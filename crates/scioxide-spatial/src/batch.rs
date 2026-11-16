//! Batches of KD-tree queries, optionally in parallel.

use crate::search::{Query, SearchPoll, SearchState, order, validate_radius};
use crate::search_workspace::{MAX_WORKERS, SearchWorker};
use crate::workspace::{add, require};
use crate::{
    KdTree, Neighbor, PointSet, SearchOptions, SearchWorkspace, SearchWorkspaceRequest,
    SpatialExecutionContext,
};
use core::num::NonZeroUsize;
use scioxide_core::backend::PartitionSlot;
use scioxide_core::{
    ExecutionContext, ExecutionPlan, ExecutionPolicy, PlanLimits, RealScalar, SciOxideError,
};

// Estimated scalar operations per item. Auto admits workers only when each
// meets the executor's minimum work; estimates never affect results.
pub(crate) fn work(units: usize) -> NonZeroUsize {
    NonZeroUsize::new(units).unwrap_or(NonZeroUsize::MIN)
}

pub(crate) fn plan(
    context: ExecutionContext<'_>,
    policy: ExecutionPolicy,
    len: usize,
    work_per_item: NonZeroUsize,
) -> ExecutionPlan {
    context.plan(
        policy,
        len,
        PlanLimits {
            work_per_item,
            auto_threshold: NonZeroUsize::new(256).unwrap_or(NonZeroUsize::MIN),
            worker_limit: NonZeroUsize::new(MAX_WORKERS).unwrap_or(NonZeroUsize::MIN),
        },
    )
}

pub(crate) fn dispatch<T: Send>(
    staging: &mut [T],
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
    work_per_item: NonZeroUsize,
    kernel: impl Fn(usize, &mut T) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    dispatch_chunks(staging, policy, context, work_per_item, |offset, values| {
        for (index, value) in values.iter_mut().enumerate() {
            context.check()?;
            kernel(offset + index, value)?;
        }
        Ok(())
    })
}

pub(crate) fn dispatch_chunks<T: Send>(
    staging: &mut [T],
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
    work_per_item: NonZeroUsize,
    kernel: impl Fn(usize, &mut [T]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    context.check()?;
    let plan = plan(context, policy, staging.len(), work_per_item);
    let mut slots: [PartitionSlot<&mut [T]>; MAX_WORKERS] =
        core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = staging;
    for partition in plan.partitions() {
        let (values, tail) = remaining.split_at_mut(partition.range().len());
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new(values);
    }
    context.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, values, child| {
            child.check()?;
            kernel(partition.range().start, values)
        },
    )
}

/// The query that [`query_batch`] runs for each row. Approximation and
/// execution are set in [`SearchOptions`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum QueryKind<S: RealScalar> {
    /// The `k` nearest neighbors of each query, or all points if there are
    /// fewer.
    Nearest {
        /// Number of neighbors.
        k: usize,
    },
    /// All points within a distance, inclusive.
    Radius {
        /// Radius, finite and nonnegative.
        radius: S,
    },
}

fn load_query<S: RealScalar>(queries: &PointSet<'_, S>, row: usize, worker: &mut SearchWorker<S>) {
    let row = if queries.contiguous() {
        queries.row::<true>(row)
    } else {
        queries.row::<false>(row)
    };
    for (axis, value) in worker.query[..queries.dimension()].iter_mut().enumerate() {
        *value = row.value(axis);
    }
}

/// Runs a nearest-neighbor or radius query for every row of `queries` and
/// returns the total number of neighbors found.
///
/// The neighbors of all queries are written to `output` one query after
/// another, each sorted as in [`KdTree::nearest_into`], and `counts[i]` is the
/// number found for query `i`. Only the first `total` entries of `output` and
/// the first `queries.len()` entries of `counts` change. A radius batch first
/// counts the matches of every query, then fills and sorts them; neither pass
/// allocates.
///
/// The workspace needs a [`SearchWorkspaceRequest::with_batch`] request: per
/// worker, `min(k, n)` neighbor slots for a nearest batch and none for a
/// radius batch. Up to 32 workers run, as allowed by the options, the
/// execution context and the number of queries. With `Auto`, a batch of at
/// least 256 queries runs in parallel when each worker gets at least the
/// executor's minimum work, estimated as `d (⌈log₂ n⌉ + leaf size + k)` units
/// per query for `n` points of dimension `d`. Each worker checks for
/// cancellation at least every 64 tree nodes.
///
///
/// # Errors
///
/// - `DimensionMismatch` if the queries and the tree differ in dimension, or
///   `output`, `counts` or the workspace is too small, including when the
///   matches of a radius batch do not fit.
/// - `InvalidInput` if the radius or `ε` is negative or not finite.
/// - `ResourceLimit` if the workspace exceeds its byte limit.
/// - `NumericalInstability` as in [`KdTree::nearest_into`].
/// - `ExecutionStopped` on cancellation or deadline, and errors of the
///   executor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query_batch<'scope, 'capability, S: RealScalar>(
    tree: &KdTree<'_, S>,
    queries: &PointSet<'_, S>,
    kind: QueryKind<S>,
    options: SearchOptions,
    output: &mut [Neighbor<S>],
    counts: Option<&mut [usize]>,
    workspace: &mut SearchWorkspace<S>,
    context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
) -> Result<usize, SciOxideError> {
    if queries.dimension() != tree.dimension() {
        return Err(SciOxideError::dimension_mismatch(
            [tree.dimension()],
            [queries.dimension()],
        ));
    }
    let factor = options.factor()?;
    let (query, candidates) = match kind {
        QueryKind::Nearest { k } => {
            let count = k.min(tree.len());
            (Query::Nearest { count, factor }, count)
        }
        QueryKind::Radius { radius } => {
            validate_radius(radius)?;
            (
                Query::Radius {
                    radius: radius.into_f64(),
                    factor,
                },
                0,
            )
        }
    };
    if let Some(counts) = counts.as_ref() {
        require(counts.len(), queries.len())?;
    }
    let context = context.into();
    let execution = context.execution();
    execution.check()?;
    let plan = plan(
        execution,
        options.execution_policy(),
        queries.len(),
        tree.query_work(candidates),
    );
    let workers = NonZeroUsize::new(plan.worker_count()).unwrap_or(NonZeroUsize::MIN);
    let request =
        SearchWorkspaceRequest::new(candidates, options).with_batch(queries.len(), 0, workers);
    let requirements = tree.query_requirements(request)?;
    if queries.is_empty() {
        let _ = workspace.check(&requirements)?;
        return Ok(0);
    }
    workspace.require(&requirements)?;
    let offsets = &mut workspace.counts[..queries.len() + 1];
    match kind {
        QueryKind::Nearest { .. } => offsets[..queries.len()].fill(candidates),
        QueryKind::Radius { .. } => {
            let mut slots: [PartitionSlot<(&mut SearchWorker<S>, &mut [usize])>; MAX_WORKERS] =
                core::array::from_fn(|_| PartitionSlot::empty());
            let mut remaining = &mut offsets[..queries.len()];
            for (partition, worker) in plan.partitions().zip(&mut workspace.workers) {
                let (values, tail) = remaining.split_at_mut(partition.range().len());
                remaining = tail;
                slots[partition.worker_index()] = PartitionSlot::new((worker, values));
            }
            execution.execute_partitioned(
                plan,
                &slots[..plan.worker_count()],
                &|partition, (worker, values), child| {
                    let mut poll = SearchPoll::new(child);
                    for (row, count) in partition.range().zip(values) {
                        load_query(queries, row, worker);
                        let mut results = SearchState::count(&mut worker.axes);
                        if !tree.is_empty() {
                            tree.search(
                                &worker.query[..tree.dimension()],
                                query,
                                &mut results,
                                &mut poll,
                            )?;
                        } else {
                            poll.visit()?;
                        }
                        *count = results.len;
                    }
                    Ok(())
                },
            )?;
        }
    }
    let mut total = 0;
    for value in &mut offsets[..queries.len()] {
        let count = *value;
        *value = total;
        total = add(total, count)?;
    }
    offsets[queries.len()] = total;
    require(output.len(), total)?;
    workspace.require(&tree.query_requirements(
        SearchWorkspaceRequest::new(candidates, options).with_batch(queries.len(), total, workers),
    )?)?;
    let offsets = &workspace.counts[..queries.len() + 1];
    let mut slots = [const { PartitionSlot::empty() }; MAX_WORKERS];
    let mut remaining = &mut workspace.output[..total];
    for (partition, worker) in plan.partitions().zip(&mut workspace.workers) {
        let range = partition.range();
        let (values, tail) = remaining.split_at_mut(offsets[range.end] - offsets[range.start]);
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new((worker, values));
    }
    execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, (worker, values), child| {
            let origin = offsets[partition.range().start];
            let mut poll = SearchPoll::new(child);
            for row in partition.range() {
                load_query(queries, row, worker);
                let destination = &mut values[offsets[row] - origin..offsets[row + 1] - origin];
                let staging = if candidates > 0 {
                    &mut worker.candidates[..candidates]
                } else {
                    &mut *destination
                };
                let mut results = SearchState::new(&mut worker.axes, staging);
                if !tree.is_empty() && !matches!(query, Query::Nearest { count: 0, .. }) {
                    tree.search(
                        &worker.query[..tree.dimension()],
                        query,
                        &mut results,
                        &mut poll,
                    )?;
                } else {
                    poll.visit()?;
                }
                results.values[..results.len].sort_unstable_by(order);
                if candidates > 0 {
                    destination.copy_from_slice(&worker.candidates[..candidates]);
                }
            }
            Ok(())
        },
    )?;
    execution.check()?;
    output[..total].copy_from_slice(&workspace.output[..total]);
    if let Some(counts) = counts {
        for (count, pair) in counts
            .iter_mut()
            .zip(workspace.counts[..queries.len() + 1].windows(2))
        {
            *count = pair[1] - pair[0];
        }
    }
    Ok(total)
}
