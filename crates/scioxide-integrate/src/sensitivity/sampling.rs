//! Node evaluation in index order, one integrand workspace per worker, with
//! every attempted call counted.

use super::{
    SensitivityIntegrand,
    provider::{self, Product},
};
use crate::gauss::kronrod::MAX_NODES;
use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicBool, Ordering},
};
use scioxide_core::backend::PartitionSlot;
use scioxide_core::{EvaluationRole, ExecutionPolicy, RealScalar, SciOxideError, SolveContext};
type SampleSlot<'a, S, W> = PartitionSlot<(&'a mut [S], &'a mut W)>;

pub(super) type Sampler<'a, S> = dyn FnMut(&[S], &mut [S], Product<'_, S>, &mut SolveContext<'_, '_>) -> Result<usize, SciOxideError>
    + 'a;

#[allow(clippy::too_many_arguments)]
pub(super) fn executor<S, I>(
    integrand: &I,
    parameters: &[S],
    product: Product<'_, S>,
    points: &[S],
    values: &mut [S],
    workspaces: &mut [I::Workspace],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError>
where
    S: RealScalar,
    I: SensitivityIntegrand<S> + Sync + ?Sized,
    I::Workspace: Send,
{
    if points.is_empty() || points.len() > MAX_NODES {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid sample batch size",
        ));
    }
    let limit = NonZeroUsize::new(workspaces.len().min(MAX_NODES)).ok_or_else(|| {
        crate::kernel::invalid(
            scioxide_core::argument!("provider_workspaces"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty provider workspace pool",
        )
    })?;
    let dimension = values.len() / points.len();
    let execution = context.execution();
    let plan = execution.plan(
        policy,
        points.len(),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: limit,
        },
    );
    let mut slots: [SampleSlot<'_, S, I::Workspace>; MAX_NODES] =
        core::array::from_fn(|_| PartitionSlot::empty());
    let attempted: [AtomicBool; MAX_NODES] = core::array::from_fn(|_| AtomicBool::new(false));
    let products: [AtomicBool; MAX_NODES] = core::array::from_fn(|_| AtomicBool::new(false));
    // Each partition owns a disjoint block of rows and one caller-provided provider
    // workspace, handed out through claim-once slots; the worker count is capped by the
    // number of workspaces so no two workers ever share provider scratch.
    let mut remaining = values;
    for (partition, workspace) in plan.partitions().zip(workspaces) {
        let (rows, tail) = remaining.split_at_mut(partition.range().len() * dimension);
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new((rows, workspace));
    }
    let outcome = execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, (rows, workspace), worker_execution| {
            for (index, row) in partition.range().zip(rows.chunks_exact_mut(dimension)) {
                worker_execution.check()?;
                attempted[index].store(true, Ordering::Relaxed);
                let mut applied = false;
                let result = provider::evaluate(
                    integrand,
                    points[index],
                    parameters,
                    product,
                    (row, workspace),
                    worker_execution,
                    &mut applied,
                );
                products[index].store(applied, Ordering::Relaxed);
                result?;
            }
            Ok(())
        },
    );
    let mut functions = 0;
    // Reconcile successful and failed attempts even when the executor returns Err.
    for i in 0..points.len() {
        if attempted[i].load(Ordering::Relaxed) {
            if !context
                .record_provider(EvaluationRole::Function)?
                .is_granted()
            {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("context"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "admitted sample budget changed",
                ));
            }
            functions += 1;
        }
        if products[i].load(Ordering::Relaxed)
            && !context
                .record_provider(EvaluationRole::LinearOperator)?
                .is_granted()
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("context"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "admitted derivative budget changed",
            ));
        }
    }
    outcome?;

    Ok(functions)
}
