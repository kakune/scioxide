//! Evaluation of values, gradients and Hessians at many points.
//!
//! A batch checks every shape first, evaluates all points into its workspace,
//! and copies the results to the output only if every point succeeds.

use crate::{
    Differentiable, ExtrapolationPolicy, ExtrapolationReport, Interpolant, TwiceDifferentiable,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use scioxide_core::backend::{PartitionSlot, checked_mul, try_reserve};
use scioxide_core::{
    Capacity, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar, SciOxideError,
};

/// The most workers a batch uses; partition slots are a fixed array.
const MAX_WORKERS: usize = 64;

/// Returns the number of workers that `policy` allows for `queries` points.
fn workers(policy: ExecutionPolicy, queries: usize) -> usize {
    let bound = match policy {
        ExecutionPolicy::Sequential => 1,
        ExecutionPolicy::Parallel(workers) => workers.get().min(MAX_WORKERS),
        ExecutionPolicy::Auto => MAX_WORKERS,
    };
    bound.min(queries)
}

/// Selects what a batch computes at each point.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BatchOperation, BatchRequest};
/// assert_eq!(
///     BatchRequest::hessians(2).operation(),
///     BatchOperation::Hessians
/// );
/// assert_eq!(
///     BatchRequest::new(BatchOperation::Values, 2),
///     BatchRequest::values(2)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchOperation {
    /// Values, with output shape `[point, component]`.
    Values,
    /// Gradients, with output shape `[point, D, component]`.
    Gradients,
    /// Hessians, with output shape `[point, D, D, component]`.
    Hessians,
}

/// Describes a batch for sizing its workspace.
///
/// A request names the operation, the number of points and whether the batch
/// also returns an [`ExtrapolationReport`] per point.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{BatchOperation, BatchRequest};
/// let request = BatchRequest::gradients(8).with_reports();
/// assert_eq!(request.operation(), BatchOperation::Gradients);
/// assert_eq!((request.queries(), request.reports()), (8, true));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct BatchRequest {
    operation: BatchOperation,
    queries: usize,
    reports: bool,
}

impl BatchRequest {
    /// Requests values at `queries` points.
    pub const fn values(queries: usize) -> Self {
        Self::new(BatchOperation::Values, queries)
    }

    /// Requests gradients at `queries` points.
    pub const fn gradients(queries: usize) -> Self {
        Self::new(BatchOperation::Gradients, queries)
    }

    /// Requests Hessians at `queries` points.
    pub const fn hessians(queries: usize) -> Self {
        Self::new(BatchOperation::Hessians, queries)
    }

    /// Requests `operation` at `queries` points, without reports.
    pub const fn new(operation: BatchOperation, queries: usize) -> Self {
        Self {
            operation,
            queries,
            reports: false,
        }
    }

    /// Also requests one [`ExtrapolationReport`] per point.
    pub const fn with_reports(mut self) -> Self {
        self.reports = true;
        self
    }

    /// Returns the operation.
    #[must_use]
    pub const fn operation(&self) -> BatchOperation {
        self.operation
    }

    /// Returns the number of points.
    #[must_use]
    pub const fn queries(&self) -> usize {
        self.queries
    }

    /// Returns whether reports are requested.
    #[must_use]
    pub const fn reports(&self) -> bool {
        self.reports
    }
}

/// Evaluates an interpolant at many points into caller storage, optionally in
/// parallel.
///
/// The evaluator binds an interpolant to the [`ExecutionPolicy`] that sets the
/// number of workers, so that sizing a workspace and running a batch use the
/// same count. `Sequential` uses one worker, `Parallel(n)` at most `min(n, 64)`
/// and `Auto` at most 64; the number of points and the executor can lower it
/// further. Each worker has its own interpolant workspace, so a batch workspace
/// for `Auto` can hold up to 64 of them. [`Interpolant::evaluate_many`] is the
/// owned form that returns the values.
///
/// The points are the rows of a `[point, D]` array view, and results go to an
/// array view of shape `[point, value_dimension]` for values, with one or two
/// extra axes of length `D` for gradients and Hessians. Views may be strided or
/// broadcast. The points are split into contiguous ranges, one per worker, as
/// described in [execution contexts](scioxide_core#execution-contexts). Each
/// point is evaluated exactly as by [`Interpolant::evaluate_into`], so the
/// result does not depend on the number of workers.
///
/// If any point fails, the batch returns that error and leaves the output and
/// the reports unchanged. The evaluator owns its interpolant; pass `&model` to
/// borrow it instead. It precomputes nothing, is immutable and is `Send`,
/// `Sync` and `Clone` whenever the interpolant is.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, ExecutionPolicy, NdView, NdViewMut, StridedLayout};
/// use scioxide_interp::{BatchEvaluator, BatchRequest, BilinearInterpolator, ExtrapolationPolicy};
/// // f(x, y) = x + 2y on the unit square.
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0])?;
/// let evaluator = BatchEvaluator::new(&grid, ExecutionPolicy::Sequential);
/// let points = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
/// let queries = NdView::new(&points, StridedLayout::new(&[3, 2], &[2, 1])?)?;
/// let mut values = [0.0; 3];
/// let mut out = NdViewMut::new(&mut values, StridedLayout::new(&[3, 1], &[1, 1])?)?;
/// let mut workspace = evaluator.workspace(BatchRequest::values(3))?;
/// evaluator.evaluate_into(
///     &queries,
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     None,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(values, [0.0, 1.5, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BatchEvaluator<S, I, const D: usize> {
    model: I,
    policy: ExecutionPolicy,
    scalar: PhantomData<fn() -> S>,
}

/// Lists the storage a [`BatchWorkspace`] needs for one [`BatchRequest`].
///
/// Requirements borrow the evaluator, so that [`BatchWorkspace::try_reserve`]
/// can create the interpolant workspace of each worker.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_interp::{BatchEvaluator, BatchRequest, BilinearInterpolator};
/// // Two-component values on a 2 × 2 grid.
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![[0.0, 1.0]; 4])?;
/// let evaluator = BatchEvaluator::new(&grid, ExecutionPolicy::Sequential);
/// let requirements = evaluator.workspace_requirements(BatchRequest::gradients(5))?;
/// // A gradient has D · 2 = 4 real components.
/// assert_eq!(
///     (
///         requirements.queries(),
///         requirements.width(),
///         requirements.workers()
///     ),
///     (5, 4, 1)
/// );
/// assert_eq!(requirements.staging_bytes()?, 5 * 4 * 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BatchRequirements<'evaluator, S, I, const D: usize> {
    evaluator: &'evaluator BatchEvaluator<S, I, D>,
    request: BatchRequest,
    width: usize,
    workers: usize,
}

impl<S: RealScalar, I: Interpolant<S, D>, const D: usize> BatchRequirements<'_, S, I, D> {
    /// Returns the number of points.
    #[must_use]
    pub const fn queries(&self) -> usize {
        self.request.queries
    }

    /// Returns the number of real components of one result.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the number of workers, each with its own interpolant workspace.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Returns the bytes needed for the results and reports of the whole batch.
    ///
    /// The interpolant workspaces of the workers are not included.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn staging_bytes(&self) -> Result<usize, SciOxideError> {
        let values = checked_mul(
            self.values()?,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("batch_value_bytes"),
        )?;
        let reports = checked_mul(
            self.reports(),
            core::mem::size_of::<ExtrapolationReport<S, D>>(),
            scioxide_core::quantity!("batch_report_bytes"),
        )?;
        scioxide_core::backend::checked_add(
            values,
            reports,
            scioxide_core::quantity!("batch_bytes"),
        )
    }

    fn values(&self) -> Result<usize, SciOxideError> {
        checked_mul(
            self.request.queries,
            self.width,
            scioxide_core::quantity!("batch_values"),
        )
    }

    const fn reports(&self) -> usize {
        if self.request.reports {
            self.request.queries
        } else {
            0
        }
    }
}

/// Holds the scratch storage of a [`BatchEvaluator`].
///
/// It holds one interpolant workspace per worker, and storage for the results
/// and reports of the whole batch until every point has succeeded. Create one
/// with [`BatchEvaluator::workspace`] and grow it with [`Self::try_reserve`].
/// One workspace serves every evaluator whose interpolant has workspace type `W`. Batches
/// that run at the same time need separate workspaces.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Capacity, ExecutionPolicy};
/// use scioxide_interp::{BatchEvaluator, BatchRequest, BilinearInterpolator};
/// let grid = BilinearInterpolator::new(vec![0.0_f64, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0])?;
/// let evaluator = BatchEvaluator::new(&grid, ExecutionPolicy::Sequential);
/// let mut workspace = evaluator.workspace(BatchRequest::values(4))?;
/// let larger = evaluator.workspace_requirements(BatchRequest::values(64).with_reports())?;
/// assert_eq!(workspace.check(&larger)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&larger)?;
/// assert_eq!(workspace.check(&larger)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct BatchWorkspace<S: RealScalar, W, const D: usize> {
    workers: Vec<W>,
    values: Vec<S>,
    reports: Vec<ExtrapolationReport<S, D>>,
}

impl<S: RealScalar, W, const D: usize> core::fmt::Debug for BatchWorkspace<S, W, D> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BatchWorkspace")
            .field("workers", &self.workers.len())
            .field("values", &self.values.len())
            .field("reports", &self.reports.len())
            .finish()
    }
}

impl<S: RealScalar, W: Send, const D: usize> BatchWorkspace<S, W, D> {
    /// Returns whether this workspace satisfies `requirements` without growing.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check<I: Interpolant<S, D, Workspace = W>>(
        &self,
        requirements: &BatchRequirements<'_, S, I, D>,
    ) -> Result<Capacity, SciOxideError> {
        let model = &requirements.evaluator.model;
        let sufficient = self.values.len() >= requirements.values()?
            && self.reports.len() >= requirements.reports()
            && self.workers.len() >= requirements.workers
            && self.workers[..requirements.workers]
                .iter()
                .all(|worker| model.workspace_capacity(worker) == Capacity::Sufficient);
        Ok(if sufficient {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows this workspace in place to satisfy `requirements`.
    ///
    /// Worker workspaces that are too small are replaced with new ones from the
    /// interpolant. If the reservation fails, the workspace may keep part of the
    /// capacity it gained and stays usable.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if storage cannot be allocated.
    /// - The errors of [`Interpolant::workspace`].
    pub fn try_reserve<I: Interpolant<S, D, Workspace = W>>(
        &mut self,
        requirements: &BatchRequirements<'_, S, I, D>,
    ) -> Result<(), SciOxideError> {
        let model = &requirements.evaluator.model;
        let values = requirements.values()?;
        if self.values.len() < values {
            let additional = values - self.values.len();
            try_reserve(
                &mut self.values,
                additional,
                scioxide_core::quantity!("batch_values"),
            )?;
            self.values.resize(values, S::ZERO);
        }
        let reports = requirements.reports();
        if self.reports.len() < reports {
            let additional = reports - self.reports.len();
            try_reserve(
                &mut self.reports,
                additional,
                scioxide_core::quantity!("batch_reports"),
            )?;
            self.reports
                .resize(reports, ExtrapolationReport::interior([S::ZERO; D]));
        }
        for worker in self.workers.iter_mut().take(requirements.workers) {
            if model.workspace_capacity(worker) != Capacity::Sufficient {
                *worker = model.workspace()?;
            }
        }
        if self.workers.len() < requirements.workers {
            let additional = requirements.workers - self.workers.len();
            try_reserve(
                &mut self.workers,
                additional,
                scioxide_core::quantity!("batch_workers"),
            )?;
            while self.workers.len() < requirements.workers {
                self.workers.push(model.workspace()?);
            }
        }
        Ok(())
    }
}

impl<S: RealScalar, I: Interpolant<S, D>, const D: usize> BatchEvaluator<S, I, D> {
    /// Binds `model` to the execution policy `policy`.
    pub const fn new(model: I, policy: ExecutionPolicy) -> Self {
        Self {
            model,
            policy,
            scalar: PhantomData,
        }
    }

    /// Returns the interpolant.
    #[must_use]
    pub const fn model(&self) -> &I {
        &self.model
    }

    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }

    /// Returns the storage that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: BatchRequest,
    ) -> Result<BatchRequirements<'_, S, I, D>, SciOxideError> {
        let factor = match request.operation {
            BatchOperation::Values => 1,
            BatchOperation::Gradients => D,
            BatchOperation::Hessians => {
                checked_mul(D, D, scioxide_core::quantity!("hessian_entries"))?
            }
        };
        let width = checked_mul(
            factor,
            self.model.value_dimension(),
            scioxide_core::quantity!("batch_width"),
        )?;
        let workers = workers(self.policy, request.queries);
        Ok(BatchRequirements {
            evaluator: self,
            request,
            width,
            workers,
        })
    }

    /// Allocates a workspace that satisfies `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`BatchWorkspace::try_reserve`].
    pub fn workspace(
        &self,
        request: BatchRequest,
    ) -> Result<BatchWorkspace<S, I::Workspace, D>, SciOxideError> {
        let mut workspace = BatchWorkspace {
            workers: Vec::new(),
            values: Vec::new(),
            reports: Vec::new(),
        };
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Evaluates values at the rows of `queries` and writes them to `out`.
    ///
    /// `queries` has shape `[point, D]` and `out` has shape
    /// `[point, value_dimension]`. If `reports` is given, it receives one
    /// [`ExtrapolationReport`] per point. Elements of `out` outside its layout
    /// are not written.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a shape is wrong, `reports` has the wrong
    ///   length, or `workspace` is too small for this batch.
    /// - The errors of [`Interpolant::evaluate_into`] at any point.
    /// - `ExecutionStopped` on cancellation or a deadline, and the errors of
    ///   the executor.
    pub fn evaluate_into(
        &self,
        queries: &NdView<'_, '_, S>,
        policy: ExtrapolationPolicy<S>,
        out: &mut NdViewMut<'_, '_, S>,
        reports: Option<&mut [ExtrapolationReport<S, D>]>,
        workspace: &mut BatchWorkspace<S, I::Workspace, D>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute(
            BatchOperation::Values,
            queries,
            policy,
            out,
            reports,
            workspace,
            context,
            &|model, point, policy, out, workspace| {
                model.evaluate_into(point, policy, out, workspace)
            },
        )
    }

    /// Evaluates gradients at the rows of `queries` and writes them to `out`.
    ///
    /// `out` has shape `[point, D, value_dimension]`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`], with those of
    ///   [`Differentiable::gradient_into`] at each point.
    pub fn gradient_into(
        &self,
        queries: &NdView<'_, '_, S>,
        policy: ExtrapolationPolicy<S>,
        out: &mut NdViewMut<'_, '_, S>,
        reports: Option<&mut [ExtrapolationReport<S, D>]>,
        workspace: &mut BatchWorkspace<S, I::Workspace, D>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        I: Differentiable<S, D>,
    {
        self.execute(
            BatchOperation::Gradients,
            queries,
            policy,
            out,
            reports,
            workspace,
            context,
            &|model, point, policy, out, workspace| {
                model.gradient_into(point, policy, out, workspace)
            },
        )
    }

    /// Evaluates Hessians at the rows of `queries` and writes them to `out`.
    ///
    /// `out` has shape `[point, D, D, value_dimension]`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::evaluate_into`], with those of
    ///   [`TwiceDifferentiable::hessian_into`] at each point.
    pub fn hessian_into(
        &self,
        queries: &NdView<'_, '_, S>,
        policy: ExtrapolationPolicy<S>,
        out: &mut NdViewMut<'_, '_, S>,
        reports: Option<&mut [ExtrapolationReport<S, D>]>,
        workspace: &mut BatchWorkspace<S, I::Workspace, D>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        I: TwiceDifferentiable<S, D>,
    {
        self.execute(
            BatchOperation::Hessians,
            queries,
            policy,
            out,
            reports,
            workspace,
            context,
            &|model, point, policy, out, workspace| {
                model.hessian_into(point, policy, out, workspace)
            },
        )
    }

    /// Checks shapes, evaluates every point into the workspace and copies the
    /// results to `out` once all have succeeded.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn execute<F>(
        &self,
        operation: BatchOperation,
        queries: &NdView<'_, '_, S>,
        policy: ExtrapolationPolicy<S>,
        out: &mut NdViewMut<'_, '_, S>,
        reports: Option<&mut [ExtrapolationReport<S, D>]>,
        workspace: &mut BatchWorkspace<S, I::Workspace, D>,
        context: ExecutionContext<'_>,
        evaluate: &F,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(
                &I,
                [S; D],
                ExtrapolationPolicy<S>,
                &mut [S],
                &mut I::Workspace,
            ) -> Result<(), SciOxideError>
            + Sync,
    {
        let shape = queries.layout().shape().extents();
        if shape.len() != 2 || shape[1] != D {
            return Err(SciOxideError::dimension_mismatch(
                [shape.first().copied().unwrap_or(0), D],
                shape,
            ));
        }
        let n = shape[0];
        let mut request = BatchRequest::new(operation, n);
        if reports.is_some() {
            request = request.with_reports();
        }
        let requirements = self.workspace_requirements(request)?;
        let components = self.model.value_dimension();
        let expected: &[usize] = match operation {
            BatchOperation::Values => &[n, components],
            BatchOperation::Gradients => &[n, D, components],
            BatchOperation::Hessians => &[n, D, D, components],
        };
        let actual = out.layout().shape().extents();
        if actual != expected {
            return Err(SciOxideError::dimension_mismatch(expected, actual));
        }
        if let Some(reports) = reports.as_deref()
            && reports.len() != n
        {
            return Err(crate::error::shape(n, reports.len()));
        }
        let width = requirements.width;
        let total = requirements.values()?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(crate::error::shape(
                requirements.staging_bytes()?,
                workspace
                    .values
                    .len()
                    .saturating_mul(core::mem::size_of::<S>()),
            ));
        }
        // Every contract violation is reported before cancellation is polled or
        // any worker runs, including for empty batches.
        context.check()?;
        if n == 0 {
            return Ok(());
        }
        let worker_limit = NonZeroUsize::new(requirements.workers).unwrap_or(NonZeroUsize::MIN);
        let plan = context.plan(
            self.policy,
            n,
            scioxide_core::PlanLimits {
                work_per_item: NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit,
            },
        );
        if plan.worker_count() > requirements.workers {
            return Err(crate::error::shape(
                plan.worker_count(),
                requirements.workers,
            ));
        }
        let staged_reports = request.reports;
        // Partition the staging buffers into disjoint, contiguous per-worker
        // chunks in query order and park each in a slot keyed by worker index;
        // the shared closure takes ownership of its chunk exactly once.
        let mut slots: [_; MAX_WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
        let mut values = &mut workspace.values[..total];
        let mut report_rows = &mut workspace.reports[..requirements.reports()];
        for (partition, worker) in plan.partitions().zip(&mut workspace.workers) {
            let range = partition.range();
            let (chunk, tail) = values.split_at_mut(range.len() * width);
            values = tail;
            let rows = if staged_reports {
                let (rows, tail) = report_rows.split_at_mut(range.len());
                report_rows = tail;
                rows
            } else {
                &mut []
            };
            slots[partition.worker_index()] = PartitionSlot::new((worker, chunk, rows));
        }
        let model = &self.model;
        // NdView proved the reachable span at construction; the rank/shape
        // checks above prove these strides describe [n, D]. Each loop index is
        // in range, so safe indexing needs no repeated coordinate decoding.
        let data = queries.storage();
        let layout = queries.layout();
        let strides = layout.strides();
        context.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|partition, (worker, chunk, rows), _| {
                for (offset, query) in partition.range().enumerate() {
                    // Coordinates and model semantics remain authoritative even
                    // when each query has no value components.
                    let result = &mut chunk[offset * width..(offset + 1) * width];
                    context.check()?;
                    let point =
                        core::array::from_fn(|axis| data[query * strides[0] + axis * strides[1]]);
                    if let Some(report) = rows.get_mut(offset) {
                        if operation == BatchOperation::Values {
                            *report =
                                model.evaluate_into_with_report(point, policy, result, worker)?;
                            continue;
                        }
                        *report = model.report(point, policy, worker)?;
                    }
                    evaluate(model, point, policy, result, worker)?;
                }
                Ok(())
            },
        )?;
        // Last cancellation point; afterwards the commit cannot fail. Staging is
        // dense row-major and `for_each_mut` visits the output in the same
        // logical order.
        context.check()?;
        let mut staged = workspace.values[..total].iter();
        out.for_each_mut(|slot| {
            if let Some(value) = staged.next() {
                *slot = *value;
            }
        });
        if let Some(reports) = reports {
            reports.copy_from_slice(&workspace.reports[..n]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_bound_reserved_workers() {
        let parallel =
            |n| ExecutionPolicy::Parallel(NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN));
        let workers = |policy, queries| workers(policy, queries);
        assert_eq!(workers(ExecutionPolicy::Sequential, 10), 1);
        assert_eq!(workers(parallel(4), 10), 4);
        assert_eq!(workers(parallel(1000), 1000), MAX_WORKERS);
        assert_eq!(workers(ExecutionPolicy::Auto, 3), 3);
        assert_eq!(workers(ExecutionPolicy::Sequential, 0), 0);
    }

    #[test]
    fn requests_carry_operation_queries_and_reports() {
        let request = BatchRequest::new(BatchOperation::Gradients, 7);
        assert_eq!(request, BatchRequest::gradients(7));
        assert!(!request.reports() && request.with_reports().reports());
        assert_eq!(BatchRequest::hessians(2).queries(), 2);
    }
}
