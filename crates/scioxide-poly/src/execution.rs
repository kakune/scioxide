//! Execution options for polynomial plans and batch evaluation.
use crate::util::{invalid, minimum_len};
use core::num::NonZeroUsize;
use scioxide_core::backend::{PartitionSlot, checked_add, checked_mul};
use scioxide_core::{ExecutionContext, NumericScalar};
use scioxide_core::{ExecutionPolicy, InvalidInputReason, SciOxideError};

/// Selects how many workers a polynomial plan or tensor evaluation view may use.
///
/// The options hold an [`ExecutionPolicy`] and an upper bound on the number of
/// workers. The default is sequential execution with a bound of 64. With
/// `Parallel(n)` a plan uses `min(n, bound)` workers; with `Auto` it may use up
/// to the bound. The executor, cancellation flag and deadline come from the
/// [`ExecutionContext`] passed on each call (see
/// [Execution contexts](scioxide_core#execution-contexts)).
///
/// # Errors
///
/// The options are checked when a plan is prepared. A worker bound above 64 is
/// `InvalidInput` (`options.worker_bound`).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_poly::PolynomialExecutionOptions;
///
/// let options = PolynomialExecutionOptions::new().with_execution_policy(ExecutionPolicy::Auto);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Auto);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolynomialExecutionOptions {
    policy: ExecutionPolicy,
    worker_bound: NonZeroUsize,
}

impl PolynomialExecutionOptions {
    /// Returns sequential execution with a worker bound of 64.
    ///
    /// ```
    /// use scioxide_core::ExecutionPolicy;
    /// use scioxide_poly::PolynomialExecutionOptions;
    ///
    /// let options = PolynomialExecutionOptions::new();
    /// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
    /// ```
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
            worker_bound: NonZeroUsize::MIN.saturating_add(63),
        }
    }
    /// Sets the execution policy.
    ///
    /// ```
    /// use scioxide_core::ExecutionPolicy;
    /// use scioxide_poly::PolynomialExecutionOptions;
    ///
    /// let options = PolynomialExecutionOptions::new().with_execution_policy(ExecutionPolicy::Auto);
    /// assert_eq!(options.execution_policy(), ExecutionPolicy::Auto);
    /// ```
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the upper bound on the number of workers.
    ///
    /// A bound above 64 is rejected when a plan is prepared with these options.
    ///
    /// ```
    /// use core::num::NonZeroUsize;
    /// use scioxide_poly::PolynomialExecutionOptions;
    ///
    /// let workers = NonZeroUsize::MIN;
    /// let options = PolynomialExecutionOptions::new().with_worker_bound(workers);
    /// assert_eq!(options.worker_bound(), workers);
    /// ```
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.worker_bound = workers;
        self
    }
    /// Returns the execution policy.
    ///
    /// ```
    /// use scioxide_core::ExecutionPolicy;
    /// use scioxide_poly::PolynomialExecutionOptions;
    ///
    /// let options = PolynomialExecutionOptions::new();
    /// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
    /// ```
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the upper bound on the number of workers.
    ///
    /// ```
    /// use scioxide_poly::PolynomialExecutionOptions;
    ///
    /// let options = PolynomialExecutionOptions::new();
    /// assert_eq!(options.worker_bound().get(), 64);
    /// ```
    pub const fn worker_bound(self) -> NonZeroUsize {
        self.worker_bound
    }
    pub(crate) fn validate(self) -> Result<(), SciOxideError> {
        if self.worker_bound.get() > 64 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options.worker_bound"),
                InvalidInputReason::Inconsistent,
                "polynomial worker bound exceeds 64",
            ));
        }
        Ok(())
    }
    pub(crate) fn workers(self) -> NonZeroUsize {
        match self.policy {
            ExecutionPolicy::Sequential => NonZeroUsize::MIN,
            ExecutionPolicy::Parallel(workers) => workers.min(self.worker_bound),
            ExecutionPolicy::Auto => self.worker_bound,
        }
    }
    pub(crate) fn resolved_policy(self) -> ExecutionPolicy {
        match self.policy {
            ExecutionPolicy::Parallel(_) => ExecutionPolicy::Parallel(self.workers()),
            policy => policy,
        }
    }
}

impl Default for PolynomialExecutionOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluates `compute(i, work)` for every output index, staging results in
/// `scratch` and copying them to `output` only after all of them succeed.
pub(crate) fn dispatch<C: NumericScalar>(
    output: &mut [C],
    scratch: &mut [C],
    per_worker: usize,
    policy: ExecutionPolicy,
    workers: NonZeroUsize,
    context: ExecutionContext<'_>,
    compute: impl Fn(usize, &mut [C]) -> Result<C, SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    // Partitions are handed to workers through a fixed 64-entry slot array to
    // avoid allocation, which caps the worker count.
    if workers.get() > 64 {
        return Err(invalid(
            scioxide_core::argument!("workers"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "polynomial batch worker cap exceeds 64",
        ));
    }
    context.check()?;
    // The execution plan never creates more partitions than work items, so only
    // `min(workers, items)` worker areas can be active.
    let required = checked_add(
        output.len(),
        checked_mul(
            workers.get().min(output.len()),
            per_worker,
            scioxide_core::quantity!("worker_scratch"),
        )?,
        scioxide_core::quantity!("batch_scratch"),
    )?;
    minimum_len(required, scratch.len())?;
    // Scratch layout: `[staged results (output.len()) | per-worker work areas]`.
    // Each partition receives disjoint mutable slices through a `PartitionSlot`,
    // which can be claimed only once; this is what lets the shared closure write
    // without aliasing.
    let (staged, workspace) = scratch.split_at_mut(output.len());
    let plan = context.plan(
        policy,
        output.len(),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    );
    type WorkerSlot<'a, C> = PartitionSlot<(&'a mut [C], &'a mut [C])>;
    let mut slots: [WorkerSlot<'_, C>; 64] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = staged;
    let mut remaining_work = workspace;
    for partition in plan.partitions() {
        let (part, tail) = remaining.split_at_mut(partition.range().len());
        remaining = tail;
        let (work, tail) = remaining_work.split_at_mut(per_worker);
        remaining_work = tail;
        slots[partition.worker_index()] = PartitionSlot::new((part, work));
    }
    context.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, (part, work), _| {
            for (i, value) in part.iter_mut().enumerate() {
                context.check()?;
                *value = compute(partition.range().start + i, work)?;
            }
            Ok(())
        },
    )?;
    // The execution context validates partition completion before publication.

    context.check()?;
    output.copy_from_slice(&scratch[..output.len()]);
    Ok(())
}
