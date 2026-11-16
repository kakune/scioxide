//! Workspace-owned solve state, the one batch shell and shared errors.
//!
//! Every validated workspace owns a [`Shell`]: its reusable [`SolveState`], so epochs
//! stay monotone across calls, and the record of its most recent attempt. A call
//! validates and grows storage first; a failure there is recorded as a preflight
//! rejection and leaves every output unchanged. The kernel then runs in a fresh root
//! or child epoch of the caller's scope, and the attempt records its epoch, counts and
//! one outcome.
//!
//! [`dispatch`] runs independent proofs of a batch in child epochs of one batch epoch.
//! Shared ancestor limits are divided by proof index, so outcomes do not depend on the
//! execution policy or the worker count.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::backend::{
    PartitionSlot, PartitionSolveBatch, PartitionedSolve, checked_add, checked_mul, try_reserve,
    with_partitioned_solves,
};
use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, ExecutionContext, ExecutionPolicy,
    ExecutionStopReason, PlanLimits, Quantity, SciOxideError, SciOxideErrorView, SolveBudgets,
    SolveContext, SolveCounts, SolveIdentity, SolveScope, SolveState,
};

type Result<T> = core::result::Result<T, SciOxideError>;

/// Inconsistent caller input.
pub(crate) fn invalid(argument: scioxide_core::Argument, message: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        message,
    )
}

/// Failure of an internal numerical invariant.
pub(crate) fn breakdown(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, message)
}

/// Largest number of executor partitions of one batch.
const MAX_WORKERS: usize = 64;
const LIMITS: PlanLimits = PlanLimits {
    auto_threshold: NonZeroUsize::MIN,
    work_per_item: NonZeroUsize::MIN,
    worker_limit: match NonZeroUsize::new(MAX_WORKERS) {
        Some(limit) => limit,
        None => NonZeroUsize::MIN,
    },
};

/// A proof outcome that a batch can mark as stopped and classify.
pub(crate) trait Outcome: Copy + Send {
    /// Initial value of a report slot before execution.
    fn not_run() -> Self;
    /// The outcome of a proof interrupted by cancellation or a deadline.
    fn stopped(reason: ExecutionStopReason) -> Self;
    /// Whether the proof met its success criterion.
    fn succeeded(&self) -> bool;
}

/// Workspace outcome storage that stages a family report `T` and reads it back intact.
pub(crate) trait Ledger<T>: Outcome {
    /// Stage `report`.
    fn stage(report: T) -> Self;
    /// The staged report.
    fn report(self) -> T;
}

impl<T: Outcome> Ledger<T> for T {
    fn stage(report: T) -> Self {
        report
    }
    fn report(self) -> T {
        self
    }
}

/// Family-owned scratch that grows monotonically to a checked shape.
pub(crate) trait Storage {
    type Shape: Copy + Default;
    fn empty() -> Self;
    fn projected_bytes(&self, shape: Self::Shape) -> Result<usize>;
    fn fits(&self, shape: Self::Shape) -> Result<bool>;
    fn reserve(&mut self, shape: Self::Shape) -> Result<()>;
}

/// Shapes and limit of one family requirements value.
pub(crate) trait Requirements<L: Storage, A: Storage> {
    fn lanes(&self) -> usize;
    fn lane_shape(&self) -> L::Shape;
    fn shared_shape(&self) -> A::Shape;
    fn limit(&self) -> usize;
}

/// Result of common batch preparation, before independent proofs run.
pub(crate) enum Prepared<P, O> {
    Ready(P),
    Inconclusive(O),
}

/// Reusable proof lanes, family scratch, child ledgers and one attempt ledger.
#[derive(Debug)]
pub(crate) struct Workspace<L, A, O> {
    pub(crate) lanes: Vec<L>,
    pub(crate) shared: A,
    pub(crate) batch: Batch<O>,
    pub(crate) shell: Shell,
}

impl<L: Storage + Send, A: Storage + Sync, O: Outcome> Workspace<L, A, O> {
    pub(crate) const fn new(shared: A) -> Self {
        Self {
            lanes: Vec::new(),
            shared,
            batch: Batch::new(),
            shell: Shell::new(),
        }
    }

    pub(crate) fn projected_bytes(&self, request: &impl Requirements<L, A>) -> Result<usize> {
        let count = request.lanes();
        let mut bytes = add(
            grown_bytes(&self.lanes, count)?,
            self.batch.projected_bytes(count)?,
        )?;
        bytes = add(bytes, self.shared.projected_bytes(request.shared_shape())?)?;
        for (index, lane) in self.lanes.iter().enumerate() {
            let shape = if index < count {
                request.lane_shape()
            } else {
                L::Shape::default()
            };
            bytes = add(bytes, lane.projected_bytes(shape)?)?;
        }
        let missing = count.saturating_sub(self.lanes.len());
        if missing != 0 {
            bytes = add(
                bytes,
                mul(missing, L::empty().projected_bytes(request.lane_shape())?)?,
            )?;
        }
        Ok(bytes)
    }

    pub(crate) fn check(
        &self,
        request: &impl Requirements<L, A>,
    ) -> Result<scioxide_core::Capacity> {
        within_limit(self.projected_bytes(request)?, request.limit())?;
        let count = request.lanes();
        let mut fits = self.lanes.len() >= count
            && self.batch.fits(count)
            && self.shared.fits(request.shared_shape())?;
        if fits {
            for lane in &self.lanes[..count] {
                fits &= lane.fits(request.lane_shape())?;
            }
        }
        Ok(if fits {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }

    pub(crate) fn reserve(&mut self, request: &impl Requirements<L, A>) -> Result<()> {
        if self.check(request)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let count = request.lanes();
        self.shared.reserve(request.shared_shape())?;
        if let Some(additional) = count.checked_sub(self.lanes.len()) {
            try_reserve(
                &mut self.lanes,
                additional,
                scioxide_core::quantity!("workspace.lanes"),
            )?;
            self.lanes.resize_with(count, L::empty);
        }
        for lane in &mut self.lanes[..count] {
            lane.reserve(request.lane_shape())?;
        }
        self.batch.try_reserve(count, O::not_run())
    }

    /// Compute requirements, validate and reserve before starting an epoch.
    #[expect(
        clippy::too_many_arguments,
        reason = "preflight, scratch admission and kernel share one attempt boundary"
    )]
    pub(crate) fn run<T, Q: Requirements<L, A>>(
        &mut self,
        scope: SolveScope<'_, '_>,
        identity: SolveIdentity,
        options: &crate::ValidatedOptions,
        preflight: impl FnOnce() -> Result<Q>,
        admit: impl FnOnce(&mut A, &Q) -> Result<()>,
        kernel: impl FnOnce(&mut [L], &mut A, &mut Batch<O>, &Q, &mut SolveContext<'_, '_>) -> Result<T>,
        success: impl FnOnce(&T) -> bool,
    ) -> Result<T> {
        let request = match crate::environment::check()
            .and_then(|()| preflight())
            .and_then(|request| {
                self.reserve(&request)?;
                admit(&mut self.shared, &request)?;
                Ok(request)
            }) {
            Ok(request) => request,
            Err(error) => return self.shell.reject(error),
        };
        self.shell.execute(
            scope,
            identity,
            options.budgets(),
            |cx| {
                kernel(
                    &mut self.lanes[..request.lanes()],
                    &mut self.shared,
                    &mut self.batch,
                    &request,
                    cx,
                )
            },
            success,
        )
    }

    /// Prepare common proof state, dispatch children and write all reports together.
    /// Reports of type `T` are staged as the workspace outcome `O`.
    #[expect(
        clippy::too_many_arguments,
        reason = "one batch boundary with family preparation and proof callbacks"
    )]
    pub(crate) fn verify_batch<Q, P, T, F>(
        &mut self,
        scope: SolveScope<'_, '_>,
        identity: SolveIdentity,
        options: &crate::ValidatedOptions,
        preflight: impl FnOnce() -> Result<Q>,
        reports: &mut [T],
        prepare: impl FnOnce(&mut A, &mut SolveContext<'_, '_>) -> Result<Prepared<P, T>>,
        verify: &F,
    ) -> Result<bool>
    where
        Q: Requirements<L, A>,
        P: Sync,
        T: Copy,
        O: Ledger<T>,
        F: Fn(usize, &mut L, &A, &P, &mut SolveContext<'_, '_>) -> Result<T> + Sync,
    {
        let count = reports.len();
        let verified = self.run(
            scope,
            identity,
            options,
            preflight,
            |_, _| Ok(()),
            |lanes, shared, batch, _, cx| {
                if count == 0 {
                    return Ok(true);
                }
                let prepared = match prepare(shared, cx)? {
                    Prepared::Ready(prepared) => prepared,
                    Prepared::Inconclusive(outcome) => {
                        batch.fill(count, O::stage(outcome));
                        return Ok(false);
                    }
                };
                let verified = batch.dispatch(
                    cx,
                    lanes,
                    options.execution_policy(),
                    identity,
                    &|index, lane, cx| verify(index, lane, shared, &prepared, cx).map(O::stage),
                )?;
                if verified {
                    cx.commit_accepted(scioxide_core::PartialResultTrust::CertificateVerified)?;
                }
                Ok(verified)
            },
            |&verified| verified,
        )?;
        for (report, &staged) in reports.iter_mut().zip(self.batch.staged(count)) {
            *report = staged.report();
        }
        Ok(verified)
    }
}

/// Solve state and the record of the most recent attempt.
#[derive(Debug, Clone)]
pub(crate) struct Shell {
    state: SolveState,
    attempt: Option<AttemptRecord<()>>,
}

impl Shell {
    pub(crate) const fn new() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
        }
    }

    /// The last attempt, if any.
    pub(crate) const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.attempt.as_ref()
    }

    /// Record a preflight rejection and return `error`. No callback ran.
    pub(crate) fn reject<T>(&mut self, error: SciOxideError) -> Result<T> {
        self.attempt = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            SolveCounts::new(),
            None,
        ));
        Err(error)
    }

    /// Run `kernel` in a root or child epoch of `scope` and record the attempt.
    /// `success` classifies an `Ok` value as succeeded or stopped.
    pub(crate) fn execute<T>(
        &mut self,
        scope: SolveScope<'_, '_>,
        identity: SolveIdentity,
        budgets: SolveBudgets,
        kernel: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T>,
        success: impl FnOnce(&T) -> bool,
    ) -> Result<T> {
        let before = self.state.epoch();
        let result = scope.run(&mut self.state, identity, budgets, kernel);
        let outcome = match &result {
            _ if self.state.epoch() == before => AttemptOutcome::PreflightRejected,
            Ok(value) if success(value) => AttemptOutcome::Succeeded,
            Ok(_) => AttemptOutcome::Stopped,
            Err(_) => AttemptOutcome::Failed,
        };
        self.attempt = Some(AttemptRecord::new(
            if outcome == AttemptOutcome::PreflightRejected {
                None
            } else {
                self.state.epoch()
            },
            AttemptKind::Call,
            outcome,
            if outcome == AttemptOutcome::PreflightRejected {
                SolveCounts::new()
            } else {
                self.state.counts()
            },
            None,
        ));
        result
    }
}

/// Run a closure in a fresh epoch of `scope` with local solve state, for calls that
/// need no workspace.
pub(crate) fn run_local<T>(
    scope: SolveScope<'_, '_>,
    identity: SolveIdentity,
    budgets: SolveBudgets,
    kernel: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T>,
) -> Result<T> {
    scope.run(&mut SolveState::new(), identity, budgets, kernel)
}

/// Child ledgers and staged outcomes of independent proofs over caller lanes.
#[derive(Debug)]
pub(crate) struct Batch<R> {
    children: Vec<PartitionedSolve>,
    staged: Vec<R>,
}

impl<R: Outcome> Batch<R> {
    pub(crate) const fn new() -> Self {
        Self {
            children: Vec::new(),
            staged: Vec::new(),
        }
    }
    /// Stored bytes after growing to `count` proofs.
    pub(crate) fn projected_bytes(&self, count: usize) -> Result<usize> {
        add(
            grown_bytes(&self.children, count)?,
            grown_bytes(&self.staged, count)?,
        )
    }
    /// Whether `count` proofs fit without growth.
    pub(crate) fn fits(&self, count: usize) -> bool {
        self.children.len() >= count && self.staged.len() >= count
    }
    /// Grow to `count` proofs; `blank` fills new staged outcomes.
    pub(crate) fn try_reserve(&mut self, count: usize, blank: R) -> Result<()> {
        if let Some(additional) = count.checked_sub(self.children.len()) {
            try_reserve(
                &mut self.children,
                additional,
                scioxide_core::quantity!("workspace.children"),
            )?;
            self.children.resize_with(count, PartitionedSolve::default);
        }
        grow(
            &mut self.staged,
            count,
            blank,
            scioxide_core::quantity!("workspace.reports"),
        )
    }

    /// Run `run(index, lane, cx)` for every lane in its own child epoch of `cx` and
    /// stage each outcome. A pending stop afterwards marks every outcome stopped.
    /// Returns whether every proof succeeded; the staged outcomes are read with
    /// [`Self::staged`].
    pub(crate) fn dispatch<L, F>(
        &mut self,
        cx: &mut SolveContext<'_, '_>,
        lanes: &mut [L],
        policy: ExecutionPolicy,
        identity: SolveIdentity,
        run: &F,
    ) -> Result<bool>
    where
        L: Send,
        F: Fn(usize, &mut L, &mut SolveContext<'_, '_>) -> Result<R> + Sync + ?Sized,
    {
        let count = lanes.len();
        let (children, staged) = (&mut self.children[..count], &mut self.staged[..count]);
        if count != 0 {
            with_partitioned_solves(
                cx,
                children,
                SolveBudgets::unbounded(),
                |execution, batch| {
                    partitioned(execution, batch, policy, identity, lanes, staged, run)
                },
            )?;
        }
        if let Some(reason) = cx.stop_reason()? {
            staged.fill(R::stopped(reason));
        }
        Ok(staged.iter().all(R::succeeded))
    }

    /// Outcomes staged by the last dispatch of `count` proofs.
    pub(crate) fn staged(&self, count: usize) -> &[R] {
        &self.staged[..count]
    }

    /// Stage a shared failure before any independent proof can run.
    pub(crate) fn fill(&mut self, count: usize, outcome: R) {
        self.staged[..count].fill(outcome);
    }
}

/// Partition lanes under `policy` and run each in its own child epoch.
fn partitioned<L, R, F>(
    execution: ExecutionContext<'_>,
    batch: PartitionSolveBatch<'_>,
    policy: ExecutionPolicy,
    identity: SolveIdentity,
    lanes: &mut [L],
    staged: &mut [R],
    run: &F,
) -> Result<()>
where
    L: Send,
    R: Send,
    F: Fn(usize, &mut L, &mut SolveContext<'_, '_>) -> Result<R> + Sync + ?Sized,
{
    let plan = execution.plan(policy, lanes.len(), LIMITS);
    let mut partitions: [_; MAX_WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
    let (mut lanes, mut batch, mut staged) = (lanes, batch, staged);
    for partition in plan.partitions() {
        let len = partition.range().len();
        let (lane_part, rest) = lanes.split_at_mut(len);
        lanes = rest;
        let (batch_part, rest) = batch.split_at(len)?;
        batch = rest;
        let (staged_part, rest) = staged.split_at_mut(len);
        staged = rest;
        partitions[partition.worker_index()] =
            PartitionSlot::new((lane_part, batch_part, staged_part));
    }
    execution.execute_partitioned(
        plan,
        &partitions[..plan.worker_count()],
        &|partition, (lanes, permits, staged), _| {
            let slots = lanes.iter_mut().zip(permits).zip(staged.iter_mut());
            for (index, ((lane, permit), outcome)) in partition.range().zip(slots) {
                // Proofs run under the batch's context, not the partition's, so their
                // capabilities do not depend on the schedule.
                *outcome = permit.run(execution, identity, |cx| run(index, lane, cx))?;
            }
            Ok(())
        },
    )
}

const BYTES: Quantity = scioxide_core::quantity!("workspace_bytes");

/// Checked byte sum.
pub(crate) fn add(a: usize, b: usize) -> Result<usize> {
    checked_add(a, b, BYTES)
}

/// Checked byte product.
pub(crate) fn mul(a: usize, b: usize) -> Result<usize> {
    checked_mul(a, b, BYTES)
}

/// Bytes `values` keeps after growing to at least `len` elements.
pub(crate) fn grown_bytes<T>(values: &[T], len: usize) -> Result<usize> {
    mul(values.len().max(len), core::mem::size_of::<T>())
}

/// Grow `values` to at least `len` elements filled with `fill`; never shrinks.
/// # Errors
/// Returns `AllocationFailure` when the storage cannot be reserved.
pub(crate) fn grow<T: Clone>(
    values: &mut Vec<T>,
    len: usize,
    fill: T,
    quantity: Quantity,
) -> Result<()> {
    if let Some(additional) = len.checked_sub(values.len()) {
        try_reserve(values, additional, quantity)?;
        values.resize(len, fill);
    }
    Ok(())
}

/// Reject kept storage beyond the configured byte limit.
/// # Errors
/// Returns `ResourceLimit` when `bytes` exceeds `limit`.
pub(crate) fn within_limit(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        Err(SciOxideError::resource_limit(BYTES, bytes, limit))
    } else {
        Ok(())
    }
}

/// Require an exact length.
/// # Errors
/// Returns `DimensionMismatch` when `actual != expected`.
pub(crate) fn exact(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}

/// Require borrowed storage of at least `required` elements.
/// # Errors
/// Returns `DimensionMismatch` when `actual < required`.
pub(crate) fn at_least(required: usize, actual: usize) -> Result<()> {
    if actual >= required {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(required, actual))
    }
}

/// The stop reason carried by an `ExecutionStopped` error.
pub(crate) fn stopped(error: &SciOxideError) -> Option<ExecutionStopReason> {
    if let SciOxideErrorView::ExecutionStopped { reason, .. } = error.view() {
        Some(reason)
    } else {
        None
    }
}
