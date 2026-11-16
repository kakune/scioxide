//! Ensembles: many independent solves, run sequentially or in parallel.

use crate::BvpReport;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use scioxide_core::backend::{
    PartitionSlot, PartitionSolveBatch, PartitionedSolve, checked_mul, try_reserve,
    with_partitioned_solves,
};
use scioxide_core::{
    AlgorithmId, AttemptRecord, Capacity, ExecutionContext, ExecutionPolicy, NonFiniteOrigin,
    PlanLimits, ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveCounts,
    SolveIdentity, SolveScope,
};
use scioxide_linalg::MatrixView;

use crate::shell::Shell;
use crate::{AdjointReport, DaeReport, DdeReport, MechanicalReport, OdeSolveReport, SdeReport};

const ENSEMBLE: AlgorithmId = scioxide_core::algorithm_id!("ode", "ensemble", 1);
const MEMBER: AlgorithmId = scioxide_core::algorithm_id!("ode", "ensemble-member", 1);
/// Largest number of executor partitions of one ensemble run.
const MAX_WORKERS: usize = 64;
const LIMITS: PlanLimits = PlanLimits {
    auto_threshold: NonZeroUsize::MIN,
    work_per_item: NonZeroUsize::MIN,
    worker_limit: match NonZeroUsize::new(MAX_WORKERS) {
        Some(limit) => limit,
        None => NonZeroUsize::MIN,
    },
};

/// Says whether the result of one ensemble member is a success.
///
/// The ODE, SDE, DDE, DAE, mechanical, BVP and adjoint reports implement it. A member
/// can also return its own type, such as a report paired with values extracted from
/// the solution.
pub trait EnsembleOutcome {
    /// Returns `true` if the member reached its stopping condition.
    fn is_success(&self) -> bool;
}

macro_rules! report_outcome {
    ($($report:ident),*) => {$(
        impl<S: RealScalar> EnsembleOutcome for $report<S> {
            fn is_success(&self) -> bool {
                $report::is_success(self)
            }
        }
    )*};
}
report_outcome!(
    OdeSolveReport,
    SdeReport,
    DdeReport,
    DaeReport,
    MechanicalReport,
    AdjointReport,
    BvpReport
);
/// Gives the parameters of the ensemble members: one vector for all of them, or one
/// row per member.
#[derive(Debug, Clone, Copy)]
pub enum EnsembleParameters<'a, S: RealScalar> {
    /// Every member receives this vector; nothing is copied.
    Shared(&'a [S]),
    /// A `[members, parameters]` matrix; member `i` receives row `i`. Contiguous rows
    /// are passed directly, and rows of any other layout are copied into the
    /// workspace once per run.
    Independent(MatrixView<'a, S>),
}

/// Returns whether every row of `view` is a contiguous slice; rows share one layout.
fn borrows_rows<S: RealScalar>(view: MatrixView<'_, S>) -> bool {
    view.rows() == 0
        || view
            .row(0)
            .is_ok_and(|row| row.as_contiguous_slice().is_some())
}
/// Describes the workspace an [`solve_ensemble_into`] needs, returned by
/// [`EnsembleRequirements::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnsembleRequirements {
    members: usize,
    staged: usize,
}

impl EnsembleRequirements {
    /// Returns the number of members, each of which needs one accounting record.
    pub const fn members(&self) -> usize {
        self.members
    }
    /// Returns the number of parameter values copied from rows that are not
    /// contiguous.
    pub const fn staged_parameters(&self) -> usize {
        self.staged
    }
}

/// Summarizes one ensemble run: how many members finished and succeeded, and the
/// work done.
///
/// The outcome of each member, in the `outcomes` slice of
/// [`solve_ensemble_into`], has the details.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct EnsembleReport {
    requested: usize,
    completed: usize,
    successful: usize,
    counts: SolveCounts,
}

impl EnsembleReport {
    fn tally<R: EnsembleOutcome>(
        outcomes: &[Option<Result<R, SciOxideError>>],
        counts: SolveCounts,
    ) -> Self {
        Self {
            requested: outcomes.len(),
            completed: outcomes.iter().filter(|outcome| outcome.is_some()).count(),
            successful: outcomes
                .iter()
                .filter(|outcome| matches!(outcome, Some(Ok(result)) if result.is_success()))
                .count(),
            counts,
        }
    }
    /// Returns `true` if every member ran and succeeded.
    pub const fn is_success(&self) -> bool {
        self.completed == self.requested && self.successful == self.requested
    }
    /// Returns the number of members that ran and have an outcome.
    pub const fn completed(&self) -> usize {
        self.completed
    }
    /// Returns the number of members whose result is a success, as
    /// [`EnsembleOutcome::is_success`] defines it.
    pub const fn successful(&self) -> usize {
        self.successful
    }
    /// Returns the work of the run, which includes the work of every member.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
}

/// Holds the per-member accounting records, copied parameter rows and the record of
/// the last run of an [`solve_ensemble_into`].
///
/// The numerical storage of each member, such as its own solver workspace, belongs to
/// the member. One workspace can serve several plans and keeps the largest size it
/// has reached.
pub struct EnsembleWorkspace<S> {
    staging: Vec<S>,
    children: Vec<PartitionedSolve>,
    shell: Shell<EnsembleReport>,
}

impl<S: RealScalar> Default for EnsembleWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> core::fmt::Debug for EnsembleWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EnsembleWorkspace")
            .field("members", &self.children.len())
            .field("staged_parameters", &self.staging.len())
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> EnsembleWorkspace<S> {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            staging: Vec::new(),
            children: Vec::new(),
            shell: Shell::new(),
        }
    }

    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Does not fail. The `Result` follows the shared
    /// [workspace protocol](scioxide_core#plans-and-workspaces).
    pub fn check(&self, requirements: &EnsembleRequirements) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.children.len() >= requirements.members
                && self.staging.len() >= requirements.staged
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the workspace to meet `requirements`.
    ///
    /// After a failure the workspace may keep part of the growth.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &EnsembleRequirements,
    ) -> Result<(), SciOxideError> {
        if self.children.len() < requirements.members {
            let additional = requirements.members - self.children.len();
            try_reserve(
                &mut self.children,
                additional,
                scioxide_core::quantity!("ensemble_members"),
            )?;
            self.children
                .resize_with(requirements.members, PartitionedSolve::default);
        }
        if self.staging.len() < requirements.staged {
            let additional = requirements.staged - self.staging.len();
            try_reserve(
                &mut self.staging,
                additional,
                scioxide_core::quantity!("ensemble_parameters"),
            )?;
            self.staging.resize(requirements.staged, S::ZERO);
        }
        Ok(())
    }

    /// Returns the record of the last run, with its [`EnsembleReport`] when the run
    /// started.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<EnsembleReport>> {
        self.shell.last_attempt()
    }
}

/// Parameter rows as the members receive them.
#[derive(Clone, Copy)]
enum Rows<'a, S: RealScalar> {
    Shared(&'a [S]),
    Borrowed(MatrixView<'a, S>),
    Staged { values: &'a [S], width: usize },
}

impl<'a, S: RealScalar> Rows<'a, S> {
    fn row(self, index: usize) -> Result<&'a [S], SciOxideError> {
        match self {
            Self::Shared(values) => Ok(values),
            Self::Borrowed(view) => view.row(index)?.as_contiguous_slice().ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("parameters"),
                    "borrowed ensemble parameter rows must be contiguous",
                )
            }),
            Self::Staged { values, width } => Ok(&values[index * width..(index + 1) * width]),
        }
    }
}

fn preflight<S: RealScalar>(
    parameters: EnsembleParameters<'_, S>,
    members: usize,
    outcomes: usize,
    workspace: &mut EnsembleWorkspace<S>,
) -> Result<(), SciOxideError> {
    let count = parameters.count();
    let requirements = EnsembleRequirements::new(members, parameters)?;
    if outcomes != members {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(members),
            scioxide_core::Dimensions::one(outcomes),
        ));
    }
    match parameters {
        EnsembleParameters::Shared(values) => {
            crate::safety::finite(values, NonFiniteOrigin::Input)?;
        }
        EnsembleParameters::Independent(view) => {
            for row in 0..members {
                for col in 0..count {
                    crate::safety::finite(&[view.get(row, col)?], NonFiniteOrigin::Input)?;
                }
            }
        }
    }
    if workspace.check(&requirements)? == Capacity::NeedsGrowth {
        workspace.try_reserve(&requirements)?;
    }
    if let EnsembleParameters::Independent(view) = parameters
        && requirements.staged > 0
    {
        let staged = workspace.staging[..requirements.staged].chunks_exact_mut(count);
        for (row, values) in staged.enumerate() {
            for (col, value) in values.iter_mut().enumerate() {
                *value = view.get(row, col)?;
            }
        }
    }
    Ok(())
}
/// Calls `run(index, parameters, member, cx)` for every member and stores its result
/// in `outcomes[index]`, then returns the [`EnsembleReport`].
///
/// The run is one solve in `scope`, and each member is a solve nested in it, with
/// the context `cx` that `run` receives; pass `cx` to the member's own solver, for
/// example to its plan's `advance`. See
/// [Solve accounting](scioxide_core#solve-accounting). With
/// [`ExecutionPolicy::Sequential`] the members run in index order; other policies
/// divide them among the workers of the executor in the context.
///
/// The method checks the shapes and parameters first; if that fails, no member runs
/// and `members` and `outcomes` are unchanged. Otherwise it sets every outcome to
/// `None` and then runs the members. An error of one member is stored as that
/// member's outcome and does not stop the others, and the method returns `Ok` with
/// an unsuccessful report. If the run itself stops, for example on cancellation
/// between parts, the outcomes of finished members are kept, the others stay `None`,
/// and [`EnsembleWorkspace::last_attempt`] holds the report. With a large enough
/// workspace the method itself does not allocate.
///
/// # Errors
///
/// - `DimensionMismatch` if `members`, `outcomes` or the parameters do not have the
///   plan's shape.
/// - `NonFinite` (origin `Input`) if a parameter is NaN or infinite.
/// - `AllocationFailure` if the workspace must grow and cannot.
/// - `ExecutionStopped` for cancellation or a deadline, and errors of the executor.
pub fn solve_ensemble_into<'scope, 'capability, S, M, R, F>(
    parameters: EnsembleParameters<'_, S>,
    run: &F,
    policy: ExecutionPolicy,
    members: &mut [M],
    outcomes: &mut [Option<Result<R, SciOxideError>>],
    workspace: &mut EnsembleWorkspace<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<EnsembleReport, SciOxideError>
where
    S: RealScalar,
    M: Send,
    R: EnsembleOutcome + Send,
    F: Fn(usize, &[S], &mut M, &mut SolveContext<'_, '_>) -> Result<R, SciOxideError>
        + Sync
        + ?Sized,
{
    let count = parameters.count();
    let member_count = members.len();
    let scope = scope.into();
    if let Err(error) = preflight(parameters, members.len(), outcomes.len(), workspace) {
        return workspace.shell.reject(error);
    }
    let EnsembleWorkspace {
        staging,
        children,
        shell,
    } = workspace;
    let rows = match parameters {
        EnsembleParameters::Shared(values) => Rows::Shared(values),
        EnsembleParameters::Independent(view) if borrows_rows(view) => Rows::Borrowed(view),
        EnsembleParameters::Independent(_) => Rows::Staged {
            values: &staging[..member_count * count],
            width: count,
        },
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        ENSEMBLE,
        ((member_count as u128) << 64) | count as u128,
        0,
    );
    let children = &mut children[..member_count];
    let mut summary = None;
    let executed = shell.execute(scope, identity, SolveBudgets::unbounded(), |cx| {
        outcomes.iter_mut().for_each(|outcome| *outcome = None);
        let dispatched = with_partitioned_solves(
            cx,
            children,
            SolveBudgets::unbounded(),
            |execution, batch| dispatch(execution, batch, policy, rows, run, members, outcomes),
        );
        let report = EnsembleReport::tally(outcomes, cx.counts());
        summary = Some(report);
        dispatched.map(|()| report)
    });
    shell.conclude(executed, EnsembleReport::is_success, |_| summary)
}

/// Partition members under `policy` and run each in its own child epoch.
fn dispatch<S, M, R, F>(
    execution: ExecutionContext<'_>,
    batch: PartitionSolveBatch<'_>,
    policy: ExecutionPolicy,
    rows: Rows<'_, S>,
    run: &F,
    members: &mut [M],
    outcomes: &mut [Option<Result<R, SciOxideError>>],
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    M: Send,
    R: Send,
    F: Fn(usize, &[S], &mut M, &mut SolveContext<'_, '_>) -> Result<R, SciOxideError>
        + Sync
        + ?Sized,
{
    let plan = execution.plan(policy, members.len(), LIMITS);
    let mut partitions: [_; MAX_WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
    let (mut members, mut batch, mut outcomes) = (members, batch, outcomes);
    for partition in plan.partitions() {
        let len = partition.range().len();
        let (member_part, rest) = members.split_at_mut(len);
        members = rest;
        let (batch_part, rest) = batch.split_at(len)?;
        batch = rest;
        let (outcome_part, rest) = outcomes.split_at_mut(len);
        outcomes = rest;
        partitions[partition.worker_index()] =
            PartitionSlot::new((member_part, batch_part, outcome_part));
    }
    execution.execute_partitioned(
        plan,
        &partitions[..plan.worker_count()],
        &|partition, (members, permits, outcomes), _| {
            let slots = members.iter_mut().zip(permits).zip(outcomes);
            for (index, ((member, permit), outcome)) in partition.range().zip(slots) {
                let parameters = rows.row(index)?;
                let identity = SolveIdentity::new(ProblemToken::new(0), MEMBER, index as u128, 0);
                // Members run under the ensemble's context, not the partition's, so their
                // capabilities and partition identities do not depend on the schedule.
                permit.run(execution, identity, |cx| {
                    *outcome = Some(run(index, parameters, member, cx));
                    Ok(())
                })?;
            }
            Ok(())
        },
    )
}

impl<S: RealScalar> EnsembleParameters<'_, S> {
    fn count(self) -> usize {
        match self {
            Self::Shared(p) => p.len(),
            Self::Independent(p) => p.cols(),
        }
    }
}
impl EnsembleRequirements {
    /// Computes retained storage for member accounting and strided parameter rows.
    /// # Errors
    /// Rejects no members, mismatched parameter rows and size overflow.
    pub fn new<S: RealScalar>(
        members: usize,
        parameters: EnsembleParameters<'_, S>,
    ) -> Result<Self, SciOxideError> {
        if members == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("members"),
                scioxide_core::InvalidInputReason::Empty,
                "an ensemble must contain at least one member",
            ));
        }
        let length = checked_mul(
            members,
            parameters.count(),
            scioxide_core::quantity!("ensemble_parameters"),
        )?;
        if let EnsembleParameters::Independent(view) = parameters
            && view.rows() != members
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(members),
                scioxide_core::Dimensions::one(view.rows()),
            ));
        }
        let staged = match parameters {
            EnsembleParameters::Independent(view) if !borrows_rows(view) => length,
            _ => 0,
        };
        Ok(Self { members, staged })
    }
}
