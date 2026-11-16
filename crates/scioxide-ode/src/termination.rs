//! Work counting and the mapping from errors to termination statuses, for every
//! time-evolution family.
//!
//! Families report budget exhaustion, step-size exhaustion and cooperative stops as
//! unsuccessful terminations with their last accepted state. Every other error is a
//! failure. Work is counted only through [`admit`] and [`record`]: a denial is the
//! value [`Fault::Denied`], so a family loop classifies it without building an error,
//! and `?` turns it into the core budget error where an error must unwind.

use core::cell::Cell;

use scioxide_core::{
    Admission, BudgetKind, BudgetLimit, BudgetReport, ConvergenceReason, EvaluationRole,
    ExecutionStopReason, SciOxideError, SciOxideErrorView, SolveContext,
};

/// A stop that every family's termination enum can express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// This budget denied further work.
    Budget(BudgetKind),
    /// No representable or permitted smaller step remains.
    StepSize,
    /// Cancellation or a deadline.
    Execution(ExecutionStopReason),
}

/// Classify `error` as a shared stop, or `None` when it is a failure.
///
/// Budget stops are denied work, recognized by the budget report that
/// [`scioxide_core::Admission::require`] attaches; inner iteration limits without a
/// report remain numerical failures.
pub(crate) fn stop(error: &SciOxideError) -> Option<Stop> {
    match error.view() {
        SciOxideErrorView::ExecutionStopped { reason, .. } => Some(Stop::Execution(reason)),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::StepSizeLimit,
            ..
        } => Some(Stop::StepSize),
        SciOxideErrorView::ConvergenceFailure { diagnostics, .. } => diagnostics
            .budget()
            .map(|report| Stop::Budget(report.kind())),
        _ => None,
    }
}

/// Work charged to the solve budgets by one [`admit`] or [`record`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Work {
    /// One provider evaluation in this role.
    Provider(EvaluationRole),
    /// One iteration of this solve.
    Iteration,
    /// One iteration of an inner solve, counted as nested work.
    NestedIteration,
    /// One step proposal.
    Proposal,
    /// One step attempt: an iteration and a proposal, counted together.
    Step,
    /// This many mesh subdivisions.
    Subdivisions(usize),
}

impl From<EvaluationRole> for Work {
    fn from(role: EvaluationRole) -> Self {
        Self::Provider(role)
    }
}

/// Internal control flow of a family kernel: denied work or an error.
#[derive(Debug)]
pub(crate) enum Fault {
    /// A budget denied the work; no count changed.
    Denied(BudgetReport),
    /// Any other failure or stop, classified by [`stop`].
    Error(SciOxideError),
}

impl Fault {
    /// The shared stop of this fault, or `None` for a failure.
    pub(crate) fn stop(&self) -> Option<Stop> {
        match self {
            Self::Denied(report) => Some(Stop::Budget(report.kind())),
            Self::Error(error) => stop(error),
        }
    }
}

impl From<SciOxideError> for Fault {
    fn from(error: SciOxideError) -> Self {
        Self::Error(error)
    }
}

impl From<Fault> for SciOxideError {
    /// A denial becomes the core budget error of [`Admission::require`], which
    /// [`stop`] recognizes.
    fn from(fault: Fault) -> Self {
        match fault {
            Fault::Error(error) => error,
            Fault::Denied(report) => match Admission::<()>::Denied(report).require() {
                Err(error) => error,
                Ok(()) => {
                    SciOxideError::accounting_invariant("a denied admission was granted", None)
                }
            },
        }
    }
}

/// Carries [`Fault::Denied`] through a callback boundary that propagates only
/// [`SciOxideError`], such as a provider trait or a nested solver, without building the
/// core budget error.
///
/// [`Self::carry`] stashes a denial and lets an inline marker error cross the boundary;
/// [`Self::restore`] returns the stashed denial after the boundary, so the marker never
/// leaves the crate.
#[derive(Debug, Default)]
pub(crate) struct Tunnel(Cell<Option<BudgetReport>>);

impl Tunnel {
    /// The error that crosses the boundary for `fault`.
    pub(crate) fn carry(&self, fault: Fault) -> SciOxideError {
        match fault {
            Fault::Denied(report) => {
                self.0.set(Some(report));
                let limit = match report.limit() {
                    BudgetLimit::Finite(limit) => limit,
                    BudgetLimit::Unbounded => usize::MAX,
                };
                SciOxideError::resource_limit(
                    scioxide_core::quantity!("solve_budget"),
                    report.consumed().saturating_add(1),
                    limit,
                )
            }
            Fault::Error(error) => error,
        }
    }

    /// The fault of `result` after the boundary: a stashed denial takes precedence
    /// over the error that carried it, and the stash is cleared.
    pub(crate) fn restore<R>(&self, result: Result<R, SciOxideError>) -> Result<R, Fault> {
        match (result, self.0.take()) {
            (Err(_), Some(report)) => Err(Fault::Denied(report)),
            (result, _) => result.map_err(Fault::Error),
        }
    }

    /// The stashed denial, if any, without clearing it.
    pub(crate) fn denied(&self) -> Option<BudgetReport> {
        self.0.get()
    }

    /// Record a denial reported through another error.
    pub(crate) fn mark(&self, report: BudgetReport) {
        self.0.set(Some(report));
    }

    /// Forget any stashed denial.
    pub(crate) fn clear(&self) {
        self.0.set(None);
    }
}

/// Count `work` in `cx` without polling execution capabilities, as per-evaluation
/// boundaries do.
///
/// # Errors
/// Returns [`Fault::Denied`] when a budget of `cx` or an ancestor denies the work, and
/// accounting errors.
#[inline(always)]
pub(crate) fn record(cx: &mut SolveContext<'_, '_>, work: impl Into<Work>) -> Result<(), Fault> {
    let admission = match work.into() {
        Work::Provider(role) => cx.record_provider(role)?,
        Work::Iteration => cx.record_iteration()?,
        Work::NestedIteration => cx.record_nested_iterations(1)?,
        Work::Proposal => cx.record_proposal()?,
        Work::Step => cx.record_work_unit(0, 0)?,
        Work::Subdivisions(count) => cx.try_consume(BudgetKind::Subdivision, count)?,
    };
    match admission {
        Admission::Granted(()) => Ok(()),
        Admission::Denied(report) => Err(Fault::Denied(report)),
    }
}

/// Poll execution capabilities, then [`record`] `work`.
///
/// # Errors
/// Returns execution stops and the [`record`] faults.
#[inline]
pub(crate) fn admit(cx: &mut SolveContext<'_, '_>, work: impl Into<Work>) -> Result<(), Fault> {
    cx.execution().check()?;
    record(cx, work)
}

/// Implemented by every family termination enum.
pub(crate) trait Terminal: Copy {
    fn from_stop(stop: Stop) -> Self;
    fn stop(self) -> Option<Stop>;
}

/// Implement [`Terminal`] for an enum with the shared variants.
macro_rules! shared_termination {
    ($name:ty) => {
        impl $name {
            /// Returns whether the integration reached the end of the interval or a
            /// terminal event (`ReachedEnd` or `TerminalEvent`).
            #[must_use]
            pub const fn is_success(self) -> bool {
                matches!(self, Self::ReachedEnd | Self::TerminalEvent)
            }
        }
        /// Writes the variant name, with the budget kind or stop reason in
        /// parentheses, such as `BudgetLimit(Evaluation)`.
        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self {
                    Self::ExecutionStopped(reason) => write!(f, "ExecutionStopped({reason})"),
                    other => write!(f, "{other:?}"),
                }
            }
        }
        impl $crate::termination::Terminal for $name {
            fn stop(self) -> Option<$crate::termination::Stop> {
                match self {
                    Self::BudgetLimit(k) => Some($crate::termination::Stop::Budget(k)),
                    Self::StepSizeLimit => Some($crate::termination::Stop::StepSize),
                    Self::ExecutionStopped(r) => Some($crate::termination::Stop::Execution(r)),
                    _ => None,
                }
            }
            fn from_stop(stop: $crate::termination::Stop) -> Self {
                match stop {
                    $crate::termination::Stop::Budget(kind) => Self::BudgetLimit(kind),
                    $crate::termination::Stop::StepSize => Self::StepSizeLimit,
                    $crate::termination::Stop::Execution(reason) => Self::ExecutionStopped(reason),
                }
            }
        }
    };
}
pub(crate) use shared_termination;

/// Converts an unsuccessful shared termination into an owned one-shot error.
pub(crate) fn stop_error(stop: Stop, counts: scioxide_core::SolveCounts) -> SciOxideError {
    let (reason, detail) = match stop {
        Stop::Execution(r) => return SciOxideError::execution_stopped(r),
        Stop::Budget(BudgetKind::Evaluation) => (
            ConvergenceReason::EvaluationLimit,
            "evaluation budget exhausted",
        ),
        Stop::Budget(BudgetKind::Iteration) => (
            ConvergenceReason::IterationLimit,
            "iteration budget exhausted",
        ),
        Stop::Budget(k) => (ConvergenceReason::ResourceLimit(k), "budget exhausted"),
        Stop::StepSize => (
            ConvergenceReason::StepSizeLimit,
            "no representable or permitted smaller step remains",
        ),
    };
    SciOxideError::convergence_failure(
        reason,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts),
        detail,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_denial_is_a_value_and_converts_to_the_core_budget_error() {
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "admission-test", 1),
            0,
            0,
        );
        let outcome = scioxide_core::SolveScope::from(scioxide_core::ExecutionContext::sequential())
            .run(
                &mut scioxide_core::SolveState::new(),
                identity,
                scioxide_core::SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, scioxide_core::Budget::ONE),
                |cx| {
                    admit(cx, EvaluationRole::RightHandSide)?;
                    let denied = admit(cx, EvaluationRole::RightHandSide);
                    Ok((
                        matches!(denied, Err(Fault::Denied(report)) if report.kind() == BudgetKind::Evaluation),
                        cx.counts().own().total_providers()?,
                    ))
                },
            );
        assert!(matches!(outcome, Ok((true, 1))));
        let error = SciOxideError::from(Fault::Denied(
            scioxide_core::SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, scioxide_core::Budget::ONE)
                .report(BudgetKind::Iteration),
        ));
        assert_eq!(stop(&error), Some(Stop::Budget(BudgetKind::Iteration)));
    }
}
