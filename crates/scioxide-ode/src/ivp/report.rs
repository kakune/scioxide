//! Termination, reports, progress and owned solutions of initial-value integration.

use scioxide_core::{
    BudgetKind, EvaluationCounts, ExecutionStopReason, NumericScalar, PartialResultTrust,
    RealScalar, SciOxideError, SolveCounts,
};
use scioxide_linalg::Vector;

use crate::{EventSearchEvidence, OdeMethodSwitch, OdeState, OdeTrajectory};

/// Says why an initial-value integration stopped.
///
/// `ReachedEnd` and `TerminalEvent` are successful. The other variants end the
/// integration early with the last accepted state; see
/// [Termination](crate#termination). Failures are returned as errors instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OdeTermination {
    /// The integration reached the end time of the problem.
    ReachedEnd,
    /// A terminal event stopped the integration at the event time.
    TerminalEvent,
    /// A limit was reached: `Proposal` for attempted steps, `Evaluation` for
    /// callback evaluations, or a limit of an enclosing solve.
    BudgetLimit(BudgetKind),
    /// A step had to be retried, and no smaller step remains: the retry would be
    /// shorter than the minimum step, or would not advance the time to a different
    /// floating-point number.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the integration.
    ExecutionStopped(ExecutionStopReason),
}

crate::termination::shared_termination!(OdeTermination);

/// Summarizes one call of [`OdeMethod::advance`](crate::OdeMethod::advance): why it
/// stopped, the steps and evaluations, and the events.
///
/// The states and events themselves went to the output sink, and the final state
/// is in the [`OdeState`] passed to `advance`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OdeSolveReport<S: RealScalar> {
    pub(crate) termination: OdeTermination,
    pub(crate) final_time: S,
    pub(crate) last_step_error_estimate: Option<S>,
    pub(crate) counts: SolveCounts,
    pub(crate) steps: EvaluationCounts,
    pub(crate) trust: PartialResultTrust,
    pub(crate) events: usize,
    pub(crate) terminal_event: Option<usize>,
    pub(crate) event_search: Option<EventSearchEvidence>,
    pub(crate) method_switches: usize,
    pub(crate) last_method_switch: Option<OdeMethodSwitch<S>>,
}

impl<S: RealScalar> OdeSolveReport<S> {
    /// Returns why the integration stopped.
    ///
    /// It is always `Some`.
    pub const fn termination(&self) -> Option<OdeTermination> {
        Some(self.termination)
    }
    /// Returns whether the integration reached the end time or a terminal event.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the time of the last accepted state.
    pub const fn final_time(&self) -> S {
        self.final_time
    }
    /// Returns the normalized local error estimate of the last attempted step.
    ///
    /// A step is accepted when this value is at most one. It is `None` for
    /// methods without an error estimate, or when no step was attempted.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
    /// Returns the number of attempted steps, accepted or rejected.
    pub const fn attempted_steps(&self) -> usize {
        self.steps.proposals()
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.steps.accepted()
    }
    /// Returns the number of rejected steps.
    pub const fn rejected_steps(&self) -> usize {
        self.steps.rejected()
    }
    /// Returns the iterations and callback evaluations of this call, without nested
    /// solves.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the counts of this call, including nested solves; see
    /// [Solve accounting](scioxide_core#solve-accounting).
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns how far the last accepted state can be relied on.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the number of events sent to the sink.
    pub const fn events(&self) -> usize {
        self.events
    }
    /// Returns the index of the event that stopped the integration, if any.
    pub const fn terminal_event(&self) -> Option<usize> {
        self.terminal_event
    }
    /// Returns the event search policy and the number of event function
    /// evaluations; `None` for a problem without events.
    pub const fn event_search(&self) -> Option<EventSearchEvidence> {
        self.event_search
    }
    /// Returns the number of switches of an automatic method such as
    /// [`AdamsBdf`](crate::AdamsBdf).
    pub const fn method_switches(&self) -> usize {
        self.method_switches
    }
    /// Returns the last switch of an automatic method, if any.
    pub const fn last_method_switch(&self) -> Option<OdeMethodSwitch<S>> {
        self.last_method_switch
    }
}

/// Records the last accepted point and the step counts of one integration call.
///
/// [`OdeWorkspace::last_attempt`](crate::OdeWorkspace::last_attempt) holds it after
/// each call, and the [`scioxide_core::PartialFailure`] of
/// [`OdeMethod::solve`](crate::OdeMethod::solve) carries it, so that you can inspect or
/// continue an integration that failed.
#[derive(Debug, Clone, PartialEq)]
pub struct OdeProgress<S, T: NumericScalar = S> {
    pub(crate) time: S,
    pub(crate) state: Vector<T>,
    pub(crate) steps: EvaluationCounts,
    pub(crate) last_step_error_estimate: Option<S>,
    pub(crate) trust: PartialResultTrust,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeProgress<S, T> {
    pub(crate) fn new(dimension: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            time: S::ZERO,
            state: Vector::zeros(dimension)?,
            steps: EvaluationCounts::new(),
            last_step_error_estimate: None,
            trust: PartialResultTrust::DiagnosticOnly,
        })
    }
    /// Returns the time of the last accepted state.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Returns the last accepted state values.
    pub fn state(&self) -> &[T] {
        self.state.as_slice()
    }
    /// Returns the number of attempted steps, accepted or rejected.
    pub const fn attempted_steps(&self) -> usize {
        self.steps.proposals()
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.steps.accepted()
    }
    /// Returns the number of rejected steps.
    pub const fn rejected_steps(&self) -> usize {
        self.steps.rejected()
    }
    /// Returns the normalized local error estimate of the last attempted step.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
    /// Returns how far the last accepted state can be relied on.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the last accepted point as a state, to continue with
    /// [`OdeMethod::advance`](crate::OdeMethod::advance).
    pub fn into_state(self) -> OdeState<S, T> {
        OdeState::new(self.time, self.state)
    }
}
/// Holds the result of [`OdeMethod::solve`](crate::OdeMethod::solve): the final state,
/// the report and the trajectory.
#[derive(Debug, Clone)]
#[must_use]
pub struct OdeSolution<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    pub(crate) state: OdeState<S, T>,
    pub(crate) report: OdeSolveReport<S>,
    pub(crate) trajectory: OdeTrajectory<S, T>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> OdeSolution<S, T> {
    /// Returns the final accepted state.
    pub const fn state(&self) -> &OdeState<S, T> {
        &self.state
    }
    /// Returns the report.
    pub const fn report(&self) -> &OdeSolveReport<S> {
        &self.report
    }
    /// Returns the states and events chosen by the output schedule.
    pub const fn trajectory(&self) -> &OdeTrajectory<S, T> {
        &self.trajectory
    }
    /// Returns the final state, the report and the trajectory.
    pub fn into_parts(self) -> (OdeState<S, T>, OdeSolveReport<S>, OdeTrajectory<S, T>) {
        (self.state, self.report, self.trajectory)
    }
}
