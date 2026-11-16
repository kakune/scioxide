//! Termination status, initialization diagnostics and report of residual DAE
//! integration.

use scioxide_core::{BudgetKind, EvaluationCounts, ExecutionStopReason, RealScalar, SolveCounts};

/// Says why a DAE integration stopped.
///
/// The first five variants have the same meaning in every time-evolution family (see
/// [termination](crate#termination)). Every status follows an accepted initial pair, and
/// the [`DaeState`](crate::DaeState) holds the last accepted pair. Inconsistent initial
/// data, failed projection and stops before the initial pair is accepted are errors
/// instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DaeTermination {
    /// The integration reached the end time. For a call whose end time equals its start
    /// time, the initial pair was checked or projected.
    ReachedEnd,
    /// Not produced: DAE integration has no events.
    TerminalEvent,
    /// A budget of the options or of an enclosing solve ran out: `Proposal` for steps,
    /// `Evaluation` for callback evaluations or `Iteration` for Newton iterations.
    BudgetLimit(BudgetKind),
    /// No representable step advances the time toward the end.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the integration.
    ExecutionStopped(ExecutionStopReason),
    /// The number of consecutive rejected steps reached
    /// [`DaeNumericalOptions::with_rejection_limit`](crate::DaeNumericalOptions::with_rejection_limit).
    RejectionLimit,
}

crate::termination::shared_termination!(DaeTermination);

/// Reports the residuals and corrections of the initialization.
///
/// All values are infinity norms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DaeConsistencyReport<S: RealScalar> {
    pub(crate) initial_differential_residual: S,
    pub(crate) initial_algebraic_residual: S,
    pub(crate) differential_residual: S,
    pub(crate) algebraic_residual: S,
    pub(crate) state_correction: S,
    pub(crate) derivative_correction: S,
}

impl<S: RealScalar> DaeConsistencyReport<S> {
    /// Returns the differential residual of the supplied pair.
    pub const fn initial_differential_residual(&self) -> S {
        self.initial_differential_residual
    }
    /// Returns the algebraic residual of the supplied pair.
    pub const fn initial_algebraic_residual(&self) -> S {
        self.initial_algebraic_residual
    }
    /// Returns the differential residual of the accepted initial pair.
    pub const fn differential_residual(&self) -> S {
        self.differential_residual
    }
    /// Returns the algebraic residual of the accepted initial pair.
    pub const fn algebraic_residual(&self) -> S {
        self.algebraic_residual
    }
    /// Returns the change of the algebraic states.
    pub const fn state_correction(&self) -> S {
        self.state_correction
    }
    /// Returns the change of the differential derivatives.
    pub const fn derivative_correction(&self) -> S {
        self.derivative_correction
    }
}

/// Reports the status, work and error diagnostics of one
/// [`DaeBdf::advance`](crate::DaeBdf::advance) call.
///
/// The same report describes the progress of a failed call in
/// [`DaeWorkspace::last_attempt`](crate::DaeWorkspace::last_attempt) and in the
/// [`scioxide_core::PartialFailure`] of [`DaeBdf::solve`](crate::DaeBdf::solve); the
/// returned error then says why the call failed. Error estimates are local and do not
/// bound the global error.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DaeReport<S: RealScalar> {
    pub(crate) termination: Option<DaeTermination>,
    pub(crate) time: S,
    pub(crate) has_accepted_state: bool,
    pub(crate) steps: usize,
    pub(crate) counts: SolveCounts,
    pub(crate) consistency: Option<DaeConsistencyReport<S>>,
    pub(crate) differential_residual: S,
    pub(crate) algebraic_residual: S,
    pub(crate) last_step_error_estimate: Option<S>,
    pub(crate) differential_error: S,
    pub(crate) algebraic_error: S,
}

impl<S: RealScalar> DaeReport<S> {
    /// A report before initialization at `time`.
    pub(crate) fn initial(time: S) -> Self {
        Self {
            termination: None,
            time,
            has_accepted_state: false,
            steps: 0,
            counts: SolveCounts::new(),
            consistency: None,
            differential_residual: S::ZERO,
            algebraic_residual: S::ZERO,
            last_step_error_estimate: None,
            differential_error: S::ZERO,
            algebraic_error: S::ZERO,
        }
    }

    /// Returns why the integration stopped.
    ///
    /// It is `Some` for every report that [`DaeBdf::advance`](crate::DaeBdf::advance)
    /// returns. In the progress of a failed call it is `None`, unless a stop condition was
    /// reached before the failure.
    pub const fn termination(&self) -> Option<DaeTermination> {
        self.termination
    }
    /// Returns `true` if the integration reached the end time.
    pub const fn is_success(&self) -> bool {
        matches!(self.termination, Some(termination) if termination.is_success())
    }
    /// Returns the time of the last accepted pair, or the start time if none was accepted.
    pub const fn final_time(&self) -> S {
        self.time
    }
    /// Returns `true` if the state holds a pair accepted by this call, which meets the
    /// residual tolerances.
    pub const fn has_accepted_state(&self) -> bool {
        self.has_accepted_state
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Returns the number of rejected steps.
    pub const fn rejected_steps(&self) -> usize {
        self.counts.own().rejected()
    }

    /// Returns the evaluations and iterations of this call, including failed attempts and
    /// excluding nested solves.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this call and of nested solves.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns `ResidualChecked` once a pair has been accepted, otherwise
    /// `DiagnosticOnly`.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        if self.has_accepted_state {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        }
    }
    /// Returns the normalized local error estimate of the last attempted step, the larger
    /// of the differential and algebraic values.
    ///
    /// A value above one means the step was rejected; `None` means its Newton iteration did not
    /// converge.
    pub fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
    /// Returns the initialization diagnostics, once the initial residual has been
    /// evaluated.
    pub const fn consistency(&self) -> Option<DaeConsistencyReport<S>> {
        self.consistency
    }
    /// Returns the infinity norm of the differential residual at the last accepted pair.
    pub const fn differential_residual(&self) -> S {
        self.differential_residual
    }
    /// Returns the infinity norm of the algebraic residual at the last accepted pair.
    pub const fn algebraic_residual(&self) -> S {
        self.algebraic_residual
    }
    /// Returns the normalized local error estimate of the differential states at the last
    /// accepted step.
    pub const fn differential_error(&self) -> S {
        self.differential_error
    }
    /// Returns the normalized local error estimate of the algebraic states at the last
    /// accepted step.
    pub const fn algebraic_error(&self) -> S {
        self.algebraic_error
    }
}
