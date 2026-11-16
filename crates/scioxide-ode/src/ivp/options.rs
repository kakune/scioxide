//! Options of initial-value plans: error control, step bounds, output and budgets.

use alloc::vec::Vec;

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, NormSelection, NumericScalar, RealScalar, SciOxideError,
    SolveBudgets, Tolerance,
};

use crate::step_control::ProportionalIntegralController;
use crate::{EventSearchPolicy, OdeEventOptions};

/// Selects the step-size controller of an adaptive method.
///
/// The step controller turns the normalized local error estimate `r` of a step
/// into the next step size. A step is accepted when `r ≤ 1`.
/// [`OdeStepControl::MethodDefault`] uses integral control for the embedded
/// explicit Runge–Kutta methods, predictive control for [`RadauIia`](crate::RadauIia)
/// and PI control with [`ProportionalIntegralController::default`] for every
/// other adaptive method. Any other variant requires an adaptive method.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub enum OdeStepControl<S: RealScalar> {
    /// Uses the controller of the method.
    ///
    /// The integral control of the explicit Runge–Kutta methods keeps the step
    /// unchanged while `0.5 < r ≤ 0.75`, so that small fluctuations of the error
    /// estimate do not change the step.
    #[default]
    MethodDefault,
    /// Uses proportional–integral control on the current and the previous accepted
    /// error, with no hysteresis band. Set `alpha = 1` and `beta = 0` for integral
    /// control.
    ProportionalIntegral(ProportionalIntegralController<S>),
    /// Uses Gustafsson's predictive control, which takes the last accepted step and
    /// error and the number of Newton iterations of the step into account.
    ///
    /// The step ratio uses the fourth root of the current error and lies in
    /// `[0.2, 8]` (Hairer and Wanner, *Solving Ordinary Differential Equations II*,
    /// 2nd ed., Springer, 1996, §IV.8).
    Radau,
}

/// Selects which states an initial-value integration sends to the output sink.
#[derive(Debug, Clone, Default, PartialEq)]
#[must_use]
pub enum OdeOutputSchedule<S: RealScalar> {
    /// Sends only the final accepted state.
    #[default]
    FinalOnly,
    /// Sends the initial state and the end point of every accepted step.
    AcceptedSteps,
    /// Sends the dense-output interpolant at each of `times`, in integration order.
    ///
    /// When `initial` or `final_state` is set, the initial or final state is also
    /// sent, unless that time is already in `times`. Duplicate times are kept and
    /// reported with their index through
    /// [`OdeOutputSink::emit_requested`](crate::OdeOutputSink::emit_requested); the
    /// end points go to [`OdeOutputSink::emit`](crate::OdeOutputSink::emit). The
    /// times must be finite, lie in the integration span and be monotone in the
    /// integration direction. Requires a method with dense output.
    Times {
        /// Output times.
        times: Vec<S>,
        /// Also send the initial state.
        initial: bool,
        /// Also send the final accepted state.
        final_state: bool,
    },
}

impl<S: RealScalar> OdeOutputSchedule<S> {
    /// Returns [`Self::Times`] at `times`, without the initial and final states.
    pub fn times(times: Vec<S>) -> Self {
        Self::Times {
            times,
            initial: false,
            final_state: false,
        }
    }
}

/// Configures an initial-value integration: tolerances, step bounds, step control,
/// output, events and budgets.
///
/// The setters do not check their arguments; [`OdeMethod::validate`](crate::OdeMethod::validate)
/// validates the whole configuration.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct OdeOptions<S: RealScalar> {
    error_tolerance: ErrorTolerance<S>,
    initial_step: Option<S>,
    minimum_step: Option<S>,
    maximum_step: Option<S>,
    step_control: OdeStepControl<S>,
    output: OdeOutputSchedule<S>,
    events: OdeEventOptions<S>,
    max_steps: Option<Budget>,
    max_evaluations: Option<Budget>,
}

impl<S: RealScalar> Default for OdeOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> OdeOptions<S> {
    /// Returns the default options.
    ///
    /// The defaults are absolute tolerance `1e-6`, relative tolerance `1e-3` with the
    /// root-mean-square norm, an estimated initial step, no step bounds, the
    /// method's step controller, final-state output, default event options and no
    /// step or evaluation limit.
    pub fn new() -> Self {
        Self {
            error_tolerance: ErrorTolerance::from_tolerance(
                Tolerance::from_decimal_places(6, 3),
                NormSelection::RootMeanSquare,
            ),
            initial_step: None,
            minimum_step: None,
            maximum_step: None,
            step_control: OdeStepControl::MethodDefault,
            output: OdeOutputSchedule::FinalOnly,
            events: OdeEventOptions::new(),
            max_steps: None,
            max_evaluations: None,
        }
    }

    /// Sets the local error tolerance.
    ///
    /// A step with local error estimate `e` is accepted when the norm of
    /// `eᵢ / (atolᵢ + rtol · max(|yᵢ_old|, |yᵢ_new|))` over the components `i` is at
    /// most one. A vector absolute tolerance must have one entry per component.
    pub fn with_error_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.error_tolerance = tolerance;
        self
    }

    /// Sets the magnitude of the first step.
    ///
    /// Without it, adaptive methods estimate the first step from `f(t₀, y₀)` and
    /// one explicit Euler trial step (Hairer, Nørsett and Wanner, *Solving Ordinary
    /// Differential Equations I*, 2nd ed., Springer, 1993, §II.4). Fixed-step methods
    /// take every step at this size, or at the estimate, except a shortened last step.
    /// A step that would stop short of the end time by at most
    /// `1024·ε·max(|t|, |t_end|)`, a bound on the rounding that about two thousand
    /// accumulated step additions leave in `t`, ends at the end time instead, so `n`
    /// steps of `(t_end − t₀)/n` cover the span without a rounding-sized extra step.
    pub fn with_initial_step(mut self, step: S) -> Self {
        self.initial_step = Some(step);
        self
    }

    /// Sets the smallest step magnitude.
    ///
    /// Proposed steps below this size are raised to it. Only a step shortened to
    /// end at the end of the span, or at a breaking point of a delay equation, is
    /// shorter. A rejected step, or a step whose state the system rejects, is retried
    /// once at exactly this size if the retry would otherwise be shorter.
    /// Integration ends with
    /// [`OdeTermination::StepSizeLimit`](crate::OdeTermination::StepSizeLimit) when a
    /// step of this size, or a shorter final step, is rejected.
    pub fn with_minimum_step(mut self, step: S) -> Self {
        self.minimum_step = Some(step);
        self
    }

    /// Sets the largest step magnitude.
    pub fn with_maximum_step(mut self, step: S) -> Self {
        self.maximum_step = Some(step);
        self
    }

    /// Sets the step controller; see [`OdeStepControl`].
    pub fn with_step_control(mut self, control: OdeStepControl<S>) -> Self {
        self.step_control = control;
        self
    }

    /// Sets which states are sent to the output sink; see [`OdeOutputSchedule`].
    pub fn with_output_schedule(mut self, schedule: OdeOutputSchedule<S>) -> Self {
        self.output = schedule;
        self
    }

    /// Sets how events are detected and located; see [`OdeEventOptions`].
    pub fn with_event_options(mut self, options: OdeEventOptions<S>) -> Self {
        self.events = options;
        self
    }

    /// Limits the number of attempted steps, counting rejected and retried steps.
    ///
    /// When the limit is reached, integration ends with
    /// [`OdeTermination::BudgetLimit`](crate::OdeTermination::BudgetLimit) of kind
    /// `Proposal`.
    pub fn with_max_steps(mut self, steps: Budget) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Limits the number of callback evaluations: right-hand side, Jacobian, mass
    /// matrix, event functions and stage solvers.
    ///
    /// When the limit is reached, integration ends with
    /// [`OdeTermination::BudgetLimit`](crate::OdeTermination::BudgetLimit) of kind
    /// `Evaluation`.
    pub fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    /// Returns the local error tolerance.
    pub const fn error_tolerance(&self) -> &ErrorTolerance<S> {
        &self.error_tolerance
    }
    /// Returns the magnitude of the first step, if set.
    pub const fn initial_step(&self) -> Option<S> {
        self.initial_step
    }
    /// Returns the smallest step magnitude, if set.
    pub const fn minimum_step(&self) -> Option<S> {
        self.minimum_step
    }
    /// Returns the largest step magnitude, if set.
    pub const fn maximum_step(&self) -> Option<S> {
        self.maximum_step
    }
    /// Returns the step controller.
    pub const fn step_control(&self) -> &OdeStepControl<S> {
        &self.step_control
    }
    /// Returns the output schedule.
    pub const fn output_schedule(&self) -> &OdeOutputSchedule<S> {
        &self.output
    }
    /// Returns the event options.
    pub const fn event_options(&self) -> OdeEventOptions<S> {
        self.events
    }
    /// Returns the limit on attempted steps, if set.
    pub const fn max_steps(&self) -> Option<Budget> {
        self.max_steps
    }
    /// Returns the limit on callback evaluations, if set.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    /// Budgets of one call.
    pub(crate) fn budgets(&self) -> SolveBudgets {
        let budgets = SolveBudgets::unbounded();
        let budgets = match self.max_steps {
            Some(steps) => budgets.with_limit(BudgetKind::Proposal, steps),
            None => budgets,
        };
        match self.max_evaluations {
            Some(evaluations) => budgets.with_limit(BudgetKind::Evaluation, evaluations),
            None => budgets,
        }
    }

    /// Requested output times, if any.
    pub(crate) fn requested_times(&self) -> Option<&[S]> {
        match &self.output {
            OdeOutputSchedule::Times { times, .. } => Some(times),
            _ => None,
        }
    }

    /// Whether requested-time output also emits the initial and final states.
    pub(crate) const fn output_endpoints(&self) -> (bool, bool) {
        match &self.output {
            OdeOutputSchedule::Times {
                initial,
                final_state,
                ..
            } => (*initial, *final_state),
            _ => (false, false),
        }
    }

    /// Validate everything that does not depend on the problem.
    pub(crate) fn validate(&self, dimension: usize) -> Result<(), SciOxideError>
    where
        S: NumericScalar<Real = S>,
    {
        if let Some(absolute) = self.error_tolerance.absolute().as_vector()
            && absolute.len() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(dimension),
                scioxide_core::Dimensions::one(absolute.len()),
            ));
        }
        for step in [self.initial_step, self.minimum_step, self.maximum_step]
            .into_iter()
            .flatten()
        {
            crate::safety::validate_positive_finite(step, scioxide_core::argument!("options"))?;
        }
        crate::safety::validate_step_bounds(self.minimum_step, self.maximum_step)?;
        if let Some(times) = self.requested_times()
            && times.iter().any(|time| !time.is_finite())
        {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "requested output times must be finite",
            ));
        }
        self.events.validate()?;
        S::try_from_usize_exact(self.events.search().subdivisions())?;
        Ok(())
    }

    /// Maximum event records one accepted step can produce for `events` functions.
    pub(crate) fn event_records(&self, events: usize) -> Result<usize, SciOxideError> {
        let per_cell = match self.events.search() {
            EventSearchPolicy::DerivativeAssisted(_) => 2,
            _ => 1,
        };
        events
            .checked_mul(self.events.search().subdivisions())
            .and_then(|records| records.checked_mul(per_cell))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ode_event_records"))
            })
    }
}
