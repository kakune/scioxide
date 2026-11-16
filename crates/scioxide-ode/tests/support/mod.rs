#![allow(dead_code)]

use std::boxed::Box;
use std::fmt;
use std::vec::Vec;

use scioxide_core::{
    Budget, ErrorTolerance, EvaluationCounts, EvaluationRole, ExecutionContext, NormSelection,
    NumericScalar, RealScalar, SciOxideError, StepTolerance, Tolerance,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    EventFunction, EventInfo, ForwardEuler, NonlinearStageSolver, OdeEventOptions, OdeMethod,
    OdeOptions, OdeOutputSchedule, OdeProblem, OdeSolveReport, OdeState, OdeStepControl, OdeSystem,
    OdeTermination, OdeTrajectory, OdeWorkspace, StageLinearSolver,
};

/// Scenario options for [`run_solve`]: accepted-step output unless evaluation times are
/// requested, RMS error control and a generous attempted-step budget.
pub struct ScenarioConfig<S: RealScalar + NumericScalar<Real = S>> {
    tolerance: Tolerance<S>,
    max_steps: Budget,
    initial_step: Option<S>,
    maximum_step: Option<S>,
    minimum_step: Option<S>,
    evaluation_times: Option<Vec<S>>,
    events: Vec<Box<dyn EventFunction<S>>>,
    event_tolerance: Tolerance<S>,
    max_event_iterations: Budget,
    step_control: OdeStepControl<S>,
    stage_solver: ScenarioStageSolver<S>,
}

pub enum ScenarioStageSolver<S: RealScalar + NumericScalar<Real = S>> {
    Native,
    Linear(Box<dyn StageLinearSolver<S>>),
    Nonlinear(Box<dyn NonlinearStageSolver<S>>),
}

/// Validate step options through the plan, as the solve will.
fn validate_steps<S: RealScalar + NumericScalar<Real = S>>(
    options: OdeOptions<S>,
) -> Result<(), SciOxideError> {
    (ForwardEuler::<S>::new())
        .validate(&(options), 1)
        .map(|_| ())
}

impl<S: RealScalar + NumericScalar<Real = S>> ScenarioConfig<S> {
    pub const fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub const fn with_max_steps(mut self, max_steps: Budget) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_initial_step(mut self, step: S) -> Result<Self, SciOxideError> {
        validate_steps(OdeOptions::new().with_initial_step(step))?;
        self.initial_step = Some(step);
        Ok(self)
    }

    pub fn with_max_step(mut self, step: S) -> Result<Self, SciOxideError> {
        let mut options = OdeOptions::new().with_maximum_step(step);
        if let Some(minimum) = self.minimum_step {
            options = options.with_minimum_step(minimum);
        }
        validate_steps(options)?;
        self.maximum_step = Some(step);
        Ok(self)
    }

    pub fn with_min_step(mut self, step: S) -> Result<Self, SciOxideError> {
        let mut options = OdeOptions::new().with_minimum_step(step);
        if let Some(maximum) = self.maximum_step {
            options = options.with_maximum_step(maximum);
        }
        validate_steps(options)?;
        self.minimum_step = Some(step);
        Ok(self)
    }

    pub fn with_evaluation_times(mut self, times: Vec<S>) -> Result<Self, SciOxideError> {
        if times.iter().copied().any(|time| !time.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "evaluation times must be finite",
            ));
        }
        self.evaluation_times = Some(times);
        Ok(self)
    }

    pub fn with_event(mut self, event: Box<dyn EventFunction<S>>) -> Self {
        self.events.push(event);
        self
    }

    pub fn with_events(mut self, events: Vec<Box<dyn EventFunction<S>>>) -> Self {
        self.events = events;
        self
    }

    pub const fn with_event_localization(
        mut self,
        tolerance: Tolerance<S>,
        max_iterations: Budget,
    ) -> Self {
        self.event_tolerance = tolerance;
        self.max_event_iterations = max_iterations;
        self
    }

    pub fn with_step_control(mut self, control: OdeStepControl<S>) -> Self {
        self.step_control = control;
        self
    }

    pub fn with_stage_solver(mut self, solver: ScenarioStageSolver<S>) -> Self {
        self.stage_solver = solver;
        self
    }

    /// The plan options this scenario configures.
    pub fn options(&self) -> OdeOptions<S> {
        let mut options = OdeOptions::new()
            .with_error_tolerance(ErrorTolerance::from_tolerance(
                self.tolerance,
                NormSelection::RootMeanSquare,
            ))
            .with_event_options(OdeEventOptions::new().with_localization(
                StepTolerance::from_tolerance(self.event_tolerance),
                self.max_event_iterations,
            ))
            .with_max_steps(self.max_steps)
            .with_step_control(self.step_control.clone())
            .with_output_schedule(
                self.evaluation_times
                    .clone()
                    .map_or(OdeOutputSchedule::AcceptedSteps, OdeOutputSchedule::times),
            );
        if let Some(step) = self.initial_step {
            options = options.with_initial_step(step);
        }
        if let Some(step) = self.minimum_step {
            options = options.with_minimum_step(step);
        }
        if let Some(step) = self.maximum_step {
            options = options.with_maximum_step(step);
        }
        options
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for ScenarioConfig<S> {
    fn default() -> Self {
        Self {
            tolerance: Tolerance::from_decimal_places(6, 3),
            max_steps: Budget::ONE.saturating_add(99_999),
            initial_step: None,
            maximum_step: None,
            minimum_step: None,
            evaluation_times: None,
            events: Vec::new(),
            event_tolerance: Tolerance::from_sqrt_epsilon(),
            max_event_iterations: Budget::ONE.saturating_add(63),
            step_control: OdeStepControl::MethodDefault,
            stage_solver: ScenarioStageSolver::Native,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> fmt::Debug for ScenarioConfig<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScenarioConfig")
            .field("tolerance", &self.tolerance)
            .field("max_steps", &self.max_steps)
            .field("initial_step", &self.initial_step)
            .field("maximum_step", &self.maximum_step)
            .field("minimum_step", &self.minimum_step)
            .field("evaluation_times", &self.evaluation_times)
            .field("event_count", &self.events.len())
            .field("event_tolerance", &self.event_tolerance)
            .field("max_event_iterations", &self.max_event_iterations)
            .field("step_control", &self.step_control)
            .field(
                "has_linear_solver",
                &matches!(self.stage_solver, ScenarioStageSolver::Linear(_)),
            )
            .field(
                "has_nonlinear_solver",
                &matches!(self.stage_solver, ScenarioStageSolver::Nonlinear(_)),
            )
            .finish()
    }
}

/// Epoch work of one solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioStats(EvaluationCounts);

impl ScenarioStats {
    pub const fn counts(self) -> EvaluationCounts {
        self.0
    }

    pub const fn num_steps(self) -> usize {
        self.0.iterations()
    }

    pub const fn num_accepted(self) -> usize {
        self.0.accepted()
    }

    pub const fn num_rejected(self) -> usize {
        self.0.rejected()
    }

    pub const fn num_rhs_evals(self) -> usize {
        self.0.provider(EvaluationRole::RightHandSide)
    }

    pub const fn num_jac_evals(self) -> usize {
        self.0.provider(EvaluationRole::Jacobian)
    }

    pub const fn num_linear_solves(self) -> usize {
        self.0.provider(EvaluationRole::LinearSolve)
    }

    pub const fn num_factorizations(self) -> usize {
        self.0.provider(EvaluationRole::Factorization)
    }

    pub const fn num_nonlinear_iters(self) -> usize {
        0
    }

    pub const fn num_jacobian_cache_hits(self) -> usize {
        self.0.cache_hits()
    }

    pub const fn num_factor_cache_hits(self) -> usize {
        self.0.cache_hits()
    }

    pub const fn num_jacobian_cache_invalidations(self) -> usize {
        0
    }

    pub const fn num_factor_cache_invalidations(self) -> usize {
        0
    }
}

/// Samples, events and report of one scenario solve.
#[derive(Debug, Clone)]
pub struct ObservedTrajectory<S: RealScalar + NumericScalar<Real = S>> {
    times: Vec<S>,
    states: Vec<Vector<S>>,
    events: Vec<EventInfo<S>>,
    final_state: OdeState<S>,
    report: OdeSolveReport<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> ObservedTrajectory<S> {
    pub fn times(&self) -> &[S] {
        &self.times
    }

    pub fn states(&self) -> &[Vector<S>] {
        &self.states
    }

    pub fn events(&self) -> &[EventInfo<S>] {
        &self.events
    }

    /// The caller-owned state after the solve.
    pub const fn final_state(&self) -> &OdeState<S> {
        &self.final_state
    }

    pub const fn report(&self) -> &OdeSolveReport<S> {
        &self.report
    }

    pub const fn termination(&self) -> Option<OdeTermination> {
        self.report.termination()
    }

    pub const fn is_success(&self) -> bool {
        self.report.is_success()
    }

    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.report.last_step_error_estimate()
    }

    pub const fn stats(&self) -> ScenarioStats {
        ScenarioStats(self.report.counts())
    }

    pub fn message(&self) -> &'static str {
        match self.report.termination() {
            Some(OdeTermination::ReachedEnd) => "reached end",
            Some(OdeTermination::TerminalEvent) => "terminal event",
            Some(OdeTermination::BudgetLimit(scioxide_core::BudgetKind::Proposal)) => "step limit",
            Some(OdeTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)) => {
                "evaluation limit"
            }
            Some(OdeTermination::BudgetLimit(_)) => "budget limit",
            Some(OdeTermination::StepSizeLimit) => "step size limit",
            Some(OdeTermination::ExecutionStopped(_)) => "execution stopped",
            _ => "unknown termination",
        }
    }
}

/// Integrate `system` over `time_span` with a fresh workspace under sequential native
/// execution, collecting every emitted sample and event.
pub fn run_solve<S, System, M>(
    system: &System,
    method: &M,
    time_span: (S, S),
    initial_state: Vector<S>,
    config: ScenarioConfig<S>,
) -> Result<ObservedTrajectory<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S> + ?Sized,
    M: OdeMethod<S> + Clone,
{
    run_solve_with_workspace(
        system,
        method,
        &mut OdeWorkspace::new(),
        time_span,
        initial_state,
        config,
    )
}

/// [`run_solve`] with a caller-owned (possibly reused) workspace.
pub fn run_solve_with_workspace<S, System, M>(
    system: &System,
    method: &M,
    workspace: &mut OdeWorkspace<S, M>,
    time_span: (S, S),
    initial_state: Vector<S>,
    config: ScenarioConfig<S>,
) -> Result<ObservedTrajectory<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S> + ?Sized,
    M: OdeMethod<S> + Clone,
{
    let plan = (method.clone(), config.options());
    let events: Vec<&dyn EventFunction<S>> = config.events.iter().map(Box::as_ref).collect();
    let problem = OdeProblem::new(system, time_span.1).with_events(&events);
    let problem = match &config.stage_solver {
        ScenarioStageSolver::Native => problem,
        ScenarioStageSolver::Linear(solver) => {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Linear(solver.as_ref()))
        }
        ScenarioStageSolver::Nonlinear(solver) => {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Nonlinear(solver.as_ref()))
        }
    };
    let mut state = OdeState::new(time_span.0, initial_state);
    let mut trajectory = OdeTrajectory::new();
    let report = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut trajectory,
        workspace,
        ExecutionContext::sequential_native(),
    )?;
    let (times, states, events) = trajectory.into_parts();
    Ok(ObservedTrajectory {
        times,
        states,
        events,
        final_state: state,
        report,
    })
}
