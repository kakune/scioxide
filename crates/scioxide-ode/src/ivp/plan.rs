//! Initial-value methods and their `advance` and `solve` forms.

use core::marker::PhantomData;

use scioxide_core::{
    ExecutionContext, NumericScalar, PartialFailure, PartialResultTrust, ProblemToken, RealScalar,
    SciOxideError, SolveIdentity, SolveScope,
};

use super::driver::{Drive, Finish, Outcome, drive, validate_requested_times};
use super::workspace::{IvpStorage, OdeRequirements};
use crate::stepper::{MethodInfo, Stepper};
use crate::termination::Terminal;
use crate::{
    EventSearchEvidence, EventSearchPolicy, OdeOptions, OdeOutputSink, OdeProblem, OdeProgress,
    OdeSolution, OdeSolveReport, OdeState, OdeStepControl, OdeSystem, OdeTermination,
    OdeTrajectory, OdeWorkspace,
};
/// The error for a feature that the problem or options need and the method lacks.
pub(crate) fn unsupported(name: &'static str, feature: &str) -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("method"),
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        alloc::format!("{name} does not support the requested feature: {feature}"),
    )
}
/// Integrates an initial-value problem using a built-in method.
///
/// Use `validate` to check options, `workspace_requirements` to size reusable
/// storage, `workspace` to allocate it, `advance` to integrate in place, and
/// `solve` to allocate a complete solution. Options are supplied per call.
/// Implemented by every built-in method, including `DormandPrince5`, `RadauIia`
/// and `BuiltinOdeMethod`; external implementations are sealed.
#[allow(private_bounds)]
pub trait OdeMethod<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S>:
    Stepper<S, T>
{
    /// Returns the linear stability along the real and imaginary axes.
    fn linear_stability(&self) -> crate::OdeLinearStability<S> {
        self.stability()
    }
    /// Validates this method and `options` for `dimension` state components.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a vector absolute tolerance does not have
    ///   `dimension` entries.
    /// - `NonFinite` (`Input`) if a step size or an output time is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a step size is not positive, or the event
    ///   contact scale is not positive.
    /// - `InvalidInput` (`Inconsistent`) if the minimum step exceeds the maximum
    ///   step.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the options request output
    ///   times and the method has no dense output, or select a step controller other
    ///   than [`OdeStepControl::MethodDefault`] for a fixed-step method.
    fn validate(&self, options: &OdeOptions<S>, dimension: usize) -> Result<(), SciOxideError>
    where
        Self: Sized + Clone,
    {
        let info = Stepper::<S, T>::info(self);
        if info.order == 0 || info.initial_step_order == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("method"),
                scioxide_core::InvalidInputReason::Empty,
                "method order must be at least 1",
            ));
        }
        options.validate(dimension)?;
        if options.requested_times().is_some() && !info.dense_output {
            return Err(unsupported(
                info.name,
                "requested output times need dense output",
            ));
        }
        if !matches!(options.step_control(), OdeStepControl::MethodDefault) && !info.adaptive {
            return Err(unsupported(
                info.name,
                "explicit step control needs adaptive control",
            ));
        }
        Ok(())
    }
    /// Returns the workspace size needed to integrate `problem`.
    ///
    /// The size depends on the method, the dimension, the number of event
    /// functions and whether the problem supplies a stage solver.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::validate`].
    /// - `SizeOverflow` if the number of event records overflows `usize`.
    fn workspace_requirements<Sys: OdeSystem<S, T> + ?Sized>(
        &self,
        problem: &OdeProblem<'_, S, Sys, T>,
        options: &OdeOptions<S>,
    ) -> Result<OdeRequirements<S, Self, T>, SciOxideError>
    where
        Self: Sized + Clone,
    {
        self.validate(options, problem.system().dimension())?;
        let events = problem.events().len();
        Ok(OdeRequirements {
            dimension: problem.system().dimension(),
            events,
            event_records: options.event_records(events)?,
            provided: problem.has_stage_solver(),
            method: self.clone(),
            scalar: PhantomData,
        })
    }

    /// Allocates a workspace large enough to integrate `problem`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::validate`].
    /// - `SizeOverflow` if a workspace size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    fn workspace<Sys: OdeSystem<S, T> + ?Sized>(
        &self,
        problem: &OdeProblem<'_, S, Sys, T>,
        options: &OdeOptions<S>,
    ) -> Result<OdeWorkspace<S, Self, T>, SciOxideError>
    where
        Self: Sized + Clone,
    {
        let mut workspace = OdeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem, options)?)?;
        Ok(workspace)
    }
    /// Integrates `problem` from `state` to the end time of the problem.
    ///
    /// On return, `state` holds the last accepted point: the end time, the time of
    /// a terminal event, or the point where a limit stopped integration. `sink`
    /// receives the states chosen by the [`OdeOutputSchedule`](crate::OdeOutputSchedule)
    /// and the events of each step as the step is accepted; pass `&mut ()` to
    /// discard them. Each call starts the method afresh from `state`, without the
    /// step size or history of an earlier call. A workspace that is too small for
    /// `problem` is enlarged first.
    ///
    /// `scope` is an [`ExecutionContext`], which starts a new solve, or the
    /// [`SolveContext`](scioxide_core::SolveContext) of an enclosing solve, whose
    /// limits and cancellation then also apply; see
    /// [Solve accounting](scioxide_core#solve-accounting). A step or evaluation limit,
    /// a step size below the minimum, cancellation or a deadline returns `Ok` with
    /// an unsuccessful [`OdeTermination`]. With a large enough workspace, and
    /// callbacks and a sink that do not allocate, `advance` does not allocate.
    ///
    /// # Errors
    ///
    /// Checked before any callback runs, leaving `state` unchanged:
    ///
    /// - The errors of [`Self::validate`] for this call's options.
    ///
    /// - `DimensionMismatch` if the system dimension or the length of `state` is not
    ///   the problem's dimension.
    /// - `NonFinite` (`Input`) if the start time, the end time or a state component
    ///   is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if the start time equals the end time.
    /// - `InvalidInput` (`OutOfRange`) if an output time lies outside the span or
    ///   the output times are not monotone in the integration direction.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the problem has events, a
    ///   mass matrix or a stage solver that the method does not support, or if
    ///   derivative-assisted event search is selected and an event function has no
    ///   total derivative.
    ///
    /// During integration, `state` holds the last accepted point and
    /// [`OdeWorkspace::last_attempt`] records the progress:
    ///
    /// - errors returned by the system, an event function or the sink, unchanged;
    /// - `NonFinite` (`Callback`) if a callback returns NaN or infinity at an
    ///   accepted state, or at a trial state of a fixed-step method;
    /// - `ConvergenceFailure`, `Singular` or `NumericalInstability` if a step of a
    ///   fixed-step method fails.
    ///
    /// An adaptive method retries a failed trial step at half the size and ends
    /// with `StepSizeLimit` when no smaller step remains.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, SciOxideError};
    /// use scioxide_linalg::Vector;
    /// use scioxide_ode::{DormandPrince5, OdeFn, OdeMethod, OdeOptions, OdeProblem, OdeState};
    ///
    /// // y′ = −y, y(0) = 1, so y(1) = e⁻¹.
    /// let rhs = OdeFn::new(1, |_t: f64, y: &[f64], dy: &mut [f64]| {
    ///     dy[0] = -y[0];
    ///     Ok(())
    /// })?;
    /// let (method, options) = (DormandPrince5::new(), OdeOptions::new());
    /// let problem = OdeProblem::new(&rhs, 1.0);
    /// let mut workspace = method.workspace(&problem, &options)?;
    /// let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    /// let cx = ExecutionContext::sequential_native();
    /// let report = method.advance(&problem, &options, &mut state, &mut (), &mut workspace, cx)?;
    /// assert!(report.is_success());
    /// assert!((state.values()[0] - (-1.0_f64).exp()).abs() < 1e-3);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    fn advance<'scope, 'capability, Sys, K>(
        &self,
        problem: &OdeProblem<'_, S, Sys, T>,
        options: &OdeOptions<S>,
        state: &mut OdeState<S, T>,
        sink: &mut K,
        workspace: &mut OdeWorkspace<S, Self, T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<OdeSolveReport<S>, SciOxideError>
    where
        Self: Sized + Clone,
        Sys: OdeSystem<S, T> + ?Sized,
        K: OdeOutputSink<S, T> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = preflight(self, problem, options, state, workspace) {
            return workspace.shell.reject(error);
        }
        let run = Drive {
            method: self,
            info: Stepper::<S, T>::info(self),
            options,
            tolerance: options.error_tolerance(),
            system: problem.system(),
            events: problem.events(),
            delay: None,
            provider: problem.stage_provider(),
            end: problem.end(),
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            Stepper::<S, T>::info(self).algorithm,
            problem.system().dimension() as u128,
            0,
        );
        let OdeWorkspace { storage, shell } = workspace;
        let (time, values) = state.parts_mut();
        let search = options.event_options().search();
        let has_events = !problem.events().is_empty();
        let executed = shell.execute(scope, identity, options.budgets(), |cx| {
            let outcome = drive(&run, storage, time, values, sink, &mut (), cx)?;
            Ok(report(
                outcome,
                ode_termination(outcome.finish)?,
                *time,
                storage,
                cx,
                has_events.then_some(search),
            ))
        });
        shell.conclude(executed, OdeSolveReport::is_success, |recycled| {
            progress(recycled, *time, values.as_slice(), storage)
        })
    }

    /// Integrates `problem` from `initial` and returns the final state, the report
    /// and an [`OdeTrajectory`] of the output.
    ///
    /// This is [`Self::advance`] with a new workspace, an [`OdeTrajectory`] as sink
    /// and [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::advance`]. The
    /// [`PartialFailure`] carries the last accepted [`OdeProgress`] when the error
    /// occurred after integration started.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::SciOxideError;
    /// use scioxide_linalg::Vector;
    /// use scioxide_ode::{DormandPrince5, OdeFn, OdeOptions, OdeOutputSchedule, OdeMethod};
    /// use scioxide_ode::{OdeProblem, OdeState};
    ///
    /// // y′ = −2y, y(0) = 1, so y(1) = e⁻².
    /// let rhs = OdeFn::new(1, |_t: f64, y: &[f64], dy: &mut [f64]| {
    ///     dy[0] = -2.0 * y[0];
    ///     Ok(())
    /// })?;
    /// let schedule = OdeOutputSchedule::AcceptedSteps;
    /// let options = OdeOptions::new().with_output_schedule(schedule);
    /// let plan = (DormandPrince5::new(), options);
    /// let problem = OdeProblem::new(&rhs, 1.0);
    /// let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    /// let solution = plan.0.solve(&problem, &plan.1, initial).map_err(SciOxideError::from)?;
    /// assert!((solution.state().values()[0] - (-2.0_f64).exp()).abs() < 1e-3);
    /// assert!(solution.trajectory().times().len() > 2);
    /// # Ok::<(), SciOxideError>(())
    /// ```
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    fn solve<Sys>(
        &self,
        problem: &OdeProblem<'_, S, Sys, T>,
        options: &OdeOptions<S>,
        initial: OdeState<S, T>,
    ) -> Result<OdeSolution<S, T>, PartialFailure<OdeProgress<S, T>>>
    where
        Self: Sized + Clone,
        Sys: OdeSystem<S, T> + ?Sized,
    {
        let mut workspace = self
            .workspace(problem, options)
            .map_err(PartialFailure::new)?;
        let mut state = initial;
        let mut trajectory = OdeTrajectory::new();
        match self.advance(
            problem,
            options,
            &mut state,
            &mut trajectory,
            &mut workspace,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(OdeSolution {
                state,
                report,
                trajectory,
            }),
            Err(error) => Err(workspace.shell.into_failure(error)),
        }
    }
}

impl<S, T, M> OdeMethod<S, T> for M
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: Stepper<S, T>,
{
}

fn preflight<S, T, M, Sys>(
    method: &M,
    problem: &OdeProblem<'_, S, Sys, T>,
    options: &OdeOptions<S>,
    state: &OdeState<S, T>,
    workspace: &mut OdeWorkspace<S, M, T>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T> + Clone,
    Sys: OdeSystem<S, T> + ?Sized,
{
    method.validate(options, problem.system().dimension())?;
    let n = problem.system().dimension();
    if state.values().len() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(n),
            scioxide_core::Dimensions::one(state.values().len()),
        ));
    }
    let (start, end) = (state.time(), problem.end());
    if !RealScalar::is_finite(start) || !RealScalar::is_finite(end) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "integration times must be finite",
        ));
    }
    if start == end {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("problem"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "the state time and the problem end time must differ",
        ));
    }
    crate::safety::finite(state.values(), scioxide_core::NonFiniteOrigin::Input)?;
    if let Some(times) = options.requested_times() {
        validate_requested_times(times, start, end, scioxide_core::argument!("options"))?;
    }
    validate_problem_features(Stepper::<S, T>::info(method), options, problem)?;
    let events = problem.events().len();
    workspace.prepare(
        method,
        n,
        events,
        options.event_records(events)?,
        problem.has_stage_solver(),
    )
}
/// Checks that the method supports the events, mass matrix and stage solver of `problem`.
pub(crate) fn validate_problem_features<S, T, Sys>(
    info: MethodInfo,
    options: &OdeOptions<S>,
    problem: &OdeProblem<'_, S, Sys, T>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    Sys: OdeSystem<S, T> + ?Sized,
{
    let events = problem.events();
    if !(events.is_empty() || info.events && info.dense_output) {
        return Err(unsupported(
            info.name,
            "event localization needs event and dense-output support",
        ));
    }
    if matches!(
        options.event_options().search(),
        EventSearchPolicy::DerivativeAssisted(_)
    ) && events.iter().any(|event| !event.has_total_derivative())
    {
        return Err(unsupported(
            info.name,
            "derivative-assisted search needs every event's total derivative",
        ));
    }
    if problem.system().mass_matrix_revision().is_some() && !info.mass_matrices {
        return Err(unsupported(info.name, "a non-identity mass matrix"));
    }
    if problem.has_stage_solver() && !info.injectable {
        return Err(unsupported(info.name, "an injected stage solver"));
    }
    Ok(())
}

pub(crate) fn ode_termination(finish: Finish) -> Result<OdeTermination, SciOxideError> {
    match finish {
        Finish::ReachedEnd => Ok(OdeTermination::ReachedEnd),
        Finish::TerminalEvent => Ok(OdeTermination::TerminalEvent),
        Finish::Stop(stop) => Ok(OdeTermination::from_stop(stop)),
        Finish::HistoryCapacity => Err(crate::safety::invalid(
            scioxide_core::argument!("method"),
            "history capacity applies only to delay integration",
        )),
    }
}

pub(crate) fn report<S, W, T>(
    outcome: Outcome<S>,
    termination: OdeTermination,
    final_time: S,
    storage: &IvpStorage<S, W, T>,
    cx: &scioxide_core::SolveContext<'_, '_>,
    search: Option<EventSearchPolicy>,
) -> OdeSolveReport<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    OdeSolveReport {
        termination,
        final_time,
        last_step_error_estimate: storage.last_step_error_estimate,
        counts: cx.counts(),
        steps: storage.steps.counts(),
        trust: cx
            .accepted_state()
            .map_or(PartialResultTrust::DiagnosticOnly, |accepted| {
                accepted.trust()
            }),
        events: outcome.events,
        terminal_event: outcome.terminal_event,
        event_search: search.map(|policy| EventSearchEvidence {
            policy,
            evaluations: storage.events.evaluations,
        }),
        method_switches: outcome.method_switches,
        last_method_switch: outcome.last_method_switch,
    }
}

/// Copies the accepted point into reused or new progress storage.
pub(crate) fn progress<S, W, T>(
    recycled: Option<OdeProgress<S, T>>,
    time: S,
    values: &[T],
    storage: &IvpStorage<S, W, T>,
) -> Option<OdeProgress<S, T>>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let mut progress = recycled
        .filter(|progress| progress.state.len() == values.len())
        .or_else(|| OdeProgress::new(values.len()).ok())?;
    progress.time = time;
    progress.state.as_mut_slice().copy_from_slice(values);
    progress.steps = storage.steps.counts();
    progress.last_step_error_estimate = storage.last_step_error_estimate;
    progress.trust = storage.trust;
    Some(progress)
}
