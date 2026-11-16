//! The progressive initial-value driver shared by ODE plans, DDE plans and adjoint sweeps.
//!
//! The driver owns step control, event detection, output and commitment. Each attempt
//! writes only the workspace candidate; an accepted step is committed by swapping it into
//! the caller's state after all fallible work for the step succeeded.

use scioxide_core::{
    ErrorTolerance, NumericScalar, PartialResultTrust, RealScalar, SciOxideError, SolveContext,
    checked_difference, checked_float_product,
};
use scioxide_linalg::Vector;

use super::workspace::IvpStorage;
use crate::dde::DelayRuntime;
use crate::result::StepResult;
use crate::safety::{adjacent_time_step, select_step, validate_finite_state};
use crate::step_control::{
    ManagedController, PiController, ProportionalIntegralController, RadauController,
};
use crate::stepper::{DefaultController, MethodInfo, StepRequest, StepTrace, Stepper};
use crate::termination::{Fault, Stop, Work};
use crate::{
    EventFunction, OdeEvent, OdeMethodSwitch, OdeOptions, OdeOutputSchedule, OdeOutputSink,
    OdeStepControl, OdeSystem,
};

/// Optional recording of accepted steps (adjoint checkpoints). Hooks share the epoch.
#[allow(clippy::too_many_arguments)]
pub(crate) trait Recorder<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    /// Whether this invocation starts an independent method epoch.
    fn restart(&self) -> bool {
        true
    }
    fn self_metered(&self) -> bool {
        false
    }
    /// Replace the usual state norm, for partitioned augmented systems.
    fn error_norm(
        &mut self,
        _initial: &Vector<T>,
        _candidate: &Vector<T>,
        _step: &StepResult<'_, S, T>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok(None)
    }
    fn boundary(
        &mut self,
        _time: S,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok(None)
    }
    /// Counts one step attempt against the budgets; the default counts one iteration
    /// and one proposal.
    fn admit(&mut self, cx: &mut SolveContext<'_, '_>) -> Result<(), Fault> {
        crate::termination::record(cx, Work::Step)
    }
    fn fixed_step(&self) -> bool {
        false
    }
    fn requires_dense(&self) -> bool {
        false
    }
    fn trial(
        &mut self,
        _time: S,
        _width: S,
        _initial: &Vector<T>,
        _candidate: &Vector<T>,
        _step: &StepResult<'_, S, T>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn accepted(
        &mut self,
        _start: S,
        _width: S,
        _end: S,
        _state: &Vector<T>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Recorder<S, T> for () {}

/// Why the driver stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    ReachedEnd,
    TerminalEvent,
    Stop(Stop),
    /// A delay driver has no room for more history segments.
    HistoryCapacity,
}

/// Diagnostics of one drive; state, time and the step ledger live in the storage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Outcome<S: RealScalar> {
    pub(crate) finish: Finish,
    pub(crate) events: usize,
    pub(crate) terminal_event: Option<usize>,
    pub(crate) method_switches: usize,
    pub(crate) last_method_switch: Option<OdeMethodSwitch<S>>,
}

/// Borrowed configuration of one drive.
pub(crate) struct Drive<'a, S, T, M, Sys: ?Sized>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    pub(crate) method: &'a M,
    pub(crate) info: MethodInfo,
    pub(crate) options: &'a OdeOptions<S>,
    pub(crate) tolerance: &'a ErrorTolerance<S>,
    pub(crate) system: &'a Sys,
    pub(crate) events: &'a [&'a dyn EventFunction<S, T>],
    pub(crate) delay: Option<&'a dyn DelayRuntime<S, T>>,
    pub(crate) provider: crate::stiff::stage::Provider<'a, S, T>,
    pub(crate) end: S,
}

/// Configuration independent of the numerical trial kernel.
pub(crate) struct TrialDrive<
    'a,
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
> {
    pub(crate) info: MethodInfo,
    pub(crate) options: &'a OdeOptions<S>,
    pub(crate) tolerance: &'a ErrorTolerance<S>,
    pub(crate) events: &'a [&'a dyn EventFunction<S, T>],
    pub(crate) delay: Option<&'a dyn DelayRuntime<S, T>>,
    pub(crate) provider: crate::stiff::stage::Provider<'a, S, T>,
    pub(crate) end: S,
}

/// One numerical proposal, separated from adaptive control and commitment.
/// Residual adjoints supply their full/half-step trial directly.
#[allow(clippy::too_many_arguments)]
pub(crate) trait TrialKernel<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    const ADAPTIVE: Option<bool> = None;
    const BORROWS_BASE_STATE: bool = false;
    const CHECKED_RESULT: bool = false;
    /// [`Self::initial_step`] leaves `f(t0, y0)` in the first probe vector of the storage.
    const INITIAL_STEP_DERIVATIVE: bool = false;
    type Workspace: Default;
    fn begin(&self, dimension: usize, workspace: &mut Self::Workspace)
    -> Result<(), SciOxideError>;
    fn initial_step(
        &self,
        storage: &mut IvpStorage<S, Self::Workspace, T>,
        time: S,
        values: &Vector<T>,
        direction: S,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError>;
    fn step_limit(
        &self,
        time: S,
        values: &[T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError>;
    fn rhs_with_step_limit(
        &self,
        _time: S,
        _values: &[T],
        _derivative: &mut [T],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<crate::OdeRhsStepLimit<S>, SciOxideError> {
        Ok(crate::OdeRhsStepLimit::Unavailable)
    }
    fn admits(
        &self,
        time: S,
        values: &[T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError>;
    fn step<'w>(
        &self,
        time: S,
        candidate: &mut Vector<T>,
        width: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError>;
    /// Complete the dense output an accepted proposal deferred; see
    /// `Stepper::complete_dense`.
    fn complete_dense<'w>(
        &self,
        _time: S,
        _width: S,
        _request: StepRequest<'_, S, T>,
        _workspace: &'w mut Self::Workspace,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
        Err(crate::safety::invalid(
            scioxide_core::argument!("method"),
            "the trial deferred no dense output",
        ))
    }
    fn accept(&self, _workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn discard(&self, _workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<S, T, M, Sys> TrialKernel<S, T> for Drive<'_, S, T, M, Sys>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: Stepper<S, T>,
    Sys: OdeSystem<S, T> + ?Sized,
{
    const ADAPTIVE: Option<bool> = M::ADAPTIVE;
    const BORROWS_BASE_STATE: bool = M::BORROWS_BASE_STATE;
    const CHECKED_RESULT: bool = M::CHECKED_RESULT;
    const INITIAL_STEP_DERIVATIVE: bool = true;
    type Workspace = M::Workspace;
    fn begin(
        &self,
        dimension: usize,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.method
            .begin(dimension, self.provider.is_provided(), workspace)
    }
    fn initial_step(
        &self,
        storage: &mut IvpStorage<S, Self::Workspace, T>,
        time: S,
        values: &Vector<T>,
        direction: S,
        request: StepRequest<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        estimate_initial_step(self, storage, time, values, direction, request, cx)
    }
    fn step_limit(
        &self,
        time: S,
        values: &[T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        self.system.step_limit(time, values, cx)
    }
    fn admits(
        &self,
        time: S,
        values: &[T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        self.system.admits(time, values, cx)
    }
    fn rhs_with_step_limit(
        &self,
        time: S,
        values: &[T],
        derivative: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<crate::OdeRhsStepLimit<S>, SciOxideError> {
        if self.method.accepts_initial_rhs() {
            self.system
                .rhs_with_step_limit(time, values, derivative, cx)
        } else {
            Ok(crate::OdeRhsStepLimit::Unavailable)
        }
    }
    fn step<'w>(
        &self,
        time: S,
        candidate: &mut Vector<T>,
        width: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError> {
        self.method
            .step(self.system, time, candidate, width, request, workspace, cx)
    }
    fn complete_dense<'w>(
        &self,
        time: S,
        width: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
        self.method
            .complete_dense(self.system, time, width, request, workspace, cx)
    }
    fn accept(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        self.method.accept(workspace)
    }
    fn discard(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        self.method.discard(workspace)
    }
}

/// Built-in controller selected by the options or the method.
pub(crate) fn controller<S: RealScalar + NumericScalar<Real = S>>(
    control: &OdeStepControl<S>,
    default: DefaultController,
) -> ManagedController<S> {
    match control {
        OdeStepControl::ProportionalIntegral(controller) => {
            ManagedController::ProportionalIntegral(PiController::new(controller.clone()))
        }
        OdeStepControl::Radau => ManagedController::Radau(RadauController::default()),
        OdeStepControl::MethodDefault => match default {
            DefaultController::Integral => ManagedController::DefaultIntegral(PiController::new(
                ProportionalIntegralController::integral_defaults(),
            )),
            DefaultController::ProportionalIntegral => ManagedController::ProportionalIntegral(
                PiController::new(ProportionalIntegralController::default()),
            ),
            DefaultController::Radau => ManagedController::Radau(RadauController::default()),
        },
    }
}

// Only internal trial failures are eligible. The trace separately records explicit
// provider errors, including errors with these same numerical kinds.
fn recoverable_trial_failure(error: &SciOxideError, nonfinite_rhs: bool) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit
                | scioxide_core::ConvergenceReason::Divergence
                | scioxide_core::ConvergenceReason::Stagnation
                | scioxide_core::ConvergenceReason::NestedSolver,
            ..
        } | scioxide_core::SciOxideErrorView::Singular { .. }
            | scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Intermediate,
                ..
            }
            | scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ) || (nonfinite_rhs
        && matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ))
}

/// A stop that ends integration with the accepted state, or `None` for a failure.
fn halt(error: &SciOxideError, trace: &StepTrace) -> Option<Stop> {
    match trace.denial.denied() {
        Some(report) => Some(Stop::Budget(report.kind())),
        None => crate::termination::stop(error),
    }
}

fn increment(value: usize, quantity: scioxide_core::Quantity) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_add(value, 1, quantity)
}

/// Emit the events recorded for the current step, with states from its `dense` output.
fn emit_events<S, T, K>(
    records: &[super::events::EventRecord<S>],
    buffer: &mut Vector<T>,
    dense: &crate::dense_output::DenseOutputStorage<S, T>,
    sink: &mut K,
) -> Result<usize, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    K: OdeOutputSink<S, T> + ?Sized,
{
    for record in records {
        dense.eval_into(record.time, buffer.as_mut_slice())?;
        crate::safety::finite(buffer.as_slice(), scioxide_core::NonFiniteOrigin::Callback)?;
        sink.event(OdeEvent {
            index: record.index,
            time: record.time,
            state: buffer.as_slice(),
            occurrence: record.occurrence,
            terminal: record.terminal,
            evidence: record.evidence,
        })?;
    }
    Ok(records.len())
}

/// Integrate from `(*time, values)` to `run.end`.
///
/// Requires a preflight-validated configuration and storage that fits the method. On
/// return or error, `(*time, values)` hold the last accepted point.
pub(crate) fn drive<S, T, M, Sys, K, R>(
    run: &Drive<'_, S, T, M, Sys>,
    storage: &mut IvpStorage<S, M::Workspace, T>,
    time: &mut S,
    values: &mut Vector<T>,
    sink: &mut K,
    recorder: &mut R,
    cx: &mut SolveContext<'_, '_>,
) -> Result<Outcome<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: Stepper<S, T>,
    Sys: OdeSystem<S, T> + ?Sized,
    K: OdeOutputSink<S, T> + ?Sized,
    R: Recorder<S, T>,
{
    drive_trials(
        &TrialDrive {
            info: run.info,
            options: run.options,
            tolerance: run.tolerance,
            events: run.events,
            delay: run.delay,
            provider: run.provider,
            end: run.end,
        },
        run,
        storage,
        time,
        values,
        sink,
        recorder,
        cx,
    )
}

/// Integrate numerical trials through the shared adaptive driver.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(crate) fn drive_trials<S, T, M, K, R>(
    run: &TrialDrive<'_, S, T>,
    kernel: &M,
    storage: &mut IvpStorage<S, M::Workspace, T>,
    time: &mut S,
    values: &mut Vector<T>,
    sink: &mut K,
    recorder: &mut R,
    cx: &mut SolveContext<'_, '_>,
) -> Result<Outcome<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: TrialKernel<S, T>,
    K: OdeOutputSink<S, T> + ?Sized,
    R: Recorder<S, T>,
{
    let zero = S::ZERO;
    let options = run.options;
    let info = run.info;
    let adaptive = M::ADAPTIVE.unwrap_or(info.adaptive);
    let t0 = *time;
    let t_end = run.end;
    let backward = t_end < t0;
    let direction = if backward { -S::ONE } else { S::ONE };
    let n = values.len();
    let own_trace = StepTrace::default();
    let trace = run.delay.map_or(&own_trace, |delay| delay.trace());
    let has_events = !run.events.is_empty();
    let requested = options.requested_times();
    let accepted_output = matches!(options.output_schedule(), OdeOutputSchedule::AcceptedSteps);
    let final_only = matches!(options.output_schedule(), OdeOutputSchedule::FinalOnly);
    let (include_initial, include_final) = options.output_endpoints();
    let mut outcome = Outcome {
        finish: Finish::ReachedEnd,
        events: 0,
        terminal_event: None,
        method_switches: 0,
        last_method_switch: None,
    };

    let restart = recorder.restart();
    if restart {
        storage.begin();
    }
    if let Some(delay) = run.delay {
        delay.initialize(values, cx)?;
    }
    if restart {
        kernel.begin(n, &mut storage.method)?;
    }
    let mut controller = controller(options.step_control(), info.controller);

    let mut last_output_time = None;
    if accepted_output {
        sink.emit(t0, values.as_slice())?;
        last_output_time = Some(t0);
    }
    let mut requested_index = 0;
    if let Some(times) = requested {
        while times.get(requested_index).is_some_and(|time| *time == t0) {
            sink.emit_requested(requested_index, t0, values.as_slice())?;
            last_output_time = Some(t0);
            requested_index += 1;
        }
        if include_initial && last_output_time != Some(t0) {
            sink.emit(t0, values.as_slice())?;
            last_output_time = Some(t0);
        }
    }

    if has_events {
        storage
            .events
            .init(run.events, t0, values.as_slice(), trace, cx)?;
    }

    let request = StepRequest::new(trace).with_error_tolerance(run.tolerance);
    let request =
        if has_events || requested.is_some() || run.delay.is_some() || recorder.requires_dense() {
            request.with_dense_output()
        } else {
            request
        };
    let request = StepRequest {
        self_metered: run.delay.is_some() || recorder.self_metered(),
        provider: run.provider,
        ..request
    };

    // The automatic initial-step heuristic evaluates f(t0, y0); the first step
    // reuses it as its base derivative.
    let mut initial_derivative = false;
    let initial_proposal = match options.initial_step() {
        Some(h) => NumericScalar::abs(h),
        None => match run.delay {
            Some(delay) => checked_difference(delay.limit(t0)?, t0)?,
            None => {
                let h = kernel.initial_step(storage, t0, values, direction, request, cx)?;
                initial_derivative = M::INITIAL_STEP_DERIVATIVE;
                h
            }
        },
    };
    if !RealScalar::is_finite(initial_proposal) || initial_proposal <= zero {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "initial step-size estimate is non-finite or zero",
        ));
    }

    let IvpStorage {
        method: workspace,
        candidate,
        sample,
        base_rhs,
        probe: [initial_rhs, ..],
        events,
        steps,
        last_step_error_estimate,
        trust,
        ..
    } = storage;
    let mut t = t0;
    let mut proposed_step = initial_proposal;
    let mut retrying_rejection = false;
    let mut forced = None;
    // These handoffs belong only to the current accepted state within this epoch.
    // Rejections preserve them; acceptance below invalidates them.
    let mut prepared_rhs = crate::OdeRhsStepLimit::Unavailable;

    loop {
        let reached_end = if backward { t <= t_end } else { t >= t_end };
        if reached_end {
            break;
        }
        let boundary = recorder.boundary(t, cx)?;
        if let Some(reason) = cx.stop_reason()? {
            forced = Some(Finish::Stop(Stop::Execution(reason)));
            break;
        }
        if let Some(delay) = run.delay
            && !delay.reserve_step(t)?
        {
            forced = Some(Finish::HistoryCapacity);
            break;
        }
        match recorder.admit(cx) {
            Ok(()) => {}
            Err(Fault::Denied(report)) => {
                forced = Some(Finish::Stop(Stop::Budget(report.kind())));
                break;
            }
            Err(Fault::Error(error)) => return Err(error),
        }
        // The attempt ledger excludes nested work.
        {
            let mut ledger = IvpLedger {
                steps,
                cx: &mut *cx,
            };
            ledger.attempt()?;
        }
        let t_old = t;
        let step_end = match run.delay {
            Some(delay) => delay.limit(t_old)?,
            None => t_end,
        };
        let step_end = boundary.map_or(step_end, |boundary| {
            if backward {
                boundary.max(step_end)
            } else {
                boundary.min(step_end)
            }
        });
        let preparation = if matches!(prepared_rhs, crate::OdeRhsStepLimit::Evaluated { .. }) {
            cx.record_cache_hit()?;
            Ok(prepared_rhs)
        } else if request.self_metered {
            Ok(crate::OdeRhsStepLimit::Unavailable)
        } else {
            base_rhs.as_mut_slice().fill(T::from_real(S::NAN));
            kernel.rhs_with_step_limit(t_old, values.as_slice(), base_rhs.as_mut_slice(), cx)
        };
        let limit = preparation.and_then(|prepared| match prepared {
            crate::OdeRhsStepLimit::Unavailable => kernel.step_limit(t_old, values.as_slice(), cx),
            crate::OdeRhsStepLimit::Evaluated { maximum_step } => {
                validate_finite_state(base_rhs, n, scioxide_core::NonFiniteOrigin::Callback)?;
                prepared_rhs = prepared;
                Ok(maximum_step)
            }
        });
        let system_limit = match limit {
            Ok(limit) => limit,
            Err(error) => match crate::termination::stop(&error) {
                Some(stop) => {
                    forced = Some(Finish::Stop(stop));
                    break;
                }
                None => return Err(error),
            },
        };
        if system_limit.is_some_and(|limit| !limit.is_finite() || limit <= zero) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("system"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "system step limits must be positive and finite",
            ));
        }
        let maximum_step = match (options.maximum_step(), system_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let Some((dt, next_time)) = select_step(
            t_old,
            step_end,
            direction,
            proposed_step,
            options.minimum_step(),
            maximum_step,
            scioxide_core::argument!("options"),
        )?
        else {
            forced = Some(Finish::Stop(Stop::StepSize));
            break;
        };
        // Keep a fixed method's nominal coefficients when addition reaches the
        // same checked endpoint. Subtracting rounded times otherwise perturbs h
        // on ordinary fixed grids, needlessly changing exact stage factors.
        let nominal = direction * proposed_step;
        let dt = if !adaptive
            && maximum_step.is_none_or(|maximum| proposed_step <= maximum)
            && options
                .minimum_step()
                .is_none_or(|minimum| proposed_step >= minimum)
            && t_old + nominal == next_time
        {
            nominal
        } else {
            dt
        };
        if !M::BORROWS_BASE_STATE {
            candidate.as_mut_slice().copy_from_slice(values.as_slice());
        }
        trace.reset();
        let attempt = StepRequest {
            accepted_step: Some(steps.accepted),
            base: Some((t_old, values)),
            initial_rhs: if matches!(prepared_rhs, crate::OdeRhsStepLimit::Evaluated { .. }) {
                Some(base_rhs.as_slice())
            } else {
                initial_derivative.then(|| initial_rhs.as_slice())
            },
            ..request
        };
        let attempted = kernel.step(t_old, candidate, dt, attempt, workspace, cx);
        let step = match attempted {
            Ok(step) => step,
            Err(error) => {
                if let Some(stop @ (Stop::Budget(_) | Stop::Execution(_))) = halt(&error, trace) {
                    forced = Some(Finish::Stop(stop));
                    break;
                }
                // Inadmissible trial states retry for every method; other internal trial
                // failures retry for adaptive methods only.
                if trace.inadmissible.get()
                    || (adaptive
                        && !trace.provider_failed.get()
                        && recoverable_trial_failure(&error, trace.trial_rhs.get()))
                {
                    kernel.discard(workspace)?;
                    IvpLedger {
                        steps,
                        cx: &mut *cx,
                    }
                    .reject()?;
                    let Some(smaller) = halved(
                        dt,
                        t_old,
                        step_end,
                        next_time,
                        direction,
                        options.minimum_step(),
                        options.maximum_step(),
                    )?
                    else {
                        forced = Some(Finish::Stop(Stop::StepSize));
                        break;
                    };
                    retrying_rejection = true;
                    proposed_step = smaller;
                    continue;
                }
                return match halt(&error, trace) {
                    Some(stop) => {
                        forced = Some(Finish::Stop(stop));
                        break;
                    }
                    None => Err(error),
                };
            }
        };
        if let Some(change) = step.method_switch {
            controller.reset();
            outcome.method_switches = increment(
                outcome.method_switches,
                scioxide_core::quantity!("ode_method_switches"),
            )?;
            outcome.last_method_switch = Some(change);
        }
        if !M::CHECKED_RESULT {
            step.validate_for_step(request, t_old, next_time, n)?;
            validate_finite_state(candidate, n, scioxide_core::NonFiniteOrigin::Callback)?;
        }
        let cache_hits = step
            .num_jacobian_cache_hits
            .checked_add(step.num_factor_cache_hits)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ode_cache_hit_count"))
            })?;
        (0..cache_hits).try_for_each(|_| cx.record_cache_hit())?;

        let override_error = recorder.error_norm(values, candidate, &step)?;
        let err = if let Some(error) = override_error {
            error
        } else if M::CHECKED_RESULT
            && let Some(error) = step.normalized_error
        {
            error
        } else if let Some(errors) = step.raw_error {
            if n == 4 {
                let old = &values.as_slice()[..4];
                let new = &candidate.as_slice()[..4];
                let errors = &errors[..4];
                run.tolerance.normalized_norm_by(4, |i| {
                    let old = old[i].abs();
                    let new = new[i].abs();
                    (errors[i].abs(), if new > old { new } else { old })
                })?
            } else {
                run.tolerance.normalized_norm_by(n, |i| {
                    let old = values.as_slice()[i].abs();
                    let new = candidate.as_slice()[i].abs();
                    (errors[i].abs(), if new > old { new } else { old })
                })?
            }
        } else if adaptive {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("method"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "adaptive methods must supply component errors",
            ));
        } else {
            zero
        };
        let extra = recorder.trial(t_old, dt, values, candidate, &step, cx)?;
        let err = err.max(extra);
        *last_step_error_estimate = step.raw_error.map(|_| err);

        // Fixed-step methods accept their checked result unchanged. Adaptive methods
        // route every estimate, including exact zero, through the controller.
        let (controller_step, accept) = if recorder.fixed_step() {
            (initial_proposal * direction, true)
        } else if !adaptive {
            // Fixed-step methods keep proposing the step they attempted: admissibility
            // retries persist, while a step shortened by the span end, a system limit or
            // time rounding does not shrink later proposals.
            (proposed_step * direction, true)
        } else {
            controller.compute_step(
                err,
                dt,
                step.error_order.unwrap_or(info.order),
                step.num_nonlinear_iters,
            )?
        };
        let mut next_proposal = validate_controller_proposal(controller_step, dt)?;
        if let Some(growth) = step.maximum_step_growth {
            next_proposal = next_proposal.min(crate::step_control::bounded_scale_step(
                NumericScalar::abs(dt),
                growth,
            ));
        }
        if retrying_rejection || !accept {
            next_proposal = next_proposal.min(NumericScalar::abs(dt));
        }

        if !accept {
            retrying_rejection = true;
            kernel.discard(workspace)?;
            IvpLedger {
                steps,
                cx: &mut *cx,
            }
            .reject()?;
            let Some(retry) = retry_step(
                next_proposal,
                dt,
                t_old,
                step_end,
                next_time,
                direction,
                options.minimum_step(),
                options.maximum_step(),
            )?
            else {
                forced = Some(Finish::Stop(Stop::StepSize));
                break;
            };
            proposed_step = retry;
            continue;
        }

        // Endpoint event values decide whether the event search reads the step's
        // continuous extension.
        let search = if has_events {
            match events.endpoint(
                run.events,
                options.event_options(),
                next_time,
                candidate.as_slice(),
                trace,
                cx,
            ) {
                Ok(search) => search,
                Err(error) => match halt(&error, trace) {
                    Some(stop) => {
                        forced = Some(Finish::Stop(stop));
                        break;
                    }
                    None => return Err(error),
                },
            }
        } else {
            false
        };
        // A requested time at the step end takes the endpoint state.
        let interior_output = requested.is_some_and(|times| {
            times.get(requested_index).is_some_and(|&te| {
                te != next_time
                    && if backward {
                        te <= t_old && te >= next_time
                    } else {
                        te >= t_old && te <= next_time
                    }
            })
        });
        // Methods may defer dense work, such as extension stages, until acceptance.
        // It is completed only when this step's interpolant is read, and its
        // failures are those of the trial.
        let mut dense_output = step.dense_output;
        if step.dense_deferred && (search || interior_output || run.delay.is_some()) {
            dense_output = match kernel.complete_dense(t_old, dt, attempt, workspace, cx) {
                Ok(dense) => Some(dense),
                Err(error) => {
                    if let Some(stop @ (Stop::Budget(_) | Stop::Execution(_))) = halt(&error, trace)
                    {
                        forced = Some(Finish::Stop(stop));
                        break;
                    }
                    if trace.inadmissible.get()
                        || (adaptive
                            && !trace.provider_failed.get()
                            && recoverable_trial_failure(&error, trace.trial_rhs.get()))
                    {
                        kernel.discard(workspace)?;
                        IvpLedger {
                            steps,
                            cx: &mut *cx,
                        }
                        .reject()?;
                        let Some(smaller) = halved(
                            dt,
                            t_old,
                            step_end,
                            next_time,
                            direction,
                            options.minimum_step(),
                            options.maximum_step(),
                        )?
                        else {
                            forced = Some(Finish::Stop(Stop::StepSize));
                            break;
                        };
                        retrying_rejection = true;
                        proposed_step = smaller;
                        continue;
                    }
                    return match halt(&error, trace) {
                        Some(stop) => {
                            forced = Some(Finish::Stop(stop));
                            break;
                        }
                        None => Err(error),
                    };
                }
            };
        }

        // Detect events before output so a terminal event can truncate this step.
        let terminal = if search {
            let dense = dense_output.ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("method"),
                    "event search requires dense output",
                )
            })?;
            if let Err(error) = events.detect(
                run.events,
                options.event_options(),
                t_old,
                next_time,
                dense,
                trace,
                cx,
            ) {
                match halt(&error, trace) {
                    Some(stop) => {
                        forced = Some(Finish::Stop(stop));
                        break;
                    }
                    None => return Err(error),
                }
            }
            let terminal = events.retain_terminal_group(options.event_options().localization())?;
            if let Some(record) = terminal {
                dense.eval_into(record.time, events.terminal.as_mut_slice())?;
                validate_finite_state(
                    &events.terminal,
                    n,
                    scioxide_core::NonFiniteOrigin::Callback,
                )?;
            }
            terminal
        } else {
            None
        };
        let (effective_t, effective_y) = match terminal {
            Some(record) => (record.time, &events.terminal),
            None => (next_time, &*candidate),
        };
        if !trace.provider(kernel.admits(effective_t, effective_y.as_slice(), cx))? {
            kernel.discard(workspace)?;
            IvpLedger {
                steps,
                cx: &mut *cx,
            }
            .reject()?;
            let Some(smaller) = halved(
                dt,
                t_old,
                step_end,
                next_time,
                direction,
                options.minimum_step(),
                options.maximum_step(),
            )?
            else {
                forced = Some(Finish::Stop(Stop::StepSize));
                break;
            };
            retrying_rejection = true;
            proposed_step = smaller;
            continue;
        }

        if terminal.is_none()
            && let Some(delay) = run.delay
        {
            let segment = dense_output.ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("method"),
                    "DDE history requires owned method coefficients",
                )
            })?;
            delay.stage(segment)?;
        }

        if let Some(times) = requested {
            while requested_index < times.len() {
                let te = times[requested_index];
                let in_step = if backward {
                    te <= t_old && te >= effective_t
                } else {
                    te >= t_old && te <= effective_t
                };
                if !in_step {
                    break;
                }
                if te == effective_t {
                    sample
                        .as_mut_slice()
                        .copy_from_slice(effective_y.as_slice());
                } else {
                    dense_output
                        .ok_or_else(|| {
                            crate::safety::invalid(
                                scioxide_core::argument!("method"),
                                "requested output times require dense output",
                            )
                        })?
                        .eval_into(te, sample.as_mut_slice())?;
                }
                validate_finite_state(sample, n, scioxide_core::NonFiniteOrigin::Callback)?;
                sink.emit_requested(requested_index, te, sample.as_slice())?;
                last_output_time = Some(te);
                requested_index = increment(
                    requested_index,
                    scioxide_core::quantity!("ode_evaluation_time_index"),
                )?;
            }
        } else if accepted_output && last_output_time != Some(effective_t) {
            sink.emit(effective_t, effective_y.as_slice())?;
            last_output_time = Some(effective_t);
        }
        if has_events && !events.records.is_empty() {
            let dense = dense_output.ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("method"),
                    "event output requires dense output",
                )
            })?;
            let emitted = emit_events(&events.records, &mut events.state, dense, sink)?;
            outcome.events = outcome.events.checked_add(emitted).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("ode_events"))
            })?;
        }

        recorder.accepted(t_old, dt, effective_t, effective_y, cx)?;
        if let Some(record) = terminal {
            kernel.discard(workspace)?;
            IvpLedger {
                steps,
                cx: &mut *cx,
            }
            .accept(trust)?;
            values
                .as_mut_slice()
                .copy_from_slice(events.terminal.as_slice());
            t = record.time;
            *time = t;
            outcome.finish = Finish::TerminalEvent;
            outcome.terminal_event = Some(record.index);
            break;
        }
        kernel.accept(workspace)?;
        IvpLedger {
            steps,
            cx: &mut *cx,
        }
        .accept(trust)?;
        core::mem::swap(values, candidate);
        t = next_time;
        prepared_rhs = crate::OdeRhsStepLimit::Unavailable;
        initial_derivative = false;
        *time = t;
        if has_events {
            events.commit();
        }
        if let Some(delay) = run.delay {
            delay.commit();
            if delay.is_break(t) {
                kernel.begin(n, workspace)?;
                controller.reset();
            }
        }
        retrying_rejection = false;
        proposed_step = next_proposal;
    }

    if let Some(finish) = forced {
        outcome.finish = finish;
    }
    if final_only || (requested.is_some() && include_final && last_output_time != Some(t)) {
        sink.emit(t, values.as_slice())?;
    }
    Ok(outcome)
}

/// The retry magnitude after a failed or inadmissible attempt `dt`: half of it, raised
/// to the minimum step (see [`retry_step`]).
#[allow(clippy::too_many_arguments)]
fn halved<S: RealScalar>(
    dt: S,
    t_old: S,
    step_end: S,
    next_time: S,
    direction: S,
    minimum: Option<S>,
    maximum: Option<S>,
) -> Result<Option<S>, SciOxideError> {
    retry_step(
        NumericScalar::abs(dt) / S::TWO,
        dt,
        t_old,
        step_end,
        next_time,
        direction,
        minimum,
        maximum,
    )
}

/// The retry magnitude after rejecting the attempt `dt` from `t_old`: `proposal` raised
/// to the minimum step, so a rejection above the minimum is retried once exactly at it.
/// `None` when that retry would not be shorter than the attempt or would not reach a new
/// representable endpoint.
#[allow(clippy::too_many_arguments)]
fn retry_step<S: RealScalar>(
    proposal: S,
    dt: S,
    t_old: S,
    step_end: S,
    next_time: S,
    direction: S,
    minimum: Option<S>,
    maximum: Option<S>,
) -> Result<Option<S>, SciOxideError> {
    let retry = minimum.map_or(proposal, |minimum| proposal.max(minimum));
    if retry <= S::ZERO || retry >= NumericScalar::abs(dt) {
        return Ok(None);
    }
    let selected = select_step(
        t_old,
        step_end,
        direction,
        retry,
        None,
        maximum,
        scioxide_core::argument!("options"),
    )?;
    Ok(selected
        .filter(|&(_, endpoint)| endpoint != next_time)
        .map(|_| retry))
}

/// The driver's attempt ledger, kept beside the epoch's own counts.
struct IvpLedger<'a, 'b, 'c, 'd> {
    steps: &'a mut super::workspace::StepCounts,
    cx: &'b mut SolveContext<'c, 'd>,
}

impl IvpLedger<'_, '_, '_, '_> {
    fn attempt(&mut self) -> Result<(), SciOxideError> {
        self.steps.attempts = increment(
            self.steps.attempts,
            scioxide_core::quantity!("ode_step_attempts"),
        )?;
        Ok(())
    }
    fn reject(&mut self) -> Result<(), SciOxideError> {
        self.cx.record_rejected()?;
        self.steps.rejected = increment(
            self.steps.rejected,
            scioxide_core::quantity!("ode_step_rejections"),
        )?;
        Ok(())
    }
    fn accept(&mut self, trust: &mut PartialResultTrust) -> Result<(), SciOxideError> {
        self.cx.commit_accepted(PartialResultTrust::Admissible)?;
        self.steps.accepted = increment(
            self.steps.accepted,
            scioxide_core::quantity!("ode_step_acceptances"),
        )?;
        *trust = PartialResultTrust::Admissible;
        Ok(())
    }
}

/// Estimate the initial step size (Hairer–Nørsett–Wanner), leaving `f(t0, y0)`
/// in the first probe vector for the first step.
pub(super) fn estimate_initial_step<S, T, M, Sys>(
    run: &Drive<'_, S, T, M, Sys>,
    storage: &mut IvpStorage<S, M::Workspace, T>,
    t0: S,
    y0: &Vector<T>,
    direction: S,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: Stepper<S, T>,
    Sys: OdeSystem<S, T> + ?Sized,
{
    let t_end = run.end;
    let tolerance = run.tolerance;
    let IvpStorage {
        probe: [f0, normalized, y1, f1],
        scales,
        magnitudes,
        ..
    } = storage;
    crate::traits::evaluate_rhs_staged(run.system, t0, y0, f0, normalized, request, cx)?;
    for (scale, state) in scales.as_mut_slice().iter_mut().zip(y0.as_slice()) {
        *scale = state.abs();
    }
    magnitudes.as_mut_slice().copy_from_slice(scales.as_slice());
    // A startup ratio at an exact-zero threshold has no information about the
    // next state's scale. Mask it only in this heuristic; acceptance still checks
    // every component against max(|y_old|, |y_new|).
    let startup_norm = |values: &mut [S]| -> Result<S, SciOxideError> {
        let dimension = values.len();
        for (i, value) in values.iter_mut().enumerate() {
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "initial-step magnitude must be finite",
                ));
            }
            if tolerance.threshold(i, dimension, scales.as_slice()[i])? == S::ZERO {
                *value = S::ZERO;
            }
        }
        tolerance.normalized_norm(values, scales.as_slice())
    };
    let d0 = startup_norm(magnitudes.as_mut_slice())?;
    for (out, derivative) in magnitudes.as_mut_slice().iter_mut().zip(f0.as_slice()) {
        *out = derivative.abs();
    }
    let d1 = startup_norm(magnitudes.as_mut_slice())?;

    let small = S::try_from_f64_lossy(1e-5)?;
    let tiny_step = S::try_from_f64_lossy(1e-6)?;
    let ratio = S::try_from_f64_lossy(0.01)?;
    let hundred = S::try_from_f64_lossy(100.0)?;
    let representable = adjacent_time_step(t0, t_end)?.abs();
    let h0 = if d0 < small || d1 < small {
        tiny_step.max(representable)
    } else {
        checked_float_product(ratio, d0 / d1)?.max(representable)
    };

    // Use the actual bounded increment for both the sample state and time.
    let Some((signed_h0, sample_time)) = select_step(
        t0,
        t_end,
        direction,
        h0,
        None,
        run.options.maximum_step(),
        scioxide_core::argument!("options"),
    )?
    else {
        // The driver reports StepSizeLimit with the accepted initial state.
        return Ok(h0);
    };
    let h0 = NumericScalar::abs(signed_h0);
    for ((next, &state), &derivative) in y1
        .as_mut_slice()
        .iter_mut()
        .zip(y0.as_slice())
        .zip(f0.as_slice())
    {
        *next = crate::safety::state_increment(state, signed_h0, derivative)?;
    }
    crate::traits::evaluate_rhs_staged(run.system, sample_time, y1, f1, normalized, request, cx)?;
    for ((out, &next), &initial) in magnitudes
        .as_mut_slice()
        .iter_mut()
        .zip(f1.as_slice())
        .zip(f0.as_slice())
    {
        *out = crate::safety::state_difference(next, initial)?.abs();
    }
    let d2 = startup_norm(magnitudes.as_mut_slice())? / h0;
    let order = S::try_from_usize_exact(run.info.initial_step_order)?;
    let max_d = d1.max(d2);
    let h1 = if max_d <= S::try_from_f64_lossy(1e-15)? {
        let reduced_h0 = checked_float_product(S::try_from_f64_lossy(1e-3)?, h0)?;
        tiny_step.max(reduced_h0)
    } else {
        (ratio / max_d).powf(S::ONE / order)
    };
    let growth_limit = if h0 > S::MAX / hundred {
        S::MAX
    } else {
        h0 * hundred
    };
    Ok(h1.min(growth_limit).max(representable))
}

fn validate_controller_proposal<S: RealScalar>(
    proposal: S,
    current: S,
) -> Result<S, SciOxideError> {
    if !proposal.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Callback,
            "step controller returned a non-finite step",
        ));
    }
    if proposal == S::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "step controller returned a zero step",
        ));
    }
    if (proposal > S::ZERO) != (current > S::ZERO) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "step controller reversed the integration direction",
        ));
    }
    Ok(NumericScalar::abs(proposal))
}

/// Requested times lie in the span and are monotone in the integration direction.
pub(crate) fn validate_requested_times<S: RealScalar>(
    times: &[S],
    t0: S,
    t_end: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    let backward = t_end < t0;
    for (idx, &t_i) in times.iter().enumerate() {
        if !t_i.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "requested times must be finite",
            ));
        }
        let in_span = if backward {
            t_i <= t0 && t_i >= t_end
        } else {
            t_i >= t0 && t_i <= t_end
        };
        if !in_span {
            return Err(SciOxideError::invalid_input(
                argument,
                scioxide_core::InvalidInputReason::OutOfRange,
                "requested output time is outside the integration span",
            ));
        }
        if idx > 0 {
            let prev = times[idx - 1];
            let monotone = if backward { t_i <= prev } else { t_i >= prev };
            if !monotone {
                return Err(SciOxideError::invalid_input(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "requested output times must be monotone in the integration direction",
                ));
            }
        }
    }
    Ok(())
}
