//! Callback and nonlinear-provider validation and transactionality contracts.

#![allow(clippy::panic, clippy::unwrap_used)]

use std::cell::Cell;

use scioxide_core::{
    AttemptOutcome, ErrorTolerance, ExecutionContext, InvalidInputReason, NonFiniteOrigin,
    NormSelection, SciOxideError, SolveContext,
};
use scioxide_ode::{
    AdamsBashforth, AdamsMoulton, BackwardEuler, Bdf, ClassicalRk4, DormandPrince5,
    DormandPrince853, ExplicitMidpoint, ForwardEuler, GaussRk, Heun, ImplicitTrapezoid,
    NonlinearStageSolver, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState,
    OdeSystem, OdeWorkspace, RadauIia, Rkf45, Sdirk, Verner65,
};

#[derive(Clone, Copy)]
enum CallbackOutcome {
    Error,
    NonFinite,
    MalformedShape,
}

/// A physical trial constraint without a time-error estimator: a candidate more than
/// 0.125 away from the last accepted state is inadmissible.
struct RejectLarge {
    accepted: Cell<f64>,
}
impl OdeSystem<f64> for RejectLarge {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(1.);
        Ok(())
    }
    fn admits(
        &self,
        _: f64,
        state: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok((state[0] - self.accepted.get()).abs() <= 0.125)
    }
}

#[test]
fn explicit_rejection_retries_fixed_methods_without_publishing_trials() {
    let run = |start: f64, end: f64, minimum: Option<f64>| {
        let system = RejectLarge {
            accepted: Cell::new(start),
        };
        let options = OdeOptions::new()
            .with_initial_step(0.5)
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps);
        let options = match minimum {
            Some(minimum) => options.with_minimum_step(minimum),
            None => options,
        };
        let plan = (ForwardEuler::new(), options);
        let problem = OdeProblem::new(&system, end);
        let mut state = OdeState::from_slice(start, &[start]).unwrap();
        let mut sink = |t: f64, y: &[f64]| {
            assert!((t - y[0]).abs() < 1e-15);
            system.accepted.set(y[0]);
            Ok(())
        };
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut sink,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        (report, state)
    };
    for (start, end) in [(0., 1.), (1., 0.)] {
        let (report, state) = run(start, end, None);
        assert!(report.is_success());
        assert_eq!((state.time(), state.values()), (end, &[end][..]));
        assert_eq!(report.rejected_steps(), 2);
        assert_eq!(report.accepted_steps(), 8);
    }
    let (report, state) = run(0., 1., Some(0.25));
    assert_eq!(
        report.termination(),
        Some(scioxide_ode::OdeTermination::StepSizeLimit)
    );
    assert_eq!((state.time(), state.values()), (0., &[0.][..]));
    assert_eq!(report.rejected_steps(), 2);
    assert_eq!(report.accepted_steps(), 0);
}

struct StageFailure {
    fail_at: usize,
    calls: Cell<usize>,
    outcome: CallbackOutcome,
}

impl OdeSystem<f64> for StageFailure {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            return match self.outcome {
                CallbackOutcome::Error => Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("derivative"),
                    InvalidInputReason::Inconsistent,
                    "deliberate callback failure",
                )),
                CallbackOutcome::NonFinite => {
                    derivative.fill(f64::NAN);
                    Ok(())
                }
                CallbackOutcome::MalformedShape => {
                    Err(SciOxideError::dimension_mismatch(
                        scioxide_core::Dimensions::one(1),
                        scioxide_core::Dimensions::one(0),
                    ))?;
                    Ok(())
                }
            };
        }
        derivative
            .iter_mut()
            .zip(state)
            .for_each(|(output, &value)| *output = -value);
        Ok(())
    }
}

fn returned_error<T>(result: Result<T, SciOxideError>) -> SciOxideError {
    let Err(error) = result else {
        panic!("failure fixture unexpectedly succeeded");
    };
    error
}

fn assert_callback_error_kind(error: &SciOxideError, outcome: CallbackOutcome) {
    match outcome {
        CallbackOutcome::Error => assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        )),
        CallbackOutcome::NonFinite => assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        )),
        CallbackOutcome::MalformedShape => {
            assert!(matches!(
                (error).view(),
                scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
            ));
        }
    }
}

/// Advance `system` from `(0, state)` to `end`; returns the result and the caller state.
fn advance<M, Sys>(
    method: M,
    options: OdeOptions<f64>,
    problem: &OdeProblem<'_, f64, Sys>,
    state: &[f64],
    workspace: &mut OdeWorkspace<f64, M>,
) -> (
    Result<scioxide_ode::OdeSolveReport<f64>, SciOxideError>,
    OdeState<f64>,
)
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64>,
{
    let plan = (method, options);
    let mut state = OdeState::from_slice(0.0, state).unwrap();
    let result = plan.0.advance(
        problem,
        &plan.1,
        &mut state,
        &mut (),
        workspace,
        ExecutionContext::sequential(),
    );
    (result, state)
}

/// One step of 0.1 whose `fail_at`-th RHS evaluation fails.
///
/// Explicit callback errors and malformed shapes are returned unchanged. Adaptive
/// methods reject a nonfinite derivative at a trial point and retry a smaller step;
/// every other failure is returned with the caller state untouched.
fn assert_explicit_failure<M, Factory>(
    factory: Factory,
    rhs_calls: usize,
    adaptive: bool,
    outcome: CallbackOutcome,
) where
    M: OdeMethod<f64> + Clone,
    Factory: Fn() -> M,
{
    let options = OdeOptions::new().with_initial_step(0.1);
    // Adaptive methods are the dense-capable ones here; request their extension.
    let options = if adaptive {
        options.with_output_schedule(OdeOutputSchedule::times(vec![0.05]))
    } else {
        options
    };
    for fail_at in 1..=rhs_calls {
        let original = [1.0, -2.0];

        let system = StageFailure {
            fail_at,
            calls: Cell::new(0),
            outcome,
        };
        let mut workspace = OdeWorkspace::new();
        let (result, state) = advance(
            factory(),
            options.clone(),
            &OdeProblem::new(&system, 0.1),
            &original,
            &mut workspace,
        );
        if adaptive && fail_at > 1 && matches!(outcome, CallbackOutcome::NonFinite) {
            let report = result.unwrap();
            assert!(report.is_success());
            assert_eq!(report.rejected_steps(), 1);
            assert!(system.calls.get() > fail_at);
            assert_eq!(state.time(), 0.1);
            continue;
        }
        assert_callback_error_kind(&returned_error(result), outcome);
        assert_eq!(system.calls.get(), fail_at);
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), original);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        assert_eq!(attempt.progress().unwrap().state(), original);
    }
}

fn assert_all_explicit_failures(outcome: CallbackOutcome) {
    assert_explicit_failure(ForwardEuler::<f64>::new, 1, false, outcome);
    assert_explicit_failure(ExplicitMidpoint::<f64>::new, 2, false, outcome);
    assert_explicit_failure(Heun::<f64>::new, 2, false, outcome);
    assert_explicit_failure(ClassicalRk4::<f64>::new, 4, false, outcome);
    assert_explicit_failure(Rkf45::<f64>::new, 7, true, outcome);
    assert_explicit_failure(DormandPrince5::<f64>::new, 7, true, outcome);
    assert_explicit_failure(DormandPrince853::<f64>::new, 16, true, outcome);
    assert_explicit_failure(Verner65::<f64>::new, 9, true, outcome);
}

#[test]
fn callback_errors_preserve_explicit_state() {
    assert_all_explicit_failures(CallbackOutcome::Error);
}

#[test]
fn non_finite_callback_outputs_are_typed_and_transactional() {
    assert_all_explicit_failures(CallbackOutcome::NonFinite);
}

#[test]
fn malformed_callback_shapes_are_typed_and_transactional() {
    assert_all_explicit_failures(CallbackOutcome::MalformedShape);
}

struct LinearDecay;

impl OdeSystem<f64> for LinearDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = jacobian.as_contiguous_slice_mut().ok_or_else(|| {
            scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("jacobian"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "row-major output required",
            )
        })?;
        jacobian_data[0] = -1.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

/// [`LinearDecay`] whose derivative fails at every time after `accepted`.
struct FailAfter {
    accepted: f64,
}

impl OdeSystem<f64> for FailAfter {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64],
        derivative: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if time > self.accepted {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("derivative"),
                InvalidInputReason::Inconsistent,
                "deliberate callback failure",
            ));
        }
        LinearDecay.rhs(time, state, derivative, cx)
    }

    fn jacobian(
        &self,
        time: f64,
        state: &[f64],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        LinearDecay.jacobian(time, state, jacobian, cx)
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

/// Startup steps of exactly 1/128 under a loose tolerance, then a failing callback.
///
/// Explicit Adams methods first evaluate the derivative at a new point in the next
/// step, so the failure surfaces one accepted step later for them.
fn assert_multistep_history_transaction<M, Factory>(factory: Factory, startup_steps: usize)
where
    M: OdeMethod<f64> + Clone,
    Factory: Fn() -> M,
{
    let step = 1.0 / 128.0;
    let failing = FailAfter {
        accepted: startup_steps as f64 * step,
    };
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(1.0, 1.0, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(step)
        .with_maximum_step(step);
    let mut workspace = OdeWorkspace::new();
    let (result, state) = advance(
        factory(),
        options.clone(),
        &OdeProblem::new(&failing, 1.0),
        &[1.0],
        &mut workspace,
    );
    assert!(matches!(
        returned_error(result).view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let accepted = workspace
        .last_attempt()
        .unwrap()
        .progress()
        .unwrap()
        .accepted_steps();
    assert!((startup_steps..=startup_steps + 1).contains(&accepted));
    assert_eq!(state.time(), accepted as f64 * step);

    // The failed attempt left exactly the state of an uninterrupted integration.
    let (control, control_state) = advance(
        factory(),
        options,
        &OdeProblem::new(&LinearDecay, state.time()),
        &[1.0],
        &mut OdeWorkspace::new(),
    );
    assert_eq!(control.unwrap().accepted_steps(), accepted);
    assert_eq!(state, control_state);
    let progress = workspace.last_attempt().unwrap().progress().unwrap();
    assert_eq!(progress.time(), control_state.time());
    assert_eq!(progress.state(), control_state.values());
}

#[test]
fn multistep_callback_failures_preserve_state_and_accepted_history() {
    assert_multistep_history_transaction(|| AdamsBashforth::<f64>::new(4).unwrap(), 4);
    assert_multistep_history_transaction(|| AdamsMoulton::<f64>::new(4).unwrap(), 4);
    assert_multistep_history_transaction(|| Bdf::<f64>::new(4).unwrap(), 4);
}

#[derive(Clone, Copy)]
enum NonlinearOutcome {
    Error,
    NonFinite,
    MalformedShape,
}

struct InvalidNonlinearProvider {
    outcome: NonlinearOutcome,
}

impl NonlinearStageSolver<f64> for InvalidNonlinearProvider {
    fn solve_in_place(
        &self,
        _residual_fn: &mut dyn FnMut(
            &[f64],
            &mut [f64],
            &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError>,
        _operator: &dyn scioxide_core::LinearOperator<f64>,
        state: &mut [f64],
        _residual: &mut [f64],
        _correction: &mut [f64],
        _accuracy: scioxide_ode::StageAccuracy<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<scioxide_ode::NonlinearStageReport, SciOxideError> {
        match self.outcome {
            NonlinearOutcome::Error => Err(SciOxideError::invalid_input(
                scioxide_core::argument!("state"),
                InvalidInputReason::Inconsistent,
                "deliberate nonlinear-provider failure",
            )),
            NonlinearOutcome::NonFinite => {
                state.fill(f64::NAN);
                Ok(scioxide_ode::NonlinearStageReport::new(1, 0, 0))
            }
            NonlinearOutcome::MalformedShape => Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(1),
                scioxide_core::Dimensions::one(0),
            )),
        }
    }
}

fn assert_nonlinear_error_kind(error: &SciOxideError, outcome: NonlinearOutcome) {
    match outcome {
        NonlinearOutcome::Error => assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        )),
        NonlinearOutcome::NonFinite => assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        )),
        NonlinearOutcome::MalformedShape => {
            assert!(matches!(
                (error).view(),
                scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
            ));
        }
    }
}

fn assert_nonlinear_transaction<M: OdeMethod<f64> + Clone>(method: M) {
    for outcome in [
        NonlinearOutcome::Error,
        NonlinearOutcome::MalformedShape,
        NonlinearOutcome::NonFinite,
    ] {
        let original = [1.0];
        let provider = InvalidNonlinearProvider { outcome };
        let mut workspace = OdeWorkspace::new();
        let (result, state) = advance(
            method.clone(),
            OdeOptions::new().with_initial_step(0.1),
            &OdeProblem::new(&LinearDecay, 0.1)
                .with_stage_solver(scioxide_ode::OdeStageSolver::Nonlinear(&provider)),
            &original,
            &mut workspace,
        );
        assert_nonlinear_error_kind(&returned_error(result), outcome);
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), original);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
}

#[test]
fn nonlinear_provider_failures_are_validated_before_state_commit() {
    assert_nonlinear_transaction(BackwardEuler::<f64>::new());
    assert_nonlinear_transaction(ImplicitTrapezoid::<f64>::new());
    assert_nonlinear_transaction(GaussRk::<f64>::new());
    assert_nonlinear_transaction(RadauIia::<f64>::new());
    assert_nonlinear_transaction(Sdirk::<f64>::new());
}

#[test]
fn termination_display_names_variant_and_kind() {
    use scioxide_core::{BudgetKind, ExecutionStopReason};
    use scioxide_ode::{
        AdjointTermination, DaeTermination, DdeTermination, MechanicalTermination, OdeTermination,
        SdeTermination,
    };
    assert_eq!(OdeTermination::ReachedEnd.to_string(), "ReachedEnd");
    assert_eq!(
        DdeTermination::BudgetLimit(BudgetKind::Evaluation).to_string(),
        "BudgetLimit(Evaluation)"
    );
    assert_eq!(
        DdeTermination::HistoryCapacity.to_string(),
        "HistoryCapacity"
    );
    assert_eq!(SdeTermination::StepSizeLimit.to_string(), "StepSizeLimit");
    assert_eq!(DaeTermination::RejectionLimit.to_string(), "RejectionLimit");
    assert_eq!(
        MechanicalTermination::ProjectionLimit.to_string(),
        "ProjectionLimit"
    );
    assert_eq!(
        AdjointTermination::ExecutionStopped(ExecutionStopReason::Cancelled).to_string(),
        format!("ExecutionStopped({})", ExecutionStopReason::Cancelled)
    );
}
