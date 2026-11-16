//! Adaptive Radau estimator, rejection history, and provider-failure contracts.
#![allow(clippy::unwrap_used)]

use core::cell::Cell;
use scioxide_core::{
    Budget, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeSolveReport, OdeState, OdeStepControl,
    OdeSystem, OdeTermination, OdeWorkspace, ProportionalIntegralController, RadauIia,
};

struct IdentityCorrections;
impl scioxide_ode::StageLinearSolver<f64> for IdentityCorrections {
    fn solve_into(
        &self,
        _: &dyn scioxide_core::LinearOperator<f64>,
        _: &[f64],
        rhs: &[f64],
        out: &mut [f64],
        _: scioxide_ode::StageAccuracy<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(rhs);
        Ok(())
    }
}

struct ConstantStages;
impl scioxide_ode::NonlinearStageSolver<f64> for ConstantStages {
    fn solve_in_place(
        &self,
        _: &mut dyn FnMut(
            &[f64],
            &mut [f64],
            &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError>,
        _: &dyn scioxide_core::LinearOperator<f64>,
        state: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: scioxide_ode::StageAccuracy<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<scioxide_ode::NonlinearStageReport, SciOxideError> {
        let root = 6.0_f64.sqrt();
        state.copy_from_slice(&[(4.0 - root) * 0.01, (4.0 + root) * 0.01, 0.1]);
        Ok(scioxide_ode::NonlinearStageReport::new(1, 0, 0))
    }
}

fn scalar(absolute: f64, relative: f64) -> ErrorTolerance<f64> {
    ErrorTolerance::scalar(absolute, relative, NormSelection::RootMeanSquare).unwrap()
}

/// Raw one-component error of the last attempt, recovered from the normalized
/// estimate `|e| / (atol + rtol * max(|y0|, |y1|))`.
fn raw_error(report: &OdeSolveReport<f64>, threshold: f64) -> f64 {
    report.last_step_error_estimate().unwrap() * threshold
}

#[test]
fn injected_stage_solvers_keep_the_native_estimator_and_dense_output() {
    let system = scioxide_ode::OdeFn::new(1, |_: f64, _: &[f64], out: &mut [f64]| {
        out[0] = 1.0;
        Ok(())
    })
    .unwrap();
    let options = OdeOptions::new()
        .with_error_tolerance(scalar(1e-6, 1e-3))
        .with_initial_step(0.1)
        .with_output_schedule(OdeOutputSchedule::times(vec![0.05]));
    let plan = (RadauIia::new(), options);
    for nonlinear in [false, true] {
        let problem = OdeProblem::new(&system, 0.1);
        let problem = if nonlinear {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Nonlinear(&ConstantStages))
        } else {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Linear(&IdentityCorrections))
        };
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
        let mut dense = Vec::new();
        let mut sink = |t: f64, y: &[f64]| {
            dense.push((t, y[0]));
            Ok(())
        };
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(report.counts().proposals(), 1);
        assert_eq!(
            report.counts().provider(EvaluationRole::Factorization),
            1,
            "only the estimator factor is native"
        );
        assert!(raw_error(&report, 1e-6 + 1e-3 * 0.1).abs() < 1e-15);
        assert_eq!(dense.len(), 1);
        assert_eq!(dense[0].0, 0.05);
        assert!((dense[0].1 - 0.05).abs() < 1e-15);
        assert!((state.values()[0] - 0.1).abs() < 1e-15);
    }
}

#[test]
fn caller_controller_overrides_radau_predictive_control() {
    // A unit-factor integral controller keeps the step fixed, which Radau's predictive
    // controller would grow on this smooth decay.
    let fixed = ProportionalIntegralController::new(1.0, 1.0, 1.0, 1.0, 0.0).unwrap();
    let system = Decay::new(1.0);
    let problem = OdeProblem::new(&system, 1.0);
    let mut steps = Vec::new();
    for control in [
        OdeStepControl::MethodDefault,
        OdeStepControl::ProportionalIntegral(fixed),
        OdeStepControl::Radau,
    ] {
        let plan = (
            RadauIia::new(),
            OdeOptions::new()
                .with_initial_step(0.125)
                .with_step_control(control),
        );
        let solution = plan
            .0
            .solve(
                &problem,
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![1.0])),
            )
            .unwrap();
        assert_eq!(
            solution.report().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        steps.push(solution.report().accepted_steps());
    }
    assert_eq!(steps[0], steps[2]);
    assert_eq!(steps[1], 8);
    assert!(steps[0] < steps[1], "{steps:?}");
}

struct Decay {
    mass: f64,
    active: Cell<bool>,
    anchor: Cell<f64>,
    anchor_calls: Cell<usize>,
    fail_refinement: Cell<bool>,
}

impl Decay {
    fn new(mass: f64) -> Self {
        Self {
            mass,
            active: Cell::new(true),
            anchor: Cell::new(0.0),
            anchor_calls: Cell::new(0),
            fail_refinement: Cell::new(false),
        }
    }
    fn start(&self, t: f64) {
        self.anchor.set(t);
        self.anchor_calls.set(0);
    }
}

impl OdeSystem<f64> for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if t == self.anchor.get() {
            self.anchor_calls.set(self.anchor_calls.get() + 1);
            if self.fail_refinement.get() && self.anchor_calls.get() == 2 {
                return Err(SciOxideError::singular(None));
            }
        }
        out[0] = if self.active.get() {
            -self.mass * y[0]
        } else {
            0.0
        };
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, -self.mass)?;
        Ok(Some(()))
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        if self.mass == 1.0 { None } else { Some(0) }
    }
    fn mass_matrix(
        &self,
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.set(0, 0, self.mass)
    }
}

#[test]
fn embedded_error_has_power_four_and_mass_scaling_preserves_it() {
    let mut identity = Vec::new();
    for mass in [1.0, 5.0] {
        let system = Decay::new(mass);
        let mut workspace = OdeWorkspace::new();
        let mut errors = Vec::new();
        for (i, h) in [0.2, 0.1, 0.05].into_iter().enumerate() {
            let plan = (
                RadauIia::new(),
                OdeOptions::new()
                    .with_error_tolerance(scalar(1e-3, 1e-3))
                    .with_initial_step(h),
            );
            let problem = OdeProblem::new(&system, h);
            let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.counts().proposals(), 1);
            // max(|y0|, |y1|) = 1, so the threshold is atol + rtol.
            let error = raw_error(&report, 2e-3);
            errors.push(error);
            assert!((state.values()[0] - (-h).exp()).abs() < 1e-8);
            if mass == 1.0 {
                identity.push(error);
            } else {
                assert!((error - identity[i]).abs() < 2e-15);
            }
        }
        for pair in errors.windows(2) {
            assert!((13.0..18.0).contains(&(pair[0] / pair[1])), "{errors:?}");
        }
    }
}

/// Integrate `system` from `(0, 1)` with steps of at most `0.4` and at most `attempts`
/// step attempts under a tight tolerance. When `continuation` is set, the first step
/// integrates a zero right-hand side (an exact, accepted step); the sink then activates
/// the decay and restarts the anchor counter at `t = 0.4`, so the next trial is an
/// ordinary continuation of accepted history.
fn tight_trials(
    system: &Decay,
    attempts: usize,
    continuation: bool,
    workspace: &mut OdeWorkspace<f64, RadauIia<f64>>,
) -> OdeSolveReport<f64> {
    let plan = (
        RadauIia::new(),
        OdeOptions::new()
            .with_error_tolerance(scalar(1e-12, 1e-12))
            .with_initial_step(0.4)
            .with_maximum_step(0.4)
            .with_max_steps(Budget::try_from(attempts).unwrap())
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    );
    system.active.set(!continuation);
    system.start(0.0);
    let problem = OdeProblem::new(system, 2.0);
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let mut sink = |t: f64, y: &[f64]| {
        if t == 0.4 {
            assert_eq!(y, &[1.0]);
            system.active.set(true);
            system.start(0.4);
        }
        Ok(())
    };
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut sink,
            workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(
            scioxide_core::BudgetKind::Proposal
        ))
    );
    assert_eq!(report.counts().proposals(), attempts);
    assert_eq!(
        report.accepted_steps(),
        usize::from(continuation),
        "only the exact zero-RHS step is accepted"
    );
    report
}

#[test]
fn first_and_rejected_trials_refine_and_new_epochs_reset_history() {
    let system = Decay::new(1.0);
    let mut workspace = OdeWorkspace::new();
    // max(|y0|, |y1|) = 1 for every trial from y = 1.
    let threshold = 2e-12;

    let first = tight_trials(&system, 1, false, &mut workspace);
    assert_eq!(
        system.anchor_calls.get(),
        2,
        "first trial must refine a large estimate"
    );
    let refined = raw_error(&first, threshold);

    let continued = tight_trials(&system, 2, true, &mut workspace);
    assert_eq!(
        system.anchor_calls.get(),
        1,
        "ordinary continuation uses one filter"
    );
    let unrefined = raw_error(&continued, threshold);

    let retried = tight_trials(&system, 3, true, &mut workspace);
    assert!(retried.last_step_error_estimate().unwrap() > 1.0);
    // The retry reuses f(tₙ, yₙ) of its base and evaluates only the refinement.
    assert_eq!(
        system.anchor_calls.get(),
        1 + 1,
        "a rejected trial must refine again"
    );

    let mu = 3.0 + 3.0_f64.cbrt().powi(2) - 3.0_f64.cbrt();
    assert!(first.last_step_error_estimate().unwrap() > 1.0);
    assert!(unrefined > refined);
    assert!((refined - unrefined / (1.0 + 0.4 / mu)).abs() < 2e-15);

    let fresh = tight_trials(&system, 1, false, &mut workspace);
    assert_eq!(system.anchor_calls.get(), 2);
    assert_eq!(raw_error(&fresh, threshold), refined);
}

#[test]
fn estimator_callback_failure_preserves_the_input_state() {
    let system = Decay::new(1.0);
    system.fail_refinement.set(true);
    let plan = (
        RadauIia::new(),
        OdeOptions::new()
            .with_error_tolerance(scalar(1e-12, 1e-12))
            .with_initial_step(0.4),
    );
    let problem = OdeProblem::new(&system, 0.4);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    assert!(
        plan.0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(state.time(), 0.0);
    assert_eq!(state.values(), &[1.0]);
}

#[test]
fn tighter_tolerance_controls_adaptive_steps_and_repeated_epochs() {
    let system = Decay::new(1.0);
    let problem = OdeProblem::new(&system, 3.0);
    let mut steps = Vec::new();
    for rtol in [1e-5, 1e-8] {
        let plan = (
            RadauIia::new(),
            OdeOptions::new()
                .with_initial_step(1.0)
                .with_error_tolerance(scalar(rtol * 1e-4, rtol)),
        );
        let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
        let mut first = None;
        for _ in 0..2 {
            let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
            assert!(report.rejected_steps() > 0);
            let error = (state.values()[0] - (-3.0_f64).exp()).abs();
            assert!(error < rtol * (-3.0_f64).exp());
            let result = (report.accepted_steps(), error);
            if let Some(first) = first {
                assert_eq!(result, first);
            } else {
                first = Some(result);
                steps.push(result.0);
            }
        }
    }
    assert!(steps[1] > steps[0]);
}
