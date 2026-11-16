//! Stiff/nonstiff transitions retain accuracy and explicit method diagnostics.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, ExecutionContext, NormSelection, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;
struct Transition;
impl OdeSystem<f64> for Transition {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = -rate(t) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        t: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -rate(t))?;
        Ok(Some(()))
    }
}
fn rate(t: f64) -> f64 {
    if t < 0.1 { 100. } else { 1. }
}

/// Every method switch of one integration. Reports carry only the most recent switch, so
/// the history is read from step-budgeted prefixes of the same deterministic epoch: each
/// attempt reports at most one switch.
fn switch_history<M, Sys>(
    method: &M,
    options: &OdeOptions<f64>,
    system: &Sys,
    end: f64,
    workspace: &mut OdeWorkspace<f64, M>,
) -> Vec<OdeMethodSwitch<f64>>
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64>,
{
    let problem = OdeProblem::new(system, end);
    let mut switches = Vec::new();
    for steps in 1.. {
        let plan = (
            method.clone(),
            options
                .clone()
                .with_max_steps(Budget::try_from(steps).unwrap()),
        );
        let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.method_switches() <= switches.len() + 1);
        if report.method_switches() > switches.len() {
            switches.push(report.last_method_switch().unwrap());
        }
        if report.termination() != Some(OdeTermination::BudgetLimit(BudgetKind::Proposal)) {
            break;
        }
    }
    switches
}

#[test]
fn adams_bdf_transition_and_epoch_reset() {
    let options = OdeOptions::new()
        .with_initial_step(0.02)
        .with_maximum_step(0.02)
        .with_error_tolerance(ErrorTolerance::scalar(1e-8, 1e-5, NormSelection::Infinity).unwrap());
    let method = AdamsBdf::<f64>::adams_bdf().unwrap();
    let plan = (method.clone(), options.clone());
    let problem = OdeProblem::new(&Transition, 0.5);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    for _ in 0..2 {
        let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.is_success(), "{:?}", report.termination());
        assert!((state.values()[0] - (-10.4_f64).exp()).abs() < 2e-6);
        assert!(report.method_switches() > 0);
        let switches = switch_history(&method, &options, &Transition, 0.5, &mut workspace);
        assert_eq!(switches.len(), report.method_switches());
        assert_eq!(switches.last().copied(), report.last_method_switch());
        assert!(switches.iter().any(|s| s.to() == "Bdf"));
        assert!(switches.iter().any(|s| s.from() == "Bdf"));
    }
}
#[test]
fn invalid_hysteresis_rejected() {
    assert!(
        AutoStiff::new(
            DormandPrince5::<f64>::new(),
            Bdf::new(2).unwrap(),
            StiffnessSwitchOptions::new()
                .with_thresholds(0.5_f64, 0.5)
                .with_evidence_steps(1, 1)
                .with_residence(1)
                .with_rejections(1)
        )
        .is_err()
    );
}

struct ComplexStiff;
impl OdeSystem<f64, scioxide_core::Complex<f64>> for ComplexStiff {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[scioxide_core::Complex<f64>],
        output: &mut [scioxide_core::Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(scioxide_core::Complex::new(0., 0.));
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[scioxide_core::Complex<f64>],
        out: &mut [scioxide_core::Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = scioxide_core::Complex::new(-1000., 20.) * y[0];
        Ok(())
    }
    fn jacobian_product(
        &self,
        t: f64,
        _: &[scioxide_core::Complex<f64>],
        v: &[scioxide_core::Complex<f64>],
        out: &mut [scioxide_core::Complex<f64>],
        _: &mut [scioxide_core::Complex<f64>],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, v, out, cx).map(Some)
    }
}
#[test]
fn native_complex_switching_preserves_accuracy() {
    use scioxide_core::Complex;
    let stiff = RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
        .unwrap()
        .with_backend(OdeStageBackend::Krylov {
            iterations: Budget::try_from(20).unwrap(),
            restart: Budget::try_from(4).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        });
    let method = AutoStiff::new(
        DormandPrince5::<f64, Complex<f64>>::new(),
        stiff,
        StiffnessSwitchOptions::default(),
    )
    .unwrap();
    let plan = (
        method,
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-8, 1e-5, NormSelection::Infinity).unwrap(),
            ),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&ComplexStiff, 0.01),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![Complex::new(1., 0.)])),
        )
        .unwrap();
    assert!(solution.report().is_success());
    assert!(solution.report().method_switches() > 0);
    let exact = Complex::new(
        (-10.0_f64).exp() * 0.2_f64.cos(),
        (-10.0_f64).exp() * 0.2_f64.sin(),
    );
    assert!((solution.state().values()[0] - exact).norm() < 1e-6);
}

struct SlowManifold;
impl OdeSystem<f64> for SlowManifold {
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
        let lambda = if (0.2..0.5).contains(&t) { 1000. } else { 0.1 };
        out[0] = lambda * (t.cos() - y[0]) - t.sin();
        Ok(())
    }
    fn jacobian(
        &self,
        t: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(
            0,
            0,
            if (0.2..0.5).contains(&t) {
                -1000.
            } else {
                -0.1
            },
        )?;
        Ok(Some(()))
    }
}
#[test]
fn unexcited_fast_mode_switches_in_both_directions() {
    let method = AdamsBdf::adams_bdf().unwrap();
    let options = OdeOptions::new()
        .with_initial_step(0.01)
        .with_maximum_step(0.02)
        .with_error_tolerance(
            ErrorTolerance::scalar(1e-9, 1e-6, NormSelection::RootMeanSquare).unwrap(),
        );
    let plan = (method.clone(), options.clone());
    let problem = OdeProblem::new(&SlowManifold, 1.);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(
        (state.values()[0] - 1_f64.cos()).abs() < 3e-6,
        "error={}",
        (state.values()[0] - 1_f64.cos()).abs()
    );
    let switches = switch_history(&method, &options, &SlowManifold, 1., &mut workspace);
    assert_eq!(switches.len(), report.method_switches());
    assert!(
        switches
            .iter()
            .any(|s| s.to() == "Bdf" && s.time() >= 0.18 && s.time() < 0.5),
        "{switches:?}"
    );
    assert!(
        switches
            .iter()
            .any(|s| s.from() == "Bdf" && s.time() >= 0.5)
    );
}
struct OverflowTrial {
    overflows: core::cell::Cell<usize>,
}
impl OdeSystem<f64> for OverflowTrial {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if y[0].abs() > 2. {
            self.overflows.set(self.overflows.get() + 1);
            out[0] = f64::INFINITY;
        } else {
            out[0] = -1000. * y[0];
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -1000.)?;
        Ok(Some(()))
    }
}
#[test]
fn overflowing_explicit_trial_is_rejected_and_recovers() {
    let problem = OverflowTrial {
        overflows: core::cell::Cell::new(0),
    };
    let plan = (
        AdamsBdf::adams_bdf().unwrap(),
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-10, 1e-6, NormSelection::RootMeanSquare).unwrap(),
            ),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&problem, 0.01),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(problem.overflows.get() > 0);
    assert!(report.counts().rejected() > 0);
    assert!((solution.state().values()[0] - (-10_f64).exp()).abs() < 1e-8);
}
struct Gentle;
impl OdeSystem<f64> for Gentle {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -0.1 * y[0];
        Ok(())
    }
}
#[test]
fn difference_probes_are_spaced_and_share_the_base_derivative() {
    use scioxide_core::EvaluationRole;
    let plan = (
        AdamsBdf::adams_bdf().unwrap(),
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_maximum_step(0.01)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-6, 1e-4, NormSelection::RootMeanSquare).unwrap(),
            ),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&Gentle, 0.2),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(report.counts().rejected(), 0);
    assert_eq!(report.method_switches(), 0);
    // Two PECE evaluations per step and the initial derivative, which the first
    // difference product evaluates and hands to the inner method. Far from the
    // switching threshold, one perturbed evaluation follows at bases 0, 2, 6 and
    // 14 of the 20 steps: the spacing doubles after each estimate.
    let accepted = report.counts().accepted();
    assert_eq!(accepted, 20);
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        2 * accepted + 1 + 4
    );
}

struct Mass;
impl OdeSystem<f64> for Mass {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -2. * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -2.)?;
        Ok(Some(()))
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(1)
    }
    fn mass_matrix(
        &self,
        m: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        m.set(0, 0, 2.)
    }
}
#[test]
fn mass_matrix_has_a_distinct_switch_reason() {
    let plan = (
        AdamsBdf::adams_bdf().unwrap(),
        OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(1e-10, 1e-7, NormSelection::RootMeanSquare).unwrap(),
        ),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&Mass, 0.1),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(report.method_switches(), 1);
    assert_eq!(
        report.last_method_switch().unwrap().reason(),
        OdeSwitchReason::MassMatrix
    );
    assert!(
        (solution.state().values()[0] - (-0.1_f64).exp()).abs() < 1e-5,
        "{:?}",
        solution.state()
    );
}
#[test]
fn hysteresis_errors_name_the_rejected_argument() {
    let cases = [
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(f64::NAN, 0.1)
                .with_evidence_steps(1, 1)
                .with_residence(1)
                .with_rejections(1),
            scioxide_core::argument!("enter"),
        ),
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(1., -0.1)
                .with_evidence_steps(1, 1)
                .with_residence(1)
                .with_rejections(1),
            scioxide_core::argument!("leave"),
        ),
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(1., 0.1)
                .with_evidence_steps(0, 1)
                .with_residence(1)
                .with_rejections(1),
            scioxide_core::argument!("enter_steps"),
        ),
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(1., 0.1)
                .with_evidence_steps(1, 0)
                .with_residence(1)
                .with_rejections(1),
            scioxide_core::argument!("leave_steps"),
        ),
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(1., 0.1)
                .with_evidence_steps(1, 1)
                .with_residence(0)
                .with_rejections(1),
            scioxide_core::argument!("residence"),
        ),
        (
            StiffnessSwitchOptions::new()
                .with_thresholds(1., 0.1)
                .with_evidence_steps(1, 1)
                .with_residence(1)
                .with_rejections(0),
            scioxide_core::argument!("rejections"),
        ),
    ];
    for (result, expected) in cases {
        let error =
            AutoStiff::new(DormandPrince5::<f64>::new(), Bdf::new(2).unwrap(), result).unwrap_err();
        assert!(
            matches!(error.view(),scioxide_core::SciOxideErrorView::InvalidInput { argument, .. } if argument == expected)
        );
    }
}
