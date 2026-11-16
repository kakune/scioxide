//! Callback work and prepared-storage contracts of automatic switching.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;

use scioxide_core::{
    Budget, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError,
    SolveContext,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::*;

struct Decay {
    rate: f64,
    analytic: bool,
    product: bool,
    calls: Cell<usize>,
}
impl OdeSystem<f64> for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        state: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = -self.rate * state[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if self.analytic {
            j.set(0, 0, -self.rate)?;
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn jacobian_product(
        &self,
        _: f64,
        _: &[f64],
        v: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if self.product {
            assert_eq!(scratch.len(), 3);
            out[0] = -self.rate * v[0];
            Ok(Some(()))
        } else {
            Ok(None)
        }
    }
}

fn options(h: f64, tolerance: f64) -> OdeOptions<f64> {
    OdeOptions::new()
        .with_initial_step(h)
        .with_maximum_step(h)
        .with_error_tolerance(
            ErrorTolerance::scalar(tolerance, tolerance, NormSelection::RootMeanSquare).unwrap(),
        )
}

fn run<M: OdeMethod<f64> + Clone>(
    method: M,
    system: &Decay,
    options: OdeOptions<f64>,
    end: f64,
) -> OdeSolveReport<f64> {
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(system, end),
            &plan.1,
            OdeState::from_slice(0., &[1.]).unwrap(),
        )
        .unwrap();
    assert!(solution.report().is_success(), "{:?}", solution.report());
    *solution.report()
}

#[test]
fn finite_difference_stiff_linearization_replaces_switcher_probes() {
    let system = Decay {
        rate: 100.,
        analytic: false,
        product: false,
        calls: Cell::new(0),
    };
    for backend in [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: scioxide_core::StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
            limits: scioxide_sparse::SparseLuLimits::new(8, 1 << 20),
        },
    ] {
        let stiff = Bdf::with_order_policy(MultistepOrder::Automatic(5))
            .unwrap()
            .with_backend(backend);
        let method = AutoStiff::new(
            AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
            stiff.clone(),
            StiffnessSwitchOptions::new()
                .with_thresholds(0.001, 0.0001)
                .with_evidence_steps(1, 6)
                .with_residence(8)
                .with_rejections(3),
        )
        .unwrap();
        let auto = run(method, &system, options(0.001, 0.1), 0.02);
        let inner = run(stiff, &system, options(0.001, 0.1), 0.02);
        assert_eq!(auto.method_switches(), 1);
        assert_eq!(
            auto.last_method_switch().unwrap().reason(),
            OdeSwitchReason::Stiffness
        );
        assert_eq!(auto.counts().accepted(), inner.counts().accepted());
        // Only the first, perturbed evaluation of the initial difference product:
        // its base derivative becomes the stiff method's initial derivative, and
        // all later power products use the stiff method's finite-difference
        // Jacobian with no RHS work.
        assert_eq!(
            auto.counts().provider(EvaluationRole::RightHandSide),
            inner.counts().provider(EvaluationRole::RightHandSide) + 1
        );
    }
}

#[test]
fn a_new_epoch_discards_diagnostic_estimates() {
    let mut system = Decay {
        rate: 0.1,
        analytic: true,
        product: false,
        calls: Cell::new(0),
    };
    let plan = (AdamsBdf::adams_bdf().unwrap(), options(0.01, 1e-4));
    let mut workspace = plan
        .0
        .workspace(&OdeProblem::new(&system, 0.2), &plan.1)
        .unwrap();
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    let first = plan
        .0
        .advance(
            &OdeProblem::new(&system, 0.2),
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(first.is_success());
    system.rate = 100.;
    state.reset(0., &[1.]).unwrap();
    let reused = plan
        .0
        .advance(
            &OdeProblem::new(&system, 0.2),
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let fresh = plan
        .0
        .solve(
            &OdeProblem::new(&system, 0.2),
            &plan.1,
            OdeState::from_slice(0., &[1.]).unwrap(),
        )
        .unwrap();
    assert!(reused.is_success());
    assert_eq!(
        reused.counts().provider(EvaluationRole::RightHandSide),
        fresh
            .report()
            .counts()
            .provider(EvaluationRole::RightHandSide)
    );
    assert_eq!(
        reused.last_method_switch(),
        fresh.report().last_method_switch()
    );
    assert!((state.values()[0] - fresh.state().values()[0]).abs() < 1e-14);
}

#[test]
fn diagnostic_budget_exhaustion_keeps_the_accepted_state() {
    let system = Decay {
        rate: 0.1,
        analytic: true,
        product: false,
        calls: Cell::new(0),
    };
    let plan = (
        AdamsBdf::adams_bdf().unwrap(),
        options(0.01, 1e-4).with_max_evaluations(Budget::ONE),
    );
    let problem = OdeProblem::new(&system, 0.2);
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut plan.0.workspace(&problem, &plan.1).unwrap(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(
            scioxide_core::BudgetKind::Evaluation
        ))
    );
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), &[1.]);
    assert_eq!(system.calls.get(), 0);
}

#[test]
fn unavailable_products_are_requested_once_per_epoch() {
    let system = Decay {
        rate: 0.1,
        analytic: false,
        product: false,
        calls: Cell::new(0),
    };
    let plan = (AdamsBdf::adams_bdf().unwrap(), options(0.01, 1e-4));
    let problem = OdeProblem::new(&system, 0.2);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    for _ in 0..2 {
        state.reset(0., &[1.]).unwrap();
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
        assert!(report.is_success());
        assert!(report.counts().accepted() > 16);
        // The first sample asks for a product; later difference probes do not.
        assert_eq!(report.counts().provider(EvaluationRole::LinearOperator), 1);
        assert_eq!(report.counts().provider(EvaluationRole::Jacobian), 0);
        assert!((state.values()[0] - (-0.02_f64).exp()).abs() < 1e-5);
    }
}
