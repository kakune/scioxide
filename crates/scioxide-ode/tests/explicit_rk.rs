//! Shared explicit workspace and continuous-extension failure regressions.
#![allow(clippy::unwrap_used)]

use core::cell::Cell;
use scioxide_core::{EvaluationRole, ExecutionContext, SciOxideError};
use scioxide_linalg::Vector;
use scioxide_ode::{
    BuiltinOdeMethod, ClassicalRk4, DormandPrince5, DormandPrince853, ExplicitMidpoint,
    ForwardEuler, Heun, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
    OdeTrajectory, OdeWorkspace, Rkf45, Verner65,
};

struct Decay {
    calls: Cell<usize>,
    fail_at: Option<usize>,
}
impl Decay {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at,
        }
    }
}
impl OdeSystem<f64> for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.fail_at == Some(call) {
            out[0] = f64::NAN;
        } else {
            out[0] = -y[0];
        }
        Ok(())
    }
}

/// One accepted step of 0.125 through a shared workspace; dense methods also sample
/// the midpoint.
fn reuse(
    method: impl Into<BuiltinOdeMethod<f64>>,
    dense: bool,
    workspace: &mut OdeWorkspace<f64, BuiltinOdeMethod<f64>>,
    tolerance: f64,
) {
    let system = Decay::new(None);
    let schedule = if dense {
        OdeOutputSchedule::times(vec![0.0625])
    } else {
        OdeOutputSchedule::FinalOnly
    };
    let plan = (
        method.into(),
        OdeOptions::new()
            .with_initial_step(0.125)
            .with_output_schedule(schedule),
    );
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let mut trajectory = OdeTrajectory::new();
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&system, 0.125),
            &plan.1,
            &mut state,
            &mut trajectory,
            workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(report.rejected_steps(), 0);
    assert!((state.values()[0] - (-0.125_f64).exp()).abs() < tolerance);
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.calls.get()
    );
    if dense {
        assert_eq!(trajectory.times(), [0.0625]);
        let middle = trajectory.states()[0].as_slice()[0];
        assert!((middle - (-0.0625_f64).exp()).abs() < 1e-5);
    }
}

#[test]
fn smaller_methods_reuse_a_workspace_containing_dop853_extension_stages() {
    let mut workspace = OdeWorkspace::new();
    reuse(DormandPrince853::new(), true, &mut workspace, 1e-12);
    reuse(DormandPrince5::new(), true, &mut workspace, 1e-8);
    reuse(Rkf45::new(), true, &mut workspace, 1e-8);
    reuse(Verner65::new(), true, &mut workspace, 1e-8);
    reuse(ClassicalRk4::new(), false, &mut workspace, 1e-6);
    reuse(Heun::new(), false, &mut workspace, 4e-4);
    reuse(ExplicitMidpoint::new(), false, &mut workspace, 4e-4);
    reuse(ForwardEuler::new(), false, &mut workspace, 8e-3);
}

#[test]
fn failed_dop853_dense_stage_keeps_state_and_starting_derivative() {
    // Main stages and the endpoint succeed; the first extra dense stage of the first
    // attempt returns NaN. The driver rejects that attempt without committing it and
    // retries half the step, reusing the starting derivative.
    let system = Decay::new(Some(13));
    let plan = (
        DormandPrince853::new(),
        OdeOptions::new()
            .with_initial_step(0.125)
            .with_output_schedule(OdeOutputSchedule::times(vec![0.0625])),
    );
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let mut accepted = OdeTrajectory::new();
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&system, 0.125),
            &plan.1,
            &mut state,
            &mut accepted,
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report.accepted_steps(), 2);
    assert_eq!(report.rejected_steps(), 1);
    // 13 successful calls and the failing dense stage, then two 12-call steps, each
    // reusing its starting derivative. The requested time ends the first of them, so
    // neither reads an interpolant or builds extension stages.
    assert_eq!(system.calls.get(), 14 + 12 + 12);
    assert_eq!(
        report.counts().provider(EvaluationRole::RightHandSide),
        system.calls.get()
    );
    assert_eq!(accepted.times(), [0.0625]);
    assert!((accepted.states()[0].as_slice()[0] - (-0.0625_f64).exp()).abs() < 1e-12);
    assert!((state.values()[0] - (-0.125_f64).exp()).abs() < 1e-12);
}

fn fixed_zero_node<M: OdeMethod<f64> + Clone>(method: M) {
    struct ZeroDerivative(Cell<Option<u64>>);
    impl OdeSystem<f64> for ZeroDerivative {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            time: f64,
            _: &[f64],
            out: &mut [f64],
            _cx: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            if self.0.get().is_none() {
                self.0.set(Some(time.to_bits()));
            }
            out.fill(0.0);
            Ok(())
        }
    }
    let system = ZeroDerivative(Cell::new(None));
    let plan = (method, OdeOptions::new().with_initial_step(0.125));
    let mut state = OdeState::new(-0.0, Vector::from_vec(vec![1.0]));
    let report = plan
        .0
        .advance(
            &OdeProblem::new(&system, -0.125),
            &plan.1,
            &mut state,
            &mut (),
            &mut OdeWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(system.0.get(), Some(0.0_f64.to_bits()));
    assert_eq!(state.values(), &[1.0]);
}

#[test]
fn fixed_zero_nodes_retain_checked_time_rounding() {
    fixed_zero_node(ForwardEuler::new());
    fixed_zero_node(ExplicitMidpoint::new());
    fixed_zero_node(Heun::new());
    fixed_zero_node(ClassicalRk4::new());
}

#[test]
fn fixed_grids_from_rounded_steps_end_without_a_spurious_step() {
    use scioxide_ode::{ClassicalRk4, GaussRk, OdeFn, OdeOptions, OdeProblem, OdeState};
    let rhs = OdeFn::new(1, |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -y[0];
        Ok(())
    })
    .unwrap();
    let problem = OdeProblem::new(&rhs, 3.0);
    // 3/100 is not exact in binary; summing it 100 times misses 3 by a few ulps.
    let options = OdeOptions::new().with_initial_step(3.0 / 100.0);
    let initial = || OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let rk4 = ClassicalRk4::new()
        .solve(&problem, &options, initial())
        .unwrap();
    assert_eq!(rk4.report().accepted_steps(), 100);
    assert_eq!(rk4.state().time(), 3.0);
    let gauss = GaussRk::new().solve(&problem, &options, initial()).unwrap();
    assert_eq!(gauss.report().accepted_steps(), 100);
    assert!((rk4.state().values()[0] - (-3.0_f64).exp()).abs() < 1e-8);
}
