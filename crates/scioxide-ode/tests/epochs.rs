//! Solve-epoch and terminal-continuation tests.
//!
//! Raw component errors, FSAL reuse, multistep restarts, custom component norms and
//! derivative-policy cache keys use direct stepping and are covered by the crate's
//! internal driver tests.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, NumericScalar,
    RealScalar, SciOxideError, SolveContext, StepTolerance,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    BackwardEuler, DormandPrince5, EventDirection, EventFunction, OdeEventOptions, OdeMethod,
    OdeOptions, OdeProblem, OdeState, OdeSystem, OdeTermination, OdeWorkspace,
};

struct Linear<S> {
    rate: S,
    dimension: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Linear<S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.iter_mut()
            .zip(y)
            .for_each(|(out, y)| *out = self.rate * *y);
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        out: &mut scioxide_linalg::MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let out_data = out.as_contiguous_slice_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "row-major output required",
            )
        })?;
        out_data.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        for i in 0..self.dimension {
            out_data[i * self.dimension + i] = self.rate;
        }
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn same_shape_different_systems_reset_implicit_solve_caches() {
    let plan = (
        BackwardEuler::<f64>::new(),
        OdeOptions::new().with_initial_step(0.125),
    );
    let mut workspace = OdeWorkspace::new();
    for rate in [-1.0, -8.0] {
        let system = Linear { rate, dimension: 1 };
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
        let report = plan
            .0
            .advance(
                &OdeProblem::new(&system, 0.125),
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((state.values()[0] - 1.0 / (1.0 - 0.125 * rate)).abs() < 1e-12);
    }
}

struct TerminalTime(std::rc::Rc<std::cell::Cell<bool>>);
impl EventFunction<f64> for TerminalTime {
    fn eval(&self, t: f64, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(if self.0.get() { t - 0.3 } else { 1.0 })
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

#[test]
fn terminal_continuation_matches_fresh_solve_from_localized_state() {
    let enabled = std::rc::Rc::new(std::cell::Cell::new(true));
    let options = OdeOptions::new()
        .with_initial_step(0.5)
        .with_error_tolerance(ErrorTolerance::scalar(1e-3, 1e-3, NormSelection::Infinity).unwrap());
    let plan = (
        DormandPrince5::new(),
        options
            .clone()
            .with_event_options(OdeEventOptions::new().with_localization(
                StepTolerance::new(1e-13, 1e-13).unwrap(),
                Budget::ONE.saturating_add(63),
            )),
    );
    let system = Linear {
        rate: 1.0,
        dimension: 1,
    };
    let event = TerminalTime(std::rc::Rc::clone(&enabled));
    let events: [&dyn EventFunction<f64>; 1] = [&event];
    let problem = OdeProblem::new(&system, 1.0).with_events(&events);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let terminal = plan
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
    assert_eq!(terminal.termination(), Some(OdeTermination::TerminalEvent));
    assert!((terminal.final_time() - 0.3).abs() < 1e-12);
    assert_eq!(state.time(), terminal.final_time());
    enabled.set(false);
    let initial = state.clone();
    let continued = plan
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
    let expected = DormandPrince5::new()
        .solve(&OdeProblem::new(&system, 1.0), &(options), initial)
        .unwrap();
    assert_eq!(state.values(), expected.state().values());
    assert_eq!(
        continued.counts().provider(EvaluationRole::RightHandSide),
        expected
            .report()
            .counts()
            .provider(EvaluationRole::RightHandSide)
    );
}
