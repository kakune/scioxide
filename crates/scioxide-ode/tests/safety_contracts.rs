//! Boundary and lifecycle regressions for the checked ODE driver.
//!
//! Malformed method workspaces and error estimates use crate-private steppers
//! and are covered by the crate's internal driver tests.

#![allow(clippy::unwrap_used)]

use std::cell::Cell;
use std::rc::Rc;

use scioxide_core::{AttemptOutcome, SciOxideError, SolveContext};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, EventDirection, EventFunction, ForwardEuler, OdeMethod, OdeOptions,
    OdeStepControl, OdeSystem, OdeTermination, OdeWorkspace, ProportionalIntegralController,
};
mod support;
use support::{ScenarioConfig, run_solve, run_solve_with_workspace};

struct ZeroSystem {
    calls: Option<Rc<Cell<usize>>>,
}

impl ZeroSystem {
    const fn new() -> Self {
        Self { calls: None }
    }

    fn counting(calls: Rc<Cell<usize>>) -> Self {
        Self { calls: Some(calls) }
    }
}

impl OdeSystem<f64> for ZeroSystem {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if let Some(calls) = &self.calls {
            calls.set(calls.get() + 1);
        }
        derivative.fill(0.0);
        Ok(())
    }
}

struct CountingTerminalEvent {
    calls: Rc<Cell<usize>>,
}

impl EventFunction<f64> for CountingTerminalEvent {
    fn eval(&self, _time: f64, _state: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        Ok(0.0)
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        true
    }
}

fn initial_step_options(step: f64) -> ScenarioConfig<f64> {
    ScenarioConfig::default().with_initial_step(step).unwrap()
}

#[test]
fn zero_error_grows_to_both_endpoints_efficiently() {
    for span in [(0.0, 1.0), (1.0, 0.0)] {
        let solution = run_solve(
            &ZeroSystem::new(),
            &DormandPrince5::new(),
            span,
            Vector::from_vec(vec![2.0]),
            ScenarioConfig::default(),
        )
        .unwrap();
        assert_eq!(solution.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(solution.times().last().copied(), Some(span.1));
        assert!(solution.stats().num_steps() < 20);
    }
}

#[test]
fn first_and_later_steps_obey_the_maximum_bound() {
    let options = initial_step_options(0.75).with_max_step(0.2).unwrap();
    let solution = run_solve(
        &ZeroSystem::new(),
        &ForwardEuler::new(),
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    )
    .unwrap();

    assert_eq!(solution.times().last().copied(), Some(1.0));
    for times in solution.times().windows(2) {
        assert!(times[1] - times[0] <= 0.2 + f64::EPSILON);
    }
}

#[test]
fn opposite_extreme_endpoints_do_not_overflow_span_or_step_arithmetic() {
    for span in [(-f64::MAX, f64::MAX), (f64::MAX, -f64::MAX)] {
        let solution = run_solve(
            &ZeroSystem::new(),
            &DormandPrince5::new(),
            span,
            Vector::from_vec(vec![0.0]),
            initial_step_options(f64::MAX / 4.0),
        )
        .unwrap();
        assert_eq!(solution.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(solution.times().last().copied(), Some(span.1));
        assert!(solution.stats().num_steps() <= 4);
    }
}

#[test]
fn time_addition_collapse_is_explicit() {
    let start = 1.0e300_f64;
    let end = f64::from_bits(start.to_bits() + 1);
    let result = run_solve(
        &ZeroSystem::new(),
        &ForwardEuler::new(),
        (start, end),
        Vector::from_vec(vec![0.0]),
        initial_step_options(1.0),
    );
    let result = result.unwrap();
    assert_eq!(result.termination(), Some(OdeTermination::StepSizeLimit));
    assert_eq!(result.stats().num_accepted(), 0);
    assert_eq!(result.final_state().time(), start);
}

#[test]
fn invalid_initial_state_and_controller_precede_all_callbacks() {
    let rhs_calls = Rc::new(Cell::new(0));
    let event_calls = Rc::new(Cell::new(0));
    let options = ScenarioConfig::default().with_event(Box::new(CountingTerminalEvent {
        calls: Rc::clone(&event_calls),
    }));
    let mut workspace = OdeWorkspace::new();
    let result = run_solve_with_workspace(
        &ZeroSystem::counting(Rc::clone(&rhs_calls)),
        &DormandPrince5::new(),
        &mut workspace,
        (0.0, 1.0),
        Vector::from_vec(vec![f64::NAN]),
        options,
    );
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(rhs_calls.get(), 0);
    assert_eq!(event_calls.get(), 0);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.progress().is_none());

    // Controllers are validated when constructed, and explicit control is rejected for
    // a fixed-step method when the plan is built: no callback can observe either.
    assert!(matches!(
        ProportionalIntegralController::new(-0.5, 5.0, 0.2, 1.0, 0.0)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        ProportionalIntegralController::new(0.9, 0.5, 2.0, 0.7, 0.4)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let control = OdeStepControl::ProportionalIntegral(
        ProportionalIntegralController::new(0.9, 5.0, 0.2, 1.0, 0.0).unwrap(),
    );
    assert!(matches!(
        (ForwardEuler::<f64>::new())
            .validate(&(OdeOptions::new().with_step_control(control)), 1)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}
