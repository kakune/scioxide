//! Integration tests for scioxide-ode.
//!
//! Tests cover: exponential decay, harmonic oscillator, stiff problems,
//! event detection, backward integration, requested output interpolation, built-in method selection,
//! and error paths.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::used_underscore_binding,
    clippy::field_reassign_with_default,
    clippy::manual_inspect,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::cell::Cell;
use std::rc::Rc;

use scioxide_core::LinearOperator;
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, EvaluationRole, ExecutionContext, NonFiniteOrigin,
    NumericScalar, RealScalar, SciOxideError, SolveContext, Tolerance,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    AdamsBashforth, AdamsMoulton, BackwardEuler, Bdf, BuiltinOdeMethod, ClassicalRk4,
    DormandPrince5, DormandPrince853, EventDirection, EventFunction, EventOccurrence,
    ExplicitMidpoint, ForwardEuler, GaussRk, Heun, ImplicitTrapezoid, NonlinearStageSolver,
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeStepControl, OdeSystem,
    OdeTermination, OdeWorkspace, ProportionalIntegralController, RadauIia, Rkf45, Sdirk, Verner65,
};

mod support;
use support::{ObservedTrajectory, ScenarioConfig, run_solve};

// ============================================================================
// Test ODE systems
// ============================================================================

/// Exponential decay: y' = -y, y(0) = 1. Solution: y = exp(-t).
struct ExponentialDecay;

impl OdeSystem<f64> for ExponentialDecay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _t: f64,
        y: &[f64],
        dydt: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        dydt[0] = -y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _t: f64,
        _y: &[f64],
        jac: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        jac.set(0, 0, -1.0)?;
        Ok(Some(()))
    }
}

/// Zero dynamics used when only time-driver behavior is under test.
struct ZeroDynamics;

impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for ZeroDynamics {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _t: S,
        _y: &[S],
        dydt: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        dydt.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        Ok(())
    }
}

struct CountingSystem {
    rhs_calls: Rc<Cell<usize>>,
}

struct FailingSystem;

impl OdeSystem<f64> for FailingSystem {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        _derivative: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("derivative"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "intentional callback failure",
        ))
    }
}

impl OdeSystem<f64> for CountingSystem {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _t: f64,
        _y: &[f64],
        dydt: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs_calls.set(self.rhs_calls.get() + 1);
        dydt[0] = 0.0;
        Ok(())
    }
}

struct CountingMassSystem {
    rhs_calls: Rc<Cell<usize>>,
}

impl OdeSystem<f64> for CountingMassSystem {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _t: f64,
        _y: &[f64],
        dydt: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.rhs_calls.set(self.rhs_calls.get() + 1);
        dydt[0] = 0.0;
        Ok(())
    }

    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(0)
    }
}

struct CountingEvent {
    calls: Rc<Cell<usize>>,
}

impl EventFunction<f64> for CountingEvent {
    fn eval(&self, t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        Ok(t - 0.5)
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        true
    }
}

/// Stiff Van der Pol oscillator: y1' = y2, y2' = mu*(1-y1^2)*y2 - y1.
struct VanDerPol {
    mu: f64,
}

impl OdeSystem<f64> for VanDerPol {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _t: f64,
        y: &[f64],
        dydt: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let s = y;
        let d = dydt;
        d[0] = s[1];
        d[1] = self.mu * (1.0 - s[0] * s[0]) * s[1] - s[0];
        Ok(())
    }
}

/// Bouncing ball: y1' = y2, y2' = -g. Event at y1 = 0 (ground contact).
struct FreeFall;

impl OdeSystem<f64> for FreeFall {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _t: f64,
        _y: &[f64],
        dydt: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let d = dydt;
        d[0] = _y[1]; // velocity
        d[1] = -9.81; // gravity
        Ok(())
    }
}

/// Ground contact event: triggers when y1 = 0 (falling direction).
struct GroundContact;

impl EventFunction<f64> for GroundContact {
    fn eval(&self, _t: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(y[0])
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Falling
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

/// Event at a fixed time `t = 0.3` for terminal event/requested_output tests.
struct TimeEvent;

impl EventFunction<f64> for TimeEvent {
    fn eval(&self, t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(t - 0.3)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Rising
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

/// Configurable time crossing used to exercise event ordering policies.
struct TimeCrossing<S> {
    root: S,
    direction: EventDirection,
    terminal: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>> EventFunction<S> for TimeCrossing<S> {
    fn eval(&self, t: S, _y: &[S]) -> Result<S, SciOxideError> {
        Ok(t - self.root)
    }

    fn direction(&self) -> EventDirection {
        self.direction
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

/// `t (t - 0.5)`: zero at `t = 0`, negative until a rising root at `t = 0.5`.
struct InitialAndLaterRoot;

impl EventFunction<f64> for InitialAndLaterRoot {
    fn eval(&self, t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(t * (t - 0.5))
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        false
    }
}

/// Event function that remains at zero for the whole solve.
struct ConstantRoot {
    terminal: bool,
}

impl EventFunction<f64> for ConstantRoot {
    fn eval(&self, _t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

/// Terminal event that is identically zero on `[0, 0.25]`, then `(t - 0.25)(t - 0.75)`:
/// it leaves zero downwards and rises through zero at `t = 0.75`.
struct DelayedRisingRoot;

impl EventFunction<f64> for DelayedRisingRoot {
    fn eval(&self, t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(if t <= 0.25 {
            0.0
        } else {
            (t - 0.25) * (t - 0.75)
        })
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        true
    }
}

/// Terminal ground contact `g = y₁` with a configurable direction filter.
struct Ground(EventDirection);

impl EventFunction<f64> for Ground {
    fn eval(&self, _t: f64, y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(y[0])
    }

    fn direction(&self) -> EventDirection {
        self.0
    }

    fn is_terminal(&self) -> bool {
        true
    }
}

/// Tangential contact at a configured time.
struct TangentialTimeContact {
    root: f64,
}

impl EventFunction<f64> for TangentialTimeContact {
    fn eval(&self, t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok((t - self.root).powi(2))
    }

    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }

    fn is_terminal(&self) -> bool {
        false
    }
}

/// Event function that returns NaN for validation testing.
struct NaNEvent;

impl EventFunction<f64> for NaNEvent {
    fn eval(&self, _t: f64, _y: &[f64]) -> Result<f64, SciOxideError> {
        Ok(f64::NAN)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

/// Nonlinear solver that always fails (to verify solver injection paths are used).
struct RejectingNonlinearSolver;

impl NonlinearStageSolver<f64> for RejectingNonlinearSolver {
    fn solve_in_place(
        &self,
        _residual_fn: &mut dyn FnMut(
            &[f64],
            &mut [f64],
            &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError>,
        _operator: &dyn scioxide_core::LinearOperator<f64>,
        _state: &mut [f64],
        _residual: &mut [f64],
        _correction: &mut [f64],
        _accuracy: scioxide_ode::StageAccuracy<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<scioxide_ode::NonlinearStageReport, SciOxideError> {
        Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_counts((scioxide_core::EvaluationCounts::from_iterations(0)).into()),
            "forced nonlinear failure for test",
        ))
    }
}

#[derive(Clone, Copy)]
enum InvalidNonlinearOutcome {
    MalformedShape,
    NonFinite,
}

struct InvalidNonlinearSolver {
    outcome: InvalidNonlinearOutcome,
}

impl NonlinearStageSolver<f64> for InvalidNonlinearSolver {
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
            InvalidNonlinearOutcome::MalformedShape => Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(1),
                scioxide_core::Dimensions::one(0),
            )),
            InvalidNonlinearOutcome::NonFinite => {
                state.fill(f64::NAN);
                Ok(scioxide_ode::NonlinearStageReport::new(1, 0, 0))
            }
        }
    }
}

/// Iterative linear solver that always reports non-convergence.
struct RejectingLinearSolver;

impl scioxide_ode::StageLinearSolver<f64> for RejectingLinearSolver {
    fn solve_into(
        &self,
        _operator: &dyn LinearOperator<f64>,
        _state: &[f64],
        _residual: &[f64],
        _correction: &mut [f64],
        _accuracy: scioxide_ode::StageAccuracy<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_counts((scioxide_core::EvaluationCounts::from_iterations(7)).into()),
            "linear iteration limit",
        ))
    }
}

// ============================================================================
// Helper
// ============================================================================

fn dense_capable_solvers() -> Vec<BuiltinOdeMethod<f64>> {
    vec![
        BuiltinOdeMethod::from(RadauIia::<f64>::new()),
        BuiltinOdeMethod::from(Rkf45::<f64>::new()),
        BuiltinOdeMethod::from(DormandPrince5::<f64>::new()),
        BuiltinOdeMethod::from(DormandPrince853::<f64>::new()),
        BuiltinOdeMethod::from(Verner65::<f64>::new()),
        BuiltinOdeMethod::from(
            AdamsBashforth::<f64>::with_order_policy(scioxide_ode::MultistepOrder::Automatic(5))
                .unwrap(),
        ),
        BuiltinOdeMethod::from(
            AdamsMoulton::<f64>::with_order_policy(scioxide_ode::MultistepOrder::Automatic(5))
                .unwrap(),
        ),
        BuiltinOdeMethod::from(
            Bdf::<f64>::with_order_policy(scioxide_ode::MultistepOrder::Automatic(5)).unwrap(),
        ),
    ]
}

fn non_dense_solvers() -> Vec<BuiltinOdeMethod<f64>> {
    vec![
        BuiltinOdeMethod::from(ForwardEuler::<f64>::new()),
        BuiltinOdeMethod::from(ExplicitMidpoint::<f64>::new()),
        BuiltinOdeMethod::from(Heun::<f64>::new()),
        BuiltinOdeMethod::from(ClassicalRk4::<f64>::new()),
        BuiltinOdeMethod::from(BackwardEuler::<f64>::new()),
        BuiltinOdeMethod::from(ImplicitTrapezoid::<f64>::new()),
        BuiltinOdeMethod::from(GaussRk::<f64>::new()),
        BuiltinOdeMethod::from(Sdirk::<f64>::new()),
    ]
}

fn solve_time_events(
    interval: (f64, f64),
    events: Vec<Box<dyn EventFunction<f64>>>,
    requested_output: Option<Vec<f64>>,
) -> Result<ObservedTrajectory<f64>, SciOxideError> {
    let solver = DormandPrince5::<f64>::new();
    let options = ScenarioConfig::default()
        .with_initial_step((interval.1 - interval.0).abs())?
        .with_events(events)
        .with_event_localization(Tolerance::new(1e-12, 1e-12)?, Budget::try_from(128)?);
    let options = match requested_output {
        Some(times) => options.with_evaluation_times(times)?,
        None => options,
    };
    run_solve(
        &ZeroDynamics,
        &solver,
        interval,
        Vector::from_vec(vec![0.0]),
        options,
    )
}

fn solve_zero_dynamics<S>(interval: (S, S)) -> Result<ObservedTrajectory<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let (t0, t_end) = interval;
    let span = if t_end < t0 { t0 - t_end } else { t_end - t0 };
    let y0 = Vector::from_vec(vec![<S as scioxide_core::AlgebraicScalar>::ZERO]);
    let solver = ForwardEuler::<S>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(span).unwrap();
    run_solve(&ZeroDynamics, &solver, interval, y0, opts)
}

/// Dormand–Prince on exponential decay with a fixed first step and minimum step.
fn run_minimum_step_driver(
    interval: (f64, f64),
    initial_step: f64,
    min_step: f64,
    tolerance: Tolerance<f64>,
) -> Result<ObservedTrajectory<f64>, SciOxideError> {
    let decay = (-interval.0).exp();
    let opts = ScenarioConfig::default()
        .with_tolerance(tolerance)
        .with_initial_step(initial_step)?
        .with_min_step(min_step)?;
    run_solve(
        &ExponentialDecay,
        &DormandPrince5::new(),
        interval,
        Vector::from_vec(vec![decay]),
        opts,
    )
}

// ============================================================================
// Stiff problem: Van der Pol with implicit solver
// ============================================================================

#[test]
fn test_backward_euler_van_der_pol() {
    let sys = VanDerPol { mu: 10.0 };
    let y0 = Vector::from_vec(vec![2.0, 0.0]);
    let solver = BackwardEuler::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.001).unwrap();

    // Just verify it doesn't blow up — stiff problem
    let sol = run_solve(&sys, &solver, (0.0, 1.0), y0, opts).unwrap();
    assert!(sol.times().len() > 1);
    assert!(
        sol.states()
            .last()
            .unwrap()
            .as_slice()
            .iter()
            .all(|v| v.is_finite())
    );
}

// ============================================================================
// Event detection
// ============================================================================

#[test]
fn test_event_detection_bouncing_ball() {
    let sys = FreeFall;
    // Initial height 10, velocity 0 → hits ground at t = sqrt(2*10/9.81) ≈ 1.428
    let y0 = Vector::from_vec(vec![10.0, 0.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();
    opts = opts.with_event(Box::new(GroundContact));

    let sol = run_solve(&sys, &solver, (0.0, 5.0), y0, opts).unwrap();
    assert_eq!(sol.termination(), Some(OdeTermination::TerminalEvent));
    assert!(sol.is_success());

    // Should have detected the terminal event
    assert!(!sol.events().is_empty(), "no events detected");

    // The event should occur around t ≈ 1.428
    let t_event = sol.events()[0].time();
    let expected_t = (2.0 * 10.0 / 9.81_f64).sqrt();
    assert!(
        (t_event - expected_t).abs() < 0.1,
        "event time {t_event}, expected ~{expected_t}"
    );

    // Integration should have stopped at the event
    let t_final = *sol.times().last().unwrap();
    assert!(
        t_final < 2.0,
        "integration should have stopped at event, but t_final={t_final}"
    );
}

#[test]
fn test_close_events_stop_at_first_terminal_in_integration_order() {
    let cases = [
        (
            (0.0, 1.0),
            [0.3004, 0.3000, 0.3002, 0.3001, 0.3003],
            EventOccurrence::Rising,
        ),
        (
            (1.0, 0.0),
            [0.6996, 0.7000, 0.6998, 0.6999, 0.6997],
            EventOccurrence::Falling,
        ),
    ];

    for (interval, roots, occurrence) in cases {
        let events = roots
            .into_iter()
            .enumerate()
            .map(|(index, root)| {
                Box::new(TimeCrossing {
                    root,
                    direction: EventDirection::Either,
                    terminal: index == 0 || index == 2,
                }) as Box<dyn EventFunction<f64>>
            })
            .collect();

        let solution = solve_time_events(interval, events, None).unwrap();
        assert_eq!(
            solution
                .events()
                .iter()
                .map(|event| event.index())
                .collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
        assert!(
            solution
                .events()
                .iter()
                .all(|event| event.occurrence() == occurrence)
        );
        assert!((solution.times().last().unwrap() - roots[2]).abs() < 1e-10);

        let chronological = if interval.0 < interval.1 {
            solution.times().windows(2).all(|pair| pair[0] <= pair[1])
        } else {
            solution.times().windows(2).all(|pair| pair[0] >= pair[1])
        };
        assert!(chronological);
    }
}

#[test]
fn test_simultaneous_events_retain_all_terminal_ties() {
    for (interval, occurrence) in [
        ((0.0, 1.0), EventOccurrence::Rising),
        ((1.0, 0.0), EventOccurrence::Falling),
    ] {
        let events = [false, true, false, true]
            .into_iter()
            .map(|terminal| {
                Box::new(TimeCrossing {
                    root: 0.5,
                    direction: EventDirection::Either,
                    terminal,
                }) as Box<dyn EventFunction<f64>>
            })
            .collect();

        let solution = solve_time_events(interval, events, None).unwrap();
        assert_eq!(solution.events().len(), 4);
        assert_eq!(solution.events()[0].index(), 0);
        assert_eq!(solution.events()[1].index(), 1);
        assert!(!solution.events()[0].is_terminal());
        assert!(solution.events()[1].is_terminal());
        assert!(
            solution
                .events()
                .iter()
                .all(|event| event.occurrence() == occurrence)
        );
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1e-10);
    }
}

#[test]
fn test_terminal_event_truncates_requested_output_in_both_directions() {
    let cases = [
        ((0.0, 1.0), vec![0.1, 0.2, 0.3, 0.4, 0.9], 0.35),
        ((1.0, 0.0), vec![0.9, 0.8, 0.7, 0.6, 0.1], 0.65),
    ];

    for (interval, requested_output, root) in cases {
        let events: Vec<Box<dyn EventFunction<f64>>> = vec![Box::new(TimeCrossing {
            root,
            direction: EventDirection::Either,
            terminal: true,
        })];
        let solution = solve_time_events(interval, events, Some(requested_output)).unwrap();

        // Only the reached requested prefix is emitted.
        assert_eq!(solution.times().len(), 3);
        assert!((solution.events()[0].time() - root).abs() < 1e-10);
        let chronological = if interval.0 < interval.1 {
            solution.times().windows(2).all(|pair| pair[0] < pair[1])
        } else {
            solution.times().windows(2).all(|pair| pair[0] > pair[1])
        };
        assert!(chronological);
    }
}

#[test]
fn test_exact_zero_at_start_is_not_an_event_in_either_direction() {
    // Every function is exactly zero at the start; the time roots leave zero in their
    // own and in the excluded direction, and the constant one never leaves it.
    for (interval, root) in [((0.0, 1.0), 0.0), ((1.0, 0.0), 1.0)] {
        let events: Vec<Box<dyn EventFunction<f64>>> = vec![
            Box::new(TimeCrossing {
                root,
                direction: EventDirection::Rising,
                terminal: true,
            }),
            Box::new(TimeCrossing {
                root,
                direction: EventDirection::Falling,
                terminal: true,
            }),
            Box::new(TimeCrossing {
                root,
                direction: EventDirection::Either,
                terminal: true,
            }),
            Box::new(ConstantRoot { terminal: true }),
        ];

        let solution = solve_time_events(interval, events, None).unwrap();
        assert_eq!(solution.termination(), Some(OdeTermination::ReachedEnd));
        assert!(solution.events().is_empty());
        assert_eq!(solution.times().last().copied(), Some(interval.1));
        assert!(solution.stats().num_accepted() >= 1);
    }
}

#[test]
fn test_root_after_an_initial_zero_is_reported_once() {
    let solver = DormandPrince5::<f64>::new();
    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(0.25).unwrap();
    options = options.with_events(vec![
        Box::new(ConstantRoot { terminal: false }),
        Box::new(InitialAndLaterRoot),
    ]);

    let solution = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    )
    .unwrap();

    assert_eq!(solution.events().len(), 1);
    assert_eq!(solution.events()[0].index(), 1);
    assert_eq!(solution.events()[0].occurrence(), EventOccurrence::Rising);
    assert!((solution.events()[0].time() - 0.5).abs() <= f64::SQRT_EPSILON);
}

#[test]
fn test_function_identically_zero_near_the_start_does_not_stall() {
    let solver = DormandPrince5::<f64>::new();
    let options = ScenarioConfig::default()
        .with_initial_step(0.1)
        .unwrap()
        .with_max_step(0.1)
        .unwrap()
        .with_event(Box::new(DelayedRisingRoot));

    let solution = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    )
    .unwrap();

    // Leaving zero downwards after t = 0.25 is not a crossing; the rise at 0.75 is.
    assert_eq!(solution.termination(), Some(OdeTermination::TerminalEvent));
    assert_eq!(solution.events().len(), 1);
    assert_eq!(solution.events()[0].occurrence(), EventOccurrence::Rising);
    assert!((solution.events()[0].time() - 0.75).abs() < 1e-7);
    assert!(solution.stats().num_accepted() >= 7);
}

/// Integrate the free-falling ball from `(t0, y0)` until its terminal ground event.
fn bounce(t0: f64, y0: [f64; 2], event: Ground) -> ObservedTrajectory<f64> {
    let options = ScenarioConfig::default()
        .with_tolerance(Tolerance::new(1e-10, 1e-10).unwrap())
        .with_event(Box::new(event));
    run_solve(
        &FreeFall,
        &DormandPrince5::new(),
        (t0, 50.0),
        Vector::from_vec(y0.to_vec()),
        options,
    )
    .unwrap()
}

#[test]
fn test_falling_restart_from_the_ground_proceeds_to_the_next_bounce() {
    // Hybrid restart: reflect and damp the velocity, and put the ball exactly on the
    // ground. Each flight then lasts 2v/g, whatever the excluded rising direction.
    let mut time = 0.0;
    let mut state = [1.0, 0.0];
    let mut expected = (2.0 / 9.81_f64).sqrt();
    for _ in 0..6 {
        let solution = bounce(time, state, Ground(EventDirection::Falling));
        assert_eq!(solution.termination(), Some(OdeTermination::TerminalEvent));
        assert!(solution.stats().num_accepted() >= 1);
        assert_eq!(solution.events().len(), 1);
        assert_eq!(solution.events()[0].occurrence(), EventOccurrence::Falling);
        let landing = solution.final_state();
        assert!((landing.time() - expected).abs() < 1e-7);
        let speed = 0.9 * landing.values()[1].abs();
        time = landing.time();
        state = [0.0, speed];
        expected = time + 2.0 * speed / 9.81;
    }
}

#[test]
fn test_either_restart_from_the_ground_fires_at_the_next_crossing() {
    let speed = 3.0;
    let solution = bounce(1.0, [0.0, speed], Ground(EventDirection::Either));
    assert_eq!(solution.termination(), Some(OdeTermination::TerminalEvent));
    assert!(solution.stats().num_accepted() >= 1);
    assert_eq!(solution.events().len(), 1);
    assert_eq!(solution.events()[0].occurrence(), EventOccurrence::Falling);
    assert!((solution.events()[0].time() - (1.0 + 2.0 * speed / 9.81)).abs() < 1e-7);
    assert!(solution.final_state().values()[0].abs() < 1e-7);
}

#[test]
fn test_direction_filters_follow_integration_order() {
    for (interval, expected_index, occurrence) in [
        ((0.0, 1.0), 0, EventOccurrence::Rising),
        ((1.0, 0.0), 1, EventOccurrence::Falling),
    ] {
        let events: Vec<Box<dyn EventFunction<f64>>> = vec![
            Box::new(TimeCrossing {
                root: 0.5,
                direction: EventDirection::Rising,
                terminal: false,
            }),
            Box::new(TimeCrossing {
                root: 0.5,
                direction: EventDirection::Falling,
                terminal: false,
            }),
        ];
        let solution = solve_time_events(interval, events, None).unwrap();

        assert_eq!(solution.events().len(), 1);
        assert_eq!(solution.events()[0].index(), expected_index);
        assert_eq!(solution.events()[0].occurrence(), occurrence);
    }
}

#[test]
fn test_tangential_contacts_follow_the_documented_endpoint_policy() {
    let interior_events: Vec<Box<dyn EventFunction<f64>>> =
        vec![Box::new(TangentialTimeContact { root: 0.5 })];
    let interior = solve_time_events((0.0, 1.0), interior_events, None).unwrap();
    assert!(interior.events().is_empty());

    for interval in [(0.0, 1.0), (1.0, 0.0)] {
        let solver = DormandPrince5::<f64>::new();
        let mut options = ScenarioConfig::default();
        options = options.with_initial_step(0.5).unwrap();
        options = options.with_events(vec![Box::new(TangentialTimeContact { root: 0.5 })]);

        let solution = run_solve(
            &ZeroDynamics,
            &solver,
            interval,
            Vector::from_vec(vec![0.0]),
            options,
        )
        .unwrap();
        assert_eq!(solution.events().len(), 1);
        assert_eq!(solution.events()[0].occurrence(), EventOccurrence::Falling);
        assert!((solution.events()[0].time() - 0.5).abs() < f64::EPSILON);
    }
}

#[test]
fn test_event_localization_configuration_and_failure_are_explicit() {
    struct NonlinearCrossing;
    impl EventFunction<f64> for NonlinearCrossing {
        fn eval(&self, t: f64, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(t * t - 0.09)
        }
        fn direction(&self) -> EventDirection {
            EventDirection::Either
        }
        fn is_terminal(&self) -> bool {
            true
        }
    }
    let solver = DormandPrince5::<f64>::new();
    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(1.0).unwrap();
    options = options
        .with_events(vec![Box::new(TimeCrossing {
            root: 0.3,
            direction: EventDirection::Either,
            terminal: true,
        })])
        .with_event_localization(
            Tolerance::new(1e-12, 0.0).unwrap(),
            Budget::try_from(128).unwrap(),
        );

    let solution = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    )
    .unwrap();
    assert!((solution.events()[0].time() - 0.3).abs() < 1e-12);

    let solver = DormandPrince5::<f64>::new();
    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(1.0).unwrap();
    options = options
        .with_events(vec![Box::new(NonlinearCrossing)])
        .with_event_localization(Tolerance::new(0.0, 0.0).unwrap(), Budget::ONE);

    let result = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    );
    assert!(
        matches ! (& (result) , Err (error) if if let scioxide_core :: SciOxideErrorView :: ConvergenceFailure { diagnostics , .. } = error . view () { diagnostics . counts ().own().iterations() == 1 } else { false })
    );
}

#[test]
fn test_invalid_event_localization_and_missing_dense_output_are_errors() {
    assert!(matches!(
        (Budget::try_from(0)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));

    let solver = ForwardEuler::<f64>::new();
    let missing_dense_options = ScenarioConfig::default()
        .with_initial_step(1.0)
        .unwrap()
        .with_event(Box::new(TimeCrossing {
            root: 0.4,
            direction: EventDirection::Either,
            terminal: false,
        }));
    let missing_dense = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        missing_dense_options,
    );
    assert!(
        matches ! (& (missing_dense) , Err (error) if if let scioxide_core :: SciOxideErrorView :: InvalidInput { reason : scioxide_core :: InvalidInputReason :: UnsupportedConfiguration , detail , .. } = error . view () { detail . contains ("ForwardEuler") && detail . contains ("event localization") } else { false })
    );
}

#[test]
fn test_event_localization_supports_f32() {
    let solver = DormandPrince5::<f32>::new();
    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(1.0_f32).unwrap();
    options = options
        .with_events(vec![Box::new(TimeCrossing {
            root: 0.3_f32,
            direction: EventDirection::Either,
            terminal: true,
        })])
        .with_event_localization(
            Tolerance::new(1e-5_f32, 0.0).unwrap(),
            Budget::try_from(64).unwrap(),
        );

    let solution = run_solve(
        &ZeroDynamics,
        &solver,
        (0.0_f32, 1.0_f32),
        Vector::from_vec(vec![0.0_f32]),
        options,
    )
    .unwrap();
    assert!((solution.events()[0].time() - 0.3).abs() < 1e-5);
}

// ============================================================================
// requested_output interpolation
// ============================================================================

#[test]
fn test_requested_output_interpolation() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();
    opts = opts
        .with_evaluation_times(vec![0.0, 0.25, 0.5, 0.75, 1.0])
        .unwrap();

    let sol = run_solve(&sys, &solver, (0.0, 1.0), y0, opts).unwrap();

    // Should have exactly 5 output points (requested_output points)
    assert!(
        sol.times().len() >= 5,
        "expected at least 5 output points, got {}",
        sol.times().len()
    );

    // Check values at output points
    for (t_i, y_i) in sol.times().iter().zip(sol.states().iter()) {
        let expected = (-t_i).exp();
        let actual = y_i.as_slice()[0];
        assert!(
            (actual - expected).abs() < 1e-4,
            "at t={t_i}: actual={actual}, expected={expected}"
        );
    }
}

// ============================================================================
// Backward integration
// ============================================================================

#[test]
fn test_backward_integration() {
    let sys = ExponentialDecay;
    // Start from y(1) = exp(-1) and integrate backward to t=0
    let y0 = Vector::from_vec(vec![(-1.0_f64).exp()]);
    let solver = ClassicalRk4::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.01).unwrap();

    let sol = run_solve(&sys, &solver, (1.0, 0.0), y0, opts).unwrap();

    // Should recover y(0) ≈ 1
    let y_final = sol.states().last().unwrap().as_slice()[0];
    assert!(
        (y_final - 1.0).abs() < 1e-4,
        "backward integration: y(0) = {y_final}, expected 1.0"
    );
}

// ============================================================================
// Error paths
// ============================================================================

#[test]
fn test_distinct_tiny_spans_are_accepted_in_both_directions() {
    let tiny_f64 = f64::EPSILON / 2.0;
    let forward_f64 = solve_zero_dynamics((0.0_f64, tiny_f64)).unwrap();
    let backward_f64 = solve_zero_dynamics((tiny_f64, 0.0_f64)).unwrap();
    assert_eq!(forward_f64.times().last().copied(), Some(tiny_f64));
    assert_eq!(backward_f64.times().last().copied(), Some(0.0));

    let tiny_f32 = f32::EPSILON / 2.0;
    let forward_f32 = solve_zero_dynamics((0.0_f32, tiny_f32)).unwrap();
    let backward_f32 = solve_zero_dynamics((tiny_f32, 0.0_f32)).unwrap();
    assert_eq!(forward_f32.times().last().copied(), Some(tiny_f32));
    assert_eq!(backward_f32.times().last().copied(), Some(0.0));
}

#[test]
fn test_adjacent_times_at_large_origin_are_accepted() {
    let t0 = 2.0_f64.powi(53);
    let t_end = t0.next_up();
    assert_ne!(t0, t_end);

    let forward = solve_zero_dynamics((t0, t_end)).unwrap();
    let backward = solve_zero_dynamics((t_end, t0)).unwrap();
    assert_eq!(forward.times().last().copied(), Some(t_end));
    assert_eq!(backward.times().last().copied(), Some(t0));
}

#[test]
fn test_dimension_mismatch() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0, 2.0]); // dim 2 but system is dim 1
    let solver = ForwardEuler::<f64>::new();
    let opts = ScenarioConfig::default();

    let result = run_solve(&sys, &solver, (0.0, 1.0), y0, opts);
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
}

#[test]
fn test_invalid_interval_nan() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = ForwardEuler::<f64>::new();
    let opts = ScenarioConfig::default();

    let result = run_solve(&sys, &solver, (0.0, f64::NAN), y0, opts);
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn test_invalid_interval_equal() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = ForwardEuler::<f64>::new();
    let opts = ScenarioConfig::default();

    let result = run_solve(&sys, &solver, (1.0, 1.0), y0, opts);
    assert!(
        matches ! (& (result) , Err (error) if if let scioxide_core :: SciOxideErrorView :: InvalidInput { reason : scioxide_core :: InvalidInputReason :: Inconsistent , detail , .. } = error . view () { detail == "the state time and the problem end time must differ" } else { false })
    );
}

#[test]
fn test_rejection_below_minimum_retries_once_at_the_minimum_in_both_directions() {
    // A 1e-12 tolerance rejects a half-unit Dormand–Prince step by orders of magnitude,
    // so the controller proposes less than the 0.25 minimum: the retry runs at 0.25 and
    // its rejection ends the solve.
    for interval in [(0.0, 1.0), (1.0, 0.0)] {
        let result =
            run_minimum_step_driver(interval, 0.5, 0.25, Tolerance::new(1e-12, 1e-12).unwrap())
                .unwrap();
        assert_eq!(
            result.termination(),
            Some(scioxide_ode::OdeTermination::StepSizeLimit)
        );
        assert_eq!(result.stats().num_steps(), 2);
        assert_eq!(result.stats().num_rejected(), 2);
        assert!(result.last_step_error_estimate().unwrap() > 1.0);
        assert_eq!(result.final_state().time(), interval.0);
    }
}

#[test]
fn test_rejection_at_minimum_stops_without_spinning() {
    let result = run_minimum_step_driver(
        (0.0, 1.0),
        0.25,
        0.25,
        Tolerance::new(1e-12, 1e-12).unwrap(),
    )
    .unwrap();
    assert_eq!(
        result.termination(),
        Some(scioxide_ode::OdeTermination::StepSizeLimit)
    );
    assert_eq!(result.stats().num_steps(), 1);
    assert_eq!(result.stats().num_accepted(), 0);
}

#[test]
fn test_steps_at_minimum_can_be_accepted_in_both_directions() {
    for interval in [(0.0, 0.25), (0.25, 0.0)] {
        let result =
            run_minimum_step_driver(interval, 0.25, 0.25, Tolerance::new(1e-3, 1e-3).unwrap());
        let solution = result.unwrap();
        assert_eq!(solution.times().last().copied(), Some(interval.1));
        assert_eq!(solution.stats().num_steps(), 1);
        assert_eq!(solution.stats().num_accepted(), 1);
        assert_eq!(solution.stats().num_rejected(), 0);
    }
}

#[test]
fn test_invalid_initial_step_zero() {
    assert!(matches!(
        (ScenarioConfig::<f64>::default().with_initial_step(0.0))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn test_requested_output_requires_dense_output_capability() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = ForwardEuler::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();
    opts = opts.with_evaluation_times(vec![0.25, 0.5, 0.75]).unwrap();

    let result = run_solve(&sys, &solver, (0.0, 1.0), y0, opts);
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
}

#[test]
fn test_requested_output_must_be_monotonic() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();
    opts = opts.with_evaluation_times(vec![0.5, 0.25, 0.75]).unwrap();

    let result = run_solve(&sys, &solver, (0.0, 1.0), y0, opts);
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn test_step_limit_returns_partial_solution() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = ForwardEuler::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(1e-12).unwrap();
    opts = opts.with_max_step(1e-12).unwrap();
    opts = opts.with_max_steps(Budget::ONE);

    let solution = run_solve(&sys, &solver, (0.0, 1.0), y0, opts).unwrap();
    assert_eq!(
        solution.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert!(!solution.is_success());
    assert_eq!(solution.stats().num_steps(), 1);
    assert_eq!(solution.last_step_error_estimate(), None);
    assert_eq!(solution.times().len(), 2);
    assert!(*solution.times().last().unwrap() < 1.0);
    assert!(solution.message().contains("step limit"));
}

#[test]
fn test_step_budget_rejects_zero() {
    assert!(matches!(
        (Budget::try_from(0)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn test_non_finite_event_value_returns_error() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();
    opts = opts.with_event(Box::new(NaNEvent));

    let result = run_solve(&sys, &solver, (0.0, 1.0), y0, opts);
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
}

#[test]
fn test_terminal_event_with_requested_output_keeps_prior_samples() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.5).unwrap();
    opts = opts.with_evaluation_times(vec![0.1, 0.2]).unwrap();
    opts = opts.with_event(Box::new(TimeEvent));

    let sol = run_solve(&sys, &solver, (0.0, 1.0), y0, opts).unwrap();
    assert!(
        sol.times().iter().any(|t| (*t - 0.2).abs() < 1e-10),
        "expected requested output sample 0.2 to remain present"
    );
    assert_eq!(sol.times(), &[0.1, 0.2]);
    assert!((sol.events()[0].time() - 0.3).abs() < 1e-5);
}

#[test]
fn implicit_methods_respect_injected_nonlinear_solver() {
    for method in [
        BuiltinOdeMethod::from(GaussRk::<f64>::new()),
        BuiltinOdeMethod::from(RadauIia::<f64>::new()),
        BuiltinOdeMethod::from(Sdirk::<f64>::new()),
    ] {
        let options = ScenarioConfig::default()
            .with_stage_solver(support::ScenarioStageSolver::Nonlinear(Box::new(
                RejectingNonlinearSolver,
            )))
            .with_initial_step(0.01)
            .unwrap();
        let error = run_solve(
            &ExponentialDecay,
            &method,
            (0.0, 0.1),
            Vector::from_vec(vec![1.0]),
            options,
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}

#[test]
fn invalid_borrowed_nonlinear_solutions_are_typed_and_transactional() {
    for outcome in [
        InvalidNonlinearOutcome::MalformedShape,
        InvalidNonlinearOutcome::NonFinite,
    ] {
        let provider = InvalidNonlinearSolver { outcome };
        let plan = (
            BackwardEuler::<f64>::new(),
            OdeOptions::new().with_initial_step(0.1),
        );
        let problem = OdeProblem::new(&ExponentialDecay, 0.1)
            .with_stage_solver(scioxide_ode::OdeStageSolver::Nonlinear(&provider));
        // A prepared workspace and an empty one grown at preflight behave identically.
        for mut workspace in [
            plan.0.workspace(&problem, &plan.1).unwrap(),
            OdeWorkspace::new(),
        ] {
            let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
            let error = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            match outcome {
                InvalidNonlinearOutcome::MalformedShape => {
                    assert!(matches!(
                        (error).view(),
                        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
                    ));
                }
                InvalidNonlinearOutcome::NonFinite => assert!(matches!(
                    (error).view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Callback,
                        ..
                    }
                )),
            }
            assert_eq!(state.time(), 0.0);
            assert_eq!(state.values(), &[1.0]);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        }
    }
}

#[test]
fn test_implicit_methods_reject_unconverged_injected_linear_results() {
    let solvers = vec![
        BuiltinOdeMethod::from(BackwardEuler::<f64>::new()),
        BuiltinOdeMethod::from(ImplicitTrapezoid::<f64>::new()),
        BuiltinOdeMethod::from(GaussRk::<f64>::new()),
        BuiltinOdeMethod::from(RadauIia::<f64>::new()),
        BuiltinOdeMethod::from(Sdirk::<f64>::new()),
    ];

    for solver in solvers {
        let mut opts = ScenarioConfig::default().with_stage_solver(
            support::ScenarioStageSolver::Linear(Box::new(RejectingLinearSolver)),
        );
        opts = opts.with_initial_step(0.01).unwrap();
        let result = run_solve(
            &ExponentialDecay,
            &solver,
            (0.0, 0.1),
            Vector::from_vec(vec![1.0]),
            opts,
        );

        assert!(
            matches ! (& (result) , Err (error) if if let scioxide_core :: SciOxideErrorView :: ConvergenceFailure { diagnostics , detail , .. } = error . view () { diagnostics . counts ().own().iterations() == 7 && detail == "linear iteration limit" } else { false })
        );
    }
}

#[test]
fn test_invalid_adams_bashforth_order() {
    assert!(matches!(
        (AdamsBashforth::<f64>::new(0)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (AdamsBashforth::<f64>::new(6)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

// ============================================================================
// Runtime solver dispatch via the opaque erased adapter
// ============================================================================

#[test]
fn test_erased_solver_dispatch() {
    let sys = ExponentialDecay;
    let solvers = vec![
        BuiltinOdeMethod::from(ForwardEuler::new()),
        BuiltinOdeMethod::from(DormandPrince5::new()),
        BuiltinOdeMethod::from(ClassicalRk4::new()),
    ];

    for solver in &solvers {
        let options = ScenarioConfig::default().with_initial_step(0.01).unwrap();
        let solution = run_solve(
            &sys,
            solver,
            (0.0, 1.0),
            Vector::from_vec(vec![1.0]),
            options,
        )
        .unwrap();
        let final_value = solution.states().last().unwrap().as_slice()[0];
        assert!((final_value - (-1.0_f64).exp()).abs() < 0.01);
    }
}

#[test]
fn typed_and_erased_routes_return_identical_solutions_and_counters() {
    fn options() -> ScenarioConfig<f64> {
        ScenarioConfig::default()
            .with_tolerance(Tolerance::new(1e-9, 1e-9).unwrap())
            .with_initial_step(0.2)
            .unwrap()
            .with_evaluation_times(vec![0.0, 0.1, 0.2, 0.3])
            .unwrap()
            .with_event(Box::new(TimeEvent))
    }

    let typed_solver = DormandPrince5::new();
    let typed = run_solve(
        &ExponentialDecay,
        &typed_solver,
        (0.0, 1.0),
        Vector::from_vec(vec![1.0]),
        options(),
    )
    .unwrap();
    let erased_solver = BuiltinOdeMethod::from(DormandPrince5::new());
    let erased = run_solve(
        &ExponentialDecay,
        &erased_solver,
        (0.0, 1.0),
        Vector::from_vec(vec![1.0]),
        options(),
    )
    .unwrap();

    assert_eq!(typed.times(), erased.times());
    assert_eq!(typed.states(), erased.states());
    assert_eq!(typed.termination(), erased.termination());
    assert_eq!(typed.message(), erased.message());
    assert_eq!(
        typed.last_step_error_estimate(),
        erased.last_step_error_estimate()
    );
    assert_eq!(typed.stats(), erased.stats());
    assert_eq!(typed.events().len(), erased.events().len());
    typed
        .events()
        .iter()
        .zip(erased.events())
        .for_each(|(typed, erased)| {
            assert_eq!(typed.index(), erased.index());
            assert_eq!(typed.time(), erased.time());
            assert_eq!(typed.state(), erased.state());
            assert_eq!(typed.occurrence(), erased.occurrence());
            assert_eq!(typed.is_terminal(), erased.is_terminal());
        });
}

#[test]
fn typed_and_erased_routes_propagate_identical_callback_errors() {
    let options = || ScenarioConfig::default().with_initial_step(0.1).unwrap();
    let typed_solver = ForwardEuler::new();
    let typed_error = run_solve(
        &FailingSystem,
        &typed_solver,
        (0.0, 1.0),
        Vector::from_vec(vec![1.0]),
        options(),
    )
    .unwrap_err();
    let erased_solver = BuiltinOdeMethod::from(ForwardEuler::new());
    let erased_error = run_solve(
        &FailingSystem,
        &erased_solver,
        (0.0, 1.0),
        Vector::from_vec(vec![1.0]),
        options(),
    )
    .unwrap_err();

    assert_eq!(typed_error.to_string(), erased_error.to_string());
    assert!(matches!(
        (typed_error).view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (erased_error).view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

// ============================================================================
// All solver names accessible from crate root
// ============================================================================

#[test]
fn test_all_solver_names() {
    let names: Vec<&str> = vec![
        BuiltinOdeMethod::from(ForwardEuler::<f64>::new()).name(),
        BuiltinOdeMethod::from(ExplicitMidpoint::<f64>::new()).name(),
        BuiltinOdeMethod::from(Heun::<f64>::new()).name(),
        BuiltinOdeMethod::from(ClassicalRk4::<f64>::new()).name(),
        BuiltinOdeMethod::from(Rkf45::<f64>::new()).name(),
        BuiltinOdeMethod::from(DormandPrince5::<f64>::new()).name(),
        BuiltinOdeMethod::from(DormandPrince853::<f64>::new()).name(),
        BuiltinOdeMethod::from(Verner65::<f64>::new()).name(),
        BuiltinOdeMethod::from(BackwardEuler::<f64>::new()).name(),
        BuiltinOdeMethod::from(ImplicitTrapezoid::<f64>::new()).name(),
        BuiltinOdeMethod::from(GaussRk::<f64>::new()).name(),
        BuiltinOdeMethod::from(RadauIia::<f64>::new()).name(),
        BuiltinOdeMethod::from(Sdirk::<f64>::new()).name(),
        BuiltinOdeMethod::from(AdamsBashforth::<f64>::new(4).unwrap()).name(),
        BuiltinOdeMethod::from(AdamsMoulton::<f64>::new(4).unwrap()).name(),
        BuiltinOdeMethod::from(Bdf::<f64>::new(2).unwrap()).name(),
    ];
    assert_eq!(names.len(), 16);
    for name in &names {
        assert!(!name.is_empty());
    }
}

/// Capabilities observed through plan validation and preflight:
/// `(dense output, events, adaptive control, mass matrices)`.
fn observed_capabilities(method: &BuiltinOdeMethod<f64>) -> (bool, bool, bool, bool) {
    fn unsupported<R>(result: Result<R, SciOxideError>) -> bool {
        match result {
            Ok(_) => false,
            Err(error) => {
                assert!(
                    matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::InvalidInput {
                            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                            ..
                        }
                    ),
                    "{error}"
                );
                true
            }
        }
    }
    let base = || OdeOptions::new().with_initial_step(0.25);
    let dense = !unsupported(method.validate(
        &base().with_output_schedule(OdeOutputSchedule::times(vec![0.5])),
        1,
    ));
    let adaptive = !unsupported(method.validate(
        &base().with_step_control(OdeStepControl::ProportionalIntegral(
            ProportionalIntegralController::new(0.9, 5.0, 0.2, 1.0, 0.0).unwrap(),
        )),
        1,
    ));
    let plan = (method.clone(), base());
    let event = TimeCrossing {
        root: 0.5,
        direction: EventDirection::Either,
        terminal: false,
    };
    let events: [&dyn EventFunction<f64>; 1] = [&event];
    let events = !unsupported(
        plan.0
            .solve(
                &OdeProblem::new(&ZeroDynamics, 1.0).with_events(&events),
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![0.0])),
            )
            .map_err(SciOxideError::from),
    );
    let mass_system = CountingMassSystem {
        rhs_calls: Rc::new(Cell::new(0)),
    };
    let mass = !unsupported(
        plan.0
            .solve(
                &OdeProblem::new(&mass_system, 1.0),
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![0.0])),
            )
            .map_err(SciOxideError::from),
    );
    (dense, events, adaptive, mass)
}

#[test]
fn test_all_solver_capabilities_are_declared() {
    for solver in dense_capable_solvers() {
        let (dense, events, adaptive, mass) = observed_capabilities(&solver);
        assert!(dense, "{}", solver.name());
        assert!(events, "{}", solver.name());
        assert!(adaptive, "{}", solver.name());
        assert_eq!(
            mass,
            matches!(solver.name(), "Bdf" | "RadauIia"),
            "{}",
            solver.name()
        );
    }

    for solver in non_dense_solvers() {
        let (dense, events, adaptive, mass) = observed_capabilities(&solver);
        assert!(!dense, "{}", solver.name());
        assert!(!events, "{}", solver.name());
        assert!(!adaptive, "{}", solver.name());
        let supports_mass_matrix = matches!(
            solver.name(),
            "BackwardEuler" | "ImplicitTrapezoid" | "GaussRk" | "Sdirk" | "Bdf"
        );
        assert_eq!(mass, supports_mass_matrix, "{}", solver.name());
    }
}

#[test]
fn unsupported_mass_matrix_is_preflighted_before_rhs_evaluation() {
    let rhs_calls = Rc::new(Cell::new(0));
    let system = CountingMassSystem {
        rhs_calls: Rc::clone(&rhs_calls),
    };
    let result = run_solve(
        &system,
        &ForwardEuler::new(),
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        ScenarioConfig::default(),
    );
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
    assert_eq!(rhs_calls.get(), 0);

    // Progressive execution leaves the caller state untouched and records the rejection.
    let plan = (
        ForwardEuler::new(),
        OdeOptions::new().with_initial_step(0.1),
    );
    let problem = OdeProblem::new(&system, 1.0);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
    let result = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut (),
        &mut workspace,
        ExecutionContext::sequential(),
    );
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
    assert_eq!(rhs_calls.get(), 0);
    assert_eq!(state.values(), &[0.0]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn test_dense_capable_solvers_support_requested_output_and_events_together() {
    for solver in dense_capable_solvers() {
        let mut options = ScenarioConfig::default();
        options = options.with_initial_step(1.0).unwrap();
        options = options
            .with_evaluation_times(vec![0.0, 0.25, 0.75])
            .unwrap();
        options = options
            .with_events(vec![Box::new(TimeCrossing {
                root: 0.5,
                direction: EventDirection::Either,
                terminal: true,
            })])
            .with_event_localization(
                Tolerance::new(1e-12, 0.0).unwrap(),
                Budget::try_from(128).unwrap(),
            );

        let solution = run_solve(
            &ZeroDynamics,
            &solver,
            (0.0, 1.0),
            Vector::from_vec(vec![0.0]),
            options,
        )
        .unwrap();
        assert_eq!(solution.termination(), Some(OdeTermination::TerminalEvent));
        assert_eq!(solution.events().len(), 1);
        assert!((solution.events()[0].time() - 0.5).abs() < 1e-10);
        assert!(
            solution
                .times()
                .iter()
                .any(|time| (*time - 0.25).abs() < 1e-12)
        );
    }
}

#[test]
fn test_all_non_dense_solvers_reject_requested_output_before_rhs() {
    for solver in non_dense_solvers() {
        let rhs_calls = Rc::new(Cell::new(0));
        let system = CountingSystem {
            rhs_calls: Rc::clone(&rhs_calls),
        };
        let mut options = ScenarioConfig::default();
        options = options.with_initial_step(1.0).unwrap();
        options = options.with_evaluation_times(vec![0.5]).unwrap();

        let result = run_solve(
            &system,
            &solver,
            (0.0, 1.0),
            Vector::from_vec(vec![0.0]),
            options,
        );
        assert!(matches!(
            (result).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
        assert_eq!(rhs_calls.get(), 0, "{} evaluated the RHS", solver.name());
    }
}

#[test]
fn test_all_non_event_solvers_reject_events_before_callbacks() {
    for solver in non_dense_solvers() {
        let rhs_calls = Rc::new(Cell::new(0));
        let event_calls = Rc::new(Cell::new(0));
        let system = CountingSystem {
            rhs_calls: Rc::clone(&rhs_calls),
        };
        let mut options = ScenarioConfig::default();
        options = options.with_initial_step(1.0).unwrap();
        options = options.with_events(vec![Box::new(CountingEvent {
            calls: Rc::clone(&event_calls),
        })]);

        let result = run_solve(
            &system,
            &solver,
            (0.0, 1.0),
            Vector::from_vec(vec![0.0]),
            options,
        );
        assert!(matches!(
            (result).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
        assert_eq!(rhs_calls.get(), 0, "{} evaluated the RHS", solver.name());
        assert_eq!(event_calls.get(), 0, "{} evaluated an event", solver.name());
    }
}

#[test]
fn test_explicit_controller_capability_is_preflighted() {
    let rhs_calls = Rc::new(Cell::new(0));
    let system = CountingSystem {
        rhs_calls: Rc::clone(&rhs_calls),
    };
    let control = || {
        OdeStepControl::ProportionalIntegral(
            ProportionalIntegralController::new(0.9, 5.0, 0.2, 1.0, 0.0).unwrap(),
        )
    };
    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(0.5).unwrap();
    options = options.with_step_control(control());
    let result = run_solve(
        &system,
        &ForwardEuler::new(),
        (0.0, 1.0),
        Vector::from_vec(vec![0.0]),
        options,
    );
    assert!(matches!(
        (result).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
    assert_eq!(rhs_calls.get(), 0);

    let mut options = ScenarioConfig::default();
    options = options.with_initial_step(0.5).unwrap();
    options = options.with_step_control(control());
    let solution = run_solve(
        &ExponentialDecay,
        &Rkf45::new(),
        (0.0, 1.0),
        Vector::from_vec(vec![1.0]),
        options,
    )
    .unwrap();
    assert!(solution.is_success());
    assert!(solution.stats().num_accepted() > 0);
}

#[test]
fn test_requested_output_adds_only_the_dense_stage_evaluation() {
    // One accepted Rkf45 step of the fixed initial size: six stages without dense
    // output, one more RHS evaluation when requested-time output needs the interpolant.
    let rhs_evaluations = |schedule: OdeOutputSchedule<f64>| {
        let plan = (
            Rkf45::<f64>::new(),
            OdeOptions::new()
                .with_initial_step(0.1)
                .with_output_schedule(schedule),
        );
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&ExponentialDecay, 0.1),
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![1.0])),
            )
            .unwrap();
        assert_eq!(solution.report().accepted_steps(), 1);
        assert_eq!(solution.report().rejected_steps(), 0);
        solution
            .report()
            .counts()
            .provider(EvaluationRole::RightHandSide)
    };
    assert_eq!(rhs_evaluations(OdeOutputSchedule::FinalOnly), 6);
    assert_eq!(rhs_evaluations(OdeOutputSchedule::times(vec![0.05])), 7);
}

// ============================================================================
// Statistics tracking
// ============================================================================

#[test]
fn test_solver_stats() {
    let sys = ExponentialDecay;
    let y0 = Vector::from_vec(vec![1.0]);
    let solver = DormandPrince5::<f64>::new();
    let mut opts = ScenarioConfig::default();
    opts = opts.with_initial_step(0.1).unwrap();

    let sol = run_solve(&sys, &solver, (0.0, 1.0), y0, opts).unwrap();

    assert!(sol.stats().num_steps() > 0);
    assert!(sol.stats().num_accepted() > 0);
    assert!(sol.stats().num_rhs_evals() > 0);
}
