//! Analytic coverage of the DOP853 extension and shared acceptance path.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection,
    NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince853, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
    OdeTermination, OdeTrajectory, OdeWorkspace,
};
use std::cell::Cell;

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

struct Polynomial {
    calls: Cell<usize>,
    degree: i32,
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Polynomial {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        t: S,
        _: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = scalar::<S>(f64::from(self.degree)) * t.powi(self.degree - 1);
        Ok(())
    }
}

fn dense_refinement<S: RealScalar + NumericScalar<Real = S>>() {
    // A loose tolerance accepts every single step at its fixed size.
    let tolerance =
        ErrorTolerance::scalar(scalar::<S>(1.0), scalar(1.0), NormSelection::RootMeanSquare)
            .unwrap();
    let mut workspace = OdeWorkspace::new();
    for direction in [-1.0, 1.0] {
        let mut errors = Vec::new();
        for h in [1.0, 0.5, 0.25] {
            let h = scalar::<S>(direction * h);
            let system = Polynomial {
                calls: Cell::new(0),
                degree: 8,
            };
            let zero = scalar::<S>(0.0);
            let thetas = [0.25, 0.5, 0.75];
            let times = thetas
                .iter()
                .map(|&theta| h * scalar::<S>(theta))
                .chain([h]);
            let plan = (
                DormandPrince853::<S>::new(),
                OdeOptions::new()
                    .with_error_tolerance(tolerance.clone())
                    .with_initial_step(NumericScalar::abs(h))
                    .with_output_schedule(OdeOutputSchedule::times(times.collect())),
            );
            let problem = OdeProblem::new(&system, h);
            let mut state = OdeState::new(zero, Vector::from_vec(vec![zero]));
            let mut output = OdeTrajectory::new();
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.accepted_steps(), 1);
            assert_eq!(report.rejected_steps(), 0);
            assert_eq!(report.counts().provider(EvaluationRole::RightHandSide), 16);
            assert_eq!(system.calls.get(), 16);
            assert_eq!(output.states()[3].as_slice(), state.values());
            let endpoint_error = NumericScalar::abs(state.values()[0] - h.powi(8));
            assert!(
                endpoint_error <= scalar::<S>(32.0) * S::EPSILON * NumericScalar::abs(h.powi(8))
            );
            let error = output.times()[..3]
                .iter()
                .zip(output.states())
                .map(|(&t, value)| {
                    NumericScalar::abs(value.as_slice()[0] - t.powi(8))
                        .try_to_f64_exact()
                        .unwrap()
                })
                .fold(0.0_f64, f64::max);
            errors.push(error);
            // Requested samples outside the span or nonfinite are rejected before stepping.
            for invalid in [h + h, S::NAN, S::INFINITY] {
                let rejected = Ok::<_, scioxide_core::SciOxideError>((
                    DormandPrince853::<S>::new(),
                    OdeOptions::new()
                        .with_initial_step(NumericScalar::abs(h))
                        .with_output_schedule(OdeOutputSchedule::times(vec![invalid])),
                ))
                .and_then(|plan| {
                    plan.0.advance(
                        &problem,
                        &plan.1,
                        &mut OdeState::new(zero, Vector::from_vec(vec![zero])),
                        &mut (),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                });
                assert!(rejected.is_err());
            }
            assert_eq!(system.calls.get(), 16);
        }
        for pair in errors.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!(
                (240.0..272.0).contains(&ratio),
                "seventh-order extension must have O(h^8) local error: {errors:?}"
            );
        }
    }
}

struct Oscillator;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Oscillator {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[1];
        out[1] = -y[0];
        Ok(())
    }
}

fn norms_and_epochs<S: RealScalar + NumericScalar<Real = S>>() {
    let h = scalar::<S>(0.25);
    let initial = vec![scalar::<S>(1.0), scalar::<S>(0.0)];
    let mut workspace = OdeWorkspace::new();
    let mut estimates = Vec::new();
    for norm in [
        NormSelection::L1,
        NormSelection::L2,
        NormSelection::Infinity,
        NormSelection::RootMeanSquare,
    ] {
        let tolerance =
            ErrorTolerance::try_vector(&[scalar::<S>(0.1), scalar(0.2)], scalar(0.1), norm)
                .unwrap();
        let options = OdeOptions::new()
            .with_initial_step(h)
            .with_error_tolerance(tolerance);
        let problem = OdeProblem::new(&Oscillator, h);
        let run = |plan: &(DormandPrince853<S>, OdeOptions<S>),
                   output: &mut OdeTrajectory<S>,
                   workspace: &mut OdeWorkspace<S, DormandPrince853<S>>| {
            let mut state = OdeState::from_slice(scalar(0.0), &initial).unwrap();
            plan.0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    output,
                    workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap()
        };
        // Without dense output the single proposal costs 12 stages and the endpoint.
        let proposal = (DormandPrince853::<S>::new(), options.clone());
        let proposal = run(&proposal, &mut OdeTrajectory::new(), &mut workspace);
        assert_eq!(
            proposal.counts().provider(EvaluationRole::RightHandSide),
            13
        );
        let expected = proposal.last_step_error_estimate().unwrap();
        estimates.push(expected.try_to_f64_exact().unwrap());
        let plan = (
            DormandPrince853::<S>::new(),
            options.with_output_schedule(OdeOutputSchedule::times(vec![h * scalar::<S>(0.5), h])),
        );
        for _ in 0..2 {
            let mut output = OdeTrajectory::new();
            let report = run(&plan, &mut output, &mut workspace);
            assert!(report.is_success());
            assert_eq!(report.counts().provider(EvaluationRole::RightHandSide), 16);
            assert_eq!(report.last_step_error_estimate(), Some(expected));
            for (&time, state) in output.times().iter().zip(output.states()) {
                let bound = scalar::<S>(1e-9) + scalar::<S>(64.0) * S::EPSILON;
                assert!(NumericScalar::abs(state.as_slice()[0] - time.cos()) < bound);
                assert!(NumericScalar::abs(state.as_slice()[1] + time.sin()) < bound);
            }
        }
    }
    // Every norm is applied to the same two normalized component errors.
    let [l1, l2, infinity, rms] = estimates[..] else {
        unreachable!()
    };
    let relative = 64.0 * S::EPSILON.try_to_f64_exact().unwrap();
    assert!(infinity <= l2 && l2 <= l1 && l1 <= 2.0 * infinity);
    assert!((rms - l2 / 2.0_f64.sqrt()).abs() <= relative * l2);
    let smaller = l1 - infinity;
    assert!((l2 - infinity.hypot(smaller)).abs() <= relative * l1);
}

#[test]
fn f32_extension_and_acceptance() {
    dense_refinement::<f32>();
    norms_and_epochs::<f32>();
}
#[test]
fn f64_extension_and_acceptance() {
    dense_refinement::<f64>();
    norms_and_epochs::<f64>();
}

#[test]
fn extension_budget_stops_before_each_unadmitted_callback() {
    for limit in 12..16 {
        let system = Polynomial {
            calls: Cell::new(0),
            degree: 1,
        };
        let options = OdeOptions::new()
            .with_initial_step(1.0)
            .with_output_schedule(OdeOutputSchedule::times(vec![0.5, 1.0]))
            .with_max_evaluations(Budget::try_from(limit).unwrap());
        let plan = (DormandPrince853::new(), options);
        let mut output = OdeTrajectory::new();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
        let report = plan
            .0
            .advance(
                &OdeProblem::new(&system, 1.0),
                &plan.1,
                &mut state,
                &mut output,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
        );
        assert_eq!(report.final_time(), 0.0);
        assert_eq!(state.time(), 0.0);
        assert_eq!(state.values(), &[0.0]);
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            limit
        );
        assert_eq!(system.calls.get(), limit);
        assert!(output.times().is_empty());
    }
}

/// `y' = 1` whose derivative is non-finite on one chosen call.
struct FailingCall {
    calls: Cell<usize>,
    fail: usize,
}
impl OdeSystem<f64> for FailingCall {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        out[0] = if self.calls.get() == self.fail {
            f64::NAN
        } else {
            1.0
        };
        Ok(())
    }
}

#[test]
fn extension_stages_follow_acceptance_and_retry_a_failed_stage() {
    // The first attempt uses 13 calls, so call 14 is its first extension stage.
    for fail in [0, 14] {
        let system = FailingCall {
            calls: Cell::new(0),
            fail,
        };
        let options = OdeOptions::new()
            .with_initial_step(1.0)
            .with_output_schedule(OdeOutputSchedule::times(vec![0.25, 0.75, 1.0]));
        let plan = (DormandPrince853::new(), options);
        let mut output = OdeTrajectory::new();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
        let report = plan
            .0
            .advance(
                &OdeProblem::new(&system, 1.0),
                &plan.1,
                &mut state,
                &mut output,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(output.times(), &[0.25, 0.75, 1.0]);
        for (&time, value) in output.times().iter().zip(output.states()) {
            assert!((value.as_slice()[0] - time).abs() <= 4.0 * f64::EPSILON);
        }
        // A failed extension stage rejects the trial; the retry halves the step and
        // reuses the accepted initial derivative.
        let (rejected, accepted, calls) = if fail == 0 {
            (0, 1, 13 + 3)
        } else {
            (1, 2, 14 + 2 * (12 + 3))
        };
        assert_eq!(report.rejected_steps(), rejected);
        assert_eq!(report.accepted_steps(), accepted);
        assert_eq!(system.calls.get(), calls);
        assert_eq!(
            report.counts().provider(EvaluationRole::RightHandSide),
            calls
        );
    }
}

/// `y(t) = t` crosses a quarter.
struct QuarterCrossing;
impl scioxide_ode::EventFunction<f64> for QuarterCrossing {
    fn eval(&self, _: f64, state: &[f64]) -> Result<f64, SciOxideError> {
        Ok(state[0] - 0.25)
    }
    fn direction(&self) -> scioxide_ode::EventDirection {
        scioxide_ode::EventDirection::Rising
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

#[test]
fn event_crossings_complete_the_extension_and_retry_a_failed_stage() {
    for fail in [0, 14] {
        let system = FailingCall {
            calls: Cell::new(0),
            fail,
        };
        let plan = (
            DormandPrince853::new(),
            OdeOptions::new().with_initial_step(1.0),
        );
        let events: [&dyn scioxide_ode::EventFunction<f64>; 1] = [&QuarterCrossing];
        let mut output = OdeTrajectory::new();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![0.0]));
        let report = plan
            .0
            .advance(
                &OdeProblem::new(&system, 1.0).with_events(&events),
                &plan.1,
                &mut state,
                &mut output,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(output.events().len(), 1);
        assert!((output.events()[0].time() - 0.25).abs() < 1e-12);
        // Only the step holding the crossing builds its extension. A failed
        // extension stage rejects the trial; of the two halves only the first
        // crosses.
        let calls = if fail == 0 { 13 + 3 } else { 14 + 12 + 3 + 12 };
        assert_eq!(report.rejected_steps(), usize::from(fail != 0));
        assert_eq!(system.calls.get(), calls);
    }
}

struct OscillatorCrossing;
impl scioxide_ode::EventFunction<f64> for OscillatorCrossing {
    fn eval(&self, _: f64, state: &[f64]) -> Result<f64, SciOxideError> {
        Ok(state[0] - 0.3_f64.cos())
    }
    fn direction(&self) -> scioxide_ode::EventDirection {
        scioxide_ode::EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        true
    }
}

#[test]
fn terminal_event_uses_extension_in_both_directions_and_resets_epoch() {
    let tolerance = ErrorTolerance::scalar(1e-11, 1e-11, NormSelection::RootMeanSquare).unwrap();
    let plan = (
        DormandPrince853::new(),
        OdeOptions::new()
            .with_initial_step(0.5)
            .with_error_tolerance(tolerance),
    );
    let events: [&dyn scioxide_ode::EventFunction<f64>; 1] = [&OscillatorCrossing];
    let mut workspace = OdeWorkspace::new();
    for (start, end) in [(0.0_f64, 1.0), (1.0_f64, 0.0), (0.0_f64, 1.0)] {
        let problem = OdeProblem::new(&Oscillator, end).with_events(&events);
        let mut state = OdeState::new(start, Vector::from_vec(vec![start.cos(), -start.sin()]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut OdeTrajectory::new(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::TerminalEvent));
        assert_eq!(report.terminal_event(), Some(0));
        assert!((report.final_time() - 0.3).abs() < 1e-8);
        assert!((state.values()[1] + 0.3_f64.sin()).abs() < 1e-8);
    }
}

/// `cos t` changes sign at `π/2 + kπ`.
struct CosineCrossing;
impl scioxide_ode::EventFunction<f64> for CosineCrossing {
    fn eval(&self, _: f64, state: &[f64]) -> Result<f64, SciOxideError> {
        Ok(state[0])
    }
    fn direction(&self) -> scioxide_ode::EventDirection {
        scioxide_ode::EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

#[test]
fn extension_stages_run_only_for_steps_whose_interpolant_is_read() {
    let run = |schedule: OdeOutputSchedule<f64>, events: bool| {
        let plan = (
            DormandPrince853::new(),
            OdeOptions::new()
                .with_initial_step(0.1)
                .with_error_tolerance(
                    ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::Infinity).unwrap(),
                )
                .with_output_schedule(schedule),
        );
        let crossing: [&dyn scioxide_ode::EventFunction<f64>; 1] = [&CosineCrossing];
        let problem = OdeProblem::new(&Oscillator, 10.0).with_events(if events {
            &crossing[..]
        } else {
            &[]
        });
        let mut output = OdeTrajectory::new();
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0, 0.0]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut output,
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        let rhs = report.counts().provider(EvaluationRole::RightHandSide);
        (
            report.accepted_steps() + report.rejected_steps(),
            rhs,
            output,
        )
    };
    // Without dense output each attempt costs 12 stages after the initial derivative.
    let (attempts, plain, _) = run(OdeOutputSchedule::FinalOnly, false);
    assert_eq!(plain, 1 + 12 * attempts);
    // An output time at the end of the span reads only the endpoint state.
    assert_eq!(run(OdeOutputSchedule::times(vec![10.0]), false).1, plain);
    // Three interior output times, or three crossings, complete three steps.
    let (_, rhs, output) = run(OdeOutputSchedule::times(vec![1.0, 2.5, 7.75]), false);
    assert_eq!(rhs, plain + 3 * 3);
    for (&time, state) in output.times().iter().zip(output.states()) {
        assert!((state.as_slice()[0] - time.cos()).abs() < 1e-7);
    }
    let (_, rhs, output) = run(OdeOutputSchedule::FinalOnly, true);
    assert_eq!(rhs, plain + 3 * 3);
    assert_eq!(output.events().len(), 3);
    for (k, event) in output.events().iter().enumerate() {
        let root = core::f64::consts::FRAC_PI_2 + core::f64::consts::PI * k as f64;
        assert!((event.time() - root).abs() < 1e-8);
    }
}

#[test]
fn empty_and_resized_workspaces_preserve_shape_and_storage_contracts() {
    struct Empty;
    impl OdeSystem<f64> for Empty {
        fn dimension(&self) -> usize {
            0
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            _: &mut [f64],
            _cx: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    let dense = |step: f64| {
        OdeOptions::new()
            .with_initial_step(step)
            .with_output_schedule(OdeOutputSchedule::times(vec![step / 2.0]))
    };
    let mut workspace = OdeWorkspace::new();
    let empty = (DormandPrince853::<f64>::new(), dense(1.0));
    let mut state = OdeState::new(0.0, Vector::empty());
    let mut output = OdeTrajectory::new();
    let report = empty
        .0
        .advance(
            &OdeProblem::new(&Empty, 1.0),
            &empty.1,
            &mut state,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report.last_step_error_estimate(), Some(0.0));
    assert_eq!(output.times(), [0.5]);
    assert_eq!(output.states()[0].len(), 0);
    // The same workspace grows for a two-component plan.
    let resized = (DormandPrince853::<f64>::new(), dense(0.1));
    let problem = OdeProblem::new(&Oscillator, 0.1);
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0, 0.0]));
    let mut output = OdeTrajectory::new();
    let report = resized
        .0
        .advance(
            &problem,
            &resized.1,
            &mut state,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.states()[0].len(), 2);
    assert_eq!(report.counts().provider(EvaluationRole::RightHandSide), 16);
    assert_eq!(
        workspace
            .check(
                &resized
                    .0
                    .workspace_requirements(&problem, &resized.1)
                    .unwrap()
            )
            .unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    // Unrepresentable stage storage is a typed resource error; the driver's state
    // vectors are reserved before the stage table, so either error may surface first.
    let huge = (DormandPrince853::<f64>::new(), OdeOptions::new());
    let error = huge
        .0
        .workspace(
            &OdeProblem::new(
                &scioxide_ode::OdeFn::new(usize::MAX, |_: f64, _: &[f64], _: &mut [f64]| Ok(()))
                    .unwrap(),
                1.0,
            ),
            &huge.1,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
            | scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
}
