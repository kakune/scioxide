//! Budget admission, cold initial steps, caller-context budgets and minimum-step retries.
//!
//! Estimator powers, rejection caps and cleanup precedence use crate-private steppers
//! and are covered by the crate's internal driver tests.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, ExecutionContext, NormSelection, NumericScalar,
    ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    DormandPrince5, ForwardEuler, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem,
    OdeSolveReport, OdeState, OdeSystem, OdeTermination, OdeWorkspace,
};

struct Decay;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Decay {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.iter_mut().zip(y).for_each(|(out, y)| *out = -*y);
        Ok(())
    }
}

fn fixed(options: OdeOptions<f64>) -> (ForwardEuler<f64>, OdeOptions<f64>) {
    (ForwardEuler::new(), options.with_initial_step(5e-6))
}
fn initial<S: RealScalar + NumericScalar<Real = S>>() -> Vector<S> {
    Vector::from_vec(vec![S::ZERO, S::ONE])
}

#[test]
fn unbounded_admission_has_no_hidden_step_cap_or_stale_limit() {
    let problem = OdeProblem::new(&Decay, 1.);
    let limited = fixed(OdeOptions::new().with_max_steps(Budget::try_from(10).unwrap()));
    let mut workspace = limited.0.workspace(&problem, &limited.1).unwrap();
    let mut state = OdeState::new(0., initial());
    let report = limited
        .0
        .advance(
            &problem,
            &limited.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.counts().accepted(), 10);
    // The same workspace under an unbounded plan keeps no stale limit.
    let unbounded = fixed(OdeOptions::new());
    state.reset(0., &[0., 1.]).unwrap();
    let report = unbounded
        .0
        .advance(
            &problem,
            &unbounded.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(report.counts().accepted() > 100_000);
    assert_eq!(report.final_time(), 1.);
    assert!((state.values()[1] - (-1_f64).exp()).abs() < 2e-6);
}

/// A child epoch is charged against the caller's shared budgets (evaluations), while
/// iteration and proposal allowances stay local to each epoch.
#[test]
fn caller_context_honors_each_outer_budget_independently() {
    let plan = fixed(OdeOptions::new());
    for kind in [
        BudgetKind::Evaluation,
        BudgetKind::Iteration,
        BudgetKind::Proposal,
    ] {
        let shared = kind == BudgetKind::Evaluation;
        // Twenty fixed steps exceed the outer limit of seven.
        let problem = OdeProblem::new(&Decay, 1e-4);
        let mut workspace = OdeWorkspace::new();
        let mut state = OdeState::new(0., initial());
        let mut outer = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::try_from(7).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(2287),
            scioxide_core::algorithm_id!("test", "outer", 1),
            2,
            0,
        );
        let mut context = outer
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert_eq!(state.time(), report.final_time());
        if shared {
            assert_eq!(
                report.termination(),
                Some(OdeTermination::BudgetLimit(kind))
            );
            assert_eq!(report.counts().accepted(), 7);
            assert!((report.final_time() - 7. * 5e-6).abs() < 1e-18);
        } else {
            assert_eq!(
                report.termination(),
                Some(OdeTermination::ReachedEnd),
                "{kind:?}"
            );
            assert_eq!(report.counts().accepted(), 20);
        }
    }
}

fn relative_initial_step<S: RealScalar + NumericScalar<Real = S>>() {
    let cast = |x| S::try_from_f64_lossy(x).unwrap();
    for explicit in [false, true] {
        let options = OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(S::ZERO, cast(1e-6), NormSelection::RootMeanSquare).unwrap(),
        );
        let options = if explicit {
            options.with_initial_step(cast(1e-3))
        } else {
            options
        };
        let solution = DormandPrince5::new()
            .solve(
                &OdeProblem::new(&Decay, S::ONE),
                &(options),
                OdeState::new(S::ZERO, initial()),
            )
            .unwrap();
        assert_eq!(
            solution.report().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        let values = solution.state().values();
        assert_eq!(values[0], S::ZERO);
        assert!(NumericScalar::abs(values[1] - RealScalar::exp(-S::ONE)) < cast(3e-6));
    }
}
#[test]
fn pure_relative_initial_steps_f64() {
    relative_initial_step::<f64>();
}
#[test]
fn pure_relative_initial_steps_f32() {
    relative_initial_step::<f32>();
}

struct GrowingZero<S>(S);
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for GrowingZero<S> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = self.0;
        out[1] = -y[1];
        Ok(())
    }
}
fn growing_zero_method<
    S: RealScalar + NumericScalar<Real = S>,
    M: scioxide_ode::OdeMethod<S> + Clone + core::fmt::Debug,
>(
    method: M,
) {
    let cast = |x| S::try_from_f64_lossy(x).unwrap();
    for backward in [false, true] {
        let end = if backward { -S::ONE } else { S::ONE };
        let options = OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(S::ZERO, cast(1e-6), NormSelection::RootMeanSquare).unwrap(),
        );
        let solution = method
            .clone()
            .solve(
                &OdeProblem::new(&GrowingZero(S::ONE), end),
                &(options),
                OdeState::new(S::ZERO, initial()),
            )
            .unwrap();
        assert_eq!(
            solution.report().termination(),
            Some(OdeTermination::ReachedEnd),
            "{method:?}"
        );
        let y = solution.state().values();
        assert!(NumericScalar::abs(y[0] - end) < cast(1e-5), "{method:?}");
        assert!(
            NumericScalar::abs(y[1] - RealScalar::exp(-end)) < cast(1e-4),
            "{method:?}"
        );
    }
}
fn growing_zero<S: RealScalar + NumericScalar<Real = S>>() {
    use scioxide_ode::{Bdf, DormandPrince853, MultistepOrder, RadauIia, Rkf45};
    growing_zero_method(DormandPrince5::<S>::new());
    growing_zero_method(DormandPrince853::<S>::new());
    growing_zero_method(Rkf45::<S>::new());
    growing_zero_method(Bdf::<S>::with_order_policy(MultistepOrder::Automatic(5)).unwrap());
    growing_zero_method(RadauIia::<S>::new());
}

#[test]
fn pure_relative_growing_zero_f64() {
    growing_zero::<f64>();
}
#[test]
fn pure_relative_growing_zero_f32() {
    growing_zero::<f32>();
}

#[test]
fn pure_relative_tiny_slope_has_an_ordinary_initial_step() {
    let system = GrowingZero(1e-300);
    let options = OdeOptions::new()
        .with_maximum_step(0.1)
        .with_error_tolerance(
            ErrorTolerance::scalar(0., 1e-6, NormSelection::RootMeanSquare).unwrap(),
        );
    let plan = (DormandPrince5::new(), options);
    let problem = OdeProblem::new(&system, 1.);
    let mut first = None;
    let mut state = OdeState::new(0., Vector::from_vec(vec![0., 0.]));
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut |t: f64, _: &[f64]| {
                if t > 0. && first.is_none() {
                    first = Some(t);
                }
                Ok(())
            },
            &mut plan.0.workspace(&problem, &plan.1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(first.unwrap() >= 1e-6);
    assert!((state.values()[0] / 1e-300 - 1.).abs() < 1e-6);
}

/// `y' = -λy`.
struct Rate(f64);
impl OdeSystem<f64> for Rate {
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
        out[0] = -self.0 * y[0];
        Ok(())
    }
}

/// Dormand–Prince on `y' = -λy, y(0) = 1` over `[0, 1]` from a 0.1 first step; returns
/// the report, the accepted times and the final state.
fn decay_from_a_rejected_step(
    rate: f64,
    minimum: Option<f64>,
) -> (OdeSolveReport<f64>, Vec<f64>, f64) {
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(1e-7, 1e-7, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(0.1)
        .with_output_schedule(OdeOutputSchedule::AcceptedSteps);
    let options = match minimum {
        Some(minimum) => options.with_minimum_step(minimum),
        None => options,
    };
    let plan = (DormandPrince5::new(), options);
    let system = Rate(rate);
    let problem = OdeProblem::new(&system, 1.);
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let mut times = Vec::new();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut |t: f64, _: &[f64]| {
                times.push(t);
                Ok(())
            },
            &mut plan.0.workspace(&problem, &plan.1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    (report, times, state.values()[0])
}

#[test]
fn rejection_below_the_minimum_retries_exactly_at_the_minimum() {
    let rate = 3.5;
    // Without a minimum, the 0.1 step is rejected and retried below 0.05.
    let (free, times, _) = decay_from_a_rejected_step(rate, None);
    assert_eq!(free.termination(), Some(OdeTermination::ReachedEnd));
    assert!(free.rejected_steps() >= 1);
    assert!(times[1] < 0.05);
    // With a 0.05 minimum, the retry runs at exactly 0.05 and is accepted.
    let (report, times, value) = decay_from_a_rejected_step(rate, Some(0.05));
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(report.rejected_steps() >= 1);
    assert_eq!(times[1], 0.05);
    assert_eq!(times.last().copied(), Some(1.));
    let steps: Vec<f64> = times.windows(2).map(|pair| pair[1] - pair[0]).collect();
    assert!(
        steps[..steps.len() - 1]
            .iter()
            .all(|&h| h >= 0.05 * (1. - 1e-12))
    );
    assert!((value - (-rate).exp()).abs() < 1e-6);
}

#[test]
fn rejection_at_the_minimum_ends_with_step_size_limit() {
    // A lone 0.05 step is rejected at this rate too: one retry, then the stop.
    let (report, times, value) = decay_from_a_rejected_step(5., Some(0.05));
    assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
    assert_eq!(report.accepted_steps(), 0);
    assert_eq!(report.rejected_steps(), 2);
    assert_eq!(report.final_time(), 0.);
    assert_eq!(times, [0.]);
    assert_eq!(value, 1.);
}

/// `y' = 1`, admitting only states up to 0.125.
struct Capped;
impl OdeSystem<f64> for Capped {
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
        out[0] = 1.;
        Ok(())
    }
    fn admits(
        &self,
        _: f64,
        state: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok(state[0] <= 0.125)
    }
}

#[test]
fn inadmissible_halving_is_raised_to_the_minimum() {
    // 0.3 and 0.15 are inadmissible; the next half, 0.075, is raised to the 0.125
    // minimum and accepted. The following 0.125 step is inadmissible at the minimum.
    let options = OdeOptions::new()
        .with_initial_step(0.3)
        .with_minimum_step(0.125);
    let plan = (ForwardEuler::new(), options);
    let problem = OdeProblem::new(&Capped, 1.);
    let mut state = OdeState::new(0., Vector::from_vec(vec![0.]));
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut plan.0.workspace(&problem, &plan.1).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(report.rejected_steps(), 3);
    assert_eq!(state.time(), 0.125);
    assert_eq!(state.values(), [0.125]);
}
