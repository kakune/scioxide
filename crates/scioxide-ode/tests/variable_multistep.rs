//! Independent variable-history, order-control and transaction tests.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use core::cell::RefCell;

use scioxide_core::{
    AttemptOutcome, Budget, Capacity, ErrorTolerance, ExecutionContext, NormSelection,
    NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    AdamsBashforth, AdamsMoulton, Bdf, BdfFormula, EventDirection, EventFunction,
    EventSearchPolicy, MultistepOrder, OdeEventOptions, OdeMethod, OdeOptions, OdeOutputSchedule,
    OdeProblem, OdeState, OdeSystem, OdeTrajectory, OdeWorkspace,
};

struct Decay;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Decay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        f: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = -y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        j: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -S::ONE)?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

/// Derivative-assisted search with two cells queries the dense state and slope at the
/// start, midpoint and end of every accepted step.
fn dense_queries<S: RealScalar>() -> OdeEventOptions<S> {
    OdeEventOptions::new().with_search(EventSearchPolicy::DerivativeAssisted(
        Budget::try_from(2).unwrap(),
    ))
}

/// A never-crossing event that records every dense `(time, state, slope)` query.
struct Probe<S> {
    samples: RefCell<Vec<(S, S, S)>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> EventFunction<S> for Probe<S> {
    fn eval(&self, _: S, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
    fn has_total_derivative(&self) -> bool {
        true
    }
    fn total_derivative(&self, t: S, state: &[S], slope: &[S]) -> Result<S, SciOxideError> {
        self.samples.borrow_mut().push((t, state[0], slope[0]));
        Ok(S::ONE)
    }
    fn direction(&self) -> EventDirection {
        EventDirection::Either
    }
    fn is_terminal(&self) -> bool {
        false
    }
}

/// Accepted step endpoints, dense midpoint samples, final state and step counts.
struct Variable<S> {
    endpoints: Vec<S>,
    samples: Vec<(S, S, S)>,
    state: S,
    accepted: usize,
    rejected: usize,
}

fn integrate_variable<S, M>(method: M, tolerance: f64) -> Variable<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(
                scalar(tolerance),
                scalar(tolerance),
                NormSelection::Infinity,
            )
            .unwrap(),
        )
        .with_initial_step(scalar(0.1))
        .with_output_schedule(OdeOutputSchedule::AcceptedSteps)
        .with_event_options(dense_queries());
    let plan = (method, options);
    let probe = Probe {
        samples: RefCell::new(Vec::new()),
    };
    let events: [&dyn EventFunction<S>; 1] = [&probe];
    let problem = OdeProblem::new(&Decay, S::ONE).with_events(&events);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(S::ZERO, Vector::from_vec(vec![S::ONE]));
    let mut endpoints = Vec::new();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut |t: S, _: &[S]| {
                endpoints.push(t);
                Ok::<(), SciOxideError>(())
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success(), "{:?}", report.termination());
    assert_eq!(state.time(), S::ONE);
    Variable {
        endpoints,
        samples: probe.samples.into_inner(),
        state: state.values()[0],
        accepted: report.accepted_steps(),
        rejected: report.rejected_steps(),
    }
}

fn variable<S, M>(name: &str, factory: impl Fn(MultistepOrder) -> M)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
{
    let tolerance = if S::EPSILON > scalar(1e-10) {
        2e-4
    } else {
        1e-8
    };
    let run = integrate_variable::<S, _>(factory(MultistepOrder::Automatic(5)), tolerance);
    let mut midpoints = 0;
    for &(time, value, slope) in &run.samples {
        assert!(RealScalar::is_finite(value) && RealScalar::is_finite(slope));
        let Some(step) = run
            .endpoints
            .windows(2)
            .find(|pair| pair[0] < time && time < pair[1])
        else {
            continue;
        };
        midpoints += 1;
        let h = step[1] - step[0];
        let exact = (-time).exp();
        assert!(
            (value - exact).abs() < scalar(tolerance * 30.0),
            "{name} t={time:?} value={value:?}"
        );
        // Differentiating history amplifies state/inner-solve uncertainty
        // by 1/h; at startup h can be much smaller than the state tolerance.
        let derivative_scale =
            scalar::<S>(tolerance).sqrt() * scalar(10.0) + scalar::<S>(2.0 * tolerance) / h.abs();
        assert!(
            (slope + exact).abs() < derivative_scale,
            "{name} t={time:?} h={h:?} slope={slope:?} bound={derivative_scale:?}"
        );
    }
    assert_eq!(
        midpoints, run.accepted,
        "{name}: every accepted step is probed"
    );
    let error = (run.state - (-S::ONE).exp()).abs();
    assert!(
        error < scalar(tolerance * 30.0),
        "{name}: error={error:?}, accepted={}, rejected={}",
        run.accepted,
        run.rejected
    );
    assert!(run.rejected > 0, "{name}");
    // Automatic order selection starts at order one; never raising the order would
    // reproduce the first-order step sequence.
    let first_order = integrate_variable::<S, _>(factory(MultistepOrder::Automatic(1)), tolerance);
    assert!(
        2 * run.accepted < first_order.accepted,
        "{name} never increased order: accepted={}, first order={}",
        run.accepted,
        first_order.accepted
    );
}
fn automatic<S: RealScalar + NumericScalar<Real = S>>() {
    variable::<S, _>("AdamsBashforth", |policy| {
        AdamsBashforth::with_order_policy(policy).unwrap()
    });
    variable::<S, _>("AdamsMoulton", |policy| {
        AdamsMoulton::with_order_policy(policy).unwrap()
    });
    variable::<S, _>("Bdf", |policy| Bdf::with_order_policy(policy).unwrap());
    variable::<S, _>("Ndf", |policy| {
        Bdf::with_order_policy(policy)
            .unwrap()
            .with_formula(BdfFormula::Ndf)
    });
}
#[test]
fn automatic_f64() {
    automatic::<f64>();
}
#[test]
fn automatic_f32() {
    automatic::<f32>();
}

struct Stiff;
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Stiff {
    fn dimension(&self) -> usize {
        1
    }
    // Halve the fixed step once mid-history so the variable-step formulas run too.
    fn step_limit(
        &self,
        time: S,
        _: &[S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok((time >= scalar(0.5)).then(|| scalar(0.01)))
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        f: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = -scalar::<S>(1000.0) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        j: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, scalar(-1000.0))?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
fn stiff<S: RealScalar + NumericScalar<Real = S>>() {
    for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
        for order in 1..=5 {
            let plan = (
                Bdf::<S>::new(order).unwrap().with_formula(formula),
                OdeOptions::new()
                    .with_initial_step(scalar(0.02))
                    .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
            );
            let problem = OdeProblem::new(&Stiff, scalar(1.5));
            let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
            let mut state = OdeState::new(S::ZERO, Vector::from_vec(vec![S::ONE]));
            let mut steps = 0;
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut |t: S, y: &[S]| {
                        if t > S::ZERO {
                            assert!(y[0].abs() < S::ONE, "{formula:?} order {order} t={t:?}");
                            steps += 1;
                        }
                        Ok::<(), SciOxideError>(())
                    },
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(steps, report.accepted_steps());
            assert!(steps >= 125, "{formula:?} order {order}: {steps} steps");
            // exp(-1500) is negligible in both supported precisions.
            assert!(
                state.values()[0].abs() < scalar(1e-6),
                "{formula:?} order {order}"
            );
        }
    }
}
#[test]
fn stiff_decay_all_orders_f64() {
    stiff::<f64>();
}
#[test]
fn stiff_decay_all_orders_f32() {
    stiff::<f32>();
}

fn cold_start_orders<S: RealScalar + NumericScalar<Real = S>>(maximum: usize, coarse: usize) {
    for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
        let mut previous = f64::INFINITY;
        for order in 1..=maximum {
            let errors: Vec<_> = [coarse, 2 * coarse, 4 * coarse]
                .into_iter()
                .map(|steps| {
                    let plan = (
                        Bdf::<S>::new(order).unwrap().with_formula(formula),
                        OdeOptions::new().with_initial_step(scalar(1. / steps as f64)),
                    );
                    let solution = plan
                        .0
                        .solve(
                            &OdeProblem::new(&Decay, S::ONE),
                            &plan.1,
                            OdeState::new(S::ZERO, Vector::from_vec(vec![S::ONE])),
                        )
                        .unwrap();
                    assert_eq!(solution.state().time(), S::ONE);
                    NumericScalar::abs(solution.state().values()[0] - RealScalar::exp(-S::ONE))
                        .try_to_f64_exact()
                        .unwrap()
                })
                .collect();
            for pair in errors.windows(2) {
                let observed = (pair[0] / pair[1]).log2();
                assert!(
                    observed >= order as f64 - 0.2,
                    "{formula:?} order={order}, observed={observed}, errors={errors:?}"
                );
            }
            assert!(
                errors[2] < previous,
                "{formula:?} order={order}, errors={errors:?}, previous={previous}"
            );
            previous = errors[2];
        }
    }
}

#[test]
fn cold_start_bdf_ndf_orders_one_through_five_f64() {
    cold_start_orders::<f64>(5, 20);
}
#[test]
fn cold_start_bdf_ndf_orders_one_through_three_f32() {
    cold_start_orders::<f32>(3, 8);
}

struct ComplexDecay;
impl OdeSystem<f64, scioxide_core::Complex<f64>> for ComplexDecay {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[scioxide_core::Complex<f64>],
        f: &mut [scioxide_core::Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = scioxide_core::Complex::new(-1., 1.) * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[scioxide_core::Complex<f64>],
        j: &mut MatrixViewMut<'_, scioxide_core::Complex<f64>>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, scioxide_core::Complex::new(-1., 1.))?;
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn complex_bdf_ndf_reach_third_order_from_cold_start() {
    use scioxide_core::Complex;
    let exact = Complex::new((-1_f64).exp() * 1_f64.cos(), (-1_f64).exp() * 1_f64.sin());
    for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
        let errors: Vec<_> = [20, 40, 80]
            .into_iter()
            .map(|steps| {
                let plan = (
                    Bdf::new(3).unwrap().with_formula(formula),
                    OdeOptions::new().with_initial_step(1. / steps as f64),
                );
                let solution = plan
                    .0
                    .solve(
                        &OdeProblem::new(&ComplexDecay, 1.),
                        &plan.1,
                        OdeState::new(0., Vector::from_vec(vec![Complex::new(1., 0.)])),
                    )
                    .unwrap();
                assert_eq!(solution.state().time(), 1.);
                NumericScalar::abs(solution.state().values()[0] - exact)
            })
            .collect();
        for pair in errors.windows(2) {
            assert!((pair[0] / pair[1]).log2() >= 2.8, "{formula:?}: {errors:?}");
        }
    }
}

#[test]
fn failed_cold_start_retains_state_and_allows_a_fresh_retry() {
    // An undeclared constant Jacobian keeps startup on correction-based Newton.
    let system = scioxide_ode::OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = -y[0];
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: f64, _: &[f64], out: &mut MatrixViewMut<'_, f64>| out.set(0, 0, -1.0));
    let options = OdeOptions::new().with_initial_step(0.1);
    let failing = (
        Bdf::<f64>::new(5).unwrap().with_max_iterations(Budget::ONE),
        options.clone(),
    );
    let problem = OdeProblem::new(&system, 0.1);
    let mut workspace = failing.0.workspace(&problem, &failing.1).unwrap();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    assert!(
        failing
            .0
            .advance(
                &problem,
                &failing.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(state.time(), 0.);
    assert_eq!(state.values(), &[1.]);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().accepted_steps(), 0);
    assert_eq!(attempt.progress().unwrap().state(), &[1.]);
    let plan = (Bdf::<f64>::new(5).unwrap(), options);
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
    assert_eq!(report.accepted_steps(), 1);
    assert!((state.values()[0] - (-0.1_f64).exp()).abs() < 1e-7);
}

#[test]
fn owned_driver_retries_nonlinear_failure_and_obeys_requested_output() {
    let tolerance = ErrorTolerance::scalar(1e-7, 1e-7, NormSelection::Infinity).unwrap();
    // Fixed-point Euler iteration at h=2 is not contractive; the managed
    // automatic method must retry without publishing the failed proposal.
    let solver = AdamsMoulton::with_order_policy(MultistepOrder::Automatic(4)).unwrap();
    let plan = (
        solver,
        OdeOptions::new()
            .with_initial_step(2.0)
            .with_error_tolerance(tolerance)
            .with_output_schedule(OdeOutputSchedule::times(vec![0.0, 0.1, 0.9, 2.0])),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&Decay, 2.0),
            &plan.1,
            OdeState::new(0.0, Vector::from_vec(vec![1.0])),
        )
        .unwrap();
    let report = solution.report();
    assert!(report.is_success());
    assert!(report.counts().rejected() > 0);
    let output: &OdeTrajectory<f64> = solution.trajectory();
    assert_eq!(output.times(), &[0.0, 0.1, 0.9, 2.0]);
    for (&t, y) in output.times().iter().zip(output.states()) {
        assert!((y.as_slice()[0] - (-t).exp()).abs() < 3e-6);
    }
}

fn managed_accuracy<M: OdeMethod<f64> + Clone>(name: &str, factory: impl Fn() -> M) {
    let mut errors = Vec::new();
    let mut accepted = Vec::new();
    for tolerance in [1e-6, 1e-8] {
        let options = OdeOptions::new()
            .with_initial_step(0.1)
            .with_error_tolerance(
                ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap(),
            );
        let plan = (factory(), options);
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&Decay, 1.0),
                &plan.1,
                OdeState::new(0.0, Vector::from_vec(vec![1.0])),
            )
            .unwrap();
        assert!(solution.report().is_success());
        errors.push((solution.state().values()[0] - (-1.0_f64).exp()).abs());
        accepted.push(solution.report().counts().accepted());
    }
    assert!(
        errors[1] < errors[0] && errors[1] < 3e-7,
        "{name} errors={errors:?}, accepted={accepted:?}"
    );
    assert!(accepted[1] > accepted[0]);
}
#[test]
fn managed_pi_controller_tolerance_refinement_all_families() {
    let policy = MultistepOrder::Automatic(5);
    managed_accuracy("AdamsBashforth", || {
        AdamsBashforth::with_order_policy(policy).unwrap()
    });
    managed_accuracy("AdamsMoulton", || {
        AdamsMoulton::with_order_policy(policy).unwrap()
    });
    managed_accuracy("Bdf", || Bdf::with_order_policy(policy).unwrap());
    managed_accuracy("Ndf", || {
        Bdf::with_order_policy(policy)
            .unwrap()
            .with_formula(BdfFormula::Ndf)
    });
}

#[test]
fn failed_workspace_growth_cannot_reuse_partially_resized_history() {
    let plan = (
        AdamsBashforth::<f64>::new(4).unwrap(),
        OdeOptions::new().with_initial_step(0.01),
    );
    let first = OdeProblem::new(&Decay, 0.05);
    let mut workspace = plan.0.workspace(&first, &plan.1).unwrap();
    let mut state = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    let report = plan
        .0
        .advance(
            &first,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    // An impossible reservation fails after it started re-preparing the storage.
    let impossible = (AdamsBashforth::<f64>::new(4).unwrap(), OdeOptions::new());
    assert!(
        workspace
            .try_reserve(
                &impossible
                    .0
                    .workspace_requirements(
                        &OdeProblem::new(
                            &scioxide_ode::OdeFn::new(
                                usize::MAX,
                                |_: f64, _: &[f64], _: &mut [f64]| Ok(())
                            )
                            .unwrap(),
                            1.0
                        ),
                        &impossible.1
                    )
                    .unwrap()
            )
            .is_err()
    );
    let second = OdeProblem::new(&Decay, 0.1);
    assert_eq!(
        workspace
            .check(&plan.0.workspace_requirements(&second, &plan.1).unwrap())
            .unwrap(),
        Capacity::NeedsGrowth
    );
    let mut expected = state.clone();
    let mut fresh = OdeWorkspace::new();
    let fresh_report = plan
        .0
        .advance(
            &second,
            &plan.1,
            &mut expected,
            &mut (),
            &mut fresh,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let report = plan
        .0
        .advance(
            &second,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report, fresh_report);
    assert_eq!(state, expected);
}
