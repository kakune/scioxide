//! Deterministic convergence-order, adaptivity, and conservation invariants.

#![allow(
    clippy::cast_precision_loss,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]

use std::fmt::Debug;

use scioxide_core::{
    ErrorTolerance, ExecutionContext, NormSelection, NumericScalar, RealScalar, SciOxideError,
    Tolerance,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    AdamsBashforth, AdamsMoulton, BackwardEuler, Bdf, BuiltinOdeMethod, ClassicalRk4,
    DormandPrince5, DormandPrince853, ExplicitMidpoint, ForwardEuler, GaussRk, Heun,
    ImplicitTrapezoid, OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeState, OdeSystem,
    OdeTermination, OdeWorkspace, Rkf45, Sdirk, Verner65,
};
mod support;
use support::{ObservedTrajectory, ScenarioConfig, run_solve, run_solve_with_workspace};

trait TestScalar: RealScalar + NumericScalar<Real = Self> + Debug {
    const NAME: &str;
    const STEP_COUNTS: &'static [usize];

    fn scalar_from_f64(value: f64) -> Self {
        Self::try_from_f64_lossy(value).unwrap()
    }

    fn to_f64(self) -> f64 {
        self.try_to_f64_exact().unwrap()
    }
}

impl TestScalar for f64 {
    const NAME: &str = "f64";
    const STEP_COUNTS: &'static [usize] = &[3, 6, 12, 24, 48, 96];
}

impl TestScalar for f32 {
    const NAME: &str = "f32";
    const STEP_COUNTS: &'static [usize] = &[2, 4, 8, 16, 32];
}

trait AnalyticProblem<S: TestScalar>: OdeSystem<S> {
    const NAME: &str;

    fn exact_state(time: S) -> Vector<S>;
}

struct ExponentialDecay;

impl<S: TestScalar> OdeSystem<S> for ExponentialDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: S,
        state: &[S],
        derivative: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: S,
        _state: &[S],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, S>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = jacobian.as_contiguous_slice_mut().ok_or_else(|| {
            scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("jacobian"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "row-major output required",
            )
        })?;
        jacobian_data[0] = -<S as scioxide_core::AlgebraicScalar>::ONE;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

impl<S: TestScalar> AnalyticProblem<S> for ExponentialDecay {
    const NAME: &str = "exponential_decay";

    fn exact_state(time: S) -> Vector<S> {
        Vector::from_vec(vec![RealScalar::exp(-time)])
    }
}

struct LogisticGrowth;

impl<S: TestScalar> OdeSystem<S> for LogisticGrowth {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: S,
        state: &[S],
        derivative: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let value = state[0];
        derivative[0] = value * (<S as scioxide_core::AlgebraicScalar>::ONE - value);
        Ok(())
    }

    fn jacobian(
        &self,
        _time: S,
        state: &[S],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, S>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = jacobian.as_contiguous_slice_mut().ok_or_else(|| {
            scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("jacobian"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "row-major output required",
            )
        })?;
        let two = S::scalar_from_f64(2.0);
        jacobian_data[0] = <S as scioxide_core::AlgebraicScalar>::ONE - two * state[0];
        Ok(Some(()))
    }
}

impl<S: TestScalar> AnalyticProblem<S> for LogisticGrowth {
    const NAME: &str = "logistic_growth";

    fn exact_state(time: S) -> Vector<S> {
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let three = S::scalar_from_f64(3.0);
        Vector::from_vec(vec![one / (one + three * RealScalar::exp(-time))])
    }
}

#[derive(Debug, Clone, Copy)]
struct RefinementSample {
    steps: usize,
    endpoint: f64,
    error: f64,
}

/// End time of `steps` driver steps of `step` from zero, accumulated as the driver does.
fn accumulated_end<S: TestScalar>(step: S, steps: usize) -> S {
    (0..steps).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |time, _| {
        time + step
    })
}

/// Uniform steps of `step`: fixed-step methods always take it, and for adaptive methods
/// a unit tolerance never rejects while the maximum caps every proposal. Linear test
/// problems with exact constant Jacobians make implicit stage solves exact regardless of
/// the stage tolerance.
fn fixed_step_options<S: TestScalar>(step: S) -> OdeOptions<S> {
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(one, one, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(step)
        .with_maximum_step(step)
}

/// Endpoint after `steps` uniform steps over the unit interval.
///
/// The driver never exceeds the maximum step, so where `t + h` rounds up it shortens the
/// step by one ulp and may finish with one additional, rounding-sized step.
fn fixed_step_endpoint<S, Problem, M>(method: M, steps: usize) -> S
where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S> + Default,
    M: OdeMethod<S> + Clone,
{
    let problem = Problem::default();
    let initial = Problem::exact_state(<S as scioxide_core::AlgebraicScalar>::ZERO);
    let step = S::scalar_from_f64(1.0 / steps as f64);
    let plan = (method, fixed_step_options(step));
    let problem = OdeProblem::new(&problem, <S as scioxide_core::AlgebraicScalar>::ONE);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(<S as scioxide_core::AlgebraicScalar>::ZERO, initial);
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
    assert!((steps..=steps + 1).contains(&report.accepted_steps()));
    assert_eq!(report.rejected_steps(), 0);
    state.values()[0]
}

fn erased_endpoint<S, Problem, M, Factory>(factory: &Factory, steps: usize) -> S
where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S> + Default,
    M: Into<BuiltinOdeMethod<S>>,
    Factory: Fn() -> M,
{
    fixed_step_endpoint::<S, Problem, _>(factory().into(), steps)
}

fn typed_workspace_endpoint<S, Problem, M, Factory>(factory: &Factory, steps: usize) -> S
where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S> + Default,
    M: OdeMethod<S> + Clone,
    Factory: Fn() -> M,
{
    fixed_step_endpoint::<S, Problem, _>(factory(), steps)
}

fn refinement_samples<S, Problem>(
    step_counts: &[usize],
    runner: impl Fn(usize) -> S,
) -> Vec<RefinementSample>
where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S>,
{
    let exact =
        Problem::exact_state(<S as scioxide_core::AlgebraicScalar>::ONE).as_slice()[0].to_f64();
    step_counts
        .iter()
        .map(|&steps| {
            let endpoint = runner(steps).to_f64();
            RefinementSample {
                steps,
                endpoint,
                error: (endpoint - exact).abs(),
            }
        })
        .collect()
}

fn assert_observed_order<S, Problem>(
    method: &str,
    path: &str,
    declared_order: usize,
    samples: &[RefinementSample],
) where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S>,
{
    let exact =
        Problem::exact_state(<S as scioxide_core::AlgebraicScalar>::ONE).as_slice()[0].to_f64();
    let roundoff_floor = 8.0 * <S as RealScalar>::EPSILON.to_f64() * exact.abs().max(1.0);
    assert!(
        samples
            .iter()
            .all(|sample| sample.endpoint.is_finite() && sample.error.is_finite()),
        "non-finite refinement evidence: scalar={}, problem={}, method={method}, path={path}, samples={samples:?}",
        S::NAME,
        Problem::NAME,
    );

    let pairs: Vec<_> = samples
        .windows(2)
        .filter(|pair| pair[0].error > roundoff_floor && pair[1].error > roundoff_floor)
        .collect();
    assert!(
        !pairs.is_empty(),
        "no pre-roundoff refinement pair: scalar={}, problem={}, method={method}, path={path}, floor={roundoff_floor:e}, samples={samples:?}",
        S::NAME,
        Problem::NAME
    );
    for pair in pairs {
        let observed_order = (pair[0].error / pair[1].error).log2();
        let required_order = declared_order as f64 - 0.5;
        assert!(
            observed_order >= required_order,
            "declared-order regression: scalar={}, problem={}, method={method}, path={path}, coarse_steps={}, fine_steps={}, observed_order={observed_order:e}, required_order={required_order:e}, samples={samples:?}",
            S::NAME,
            Problem::NAME,
            pair[0].steps,
            pair[1].steps
        );
    }
}

fn assert_problem_refinement<S, Problem, M, Factory>(
    factory: &Factory,
    method: &str,
    declared_order: Option<usize>,
) where
    S: TestScalar + 'static,
    Problem: AnalyticProblem<S> + Default,
    M: OdeMethod<S> + Clone + Into<BuiltinOdeMethod<S>>,
    Factory: Fn() -> M,
{
    let erased = refinement_samples::<S, Problem>(S::STEP_COUNTS, |steps| {
        erased_endpoint::<S, Problem, M, Factory>(factory, steps)
    });
    let typed_workspace = refinement_samples::<S, Problem>(S::STEP_COUNTS, |steps| {
        typed_workspace_endpoint::<S, Problem, M, Factory>(factory, steps)
    });
    if let Some(order) = declared_order {
        assert_observed_order::<S, Problem>(method, "erased", order, &erased);
        assert_observed_order::<S, Problem>(method, "typed_workspace", order, &typed_workspace);
    } else {
        assert!(
            erased
                .iter()
                .chain(&typed_workspace)
                .all(|sample| sample.endpoint.is_finite() && sample.error.is_finite()),
            "non-finite analytic evidence: scalar={}, problem={}, method={method}",
            S::NAME,
            Problem::NAME,
        );
    }

    let epsilon = <S as RealScalar>::EPSILON.to_f64();
    erased
        .iter()
        .zip(&typed_workspace)
        .for_each(|(erased, typed)| {
            let scale = erased.endpoint.abs().max(typed.endpoint.abs()).max(1.0);
            assert!(
                (erased.endpoint - typed.endpoint).abs() <= 32.0 * epsilon * scale,
                "route disagreement: scalar={}, problem={}, method={method}, steps={}, erased={:e}, typed_workspace={:e}",
                S::NAME,
                Problem::NAME,
                erased.steps,
                erased.endpoint,
                typed.endpoint,
            );
        });
}

fn assert_solver_refinement<S, M, Factory>(factory: Factory, declared_order: usize)
where
    S: TestScalar + 'static,
    M: OdeMethod<S> + Clone + Into<BuiltinOdeMethod<S>>,
    Factory: Fn() -> M,
{
    let method = factory().into().name();
    assert_problem_refinement::<S, ExponentialDecay, M, Factory>(
        &factory,
        method,
        Some(declared_order),
    );
    assert_problem_refinement::<S, LogisticGrowth, M, Factory>(&factory, method, None);
}

impl Default for ExponentialDecay {
    fn default() -> Self {
        Self
    }
}

impl Default for LogisticGrowth {
    fn default() -> Self {
        Self
    }
}

#[test]
fn rkf45_f32_meets_its_declared_fifth_order() {
    // Fifth-order errors reach roundoff at four steps; start with a coarse pair.
    let counts = [1, 2, 4, 8];
    let factory = Rkf45::<f32>::new;
    let erased = refinement_samples::<f32, ExponentialDecay>(&counts, |steps| {
        erased_endpoint::<f32, ExponentialDecay, Rkf45<f32>, _>(&factory, steps)
    });
    let typed = refinement_samples::<f32, ExponentialDecay>(&counts, |steps| {
        typed_workspace_endpoint::<f32, ExponentialDecay, Rkf45<f32>, _>(&factory, steps)
    });
    assert_observed_order::<f32, ExponentialDecay>("Rkf45", "erased", 5, &erased);
    assert_observed_order::<f32, ExponentialDecay>("Rkf45", "typed_workspace", 5, &typed);
}

#[test]
fn sdirk_meets_its_declared_third_order() {
    assert_solver_refinement::<f32, _, _>(Sdirk::new, 3);
}

#[test]
fn one_step_orders_through_four_have_f32_refinement_evidence() {
    assert_solver_refinement::<f32, _, _>(ForwardEuler::new, 1);
    assert_solver_refinement::<f32, _, _>(ExplicitMidpoint::new, 2);
    assert_solver_refinement::<f32, _, _>(Heun::new, 2);
    assert_solver_refinement::<f32, _, _>(ClassicalRk4::new, 4);
    assert_solver_refinement::<f32, _, _>(BackwardEuler::new, 1);
    assert_solver_refinement::<f32, _, _>(ImplicitTrapezoid::new, 2);
    assert_solver_refinement::<f32, _, _>(GaussRk::new, 4);
}

#[test]
fn f32_multistep_refinements_agree_across_dispatch_routes() {
    // Nonlinear f64 tests establish formula orders without f32 startup cancellation
    // and roundoff. These grids retain scalar and dispatch-route coverage.
    for order in 1..=4 {
        assert_problem_refinement::<f32, ExponentialDecay, _, _>(
            &|| AdamsBashforth::new(order).unwrap(),
            "AdamsBashforth",
            None,
        );
        assert_problem_refinement::<f32, ExponentialDecay, _, _>(
            &|| AdamsMoulton::new(order).unwrap(),
            "AdamsMoulton",
            None,
        );
        assert_problem_refinement::<f32, ExponentialDecay, _, _>(
            &|| Bdf::new(order).unwrap(),
            "Bdf",
            None,
        );
        assert_problem_refinement::<f32, LogisticGrowth, _, _>(
            &|| AdamsBashforth::new(order).unwrap(),
            "AdamsBashforth",
            None,
        );
        assert_problem_refinement::<f32, LogisticGrowth, _, _>(
            &|| AdamsMoulton::new(order).unwrap(),
            "AdamsMoulton",
            None,
        );
        assert_problem_refinement::<f32, LogisticGrowth, _, _>(
            &|| Bdf::new(order).unwrap(),
            "Bdf",
            None,
        );
    }
}

fn adaptive_options<S: TestScalar>(tolerance: Tolerance<S>) -> ScenarioConfig<S> {
    ScenarioConfig::default()
        .with_tolerance(tolerance)
        .with_initial_step(S::scalar_from_f64(4.0))
        .unwrap()
        .with_max_step(S::scalar_from_f64(4.0))
        .unwrap()
}

fn solve_adaptive_solver_owned<S, M, Factory>(
    factory: &Factory,
    tolerance: S,
) -> ObservedTrajectory<S>
where
    S: TestScalar,
    M: OdeMethod<S> + Clone,
    Factory: Fn(Tolerance<S>) -> M,
{
    let tolerance = Tolerance::new(tolerance, tolerance).unwrap();
    let solver = factory(tolerance);
    run_solve(
        &LogisticGrowth,
        &solver,
        (
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            S::scalar_from_f64(4.0),
        ),
        LogisticGrowth::exact_state(<S as scioxide_core::AlgebraicScalar>::ZERO),
        adaptive_options(tolerance),
    )
    .unwrap()
}

fn solve_adaptive_with_workspace<S, M, Factory>(
    factory: &Factory,
    tolerance: S,
) -> ObservedTrajectory<S>
where
    S: TestScalar,
    M: OdeMethod<S> + Clone,
    Factory: Fn(Tolerance<S>) -> M,
{
    let tolerance = Tolerance::new(tolerance, tolerance).unwrap();
    let solver = factory(tolerance);
    // A caller-owned workspace, used once before so its storage is reused.
    let mut workspace = OdeWorkspace::new();
    run_solve_with_workspace(
        &LogisticGrowth,
        &solver,
        &mut workspace,
        (
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            S::scalar_from_f64(4.0),
        ),
        LogisticGrowth::exact_state(<S as scioxide_core::AlgebraicScalar>::ZERO),
        adaptive_options(tolerance),
    )
    .unwrap();
    run_solve_with_workspace(
        &LogisticGrowth,
        &solver,
        &mut workspace,
        (
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            S::scalar_from_f64(4.0),
        ),
        LogisticGrowth::exact_state(<S as scioxide_core::AlgebraicScalar>::ZERO),
        adaptive_options(tolerance),
    )
    .unwrap()
}

fn adaptive_global_error<S: TestScalar>(
    method: &str,
    path: &str,
    solution: &ObservedTrajectory<S>,
) -> f64 {
    assert_eq!(
        solution.termination(),
        Some(OdeTermination::ReachedEnd),
        "adaptive solve did not reach the end: scalar={}, method={method}, path={path}, solution={solution:?}",
        S::NAME,
    );
    assert!(solution.is_success());
    assert_eq!(
        solution.stats().num_steps(),
        solution.stats().num_accepted() + solution.stats().num_rejected(),
        "inconsistent adaptive attempt counters: scalar={}, method={method}, path={path}",
        S::NAME,
    );
    assert_eq!(
        solution.stats().num_accepted(),
        solution.times().len() - 1,
        "accepted-step/trajectory mismatch: scalar={}, method={method}, path={path}",
        S::NAME,
    );
    assert!(solution.stats().num_accepted() > 0);
    assert!(solution.stats().num_rhs_evals() > 0);
    assert!(solution.times().iter().all(|time| time.is_finite()));
    assert!(
        solution
            .states()
            .iter()
            .flat_map(|state| state.as_slice())
            .all(|value| value.is_finite())
    );
    assert!(
        solution
            .last_step_error_estimate()
            .is_some_and(|value| value.is_finite())
    );

    let endpoint = solution.states().last().unwrap().as_slice()[0];
    let exact = LogisticGrowth::exact_state(S::scalar_from_f64(4.0)).as_slice()[0];
    scioxide_core::NumericScalar::abs(endpoint - exact).to_f64()
}

fn stats_tuple<S: TestScalar>(solution: &ObservedTrajectory<S>) -> [usize; 12] {
    let stats = solution.stats();
    [
        stats.num_steps(),
        stats.num_accepted(),
        stats.num_rejected(),
        stats.num_rhs_evals(),
        stats.num_jac_evals(),
        stats.num_linear_solves(),
        stats.num_factorizations(),
        stats.num_nonlinear_iters(),
        stats.num_jacobian_cache_hits(),
        stats.num_factor_cache_hits(),
        stats.num_jacobian_cache_invalidations(),
        stats.num_factor_cache_invalidations(),
    ]
}

fn assert_adaptive_tolerance_monotonicity<S, M, Factory>(
    factory: Factory,
    loose_tolerance: f64,
    tight_tolerance: f64,
) where
    S: TestScalar,
    M: OdeMethod<S> + Clone + Into<BuiltinOdeMethod<S>>,
    Factory: Fn(Tolerance<S>) -> M,
{
    let method = factory(
        Tolerance::new(
            S::scalar_from_f64(loose_tolerance),
            S::scalar_from_f64(loose_tolerance),
        )
        .unwrap(),
    )
    .into()
    .name();
    let owned_loose =
        solve_adaptive_solver_owned::<S, M, Factory>(&factory, S::scalar_from_f64(loose_tolerance));
    let owned_tight =
        solve_adaptive_solver_owned::<S, M, Factory>(&factory, S::scalar_from_f64(tight_tolerance));
    let workspace_loose = solve_adaptive_with_workspace::<S, M, Factory>(
        &factory,
        S::scalar_from_f64(loose_tolerance),
    );
    let workspace_tight = solve_adaptive_with_workspace::<S, M, Factory>(
        &factory,
        S::scalar_from_f64(tight_tolerance),
    );

    let owned_loose_error = adaptive_global_error(method, "solver_owned_loose", &owned_loose);
    let owned_tight_error = adaptive_global_error(method, "solver_owned_tight", &owned_tight);
    let workspace_loose_error =
        adaptive_global_error(method, "caller_owned_loose", &workspace_loose);
    let workspace_tight_error =
        adaptive_global_error(method, "caller_owned_tight", &workspace_tight);

    assert!(
        owned_tight_error < owned_loose_error,
        "tighter tolerance did not reduce error: scalar={}, method={method}, loose_error={owned_loose_error:e}, tight_error={owned_tight_error:e}",
        S::NAME,
    );
    assert!(workspace_tight_error < workspace_loose_error);
    assert!(owned_tight.stats().num_accepted() >= owned_loose.stats().num_accepted());
    assert!(workspace_tight.stats().num_accepted() >= workspace_loose.stats().num_accepted());
    assert!(owned_tight.stats().num_rejected() > 0);
    assert!(workspace_tight.stats().num_rejected() > 0);
    assert_eq!(stats_tuple(&owned_loose), stats_tuple(&workspace_loose));
    assert_eq!(stats_tuple(&owned_tight), stats_tuple(&workspace_tight));
    assert_eq!(owned_loose.times(), workspace_loose.times());
    assert_eq!(owned_tight.times(), workspace_tight.times());
    owned_loose
        .states()
        .iter()
        .zip(workspace_loose.states())
        .chain(owned_tight.states().iter().zip(workspace_tight.states()))
        .for_each(|(owned, caller)| assert_eq!(owned, caller));
}

#[test]
fn rkf45_adaptive_error_improves_with_tighter_tolerances() {
    assert_adaptive_tolerance_monotonicity::<f64, _, _>(|_tolerance| Rkf45::new(), 1.0e-5, 1.0e-9);
    assert_adaptive_tolerance_monotonicity::<f32, _, _>(|_tolerance| Rkf45::new(), 1.0e-2, 1.0e-4);
}

#[test]
fn dormand_prince5_adaptive_error_improves_with_tighter_tolerances() {
    assert_adaptive_tolerance_monotonicity::<f64, _, _>(
        |_tolerance| DormandPrince5::new(),
        1.0e-5,
        1.0e-9,
    );
}

#[test]
fn dormand_prince853_adaptive_error_improves_with_tighter_tolerances() {
    assert_adaptive_tolerance_monotonicity::<f64, _, _>(
        |_tolerance| DormandPrince853::new(),
        1.0e-5,
        1.0e-9,
    );
}

#[test]
fn verner65_adaptive_error_improves_with_tighter_tolerances() {
    assert_adaptive_tolerance_monotonicity::<f64, _, _>(
        |_tolerance| Verner65::new(),
        1.0e-5,
        1.0e-9,
    );
}

struct HarmonicOscillator;

impl<S: TestScalar> OdeSystem<S> for HarmonicOscillator {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(
        &self,
        _time: S,
        state: &[S],
        derivative: &mut [S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = state[1];
        derivative[1] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: S,
        _state: &[S],
        jacobian: &mut scioxide_linalg::MatrixViewMut<'_, S>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        jacobian
            .as_contiguous_slice_mut()
            .unwrap()
            .copy_from_slice(&[zero, one, -one, zero]);
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct InvariantEvidence {
    final_first: f64,
    final_second: f64,
    max_drift: f64,
}

fn energy<S: TestScalar>(state: &[S]) -> S {
    state.iter().fold(
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        |sum, &value| sum + value * value,
    )
}

/// Integrate the oscillator over `steps` fixed steps, tracking energy at every
/// accepted endpoint.
fn fixed_step_invariant<S, M>(method: M, steps: usize, step_size: f64) -> InvariantEvidence
where
    S: TestScalar + 'static,
    M: OdeMethod<S> + Clone,
{
    let step = S::scalar_from_f64(step_size);
    let plan = (
        method,
        OdeOptions::new()
            .with_initial_step(step)
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    );
    let problem = OdeProblem::new(&HarmonicOscillator, accumulated_end(step, steps));
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        Vector::from_vec(vec![
            <S as scioxide_core::AlgebraicScalar>::ONE,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        ]),
    );
    let mut max_drift = 0.0_f64;
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut |_: S, values: &[S]| {
                assert!(values.iter().all(|value| value.is_finite()));
                let current_energy = energy(values).to_f64();
                assert!(current_energy.is_finite());
                max_drift = max_drift.max((current_energy - 1.0).abs());
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    // Fixed-step methods re-propose the previous evaluated increment, so at large |t|
    // (notably in f32) rounding can change the step count slightly.
    assert!(report.is_success());
    assert_eq!(state.time(), problem.end());
    InvariantEvidence {
        final_first: state.values()[0].to_f64(),
        final_second: state.values()[1].to_f64(),
        max_drift,
    }
}

fn assert_invariant_paths_agree<S, M, Factory>(
    factory: Factory,
    steps: usize,
    step_size: f64,
    drift_bound: f64,
) where
    S: TestScalar + 'static,
    M: OdeMethod<S> + Clone + Into<BuiltinOdeMethod<S>>,
    Factory: Fn() -> M,
{
    let method = factory().into().name();
    let erased = fixed_step_invariant::<S, _>(factory().into(), steps, step_size);
    let typed = fixed_step_invariant::<S, _>(factory(), steps, step_size);
    assert!(
        erased.max_drift <= drift_bound,
        "invariant drift exceeded bound: scalar={}, method={method}, path=erased, evidence={erased:?}, bound={drift_bound:e}",
        S::NAME,
    );
    assert!(typed.max_drift <= drift_bound);
    let epsilon = <S as RealScalar>::EPSILON.to_f64();
    let scale = erased
        .final_first
        .abs()
        .max(erased.final_second.abs())
        .max(1.0);
    assert!((erased.final_first - typed.final_first).abs() <= 32.0 * epsilon * scale);
    assert!((erased.final_second - typed.final_second).abs() <= 32.0 * epsilon * scale);
}

#[test]
fn gauss_rk_preserves_quadratic_energy_over_a_long_horizon() {
    const STEPS: usize = 10_000;
    let f64_bound = 16.0 * (f64::SQRT_EPSILON + STEPS as f64 * f64::EPSILON);
    let f32_bound = 16.0 * (f32::SQRT_EPSILON as f64 + STEPS as f64 * f32::EPSILON as f64);
    assert_invariant_paths_agree::<f64, _, _>(GaussRk::new, STEPS, 0.1, f64_bound);
    assert_invariant_paths_agree::<f32, _, _>(GaussRk::new, STEPS, 0.1, f32_bound);
}

#[test]
fn rk4_harmonic_oscillator_energy_regression_is_retained() {
    assert_invariant_paths_agree::<f64, _, _>(ClassicalRk4::new, 1_000, 0.01, 0.01);
}
