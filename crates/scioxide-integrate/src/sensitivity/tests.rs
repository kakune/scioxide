#![allow(clippy::unwrap_used, clippy::too_many_arguments)]

use super::*;
use scioxide_core::{
    AdjointOperator, Budget, ExecutionContext, LinearOperator, NormSelection, NumericScalar,
    ProblemToken, SolveBudgets, SolveIdentity, SolveState,
};
use std::sync::atomic::AtomicUsize as Cell;

fn number<S: RealScalar + Default>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}
fn tolerance<S: RealScalar + Default>() -> ErrorTolerance<S> {
    ErrorTolerance::scalar(
        number::<S>(2e-12).max(number::<S>(1000.0) * S::EPSILON),
        S::ZERO,
        NormSelection::Infinity,
    )
    .unwrap()
}

fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(597),
        scioxide_core::algorithm_id!("integrate", "sensitivity-test", 1),
        0,
        0,
    );
    f(&mut state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap())
}
fn sequential<T>(f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    run(ExecutionContext::sequential(), SolveBudgets::unbounded(), f)
}

struct Polynomial;

#[test]
fn invalid_query_values_are_input_errors_before_the_solve_epoch() {
    for mode in [SensitivityMode::Jvp, SensitivityMode::Vjp] {
        let plan = QuadratureSensitivityPlan::fixed(
            &Polynomial,
            Interval::new(0.0_f64, 1.0).unwrap(),
            GaussKronrodRule::G7K15,
            SensitivityOptions::default().with_mode(mode),
        )
        .unwrap();
        let mut workspace = plan.workspace().unwrap();
        for field in 0..3 {
            let mut parameters = [1.0, 2.0];
            let mut seed = [1.0, 1.0];
            let mut endpoint = [0.0, 0.0];
            match field {
                0 => parameters[0] = f64::NAN,
                1 => seed[0] = f64::INFINITY,
                _ => endpoint[0] = f64::NEG_INFINITY,
            }
            let (mut value, mut derivative) = ([7.0; 2], [8.0; 2]);
            let mut providers = [()];
            let result = match mode {
                SensitivityMode::Jvp => plan.jvp_into(
                    &Polynomial,
                    JvpQuery::new(&parameters, &seed).with_endpoint_tangent(Some(endpoint)),
                    (&mut value, &mut derivative),
                    (&mut workspace, &mut providers),
                    ExecutionContext::sequential_native(),
                ),
                SensitivityMode::Vjp => plan.vjp_into(
                    &Polynomial,
                    VjpQuery::new(&parameters, &seed)
                        .with_endpoint_gradients(Some((&endpoint, &[0.0; 2]))),
                    (&mut value, &mut derivative),
                    (&mut workspace, &mut providers),
                    ExecutionContext::sequential_native(),
                ),
            };
            assert!(matches!(
                result.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
            assert_eq!(value, [7.0; 2]);
            assert_eq!(derivative, [8.0; 2]);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(
                attempt.outcome(),
                scioxide_core::AttemptOutcome::PreflightRejected
            );
            assert!(attempt.epoch().is_none());
            assert_eq!(
                attempt.counts().total().unwrap().total_providers().unwrap(),
                0
            );
        }
    }
}
struct Jacobian<S> {
    x: S,
    p: [S; 2],
}
impl<S: RealScalar + Default + NumericScalar<Real = S>> LinearOperator<S> for Jacobian<S> {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, v: &[S], y: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        y[0] = self.x * self.x * v[0] + v[1];
        y[1] = self.x * (self.p[1] * v[0] + self.p[0] * v[1]);
        Ok::<(), SciOxideError>(())
    }
}
impl<S: RealScalar + Default + NumericScalar<Real = S>> AdjointOperator<S> for Jacobian<S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(&self, w: &[S], y: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        y[0] = self.x * self.x * w[0] + self.p[1] * self.x * w[1];
        y[1] = w[0] + self.p[0] * self.x * w[1];
        Ok::<(), SciOxideError>(())
    }
}
impl<S: RealScalar + Default + NumericScalar<Real = S>> SensitivityIntegrand<S> for Polynomial {
    type Workspace = ();
    fn parameter_dimension(&self) -> usize {
        2
    }
    fn value_dimension(&self) -> usize {
        2
    }
    fn supports(&self, _: SensitivityMode) -> bool {
        true
    }
    fn evaluate(
        &self,
        x: S,
        p: &[S],
        sample: &mut SensitivitySample<'_, S>,
        _: &mut (),
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        sample
            .value_mut()
            .copy_from_slice(&[p[0] * x * x + p[1], p[0] * p[1] * x]);
        sample.apply(&Jacobian { x, p: [p[0], p[1]] }, &mut [])
    }
}
fn polynomial_integral(p: [f64; 2]) -> [f64; 2] {
    let [b, a] = p;
    [
        p[0] * (b.powi(3) - a.powi(3)) / 3.0 + p[1] * (b - a),
        p[0] * p[1] * (b * b - a * a) / 2.0,
    ]
}

#[test]
fn independent_moving_finite_difference() {
    let p = [2.0, 0.5];
    let v = [0.25, -0.1];
    let h = 1e-4;
    let plus = polynomial_integral([p[0] + h * v[0], p[1] + h * v[1]]);
    let minus = polynomial_integral([p[0] - h * v[0], p[1] - h * v[1]]);
    for ((plus, minus), expected) in plus.into_iter().zip(minus).zip([2.73125, 0.409375]) {
        assert!(((plus - minus) / (2.0 * h) - expected).abs() < 2e-8);
    }
}

struct Peak {
    scale: f64,
    center: f64,
}
impl SensitivityIntegrand<f64> for Peak {
    type Workspace = ();
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn value_dimension(&self) -> usize {
        1
    }
    fn supports(&self, _: SensitivityMode) -> bool {
        true
    }
    fn evaluate(
        &self,
        x: f64,
        p: &[f64],
        sample: &mut SensitivitySample<'_, f64>,
        _: &mut (),
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let h = 1.0 / (1.0 + (self.scale * (x - self.center)).powi(2));
        sample.value_mut()[0] = p[0] * h;
        sample.apply(&ScalarOperator(h), &mut [])
    }
}
struct ScalarOperator(f64);
impl LinearOperator<f64> for ScalarOperator {
    fn rows(&self) -> usize {
        1
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = self.0 * x[0];
        Ok::<(), SciOxideError>(())
    }
}
impl AdjointOperator<f64> for ScalarOperator {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, w)
    }
}
fn peak_plan(peak: &Peak, options: SubdivisionOptions<f64>) -> QuadratureSensitivityPlan<f64> {
    QuadratureSensitivityPlan::adaptive(
        peak,
        Interval::new(0.0, 1.0).unwrap(),
        SensitivityOptions::new(options)
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap()
}
#[test]
fn plans_and_workspaces_summarize_their_shape_for_debugging() {
    let peak = Peak {
        scale: 80.0,
        center: 0.31,
    };
    let plan = peak_plan(&peak, SubdivisionOptions::default());
    let plan_text = alloc::format!("{plan:?}");
    assert!(plan_text.starts_with("QuadratureSensitivityPlan") && plan_text.contains("Adaptive"));
    let workspace = alloc::format!("{:?}", plan.workspace().unwrap());
    assert!(workspace.starts_with("QuadratureSensitivityWorkspace"));
    assert!(workspace.contains("captured: false"));
}

#[test]
fn budgets_reserve_complete_primal_derivative_pairs_and_splits() {
    let peak = Peak {
        scale: 80.0,
        center: 0.31,
    };
    let plan = peak_plan(&peak, SubdivisionOptions::default());
    let mut ws = plan.workspace().unwrap();
    let (mut y, mut dy) = (77.0, 88.0);
    for (budget, expected_calls, measured) in [
        (29, 0, false),
        (30, 15, true),
        (89, 15, true),
        (90, 45, true),
    ] {
        let result = run(
            ExecutionContext::sequential(),
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(budget).unwrap()),
            |parent| {
                let result = plan.jvp_into(
                    &peak,
                    JvpQuery::new(&[0.0], &[1.0]),
                    (&mut y, &mut dy),
                    (&mut ws, core::slice::from_mut(&mut ())),
                    &mut *parent,
                );
                let total = parent.counts().total().unwrap();
                assert_eq!(total.provider(EvaluationRole::Function), expected_calls);
                assert_eq!(
                    total.provider(EvaluationRole::LinearOperator),
                    expected_calls
                );
                assert_eq!(parent.counts().own().accepted(), 0);
                result
            },
        );
        assert_eq!(result.is_ok(), measured);
        assert_eq!(ws.value_errors().is_some(), measured);
        if measured {
            assert_eq!(
                result.unwrap().termination(),
                IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
            );
            assert!(plan.partition(&ws).is_ok());
        } else {
            assert_eq!((y, dy), (77.0, 88.0));
            assert!(plan.partition(&ws).is_err());
            assert!(matches!(
                result.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                    ..
                }
            ));
        }
    }
}
#[test]
fn collapsed_interval_keeps_moving_boundary_derivative() {
    let peak = Peak {
        scale: 1.0,
        center: 0.0,
    };
    let plan = QuadratureSensitivityPlan::fixed(
        &peak,
        Interval::new(0.0, 0.0).unwrap(),
        GaussKronrodRule::G7K15,
        SensitivityOptions::default()
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut y = 99.0;
    let mut dy = 99.0;
    let report = sequential(|ctx| {
        plan.jvp_into(
            &peak,
            JvpQuery::new(&[2.0], &[1.0]).with_endpoint_tangent(Some([1.0, 3.0])),
            (&mut y, &mut dy),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx,
        )
    })
    .unwrap();
    assert_eq!((y, dy), (0.0, 4.0));
    assert_eq!(report.function_evaluations, 2);
    assert_eq!(report.derivative_evaluations, 0);
    assert_eq!(report.termination, IntegrationTermination::DegenerateDomain);
}

struct Failing {
    calls: Cell,
    mode: u8,
}
impl SensitivityIntegrand<f64> for Failing {
    type Workspace = ();
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn value_dimension(&self) -> usize {
        1
    }
    fn supports(&self, _: SensitivityMode) -> bool {
        self.mode != 5
    }
    fn evaluate(
        &self,
        _: f64,
        _: &[f64],
        sample: &mut SensitivitySample<'_, f64>,
        _: &mut (),
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        sample.value_mut()[0] = 1.0;
        match self.mode {
            0 => Err(crate::kernel::invalid(
                scioxide_core::argument!("integrand"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "injected primal failure",
            )),
            1 => sample.apply(&ScalarOperator(f64::NAN), &mut []),
            2 => Ok::<(), SciOxideError>(()),
            3 => {
                sample.value_mut()[0] = f64::NAN;
                sample.apply(&ScalarOperator(1.0), &mut [])
            }
            _ => sample.apply(&ScalarOperator(1.0), &mut []),
        }
    }
}
#[test]
fn errors_preserve_outputs_count_attempts_and_allow_reuse() {
    for mode in 0..4 {
        let provider = Failing {
            calls: Cell::new(0),
            mode,
        };
        let plan = QuadratureSensitivityPlan::fixed(
            &provider,
            Interval::new(0.0, 1.0).unwrap(),
            GaussKronrodRule::G7K15,
            SensitivityOptions::default()
                .with_mode(SensitivityMode::Jvp)
                .with_primal_tolerance(tolerance())
                .with_derivative_tolerance(tolerance()),
        )
        .unwrap();
        let mut ws = plan.workspace().unwrap();
        let mut y = 77.0;
        let mut dy = 88.0;
        sequential(|ctx| {
            assert!(
                plan.jvp_into(
                    &provider,
                    JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(None),
                    (&mut y, &mut dy),
                    (&mut ws, core::slice::from_mut(&mut ())),
                    &mut *ctx
                )
                .is_err()
            );
            assert_eq!(
                ctx.counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Function),
                1
            );
            assert_eq!(
                ctx.counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                usize::from(mode == 1)
            );
            assert_eq!(ctx.counts().own().accepted(), 0);
        });
        assert_eq!((y, dy), (77.0, 88.0));
        assert!(plan.partition(&ws).is_err());
        let good = Failing {
            calls: Cell::new(0),
            mode: 4,
        };
        assert!(
            sequential(|ctx| plan.jvp_into(
                &good,
                JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(None),
                (&mut y, &mut dy),
                (&mut ws, core::slice::from_mut(&mut ())),
                &mut *ctx
            ))
            .unwrap()
            .termination()
            .is_success()
        );
        assert!((y - 1.0).abs() < 1e-14 && (dy - 1.0).abs() < 1e-14);
    }
}
#[test]
fn endpoint_budget_and_bad_shapes_fail_before_providers() {
    let p = Failing {
        calls: Cell::new(0),
        mode: 4,
    };
    let plan = QuadratureSensitivityPlan::fixed(
        &p,
        Interval::new(0.0, 1.0).unwrap(),
        GaussKronrodRule::G7K15,
        SensitivityOptions::default()
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut y = 77.0;
    let mut dy = 88.0;
    let error = run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(31).unwrap()),
        |ctx| {
            plan.jvp_into(
                &p,
                JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(Some([1.0, 1.0])),
                (&mut y, &mut dy),
                (&mut ws, core::slice::from_mut(&mut ())),
                &mut *ctx,
            )
        },
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    assert!(ws.value_errors().is_none());
    assert_eq!((y, dy), (77.0, 88.0));
    assert_eq!(p.calls.load(core::sync::atomic::Ordering::Relaxed), 0);
    assert!(
        sequential(|ctx| plan.jvp_into(
            &p,
            JvpQuery::new(&[1.0], &[]).with_endpoint_tangent(None),
            (&mut y, &mut dy),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx
        ))
        .is_err()
    );
    assert_eq!(p.calls.load(core::sync::atomic::Ordering::Relaxed), 0);
    let unavailable = Failing {
        calls: Cell::new(0),
        mode: 5,
    };
    assert!(
        QuadratureSensitivityPlan::fixed(
            &unavailable,
            Interval::new(0.0, 1.0).unwrap(),
            GaussKronrodRule::G7K15,
            SensitivityOptions::default()
                .with_mode(SensitivityMode::Jvp)
                .with_primal_tolerance(tolerance())
                .with_derivative_tolerance(tolerance())
        )
        .is_err()
    );
}

struct LinearMap<'a>(&'a (dyn AdjointOperator<f64> + Sync));
impl SensitivityIntegrand<f64> for LinearMap<'_> {
    type Workspace = Vec<f64>;
    fn parameter_dimension(&self) -> usize {
        self.0.cols()
    }
    fn value_dimension(&self) -> usize {
        self.0.rows()
    }
    fn supports(&self, _: SensitivityMode) -> bool {
        true
    }
    fn evaluate(
        &self,
        x: f64,
        p: &[f64],
        sample: &mut SensitivitySample<'_, f64>,
        scratch: &mut Vec<f64>,
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.apply_into(
            p,
            sample.value_mut(),
            &mut scratch[..self.0.apply_workspace_len()?],
        )?;
        for v in sample.value_mut() {
            *v += x * x;
        }
        sample.apply(self.0, scratch)
    }
}

#[test]
fn zero_parameters_and_empty_values_have_explicit_identities() {
    use scioxide_linalg::Matrix;
    for (m, p, mode) in [
        (0, 3, SensitivityMode::Vjp),
        (2, 0, SensitivityMode::Vjp),
        (2, 0, SensitivityMode::Jvp),
        (0, 0, SensitivityMode::Jvp),
    ] {
        let matrix = Matrix::from_vec(m, p, alloc::vec![]).unwrap();
        let provider = LinearMap(&matrix.view().try_finite().unwrap());
        let plan = QuadratureSensitivityPlan::fixed(
            &provider,
            Interval::new(0.0, 1.0).unwrap(),
            GaussKronrodRule::G7K15,
            SensitivityOptions::default()
                .with_mode(mode)
                .with_primal_tolerance(tolerance())
                .with_derivative_tolerance(tolerance()),
        )
        .unwrap();
        let mut ws = plan.workspace().unwrap();
        let mut scratch = alloc::vec![0.0;m.max(p)];
        let mut y = alloc::vec![99.0;m];
        let mut dy = alloc::vec![99.0;if mode==SensitivityMode::Jvp{m}else{p}];
        let params = alloc::vec![0.0;p];
        let tangent = params.clone();
        let cotangent = alloc::vec![1.0;m];
        let report = sequential(|ctx| {
            if mode == SensitivityMode::Jvp {
                plan.jvp_into(
                    &provider,
                    JvpQuery::new(&params, &tangent).with_endpoint_tangent(None),
                    (&mut y, &mut dy),
                    (&mut ws, core::slice::from_mut(&mut scratch)),
                    &mut *ctx,
                )
            } else {
                plan.vjp_into(
                    &provider,
                    VjpQuery::new(&params, &cotangent).with_endpoint_gradients(None),
                    (&mut y, &mut dy),
                    (&mut ws, core::slice::from_mut(&mut scratch)),
                    &mut *ctx,
                )
            }
        })
        .unwrap();
        assert!(report.function_evaluations() > 0 || report.termination().is_success());
        assert!(dy.iter().all(|&v| v == 0.0));
        assert!(y.iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-14));
        if m == 0 {
            assert_eq!(report.function_evaluations, 0);
        }
    }
}

struct BadExecutor {
    duplicate: bool,
}
impl scioxide_core::Executor for BadExecutor {
    fn worker_budget(&self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        scioxide_core::SequentialExecutor.execute(ctx, plan, &|p, (), scope| {
            if self.duplicate {
                operation(p.clone(), (), scope)?;
                operation(p, (), scope)
            } else {
                Ok::<(), SciOxideError>(())
            }
        })
    }
}
struct CountedPeak<'a> {
    peak: &'a Peak,
    calls: &'a core::sync::atomic::AtomicUsize,
}
impl SensitivityIntegrand<f64> for CountedPeak<'_> {
    type Workspace = ();
    fn parameter_dimension(&self) -> usize {
        self.peak.parameter_dimension()
    }
    fn value_dimension(&self) -> usize {
        self.peak.value_dimension()
    }
    fn supports(&self, mode: SensitivityMode) -> bool {
        self.peak.supports(mode)
    }
    fn evaluate(
        &self,
        x: f64,
        p: &[f64],
        sample: &mut SensitivitySample<'_, f64>,
        workspace: &mut (),
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.calls
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // Peak evaluates exactly one function and applies exactly one scalar
        // operator. Count attempts independently of solver accounting or the
        // execution policy's chosen partition geometry.
        self.peak.evaluate(x, p, sample, workspace, execution)
    }
}
#[test]
fn malformed_executor_rolls_back_and_reconciles_attempts() {
    for duplicate in [false, true] {
        let executor = BadExecutor { duplicate };
        let peak = Peak {
            scale: 1.0,
            center: 0.3,
        };
        let plan = peak_plan(
            &peak,
            SubdivisionOptions::default().with_execution_policy(ExecutionPolicy::Parallel(
                std::num::NonZeroUsize::new(2).unwrap(),
            )),
        );
        let calls = core::sync::atomic::AtomicUsize::new(0);
        let counted = CountedPeak {
            peak: &peak,
            calls: &calls,
        };
        let mut ws = plan.workspace().unwrap();
        let mut y = 77.0;
        let mut dy = 88.0;
        let mut providers = [(); 2];
        run(
            ExecutionContext::new(&executor),
            SolveBudgets::unbounded(),
            |ctx| {
                assert!(
                    plan.jvp_into(
                        &counted,
                        JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(None),
                        (&mut y, &mut dy),
                        (&mut ws, &mut providers),
                        &mut *ctx
                    )
                    .is_err()
                );
                let attempted = calls.load(core::sync::atomic::Ordering::Relaxed);
                assert_eq!(attempted > 0, duplicate);
                assert_eq!(
                    ctx.counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::Function),
                    attempted
                );
                assert_eq!(
                    ctx.counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::LinearOperator),
                    attempted
                );
                assert_eq!(ctx.counts().own().accepted(), 0);
            },
        );
        assert_eq!((y, dy), (77.0, 88.0));
        assert!(plan.partition(&ws).is_err());
    }
}
#[test]
fn changing_adaptive_decisions_does_not_differentiate_control_flow() {
    let peak = Peak {
        scale: 5.0,
        center: 0.31,
    };
    let fixed = QuadratureSensitivityPlan::fixed(
        &peak,
        Interval::new(0.0, 1.0).unwrap(),
        GaussKronrodRule::G7K15,
        SensitivityOptions::default()
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap();
    let mut ws = fixed.workspace().unwrap();
    let mut y = 0.0;
    let mut dy = 0.0;
    let _ = sequential(|ctx| {
        fixed.jvp_into(
            &peak,
            JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(None),
            (&mut y, &mut dy),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx,
        )
    })
    .unwrap();
    let first_error = ws.value_errors().unwrap()[0].total().unwrap();
    assert!(first_error > 0.0);
    let threshold = 1e-6;
    let parameter = threshold / first_error;
    let plan = QuadratureSensitivityPlan::adaptive(
        &peak,
        fixed.interval(),
        SensitivityOptions::new(SubdivisionOptions::default())
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(
                ErrorTolerance::scalar(threshold, 0.0, NormSelection::Infinity).unwrap(),
            )
            .with_derivative_tolerance(
                ErrorTolerance::scalar(1.0, 0.0, NormSelection::Infinity).unwrap(),
            ),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut splits = [0; 2];
    let exact = ((5.0 * (1.0 - 0.31_f64)).atan() + (5.0 * 0.31_f64).atan()) / 5.0;
    for (i, factor) in [1.0 - 1e-4, 1.0 + 1e-4].into_iter().enumerate() {
        let report = sequential(|ctx| {
            plan.jvp_into(
                &peak,
                JvpQuery::new(&[parameter * factor], &[1.0]).with_endpoint_tangent(None),
                (&mut y, &mut dy),
                (&mut ws, core::slice::from_mut(&mut ())),
                &mut *ctx,
            )
        })
        .unwrap();
        splits[i] = report.subdivisions;
        assert!((dy - exact).abs() < 2e-4);
    }
    assert_eq!(splits[0], 0);
    assert!(splits[1] > 0);
}
#[test]
fn endpoint_cancellation_scales_derivative_error_against_corrected_value() {
    // For f=p and b'=-1/p on [0,1], the derivative cancels exactly.
    let provider = Failing {
        calls: Cell::new(0),
        mode: 4,
    };
    let plan = QuadratureSensitivityPlan::adaptive(
        &provider,
        Interval::new(0.0, 1.0).unwrap(),
        SensitivityOptions::new(SubdivisionOptions::default())
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(
                ErrorTolerance::scalar(0.0, 1e-2, NormSelection::Infinity).unwrap(),
            ),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut y = 77.0;
    let mut dy = 88.0;
    let r = sequential(|ctx| {
        plan.jvp_into(
            &provider,
            JvpQuery::new(&[1.0], &[1.0]).with_endpoint_tangent(Some([0.0, -1.0])),
            (&mut y, &mut dy),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx,
        )
    })
    .unwrap();
    assert_eq!(dy, 0.0);
    assert!(r.function_evaluations() > 0 || r.termination().is_success());
    assert!(!r.derivative_tolerance_met);
    assert_eq!(r.termination, IntegrationTermination::Stagnation);
}

fn executor_pullback<S: RealScalar + Default + NumericScalar<Real = S>>() {
    let plan = QuadratureSensitivityPlan::fixed(
        &Polynomial,
        Interval::new(number(0.5), number(2.0)).unwrap(),
        GaussKronrodRule::G25K51,
        SensitivityOptions::default()
            .with_mode(SensitivityMode::Vjp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut value = [number::<S>(0.0); 2];
    let mut gradient = value;
    let p = [number(2.0), number(0.5)];
    let seed = [number(0.7), number(-0.3)];
    let lower = [number(0.0), number(1.0)];
    let upper = [number(1.0), number(0.0)];
    let serial = sequential(|ctx| {
        plan.vjp_into(
            &Polynomial,
            VjpQuery::new(&p, &seed).with_endpoint_gradients(Some((&lower, &upper))),
            (&mut value, &mut gradient),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx,
        )
    })
    .unwrap();
    let expected = (value, gradient);
    let executor = sequential(|ctx| {
        plan.vjp_into(
            &Polynomial,
            VjpQuery::new(&p, &seed).with_endpoint_gradients(Some((&lower, &upper))),
            (&mut value, &mut gradient),
            (&mut ws, &mut [(); 2]),
            &mut *ctx,
        )
    })
    .unwrap();
    assert_eq!((value, gradient), expected);
    assert_eq!(executor.function_evaluations, serial.function_evaluations);
    assert_eq!(
        executor.derivative_evaluations,
        serial.derivative_evaluations
    );
}
#[test]
fn executor_pullback_f32() {
    executor_pullback::<f32>();
}
#[test]
fn executor_pullback_f64() {
    executor_pullback::<f64>();
}

struct ForwardOnly;
impl SensitivityIntegrand<f64> for ForwardOnly {
    type Workspace = ();
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn value_dimension(&self) -> usize {
        1
    }
    fn supports(&self, mode: SensitivityMode) -> bool {
        mode == SensitivityMode::Jvp
    }
    fn evaluate(
        &self,
        x: f64,
        p: &[f64],
        sample: &mut SensitivitySample<'_, f64>,
        _: &mut (),
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        sample.value_mut()[0] = p[0] * x;
        let op = scioxide_core::ClosureOperator::new(1, 1, move |v: &[f64], out: &mut [f64]| {
            out[0] = x * v[0];
            Ok(())
        });
        sample.apply_forward(&op, &mut [0.0])
    }
}
#[test]
fn forward_only_provider_needs_no_adjoint_capability() {
    let plan = QuadratureSensitivityPlan::fixed(
        &ForwardOnly,
        Interval::new(0.0, 1.0).unwrap(),
        GaussKronrodRule::G7K15,
        SensitivityOptions::default()
            .with_mode(SensitivityMode::Jvp)
            .with_primal_tolerance(tolerance())
            .with_derivative_tolerance(tolerance()),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut y = 0.0;
    let mut dy = 0.0;
    let r = sequential(|ctx| {
        plan.jvp_into(
            &ForwardOnly,
            JvpQuery::new(&[3.0], &[1.0]).with_endpoint_tangent(None),
            (&mut y, &mut dy),
            (&mut ws, core::slice::from_mut(&mut ())),
            &mut *ctx,
        )
    })
    .unwrap();
    assert!(r.function_evaluations() > 0 || r.termination().is_success());
    assert!((y - 1.5).abs() < 1e-14 && (dy - 0.5).abs() < 1e-14);
    assert!(
        QuadratureSensitivityPlan::frozen(
            &ForwardOnly,
            &plan.partition(&ws).unwrap(),
            plan.options()
                .clone()
                .with_mode(SensitivityMode::Vjp)
                .with_derivative_tolerance(tolerance())
        )
        .is_err()
    );
}

#[test]
fn constructors_validate_every_initial_geometry_in_both_precisions() {
    fn check<S: RealScalar + Default + NumericScalar<Real = S>>() {
        let interval = Interval::new(S::ONE, S::ONE.next_after(S::TWO)).unwrap();
        for rule in [
            GaussKronrodRule::G7K15,
            GaussKronrodRule::G10K21,
            GaussKronrodRule::G15K31,
            GaussKronrodRule::G20K41,
            GaussKronrodRule::G25K51,
        ] {
            assert!(
                QuadratureSensitivityPlan::fixed(
                    &Polynomial,
                    interval,
                    rule,
                    SensitivityOptions::default()
                )
                .is_err()
            );
        }
        assert!(
            QuadratureSensitivityPlan::adaptive(
                &Polynomial,
                interval,
                SensitivityOptions::default()
            )
            .is_err()
        );
        // The overall support is representable; a later frozen segment is not.
        let support = Interval::new(S::ZERO, interval.upper()).unwrap();
        let partition = SensitivityPartition {
            interval: support,
            rule: GaussKronrodRule::G7K15,
            intervals: alloc::vec![Interval::new(S::ZERO, S::ONE).unwrap(), interval],
        };
        assert!(
            QuadratureSensitivityPlan::frozen(
                &Polynomial,
                &partition,
                SensitivityOptions::default()
            )
            .is_err()
        );
    }
    check::<f32>();
    check::<f64>();
}
