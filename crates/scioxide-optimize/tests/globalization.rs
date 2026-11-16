//! Independent numerical and resource tests for globalization.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, GradientTolerance, NumericScalar,
    ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveState, StepTolerance,
};
use scioxide_linalg::{Matrix, MatrixView, MatrixViewMut};
use scioxide_optimize::{
    Bfgs, BfgsOptions, CurvaturePolicy, HessianForm, HessianUpdate, HessianUpdateReport,
    HessianUpdateWorkspace, Lbfgs, LbfgsOptions, LineSearch, NonlinearCg, NonlinearCgOptions,
    ObjectiveFunction, OptimizeTermination, StrongWolfeLineSearch, UpdateSkipReason,
};

#[test]
fn scaled_rosenbrock_reports_local_line_search_exhaustion() {
    let objective = scioxide_optimize::ObjectiveFn::new(2, |x: &[f64]| {
        let a = x[0] / 1e6;
        let b = x[1] / 1e6;
        Ok((1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2))
    })
    .with_gradient(|x: &[f64], output| {
        let a = x[0] / 1e6;
        let b = x[1] / 1e6;
        {
            output.copy_from_slice(&[
                (-2.0 * (1.0 - a) - 400.0 * a * (b - a * a)) / 1e6,
                200.0 * (b - a * a) / 1e6,
            ]);
            Ok(())
        }
    });
    let result = NonlinearCg::default()
        .minimize(&objective, &[-1.2e6, 1e6])
        .unwrap();
    assert!(
        matches!(
            result.termination(),
            OptimizeTermination::LineSearchFailed(_)
        ),
        "{result:?}"
    );
    assert!(result.iterations() < 1000);
    assert!(!result.is_success());
}

struct Quadratic<S> {
    scale: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> ObjectiveFunction<S> for Quadratic<S> {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(self.scale * (x[0] * x[0] + S::try_from_usize_exact(4)? * x[1] * x[1]))
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        out[0] = self.scale * S::TWO * x[0];
        out[1] = self.scale * S::try_from_usize_exact(8)? * x[1];
        Ok(true)
    }
}
fn scaled_solvers<S: RealScalar + NumericScalar<Real = S>>(scales: &[f64]) {
    let cv = |x| S::try_from_f64_lossy(x).unwrap();
    for &scale in scales {
        let objective = Quadratic { scale: cv(scale) };
        let accuracy = cv(200.) * <S as RealScalar>::EPSILON;
        let tol = GradientTolerance::new(cv(scale) * accuracy, cv(0.)).unwrap();
        macro_rules! check {
            ($name:literal, $solver:expr) => {{
                let solver = $solver;
                let result = solver.minimize(&objective, &[cv(1.), cv(-1.)]).unwrap();
                assert!(result.is_success(), "{}: {:?}", $name, result);
                // Analytic gradient evaluated independently of the reported gradient.
                let residual = ((cv(2.) * result.x()[0]) * (cv(2.) * result.x()[0])
                    + (cv(8.) * result.x()[1]) * (cv(8.) * result.x()[1]))
                    .sqrt();
                assert!(residual <= accuracy, "{}: {:?}", $name, result);
                assert_eq!(result.termination(), OptimizeTermination::GradientSmall);
            }};
        }
        check!(
            "BFGS",
            Bfgs::new(BfgsOptions::default().with_gradient_tolerance(tol))
        );
        check!(
            "L-BFGS",
            Lbfgs::new(LbfgsOptions::default().with_gradient_tolerance(tol))
        );
        check!(
            "Nonlinear CG",
            NonlinearCg::new(NonlinearCgOptions::default().with_gradient_tolerance(tol))
        );
    }
}
#[test]
fn scale_equivalent_solvers_f64() {
    scaled_solvers::<f64>(&[1e-200, 1e-150, 1. / 1024., 1., 1024., 1e150, 1e200]);
}
#[test]
fn scale_equivalent_solvers_f32() {
    scaled_solvers::<f32>(&[1e-30, 1e-15, 1. / 1024., 1., 1024., 1e15, 1e30]);
}

#[test]
fn bfgs_matches_exact_rational_matrix_and_inverse() {
    // Exact arithmetic gives B+ = [[219,27],[27,214]]/91 and det(B+) = 39/7.
    // Its independently inverted matrix is [[214,-27],[-27,219]]/507.
    let expected = [219., 27., 27., 214.].map(|x| x / 91.);
    let inverse = [214., -27., -27., 219.].map(|x| x / 507.);
    let mut workspace = HessianUpdateWorkspace::new();
    workspace.try_reserve(2).unwrap();
    for (form, source, expected) in [
        (HessianForm::Hessian, [2., 0., 0., 3.], expected),
        (HessianForm::Inverse, [0.5, 0., 0., 1. / 3.], inverse),
    ] {
        let mut output = [99.; 4];
        workspace
            .update_into(
                MatrixView::from_slice(2, 2, &source).unwrap(),
                MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
                &[1., 2.],
                &[3., 5.],
                form,
                HessianUpdate::Bfgs(CurvaturePolicy::default()),
                ExecutionContext::sequential(),
            )
            .unwrap();
        for (actual, expected) in output.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= 4e-15);
        }
    }
}

#[test]
fn sr1_denominator_safeguard_and_invalid_views_preserve_output() {
    let source = [1., 0., 0., 1.];
    let mut output = [99.; 4];
    let mut workspace = HessianUpdateWorkspace::new();
    workspace.try_reserve(2).unwrap();
    let report = workspace
        .update_into(
            MatrixView::from_slice(2, 2, &source).unwrap(),
            MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
            &[1., 0.],
            &[1. + 1e-12, 1.],
            HessianForm::Hessian,
            HessianUpdate::Sr1,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report,
        HessianUpdateReport::Skipped(UpdateSkipReason::SmallDenominator)
    );
    assert_eq!(output, [99.; 4]);
    let result = workspace.update_into(
        MatrixView::from_slice(2, 2, &source).unwrap(),
        MatrixViewMut::from_slice(2, 2, &mut output).unwrap(),
        &[1.],
        &[2.],
        HessianForm::Hessian,
        HessianUpdate::Sr1,
        ExecutionContext::sequential(),
    );
    assert!(matches!(
        (result)
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
    assert_eq!(output, [99.; 4]);
}

#[test]
fn prepared_updates_do_not_allocate_even_before_first_update() {
    for rule in [
        HessianUpdate::Bfgs(CurvaturePolicy::default()),
        HessianUpdate::Sr1,
    ] {
        let mut workspace = HessianUpdateWorkspace::new();
        workspace.try_reserve(2).unwrap();
        let mut matrix = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap();
        {
            let report = workspace
                .update(
                    &mut matrix,
                    &[1., 0.],
                    &[2., 0.],
                    HessianForm::Hessian,
                    rule,
                )
                .unwrap();
            assert_eq!(report, HessianUpdateReport::Updated);
        };
    }
}

#[test]
fn update_budget_cannot_publish_an_unaccepted_iterate() {
    struct Sphere;
    impl ObjectiveFunction<f64> for Sphere {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            out[0] = 2. * x[0];
            Ok(true)
        }
    }
    let solver = Bfgs::<f64>::default();
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = scioxide_optimize::OptimizeResult::<f64>::empty();
    for limit in [9, 10] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                scioxide_core::SolveIdentity::new(
                    ProblemToken::new(598),
                    scioxide_core::algorithm_id!("test", "parent", 1),
                    1,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(&Sphere, &[1.], &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(
            result
                .counts()
                .own()
                .provider(EvaluationRole::Factorization),
            0,
            "limit {limit}: {result:?}"
        );
        if limit < 10 {
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[1.]);
            assert_eq!(first_order_of(&result).updates(), 0);
        } else {
            assert_eq!(result.termination(), OptimizeTermination::GradientSmall);
            assert_eq!(result.x(), &[0.]);
            assert_eq!(first_order_of(&result).updates(), 1);
        }
    }
}

#[test]
fn step_tolerance_is_not_gradient_convergence() {
    let search =
        StrongWolfeLineSearch::default().with_step_tolerance(StepTolerance::new(2., 0.).unwrap());
    let result = standalone(
        &search,
        &Quadratic { scale: 1. },
        &[1., 1.],
        5.,
        &[-2., -8.],
        -68.,
    );
    assert!(!result.termination().is_success());
}

#[test]
fn sr1_f32_accepts_negative_denominator() {
    let mut matrix = Matrix::from_vec(2, 2, vec![1_f32, 0., 0., 1.]).unwrap();
    let mut workspace = HessianUpdateWorkspace::new();
    workspace.try_reserve(2).unwrap();
    let report = workspace
        .update(
            &mut matrix,
            &[1., 0.],
            &[-2., 0.],
            HessianForm::Hessian,
            HessianUpdate::Sr1,
        )
        .unwrap();
    assert_eq!(report, HessianUpdateReport::Updated);
    assert_eq!(matrix.as_slice(), &[-2., 0., 0., 1.]);
}

#[test]
fn expansion_and_zoom_share_the_line_search_budget() {
    let search = StrongWolfeLineSearch::new(1., f64::EPSILON, 1e-4, 0.9, Budget::ONE).unwrap();
    let result = standalone(
        &search,
        &Quadratic { scale: 1. },
        &[1., 0.],
        1.,
        &[-2., 0.],
        -4.,
    );
    assert_eq!(
        result.termination().failure(),
        Some(scioxide_optimize::LineSearchFailure::IterationLimit)
    );
    assert_eq!(result.fun_evals(), 1);
    assert_eq!(result.grad_evals(), 1);
    assert_eq!(result.rejected_trials(), 1);
}

fn first_order_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::GlobalizationDiagnostics<S> {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::FirstOrder(diagnostics) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}

fn standalone<L: LineSearch<f64>>(
    search: &L,
    f: &impl ObjectiveFunction<f64>,
    x: &[f64],
    f_x: f64,
    direction: &[f64],
    slope: f64,
) -> scioxide_optimize::LineSearchResult<f64> {
    let n = x.len();
    let (mut trial, mut gradient) = (vec![0.0; n], vec![0.0; n]);
    let (mut candidate, mut candidate_gradient) = (vec![0.0; n], vec![0.0; n]);
    let mut derivatives = scioxide_optimize::DerivativeWorkspace::new(
        scioxide_differentiate::DerivativeOperation::Gradient,
        n,
        1,
    )
    .unwrap();
    search
        .search_into(
            f,
            x,
            f_x,
            direction,
            slope,
            scioxide_differentiate::DerivativeStrategy::default(),
            &mut trial,
            &mut gradient,
            &mut candidate,
            &mut candidate_gradient,
            &mut derivatives,
            scioxide_core::ExecutionContext::sequential().into(),
            None,
        )
        .unwrap()
}

#[test]
fn initial_step_hints_preserve_conditions_counts_and_error_transactions() {
    use scioxide_optimize::{DerivativeWorkspace, LineSearch, ObjectiveFunction};
    struct Parabola;
    impl ObjectiveFunction<f64> for Parabola {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, scioxide_core::SciOxideError> {
            Ok(x[0] * x[0])
        }
        fn gradient_into(
            &self,
            x: &[f64],
            output: &mut [f64],
        ) -> Result<bool, scioxide_core::SciOxideError> {
            output[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    fn check(search: impl LineSearch<f64>) {
        let mut derivatives =
            DerivativeWorkspace::new(scioxide_differentiate::DerivativeOperation::Gradient, 1, 1)
                .unwrap();
        for (hint, evaluations) in [(None, 2), (Some(0.5), 1), (Some(100.0), 2)] {
            let (mut point, mut gradient, mut candidate, mut candidate_gradient) =
                ([7.0], [8.0], [0.0], [0.0]);
            let result = search
                .search_into(
                    &Parabola,
                    &[1.0],
                    1.0,
                    &[-2.0],
                    -4.0,
                    scioxide_differentiate::DerivativeStrategy::default(),
                    &mut point,
                    &mut gradient,
                    &mut candidate,
                    &mut candidate_gradient,
                    &mut derivatives,
                    scioxide_core::ExecutionContext::sequential().into(),
                    hint,
                )
                .unwrap();
            assert!(result.is_success());
            assert_eq!(point, [0.0]);
            assert_eq!(result.fun_evals(), evaluations);
        }
        for hint in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (mut point, mut gradient, mut candidate, mut candidate_gradient) =
                ([7.0], [8.0], [0.0], [0.0]);
            assert!(
                search
                    .search_into(
                        &Parabola,
                        &[1.0],
                        1.0,
                        &[-2.0],
                        -4.0,
                        scioxide_differentiate::DerivativeStrategy::default(),
                        &mut point,
                        &mut gradient,
                        &mut candidate,
                        &mut candidate_gradient,
                        &mut derivatives,
                        scioxide_core::ExecutionContext::sequential().into(),
                        Some(hint)
                    )
                    .is_err()
            );
            assert_eq!(point, [7.0]);
            assert_eq!(gradient, [8.0]);
        }
    }
    check(scioxide_optimize::BacktrackingLineSearch::default());
    check(scioxide_optimize::StrongWolfeLineSearch::default());
}
