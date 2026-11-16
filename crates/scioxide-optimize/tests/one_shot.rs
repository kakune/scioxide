//! Default forms preserve configured results and reject unsuccessful terminations.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Bounds, Budget, ConvergenceReason, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::*;

macro_rules! default_forms {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            fn bowl(x: &[$s]) -> Result<$s, SciOxideError> {
                Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
            }
            fn same_point(a: &[$s], b: &[$s]) {
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            fn same(a: OptimizeResult<$s>, b: OptimizeResult<$s>) {
                assert_eq!(
                    a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(a.fun().to_bits(), b.fun().to_bits());
                assert_eq!(a.termination(), b.termination());
                assert_eq!(a.counts(), b.counts());
            }
            fn compare(
                actual: Result<OptimizeResult<$s>, SciOxideError>,
                configured: OptimizeResult<$s>,
            ) {
                if configured.is_success() {
                    same(actual.unwrap(), configured);
                } else {
                    assert_eq!(
                        actual.unwrap_err().kind(),
                        configured.require_success().unwrap_err().kind()
                    );
                }
            }
            let objective = ObjectiveFn::new(2, bowl);
            let x0 = [0.0, 0.0];
            compare(
                minimize(bowl, &x0),
                Bfgs::default().minimize(&objective, &x0).unwrap(),
            );
            let bounds = Bounds::new(vec![(Some(-3.0), Some(3.0)); 2]).unwrap();
            let bounded = BoundedObjective::new(&objective, &bounds).unwrap();
            compare(
                minimize_bounded(bowl, &x0, &bounds),
                LbfgsB::default().minimize(&bounded, &x0).unwrap(),
            );
            let constraints = Constraints::new(2)
                .unwrap()
                .equality(ConstraintFn::new(2, |x: &[$s]| Ok(x[0] + x[1] + 1.0)))
                .unwrap();
            let constrained = ConstrainedProblem::new(&objective, &constraints).unwrap();
            compare(
                minimize_constrained(bowl, &x0, &constraints),
                Sqp::default().minimize(&constrained, &x0).unwrap(),
            );
            let global = minimize_global(bowl, &[(-3.0, 3.0); 2], 42).unwrap();
            let configured = DifferentialEvolution::default()
                .minimize(&bounded, &x0)
                .unwrap();
            assert!(matches!(
                global.termination(),
                OptimizeTermination::PopulationStagnation
                    | OptimizeTermination::SearchBudgetCompleted
            ));
            same(global, configured);
            same(
                minimize_global(bowl, &[(-3.0, 3.0); 2], 42).unwrap(),
                minimize_global(bowl, &[(-3.0, 3.0); 2], 42).unwrap(),
            );

            let f = |x: $s| Ok((x - 0.25).powi(2));
            let actual = minimize_scalar(f, -1.0, 2.0).unwrap();
            let configured = ScalarMinimizer::default()
                .minimize(
                    &f,
                    ScalarMinimumStart::Bounded(scioxide_core::Interval::new(-1.0, 2.0).unwrap()),
                )
                .unwrap();
            assert_eq!(actual.x().to_bits(), configured.x().to_bits());
            assert_eq!(actual.fun().to_bits(), configured.fun().to_bits());
            assert_eq!(actual.termination(), configured.termination());
            assert_eq!(actual.counts(), configured.counts());

            let residual = |x: &[$s], r: &mut [$s]| {
                r[0] = x[0] - 1.0;
                r[1] = x[1] + 2.0;
                Ok(())
            };
            let actual = least_squares(residual, &x0, 2);
            let configured = LeastSquares::default()
                .solve(&LeastSquaresFn::new(2, 2, residual).unwrap(), &x0)
                .unwrap();
            if configured.is_success() {
                let actual = actual.unwrap();
                same_point(actual.x(), configured.x());
                assert_eq!(
                    actual.cost().unwrap().to_bits(),
                    configured.cost().unwrap().to_bits()
                );
                assert_eq!(actual.termination(), configured.termination());
                assert_eq!(actual.counts(), configured.counts());
            } else {
                assert_eq!(
                    actual.unwrap_err().kind(),
                    configured.require_success().unwrap_err().kind()
                );
            }
            let data: [$s; 3] = [0.0, 1.0, 2.0];
            let ys: [$s; 3] = [1.0, 3.0, 5.0];
            let fit = curve_fit(|t: $s, p: &[$s]| Ok(p[0] * t + p[1]), &data, &ys, &x0);
            let residual = |p: &[$s], r: &mut [$s]| {
                for ((v, &t), &y) in r.iter_mut().zip(&data).zip(&ys) {
                    *v = p[0] * t + p[1] - y;
                }
                Ok(())
            };
            let configured = LeastSquares::default()
                .solve(&LeastSquaresFn::new(2, 3, residual).unwrap(), &x0)
                .unwrap();
            if configured.is_success() {
                let fit = fit.unwrap();
                same_point(fit.x(), configured.x());
                assert_eq!(
                    fit.cost().unwrap().to_bits(),
                    configured.cost().unwrap().to_bits()
                );
                assert_eq!(fit.counts(), configured.counts());
                assert_eq!(fit.termination(), configured.termination());
            } else {
                assert_eq!(
                    fit.unwrap_err().kind(),
                    configured.require_success().unwrap_err().kind()
                );
            }

            let a = Matrix::from_rows(&[[1.0 as $s, 0.0], [0.0, 1.0]]).unwrap();
            let rhs = [2.0, 2.0];
            let linear = LinearLeastSquaresProblem::from_dense(&a, &rhs).unwrap();
            let box_bounds = Bounds::new(vec![(Some(0.0), Some(1.0)); 2]).unwrap();
            for (actual, configured) in [
                (
                    nnls(&a, &rhs).unwrap(),
                    Nnls::default().solve(&linear).unwrap(),
                ),
                (
                    bvls(&a, &rhs, &box_bounds).unwrap(),
                    BoundedLinearLeastSquares::default()
                        .solve(&linear.with_bounds(&box_bounds).unwrap())
                        .unwrap(),
                ),
            ] {
                same_point(actual.x(), configured.x());
                assert_eq!(
                    actual.objective().to_bits(),
                    configured.objective().to_bits()
                );
                assert_eq!(actual.termination(), configured.termination());
                assert_eq!(actual.counts(), configured.counts());
            }
            let lp = LpProblem::new(vec![-1.0 as $s, -2.0])
                .unwrap()
                .with_constraints(
                    LinearConstraints::less_equal(
                        Matrix::from_rows(&[[1.0 as $s, 1.0], [1.0, 0.0], [0.0, 1.0]]).unwrap(),
                        vec![4.0, 3.0, 2.0],
                    )
                    .unwrap(),
                )
                .unwrap();
            let qp = QpProblem::new(a.clone(), vec![-1.0 as $s, -2.0]).unwrap();
            for (actual, configured) in [
                (
                    lp.solve().unwrap(),
                    SimplexSolver::default().solve(&lp).unwrap(),
                ),
                (qp.solve().unwrap(), QpSolver::default().solve(&qp).unwrap()),
            ] {
                same_point(actual.x(), configured.x());
                assert_eq!(actual.fun().to_bits(), configured.fun().to_bits());
                assert_eq!(actual.termination(), configured.termination());
                assert_eq!(actual.counts(), configured.counts());
            }
            let mip = MipProblem::new(lp, vec![0, 1]).unwrap();
            let actual = mip.solve().unwrap();
            let configured = BranchAndBound::default().solve(&mip).unwrap();
            same_point(actual.x(), configured.x());
            assert_eq!(actual.fun(), configured.fun());
            assert_eq!(actual.termination(), configured.termination());
            assert_eq!(actual.counts(), configured.counts());

            let b = Matrix::from_rows(&[[0.0 as $s, 1.0], [1.0, 0.0]]).unwrap();
            let plan = QapProblemView::new(a.view(), b.view(), &[])
                .unwrap()
                .prepare()
                .unwrap();
            for direction in [QapDirection::Minimize, QapDirection::Maximize] {
                let actual = quadratic_assignment(a.view(), b.view(), direction).unwrap();
                let configured = plan
                    .solve_with_options(direction, None, &QapOptions::default())
                    .unwrap();
                assert_eq!(actual.permutation(), configured.permutation());
                assert_eq!(actual.objective(), configured.objective());
                assert_eq!(actual.termination(), configured.termination());
                assert_eq!(actual.fw_iterations(), configured.fw_iterations());
                assert_eq!(actual.swap_checks(), configured.swap_checks());
            }
        }
    };
}
default_forms!(defaults_f32, f32);
default_forms!(defaults_f64, f64);

#[test]
fn gradient_free_minimizers_pass_the_default_stopping_test() {
    let rosenbrock = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
    });
    let result = minimize(|x: &[f64]| rosenbrock.eval(x), &[-1.2, 1.0]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::GradientSmall);
    assert!(result.x().iter().all(|x| (x - 1.0).abs() < 1e-6));
    for offset in [0.0, 1.0] {
        let result = minimize(
            |x: &[f64]| Ok(offset + (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)),
            &[0.0, 0.0],
        )
        .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::GradientSmall);
        assert!((result.x()[0] - 1.0).abs() < 1e-6);
        assert!((result.x()[1] + 2.0).abs() < 1e-6);
    }
}

#[test]
fn exponential_curve_fit_exact_and_perturbed() {
    let times = [0.0_f64, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0];
    for perturbation in [0.0, 0.001] {
        let samples: Vec<_> = times
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                2.0 * (-1.3 * t).exp()
                    + 0.5
                    + if i % 2 == 0 {
                        perturbation
                    } else {
                        -perturbation
                    }
            })
            .collect();
        let fit = curve_fit(
            |t: f64, p: &[f64]| Ok(p[0] * (-p[1] * t).exp() + p[2]),
            &times,
            &samples,
            &[1.0, 1.0, 0.0],
        )
        .unwrap();
        assert!(
            fit.x()
                .iter()
                .zip([2.0, 1.3, 0.5])
                .all(|(&p, expected)| (p - expected).abs() < 0.005)
        );
        if perturbation > 0.0 {
            assert!(fit.cost().unwrap() > 0.0);
        }
    }
}

fn invalid(
    error: SciOxideError,
    expected: scioxide_core::Argument,
    reason: scioxide_core::InvalidInputReason,
) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason: actual, .. } if argument == expected && actual == reason),
        "{error:?}"
    );
}

#[test]
fn input_errors_and_callback_failure() {
    use scioxide_core::{InvalidInputReason as I, argument};
    let f = |x: &[f64]| Ok(x.iter().map(|v| v * v).sum());
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    let constraints = Constraints::<f64>::new(1).unwrap();
    invalid(minimize(f, &[]).unwrap_err(), argument!("x0"), I::Empty);
    invalid(
        minimize_bounded(f, &[], &bounds).unwrap_err(),
        argument!("x0"),
        I::Empty,
    );
    invalid(
        minimize_constrained(f, &[], &constraints).unwrap_err(),
        argument!("x0"),
        I::Empty,
    );
    assert_eq!(
        minimize(f, &[f64::NAN]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        minimize_bounded(f, &[0.0, 0.0], &bounds)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    invalid(
        minimize_bounded(f, &[2.0], &bounds).unwrap_err(),
        argument!("x0"),
        I::OutOfRange,
    );
    assert_eq!(
        minimize_constrained(f, &[0.0, 0.0], &constraints)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    invalid(
        minimize_global(f, &[], 42).unwrap_err(),
        argument!("bounds"),
        I::Empty,
    );
    invalid(
        minimize_global(f, &[(1.0, 1.0)], 42).unwrap_err(),
        argument!("bounds"),
        I::OutOfRange,
    );
    assert_eq!(
        minimize_global(f, &[(0.0, f64::INFINITY)], 42)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    invalid(
        minimize_scalar(|x: f64| Ok(x * x), 2.0, 1.0).unwrap_err(),
        argument!("lower"),
        I::Inconsistent,
    );
    assert_eq!(
        minimize_scalar(|x: f64| Ok(x * x), 0.0, f64::NAN)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    let r = |_: &[f64], _: &mut [f64]| Ok(());
    invalid(
        least_squares(r, &[], 1).unwrap_err(),
        argument!("x0"),
        I::Empty,
    );
    invalid(
        least_squares(r, &[0.0], 0).unwrap_err(),
        argument!("residuals"),
        I::Empty,
    );
    assert_eq!(
        least_squares(r, &[f64::NAN], 1).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let model = |t: f64, _: &[f64]| Ok(t);
    assert_eq!(
        curve_fit(model, &[1.0], &[], &[0.0]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    invalid(
        curve_fit(model, &[], &[], &[0.0]).unwrap_err(),
        argument!("xdata"),
        I::Empty,
    );
    invalid(
        curve_fit(model, &[1.0], &[1.0], &[]).unwrap_err(),
        argument!("p0"),
        I::Empty,
    );
    for (x, y) in [(f64::NAN, 1.0), (1.0, f64::NAN)] {
        assert_eq!(
            curve_fit(model, &[x], &[y], &[0.0]).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    let a = Matrix::from_rows(&[[1.0_f64, 0.0], [0.0, 1.0]]).unwrap();
    for error in [
        nnls(&a, &[1.0]).unwrap_err(),
        bvls(&a, &[1.0], &bounds).unwrap_err(),
        bvls(&a, &[1.0, 1.0], &bounds).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    }
    assert_eq!(
        nnls(&a, &[f64::NAN, 1.0]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    invalid(
        minimize(
            |_: &[f64]| {
                Err(SciOxideError::invalid_input(
                    argument!("callback"),
                    I::OutOfRange,
                    "injected",
                ))
            },
            &[0.0],
        )
        .unwrap_err(),
        argument!("callback"),
        I::OutOfRange,
    );
    let result = Bfgs::new(BfgsOptions::default().with_max_iterations(Budget::ONE))
        .minimize(
            &ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + 4.0 * x[1] * x[1])),
            &[2.0, 3.0],
        )
        .unwrap();
    assert!(matches!(
        result.require_success().unwrap_err().view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        }
    ));
}

#[test]
fn known_linear_and_quadratic_assignment_answers() {
    let lp = LpProblem::new(vec![-1.0_f64, -2.0])
        .unwrap()
        .with_constraints(
            LinearConstraints::less_equal(
                Matrix::from_rows(&[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]).unwrap(),
                vec![4.0, 3.0, 2.0],
            )
            .unwrap(),
        )
        .unwrap();
    let result = lp.solve().unwrap();
    assert!(result.x().iter().all(|x| (x - 2.0).abs() < 1e-10));
    assert!((result.fun() + 6.0).abs() < 1e-10);
    let a = Matrix::from_rows(&[[0.0_f64, 3.0, 1.0], [8.0, 0.0, 2.0], [4.0, 6.0, 0.0]]).unwrap();
    let b = Matrix::from_rows(&[[0.0, 5.0, 7.0], [2.0, 0.0, 9.0], [3.0, 1.0, 0.0]]).unwrap();
    let result = quadratic_assignment(a.view(), b.view(), QapDirection::Minimize).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Converged);
    assert_eq!(result.permutation(), Some(&[0, 1, 2][..]));
    assert!((result.objective().unwrap() - 74.0).abs() < 1e-10);
}

#[test]
fn matrix_and_model_failures_preserve_their_error_kinds() {
    let square = Matrix::from_rows(&[[1.0_f64, 0.0], [0.0, 1.0]]).unwrap();
    let rectangular = Matrix::from_rows(&[[1.0_f64, 0.0]]).unwrap();
    assert!(
        matches!(quadratic_assignment(square.view(),rectangular.view(),QapDirection::Minimize).unwrap_err().view(),SciOxideErrorView::InvalidInput {argument,..} if argument==scioxide_core::argument!("b"))
    );
    let bad = Matrix::from_rows(&[[f64::NAN, 0.0], [0.0, 1.0]]).unwrap();
    assert_eq!(
        quadratic_assignment(bad.view(), square.view(), QapDirection::Minimize)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        nnls(&bad, &[0.0, 0.0]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0)); 2]).unwrap();
    assert_eq!(
        bvls(&square, &[0.0, f64::NAN], &bounds).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let failure = || {
        SciOxideError::invalid_input(
            scioxide_core::argument!("provider"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "original callback error",
        )
    };
    for error in [
        least_squares(|_: &[f64], _: &mut [f64]| Err(failure()), &[0.0], 1).unwrap_err(),
        curve_fit(|_: f64, _: &[f64]| Err(failure()), &[0.0], &[0.0], &[0.0]).unwrap_err(),
        minimize_global(|_: &[f64]| Err(failure()), &[(-1.0, 1.0)], 42).unwrap_err(),
    ] {
        invalid(
            error,
            scioxide_core::argument!("provider"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        );
    }
}
