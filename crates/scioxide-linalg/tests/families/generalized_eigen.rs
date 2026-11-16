#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{
    GeneralizedHermitianEigenDecomposition as Eigen,
    GeneralizedHermitianEigenWorkspaceRequest as Request,
    GeneralizedHermitianEigenvaluesBuffer as Values, HermitianEigenOptions, LinalgWorkspace,
    Matrix,
};

#[test]
fn extreme_diagonal_and_empty() {
    for scale in [1e-300_f64, 1., 1e300] {
        let a = Matrix::from_vec(2, 2, vec![2. * scale, 0., 0., 3. * scale]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![scale, 0., 0., scale]).unwrap();
        let result = Eigen::decompose(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
        )
        .unwrap();
        assert!((result.eigenvalues()[0] - 2.).abs() < 1e-13);
        assert!((result.eigenvalues()[1] - 3.).abs() < 1e-13);
        for i in 0..2 {
            assert!(
                (result.eigenvectors().view().get(i, i).unwrap().abs() * scale.sqrt() - 1.).abs()
                    < 1e-13
            );
        }
    }
    let empty = Matrix::<f64>::zeros(0, 0).unwrap();
    assert!(
        Eigen::decompose(
            &empty,
            &empty,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact
        )
        .unwrap()
        .eigenvalues()
        .is_empty()
    );
}

fn analytic_pair<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let c = |v| R::try_from_f64_lossy(v).unwrap();
    // A unitary diagonal phase turns the exact real pair into a genuinely
    // complex pair without changing its determinant polynomial 2*l²-9*l+8.
    let z = |r, i| Complex::new(c(r), c(i));
    let a = Matrix::from_vec(2, 2, vec![z(3., 0.), z(0., -1.), z(0., 1.), z(3., 0.)]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![z(2., 0.), z(0., 0.), z(0., 0.), z(1., 0.)]).unwrap();
    let result = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let exact = [
        (c(9.) - c(17.).sqrt()) / c(4.),
        (c(9.) + c(17.).sqrt()) / c(4.),
    ];
    let tol = c(100.) * <R as RealScalar>::EPSILON;
    for (&actual, &expected) in result.eigenvalues().iter().zip(&exact) {
        assert!((actual - expected).abs() < tol);
    }
    for j in 0..2 {
        for k in 0..2 {
            let dot = result.eigenvectors().view().get(0, j).unwrap().conj()
                * result.eigenvectors().view().get(0, k).unwrap()
                * z(2., 0.)
                + result.eigenvectors().view().get(1, j).unwrap().conj()
                    * result.eigenvectors().view().get(1, k).unwrap();
            assert!((dot - if j == k { z(1., 0.) } else { z(0., 0.) }).norm() < tol);
        }
    }
}
#[test]
fn analytic_complex32() {
    analytic_pair::<f32>();
}
#[test]
fn analytic_complex64() {
    analytic_pair::<f64>();
}

#[test]
fn prepared_errors_views_and_budgets() {
    use scioxide_core::{
        AttemptOutcome, Budget, BudgetKind, CancellationToken, ExecutionContext, ProblemToken,
        SolveBudgets, SolveIdentity, SolveState,
    };
    use scioxide_linalg::{EigenIndexRange, MatrixView};
    let a = Matrix::<f64>::from_vec(3, 3, vec![6., 2., 1., 2., 5., 2., 1., 2., 4.]).unwrap();
    let b = Matrix::identity(3).unwrap();
    let options = HermitianEigenOptions::default();
    let mut result = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let before = result.eigenvalues().to_vec();
    let mut work = LinalgWorkspace::new();
    // A parent's one-iteration allowance no longer bounds the child epoch.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(474),
        scioxide_core::algorithm_id!("linalg", "generalized", 1),
        3,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut result,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    let used = context.counts().nested().iterations();
    assert!(used > 1);
    let attempt = work.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), used);
    // The options' budget bounds the work and preserves the output.
    assert!(
        Eigen::decompose_into(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_iteration_budget(Budget::ONE)
            ),
            &mut result,
            &mut work,
            &mut context
        )
        .is_err()
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert_eq!(context.counts().nested().iterations(), used + 1);
    assert_eq!(result.eigenvalues(), before);
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        Eigen::decompose_into(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &options,
            &mut result,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(result.eigenvalues(), before);
    let selected = options.with_index_range(EigenIndexRange::new(0, 1).unwrap());
    assert!(
        Eigen::decompose_with_options(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &selected
        )
        .is_err()
    );
    assert!(
        scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute_with_options(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &selected
        )
        .is_err()
    );
    Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut result,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let data = [3.0_f64, 1., 99., 1., 3., 99.];
    let view = MatrixView::from_slice_with_strides(2, 2, &data, 3, 1).unwrap();
    let metric = Matrix::from_vec(2, 2, vec![2., 0., 0., 1.]).unwrap();
    let result = Eigen::decompose(
        view,
        &metric,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    assert!((result.eigenvalues()[0] - (9. - 17.0_f64.sqrt()) / 4.).abs() < 1e-13);
    for data in [
        vec![1., 0., 0., 0.],
        vec![-1., 0., 0., 1.],
        vec![1., f64::NAN, f64::NAN, 1.],
        vec![1., 1., 0., 1.],
    ] {
        let bad = Matrix::from_vec(2, 2, data).unwrap();
        assert!(
            Eigen::decompose(
                view,
                &bad,
                scioxide_linalg::HermitianInput::Exact,
                scioxide_linalg::HermitianInput::Exact
            )
            .is_err()
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_equivalence() {
    let n = 40;
    let a = Matrix::<f64>::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    40. + (i / n) as f64
                } else {
                    0.1
                }
            })
            .collect(),
    )
    .unwrap();
    let b = Matrix::identity(n).unwrap();
    let options = HermitianEigenOptions::default();
    let expected = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let mut actual = Eigen::empty();
    let mut work = LinalgWorkspace::new();
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut actual,
        &mut work,
        scioxide_core::ExecutionContext::new(&pool).with_processor(&scioxide_core::StdProcessor),
    )
    .unwrap();
    assert_eq!(actual.eigenvalues(), expected.eigenvalues());
}

#[test]
fn clustered_and_subnormal_pairs() {
    let tiny = f64::from_bits(1);
    let a = Matrix::<f64>::from_vec(2, 2, vec![2. * tiny, 0., 0., 3. * tiny]).unwrap();
    let b = Matrix::<f64>::from_vec(2, 2, vec![tiny, 0., 0., tiny]).unwrap();
    let r = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    assert!((r.eigenvalues()[0] - 2.0_f64).abs() < 1e-13);
    assert!((r.eigenvalues()[1] - 3.).abs() < 1e-13);
    let b = Matrix::<f64>::from_vec(3, 3, vec![4., 1., 0., 1., 3., 1., 0., 1., 2.]).unwrap();
    let a = Matrix::<f64>::from_vec(3, 3, b.as_slice().iter().map(|x| 3. * x).collect()).unwrap();
    let r = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    assert!(r.eigenvalues().iter().all(|x| (*x - 3.0_f64).abs() < 1e-13));
}

fn pair(n: usize) -> (Matrix<f64>, Matrix<f64>) {
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                if row == col {
                    4.0 + row as f64
                } else {
                    1.0 / (1 + row + col) as f64
                }
            })
            .collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                if row == col {
                    2.0
                } else if row.abs_diff(col) == 1 {
                    0.5
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    (a, b)
}

#[test]
fn generalized_requirements_report_limits_overflow_and_kind() {
    use scioxide_core::SciOxideErrorKind;
    let options = HermitianEigenOptions::<f64>::default();
    let (a, b) = pair(4);
    let limited = options.with_execution_options((options).execution_options().with_byte_limit(1));
    assert_eq!(
        Eigen::<f64>::workspace_requirements(&Request::new(4, &limited))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut eigen = Eigen::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let before = eigen.clone();
    let error = Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &limited,
        &mut eigen,
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(eigen.eigenvalues(), before.eigenvalues());
    assert_eq!(eigen.eigenvectors(), before.eigenvectors());
    assert_eq!(
        Values::<f64>::workspace_requirements::<f64>(&Request::new(usize::MAX, &options))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let mut short = [0.0_f64; 4];
    let error = Eigen::decompose_into(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut eigen,
        &mut LinalgWorkspace::from_scalar_slice(&mut short),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(eigen.eigenvalues(), before.eigenvalues());
    let timed = options.with_execution_options(
        (options)
            .execution_options()
            .with_time_limit(core::time::Duration::from_secs(1)),
    );
    assert_eq!(
        Eigen::decompose_with_options(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &timed
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        scioxide_linalg::GeneralizedHermitianEigenvaluesResult::compute_with_options(
            &a,
            &b,
            scioxide_linalg::HermitianInput::Exact,
            scioxide_linalg::HermitianInput::Exact,
            &timed
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
}

#[test]
fn exact_hermitian_admission_preserves_output_for_both_operands() {
    use scioxide_core::{ExecutionContext, SciOxideErrorKind};
    let valid = Matrix::<Complex<f64>>::identity(2).unwrap();
    let options = HermitianEigenOptions::default();
    let mut result = Eigen::decompose(
        &valid,
        &valid,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let saved_values = result.eigenvalues().to_vec();
    let saved_vectors = result.eigenvectors().clone();
    let mut workspace = LinalgWorkspace::new();
    for (row, col, value, kind) in [
        (
            0,
            0,
            Complex::new(1.0, 0.25),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            0,
            1,
            Complex::new(0.0, 0.25),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            1,
            0,
            Complex::new(0.0, -0.25),
            SciOxideErrorKind::InvalidInput,
        ),
        (
            0,
            1,
            Complex::new(f64::NAN, 0.0),
            SciOxideErrorKind::NonFinite,
        ),
        (
            1,
            0,
            Complex::new(0.0, f64::INFINITY),
            SciOxideErrorKind::NonFinite,
        ),
    ] {
        let mut invalid = valid.clone();
        invalid.view_mut().set(row, col, value).unwrap();
        for (a, b) in [(&invalid, &valid), (&valid, &invalid)] {
            let error = Eigen::decompose_into(
                a,
                b,
                scioxide_linalg::HermitianInput::Exact,
                scioxide_linalg::HermitianInput::Exact,
                &options,
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert_eq!(result.eigenvalues(), saved_values);
            assert_eq!(result.eigenvectors(), &saved_vectors);
        }
    }
}
