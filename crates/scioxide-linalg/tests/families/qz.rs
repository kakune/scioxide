#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;
use scioxide_core::{
    Budget, Complex, ExecutionContext, RealScalar, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::{
    Estimation, GeneralizedSchurDecomposition as Qz, GeneralizedSchurWorkspaceRequest,
    LinalgWorkspace, Matrix, SchurOptions, SchurReorderOptions,
};

fn qz<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>) -> Result<Qz<S>, SciOxideError> {
    Qz::decompose(a, b)
}
fn estimates() -> SchurReorderOptions {
    SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate)
}

fn c<R: RealScalar>(v: f64) -> R {
    R::try_from_f64_lossy(v).unwrap()
}
fn matrix<S: NumericScalar>(n: usize, seed: usize) -> Matrix<S> {
    Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let mut bits =
                    (i as u64).wrapping_add((seed as u64).wrapping_mul(0x9e3779b97f4a7c15));
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                let re = (bits >> 32) as f64 / 4294967296. - 0.5;
                let im = if S::IS_REAL {
                    0.
                } else {
                    (bits & 0xffff_ffff) as f64 / 4294967296. - 0.5
                };
                S::from_components(c(re), c(im))
            })
            .collect(),
    )
    .unwrap()
}
fn residual<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>, p: &Qz<S>) -> f64 {
    let n = a.rows();
    let mut error = 0.0_f64;
    for (input, factor) in [(a, p.s()), (b, p.t())] {
        let scale = input
            .as_slice()
            .iter()
            .fold(S::Real::ZERO, |x, v| x.max(v.abs()))
            .max(S::Real::MIN_POSITIVE);
        for i in 0..n {
            for j in 0..n {
                let mut actual = S::ZERO;
                for k in 0..n {
                    for l in 0..n {
                        actual = actual
                            + p.q().view().get(k, i).unwrap().conj()
                                * (input.view().get(k, l).unwrap() / S::from_real(scale))
                                * p.z().view().get(l, j).unwrap();
                    }
                }
                error = error.max(
                    (actual - factor.view().get(i, j).unwrap() / S::from_real(scale))
                        .abs()
                        .try_to_f64_exact()
                        .unwrap(),
                );
            }
        }
    }
    for u in [p.q(), p.z()] {
        for i in 0..n {
            for j in 0..n {
                let dot = (0..n).fold(S::ZERO, |sum, k| {
                    sum + u.view().get(k, i).unwrap().conj() * u.view().get(k, j).unwrap()
                });
                error = error.max(
                    (dot - if i == j { S::ONE } else { S::ZERO })
                        .abs()
                        .try_to_f64_exact()
                        .unwrap(),
                );
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            assert_eq!(p.t().view().get(i, j).unwrap(), S::ZERO);
            if i > j + 1 || !S::IS_REAL {
                assert_eq!(p.s().view().get(i, j).unwrap(), S::ZERO);
            }
        }
    }
    error
}
fn random_pencils<S: NumericScalar>() {
    for n in 0..13 {
        for seed in 1..8 {
            let a = matrix::<S>(n, seed);
            let b = matrix::<S>(n, seed + 7);
            let p = qz(&a, &b).unwrap();
            let error = residual(&a, &b, &p);
            assert!(
                error < 1000. * S::Real::EPSILON.try_to_f64_exact().unwrap(),
                "n={n} seed={seed} residual={error}"
            );
            assert_eq!(p.eigenvalues().len(), n);
        }
    }
}
#[test]
fn real64() {
    random_pencils::<f64>();
}
#[test]
fn real32() {
    random_pencils::<f32>();
}
#[test]
fn complex64() {
    random_pencils::<Complex<f64>>();
}
#[test]
fn complex32() {
    random_pencils::<Complex<f32>>();
}

#[test]
fn infinite_and_singular() {
    let a = Matrix::from_vec(3, 3, vec![1., 2., 3., 0., 4., 5., 0., 0., 6.]).unwrap();
    let b = Matrix::from_vec(3, 3, vec![0., 1., 2., 0., 2., 3., 0., 0., 0.]).unwrap();
    let p = qz(&a, &b).unwrap();
    assert_eq!(p.report().unwrap().infinite_eigenvalues(), 2);
    assert!(residual(&a, &b, &p) < 1e-13);
    let zero = Matrix::<f64>::zeros(3, 3).unwrap();
    assert!(matches!(
        (qz(&zero, &zero)).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
}
#[test]
fn extreme_homogeneous_values() {
    let a = Matrix::from_vec(1, 1, vec![1e300]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1e-300]).unwrap();
    let p = qz(&a, &b).unwrap();
    let v = p.eigenvalues()[0];
    assert!(v.is_finite());
    assert!(v.ratio().is_err());
    assert_eq!(v.alpha().re, 1e300);
    assert_eq!(v.beta(), 1e-300);
}
#[test]
fn prepared_failure_preserves_result() {
    let a = matrix::<f64>(6, 1);
    let b = matrix::<f64>(6, 9);
    let mut p = qz(&a, &b).unwrap();
    let before = p.s().as_slice().to_vec();
    let mut w = LinalgWorkspace::new();
    let opts = SchurOptions::default().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
    );
    assert!(
        Qz::decompose_into(
            &a,
            &b,
            &opts,
            &mut p,
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(p.s().as_slice(), before);
}

fn reordered<S: NumericScalar>() {
    for n in 2..10 {
        for seed in 1..5 {
            let a = matrix::<S>(n, seed);
            let b = matrix::<S>(n, seed + 7);
            let p = qz(&a, &b).unwrap();
            let selection: Vec<_> = (0..n).map(|i| i % 2 == 1).collect();
            let expected: Vec<_> = p
                .blocks()
                .iter()
                .filter(|block| {
                    selection[block.start()..block.start() + block.size()]
                        .iter()
                        .any(|&x| x)
                })
                .flat_map(|block| {
                    p.eigenvalues()[block.start()..block.start() + block.size()]
                        .iter()
                        .map(|v| v.ratio().unwrap())
                })
                .collect();
            let (p, report) = p.reorder_with_options(&selection, &estimates()).unwrap();
            assert_eq!(report.selected_dimension(), expected.len());
            let error = residual(&a, &b, &p);
            assert!(
                error < 1000. * S::Real::EPSILON.try_to_f64_exact().unwrap(),
                "n={n} seed={seed} residual={error}"
            );
            for (v, e) in p.eigenvalues().iter().zip(expected) {
                assert!(
                    (v.ratio().unwrap() - e).norm()
                        < c::<S::Real>(5000.) * S::Real::EPSILON * (S::Real::ONE + e.norm()),
                    "n={n} seed={seed} actual={:?} expected={e:?}",
                    v.ratio().unwrap()
                );
            }
            assert_eq!(
                p.left_deflating_subspace(report.selected_dimension())
                    .unwrap()
                    .cols(),
                report.selected_dimension()
            );
            assert!(report.separation().is_some());
        }
    }
}
#[test]
fn reorder_real64() {
    reordered::<f64>();
}
#[test]
fn reorder_real32() {
    reordered::<f32>();
}
#[test]
fn reorder_complex64() {
    reordered::<Complex<f64>>();
}
#[test]
fn reorder_complex32() {
    reordered::<Complex<f32>>();
}

#[test]
fn zeros_inside_hessenberg_pencils() {
    for k in 0..4 {
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                1., 2., 3., 4., 2., -3., 1., 5., 0., 4., 5., 6., 0., 0., -2., 8.,
            ],
        )
        .unwrap();
        let mut b = Matrix::from_vec(
            4,
            4,
            vec![
                2., 1., 2., 3., 0., 3., 1., 2., 0., 0., 4., 1., 0., 0., 0., 5.,
            ],
        )
        .unwrap();
        b.as_mut_slice()[k * 4 + k] = 0.;
        let p = qz(&a, &b).unwrap();
        assert!(residual(&a, &b, &p) < 1e-12);
        assert_eq!(p.report().unwrap().infinite_eigenvalues(), 1);
        let select: Vec<_> = p.eigenvalues().iter().map(|v| v.is_infinite()).collect();
        let (p, _) = p.reorder(&select).unwrap();
        assert!(p.eigenvalues()[0].is_infinite());
        assert!(residual(&a, &b, &p) < 1e-12);
    }
}
#[test]
fn singular_and_ambiguous_pencils() {
    for data in [vec![1., 2., 2., 4.], vec![1., 0., 0., 1e-20]] {
        let a = Matrix::from_vec(2, 2, data).unwrap();
        assert!(matches!(
            (qz(&a, &a)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
                | scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }
    let a = Matrix::from_vec(1, 1, vec![1e-300]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1e300]).unwrap();
    let p = qz(&a, &b).unwrap();
    assert!(p.eigenvalues()[0].is_finite());
    assert!(p.eigenvalues()[0].ratio().is_err());
}
#[test]
fn budget_cancellation_and_rollback() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let a = matrix::<f64>(6, 2);
    let b = matrix::<f64>(6, 9);
    let options = SchurOptions::default();
    let mut p = qz(&a, &b).unwrap();
    let before = p.clone();
    let mut w = LinalgWorkspace::new();
    // The QZ budget belongs to the options; a parent's own iteration allowance
    // is not consumed and its counts land in `nested()`.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("linalg", "qz", 1),
        6,
        0,
    );
    let mut solve = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let failure = Qz::decompose_into(
        &a,
        &b,
        &options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::ONE),
        ),
        &mut p,
        &mut w,
        &mut solve,
    )
    .unwrap_err();
    assert!(
        matches!(failure.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() == 1)
    );

    assert_eq!(solve.counts().own().iterations(), 0);
    assert_eq!(solve.counts().nested().iterations(), 1);
    let attempt = w.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), scioxide_core::AttemptOutcome::Stopped);
    assert_eq!(attempt.counts().own().iterations(), 1);
    Qz::decompose_into(&a, &b, &options, &mut p, &mut w, &mut solve).unwrap();
    assert_eq!(solve.counts().own().iterations(), 0);
    assert!(solve.counts().nested().iterations() > 1);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Succeeded
    );
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(
        Qz::decompose_into(
            &a,
            &b,
            &options,
            &mut p,
            &mut w,
            ExecutionContext::sequential().with_cancellation(&cancel)
        )
        .is_err()
    );
    let reorder = estimates().with_execution_options(
        (estimates())
            .execution_options()
            .with_iteration_budget(Budget::ONE),
    );
    assert!(
        p.reorder_in_place(
            &[false, false, false, false, false, true],
            &reorder,
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        p.reorder_in_place(
            &[true; 6],
            &reorder,
            &mut w,
            ExecutionContext::sequential().with_cancellation(&cancel)
        )
        .is_err()
    );
    for (actual, expected) in [
        (p.s(), before.s()),
        (p.t(), before.t()),
        (p.q(), before.q()),
        (p.z(), before.z()),
    ] {
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
    let (_, report) = p.reorder_with_options(&[false; 6], &reorder).unwrap();
    assert_eq!(report.selected_dimension(), 0);
    let (_, report) = p.reorder_with_options(&[true; 6], &reorder).unwrap();
    assert_eq!(report.selected_dimension(), 6);
    assert_eq!(
        p.reorder_with_options(&[true], &reorder)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let bad = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
    assert!(qz(&bad, &bad).is_err());
    assert!(qz(&a, &bad).is_err());
}

#[test]
fn workspace_requirements_errors_preserve_output() {
    let a = matrix::<f64>(4, 3);
    let b = matrix::<f64>(4, 5);
    let options = SchurOptions::<f64>::new();
    let mut p = qz(&a, &b).unwrap();
    let before = p.clone();
    let unchanged = |p: &Qz<f64>| {
        p.s().as_slice() == before.s().as_slice()
            && p.t().as_slice() == before.t().as_slice()
            && p.q().as_slice() == before.q().as_slice()
            && p.z().as_slice() == before.z().as_slice()
            && p.eigenvalues() == before.eigenvalues()
    };
    let mut w = LinalgWorkspace::new();
    for (options, kind) in [
        (
            options.with_execution_options((options).execution_options().with_byte_limit(1)),
            SciOxideErrorKind::ResourceLimit,
        ),
        (
            options.with_execution_options(
                (options)
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1)),
            ),
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        assert_eq!(
            Qz::decompose_into(
                &a,
                &b,
                &options,
                &mut p,
                &mut w,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            kind
        );
        assert!(unchanged(&p));
        assert!(w.last_attempt().unwrap().epoch().is_none());
    }
    let mut short = [0.0; 4];
    assert_eq!(
        Qz::decompose_into(
            &a,
            &b,
            &options,
            &mut p,
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(unchanged(&p));
    let mask = [false, false, true, true];
    for (reorder, kind) in [
        (
            estimates()
                .with_execution_options((estimates()).execution_options().with_byte_limit(1)),
            SciOxideErrorKind::ResourceLimit,
        ),
        (
            estimates().with_execution_options(
                (estimates())
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1)),
            ),
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        assert_eq!(
            p.reorder_in_place(&mask, &reorder, &mut w, ExecutionContext::sequential())
                .unwrap_err()
                .kind(),
            kind
        );
        assert!(unchanged(&p));
    }
    assert_eq!(
        p.reorder_in_place(
            &mask,
            &estimates(),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(unchanged(&p));
    assert_eq!(
        qz(&a, &Matrix::zeros(3, 3).unwrap()).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        Qz::decompose_with_options(
            &a,
            &b,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_time_limit(core::time::Duration::from_secs(1))
            )
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    for request in [
        GeneralizedSchurWorkspaceRequest::new(usize::MAX, &options),
        GeneralizedSchurWorkspaceRequest::reorder(usize::MAX, &estimates()),
    ] {
        assert_eq!(
            Qz::<f64>::workspace_requirements(&request)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    }
    let skip = GeneralizedSchurWorkspaceRequest::reorder(4, &SchurReorderOptions::new());
    let estimate = GeneralizedSchurWorkspaceRequest::reorder(4, &estimates());
    assert!(
        Qz::<f64>::workspace_requirements(&estimate)
            .unwrap()
            .required_bytes::<f64>()
            .unwrap()
            > Qz::<f64>::workspace_requirements(&skip)
                .unwrap()
                .required_bytes::<f64>()
                .unwrap()
    );
}
#[test]
fn inseparable_exchange_is_transactional() {
    let a = Matrix::from_vec(2, 2, vec![1., 1., 0., 1.]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap();
    let mut p = qz(&a, &b).unwrap();
    let old = p.s().as_slice().to_vec();
    let mut w = LinalgWorkspace::new();
    assert!(
        p.reorder_in_place(
            &[false, true],
            &estimates(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(p.s().as_slice(), old);
    let (_, report) = p
        .reorder_with_options(&[true, false], &estimates())
        .unwrap();
    assert_eq!(report.separation().unwrap()[0].value(), 0.);
}

#[test]
fn conjugate_pairs_retain_extreme_homogeneous_scale() {
    for scale in [1e300, 1e-300] {
        let a = Matrix::from_vec(2, 2, vec![0., -scale, scale, 0.]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1. / scale, 0., 0., 1. / scale]).unwrap();
        let p = qz(&a, &b).unwrap();
        assert_eq!(p.blocks().len(), 1);
        assert!(
            p.eigenvalues()
                .iter()
                .all(|v| v.is_finite() && v.beta() > 0. && v.alpha().im != 0.)
        );
        assert!(p.eigenvalues().iter().all(|v| v.ratio().is_err()));
    }
}
fn swap_sizes<R: RealScalar + NumericScalar<Real = R>>() {
    for p in 1..=2 {
        for q in 1..=2 {
            let n = p + q;
            let mut a = vec![R::ZERO; n * n];
            let mut b = vec![R::ZERO; n * n];
            for i in 0..n {
                a[i * n + i] = c(if i < p { 1. } else { 5. });
                b[i * n + i] = c(1.);
            }
            for (start, size) in [(0, p), (p, q)] {
                if size == 2 {
                    a[start * n + start + 1] = c(2.);
                    a[(start + 1) * n + start] = c(-1.);
                }
            }
            for i in 0..p {
                for j in p..n {
                    a[i * n + j] = c((i + j + 1) as f64 / 3.);
                    b[i * n + j] = c((i + j + 1) as f64 / 7.);
                }
            }
            let a = Matrix::from_vec(n, n, a).unwrap();
            let b = Matrix::from_vec(n, n, b).unwrap();
            let pencil = qz(&a, &b).unwrap();
            let mut select = vec![false; n];
            select[n - 1] = true;
            let (pencil, report) = pencil.reorder_with_options(&select, &estimates()).unwrap();
            assert_exact_real_pairs(&pencil);
            assert_eq!(report.swaps(), 1);
            assert_eq!(report.selected_dimension(), q);
            assert!(
                residual(&a, &b, &pencil)
                    < 1000. * <R as RealScalar>::EPSILON.try_to_f64_exact().unwrap()
            );
            if q == 2 {
                assert!(pencil.right_deflating_subspace(1).is_err());
            }
        }
    }
}

fn assert_exact_real_pairs<R: RealScalar + NumericScalar<Real = R>>(pencil: &Qz<R>) {
    for block in pencil.blocks().iter().filter(|block| block.size() == 2) {
        let pair = &pencil.eigenvalues()[block.start()..block.start() + 2];
        assert_eq!(pair[0].alpha(), pair[1].alpha().conj());
        assert_eq!(pair[0].beta(), pair[1].beta());
        assert!(pair[0].alpha().im > R::ZERO);
        if let (Ok(first), Ok(second)) = (pair[0].ratio(), pair[1].ratio()) {
            assert_eq!(first, second.conj());
        }
    }
}

#[test]
fn reordered_pairs_remain_exact_conjugates_at_extreme_scales() {
    for scale in [1e-150_f64, 1.0, 1e150] {
        let a = Matrix::from_vec(
            3,
            3,
            [1., 2., 3., 0., 4., -6., 0., 5., 4.]
                .map(|x| x * scale)
                .to_vec(),
        )
        .unwrap();
        let b = Matrix::from_vec(
            3,
            3,
            [1., 0.5, -0.25, 0., 1., 0., 0., 0., 1.]
                .map(|x| x / scale)
                .to_vec(),
        )
        .unwrap();
        let mut pencil = qz(&a, &b).unwrap();
        for select in [
            [false, true, false],
            [false, false, true],
            [false, true, false],
        ] {
            (pencil, _) = pencil.reorder(&select).unwrap();
            assert_exact_real_pairs(&pencil);
            assert!(residual(&a, &b, &pencil) <= 1024.0 * f64::EPSILON);
        }
    }
}
#[test]
fn all_real_swaps64() {
    swap_sizes::<f64>();
}
#[test]
fn all_real_swaps32() {
    swap_sizes::<f32>();
}

fn clustered<R: RealScalar + NumericScalar<Real = R>>() {
    let delta = c::<R>(128.) * <R as RealScalar>::EPSILON;
    let one = R::ONE;
    let zero = R::ZERO;
    let a = Matrix::from_vec(2, 2, vec![one, one, zero, one + delta]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![one, zero, zero, one]).unwrap();
    let p = qz(&a, &b).unwrap();
    let (p, report) = p
        .reorder_with_options(&[false, true], &estimates())
        .unwrap();
    let separation = report.separation().unwrap()[0].value();
    assert!(separation < delta && separation > delta / c(4.));
    assert!(report.right_condition().unwrap().reciprocal().value() < delta * c(2.));
    assert!(residual(&a, &b, &p) < 1000. * <R as RealScalar>::EPSILON.try_to_f64_exact().unwrap());
}
#[test]
fn clustered64() {
    clustered::<f64>();
}
#[test]
fn clustered32() {
    clustered::<f32>();
}

fn order64<S: NumericScalar>() {
    let n = 64;
    let mut a = matrix::<S>(n, 1);
    let mut b = matrix::<S>(n, 2);
    for i in 0..n {
        a.as_mut_slice()[i * n + i] = S::from_real(c((n + i + 1) as f64));
        b.as_mut_slice()[i * n + i] = S::from_real(c((n + i + 2) as f64));
    }
    let p = qz(&a, &b).unwrap();
    let mask: Vec<_> = (0..n).map(|i| i >= n / 2).collect();
    let (p, _) = p.reorder(&mask).unwrap();
    let tolerance = if S::Real::EPSILON > c(1e-10) {
        c(2e-4)
    } else {
        c(3e-11)
    };
    for (input, factor) in [(&a, p.s()), (&b, p.t())] {
        let scale = input
            .as_slice()
            .iter()
            .fold(S::Real::ZERO, |v, x| v.max(x.abs()));
        for i in 0..n {
            for j in 0..n {
                let lhs = (0..n).fold(S::ZERO, |sum, k| {
                    sum + input.view().get(i, k).unwrap() * p.z().view().get(k, j).unwrap()
                });
                let rhs = (0..n).fold(S::ZERO, |sum, k| {
                    sum + p.q().view().get(i, k).unwrap() * factor.view().get(k, j).unwrap()
                });
                assert!((lhs - rhs).abs() / scale < tolerance);
            }
        }
    }
}
#[test]
fn order64_real32() {
    order64::<f32>();
}
#[test]
fn order64_real64() {
    order64::<f64>();
}
#[test]
fn order64_complex32() {
    order64::<Complex<f32>>();
}
#[test]
fn order64_complex64() {
    order64::<Complex<f64>>();
}

#[test]
fn deadline_during_reduction_preserves_output() {
    use scioxide_core::{CapabilityError, Clock, Deadline, MonotonicInstant};
    use std::sync::atomic::{AtomicU64, Ordering};
    struct Tick(AtomicU64);
    impl Clock for Tick {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                std::time::Duration::from_secs(self.0.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }
    let a = matrix::<f64>(6, 1);
    let b = matrix::<f64>(6, 8);
    let mut p = qz(&a, &b).unwrap();
    let old = p.s().as_slice().to_vec();
    let mut w = LinalgWorkspace::new();
    let clock = Tick(AtomicU64::new(0));
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let context = ExecutionContext::sequential()
        .with_deadline(Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(std::time::Duration::from_secs(3)),
        ))
        .unwrap();
    assert!(matches!(
        (Qz::decompose_into(&a, &b, &SchurOptions::default(), &mut p, &mut w, context))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(p.s().as_slice(), old);
}

/// Largest normwise backward error of `Q S Z^H = A`, `Q T Z^H = B` (each
/// relative to its input's largest entry) and departure of Q and Z from
/// unitarity, in O(n^3).
fn backward_error<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>, p: &Qz<S>) -> f64 {
    let n = a.rows();
    let at = |m: &Matrix<S>, i: usize, j: usize| m.view().get(i, j).unwrap();
    let mut error = 0.0_f64;
    for (input, factor) in [(a, p.s()), (b, p.t())] {
        let scale = input
            .as_slice()
            .iter()
            .fold(S::Real::ZERO, |x, v| x.max(v.abs()))
            .max(S::Real::MIN_POSITIVE);
        let product: Vec<S> = (0..n * n)
            .map(|index| {
                let (k, j) = (index / n, index % n);
                (0..n).fold(S::ZERO, |sum, l| {
                    sum + at(factor, k, l) * at(p.z(), j, l).conj()
                })
            })
            .collect();
        for i in 0..n {
            for j in 0..n {
                let rebuilt =
                    (0..n).fold(S::ZERO, |sum, k| sum + at(p.q(), i, k) * product[k * n + j]);
                error = error.max(
                    ((rebuilt - at(input, i, j)) / S::from_real(scale))
                        .abs()
                        .try_to_f64_exact()
                        .unwrap(),
                );
            }
        }
    }
    for u in [p.q(), p.z()] {
        for i in 0..n {
            for j in 0..n {
                let dot = (0..n).fold(S::ZERO, |sum, k| sum + at(u, k, i).conj() * at(u, k, j));
                let identity = if i == j { S::ONE } else { S::ZERO };
                error = error.max((dot - identity).abs().try_to_f64_exact().unwrap());
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            assert_eq!(at(p.t(), i, j), S::ZERO);
            if i > j + 1 || !S::IS_REAL {
                assert_eq!(at(p.s(), i, j), S::ZERO);
            }
        }
    }
    error
}

/// `D1 M D2` with diagonal gradings spanning `10^-6..10^6` along rows and
/// columns.
fn graded<S: NumericScalar>(n: usize, seed: usize) -> Matrix<S> {
    let base = matrix::<S>(n, seed);
    let grade = |index: usize, sign: f64| {
        S::from_real(c(
            10.0_f64.powf(sign * 6.0 * index as f64 / (n.max(2) - 1) as f64)
        ))
    };
    Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                grade(row, -1.0) * base.view().get(row, col).unwrap() * grade(col, 1.0)
            })
            .collect(),
    )
    .unwrap()
}

fn graded_pencils<S: NumericScalar>() {
    // Order 150 runs the blocked Householder front end.
    for n in [8, 150] {
        let a = graded::<S>(n, 3);
        let b = graded::<S>(n, 5);
        let p = qz(&a, &b).unwrap();
        let error = backward_error(&a, &b, &p);
        let bound = 100.0 * n as f64 * S::Real::EPSILON.try_to_f64_exact().unwrap();
        assert!(error < bound, "n={n} error={error}");
        assert_eq!(p.eigenvalues().len(), n);
    }
}
#[test]
fn graded_real64() {
    graded_pencils::<f64>();
}
#[test]
fn graded_complex64() {
    graded_pencils::<Complex<f64>>();
}

#[test]
fn blocked_front_end_keeps_exact_infinite_eigenvalues() {
    // Hessenberg A and triangular B whose diagonal has exact zeros: the QR
    // front end and the reduction leave the zeros exact, so each is an
    // infinite eigenvalue.
    let n = 150;
    let mut a = matrix::<f64>(n, 7);
    let mut b = matrix::<f64>(n, 9);
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 {
                a.as_mut_slice()[i * n + j] = 0.0;
            }
            if i > j || (i == j && [3, 70, 140].contains(&i)) {
                b.as_mut_slice()[i * n + j] = 0.0;
            }
        }
    }
    let p = qz(&a, &b).unwrap();
    assert_eq!(p.report().unwrap().infinite_eigenvalues(), 3);
    let error = backward_error(&a, &b, &p);
    assert!(error < 100.0 * n as f64 * f64::EPSILON, "error={error}");
}
