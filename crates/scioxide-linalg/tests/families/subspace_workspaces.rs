#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{
    AlgebraicScalar, Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorKind,
};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy, SubspaceBasisWorkspaceRequest,
    null_space, null_space_into, orth,
};

fn scalar<S: NumericScalar>(x: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(x).unwrap())
}
fn tolerance<S: NumericScalar>() -> f64 {
    S::Real::EPSILON.into_f64() * 200.0
}
fn assert_unitary_columns<S: NumericScalar>(q: &Matrix<S>) {
    for i in 0..q.cols() {
        for j in 0..q.cols() {
            let sum = (0..q.rows()).fold(S::ZERO, |sum, k| {
                sum + q.view().get(k, i).unwrap().conj() * q.view().get(k, j).unwrap()
            });
            let expected = if i == j { S::ONE } else { S::ZERO };
            assert!((sum - expected).abs().into_f64() < tolerance::<S>());
        }
    }
}
fn bases<S: NumericScalar>() {
    for (m, n) in [(3, 3), (5, 3), (3, 5), (0, 3), (3, 0), (0, 0)] {
        let mut a = Matrix::<S>::zeros(m, n).unwrap();
        for i in 0..m.min(n).min(2) {
            a.view_mut().set(i, i, scalar::<S>((i + 1) as f64)).unwrap();
        }
        let o = orth(a.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        let z = null_space(a.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        assert_eq!(o.basis().cols(), m.min(n).min(2));
        assert_eq!(z.basis().cols(), n - o.basis().cols());
        assert_unitary_columns(o.basis());
        assert_unitary_columns(z.basis());
        assert!(
            a.view()
                .mul_mat(z.basis())
                .unwrap()
                .as_slice()
                .iter()
                .all(|x| x.abs().into_f64() < tolerance::<S>())
        );
        let left = null_space(a.view().adjoint(), RankPolicy::Exact).unwrap();
        assert_eq!(left.basis().cols(), m - o.basis().cols());
        assert_unitary_columns(left.basis());
        assert!(
            a.view()
                .adjoint()
                .mul_mat(left.basis())
                .unwrap()
                .as_slice()
                .iter()
                .all(|x| x.abs().into_f64() < tolerance::<S>())
        );
    }
}
#[test]
fn bases_all_scalars_and_shapes() {
    bases::<f32>();
    bases::<f64>();
    bases::<Complex<f32>>();
    bases::<Complex<f64>>();
}
#[test]
fn reuse_rank_loss_and_failure() {
    let mut a = Matrix::<f64>::identity(3).unwrap();
    let mut out = Matrix::empty();
    let mut work = LinalgWorkspace::new();
    for rank in [3, 2, 1, 0, 3] {
        a.as_mut_slice().fill(0.0);
        for i in 0..rank {
            a.view_mut().set(i, i, 1.0).unwrap();
        }
        let report = null_space_into(
            &a,
            RankPolicy::Exact,
            &LinalgExecutionOptions::new(),
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(report.rank().rank(), rank);
        assert_eq!(out.cols(), 3 - rank);
    }
    let before = out.clone();
    a.view_mut().set(0, 0, f64::NAN).unwrap();
    assert_eq!(
        null_space_into(
            &a,
            RankPolicy::Exact,
            &LinalgExecutionOptions::new(),
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out.as_slice(), before.as_slice());
    assert_eq!(out.cols(), before.cols());
}

use scioxide_linalg::{
    ProcrustesUniqueness, orthogonal_procrustes, subspace_angles, subspace_angles_into,
};

fn angles<S: NumericScalar>() {
    let a = Matrix::<S>::from_vec(
        4,
        2,
        vec![
            S::ONE,
            S::ZERO,
            S::ZERO,
            S::ONE,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ZERO,
        ],
    )
    .unwrap();
    for theta in [
        0.0_f64,
        1.0e-6,
        0.3,
        core::f64::consts::FRAC_PI_4,
        core::f64::consts::FRAC_PI_2 - 1.0e-6,
        core::f64::consts::FRAC_PI_2,
    ] {
        let b = Matrix::from_vec(
            4,
            2,
            vec![
                scalar::<S>(theta.cos()),
                S::ZERO,
                S::ZERO,
                S::ONE,
                scalar::<S>(theta.sin()),
                S::ZERO,
                S::ZERO,
                S::ZERO,
            ],
        )
        .unwrap();
        let result =
            subspace_angles(a.view(), b.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        assert_eq!(result.angles().len(), 2);
        assert!(
            (result.angles()[0].into_f64() - theta).abs() < tolerance::<S>(),
            "theta={theta}, got={:?}",
            result.angles()
        );
        assert!(result.angles()[1].into_f64() < tolerance::<S>());
        let mixing = Matrix::from_vec(
            2,
            2,
            vec![
                scalar::<S>(2.0),
                scalar::<S>(1.0),
                scalar::<S>(-1.0),
                scalar::<S>(3.0),
            ],
        )
        .unwrap();
        let changed = b.view().mul_mat(&mixing).unwrap();
        let result2 =
            subspace_angles(a.view(), changed.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        for (&x, &y) in result.angles().iter().zip(result2.angles()) {
            assert!((x - y).abs().into_f64() < tolerance::<S>());
        }
        let swapped =
            subspace_angles(b.view(), a.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        for (&x, &y) in result.angles().iter().zip(swapped.angles()) {
            assert!((x - y).abs().into_f64() < tolerance::<S>());
        }
    }
}
#[test]
fn principal_angles_all_scalars() {
    angles::<f32>();
    angles::<f64>();
    angles::<Complex<f32>>();
    angles::<Complex<f64>>();
}
fn procrustes<S: NumericScalar>() {
    for (m, n) in [(5, 3), (3, 3), (2, 3), (0, 3), (3, 0), (0, 0)] {
        let a = Matrix::from_vec(
            m,
            n,
            (0..m * n)
                .map(|i| scalar::<S>(((i * 7 + 3) % 11) as f64 - 5.0))
                .collect(),
        )
        .unwrap();
        let mut rotation = Matrix::<S>::identity(n).unwrap();
        if n >= 2 {
            rotation.view_mut().set(0, 0, S::ZERO).unwrap();
            rotation.view_mut().set(1, 1, S::ZERO).unwrap();
            rotation.view_mut().set(0, 1, S::ONE).unwrap();
            rotation.view_mut().set(1, 0, S::ONE).unwrap();
        }
        let b = a.view().mul_mat(&rotation).unwrap();
        let aligned =
            orthogonal_procrustes(a.view(), b.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        assert_unitary_columns(aligned.transformation());
        assert!(
            aligned.report().residual_norm().into_f64() < tolerance::<S>() * 30.0,
            "{:?}",
            aligned.report()
        );
        let recovered = a.view().mul_mat(aligned.transformation()).unwrap();
        for (&x, &y) in recovered.as_slice().iter().zip(b.as_slice()) {
            assert!((x - y).abs().into_f64() < tolerance::<S>() * 10.0);
        }
    }
}
#[test]
fn procrustes_all_scalars_and_shapes() {
    procrustes::<f32>();
    procrustes::<f64>();
    procrustes::<Complex<f32>>();
    procrustes::<Complex<f64>>();
}
#[test]
fn complex_adjoint_convention_and_strided_inputs() {
    let a = Matrix::from_vec(
        3,
        2,
        vec![
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(0.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(-2.0, 0.0),
            Complex::new(1.0, 3.0),
        ],
    )
    .unwrap();
    let r = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let b = a.view().mul_mat(&r).unwrap();
    let result = orthogonal_procrustes(a.view(), b.view(), RankPolicy::Exact).unwrap();
    for (&x, &y) in result.transformation().as_slice().iter().zip(r.as_slice()) {
        assert!((x - y).norm() < 1e-12);
    }
    // Exercise non-real phases at the narrower precision as well.
    let a32 = Matrix::from_vec(
        3,
        2,
        a.as_slice()
            .iter()
            .map(|x| Complex::new(x.real_component() as f32, x.imaginary_component() as f32))
            .collect(),
    )
    .unwrap();
    let b32 = Matrix::from_vec(
        3,
        2,
        b.as_slice()
            .iter()
            .map(|x| Complex::new(x.real_component() as f32, x.imaginary_component() as f32))
            .collect(),
    )
    .unwrap();
    let aligned32 =
        orthogonal_procrustes(a32.view(), b32.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
    assert_unitary_columns(aligned32.transformation());
    for (x, y) in aligned32
        .transformation()
        .as_slice()
        .iter()
        .zip(r.as_slice())
    {
        let expected = Complex::new(y.real_component() as f32, y.imaginary_component() as f32);
        assert!((*x - expected).norm() < 2e-5);
    }
    let angles32 =
        subspace_angles(a32.view(), b32.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
    assert!(angles32.angles().iter().all(|&angle| angle < 2e-5));
    let transposed = a.view().adjoint().try_to_matrix().unwrap();
    let z = null_space(
        transposed.view().adjoint(),
        RankPolicy::DimensionScaledEpsilon,
    )
    .unwrap();
    assert_eq!(z.basis().cols(), 0);
    let mut storage = Matrix::zeros(6, 4).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            storage
                .view_mut()
                .set(i * 2, j * 2, a.view().get(i, j).unwrap())
                .unwrap();
        }
    }
    let view = scioxide_linalg::MatrixView::from_slice_with_strides(3, 2, storage.as_slice(), 8, 2)
        .unwrap();
    let angles = subspace_angles(view, b.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
    assert!(angles.angles().iter().all(|x| x.abs() < 1e-12));
}
#[test]
fn independent_optimum_and_nonuniqueness() {
    let a = Matrix::<f64>::identity(2).unwrap();
    let b = Matrix::from_vec(2, 2, vec![3.0, 0.0, 0.0, -2.0]).unwrap();
    let result = orthogonal_procrustes(a.view(), b.view(), RankPolicy::Exact).unwrap();
    assert!((result.report().scale() - 5.0).abs() < 1e-13);
    assert!((result.report().residual_norm() - 5.0_f64.sqrt()).abs() < 1e-13);
    assert_eq!(result.report().uniqueness(), ProcrustesUniqueness::Unique);
    let deficient = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let result = orthogonal_procrustes(a.view(), deficient.view(), RankPolicy::Exact).unwrap();
    assert_eq!(
        result.report().uniqueness(),
        ProcrustesUniqueness::Nonunique
    );
}
#[test]
fn range_safe_cross_product_and_unrepresentable_output() {
    let a = Matrix::from_vec(2, 1, vec![1e200_f64, 1e200]).unwrap();
    let b = Matrix::from_vec(2, 1, vec![1e108_f64, -1e108]).unwrap();
    let result = orthogonal_procrustes(a.view(), b.view(), RankPolicy::Exact).unwrap();
    // The cancelling 1e308 - 1e308 cross product is a Standard product, within
    // gamma_2 (|A|^T |B|) = 4 epsilon 1e308 of its exact zero.
    assert!(result.report().scale() <= 4.0 * f64::EPSILON * 1e308);
    assert!((result.report().residual_norm() / 1e200 - 2.0_f64.sqrt()).abs() < 1e-14);
    let b = Matrix::from_vec(2, 1, vec![1e108_f64, 1e108]).unwrap();
    assert!(orthogonal_procrustes(a.view(), b.view(), RankPolicy::Exact).is_err());
    let tiny = Matrix::from_vec(1, 1, vec![1e-150_f64]).unwrap();
    let result = orthogonal_procrustes(tiny.view(), tiny.view(), RankPolicy::Exact).unwrap();
    assert!((result.report().scale() / 1e-300 - 1.0).abs() < 1e-14);
}

#[test]
fn rank_policy_selects_basis_without_truncating_alignment() {
    let a = Matrix::from_vec(3, 3, vec![1.0_f64, 0.0, 0.0, 0.0, 1e-8, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let policy = RankPolicy::Threshold(scioxide_core::RankThreshold::new(1e-7, 0.0).unwrap());
    assert_eq!(orth(&a, policy).unwrap().basis().cols(), 1);
    assert_eq!(orth(&a, RankPolicy::Exact).unwrap().basis().cols(), 2);
    let b = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, -1e-8]).unwrap();
    let identity = Matrix::identity(2).unwrap();
    let r = orthogonal_procrustes(identity.view(), b.view(), policy).unwrap();
    assert_eq!(
        r.report().uniqueness(),
        ProcrustesUniqueness::NumericallyUnresolved
    );
    assert_eq!(r.report().cross_product().rank().rank(), 1);
    assert!((r.transformation().view().get(1, 1).unwrap() + 1.0).abs() < 1e-12);
}

#[test]
fn stage_budgets_cancellation_and_empty_angles() {
    use scioxide_core::{
        AttemptOutcome, Budget, BudgetKind, CancellationToken, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let a = Matrix::<f64>::identity(3).unwrap();
    // A parent's scope-local iteration allowance does not limit the nested
    // stages: the operation runs as a child epoch and reports its own counts.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(667),
        scioxide_core::algorithm_id!("linalg", "subspaces", 1),
        3,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let mut out = vec![99.0];
    let mut workspace = LinalgWorkspace::new();
    subspace_angles_into(
        &a,
        &a,
        RankPolicy::Exact,
        &LinalgExecutionOptions::new(),
        &mut out,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(out, vec![0.0; 3]);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    // One admitted iteration per SVD stage: two ranges, cross product, residual.
    assert_eq!(attempt.counts().own().iterations(), 4);
    let stages = attempt.counts().total().unwrap().iterations();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), stages);
    // Without an angle below 45 degrees the sine stage does not run.
    let x = Matrix::from_vec(3, 1, vec![1.0, 0.0, 0.0]).unwrap();
    let y = Matrix::from_vec(3, 1, vec![0.0, 1.0, 1.0]).unwrap();
    subspace_angles_into(
        &x,
        &y,
        RankPolicy::Exact,
        &LinalgExecutionOptions::new(),
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((out[0] - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.counts().own().iterations(), 3);
    // The iteration bound of the nested SVD stages comes from the options.
    let hard = Matrix::from_vec(
        5,
        4,
        (0..20).map(|i| ((i * 7 + 3) % 17) as f64 - 8.0).collect(),
    )
    .unwrap();
    let before = out.clone();
    assert_eq!(
        subspace_angles_into(
            &hard,
            &hard,
            RankPolicy::Exact,
            &LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
            &mut out,
            &mut workspace,
            &mut context,
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out, before);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let token = CancellationToken::new();
    token.cancel();
    let mut out = Matrix::<f64>::identity(1).unwrap();
    assert!(
        scioxide_linalg::orthogonal_procrustes_into(
            &a,
            &a,
            RankPolicy::Exact,
            &LinalgExecutionOptions::new(),
            &mut out,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(out.as_slice(), &[1.0]);
    for (m, n, k) in [(0, 3, 2), (3, 0, 2), (3, 2, 0), (3, 2, 3)] {
        let a = Matrix::<f64>::zeros(m, n).unwrap();
        let b = Matrix::<f64>::zeros(m, k).unwrap();
        assert!(
            subspace_angles(a.view(), b.view(), RankPolicy::Exact)
                .unwrap()
                .angles()
                .is_empty()
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_agrees_with_sequential() {
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let context = scioxide_core::ExecutionContext::new(&pool);
    let a = Matrix::from_vec(
        12,
        4,
        (0..48).map(|i| ((i * 13 + 7) % 23) as f64 - 11.0).collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        12,
        4,
        (0..48).map(|i| ((i * 11 + 3) % 19) as f64 - 9.0).collect(),
    )
    .unwrap();
    let expected = subspace_angles(a.view(), b.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
    let mut output = Vec::new();
    let mut w = LinalgWorkspace::new();
    let options = LinalgExecutionOptions::new().with_execution_policy(
        scioxide_core::ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap()),
    );
    subspace_angles_into(
        &a,
        &b,
        RankPolicy::DimensionScaledEpsilon,
        &options,
        &mut output,
        &mut w,
        context,
    )
    .unwrap();
    assert_eq!(output.len(), expected.angles().len());
    for (&x, &y) in output.iter().zip(expected.angles()) {
        assert!((x - y).abs() < 1e-12);
    }
    let expected =
        orthogonal_procrustes(a.view(), b.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
    let mut out = Matrix::empty();
    orthogonal_procrustes_into(
        &a,
        &b,
        RankPolicy::DimensionScaledEpsilon,
        &options,
        &mut out,
        &mut w,
        context,
    )
    .unwrap();
    for (&x, &y) in out
        .as_slice()
        .iter()
        .zip(expected.transformation().as_slice())
    {
        assert!((x - y).abs() < 1e-12);
    }
}

#[test]
fn dense_range_reconstruction_and_unequal_dimension_angles() {
    for (m, n) in [(5, 3), (3, 5), (4, 4)] {
        let a = Matrix::from_vec(
            m,
            n,
            (0..m * n)
                .map(|i| {
                    Complex::new(
                        ((i * 7 + 3) % 13) as f64 - 6.0,
                        ((i * 11 + 5) % 17) as f64 - 8.0,
                    )
                })
                .collect(),
        )
        .unwrap();
        let range = orth(a.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        assert_unitary_columns(range.basis());
        let coefficients = range
            .basis()
            .view()
            .adjoint()
            .try_to_matrix()
            .unwrap()
            .view()
            .mul_mat(&a)
            .unwrap();
        let reconstructed = range.basis().view().mul_mat(&coefficients).unwrap();
        for (&x, &y) in reconstructed.as_slice().iter().zip(a.as_slice()) {
            assert!((x - y).norm() < 1e-11);
        }
        let z = null_space(a.view(), RankPolicy::DimensionScaledEpsilon).unwrap();
        assert_unitary_columns(z.basis());
        for x in a.view().mul_mat(z.basis()).unwrap().as_slice() {
            assert!(x.norm() < 1e-11);
        }
    }
    let a = Matrix::<f64>::identity(4).unwrap();
    let b = a.view().block(0, 0, 4, 2).unwrap();
    let result = subspace_angles(a.view(), b, RankPolicy::Exact).unwrap();
    assert_eq!(result.angles(), vec![0.0, 0.0]);
    assert!(
        subspace_angles(
            a.view(),
            Matrix::<f64>::identity(3).unwrap().view(),
            RankPolicy::Exact
        )
        .is_err()
    );
    assert!(orthogonal_procrustes(a.view(), b, RankPolicy::Exact).is_err());
}

#[test]
fn left_null_space_uses_complex_adjoint() {
    let a = Matrix::from_vec(
        2,
        1,
        vec![Complex::new(1.0_f64, 0.0), Complex::new(0.0, 1.0)],
    )
    .unwrap();
    let left = null_space(a.view().adjoint(), RankPolicy::DimensionScaledEpsilon).unwrap();
    assert_eq!((left.basis().rows(), left.basis().cols()), (2, 1));
    assert_unitary_columns(left.basis());
    let residual = a.view().adjoint().mul_mat(left.basis()).unwrap();
    assert!(residual.as_slice()[0].norm() < 1e-12);
    let unconjugated = a.view().transpose().mul_mat(left.basis()).unwrap();
    assert!(unconjugated.as_slice()[0].norm() > 1.0);
}

#[test]
fn procrustes_cross_product_scale_can_round_to_zero() {
    let a = Matrix::from_vec(1, 1, vec![1e-200_f64]).unwrap();
    let result = orthogonal_procrustes(&a, &a, RankPolicy::Exact).unwrap();
    assert!((result.transformation().as_slice()[0] - 1.0).abs() <= f64::EPSILON);
    assert_eq!(result.report().scale(), 0.0);
    assert_eq!(result.report().residual_norm(), 0.0);
}

#[test]
fn tall_null_space_reserves_a_thin_svd() {
    let options = LinalgExecutionOptions::new();
    let bytes = |request| {
        LinalgWorkspace::<f64>::workspace_requirements(&request)
            .unwrap()
            .required_bytes::<f64>()
            .unwrap()
    };
    let null = bytes(SubspaceBasisWorkspaceRequest::null_space(
        20_000, 10, &options,
    ));
    let range = bytes(SubspaceBasisWorkspaceRequest::orth(20_000, 10, &options));
    assert!(null <= 2 * range, "null space {null} > 2 x orth {range}");
}

// `B = A Q` for a signed (real) or phased (complex) cyclic permutation `Q`.
fn blocked_alignment<S: NumericScalar>() {
    let (rows, cols) = (600, 24);
    let entry = |i: usize, j: usize| {
        let base = ((i * 7 + j * 13) % 29) as f64 / 29.0 - 0.5;
        let diagonal = if i % cols == j { 2.0 } else { 0.0 };
        S::from_components(
            S::Real::try_from_f64_lossy(base + diagonal).unwrap(),
            S::Real::try_from_f64_lossy(((i + 3 * j) % 11) as f64 / 11.0).unwrap(),
        )
    };
    let phase = |j: usize| {
        let sign = if j.is_multiple_of(2) {
            S::Real::ONE
        } else {
            -S::Real::ONE
        };
        if S::IS_REAL {
            S::from_real(sign)
        } else {
            S::from_components(S::Real::ZERO, sign)
        }
    };
    let mut a = Matrix::<S>::zeros(rows, cols).unwrap();
    let mut b = Matrix::<S>::zeros(rows, cols).unwrap();
    for i in 0..rows {
        for j in 0..cols {
            a.view_mut().set(i, j, entry(i, j)).unwrap();
            b.view_mut()
                .set(i, (j + 1) % cols, entry(i, j) * phase(j))
                .unwrap();
        }
    }
    let alignment = orthogonal_procrustes(&a, &b, RankPolicy::DimensionScaledEpsilon).unwrap();
    assert_eq!(
        alignment.report().uniqueness(),
        ProcrustesUniqueness::Unique
    );
    let norm = a
        .as_slice()
        .iter()
        .map(|x| x.abs().into_f64().powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(alignment.report().residual_norm().into_f64() < tolerance::<S>() * norm);
    for i in 0..cols {
        for j in 0..cols {
            let expected = if j == (i + 1) % cols {
                phase(i)
            } else {
                S::ZERO
            };
            let actual = alignment.transformation().view().get(i, j).unwrap();
            assert!((actual - expected).abs().into_f64() < tolerance::<S>());
        }
    }
    let angles = subspace_angles(&a, &b, RankPolicy::DimensionScaledEpsilon).unwrap();
    assert_eq!(angles.angles().len(), cols);
    assert!(
        angles
            .angles()
            .iter()
            .all(|angle| angle.into_f64() < tolerance::<S>().sqrt())
    );
}

#[test]
fn blocked_products_recover_known_alignment() {
    blocked_alignment::<f32>();
    blocked_alignment::<f64>();
    blocked_alignment::<Complex<f32>>();
    blocked_alignment::<Complex<f64>>();
}

#[cfg(feature = "std")]
use scioxide_linalg::orthogonal_procrustes_into;
