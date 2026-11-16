//! Independent LDL reconstruction, solves, and lifecycle regression coverage.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{LdlDecomposition, Matrix, Vector};

fn scalar<S: NumericScalar>(x: f64, y: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_lossy(x).unwrap(),
        S::Real::try_from_f64_lossy(y).unwrap(),
    )
}
fn check<S: NumericScalar>(a: &Matrix<S>, tol: f64, solve: bool) {
    let f = LdlDecomposition::factor(a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let n = a.rows();
    let l = f.l_matrix().view();
    let mut d = Matrix::zeros(n, n).unwrap();
    for block in f.pivot_blocks() {
        let k = block.start();
        d.view_mut().set(k, k, f.diagonal()[k]).unwrap();
        if block.size() == 2 {
            d.view_mut().set(k + 1, k + 1, f.diagonal()[k + 1]).unwrap();
            d.view_mut().set(k + 1, k, f.off_diagonal()[k]).unwrap();
            d.view_mut()
                .set(k, k + 1, f.off_diagonal()[k].conj())
                .unwrap();
        }
    }
    let mut scale = S::Real::ZERO;
    let mut error = S::Real::ZERO;
    for i in 0..n {
        for j in 0..n {
            let original = a
                .view()
                .get(f.permutation()[i], f.permutation()[j])
                .unwrap();
            let mut reconstructed = S::ZERO;
            for p in 0..n {
                for q in 0..n {
                    reconstructed = reconstructed
                        + l.get(i, p).unwrap()
                            * d.view().get(p, q).unwrap()
                            * l.get(j, q).unwrap().conj();
                }
            }
            error = error.max((original - reconstructed).abs());
            scale = scale.max(original.abs());
        }
    }
    assert!(
        error <= S::Real::try_from_f64_lossy(tol).unwrap() * scale.max(S::Real::ONE),
        "reconstruction {error:?} scale {scale:?}"
    );
    if solve
        && f.inertia(scioxide_linalg::RankPolicy::Exact)
            .unwrap()
            .zero()
            == 0
    {
        let exact: Vec<_> = (0..n).map(|i| scalar::<S>((i + 1) as f64, 0.0)).collect();
        let b = Vector::from_vec(
            (0..n)
                .map(|i| (0..n).fold(S::ZERO, |v, j| v + a.view().get(i, j).unwrap() * exact[j]))
                .collect(),
        );
        let x = f.solve(&b).unwrap();
        for (&x, &y) in x.as_slice().iter().zip(&exact) {
            assert!(
                (x - y).abs()
                    <= S::Real::try_from_f64_lossy(tol * 100.0).unwrap()
                        * y.abs().max(S::Real::ONE),
                "solution {x:?} expected {y:?}"
            );
        }
    }
}
fn cases<S: NumericScalar>() {
    for n in 1..10 {
        let a = Matrix::<S>::from_vec(
            n,
            n,
            (0..n * n)
                .map(|p| {
                    let i = p / n;
                    let j = p % n;
                    let re = if i == j {
                        if i % 3 == 0 { 0.0 } else { (i as f64) - 3.0 }
                    } else {
                        (((i + j) * 7 + i * j * 3) % 11) as f64 - 5.0
                    };
                    let im = if i == j {
                        0.0
                    } else {
                        ((i.abs_diff(j) * 3) % 7) as f64 * if i > j { 1.0 } else { -1.0 }
                    };
                    scalar::<S>(re, im)
                })
                .collect(),
        )
        .unwrap();
        check(
            &a,
            if S::Real::EPSILON > S::Real::try_from_f64_lossy(1e-10).unwrap() {
                2e-5
            } else {
                5e-14
            },
            true,
        );
    }
}
#[test]
fn all_scalars_reconstruction_and_solve() {
    cases::<f32>();
    cases::<f64>();
    cases::<Complex<f32>>();
    cases::<Complex<f64>>();
}
#[test]
fn zero_diagonal_requires_two_by_two() {
    let a = Matrix::<f64>::from_vec(2, 2, vec![0.0, 2.0, 2.0, 0.0]).unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    assert_eq!(f.pivot_blocks().next().unwrap().size(), 2);
    assert_eq!(
        (
            f.inertia(scioxide_linalg::RankPolicy::Exact)
                .unwrap()
                .positive(),
            f.inertia(scioxide_linalg::RankPolicy::Exact)
                .unwrap()
                .negative()
        ),
        (1, 1)
    );
    assert!((f.condition_estimate().unwrap().reciprocal().value() - 1.0_f64).abs() < 1e-14);
    check(&a, 1e-14, true);
}
#[test]
fn singular_factor_retains_inertia() {
    let a =
        Matrix::<f64>::from_vec(3, 3, vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -2.0]).unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    assert_eq!(
        f.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
        scioxide_sparse::LdlInertia::new(1, 1, 1).unwrap()
    );
    assert!(matches!(
        (f.solve(&Vector::from_vec(vec![1.0; 3])))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
    check(&a, 1e-14, false);
}
#[test]
fn scaling_and_near_singularity() {
    for scale in [1e-300, 1e300] {
        let a = Matrix::<f64>::from_vec(2, 2, vec![0.0, scale, scale, 0.0]).unwrap();
        let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
        let x = f.solve(&Vector::from_vec(vec![scale, scale])).unwrap();
        assert_eq!(x.as_slice(), &[1.0, 1.0]);
        assert_eq!(
            f.inertia(scioxide_linalg::RankPolicy::Exact)
                .unwrap()
                .negative(),
            1
        );
        assert!((f.condition_estimate().unwrap().reciprocal().value() - 1.0_f64).abs() < 1e-14);
    }
    let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1e-30]).unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    assert_eq!(
        f.inertia(scioxide_linalg::RankPolicy::Exact)
            .unwrap()
            .negative(),
        1
    );
    assert_eq!(
        f.inertia(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)
            .unwrap()
            .zero(),
        1
    );
}

#[test]
fn extreme_diagonal_condition_and_subnormal_pivot() {
    for values in [
        vec![1e300_f64, 0.0, 0.0, -1e-300],
        vec![f64::from_bits(1), 0.0, 0.0, -f64::from_bits(1)],
    ] {
        let a = Matrix::from_vec(2, 2, values.clone()).unwrap();
        let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
        assert_eq!(
            f.solve(&Vector::from_vec(vec![values[0], values[3]]))
                .unwrap()
                .as_slice(),
            &[1.0, 1.0]
        );
        assert!(f.condition_estimate().unwrap().reciprocal().value() > 0.0);
    }
}

fn congruence<S: NumericScalar>() {
    let n = 7;
    let t: Vec<S> = (0..n * n)
        .map(|p| {
            let i = p / n;
            let j = p % n;
            if i == j {
                S::ONE
            } else if i > j {
                scalar::<S>(((i + j) % 3) as f64 - 1.0, ((i + j) % 2) as f64)
            } else {
                S::ZERO
            }
        })
        .collect();
    let signs = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.0];
    let data = (0..n * n)
        .map(|p| {
            (0..n).fold(S::ZERO, |sum, k| {
                sum + t[(p / n) * n + k] * scalar::<S>(signs[k], 0.0) * t[(p % n) * n + k].conj()
            })
        })
        .collect();
    let a = Matrix::from_vec(n, n, data).unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    // Integer construction is exact, but elimination may perturb the null direction.
    let inertia = f
        .inertia(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)
        .unwrap();
    assert_eq!(inertia, scioxide_sparse::LdlInertia::new(3, 3, 1).unwrap());
    check(
        &a,
        if S::Real::EPSILON > S::Real::try_from_f64_lossy(1e-10).unwrap() {
            2e-5
        } else {
            2e-13
        },
        false,
    );
}
#[test]
fn known_inertia_under_real_and_complex_congruence() {
    congruence::<f32>();
    congruence::<f64>();
    congruence::<Complex<f32>>();
    congruence::<Complex<f64>>();
}

#[test]
fn kkt_inertia_and_independent_residual() {
    // [H B^T; B 0], H positive definite and B full row rank: inertia (3,2,0).
    let a = Matrix::from_vec(
        5,
        5,
        vec![
            4.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    assert_eq!(
        f.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
        scioxide_sparse::LdlInertia::new(3, 2, 0).unwrap()
    );
    check(&a, 2e-14, true);
    let b = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = f
        .solve_with_options(
            &b,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
                .with_residual(scioxide_linalg::Estimation::Estimate),
        )
        .unwrap();
    assert!(result.report().residual().unwrap().scaled_norm() < 1e-14);
    let rejected = f
        .solve_with_options(
            &b,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact)
                .with_residual(scioxide_linalg::Estimation::Estimate)
                .with_minimum_reciprocal_condition(
                    scioxide_linalg::ReciprocalConditionThreshold::new(1.0).unwrap(),
                ),
        )
        .unwrap();
    assert_eq!(
        rejected.report().termination(),
        scioxide_linalg::DirectSolveTermination::IllConditioned
    );
    assert!(rejected.solution().is_none());
}

#[test]
fn views_triangles_operator_and_transactions() {
    use scioxide_core::Triangle;
    use scioxide_core::{AdjointOperator, ExecutionContext, LinearOperator, Preconditioner};
    use scioxide_linalg::{LinalgWorkspace, MatrixView};
    let storage = [0.0, 99.0, 2.0, 99.0, 2.0, 99.0, 0.0];
    let view = MatrixView::from_slice_with_strides(2, 2, &storage, 4, 2).unwrap();
    let f = LdlDecomposition::factor(view, scioxide_linalg::HermitianInput::Exact).unwrap();
    let mut w = LinalgWorkspace::new();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::factor(
                2,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                2,
                2,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    let mut out = f.clone();
    let wrong = Matrix::from_vec(2, 2, vec![0.0, 3.0, 2.0, 0.0]).unwrap();
    assert!(
        LdlDecomposition::factor_into(
            wrong.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut out,
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, f);
    let ignored = Matrix::from_vec(2, 2, vec![0.0, f64::NAN, 2.0, 0.0]).unwrap();
    let lower = LdlDecomposition::factor(
        &ignored,
        scioxide_linalg::HermitianInput::Triangle(Triangle::Lower),
    )
    .unwrap();
    assert_eq!(lower.diagonal(), f.diagonal());
    let upper = LdlDecomposition::factor(
        &Matrix::from_vec(2, 2, vec![0.0, 2.0, f64::NAN, 0.0]).unwrap(),
        scioxide_linalg::HermitianInput::Triangle(Triangle::Upper),
    )
    .unwrap();
    assert_eq!(upper.diagonal(), f.diagonal());
    let rhs = Matrix::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    let solution = f.solve_matrix(&rhs).unwrap();
    assert_eq!(solution.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
    let inverse = f.inverse_operator();
    let mut scratch = vec![0.0; inverse.apply_workspace_len().unwrap()];
    let mut x = [7.0; 2];
    inverse
        .apply_into(&[2.0, 6.0], &mut x, &mut scratch)
        .unwrap();
    assert_eq!(x, [3.0, 1.0]);
    inverse
        .apply_adjoint_into(&[2.0, 6.0], &mut x, &mut scratch)
        .unwrap();
    assert_eq!(x, [3.0, 1.0]);
    inverse
        .precondition_into(&[2.0, 6.0], &mut x, &mut scratch)
        .unwrap();
    assert!(
        inverse
            .apply_into(&[f64::NAN, 1.0], &mut x, &mut scratch)
            .is_err()
    );
    assert_eq!(x, [3.0, 1.0]);
    for error in [
        inverse
            .apply_into(&[2.0, 6.0], &mut x, &mut [])
            .unwrap_err(),
        inverse
            .apply_into(&[2.0], &mut x, &mut scratch)
            .unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }
    assert_eq!(x, [3.0, 1.0]);
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert!(
        LdlDecomposition::factor_into(
            view,
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut out,
            &mut w,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(out, f);
    let empty = LdlDecomposition::<f64>::empty();
    assert_eq!(
        empty.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
        scioxide_sparse::LdlInertia::default()
    );
    assert!(empty.solve(&Vector::empty()).unwrap().is_empty());
    assert!(empty.condition_estimate().is_err());
    assert!(
        LdlDecomposition::factor(
            &Matrix::<f64>::zeros(2, 3).unwrap(),
            scioxide_linalg::HermitianInput::Exact
        )
        .is_err()
    );
    assert!(
        LdlDecomposition::factor(
            &Matrix::from_vec(1, 1, vec![Complex::new(1.0_f64, 1.0)]).unwrap(),
            scioxide_linalg::HermitianInput::Exact
        )
        .is_err()
    );
}

#[test]
fn local_pivot_budget_preserves_factor_and_next_child_can_solve() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    use scioxide_linalg::LinalgWorkspace;
    let a = Matrix::from_vec(3, 3, vec![1.0_f64, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
    let mut f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let before = f.clone();
    let mut w = LinalgWorkspace::new();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::factor(
                3,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                3,
                1,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("linalg", "ldl", 1),
        3,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(
        LdlDecomposition::factor_into(
            a.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
            &mut f,
            &mut w,
            &mut context
        )
        .is_err()
    );
    assert_eq!(f, before);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
    let rhs = Vector::from_vec(vec![1.0; 3]);
    let mut out = Vector::from_vec(vec![8.0; 3]);
    f.solve_into(
        rhs.view(),
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        out.view_mut(),
        &mut w,
        &mut context,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(out.as_slice(), &[1.0, -0.5, 1.0 / 3.0]);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 2);
}

#[cfg(feature = "std")]
#[test]
fn executor_matches_sequential_factors_and_multiple_rhs() {
    use scioxide_core::{ExecutionContext, ThreadExecutor};
    use scioxide_linalg::LinalgWorkspace;
    let n = 12;
    let a = Matrix::<f64>::from_vec(
        n,
        n,
        (0..n * n)
            .map(|p| {
                if p / n == p % n {
                    if p / n % 2 == 0 { 30.0 } else { -30.0 }
                } else {
                    ((p / n + p % n) % 5) as f64
                }
            })
            .collect(),
    )
    .unwrap();
    let expected = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    let execution = ExecutionContext::new(&pool).with_processor(&scioxide_core::StdProcessor);
    let mut f = scioxide_linalg::LdlDecomposition::factor(
        scioxide_linalg::MatrixView::from_slice_with_strides(
            n,
            n,
            &[<f64 as scioxide_core::AlgebraicScalar>::ZERO],
            0,
            0,
        )
        .unwrap(),
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let mut w = LinalgWorkspace::new();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::factor(
                n,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    LdlDecomposition::factor_into(
        a.view(),
        scioxide_linalg::HermitianInput::Exact,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        &mut f,
        &mut w,
        execution,
    )
    .unwrap();
    assert_eq!(f, expected);
    let rhs = Matrix::from_vec(n, 5, (0..n * 5).map(|i| i as f64).collect()).unwrap();
    let expected = f.solve_matrix(&rhs).unwrap();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                n,
                5,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    let mut output = Matrix::zeros(n, 5).unwrap();
    f.solve_matrix_into(
        rhs.view(),
        &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
        output.view_mut(),
        &mut w,
        execution,
    )
    .and_then(|report| {
        if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
            Ok(())
        } else {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    })
    .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn cloned_factor_grows_transactionally() {
    let options = scioxide_linalg::LinalgExecutionOptions::new();
    let small = Matrix::<f64>::identity(2).unwrap();
    let original =
        LdlDecomposition::factor(&small, scioxide_linalg::HermitianInput::Exact).unwrap();
    let mut copy = original.clone();
    let larger = Matrix::<f64>::identity(4).unwrap();
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    LdlDecomposition::factor_into(
        &larger,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut copy,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(copy.order(), 4);
    assert_eq!(original.order(), 2);
    let before = copy.clone();
    let malformed = Matrix::from_vec(5, 5, vec![f64::NAN; 25]).unwrap();
    assert!(
        LdlDecomposition::factor_into(
            &malformed,
            scioxide_linalg::HermitianInput::Exact,
            &options,
            &mut copy,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(copy, before);
}

#[test]
fn omitted_executor_work_is_an_error_without_publication() {
    struct Omitted;
    impl scioxide_core::Executor for Omitted {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: scioxide_core::ExecutionContext<'_>,
            _: scioxide_core::ExecutionPlan,
            _: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            Ok(())
        }
    }
    let a = Matrix::<f64>::identity(4).unwrap();
    let mut f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let before = f.clone();
    let mut w = scioxide_linalg::LinalgWorkspace::new();
    w.try_reserve(
        &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::factor(
                4,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            ),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        LdlDecomposition::factor_into(
            a.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut f,
            &mut w,
            scioxide_core::ExecutionContext::new(&Omitted)
        )
        .is_err()
    );
    assert_eq!(f, before);
}

fn complex_extreme<S: NumericScalar>(scales: &[f64]) {
    for &scale in scales {
        let b = scalar::<S>(scale, scale);
        let a = Matrix::from_vec(2, 2, vec![S::ZERO, b.conj(), b, S::ZERO]).unwrap();
        let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
        let x = f.solve(&Vector::from_vec(vec![b.conj(), b])).unwrap();
        for &value in x.as_slice() {
            assert!(
                (value - S::ONE).abs()
                    <= S::Real::EPSILON * S::Real::try_from_usize_exact(8).unwrap()
            );
        }
        assert_eq!(
            f.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
            scioxide_sparse::LdlInertia::new(1, 1, 0).unwrap()
        );
        let condition = f.condition_estimate().unwrap().reciprocal().value();
        assert!(
            (condition - S::Real::ONE).abs()
                <= S::Real::EPSILON * S::Real::try_from_usize_exact(64).unwrap()
        );
    }
}
#[test]
fn complex_blocks_across_the_exponent_range() {
    complex_extreme::<Complex<f32>>(&[1e-37, 1e37]);
    complex_extreme::<Complex<f64>>(&[1e-307, 1.5e308]);
}

#[test]
fn zero_order_rhs_shapes_do_not_overflow_workspace_queries() {
    assert_eq!(
        LdlDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                0,
                usize::MAX,
                &scioxide_linalg::LinalgExecutionOptions::new()
            )
        )
        .unwrap()
        .scalar_len()
        .unwrap(),
        0
    );
    let rhs = Matrix::<f64>::zeros(0, usize::MAX).unwrap();
    let output = LdlDecomposition::empty().solve_matrix(&rhs).unwrap();
    assert_eq!((output.rows(), output.cols()), (0, usize::MAX));
}

#[test]
fn larger_reservations_support_smaller_factors_and_solves() {
    let mut factor = scioxide_linalg::LdlDecomposition::factor(
        scioxide_linalg::MatrixView::from_slice_with_strides(
            12,
            12,
            &[<f64 as scioxide_core::AlgebraicScalar>::ZERO],
            0,
            0,
        )
        .unwrap(),
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    workspace
        .try_reserve(
            &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    12,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    workspace
        .try_reserve(
            &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                    12,
                    3,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    for order in [8, 3, 7, 1, 0] {
        let matrix = Matrix::<f64>::identity(order).unwrap();
        LdlDecomposition::factor_into(
            matrix.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut factor,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            factor,
            LdlDecomposition::factor(&matrix, scioxide_linalg::HermitianInput::Exact).unwrap()
        );
        let rhs = Vector::from_vec(vec![1.0; order]);
        let mut output = Vector::zeros(order).unwrap();
        factor
            .solve_into(
                rhs.view(),
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                output.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .and_then(|report| {
                if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                    Ok(())
                } else {
                    Err(scioxide_core::SciOxideError::singular(None))
                }
            })
            .unwrap();
        assert_eq!(output, rhs);
    }
}

fn check_blocked<S: NumericScalar>(a: &Matrix<S>, tolerance: f64) -> LdlDecomposition<S> {
    let f = LdlDecomposition::factor(a, scioxide_linalg::HermitianInput::Exact).unwrap();
    reconstruct_blocked(a, &f, tolerance);
    #[cfg(feature = "std")]
    {
        static PROCESSOR: scioxide_core::StdProcessor = scioxide_core::StdProcessor;
        let mut native = scioxide_linalg::LdlDecomposition::factor(
            scioxide_linalg::MatrixView::from_slice_with_strides(
                a.rows(),
                a.rows(),
                &[<S as scioxide_core::AlgebraicScalar>::ZERO],
                0,
                0,
            )
            .unwrap(),
            scioxide_linalg::HermitianInput::Exact,
        )
        .unwrap();
        let mut workspace = scioxide_linalg::LinalgWorkspace::new();
        workspace
            .try_reserve(
                &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                    &scioxide_linalg::LdlWorkspaceRequest::factor(
                        a.rows(),
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )
                .unwrap(),
            )
            .unwrap();
        LdlDecomposition::factor_into(
            a.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut native,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential().with_processor(&PROCESSOR),
        )
        .unwrap();
        reconstruct_blocked(a, &native, tolerance);
        assert_eq!(
            native.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
            f.inertia(scioxide_linalg::RankPolicy::Exact).unwrap()
        );
    }
    f
}

fn reconstruct_blocked<S: NumericScalar>(a: &Matrix<S>, f: &LdlDecomposition<S>, tolerance: f64) {
    let l = f.l_matrix().view();
    let mut error = S::Real::ZERO;
    let mut scale = S::Real::ZERO;
    for i in 0..a.rows() {
        for j in 0..a.rows() {
            let value = f.pivot_blocks().fold(S::ZERO, |sum, block| {
                let k = block.start();
                let first = l.get(i, k).unwrap() * f.diagonal()[k] * l.get(j, k).unwrap().conj();
                if block.size() == 1 {
                    sum + first
                } else {
                    sum + first
                        + l.get(i, k).unwrap()
                            * f.off_diagonal()[k].conj()
                            * l.get(j, k + 1).unwrap().conj()
                        + l.get(i, k + 1).unwrap()
                            * f.off_diagonal()[k]
                            * l.get(j, k).unwrap().conj()
                        + l.get(i, k + 1).unwrap()
                            * f.diagonal()[k + 1]
                            * l.get(j, k + 1).unwrap().conj()
                }
            });
            let expected = a
                .view()
                .get(f.permutation()[i], f.permutation()[j])
                .unwrap();
            error = error.max((value - expected).abs());
            scale = scale.max(expected.abs());
        }
    }
    assert!(
        error <= S::Real::try_from_f64_lossy(tolerance).unwrap() * scale.max(S::Real::ONE),
        "blocked reconstruction {error:?}, scale {scale:?}"
    );
}

fn blocked_pivots<S: NumericScalar>() {
    let n = 97;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                if i == j {
                    scalar::<S>(
                        if [31, 32, 63, 95].contains(&i) {
                            0.0
                        } else if i.is_multiple_of(2) {
                            200.0
                        } else {
                            -200.0
                        },
                        0.0,
                    )
                } else if [(31, 32), (32, 31), (63, 95), (95, 63)].contains(&(i, j)) {
                    scalar::<S>(300.0, if i > j { 20.0 } else { -20.0 })
                } else {
                    scalar::<S>(
                        ((i + j) % 7) as f64 / 128.0,
                        if i > j { 0.015625 } else { -0.015625 },
                    )
                }
            })
            .collect(),
    )
    .unwrap();
    let tol = if S::Real::EPSILON > S::Real::try_from_f64_lossy(1e-10).unwrap() {
        2e-5
    } else {
        5e-14
    };
    let factor = check_blocked(&a, tol);
    assert!(
        factor
            .pivot_blocks()
            .filter(|block| block.size() == 2)
            .count()
            >= 2
    );
    assert!(
        factor
            .permutation()
            .iter()
            .enumerate()
            .any(|(i, &p)| i != p)
    );
    let expected: Vec<_> = (0..n)
        .map(|i| scalar::<S>((i % 5) as f64 - 2.0, 0.0))
        .collect();
    let rhs = Vector::from_vec(
        (0..n)
            .map(|i| {
                (0..n).fold(S::ZERO, |v, j| {
                    v + a.view().get(i, j).unwrap() * expected[j]
                })
            })
            .collect(),
    );
    let solution = factor.solve(&rhs).unwrap();
    for (&actual, &expected) in solution.as_slice().iter().zip(&expected) {
        assert!((actual - expected).abs() <= S::Real::try_from_f64_lossy(tol * 100.0).unwrap());
    }
}

#[test]
fn blocked_rook_pivots_cross_panels_and_swap_nonlocal_columns() {
    blocked_pivots::<f32>();
    blocked_pivots::<f64>();
    blocked_pivots::<Complex<f32>>();
    blocked_pivots::<Complex<f64>>();
}

fn graded_inertia<S: NumericScalar>() {
    let n = 97;
    let low_precision = S::Real::EPSILON > S::Real::try_from_f64_lossy(1e-10).unwrap();
    let span = if low_precision { 12 } else { 120 };
    let scale = |i: usize| S::Real::ONE.scalbn((i as i32 * 17) % (2 * span + 1) - span);
    let diagonal = |i: usize| {
        if i == n - 1 {
            S::ZERO
        } else if i.is_multiple_of(2) {
            S::ONE
        } else {
            -S::ONE
        }
    };
    let t = scalar::<S>(0.125, 0.0625);
    // A = S T D Tᴴ S, with unit lower-bidiagonal T and an isolated null row.
    // Dyadic coefficients make the construction independent of a factorizer.
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                let value = if i == n - 1 || j == n - 1 {
                    S::ZERO
                } else if i == j {
                    diagonal(i)
                        + if i > 0 {
                            t * t.conj() * diagonal(i - 1)
                        } else {
                            S::ZERO
                        }
                } else if i == j + 1 {
                    t * diagonal(j)
                } else if j == i + 1 {
                    t.conj() * diagonal(i)
                } else {
                    S::ZERO
                };
                value * S::from_real(scale(i)) * S::from_real(scale(j))
            })
            .collect(),
    )
    .unwrap();
    let factor = check_blocked(&a, if low_precision { 2e-5 } else { 5e-14 });
    assert_eq!(
        factor.inertia(scioxide_linalg::RankPolicy::Exact).unwrap(),
        scioxide_sparse::LdlInertia::new(48, 48, 1).unwrap()
    );
    assert_eq!(
        factor
            .solve(&Vector::from_vec(vec![S::ONE; n]))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::Singular
    );
}

#[test]
fn blocked_graded_congruences_preserve_positive_negative_and_zero_inertia() {
    graded_inertia::<f32>();
    graded_inertia::<f64>();
    graded_inertia::<Complex<f32>>();
    graded_inertia::<Complex<f64>>();
}

#[test]
fn failed_later_panel_preserves_factor_and_workspace_can_be_reused() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let n = 97;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| if i / n == i % n { 2.0 } else { 0.015625 })
            .collect(),
    )
    .unwrap();
    let mut factor = LdlDecomposition::factor(
        &Matrix::<f64>::identity(2).unwrap(),
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();

    let before = factor.clone();
    let mut workspace = scioxide_linalg::LinalgWorkspace::new();
    workspace
        .try_reserve(
            &scioxide_linalg::LdlDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    n,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(64).unwrap());
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("linalg", "ldl", 1),
        n as u128,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(
        LdlDecomposition::factor_into(
            a.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_iteration_budget(Budget::try_from(64).unwrap()),
            &mut factor,
            &mut workspace,
            &mut context
        )
        .is_err()
    );
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 64);
    assert_eq!(factor, before);
    LdlDecomposition::factor_into(
        a.view(),
        scioxide_linalg::HermitianInput::Exact,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        &mut factor,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    reconstruct_blocked(&a, &factor, 5e-14);
}

#[test]
fn owned_factor_storage_matches_the_published_minimum() {
    let a = Matrix::from_vec(3, 3, vec![4.0, 1.0, 0.0, 1.0, -3.0, 2.0, 0.0, 2.0, 5.0]).unwrap();
    let f = LdlDecomposition::factor(&a, scioxide_linalg::HermitianInput::Exact).unwrap();
    let minimum = LdlDecomposition::<f64>::factor_storage_bytes(3).unwrap();
    assert!(minimum > 0);
    assert!(f.retained_bytes().unwrap() >= minimum);
    assert_eq!(
        LdlDecomposition::<f64>::empty().retained_bytes().unwrap(),
        0
    );
    assert_eq!(
        LdlDecomposition::<f64>::factor_storage_bytes(usize::MAX)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
}
