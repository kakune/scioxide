#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar};
use scioxide_linalg::{GsvdDecomposition, GsvdOptions, LinalgWorkspace, Matrix, RankPolicy};
fn matrix<S: NumericScalar>(m: usize, n: usize, seed: usize) -> Matrix<S> {
    Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|i| {
                let re = ((i * 17 + seed * 13) % 29) as f64 / 19.0 - 0.7;
                let im = if core::mem::size_of::<S>() > core::mem::size_of::<S::Real>() {
                    ((i * 7 + seed * 3) % 13) as f64 / 17.0
                } else {
                    0.0
                };
                S::from_components(
                    S::Real::try_from_f64_lossy(re).unwrap(),
                    S::Real::try_from_f64_lossy(im).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap()
}
fn unitary<S: NumericScalar>(q: &Matrix<S>) {
    let tolerance = 32.0 * S::Real::EPSILON.into_f64() * (q.rows().max(1) as f64);
    for i in 0..q.cols() {
        for j in 0..q.cols() {
            let sum = (0..q.rows()).fold(S::ZERO, |s, k| {
                s + q.view().get(k, i).unwrap().conj() * q.view().get(k, j).unwrap()
            });
            assert!(
                (sum - if i == j { S::ONE } else { S::ZERO })
                    .abs()
                    .into_f64()
                    < tolerance,
                "unitarity {i} {j}: {:?}",
                sum
            );
        }
    }
}
fn verify<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>, g: &GsvdDecomposition<S>) {
    for q in [g.u_matrix(), g.v_matrix(), g.q_matrix()] {
        unitary(q);
    }
    let r = g.r_matrix();
    let rank = r.rows();
    let n = a.cols();
    let mut h = Matrix::<S>::zeros(rank, n).unwrap();
    for i in 0..rank {
        for j in 0..rank {
            if i > j {
                assert!(r.view().get(i, j).unwrap().abs().into_f64() < 1e-6);
            }
            h.view_mut()
                .set(i, n - rank + j, r.view().get(i, j).unwrap())
                .unwrap();
        }
    }
    let mut qh = Matrix::<S>::zeros(n, n).unwrap();
    for i in 0..n {
        for j in 0..n {
            qh.view_mut()
                .set(i, j, g.q_matrix().view().get(j, i).unwrap().conj())
                .unwrap();
        }
    }
    let h = h.view().mul_mat(&qh).unwrap();
    for (a, u, d) in [
        (a, g.u_matrix(), g.d1_matrix()),
        (b, g.v_matrix(), g.d2_matrix()),
    ] {
        let reconstructed = u.view().mul_mat(d).unwrap().view().mul_mat(&h).unwrap();
        let error = a
            .as_slice()
            .iter()
            .zip(reconstructed.as_slice())
            .fold(0.0_f64, |s, (&x, &y)| {
                s.hypot((x / S::from_real(g.input_scale()) - y).abs().into_f64())
            });
        let tolerance =
            128.0 * S::Real::EPSILON.into_f64() * (a.rows() + b.rows() + n).max(1) as f64;
        assert!(
            error < tolerance,
            "reconstruction {}x{} error {error} bound {tolerance}",
            a.rows(),
            n
        );
        let z = g.common_null_space();
        for i in 0..a.rows() {
            for j in 0..z.cols() {
                let sum = (0..n).fold(S::ZERO, |s, k| {
                    s + a.view().get(i, k).unwrap() / S::from_real(g.input_scale())
                        * z.get(k, j).unwrap()
                });
                assert!(sum.abs().into_f64() < tolerance);
            }
        }
    }
    for pair in g.generalized_values() {
        assert!(
            (pair.alpha() * pair.alpha() + pair.beta() * pair.beta() - S::Real::ONE)
                .abs()
                .into_f64()
                < 100.0 * S::Real::EPSILON.into_f64()
        );
    }
}
fn shapes<S: NumericScalar>() {
    for (m, p, n) in [
        (3, 3, 3),
        (7, 5, 3),
        (2, 3, 7),
        (1, 4, 4),
        (4, 1, 4),
        (8, 7, 8),
        (40, 36, 33),
        (0, 3, 4),
        (3, 0, 4),
        (0, 0, 3),
        (3, 4, 0),
        (0, 0, 0),
    ] {
        let a = matrix::<S>(m, n, 1);
        let b = matrix::<S>(p, n, 2);
        let g = GsvdDecomposition::factor_with_options(
            a.view(),
            b.view(),
            &(GsvdOptions::new(RankPolicy::DimensionScaledEpsilon)),
        )
        .unwrap();
        verify(&a, &b, &g);
    }
}
#[test]
fn all_domains_and_shapes() {
    shapes::<f32>();
    shapes::<f64>();
    shapes::<Complex<f32>>();
    shapes::<Complex<f64>>();
}
#[test]
fn analytic_endpoints_and_rank_reuse() {
    let mut a = Matrix::<f64>::zeros(4, 5).unwrap();
    let mut b = a.clone();
    let mut out = GsvdDecomposition::empty();
    let mut work = LinalgWorkspace::new();
    let options = GsvdOptions::new(RankPolicy::DimensionScaledEpsilon);
    work.try_reserve(
        &GsvdDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::GsvdWorkspaceRequest::new(4, 4, 5, &options),
        )
        .unwrap(),
    )
    .unwrap();
    for rank in [4, 2, 0, 3, 4] {
        a.as_mut_slice().fill(0.0);
        b.as_mut_slice().fill(0.0);
        for i in 0..rank {
            a.view_mut()
                .set(i, i, if i == 0 { 0.0 } else { 3.0 })
                .unwrap();
            b.view_mut()
                .set(i, i, if i == 1 { 0.0 } else { 4.0 })
                .unwrap();
        }
        GsvdDecomposition::factor_into(
            a.view(),
            b.view(),
            &(options),
            &mut out,
            &mut work,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(out.report().unwrap().rank().rank(), rank);
        verify(&a, &b, &out);
        let mut ratios: Vec<_> = out
            .generalized_values()
            .iter()
            .map(|v| {
                if v.beta() == 0.0 {
                    f64::INFINITY
                } else {
                    v.ratio().unwrap()
                }
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        if rank == 4 {
            assert_eq!(ratios[0], 0.0);
            assert!((ratios[1] - 0.75).abs() < 1e-14);
            assert_eq!(ratios[3], f64::INFINITY);
        }
    }
    let before = out.r_matrix().clone();
    b.view_mut().set(0, 1, 2.0).unwrap();
    // One iteration covers the stacked SVD and leaves no Jacobi sweep.
    assert!(
        GsvdDecomposition::factor_into(
            a.view(),
            b.view(),
            &(options.with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE)
            )),
            &mut out,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out.r_matrix(), &before);
    a.view_mut().set(0, 0, f64::NAN).unwrap();
    assert!(matches!(
        (GsvdDecomposition::factor_into(
            a.view(),
            b.view(),
            &(options),
            &mut out,
            &mut work,
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(out.r_matrix(), &before);
}
#[test]
fn extreme_common_scales_and_overflowing_ratio() {
    for scale in [1e-300, 1.0, 1e300] {
        let a = Matrix::from_vec(2, 2, vec![3.0 * scale, 0.0, 0.0, scale]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![4.0 * scale, 0.0, 0.0, scale]).unwrap();
        let g = GsvdDecomposition::factor_with_options(
            a.view(),
            b.view(),
            &(GsvdOptions::new(RankPolicy::Exact)),
        )
        .unwrap();
        verify(&a, &b, &g);
    }
}
#[test]
fn common_normalization_flushes_entries_below_the_subnormal_range() {
    // cond(A) = 1, but 1e-200 / 1e200 underflows past the subnormal range.
    let a = Matrix::from_vec(2, 2, vec![1e200, 1e-200, 0.0, 1e200]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let g = GsvdDecomposition::factor(a.view(), b.view()).unwrap();
    assert_eq!(g.input_scale(), 1e200);
    assert_eq!(g.report().unwrap().rank().rank(), 2);
    verify(&a, &b, &g);
}

fn manufactured<S: NumericScalar>() {
    let cs = [(0.0, 1.0), (0.6, 0.8), (0.6, 0.8), (1.0, 0.0)];
    let mut a = Matrix::<S>::zeros(4, 6).unwrap();
    let mut b = a.clone();
    // A common nonsingular row factor; two extra columns make the nullspace nontrivial.
    for (i, &(c, s)) in cs.iter().enumerate() {
        for j in 0..6 {
            let h = if i == j {
                2.0
            } else if j > i {
                ((i + 2 * j) % 7) as f64 / 11.0
            } else {
                0.0
            };
            a.view_mut()
                .set(
                    i,
                    j,
                    S::from_real(S::Real::try_from_f64_lossy(c * h).unwrap()),
                )
                .unwrap();
            b.view_mut()
                .set(
                    i,
                    j,
                    S::from_real(S::Real::try_from_f64_lossy(s * h).unwrap()),
                )
                .unwrap();
        }
    }
    // Independent unitary row changes, including genuine complex phases.
    let mut u = Matrix::<S>::identity(4).unwrap();
    for (i, j) in [(0, 1), (2, 3)] {
        let c = S::from_real(S::Real::try_from_f64_lossy(0.6).unwrap());
        let s = S::from_real(S::Real::try_from_f64_lossy(0.8).unwrap());
        u.view_mut().set(i, i, c).unwrap();
        u.view_mut().set(j, j, c).unwrap();
        u.view_mut().set(i, j, s).unwrap();
        u.view_mut().set(j, i, -s).unwrap();
    }
    if core::mem::size_of::<S>() > core::mem::size_of::<S::Real>() {
        for j in 0..4 {
            let value =
                u.view().get(0, j).unwrap() * S::from_components(S::Real::ZERO, S::Real::ONE);
            u.view_mut().set(0, j, value).unwrap();
        }
    }
    a = u.view().mul_mat(&a).unwrap();
    b = u
        .view()
        .adjoint()
        .try_to_matrix()
        .unwrap()
        .view()
        .mul_mat(&b)
        .unwrap();
    let options = GsvdOptions::new(RankPolicy::DimensionScaledEpsilon);
    let g = GsvdDecomposition::factor_with_options(a.view(), b.view(), &(options)).unwrap();
    verify(&a, &b, &g);
    assert_eq!(g.report().unwrap().rank().rank(), 4);
    let mut alphas: Vec<_> = g
        .generalized_values()
        .iter()
        .map(|x| x.alpha().into_f64())
        .collect();
    alphas.sort_by(f64::total_cmp);
    for (x, y) in alphas.iter().zip([0.0, 0.6, 0.6, 1.0]) {
        assert!(
            (x - y).abs() < 100.0 * S::Real::EPSILON.into_f64(),
            "{alphas:?}"
        );
    }
    // A shared, well-conditioned invertible right change preserves the GS values.
    let mut t = Matrix::<S>::identity(6).unwrap();
    t.view_mut()
        .set(
            0,
            2,
            S::from_real(S::Real::try_from_f64_lossy(0.25).unwrap()),
        )
        .unwrap();
    let aa = a.view().mul_mat(&t).unwrap();
    let bb = b.view().mul_mat(&t).unwrap();
    let changed = GsvdDecomposition::factor_with_options(aa.view(), bb.view(), &(options)).unwrap();
    verify(&aa, &bb, &changed);
    let mut changed_values: Vec<_> = changed
        .generalized_values()
        .iter()
        .map(|x| x.alpha().into_f64())
        .collect();
    changed_values.sort_by(f64::total_cmp);
    for (x, y) in alphas.iter().zip(changed_values) {
        assert!((x - y).abs() < 100.0 * S::Real::EPSILON.into_f64());
    }
}
#[test]
fn independent_known_values_and_basis_changes() {
    manufactured::<f32>();
    manufactured::<f64>();
    manufactured::<Complex<f32>>();
    manufactured::<Complex<f64>>();
}

#[test]
fn policy_diagnostics_and_strided_adjoint_inputs() {
    let a = Matrix::from_vec(2, 3, vec![1.0_f64, 0.0, 0.0, 0.0, 1e-8, 0.0]).unwrap();
    let b = Matrix::<f64>::zeros(1, 3).unwrap();
    let exact = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &(GsvdOptions::new(RankPolicy::Exact)),
    )
    .unwrap();
    let truncated = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &(GsvdOptions::new(RankPolicy::Threshold(
            scioxide_core::RankThreshold::new(1e-7, 0.0).unwrap(),
        ))),
    )
    .unwrap();
    assert_eq!(exact.report().unwrap().rank().rank(), 2);
    let report = truncated.report().unwrap();
    assert_eq!(report.rank().rank(), 1);
    assert_eq!(report.rank().threshold(), 1e-7);
    assert!((report.discarded_norm() - 1e-8).abs() < 1e-22);
    assert!((report.residual_a() - 1e-8).abs() < 1e-22);
    assert_eq!(exact.report().unwrap().condition(), report.condition());
    assert_eq!(report.retained_reciprocal_condition(), 1.0);
    let a = matrix::<Complex<f64>>(4, 3, 2);
    let b = matrix::<Complex<f64>>(5, 3, 3);
    let mut storage = Matrix::zeros(8, 6).unwrap();
    for i in 0..4 {
        for j in 0..3 {
            storage
                .view_mut()
                .set(2 * i, 2 * j, a.view().get(i, j).unwrap())
                .unwrap();
        }
    }
    let view =
        scioxide_linalg::MatrixView::from_slice_with_strides(4, 3, storage.as_slice(), 12, 2)
            .unwrap();
    let bh = b.view().adjoint().try_to_matrix().unwrap();
    let g = GsvdDecomposition::factor_with_options(
        view,
        bh.view().adjoint(),
        &(GsvdOptions::new(RankPolicy::DimensionScaledEpsilon)),
    )
    .unwrap();
    verify(&a, &b, &g);
}

#[test]
fn stage_budgets_cancellation_shape_and_range_failures() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    let a = Matrix::<f64>::identity(3).unwrap();
    let options = GsvdOptions::new(RankPolicy::Exact);
    let mut out = GsvdDecomposition::factor_with_options(a.view(), a.view(), &(options)).unwrap();
    let before = out.r_matrix().clone();
    let mut work = LinalgWorkspace::new();
    for limit in [1, 2] {
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap());
        let mut state = SolveState::new();
        let id = SolveIdentity::new(
            ProblemToken::new(666),
            scioxide_core::algorithm_id!("linalg", "gsvd", 1),
            3,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
            .unwrap();
        let result = GsvdDecomposition::factor_into(
            a.view(),
            a.view(),
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_iteration_budget(Budget::try_from(limit).unwrap()),
            ),
            &mut out,
            &mut work,
            &mut context,
        );
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(
            work.last_attempt().unwrap().counts().own().iterations(),
            limit
        );
        if limit == 1 {
            assert!(matches!(
                (result).as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() > 0
            ));
            assert_eq!(out.r_matrix(), &before);
        } else {
            result.unwrap();
            verify(&a, &a, &out);
        }
    }
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        GsvdDecomposition::factor_into(
            a.view(),
            a.view(),
            &(options),
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(out.r_matrix(), &before);
    assert!(
        GsvdDecomposition::factor_with_options(
            a.view(),
            Matrix::<f64>::zeros(3, 2).unwrap().view(),
            &(options)
        )
        .is_err()
    );
    assert!(
        GsvdDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::GsvdWorkspaceRequest::new(usize::MAX, 1, 2, &options)
        )
        .is_err()
    );
    let huge = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
    let tiny = Matrix::from_vec(1, 1, vec![f64::from_bits(1)]).unwrap();
    // The subnormal entry flushes under the common MAX scale: B / scale is zero,
    // so the pair is an infinite generalized value.
    let flushed = GsvdDecomposition::factor(huge.view(), tiny.view()).unwrap();
    let value = flushed.generalized_values()[0];
    assert_eq!((value.alpha(), value.beta()), (1.0, 0.0));
    assert!(value.ratio().is_err());
    assert_eq!(flushed.report().unwrap().rank().rank(), 1);
    // The pair is retained even when its quotient overflows.
    let unit = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let g = GsvdDecomposition::factor_with_options(unit.view(), tiny.view(), &(options)).unwrap();
    assert!(g.generalized_values()[0].beta() > 0.0);
    assert!(g.generalized_values()[0].ratio().is_err());
}

#[cfg(feature = "std")]
#[test]
fn executor_matches_sequential_and_reserves_every_buffer() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy};
    let workers = core::num::NonZeroUsize::new(2).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);
    let options = GsvdOptions::new(RankPolicy::DimensionScaledEpsilon).with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    );
    let a = matrix::<Complex<f64>>(17, 8, 5);
    let b = matrix::<Complex<f64>>(13, 8, 7);
    let expected = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &(GsvdOptions::new(RankPolicy::DimensionScaledEpsilon)),
    )
    .unwrap();
    let mut out = GsvdDecomposition::empty();
    let mut work = LinalgWorkspace::new();
    work.try_reserve(
        &GsvdDecomposition::<Complex<f64>>::workspace_requirements(
            &scioxide_linalg::GsvdWorkspaceRequest::new(17, 13, 8, &options),
        )
        .unwrap(),
    )
    .unwrap();
    GsvdDecomposition::factor_into(a.view(), b.view(), &(options), &mut out, &mut work, context)
        .unwrap();
    verify(&a, &b, &out);
    for (x, y) in expected
        .generalized_values()
        .iter()
        .zip(out.generalized_values())
    {
        assert!((x.alpha() - y.alpha()).abs() < 1e-12);
        assert!((x.beta() - y.beta()).abs() < 1e-12);
    }
}

#[test]
fn seeded_rectangular_reference() {
    for seed in 0..8 {
        for (m, p, n) in [(19, 17, 16), (3, 13, 15), (13, 3, 15), (5, 7, 16)] {
            let a = matrix::<f64>(m, n, seed);
            let b = matrix::<f64>(p, n, seed + 3);
            let g = GsvdDecomposition::factor_with_options(
                a.view(),
                b.view(),
                &(GsvdOptions::new(RankPolicy::DimensionScaledEpsilon)),
            )
            .unwrap();
            verify(&a, &b, &g);
        }
    }
}

fn subnormal_denominator<S: NumericScalar>() {
    let tiny = S::Real::ZERO.next_after(S::Real::ONE);
    let a = Matrix::from_vec(1, 1, vec![S::ONE]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![S::from_real(tiny)]).unwrap();
    let g = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &(GsvdOptions::new(RankPolicy::Exact)),
    )
    .unwrap();
    assert_eq!(g.generalized_values()[0].beta(), tiny);
    assert!(g.generalized_values()[0].ratio().is_err());
    unitary(g.u_matrix());
    unitary(g.v_matrix());
}
#[test]
fn smallest_denominator_all_domains() {
    subnormal_denominator::<f32>();
    subnormal_denominator::<f64>();
    subnormal_denominator::<Complex<f32>>();
    subnormal_denominator::<Complex<f64>>();
}

#[test]
fn wide_endpoint_roundoff_converges_without_chasing_both_grams() {
    let a = matrix::<f64>(5, 16, 1);
    let b = matrix::<f64>(7, 16, 2);
    let g = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &(GsvdOptions::new(RankPolicy::DimensionScaledEpsilon)),
    )
    .unwrap();
    verify(&a, &b, &g);
    assert!(
        g.report()
            .unwrap()
            .residual_a()
            .hypot(g.report().unwrap().residual_b())
            < 1e-12
    );
}

#[test]
fn blocks_with_fewer_rows_than_the_rank_converge_in_few_sweeps() {
    // With r = rank [A; B] above m or p, a block Gram matrix has a multiple
    // zero eigenvalue, where pairwise Jacobi alone converges only linearly.
    // `n − p` values are infinite (β = 0) and `n − m` are zero (α = 0).
    for (m, p, n) in [(40, 40, 80), (20, 50, 60)] {
        let a = crate::support::decomposition::generated_matrix::<f64>(m, n, 11);
        let b = crate::support::decomposition::generated_matrix::<f64>(p, n, 17);
        let g = GsvdDecomposition::factor_with_options(
            a.view(),
            b.view(),
            &GsvdOptions::new(RankPolicy::DimensionScaledEpsilon),
        )
        .unwrap();
        verify(&a, &b, &g);
        assert_eq!(g.r_matrix().rows(), n);
        assert!(g.report().unwrap().sweeps() <= 3);
        let count = |small: &dyn Fn(f64, f64) -> bool| {
            g.generalized_values()
                .iter()
                .filter(|value| small(value.alpha(), value.beta()))
                .count()
        };
        assert_eq!(count(&|alpha, _| alpha < 1e-12), n - m);
        assert_eq!(count(&|_, beta| beta < 1e-12), n - p);
    }
}
