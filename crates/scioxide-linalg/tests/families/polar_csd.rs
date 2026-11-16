//! Analytic and independently reconstructed tests for both factor families.
#![allow(clippy::unwrap_used)]
use crate::support::svd_cases;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{
    CsdDecomposition, LinalgExecutionOptions, Matrix, PolarDecomposition, PolarOptions, PolarSide,
    RankPolicy,
};

#[test]
fn polar_zero_diagonals_across_splits_and_leaves() {
    fn check<S: NumericScalar>() {
        svd_cases::zero_diagonals::<S>(|name, a| {
            let bound = tolerance::<S>(a.rows().max(a.cols())) * 32.0;
            for side in [PolarSide::Left, PolarSide::Right] {
                let factor = PolarDecomposition::factor_with_options(
                    &a,
                    side,
                    &PolarOptions::new(RankPolicy::DimensionScaledEpsilon),
                )
                .unwrap();
                let u = factor.u_matrix();
                let h = factor.h_matrix();
                let reconstructed = if side == PolarSide::Right {
                    u.view().mul_mat(h).unwrap()
                } else {
                    h.view().mul_mat(u).unwrap()
                };
                close(&a, &reconstructed, bound);
                close(h, &adjoint(h), 0.0);
                if a.rows() >= a.cols() {
                    unitary(u);
                } else {
                    unitary(&adjoint(u));
                }
                let gram = if side == PolarSide::Right {
                    adjoint(&a).view().mul_mat(&a).unwrap()
                } else {
                    a.view().mul_mat(&adjoint(&a)).unwrap()
                };
                close(&h.view().mul_mat(h).unwrap(), &gram, bound);
                assert!(factor.report().unwrap().completion_nonunique(), "{name}");
                // H is the positive square root: test its eigenvalues as well
                // as H^2, which alone would also admit a negative root.
                let eigen = scioxide_linalg::HermitianEigenvaluesResult::compute(
                    h,
                    scioxide_linalg::HermitianInput::Exact,
                )
                .unwrap();
                assert!(
                    eigen
                        .eigenvalues()
                        .iter()
                        .all(|value| value.into_f64() >= -bound)
                );
            }
        });
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
fn real<S: NumericScalar>(x: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(x).unwrap())
}
fn adjoint<S: NumericScalar>(a: &Matrix<S>) -> Matrix<S> {
    Matrix::from_vec(
        a.cols(),
        a.rows(),
        (0..a.cols() * a.rows())
            .map(|k| a.view().get(k % a.rows(), k / a.rows()).unwrap().conj())
            .collect(),
    )
    .unwrap()
}
fn tolerance<S: NumericScalar>(n: usize) -> f64 {
    32.0 * S::Real::EPSILON.into_f64() * n.max(1) as f64
}
use crate::support::comparisons::matrix_within as close;
fn unitary<S: NumericScalar>(a: &Matrix<S>) {
    close(
        &adjoint(a).view().mul_mat(a).unwrap(),
        &Matrix::identity(a.cols()).unwrap(),
        tolerance::<S>(a.cols()),
    );
}
fn rotation<S: NumericScalar>(n: usize, seed: usize) -> Matrix<S> {
    let mut a = Matrix::<S>::identity(n).unwrap();
    for round in 0..3 {
        for i in 0..n {
            for j in i + 1..n {
                let t = (seed + round * 13 + i * 7 + j * 3) as f64 * 0.19;
                let c = real::<S>(t.cos());
                let mut s = real::<S>(t.sin());
                if core::mem::size_of::<S>() > core::mem::size_of::<S::Real>() {
                    s = S::from_components(
                        S::Real::ZERO,
                        S::Real::try_from_f64_lossy(t.sin()).unwrap(),
                    );
                }
                for row in 0..n {
                    let x = a.view().get(row, i).unwrap();
                    let y = a.view().get(row, j).unwrap();
                    a.view_mut().set(row, i, c * x - s.conj() * y).unwrap();
                    a.view_mut().set(row, j, s * x + c * y).unwrap();
                }
            }
        }
    }
    a
}
fn block<S: NumericScalar>(a: &Matrix<S>, b: &Matrix<S>) -> Matrix<S> {
    let mut out = Matrix::zeros(a.rows() + b.rows(), a.cols() + b.cols()).unwrap();
    for (m, offset) in [(a, 0), (b, a.rows())] {
        for i in 0..m.rows() {
            for j in 0..m.cols() {
                out.view_mut()
                    .set(offset + i, offset + j, m.view().get(i, j).unwrap())
                    .unwrap();
            }
        }
    }
    out
}
fn polar_cases<S: NumericScalar>() {
    for (m, n) in [
        (4, 4),
        (7, 3),
        (3, 7),
        (39, 33),
        (33, 39),
        (0, 3),
        (3, 0),
        (0, 0),
    ] {
        for rank_loss in [false, true] {
            let mut d = Matrix::<S>::zeros(m, n).unwrap();
            for i in 0..m.min(n) {
                d.view_mut()
                    .set(
                        i,
                        i,
                        real::<S>(if rank_loss && i + 1 == m.min(n) {
                            0.0
                        } else {
                            // Keep the larger constructed fixtures at unit spectral
                            // scale: their existing H^2 bound is an absolute bound.
                            (i + 1) as f64 / if m.min(n) > 32 { m.min(n) as f64 } else { 1.0 }
                        }),
                    )
                    .unwrap();
            }
            let l = rotation::<S>(m, 3);
            let v = rotation::<S>(n, 8);
            let a = l
                .view()
                .mul_mat(&d)
                .unwrap()
                .view()
                .mul_mat(&adjoint(&v))
                .unwrap();
            for side in [PolarSide::Left, PolarSide::Right] {
                let f = PolarDecomposition::factor_with_options(
                    a.view(),
                    side,
                    &(PolarOptions::new(RankPolicy::DimensionScaledEpsilon)),
                )
                .unwrap();
                let reconstructed = if side == PolarSide::Right {
                    f.u_matrix().view().mul_mat(f.h_matrix()).unwrap()
                } else {
                    f.h_matrix().view().mul_mat(f.u_matrix()).unwrap()
                };
                close(
                    &a,
                    &reconstructed,
                    tolerance::<S>(m.max(n)) * m.min(n).max(1) as f64,
                );
                if m >= n {
                    unitary(f.u_matrix());
                } else {
                    unitary(&adjoint(f.u_matrix()));
                }
                close(f.h_matrix(), &adjoint(f.h_matrix()), 0.0);
                let expected = if side == PolarSide::Right {
                    adjoint(&d).view().mul_mat(&d).unwrap()
                } else {
                    d.view().mul_mat(&adjoint(&d)).unwrap()
                };
                let basis = if side == PolarSide::Right { &v } else { &l };
                let expected = basis
                    .view()
                    .mul_mat(&expected)
                    .unwrap()
                    .view()
                    .mul_mat(&adjoint(basis))
                    .unwrap();
                close(
                    &f.h_matrix().view().mul_mat(f.h_matrix()).unwrap(),
                    &expected,
                    tolerance::<S>(m.max(n)) * 32.0,
                );
                // In the known singular basis H is diag(sigma), establishing positivity.
                let diagonal = adjoint(basis)
                    .view()
                    .mul_mat(f.h_matrix())
                    .unwrap()
                    .view()
                    .mul_mat(basis)
                    .unwrap();
                for i in 0..diagonal.rows() {
                    assert!(
                        diagonal
                            .view()
                            .get(i, i)
                            .unwrap()
                            .real_component()
                            .into_f64()
                            >= -tolerance::<S>(m.max(n))
                    );
                }
                assert_eq!(
                    f.report().unwrap().completion_nonunique(),
                    rank_loss && m.min(n) > 0
                );
            }
        }
    }
}
#[test]
fn polar_shapes_rank_loss_and_all_scalars() {
    polar_cases::<f32>();
    polar_cases::<f64>();
    polar_cases::<Complex<f32>>();
    polar_cases::<Complex<f64>>();
}
fn verify_csd<S: NumericScalar>(x: &Matrix<S>, f: &CsdDecomposition<S>) {
    for u in [f.u1_matrix(), f.u2_matrix(), f.v1_matrix(), f.v2_matrix()] {
        unitary(u);
    }
    let u = block(f.u1_matrix(), f.u2_matrix());
    let v = block(f.v1_matrix(), f.v2_matrix());
    let recovered = u
        .view()
        .mul_mat(f.cs_matrix())
        .unwrap()
        .view()
        .mul_mat(&adjoint(&v))
        .unwrap();
    close(x, &recovered, tolerance::<S>(x.rows()));
    unitary(f.cs_matrix());
    assert!(f.angles().windows(2).all(|v| v[0] <= v[1]));
}
fn csd_cases<S: NumericScalar>() {
    for n in 0..9 {
        let x = rotation::<S>(n, 7);
        for p in 0..=n {
            for q in 0..=n {
                let result = CsdDecomposition::factor_with_options(
                    x.view(),
                    p,
                    q,
                    &(LinalgExecutionOptions::new()),
                );
                assert!(result.is_ok(), "{n},{p},{q}: {result:?}");
                let f = result.unwrap();
                assert_eq!(f.angles().len(), p.min(n - p).min(q).min(n - q));
                verify_csd(&x, &f);
            }
        }
    }
}
#[test]
fn csd_every_partition_all_scalars() {
    csd_cases::<f32>();
    csd_cases::<f64>();
    csd_cases::<Complex<f32>>();
    csd_cases::<Complex<f64>>();
}
fn analytic<S: NumericScalar>() {
    let mut x = Matrix::<S>::zeros(8, 8).unwrap();
    let angles = [0.0_f64, 0.37, 0.37, core::f64::consts::FRAC_PI_2];
    for (i, t) in angles.iter().enumerate() {
        let (c, s) = if i == 3 {
            (0.0, 1.0)
        } else {
            (t.cos(), t.sin())
        };
        x.view_mut().set(i, i, real::<S>(c)).unwrap();
        x.view_mut().set(i + 4, i + 4, real::<S>(c)).unwrap();
        x.view_mut().set(i, i + 4, real::<S>(-s)).unwrap();
        x.view_mut().set(i + 4, i, real::<S>(s)).unwrap();
    }
    for seed in 0..5 {
        let u1 = rotation::<S>(4, seed);
        let u2 = rotation::<S>(4, seed + 5);
        let v1 = rotation::<S>(4, seed + 9);
        let v2 = rotation::<S>(4, seed + 12);
        let changed = block(&u1, &u2)
            .view()
            .mul_mat(&x)
            .unwrap()
            .view()
            .mul_mat(&adjoint(&block(&v1, &v2)))
            .unwrap();
        let f = CsdDecomposition::factor_with_options(
            changed.view(),
            4,
            4,
            &(LinalgExecutionOptions::new()),
        )
        .unwrap();
        verify_csd(&changed, &f);
        for (&got, &want) in f.angles().iter().zip(&angles) {
            assert!((got.into_f64() - want).abs() < tolerance::<S>(8));
        }
        // Repeated-angle right invariant subspace is independent of its chosen basis.
        let projector = |v: &Matrix<S>| {
            Matrix::from_vec(
                4,
                4,
                (0..16)
                    .map(|k| {
                        (1..3).fold(S::ZERO, |s, j| {
                            s + v.view().get(k / 4, j).unwrap()
                                * v.view().get(k % 4, j).unwrap().conj()
                        })
                    })
                    .collect(),
            )
            .unwrap()
        };
        close(
            &projector(f.v1_matrix()),
            &projector(&v1),
            tolerance::<S>(8),
        );
    }
}
#[test]
fn prescribed_endpoints_repeated_angles_and_basis_invariance() {
    analytic::<f32>();
    analytic::<f64>();
    analytic::<Complex<f32>>();
    analytic::<Complex<f64>>();
}
#[test]
fn polar_extreme_scales_and_no_rank_truncation() {
    for scale in [1e-300, 1e300] {
        let a = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, scale * 0.25]).unwrap();
        let f = PolarDecomposition::factor_with_options(
            a.view(),
            scioxide_linalg::PolarSide::Right,
            &(PolarOptions::new(RankPolicy::DimensionScaledEpsilon)),
        )
        .unwrap();
        assert!((f.h_matrix().view().get(0, 0).unwrap() / scale - 1.0_f64).abs() < 1e-14);
    }
    let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-18]).unwrap();
    let f = PolarDecomposition::factor_with_options(
        a.view(),
        scioxide_linalg::PolarSide::Right,
        &(PolarOptions::new(RankPolicy::DimensionScaledEpsilon)),
    )
    .unwrap();
    assert_eq!(f.report().unwrap().rank().rank(), 1);
    assert!((f.h_matrix().view().get(1, 1).unwrap() / 1e-18 - 1.0_f64).abs() < 1e-14);
}

use scioxide_linalg::LinalgWorkspace;

#[test]
fn failures_preserve_factors_and_allow_retry() {
    use scioxide_core::{
        Budget, BudgetKind, CancellationToken, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let x = rotation::<f64>(6, 7);
    let opts = PolarOptions::new(RankPolicy::Exact);
    let mut f = PolarDecomposition::factor_with_options(
        x.view(),
        scioxide_linalg::PolarSide::Right,
        &(opts),
    )
    .unwrap();
    let mut c =
        CsdDecomposition::factor_with_options(x.view(), 3, 3, &(LinalgExecutionOptions::new()))
            .unwrap();
    let mut pw = LinalgWorkspace::new();
    let mut cw = LinalgWorkspace::new();
    let fu = f.u_matrix().clone();
    let ch = c.cs_matrix().clone();
    let cr = c.report().unwrap().residual();
    let mut bad = x.clone();
    bad.view_mut().set(0, 0, f64::NAN).unwrap();
    assert!(matches!(
        (PolarDecomposition::factor_into(
            bad.view(),
            scioxide_linalg::PolarSide::Right,
            &(opts),
            &mut f,
            &mut pw,
            scioxide_core::ExecutionContext::sequential_native()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(matches!(
        (CsdDecomposition::factor_into(
            bad.view(),
            3,
            3,
            &(LinalgExecutionOptions::new()),
            &mut c,
            &mut cw,
            scioxide_core::ExecutionContext::sequential_native()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    bad.view_mut().set(0, 0, 3.0).unwrap();
    assert!(matches!(
        (CsdDecomposition::factor_into(
            bad.view(),
            3,
            3,
            &(LinalgExecutionOptions::new()),
            &mut c,
            &mut cw,
            scioxide_core::ExecutionContext::sequential_native()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (CsdDecomposition::factor_into(
            x.view(),
            7,
            3,
            &(LinalgExecutionOptions::new()),
            &mut c,
            &mut cw,
            scioxide_core::ExecutionContext::sequential_native()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let rectangular = Matrix::<f64>::zeros(3, 4).unwrap();
    assert!(matches!(
        (CsdDecomposition::factor_with_options(
            rectangular.view(),
            1,
            1,
            &(LinalgExecutionOptions::new())
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (CsdDecomposition::factor_into(
            x.view(),
            3,
            3,
            &(LinalgExecutionOptions::new().with_iteration_budget(Budget::try_from(2).unwrap())),
            &mut c,
            &mut cw,
            scioxide_core::ExecutionContext::sequential_native()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() > 0
    ));
    let token = CancellationToken::new();
    token.cancel();
    let context = ExecutionContext::sequential_native().with_cancellation(&token);
    assert!(
        PolarDecomposition::factor_into(
            x.view(),
            scioxide_linalg::PolarSide::Right,
            &(opts),
            &mut f,
            &mut pw,
            context
        )
        .is_err()
    );
    assert!(
        CsdDecomposition::factor_into(
            x.view(),
            3,
            3,
            &(LinalgExecutionOptions::new()),
            &mut c,
            &mut cw,
            context
        )
        .is_err()
    );
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(665),
        scioxide_core::algorithm_id!("linalg", "csd", 1),
        6,
        0,
    );
    let mut solve = state
        .begin_solve(ExecutionContext::sequential_native(), id, &mut budgets)
        .unwrap();
    assert!(matches!(
        (CsdDecomposition::factor_into(
            x.view(),
            3,
            3,
            &LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
            &mut c,
            &mut cw,
            &mut solve
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } if diagnostics.counts().own().iterations() > 0
    ));
    let before_iterations = solve.counts().own().iterations();
    PolarDecomposition::factor_into(
        x.view(),
        PolarSide::Right,
        &opts,
        &mut f,
        &mut pw,
        &mut solve,
    )
    .unwrap();
    assert_eq!(solve.counts().own().iterations(), before_iterations);
    assert!(solve.counts().nested().iterations() >= 1);
    assert!(
        PolarDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::PolarWorkspaceRequest::new(usize::MAX, 2, PolarSide::Right, &opts)
        )
        .is_err()
    );
    assert!(
        CsdDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::CsdWorkspaceRequest::new(
                usize::MAX,
                1,
                1,
                &LinalgExecutionOptions::new()
            )
        )
        .is_err()
    );
    assert_eq!(f.u_matrix(), &fu);
    assert_eq!(c.cs_matrix(), &ch);
    assert_eq!(c.report().unwrap().residual(), cr);
    PolarDecomposition::factor_into(
        x.view(),
        scioxide_linalg::PolarSide::Right,
        &(opts),
        &mut f,
        &mut pw,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    CsdDecomposition::factor_into(
        x.view(),
        3,
        3,
        &(LinalgExecutionOptions::new()),
        &mut c,
        &mut cw,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    verify_csd(&x, &c);
    // The subnormal entry is below 2^-1074 relative to MAX and flushes: the
    // normalized input has rank one, and H keeps only the representable entry.
    let underflow = Matrix::from_vec(2, 2, vec![f64::MAX, 0.0, 0.0, f64::from_bits(1)]).unwrap();
    let flushed =
        PolarDecomposition::factor(underflow.view(), scioxide_linalg::PolarSide::Right).unwrap();
    let report = flushed.report().unwrap();
    assert_eq!(report.rank().rank(), 1);
    assert!(report.completion_nonunique());
    assert_eq!(flushed.h_matrix().as_slice(), &[f64::MAX, 0.0, 0.0, 0.0]);
    unitary(flushed.u_matrix());
}

#[test]
fn polar_flushes_entries_below_the_normalized_range() {
    // cond(A) = 1, but 1e-200 / 1e200 underflows past the subnormal range.
    let a = Matrix::from_vec(2, 2, vec![1e200, 1e-200, 0.0, 1e200]).unwrap();
    for side in [PolarSide::Right, PolarSide::Left] {
        let f = PolarDecomposition::factor_with_options(
            a.view(),
            side,
            &PolarOptions::new(RankPolicy::DimensionScaledEpsilon),
        )
        .unwrap();
        let report = f.report().unwrap();
        assert!(report.rank().is_full_rank());
        assert!(!report.completion_nonunique());
        assert_eq!(f.input_scale(), 1e200);
        let (u, h) = (f.u_matrix().as_slice(), f.h_matrix().as_slice());
        for (index, (&u, &h)) in u.iter().zip(h).enumerate() {
            let diagonal = index % 3 == 0;
            assert!((u - f64::from(u8::from(diagonal))).abs() <= 4.0 * f64::EPSILON);
            if diagonal {
                assert!((h / 1e200 - 1.0).abs() <= 4.0 * f64::EPSILON);
            } else {
                assert!(h.abs() <= 4.0 * f64::EPSILON * 1e200);
            }
        }
    }
}

fn tiny_endpoint<S: NumericScalar>() {
    let tiny = S::Real::ZERO.next_after(S::Real::ONE);
    let x = Matrix::from_vec(
        2,
        2,
        vec![S::ONE, -S::from_real(tiny), S::from_real(tiny), S::ONE],
    )
    .unwrap();
    let f = CsdDecomposition::factor_with_options(x.view(), 1, 1, &(LinalgExecutionOptions::new()))
        .unwrap();
    assert_eq!(f.sines()[0], tiny);
    assert_eq!(f.cosines()[0], S::Real::ONE);
    verify_csd(&x, &f);
    let a = Matrix::from_vec(1, 1, vec![S::from_real(tiny)]).unwrap();
    let f = PolarDecomposition::factor_with_options(
        a.view(),
        scioxide_linalg::PolarSide::Right,
        &(PolarOptions::new(RankPolicy::Exact)),
    )
    .unwrap();
    assert_eq!(f.h_matrix().view().get(0, 0).unwrap(), S::from_real(tiny));
}
#[test]
fn subnormal_endpoints_are_not_recovered_by_subtraction() {
    tiny_endpoint::<f32>();
    tiny_endpoint::<f64>();
    tiny_endpoint::<Complex<f32>>();
    tiny_endpoint::<Complex<f64>>();
}

#[test]
fn polar_gates_tiny_scales_in_normalized_units() {
    // H in input units has subnormal entries; the publication gate must not
    // reject the rounding that restoring those units introduces.
    let d = f64::from_bits(1 << 10);
    let a = Matrix::from_vec(2, 2, vec![d, d, d, -d]).unwrap();
    let f = PolarDecomposition::factor_with_options(
        a.view(),
        PolarSide::Right,
        &PolarOptions::new(RankPolicy::DimensionScaledEpsilon),
    )
    .unwrap();
    assert!(f.report().unwrap().rank().is_full_rank());
    unitary(f.u_matrix());
}

#[test]
fn csd_of_wide_row_blocks_converges_in_few_sweeps() {
    // The first block column has more columns than either row block has rows.
    let x = rotation::<f64>(40, 3);
    for (p, q) in [(20, 40), (20, 30), (12, 30)] {
        let f =
            CsdDecomposition::factor_with_options(x.view(), p, q, &LinalgExecutionOptions::new())
                .unwrap();
        verify_csd(&x, &f);
        assert!(f.report().unwrap().sweeps() <= 3);
        assert_eq!(f.angles().len(), p.min(40 - p).min(q).min(40 - q));
    }
}
