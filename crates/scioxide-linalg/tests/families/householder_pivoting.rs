//! Panel boundaries and the public choice of QR/LQ pivoting.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AlgebraicScalar, Complex32, Complex64, NumericScalar, RealScalar};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting, Matrix,
};

fn boundaries<S: NumericScalar<Real = f64>>() {
    for n in [0, 1, 4, 8, 9, 31, 32, 33, 63, 64, 65] {
        for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
            let (rows, cols) = match orientation {
                HouseholderOrientation::Qr => (n + 3, n),
                HouseholderOrientation::Lq => (n, n + 3),
            };
            let data = (0..rows)
                .flat_map(|i| {
                    (0..cols).map(move |j| {
                        S::from_components(
                            ((i * 17 + j * 31) % 97) as f64 / 97.0 + if i == j { 2.0 } else { 0.0 },
                            if S::IS_REAL {
                                0.0
                            } else {
                                ((i * 11 + j * 7) % 37) as f64 / 37.0
                            },
                        )
                    })
                })
                .collect();
            let a = Matrix::from_vec(rows, cols, data).unwrap();
            for pivoting in [HouseholderPivoting::Column, HouseholderPivoting::None] {
                let factor = HouseholderDecomposition::factor(&a, orientation, pivoting).unwrap();
                if pivoting == HouseholderPivoting::None {
                    assert!(
                        factor
                            .permutation()
                            .iter()
                            .enumerate()
                            .all(|(i, &p)| i == p)
                    );
                }
                let rebuilt = factor.reconstruct().unwrap();
                for (&actual, &expected) in rebuilt.as_slice().iter().zip(a.as_slice()) {
                    assert!(
                        (actual - expected).abs() < 1e-11,
                        "n={n}, {orientation:?}, {pivoting:?}"
                    );
                }
                let q = factor.to_q_matrix().unwrap();
                let gram = q
                    .view()
                    .adjoint()
                    .try_to_matrix()
                    .unwrap()
                    .view()
                    .mul_mat(&q)
                    .unwrap();
                for i in 0..gram.rows() {
                    for j in 0..gram.cols() {
                        let expected = if i == j { S::ONE } else { S::ZERO };
                        assert!((gram.as_slice()[i * gram.cols() + j] - expected).abs() < 1e-11);
                    }
                }
            }
        }
    }
}
#[test]
fn real_panel_boundaries() {
    boundaries::<f64>();
}
#[test]
fn complex_panel_boundaries() {
    boundaries::<Complex64>();
}
#[test]
fn default_and_cancelling_norm_downdates() {
    assert_eq!(HouseholderPivoting::default(), HouseholderPivoting::Column);
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            1., 2., 3., 4., 0., 1e-12, 0., 0., 0., 0., 1e-11, 0., 0., 0., 0., 1e-10,
        ],
    )
    .unwrap();
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    assert_eq!(factor.permutation()[0], 3);
    let reconstructed = factor.reconstruct().unwrap();
    for (&actual, &expected) in reconstructed.as_slice().iter().zip(a.as_slice()) {
        assert!((actual - expected).abs() < 1e-13);
    }
}

#[test]
fn scalar_solve_keeps_subnormal_quotients() {
    use scioxide_linalg::{RankPolicy, Vector};
    let tiny = f64::from_bits(1);
    let a = Matrix::from_vec(1, 1, vec![3.0 * tiny]).unwrap();
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let solution = factor
        .into_minimum_norm(
            &((scioxide_linalg::DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()))
            .with_residual(scioxide_linalg::Estimation::Estimate),
        )
        .unwrap()
        .solve(&Vector::from_vec(vec![tiny]))
        .unwrap();
    assert!((solution.solution().unwrap().as_slice()[0] - 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn borrowed_diagonal_matches_owned_triangular_factors() {
    assert!(
        HouseholderDecomposition::<f64>::empty()
            .triangular_diagonal()
            .is_empty()
    );
    for (rows, cols) in [(0, 3), (3, 0), (1, 1), (3, 5), (5, 3)] {
        let a = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| Complex64::new((i + 1) as f64, (i % 3) as f64 - 1.0))
                .collect(),
        )
        .unwrap();
        for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
            for pivoting in [HouseholderPivoting::None, HouseholderPivoting::Column] {
                let factor = HouseholderDecomposition::factor(&a, orientation, pivoting).unwrap();
                let diagonal = factor.triangular_diagonal();
                let triangular = factor.to_triangular_matrix().unwrap();
                assert_eq!(diagonal.len(), rows.min(cols));
                for (i, value) in diagonal.iter().enumerate() {
                    assert_eq!(value, triangular.view().get(i, i).unwrap());
                }
            }
        }
    }
}

#[test]
fn graded_columns_recompute_norms_after_partial_panel_progress() {
    // The final four columns lose their common leading component after 69
    // ordinary pivots. Their true residual norms differ by orders of magnitude.
    let n = 73;
    let mut a = Matrix::<f64>::zeros(n + 3, n).unwrap();
    for i in 0..n - 4 {
        a.view_mut().set(i, i, (100 - i) as f64).unwrap();
    }
    for j in 0..4 {
        a.view_mut().set(n - 4, n - 4 + j, (j + 1) as f64).unwrap();
    }
    for (j, tiny) in [1e-12, 1e-11, 1e-10].into_iter().enumerate() {
        a.view_mut().set(n - 3 + j, n - 3 + j, tiny).unwrap();
    }
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let expected: Vec<_> = (0..n - 4).chain((n - 4..n).rev()).collect();
    assert_eq!(factor.permutation(), expected);
    let rebuilt = factor.reconstruct().unwrap();
    for (&actual, &expected) in rebuilt.as_slice().iter().zip(a.as_slice()) {
        assert!((actual - expected).abs() < 1e-13);
    }
    // Compare every chosen pivot against explicitly recomputed norms from R.
    // Orthogonality makes these the norms of the corresponding trailing
    // columns at that step, independent of the implementation's downdates.
    let r = factor.to_triangular_matrix().unwrap();
    for k in 0..n {
        let pivot = r.view().get(k, k).unwrap().abs();
        for j in k + 1..n {
            let norm = (k..n).fold(0.0_f64, |norm, i| norm.hypot(r.view().get(i, j).unwrap()));
            assert!(norm <= pivot * (1.0 + 128.0 * f64::EPSILON));
        }
    }
}

#[test]
fn multipanel_rank_deficient_qr_preserves_factors_and_rejects_full_rank_solve() {
    use scioxide_linalg::{DirectSolveTermination, LinalgWorkspace, RankPolicy, Vector};
    let (m, n) = (83, 73);
    let mut a = Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|index| {
                let (i, j) = (index / n, index % n);
                if j == 0 {
                    0.0
                } else {
                    ((i * 17 + j * 31) % 97) as f64 / 97.0 + if i == j { 4.0 } else { 0.0 }
                }
            })
            .collect(),
    )
    .unwrap();
    for i in 0..m {
        let value = 2.0 * a.view().get(i, 1).unwrap();
        a.view_mut().set(i, n - 1, value).unwrap();
    }
    let factor = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let rank = factor
        .analyze_rank(RankPolicy::DimensionScaledEpsilon)
        .unwrap();
    assert_eq!(rank.rank(), n - 2);
    let rebuilt = factor.reconstruct().unwrap();
    for (&actual, &expected) in rebuilt.as_slice().iter().zip(a.as_slice()) {
        assert!((actual - expected).abs() < 1e-11);
    }
    let mut solution = Vector::from_vec(vec![17.0; n]);
    let report = factor
        .solve_into(
            &Vector::from_vec(vec![1.0; m]),
            &(scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()),
            solution.view_mut(),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), DirectSolveTermination::RankDeficient);
    assert!(solution.as_slice().iter().all(|&x| x == 17.0));
}

fn scaled_panels<S: NumericScalar>() {
    let real = |value| S::Real::try_from_f64_lossy(value).unwrap();
    let tolerance = real(128.0) * S::Real::EPSILON;
    for n in [33, 65] {
        for scale in [
            S::Real::MIN_POSITIVE / real(8.0),
            S::Real::ONE,
            S::Real::MAX.scalbn(-12),
        ] {
            // Subnormal updates round to an absolute grid, independently of
            // the relative Standard error. Account for one quantum per row.
            let reconstruction_tolerance = tolerance
                + real((n + 3) as f64) * (S::Real::MIN_POSITIVE * S::Real::EPSILON) / scale;
            for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
                let (rows, cols) = if orientation == HouseholderOrientation::Qr {
                    (n + 3, n)
                } else {
                    (n, n + 3)
                };
                let a = Matrix::from_vec(
                    rows,
                    cols,
                    (0..rows * cols)
                        .map(|index| {
                            let (i, j) = (index / cols, index % cols);
                            S::from_components(
                                real(
                                    ((i * 17 + j * 31) % 23) as f64 / 32.0
                                        + if i == j { 4.0 } else { 0.0 },
                                ) * scale,
                                if S::IS_REAL {
                                    S::Real::ZERO
                                } else {
                                    real(((i * 7 + j * 11) % 13) as f64 / 32.0) * scale
                                },
                            )
                        })
                        .collect(),
                )
                .unwrap();
                for pivoting in [HouseholderPivoting::None, HouseholderPivoting::Column] {
                    let factor =
                        HouseholderDecomposition::factor(&a, orientation, pivoting).unwrap();
                    let rebuilt = factor.reconstruct().unwrap();
                    for (&actual, &expected) in rebuilt.as_slice().iter().zip(a.as_slice()) {
                        // Divide components separately to avoid squaring the extreme scale.
                        assert!(
                            (actual.real_component() / scale - expected.real_component() / scale)
                                .abs()
                                < reconstruction_tolerance,
                            "{} n={n} {orientation:?} {pivoting:?} scale={scale:?} actual={actual:?} expected={expected:?} tolerance={tolerance:?}",
                            core::any::type_name::<S>(),
                        );
                        assert!(
                            (actual.imaginary_component() / scale
                                - expected.imaginary_component() / scale)
                                .abs()
                                < reconstruction_tolerance
                        );
                    }
                    let q = factor.to_q_matrix().unwrap();
                    let gram = q.view().adjoint().mul_mat(&q).unwrap();
                    for i in 0..gram.rows() {
                        for j in 0..gram.cols() {
                            let expected = if i == j { S::ONE } else { S::ZERO };
                            assert!((gram.view().get(i, j).unwrap() - expected).abs() < tolerance);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn scaled_real_and_complex_panels_at_both_precisions() {
    scaled_panels::<f32>();
    scaled_panels::<f64>();
    scaled_panels::<Complex32>();
    scaled_panels::<Complex64>();
}
