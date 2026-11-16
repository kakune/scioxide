//! Dimension-scaled backward-error gates for difficult dense fixtures.
#![allow(clippy::unwrap_used)]
use crate::support::decomposition as support;
use crate::support::metrics;
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, HermitianEigenDecomposition, HouseholderDecomposition,
    HouseholderOrientation, LuDecomposition, Matrix,
};

fn fixtures(n: usize) -> Vec<(&'static str, Matrix<f64>)> {
    let hilbert = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| 1.0 / ((i / n) + (i % n) + 1) as f64)
            .collect(),
    )
    .unwrap();
    let graded = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                let diagonal = 10.0_f64.powf(-12.0 * r as f64 / (n - 1) as f64);
                if r == c {
                    diagonal
                } else if r < c {
                    -0.1 * diagonal
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    let kahan = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                let s = 0.25_f64;
                if r > c {
                    0.0
                } else {
                    s.powi(r as i32) * if r == c { 1.0 } else { -(1.0 - s * s).sqrt() }
                }
            })
            .collect(),
    )
    .unwrap();
    let q = HouseholderDecomposition::factor(
        &support::generated_matrix::<f64>(n, n, 0x827),
        HouseholderOrientation::Qr,
        scioxide_linalg::HouseholderPivoting::Column,
    )
    .unwrap()
    .to_q_matrix()
    .unwrap();
    let spectrum = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    10.0_f64.powf(-10.0 * (i / n) as f64 / (n - 1) as f64)
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    let mut prescribed = q
        .view()
        .mul_mat(&spectrum)
        .unwrap()
        .view()
        .mul_mat(&q.view().transpose().try_to_matrix().unwrap())
        .unwrap();
    for row in 0..n {
        for col in 0..row {
            let value = prescribed.as_slice()[row * n + col];
            prescribed.as_mut_slice()[col * n + row] = value;
        }
    }
    vec![
        ("Hilbert", hilbert),
        ("graded", graded),
        ("Kahan", kahan),
        ("prescribed condition", prescribed),
    ]
}

#[test]
fn difficult_dense_factors_obey_backward_error_bounds() {
    let n = 12;
    let bound = metrics::invariant_bound::<f64>(n);
    for (name, base) in fixtures(n) {
        for scale in [1.0e-100, 1.0, 1.0e100] {
            let a = Matrix::from_vec(
                n,
                n,
                base.as_slice().iter().map(|value| value * scale).collect(),
            )
            .unwrap();
            let lu = LuDecomposition::factor(&a).unwrap();
            let lower = lu.to_l_matrix().unwrap();
            let upper = lu.to_u_matrix().unwrap();
            let reconstructed = lower.view().mul_mat(&upper).unwrap();
            let permuted = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| a.as_slice()[lu.permutation()[i / n] * n + i % n])
                    .collect(),
            )
            .unwrap();
            let growth_scale = metrics::matrix_norm(&lower) * metrics::matrix_norm(&upper);
            assert!(
                metrics::matrix_difference_norm(&reconstructed, &permuted) / growth_scale <= bound,
                "{name}: LU"
            );
            let qr = HouseholderDecomposition::factor(
                &a,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
            )
            .unwrap();
            assert!(
                metrics::relative_matrix_difference(&qr.reconstruct().unwrap(), &a) <= bound,
                "{name}: QR"
            );
            assert!(
                metrics::column_orthogonality_residual(&qr.to_q_matrix().unwrap()) <= bound,
                "{name}: Q orthogonality"
            );
            support::assert_thin_svd_invariants(
                &a,
                &scioxide_linalg::ThinSvdDecomposition::compute(&a).unwrap(),
                name,
            );
            if name == "Hilbert" || name == "prescribed condition" {
                let chol = CholeskyDecomposition::factor(
                    &a,
                    scioxide_linalg::HermitianInput::Exact,
                    CholeskyPivoting::None,
                )
                .unwrap();
                let l = chol.l_matrix();
                assert!(
                    metrics::relative_matrix_difference(
                        &l.view()
                            .mul_mat(&l.view().transpose().try_to_matrix().unwrap())
                            .unwrap(),
                        &a
                    ) <= bound,
                    "{name}: Cholesky"
                );
                support::assert_hermitian_invariants(
                    &a,
                    &HermitianEigenDecomposition::decompose(
                        &a,
                        scioxide_linalg::HermitianInput::Exact,
                    )
                    .unwrap(),
                    name,
                );
            }
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn hermitian_vectors_are_bitwise_identical_across_worker_counts() {
    use scioxide_core::{ExecutionContext, StdProcessor, ThreadExecutor};
    use scioxide_linalg::{HermitianEigenOptions, LinalgWorkspace};
    use std::num::NonZeroUsize;
    // Large enough that dense updates cross the auto-parallel threshold.
    let n = 512;
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                if row == col {
                    2.0 + row as f64 / n as f64
                } else {
                    ((row.min(col) * 17 + row.max(col) * 31) % 103) as f64 / 103.0 / n as f64
                }
            })
            .collect(),
    )
    .unwrap();
    let mut expected = None;
    let processor = StdProcessor;
    for workers in [1, 2, 4, 8] {
        let pool = ThreadExecutor::new(NonZeroUsize::new(workers).unwrap());
        let context = ExecutionContext::new(&pool).with_processor(&processor);
        let mut result = HermitianEigenDecomposition::empty();
        let mut workspace = LinalgWorkspace::new();
        HermitianEigenDecomposition::decompose_into(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            &mut result,
            &mut workspace,
            context,
        )
        .unwrap();
        let bits: Vec<_> = result
            .eigenvalues()
            .iter()
            .chain(result.eigenvectors().as_slice())
            .map(|value| value.to_bits())
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&bits, expected, "worker count {workers}");
        } else {
            expected = Some(bits);
        }
    }
}
