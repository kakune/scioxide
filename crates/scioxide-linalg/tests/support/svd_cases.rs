//! Structural SVD regressions shared by its direct consumers.

#![allow(clippy::unwrap_used)]

use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar};
use scioxide_linalg::Matrix;

pub fn zero_diagonals<S: NumericScalar>(mut check: impl FnMut(&str, Matrix<S>)) {
    let scalar = |x| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
    for n in [2, 3, 33, 40, 65] {
        for (name, zero) in [
            ("leading", 0),
            ("middle", n / 2),
            ("split", n / 2),
            ("leaf", n / 2 + 1),
        ] {
            if name == "leaf" && n <= 32 {
                continue;
            }
            for (rows, cols) in [(n, n), (n + 1, n), (n, n + 1)] {
                let a = Matrix::from_vec(
                    rows,
                    cols,
                    (0..rows * cols)
                        .map(|k| {
                            let (i, j) = (k / cols, k % cols);
                            let value = if i == j && i < n && i != zero {
                                scalar(1.0 + (i % 3) as f64 / 8.0)
                            } else if i + 1 == j && j < n && !(name == "split" && j == zero) {
                                scalar(0.125)
                            } else {
                                S::ZERO
                            };
                            let phase = if !S::IS_REAL && j % 2 == 1 {
                                S::from_components(S::Real::ZERO, S::Real::ONE)
                            } else {
                                S::ONE
                            };
                            value * phase
                        })
                        .collect(),
                )
                .unwrap();
                check(name, a.clone());
                check(name, a.view().adjoint().try_to_matrix().unwrap());
            }
        }
    }
}
