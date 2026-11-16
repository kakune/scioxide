//! Hessenberg reduction and shifted Hessenberg solves.

use crate::{Matrix, MatrixView};
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};

/// Reduces a square matrix to upper Hessenberg form `A = Q H Qᴴ`.
///
/// Returns `(H, Q)`, where `H` is upper Hessenberg (zero below the first
/// subdiagonal) and `Q` is unitary (orthogonal for real scalars). The
/// reduction applies Householder reflections in O(n³) operations. A `0 × 0`
/// matrix is valid.
///
/// # Errors
/// - `DimensionMismatch` if `a` is not square.
/// - `NonFinite` if `a` contains NaN or an infinity.
/// - `ExecutionStopped` if `cx` is cancelled or its deadline passes.
/// - `AllocationFailure` if `H`, `Q` or the scratch storage cannot be allocated.
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §7.4.3.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::{Matrix, backend::hessenberg};
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let (h, q) = hessenberg::reduce(a.view(), ExecutionContext::sequential_native())?;
/// assert_eq!(h.rows(), 2);
/// assert_eq!(q.cols(), 2);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn reduce<T: NumericScalar>(
    a: MatrixView<'_, T>,
    cx: ExecutionContext<'_>,
) -> Result<(Matrix<T>, Matrix<T>), SciOxideError> {
    crate::eigen::hessenberg::prepare(a, cx)
}

/// Borrowed upper Hessenberg matrix, checked once for repeated shifted solves.
///
/// If the first subdiagonal is also zero, the matrix is upper triangular and
/// [`solve_shifted`] uses back substitution only.
#[derive(Clone, Copy, Debug)]
pub struct HessenbergView<'a, T: NumericScalar> {
    matrix: MatrixView<'a, T>,
    triangular: bool,
}

impl<'a, T: NumericScalar> HessenbergView<'a, T> {
    /// Checks that `matrix` is square, finite and upper Hessenberg.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if `matrix` contains NaN or an infinity.
    /// - `InvalidInput` if an entry below the first subdiagonal is nonzero.
    pub fn new(matrix: MatrixView<'a, T>) -> Result<Self, SciOxideError> {
        let n = matrix.rows();
        if matrix.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                [n, n],
                [n, matrix.cols()],
            ));
        }
        let _ = matrix.try_finite()?;
        let mut triangular = true;
        for row in 0..n {
            for col in 0..row {
                if matrix.get_element(row, col) != T::ZERO {
                    if row > col + 1 {
                        return Err(SciOxideError::invalid_input(
                            scioxide_core::argument!("h"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "matrix must be upper Hessenberg",
                        ));
                    }
                    triangular = false;
                }
            }
        }
        Ok(Self { matrix, triangular })
    }
}

/// Solves `(σ I − H) X = B` in place, where `σ` is `shift`.
///
/// `rhs` holds `B` as a row-major `n × columns` array and is overwritten
/// with `X`. `factor` is scratch of at least `n²` elements; it is overwritten
/// on every call. `H` itself is not modified.
///
/// The solve copies `σ I − H` into `factor` and runs Gaussian elimination.
/// Because `H` has one subdiagonal, each step only compares rows `k` and
/// `k + 1` and swaps them if the subdiagonal entry is larger. Magnitudes are
/// compared as `max(|Re z|, |Im z|)`. The cost is O(n² + n² m) operations for
/// `m = columns` right-hand sides. Triangular `H` skips the elimination.
///
/// # Errors
/// - `SizeOverflow` if `n²` or `n · columns` overflows `usize`.
/// - `DimensionMismatch` if `factor` has fewer than `n²` elements or `rhs`
///   does not have exactly `n · columns` elements.
/// - `NonFinite` if `shift` or an entry of `rhs` is NaN or infinite.
/// - `NumericalInstability` if an entry of `σ I − H` or an intermediate value
///   is not finite.
/// - `Singular` if a pivot is exactly zero.
///
/// ```
/// use scioxide_linalg::{Matrix, backend::hessenberg};
/// let h = Matrix::from_vec(2, 2, vec![-1.0_f64, 0.0, 1.0, -2.0])?;
/// let mut rhs = [1.0, 1.0];
/// let h = hessenberg::HessenbergView::new(h.view())?;
/// hessenberg::solve_shifted(h, 0.0, 1, &mut rhs, &mut [0.0; 4])?;
/// assert_eq!(rhs, [1.0, 1.0]);
/// Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn solve_shifted<T: NumericScalar>(
    h: HessenbergView<'_, T>,
    shift: T,
    columns: usize,
    rhs: &mut [T],
    factor: &mut [T],
) -> Result<(), SciOxideError> {
    let n = h.matrix.rows();
    let square = scioxide_core::backend::checked_product(
        &[n, n],
        scioxide_core::quantity!("hessenberg_factor"),
    )?;
    let rhs_len = scioxide_core::backend::checked_product(
        &[n, columns],
        scioxide_core::quantity!("hessenberg_rhs"),
    )?;
    if factor.len() < square {
        return Err(SciOxideError::dimension_mismatch([square], [factor.len()]));
    }
    if rhs.len() != rhs_len {
        return Err(SciOxideError::dimension_mismatch([rhs_len], [rhs.len()]));
    }
    let factor = &mut factor[..square];
    crate::checked::validate_input(shift, "shift")?;
    if rhs.iter().any(|v| !v.has_finite_components()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "Hessenberg right-hand side must be finite",
        ));
    }
    if h.triangular {
        for row in (0..n).rev() {
            let diagonal =
                crate::checked::finite_candidate(shift - h.matrix.get_element(row, row))?;
            if diagonal == T::ZERO {
                return Err(SciOxideError::singular(None));
            }
            for col in 0..columns {
                let value = (row + 1..n).fold(rhs[row * columns + col], |sum, k| {
                    sum + h.matrix.get_element(row, k) * rhs[k * columns + col]
                });
                rhs[row * columns + col] = crate::checked::finite_candidate(value / diagonal)?;
            }
        }
        return Ok(());
    }
    for row in 0..n {
        for col in 0..n {
            let value = h.matrix.get_element(row, col);
            factor[row * n + col] =
                crate::checked::finite_candidate(if row == col { shift - value } else { -value })?;
        }
    }
    for k in 0..n.saturating_sub(1) {
        if factor[(k + 1) * n + k] == T::ZERO {
            continue;
        }
        // Component maximum is a range-safe pivot norm; it bounds modulus
        // within sqrt(2) for complex scalars without squaring tiny/large values.
        if magnitude(factor[(k + 1) * n + k]) > magnitude(factor[k * n + k]) {
            for col in k..n {
                factor.swap(k * n + col, (k + 1) * n + col);
            }
            for col in 0..columns {
                rhs.swap(k * columns + col, (k + 1) * columns + col);
            }
        }
        let diagonal = factor[k * n + k];
        if diagonal == T::ZERO {
            return Err(SciOxideError::singular(None));
        }
        let multiplier = crate::checked::finite_candidate(factor[(k + 1) * n + k] / diagonal)?;
        factor[(k + 1) * n + k] = T::ZERO;
        for col in k + 1..n {
            factor[(k + 1) * n + col] = crate::checked::finite_candidate(
                factor[(k + 1) * n + col] - multiplier * factor[k * n + col],
            )?;
        }
        for col in 0..columns {
            rhs[(k + 1) * columns + col] = crate::checked::finite_candidate(
                rhs[(k + 1) * columns + col] - multiplier * rhs[k * columns + col],
            )?;
        }
    }
    for row in (0..n).rev() {
        let diagonal = factor[row * n + row];
        if diagonal == T::ZERO {
            return Err(SciOxideError::singular(None));
        }
        for col in 0..columns {
            let value = (row + 1..n).fold(rhs[row * columns + col], |sum, k| {
                sum - factor[row * n + k] * rhs[k * columns + col]
            });
            rhs[row * columns + col] = crate::checked::finite_candidate(value / diagonal)?;
        }
    }
    Ok(())
}

fn magnitude<T: NumericScalar>(value: T) -> T::Real {
    value
        .real_component()
        .abs()
        .max(value.imaginary_component().abs())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use scioxide_core::Complex;

    #[test]
    fn reduction_preserves_similarity_at_both_kernel_sizes_and_extreme_scales() {
        for (n, scale) in [(0, 1.), (5, 1e-200), (5, 1e200), (129, 1.)] {
            let input = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| (((i * 73 + 19) % 97) as f64 / 97. - 0.5) * scale)
                    .collect(),
            )
            .unwrap();
            let (h, q) = reduce(input.view(), ExecutionContext::sequential_native()).unwrap();
            for row in 0..n {
                for col in 0..n {
                    if row > col + 1 {
                        assert_eq!(h.as_slice()[row * n + col], 0.);
                    }
                    let actual = (0..n)
                        .map(|k| {
                            q.as_slice()[row * n + k]
                                * (0..n)
                                    .map(|j| {
                                        h.as_slice()[k * n + j] / scale * q.as_slice()[col * n + j]
                                    })
                                    .sum::<f64>()
                        })
                        .sum::<f64>();
                    assert!((actual - input.as_slice()[row * n + col] / scale).abs() < 2e-12);
                    let inner = (0..n)
                        .map(|k| q.as_slice()[k * n + row] * q.as_slice()[k * n + col])
                        .sum::<f64>();
                    assert!((inner - if row == col { 1. } else { 0. }).abs() < 2e-12);
                }
            }
        }
    }

    #[test]
    fn adjacent_pivots_solve_multiple_complex_rhs_across_scales() {
        for scale in [1., 1e-200, 1e200] {
            let h = Matrix::from_vec(
                3,
                3,
                [0., 2., 3., 1., 4., 5., 0., 6., 7.]
                    .into_iter()
                    .map(|v| Complex::new(v * scale, v * scale * 0.1))
                    .collect(),
            )
            .unwrap();
            let expected = [
                Complex::new(1., 0.5),
                Complex::new(2., -0.5),
                Complex::new(-1., 0.2),
                Complex::new(0.5, 1.),
                Complex::new(2., -1.),
                Complex::new(0., 1.),
            ];
            let mut rhs = vec![Complex::new(0., 0.); 6];
            for row in 0..3 {
                for col in 0..2 {
                    rhs[row * 2 + col] = -(0..3).fold(Complex::new(0., 0.), |sum, k| {
                        sum + h.as_slice()[row * 3 + k] * expected[k * 2 + col]
                    });
                }
            }
            solve_shifted(
                HessenbergView::new(h.view()).unwrap(),
                Complex::new(0., 0.),
                2,
                &mut rhs,
                &mut [Complex::new(0., 0.); 9],
            )
            .unwrap();
            for (actual, expected) in rhs.iter().zip(expected) {
                assert!((*actual - expected).norm() < 5e-13);
            }
        }
    }

    #[test]
    fn complex_reduction_preserves_unitary_similarity() {
        let n = 4;
        let input = Matrix::from_vec(
            n,
            n,
            (0..n * n)
                .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
                .collect(),
        )
        .unwrap();
        let (h, q) = reduce(input.view(), ExecutionContext::sequential_native()).unwrap();
        for row in 0..n {
            for col in 0..n {
                let actual = (0..n).fold(Complex::new(0., 0.), |sum, k| {
                    sum + q.as_slice()[row * n + k]
                        * (0..n).fold(Complex::new(0., 0.), |sum, j| {
                            sum + h.as_slice()[k * n + j] * q.as_slice()[col * n + j].conj()
                        })
                });
                assert!((actual - input.as_slice()[row * n + col]).norm() < 2e-13);
            }
        }
    }

    #[test]
    fn triangular_shifts_match_known_complex_solutions() {
        let h = Matrix::from_vec(
            2,
            2,
            vec![
                Complex::new(-1., 0.2),
                Complex::new(3., -0.5),
                Complex::new(0., 0.),
                Complex::new(-2., -0.1),
            ],
        )
        .unwrap();
        let expected = [Complex::new(1., 0.5), Complex::new(-1., 0.2)];
        let shift = Complex::new(0., 0.7);
        let mut rhs = [
            shift * expected[0] - h.as_slice()[0] * expected[0] - h.as_slice()[1] * expected[1],
            (shift - h.as_slice()[3]) * expected[1],
        ];
        solve_shifted(
            HessenbergView::new(h.view()).unwrap(),
            shift,
            1,
            &mut rhs,
            &mut [Complex::new(0., 0.); 4],
        )
        .unwrap();
        for (actual, expected) in rhs.into_iter().zip(expected) {
            assert!((actual - expected).norm() < 1e-14);
        }
    }

    #[test]
    fn malformed_and_singular_problems_fail_explicitly() {
        let zero = Matrix::<f64>::zeros(2, 2).unwrap();
        assert!(
            solve_shifted(
                HessenbergView::new(zero.view()).unwrap(),
                0.,
                1,
                &mut [1.; 2],
                &mut [0.; 4]
            )
            .is_err()
        );
        assert!(
            solve_shifted(
                HessenbergView::new(zero.view()).unwrap(),
                1.,
                1,
                &mut [1.; 2],
                &mut [0.; 3]
            )
            .is_err()
        );
        for shift in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                solve_shifted(
                    HessenbergView::new(zero.view()).unwrap(),
                    shift,
                    1,
                    &mut [1.; 2],
                    &mut [0.; 4]
                )
                .unwrap_err()
                .kind(),
                scioxide_core::SciOxideErrorKind::NonFinite
            );
        }
        let non_hessenberg = Matrix::from_vec(3, 3, vec![1.; 9]).unwrap();
        assert!(HessenbergView::new(non_hessenberg.view()).is_err());
        let nonsquare = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(reduce(nonsquare.view(), ExecutionContext::sequential_native()).is_err());
    }
}
