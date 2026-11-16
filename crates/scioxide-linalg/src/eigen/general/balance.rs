//! Power-of-two balancing and isolation permutations for general eigenproblems.

use core::ops::Range;

use crate::GeneralEigenBalancing;
use crate::kernel::standard::Blue;
use crate::rotation::scale_real;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

/// Component range of the off-diagonal entries a scaling step changes.
/// Component minima protect even a tiny imaginary part of a predominantly
/// real entry from disappearing.
struct Extent<R> {
    minimum: R,
    maximum: R,
}

impl<R: RealScalar> Extent<R> {
    fn empty() -> Self {
        Self {
            minimum: R::MAX,
            maximum: R::ZERO,
        }
    }

    fn include<W: NumericScalar<Real = R>>(mut self, value: W) -> Self {
        for component in [value.real_component(), value.imaginary_component()] {
            let magnitude = component.abs();
            if magnitude != R::ZERO {
                self.minimum = self.minimum.min(magnitude);
                self.maximum = self.maximum.max(magnitude);
            }
        }
        self
    }

    fn is_empty(&self) -> bool {
        self.maximum == R::ZERO
    }

    fn scale(&mut self, factor: R) {
        self.minimum *= factor;
        self.maximum *= factor;
    }
}

/// Balance a row-major square matrix in place (Parlett and Reinsch, with
/// 2-norms that include the diagonal), recording the power-of-two scales and
/// the permutation.
///
/// # Errors
/// Returns `NumericalInstability(Overflow)` if a row or column norm is not
/// representable.
pub(super) fn balance<W: NumericScalar>(
    matrix: &mut [W],
    order: usize,
    scales: &mut [W::Real],
    permutation: &mut [usize],
    balancing: GeneralEigenBalancing,
) -> Result<(), SciOxideError> {
    scales.fill(W::Real::ONE);
    permutation
        .iter_mut()
        .enumerate()
        .for_each(|(index, value)| *value = index);
    let window = if matches!(
        balancing,
        GeneralEigenBalancing::Permute | GeneralEigenBalancing::PermuteAndScale
    ) {
        isolate_eigenvalues(matrix, order, permutation)
    } else {
        0..order
    };
    if matches!(
        balancing,
        GeneralEigenBalancing::None | GeneralEigenBalancing::Permute
    ) {
        return Ok(());
    }

    // Power-of-two balancing (James, Langou and Lowery, 2014): each
    // index of the window left by isolation gets a power-of-two factor
    // (applied exactly, so balancing itself introduces no rounding) that
    // brings the Euclidean norms of its window row and column within a factor
    // of two, and is kept only if it cuts their sum by at least 5%. The norms
    // include the diagonal, so diagonally dominant indices stay unscaled:
    // off-diagonal balancing would replace a well-conditioned eigenproblem by
    // a graded similarity that amplifies the QR backward error (Watkins,
    // 2006). The trial norms scale linearly with the factor.
    // The pass cap guarantees termination even if the iteration would keep
    // cycling.
    let radix = W::Real::TWO;
    // xGEBAL's precision margin keeps factors, their reciprocals and norms
    // away from overflow. The component minimum additionally prevents loss
    // of small couplings, which guarding just the largest entry cannot do.
    let safe_minimum = W::Real::MIN_POSITIVE / W::Real::EPSILON;
    let safe_maximum = W::Real::ONE / safe_minimum;
    let reduction = W::Real::try_from_f64_lossy(0.95).unwrap_or(W::Real::ONE);
    let empty_norm = Blue::new()?;
    for _ in 0..order.saturating_mul(8).max(1) {
        let mut changed = false;
        for index in window.clone() {
            // Scaling changes the whole off-diagonal row and column; entries
            // outside the window couple only to isolated eigenvalues.
            let (mut row, mut column) = (Extent::empty(), Extent::empty());
            let (mut row_norm, mut column_norm) = (empty_norm, empty_norm);
            for other in 0..order {
                let r = matrix[index * order + other];
                let c = matrix[other * order + index];
                if other != index {
                    row = row.include(r);
                    column = column.include(c);
                }
                if window.contains(&other) {
                    row_norm = row_norm.add(r);
                    column_norm = column_norm.add(c);
                }
            }
            if row.is_empty() || column.is_empty() {
                continue;
            }
            let mut row_norm = row_norm.finish()?;
            let mut column_norm = column_norm.finish()?;
            let original_sum = row_norm + column_norm;
            let mut factor = W::Real::ONE;
            let mut scale = scales[index];
            while column_norm < row_norm / radix
                && factor <= safe_maximum / radix
                && scale <= safe_maximum / radix
                && column_norm <= safe_maximum / radix
                && column.maximum <= safe_maximum / radix
                && row_norm >= safe_minimum * radix
                && row.minimum >= W::Real::MIN_POSITIVE * radix
            {
                row_norm /= radix;
                row.scale(W::Real::ONE / radix);
                column_norm *= radix;
                column.scale(radix);
                factor *= radix;
                scale *= radix;
            }
            while row_norm <= column_norm / radix
                && factor >= safe_minimum * radix
                && scale >= safe_minimum * radix
                && row_norm <= safe_maximum / radix
                && row.maximum <= safe_maximum / radix
                && column_norm >= safe_minimum * radix
                && column.minimum >= W::Real::MIN_POSITIVE * radix
            {
                row_norm *= radix;
                row.scale(radix);
                column_norm /= radix;
                column.scale(W::Real::ONE / radix);
                factor /= radix;
                scale /= radix;
            }
            if row_norm + column_norm < reduction * original_sum {
                changed = true;
                scales[index] = scale;
                let inverse = W::Real::ONE / factor;
                for other in (0..order).filter(|&other| other != index) {
                    let row_position = index * order + other;
                    let column_position = other * order + index;
                    matrix[row_position] = scale_real(matrix[row_position], inverse);
                    matrix[column_position] = scale_real(matrix[column_position], factor);
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

/// Permutation phase of balancing; returns the active window `low..high`.
fn isolate_eigenvalues<W: NumericScalar>(
    matrix: &mut [W],
    order: usize,
    permutation: &mut [usize],
) -> Range<usize> {
    // A row whose off-diagonal entries in the active window are all zero
    // exposes its diagonal as an eigenvalue and is moved to the bottom; a
    // column with that property moves to the top. The active window
    // `low..high` shrinks until no such row or column remains, so the
    // eigenvalue iteration never has to rediscover these values.
    let mut low = 0_usize;
    let mut high = order;
    loop {
        let isolated_row = (low..high).find(|&row| {
            (low..high).all(|column| row == column || matrix[row * order + column] == W::ZERO)
        });
        if let Some(row) = isolated_row {
            high -= 1;
            swap_similarity(matrix, order, row, high);
            permutation.swap(row, high);
            if low == high {
                break;
            }
            continue;
        }
        let isolated_column = (low..high).find(|&column| {
            (low..high).all(|row| row == column || matrix[row * order + column] == W::ZERO)
        });
        if let Some(column) = isolated_column {
            swap_similarity(matrix, order, column, low);
            permutation.swap(column, low);
            low += 1;
            if low == high {
                break;
            }
            continue;
        }
        break;
    }
    low..high
}

fn swap_similarity<W: NumericScalar>(matrix: &mut [W], order: usize, first: usize, second: usize) {
    if first == second {
        return;
    }
    for column in 0..order {
        matrix.swap(first * order + column, second * order + column);
    }
    for row in 0..order {
        matrix.swap(row * order + first, row * order + second);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use scioxide_core::{Complex, checked_product_quotient};

    fn check_similarity<W: NumericScalar>(input: &[W], n: usize) {
        for strategy in [
            GeneralEigenBalancing::None,
            GeneralEigenBalancing::Permute,
            GeneralEigenBalancing::Scale,
            GeneralEigenBalancing::PermuteAndScale,
        ] {
            let mut matrix = input.to_vec();
            let mut scales = vec![W::Real::ZERO; n];
            let mut permutation = vec![0; n];
            balance(&mut matrix, n, &mut scales, &mut permutation, strategy).unwrap();
            assert!(scales.iter().all(|&s| s.is_finite()
                && s >= W::Real::MIN_POSITIVE / W::Real::EPSILON
                && s <= W::Real::EPSILON / W::Real::MIN_POSITIVE));
            for (index, value) in matrix.iter().enumerate() {
                let (i, j) = (index / n, index % n);
                let original = input[permutation[i] * n + permutation[j]];
                for (actual, source) in [
                    (value.real_component(), original.real_component()),
                    (value.imaginary_component(), original.imaginary_component()),
                ] {
                    let expected = checked_product_quotient(source, scales[j], scales[i]).unwrap();
                    assert_eq!(actual, expected, "{strategy:?}: entry ({i}, {j})");
                    if source != W::Real::ZERO {
                        assert!(actual != W::Real::ZERO && actual.is_finite());
                        if source.abs() >= W::Real::MIN_POSITIVE {
                            assert!(actual.abs() >= W::Real::MIN_POSITIVE);
                        } else {
                            assert!(actual.abs() >= source.abs());
                        }
                    }
                }
            }
        }
    }

    fn exercise_ranges<R: RealScalar>(n: usize, small: R)
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        // More grading than even the full safe scale interval can accommodate.
        let chain: Vec<R> = (0..n * n)
            .map(|i| {
                let (row, column) = (i / n, i % n);
                if column == row + 1 {
                    R::ONE
                } else if row == column + 1 {
                    small
                } else {
                    R::ZERO
                }
            })
            .collect();
        check_similarity(&chain, n);
        for tiny in [R::MIN_POSITIVE, R::MIN_SUBNORMAL] {
            // The tiny coupling must constrain row scaling even though its
            // contribution to the norm is swallowed by the unit entry. A zero
            // diagonal keeps index 0 eligible for scaling.
            let mixed = [
                R::ZERO,
                R::ONE,
                tiny,
                small,
                R::ONE,
                small,
                small,
                R::ONE,
                R::ONE,
            ];
            check_similarity(&mixed, 3);
            let complex: Vec<_> = mixed
                .iter()
                .enumerate()
                .map(|(i, &re)| Complex::new(re, if i == 1 { -tiny } else { R::ZERO }))
                .collect();
            check_similarity(&complex, 3);
        }
        check_similarity::<R>(&[], 0);
        check_similarity(&[R::MIN_SUBNORMAL], 1);
        check_similarity(&[R::ONE, R::ONE, R::ZERO, R::TWO], 2);
    }

    #[test]
    fn balancing_preserves_f64_ranges_and_similarity() {
        exercise_ranges::<f64>(12, 1e-150);
    }

    #[test]
    fn balancing_preserves_f32_ranges_and_similarity() {
        exercise_ranges::<f32>(8, 1e-25);
    }

    #[test]
    fn scaling_reduces_norm_in_either_direction_without_changing_diagonals() {
        for input in [
            [f64::MAX, 1.0, 1e-150, f64::MIN_SUBNORMAL],
            [f64::MAX, 1e-150, 1.0, f64::MIN_SUBNORMAL],
        ] {
            let mut matrix = input;
            let mut scales = [0.0; 2];
            balance(
                &mut matrix,
                2,
                &mut scales,
                &mut [0; 2],
                GeneralEigenBalancing::Scale,
            )
            .unwrap();
            assert_eq!(matrix[0], input[0]);
            assert_eq!(matrix[3], input[3]);
            assert!(matrix[1] <= 2.0 * matrix[2] && matrix[2] <= 2.0 * matrix[1]);
            assert!(matrix[1] + matrix[2] < input[1] + input[2]);
        }
    }

    #[test]
    fn diagonal_dominance_suppresses_scaling() {
        // Off-diagonal norms alone would scale both cases by large factors.
        for (input, n) in [
            (vec![1.0, 1.0, 2.0_f64.powi(-60), 1.0], 2),
            (
                vec![f64::MAX, 1.0, 1e-300, 1e-150, 1.0, 1e-150, 1e-150, 1.0, 1.0],
                3,
            ),
        ] {
            let mut matrix = input.clone();
            let mut scales = vec![0.0; n];
            let mut permutation = vec![0; n];
            balance(
                &mut matrix,
                n,
                &mut scales,
                &mut permutation,
                GeneralEigenBalancing::PermuteAndScale,
            )
            .unwrap();
            assert!(scales.iter().all(|&s| s == 1.0), "{scales:?}");
            assert_eq!(matrix, input);
        }
    }

    #[test]
    fn window_ignores_isolated_couplings() {
        // Row 2 isolates `5`; its large column couples only to that eigenvalue.
        let big = 2.0_f64.powi(40);
        let input = [1.0, 1.0, big, 1.0, 1.0, big, 0.0, 0.0, 5.0];
        let run = |strategy| {
            let mut matrix = input;
            let mut scales = [0.0; 3];
            let mut permutation = [0; 3];
            balance(&mut matrix, 3, &mut scales, &mut permutation, strategy).unwrap();
            (matrix, scales)
        };
        let (matrix, scales) = run(GeneralEigenBalancing::PermuteAndScale);
        assert_eq!(scales, [1.0; 3]);
        assert_eq!(matrix, input);
        // Without isolation the coupling is part of the window and is balanced.
        let (matrix, scales) = run(GeneralEigenBalancing::Scale);
        assert!(scales.iter().any(|&s| s != 1.0));
        assert!(matrix[2] < big);
    }
}
