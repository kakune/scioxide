//! Scaling, storage and copying helpers for the dense factorizations.

use alloc::vec::Vec;

use core::ops::Range;

use scioxide_core::backend::try_reserve;
use scioxide_core::{AlgebraicScalar, NumericScalar, Quantity, RealScalar, SciOxideError};

use crate::dense::{MatrixView, MatrixViewMut};
use crate::workspace::storage::MatrixStorage;

/// Apply `2^exponent` using its precomputed scalar multiplier when representable.
/// The multiplier must be the rounded power of two. A finite nonzero power is
/// exact, so component multiplication has the same single rounding as `scalbn`;
/// an overflowed or flushed multiplier uses component exponent scaling instead.
#[inline]
pub(super) fn scale_with_power<S: NumericScalar>(value: S, power: S::Real, exponent: i32) -> S {
    if power.is_finite() && power != S::Real::ZERO {
        scale_components(value, power)
    } else {
        crate::kernel::standard::scale(value, exponent)
    }
}

/// Validate values while computing their largest modulus for equilibration.
pub(super) fn largest_abs<S: NumericScalar>(values: &[S]) -> Result<S::Real, SciOxideError> {
    if S::IS_REAL {
        return crate::kernel::standard::finite_slice_maximum(values).map_err(|_| {
            SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "LU matrix must be finite",
            )
        });
    }
    values.iter().try_fold(S::Real::ZERO, |maximum, &value| {
        if !value.has_finite_components() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "LU matrix must be finite",
            ));
        }
        Ok(modulus_above(value, maximum).map_or(maximum, |modulus| maximum.max(modulus)))
    })
}

/// `abs()` of a finite value unless its square proves `abs() <= bound`.
///
/// Complex moduli are evaluated with a careful hypotenuse, which dominates
/// maximum scans. With every operand normal, the rounded `re^2 + im^2` is within
/// `3*epsilon` of `|value|^2` and the hypotenuse within one ulp, so a square at
/// most `(1 - 16*epsilon) bound^2` proves the computed modulus does not exceed
/// `bound`. Skipping such values leaves a maximum, and the first index of a
/// strictly larger value, exactly unchanged. Real values are always returned.
#[inline]
pub(super) fn modulus_above<S: NumericScalar>(value: S, bound: S::Real) -> Option<S::Real> {
    if !S::IS_REAL {
        let (re, im) = (value.real_component(), value.imaginary_component());
        let limit = bound * bound * (S::Real::ONE - S::Real::EPSILON.scalbn(4));
        if limit.is_finite()
            && limit >= S::Real::MIN_POSITIVE.scalbn(4)
            && re * re + im * im <= limit
        {
            return None;
        }
    }
    Some(value.abs())
}

/// Multiply every value by `2^exponent` exactly as [`scale_with_power`] does.
pub(super) fn scale_slice<S: NumericScalar>(values: &mut [S], exponent: i32) {
    let power = S::Real::ONE.scalbn(exponent);
    if power.is_finite() && power != S::Real::ZERO {
        values
            .iter_mut()
            .for_each(|value| *value = scale_components(*value, power));
    } else {
        values
            .iter_mut()
            .for_each(|value| *value = crate::kernel::standard::scale(*value, exponent));
    }
}

/// Multiply column `j` of a row-major matrix by its power `2^exponent(j)`,
/// stored as the real part of `powers[j]`, exactly as [`scale_with_power`] does.
pub(super) fn scale_columns<S: NumericScalar>(
    matrix: &mut [S],
    powers: &[S],
    exponent: impl Fn(usize) -> i32,
) {
    let cols = powers.len();
    if cols == 0 {
        return;
    }
    let representable = powers.iter().all(|power| {
        let power = power.real_component();
        power.is_finite() && power != S::Real::ZERO
    });
    matrix.chunks_exact_mut(cols).for_each(|row| {
        if representable {
            row.iter_mut().zip(powers).for_each(|(value, power)| {
                *value = scale_components(*value, power.real_component())
            });
        } else {
            row.iter_mut()
                .zip(powers)
                .enumerate()
                .for_each(|(col, (value, power))| {
                    *value = scale_with_power(*value, power.real_component(), exponent(col));
                });
        }
    });
}

#[inline]
fn scale_components<S: NumericScalar>(value: S, power: S::Real) -> S {
    S::from_components(
        value.real_component() * power,
        value.imaginary_component() * power,
    )
}

pub(super) fn storage_bytes<const N: usize>(
    regions: [(usize, usize); N],
) -> Result<usize, SciOxideError> {
    regions.into_iter().try_fold(0usize, |total, (len, width)| {
        len.checked_mul(width)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("factor_storage_bytes"))
            })
    })
}

pub(super) fn matrix_block<S: NumericScalar>(
    matrix: &impl MatrixStorage<S>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Result<MatrixView<'_, S>, SciOxideError> {
    let height = rows
        .end
        .checked_sub(rows.start)
        .ok_or_else(|| SciOxideError::dimension_mismatch([rows.start], [rows.end]))?;
    let width = cols
        .end
        .checked_sub(cols.start)
        .ok_or_else(|| SciOxideError::dimension_mismatch([cols.start], [cols.end]))?;
    matrix
        .try_view()?
        .block(rows.start, cols.start, height, width)
}

pub(super) fn matrix_block_mut<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Result<MatrixViewMut<'_, S>, SciOxideError> {
    let height = rows
        .end
        .checked_sub(rows.start)
        .ok_or_else(|| SciOxideError::dimension_mismatch([rows.start], [rows.end]))?;
    let width = cols
        .end
        .checked_sub(cols.start)
        .ok_or_else(|| SciOxideError::dimension_mismatch([cols.start], [cols.end]))?;
    // Consume the checked view so the block's lifetime belongs to storage.
    matrix
        .try_view_mut()?
        .into_block(rows.start, cols.start, height, width)
}

pub(super) fn copy_block<S: NumericScalar>(
    source: MatrixView<'_, S>,
    destination: &mut impl MatrixStorage<S>,
) -> Result<(), SciOxideError> {
    destination.try_resize_flat(source.rows(), source.cols())?;
    copy_rows(source, destination.as_mut_slice());
    Ok(())
}

/// Copy the logical entries of `source` row-major into `destination`.
pub(crate) fn copy_rows<S: NumericScalar>(source: MatrixView<'_, S>, destination: &mut [S]) {
    if let Some(values) = source.as_contiguous_slice() {
        destination[..values.len()].copy_from_slice(values);
        return;
    }
    let cols = source.cols();
    if cols == 0 {
        return;
    }
    let rows = destination[..source.rows() * cols].chunks_exact_mut(cols);
    if source.col_stride() == 1 && !source.is_conjugated() {
        rows.enumerate().for_each(|(row, destination_row)| {
            let start = row * source.row_stride();
            destination_row.copy_from_slice(&source.data[start..start + cols]);
        });
    } else {
        rows.enumerate().for_each(|(row, destination_row)| {
            destination_row
                .iter_mut()
                .enumerate()
                .for_each(|(col, value)| *value = source.get_element(row, col));
        });
    }
}

#[inline(always)]
pub(super) fn resize_vec<T: Copy>(
    values: &mut Vec<T>,
    len: usize,
    fill: T,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.capacity() < len {
        try_reserve(values, len.saturating_sub(values.len()), quantity)?;
    }
    values.resize(len, fill);
    Ok(())
}

pub(super) fn reserve_vec<T>(
    values: &mut Vec<T>,
    len: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.capacity() < len {
        try_reserve(values, len.saturating_sub(values.len()), quantity)?;
    }
    Ok(())
}

/// Copy rows `columns.start..` of the row-major `matrix` with row length
/// `stride`, restricted to `columns`, into a contiguous row-major panel, or
/// back with `gather == false`.
pub(super) fn transfer_panel<S: NumericScalar>(
    matrix: &mut [S],
    stride: usize,
    columns: Range<usize>,
    panel: &mut [S],
    gather: bool,
) {
    matrix[columns.start * stride..]
        .chunks_exact_mut(stride)
        .zip(panel.chunks_exact_mut(columns.len()))
        .for_each(|(row, panel_row)| {
            if gather {
                panel_row.copy_from_slice(&row[columns.clone()]);
            } else {
                row[columns.clone()].copy_from_slice(panel_row);
            }
        });
}

pub(super) fn copy_panel<'a, S: NumericScalar>(
    source: crate::MatrixView<'_, S>,
    destination: &'a mut [S],
) -> Result<crate::FiniteMatrixView<'a, S>, SciOxideError> {
    let destination = &mut destination[..source.rows() * source.cols()];
    copy_rows(source, destination);
    let copied = crate::MatrixView::from_slice(source.rows(), source.cols(), destination)?;
    let maximum = crate::kernel::standard::finite_component_maximum(copied)?;
    Ok(crate::FiniteMatrixView::from_validated(copied, maximum))
}

/// Transpose a square packed matrix using paired cache-sized tiles.
pub(crate) fn transpose_square_in_place<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>) {
    let order = matrix.rows();
    debug_assert_eq!(order, matrix.cols());
    // Keep both sides of each swap in a pair of cache-sized tiles.
    for first_row in (0..order).step_by(32) {
        for first_column in (first_row..order).step_by(32) {
            for row in first_row..(first_row + 32).min(order) {
                for column in first_column.max(row + 1)..(first_column + 32).min(order) {
                    matrix
                        .as_mut_slice()
                        .swap(row * order + column, column * order + row);
                }
            }
        }
    }
}

/// Frobenius norm of the Gram defect `UᴴU - I` (or `UUᴴ - I` for a wide
/// factor), with one stop check per Gram row.
pub(super) fn orthogonality<S: NumericScalar>(
    u: MatrixView<'_, S>,
    context: crate::KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    let (m, n) = (u.rows(), u.cols());
    let k = m.min(n);
    crate::kernel::standard::try_norm((0..k).map(|i| {
        context.execution().check()?;
        crate::kernel::standard::norm_workspace((0..k).map(|j| {
            let dot = if m >= n {
                (0..m).fold(S::ZERO, |s, l| {
                    s + u.get_element(l, i).conj() * u.get_element(l, j)
                })
            } else {
                (0..n).fold(S::ZERO, |s, l| {
                    s + u.get_element(i, l) * u.get_element(j, l).conj()
                })
            };
            dot - if i == j { S::ONE } else { S::ZERO }
        }))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;

    #[test]
    fn internal_blocks_validate_ranges_and_preserve_storage_on_rejection()
    -> Result<(), SciOxideError> {
        let mut matrix = Matrix::from_vec(2, 3, alloc::vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let block = matrix_block(&matrix, 1..2, 1..3)?;
        assert_eq!(block.get(0, 0)?, 5.0);
        assert_eq!(block.get(0, 1)?, 6.0);
        assert_eq!(block.row_stride(), 3);
        let empty = matrix_block(&matrix, 2..2, 3..3)?;
        assert_eq!((empty.rows(), empty.cols()), (0, 0));
        let reversed = core::ops::Range { start: 2, end: 1 };
        assert!(matrix_block(&matrix, reversed, 0..1).is_err());
        assert!(matrix_block_mut(&mut matrix, 0..3, 0..1).is_err());
        assert_eq!(matrix.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        matrix_block_mut(&mut matrix, 0..2, 1..2)?.fill(9.0);
        assert_eq!(matrix.as_slice(), &[1.0, 9.0, 3.0, 4.0, 9.0, 6.0]);
        Ok(())
    }
    fn screened_moduli<R: RealScalar + NumericScalar<Real = R>>() {
        use scioxide_core::Complex;
        let value = |x: f64| R::try_from_f64_lossy(x).unwrap();
        let (tiny, huge) = (R::MIN_POSITIVE.scalbn(2), R::MAX.scalbn(-2));
        let mut values: Vec<Complex<R>> = (0..4096)
            .map(|i| {
                let scale = value(2.0_f64.powi((i % 41) - 20));
                let (re, im) = (
                    value(((i * 37) % 101) as f64 - 50.0),
                    value(((i * 53) % 89) as f64 - 44.0),
                );
                Complex::new(re * scale, im * scale)
            })
            .collect();
        values.extend([
            Complex::new(R::ZERO, R::ZERO),
            Complex::new(tiny, tiny),
            Complex::new(huge, huge),
            Complex::new(huge, tiny),
            Complex::new(R::ONE, R::EPSILON),
        ]);
        for &z in &values {
            let modulus = z.abs();
            // Never skip a value whose computed modulus exceeds the bound.
            let below = if modulus > R::ZERO {
                modulus.scalbn(-1).max(modulus * (R::ONE - R::EPSILON))
            } else {
                R::ZERO
            };
            for bound in [R::ZERO, below, modulus, modulus * value(1.5), tiny, huge] {
                match modulus_above(z, bound) {
                    None => assert!(modulus <= bound, "{z:?} skipped at {bound:?}"),
                    Some(evaluated) => assert_eq!(evaluated, modulus),
                }
            }
        }
        // Screened scans equal the plain fold exactly.
        let plain = values
            .iter()
            .fold(R::ZERO, |maximum, value| maximum.max(value.abs()));
        assert_eq!(largest_abs(&values).unwrap(), plain);
        assert_eq!(modulus_above(value(-3.0), value(5.0)), Some(value(3.0)));
        // Ordinary smaller moduli are screened out; out-of-range bounds are not.
        assert_eq!(
            modulus_above(Complex::new(R::ONE, R::ONE), value(2.0)),
            None
        );
        assert!(modulus_above(Complex::new(tiny, R::ZERO), tiny.scalbn(1)).is_some());
        assert!(modulus_above(Complex::new(R::ONE, R::ZERO), huge.scalbn(1)).is_some());
    }

    #[test]
    fn screened_moduli_never_change_a_maximum() {
        screened_moduli::<f32>();
        screened_moduli::<f64>();
    }
}
