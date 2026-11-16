//! Checked dense matrix-vector and transpose-copy dispatch.

use scioxide_core::{Dimensions, ExecutionPolicy, NumericScalar, ProductSum, SciOxideError};

use super::standard;
use crate::dense::{MatrixView, MatrixViewMut, VectorView};
use crate::{Accuracy, FiniteMatrixView, FiniteVectorView, KernelContext};

/// Edge length used by the cache-tiled transpose copy.
///
/// Two f64 tiles occupy 16 KiB, leaving cache capacity for loop state and
/// neighboring data.
pub(crate) const TRANSPOSE_TILE_EDGE: usize = 32;

pub(crate) fn gemv_candidate<'a, S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    input: VectorView<'_, S>,
    scratch: &mut [S],
    policy: ExecutionPolicy,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    let context: crate::KernelContext<'_> = context.into();
    validate_gemv_input_and_scratch(matrix, input, scratch.len())?;
    #[cfg(feature = "blas")]
    let native = context.blas().is_none();
    #[cfg(not(feature = "blas"))]
    let native = true;
    if native && context.accuracy() == Accuracy::Standard {
        let input = input.try_finite_dispatched(context.dispatch())?;
        context.execution().check()?;
        if matrix.cols == 0 {
            scratch[..matrix.rows].fill(S::ZERO);
            return Ok(());
        }
        // Validate and consume four rows while they are still in cache, sharing
        // a range frame. A whole-matrix proof would stream large inputs twice.
        for start in (0..matrix.rows).step_by(4) {
            let end = (start + 4).min(matrix.rows);
            let block = matrix
                .block(start, 0, end - start, matrix.cols)?
                .try_finite_dispatched(context.dispatch())?;
            gemv_bounded(
                block,
                input,
                &mut scratch[start..end],
                ExecutionPolicy::Sequential,
                context,
            )?;
        }
        return Ok(());
    }
    let matrix = matrix.try_finite_dispatched(context.dispatch())?;
    let input = input.try_finite_dispatched(context.dispatch())?;
    gemv_bounded(matrix, input, scratch, policy, context)
}

pub(crate) fn gemv_bounded<S: NumericScalar>(
    matrix: FiniteMatrixView<'_, S>,
    input: FiniteVectorView<'_, S>,
    scratch: &mut [S],
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    #[cfg(not(feature = "blas"))]
    let _ = policy;
    let (a, b) = standard::product_scales(matrix.maximum, input.maximum, matrix.cols())?;
    let matrix = matrix.as_view();
    let input = input.as_view();
    context.execution().check()?;
    if matrix.cols == 0 {
        scratch[..matrix.rows].fill(S::ZERO);
        return Ok(());
    }
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && crate::blas::dispatch::gemv(
            matrix,
            input,
            &mut scratch[..matrix.rows],
            dispatch,
            context.accuracy(),
            crate::kernel::policy_workers(policy, context),
        )? == crate::blas::dispatch::BlasRoute::Provider
    {
        return Ok(());
    }
    if context.accuracy() == Accuracy::Standard {
        // Adjoint applications of row-major real matrices stream contiguous
        // columns of the logical view. Keep a small output tile resident rather
        // than repeatedly traversing the entire matrix with a long stride.
        // The unit-frame range proof covers every product and partial sum;
        // ordinary multiply/add has the documented componentwise Standard bound.
        if S::IS_REAL && matrix.row_stride == 1 && a == 0 && b == 0 {
            for (tile, output) in scratch[..matrix.rows].chunks_mut(64).enumerate() {
                output.fill(S::ZERO);
                for col in 0..matrix.cols {
                    let value = input.get_element(col);
                    let start = col * matrix.col_stride + tile * 64;
                    let end = start + output.len();
                    for (out, &coefficient) in output.iter_mut().zip(&matrix.data[start..end]) {
                        *out = *out + coefficient * value;
                    }
                }
                for value in output {
                    *value = crate::checked::finite_candidate(*value)?;
                }
            }
            return Ok(());
        }
        for (row, candidate) in scratch[..matrix.rows].iter_mut().enumerate() {
            if a == 0
                && b == 0
                && matrix.col_stride == 1
                && !matrix.conjugated
                && let Some(input) = input.as_contiguous_slice()
                && let Some(sum) = super::arch::dot(
                    &matrix.data[row * matrix.row_stride..row * matrix.row_stride + matrix.cols],
                    input,
                    context.dispatch(),
                )?
            {
                *candidate = standard::restore(sum, 0)?;
                continue;
            }
            *candidate = standard::unit_then_normalized(a, b, |a, b| {
                standard::dot((0..matrix.cols).map(|col| {
                    (
                        standard::scale(matrix.get_element(row, col), -a),
                        standard::scale(input.get_element(col), -b),
                    )
                }))
            })?;
        }
        return Ok(());
    }
    // Each row's dot product is accumulated exactly and rounded once, which
    // makes the result independent of summation order and reports overflow
    // only when the true result is unrepresentable.
    (0..matrix.rows).try_for_each(|row| {
        let mut sum = ProductSum::new();
        (0..matrix.cols).try_for_each(|col| {
            sum.add_product(matrix.get_element(row, col), input.get_element(col))
        })?;
        scratch[row] = sum.finish()?;
        Ok::<(), SciOxideError>(())
    })?;
    Ok(())
}

/// Borrowed columns in the two stored Krylov basis layouts.
#[derive(Clone, Copy)]
pub(crate) enum Columns<'a, S: NumericScalar> {
    Packed(&'a [S], usize),
    Vectors(&'a [crate::Vector<S>]),
}
impl<'a, S: NumericScalar> Columns<'a, S> {
    pub(crate) fn maximum(
        self,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<S::Real, SciOxideError> {
        use scioxide_core::{AlgebraicScalar, RealScalar};
        self.iter().try_fold(S::Real::ZERO, |bound, column| {
            Ok(bound.max(super::finite::maximum(column, dispatch)?))
        })
    }

    pub(crate) fn iter(self) -> impl ExactSizeIterator<Item = &'a [S]> + Clone {
        let count = match self {
            Self::Packed(data, rows) => data.len().checked_div(rows).unwrap_or(0),
            Self::Vectors(vectors) => vectors.len(),
        };
        (0..count).map(move |i| match self {
            Self::Packed(data, rows) => &data[i * rows..(i + 1) * rows],
            Self::Vectors(vectors) => vectors[i].as_slice(),
        })
    }
}

/// Block GEMV `coefficients = Q^H image` over validated workspace columns,
/// given component bounds of both operands.
pub(crate) fn project_columns<S: NumericScalar>(
    columns: Columns<'_, S>,
    maximum: S::Real,
    image: &[S],
    image_maximum: S::Real,
    coefficients: &mut [S],
    dispatch: scioxide_core::backend::KernelDispatch,
) -> Result<(), SciOxideError> {
    let (a, b) = standard::product_scales(maximum, image_maximum, image.len())?;
    for (column, coefficient) in columns.iter().zip(coefficients.iter_mut()) {
        if a == 0
            && b == 0
            && let Some(sum) = super::arch::dot(column, image, dispatch)?
        {
            *coefficient = standard::restore(sum, 0)?;
            continue;
        }
        *coefficient = standard::unit_then_normalized(a, b, |a, b| {
            standard::dot(column.iter().zip(image).map(|(&entry, &value)| {
                (
                    standard::scale(entry.conj(), -a),
                    standard::scale(value, -b),
                )
            }))
        })?;
    }
    Ok(())
}

/// Block GEMV update `image -= Q coefficients`, with no public revalidation.
pub(crate) fn subtract_columns<S: NumericScalar>(
    columns: Columns<'_, S>,
    maximum: S::Real,
    coefficients: &[S],
    image: &mut [S],
) -> Result<(), SciOxideError> {
    let (a, b) = standard::product_scales(
        maximum,
        super::standard::slice_maximum(coefficients),
        columns.iter().len(),
    )?;
    if a == 0 && b == 0 {
        // Stream contiguous slices of each basis column. Accumulation order per
        // entry is unchanged, but a small projection tile stays in cache.
        for (tile, output) in image.chunks_mut(64).enumerate() {
            let mut projection = [S::ZERO; 64];
            let start = tile * 64;
            for (column, &coefficient) in columns.iter().zip(coefficients) {
                for (sum, &value) in projection
                    .iter_mut()
                    .zip(&column[start..start + output.len()])
                {
                    *sum = value * coefficient + *sum;
                }
            }
            for (out, projection) in output.iter_mut().zip(projection) {
                *out = crate::checked::finite_candidate(*out - projection)?;
            }
        }
        return Ok(());
    }
    for (row, out) in image.iter_mut().enumerate() {
        let projection = standard::unit_then_normalized(a, b, |a, b| {
            columns
                .iter()
                .zip(coefficients)
                .fold(S::ZERO, |sum, (column, &coefficient)| {
                    standard::scale(column[row], -a) * standard::scale(coefficient, -b) + sum
                })
        })?;
        *out = crate::checked::finite_candidate(*out - projection)?;
    }
    Ok(())
}

fn validate_gemv_input_and_scratch<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    input: VectorView<'_, S>,
    scratch_len: usize,
) -> Result<(), SciOxideError> {
    if input.len != matrix.cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(matrix.cols),
            Dimensions::one(input.len),
        ));
    }
    if scratch_len < matrix.rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(matrix.rows),
            Dimensions::one(scratch_len),
        ));
    }
    Ok(())
}

pub(crate) fn copy_into<S: NumericScalar>(
    source: MatrixView<'_, S>,
    output: MatrixViewMut<'_, S>,
    policy: crate::CopyPolicy,
) -> Result<(), SciOxideError> {
    validate_copy_output(source, &output)?;
    if source.is_empty() {
        return Ok(());
    }
    if policy == crate::CopyPolicy::RequireFinite {
        let _ = source.try_finite()?;
    }
    tiled_copy(source, output);
    Ok(())
}

fn validate_copy_output<S: NumericScalar>(
    source: MatrixView<'_, S>,
    output: &MatrixViewMut<'_, S>,
) -> Result<(), SciOxideError> {
    if output.rows != source.rows || output.cols != source.cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(source.rows, source.cols),
            Dimensions::two(output.rows, output.cols),
        ));
    }
    Ok(())
}

fn tiled_copy<S: NumericScalar>(source: MatrixView<'_, S>, mut output: MatrixViewMut<'_, S>) {
    // When source and output have opposite orientations (a transpose copy),
    // one side is always strided. Working in square tiles keeps both the
    // source and output lines of a tile resident in cache while it is copied.
    let mut row_start = 0;
    while row_start < source.rows {
        let row_end = row_start + (source.rows - row_start).min(TRANSPOSE_TILE_EDGE);
        let mut col_start = 0;
        while col_start < source.cols {
            let col_end = col_start + (source.cols - col_start).min(TRANSPOSE_TILE_EDGE);
            let mut col = col_start;
            while col < col_end {
                let mut row = row_start;
                while row < row_end {
                    output.set_element(row, col, source.get_element(row, col));
                    row += 1;
                }
                col += 1;
            }
            col_start = col_end;
        }
        row_start = row_end;
    }
}

#[cfg(test)]
mod tests {
    use crate::LinalgWorkspace;
    use crate::VectorViewMut;
    use alloc::vec::Vec;

    use super::*;
    use crate::dense::{Matrix, Vector};
    use scioxide_core::Complex;

    fn assert_f64_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .for_each(|(&actual, &expected)| {
                assert!((actual - expected).abs() <= 2.0e-12 * (1.0 + expected.abs()));
            });
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .for_each(|(&actual, &expected)| {
                assert!((actual - expected).abs() <= 2.0e-4 * (1.0 + expected.abs()));
            });
    }

    fn assert_complex_f64_close(actual: &[Complex<f64>], expected: &[Complex<f64>]) {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .for_each(|(&actual, &expected)| {
                assert!((actual - expected).norm() <= 2.0e-12 * (1.0 + expected.norm()));
            });
    }

    fn reference_gemv<S: NumericScalar>(
        matrix: MatrixView<'_, S>,
        input: VectorView<'_, S>,
    ) -> Vec<S> {
        (0..matrix.rows())
            .map(|row| {
                let mut sum = ProductSum::new();
                (0..matrix.cols()).for_each(|col| {
                    sum.add_product(matrix.get_element(row, col), input.get_element(col))
                        .unwrap()
                });
                sum.finish().unwrap()
            })
            .collect()
    }

    #[test]
    fn standard_gemv_and_projections_normalize_only_overflowing_entries() {
        // Row 0 multiplies MAX by MIN_POSITIVE, which is ordinary in the unit
        // frame; row 1 overflows there but cancels once normalized.
        let matrix = Matrix::from_vec(
            2,
            4,
            vec![
                f64::MAX,
                f64::MIN_POSITIVE,
                0.0,
                0.0,
                0.0,
                0.0,
                f64::MAX,
                -f64::MAX,
            ],
        )
        .unwrap();
        let input = Vector::from_vec(vec![f64::MIN_POSITIVE, f64::MAX, 2.0, 2.0]);
        let mut output = Vector::from_vec(vec![7.0; 2]);
        matrix
            .view()
            .mul_vec_into(
                input.view(),
                &crate::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut LinalgWorkspace::new(),
                crate::LinalgExecutionContext::unscoped(
                    KernelContext::sequential().with_accuracy(Accuracy::Standard),
                ),
            )
            .unwrap();
        assert_eq!(output.as_slice()[0].to_bits(), 0x401f_ffff_ffff_ffff);
        assert_eq!(output.as_slice()[1].to_bits(), 0);

        let mut coefficient = [0.0];
        project_columns(
            Columns::Vectors(&[Vector::from_vec(vec![f64::MAX, f64::MIN_POSITIVE])]),
            f64::MAX,
            &[f64::MIN_POSITIVE, f64::MAX],
            f64::MAX,
            &mut coefficient,
            KernelContext::sequential().dispatch(),
        )
        .unwrap();
        assert_eq!(coefficient[0].to_bits(), 0x401f_ffff_ffff_ffff);
        let columns = [
            Vector::from_vec(vec![f64::MAX, 0.0]),
            Vector::from_vec(vec![f64::MIN_POSITIVE, 0.0]),
        ];
        let mut image = [8.0, 3.0];
        subtract_columns(
            Columns::Vectors(&columns),
            f64::MAX,
            &[f64::MIN_POSITIVE, f64::MAX],
            &mut image,
        )
        .unwrap();
        assert_eq!(image, [2.0_f64.powi(-50), 3.0]);
    }

    #[test]
    fn dense_gemv_rejects_raw_non_finite_and_overflow_without_mutation() {
        let non_finite = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
        let input = Vector::from_vec(vec![1.0]);
        let mut output = Vector::from_vec(vec![7.0]);
        assert!(matches!(
            (non_finite.view().mul_vec_into(
                input.view(),
                &crate::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut LinalgWorkspace::new(),
                crate::kernel::tests::test_context()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        assert_eq!(output.as_slice(), &[7.0]);

        let overflowing = Matrix::from_vec(1, 2, vec![f64::MAX, f64::MAX]).unwrap();
        let input = Vector::from_vec(vec![2.0, 2.0]);
        assert!(matches!(
            (overflowing.view().mul_vec_into(
                input.view(),
                &crate::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut LinalgWorkspace::new(),
                crate::kernel::tests::test_context()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(output.as_slice(), &[7.0]);
    }

    #[test]
    fn f64_matches_reference_on_unaligned_tails() {
        for cols in [31, 32, 33, 127, 128, 129] {
            let rows = 5;
            let matrix_storage = std::iter::once(-99.0)
                .chain((0..rows * cols).map(|index| (index as f64 - 19.0) / 23.0))
                .collect::<Vec<_>>();
            let input_storage = std::iter::once(-88.0)
                .chain((0..cols).map(|index| (index as f64 + 3.0) / 29.0))
                .collect::<Vec<_>>();
            let matrix = MatrixView::from_slice(rows, cols, &matrix_storage[1..]).unwrap();
            let input = VectorView::from_slice(&input_storage[1..]);
            let expected = reference_gemv(matrix, input);

            let mut actual = vec![0.0; rows];
            matrix
                .mul_vec_into(
                    input,
                    &crate::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut actual),
                    &mut LinalgWorkspace::new(),
                    crate::kernel::tests::test_context(),
                )
                .unwrap();
            assert_f64_close(&actual, &expected);
        }
    }

    #[test]
    fn f64_matches_reference_for_transposed_tails() {
        let rows = 17;
        let cols = 131;
        let matrix_storage = std::iter::once(-99.0)
            .chain((0..rows * cols).map(|index| (index as f64 - 19.0) / 23.0))
            .collect::<Vec<_>>();
        let matrix = MatrixView::from_slice(rows, cols, &matrix_storage[1..]).unwrap();
        let input = Vector::from_vec((0..rows).map(|index| (index as f64 + 3.0) / 29.0).collect());

        for transformed in [matrix.transpose(), matrix.adjoint()] {
            let expected = reference_gemv(transformed, input.view());

            let mut actual = vec![0.0; cols];
            transformed
                .mul_vec_into(
                    input.view(),
                    &crate::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut actual),
                    &mut LinalgWorkspace::new(),
                    crate::kernel::tests::test_context(),
                )
                .unwrap();
            assert_f64_close(&actual, &expected);
        }
    }

    #[test]
    fn f32_matches_reference_for_all_operations() {
        let rows = 17;
        let cols = 137;
        let matrix = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|index| (index as f32 - 31.0) / 37.0)
                .collect(),
        )
        .unwrap();
        let ordinary_input =
            Vector::from_vec((0..cols).map(|index| (index as f32 + 1.0) / 41.0).collect());
        let transposed_input =
            Vector::from_vec((0..rows).map(|index| (index as f32 + 2.0) / 43.0).collect());

        for (transformed, input, output_len) in [
            (matrix.view(), ordinary_input.view(), rows),
            (matrix.view().transpose(), transposed_input.view(), cols),
            (matrix.view().adjoint(), transposed_input.view(), cols),
        ] {
            let expected = reference_gemv(transformed, input);

            let mut actual = vec![0.0; output_len];
            transformed
                .mul_vec_into(
                    input,
                    &crate::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut actual),
                    &mut LinalgWorkspace::new(),
                    crate::kernel::tests::test_context(),
                )
                .unwrap();
            assert_f32_close(&actual, &expected);
        }
    }

    #[test]
    fn complex_adjoint_matches_reference_for_tails() {
        let rows = 17;
        let cols = 131;
        let matrix_storage = std::iter::once(Complex::new(-99.0, 88.0))
            .chain((0..rows * cols).map(|index| {
                Complex::new((index as f64 - 31.0) / 37.0, (index as f64 + 17.0) / 43.0)
            }))
            .collect::<Vec<_>>();
        let matrix = MatrixView::from_slice(rows, cols, &matrix_storage[1..]).unwrap();
        let input = Vector::from_vec(
            (0..rows)
                .map(|index| Complex::new((index as f64 + 1.0) / 41.0, (index as f64 - 3.0) / 47.0))
                .collect(),
        );
        let expected = reference_gemv(matrix.adjoint(), input.view());

        let mut actual = vec![Complex::new(0.0, 0.0); cols];
        matrix
            .adjoint()
            .mul_vec_into(
                input.view(),
                &crate::LinalgExecutionOptions::new(),
                VectorViewMut::from_slice(&mut actual),
                &mut LinalgWorkspace::new(),
                crate::kernel::tests::test_context(),
            )
            .unwrap();
        assert_complex_f64_close(&actual, &expected);
    }
}
