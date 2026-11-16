//! Convolution and correlation of records and matrices.
pub(crate) mod nd;
mod runs;
pub(crate) mod separable;
pub(crate) mod stream;
use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    ExecutionContext, NdView, NdViewMut, NumericScalar, SciOxideError, StridedLayout,
};
use scioxide_linalg::{Matrix, MatrixView};
/// Computes the full two-dimensional linear convolution of two real or complex
/// matrices.
///
/// ```text
/// c[i, j] = Σₚ Σ_q a[i − p, j − q] b[p, q]
/// ```
///
/// with `a` zero outside its bounds. The result has
/// `a.rows() + b.rows() − 1` rows and `a.cols() + b.cols() − 1` columns. For
/// repeated convolutions with one kernel, other extents or boundaries, use
/// [`ConvolutionNdPlan`](crate::ConvolutionNdPlan), which this function runs
/// once with default options.
///
/// # Errors
///
/// - `InvalidInput` if `a` or `b` has no rows or no columns.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let b = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0])?;
/// let c = scioxide_signal::convolve2d(a.view(), b.view())?;
/// assert_eq!((c.rows(), c.cols()), (3, 3));
/// let expected = [1.0, 2.0, 0.0, 3.0, 3.0, -2.0, 0.0, -3.0, -4.0];
/// assert!(c.as_slice().iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn convolve2d<T: NumericScalar>(
    a: MatrixView<'_, T>,
    b: MatrixView<'_, T>,
) -> Result<Matrix<T>, SciOxideError> {
    let gather = |v: MatrixView<'_, T>| -> Result<_, SciOxideError> {
        let mut data = try_with_capacity(
            checked_mul(
                v.rows(),
                v.cols(),
                scioxide_core::quantity!("matrix_signal"),
            )?,
            scioxide_core::quantity!("matrix_signal"),
        )?;
        for row in 0..v.rows() {
            data.extend(v.row(row)?.iter());
        }
        Ok(data)
    };
    let left = gather(a)?;
    let right = gather(b)?;
    let shape = [a.rows(), a.cols()];
    let kernel_shape = [b.rows(), b.cols()];
    let plan = nd::ConvolutionNdPlan::new(
        &shape,
        NdView::new(&right, StridedLayout::new(&kernel_shape, &[b.cols(), 1])?)?,
        &[0, 1],
        nd::ConvolutionOptions::default(),
    )?;
    let rows = plan.output_shape()[0];
    let cols = plan.output_shape()[1];
    let mut result = try_filled(
        checked_mul(rows, cols, scioxide_core::quantity!("matrix_output"))?,
        T::ZERO,
        scioxide_core::quantity!("matrix_output"),
    )?;
    let mut workspace = plan.workspace(nd::ConvolutionKernel::Prepared)?;
    plan.convolve_into(
        NdView::new(&left, StridedLayout::new(&shape, &[a.cols(), 1])?)?,
        NdViewMut::new(&mut result, StridedLayout::new(&[rows, cols], &[cols, 1])?)?,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Matrix::from_vec(rows, cols, result)
}

/// Computes the full linear convolution of two real or complex records.
///
/// Entry `n` is `Σₖ a[n − k] b[k]`, with both records zero outside their
/// bounds, so the result has `a.len() + b.len() − 1` entries. With `n = a.len()`
/// and `k = b.len()`, the [`Same`](crate::ConvolutionExtent::Same) extent is
/// the slice `⌊(k − 1)/2⌋..⌊(k − 1)/2⌋ + n` of this result, and the
/// [`Valid`](crate::ConvolutionExtent::Valid) extent, for `n ≥ k`, the slice
/// `k − 1..n`. The method is direct summation or FFT multiplication, whichever
/// needs fewer operations; see
/// [`ConvolutionMethod::Automatic`](crate::ConvolutionMethod::Automatic). Use
/// [`ConvolutionNdPlan`](crate::ConvolutionNdPlan) to reuse one kernel or to
/// choose the method.
///
/// # Errors
///
/// - `InvalidInput` if `a` or `b` is empty.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// // (1 + 2z⁻¹)(1 − z⁻¹) = 1 + z⁻¹ − 2z⁻²
/// assert_eq!(
///     scioxide_signal::convolve(&[1.0_f64, 2.0], &[1.0, -1.0])?,
///     [1.0, 1.0, -2.0]
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn convolve<T: NumericScalar>(a: &[T], b: &[T]) -> Result<alloc::vec::Vec<T>, SciOxideError> {
    product(a, b, nd::ConvolutionOptions::default())
}
/// Computes the full cross-correlation of two real or complex records.
///
/// The value at lag `ℓ` is `Σₖ a[k] conj(b[k − ℓ])`. The result lists the lags
/// `−(b.len() − 1), …, a.len() − 1` in ascending order, so it has
/// `a.len() + b.len() − 1` entries and index `b.len() − 1` is lag zero. With
/// `n = a.len()` and `k = b.len()`, the [`Same`](crate::ConvolutionExtent::Same)
/// extent is the slice `⌊(k − 1)/2⌋..⌊(k − 1)/2⌋ + n` of this result, and the
/// [`Valid`](crate::ConvolutionExtent::Valid) extent, for `n ≥ k`, the slice
/// `k − 1..n` of the lags `0, …, n − k`. The method is chosen as in
/// [`convolve`].
///
/// # Errors
///
/// - `InvalidInput` if `a` or `b` is empty.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// let correlation = scioxide_signal::correlate(&[1.0_f64, 2.0, 3.0], &[1.0, 2.0])?;
/// // Lags -1, 0, 1 and 2.
/// let expected = [2.0, 5.0, 8.0, 3.0];
/// assert!(correlation.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn correlate<T: NumericScalar>(a: &[T], b: &[T]) -> Result<alloc::vec::Vec<T>, SciOxideError> {
    product(
        a,
        b,
        nd::ConvolutionOptions::new().with_product(nd::SignalProduct::Correlation),
    )
}
/// Computes the circular convolution of two real or complex records of equal
/// length.
///
/// Entry `n` is `Σₖ a[(n − k) mod N] b[k]`, where `N` is the common length, so
/// the result has `N` entries.
///
/// # Errors
///
/// - `DimensionMismatch` if `a` and `b` have different lengths.
/// - `InvalidInput` if they are empty.
/// - `NonFinite` if `a` or `b` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// // Convolving with a delayed unit impulse rotates the record by one sample.
/// let rotated = scioxide_signal::convolve_circular(&[1.0_f64, 2.0, 3.0], &[0.0, 1.0, 0.0])?;
/// assert!(rotated.iter().zip([3.0, 1.0, 2.0]).all(|(x, e)| (x - e).abs() < 1e-12));
/// let unequal = scioxide_signal::convolve_circular(&[1.0_f64, 2.0], &[1.0]);
/// assert_eq!(unequal.unwrap_err().kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn convolve_circular<T: NumericScalar>(
    a: &[T],
    b: &[T],
) -> Result<alloc::vec::Vec<T>, SciOxideError> {
    crate::util::exact_len(a.len(), b.len())?;
    product(
        a,
        b,
        nd::ConvolutionOptions::new()
            .with_extent(nd::ConvolutionExtent::Circular)
            .with_boundary(nd::SignalBoundary::Periodic),
    )
}
fn product<T: NumericScalar>(
    a: &[T],
    b: &[T],
    options: nd::ConvolutionOptions,
) -> Result<alloc::vec::Vec<T>, SciOxideError> {
    nd::ConvolutionNdPlan::new(
        &[a.len()],
        NdView::new(b, StridedLayout::new(&[b.len()], &[1])?)?,
        &[0],
        options,
    )?
    .convolve(NdView::new(a, StridedLayout::new(&[a.len()], &[1])?)?)
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::util::error_kind;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn two_dimensional_convolution_retains_shape_and_delta_behavior() {
        let input = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let delta = Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap();
        let output = convolve2d(input.view(), delta.view()).unwrap();
        assert_eq!((output.rows(), output.cols()), (2, 2));
        output
            .as_slice()
            .iter()
            .zip(input.as_slice())
            .for_each(|(actual, expected)| assert!((actual - expected).abs() < 1e-12));
    }

    #[test]
    fn two_dimensional_convolution_rejects_empty_and_non_finite_inputs() {
        let empty = Matrix::<f64>::zeros(0, 0).unwrap();
        let unit = Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap();
        assert_eq!(
            error_kind(convolve2d(empty.view(), unit.view())),
            SciOxideErrorKind::InvalidInput
        );
        let invalid = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
        assert_eq!(
            error_kind(convolve2d(invalid.view(), unit.view())),
            SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn two_dimensional_convolution_scales_frequency_products_before_multiplication() {
        let root = f64::MAX.sqrt();
        let input = Matrix::from_vec(2, 2, vec![root * 0.5; 4]).unwrap();
        let kernel = Matrix::from_vec(1, 1, vec![root]).unwrap();
        let output = convolve2d(input.view(), kernel.view()).unwrap();
        let expected = f64::MAX * 0.5;
        assert_eq!((output.rows(), output.cols()), (2, 2));
        output.as_slice().iter().for_each(|value| {
            assert!(value.is_finite());
            assert!((value - expected).abs() <= expected * 1e-14);
        });
    }
}
