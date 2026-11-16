//! Softmax, log-softmax and log-sum-exp functions that return a `Vec`.
use super::{Normalization, normalize, stage_logsumexp, stage_weighted_logsumexp, weighted_lanes};
use crate::backend::try_filled;
use crate::{
    ExecutionContext, ExecutionPolicy, LogValue, NdView, NdViewMut, NumericScalar, RealScalar,
    SciOxideError, StridedLayout,
};
use alloc::vec::Vec;

fn normalize_slice<S: RealScalar>(
    input: &[S],
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    let mut output = try_filled(
        input.len(),
        S::ZERO,
        crate::quantity!("normalization_output"),
    )?;
    normalize(input.iter().copied(), &mut output, normalization)?;
    Ok(output)
}

/// Returns the softmax `pᵢ = e^(xᵢ − m) / Σⱼ e^(xⱼ − m)`, `m = maxⱼ xⱼ`, of a slice.
///
/// Masked entries (`−∞`) receive `+0`. Use [`softmax_into`](crate::softmax_into)
/// to write into a buffer you provide.
///
/// # Errors
///
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `DomainError` if the input is empty or every entry is masked.
/// - `NumericalInstability` if a result overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// let p = scioxide_core::softmax(&[0.0_f64, f64::NEG_INFINITY])?;
/// assert_eq!(p, [1.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn softmax<S: RealScalar>(input: &[S]) -> Result<Vec<S>, SciOxideError> {
    normalize_slice(input, Normalization::Probability)
}
/// Returns the log-softmax `ln pᵢ = xᵢ − ln Σⱼ e^(xⱼ)` of a slice.
///
/// Masked entries (`−∞`) receive `−∞`. Use
/// [`log_softmax_into`](crate::log_softmax_into) to write into a buffer you
/// provide.
///
/// # Errors
///
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `DomainError` if the input is empty or every entry is masked.
/// - `NumericalInstability` if a result overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// let log_p = scioxide_core::log_softmax(&[0.0_f64, f64::NEG_INFINITY])?;
/// assert_eq!(log_p, [0.0, f64::NEG_INFINITY]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_softmax<S: RealScalar>(input: &[S]) -> Result<Vec<S>, SciOxideError> {
    normalize_slice(input, Normalization::Log)
}

fn normalize_axis<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    let layout = input.layout();
    if layout.axis(axis)?.is_empty() {
        return Ok(Vec::new());
    }
    let n = layout.shape().len();
    if n == 0 {
        return Err(crate::stable::domain(crate::argument!("input")));
    }
    let mut output = try_filled(n, S::ZERO, crate::quantity!("normalization_output"))?;
    let mut scratch = try_filled(n, S::ZERO, crate::quantity!("normalization_scratch"))?;
    // The rank is unbounded, so the row-major strides of the result are stored
    // here rather than in a `StridedLayout::row_major`.
    let extents = layout.shape().extents();
    let mut strides = try_filled(extents.len(), 0, crate::quantity!("normalization_strides"))?;
    strides
        .iter_mut()
        .zip(extents)
        .rev()
        .fold(1, |stride, (slot, &extent)| {
            *slot = stride;
            stride * extent
        });
    super::axis_normalize(
        input,
        axis,
        NdViewMut::new(&mut output, StridedLayout::new(extents, &strides)?)?,
        &mut scratch,
        normalization,
        ExecutionPolicy::Sequential,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

/// Returns the softmax of every lane of `input` along `axis`.
///
/// Each lane `x` along `axis` is normalized independently, as in [`softmax`]:
///
/// ```text
/// pᵢ = e^(xᵢ − m) / Σⱼ e^(xⱼ − m),   m = maxⱼ xⱼ
/// ```
///
/// A `−∞` logit is a masked entry and receives `+0`. The result has one value
/// per element of `input`, in row-major order (last axis fastest); an array
/// with no lanes gives an empty vector. It runs sequentially and allocates the
/// result, a staging buffer of the same length and one stride per axis. Use
/// [`softmax_axis_into`](crate::softmax_axis_into) to run in parallel and
/// write into buffers you provide.
///
/// # Errors
///
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `DomainError` if a lane is empty or fully masked.
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `NumericalInstability` if a result overflows.
/// - `AllocationFailure` if the result or its staging cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, softmax_axis};
///
/// let x = [0.0, 1.0, 2.0, 5.0, 5.0, f64::NEG_INFINITY];
/// let p = softmax_axis(NdView::row_major(&x, &[2, 3])?, 1)?;
/// let total = 1.0 + 1.0_f64.exp() + 2.0_f64.exp();
/// assert!((p[2] - 2.0_f64.exp() / total).abs() < 1.0e-14);
/// assert!((p[3] - 0.5).abs() < 1.0e-14 && p[5] == 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn softmax_axis<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
) -> Result<Vec<S>, SciOxideError> {
    normalize_axis(input, axis, Normalization::Probability)
}

/// Returns the log-softmax of every lane of `input` along `axis`.
///
/// Each lane `x` along `axis` receives `ln pᵢ = xᵢ − m − ln Σⱼ e^(xⱼ − m)`,
/// `m = maxⱼ xⱼ`, as in [`log_softmax`]. A `−∞` logit is a masked entry and
/// receives `−∞`. Order, execution and allocation are those of
/// [`softmax_axis`]; [`log_softmax_axis_into`](crate::log_softmax_axis_into)
/// writes into buffers you provide.
///
/// # Errors
///
/// The errors of [`softmax_axis`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, log_softmax_axis};
///
/// let x = [0.0, 1.0, 2.0, 5.0, 5.0, f64::NEG_INFINITY];
/// let log_p = log_softmax_axis(NdView::row_major(&x, &[2, 3])?, 1)?;
/// let log_total = (1.0 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
/// assert!((log_p[2] - (2.0 - log_total)).abs() < 1.0e-14);
/// assert!((log_p[3] + core::f64::consts::LN_2).abs() < 1.0e-14);
/// assert_eq!(log_p[5], f64::NEG_INFINITY);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_softmax_axis<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
) -> Result<Vec<S>, SciOxideError> {
    normalize_axis(input, axis, Normalization::Log)
}

/// Returns the log-sum-exp `ln Σᵢ e^(xᵢ)` of every lane of `input` along `axis`.
///
/// Each lane is reduced as in [`logsumexp`](crate::logsumexp), with its
/// accuracy; an empty or fully masked lane gives `−∞` (`(−∞, +0)` for complex
/// values). The result has one value per lane, in row-major order of the
/// remaining axes (the order of [`StridedLayout::axis`]); an array with no
/// lanes gives an empty vector. It runs sequentially; use
/// [`logsumexp_axis_into`](crate::logsumexp_axis_into) to run in parallel and
/// write into buffers you provide.
///
/// # Errors
///
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `NonFinite` if a logit is NaN or `+∞`, or an imaginary part is not finite.
/// - `DomainError` if the terms of a complex lane cancel exactly.
/// - `NumericalInstability` if a result overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, logsumexp_axis};
///
/// let x = [0.0, 1.0, 2.0, 5.0, 5.0, f64::NEG_INFINITY];
/// let s = logsumexp_axis(NdView::row_major(&x, &[2, 3])?, 1)?;
/// let expected = (1.0 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
/// assert!((s[0] - expected).abs() < 1.0e-14);
/// assert!((s[1] - (5.0 + core::f64::consts::LN_2)).abs() < 1.0e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn logsumexp_axis<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    axis: usize,
) -> Result<Vec<T>, SciOxideError> {
    let batches = input.layout().axis(axis)?;
    let mut output = try_filled(batches.len(), T::ZERO, crate::quantity!("logsumexp_output"))?;
    stage_logsumexp(
        input,
        batches,
        &mut output,
        ExecutionPolicy::Sequential,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

/// Returns the weighted log-sum-exp `ln Σᵢ wᵢ e^(xᵢ)` of every lane along `axis`.
///
/// For real logits `x` and real or complex weights `w` of the same extents,
/// each lane is reduced as in
/// [`weighted_logsumexp`](crate::weighted_logsumexp), with its accuracy. The
/// result is a [`LogValue`], which represents an exact zero sum, such as an
/// empty or fully masked lane or exact cancellation, without an infinity.
/// Weights must be finite, also on masked terms, and may use other strides than
/// `input`. Order, execution and allocation are those of [`logsumexp_axis`];
/// [`weighted_logsumexp_axis_into`](crate::weighted_logsumexp_axis_into)
/// writes into buffers you provide.
///
/// # Errors
///
/// - `DimensionMismatch` if the extents of `weights` differ from those of
///   `input`.
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `NonFinite` if a logit is NaN or `+∞`, or a weight is not finite.
/// - `NumericalInstability` if a result overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, weighted_logsumexp_axis};
///
/// let x = [0.0_f64, 1.0, 2.0, 0.0, 0.0, 0.0];
/// let w = [1.0_f64, 1.0, 1.0, 1.0, -1.0, 0.0];
/// let input = NdView::row_major(&x, &[2, 3])?;
/// let weights = NdView::row_major(&w, &[2, 3])?;
/// let s = weighted_logsumexp_axis(input, weights, 1)?;
/// let expected = (1.0 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
/// assert!(s[0].log_magnitude().is_some_and(|log| (log - expected).abs() < 1.0e-14));
/// // Equal terms with opposite signs cancel exactly.
/// assert!(s[1].log_magnitude().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn weighted_logsumexp_axis<T: NumericScalar>(
    input: NdView<'_, '_, T::Real>,
    weights: NdView<'_, '_, T>,
    axis: usize,
) -> Result<Vec<LogValue<T, T::Real>>, SciOxideError> {
    let lanes = weighted_lanes(input, weights, axis)?;
    let mut output = try_filled(
        lanes.0.len(),
        LogValue::ZERO,
        crate::quantity!("logsumexp_output"),
    )?;
    stage_weighted_logsumexp(
        input,
        weights,
        lanes,
        &mut output,
        ExecutionPolicy::Sequential,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
