//! Analytic signals and envelopes of one-dimensional records.
use crate::util::{capacity, finite};
use crate::{AnalyticSignalNdPlan, SignalExecutionOptions};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, RealScalar, SciOxideError, StridedLayout,
};

/// Computes the analytic signal `z = x + i H(x)` of a real record `x`, treated
/// as one period of a periodic signal.
///
/// `H(x)` is the discrete periodic Hilbert transform. With `X` the DFT of the
/// `n` samples, `z` is the inverse DFT of `w[k] X[k]`, where `w[0] = 1`,
/// `w[k] = 2` for `0 < k < n/2`, `w[n/2] = 1` for even `n`, and `w[k] = 0` for
/// the negative frequencies (see [Conventions](crate#conventions)). The real
/// part of the result is `x` exactly. The cost is `O(n log n)`. For repeated
/// calls or arrays, use [`AnalyticSignalNdPlan`]. This function runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty.
/// - `NonFinite` if `input` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// // The analytic signal of a sampled cosine is the complex exponential.
/// let analytic = scioxide_signal::hilbert(&[1.0_f64, 0.0, -1.0, 0.0])?;
/// let expected = [
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 1.0),
///     Complex::new(-1.0, 0.0),
///     Complex::new(0.0, -1.0),
/// ];
/// assert!(analytic.into_iter().zip(expected).all(|(z, e)| (z - e).norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. L. Marple, "Computing the discrete-time analytic signal via FFT", *IEEE
/// Trans. Signal Process.* 47(9), 2600–2603, 1999.
pub fn hilbert<S: RealScalar>(input: &[S]) -> Result<Vec<Complex<S>>, SciOxideError> {
    let plan = AnalyticSignalNdPlan::new(&[input.len()], &[0], SignalExecutionOptions::new())?;
    let mut output = try_filled(
        input.len(),
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("analytic_output"),
    )?;
    plan.analytic_into(
        NdView::new(input, StridedLayout::new(&[input.len()], &[1])?)?,
        NdViewMut::new(&mut output, StridedLayout::new(&[input.len()], &[1])?)?,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes the envelope `|z[k]|` of a real record, where `z` is its analytic
/// signal as computed by [`hilbert`].
///
/// This function runs with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty.
/// - `NonFinite` if `input` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
///
/// # Examples
///
/// ```
/// // A sampled cosine has a constant unit envelope.
/// let envelope = scioxide_signal::envelope(&[1.0_f64, 0.0, -1.0, 0.0])?;
/// assert!(envelope.iter().all(|magnitude| (magnitude - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn envelope<S: RealScalar>(input: &[S]) -> Result<Vec<S>, SciOxideError> {
    let plan = AnalyticSignalNdPlan::new(&[input.len()], &[0], SignalExecutionOptions::new())?;
    let mut output = try_filled(
        input.len(),
        S::ZERO,
        scioxide_core::quantity!("envelope_output"),
    )?;
    plan.envelope_into(
        NdView::new(input, StridedLayout::new(&[input.len()], &[1])?)?,
        NdViewMut::new(&mut output, StridedLayout::new(&[input.len()], &[1])?)?,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Writes the magnitudes `|z[k]|` of complex samples, such as an analytic or
/// baseband signal, to the start of `output`.
///
/// No Hilbert transform is applied. Use [`envelope`] for a real record.
///
/// # Errors
///
/// - `DimensionMismatch` if `output` is shorter than `input`.
/// - `NonFinite` if `input` contains NaN or infinity.
/// - `NumericalInstability` if a magnitude overflows.
/// - `ExecutionStopped` if `cx` is cancelled or past its deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// let mut out = [0.0; 2];
/// scioxide_signal::complex_envelope_into(
///     &[Complex::new(3.0, 4.0), Complex::new(0.0, -1.0)],
///     &mut out,
///     ExecutionContext::sequential_native(),
/// )?;
/// // |3 + 4i| = 5 and |−i| = 1.
/// assert_eq!(out, [5.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn complex_envelope_into<S: RealScalar>(
    input: &[Complex<S>],
    output: &mut [S],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    capacity(input.len(), output.len())?;
    cx.check()?;
    for z in input {
        crate::util::inputs(core::slice::from_ref(z))?;
        finite(z.norm())?;
    }
    cx.check()?;
    output
        .iter_mut()
        .zip(input)
        .for_each(|(y, z)| *y = z.norm());
    Ok(())
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::util::error_kind;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn rejects_empty_and_non_finite_input() {
        assert_eq!(
            error_kind(hilbert::<f64>(&[])),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            error_kind(hilbert(&[1.0_f64, f64::NAN])),
            SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn preserves_real_part_for_even_and_odd_lengths() {
        // Single-axis analytic signals copy the real input exactly.
        for input in [
            vec![1.0_f64, 2.0, 3.0, 4.0],
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![1e-310, -0.0, 1e300, 0.1],
        ] {
            let result = hilbert(&input).unwrap();
            result.iter().zip(&input).for_each(|(actual, expected)| {
                assert_eq!(actual.re.to_bits(), expected.to_bits());
            });
        }
    }

    #[test]
    fn constant_and_extreme_normalized_signals_remain_finite() {
        let constant = hilbert(&[3.0_f64; 8]).unwrap();
        assert!(constant.iter().all(|value| value.im.abs() < 1e-12));

        let extreme = hilbert(&[f64::MAX * 0.25, 0.0, -f64::MAX * 0.25, 0.0]).unwrap();
        assert!(
            extreme
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }
}
