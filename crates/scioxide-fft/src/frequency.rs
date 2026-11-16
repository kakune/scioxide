//! Frequency grids, frequency shifts and smooth transform lengths.
use crate::smooth_len::fast_len;
use crate::validation::arithmetic_overflow;
use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{NumericalIssue, ScalarConversionReason, SciOxideError};
fn cast_isize<S: scioxide_core::RealScalar>(value: isize) -> Result<S, SciOxideError> {
    S::try_from_i64_exact(value as i64)
}

/// Returns the sample frequencies of the bins of an `n`-point DFT.
///
/// For sample spacing `d`, bin `k` of [`fft`](crate::fft) has frequency
///
/// ```text
/// f = [0, 1, …, ⌈n/2⌉ − 1, −⌊n/2⌋, …, −1] / (n d),
/// ```
///
/// in cycles per unit of `d`. For even `n` the Nyquist bin `n/2` is listed as
/// the negative frequency `−1/(2d)`. A negative `d` reverses the signs. Each
/// entry is computed as `(k/n)/d` from the exact integer `k`, so the grid does
/// not accumulate rounding error. For `n = 1` the result is `[0]`.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero or `d` is zero.
/// - `NonFinite` if `d` is NaN or infinite.
/// - `ScalarConversion` if `n` or a bin index is not exactly representable in
///   `S`, or `S` cannot resolve two adjacent frequencies.
/// - `NumericalInstability` if `1/(n d)` or a frequency overflows or underflows
///   to zero.
///
/// # Examples
///
/// For even lengths the Nyquist bin starts the negative-frequency half.
///
/// ```
/// let frequencies = scioxide_fft::fftfreq(4, 0.5_f64)?;
/// let expected = [0.0, 0.5, -1.0, -0.5];
/// assert_eq!(frequencies.len(), expected.len());
/// for (a, b) in frequencies.iter().zip(expected) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fftfreq<S: scioxide_core::RealScalar>(n: usize, d: S) -> Result<Vec<S>, SciOxideError> {
    let scaled = FrequencyScale::<S>::new(n, d)?;
    let mut freqs = try_with_capacity(n, scioxide_core::quantity!("fft_frequency_grid"))?;
    let pos_count = n.div_ceil(2);
    for i in 0..pos_count {
        freqs.push(scaled.bin(isize::try_from(i).map_err(|_| {
            SciOxideError::scalar_conversion(
                scioxide_core::ConversionType::Usize,
                scioxide_core::ConversionType::Isize,
                ScalarConversionReason::OutOfRange,
            )
        })?)?);
    }
    for magnitude in (1..=n / 2).rev() {
        let index = isize::try_from(magnitude).map_err(|_| {
            SciOxideError::scalar_conversion(
                scioxide_core::ConversionType::Usize,
                scioxide_core::ConversionType::Isize,
                ScalarConversionReason::OutOfRange,
            )
        })?;
        freqs.push(scaled.bin(-index)?);
    }
    scaled.validate_fft_grid(&freqs, pos_count)?;
    Ok(freqs)
}

/// Returns the sample frequencies of the bins kept by [`rfft`](crate::rfft).
///
/// For sample spacing `d`, the `⌊n/2⌋ + 1` bins have frequencies
///
/// ```text
/// f = [0, 1, …, ⌊n/2⌋] / (n d),
/// ```
///
/// in cycles per unit of `d`. For even `n` the last bin is the Nyquist
/// frequency `+1/(2d)`. Entries are computed as in [`fftfreq`]. For `n = 1` the
/// result is `[0]`.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero or `d` is zero.
/// - `NonFinite` if `d` is NaN or infinite.
/// - `ScalarConversion` if `n` or a bin index is not exactly representable in
///   `S`, or `S` cannot resolve two adjacent frequencies.
/// - `NumericalInstability` if `1/(n d)` or a frequency overflows or underflows
///   to zero.
///
/// # Examples
///
/// Packed bins have nonnegative frequencies, including the even-length Nyquist
/// bin.
///
/// ```
/// let frequencies = scioxide_fft::rfftfreq(4, 0.5_f64)?;
/// let expected = [0.0, 0.5, 1.0];
/// assert_eq!(frequencies.len(), expected.len());
/// for (a, b) in frequencies.iter().zip(expected) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn rfftfreq<S: scioxide_core::RealScalar>(n: usize, d: S) -> Result<Vec<S>, SciOxideError> {
    let scaled = FrequencyScale::<S>::new(n, d)?;
    let m = n / 2 + 1;
    let mut freqs = try_with_capacity(m, scioxide_core::quantity!("real_fft_frequency_grid"))?;
    for i in 0..m {
        freqs.push(scaled.bin(isize::try_from(i).map_err(|_| {
            SciOxideError::scalar_conversion(
                scioxide_core::ConversionType::Usize,
                scioxide_core::ConversionType::Isize,
                ScalarConversionReason::OutOfRange,
            )
        })?)?);
    }
    scaled.validate_monotonic_branch(&freqs)?;
    Ok(freqs)
}

struct FrequencyScale<S> {
    n: S,
    spacing: S,
}

impl<S: scioxide_core::RealScalar> FrequencyScale<S> {
    fn new(n: usize, spacing: S) -> Result<Self, SciOxideError> {
        if n == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("n"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "frequency-grid length must be greater than zero",
            ));
        }
        if spacing == S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("d"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "sample spacing must be non-zero",
            ));
        }
        if !spacing.is_finite() {
            return Err(crate::validation::nonfinite());
        }
        let n = S::try_from_usize_exact(n)?;
        // A singleton has only the zero bin and needs no reciprocal.
        // `(1 / n) / d` is the scaled reciprocal of `n * d`: it avoids both
        // an overflowing product for large spacing and an overflowing `1 / d`
        // for subnormal spacing when the final reciprocal is representable.
        if n != S::ONE {
            let reciprocal = (S::ONE / n) / spacing;
            if !reciprocal.is_finite() {
                return Err(arithmetic_overflow(
                    "the reciprocal of n*d is outside the finite range",
                ));
            }
            if reciprocal == S::ZERO {
                return Err(crate::validation::numerical(
                    NumericalIssue::Underflow,
                    "the reciprocal of n*d is not representable as a non-zero frequency step",
                ));
            }
        }
        Ok(Self { n, spacing })
    }

    fn bin(&self, index: isize) -> Result<S, SciOxideError> {
        if index == 0 {
            return Ok(S::ZERO);
        }
        let exact_index = cast_isize::<S>(index)?;
        // Use the exact signed integer and one scaled quotient rather than an
        // incrementally accumulated grid, which would drift or collapse.
        let value = (exact_index / self.n) / self.spacing;
        if !value.is_finite() {
            return Err(arithmetic_overflow(
                "a generated frequency bin is outside the finite range",
            ));
        }
        if value == S::ZERO {
            return Err(crate::validation::numerical(
                NumericalIssue::Underflow,
                "a non-zero frequency bin collapsed to zero",
            ));
        }
        Ok(value)
    }

    fn validate_fft_grid(&self, values: &[S], positive_count: usize) -> Result<(), SciOxideError> {
        self.validate_monotonic_branch(&values[..positive_count])?;
        self.validate_monotonic_branch(&values[positive_count..])
    }

    fn validate_monotonic_branch(&self, values: &[S]) -> Result<(), SciOxideError> {
        // Distinct integer bins must stay strictly ordered after scaling
        // (decreasing when the spacing is negative). Equal or reordered
        // neighbours mean the scalar cannot resolve adjacent bins, which is
        // reported as an inexact index conversion rather than a silent grid
        // with duplicated frequencies.
        let increasing = self.spacing > S::ZERO;
        if values.windows(2).all(|pair| {
            if increasing {
                pair[0] < pair[1]
            } else {
                pair[0] > pair[1]
            }
        }) {
            Ok(())
        } else {
            Err(SciOxideError::scalar_conversion(
                scioxide_core::ConversionType::Isize,
                <S as scioxide_core::backend::Scalar>::kind().into(),
                ScalarConversionReason::Inexact,
            ))
        }
    }
}

/// Moves the zero-frequency bin to the center of a spectrum.
///
/// The result is `input` rotated left by `⌈n/2⌉`, so bin 0 moves to index
/// `⌊n/2⌋` and, for `d > 0`, the frequencies of [`fftfreq`] increase. An
/// empty input gives an empty result. [`ifftshift`] is the inverse;
/// [`fftshift_nd`](crate::fftshift_nd) shifts selected axes of an array.
///
/// # Errors
///
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// For odd lengths the zero-frequency bin moves to the middle entry.
///
/// ```
/// let output = scioxide_fft::fftshift(&[0, 1, 2, 3, 4])?;
/// assert_eq!(output, [3, 4, 0, 1, 2]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fftshift<T: Clone>(input: &[T]) -> Result<Vec<T>, SciOxideError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mid = input.len().div_ceil(2);
    let mut out = try_with_capacity(input.len(), scioxide_core::quantity!("fft_shift_output"))?;
    out.extend_from_slice(&input[mid..]);
    out.extend_from_slice(&input[..mid]);
    Ok(out)
}

/// Moves the center bin of a shifted spectrum back to index zero.
///
/// The result is `input` rotated left by `⌊n/2⌋`, which inverts [`fftshift`].
/// The two rotations differ for odd `n`. An empty input gives an empty result.
///
/// # Errors
///
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// This undoes the [`fftshift`] example.
///
/// ```
/// let output = scioxide_fft::ifftshift(&[3, 4, 0, 1, 2])?;
/// assert_eq!(output, [0, 1, 2, 3, 4]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ifftshift<T: Clone>(input: &[T]) -> Result<Vec<T>, SciOxideError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mid = input.len() / 2;
    let mut out = try_with_capacity(
        input.len(),
        scioxide_core::quantity!("inverse_fft_shift_output"),
    )?;
    out.extend_from_slice(&input[mid..]);
    out.extend_from_slice(&input[..mid]);
    Ok(out)
}

/// Returns the smallest 2/3/5/7-smooth length that is at least `n`.
///
/// A length `2^a 3^b 5^c 7^d` transforms with mixed-radix Cooley–Tukey stages
/// only, so zero-padding a sequence to this length is usually the cheapest
/// choice. Every other length is also supported, through Rader's or
/// Bluestein's algorithm. The result is the same for real and complex
/// transforms; it does not guarantee the fastest length on every processor.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `n` is zero.
/// - `SizeOverflow` if no 2/3/5/7-smooth length at least `n` fits in `usize`.
///
/// # Examples
///
/// ```
/// // 11 is prime; 12 = 2² · 3 is the next smooth length.
/// assert_eq!(scioxide_fft::next_fast_len(11)?, 12);
/// assert_eq!(scioxide_fft::next_fast_len(1)?, 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn next_fast_len(n: usize) -> Result<usize, SciOxideError> {
    fast_len(n, true, scioxide_core::argument!("n"))
}

/// Returns the largest 2/3/5/7-smooth length that is at most `n`.
///
/// Use it to truncate a sequence to a length that transforms with mixed-radix
/// Cooley–Tukey stages only; see [`next_fast_len`].
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `n` is zero.
///
/// # Examples
///
/// ```
/// // 11 is prime; 10 = 2 · 5 is the previous smooth length.
/// assert_eq!(scioxide_fft::prev_fast_len(11)?, 10);
/// assert_eq!(scioxide_fft::prev_fast_len(1)?, 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn prev_fast_len(n: usize) -> Result<usize, SciOxideError> {
    fast_len(n, false, scioxide_core::argument!("n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{NumericalIssue, ScalarConversionReason};

    #[test]
    fn test_fftfreq_even_f64() {
        let f: Vec<f64> = fftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        assert_eq!(f.len(), 8);
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn test_fftfreq_odd_f64() {
        let f: Vec<f64> = fftfreq(5, 1.0).unwrap();
        let expected = [0.0, 0.2, 0.4, -0.4, -0.2];
        assert_eq!(f.len(), 5);
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn test_fftfreq_with_spacing() {
        let f: Vec<f64> = fftfreq(4, 0.5).unwrap();
        // [0, 1, -2, -1] / (4*0.5) = [0, 0.5, -1.0, -0.5]
        let expected = [0.0, 0.5, -1.0, -0.5];
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fftfreq_n_zero() {
        let error = fftfreq::<f64>(0, 1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_fftfreq_d_zero() {
        let error = fftfreq::<f64>(8, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_fftfreq_d_non_finite() {
        let error = fftfreq::<f64>(8, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = fftfreq::<f64>(8, f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_fftfreq_f32() {
        let f: Vec<f32> = fftfreq(4, 1.0_f32).unwrap();
        let expected: [f32; 4] = [0.0, 0.25, -0.5, -0.25];
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn frequency_grids_preserve_odd_bin_placement_and_monotonic_branches() {
        for n in [3, 5, 7, 11] {
            let full = fftfreq::<f64>(n, 1.0).unwrap();
            let positive_count = n.div_ceil(2);
            assert_eq!(full[0], 0.0);
            assert!(
                full[..positive_count]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            assert!(
                full[positive_count..]
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            assert!(full[positive_count] < 0.0);

            let real = rfftfreq::<f64>(n, 1.0).unwrap();
            assert_eq!(real.len(), n / 2 + 1);
            assert!(real.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn frequency_grids_handle_extreme_spacing_without_silent_collapse() {
        let full = fftfreq::<f64>(11, f64::MAX).unwrap();
        assert!(full.iter().all(|value| value.is_finite()));
        assert!(full.iter().skip(1).all(|value| *value != 0.0));
        let real = rfftfreq::<f32>(3, f32::MAX).unwrap();
        assert!(real[1] > 0.0 && real[1].is_finite());

        let subnormal = f64::from_bits(1);
        let error = fftfreq::<f64>(11, subnormal).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn frequency_grids_reject_inexact_integer_indices_before_allocation() {
        let inexact_f32_length = (1_usize << 25) + 1;
        let error = rfftfreq::<f32>(inexact_f32_length, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ScalarConversion {
                reason: ScalarConversionReason::Inexact,
                ..
            }
        ));
    }

    #[test]
    fn test_rfftfreq_even_f64() {
        let f: Vec<f64> = rfftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        assert_eq!(f.len(), 5);
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rfftfreq_odd_f64() {
        let f: Vec<f64> = rfftfreq(5, 1.0).unwrap();
        let expected = [0.0, 0.2, 0.4];
        assert_eq!(f.len(), 3);
        for (a, b) in f.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rfftfreq_errors() {
        let error = rfftfreq::<f64>(0, 1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = rfftfreq::<f64>(4, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = rfftfreq::<f64>(4, f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_fftshift_even() {
        let input = [0, 1, 2, 3, 4, 5];
        let shifted = fftshift(&input).unwrap();
        assert_eq!(shifted, vec![3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn test_fftshift_odd() {
        let input = [0, 1, 2, 3, 4];
        let shifted = fftshift(&input).unwrap();
        assert_eq!(shifted, vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn test_fftshift_empty() {
        let input: [i32; 0] = [];
        assert!(fftshift(&input).unwrap().is_empty());
    }

    #[test]
    fn test_ifftshift_even() {
        let input = [3, 4, 5, 0, 1, 2];
        let unshifted = ifftshift(&input).unwrap();
        assert_eq!(unshifted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_ifftshift_odd() {
        let input = [0, 1, 2, 3, 4];
        let unshifted = ifftshift(&input).unwrap();
        assert_eq!(unshifted, vec![2, 3, 4, 0, 1]);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip_even() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(ifftshift(&fftshift(&input).unwrap()).unwrap(), input);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip_odd() {
        for n in [3, 5, 7, 11] {
            let input: Vec<_> = (0..n).collect();
            let shifted = fftshift(&input).unwrap();
            assert_eq!(shifted[n / 2], 0);
            assert_eq!(ifftshift(&shifted).unwrap(), input);
            assert_eq!(fftshift(&ifftshift(&input).unwrap()).unwrap(), input);
        }
    }

    #[test]
    fn exhaustive_small_and_integer_boundaries() {
        fn smooth(mut n: usize) -> bool {
            for p in [2, 3, 5, 7] {
                while n.is_multiple_of(p) {
                    n /= p;
                }
            }
            n == 1
        }
        for n in 1..2048 {
            let next = super::next_fast_len(n).unwrap();
            let prev = super::prev_fast_len(n).unwrap();
            assert!(smooth(next) && smooth(prev) && prev <= n && next >= n);
            assert!((prev + 1..n).all(|v| !smooth(v)));
            assert!((n..next).all(|v| !smooth(v)));
        }
        assert!(super::next_fast_len(0).is_err());
        assert!(super::prev_fast_len(0).is_err());
        assert!(super::next_fast_len(usize::MAX).is_err());
        assert!(smooth(super::prev_fast_len(usize::MAX).unwrap()));
    }
}
