//! Internal utility helpers shared across all `scioxide-random` modules.
//!
//! All items are `pub(crate)` — they are not part of the public API.

#![allow(clippy::redundant_pub_crate)]

use scioxide_core::{InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError};

use crate::traits::Rng;

/// Fixed SplitMix increment, shared by engine seeding and seed expansion.
pub(crate) const SPLITMIX64_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// Pure SplitMix finalizer; does not add entropy to its input.
#[inline]
pub(crate) const fn splitmix64_mix(mut word: u64) -> u64 {
    // Stafford's "Mix13" xorshift-multiply finalizer as used by Vigna's
    // SplitMix64. It is a bijection on u64, so distinct inputs never collide;
    // the shifts and odd multipliers must stay paired with each other and
    // with published SplitMix64 vectors.
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

/// Fill little-endian native words, discarding the unused final bytes.
/// Width is an internal native-protocol constant in 1..=8. Completed chunks
/// stay written on provider failure; draw accounting belongs to the provider.
/// Empty output never calls `draw`.
#[inline]
pub(crate) fn fill_bytes_with(
    output: &mut [u8],
    width: usize,
    mut draw: impl FnMut() -> Result<u64, SciOxideError>,
) -> Result<(), SciOxideError> {
    debug_assert!((1..=8).contains(&width));
    let mut chunks = output.chunks_exact_mut(width);
    for chunk in chunks.by_ref() {
        let word = draw()?.to_le_bytes();
        chunk.copy_from_slice(&word[..width]);
    }
    let tail = chunks.into_remainder();
    if !tail.is_empty() {
        let word = draw()?.to_le_bytes();
        tail.copy_from_slice(&word[..tail.len()]);
    }
    Ok(())
}

/// Full-width position with a native-word fast path. The optional high word
/// records tracking status; a carry reaches it only once per 2^64 transitions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TransitionCounter {
    low: u64,
    high: Option<u64>,
}

impl TransitionCounter {
    pub(crate) const fn from_position(position: Option<u128>) -> Self {
        match position {
            Some(position) => Self {
                low: position as u64,
                high: Some((position >> 64) as u64),
            },
            None => Self { low: 0, high: None },
        }
    }

    pub(crate) const fn position(self) -> Option<u128> {
        match self.high {
            Some(high) => Some(((high as u128) << 64) | self.low as u128),
            None => None,
        }
    }

    #[inline]
    pub(crate) fn increment(&mut self) {
        self.increment_by(1);
    }

    #[inline]
    pub(crate) fn increment_by(&mut self, count: u64) {
        let (low, carry) = self.low.overflowing_add(count);
        self.low = low;
        if carry {
            self.high = self.high.and_then(|high| high.checked_add(1));
        }
    }

    pub(crate) fn advanced(self, count: u128) -> Self {
        Self::from_position(
            self.position()
                .and_then(|position| position.checked_add(count)),
        )
    }
}

/// Exact attempt budget shared by every provider-driven rejection loop.
pub(crate) const MAX_SAMPLING_ATTEMPTS: usize = 1_024;

/// Repeat one rejection proposal until it accepts, at most
/// [`MAX_SAMPLING_ATTEMPTS`] times; exhaustion reports the attempted count.
#[inline]
pub(crate) fn rejection<T>(
    mut propose: impl FnMut() -> Result<Option<T>, SciOxideError>,
) -> Result<T, SciOxideError> {
    for _ in 0..MAX_SAMPLING_ATTEMPTS {
        if let Some(value) = propose()? {
            return Ok(value);
        }
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

/// One Neumaier-compensated addition of `value` to `(sum, correction)`.
///
/// The correction captures each rounding error regardless of which operand is
/// larger, so long running sums keep the accuracy of their final rounding.
#[inline]
pub(crate) fn neumaier<S: RealScalar>((sum, correction): (S, S), value: S) -> (S, S) {
    let next = sum + value;
    let error = if sum.abs() >= value.abs() {
        (sum - next) + value
    } else {
        (value - next) + sum
    };
    (next, correction + error)
}

/// Draw and validate one provider value from the half-open unit interval.
///
/// Every distribution and sampler reads uniforms through this check. Values
/// from built-in decoders are in range by construction and skip it; values
/// from every other [`Rng`] are checked.
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::NonFinite`] with [`NonFiniteOrigin::Callback`] if the
///   provider returns NaN or infinity.
/// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if the provider returns a finite value
///   outside `[0, 1)`.
///
/// An out-of-range value is attributed to `argument`, the caller's provider parameter.
#[inline]
pub(crate) fn validated_unit_float<S: RealScalar, R: Rng<S> + ?Sized>(
    rng: &mut R,
    argument: scioxide_core::Argument,
) -> Result<S, SciOxideError> {
    rng.checked_float(argument, crate::traits::private::Token)
}

/// Validate even an overridden open-uniform provider before a transform uses it.
#[inline]
pub(crate) fn validated_open_float<S: RealScalar, R: Rng<S> + ?Sized>(
    rng: &mut R,
) -> Result<S, SciOxideError> {
    rng.checked_open_float(crate::traits::private::Token)
}

/// Validate one provider value against the half-open unit interval.
///
/// Downstream transforms such as `-ln(1 - u)` or `u * width` rely on the
/// half-open range: `u == 1` could produce infinities or out-of-support
/// samples, so user-supplied providers are checked here once instead of in
/// every sampler. Built-in decoders prove the range and skip this check.
#[inline]
pub(crate) fn checked_unit_float<S: RealScalar>(
    value: S,
    argument: scioxide_core::Argument,
) -> Result<S, SciOxideError> {
    if !value.is_finite() {
        return Err(non_finite_provider_value());
    }
    if value < S::ZERO || value >= S::ONE {
        return Err(invalid_provider_value(value, argument));
    }
    Ok(value)
}

/// Guard that a parameter is finite.
///
/// # Errors
///
/// - `scioxide_core::SciOxideErrorKind::NonFinite` — if `value` is NaN or infinite.
#[inline]
pub(crate) fn require_finite_param<S: RealScalar>(
    value: S,
    param: &str,
) -> Result<(), SciOxideError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(non_finite_parameter(value, param))
    }
}

// Keep formatting and its stack scratch out of inlined numerical/provider
// checks. Ordinary success paths neither construct diagnostics nor allocate.
#[cold]
#[inline(never)]
fn non_finite_parameter<S: RealScalar>(value: S, param: &str) -> SciOxideError {
    SciOxideError::non_finite(
        scioxide_core::NonFiniteOrigin::Input,
        alloc::format!("{param} is non-finite: {value:?}"),
    )
}

#[cold]
#[inline(never)]
fn non_finite_provider_value() -> SciOxideError {
    SciOxideError::non_finite(
        NonFiniteOrigin::Callback,
        "Rng::next_float returned a non-finite value",
    )
}

#[cold]
#[inline(never)]
fn invalid_provider_value<S: RealScalar>(
    value: S,
    argument: scioxide_core::Argument,
) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        InvalidInputReason::OutOfRange,
        alloc::format!("provider value must be in [0, 1), got {value:?}"),
    )
}

/// Bounded rejection of the zero endpoint. The provider closure validates or
/// constructs a half-open unit value before it reaches this shared loop.
#[inline]
pub(crate) fn open_probability<S: RealScalar>(
    limit: usize,
    mut next: impl FnMut() -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    for _ in 0..limit {
        let value = next()?;
        if value > S::ZERO {
            return Ok(value);
        }
    }
    Err(crate::law::iteration_limit(limit))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{decode_u32, decode_u64};
    use alloc::vec::Vec;

    #[test]
    fn native_byte_widths_discard_tails_and_never_draw_for_empty_output() {
        for width in 1..=8 {
            for length in 0..=2 * width + 1 {
                let mut draws = 0;
                let mut output = vec![0xa5; length];
                fill_bytes_with(&mut output, width, || {
                    draws += 1;
                    Ok(0x0807_0605_0403_0201)
                })
                .unwrap();
                assert_eq!(draws, length.div_ceil(width));
                let expected: Vec<_> = (1_u8..=width as u8).cycle().take(length).collect();
                assert_eq!(output, expected);
            }
        }
    }

    #[test]
    fn byte_provider_failure_publishes_only_completed_chunks() {
        for width in 1..=8 {
            let mut draws = 0;
            let mut output = vec![0xa5; 2 * width + 1];
            let error = fill_bytes_with(&mut output, width, || {
                draws += 1;
                if draws == 2 {
                    Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("provider"),
                        InvalidInputReason::Inconsistent,
                        "injected draw failure",
                    ))
                } else {
                    Ok(0x0807_0605_0403_0201)
                }
            })
            .unwrap_err();
            assert_eq!(draws, 2);
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
            assert_eq!(&output[..width], &(1_u8..=width as u8).collect::<Vec<_>>());
            assert!(output[width..].iter().all(|&byte| byte == 0xa5));
        }
    }

    struct FixedProvider(f64);

    impl crate::traits::BitGenerator for FixedProvider {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for FixedProvider {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            Ok(self.0)
        }
        fn next_open_float(&mut self) -> Result<f64, SciOxideError> {
            Ok(self.0)
        }
    }

    #[test]
    fn overridden_open_providers_are_validated_before_sampling() {
        use crate::Distribution;
        use scioxide_core::SciOxideErrorKind;
        let histogram = crate::Histogram::from_masses(&[0.0, 1.0], &[1.0]).unwrap();
        for value in [
            0.0,
            -1.0,
            1.0,
            2.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let expected = if value.is_finite() {
                SciOxideErrorKind::InvalidInput
            } else {
                SciOxideErrorKind::NonFinite
            };
            let mut provider = FixedProvider(value);
            assert_eq!(
                crate::GumbelRight::new()
                    .sample(&mut provider)
                    .unwrap_err()
                    .kind(),
                expected
            );
            assert_eq!(
                histogram.sample(&mut &mut provider).unwrap_err().kind(),
                expected
            );
        }
        let mut provider = FixedProvider(0.5);
        assert_eq!(histogram.sample(&mut provider).unwrap(), 0.5);
    }

    #[test]
    fn validated_unit_float_accepts_both_valid_boundaries() {
        let mut zero = FixedProvider(0.0);
        let mut below_one = FixedProvider(1.0_f64.next_down());
        assert_eq!(
            validated_unit_float(&mut zero, scioxide_core::argument!("rng")).unwrap(),
            0.0
        );
        assert_eq!(
            validated_unit_float(&mut below_one, scioxide_core::argument!("rng")).unwrap(),
            1.0_f64.next_down()
        );
    }

    #[test]
    fn validated_unit_float_classifies_non_finite_provider_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut provider = FixedProvider(value);
            let error =
                validated_unit_float(&mut provider, scioxide_core::argument!("rng")).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ));
        }
    }

    #[test]
    fn validated_unit_float_classifies_finite_range_violations() {
        for value in [-f64::MIN_POSITIVE, 1.0, 1.0 + f64::EPSILON] {
            let mut provider = FixedProvider(value);
            let error =
                validated_unit_float(&mut provider, scioxide_core::argument!("rng")).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::OutOfRange,
                    ..
                }
            ));
        }
    }

    // ---- require_finite_param ----

    #[test]
    fn test_require_finite_param_ok() {
        assert!(require_finite_param(1.0_f64, "x").is_ok());
    }

    #[test]
    fn test_require_finite_param_nan() {
        let error = require_finite_param(f64::NAN, "x").unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_require_finite_param_inf() {
        let error = require_finite_param(f64::INFINITY, "x").unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    // ---- decode_u64 ----

    #[test]
    fn test_decode_u64_zero_for_both_scalars() {
        let f32_value: f32 = decode_u64(0);
        let f64_value: f64 = decode_u64(0);
        assert_eq!(f32_value, 0.0);
        assert_eq!(f64_value, 0.0);
    }

    #[test]
    fn test_decode_u64_max_for_f32() {
        let value: f32 = decode_u64(u64::MAX);
        assert_eq!(value, f32::from_bits(1.0_f32.to_bits() - 1));
    }

    #[test]
    fn test_decode_u64_max_for_f64() {
        let value: f64 = decode_u64(u64::MAX);
        assert_eq!(value, f64::from_bits(1.0_f64.to_bits() - 1));
    }

    // ---- decode_u32 ----

    #[test]
    fn test_decode_u32_zero_for_both_scalars() {
        let f32_value: f32 = decode_u32(0);
        let f64_value: f64 = decode_u32(0);
        assert_eq!(f32_value, 0.0);
        assert_eq!(f64_value, 0.0);
    }

    #[test]
    fn test_decode_u32_max_for_f32() {
        let value: f32 = decode_u32(u32::MAX);
        assert_eq!(value, f32::from_bits(1.0_f32.to_bits() - 1));
    }

    #[test]
    fn test_decode_u32_max_for_f64() {
        let value: f64 = decode_u32(u32::MAX);
        assert_eq!(value, 1.0 - 1.0 / 4_294_967_296.0);
    }
}
