//! FFT-local integer-reduced roots, rounded ratios and compensated bounded sums.
use crate::validation::arithmetic_overflow;
pub(crate) use scioxide_core::backend::scalar_from_u128_rounded;
use scioxide_core::{Argument, Complex, RealScalar, SciOxideError};

pub(crate) fn scalar_from_usize_rounded<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    scalar_from_u128_rounded(value as u128)
}

/// Compute a finite scalar ratio from integral numerator and denominator.
///
/// Both operands are reduced by their common power of two before independent
/// rounded conversion. No integer index is round-tripped through a float.
/// A zero denominator is attributed to `argument`, the caller parameter the denominator derives from.
pub(crate) fn scalar_ratio_u128<S: RealScalar>(
    numerator: u128,
    denominator: u128,
    argument: Argument,
) -> Result<S, SciOxideError> {
    if denominator == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "FFT integer ratio denominator must be non-zero",
        ));
    }
    if numerator == 0 {
        return Ok(S::ZERO);
    }
    // Removing the shared power of two is exact and keeps each operand as
    // small as possible before the (possibly rounding) conversion; e.g. a
    // dyadic ratio such as k / 2^m then divides two small exact integers.
    let common_shift = numerator.trailing_zeros().min(denominator.trailing_zeros());
    let numerator = scalar_from_u128_rounded::<S>(numerator >> common_shift)?;
    let denominator = scalar_from_u128_rounded::<S>(denominator >> common_shift)?;
    let ratio = numerator / denominator;
    if ratio.is_finite() {
        Ok(ratio)
    } else {
        Err(arithmetic_overflow(
            "the reduced integer ratio is outside the finite scalar range",
        ))
    }
}

/// Add one bounded scalar term with cascaded (Neumaier) compensation; finish with
/// sum + correction.
///
/// TwoSum yields the exact rounding error of `sum + term` for every finite,
/// non-overflowing pair, so this equals the magnitude-ordered Neumaier update
/// bit for bit. Avoiding its data-dependent branch keeps hot accumulation loops
/// free of mispredictions and lets the compiler combine independent scalar lanes.
#[inline]
pub(crate) fn compensated_term<S: RealScalar>(sum: &mut S, correction: &mut S, term: S) {
    let next = *sum + term;
    let part = next - *sum;
    *correction += (*sum - (next - part)) + (term - part);
    *sum = next;
}

/// Return `exp(-+2*pi*i * numerator / denominator)` from an exactly reduced
/// integer exponent. Centre in integers before forming an f64 angle, including
/// for f32 outputs. Cardinal roots are exact; no root depends on another root.
/// A zero denominator is attributed to `argument`.
#[inline]
pub(crate) fn reduced_root<S: scioxide_core::RealScalar>(
    numerator: u128,
    denominator: u128,
    inverse: bool,
    argument: Argument,
) -> Result<Complex<S>, SciOxideError> {
    if denominator == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "FFT phase denominator must be non-zero",
        ));
    }
    let reduced = if numerator < denominator {
        numerator
    } else {
        numerator % denominator
    };
    let reflected = reduced > denominator / 2;
    let magnitude = if reflected {
        denominator - reduced
    } else {
        reduced
    };
    let (sine, cosine) = if magnitude == 0 {
        (0.0, 1.0)
    } else if denominator.is_multiple_of(2) && magnitude == denominator / 2 {
        (0.0, -1.0)
    } else if denominator.is_multiple_of(4) && magnitude == denominator / 4 {
        (1.0, 0.0)
    } else {
        // Every u128 is finite in f64. These conversions round directly once;
        // integer centring above avoids cancellation near a complete turn.
        let fraction = magnitude as f64 / denominator as f64;
        (core::f64::consts::TAU * fraction).sin_cos()
    };
    let sine = if inverse != reflected { sine } else { -sine };
    Ok(Complex::new(
        S::try_from_f64_lossy(cosine)?,
        S::try_from_f64_lossy(sine)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_roots<S: RealScalar>() {
        let argument = scioxide_core::argument!("n");
        for (k, re, im) in [(0, 1.0, 0.0), (1, 0.0, -1.0), (2, -1.0, 0.0), (3, 0.0, 1.0)] {
            let expected = Complex::new(
                S::try_from_f64_lossy(re).unwrap(),
                S::try_from_f64_lossy(im).unwrap(),
            );
            assert_eq!(reduced_root::<S>(k, 4, false, argument).unwrap(), expected);
            assert_eq!(
                reduced_root::<S>(k, 4, true, argument).unwrap(),
                expected.conj()
            );
            assert_eq!(
                reduced_root::<S>((1_u128 << 100) + k, 4, false, argument).unwrap(),
                expected
            );
        }
        for denominator in [3, 5, 7, 1009, 1019, 1_u128 << 60, u128::MAX] {
            let left = reduced_root::<S>(1, denominator, false, argument).unwrap();
            let right = reduced_root::<S>(denominator - 1, denominator, false, argument).unwrap();
            assert_eq!(left, right.conj());
            assert!(
                right.im > S::ZERO,
                "integer centring must preserve near-full turns"
            );
        }
        assert!(matches!(
            reduced_root::<S>(1, 0, false, argument).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn roots_reduce_exactly_and_preserve_cardinal_and_conjugate_symmetry() {
        check_roots::<f32>();
        check_roots::<f64>();
    }

    #[test]
    fn compensated_terms_retain_cancellation_residuals() {
        for terms in [[1e16_f64, 1.0, -1e16], [1.0, 1e16, -1e16]] {
            let (mut sum, mut correction) = (0.0, 0.0);
            for term in terms {
                compensated_term(&mut sum, &mut correction, term);
            }
            assert_eq!(sum + correction, 1.0);
        }
        let (mut sum, mut correction) = (0.0_f32, 0.0_f32);
        for term in [1e8, 1.0, -1e8] {
            compensated_term(&mut sum, &mut correction, term);
        }
        assert_eq!(sum + correction, 1.0);
    }

    fn matches_ordered_neumaier<S: RealScalar>(terms: &[S]) {
        let (mut sum, mut correction) = (S::ZERO, S::ZERO);
        let (mut ordered, mut ordered_correction) = (S::ZERO, S::ZERO);
        for &term in terms {
            compensated_term(&mut sum, &mut correction, term);
            let next = ordered + term;
            ordered_correction += if ordered.abs() >= term.abs() {
                (ordered - next) + term
            } else {
                (term - next) + ordered
            };
            ordered = next;
            for (actual, expected) in [(sum, ordered), (correction, ordered_correction)] {
                // Equal non-NaN values with equal signs have identical bits.
                assert!(
                    actual == expected && actual.is_sign_negative() == expected.is_sign_negative()
                );
            }
        }
    }

    #[test]
    fn branch_free_terms_equal_magnitude_ordered_neumaier_bits() {
        // Mixed signs and binades spanning 2^-60..2^60 exercise both orderings.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let wide: [f64; 4096] = core::array::from_fn(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let mantissa = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            mantissa * f64::from_bits((1023 - 60 + (state >> 3) % 120) << 52)
        });
        matches_ordered_neumaier(&wide);
        matches_ordered_neumaier(&wide.map(|x| x as f32));
        matches_ordered_neumaier(&[
            -0.0,
            0.0,
            f64::from_bits(1),
            -f64::MIN_POSITIVE,
            1.0,
            -1.0,
            1e-300,
        ]);
        matches_ordered_neumaier(&[-0.0_f32, f32::from_bits(1), 1e8, 1.0, -1e8, -1.0]);
    }

    #[test]
    fn rounded_integer_conversion_to_f64() {
        let v: f64 = scalar_from_usize_rounded(5).unwrap();
        assert!((v - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rounded_integer_conversion_accepts_f32_beyond_exact_integer_limit() {
        let v: f32 = scalar_from_usize_rounded((1_usize << 24) + 1).unwrap();
        assert_eq!(v, (1_u32 << 24) as f32);
        let reciprocal = scalar_ratio_u128::<f32>(
            1,
            ((1_u32 << 24) + 1).into(),
            scioxide_core::argument!("denominator"),
        )
        .unwrap();
        assert!(reciprocal > 0.0);
    }
}
