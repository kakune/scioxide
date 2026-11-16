//! Independent exponent reasoning and IEEE boundary regressions.
#![allow(clippy::unwrap_used)]
use core::cmp::Ordering;
use core::ops::Neg;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::{Complex, NonFiniteOrigin, NumericalIssue, SciOxideError, SciOxideErrorView};
use scioxide_core::{Scaled, ScaledAccumulator, ScaledValue};

fn exponent_failure(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::ExponentOutOfRange,
            ..
        }
    ));
}

macro_rules! qualify {
    ($name:ident, $s:ty, $bits:ty) => {
        #[test]
        fn $name() {
            let one = Scaled::from_value(1.0 as $s).unwrap();
            let tiny = one.shifted(i64::MIN + 100).unwrap();
            let huge = one.shifted(i64::MAX).unwrap();
            assert_eq!(tiny.add(huge).unwrap(), huge);
            assert_eq!(huge.sub(tiny).unwrap(), huge);
            assert_eq!(huge.sub(huge).unwrap().round().unwrap().to_bits(), 0);
            assert_eq!(
                tiny.neg().round().unwrap().to_bits(),
                (-0.0 as $s).to_bits()
            );
            assert!(huge.round().is_err());
            assert!(tiny.try_value().is_err());

            let exponents = [
                i64::MIN,
                i64::MIN + 1,
                -4000,
                -1,
                0,
                1,
                4000,
                i64::MAX - 1,
                i64::MAX,
            ];
            for a in exponents {
                let left = one.shifted(a).unwrap();
                assert_eq!(left.significand(), 1.0);
                assert_eq!(left.exponent(), a);
                let root = left.sqrt().unwrap();
                assert_eq!(root.exponent(), a.div_euclid(2));
                assert!(root.significand() >= 1.0 && root.significand() < 2.0);
                for b in exponents {
                    let right = one.shifted(b).unwrap();
                    assert_eq!(left.abs_cmp(right), a.cmp(&b));
                    for (actual, exponent) in [
                        (left.mul(right), i128::from(a) + i128::from(b)),
                        (left.div(right), i128::from(a) - i128::from(b)),
                        (left.shifted(b), i128::from(a) + i128::from(b)),
                    ] {
                        match i64::try_from(exponent) {
                            Ok(expected) => {
                                let actual = actual.unwrap();
                                assert_eq!(actual.exponent(), expected);
                                assert_eq!(actual.significand(), 1.0);
                            }
                            Err(_) => exponent_failure(actual.unwrap_err()),
                        }
                    }
                }
            }
            // Normalization rescues an intermediate exponent outside i64.
            let a = Scaled::from_parts(1.5 as $s, i64::MIN).unwrap();
            let b = Scaled::from_parts(1.5 as $s, -1).unwrap();
            let product = a.mul(b).unwrap();
            assert_eq!(
                (product.significand(), product.exponent()),
                (1.125, i64::MIN)
            );
            let quotient = huge.div(b).unwrap();
            assert_eq!(quotient.exponent(), i64::MAX);
            assert_eq!(quotient.significand(), (4.0 as $s) / 3.0);
            exponent_failure(huge.add(huge).unwrap_err());
            assert!(
                huge.difference_of_products(huge, huge, huge)
                    .unwrap()
                    .is_zero()
            );

            let complex = ScaledValue::<Complex<$s>>::from_components(huge, huge);
            assert_eq!(
                complex.div(complex).unwrap().round().unwrap(),
                Complex::new(1.0, 0.0)
            );
            let minimum = one.shifted(i64::MIN).unwrap();
            let (factored, exponent) = factor_common_scale([minimum, minimum]).unwrap();
            assert_eq!(exponent, i64::MIN);
            assert_eq!(factored, [one, one]);
            exponent_failure(factor_common_scale([minimum, huge]).unwrap_err());

            let positive_zero = Scaled::from_value(0.0 as $s).unwrap();
            let negative_zero = Scaled::from_value(-0.0 as $s).unwrap();
            assert_eq!(positive_zero.cmp_value(negative_zero), Ordering::Equal);
            assert_eq!(
                negative_zero.sqrt().unwrap().round().unwrap().to_bits(),
                (-0.0 as $s).to_bits()
            );
            assert_eq!(
                negative_zero
                    .add(positive_zero)
                    .unwrap()
                    .round()
                    .unwrap()
                    .to_bits(),
                0
            );
            for nonfinite in [<$s>::NAN, <$s>::INFINITY, <$s>::NEG_INFINITY] {
                assert!(matches!(
                    Scaled::from_value(nonfinite).unwrap_err().view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
            }
            let mut accumulator = ScaledAccumulator::from_scaled(huge);
            let before = accumulator.parts();
            exponent_failure(accumulator.add_scaled(huge).unwrap_err());
            assert_eq!(accumulator.parts(), before);

            // Deterministic bit sampling includes both signs and arbitrary
            // significands; the reference is the source encoding, not decode().
            let mut bits = 0x475241444941_u64;
            for _ in 0..8192 {
                bits ^= bits << 13;
                bits ^= bits >> 7;
                bits ^= bits << 17;
                let value = <$s>::from_bits(bits as $bits);
                if value.is_finite() {
                    let scaled = Scaled::from_value(value).unwrap();
                    assert_eq!(scaled.round().unwrap().to_bits(), value.to_bits());
                    assert!(
                        scaled.is_zero()
                            || (scaled.significand().abs() >= 1.0
                                && scaled.significand().abs() < 2.0)
                    );
                }
            }
            for value in [0.0, -0.0, <$s>::from_bits(1), <$s>::MIN_POSITIVE, <$s>::MAX] {
                assert_eq!(
                    Scaled::from_value(value)
                        .unwrap()
                        .round()
                        .unwrap()
                        .to_bits(),
                    value.to_bits()
                );
            }
        }
    };
}
qualify!(binary32, f32, u32);
qualify!(binary64, f64, u64);

#[test]
fn destination_width_rounding_uses_the_final_subnormal_grid() {
    // Casting the significand before scaling would round the first input to
    // 1.5, then round that halfway case to two quanta instead of one.
    for significand in [1.5_f64.next_down(), 1.5, 1.5_f64.next_up()] {
        for sign in [-1.0, 1.0] {
            let source = significand * sign;
            let actual = Scaled::from_parts(source, -149).unwrap();
            let expected = (source * 2.0_f64.powi(-149)) as f32;
            assert_eq!(
                actual.round_to::<f32>().unwrap().to_bits(),
                expected.to_bits()
            );
        }
    }
    let mut state = 0x873_882_885_u64;
    for _ in 0..8192 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if value.is_finite() {
            let expected = value as f32;
            let actual = Scaled::from_value(value).unwrap().round_to::<f32>();
            if expected.is_finite() {
                assert_eq!(actual.unwrap().to_bits(), expected.to_bits());
            } else {
                assert!(matches!(
                    actual.unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ));
            }
        }
        let value = f32::from_bits(state as u32);
        if value.is_finite() {
            assert_eq!(
                Scaled::from_value(value)
                    .unwrap()
                    .round_to::<f64>()
                    .unwrap()
                    .to_bits(),
                f64::from(value).to_bits()
            );
        }
    }
}

#[test]
fn compound_accumulation_checks_exponents_after_cancellation() {
    let huge = Scaled::<f64>::ONE.shifted(i64::MAX).unwrap();
    let mut sum = ScaledAccumulator::from_parts(huge, huge.neg());
    sum.add_scaled(huge).unwrap();
    assert_eq!(sum.total().unwrap(), huge);
    let zero = ScaledAccumulator::from_parts(huge, huge.neg());
    assert!(zero.shifted(i64::MAX).unwrap().total().unwrap().is_zero());
    assert!(zero.multiplied(huge).unwrap().total().unwrap().is_zero());
    let mut sum = ScaledAccumulator::from_parts(huge.neg(), huge.neg());
    sum.add_product(huge, Scaled::from_value(2.0).unwrap())
        .unwrap();
    assert!(sum.total().unwrap().is_zero());
}

#[test]
fn remote_product_resolves_midpoint_ties_without_disappearing() {
    // The exact leading product is 1 + 3*2^-27 + 2^-53, a midpoint.
    // An arbitrarily remote positive term breaks the tie toward the upper value.
    let a = Scaled::from_value(1.0_f64 + 2.0_f64.powi(-27)).unwrap();
    let b = Scaled::from_value(1.0_f64 + 2.0_f64.powi(-26)).unwrap();
    let tiny = Scaled::ONE.shifted(i64::MIN).unwrap();
    let lower = 1.0_f64 + 3.0 * 2.0_f64.powi(-27);
    let above = a
        .difference_of_products(b, tiny.neg(), Scaled::ONE)
        .unwrap();
    let below = a.difference_of_products(b, tiny, Scaled::ONE).unwrap();
    assert_eq!(above.round().unwrap().to_bits(), lower.to_bits() + 1);
    assert_eq!(below.round().unwrap().to_bits(), lower.to_bits());
    let zero = Scaled::<f64>::ZERO;
    assert_eq!(
        zero.difference_of_products(zero, zero, zero)
            .unwrap()
            .round()
            .unwrap()
            .to_bits(),
        0
    );
}

#[test]
fn domains_zero_signs_and_tolerance_compounds_are_explicit() {
    use scioxide_core::backend::scaled::scaled_tolerance_threshold;
    let one = Scaled::<f64>::ONE;
    assert!(matches!(
        one.div(Scaled::ZERO).unwrap_err().view(),
        SciOxideErrorView::DomainError { .. }
    ));
    assert!(matches!(
        one.neg().sqrt().unwrap_err().view(),
        SciOxideErrorView::DomainError { .. }
    ));
    let zero = ScaledValue::from_value(Complex::new(-0.0_f64, -0.0)).unwrap();
    let divided = zero
        .div(
            ScaledValue::from_value(scioxide_core::Complex::new(
                1.0,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))
            .unwrap(),
        )
        .unwrap()
        .round()
        .unwrap();
    assert_eq!(divided.re.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(divided.im.to_bits(), (-0.0_f64).to_bits());
    let tiny = one.shifted(i64::MIN).unwrap();
    let tolerance = scioxide_core::Tolerance::new(1.0, f64::EPSILON).unwrap();
    assert_eq!(scaled_tolerance_threshold(&tolerance, tiny).unwrap(), one);
}
