//! Component-bit conversion and logarithmic representation regressions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, LogValue, NonFiniteOrigin, NumericalIssue, ScaledValue, SciOxideErrorView,
};

#[test]
fn projective_frames_reject_bit_loss_even_when_both_coordinates_survive() {
    let value =
        ScaledValue::from_component_parts(Complex::new(1.0_f64, 1.0 + f64::EPSILON), [1023, -50])
            .unwrap();
    assert!(matches!(
        value.try_projective().unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::LossOfSignificance,
            ..
        }
    ));
    for value in [
        Complex::new(-0.0_f64, f64::from_bits(1)),
        Complex::new(0.0, -0.0),
        Complex::new(-3.0, 4.0),
    ] {
        let decoded = ScaledValue::from_value(value).unwrap();
        let restored = decoded
            .try_projective()
            .unwrap()
            .into_scaled()
            .unwrap()
            .try_value()
            .unwrap();
        assert_eq!(restored.re.to_bits(), value.re.to_bits());
        assert_eq!(restored.im.to_bits(), value.im.to_bits());
    }
}

macro_rules! qualify {
    ($name:ident, $s:ty, $bits:ty) => {
        #[test]
        fn $name() {
            let values = [
                0.0 as $s,
                -0.0,
                <$s>::from_bits(1),
                -<$s>::from_bits(1),
                <$s>::MIN_POSITIVE,
                -<$s>::MIN_POSITIVE,
                1.0,
                -1.0,
                <$s>::MAX,
                -<$s>::MAX,
                0.75 * <$s>::MAX,
            ];
            for re in values {
                let value = ScaledValue::from_value(re).unwrap();
                assert_eq!(value.try_value().unwrap().to_bits(), re.to_bits());
                for im in values {
                    let original = Complex::new(re, im);
                    let value = ScaledValue::from_value(original).unwrap();
                    let projected = value.try_value().unwrap();
                    assert_eq!(
                        (projected.re.to_bits(), projected.im.to_bits()),
                        (re.to_bits(), im.to_bits())
                    );
                    let log = LogValue::from_value(original).unwrap();
                    assert_eq!(log.scaled_phase().real().is_zero(), re == 0.0);
                    assert_eq!(log.scaled_phase().imaginary().is_zero(), im == 0.0);
                    if re == 0.0 && im == 0.0 {
                        let projected = log.try_value().unwrap();
                        assert_eq!(
                            (projected.re.to_bits(), projected.im.to_bits()),
                            (re.to_bits(), im.to_bits())
                        );
                    }
                }
            }
            let mixed =
                ScaledValue::from_value(Complex::new(<$s>::MAX, <$s>::from_bits(1))).unwrap();
            assert!(mixed.try_projective().is_err());
            let value =
                ScaledValue::from_parts(Complex::new(1.0 as $s, <$s>::MIN_POSITIVE), -64).unwrap();
            assert!(value.try_value().is_err());
            assert_eq!(value.round().unwrap().im, 0.0);
            assert!(value.round().unwrap().re > 0.0);
            for bad in [<$s>::NAN, <$s>::INFINITY, <$s>::NEG_INFINITY] {
                assert!(matches!(
                    ScaledValue::from_value(bad).unwrap_err().view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
                assert!(matches!(
                    LogValue::from_value(Complex::new(1.0, bad))
                        .unwrap_err()
                        .view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
            }
            // Exercise both signs, normals in every exponent bin, and the
            // subnormal endpoints, without deriving expectations via normalization.
            let mantissa_bits = <$s>::MANTISSA_DIGITS - 1;
            let exponent_count = (1_u64 << (<$bits>::BITS - mantissa_bits - 1)) - 1;
            for exponent in 0..exponent_count {
                for fraction in [0, 1, (1_u64 << mantissa_bits) - 1] {
                    let bits = (exponent << mantissa_bits) | fraction;
                    for sign in [0, 1_u64 << (<$bits>::BITS - 1)] {
                        let original = <$s>::from_bits((bits | sign) as $bits);
                        let value = ScaledValue::from_value(original).unwrap();
                        assert_eq!(value.try_value().unwrap().to_bits(), original.to_bits());
                    }
                }
            }
        }
    };
}
qualify!(binary32_components, f32, u32);
qualify!(binary64_components, f64, u64);

#[test]
fn overflowing_modulus_does_not_overflow_representable_components() {
    let input = Complex::new(0.75 * f64::MAX, 0.75 * f64::MAX);
    let encoded = LogValue::from_value(input).unwrap();
    let actual = encoded.try_value().unwrap();
    assert!((actual.re / input.re - 1.0).abs() < 5e-13);
    assert!((actual.im / input.im - 1.0).abs() < 5e-13);
    let phase = Complex::new(
        core::f64::consts::FRAC_1_SQRT_2,
        core::f64::consts::FRAC_1_SQRT_2,
    );
    let value = LogValue::from_log(phase, f64::MAX.ln() + 0.2)
        .unwrap()
        .try_value()
        .unwrap();
    assert!(value.re > 0.85 * f64::MAX && value.re < 0.87 * f64::MAX);
    assert_eq!(value.re, value.im);
}

#[test]
fn phase_underflow_is_classified_at_the_ieee_boundary() {
    use scioxide_core::Scaled;
    let tiny = Scaled::<f64>::ONE.shifted(i64::MIN).unwrap();
    let phase = ScaledValue::<Complex<f64>>::from_components(Scaled::ONE, tiny);
    let value = LogValue::from_scaled_log(phase, -1.0).unwrap();
    let rounded = value.round().unwrap();
    assert!(rounded.re > 0.0);
    assert_eq!(rounded.im.to_bits(), 0);
    assert!(matches!(
        value.try_value().unwrap_err().view(),
        SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Underflow,
            ..
        }
    ));
}
