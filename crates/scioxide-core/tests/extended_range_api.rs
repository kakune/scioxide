//! Shared extended-range construction vocabulary and typed failures.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Complex, LogValue, NumericalIssue, Scaled, ScaledAccumulator, ScaledValue, SciOxideError,
    SciOxideErrorKind, SciOxideErrorView,
};

fn failure<T: core::fmt::Debug>(
    result: Result<T, SciOxideError>,
    kind: SciOxideErrorKind,
    issue: Option<NumericalIssue>,
) {
    let error = result.unwrap_err();
    assert_eq!(error.kind(), kind);
    if let Some(issue) = issue {
        assert!(
            matches!(error.view(), SciOxideErrorView::NumericalInstability { issue: actual, .. } if actual == issue)
        );
    }
}

macro_rules! qualify {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            use NumericalIssue::{ExponentOutOfRange, LossOfSignificance, Overflow, Underflow};
            use SciOxideErrorKind::{DomainError, InvalidInput, NonFinite, NumericalInstability};

            let huge = Scaled::<$s>::ONE.shifted(i64::MAX).unwrap();
            let tiny = Scaled::<$s>::ONE.shifted(i64::MIN).unwrap();
            let negative_zero = Scaled::from_value(-0.0 as $s).unwrap();
            let complex = ScaledValue::<Complex<$s>>::from_components(huge, tiny);
            let real = ScaledValue::<$s>::from_components(negative_zero, huge);
            assert_eq!(real.real(), negative_zero);
            assert_eq!(real.imaginary(), Scaled::ZERO);
            assert!(!real.imaginary().is_sign_negative());
            assert_eq!(real.round().unwrap().to_bits(), (-0.0 as $s).to_bits());
            assert_eq!(complex.real(), huge);
            assert_eq!(complex.imaginary(), tiny);
            assert_eq!(complex.component_exponents(), [i64::MAX, i64::MIN]);
            assert_eq!(ScaledAccumulator::<$s>::ZERO.round().unwrap(), 0.0);
            assert_eq!(LogValue::<$s, $s>::ZERO.log_magnitude(), None);

            let exact = ScaledValue::from_value(Complex::new(-3.0 as $s, -0.0)).unwrap();

            let frame = exact.try_projective().unwrap();
            let restored = frame.into_scaled().unwrap().try_value().unwrap();
            assert_eq!(restored.re.to_bits(), (-3.0 as $s).to_bits());
            assert_eq!(restored.im.to_bits(), (-0.0 as $s).to_bits());
            assert_eq!(
                Scaled::<$s>::from_log(-1.0, 0.0).unwrap().round().unwrap(),
                -1.0
            );
            assert_eq!(
                ScaledValue::<$s>::from_log(-1.0, 0.0)
                    .unwrap()
                    .round()
                    .unwrap(),
                -1.0
            );
            assert_eq!(
                LogValue::<$s, $s>::from_log(-1.0, 0.0)
                    .unwrap()
                    .round()
                    .unwrap(),
                -1.0
            );

            for bad in [<$s>::NAN, <$s>::INFINITY, <$s>::NEG_INFINITY] {
                failure(Scaled::from_value(bad), NonFinite, None);
                failure(Scaled::from_parts(bad, 0), NonFinite, None);
                failure(Scaled::from_log(1.0, bad), NonFinite, None);
                failure(
                    ScaledValue::from_value(Complex::new(1.0, bad)),
                    NonFinite,
                    None,
                );
                failure(ScaledValue::from_parts(bad, 0), NonFinite, None);
                failure(
                    ScaledValue::from_component_parts(Complex::new(1.0, bad), [0, 0]),
                    NonFinite,
                    None,
                );
                failure(ScaledValue::from_log(1.0, bad), NonFinite, None);
                failure(ScaledAccumulator::from_value(bad), NonFinite, None);
                failure(
                    LogValue::from_value(Complex::new(bad, 1.0)),
                    NonFinite,
                    None,
                );
                failure(LogValue::from_log(bad, 0.0), NonFinite, None);
                failure(
                    LogValue::from_scaled_log(ScaledValue::<$s>::ONE, bad),
                    NonFinite,
                    None,
                );
                failure(real.scale_absolute_error(bad), NonFinite, None);
            }
            failure(Scaled::<$s>::from_log(0.0, 0.0), InvalidInput, None);
            failure(ScaledValue::<$s>::from_log(0.0, 0.0), InvalidInput, None);
            failure(LogValue::<$s, $s>::from_log(0.5, 0.0), InvalidInput, None);
            failure(
                LogValue::from_scaled_log(ScaledValue::<Complex<$s>>::ZERO, 0.0),
                InvalidInput,
                None,
            );
            failure(real.scale_absolute_error(-1.0), InvalidInput, None);

            for result in [
                Scaled::<$s>::from_parts(2.0, i64::MAX),
                Scaled::from_log(1.0, <$s>::MAX),
                huge.shifted(1),
                huge.add(huge),
                huge.sub(-huge),
                huge.mul(huge),
                huge.square(),
                huge.div(tiny),
                huge.difference_of_products(huge, tiny, tiny),
            ] {
                failure(result, NumericalInstability, Some(ExponentOutOfRange));
            }
            failure(huge.div(Scaled::ZERO), DomainError, None);
            failure((-Scaled::<$s>::ONE).sqrt(), DomainError, None);
            failure(Scaled::<$s>::ZERO.ln_abs(), DomainError, None);
            failure(huge.round(), NumericalInstability, Some(Overflow));
            failure(huge.try_value(), NumericalInstability, Some(Overflow));
            failure(tiny.try_value(), NumericalInstability, Some(Underflow));
            failure(huge.round_to::<f32>(), NumericalInstability, Some(Overflow));
            failure(
                tiny.try_value_to::<f64>(),
                NumericalInstability,
                Some(Underflow),
            );
            failure(
                complex.shifted(1),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            failure(
                complex.mul_real(huge),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            failure(complex.try_value(), NumericalInstability, Some(Overflow));
            failure(
                complex.try_projective(),
                NumericalInstability,
                Some(Underflow),
            );

            let disappearing = ScaledValue::<$s>::from_components(-tiny, Scaled::ZERO);
            assert_eq!(
                disappearing.round().unwrap().to_bits(),
                (-0.0 as $s).to_bits()
            );
            failure(
                disappearing.try_value(),
                NumericalInstability,
                Some(Underflow),
            );
            failure(
                disappearing.scale_absolute_error(1.0),
                NumericalInstability,
                Some(Overflow),
            );
            let bits_lost = ScaledValue::from_component_parts(
                Complex::new(1.0 as $s, 1.0 + <$s>::EPSILON),
                [
                    i64::from(<$s>::MAX_EXP - 1),
                    3 - i64::from(<$s>::MANTISSA_DIGITS),
                ],
            )
            .unwrap();
            failure(
                bits_lost.try_projective(),
                NumericalInstability,
                Some(LossOfSignificance),
            );

            let mut sum = ScaledAccumulator::from_scaled(huge);
            let before = sum.parts();
            failure(sum.add_value(<$s>::NAN), NonFinite, None);
            assert_eq!(sum.parts(), before);
            failure(
                sum.add_scaled(huge),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            assert_eq!(sum.parts(), before);
            failure(
                sum.add_product(huge, huge),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            assert_eq!(sum.parts(), before);
            failure(
                sum.shifted(1),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            failure(
                sum.multiplied(huge),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );
            failure(sum.round(), NumericalInstability, Some(Overflow));
            failure(
                ScaledAccumulator::from_parts(huge, huge).total(),
                NumericalInstability,
                Some(ExponentOutOfRange),
            );

            let overflow = LogValue::<$s, $s>::from_log(1.0, <$s>::MAX).unwrap();
            failure(overflow.round(), NumericalInstability, Some(Overflow));
            failure(overflow.try_value(), NumericalInstability, Some(Overflow));
            let underflow = LogValue::<$s, $s>::from_log(-1.0, -<$s>::MAX).unwrap();
            assert_eq!(underflow.round().unwrap().to_bits(), (-0.0 as $s).to_bits());
            failure(underflow.try_value(), NumericalInstability, Some(Underflow));
        }
    };
}

qualify!(binary32, f32);
qualify!(binary64, f64);
