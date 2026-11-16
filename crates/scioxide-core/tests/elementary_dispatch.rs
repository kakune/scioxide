//! Guard behavior is independent of the selected elementary implementation.
#![allow(clippy::float_cmp)]
use scioxide_core::RealScalar;

macro_rules! special_values {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for zero in [0.0 as $ty, -0.0] {
                for value in [<$ty>::NAN, <$ty>::INFINITY, <$ty>::NEG_INFINITY, -1.0, zero] {
                    assert_eq!(
                        RealScalar::powf(value, zero).to_bits(),
                        (1.0 as $ty).to_bits()
                    );
                }
                assert_eq!(RealScalar::sqrt(zero).to_bits(), zero.to_bits());
                assert_eq!(RealScalar::cbrt(zero).to_bits(), zero.to_bits());
                assert_eq!(RealScalar::sin(zero).to_bits(), zero.to_bits());
                assert_eq!(RealScalar::asinh(zero).to_bits(), zero.to_bits());
                assert_eq!(RealScalar::atanh(zero).to_bits(), zero.to_bits());
                assert_eq!(RealScalar::clamp(zero, -1.0, 1.0).to_bits(), zero.to_bits());
            }
            for exponent in [<$ty>::NAN, <$ty>::INFINITY, <$ty>::NEG_INFINITY] {
                assert_eq!(RealScalar::powf(1.0 as $ty, exponent), 1.0);
            }
            for infinite in [<$ty>::INFINITY, <$ty>::NEG_INFINITY] {
                for other in [
                    <$ty>::NAN,
                    <$ty>::INFINITY,
                    <$ty>::NEG_INFINITY,
                    -0.0,
                    0.0,
                    <$ty>::MAX,
                ] {
                    assert_eq!(RealScalar::hypot(infinite, other), <$ty>::INFINITY);
                    assert_eq!(RealScalar::hypot(other, infinite), <$ty>::INFINITY);
                }
            }
            for (minimum, maximum) in [(1.0, -1.0), (<$ty>::NAN, 1.0), (-1.0, <$ty>::NAN)] {
                assert!(RealScalar::clamp(0.0 as $ty, minimum, maximum).is_nan());
            }
            assert!(RealScalar::clamp(<$ty>::NAN, -1.0, 1.0).is_nan());
            assert_eq!(RealScalar::clamp(<$ty>::INFINITY, -1.0, 1.0), 1.0);
            let tiny = <$ty>::from_bits(1);
            assert_eq!(RealScalar::exp_m1(tiny).to_bits(), tiny.to_bits());
            assert_eq!(RealScalar::ln_1p(tiny).to_bits(), tiny.to_bits());
            assert_eq!(RealScalar::sin(tiny).to_bits(), tiny.to_bits());
        }
    };
}
special_values!(binary32_guards, f32);
special_values!(binary64_guards, f64);
