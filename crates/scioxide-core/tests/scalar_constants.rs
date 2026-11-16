//! Independently derived destination bits, also exercised without default features.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{AlgebraicScalar, Complex32, Complex64, RealScalar};

macro_rules! constants {
    ($test:ident, $ty:ty, $width:literal) => {
        #[test]
        fn $test() {
            for line in include_str!("data/scalar_constants.txt")
                .lines()
                .filter(|l| !l.starts_with('#'))
            {
                let row: Vec<_> = line.split_whitespace().collect();
                if row[0] != $width {
                    continue;
                }
                let expected = u64::from_str_radix(row[2], 16).unwrap();
                let actual = match row[1] {
                    "ZERO" => <$ty as scioxide_core::AlgebraicScalar>::ZERO,
                    "ONE" => <$ty as scioxide_core::AlgebraicScalar>::ONE,
                    "TWO" => <$ty as RealScalar>::TWO,
                    "TEN" => <$ty as RealScalar>::TEN,
                    "HALF" => <$ty as RealScalar>::HALF,
                    "PI" => <$ty as RealScalar>::PI,
                    "TAU" => <$ty as RealScalar>::TAU,
                    "EPSILON" => <$ty as RealScalar>::EPSILON,
                    "SQRT_EPSILON" => <$ty as RealScalar>::SQRT_EPSILON,
                    "CBRT_EPSILON" => <$ty as RealScalar>::CBRT_EPSILON,
                    "MIN" => <$ty as RealScalar>::MIN,
                    "MAX" => <$ty as RealScalar>::MAX,
                    "MIN_POSITIVE" => <$ty as RealScalar>::MIN_POSITIVE,
                    "MIN_SUBNORMAL" => <$ty as RealScalar>::MIN_SUBNORMAL,
                    "LN_2" => <$ty as RealScalar>::LN_2,
                    "LN_10" => <$ty as RealScalar>::LN_10,
                    "LOG2_E" => <$ty as RealScalar>::LOG2_E,
                    "LOG10_E" => <$ty as RealScalar>::LOG10_E,
                    "NAN" => <$ty as RealScalar>::NAN,
                    "INFINITY" => <$ty as RealScalar>::INFINITY,
                    "NEG_INFINITY" => <$ty as RealScalar>::NEG_INFINITY,
                    "NEG_ZERO" => <$ty as RealScalar>::NEG_ZERO,
                    "MANTISSA_DIGITS" => {
                        assert_eq!(u64::from(<$ty as RealScalar>::MANTISSA_DIGITS), expected);
                        continue;
                    }
                    name => panic!("unknown constant {name}"),
                };
                assert_eq!(actual.to_bits() as u64, expected, "{line}");
            }
        }
    };
}
constants!(binary32_constants, f32, "32");
constants!(binary64_constants, f64, "64");

#[test]
fn identities_have_one_owner() {
    fn two<S: AlgebraicScalar>() -> S {
        S::ONE + S::ONE
    }
    assert_eq!(two::<Complex32>(), Complex32::new(2.0, 0.0));
    assert_eq!(two::<Complex64>(), Complex64::new(2.0, 0.0));
    assert!(Complex64::new(-0.0, 0.0).is_zero());
    assert!(!Complex64::new(0.0, 1.0).is_zero());
    assert!((-0.0_f64).is_zero());
    assert!(!f64::NAN.is_zero());
    assert_eq!(Complex32::ZERO.re.to_bits(), 0);
    assert_eq!(Complex32::ONE.im.to_bits(), 0);
}
