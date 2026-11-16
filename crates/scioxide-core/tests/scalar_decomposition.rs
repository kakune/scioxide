//! Bit-exact reconstruction across every IEEE exponent and class.
#![allow(clippy::panic)]
use scioxide_core::{FloatDecomposition, RealScalar};

macro_rules! decomposition {
    ($name:ident, $ty:ty, $bits:ty, $fraction:literal, $exponents:literal) => {
        #[test]
        fn $name() {
            let fraction_mask = ((1 as $bits) << $fraction) - 1;
            for negative in [false, true] {
                let sign = <$ty>::from_bits((negative as $bits) << (<$bits>::BITS - 1));
                for exponent in 0..=$exponents {
                    for fraction in [0, 1, fraction_mask / 2, fraction_mask - 1, fraction_mask] {
                        let bits = ((exponent as $bits) << $fraction) | fraction | sign.to_bits();
                        let value = <$ty>::from_bits(bits);
                        match value.decompose() {
                            FloatDecomposition::Finite {
                                significand,
                                exponent,
                                negative: decoded_sign,
                            } => {
                                assert!(value.is_finite());
                                assert_eq!(decoded_sign, negative);
                                assert!(significand < (1_u64 << <$ty>::MANTISSA_DIGITS));
                                let recomposed =
                                    RealScalar::scalbn(significand as $ty, i32::from(exponent))
                                        .copysign(sign);
                                assert_eq!(recomposed.to_bits(), bits);
                                if value == 0.0 {
                                    assert_eq!(significand, 0);
                                }
                            }
                            FloatDecomposition::Infinite {
                                negative: decoded_sign,
                            } => {
                                assert!(value.is_infinite());
                                assert_eq!(decoded_sign, negative);
                            }
                            FloatDecomposition::Nan => assert!(value.is_nan()),
                        }
                    }
                }
            }
        }
    };
}
decomposition!(binary32_classes_and_recomposition, f32, u32, 23, 255);
decomposition!(binary64_classes_and_recomposition, f64, u64, 52, 2047);
