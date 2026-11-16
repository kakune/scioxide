//! Public scalar tests also run against the actual no-default-features library.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, RealScalar};

mod atan2_boundaries {
    include!("support/atan2_cases.rs");
}

#[test]
fn independent_exact_references() {
    for line in include_str!("data/ieee.txt").lines().skip(1) {
        let columns: Vec<_> = line.split_whitespace().collect();
        let bits: Vec<_> = columns[2..]
            .iter()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        let actual = if columns[0] == "32" {
            let a = f32::from_bits(bits[0] as u32);
            let result = match columns[1] {
                "sqrt" => RealScalar::sqrt(a),
                "round" => RealScalar::round(a),
                "floor" => RealScalar::floor(a),
                "ceil" => RealScalar::ceil(a),
                "trunc" => RealScalar::trunc(a),
                "fract" => RealScalar::fract(a),
                _ => RealScalar::mul_add(
                    a,
                    f32::from_bits(bits[1] as u32),
                    f32::from_bits(bits[2] as u32),
                ),
            };
            u64::from(result.to_bits())
        } else {
            let a = f64::from_bits(bits[0]);
            let result = match columns[1] {
                "sqrt" => RealScalar::sqrt(a),
                "round" => RealScalar::round(a),
                "floor" => RealScalar::floor(a),
                "ceil" => RealScalar::ceil(a),
                "trunc" => RealScalar::trunc(a),
                "fract" => RealScalar::fract(a),
                _ => RealScalar::mul_add(a, f64::from_bits(bits[1]), f64::from_bits(bits[2])),
            };
            result.to_bits()
        };
        assert_eq!(actual, bits[3], "{line}");
    }
}

macro_rules! scalar_cases {
    ($name:ident, $ty:ty, $bits:ty) => {
        #[test]
        fn $name() {
            let values: &[$ty] = &[
                0.0,
                -0.0,
                <$ty>::from_bits(1),
                -<$ty>::from_bits(1),
                <$ty>::MIN_POSITIVE,
                <$ty>::MAX,
                -<$ty>::MAX,
                0.5,
                -0.5,
                1.5,
                -1.5,
                2.5,
                -2.5,
                <$ty>::INFINITY,
                <$ty>::NEG_INFINITY,
                <$ty>::NAN,
            ];
            let equal = |a: $ty, b: $ty| {
                if b.is_nan() {
                    assert!(a.is_nan());
                } else {
                    assert_eq!(a.to_bits(), b.to_bits(), "{a:?} != {b:?}");
                }
            };
            for &a in values {
                equal(RealScalar::sqrt(a), a.sqrt());
                equal(RealScalar::round(a), a.round());
                equal(RealScalar::floor(a), a.floor());
                equal(RealScalar::ceil(a), a.ceil());
                equal(RealScalar::trunc(a), a.trunc());
                equal(RealScalar::fract(a), a.fract());
                for &b in values {
                    for &c in values {
                        equal(RealScalar::mul_add(a, b, c), a.mul_add(b, c));
                    }
                }
            }
            for exponent in 0..<$ty>::MANTISSA_DIGITS + 2 {
                let integer = (2.0 as $ty).powi(exponent as i32);
                for a in [integer, integer + 1.0, -integer, -integer - 1.0] {
                    equal(RealScalar::round(a), a.round());
                }
            }
            for midpoint in [0.5 as $ty, 1.5, 2.5, -0.5, -1.5, -2.5] {
                for bits in [
                    midpoint.to_bits() - 1,
                    midpoint.to_bits(),
                    midpoint.to_bits() + 1,
                ] {
                    let a = <$ty>::from_bits(bits);
                    equal(RealScalar::round(a), a.round());
                }
            }
            // Supplement exact fixtures with deterministic host-leaf checks.
            let mut state = 0x4752_4144_4941_0206_u64;
            for _ in 0..20_000 {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    <$ty>::from_bits(state as $bits)
                };
                let (a, b, c) = (next(), next(), next());
                equal(RealScalar::sqrt(a), a.sqrt());
                equal(RealScalar::mul_add(a, b, c), a.mul_add(b, c));
                equal(RealScalar::round(a), a.round());
            }
            for zero in [0.0 as $ty, -0.0] {
                let z = Complex::new(-4.0 as $ty, zero);
                equal(z.arg(), <$ty as RealScalar>::PI.copysign(zero));
                equal(z.sqrt().im, (2.0 as $ty).copysign(zero));
                let tiny = <$ty>::from_bits(1);
                equal(Complex::new(tiny, zero).sqrt().re, tiny.sqrt());
                equal(
                    Complex::new(-tiny, zero).sqrt().im,
                    tiny.sqrt().copysign(zero),
                );
            }
        }
    };
}

scalar_cases!(binary32, f32, u32);
scalar_cases!(binary64, f64, u64);
