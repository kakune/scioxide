//! Bit-level checked arithmetic contracts, shared by native and portable builds.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{
    Complex, ProductSum, RangeAccumulator, ScaledNorm, SciOxideErrorKind, checked_affine,
    checked_difference, checked_difference_of_products, checked_float_product,
    checked_float_product_factors, checked_float_quotient, checked_midpoint, checked_mul_add,
    checked_next_after, checked_product_quotient, checked_quadratic_roots, checked_scalbn,
    checked_scaled_norm, checked_sum, checked_sum_of_products,
};

macro_rules! short_reductions {
    ($name:ident, $s:ty, $bits:ty) => {
        #[test]
        fn $name() {
            let tiny = <$s>::from_bits(1);
            for (weights, values, expected) in [
                (
                    [1.0, 1.0, -1.0],
                    [<$s>::MAX, <$s>::MAX, <$s>::MAX],
                    <$s>::MAX,
                ),
                ([1.0, 1.0, -1.0], [1.0, tiny, 1.0], tiny),
                ([0.75, 0.75, 0.75], [tiny, tiny, tiny], 2.0 * tiny),
                ([0.5, 0.0, 0.0], [-tiny, 0.0, 0.0], -0.0),
            ] {
                assert_eq!(
                    checked_sum_of_products(&weights, &values)
                        .unwrap()
                        .to_bits(),
                    expected.to_bits()
                );
            }
            let mut state = 0x1234_5678_9abc_def0_u64;
            for length in 0..=4 {
                for _ in 0..2048 {
                    let mut weights = [0.0 as $s; 4];
                    let mut values = weights;
                    let mut reference = RangeAccumulator::<$s>::new();
                    for i in 0..length {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        weights[i] = <$s>::from_bits(state as $bits);
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        values[i] = <$s>::from_bits(state as $bits);
                        if !weights[i].is_finite() {
                            weights[i] = 0.5;
                        }
                        if !values[i].is_finite() {
                            values[i] = -0.5;
                        }
                        reference.add_product(weights[i], values[i]).unwrap();
                    }
                    match (
                        checked_sum_of_products(&weights[..length], &values[..length]),
                        reference.finish(),
                    ) {
                        (Ok(actual), Ok(expected)) => {
                            assert_eq!(actual.to_bits(), expected.to_bits())
                        }
                        (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
                        pair => panic!("short reduction differs: {pair:?}"),
                    }
                }
            }
            assert!(checked_sum_of_products(&[0.0 as $s], &[<$s>::NAN]).is_err());
        }
    };
}
short_reductions!(short_reductions_f32, f32, u32);
short_reductions!(short_reductions_f64, f64, u64);

macro_rules! finite_updates {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let tiny = <$s>::from_bits(1);
            for left in [0.0 as $s, -0.0, tiny, -tiny, 1.5, -1.5] {
                for right in [0.5 as $s, -0.5, 2.0, -2.0] {
                    let mut pair = RangeAccumulator::new();
                    pair.add_product(left, right).unwrap();
                    let expected = if left == 0.0 { 0.0 } else { left * right };
                    assert_eq!(pair.finish().unwrap().to_bits(), expected.to_bits());
                    let mut identity = RangeAccumulator::new();
                    identity.add_triple_product(1.0, left, right).unwrap();
                    assert_eq!(identity, pair);
                    pair.subtract_product(left, right).unwrap();
                    identity.subtract_triple_product(1.0, left, right).unwrap();
                    assert_eq!(identity, pair);
                    assert_eq!(pair.finish().unwrap().to_bits(), 0);
                }
            }
            for first in [-2.0 as $s, 2.0] {
                for second in [-1.5 as $s, 1.5] {
                    for third in [-4.0 as $s, 4.0] {
                        let mut sum = RangeAccumulator::new();
                        sum.add_triple_product(first, second, third).unwrap();
                        assert_eq!(
                            sum.finish().unwrap().to_bits(),
                            (first * second * third).to_bits()
                        );
                        sum.subtract_triple_product(first, second, third).unwrap();
                        assert_eq!(sum.finish().unwrap().to_bits(), 0);
                    }
                }
            }
            for invalid in [<$s>::NAN, <$s>::INFINITY, <$s>::NEG_INFINITY] {
                let mut sum = RangeAccumulator::new();
                sum.add(7.0 as $s).unwrap();
                let previous = sum.clone();
                for result in [
                    sum.add(invalid),
                    sum.add_product(0.0, invalid),
                    sum.subtract_product(invalid, -0.0),
                    sum.add_scaled_product(0.0, invalid, i32::MAX),
                    sum.add_triple_product(invalid, 0.0, 0.0),
                    sum.add_triple_product(1.0, invalid, 0.0),
                    sum.subtract_triple_product(1.0, 0.0, invalid),
                    sum.add_triple_products(invalid, []),
                ] {
                    assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
                }
                assert_eq!(sum, previous);
                for scale in [0.0 as $s, 1.0, 2.0] {
                    let mut stream = previous.clone();
                    assert!(
                        stream
                            .add_triple_products(scale, [(2.0, 3.0), (0.0, invalid)])
                            .is_err()
                    );
                    assert_eq!(stream.finish().unwrap(), 7.0 + scale * 6.0);
                }
                for length in 1..=5 {
                    for position in 0..length {
                        let mut factors = [0.0 as $s; 5];
                        factors[position] = invalid;
                        let zeros = [0.0 as $s; 5];
                        for result in [
                            checked_sum_of_products(&zeros[..length], &factors[..length]),
                            checked_sum_of_products(&factors[..length], &zeros[..length]),
                        ] {
                            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
                        }
                    }
                }
            }
            let mut complex = ProductSum::new();
            let left = Complex::new(1.5 as $s, -2.0);
            let right = Complex::new(-4.0 as $s, 0.5);
            complex
                .add_triple_product(Complex::new(1.0, -0.0), left, right)
                .unwrap();
            assert_eq!(complex.finish().unwrap(), left * right);
            complex.subtract_product(left, right).unwrap();
            assert_eq!(complex.finish().unwrap(), Complex::new(0.0, 0.0));
        }
    };
}
finite_updates!(binary32_finite_updates, f32);
finite_updates!(binary64_finite_updates, f64);

macro_rules! zero_contract {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let z = -0.0 as $s;
            let tiny = <$s>::from_bits(1);
            let negative_zero = z.to_bits();
            let exact_zero = [
                checked_difference(z, 0.0),
                checked_float_product(z, -1.0),
                checked_float_quotient(z, -1.0),
                checked_product_quotient(z, 1.0, -1.0),
                checked_float_product_factors(&[z, -1.0]),
                checked_sum(&[z, z]),
                checked_sum::<$s>(&[]),
                checked_sum_of_products(&[z], &[-1.0]),
                checked_difference_of_products(1.0 as $s, 1.0, 1.0, 1.0),
                checked_mul_add(z, 1.0, z),
                checked_mul_add(1.0 as $s, 1.0, -1.0),
                checked_affine(-1.0 as $s, 1.0, 0.5),
                checked_midpoint(-1.0 as $s, 1.0),
                checked_scaled_norm(&[z]),
                checked_scaled_norm::<$s>(&[]),
                RangeAccumulator::<$s>::new().finish(),
                RangeAccumulator::<$s>::new().finish_quotient(-1.0),
                ProductSum::<$s>::new().finish(),
            ];
            for value in exact_zero {
                assert_eq!(value.unwrap().to_bits(), 0);
            }
            for value in [
                checked_float_product(-tiny, 0.25),
                checked_float_quotient(-tiny, 4.0),
                checked_product_quotient(-tiny, 0.25, 1.0),
                checked_float_product_factors(&[-tiny, 0.25]),
                checked_mul_add(-tiny, 0.25, 0.0),
                checked_midpoint(-tiny, 0.0),
                checked_affine(-tiny, 0.0, 0.75),
                checked_scalbn(-tiny, -2),
            ] {
                assert_eq!(value.unwrap().to_bits(), negative_zero);
            }
            let mut sum = RangeAccumulator::<$s>::new();
            sum.add_product(-tiny, 0.25).unwrap();
            assert_eq!(sum.finish().unwrap().to_bits(), negative_zero);
            assert_eq!(sum.finish_quotient(2.0).unwrap().to_bits(), negative_zero);
            let mut complex = ProductSum::<Complex<$s>>::new();
            complex
                .add_product(Complex::new(-tiny, -tiny), Complex::new(0.25, 0.0))
                .unwrap();
            let result = complex.finish().unwrap();
            assert_eq!(result.re.to_bits(), negative_zero);
            assert_eq!(result.im.to_bits(), negative_zero);
            let norm = ScaledNorm::from_parts(z, z).unwrap();
            assert_eq!(norm.scale().to_bits(), 0);
            assert_eq!(norm.sum_squares().to_bits(), 0);
            assert_eq!(norm.finish().unwrap().to_bits(), 0);
            assert_eq!(checked_float_product_factors::<$s>(&[]).unwrap(), 1.0);
            // Representation/endpoint primitives explicitly retain their input bits.
            assert_eq!(
                checked_affine(z, 1.0, 0.0).unwrap().to_bits(),
                negative_zero
            );
            assert_eq!(
                checked_affine(1.0, z, 1.0).unwrap().to_bits(),
                negative_zero
            );
            assert_eq!(checked_midpoint(z, z).unwrap().to_bits(), negative_zero);
            assert_eq!(checked_scalbn(z, 20).unwrap().to_bits(), negative_zero);
            assert_eq!(
                checked_next_after(0.0 as $s, z).unwrap().to_bits(),
                negative_zero
            );
            for (a, b, c) in [(1.0 as $s, z, z), (1.0, 1.0, z), (1.0, -1.0, z)] {
                for root in checked_quadratic_roots(a, b, c).unwrap() {
                    if root.re == 0.0 {
                        assert_eq!(root.re.to_bits(), 0);
                    }
                    assert_eq!(root.im.to_bits(), 0);
                }
            }
            // Zero absorbs finite range, but cannot hide malformed input.
            assert_eq!(
                checked_float_product_factors(&[<$s>::MAX, <$s>::MAX, z])
                    .unwrap()
                    .to_bits(),
                0
            );
            assert_eq!(
                checked_float_product_factors(&[z, <$s>::NAN])
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
            assert!(checked_float_quotient(z, z).is_err());
            assert!(checked_product_quotient(z, 1.0, z).is_err());
            assert!(checked_affine(0.0 as $s, 1.0, -tiny).is_err());
            assert!(checked_affine(0.0 as $s, 1.0, <$s>::INFINITY).is_err());
            let mut norm = ScaledNorm::from_parts(1.0 as $s, <$s>::MAX).unwrap();
            let before = norm;
            assert!(norm.merge(before).is_err());
            assert_eq!(norm, before);
            assert!(norm.add_magnitude(-1.0).is_err());
            assert_eq!(norm, before);
        }
    };
}
zero_contract!(binary32_zero_contract, f32);
zero_contract!(binary64_zero_contract, f64);

#[test]
fn factor_products_normalize_every_subnormal_before_multiplication() {
    let mut binary32 = [f32::from_bits(0x7f00_0000); 16];
    binary32[..7].fill(f32::from_bits(1));
    assert_eq!(
        checked_float_product_factors(&binary32).unwrap().to_bits(),
        0x7180_0000
    );
    binary32[0] = -binary32[0];
    assert_eq!(
        checked_float_product_factors(&binary32).unwrap().to_bits(),
        0xf180_0000
    );
    let mut binary64 = [f64::from_bits(0x7fe0_0000_0000_0000); 44];
    binary64[..21].fill(f64::from_bits(1));
    assert_eq!(
        checked_float_product_factors(&binary64).unwrap().to_bits(),
        0x7ce0_0000_0000_0000
    );
    binary64[0] = -binary64[0];
    assert_eq!(
        checked_float_product_factors(&binary64).unwrap().to_bits(),
        0xfce0_0000_0000_0000
    );
}

#[test]
fn power_of_two_scaling_rounds_only_on_the_final_subnormal_grid() {
    assert_eq!(
        checked_scalbn(f64::from_bits(0x3ff0_0000_0000_0009), -1026)
            .unwrap()
            .to_bits(),
        0x0001_0000_0000_0001
    );
    assert_eq!(
        checked_scalbn(f32::from_bits(0x3f80_0009), -130)
            .unwrap()
            .to_bits(),
        0x0008_0001
    );
    assert_eq!(
        checked_scalbn(-1.0_f64, i32::MIN).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(checked_scalbn(1.0_f64, i32::MAX).is_err());
}

/// Long sums take a certified compensated path; every result must equal the
/// exact accumulator's single rounding, including cancellation, ties and
/// magnitudes near overflow.
fn long_sums_match_the_exact_accumulator<S: scioxide_core::RealScalar>() {
    let exact = |values: &[S]| {
        let mut accumulator = scioxide_core::RangeAccumulator::new();
        accumulator.add_weighted_values(S::ONE, values)?;
        accumulator.finish()
    };
    // A small deterministic generator keeps the cases reproducible.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let (minimum_exponent, maximum_exponent) = if S::MANTISSA_DIGITS == 24 {
        (-149, 127)
    } else {
        (-1074, 1023)
    };
    for case in 0..8000 {
        let len = 4 + (next() % 61) as usize;
        let spread = (next() % 5) as i32;
        let mut values: Vec<S> = (0..len)
            .map(|_| {
                let mantissa = (next() % 2_000_001) as f64 / 1_000_000.0 - 1.0;
                let exponent = (next() % (1 + 20 * spread as u64)) as i32 - 10 * spread;
                scalar(mantissa * 2.0_f64.powi(exponent))
            })
            .collect();
        match case % 10 {
            // Exact cancellation around a small residue.
            1 => {
                let half = len / 2;
                for i in 0..half {
                    values[half + i] = -values[i];
                }
            }
            // Halfway-style ties built from a power of two and its ulp halves.
            2 => {
                values.fill(S::ZERO);
                values[0] = S::ONE;
                values[1] = S::EPSILON * S::HALF;
                values[2] = S::EPSILON * S::HALF * S::HALF;
                values[3] = -(S::EPSILON * S::HALF * S::HALF);
            }
            // Magnitudes near the overflow guard.
            3 => {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = S::MAX * scalar(0.2) * if i % 2 == 0 { S::ONE } else { -S::ONE };
                }
            }
            4 => values.iter_mut().for_each(|value| *value = value.abs()),
            // Exercise the entire exponent range, not just ordinary magnitudes.
            5 => {
                for value in &mut values {
                    let exponent = minimum_exponent
                        + (next() % (maximum_exponent - minimum_exponent + 1) as u64) as i32;
                    *value = S::from_u64_half_open01(next()).scalbn(exponent)
                        * if next() & 1 == 0 { S::ONE } else { -S::ONE };
                }
            }
            6 => {
                for value in &mut values {
                    *value = scalar((next() % 2049) as f64 - 1024.0) * S::MIN_SUBNORMAL;
                }
            }
            // Powers of two have unequal neighboring spacings. Test either sign
            // and both sides of their midpoints, down to the subnormal boundary.
            7 | 8 => {
                let exponent = minimum_exponent
                    + (next() % (maximum_exponent - minimum_exponent + 1) as u64) as i32;
                let power = S::ONE.scalbn(exponent);
                let base = if case % 10 == 7 {
                    power
                } else {
                    power.next_after(S::MAX)
                };
                let neighbor = base.next_after(if next() & 1 == 0 { S::ZERO } else { S::MAX });
                let half_gap = (neighbor - base) * S::HALF;
                let residual = match (case / 10) % 3 {
                    0 => half_gap,
                    1 => half_gap.next_after(S::NEG_INFINITY),
                    _ => half_gap.next_after(S::INFINITY),
                };
                let sign = if next() & 1 == 0 { S::ONE } else { -S::ONE };
                values.fill(S::ZERO);
                values[0] = sign * base;
                values[1] = sign * residual;
            }
            9 => values.fill(S::MAX * scalar(0.2)),
            _ => {}
        }
        values.rotate_left(case % len);
        let expected = exact(&values);
        let actual = scioxide_core::checked_sum(&values);
        match (actual, expected) {
            (Ok(a), Ok(e)) => assert!(
                a == e && a.is_sign_negative() == e.is_sign_negative(),
                "case {case}: {a:?} != {e:?} for {values:?}"
            ),
            (Err(a), Err(e)) => assert_eq!(a.kind(), e.kind(), "case {case}"),
            (a, e) => panic!("case {case}: {a:?} vs {e:?}"),
        }
    }
    for len in [65, 257, 1024, 16384] {
        let values: Vec<_> = (0..len)
            .map(|i| S::from_u64_half_open01((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect();
        assert_eq!(
            scioxide_core::checked_sum(&values).unwrap(),
            exact(&values).unwrap()
        );
    }
    // Every lane and the unaligned remainder must retain input validation.
    for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
        for index in 0..9 {
            let mut values = [S::ONE; 9];
            values[index] = bad;
            assert_eq!(
                scioxide_core::checked_sum(&values).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
        }
    }
}

#[test]
fn certified_long_sums_are_correctly_rounded() {
    long_sums_match_the_exact_accumulator::<f64>();
    long_sums_match_the_exact_accumulator::<f32>();
}
