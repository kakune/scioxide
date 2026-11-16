//! Independent nearest-even fixtures and direct portable/fused certification.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::backend::{
    certified_short_sum_products, certified_sum_two_products, certified_sum_two_products_with,
};
use scioxide_core::{RangeAccumulator, checked_sum_of_products};

macro_rules! checks {
    ($name:ident, $s:ty, $bits:ty, $width:literal) => {
        #[test]
        fn $name() {
            let verify = |terms: &[($s, $s)], expected: $s| {
                let mut accumulator = RangeAccumulator::new();
                let mut left = [0.0 as $s; 4];
                let mut right = left;
                for (i, &(a, b)) in terms.iter().enumerate() {
                    accumulator.add_product(a, b).unwrap();
                    left[i] = a;
                    right[i] = b;
                }
                for result in [
                    accumulator.finish(),
                    checked_sum_of_products(&left[..terms.len()], &right[..terms.len()]),
                    scioxide_core::backend::checked_product_sum(terms.iter().copied()),
                ] {
                    if expected.is_infinite() {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().to_bits(), expected.to_bits(), "{terms:?}");
                    }
                }
                let mut candidates = [certified_short_sum_products(terms), None, None];
                if let [(a, b), (c, d)] = terms {
                    candidates[1] = certified_sum_two_products(*a, *b, *c, *d);
                    candidates[2] = certified_sum_two_products_with(*a, *b, *c, *d, <$s>::mul_add);
                }
                for value in candidates.into_iter().flatten() {
                    assert_eq!(value.to_bits(), expected.to_bits(), "certificate {terms:?}");
                }
            };
            let mut fixture_count = 0;
            let mut lengths = 0_u32;
            for row in include_str!("data/contractions.txt")
                .lines()
                .filter(|r| !r.starts_with('#'))
            {
                let mut fields = row.split_whitespace();
                if fields.next().unwrap() != $width {
                    continue;
                }
                let count = fields.next().unwrap().parse::<usize>().unwrap();
                let _stratum = fields.next().unwrap();
                let mut read = || {
                    <$s>::from_bits(<$bits>::from_str_radix(fields.next().unwrap(), 16).unwrap())
                };
                let mut terms = [(0.0 as $s, 0.0 as $s); 4];
                for term in &mut terms[..count] {
                    *term = (read(), read());
                }
                verify(&terms[..count], read());
                fixture_count += 1;
                lengths |= 1 << count;
            }
            assert!(fixture_count >= 2000);
            assert_eq!(lengths, 0b11100);
            let mut state = 0xd314_87c2_831f_4eb9_u64;
            let mut admitted = [0_usize; 2];
            for i in 0..50000 {
                let terms: [$s; 4] = core::array::from_fn(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    if i % 2 == 0 {
                        (state as i64 as $s) / (i64::MAX as $s) * 32.0
                    } else {
                        let x = <$s>::from_bits(state as $bits);
                        if x.is_finite() { x } else { 1.5 }
                    }
                });
                let [a, b, c, d] = terms;
                let mut accumulator = RangeAccumulator::new();
                accumulator.add_product(a, b).unwrap();
                accumulator.add_product(c, d).unwrap();
                let expected = accumulator.finish().unwrap_or(<$s>::INFINITY);
                verify(&[(a, b), (c, d)], expected);
                admitted[0] += usize::from(certified_sum_two_products(a, b, c, d).is_some());
                admitted[1] += usize::from(
                    certified_sum_two_products_with(a, b, c, d, <$s>::mul_add).is_some(),
                );
            }
            assert!(
                admitted.into_iter().all(|n| {
                    if cfg!(any(not(target_arch = "x86"), target_feature = "sse2")) {
                        n >= 24000
                    } else {
                        n == 0
                    }
                }),
                "ordinary contractions must use the certificate"
            );
            for count in [3, 4] {
                for i in 0..10000 {
                    let terms: [($s, $s); 4] = core::array::from_fn(|_| {
                        let mut next = || {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            if i % 2 == 0 {
                                (state as i64 as $s) / (i64::MAX as $s) * 32.0
                            } else {
                                let x = <$s>::from_bits(state as $bits);
                                if x.is_finite() { x } else { 1.5 }
                            }
                        };
                        (next(), next())
                    });
                    let mut exact = RangeAccumulator::new();
                    for &(a, b) in &terms[..count] {
                        exact.add_product(a, b).unwrap();
                    }
                    verify(&terms[..count], exact.finish().unwrap_or(<$s>::INFINITY));
                }
            }
            // Both parities of a midpoint require exact fallback.
            for a in [1.0 as $s, 1.0 + <$s>::EPSILON] {
                assert!(certified_sum_two_products(a, 1.0, <$s>::EPSILON * 0.5, 1.0).is_none());
            }
        }
    };
}
checks!(binary32, f32, u32, "32");
checks!(binary64, f64, u64, "64");

fn exact_products<S: scioxide_core::RealScalar>() {
    use scioxide_core::backend::two_product;
    let s = |value: f64| S::try_from_f64_lossy(value).unwrap();
    let bound = if S::MANTISSA_DIGITS == 24 { 104 } else { 971 };
    let limit = S::ONE.scalbn(bound);
    let third = S::ONE / s(3.0);
    for (a, b) in [
        (third, s(3.0)),
        (s(0.1), s(-0.7)),
        (S::ONE + S::EPSILON, S::ONE - S::EPSILON),
        (limit, third),
        (-limit, s(0.75).scalbn(-bound)),
        (s(5.0), S::ZERO),
        (S::ONE, s(0.3)),
    ] {
        let (product, residual) = two_product(a, b);
        assert_eq!(product, a * b);
        assert_eq!(residual, a.mul_add(b, -product));
        let mut exact = RangeAccumulator::<S>::new();
        exact.add_product(a, b).unwrap();
        exact.subtract_product(product, S::ONE).unwrap();
        assert_eq!(exact.finish().unwrap(), residual, "{a:?} * {b:?}");
    }
}

#[test]
fn two_products_return_exact_rounding_residuals() {
    exact_products::<f32>();
    exact_products::<f64>();
}

#[test]
fn indexed_lanes_request_every_row_once_in_order() {
    for count in 0..=64 {
        let mut next = 0;
        let [sum] = scioxide_core::backend::four_lane_sums(count, |index| {
            assert_eq!(index, next);
            next += 1;
            [1.0_f64]
        });
        assert_eq!(next, count);
        assert_eq!(sum, count as f64);
    }
}
