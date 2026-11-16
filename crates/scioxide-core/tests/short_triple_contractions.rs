//! Independent MPFR qualification of the guarded exact triple expansion.
#![allow(clippy::unwrap_used)]
use scioxide_core::{RangeAccumulator, backend::certified_short_sum_triples};

macro_rules! qualify {
    ($name:ident,$s:ty,$bits:ty,$width:literal) => {
        #[test]
        fn $name() {
            let verify = |terms: &[($s, $s, $s)], expected: $s| {
                let mut exact = RangeAccumulator::new();
                for &(a, b, c) in terms {
                    exact.add_triple_product(a, b, c).unwrap();
                }
                let actual = exact.finish();
                if expected.is_infinite() {
                    assert!(actual.is_err());
                } else {
                    assert_eq!(actual.unwrap().to_bits(), expected.to_bits(), "{terms:?}");
                }
                if let Some(actual) = certified_short_sum_triples(terms) {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "certificate {terms:?}"
                    );
                }
            };
            let mut rows = 0;
            let mut lengths = 0_u32;
            for row in include_str!("data/triple_contractions.txt")
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
                let mut terms = [(0.0 as $s, 0.0 as $s, 0.0 as $s); 8];
                for term in &mut terms[..count] {
                    *term = (read(), read(), read());
                }
                verify(&terms[..count], read());
                rows += 1;
                lengths |= 1 << count;
            }
            assert!(rows >= 2048);
            assert_eq!(lengths, 0b111111110);
            let mut state = 0xa719_cc01_42a9_b391_u64;
            let mut admitted = 0;
            for count in 1..=8 {
                for i in 0..8000 {
                    let mut next = || {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if i % 2 == 0 {
                            (state as i64 as $s) / (i64::MAX as $s) * 4.0
                        } else {
                            let x = <$s>::from_bits(state as $bits);
                            if x.is_finite() { x } else { 1.5 }
                        }
                    };
                    let terms: [($s, $s, $s); 8] =
                        core::array::from_fn(|_| (next(), next(), next()));
                    let mut exact = RangeAccumulator::new();
                    for &(a, b, c) in &terms[..count] {
                        exact.add_triple_product(a, b, c).unwrap();
                    }
                    verify(&terms[..count], exact.finish().unwrap_or(<$s>::INFINITY));
                    admitted += usize::from(certified_short_sum_triples(&terms[..count]).is_some());
                }
            }
            if cfg!(any(not(target_arch = "x86"), target_feature = "sse2")) {
                assert!(admitted > 28000);
            }
            assert!(certified_short_sum_triples(&[(1.0 as $s, 1.0, 1.0); 9]).is_none());
            assert!(certified_short_sum_triples(&[(0.0 as $s, <$s>::NAN, 0.0)]).is_none());
        }
    };
}
qualify!(binary32, f32, u32, "32");
qualify!(binary64, f64, u64, "64");
