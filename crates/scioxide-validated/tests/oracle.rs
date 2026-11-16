//! Independent directed MPFR bounds, including adjacent-endpoint tightness.
#![cfg(test)]
#![allow(clippy::panic)]

use scioxide_validated::{DecoratedInterval, Decoration};

macro_rules! oracle {
    ($name:ident, $scalar:ty, $bits:ty, $width:literal) => {
        #[test]
        fn $name() {
            let mut count = 0;
            for (line_number, line) in include_str!("data/arithmetic.txt").lines().enumerate() {
                let fields: Vec<_> = line.split_whitespace().collect();
                if fields.first() != Some(&$width) {
                    continue;
                }
                let values: Vec<_> = fields[2..8]
                    .iter()
                    .map(|x| <$scalar>::from_bits(<$bits>::from_str_radix(x, 16).unwrap()))
                    .collect();
                let a = DecoratedInterval::new(values[0], values[1]).unwrap();
                let b = DecoratedInterval::new(values[2], values[3]).unwrap();
                let result = match fields[1] {
                    "add" => a.add(&b),
                    "sub" => a.sub(&b),
                    "mul" => a.mul(&b),
                    "div" => a.div(&b),
                    "sqrt" => a.sqrt(),
                    "square" => a.square(),
                    "abs" => a.abs(),
                    _ => panic!("bad fixture operation"),
                }
                .unwrap();
                let (lo, hi) = result.set().bounds().unwrap();
                assert_eq!(
                    lo.unwrap_or(<$scalar>::NEG_INFINITY).to_bits(),
                    values[4].to_bits(),
                    "lower: line {}: {line}",
                    line_number + 1
                );
                assert_eq!(
                    hi.unwrap_or(<$scalar>::INFINITY).to_bits(),
                    values[5].to_bits(),
                    "upper: line {}: {line}",
                    line_number + 1
                );
                let expected = match fields[8] {
                    "Com" => Decoration::Com,
                    "Dac" => Decoration::Dac,
                    "Trv" => Decoration::Trv,
                    _ => panic!("bad decoration"),
                };
                assert_eq!(result.decoration(), expected, "decoration: {line}");
                count += 1;
            }
            assert!(count > 3000);
        }
    };
}
oracle!(exact_f32, f32, u32, "32");
oracle!(exact_f64, f64, u64, "64");
