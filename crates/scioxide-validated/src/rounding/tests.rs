//! Independent exact-integer residuals exercise every destination exponent.
use super::*;
use scioxide_core::RangeAccumulator;

fn reference<S: RealScalar>(a: S, b: S, op: Operation, up: bool) -> S {
    let candidate = match op {
        Operation::Add => a + b,
        Operation::Multiply => a * b,
        Operation::Divide => a / b,
        Operation::Sqrt => a.sqrt(),
    };
    if candidate.is_infinite() {
        return match (candidate > S::ZERO, up) {
            (true, true) => S::INFINITY,
            (true, false) => S::MAX,
            (false, true) => -S::MAX,
            (false, false) => -S::INFINITY,
        };
    }
    let mut sum = RangeAccumulator::new();
    match op {
        Operation::Add => {
            sum.add(a).unwrap();
            sum.add(b).unwrap();
            sum.add(-candidate).unwrap();
        }
        Operation::Multiply => {
            sum.add_product(a, b).unwrap();
            sum.add(-candidate).unwrap();
        }
        Operation::Divide => {
            sum.add(a).unwrap();
            sum.add_product(-candidate, b).unwrap();
        }
        Operation::Sqrt => {
            sum.add(a).unwrap();
            sum.add_product(-candidate, candidate).unwrap();
        }
    }
    let sign = if matches!(op, Operation::Divide) && b < S::ZERO {
        sum.sign().reverse()
    } else {
        sum.sign()
    };
    match (sign, up) {
        (Ordering::Less, false) => candidate.next_after(-S::INFINITY),
        (Ordering::Greater, true) => candidate.next_after(S::INFINITY),
        _ => candidate,
    }
}

fn check<S: RealScalar>(a: S, b: S) {
    for op in [
        Operation::Add,
        Operation::Multiply,
        Operation::Divide,
        Operation::Sqrt,
    ] {
        if matches!(op, Operation::Divide) && b == S::ZERO {
            continue;
        }
        if matches!(op, Operation::Sqrt) && a < S::ZERO {
            continue;
        }
        for up in [false, true] {
            let expected = reference(a, b, op, up).into_f64().to_bits();
            let actual = directed(a, b, op, up).into_f64().to_bits();
            assert_eq!(actual, expected, "{op:?}, a={a:?}, b={b:?}, up={up}");
            let split = directed_with(a, b, op, up, false).into_f64().to_bits();
            assert_eq!(
                split, expected,
                "non-FMA correction: {op:?}, a={a:?}, b={b:?}, up={up}"
            );
            // mul_add rounds once on every host, so the fused residual signs
            // are exercised without hardware FMA.
            let fused = directed_with(a, b, op, up, true).into_f64().to_bits();
            assert_eq!(
                fused, expected,
                "fused correction: {op:?}, a={a:?}, b={b:?}, up={up}"
            );
        }
    }
    let x = a.abs();
    let (lo, hi) = square_bounds(x, x);
    for (actual, up) in [(lo, false), (hi, true)] {
        assert_eq!(
            actual.into_f64().to_bits(),
            reference(x, x, Operation::Multiply, up)
                .into_f64()
                .to_bits(),
            "square correction: x={x:?}, up={up}"
        );
    }
    let mut comparison = RangeAccumulator::new();
    comparison.add_product(a, b).unwrap();
    comparison.subtract_product(a, a).unwrap();
    assert_eq!(compare_products(a, b, a, a), comparison.sign());
}

fn boundaries<S: RealScalar>() {
    let p = S::MANTISSA_DIGITS as i32;
    let values = [
        S::ZERO,
        -S::ZERO,
        S::MIN_SUBNORMAL,
        -S::MIN_SUBNORMAL,
        S::MIN_SUBNORMAL * S::TWO,
        S::MIN_POSITIVE,
        S::MIN_POSITIVE.next_after(S::ZERO),
        S::MIN_POSITIVE.scalbn(p + 2),
        S::MIN_POSITIVE.scalbn(2 * p + 1),
        S::ONE - S::EPSILON,
        S::ONE,
        S::ONE + S::ONE / S::TWO,
        S::ONE + S::EPSILON,
        S::TWO,
        S::MAX.scalbn(-p),
        S::MAX.next_after(S::ZERO),
        S::MAX,
    ];
    for a in values {
        for b in values {
            check(a, b);
            check(-a, b);
            check(a, -b);
        }
    }
}
macro_rules! exercise {
    ($name:ident,$scalar:ty,$bits:ty) => {
        #[test]
        fn $name() {
            boundaries::<$scalar>();
            let mut seed = 2123_u64;
            for _ in 0..16384 {
                let mut next = || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    <$scalar>::from_bits(seed as $bits)
                };
                let a = next();
                let b = next();
                if a.is_finite() && b.is_finite() {
                    check(a, b);
                }
            }
        }
    };
}
exercise!(exact_residuals_f32, f32, u32);
exercise!(exact_residuals_f64, f64, u64);

fn corners<S: RealScalar>() {
    let values = [
        -S::MAX,
        -S::ONE,
        -S::MIN_SUBNORMAL,
        S::ZERO,
        S::MIN_SUBNORMAL,
        S::ONE,
        S::MAX,
    ];
    for (i, &a) in values.iter().enumerate() {
        for &b in &values[i..] {
            for (j, &c) in values.iter().enumerate() {
                for &d in &values[j..] {
                    let (lo, hi) = product_corners(a, b, c, d);
                    for (x, y) in [(a, c), (a, d), (b, c), (b, d)] {
                        let mut difference = RangeAccumulator::new();
                        difference.add_product(x, y).unwrap();
                        difference.subtract_product(lo.0, lo.1).unwrap();
                        assert!(!difference.sign().is_lt());
                        let mut difference = RangeAccumulator::new();
                        difference.add_product(x, y).unwrap();
                        difference.subtract_product(hi.0, hi.1).unwrap();
                        assert!(!difference.sign().is_gt());
                    }
                }
            }
        }
    }
}
#[test]
fn exact_corner_selection_including_overflow_and_underflow() {
    corners::<f32>();
    corners::<f64>();
}
