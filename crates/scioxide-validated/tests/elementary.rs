//! Public certified ranges, independent MPFR point witnesses and domain outcomes.
#![cfg(test)]
#![allow(clippy::panic)]
use scioxide_core::RealScalar;
use scioxide_validated::{DecoratedInterval as D, ElementaryEnclosure, EnclosureStatus as Status};
fn unary<S: RealScalar>(op: &str, x: D<S>) -> ElementaryEnclosure<S> {
    match op {
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "exp10" => x.exp10(),
        "exp_m1" => x.exp_m1(),
        "ln" => x.ln(),
        "ln_1p" => x.ln_1p(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        "cbrt" => x.cbrt(),
        _ => unreachable!(),
    }
    .unwrap()
}
#[test]
fn domains_and_splits() {
    assert_eq!(
        D::new(-2.0, 0.0).unwrap().ln().unwrap().status(),
        Status::Undefined
    );
    let clipped = D::new(-2.0, 2.0).unwrap().ln().unwrap();
    assert_eq!(clipped.status(), Status::PartiallyDefined);
    assert!(clipped.contains(-1000.0));
    let split = D::new(1.0, 2.0).unwrap().tan().unwrap();
    assert_eq!(split.status(), Status::PartiallyDefined);
    assert_eq!(split.pieces().len(), 2);
    assert!(!split.contains(0.0));
    assert!(split.contains(-1000.0) && split.contains(1000.0));
    for (a, b) in [(1.5, 4.75), (-4.75, -1.5)] {
        let multiple = D::new(a, b).unwrap().tan().unwrap();
        assert_eq!(multiple.status(), Status::PartiallyDefined);
        assert_eq!(multiple.hull().set(), D::<f64>::entire().set());
    }
    // The coarse lattice-distance bound overlaps this endpoint although its
    // exact value is below pi/2. Classification must remain explicitly unknown.
    let unresolved = D::new(0.0, f64::from_bits(0x3ff921fb54442d18))
        .unwrap()
        .tan()
        .unwrap();
    assert_eq!(unresolved.status(), Status::Inconclusive);
    assert!(unresolved.contains(-f64::MAX) && unresolved.contains(f64::MAX));
    assert_eq!(
        D::<f64>::empty().exp().unwrap().status(),
        Status::EmptyInput
    );
    assert_eq!(
        D::new(-1.0, 1.0).unwrap().atanh().unwrap().status(),
        Status::PartiallyDefined
    );
    assert!(D::new(-10.0, 10.0).unwrap().cosh().unwrap().contains(1.0));
    assert!(D::new(1.0, 2.0).unwrap().sin().unwrap().contains(1.0));
    assert!(D::<f64>::entire().atan().unwrap().contains(0.0));
    for op in [
        "exp", "exp2", "exp10", "exp_m1", "ln", "ln_1p", "log2", "log10", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt",
    ] {
        let _ = unary(op, D::<f64>::entire());
    }
}
#[test]
fn independent_core_mpfr_witnesses() {
    // This additional independent corpus predates the interval implementation.
    // Stratified replay keeps this smoke test inexpensive; the directed oracle
    // corpus separately covers each operation's adversarial interval cases.
    let data = include_str!("../../scioxide-core/tests/data/elementary.txt");
    let mut count = 0;
    for (index, line) in data
        .lines()
        .filter(|line| !line.starts_with('#'))
        .enumerate()
    {
        if index % 79 != 0 {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        let op = fields[1];
        if ["hypot", "atan2", "powf", "log"].contains(&op) {
            continue;
        }
        let a = u64::from_str_radix(fields[2], 16).unwrap();
        let expected = u64::from_str_radix(fields[4], 16).unwrap();
        if fields[0] == "64" {
            let x = f64::from_bits(a);
            let y = f64::from_bits(expected);
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let result = unary(op, D::point(x).unwrap());
            assert!(result.contains(y), "{op}({x:?}) = {y:?}; {result:?}");
        } else {
            let x = f32::from_bits(a as u32);
            let y = f32::from_bits(expected as u32);
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let result = unary(op, D::point(x).unwrap());
            assert!(result.contains(y), "{op}({x:?}) = {y:?}; {result:?}");
        }
        count += 1;
    }
    assert!(count > 300);
}
#[test]
fn binary_domains() {
    let zero_power = D::point(0.0)
        .unwrap()
        .powf(&D::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(zero_power.status(), Status::Defined);
    assert_eq!(zero_power.pieces().len(), 2);
    assert!(!zero_power.contains(0.5));
    assert!(
        zero_power
            .pieces()
            .iter()
            .all(|p| p.decoration() == scioxide_validated::Decoration::Def)
    );
    let joint_power = D::new(0.0, 2.0)
        .unwrap()
        .powf(&D::new(0.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        joint_power.hull().decoration(),
        scioxide_validated::Decoration::Def
    );
    let negative_axis = D::point(0.0)
        .unwrap()
        .atan2(&D::new(-2.0, -1.0).unwrap())
        .unwrap();
    assert_eq!(
        negative_axis.hull().decoration(),
        scioxide_validated::Decoration::Def
    );
    let split = D::new(-1.0, 1.0).unwrap().powi(-3).unwrap();
    assert_eq!(split.pieces().len(), 2);
    assert_eq!(split.status(), Status::PartiallyDefined);
    assert!(!split.contains(0.0));
    let powers = D::point(-2.0)
        .unwrap()
        .powf(&D::new(1.0, 4.0).unwrap())
        .unwrap();
    assert!(powers.gaps_filled());
    assert!(powers.contains(-8.0) && powers.contains(16.0));
    assert_eq!(powers.status(), Status::PartiallyDefined);
    assert_eq!(
        D::new(-2.0, -1.0)
            .unwrap()
            .powf(&D::new(0.1, 0.9).unwrap())
            .unwrap()
            .status(),
        Status::Undefined
    );
    let angles = D::new(-1.0, 1.0)
        .unwrap()
        .atan2(&D::new(-2.0, -1.0).unwrap())
        .unwrap();
    assert_eq!(angles.pieces().len(), 2);
    assert!(!angles.contains(0.0));
    assert_eq!(
        D::point(0.0)
            .unwrap()
            .atan2(&D::point(0.0).unwrap())
            .unwrap()
            .status(),
        Status::Undefined
    );
    let axis = D::point(0.0)
        .unwrap()
        .atan2(&D::new(-1.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(axis.pieces().len(), 2);
    assert!(axis.contains(0.0));
    assert!(!axis.contains(1.0));
    assert!(
        D::point(3.0)
            .unwrap()
            .hypot(&D::point(4.0).unwrap())
            .unwrap()
            .contains(5.0)
    );
    let logs = D::new(2.0, 3.0)
        .unwrap()
        .log(&D::new(0.5, 2.0).unwrap())
        .unwrap();
    assert_eq!(logs.pieces().len(), 2);
    assert!(!logs.contains(0.0));
    for n in [i32::MIN, -3, -2, 0, 1, 2, 3, i32::MAX] {
        let _ = D::<f64>::entire().powi(n).unwrap();
    }
    let _ = D::<f64>::entire().powf(&D::entire()).unwrap();
    let _ = D::<f64>::entire().atan2(&D::entire()).unwrap();
    let _ = D::<f64>::entire().hypot(&D::entire()).unwrap();
    let _ = D::<f64>::entire().log(&D::entire()).unwrap();
}
#[test]
fn independent_binary_mpfr_witnesses() {
    let data = include_str!("../../scioxide-core/tests/data/elementary.txt");
    let mut count = 0;
    for fields in data
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
    {
        let op = fields[1];
        if !["hypot", "atan2", "powf", "log"].contains(&op) {
            continue;
        }
        let decode = |s: &str| {
            let bits = u64::from_str_radix(s, 16).unwrap();
            if fields[0] == "64" {
                f64::from_bits(bits)
            } else {
                f64::from(f32::from_bits(bits as u32))
            }
        };
        let a = decode(fields[2]);
        let b = decode(fields[3]);
        let expected = decode(fields[4]);
        if !a.is_finite() || !b.is_finite() || !expected.is_finite() {
            continue;
        }
        if op == "atan2" && a == 0.0 && b == 0.0
            || op == "log" && (a <= 0.0 || b <= 0.0 || b == 1.0)
        {
            continue;
        }
        // The real negative axis uses +pi; IEEE negative zero selects -pi.
        if op == "atan2" && a == 0.0 && a.is_sign_negative() && b < 0.0 {
            continue;
        }
        let x = D::point(a).unwrap();
        let y = D::point(b).unwrap();
        let result = match op {
            "hypot" => x.hypot(&y),
            "atan2" => x.atan2(&y),
            "powf" => x.powf(&y),
            _ => x.log(&y),
        }
        .unwrap();
        // Binary32 oracle rounding can lie outside binary64 endpoints; replay
        // those cases through their actual destination width.
        if fields[0] == "64" {
            assert!(
                result.contains(expected),
                "{op}({a},{b})={expected}; {result:?}"
            );
        } else {
            let x = D::point(a as f32).unwrap();
            let y = D::point(b as f32).unwrap();
            let result = match op {
                "hypot" => x.hypot(&y),
                "atan2" => x.atan2(&y),
                "powf" => x.powf(&y),
                _ => x.log(&y),
            }
            .unwrap();
            assert!(
                result.contains(expected as f32),
                "{op}({a},{b})={expected}; {result:?}"
            );
        }
        count += 1;
    }
    assert!(count > 500);
}
#[test]
fn signed_zero_fixture_boundaries() {
    let mut covered = std::collections::BTreeSet::new();
    for line in include_str!("data/elementary.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let width = fields[0];
        let op = fields[1];
        if !["atan2", "hypot", "powf", "powi"].contains(&op) {
            continue;
        }
        let sign = 1_u64 << if width == "32" { 31 } else { 63 };
        let exponent_bits = u64::from_str_radix(fields[4], 16).unwrap();
        let exponent = if width == "32" {
            f64::from(f32::from_bits(exponent_bits as u32))
        } else {
            f64::from_bits(exponent_bits)
        };
        let power = if exponent < 0.0 && exponent % 2.0 == -1.0 {
            "odd"
        } else if exponent < 0.0 && exponent.fract() != 0.0 {
            "fractional"
        } else {
            "other"
        };
        let kind = if ["powf", "powi"].contains(&op) {
            power
        } else {
            "binary"
        };
        for (operand, endpoints) in fields[2..6].chunks_exact(2).enumerate() {
            for endpoint in endpoints {
                let bits = u64::from_str_radix(endpoint, 16).unwrap();
                if bits & (sign - 1) == 0 {
                    covered.insert((width, op, bits & sign != 0, operand, kind));
                }
            }
        }
    }
    for width in ["32", "64"] {
        for negative in [false, true] {
            for op in ["atan2", "hypot"] {
                for operand in [0, 1] {
                    assert!(covered.contains(&(width, op, negative, operand, "binary")));
                }
            }
            for (op, kind) in [("powi", "odd"), ("powf", "odd"), ("powf", "fractional")] {
                assert!(covered.contains(&(width, op, negative, 0, kind)));
            }
        }
    }
}

#[test]
fn directed_mpfr_ranges() {
    let data = include_str!("data/elementary.txt");
    let mut count = 0;
    let mut inconclusive = 0;
    let mut widths = std::collections::BTreeMap::<String, u64>::new();
    for line in data.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = line.split_whitespace().collect();
        let width = fields[0];
        let op = fields[1];
        let decode = |s: &str| {
            let bits = u64::from_str_radix(s, 16).unwrap();
            if width == "64" {
                f64::from_bits(bits)
            } else {
                f64::from(f32::from_bits(bits as u32))
            }
        };
        let inputs: [f64; 4] = core::array::from_fn(|i| decode(fields[i + 2]));
        fn check<S: RealScalar>(
            op: &str,
            inputs: [f64; 4],
            fields: &[&str],
            decode: &dyn Fn(&str) -> f64,
        ) -> (bool, Option<u64>) {
            let make = |a: f64, b: f64| {
                D::from(
                    scioxide_core::ExtendedInterval::new(
                        if a == f64::NEG_INFINITY {
                            None
                        } else {
                            Some(S::try_from_f64_exact(a).unwrap())
                        },
                        if b == f64::INFINITY {
                            None
                        } else {
                            Some(S::try_from_f64_exact(b).unwrap())
                        },
                    )
                    .unwrap(),
                )
            };
            let [a, b, c, d] = inputs;
            let x = make(a, b);
            let y = make(c, d);
            let result = match op {
                "hypot" => x.hypot(&y).unwrap(),
                "atan2" => x.atan2(&y).unwrap(),
                "powi" => x.powi(c as i32).unwrap(),
                "powf" => x.powf(&y).unwrap(),
                "log" => x.log(&y).unwrap(),
                _ => unary(op, x),
            };
            if result.status() != Status::Inconclusive {
                assert_eq!(
                    format!("{:?}", result.status()),
                    fields[6],
                    "{op} {inputs:?}: {result:?}"
                );
            } else {
                assert_eq!(op, "tan");
            }
            let n: usize = fields[7].parse().unwrap();
            for i in 0..n {
                let expected = make(decode(fields[8 + 2 * i]), decode(fields[9 + 2 * i]));
                assert!(
                    result
                        .pieces()
                        .iter()
                        .any(|piece| piece.contains_interval(&expected)),
                    "{op} {inputs:?}: {result:?} excludes {expected:?}"
                );
            }
            let point_width =
                if a == b && !["hypot", "atan2", "powi", "powf", "log"].contains(&op) && n == 1 {
                    let oracle_lo = decode(fields[8]);
                    let oracle_hi = decode(fields[9]);
                    if oracle_lo.is_finite() && oracle_hi.is_finite() {
                        let set = result.hull().set();
                        let (Some(lo), Some(hi)) = set.bounds().unwrap() else {
                            panic!("uninformative finite point: {op} {inputs:?}")
                        };
                        // Use sign-magnitude monotone keys without overflowing the
                        // mask for binary64; signed zero occupies adjacent keys.
                        let ordered = |x: S| {
                            let (bits, sign, mask) = if core::mem::size_of::<S>() == 4 {
                                (
                                    (x.into_f64() as f32).to_bits() as u64,
                                    1_u64 << 31,
                                    u64::from(u32::MAX),
                                )
                            } else {
                                (x.into_f64().to_bits(), 1_u64 << 63, u64::MAX)
                            };
                            if bits & sign != 0 {
                                (!bits) & mask
                            } else {
                                bits | sign
                            }
                        };
                        let diameter = ordered(hi) - ordered(lo);
                        assert!(
                            diameter <= 2,
                            "point enclosure exceeds 2 ULP: {op} {inputs:?}, width={diameter}"
                        );
                        Some(diameter)
                    } else {
                        None
                    }
                } else {
                    None
                };
            (result.status() == Status::Inconclusive, point_width)
        }
        let (uncertain, point_width) = if width == "64" {
            check::<f64>(op, inputs, &fields, &decode)
        } else {
            check::<f32>(op, inputs, &fields, &decode)
        };
        inconclusive += usize::from(uncertain);
        if let Some(value) = point_width {
            widths
                .entry(format!("f{width}/{op}"))
                .and_modify(|maximum| *maximum = (*maximum).max(value))
                .or_insert(value);
        }
        count += 1;
    }
    assert!(count > 6000);
    // Unresolved branch classification is permitted and counted, not disguised
    // as domain success. The corpus is expected to resolve every chosen case.
    assert_eq!(inconclusive, 0);
    println!(
        "directed fixtures={count}, inconclusive={inconclusive}, maximum point widths in destination steps: {widths:?}"
    );
}

#[test]
fn interval_inclusion_and_continuity() {
    for op in [
        "exp", "exp2", "exp10", "exp_m1", "ln", "ln_1p", "log2", "log10", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt",
    ] {
        for (a, b) in [
            (-4.0, -0.1),
            (-1.0, 1.0),
            (0.1, 0.9),
            (1.0, 3.0),
            (100.0, 101.0),
        ] {
            let outer = unary(op, D::new(a, b).unwrap());
            for t in [0.0, 0.125, 0.5, 0.875, 1.0] {
                let x = (1.0 - t) * a + t * b;
                let inner = unary(op, D::point(x).unwrap());
                for piece in inner.pieces() {
                    // Independently rounded endpoint bounds can differ in width
                    // at boundaries; the fixed interior midpoint is a witness.
                    let set = piece.set();
                    if let Some((Some(lo), Some(hi))) = set.bounds() {
                        let witness = lo * 0.5 + hi * 0.5;
                        assert!(
                            outer.contains(witness),
                            "{op} [{a},{b}], x={x}, {outer:?} excludes {inner:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn directed_mpfr_points_and_exp2_cancellation() {
    fn check<S: RealScalar>(op: &str, x: f64, lo: f64, hi: f64) {
        let value = S::try_from_f64_exact(x).unwrap();
        let result = unary(op, D::point(value).unwrap());
        for expected in [lo, hi].into_iter().filter(|x| x.is_finite()) {
            assert!(
                result.contains(S::try_from_f64_exact(expected).unwrap()),
                "{op}({x:e}) excludes {expected:e}: {result:?}"
            );
        }
        if lo.is_finite() && hi.is_finite() {
            let (Some(a), Some(b)) = result.hull().set().bounds().unwrap() else {
                panic!("finite MPFR point has unbounded enclosure: {op}({x:e})");
            };
            let ordered = |v: S| {
                let (bits, sign, mask) = if core::mem::size_of::<S>() == 4 {
                    (
                        u64::from((v.into_f64() as f32).to_bits()),
                        1_u64 << 31,
                        u64::from(u32::MAX),
                    )
                } else {
                    (v.into_f64().to_bits(), 1_u64 << 63, u64::MAX)
                };
                if bits & sign == 0 {
                    bits | sign
                } else {
                    (!bits) & mask
                }
            };
            let steps = ordered(b) - ordered(a);
            assert!(steps <= 2, "{op}({x:e}) width={steps}: {result:?}");
        }
    }
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for line in include_str!("data/elementary-points.txt")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let decode = |word| {
            let bits = u64::from_str_radix(word, 16).unwrap();
            if fields[0] == "32" {
                f64::from(f32::from_bits(bits as u32))
            } else {
                f64::from_bits(bits)
            }
        };
        let [x, lo, hi] = core::array::from_fn(|i| decode(fields[i + 2]));
        if fields[0] == "32" {
            check::<f32>(fields[1], x, lo, hi);
        } else {
            check::<f64>(fields[1], x, lo, hi);
        }
        *counts
            .entry(format!("{}/{}", fields[0], fields[1]))
            .or_default() += 1;
    }
    assert_eq!(counts.len(), 42);
    assert!(counts["64/exp2"] >= 8192);
}
