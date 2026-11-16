//! Elementary rules against the independent high-precision references in `tests/data/`.
#![allow(clippy::unwrap_used, clippy::panic)]

mod support;

use std::collections::BTreeSet;

use scioxide_ad::{Dual, Real};
use scioxide_core::RealScalar;
use support::{close, decode, moderate, rows};

fn unary<S: RealScalar, R: Real<S>>(name: &str, x: R) -> R {
    match name {
        "sqrt" => x.sqrt(),
        "cbrt" => x.cbrt(),
        "recip" => x.recip(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "asinh" => x.asinh(),
        "acosh" => x.acosh(),
        "atanh" => x.atanh(),
        "expm1" => x.exp_m1(),
        "ln1p" => x.ln_1p(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "exp2" => x.exp2(),
        "tanh" => x.tanh(),
        "softplus" => x.softplus(),
        "sigmoid" => x.sigmoid(),
        "log_sigmoid" => x.log_sigmoid(),
        "logit" => x.logit(),
        "exprel" => x.exprel(),
        "cosm1" => x.cosm1(),
        "sinc" => x.sinc(),
        "sincpi" => x.sincpi(),
        "sin_pi" => x.sin_pi(),
        "cos_pi" => x.cos_pi(),
        "tan_pi" => x.tan_pi(),
        "sin_degrees" => x.sin_degrees(),
        "cos_degrees" => x.cos_degrees(),
        "tan_degrees" => x.tan_degrees(),
        _ => panic!("unknown unary {name}"),
    }
}

fn binary<S: RealScalar, R: Real<S>>(name: &str, x: R, y: R) -> R {
    match name {
        "powf" => x.powf(y),
        "powm1" => x.powm1(y),
        "hypot" => x.hypot(y),
        "atan2" => x.atan2(y),
        "logaddexp" => x.logaddexp(y),
        "xlogy" => x.xlogy(y),
        "xlog1py" => x.xlog1py(y),
        "div" => x / y,
        _ => panic!("unknown binary {name}"),
    }
}

const BINARY: [&str; 8] = [
    "powf",
    "powm1",
    "hypot",
    "atan2",
    "logaddexp",
    "xlogy",
    "xlog1py",
    "div",
];

/// `x` seeded by `a` in the inner and `b` in the outer direction.
fn seed2<S: RealScalar>(x: S, a: S, b: S) -> Dual<Dual<S>> {
    Dual::new(Dual::new(x, a), Dual::new(b, S::ZERO))
}

fn unary_first<S: RealScalar>(name: &str, x: S) -> S {
    unary(name, Dual::variable(x)).derivative()
}

fn unary_second<S: RealScalar>(name: &str, x: S) -> S {
    unary(name, seed2(x, S::ONE, S::ONE))
        .derivative()
        .derivative()
}

fn local<S: RealScalar>(width: usize) {
    let mut coverage = BTreeSet::new();
    for (row, fields) in rows(include_str!("data/local_derivatives.txt")) {
        if fields[0].parse::<usize>().unwrap() != width {
            continue;
        }
        let name = fields[1];
        coverage.insert(name);
        let x = decode::<S>(fields[2], width);
        let y = decode::<S>(fields[3], width);
        let e: Vec<S> = fields[4..].iter().map(|v| decode(v, width)).collect();
        let (zero, one) = (S::ZERO, S::ONE);
        if BINARY.contains(&name) {
            let value = binary(name, x, y);
            let dx = binary(name, Dual::new(x, one), Dual::new(y, zero));
            let dy = binary(name, Dual::new(x, zero), Dual::new(y, one));
            assert_eq!(support::bits(dx.value()), support::bits(value), "{row}");
            if moderate(&[x, y, e[0], e[1], e[2]]) {
                close(dx.derivative(), e[1], 64, row);
                close(dy.derivative(), e[2], 64, row);
            }
            if moderate(&[x, y, e[0], e[1], e[2], e[3], e[4], e[5]]) {
                let second = |a: [S; 2], b: [S; 2]| {
                    binary(name, seed2(x, a[0], b[0]), seed2(y, a[1], b[1]))
                        .derivative()
                        .derivative()
                };
                close(second([one, zero], [one, zero]), e[3], 256, row);
                close(second([one, zero], [zero, one]), e[4], 256, row);
                close(second([zero, one], [zero, one]), e[5], 256, row);
            }
        } else {
            assert_eq!(
                unary(name, Dual::variable(x)).value(),
                unary(name, x),
                "{row}"
            );
            close(unary_first(name, x), e[1], 64, row);
            if moderate(&[x, e[0], e[1], e[3]]) {
                close(unary_second(name, x), e[3], 256, row);
            }
        }
    }
    assert_eq!(coverage.len(), 19, "{coverage:?}");
}

#[test]
fn local_partials_match_references() {
    local::<f32>(32);
    local::<f64>(64);
}

fn boundaries<S: RealScalar>(width: usize) -> usize {
    let mut count = 0;
    for (row, fields) in rows(include_str!("data/elementary_boundaries.txt")) {
        if fields[0].parse::<usize>().unwrap() != width {
            continue;
        }
        count += 1;
        let name = fields[1];
        let x = decode::<S>(fields[2], width);
        let e: Vec<S> = fields[3..].iter().map(|v| decode(v, width)).collect();
        close(unary(name, x), e[0], 4, row);
        close(unary_first(name, x), e[1], 4, row);
        if moderate(&[x, e[0], e[1], e[2]]) {
            close(unary_second(name, x), e[2], 256, row);
        }
    }
    count
}

#[test]
fn boundary_partials_match_references() {
    assert_eq!(boundaries::<f32>(32) + boundaries::<f64>(64), 178);
}

fn powers<S: RealScalar>(width: usize) {
    for (row, fields) in rows(include_str!("data/power_boundaries.txt")) {
        if fields[0].parse::<usize>().unwrap() != width {
            continue;
        }
        let name = fields[1];
        let x = decode::<S>(fields[2], width);
        let p = decode::<S>(fields[3], width);
        let e: Vec<S> = fields[4..].iter().map(|v| decode(v, width)).collect();
        let (zero, one) = (S::ZERO, S::ONE);
        if moderate(&[x, p, e[0], e[1], e[2]]) {
            let dx = binary(name, Dual::new(x, one), Dual::new(p, zero));
            let dp = binary(name, Dual::new(x, zero), Dual::new(p, one));
            close(dx.derivative(), e[1], 64, row);
            close(dp.derivative(), e[2], 64, row);
        }
        if moderate(&[x, p, e[0], e[1], e[2], e[3], e[4], e[5]]) {
            let second = |a: [S; 2], b: [S; 2]| {
                binary(name, seed2(x, a[0], b[0]), seed2(p, a[1], b[1]))
                    .derivative()
                    .derivative()
            };
            close(second([one, zero], [one, zero]), e[3], 256, row);
            close(second([one, zero], [zero, one]), e[4], 256, row);
            close(second([zero, one], [zero, one]), e[5], 256, row);
        }
    }
}

#[test]
fn power_partials_match_references() {
    powers::<f32>(32);
    powers::<f64>(64);
}

fn scalar_constants<S: RealScalar>(width: usize) -> usize {
    let mut count = 0;
    for (row, fields) in rows(include_str!("data/scalar_derivatives.txt")) {
        if fields[0].parse::<usize>().unwrap() != width {
            continue;
        }
        count += 1;
        let x = decode::<S>(fields[2], width);
        close(unary_first(fields[1], x), decode(fields[3], width), 1, row);
    }
    count
}

#[test]
fn constant_scaled_derivatives_are_within_one_ulp() {
    assert!(scalar_constants::<f32>(32) > 0);
    assert!(scalar_constants::<f64>(64) > 0);
}

fn stable<S: RealScalar>(width: usize) -> BTreeSet<String> {
    let mut coverage = BTreeSet::new();
    for (row, fields) in rows(include_str!("data/stable_derivatives.txt")) {
        if fields.len() != 6 || fields[0].parse::<usize>().ok() != Some(width) {
            continue;
        }
        let name = fields[1];
        coverage.insert(name.to_owned());
        let x = decode::<S>(fields[2], width);
        let value = decode::<S>(fields[3], width);
        let first = decode::<S>(fields[4], width);
        let second = decode::<S>(fields[5], width);
        close(unary(name, x), value, 64, row);
        close(unary_first(name, x), first, 64, row);
        close(unary_second(name, x), second, 64, row);
    }
    coverage
}

#[test]
fn stable_transform_derivatives_match_references() {
    assert_eq!(stable::<f32>(32).len(), 14);
    assert_eq!(stable::<f64>(64).len(), 14);
}
