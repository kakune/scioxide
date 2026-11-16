//! Active independent FLINT ball/exact fixtures; comparison never generates its own oracle.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
use std::collections::BTreeMap;

fn ordered(x: f64, width: u32) -> u64 {
    let bits = if width == 32 {
        u64::from((x as f32).to_bits())
    } else {
        x.to_bits()
    };
    let sign = 1_u64 << (width - 1);
    if bits & sign != 0 {
        sign - (bits & !sign)
    } else {
        sign + bits
    }
}
fn decode(bits: u64, width: u32) -> f64 {
    if width == 32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    }
}
fn spacing(x: f64, width: u32) -> f64 {
    if width == 32 {
        let x = (x as f32).abs();
        let next = x.next_after(f32::INFINITY);
        f64::from(if next.is_finite() {
            next - x
        } else {
            x - x.next_after(0.0)
        })
    } else {
        let x = x.abs();
        let next = x.next_after(f64::INFINITY);
        if next.is_finite() {
            next - x
        } else {
            x - x.next_after(0.0)
        }
    }
}
/// Kinds 7–10 accept real and complex arguments.
fn analytic<T: NumericScalar>(kind: usize, k: usize, a: T, x: T) -> SpecialFunctionResult<T> {
    let options = CombinatorialOptions::default();
    match kind {
        7 => pochhammer_with_options(a, x, options),
        8 => pochhammer_relative_with_options(a, x, options),
        9 => generalized_binomial_with_options(a, k, options),
        _ => gamma_with_options(a + T::ONE, GammaOptions::default()),
    }
    .unwrap()
}
fn real<S: RealScalar>(
    kind: usize,
    n: usize,
    k: usize,
    a: [f64; 2],
    x: [f64; 2],
) -> SpecialFunctionResult<S> {
    let options = CombinatorialOptions::default();
    match kind {
        0 => factorial_with_options(n, CombinatorialOptions::default()),
        1 => multifactorial_with_options(n, k, options),
        2 => binomial_with_options(n, k, CombinatorialOptions::default()),
        3 => bernoulli_with_options(n, options),
        4 => euler_with_options(n, options),
        5 => stirling_first_kind_with_options(n, k, options),
        6 => stirling_second_kind_with_options(n, k, options),
        _ => {
            let value = |v: [f64; 2]| S::try_from_f64_lossy(v[0]).unwrap();
            return analytic(kind, k, value(a), value(x));
        }
    }
    .unwrap()
}
fn complex_case<S: RealScalar>(
    kind: usize,
    k: usize,
    a: [f64; 2],
    x: [f64; 2],
) -> SpecialFunctionResult<Complex<S>>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = |v: [f64; 2]| {
        Complex::new(
            S::try_from_f64_lossy(v[0]).unwrap(),
            S::try_from_f64_lossy(v[1]).unwrap(),
        )
    };
    analytic(kind, k, z(a), z(x))
}
fn check<T: NumericScalar>(
    r: SpecialFunctionResult<T>,
    expected: [f64; 2],
    log: f64,
    width: u32,
    complex: bool,
    case: &str,
) -> (u64, u64, bool) {
    let target = if complex { 256 } else { 64 };
    let mut okay = r.termination().is_success();
    let mut distance = 0;
    if expected.iter().all(|v| v.is_finite()) && (expected != [0.0, 0.0] || log == 0.0) {
        if let Some(v) = r.value() {
            let actual = [
                v.value().real_component().into_f64(),
                v.value().imaginary_component().into_f64(),
            ];
            distance = actual
                .iter()
                .zip(expected)
                .map(|(&a, b)| ordered(a, width).abs_diff(ordered(b, width)))
                .max()
                .unwrap();
            okay &= distance <= target;
            let error = (actual[0] - expected[0]).hypot(actual[1] - expected[1]);
            okay &= error
                <= v.absolute_error().into_f64()
                    + expected[0].abs().max(expected[1].abs()) * 2.0 * T::Real::EPSILON.into_f64();
        } else {
            okay = false;
        }
    } else {
        okay &= r.value().is_none() && r.scaled().is_some();
    }
    let mut ld = 0;
    if expected != [0.0, 0.0] {
        if let Some(l) = r.logarithmic().and_then(|v| v.value().log_magnitude()) {
            ld = ordered(l.into_f64(), width).abs_diff(ordered(log, width));
            // Fixed independent acceptance bounds, separate from the model's
            // error estimate. A log zero is ill-conditioned in relative units.
            let err = (l.into_f64() - log).abs();
            let eps = T::Real::EPSILON.into_f64();
            let mut bound = 64.0 * spacing(log, width);
            if expected.iter().all(|v| v.is_finite()) {
                let norm = expected[0].hypot(expected[1]);
                let perturbation = (target as f64)
                    * spacing(expected[0], width).hypot(spacing(expected[1], width));
                if perturbation < norm {
                    // log's conditioning maps the independently fixed value-ULP
                    // allowance into log units; it never uses the model estimate.
                    bound += -(-(perturbation / norm)).ln_1p();
                }
            }
            okay &= err <= bound;
            okay &=
                err <= r.logarithmic().unwrap().absolute_error().into_f64() + 2.0 * eps * log.abs();
        } else {
            okay = false;
        }
    }
    if !okay {
        eprintln!("FAIL {case}: ulp={distance} log_ulp={ld} expected={expected:?} actual={r:?}");
    }
    (distance, ld, okay)
}
#[test]
fn independent_combinatorial_reference() {
    type Metrics = (usize, u64, u64, usize);
    let mut stats: BTreeMap<(u32, usize, bool), Metrics> = BTreeMap::new();
    for line in include_str!("data/combinatorial.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let width: u32 = f[0].parse().unwrap();
        let kind = f[1].parse().unwrap();
        let n = f[2].parse().unwrap();
        let k = f[3].parse().unwrap();
        let b: Vec<_> = f[4..]
            .iter()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        let a = [decode(b[0], width), decode(b[1], width)];
        let x = [decode(b[2], width), decode(b[3], width)];
        let expected = [decode(b[4], width), decode(b[5], width)];
        let log = decode(b[6], width);
        let complex = a[1] != 0.0 || x[1] != 0.0;
        let (d, ld, okay) = match (width, complex) {
            (32, false) => check(
                real::<f32>(kind, n, k, a, x),
                expected,
                log,
                width,
                complex,
                line,
            ),
            (64, false) => check(
                real::<f64>(kind, n, k, a, x),
                expected,
                log,
                width,
                complex,
                line,
            ),
            (32, true) => check(
                complex_case::<f32>(kind, k, a, x),
                expected,
                log,
                width,
                complex,
                line,
            ),
            _ => check(
                complex_case::<f64>(kind, k, a, x),
                expected,
                log,
                width,
                complex,
                line,
            ),
        };
        let s = stats.entry((width, kind, complex)).or_default();
        s.0 += 1;
        s.1 = s.1.max(d);
        s.2 = s.2.max(ld);
        s.3 += usize::from(!okay);
    }
    println!("width\tkind\tcomplex\tcases\tmax_ulp\tmax_log_ulp\tfailures");
    for ((width, kind, complex), (cases, d, ld, fail)) in &stats {
        println!("{width}\t{kind}\t{complex}\t{cases}\t{d}\t{ld}\t{fail}");
    }
    assert!(
        stats.values().all(|s| s.3 == 0),
        "independent accuracy reference check failed"
    );
}
