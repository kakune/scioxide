//! Independent two-precision boundary coverage; no oracle library at test time.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::*;
use std::collections::{BTreeMap, BTreeSet};

const REAL_OPS: [&str; 17] = [
    "softplus",
    "sigmoid",
    "log_sigmoid",
    "logit",
    "exprel",
    "cosm1",
    "sinc",
    "sincpi",
    "sin_pi",
    "cos_pi",
    "tan_pi",
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
    "xlogy",
    "xlog1py",
    "powm1",
];
const COMPLEX_OPS: [&str; 19] = [
    "softplus",
    "sigmoid",
    "log_sigmoid",
    "logit",
    "exprel",
    "cosm1",
    "sinc",
    "sincpi",
    "sin_pi",
    "cos_pi",
    "tan_pi",
    "sin_degrees",
    "cos_degrees",
    "tan_degrees",
    "xlogy",
    "xlog1py",
    "powm1",
    "sin_cos_pi",
    "sin_cos_degrees",
];
fn inventory(data: &str, complex: bool) -> Result<(), String> {
    let mut counts = BTreeMap::<(&str, &str, &str), usize>::new();
    let mut seen = BTreeSet::new();
    for line in data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let expected = if complex {
            if f.get(1).is_some_and(|op| op.starts_with("sin_cos")) {
                11
            } else {
                9
            }
        } else {
            6
        };
        if f.len() != expected {
            return Err(format!("invalid row {line}"));
        }
        seen.insert((f[0], f[1]));
        *counts.entry((f[0], f[1], f[expected - 1])).or_default() += 1;
    }
    let ops: &[&str] = if complex { &COMPLEX_OPS } else { &REAL_OPS };
    for width in ["32", "64"] {
        for op in ops {
            if !seen.contains(&(width, *op)) {
                return Err(format!("missing {width}/{op}"));
            }
            let requirements: &[(&str, usize)] = if complex {
                &[("boundary", 114), ("cancellation", 64)]
            } else {
                &[("transition", 84), ("adversarial", 256), ("tiny", 22)]
            };
            for &(stratum, n) in requirements {
                if counts.get(&(width, *op, stratum)).copied() != Some(n) {
                    return Err(format!("missing {width}/{op}/{stratum}"));
                }
            }
            if !complex && *op == "logit" {
                for (stratum, n) in [("center-lattice", 129), ("center-dense", 4097)] {
                    if counts.get(&(width, *op, stratum)).copied() != Some(n) {
                        return Err(format!("missing {width}/{op}/{stratum}"));
                    }
                }
            }
            if ["xlogy", "xlog1py", "powm1"].contains(op)
                && !counts.contains_key(&(width, *op, "joint"))
            {
                return Err(format!("missing joint {width}/{op}"));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Observation {
    count: usize,
    maximum: f64,
    worst: String,
    component_ulp: f64,
    component_worst: String,
}
fn ulp<S: RealScalar>(expected: S) -> f64 {
    let a = expected.abs();
    let next = a.next_after(S::INFINITY);
    if next.is_finite() {
        (next - a).into_f64()
    } else {
        (a - a.next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)).into_f64()
    }
}
fn scalar<S: RealScalar>(op: &str, x: S, y: S) -> Result<S, SciOxideError> {
    match op {
        "softplus" => softplus(x),
        "sigmoid" => sigmoid(x),
        "log_sigmoid" => log_sigmoid(x),
        "logit" => logit(x),
        "cosm1" => cosm1(x),
        "sinc" => sinc(x),
        "sincpi" => sincpi(x),
        "exprel" => exprel(x),
        "sin_pi" => sin_pi(x),
        "cos_pi" => cos_pi(x),
        "tan_pi" => tan_pi(x),
        "sin_degrees" => sin_degrees(x),
        "cos_degrees" => cos_degrees(x),
        "tan_degrees" => tan_degrees(x),
        "xlogy" => xlogy(x, y),
        "xlog1py" => xlog1py(x, y),
        "powm1" => powm1(x, y),
        _ => panic!("unknown operation {op}"),
    }
}
fn complex<S: RealScalar>(
    op: &str,
    z: Complex<S>,
    w: Complex<S>,
) -> Result<(Complex<S>, Option<Complex<S>>), SciOxideError> {
    if op == "sin_cos_pi" {
        return sin_cos_pi(z).map(|(s, c)| (s, Some(c)));
    }
    if op == "sin_cos_degrees" {
        return sin_cos_degrees(z).map(|(s, c)| (s, Some(c)));
    }
    let r = match op {
        "softplus" => softplus(z),
        "sigmoid" => sigmoid(z),
        "log_sigmoid" => log_sigmoid(z),
        "logit" => logit(z),
        "cosm1" => cosm1(z),
        "sinc" => sinc(z),
        "sincpi" => sincpi(z),
        "exprel" => exprel(z),
        "sin_pi" => sin_pi(z),
        "cos_pi" => cos_pi(z),
        "tan_pi" => tan_pi(z),
        "sin_degrees" => sin_degrees(z),
        "cos_degrees" => cos_degrees(z),
        "tan_degrees" => tan_degrees(z),
        "xlogy" => xlogy(z, w),
        "xlog1py" => xlog1py(z, w),
        "powm1" => powm1(z, w),
        _ => panic!("unknown complex operation {op}"),
    }?;
    Ok((r, None))
}
fn decode<S: RealScalar>(bits: &str) -> S {
    let b = u64::from_str_radix(bits, 16).unwrap();
    S::try_from_f64_lossy(if S::MANTISSA_DIGITS == 24 {
        f32::from_bits(b as u32) as f64
    } else {
        f64::from_bits(b)
    })
    .unwrap()
}
fn real_cases<S: RealScalar>(data: &str) -> usize {
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let mut reports = BTreeMap::<String, Observation>::new();
    let mut failures = 0;
    for line in data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let r = scalar(f[1], decode::<S>(f[2]), decode::<S>(f[3]));
        let o = reports.entry(f[1].to_owned()).or_default();
        o.count += 1;
        if f[4] == "domain" || f[4] == "overflow" {
            if r.is_ok()
                || (f[4] == "overflow"
                    && r.as_ref().err().is_some_and(|e| {
                        !matches!(
                            e.view(),
                            scioxide_core::SciOxideErrorView::NumericalInstability {
                                issue: scioxide_core::NumericalIssue::Overflow,
                                ..
                            }
                        )
                    }))
                || (f[4] == "domain"
                    && r.as_ref()
                        .err()
                        .is_some_and(|e| e.kind() != SciOxideErrorKind::DomainError))
            {
                failures += 1;
                if failures <= 30 {
                    eprintln!("expected {}: {line}, got {r:?}", f[4]);
                }
            }
            continue;
        }
        let expected = decode::<S>(f[4]);
        let error = match r {
            Ok(a) => (a.into_f64() - expected.into_f64()).abs() / ulp(expected),
            Err(_) => f64::INFINITY,
        };
        if error > o.maximum || o.worst.is_empty() {
            o.maximum = error;
            o.worst = line.to_owned();
        }
        if error > 16.0 {
            failures += 1;
            if failures <= 30 {
                eprintln!("real error {error} ULP: {line}, got {r:?}");
            }
        }
    }
    for (op, o) in reports {
        println!(
            "real/{width}/{op}: n={} max_ulp={} worst={}",
            o.count, o.maximum, o.worst
        );
    }
    failures
}
fn complex_cases<S: RealScalar>(data: &str) -> usize {
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let mut reports = BTreeMap::<String, Observation>::new();
    let mut failures = 0;
    for line in data
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let z = Complex::new(decode::<S>(f[2]), decode::<S>(f[3]));
        let w = Complex::new(decode::<S>(f[4]), decode::<S>(f[5]));
        let r = complex(f[1], z, w);
        let o = reports.entry(f[1].to_owned()).or_default();
        o.count += 1;
        if f[6] == "domain"
            || f[6] == "overflow"
            || (f.len() > 10 && (f[8] == "domain" || f[8] == "overflow"))
        {
            if r.is_ok() {
                failures += 1;
                if failures <= 30 {
                    eprintln!("expected complex error: {line}, got {r:?}");
                }
            }
            continue;
        }
        let growth = match f[1] {
            "sinc" | "cosm1" => 1.0 + z.im.into_f64().abs(),
            "sincpi" | "sin_pi" | "cos_pi" | "tan_pi" | "sin_cos_pi" => {
                1.0 + core::f64::consts::PI * z.im.into_f64().abs()
            }
            "sin_degrees" | "cos_degrees" | "tan_degrees" | "sin_cos_degrees" => {
                1.0 + (core::f64::consts::PI / 180.0) * z.im.into_f64().abs()
            }
            "exprel" => 1.0 + z.re.into_f64().abs(),
            "powm1" => 1.0 + w.norm().into_f64() * z.ln().norm().into_f64(),
            _ => 1.0,
        };
        let mut error = 0.0_f64;
        if let Ok((a, pair)) = r {
            for (index, actual) in [(6, Some(a)), (8, pair)] {
                if let Some(actual) = actual {
                    let re = decode::<S>(f[index]);
                    let im = decode::<S>(f[index + 1]);
                    let magnitude = re.into_f64().abs().max(im.into_f64().abs());
                    let scale = 32.0 * S::EPSILON.into_f64() * growth;
                    let tiny = S::ZERO.next_after(S::ONE).into_f64();
                    // Divide before multiplying: computing the bound can overflow near MAX.
                    for (a, b) in [(actual.re, re), (actual.im, im)] {
                        let diff = (a.into_f64() - b.into_f64()).abs();
                        let normalized = if magnitude > 0.0 {
                            (diff / magnitude) / (scale + 8.0 * (tiny / magnitude))
                        } else {
                            (diff / tiny) / 8.0
                        };
                        error = error.max(normalized);
                        let cu = diff / ulp(b);
                        if cu > o.component_ulp {
                            o.component_ulp = cu;
                            o.component_worst = line.to_owned();
                        }
                    }
                }
            }
        } else {
            error = f64::INFINITY;
        }
        if error > o.maximum || o.worst.is_empty() {
            o.maximum = error;
            o.worst = line.to_owned();
        }
        if error > 1.0 {
            failures += 1;
            if failures <= 30 {
                eprintln!("complex error {error}: {line}, got {r:?}");
            }
        }
    }
    for (op, o) in reports {
        println!(
            "complex/{width}/{op}: n={} max_normalized={} worst={} component_ulp={} component_worst={}",
            o.count, o.maximum, o.worst, o.component_ulp, o.component_worst
        );
    }
    failures
}
#[test]
fn real_boundary_envelopes() {
    let data = include_str!("data/stable_boundaries.txt");
    inventory(data, false).unwrap();
    let failures = real_cases::<f32>(data) + real_cases::<f64>(data);
    assert_eq!(failures, 0, "real boundary failures");
}
#[test]
fn complex_boundary_envelopes() {
    let data = include_str!("data/stable_complex_boundaries.txt");
    inventory(data, true).unwrap();
    let failures = complex_cases::<f32>(data) + complex_cases::<f64>(data);
    assert_eq!(failures, 0, "complex boundary failures");
}
#[test]
fn degree_subnormal_regressions() {
    for (input, expected) in [(90, 2), (1000, 0x11)] {
        for sign in [0, 1_u64 << 63] {
            let x = f64::from_bits(input | sign);
            let bits = expected | sign;
            assert_eq!(sin_degrees(x).unwrap().to_bits(), bits);
            assert_eq!(tan_degrees(x).unwrap().to_bits(), bits);
            assert_eq!(sin_cos_degrees(x).unwrap().0.to_bits(), bits);
            assert_eq!(
                sin_degrees(Complex::new(x, 0.0)).unwrap().re.to_bits(),
                bits
            );
        }
    }
}

#[test]
fn fixtures_reject_perturbed_outputs_and_missing_strata() {
    let data = include_str!("data/stable_boundaries.txt");
    let filtered = data
        .lines()
        .filter(|line| !line.ends_with("center-lattice"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(inventory(&filtered, false).is_err());
    let complex = include_str!("data/stable_complex_boundaries.txt");
    let filtered = complex
        .lines()
        .filter(|line| !line.ends_with("cancellation"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(inventory(&filtered, true).is_err());
    assert_eq!(
        real_cases::<f64>("64 logit 3fe0000000000000 0 3ff0000000000000 negative"),
        1
    );
    assert_eq!(
        complex_cases::<f64>("64 sigmoid 0 0 0 0 3ff0000000000000 0 negative"),
        1
    );
}
#[test]
fn scalar_transforms_execute_on_success_and_failure() {
    fn check<S: RealScalar>() {
        let x = S::try_from_f64_lossy(0.25).unwrap();
        let y = S::try_from_f64_lossy(0.75).unwrap();
        let z = Complex::new(x, x);
        let w = Complex::new(y, -x);
        {
            for op in REAL_OPS {
                std::hint::black_box(scalar(op, std::hint::black_box(x), y)).unwrap();
                assert!(std::hint::black_box(scalar(op, S::NAN, y)).is_err());
            }
            for op in COMPLEX_OPS {
                std::hint::black_box(complex(op, std::hint::black_box(z), w)).unwrap();
                assert!(std::hint::black_box(complex(op, Complex::new(S::NAN, x), w)).is_err());
            }
            assert!(logit(S::ZERO).is_err());
            assert!(tan_pi(S::try_from_f64_lossy(0.5).unwrap()).is_err());
        };
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn exact_zero_and_phase_bits() {
    fn check<S: RealScalar>() {
        assert_eq!(
            logit(S::try_from_f64_lossy(0.5).unwrap())
                .unwrap()
                .decompose(),
            S::ZERO.decompose()
        );
        let negative_zero = -S::ZERO;
        for actual in [
            sin_pi(negative_zero).unwrap(),
            sin_degrees(negative_zero).unwrap(),
            tan_pi(negative_zero).unwrap(),
            tan_degrees(negative_zero).unwrap(),
            xlogy(negative_zero, -S::ONE).unwrap(),
            xlog1py(negative_zero, -S::ONE).unwrap(),
        ] {
            assert_eq!(actual.decompose(), negative_zero.decompose());
        }
        assert!(
            cos_pi(S::try_from_f64_lossy(0.5).unwrap())
                .unwrap()
                .is_zero()
        );
        assert!(
            !cos_pi(S::try_from_f64_lossy(0.5).unwrap())
                .unwrap()
                .is_sign_negative()
        );
        assert!(tan_degrees(S::try_from_f64_lossy(90.0).unwrap()).is_err());
        assert_eq!(sin_pi(S::MAX).unwrap().decompose(), S::ZERO.decompose());
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn complex_logistic_real_axis_retains_real_envelope() {
    fn check<S: RealScalar>() {
        let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
        for line in include_str!("data/stable_boundaries.txt")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let f: Vec<_> = line.split_whitespace().collect();
            if f[0] != width
                || !["sigmoid", "softplus", "log_sigmoid", "logit"].contains(&f[1])
                || ["domain", "overflow"].contains(&f[4])
            {
                continue;
            }
            let x = decode::<S>(f[2]);
            let expected = decode::<S>(f[4]);
            for im in [S::ZERO, -S::ZERO] {
                let (actual, _) =
                    complex(f[1], Complex::new(x, im), Complex::new(S::ZERO, S::ZERO)).unwrap();
                let error = (actual.re.into_f64() - expected.into_f64()).abs() / ulp(expected);
                assert!(
                    error <= 16.0,
                    "complex real-axis {line}: {actual:?}, {error} ULP"
                );
                assert!(actual.im.is_zero());
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
