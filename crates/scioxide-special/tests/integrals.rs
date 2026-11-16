#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_special::*;

fn evaluate<S: RealScalar>(
    name: &str,
    n: u32,
    x: S,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    let o = IntegralOptions::default();
    match name {
        "expint_e1" => expint_e1_with_options(x, o),
        "expint_en" => expint_en_with_options(n, x, o),
        "expint_en_scaled" => expint_en_scaled_with_options(n, x, o),
        "expint_ei" => expint_ei_with_options(x, o),
        "expint_ei_scaled" => expint_ei_scaled_with_options(x, o),
        "si" => si_with_options(x, o),
        "ci" => ci_with_options(x, o),
        "shi" => shi_with_options(x, o),
        "chi" => chi_with_options(x, o),
        "shi_scaled" => shi_scaled_with_options(x, o),
        "chi_scaled" => chi_scaled_with_options(x, o),
        "dawson" => dawson_with_options(
            x,
            scioxide_special::ErrorFunctionOptions::new((o).iteration_budget()),
        ),
        "fresnel_c" => fresnel_c_with_options(x, o),
        "fresnel_s" => fresnel_s_with_options(x, o),
        "atanint" => atanint_with_options(x, o),
        "expint_3" => expint_3_with_options(x, o),
        "logarithmic_integral" => logarithmic_integral_with_options(x, o),
        "logarithmic_integral_offset" => logarithmic_integral_offset_with_options(x, o),
        _ => unreachable!("unknown fixture {name}"),
    }
}

fn check_oracle<S: RealScalar>(
    name: &str,
    n: u32,
    x: S,
    y: S,
    derivative: S,
    logarithm: S,
) -> Result<(), String> {
    let r = evaluate(name, n, x).map_err(|e| format!("{e}"))?;
    if !r.termination().is_success() {
        return Err(format!("{:?}", r.termination()));
    }
    let f = |v: S| v.try_to_f64_exact().unwrap();
    let (rel, abs, ulps) = if S::MANTISSA_DIGITS == 24 {
        (2e-4_f64, 5e-6_f64, 16.)
    } else {
        (5e-11_f64, 5e-13_f64, 8.)
    };
    if y.is_finite() {
        let v = r.value().ok_or("missing ordinary value")?;
        let error = (f(*v.value()) - f(y)).abs();
        if name == "expint_3" && error > 8.0 * f(y.next_after(S::INFINITY) - y) {
            return Err(format!(
                "expint_3 value exceeds eight ULPs: {} expected {}",
                f(*v.value()),
                f(y)
            ));
        }
        if error > abs.max(rel * f(y).abs()) {
            return Err(format!("value {} expected {}", f(*v.value()), f(y)));
        }
        if error > f(v.absolute_error()) + f(S::EPSILON) * f(y).abs() {
            return Err(format!(
                "value error {error} exceeds estimate {}",
                f(v.absolute_error())
            ));
        }
        if y.abs() < S::MIN_POSITIVE {
            let floor = f(S::ZERO.next_after(S::ONE));
            if error > ulps * floor || v.value().is_sign_negative() != y.is_sign_negative() {
                return Err(format!("subnormal {} expected {}", f(*v.value()), f(y)));
            }
        }
    } else if r.scalar_representation() != ScalarRepresentation::Overflow {
        return Err("expected overflow status".into());
    }
    if derivative.is_finite() {
        let d = r
            .derivative()
            .ok_or_else(|| format!("missing derivative, expected {}", f(derivative)))?;
        let error = (f(*d.value()) - f(derivative)).abs();
        if error > abs.max(rel * f(derivative).abs()) {
            return Err(format!(
                "derivative {} expected {}",
                f(*d.value()),
                f(derivative)
            ));
        }
        if error > f(d.absolute_error()) + f(S::EPSILON) * f(derivative).abs() {
            return Err(format!(
                "derivative error {error} exceeds estimate {}",
                f(d.absolute_error())
            ));
        }
    }
    if let Some(log) = r.logarithmic()
        && let Some(actual) = log.value().log_magnitude()
    {
        let error = (f(actual) - f(logarithm)).abs();
        if (r
            .value()
            .is_some_and(|v| f(v.absolute_error()) < rel * f(*v.value()).abs())
            && error > abs.max(rel * f(logarithm).abs()))
            || error > f(log.absolute_error()) + f(S::EPSILON) * f(logarithm).abs()
        {
            return Err(format!(
                "log {} expected {} (estimate {})",
                f(actual),
                f(logarithm),
                f(log.absolute_error())
            ));
        }
    }
    Ok(())
}
#[test]
fn independent_oracles() {
    let mut failures = Vec::new();
    for line in include_str!("data/integrals.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        let n = p[2].parse().unwrap();
        let b: Vec<_> = p[3..7]
            .iter()
            .map(|v| u64::from_str_radix(v, 16).unwrap())
            .collect();
        let result = if p[1] == "64" {
            check_oracle(
                p[0],
                n,
                f64::from_bits(b[0]),
                f64::from_bits(b[1]),
                f64::from_bits(b[2]),
                f64::from_bits(b[3]),
            )
        } else {
            check_oracle(
                p[0],
                n,
                f32::from_bits(b[0] as u32),
                f32::from_bits(b[1] as u32),
                f32::from_bits(b[2] as u32),
                f32::from_bits(b[3] as u32),
            )
        };
        if let Err(e) = result {
            failures.push(format!("{} {} order {n} input {}: {e}", p[0], p[1], p[3]));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn domains_and_budgets() {
    let exact_options = IntegralOptions::new(Budget::ONE);
    let exact = si_with_options(0.0, exact_options).unwrap();
    assert!(exact.termination().is_success());
    assert_eq!(exact.value().unwrap().absolute_error(), 0.0);
    let partial = expint_en_with_options(32, 0.5_f64, exact_options).unwrap();
    let estimate = partial.value().unwrap();
    let lower = (-1.0_f64).exp() * (1.0 - 2.0_f64.powi(-31)) / 31.0;
    assert!(*estimate.value() + estimate.absolute_error() >= lower);
    assert!(partial.derivative().is_none());
    assert!(ci(-1.0_f64).is_err());
    assert!(expint_en(1, 0.0_f64).is_err());
    assert_eq!(si(-0.0_f64).unwrap().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(expint_en(2, 0.0_f64).unwrap(), 1.0);
    let result = expint_en_with_options(1, 2.0_f64, exact_options).unwrap();
    assert_eq!(
        result.termination(),
        SpecialFunctionTermination::IterationLimit
    );
}

#[test]
fn full_range_representations() {
    for x in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-160,
        1e-100,
        1e100,
        f64::MAX,
    ] {
        for name in [
            "expint_en_scaled",
            "expint_ei_scaled",
            "si",
            "ci",
            "shi_scaled",
            "chi_scaled",
            "dawson",
            "fresnel_c",
            "fresnel_s",
            "atanint",
            "expint_3",
            "logarithmic_integral",
        ] {
            let r = evaluate(name, 1, x).unwrap_or_else(|e| panic!("{name}({x}): {e}"));
            assert!(r.termination().is_success(), "{name}({x}): {r:?}");
        }
    }
    for x in [f32::from_bits(1), f32::MIN_POSITIVE, 1e-20, 1e20, f32::MAX] {
        for name in [
            "expint_en_scaled",
            "expint_ei_scaled",
            "si",
            "ci",
            "shi_scaled",
            "chi_scaled",
            "dawson",
            "fresnel_c",
            "fresnel_s",
            "atanint",
            "expint_3",
            "logarithmic_integral",
        ] {
            let r = evaluate(name, 1, x).unwrap_or_else(|e| panic!("{name}({x}): {e}"));
            assert!(r.termination().is_success(), "{name}({x}): {r:?}");
        }
    }
    let r = expint_ei_with_options(1000.0, IntegralOptions::default()).unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.logarithmic().is_some());
    assert!(r.scaled().is_some());
    let r = expint_en_with_options(1, 1000.0, IntegralOptions::default()).unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Finite);
    assert!(r.logarithmic().is_some());
    assert!(r.scaled().is_some());
}

#[test]
fn identities_continuity_and_allocations() {
    let names = [
        "expint_en",
        "expint_en_scaled",
        "expint_ei",
        "expint_ei_scaled",
        "si",
        "ci",
        "shi",
        "chi",
        "shi_scaled",
        "chi_scaled",
        "dawson",
        "fresnel_c",
        "fresnel_s",
        "atanint",
        "expint_3",
        "logarithmic_integral",
        "logarithmic_integral_offset",
    ];
    for name in names {
        for x in [0.1_f64, 0.5, 1.5, 2.5, 4.5, 8.5, 32.5, 64.] {
            let r = evaluate(name, 3, x).unwrap();
            assert!(r.termination().is_success());
            let h = 1e-5 / x.max(1.);
            let lo = evaluate(name, 3, x - h).unwrap().try_value().unwrap();
            let hi = evaluate(name, 3, x + h).unwrap().try_value().unwrap();
            let actual = (hi - lo) / (2. * h);
            let derivative = *r.derivative().unwrap().value();
            assert!(
                (actual - derivative).abs() < 3e-7 * (1. + derivative.abs()),
                "{name} {x}: {actual} {derivative}"
            );
            let _ = evaluate(name, 3, x as f32).unwrap();
        }
        for x in [
            0.5_f64, 1., 1.5, 2., 2.5, 3., 4., 6., 8., 10., 12., 20., 32., 40., 48.,
        ] {
            if x == 1. && name.starts_with("logarithmic") {
                continue;
            }
            let lo = evaluate(name, 3, x.next_after(0.))
                .unwrap()
                .try_value()
                .unwrap();
            let hi = evaluate(name, 3, x.next_after(f64::INFINITY))
                .unwrap()
                .try_value()
                .unwrap();
            assert!(
                (lo - hi).abs() <= 2e-11 * (1. + hi.abs()),
                "{name} at {x}: {lo} {hi}"
            );
        }
        let _ = evaluate(name, 3, f64::NAN).unwrap_err();
    }
    for x in [0.01_f64, 0.5, 2., 10., 100.] {
        for n in [1, 2, 3, 10, 100] {
            let left = f64::from(n) * expint_en_scaled(n + 1, x).unwrap()
                + x * expint_en_scaled(n, x).unwrap();
            assert!(
                (left - 1.).abs() < 1e-12,
                "En recurrence n={n} x={x}: {left}"
            );
        }
        assert!((expint_e1(x).unwrap() + expint_ei(-x).unwrap()).abs() < 1e-13);
        assert!(
            (expint_e1(-x).unwrap() + expint_ei(x).unwrap()).abs()
                < 1e-13 * expint_ei(x).unwrap().abs()
        );
        assert!(
            (shi(x).unwrap() + chi(x).unwrap() - expint_ei(x).unwrap()).abs()
                < 1e-12 * expint_ei(x).unwrap().abs().max(1.)
        );
        for name in [
            "si",
            "shi",
            "shi_scaled",
            "dawson",
            "fresnel_c",
            "fresnel_s",
            "atanint",
        ] {
            assert_eq!(
                evaluate(name, 0, -x).unwrap().try_value().unwrap(),
                -evaluate(name, 0, x).unwrap().try_value().unwrap()
            );
        }
    }
}

#[test]
fn subnormal_error_representations() {
    let x = 1.5e-15_f32;
    let r = fresnel_s_with_options(x, IntegralOptions::default()).unwrap();
    // The first omitted normalized Fresnel term is O(x^7), far below
    // binary64 rounding here; this independently resolves binary32 underflow.
    let reference = core::f64::consts::PI / 6.0 * f64::from(x).powi(3);
    let value = r.value().unwrap();
    let observed = (f64::from(*value.value()) - reference).abs() / reference.abs();
    assert!(f64::from(value.relative_error().unwrap()) >= observed);
    assert!(f64::from(r.scaled().unwrap().relative_error().unwrap()) >= observed);
    // A bound spanning zero has no meaningful logarithmic error estimate.
    assert!(value.relative_error().unwrap() >= 0.5);
    assert!(r.logarithmic().is_none());
    let zero = expint_en_with_options(1, 1000.0_f64, IntegralOptions::default()).unwrap();
    assert_eq!(*zero.value().unwrap().value(), 0.0);
    assert!(zero.value().unwrap().relative_error().is_none());
}

/// Tight ordinary-input qualification, with absolute bounds at Ci zeros.
#[test]
fn affected_integrals_meet_eight_ulp_target() {
    let mut failures = Vec::new();
    let mut worst = std::collections::BTreeMap::<&str, f64>::new();
    for line in include_str!("data/integrals.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        if p[1] != "64"
            || !matches!(
                p[0],
                "expint_e1"
                    | "expint_ei"
                    | "expint_ei_scaled"
                    | "expint_en"
                    | "expint_en_scaled"
                    | "expint_3"
                    | "si"
                    | "ci"
            )
        {
            continue;
        }
        let bits = |i: usize| u64::from_str_radix(p[i], 16).unwrap();
        let (x, expected) = (f64::from_bits(bits(3)), f64::from_bits(bits(4)));
        if !expected.is_finite() || expected == 0.0 {
            continue;
        }
        let result = evaluate(p[0], p[2].parse().unwrap(), x).unwrap();
        let actual = *result.value().unwrap().value();
        let spacing = f64::from_bits(expected.abs().to_bits() + 1) - expected.abs();
        let offset = p.get(7).map_or(0.0, |s| s.parse::<f64>().unwrap());
        let error = ((actual - expected) / spacing - offset).abs();
        if p[0] == "ci" && expected.abs() < 0.125 / (1.0 + x) {
            continue;
        }
        worst
            .entry(p[0])
            .and_modify(|v| *v = v.max(error))
            .or_insert(error);
        if error > 8.0 {
            failures.push(format!("{}({x},n={}) error={error:.3} ULP", p[0], p[2]));
        }
    }
    eprintln!("qualified maxima: {worst:?}");
    assert!(
        failures.is_empty(),
        "{} failures: {}",
        failures.len(),
        failures.into_iter().take(30).collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn qualified_bounds_and_interrupted_approximations() {
    for (name, n, x) in [
        ("expint_e1", 0, 1.358),
        ("expint_ei", 0, -1.63),
        ("expint_en", 2, 1.255),
        ("si", 0, 33.2),
        ("ci", 0, 33.2),
    ] {
        let result = evaluate(name, n, x).unwrap();
        assert_eq!(
            result.value().unwrap().reliability(),
            scioxide_core::EstimateReliability::StatisticallyCalibrated
        );
    }
    for result in [
        expint_e1_with_options(2.0, IntegralOptions::new(Budget::ONE)),
        expint_ei_with_options(3.0, IntegralOptions::new(Budget::ONE)),
        ci_with_options(33.2, IntegralOptions::new(Budget::ONE)),
        si_with_options(33.2, IntegralOptions::new(Budget::ONE)),
    ] {
        let result = result.unwrap();
        assert_eq!(
            result.termination(),
            SpecialFunctionTermination::IterationLimit
        );
        assert_eq!(
            result.value().unwrap().reliability(),
            scioxide_core::EstimateReliability::ModelBased
        );
    }
}

#[test]
fn integral_poles_and_exhausted_projection_execute() {
    assert!(expint_e1(0.0_f64).is_err());
    assert!(
        fresnel_c_with_options(
            1.25_f64,
            IntegralOptions::default().with_iteration_budget(scioxide_core::Budget::ONE),
        )
        .and_then(|report| report.try_value())
        .is_err()
    );
}
