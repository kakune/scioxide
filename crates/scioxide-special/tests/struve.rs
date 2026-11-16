#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;
fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("struve", width);
    let o = StruveOptions::default();
    for line in include_str!("data/struve.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        if p[1] != width {
            continue;
        }
        let n = decode(p[2]);
        let x = decode(p[3]);
        let expected = decode(p[4]);
        let derivative = decode(p[5]);
        let r = if p[0] == "h" {
            struve_with_options(n, x, o)
        } else {
            modstruve_with_options(n, x, o)
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        let v = r.try_value().unwrap_or_else(|e| panic!("{line}: {e}"));
        checker.scalar(
            "value",
            v,
            expected,
            Some(r.value().unwrap().absolute_error()),
            line,
        );
        let d = *r.derivative().unwrap().value();
        checker.scalar(
            "derivative",
            d,
            derivative,
            Some(r.derivative().unwrap().absolute_error()),
            line,
        );
        // The reported errors cover the rounded references.
        let rounding = |reference: S| S::EPSILON * reference.abs() + S::ZERO.next_after(S::ONE);
        let reported = r.value().unwrap().absolute_error();
        assert!(
            (v - expected).abs() <= reported + rounding(expected),
            "{line}: value error {reported:?}"
        );
        let reported = r.derivative().unwrap().absolute_error();
        assert!(
            (d - derivative).abs() <= reported + rounding(derivative),
            "{line}: derivative error {reported:?}"
        );
        assert!(r.counts().iterations() <= o.iteration_budget().get());
    }
}
#[test]
fn independent_f64() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn independent_f32() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn domains_and_budgets() {
    assert_eq!(struve(0.0_f64, 0.0).unwrap(), 0.0);
    assert!((struve(-1.0_f64, 0.0).unwrap() - 2.0 / core::f64::consts::PI).abs() < 1e-15);
    assert_eq!(struve(-1.5_f64, 0.0).unwrap(), 0.0);
    assert!(struve(-2.0_f64, 0.0).is_err());
    assert!(struve(0.5_f64, -1.0).is_err());
    assert_eq!(
        struve(0.0_f64, -2.0).unwrap(),
        -struve(0.0_f64, 2.0).unwrap()
    );
    let r = struve_with_options(
        0.0_f64,
        2.0,
        StruveOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
    assert!(
        struve_with_options(0.0_f64, 2.0, StruveOptions::default())
            .unwrap()
            .termination()
            .is_success()
    );
    // The large-argument regime also stops on its budget.
    let r = struve_with_options(
        0.0_f64,
        50.0,
        StruveOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
}

#[test]
fn identities_across_regimes() {
    let close = |a: f64, b: f64, ulps: f64| (a - b).abs() <= ulps * f64::EPSILON * b.abs();
    for x in [0.5_f64, 3.0, 24.0, 50.0, 90.0] {
        // H_-1 + H_1 = 2/pi (DLMF 11.4.23 at nu = 0).
        let sum = struve(-1.0_f64, x).unwrap() + struve(1.0, x).unwrap();
        assert!(close(sum, core::f64::consts::FRAC_2_PI, 8.0), "{x}: {sum}");
        // H_1/2 and L_1/2 in closed form; K_1/2 terminates after one term.
        let root = (2.0 / (core::f64::consts::PI * x)).sqrt();
        let h = struve(0.5_f64, x).unwrap();
        assert!(close(h, root * (1.0 - x.cos()), 16.0), "{x}: {h}");
        let l = modstruve(0.5_f64, x).unwrap();
        assert!(close(l, root * (x.cosh() - 1.0), 16.0), "{x}: {l}");
    }
}

/// Monotone integer image of binary64 bits, for ULP distances.
fn ordered(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

/// Seeded mpmath references (`generate_struve.py`) on
/// nu in [-5, 10], x in [1e-3, 100]: within 64 ULP, or near a zero of H_nu an
/// absolute error of at most 4 eps max(1, |H_nu|). Reported errors cover the
/// rounded references.
#[test]
fn seeded_references() {
    let o = StruveOptions::default();
    let decode = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
    let half_unit =
        |reference: f64| 0.5 * (reference.abs().next_up() - reference.abs()).max(f64::from_bits(1));
    let (mut rows, mut misses, mut worst) = ([0_usize; 2], Vec::new(), 0_u64);
    for line in include_str!("data/struve_reference.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        let h = p[0] == "h";
        let (nu, x, expected, slope) = (decode(p[2]), decode(p[3]), decode(p[4]), decode(p[5]));
        let r = if h {
            struve_with_options(nu, x, o)
        } else {
            modstruve_with_options(nu, x, o)
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        let v = r.try_value().unwrap_or_else(|e| panic!("{line}: {e}"));
        let error = (v - expected).abs();
        let ulps = ordered(v).abs_diff(ordered(expected));
        // Near a zero: sensitive in x (power laws give |x H'/H| = |nu+1| <= 11)
        // and small against the oscillation amplitude sqrt(2/(pi x)).
        let amplitude = (2.0 / (core::f64::consts::PI * x)).sqrt();
        let near_zero =
            h && (x * slope).abs() >= 16.0 * expected.abs() && expected.abs() < amplitude / 16.0;
        if near_zero {
            if error > 4.0 * f64::EPSILON * expected.abs().max(1.0) {
                misses.push(format!("{line}: near-zero error {error:e}"));
            }
        } else {
            worst = worst.max(ulps);
            if ulps > 64 {
                misses.push(format!("{line}: {ulps} ULP ({v:e})"));
            }
        }
        let reported = r.value().unwrap().absolute_error();
        if error > reported + half_unit(expected) {
            misses.push(format!(
                "{line}: value error {error:e} > reported {reported:e}"
            ));
        }
        let derivative = r.derivative().unwrap();
        let d_error = (*derivative.value() - slope).abs();
        if d_error > derivative.absolute_error() + half_unit(slope) {
            misses.push(format!(
                "{line}: derivative error {d_error:e} > reported {:e}",
                derivative.absolute_error()
            ));
        }
        rows[usize::from(h)] += 1;
    }
    assert!(rows.iter().all(|&n| n >= 4000), "{rows:?}");
    assert!(
        misses.is_empty(),
        "{} misses (worst {worst} ULP):\n{}",
        misses.len(),
        misses.join("\n")
    );
}

#[test]
#[should_panic(expected = "fixture accuracy")]
fn fixture_checker_rejects_value_regressions_with_matching_error_inflation() {
    let mut checker = fixture::Checker::new("struve", "64");
    checker.scalar(
        "value",
        1.0 + 2048.0 * f64::EPSILON,
        1.0,
        Some(2048.0 * f64::EPSILON),
        "inflated value and error",
    );
}

#[test]
#[should_panic(expected = "reported error")]
fn fixture_checker_caps_reported_error_even_for_an_exact_value() {
    let mut checker = fixture::Checker::new("struve", "64");
    checker.scalar(
        "value",
        1.0,
        1.0,
        Some(4096.0 * f64::EPSILON),
        "inflated error only",
    );
}
