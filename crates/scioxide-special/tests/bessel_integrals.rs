#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;
fn evaluate<S: RealScalar>(
    name: &str,
    x: S,
    o: BesselIntegralOptions,
) -> Result<SpecialFunctionResult<S>, scioxide_core::SciOxideError> {
    match name {
        "jint" => bessel_j0_integral_with_options(x, o),
        "yint" => bessel_y0_integral_with_options(x, o),
        "iint" => bessel_i0_integral_with_options(x, o),
        "kint" => bessel_k0_integral_with_options(x, o),
        "jcompl" => bessel_j0_complement_integral_with_options(x, o),
        "ytail" => bessel_y0_tail_integral_with_options(x, o),
        "iremainder" => bessel_i0_remainder_integral_with_options(x, o),
        "ktail" => bessel_k0_tail_integral_with_options(x, o),
        "hint" => struve_h0_integral_with_options(x, o),
        "htail" => struve_h0_tail_integral_with_options(x, o),
        _ => struve_l0_integral_with_options(x, o),
    }
}
fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("bessel_integrals", width);
    for row in include_str!("data/bessel_integrals.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = row.split_whitespace().collect();
        if p[1] != width {
            continue;
        }
        let x = decode(p[2]);
        let reference = decode(p[3]);
        let derivative = decode(p[4]);
        let r = evaluate(p[0], x, BesselIntegralOptions::default())
            .unwrap_or_else(|e| panic!("{row}: {e}"));
        let value = r.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
        let d = *r.derivative().unwrap().value();
        checker.scalar(
            &format!("{}:value", p[0]),
            value,
            reference,
            Some(r.value().unwrap().absolute_error()),
            row,
        );
        checker.scalar(
            &format!("{}:derivative", p[0]),
            d,
            derivative,
            Some(r.derivative().unwrap().absolute_error()),
            row,
        );
        // The reported errors cover the rounded references.
        let half_spacing = |reference: S| S::EPSILON * reference.abs() + S::ZERO.next_after(S::ONE);
        let value_error = r.value().unwrap().absolute_error();
        assert!(
            (value - reference).abs() <= value_error + half_spacing(reference),
            "{row}: value error {value_error:?}"
        );
        let derivative_error = r.derivative().unwrap().absolute_error();
        assert!(
            (d - derivative).abs() <= derivative_error + half_spacing(derivative),
            "{row}: derivative error {derivative_error:?}"
        );
        if reference != S::ZERO && reference.abs() < S::MIN_POSITIVE {
            assert!(
                (value - reference).abs()
                    <= S::try_from_f64_lossy(if width == "32" { 16.0 } else { 8.0 }).unwrap()
                        * S::ZERO.next_after(S::ONE),
                "{row}: subnormal accuracy"
            );
        }
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
fn domains_symmetry_and_work() {
    assert_eq!(bessel_j0_integral(0.0_f64).unwrap(), 0.0);
    assert_eq!(bessel_y0_integral(0.0_f64).unwrap(), 0.0);
    assert!(
        (struve_h0_tail_integral(0.0_f64).unwrap() - core::f64::consts::FRAC_PI_2).abs() < 1e-15
    );
    assert_eq!(
        bessel_j0_integral(-2.0_f64).unwrap(),
        -bessel_j0_integral(2.0).unwrap()
    );
    assert_eq!(
        struve_h0_integral(-2.0_f64).unwrap(),
        struve_h0_integral(2.0).unwrap()
    );
    assert!(
        (struve_h0_tail_integral(-2.0_f64).unwrap() + struve_h0_tail_integral(2.0).unwrap()
            - core::f64::consts::PI)
            .abs()
            < 1e-14
    );
    assert!(bessel_k0_tail_integral(0.0_f64).is_err());
    assert!(bessel_y0_integral(-1.0_f64).is_err());
    let r = bessel_j0_integral_with_options(
        20.0_f64,
        BesselIntegralOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&r);
}
#[test]
fn weighted_integral_and_log_k() {
    assert_eq!(
        bessel_weighted_integral(0.0_f64, 2.0, 0.0).unwrap(),
        1.0 / 3.0
    );
    for a in [0.5_f64, 2.0, 8.0] {
        let expected = bessel_j0_integral(2.0 * a).unwrap() / (2.0 * a);
        let actual = bessel_weighted_integral(a, 0.0, 0.0).unwrap();
        assert!((actual - expected).abs() < 1e-10);
    }
    assert!(bessel_weighted_integral(1.0_f64, -2.0, 0.0).is_err());
    assert!(
        (bessel_k_log(0.5_f64, 1000.0).unwrap()
            - (-1000.0 + 0.5 * (core::f64::consts::PI / 2000.0).ln()))
        .abs()
            < 1e-10
    );
    assert!(bessel_k_log(100.0_f64, 1e-100).unwrap().is_finite());
}

#[test]
fn independent_weighted_hypergeometric_fixtures() {
    let mut checker64 = fixture::Checker::new("weighted_bessel", "64");
    let mut checker32 = fixture::Checker::new("weighted_bessel", "32");
    for row in include_str!("data/weighted_bessel.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = row.split_whitespace().collect();
        if p[0] == "64" {
            let decode = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
            let options = BesselIntegralOptions::default()
                .with_iteration_budget(Budget::try_from(1_000_000).unwrap());
            let report = bessel_weighted_integral_with_options(
                decode(p[1]),
                decode(p[2]),
                decode(p[3]),
                options,
            )
            .unwrap_or_else(|e| panic!("{row}: {e}"));
            let actual = report.try_value().unwrap();
            let reference = decode(p[4]);
            checker64.scalar(
                "value",
                actual,
                reference,
                Some(report.value().unwrap().absolute_error()),
                row,
            );
        } else {
            let decode = |s: &str| f32::from_bits(u32::from_str_radix(s, 16).unwrap());
            let options = BesselIntegralOptions::default()
                .with_iteration_budget(Budget::try_from(1_000_000).unwrap());
            let report = bessel_weighted_integral_with_options(
                decode(p[1]),
                decode(p[2]),
                decode(p[3]),
                options,
            )
            .unwrap_or_else(|e| panic!("{row}: {e}"));
            let actual = report.try_value().unwrap();
            let reference = decode(p[4]);
            checker32.scalar(
                "value",
                actual,
                reference,
                Some(report.value().unwrap().absolute_error()),
                row,
            );
        }
    }
}

#[test]
fn weighted_quadrature_completes_with_the_default_budget() {
    // mpmath 1.3.0, 70 digits, exact binary64 arguments; integrated 1F2 series.
    for (a, exponent, order, reference) in [
        (
            4.110046448025208_f64,
            1.3611827977010207,
            2.248153721199336,
            -0.012689757578138633,
        ),
        (5.0, 3.0, 3.0, -0.022347115254249804),
        (8.0, 1.25, 0.5, 0.012327788687103495),
        (2.0, -0.9, 0.0, 8.827847499539036),
    ] {
        let value = bessel_weighted_integral(a, exponent, order).unwrap();
        assert!(
            (value - reference).abs() <= 5e-12_f64.max(5e-10 * reference.abs()),
            "{a}, {exponent}, {order}: {value}"
        );
        let value = bessel_weighted_integral(a as f32, exponent as f32, order as f32).unwrap();
        assert!((f64::from(value) - reference).abs() <= 2e-6 * reference.abs().max(1.0));
    }
    let report = bessel_weighted_integral_with_options(
        4.110046448025208,
        1.3611827977010207,
        2.248153721199336,
        BesselIntegralOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&report);
}
