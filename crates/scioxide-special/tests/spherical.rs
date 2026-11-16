#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_special::*;

type Evaluation<S> = SpecialFunctionResult<S>;
fn evaluate<S: RealScalar>(
    kind: &str,
    n: u32,
    x: S,
    options: SphericalBesselOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    match kind {
        "rj" => riccati_jn_with_options(n, x, options),
        "ry" => riccati_yn_with_options(n, x, options),
        "j" => spherical_jn_with_options(n, x, options),
        "y" => spherical_yn_with_options(n, x, options),
        "i" => spherical_in_with_options(n, x, options),
        "k" => spherical_kn_with_options(n, x, options),
        "is" => spherical_in_scaled_with_options(n, x, options),
        "ks" => spherical_kn_scaled_with_options(n, x, options),
        _ => unreachable!(),
    }
}

fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("spherical", width);
    let options = SphericalBesselOptions::default();
    let mut count = 0;
    for row in include_str!("data/spherical.tsv")
        .lines()
        .chain(include_str!("data/riccati.tsv").lines())
        .chain(include_str!("data/spherical_zeros.tsv").lines())
        .filter(|r| !r.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[1] != width {
            continue;
        }
        let n = fields[2].parse().unwrap();
        let x = decode(fields[3]);
        let reference = decode(fields[4]);
        let derivative = decode(fields[5]);
        let result = evaluate(fields[0], n, x, options).unwrap_or_else(|e| panic!("{row}: {e}"));
        let actual = result.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
        checker.scalar(
            "value",
            actual,
            reference,
            Some(result.value().unwrap().absolute_error()),
            row,
        );
        let d = *result
            .derivative()
            .unwrap_or_else(|| panic!("missing derivative: {row}"))
            .value();
        checker.scalar(
            "derivative",
            d,
            derivative,
            Some(result.derivative().unwrap().absolute_error()),
            row,
        );
        // The reported errors cover the rounded references.
        let rounding = |reference: S| S::EPSILON * reference.abs() + S::ZERO.next_after(S::ONE);
        let reported = result.value().unwrap().absolute_error();
        assert!(
            (actual - reference).abs() <= reported + rounding(reference),
            "{row}: value error {reported:?}"
        );
        let reported = result.derivative().unwrap().absolute_error();
        assert!(
            (d - derivative).abs() <= reported + rounding(derivative),
            "{row}: derivative error {reported:?}"
        );
        assert!(result.counts().iterations() <= options.iteration_budget().get());
        if reference != S::ZERO && reference.abs() < S::MIN_POSITIVE {
            assert_eq!(
                actual.is_sign_negative(),
                reference.is_sign_negative(),
                "{row}: subnormal sign"
            );
            let ulp = S::ZERO.next_after(S::ONE);
            assert!(
                (actual - reference).abs()
                    <= S::try_from_f64_lossy(if width == "32" { 16.0 } else { 8.0 }).unwrap() * ulp,
                "{row}: subnormal value"
            );
            assert!(actual != S::ZERO, "{row}: destructive value underflow");
        }
        if derivative != S::ZERO && derivative.abs() < S::MIN_POSITIVE {
            let ulp = S::ZERO.next_after(S::ONE);
            assert_eq!(
                d.is_sign_negative(),
                derivative.is_sign_negative(),
                "{row}: derivative sign"
            );
            assert!(
                (d - derivative).abs()
                    <= S::try_from_f64_lossy(if width == "32" { 16.0 } else { 8.0 }).unwrap() * ulp,
                "{row}: subnormal derivative"
            );
            assert!(d != S::ZERO, "{row}: destructive derivative underflow");
        }
        count += 1;
    }
    assert!(count > 300);
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
fn origins_parity_and_poles() {
    assert_eq!(spherical_jn(0, 0.0_f64).unwrap(), 1.0);
    assert_eq!(spherical_in(0, 0.0_f32).unwrap(), 1.0);
    for n in 0..8 {
        let positive = spherical_jn(n, 0.5_f64).unwrap();
        assert_eq!(
            spherical_jn(n, -0.5_f64).unwrap(),
            if n % 2 == 0 { positive } else { -positive }
        );
    }
    assert!(spherical_jn(1, -0.0_f64).unwrap().is_sign_negative());
    {
        let outcome = &(spherical_yn(0, 0.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(spherical_kn(0, -1.0_f32));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(spherical_jn(0, f64::NAN));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    assert!((spherical_kn_scaled(0, 2.0_f64).unwrap() - std::f64::consts::PI / 4.0).abs() < 1e-15);
}

#[test]
fn exhausted_budget_keeps_a_single_termination_status() {
    let options = SphericalBesselOptions::default().with_iteration_budget(Budget::ONE);
    let result = spherical_jn_with_options(100, 20.0_f64, options).unwrap_err();
    let result_counts = publication::exhausted_counts(&result);
    assert_eq!(result_counts.iterations(), 1);
}
#[test]
fn wronskians() {
    for n in [0, 1, 5, 20, 100] {
        for x in [10.0_f64, 32.0, 100.0, 1000.0] {
            let options = SphericalBesselOptions::default();
            let j = spherical_jn_with_options(n, x, options).unwrap();
            let y = spherical_yn_with_options(n, x, options).unwrap();
            let w = j.try_value().unwrap() * y.derivative().unwrap().value()
                - j.derivative().unwrap().value() * y.try_value().unwrap();
            assert!(
                (w * x * x - 1.0).abs() < 2048.0 * f64::EPSILON,
                "n={n} x={x} w={w}"
            );
        }
    }
}

#[test]
fn derivative_overflow_preserves_finite_value() {
    let options = SphericalBesselOptions::default();
    for result in [
        spherical_yn_with_options(0, 1e-200_f64, options),
        spherical_kn_with_options(0, 1e-200, options),
        spherical_yn_with_options(1, 1e-150, options),
    ] {
        let result = result.unwrap();
        assert!(result.try_value().unwrap().is_finite());
        assert!(result.derivative().is_none());
    }
}

fn check_h3d<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("h3d", width);
    let options = SphericalBesselOptions::default();
    for line in include_str!("data/h3d.tsv")
        .lines()
        .chain(include_str!("data/h3d_zeros.tsv").lines())
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        if p[0] != width {
            continue;
        }
        let n = p[1].parse().unwrap();
        let lambda = decode(p[2]);
        let eta = decode(p[3]);
        let oracle = decode(p[4]);
        let result = legendre_h3d_with_options(n, lambda, eta, options)
            .unwrap_or_else(|e| panic!("{line}: {e}"));
        let actual = result.try_value().unwrap_or_else(|e| panic!("{line}: {e}"));
        checker.scalar(
            "value",
            actual,
            oracle,
            Some(result.value().unwrap().absolute_error()),
            line,
        );
        assert!(result.counts().iterations() <= options.iteration_budget().get());
    }
}
#[test]
fn h3d_independent_f64() {
    check_h3d("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn h3d_independent_f32() {
    check_h3d("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}

#[test]
fn h3d_limits() {
    assert_eq!(legendre_h3d(0, 2.0_f64, 0.0).unwrap(), 1.0);
    assert!((legendre_h3d(0, 0.0_f64, 2.0).unwrap() - 2.0 / 2.0_f64.sinh()).abs() < 1e-14);
    assert_eq!(
        legendre_h3d(5, -2.0_f64, 1.0).unwrap(),
        legendre_h3d(5, 2.0, 1.0).unwrap()
    );
    assert!(legendre_h3d(0, 1.0_f64, -1.0).is_err());
}

#[test]
fn riccati_endpoints_and_range() {
    let o = SphericalBesselOptions::default();
    assert_eq!(riccati_jn(0, 0.0_f64).unwrap(), 0.0);
    assert_eq!(riccati_yn(0, 0.0_f64).unwrap(), -1.0);
    assert_eq!(
        *riccati_jn_with_options(0, 0.0_f64, o)
            .unwrap()
            .derivative()
            .unwrap()
            .value(),
        1.0
    );
    assert_eq!(riccati_yn(0, 1e-300_f64).unwrap(), -1.0);
    assert!((riccati_yn(1, 1e-200_f64).unwrap() * 1e-200 + 1.0).abs() < 1e-14);
    assert!(riccati_yn(1, 0.0_f64).is_err());
    assert_eq!(
        spherical_yn(2, -2.0_f64).unwrap(),
        -spherical_yn(2, 2.0_f64).unwrap()
    );
}

#[test]
fn complex_spherical_origin_and_laurent_terms() {
    use scioxide_core::Complex;
    let z = Complex::new(2.0_f64, 1.0);
    assert!((spherical_jn(0, z).unwrap() - z.sin() / z).norm() < 1e-10);
    assert!((spherical_yn(0, z).unwrap() + z.cos() / z).norm() < 1e-10);
    assert!(
        (spherical_kn(0, z).unwrap()
            - (-z).exp() / z * Complex::new(core::f64::consts::FRAC_PI_2, 0.0))
        .norm()
            < 1e-10
    );
    let small = Complex::new(0.0_f64, 1e-10);
    let k1 = spherical_kn(1, small).unwrap();
    assert!((k1.im - core::f64::consts::PI * 1e-10 / 6.0).abs() < 1e-20);
    assert_eq!(
        spherical_jn(0, Complex::new(0.0_f64, 0.0)).unwrap(),
        Complex::new(1.0, 0.0)
    );
}

#[test]
fn spherical_hankel_recessive_scaling() {
    use scioxide_core::Complex;
    for z in [
        Complex::new(2.0_f64, 1.0),
        Complex::new(-2.0, 1.0),
        Complex::new(2.0, -1.0),
    ] {
        let h1 = spherical_hankel_1(0, z).unwrap();
        let h2 = spherical_hankel_2(0, z).unwrap();
        assert!(
            (h1 - Complex::new(0.0, -1.0) * (Complex::new(0.0, 1.0) * z).exp() / z).norm() < 1e-10
        );
        assert!(
            (h2 - Complex::new(0.0, 1.0) * (Complex::new(0.0, -1.0) * z).exp() / z).norm() < 1e-10
        );
    }
    let z = Complex::new(2.0_f64, 1000.0);
    assert!(
        (spherical_hankel_1_scaled(0, z).unwrap() - Complex::new(0.0, -1.0) / z).norm() < 1e-12
    );
}

#[test]
fn large_real_axis_hankel_retains_scaled_reports() {
    use scioxide_core::{Complex, NumericalIssue, SciOxideErrorView};
    let options = SphericalBesselOptions::default();
    for z in [Complex::new(1000.0_f64, 0.0), Complex::new(-1000.0, -0.0)] {
        for report in [
            spherical_hankel_1_with_options(2000, z, options),
            spherical_hankel_2_with_options(2000, z, options),
            spherical_hankel_1_scaled_with_options(2000, z, options),
            spherical_hankel_2_scaled_with_options(2000, z, options),
        ] {
            let report = report.unwrap();
            assert!(report.termination().is_success());
            assert!(report.value().is_none());
            assert!(report.scaled().unwrap().absolute_error().is_finite());
            assert!(report.logarithmic().unwrap().absolute_error().is_finite());
            assert!(report.derivative().is_none());
        }
        for plain in [
            spherical_hankel_1(2000, z),
            spherical_hankel_2(2000, z),
            spherical_hankel_1_scaled(2000, z),
            spherical_hankel_2_scaled(2000, z),
        ] {
            assert!(matches!(
                plain.unwrap_err().view(),
                SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ));
        }
    }
}
