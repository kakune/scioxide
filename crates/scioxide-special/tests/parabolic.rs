#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;

fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("parabolic", width);
    for row in include_str!("data/parabolic.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = row.split_whitespace().collect();
        if p[1] != width {
            continue;
        }
        let n = decode(p[2]);
        let x = decode(p[3]);
        let reference = decode(p[4]);
        let derivative = decode(p[5]);
        let o = ParabolicCylinderOptions::default();
        let r = match p[0] {
            "d" => parabolic_d_with_options(n, x, o),
            "v" => parabolic_v_with_options(n, x, o),
            _ => parabolic_w_with_options(n, x, o),
        }
        .unwrap_or_else(|e| panic!("{row}: {e}"));
        let actual = r.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
        let plain = match p[0] {
            "d" => parabolic_d(n, x),
            "v" => parabolic_v(n, x),
            _ => parabolic_w(n, x),
        }
        .unwrap();
        assert_eq!(plain, actual, "{row}: plain/detailed projection");
        assert!(r.termination().is_success(), "{row}");
        checker.scalar(
            "value",
            actual,
            reference,
            Some(r.value().unwrap().absolute_error()),
            row,
        );
        let d = *r.derivative().unwrap().value();
        checker.scalar(
            "derivative",
            d,
            derivative,
            Some(r.derivative().unwrap().absolute_error()),
            row,
        );
        // The reported errors cover the rounded references.
        let rounding = |reference: S| S::EPSILON * reference.abs() + S::ZERO.next_after(S::ONE);
        let reported = r.value().unwrap().absolute_error();
        assert!(
            (actual - reference).abs() <= reported + rounding(reference),
            "{row}: value error {reported:?}"
        );
        let reported = r.derivative().unwrap().absolute_error();
        assert!(
            (d - derivative).abs() <= reported + rounding(derivative),
            "{row}: derivative error {reported:?}"
        );
        assert!(r.counts().iterations() <= o.iteration_budget().get());
    }
}

#[test]
fn origin_series_derivative_recurrence_and_forms() {
    for nu in [-2.0_f64, -1.75, -1.0, -0.5, 0.5, 1.25, 1.75] {
        for x in [-3.0_f64, -0.75, -1e-200, 1e-200, 0.75, 3.0] {
            let r = parabolic_d_with_options(nu, x, ParabolicCylinderOptions::default()).unwrap();
            let value = r.try_value().unwrap();
            let derivative = *r.derivative().unwrap().value();
            let following = parabolic_d(nu + 1.0, x).unwrap();
            let expected = x * 0.5 * value - following;
            assert!((derivative - expected).abs() <= 5e-12_f64.max(5e-10 * expected.abs()));
            let scaled = r.scaled().unwrap().value().try_value().unwrap();
            let logarithmic = r.logarithmic().unwrap().value().try_value().unwrap();
            for projected in [scaled, logarithmic] {
                assert!((projected - value).abs() <= 16.0 * f64::EPSILON * value.abs());
            }
        }
    }
}

#[test]
fn origin_series_budget_and_allocations() {
    use scioxide_core::SciOxideErrorKind;

    fn check<S: RealScalar>() {
        for (nu, x) in [(-2.0, 3.0), (-0.5, -3.0), (0.5, 2.0), (1.75, -0.25)] {
            let nu = S::try_from_f64_lossy(nu).unwrap();
            let x = S::try_from_f64_lossy(x).unwrap();
            let options = ParabolicCylinderOptions::default();
            let complete = parabolic_d_with_options(nu, x, options).unwrap();
            let count = complete.counts().iterations();
            assert!(count > 2);
            for limit in [1, 2, count - 1, count] {
                let options = options.with_iteration_budget(Budget::try_from(limit).unwrap());
                let result = parabolic_d_with_options(nu, x, options);
                if limit == count {
                    let result = result.unwrap();
                    assert_eq!(result.counts().iterations(), limit);
                    assert_eq!(result.termination(), complete.termination());
                    assert_eq!(result.value(), complete.value());
                    assert_eq!(result.derivative(), complete.derivative());
                } else {
                    assert_eq!(
                        publication::exhausted_counts(&result.unwrap_err()).iterations(),
                        limit
                    );
                }
            }
            let plain = parabolic_d(nu, x).unwrap();
            assert_eq!(plain, complete.try_value().unwrap());
        }
        for x in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            let error = parabolic_d_with_options(S::HALF, x, ParabolicCylinderOptions::default())
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        }
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn independent_f32() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn independent_f64() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn identities_and_work() {
    for x in [-5.0_f64, -1.0, 0.0, 1.0, 5.0] {
        assert!((parabolic_d(0.0, x).unwrap() - (-x * x / 4.0).exp()).abs() < 1e-14);
        let o = ParabolicCylinderOptions::default();
        let d = parabolic_d_with_options(0.5, x, o).unwrap();
        let v = parabolic_v_with_options(0.5, x, o).unwrap();
        let wronskian = d.try_value().unwrap() * v.derivative().unwrap().value()
            - d.derivative().unwrap().value() * v.try_value().unwrap();
        assert!((wronskian - (2.0 / core::f64::consts::PI).sqrt()).abs() < 1e-8);
    }
    let o = ParabolicCylinderOptions::default().with_iteration_budget(Budget::ONE);
    for r in [
        parabolic_d_with_options(-2.5_f64, 2.0, o),
        parabolic_v_with_options(2.5, 2.0, o),
        parabolic_w_with_options(2.5, 2.0, o),
    ] {
        let r = r.unwrap_err();
        let r_counts = publication::exhausted_counts(&r);
        assert!(r_counts.iterations() <= 1);
    }
    assert!(parabolic_w(6.0_f64, 0.0).is_err());
    assert!(parabolic_d(f64::NAN, 1.0).is_err());
}

#[test]
fn complex_integer_orders_use_hermite_values() {
    use scioxide_core::Complex;
    let z = Complex::new(2.0_f64, 1.0);
    let base = (-z * z * Complex::new(0.25, 0.0)).exp();
    assert!((parabolic_d_complex(0, z).unwrap() - base).norm() < 1e-14);
    assert!((parabolic_d_complex(1, z).unwrap() - z * base).norm() < 1e-14);
    assert!(
        (parabolic_d_complex(2, z).unwrap() - (z * z - Complex::new(1.0, 0.0)) * base).norm()
            < 1e-13
    );
    let z32 = Complex::new(2.0_f32, 1.0);
    assert!(
        (parabolic_d_complex(2, z32).unwrap().re - parabolic_d_complex(2, z).unwrap().re as f32)
            .abs()
            < 1e-6
    );
}
