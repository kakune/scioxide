//! Principal complex Carlson lips and defining-contour RJ references.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn references<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/complex_carlson.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap();
        let c = |i| {
            Complex::new(
                S::try_from_f64_lossy(n(i)).unwrap(),
                S::try_from_f64_lossy(n(i + 1)).unwrap(),
            )
        };
        let [x, y, z, p] = [c(2), c(4), c(6), c(8)];
        let options = EllipticOptions::default();
        let r = match f[1] {
            "rf" => carlson_rf_with_options(x, y, z, options),
            "rd" => carlson_rd_with_options(x, y, z, options),
            "rg" => carlson_rg_with_options(x, y, z, options),
            "rc" => carlson_rc_with_options(x, y, options),
            _ => carlson_rj_with_options(x, y, z, p, options),
        }
        .unwrap_or_else(|error| panic!("{row}: {error:?}"));
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let s = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(s, f[12].parse().unwrap());
        let expected = Complex::new(n(10), n(11));
        // The reported scaled error covers the reference, allowing its rounding.
        let scaled = r.scaled().unwrap();
        let reported = scaled_reference::error_at_exponent(
            scaled.value(),
            scaled.absolute_error().into_f64(),
            f[12].parse().unwrap(),
        );
        let precision = if width == "32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        assert!(
            (actual - expected).norm() <= reported + 2.0 * precision * expected.norm(),
            "{row}: reported scaled error {reported:e}"
        );
        assert!(
            (actual - expected).norm() < if width == "32" { 3e-6 } else { 5e-12 },
            "{row}: {actual:?} expected={expected:?}"
        );
    }
}
#[test]
fn independent_f64() {
    references::<f64>("64");
}
#[test]
fn independent_f32() {
    references::<f32>("32");
}

type C = Complex<f64>;

fn reports([x, y, z, p]: [C; 4], options: EllipticOptions) -> [SpecialFunctionResult<C>; 4] {
    [
        carlson_rc_with_options(x, y, options).unwrap(),
        carlson_rf_with_options(x, y, z, options).unwrap(),
        carlson_rd_with_options(x, y, z, options).unwrap(),
        carlson_rj_with_options(x, y, z, p, options).unwrap(),
    ]
}

#[test]
fn homogeneity_across_component_and_nested_rc_range_boundaries() {
    let a = [
        C::new(1.0, 0.25),
        C::new(2.0, -0.5),
        C::new(4.0, 0.125),
        C::new(0.5, -0.25),
    ];
    let options = EllipticOptions::default();
    let reference = reports(a, options).map(|r| r.try_value().unwrap());
    // The outer guard checks every component, while RJ's cubic RC arguments
    // cross it sooner. Both neighbours of each power also exercise the guard.
    for exponent in [
        -1000, -102, -100, -98, -40, -34, -32, 0, 32, 34, 40, 98, 100, 102, 1000,
    ] {
        let power = 2.0_f64.powi(exponent);
        for scale in [power.next_down(), power, power.next_up()] {
            for ((report, expected), degree) in reports(a.map(|z| z * scale), options)
                .into_iter()
                .zip(reference)
                .zip([-1, -1, -3, -3])
            {
                assert!(report.termination().is_success(), "{exponent}: {report:?}");
                let actual = scaled_reference::at_exponent(
                    report.scaled().unwrap().value(),
                    i64::from(degree * exponent / 2),
                );
                let expected = expected * scale.scalbn(-exponent).sqrt().powi(degree);
                assert!(
                    (actual - expected).norm() <= 8e-14 * expected.norm(),
                    "exponent={exponent}, degree={degree}: {actual:?} != {expected:?}"
                );
                assert!(report.derivative().is_none());
                // Degree -3 at 2^±1000 overflows, or rounds to zero.
                if exponent == -1000 && degree == -3 {
                    assert!(report.value().is_none());
                }
                if exponent == 1000 && degree == -3 {
                    assert_eq!(*report.value().unwrap().value(), Complex::new(0.0, 0.0));
                }
            }
        }
    }
}

#[test]
fn ordinary_identities_and_each_nested_work_boundary() {
    use scioxide_core::Budget;
    let a = [
        C::new(0.5, 0.25),
        C::new(2.0, -0.5),
        C::new(3.0, 1.0),
        C::new(0.25, -0.125),
    ];
    let [x, y, z, p] = a;
    let options = EllipticOptions::default();
    assert!((carlson_rc(x, y).unwrap() - carlson_rf(x, y, y).unwrap()).norm() < 2e-14);
    assert!((carlson_rd(x, y, z).unwrap() - carlson_rj(x, y, z, z).unwrap()).norm() < 2e-14);
    for (report, plain) in reports(a, options).into_iter().zip([
        carlson_rc(x, y).unwrap(),
        carlson_rf(x, y, z).unwrap(),
        carlson_rd(x, y, z).unwrap(),
        carlson_rj(x, y, z, p).unwrap(),
    ]) {
        let value = report.try_value().unwrap();
        assert_eq!(value.re.to_bits(), plain.re.to_bits());
        assert_eq!(value.im.to_bits(), plain.im.to_bits());
    }
    let expected = carlson_rj_with_options(x, y, z, p, options).unwrap();
    let used = expected.counts().iterations();
    for limit in 1..=used {
        let report = carlson_rj_with_options(
            x,
            y,
            z,
            p,
            options.with_iteration_budget(Budget::try_from(limit).unwrap()),
        );
        if limit == used {
            assert_eq!(report.unwrap(), expected);
        } else {
            assert_eq!(
                publication::exhausted_counts(&report.unwrap_err()).iterations(),
                limit
            );
        }
    }
}

#[test]
fn separated_components_and_equal_arguments_preserve_analytic_values() {
    for exponent in [-101, -100, -99, 0, 99, 100, 101] {
        let x = C::new(2.0_f64.powi(exponent), 2.0_f64.powi(-exponent));
        let rf = C::new(1.0, 0.0) / x.sqrt();
        let rd = rf / x;
        for (r, expected) in reports([x; 4], EllipticOptions::default())
            .into_iter()
            .zip([rf, rf, rd, rd])
        {
            let actual = r.try_value().unwrap();
            assert!((actual - expected).norm() <= 8e-14 * expected.norm());
            for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
                assert!(
                    (actual - expected).abs() <= 8e-14 * expected.abs(),
                    "{exponent}: {actual:e} != {expected:e}"
                );
            }
        }
    }
}
