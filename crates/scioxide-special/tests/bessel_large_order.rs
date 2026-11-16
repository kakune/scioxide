//! Uniform large-order values and range classifications against contour integrals.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;

fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S) {
    for row in include_str!("data/bessel_large_order.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != scalar {
            continue;
        }
        let order = decode(f[1]);
        let x = decode(f[2]);
        let expected_j = decode(f[3]);
        let expected_y = decode(f[4]);
        for (second, expected, logarithm) in [
            (false, expected_j, decode(f[5])),
            (true, expected_y, decode(f[6])),
        ] {
            let result = if second {
                bessel_y_with_options(order, x, BesselOptions::default())
            } else {
                bessel_j_with_options(order, x, BesselOptions::default())
            };
            let r = result.unwrap_or_else(|e| panic!("{row} second={second}: {e}"));
            assert!(r.termination().is_success(), "{row}");
            assert!(
                r.counts().iterations() < 4096,
                "order-independent work {row}"
            );
            if expected.is_finite() {
                let actual = r.try_value().unwrap();
                let rounding = S::EPSILON * expected.abs() + S::MIN_SUBNORMAL;
                assert!(
                    (actual - expected).abs() <= r.value().unwrap().absolute_error() + rounding,
                    "coverage {row} second={second}: actual={actual:?}, bound={:?}",
                    r.value().unwrap().absolute_error()
                );
                let amplitude = if expected_j.is_finite() && expected_y.is_finite() {
                    expected_j.abs().max(expected_y.abs())
                } else {
                    expected.abs()
                };
                assert!(
                    (actual - expected).abs()
                        <= S::try_from_f64_lossy(64.).unwrap() * S::EPSILON * amplitude + rounding,
                    "accuracy {row} second={second}: {actual:?}"
                );
            } else {
                assert_eq!(
                    r.scalar_representation(),
                    ScalarRepresentation::Overflow,
                    "{row}"
                );
                assert!(r.try_value().is_err());
            }
            let log = r
                .logarithmic()
                .unwrap_or_else(|| panic!("missing log {row}"));
            let actual = log.value().log_magnitude().unwrap();
            assert!(
                (actual - logarithm).abs() <= log.absolute_error() + S::EPSILON * logarithm.abs(),
                "log {row}: {actual:?}"
            );
        }
    }
}

#[test]
fn binary64_large_order() {
    check("f64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn binary32_large_order() {
    check("f32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}

#[test]
fn mathematical_range_and_actual_exhaustion_are_distinct() {
    for x in [1., 2., 2.5] {
        assert_eq!(bessel_j(70000.0_f64, x).unwrap(), 0.0);
    }
    let rounded =
        bessel_j_with_options(70000.0_f64, f64::from_bits(1), BesselOptions::default()).unwrap();
    assert_eq!(rounded.try_value().unwrap(), 0.0);
    assert!(rounded.value().unwrap().absolute_error() > 0.0);
    assert!(rounded.scaled().is_some() && rounded.logarithmic().is_some());
    let error = bessel_y(70000.0_f64, 1.).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    let error = bessel_j_with_options(
        10000.0_f64,
        10000.,
        BesselOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
}
