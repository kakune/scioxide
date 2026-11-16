#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;

use scioxide_core::{Budget, NumericalIssue, RealScalar, ScaledValue, SciOxideErrorView};
use scioxide_special::{
    ParabolicCylinderOptions, ScalarRepresentation, parabolic_d, parabolic_d_with_options,
    parabolic_v, parabolic_v_with_options,
};

fn fixtures<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let epsilon = S::EPSILON;
    let quantum = S::ZERO.next_after(S::ONE);
    let tolerance = S::try_from_f64_lossy(if width == "32" { 2e-5 } else { 5e-9 }).unwrap();
    for row in include_str!("data/parabolic_range.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let columns: Vec<_> = row.split_whitespace().collect();
        if columns[1] != width {
            continue;
        }
        let nu = decode(columns[2]);
        let x = decode(columns[3]);
        let expected = decode(columns[4]);
        let d_expected = decode(columns[5]);
        let sign = S::try_from_f64_lossy(columns[6].parse::<f64>().unwrap()).unwrap();
        let log_expected = decode(columns[7]);
        let options = ParabolicCylinderOptions::default();
        let result = if columns[0] == "d" {
            parabolic_d_with_options(nu, x, options)
        } else {
            parabolic_v_with_options(nu, x, options)
        }
        .unwrap_or_else(|error| panic!("{row}: {error}"));
        assert!(result.termination().is_success(), "{row}");
        assert!(
            result.counts().iterations() <= options.iteration_budget().get(),
            "{row}"
        );
        if expected.is_finite() {
            let value = result
                .value()
                .unwrap_or_else(|| panic!("{row}: no ordinary value"));
            let difference = (*value.value() - expected).abs();
            assert!(
                difference
                    <= tolerance * expected.abs() + S::try_from_f64_lossy(5e-12).unwrap() + quantum,
                "{row}: {:?} versus {expected:?}",
                value.value()
            );
            assert!(
                difference <= value.absolute_error() + epsilon * expected.abs() + quantum,
                "{row}: value error {:?}, difference {difference:?}",
                value.absolute_error()
            );
            let plain = if columns[0] == "d" {
                parabolic_d(nu, x)
            } else {
                parabolic_v(nu, x)
            }
            .unwrap();
            assert_eq!(plain, *value.value(), "{row}: plain/detailed");
        } else {
            assert_eq!(
                result.scalar_representation(),
                ScalarRepresentation::Overflow,
                "{row}"
            );
            assert!(result.value().is_none(), "{row}");
            assert!(
                matches!(
                    result.try_value().unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ),
                "{row}"
            );
            assert!(
                result.logarithmic().is_some(),
                "{row}: missing overflow logarithm"
            );
        }
        if d_expected.is_finite() {
            let derivative = result
                .derivative()
                .unwrap_or_else(|| panic!("{row}: no derivative"));
            let difference = (*derivative.value() - d_expected).abs();
            assert!(
                difference
                    <= tolerance * d_expected.abs()
                        + S::try_from_f64_lossy(5e-12).unwrap()
                        + quantum,
                "{row}: derivative {:?} versus {d_expected:?}",
                derivative.value()
            );
            assert!(
                difference <= derivative.absolute_error() + epsilon * d_expected.abs() + quantum,
                "{row}: derivative error {:?}, difference {difference:?}",
                derivative.absolute_error()
            );
        } else {
            assert!(result.derivative().is_none(), "{row}");
        }
        if let Some(log) = result.logarithmic() {
            assert_eq!(log.value().sign(), sign, "{row}");
            if sign != S::ZERO {
                let actual = log.value().log_magnitude().unwrap();
                assert!(
                    (actual - log_expected).abs()
                        <= log.absolute_error() + epsilon * log_expected.abs() + quantum,
                    "{row}: logarithm {actual:?} versus {log_expected:?}, error {:?}",
                    log.absolute_error()
                );
            }
        }
        let scaled = result
            .scaled()
            .unwrap_or_else(|| panic!("{row}: no scaled value"));
        let reference =
            ScaledValue::from_parts(decode(columns[10]), columns[11].parse().unwrap()).unwrap();
        if sign == S::ZERO {
            assert_eq!(scaled.value().component_significands(), S::ZERO, "{row}");
        } else {
            let difference = reference.scale_exponent() - scaled.value().scale_exponent();
            assert!(
                difference.abs() <= 1,
                "{row}: scaled exponent gap {difference}"
            );
            let ratio = (reference.component_significands()
                / scaled.value().component_significands())
                * S::try_from_f64_lossy(2.0_f64.powi(difference as i32)).unwrap();
            assert!(
                (ratio - S::ONE).abs()
                    <= scaled.relative_error().unwrap()
                        + S::try_from_f64_lossy(4.0).unwrap() * epsilon,
                "{row}: scaled relative difference {:?}, bound {:?}",
                (ratio - S::ONE).abs(),
                scaled.relative_error()
            );
        }
    }
}

#[test]
fn range_fixtures_f64() {
    fixtures("64", |bits| {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    });
}

#[test]
fn range_fixtures_f32() {
    fixtures("32", |bits| {
        f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
    });
}

#[test]
fn default_budget_covers_order_and_coordinate_range() {
    fn sweep<S: RealScalar>() {
        for index in 0..=80 {
            let nu = S::try_from_f64_lossy(-20.0 + f64::from(index) * 0.5).unwrap();
            for magnitude in [
                0.0, 1e-20, 0.01, 0.5, 2.0, 3.0, 3.5, 8.0, 15.0, 25.0, 31.0, 32.0, 40.0, 42.0,
                55.0, 104.0, 1000.0, 10000.0,
            ] {
                for x in [magnitude, -magnitude] {
                    let x = S::try_from_f64_lossy(x).unwrap();
                    for result in [
                        parabolic_d_with_options(nu, x, ParabolicCylinderOptions::default()),
                        parabolic_v_with_options(nu, x, ParabolicCylinderOptions::default()),
                    ] {
                        let result =
                            result.unwrap_or_else(|error| panic!("nu={nu:?}, x={x:?}: {error}"));
                        assert!(result.termination().is_success(), "nu={nu:?}, x={x:?}");
                        if magnitude >= 104.0 {
                            assert!(result.counts().iterations() <= 128, "nu={nu:?}, x={x:?}");
                        }
                    }
                }
            }
        }
    }
    sweep::<f32>();
    sweep::<f64>();
}

#[test]
fn asymptotic_budgets_allocations_and_overflow_forms() {
    for (nu, x, v) in [
        (0.0_f64, 104.0_f64, true),
        (0.5, -104.0, false),
        (19.5, -10000.0, true),
        (-19.5, 10000.0, false),
    ] {
        let evaluate = |options| {
            if v {
                parabolic_v_with_options(nu, x, options)
            } else {
                parabolic_d_with_options(nu, x, options)
            }
        };
        let options = ParabolicCylinderOptions::default();
        let complete = evaluate(options).unwrap();
        let count = complete.counts().iterations();
        for limit in [1, count - 1, count] {
            let options = options.with_iteration_budget(Budget::try_from(limit).unwrap());
            let result = evaluate(options);
            if limit == count {
                assert_eq!(result.unwrap(), complete);
            } else {
                assert!(publication::exhausted_counts(&result.unwrap_err()).iterations() <= limit);
            }
        }
        assert!(complete.scaled().is_some());
        assert!(complete.logarithmic().is_some());
        if x.abs() == 104.0 {
            assert_eq!(
                complete.scalar_representation(),
                ScalarRepresentation::Overflow
            );
        }
    }
    let witness = parabolic_d(-0.1_f64, 3.5).unwrap();
    assert!((witness - 0.041_097_872_365_129_65).abs() <= 64.0 * f64::EPSILON * witness.abs());
}
