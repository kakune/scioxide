//! Independent AS R94 references evaluated on exact binary input samples.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::RealScalar;
use scioxide_stats::{
    ClassicalOptions, TestAlternative, TestCalibration, shapiro_wilk_statistic_with_options,
    shapiro_wilk_test_with_options,
};

fn references<S: RealScalar>(scalar: &str, statistic_tolerance: f64) {
    for row in include_str!("data/shapiro.tsv")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        if fields[0] != scalar {
            continue;
        }
        let values: Vec<_> = fields[2]
            .split(',')
            .map(|bits| {
                let value = if scalar == "f32" {
                    f64::from(f32::from_bits(u32::from_str_radix(bits, 16).unwrap()))
                } else {
                    f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
                };
                S::try_from_f64_lossy(value).unwrap()
            })
            .collect();
        let expected_w: f64 = fields[3].parse().unwrap();
        let expected_p: f64 = fields[4].parse().unwrap();
        let actual = shapiro_wilk_test_with_options(&values, ClassicalOptions::default()).unwrap();
        let statistic = shapiro_wilk_statistic_with_options(
            &values,
            scioxide_stats::ClassicalOptions::default(),
        )
        .unwrap()
        .try_to_f64_exact()
        .unwrap();
        for w in [statistic, actual.statistic().try_to_f64_exact().unwrap()] {
            assert!(
                (w - expected_w).abs() <= statistic_tolerance,
                "{scalar} {}: W {w} != {expected_w}",
                fields[1]
            );
        }
        // p-values remain binary64 for either input precision. Allow the
        // normal-quantile approximation used by the independent AS R94 oracle.
        assert!(
            (actual.p_value() - expected_p).abs() <= 2e-8,
            "{scalar} {}: p {} != {expected_p}",
            fields[1],
            actual.p_value()
        );
    }
}

#[test]
fn independent_f64_normality_references() {
    references::<f64>("f64", 2e-9);
}

#[test]
fn independent_f32_normality_references() {
    references::<f32>("f32", 2.0 * f64::from(f32::EPSILON));
}

fn boundaries<S: RealScalar>() {
    let convert = |value| S::try_from_f64_lossy(value).unwrap();
    let options = ClassicalOptions::default();
    let values = [0., 1., 2., 3., 4., 10.].map(convert);
    let bytes =
        values.len() * core::mem::size_of::<S>() + (values.len() / 2) * core::mem::size_of::<f64>();
    for length in [0, 1, 2] {
        assert!(
            shapiro_wilk_statistic_with_options(
                &values[..length],
                scioxide_stats::ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(shapiro_wilk_test_with_options(&values[..length], options).is_err());
    }
    for bad in [S::ZERO, convert(1e30), convert(1e-40)] {
        let constant = [bad; 6];
        assert!(
            shapiro_wilk_statistic_with_options(
                &constant,
                scioxide_stats::ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(shapiro_wilk_test_with_options(&constant, options).is_err());
    }
    assert!(
        shapiro_wilk_statistic_with_options(
            &values,
            scioxide_stats::ClassicalOptions::default().with_max_workspace_bytes(bytes)
        )
        .is_ok()
    );
    assert!(
        shapiro_wilk_statistic_with_options(
            &values,
            scioxide_stats::ClassicalOptions::default().with_max_workspace_bytes(bytes - 1)
        )
        .is_err()
    );
    assert!(
        shapiro_wilk_test_with_options(&values, options.with_max_workspace_bytes(bytes)).is_ok()
    );
    assert!(
        shapiro_wilk_test_with_options(&values, options.with_max_workspace_bytes(bytes - 1))
            .is_err()
    );
    assert!(
        shapiro_wilk_test_with_options(&values, options.with_calibration(TestCalibration::Exact))
            .is_err()
    );
    assert!(
        shapiro_wilk_test_with_options(&values, options.with_alternative(TestAlternative::Greater))
            .is_err()
    );

    let mut reversed = values;
    reversed.reverse();
    let forward = shapiro_wilk_test_with_options(&values, options).unwrap();
    let backward = shapiro_wilk_test_with_options(&reversed, options).unwrap();
    assert!((forward.statistic() - backward.statistic()).abs() <= S::EPSILON);
    assert!((forward.p_value() - backward.p_value()).abs() <= f64::EPSILON);

    let large: Vec<_> = (0..5001).map(|i| convert(f64::from(i))).collect();
    assert!(shapiro_wilk_test_with_options(&large[..5000], options).is_ok());
    assert!(shapiro_wilk_test_with_options(&large, options).is_err());
    assert!(
        shapiro_wilk_statistic_with_options(&large, scioxide_stats::ClassicalOptions::default())
            .is_ok()
    );
}

#[test]
fn input_calibration_and_workspace_boundaries() {
    boundaries::<f32>();
    boundaries::<f64>();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            shapiro_wilk_statistic_with_options(
                &[0., bad, 1.],
                scioxide_stats::ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(
            shapiro_wilk_test_with_options(&[0., bad, 1.], ClassicalOptions::default()).is_err()
        );
        assert!(
            shapiro_wilk_statistic_with_options(
                &[0_f32, bad as f32, 1.],
                scioxide_stats::ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(
            shapiro_wilk_test_with_options(&[0_f32, bad as f32, 1.], ClassicalOptions::default())
                .is_err()
        );
    }
}
