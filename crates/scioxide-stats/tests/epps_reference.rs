//! Independent Epps–Singleton references on exactly decoded binary observations.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::RealScalar;
use scioxide_stats::{ClassicalOptions, epps_singleton_two_sample_test_with_options};

fn references<S: RealScalar>(scalar: &str, statistic_tolerance: f64) {
    for row in include_str!("data/epps.tsv")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        if fields[0] != scalar {
            continue;
        }
        let decode = |field: &str| -> Vec<S> {
            field
                .split(',')
                .map(|bits| {
                    let value = if scalar == "f32" {
                        f64::from(f32::from_bits(u32::from_str_radix(bits, 16).unwrap()))
                    } else {
                        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
                    };
                    S::try_from_f64_lossy(value).unwrap()
                })
                .collect()
        };
        let x = decode(fields[2]);
        let y = decode(fields[3]);
        let points: Vec<f64> = fields[4].split(',').map(|p| p.parse().unwrap()).collect();
        let expected: f64 = fields[5].parse().unwrap();
        let expected_p: f64 = fields[6].parse().unwrap();
        for (a, b) in [(&x, &y), (&y, &x)] {
            let actual = epps_singleton_two_sample_test_with_options(
                a,
                b,
                [points[0], points[1]],
                ClassicalOptions::default(),
            )
            .unwrap();
            let statistic = actual.statistic().try_to_f64_exact().unwrap();
            assert!(
                (statistic - expected).abs() <= statistic_tolerance * (1.0 + expected.abs()),
                "{scalar} {}: statistic {statistic} != {expected}",
                fields[1]
            );
            assert!(
                (actual.p_value() - expected_p).abs() <= 2e-10,
                "{scalar} {}: p {} != {expected_p}",
                fields[1],
                actual.p_value()
            );
            assert_eq!(actual.observations(), x.len() + y.len());
            assert_eq!(actual.groups(), 2);
        }
    }
}

#[test]
fn independent_f64_references() {
    references::<f64>("f64", 2e-10);
}

#[test]
fn independent_f32_references() {
    references::<f32>("f32", 2.0 * f64::from(f32::EPSILON));
}

fn affine_invariance<S: RealScalar>(shift: f64, large: f64, tiny: f64) {
    let convert = |v| S::try_from_f64_lossy(v).unwrap();
    let x = [0., 1., 2., 3., 4., 5.];
    let y = [1., 2., 3., 4., 5., 6.];
    let test = |x: &[S], y: &[S]| {
        epps_singleton_two_sample_test_with_options(x, y, [0.4, 0.8], ClassicalOptions::default())
            .unwrap()
    };
    let base = test(&x.map(convert), &y.map(convert));
    for (origin, scale) in [
        (shift, 1.),
        (-shift, 1.),
        (0., large),
        (0., tiny),
        (0., -1.),
        (0., -tiny),
    ] {
        let actual = test(
            &x.map(|v| convert(origin + scale * v)),
            &y.map(|v| convert(origin + scale * v)),
        );
        assert!(
            (actual.statistic().try_to_f64_exact().unwrap()
                - base.statistic().try_to_f64_exact().unwrap())
            .abs()
                <= 8.0 * S::EPSILON.try_to_f64_exact().unwrap()
        );
        assert!((actual.p_value() - base.p_value()).abs() <= 1e-12);
    }
}

#[test]
fn exactly_represented_translation_scale_and_reflection() {
    affine_invariance::<f64>(2_f64.powi(50), 2_f64.powi(1000), f64::from_bits(1));
    affine_invariance::<f32>(
        2_f64.powi(20),
        2_f64.powi(120),
        f64::from(f32::from_bits(1)),
    );
}
