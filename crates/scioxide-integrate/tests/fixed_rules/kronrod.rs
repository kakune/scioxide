//! Public numerical and allocation regressions for canonical Gauss-Kronrod pairs.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{Interval, NonFiniteOrigin, SciOxideError};
use scioxide_integrate::{FixedRule, GaussKronrod, GaussKronrodRule, IntegrationTermination};

const PAIRS: [(GaussKronrodRule, usize, usize); 5] = [
    (GaussKronrodRule::G7K15, 15, 23),
    (GaussKronrodRule::G10K21, 21, 31),
    (GaussKronrodRule::G15K31, 31, 47),
    (GaussKronrodRule::G20K41, 41, 61),
    (GaussKronrodRule::G25K51, 51, 77),
];

#[test]
fn reported_callback_counts_match_observed_calls_for_every_pair() {
    let interval_f64 = Interval::new(0.0_f64, 1.0).unwrap();
    let interval_f32 = Interval::new(0.0_f32, 1.0).unwrap();
    for (pair, evaluations, _) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        let integrator_f32 = GaussKronrod::<f32>::new(pair).unwrap();
        let calls_f64 = AtomicUsize::new(0);
        let function_f64 = |x: f64| -> Result<f64, SciOxideError> {
            calls_f64.fetch_add(1, Ordering::Relaxed);
            Ok(x)
        };
        let (_, report_f64) = integrator.integrate(&function_f64, interval_f64).unwrap();
        assert_eq!(calls_f64.load(Ordering::Relaxed), evaluations);
        assert_eq!(report_f64.evaluations(), evaluations);

        let calls_f32 = AtomicUsize::new(0);
        let function_f32 = |x: f32| -> Result<f32, SciOxideError> {
            calls_f32.fetch_add(1, Ordering::Relaxed);
            Ok(x)
        };
        let (_, report_f32) = integrator_f32
            .integrate(&function_f32, interval_f32)
            .unwrap();
        assert_eq!(calls_f32.load(Ordering::Relaxed), evaluations);
        assert_eq!(report_f32.evaluations(), evaluations);
    }
}

#[test]
fn every_pair_integrates_all_documented_polynomials_for_f64() {
    let interval = Interval::new(-1.0_f64, 1.0).unwrap();
    for (pair, evaluations, exactness) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        for degree in 0..=exactness {
            let exponent = degree as i32;
            let polynomial = |x: f64| -> Result<f64, SciOxideError> { Ok(x.powi(exponent)) };
            let (value, report) = integrator.integrate(&polynomial, interval).unwrap();
            let expected = if degree % 2 == 0 {
                2.0 / (degree + 1) as f64
            } else {
                0.0
            };
            assert!(
                (value - expected).abs() <= 1.0e-12,
                "degree {degree} for {pair:?}: {value} != {expected}"
            );
            assert_eq!(report.evaluations(), evaluations);
            let error = report.error_estimate().unwrap();
            assert!(error.is_finite() && error >= 0.0);
        }
    }
}

#[test]
fn every_pair_integrates_all_documented_polynomials_for_f32() {
    let interval = Interval::new(-1.0_f32, 1.0).unwrap();
    for (pair, evaluations, exactness) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        for degree in 0..=exactness {
            let exponent = degree as i32;
            let polynomial = |x: f32| -> Result<f32, SciOxideError> { Ok(x.powi(exponent)) };
            let (value, report) = integrator.integrate(&polynomial, interval).unwrap();
            let expected = if degree % 2 == 0 {
                2.0 / (degree + 1) as f32
            } else {
                0.0
            };
            assert!(
                (value - expected).abs() <= 5.0e-5,
                "degree {degree} for {pair:?}: {value} != {expected}"
            );
            assert_eq!(report.evaluations(), evaluations);
            let error = report.error_estimate().unwrap();
            assert!(error.is_finite() && error >= 0.0);
        }
    }
}

#[test]
fn gaussian_oracle_matches_for_every_pair_and_scalar() {
    // generate_batteries.py, constants.csv.
    const ORACLE: f64 = 0.746_824_132_812_427;
    let interval_f64 = Interval::new(0.0_f64, 1.0).unwrap();
    let interval_f32 = Interval::new(0.0_f32, 1.0).unwrap();
    for (pair, evaluations, _) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        let integrator_f32 = GaussKronrod::<f32>::new(pair).unwrap();
        let gaussian_f64 = |x: f64| -> Result<f64, SciOxideError> { Ok((-x * x).exp()) };
        let (value_f64, report_f64) = integrator.integrate(&gaussian_f64, interval_f64).unwrap();
        assert!((value_f64 - ORACLE).abs() <= 64.0 * f64::EPSILON);
        assert_eq!(report_f64.evaluations(), evaluations);
        assert!(report_f64.error_estimate().unwrap().is_finite());

        let gaussian_f32 = |x: f32| -> Result<f32, SciOxideError> { Ok((-x * x).exp()) };
        let (value_f32, report_f32) = integrator_f32
            .integrate(&gaussian_f32, interval_f32)
            .unwrap();
        assert!((value_f32 - ORACLE as f32).abs() <= 64.0 * f32::EPSILON);
        assert_eq!(report_f32.evaluations(), evaluations);
        assert!(report_f32.error_estimate().unwrap().is_finite());
    }
}

#[test]
fn asymmetric_peaked_and_cancellation_regressions_cover_both_scalars() {
    let integrator = GaussKronrod::new(GaussKronrodRule::G25K51).unwrap();
    let integrator_f32 = GaussKronrod::<f32>::new(GaussKronrodRule::G25K51).unwrap();

    let asymmetric_f64 = |x: f64| -> Result<f64, SciOxideError> { Ok(x.exp()) };
    let asymmetric_interval_f64 = Interval::new(-0.75_f64, 1.25).unwrap();
    let asymmetric_expected_f64 = 1.25_f64.exp() - (-0.75_f64).exp();
    let (asymmetric_f64_value, asymmetric_f64_report) = integrator
        .integrate(&asymmetric_f64, asymmetric_interval_f64)
        .unwrap();
    assert!((asymmetric_f64_value - asymmetric_expected_f64).abs() <= 1.0e-10);

    let asymmetric_f32 = |x: f32| -> Result<f32, SciOxideError> { Ok(x.exp()) };
    let asymmetric_interval_f32 = Interval::new(-0.75_f32, 1.25).unwrap();
    let asymmetric_expected_f32 = 1.25_f32.exp() - (-0.75_f32).exp();
    let (asymmetric_f32_value, asymmetric_f32_report) = integrator_f32
        .integrate(&asymmetric_f32, asymmetric_interval_f32)
        .unwrap();
    assert!((asymmetric_f32_value - asymmetric_expected_f32).abs() <= 5.0e-4);

    let peaked_f64 = |x: f64| -> Result<f64, SciOxideError> {
        Ok(1.0 / 100.0_f64.mul_add((x - 0.3).powi(2), 1.0))
    };
    let peaked_expected_f64 = (7.0_f64.atan() + 3.0_f64.atan()) / 10.0;
    let (peaked_f64_value, peaked_f64_report) = integrator
        .integrate(&peaked_f64, Interval::new(0.0_f64, 1.0).unwrap())
        .unwrap();
    assert!(
        (peaked_f64_value - peaked_expected_f64).abs() <= 5.0e-10,
        "peaked result {peaked_f64_value} != {peaked_expected_f64}"
    );

    let peaked_f32 = |x: f32| -> Result<f32, SciOxideError> {
        Ok(1.0 / 100.0_f32.mul_add((x - 0.3).powi(2), 1.0))
    };
    let peaked_expected_f32 = (7.0_f32.atan() + 3.0_f32.atan()) / 10.0;
    let (peaked_f32_value, peaked_f32_report) = integrator_f32
        .integrate(&peaked_f32, Interval::new(0.0_f32, 1.0).unwrap())
        .unwrap();
    assert!((peaked_f32_value - peaked_expected_f32).abs() <= 5.0e-4);

    let cancellation_f64 = |x: f64| -> Result<f64, SciOxideError> { Ok(x.exp() - 1.0_f64.sinh()) };
    let (cancellation_f64_value, cancellation_f64_report) = integrator
        .integrate(&cancellation_f64, Interval::new(-1.0_f64, 1.0).unwrap())
        .unwrap();
    assert!(cancellation_f64_value.abs() <= 1.0e-12);

    let cancellation_f32 = |x: f32| -> Result<f32, SciOxideError> { Ok(x.exp() - 1.0_f32.sinh()) };
    let (cancellation_f32_value, cancellation_f32_report) = integrator_f32
        .integrate(&cancellation_f32, Interval::new(-1.0_f32, 1.0).unwrap())
        .unwrap();
    assert!(cancellation_f32_value.abs() <= 5.0e-4);

    for error in [
        asymmetric_f64_report.error_estimate().unwrap(),
        peaked_f64_report.error_estimate().unwrap(),
        cancellation_f64_report.error_estimate().unwrap(),
    ] {
        assert!(error.is_finite() && error >= 0.0);
    }
    for error in [
        asymmetric_f32_report.error_estimate().unwrap(),
        peaked_f32_report.error_estimate().unwrap(),
        cancellation_f32_report.error_estimate().unwrap(),
    ] {
        assert!(error.is_finite() && error >= 0.0);
    }
}

#[test]
fn concrete_and_erased_callbacks_share_values_errors_and_counts() {
    type Erased<'a, S> = &'a (dyn Fn(S) -> Result<S, SciOxideError> + Sync);
    let interval_f64 = Interval::new(-0.5_f64, 1.25).unwrap();
    let function_f64 = |x: f64| -> Result<f64, SciOxideError> { Ok(x.exp() + x * x) };
    let dynamic_f64: Erased<'_, f64> = &function_f64;
    let interval_f32 = Interval::new(-0.5_f32, 1.25).unwrap();
    let function_f32 = |x: f32| -> Result<f32, SciOxideError> { Ok(x.exp() + x * x) };
    let dynamic_f32: Erased<'_, f32> = &function_f32;

    for (pair, evaluations, _) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        let integrator_f32 = GaussKronrod::<f32>::new(pair).unwrap();
        let generic_f64 = integrator.integrate(&function_f64, interval_f64).unwrap();
        let dynamic_f64 = integrator.integrate(dynamic_f64, interval_f64).unwrap();
        assert_eq!(generic_f64.0, dynamic_f64.0);
        assert_eq!(generic_f64.1, dynamic_f64.1);
        assert_eq!(generic_f64.1.evaluations(), evaluations);

        let generic_f32 = integrator_f32
            .integrate(&function_f32, interval_f32)
            .unwrap();
        let dynamic_f32 = integrator_f32.integrate(dynamic_f32, interval_f32).unwrap();
        assert_eq!(generic_f32.0, dynamic_f32.0);
        assert_eq!(generic_f32.1, dynamic_f32.1);
        assert_eq!(generic_f32.1.evaluations(), evaluations);
    }
}

#[test]
fn degenerate_and_reversed_intervals_follow_the_shared_interval_policy() {
    assert!(matches!(
        (Interval::<f64>::new(1.0, -1.0))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (Interval::<f32>::new(1.0, -1.0))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));

    for (pair, _, _) in PAIRS {
        let integrator = GaussKronrod::new(pair).unwrap();
        let integrator_f32 = GaussKronrod::<f32>::new(pair).unwrap();
        let calls_f64 = AtomicUsize::new(0);
        let function_f64 = |x: f64| -> Result<f64, SciOxideError> {
            calls_f64.fetch_add(1, Ordering::Relaxed);
            Ok(x)
        };
        let (value_f64, report_f64) = integrator
            .integrate(&function_f64, Interval::new(2.0_f64, 2.0).unwrap())
            .unwrap();
        assert_eq!(value_f64, 0.0);
        assert_eq!(report_f64.evaluations(), 0);
        assert_eq!(
            report_f64.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert_eq!(calls_f64.load(Ordering::Relaxed), 0);

        let calls_f32 = AtomicUsize::new(0);
        let function_f32 = |x: f32| -> Result<f32, SciOxideError> {
            calls_f32.fetch_add(1, Ordering::Relaxed);
            Ok(x)
        };
        let (value_f32, report_f32) = integrator_f32
            .integrate(&function_f32, Interval::new(2.0_f32, 2.0).unwrap())
            .unwrap();
        assert_eq!(value_f32, 0.0);
        assert_eq!(report_f32.evaluations(), 0);
        assert_eq!(
            report_f32.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert_eq!(calls_f32.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn callback_and_intermediate_failures_remain_structured() {
    let interval_f64 = Interval::new(-1.0_f64, 1.0).unwrap();
    let interval_f32 = Interval::new(-1.0_f32, 1.0).unwrap();
    let integrator = GaussKronrod::new(GaussKronrodRule::G7K15).unwrap();
    let integrator_f32 = GaussKronrod::<f32>::new(GaussKronrodRule::G7K15).unwrap();

    let calls = AtomicUsize::new(0);
    let callback_error = |_x: f64| -> Result<f64, SciOxideError> {
        calls.fetch_add(1, Ordering::Relaxed);
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::Undefined,
            "sentinel callback error",
        ))
    };
    assert!(matches!(
        (integrator.integrate(&callback_error, interval_f64))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let non_finite_f64 = |_x: f64| -> Result<f64, SciOxideError> { Ok(f64::NAN) };
    assert!(matches!(
        (integrator.integrate(&non_finite_f64, interval_f64))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
    let non_finite_f32 = |_x: f32| -> Result<f32, SciOxideError> { Ok(f32::INFINITY) };
    assert!(matches!(
        (integrator_f32.integrate(&non_finite_f32, interval_f32))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));

    let overflowing_f64 = |_x: f64| -> Result<f64, SciOxideError> { Ok(f64::MAX) };
    assert!(matches!(
        (integrator.integrate(&overflowing_f64, interval_f64))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    let overflowing_f32 = |_x: f32| -> Result<f32, SciOxideError> { Ok(f32::MAX) };
    assert!(matches!(
        (integrator_f32.integrate(&overflowing_f32, interval_f32))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
}
