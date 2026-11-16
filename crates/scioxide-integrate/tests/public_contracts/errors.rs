//! Independent exact-integral and representability tests for deterministic diagnostics.
#![allow(clippy::unwrap_used)]
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use scioxide_core::{ErrorTolerance, Interval, NormSelection, SciOxideError, SciOxideErrorView};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, DoubleExponentialDomain, DoubleExponentialIntegrator,
    FixedRule, GaussKronrod, GaussKronrodRule, IntegrationTermination, LevelOptions,
    SubdivisionOptions,
};

macro_rules! precision_cases {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            let tiny = <$scalar>::from_bits(1);
            let scale = <$scalar>::MIN_POSITIVE / 2.0;
            let interval = Interval::new(0.0 as $scalar, 1.0).unwrap();
            let zero = SubdivisionOptions::default()
                .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap());
            let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, zero.clone()).unwrap();
            let kronrod = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, zero).unwrap();
            for amplitude in [tiny, scale, 1.0] {
                let f = |_: $scalar| -> Result<$scalar, SciOxideError> { Ok(amplitude) };
                for (value, report) in [
                    simpson.integrate(&f, interval).unwrap(),
                    kronrod.integrate(&f, interval).unwrap(),
                    GaussKronrod::new(GaussKronrodRule::G7K15)
                        .unwrap()
                        .integrate(&f, interval)
                        .unwrap(),
                ] {
                    let components = report.error_components().unwrap();
                    let floor = components.roundoff().unwrap();
                    assert!(floor >= tiny, "{components:?}");
                    assert!(report.error_estimate().unwrap() >= floor);
                    assert!(
                        (value - amplitude).abs()
                            <= 4.0 * tiny + 4.0 * <$scalar>::EPSILON * amplitude
                    );
                    assert_ne!(
                        report.termination(),
                        IntegrationTermination::ToleranceReached
                    );
                }
            }
            let adjacent = Interval::new(1.0 as $scalar, (1.0 as $scalar).next_up()).unwrap();
            let calls = AtomicUsize::new(0);
            let f = |_: $scalar| -> Result<$scalar, SciOxideError> {
                calls.fetch_add(1, Relaxed);
                Ok(1.0)
            };
            for error in [
                AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
                    .unwrap()
                    .integrate(&f, adjacent)
                    .unwrap_err(),
                AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
                    .unwrap()
                    .integrate(&f, adjacent)
                    .unwrap_err(),
            ] {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Breakdown,
                        ..
                    }
                ));
            }
            assert_eq!(calls.load(Relaxed), 0);
        }
    };
}
precision_cases!(f32_exact_constants_and_adjacent_intervals, f32);
precision_cases!(f64_exact_constants_and_adjacent_intervals, f64);

macro_rules! tail_cases {
    ($name:ident, $scalar:ty, $peak:expr) => {
        #[test]
        fn $name() {
            use scioxide_core::Budget;
            use scioxide_integrate::{ErrorReliability, TailStatus};
            let half_line = DoubleExponentialDomain::HalfLine { lower: 0.0 };
            // With u=log(x), the compact parabola integrates to exactly one.
            // Its support lies beyond the explored transformed grid. No finite
            // collection of apparently decayed samples can certify its absence.
            let hidden = |x: $scalar| -> Result<$scalar, SciOxideError> {
                let z = x.ln() - $peak;
                Ok(if z.abs() < 1.0 {
                    0.75 * (1.0 - z * z) / x
                } else {
                    0.0
                })
            };
            let (value, report) = DoubleExponentialIntegrator::new(
                half_line,
                LevelOptions::default().with_max_levels(Budget::try_from(4).unwrap()),
            )
            .unwrap()
            .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| {
                hidden(point.coordinate())
            })
            .unwrap();
            assert_eq!(
                report.termination(),
                IntegrationTermination::RefinementLimit
            );
            assert!(!report.is_success());
            assert!((value - 1.0).abs() > 0.9);
            let components = report.error_components().unwrap();
            assert_eq!(
                components.tail().unwrap().reliability(),
                ErrorReliability::Heuristic
            );
            assert_eq!(
                components.tail_diagnostics().unwrap().status(),
                TailStatus::HeuristicallyTruncated
            );
            assert!(components.tail_diagnostics().unwrap().shells() >= 3);

            // Cancellation must not inflate the relative stopping scale to L1.
            let odd = |x: $scalar| -> Result<$scalar, SciOxideError> { Ok(x * (-x * x).exp()) };
            let solver = DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::RealLine,
                LevelOptions::default().with_tolerance(
                    ErrorTolerance::scalar(0.0, 0.1, NormSelection::Infinity).unwrap(),
                ),
            )
            .unwrap();
            let (value, cancellation) = solver
                .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| {
                    odd(point.coordinate())
                })
                .unwrap();
            assert!(!cancellation.is_success());
            assert_ne!(
                cancellation.termination(),
                IntegrationTermination::HeuristicToleranceReached
            );
            assert_eq!(value, 0.0);
            assert!(cancellation.error_components().unwrap().roundoff().unwrap() > 0.0);

            for budget in 1..=65 {
                let calls = AtomicUsize::new(0);
                let f = |x: $scalar| -> Result<$scalar, SciOxideError> {
                    calls.fetch_add(1, Relaxed);
                    Ok((-x).exp())
                };
                let solver = DoubleExponentialIntegrator::new(
                    half_line,
                    LevelOptions::default()
                        .with_max_evaluations(Some(Budget::try_from(budget).unwrap())),
                )
                .unwrap();
                let (value, report) = solver
                    .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| {
                        f(point.coordinate())
                    })
                    .unwrap();
                assert_eq!(report.evaluations(), calls.load(Relaxed));
                assert!(calls.load(Relaxed) <= budget);
                if report.is_success() {
                    assert_eq!(
                        report.termination(),
                        IntegrationTermination::HeuristicToleranceReached
                    );
                    assert!((value - 1.0).abs() <= <$scalar>::EPSILON.sqrt() * 2.0);
                }
                if budget == 1 {
                    assert_eq!(
                        report.termination(),
                        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
                    );
                    let components = report.error_components().unwrap();
                    assert!(components.discretization().is_none());
                    assert!(components.tail().is_none());
                    assert!(report.error_estimate().is_none());
                }
            }
        }
    };
}
tail_cases!(f64_hidden_tail_cancellation_and_budgets, f64, 200.0);
tail_cases!(f32_hidden_tail_cancellation_and_budgets, f32, 60.0);
