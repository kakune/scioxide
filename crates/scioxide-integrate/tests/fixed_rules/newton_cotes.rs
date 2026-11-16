//! Composite Newton-Cotes grids, exactness and failure contracts.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{Budget, Interval, NonFiniteOrigin, RealScalar, SciOxideError};
use scioxide_integrate::{FixedRule, IntegrationTermination, NewtonCotes, NewtonCotesEndpoints};

fn constant_f64(_: f64) -> Result<f64, SciOxideError> {
    Ok(1.0)
}

fn constant_f32(_: f32) -> Result<f32, SciOxideError> {
    Ok(1.0)
}

type Stencil = (usize, NewtonCotesEndpoints);
const MIDPOINT: Stencil = (1, NewtonCotesEndpoints::Open);
const TRAPEZOID: Stencil = (2, NewtonCotesEndpoints::Closed);
const SIMPSON: Stencil = (3, NewtonCotesEndpoints::Closed);

fn composite<S: RealScalar>((nodes, endpoints): Stencil, panels: usize) -> NewtonCotes<S> {
    NewtonCotes::new(nodes, endpoints, Budget::try_from(panels).unwrap()).unwrap()
}

#[test]
fn composite_callback_counts_are_exact_for_concrete_and_erased_callbacks() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    for (base, expected) in [(MIDPOINT, 8_usize), (TRAPEZOID, 9), (SIMPSON, 17)] {
        let calls = AtomicUsize::new(0);
        let function = |x: f64| -> Result<f64, SciOxideError> {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(x * x)
        };
        let integrator = composite::<f64>(base, 8);
        assert_eq!(integrator.node_count(), expected);

        let generic = integrator.integrate(&function, interval).unwrap();
        assert_eq!(generic.1.evaluations(), expected);
        assert_eq!(calls.load(Ordering::Relaxed), expected);

        calls.store(0, Ordering::Relaxed);
        let dynamic: &(dyn Fn(f64) -> Result<f64, SciOxideError> + Sync) = &function;
        let dynamic_result = integrator.integrate(dynamic, interval).unwrap();
        assert_eq!(dynamic_result.1.evaluations(), expected);
        assert_eq!(calls.load(Ordering::Relaxed), expected);
        assert_eq!(generic, dynamic_result);
    }
}

#[test]
fn composite_rules_integrate_constants_and_their_exact_polynomials() {
    let interval_f64 = Interval::new(0.0_f64, 2.0).unwrap();
    let interval_f32 = Interval::new(0.0_f32, 2.0).unwrap();
    for base in [MIDPOINT, TRAPEZOID, SIMPSON] {
        let (f64_value, _) = composite(base, 7)
            .integrate(&constant_f64, interval_f64)
            .unwrap();
        let (f32_value, _) = composite(base, 7)
            .integrate(&constant_f32, interval_f32)
            .unwrap();
        assert!((f64_value - 2.0).abs() < 1.0e-14);
        assert!((f32_value - 2.0).abs() < 2.0e-6);
    }

    for base in [MIDPOINT, TRAPEZOID] {
        let (linear_f64, _) = composite(base, 9)
            .integrate(
                |x: f64| -> Result<f64, SciOxideError> { Ok(x) },
                interval_f64,
            )
            .unwrap();
        let (linear_f32, _) = composite(base, 9)
            .integrate(
                |x: f32| -> Result<f32, SciOxideError> { Ok(x) },
                interval_f32,
            )
            .unwrap();
        assert!((linear_f64 - 2.0).abs() < 1.0e-14);
        assert!((linear_f32 - 2.0).abs() < 2.0e-6);
    }

    let (cubic_f64, _) = composite(SIMPSON, 5)
        .integrate(
            |x: f64| -> Result<f64, SciOxideError> { Ok(x * x * x) },
            interval_f64,
        )
        .unwrap();
    let (cubic_f32, _) = composite(SIMPSON, 5)
        .integrate(
            |x: f32| -> Result<f32, SciOxideError> { Ok(x * x * x) },
            interval_f32,
        )
        .unwrap();
    assert!((cubic_f64 - 4.0).abs() < 1.0e-14);
    assert!((cubic_f32 - 4.0).abs() < 5.0e-6);
}

#[test]
fn composite_interval_boundaries_are_preserved() {
    assert!(matches!(
        (Interval::<f64>::new(1.0, 0.0))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));

    let interval = Interval::new(2.0_f64, 2.0).unwrap();
    for base in [MIDPOINT, TRAPEZOID, SIMPSON] {
        let calls = AtomicUsize::new(0);
        let (value, report) = composite(base, 4)
            .integrate(
                |x: f64| -> Result<f64, SciOxideError> {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(x)
                },
                interval,
            )
            .unwrap();
        assert_eq!(value, 0.0);
        assert_eq!(report.evaluations(), 0);
        assert_eq!(
            report.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn composite_compensation_retains_small_cancellation_terms() {
    let (midpoint, _) = composite(MIDPOINT, 3)
        .integrate(
            |x: f64| -> Result<f64, SciOxideError> {
                Ok(if x == 0.5 {
                    1.0e16
                } else if x == 1.5 {
                    1.0
                } else {
                    -1.0e16
                })
            },
            Interval::new(0.0, 3.0).unwrap(),
        )
        .unwrap();
    assert_eq!(midpoint, 1.0);

    let (trapezoidal, _) = composite(TRAPEZOID, 2)
        .integrate(
            |x: f64| -> Result<f64, SciOxideError> {
                Ok(if x == 0.0 {
                    2.0e16
                } else if x == 1.0 {
                    1.0
                } else {
                    -2.0e16
                })
            },
            Interval::new(0.0, 2.0).unwrap(),
        )
        .unwrap();
    assert_eq!(trapezoidal, 1.0);

    let (simpson, _) = composite(SIMPSON, 1)
        .integrate(
            |x: f64| -> Result<f64, SciOxideError> {
                Ok(if x == 0.0 {
                    1.0e16
                } else if x == 3.0 {
                    0.25
                } else {
                    -1.0e16
                })
            },
            Interval::new(0.0, 6.0).unwrap(),
        )
        .unwrap();
    assert_eq!(simpson, 1.0);
}

#[test]
fn composite_errors_stop_without_unreserved_or_duplicate_callbacks() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    for base in [MIDPOINT, TRAPEZOID, SIMPSON] {
        let calls = AtomicUsize::new(0);
        let callback_error = composite::<f64>(base, 4).integrate(
            |x: f64| -> Result<f64, SciOxideError> {
                calls.fetch_add(1, Ordering::Relaxed);
                if calls.load(Ordering::Relaxed) == 2 {
                    Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("callback"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "intentional callback failure",
                    ))
                } else {
                    Ok(x)
                }
            },
            interval,
        );
        assert!(matches!(
            (callback_error).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        let non_finite = composite::<f64>(base, 4)
            .integrate(
                |_: f64| -> Result<f64, SciOxideError> { Ok(f64::INFINITY) },
                interval,
            )
            .unwrap_err();
        assert!(matches!(
            non_finite.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));
    }

    let intermediate = composite::<f64>(MIDPOINT, 1)
        .integrate(
            |_: f64| -> Result<f64, SciOxideError> { Ok(f64::MAX) },
            Interval::new(0.0, 2.0).unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        intermediate.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
}

#[test]
fn composite_panel_cost_overflow_is_rejected_at_construction() {
    for ((nodes, endpoints), panels) in [
        (TRAPEZOID, usize::MAX),
        (SIMPSON, usize::MAX / 2 + 1),
        (MIDPOINT, usize::MAX / 2 + 1),
    ] {
        let result = NewtonCotes::<f64>::new(nodes, endpoints, Budget::try_from(panels).unwrap());
        assert!(matches!(
            result.err().unwrap().view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
    }
}
