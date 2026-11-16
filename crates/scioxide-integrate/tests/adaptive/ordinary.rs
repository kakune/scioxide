//! Public contracts for globally adaptive one-dimensional integration.

#![allow(clippy::unwrap_used)]

use core::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use scioxide_core::{
    Budget, BudgetKind, ConvergenceReason, ErrorTolerance, Interval, NormSelection, RealScalar,
    SciOxideError, SciOxideErrorView,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, IntegrationReport, IntegrationTermination, SubdivisionOptions,
};

fn options<S: RealScalar>(absolute: S, relative: S) -> SubdivisionOptions<S> {
    SubdivisionOptions::default().with_tolerance(
        ErrorTolerance::scalar(absolute, relative, NormSelection::Infinity).unwrap(),
    )
}

fn assert_converged<S: RealScalar + Debug>(
    result: Result<(S, IntegrationReport<S>), SciOxideError>,
    expected: S,
    value_tolerance: S,
) {
    let (value, report) = result.unwrap();
    assert!(report.is_success(), "termination: {}", report.termination());
    assert!((value - expected).abs() <= value_tolerance, "{value:?}");
    let error = report.error_estimate().unwrap();
    assert!(error.is_finite() && error >= S::ZERO);
}

fn domain_failure() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("x"),
        scioxide_core::DomainReason::Undefined,
        "sentinel split failure",
    )
}

#[test]
fn prepared_rules_resample_changed_callbacks_and_supports_after_failure() {
    use scioxide_core::ExecutionContext;

    fn check<S: RealScalar>() {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let plan = AdaptiveIntegrator::new(
            AdaptiveRule::GaussKronrod,
            options(scalar(1e-4), scalar(1e-4)),
        )
        .unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let attempts = AtomicUsize::new(0);
        let mut value = S::ZERO;
        for (lower, upper, amplitude) in [
            (0.0, 1.0, 1.0),
            (-1.0, 2.0, -2.0),
            (0.0, 1.0, 3.0),
            (1.0, 2.0, -1.0),
        ] {
            let lower = scalar(lower);
            let upper = scalar(upper);
            let amplitude = scalar(amplitude);
            let interval = Interval::new(lower, upper).unwrap();
            let center = lower + (upper - lower) * scalar(0.37);
            let width_squared = scalar(0.01);
            let width = width_squared.into_f64().sqrt();
            let expected = scalar(
                amplitude.into_f64() / width
                    * (((upper - center).into_f64() / width).atan()
                        - ((lower - center).into_f64() / width).atan()),
            );
            let function = |x: S| {
                attempts.fetch_add(1, Relaxed);
                if x < lower || x > upper {
                    return Err(domain_failure());
                }
                Ok(amplitude / ((x - center) * (x - center) + width_squared))
            };
            for _ in 0..2 {
                attempts.store(0, Relaxed);
                let report = plan
                    .integrate_into(
                        &function,
                        interval,
                        &mut value,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert!(report.is_success());
                assert!(report.subdivisions() > 0);
                assert_eq!(attempts.load(Relaxed), report.evaluations());
                assert!((value - expected).abs() <= scalar(1e-4) * expected.abs().max(S::ONE));
            }

            let before = value;
            attempts.store(0, Relaxed);
            let failed = plan.integrate_into(
                |x| {
                    if attempts.load(Relaxed) == 2 {
                        attempts.fetch_add(1, Relaxed);
                        return Err(domain_failure());
                    }
                    function(x)
                },
                interval,
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            );
            assert!(failed.is_err());
            assert_eq!(attempts.load(Relaxed), 3);
            assert_eq!(value, before);
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn smooth_peaked_discontinuous_cancelling_and_near_zero_f64() {
    let options = options(1.0e-7, 1.0e-7).with_max_subdivisions(Budget::try_from(500).unwrap());
    let gauss_kronrod =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone()).unwrap();
    let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options).unwrap();

    let sine = |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) };
    let sine_interval = Interval::new(0.0, std::f64::consts::PI).unwrap();
    assert_converged(gauss_kronrod.integrate(&sine, sine_interval), 2.0, 1.0e-10);
    assert_converged(simpson.integrate(&sine, sine_interval), 2.0, 1.0e-7);

    let peaked = |x: f64| -> Result<f64, SciOxideError> {
        Ok(1.0 / 100.0_f64.mul_add((x - 0.3).powi(2), 1.0))
    };
    let unit = Interval::new(0.0, 1.0).unwrap();
    let peaked_expected = (7.0_f64.atan() + 3.0_f64.atan()) / 10.0;
    assert_converged(
        gauss_kronrod.integrate(&peaked, unit),
        peaked_expected,
        1.0e-7,
    );
    assert_converged(simpson.integrate(&peaked, unit), peaked_expected, 1.0e-7);

    let discontinuous =
        |x: f64| -> Result<f64, SciOxideError> { Ok(if x < 0.3 { 0.0 } else { 1.0 }) };
    assert_converged(gauss_kronrod.integrate(&discontinuous, unit), 0.7, 2.0e-6);
    assert_converged(simpson.integrate(&discontinuous, unit), 0.7, 2.0e-6);

    let cancelling = |x: f64| -> Result<f64, SciOxideError> { Ok(x.exp() - 1.0_f64.sinh()) };
    let symmetric = Interval::new(-1.0, 1.0).unwrap();
    assert_converged(
        gauss_kronrod.integrate(&cancelling, symmetric),
        0.0,
        1.0e-10,
    );
    assert_converged(simpson.integrate(&cancelling, symmetric), 0.0, 1.0e-7);

    let near_zero = |x: f64| -> Result<f64, SciOxideError> { Ok(1.0e-12 * x.powi(3)) };
    assert_converged(gauss_kronrod.integrate(&near_zero, symmetric), 0.0, 1.0e-20);
    assert_converged(simpson.integrate(&near_zero, symmetric), 0.0, 1.0e-20);
}

#[test]
fn smooth_peaked_discontinuous_cancelling_and_near_zero_f32() {
    let options = options(2.0e-4_f32, 2.0e-4).with_max_subdivisions(Budget::try_from(500).unwrap());
    let gauss_kronrod =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone()).unwrap();
    let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options).unwrap();

    let sine = |x: f32| -> Result<f32, SciOxideError> { Ok(x.sin()) };
    let sine_interval = Interval::new(0.0_f32, std::f32::consts::PI).unwrap();
    assert_converged(gauss_kronrod.integrate(&sine, sine_interval), 2.0, 5.0e-4);
    assert_converged(simpson.integrate(&sine, sine_interval), 2.0, 5.0e-4);

    let peaked = |x: f32| -> Result<f32, SciOxideError> {
        Ok(1.0 / 100.0_f32.mul_add((x - 0.3).powi(2), 1.0))
    };
    let unit = Interval::new(0.0_f32, 1.0).unwrap();
    let peaked_expected = (7.0_f32.atan() + 3.0_f32.atan()) / 10.0;
    assert_converged(
        gauss_kronrod.integrate(&peaked, unit),
        peaked_expected,
        8.0e-4,
    );
    assert_converged(simpson.integrate(&peaked, unit), peaked_expected, 8.0e-4);

    let discontinuous =
        |x: f32| -> Result<f32, SciOxideError> { Ok(if x < 0.3 { 0.0 } else { 1.0 }) };
    assert_converged(gauss_kronrod.integrate(&discontinuous, unit), 0.7, 2.0e-3);
    assert_converged(simpson.integrate(&discontinuous, unit), 0.7, 2.0e-3);

    let cancelling = |x: f32| -> Result<f32, SciOxideError> { Ok(x.exp() - 1.0_f32.sinh()) };
    let symmetric = Interval::new(-1.0_f32, 1.0).unwrap();
    assert_converged(gauss_kronrod.integrate(&cancelling, symmetric), 0.0, 1.0e-4);
    assert_converged(simpson.integrate(&cancelling, symmetric), 0.0, 5.0e-4);

    let near_zero = |x: f32| -> Result<f32, SciOxideError> { Ok(1.0e-5 * x.powi(3)) };
    assert_converged(gauss_kronrod.integrate(&near_zero, symmetric), 0.0, 1.0e-7);
    assert_converged(simpson.integrate(&near_zero, symmetric), 0.0, 1.0e-7);
}

#[test]
fn evaluation_exhaustion_stops_only_at_complete_kernel_boundaries() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    let limited = |budget: usize| {
        options(0.0, 0.0)
            .with_max_subdivisions(Budget::try_from(20).unwrap())
            .with_max_evaluations(Some(Budget::try_from(budget).unwrap()))
    };
    let assert_stopped = |(value, report): (f64, IntegrationReport<f64>), expected, calls| {
        assert_eq!(report.evaluations(), expected);
        assert_eq!(report.evaluations(), calls);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(value.is_finite());
        let error = report.error_estimate().unwrap();
        assert!(error.is_finite() && error >= 0.0);
    };

    for budget in 15..=104 {
        let calls = AtomicUsize::new(0);
        let function = |x: f64| -> Result<f64, SciOxideError> {
            calls.fetch_add(1, Relaxed);
            Ok((100.0 * x).sin())
        };
        let result = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, limited(budget))
            .unwrap()
            .integrate(&function, interval)
            .unwrap();
        assert_stopped(result, 15 + 30 * ((budget - 15) / 30), calls.load(Relaxed));
    }

    for budget in 10..=40 {
        let calls = AtomicUsize::new(0);
        let function = |x: f64| -> Result<f64, SciOxideError> {
            calls.fetch_add(1, Relaxed);
            Ok(x.powi(8))
        };
        let result = AdaptiveIntegrator::new(AdaptiveRule::Simpson, limited(budget))
            .unwrap()
            .integrate(&function, interval)
            .unwrap();
        assert_stopped(result, 10 + 8 * ((budget - 10) / 8), calls.load(Relaxed));
    }

    // A budget below the initial region publishes nothing and makes no callback.
    let calls = AtomicUsize::new(0);
    let counted = |x: f64| -> Result<f64, SciOxideError> {
        calls.fetch_add(1, Relaxed);
        Ok(x)
    };
    let budget = |limit: usize| {
        SubdivisionOptions::default().with_max_evaluations(Some(Budget::try_from(limit).unwrap()))
    };
    for error in [
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, budget(14))
            .unwrap()
            .integrate(&counted, interval)
            .unwrap_err(),
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, budget(9))
            .unwrap()
            .integrate(&counted, interval)
            .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
    }
    assert_eq!(calls.load(Relaxed), 0);
}

#[test]
fn resource_exhaustion_reports_exact_completed_split_counts() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    for subdivisions in 1..=4 {
        let options =
            options(0.0, 0.0).with_max_subdivisions(Budget::try_from(subdivisions).unwrap());
        let oscillatory = |x: f64| -> Result<f64, SciOxideError> { Ok((100.0 * x).sin()) };
        let (_, gk) = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone())
            .unwrap()
            .integrate(&oscillatory, interval)
            .unwrap();
        assert_eq!(gk.evaluations(), 15 + 30 * subdivisions);
        assert_eq!(
            gk.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
        );

        let polynomial = |x: f64| -> Result<f64, SciOxideError> { Ok(x.powi(8)) };
        let (_, simpson) = AdaptiveIntegrator::new(
            AdaptiveRule::Simpson,
            options
                .with_max_subdivisions(Budget::try_from(15).unwrap())
                .with_max_evaluations(Some(Budget::try_from(10 + 8 * subdivisions).unwrap())),
        )
        .unwrap()
        .integrate(&polynomial, interval)
        .unwrap();
        assert_eq!(simpson.evaluations(), 10 + 8 * subdivisions);
        assert_eq!(
            simpson.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
    }
}

fn erased_and_concrete_paths_match<S: RealScalar + Debug + Default>() {
    let options = options(S::ZERO, S::ZERO).with_max_subdivisions(Budget::ONE);
    let jump = S::try_from_f64_lossy(0.3).unwrap();
    let function =
        |x: S| -> Result<S, SciOxideError> { Ok(if x < jump { S::ZERO } else { S::ONE }) };
    let erased: &(dyn Fn(S) -> Result<S, SciOxideError> + Sync) = &function;
    let interval = Interval::new(S::ZERO, S::ONE).unwrap();
    let kronrod = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone()).unwrap();
    let simpson = AdaptiveIntegrator::new(
        AdaptiveRule::Simpson,
        options
            .with_max_subdivisions(Budget::try_from(15).unwrap())
            .with_max_evaluations(Some(Budget::try_from(18).unwrap())),
    )
    .unwrap();
    for (evaluations, concrete, dynamic) in [
        (
            45,
            kronrod.integrate(&function, interval).unwrap(),
            kronrod.integrate(erased, interval).unwrap(),
        ),
        (
            18,
            simpson.integrate(&function, interval).unwrap(),
            simpson.integrate(erased, interval).unwrap(),
        ),
    ] {
        assert_eq!(concrete, dynamic);
        assert_eq!(concrete.1.evaluations(), evaluations);
    }
}

#[test]
fn concrete_erased_and_scalar_paths_share_exact_schedule_semantics() {
    erased_and_concrete_paths_match::<f64>();
    erased_and_concrete_paths_match::<f32>();
}

#[test]
fn roundoff_degenerate_reversed_and_callback_failures_are_structured() {
    let zero_tolerance = options(0.0_f64, 0.0);
    let constant = |_x: f64| -> Result<f64, SciOxideError> { Ok(1.0) };
    let unit = Interval::new(0.0_f64, 1.0).unwrap();
    let (_, gk_roundoff) =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, zero_tolerance.clone())
            .unwrap()
            .integrate(&constant, unit)
            .unwrap();
    assert_eq!(gk_roundoff.evaluations(), 15);
    assert_eq!(
        gk_roundoff.termination(),
        IntegrationTermination::Stagnation
    );
    let (_, simpson_roundoff) =
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, zero_tolerance.clone())
            .unwrap()
            .integrate(&constant, unit)
            .unwrap();
    assert_eq!(simpson_roundoff.evaluations(), 10);
    assert_eq!(
        simpson_roundoff.termination(),
        IntegrationTermination::Stagnation
    );

    assert!(matches!(
        Interval::<f64>::new(1.0, -1.0).unwrap_err().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        Interval::<f32>::new(1.0, -1.0).unwrap_err().view(),
        SciOxideErrorView::InvalidInput { .. }
    ));

    let calls = AtomicUsize::new(0);
    let degenerate_function = |x: f64| -> Result<f64, SciOxideError> {
        calls.fetch_add(1, Relaxed);
        Ok(x)
    };
    let point = Interval::new(2.0, 2.0).unwrap();
    for (value, report) in [
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default())
            .unwrap()
            .integrate(&degenerate_function, point)
            .unwrap(),
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
            .unwrap()
            .integrate(&degenerate_function, point)
            .unwrap(),
    ] {
        assert_eq!(value, 0.0);
        assert_eq!(report.evaluations(), 0);
        assert_eq!(
            report.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert!(report.error_components().is_none());
    }
    assert_eq!(calls.load(Relaxed), 0);

    let gk_calls = AtomicUsize::new(0);
    let gk_failure = |x: f64| -> Result<f64, SciOxideError> {
        if gk_calls.fetch_add(1, Relaxed) + 1 == 16 {
            Err(domain_failure())
        } else {
            Ok((100.0 * x).sin())
        }
    };
    let error = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, zero_tolerance.clone())
        .unwrap()
        .integrate(&gk_failure, unit)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(gk_calls.load(Relaxed), 16);

    let simpson_calls = AtomicUsize::new(0);
    let simpson_failure = |x: f64| -> Result<f64, SciOxideError> {
        if simpson_calls.fetch_add(1, Relaxed) + 1 == 6 {
            Err(domain_failure())
        } else {
            Ok(x.powi(8))
        }
    };
    let error = AdaptiveIntegrator::new(AdaptiveRule::Simpson, zero_tolerance)
        .unwrap()
        .integrate(&simpson_failure, unit)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(simpson_calls.load(Relaxed), 6);
}
