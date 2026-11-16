//! Identical G7K15 rules must expose identical scheduling and stopping behavior.
#![allow(clippy::unwrap_used)]
use core::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use scioxide_core::{
    Budget, BudgetKind, ConvergenceReason, ErrorTolerance, Interval, NormSelection, NumericalIssue,
    RealScalar, SciOxideError, SciOxideErrorView,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, CubaturePlan, IntegrationReport, IntegrationTermination,
    SubdivisionOptions, WeightedQuadraturePlan,
};

/// Run the adaptive, weighted and cubature entry points, assert that they agree,
/// and return the adaptive outcome. The legacy entry points report an
/// unpublished stop as a value-less result instead of an error.
fn run<S: RealScalar + Debug + Default>(
    function: &(impl Fn(S) -> Result<S, SciOxideError> + Sync),
    interval: Interval<S>,
    options: SubdivisionOptions<S>,
) -> Result<(S, IntegrationReport<S>), SciOxideError> {
    let adaptive = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone())
        .unwrap()
        .integrate(function, interval);
    let nd = |point: &[S]| function(point[0]);
    let outcomes = [
        WeightedQuadraturePlan::breakpoints(
            interval,
            &[],
            scioxide_integrate::WeightedQuadratureOptions::new(options.clone()),
        )
        .unwrap()
        .integrate(function),
        CubaturePlan::new(&[interval], options)
            .unwrap()
            .integrate(&nd),
    ];
    for outcome in &outcomes {
        match (&adaptive, outcome) {
            (Ok((expected, report)), Ok((actual, other))) => {
                assert_eq!(actual, expected);
                assert_eq!(other.error_estimate(), report.error_estimate());
                assert_eq!(other.termination(), report.termination());
                assert_eq!(other.evaluations(), report.evaluations());
            }
            (Err(expected), Err(actual)) => {
                let same = match (expected.view(), actual.view()) {
                    (
                        SciOxideErrorView::ConvergenceFailure { reason: a, .. },
                        SciOxideErrorView::ConvergenceFailure { reason: b, .. },
                    ) => a == b,
                    (
                        SciOxideErrorView::NumericalInstability { issue: a, .. },
                        SciOxideErrorView::NumericalInstability { issue: b, .. },
                    ) => a == b,
                    _ => false,
                };
                assert!(same, "{expected:?} vs {actual:?}");
            }
            _ => assert!(
                adaptive.is_ok() == outcome.is_ok(),
                "{adaptive:?} vs {outcome:?}"
            ),
        }
    }
    adaptive
}
fn peak(x: f64) -> Result<f64, SciOxideError> {
    Ok(1.0 / 100.0_f64.mul_add((x - 0.3).powi(2), 1.0))
}
fn exact_options() -> SubdivisionOptions<f64> {
    SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
}
#[test]
fn all_three_entry_points_share_success_and_roundoff_stopping() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let (_, success) = run(&peak, interval, SubdivisionOptions::default()).unwrap();
    assert!(success.is_success());
    assert!(success.evaluations() > 15);
    let (_, stagnation) = run(&|_| Ok(1.0), interval, exact_options()).unwrap();
    assert_eq!(stagnation.termination(), IntegrationTermination::Stagnation);
    assert_eq!(stagnation.evaluations(), 15);
}
#[test]
fn admission_and_complete_domain_limits_match_at_every_boundary() {
    let interval = Interval::new(0.0, 1.0).unwrap();
    let error = run(
        &peak,
        interval,
        exact_options().with_max_evaluations(Some(Budget::try_from(14).unwrap())),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    for (limit, expected) in [(15, 15), (44, 15), (45, 45), (74, 45)] {
        let (_, report) = run(
            &peak,
            interval,
            exact_options().with_max_evaluations(Some(Budget::try_from(limit).unwrap())),
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert_eq!(report.evaluations(), expected);
    }
    let (_, report) = run(
        &peak,
        interval,
        exact_options().with_max_subdivisions(Budget::ONE),
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
    );
    assert_eq!(report.evaluations(), 45);
}
#[test]
fn collapsed_open_geometry_is_unmeasured_without_callbacks() {
    let interval = Interval::new(1.0, f64::from_bits(1.0_f64.to_bits() + 1)).unwrap();
    let calls = AtomicUsize::new(0);
    let error = run(
        &|_| {
            calls.fetch_add(1, Relaxed);
            Ok(1.0)
        },
        interval,
        SubdivisionOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(Relaxed), 0);
}
#[test]
fn binary32_schedules_and_results_match() {
    let interval = Interval::new(0.0_f32, 1.0).unwrap();
    let function = |x: f32| Ok(1.0 / 100.0_f32.mul_add((x - 0.3).powi(2), 1.0));
    let (_, report) = run(&function, interval, SubdivisionOptions::default()).unwrap();
    assert!(report.is_success());
}
