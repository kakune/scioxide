//! Romberg refinement, error floors and early-stop diagnostics.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Budget, CancellationToken, ErrorTolerance, ExecutionContext, Interval, NormSelection,
    RealScalar, SciOxideError,
};
use scioxide_integrate::{IntegrationTermination, LevelOptions, Romberg};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn romberg_handles_near_zero_integrals_and_preserves_the_diagonal_error() {
    let (value, report) = Romberg::new(LevelOptions::default())
        .unwrap()
        .integrate(
            |x: f64| -> Result<f64, SciOxideError> { Ok(x.sin()) },
            Interval::new(-std::f64::consts::PI, std::f64::consts::PI).unwrap(),
        )
        .unwrap();

    assert_eq!(
        report.termination(),
        IntegrationTermination::ToleranceReached
    );
    assert!(value.abs() < 1.0e-14);
    let floor = report.error_components().unwrap().roundoff().unwrap();
    assert!(floor > 0.0 && floor <= 50.0 * f64::EPSILON * 4.0);
    assert_eq!(report.error_estimate(), Some(floor));
    assert_eq!(report.evaluations(), 34);
}

#[test]
fn romberg_early_limits_return_the_completed_diagonal_with_the_smallest_difference() {
    macro_rules! check {
        ($scalar:ty) => {
            for (budget, cancelled) in [3, 5, 9]
                .into_iter()
                .flat_map(|budget| [false, true].map(|cancelled| (budget, cancelled)))
            {
                let options = LevelOptions::default()
                    .with_tolerance(
                        ErrorTolerance::scalar(0.0 as $scalar, 0.0, NormSelection::Infinity)
                            .unwrap(),
                    )
                    .with_max_evaluations(
                        (!cancelled).then_some(Budget::try_from(budget).unwrap()),
                    );
                let cancellation = CancellationToken::new();
                let calls = AtomicUsize::new(0);
                let points: [$scalar; 9] = [0.0, 1.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875];
                let rule = Romberg::new(options).unwrap();
                let mut workspace = rule.workspace(1).unwrap();
                let mut value = 0.0 as $scalar;
                let report = rule
                    .integrate_into(
                        |x: $scalar| -> Result<$scalar, SciOxideError> {
                            let index = calls.fetch_add(1, Ordering::Relaxed);
                            assert_eq!(x, points[index]);
                            if cancelled && index + 1 == budget {
                                cancellation.cancel();
                            }
                            Ok(if index < 2 {
                                0.0
                            } else if index == 2 {
                                1.0
                            } else if index < 5 {
                                100.0
                            } else {
                                -20.0
                            })
                        },
                        Interval::new(0.0, 1.0).unwrap(),
                        &mut value,
                        &mut workspace,
                        ExecutionContext::sequential_native().with_cancellation(&cancellation),
                    )
                    .unwrap();

                assert_eq!(
                    report.termination(),
                    if cancelled {
                        IntegrationTermination::ExecutionStopped(
                            scioxide_core::ExecutionStopReason::Cancelled,
                        )
                    } else {
                        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
                    }
                );
                assert!((value - 2.0 / 3.0).abs() <= <$scalar>::EPSILON);
                assert!(
                    (report.error_estimate().unwrap() - 8.0 / 3.0).abs()
                        <= 4.0 * <$scalar>::EPSILON
                );
                let floor = report.error_components().unwrap().roundoff().unwrap();
                assert!((floor - 25.0 * <$scalar>::EPSILON).abs() <= <$scalar>::EPSILON * floor);
                assert_eq!(report.evaluations(), budget);
                assert_eq!(calls.load(Ordering::Relaxed), budget);
            }
        };
    }
    check!(f32);
    check!(f64);
}

#[test]
fn romberg_same_sign_blocks_keep_the_floor_and_component_order() {
    fn check<S: RealScalar>() {
        let options = LevelOptions::default().with_tolerance(
            ErrorTolerance::scalar(S::ZERO, S::ZERO, NormSelection::Infinity).unwrap(),
        );
        let plan = Romberg::new(options).unwrap();
        let interval = Interval::new(S::ZERO, S::ONE).unwrap();
        for scale in [
            S::ONE,
            S::MIN_POSITIVE / S::EPSILON,
            S::MAX / S::try_from_usize_exact(1024).unwrap(),
        ] {
            for sign in [S::ONE, -S::ONE] {
                let function = |x: S| Ok(sign * scale * (S::ONE + x * x));
                let mut scalar = S::ZERO;
                let mut scalar_workspace = plan.workspace(1).unwrap();
                let report = plan
                    .integrate_into(
                        &function,
                        interval,
                        &mut scalar,
                        &mut scalar_workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert_eq!(report.termination(), IntegrationTermination::Stagnation);
                let expected = sign
                    * scale
                    * (S::try_from_usize_exact(4).unwrap() / S::try_from_usize_exact(3).unwrap());
                let floor = report.error_components().unwrap().roundoff().unwrap();
                assert!(floor > S::ZERO);
                assert!((scalar - expected).abs() <= floor);
                let mut vector = [S::ZERO; 2];
                let mut workspace = plan.workspace(2).unwrap();
                let vector_report = plan
                    .integrate_into(
                        |x: S| {
                            let value = function(x)?;
                            Ok([value, -value])
                        },
                        interval,
                        &mut vector,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert_eq!(vector_report.termination(), report.termination());
                assert_eq!(vector, [scalar, -scalar]);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
