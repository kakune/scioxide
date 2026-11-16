//! Termination to error: `require_success` and `into_values` for every
//! termination, produced by real evaluations.
#![allow(clippy::unwrap_used, clippy::panic)]
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, CapabilityError, Clock, ClockBinding, ConvergenceReason,
    Deadline, EvaluationRole, ExecutionContext, ExecutionStopReason, MonotonicInstant,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, SolveBudgets,
};
use scioxide_differentiate::*;

fn sum_of_squares(x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    y[0] = x.iter().map(|v| v * v).sum();
    Ok(())
}
fn limited(kind: BudgetKind, limit: usize) -> SolveBudgets {
    SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap())
}
fn convergence(error: &SciOxideError) -> (ConvergenceReason, usize) {
    match error.view() {
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => (
            reason,
            diagnostics
                .counts()
                .own()
                .provider(EvaluationRole::Function),
        ),
        _ => panic!("{error:?}"),
    }
}
fn stopped(error: &SciOxideError) -> ExecutionStopReason {
    match error.view() {
        SciOxideErrorView::ExecutionStopped { reason, .. } => reason,
        _ => panic!("{error:?}"),
    }
}
// Checks both forms of an owned result against the expected reason.
fn assert_failure<T: scioxide_core::NumericScalar>(
    result: DifferentiationResult<T>,
    termination: DifferentiationTermination,
    reason: ConvergenceReason,
) {
    let report = *result.report();
    assert_eq!(report.termination(), termination);
    let error = report.require_success().unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(convergence(&error), (reason, report.evaluations()));
    let error = result.into_values().unwrap_err();
    assert_eq!(convergence(&error), (reason, report.evaluations()));
}

#[test]
fn completed_and_converged_return_the_values() {
    let result = FiniteDifference::default()
        .gradient(sum_of_squares, &[1.0, 2.0], None, None)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::Completed
    );
    result.report().require_success().unwrap();
    let expected = result.values().unwrap().to_vec();
    assert_eq!(result.into_values().unwrap(), expected);
    let result = AdaptiveDifference::default()
        .gradient(sum_of_squares, &[1.0, 2.0], None)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::Converged
    );
    result.report().require_success().unwrap();
    let values = result.into_values().unwrap();
    assert!((values[0] - 2.0).abs() < 1e-8 && (values[1] - 4.0).abs() < 1e-8);
}

#[test]
fn extrapolation_limits_are_convergence_failures() {
    // Three levels are too few for e^(5x) at the default tolerance.
    let refinement = AdaptiveDifference::default()
        .with_max_levels(3)
        .derivative(|x: f64| Ok((5.0 * x).exp()), 1.0)
        .unwrap();
    assert!(refinement.report().output_available());
    assert_failure(
        refinement,
        DifferentiationTermination::RefinementLimit,
        ConvergenceReason::TruncationLimit,
    );
    // Values rounded to `f32` make every smaller step worse.
    let roundoff = AdaptiveDifference::default()
        .derivative(|x: f64| Ok(f64::from(x as f32)), 1.0 / 3.0)
        .unwrap();
    assert_failure(
        roundoff,
        DifferentiationTermination::RoundoffLimit,
        ConvergenceReason::Stagnation,
    );
    // Equal bounds leave no step.
    let bounds = scioxide_core::Bounds::new(vec![(Some(1.0), Some(1.0))]).unwrap();
    let step = AdaptiveDifference::default()
        .gradient(sum_of_squares, &[1.0], Some(&bounds))
        .unwrap();
    assert!(step.values().is_none());
    assert_failure(
        step,
        DifferentiationTermination::StepSizeLimit,
        ConvergenceReason::StepSizeLimit,
    );
}

#[test]
fn budget_limits_are_convergence_failures_with_the_evaluation_count() {
    let evaluation = FiniteDifference::default()
        .with_budgets(limited(BudgetKind::Evaluation, 2))
        .gradient(sum_of_squares, &[1.0, 2.0], None, None)
        .unwrap();
    assert_eq!(evaluation.report().evaluations(), 2);
    assert!(evaluation.values().is_none());
    assert_failure(
        evaluation,
        DifferentiationTermination::EvaluationLimit,
        ConvergenceReason::EvaluationLimit,
    );
    let iteration = AdaptiveDifference::default()
        .with_budgets(limited(BudgetKind::Iteration, 1))
        .gradient(sum_of_squares, &[1.0, 2.0], None)
        .unwrap();
    assert_failure(
        iteration,
        DifferentiationTermination::IterationLimit,
        ConvergenceReason::IterationLimit,
    );
}

#[test]
fn cancellation_is_an_execution_stop() {
    let token = CancellationToken::new();
    let method = FiniteDifference::default();
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(2));
    let mut workspace = FiniteDifferenceWorkspace::workspace(request).unwrap();
    let mut out = [7.0; 2];
    let report = method
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                token.cancel();
                sum_of_squares(x, y)
            },
            &[1.0, 2.0],
            None,
            None,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert_eq!(out, [7.0; 2]);
    let error = report.require_success().unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(stopped(&error), ExecutionStopReason::Cancelled);
}

struct StepClock(AtomicBool);
impl Clock for StepClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            Duration::from_secs(u64::from(self.0.load(Ordering::Relaxed))),
        ))
    }
}

#[test]
fn a_passed_deadline_is_an_execution_stop() {
    let clock = StepClock(AtomicBool::new(false));
    let binding = ClockBinding::new(&clock);
    let context = ExecutionContext::sequential()
        .with_deadline(Deadline::at(
            &binding,
            MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
        ))
        .unwrap();
    let method = AdaptiveDifference::default();
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
    let mut workspace = AdaptiveDifferenceWorkspace::workspace(request).unwrap();
    let (mut out, mut errors) = ([7.0], [8.0]);
    let report = method
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                clock.0.store(true, Ordering::Relaxed);
                sum_of_squares(x, y)
            },
            &[1.0],
            None,
            &mut out,
            &mut errors,
            &mut workspace,
            context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::DeadlineReached
    );
    assert_eq!((out, errors), ([7.0], [8.0]));
    let error = report.require_success().unwrap_err();
    assert_eq!(stopped(&error), ExecutionStopReason::DeadlineReached);
}
