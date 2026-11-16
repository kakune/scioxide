//! Assertions shared by work-exhaustion regression tests.
#![allow(dead_code, clippy::unwrap_used, clippy::panic)]
pub(crate) fn exhausted_counts(
    error: &scioxide_core::SciOxideError,
) -> scioxide_core::EvaluationCounts {
    match error.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => {
            assert_eq!(reason, scioxide_core::ConvergenceReason::IterationLimit);
            diagnostics.counts().total().unwrap()
        }
        other => panic!("expected iteration exhaustion, got {other:?}"),
    }
}

pub(crate) fn budget_counts<V: scioxide_core::NumericScalar>(
    result: Result<scioxide_special::SpecialFunctionResult<V>, scioxide_core::SciOxideError>,
) -> scioxide_core::EvaluationCounts {
    match result {
        Ok(report) => {
            assert!(
                report.value().is_some()
                    || report.scaled().is_some()
                    || report.logarithmic().is_some()
            );
            report.counts()
        }
        Err(error) => exhausted_counts(&error),
    }
}
