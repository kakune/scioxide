//! Default termination tolerances, the scaled residual test and the error that
//! a one-shot form reports for an unsuccessful termination.
//!
//! The residual reference of each solve is fixed when the solve starts.
use scioxide_core::{
    BudgetKind, ConvergenceDiagnostics, ConvergenceReason, ExecutionStopReason, RealScalar,
    ResidualTolerance, SciOxideError, SolveCounts, StepTolerance, Tolerance,
};

use crate::RootTermination;

/// Default residual tolerance: reduction of the initial residual by `√ε`, with
/// zero absolute tolerance so that the test does not depend on function units.
#[expect(
    clippy::expect_used,
    reason = "Scalar precision constants define valid tolerances"
)]
pub(crate) fn relative_tolerance<S: RealScalar>() -> ResidualTolerance<S> {
    ResidualTolerance::from_tolerance(
        Tolerance::from_sqrt_epsilon()
            .with_absolute(S::ZERO)
            .expect("valid scalar precision tolerance"),
    )
}

/// Default scalar step tolerance `4ε + 4ε|x|`, which lets a bracket contract to
/// full precision.
#[expect(
    clippy::expect_used,
    reason = "Four machine epsilons are finite and nonnegative"
)]
pub(crate) fn scalar_step_tolerance<S: RealScalar>() -> StepTolerance<S> {
    let four_epsilon = (S::EPSILON + S::EPSILON) + (S::EPSILON + S::EPSILON);
    StepTolerance::from_tolerance(
        Tolerance::from_machine_epsilon()
            .with_absolute(four_epsilon)
            .and_then(|t| t.with_relative(four_epsilon))
            .expect("valid scalar precision tolerance"),
    )
}

/// Tests `|value| ≤ absolute + relative · scale` without rounding the threshold.
pub(crate) fn residual_ok<S: RealScalar>(
    tolerance: &Tolerance<S>,
    value: S,
    scale: S,
) -> Result<bool, SciOxideError> {
    if value == S::ZERO {
        return Ok(true);
    }
    if tolerance.relative() == S::ZERO {
        return Ok(value.abs() <= tolerance.absolute());
    }
    let relative = tolerance.relative() * scale;
    let threshold = tolerance.absolute() + relative;
    // Away from a rounding boundary, normal products and finite sums give the
    // same comparison as the extended threshold. Retry close comparisons as
    // well as exceptional ranges so threshold rounding cannot change success.
    if relative.is_normal() && threshold.is_finite() {
        let margin = threshold * (S::EPSILON + S::EPSILON);
        if (value.abs() - threshold).abs() > margin {
            return Ok(value.abs() <= threshold);
        }
    }
    scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
        scioxide_core::Scaled::from_value(value.abs())?,
        tolerance,
        scioxide_core::Scaled::from_value(scale)?,
    )
}

/// Reason of the `ConvergenceFailure` reported when a budget of `kind` ends a
/// solve.
pub(crate) const fn budget_reason(kind: BudgetKind) -> ConvergenceReason {
    match kind {
        BudgetKind::Iteration => ConvergenceReason::IterationLimit,
        BudgetKind::Evaluation => ConvergenceReason::EvaluationLimit,
        kind => ConvergenceReason::ResourceLimit(kind),
    }
}

/// Returns `Ok(())` for a successful `termination` and otherwise the error that
/// a one-shot form reports instead of a partial result.
///
/// Cancellation and deadlines are `ExecutionStopped`, a numerical breakdown is
/// `NumericalInstability`, and every other status is a `ConvergenceFailure`
/// whose diagnostics carry `counts`: iteration and work limits keep their
/// budget reason and the remaining statuses are `Stagnation`.
pub(crate) fn require_success(
    termination: RootTermination,
    counts: SolveCounts,
) -> Result<(), SciOxideError> {
    if termination.is_success() {
        return Ok(());
    }
    Err(match termination {
        RootTermination::Cancelled => {
            SciOxideError::execution_stopped(ExecutionStopReason::Cancelled)
        }
        RootTermination::TimeLimit => {
            SciOxideError::execution_stopped(ExecutionStopReason::DeadlineReached)
        }
        RootTermination::NumericalFailure(issue) => {
            SciOxideError::numerical_instability(issue, "root computation broke down")
        }
        other => SciOxideError::convergence_failure(
            match other {
                RootTermination::IterationLimit => ConvergenceReason::IterationLimit,
                RootTermination::WorkLimit(kind) => budget_reason(kind),
                _ => ConvergenceReason::Stagnation,
            },
            ConvergenceDiagnostics::new().with_counts(counts),
            "root solve ended without a successful termination",
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{EvaluationCounts, NumericalIssue, SciOxideErrorView};

    fn counts() -> SolveCounts {
        SolveCounts::from_own(EvaluationCounts::from_iterations(7))
    }

    /// Reason and diagnostic counts of the `ConvergenceFailure` for `termination`.
    fn convergence_failure(
        termination: RootTermination,
    ) -> Option<(ConvergenceReason, SolveCounts)> {
        match require_success(termination, counts()).unwrap_err().view() {
            SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } => Some((reason, diagnostics.counts())),
            _ => None,
        }
    }

    #[test]
    fn successful_terminations_are_ok() {
        for termination in [
            RootTermination::ResidualTolerance,
            RootTermination::StepTolerance,
            RootTermination::BracketTolerance,
            RootTermination::Complete,
            RootTermination::NoZeros,
        ] {
            assert!(
                require_success(termination, counts()).is_ok(),
                "{termination:?}"
            );
        }
    }

    #[test]
    fn stops_are_execution_stopped() {
        for (termination, expected) in [
            (RootTermination::Cancelled, ExecutionStopReason::Cancelled),
            (
                RootTermination::TimeLimit,
                ExecutionStopReason::DeadlineReached,
            ),
        ] {
            let error = require_success(termination, counts()).unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::ExecutionStopped { reason } if reason == expected
            ));
        }
    }

    #[test]
    fn numerical_failure_keeps_its_issue() {
        let termination = RootTermination::NumericalFailure(NumericalIssue::Overflow);
        let error = require_success(termination, counts()).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn limits_keep_their_budget_reason() {
        for (termination, expected) in [
            (
                RootTermination::IterationLimit,
                ConvergenceReason::IterationLimit,
            ),
            (
                RootTermination::WorkLimit(BudgetKind::Iteration),
                ConvergenceReason::IterationLimit,
            ),
            (
                RootTermination::WorkLimit(BudgetKind::Evaluation),
                ConvergenceReason::EvaluationLimit,
            ),
            (
                RootTermination::WorkLimit(BudgetKind::Proposal),
                ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
            ),
        ] {
            assert_eq!(
                convergence_failure(termination),
                Some((expected, counts())),
                "{termination:?}"
            );
        }
    }

    /// `x² + y − 1 = 0`, `x + y² − 1 = 0`, solved from `(0.5, 0.5)`.
    fn parabolas(x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0] + x[1] - 1.0;
        out[1] = x[0] + x[1] * x[1] - 1.0;
        Ok(())
    }

    /// `G(x) = x/2 + 1`, solved from 0.
    fn halving(x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = 0.5 * x[0] + 1.0;
        Ok(())
    }

    /// `G(x) = x + 1`, which has no fixed point.
    fn shift(x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = x[0] + 1.0;
        Ok(())
    }

    fn fixed_point_result(
        map: fn(&[f64], &mut [f64]) -> Result<(), SciOxideError>,
        options: crate::FixedPointOptions<f64>,
    ) -> crate::FixedPointResult<f64> {
        crate::FixedPointSolver::<f64>::new(
            crate::FixedPointMethod::Anderson,
            crate::FixedPointInput::Map,
            options,
        )
        .unwrap()
        .solve(&crate::ResidualFn::new(1, map), &[0.0])
        .unwrap()
    }

    #[test]
    fn iteration_limits_of_real_solves_keep_their_counts() {
        let options = crate::TrustRegionSystemOptions::default()
            .with_max_iterations(scioxide_core::Budget::ONE);
        let system = crate::SystemRootSolver::solve(
            &crate::TrustRegionSystem::new(options).unwrap(),
            &crate::ResidualFn::new(2, parabolas),
            &[0.5, 0.5],
        )
        .unwrap();
        let options =
            crate::FixedPointOptions::default().with_max_iterations(scioxide_core::Budget::ONE);
        let fixed = fixed_point_result(halving, options);
        for (termination, accounting) in [
            (system.termination(), system.accounting()),
            (fixed.termination(), fixed.accounting()),
        ] {
            assert_eq!(termination, RootTermination::IterationLimit);
            assert_eq!(accounting.own().iterations(), 1);
            let error = require_success(termination, accounting).unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    diagnostics,
                    ..
                } if diagnostics.counts() == accounting
            ));
        }
    }

    #[test]
    fn one_shot_errors_are_those_of_require_success() {
        let no_root = |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] * x[0] + 1.0;
            Ok(())
        };
        let system = crate::SystemRootSolver::solve(
            &crate::TrustRegionSystem::new(crate::TrustRegionSystemOptions::default()).unwrap(),
            &crate::ResidualFn::new(1, no_root),
            &[1.0],
        )
        .unwrap();
        assert_eq!(system.termination(), RootTermination::Stagnation);
        assert_eq!(
            crate::solve_system(no_root, &[1.0]).unwrap_err(),
            require_success(system.termination(), system.accounting()).unwrap_err()
        );

        let fixed = fixed_point_result(shift, crate::FixedPointOptions::default());
        assert_eq!(fixed.termination(), RootTermination::IterationLimit);
        assert_eq!(
            crate::fixed_point(shift, &[0.0]).unwrap_err(),
            require_success(fixed.termination(), fixed.accounting()).unwrap_err()
        );
    }

    #[test]
    fn other_statuses_are_stagnation() {
        for termination in [
            RootTermination::NotStarted,
            RootTermination::Advanced,
            RootTermination::ComplexStepRequired,
            RootTermination::GlobalizationFailure,
            RootTermination::Stagnation,
            RootTermination::LinearFailure(scioxide_linalg::LinearSolveTermination::Stagnation),
            RootTermination::HistoryLimit,
            RootTermination::StepLimit,
            RootTermination::TangentFailure,
            RootTermination::Unverified,
        ] {
            assert_eq!(
                convergence_failure(termination),
                Some((ConvergenceReason::Stagnation, counts())),
                "{termination:?}"
            );
        }
    }
}
