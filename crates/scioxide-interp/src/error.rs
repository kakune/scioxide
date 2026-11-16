//! Crate-private error constructors and budget checks.
//!
//! Every family builds its `InvalidInput`, shape and instability errors here,
//! so one condition has one kind and reason across the crate. Budget helpers
//! check for cancellation and deadlines before counting work. A budget that
//! is exhausted before any usable result exists is a `ConvergenceFailure`.

use scioxide_core::{
    AlgorithmId, Argument, BudgetKind, ConvergenceDiagnostics, ConvergenceReason, EvaluationCounts,
    InvalidInputReason, NumericalIssue, SciOxideError, SolveContext, SolveCounts,
};

/// Arguments contradicting each other or their preparation.
pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}

/// An argument value outside its accepted range or set.
pub(crate) fn out_of_range(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::OutOfRange, detail)
}

/// A required collection is empty.
pub(crate) fn empty(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Empty, detail)
}

/// One length disagreeing with its expected value.
pub(crate) fn shape(expected: usize, actual: usize) -> SciOxideError {
    SciOxideError::dimension_mismatch([expected], [actual])
}

/// A numerical invariant lost during construction or evaluation.
pub(crate) fn instability(issue: NumericalIssue, detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(issue, detail)
}

/// Poll execution stops and record one iteration; `false` when denied.
pub(crate) fn admit_iteration(context: &mut SolveContext<'_, '_>) -> Result<bool, SciOxideError> {
    context.execution().check()?;
    Ok(context.record_iteration()?.is_granted())
}

/// The iteration budget denied work before any accepted state.
pub(crate) fn iteration_denied(
    context: &SolveContext<'_, '_>,
    detail: &'static str,
) -> SciOxideError {
    denied(
        context,
        BudgetKind::Iteration,
        ConvergenceReason::IterationLimit,
        detail,
    )
}

/// Record one iteration or fail with [`iteration_denied`].
pub(crate) fn require_iteration(
    context: &mut SolveContext<'_, '_>,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    if admit_iteration(context)? {
        Ok(())
    } else {
        Err(iteration_denied(context, detail))
    }
}

/// Poll execution stops and record one proposal, failing with
/// `ResourceLimit(Proposal)` when the allowance is exhausted.
pub(crate) fn require_proposal(
    context: &mut SolveContext<'_, '_>,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    context.execution().check()?;
    if context.record_proposal()?.is_granted() {
        Ok(())
    } else {
        Err(denied(
            context,
            BudgetKind::Proposal,
            ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
            detail,
        ))
    }
}

/// An iterative fit that ended without success, reported by a one-shot form.
///
/// The configured forms return the fit with its unsuccessful status; the
/// one-shot forms, which cannot return a partial result, turn that status into
/// this `ConvergenceFailure` with the algorithm and its iteration count.
pub(crate) fn unconverged(
    reason: ConvergenceReason,
    algorithm: AlgorithmId,
    iterations: usize,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::convergence_failure(
        reason,
        ConvergenceDiagnostics::new()
            .with_algorithm(algorithm)
            .with_counts(SolveCounts::from_own(EvaluationCounts::from_iterations(
                iterations,
            ))),
        detail,
    )
}

fn denied(
    context: &SolveContext<'_, '_>,
    budget: BudgetKind,
    reason: ConvergenceReason,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::convergence_failure(
        reason,
        ConvergenceDiagnostics::new()
            .with_counts(context.counts())
            .with_budget(context.effective_budget_report(budget)),
        detail,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{
        Budget, ExecutionContext, ProblemToken, SciOxideErrorKind, SciOxideErrorView, SolveBudgets,
        SolveIdentity, SolveScope, SolveState,
    };

    #[test]
    fn constructors_have_one_kind_and_reason() {
        let argument = scioxide_core::argument!("x");
        for (error, reason) in [
            (
                invalid(argument, "detail"),
                InvalidInputReason::Inconsistent,
            ),
            (
                out_of_range(argument, "detail"),
                InvalidInputReason::OutOfRange,
            ),
            (empty(argument, "detail"), InvalidInputReason::Empty),
        ] {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput { reason: actual, .. } if actual == reason
            ));
        }
        assert_eq!(shape(2, 3).kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(
            instability(NumericalIssue::Overflow, "detail").kind(),
            SciOxideErrorKind::NumericalInstability
        );
    }

    #[test]
    fn unconverged_fits_report_reason_algorithm_and_iterations() {
        let algorithm = scioxide_core::algorithm_id!("interp", "error-test", 1);
        let error = unconverged(ConvergenceReason::TruncationLimit, algorithm, 7, "detail");
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::TruncationLimit,
                diagnostics,
                ..
            } if diagnostics.algorithm() == Some(algorithm)
                && diagnostics.counts().own().iterations() == 7
        ));
    }

    fn run<T>(
        budgets: SolveBudgets,
        work: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("interp", "error-test", 1),
            0,
            0,
        );
        SolveScope::from(ExecutionContext::sequential()).run(
            &mut SolveState::new(),
            identity,
            budgets,
            work,
        )
    }

    #[test]
    fn budget_denials_are_convergence_failures() {
        let granted = run(SolveBudgets::unbounded(), |context| {
            require_iteration(context, "iteration")?;
            require_proposal(context, "proposal")?;
            admit_iteration(context)
        });
        assert!(granted.unwrap());
        let one = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::ONE)
            .with_limit(BudgetKind::Proposal, Budget::ONE);
        let iteration = run(one, |context| {
            require_iteration(context, "iteration")?;
            require_iteration(context, "iteration")
        });
        assert!(matches!(
            iteration.unwrap_err().view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                ..
            }
        ));
        let proposal = run(one, |context| {
            require_proposal(context, "proposal")?;
            require_proposal(context, "proposal")
        });
        assert!(matches!(
            proposal.unwrap_err().view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
                ..
            }
        ));
        assert!(
            !run(one, |context| {
                admit_iteration(context)?;
                admit_iteration(context)
            })
            .unwrap()
        );
    }
}
