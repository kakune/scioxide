//! Shared transform resource checks and preparation work.
use scioxide_core::{Budget, ExecutionContext, SciOxideError, SolveContext};

use crate::validation::numerical;

// Per-call preparation work charged to the enclosing solve scope.
pub(super) struct Work<'a, 's, 'c> {
    remaining: usize,
    pub used: usize,
    pub parent: &'a mut SolveContext<'s, 'c>,
}
impl<'a, 's, 'c> Work<'a, 's, 'c> {
    fn exhausted(&self) -> SciOxideError {
        let diagnostics =
            scioxide_core::ConvergenceDiagnostics::new().with_counts(self.parent.counts());
        SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            diagnostics,
            "preparation work exhausted",
        )
    }

    pub(super) fn new(budget: Budget, parent: &'a mut SolveContext<'s, 'c>) -> Self {
        Self {
            remaining: budget.get(),
            used: 0,
            parent,
        }
    }
    pub(super) fn step(&mut self, context: ExecutionContext<'_>) -> Result<(), SciOxideError> {
        self.check(context)?;
        if self.remaining == 0 {
            return Err(self.exhausted());
        }
        admit_iterations(self.parent, 1)?;
        self.remaining -= 1;
        self.used += 1;
        Ok(())
    }
    /// Return a nonempty remaining allowance for one nested special solver.
    pub(super) fn remaining_budget(
        &mut self,
        context: ExecutionContext<'_>,
    ) -> Result<Budget, SciOxideError> {
        if self.remaining == 0 {
            self.step(context)?;
        }
        Budget::try_from(self.remaining)
    }
    pub(super) fn check(&self, context: ExecutionContext<'_>) -> Result<(), SciOxideError> {
        context.check()?;
        self.parent.execution().check()?;
        Ok(())
    }
    /// Charge special-function work that ran outside the parent ledger.
    pub(super) fn charge_parent(&mut self, used: usize) -> Result<(), SciOxideError> {
        admit_iterations(self.parent, used)
    }
    pub(super) fn record_special_iterations(
        &mut self,
        used: usize,
        exhausted: bool,
    ) -> Result<(), SciOxideError> {
        self.remaining = self.remaining.checked_sub(used).ok_or_else(|| {
            numerical(
                scioxide_core::NumericalIssue::Breakdown,
                "invalid special-function work report",
            )
        })?;
        self.used += used;
        if exhausted {
            return Err(self.exhausted());
        }
        Ok(())
    }
    pub(crate) fn bessel(
        &mut self,
        nu: f64,
        x: f64,
        context: ExecutionContext<'_>,
    ) -> Result<f64, SciOxideError> {
        self.bessel_report(nu, x, context)?.try_value()
    }
    pub(crate) fn bessel_report(
        &mut self,
        nu: f64,
        x: f64,
        context: ExecutionContext<'_>,
    ) -> Result<scioxide_special::SpecialFunctionResult<f64>, SciOxideError> {
        // Charge one unit for the call itself, then hand the remaining
        // preparation budget to the special-function solver and deduct what
        // it reports using. Stepping again on an empty budget produces the
        // typed exhaustion error here rather than an invalid zero budget.
        self.step(context)?;
        let options = scioxide_special::BesselOptions::default()
            .with_iteration_budget(self.remaining_budget(context)?);
        let before = self.parent.counts().nested().iterations();
        let result = scioxide_special::bessel_j_scoped(nu, x, options, &mut *self.parent);
        let used = self.parent.counts().nested().iterations() - before;
        self.record_special_iterations(
            used,
            match &result {
                Ok(report) => {
                    report.termination()
                        == scioxide_special::SpecialFunctionTermination::IterationLimit
                }
                Err(error) => matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: scioxide_core::ConvergenceReason::IterationLimit,
                        ..
                    }
                ),
            },
        )?;
        let report = result?;
        context.check()?;
        Ok(report)
    }
}
pub(super) fn admit_iterations(
    context: &mut SolveContext<'_, '_>,
    amount: usize,
) -> Result<(), SciOxideError> {
    match context.record_iterations(amount)? {
        scioxide_core::Admission::Granted(()) => Ok(()),
        scioxide_core::Admission::Denied(report) => Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_budget(report)
                .with_counts(context.counts()),
            "transform iteration allowance exhausted",
        )),
    }
}

/// Count one operator evaluation, with full diagnostics if the budget refuses it.
pub(super) fn admit_operator(context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    match context.record_provider(scioxide_core::EvaluationRole::LinearOperator)? {
        scioxide_core::Admission::Granted(()) => Ok(()),
        scioxide_core::Admission::Denied(report) => Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::EvaluationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_budget(report)
                .with_counts(context.counts()),
            "transform operator evaluation allowance exhausted",
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_bessel_charges_admitted_child_work() {
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(1508),
            scioxide_core::algorithm_id!("fft", "special_work_test", 1),
            0,
            0,
        );
        let execution = ExecutionContext::sequential();
        let mut parent = state
            .begin_solve(execution, identity, &mut budgets)
            .unwrap();
        let mut work = Work::new(Budget::try_from(2).unwrap(), &mut parent);
        let error = work.bessel_report(2.5, 2.0, execution).unwrap_err();
        assert_eq!(work.used, 2);
        assert_eq!(work.remaining, 0);
        assert_eq!(work.parent.counts().own().iterations(), 1);
        assert_eq!(work.parent.counts().nested().iterations(), 1);
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit, diagnostics, ..
        } if diagnostics.counts().total().unwrap().iterations() == 2)
        );
    }
}
