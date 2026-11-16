//! Termination statuses, reports and owned results of iterative linear solves.
//!
//! An unsuccessful solve returns the verified iterate with the smallest true
//! residual, the latest one on ties. Residual diagnostics describe that iterate;
//! iteration counts include all work.

use alloc::vec::Vec;

use scioxide_core::{
    NormSelection, NumericScalar, PartialResultTrust, RealScalar, ResidualReport, ResidualScaling,
    ResidualScalingRule, SciOxideError,
};

use crate::dense::Vector;

/// Reason an iterative linear solve stopped.
///
/// Only `Converged` and `LeastSquaresConverged` are successful; test with
/// [`Self::is_success`]. With any other status the solver returns the verified
/// iterate with the smallest true residual, if one was computed, as described
/// in [Iterative solves](crate#iterative-solves).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::LinearSolveTermination;
/// assert!(LinearSolveTermination::Converged.is_success());
/// assert!(!LinearSolveTermination::IterationLimit.is_success());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum LinearSolveTermination {
    /// The true residual `‖b − A x‖₂`, recomputed from the returned iterate,
    /// met the residual tolerance.
    ///
    /// A rounding-level floor never relaxes the tolerance.
    Converged,
    /// The normal residual `‖Aᴴ r‖₂`, recomputed from the returned iterate, met
    /// the least-squares tolerance relative to `‖A‖·‖r‖₂`.
    ///
    /// Only MINRES, LSQR and LSMR return it, MINRES only without a
    /// preconditioner. `‖A‖` is the norm estimate built by the recurrence.
    LeastSquaresConverged,
    /// The reciprocal condition estimate of the projected problem fell below the
    /// limit set with `with_condition_limit` (MINRES, LSQR, LSMR).
    ConditionLimit,
    /// The iteration limit was reached.
    IterationLimit,
    /// The limit on operator and preconditioner evaluations was reached.
    EvaluationLimit,
    /// The iteration stopped making progress before meeting the tolerance.
    ///
    /// Either an update left the iterate unchanged, or a verification did not
    /// improve on the best verified residual while that residual lay within
    /// the rounding floor `ε c (‖b‖₂ + ‖A (s ∘ x)‖₂)` at the best verified
    /// iterate `x`. Here `s` is a fixed ±1 sign pattern and `c` is 16 for real
    /// and 32 for complex scalars. The product with `A` counts toward the
    /// evaluation limit.
    Stagnation,
    /// The recurrence broke down after a usable iterate was found.
    Breakdown(LinearSolveBreakdown),
    /// Cancellation or a deadline stopped the solve.
    ExecutionStopped(scioxide_core::ExecutionStopReason),
}

/// Cause of a Krylov recurrence breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum LinearSolveBreakdown {
    /// A recurrence denominator became too small to divide by.
    RecurrenceDenominator,
    /// CG found a direction `p` with `pᴴ A p ≤ 0`, so `A` is not positive definite.
    NonPositiveCurvature,
    /// Arnoldi orthogonalization produced no new usable direction.
    ArnoldiInvariant,
    /// The preconditioner is not Hermitian positive definite, as CG and MINRES require.
    Preconditioner,
}

impl LinearSolveTermination {
    /// Returns `true` for `Converged` and `LeastSquaresConverged`.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Converged | Self::LeastSquaresConverged)
    }
}

/// Extra diagnostics of Krylov solvers.
///
/// MINRES, LSQR and LSMR fill all fields; their estimates describe the current
/// Lanczos or bidiagonalization segment, not the numerical rank of `A`. The
/// GMRES family, BiCGSTAB, BiCG, CGS, QMR and TFQMR fill only the restart count
/// and the direction-loss flag.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct KrylovDiagnostics<R: RealScalar> {
    /// Normal residual of the returned solution.
    pub(crate) normal_residual: Option<ResidualReport<R>>,
    /// Reciprocal condition estimate of the projected problem.
    pub(crate) condition_estimate: Option<crate::ConditionEstimate<R>>,
    /// Whether a recurrence direction vanished or a denominator became unusable.
    pub(crate) rank_loss_detected: bool,
    /// Number of recurrence or Arnoldi restarts.
    pub(crate) restarts: usize,
}

impl<R: RealScalar> KrylovDiagnostics<R> {
    /// Returns the normal residual `‖Aᴴ r‖₂` of the returned solution, recomputed from it.
    ///
    /// MINRES computes `‖A r‖₂`, which equals `‖Aᴴ r‖₂` for Hermitian `A`.
    pub const fn normal_residual(&self) -> Option<ResidualReport<R>> {
        self.normal_residual
    }

    /// Returns the reciprocal condition estimate of the projected problem.
    ///
    /// It is a heuristic, not a bound on the condition number of `A`.
    pub const fn condition_estimate(&self) -> Option<crate::ConditionEstimate<R>> {
        self.condition_estimate
    }

    /// Returns whether a recurrence direction vanished at working precision.
    ///
    /// The direction is a Lanczos, bidiagonalization or Arnoldi vector, or for
    /// biorthogonal solvers a recurrence denominator. A vanished direction
    /// does not establish that `A` is rank deficient.
    pub const fn rank_loss_detected(&self) -> bool {
        self.rank_loss_detected
    }

    /// Returns the number of recurrence or Arnoldi restarts, including restarts after breakdown.
    ///
    /// Periodic residual verification and residual replacement that keeps the
    /// recurrence do not count.
    pub const fn restarts(&self) -> usize {
        self.restarts
    }
}

/// Termination status, iteration count and residual of one iterative solve.
///
/// The residual history and the returned iterate are read from the workspace
/// with [`IterativeWorkspace::last_accepted`](crate::IterativeWorkspace::last_accepted).
///
/// # Examples
///
/// ```
/// use scioxide_core::{NormSelection, PartialResultTrust, ResidualReport};
/// use scioxide_linalg::{IterativeSolveReport, LinearSolveTermination};
/// let residual = ResidualReport::unscaled(1e-12, NormSelection::L2)?;
/// let report = IterativeSolveReport::new(
///     LinearSolveTermination::Converged,
///     3,
///     Some(residual),
///     PartialResultTrust::ResidualChecked,
/// );
/// assert!(report.is_success());
/// assert_eq!(report.diagnostic_residual_norm(), Some(1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct IterativeSolveReport<R: RealScalar> {
    termination: LinearSolveTermination,
    krylov: Option<KrylovDiagnostics<R>>,
    iterations: usize,
    true_residual: Option<ResidualReport<R>>,
    trust: PartialResultTrust,
}

impl<R: RealScalar> IterativeSolveReport<R> {
    /// Creates a report; `true_residual`, if present, must be a true residual `‖b − A x‖₂`.
    ///
    /// Solvers implemented outside this crate use this constructor. The report
    /// has no Krylov diagnostics.
    pub const fn new(
        termination: LinearSolveTermination,
        iterations: usize,
        true_residual: Option<ResidualReport<R>>,
        trust: PartialResultTrust,
    ) -> Self {
        Self {
            termination,
            krylov: None,
            iterations,
            true_residual,
            trust,
        }
    }

    pub(crate) fn from_summary(summary: IterativeSolveSummary<R>) -> Result<Self, SciOxideError> {
        let true_residual = Some(ResidualReport::new(
            summary.raw_residual_norm,
            NormSelection::L2,
            ResidualScaling::new(ResidualScalingRule::Reference, summary.residual_reference)?,
        )?);
        Ok(Self {
            termination: summary.termination,
            krylov: None,
            iterations: summary.iterations,
            true_residual,
            trust: summary.trust,
        })
    }

    pub(crate) const fn with_krylov(mut self, krylov: KrylovDiagnostics<R>) -> Self {
        self.krylov = Some(krylov);
        self
    }

    /// Returns the reason the solver stopped.
    pub const fn termination(&self) -> LinearSolveTermination {
        self.termination
    }

    /// Returns extra Krylov diagnostics; `None` for CG and the stationary solvers.
    pub const fn krylov(&self) -> Option<KrylovDiagnostics<R>> {
        self.krylov
    }

    /// Returns the number of iterations counted toward this solve's iteration limit.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the true residual `‖b − A x‖₂` of the returned solution.
    ///
    /// The report is scaled by `‖b‖₂`, or by one when `b = 0`. It is `None` when
    /// the solve stopped before an operator evaluation was available.
    pub const fn true_residual(&self) -> Option<ResidualReport<R>> {
        self.true_residual
    }

    /// Returns how far the returned iterate was checked.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }

    /// Returns whether the solver met its convergence criterion.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Returns the unscaled norm of [`Self::true_residual`], if present.
    pub fn diagnostic_residual_norm(&self) -> Option<R> {
        self.true_residual.map(ResidualReport::raw_norm)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IterativeSolveSummary<R: RealScalar> {
    pub(crate) termination: LinearSolveTermination,
    pub(crate) iterations: usize,
    pub(crate) raw_residual_norm: R,
    pub(crate) residual_reference: R,
    pub(crate) trust: PartialResultTrust,
}

/// Solution, report and residual history returned by an owned iterative solve.
///
/// When [`Self::is_success`] is `false`, the solution is the verified iterate
/// with the smallest true residual (the latest one on ties). If the solve stops
/// before the first residual is computed, the solution is the initial guess and
/// the report has no residual.
#[derive(Debug, Clone)]
#[must_use]
pub struct IterativeSolveResult<S: NumericScalar> {
    algorithm: scioxide_core::AlgorithmId,
    solution: Vector<S>,
    report: IterativeSolveReport<S::Real>,
    history: Option<Vec<S::Real>>,
}

impl<S: NumericScalar> IterativeSolveResult<S> {
    pub(crate) const fn new(
        algorithm: scioxide_core::AlgorithmId,
        solution: Vector<S>,
        report: IterativeSolveReport<S::Real>,
        history: Option<Vec<S::Real>>,
    ) -> Self {
        Self {
            algorithm,
            solution,
            report,
            history,
        }
    }

    /// Returns the identifier of the algorithm that produced this result.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Budget, ResidualTolerance, algorithm_id};
    /// use scioxide_linalg::{CgSolver, Matrix, Vector};
    /// let matrix = Matrix::<f64>::identity(1)?;
    /// let rhs = Vector::from_vec(vec![1.0]);
    /// let solver = CgSolver::new(ResidualTolerance::new(1e-12, 1e-12)?, Budget::ONE);
    /// let result = solver.solve_with_options(
    ///     &matrix.view().try_finite()?,
    ///     &rhs,
    ///     scioxide_linalg::Preconditioning::None,
    ///     None,
    /// )?;
    /// assert_eq!(result.algorithm(), algorithm_id!("linear", "CG", 1));
    /// let solution = result.into_converged_solution()?;
    /// assert_eq!(solution.len(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn algorithm(&self) -> scioxide_core::AlgorithmId {
        self.algorithm
    }

    /// Returns the approximate solution `x`.
    pub const fn solution(&self) -> &Vector<S> {
        &self.solution
    }

    /// Returns the solution whatever the termination status.
    pub fn into_solution(self) -> Vector<S> {
        self.solution
    }

    /// Splits the result into solution, report and residual history.
    #[expect(
        clippy::type_complexity,
        reason = "the tuple groups the distinct numerical outputs of this operation"
    )] // The three owned parts are named by the getters above.
    pub fn into_parts(
        self,
    ) -> (
        Vector<S>,
        IterativeSolveReport<S::Real>,
        Option<Vec<S::Real>>,
    ) {
        (self.solution, self.report, self.history)
    }

    /// Returns the report of the solve.
    pub const fn report(&self) -> IterativeSolveReport<S::Real> {
        self.report
    }

    /// Returns the history of unpreconditioned residual norms, if recording was enabled.
    ///
    /// The first entry is the true initial residual `‖b − A x₀‖₂`, followed by
    /// one entry per completed iteration. Between verification checkpoints the
    /// entries are recursively updated residual norms or estimates of the
    /// projected problem; TFQMR records its quasi-minimal residual bound.
    /// Entries at checkpoints, at convergence and at the end are true residuals.
    /// An interrupted solve can have fewer entries than counted iterations. When
    /// the report has a true residual, the last entry is the residual of the
    /// returned solution.
    pub fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }

    /// Returns whether the solver met its convergence criterion.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.report.is_success()
    }

    /// Returns the solution if the solve succeeded, and an error otherwise.
    ///
    /// `LeastSquaresConverged` counts as success. To use the iterate of an
    /// unsuccessful solve, read it with [`Self::solution`] or [`Self::into_parts`]
    /// instead, which keeps the residual and iteration diagnostics.
    ///
    /// # Errors
    ///
    /// - [`ExecutionStopped`](scioxide_core::SciOxideErrorKind::ExecutionStopped) with
    ///   the original reason if the solve was cancelled or reached its deadline.
    /// - [`ConvergenceFailure`](scioxide_core::SciOxideErrorKind::ConvergenceFailure)
    ///   (`NestedSolver`) for any other unsuccessful termination. Its diagnostics
    ///   name the solver and carry the iteration count and residual norm.
    pub fn into_converged_solution(self) -> Result<Vector<S>, SciOxideError> {
        self.report.require_success(self.algorithm)?;
        Ok(self.solution)
    }
}

impl<R: RealScalar> IterativeSolveReport<R> {
    /// Nested-solver convergence failure for an unsuccessful report.
    pub(crate) fn require_success(
        &self,
        solver: scioxide_core::AlgorithmId,
    ) -> Result<(), SciOxideError> {
        if self.is_success() {
            return Ok(());
        }
        if let LinearSolveTermination::ExecutionStopped(reason) = self.termination {
            return Err(SciOxideError::execution_stopped(reason));
        }
        let mut inner_diagnostics = scioxide_core::ConvergenceDiagnostics::new().with_counts(
            (scioxide_core::EvaluationCounts::from_iterations(self.iterations)).into(),
        );
        // A stop before any residual was measured carries no residual report.
        if let Some(residual) = self.diagnostic_residual_norm() {
            inner_diagnostics =
                inner_diagnostics.with_residual(scioxide_core::ResidualReport::unscaled(
                    residual.try_to_f64_exact()?,
                    scioxide_core::NormSelection::L2,
                )?);
        }
        let diagnostics =
            scioxide_core::ConvergenceDiagnostics::new().with_nested(solver, inner_diagnostics);
        Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::NestedSolver,
            diagnostics,
            "nested linear solve did not converge",
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn converged_result_returns_solution_unchanged() {
        let result = IterativeSolveResult::new(
            scioxide_core::algorithm_id!("linear", "test-solver", 1),
            Vector::from_vec(vec![1.0_f64, 2.0]),
            IterativeSolveReport::new(
                LinearSolveTermination::Converged,
                3,
                Some(ResidualReport::unscaled(1.0e-12, NormSelection::L2).unwrap()),
                PartialResultTrust::ResidualChecked,
            ),
            None,
        );

        assert_eq!(
            result.algorithm(),
            scioxide_core::algorithm_id!("linear", "test-solver", 1)
        );
        let solution = result.into_converged_solution().unwrap();

        assert_eq!(solution.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    #[allow(clippy::panic)]
    fn unconverged_result_preserves_convergence_metadata() {
        let result = IterativeSolveResult::new(
            scioxide_core::algorithm_id!("linear", "test-solver", 1),
            Vector::from_vec(vec![1.0_f32, 2.0]),
            IterativeSolveReport::new(
                LinearSolveTermination::IterationLimit,
                7,
                Some(ResidualReport::unscaled(0.25, NormSelection::L2).unwrap()),
                PartialResultTrust::Admissible,
            ),
            Some(vec![1.0, 0.5, 0.25]),
        );

        let error = result.into_converged_solution().unwrap_err();

        let scioxide_core::SciOxideErrorView::ConvergenceFailure {
            diagnostics,
            detail,
            ..
        } = error.view()
        else {
            panic!("expected convergence failure")
        };
        assert_eq!(diagnostics.counts().own().iterations(), 0);
        assert!(diagnostics.residual().is_none());
        let nested = diagnostics.nested_levels().next().unwrap();
        assert_eq!(nested.algorithm().method(), "test-solver");
        assert_eq!(nested.counts().own().iterations(), 7);
        assert_eq!(
            nested
                .residual()
                .map(scioxide_core::ResidualReport::raw_norm),
            Some(0.25)
        );
        assert_eq!(detail, "nested linear solve did not converge");
    }

    #[test]
    #[allow(clippy::panic)]
    fn unmeasured_stops_report_no_nested_residual() {
        let result = IterativeSolveResult::new(
            scioxide_core::algorithm_id!("linear", "test-solver", 1),
            Vector::from_vec(vec![1.0_f64]),
            IterativeSolveReport::new(
                LinearSolveTermination::EvaluationLimit,
                0,
                None,
                PartialResultTrust::Admissible,
            ),
            None,
        );
        let error = result.into_converged_solution().unwrap_err();
        let scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } = error.view()
        else {
            panic!("expected convergence failure")
        };
        assert!(
            diagnostics
                .nested_levels()
                .next()
                .unwrap()
                .residual()
                .is_none()
        );
    }

    #[test]
    fn conversion_accepts_exactly_the_successful_terminations() {
        for termination in [
            LinearSolveTermination::Converged,
            LinearSolveTermination::LeastSquaresConverged,
            LinearSolveTermination::ConditionLimit,
            LinearSolveTermination::IterationLimit,
            LinearSolveTermination::EvaluationLimit,
            LinearSolveTermination::Stagnation,
            LinearSolveTermination::Breakdown(LinearSolveBreakdown::ArnoldiInvariant),
            LinearSolveTermination::ExecutionStopped(scioxide_core::ExecutionStopReason::Cancelled),
        ] {
            let result = IterativeSolveResult::new(
                scioxide_core::algorithm_id!("linear", "test-solver", 1),
                Vector::from_vec(vec![1.0_f64]),
                IterativeSolveReport::new(termination, 1, None, PartialResultTrust::Admissible),
                None,
            );
            let converted = result.into_converged_solution();
            assert_eq!(
                converted.is_ok(),
                termination.is_success(),
                "{termination:?}"
            );
        }
    }

    #[test]
    fn nested_results_preserve_both_execution_stop_reasons() {
        for reason in [
            scioxide_core::ExecutionStopReason::Cancelled,
            scioxide_core::ExecutionStopReason::DeadlineReached,
        ] {
            let result = IterativeSolveResult::new(
                scioxide_core::algorithm_id!("linear", "test-solver", 1),
                Vector::from_vec(vec![1_f64]),
                IterativeSolveReport::new(
                    LinearSolveTermination::ExecutionStopped(reason),
                    1,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ),
                None,
            );
            assert!(
                matches!(result.into_converged_solution().unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason: actual } if actual == reason)
            );
        }
    }

    #[test]
    fn termination_is_the_only_success_state() {
        assert!(LinearSolveTermination::Converged.is_success());
        assert!(!LinearSolveTermination::IterationLimit.is_success());
        assert!(!LinearSolveTermination::EvaluationLimit.is_success());
        assert!(!LinearSolveTermination::Stagnation.is_success());
        assert!(
            !LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)
                .is_success()
        );
    }
}
