//! Result of a nonlinear least-squares solve.

use alloc::string::String;
use alloc::vec::Vec;

use scioxide_core::backend::try_reserve;
use scioxide_core::{RealScalar, SciOxideError};

use crate::{OptimalityDiagnostics, OptimizeTermination};

/// Holds the result of a [`LeastSquares`](crate::LeastSquares) solve: parameters,
/// residual, objective, termination and diagnostics.
///
/// When the solve stops without success, the result holds the last accepted
/// parameters. See [Termination](crate#termination).
#[derive(Debug, Clone)]
#[must_use]
pub struct LeastSquaresResult<S: RealScalar> {
    pub(crate) counts: scioxide_core::SolveCounts,
    pub(crate) algorithm: Option<scioxide_core::AlgorithmId>,
    pub(crate) trust: scioxide_core::PartialResultTrust,

    /// Last accepted parameters.
    pub(crate) x: Vec<S>,
    /// Objective at the returned parameters, in scaled arithmetic.
    pub(crate) cost: scioxide_core::Scaled<S>,
    pub(crate) condition: Option<scioxide_linalg::ConditionEstimate<S>>,
    pub(crate) damping: S,
    pub(crate) last_gain: Option<scioxide_core::Scaled<S>>,
    pub(crate) inner_termination: Option<scioxide_linalg::LinearSolveTermination>,
    pub(crate) rank: Option<scioxide_linalg::RankReport<S>>,
    pub(crate) gradient_norm: Option<scioxide_core::Scaled<S>>,
    /// Unweighted residual at the returned parameters.
    pub(crate) residual: Vec<S>,
    /// Number of iterations performed.
    pub(crate) iterations: usize,
    /// Number of residual evaluations.
    pub(crate) fun_evals: usize,
    /// Number of Jacobian evaluations.
    pub(crate) jac_evals: usize,
    /// Termination reason.
    pub(crate) termination: OptimizeTermination,
    /// Human-readable status message.
    pub(crate) message: String,
    /// Optimality diagnostics.
    pub(crate) diagnostics: OptimalityDiagnostics<S>,
}

impl<S: RealScalar> LeastSquaresResult<S> {
    /// Returns the iteration and evaluation counts of the solve. `own()` and
    /// `nested()` are disjoint; `total()` adds them with overflow checks.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the algorithm (Gauss–Newton or Levenberg–Marquardt), or `None`
    /// before the first solve.
    pub const fn algorithm(&self) -> Option<scioxide_core::AlgorithmId> {
        self.algorithm
    }
    /// Returns which checks the returned parameters and residual passed.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }

    /// Creates an empty result to pass to
    /// [`LeastSquares::solve_into`](crate::LeastSquares::solve_into).
    pub fn empty() -> Self {
        Self {
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,

            x: Vec::new(),
            cost: scioxide_core::Scaled::ZERO,
            condition: None,
            damping: S::ZERO,
            last_gain: None,
            inner_termination: None,
            rank: None,
            gradient_norm: None,
            residual: Vec::new(),
            iterations: 0,
            fun_evals: 0,
            jac_evals: 0,
            termination: OptimizeTermination::NotStarted,
            message: String::new(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        }
    }

    /// Clears the result and keeps the allocated capacity.
    pub(crate) fn reset_for_fit(&mut self) {
        let mut next = Self::empty();
        next.x = core::mem::take(&mut self.x);
        next.x.clear();
        next.residual = core::mem::take(&mut self.residual);
        next.residual.clear();
        next.message = core::mem::take(&mut self.message);
        next.message.clear();
        *self = next;
    }

    /// Reserves storage for `parameter_dimension` parameters and
    /// `residual_dimension` residual components, so that a later solve does
    /// not allocate.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(
        &mut self,
        parameter_dimension: usize,
        residual_dimension: usize,
    ) -> Result<(), SciOxideError> {
        if parameter_dimension > self.x.capacity() {
            let additional = parameter_dimension - self.x.len();
            try_reserve(
                &mut self.x,
                additional,
                scioxide_core::quantity!("least_squares_result_parameters"),
            )?;
        }
        if residual_dimension > self.residual.capacity() {
            let additional = residual_dimension - self.residual.len();
            try_reserve(
                &mut self.residual,
                additional,
                scioxide_core::quantity!("least_squares_result_residual"),
            )?;
        }
        if self.message.capacity() < 128 {
            self.message
                .try_reserve(128 - self.message.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("least_squares_result_message"),
                        128,
                    )
                })?;
        }
        Ok(())
    }

    /// Returns whether the capacity holds a result of this shape without growth.
    pub(crate) fn fits(&self, parameters: usize, residuals: usize) -> bool {
        self.x.capacity() >= parameters
            && self.residual.capacity() >= residuals
            && self.message.capacity() >= 128
    }

    /// Copies `source` into `self`; on allocation failure `self` is unchanged.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        if !self.fits(source.x.len(), source.residual.len())
            || self.message.capacity() < source.message.len()
        {
            let mut candidate = Self::empty();
            candidate.try_reserve(source.x.len(), source.residual.len())?;
            candidate.copy_fields(source);
            *self = candidate;
        } else {
            self.copy_fields(source);
        }
        Ok(())
    }

    fn copy_fields(&mut self, source: &Self) {
        self.counts = source.counts;
        self.algorithm = source.algorithm;
        self.trust = source.trust;
        self.x.clear();
        self.x.extend_from_slice(&source.x);
        self.cost = source.cost;
        self.condition = source.condition;
        self.damping = source.damping;
        self.last_gain = source.last_gain;
        self.inner_termination = source.inner_termination;
        self.rank = source.rank;
        self.gradient_norm = source.gradient_norm;
        self.residual.clear();
        self.residual.extend_from_slice(&source.residual);
        self.iterations = source.iterations;
        self.fun_evals = source.fun_evals;
        self.jac_evals = source.jac_evals;
        self.termination = source.termination;
        self.message.clear();
        self.message.push_str(&source.message);
        self.diagnostics = source.diagnostics;
    }

    /// Returns the solution, or the last accepted parameters after an
    /// unsuccessful stop.
    #[must_use]
    pub fn x(&self) -> &[S] {
        &self.x
    }

    /// Returns the objective `f(x)` at the returned parameters.
    ///
    /// # Errors
    ///
    /// Returns `NumericalInstability` (`Overflow`) if `f(x)` exceeds the range of
    /// `S`. [`Self::scaled_cost`] returns the value in that case.
    pub fn cost(&self) -> Result<S, SciOxideError> {
        self.cost.round()
    }

    /// Returns the objective as a scaled value, which also represents values
    /// outside the range of `S`.
    pub const fn scaled_cost(&self) -> scioxide_core::Scaled<S> {
        self.cost
    }

    /// Returns the numerical rank of the last factored Jacobian.
    ///
    /// `None` for LSMR solves and when convergence did not need a final
    /// factorization. Request a condition estimate with
    /// [`LeastSquaresOptions::with_condition_estimation`](crate::LeastSquaresOptions::with_condition_estimation)
    /// to always get it for dense Jacobians.
    pub const fn rank(&self) -> Option<scioxide_linalg::RankReport<S>> {
        self.rank
    }

    /// Returns the norm of the objective gradient as a scaled value, if it was
    /// evaluated.
    pub const fn scaled_gradient_norm(&self) -> Option<scioxide_core::Scaled<S>> {
        self.gradient_norm
    }

    /// Returns the condition estimate of the Jacobian, if available.
    ///
    /// For dense Jacobians it is computed only on request (off by default).
    /// For LSMR it is the estimate of the last LSMR solve, which describes the
    /// Krylov subspace explored, not the full spectrum of the Jacobian.
    pub const fn condition_estimate(&self) -> Option<scioxide_linalg::ConditionEstimate<S>> {
        self.condition
    }

    /// Returns the Levenberg–Marquardt damping parameter in scaled coordinates;
    /// zero for Gauss–Newton.
    pub const fn damping(&self) -> S {
        self.damping
    }

    /// Returns the ratio of actual to predicted reduction of the last trial step.
    /// It says nothing about the optimality of the returned parameters.
    pub const fn last_model_gain(&self) -> Option<scioxide_core::Scaled<S>> {
        self.last_gain
    }

    /// Returns the termination of the last LSMR solve; `None` for dense solves.
    pub const fn inner_termination(&self) -> Option<scioxide_linalg::LinearSolveTermination> {
        self.inner_termination
    }

    /// Returns the unweighted residual `r(x)` at the returned parameters.
    #[must_use]
    pub fn residual(&self) -> &[S] {
        &self.residual
    }

    /// Returns the number of iterations.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the number of residual evaluations.
    #[must_use]
    pub const fn fun_evals(&self) -> usize {
        self.fun_evals
    }

    /// Returns the number of Jacobian evaluations.
    #[must_use]
    pub const fn jac_evals(&self) -> usize {
        self.jac_evals
    }

    /// Returns why the solve stopped.
    #[must_use]
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }

    /// Returns a human-readable description of the termination.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns whether the termination is a success.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            self.algorithm(),
        )
        .map_or(Ok(()), Err)
    }

    /// Returns the optimality diagnostics at the returned parameters.
    pub const fn diagnostics(&self) -> OptimalityDiagnostics<S> {
        self.diagnostics
    }

    /// Returns the largest scaled constraint violation, if available.
    pub const fn primal_feasibility(&self) -> Option<S> {
        self.diagnostics.primal_feasibility()
    }

    /// Returns the Euclidean norm of the objective gradient, if available.
    pub const fn stationarity(&self) -> Option<S> {
        self.diagnostics.stationarity()
    }

    /// Returns the infinity norm of the dual-feasibility violation, if available.
    pub const fn dual_feasibility(&self) -> Option<S> {
        self.diagnostics.dual_feasibility()
    }

    /// Returns the infinity norm of the complementarity products, if available.
    pub const fn complementarity(&self) -> Option<S> {
        self.diagnostics.complementarity()
    }

    /// Returns the largest distance of an integer variable from an integer, if
    /// available.
    pub const fn integrality_violation(&self) -> Option<S> {
        self.diagnostics.integrality_violation()
    }

    /// Returns the largest scaled bound violation, if available.
    pub const fn bound_violation(&self) -> Option<S> {
        self.diagnostics.bound_violation()
    }

    /// Returns the certified absolute optimality gap, if available.
    pub const fn certified_optimality_gap(&self) -> Option<S> {
        self.diagnostics.certified_optimality_gap()
    }

    /// Splits the result into parameters, scaled objective, residual, iteration,
    /// residual-evaluation and Jacobian-evaluation counts, termination,
    /// diagnostics and message.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Vec<S>,
        scioxide_core::Scaled<S>,
        Vec<S>,
        usize,
        usize,
        usize,
        OptimizeTermination,
        OptimalityDiagnostics<S>,
        String,
    ) {
        (
            self.x,
            self.cost,
            self.residual,
            self.iterations,
            self.fun_evals,
            self.jac_evals,
            self.termination,
            self.diagnostics,
            self.message,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use alloc::borrow::ToOwned;
    use alloc::vec;

    #[test]
    fn test_least_squares_result_fields() {
        let r = LeastSquaresResult {
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            x: vec![1.0_f64, 2.0],
            cost: scioxide_core::Scaled::from_value(0.001).unwrap(),
            condition: None,
            damping: 0.0,
            last_gain: None,
            inner_termination: None,
            rank: None,
            gradient_norm: None,
            residual: vec![0.01, -0.01, 0.0],
            iterations: 15,
            fun_evals: 30,
            jac_evals: 15,
            termination: OptimizeTermination::Converged,
            message: "converged".to_owned(),
            diagnostics: OptimalityDiagnostics::unavailable(),
        };
        assert!(r.is_success());
        assert_eq!(r.iterations, 15);
    }
}
