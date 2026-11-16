//! Options for polynomial root solves.
use scioxide_core::{Budget, RealScalar};

use crate::PolynomialRootMethod;

/// Holds the method, tolerances and limits of polynomial root solves with
/// [`crate::polynomial_roots_with_options`] and
/// [`crate::polynomial_roots_into`].
///
/// | Option | Default |
/// |---|---|
/// | [`Self::method`] | [`PolynomialRootMethod::Auto`] |
/// | [`Self::backward_error_tolerance`] | `atol = rtol = √ε`: accepts backward errors up to `2√ε` |
/// | [`Self::step_tolerance`] | `atol = rtol = √ε` |
/// | [`Self::max_iterations`] | 200 per start |
/// | [`Self::restart_budget`] | 4 starts |
/// | [`Self::polish`] | `true` |
/// | [`Self::rank_threshold`] | none |
/// | [`Self::budgets`] | unbounded |
/// | [`Self::execution_policy`] | sequential |
/// | [`Self::workspace_byte_limit`] | `usize::MAX` |
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions};
///
/// // Aberth–Ehrlich iteration with one start of at most 50 iterations,
/// // without final polishing.
/// let options = PolynomialRootOptions::<f64>::default()
///     .with_method(PolynomialRootMethod::Aberth)
///     .with_max_iterations(Budget::ONE.saturating_add(49))
///     .with_restart_budget(Budget::ONE)
///     .with_polish(false);
/// assert_eq!(options.method(), PolynomialRootMethod::Aberth);
/// assert_eq!(options.max_iterations().get(), 50);
/// assert!(!options.polish());
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialRootOptions<S: RealScalar> {
    pub(crate) method: PolynomialRootMethod,
    pub(crate) backward_error_tolerance: scioxide_core::ResidualTolerance<S>,
    pub(crate) step_tolerance: scioxide_core::StepTolerance<S>,
    /// Threshold for the effective degree and rank in the report.
    pub(crate) rank_threshold: Option<scioxide_core::RankThreshold<S>>,
    /// Iteration limit of each start.
    pub(crate) max_iterations: Budget,
    /// Whether to apply Newton steps on the original coefficients.
    pub(crate) polish: bool,
    /// Number of starts, including the first.
    pub(crate) restart_budget: Budget,
    pub(crate) budgets: scioxide_core::SolveBudgets,
    pub(crate) policy: scioxide_core::ExecutionPolicy,
    pub(crate) byte_limit: usize,
}

impl<S: RealScalar> PolynomialRootOptions<S> {
    /// Sets the method that generates the root candidates.
    ///
    /// The method affects cost and robustness, not the acceptance test (see
    /// [`PolynomialRootMethod`]). [`PolynomialRootMethod::Laguerre`] and
    /// [`PolynomialRootMethod::DurandKerner`] accept only monomial input.
    pub const fn with_method(mut self, method: PolynomialRootMethod) -> Self {
        self.method = method;
        self
    }
    /// Returns the method that generates the root candidates.
    pub const fn method(&self) -> PolynomialRootMethod {
        self.method
    }
    /// Sets the iteration and evaluation limits of each solve.
    ///
    /// The limits of a parent [`scioxide_core::SolveContext`] also apply (see
    /// [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn with_budgets(mut self, budgets: scioxide_core::SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the iteration and evaluation limits of each solve.
    pub const fn budgets(&self) -> scioxide_core::SolveBudgets {
        self.budgets
    }
    /// Sets the execution policy of the companion-matrix eigenvalue stage.
    pub const fn with_execution_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy of the eigenvalue stage.
    pub const fn execution_policy(&self) -> scioxide_core::ExecutionPolicy {
        self.policy
    }
    /// Sets the largest workspace, in bytes, that a solve may require or a
    /// workspace may hold.
    pub const fn with_workspace_byte_limit(mut self, limit: usize) -> Self {
        self.byte_limit = limit;
        self
    }
    /// Returns the workspace byte limit.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }

    /// Sets the threshold that classifies coefficients as significant.
    ///
    /// A coefficient `aₖ` is significant if `|aₖ| > atol + rtol · maxⱼ |aⱼ|`.
    /// The report gives the highest significant index
    /// ([`crate::PolynomialRootReport::effective_degree`]) and the number of
    /// significant coefficients
    /// ([`crate::PolynomialRootReport::effective_rank`]). The classification is
    /// diagnostic only: coefficients are never dropped, and a polynomial of
    /// degree `n` always has `n` roots.
    #[must_use]
    pub const fn with_rank_threshold(mut self, threshold: scioxide_core::RankThreshold<S>) -> Self {
        self.rank_threshold = Some(threshold);
        self
    }
    /// Returns the coefficient significance threshold, if set.
    pub const fn rank_threshold(&self) -> Option<scioxide_core::RankThreshold<S>> {
        self.rank_threshold
    }

    /// Sets the step tolerance of the iteration and of Newton polishing.
    ///
    /// A candidate `z` has converged when its Newton correction `δ` satisfies
    /// `|δ| ≤ atol + rtol (1 + |z|)` and its residual is at the rounding level.
    #[must_use]
    pub const fn with_step_tolerance(mut self, tolerance: scioxide_core::StepTolerance<S>) -> Self {
        self.step_tolerance = tolerance;
        self
    }
    /// Returns the step tolerance.
    pub const fn step_tolerance(&self) -> scioxide_core::StepTolerance<S> {
        self.step_tolerance
    }

    /// Sets the tolerance on the normalized backward error of each root.
    ///
    /// A root `z` is accepted when its backward error `η(z)` (see
    /// [`crate::polynomial_roots_with_options`]) satisfies
    /// `η(z) ≤ atol + rtol`. The same tolerance sets the deflation threshold of
    /// the eigenvalue stage.
    #[must_use]
    pub const fn with_backward_error_tolerance(
        mut self,
        tolerance: scioxide_core::ResidualTolerance<S>,
    ) -> Self {
        self.backward_error_tolerance = tolerance;
        self
    }

    /// Sets the iteration limit of each start.
    ///
    /// Each restart, and the eigenvalue stage, gets its own limit derived from
    /// this value; polishing uses at most `min(10, max_iterations)` steps per
    /// root. To bound the total work of a solve, use [`Self::with_budgets`].
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Enables or disables Newton polishing on the original coefficients.
    #[must_use]
    pub const fn with_polish(mut self, polish: bool) -> Self {
        self.polish = polish;
        self
    }

    /// Sets the number of starts of the simultaneous iteration, including the
    /// first.
    ///
    /// Restarts use deterministic initial points, so results are reproducible.
    /// A budget of one disables restarts.
    #[must_use]
    pub const fn with_restart_budget(mut self, restart_budget: Budget) -> Self {
        self.restart_budget = restart_budget;
        self
    }

    /// Returns the backward-error tolerance.
    pub const fn backward_error_tolerance(&self) -> scioxide_core::ResidualTolerance<S> {
        self.backward_error_tolerance
    }

    /// Returns the iteration limit of each start.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }

    /// Returns whether Newton polishing is enabled.
    #[must_use]
    pub const fn polish(&self) -> bool {
        self.polish
    }

    /// Returns the number of starts.
    pub const fn restart_budget(&self) -> Budget {
        self.restart_budget
    }
}

impl<S: RealScalar> Default for PolynomialRootOptions<S> {
    fn default() -> Self {
        Self {
            method: PolynomialRootMethod::Auto,
            budgets: scioxide_core::SolveBudgets::unbounded(),
            policy: scioxide_core::ExecutionPolicy::Sequential,
            byte_limit: usize::MAX,
            rank_threshold: None,
            backward_error_tolerance: scioxide_core::ResidualTolerance::default(),
            step_tolerance: scioxide_core::StepTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(199),
            polish: true,
            restart_budget: Budget::ONE.saturating_add(3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_root_options_default() {
        let opts = PolynomialRootOptions::<f64>::default();
        assert_eq!(opts.method(), PolynomialRootMethod::Auto);
        assert!(opts.polish());
        assert_eq!(opts.restart_budget().get(), 4);
    }
}
