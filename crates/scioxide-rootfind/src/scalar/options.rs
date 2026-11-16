//! Options and stopping tests of the scalar root solvers.
use crate::RootTermination;
use scioxide_core::{Budget, RealScalar, SciOxideError};

/// Holds the stopping tests and limits of the real and complex scalar root
/// solvers.
///
/// Every solver stops with success when it evaluates `f` exactly zero. The
/// other tests depend on the method, with `ε` the machine epsilon of `S`.
///
/// Bracketing methods, including [`SafeguardedNewton`](crate::SafeguardedNewton)
/// and [`SafeguardedHalley`](crate::SafeguardedHalley), succeed with
/// `BracketTolerance` when the bracket `[a, b]` satisfies
///
/// ```text
/// b − a ≤ τₐ + τᵣ max(|a|, |b|),
/// ```
///
/// where `τₐ` and `τᵣ` are the absolute and relative parts of the step
/// tolerance, both `4ε` by default. Open methods (Newton, Halley, secant and
/// Muller) succeed with `StepTolerance` when a step `|xₖ₊₁ − xₖ|` meets the
/// same test and `|f(xₖ₊₁)| ≤ √ε s`. The residual scale `s` is
/// [`residual_scale`](Self::residual_scale) when set, and otherwise the
/// largest `|f|` at the starting points or bracket endpoints; it stays fixed
/// during the solve, so the test is independent of the units of `f`.
///
/// [`with_residual_tolerance`](Self::with_residual_tolerance) adds a residual
/// test for every method: success with `ResidualTolerance` as soon as an
/// accepted point, including a starting point, has `|f| ≤ ρₐ + ρᵣ s`. Setting
/// only a residual scale has the same effect for bracketing methods with
/// `ρₐ = 0` and `ρᵣ = √ε`.
///
/// A solve ends without success with `Stagnation` when it cannot make
/// progress: a step is undefined (for example `f′ = 0`), repeats a point, or,
/// in an open method, meets the step tolerance without reducing `|f|`, or a
/// bracket has no floating-point number strictly inside it. It ends with
/// `IterationLimit` after [`max_iterations`](Self::max_iterations)
/// iterations (100 by default). See [Termination](crate#termination) for the
/// other statuses.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_rootfind::{Brent, ScalarFn, ScalarRootOptions, ScalarRootSolver};
///
/// // The bracket test does not depend on the scale of f: 10⁻⁹ (x − 1) has its zero at 1.
/// let f = ScalarFn::new(|x: f64| Ok(1e-9 * (x - 1.0)));
/// let result = Brent.solve(&f, Interval::new(0.0, 3.0)?, &ScalarRootOptions::default())?;
/// assert!(result.is_success());
/// assert!((result.point() - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ScalarRootOptions<S: RealScalar> {
    pub(crate) residual_scale: Option<S>,
    /// Explicit residual tolerance, if any.
    pub(crate) residual_tolerance: Option<scioxide_core::ResidualTolerance<S>>,
    pub(crate) step_tolerance: scioxide_core::StepTolerance<S>,
    /// Maximum number of iterations (default 100).
    pub(crate) max_iterations: Budget,
    evaluation_role: scioxide_core::EvaluationRole,
    budgets: scioxide_core::SolveBudgets,
}

impl<S: RealScalar> ScalarRootOptions<S> {
    /// Sets the role under which calls of `f` are counted, including samples
    /// for difference quotients.
    ///
    /// The default is [`EvaluationRole::Function`](scioxide_core::EvaluationRole::Function).
    /// Calls of `f′` count as `Jacobian` and calls of `f″` as `Hessian`. The
    /// role changes how [`SolveCounts`](scioxide_core::SolveCounts) reports the
    /// calls, for example `Event` when `f` is an event function. All roles count
    /// toward the same evaluation limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::EvaluationRole;
    /// use scioxide_rootfind::ScalarRootOptions;
    /// let options = ScalarRootOptions::<f64>::default().with_evaluation_role(EvaluationRole::Event);
    /// assert_eq!(options.evaluation_role(), EvaluationRole::Event);
    /// ```
    #[must_use]
    pub const fn with_evaluation_role(mut self, role: scioxide_core::EvaluationRole) -> Self {
        self.evaluation_role = role;
        self
    }

    /// Returns the role under which calls of `f` are counted.
    pub const fn evaluation_role(&self) -> scioxide_core::EvaluationRole {
        self.evaluation_role
    }

    /// Sets the residual scale `s`, in the units of `f`, used by the relative
    /// residual tests.
    ///
    /// For bracketing methods it also enables the residual test
    /// `|f| ≤ ρₐ + ρᵣ s`, with `ρₐ = 0` and `ρᵣ = √ε` unless a residual
    /// tolerance is set. The solve returns `InvalidInput` before calling `f`
    /// if `scale` is not finite and positive.
    #[must_use]
    pub const fn with_residual_scale(mut self, scale: S) -> Self {
        self.residual_scale = Some(scale);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        let Some(scale) = self.residual_scale else {
            return Ok(());
        };
        if !scale.is_finite() || scale <= S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("scale"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "scale must be finite and positive",
            ));
        }
        Ok(())
    }
    /// Returns the residual scale, or `None` to use the largest `|f|` at the
    /// starting points.
    pub const fn residual_scale(&self) -> Option<S> {
        self.residual_scale
    }

    /// Sets the step tolerance `τₐ + τᵣ max(|a|, |b|)` on the bracket width or
    /// the step length.
    #[must_use]
    pub const fn with_step_tolerance(mut self, tolerance: scioxide_core::StepTolerance<S>) -> Self {
        self.step_tolerance = tolerance;
        self
    }
    /// Returns the step tolerance; `τₐ = τᵣ = 4ε` by default.
    pub const fn step_tolerance(&self) -> scioxide_core::StepTolerance<S> {
        self.step_tolerance
    }

    /// Sets a residual tolerance `ρₐ + ρᵣ s`; any method then stops with
    /// success when `|f| ≤ ρₐ + ρᵣ s` at an accepted point, including a starting
    /// point.
    ///
    /// `ρₐ` is in the units of `f`. `s` is the residual scale described on
    /// [`ScalarRootOptions`].
    #[must_use]
    pub const fn with_residual_tolerance(
        mut self,
        tolerance: scioxide_core::ResidualTolerance<S>,
    ) -> Self {
        self.residual_tolerance = Some(tolerance);
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns the residual tolerance, or `None` when only the default tests
    /// apply.
    pub const fn residual_tolerance(&self) -> Option<scioxide_core::ResidualTolerance<S>> {
        self.residual_tolerance
    }

    pub(crate) fn effective_residual_tolerance(&self) -> scioxide_core::ResidualTolerance<S> {
        self.residual_tolerance
            .unwrap_or_else(crate::termination::relative_tolerance)
    }

    /// Residual and step test of a method family. Brackets test their width
    /// separately.
    #[inline(always)]
    pub(crate) fn residual_termination(
        &self,
        value: S,
        scale: S,
        small_step: bool,
        bracketed: bool,
    ) -> Result<Option<RootTermination>, SciOxideError> {
        if value == S::ZERO {
            return Ok(Some(RootTermination::ResidualTolerance));
        }
        let explicit =
            self.residual_tolerance.is_some() || (bracketed && self.residual_scale.is_some());
        if (explicit || (!bracketed && small_step))
            && crate::termination::residual_ok(
                &self.effective_residual_tolerance().tolerance(),
                value,
                scale,
            )?
        {
            Ok(Some(if explicit {
                RootTermination::ResidualTolerance
            } else {
                RootTermination::StepTolerance
            }))
        } else {
            Ok(None)
        }
    }

    /// Returns the maximum number of iterations.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }

    /// Sets limits on iterations, evaluations and nested work for each solve.
    ///
    /// [`Self::max_iterations`] still applies, so an iteration limit here only
    /// stops a solve earlier. A solve that reaches a limit after its first
    /// evaluation returns the best point so far with `WorkLimit`. A nested
    /// solve is also bound by its parent's limits (see
    /// [Solve accounting](scioxide_core#solve-accounting)).
    #[must_use]
    pub const fn with_budgets(mut self, budgets: scioxide_core::SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the work limits; unlimited by default.
    pub const fn budgets(&self) -> scioxide_core::SolveBudgets {
        self.budgets
    }
}

impl<S: RealScalar> Default for ScalarRootOptions<S> {
    fn default() -> Self {
        Self {
            residual_scale: None,
            residual_tolerance: None,
            step_tolerance: crate::termination::scalar_step_tolerance(),
            max_iterations: Budget::ONE.saturating_add(99),
            evaluation_role: scioxide_core::EvaluationRole::Function,
            budgets: scioxide_core::SolveBudgets::unbounded(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_root_options_default_f64() {
        let opts = ScalarRootOptions::<f64>::default();
        assert_eq!(opts.max_iterations().get(), 100);
    }

    #[test]
    fn test_scalar_root_options_default_f32() {
        let opts = ScalarRootOptions::<f32>::default();
        assert_eq!(opts.max_iterations().get(), 100);
    }
}
