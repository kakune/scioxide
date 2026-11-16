//! Options of the globally adaptive subdivision methods.

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, ExecutionPolicy, NormSelection, RealScalar, SolveBudgets,
    Tolerance,
};

use crate::workspace::limit;

/// Options for methods that bisect regions adaptively: error tolerance,
/// subdivision and evaluation limits, and the execution policy for the samples
/// of a region.
///
/// The defaults are absolute and relative tolerances of `√ε` with the infinity
/// norm, where `ε` is the machine epsilon of `S`; 50 subdivisions; no limit on
/// callbacks; and sequential sampling.
///
/// A scalar absolute tolerance applies to every real component of the value. A
/// vector absolute tolerance has one entry per component; a call checks its
/// length before the first callback and returns `DimensionMismatch` if it
/// differs. A tolerance below the rounding-error floor is used as given; the
/// integration then stops with [`Stagnation`](crate::IntegrationTermination::Stagnation).
///
/// Limits inherited from an enclosing [`SolveScope`](scioxide_core::SolveScope)
/// also apply. Reaching any limit ends the call with
/// [`BudgetLimit`](crate::IntegrationTermination::BudgetLimit); see
/// [Solve accounting](scioxide_core#solve-accounting).
///
/// ```
/// use scioxide_core::{Budget, ErrorTolerance, ExecutionPolicy, NormSelection};
/// use scioxide_integrate::SubdivisionOptions;
///
/// // Absolute tolerances 1e-10 and 1e-6 for two components, relative tolerance 1e-8.
/// let tolerance = ErrorTolerance::try_vector(&[1e-10, 1e-6], 1e-8, NormSelection::Infinity)?;
/// let options = SubdivisionOptions::<f64>::default()
///     .with_tolerance(tolerance)
///     .with_max_subdivisions(Budget::try_from(100)?)
///     .with_max_evaluations(Some(Budget::try_from(5_000)?))
///     .with_execution_policy(ExecutionPolicy::Sequential);
/// assert_eq!(options.max_subdivisions().get(), 100);
/// assert_eq!(
///     options.tolerance().absolute().as_vector(),
///     Some(&[1e-10, 1e-6][..])
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct SubdivisionOptions<S: RealScalar> {
    tolerance: ErrorTolerance<S>,
    max_subdivisions: Budget,
    max_evaluations: Option<Budget>,
    policy: ExecutionPolicy,
}

impl<S: RealScalar> SubdivisionOptions<S> {
    /// Sets the scalar or per-component error tolerance and the norm that
    /// combines the components.
    pub fn with_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of bisections.
    pub const fn with_max_subdivisions(mut self, max_subdivisions: Budget) -> Self {
        self.max_subdivisions = max_subdivisions;
        self
    }

    /// Sets or removes the limit on integrand callbacks.
    ///
    /// A bisection that would exceed the limit is not started.
    pub const fn with_max_evaluations(mut self, max_evaluations: Option<Budget>) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Sets how the samples of one region are distributed over workers.
    ///
    /// The result and the callback count do not depend on the policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the error tolerance.
    pub const fn tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }

    /// Returns the maximum number of bisections.
    pub const fn max_subdivisions(&self) -> Budget {
        self.max_subdivisions
    }

    /// Returns the limit on integrand callbacks, if any.
    #[must_use]
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    /// Returns the execution policy for the samples of a region.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }

    /// Subdivision limit, without the evaluation limit.
    pub(crate) const fn work_allowance(&self) -> SolveBudgets {
        limit(
            SolveBudgets::unbounded(),
            BudgetKind::Subdivision,
            Some(self.max_subdivisions),
        )
    }
    /// Subdivision and evaluation limits of one call.
    pub(crate) const fn allowance(&self) -> SolveBudgets {
        limit(
            self.work_allowance(),
            BudgetKind::Evaluation,
            self.max_evaluations,
        )
    }
}

impl<S: RealScalar> Default for SubdivisionOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: ErrorTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
                NormSelection::Infinity,
            ),
            max_subdivisions: Budget::ONE.saturating_add(49),
            max_evaluations: None,
            policy: ExecutionPolicy::Sequential,
        }
    }
}
