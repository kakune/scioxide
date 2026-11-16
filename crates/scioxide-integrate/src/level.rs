//! Options of level-based one-dimensional integration.
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionPolicy, NormSelection, RealScalar, Tolerance,
};

/// Options of [`Romberg`](crate::Romberg) and
/// [`DoubleExponentialIntegrator`](crate::DoubleExponentialIntegrator), which
/// halve the step size at each level.
///
/// The defaults are absolute and relative tolerances of `√ε` in the maximum
/// norm, where `ε` is the machine epsilon, at most 15 levels, no evaluation
/// limit and sequential evaluation. A vector absolute tolerance needs one entry
/// per real value component. [`Romberg::new`](crate::Romberg::new) accepts 6 to
/// 30 levels and
/// [`DoubleExponentialIntegrator::new`](crate::DoubleExponentialIntegrator::new)
/// accepts 4 to 15.
///
/// ```
/// use scioxide_core::{Budget, ErrorTolerance, NormSelection};
/// use scioxide_integrate::LevelOptions;
///
/// let options = LevelOptions::<f64>::default()
///     .with_tolerance(ErrorTolerance::scalar(1e-12, 0.0, NormSelection::Infinity)?)
///     .with_max_levels(Budget::try_from(20)?)
///     .with_max_evaluations(Some(Budget::try_from(1 << 16)?));
/// assert_eq!(options.max_levels().get(), 20);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct LevelOptions<S: RealScalar> {
    tolerance: ErrorTolerance<S>,
    max_levels: Budget,
    max_evaluations: Option<Budget>,
    policy: ExecutionPolicy,
}

impl<S: RealScalar> LevelOptions<S> {
    /// Sets the absolute and relative tolerance and the norm over value components.
    pub fn with_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of levels. Each level halves the step size.
    pub const fn with_max_levels(mut self, max_levels: Budget) -> Self {
        self.max_levels = max_levels;
        self
    }

    /// Sets or removes the limit on integrand evaluations.
    pub const fn with_max_evaluations(mut self, max_evaluations: Option<Budget>) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Sets how the evaluations of one level are spread over workers. The result
    /// and the evaluation count do not depend on the policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the error tolerance.
    pub const fn tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }

    /// Returns the maximum number of levels.
    pub const fn max_levels(&self) -> Budget {
        self.max_levels
    }

    /// Returns the limit on integrand evaluations, if any.
    #[must_use]
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
}

impl<S: RealScalar> Default for LevelOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: ErrorTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
                NormSelection::Infinity,
            ),
            max_levels: Budget::ONE.saturating_add(14),
            max_evaluations: None,
            policy: ExecutionPolicy::Sequential,
        }
    }
}
