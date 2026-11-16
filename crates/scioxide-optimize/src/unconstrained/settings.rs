//! Configuration shared by the line-search minimizers.

use core::time::Duration;

use scioxide_core::{Budget, GradientTolerance, NumericScalar, RealScalar};
use scioxide_differentiate::DerivativeStrategy;

/// Stopping, derivative and line-search settings read by every line-search minimizer.
#[derive(Debug, Clone)]
pub(crate) struct LineSearchSettings<S: RealScalar + NumericScalar<Real = S>, L> {
    pub(crate) gradient_tolerance: GradientTolerance<S>,
    pub(crate) max_iterations: Budget,
    pub(crate) line_search: L,
    pub(crate) derivative_strategy: DerivativeStrategy,
    pub(crate) time_limit: Option<Duration>,
}

impl<S: RealScalar + NumericScalar<Real = S>, L> LineSearchSettings<S, L> {
    pub(crate) fn new(line_search: L) -> Self {
        Self {
            gradient_tolerance: GradientTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(999),
            line_search,
            derivative_strategy: DerivativeStrategy::AnalyticOrCentral,
            time_limit: None,
        }
    }

    pub(crate) fn with_line_search<L2>(self, line_search: L2) -> LineSearchSettings<S, L2> {
        LineSearchSettings {
            gradient_tolerance: self.gradient_tolerance,
            max_iterations: self.max_iterations,
            line_search,
            derivative_strategy: self.derivative_strategy,
            time_limit: self.time_limit,
        }
    }
}

/// Generate the shared setters and accessors over a `settings` field.
macro_rules! line_search_settings {
    ($options:ident) => {
        impl<S, L> $options<S, L>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
        {
            /// Sets the gradient tolerance.
            #[must_use]
            pub fn with_gradient_tolerance(
                mut self,
                gradient_tolerance: scioxide_core::GradientTolerance<S>,
            ) -> Self {
                self.settings.gradient_tolerance = gradient_tolerance;
                self
            }

            /// Sets the iteration limit.
            #[must_use]
            pub fn with_max_iterations(mut self, max_iterations: scioxide_core::Budget) -> Self {
                self.settings.max_iterations = max_iterations;
                self
            }

            /// Sets the order in which derivative sources are tried.
            #[must_use]
            pub fn with_derivative_strategy(
                mut self,
                strategy: scioxide_differentiate::DerivativeStrategy,
            ) -> Self {
                self.settings.derivative_strategy = strategy;
                self
            }

            /// Sets a wall-clock time limit. The execution context must provide a clock.
            #[must_use]
            pub fn with_time_limit(mut self, time_limit: core::time::Duration) -> Self {
                self.settings.time_limit = Some(time_limit);
                self
            }

            /// Returns the gradient tolerance.
            pub const fn gradient_tolerance(&self) -> scioxide_core::GradientTolerance<S> {
                self.settings.gradient_tolerance
            }

            /// Returns the iteration limit.
            pub const fn max_iterations(&self) -> scioxide_core::Budget {
                self.settings.max_iterations
            }

            /// Returns the line search.
            pub const fn line_search(&self) -> &L {
                &self.settings.line_search
            }

            /// Returns the order in which derivative sources are tried.
            pub const fn derivative_strategy(&self) -> scioxide_differentiate::DerivativeStrategy {
                self.settings.derivative_strategy
            }

            /// Returns the time limit, if set.
            pub const fn time_limit(&self) -> Option<core::time::Duration> {
                self.settings.time_limit
            }
        }
    };
}
pub(crate) use line_search_settings;

/// Iteration allowance of one local epoch.
pub(crate) fn iteration_budgets(iterations: Budget) -> scioxide_core::SolveBudgets {
    scioxide_core::SolveBudgets::unbounded()
        .with_limit(scioxide_core::BudgetKind::Iteration, iterations)
}
