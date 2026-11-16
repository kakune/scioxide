//! Options of the dense nonlinear-system solvers.
use crate::{BacktrackingLineSearch, RootLineSearch};
use scioxide_core::{
    Budget, ExecutionPolicy, NumericScalar, RealScalar, ResidualTolerance, SciOxideError,
    SolveBudgets, StepTolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{GmresSolver, IterativeLinearSolver};

#[derive(Clone, Debug)]
pub(crate) struct Settings<S: RealScalar> {
    pub(crate) strategy: DerivativeStrategy,
    pub(crate) scale: Option<S>,
    pub(crate) residual: ResidualTolerance<S>,
    pub(crate) step: StepTolerance<S>,
    pub(crate) iterations: Budget,
    pub(crate) budgets: SolveBudgets,
    pub(crate) policy: ExecutionPolicy,
    pub(crate) byte_limit: usize,
}
impl<S: RealScalar> Default for Settings<S> {
    fn default() -> Self {
        Self {
            strategy: DerivativeStrategy::AnalyticOrForward,
            scale: None,
            residual: crate::termination::relative_tolerance(),
            step: StepTolerance::default(),
            iterations: Budget::ONE.saturating_add(99),
            budgets: SolveBudgets::unbounded(),
            policy: ExecutionPolicy::Sequential,
            byte_limit: usize::MAX,
        }
    }
}
impl<S: RealScalar> Settings<S> {
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        if self.scale.is_some_and(|s| !s.is_finite() || s <= S::ZERO) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options.residual_scale"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "residual scale must be positive and finite",
            ));
        }
        Ok(())
    }
    /// Borrow the kernel view with the solver's optional providers.
    pub(crate) fn view<'a>(
        &self,
        line_search: Option<&'a dyn RootLineSearch<S>>,
        linear_solver: Option<&'a dyn IterativeLinearSolver<S>>,
    ) -> SystemRootOptions<'a, S>
    where
        S: NumericScalar<Real = S>,
    {
        SystemRootOptions {
            policy: self.policy,
            residual_scale: self.scale,
            residual_tolerance: self.residual,
            step_tolerance: self.step,
            max_iterations: self.iterations,
            line_search,
            linear_solver,
        }
    }
}
/// Declares the builder setters and getters of the `settings` field of a dense-system options type.
macro_rules! settings_accessors {
    () => {
        /// Sets how the Jacobian is obtained (default
        /// [`DerivativeStrategy::AnalyticOrForward`]: the analytic Jacobian if the
        /// problem supplies one, otherwise forward differences). Finite-difference
        /// strategies cost `n` (forward) or `2n` (central) residual evaluations
        /// per Jacobian. [`DerivativeStrategy::Analytic`] rejects a problem
        /// without an analytic Jacobian.
        pub const fn with_derivative_strategy(mut self, value: DerivativeStrategy) -> Self {
            self.settings.strategy = value;
            self
        }
        /// Returns how the Jacobian is obtained.
        pub const fn derivative_strategy(&self) -> DerivativeStrategy {
            self.settings.strategy
        }
        /// Sets the reference scale `s` of the relative residual test. By default
        /// `s = ‖F(x₀)‖₂`. The scale must be positive and finite; the solver's
        /// `new` checks it.
        pub const fn with_residual_scale(mut self, value: S) -> Self {
            self.settings.scale = Some(value);
            self
        }
        /// Returns the residual reference scale, or `None` for `‖F(x₀)‖₂`.
        pub const fn residual_scale(&self) -> Option<S> {
            self.settings.scale
        }
        /// Sets the residual tolerance. The solve succeeds when
        /// `‖F(x)‖₂ ≤ atol + rtol · s`, with `s` the residual reference scale.
        /// The default is `atol = 0`, `rtol = √ε`, which does not depend on the
        /// units of `F`.
        pub const fn with_residual_tolerance(mut self, value: ResidualTolerance<S>) -> Self {
            self.settings.residual = value;
            self
        }
        /// Returns the residual tolerance.
        pub const fn residual_tolerance(&self) -> ResidualTolerance<S> {
            self.settings.residual
        }
        /// Sets the step tolerance (default `atol = rtol = √ε`). A step `Δx` with
        /// `‖Δx‖₂ ≤ atol + rtol · max(1, ‖x‖₂)` that does not reduce `‖F‖₂` stops
        /// the solve with `Stagnation`. A zero tolerance disables the test.
        pub const fn with_step_tolerance(mut self, value: StepTolerance<S>) -> Self {
            self.settings.step = value;
            self
        }
        /// Returns the step tolerance.
        pub const fn step_tolerance(&self) -> StepTolerance<S> {
            self.settings.step
        }
        /// Sets the iteration limit (default 100).
        pub const fn with_max_iterations(mut self, value: Budget) -> Self {
            self.settings.iterations = value;
            self
        }
        /// Returns the iteration limit.
        pub const fn max_iterations(&self) -> Budget {
            self.settings.iterations
        }
        /// Sets limits on iterations, evaluations and nested work (default
        /// unlimited). See [Solve accounting](scioxide_core#solve-accounting).
        pub const fn with_budgets(mut self, value: SolveBudgets) -> Self {
            self.settings.budgets = value;
            self
        }
        /// Returns the work limits.
        pub const fn budgets(&self) -> SolveBudgets {
            self.settings.budgets
        }
        /// Sets the execution policy of the linear algebra (default
        /// `Sequential`).
        pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
            self.settings.policy = value;
            self
        }
        /// Returns the execution policy.
        pub const fn execution_policy(&self) -> ExecutionPolicy {
            self.settings.policy
        }
        /// Sets the largest workspace size in bytes (default unlimited). A larger
        /// requirement is a `ResourceLimit` error.
        pub const fn with_workspace_byte_limit(mut self, value: usize) -> Self {
            self.settings.byte_limit = value;
            self
        }
        /// Returns the workspace byte limit.
        pub const fn workspace_byte_limit(&self) -> usize {
            self.settings.byte_limit
        }
    };
}

/// Kernel view of the options; the line search and linear solver stay in the solver.
pub(crate) struct SystemRootOptions<'a, S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) residual_scale: Option<S>,
    pub(crate) residual_tolerance: ResidualTolerance<S>,
    pub(crate) step_tolerance: StepTolerance<S>,
    pub(crate) max_iterations: Budget,
    pub(crate) line_search: Option<&'a dyn RootLineSearch<S>>,
    pub(crate) linear_solver: Option<&'a dyn IterativeLinearSolver<S>>,
    pub(crate) policy: ExecutionPolicy,
}
impl<S: RealScalar + NumericScalar<Real = S>> SystemRootOptions<'_, S> {
    pub(crate) const fn residual_scale(&self) -> Option<S> {
        self.residual_scale
    }
}

/// Holds the options of [`crate::NewtonSystem`].
///
/// By default there is no line search, the step comes from an LU
/// factorization, and the Jacobian is the analytic one if the problem supplies
/// it and a forward-difference approximation otherwise.
#[derive(Clone, Debug)]
pub struct NewtonSystemOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = BacktrackingLineSearch<S>,
    I = GmresSolver<S>,
> {
    pub(crate) settings: Settings<S>,
    line_search: Option<L>,
    linear_solver: Option<I>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for NewtonSystemOptions<S> {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            line_search: None,
            linear_solver: None,
        }
    }
}
impl<S, L, I> NewtonSystemOptions<S, L, I>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    settings_accessors!();
    /// Sets a line search, usually [`crate::BacktrackingLineSearch`]. Without one,
    /// every step is a full step.
    pub fn with_line_search<N>(self, search: N) -> NewtonSystemOptions<S, N, I> {
        NewtonSystemOptions {
            settings: self.settings,
            line_search: Some(search),
            linear_solver: self.linear_solver,
        }
    }
    /// Solves the Newton equations with an iterative linear solver, such as
    /// GMRES, instead of an LU factorization.
    pub fn with_linear_solver<N>(self, solver: N) -> NewtonSystemOptions<S, L, N> {
        NewtonSystemOptions {
            settings: self.settings,
            linear_solver: Some(solver),
            line_search: self.line_search,
        }
    }
}
impl<S, L, I> NewtonSystemOptions<S, L, I>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: RootLineSearch<S>,
    I: IterativeLinearSolver<S>,
{
    pub(crate) fn view(&self) -> SystemRootOptions<'_, S> {
        self.settings.view(
            self.line_search
                .as_ref()
                .map(|s| s as &dyn RootLineSearch<S>),
            self.linear_solver
                .as_ref()
                .map(|s| s as &dyn IterativeLinearSolver<S>),
        )
    }
}

/// Holds the options of [`crate::Broyden`].
///
/// Broyden's method solves with its own Jacobian model, so it takes no
/// iterative linear solver. By default there is no line search.
///
/// ```compile_fail
/// use scioxide_core::{Budget, ResidualTolerance};
/// use scioxide_linalg::GmresSolver;
/// use scioxide_rootfind::BroydenOptions;
/// let linear = GmresSolver::<f64>::new(ResidualTolerance::new(1e-8, 0.0).unwrap(), Budget::ONE);
/// let _ = BroydenOptions::<f64>::default().with_linear_solver(linear);
/// ```
#[derive(Clone, Debug)]
pub struct BroydenOptions<S: RealScalar + NumericScalar<Real = S>, L = BacktrackingLineSearch<S>> {
    pub(crate) settings: Settings<S>,
    line_search: Option<L>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for BroydenOptions<S> {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            line_search: None,
        }
    }
}
impl<S, L> BroydenOptions<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    settings_accessors!();
    /// Sets a line search, usually [`crate::BacktrackingLineSearch`]. Without one,
    /// every step is a full step.
    pub fn with_line_search<N>(self, search: N) -> BroydenOptions<S, N> {
        BroydenOptions {
            settings: self.settings,
            line_search: Some(search),
        }
    }
}
impl<S, L> BroydenOptions<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: RootLineSearch<S>,
{
    pub(crate) fn view(&self) -> SystemRootOptions<'_, S> {
        self.settings.view(
            self.line_search
                .as_ref()
                .map(|s| s as &dyn RootLineSearch<S>),
            None,
        )
    }
}

/// Holds the options of [`crate::TrustRegionSystem`].
///
/// The trust region controls the step length, so these options take neither a
/// line search nor an iterative linear solver.
///
/// ```compile_fail
/// use scioxide_rootfind::TrustRegionSystemOptions;
/// let _ = TrustRegionSystemOptions::<f64>::default().with_linear_solver(());
/// ```
///
/// ```compile_fail
/// use scioxide_rootfind::{BacktrackingLineSearch, TrustRegionSystemOptions};
/// let _ = TrustRegionSystemOptions::<f64>::default()
///     .with_line_search(BacktrackingLineSearch::<f64>::default());
/// ```
#[derive(Clone, Debug)]
pub struct TrustRegionSystemOptions<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) settings: Settings<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for TrustRegionSystemOptions<S> {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
        }
    }
}
impl<S> TrustRegionSystemOptions<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    settings_accessors!();
    pub(crate) fn view(&self) -> SystemRootOptions<'_, S> {
        self.settings.view(None, None)
    }
}
