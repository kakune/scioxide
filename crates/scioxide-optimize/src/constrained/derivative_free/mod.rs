//! Derivative-free trust-region methods for nonlinear programs: COBYLA and COBYQA.
pub(super) mod engine;
pub(super) mod lp;
mod lp_algebra;

/// Declares the public types of one derivative-free solver over the shared engine.
macro_rules! solver {
    ($solver:ident, $options:ident, $workspace:ident, $requirements:ident, $diagnostics:ident, $kind:ident, $description:literal) => {
        #[doc = concat!("Sets the trust radii, tolerances and limits of [`", stringify!($solver), "`].\n\nThe radii are in the units of `x`. The feasibility tolerance applies to the constraint rows after scaling, as described in [Termination](crate#termination). [`", stringify!($solver), "::new`] checks the radii.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $options<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> {
            inner: $crate::constrained::derivative_free::engine::DerivativeFreeOptions<S>,
        }
        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> Default for $options<S> {
            fn default() -> Self {
                Self { inner: Default::default() }
            }
        }
        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> $options<S> {
            /// Sets the initial and final trust-region radii (default 1 and
            /// 10⁻⁴). They must satisfy `0 < final ≤ initial`.
            #[must_use]
            pub const fn with_radii(self, initial: S, final_radius: S) -> Self {
                Self { inner: self.inner.with_radii(initial, final_radius) }
            }
            /// Sets the feasibility tolerance for the scaled constraint rows.
            #[must_use]
            pub const fn with_feasibility_tolerance(self, value: scioxide_core::FeasibilityTolerance<S>) -> Self {
                Self { inner: self.inner.with_feasibility_tolerance(value) }
            }
            /// Sets the threshold of the rank test on the interpolation points.
            #[must_use]
            pub const fn with_rank_threshold(self, value: scioxide_core::RankThreshold<S>) -> Self {
                Self { inner: self.inner.with_rank_threshold(value) }
            }
            /// Sets the limit on trust-region steps and geometry steps (default 1000).
            #[must_use]
            pub const fn with_max_iterations(self, value: scioxide_core::Budget) -> Self {
                Self { inner: self.inner.with_max_iterations(value) }
            }
            /// Sets the limit on objective evaluations, including the initial
            /// interpolation points and geometry steps (default 1000).
            #[must_use]
            pub const fn with_max_evaluations(self, value: scioxide_core::Budget) -> Self {
                Self { inner: self.inner.with_max_evaluations(value) }
            }
            /// Sets a wall-clock limit. The execution context must provide a clock.
            #[must_use]
            pub const fn with_time_limit(self, value: core::time::Duration) -> Self {
                Self { inner: self.inner.with_time_limit(value) }
            }
            /// Returns the initial and final trust-region radii.
            pub const fn radii(&self) -> (S, S) {
                self.inner.radii()
            }
            /// Returns the feasibility tolerance.
            pub const fn feasibility_tolerance(&self) -> scioxide_core::FeasibilityTolerance<S> {
                self.inner.feasibility_tolerance()
            }
            /// Returns the rank threshold.
            pub const fn rank_threshold(&self) -> scioxide_core::RankThreshold<S> {
                self.inner.rank_threshold()
            }
            /// Returns the iteration limit.
            pub const fn max_iterations(&self) -> scioxide_core::Budget {
                self.inner.max_iterations()
            }
            /// Returns the evaluation limit.
            pub const fn max_evaluations(&self) -> scioxide_core::Budget {
                self.inner.max_evaluations()
            }
            /// Returns the wall-clock limit, if set.
            pub const fn time_limit(&self) -> Option<core::time::Duration> {
                self.inner.time_limit()
            }
        }

        /// Reports the trust region, merit penalty and interpolation model at the
        /// end of a derivative-free solve.
        ///
        /// Read it through [`OptimizeResult::method_diagnostics`](crate::OptimizeResult::method_diagnostics).
        /// The model quantities are estimates from interpolation; they are not
        /// derivatives of the original problem.
        #[derive(Debug, Clone, Copy)]
        pub struct $diagnostics<S: scioxide_core::RealScalar> {
            radius: S,
            resolution: S,
            penalty: S,
            model_error: Option<S>,
            model_stationarity: Option<S>,
            geometry_repairs: usize,
            poised: bool,
        }
        impl<S: scioxide_core::RealScalar> $diagnostics<S> {
            /// Returns the last trust-region radius.
            pub const fn radius(&self) -> S {
                self.radius
            }
            /// Returns the last resolution, the lower limit of the trust-region
            /// radius at that stage.
            pub const fn resolution(&self) -> S {
                self.resolution
            }
            /// Returns the penalty parameter of the merit function.
            pub const fn penalty(&self) -> S {
                self.penalty
            }
            /// Returns the largest difference between the model and the scaled
            /// objective or constraint values at the last trial point, if any.
            pub const fn model_error(&self) -> Option<S> {
                self.model_error
            }
            /// Returns the decrease of the model merit function predicted by the
            /// trust-region LP step, divided by the LP radius, or `None` before
            /// the first step.
            ///
            /// The LP radius is the trust radius `Δ` for COBYLA and the
            /// resolution `ρ` for COBYQA, and the merit function is the
            /// normalized objective plus the penalty times the normalized
            /// constraint violation. The value tends to zero near a KKT point of
            /// the model; it is not a KKT residual.
            pub const fn model_stationarity(&self) -> Option<S> {
                self.model_stationarity
            }
            /// Returns the number of geometry steps that replace one
            /// interpolation point plus the number of rebuilds of the whole
            /// interpolation set, including the initial one.
            pub const fn geometry_repairs(&self) -> usize {
                self.geometry_repairs
            }
            /// Returns `true` if the interpolation points passed the rank test.
            pub const fn poised(&self) -> bool {
                self.poised
            }
        }
        impl<S: scioxide_core::RealScalar>
            From<$crate::constrained::derivative_free::engine::DerivativeFreeDiagnostics<S>>
            for $diagnostics<S>
        {
            fn from(d: $crate::constrained::derivative_free::engine::DerivativeFreeDiagnostics<S>) -> Self {
                Self {
                    radius: d.radius,
                    resolution: d.resolution,
                    penalty: d.penalty,
                    model_error: d.model_error,
                    model_stationarity: d.model_stationarity,
                    geometry_repairs: d.geometry_repairs,
                    poised: d.poised,
                }
            }
        }

        #[doc = concat!("Describes the workspace size of a [`", stringify!($solver), "`] solve.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $requirements {
            inner: $crate::constrained::derivative_free::engine::DerivativeFreeShape,
        }
        impl $requirements {
            /// Returns the number of variables.
            pub const fn dimension(self) -> usize {
                self.inner.dimension
            }
            /// Returns the number of nonlinear rows. Bounds are not counted.
            pub const fn constraints(self) -> usize {
                self.inner.constraints
            }
            /// Returns the number of nonlinear equality rows.
            pub const fn equalities(self) -> usize {
                self.inner.equalities
            }
            /// Returns the number of variables that the bounds do not fix.
            pub const fn free_variables(self) -> usize {
                self.inner.free
            }
        }
        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>>
            $crate::attempt::Shape<$crate::constrained::derivative_free::engine::DerivativeFreeStorage<S>>
            for $requirements
        {
            fn fits(&self, storage: &$crate::constrained::derivative_free::engine::DerivativeFreeStorage<S>) -> bool {
                storage.fits(self.inner)
            }
            fn prepare(
                &self,
                storage: &mut $crate::constrained::derivative_free::engine::DerivativeFreeStorage<S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                storage.reserve(self.inner)
            }
        }

        $crate::attempt::local_workspace!(
            #[doc = concat!("Stores the interpolation points, models and trust-region subproblem of a [`", stringify!($solver), "`] solve.")]
            $workspace,
            $crate::constrained::derivative_free::engine::DerivativeFreeStorage<S>,
            $requirements
        );

        #[doc = concat!($description, "\n\n# Examples\n\n```\nuse scioxide_optimize::{ConstrainedProblem, Constraints, ObjectiveFn, ", stringify!($solver), "};\n\n// minimize (x − 2)²; the minimizer is x = 2\nlet f = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 2.0).powi(2)));\nlet constraints = Constraints::new(1)?;\nlet problem = ConstrainedProblem::new(&f, &constraints)?;\nlet result = ", stringify!($solver), "::default().minimize(&problem, &[0.0])?;\nassert!((result.x()[0] - 2.0).abs() < 1e-3);\n// No gradient is available, so no stationarity measure is reported.\nassert_eq!(result.stationarity(), None);\n# Ok::<(), scioxide_core::SciOxideError>(())\n```")]
        #[derive(Debug, Clone, Copy)]
        pub struct $solver<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> {
            options: $options<S>,
        }
        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> Default for $solver<S> {
            fn default() -> Self {
                Self { options: $options::default() }
            }
        }
        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> $solver<S> {
            /// Creates a solver with `options`.
            ///
            /// # Errors
            ///
            /// - `NonFinite` if a radius is NaN or infinite.
            /// - `InvalidInput` if a radius is not positive or the final radius
            ///   exceeds the initial radius.
            pub fn new(options: $options<S>) -> Result<Self, scioxide_core::SciOxideError> {
                options.inner.validate()?;
                Ok(Self { options })
            }
            /// Returns the options.
            pub const fn options(&self) -> &$options<S> {
                &self.options
            }
            const fn engine(&self) -> $crate::constrained::derivative_free::engine::DerivativeFree<S> {
                $crate::constrained::derivative_free::engine::DerivativeFree::new(
                    self.options.inner,
                    $crate::constrained::derivative_free::engine::ModelKind::$kind,
                )
            }
            /// Returns the workspace size needed for `problem`.
            ///
            /// # Errors
            ///
            /// This method does not fail.
            pub fn workspace_requirements(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
            ) -> Result<$requirements, scioxide_core::SciOxideError> {
                Ok($requirements { inner: self.engine().shape(problem) })
            }
            /// Allocates a workspace for `problem`.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
            pub fn workspace(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
            ) -> Result<$workspace<S>, scioxide_core::SciOxideError> {
                let mut workspace = $workspace::new();
                workspace.try_reserve(&self.workspace_requirements(problem)?)?;
                Ok(workspace)
            }
            /// Minimizes `problem` from `x0` and writes the result to `output`.
            ///
            /// The starting point is projected into the bounds first. See
            /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `x0` does not have the problem's dimension.
            /// - `NonFinite` if `x0` contains NaN or infinity, or if the objective
            ///   or a constraint is not finite at the start or an initial
            ///   interpolation point.
            /// - `InvalidInput` if the projected start is outside the objective
            ///   domain.
            /// - `CapabilityUnavailable` if a time limit is set and `scope` has no
            ///   clock.
            /// - Errors from the callbacks, unchanged.
            /// - `ConvergenceFailure` or `ExecutionStopped` if a limit,
            ///   cancellation or deadline stops the solve before the start is
            ///   evaluated, as described in [Termination](crate#termination).
            pub fn minimize_into<'scope, 'capability>(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
                x0: &[S],
                output: &mut $crate::OptimizeResult<S>,
                workspace: &mut $workspace<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                workspace.shell.reject(scioxide_core::AttemptKind::Call);
                let scope = scope.into();
                $crate::utils::validate_x0(problem.objective(), x0)?;
                workspace.try_reserve(&self.workspace_requirements(problem)?)?;
                let engine = self.engine();
                $crate::attempt::preflight(&scope, self.options.inner.time)?;
                let entry = $crate::attempt::Entry::Call {
                    identity: engine.identity(problem),
                    budgets: scioxide_core::SolveBudgets::unbounded().with_limit(
                        scioxide_core::BudgetKind::Iteration,
                        self.options.inner.iterations,
                    ),
                };
                $crate::attempt::minimize(
                    &mut workspace.shell,
                    &mut workspace.storage,
                    &mut workspace.staged,
                    output,
                    scope,
                    entry,
                    |context, storage, staged| {
                        engine.minimize_in_context(problem, x0, staged, storage, context)
                    },
                )
            }
        }
        impl<S> $solver<S>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
        {
            /// Minimizes the objective with this method's settings and owned workspace.
            ///
            /// Returns the complete result, including unsuccessful termination. Call
            /// `require_success()` to require the stopping test to pass.
            ///
            /// # Errors
            ///
            /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
            pub fn minimize(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
                x0: &[S],
            ) -> Result<$crate::OptimizeResult<S>, scioxide_core::SciOxideError> {
                let mut workspace = self.workspace(problem)?;
                let mut output = $crate::OptimizeResult::empty();
                self.minimize_into(
                    problem,
                    x0,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
        }

        impl<'a, S> $crate::Optimizer<S, $crate::ConstrainedProblem<'a, S>> for $solver<S>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
        {
            fn minimize(
                &self,
                problem: &$crate::ConstrainedProblem<'a, S>,
                x0: &[S],
            ) -> Result<$crate::OptimizeResult<S>, scioxide_core::SciOxideError> {
                self.minimize(problem, x0)
            }
        }
    };
}
pub(super) use solver;
