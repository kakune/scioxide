//! Sequential quadratic programming for nonlinear programs with bounds.
mod subproblem;
use super::nonlinear::Observation;
use crate::attempt::{Entry, Shape};
use crate::context::SolveSession;
use crate::derivatives::Differences;
use crate::{
    ConstrainedProblem, Constraints, OptimalityDiagnostics, OptimalityTolerances, OptimizeResult,
    OptimizeTermination,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::time::Duration;
use scioxide_core::{
    AttemptKind, Budget, BudgetKind, NumericScalar, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveScope, SolveState, StepTolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;
pub use subproblem::{
    DenseSqpSubproblem, SqpSubproblem, SqpSubproblemObjective, SqpSubproblemProvider,
    SqpSubproblemReport,
};

pub(super) use crate::storage::resize;
pub(crate) use crate::validation::invalid;

/// Stores a step, Lagrange multipliers and the active set of an [`Sqp`] iteration.
///
/// [`SqpWorkspace::state`] returns the state at the last accepted point, and
/// [`SqpWorkspace::set_warm_start`] starts the next solve from a saved state.
/// An [`SqpSubproblemProvider`] writes its solution into this type.
///
/// Equality multipliers have any sign. Multipliers of inequality rows and of
/// bounds are nonnegative. On a fixed variable (`lᵢ = uᵢ`) without an available
/// derivative, the bound multipliers refer to the reduced problem.
#[derive(Debug, Clone)]
pub struct SqpState<S: RealScalar> {
    pub(super) step: Vec<S>,
    pub(super) multipliers: Vec<S>,
    pub(super) lower: Vec<S>,
    pub(super) upper: Vec<S>,
    pub(super) active: Vec<usize>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for SqpState<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SqpState<S> {
    /// Creates an empty state.
    pub const fn new() -> Self {
        Self {
            step: Vec::new(),
            multipliers: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            active: Vec::new(),
        }
    }
    /// Resizes the state for `n` variables and `m` nonlinear rows.
    ///
    /// A change of either size clears the active set.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn reserve(&mut self, n: usize, m: usize) -> Result<(), SciOxideError> {
        if self.step.len() != n || self.multipliers.len() != m {
            self.active.clear();
        }
        for v in [&mut self.step, &mut self.lower, &mut self.upper] {
            resize(v, n, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        }
        resize(
            &mut self.multipliers,
            m,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        let count = scioxide_core::backend::checked_add(
            m,
            scioxide_core::backend::checked_product(
                &[2, n],
                scioxide_core::quantity!("sqp_active_capacity"),
            )?,
            scioxide_core::quantity!("sqp_active_capacity"),
        )?;
        if count > self.active.capacity() {
            let extra = count - self.active.len();
            scioxide_core::backend::try_reserve(
                &mut self.active,
                extra,
                scioxide_core::quantity!("sqp_active_state"),
            )?;
        }
        Ok(())
    }
    /// Returns the step `d` from the linearization point.
    pub fn step(&self) -> &[S] {
        &self.step
    }
    /// Returns the multipliers of the nonlinear rows, equality rows first.
    ///
    /// The multipliers `λ` of the rows `c`, `λˡ` of the lower bounds and `λᵘ`
    /// of the upper bounds satisfy the stationarity condition
    /// `∇f(x) + Σᵢ λᵢ ∇cᵢ(x) − λˡ + λᵘ = 0` at a KKT point. The multipliers of
    /// inequality rows `gᵢ(x) ≤ 0` and of the bounds are nonnegative.
    pub fn multipliers(&self) -> &[S] {
        &self.multipliers
    }
    /// Returns the lower-bound multipliers, one per variable.
    pub fn lower_multipliers(&self) -> &[S] {
        &self.lower
    }
    /// Returns the upper-bound multipliers, one per variable.
    pub fn upper_multipliers(&self) -> &[S] {
        &self.upper
    }
    /// Returns the indices of the active constraints.
    ///
    /// Index `i < m` is nonlinear row `i`. Index `m + 2j` is the lower bound and
    /// `m + 2j + 1` the upper bound of variable `j`, where `m` is the number of
    /// nonlinear rows.
    pub fn active_constraints(&self) -> &[usize] {
        &self.active
    }
    /// Returns the step, multiplier, lower-bound, upper-bound and active-set
    /// buffers for a provider to write its solution.
    #[expect(clippy::type_complexity, reason = "preserve the public buffer tuple")]
    pub fn buffers_mut(&mut self) -> (&mut [S], &mut [S], &mut [S], &mut [S], &mut Vec<usize>) {
        (
            &mut self.step,
            &mut self.multipliers,
            &mut self.lower,
            &mut self.upper,
            &mut self.active,
        )
    }
    fn clear(&mut self) {
        self.step.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        self.multipliers
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        self.lower.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        self.upper.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        self.active.clear();
    }
    pub(super) fn copy_from(&mut self, other: &Self) {
        self.step.copy_from_slice(&other.step);
        self.multipliers.copy_from_slice(&other.multipliers);
        self.lower.copy_from_slice(&other.lower);
        self.upper.copy_from_slice(&other.upper);
        self.active.clear();
        self.active.extend_from_slice(&other.active);
    }
}
/// Sets the tolerances and iteration limits of [`Sqp`].
///
/// The defaults are the default [`OptimalityTolerances`], at most 1000
/// iterations, 1000 active-set iterations per subproblem and 20 restoration iterations.
/// See [Termination](crate#termination) for how the tolerances are scaled.
#[derive(Debug, Clone, Copy)]
pub struct SqpOptions<S: RealScalar + NumericScalar<Real = S>> {
    step_tolerance: StepTolerance<S>,
    tolerances: OptimalityTolerances<S>,
    max_iterations: Budget,
    max_subproblem_iterations: Budget,
    max_restoration_iterations: Budget,
    derivative_strategy: DerivativeStrategy,
    time_limit: Option<Duration>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for SqpOptions<S> {
    fn default() -> Self {
        Self {
            tolerances: OptimalityTolerances::default(),
            step_tolerance: StepTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(999),
            max_subproblem_iterations: Budget::ONE.saturating_add(999),
            max_restoration_iterations: Budget::ONE.saturating_add(19),
            derivative_strategy: DerivativeStrategy::default(),
            time_limit: None,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> SqpOptions<S> {
    /// Sets the step tolerance. The solver stops with `StepSmall` after two
    /// consecutive feasible steps below it.
    #[must_use]
    pub const fn with_step_tolerance(mut self, tolerance: StepTolerance<S>) -> Self {
        self.step_tolerance = tolerance;
        self
    }
    /// Sets the convergence tolerances.
    #[must_use]
    pub const fn with_tolerances(mut self, tolerances: OptimalityTolerances<S>) -> Self {
        self.tolerances = tolerances;
        self
    }
    /// Sets the iteration limit (default 1000).
    #[must_use]
    pub const fn with_max_iterations(mut self, budget: Budget) -> Self {
        self.max_iterations = budget;
        self
    }
    /// Sets the iteration limit of each QP subproblem (default 1000).
    #[must_use]
    pub const fn with_max_subproblem_iterations(mut self, budget: Budget) -> Self {
        self.max_subproblem_iterations = budget;
        self
    }
    /// Sets the iteration limit of each feasibility restoration (default 20).
    #[must_use]
    pub const fn with_max_restoration_iterations(mut self, budget: Budget) -> Self {
        self.max_restoration_iterations = budget;
        self
    }
    /// Sets how objective and constraint gradients are computed when no
    /// analytic gradient is available.
    #[must_use]
    pub const fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.derivative_strategy = strategy;
        self
    }
    /// Sets a wall-clock limit. The execution context must provide a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }
    /// Returns the step tolerance.
    pub const fn step_tolerance(&self) -> StepTolerance<S> {
        self.step_tolerance
    }
    /// Returns the convergence tolerances.
    pub const fn tolerances(&self) -> OptimalityTolerances<S> {
        self.tolerances
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
    /// Returns the iteration limit of each QP subproblem.
    pub const fn max_subproblem_iterations(&self) -> Budget {
        self.max_subproblem_iterations
    }
    /// Returns the iteration limit of each feasibility restoration.
    pub const fn max_restoration_iterations(&self) -> Budget {
        self.max_restoration_iterations
    }
    /// Returns the derivative strategy.
    pub const fn derivative_strategy(&self) -> DerivativeStrategy {
        self.derivative_strategy
    }
    /// Returns the wall-clock limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }
}

/// Reports method-specific counts and the state of the last QP subproblem of
/// an [`Sqp`] solve.
///
/// Read it through [`OptimizeResult::method_diagnostics`]. The KKT residuals
/// are in [`OptimizeResult::diagnostics`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SqpDiagnostics {
    active_rank: usize,
    multipliers_ambiguous: bool,
    inertia: Option<scioxide_sparse::LdlInertia>,
    second_order_corrections: usize,
    accepted_corrections: usize,
    restoration_solves: usize,
    fixed_derivatives_available: bool,
}
impl SqpDiagnostics {
    /// Returns the numerical rank of the active constraint gradients in the
    /// last accepted subproblem.
    pub const fn active_rank(&self) -> usize {
        self.active_rank
    }
    /// Returns `true` if the active constraint gradients were linearly
    /// dependent, so the multipliers may not be unique.
    pub const fn multipliers_ambiguous(&self) -> bool {
        self.multipliers_ambiguous
    }
    /// Returns the inertia of the last symmetric system solved for an
    /// accepted subproblem, if one was solved.
    pub const fn inertia(&self) -> Option<scioxide_sparse::LdlInertia> {
        self.inertia
    }
    /// Returns the number of second-order corrections tried.
    pub const fn second_order_corrections(&self) -> usize {
        self.second_order_corrections
    }
    /// Returns the number of second-order corrections accepted.
    pub const fn accepted_corrections(&self) -> usize {
        self.accepted_corrections
    }
    /// Returns the number of feasibility restorations.
    pub const fn restoration_solves(&self) -> usize {
        self.restoration_solves
    }
    /// Returns `true` if derivatives with respect to fixed variables were
    /// available, so the reported gradient covers every variable.
    pub const fn fixed_derivatives_available(&self) -> bool {
        self.fixed_derivatives_available
    }
}

/// Solver storage behind [`SqpWorkspace`]: accepted, trial and restoration
/// observations, multiplier states, the dense default subproblem and an
/// optional injected subproblem provider.
#[derive(Debug)]
pub(crate) struct SqpStorage<S: RealScalar + NumericScalar<Real = S>> {
    shape: Option<SqpWorkspaceRequirements>,
    accepted: Observation<S>,
    trial: Observation<S>,
    restored: Observation<S>,
    differences: Differences<S>,
    bounds: Vec<(Option<S>, Option<S>)>,
    step_bounds: Vec<(Option<S>, Option<S>)>,
    model_state: SqpState<S>,
    state: SqpState<S>,
    proposal: SqpState<S>,
    correction: SqpState<S>,
    warm: bool,
    hessian: Matrix<S>,
    identity: Matrix<S>,
    empty_jacobian: Matrix<S>,
    restoration_design: Matrix<S>,
    restoration_target: Vec<S>,
    zeros: Vec<S>,
    direction: Vec<S>,
    scratch: Vec<S>,
    old_lagrangian: Vec<S>,
    new_lagrangian: Vec<S>,
    update: crate::HessianUpdateWorkspace<S>,
    globalization: crate::GlobalizationDiagnostics<S>,
    provider: Option<DenseSqpSubproblem<S>>,
    injected: Option<Box<dyn SqpSubproblemProvider<S> + Send>>,
    diagnostics: SqpDiagnostics,
    optimality: OptimalityDiagnostics<S>,
    iterations: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for SqpStorage<S> {
    fn default() -> Self {
        Self {
            shape: None,
            accepted: Observation::new(),
            trial: Observation::new(),
            restored: Observation::new(),
            differences: Differences::new(),
            bounds: Vec::new(),
            step_bounds: Vec::new(),
            model_state: SqpState::new(),
            state: SqpState::new(),
            proposal: SqpState::new(),
            correction: SqpState::new(),
            warm: false,
            hessian: Matrix::empty(),
            identity: Matrix::empty(),
            empty_jacobian: Matrix::empty(),
            restoration_design: Matrix::empty(),
            restoration_target: Vec::new(),
            zeros: Vec::new(),
            direction: Vec::new(),
            scratch: Vec::new(),
            old_lagrangian: Vec::new(),
            new_lagrangian: Vec::new(),
            update: crate::HessianUpdateWorkspace::new(),
            globalization: crate::GlobalizationDiagnostics::default(),
            provider: None,
            injected: None,
            diagnostics: SqpDiagnostics::default(),
            optimality: OptimalityDiagnostics::unavailable(),
            iterations: 0,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SqpStorage<S> {
    /// Size every buffer and the dense default provider for `shape`.
    fn reserve(&mut self, shape: SqpWorkspaceRequirements) -> Result<(), SciOxideError> {
        let (n, m) = (shape.dimension, shape.constraints);
        self.shape = None;
        for o in [&mut self.accepted, &mut self.trial, &mut self.restored] {
            o.reserve(n, m)?;
        }
        for s in [
            &mut self.model_state,
            &mut self.state,
            &mut self.proposal,
            &mut self.correction,
        ] {
            s.reserve(n, m)?;
        }
        for v in [
            &mut self.zeros,
            &mut self.direction,
            &mut self.scratch,
            &mut self.old_lagrangian,
            &mut self.new_lagrangian,
        ] {
            resize(v, n, <_ as scioxide_core::AlgebraicScalar>::ZERO)?;
        }
        resize(&mut self.bounds, n, (None, None))?;
        resize(&mut self.step_bounds, n, (None, None))?;
        resize(
            &mut self.restoration_target,
            m,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        self.hessian.try_resize_flat(n, n)?;
        self.identity.try_resize_flat(n, n)?;
        self.identity
            .as_mut_slice()
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        for j in 0..n {
            self.identity.as_mut_slice()[j * n + j] = <_ as scioxide_core::AlgebraicScalar>::ONE;
        }
        self.empty_jacobian.try_resize_flat(0, n)?;
        self.restoration_design.try_resize_flat(m, n)?;
        self.update.try_reserve(n)?;
        if self
            .provider
            .as_ref()
            .is_none_or(|p| p.dimensions() != (n, m, m))
        {
            self.provider = Some(DenseSqpSubproblem::new(n, m, m)?);
        }
        self.shape = Some(shape);
        Ok(())
    }
    /// Load the problem's intersected box and key the inward differences to it.
    fn load_bounds(&mut self, constraints: &Constraints<S>) -> Result<(), SciOxideError> {
        self.bounds.copy_from_slice(constraints.hard_box());
        self.differences.prepare(&self.bounds)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::attempt::Accepted<S> for SqpStorage<S> {
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        self.accepted.value?;
        Some(crate::attempt::AcceptedState {
            point: &self.accepted.x,
            value: self.accepted.original_value?,
            gradient: (self.accepted.derivatives && self.accepted.fixed_derivatives)
                .then_some(self.accepted.original_gradient.as_slice()),
            iterations: self.iterations,
        })
    }
}

/// Describes the workspace size of an [`Sqp`] solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqpWorkspaceRequirements {
    dimension: usize,
    equalities: usize,
    constraints: usize,
}
impl SqpWorkspaceRequirements {
    fn of<S: RealScalar + NumericScalar<Real = S>>(constraints: &Constraints<S>) -> Self {
        let (equalities, rows) = shape(constraints);
        Self {
            dimension: constraints.dimension(),
            equalities,
            constraints: rows,
        }
    }
    /// Number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Number of nonlinear equality rows.
    pub const fn equalities(self) -> usize {
        self.equalities
    }
    /// Returns the number of nonlinear rows. Bounds are not counted.
    pub const fn constraints(self) -> usize {
        self.constraints
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Shape<SqpStorage<S>> for SqpWorkspaceRequirements {
    fn fits(&self, storage: &SqpStorage<S>) -> bool {
        storage.shape == Some(*self)
    }
    fn prepare(&self, storage: &mut SqpStorage<S>) -> Result<(), SciOxideError> {
        storage.reserve(*self)
    }
}

crate::attempt::local_workspace!(
    /// Stores the iterates, multipliers and QP subproblem solver of an [`Sqp`]
    /// solve.
    ///
    /// Subproblems are solved by [`DenseSqpSubproblem`] unless
    /// [`with_subproblem_provider`](SqpWorkspace::with_subproblem_provider)
    /// sets another solver.
    SqpWorkspace,
    SqpStorage<S>,
    SqpWorkspaceRequirements
);
impl<S: RealScalar + NumericScalar<Real = S>> SqpWorkspace<S> {
    /// Solves every later QP subproblem with `provider`.
    ///
    /// The solver checks each returned step against the linearized constraints
    /// before using it.
    #[must_use]
    pub fn with_subproblem_provider(
        mut self,
        provider: impl SqpSubproblemProvider<S> + Send + 'static,
    ) -> Self {
        self.storage.injected = Some(Box::new(provider));
        self
    }
    /// Returns the multipliers and active set at the last accepted point.
    pub fn state(&self) -> &SqpState<S> {
        &self.storage.state
    }
    /// Starts the next solve from the multipliers and active set in `state`.
    ///
    /// The quasi-Newton Hessian approximation starts again from the identity.
    /// The next [`Sqp::minimize_into`] call returns `InvalidInput` if the sizes
    /// of `state` do not match its problem.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn set_warm_start(&mut self, state: &SqpState<S>) -> Result<(), SciOxideError> {
        self.storage
            .model_state
            .reserve(state.step.len(), state.multipliers.len())?;
        self.storage.model_state.copy_from(state);
        self.storage.warm = true;
        Ok(())
    }
}

/// Minimizes a smooth nonlinear program by sequential quadratic programming (SQP).
///
/// ```text
/// minimize f(x)  subject to  h(x) = 0,  g(x) ≤ 0,  l ≤ x ≤ u
/// ```
///
/// Each iteration solves a QP subproblem built from the gradient of `f`, the
/// linearized constraints and a damped BFGS approximation `B` of the Hessian of
/// the Lagrangian. The step is accepted by an Armijo backtracking search on the
/// ℓ₁ merit function `f(x) + ρ (Σ|hᵢ(x)| + Σ max(gⱼ(x), 0))`, where the penalty
/// `ρ` is kept at least one larger than every multiplier magnitude. If the
/// first trial is rejected, one
/// second-order correction is tried. If the subproblem is inconsistent or no
/// step is accepted, a feasibility restoration phase reduces the constraint
/// violation.
///
/// The bounds are the intersected box of the problem's [`Constraints`]. The
/// starting point is projected into the box, and the objective and constraints
/// are only evaluated inside it. QP subproblems use the dense active-set solver
/// [`DenseSqpSubproblem`] unless the workspace has another
/// [`SqpSubproblemProvider`].
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn, Sqp};
///
/// // minimize x² + y² subject to x + y = 1; the solution is (1/2, 1/2)
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
/// let constraints = Constraints::new(2)?
///     .equality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)))?;
/// let problem = ConstrainedProblem::new(&objective, &constraints)?;
/// let result = Sqp::default().minimize(&problem, &[2.0, 0.0])?;
/// assert!((result.x()[0] - 0.5).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
///   2006, Chapter 18.
/// - M. J. D. Powell, "A fast algorithm for nonlinearly constrained
///   optimization calculations", in *Numerical Analysis*, Lecture Notes in
///   Mathematics 630, Springer, 1978, 144–157.
#[derive(Debug, Clone)]
pub struct Sqp<S: RealScalar + NumericScalar<Real = S>> {
    options: SqpOptions<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for Sqp<S> {
    fn default() -> Self {
        Self::new(SqpOptions::default())
    }
}
fn shape<S: RealScalar + NumericScalar<Real = S>>(constraints: &Constraints<S>) -> (usize, usize) {
    (
        constraints.profile().equality_count(),
        constraints.nonlinear_functions().count(),
    )
}
impl<S: RealScalar + NumericScalar<Real = S>> Sqp<S> {
    /// Creates a solver with `options`.
    pub const fn new(options: SqpOptions<S>) -> Self {
        Self { options }
    }
    /// Returns the options.
    pub const fn options(&self) -> &SqpOptions<S> {
        &self.options
    }
    /// Returns the workspace size needed for `problem`.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    pub fn workspace_requirements(
        &self,
        problem: &ConstrainedProblem<'_, S>,
    ) -> Result<SqpWorkspaceRequirements, SciOxideError> {
        Ok(SqpWorkspaceRequirements::of(problem.constraints()))
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(
        &self,
        problem: &ConstrainedProblem<'_, S>,
    ) -> Result<SqpWorkspace<S>, SciOxideError> {
        self.workspace_for(problem.constraints())
    }
    /// Allocate storage for problems over `constraints`.
    pub(crate) fn workspace_for(
        &self,
        constraints: &Constraints<S>,
    ) -> Result<SqpWorkspace<S>, SciOxideError> {
        let mut workspace = SqpWorkspace::new();
        workspace.try_reserve(&SqpWorkspaceRequirements::of(constraints))?;
        workspace.storage.load_bounds(constraints)?;
        Ok(workspace)
    }
    /// Minimizes `problem` from `x0` and writes the result to `output`.
    ///
    /// QP subproblems and restoration phases run as nested solves of `scope`.
    /// See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0` does not have the problem's dimension.
    /// - `NonFinite` if `x0` contains NaN or infinity, or if the objective or a
    ///   constraint is not finite at the projected start.
    /// - `InvalidInput` if the projected start is outside the objective domain,
    ///   or a warm start set with [`SqpWorkspace::set_warm_start`] has other
    ///   sizes than `problem`.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - Errors from the callbacks and the subproblem provider, unchanged.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the solve before the first accepted point, as described
    ///   in [Termination](crate#termination).
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut SqpWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::utils::validate_x0(problem.objective(), x0)?;
        let requirements = self.workspace_requirements(problem)?;
        let storage = &workspace.storage;
        if storage.warm
            && (storage.model_state.step.len() != requirements.dimension
                || storage.model_state.multipliers.len() != requirements.constraints)
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "warm state shape differs from problem",
            ));
        }
        workspace.try_reserve(&requirements)?;
        crate::attempt::preflight(&scope, self.options.time_limit)?;
        let entry = Entry::Call {
            identity: self.identity(problem),
            budgets: SolveBudgets::unbounded(),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                self.minimize_in_context(problem, x0, staged, storage, context)
            },
        )
    }
    /// Identify method, constraint shape, box and options for a solve epoch.
    pub(crate) fn identity(
        &self,
        problem: &ConstrainedProblem<'_, S>,
    ) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "Sqp", 3),
            problem.dim(),
            &(
                &self.options,
                problem.constraints().profile(),
                problem.constraints().hard_box(),
            ),
        )
    }
    /// Run one epoch in `context` with the workspace's provider.
    pub(crate) fn minimize_in_context(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        out: &mut OptimizeResult<S>,
        w: &mut SqpStorage<S>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        w.accepted.value = None;
        if let Some(limit) = self.options.time_limit() {
            ctx.limit_time(limit)?;
        }
        let session = SolveSession::new(ctx, scioxide_core::argument!("scope"));
        // Lend the provider without leaving the workspace unprepared on error.
        let mut injected = w.injected.take();
        let mut dense = w.provider.take();
        let outcome = match (injected.as_deref_mut(), dense.as_mut()) {
            (Some(provider), _) => self.kernel(problem, x0, out, w, provider, &session),
            (None, Some(provider)) => self.kernel(problem, x0, out, w, provider, &session),
            (None, None) => Err(invalid(
                scioxide_core::argument!("workspace"),
                "unprepared subproblem provider",
            )),
        };
        w.injected = injected;
        w.provider = dense;
        if let Err(error) = &outcome {
            session.record_execution_stop(error);
        }
        let eq = shape(problem.constraints()).0;
        match outcome {
            Ok(stop) => self.finish(out, w, eq, session.stopped().unwrap_or(stop), &session),
            Err(error) => {
                if let (Some(stop), Some(_)) = (session.stopped(), w.accepted.value) {
                    self.finish(out, w, eq, stop, &session)
                } else {
                    Err(error)
                }
            }
        }
    }
    fn evaluate(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        o: &mut Observation<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        o.evaluate(problem.objective(), Some(problem.constraints()), session)
    }
    fn derivatives(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        o: &mut Observation<S>,
        d: &mut Differences<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        o.derivatives(
            problem.objective(),
            Some(problem.constraints()),
            d,
            self.options.derivative_strategy(),
            session,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn kernel(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        out: &mut OptimizeResult<S>,
        w: &mut SqpStorage<S>,
        provider: &mut dyn SqpSubproblemProvider<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        crate::utils::validate_x0(problem.objective(), x0)?;
        let (eq, m) = shape(problem.constraints());
        let n = x0.len();
        let warm = w.warm;
        if warm && (w.model_state.step.len() != n || w.model_state.multipliers.len() != m) {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "warm state shape differs from problem",
            ));
        }
        if w.shape
            .is_none_or(|shape| shape.dimension != n || shape.constraints != m)
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace shape differs from problem",
            ));
        }
        w.load_bounds(problem.constraints())?;
        out.try_reserve(n)?;
        w.accepted.value = None;
        w.iterations = 0;
        w.diagnostics = SqpDiagnostics::default();
        w.globalization = crate::GlobalizationDiagnostics::default();
        w.optimality = OptimalityDiagnostics::unavailable();
        w.warm = false;
        if !warm {
            w.model_state.clear();
        } else {
            if w.model_state.active.iter().any(|&id| id >= m + 2 * n) {
                return Err(invalid(
                    scioxide_core::argument!("workspace"),
                    "warm active-set index is outside the problem",
                ));
            }
            for v in [
                &w.model_state.multipliers,
                &w.model_state.lower,
                &w.model_state.upper,
            ] {
                crate::utils::validate_finite(v)?;
            }
            if w.model_state.multipliers[eq..]
                .iter()
                .chain(&w.model_state.lower)
                .chain(&w.model_state.upper)
                .any(|&v| v < <_ as scioxide_core::AlgebraicScalar>::ZERO)
            {
                return Err(invalid(
                    scioxide_core::argument!("workspace"),
                    "negative warm inequality multiplier",
                ));
            }
        }
        w.hessian
            .as_mut_slice()
            .copy_from_slice(w.identity.as_slice());
        w.accepted.objective_scale = <S as scioxide_core::AlgebraicScalar>::ZERO;
        w.accepted
            .row_scales
            .fill(<S as scioxide_core::AlgebraicScalar>::ONE);
        w.accepted.at(x0);
        project(&mut w.accepted.x, &w.bounds);
        w.accepted
            .evaluate_start(problem.objective(), Some(problem.constraints()), session)?;
        self.derivatives(problem, &mut w.accepted, &mut w.differences, session)?;
        w.trial.inherit_scales(&w.accepted);
        w.restored.inherit_scales(&w.accepted);
        if warm {
            for (i, lambda) in w.model_state.multipliers.iter_mut().enumerate() {
                *lambda = scioxide_core::checked_float_product(
                    scioxide_core::checked_float_quotient(*lambda, w.accepted.objective_scale)?,
                    w.accepted.row_scales[i],
                )?;
            }
            for lambda in w
                .model_state
                .lower
                .iter_mut()
                .chain(&mut w.model_state.upper)
            {
                *lambda =
                    scioxide_core::checked_float_quotient(*lambda, w.accepted.objective_scale)?;
            }
        }
        let coordinate = w
            .accepted
            .x
            .iter()
            .fold(S::ONE, |a, &v| a.max(NumericScalar::abs(v)));
        for value in w.hessian.as_mut_slice() {
            *value /= coordinate;
        }
        let mut penalty = S::TWO;
        let mut small_steps = 0usize;
        for iteration in 0..self.options.max_iterations().get() {
            w.optimality = diagnostics(&w.accepted, &w.model_state, &w.bounds, eq, &mut w.scratch)?;
            if satisfies(
                w.optimality,
                self.options.tolerances(),
                &w.accepted,
                &w.model_state,
            )? {
                return Ok(OptimizeTermination::Converged);
            }
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            step_bounds(&w.bounds, &w.accepted.x, &mut w.step_bounds)?;
            let model = SqpSubproblem {
                objective: SqpSubproblemObjective::Quadratic {
                    hessian: w.hessian.view(),
                    gradient: &w.accepted.gradient,
                },
                jacobian: w.accepted.jacobian.view(),
                values: &w.accepted.values,
                equalities: eq,
                bounds: &w.step_bounds,
                tolerances: self.options.tolerances(),
                max_iterations: self.options.max_subproblem_iterations(),
            };
            w.proposal.copy_from(&w.model_state);
            let report = solve_child(provider, model, &mut w.proposal, session)?;
            let mut restored = false;
            let mut accepted = false;
            let mut corrected = false;
            if report.termination == OptimizeTermination::Converged {
                // A zero primal step can still establish new, valid dual multipliers.
                let trial_diagnostics =
                    diagnostics(&w.accepted, &w.proposal, &w.bounds, eq, &mut w.scratch)?;
                if satisfies(
                    trial_diagnostics,
                    self.options.tolerances(),
                    &w.accepted,
                    &w.proposal,
                )? {
                    w.model_state.copy_from(&w.proposal);
                    w.optimality = trial_diagnostics;
                    record_report(&mut w.diagnostics, report);
                    return Ok(OptimizeTermination::Converged);
                }
                let small = self.options.step_tolerance().is_satisfied(
                    crate::utils::norm2(&w.proposal.step)?,
                    crate::utils::norm2(&w.accepted.x)?,
                )? && violation(&w.accepted.values, eq)
                    <= self
                        .options
                        .tolerances()
                        .feasibility()
                        .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
                // Take one small step before stopping: near a solution it is
                // what finally meets the stationarity tolerance.
                if small && small_steps > 0 {
                    return Ok(OptimizeTermination::StepSmall);
                }
                small_steps = if small { small_steps + 1 } else { 0 };
                penalty = penalty.max(
                    w.proposal
                        .multipliers
                        .iter()
                        .chain(&w.proposal.lower)
                        .chain(&w.proposal.upper)
                        .fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)))
                        + <_ as scioxide_core::AlgebraicScalar>::ONE,
                );
                let current = merit(&w.accepted, eq, penalty)?;
                let slope = directional(&w.accepted, &w.proposal.step, eq, penalty)?;
                if slope < <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    let searched = crate::globalization::backtrack(
                        <_ as scioxide_core::AlgebraicScalar>::ONE,
                        S::EPSILON,
                        S::HALF,
                        crate::globalization::TRIALS,
                        |attempt, alpha| {
                            session.proposal()?;
                            for j in 0..n {
                                w.trial.x[j] = crate::utils::trial_component(
                                    w.accepted.x[j],
                                    alpha,
                                    w.proposal.step[j],
                                )?;
                            }
                            project(&mut w.trial.x, &w.bounds);
                            let domain = self.evaluate(problem, &mut w.trial, session)?;
                            if domain
                                && crate::utils::armijo_satisfied(
                                    merit(&w.trial, eq, penalty)?,
                                    current,
                                    crate::globalization::sufficient_decrease(),
                                    alpha,
                                    slope,
                                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                                )?
                            {
                                accepted = true;
                                return Ok(Some(None));
                            }
                            session.reject()?;
                            if attempt == 0 && domain && m > 0 {
                                w.diagnostics.second_order_corrections += 1;
                                step_bounds(&w.bounds, &w.trial.x, &mut w.step_bounds)?;
                                let correction_model = SqpSubproblem {
                                    objective: SqpSubproblemObjective::Quadratic {
                                        hessian: w.identity.view(),
                                        gradient: &w.zeros,
                                    },
                                    jacobian: w.accepted.jacobian.view(),
                                    values: &w.trial.values,
                                    equalities: eq,
                                    bounds: &w.step_bounds,
                                    tolerances: self.options.tolerances(),
                                    max_iterations: self.options.max_subproblem_iterations(),
                                };
                                let correction = solve_child(
                                    provider,
                                    correction_model,
                                    &mut w.correction,
                                    session,
                                )?;
                                if correction.termination == OptimizeTermination::Converged {
                                    if crate::utils::norm2(&w.correction.step)?
                                        <= crate::utils::norm2(&w.proposal.step)?
                                    {
                                        session.proposal()?;
                                        for j in 0..n {
                                            w.trial.x[j] = crate::utils::trial_component(
                                                w.trial.x[j],
                                                <_ as scioxide_core::AlgebraicScalar>::ONE,
                                                w.correction.step[j],
                                            )?;
                                        }
                                        project(&mut w.trial.x, &w.bounds);
                                        if self.evaluate(problem, &mut w.trial, session)?
                                            && crate::utils::armijo_satisfied(
                                                merit(&w.trial, eq, penalty)?,
                                                current,
                                                crate::globalization::sufficient_decrease(),
                                                <_ as scioxide_core::AlgebraicScalar>::ONE,
                                                slope,
                                                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                                            )?
                                        {
                                            accepted = true;
                                            corrected = true;
                                            return Ok(Some(None));
                                        }
                                        session.reject()?;
                                    }
                                } else if correction.termination
                                    != OptimizeTermination::SuspectedInfeasible
                                {
                                    return Ok(Some(Some(correction.termination)));
                                }
                            }

                            Ok(None)
                        },
                    )?;
                    if let Ok(Some(stop)) = searched {
                        return Ok(stop);
                    }
                }
            } else if report.termination != OptimizeTermination::SuspectedInfeasible {
                return Ok(report.termination);
            }
            if !accepted {
                w.diagnostics.restoration_solves += 1;
                let status = self.restore(problem, eq, m, w, provider, session)?;
                if status != OptimizeTermination::Converged {
                    return Ok(status);
                }
                restored = true;
                w.proposal.clear();
            }
            self.derivatives(problem, &mut w.trial, &mut w.differences, session)?;
            for j in 0..n {
                w.direction[j] = scioxide_core::checked_difference(w.trial.x[j], w.accepted.x[j])?;
            }
            let next = diagnostics(&w.trial, &w.proposal, &w.bounds, eq, &mut w.scratch)?;
            let update = if restored {
                None
            } else {
                lagrangian(&w.accepted, &w.proposal, &mut w.old_lagrangian)?;
                lagrangian(&w.trial, &w.proposal, &mut w.new_lagrangian)?;
                for j in 0..n {
                    w.scratch[j] = scioxide_core::checked_difference(
                        w.new_lagrangian[j],
                        w.old_lagrangian[j],
                    )?;
                }
                Some(w.update.prepare(
                    w.hessian.view(),
                    &w.direction,
                    &w.scratch,
                    crate::HessianForm::Hessian,
                    crate::HessianUpdate::Bfgs(crate::CurvaturePolicy::default()),
                    session.execution()?,
                    Some(session),
                    crate::quasi_newton::UpdateSource::Verified,
                    scioxide_core::argument!("w"),
                )?)
            };
            session.accept(scioxide_core::PartialResultTrust::Admissible)?;
            if let Some(update) = update {
                w.update.commit(&mut w.hessian, update);
                w.globalization.record_update(update);
            } else {
                w.hessian
                    .as_mut_slice()
                    .copy_from_slice(w.identity.as_slice());
            }
            w.accepted.copy_from(&w.trial);
            w.model_state.copy_from(&w.proposal);
            w.model_state.step.copy_from_slice(&w.direction);
            w.optimality = next;
            w.iterations = iteration + 1;
            if !restored {
                record_report(&mut w.diagnostics, report);
            }
            if corrected {
                w.diagnostics.accepted_corrections += 1;
            }
            w.diagnostics.fixed_derivatives_available = w.accepted.fixed_derivatives;
        }
        Ok(
            if satisfies(
                w.optimality,
                self.options.tolerances(),
                &w.accepted,
                &w.model_state,
            )? {
                OptimizeTermination::Converged
            } else {
                OptimizeTermination::IterationLimit
            },
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn restore(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        eq: usize,
        m: usize,
        w: &mut SqpStorage<S>,
        provider: &mut dyn SqpSubproblemProvider<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let n = w.accepted.x.len();
        let initial = violation(&w.accepted.values, eq);
        let target = self
            .options
            .tolerances()
            .feasibility()
            .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
        if initial <= target {
            return Ok(OptimizeTermination::Stagnation);
        }
        w.restored.copy_from(&w.accepted);
        for _ in 0..self.options.max_restoration_iterations().get() {
            if let Some(stop) = session.trial_iteration()? {
                return Ok(stop);
            }
            for i in 0..m {
                let active =
                    i < eq || w.restored.values[i] > <_ as scioxide_core::AlgebraicScalar>::ZERO;
                w.restoration_target[i] = if active {
                    -w.restored.values[i]
                } else {
                    <_ as scioxide_core::AlgebraicScalar>::ZERO
                };
                for j in 0..n {
                    w.restoration_design.as_mut_slice()[i * n + j] = if active {
                        w.restored.jacobian.as_slice()[i * n + j]
                    } else {
                        <_ as scioxide_core::AlgebraicScalar>::ZERO
                    };
                }
            }
            step_bounds(&w.bounds, &w.restored.x, &mut w.step_bounds)?;
            w.correction.reserve(n, 0)?;
            let model = SqpSubproblem {
                objective: SqpSubproblemObjective::LeastSquares {
                    design: w.restoration_design.view(),
                    target: &w.restoration_target,
                },
                jacobian: w.empty_jacobian.view(),
                values: &[],
                equalities: 0,
                bounds: &w.step_bounds,
                tolerances: self.options.tolerances(),
                max_iterations: self.options.max_subproblem_iterations(),
            };
            let report = solve_child(provider, model, &mut w.correction, session)?;
            if report.termination != OptimizeTermination::Converged {
                return Ok(report.termination);
            }
            let current = violation(&w.restored.values, eq);

            let searched = crate::globalization::backtrack(
                <_ as scioxide_core::AlgebraicScalar>::ONE,
                S::EPSILON,
                S::HALF,
                crate::globalization::TRIALS,
                |_attempt, alpha| {
                    session.proposal()?;
                    for j in 0..n {
                        w.trial.x[j] = crate::utils::trial_component(
                            w.restored.x[j],
                            alpha,
                            w.correction.step[j],
                        )?;
                    }
                    project(&mut w.trial.x, &w.bounds);
                    if self.evaluate(problem, &mut w.trial, session)? {
                        let v = violation(&w.trial.values, eq);
                        if v <= target || v <= initial * S::HALF {
                            return Ok(Some(true));
                        }
                        if v < current {
                            return Ok(Some(false));
                        }
                    }
                    session.reject()?;

                    Ok(None)
                },
            )?;
            if searched == Ok(true) {
                w.correction.reserve(n, m)?;
                return Ok(OptimizeTermination::Converged);
            }
            let progress = searched.is_ok();
            if !progress {
                break;
            }
            self.derivatives(problem, &mut w.trial, &mut w.differences, session)?;
            w.restored.copy_from(&w.trial);
        }
        w.correction.reserve(n, m)?;
        Ok(OptimizeTermination::Stagnation)
    }
    fn finish(
        &self,
        out: &mut OptimizeResult<S>,
        w: &mut SqpStorage<S>,
        eq: usize,
        stop: OptimizeTermination,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if stop.is_success()
            && (stop != OptimizeTermination::Converged
                || !w.accepted.derivatives
                || !satisfies(
                    w.optimality,
                    self.options.tolerances(),
                    &w.accepted,
                    &w.model_state,
                )?)
        {
            return Err(invalid(
                scioxide_core::argument!("f"),
                "SQP success requires accepted original-model KKT residuals",
            ));
        }
        let value = w.accepted.original_value.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("f"),
                "no accepted objective observation",
            )
        })?;
        unscale_state(&w.accepted, &w.model_state, &mut w.state)?;
        crate::utils::commit_optimize_result(
            out,
            &w.accepted.x,
            value,
            (w.accepted.derivatives && w.accepted.fixed_derivatives)
                .then_some(w.accepted.original_gradient.as_slice()),
            w.iterations,
            stop,
            "SQP completed; inspect termination and original-coordinate residuals",
        )?;
        out.diagnostics = original_diagnostics(&w.accepted, &w.model_state, w.optimality, eq)?;
        let mut diag = w.diagnostics;
        diag.fixed_derivatives_available = w.accepted.fixed_derivatives;
        out.method = crate::OptimizeDiagnostics::Sqp(diag, w.globalization);
        session.assess(if out.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        })?;
        crate::utils::synchronize_metadata(
            out,
            session,
            if out.is_success() {
                scioxide_core::PartialResultTrust::ResidualChecked
            } else {
                scioxide_core::PartialResultTrust::DiagnosticOnly
            },
        )
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Sqp<S> {
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
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut output = OptimizeResult::empty();
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

impl<'a, S: RealScalar + NumericScalar<Real = S>>
    crate::traits::Optimizer<S, ConstrainedProblem<'a, S>> for Sqp<S>
{
    fn minimize(
        &self,
        problem: &ConstrainedProblem<'a, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(problem, x0)
    }
}
pub(super) fn solve_child<S: RealScalar + NumericScalar<Real = S>>(
    provider: &mut dyn SqpSubproblemProvider<S>,
    model: SqpSubproblem<'_, S>,
    state: &mut SqpState<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<SqpSubproblemReport, SciOxideError> {
    let mut child = SolveState::new();
    let identity = crate::context::local_identity(
        scioxide_core::algorithm_id!("optimize", "SqpSubproblem", 2),
        model.bounds.len(),
        &provider,
    );
    let mut result = session.with_child(
        &mut child,
        identity,
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, model.max_iterations),
        |ctx| {
            let mut report = provider.solve_into(model, state, ctx)?;
            if report.termination == OptimizeTermination::Converged
                && !subproblem::verify(model, state, scioxide_core::argument!("provider"))?
            {
                // Retry once with a fresh working set under the same child
                // allowance. Rejected finite residuals are not input errors.
                state.clear();
                report = provider.solve_into(model, state, ctx)?;
            }
            Ok(report)
        },
    );
    if let Ok(report) = &mut result {
        if report.termination == OptimizeTermination::Converged
            && !subproblem::verify(model, state, scioxide_core::argument!("provider"))?
        {
            report.termination = OptimizeTermination::Stagnation;
        }
        if report.termination.is_success() && report.termination != OptimizeTermination::Converged {
            return Err(invalid(
                scioxide_core::argument!("provider"),
                "SQP providers must report verified success as Converged",
            ));
        }
        if report.termination != OptimizeTermination::Converged
            && report.termination != OptimizeTermination::SuspectedInfeasible
        {
            session.inherit_child_stop(Some(report.termination))?;
        }
    }
    session.check()?;
    result
}
fn record_report(d: &mut SqpDiagnostics, r: SqpSubproblemReport) {
    d.active_rank = r.active_rank;
    d.multipliers_ambiguous = r.multipliers_ambiguous;
    d.inertia = r.inertia;
}
pub(super) use crate::geometry::project;
pub(super) fn step_bounds<S: RealScalar + NumericScalar<Real = S>>(
    bounds: &[(Option<S>, Option<S>)],
    x: &[S],
    out: &mut [(Option<S>, Option<S>)],
) -> Result<(), SciOxideError> {
    for ((out, &(lo, hi)), &x) in out.iter_mut().zip(bounds).zip(x) {
        *out = (
            lo.map(|v| scioxide_core::checked_difference(v, x))
                .transpose()?,
            hi.map(|v| scioxide_core::checked_difference(v, x))
                .transpose()?,
        );
    }
    Ok(())
}
pub(super) use crate::geometry::violation;
pub(super) fn lagrangian<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    s: &SqpState<S>,
    out: &mut [S],
) -> Result<(), SciOxideError> {
    let n = o.x.len();
    for (j, slot) in out.iter_mut().enumerate().take(n) {
        let mut sum = scioxide_core::ProductSum::new();
        sum.add_product(o.gradient[j], <_ as scioxide_core::AlgebraicScalar>::ONE)?;
        sum.add_product(
            s.upper[j] - s.lower[j],
            <_ as scioxide_core::AlgebraicScalar>::ONE,
        )?;
        for (i, &lambda) in s.multipliers.iter().enumerate() {
            sum.add_product(lambda, o.jacobian.as_slice()[i * n + j])?;
        }
        *slot = sum.finish()?;
    }
    Ok(())
}
pub(super) fn diagnostics<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    s: &SqpState<S>,
    bounds: &[(Option<S>, Option<S>)],
    eq: usize,
    scratch: &mut [S],
) -> Result<OptimalityDiagnostics<S>, SciOxideError> {
    lagrangian(o, s, scratch)?;
    let mut dual = S::ZERO;
    let mut comp = S::ZERO;
    let mut bound = S::ZERO;
    for (&v, &l) in o.values[eq..].iter().zip(&s.multipliers[eq..]) {
        dual = dual.max(-l);
        comp = comp.max(NumericScalar::abs(scioxide_core::checked_float_product(
            v, l,
        )?));
    }
    for (j, &(lo, hi)) in bounds.iter().enumerate() {
        for (endpoint, lambda, sign) in [
            (lo, s.lower[j], -S::ONE),
            (hi, s.upper[j], <_ as scioxide_core::AlgebraicScalar>::ONE),
        ] {
            dual = dual.max(-lambda);
            if let Some(b) = endpoint {
                let v = sign * (o.x[j] - b);
                bound = bound.max(v);
                comp = comp.max(NumericScalar::abs(scioxide_core::checked_float_product(
                    v, lambda,
                )?));
            } else {
                dual = dual.max(NumericScalar::abs(lambda));
            }
        }
    }
    Ok(OptimalityDiagnostics {
        primal_feasibility: Some(violation(&o.values, eq).max(bound)),
        stationarity: Some(crate::utils::norm2(scratch)?),
        dual_feasibility: Some(dual),
        complementarity: Some(comp),
        bound_violation: Some(bound),
        ..OptimalityDiagnostics::unavailable()
    })
}
pub(super) fn satisfies<S: RealScalar + NumericScalar<Real = S>>(
    d: OptimalityDiagnostics<S>,
    t: crate::OptimalityTolerances<S>,
    o: &Observation<S>,
    state: &SqpState<S>,
) -> Result<bool, SciOxideError> {
    let scales = crate::utils::optimality_scales(
        &o.x,
        o.value
            .ok_or_else(|| invalid(scioxide_core::argument!("f"), "missing KKT objective"))?,
        &o.gradient,
        o.jacobian.view(),
        &state.multipliers,
        &state.lower,
        &state.upper,
    )?;
    t.satisfied(d, scales)
}

/// Re-express normalized model residuals in original objective/row units.
pub(super) fn original_diagnostics<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    state: &SqpState<S>,
    mut diagnostics: OptimalityDiagnostics<S>,
    equalities: usize,
) -> Result<OptimalityDiagnostics<S>, SciOxideError> {
    diagnostics.stationarity = diagnostics
        .stationarity
        .map(|v| scioxide_core::checked_float_product(v, o.objective_scale))
        .transpose()?;
    diagnostics.complementarity = diagnostics
        .complementarity
        .map(|v| scioxide_core::checked_float_product(v, o.objective_scale))
        .transpose()?;
    diagnostics.primal_feasibility = Some(
        violation(&o.original_values, equalities).max(
            diagnostics
                .bound_violation
                .unwrap_or(<S as scioxide_core::AlgebraicScalar>::ZERO),
        ),
    );
    let mut dual = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for (i, &lambda) in state.multipliers.iter().enumerate().skip(equalities) {
        dual = dual.max(-scioxide_core::checked_float_product(
            scioxide_core::checked_float_quotient(lambda, o.row_scales[i])?,
            o.objective_scale,
        )?);
    }
    for &lambda in state.lower.iter().chain(&state.upper) {
        dual = dual.max(-scioxide_core::checked_float_product(
            lambda,
            o.objective_scale,
        )?);
    }
    diagnostics.dual_feasibility = Some(dual);
    Ok(diagnostics)
}

pub(super) fn unscale_state<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    source: &SqpState<S>,
    target: &mut SqpState<S>,
) -> Result<(), SciOxideError> {
    target.copy_from(source);
    for (i, lambda) in target.multipliers.iter_mut().enumerate() {
        *lambda = scioxide_core::checked_float_product(
            scioxide_core::checked_float_quotient(*lambda, o.row_scales[i])?,
            o.objective_scale,
        )?;
    }
    for lambda in target.lower.iter_mut().chain(&mut target.upper) {
        *lambda = scioxide_core::checked_float_product(*lambda, o.objective_scale)?;
    }
    Ok(())
}
fn merit<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    eq: usize,
    penalty: S,
) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(
        o.value
            .ok_or_else(|| invalid(scioxide_core::argument!("f"), "missing merit objective"))?,
        <_ as scioxide_core::AlgebraicScalar>::ONE,
    )?;
    for (i, &v) in o.values.iter().enumerate() {
        sum.add_product(
            penalty,
            if i < eq {
                NumericScalar::abs(v)
            } else {
                v.max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
            },
        )?;
    }
    sum.finish()
}
fn directional<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    p: &[S],
    eq: usize,
    penalty: S,
) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(
        crate::utils::dot(&o.gradient, p)?,
        <_ as scioxide_core::AlgebraicScalar>::ONE,
    )?;
    for (i, &v) in o.values.iter().enumerate() {
        let slope = crate::utils::dot(&o.jacobian.as_slice()[i * p.len()..(i + 1) * p.len()], p)?;
        let d = if i < eq {
            crate::geometry::absolute_directional(v, slope)
        } else {
            crate::geometry::positive_directional(v, slope)
        };
        sum.add_product(penalty, d)?;
    }
    sum.finish()
}

#[cfg(test)]
mod tests;
