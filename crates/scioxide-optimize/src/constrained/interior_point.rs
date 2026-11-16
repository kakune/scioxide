//! Primal-dual interior-point method for nonlinear programs.
mod engine;
use super::nonlinear::Observation;
use super::sqp::SqpOptions;
use crate::attempt::{Entry, Shape};
use crate::context::SolveSession;
use crate::derivatives::Differences;
use crate::{
    ConstrainedProblem, Constraints, DenseKktProvider, DenseSqpSubproblem, KktProvider,
    OptimalityDiagnostics, OptimalityTolerances, OptimizeResult, OptimizeTermination, SqpState,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use scioxide_core::{
    AttemptKind, Budget, EvaluationRole, NumericScalar, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveScope,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, MatrixViewMut};

use crate::validation::invalid;
use crate::validation::number;

/// Computes the Hessian of the Lagrangian of a nonlinear program.
///
/// ```text
/// ∇²ₓₓ L(x, λ) = ∇²f(x) + Σᵢ λᵢ ∇²cᵢ(x)
/// ```
///
/// where `cᵢ` are the nonlinear rows of the [`Constraints`], equality rows
/// first, and `λᵢ` their multipliers. Bounds do not contribute. Attach an
/// implementation with [`ConstrainedProblem::with_lagrangian_hessian`];
/// [`InteriorPoint`] then uses it in place of a damped BFGS approximation.
pub trait LagrangianHessian<S: RealScalar + NumericScalar<Real = S>> {
    /// Writes the full symmetric `n × n` matrix `∇²ₓₓ L(x, λ)` to `output`.
    ///
    /// # Errors
    ///
    /// Return an error if the Hessian cannot be computed. The solver passes it
    /// on unchanged. It reports a non-finite or nonsymmetric matrix as an error
    /// of its own.
    fn hessian_into(
        &self,
        x: &[S],
        multipliers: &[S],
        output: MatrixViewMut<'_, S>,
    ) -> Result<(), SciOxideError>;
}

/// Sets the tolerances and iteration limits of [`InteriorPoint`].
///
/// The defaults are the default [`OptimalityTolerances`], at most 1000
/// iterations, 20 iterations per feasibility restoration and 1000 iterations
/// per restoration subproblem. See [Termination](crate#termination) for how
/// the tolerances are scaled.
#[derive(Debug, Clone, Copy, Default)]
pub struct InteriorPointOptions<S: RealScalar + NumericScalar<Real = S>> {
    model: SqpOptions<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointOptions<S> {
    /// Sets the convergence tolerances.
    #[must_use]
    pub const fn with_tolerances(mut self, tolerances: OptimalityTolerances<S>) -> Self {
        self.model = self.model.with_tolerances(tolerances);
        self
    }
    /// Sets the iteration limit (default 1000).
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: Budget) -> Self {
        self.model = self.model.with_max_iterations(limit);
        self
    }
    /// Sets the iteration limit of each feasibility restoration (default 20).
    #[must_use]
    pub const fn with_max_restoration_iterations(mut self, limit: Budget) -> Self {
        self.model = self.model.with_max_restoration_iterations(limit);
        self
    }
    /// Sets the iteration limit of each linear least-squares subproblem of a
    /// restoration (default 1000).
    #[must_use]
    pub const fn with_max_subproblem_iterations(mut self, limit: Budget) -> Self {
        self.model = self.model.with_max_subproblem_iterations(limit);
        self
    }
    /// Sets how objective and constraint gradients are computed when no
    /// analytic gradient is available.
    #[must_use]
    pub const fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.model = self.model.with_derivative_strategy(strategy);
        self
    }
    /// Sets a wall-clock limit. The execution context must provide a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, limit: core::time::Duration) -> Self {
        self.model = self.model.with_time_limit(limit);
        self
    }
    /// Returns the convergence tolerances.
    pub const fn tolerances(&self) -> OptimalityTolerances<S> {
        self.model.tolerances()
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.model.max_iterations()
    }
    /// Returns the iteration limit of each feasibility restoration.
    pub const fn max_restoration_iterations(&self) -> Budget {
        self.model.max_restoration_iterations()
    }
    /// Returns the iteration limit of each restoration subproblem.
    pub const fn max_subproblem_iterations(&self) -> Budget {
        self.model.max_subproblem_iterations()
    }
    /// Returns the derivative strategy.
    pub const fn derivative_strategy(&self) -> DerivativeStrategy {
        self.model.derivative_strategy()
    }
    /// Returns the wall-clock limit, if set.
    pub const fn time_limit(&self) -> Option<core::time::Duration> {
        self.model.time_limit()
    }
}

/// Holds the multipliers and slacks at the last accepted point of an
/// [`InteriorPoint`] solve.
///
/// Read it with [`InteriorPointWorkspace::state`].
#[derive(Debug)]
pub struct InteriorPointState<S: RealScalar + NumericScalar<Real = S>> {
    dual: SqpState<S>,
    lambda: Vec<S>,
    slack: Vec<S>,
    equalities: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointState<S> {
    const fn empty() -> Self {
        Self {
            dual: SqpState::new(),
            lambda: Vec::new(),
            slack: Vec::new(),
            equalities: 0,
        }
    }
    fn new(n: usize, m: usize, rows: usize, equalities: usize) -> Result<Self, SciOxideError> {
        let mut dual = SqpState::new();
        dual.reserve(n, m)?;
        Ok(Self {
            dual,
            lambda: scioxide_core::backend::try_filled(
                rows,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("interior_duals"),
            )?,
            slack: scioxide_core::backend::try_filled(
                rows,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("interior_slacks"),
            )?,
            equalities,
        })
    }
    fn copy_from(&mut self, other: &Self) {
        self.dual.copy_from(&other.dual);
        self.lambda.copy_from_slice(&other.lambda);
        self.slack.copy_from_slice(&other.slack);
    }
    /// Returns the multipliers of the nonlinear rows, equality rows first.
    pub fn multipliers(&self) -> &[S] {
        self.dual.multipliers()
    }
    /// Returns the lower-bound multipliers, one per variable; zero where
    /// there is no lower bound.
    pub fn lower_multipliers(&self) -> &[S] {
        self.dual.lower_multipliers()
    }
    /// Returns the upper-bound multipliers, one per variable; zero where
    /// there is no upper bound.
    pub fn upper_multipliers(&self) -> &[S] {
        self.dual.upper_multipliers()
    }
    /// Returns the positive slacks `sᵢ` with `gᵢ(x) + sᵢ = 0`: first the
    /// nonlinear inequality rows, then the finite bound sides, lower before
    /// upper, in variable order.
    pub fn slacks(&self) -> &[S] {
        &self.slack[self.equalities..]
    }
}
/// Reports the barrier parameter, regularization and method-specific counts of
/// an [`InteriorPoint`] solve.
///
/// Read it through [`OptimizeResult::method_diagnostics`]. The KKT residuals
/// are in [`OptimizeResult::diagnostics`].
#[derive(Debug, Clone, Copy)]
pub struct InteriorPointDiagnostics<S: RealScalar> {
    inertia: Option<scioxide_sparse::LdlInertia>,
    barrier_parameter: S,
    primal_regularization: S,
    dual_regularization: S,
    regularization_attempts: usize,
    restoration_solves: usize,
    second_order_corrections: usize,
    accepted_corrections: usize,
    fixed_derivatives_available: bool,
}
impl<S: RealScalar> Default for InteriorPointDiagnostics<S> {
    fn default() -> Self {
        Self {
            inertia: None,
            barrier_parameter: S::ZERO,
            primal_regularization: S::ZERO,
            dual_regularization: S::ZERO,
            regularization_attempts: 0,
            restoration_solves: 0,
            second_order_corrections: 0,
            accepted_corrections: 0,
            fixed_derivatives_available: true,
        }
    }
}
impl<S: RealScalar> InteriorPointDiagnostics<S> {
    /// Returns the inertia of the last KKT matrix factored.
    pub const fn inertia(&self) -> Option<scioxide_sparse::LdlInertia> {
        self.inertia
    }
    /// Returns the last barrier parameter `μ`.
    pub const fn barrier_parameter(&self) -> S {
        self.barrier_parameter
    }
    /// Returns the last diagonal shift added to the Hessian block.
    pub const fn primal_regularization(&self) -> S {
        self.primal_regularization
    }
    /// Returns the last diagonal shift added to the equality block; nonzero
    /// when the equality Jacobian is rank deficient.
    pub const fn dual_regularization(&self) -> S {
        self.dual_regularization
    }
    /// Returns the number of factorizations whose inertia was wrong and that
    /// were repeated with a larger shift.
    pub const fn regularization_attempts(&self) -> usize {
        self.regularization_attempts
    }
    /// Returns the number of feasibility restorations, including one at the
    /// start if the starting point is infeasible.
    pub const fn restoration_solves(&self) -> usize {
        self.restoration_solves
    }
    /// Returns the number of second-order corrections tried.
    pub const fn second_order_corrections(&self) -> usize {
        self.second_order_corrections
    }
    /// Returns the number of second-order corrections accepted.
    pub const fn accepted_corrections(&self) -> usize {
        self.accepted_corrections
    }
    /// Returns `true` if derivatives with respect to fixed variables were
    /// available, so the reported gradient covers every variable.
    pub const fn fixed_derivatives_available(&self) -> bool {
        self.fixed_derivatives_available
    }
}

/// Describes the workspace size of an [`InteriorPoint`] solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteriorPointWorkspaceRequirements {
    dimension: usize,
    equalities: usize,
    constraints: usize,
    bound_sides: usize,
}
impl InteriorPointWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the number of nonlinear equality rows.
    pub const fn equalities(self) -> usize {
        self.equalities
    }
    /// Returns the number of nonlinear rows, equality rows included.
    pub const fn constraints(self) -> usize {
        self.constraints
    }
    /// Returns the number of finite bound sides on variables that are not fixed.
    pub const fn bound_sides(self) -> usize {
        self.bound_sides
    }
    const fn rows(self) -> usize {
        self.constraints + self.bound_sides
    }
}

/// Dense nonlinear and KKT storage behind [`InteriorPointWorkspace`].
pub(crate) struct InteriorPointStorage<S: RealScalar + NumericScalar<Real = S>> {
    shape: Option<InteriorPointWorkspaceRequirements>,
    accepted: Observation<S>,
    trial: Observation<S>,
    restored: Observation<S>,
    differences: Differences<S>,
    bounds: Vec<(Option<S>, Option<S>)>,
    sides: Vec<(usize, S, S)>,
    model_state: InteriorPointState<S>,
    state: InteriorPointState<S>,
    proposal: InteriorPointState<S>,
    hessian: Matrix<S>,
    trial_hessian: Matrix<S>,
    matrix: Matrix<S>,
    base_matrix: Matrix<S>,
    rhs: Vec<S>,
    direction: Vec<S>,
    slack_direction: Vec<S>,
    scratch: Vec<S>,
    displacement: Vec<S>,
    secant: Vec<S>,
    old_gradient: Vec<S>,
    update: crate::HessianUpdateWorkspace<S>,
    provider: Option<DenseKktProvider<S>>,
    injected: Option<Box<dyn KktProvider<S> + Send>>,
    restoration: Option<DenseSqpSubproblem<S>>,
    restoration_state: SqpState<S>,
    restoration_design: Matrix<S>,
    restoration_target: Vec<S>,
    empty_jacobian: Matrix<S>,
    step_bounds: Vec<(Option<S>, Option<S>)>,
    diagnostics: InteriorPointDiagnostics<S>,
    optimality: OptimalityDiagnostics<S>,
    iterations: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for InteriorPointStorage<S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("InteriorPointStorage")
            .field("shape", &self.shape)
            .field("injected_kkt_provider", &self.injected.is_some())
            .field("iterations", &self.iterations)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for InteriorPointStorage<S> {
    fn default() -> Self {
        Self {
            shape: None,
            accepted: Observation::new(),
            trial: Observation::new(),
            restored: Observation::new(),
            differences: Differences::new(),
            bounds: Vec::new(),
            sides: Vec::new(),
            model_state: InteriorPointState::empty(),
            state: InteriorPointState::empty(),
            proposal: InteriorPointState::empty(),
            hessian: Matrix::empty(),
            trial_hessian: Matrix::empty(),
            matrix: Matrix::empty(),
            base_matrix: Matrix::empty(),
            rhs: Vec::new(),
            direction: Vec::new(),
            slack_direction: Vec::new(),
            scratch: Vec::new(),
            displacement: Vec::new(),
            secant: Vec::new(),
            old_gradient: Vec::new(),
            update: crate::HessianUpdateWorkspace::new(),
            provider: None,
            injected: None,
            restoration: None,
            restoration_state: SqpState::new(),
            restoration_design: Matrix::empty(),
            restoration_target: Vec::new(),
            empty_jacobian: Matrix::empty(),
            step_bounds: Vec::new(),
            diagnostics: InteriorPointDiagnostics::default(),
            optimality: OptimalityDiagnostics::unavailable(),
            iterations: 0,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointStorage<S> {
    /// Allocate every buffer, the dense KKT provider and restoration storage.
    fn reserve(&mut self, shape: InteriorPointWorkspaceRequirements) -> Result<(), SciOxideError> {
        let (n, eq, m) = (shape.dimension, shape.equalities, shape.constraints);
        let rows = shape.rows();
        let order = scioxide_core::backend::checked_add(
            n,
            rows,
            scioxide_core::quantity!("interior_kkt_order"),
        )?;
        let vector = |len| {
            scioxide_core::backend::try_filled(
                len,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("interior_workspace"),
            )
        };
        let mut sides = Vec::new();
        scioxide_core::backend::try_reserve(
            &mut sides,
            scioxide_core::backend::checked_product(
                &[2, n],
                scioxide_core::quantity!("bound_side_count"),
            )?,
            scioxide_core::quantity!("bound_sides"),
        )?;
        let mut next = Self {
            shape: Some(shape),
            bounds: scioxide_core::backend::try_filled(
                n,
                (None, None),
                scioxide_core::quantity!("interior_bounds"),
            )?,
            sides,
            model_state: InteriorPointState::new(n, m, rows, eq)?,
            state: InteriorPointState::new(n, m, rows, eq)?,
            proposal: InteriorPointState::new(n, m, rows, eq)?,
            hessian: Matrix::zeros(n, n)?,
            trial_hessian: Matrix::zeros(n, n)?,
            matrix: Matrix::zeros(order, order)?,
            base_matrix: Matrix::zeros(order, order)?,
            rhs: vector(order)?,
            direction: vector(order)?,
            slack_direction: vector(rows)?,
            scratch: vector(n)?,
            displacement: vector(n)?,
            secant: vector(n)?,
            old_gradient: vector(n)?,
            provider: Some(DenseKktProvider::new(order)?),
            injected: None,
            restoration: Some(DenseSqpSubproblem::new(n, 0, m)?),
            restoration_design: Matrix::zeros(m, n)?,
            restoration_target: vector(m)?,
            empty_jacobian: Matrix::zeros(0, n)?,
            step_bounds: scioxide_core::backend::try_filled(
                n,
                (None, None),
                scioxide_core::quantity!("restoration_bounds"),
            )?,
            ..Self::default()
        };
        for observation in [&mut next.accepted, &mut next.trial, &mut next.restored] {
            observation.reserve(n, m)?;
        }
        next.update.try_reserve(n)?;
        next.restoration_state.reserve(n, 0)?;
        next.injected = self.injected.take();
        *self = next;
        Ok(())
    }
    /// Load the problem's intersected box, its bound sides and inward differences.
    fn load(&mut self, constraints: &Constraints<S>) -> Result<(), SciOxideError> {
        self.bounds.copy_from_slice(constraints.hard_box());
        self.sides.clear();
        for (j, &(lo, hi)) in self.bounds.iter().enumerate() {
            if lo.is_some() && lo == hi {
                continue;
            }
            if let Some(v) = lo {
                self.sides.push((j, -S::ONE, v));
            }
            if let Some(v) = hi {
                self.sides.push((j, S::ONE, v));
            }
        }
        if self
            .shape
            .is_none_or(|shape| shape.bound_sides != self.sides.len())
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace shape differs from the problem's box",
            ));
        }
        self.differences.prepare(&self.bounds)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::attempt::Accepted<S>
    for InteriorPointStorage<S>
{
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
impl<S: RealScalar + NumericScalar<Real = S>> Shape<InteriorPointStorage<S>>
    for InteriorPointWorkspaceRequirements
{
    fn fits(&self, storage: &InteriorPointStorage<S>) -> bool {
        storage.shape == Some(*self)
    }
    fn prepare(&self, storage: &mut InteriorPointStorage<S>) -> Result<(), SciOxideError> {
        storage.reserve(*self)
    }
}

crate::attempt::local_workspace!(
    /// Stores the iterates, multipliers and KKT solver of an [`InteriorPoint`]
    /// solve.
    ///
    /// KKT systems are solved by [`DenseKktProvider`] unless
    /// [`with_kkt_provider`](InteriorPointWorkspace::with_kkt_provider) sets
    /// another solver.
    InteriorPointWorkspace,
    InteriorPointStorage<S>,
    InteriorPointWorkspaceRequirements
);
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointWorkspace<S> {
    /// Solves every later KKT system with `provider`.
    ///
    /// The systems have order `dimension + constraints + bound_sides`, as given
    /// by [`InteriorPointWorkspaceRequirements`]. The solver checks the residual
    /// of each direction.
    #[must_use]
    pub fn with_kkt_provider(mut self, provider: impl KktProvider<S> + Send + 'static) -> Self {
        self.storage.injected = Some(Box::new(provider));
        self
    }
    /// Returns the multipliers and slacks at the last accepted point.
    pub fn state(&self) -> &InteriorPointState<S> {
        &self.storage.state
    }
}

/// Minimizes a smooth nonlinear program by a primal-dual interior-point method.
///
/// ```text
/// minimize f(x)  subject to  h(x) = 0,  g(x) ≤ 0,  l ≤ x ≤ u
/// ```
///
/// The method adds a slack `sᵢ > 0` to each inequality row and finite bound
/// side and follows the central path of the log-barrier problem as the
/// barrier parameter `μ` decreases. It starts at `μ = 0.1` and multiplies `μ`
/// by 0.2 whenever the barrier problem is solved to within `10 μ`. Each
/// iteration solves one symmetric indefinite KKT system for a Newton direction.
/// If the inertia of that system is wrong, the Hessian block is shifted and the
/// system factored again. A fraction-to-the-boundary rule with `τ = 0.995`
/// keeps slacks positive, and a backtracking search on the merit function
/// `f(x) − μ Σ ln sᵢ + ρ Σ |cᵢ(x) + sᵢ|` accepts the step, with one
/// second-order correction. An infeasible start, or a step that is not
/// accepted, triggers a feasibility restoration by bounded linear least
/// squares that stops halfway to each bound.
///
/// The bounds are the intersected box of the problem's [`Constraints`].
/// Iterates stay strictly inside every bound of a variable that is not fixed.
/// The Hessian of the Lagrangian comes from a [`LagrangianHessian`] attached
/// to the problem, or from a damped BFGS approximation otherwise. KKT systems
/// are solved with [`DenseKktProvider`] unless the workspace has another
/// [`KktProvider`].
///
/// # References
///
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
///   2006, Chapter 19.
/// - A. Wächter and L. T. Biegler, "On the implementation of an interior-point
///   filter line-search algorithm for large-scale nonlinear programming",
///   *Math. Program.* 106(1), 25–57, 2006 (inertia correction).
#[derive(Debug, Clone, Default)]
pub struct InteriorPoint<S: RealScalar + NumericScalar<Real = S>> {
    options: InteriorPointOptions<S>,
}
fn shape<S: RealScalar + NumericScalar<Real = S>>(constraints: &Constraints<S>) -> (usize, usize) {
    (
        constraints.profile().equality_count(),
        constraints.nonlinear_functions().count(),
    )
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPoint<S> {
    /// Creates a solver with `options`.
    pub const fn new(options: InteriorPointOptions<S>) -> Self {
        Self { options }
    }
    /// Returns the options.
    pub const fn options(&self) -> &InteriorPointOptions<S> {
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
    ) -> Result<InteriorPointWorkspaceRequirements, SciOxideError> {
        let (equalities, constraints) = shape(problem.constraints());
        let bound_sides = problem
            .constraints()
            .hard_box()
            .iter()
            .filter(|(lo, hi)| lo.is_none() || lo != hi)
            .map(|(lo, hi)| usize::from(lo.is_some()) + usize::from(hi.is_some()))
            .sum();
        Ok(InteriorPointWorkspaceRequirements {
            dimension: problem.dim(),
            equalities,
            constraints,
            bound_sides,
        })
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(
        &self,
        problem: &ConstrainedProblem<'_, S>,
    ) -> Result<InteriorPointWorkspace<S>, SciOxideError> {
        let mut workspace = InteriorPointWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        workspace.storage.load(problem.constraints())?;
        Ok(workspace)
    }
    /// Minimizes `problem` from `x0` and writes the result to `output`.
    ///
    /// The starting point is moved strictly inside the bounds first.
    /// Restoration subproblems run as nested solves of `scope`. See
    /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0` does not have the problem's dimension.
    /// - `NonFinite` if `x0` contains NaN or infinity, or if the objective or a
    ///   constraint is not finite at the moved start.
    /// - `InvalidInput` if the moved start is outside the objective domain, or
    ///   the [`LagrangianHessian`] returns a nonsymmetric matrix.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - Errors from the callbacks and the KKT provider, unchanged.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the solve before the first accepted point, as described
    ///   in [Termination](crate#termination).
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut InteriorPointWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::utils::validate_x0(problem.objective(), x0)?;
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        crate::attempt::preflight(&scope, self.options.time_limit())?;
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
    /// Identify method, constraint shape, box, curvature source and options.
    fn identity(&self, problem: &ConstrainedProblem<'_, S>) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "InteriorPoint", 2),
            problem.dim(),
            &(
                &self.options,
                problem.constraints().profile(),
                problem.constraints().hard_box(),
                problem.lagrangian_hessian().is_some(),
            ),
        )
    }
    fn minimize_in_context(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        out: &mut OptimizeResult<S>,
        w: &mut InteriorPointStorage<S>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        w.accepted.value = None;
        if let Some(limit) = self.options.time_limit() {
            ctx.limit_time(limit)?;
        }
        out.try_reserve(problem.dim())?;
        let session = SolveSession::new(ctx, scioxide_core::argument!("scope"));
        // Lend the provider without leaving the workspace unprepared on error.
        let mut injected = w.injected.take();
        let mut dense = w.provider.take();
        let result = w.load(problem.constraints()).and_then(|()| {
            match (injected.as_deref_mut(), dense.as_mut()) {
                (Some(provider), _) => self.kernel(problem, x0, w, provider, &session),
                (None, Some(provider)) => self.kernel(problem, x0, w, provider, &session),
                (None, None) => Err(invalid(
                    scioxide_core::argument!("workspace"),
                    "workspace dense provider unavailable",
                )),
            }
        });
        w.injected = injected;
        w.provider = dense;
        if let Err(e) = &result {
            session.record_execution_stop(e);
        }
        let stop = match result {
            Ok(s) => session.stopped().unwrap_or(s),
            Err(e) => match (session.stopped(), w.accepted.value) {
                (Some(s), Some(_)) => s,
                _ => return Err(e),
            },
        };
        self.finish(problem, out, w, stop, &session)
    }
    fn finish(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        out: &mut OptimizeResult<S>,
        w: &mut InteriorPointStorage<S>,
        stop: OptimizeTermination,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        w.state.copy_from(&w.model_state);
        super::sqp::unscale_state(&w.accepted, &w.model_state.dual, &mut w.state.dual)?;
        for i in 0..w.model_state.lambda.len() {
            let row = w
                .accepted
                .row_scales
                .get(i)
                .copied()
                .unwrap_or(<S as scioxide_core::AlgebraicScalar>::ONE);
            w.state.lambda[i] = scioxide_core::checked_float_product(
                scioxide_core::checked_float_quotient(w.model_state.lambda[i], row)?,
                w.accepted.objective_scale,
            )?;
            w.state.slack[i] = scioxide_core::checked_float_product(w.model_state.slack[i], row)?;
        }
        crate::utils::commit_optimize_result(
            out,
            &w.accepted.x,
            w.accepted
                .original_value
                .ok_or_else(|| invalid(scioxide_core::argument!("f"), "no accepted objective"))?,
            (w.accepted.derivatives && w.accepted.fixed_derivatives)
                .then_some(w.accepted.original_gradient.as_slice()),
            w.iterations,
            stop,
            "Interior point completed; inspect original-coordinate KKT residuals",
        )?;
        out.diagnostics = super::sqp::original_diagnostics(
            &w.accepted,
            &w.model_state.dual,
            w.optimality,
            shape(problem.constraints()).0,
        )?;
        let mut diagnostics = w.diagnostics;
        diagnostics.fixed_derivatives_available = w.accepted.fixed_derivatives;
        out.method = crate::OptimizeDiagnostics::InteriorPoint(diagnostics);
        let trust = if out.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        };
        session.assess(trust)?;
        crate::utils::synchronize_metadata(out, session, trust)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPoint<S> {
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
    crate::traits::Optimizer<S, ConstrainedProblem<'a, S>> for InteriorPoint<S>
{
    fn minimize(
        &self,
        problem: &ConstrainedProblem<'a, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(problem, x0)
    }
}
#[cfg(test)]
mod tests;
