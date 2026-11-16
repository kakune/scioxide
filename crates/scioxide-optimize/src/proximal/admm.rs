//! Two-block alternating direction method of multipliers in scaled form.

use alloc::vec::Vec;
use core::time::Duration;
use scioxide_core::AlgebraicScalar;
use scioxide_core::Argument;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AdjointOperator, AttemptKind, AttemptRecord, Budget, BudgetKind, EvaluationRole,
    GradientTolerance, LinearOperator, NonFiniteOrigin, NumericScalar, PartialResultTrust,
    RealScalar, ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveScope,
};

use super::{finite, norm, positive, shape};
use crate::OptimizeTermination;
use crate::attempt::{Entry, Shape, Shell, Staged, assign, preflight, reserve_len};

/// Names the block that an ADMM update minimizes over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmmBlock {
    /// The `x` block.
    X,
    /// The `z` block.
    Z,
}

/// Passes the problem data and the current iterate to an [`AdmmBlockUpdate`].
///
/// For the `x` block the update minimizes over `x` with `other = z` fixed, and
/// for the `z` block over `z` with `other = x` fixed, as described on
/// [`AdmmBlockUpdate`].
#[derive(Clone, Copy)]
pub struct AdmmUpdateInput<'a, S: NumericScalar> {
    block: AdmmBlock,
    current: &'a [S],
    other: &'a [S],
    scaled_dual: &'a [S],
    rhs: &'a [S],
    operator: &'a dyn LinearOperator<S>,
    other_operator: &'a dyn LinearOperator<S>,
    rho: S::Real,
}

impl<'a, S: NumericScalar> AdmmUpdateInput<'a, S> {
    /// Returns the block being updated.
    pub const fn block(&self) -> AdmmBlock {
        self.block
    }
    /// Returns the current value of the block being updated.
    pub const fn current(&self) -> &'a [S] {
        self.current
    }
    /// Returns the latest value of the other block.
    pub const fn other(&self) -> &'a [S] {
        self.other
    }
    /// Returns the scaled dual variable `u = y/ρ`.
    pub const fn scaled_dual(&self) -> &'a [S] {
        self.scaled_dual
    }
    /// Returns the right-hand side `c`.
    pub const fn rhs(&self) -> &'a [S] {
        self.rhs
    }
    /// Returns the operator of this block (`A` or `B`).
    pub fn operator(&self) -> &'a dyn LinearOperator<S> {
        self.operator
    }
    /// Returns the operator of the other block.
    pub fn other_operator(&self) -> &'a dyn LinearOperator<S> {
        self.other_operator
    }
    /// Returns the current penalty parameter `ρ > 0`.
    pub const fn rho(&self) -> S::Real {
        self.rho
    }
}

/// Minimizes the augmented Lagrangian over one ADMM block.
///
/// For the `x` block, write to `output` an approximate solution of
///
/// ```text
/// minₓ f(x) + (ρ/2) ‖A x + B z − c + u‖²,
/// ```
///
/// and for the `z` block of `min_z g(z) + (ρ/2) ‖A x + B z − c + u‖²`, using
/// the values in [`AdmmUpdateInput`]. Return an [`AdmmUpdateReport`] that says
/// whether the inner method finished and how large its stationarity residual
/// is. An inner solver that runs in `context` counts its work toward the
/// limits of the ADMM solve.
pub trait AdmmBlockUpdate<S: NumericScalar> {
    /// Writes the updated block to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    fn update_into(
        &self,
        input: AdmmUpdateInput<'_, S>,
        output: &mut [S],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<S::Real>, SciOxideError>;
}

/// Reports how accurately an [`AdmmBlockUpdate`] solved its subproblem.
///
/// The ADMM solve accepts the update only if the inner method finished and
/// the stationarity residual meets the subproblem tolerance of
/// [`AdmmOptions`]. The solver does not check the residual itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmmUpdateReport<R: RealScalar> {
    converged: bool,
    stationarity_norm: R,
}

impl<R: RealScalar> AdmmUpdateReport<R> {
    /// Creates a report from whether the inner method finished and its
    /// nonnegative stationarity residual.
    pub const fn new(converged: bool, stationarity_norm: R) -> Self {
        Self {
            converged,
            stationarity_norm,
        }
    }
    /// Returns whether the inner method finished.
    pub const fn converged(self) -> bool {
        self.converged
    }
    /// Returns the stationarity residual of the subproblem.
    pub const fn stationarity_norm(self) -> R {
        self.stationarity_norm
    }
}

/// Borrows a problem `min f(x) + g(z)` subject to `A x + B z = c`.
///
/// `f` and `g` enter only through their block updates. `A` must provide
/// adjoint products, because the dual residual uses `Aᴴ`.
pub struct AdmmProblem<'a, S: NumericScalar> {
    a: &'a dyn AdjointOperator<S>,
    b: &'a dyn LinearOperator<S>,
    c: &'a [S],
    x_update: &'a dyn AdmmBlockUpdate<S>,
    z_update: &'a dyn AdmmBlockUpdate<S>,
}

impl<'a, S: NumericScalar> AdmmProblem<'a, S> {
    /// Creates a problem from `A`, `B`, `c` and the two block updates.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `A`, `B` and `c` do not have the same number of
    ///   rows.
    /// - `NonFinite` if `c` contains NaN or infinity.
    pub fn new(
        a: &'a dyn AdjointOperator<S>,
        b: &'a dyn LinearOperator<S>,
        c: &'a [S],
        x_update: &'a dyn AdmmBlockUpdate<S>,
        z_update: &'a dyn AdmmBlockUpdate<S>,
    ) -> Result<Self, SciOxideError> {
        shape(a.rows(), b.rows())?;
        shape(a.rows(), c.len())?;
        finite(c, NonFiniteOrigin::Input)?;
        Ok(Self {
            a,
            b,
            c,
            x_update,
            z_update,
        })
    }
    /// Returns the number of `x` variables.
    pub fn x_dimension(&self) -> usize {
        self.a.cols()
    }
    /// Returns the number of `z` variables.
    pub fn z_dimension(&self) -> usize {
        self.b.cols()
    }
    /// Returns the number of equality constraints.
    pub fn constraints(&self) -> usize {
        self.c.len()
    }
    fn operator_scratch(&self) -> Result<usize, SciOxideError> {
        Ok(self
            .a
            .apply_workspace_len()?
            .max(self.a.apply_adjoint_workspace_len()?)
            .max(self.b.apply_workspace_len()?))
    }
}

/// Borrows the starting point `x`, `z` and scaled dual `u = y/ρ` of an ADMM
/// solve.
#[derive(Debug, Clone, Copy)]
pub struct AdmmStart<'a, S> {
    x: &'a [S],
    z: &'a [S],
    scaled_dual: &'a [S],
}

impl<'a, S> AdmmStart<'a, S> {
    /// Creates a start from `x`, `z` and the scaled dual `u`.
    pub const fn new(x: &'a [S], z: &'a [S], scaled_dual: &'a [S]) -> Self {
        Self { x, z, scaled_dual }
    }
    /// Returns the starting `x`.
    pub const fn x(&self) -> &'a [S] {
        self.x
    }
    /// Returns the starting `z`.
    pub const fn z(&self) -> &'a [S] {
        self.z
    }
    /// Returns the starting scaled dual `u`.
    pub const fn scaled_dual(&self) -> &'a [S] {
        self.scaled_dual
    }
}

/// Configures [`Admm`].
///
/// The solve stops successfully when both residuals meet the residual
/// tolerance: the primal residual `r = A x + B z − c` relative to
/// `max(‖A x‖, ‖B z‖, ‖c‖)`, and the dual residual `s = ρ Aᴴ B (z − z_prev)`
/// relative to `‖ρ Aᴴ u‖`, all in the Euclidean norm. See also
/// [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct AdmmOptions<R: RealScalar> {
    rho: R,
    residual: ResidualTolerance<R>,
    subproblem: GradientTolerance<R>,
    iterations: Budget,
    balancing: Option<(R, R)>,
    time_limit: Option<Duration>,
}

impl<R: RealScalar> AdmmOptions<R> {
    /// Creates options with the fixed penalty `rho`, at most 1000 iterations and
    /// default tolerances. The solver constructor checks that `rho` is positive
    /// and finite.
    pub fn new(rho: R) -> Self {
        Self {
            rho,
            residual: ResidualTolerance::default(),
            subproblem: GradientTolerance::default(),
            iterations: Budget::ONE.saturating_add(999),
            balancing: None,
            time_limit: None,
        }
    }
    /// Sets the tolerance on the primal and dual residuals.
    #[must_use]
    pub const fn with_residual_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.residual = tolerance;
        self
    }
    /// Sets the tolerance on the stationarity residual reported by each block
    /// update.
    #[must_use]
    pub const fn with_subproblem_tolerance(mut self, tolerance: GradientTolerance<R>) -> Self {
        self.subproblem = tolerance;
        self
    }
    /// Sets the iteration limit.
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: Budget) -> Self {
        self.iterations = limit;
        self
    }
    /// Enables residual balancing: when one residual norm exceeds `ratio` times
    /// the other, `ρ` is multiplied or divided by `factor`, and `u` is rescaled
    /// so that the unscaled dual `y = ρu` does not change. The solver
    /// constructor requires both values to be finite and greater than one.
    #[must_use]
    pub const fn with_residual_balancing(mut self, ratio: R, factor: R) -> Self {
        self.balancing = Some((ratio, factor));
        self
    }
    /// Sets a time limit. A solve then needs an execution context with a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, limit: Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }
    /// Returns the initial penalty `ρ`.
    pub const fn rho(&self) -> R {
        self.rho
    }
    /// Returns the residual tolerance.
    pub const fn residual_tolerance(&self) -> ResidualTolerance<R> {
        self.residual
    }
    /// Returns the subproblem tolerance.
    pub const fn subproblem_tolerance(&self) -> GradientTolerance<R> {
        self.subproblem
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.iterations
    }
    /// Returns the residual-balancing ratio and factor, or `None` if disabled.
    pub const fn residual_balancing(&self) -> Option<(R, R)> {
        self.balancing
    }
    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        positive(scioxide_core::argument!("rho"), self.rho)?;
        if let Some((ratio, factor)) = self.balancing {
            positive(scioxide_core::argument!("ratio"), ratio)?;
            positive(scioxide_core::argument!("factor"), factor)?;
            for (argument, value) in [
                (scioxide_core::argument!("ratio"), ratio),
                (scioxide_core::argument!("factor"), factor),
            ] {
                if value <= R::ONE {
                    return Err(SciOxideError::invalid_input(
                        argument,
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "balancing ratio and factor must exceed one",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Describes the workspace size of an [`Admm`] solve for one problem shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmmWorkspaceRequirements {
    x: usize,
    z: usize,
    constraints: usize,
    operator: usize,
    bytes: usize,
}

impl AdmmWorkspaceRequirements {
    fn new<S: NumericScalar>(problem: &AdmmProblem<'_, S>) -> Result<Self, SciOxideError> {
        let (x, z, constraints) = (
            problem.x_dimension(),
            problem.z_dimension(),
            problem.constraints(),
        );
        let operator = problem.operator_scratch()?;
        let q = scioxide_core::quantity!("admm_workspace");
        // Workspace: 5 x-, 3 z- and 9 constraint-length vectors plus operator
        // scratch; staged result: 2 x-, 1 z- and 2 constraint-length vectors.
        let elements = [
            checked_mul(x, 7, q)?,
            checked_mul(z, 4, q)?,
            checked_mul(constraints, 11, q)?,
            operator,
        ]
        .into_iter()
        .try_fold(0_usize, |total, part| checked_add(total, part, q))?;
        Ok(Self {
            x,
            z,
            constraints,
            operator,
            bytes: checked_mul(elements, core::mem::size_of::<S>(), q)?,
        })
    }
    /// Returns the number of `x` variables.
    pub const fn x_dimension(self) -> usize {
        self.x
    }
    /// Returns the number of `z` variables.
    pub const fn z_dimension(self) -> usize {
        self.z
    }
    /// Returns the number of equality constraints.
    pub const fn constraints(self) -> usize {
        self.constraints
    }
    /// Returns the workspace size in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Summarizes the last accepted iterate of an ADMM call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmmProgress<R> {
    iterations: usize,
    primal_norm: R,
    dual_norm: R,
    trust: PartialResultTrust,
}

impl<R: Copy> AdmmProgress<R> {
    /// Returns the number of completed iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the primal residual norm `‖A x + B z − c‖`.
    pub const fn primal_norm(self) -> R {
        self.primal_norm
    }
    /// Returns the dual residual norm `‖ρ Aᴴ B (z − z_prev)‖`.
    pub const fn dual_norm(self) -> R {
        self.dual_norm
    }
    /// Returns how far the accepted iterate has been checked.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

#[derive(Debug)]
struct Storage<S: NumericScalar> {
    x: Vec<S>,
    z: Vec<S>,
    u: Vec<S>,
    primal: Vec<S>,
    dual: Vec<S>,
    trial_u: Vec<S>,
    trial_primal: Vec<S>,
    trial_dual: Vec<S>,
    trial_x: Vec<S>,
    trial_z: Vec<S>,
    ax: Vec<S>,
    bz: Vec<S>,
    previous_bz: Vec<S>,
    difference: Vec<S>,
    adjoint: Vec<S>,
    operator: Vec<S>,
    completed: usize,
    norms: (S::Real, S::Real),
}

impl<S: NumericScalar> Shape<Storage<S>> for AdmmWorkspaceRequirements {
    fn fits(&self, w: &Storage<S>) -> bool {
        (w.x.len(), w.z.len(), w.u.len()) == (self.x, self.z, self.constraints)
            && w.operator.len() >= self.operator
    }
    fn prepare(&self, storage: &mut Storage<S>) -> Result<(), SciOxideError> {
        let (n, p, m) = (self.x, self.z, self.constraints);
        let allocate = |length| {
            try_filled(
                length,
                S::ZERO,
                scioxide_core::quantity!("workspace_vectors"),
            )
        };
        *storage = Storage {
            x: allocate(n)?,
            z: allocate(p)?,
            u: allocate(m)?,
            primal: allocate(m)?,
            dual: allocate(n)?,
            trial_u: allocate(m)?,
            trial_primal: allocate(m)?,
            trial_dual: allocate(n)?,
            trial_x: allocate(n)?,
            trial_z: allocate(p)?,
            ax: allocate(m)?,
            bz: allocate(m)?,
            previous_bz: allocate(m)?,
            difference: allocate(m)?,
            adjoint: allocate(n)?,
            operator: allocate(self.operator)?,
            completed: 0,
            norms: (S::Real::ZERO, S::Real::ZERO),
        };
        Ok(())
    }
}

/// Holds the storage of an [`Admm`] solve and a record of the last call.
#[derive(Debug)]
pub struct AdmmWorkspace<S: NumericScalar> {
    storage: Storage<S>,
    staged: AdmmResult<S>,
    shell: Shell<AdmmProgress<S::Real>>,
}

impl<S: NumericScalar> Default for AdmmWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: NumericScalar> AdmmWorkspace<S> {
    /// Creates an empty workspace; reserve storage with [`Self::try_reserve`].
    pub fn new() -> Self {
        Self {
            storage: Storage {
                x: Vec::new(),
                z: Vec::new(),
                u: Vec::new(),
                primal: Vec::new(),
                dual: Vec::new(),
                trial_u: Vec::new(),
                trial_primal: Vec::new(),
                trial_dual: Vec::new(),
                trial_x: Vec::new(),
                trial_z: Vec::new(),
                ax: Vec::new(),
                bz: Vec::new(),
                previous_bz: Vec::new(),
                difference: Vec::new(),
                adjoint: Vec::new(),
                operator: Vec::new(),
                completed: 0,
                norms: (S::Real::ZERO, S::Real::ZERO),
            },
            staged: AdmmResult::empty(),
            shell: Shell::new(),
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Never fails; this workspace has no byte limit.
    pub fn check(
        &self,
        requirements: &AdmmWorkspaceRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        Ok(
            if requirements.fits(&self.storage) && self.staged.fits(requirements) {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Reserves storage for `requirements`. On failure the existing storage is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(
        &mut self,
        requirements: &AdmmWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        if !requirements.fits(&self.storage) {
            requirements.prepare(&mut self.storage)?;
        }
        self.staged.try_reserve(requirements)
    }
    /// Returns the record of the most recent call, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<AdmmProgress<S::Real>>> {
        self.shell.last_attempt()
    }
    /// Returns the last accepted `x` and `z` of the most recent call, or `None`.
    pub fn last_accepted(&self) -> Option<(&[S], &[S])> {
        self.shell
            .progress()
            .filter(|progress| progress.iterations > 0)
            .map(|_| (self.storage.x.as_slice(), self.storage.z.as_slice()))
    }
}

/// Holds the result of an [`Admm`] solve.
///
/// The residuals are computed by the solver from `A`, `B` and `c`, not taken
/// from the block updates.
#[derive(Debug, Clone)]
pub struct AdmmResult<S: NumericScalar> {
    x: Vec<S>,
    z: Vec<S>,
    scaled_dual: Vec<S>,
    primal: Vec<S>,
    dual: Vec<S>,
    primal_norm: S::Real,
    dual_norm: S::Real,
    rho: S::Real,
    iterations: usize,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
}

impl<S: NumericScalar> Default for AdmmResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> AdmmResult<S> {
    /// Returns an empty result, without allocating, to pass to
    /// [`Admm::solve_into`].
    pub fn empty() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            scaled_dual: Vec::new(),
            primal: Vec::new(),
            dual: Vec::new(),
            primal_norm: S::Real::ZERO,
            dual_norm: S::Real::ZERO,
            rho: S::Real::ZERO,
            iterations: 0,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    /// Reserves storage for a result of the shape in `requirements`.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &AdmmWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("admm_result");
        reserve_len(&mut self.x, requirements.x, q)?;
        reserve_len(&mut self.z, requirements.z, q)?;
        reserve_len(&mut self.scaled_dual, requirements.constraints, q)?;
        reserve_len(&mut self.primal, requirements.constraints, q)?;
        reserve_len(&mut self.dual, requirements.x, q)
    }
    fn fits(&self, requirements: &AdmmWorkspaceRequirements) -> bool {
        self.x.capacity() >= requirements.x
            && self.dual.capacity() >= requirements.x
            && self.z.capacity() >= requirements.z
            && self.scaled_dual.capacity() >= requirements.constraints
            && self.primal.capacity() >= requirements.constraints
    }
    /// Returns the last accepted `x`.
    pub fn x(&self) -> &[S] {
        &self.x
    }
    /// Returns the last accepted `z`.
    pub fn z(&self) -> &[S] {
        &self.z
    }
    /// Returns the last accepted scaled dual `u = y/ρ`.
    pub fn scaled_dual(&self) -> &[S] {
        &self.scaled_dual
    }
    /// Returns the primal residual `A x + B z − c`, or `None` before the first
    /// complete iteration.
    pub fn primal_residual(&self) -> Option<&[S]> {
        (self.iterations > 0).then_some(&self.primal)
    }
    /// Returns the dual residual `ρ Aᴴ B (z − z_prev)`, or `None` before the
    /// first complete iteration.
    pub fn dual_residual(&self) -> Option<&[S]> {
        (self.iterations > 0).then_some(&self.dual)
    }
    /// Returns the Euclidean norm of the primal residual, or `None` before the
    /// first complete iteration.
    pub const fn primal_norm(&self) -> Option<S::Real> {
        if self.iterations == 0 {
            None
        } else {
            Some(self.primal_norm)
        }
    }
    /// Returns the Euclidean norm of the dual residual, or `None` before the
    /// first complete iteration.
    pub const fn dual_norm(&self) -> Option<S::Real> {
        if self.iterations == 0 {
            None
        } else {
            Some(self.dual_norm)
        }
    }
    /// Returns the final penalty `ρ`.
    pub const fn rho(&self) -> S::Real {
        self.rho
    }
    /// Returns the number of completed iterations.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns why the solver stopped.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the iteration and evaluation counts.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns whether both residual tests passed.
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
            None,
        )
        .map_or(Ok(()), Err)
    }
}

impl<S: NumericScalar> Staged for AdmmResult<S> {
    fn reset(&mut self) {
        for values in [
            &mut self.x,
            &mut self.z,
            &mut self.scaled_dual,
            &mut self.primal,
            &mut self.dual,
        ] {
            values.clear();
        }
        self.iterations = 0;
        self.termination = OptimizeTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("admm_result");
        reserve_len(&mut self.x, source.x.len(), q)?;
        reserve_len(&mut self.z, source.z.len(), q)?;
        reserve_len(&mut self.scaled_dual, source.scaled_dual.len(), q)?;
        reserve_len(&mut self.primal, source.primal.len(), q)?;
        reserve_len(&mut self.dual, source.dual.len(), q)?;
        assign(&mut self.x, &source.x);
        assign(&mut self.z, &source.z);
        assign(&mut self.scaled_dual, &source.scaled_dual);
        assign(&mut self.primal, &source.primal);
        assign(&mut self.dual, &source.dual);
        self.primal_norm = source.primal_norm;
        self.dual_norm = source.dual_norm;
        self.rho = source.rho;
        self.iterations = source.iterations;
        self.termination = source.termination;
        self.counts = source.counts;
        Ok(())
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}

/// Solves `min f(x) + g(z)` subject to `A x + B z = c` by the alternating
/// direction method of multipliers (ADMM).
///
/// Each iteration of the scaled form updates
///
/// ```text
/// x ← argminₓ f(x) + (ρ/2) ‖A x + B z − c + u‖²
/// z ← argmin_z g(z) + (ρ/2) ‖A x + B z − c + u‖²
/// u ← u + A x + B z − c
/// ```
///
/// where `u = y/ρ` is the scaled dual variable. You supply the two
/// minimizations as [`AdmmBlockUpdate`]s, so `f` and `g` may be nonsmooth.
/// Besides the two block updates, each iteration applies `A` once, `B` twice
/// and `Aᴴ` twice to form the residuals. For closed proper convex `f` and `g`
/// the residuals converge to zero; the convergence rate is typically linear at
/// best and depends strongly on `ρ`. Variables may be real or complex.
/// The primal residual sums `A x + B z − c` before rounding each component,
/// so cancelling terms do not overflow prematurely.
///
/// # Errors
///
/// - `InvalidInput` from [`Self::new`] if `ρ` is not positive and finite, or a
///   balancing ratio or factor is not finite and greater than one.
/// - `DimensionMismatch` if the start does not match the problem shape.
/// - `NonFinite` if the start is not finite, or an update or operator returns
///   NaN or infinity.
/// - `InvalidInput` (`Inconsistent`) if a block update reports that it did not
///   finish, or that its stationarity residual misses the subproblem tolerance.
/// - `CapabilityUnavailable` if a time limit is set and the context has no
///   clock.
/// - Any error returned by a block update or an operator.
///
/// # References
///
/// S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein, "Distributed
/// optimization and statistical learning via the alternating direction method
/// of multipliers", *Foundations and Trends in Machine Learning* 3(1), 1–122,
/// 2011, §3. <https://doi.org/10.1561/2200000016>
#[derive(Debug, Clone)]
pub struct Admm<R: RealScalar> {
    options: AdmmOptions<R>,
}

impl<R: RealScalar> Admm<R> {
    /// Creates a solver with checked options.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the penalty is not positive and finite, or if
    /// a balancing ratio or factor is not finite or not greater than one.
    pub fn new(options: AdmmOptions<R>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self { options })
    }
    /// Returns the options.
    pub const fn options(&self) -> &AdmmOptions<R> {
        &self.options
    }
    /// Returns the workspace size for `problem`, including operator scratch.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow`, or an error from an operator's scratch-size
    /// query.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        problem: &AdmmProblem<'_, S>,
    ) -> Result<AdmmWorkspaceRequirements, SciOxideError> {
        AdmmWorkspaceRequirements::new(problem)
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub fn workspace<S: NumericScalar<Real = R>>(
        &self,
        problem: &AdmmProblem<'_, S>,
    ) -> Result<AdmmWorkspace<S>, SciOxideError> {
        let mut workspace = AdmmWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(workspace)
    }
    /// Solves from `start` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// As [`Self::solve_into`].
    pub fn solve<S: NumericScalar<Real = R>>(
        &self,
        problem: &AdmmProblem<'_, S>,
        start: AdmmStart<'_, S>,
    ) -> Result<AdmmResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut output = AdmmResult::empty();
        self.solve_into(
            problem,
            start,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Solves from `start` and writes the result to `output`.
    ///
    /// Block updates receive the solve context and may run nested solves in
    /// it. An iteration or evaluation limit, cancellation or a deadline ends
    /// the solve with an unsuccessful result. On error, `output` is unchanged.
    ///
    /// # Errors
    ///
    /// See the type-level `# Errors` section. Also returns
    /// `AllocationFailure` if the workspace must grow and cannot.
    pub fn solve_into<'scope, 'capability, S: NumericScalar<Real = R>>(
        &self,
        problem: &AdmmProblem<'_, S>,
        start: AdmmStart<'_, S>,
        output: &mut AdmmResult<S>,
        workspace: &mut AdmmWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let requirements = self.workspace_requirements(problem)?;
        for (expected, actual) in [
            (requirements.x, start.x.len()),
            (requirements.z, start.z.len()),
            (requirements.constraints, start.scaled_dual.len()),
        ] {
            shape(expected, actual)?;
        }
        for values in [start.x, start.z, start.scaled_dual] {
            finite(values, NonFiniteOrigin::Input)?;
        }
        workspace.try_reserve(&requirements)?;
        preflight(&scope, self.options.time_limit)?;
        let dimension = checked_add(
            checked_add(
                requirements.x,
                requirements.z,
                scioxide_core::quantity!("solve_dimension"),
            )?,
            requirements.constraints,
            scioxide_core::quantity!("solve_dimension"),
        )?;
        let entry = Entry::Call {
            identity: crate::context::local_identity(
                scioxide_core::algorithm_id!("optimize", "Admm", 1),
                dimension,
                &self.options,
            ),
            budgets: SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, self.options.iterations),
        };
        crate::attempt::run(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| run(problem, start, &self.options, staged, storage, context),
            |storage, succeeded| {
                Some(AdmmProgress {
                    iterations: storage.completed,
                    primal_norm: storage.norms.0,
                    dual_norm: storage.norms.1,
                    trust: crate::attempt::trust(succeeded),
                })
            },
        )
    }
}

fn apply<S: NumericScalar>(
    op: &dyn LinearOperator<S>,
    x: &[S],
    out: &mut [S],
    scratch: &mut [S],
    context: &mut SolveContext<'_, '_>,
) -> Result<bool, SciOxideError> {
    if !context
        .record_provider(EvaluationRole::LinearOperator)?
        .is_granted()
    {
        return Ok(false);
    }
    op.apply_into_scoped(
        x,
        out,
        &mut scratch[..op.apply_workspace_len()?],
        &mut scioxide_core::OperatorContext::new(context),
    )
    .and_then(scioxide_core::Invocation::require)?;
    finite(out, NonFiniteOrigin::Callback)?;
    Ok(true)
}

fn publish<S: NumericScalar>(
    output: &mut AdmmResult<S>,
    w: &mut Storage<S>,
    (pnorm, dnorm, rho, iterations): (S::Real, S::Real, S::Real, usize),
    termination: OptimizeTermination,
    context: &SolveContext<'_, '_>,
) {
    w.completed = iterations;
    w.norms = (pnorm, dnorm);
    assign(&mut output.x, &w.x);
    assign(&mut output.z, &w.z);
    assign(&mut output.scaled_dual, &w.u);
    assign(&mut output.primal, &w.primal);
    assign(&mut output.dual, &w.dual);
    output.primal_norm = pnorm;
    output.dual_norm = dnorm;
    output.rho = rho;
    output.iterations = iterations;
    output.termination = termination;
    output.counts = context.counts();
}

fn run<S: NumericScalar>(
    problem: &AdmmProblem<'_, S>,
    start: AdmmStart<'_, S>,
    options: &AdmmOptions<S::Real>,
    output: &mut AdmmResult<S>,
    w: &mut Storage<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    if let Some(limit) = options.time_limit {
        context.limit_time(limit)?;
    }
    let (x_update, z_update) = (problem.x_update, problem.z_update);
    w.completed = 0;
    w.norms = (S::Real::ZERO, S::Real::ZERO);
    w.x.copy_from_slice(start.x);
    w.z.copy_from_slice(start.z);
    w.u.copy_from_slice(start.scaled_dual);
    w.primal.fill(S::ZERO);
    w.dual.fill(S::ZERO);
    let mut rho = options.rho;
    let mut completed = 0;
    let mut pnorm = S::Real::ZERO;
    let mut dnorm = S::Real::ZERO;
    let mut termination = OptimizeTermination::IterationLimit;
    'iterations: for _ in 0..options.iterations.get() {
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break;
        }
        if !context.record_iteration()?.is_granted() {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::IterationLimit,
                context,
            );
            return Ok(());
        }
        if !context
            .record_provider(EvaluationRole::Function)?
            .is_granted()
        {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        let x_report = x_update.update_into(
            AdmmUpdateInput {
                block: AdmmBlock::X,
                current: &w.x,
                other: &w.z,
                scaled_dual: &w.u,
                rhs: problem.c,
                operator: problem.a,
                other_operator: problem.b,
                rho,
            },
            &mut w.trial_x,
            context,
        )?;
        check_report(
            x_report,
            options.subproblem,
            scioxide_core::argument!("x_update"),
        )?;
        finite(&w.trial_x, NonFiniteOrigin::Callback)?;
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        if !context
            .record_provider(EvaluationRole::Function)?
            .is_granted()
        {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        let z_report = z_update.update_into(
            AdmmUpdateInput {
                block: AdmmBlock::Z,
                current: &w.z,
                other: &w.trial_x,
                scaled_dual: &w.u,
                rhs: problem.c,
                operator: problem.b,
                other_operator: problem.a,
                rho,
            },
            &mut w.trial_z,
            context,
        )?;
        check_report(
            z_report,
            options.subproblem,
            scioxide_core::argument!("z_update"),
        )?;
        finite(&w.trial_z, NonFiniteOrigin::Callback)?;
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        if !apply(
            problem.b,
            &w.z,
            &mut w.previous_bz,
            &mut w.operator,
            context,
        )? {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        if !apply(problem.a, &w.trial_x, &mut w.ax, &mut w.operator, context)? {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        if !apply(problem.b, &w.trial_z, &mut w.bz, &mut w.operator, context)? {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        for ((difference, &bz), &before) in w.difference.iter_mut().zip(&w.bz).zip(&w.previous_bz) {
            *difference = bz - before;
        }
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        problem
            .a
            .apply_adjoint_into_scoped(
                &w.difference,
                &mut w.adjoint,
                &mut w.operator[..problem.a.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite(&w.adjoint, NonFiniteOrigin::Callback)?;
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        let rho_scalar = S::from_real(rho);
        for (dual, &v) in w.trial_dual.iter_mut().zip(&w.adjoint) {
            *dual = rho_scalar * v;
        }
        for (((primal, &ax), &bz), &c) in w
            .trial_primal
            .iter_mut()
            .zip(&w.ax)
            .zip(&w.bz)
            .zip(problem.c)
        {
            *primal = scioxide_core::backend::checked_product_sum([
                (ax, S::ONE),
                (bz, S::ONE),
                (-c, S::ONE),
            ])?;
        }
        finite(&w.trial_primal, NonFiniteOrigin::Callback)?;
        finite(&w.trial_dual, NonFiniteOrigin::Callback)?;
        let trial_pnorm = norm(&w.trial_primal)?;
        let trial_dnorm = norm(&w.trial_dual)?;
        let axnorm = norm(&w.ax)?;
        let bznorm = norm(&w.bz)?;
        let cnorm = norm(problem.c)?;
        let scale = if axnorm > bznorm { axnorm } else { bznorm };
        let scale = if scale > cnorm { scale } else { cnorm };
        let primal_ok = options.residual.is_satisfied(trial_pnorm, scale)?;
        for ((u, &old), &r) in w.trial_u.iter_mut().zip(&w.u).zip(&w.trial_primal) {
            *u = old + r;
        }
        finite(&w.trial_u, NonFiniteOrigin::Callback)?;
        for (difference, &u) in w.difference.iter_mut().zip(&w.trial_u) {
            *difference = rho_scalar * u;
        }
        finite(&w.difference, NonFiniteOrigin::Callback)?;
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        problem
            .a
            .apply_adjoint_into_scoped(
                &w.difference,
                &mut w.adjoint,
                &mut w.operator[..problem.a.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite(&w.adjoint, NonFiniteOrigin::Callback)?;
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        let dual_ok = options
            .residual
            .is_satisfied(trial_dnorm, norm(&w.adjoint)?)?;
        w.x.copy_from_slice(&w.trial_x);
        w.z.copy_from_slice(&w.trial_z);
        w.u.copy_from_slice(&w.trial_u);
        w.primal.copy_from_slice(&w.trial_primal);
        w.dual.copy_from_slice(&w.trial_dual);
        pnorm = trial_pnorm;
        dnorm = trial_dnorm;
        completed += 1;
        (w.completed, w.norms) = (completed, (pnorm, dnorm));
        context.commit_accepted(PartialResultTrust::Admissible)?;
        if primal_ok && dual_ok {
            context.assess_accepted(PartialResultTrust::ResidualChecked)?;
            publish(
                output,
                w,
                (pnorm, dnorm, rho, completed),
                OptimizeTermination::Converged,
                context,
            );
            return Ok(());
        }
        if let Some((ratio, factor)) = options.balancing {
            let next = if pnorm > ratio * dnorm {
                rho * factor
            } else if dnorm > ratio * pnorm {
                rho / factor
            } else {
                rho
            };
            positive(scioxide_core::argument!("self"), next)?;
            if next != rho {
                let adjust = S::from_real(rho / next);
                for u in &mut w.u {
                    *u = adjust * *u;
                }
                finite(&w.u, NonFiniteOrigin::Intermediate)?;
                rho = next;
            }
        }
    }
    publish(
        output,
        w,
        (pnorm, dnorm, rho, completed),
        termination,
        context,
    );
    Ok(())
}

/// Check a block report returned by the entry point's update `argument`.
fn check_report<R: RealScalar>(
    report: AdmmUpdateReport<R>,
    tolerance: GradientTolerance<R>,
    argument: Argument,
) -> Result<(), SciOxideError> {
    if !report.converged
        || !report.stationarity_norm.is_finite()
        || report.stationarity_norm < R::ZERO
        || !tolerance.is_satisfied(report.stationarity_norm, R::ONE)?
    {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "block update did not report required stationarity",
        ));
    }
    Ok(())
}
