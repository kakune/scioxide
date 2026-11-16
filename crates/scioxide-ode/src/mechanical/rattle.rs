//! The RATTLE method for Hamiltonian systems with holonomic constraints.

use alloc::vec::Vec;

use scioxide_core::{
    AttemptRecord, Capacity, EvaluationRole, ExecutionContext, ExecutionPolicy, LinearOperator,
    NonFiniteOrigin, NumericScalar, PartialFailure, PartialResultTrust, RealScalar,
    ResidualTolerance, SciOxideError, SolveContext, SolveScope, Triangle,
};
use scioxide_linalg::{
    LdlDecomposition, LinalgWorkspace, LuDecomposition, Matrix, MatrixViewMut, Vector,
};

use super::{
    MechanicalOptions, MechanicalOutputSink, MechanicalReport, MechanicalSolution,
    MechanicalTermination, PhaseSpace, accept, admit, attempt, identity, partial, poison, proposal,
    validate_state,
};
use crate::safety::{finite, invalid};
use crate::shell::Shell;
use crate::termination::Fault;

/// Defines a real Hamiltonian system with holonomic constraints for [`Rattle`].
///
/// ```text
/// H(q, p) = pᵀ M⁻¹ p / 2 + V(q),   g(q) = 0,
/// q′ = M⁻¹ p,   p′ = −∇V(q) − G(q)ᵀ λ,   G(q) = ∂g/∂q,
/// ```
///
/// where `q` has `n` = [`Self::dimension`] positions, `p` the conjugate momenta, `M`
/// the constant mass matrix, `V` the potential, `g` the `m` = [`Self::constraints`]
/// constraint functions and `λ` the Lagrange multipliers. The exact solution also
/// satisfies the hidden constraint `G(q) M⁻¹ p = 0`. The [`RattleProblem`] supplies
/// `M⁻¹`, which must be symmetric positive definite, and `G(q)` must have full rank
/// `m` near the trajectory.
///
/// The callbacks must write every output entry; unwritten, NaN and infinite entries
/// are a `NonFinite` error with origin `Callback`. Each call counts as one evaluation
/// of the step that makes it; charge only nested work to `cx`.
pub trait HolonomicConstraints<S: RealScalar + NumericScalar<Real = S>> {
    /// Returns the number of positions `n`, which equals the number of momenta.
    fn dimension(&self) -> usize;
    /// Returns the number of constraints `m`, at most `n`.
    fn constraints(&self) -> usize;
    /// Evaluates `∇V(q)`, the negative force, into `output`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the call returns unchanged.
    fn gradient(
        &self,
        position: &[S],
        output: &mut [S],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates the `m` constraint values `g(q)` into `output`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the call returns unchanged.
    fn values(
        &self,
        position: &[S],
        output: &mut [S],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates the `m × n` constraint Jacobian `G(q) = ∂g/∂q` into `output`, one row
    /// per constraint.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the call returns unchanged.
    fn jacobian(
        &self,
        position: &[S],
        output: &mut MatrixViewMut<'_, S>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Pairs a borrowed [`HolonomicConstraints`] system with the inverse mass matrix `M⁻¹`
/// for [`Rattle`].
#[must_use]
pub struct RattleProblem<'a, S: RealScalar + NumericScalar<Real = S>, P: ?Sized> {
    constraints: &'a P,
    inverse_mass: &'a dyn LinearOperator<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>, P: ?Sized> Clone for RattleProblem<'_, S, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, P: ?Sized> Copy for RattleProblem<'_, S, P> {}

impl<'a, S, P> RattleProblem<'a, S, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: HolonomicConstraints<S> + ?Sized,
{
    /// Borrows the system and the operator that applies the constant inverse mass
    /// matrix `M⁻¹`.
    pub const fn new(constraints: &'a P, inverse_mass: &'a dyn LinearOperator<S>) -> Self {
        Self {
            constraints,
            inverse_mass,
        }
    }
    /// Returns the system.
    pub const fn constraints(&self) -> &'a P {
        self.constraints
    }
    /// Returns the inverse mass operator.
    pub const fn inverse_mass(&self) -> &'a dyn LinearOperator<S> {
        self.inverse_mass
    }
}

/// Sets the work limits, the constraint tolerances and the Newton iteration limit of
/// [`Rattle`].
///
/// A step is accepted when `‖g(q)‖∞` and `‖G(q) M⁻¹ p‖∞` are at most their thresholds
/// `atol + rtol · scale`. The thresholds are computed once by [`Rattle::advance`] and are
/// in the units of `g`, so choose each scale in those units. The setters do not check
/// their arguments; [`Rattle::advance`] validates the options.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct RattleOptions<S: RealScalar> {
    integration: MechanicalOptions,
    position: ResidualTolerance<S>,
    position_scale: S,
    momentum: ResidualTolerance<S>,
    momentum_scale: S,
    newton_iterations: usize,
}

impl<S: RealScalar> RattleOptions<S> {
    /// The default limit on Newton iterations of each position projection.
    pub const DEFAULT_NEWTON_ITERATIONS: usize = 10;

    /// Creates options with unlimited integration budgets, the default
    /// [`ResidualTolerance`] at scale one for both constraints, and
    /// [`Self::DEFAULT_NEWTON_ITERATIONS`].
    pub fn new() -> Self {
        let integration = MechanicalOptions::new();
        Self {
            integration,
            position: ResidualTolerance::default(),
            position_scale: S::ONE,
            momentum: ResidualTolerance::default(),
            momentum_scale: S::ONE,
            newton_iterations: Self::DEFAULT_NEWTON_ITERATIONS,
        }
    }
    /// Sets the integration work limits.
    pub fn with_integration(mut self, integration: MechanicalOptions) -> Self {
        self.integration = integration;
        self
    }
    /// Sets the tolerance of `‖g(q)‖∞` and the scale of its relative part.
    pub fn with_position_tolerance(mut self, tolerance: ResidualTolerance<S>, scale: S) -> Self {
        self.position = tolerance;
        self.position_scale = scale;
        self
    }
    /// Sets the tolerance of `‖G(q) M⁻¹ p‖∞` and the scale of its relative part.
    pub fn with_momentum_tolerance(mut self, tolerance: ResidualTolerance<S>, scale: S) -> Self {
        self.momentum = tolerance;
        self.momentum_scale = scale;
        self
    }
    /// Sets the largest number of Newton iterations of each position projection.
    ///
    /// With zero, a step is accepted only if the unprojected position already meets the tolerance.
    pub fn with_newton_iterations(mut self, iterations: usize) -> Self {
        self.newton_iterations = iterations;
        self
    }

    /// Returns the work limits.
    pub const fn integration(&self) -> &MechanicalOptions {
        &self.integration
    }
    /// Returns the position constraint tolerance and its scale.
    pub const fn position_tolerance(&self) -> (ResidualTolerance<S>, S) {
        (self.position, self.position_scale)
    }
    /// Returns the momentum constraint tolerance and its scale.
    pub const fn momentum_tolerance(&self) -> (ResidualTolerance<S>, S) {
        (self.momentum, self.momentum_scale)
    }
    /// Returns the Newton iteration limit of each position projection.
    pub const fn newton_iterations(&self) -> usize {
        self.newton_iterations
    }
}

/// Borrows or stores call-local execution parameters.
#[derive(Debug, Clone)]
#[must_use]
struct RattleRun<S: RealScalar + NumericScalar<Real = S>> {
    step: S,
    steps: usize,
    dimension: usize,
    constraints: usize,
    options: RattleOptions<S>,
    position_threshold: S,
    momentum_threshold: S,
}

/// Describes the workspace storage of a [`Rattle`] method for one problem; returned by
/// [`Rattle::workspace_requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RattleRequirements {
    dimension: usize,
    constraints: usize,
    operator_scratch: usize,
}

impl RattleRequirements {
    /// Returns the number of positions `n`.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of constraints `m`.
    pub const fn constraints(&self) -> usize {
        self.constraints
    }
    /// Returns the scratch length of the inverse mass operator, in scalars.
    pub const fn operator_scratch(&self) -> usize {
        self.operator_scratch
    }
}

/// Holds the step, projection and factorization storage of [`Rattle`] calls and the
/// record of the last attempt.
///
/// Create one with [`Rattle::workspace`], or with [`Self::new`] and
/// [`Self::try_reserve`]. The operator scratch may be longer than a problem needs.
#[derive(Debug)]
pub struct RattleWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    storage: Storage<S>,
    shell: Shell<MechanicalReport<S>>,
}

#[derive(Debug)]
struct Storage<S: RealScalar + NumericScalar<Real = S>> {
    /// Dimension and constraint count of the prepared storage.
    prepared: Option<(usize, usize)>,
    q: Vec<S>,
    p: Vec<S>,
    force: Vec<S>,
    velocity: Vec<S>,
    column: Vec<S>,
    scratch: Vec<S>,
    initial_j: Matrix<S>,
    j: Matrix<S>,
    mass_j: Matrix<S>,
    correction: Matrix<S>,
    residual: Vector<S>,
    delta: Vector<S>,
    lu: LuDecomposition<S>,
    lu_work: LinalgWorkspace<'static, S>,
    ldl: LdlDecomposition<S>,
    ldl_work: LinalgWorkspace<'static, S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for RattleWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> RattleWorkspace<S> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: Storage {
                prepared: None,
                q: Vec::new(),
                p: Vec::new(),
                force: Vec::new(),
                velocity: Vec::new(),
                column: Vec::new(),
                scratch: Vec::new(),
                initial_j: Matrix::empty(),
                j: Matrix::empty(),
                mass_j: Matrix::empty(),
                correction: Matrix::empty(),
                residual: Vector::empty(),
                delta: Vector::empty(),
                lu: LuDecomposition::empty(),
                lu_work: LinalgWorkspace::default(),
                ldl: LdlDecomposition::empty(),
                ldl_work: LinalgWorkspace::default(),
            },
            shell: Shell::new(),
        }
    }

    /// Returns whether a call with `requirements` can run without growing this
    /// workspace.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the other workspaces.
    pub fn check(&self, requirements: &RattleRequirements) -> Result<Capacity, SciOxideError> {
        Ok(if self.storage.fits(requirements) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows this workspace to meet `requirements`, including the storage of both
    /// factorizations.
    ///
    /// After a failed reservation the workspace may keep part of the new storage.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` or `SizeOverflow` if a buffer cannot be allocated.
    pub fn try_reserve(&mut self, requirements: &RattleRequirements) -> Result<(), SciOxideError> {
        self.storage.reserve(requirements)
    }

    /// Returns the record of the last call: its outcome, counts and the report at the
    /// last accepted step.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MechanicalReport<S>>> {
        self.shell.last_attempt()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> RattleRun<S> {
    /// Creates a plan for `dimension` positions and `constraints` constraints.
    ///
    /// The constraint thresholds `atol + rtol · scale` of `options` are computed here.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `constraints > dimension`, if the step is zero or half of it
    ///   underflows to zero, if a tolerance scale is negative, or if a threshold is
    ///   zero.
    /// - `NonFinite` (origin `Input`) if the step or a scale is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a threshold overflows.
    fn new(
        problem: &RattleProblem<'_, S, impl HolonomicConstraints<S> + ?Sized>,
        step: S,
        steps: usize,
        options: RattleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let dimension = problem.constraints.dimension();
        let constraints = problem.constraints.constraints();
        if constraints > dimension {
            return Err(invalid(
                scioxide_core::argument!("constraints"),
                "more constraints than coordinates",
            ));
        }
        super::validate_step(step)?;
        let position_threshold = options.position.threshold(options.position_scale)?;
        let momentum_threshold = options.momentum.threshold(options.momentum_scale)?;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        if position_threshold <= zero || momentum_threshold <= zero {
            return Err(invalid(
                if position_threshold <= zero {
                    scioxide_core::argument!("position")
                } else {
                    scioxide_core::argument!("momentum")
                },
                "projection thresholds must be positive",
            ));
        }
        Ok(Self {
            step,
            steps,
            dimension,
            constraints,
            options,
            position_threshold,
            momentum_threshold,
        })
    }

    /// Returns the workspace storage needed to integrate `problem`, including the
    /// scratch of its inverse mass operator.
    ///
    /// # Errors
    ///
    /// The error of the operator's scratch length query.
    fn workspace_requirements<P: HolonomicConstraints<S> + ?Sized>(
        &self,
        problem: &RattleProblem<'_, S, P>,
    ) -> Result<RattleRequirements, SciOxideError> {
        Ok(RattleRequirements {
            dimension: self.dimension,
            constraints: self.constraints,
            operator_scratch: problem.inverse_mass.apply_workspace_len()?,
        })
    }

    fn preflight<P: HolonomicConstraints<S> + ?Sized>(
        &self,
        problem: &RattleProblem<'_, S, P>,
        state: &PhaseSpace<S>,
        workspace: &mut RattleWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let n = self.dimension;
        validate_state(n, state.position(), state.momentum())?;
        for (expected, actual) in [
            (n, problem.constraints.dimension()),
            (self.constraints, problem.constraints.constraints()),
        ] {
            if expected != actual {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(expected),
                    scioxide_core::Dimensions::one(actual),
                ));
            }
        }
        let mass = problem.inverse_mass;
        if mass.rows() != n || mass.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, n),
                scioxide_core::Dimensions::two(mass.rows(), mass.cols()),
            ));
        }
        let requirements = self.workspace_requirements(problem)?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            workspace.try_reserve(&requirements)?;
        }
        Ok(())
    }

    /// Takes the configured number of steps from `state` and passes each accepted state
    /// to `sink`.
    ///
    /// The call continues from the time and values in `state`, and runs in `scope`, an
    /// [`ExecutionContext`] or the `&mut SolveContext` of an enclosing solve (see
    /// [Solve accounting](scioxide_core#solve-accounting)). It first checks that `state`
    /// meets both constraint tolerances, also when the step count is zero. A workspace
    /// that is too small is grown first. A work limit, cancellation, a deadline or
    /// [`MechanicalTermination::ProjectionLimit`] ends the call with an unsuccessful
    /// report. On return, also after an error, `state` holds the last accepted step.
    /// With a large enough workspace and a sink, operator and callbacks that do not
    /// allocate, the call does not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `state`, `problem` or the inverse mass operator does not
    ///   match the plan's dimensions, and `NonFinite` (origin `Input`) if `state` has a
    ///   NaN or infinite entry. No callback runs and `state` is unchanged.
    /// - `InvalidInput` if the initial state does not meet a constraint tolerance, or
    ///   if `G M⁻¹ Gᵀ` is not positive definite.
    /// - `Singular` if `G M⁻¹ Gᵀ` or the Newton matrix is singular.
    /// - `NonFinite` (origin `Callback`) if a callback or the operator returns a NaN,
    ///   infinite or unwritten entry, and (origin `Intermediate`) if a position,
    ///   momentum or matrix entry overflows.
    /// - `NumericalInstability` (`Overflow`) if a kick or drift result is outside
    ///   the finite scalar range.
    /// - Errors of `problem`, the operator and `sink`, unchanged, and
    ///   `AllocationFailure` if the workspace cannot grow.
    fn advance<'scope, 'capability, P, K>(
        &self,
        problem: &RattleProblem<'_, S, P>,
        state: &mut PhaseSpace<S>,
        sink: &mut K,
        workspace: &mut RattleWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<MechanicalReport<S>, SciOxideError>
    where
        P: HolonomicConstraints<S> + ?Sized,
        K: MechanicalOutputSink<S> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, state, workspace) {
            return workspace.shell.reject(error);
        }
        let RattleWorkspace { storage, shell } = workspace;
        let initial_time = state.time();
        let (position, momentum, time) = state.parts_mut();
        attempt(
            shell,
            scope,
            identity(
                scioxide_core::algorithm_id!("ode-mechanical", "rattle", 1),
                self.dimension,
            ),
            self.options.integration.budgets(),
            initial_time,
            |report, cx| storage.steps(self, problem, position, momentum, time, sink, report, cx),
        )
    }
}

fn apply<S: RealScalar + NumericScalar<Real = S>>(
    mass: &dyn LinearOperator<S>,
    input: &[S],
    output: &mut [S],
    scratch: &mut [S],
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), Fault> {
    admit(cx, EvaluationRole::LinearOperator)?;
    poison(output);
    mass.apply_into(input, output, scratch)?;
    Ok(finite(output, NonFiniteOrigin::Callback)?)
}
fn values<S: RealScalar + NumericScalar<Real = S>, P: HolonomicConstraints<S> + ?Sized>(
    problem: &P,
    q: &[S],
    out: &mut [S],
    cx: &mut SolveContext<'_, '_>,
) -> Result<S, Fault> {
    admit(cx, EvaluationRole::Constraint)?;
    poison(out);
    problem.values(q, out, cx)?;
    finite(out, NonFiniteOrigin::Callback)?;
    Ok(crate::safety::max_abs(out.iter().copied()))
}
fn jacobian<S: RealScalar + NumericScalar<Real = S>, P: HolonomicConstraints<S> + ?Sized>(
    problem: &P,
    q: &[S],
    out: &mut Matrix<S>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), Fault> {
    admit(cx, EvaluationRole::ConstraintJacobian)?;
    poison(out.as_mut_slice());
    problem.jacobian(q, &mut out.view_mut(), cx)?;
    Ok(finite(out.as_slice(), NonFiniteOrigin::Callback)?)
}
fn gradient<S: RealScalar + NumericScalar<Real = S>, P: HolonomicConstraints<S> + ?Sized>(
    problem: &P,
    q: &[S],
    out: &mut [S],
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), Fault> {
    admit(cx, EvaluationRole::Gradient)?;
    poison(out);
    problem.gradient(q, out, cx)?;
    Ok(finite(out, NonFiniteOrigin::Callback)?)
}

impl<S: RealScalar + NumericScalar<Real = S>> Storage<S> {
    fn fits(&self, requirements: &RattleRequirements) -> bool {
        self.prepared == Some((requirements.dimension, requirements.constraints))
            && self.scratch.len() >= requirements.operator_scratch
    }

    fn reserve(&mut self, requirements: &RattleRequirements) -> Result<(), SciOxideError> {
        let RattleRequirements {
            dimension,
            constraints,
            operator_scratch,
        } = *requirements;
        if self.prepared != Some((dimension, constraints)) {
            self.prepared = None;
            self.prepare(dimension, constraints)?;
            self.prepared = Some((dimension, constraints));
        }
        if self.scratch.len() < operator_scratch {
            self.scratch = scioxide_core::backend::try_filled(
                operator_scratch,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("operator_scratch"),
            )?;
        }
        Ok(())
    }

    /// Allocates every step and projection buffer and the storage of both
    /// factorizations.
    fn prepare(&mut self, dimension: usize, constraints: usize) -> Result<(), SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let sequential = scioxide_linalg::LinalgExecutionOptions::new()
            .with_execution_policy(ExecutionPolicy::Sequential);
        let mut correction = Matrix::identity(constraints)?;
        LuDecomposition::factor_into(
            &correction,
            &sequential,
            &mut self.lu,
            &mut self.lu_work,
            ExecutionContext::sequential(),
        )?;
        self.lu_work
            .try_reserve(&LuDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::solve(
                    constraints,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?)?;
        // LU swaps its candidate and published factor. Warm both sets of storage.
        LuDecomposition::factor_into(
            &correction,
            &sequential,
            &mut self.lu,
            &mut self.lu_work,
            ExecutionContext::sequential(),
        )?;
        correction.as_mut_slice().fill(zero);
        self.ldl = LdlDecomposition::factor(
            scioxide_linalg::MatrixView::from_slice_with_strides(
                constraints,
                constraints,
                &[zero],
                0,
                0,
            )?,
            scioxide_linalg::HermitianInput::Exact,
        )?;
        self.ldl_work
            .try_reserve(&LdlDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    constraints,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?)?;
        self.ldl_work
            .try_reserve(&LdlDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                    constraints,
                    1,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?)?;
        for (buffer, quantity) in [
            (&mut self.q, scioxide_core::quantity!("position_scratch")),
            (&mut self.p, scioxide_core::quantity!("momentum_scratch")),
            (&mut self.force, scioxide_core::quantity!("force_scratch")),
            (
                &mut self.velocity,
                scioxide_core::quantity!("velocity_scratch"),
            ),
            (&mut self.column, scioxide_core::quantity!("column_scratch")),
        ] {
            *buffer = scioxide_core::backend::try_filled(dimension, zero, quantity)?;
        }
        self.initial_j = Matrix::zeros(constraints, dimension)?;
        self.j = Matrix::zeros(constraints, dimension)?;
        self.mass_j = Matrix::zeros(constraints, dimension)?;
        self.correction = correction;
        self.residual = Vector::zeros(constraints)?;
        self.delta = Vector::zeros(constraints)?;
        Ok(())
    }

    // Store M^-1 J^T as rows so each operator output is contiguous.
    fn mass_rows(
        &mut self,
        mass: &dyn LinearOperator<S>,
        initial: bool,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        let scratch_len = mass.apply_workspace_len()?;
        if scratch_len > self.scratch.len() {
            return Err(invalid(
                scioxide_core::argument!("inverse_mass"),
                "operator scratch requirement changed",
            )
            .into());
        }
        let j = if initial { &self.initial_j } else { &self.j };
        let n = self.q.len();
        for row in 0..j.rows() {
            apply(
                mass,
                &j.as_slice()[row * n..(row + 1) * n],
                &mut self.mass_j.as_mut_slice()[row * n..(row + 1) * n],
                &mut self.scratch[..scratch_len],
                cx,
            )?;
        }
        Ok(())
    }

    fn correction_matrix(&mut self) -> Result<(), SciOxideError> {
        let n = self.q.len();
        let m = self.j.rows();
        for i in 0..m {
            for k in 0..m {
                let value = (0..n).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, col| {
                    sum + self.j.as_slice()[i * n + col] * self.mass_j.as_slice()[k * n + col]
                });
                self.correction.as_mut_slice()[i * m + k] = value;
            }
        }
        finite(self.correction.as_slice(), NonFiniteOrigin::Intermediate)
    }

    fn tangent_residual(&mut self) -> Result<S, SciOxideError> {
        let n = self.q.len();
        for (row, out) in self.residual.as_mut_slice().iter_mut().enumerate() {
            *out = (0..n).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, col| {
                sum + self.j.as_slice()[row * n + col] * self.velocity[col]
            });
        }
        finite(self.residual.as_slice(), NonFiniteOrigin::Intermediate)?;
        Ok(crate::safety::max_abs(
            self.residual.as_slice().iter().copied(),
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn steps<P, K>(
        &mut self,
        plan: &RattleRun<S>,
        problem: &RattleProblem<'_, S, P>,
        position: &mut [S],
        momentum: &mut [S],
        time: &mut S,
        sink: &mut K,
        report: &mut MechanicalReport<S>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault>
    where
        P: HolonomicConstraints<S> + ?Sized,
        K: MechanicalOutputSink<S> + ?Sized,
    {
        let constraints = problem.constraints;
        let mass = problem.inverse_mass;
        let n = self.q.len();
        let scratch = mass.apply_workspace_len()?;
        if scratch > self.scratch.len() {
            return Err(invalid(
                scioxide_core::argument!("inverse_mass"),
                "operator scratch requirement changed",
            )
            .into());
        }
        let h = plan.step;
        let half = h
            / (<S as scioxide_core::AlgebraicScalar>::ONE
                + <S as scioxide_core::AlgebraicScalar>::ONE);
        // Even zero-step requests validate initial feasibility; no silent projection.
        let initial_position = values(constraints, position, self.residual.as_mut_slice(), cx)?;
        jacobian(constraints, position, &mut self.j, cx)?;
        apply(
            mass,
            momentum,
            &mut self.velocity,
            &mut self.scratch[..scratch],
            cx,
        )?;
        let initial_momentum = self.tangent_residual()?;
        if initial_position > plan.position_threshold || initial_momentum > plan.momentum_threshold
        {
            return Err(invalid(
                if initial_position > plan.position_threshold {
                    scioxide_core::argument!("position")
                } else {
                    scioxide_core::argument!("momentum")
                },
                "initial position and momentum must satisfy constraints",
            )
            .into());
        }
        report.position_residual = Some(initial_position);
        report.momentum_residual = Some(initial_momentum);
        for step in 0..plan.steps {
            proposal(cx)?;
            let Some(next_time) = super::next_time(*time, plan.step) else {
                report.termination = Some(MechanicalTermination::StepSizeLimit);
                return Ok(());
            };
            self.initial_j
                .as_mut_slice()
                .copy_from_slice(self.j.as_slice());
            if step == 0 {
                gradient(constraints, position, &mut self.force, cx)?;
            }
            self.p
                .iter_mut()
                .zip(momentum.iter().zip(&self.force))
                .try_for_each(|(out, (&p, &f))| {
                    *out = crate::safety::state_increment(p, -half, f)?;
                    Ok::<_, SciOxideError>(())
                })?;
            finite(&self.p, NonFiniteOrigin::Intermediate)?;
            apply(
                mass,
                &self.p,
                &mut self.velocity,
                &mut self.scratch[..scratch],
                cx,
            )?;
            self.q
                .iter_mut()
                .zip(position.iter().zip(&self.velocity))
                .try_for_each(|(out, (&q, &v))| {
                    *out = crate::safety::state_increment(q, h, v)?;
                    Ok::<_, SciOxideError>(())
                })?;
            finite(&self.q, NonFiniteOrigin::Intermediate)?;
            self.mass_rows(mass, true, cx)?;
            let mut position_residual =
                values(constraints, &self.q, self.residual.as_mut_slice(), cx)?;
            for _ in 0..plan.options.newton_iterations {
                if position_residual <= plan.position_threshold {
                    break;
                }
                crate::termination::admit(cx, crate::termination::Work::Iteration)?;
                jacobian(constraints, &self.q, &mut self.j, cx)?;
                self.correction_matrix()?;
                admit(cx, EvaluationRole::Factorization)?;
                LuDecomposition::factor_into(
                    &self.correction,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut self.lu,
                    &mut self.lu_work,
                    cx.execution(),
                )?;
                admit(cx, EvaluationRole::LinearSolve)?;
                self.lu
                    .solve_into(
                        &self.residual,
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        self.delta.view_mut(),
                        &mut self.lu_work,
                        cx.execution(),
                    )
                    .and_then(|report| report.termination().require_accepted())?;
                let m = self.j.rows();
                for col in 0..n {
                    let dq =
                        (0..m).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, row| {
                            sum + self.mass_j.as_slice()[row * n + col] * self.delta.as_slice()[row]
                        });
                    let dp =
                        (0..m).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, row| {
                            sum + self.initial_j.as_slice()[row * n + col]
                                * self.delta.as_slice()[row]
                        });
                    self.q[col] -= dq;
                    self.p[col] -= dp / h;
                }
                finite(&self.q, NonFiniteOrigin::Intermediate)?;
                finite(&self.p, NonFiniteOrigin::Intermediate)?;
                position_residual = values(constraints, &self.q, self.residual.as_mut_slice(), cx)?;
            }
            if position_residual > plan.position_threshold {
                cx.record_rejected()?;
                report.termination = Some(MechanicalTermination::ProjectionLimit);
                return Ok(());
            }
            gradient(constraints, &self.q, &mut self.force, cx)?;
            self.p.iter_mut().zip(&self.force).try_for_each(|(p, &f)| {
                *p = crate::safety::state_increment(*p, -half, f)?;
                Ok::<_, SciOxideError>(())
            })?;
            finite(&self.p, NonFiniteOrigin::Intermediate)?;
            jacobian(constraints, &self.q, &mut self.j, cx)?;
            self.mass_rows(mass, false, cx)?;
            self.correction_matrix()?;
            // This Gram matrix is symmetric mathematically. Select its lower
            // triangle explicitly; no nonsymmetric Newton matrix enters LDL.
            admit(cx, EvaluationRole::Factorization)?;
            LdlDecomposition::factor_into(
                self.correction.view(),
                scioxide_linalg::HermitianInput::Triangle(Triangle::Lower),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut self.ldl,
                &mut self.ldl_work,
                cx.execution(),
            )?;
            let inertia = self.ldl.inertia(scioxide_linalg::RankPolicy::Exact)?;
            if inertia.zero() != 0 {
                return Err(SciOxideError::singular(None).into());
            }
            if inertia.positive() != self.j.rows() {
                return Err(invalid(
                    scioxide_core::argument!("problem"),
                    "constraint Gram matrix must be positive definite",
                )
                .into());
            }
            apply(
                mass,
                &self.p,
                &mut self.velocity,
                &mut self.scratch[..scratch],
                cx,
            )?;
            self.tangent_residual()?;
            admit(cx, EvaluationRole::LinearSolve)?;
            self.ldl
                .solve_into(
                    self.residual.view(),
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    self.delta.view_mut(),
                    &mut self.ldl_work,
                    cx.execution(),
                )
                .and_then(|report| report.termination().require_accepted())?;
            for col in 0..n {
                self.column[col] = (0..self.j.rows())
                    .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, row| {
                        sum + self.j.as_slice()[row * n + col] * self.delta.as_slice()[row]
                    });
            }
            self.p
                .iter_mut()
                .zip(&self.column)
                .for_each(|(p, &d)| *p -= d);
            finite(&self.p, NonFiniteOrigin::Intermediate)?;
            apply(
                mass,
                &self.p,
                &mut self.velocity,
                &mut self.scratch[..scratch],
                cx,
            )?;
            let momentum_residual = self.tangent_residual()?;
            if momentum_residual > plan.momentum_threshold {
                cx.record_rejected()?;
                report.termination = Some(MechanicalTermination::ProjectionLimit);
                return Ok(());
            }
            accept(cx, PartialResultTrust::ResidualChecked)?;
            *time = next_time;
            report.time = next_time;
            report.trust = PartialResultTrust::ResidualChecked;
            position.copy_from_slice(&self.q);
            momentum.copy_from_slice(&self.p);
            report.steps += 1;
            report.position_residual = Some(position_residual);
            report.momentum_residual = Some(momentum_residual);
            sink.emit(report.steps, position, momentum)?;
        }
        Ok(())
    }
}

impl<S: RealScalar> Default for RattleOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Integrates a [`HolonomicConstraints`] system with the RATTLE method at a fixed step.
///
/// One step of size `h` from `(qₙ, pₙ)` is
///
/// ```text
/// p½ = pₙ − (h/2) ∇V(qₙ) − G(qₙ)ᵀ λ,
/// qₙ₊₁ = qₙ + h M⁻¹ p½,                    g(qₙ₊₁) = 0,
/// pₙ₊₁ = p½ − (h/2) ∇V(qₙ₊₁) − G(qₙ₊₁)ᵀ μ,   G(qₙ₊₁) M⁻¹ pₙ₊₁ = 0,
/// ```
///
/// with multipliers `λ` and `μ`. The position constraint is solved for `λ` by Newton's
/// method, at most [`RattleOptions::newton_iterations`] iterations, each with an LU
/// factorization of `G(q̃) M⁻¹ G(qₙ)ᵀ` at the current iterate `q̃`. The momentum
/// constraint is linear
/// in `μ` and is solved with an `LDLᵀ` factorization of `G(qₙ₊₁) M⁻¹ G(qₙ₊₁)ᵀ`. The
/// method is of order 2, symplectic and time-reversible on the constraint manifold.
///
/// Each step applies `M⁻¹` `2m + 3` times and evaluates `∇V`, `g` and `G` once each,
/// plus `g` and `G` once per Newton iteration. Each call also evaluates `g`, `G` and
/// `M⁻¹ p` at the initial state, and `∇V(q₀)`. The linear algebra costs
/// `O(m² n + m³)` per Newton iteration.
/// The workspace stores `O(m n + m²)` values, and a reserved workspace is ready for the
/// first step without further allocation.
///
/// A call first checks that the initial state meets both tolerances; it does not
/// project an infeasible state. A step that misses a tolerance ends the call with
/// [`MechanicalTermination::ProjectionLimit`].
///
/// # References
///
/// - H. C. Andersen, "Rattle: A 'velocity' version of the Shake algorithm for molecular
///   dynamics calculations", *J. Comput. Phys.* 52(1), 24–34, 1983.
/// - E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical Integration*, 2nd ed.,
///   Springer, 2006, §VII.1.
/// - B. Leimkuhler and S. Reich, *Simulating Hamiltonian Dynamics*, Cambridge
///   University Press, 2004.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rattle;
impl Rattle {
    /// Computes reusable storage requirements from the problem dimensions.
    /// # Errors
    /// Returns operator scratch-query errors and storage size overflow.
    pub fn workspace_requirements<
        S: RealScalar + NumericScalar<Real = S>,
        P: HolonomicConstraints<S> + ?Sized,
    >(
        &self,
        problem: &RattleProblem<'_, S, P>,
    ) -> Result<RattleRequirements, SciOxideError> {
        Ok(RattleRequirements {
            dimension: problem.constraints.dimension(),
            constraints: problem.constraints.constraints(),
            operator_scratch: problem.inverse_mass.apply_workspace_len()?,
        })
    }
    /// Allocates reusable storage for the problem.
    /// # Errors
    /// Returns requirement, allocation and factorization setup errors.
    pub fn workspace<
        S: RealScalar + NumericScalar<Real = S>,
        P: HolonomicConstraints<S> + ?Sized,
    >(
        &self,
        problem: &RattleProblem<'_, S, P>,
    ) -> Result<RattleWorkspace<S>, SciOxideError> {
        let mut ws = RattleWorkspace::new();
        ws.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(ws)
    }
    /// Takes `steps` signed steps of size `step`, retaining each accepted state.
    /// Prepared execution allocates nothing when callbacks and sinks do not allocate.
    /// # Errors
    /// Rejects non-finite, zero or unrepresentable half steps, invalid shapes and
    /// options before changing state; returns callback and numerical errors unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn advance<
        'scope,
        'capability,
        S: RealScalar + NumericScalar<Real = S>,
        P: HolonomicConstraints<S> + ?Sized,
        K: MechanicalOutputSink<S> + ?Sized,
    >(
        &self,
        problem: &RattleProblem<'_, S, P>,
        step: S,
        steps: usize,
        options: &RattleOptions<S>,
        state: &mut PhaseSpace<S>,
        sink: &mut K,
        workspace: &mut RattleWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<MechanicalReport<S>, SciOxideError> {
        let run = match RattleRun::new(problem, step, steps, *options) {
            Ok(run) => run,
            Err(error) => return workspace.shell.reject(error),
        };
        run.advance(problem, state, sink, workspace, scope)
    }
    /// Allocates storage and returns the final state and termination report.
    /// # Errors
    /// Returns the errors of [`Self::advance`] with accepted progress.
    #[allow(
        clippy::result_large_err,
        reason = "owned evolution retains accepted progress"
    )]
    pub fn solve<S: RealScalar + NumericScalar<Real = S>, P: HolonomicConstraints<S> + ?Sized>(
        &self,
        problem: &RattleProblem<'_, S, P>,
        step: S,
        steps: usize,
        options: &RattleOptions<S>,
        initial: PhaseSpace<S>,
    ) -> Result<MechanicalSolution<S>, PartialFailure<MechanicalReport<S>>> {
        let mut ws = self.workspace(problem).map_err(PartialFailure::new)?;
        let mut state = initial;
        match self.advance(
            problem,
            step,
            steps,
            options,
            &mut state,
            &mut (),
            &mut ws,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(MechanicalSolution { state, report }),
            Err(error) => Err(partial(error, ws.shell)),
        }
    }
}
