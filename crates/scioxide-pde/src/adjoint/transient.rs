//! Transient adjoints of the method-of-lines adapter over the crate's own plans.
use super::*;
use scioxide_core::{PartialFailure, SolveScope};
use scioxide_ode::{
    AdjointMode, AdjointOptions, AdjointReport, AdjointResult, DaeAdjointOptions,
    DaeAdjointProblem, DaeAdjointRequirements, DaeAdjointSolver, DaeAdjointSystem,
    DaeAdjointWorkspace, DaeInitialization, DaeSystem, DormandPrince5, OdeAdjointOptions,
    OdeAdjointProblem, OdeAdjointRequirements, OdeAdjointResidual, OdeAdjointSolver,
    OdeAdjointSystem, OdeAdjointWorkspace, OdeMethod, OdeStageBackend, OdeSystem,
};

use crate::mol::{CflPolicy, MethodOfLinesOrder, MethodOfLinesPlan, MethodOfLinesWorkspace};
use crate::{DifferentiableDiscretization, ParabolicProblem, ParabolicScheme, PdeParameterization};

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
enum TimePlan<R: RealScalar + NumericScalar<Real = R>> {
    Explicit(OdeAdjointSolver<R>, OdeAdjointRequirements<R>),
    Residual(DaeAdjointSolver<R>, DaeAdjointRequirements<R>),
}

#[allow(clippy::large_enum_variant)]
enum TimeWorkspace<R: RealScalar + NumericScalar<Real = R>> {
    Explicit(OdeAdjointWorkspace<R>),
    Residual(DaeAdjointWorkspace<R>),
}

/// Discrete gradients of a parabolic [`ParabolicProblem`] with respect to the real
/// parameters of its plan's [`PdeParameterization`].
///
/// The plan differentiates exactly the fixed-step time discretization that
/// [`crate::ParabolicProblem`] integrates for the same [`ParabolicScheme`], on the
/// method-of-lines system `M u_f' = b_f(t, p) − (A(p) u)_f` over the free unknowns
/// (state coordinates are [`crate::PdeState::coordinates`]). Time traversal, checkpoints
/// and reverse sweeps are `scioxide-ode`'s:
///
/// | Scheme | `scioxide-ode` plan |
/// |---|---|
/// | [`ParabolicScheme::BackwardEuler`] | [`DaeAdjointSolver::backward_euler`] on [`OdeAdjointResidual`] |
/// | [`ParabolicScheme::CrankNicolson`] | [`DaeAdjointSolver::trapezoid`] on [`OdeAdjointResidual`] |
/// | [`ParabolicScheme::ExplicitDopri5`] | [`OdeAdjointSolver`] with the adapter's step limit |
///
/// Implicit schemes require [`AdjointOptions::with_fixed_step`] and
/// [`DaeInitialization::Project`]; their initial rate
/// is projected from the semidiscrete equations ([`DaeInitialization::Project`]),
/// so callers supply only the initial state. Only the discrete mode is accepted.
/// The interval is `[0, end]` of the problem. A Rannacher startup is not
/// differentiated. `Send + Sync + Clone`.
///
/// [`AdjointOptions::with_max_workspace_bytes`] bounds the whole plan: the
/// method-of-lines storage is reserved first, and the `scioxide-ode` plan, including a
/// [`CheckpointPolicy::MemoryBudget`](scioxide_ode::CheckpointPolicy::MemoryBudget)
/// schedule, fits the remainder.
///
/// ```
/// use scioxide_core::{SciOxideError, SolveContext};
/// use scioxide_ode::{AdjointMode, AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue,
///     AdjointOptions, DaeAdjointOptions, DaeNumericalOptions, DaeInitialization};
/// use scioxide_pde::{
///     BoundaryCondition, BoundaryData, DifferentiableDiscretization, FieldData,
///     PdeParameterTangent, ParabolicProblem, PdeTransientAdjointPlan, SpatialLimits,
///     StructuredBoundaries, StructuredFdmCoefficients, StructuredFdmPlan,
/// };
/// use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};
/// struct Unit;
/// impl StructuredFdmCoefficients<f64, f64, 1> for Unit {
///     fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
///         Ok(1.0)
///     }
/// }
/// /// J(u(T)) = Σ u_i(T).
/// struct Total;
/// impl AdjointObjective<f64> for Total {
///     fn evaluate(&self, part: AdjointObjectivePart, _: f64, u: &[f64], _: &[f64],
///         gu: &mut [f64], gp: &mut [f64], _: &mut [f64], _: &mut SolveContext<'_, '_>)
///         -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
///         let terminal = part == AdjointObjectivePart::Terminal;
///         gu.fill(if terminal { 1.0 } else { 0.0 });
///         gp.fill(0.0);
///         let value = if terminal { u.iter().sum() } else { 0.0 };
///         Ok(AdjointObjectiveValue { value, time_derivative: 0.0 })
///     }
/// }
/// let mesh = SegmentMesh::new(Grid1d::new(
///     [CoordinateAxis::uniform(0.0, 1.0, 9)?], [AxisTopology::Bounded])?)?;
/// let side = |s| BoundarySet::axis_side(&mesh, 0, s);
/// let boundaries = StructuredBoundaries::new(&mesh, 1, vec![
///     BoundaryCondition::dirichlet(side(AxisSide::Start)?, BoundaryData::zero()),
///     BoundaryCondition::dirichlet(side(AxisSide::End)?, BoundaryData::zero()),
/// ])?;
/// let plan = StructuredFdmPlan::with_options(mesh, &Unit, &boundaries,
///     SpatialLimits::new(16, 256, 1 << 20))?;
/// // One parameter: the source amplitude s in f = s.
/// let source = FieldData::constant(1.0);
/// let problem = ParabolicProblem::new(&plan, &source, 0.25)?;
/// let parameterization = plan.parameterize(&[1.0], &[PdeParameterTangent::new()
///     .with_source(FieldData::constant(1.0))])?;
/// let options = AdjointOptions::new(AdjointMode::Discrete, 0.0625)?.with_fixed_step(true);
/// let adjoint = PdeTransientAdjointPlan::crank_nicolson(&problem, &parameterization, &Total,
///     None, DaeAdjointOptions::new(options, DaeNumericalOptions::new()
///         .with_initialization(DaeInitialization::Project))?)?;
/// let initial = vec![0.0; problem.system().free_indices().len()];
/// let result = adjoint.solve(&problem, &parameterization, &Total, &initial)
///     .map_err(SciOxideError::from)?;
/// // Starting from rest, u(T) is linear in s, so dJ/ds = J at s = 1.
/// let objective = result.report().objective().unwrap();
/// assert!((result.parameter_gradient()[0] - objective).abs() < 1e-9 * objective);
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct PdeTransientAdjointPlan<R: RealScalar + NumericScalar<Real = R>> {
    preparation: alloc::sync::Arc<()>,
    scheme: ParabolicScheme,
    mol: MethodOfLinesPlan<R>,
    time: TimePlan<R>,
    token: MeshToken,
    unknowns: usize,
    parameters: usize,
    end: R,
    bytes: usize,
}

/// Adapter, checkpoint and reverse storage of one prepared transient adjoint.
pub struct PdeTransientAdjointWorkspace<R: RealScalar + NumericScalar<Real = R>> {
    mol: MethodOfLinesWorkspace<R>,
    rate: Vec<R>,
    time: TimeWorkspace<R>,
}

impl<R: RealScalar + NumericScalar<Real = R>> PdeTransientAdjointWorkspace<R> {
    /// Record of the most recent reverse attempt; its progress is the ODE-owned
    /// report whose termination classifies a failure. Absent before the first call.
    pub const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<AdjointReport<R>>> {
        match &self.time {
            TimeWorkspace::Explicit(w) => w.last_attempt(),
            TimeWorkspace::Residual(w) => w.last_attempt(),
        }
    }
}

/// Reject continuous adjoints: PDE plans differentiate discrete equations.
fn discrete<S: RealScalar>(
    options: &AdjointOptions<S>,
    implicit: bool,
) -> Result<(), SciOxideError> {
    if options.mode() != AdjointMode::Discrete {
        return Err(invalid(
            scioxide_core::argument!("options.mode"),
            "PDE plans differentiate discrete equations",
        ));
    }
    if implicit && !options.fixed_step() {
        return Err(invalid(
            scioxide_core::argument!("options.fixed_step"),
            "implicit parabolic schemes advance with a fixed step",
        ));
    }
    Ok(())
}

const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("adjoint_workspace_bytes");

/// Bytes of the method-of-lines adapter workspace and state that the plan owns beside
/// the time-integration workspace; the latter plans against the remaining limit.
fn owned_bytes<R: RealScalar + NumericScalar<Real = R>>(
    mol: &MethodOfLinesPlan<R>,
    parameters: usize,
    max_bytes: usize,
) -> Result<usize, SciOxideError> {
    let owned = checked_add(
        mol.workspace_bytes::<R>(parameters)?,
        checked_mul(mol.dimension(), core::mem::size_of::<R>(), BYTES)?,
        BYTES,
    )?;
    if owned > max_bytes {
        return Err(SciOxideError::resource_limit(BYTES, owned, max_bytes));
    }
    Ok(owned)
}

impl<R: RealScalar + NumericScalar<Real = R>> PdeTransientAdjointPlan<R> {
    /// Prepare the backward-Euler adjoint. `backend` defaults to sparse LU over the
    /// semidiscrete Jacobian pattern, as in [`crate::ParabolicProblem`].
    /// # Errors
    /// Rejects continuous mode, adaptive steps, a parameterization of another plan,
    /// and the option, checkpoint and resource errors of
    /// [`DaeAdjointSolver::backward_euler`].
    pub fn backward_euler<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        backend: Option<OdeStageBackend>,
        options: DaeAdjointOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::implicit(
            ParabolicScheme::BackwardEuler,
            problem,
            parameterization,
            objective,
            backend,
            options,
        )
    }

    /// Prepare the Crank–Nicolson (implicit trapezoid) adjoint; arguments and
    /// defaults as [`Self::backward_euler`].
    /// # Errors
    /// As [`Self::backward_euler`], with [`DaeAdjointSolver::trapezoid`].
    pub fn crank_nicolson<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        backend: Option<OdeStageBackend>,
        options: DaeAdjointOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::implicit(
            ParabolicScheme::CrankNicolson,
            problem,
            parameterization,
            objective,
            backend,
            options,
        )
    }

    /// Prepare the seven-stage discrete DP5 adjoint with the adapter's explicit
    /// step limit `cfl` (use a fixed maximum step for reproducible step sequences).
    /// # Errors
    /// Rejects continuous mode, a parameterization of another plan, invalid step
    /// limits and the errors of [`OdeAdjointSolver::new`].
    pub fn dopri5<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        cfl: CflPolicy<R>,
        options: impl Into<OdeAdjointOptions<R>>,
    ) -> Result<Self, SciOxideError> {
        let options: OdeAdjointOptions<R> = options.into();
        discrete(options.common(), false)?;
        let semi = problem.system();
        let adjoint = semi.adjoint(parameterization)?;
        let mol = semi
            .method_of_lines(MethodOfLinesOrder::First)?
            .with_step_limit(cfl, DormandPrince5::<R>::new().linear_stability())?;
        let max_bytes = options.common().max_workspace_bytes();
        let owned = owned_bytes(&mol, parameterization.len(), max_bytes)?;
        let options = options.with_max_workspace_bytes(max_bytes - owned);
        let mut workspace = mol.workspace::<R>()?;
        let (plan, requirements) = {
            let bound = mol.bind_adjoint(&adjoint, &mut workspace)?;
            let plan =
                OdeAdjointSolver::new(bound.dimension(), bound.parameter_dimension(), options)?;
            let requirements = plan.workspace_requirements(&OdeAdjointProblem::new(
                &bound,
                objective,
                &[],
                (R::ZERO, problem.end()),
            ))?;
            (plan, requirements)
        };
        let bytes = requirements.required_bytes();
        Self::assemble(
            ParabolicScheme::ExplicitDopri5,
            problem,
            parameterization,
            mol,
            TimePlan::Explicit(plan, requirements),
            checked_add(bytes, owned, BYTES)?,
            max_bytes,
        )
    }

    fn implicit<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        scheme: ParabolicScheme,
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        backend: Option<OdeStageBackend>,
        options: DaeAdjointOptions<R>,
    ) -> Result<Self, SciOxideError> {
        discrete(options.common(), true)?;
        let semi = problem.system();
        let adjoint = semi.adjoint(parameterization)?;
        let backend = match backend {
            Some(backend) => backend,
            None => crate::semidiscrete::sparse_backend(semi.jacobian_pattern())?,
        };
        if options.numerics().initialization() != DaeInitialization::Project {
            return Err(invalid(
                scioxide_core::argument!("options.initialization"),
                "implicit PDE adjoints require projection of the initial rate",
            ));
        }
        let mol = semi.method_of_lines(MethodOfLinesOrder::First)?;
        let max_bytes = options.common().max_workspace_bytes();
        let owned = owned_bytes(&mol, parameterization.len(), max_bytes)?;
        let options = options.with_max_workspace_bytes(max_bytes - owned);
        let mut workspace = mol.workspace::<R>()?;
        let (plan, requirements) = {
            let bound = mol.bind_adjoint(&adjoint, &mut workspace)?;
            let residual = OdeAdjointResidual::new(&bound)?;
            let plan = match scheme {
                ParabolicScheme::CrankNicolson => DaeAdjointSolver::trapezoid(
                    residual.dimension(),
                    residual.parameter_dimension(),
                    backend,
                    options,
                )?,
                _ => DaeAdjointSolver::backward_euler(
                    residual.dimension(),
                    residual.parameter_dimension(),
                    backend,
                    options,
                )?,
            };
            let requirements = plan.workspace_requirements(&DaeAdjointProblem::new(
                &residual,
                objective,
                &[],
                &[],
                (R::ZERO, problem.end()),
            ))?;
            (plan, requirements)
        };
        let bytes = requirements.required_bytes();
        Self::assemble(
            scheme,
            problem,
            parameterization,
            mol,
            TimePlan::Residual(plan, requirements),
            checked_add(bytes, owned, BYTES)?,
            max_bytes,
        )
    }

    fn assemble<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        scheme: ParabolicScheme,
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        mol: MethodOfLinesPlan<R>,
        time: TimePlan<R>,
        bytes: usize,
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        if bytes > max_bytes {
            return Err(SciOxideError::resource_limit(BYTES, bytes, max_bytes));
        }
        Ok(Self {
            preparation: alloc::sync::Arc::clone(&parameterization.preparation),
            scheme,
            mol,
            time,
            token: parameterization.mesh_token().clone(),
            unknowns: problem.system().free_indices().len(),
            parameters: parameterization.len(),
            end: problem.end(),
            bytes,
        })
    }

    /// The differentiated time discretization.
    pub const fn scheme(&self) -> ParabolicScheme {
        self.scheme
    }

    /// Admitted byte bound including adapter staging, initial rates, ODE reverse
    /// frames, observation seeds and checkpoints.
    pub const fn workspace_bytes(&self) -> usize {
        self.bytes
    }

    /// Allocate all adapter and ODE numerical storage once.
    /// # Errors
    /// Propagates checked allocation failures.
    pub fn workspace(&self) -> Result<PdeTransientAdjointWorkspace<R>, SciOxideError> {
        let mut mol = self.mol.workspace::<R>()?;
        mol.reserve_parameters(self.parameters)?;
        Ok(PdeTransientAdjointWorkspace {
            mol,
            rate: try_filled(
                self.unknowns,
                R::ZERO,
                scioxide_core::quantity!("initial_rate"),
            )?,
            time: match &self.time {
                TimePlan::Explicit(_, requirements) => {
                    let mut workspace = OdeAdjointWorkspace::new();
                    workspace.try_reserve(requirements)?;
                    TimeWorkspace::Explicit(workspace)
                }
                TimePlan::Residual(_, requirements) => {
                    let mut workspace = DaeAdjointWorkspace::new();
                    workspace.try_reserve(requirements)?;
                    TimeWorkspace::Residual(workspace)
                }
            },
        })
    }

    fn check<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        &self,
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
    ) -> Result<(), SciOxideError> {
        parameterization.check_plan(problem.discretization())?;
        if !alloc::sync::Arc::ptr_eq(&self.preparation, problem.discretization().preparation()) {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "transient adjoint belongs to another prepared operator",
            ));
        }
        self.token
            .ensure_compatible(problem.discretization().mesh_token())?;
        self.token
            .ensure_compatible(parameterization.mesh_token())?;
        if problem.system().free_indices().len() != self.unknowns
            || parameterization.len() != self.parameters
            || problem.end() != self.end
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "problem or parameterization differs from the preparation",
            ));
        }
        Ok(())
    }

    /// Initial and parameter cotangents into caller slices as one epoch of `scope`
    /// (a root epoch for an execution context, a child epoch for a solve context).
    /// `initial` holds the free unknowns at time zero; `initial_gradient` receives
    /// `dJ/du_f(0)` and `parameter_gradient` `dJ/dp`.
    /// # Errors
    /// Every failure preserves both outputs; ODE failures after preflight leave the
    /// attempt's progress in [`PdeTransientAdjointWorkspace::last_attempt`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub fn solve_into<
        'scope,
        'capability,
        P: DifferentiableDiscretization<R, R, D>,
        const D: usize,
    >(
        &self,
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        initial: &[R],
        initial_gradient: &mut [R],
        parameter_gradient: &mut [R],
        workspace: &mut PdeTransientAdjointWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdjointReport<R>, SciOxideError> {
        self.check(problem, parameterization)?;
        let adjoint = problem.system().adjoint(parameterization)?;
        let interval = (R::ZERO, self.end);
        let PdeTransientAdjointWorkspace { mol, rate, time } = workspace;
        let bound = self.mol.bind_adjoint(&adjoint, mol)?;
        match (&self.time, time) {
            (TimePlan::Explicit(plan, _), TimeWorkspace::Explicit(time)) => plan.solve_into(
                &OdeAdjointProblem::new(&bound, objective, initial, interval),
                initial_gradient,
                parameter_gradient,
                time,
                scope,
            ),
            (TimePlan::Residual(plan, _), TimeWorkspace::Residual(time)) => {
                rate.fill(R::ZERO);
                plan.solve_into(
                    &DaeAdjointProblem::new(
                        &OdeAdjointResidual::new(&bound)?,
                        objective,
                        initial,
                        rate,
                        interval,
                    ),
                    initial_gradient,
                    parameter_gradient,
                    time,
                    scope,
                )
            }
            _ => Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace belongs to another time discretization",
            )),
        }
    }

    /// Allocate result and workspace and solve under sequential native execution.
    /// # Errors
    /// Returns allocation, binding and shared ODE failures with their progress.
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    pub fn solve<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        &self,
        problem: &ParabolicProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        initial: &[R],
    ) -> Result<AdjointResult<R>, PartialFailure<AdjointReport<R>>> {
        self.check(problem, parameterization)
            .map_err(PartialFailure::new)?;
        let adjoint = problem
            .system()
            .adjoint(parameterization)
            .map_err(PartialFailure::new)?;
        let mut mol = self.mol.workspace::<R>().map_err(PartialFailure::new)?;
        let bound = self
            .mol
            .bind_adjoint(&adjoint, &mut mol)
            .map_err(PartialFailure::new)?;
        let interval = (R::ZERO, self.end);
        match &self.time {
            TimePlan::Explicit(plan, _) => plan.solve(&OdeAdjointProblem::new(
                &bound, objective, initial, interval,
            )),
            TimePlan::Residual(plan, _) => {
                let rate = try_filled(self.unknowns, R::ZERO, scioxide_core::quantity!("rate"))
                    .map_err(PartialFailure::new)?;
                let residual = OdeAdjointResidual::new(&bound).map_err(PartialFailure::new)?;
                plan.solve(&DaeAdjointProblem::new(
                    &residual, objective, initial, &rate, interval,
                ))
            }
        }
    }
}
