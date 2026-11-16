//! Steady adjoints of the crate's own linear elliptic plans.
use super::*;
use alloc::sync::Arc;
use scioxide_core::{ExecutionContext, ResidualOperator, SolveScope};
use scioxide_linalg::{VectorView, VectorViewMut};
use scioxide_sparse::{
    CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseOperation, SparseOrdering,
    SparseSolveOptions, SparseSolveWorkspace,
};

use crate::{DifferentiableDiscretization, EllipticProblem, PdeParameterization};

/// Steady objective gradients of an [`EllipticProblem`] over its own linear plan.
///
/// For the eliminated residual `R(u, p) = A(p) u − b(p)` of a
/// [`DifferentiableDiscretization`] and an objective `J(u, p)` without time samples,
/// the plan solves `Aᵀ λ = ∂J/∂u` and returns `dJ/dp = ∂J/∂p − λᵀ ∂R/∂p`, with
/// `∂R/∂p` from the plan's [`PdeParameterization`]. Preparation assembles and
/// factors `A` once with fill-reducing sparse LU; every call evaluates `b`, checks
/// the supplied equilibrium against the primal tolerance and independently
/// residual-checks the transpose solve, exactly as [`PdeDiscretizationDerivatives`] does.
/// Data are evaluated at time zero, as in [`crate::EllipticProblem`]. Problems with
/// known constant modes or supplied nullspaces are rejected: their gradients
/// depend on a gauge. The plan is real-valued, `Send + Sync + Clone`, and calls
/// with a prepared workspace do not allocate.
///
/// ```
/// use scioxide_core::{SciOxideError, ResidualTolerance, SolveContext};
/// use scioxide_ode::{AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue};
/// use scioxide_pde::{
///     BoundaryCondition, BoundaryData, DifferentiableDiscretization, EllipticOptions,
///     EllipticProblem, FieldData, PdeEllipticAdjointPlan, PdeParameterTangent,
///     PdeSteadyAdjointOptions, SpatialLimits, StructuredBoundaries, StructuredFdmCoefficients,
///     StructuredFdmPlan,
/// };
/// use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};
///
/// struct Unit;
/// impl StructuredFdmCoefficients<f64, f64, 1> for Unit {
///     fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
///         Ok(1.0)
///     }
/// }
/// /// J(u) = Σ u_i.
/// struct Total;
/// impl AdjointObjective<f64> for Total {
///     fn evaluate(&self, _: AdjointObjectivePart, _: f64, u: &[f64], _: &[f64],
///         gu: &mut [f64], gp: &mut [f64], _: &mut [f64], _: &mut SolveContext<'_, '_>)
///         -> Result<AdjointObjectiveValue<f64>, SciOxideError> {
///         gu.fill(1.0);
///         gp.fill(0.0);
///         Ok(AdjointObjectiveValue { value: u.iter().sum(), time_derivative: 0.0 })
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
/// let s = 2.0;
/// let source = FieldData::constant(s);
/// let problem = EllipticProblem::new(&plan, &source);
/// let solution = (&problem).solve_with_options(&(EllipticOptions::new()))?;
/// let parameterization = plan.parameterize(&[s], &[PdeParameterTangent::new()
///     .with_source(FieldData::constant(1.0))])?;
/// let tolerance = ResidualTolerance::new(1e-10, 1e-10)?;
/// let adjoint = PdeEllipticAdjointPlan::new(&problem, &parameterization, &Total,
///     PdeSteadyAdjointOptions::new().with_primal_tolerance(tolerance).with_linear_tolerance(tolerance))?;
/// let gradient = adjoint.solve(&problem, &parameterization, solution.values(), &Total)?;
/// // u is linear in s, so dJ/ds = J/s.
/// let total: f64 = solution.values().iter().sum();
/// assert!((gradient.parameter_gradient()[0] - total / s).abs() < 1e-10);
/// # Ok::<(), SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct PdeEllipticAdjointPlan<R: RealScalar + NumericScalar<Real = R>> {
    preparation: Arc<()>,
    options: PdeSteadyAdjointOptions<R>,
    steady: PdeSteadyAdjointPlan<R>,
    token: MeshToken,
    matrix: CsrMatrix<R>,
    factor: Arc<SparseLu<R>>,
    bytes: usize,
}

/// Reusable steady staging, load and factor-solve storage.
pub struct PdeEllipticAdjointWorkspace<R: RealScalar + NumericScalar<Real = R>> {
    steady: PdeSteadyAdjointWorkspace<R>,
    load: Vec<R>,
    load_scratch: Vec<R>,
    solve: SparseSolveWorkspace<'static, R>,
}

impl<R: RealScalar + NumericScalar<Real = R>> core::fmt::Debug for PdeEllipticAdjointWorkspace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PdeEllipticAdjointWorkspace")
            .field("unknowns", &self.load.len())
            .finish_non_exhaustive()
    }
}

/// `A u − b` with a borrowed load.
struct Residual<'a, R: NumericScalar> {
    matrix: &'a CsrMatrix<R>,
    load: &'a [R],
}

impl<R: NumericScalar> ResidualOperator<R> for Residual<'_, R> {
    fn input_dimension(&self) -> usize {
        self.load.len()
    }
    fn output_dimension(&self) -> usize {
        self.load.len()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_workspace_len()
    }
    fn residual_into(
        &self,
        point: &[R],
        output: &mut [R],
        scratch: &mut [R],
    ) -> Result<(), SciOxideError> {
        self.matrix.apply_into(point, output, scratch)?;
        for (out, &b) in output.iter_mut().zip(self.load) {
            *out = scioxide_core::backend::standard::sum_products(
                [(*out, R::ONE), (-b, R::ONE)].into_iter(),
                scioxide_core::Accuracy::Standard,
            )?;
        }
        Ok(())
    }
}

/// `∂R/∂p` at one state as a real-column operator: columns are parameters.
struct Parameters<'a, R: RealScalar + NumericScalar<Real = R>, const D: usize> {
    parameterization: &'a PdeParameterization<R, R, D>,
    state: &'a [R],
    time: R,
}

impl<R: RealScalar + NumericScalar<Real = R>, const D: usize> LinearOperator<R>
    for Parameters<'_, R, D>
{
    fn rows(&self) -> usize {
        self.parameterization.unknowns()
    }
    fn cols(&self) -> usize {
        self.parameterization.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            self.parameterization.unknowns(),
            self.parameterization.scratch_len(),
            scioxide_core::quantity!("parameter_scratch"),
        )
    }
    fn apply_into(
        &self,
        input: &[R],
        output: &mut [R],
        workspace: &mut [R],
    ) -> Result<(), SciOxideError> {
        let n = self.rows();
        if input.len() != self.cols()
            || output.len() != n
            || workspace.len() < self.apply_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "parameter derivative product shape",
            ));
        }
        let (column, scratch) = workspace.split_at_mut(n);
        output.fill(R::ZERO);
        for (j, &direction) in input.iter().enumerate() {
            if direction == R::ZERO {
                continue;
            }
            self.parameterization
                .residual_tangent_into(j, self.state, self.time, column, scratch)?;
            for (out, &value) in output.iter_mut().zip(column.iter()) {
                *out = scioxide_core::backend::standard::sum_products(
                    [(*out, R::ONE), (direction, value)].into_iter(),
                    scioxide_core::Accuracy::Standard,
                )?;
            }
        }
        Ok(())
    }
}

impl<R: RealScalar + NumericScalar<Real = R>, const D: usize> AdjointOperator<R>
    for Parameters<'_, R, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        input: &[R],
        output: &mut [R],
        workspace: &mut [R],
    ) -> Result<(), SciOxideError> {
        let n = self.rows();
        if input.len() != n
            || output.len() != self.cols()
            || workspace.len() < self.apply_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "parameter derivative product shape",
            ));
        }
        let (column, scratch) = workspace.split_at_mut(n);
        for (j, out) in output.iter_mut().enumerate() {
            self.parameterization
                .residual_tangent_into(j, self.state, self.time, column, scratch)?;
            *out = scioxide_core::backend::standard::sum_products(
                column.iter().copied().zip(input.iter().copied()),
                scioxide_core::Accuracy::Standard,
            )?;
        }
        Ok(())
    }
}

impl<R: RealScalar + NumericScalar<Real = R>> PdeEllipticAdjointPlan<R> {
    /// Assemble and factor the problem's operator and admit all storage.
    /// # Errors
    /// Rejects nullspace problems and plans with constant modes, a parameterization
    /// of another mesh revision or size, time-sampled objectives, invalid options
    /// and byte limits; propagates assembly and factorization failures (a singular
    /// operator is `Singular`).
    pub fn new<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        objective: &impl AdjointObjective<R>,
        options: PdeSteadyAdjointOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let plan = problem.discretization();
        Self::check(plan, problem, parameterization)?;
        if plan.has_constant_modes() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "steady adjoints need an operator without constant modes",
            ));
        }
        let n = plan.unknowns();
        let matrix = plan.assemble_csr()?;
        let entries = matrix
            .nnz()
            .saturating_add(n)
            .saturating_mul(64)
            .max(1 << 12);
        let symbolic = SparseLuSymbolic::analyze(
            matrix.structural_pattern(),
            SparseOrdering::FillReducing,
            SparseLuLimits::new(
                entries,
                entries
                    .saturating_mul(64)
                    .min(options.max_workspace_bytes()),
            ),
        )?;
        let scratch = checked_add(
            n,
            parameterization.scratch_len(),
            scioxide_core::quantity!("parameter_scratch"),
        )?
        .max(matrix.apply_workspace_len()?)
        .max(matrix.apply_adjoint_workspace_len()?);
        const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("workspace_bytes");
        // CSR retains native and opposite traversal indices. Sparse LU's owner
        // accounts for reserved factor capacity and temporary numerical storage.
        let indices = checked_add(
            checked_mul(matrix.nnz(), 3, BYTES)?,
            checked_mul(checked_add(n, 1, BYTES)?, 2, BYTES)?,
            BYTES,
        )?;
        let matrix_bytes = checked_add(
            checked_mul(matrix.nnz(), core::mem::size_of::<R>(), BYTES)?,
            checked_mul(indices, core::mem::size_of::<usize>(), BYTES)?,
            BYTES,
        )?;
        let owned = checked_add(
            matrix_bytes,
            checked_add(
                symbolic.workspace_requirements::<R>(())?.total_bytes(),
                checked_add(
                    // Load and staging, plus the sparse solve's candidate and work.
                    checked_mul(checked_mul(n, 4, BYTES)?, core::mem::size_of::<R>(), BYTES)?,
                    core::mem::size_of::<SparseLu<R>>() + 2 * core::mem::size_of::<usize>(),
                    BYTES,
                )?,
                BYTES,
            )?,
            BYTES,
        )?;
        if owned > options.max_workspace_bytes() {
            return Err(SciOxideError::resource_limit(
                BYTES,
                owned,
                options.max_workspace_bytes(),
            ));
        }
        let steady = PdeSteadyAdjointPlan::prepared(
            n,
            parameterization.len(),
            scratch,
            objective,
            options.with_max_workspace_bytes(options.max_workspace_bytes().saturating_sub(owned)),
        )?;
        let factor = SparseLu::factor(symbolic, matrix.view())?;
        Ok(Self {
            preparation: Arc::clone(plan.preparation()),
            options,
            bytes: checked_add(steady.workspace_bytes(), owned, BYTES)?,
            steady,
            token: plan.mesh_token().clone(),
            matrix,
            factor: Arc::new(factor),
        })
    }

    fn check<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        plan: &P,
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
    ) -> Result<(), SciOxideError> {
        parameterization.check_plan(plan)?;
        plan.mesh_token()
            .ensure_compatible(parameterization.mesh_token())?;
        if problem.has_nullspace() || parameterization.unknowns() != plan.unknowns() {
            return Err(invalid(
                scioxide_core::argument!("parameterization"),
                "parameterization or nullspace incompatible with the steady adjoint",
            ));
        }
        Ok(())
    }

    /// The validated options.
    pub const fn options(&self) -> &PdeSteadyAdjointOptions<R> {
        &self.options
    }

    /// Conservative library byte bound including CSR traversals, reserved LU
    /// factors and preparation scratch, and the numerical workspace payload.
    pub const fn workspace_bytes(&self) -> usize {
        self.bytes
    }

    /// Allocate reusable storage; calls with it allocate nothing.
    /// # Errors
    /// Propagates checked allocation failures.
    pub fn workspace(&self) -> Result<PdeEllipticAdjointWorkspace<R>, SciOxideError> {
        Ok(PdeEllipticAdjointWorkspace {
            steady: self.steady.workspace()?,
            load: try_filled(
                self.matrix.rows(),
                R::ZERO,
                scioxide_core::quantity!("workspace.load"),
            )?,
            load_scratch: try_filled(
                self.matrix.rows(),
                R::ZERO,
                scioxide_core::quantity!("load_scratch"),
            )?,
            solve: self.factor.workspace(1)?,
        })
    }

    fn bind<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        &self,
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        solution: &[R],
        load: &mut [R],
        load_scratch: &mut [R],
    ) -> Result<(), SciOxideError> {
        let plan = problem.discretization();
        Self::check(plan, problem, parameterization)?;
        if !Arc::ptr_eq(&self.preparation, plan.preparation()) {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "adjoint belongs to another prepared operator",
            ));
        }
        self.token.ensure_compatible(plan.mesh_token())?;
        if plan.unknowns() != self.matrix.rows() || solution.len() != self.matrix.rows() {
            return Err(invalid(
                scioxide_core::argument!("solution"),
                "solution or problem size differs from the preparation",
            ));
        }
        plan.load_into(problem.source(), R::ZERO, load, load_scratch)
    }

    /// Solve `Aᵀ λ = ∂J/∂u` at the supplied equilibrium as one epoch of `scope` and
    /// publish multiplier and gradient together.
    /// # Errors
    /// Shape, identity, provider, budget, finite-value and accuracy failures
    /// (including a `solution` that is not an equilibrium) preserve both outputs.
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
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        solution: &[R],
        objective: &impl AdjointObjective<R>,
        adjoint: &mut VectorViewMut<'_, R>,
        gradient: &mut VectorViewMut<'_, R>,
        workspace: &mut PdeEllipticAdjointWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<R>, SciOxideError> {
        crate::validation::shape(adjoint.len(), self.matrix.rows())?;
        crate::validation::shape(gradient.len(), parameterization.len())?;
        finite_slice(solution, scioxide_core::NonFiniteOrigin::Input)?;
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "elliptic-adjoint", 1),
            |context| {
                admit(context, EvaluationRole::Function)?;
                self.bind(
                    problem,
                    parameterization,
                    solution,
                    &mut workspace.load,
                    &mut workspace.load_scratch,
                )?;
                let PdeEllipticAdjointWorkspace {
                    steady,
                    load,
                    solve,
                    load_scratch: _,
                } = workspace;
                let residual = Residual {
                    matrix: &self.matrix,
                    load,
                };
                let parameters = Parameters {
                    parameterization,
                    state: solution,
                    time: R::ZERO,
                };
                let derivatives = PdeDiscretizationDerivatives::new(
                    VectorView::from_slice(solution),
                    parameterization.values(),
                    &residual,
                    &self.matrix,
                    &parameters,
                )?;
                let factor = &self.factor;
                let mut transpose =
                    |_: &dyn LinearOperator<R>,
                     rhs: &[R],
                     x: &mut [R],
                     context: &mut SolveContext<'_, '_>| {
                        factor.solve_into(
                            rhs,
                            SparseOperation::Transpose,
                            SparseSolveOptions::new(),
                            x,
                            solve,
                            &mut *context,
                        )
                    };

                self.steady.adjoint(
                    &derivatives,
                    objective,
                    R::ZERO,
                    &mut transpose,
                    adjoint,
                    gradient,
                    steady,
                    context,
                )
            },
        )
    }

    /// Allocate outputs and workspace and solve under sequential native execution.
    /// # Errors
    /// Returns allocation and [`Self::solve_into`] failures.
    pub fn solve<P: DifferentiableDiscretization<R, R, D>, const D: usize>(
        &self,
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        solution: &[R],
        objective: &impl AdjointObjective<R>,
    ) -> Result<PdeSteadyAdjointResult<R>, SciOxideError> {
        let mut adjoint = try_filled(
            self.matrix.rows(),
            R::ZERO,
            scioxide_core::quantity!("adjoint"),
        )?;
        let mut gradient = try_filled(
            parameterization.len(),
            R::ZERO,
            scioxide_core::quantity!("parameter_gradient"),
        )?;
        let report = self.solve_into(
            problem,
            parameterization,
            solution,
            objective,
            &mut VectorViewMut::from_slice(&mut adjoint),
            &mut VectorViewMut::from_slice(&mut gradient),
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(PdeSteadyAdjointResult::new(adjoint, gradient, report))
    }

    /// Apply one real parameter direction through the implicit solution map,
    /// `du = −A⁻¹ (∂R/∂p) δp`, as one epoch of `scope`.
    /// # Errors
    /// Uses the same checks, budgets and output transaction as [`Self::solve_into`].
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub fn sensitivity_into<
        'scope,
        'capability,
        P: DifferentiableDiscretization<R, R, D>,
        const D: usize,
    >(
        &self,
        problem: &EllipticProblem<'_, R, P, D>,
        parameterization: &PdeParameterization<R, R, D>,
        solution: &[R],
        direction: &[R],
        output: &mut VectorViewMut<'_, R>,
        workspace: &mut PdeEllipticAdjointWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<R>, SciOxideError> {
        crate::validation::shape(output.len(), self.matrix.rows())?;
        crate::validation::shape(direction.len(), parameterization.len())?;
        finite_slice(direction, scioxide_core::NonFiniteOrigin::Input)?;
        finite_slice(solution, scioxide_core::NonFiniteOrigin::Input)?;
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "elliptic-sensitivity", 1),
            |context| {
                admit(context, EvaluationRole::Function)?;
                self.bind(
                    problem,
                    parameterization,
                    solution,
                    &mut workspace.load,
                    &mut workspace.load_scratch,
                )?;
                let PdeEllipticAdjointWorkspace {
                    steady,
                    load,
                    solve,
                    load_scratch: _,
                } = workspace;
                let residual = Residual {
                    matrix: &self.matrix,
                    load,
                };
                let parameters = Parameters {
                    parameterization,
                    state: solution,
                    time: R::ZERO,
                };
                let derivatives = PdeDiscretizationDerivatives::new(
                    VectorView::from_slice(solution),
                    parameterization.values(),
                    &residual,
                    &self.matrix,
                    &parameters,
                )?;
                let factor = &self.factor;
                let mut normal =
                    |_: &dyn LinearOperator<R>,
                     rhs: &[R],
                     x: &mut [R],
                     context: &mut SolveContext<'_, '_>| {
                        factor.solve_into(
                            rhs,
                            SparseOperation::Normal,
                            SparseSolveOptions::new(),
                            x,
                            solve,
                            &mut *context,
                        )
                    };

                self.steady.sensitivity(
                    &derivatives,
                    direction,
                    &mut normal,
                    output,
                    steady,
                    context,
                )
            },
        )
    }
}
