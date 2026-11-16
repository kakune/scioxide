//! Steady elliptic solves of prepared spatial discretizations.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use scioxide_core::{
    AttemptKind, AttemptRecord, BlockVectorView, Budget, BudgetKind, BudgetLimit, ExecutionContext,
    NumericScalar, PartialResultTrust, RealScalar, ResidualTolerance, SciOxideError, SolveBudgets,
    SolveScope,
};
use scioxide_linalg::{IterativeLinearSolver, Vector};

use crate::attempt::{Shell, identity};
use crate::boundary::FieldData;
use crate::discretization::SpatialDiscretization;
use crate::result::{PdeKernelOptions, PdeSolveReport};

/// Explicit gauge for known elliptic nullspaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EllipticGauge<S> {
    /// Reject known-nullspace problems until a gauge is selected.
    Unspecified,
    /// Set all supplied orthonormal right-basis coordinates to zero.
    Orthogonal,
    /// Set the quadrature-weighted mean for a structural constant mode.
    WeightedMean(S),
}

/// Configuration of an elliptic solve: gauge, compatibility, tolerances and budgets.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct EllipticOptions<S: RealScalar + NumericScalar<Real = S>> {
    residual_tolerance: ResidualTolerance<S>,
    gauge: EllipticGauge<S>,
    compatibility_policy: scioxide_linalg::CompatibilityPolicy,
    compatibility_tolerance: ResidualTolerance<S>,
    max_linear_iterations: Budget,
    direct_limits: Option<scioxide_sparse::SparseLuLimits>,
    budgets: SolveBudgets,
}

impl<S: RealScalar + NumericScalar<Real = S>> EllipticOptions<S> {
    /// Square-root-epsilon residual/compatibility tolerances, no gauge, unbounded
    /// budgets and a 1,000-iteration cap for workspace-injected iterative solvers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the explicit gauge for known nullspaces.
    pub const fn with_gauge(mut self, gauge: EllipticGauge<S>) -> Self {
        self.gauge = gauge;
        self
    }

    /// Select load compatibility handling and its independent tolerance.
    pub const fn with_compatibility(
        mut self,
        policy: scioxide_linalg::CompatibilityPolicy,
        tolerance: ResidualTolerance<S>,
    ) -> Self {
        self.compatibility_policy = policy;
        self.compatibility_tolerance = tolerance;
        self
    }

    /// Replace direct-solve backward-error and gauge tolerances.
    /// Injected iterative solvers retain their own residual tolerance.
    pub const fn with_residual_tolerance(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.residual_tolerance = tolerance;
        self
    }

    /// Replace the inner linear-iteration limit of injected iterative solvers.
    pub const fn with_max_linear_iterations(mut self, limit: Budget) -> Self {
        self.max_linear_iterations = limit;
        self
    }

    /// Bound the default sparse LU factorization used beyond compact bands.
    /// Without limits, factors may hold `64·(nnz + n)` entries.
    pub const fn with_direct_limits(mut self, limits: scioxide_sparse::SparseLuLimits) -> Self {
        self.direct_limits = Some(limits);
        self
    }

    /// Replace the local solve budgets; a finite iteration budget also caps
    /// [`Self::max_linear_iterations`].
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Configured gauge.
    pub const fn gauge(&self) -> EllipticGauge<S> {
        self.gauge
    }

    /// Configured residual tolerance.
    pub const fn residual_tolerance(&self) -> ResidualTolerance<S> {
        self.residual_tolerance
    }

    /// Maximum inner linear iterations.
    pub const fn max_linear_iterations(&self) -> Budget {
        self.max_linear_iterations
    }

    /// Configured local budgets.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    fn kernel(&self) -> Result<PdeKernelOptions<S>, SciOxideError> {
        let mut iterations = self.max_linear_iterations;
        if let BudgetLimit::Finite(limit) = self.budgets.report(BudgetKind::Iteration).limit() {
            iterations = iterations.min(Budget::try_from(limit)?);
        }
        Ok(PdeKernelOptions::default()
            .with_nullspace(
                self.gauge,
                self.compatibility_policy,
                self.compatibility_tolerance.tolerance(),
            )
            .with_tolerance(self.residual_tolerance.tolerance())
            .with_max_linear_iterations(iterations))
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for EllipticOptions<S> {
    fn default() -> Self {
        let defaults = PdeKernelOptions::<S>::default();
        Self {
            gauge: defaults.gauge,
            compatibility_policy: defaults.compatibility_policy,
            compatibility_tolerance: ResidualTolerance::from_tolerance(
                defaults.compatibility_tolerance,
            ),
            residual_tolerance: ResidualTolerance::from_tolerance(defaults.tolerance()),
            max_linear_iterations: defaults.max_linear_iterations(),
            direct_limits: None,
            budgets: SolveBudgets::unbounded(),
        }
    }
}

/// A steady problem `A u = b(0)`: a prepared discretization, its source and, for a
/// singular operator, supplied orthonormal left/right nullspace bases of the
/// eliminated operator (orthogonal gauge only).
#[must_use]
pub struct EllipticProblem<'a, S: RealScalar, P, const D: usize> {
    discretization: &'a P,
    source: &'a FieldData<S, S, D>,
    nullspace: Option<(BlockVectorView<'a, S>, BlockVectorView<'a, S>)>,
}

impl<'a, S: RealScalar + NumericScalar<Real = S>, P: SpatialDiscretization<S, S, D>, const D: usize>
    EllipticProblem<'a, S, P, D>
{
    /// Pair a prepared discretization with its source.
    pub const fn new(discretization: &'a P, source: &'a FieldData<S, S, D>) -> Self {
        Self {
            discretization,
            source,
            nullspace: None,
        }
    }

    /// Declare orthonormal left and right nullspace bases of the eliminated operator.
    pub const fn with_nullspace(
        mut self,
        left: BlockVectorView<'a, S>,
        right: BlockVectorView<'a, S>,
    ) -> Self {
        self.nullspace = Some((left, right));
        self
    }

    /// The prepared discretization.
    pub const fn discretization(&self) -> &'a P {
        self.discretization
    }

    /// The source data.
    pub const fn source(&self) -> &'a FieldData<S, S, D> {
        self.source
    }
    /// Whether orthonormal nullspace bases were declared.
    pub const fn has_nullspace(&self) -> bool {
        self.nullspace.is_some()
    }
}

/// Reusable factor storage, injected solver and last attempt of elliptic solves.
///
/// [`Self::new`] allocates nothing. A compact (tridiagonal) factor is refactored in
/// place on later solves of the same size.
pub struct EllipticWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    storage: crate::elliptic::EllipticStorage<S>,
    solver: Option<Box<dyn IterativeLinearSolver<S> + Send>>,
    shell: Shell<()>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for EllipticWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> fmt::Debug for EllipticWorkspace<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EllipticWorkspace")
            .field("injected_solver", &self.solver.is_some())
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> EllipticWorkspace<S> {
    /// An empty workspace using the default direct solves; allocates nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: crate::elliptic::EllipticStorage::default(),
            solver: None,
            shell: Shell::new(),
        }
    }

    /// Solve the eliminated operator with `solver` for every later call. The solver
    /// runs in the solve's scope with the smaller of its own and the configured
    /// iteration allowance.
    #[must_use]
    pub fn with_linear_solver(mut self, solver: Box<dyn IterativeLinearSolver<S> + Send>) -> Self {
        self.solver = Some(solver);
        self
    }

    /// Record of the most recent call; atomic solves retain no partial progress.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.shell.last_attempt()
    }
}

/// Owned result of [`EllipticProblem::solve_with_options`].
#[derive(Debug, Clone)]
#[must_use]
pub struct EllipticSolution<S: RealScalar> {
    values: Vec<S>,
    report: PdeSolveReport<S>,
}

impl<S: RealScalar> EllipticSolution<S> {
    /// Full physical field in discretization unknown order.
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Termination, diagnostics and trust.
    pub const fn report(&self) -> &PdeSolveReport<S> {
        &self.report
    }

    /// Separate the field and the report.
    pub fn into_parts(self) -> (Vec<S>, PdeSolveReport<S>) {
        (self.values, self.report)
    }
}

impl<'a, S: RealScalar + NumericScalar<Real = S>, P: SpatialDiscretization<S, S, D>, const D: usize>
    EllipticProblem<'a, S, P, D>
{
    /// Solve `A u = b(0)` with square-root-epsilon tolerances and direct factors.
    /// Uses pivoted tridiagonal factors for compact bands and fill-reducing sparse LU otherwise.
    /// Work and storage depend on sparse fill; tridiagonal problems take `O(n)`.
    /// # Errors
    /// A known nullspace requires `solve_with_options` and an explicit `EllipticGauge`.
    /// Propagates non-finite data, callback, singularity, accuracy and allocation errors.
    /// ```
    /// use scioxide_pde::{EllipticProblem, FieldData, StructuredFdmPlan};
    /// let fdm = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [9])?;
    /// let u = EllipticProblem::new(&fdm, &FieldData::constant(2.0)).solve()?;
    /// assert!((u[4] - 0.25_f64).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve(&self) -> Result<Vec<S>, SciOxideError> {
        let solution = self.solve_owned(
            &EllipticOptions::default(),
            scioxide_core::argument!("self"),
        )?;
        solution.report.require_success()?;
        Ok(solution.values)
    }
    /// Solve into owned storage with configured tolerances, gauge and budgets.
    /// # Errors
    /// Returns the errors of [`Self::solve_into`] and allocation failures.
    pub fn solve_with_options(
        &self,
        options: &EllipticOptions<S>,
    ) -> Result<EllipticSolution<S>, SciOxideError> {
        self.solve_owned(options, scioxide_core::argument!("options.gauge"))
    }
    fn solve_owned(
        &self,
        options: &EllipticOptions<S>,
        argument: scioxide_core::Argument,
    ) -> Result<EllipticSolution<S>, SciOxideError> {
        let mut values = scioxide_core::backend::try_filled(
            self.discretization.unknowns(),
            S::ZERO,
            scioxide_core::quantity!("elliptic_solution"),
        )?;
        let report = self.solve_impl(
            options,
            argument,
            &mut values,
            &mut EllipticWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(EllipticSolution { values, report })
    }
    /// Solve into the full physical field as one epoch of the caller's scope.
    /// Errors leave output unchanged; an injected solver may return an unsuccessful report.
    /// # Errors
    /// Returns configuration, shape, callback, factorization, accuracy and execution errors.
    /// Records preflight failures in [`EllipticWorkspace::last_attempt`].
    pub fn solve_into<'scope, 'capability>(
        &self,
        options: &EllipticOptions<S>,
        out: &mut [S],
        workspace: &mut EllipticWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        self.solve_impl(
            options,
            scioxide_core::argument!("options.gauge"),
            out,
            workspace,
            scope,
        )
    }
    fn preflight(
        &self,
        options: &EllipticOptions<S>,
        gauge_argument: scioxide_core::Argument,
        out: &[S],
    ) -> Result<(), SciOxideError> {
        crate::discretization::validate_structure(self.discretization)?;
        if let EllipticGauge::WeightedMean(target) = options.gauge {
            crate::spatial::finite_callback(target)?;
        }
        options.kernel().map(drop)?;
        let unknowns = self.discretization.unknowns();
        if let Some((left, right)) = self.nullspace {
            crate::validation::shape(left.rows(), unknowns)?;
            crate::validation::shape(right.rows(), unknowns)?;
            crate::validation::shape(left.right_hand_sides(), right.right_hand_sides())?;
            crate::validation::finite_slice(
                left.as_slice(),
                scioxide_core::NonFiniteOrigin::Input,
            )?;
            crate::validation::finite_slice(
                right.as_slice(),
                scioxide_core::NonFiniteOrigin::Input,
            )?;
        }
        if out.len() != unknowns {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(unknowns),
                scioxide_core::Dimensions::one(out.len()),
            ));
        }
        let known = self.discretization.has_constant_modes();
        let supplied = self.nullspace.is_some();
        let inapplicable = match options.gauge {
            EllipticGauge::Unspecified => false,
            EllipticGauge::Orthogonal => !supplied && !known,
            EllipticGauge::WeightedMean(_) => supplied || !known,
        };
        if inapplicable
            || ((known || supplied) && matches!(options.gauge, EllipticGauge::Unspecified))
        {
            return Err(SciOxideError::invalid_input(
                gauge_argument,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "use solve_with_options with an applicable EllipticGauge for these boundaries and declared bases",
            ));
        }
        Ok(())
    }

    /// Solve `problem` into `out` (the full physical field) as one epoch of `scope`.
    ///
    /// A preflight rejection (shapes, an inapplicable gauge) runs no callback. Every
    /// error leaves `out` unchanged. An injected iterative solver that stops on its
    /// iteration, condition or stagnation limit publishes its unconverged iterate with
    /// an unsuccessful report and `DiagnosticOnly` trust.
    ///
    /// # Errors
    /// Returns shape, gauge, callback, assembly, factorization, accuracy and budget
    /// errors; [`EllipticWorkspace::last_attempt`] records the outcome.
    fn solve_impl<'scope, 'capability>(
        &self,
        options: &EllipticOptions<S>,
        gauge_argument: scioxide_core::Argument,
        out: &mut [S],
        workspace: &mut EllipticWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        let scope = scope.into();
        if let Err(error) = self.preflight(options, gauge_argument, out) {
            return workspace.shell.reject(AttemptKind::Call, error);
        }
        let kernel = match options.kernel() {
            Ok(kernel) => kernel,
            Err(error) => return workspace.shell.reject(AttemptKind::Call, error),
        };
        let EllipticWorkspace {
            storage,
            solver,
            shell,
        } = workspace;
        let plan = self.discretization;
        let executed = shell.execute(
            scope,
            identity(
                scioxide_core::algorithm_id!("pde", "elliptic", 4),
                plan.unknowns() as u128,
            ),
            options.budgets,
            |context| {
                context.execution().check()?;
                let known = plan.has_constant_modes();
                crate::adjoint::admit(context, scioxide_core::EvaluationRole::Function)?;
                let rhs = Vector::from_vec(plan.load(self.source, S::ZERO)?);
                crate::validation::shape(rhs.len(), plan.unknowns())?;
                crate::validation::finite_slice(
                    rhs.as_slice(),
                    scioxide_core::NonFiniteOrigin::Callback,
                )?;
                let matrix = plan.assemble_csr()?;
                crate::validation::shape(matrix.rows(), plan.unknowns())?;
                crate::validation::shape(matrix.cols(), plan.unknowns())?;
                let bands = crate::elliptic::EllipticMatrix::from_csr(&matrix)?;
                let system = crate::helpers::EllipticSystem {
                    gauge_argument,
                    matrix: &matrix,
                    bands,
                    constant_components: known.then(|| plan.components()),
                    weights: plan.lumped_mass(),
                    direct_limits: options.direct_limits,
                };
                let (values, diagnostics) = crate::helpers::solve_elliptic_system(
                    &system,
                    &rhs,
                    self.nullspace,
                    &kernel,
                    solver
                        .as_deref()
                        .map(|solver| solver as &dyn IterativeLinearSolver<S>),
                    storage,
                    context,
                )?;
                crate::validation::finite_slice(&values, scioxide_core::NonFiniteOrigin::Output)?;
                let trust = if !diagnostics.termination().is_success() {
                    PartialResultTrust::DiagnosticOnly
                } else if diagnostics.residual_norm().is_some() {
                    PartialResultTrust::ResidualChecked
                } else {
                    PartialResultTrust::Admissible
                };
                context.execution().check()?;
                out.copy_from_slice(&values);
                Ok(PdeSolveReport::new(diagnostics, trust))
            },
        );
        shell.conclude(
            AttemptKind::Call,
            executed,
            PdeSolveReport::is_success,
            |_| None,
        )
    }
}
