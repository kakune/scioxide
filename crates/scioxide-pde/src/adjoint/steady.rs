use super::*;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    ExecutionContext, ResidualOperator, ResidualTolerance, SolveCounts, SolveScope,
};
use scioxide_linalg::{VectorView, VectorViewMut};

/// Discrete residual derivatives at one immutable supplied solution and parameter point.
///
/// `state` represents R_u and `parameters` R_p for the same residual, including
/// boundary lifting. Parameter columns are real directions even for complex
/// residuals. Dense/CSR views and AD/core operators share this interface; the
/// crate's own linear plans supply all three through [`super::PdeEllipticAdjointPlan`].
/// A provider must remain fixed while this binding is used.
pub struct PdeDiscretizationDerivatives<'a, T: NumericScalar> {
    solution: VectorView<'a, T>,
    parameter_values: &'a [T::Real],
    residual: &'a dyn ResidualOperator<T>,
    state: &'a dyn AdjointOperator<T>,
    parameters: &'a dyn AdjointOperator<T>,
}
impl<'a, T: NumericScalar> PdeDiscretizationDerivatives<'a, T> {
    /// Bind borrowed operators at the supplied state without allocating.
    /// # Errors
    /// Rejects incompatible dimensions and nonfinite parameter values.
    pub fn new(
        solution: VectorView<'a, T>,
        parameter_values: &'a [T::Real],
        residual: &'a dyn ResidualOperator<T>,
        state: &'a dyn AdjointOperator<T>,
        parameters: &'a dyn AdjointOperator<T>,
    ) -> Result<Self, SciOxideError> {
        let n = solution.len();
        let p = parameter_values.len();

        if n == 0
            || residual.input_dimension() != n
            || residual.output_dimension() != n
            || state.rows() != n
            || state.cols() != n
            || parameters.rows() != n
            || parameters.cols() != p
        {
            return Err(invalid(
                if n == 0 {
                    scioxide_core::argument!("solution")
                } else if residual.input_dimension() != n || residual.output_dimension() != n {
                    scioxide_core::argument!("residual")
                } else if state.rows() != n || state.cols() != n {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("parameters")
                },
                "discretization derivative shapes differ",
            ));
        }
        finite_slice(parameter_values, scioxide_core::NonFiniteOrigin::Input)?;
        Ok(Self {
            solution,
            parameter_values,
            residual,
            state,
            parameters,
        })
    }
}
/// Algebraic diagnostics of a steady objective gradient. They certify the
/// discrete gradient only; no residual observation bounds continuum accuracy.
#[derive(Clone, Debug)]
pub struct PdeSteadyAdjointReport<S: RealScalar> {
    objective: Option<S>,
    primal_residual: S,
    derivative_residual: S,
    counts: SolveCounts,
    workspace_bytes: usize,
}
impl<S: RealScalar> PdeSteadyAdjointReport<S> {
    /// Objective at the supplied equilibrium; absent for a forward product.
    pub const fn objective(&self) -> Option<S> {
        self.objective
    }
    /// Norm of the independently evaluated primal residual.
    pub const fn primal_residual(&self) -> S {
        self.primal_residual
    }
    /// Norm of `R_u* lambda - J_u` (or the forward tangent equation).
    pub const fn derivative_residual(&self) -> S {
        self.derivative_residual
    }
    /// Ledger of the epoch after successful publication.
    pub const fn counts(&self) -> SolveCounts {
        self.counts
    }
    /// Admitted library workspace payload; caller solver storage is separate.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}
/// Owned steady adjoint result of [`PdeDiscretizationDerivatives::solve_adjoint`].
#[derive(Debug, Clone)]
pub struct PdeSteadyAdjointResult<T: NumericScalar> {
    adjoint: Vec<T>,
    parameter_gradient: Vec<T::Real>,
    report: PdeSteadyAdjointReport<T::Real>,
}
impl<T: NumericScalar> PdeSteadyAdjointResult<T> {
    pub(super) const fn new(
        adjoint: Vec<T>,
        parameter_gradient: Vec<T::Real>,
        report: PdeSteadyAdjointReport<T::Real>,
    ) -> Self {
        Self {
            adjoint,
            parameter_gradient,
            report,
        }
    }

    /// Residual-equation multiplier, not a state or initial-condition gradient.
    pub fn adjoint(&self) -> &[T] {
        &self.adjoint
    }
    /// Total real parameter gradient `J_p - R_p* lambda`.
    pub fn parameter_gradient(&self) -> &[T::Real] {
        &self.parameter_gradient
    }
    /// Checked residual diagnostics.
    pub const fn report(&self) -> &PdeSteadyAdjointReport<T::Real> {
        &self.report
    }
}
/// Tolerances, residual units and the workspace byte limit of steady adjoints.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PdeSteadyAdjointOptions<R: RealScalar> {
    primal: ResidualTolerance<R>,
    linear: ResidualTolerance<R>,
    primal_scale: R,
    max_workspace_bytes: usize,
}
impl<R: RealScalar> PdeSteadyAdjointOptions<R> {
    /// Check the supplied equilibrium against `primal` (in units of a unit residual
    /// scale) and every caller linear solve against `linear`; no byte limit.
    pub fn new() -> Self {
        Self {
            primal: ResidualTolerance::from_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon()),
            linear: ResidualTolerance::from_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon()),
            primal_scale: R::ONE,
            max_workspace_bytes: usize::MAX,
        }
    }
    /// Set the tolerance for the supplied equilibrium.
    pub const fn with_primal_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.primal = tolerance;
        self
    }
    /// Set the independently checked linear residual tolerance.
    pub const fn with_linear_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.linear = tolerance;
        self
    }
    /// Units of the primal residual tolerance; finite and nonnegative.
    pub const fn with_primal_scale(mut self, scale: R) -> Self {
        self.primal_scale = scale;
        self
    }
    /// Ceiling on the library workspace payload in bytes.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    /// Primal equilibrium tolerance.
    pub const fn primal_tolerance(&self) -> ResidualTolerance<R> {
        self.primal
    }
    /// Linear-solve residual tolerance.
    pub const fn linear_tolerance(&self) -> ResidualTolerance<R> {
        self.linear
    }
    /// Primal residual units.
    pub const fn primal_scale(&self) -> R {
        self.primal_scale
    }
    /// Workspace byte ceiling.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
}
/// A caller linear solve `A x = b` of the supplied operator from a zero solution.
/// It receives the epoch's context and owns its factors or Krylov storage; its
/// result is independently residual-checked.
pub type PdeLinearSolve<'f, T> = dyn FnMut(
        &dyn LinearOperator<T>,
        &[T],
        &mut [T],
        &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    + 'f;
/// Immutable implicit differentiation of existing equilibria of a bound residual;
/// owns no nonlinear solve and no caller provider, and is `Send + Sync + Clone`.
///
/// Every call takes the [`PdeDiscretizationDerivatives`] of its equilibrium, whose
/// shapes and scratch must match the preparation.
#[derive(Clone, Debug)]
#[must_use]
pub(super) struct PdeSteadyAdjointPlan<T: NumericScalar> {
    options: PdeSteadyAdjointOptions<T::Real>,
    unknowns: usize,
    parameters: usize,
    scratch: usize,
    objective_scratch: usize,
    bytes: usize,
}
/// Reusable transactional state, product and output staging.
pub struct PdeSteadyAdjointWorkspace<T: NumericScalar> {
    point: Vec<T>,
    rhs: Vec<T>,
    solution: Vec<T>,
    residual: Vec<T>,
    parameter: Vec<T>,
    gradient: Vec<T::Real>,
    scratch: Vec<T>,
    objective: Vec<T>,
}
impl<T: NumericScalar> PdeSteadyAdjointPlan<T> {
    /// Admit exact adapter storage for `derivatives` and `objective` before any
    /// allocation or numerical callback.
    /// # Errors
    /// Rejects time-sampled objectives, unavailable products, an invalid residual
    /// scale and insufficient bytes. Fixed-point and inverse regularity remain
    /// assumptions.
    pub(super) fn new(
        derivatives: &PdeDiscretizationDerivatives<'_, T>,
        objective: &impl AdjointObjective<T>,
        options: PdeSteadyAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::prepared(
            derivatives.solution.len(),
            derivatives.parameter_values.len(),
            Self::scratch(derivatives)?,
            objective,
            options,
        )
    }

    /// Admit storage for `n` unknowns, `p` parameters and `scratch` operator scratch.
    pub(super) fn prepared(
        n: usize,
        p: usize,
        scratch: usize,
        objective: &impl AdjointObjective<T>,
        options: PdeSteadyAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let primal_scale = options.primal_scale;
        if !objective.observation_times().is_empty()
            || !primal_scale.is_finite()
            || primal_scale < T::Real::ZERO
        {
            return Err(invalid(
                if objective.observation_times().is_empty() {
                    scioxide_core::argument!("options")
                } else {
                    scioxide_core::argument!("objective")
                },
                "steady objective requires no time samples and a finite nonnegative residual scale",
            ));
        }
        Self::sized(n, p, scratch, objective.workspace_len(), options)
    }
    fn sized(
        n: usize,
        p: usize,
        scratch: usize,
        objective_scratch: usize,
        options: PdeSteadyAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        if !options.primal_scale.is_finite() || options.primal_scale < T::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "residual scale must be finite and nonnegative",
            ));
        }
        const SLOTS: scioxide_core::Quantity = scioxide_core::quantity!("workspace_slots");
        const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("workspace_bytes");
        let slots = checked_add(
            checked_add(checked_mul(n, 4, SLOTS)?, p, SLOTS)?,
            checked_add(scratch, objective_scratch, SLOTS)?,
            SLOTS,
        )?;
        let bytes = checked_add(
            checked_mul(slots, core::mem::size_of::<T>(), BYTES)?,
            checked_mul(p, core::mem::size_of::<T::Real>(), BYTES)?,
            BYTES,
        )?;
        if bytes > options.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                BYTES,
                bytes,
                options.max_workspace_bytes,
            ));
        }
        Ok(Self {
            options,
            unknowns: n,
            parameters: p,
            scratch,
            objective_scratch,
            bytes,
        })
    }
    pub(super) fn scratch(
        derivatives: &PdeDiscretizationDerivatives<'_, T>,
    ) -> Result<usize, SciOxideError> {
        Ok([
            derivatives.residual.residual_workspace_len()?,
            derivatives.state.apply_workspace_len()?,
            derivatives.state.apply_adjoint_workspace_len()?,
            derivatives.parameters.apply_workspace_len()?,
            derivatives.parameters.apply_adjoint_workspace_len()?,
        ]
        .into_iter()
        .max()
        .unwrap_or(0))
    }
    /// Exact numerical payload allocated by `workspace`.
    pub(super) const fn workspace_bytes(&self) -> usize {
        self.bytes
    }
    /// Allocate reusable numerical storage. Execution performs no adapter allocation.
    /// # Errors
    /// Propagates checked allocation failure.
    pub(super) fn workspace(&self) -> Result<PdeSteadyAdjointWorkspace<T>, SciOxideError> {
        let (n, p) = (self.unknowns, self.parameters);
        Ok(PdeSteadyAdjointWorkspace {
            point: try_filled(n, T::ZERO, scioxide_core::quantity!("workspace.point"))?,
            rhs: try_filled(n, T::ZERO, scioxide_core::quantity!("workspace.rhs"))?,
            solution: try_filled(n, T::ZERO, scioxide_core::quantity!("workspace.solution"))?,
            residual: try_filled(n, T::ZERO, scioxide_core::quantity!("workspace.residual"))?,
            parameter: try_filled(p, T::ZERO, scioxide_core::quantity!("workspace.parameter"))?,
            gradient: try_filled(
                p,
                T::Real::ZERO,
                scioxide_core::quantity!("workspace.gradient"),
            )?,
            scratch: try_filled(
                self.scratch,
                T::ZERO,
                scioxide_core::quantity!("workspace.scratch"),
            )?,
            objective: try_filled(
                self.objective_scratch,
                T::ZERO,
                scioxide_core::quantity!("workspace.objective"),
            )?,
        })
    }
    /// Check `d` against the preparation, then evaluate the supplied equilibrium.
    fn prepare(
        &self,
        d: &PdeDiscretizationDerivatives<'_, T>,
        work: &mut PdeSteadyAdjointWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        let (n, p) = (self.unknowns, self.parameters);
        if d.solution.len() != n
            || d.parameter_values.len() != p
            || work.point.len() != n
            || work.rhs.len() != n
            || work.solution.len() != n
            || work.residual.len() != n
            || work.parameter.len() != p
            || work.gradient.len() != p
            || work.scratch.len() != self.scratch
            || work.objective.len() != self.objective_scratch
            || d.state.rows() != n
            || d.state.cols() != n
            || d.parameters.rows() != n
            || d.parameters.cols() != p
            || d.residual.input_dimension() != n
            || d.residual.output_dimension() != n
        {
            return Err(invalid(
                scioxide_core::argument!("derivatives"),
                "steady adjoint workspace or provider dimensions differ from the plan",
            ));
        }
        if Self::scratch(d)? > self.scratch {
            return Err(invalid(
                scioxide_core::argument!("derivatives"),
                "derivative scratch exceeds the preparation",
            ));
        }
        for (i, out) in work.point.iter_mut().enumerate() {
            *out = d.solution.get(i)?;
        }
        finite_slice(&work.point, scioxide_core::NonFiniteOrigin::Input)?;
        finite_slice(d.parameter_values, scioxide_core::NonFiniteOrigin::Input)?;
        poison(&mut work.residual);
        admit(context, EvaluationRole::Residual)?;
        d.residual.residual_into_scoped(
            &work.point,
            &mut work.residual,
            &mut work.scratch[..d.residual.residual_workspace_len()?],
            context,
        )?;
        finite_slice(&work.residual, scioxide_core::NonFiniteOrigin::Callback)?;
        let error = norm(&work.residual)?;
        if error > self.options.primal.threshold(self.options.primal_scale)? {
            return Err(invalid(
                scioxide_core::argument!("derivatives"),
                "supplied equilibrium residual exceeds tolerance",
            ));
        }
        Ok(error)
    }
    /// Solve the transpose equation as one epoch of `scope` and publish multiplier and
    /// gradient together. `solve` must solve the supplied operator; its result is
    /// independently residual-checked, even if it reports success.
    /// # Errors
    /// Provider, budget, shape, finite-value and accuracy failures preserve both outputs.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub(super) fn solve_into<'scope, 'capability>(
        &self,
        derivatives: &PdeDiscretizationDerivatives<'_, T>,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
        adjoint: &mut VectorViewMut<'_, T>,
        gradient: &mut VectorViewMut<'_, T::Real>,
        workspace: &mut PdeSteadyAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "steady-adjoint", 2),
            |context| {
                self.adjoint(
                    derivatives,
                    objective,
                    time,
                    solve,
                    adjoint,
                    gradient,
                    workspace,
                    context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub(super) fn adjoint(
        &self,
        d: &PdeDiscretizationDerivatives<'_, T>,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
        adjoint: &mut VectorViewMut<'_, T>,
        gradient: &mut VectorViewMut<'_, T::Real>,
        work: &mut PdeSteadyAdjointWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        if !time.is_finite()
            || adjoint.len() != d.solution.len()
            || gradient.len() != d.parameter_values.len()
            || objective.workspace_len() > self.objective_scratch
            || !objective.observation_times().is_empty()
        {
            return Err(invalid(
                if !time.is_finite() {
                    scioxide_core::argument!("time")
                } else if adjoint.len() != d.solution.len() {
                    scioxide_core::argument!("adjoint")
                } else if gradient.len() != d.parameter_values.len() {
                    scioxide_core::argument!("gradient")
                } else {
                    scioxide_core::argument!("objective")
                },
                "steady objective/output shape or time",
            ));
        }
        let primal_residual = self.prepare(d, work, context)?;
        poison(&mut work.rhs);
        work.gradient.fill(T::Real::NAN);
        admit(context, EvaluationRole::Objective)?;
        let value = objective.evaluate(
            AdjointObjectivePart::Terminal,
            time,
            &work.point,
            d.parameter_values,
            &mut work.rhs,
            &mut work.gradient,
            &mut work.objective,
            context,
        )?;
        finite_slice(&work.rhs, scioxide_core::NonFiniteOrigin::Callback)?;
        finite_slice(&work.gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        if !value.value.is_finite() || !value.time_derivative.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "nonfinite objective",
            ));
        }
        work.solution.fill(T::ZERO);
        admit(context, EvaluationRole::LinearSolve)?;
        solve(&Transpose(d.state), &work.rhs, &mut work.solution, context)?;
        finite_slice(&work.solution, scioxide_core::NonFiniteOrigin::Callback)?;
        poison(&mut work.residual);
        admit(context, EvaluationRole::LinearOperator)?;
        d.state
            .apply_adjoint_into_scoped(
                &work.solution,
                &mut work.residual,
                &mut work.scratch[..d.state.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(&work.residual, scioxide_core::NonFiniteOrigin::Callback)?;
        let derivative_residual = difference(&work.residual, &work.rhs)?;
        if derivative_residual > self.options.linear.threshold(norm(&work.rhs)?)? {
            return Err(invalid(
                scioxide_core::argument!("solve"),
                "transpose solve residual exceeds tolerance",
            ));
        }
        poison(&mut work.parameter);
        admit(context, EvaluationRole::LinearOperator)?;
        d.parameters
            .apply_adjoint_into_scoped(
                &work.solution,
                &mut work.parameter,
                &mut work.scratch[..d.parameters.apply_adjoint_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(&work.parameter, scioxide_core::NonFiniteOrigin::Callback)?;
        for (out, &v) in work.gradient.iter_mut().zip(&work.parameter) {
            *out -= v.real_component();
        }
        finite_slice(&work.gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        context.execution().check()?;
        for (i, &v) in work.solution.iter().enumerate() {
            adjoint.set(i, v)?;
        }
        for (i, &v) in work.gradient.iter().enumerate() {
            gradient.set(i, v)?;
        }
        Ok(PdeSteadyAdjointReport {
            objective: Some(value.value),
            primal_residual,
            derivative_residual,
            counts: context.counts(),
            workspace_bytes: self.bytes,
        })
    }
    /// Allocate multiplier/gradient outputs and use the checked borrowed path under
    /// sequential native execution.
    /// # Errors
    /// Returns allocation and `solve_into` failures.
    pub(super) fn solve(
        &self,
        derivatives: &PdeDiscretizationDerivatives<'_, T>,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
    ) -> Result<PdeSteadyAdjointResult<T>, SciOxideError> {
        let mut adjoint = try_filled(self.unknowns, T::ZERO, scioxide_core::quantity!("adjoint"))?;
        let mut parameter_gradient = try_filled(
            self.parameters,
            T::Real::ZERO,
            scioxide_core::quantity!("parameter_gradient"),
        )?;
        let report = self.solve_into(
            derivatives,
            objective,
            time,
            solve,
            &mut VectorViewMut::from_slice(&mut adjoint),
            &mut VectorViewMut::from_slice(&mut parameter_gradient),
            &mut self.workspace()?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(PdeSteadyAdjointResult {
            adjoint,
            parameter_gradient,
            report,
        })
    }
    /// Apply one real parameter direction through the implicit solution map as one
    /// epoch of `scope`.
    /// # Errors
    /// Uses the same residual checks, budgets and output transaction as `solve_into`.
    pub(super) fn sensitivity_into<'scope, 'capability>(
        &self,
        derivatives: &PdeDiscretizationDerivatives<'_, T>,
        direction: &[T::Real],
        solve: &mut PdeLinearSolve<'_, T>,
        output: &mut VectorViewMut<'_, T>,
        workspace: &mut PdeSteadyAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "steady-sensitivity", 2),
            |context| self.sensitivity(derivatives, direction, solve, output, workspace, context),
        )
    }
    pub(super) fn sensitivity(
        &self,
        d: &PdeDiscretizationDerivatives<'_, T>,
        direction: &[T::Real],
        solve: &mut PdeLinearSolve<'_, T>,
        output: &mut VectorViewMut<'_, T>,
        work: &mut PdeSteadyAdjointWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        if direction.len() != d.parameter_values.len() || output.len() != d.solution.len() {
            return Err(invalid(
                if direction.len() != d.parameter_values.len() {
                    scioxide_core::argument!("direction")
                } else {
                    scioxide_core::argument!("output")
                },
                "sensitivity direction/output shape",
            ));
        }
        finite_slice(direction, scioxide_core::NonFiniteOrigin::Input)?;
        let primal_residual = self.prepare(d, work, context)?;
        for (out, &v) in work.parameter.iter_mut().zip(direction) {
            *out = T::from_real(v);
        }
        poison(&mut work.rhs);
        admit(context, EvaluationRole::LinearOperator)?;
        d.parameters
            .apply_into_scoped(
                &work.parameter,
                &mut work.rhs,
                &mut work.scratch[..d.parameters.apply_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(&work.rhs, scioxide_core::NonFiniteOrigin::Callback)?;
        for v in &mut work.rhs {
            *v = -*v;
        }
        work.solution.fill(T::ZERO);
        admit(context, EvaluationRole::LinearSolve)?;
        solve(d.state, &work.rhs, &mut work.solution, context)?;
        finite_slice(&work.solution, scioxide_core::NonFiniteOrigin::Callback)?;
        poison(&mut work.residual);
        admit(context, EvaluationRole::LinearOperator)?;
        d.state
            .apply_into_scoped(
                &work.solution,
                &mut work.residual,
                &mut work.scratch[..d.state.apply_workspace_len()?],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        finite_slice(&work.residual, scioxide_core::NonFiniteOrigin::Callback)?;
        let derivative_residual = difference(&work.residual, &work.rhs)?;
        if derivative_residual > self.options.linear.threshold(norm(&work.rhs)?)? {
            return Err(invalid(
                scioxide_core::argument!("solve"),
                "forward sensitivity residual exceeds tolerance",
            ));
        }
        context.execution().check()?;
        for (i, &v) in work.solution.iter().enumerate() {
            output.set(i, v)?;
        }
        Ok(PdeSteadyAdjointReport {
            objective: None,
            primal_residual,
            derivative_residual,
            counts: context.counts(),
            workspace_bytes: self.bytes,
        })
    }
}

impl<R: RealScalar> Default for PdeSteadyAdjointOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NumericScalar> PdeDiscretizationDerivatives<'_, T> {
    /// Allocate adjoint staging for this binding and objective without numerical callbacks.
    /// # Errors
    /// Rejects incompatible objective, unavailable products, invalid options and allocation failures.
    pub fn workspace(
        &self,
        objective: &impl AdjointObjective<T>,
        options: &PdeSteadyAdjointOptions<T::Real>,
    ) -> Result<PdeSteadyAdjointWorkspace<T>, SciOxideError> {
        PdeSteadyAdjointPlan::new(self, objective, *options)?.workspace()
    }
    /// Compute a steady objective gradient with square-root-epsilon residual tolerances.
    /// # Errors
    /// Propagates provider, shape, allocation and independently checked residual errors.
    pub fn solve_adjoint(
        &self,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
    ) -> Result<PdeSteadyAdjointResult<T>, SciOxideError> {
        self.solve_adjoint_with_options(objective, time, solve, &PdeSteadyAdjointOptions::default())
    }
    /// Compute the steady adjoint and parameter gradient with explicit tolerances and limits.
    /// # Errors
    /// Propagates configuration, provider, allocation and residual-check failures.
    pub fn solve_adjoint_with_options(
        &self,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
        options: &PdeSteadyAdjointOptions<T::Real>,
    ) -> Result<PdeSteadyAdjointResult<T>, SciOxideError> {
        PdeSteadyAdjointPlan::new(self, objective, *options)?.solve(self, objective, time, solve)
    }
    /// Publish the multiplier and gradient atomically inside the caller's solve.
    /// # Errors
    /// Provider, budget, shape, finite-value and accuracy failures preserve both outputs.
    #[expect(
        clippy::too_many_arguments,
        reason = "operands, output staging and scope are explicit"
    )]
    pub fn solve_adjoint_into<'scope, 'capability>(
        &self,
        objective: &impl AdjointObjective<T>,
        time: T::Real,
        solve: &mut PdeLinearSolve<'_, T>,
        options: &PdeSteadyAdjointOptions<T::Real>,
        adjoint: &mut VectorViewMut<'_, T>,
        gradient: &mut VectorViewMut<'_, T::Real>,
        workspace: &mut PdeSteadyAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        PdeSteadyAdjointPlan::new(self, objective, *options)?.solve_into(
            self, objective, time, solve, adjoint, gradient, workspace, scope,
        )
    }
    /// Apply a real parameter direction through the implicit solution map.
    /// Reuses any objective staging already retained by the workspace.
    /// # Errors
    /// Preserves output on shape, provider, allocation, budget and residual-check failures.
    pub fn sensitivity_into<'scope, 'capability>(
        &self,
        direction: &[T::Real],
        solve: &mut PdeLinearSolve<'_, T>,
        options: &PdeSteadyAdjointOptions<T::Real>,
        output: &mut VectorViewMut<'_, T>,
        workspace: &mut PdeSteadyAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSteadyAdjointReport<T::Real>, SciOxideError> {
        PdeSteadyAdjointPlan::sized(
            self.solution.len(),
            self.parameter_values.len(),
            PdeSteadyAdjointPlan::scratch(self)?,
            workspace.objective.len(),
            *options,
        )?
        .sensitivity_into(self, direction, solve, output, workspace, scope)
    }
}
