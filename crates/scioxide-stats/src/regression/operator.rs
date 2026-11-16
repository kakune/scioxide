//! Linear least squares with a matrix-free design, solved by LSQR or LSMR.
use super::{LinearRegressionFit, budget_bytes, invalid};
use crate::RegressionWeights;
use scioxide_core::AlgebraicScalar;

use scioxide_core::{
    AdjointOperator, Budget, EvaluationRole, GradientTolerance, LinearOperator, NumericScalar,
    ProblemToken, RealScalar, ResidualTolerance, SciOxideError, SolveContext, SolveIdentity,
};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, KrylovDiagnostics, LinearSolveTermination,
    LsmrSolver, LsqrSolver, Vector,
};

/// Selects the Krylov method of a [`PreparedOperatorRegression`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OperatorRegressionMethod {
    /// LSQR: Golub–Kahan bidiagonalization, equivalent to conjugate gradients
    /// on the normal equations. `‖r‖₂` decreases monotonically.
    ///
    /// # References
    ///
    /// C. C. Paige and M. A. Saunders, "LSQR: An algorithm for sparse linear
    /// equations and sparse least squares", *ACM Trans. Math. Software* 8(1),
    /// 43–71, 1982.
    Lsqr,
    /// LSMR: Golub–Kahan bidiagonalization that minimizes `‖Aᴴ r‖₂` at each
    /// step, equivalent to MINRES on the normal equations.
    ///
    /// # References
    ///
    /// D. C.-L. Fong and M. A. Saunders, "LSMR: An iterative algorithm for
    /// sparse least-squares problems", *SIAM J. Sci. Comput.* 33(5),
    /// 2950–2971, 2011.
    #[default]
    Lsmr,
}
/// Options of a [`PreparedOperatorRegression`]: the method, stopping
/// tolerances, an iteration limit and a byte limit.
///
/// Defaults to LSMR, the scalar-derived residual and gradient tolerances,
/// 4096 iterations and no byte limit, matching the inner GLM solve.
///
/// # References
///
/// D. C.-L. Fong and M. A. Saunders, SIAM J. Sci. Comput. 33 (2011), 2950–2971.
#[derive(Clone, Copy, Debug)]
pub struct OperatorRegressionOptions<S: RealScalar> {
    /// Krylov method.
    method: OperatorRegressionMethod,
    /// Tolerance on the residual norm.
    residual: ResidualTolerance<S>,
    /// Tolerance on the normal-equation residual norm.
    normal: GradientTolerance<S>,
    /// Iteration limit. The solve scope can impose a lower one.
    iterations: Budget,
    max_bytes: usize,
}
impl<S: RealScalar> Default for OperatorRegressionOptions<S> {
    fn default() -> Self {
        Self {
            method: OperatorRegressionMethod::Lsmr,
            residual: Default::default(),
            normal: Default::default(),
            iterations: Budget::ONE.saturating_add(4095),
            max_bytes: usize::MAX,
        }
    }
}
impl<S: RealScalar> OperatorRegressionOptions<S> {
    /// Sets the Krylov method.
    pub fn with_method(mut self, value: OperatorRegressionMethod) -> Self {
        self.method = value;
        self
    }
    /// Returns the Krylov method.
    pub fn method(&self) -> OperatorRegressionMethod {
        self.method
    }
    /// Sets the tolerance on the residual norm `‖W^(1/2) (y − X β)‖₂`.
    pub fn with_residual(mut self, value: ResidualTolerance<S>) -> Self {
        self.residual = value;
        self
    }
    /// Returns the tolerance on the residual norm.
    pub fn residual(&self) -> ResidualTolerance<S> {
        self.residual
    }
    /// Sets the tolerance on the normal-equation residual norm
    /// `‖Xᴴ W (y − X β)‖₂`.
    pub fn with_normal(mut self, value: GradientTolerance<S>) -> Self {
        self.normal = value;
        self
    }
    /// Returns the tolerance on the normal-equation residual norm.
    pub fn normal(&self) -> GradientTolerance<S> {
        self.normal
    }
    /// Sets the iteration limit.
    pub fn with_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Returns the iteration limit.
    pub fn iterations(&self) -> Budget {
        self.iterations
    }
    /// Sets the byte limit for workspaces and outputs. The default is
    /// `usize::MAX`.
    pub fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the byte limit.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}
#[derive(Clone)]
enum Solver<T: NumericScalar> {
    Lsqr(LsqrSolver<T>),
    Lsmr(LsmrSolver<T>),
}
/// Holds a borrowed design operator and row weights for weighted least squares
/// by LSQR or LSMR.
///
/// A fit solves `minimize ‖W^(1/2) (y − X β)‖₂` over `β` from the starting
/// point `β = 0`, where `X` is the `n × p` operator and `W = diag(w₁, …, wₙ)`.
/// The design is used only through products `X v` and `Xᴴ u`, so it can be
/// dense, sparse or matrix-free. Each iteration costs one product with `X`, one
/// with `Xᴴ` and O(n + p) work. The rank of the design is not computed, so fits
/// report no rank or degrees of freedom. For a dense design that fits in
/// memory, [`crate::PreparedRegression`] solves by a direct factorization and
/// supports inference.
///
/// The plan is `Clone`, and `Send + Sync` when `O` is `Sync`.
pub struct PreparedOperatorRegression<'a, T: NumericScalar, O: ?Sized> {
    operator: &'a O,
    roots: Vector<T::Real>,
    solver: Solver<T>,
    options: OperatorRegressionOptions<T::Real>,
}
/// Holds the scratch storage of [`PreparedOperatorRegression::fit_into`] and a
/// record of the last fit.
///
/// Create it with [`PreparedOperatorRegression::workspace`].
pub struct OperatorRegressionWorkspace<T: NumericScalar> {
    engine: IterativeWorkspace<T>,
    rhs: Vector<T>,
    stage: LinearRegressionFit<T>,
    provider: Vector<T>,
    normal: Vector<T>,
    slot: crate::execution::SolveSlot<OperatorRegressionReport<T::Real>>,
    retained_bytes: usize,
}
/// Describes how a [`PreparedOperatorRegression`] fit ended.
#[derive(Clone, Copy, Debug)]
pub struct OperatorRegressionReport<S: RealScalar> {
    /// Reason the solve stopped.
    pub termination: LinearSolveTermination,
    /// Number of Krylov iterations.
    pub iterations: usize,
    /// Condition and rank-loss estimates of the projected problem, when the
    /// method provides them. They do not give the rank of the design.
    pub krylov: Option<KrylovDiagnostics<S>>,
    /// Whether the fit was written to the output.
    ///
    /// It is `false` when execution stopped or the evaluation limit left no
    /// room for the two products that compute the residuals. It says nothing
    /// about convergence: check `termination` for that.
    pub output_committed: bool,
}
impl<T: NumericScalar, O: ?Sized> Clone for PreparedOperatorRegression<'_, T, O> {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator,
            roots: self.roots.clone(),
            solver: self.solver.clone(),
            options: self.options,
        }
    }
}
impl<'a, T: NumericScalar, O: AdjointOperator<T> + ?Sized> PreparedOperatorRegression<'a, T, O> {
    /// Prepares a least-squares fit for a borrowed operator and row weights.
    ///
    /// `weights` is [`crate::Unweighted`], [`crate::Frequency`] (integer
    /// replication counts) or [`crate::Analytic`] (relative inverse variances).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operator has no rows or no columns, or the
    ///   weights have a different length than the operator has rows.
    /// - `InvalidInput` if a weight is negative or not finite, or a frequency
    ///   weight is not an integer.
    /// - `InvalidInput` if no weight is positive.
    /// - `ResourceLimit` if the weights exceed `options.max_bytes()`.
    pub fn new<'w>(
        operator: &'a O,
        weights: impl RegressionWeights<'w, T::Real>,
        options: OperatorRegressionOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let max_bytes = options.max_bytes;
        let rows = weights.rows();
        let n = operator.rows();
        if n == 0 || operator.cols() == 0 || !rows.matches(n) {
            return Err(invalid(
                scioxide_core::argument!("operator"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid operator regression design",
            ));
        }
        budget_bytes(n, core::mem::size_of::<T::Real>(), max_bytes)?;
        let mut roots = Vector::zeros(n)?;
        let mut positive = false;
        for (i, r) in roots.as_mut_slice().iter_mut().enumerate() {
            let w = rows.weight(i)?;
            *r = w.sqrt();
            positive |= w > T::Real::ZERO;
        }
        if !positive {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "at least one positive weight required",
            ));
        }
        let solver = match options.method {
            OperatorRegressionMethod::Lsqr => Solver::Lsqr(
                LsqrSolver::new(options.residual, options.iterations)
                    .with_normal_tolerance(options.normal),
            ),
            OperatorRegressionMethod::Lsmr => Solver::Lsmr(
                LsmrSolver::new(options.residual, options.iterations)
                    .with_normal_tolerance(options.normal),
            ),
        };
        Ok(Self {
            operator,
            roots,
            solver,
            options,
        })
    }
    /// Returns the solve identity of [`Self::fit_into`]. The Krylov solve runs
    /// in a child scope of it.
    ///
    /// # Errors
    ///
    /// Does not fail.
    fn identity(&self, problem: ProblemToken) -> Result<SolveIdentity, SciOxideError> {
        let algorithm = match &self.solver {
            Solver::Lsqr(_) => scioxide_core::algorithm_id!("stats", "operator-regression-lsqr", 1),
            Solver::Lsmr(_) => scioxide_core::algorithm_id!("stats", "operator-regression-lsmr", 1),
        };
        Ok(SolveIdentity::new(
            problem,
            algorithm,
            ((self.rows() as u128) << 64) | self.cols() as u128,
            0,
        ))
    }
    /// Allocates the workspace for [`Self::fit_into`].
    ///
    /// Its size is O(n + p + s), where `s` is the scratch length the operator
    /// requests for its products.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the plan's byte limit.
    /// - The error of the operator's scratch-length query.
    pub fn workspace(&self) -> Result<OperatorRegressionWorkspace<T>, SciOxideError> {
        let max_bytes = self.options.max_bytes;
        let scratch = self
            .operator
            .apply_workspace_len()?
            .max(self.operator.apply_adjoint_workspace_len()?);
        let size = self
            .rows()
            .checked_add(self.cols())
            .and_then(|v| v.checked_add(scratch))
            .and_then(|v| v.checked_mul(64))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(size, core::mem::size_of::<T>(), max_bytes)?;
        let request = IterativeWorkspaceRequest::adjoint(self)?;
        let engine = match &self.solver {
            Solver::Lsqr(s) => s.workspace(&request)?,
            Solver::Lsmr(s) => s.workspace(&request)?,
        };
        Ok(OperatorRegressionWorkspace {
            engine,
            rhs: Vector::zeros(self.rows())?,
            stage: LinearRegressionFit::reserve(self.rows(), self.cols())?,
            provider: Vector::zeros(scratch)?,
            normal: Vector::zeros(self.cols())?,
            slot: Default::default(),
            retained_bytes: size.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "operator_regression_workspace_bytes"
                ))
            })?,
        })
    }
    /// Fits one response and writes the result to `output`.
    ///
    /// The Krylov solve starts from `β = 0` and stops at the tolerances or the
    /// iteration limit. The method then applies `X` once to compute the fitted
    /// values and residuals, and `Xᴴ` once to compute `‖Xᴴ W r‖₂`. These two
    /// products count toward the evaluation limit of `scope`. A solve that
    /// reaches the iteration limit still writes its last iterate. If execution
    /// is stopped or the evaluation limit is exhausted, `output` is unchanged
    /// and `output_committed` in the report is `false`. See
    /// [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `response`, `output` or `work` does not match the
    ///   operator shape.
    /// - `InvalidInput` if `response` contains a NaN or infinite value.
    /// - `ResourceLimit` if `work` exceeds the plan's byte limit.
    /// - `NumericalInstability` if a residual overflows.
    /// - The error of the operator or of the Krylov solver.
    pub fn fit_into<'scope, 'capability>(
        &self,
        response: &[T],
        output: &mut LinearRegressionFit<T>,
        work: &mut OperatorRegressionWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<OperatorRegressionReport<T::Real>, SciOxideError> {
        work.slot.reject();
        crate::error_helpers::limit_bytes(work.retained_bytes, self.options.max_bytes)?;
        let required_provider = self
            .operator
            .apply_workspace_len()?
            .max(self.operator.apply_adjoint_workspace_len()?);
        if work.provider.len() < required_provider {
            return Err(invalid(
                scioxide_core::argument!("work"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "operator provider workspace too small",
            ));
        }
        if response.len() != self.rows()
            || response.iter().any(|v| !v.has_finite_components())
            || output.coefficients.len() != self.cols()
            || output.fitted.len() != self.rows()
            || output.residuals.len() != self.rows()
            || output.weighted_residuals.len() != self.rows()
            || work.rhs.len() != self.rows()
            || work.normal.len() != self.cols()
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "operator regression response, output or workspace",
            ));
        }
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope.into(),
            self.identity(ProblemToken::new(0))?,
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Iteration,
                self.options.iterations,
            ),
            |context| self.fit_attempt(response, output, work, context),
            |result| match result {
                Ok(report) => (
                    if matches!(
                        report.termination,
                        LinearSolveTermination::Converged
                            | LinearSolveTermination::LeastSquaresConverged
                    ) {
                        scioxide_core::AttemptOutcome::Succeeded
                    } else {
                        scioxide_core::AttemptOutcome::Stopped
                    },
                    report.output_committed.then_some(*report),
                ),
                Err(_) => (scioxide_core::AttemptOutcome::Failed, None),
            },
        );
        work.slot = slot;
        result
    }
    /// Allocates an output for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds the plan's byte limit.
    pub fn output(&self) -> Result<LinearRegressionFit<T>, SciOxideError> {
        crate::error_helpers::limit_bytes(
            LinearRegressionFit::<T>::bytes(self.rows(), self.cols())?,
            self.options.max_bytes,
        )?;
        LinearRegressionFit::reserve(self.rows(), self.cols())
    }
    /// Fits one response sequentially and returns a new result.
    ///
    /// Allocates the workspace and the output. Use [`Self::fit_into`] to reuse
    /// storage.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`Self::output`] and
    /// [`Self::fit_into`].
    pub fn fit(&self, response: &[T]) -> Result<OperatorRegressionResult<T>, SciOxideError> {
        let mut output = self.output()?;
        let mut work = self.workspace()?;
        let report = self.fit_into(
            response,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(OperatorRegressionResult {
            fit: report.output_committed.then_some(output),
            report,
        })
    }
    fn fit_attempt(
        &self,
        response: &[T],
        output: &mut LinearRegressionFit<T>,
        work: &mut OperatorRegressionWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<OperatorRegressionReport<T::Real>, SciOxideError> {
        let (n, p) = (self.rows(), self.cols());
        if work.provider.len()
            < self
                .operator
                .apply_workspace_len()?
                .max(self.operator.apply_adjoint_workspace_len()?)
        {
            return Err(invalid(
                scioxide_core::argument!("work"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "operator provider workspace too small",
            ));
        }
        if response.len() != n
            || work.rhs.len() != n
            || work.stage.coefficients.len() != p
            || output.coefficients.len() != p
            || output.fitted.len() != n
            || output.residuals.len() != n
            || output.weighted_residuals.len() != n
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "operator fit shape mismatch",
            ));
        }
        for (i, &y) in response.iter().enumerate() {
            if !y.has_finite_components() {
                return Err(invalid(
                    scioxide_core::argument!("response"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "non-finite operator response",
                ));
            }
            work.rhs
                .view_mut()
                .set(i, y * T::from_real(self.roots.view().get(i)?))?;
        }
        work.stage.coefficients.as_mut_slice().fill(T::ZERO);
        let report = match &self.solver {
            Solver::Lsqr(s) => s.solve_into(
                self,
                &work.rhs,
                work.stage.coefficients.view_mut(),
                &mut work.engine,
                &mut *context,
            )?,
            Solver::Lsmr(s) => s.solve_into(
                self,
                &work.rhs,
                work.stage.coefficients.view_mut(),
                &mut work.engine,
                &mut *context,
            )?,
        };
        let mut result = OperatorRegressionReport {
            termination: report.termination(),
            iterations: report.iterations(),
            krylov: report.krylov(),
            output_committed: false,
        };
        if let Some(stop) = context.stop_reason()? {
            result.termination = LinearSolveTermination::ExecutionStopped(stop);
            return Ok(result);
        }
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            result.termination = LinearSolveTermination::EvaluationLimit;
            return Ok(result);
        }
        let forward = self.operator.apply_workspace_len()?;
        self.operator
            .apply_into_scoped(
                work.stage.coefficients.as_slice(),
                work.stage.fitted.as_mut_slice(),
                &mut work.provider.as_mut_slice()[..forward],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        for (i, &y) in response.iter().enumerate() {
            let r = y - work.stage.fitted.view().get(i)?;
            crate::error_helpers::finite(r)?;
            work.stage.residuals.view_mut().set(i, r)?;
            work.stage
                .weighted_residuals
                .view_mut()
                .set(i, r * T::from_real(self.roots.view().get(i)?))?;
            work.rhs.view_mut().set(
                i,
                work.stage.weighted_residuals.view().get(i)?
                    * T::from_real(self.roots.view().get(i)?),
            )?;
        }
        if !context
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            result.termination = LinearSolveTermination::EvaluationLimit;
            return Ok(result);
        }
        let adjoint = self.operator.apply_adjoint_workspace_len()?;
        self.operator
            .apply_adjoint_into_scoped(
                work.rhs.as_slice(),
                work.normal.as_mut_slice(),
                &mut work.provider.as_mut_slice()[..adjoint],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        work.stage.residual_norm = work.stage.residuals.view().norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        work.stage.weighted_residual_norm = work.stage.weighted_residuals.view().norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        work.stage.normal_residual_norm = work.normal.view().norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        work.stage.rank = None;
        work.stage.degrees_of_freedom = None;
        output.copy_from(&work.stage)?;
        result.output_committed = true;
        Ok(result)
    }
}
impl<T: NumericScalar, O: AdjointOperator<T> + ?Sized> PreparedOperatorRegression<'_, T, O> {
    fn apply_forward(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        if input.len() != self.cols()
            || output.len() != self.rows()
            || workspace.len() != self.apply_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "weighted operator shape",
            ));
        }
        let outcome = match context {
            Some(context) => self.operator.apply_into_scoped(
                input,
                output,
                workspace,
                &mut scioxide_core::OperatorContext::new(context),
            ),
            None => self
                .operator
                .apply_into(input, output, workspace)
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        for (v, &w) in output.iter_mut().zip(self.roots.as_slice()) {
            *v = *v * T::from_real(w);
            crate::error_helpers::finite(*v)?;
        }
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
    fn apply_adjoint(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        if input.len() != self.rows()
            || output.len() != self.cols()
            || workspace.len() != self.apply_adjoint_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "weighted adjoint shape",
            ));
        }
        let (scaled, scratch) = workspace.split_at_mut(self.rows());
        for ((v, &x), &w) in scaled.iter_mut().zip(input).zip(self.roots.as_slice()) {
            *v = x * T::from_real(w);
            if !v.has_finite_components() {
                return Err(invalid(
                    scioxide_core::argument!("input"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "non-finite weighted adjoint input",
                ));
            }
        }
        match context {
            Some(context) => self.operator.apply_adjoint_into_scoped(
                scaled,
                output,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            ),
            None => self
                .operator
                .apply_adjoint_into(scaled, output, scratch)
                .map(scioxide_core::Invocation::Evaluated),
        }
    }
}
impl<T: NumericScalar, O: AdjointOperator<T> + ?Sized> LinearOperator<T>
    for PreparedOperatorRegression<'_, T, O>
{
    fn rows(&self) -> usize {
        self.operator.rows()
    }
    fn cols(&self) -> usize {
        self.operator.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.operator.apply_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply_forward(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_forward(input, output, workspace, Some(context))
    }
}
impl<T: NumericScalar, O: AdjointOperator<T> + ?Sized> AdjointOperator<T>
    for PreparedOperatorRegression<'_, T, O>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.rows()
            .checked_add(self.operator.apply_adjoint_workspace_len()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
    }
    fn apply_adjoint_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply_adjoint(input, output, workspace, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_adjoint(input, output, workspace, Some(context))
    }
}

/// Holds the fit and the report of [`PreparedOperatorRegression::fit`].
pub struct OperatorRegressionResult<T: NumericScalar> {
    fit: Option<LinearRegressionFit<T>>,
    report: OperatorRegressionReport<T::Real>,
}
impl<T: NumericScalar> OperatorRegressionResult<T> {
    /// Returns the fit, or `None` when it was not written (see
    /// `OperatorRegressionReport::output_committed`).
    pub fn fit(&self) -> Option<&LinearRegressionFit<T>> {
        self.fit.as_ref()
    }
    /// Returns the termination status and iteration count.
    pub fn report(&self) -> &OperatorRegressionReport<T::Real> {
        &self.report
    }
}
impl<T: NumericScalar> OperatorRegressionWorkspace<T> {
    /// Returns the record of the last fit run with this workspace.
    pub fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<OperatorRegressionReport<T::Real>>> {
        self.slot.last_attempt()
    }
    /// Returns the report of the last fit, if it wrote its output.
    pub fn last_accepted(&self) -> Option<&OperatorRegressionReport<T::Real>> {
        self.slot
            .last_attempt()
            .and_then(scioxide_core::AttemptRecord::progress)
    }
}
