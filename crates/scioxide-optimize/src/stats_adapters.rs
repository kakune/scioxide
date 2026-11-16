//! Optimizers for the model-fitting interface of `scioxide-stats`.
use crate::{
    BoundedObjective, ConstrainedProblem, Constraints, LbfgsB, LbfgsBWorkspace, LeastSquares,
    LeastSquaresProblem, LeastSquaresResult, LeastSquaresWorkspace, LeastSquaresWorkspaceRequest,
    ObjectiveFunction, OptimizeResult, OptimizeTermination, Sqp, SqpWorkspace,
};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::{
    Budget, NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind, SolveContext,
};
use scioxide_linalg::Matrix;
use scioxide_stats::{FitRequest, FitSolver, FitSolverCapabilities, StatsTermination};

use crate::validation::invalid;
// Iteration allowances are scope-local in core. An adapter caps the configured
// iteration limit by the allowance of the context it is given;
// `None` means that allowance is exhausted. Provider budgets stay shared.
pub(crate) fn iteration_limit(
    context: &SolveContext<'_, '_>,
    configured: Budget,
) -> Result<Option<Budget>, SciOxideError> {
    Ok(
        match context
            .budget_report(scioxide_core::BudgetKind::Iteration)
            .remaining()
        {
            None => Some(configured),
            Some(0) => None,
            Some(remaining) => Some(Budget::try_from(remaining)?.min(configured)),
        },
    )
}
fn resource_failure(error: SciOxideError) -> Result<StatsTermination, SciOxideError> {
    match error.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. }
            if diagnostics
                .budget()
                .is_some_and(|b| b.remaining() == Some(0)) =>
        {
            Ok(StatsTermination::BudgetExhausted)
        }
        scioxide_core::SciOxideErrorView::ExecutionStopped { reason: stop } => {
            Ok(StatsTermination::Stopped(stop))
        }
        _ => Err(error),
    }
}
fn preflight(context: &SolveContext<'_, '_>) -> Result<OptimizeTermination, SciOxideError> {
    if let Some(stop) = context.stop_reason()? {
        return Ok(match stop {
            scioxide_core::ExecutionStopReason::DeadlineReached => OptimizeTermination::TimeLimit,
            _ => OptimizeTermination::Cancelled,
        });
    }
    if context
        .effective_budget_report(scioxide_core::BudgetKind::Evaluation)
        .exhausted()
    {
        return Ok(OptimizeTermination::EvaluationLimit);
    }
    Ok(OptimizeTermination::NotStarted)
}
fn termination(t: OptimizeTermination) -> StatsTermination {
    match t {
        OptimizeTermination::Converged | OptimizeTermination::GradientSmall => {
            StatsTermination::Converged
        }
        OptimizeTermination::IterationLimit
        | OptimizeTermination::EvaluationLimit
        | OptimizeTermination::MemoryLimit
        | OptimizeTermination::NodeLimit
        | OptimizeTermination::DepthLimit
        | OptimizeTermination::ProposalLimit
        | OptimizeTermination::SampleLimit => StatsTermination::BudgetExhausted,
        OptimizeTermination::Cancelled => {
            StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
        }
        OptimizeTermination::TimeLimit => {
            StatsTermination::Stopped(scioxide_core::ExecutionStopReason::DeadlineReached)
        }
        _ => StatsTermination::NotConverged,
    }
}
fn validate<S: RealScalar, W: Send>(
    r: FitRequest<'_, S, W>,
    candidate: &[S],
    n: usize,
    bounded: bool,
) -> Result<(), SciOxideError> {
    if n == 0
        || r.objective.dimension() != n
        || r.initial.len() != n
        || candidate.len() != n
        || r.bounds.len() != n
        || r.integrality.len() != n
    {
        return Err(invalid(
            scioxide_core::argument!("request"),
            "request and prepared dimensions differ",
        ));
    }
    if r.integrality.contains(&true)
        || (!bounded
            && r.bounds
                .as_slice()
                .iter()
                .any(|(a, b)| a.is_some() || b.is_some()))
    {
        return Err(SciOxideError::not_implemented(
            "unsupported fitting constraints",
        ));
    }
    if r.initial
        .iter()
        .zip(r.bounds.as_slice())
        .any(|(&x, &(a, b))| !x.is_finite() || a.is_some_and(|a| x < a) || b.is_some_and(|b| x > b))
    {
        return Err(invalid(
            scioxide_core::argument!("request.initial"),
            "initial parameters must be finite and feasible",
        ));
    }
    Ok(())
}
struct Objective<'a, S: RealScalar, W: Send> {
    request: FitRequest<'a, S, W>,
    workspace: RefCell<&'a mut W>,
}
impl<S: RealScalar + NumericScalar<Real = S>, W: Send> ObjectiveFunction<S>
    for Objective<'_, S, W>
{
    fn dim(&self) -> usize {
        self.request.objective.dimension()
    }
    fn is_in_domain(&self, x: &[S]) -> Result<bool, SciOxideError> {
        let mut w = self.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("workspace"),
                "reentrant objective workspace borrow",
            )
        })?;
        match self.request.objective.validate(x, &mut w) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == SciOxideErrorKind::InvalidInput => Ok(false),
            Err(e) => Err(e),
        }
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        let mut w = self.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("workspace"),
                "reentrant objective workspace borrow",
            )
        })?;
        self.request.objective.value(x, &mut w)
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        if let Some(d) = self
            .request
            .derivatives
            .filter(|d| d.capabilities().gradient)
        {
            d.gradient(x, out)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
struct Residuals<'a, S: RealScalar, W: Send> {
    objective: Objective<'a, S, W>,
    rows: usize,
    jacobian: RefCell<&'a mut [S]>,
}
impl<S: RealScalar + NumericScalar<Real = S>, W: Send> LeastSquaresProblem<S>
    for Residuals<'_, S, W>
{
    fn param_dim(&self) -> usize {
        self.objective.request.objective.dimension()
    }
    fn residual_dim(&self) -> usize {
        self.rows
    }
    fn residual_into(&self, x: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        let mut w = self.objective.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("workspace"),
                "reentrant objective workspace borrow",
            )
        })?;
        self.objective.request.objective.residuals(x, out, &mut w)
    }
    fn jacobian_into(&self, x: &[S], out: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        let Some(d) = self
            .objective
            .request
            .derivatives
            .filter(|d| d.capabilities().jacobian)
        else {
            return Ok(false);
        };
        let mut j = self.jacobian.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("workspace"),
                "reentrant Jacobian borrow",
            )
        })?;
        d.jacobian(x, &mut j)?;
        for (i, row) in j.chunks_exact(self.param_dim()).enumerate() {
            for (k, &v) in row.iter().enumerate() {
                out.view_mut().set(i, k, v)?;
            }
        }
        Ok(true)
    }
}

/// Fits a statistical model by nonlinear least squares, as a
/// [`FitSolver`](scioxide_stats::FitSolver) for `scioxide-stats`.
///
/// Each fit minimizes `½ Σᵢ rᵢ(θ)²` over the model residuals `rᵢ` with the
/// [`LeastSquares`] solver it wraps, using a dense Jacobian. It does not
/// support bounds, integer parameters or Jacobian operators. Storage is
/// allocated by [`Self::new`] for one parameter and residual count. The
/// iteration limit is the solver's own limit, lowered to the iteration
/// allowance of the solve context passed to
/// [`FitSolver::solve`](scioxide_stats::FitSolver::solve) when that context has
/// one. Iteration limits of enclosing solves do not apply, and
/// [`fit_into`](scioxide_stats::fit_into) runs the fit without one.
pub struct LeastSquaresFitSolver<S: RealScalar + NumericScalar<Real = S>> {
    solver: LeastSquares<S>,
    workspace: LeastSquaresWorkspace<S>,
    result: LeastSquaresResult<S>,
    jacobian: Vec<S>,
    parameters: usize,
    residuals: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> LeastSquaresFitSolver<S> {
    /// Creates the adapter for `parameters` parameters and `residuals`
    /// residuals. Tolerances and globalization come from the options of
    /// `solver`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `parameters` or `residuals` is zero, or `solver`
    ///   uses a loss other than [`LeastSquaresLoss::Quadratic`](crate::LeastSquaresLoss::Quadratic)
    ///   or the LSMR linear solver.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(
        solver: LeastSquares<S>,
        parameters: usize,
        residuals: usize,
    ) -> Result<Self, SciOxideError> {
        if parameters == 0 || residuals == 0 || !solver.supports_statistical_residuals() {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                "empty least-squares shape or incompatible loss/operator policy",
            ));
        }
        let len = parameters.checked_mul(residuals).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("fit_jacobian"))
        })?;
        let workspace =
            solver.workspace(LeastSquaresWorkspaceRequest::dense(parameters, residuals))?;
        let mut result = LeastSquaresResult::empty();
        result.try_reserve(parameters, residuals)?;
        Ok(Self {
            solver,
            workspace,
            result,
            jacobian: scioxide_core::backend::try_filled(
                len,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("fit_jacobian"),
            )?,
            parameters,
            residuals,
        })
    }
    /// Returns the result of the last fit. Its cost is `½ Σᵢ rᵢ²`.
    pub fn result(&self) -> &LeastSquaresResult<S> {
        &self.result
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, W: Send> FitSolver<S, W>
    for LeastSquaresFitSolver<S>
{
    fn capabilities(&self) -> FitSolverCapabilities {
        FitSolverCapabilities {
            requires_residuals: true,
            ..FitSolverCapabilities::default()
        }
    }
    fn solve(
        &mut self,
        request: FitRequest<'_, S, W>,
        candidate: &mut [S],
        workspace: &mut W,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        validate(request, candidate, self.parameters, false)?;
        if request.objective.residual_count() != Some(self.residuals) {
            return Err(invalid(
                scioxide_core::argument!("request"),
                "prepared residual count differs",
            ));
        }
        let p = Residuals {
            objective: Objective {
                request,
                workspace: RefCell::new(workspace),
            },
            rows: self.residuals,
            jacobian: RefCell::new(&mut self.jacobian),
        };
        self.result.reset_for_fit();
        self.result.termination = preflight(context)?;
        if self.result.termination != OptimizeTermination::NotStarted {
            return Ok(termination(self.result.termination));
        }
        let options = *self.solver.options();
        let Some(iterations) = iteration_limit(context, options.max_iterations())? else {
            self.result.termination = OptimizeTermination::IterationLimit;
            return Ok(StatsTermination::BudgetExhausted);
        };
        let solver = LeastSquares::new(options.with_max_iterations(iterations))?;
        let outcome = solver.solve_into(
            &p,
            request.initial,
            &mut self.result,
            &mut self.workspace,
            &mut *context,
        );
        if let Err(error) = outcome {
            return resource_failure(error);
        }
        if self.result.x().len() == candidate.len() {
            candidate.copy_from_slice(self.result.x());
        }
        Ok(termination(self.result.termination()))
    }
}

/// Fits a statistical model by minimizing its objective with
/// [`LbfgsB`], as a [`FitSolver`](scioxide_stats::FitSolver) for `scioxide-stats`.
///
/// It supports finite, one-sided and fixed bounds on the parameters, but not
/// integer parameters. An analytic gradient is optional; without one, the
/// derivative strategy of the solver's options selects the approximation. The
/// iteration limit is the solver's own limit, lowered to the iteration
/// allowance of the solve context passed to
/// [`FitSolver::solve`](scioxide_stats::FitSolver::solve) when that context has
/// one. Iteration limits of enclosing solves do not apply, and
/// [`fit_into`](scioxide_stats::fit_into) runs the fit without one.
pub struct LbfgsBFitSolver<S: RealScalar + NumericScalar<Real = S>> {
    solver: LbfgsB<S>,
    workspace: LbfgsBWorkspace<S>,
    result: OptimizeResult<S>,
    dimension: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> LbfgsBFitSolver<S> {
    /// Creates the adapter for `dimension` parameters.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(solver: LbfgsB<S>, dimension: usize) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "empty fitting dimension",
            ));
        }
        let workspace = solver.workspace(dimension)?;
        let mut result = OptimizeResult::empty();
        result.try_reserve(dimension)?;
        Ok(Self {
            solver,
            workspace,
            result,
            dimension,
        })
    }
    /// Returns the result of the last fit.
    pub fn result(&self) -> &OptimizeResult<S> {
        &self.result
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, W: Send> FitSolver<S, W> for LbfgsBFitSolver<S> {
    fn capabilities(&self) -> FitSolverCapabilities {
        FitSolverCapabilities {
            bounds: true,
            ..FitSolverCapabilities::default()
        }
    }
    fn solve(
        &mut self,
        request: FitRequest<'_, S, W>,
        candidate: &mut [S],
        workspace: &mut W,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        validate(request, candidate, self.dimension, true)?;
        let f = Objective {
            request,
            workspace: RefCell::new(workspace),
        };
        let p = BoundedObjective::new(&f, request.bounds)?;
        self.result.reset_for_fit();
        self.result.termination = preflight(context)?;
        if self.result.termination != OptimizeTermination::NotStarted {
            return Ok(termination(self.result.termination));
        }
        let options = *self.solver.options();
        let Some(iterations) = iteration_limit(context, options.max_iterations())? else {
            self.result.termination = OptimizeTermination::IterationLimit;
            return Ok(StatsTermination::BudgetExhausted);
        };
        let outcome = LbfgsB::new(options.with_max_iterations(iterations)).minimize_into(
            &p,
            request.initial,
            &mut self.result,
            &mut self.workspace,
            &mut *context,
        );
        if let Err(error) = outcome {
            return resource_failure(error);
        }
        if self.result.x().len() == candidate.len() {
            candidate.copy_from_slice(self.result.x());
        }
        Ok(termination(self.result.termination()))
    }
}

/// Fits a statistical model subject to nonlinear constraints with [`Sqp`], as a
/// [`FitSolver`](scioxide_stats::FitSolver) for `scioxide-stats`.
///
/// Each fit minimizes the model objective subject to the [`Constraints`] given
/// to [`Self::new`], as a [`ConstrainedProblem`]. The bounds of the fit
/// request must equal the bounds of the constraints. Integer parameters are
/// not supported. The
/// iteration limit is the solver's own limit, lowered to the iteration
/// allowance of the solve context passed to
/// [`FitSolver::solve`](scioxide_stats::FitSolver::solve) when that context has
/// one. Iteration limits of enclosing solves do not apply, and
/// [`fit_into`](scioxide_stats::fit_into) runs the fit without one.
pub struct SqpFitSolver<S: RealScalar + NumericScalar<Real = S>> {
    solver: Sqp<S>,
    constraints: Constraints<S>,
    workspace: SqpWorkspace<S>,
    result: OptimizeResult<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> SqpFitSolver<S> {
    /// Creates the adapter. The dimension of `constraints` is the number of
    /// parameters.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn new(solver: Sqp<S>, constraints: Constraints<S>) -> Result<Self, SciOxideError> {
        let workspace = solver.workspace_for(&constraints)?;
        let mut result = OptimizeResult::empty();
        result.try_reserve(constraints.dimension())?;
        Ok(Self {
            solver,
            constraints,
            workspace,
            result,
        })
    }
    /// Returns the result of the last fit, including KKT and constraint
    /// residuals.
    pub fn result(&self) -> &OptimizeResult<S> {
        &self.result
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, W: Send> FitSolver<S, W> for SqpFitSolver<S> {
    fn capabilities(&self) -> FitSolverCapabilities {
        FitSolverCapabilities {
            bounds: true,
            ..FitSolverCapabilities::default()
        }
    }
    fn solve(
        &mut self,
        request: FitRequest<'_, S, W>,
        candidate: &mut [S],
        workspace: &mut W,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        validate(request, candidate, self.constraints.dimension(), true)?;
        if request.bounds.as_slice() != self.constraints.hard_box() {
            return Err(invalid(
                scioxide_core::argument!("request.bounds"),
                "request bounds differ from the constraints' box",
            ));
        }
        let f = Objective {
            request,
            workspace: RefCell::new(workspace),
        };
        let problem = ConstrainedProblem::new(&f, &self.constraints)?;
        self.result.reset_for_fit();
        self.result.termination = preflight(context)?;
        if self.result.termination != OptimizeTermination::NotStarted {
            return Ok(termination(self.result.termination));
        }
        let options = *self.solver.options();
        let Some(iterations) = iteration_limit(context, options.max_iterations())? else {
            self.result.termination = OptimizeTermination::IterationLimit;
            return Ok(StatsTermination::BudgetExhausted);
        };
        let outcome = Sqp::new(options.with_max_iterations(iterations)).minimize_into(
            &problem,
            request.initial,
            &mut self.result,
            &mut self.workspace,
            &mut *context,
        );
        if let Err(error) = outcome {
            return resource_failure(error);
        }
        if self.result.x().len() == candidate.len() {
            candidate.copy_from_slice(self.result.x());
        }
        Ok(termination(self.result.termination()))
    }
}
