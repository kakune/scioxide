use super::problem::validate_parameters;
use super::{FitObjective, finite, invalid};
use crate::StatsTermination;
use alloc::vec::Vec;

use scioxide_core::{Bounds, EvaluationRole, RealScalar, SciOxideError, SolveContext};

/// Derivatives of a [`FitObjective`], passed to the solver with it.
///
/// The derivatives must be those of the same objective, with the same data and
/// normalization. This crate does not approximate missing derivatives by
/// finite differences. Each method's default returns `NotImplemented`.
pub trait FitDerivatives<S: RealScalar>: Sync {
    /// Returns which derivatives this provider computes.
    fn capabilities(&self) -> FitDerivativeCapabilities;
    /// Writes the gradient of the objective at the parameters into the output.
    ///
    /// # Errors
    ///
    /// `NotImplemented` if not provided, or an error if the shapes disagree or
    /// the evaluation fails.
    fn gradient(&self, _parameters: &[S], _output: &mut [S]) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented("fitting gradient"))
    }
    /// Writes the `d × d` Hessian of the objective at the parameters, row-major,
    /// into the output.
    ///
    /// # Errors
    ///
    /// `NotImplemented` if not provided, or an error if the shapes disagree or
    /// the evaluation fails.
    fn hessian(&self, _parameters: &[S], _output: &mut [S]) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented("fitting Hessian"))
    }
    /// Writes the Jacobian of the residuals at the parameters, with one row per
    /// residual and one column per parameter, row-major, into the output.
    ///
    /// # Errors
    ///
    /// `NotImplemented` if not provided, or an error if the shapes disagree or
    /// the evaluation fails.
    fn jacobian(&self, _parameters: &[S], _output: &mut [S]) -> Result<(), SciOxideError> {
        Err(SciOxideError::not_implemented("fitting Jacobian"))
    }
}
/// The derivatives that a [`FitDerivatives`] provider computes, or that a
/// [`FitSolver`] requires.
#[derive(Clone, Copy, Debug, Default)]
pub struct FitDerivativeCapabilities {
    /// Gradient of the objective.
    pub gradient: bool,
    /// Hessian of the objective.
    pub hessian: bool,
    /// Jacobian of the residuals.
    pub jacobian: bool,
}
impl<S: RealScalar> FitDerivatives<S> for super::ExactFitDerivatives<'_, S> {
    fn capabilities(&self) -> FitDerivativeCapabilities {
        let regular = !matches!(self.model, super::DistributionFitModel::Uniform);
        FitDerivativeCapabilities {
            gradient: regular,
            hessian: regular,
            jacobian: false,
        }
    }
    fn gradient(&self, p: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        let mut h = [S::ZERO; 4];
        let n = p
            .len()
            .checked_mul(p.len())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("fit_hessian")))?;
        if n > 4 {
            return Err(invalid(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "built-in derivative dimensions",
            ));
        }
        self.evaluate(p, out, &mut h[..n])
    }
    fn hessian(&self, p: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        let mut g = [S::ZERO; 2];
        if p.len() > 2 {
            return Err(invalid(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "built-in derivative dimensions",
            ));
        }
        self.evaluate(p, &mut g[..p.len()], out)
    }
}
/// What a [`FitSolver`] supports and requires.
///
/// [`fit_into`] checks these against the request before it calls the solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct FitSolverCapabilities {
    /// The solver respects parameter bounds, including fixed parameters.
    pub bounds: bool,
    /// The solver respects integer constraints.
    pub integrality: bool,
    /// The solver needs residuals, as in nonlinear least squares.
    pub requires_residuals: bool,
    /// The derivatives the solver needs.
    pub derivatives: FitDerivativeCapabilities,
}
/// A minimization problem passed to a [`FitSolver`]: objective, optional
/// derivatives, bounds, integer constraints and starting point.
///
/// The solver counts each call of the objective or its derivatives in the
/// solve context, with its evaluation role, and records each iteration.
pub struct FitRequest<'a, S: RealScalar, W: Send = ()> {
    /// The objective to minimize.
    pub objective: &'a dyn FitObjective<S, Workspace = W>,
    /// Derivatives of the objective, if available.
    pub derivatives: Option<&'a dyn FitDerivatives<S>>,
    /// Parameter bounds. Equal lower and upper bounds fix a parameter.
    pub bounds: &'a Bounds<S>,
    /// Flags that mark integer parameters, one per parameter.
    pub integrality: &'a [bool],
    /// Starting point: finite, within the bounds and integer where required.
    pub initial: &'a [S],
}
impl<S: RealScalar, W: Send> Copy for FitRequest<'_, S, W> {}
impl<S: RealScalar, W: Send> Clone for FitRequest<'_, S, W> {
    fn clone(&self) -> Self {
        *self
    }
}
/// An optimizer that minimizes a [`FitObjective`].
///
/// This crate contains no optimizer of its own. Implement this trait to fit with
/// any minimizer, for example one from `scioxide-optimize`. The implementation
/// keeps its own scratch storage.
pub trait FitSolver<S: RealScalar, W: Send = ()> {
    /// Returns the constraints the solver supports and the derivatives it needs.
    fn capabilities(&self) -> FitSolverCapabilities;
    /// Minimizes the objective, starting from `candidate`, which holds the
    /// request's starting point, and leaves the result in `candidate`.
    ///
    /// The solver checks stop requests and shared evaluation budgets before each
    /// callback. Its own configured iteration limit bounds this child epoch;
    /// an ancestor iteration limit does not cap it. It returns `Converged`, `BudgetExhausted`,
    /// `Stopped` or `NotConverged`. [`fit_into`] rejects any other status with
    /// `InvalidInput`.
    ///
    /// # Errors
    ///
    /// The errors of the objective, of its derivatives, and of the solver itself.
    fn solve(
        &mut self,
        request: FitRequest<'_, S, W>,
        candidate: &mut [S],
        objective_workspace: &mut W,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError>;
}
/// Scratch storage for [`fit_into`]: one parameter vector and the objective's
/// workspace.
///
/// The solver's own scratch is not included.
pub struct FitWorkspace<S: RealScalar, W: Send = ()> {
    solve: crate::execution::SolveSlot<()>,
    candidate: Vec<S>,
    pub(crate) objective: W,
}
impl<S: RealScalar> FitWorkspace<S> {
    /// Creates a workspace for an objective with `dimension` parameters and no
    /// workspace of its own.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    /// - `ResourceLimit` if the vector exceeds
    ///   [`StatisticsOptions::max_bytes`](crate::StatisticsOptions::max_bytes).
    pub fn new(dimension: usize, options: crate::StatisticsOptions) -> Result<Self, SciOxideError> {
        let bytes = dimension
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("fit_candidate"))
            })?;
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "candidate dimension must be positive",
            ));
        }
        crate::error_helpers::limit(
            bytes,
            options.max_bytes(),
            scioxide_core::quantity!("fit_candidate_bytes"),
        )?;
        Ok(Self {
            solve: Default::default(),
            objective: (),
            candidate: scioxide_core::backend::try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("fit_candidate"),
            )?,
        })
    }
}
impl<S: RealScalar, W: Send> FitWorkspace<S, W> {
    /// Returns diagnostics of the last attempted fit.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.solve.last_attempt()
    }
    /// Creates a workspace for `objective`, including the objective's own workspace.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the objective has no parameters.
    /// - `ResourceLimit` if the vector and the objective's workspace together
    ///   exceed [`StatisticsOptions::max_bytes`](crate::StatisticsOptions::max_bytes).
    /// - The errors of [`FitObjective::workspace`].
    pub fn for_objective(
        objective: &dyn FitObjective<S, Workspace = W>,
        options: crate::StatisticsOptions,
    ) -> Result<Self, SciOxideError> {
        let n = objective.dimension();
        let bytes = n.checked_mul(core::mem::size_of::<S>()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("fit_candidate"))
        })?;
        if n == 0 {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "objective dimension must be positive",
            ));
        }
        let required = scioxide_core::backend::checked_add(
            bytes,
            objective.workspace_bytes()?,
            scioxide_core::quantity!("fit_workspace_bytes"),
        )?;
        crate::error_helpers::limit(
            required,
            options.max_bytes(),
            scioxide_core::quantity!("fit_workspace_bytes"),
        )?;
        Ok(Self {
            solve: Default::default(),
            candidate: scioxide_core::backend::try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("fit_candidate"),
            )?,
            objective: objective.workspace()?,
        })
    }
}
/// Report of a fit: termination status, final objective value, active bounds
/// and evaluation counts.
///
/// If the fit stopped before the final objective evaluation, [`Self::objective`]
/// is `None` and no parameters were written.
#[derive(Clone, Copy, Debug)]
pub struct FitReport<S: RealScalar> {
    termination: StatsTermination,
    objective: Option<S>,
    active_bounds: usize,
    counts: scioxide_core::SolveCounts,
}
impl<S: RealScalar> FitReport<S> {
    /// Returns the termination status.
    pub const fn termination(&self) -> StatsTermination {
        self.termination
    }
    /// Returns the objective at the fitted parameters, evaluated again after the
    /// solver returned, or `None` if the fit stopped before that evaluation.
    pub const fn objective(&self) -> Option<S> {
        self.objective
    }
    /// Returns the number of fitted parameters that lie exactly on a bound,
    /// including fixed parameters, or zero if no parameters were written.
    pub const fn active_bounds(&self) -> usize {
        self.active_bounds
    }
    /// Returns the iteration and evaluation counts of the solve context, including
    /// the final evaluation.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
}
/// Minimizes an objective with the given solver and writes the parameters to `output`.
///
/// The fit runs in four steps:
///
/// 1. Checks the shapes, the starting point, and that the solver supports the
///    request's bounds and integer constraints and gets the residuals and
///    derivatives it needs.
/// 2. Checks the starting point with [`FitObjective::validate`], counted as one
///    domain evaluation.
/// 3. Runs the solver.
/// 4. Checks the solver's result against the bounds and integer constraints
///    and evaluates the objective there, counted as one objective evaluation.
///
/// `output` receives the result only after step 4 succeeds. If the evaluation
/// budget or a stop request prevents step 2 or 4, the function returns `Ok`
/// with status `BudgetExhausted` or `Stopped`, no objective value, and `output`
/// unchanged. See [Results and termination](crate#results-and-termination).
///
/// # Errors
///
/// - `InvalidInput` if `request.initial`, `output` and `workspace` do not all
///   have the objective's dimension.
/// - An error if `request.initial` or the solver's result is not finite, lies
///   outside `request.bounds`, or is not an integer where required.
/// - `NotImplemented` if the request has bounds or integer constraints that the
///   solver does not support, or lacks residuals or derivatives that it needs.
/// - `InvalidInput` if the solver returns a status other than `Converged`,
///   `BudgetExhausted`, `Stopped` or `NotConverged`.
/// - The errors of the objective, its derivatives and the solver, and an error
///   if the final objective value is not finite.
pub fn fit_into<'scope, 'capability, S: RealScalar, W: Send>(
    request: FitRequest<'_, S, W>,
    solver: &mut dyn FitSolver<S, W>,
    output: &mut [S],
    workspace: &mut FitWorkspace<S, W>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<FitReport<S>, SciOxideError> {
    workspace.solve.reject();

    let d = request.objective.dimension();
    if d != request.initial.len() || d != output.len() || d != workspace.candidate.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "initial point, output and workspace dimensions differ from the objective dimension",
        ));
    }
    validate_parameters(request.initial, request.bounds, request.integrality)?;
    let caps = solver.capabilities();
    let dc = request
        .derivatives
        .map(|d| d.capabilities())
        .unwrap_or_default();
    if (!caps.bounds
        && request
            .bounds
            .as_slice()
            .iter()
            .any(|(a, b)| a.is_some() || b.is_some()))
        || (!caps.integrality && request.integrality.contains(&true))
        || (caps.requires_residuals && request.objective.residual_count().is_none())
        || (caps.derivatives.gradient && !dc.gradient)
        || (caps.derivatives.hessian && !dc.hessian)
        || (caps.derivatives.jacobian && !dc.jacobian)
    {
        return Err(SciOxideError::not_implemented(
            "incompatible fitting solver capabilities",
        ));
    }
    let mut slot = core::mem::take(&mut workspace.solve);
    let result = slot.run(
        scope.into(),
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "fit", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            let empty = |termination, counts| FitReport {
                termination,
                objective: None,
                active_bounds: 0,
                counts,
            };
            if let Some(stop) = context.stop_reason()? {
                return Ok(empty(StatsTermination::Stopped(stop), context.counts()));
            }
            if context
                .invoke_provider(EvaluationRole::Domain, || {
                    request
                        .objective
                        .validate(request.initial, &mut workspace.objective)
                })?
                .is_exhausted()
            {
                return Ok(empty(StatsTermination::BudgetExhausted, context.counts()));
            }
            workspace.candidate.copy_from_slice(request.initial);
            let termination = solver
                .solve(
                    request,
                    &mut workspace.candidate,
                    &mut workspace.objective,
                    context,
                )?
                .validate_fit()?;
            if let Some(stop) = context.stop_reason()? {
                return Ok(empty(StatsTermination::Stopped(stop), context.counts()));
            }
            validate_parameters(&workspace.candidate, request.bounds, request.integrality)?;
            let scioxide_core::Invocation::Evaluated(value) =
                context.invoke_provider(EvaluationRole::Objective, || {
                    request
                        .objective
                        .value(&workspace.candidate, &mut workspace.objective)
                })?
            else {
                return Ok(empty(StatsTermination::BudgetExhausted, context.counts()));
            };
            finite(value)?;
            if let Some(stop) = context.stop_reason()? {
                return Ok(empty(StatsTermination::Stopped(stop), context.counts()));
            }
            let active_bounds = workspace
                .candidate
                .iter()
                .zip(request.bounds.as_slice())
                .filter(|(x, (a, b))| a.is_some_and(|a| **x == a) || b.is_some_and(|b| **x == b))
                .count();
            output.copy_from_slice(&workspace.candidate);
            Ok(FitReport {
                termination,
                objective: Some(value),
                active_bounds,
                counts: context.counts(),
            })
        },
        |result| {
            (
                match result {
                    Ok(report) if report.termination().is_success() => {
                        scioxide_core::AttemptOutcome::Succeeded
                    }
                    Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                    Err(_) => scioxide_core::AttemptOutcome::Failed,
                },
                None,
            )
        },
    );
    workspace.solve = slot;
    result
}

/// Fitted parameters and the report of a fit.
pub struct FitResult<S: RealScalar> {
    /// Fitted parameters, `None` if the fit stopped before the final evaluation.
    parameters: Option<Vec<S>>,
    /// Status and diagnostics.
    report: FitReport<S>,
}
impl<S: RealScalar> FitResult<S> {
    /// Returns the fitted parameters, or `None` if the fit stopped before the
    /// final objective evaluation.
    pub fn parameters(&self) -> Option<&[S]> {
        self.parameters.as_deref()
    }
    /// Returns the report of the fit.
    pub const fn report(&self) -> &FitReport<S> {
        &self.report
    }
    /// Returns the parameters and the report.
    pub fn into_parts(self) -> (Option<Vec<S>>, FitReport<S>) {
        (self.parameters, self.report)
    }
}

/// Executes [`fit_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn fit_with_options<S: RealScalar, W: Send>(
    request: FitRequest<'_, S, W>,
    solver: &mut dyn FitSolver<S, W>,
    options: crate::StatisticsOptions,
) -> Result<FitResult<S>, SciOxideError> {
    fit_scoped(
        request,
        solver,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Executes [`fit_scoped`] with the [one-shot defaults](crate#one-shot-forms)
/// and sequential native capabilities, requiring successful termination.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors, and maps
/// every unsuccessful termination to an error under the one-shot contract.
pub fn fit<S: RealScalar, W: Send>(
    request: FitRequest<'_, S, W>,
    solver: &mut dyn FitSolver<S, W>,
) -> Result<FitResult<S>, SciOxideError> {
    let result = fit_with_options(request, solver, Default::default())?;
    let report = result.report();
    if let Some(error) = report.termination().one_shot_error(
        scioxide_core::ConvergenceReason::NestedSolver,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
    ) {
        return Err(error);
    }
    Ok(result)
}
/// Minimizes an objective with the given solver and returns the parameters.
///
/// Allocates the output and the workspace for one call, then runs [`fit_into`].
/// [`StatisticsOptions::max_bytes`](crate::StatisticsOptions::max_bytes) limits
/// the two parameter vectors and the objective's workspace together.
///
/// # Errors
///
/// The errors of [`fit_into`] and [`FitWorkspace::for_objective`], and
/// `ResourceLimit` if the memory needed exceeds the byte limit.
pub fn fit_scoped<'scope, 'capability, S: RealScalar, W: Send>(
    request: FitRequest<'_, S, W>,
    solver: &mut dyn FitSolver<S, W>,
    options: crate::StatisticsOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<FitResult<S>, SciOxideError> {
    let mut state = scioxide_core::SolveState::new();
    scope.into().run(
        &mut state,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "fit", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            let output_bytes = request
                .objective
                .dimension()
                .checked_mul(core::mem::size_of::<S>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("fit_output"))
                })?;
            // Output, candidate and objective scratch are admitted together.
            let scratch = request.objective.workspace_bytes()?;
            let required = output_bytes
                .checked_mul(2)
                .and_then(|vectors| vectors.checked_add(scratch))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("fit_workspace_bytes"))
                })?;
            crate::error_helpers::limit(
                required,
                options.max_bytes(),
                scioxide_core::quantity!("fit_workspace_bytes"),
            )?;
            let mut workspace = FitWorkspace::for_objective(
                request.objective,
                options.with_max_bytes(options.max_bytes() - output_bytes),
            )?;
            let mut output = scioxide_core::backend::try_filled(
                request.objective.dimension(),
                S::ZERO,
                scioxide_core::quantity!("fit_result"),
            )?;
            let report = fit_into(request, solver, &mut output, &mut workspace, context)?;
            Ok(FitResult {
                parameters: report.objective.map(|_| output),
                report,
            })
        },
    )
}
/// Classification of the observed information matrix at a fit, from
/// [`fit_information`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitIdentifiability {
    /// The information is positive definite in the free parameters.
    Identifiable,
    /// The information is numerically rank deficient.
    RankDeficient,
    /// The information has a negative eigenvalue, so the fit is not a local
    /// minimum of the negative log likelihood.
    Indefinite,
    /// A free parameter is on a bound or is an integer, so the usual asymptotic
    /// covariance does not apply.
    Nonregular,
}
/// Diagnostics of the observed information matrix at a fit: classification,
/// rank, condition number and asymptotic covariance.
///
/// The covariance is present only when the classification is
/// [`FitIdentifiability::Identifiable`].
pub struct FitInformation<S: RealScalar> {
    identifiability: FitIdentifiability,
    rank: usize,
    condition: Option<S>,
    covariance: Option<Vec<S>>,
}
impl<S: RealScalar> FitInformation<S> {
    /// Returns the classification. It is a numerical diagnosis at this point, not
    /// a proof that the model is identifiable.
    pub const fn identifiability(&self) -> FitIdentifiability {
        self.identifiability
    }
    /// Returns the numerical rank of the information in the free parameters.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Returns the condition number `λ_max/λ_min` of the information in the free
    /// parameters, or `None` if it is rank deficient or indefinite.
    pub const fn condition(&self) -> Option<S> {
        self.condition
    }
    /// Returns the `d × d` covariance, row-major, with zeros in the rows and
    /// columns of fixed parameters, or `None` unless the fit is identifiable.
    pub fn covariance(&self) -> Option<&[S]> {
        self.covariance.as_deref()
    }
    /// Returns the classification, rank, condition number and covariance.
    pub fn into_parts(self) -> (FitIdentifiability, usize, Option<S>, Option<Vec<S>>) {
        (
            self.identifiability,
            self.rank,
            self.condition,
            self.covariance,
        )
    }
}
/// Analyzes the observed information matrix at a maximum likelihood fit.
///
/// `hessian` is the `d × d` Hessian of the negative log likelihood at
/// `parameters`, row-major, for example from
/// [`ExactFitDerivatives`](crate::ExactFitDerivatives). The Hessian of a moment
/// or quantile matching objective is not an information matrix. Parameters
/// with equal bounds are fixed and left out. The function computes the
/// eigenvalues `λ` of the free block `I`, counts those with `|λ| > τ` as its
/// rank, where `τ` comes from `rank_tolerance` relative to the largest `|λ|`,
/// and classifies `I`:
///
/// - `Nonregular` if a free parameter is integer or lies on a bound.
/// - `Indefinite` if an eigenvalue is below `−τ`.
/// - `RankDeficient` if the rank is below the number of free parameters.
/// - `Identifiable` otherwise. The covariance is then `I⁻¹`.
///
/// The cost is one symmetric eigendecomposition, O(d³).
///
/// # Errors
///
/// - `InvalidInput` if `bounds`, `integrality` or `hessian` (which needs `d²`
///   entries) has the wrong length.
/// - An error if `parameters` or `hessian` is not finite, or `parameters` lies
///   outside the bounds or is not integer where required.
/// - `InvalidInput` if the free block of `hessian` is not exactly symmetric.
/// - The errors of the symmetric eigendecomposition, and an error if the
///   condition number or covariance overflows.
pub fn fit_information<S>(
    parameters: &[S],
    bounds: &Bounds<S>,
    integrality: &[bool],
    hessian: &[S],
    rank_tolerance: scioxide_core::RankThreshold<S>,
    options: &scioxide_linalg::HermitianEigenOptions<S>,
) -> Result<FitInformation<S>, SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
{
    validate_parameters(parameters, bounds, integrality)?;
    let d = parameters.len();
    let size = d.checked_mul(d).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("information_hessian"))
    })?;
    if hessian.len() != size {
        return Err(invalid(
            scioxide_core::argument!("hessian"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "information Hessian shape",
        ));
    }
    for &x in hessian {
        finite(x)?;
    }
    let mut free =
        scioxide_core::backend::try_with_capacity(d, scioxide_core::quantity!("free_parameters"))?;
    let mut nonregular = false;
    for (i, ((&x, &(a, b)), &integer)) in parameters
        .iter()
        .zip(bounds.as_slice())
        .zip(integrality)
        .enumerate()
    {
        if a.is_some() && a == b {
            continue;
        }
        free.push(i);
        nonregular |= integer || a.is_some_and(|a| a == x) || b.is_some_and(|b| b == x);
    }
    let n = free.len();
    let mut covariance = scioxide_core::backend::try_filled(
        size,
        S::ZERO,
        scioxide_core::quantity!("fit_covariance"),
    )?;
    if n == 0 {
        return Ok(FitInformation {
            identifiability: FitIdentifiability::Identifiable,
            rank: 0,
            condition: None,
            covariance: Some(covariance),
        });
    }
    let mut entries = scioxide_core::backend::try_with_capacity(
        n * n,
        scioxide_core::quantity!("free_information"),
    )?;
    for &j in &free {
        for &i in &free {
            entries.push(hessian[i * d + j]);
        }
    }
    let matrix = scioxide_linalg::Matrix::from_vec(n, n, entries)?;
    let eig = scioxide_linalg::HermitianEigenDecomposition::decompose_with_options(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        options,
    )?;
    let scale = eig
        .eigenvalues()
        .iter()
        .fold(S::ZERO, |s, &x| s.max(x.abs()));
    let threshold = rank_tolerance.threshold(scale)?;
    let rank = eig
        .eigenvalues()
        .iter()
        .filter(|&&x| x.abs() > threshold)
        .count();
    let indefinite = eig.eigenvalues().iter().any(|&x| x < -threshold);
    let status = if nonregular {
        FitIdentifiability::Nonregular
    } else if indefinite {
        FitIdentifiability::Indefinite
    } else if rank < n {
        FitIdentifiability::RankDeficient
    } else {
        FitIdentifiability::Identifiable
    };
    let condition = if rank == n && !indefinite {
        let low = eig.eigenvalues().iter().fold(scale, |s, &x| s.min(x));
        Some(finite(scale / low)?)
    } else {
        None
    };
    if status == FitIdentifiability::Identifiable {
        for (i, &row) in free.iter().enumerate() {
            for (j, &col) in free.iter().enumerate() {
                let mut value = S::ZERO;
                for (k, &lambda) in eig.eigenvalues().iter().enumerate() {
                    value += eig.eigenvectors().view().get(i, k)?
                        * eig.eigenvectors().view().get(j, k)?
                        / lambda;
                }
                covariance[row * d + col] = finite(value)?;
            }
        }
    }
    Ok(FitInformation {
        identifiability: status,
        rank,
        condition,
        covariance: (status == FitIdentifiability::Identifiable).then_some(covariance),
    })
}
