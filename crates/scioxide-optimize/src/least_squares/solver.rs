//! Nonlinear least squares with Gauss–Newton or Levenberg–Marquardt steps.

use scioxide_core::{
    AttemptKind, AttemptRecord, Capacity, ExecutionContext, NumericScalar, PartialResultTrust,
    ProblemToken, RealScalar, Scaled, SciOxideError, SolveBudgets, SolveContext, SolveIdentity,
    SolveScope,
};

use super::{LeastSquaresGlobalization, LeastSquaresLinearSolver, LeastSquaresOptions};
use crate::LeastSquaresResult;
use crate::attempt::{Entry, Shell};
use crate::least_squares::{LeastSquaresShape, LeastSquaresStorage};
use crate::traits::LeastSquaresProblem;

/// Describes the accepted state of a least-squares solve.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresProgress<S: RealScalar> {
    cost: Scaled<S>,
    iterations: usize,
    gradient_norm: Option<Scaled<S>>,
    trust: PartialResultTrust,
}

impl<S: RealScalar> LeastSquaresProgress<S> {
    /// Returns the objective at the accepted parameters, as a scaled value that
    /// cannot overflow.
    pub const fn cost(self) -> Scaled<S> {
        self.cost
    }
    /// Returns the number of accepted iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the norm of the objective gradient at the accepted parameters, if
    /// it was evaluated.
    pub const fn gradient_norm(self) -> Option<Scaled<S>> {
        self.gradient_norm
    }
    /// Returns which checks the accepted parameters passed.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

/// Borrows the accepted parameters, residual and progress of the last solve.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresAccepted<'a, S: RealScalar> {
    parameters: &'a [S],
    residual: &'a [S],
    progress: LeastSquaresProgress<S>,
}

impl<'a, S: RealScalar> LeastSquaresAccepted<'a, S> {
    /// Returns the accepted parameters.
    pub const fn parameters(&self) -> &'a [S] {
        self.parameters
    }
    /// Returns the residual at the accepted parameters.
    pub const fn residual(&self) -> &'a [S] {
        self.residual
    }
    /// Returns the progress at the accepted parameters.
    pub const fn progress(&self) -> LeastSquaresProgress<S> {
        self.progress
    }
}

/// Describes the problem a [`LeastSquaresWorkspace`] is prepared for: its
/// dimensions and whether it provides Jacobian products.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeastSquaresWorkspaceRequest {
    parameters: usize,
    residuals: usize,
    operator_scratch: Option<usize>,
}

impl LeastSquaresWorkspaceRequest {
    /// Requests storage for a dense Jacobian with `residuals` rows and
    /// `parameters` columns.
    pub const fn dense(parameters: usize, residuals: usize) -> Self {
        Self {
            parameters,
            residuals,
            operator_scratch: None,
        }
    }

    /// Returns the dimensions of `problem` and whether it provides Jacobian
    /// products.
    ///
    /// # Errors
    ///
    /// Returns the error of
    /// [`LeastSquaresProblem::jacobian_operator_workspace_len`].
    pub fn of<S, P>(problem: &P) -> Result<Self, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        P: LeastSquaresProblem<S> + ?Sized,
    {
        Ok(Self {
            parameters: problem.param_dim(),
            residuals: problem.residual_dim(),
            operator_scratch: problem.jacobian_operator_workspace_len()?,
        })
    }
}

/// Lists the storage a [`LeastSquares`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeastSquaresWorkspaceRequirements {
    shape: LeastSquaresShape,
}

impl LeastSquaresWorkspaceRequirements {
    /// Returns the number of parameters.
    pub const fn parameters(self) -> usize {
        self.shape.parameters
    }
    /// Returns the number of residual components.
    pub const fn residuals(self) -> usize {
        self.shape.residuals
    }
    /// Returns whether the solve forms a dense Jacobian.
    pub const fn dense(self) -> bool {
        self.shape.operator_scratch.is_none()
    }
}

/// Holds the Jacobian, factors and trial storage of a [`LeastSquares`] solve,
/// and a record of the last call.
///
/// A dense workspace holds all storage, so a solve with a prepared workspace
/// and output does not allocate. An operator workspace sizes its LSMR storage
/// on the first solve of a problem.
#[derive(Debug)]
pub struct LeastSquaresWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) storage: LeastSquaresStorage<S>,
    staged: LeastSquaresResult<S>,
    shell: Shell<LeastSquaresProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for LeastSquaresWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LeastSquaresWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            storage: LeastSquaresStorage::new(),
            staged: LeastSquaresResult::empty(),
            shell: Shell::new(),
        }
    }

    /// Reports whether the workspace already fits `requirements`.
    ///
    /// # Errors
    ///
    /// Never fails; the workspace has no byte limit.
    pub fn check(
        &self,
        requirements: &LeastSquaresWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        let shape = requirements.shape;
        Ok(
            if self.storage.fits(shape) && self.staged.fits(shape.parameters, shape.residuals) {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Reserves storage for `requirements`.
    ///
    /// A different shape replaces the storage. On failure the existing storage is
    /// kept.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(
        &mut self,
        requirements: &LeastSquaresWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let shape = requirements.shape;
        self.storage.prepare(shape)?;
        self.staged.try_reserve(shape.parameters, shape.residuals)
    }

    /// Returns the record of the last call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<LeastSquaresProgress<S>>> {
        self.shell.last_attempt()
    }

    /// Returns the accepted parameters and residual of the last call, if it
    /// accepted any.
    pub fn last_accepted(&self) -> Option<LeastSquaresAccepted<'_, S>> {
        Some(LeastSquaresAccepted {
            parameters: &self.storage.x,
            residual: &self.storage.residual,
            progress: *self.shell.progress()?,
        })
    }
}

/// Minimizes a sum of squared residuals, optionally with a robust loss.
///
/// The objective is
///
/// ```text
/// f(x) = ½ Σᵢ wᵢ c² ρ(|rᵢ(x) / σᵢ|² / c²),
/// ```
///
/// where `r: ℝⁿ → ℝᵐ` is the residual of a [`LeastSquaresProblem`], `wᵢ ≥ 0`
/// are weights, `σᵢ > 0` residual scales, and `ρ` the
/// [`LeastSquaresLoss`](crate::LeastSquaresLoss) with scale `c`. With the
/// default quadratic loss and unit weights, `f(x) = ½ ‖r(x)‖²`.
///
/// Each iteration linearizes `r` and solves the linear least-squares
/// subproblem by column-pivoted QR, a truncated SVD for rank-deficient or wide
/// Jacobians, or LSMR when the problem provides only Jacobian products. Robust
/// losses use iteratively reweighted residuals with a curvature correction.
/// [`LeastSquaresGlobalization`] selects how the step is safeguarded:
/// Levenberg–Marquardt with a trust region (the default), or Gauss–Newton with
/// backtracking. A missing analytic Jacobian is replaced by automatic
/// differentiation or finite differences, as set by the derivative strategy.
/// Parameters are scaled by the norms of the Jacobian columns unless the
/// problem supplies fixed scales.
///
/// For linear residuals with bounds, use [`Nnls`](crate::Nnls) or
/// [`BoundedLinearLeastSquares`](crate::BoundedLinearLeastSquares).
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{LeastSquares, LeastSquaresFn};
/// let problem = LeastSquaresFn::new(2, 2, |x: &[f64], r: &mut [f64]| {
///     r[0] = x[0] - 1.0;
///     r[1] = x[1] + 2.0;
///     Ok(())
/// })?;
/// let result = LeastSquares::default().solve(&problem, &[0.0, 0.0])?;
/// result.require_success()?;
/// assert!((result.x()[0] - 1.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and
///   theory", in *Numerical Analysis*, Lecture Notes in Math. 630, Springer,
///   1978, 105–116.
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
///   2006, ch. 10.
/// - D. C.-L. Fong and M. Saunders, "LSMR: An iterative algorithm for sparse
///   least-squares problems", *SIAM J. Sci. Comput.* 33(5), 2950–2971, 2011.
#[derive(Clone, Copy)]
pub struct LeastSquares<S: RealScalar> {
    options: LeastSquaresOptions<S>,
    configuration: u128,
}

impl<S: RealScalar> core::fmt::Debug for LeastSquares<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LeastSquares")
            .field("options", &self.options)
            .finish()
    }
}

impl<S: RealScalar> Default for LeastSquares<S> {
    fn default() -> Self {
        let options = LeastSquaresOptions::default();
        Self {
            configuration: crate::context::options_token(&options),
            options,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LeastSquares<S> {
    /// Creates a solver after checking `options`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the loss scale is not positive and finite.
    /// - `InvalidInput` if the Levenberg–Marquardt initial damping is negative
    ///   or not finite.
    pub fn new(options: LeastSquaresOptions<S>) -> Result<Self, SciOxideError> {
        super::options::positive(options.loss_scale, scioxide_core::argument!("scale"))?;
        if let LeastSquaresGlobalization::LevenbergMarquardt { initial_damping } =
            options.globalization
            && (!initial_damping.is_finite()
                || initial_damping < <S as scioxide_core::AlgebraicScalar>::ZERO)
        {
            return Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("damping"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "damping must be finite and nonnegative",
            ));
        }
        Ok(Self {
            configuration: crate::context::options_token(&options),
            options,
        })
    }

    /// Returns the options.
    pub const fn options(&self) -> &LeastSquaresOptions<S> {
        &self.options
    }

    /// Returns the storage a solve of the problem described by `request` needs.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if the linear solver is
    ///   [`LeastSquaresLinearSolver::Lsmr`] but the problem provides no Jacobian
    ///   products.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if
    ///   [`LeastSquaresStopping::Minpack`](crate::LeastSquaresStopping::Minpack)
    ///   is selected for a problem solved with Jacobian products.
    pub fn workspace_requirements(
        &self,
        request: LeastSquaresWorkspaceRequest,
    ) -> Result<LeastSquaresWorkspaceRequirements, SciOxideError> {
        let operator_scratch = match (self.options.linear_solver, request.operator_scratch) {
            (LeastSquaresLinearSolver::DenseSvd, _) | (LeastSquaresLinearSolver::Auto, None) => {
                None
            }
            (_, Some(scratch)) => Some(scratch),
            (LeastSquaresLinearSolver::Lsmr, None) => {
                return Err(SciOxideError::not_implemented(
                    "LSMR requires Jacobian and adjoint products",
                ));
            }
        };
        if operator_scratch.is_some()
            && matches!(
                self.options.stopping,
                super::LeastSquaresStopping::Minpack { .. }
            )
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("stopping"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "MINPACK orthogonality stopping requires a dense Jacobian",
            ));
        }
        Ok(LeastSquaresWorkspaceRequirements {
            shape: LeastSquaresShape {
                parameters: request.parameters,
                residuals: request.residuals,
                operator_scratch,
            },
        })
    }

    /// Allocates a workspace for the problem described by `request`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace_requirements`], and
    /// `SizeOverflow` or `AllocationFailure`.
    pub fn workspace(
        &self,
        request: LeastSquaresWorkspaceRequest,
    ) -> Result<LeastSquaresWorkspace<S>, SciOxideError> {
        let mut workspace = LeastSquaresWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Solves `problem` from `x0` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace`] and [`Self::solve_into`].
    pub fn solve<P: LeastSquaresProblem<S> + ?Sized>(
        &self,
        problem: &P,
        x0: &[S],
    ) -> Result<LeastSquaresResult<S>, SciOxideError> {
        let mut workspace = self.workspace(LeastSquaresWorkspaceRequest::of(problem)?)?;
        let mut output = LeastSquaresResult::empty();
        self.solve_into(
            problem,
            x0,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Solves `problem` from `x0` and writes the result to `output`.
    ///
    /// The workspace grows if it is smaller than
    /// [`LeastSquaresWorkspaceRequest::of`]`(problem)` requires. A rejected
    /// trial never changes the accepted parameters. Stopping rules, results on
    /// a budget stop and the workspace record are described under
    /// [Termination](crate#termination) and
    /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0.len()` differs from the number of
    ///   parameters.
    /// - `NonFinite` if `x0` contains NaN or infinity.
    /// - `InvalidInput` if a parameter scale or residual scale is not positive
    ///   and finite, a weight is negative or not finite, or the residual group
    ///   size is not 1 or 2.
    /// - The errors of [`Self::workspace_requirements`].
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the solve before the first accepted point.
    /// - Errors returned by the residual or Jacobian callbacks, unchanged.
    pub fn solve_into<'scope, 'capability, P>(
        &self,
        problem: &P,
        x0: &[S],
        output: &mut LeastSquaresResult<S>,
        workspace: &mut LeastSquaresWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        P: LeastSquaresProblem<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        super::engine::validate(problem)?;
        if x0.len() != problem.param_dim() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(problem.param_dim()),
                scioxide_core::Dimensions::one(x0.len()),
            ));
        }
        crate::utils::validate_finite_origin(x0, scioxide_core::NonFiniteOrigin::Input)?;
        let requirements =
            self.workspace_requirements(LeastSquaresWorkspaceRequest::of(problem)?)?;
        workspace.try_reserve(&requirements)?;
        if !requirements.dense() {
            super::operator::prepare(problem, &self.options, &mut workspace.storage)?;
        }
        crate::attempt::preflight(&scope, None)?;
        let entry = Entry::Call {
            identity: self.identity(
                ProblemToken::new(0),
                problem.param_dim(),
                problem.residual_dim(),
            ),
            budgets: self.budgets(),
        };
        crate::attempt::run(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| self.solve_in_context(problem, x0, staged, storage, context),
            |storage, succeeded| {
                Some(LeastSquaresProgress {
                    cost: storage.accepted_value?,
                    iterations: storage.accepted_iterations,
                    gradient_norm: storage.gradient_norm,
                    trust: if succeeded
                        && matches!(
                            self.options.stopping,
                            super::LeastSquaresStopping::OriginalGradient
                        ) {
                        PartialResultTrust::ResidualChecked
                    } else {
                        PartialResultTrust::Admissible
                    },
                })
            },
        )
    }

    /// Identifies a solve by globalization, problem shape and options.
    pub(crate) fn identity(
        &self,
        problem: ProblemToken,
        parameters: usize,
        residuals: usize,
    ) -> SolveIdentity {
        SolveIdentity::new(
            problem,
            match self.options.globalization {
                LeastSquaresGlobalization::GaussNewton { .. } => {
                    scioxide_core::algorithm_id!("optimize", "GaussNewton", 1)
                }
                LeastSquaresGlobalization::LevenbergMarquardt { .. } => {
                    scioxide_core::algorithm_id!("optimize", "LevenbergMarquardt", 1)
                }
            },
            ((parameters as u128) << 64) | residuals as u128,
            self.configuration,
        )
    }

    /// Returns the iteration limit of one solve.
    pub(crate) fn budgets(&self) -> SolveBudgets {
        super::super::unconstrained::iteration_budgets(self.options.max_iterations)
    }

    /// Solves within a solve context the caller has already started.
    pub(crate) fn solve_in_context<P: LeastSquaresProblem<S> + ?Sized>(
        &self,
        problem: &P,
        x0: &[S],
        output: &mut LeastSquaresResult<S>,
        storage: &mut LeastSquaresStorage<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let session = crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
        let observed = session.observe_least_squares(problem);
        let outcome = super::engine::solve(&observed, x0, &self.options, output, storage, &session);
        crate::utils::complete_least_squares(outcome, storage, output, &session)
    }

    /// Returns whether residuals keep the statistical meaning that fitting
    /// adapters need.
    pub(crate) fn supports_statistical_residuals(&self) -> bool {
        self.options.loss == super::LeastSquaresLoss::Quadratic
            && self.options.linear_solver != LeastSquaresLinearSolver::Lsmr
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use scioxide_core::{EvaluationRole, GradientTolerance, StepTolerance, Tolerance};
    use scioxide_linalg::Matrix;

    struct LinearLsp;
    impl LeastSquaresProblem<f64> for LinearLsp {
        fn param_dim(&self) -> usize {
            2
        }
        fn residual_dim(&self) -> usize {
            3
        }
        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]);
            Ok(())
        }
        fn jacobian_into(
            &self,
            _x: &[f64],
            output: &mut Matrix<f64>,
        ) -> Result<bool, SciOxideError> {
            output
                .as_mut_slice()
                .copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            Ok(true)
        }
    }

    struct NonlinearLsp;
    impl LeastSquaresProblem<f64> for NonlinearLsp {
        fn param_dim(&self) -> usize {
            2
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output.copy_from_slice(&[x[0] * x[0] - 1.0, x[1] * x[1] - 4.0]);
            Ok(())
        }
    }

    struct RosenbrockLsp;
    impl LeastSquaresProblem<f64> for RosenbrockLsp {
        fn param_dim(&self) -> usize {
            2
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output.copy_from_slice(&[10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]);
            Ok(())
        }
    }

    fn solver(globalization: LeastSquaresGlobalization<f64>) -> LeastSquares<f64> {
        let tolerance = Tolerance::from_decimal_places(8, 8);
        LeastSquares::new(
            LeastSquaresOptions::default()
                .with_gradient_tolerance(GradientTolerance::from_tolerance(tolerance))
                .with_step_tolerance(StepTolerance::from_tolerance(tolerance))
                .with_globalization(globalization),
        )
        .unwrap()
    }

    const GLOBALIZATIONS: [LeastSquaresGlobalization<f64>; 2] = [
        LeastSquaresGlobalization::gauss_newton(),
        LeastSquaresGlobalization::levenberg_marquardt(),
    ];

    #[test]
    fn linear_problem_converges_and_counts_factorization_work() {
        for globalization in GLOBALIZATIONS {
            let solver = solver(globalization);
            let mut workspace = solver
                .workspace(LeastSquaresWorkspaceRequest::dense(2, 3))
                .unwrap();
            let mut result = LeastSquaresResult::empty();
            solver
                .solve_into(
                    &LinearLsp,
                    &[0.0, 0.0],
                    &mut result,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!((result.x[0] - 1.0).abs() < 1e-6);
            assert!((result.x[1] - 2.0).abs() < 1e-6);
            let counts = workspace.last_attempt().unwrap().counts().own();
            assert!(counts.provider(EvaluationRole::Factorization) > 0);
            assert_eq!(
                counts.provider(EvaluationRole::Factorization),
                counts.provider(EvaluationRole::LinearSolve)
            );
            let accepted = workspace.last_accepted().unwrap();
            assert_eq!(accepted.parameters(), result.x());
            assert_eq!(accepted.residual(), result.residual());
        }
    }

    #[test]
    fn algorithm_identity_follows_globalization() {
        for (globalization, method) in GLOBALIZATIONS
            .into_iter()
            .zip(["GaussNewton", "LevenbergMarquardt"])
        {
            let result = solver(globalization)
                .solve(&LinearLsp, &[0.0, 0.0])
                .unwrap();
            assert_eq!(result.algorithm().unwrap().method(), method);
        }
    }

    #[test]
    fn finite_difference_problems_converge() {
        let result = solver(LeastSquaresGlobalization::gauss_newton())
            .solve(&NonlinearLsp, &[0.5, 1.5])
            .unwrap();
        assert!(result.is_success());
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 2.0).abs() < 1e-8);
        let result = solver(LeastSquaresGlobalization::levenberg_marquardt())
            .solve(&RosenbrockLsp, &[-1.0, 1.0])
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn worsening_full_step_is_rejected_before_state_commit() {
        struct ScalarSquareResidual;
        impl LeastSquaresProblem<f64> for ScalarSquareResidual {
            fn param_dim(&self) -> usize {
                1
            }
            fn residual_dim(&self) -> usize {
                1
            }
            fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
                output[0] = x[0] * x[0] - 1.0;
                Ok(())
            }
            fn jacobian_into(
                &self,
                x: &[f64],
                output: &mut Matrix<f64>,
            ) -> Result<bool, SciOxideError> {
                output.as_mut_slice()[0] = 2.0 * x[0];
                Ok(true)
            }
        }
        let solver = solver(LeastSquaresGlobalization::gauss_newton());
        let mut workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::dense(1, 1))
            .unwrap();
        let mut result = LeastSquaresResult::empty();
        let initial_cost = (0.1_f64 * 0.1 - 1.0).powi(2);
        solver
            .solve_into(
                &ScalarSquareResidual,
                &[0.1],
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(result.cost().unwrap() < initial_cost);
        assert!(workspace.last_attempt().unwrap().counts().own().rejected() > 0);
        assert!(result.x()[0].abs() < 2.0);
    }

    #[test]
    fn rejects_dimension_mismatch_non_finite_residual_and_invalid_options() {
        struct NonFinite;
        impl LeastSquaresProblem<f64> for NonFinite {
            fn param_dim(&self) -> usize {
                1
            }
            fn residual_dim(&self) -> usize {
                1
            }
            fn residual_into(&self, _x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
                output[0] = f64::NAN;
                Ok(())
            }
        }
        for globalization in GLOBALIZATIONS {
            let solver = solver(globalization);
            assert!(matches!(
                (solver.solve(&LinearLsp, &[1.0]))
                    .as_ref()
                    .map_err(SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
            ));
            assert!(matches!(
                (solver.solve(&NonFinite, &[0.0]))
                    .as_ref()
                    .map_err(SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
            ));
        }
        for options in [
            LeastSquaresOptions::<f64>::default()
                .with_loss(super::super::LeastSquaresLoss::Huber, 0.0),
            LeastSquaresOptions::default()
                .with_loss(super::super::LeastSquaresLoss::Huber, f64::NAN),
            LeastSquaresOptions::default().with_globalization(
                LeastSquaresGlobalization::LevenbergMarquardt {
                    initial_damping: -1.0,
                },
            ),
        ] {
            assert_eq!(
                LeastSquares::new(options).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn supports_f32() {
        struct LinearF32;
        impl LeastSquaresProblem<f32> for LinearF32 {
            fn param_dim(&self) -> usize {
                2
            }
            fn residual_dim(&self) -> usize {
                3
            }
            fn residual_into(&self, x: &[f32], output: &mut [f32]) -> Result<(), SciOxideError> {
                output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]);
                Ok(())
            }
            fn jacobian_into(
                &self,
                _x: &[f32],
                output: &mut Matrix<f32>,
            ) -> Result<bool, SciOxideError> {
                output
                    .as_mut_slice()
                    .copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
                Ok(true)
            }
        }
        let tolerance = Tolerance::from_decimal_places(5, 5);
        for globalization in [
            LeastSquaresGlobalization::gauss_newton(),
            LeastSquaresGlobalization::levenberg_marquardt(),
        ] {
            let result = LeastSquares::new(
                LeastSquaresOptions::default()
                    .with_gradient_tolerance(GradientTolerance::from_tolerance(tolerance))
                    .with_step_tolerance(StepTolerance::from_tolerance(tolerance))
                    .with_globalization(globalization),
            )
            .unwrap()
            .solve(&LinearF32, &[0.0, 0.0])
            .unwrap();
            assert!(result.is_success());
            assert!((result.x[0] - 1.0).abs() < 1e-4);
            assert!((result.x[1] - 2.0).abs() < 1e-4);
        }
    }
}
