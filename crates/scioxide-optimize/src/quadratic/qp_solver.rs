//! Active-set quadratic programming solver.
//!
//! Solves problems of the form:
//!
//! Minimize `0.5 x^T Q x + c^T x`
//! subject to `A_ub x <= b_ub`, `A_eq x = b_eq`, `lower <= x <= upper`.
//!
//! The method depends on the constraints:
//! 1. Without constraints or bounds, solve `Q x = -c` by LDLᵀ or a rank-revealing spectral solve.
//! 2. With bounds only, use projected gradient with active-set tracking.
//! 3. With general constraints, use the active-set method.
//!
//! # Complexity
//!
//! Preparation takes O(n³). Free-Hessian and general-row updates take quadratic
//! work in their dimensions; bound changes also refresh projected general rows.
//! Space: O(n^2) for the Hessian and working matrices.

use super::{common::*, model::validate_warm};
use alloc::boxed::Box;

use scioxide_core::backend::try_copy_slice;
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{HermitianEigenvaluesBuffer, Matrix};

use crate::KktProvider;
use crate::attempt::{Entry, Shell};
use crate::context::SolveSession;
use crate::problem::QpProblem;
use crate::programming::{ProgrammingAccepted, ProgrammingProgress, ProgrammingResult};
use crate::result::OptimizeTermination;
use scioxide_core::{EvaluationRole, PartialResultTrust};

/// Solves convex quadratic programs.
///
/// The problem is a [`QpProblem`]:
///
/// ```text
/// minimize    ½ xᵀQx + cᵀx
/// subject to  aᵢᵀx ≤ bᵢ,  aᵢᵀx = bᵢ  or  aᵢᵀx ≥ bᵢ   (one sense per row i)
///             l ≤ x ≤ u
/// ```
///
/// where `Q` is a symmetric positive semidefinite Hessian, `c` is the linear
/// objective, `aᵢᵀ` is row `i` of the constraint matrix, `b` is the right-hand
/// side, and `l`, `u` are optional variable bounds (variables are free when no
/// bounds are attached). Fixed bounds `lⱼ = uⱼ` are treated as equality
/// constraints. The method depends on the problem and on
/// [`QpMethod`](crate::QpMethod):
///
/// - A zero Hessian makes the problem an LP, which the bounded revised simplex
///   method of [`SimplexSolver`](crate::SimplexSolver) solves.
/// - Without constraints or bounds, `Qx = −c` is solved by LDLᵀ factorization,
///   or by a rank-revealing spectral solve when `Q` is singular.
/// - With bounds only, [`QpMethod::Automatic`](crate::QpMethod::Automatic)
///   uses projected gradient steps with Barzilai–Borwein step lengths.
/// - Otherwise a primal active-set method runs. When the Hessian on the free
///   variables is positive definite, it keeps a Cholesky factor of that
///   Hessian and QR factors of the active general constraints, and updates
///   them in O(n²) work as constraints enter and leave the working set.
///   Semidefinite subproblems use a rank-revealing factorization of the KKT
///   system. After eight working-set changes without progress, Bland's rule
///   prevents cycling.
/// - [`QpMethod::PrimalDual`](crate::QpMethod::PrimalDual) uses
///   primal-dual predictor-corrector steps on the augmented KKT system.
///
/// A successful result carries an optimality certificate that has been checked
/// against the original problem; see [`ProgrammingResult::certificate`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_optimize::{QpProblem, QpSolver};
///
/// // Minimize x² + y² − 2x − 6y; the minimizer is (1, 3) with objective −10.
/// let hessian = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 2.0])?;
/// let problem = QpProblem::new(hessian, vec![-2.0, -6.0])?;
/// let result = QpSolver::default().solve(&problem)?;
/// assert!(result.is_success());
/// assert!((result.x()[0] - 1.0).abs() < 1e-9);
/// assert!((result.x()[1] - 3.0).abs() < 1e-9);
/// assert!((result.fun() + 10.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
///   Springer, 2006, ch. 16.
/// - J. Barzilai and J. M. Borwein, "Two-point step size gradient methods",
///   *IMA J. Numer. Anal.* 8(1), 141–148, 1988.
#[derive(Debug, Clone)]
pub struct QpSolver<S: RealScalar> {
    pub(super) tolerances: crate::OptimalityTolerances<S>,
    pub(super) max_iterations: Budget,
    pub(super) method: super::QpMethod,
    pub(super) warm_start: Option<super::QpWarmStart<S>>,
}

impl<S: RealScalar> Default for QpSolver<S> {
    fn default() -> Self {
        Self {
            tolerances: crate::OptimalityTolerances::programming_default(),
            max_iterations: Budget::ONE.saturating_add(999),
            method: super::QpMethod::Automatic,
            warm_start: None,
        }
    }
}

impl<S: RealScalar> QpSolver<S> {
    /// Selects the method (default [`QpMethod::Automatic`](crate::QpMethod::Automatic)).
    pub const fn with_method(mut self, method: super::QpMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets an initial primal point; each solve checks its length and feasibility.
    pub fn with_warm_start(mut self, warm: super::QpWarmStart<S>) -> Self {
        self.warm_start = Some(warm);
        self
    }

    /// Sets the convergence tolerances.
    ///
    /// The default is `1e-8` absolute and relative for every role (`1e-6` when
    /// the machine epsilon exceeds `1e-8`). [`ProgrammingVerifier`](crate::ProgrammingVerifier)
    /// defaults to `√ε` instead.
    #[must_use]
    pub const fn with_tolerances(mut self, tolerances: crate::OptimalityTolerances<S>) -> Self {
        self.tolerances = tolerances;
        self
    }

    /// Sets the maximum number of active-set or projected-gradient iterations (default 1000).
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns the convergence tolerances.
    pub const fn tolerances(&self) -> crate::OptimalityTolerances<S> {
        self.tolerances
    }

    /// Returns the maximum number of active-set or projected-gradient iterations.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
}

/// Holds the reusable storage of [`QpSolver`].
///
/// One workspace accepts any later QP. Without an injected [`KktProvider`], the
/// positive definite active-set method updates its own factors and other KKT
/// systems use a built-in dense factorization. An injected provider receives
/// KKT systems from which the variables at active bounds have been eliminated,
/// so they contain only general constraint rows.
pub struct QpWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    provider: Option<Box<dyn KktProvider<S> + Send>>,
    active: Option<super::active_set::Storage<S>>,
    staged: ProgrammingResult<S>,
    shell: Shell<ProgrammingProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for QpWorkspace<S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("QpWorkspace")
            .field("injected_provider", &self.provider.is_some())
            .field("staged", &self.staged)
            .field("shell", &self.shell)
            .finish()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> QpWorkspace<S> {
    /// Replaces the built-in dense KKT factorization with `provider` for every later call.
    #[must_use]
    pub fn with_kkt_provider(mut self, provider: Box<dyn KktProvider<S> + Send>) -> Self {
        self.provider = Some(provider);
        self
    }
    /// Returns the record of the most recent call.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<ProgrammingProgress<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the point written by the most recent attempt, if any.
    pub fn last_accepted(&self) -> Option<ProgrammingAccepted<'_, S>> {
        crate::programming::last_accepted(&self.shell, &self.staged.x)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> QpSolver<S> {
    /// Creates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn workspace(&self, problem: &QpProblem<S>) -> Result<QpWorkspace<S>, SciOxideError> {
        let mut staged = ProgrammingResult::empty();
        staged.try_reserve(problem.dimension())?;
        Ok(QpWorkspace {
            provider: None,
            active: None,
            staged,
            shell: Shell::new(),
        })
    }

    /// Solves `problem` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_into`].
    pub fn solve(&self, problem: &QpProblem<S>) -> Result<ProgrammingResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut output = ProgrammingResult::empty();
        self.solve_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Solves `problem` into `output`, reusing `workspace`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent solve context. The iteration limit is [`Self::max_iterations`].
    /// If a limit, cancellation or deadline stops the method after it has a
    /// finite iterate, `output` holds that iterate with an unsuccessful
    /// termination.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the Hessian is not positive semidefinite or the warm
    ///   start is infeasible.
    /// - `DimensionMismatch` if the warm start has the wrong length.
    /// - `SizeOverflow` or `AllocationFailure` if a dense workspace is too large.
    /// - `NumericalInstability` if finite inputs overflow during the solve.
    /// - Errors returned by an injected [`KktProvider`].
    pub fn solve_into<'scope, 'capability>(
        &self,
        problem: &QpProblem<S>,
        output: &mut ProgrammingResult<S>,
        workspace: &mut QpWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(scioxide_core::AttemptKind::Call);
        let scope = scope.into();
        if let Some(warm) = &self.warm_start {
            validate_warm(problem, &warm.point, self.tolerances)?;
        }
        workspace.staged.try_reserve(problem.dimension())?;
        crate::attempt::preflight(&scope, None)?;
        let QpWorkspace {
            provider,
            active,
            staged,
            shell,
        } = workspace;
        crate::programming::run(
            shell,
            staged,
            output,
            scope,
            scioxide_core::AttemptKind::Call,
            Entry::Call {
                identity: self.identity(problem),
                budgets: scioxide_core::SolveBudgets::unbounded()
                    .with_limit(scioxide_core::BudgetKind::Iteration, self.max_iterations),
            },
            |context, staged| {
                *staged = self.solve_in_context(
                    problem,
                    context,
                    active,
                    provider
                        .as_deref_mut()
                        .map(|provider| provider as &mut (dyn KktProvider<S> + Send)),
                )?;
                Ok(())
            },
            ProgrammingResult::progress,
        )
    }

    /// Internal identity of the method, dimension and immutable QP controls.
    pub(crate) fn identity(&self, problem: &QpProblem<S>) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "QpSolver", 1),
            problem.dimension(),
            self,
        )
    }

    fn solve_in_context(
        &self,
        problem: &QpProblem<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
        active: &mut Option<super::active_set::Storage<S>>,
        provider: Option<&mut (dyn KktProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        session.check()?;
        let mut result = self.solve_kernel(problem, &session, active, provider)?;
        if session.stopped().is_none()
            && !matches!(
                (result.termination, result.evidence.as_ref()),
                (
                    OptimizeTermination::SuspectedUnbounded,
                    Some(crate::verification::OwnedProgrammingCertificate::Unboundedness(_))
                ) | (
                    OptimizeTermination::SuspectedInfeasible,
                    Some(crate::verification::OwnedProgrammingCertificate::Infeasibility(_))
                )
            )
            && matches!(
                result.termination,
                OptimizeTermination::SuspectedInfeasible | OptimizeTermination::SuspectedUnbounded
            )
        {
            let view = crate::QpVerificationProblem::from_problem(problem);
            let attempt = (|| {
                let evidence = if result.termination == OptimizeTermination::SuspectedInfeasible {
                    let dual = crate::verification::candidate::dual(
                        view.linear(),
                        &result.x,
                        problem.linear_objective(),
                        self.tolerances,
                        true,
                        &session,
                    )?;
                    crate::verification::OwnedProgrammingCertificate::Infeasibility(dual)
                } else {
                    let hessian = materialize_symmetric_hessian(problem.hessian())?;
                    let mut direction = session.invoke(EvaluationRole::LinearOperator, || {
                        compute_gradient(
                            &hessian,
                            problem.linear_objective(),
                            &result.x,
                            problem.dimension(),
                        )
                    })?;
                    direction.iter_mut().for_each(|value| *value = -*value);
                    crate::verification::OwnedProgrammingCertificate::Unboundedness(direction)
                };
                let mut workspace = crate::VerificationWorkspace::for_qp(&view)?;
                let report = crate::verification::kernel::qp_session(
                    &view,
                    evidence.borrow(&result.x),
                    self.tolerances,
                    &mut workspace,
                    Some(&session),
                )?;
                Ok::<_, SciOxideError>((evidence, report))
            })();
            match attempt {
                Ok((evidence, report)) => {
                    if report.is_accepted() {
                        result.termination = if matches!(
                            evidence,
                            crate::verification::OwnedProgrammingCertificate::Infeasibility(_)
                        ) {
                            OptimizeTermination::Infeasible
                        } else {
                            OptimizeTermination::Unbounded
                        };
                    }
                    result.evidence = Some(evidence);
                    result.verification = Some(report);
                }
                Err(_) if session.stopped().is_some() => {}
                Err(error) => return Err(error),
            }
        }
        if let Some(stop) = session.stopped() {
            result.termination = stop;
        }
        result.algorithm = Some(session.algorithm()?);
        result.counts = session.accounting()?;
        result.trust = if result.is_success() {
            PartialResultTrust::ResidualChecked
        } else {
            PartialResultTrust::DiagnosticOnly
        };
        session.assess(result.trust)?;
        Ok(result)
    }

    fn solve_kernel(
        &self,
        problem: &QpProblem<S>,
        session: &SolveSession<'_, '_, '_>,
        active: &mut Option<super::active_set::Storage<S>>,
        provider: Option<&mut (dyn KktProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        if let Some(warm) = &self.warm_start {
            validate_warm(problem, &warm.point, self.tolerances)?;
        }
        let hessian = materialize_symmetric_hessian(problem.hessian())?;
        self.validate_convex_hessian(&hessian, problem.dimension(), session)?;
        if provider.is_none()
            && hessian
                .iter()
                .all(|&v| v == <S as scioxide_core::AlgebraicScalar>::ZERO)
        {
            return self.solve_affine(problem, &hessian, session);
        }
        let has_constraints = problem.constraints().is_some();
        let has_bounds = problem.bounds().is_some_and(|bounds| {
            bounds
                .as_slice()
                .iter()
                .any(|(lower, upper)| lower.is_some() || upper.is_some())
        });
        if !has_constraints && !has_bounds {
            return self.solve_unconstrained(problem, &hessian, session, provider);
        }
        if self.method == super::QpMethod::PrimalDual {
            return super::primal_dual::solve(
                problem,
                &hessian,
                self.tolerances,
                self.max_iterations,
                self.warm_start.as_ref(),
                provider,
                session,
            );
        }
        if self.method == super::QpMethod::ActiveSet {
            return self.solve_active_set(problem, &hessian, active, session, provider);
        }
        if has_constraints {
            self.solve_active_set(problem, &hessian, active, session, provider)
        } else {
            self.solve_bound_constrained(problem, &hessian, session)
        }
    }

    fn validate_convex_hessian(
        &self,
        hessian: &[S],
        dimension: usize,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let matrix = Matrix::from_vec(
            dimension,
            dimension,
            try_copy_slice(hessian, scioxide_core::quantity!("qp_hessian_copy"))?,
        )?;
        let mut cholesky = scioxide_linalg::CholeskyDecomposition::empty();
        let mut cholesky_workspace = scioxide_linalg::LinalgWorkspace::new();
        if session.invoke(EvaluationRole::Factorization, || {
            if crate::dense::positive_gershgorin(&matrix)? {
                return Ok(true);
            }
            crate::dense::positive_definite(scioxide_linalg::CholeskyDecomposition::factor_into(
                &matrix,
                scioxide_linalg::HermitianInput::Exact,
                scioxide_linalg::CholeskyPivoting::None,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut cholesky,
                &mut cholesky_workspace,
                session.execution()?,
            ))
        })? {
            return Ok(());
        }
        let mut eigenvalues = HermitianEigenvaluesBuffer::empty();
        let mut workspace = scioxide_linalg::LinalgWorkspace::new();
        session.invoke(EvaluationRole::Factorization, || {
            HermitianEigenvaluesBuffer::compute_into(
                &matrix,
                scioxide_linalg::HermitianInput::Exact,
                &scioxide_linalg::HermitianEigenOptions::default(),
                &mut eigenvalues,
                &mut workspace,
                session.execution()?,
            )
        })?;
        let scale = hessian.iter().fold(
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            |scale, value| {
                let magnitude = NumericScalar::abs(*value);
                if magnitude > scale { magnitude } else { scale }
            },
        );
        let cutoff = self.tolerances.rank().threshold(scale)?;
        if eigenvalues
            .eigenvalues()
            .iter()
            .any(|&eigenvalue| eigenvalue < -cutoff)
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "the convex QP solver requires a positive-semidefinite Hessian",
            ));
        }
        Ok(())
    }
}
