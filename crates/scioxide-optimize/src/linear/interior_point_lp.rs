//! Homogeneous self-dual predictor-corrector method for linear programming.
//!
//! Solves linear programs of the form:
//!
//! ```text
//! min  c^T x
//! s.t. A_ub x <= b_ub
//!      A_eq x  = b_eq
//!      lo <= x <= hi
//! ```
//!
//! The algorithm converts to standard form (min c^T x, Ax = b, x >= 0),
//! then applies the Mehrotra predictor-corrector method to solve the
//! primal-dual system.

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

#[cfg(test)]
use scioxide_core::Tolerance;

use scioxide_core::backend::try_filled;
use scioxide_core::checked_float_quotient;
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};

use super::certificate;
use super::standard_form::{StandardForm, standardize, standardize_into};
use crate::KktProvider;
use crate::attempt::{Entry, Shell};
use crate::problem::LpProblem;
use crate::programming::{ProgrammingAccepted, ProgrammingProgress, ProgrammingResult};
use crate::result::OptimizeTermination;

// ---------------------------------------------------------------------------
// Solver struct
// ---------------------------------------------------------------------------

/// Standard form, embedding iterates and decoded scratch for one LP shape.
struct Prepared<S: RealScalar + NumericScalar<Real = S>> {
    form: StandardForm<S>,
    embedding: super::hsd::Embedding<S>,
    point: Vec<S>,
    rows: Vec<usize>,
    dual: Vec<S>,
    original: Vec<S>,
    #[cfg(test)]
    auxiliary_solves: usize,
}

/// Holds the reusable storage of [`InteriorPointLp`].
///
/// [`InteriorPointLp::workspace`] fixes the standard-form shape
/// `(columns, rows)` obtained after converting bounds. Each call converts its
/// problem into this storage and rejects a problem of another shape before any
/// factorization. Without an injected [`KktProvider`], the normal matrix of
/// order `rows` is factored by a built-in dense LDLᵀ factorization.
pub struct InteriorPointLpWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    prepared: Prepared<S>,
    shape: (usize, usize),
    provider: Option<Box<dyn KktProvider<S> + Send>>,
    staged: ProgrammingResult<S>,
    shell: Shell<ProgrammingProgress<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for InteriorPointLpWorkspace<S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("InteriorPointLpWorkspace")
            .field("shape", &self.shape)
            .field("injected_provider", &self.provider.is_some())
            .field("staged", &self.staged)
            .field("shell", &self.shell)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointLpWorkspace<S> {
    /// Returns the homogeneous embedding scalars `(τ, κ)` of the last accepted iterate.
    pub fn embedding_scalars(&self) -> (S, S) {
        (self.prepared.embedding.tau, self.prepared.embedding.kappa)
    }
    /// Returns the standard-form numbers of variables and equality rows.
    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }
    /// Replaces the built-in dense factorization with `provider` for every later call.
    ///
    /// The provider must accept the symmetric normal matrix `A diag(x/s) Aᵀ` of
    /// order `rows`, with a diagonal regularization relative to its scale.
    #[must_use]
    pub fn with_kkt_provider(mut self, provider: Box<dyn KktProvider<S> + Send>) -> Self {
        self.prepared.embedding.release_builtin();
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
    fn new(problem: &LpProblem<S>) -> Result<Self, SciOxideError> {
        let mut form = standardize(problem)?;
        form.equilibrate()?;
        let embedding = super::hsd::Embedding::new(&form)?;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let point = try_filled(
            form.columns,
            zero,
            scioxide_core::quantity!("hsd_decoded_point"),
        )?;
        let dual = try_filled(
            form.rows,
            zero,
            scioxide_core::quantity!("hsd_decoded_dual"),
        )?;
        let mut rows = scioxide_core::backend::try_with_capacity(
            form.rows,
            scioxide_core::quantity!("hsd_row_map"),
        )?;
        rows.extend(0..form.rows);
        let mut staged = ProgrammingResult::empty();
        staged.try_reserve(problem.dimension())?;
        Ok(Self {
            shape: (form.columns, form.rows),
            prepared: Prepared {
                form,
                embedding,
                point,
                rows,
                dual,
                original: try_filled(
                    problem.dimension(),
                    zero,
                    scioxide_core::quantity!("hsd_original_point"),
                )?,
                #[cfg(test)]
                auxiliary_solves: 0,
            },
            provider: None,
            staged,
            shell: Shell::new(),
        })
    }
}

/// Solves linear programs with a homogeneous self-dual interior-point method.
///
/// The problem is an [`LpProblem`], as for [`SimplexSolver`](crate::SimplexSolver).
/// It is converted to the standard form
///
/// ```text
/// minimize cᵀx   subject to   Ax = b,  x ≥ 0
/// ```
///
/// with `m` equality rows, and embedded in the homogeneous self-dual model,
/// whose scalars `τ` and `κ` detect infeasibility and unboundedness. Each
/// iteration takes a Mehrotra predictor-corrector step that keeps `x` and the
/// dual slacks `s = c − Aᵀy` positive. One factorization of the normal matrix
/// `A diag(x/s) Aᵀ` serves the predictor and the corrector, at O(m³) work per
/// iteration, and up to three refinement steps reduce the residual of the
/// original Newton equations.
///
/// The final iterate is refined on its active face to build the optimality
/// certificate. An `Unbounded` termination requires a feasible point and a
/// recession direction that pass a check against the original problem;
/// finding the feasible point may take one simplex solve, which counts toward
/// [`Self::max_iterations`]. A small `τ` alone never produces that
/// termination.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_optimize::{InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem};
///
/// // Minimize −x − 2y subject to x + y ≤ 4, x ≤ 3, y ≤ 2 and x, y ≥ 0.
/// // The optimum is x = y = 2 with objective −6.
/// let a = Matrix::from_vec(3, 2, vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0])?;
/// let senses = vec![LinearConstraintSense::LessEqual; 3];
/// let constraints = LinearConstraints::new(a, vec![4.0, 3.0, 2.0], senses)?;
/// let problem = LpProblem::new(vec![-1.0_f64, -2.0])?.with_constraints(constraints)?;
/// let result = InteriorPointLp::default().solve(&problem)?;
/// assert!(result.is_success());
/// assert!((result.fun() + 6.0).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Y. Ye, M. J. Todd and S. Mizuno, "An O(√n L)-iteration homogeneous and
///   self-dual linear programming algorithm", *Math. Oper. Res.* 19(1), 53–67,
///   1994.
/// - X. Xu, P.-F. Hung and Y. Ye, "A simplified homogeneous and self-dual
///   linear programming algorithm and its implementation", *Ann. Oper. Res.*
///   62, 151–171, 1996.
/// - S. Mehrotra, "On the implementation of a primal-dual interior point
///   method", *SIAM J. Optim.* 2(4), 575–601, 1992.
#[derive(Debug, Clone)]
pub struct InteriorPointLp<S: RealScalar> {
    tolerances: crate::OptimalityTolerances<S>,
    max_iterations: Budget,
}

impl<S: RealScalar> Default for InteriorPointLp<S> {
    fn default() -> Self {
        Self {
            tolerances: crate::OptimalityTolerances::programming_default(),
            max_iterations: Budget::ONE.saturating_add(999),
        }
    }
}

impl<S: RealScalar> InteriorPointLp<S> {
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

    /// Sets the maximum number of predictor-corrector iterations (default 1000).
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Returns the convergence tolerances.
    pub const fn tolerances(&self) -> crate::OptimalityTolerances<S> {
        self.tolerances
    }

    /// Returns the maximum number of predictor-corrector iterations.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> InteriorPointLp<S> {
    /// Creates a workspace for problems with the standard-form shape of `problem`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated, or an error if the problem cannot be equilibrated.
    pub fn workspace(
        &self,
        problem: &LpProblem<S>,
    ) -> Result<InteriorPointLpWorkspace<S>, SciOxideError> {
        InteriorPointLpWorkspace::new(problem)
    }

    /// Solves `problem` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace`] and [`Self::solve_into`].
    pub fn solve(&self, problem: &LpProblem<S>) -> Result<ProgrammingResult<S>, SciOxideError> {
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
    /// primal-dual iterate, `output` holds that iterate with an unsuccessful
    /// termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the standard-form shape of `problem` differs
    ///   from the one the workspace was created for.
    /// - `InvalidInput` if the problem dimensions are inconsistent.
    /// - `NumericalInstability` if the Newton equations of the embedding cannot
    ///   be solved.
    /// - Errors returned by an injected [`KktProvider`].
    pub fn solve_into<'scope, 'capability>(
        &self,
        problem: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        workspace: &mut InteriorPointLpWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(scioxide_core::AttemptKind::Call);
        let scope = scope.into();
        let form = &mut workspace.prepared.form;
        standardize_into(problem, form)?;
        form.equilibrate()?;
        if (form.columns, form.rows) != workspace.shape {
            return Err(SciOxideError::dimension_mismatch(
                [workspace.shape.0, workspace.shape.1],
                [form.columns, form.rows],
            ));
        }
        workspace.staged.try_reserve(problem.dimension())?;
        crate::storage::resize(
            &mut workspace.prepared.original,
            problem.dimension(),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        crate::attempt::preflight(&scope, None)?;
        let InteriorPointLpWorkspace {
            prepared,
            provider,
            staged,
            shell,
            ..
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
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                session.check()?;
                let provider = provider
                    .as_deref_mut()
                    .map(|provider| provider as &mut dyn KktProvider<S>);
                let mut result = if prepared.form.columns == 0 {
                    self.fixed_variables_only(&session, problem, &prepared.form)?
                } else if prepared.form.rows == 0 {
                    self.solve_unconstrained(&session, problem, &prepared.form)?
                } else {
                    self.solve_embedding(&session, problem, prepared, provider)?
                };
                result.algorithm = Some(session.algorithm()?);
                result.counts = session.accounting()?;
                result.trust = if result.is_success() {
                    scioxide_core::PartialResultTrust::ResidualChecked
                } else {
                    scioxide_core::PartialResultTrust::DiagnosticOnly
                };
                session.assess(result.trust)?;
                *staged = result;
                Ok(())
            },
            ProgrammingResult::progress,
        )
    }

    /// Internal identity of the method, dimension and immutable iteration policy.
    pub(crate) fn identity(&self, problem: &LpProblem<S>) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "InteriorPointLp", 1),
            problem.dimension(),
            self,
        )
    }

    fn fixed_variables_only(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        form: &StandardForm<S>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let rhs_scale = form.rhs.iter().fold(
            <S as scioxide_core::AlgebraicScalar>::ONE,
            |scale, value| {
                let magnitude = NumericScalar::abs(*value);
                if magnitude > scale { magnitude } else { scale }
            },
        );
        let threshold = self.tolerances.feasibility().threshold(rhs_scale)?;
        let feasible = form
            .rhs
            .iter()
            .all(|rhs| NumericScalar::abs(*rhs) <= threshold);
        let solution = form.reconstruct_solution(&[])?;
        let objective = objective_value(problem, &solution)?;
        self.result_with_fun(
            session,
            problem,
            solution,
            objective,
            0,
            if feasible {
                OptimizeTermination::Converged
            } else {
                OptimizeTermination::Infeasible
            },
            if feasible {
                "fixed-variable LP certified without complementarity division"
            } else {
                "fixed-variable LP violates a transformed constraint"
            },
        )
    }

    /// Handle unconstrained LP.
    #[allow(clippy::unnecessary_wraps)]
    fn solve_unconstrained(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        form: &StandardForm<S>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let standard = try_filled(
            form.columns,
            zero,
            scioxide_core::quantity!("unconstrained_lp_solution"),
        )?;
        let solution = form.reconstruct_solution(&standard)?;
        for (column, &coefficient) in form.objective.iter().enumerate() {
            if coefficient < zero {
                let mut result = self.result(
                    session,
                    problem,
                    solution,
                    0,
                    OptimizeTermination::Unbounded,
                    "problem is unbounded",
                )?;
                if session.stopped().is_some() {
                    return Ok(result);
                }
                let mut direction = try_filled(
                    form.columns,
                    zero,
                    scioxide_core::quantity!("unconstrained_hsd_ray"),
                )?;
                direction[column] = <S as scioxide_core::AlgebraicScalar>::ONE;
                let assessment = certificate::ray(
                    problem,
                    &result.x,
                    form.reconstruct_direction(&direction)?,
                    self.tolerances,
                    session,
                )?;
                result.termination = if assessment.accepted {
                    OptimizeTermination::Unbounded
                } else {
                    OptimizeTermination::SuspectedUnbounded
                };
                result.evidence = assessment.evidence;
                result.verification = assessment.report;
                result.diagnostics = assessment.diagnostics;
                return Ok(result);
            }
        }
        self.result(
            session,
            problem,
            solution,
            0,
            OptimizeTermination::Converged,
            "optimal solution at origin",
        )
    }

    /// Advance the homogeneous self-dual embedding with predictor-corrector steps.
    ///
    /// Standard form: min c^T x, Ax = b, x >= 0.
    /// Primal: x, dual: y (m-vector), slack: s = c - A^T y (n-vector).
    ///
    /// Normal factors cost O(m³) work, with O(mn + m²) storage.
    #[allow(
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::many_single_char_names,
        clippy::cast_precision_loss
    )]
    fn solve_embedding(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        prepared: &mut Prepared<S>,
        mut provider: Option<&mut dyn KktProvider<S>>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let form = &prepared.form;
        let embedding = &mut prepared.embedding;
        embedding.reset();
        let point = &mut prepared.point;
        let rows = &prepared.rows;
        let row_dual = &mut prepared.dual;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let x = &mut prepared.original;
        #[cfg(test)]
        {
            prepared.auxiliary_solves = 0;
        }
        let mut iteration = 0;
        let recovery_iteration = loop {
            embedding.residuals(form)?;
            for (x, &v) in point.iter_mut().zip(&embedding.x) {
                *x = checked_float_quotient(v, embedding.tau)?;
            }
            form.reconstruct_solution_into(point, x)?;
            let primal_scale = form
                .rhs
                .iter()
                .fold(<S as scioxide_core::AlgebraicScalar>::ONE, |v, b| {
                    v.max(NumericScalar::abs(*b))
                });
            let dual_scale = form
                .objective
                .iter()
                .fold(<S as scioxide_core::AlgebraicScalar>::ONE, |v, c| {
                    v.max(NumericScalar::abs(*c))
                });
            let primal = inf_norm(&embedding.rp)? / embedding.tau;
            let dual = inf_norm(&embedding.rd)? / embedding.tau;
            let gap = NumericScalar::abs(embedding.gap - embedding.kappa) / embedding.tau;
            if primal <= self.tolerances.feasibility().threshold(primal_scale)?
                && dual <= self.tolerances.dual_feasibility().threshold(dual_scale)?
                && gap <= self.tolerances.gap().threshold(dual_scale)?
            {
                for (d, &v) in row_dual.iter_mut().zip(&embedding.y) {
                    *d = checked_float_quotient(v, embedding.tau)?;
                }
                let dual = super::simplex::basis_dual(problem, form, rows, row_dual, true)?;
                let mut assessment = certificate::assess(
                    problem,
                    x,
                    self.tolerances,
                    OptimizeTermination::Converged,
                    Some(dual),
                    session,
                )?;
                if !assessment.accepted {
                    // Interior iterates approach active nonnegative bounds
                    // asymptotically. Try the corresponding face as a separate
                    // candidate; only original-model verification admits it.
                    let cutoff = self.tolerances.feasibility().threshold(inf_norm(point)?)?;
                    for value in point.iter_mut() {
                        if *value <= cutoff {
                            *value = zero;
                        }
                    }
                    let polishing = super::hsd::polish_point(
                        form,
                        point,
                        scioxide_linalg::RankPolicy::Threshold(self.tolerances.rank()),
                        session,
                    );
                    if let Err(error) = polishing {
                        for (target, &value) in point.iter_mut().zip(&embedding.x) {
                            *target = checked_float_quotient(value, embedding.tau)?;
                        }
                        form.reconstruct_solution_into(point, x)?;
                        return match session.stopped() {
                            Some(stop) => self.result(
                                session,
                                problem,
                                x.clone(),
                                iteration,
                                stop,
                                "HSD resource limit during active-face preparation",
                            ),
                            None => Err(error),
                        };
                    }
                    form.reconstruct_solution_into(point, x)?;
                    let dual = super::simplex::basis_dual(problem, form, rows, row_dual, true)?;
                    assessment = certificate::assess(
                        problem,
                        x,
                        self.tolerances,
                        OptimizeTermination::Converged,
                        Some(dual),
                        session,
                    )?;
                }
                if assessment.accepted {
                    return self.verified_result(
                        session,
                        problem,
                        x.clone(),
                        iteration,
                        OptimizeTermination::Converged,
                        "HSD verified optimum",
                        assessment,
                    );
                }
                // Rejected certificate candidates never replace the accepted
                // embedding used for a later resource stop.
                for (target, &value) in point.iter_mut().zip(&embedding.x) {
                    *target = checked_float_quotient(value, embedding.tau)?;
                }
                form.reconstruct_solution_into(point, x)?;
            }
            if embedding.tau < self.tolerances.gap().threshold(embedding.kappa)? {
                let dual = super::simplex::basis_dual(problem, form, rows, &embedding.y, false)?;
                let assessment = certificate::assess(
                    problem,
                    x,
                    self.tolerances,
                    OptimizeTermination::Infeasible,
                    Some(dual),
                    session,
                )?;
                if assessment.accepted {
                    return self.verified_result(
                        session,
                        problem,
                        x.clone(),
                        iteration,
                        OptimizeTermination::Infeasible,
                        "HSD verified Farkas witness",
                        assessment,
                    );
                }
                if let Some(stop) = session.stopped() {
                    return self.result(
                        session,
                        problem,
                        x.clone(),
                        iteration,
                        stop,
                        "HSD resource limit during certificate recovery",
                    );
                }
                // Require descent and a small homogeneous equality defect
                // before requesting the single feasibility recovery.
                let norm = inf_norm(&embedding.x)?;
                if crate::utils::dot(&form.objective, &embedding.x)? < zero
                    && inf_norm(&embedding.rp)? <= self.tolerances.feasibility().threshold(norm)?
                {
                    break iteration;
                }
            }
            if iteration == self.max_iterations.get() {
                return self.result(
                    session,
                    problem,
                    x.clone(),
                    iteration,
                    OptimizeTermination::IterationLimit,
                    "HSD iteration limit",
                );
            }
            if let Some(stop) = session.trial_iteration()? {
                return self.result(
                    session,
                    problem,
                    x.clone(),
                    iteration,
                    stop,
                    "HSD resource limit",
                );
            }
            if let Err(error) = embedding.advance(form, session, crate::kkt::lend(&mut provider)) {
                return match session.stopped() {
                    Some(stop) => self.result(
                        session,
                        problem,
                        x.clone(),
                        iteration,
                        stop,
                        "HSD resource limit retained last accepted embedding",
                    ),
                    None => Err(error),
                };
            }
            iteration += 1;
        };
        // Exactly one auxiliary solve is reachable: recovery is outside the loop.
        #[cfg(test)]
        {
            prepared.auxiliary_solves += 1;
        }
        let mut feasibility = LpProblem::new(try_filled(
            problem.dimension(),
            zero,
            scioxide_core::quantity!("hsd_feasible_base_objective"),
        )?)?;
        if let Some(constraints) = problem.constraints() {
            feasibility = feasibility.with_constraints(constraints.clone())?;
        }
        if let Some(bounds) = problem.bounds() {
            feasibility = feasibility.with_bounds(bounds.clone())?;
        }
        let candidate = form.reconstruct_direction(&embedding.x)?;
        let solver = super::simplex::SimplexSolver::default()
            .with_tolerances(self.tolerances)
            .with_max_iterations(self.max_iterations);
        let mut state = scioxide_core::SolveState::new();
        let recovery = session.with_child(
            &mut state,
            solver.identity(&feasibility),
            scioxide_core::SolveBudgets::unbounded()
                .with_limit(scioxide_core::BudgetKind::Iteration, self.max_iterations),
            |context| {
                let child =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                let result = solver.feasible_ray(&feasibility, &candidate, &child);
                Ok((result, child.stopped()))
            },
        )?;
        session.inherit_child_stop(recovery.1)?;
        let (base, ray) = match recovery.0 {
            Ok(result) => result,
            Err(error) => {
                return match session.stopped() {
                    Some(stop) => self.result(
                        session,
                        problem,
                        x.clone(),
                        recovery_iteration,
                        stop,
                        "HSD resource limit retained embedding during recovery",
                    ),
                    None => Err(error),
                };
            }
        };
        if !base.is_success() && session.stopped().is_none() {
            return self.result(
                session,
                problem,
                x.clone(),
                recovery_iteration,
                OptimizeTermination::SuspectedUnbounded,
                "HSD feasibility recovery was inconclusive",
            );
        }
        if base.is_success() {
            let assessment =
                match certificate::ray(problem, base.x(), ray, self.tolerances, session) {
                    Ok(assessment) => assessment,
                    Err(error) => {
                        return match session.stopped() {
                            Some(stop) => self.result(
                                session,
                                problem,
                                base.x().to_vec(),
                                recovery_iteration,
                                stop,
                                "HSD resource limit during ray verification",
                            ),
                            None => Err(error),
                        };
                    }
                };
            return self.verified_result(
                session,
                problem,
                base.x().to_vec(),
                recovery_iteration,
                if assessment.accepted {
                    OptimizeTermination::Unbounded
                } else {
                    OptimizeTermination::SuspectedUnbounded
                },
                "HSD feasible base and recession candidate",
                assessment,
            );
        }
        self.result(
            session,
            problem,
            x.clone(),
            recovery_iteration,
            session
                .stopped()
                .unwrap_or(OptimizeTermination::SuspectedUnbounded),
            "HSD feasibility recovery did not verify unboundedness",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn verified_result(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        x: Vec<S>,
        iterations: usize,
        termination: OptimizeTermination,
        message: &str,
        assessment: crate::verification::candidate::Assessment<S>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        Ok(ProgrammingResult {
            fun: objective_value(problem, &x)?,
            x,
            iterations,
            termination: session.stopped().unwrap_or(termination),
            message: message.to_owned(),
            evidence: assessment.evidence,
            verification: assessment.report,
            diagnostics: assessment.diagnostics,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn result(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        x: Vec<S>,
        iterations: usize,
        termination: OptimizeTermination,
        message: &str,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let fun = objective_value(problem, &x)?;
        self.result_with_fun(session, problem, x, fun, iterations, termination, message)
    }

    #[allow(clippy::too_many_arguments)]
    fn result_with_fun(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        x: Vec<S>,
        fun: S,
        iterations: usize,
        termination: OptimizeTermination,
        message: &str,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let assessment =
            certificate::assess(problem, &x, self.tolerances, termination, None, session)?;
        let diagnostics = assessment.diagnostics;
        let certified = assessment.accepted;
        let termination = session.stopped().unwrap_or(termination);
        let requested_success = matches!(termination, OptimizeTermination::Converged);
        Ok(ProgrammingResult {
            evidence: assessment.evidence,
            verification: assessment.report,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            diagnostics,
            x,
            fun,
            iterations,
            termination: if requested_success && !certified {
                OptimizeTermination::Stagnation
            } else {
                match termination {
                    OptimizeTermination::Infeasible if !certified => {
                        OptimizeTermination::SuspectedInfeasible
                    }
                    OptimizeTermination::Unbounded => OptimizeTermination::SuspectedUnbounded,
                    other => other,
                }
            },
            message: if requested_success && !certified {
                format!("{message}; independent certificate failed")
            } else {
                message.to_owned()
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Infinity norm of a vector.
fn inf_norm<S: RealScalar + NumericScalar<Real = S>>(v: &[S]) -> Result<S, SciOxideError> {
    if v.iter().any(|value| !value.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "residual vector contains a non-finite value",
        ));
    }
    Ok(v.iter()
        .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |acc, &vi| {
            acc.max(NumericScalar::abs(vi))
        }))
}

/// Compute primal residual `r_p = b - A x`.
#[allow(clippy::many_single_char_names)]
fn objective_value<S: RealScalar + NumericScalar<Real = S>>(
    problem: &LpProblem<S>,
    solution: &[S],
) -> Result<S, SciOxideError> {
    crate::utils::dot(problem.objective(), solution)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::problem::{LinearConstraintSense, LinearConstraints};
    use scioxide_linalg::Matrix;

    #[test]
    fn auxiliary_feasibility_is_once_per_call_and_absent_on_ordinary_paths() {
        let solver = InteriorPointLp::default();
        for (lp, expected, calls) in [
            (
                problem(
                    vec![-1.0_f64],
                    vec![1.0],
                    vec![2.0],
                    vec![LinearConstraintSense::LessEqual],
                ),
                OptimizeTermination::Converged,
                0,
            ),
            (
                problem(
                    vec![1.0_f64],
                    vec![1.0, -1.0],
                    vec![1.0, -3.0],
                    vec![LinearConstraintSense::LessEqual; 2],
                ),
                OptimizeTermination::Infeasible,
                0,
            ),
            (
                problem(
                    vec![-1.0_f64, 0.0],
                    vec![1.0, -1.0],
                    vec![1.0],
                    vec![LinearConstraintSense::Equal],
                ),
                OptimizeTermination::Unbounded,
                1,
            ),
        ] {
            let mut workspace = solver.workspace(&lp).unwrap();
            let mut result = ProgrammingResult::empty();
            for _ in 0..2 {
                solver
                    .solve_into(
                        &lp,
                        &mut result,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert_eq!(result.termination(), expected);
                assert!(result.verification_report().unwrap().is_accepted());
                assert_eq!(workspace.prepared.auxiliary_solves, calls);
            }
        }
    }

    fn problem<S>(
        objective: Vec<S>,
        coefficients: Vec<S>,
        rhs: Vec<S>,
        senses: Vec<LinearConstraintSense>,
    ) -> LpProblem<S>
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        let variables = objective.len();
        let rows = rhs.len();
        LpProblem::new(objective)
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(rows, variables, coefficients).unwrap(),
                    rhs,
                    senses,
                )
                .unwrap(),
            )
            .unwrap()
    }

    // ---- Basic LP ----

    /// min -x - 2y
    /// s.t. x + y <= 4
    ///      x     <= 3
    ///          y <= 2
    /// Solution: x=2, y=2, fun=-6
    #[test]
    fn test_interior_point_basic_lp() {
        let problem = problem(
            vec![-1.0_f64, -2.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![4.0, 3.0, 2.0],
            vec![LinearConstraintSense::LessEqual; 3],
        );

        let solver = InteriorPointLp::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(
            result.is_success(),
            "should converge; got: {}",
            result.message
        );
        assert_eq!(result.termination, OptimizeTermination::Converged);
        assert!(
            (result.fun - (-6.0)).abs() < 1e-4,
            "fun={}, expected -6",
            result.fun
        );
        assert!(
            (result.x[0] - 2.0).abs() < 1e-3,
            "x0={}, expected 2",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 1e-3,
            "x1={}, expected 2",
            result.x[1]
        );
    }

    // ---- Equality constraints ----

    /// min -x - y
    /// s.t. x + y = 3
    ///      x <= 2
    ///      y <= 2
    #[test]
    fn test_interior_point_equality_constraint() {
        let problem = problem(
            vec![-1.0_f64, -1.0],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 3.0],
            vec![
                LinearConstraintSense::LessEqual,
                LinearConstraintSense::LessEqual,
                LinearConstraintSense::Equal,
            ],
        );

        let solver = InteriorPointLp::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.fun - (-3.0)).abs() < 1e-3);
        assert!((result.x[0] + result.x[1] - 3.0).abs() < 1e-3);
    }

    // ---- Default ----

    #[test]
    fn test_interior_point_default() {
        let solver = InteriorPointLp::<f64>::default();
        assert_eq!(solver.max_iterations.get(), 1000);
    }

    // ---- Another simple LP ----

    /// min -2x - 3y
    /// s.t.  x +  y <= 4
    ///       x + 3y <= 6
    /// Solution: at (3,1), fun = -9
    #[test]
    fn test_interior_point_another_lp() {
        let problem = problem(
            vec![-2.0_f64, -3.0],
            vec![1.0, 1.0, 1.0, 3.0],
            vec![4.0, 6.0],
            vec![LinearConstraintSense::LessEqual; 2],
        );

        let solver = InteriorPointLp::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!(
            (result.fun - (-9.0)).abs() < 1e-3,
            "fun={}, expected -9",
            result.fun
        );
    }

    // ---- f32 ----

    #[test]
    fn test_interior_point_f32() {
        let problem = problem(
            vec![-1.0_f32, -2.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![4.0, 3.0, 2.0],
            vec![LinearConstraintSense::LessEqual; 3],
        );

        let solver = InteriorPointLp {
            tolerances: crate::OptimalityTolerances::new(
                scioxide_core::FeasibilityTolerance::from_tolerance(
                    Tolerance::from_decimal_places(6, 6),
                ),
                scioxide_core::GradientTolerance::from_tolerance(Tolerance::from_decimal_places(
                    6, 6,
                )),
                scioxide_core::ResidualTolerance::from_tolerance(Tolerance::from_decimal_places(
                    6, 6,
                )),
                scioxide_core::RankThreshold::from_tolerance(Tolerance::from_decimal_places(6, 6)),
            ),
            ..InteriorPointLp::default()
        };
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.fun - (-6.0_f32)).abs() < 0.1);
    }

    #[test]
    fn fixed_variable_lp_avoids_zero_column_complementarity_division() {
        let constraints = LinearConstraints::new(
            Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap(),
            vec![2.0],
            vec![LinearConstraintSense::Equal],
        )
        .unwrap();
        let problem = LpProblem::new(vec![3.0])
            .unwrap()
            .with_constraints(constraints)
            .unwrap()
            .with_bounds(scioxide_core::Bounds::new(vec![(Some(2.0), Some(2.0))]).unwrap())
            .unwrap();

        let result = InteriorPointLp::default().solve(&problem).unwrap();

        assert!(result.is_success(), "{}", result.message());
        assert_eq!(result.x(), &[2.0]);
        assert_eq!(result.fun(), 6.0);
        assert_eq!(result.certified_optimality_gap(), None);
        assert!(
            result
                .verification_report()
                .unwrap()
                .objective_gap()
                .unwrap()
                .abs()
                < 1e-12
        );
    }
}
