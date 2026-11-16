//! Bounded two-phase revised simplex with Dantzig pricing and Bland fallback.
//!
//! Solves linear programs of the form:
//!
//! ```text
//! min  c^T x
//! s.t. A_ub x <= b_ub
//!      A_eq x  = b_eq
//!      lo <= x <= hi
//! ```

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

#[cfg(test)]
use scioxide_core::Tolerance;

use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{checked_difference, checked_float_quotient};

#[path = "reoptimization.rs"]
mod reoptimization;

use super::basis::BasisProvider;
use super::certificate;
use super::standard_form::{StandardForm, standardize_bounded};
#[path = "pivot.rs"]
mod pivot;
use crate::attempt::{Entry, Shell};
use crate::problem::{LinearConstraintSense, LpProblem};
use crate::programming::{ProgrammingAccepted, ProgrammingProgress, ProgrammingResult};
use crate::result::OptimizeTermination;

// ---------------------------------------------------------------------------
// Solver struct
// ---------------------------------------------------------------------------

/// Stores a simplex basis for warm-starting [`SimplexSolver::resume_into`].
///
/// Obtain one from [`SimplexWorkspace::warm_start`] after a solve. A resume
/// starts from the stored basis when only the objective or the right-hand side
/// changed, or when inequality rows were appended; dual simplex then restores
/// primal feasibility. When coefficients, row senses, variable bounds or the
/// shape changed, the resume solves from scratch. [`Self::reused_basis`] reports
/// which of the two the recording attempt did.
#[derive(Debug)]
pub struct SimplexWarmStart<S: RealScalar + NumericScalar<Real = S>> {
    structure: Option<WarmStructure<S>>,
    record: bool,
    reused: bool,
    indices: Vec<usize>,
    rows: Vec<usize>,
    upper: Vec<bool>,
}
#[derive(Debug)]
struct WarmStructure<S: RealScalar + NumericScalar<Real = S>> {
    values: Vec<S>,
    columns: Vec<usize>,
    offsets: Vec<usize>,
    signs: Vec<S>,
    bounds: Option<scioxide_core::Bounds<S>>,
    dimension: usize,
    input_rows: usize,
    transformed_variables: usize,
    row_exponents: Vec<i32>,
    column_exponents: Vec<i32>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for SimplexWarmStart<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SimplexWarmStart<S> {
    /// Creates an empty snapshot; a resume from it solves from scratch (Phase I and II).
    pub const fn new() -> Self {
        Self {
            structure: None,
            record: true,
            reused: false,
            indices: Vec::new(),
            rows: Vec::new(),
            upper: Vec::new(),
        }
    }
    /// Returns the structural column indices of the stored feasible basis.
    pub fn basis_indices(&self) -> &[usize] {
        &self.indices
    }
    pub(super) fn retained_rows(&self) -> &[usize] {
        &self.rows
    }
    /// Discards the stored basis and its reuse report.
    pub fn clear(&mut self) {
        self.structure = None;
        self.reused = false;
        self.indices.clear();
        self.rows.clear();
        self.upper.clear();
    }
    fn save(
        &mut self,
        p: &LpProblem<S>,
        f: &StandardForm<S>,
        basis: &super::basis::Basis<'_, S>,
        upper: &[bool],
    ) -> Result<(), SciOxideError> {
        if !self.record {
            return Ok(());
        }
        use scioxide_core::backend::{try_copy_slice, try_reserve};
        let structure = if self.structure.is_none() {
            Some(WarmStructure {
                values: try_copy_slice(&f.coefficients, scioxide_core::quantity!("warm_matrix"))?,
                columns: try_copy_slice(
                    &f.column_indices,
                    scioxide_core::quantity!("warm_columns"),
                )?,
                offsets: try_copy_slice(&f.row_offsets, scioxide_core::quantity!("warm_offsets"))?,
                signs: try_copy_slice(&f.row_signs, scioxide_core::quantity!("warm_signs"))?,
                bounds: p
                    .bounds()
                    .map(|b| {
                        scioxide_core::Bounds::new(try_copy_slice(
                            b.as_slice(),
                            scioxide_core::quantity!("warm_bounds"),
                        )?)
                    })
                    .transpose()?,
                dimension: p.dimension(),
                input_rows: p.constraints().map_or(0, |c| c.len()),
                transformed_variables: f.transformed_variables,
                row_exponents: try_copy_slice(
                    &f.row_exponents,
                    scioxide_core::quantity!("warm_row_scales"),
                )?,
                column_exponents: try_copy_slice(
                    &f.column_exponents,
                    scioxide_core::quantity!("warm_column_scales"),
                )?,
            })
        } else {
            None
        };
        let extra = basis.indices.len().saturating_sub(self.indices.len());
        try_reserve(
            &mut self.indices,
            extra,
            scioxide_core::quantity!("warm_basis"),
        )?;
        let extra = basis.rows.len().saturating_sub(self.rows.len());
        try_reserve(&mut self.rows, extra, scioxide_core::quantity!("warm_rows"))?;
        let extra = upper.len().saturating_sub(self.upper.len());
        try_reserve(
            &mut self.upper,
            extra,
            scioxide_core::quantity!("warm_bound_states"),
        )?;
        if let Some(structure) = structure {
            self.structure = Some(structure);
        }
        self.upper.clear();
        self.upper.extend_from_slice(upper);
        self.indices.clear();
        self.indices.extend_from_slice(&basis.indices);
        self.rows.clear();
        self.rows.extend_from_slice(&basis.rows);
        Ok(())
    }
}

/// Solves linear programs with the bounded two-phase revised simplex method.
///
/// The problem is an [`LpProblem`]:
///
/// ```text
/// minimize    cᵀx
/// subject to  aᵢᵀx ≤ bᵢ,  aᵢᵀx = bᵢ  or  aᵢᵀx ≥ bᵢ   (one sense per row i)
///             l ≤ x ≤ u
/// ```
///
/// where `c` is the objective, `aᵢᵀ` is row `i` of the constraint matrix, `b`
/// is the right-hand side, and `l`, `u` are the variable bounds (`x ≥ 0` when
/// no bounds are attached). Phase I finds a basic feasible solution and Phase
/// II minimizes the objective. Finite variable bounds are handled directly
/// rather than as extra rows. Pricing uses Dantzig's rule, and a two-pass
/// Harris ratio test chooses large pivots. After eight consecutive pivots
/// without progress, Bland's rule prevents cycling for the rest of the phase.
/// The basis is factored by dense or sparse LU, with up to
/// [`Self::with_basis_update_limit`] eta updates between refactorizations.
///
/// The result carries a certificate in original coordinates: multipliers for an
/// optimum, Farkas multipliers for an infeasible problem or a recession
/// direction for an unbounded one; see [`ProgrammingResult::certificate`].
/// To reoptimize after a small change to the problem, use
/// [`Self::resume_into`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_optimize::{LinearConstraints, LpProblem};
/// let a = Matrix::from_rows(&[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])?;
/// let constraints = LinearConstraints::less_equal(a, vec![4.0, 3.0, 2.0])?;
/// let problem = LpProblem::new(vec![-1.0_f64, -2.0])?.with_constraints(constraints)?;
/// let result = problem.solve()?;
/// assert!((result.fun() + 6.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - V. Chvátal, *Linear Programming*, W. H. Freeman, 1983, ch. 7–8 (revised
///   simplex and bounded variables).
/// - P. M. J. Harris, "Pivot selection methods of the Devex LP code",
///   *Math. Programming* 5, 1–28, 1973.
/// - R. G. Bland, "New finite pivoting rules for the simplex method",
///   *Math. Oper. Res.* 2(2), 103–107, 1977.
#[derive(Debug, Clone)]
pub struct SimplexSolver<S: RealScalar> {
    tolerances: crate::OptimalityTolerances<S>,
    max_iterations: Option<Budget>,
    basis_updates: Budget,
    sparse_limits: Option<scioxide_sparse::SparseLuLimits>,
}

impl<S: RealScalar> Default for SimplexSolver<S> {
    fn default() -> Self {
        Self {
            tolerances: crate::OptimalityTolerances::programming_default(),
            max_iterations: None,
            basis_updates: Budget::ONE.saturating_add(31),
            sparse_limits: None,
        }
    }
}

impl<S: RealScalar> SimplexSolver<S> {
    pub(crate) fn basis_update_limit(&self) -> usize {
        self.basis_updates.get()
    }
    /// Sets the number of eta updates between full refactorizations of the basis (default 32).
    pub const fn with_basis_update_limit(mut self, limit: Budget) -> Self {
        self.basis_updates = limit;
        self
    }
    /// Sets the fill and workspace limits of the sparse LU basis factorization.
    pub const fn with_sparse_limits(mut self, limits: scioxide_sparse::SparseLuLimits) -> Self {
        self.sparse_limits = Some(limits);
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

    /// Replaces the default limit on pivots and bound flips; see [`Self::max_iterations`].
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Returns the convergence tolerances.
    pub const fn tolerances(&self) -> crate::OptimalityTolerances<S> {
        self.tolerances
    }

    /// Returns the limit on pivots and bound flips for `problem`.
    ///
    /// The default is `max(1000, 10 (n + m))`, where `n` is the number of
    /// variables and `m` the number of constraint rows, computed with saturating
    /// arithmetic. [`Self::with_max_iterations`] replaces it.
    pub fn max_iterations(&self, problem: &LpProblem<S>) -> Budget {
        self.max_iterations.unwrap_or_else(|| {
            let size = problem
                .dimension()
                .saturating_add(problem.constraints().map_or(0, |c| c.len()));
            Budget::ONE.saturating_add(size.saturating_mul(10).max(1000) - 1)
        })
    }
}

/// Holds the reusable storage of [`SimplexSolver`].
///
/// The workspace keeps the basis of the most recent attempt, an optional
/// [`BasisProvider`] and the record of the last attempt. Without a provider,
/// each call factors the basis with a built-in dense or sparse LU.
pub struct SimplexWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    warm: SimplexWarmStart<S>,
    provider: Option<Box<dyn BasisProvider<S> + Send>>,
    staged: ProgrammingResult<S>,
    shell: Shell<ProgrammingProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for SimplexWorkspace<S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SimplexWorkspace")
            .field("warm", &self.warm)
            .field("injected_provider", &self.provider.is_some())
            .field("staged", &self.staged)
            .field("shell", &self.shell)
            .finish()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> SimplexWorkspace<S> {
    /// Replaces the built-in basis factorization with `provider` for every later call.
    #[must_use]
    pub fn with_basis_provider(mut self, provider: Box<dyn BasisProvider<S> + Send>) -> Self {
        self.provider = Some(provider);
        self
    }
    /// Returns the record of the most recent call or resume.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<ProgrammingProgress<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the point written by the most recent attempt, if any.
    pub fn last_accepted(&self) -> Option<ProgrammingAccepted<'_, S>> {
        crate::programming::last_accepted(&self.shell, &self.staged.x)
    }
    /// Copies the basis of the most recent attempt for [`SimplexSolver::resume_into`].
    ///
    /// The copy is empty, so a resume from it solves from scratch, when that
    /// attempt found no feasible basis, for example because it failed.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the copy cannot be allocated. The
    /// workspace is not changed.
    pub fn warm_start(&self) -> Result<SimplexWarmStart<S>, SciOxideError> {
        let mut copy = self.warm.try_clone()?;
        copy.record = true;
        Ok(copy)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> SimplexSolver<S> {
    /// Creates a workspace for `problem`.
    ///
    /// The workspace accepts any later LP; factorization storage is sized on
    /// each call.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn workspace(&self, problem: &LpProblem<S>) -> Result<SimplexWorkspace<S>, SciOxideError> {
        let mut staged = ProgrammingResult::empty();
        staged.try_reserve(problem.dimension())?;
        Ok(SimplexWorkspace {
            warm: SimplexWarmStart::new(),
            provider: None,
            staged,
            shell: Shell::new(),
        })
    }

    /// Solves `problem` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_into`].
    pub fn solve(&self, problem: &LpProblem<S>) -> Result<ProgrammingResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        workspace.warm.record = false;
        let mut output = ProgrammingResult::empty();
        self.solve_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Solves `problem` from scratch into `output`, reusing `workspace`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent solve context. The limit on pivots and bound flips is
    /// [`Self::max_iterations`]. If a limit, cancellation or deadline stops the
    /// method after it has a feasible basis, `output` holds that point with an
    /// unsuccessful termination. The final basis is stored for
    /// [`SimplexWorkspace::warm_start`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the problem dimensions are inconsistent.
    /// - `NumericalInstability` if pivoting produces a non-finite value or the
    ///   basis cannot be factored accurately.
    /// - `AllocationFailure` if factorization storage cannot be allocated.
    /// - Errors returned by an injected [`BasisProvider`].
    pub fn solve_into<'scope, 'capability>(
        &self,
        problem: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        workspace: &mut SimplexWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(scioxide_core::AttemptKind::Call);
        workspace.warm.clear();
        let scope = scope.into();
        workspace.staged.try_reserve(problem.dimension())?;
        crate::attempt::preflight(&scope, None)?;
        self.attempt(
            problem,
            output,
            workspace,
            scope,
            scioxide_core::AttemptKind::Call,
        )
    }

    /// Reoptimizes `problem` from the basis in `warm` into `output`.
    ///
    /// If only the objective or the right-hand side changed, or inequality rows
    /// were appended, the solve starts from the stored basis. If that basis is
    /// primal infeasible, dual simplex restores feasibility or returns a Farkas
    /// certificate of infeasibility. If coefficients, row senses, variable
    /// bounds or the shape changed, the solve starts from scratch, so an
    /// outdated snapshot never causes an error. `warm` is not modified; read the
    /// new basis and [`SimplexWarmStart::reused_basis`] from
    /// [`SimplexWorkspace::warm_start`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_into`], and `AllocationFailure` if
    /// the snapshot cannot be copied.
    pub fn resume_into<'scope, 'capability>(
        &self,
        problem: &LpProblem<S>,
        warm: &SimplexWarmStart<S>,
        output: &mut ProgrammingResult<S>,
        workspace: &mut SimplexWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(scioxide_core::AttemptKind::Resume);
        workspace.warm.clear();
        let scope = scope.into();
        let mut snapshot = warm.try_clone()?;
        snapshot.record = true;
        workspace.staged.try_reserve(problem.dimension())?;
        crate::attempt::preflight(&scope, None)?;
        workspace.warm = snapshot;
        self.attempt(
            problem,
            output,
            workspace,
            scope,
            scioxide_core::AttemptKind::Resume,
        )
    }

    fn attempt(
        &self,
        problem: &LpProblem<S>,
        output: &mut ProgrammingResult<S>,
        workspace: &mut SimplexWorkspace<S>,
        scope: scioxide_core::SolveScope<'_, '_>,
        kind: scioxide_core::AttemptKind,
    ) -> Result<(), SciOxideError> {
        let SimplexWorkspace {
            warm,
            provider,
            staged,
            shell,
        } = workspace;
        let outcome = crate::programming::run(
            shell,
            staged,
            output,
            scope,
            kind,
            Entry::Call {
                identity: self.identity(problem),
                budgets: self.budgets(problem),
            },
            |context, staged| {
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                *staged = self.reoptimize_observed(
                    problem,
                    &session,
                    warm,
                    provider
                        .as_deref_mut()
                        .map(|provider| provider as &mut (dyn BasisProvider<S> + Send)),
                )?;
                Ok(())
            },
            ProgrammingResult::progress,
        );
        if outcome.is_err() {
            warm.clear();
        }
        outcome
    }

    /// Internal identity of the method, dimension and immutable pivot policy.
    pub(crate) fn identity(&self, problem: &LpProblem<S>) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "Simplex", 1),
            problem.dimension(),
            self,
        )
    }

    fn budgets(&self, problem: &LpProblem<S>) -> scioxide_core::SolveBudgets {
        scioxide_core::SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Iteration,
            self.max_iterations(problem),
        )
    }

    /// Reoptimize `warm` in an already-open (child) epoch without identity checks.
    pub(crate) fn reoptimize_in_context(
        &self,
        problem: &LpProblem<S>,
        warm: &mut SimplexWarmStart<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let session =
            crate::context::SolveSession::new(context, scioxide_core::argument!("context"));
        self.reoptimize_observed(problem, &session, warm, None)
    }

    fn reoptimize_observed(
        &self,
        problem: &LpProblem<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
        warm: &mut SimplexWarmStart<S>,
        provider: Option<&mut (dyn BasisProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        warm.reused = false;
        self.solve_observed_warm(problem, session, warm, provider)
    }

    pub(crate) fn solve_observed(
        &self,
        problem: &LpProblem<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let mut warm = SimplexWarmStart::new();
        warm.record = false;
        self.solve_observed_warm(problem, session, &mut warm, None)
    }

    /// One bounded feasibility solve, then a basis solve to clean an HSD ray.
    /// No objective optimization or repeated feasibility solve is performed.
    pub(super) fn feasible_ray(
        &self,
        problem: &LpProblem<S>,
        candidate: &[S],
        session: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<(ProgrammingResult<S>, Vec<S>), SciOxideError> {
        let mut warm = SimplexWarmStart::new();
        let result = self.solve_observed_warm(problem, session, &mut warm, None)?;
        if !result.is_success() {
            return Ok((result, Vec::new()));
        }
        let mut form = standardize_bounded(problem)?;
        form.equilibrate()?;
        let mut ray = form.encode_direction(candidate)?;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let scale = ray.iter().fold(zero, |a, &v| a.max(v.abs()));
        let threshold = RealScalar::sqrt(scioxide_core::checked_float_product(
            self.tolerances.feasibility().threshold(scale)?,
            scale,
        )?);
        // Finite two-sided variables cannot move along a recession direction.
        // Small homogeneous components are support candidates, never evidence.
        for (j, value) in ray.iter_mut().enumerate() {
            if value.abs() <= threshold || form.upper_bounds[j].is_some() {
                *value = zero;
            }
        }
        if form.rows != 0 && !warm.indices.is_empty() {
            let mut basis = super::basis::Basis::new(&form, 1, None, None)?;
            basis.indices = warm.indices;
            basis.rows = warm.rows;
            basis.artificial_rows.clear();
            basis.refactor(&form, session)?;
            for &j in &basis.indices {
                ray[j] = zero;
            }
            let rhs = basis
                .rows
                .iter()
                .map(|&i| Ok(-form.row_dot(i, &ray)?))
                .collect::<Result<Vec<_>, SciOxideError>>()?;
            let mut values =
                try_filled(rhs.len(), zero, scioxide_core::quantity!("hsd_ray_basis"))?;
            basis.solve_checked(&form, &rhs, &mut values, false, session)?;
            for (&j, value) in basis.indices.iter().zip(values) {
                ray[j] = value;
            }
        }
        Ok((result, form.reconstruct_direction(&ray)?))
    }
    fn solve_observed_warm(
        &self,
        problem: &LpProblem<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
        warm: &mut SimplexWarmStart<S>,
        provider: Option<&mut (dyn BasisProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        session.check()?;
        let mut form = standardize_bounded(problem)?;
        form.equilibrate()?;
        let mut partial = ProgrammingResult::empty();
        partial.x = form.reconstruct_solution(&try_filled(
            form.columns,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("simplex_initial_partial"),
        )?)?;
        partial.fun = objective_value(problem, &partial.x)?;
        let outcome = if form.rows == 0 {
            self.solve_unconstrained(session, problem, &form)
        } else {
            self.two_phase(session, problem, &form, warm, &mut partial, provider, 0)
        };
        let mut result = match outcome {
            Ok(result) => result,
            Err(error) => match session.stopped() {
                Some(stop) => {
                    partial.termination = stop;
                    partial
                }
                None => return Err(error),
            },
        };
        result.algorithm = Some(session.algorithm()?);
        result.trust = if result.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        };
        result.counts = session.accounting()?;
        session.assess(result.trust)?;
        Ok(result)
    }

    /// Handle unconstrained LP (either unbounded or zero objective).
    #[allow(clippy::unnecessary_wraps)]
    fn solve_unconstrained(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        form: &StandardForm<S>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let mut standard =
            try_filled(form.columns, zero, scioxide_core::quantity!("box_lp_point"))?;
        let mut ray = None;
        for (column, &cost) in form.objective.iter().enumerate() {
            if cost < zero {
                if let Some(upper) = form.upper_bounds[column] {
                    standard[column] = upper;
                } else {
                    ray = Some(column);
                }
            }
        }
        let x = form.reconstruct_solution(&standard)?;
        let result = self.result(
            session,
            problem,
            x,
            0,
            if ray.is_some() {
                OptimizeTermination::Unbounded
            } else {
                OptimizeTermination::Converged
            },
            "box linear program",
            Some(Vec::new()),
        )?;
        if let Some(column) = ray {
            standard.fill(zero);
            standard[column] = <S as scioxide_core::AlgebraicScalar>::ONE;
            self.verify_ray(
                problem,
                result,
                form.reconstruct_direction(&standard)?,
                session,
            )
        } else {
            Ok(result)
        }
    }

    /// Two-phase revised simplex using basis solves and product-form updates.
    #[allow(clippy::too_many_arguments)]
    fn two_phase(
        &self,
        session: &crate::context::SolveSession<'_, '_, '_>,
        problem: &LpProblem<S>,
        form: &StandardForm<S>,
        warm: &mut SimplexWarmStart<S>,
        partial: &mut ProgrammingResult<S>,
        mut provider: Option<&mut (dyn BasisProvider<S> + Send)>,
        mut iterations: usize,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let sparse = problem
            .constraints()
            .is_some_and(|c| c.coefficients().is_sparse())
            || self.sparse_limits.is_some();
        let limits = if sparse {
            Some(match self.sparse_limits {
                Some(limits) => limits,
                None => {
                    let entries = scioxide_core::backend::checked_product(
                        &[
                            16,
                            checked_add(
                                form.coefficients.len(),
                                form.rows,
                                scioxide_core::quantity!("sparse_basis_entries"),
                            )?,
                        ],
                        scioxide_core::quantity!("sparse_basis_fill_budget"),
                    )?;
                    scioxide_sparse::SparseLuLimits::new(
                        entries,
                        scioxide_core::backend::checked_product(
                            &[
                                128,
                                checked_add(
                                    entries,
                                    form.rows,
                                    scioxide_core::quantity!("sparse_basis_bytes"),
                                )?,
                            ],
                            scioxide_core::quantity!("sparse_basis_byte_budget"),
                        )?,
                    )
                }
            })
        } else {
            None
        };
        let mut basis = super::basis::Basis::new(
            form,
            self.basis_updates.get(),
            limits,
            super::basis::lend(&mut provider),
        )?;
        let resumed = warm.extend(problem, form)?;
        if resumed {
            basis.indices = scioxide_core::backend::try_copy_slice(
                &warm.indices,
                scioxide_core::quantity!("warm_basis_restore"),
            )?;
            basis.rows = scioxide_core::backend::try_copy_slice(
                &warm.rows,
                scioxide_core::quantity!("warm_row_restore"),
            )?;
            basis.artificial_rows.clear();
        }
        basis.refactor(form, session)?;
        let m = form.rows;
        let n = form.columns;
        let mut values = try_filled(m, zero, scioxide_core::quantity!("simplex_primal_basis"))?;
        let mut rhs = try_filled(m, zero, scioxide_core::quantity!("simplex_basis_rhs"))?;
        let mut dual = try_filled(m, zero, scioxide_core::quantity!("simplex_dual_basis"))?;
        let mut column = try_filled(m, zero, scioxide_core::quantity!("simplex_entering_column"))?;
        let mut direction = try_filled(
            m,
            zero,
            scioxide_core::quantity!("simplex_entering_direction"),
        )?;
        let mut standard = try_filled(
            n,
            zero,
            scioxide_core::quantity!("simplex_original_solution"),
        )?;
        let mut x = try_filled(
            problem.dimension(),
            zero,
            scioxide_core::quantity!("simplex_original_point"),
        )?;
        let mut upper = try_filled(n, false, scioxide_core::quantity!("simplex_bound_states"))?;
        if resumed {
            upper.copy_from_slice(&warm.upper);
        }
        let mut phase_one = !basis.artificial_rows.is_empty();
        let mut refresh = true;
        let mut stalls = 0;
        let mut bland = false;
        let limit = self.max_iterations(problem).get();
        let rhs_scale = form.rhs.iter().fold(one, |a, v| a.max(v.abs()));
        let feasibility = self.tolerances.feasibility().threshold(rhs_scale)?;
        loop {
            session.check()?;
            let count = basis.rows.len();
            let fresh = refresh;
            if refresh {
                for (target, &row) in rhs[..count].iter_mut().zip(&basis.rows) {
                    let mut sum = scioxide_core::ProductSum::new();
                    sum.add_product(form.rhs[row], one)?;
                    for (j, a) in form.row_entries(row) {
                        if upper[j] {
                            sum.add_product(-a, form.upper_bounds[j].unwrap_or(zero))?;
                        }
                    }
                    *target = sum.finish()?;
                }
                basis.solve_checked(form, &rhs[..count], &mut values[..count], false, session)?;
                refresh = false;
            }
            for (j, value) in standard.iter_mut().enumerate() {
                *value = if upper[j] {
                    form.upper_bounds[j].unwrap_or(zero)
                } else {
                    zero
                };
            }
            for (&index, &value) in basis.indices.iter().zip(&values[..count]) {
                if index < n {
                    standard[index] = value;
                }
            }
            for (target, &index) in rhs[..count].iter_mut().zip(&basis.indices) {
                *target = if phase_one {
                    if index >= n { one } else { zero }
                } else if index < n {
                    form.objective[index]
                } else {
                    zero
                };
            }
            if fresh {
                basis.solve_checked(form, &rhs[..count], &mut dual[..count], true, session)?;
            } else {
                basis.solve(&rhs[..count], &mut dual[..count], true, session)?;
            }
            let cost_scale = if phase_one {
                one
            } else {
                form.objective.iter().fold(zero, |v, c| v.max(c.abs()))
            };
            let cost_tol = self.tolerances.reduced_cost().threshold(cost_scale)?;
            let columns = if phase_one {
                n + basis.artificial_rows.len()
            } else {
                n
            };
            let mut entering: Option<(usize, S, S)> = None;
            for (j, at_upper) in upper
                .iter()
                .copied()
                .chain(core::iter::repeat(false))
                .take(columns)
                .enumerate()
            {
                if basis.contains(j) {
                    continue;
                }
                let sign = if at_upper { -one } else { one };
                let cost = if phase_one {
                    if j >= n { one } else { zero }
                } else {
                    form.objective[j]
                };
                let reduced =
                    checked_difference(cost, basis.column_dot(form, j, &dual[..count])?)? * sign;
                if reduced < -cost_tol && entering.is_none_or(|(_, best, _)| reduced < best) {
                    entering = Some((j, reduced, sign));
                    if bland {
                        break;
                    }
                }
            }
            form.reconstruct_solution_into(&standard, &mut x)?;
            partial.x.copy_from_slice(&x);
            partial.fun = objective_value(problem, &x)?;
            partial.iterations = iterations;
            let feasible = values[..count]
                .iter()
                .zip(&basis.indices)
                .all(|(&value, &j)| {
                    value >= -feasibility
                        && (j >= n || form.upper_bounds[j].is_none_or(|u| value <= u + feasibility))
                });
            if !phase_one && !feasible {
                if resumed && entering.is_none() {
                    if iterations >= limit {
                        return self.result(
                            session,
                            problem,
                            x,
                            iterations,
                            OptimizeTermination::IterationLimit,
                            "dual simplex pivot limit",
                            Some(basis_dual(
                                problem,
                                form,
                                &basis.rows,
                                &dual[..count],
                                !phase_one,
                            )?),
                        );
                    }
                    let mut leaving: Option<(usize, S, S)> = None;
                    for (i, (&value, &j)) in values[..count].iter().zip(&basis.indices).enumerate()
                    {
                        let (violation, sign) = if value < -feasibility {
                            (-value, one)
                        } else if j < n
                            && form.upper_bounds[j].is_some_and(|u| value > u + feasibility)
                        {
                            (value - form.upper_bounds[j].unwrap_or(zero), -one)
                        } else {
                            continue;
                        };
                        if leaving.is_none_or(|(row, best, _)| {
                            if bland {
                                j < basis.indices[row]
                            } else {
                                violation > best
                            }
                        }) {
                            leaving = Some((i, violation, sign));
                        }
                    }
                    let (leaving, _, row_sign) = leaving.ok_or_else(|| {
                        SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Breakdown,
                            "missing violated basic bound",
                        )
                    })?;
                    rhs[..count].fill(zero);
                    rhs[leaving] = row_sign;
                    basis.solve(&rhs[..count], &mut direction[..count], true, session)?;
                    // Dual Harris: first relax reduced-cost feasibility, then
                    // select the strongest pivot in the admissible ratio window.
                    let pivot_tol = self.tolerances.pivot().threshold(one)?;
                    let mut relaxed: Option<S> = None;
                    for (j, &at_upper) in upper.iter().enumerate() {
                        if basis.contains(j) {
                            continue;
                        }
                        let sign = if at_upper { -one } else { one };
                        let a = basis.column_dot(form, j, &direction[..count])? * sign;
                        if a >= -pivot_tol {
                            continue;
                        }
                        let reduced = checked_difference(
                            form.objective[j],
                            basis.column_dot(form, j, &dual[..count])?,
                        )? * sign;
                        let numerator = if bland {
                            reduced.max(zero)
                        } else {
                            reduced.max(zero) + cost_tol
                        };
                        let ratio = checked_float_quotient(numerator, -a)?;
                        relaxed = Some(relaxed.map_or(ratio, |r| r.min(ratio)));
                    }
                    let mut chosen: Option<(usize, S, S)> = None;
                    if let Some(relaxed) = relaxed {
                        for (j, &at_upper) in upper.iter().enumerate() {
                            if basis.contains(j) {
                                continue;
                            }
                            let sign = if at_upper { -one } else { one };
                            let a = basis.column_dot(form, j, &direction[..count])? * sign;
                            if a >= -pivot_tol {
                                continue;
                            }
                            let reduced = checked_difference(
                                form.objective[j],
                                basis.column_dot(form, j, &dual[..count])?,
                            )? * sign;
                            let ratio = checked_float_quotient(reduced.max(zero), -a)?;
                            if ratio <= relaxed
                                && chosen.is_none_or(|(c, r, strength)| {
                                    if bland {
                                        ratio < r || (ratio == r && j < c)
                                    } else {
                                        -a > strength || (-a == strength && j < c)
                                    }
                                })
                            {
                                chosen = Some((j, ratio, -a));
                            }
                        }
                    }
                    let Some((j, ratio, _)) = chosen else {
                        if !fresh {
                            basis.refactor(form, session)?;
                            refresh = true;
                            continue;
                        }
                        for v in &mut direction[..count] {
                            *v = -*v;
                        }
                        return self.result(
                            session,
                            problem,
                            x,
                            iterations,
                            OptimizeTermination::Infeasible,
                            "dual simplex infeasibility witness",
                            Some(basis_dual(
                                problem,
                                form,
                                &basis.rows,
                                &direction[..count],
                                false,
                            )?),
                        );
                    };
                    if let Some(stop) = session.trial_iteration()? {
                        return self.result(
                            session,
                            problem,
                            x,
                            iterations,
                            stop,
                            "dual simplex resource limit",
                            None,
                        );
                    }
                    let sign = if upper[j] { -one } else { one };
                    let previous = if upper[j] {
                        form.upper_bounds[j].unwrap_or(zero)
                    } else {
                        zero
                    };
                    let outgoing = basis.indices[leaving];
                    let target = if row_sign < zero {
                        form.upper_bounds[outgoing].unwrap_or(zero)
                    } else {
                        zero
                    };
                    basis.column(form, j, &mut column[..count]);
                    basis.solve(&column[..count], &mut direction[..count], false, session)?;
                    let step = checked_float_quotient(
                        values[leaving] - target,
                        direction[leaving] * sign,
                    )?;
                    for (value, &d) in values[..count].iter_mut().zip(&direction[..count]) {
                        *value = crate::utils::ordinary_trial_component(*value, -step * sign, d)?;
                    }
                    values[leaving] = crate::utils::ordinary_trial_component(previous, step, sign)?;
                    upper[outgoing] = row_sign < zero;
                    upper[j] = false;
                    let generation = basis.generation;
                    basis.replace(leaving, j, &direction[..count], form, session)?;
                    refresh = basis.generation != generation;
                    pivot::progress(
                        &mut stalls,
                        &mut bland,
                        step.abs(),
                        step * ratio,
                        feasibility,
                        cost_tol,
                    );
                    session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
                    iterations += 1;
                    continue;
                }
                if !fresh {
                    basis.refactor(form, session)?;
                    refresh = true;
                    continue;
                }
                if resumed {
                    drop(basis);
                    warm.clear();
                    return self
                        .two_phase(session, problem, form, warm, partial, provider, iterations);
                }
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::LossOfSignificance,
                    "simplex basis lost primal feasibility after factorization",
                ));
            }
            if !phase_one {
                warm.save(problem, form, &basis, &upper)?;
            }
            let Some((entering, reduced, sign)) = entering else {
                // Refresh and verify before accepting an optimum or a Phase I
                // claim, so updates never bypass independent verification.
                if !fresh {
                    basis.refactor(form, session)?;
                    refresh = true;
                    continue;
                }
                if phase_one {
                    let mut artificial = scioxide_core::ProductSum::new();
                    for (&index, &value) in basis.indices.iter().zip(&values[..count]) {
                        if index >= n {
                            artificial.add_product(value, one)?;
                        }
                    }
                    if artificial.finish()? > feasibility {
                        return self.result(
                            session,
                            problem,
                            x,
                            iterations,
                            OptimizeTermination::Infeasible,
                            "positive Phase I optimum",
                            Some(basis_dual(
                                problem,
                                form,
                                &basis.rows,
                                &dual[..count],
                                false,
                            )?),
                        );
                    }
                    if let Some(row) = basis.indices.iter().position(|&index| index >= n) {
                        let mut replacement = None;
                        for j in 0..n {
                            if basis.contains(j) {
                                continue;
                            }
                            basis.column(form, j, &mut column[..count]);
                            basis.solve(
                                &column[..count],
                                &mut direction[..count],
                                false,
                                session,
                            )?;
                            let scale = direction[..count].iter().fold(zero, |v, d| v.max(d.abs()));
                            if direction[row].abs() > self.tolerances.pivot().threshold(scale)? {
                                replacement = Some(j);
                                break;
                            }
                        }
                        if let Some(j) = replacement {
                            if iterations >= limit {
                                return self.result(
                                    session,
                                    problem,
                                    x,
                                    iterations,
                                    OptimizeTermination::IterationLimit,
                                    "Phase I cleanup limit",
                                    Some(basis_dual(
                                        problem,
                                        form,
                                        &basis.rows,
                                        &dual[..count],
                                        !phase_one,
                                    )?),
                                );
                            }
                            if let Some(stop) = session.trial_iteration()? {
                                return self.result(
                                    session,
                                    problem,
                                    x,
                                    iterations,
                                    stop,
                                    "simplex resource limit",
                                    None,
                                );
                            }
                            basis.replace(row, j, &direction[..count], form, session)?;
                            upper[j] = false;
                            session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
                            iterations += 1;
                        } else {
                            let original = basis.artificial_rows[basis.indices[row] - n];
                            let remove =
                                basis.rows.iter().position(|&r| r == original).ok_or_else(
                                    || {
                                        SciOxideError::invalid_input(
                                            scioxide_core::argument!("problem"),
                                            scioxide_core::InvalidInputReason::Inconsistent,
                                            "missing artificial row",
                                        )
                                    },
                                )?;
                            basis.indices.remove(row);
                            basis.rows.remove(remove);
                        }
                        basis.refactor(form, session)?;
                        refresh = true;
                        continue;
                    }
                    phase_one = false;
                    stalls = 0;
                    bland = false;
                    refresh = true;
                    continue;
                }
                return self.result(
                    session,
                    problem,
                    x,
                    iterations,
                    OptimizeTermination::Converged,
                    "bounded revised simplex optimum",
                    Some(basis_dual(
                        problem,
                        form,
                        &basis.rows,
                        &dual[..count],
                        true,
                    )?),
                );
            };
            if iterations >= limit {
                return self.result(
                    session,
                    problem,
                    x,
                    iterations,
                    OptimizeTermination::IterationLimit,
                    "simplex pivot limit",
                    Some(basis_dual(
                        problem,
                        form,
                        &basis.rows,
                        &dual[..count],
                        !phase_one,
                    )?),
                );
            }
            basis.column(form, entering, &mut column[..count]);
            basis.solve(&column[..count], &mut direction[..count], false, session)?;
            let scale = direction[..count].iter().fold(zero, |v, d| v.max(d.abs()));
            let pivot_tol = self.tolerances.pivot().threshold(scale)?;
            let step = pivot::primal(
                &values[..count],
                &direction[..count],
                &basis.indices,
                &form.upper_bounds,
                sign,
                if entering < n {
                    form.upper_bounds[entering]
                } else {
                    None
                },
                feasibility,
                pivot_tol,
                bland,
            )?;
            let Some(step) = step else {
                if !fresh {
                    basis.refactor(form, session)?;
                    refresh = true;
                    continue;
                }
                if phase_one {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "unbounded auxiliary problem",
                    ));
                }
                let result = self.result(
                    session,
                    problem,
                    x,
                    iterations,
                    OptimizeTermination::Unbounded,
                    "simplex recession direction",
                    None,
                )?;
                standard.fill(zero);
                standard[entering] = sign;
                for (&j, &d) in basis.indices.iter().zip(&direction[..count]) {
                    if j < n {
                        standard[j] = -d * sign;
                    }
                }
                return self.verify_ray(
                    problem,
                    result,
                    form.reconstruct_direction(&standard)?,
                    session,
                );
            };
            if let Some(stop) = session.trial_iteration()? {
                return self.result(
                    session,
                    problem,
                    x,
                    iterations,
                    stop,
                    "simplex resource limit",
                    None,
                );
            }
            for (value, &d) in values[..count].iter_mut().zip(&direction[..count]) {
                *value = crate::utils::ordinary_trial_component(*value, -step.length * sign, d)?;
            }
            if let Some(row) = step.row {
                let previous = if entering < n && upper[entering] {
                    form.upper_bounds[entering].unwrap_or(zero)
                } else {
                    zero
                };
                let outgoing = basis.indices[row];
                if outgoing < n {
                    upper[outgoing] = step.upper;
                }
                if entering < n {
                    upper[entering] = false;
                }
                values[row] = crate::utils::ordinary_trial_component(previous, step.length, sign)?;
                let generation = basis.generation;
                basis.replace(row, entering, &direction[..count], form, session)?;
                refresh = basis.generation != generation;
            } else {
                upper[entering] = !upper[entering];
            }
            pivot::progress(
                &mut stalls,
                &mut bland,
                step.length,
                reduced * step.length,
                feasibility,
                cost_tol,
            );
            session.accept(scioxide_core::PartialResultTrust::DiagnosticOnly)?;
            iterations += 1;
        }
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
        row_dual: Option<Vec<S>>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let fun = objective_value(problem, &x)?;
        self.result_with_fun(
            session,
            problem,
            x,
            fun,
            iterations,
            termination,
            message,
            row_dual,
        )
    }

    fn verify_ray(
        &self,
        problem: &LpProblem<S>,
        mut result: ProgrammingResult<S>,
        direction: Vec<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        if session.stopped().is_some() {
            return Ok(result);
        }
        match certificate::ray(problem, &result.x, direction, self.tolerances, session) {
            Ok(assessment) => {
                result.termination = if assessment.accepted {
                    OptimizeTermination::Unbounded
                } else {
                    OptimizeTermination::SuspectedUnbounded
                };
                result.diagnostics = assessment.diagnostics;
                result.evidence = assessment.evidence;
                result.verification = assessment.report;
            }
            Err(_) if session.stopped().is_some() => {
                result.termination = session
                    .stopped()
                    .unwrap_or(OptimizeTermination::EvaluationLimit);
            }
            Err(error) => return Err(error),
        }
        Ok(result)
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
        row_dual: Option<Vec<S>>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let assessment =
            certificate::assess(problem, &x, self.tolerances, termination, row_dual, session)?;
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

pub(super) fn basis_dual<S: RealScalar + NumericScalar<Real = S>>(
    problem: &LpProblem<S>,
    form: &StandardForm<S>,
    rows: &[usize],
    dual: &[S],
    optimality: bool,
) -> Result<Vec<S>, SciOxideError> {
    let original = problem.constraints().map_or(0, |c| c.len());
    let count = original;
    let mut output = try_filled(
        count,
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        scioxide_core::quantity!("original_basis_dual"),
    )?;
    basis_dual_into(problem, form, rows, dual, optimality, &mut output)?;
    Ok(output)
}

pub(super) use pivot::progress as mip_progress;

pub(super) fn basis_dual_into<S: RealScalar + NumericScalar<Real = S>>(
    problem: &LpProblem<S>,
    form: &StandardForm<S>,
    rows: &[usize],
    dual: &[S],
    optimality: bool,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    output.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    let count = output.len();
    for (&row, &value) in rows.iter().zip(dual) {
        if row >= count {
            continue;
        }
        let sense = problem
            .constraints()
            .map(|c| c.senses()[row])
            .unwrap_or(LinearConstraintSense::Equal);
        let sign = if sense == LinearConstraintSense::GreaterEqual {
            <S as scioxide_core::AlgebraicScalar>::ONE
        } else {
            -<S as scioxide_core::AlgebraicScalar>::ONE
        };
        output[row] = scioxide_core::Scaled::from_value(value)?
            .shifted(
                i64::from(form.row_exponents[row])
                    + if optimality {
                        i64::from(form.cost_exponent)
                    } else {
                        0
                    },
            )?
            .mul(scioxide_core::Scaled::from_value(
                form.row_signs[row] * sign,
            )?)?
            .round()?;
    }
    // Construct sign-admissible optimality and Farkas multipliers. Degenerate
    // basis solves can leave tiny wrong-sign components. Projection is candidate
    // construction, never verifier repair: residual support, the witness margin
    // and all KKT checks are rerun.
    if let Some(constraints) = problem.constraints() {
        for (value, sense) in output.iter_mut().zip(constraints.senses()) {
            *value = match sense {
                LinearConstraintSense::LessEqual => {
                    value.max(<S as scioxide_core::AlgebraicScalar>::ZERO)
                }
                LinearConstraintSense::GreaterEqual => {
                    value.min(<S as scioxide_core::AlgebraicScalar>::ZERO)
                }
                LinearConstraintSense::Equal => *value,
            };
        }
    }
    Ok(())
}

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

    /// min -x - 2y
    /// s.t. x + y <= 4, x <= 3, y <= 2
    /// Solution: x=2, y=2, fun=-6
    #[test]
    fn test_simplex_basic_lp() {
        let problem = problem(
            vec![-1.0_f64, -2.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![4.0, 3.0, 2.0],
            vec![LinearConstraintSense::LessEqual; 3],
        );

        let solver = SimplexSolver::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert_eq!(result.termination, OptimizeTermination::Converged);
        assert!((result.fun - (-6.0)).abs() < 1e-8);
        assert!((result.x[0] - 2.0).abs() < 1e-8);
        assert!((result.x[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn phase_one_budget_exhaustion_is_preserved() {
        let problem = problem(
            vec![-1.0_f64, -2.0],
            vec![1.0, 1.0, 1.0, 2.0],
            vec![3.0, 5.0],
            vec![LinearConstraintSense::Equal; 2],
        );
        let result = SimplexSolver::default()
            .with_max_iterations(Budget::ONE)
            .solve(&problem)
            .unwrap();

        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        assert!(!result.is_success());
        assert_eq!(result.iterations(), 1);
    }

    /// min x + y, s.t. x+y<=1, x+y>=3 (infeasible)
    #[test]
    fn test_simplex_infeasible() {
        let problem = problem(
            vec![1.0_f64, 1.0],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![1.0, -3.0],
            vec![LinearConstraintSense::LessEqual; 2],
        );

        let solver = SimplexSolver::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(!result.is_success());
        assert_eq!(result.termination, OptimizeTermination::Infeasible);
    }

    /// min -x - y, s.t. x+y=3, x<=2, y<=2
    /// Solution: fun=-3, x0+x1=3
    #[test]
    fn test_simplex_equality_constraint() {
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

        let solver = SimplexSolver::default()
            .with_tolerances(crate::OptimalityTolerances::programming_default());
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.fun - (-3.0)).abs() < 1e-8);
        assert!((result.x[0] + result.x[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_simplex_basic_lp_f32() {
        let problem = problem(
            vec![-1.0_f32, -2.0],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![4.0, 3.0, 2.0],
            vec![LinearConstraintSense::LessEqual; 3],
        );

        let solver = SimplexSolver {
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
            ..SimplexSolver::default()
        };
        let result = solver.solve(&problem).unwrap();

        assert!(result.is_success());
        assert!((result.fun - (-6.0_f32)).abs() < 1e-4);
    }

    #[test]
    fn test_simplex_default() {
        let solver = SimplexSolver::<f64>::default();
        assert_eq!(
            solver
                .max_iterations(&LpProblem::new(vec![0.0]).unwrap())
                .get(),
            1000
        );
    }
}
