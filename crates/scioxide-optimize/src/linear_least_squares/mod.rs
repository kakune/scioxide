//! Linear least squares with bounds: active-set methods with updated QR factors.
use crate::OptimizeTermination;
use alloc::vec::Vec;
use scioxide_core::{
    AdjointOperator, Bounds, Budget, FeasibilityTolerance, GradientTolerance, NumericScalar,
    RealScalar, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::{Matrix, MatrixView, RankPolicy, RankReport};
mod active_qr;
mod engine;
mod linear;
mod workspace;
pub use workspace::{BoundedLinearLeastSquaresWorkspace, NnlsWorkspace};

use crate::validation::invalid;
fn shape(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected != actual {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(expected),
            scioxide_core::Dimensions::one(actual),
        ));
    }
    Ok(())
}
fn finite<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    crate::utils::validate_finite_origin(values, scioxide_core::NonFiniteOrigin::Input)
}

fn finite_product<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    crate::utils::validate_finite_origin(values, scioxide_core::NonFiniteOrigin::Callback)
}

/// Describes the problem `min ½ ‖A x − b‖²` subject to optional bounds
/// `l ≤ x ≤ u`, for [`Nnls`] and [`BoundedLinearLeastSquares`].
///
/// `A` is an `m × n` real matrix or operator and `b ∈ ℝᵐ`. The problem
/// borrows `A`, `b` and the bounds without copying them. A dense matrix
/// ([`Self::from_dense`], [`Self::from_view`]) is solved with QR factors; an
/// operator ([`Self::new`]), such as a sparse matrix, is solved with LSMR.
/// Without bounds every variable is free. [`Nnls`] imposes `x ≥ 0` itself and
/// rejects a problem with bounds.
#[derive(Clone, Copy)]
pub struct LinearLeastSquaresProblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    operator: crate::problem::BorrowedAdjointOperator<'a, S>,
    rhs: &'a [S],
    bounds: Option<&'a Bounds<S>>,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> LinearLeastSquaresProblem<'a, S> {
    /// Creates a problem from an operator `A` and right-hand side `b`. It does
    /// not apply the operator.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs.len()` differs from the number of rows.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    pub fn new(operator: &'a dyn AdjointOperator<S>, rhs: &'a [S]) -> Result<Self, SciOxideError> {
        shape(operator.rows(), rhs.len())?;
        finite(rhs)?;
        Ok(Self {
            operator: crate::problem::BorrowedAdjointOperator::Other(operator),
            rhs,
            bounds: None,
        })
    }
    /// Creates a problem from a dense matrix `A` and right-hand side `b`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs.len()` differs from the number of rows.
    /// - `NonFinite` if `matrix` or `rhs` contains NaN or infinity.
    pub fn from_dense(matrix: &'a Matrix<S>, rhs: &'a [S]) -> Result<Self, SciOxideError> {
        Self::from_view(&matrix.view(), rhs)
    }
    /// Creates a problem from a dense matrix view, which may be strided.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs.len()` differs from the number of rows.
    /// - `NonFinite` if `matrix` or `rhs` contains NaN or infinity.
    pub fn from_view(matrix: &MatrixView<'a, S>, rhs: &'a [S]) -> Result<Self, SciOxideError> {
        shape(matrix.rows(), rhs.len())?;
        finite(rhs)?;
        Ok(Self {
            operator: crate::problem::BorrowedAdjointOperator::Dense(matrix.try_finite()?),
            rhs,
            bounds: None,
        })
    }

    fn operator(&self) -> &dyn AdjointOperator<S> {
        self.operator.as_operator()
    }
    fn dense(&self) -> Option<MatrixView<'_, S>> {
        self.operator.dense()
    }
    /// Adds bounds `l ≤ x ≤ u`. A missing endpoint is unbounded, and equal
    /// endpoints fix the variable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if bounds are already set.
    /// - `DimensionMismatch` if `bounds.len()` differs from the number of
    ///   columns.
    pub fn with_bounds(mut self, bounds: &'a Bounds<S>) -> Result<Self, SciOxideError> {
        if self.bounds.is_some() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "duplicate bounds",
            ));
        }
        shape(self.cols(), bounds.len())?;
        self.bounds = Some(bounds);
        Ok(self)
    }
    /// Returns the number of rows `m` of `A`.
    pub fn rows(&self) -> usize {
        self.operator().rows()
    }
    /// Returns the number of variables `n`.
    pub fn cols(&self) -> usize {
        self.operator().cols()
    }
    fn bound(&self, index: usize, nnls: bool) -> (Option<S>, Option<S>) {
        if nnls {
            (Some(<_ as scioxide_core::AlgebraicScalar>::ZERO), None)
        } else {
            self.bounds.map_or((None, None), |b| b.as_slice()[index])
        }
    }
}

/// Configures [`Nnls`] and [`BoundedLinearLeastSquares`]: optimality
/// tolerances, rank threshold and iteration limits.
///
/// The defaults are the default core tolerances and at most 1000 outer and
/// 1000 LSMR iterations. See [Termination](crate#termination) for the shared
/// stopping rules.
#[derive(Debug, Clone, Copy)]
pub struct LinearLeastSquaresOptions<S: RealScalar> {
    gradient: GradientTolerance<S>,
    feasibility: FeasibilityTolerance<S>,
    complementarity: ResidualTolerance<S>,
    rank: RankPolicy<S>,
    iterations: Budget,
    inner: Budget,
}
impl<S: RealScalar> Default for LinearLeastSquaresOptions<S> {
    fn default() -> Self {
        Self {
            gradient: GradientTolerance::default(),
            feasibility: FeasibilityTolerance::default(),
            complementarity: ResidualTolerance::default(),
            rank: RankPolicy::DimensionScaledEpsilon,
            iterations: Budget::ONE.saturating_add(999),
            inner: Budget::ONE.saturating_add(999),
        }
    }
}
impl<S: RealScalar> LinearLeastSquaresOptions<S> {
    /// Sets the tolerances of the KKT conditions: stationarity, bound
    /// feasibility and complementarity, in the units of the problem.
    #[must_use]
    pub const fn with_tolerances(
        mut self,
        gradient: GradientTolerance<S>,
        feasibility: FeasibilityTolerance<S>,
        complementarity: ResidualTolerance<S>,
    ) -> Self {
        self.gradient = gradient;
        self.feasibility = feasibility;
        self.complementarity = complementarity;
        self
    }
    /// Sets the numerical-rank threshold for dense problems. A rank-truncated
    /// solution must still satisfy the KKT conditions to succeed.
    #[must_use]
    pub const fn with_rank_policy(mut self, rank: RankPolicy<S>) -> Self {
        self.rank = rank;
        self
    }
    /// Sets the maximum number of outer iterations. LSMR iterations count toward
    /// the limits of an enclosing solve.
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: Budget) -> Self {
        self.iterations = limit;
        self
    }
    /// Sets the maximum number of LSMR iterations per subproblem.
    #[must_use]
    pub const fn with_max_inner_iterations(mut self, limit: Budget) -> Self {
        self.inner = limit;
        self
    }
}

/// Gives the status of a variable at the returned point.
///
/// The multiplier of a fixed variable may have either sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearLeastSquaresActivity {
    /// No bound is active.
    Free,
    /// At its lower bound.
    Lower,
    /// At its upper bound.
    Upper,
    /// Fixed by equal lower and upper bounds.
    Fixed,
}
/// Holds the infinity norms of the KKT residuals at the returned point.
///
/// The values are computed in floating point and are not rigorous bounds.
#[derive(Debug, Clone, Copy)]
pub struct LinearLeastSquaresKkt<S: RealScalar> {
    primal_feasibility: S,
    stationarity: S,
    dual_feasibility: S,
    complementarity: S,
}
impl<S: RealScalar> LinearLeastSquaresKkt<S> {
    const fn zero() -> Self {
        Self {
            primal_feasibility: S::ZERO,
            stationarity: S::ZERO,
            dual_feasibility: S::ZERO,
            complementarity: S::ZERO,
        }
    }
    /// Returns the largest bound violation.
    pub const fn primal_feasibility(self) -> S {
        self.primal_feasibility
    }
    /// Returns the largest component of the gradient of the Lagrangian, with
    /// nonnegative bound multipliers constructed from the gradient.
    pub const fn stationarity(self) -> S {
        self.stationarity
    }
    /// Returns the largest magnitude of a negative multiplier; zero for the
    /// constructed multipliers.
    pub const fn dual_feasibility(self) -> S {
        self.dual_feasibility
    }
    /// Returns the largest product of a bound multiplier and the distance to its
    /// bound.
    pub const fn complementarity(self) -> S {
        self.complementarity
    }
}
/// Holds the result of [`Nnls`] or [`BoundedLinearLeastSquares`]: solution,
/// residual, multipliers, active set and KKT residuals.
///
/// After an unsuccessful stop it holds the last accepted feasible point. See
/// [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct LinearLeastSquaresResult<S: RealScalar> {
    x: Vec<S>,
    residual: Vec<S>,
    gradient: Vec<S>,
    lower: Vec<S>,
    upper: Vec<S>,
    activity: Vec<LinearLeastSquaresActivity>,
    objective: S,
    residual_norm: S,
    kkt: LinearLeastSquaresKkt<S>,
    rank: Option<RankReport<S>>,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
    algorithm: Option<scioxide_core::AlgorithmId>,
    trust: scioxide_core::PartialResultTrust,
    iterations: usize,
}
impl<S: RealScalar> LinearLeastSquaresResult<S> {
    /// Creates an empty result to pass to `solve_into`.
    pub fn empty() -> Self {
        Self {
            x: Vec::new(),
            residual: Vec::new(),
            gradient: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            activity: Vec::new(),
            objective: S::ZERO,
            residual_norm: S::ZERO,
            kkt: LinearLeastSquaresKkt::zero(),
            rank: None,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            iterations: 0,
        }
    }

    /// Reserves storage for `rows` residual components and `cols` variables, so
    /// that a later solve does not allocate.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        use scioxide_core::backend::try_reserve;
        for (values, quantity) in [
            (&mut self.x, scioxide_core::quantity!("result.x")),
            (
                &mut self.gradient,
                scioxide_core::quantity!("result.gradient"),
            ),
            (&mut self.lower, scioxide_core::quantity!("result.lower")),
            (&mut self.upper, scioxide_core::quantity!("result.upper")),
        ] {
            let additional = cols.saturating_sub(values.len());
            try_reserve(values, additional, quantity)?;
        }
        let residual = rows.saturating_sub(self.residual.len());
        try_reserve(
            &mut self.residual,
            residual,
            scioxide_core::quantity!("result.residual"),
        )?;
        let activity = cols.saturating_sub(self.activity.len());
        try_reserve(
            &mut self.activity,
            activity,
            scioxide_core::quantity!("result.activity"),
        )
    }

    fn fits(&self, rows: usize, cols: usize) -> bool {
        [&self.x, &self.gradient, &self.lower, &self.upper]
            .iter()
            .all(|values| values.capacity() >= cols)
            && self.residual.capacity() >= rows
            && self.activity.capacity() >= cols
    }

    /// Copies `source` into `self`; on allocation failure `self` is unchanged.
    pub(crate) fn try_copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        let (rows, cols) = (source.residual.len(), source.x.len());
        if !self.fits(rows, cols) {
            let mut candidate = Self::empty();
            candidate.try_reserve(rows, cols)?;
            *self = candidate;
        }
        for (target, values) in [
            (&mut self.x, &source.x),
            (&mut self.residual, &source.residual),
            (&mut self.gradient, &source.gradient),
            (&mut self.lower, &source.lower),
            (&mut self.upper, &source.upper),
        ] {
            target.clear();
            target.extend_from_slice(values);
        }
        self.activity.clear();
        self.activity.extend_from_slice(&source.activity);
        self.objective = source.objective;
        self.residual_norm = source.residual_norm;
        self.kkt = source.kkt;
        self.rank = source.rank;
        self.termination = source.termination;
        self.counts = source.counts;
        self.algorithm = source.algorithm;
        self.trust = source.trust;
        self.iterations = source.iterations;
        Ok(())
    }

    /// Zeroed storage of exactly this size for accepted and trial states.
    pub(crate) fn sized(rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        use scioxide_core::backend::try_filled;
        Ok(Self {
            x: try_filled(
                cols,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("result.x"),
            )?,
            residual: try_filled(
                rows,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("result.residual"),
            )?,
            gradient: try_filled(
                cols,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("result.gradient"),
            )?,
            lower: try_filled(
                cols,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("result.lower"),
            )?,
            upper: try_filled(
                cols,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("result.upper"),
            )?,
            activity: try_filled(
                cols,
                LinearLeastSquaresActivity::Free,
                scioxide_core::quantity!("result.activity"),
            )?,
            objective: <_ as scioxide_core::AlgebraicScalar>::ZERO,
            residual_norm: <_ as scioxide_core::AlgebraicScalar>::ZERO,
            kkt: LinearLeastSquaresKkt::zero(),
            rank: None,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            iterations: 0,
        })
    }
    /// Returns the solution `x`, which satisfies the bounds.
    pub fn x(&self) -> &[S] {
        &self.x
    }
    /// Returns the residual `A x − b`.
    pub fn residual(&self) -> &[S] {
        &self.residual
    }
    /// Returns the gradient `Aᵀ(A x − b)`.
    pub fn gradient(&self) -> &[S] {
        &self.gradient
    }
    /// Returns the lower-bound multipliers, which are nonnegative; zero for a
    /// missing bound.
    pub fn lower_multipliers(&self) -> &[S] {
        &self.lower
    }
    /// Returns the upper-bound multipliers, which are nonnegative; zero for a
    /// missing bound.
    pub fn upper_multipliers(&self) -> &[S] {
        &self.upper
    }
    /// Returns the status of each variable, decided with the feasibility
    /// tolerance.
    pub fn activity(&self) -> &[LinearLeastSquaresActivity] {
        &self.activity
    }
    /// Returns the objective `½ ‖A x − b‖²`.
    pub const fn objective(&self) -> S {
        self.objective
    }
    /// Returns the residual norm `‖A x − b‖₂`.
    pub const fn residual_norm(&self) -> S {
        self.residual_norm
    }
    /// Returns the KKT residuals, recomputed from `A` and `b` at `x`.
    pub const fn kkt(&self) -> LinearLeastSquaresKkt<S> {
        self.kkt
    }
    /// Returns the numerical rank of the free columns in the last dense
    /// subproblem; `None` for operator problems.
    pub const fn rank(&self) -> Option<RankReport<S>> {
        self.rank
    }
    /// Returns why the solve stopped.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns whether the solve succeeded, that is, the KKT conditions hold
    /// within the tolerances.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            self.algorithm(),
        )
        .map_or(Ok(()), Err)
    }

    /// Returns the iteration and operator-evaluation counts, including LSMR
    /// subproblems.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the algorithm, or `None` before the first solve.
    pub const fn algorithm(&self) -> Option<scioxide_core::AlgorithmId> {
        self.algorithm
    }
    /// Returns which checks the returned point passed.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }
    /// Returns the number of outer iterations.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
}

macro_rules! solver {
    ($name:ident, $workspace:ident, $nnls:expr, $doc:literal) => {
        #[doc=$doc]
        #[doc = ""]
        #[doc = "The solve starts from zero projected onto the bounds and then from the"]
        #[doc = "unconstrained solution clipped to the bounds. Each outer iteration frees"]
        #[doc = "the bound variable whose multiplier has the wrong sign; inner iterations"]
        #[doc = "solve for the free variables and step back to the feasible set, fixing"]
        #[doc = "variables that reach a bound. For a dense `A`, the QR factors of the free"]
        #[doc = "columns are updated by one Householder reflection when a column enters and"]
        #[doc = "by Givens rotations when one leaves, at O(m n) cost; the normal equations"]
        #[doc = "are never formed. Rank-deficient or wide systems fall back to pivoted QR"]
        #[doc = "or the SVD. An operator `A` uses LSMR on the free variables. Success"]
        #[doc = "requires the KKT conditions, recomputed from `A` and `b`, to hold within"]
        #[doc = "the tolerances of [`LinearLeastSquaresOptions`]."]
        #[doc = ""]
        #[doc = "A workspace serves one problem shape and one kind of `A` (dense or"]
        #[doc = "operator) and does not grow."]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```"]
        #[doc = "use scioxide_linalg::Matrix;"]
        #[doc = concat!("use scioxide_optimize::{LinearLeastSquaresProblem, ", stringify!($name), "};")]
        #[doc = "// A = [1; 1] and b = [2; 2]: the least-squares solution is x = 2."]
        #[doc = "let a = Matrix::from_vec(2, 1, vec![1.0_f64, 1.0])?;"]
        #[doc = "let problem = LinearLeastSquaresProblem::from_dense(&a, &[2., 2.])?;"]
        #[doc = concat!("let result = ", stringify!($name), "::default().solve(&problem)?;")]
        #[doc = "assert!(result.is_success());"]
        #[doc = "assert!((result.x()[0] - 2.).abs() < 1e-10);"]
        #[doc = "# Ok::<(), scioxide_core::SciOxideError>(())"]
        #[doc = "```"]
        #[doc = ""]
        #[doc = "# References"]
        #[doc = ""]
        #[doc = "- C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*,"]
        #[doc = "  Prentice-Hall, 1974, ch. 23 (nonnegative least squares)."]
        #[doc = "- P. B. Stark and R. L. Parker, \"Bounded-variable least-squares: an"]
        #[doc = "  algorithm and applications\", *Comput. Statist.* 10, 129–141, 1995."]
        #[derive(Debug, Clone, Copy)]
        pub struct $name<S: RealScalar> {
            options: LinearLeastSquaresOptions<S>,
        }
        impl<S: RealScalar> Default for $name<S> {
            fn default() -> Self {
                Self::new(LinearLeastSquaresOptions::default())
            }
        }
        impl<S: RealScalar> $name<S> {
            /// Creates a solver with `options`.
            pub const fn new(options: LinearLeastSquaresOptions<S>) -> Self {
                Self { options }
            }
            /// Returns the options.
            pub const fn options(&self) -> &LinearLeastSquaresOptions<S> {
                &self.options
            }
        }
        impl<S: RealScalar + NumericScalar<Real = S>> $name<S> {
            /// Allocates a workspace for the shape of `problem` and its kind of `A`.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if the solver is [`Nnls`] and `problem` has bounds.
            /// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be
            ///   allocated.
            /// - Errors from the operator's workspace-size queries.
            pub fn workspace(
                &self,
                problem: &LinearLeastSquaresProblem<'_, S>,
            ) -> Result<$workspace<S>, SciOxideError> {
                engine::validate(problem, $nnls)?;
                Ok($workspace {
                    storage: workspace::Storage::new(problem, &self.options, $nnls)?,
                    shell: crate::attempt::Shell::new(),
                })
            }
            /// Solves `problem` on the calling thread and returns a new result.
            ///
            /// # Errors
            ///
            /// Returns the errors of `workspace` and `solve_into`.
            pub fn solve(
                &self,
                problem: &LinearLeastSquaresProblem<'_, S>,
            ) -> Result<LinearLeastSquaresResult<S>, SciOxideError> {
                let mut workspace = self.workspace(problem)?;
                let mut output = LinearLeastSquaresResult::empty();
                self.solve_into(
                    problem,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            /// Solves `problem` and writes the result to `output`.
            ///
            /// Iterations are limited by the options and by the limits of an
            /// enclosing solve in `scope`. A limit,
            /// cancellation or deadline returns the last accepted feasible point
            /// with an unsuccessful termination. With a workspace from
            /// `workspace(problem)` and a reserved `output`, the call does not
            /// allocate.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if the solver is [`Nnls`] and `problem` has bounds,
            ///   or the workspace was prepared for another solver or kind of `A`.
            /// - `DimensionMismatch` if the workspace was prepared for another shape.
            /// - Errors from operator products, unchanged, and `NonFinite` if a
            ///   product returns NaN or infinity.
            pub fn solve_into<'scope, 'capability>(
                &self,
                problem: &LinearLeastSquaresProblem<'_, S>,
                output: &mut LinearLeastSquaresResult<S>,
                workspace: &mut $workspace<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError> {
                workspace.shell.reject(scioxide_core::AttemptKind::Call);
                let scope = scope.into();
                engine::validate(problem, $nnls)?;
                engine::check_workspace(problem, $nnls, &workspace.storage)?;
                crate::attempt::preflight(&scope, None)?;
                let identity = crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", stringify!($name), 1),
                    problem.cols(),
                    &(problem.rows(), &self.options),
                );
                let storage = &mut workspace.storage;
                let (outcome, started) = workspace.shell.execute(
                    scope,
                    crate::attempt::Entry::Call {
                        identity,
                        budgets: scioxide_core::SolveBudgets::unbounded(),
                    },
                    |context| {
                        engine::solve(
                            problem,
                            &self.options,
                            $nnls,
                            output,
                            storage,
                            &crate::context::SolveSession::new(
                                context,
                                scioxide_core::argument!("scope"),
                            ),
                        )
                    },
                );
                let succeeded = outcome.is_ok() && output.is_success();
                let progress = started
                    .then(|| crate::attempt::Accepted::accepted(&workspace.storage))
                    .flatten()
                    .map(|state| crate::attempt::progress(state, succeeded));
                workspace.shell.finish(
                    scioxide_core::AttemptKind::Call,
                    started,
                    outcome.is_err(),
                    succeeded,
                    progress,
                );
                outcome
            }
        }
    };
}
solver!(
    Nnls,
    NnlsWorkspace,
    true,
    "Solves the nonnegative least-squares problem `min ½ ‖A x − b‖²` subject to `x ≥ 0`.\n\nUses the active-set method of Lawson and Hanson."
);
solver!(
    BoundedLinearLeastSquares,
    BoundedLinearLeastSquaresWorkspace,
    false,
    "Solves the bounded least-squares problem `min ½ ‖A x − b‖²` subject to `l ≤ x ≤ u`.\n\nUses the bounded-variable least-squares (BVLS) active-set method of Stark and Parker. Fixed variables are eliminated before the iteration."
);
