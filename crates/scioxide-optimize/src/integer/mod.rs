//! Deterministic mixed-integer branch-and-cut search with bounded storage.
use crate::{LpProblem, MipProblem, OptimizeTermination};
use alloc::vec::Vec;
use scioxide_core::backend::{
    checked_add, try_copy_slice as copy, try_filled as filled, try_with_capacity as capacity,
};
use scioxide_core::checked_difference;
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};
mod contracts;
mod cuts;
mod model;
mod result;
mod search;
#[cfg(test)]
mod tests;
mod validation;
pub use contracts::{
    MipBranching, MipCutKind, MipLimits, MipNode, MipRelaxationSolver, MipSearchState,
    MipStatistics,
};
use contracts::{Pseudocost, relaxation_bytes};
pub use model::MipPresolveMap;
pub use result::MipResult;
use validation::Candidate;

use crate::validation::invalid;

/// Solves mixed-integer linear programs by LP-based branch and cut.
///
/// The problem is a [`MipProblem`]: an [`LpProblem`] in which the listed
/// variables must take integer values. Each node solves an LP relaxation with
/// [`SimplexSolver`](crate::SimplexSolver), reoptimizing from the previous
/// basis, or with a custom [`MipRelaxationSolver`]. The search processes the
/// open node with the lowest bound first (ties in order of creation) and
/// branches on a fractional variable chosen by [`MipBranching`]. Nodes may add
/// cutting planes ([`MipCutKind`]) and run a round-and-fix heuristic;
/// [`MipLimits`] bounds this work and the storage of the tree. An optional root
/// presolve tightens bounds and removes fixed variables. Every incumbent is
/// rounded, checked against the original constraints and bounds, and has its
/// objective recomputed. Tree search is sequential, also when the execution
/// context has an executor.
///
/// Node bounds come from checked floating-point dual information. They are
/// estimates, not interval enclosures. A node is pruned, and the search
/// succeeds, when
///
/// ```text
/// incumbent − bound ≤ a + r · max(1, |incumbent|, |bound|)
/// ```
///
/// where `a` and `r` are the absolute and relative parts of the gap tolerance
/// in [`Self::with_relaxation_tolerances`]. [`MipResult::absolute_gap`] reports
/// the gap actually reached. Before the search, missing variable bounds that
/// follow from a constraint row are added, rounded outward. A variable that
/// still lacks a needed bound can leave a node without a verified bound, which
/// stops the search with [`OptimizeTermination::Stagnation`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Bounds;
/// use scioxide_linalg::Matrix;
/// use scioxide_optimize::{LinearConstraints, LpProblem, MipProblem};
/// let row = Matrix::from_rows(&[[2.0, 2.0]])?;
/// let rows = LinearConstraints::less_equal(row, vec![3.0])?;
/// let lp = LpProblem::new(vec![-1.0_f64, -1.0])?
///     .with_bounds(Bounds::new(vec![(Some(0.0), Some(1.0)); 2])?)?
///     .with_constraints(rows)?;
/// let result = MipProblem::new(lp, vec![0, 1])?.solve()?;
/// assert!((result.fun() + 1.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. H. Land and A. G. Doig, "An automatic method of solving discrete
///   programming problems", *Econometrica* 28(3), 497–520, 1960.
/// - T. Achterberg, T. Koch and A. Martin, "Branching rules revisited",
///   *Oper. Res. Lett.* 33(1), 42–54, 2005.
/// - L. A. Wolsey, *Integer Programming*, Wiley, 1998.
#[derive(Debug, Clone)]
pub struct BranchAndBound<S: RealScalar> {
    integrality_tolerance: scioxide_core::IntegralityTolerance<S>,
    relaxation_tolerances: crate::OptimalityTolerances<S>,
    max_nodes: Budget,
    max_lp_iterations: Budget,
    limits: MipLimits,
    branching: MipBranching,
    presolve: bool,
}
impl<S: RealScalar> Default for BranchAndBound<S> {
    fn default() -> Self {
        Self {
            integrality_tolerance: scioxide_core::IntegralityTolerance::default(),
            relaxation_tolerances: crate::OptimalityTolerances::programming_default(),
            max_nodes: Budget::ONE.saturating_add(999),
            max_lp_iterations: Budget::ONE.saturating_add(999),
            limits: MipLimits::default(),
            branching: MipBranching::Reliability,
            presolve: true,
        }
    }
}
impl<S: RealScalar> BranchAndBound<S> {
    /// Sets the largest distance from an integer at which a value counts as integral.
    pub const fn with_integrality_tolerance(
        mut self,
        value: scioxide_core::IntegralityTolerance<S>,
    ) -> Self {
        self.integrality_tolerance = value;
        self
    }
    /// Sets the tolerances of the relaxations and the gap tolerance of the search.
    ///
    /// The default is `1e-8` absolute and relative for every role (`1e-6` when
    /// the machine epsilon exceeds `1e-8`).
    pub const fn with_relaxation_tolerances(
        mut self,
        value: crate::OptimalityTolerances<S>,
    ) -> Self {
        self.relaxation_tolerances = value;
        self
    }
    /// Sets the maximum number of processed nodes, including the root (default 1000).
    pub const fn with_max_nodes(mut self, value: Budget) -> Self {
        self.max_nodes = value;
        self
    }
    /// Sets the pivot limit of each relaxation solve, including repairs (default 1000).
    pub const fn with_max_lp_iterations(mut self, value: Budget) -> Self {
        self.max_lp_iterations = value;
        self
    }
    /// Sets the storage limits and optional work.
    pub const fn with_limits(mut self, value: MipLimits) -> Self {
        self.limits = value;
        self
    }
    /// Sets the branching rule (default [`MipBranching::Reliability`]).
    pub const fn with_branching(mut self, value: MipBranching) -> Self {
        self.branching = value;
        self
    }
    /// Enables the root presolve (default true).
    ///
    /// Presolve turns singleton rows into bounds, rounds the bounds of integer
    /// variables, and removes fixed variables. Every step is exact.
    pub const fn with_presolve(mut self, value: bool) -> Self {
        self.presolve = value;
        self
    }
    /// Returns the integrality tolerance.
    pub const fn integrality_tolerance(&self) -> scioxide_core::IntegralityTolerance<S> {
        self.integrality_tolerance
    }
    /// Returns the relaxation and gap tolerances.
    pub const fn relaxation_tolerances(&self) -> crate::OptimalityTolerances<S> {
        self.relaxation_tolerances
    }
    /// Returns the maximum number of processed nodes.
    pub const fn max_nodes(&self) -> Budget {
        self.max_nodes
    }
    /// Returns the pivot limit of each relaxation solve.
    pub const fn max_lp_iterations(&self) -> Budget {
        self.max_lp_iterations
    }
    /// Returns the storage limits and optional work.
    pub const fn limits(&self) -> MipLimits {
        self.limits
    }
}
/// Holds the relaxation solver and search tree of [`BranchAndBound`].
///
/// [`BranchAndBound::workspace`] uses [`SimplexSolver`](crate::SimplexSolver)
/// with the solver's relaxation tolerances; [`Self::new`] installs a custom
/// [`MipRelaxationSolver`]. After a call, [`Self::state`] shows the open nodes
/// and the incumbent, also when a limit stopped the search.
pub struct BranchAndBoundWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    R: MipRelaxationSolver<S> = crate::SimplexSolver<S>,
> {
    relaxation: R,
    state: MipSearchState<S, R::Workspace>,
    staged: MipResult<S>,
    shell: crate::attempt::Shell<crate::ProgrammingProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>, R: MipRelaxationSolver<S>> core::fmt::Debug
    for BranchAndBoundWorkspace<S, R>
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BranchAndBoundWorkspace")
            .field("statistics", &self.state.statistics())
            .field("open_nodes", &self.state.open_nodes().len())
            .field("staged", &self.staged)
            .field("shell", &self.shell)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, R: MipRelaxationSolver<S>>
    BranchAndBoundWorkspace<S, R>
{
    /// Creates a workspace that solves relaxations with `relaxation`, without allocating.
    pub fn new(relaxation: R) -> Self {
        Self {
            relaxation,
            state: MipSearchState::new(),
            staged: MipResult::empty(),
            shell: crate::attempt::Shell::new(),
        }
    }
    /// Returns the search state of the most recent call: open nodes, incumbent and presolve map.
    pub const fn state(&self) -> &MipSearchState<S, R::Workspace> {
        &self.state
    }
    /// Returns the relaxation solver.
    pub const fn relaxation(&self) -> &R {
        &self.relaxation
    }
    /// Returns the record of the most recent call.
    pub const fn last_attempt(
        &self,
    ) -> Option<&scioxide_core::AttemptRecord<crate::ProgrammingProgress<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the incumbent written by the most recent attempt, if any.
    pub fn last_accepted(&self) -> Option<crate::ProgrammingAccepted<'_, S>> {
        crate::programming::last_accepted(&self.shell, &self.staged.x)
    }
}

impl<S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>>
    BranchAndBound<S>
{
    /// Creates a workspace that solves relaxations with [`SimplexSolver`](crate::SimplexSolver).
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn workspace(
        &self,
        problem: &MipProblem<S>,
    ) -> Result<BranchAndBoundWorkspace<S>, SciOxideError> {
        let mut workspace = BranchAndBoundWorkspace::new(
            crate::SimplexSolver::default().with_tolerances(self.relaxation_tolerances),
        );
        workspace
            .staged
            .try_reserve(problem.relaxation().dimension())?;
        Ok(workspace)
    }

    /// Solves `problem` on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_into`].
    pub fn solve(&self, problem: &MipProblem<S>) -> Result<MipResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem)?;
        let mut output = MipResult::empty();
        self.solve_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Runs a new search for `problem` into `output`, reusing `workspace`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent solve context. The node limit is [`Self::max_nodes`], and each
    /// relaxation runs as a nested solve limited by
    /// [`Self::max_lp_iterations`]. When a limit, cancellation or deadline
    /// stops the search, `output` holds the best incumbent with an unsuccessful
    /// termination, and the open nodes stay in [`BranchAndBoundWorkspace::state`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the relaxation solver returns a point or objective
    ///   that is not finite or does not match the problem.
    /// - `NumericalInstability` or `AllocationFailure` from a relaxation solve.
    /// - Errors returned by a custom [`MipRelaxationSolver`].
    pub fn solve_into<'scope, 'capability, R: MipRelaxationSolver<S>>(
        &self,
        problem: &MipProblem<S>,
        output: &mut MipResult<S>,
        workspace: &mut BranchAndBoundWorkspace<S, R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        use crate::attempt::Staged;
        workspace.shell.reject(scioxide_core::AttemptKind::Call);
        let scope = scope.into();
        workspace
            .staged
            .try_reserve(problem.relaxation().dimension())?;
        crate::attempt::preflight(&scope, None)?;
        let BranchAndBoundWorkspace {
            relaxation,
            state,
            staged,
            shell,
        } = workspace;
        state.reset();
        staged.reset();
        crate::programming::run(
            shell,
            staged,
            output,
            scope,
            scioxide_core::AttemptKind::Call,
            crate::attempt::Entry::Call {
                identity: crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "BranchAndBound", 1),
                    problem.relaxation().dimension(),
                    self,
                ),
                budgets: scioxide_core::SolveBudgets::unbounded()
                    .with_limit(scioxide_core::BudgetKind::Node, self.max_nodes),
            },
            |context, staged| {
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                let mut result = search::run(self, problem, state, relaxation, &session)?;
                result.algorithm = Some(session.algorithm()?);
                result.counts = session.accounting()?;
                session.assess(result.trust)?;
                *staged = result;
                Ok(())
            },
            MipResult::progress,
        )
    }
}
