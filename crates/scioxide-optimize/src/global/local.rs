//! Local solvers that refine the candidates of the global searches.
use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    BudgetKind, ConvergenceReason, NumericScalar, ProblemToken, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveState,
};

use super::multistart::SearchStartReport;
use crate::context::SolveSession;
use crate::validation::invalid;
use crate::{
    BoundedObjective, FiniteBounds, OptimizeResult, OptimizeTermination, Powell, PowellOptions,
    PowellWorkspace,
};

/// Refines a start point of a global search with a local minimizer.
///
/// Each call starts a new local solve: it must not depend on state from an
/// earlier call. The problem, including the search box, is passed on every
/// call. Implementations must evaluate the objective only through `context`,
/// so that every evaluation counts toward the limits of the global search.
pub trait SearchLocalSolver<S: RealScalar + NumericScalar<Real = S>> {
    /// Minimizes `problem` from `start` and writes the result to `output`.
    ///
    /// `seed` identifies the start, for local solvers that draw random
    /// numbers. `context` is a child of the global search's
    /// [solve context](scioxide_core#solve-accounting), so its evaluations also
    /// count toward the limits of the global search.
    ///
    /// # Errors
    ///
    /// Returns the local solver's errors unchanged. The global search records
    /// `ConvergenceFailure`, `NumericalInstability` and `Singular` in the
    /// start's report and continues; other errors stop the search.
    fn minimize_into(
        &mut self,
        problem: &BoundedObjective<'_, S>,
        start: &[S],
        seed: u64,
        output: &mut OptimizeResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Creates the local solvers that a global search uses to refine its starts.
///
/// The search prepares one solver per start, so different starts never share
/// solver state. The factory's `Debug` output identifies its configuration: a
/// workspace prepared with a differently configured factory is prepared again.
pub trait LocalSolverFactory<S: RealScalar + NumericScalar<Real = S>>: core::fmt::Debug {
    /// Local solver with its storage.
    type Solver: SearchLocalSolver<S> + Send;
    /// Creates a solver for `dimension` variables, before any objective
    /// evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported dimension or configuration, or when
    /// storage cannot be allocated.
    fn prepare(&self, dimension: usize) -> Result<Self::Solver, SciOxideError>;
}

/// Creates [`Powell`](crate::Powell) local solvers, bounded by the search box.
#[derive(Debug, Clone)]
pub struct PowellLocalFactory<S: RealScalar> {
    options: PowellOptions<S>,
}
impl<S: RealScalar> Default for PowellLocalFactory<S> {
    fn default() -> Self {
        Self::new(PowellOptions::default())
    }
}
impl<S: RealScalar> PowellLocalFactory<S> {
    /// Creates a factory whose solvers use `options`. Each local problem is
    /// bounded by the search box.
    pub const fn new(options: PowellOptions<S>) -> Self {
        Self { options }
    }
    /// Returns the options of the local Powell solvers.
    pub const fn options(&self) -> &PowellOptions<S> {
        &self.options
    }
}
/// Runs a [`Powell`](crate::Powell) local solve; created by [`PowellLocalFactory`].
///
/// Each local solve minimizes the global problem inside the search box.
#[derive(Debug)]
pub struct PowellSearchLocal<S: RealScalar> {
    solver: Powell<S>,
    work: PowellWorkspace<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> LocalSolverFactory<S> for PowellLocalFactory<S> {
    type Solver = PowellSearchLocal<S>;
    fn prepare(&self, dimension: usize) -> Result<Self::Solver, SciOxideError> {
        let solver = Powell::new(self.options.clone())?;
        let work = solver.workspace(dimension)?;
        Ok(PowellSearchLocal { solver, work })
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> SearchLocalSolver<S> for PowellSearchLocal<S> {
    fn minimize_into(
        &mut self,
        problem: &BoundedObjective<'_, S>,
        start: &[S],
        _: u64,
        output: &mut OptimizeResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.solver
            .prepare(problem, start, &mut self.work.storage)?;
        self.solver
            .minimize_in_context(problem, output, &mut self.work.storage, context)
    }
}

/// Prepared local solvers bound to one factory configuration and dimension.
pub(super) struct LocalPool<L> {
    pub(super) locals: Vec<L>,
    token: u128,
    dimension: usize,
}
impl<L> LocalPool<L> {
    pub(super) const fn new() -> Self {
        Self {
            locals: Vec::new(),
            token: 0,
            dimension: 0,
        }
    }
    pub(super) fn fits(&self, count: usize, dimension: usize, token: u128) -> bool {
        self.locals.len() == count && self.dimension == dimension && self.token == token
    }
    /// Prepare `count` solvers; on failure the previous solvers stay in place.
    pub(super) fn prepare<S, F>(
        &mut self,
        factory: &F,
        count: usize,
        dimension: usize,
        token: u128,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: LocalSolverFactory<S, Solver = L>,
    {
        if self.fits(count, dimension, token) {
            return Ok(());
        }
        let mut locals = try_with_capacity(count, scioxide_core::quantity!("search_local_states"))?;
        for _ in 0..count {
            locals.push(factory.prepare(dimension)?);
        }
        *self = Self {
            locals,
            token,
            dimension,
        };
        Ok(())
    }
}

/// Identity of one local child epoch: start index, dimension and factory configuration.
pub(super) fn local_identity(index: usize, dimension: usize, token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(index as u128),
        scioxide_core::algorithm_id!("optimize", "SearchLocal", 1),
        dimension as u128,
        token,
    )
}

/// Termination of a local solve stopped by a resource budget.
///
/// Uses the stored budget report, or the budget named by the convergence
/// reason when the error carries no budget report.
pub(super) fn budget_termination(error: &SciOxideError) -> Option<OptimizeTermination> {
    let scioxide_core::SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        ..
    } = error.view()
    else {
        return None;
    };
    let kind = diagnostics
        .budget()
        .map(|budget| budget.kind())
        .or(match reason {
            ConvergenceReason::IterationLimit => Some(BudgetKind::Iteration),
            ConvergenceReason::EvaluationLimit => Some(BudgetKind::Evaluation),
            ConvergenceReason::ResourceLimit(kind) => Some(kind),
            _ => None,
        });
    kind.map(|kind| match kind {
        BudgetKind::Iteration => OptimizeTermination::IterationLimit,
        BudgetKind::Proposal => OptimizeTermination::ProposalLimit,
        BudgetKind::Sample => OptimizeTermination::SampleLimit,
        _ => OptimizeTermination::EvaluationLimit,
    })
}

/// Local failures recorded per start instead of failing the search.
pub(super) fn recoverable(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            | scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            | scioxide_core::SciOxideErrorView::Singular { .. }
    )
}

/// Validate a local result before it can become a search candidate.
pub(super) fn validate_local<S: RealScalar + NumericScalar<Real = S>>(
    result: &OptimizeResult<S>,
    bounds: &FiniteBounds<S>,
) -> Result<(), SciOxideError> {
    if !bounds.contains(result.x())
        || !result.fun().is_finite()
        || result.termination() == OptimizeTermination::NotStarted
    {
        return Err(invalid(
            scioxide_core::argument!("factory"),
            "local solver returned an invalid result",
        ));
    }
    Ok(())
}

/// Refine one start in a child epoch with the configured local allowance.
///
/// Child budget stops do not stop the search while the ancestor has resources;
/// recoverable local failures are recorded in the start report.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_local<S: RealScalar + NumericScalar<Real = S>, L: SearchLocalSolver<S>>(
    local: &mut L,
    report: &mut SearchStartReport<S>,
    index: usize,
    problem: &BoundedObjective<'_, S>,
    bounds: &FiniteBounds<S>,
    allowance: SolveBudgets,
    token: u128,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let mut state = SolveState::new();
    let identity = local_identity(index, bounds.len(), token);
    let result = session.with_child(&mut state, identity, allowance, |context| {
        let outcome = local.minimize_into(
            problem,
            &report.start,
            report.seed,
            &mut report.candidate,
            context,
        );
        report.counts = context.counts();
        outcome
    });
    let stop = match &result {
        Ok(()) => Some(report.candidate.termination()),
        Err(error) => budget_termination(error),
    };
    if stop.is_some_and(|stop| {
        matches!(
            stop,
            OptimizeTermination::IterationLimit
                | OptimizeTermination::ProposalLimit
                | OptimizeTermination::EvaluationLimit
                | OptimizeTermination::SampleLimit
        )
    }) {
        session.inherit_child_stop(stop)?;
    }
    // A local deadline does not expire an ancestor with a later/no deadline.
    let _ = session.termination()?;
    match result {
        Ok(()) => {
            validate_local(&report.candidate, bounds)?;
            report.has_candidate = true;
            report.local_done = true;
        }
        Err(error)
            if recoverable(&error)
                || matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                ) =>
        {
            report.error = Some(error);
        }
        Err(error) => return Err(error),
    }
    Ok(())
}
