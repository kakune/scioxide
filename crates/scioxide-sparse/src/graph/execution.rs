//! Execution options of the graph algorithms and the attempt records of their workspaces.
use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, ExecutionPolicy, ProblemToken, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState,
};

/// Sets the execution policy and the iteration and evaluation limits of a graph
/// algorithm.
///
/// The default is sequential execution without limits. Each algorithm states
/// what counts as one iteration; exhausting the iteration limit is a
/// `ConvergenceFailure`. Johnson's shortest-path method applies the limits
/// separately to its Bellman–Ford pass and to the search from each source. Those
/// searches run in nested scopes, so their iterations do not count toward the
/// limit of the enclosing call (see
/// [Solve accounting](scioxide_core#solve-accounting)).
///
/// ```
/// use scioxide_core::{Budget, BudgetKind, SolveBudgets};
/// use scioxide_sparse::GraphExecutionOptions;
/// let options = GraphExecutionOptions::new().with_budgets(
///     SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
/// assert_eq!(options.budgets().report(BudgetKind::Iteration).consumed(), 0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GraphExecutionOptions {
    execution: ExecutionPolicy,
    budgets: SolveBudgets,
}
impl Default for GraphExecutionOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl GraphExecutionOptions {
    /// Returns options for sequential execution without limits.
    pub const fn new() -> Self {
        Self {
            execution: ExecutionPolicy::Sequential,
            budgets: SolveBudgets::unbounded(),
        }
    }
    /// Sets the execution policy. Only the all-pairs shortest-path methods use
    /// more than one worker; matching, flow and assignment run sequentially
    /// under every policy.
    pub const fn with_execution_policy(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the iteration and evaluation limits of one call.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution
    }
    /// Returns the iteration and evaluation limits.
    pub const fn budgets(self) -> SolveBudgets {
        self.budgets
    }
    pub(super) fn workers(self, vertices: usize) -> usize {
        vertices.min(crate::execution::policy_workers(self.execution).get())
    }
}

#[derive(Debug, Default)]
pub(super) struct Attempt {
    state: SolveState,
    record: Option<AttemptRecord<()>>,
}
impl Attempt {
    pub(super) fn reject(&mut self) {
        self.record = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
    }
    pub(super) fn record(&self) -> Option<&AttemptRecord<()>> {
        self.record.as_ref()
    }
    pub(super) fn succeeded(&self) -> bool {
        self.record
            .as_ref()
            .is_some_and(|r| r.outcome() == AttemptOutcome::Succeeded)
    }
    pub(super) fn run<'scope, 'capability, T>(
        &mut self,
        scope: impl Into<SolveScope<'scope, 'capability>>,
        options: GraphExecutionOptions,
        method: scioxide_core::AlgorithmId,
        run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut entered = false;
        let result = scope.into().run(
            &mut self.state,
            identity(method),
            options.budgets(),
            |context| {
                entered = true;
                context.execution().check()?;
                let value = run(context)?;
                context.execution().check()?;
                Ok(value)
            },
        );
        if !entered {
            self.reject();
            return result;
        }
        self.record = Some(AttemptRecord::new(
            self.state.epoch(),
            AttemptKind::Call,
            if result.is_ok() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Failed
            },
            self.state.counts(),
            result.as_ref().ok().map(|_| ()),
        ));
        result
    }
}
pub(super) fn identity(method: scioxide_core::AlgorithmId) -> SolveIdentity {
    SolveIdentity::new(ProblemToken::new(0), method, 0, 0)
}

pub(super) fn grow<T: Clone>(
    values: &mut alloc::vec::Vec<T>,
    len: usize,
    fill: T,
) -> Result<(), SciOxideError> {
    values
        .try_reserve_exact(len.saturating_sub(values.len()))
        .map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("graph_workspace"), len)
        })?;
    if values.len() < len {
        values.resize(len, fill);
    }
    Ok(())
}
/// Check every numerical storage extent and report the first short extent.
pub(super) fn require<const N: usize>(extents: [(usize, usize); N]) -> Result<(), SciOxideError> {
    for (actual, required) in extents {
        crate::helpers::minimum(actual, required)?;
    }
    Ok(())
}
