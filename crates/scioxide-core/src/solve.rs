//! Solve identity, work counting, budgets and checkpoints for iterative solvers.
//!
//! See [solve accounting](crate#solve-accounting) for how nested solves share
//! limits.

use core::fmt;
mod metadata;
pub use metadata::{
    AcceptedStateMetadata, AlgorithmId, EstimateReliability, PartialResultTrust, ProblemToken,
    ReliableEstimate, ResidualReport, ResidualScaling, ResidualScalingRule, SolveEpoch,
    SolveIdentity,
};

#[cfg(test)]
mod accounting_tests;

use crate::{Budget, ExecutionContext, ExecutionStopReason, InvalidInputReason, SciOxideError};

// One declaration owns variant order, dense indexing, names, and ALL.
macro_rules! indexed_roles {
    ($(#[$meta:meta])* $name:ident { $($(#[$doc:meta])* $variant:ident => $label:literal, $counter:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        #[non_exhaustive]
        pub enum $name { $($(#[$doc])* $variant),+ }
        impl $name {
            /// All roles, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            /// Dense internal index derived from the declaration.
            const fn index(self) -> usize { self as usize }
            /// Returns the stable diagnostic name of the role.
            pub const fn name(self) -> &'static str { match self { $(Self::$variant => $label),+ } }
            const fn counter_context(self) -> crate::Quantity {
                match self { $(Self::$variant => crate::quantity!($counter)),+ }
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.name()) }
        }
    };
}
indexed_roles! {
    /// Role of a numerical callback or provider invocation.
    EvaluationRole {
    /// Scalar or vector mathematical function.
    Function => "function", "solve.function.count",
    /// Optimization objective.
    Objective => "objective", "solve.objective.count",
    /// Residual provider.
    Residual => "residual", "solve.residual.count",
    /// Gradient provider.
    Gradient => "gradient", "solve.gradient.count",
    /// Jacobian provider.
    Jacobian => "jacobian", "solve.jacobian.count",
    /// Hessian provider.
    Hessian => "hessian", "solve.hessian.count",
    /// Constraint provider.
    Constraint => "constraint", "solve.constraint.count",
    /// ODE/PDE right-hand side.
    RightHandSide => "right-hand-side", "solve.right_hand_side.count",
    /// Event function.
    Event => "event", "solve.event.count",
    /// Linear operator application.
    LinearOperator => "linear-operator", "solve.linear_operator.count",
    /// Preconditioner application.
    Preconditioner => "preconditioner", "solve.preconditioner.count",
    /// Random or quasi-random sample provider.
    Sample => "sample", "solve.sample.count",
    /// Matrix factorization.
    Factorization => "factorization", "solve.factorization.count",
    /// Linear solve.
    LinearSolve => "linear-solve", "solve.linear_solve.count",
    /// Mathematical-domain predicate supplied by a problem.
    Domain => "domain", "solve.domain.count",
    /// Jacobian provider for a constraint row or block.
    ConstraintJacobian => "constraint-jacobian", "solve.constraint_jacobian.count",
    }
}
const EVALUATION_ROLE_COUNT: usize = EvaluationRole::ALL.len();

/// Counts of iterations, proposals and callback invocations by role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct EvaluationCounts {
    iterations: usize,
    proposals: usize,
    accepted: usize,
    rejected: usize,
    cache_hits: usize,
    providers: [usize; EVALUATION_ROLE_COUNT],
}

impl EvaluationCounts {
    /// Returns zero counts for a fresh solve epoch.
    #[inline]
    pub const fn new() -> Self {
        Self {
            iterations: 0,
            proposals: 0,
            accepted: 0,
            rejected: 0,
            cache_hits: 0,
            providers: [0; EVALUATION_ROLE_COUNT],
        }
    }

    /// Returns counts for an algorithm that reports only its iteration count.
    pub const fn from_iterations(iterations: usize) -> Self {
        Self {
            iterations,
            proposals: 0,
            accepted: 0,
            rejected: 0,
            cache_hits: 0,
            providers: [0; EVALUATION_ROLE_COUNT],
        }
    }

    /// Returns a copy with the count of one callback role set to `count`.
    ///
    /// Use it to convert counts recorded elsewhere into this report. Live
    /// solves count callbacks through [`SolveContext`] instead.
    pub const fn with_provider_count(mut self, role: EvaluationRole, count: usize) -> Self {
        self.providers[role.index()] = count;
        self
    }

    /// Returns a copy with the given proposal, acceptance and rejection counts.
    ///
    /// Like [`Self::with_provider_count`], this converts counts recorded
    /// elsewhere; live solves count through [`SolveContext`].
    ///
    /// ```
    /// use scioxide_core::EvaluationCounts;
    ///
    /// let counts = EvaluationCounts::from_iterations(3).with_proposals(3, 1, 2);
    /// assert_eq!(
    ///     (counts.proposals(), counts.accepted(), counts.rejected()),
    ///     (3, 1, 2)
    /// );
    /// ```
    pub const fn with_proposals(
        mut self,
        proposals: usize,
        accepted: usize,
        rejected: usize,
    ) -> Self {
        self.proposals = proposals;
        self.accepted = accepted;
        self.rejected = rejected;
        self
    }

    /// Returns the total number of callback invocations across all roles.
    pub fn total_providers(self) -> Result<usize, SciOxideError> {
        self.provider_sum().ok_or_else(|| {
            SciOxideError::size_overflow(crate::quantity!("total_provider_evaluations"))
        })
    }

    #[inline]
    const fn provider_sum(&self) -> Option<usize> {
        let mut sum = 0_usize;
        let mut index = 0;
        while index < EVALUATION_ROLE_COUNT {
            sum = match sum.checked_add(self.providers[index]) {
                Some(value) => value,
                None => return None,
            };
            index += 1;
        }
        Some(sum)
    }

    /// Returns the number of attempted iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }

    /// Returns the number of proposed states.
    pub const fn proposals(self) -> usize {
        self.proposals
    }

    /// Returns the number of accepted proposals.
    pub const fn accepted(self) -> usize {
        self.accepted
    }

    /// Returns the number of rejected proposals.
    pub const fn rejected(self) -> usize {
        self.rejected
    }

    /// Returns the number of cache hits; a cache hit is not a callback invocation.
    pub const fn cache_hits(self) -> usize {
        self.cache_hits
    }

    /// Returns the number of callback invocations for `role`.
    pub const fn provider(self, role: EvaluationRole) -> usize {
        self.providers[role.index()]
    }

    /// Returns the sum of two sets of counts.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow`, naming the first counter whose sum exceeds `usize::MAX`.
    #[inline]
    pub fn merge(self, other: Self) -> Result<Self, SciOxideError> {
        let mut next = self;
        for counter in Counter::all() {
            counter.write(
                &mut next,
                counter.checked_merge(counter.read(&self), counter.read(&other))?,
            );
        }
        next.total_providers()?;
        Ok(next)
    }
}

impl fmt::Display for EvaluationCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "iterations={}, evaluations=", self.iterations)?;
        match self.total_providers() {
            Ok(total) => write!(f, "{total}")?,
            Err(_) => f.write_str("overflow")?,
        }
        for (name, value) in [
            ("proposals", self.proposals),
            ("accepted", self.accepted),
            ("rejected", self.rejected),
            ("cache-hits", self.cache_hits),
        ] {
            if value != 0 {
                write!(f, ", {name}={value}")?;
            }
        }
        for &role in EvaluationRole::ALL {
            if self.provider(role) != 0 {
                write!(f, ", {role}={}", self.provider(role))?;
            }
        }
        Ok(())
    }
}

impl Default for EvaluationCounts {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagation policy for a solve resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NestingPolicy {
    /// Consumption belongs only to this solve's budget.
    ScopeLocal,
    /// Consumption also counts toward the limits of every enclosing solve.
    Shared,
}
indexed_roles! {
    /// Role of a finite solve resource.
    BudgetKind {
        /// Solver iterations, local to one solve.
        Iteration => "iteration", "solve.iteration.count",
        /// Callback invocations, shared with nested solves.
        Evaluation => "evaluation", "solve.evaluation.count",
        /// Candidate proposals, local to one solve.
        Proposal => "proposal", "solve.proposal.count",
        /// Adaptive subdivisions, shared with nested solves.
        Subdivision => "subdivision", "solve.subdivision.count",
        /// Random samples, shared with nested solves.
        Sample => "sample", "solve.sample.count",
        /// Search-tree nodes, shared with nested solves.
        Node => "node", "solve.node.count",
    }
}
impl BudgetKind {
    /// Exhaustive per-kind nesting policy.
    pub(crate) const fn nesting(self) -> NestingPolicy {
        match self {
            Self::Iteration | Self::Proposal => NestingPolicy::ScopeLocal,
            Self::Evaluation | Self::Subdivision | Self::Sample | Self::Node => {
                NestingPolicy::Shared
            }
        }
    }
}
const BUDGET_KIND_COUNT: usize = BudgetKind::ALL.len();

/// The limit of one budget kind: finite (possibly zero) or unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetLimit {
    /// A finite limit; zero is possible for a child's share in a partitioned batch.
    Finite(usize),
    /// No configured limit; counts are still bounded by `usize::MAX`.
    Unbounded,
}
impl fmt::Display for BudgetLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finite(n) => n.fmt(f),
            Self::Unbounded => f.write_str("unbounded"),
        }
    }
}

/// Outcome of checking or recording a unit of work against the solve budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Admission<T = ()> {
    /// All applicable limits and counter capacities permit the work.
    Granted(T),
    /// A configured limit denied the work; no count changed.
    Denied(BudgetReport),
}
impl<T> Admission<T> {
    /// Returns whether the work was granted.
    pub const fn is_granted(&self) -> bool {
        matches!(self, Self::Granted(_))
    }

    /// Returns the granted value, or an error if a limit denied the work.
    ///
    /// Use it where a partial result cannot be returned.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` for a denial, with reason `IterationLimit`,
    ///   `EvaluationLimit` or `ResourceLimit` and the limiting [`BudgetReport`].
    #[inline(always)]
    pub fn require(self) -> Result<T, SciOxideError> {
        match self {
            Self::Granted(value) => Ok(value),
            Self::Denied(report) => Err(SciOxideError::convergence_failure(
                match report.kind() {
                    BudgetKind::Iteration => crate::ConvergenceReason::IterationLimit,
                    BudgetKind::Evaluation => crate::ConvergenceReason::EvaluationLimit,
                    kind => crate::ConvergenceReason::ResourceLimit(kind),
                },
                crate::ConvergenceDiagnostics::new().with_budget(report),
                "solve allowance denied further work",
            )),
        }
    }
}

/// Outcome of a counted callback invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Invocation<T> {
    /// The callback ran and returned this value.
    Evaluated(T),
    /// A budget of this solve or a nested solve denied the work; no value is available.
    BudgetExhausted(BudgetReport),
    /// Cancellation or the deadline stopped the work before it completed.
    ExecutionStopped(ExecutionStopReason),
}
impl<T> Invocation<T> {
    /// Applies `map` to an evaluated value and passes other outcomes through unchanged.
    ///
    /// ```
    /// let value = scioxide_core::Invocation::Evaluated(2).map(|n| n + 1);
    /// assert_eq!(value.require()?, 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Invocation<U> {
        match self {
            Self::Evaluated(value) => Invocation::Evaluated(map(value)),
            Self::BudgetExhausted(report) => Invocation::BudgetExhausted(report),
            Self::ExecutionStopped(reason) => Invocation::ExecutionStopped(reason),
        }
    }

    /// Returns the evaluated value, or an error where a partial result cannot be returned.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` if a budget denied the invocation.
    /// - `ExecutionStopped` if cancellation or a deadline stopped it.
    pub fn require(self) -> Result<T, SciOxideError> {
        match self {
            Self::Evaluated(value) => Ok(value),
            Self::BudgetExhausted(report) => Admission::Denied(report).require(),
            Self::ExecutionStopped(reason) => Err(SciOxideError::execution_stopped(reason)),
        }
    }
    /// Returns whether the callback ran and returned a value.
    pub const fn is_evaluated(&self) -> bool {
        matches!(self, Self::Evaluated(_))
    }
    /// Returns whether a local or nested budget prevented the invocation.
    pub const fn is_exhausted(&self) -> bool {
        matches!(self, Self::BudgetExhausted(_))
    }
}

/// Limit and consumption of one budget kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct BudgetReport {
    kind: BudgetKind,
    limit: BudgetLimit,
    consumed: usize,
}
impl BudgetReport {
    /// Returns the budget kind.
    pub const fn kind(self) -> BudgetKind {
        self.kind
    }
    /// Returns the limit; a finite zero differs from unbounded.
    pub const fn limit(self) -> BudgetLimit {
        self.limit
    }
    /// Returns the units consumed by this solve and, for shared kinds, its nested solves.
    pub const fn consumed(self) -> usize {
        self.consumed
    }
    /// Returns the remaining finite allowance, or `None` if unbounded.
    pub const fn remaining(self) -> Option<usize> {
        match self.limit {
            BudgetLimit::Finite(limit) => Some(limit.saturating_sub(self.consumed)),
            BudgetLimit::Unbounded => None,
        }
    }
    /// Returns whether a finite limit has been reached.
    pub const fn exhausted(self) -> bool {
        matches!(self.limit, BudgetLimit::Finite(limit) if self.consumed >= limit)
    }
}

/// Whether the recorded counts describe all work of the solve epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountingIntegrity {
    /// Every counted unit of work is recorded exactly.
    Exact,
    /// Inconsistent counts from a partitioned solve could not be merged.
    Invalid,
}

/// Work counted by a solve itself and by its nested solves, kept separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SolveCounts {
    own: EvaluationCounts,
    nested: EvaluationCounts,
    // Exact aggregate for sparse admission; None retains overflowing diagnostics.
    // Public totals and record validation independently inspect the role counters.
    provider_total: Option<usize>,
    integrity: AccountingIntegrity,
}
impl SolveCounts {
    /// Iteration-only counts for this solve and its nested solves, kept separately.
    pub(crate) const fn from_iteration_counts(own: usize, nested: usize) -> Self {
        Self {
            own: EvaluationCounts::from_iterations(own),
            nested: EvaluationCounts::from_iterations(nested),
            provider_total: Some(0),
            integrity: AccountingIntegrity::Exact,
        }
    }
    /// Returns zero counts.
    #[inline]
    pub const fn new() -> Self {
        Self {
            own: EvaluationCounts::new(),
            nested: EvaluationCounts::new(),
            provider_total: Some(0),
            integrity: AccountingIntegrity::Exact,
        }
    }
    /// Returns counts with `own` as this solve's work and no nested work.
    ///
    /// The fields are kept even if their combined total overflows.
    #[inline]
    pub const fn from_own(own: EvaluationCounts) -> Self {
        Self {
            own,
            nested: EvaluationCounts::new(),
            provider_total: own.provider_sum(),
            integrity: AccountingIntegrity::Exact,
        }
    }
    /// Returns the work performed directly in this solve.
    pub const fn own(self) -> EvaluationCounts {
        self.own
    }
    /// Consult `integrity` before treating this as a complete epoch summary.
    pub const fn nested(self) -> EvaluationCounts {
        self.nested
    }
    /// Returns whether the counts describe all work of the epoch.
    pub const fn integrity(self) -> AccountingIntegrity {
        self.integrity
    }
    /// Returns the sum of this solve's work and its nested solves' work.
    ///
    /// # Errors
    ///
    /// - `AccountingInvariant` if [`Self::integrity`] is not exact.
    /// - `SizeOverflow` if a total exceeds `usize::MAX`.
    #[inline]
    pub fn total(self) -> Result<EvaluationCounts, SciOxideError> {
        self.ensure_exact()?;
        self.own.merge(self.nested)
    }

    #[inline]
    fn ensure_exact(&self) -> Result<(), SciOxideError> {
        match self.integrity {
            AccountingIntegrity::Exact => Ok(()),
            AccountingIntegrity::Invalid => Err(accounting_invariant(
                "incomplete accounting begin a fresh epoch before further work",
            )),
        }
    }
}
impl From<EvaluationCounts> for SolveCounts {
    fn from(counts: EvaluationCounts) -> Self {
        Self::from_own(counts)
    }
}
impl Default for SolveCounts {
    fn default() -> Self {
        Self::new()
    }
}

/// Limits and consumption for each [`BudgetKind`]; finite limits are positive [`Budget`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SolveBudgets {
    limits: [BudgetLimit; BUDGET_KIND_COUNT],
    consumed: [usize; BUDGET_KIND_COUNT],
}
impl SolveBudgets {
    /// Returns budgets with no limits; counters are still bounded by `usize::MAX`.
    #[inline]
    pub const fn unbounded() -> Self {
        Self {
            limits: [BudgetLimit::Unbounded; BUDGET_KIND_COUNT],
            consumed: [0; BUDGET_KIND_COUNT],
        }
    }
    /// Returns a copy with a limit for `kind`.
    pub const fn with_limit(mut self, kind: BudgetKind, limit: Budget) -> Self {
        self.limits[kind.index()] = BudgetLimit::Finite(limit.get());
        self
    }
    /// Consumes `amount` units of `kind` if the limit allows it.
    ///
    /// A denial changes nothing.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the consumed count would exceed `usize::MAX`.
    #[inline]
    pub fn try_consume(
        &mut self,
        kind: BudgetKind,
        amount: usize,
    ) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::resource(kind, amount))
    }
    /// Returns the limit and consumption of `kind`.
    pub const fn report(self, kind: BudgetKind) -> BudgetReport {
        BudgetReport {
            kind,
            limit: self.limits[kind.index()],
            consumed: self.consumed[kind.index()],
        }
    }
}
impl Default for SolveBudgets {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Versioned solve metadata stored in a solver's checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SolveCheckpointMetadata {
    schema_version: u32,
    identity: SolveIdentity,
    accepted: AcceptedStateMetadata,
    counts: SolveCounts,
    budgets: SolveBudgets,
    state_token: u128,
}

impl SolveCheckpointMetadata {
    /// Current checkpoint schema version.
    pub const SCHEMA_VERSION: u32 = 2;

    /// Returns the solve identity stored in the checkpoint.
    pub const fn identity(self) -> SolveIdentity {
        self.identity
    }

    /// Returns the metadata of the last accepted state.
    pub const fn accepted(self) -> AcceptedStateMetadata {
        self.accepted
    }

    /// Returns the counts when the checkpoint was taken.
    pub const fn counts(self) -> SolveCounts {
        self.counts
    }

    /// Returns the budgets when the checkpoint was taken.
    pub const fn budgets(self) -> SolveBudgets {
        self.budgets
    }

    /// Returns the solver's token that identifies the checkpointed numerical state.
    pub const fn state_token(self) -> u128 {
        self.state_token
    }
}

/// Reusable solve state: epoch, counts and accepted-state metadata.
///
/// Plans and workspaces store a `SolveState` and create a new [`SolveContext`]
/// from it for each solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SolveState {
    next_epoch: u64,
    identity: Option<SolveIdentity>,
    accepted: Option<AcceptedStateMetadata>,
    counts: SolveCounts,
}

impl SolveState {
    #[inline(always)]
    fn active(&mut self) -> Result<ActiveSolveState<'_>, SciOxideError> {
        let identity = self.identity.as_ref().ok_or_else(|| {
            SciOxideError::invalid_input(
                crate::argument!("self"),
                InvalidInputReason::Malformed,
                "solve state has no active identity",
            )
        })?;
        Ok(ActiveSolveState {
            identity,
            accepted: &mut self.accepted,
            counts: &mut self.counts,
        })
    }

    /// Creates state with no active solve.
    pub const fn new() -> Self {
        Self {
            next_epoch: 1,
            identity: None,
            accepted: None,
            counts: SolveCounts::new(),
        }
    }

    /// Returns the current epoch, or `None` before the first solve.
    pub const fn epoch(&self) -> Option<SolveEpoch> {
        match self.accepted {
            Some(accepted) => Some(accepted.epoch),
            None => None,
        }
    }
    /// Returns the work counted in the current epoch.
    pub const fn counts(&self) -> SolveCounts {
        self.counts
    }

    /// Starts a new solve epoch and returns its context.
    ///
    /// Counts and accepted-state metadata are reset. The context borrows this
    /// state and the execution context, so it cannot outlive the call that
    /// uses it:
    ///
    /// ```compile_fail
    /// use scioxide_core::{ExecutionContext, SolveBudgets, SolveContext, SolveIdentity, SolveState};
    ///
    /// fn escape<'a>(state: &'a mut SolveState, budgets: &'a mut SolveBudgets, identity: SolveIdentity)
    ///     -> SolveContext<'static, 'static>
    /// {
    ///     state.begin_solve(ExecutionContext::sequential(), identity, budgets).unwrap()
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the epoch counter is exhausted.
    #[inline(always)]
    pub fn begin_solve<'state, 'capability>(
        &'state mut self,
        execution: ExecutionContext<'capability>,
        identity: SolveIdentity,
        budgets: &'state mut SolveBudgets,
    ) -> Result<SolveContext<'state, 'capability>, SciOxideError> {
        let epoch = SolveEpoch(self.next_epoch);
        self.next_epoch = self
            .next_epoch
            .checked_add(1)
            .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("solve_epoch")))?;
        self.identity = Some(identity);
        self.accepted = Some(AcceptedStateMetadata {
            epoch,
            revision: 0,
            trust: PartialResultTrust::DiagnosticOnly,
        });
        self.counts = SolveCounts::new();
        // Limits persist across solves; consumption is per epoch.
        budgets.consumed.fill(0);
        Ok(SolveContext {
            execution,
            state: self.active()?,
            budgets,
            parent: None,
            headroom_limit: usize::MAX,
        })
    }

    /// Continues the solve recorded in `checkpoint` instead of starting a new epoch.
    ///
    /// The checkpoint is validated before the state or budgets change.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the schema version, solve identity or state token
    ///   does not match.
    pub fn resume<'state, 'capability>(
        &'state mut self,
        execution: ExecutionContext<'capability>,
        expected: SolveIdentity,
        expected_state_token: u128,
        checkpoint: SolveCheckpointMetadata,
        budgets: &'state mut SolveBudgets,
    ) -> Result<SolveContext<'state, 'capability>, SciOxideError> {
        if checkpoint.schema_version != SolveCheckpointMetadata::SCHEMA_VERSION
            || checkpoint.identity != expected
            || checkpoint.state_token != expected_state_token
        {
            return Err(SciOxideError::invalid_input(
                crate::argument!("checkpoint"),
                InvalidInputReason::Inconsistent,
                "checkpoint version, identity, shape, options, or state token does not match",
            ));
        }
        validate_record(checkpoint.budgets, checkpoint.counts)?;
        let next_epoch = checkpoint
            .accepted
            .epoch
            .0
            .checked_add(1)
            .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("solve_epoch")))?;
        self.identity = Some(checkpoint.identity);
        self.accepted = Some(checkpoint.accepted);
        self.counts = checkpoint.counts;
        // Later fresh solves must not reuse an epoch number from the restored
        // lineage, or stale accepted-state metadata could look current.
        self.next_epoch = self.next_epoch.max(next_epoch);
        *budgets = checkpoint.budgets;
        Ok(SolveContext {
            execution,
            state: self.active()?,
            budgets,
            parent: None,
            headroom_limit: usize::MAX,
        })
    }
}

impl Default for SolveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a solve runs: at the top level, or nested inside another solve.
///
/// A scope built from an [`ExecutionContext`] starts a top-level solve. A scope
/// built from a `&mut SolveContext` starts a nested solve whose shared work
/// counts toward the parent's limits; see
/// [solve accounting](crate#solve-accounting). The nested solve's epoch,
/// accepted state and local limits belong to its own [`SolveState`].
///
/// ```
/// use scioxide_core::{
///     ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveScope, SolveState,
///     algorithm_id,
/// };
///
/// let identity =
///     SolveIdentity::new(ProblemToken::new(1), algorithm_id!("example", "kernel", 1), 0, 0);
/// let mut state = SolveState::new();
/// SolveScope::from(ExecutionContext::sequential())
///     .run(&mut state, identity, SolveBudgets::unbounded(), |cx| {
///         assert_eq!(cx.accepted_state().unwrap().epoch().get(), 1);
///         Ok(())
///     })
///     .unwrap();
/// ```
#[derive(Debug)]
#[must_use]
pub struct SolveScope<'scope, 'capability> {
    execution: ExecutionContext<'capability>,
    parent: Option<&'scope mut dyn SolveLedger>,
}

impl<'capability> From<ExecutionContext<'capability>> for SolveScope<'_, 'capability> {
    #[inline]
    fn from(execution: ExecutionContext<'capability>) -> Self {
        Self {
            execution,
            parent: None,
        }
    }
}
impl<'scope, 'state, 'capability> From<&'scope mut SolveContext<'state, 'capability>>
    for SolveScope<'scope, 'capability>
{
    #[inline]
    fn from(parent: &'scope mut SolveContext<'state, 'capability>) -> Self {
        Self {
            execution: parent.execution(),
            parent: Some(parent),
        }
    }
}
impl<'capability> SolveScope<'_, 'capability> {
    /// Returns the execution context inherited from the enclosing solve.
    pub const fn execution(&self) -> ExecutionContext<'capability> {
        self.execution
    }

    /// Returns a shorter-lived scope for passing to one nested operation.
    ///
    /// Reborrowing starts no epoch and counts no work. The original scope is
    /// usable again once the reborrowed one is consumed or dropped.
    pub fn reborrow(&mut self) -> SolveScope<'_, 'capability> {
        SolveScope {
            execution: self.execution,
            parent: self
                .parent
                .as_mut()
                .map(|parent| &mut **parent as &mut dyn SolveLedger),
        }
    }

    /// Returns the scope with a deadline `duration` from now, keeping any earlier deadline.
    ///
    /// The parent solve is not changed.
    ///
    /// # Errors
    ///
    /// - `CapabilityUnavailable` if no clock is bound.
    /// - `CapabilityFailure` if the clock fails or the deadline overflows.
    pub fn with_time_limit(
        mut self,
        duration: core::time::Duration,
    ) -> Result<Self, SciOxideError> {
        self.execution = self.execution.with_time_limit(duration)?;
        Ok(self)
    }

    /// Starts a new epoch in `state` and calls `run` with its context.
    ///
    /// `allowance` sets the local limits. Shared work, including failed callback
    /// invocations, also counts toward the enclosing solves' limits.
    ///
    /// # Errors
    ///
    /// - `AccountingInvariant` if the enclosing solve's counts are inconsistent.
    /// - `SizeOverflow` if the epoch counter is exhausted.
    /// - Any error returned by `run`, unchanged.
    #[inline(always)]
    pub fn run<T>(
        mut self,
        state: &mut SolveState,
        identity: SolveIdentity,
        allowance: SolveBudgets,
        run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        if let Some(parent) = &self.parent {
            parent.counts_ref().ensure_exact()?;
        }
        let mut budgets = allowance;
        let mut child = state.begin_solve(self.execution, identity, &mut budgets)?;
        if let Some(parent) = &mut self.parent {
            child.parent = Some(&mut **parent);
        }
        let outcome = run(&mut child);
        finish_scope(&mut child, outcome)
    }

    /// Continues the solve recorded in `checkpoint` and calls `run` with its context.
    ///
    /// New work counts toward the enclosing solves as in [`Self::run`]. The
    /// checkpoint is validated before anything changes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the checkpoint does not match, as in [`SolveState::resume`].
    /// - Any error returned by `run`, unchanged.
    pub fn resume<T>(
        mut self,
        state: &mut SolveState,
        expected: SolveIdentity,
        expected_state_token: u128,
        checkpoint: SolveCheckpointMetadata,
        budgets: &mut SolveBudgets,
        run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        if let Some(parent) = &self.parent {
            parent.counts_ref().ensure_exact()?;
        }
        let mut child = state.resume(
            self.execution,
            expected,
            expected_state_token,
            checkpoint,
            budgets,
        )?;
        if let Some(parent) = &mut self.parent {
            child.parent = Some(&mut **parent);
        }
        let outcome = run(&mut child);
        finish_scope(&mut child, outcome)
    }
}

/// The running state of one solve: its execution context, budgets and work counts.
///
/// A solver records each iteration, proposal and callback invocation through
/// the context before doing the work. A `record_*` method returns
/// [`Admission::Denied`] when a limit would be exceeded, and then changes no
/// count. See [solve accounting](crate#solve-accounting).
///
/// The context borrows its [`SolveState`] and is created anew for each solve
/// by [`SolveState::begin_solve`] or [`SolveScope::run`].
#[derive(Debug)]
#[must_use]
pub struct SolveContext<'state, 'capability> {
    execution: ExecutionContext<'capability>,
    state: ActiveSolveState<'state>,
    budgets: &'state mut SolveBudgets,
    parent: Option<&'state mut dyn SolveLedger>,
    headroom_limit: usize,
}

/// Borrowed fields of the active epoch.
#[derive(Debug)]
struct ActiveSolveState<'state> {
    identity: &'state SolveIdentity,
    accepted: &'state mut Option<AcceptedStateMetadata>,
    counts: &'state mut SolveCounts,
}

impl<'state, 'capability> SolveContext<'state, 'capability> {
    /// Returns the execution context of this solve.
    pub const fn execution(&self) -> ExecutionContext<'capability> {
        self.execution
    }

    /// Returns the identity of the active solve.
    pub const fn identity(&self) -> SolveIdentity {
        *self.state.identity
    }

    /// Returns the work counted in this epoch.
    pub const fn counts(&self) -> SolveCounts {
        *self.state.counts
    }

    /// Returns the invocations of `role` by this solve and its nested solves.
    ///
    /// This reads one counter and is cheaper than [`SolveCounts::total`].
    ///
    /// # Errors
    ///
    /// - `AccountingInvariant` if the epoch's counts are incomplete.
    /// - `SizeOverflow` if the combined count exceeds `usize::MAX`.
    ///
    /// ```
    /// fn objectives(
    ///     c: &scioxide_core::SolveContext<'_, '_>,
    /// ) -> Result<usize, scioxide_core::SciOxideError> {
    ///     c.provider_count(scioxide_core::EvaluationRole::Objective)
    /// }
    /// ```
    #[inline]
    pub fn provider_count(&self, role: EvaluationRole) -> Result<usize, SciOxideError> {
        self.state.counts.ensure_exact()?;
        Counter::Provider(role).checked_merge(
            self.state.counts.own.provider(role),
            self.state.counts.nested.provider(role),
        )
    }

    /// Returns the metadata of the last accepted state, if any.
    pub const fn accepted_state(&self) -> Option<AcceptedStateMetadata> {
        *self.state.accepted
    }

    /// Returns why the solve should stop, if cancellation was requested or the deadline passed.
    ///
    /// Cancellation is checked before the deadline.
    ///
    /// # Errors
    ///
    /// - The clock's error, if reading the clock fails.
    #[inline]
    pub fn stop_reason(&self) -> Result<Option<ExecutionStopReason>, SciOxideError> {
        self.execution.stop_reason()
    }

    /// Counts one iteration of this solve, if its iteration limit allows it.
    #[inline]
    pub fn record_iteration(&mut self) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::count(Counter::Iteration))
    }
    /// Counts `amount` iterations of this solve, all or none.
    ///
    /// Enclosing solves record them as nested iterations, which do not count
    /// toward their own iteration limits. A denial or error changes no count.
    /// This method does not check cancellation or the deadline.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    ///
    /// ```
    /// fn rows(
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    ///     n: usize,
    /// ) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.record_iterations(n)?.is_granted())
    /// }
    /// ```
    #[inline]
    pub fn record_iterations(&mut self, amount: usize) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::Count {
            counter: Counter::Iteration,
            scope: CountScope::Own,
            amount,
        })
    }
    /// Counts `amount` iterations of a nested computation without starting a nested solve.
    ///
    /// The iterations are recorded as nested counts and do not count toward this
    /// solve's iteration limit. The nested computation enforces its own limits.
    ///
    /// # Errors
    ///
    /// - The same errors as [`Self::record_iterations`].
    ///
    /// ```
    /// # use scioxide_core::*;
    /// # fn main() -> Result<(), SciOxideError> {
    /// let mut state = SolveState::new();
    /// let mut budgets = SolveBudgets::unbounded();
    /// let algorithm = algorithm_id!("example", "nested", 1);
    /// let id = SolveIdentity::new(ProblemToken::new(1), algorithm, 0, 0);
    /// let mut context = state.begin_solve(ExecutionContext::sequential(), id, &mut budgets)?;
    /// context.record_nested_iterations(3)?.require()?;
    /// assert_eq!(context.counts().nested().iterations(), 3);
    /// # Ok(()) }
    /// ```
    #[inline]
    pub fn record_nested_iterations(&mut self, amount: usize) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::Count {
            counter: Counter::Iteration,
            scope: CountScope::Nested,
            amount,
        })
    }

    /// Checks whether [`Self::record_iterations`] would grant `amount` iterations, without counting them.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    ///
    /// ```
    /// fn can_iterate(c: &scioxide_core::SolveContext<'_, '_>) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.can_record_iterations(8)?.is_granted())
    /// }
    /// ```
    pub fn can_record_iterations(&self, amount: usize) -> Result<Admission, SciOxideError> {
        self.check(&Charge::Count {
            counter: Counter::Iteration,
            scope: CountScope::Own,
            amount,
        })
    }
    /// Counts one callback invocation of `role`, if the evaluation limit allows it.
    ///
    /// Call it before invoking the callback; a failed invocation still counts.
    #[inline]
    pub fn record_provider(&mut self, role: EvaluationRole) -> Result<Admission, SciOxideError> {
        self.record_providers(role, 1)
    }
    /// Counts `amount` callback invocations of `role`, all or none.
    ///
    /// The effect equals `amount` calls of [`Self::record_provider`]. The
    /// invocations count toward the evaluation limits of this solve and every
    /// enclosing solve. A denial or error changes no count. This method does not
    /// check cancellation or the deadline.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    ///
    /// ```
    /// fn samples(
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    ///     n: usize,
    /// ) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.record_providers(scioxide_core::EvaluationRole::Function, n)?.is_granted())
    /// }
    /// ```
    #[inline]
    pub fn record_providers(
        &mut self,
        role: EvaluationRole,
        amount: usize,
    ) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::Count {
            counter: Counter::Provider(role),
            scope: CountScope::Own,
            amount,
        })
    }
    /// Counts `amount` callback invocations of a nested computation without starting a nested solve.
    ///
    /// The invocations are recorded as nested counts and count toward the
    /// evaluation limits of this solve and every enclosing solve. Call it before
    /// invoking the callbacks. The nested computation enforces its own limits.
    ///
    /// # Errors
    ///
    /// - The same errors as [`Self::record_providers`].
    /// ```
    /// fn inner(c: &mut scioxide_core::SolveContext<'_, '_>) -> Result<(), scioxide_core::SciOxideError> {
    ///     c.record_nested_providers(scioxide_core::EvaluationRole::Function, 1)?.require()
    /// }
    /// ```
    #[inline]
    pub fn record_nested_providers(
        &mut self,
        role: EvaluationRole,
        amount: usize,
    ) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::Count {
            counter: Counter::Provider(role),
            scope: CountScope::Nested,
            amount,
        })
    }

    /// Counts one invocation of `role` and then calls `provider`.
    ///
    /// If the evaluation limit denies the invocation, `provider` is not called and
    /// the result is [`Invocation::BudgetExhausted`]. Errors from `provider` are
    /// returned unchanged.
    pub fn invoke_provider<T>(
        &mut self,
        role: EvaluationRole,
        provider: impl FnOnce() -> Result<T, SciOxideError>,
    ) -> Result<Invocation<T>, SciOxideError> {
        match self.admit(&Charge::count(Counter::Provider(role)))? {
            Admission::Granted(()) => provider().map(Invocation::Evaluated),
            Admission::Denied(report) => Ok(Invocation::BudgetExhausted(report)),
        }
    }
    /// Counts one proposal of this solve, if its proposal limit allows it.
    #[inline]
    pub fn record_proposal(&mut self) -> Result<Admission, SciOxideError> {
        self.record_proposals(1)
    }

    /// Counts `amount` proposals of this solve, all or none.
    ///
    /// Enclosing solves record them as nested proposals, which do not count
    /// toward their own proposal limits. A denial or error changes no count.
    /// This method does not check cancellation or the deadline.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    ///
    /// ```
    /// fn proposals(
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    ///     n: usize,
    /// ) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.record_proposals(n)?.is_granted())
    /// }
    /// ```
    #[inline(always)]
    pub fn record_proposals(&mut self, amount: usize) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::Count {
            counter: Counter::Proposal,
            scope: CountScope::Own,
            amount,
        })
    }

    /// Checks whether [`Self::record_proposals`] would grant `amount` proposals, without counting them.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    /// ```
    /// fn can_propose(c: &scioxide_core::SolveContext<'_, '_>) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.can_record_proposals(8)?.is_granted())
    /// }
    /// ```
    pub fn can_record_proposals(&self, amount: usize) -> Result<Admission, SciOxideError> {
        self.check(&Charge::Count {
            counter: Counter::Proposal,
            scope: CountScope::Own,
            amount,
        })
    }

    /// Counts an accepted proposal and advances the accepted-state revision.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the context has no active epoch.
    /// - `SizeOverflow` if the revision or a count would exceed `usize::MAX`.
    #[inline(always)]
    pub fn commit_accepted(&mut self, trust: PartialResultTrust) -> Result<(), SciOxideError> {
        let accepted = self.state.accepted.ok_or_else(|| {
            SciOxideError::invalid_input(
                crate::argument!("self"),
                InvalidInputReason::Malformed,
                "solve context has no active epoch",
            )
        })?;
        let revision = accepted.revision.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(crate::quantity!("accepted_state_revision"))
        })?;
        let _ = self.admit(&Charge::count(Counter::Accepted))?;
        *self.state.accepted = Some(AcceptedStateMetadata {
            epoch: accepted.epoch,
            revision,
            trust,
        });
        Ok(())
    }

    /// Updates the trust level of the current accepted state without accepting a new proposal.
    ///
    /// This also applies to the initial state at revision zero. Certification
    /// and diagnostics do not count as new iterates.
    ///
    /// # Errors
    ///
    /// - `AccountingInvariant` if the epoch's counts are incomplete.
    /// - `InvalidInput` if the context has no active epoch.
    #[inline]
    pub fn assess_accepted(&mut self, trust: PartialResultTrust) -> Result<(), SciOxideError> {
        self.state.counts.ensure_exact()?;
        let accepted = self.state.accepted.as_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                crate::argument!("self"),
                InvalidInputReason::Malformed,
                "solve context has no active epoch",
            )
        })?;
        accepted.trust = trust;
        Ok(())
    }

    /// Counts a rejected proposal; the accepted state does not change.
    pub fn record_rejected(&mut self) -> Result<(), SciOxideError> {
        self.admit(&Charge::count(Counter::Rejected)).map(|_| ())
    }

    /// Counts a cache hit; it does not count toward the evaluation limit.
    pub fn record_cache_hit(&mut self) -> Result<(), SciOxideError> {
        self.admit(&Charge::count(Counter::CacheHit)).map(|_| ())
    }

    /// Sets a deadline `duration` from now, keeping any earlier deadline.
    ///
    /// On error the context is unchanged.
    ///
    /// # Errors
    ///
    /// - `CapabilityUnavailable` if no clock is bound.
    /// - `CapabilityFailure` if the clock fails or the deadline overflows.
    pub fn limit_time(&mut self, duration: core::time::Duration) -> Result<(), SciOxideError> {
        self.execution = self.execution.with_time_limit(duration)?;
        Ok(())
    }

    /// Counts one unit of work as an iteration and a proposal, all or none.
    ///
    /// `evaluations` is checked against the evaluation limit but not counted;
    /// count the actual invocations afterwards with [`Self::record_providers`].
    /// `subdivisions` is counted immediately. Limits are checked in the order
    /// evaluation, iteration, proposal, subdivision, and a denial or error
    /// changes no count. The caller checks cancellation and the deadline.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    ///
    /// ```
    /// fn begin(
    ///     c: &mut scioxide_core::SolveContext<'_, '_>,
    ///     samples: usize,
    /// ) -> Result<bool, scioxide_core::SciOxideError> {
    ///     Ok(c.record_work_unit(samples, 0)?.is_granted())
    /// }
    /// ```
    #[inline(always)]
    pub fn record_work_unit(
        &mut self,
        evaluations: usize,
        subdivisions: usize,
    ) -> Result<Admission, SciOxideError> {
        if self.parent.is_some() {
            return self.record_nested_work_unit(evaluations, subdivisions);
        }
        self.state.counts.ensure_exact()?;
        if evaluations != 0
            && let Admission::Denied(report) = prepare_resource(
                self.budgets,
                BudgetKind::Evaluation,
                evaluations,
                self.headroom_limit,
            )?
        {
            return Ok(Admission::Denied(report));
        }
        let iteration_consumed =
            match prepare_resource(self.budgets, BudgetKind::Iteration, 1, self.headroom_limit)? {
                Admission::Granted(next) => next,
                Admission::Denied(report) => return Ok(Admission::Denied(report)),
            };
        let proposal_consumed =
            match prepare_resource(self.budgets, BudgetKind::Proposal, 1, self.headroom_limit)? {
                Admission::Granted(next) => next,
                Admission::Denied(report) => return Ok(Admission::Denied(report)),
            };
        let subdivision_consumed = if subdivisions != 0 {
            match prepare_resource(
                self.budgets,
                BudgetKind::Subdivision,
                subdivisions,
                self.headroom_limit,
            )? {
                Admission::Granted(next) => Some(next),
                Admission::Denied(report) => return Ok(Admission::Denied(report)),
            }
        } else {
            None
        };
        let iteration = prepare_counter(
            self.state.counts,
            Counter::Iteration,
            CountScope::Own,
            1,
            self.headroom_limit,
        )?;
        let proposal = prepare_counter(
            self.state.counts,
            Counter::Proposal,
            CountScope::Own,
            1,
            self.headroom_limit,
        )?;
        self.budgets.consumed[BudgetKind::Iteration.index()] = iteration_consumed;
        self.budgets.consumed[BudgetKind::Proposal.index()] = proposal_consumed;
        if let Some(next) = subdivision_consumed {
            self.budgets.consumed[BudgetKind::Subdivision.index()] = next;
        }
        self.state.counts.own.iterations = iteration.counter;
        self.state.counts.own.proposals = proposal.counter;
        Ok(Admission::Granted(()))
    }

    fn record_nested_work_unit(
        &mut self,
        evaluations: usize,
        subdivisions: usize,
    ) -> Result<Admission, SciOxideError> {
        for (kind, amount) in [
            (BudgetKind::Evaluation, evaluations),
            (BudgetKind::Iteration, 1),
            (BudgetKind::Proposal, 1),
            (BudgetKind::Subdivision, subdivisions),
        ] {
            if amount != 0
                && let Admission::Denied(report) = self.can_consume(kind, amount)?
            {
                return Ok(Admission::Denied(report));
            }
        }
        if let Admission::Denied(report) = self.can_record_iterations(1)? {
            return Ok(Admission::Denied(report));
        }
        if let Admission::Denied(report) = self.can_record_proposals(1)? {
            return Ok(Admission::Denied(report));
        }
        self.record_iteration()?.require()?;
        self.record_proposal()?.require()?;
        if subdivisions != 0 {
            self.try_consume(BudgetKind::Subdivision, subdivisions)?
                .require()?;
        }
        Ok(Admission::Granted(()))
    }

    /// Consumes `amount` units of a non-callback resource, such as search-tree nodes or samples.
    ///
    /// Shared kinds are checked against the limits of every enclosing solve
    /// before anything changes. Local kinds are checked only against this solve.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a count of this solve or an enclosing solve would
    ///   exceed `usize::MAX`.
    #[inline]
    pub fn try_consume(
        &mut self,
        kind: BudgetKind,
        amount: usize,
    ) -> Result<Admission, SciOxideError> {
        self.admit(&Charge::resource(kind, amount))
    }

    /// Checks whether [`Self::try_consume`] would grant `amount` units of `kind`, without consuming them.
    ///
    /// Use it before work that cannot stop halfway, then record the actual
    /// work with the `record_*` methods. Counter overflow of those later calls
    /// is not checked here.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the consumption would exceed `usize::MAX`.
    #[inline]
    pub fn can_consume(&self, kind: BudgetKind, amount: usize) -> Result<Admission, SciOxideError> {
        self.check(&Charge::resource(kind, amount))
    }

    /// Returns the solve metadata to store in a checkpoint.
    ///
    /// `state_token` identifies the solver's own checkpointed numerical state.
    ///
    /// # Errors
    ///
    /// - `AccountingInvariant` if the epoch's counts are incomplete.
    /// - `InvalidInput` if no state has been accepted.
    pub fn checkpoint(&self, state_token: u128) -> Result<SolveCheckpointMetadata, SciOxideError> {
        self.state.counts.ensure_exact()?;
        let identity = *self.state.identity;
        let accepted = self.state.accepted.ok_or_else(|| {
            SciOxideError::invalid_input(
                crate::argument!("self"),
                InvalidInputReason::Malformed,
                "solve context has no accepted-state metadata",
            )
        })?;
        Ok(SolveCheckpointMetadata {
            schema_version: SolveCheckpointMetadata::SCHEMA_VERSION,
            identity,
            accepted,
            counts: *self.state.counts,
            budgets: *self.budgets,
            state_token,
        })
    }

    /// Returns the limit and consumption of `kind` for this solve.
    pub const fn budget_report(&self, kind: BudgetKind) -> BudgetReport {
        self.budgets.report(kind)
    }

    /// Returns the report of the tightest limit on `kind` that applies to this solve.
    ///
    /// Shared kinds include the limits of enclosing solves; on ties the nearest
    /// solve wins. Local kinds report only this solve. The reported consumption
    /// is that of the solve whose limit is returned.
    pub fn effective_budget_report(&self, kind: BudgetKind) -> BudgetReport {
        self.limiting_report(kind)
    }
}

// Ordinary admission prepares only the scalar values changed by a sparse charge.
// Detached reconciliation separately visits the complete borrowed child record.
macro_rules! counter_fields {
    ($($variant:ident => $field:ident, $context:literal, $budget:expr);+ $(;)?) => {
        #[derive(Debug, Clone, Copy)]
        enum Counter { $($variant,)+ Provider(EvaluationRole) }
        impl Counter {
            fn all() -> impl Iterator<Item = Self> {
                [$(Self::$variant,)+].into_iter()
                    .chain(EvaluationRole::ALL.iter().copied().map(Self::Provider))
            }
            #[inline]
            fn read(self, counts: &EvaluationCounts) -> usize {
                match self { $(Self::$variant => counts.$field,)+ Self::Provider(role) => counts.providers[role.index()] }
            }
            #[inline]
            fn write(self, counts: &mut EvaluationCounts, value: usize) {
                match self { $(Self::$variant => counts.$field = value,)+ Self::Provider(role) => counts.providers[role.index()] = value }
            }
            const fn context(self) -> crate::Quantity {
                match self { $(Self::$variant => crate::quantity!($context),)+ Self::Provider(role) => role.counter_context() }
            }
            const fn budget(self) -> Option<BudgetKind> {
                match self { $(Self::$variant => $budget,)+ Self::Provider(_) => Some(BudgetKind::Evaluation) }
            }
            #[inline]
            fn checked_merge(self, old: usize, delta: usize, ) -> Result<usize, SciOxideError> {
                old.checked_add(delta).ok_or_else(|| SciOxideError::size_overflow( self.context()))
            }
        }
    };
}
counter_fields! {
    Iteration => iterations, "solve.iterations", Some(BudgetKind::Iteration);
    Proposal => proposals, "solve.proposals", Some(BudgetKind::Proposal);
    Accepted => accepted, "solve.accepted_proposals", None;
    Rejected => rejected, "solve.rejected_proposals", None;
    CacheHit => cache_hits, "solve.cache_hits", None;
}
#[derive(Debug, Clone, Copy)]
enum CountScope {
    Own,
    Nested,
}
impl CountScope {
    fn source(self, counts: &SolveCounts) -> &EvaluationCounts {
        match self {
            Self::Own => &counts.own,
            Self::Nested => &counts.nested,
        }
    }
    fn target(self, counts: &mut SolveCounts) -> &mut EvaluationCounts {
        match self {
            Self::Own => &mut counts.own,
            Self::Nested => &mut counts.nested,
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Charge {
    Resource {
        kind: BudgetKind,
        amount: usize,
    },
    Count {
        counter: Counter,
        scope: CountScope,
        amount: usize,
    },
}
impl Charge {
    const fn resource(kind: BudgetKind, amount: usize) -> Self {
        Self::Resource { kind, amount }
    }
    const fn count(counter: Counter) -> Self {
        Self::Count {
            counter,
            scope: CountScope::Own,
            amount: 1,
        }
    }
    fn nested(self) -> Self {
        match self {
            Self::Resource { kind, amount } => Self::Resource {
                kind,
                amount: if kind.nesting() == NestingPolicy::Shared {
                    amount
                } else {
                    0
                },
            },
            Self::Count {
                counter, amount, ..
            } => Self::Count {
                counter,
                scope: CountScope::Nested,
                amount,
            },
        }
    }
    #[inline(always)]
    fn resource_charge(self) -> Option<(BudgetKind, usize)> {
        match self {
            Self::Resource { kind, amount } => Some((kind, amount)),
            Self::Count {
                counter,
                scope,
                amount,
            } => counter
                .budget()
                .filter(|kind| {
                    matches!(scope, CountScope::Own) || kind.nesting() == NestingPolicy::Shared
                })
                .map(|kind| (kind, amount)),
        }
    }
}

// No record-sized variant: ancestor recursion retains just three scalar values.
#[derive(Debug, Clone, Copy)]
struct PreparedCharge {
    counter: usize,
    consumed: usize,
    provider_total: usize,
}

/// Checks counter overflow, the callback total and headroom for one counter
/// update, separately from resource consumption.
#[inline(always)]
fn prepare_counter(
    counts: &SolveCounts,
    counter: Counter,
    scope: CountScope,
    amount: usize,
    headroom: usize,
) -> Result<PreparedCharge, SciOxideError> {
    let mut next = PreparedCharge {
        counter: 0,
        consumed: 0,
        provider_total: 0,
    };
    next.counter = counter.checked_merge(counter.read(scope.source(counts)), amount)?;
    let other = match scope {
        CountScope::Own => &counts.nested,
        CountScope::Nested => &counts.own,
    };
    let combined = counter.checked_merge(next.counter, counter.read(other))?;
    next.provider_total = counts.provider_total.ok_or_else(|| {
        SciOxideError::size_overflow(crate::quantity!("total_provider_evaluations"))
    })?;
    if matches!(counter, Counter::Provider(_)) {
        next.provider_total = next.provider_total.checked_add(amount).ok_or_else(|| {
            SciOxideError::size_overflow(crate::quantity!("total_provider_evaluations"))
        })?;
    }
    if combined.max(next.provider_total) > headroom {
        return Err(SciOxideError::size_overflow(counter.context()));
    }
    Ok(next)
}

fn validate_record(budgets: SolveBudgets, counts: SolveCounts) -> Result<(), SciOxideError> {
    let total = counts.total()?;
    let provider_total = total.total_providers()?;
    if counts.provider_total != Some(provider_total) {
        return Err(accounting_invariant(
            "provider aggregate disagrees with recorded counters",
        ));
    }
    if provider_total > budgets.consumed[BudgetKind::Evaluation.index()] {
        return Err(accounting_invariant(
            "provider counts exceed recorded evaluation consumption",
        ));
    }
    if counts.own.iterations() > budgets.consumed[BudgetKind::Iteration.index()]
        || counts.own.proposals() > budgets.consumed[BudgetKind::Proposal.index()]
    {
        return Err(accounting_invariant(
            "own work exceeds recorded resource consumption",
        ));
    }
    for &kind in BudgetKind::ALL {
        if let BudgetLimit::Finite(limit) = budgets.limits[kind.index()]
            && budgets.consumed[kind.index()] > limit
        {
            return Err(accounting_invariant(
                "recorded consumption exceeds the local allowance",
            ));
        }
    }
    Ok(())
}

fn largest_quantity(budgets: &SolveBudgets, counts: SolveCounts) -> Result<usize, SciOxideError> {
    let total = counts.total()?;
    let largest = Counter::all()
        .map(|counter| counter.read(&total))
        .chain(
            BudgetKind::ALL
                .iter()
                .copied()
                .filter(|kind| kind.nesting() == NestingPolicy::Shared)
                .map(|kind| budgets.consumed[kind.index()]),
        )
        .max()
        .unwrap_or(0);
    Ok(largest.max(total.total_providers()?))
}

trait SolveLedger: fmt::Debug {
    fn budgets_ref(&self) -> &SolveBudgets;
    fn counts_ref(&self) -> &SolveCounts {
        const EMPTY: SolveCounts = SolveCounts::new();
        &EMPTY
    }
    fn headroom_limit(&self) -> usize {
        usize::MAX
    }
    fn parent(&self) -> Option<&dyn SolveLedger> {
        None
    }
    fn parent_mut(&mut self) -> Option<&mut (dyn SolveLedger + '_)> {
        None
    }
    fn commit(&mut self, charge: &Charge, next: PreparedCharge);
    fn commit_merge(&mut self, budgets: SolveBudgets, counts: SolveCounts);
    fn invalidate(&mut self) {
        if let Some(parent) = self.parent_mut() {
            parent.invalidate();
        }
    }
    // Keep the caller's charge kind visible: fixed-stencil samples then check
    // the known counter and allowance without a run-time Counter dispatch.
    // Issue #997's isolated admission/charge experiments require both hints.
    #[inline(always)]
    fn charge(&self, charge: &Charge) -> Result<Admission<PreparedCharge>, SciOxideError> {
        let counts = self.counts_ref();
        counts.ensure_exact()?;
        let mut next = PreparedCharge {
            counter: 0,
            consumed: 0,
            provider_total: 0,
        };
        if let Charge::Count {
            counter,
            scope,
            amount,
        } = *charge
        {
            next = prepare_counter(counts, counter, scope, amount, self.headroom_limit())?;
        }
        if let Some((kind, amount)) = charge.resource_charge() {
            match prepare_resource(self.budgets_ref(), kind, amount, self.headroom_limit())? {
                Admission::Granted(consumed) => next.consumed = consumed,
                Admission::Denied(report) => return Ok(Admission::Denied(report)),
            }
        }
        Ok(Admission::Granted(next))
    }
    #[inline(always)]
    fn check(&self, charge: &Charge) -> Result<Admission, SciOxideError> {
        match self.charge(charge)? {
            Admission::Denied(report) => Ok(Admission::Denied(report)),
            Admission::Granted(_) => match self.parent() {
                Some(parent) => parent.check(&charge.nested()),
                None => Ok(Admission::Granted(())),
            },
        }
    }
    // Admission must expose `charge` to the recording call's constant counter.
    // Ancestor admission remains checked before any local commit.
    #[inline(always)]
    fn admit(&mut self, charge: &Charge) -> Result<Admission, SciOxideError> {
        let next = match self.charge(charge)? {
            Admission::Denied(report) => return Ok(Admission::Denied(report)),
            Admission::Granted(next) => next,
        };
        // Descend through every ancestor before any commit, then commit while
        // unwinding the checked chain. No reservation vector is allocated.
        if let Some(parent) = self.parent_mut()
            && let Admission::Denied(report) = parent.admit(&charge.nested())?
        {
            return Ok(Admission::Denied(report));
        }
        self.commit(charge, next);
        Ok(Admission::Granted(()))
    }
    fn merge(
        &mut self,
        resources: &[usize; BUDGET_KIND_COUNT],
        child: &EvaluationCounts,
    ) -> Result<Admission, SciOxideError> {
        let mut counts = *self.counts_ref();
        counts.ensure_exact()?;
        counts.nested = counts.nested.merge(*child)?;
        let mut budgets = *self.budgets_ref();
        for &kind in BudgetKind::ALL {
            if kind.nesting() == NestingPolicy::Shared {
                match prepare_resource(
                    &budgets,
                    kind,
                    resources[kind.index()],
                    self.headroom_limit(),
                )? {
                    Admission::Granted(consumed) => budgets.consumed[kind.index()] = consumed,
                    Admission::Denied(report) => return Ok(Admission::Denied(report)),
                }
            }
        }
        counts.provider_total = Some(counts.total()?.total_providers()?);
        if largest_quantity(&budgets, counts)? > self.headroom_limit() {
            return Err(SciOxideError::size_overflow(crate::quantity!(
                "solve_counter_headroom"
            )));
        }
        if let Some(parent) = self.parent_mut()
            && let Admission::Denied(report) = parent.merge(resources, child)?
        {
            return Ok(Admission::Denied(report));
        }
        self.commit_merge(budgets, counts);
        Ok(Admission::Granted(()))
    }
    fn headroom(&self) -> Result<usize, SciOxideError> {
        validate_record(*self.budgets_ref(), *self.counts_ref())?;
        let largest = largest_quantity(self.budgets_ref(), *self.counts_ref())?;
        let remaining = self
            .headroom_limit()
            .checked_sub(largest)
            .ok_or_else(|| accounting_invariant("recorded work exceeds reserved headroom"))?;
        match self.parent() {
            Some(parent) => Ok(remaining.min(parent.headroom()?)),
            None => Ok(remaining),
        }
    }
    fn remaining(&self, kind: BudgetKind) -> Option<usize> {
        self.limiting_report(kind).remaining()
    }
    fn limiting_report(&self, kind: BudgetKind) -> BudgetReport {
        let own = self.budgets_ref().report(kind);
        if kind.nesting() == NestingPolicy::ScopeLocal {
            return own;
        }
        let Some(parent) = self.parent() else {
            return own;
        };
        let inherited = parent.limiting_report(kind);
        match (own.remaining(), inherited.remaining()) {
            (None, Some(_)) => inherited,
            (Some(a), Some(b)) if b < a => inherited,
            _ => own,
        }
    }
}

// Keep fixed counter roles visible through the complete admission/commit path.
// Scalar solvers charge several roles per iteration; outlining these tiny
// operations otherwise repeats dynamic dispatch and materializes Charge values.
#[inline(always)]
fn prepare_resource(
    budgets: &SolveBudgets,
    kind: BudgetKind,
    amount: usize,
    headroom: usize,
) -> Result<Admission<usize>, SciOxideError> {
    let index = kind.index();
    let next = budgets.consumed[index]
        .checked_add(amount)
        .ok_or_else(|| SciOxideError::size_overflow(kind.counter_context()))?;
    if kind.nesting() == NestingPolicy::Shared && next > headroom {
        return Err(SciOxideError::size_overflow(kind.counter_context()));
    }
    if let BudgetLimit::Finite(limit) = budgets.limits[index]
        && next > limit
    {
        return Ok(Admission::Denied(budgets.report(kind)));
    }
    Ok(Admission::Granted(next))
}

impl SolveLedger for SolveBudgets {
    fn budgets_ref(&self) -> &SolveBudgets {
        self
    }
    #[inline(always)]
    fn commit(&mut self, charge: &Charge, next: PreparedCharge) {
        if let Some((kind, _)) = charge.resource_charge() {
            self.consumed[kind.index()] = next.consumed;
        }
    }
    fn commit_merge(&mut self, budgets: SolveBudgets, _: SolveCounts) {
        *self = budgets;
    }
}
impl SolveLedger for SolveContext<'_, '_> {
    fn budgets_ref(&self) -> &SolveBudgets {
        self.budgets
    }
    fn counts_ref(&self) -> &SolveCounts {
        self.state.counts
    }
    fn headroom_limit(&self) -> usize {
        self.headroom_limit
    }
    fn parent(&self) -> Option<&dyn SolveLedger> {
        self.parent.as_deref()
    }
    fn parent_mut(&mut self) -> Option<&mut (dyn SolveLedger + '_)> {
        match &mut self.parent {
            Some(parent) => Some(&mut **parent),
            None => None,
        }
    }
    #[inline(always)]
    fn commit(&mut self, charge: &Charge, next: PreparedCharge) {
        self.budgets.commit(charge, next);
        if let Charge::Count { counter, scope, .. } = *charge {
            counter.write(scope.target(self.state.counts), next.counter);
            if matches!(counter, Counter::Provider(_)) {
                self.state.counts.provider_total = Some(next.provider_total);
            }
        }
    }
    fn commit_merge(&mut self, budgets: SolveBudgets, counts: SolveCounts) {
        *self.budgets = budgets;
        *self.state.counts = counts;
    }
    fn invalidate(&mut self) {
        self.state.counts.integrity = AccountingIntegrity::Invalid;
        if let Some(parent) = self.parent_mut() {
            parent.invalidate();
        }
    }
}

fn accounting_invariant(detail: &'static str) -> SciOxideError {
    SciOxideError::accounting_invariant(detail, None)
}

#[inline(always)]
fn finish_scope<T>(
    context: &mut SolveContext<'_, '_>,
    outcome: Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    if context.state.counts.integrity == AccountingIntegrity::Invalid {
        context.invalidate();
        return match outcome {
            Err(error) if error.kind() == crate::SciOxideErrorKind::AccountingInvariant => {
                Err(error)
            }
            _ => Err(accounting_invariant(
                "child epoch has incomplete accounting",
            )),
        };
    }
    outcome
}

/// Fixed-size storage for the state and counts of one child solve in a partitioned batch.
///
/// [`with_partitioned_solves`] resets the storage and gives each child a
/// share of the parent's remaining limits. A default value is empty storage,
/// not permission to run. Outside a batch only the counts can be read.
///
/// ```
/// use scioxide_core::backend::PartitionedSolve;
///
/// let storage = PartitionedSolve::default();
/// assert_eq!(storage.counts().total()?.total_providers()?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Default)]
pub struct PartitionedSolve {
    state: SolveState,
    allowance: SolveBudgets,
    headroom: usize,
}
impl PartitionedSolve {
    /// Returns the work counted by this child in the last batch.
    pub const fn counts(&self) -> SolveCounts {
        self.state.counts
    }
}

/// Permission to run one child solve of a partitioned batch at most once.
///
/// Permits come only from a [`PartitionSolveBatch`]. A permit borrows its
/// child's storage, so its share of the limits cannot move to another child.
/// Running consumes the permit:
/// ```compile_fail
/// use scioxide_core::{ExecutionContext, SolveIdentity};
/// use scioxide_core::backend::PartitionSolvePermit;
/// fn twice(permit: PartitionSolvePermit<'_>, cx: ExecutionContext<'_>, id: SolveIdentity) {
///     permit.run(cx, id, |_| Ok(())).unwrap();
///     permit.run(cx, id, |_| Ok(())).unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct PartitionSolvePermit<'a> {
    storage: &'a mut PartitionedSolve,
    execution: ExecutionContext<'a>,
}
impl PartitionSolvePermit<'_> {
    /// Returns the work counted by this child so far.
    pub const fn counts(&self) -> SolveCounts {
        self.storage.counts()
    }
    /// Starts the child's solve epoch and calls `run` with its context.
    ///
    /// The child inherits the batch's cancellation, deadline and processor
    /// features; `execution` can only narrow its parallelism and deadline.
    ///
    /// # Errors
    ///
    /// - An error if `execution` has a deadline from a different clock binding
    ///   than the batch, before `run` is called.
    /// - Any error returned by `run`, unchanged.
    pub fn run<T>(
        self,
        execution: ExecutionContext<'_>,
        identity: SolveIdentity,
        run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let execution = execution.inherit_solve_capabilities(self.execution)?;
        let child = self.storage;
        let mut context = child
            .state
            .begin_solve(execution, identity, &mut child.allowance)?;
        context.headroom_limit = child.headroom;
        let outcome = run(&mut context);
        finish_scope(&mut context, outcome)
    }
}

/// The prepared children of a partitioned batch; iterating yields one permit per child.
///
/// Created only by [`with_partitioned_solves`]; see its example. A batch can
/// be split into parts for different workers. Iteration consumes the batch, so
/// no child yields a second permit:
/// ```compile_fail
/// use scioxide_core::backend::PartitionSolveBatch;
/// fn twice(batch: PartitionSolveBatch<'_>) {
///     let first = batch.into_iter().next();
///     let second = batch.into_iter().next();
/// }
/// ```
#[derive(Debug)]
pub struct PartitionSolveBatch<'a> {
    children: &'a mut [PartitionedSolve],
    execution: ExecutionContext<'a>,
}
impl<'a> PartitionSolveBatch<'a> {
    /// Returns the number of children.
    pub const fn len(&self) -> usize {
        self.children.len()
    }
    /// Returns whether the batch has no children.
    pub const fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
    /// Splits the batch into the children before `index` and those from `index` on.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index` exceeds the batch length.
    pub fn split_at(self, index: usize) -> Result<(Self, Self), SciOxideError> {
        if index > self.children.len() {
            return Err(SciOxideError::out_of_bounds(
                crate::quantity!("children"),
                index,
                self.children.len(),
            ));
        }
        let (left, right) = self.children.split_at_mut(index);
        Ok((
            Self {
                children: left,
                execution: self.execution,
            },
            Self {
                children: right,
                execution: self.execution,
            },
        ))
    }
    /// Splits the batch into consecutive chunks of `size` children; the last may be shorter.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `size` is zero.
    pub fn into_chunks(self, size: usize) -> Result<impl Iterator<Item = Self>, SciOxideError> {
        if size == 0 {
            return Err(SciOxideError::invalid_input(
                crate::argument!("size"),
                InvalidInputReason::OutOfRange,
                "chunk size must be positive",
            ));
        }
        let execution = self.execution;
        Ok(self.children.chunks_mut(size).map(move |children| Self {
            children,
            execution,
        }))
    }
}
/// Iterator over the permits of a [`PartitionSolveBatch`], one per child.
///
/// Created by `into_iter` on the batch. See [`with_partitioned_solves`] for an
/// example.
#[derive(Debug)]
pub struct PartitionSolvePermits<'a> {
    children: core::slice::IterMut<'a, PartitionedSolve>,
    execution: ExecutionContext<'a>,
}
impl<'a> Iterator for PartitionSolvePermits<'a> {
    type Item = PartitionSolvePermit<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.children.next().map(|storage| PartitionSolvePermit {
            storage,
            execution: self.execution,
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.children.size_hint()
    }
}
impl ExactSizeIterator for PartitionSolvePermits<'_> {}
impl<'a> IntoIterator for PartitionSolveBatch<'a> {
    type Item = PartitionSolvePermit<'a>;
    type IntoIter = PartitionSolvePermits<'a>;
    fn into_iter(self) -> Self::IntoIter {
        PartitionSolvePermits {
            children: self.children.iter_mut(),
            execution: self.execution,
        }
    }
}

/// Runs independent child solves, for example on different workers, and merges their counts into `parent`.
///
/// Before `run` is called, the parent's remaining shared limits (evaluations,
/// subdivisions, samples, nodes) are divided among the children by index;
/// a share a child does not use is not passed to its siblings. `allowance`
/// sets each child's own iteration and proposal limits. The storage in
/// `children` is reset first and holds each child's counts afterwards.
///
/// Every valid child's counts are merged into `parent`, in index order, even
/// when `run` or a child fails. If a child's counts are inconsistent, the
/// parent's epoch is marked invalid: its totals, checkpoints and further
/// counting fail until a new epoch starts. Preparation and merging take O(n)
/// time for n children. Callbacks must return errors rather than panic for
/// their counts to be merged.
///
/// Run `cargo run -p scioxide-core --example workspace_operations` for a complete caller-owned storage example.
///
/// # Errors
///
/// - `InvalidInput` if `children` is empty.
/// - `AccountingInvariant` if the parent's or a child's counts are inconsistent;
///   this takes precedence over an error from `run`.
/// - `SizeOverflow` if a merged count exceeds `usize::MAX`.
/// - Any error returned by `run`, unchanged.
pub fn with_partitioned_solves<T>(
    parent: &mut SolveContext<'_, '_>,
    children: &mut [PartitionedSolve],
    allowance: SolveBudgets,
    run: impl FnOnce(ExecutionContext<'_>, PartitionSolveBatch<'_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    if children.is_empty() {
        return Err(SciOxideError::invalid_input(
            crate::argument!("children"),
            InvalidInputReason::Empty,
            "empty child batch",
        ));
    }
    let Ok(headroom) = parent.headroom() else {
        parent.invalidate();
        return Err(accounting_invariant(
            "invalid accounting before partition reservation",
        ));
    };
    let count = children.len();
    let remaining = core::array::from_fn::<_, BUDGET_KIND_COUNT, _>(|index| {
        let kind = BudgetKind::ALL[index];
        if kind.nesting() == NestingPolicy::Shared {
            parent.remaining(kind)
        } else {
            None
        }
    });
    for (index, child) in children.iter_mut().enumerate() {
        child.state = SolveState {
            next_epoch: child.state.next_epoch,
            ..SolveState::new()
        };
        child.headroom = headroom / count + usize::from(index < headroom % count);
        child.allowance = SolveBudgets {
            consumed: [0; BUDGET_KIND_COUNT],
            ..allowance
        };
        for &kind in BudgetKind::ALL {
            if let Some(remaining) = remaining[kind.index()] {
                let share = remaining / count + usize::from(index < remaining % count);
                child.allowance.limits[kind.index()] =
                    BudgetLimit::Finite(match child.allowance.limits[kind.index()] {
                        BudgetLimit::Finite(local) => local.min(share),
                        BudgetLimit::Unbounded => share,
                    });
            }
        }
    }
    let execution = parent.execution();
    let outcome = run(
        execution,
        PartitionSolveBatch {
            children,
            execution,
        },
    );
    let mut failure = None;
    for (index, child) in children.iter().enumerate() {
        let incomplete = child.state.counts.integrity == AccountingIntegrity::Invalid;
        let recorded = SolveCounts {
            integrity: AccountingIntegrity::Exact,
            ..child.state.counts
        };
        let merged = (|| {
            validate_record(child.allowance, recorded)?;
            let counts = recorded.total()?;
            match parent.merge(&child.allowance.consumed, &counts)? {
                Admission::Granted(()) => Ok(()),
                Admission::Denied(_) => Err(accounting_invariant(
                    "reserved child work was denied during reconciliation",
                )),
            }
        })();
        if failure.is_none() && (merged.is_err() || incomplete) {
            failure = Some(SciOxideError::accounting_invariant(
                "detached child accounting is incomplete",
                Some(index),
            ));
        }
    }
    if let Some(error) = failure {
        parent.invalidate();
        return Err(error);
    }
    outcome
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{CancellationToken, NormSelection, Tolerance};

    #[test]
    fn direct_descendant_charges_preserve_local_caps_and_atomic_ancestor_denial() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::ONE)
            .with_limit(BudgetKind::Evaluation, crate::budget!(2));
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        let mut child_state = SolveState::new();
        SolveScope::from(&mut root)
            .run(
                &mut child_state,
                identity(2),
                SolveBudgets::unbounded(),
                |child| {
                    child.record_nested_iterations(3)?.require()?;
                    child
                        .record_nested_providers(EvaluationRole::Function, 2)?
                        .require()?;
                    let before = child.counts();
                    assert!(
                        !child
                            .record_nested_providers(EvaluationRole::Function, 1)?
                            .is_granted()
                    );
                    assert_eq!(child.counts(), before);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(root.counts().own().iterations(), 0);
        assert_eq!(root.counts().nested().iterations(), 3);
        assert_eq!(root.provider_count(EvaluationRole::Function).unwrap(), 2);
        assert_eq!(root.budget_report(BudgetKind::Iteration).consumed(), 0);
        root.record_iteration().unwrap().require().unwrap();
        assert_eq!(root.budget_report(BudgetKind::Iteration).consumed(), 1);
    }

    #[test]
    fn work_units_admit_all_limits_in_order_and_charge_actual_providers() {
        let kinds = [
            BudgetKind::Evaluation,
            BudgetKind::Iteration,
            BudgetKind::Proposal,
            BudgetKind::Subdivision,
        ];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        for kind in kinds {
            budgets.limits[kind.index()] = BudgetLimit::Finite(0);
        }
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        for kind in kinds {
            let before = *context.budgets;
            let counts = context.counts();
            assert_eq!(
                context.record_work_unit(2, 1).unwrap(),
                Admission::Denied(context.budgets.report(kind)),
            );
            assert_eq!(*context.budgets, before);
            assert_eq!(context.counts(), counts);
            context.budgets.limits[kind.index()] = BudgetLimit::Unbounded;
        }
        context.record_work_unit(2, 1).unwrap().require().unwrap();
        assert_eq!(context.counts().own().iterations(), 1);
        assert_eq!(context.counts().own().proposals(), 1);
        assert_eq!(context.budgets.report(BudgetKind::Evaluation).consumed(), 0);
        assert_eq!(
            context.budgets.report(BudgetKind::Subdivision).consumed(),
            1
        );
        context
            .record_providers(EvaluationRole::Function, 2)
            .unwrap()
            .require()
            .unwrap();
        assert_eq!(context.budgets.report(BudgetKind::Evaluation).consumed(), 2);
    }

    #[test]
    fn work_unit_counter_overflow_is_atomic() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        context.state.counts.nested = EvaluationCounts::new().with_proposals(usize::MAX, 0, 0);
        let before = context.counts();
        assert!(context.record_work_unit(1, 1).is_err());
        assert_eq!(context.counts(), before);
        assert_eq!(context.budgets.report(BudgetKind::Iteration).consumed(), 0);
        assert_eq!(
            context.budgets.report(BudgetKind::Subdivision).consumed(),
            0
        );
    }

    #[test]
    fn nested_work_units_reconcile_shared_resources_without_local_parent_charges() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, crate::budget!(2))
            .with_limit(BudgetKind::Subdivision, Budget::ONE);
        let mut parent = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        let mut child_state = SolveState::new();
        SolveScope::from(&mut parent).run(&mut child_state, identity(2), SolveBudgets::unbounded(), |child| {
            child.record_work_unit(2, 1)?.require()?;
            child.record_providers(EvaluationRole::Function, 2)?.require()?;
            let before = child.counts();
            let denied = child.record_work_unit(1, 0)?;
            assert!(matches!(denied, Admission::Denied(report) if report.kind() == BudgetKind::Evaluation));
            assert_eq!(child.counts(), before);
            Ok(())
        }).unwrap();
        assert_eq!(parent.counts().own().iterations(), 0);
        assert_eq!(parent.counts().nested().iterations(), 1);
        assert_eq!(parent.counts().nested().proposals(), 1);
        assert_eq!(parent.provider_count(EvaluationRole::Function).unwrap(), 2);
        assert_eq!(parent.budgets.report(BudgetKind::Iteration).consumed(), 0);
        assert_eq!(parent.budgets.report(BudgetKind::Subdivision).consumed(), 1);
    }

    #[test]
    fn partitioned_children_conserve_ancestors_on_failure() {
        let mut root_state = SolveState::new();
        let mut root_budget =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(5));
        let mut root = root_state
            .begin_solve(
                ExecutionContext::sequential(),
                identity(10),
                &mut root_budget,
            )
            .unwrap();
        let mut child_state = SolveState::new();
        let mut partitions = core::array::from_fn::<_, 3, _>(|_| PartitionedSolve::default());
        let result: Result<(), SciOxideError> = crate::SolveScope::from(&mut root).run(
            &mut child_state,
            identity(11),
            SolveBudgets::unbounded(),
            |parent| {
                with_partitioned_solves(
                    parent,
                    &mut partitions,
                    SolveBudgets::unbounded(),
                    |execution, children| {
                        for (i, child) in children.into_iter().enumerate() {
                            child.run(execution, identity(i as u128), |ctx| {
                                while ctx.record_provider(EvaluationRole::Objective)?.is_granted() {
                                }
                                ctx.commit_accepted(PartialResultTrust::Admissible)?;
                                Ok(())
                            })?;
                        }
                        Err(SciOxideError::execution_failure(
                            None,
                            crate::ExecutionFailureReason::ProtocolViolation,
                            "after all work",
                        ))
                    },
                )
            },
        );
        assert!(result.is_err());
        assert_eq!(
            root.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Objective),
            5
        );
        assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 5);
        assert_eq!(root.accepted_state().unwrap().revision(), 0);
        assert_eq!(
            partitions
                .iter()
                .map(|p| p.counts().own().provider(EvaluationRole::Objective))
                .collect::<alloc::vec::Vec<_>>(),
            alloc::vec![2, 2, 1]
        );
    }
    #[test]
    fn partitioned_zero_shares_are_not_transferable() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
        let mut parent = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        let mut children = core::array::from_fn::<_, 2, _>(|_| PartitionedSolve::default());
        with_partitioned_solves(
            &mut parent,
            &mut children,
            SolveBudgets::unbounded(),
            |execution, children| {
                for (index, child) in children.into_iter().enumerate() {
                    child.run(execution, identity(2), |ctx| {
                        assert_eq!(
                            ctx.record_provider(EvaluationRole::Objective)?.is_granted(),
                            index == 0
                        );
                        Ok(())
                    })?;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            parent
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Objective),
            1
        );
    }

    fn identity(problem: u128) -> SolveIdentity {
        SolveIdentity::new(
            ProblemToken::new(problem),
            crate::algorithm_id!("test", "solver", 1),
            3,
            7,
        )
    }

    #[test]
    fn root_accounting_preserves_state_on_limit_and_counter_overflow() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::ONE)
            .with_limit(BudgetKind::Iteration, Budget::ONE)
            .with_limit(BudgetKind::Proposal, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        assert!(
            context
                .record_provider(EvaluationRole::Function)
                .unwrap()
                .is_granted()
        );
        assert!(context.record_iteration().unwrap().is_granted());
        assert!(context.record_proposal().unwrap().is_granted());
        let counts = context.counts();
        assert!(
            !context
                .record_provider(EvaluationRole::Function)
                .unwrap()
                .is_granted()
        );
        assert!(!context.record_iteration().unwrap().is_granted());
        assert!(!context.record_proposal().unwrap().is_granted());
        assert_eq!(context.counts(), counts);
        context.state.counts.own.providers[EvaluationRole::Function as usize] = usize::MAX;
        context.state.counts.own.iterations = usize::MAX;
        context.state.counts.own.proposals = usize::MAX;
        assert!(matches!(
            (context.record_provider(EvaluationRole::Function))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert!(matches!(
            (context.record_iteration()).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert!(matches!(
            (context.record_proposal()).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        for kind in [
            BudgetKind::Evaluation,
            BudgetKind::Iteration,
            BudgetKind::Proposal,
        ] {
            assert_eq!(context.budget_report(kind).consumed(), 1);
        }
    }

    #[test]
    fn work_preflight_is_read_only_and_checks_ancestor_limits_and_overflow() {
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(3));
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        assert!(
            context
                .record_provider(EvaluationRole::Function)
                .unwrap()
                .is_granted()
        );
        let mut child_state = SolveState::new();
        crate::SolveScope::from(&mut context)
            .run(
                &mut child_state,
                identity(2),
                SolveBudgets::unbounded(),
                |child| {
                    assert!(child.can_consume(BudgetKind::Evaluation, 2)?.is_granted());
                    assert!(!child.can_consume(BudgetKind::Evaluation, 3)?.is_granted());
                    assert_eq!(child.counts().own().total_providers()?, 0);
                    assert_eq!(child.budget_report(BudgetKind::Evaluation).consumed(), 0);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 1);
        assert!(
            context
                .try_consume(BudgetKind::Node, usize::MAX)
                .unwrap()
                .is_granted()
        );
        assert!(matches!(
            (context.can_consume(BudgetKind::Node, 1))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert_eq!(
            context.budget_report(BudgetKind::Node).consumed(),
            usize::MAX
        );
    }

    #[test]
    fn nested_contexts_charge_admitted_work_once_and_preserve_parent_acceptance() {
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(3));
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        context
            .commit_accepted(PartialResultTrust::Admissible)
            .unwrap();
        let accepted = context.accepted_state();
        let mut child_state = SolveState::new();
        crate::SolveScope::from(&mut context)
            .run(
                &mut child_state,
                identity(2),
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(2)),
                |child| {
                    assert_eq!(
                        child.invoke_provider(EvaluationRole::Objective, || Ok(7))?,
                        Invocation::Evaluated(7)
                    );
                    let mut grandchild_state = SolveState::new();
                    crate::SolveScope::from(&mut *child).run(
                        &mut grandchild_state,
                        identity(3),
                        SolveBudgets::unbounded(),
                        |grandchild| {
                            assert_eq!(grandchild.record_proposal()?, Admission::Granted(()));
                            grandchild.commit_accepted(PartialResultTrust::ResidualChecked)?;
                            assert!(
                                grandchild
                                    .record_provider(EvaluationRole::Gradient)?
                                    .is_granted()
                            );
                            assert!(
                                !grandchild
                                    .record_provider(EvaluationRole::Hessian)?
                                    .is_granted()
                            );
                            assert_eq!(grandchild.counts().own().total_providers()?, 1);
                            assert_eq!(
                                grandchild.budget_report(BudgetKind::Evaluation).consumed(),
                                1
                            );
                            Ok(())
                        },
                    )?;
                    assert_eq!(child.counts().total()?.total_providers()?, 2);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(context.accepted_state(), accepted);
        assert_eq!(
            context.counts().total().unwrap().total_providers().unwrap(),
            2
        );
        assert_eq!(context.counts().own().accepted(), 1);
        assert_eq!(
            context.budget_report(BudgetKind::Evaluation).remaining(),
            Some(1)
        );
        assert!(
            context
                .record_provider(EvaluationRole::Objective)
                .unwrap()
                .is_granted()
        );
        assert!(
            !context
                .record_provider(EvaluationRole::Objective)
                .unwrap()
                .is_granted()
        );
    }

    #[test]
    fn child_provider_errors_and_raw_allowances_charge_all_ancestors() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        let mut child_state = SolveState::new();
        let failure = SciOxideError::invalid_input(
            crate::argument!("injected_provider"),
            InvalidInputReason::Malformed,
            "test failure",
        );
        let result = crate::SolveScope::from(&mut context).run(
            &mut child_state,
            identity(2),
            SolveBudgets::unbounded(),
            |child| {
                assert!(child.try_consume(BudgetKind::Node, 1)?.is_granted());
                assert!(!child.try_consume(BudgetKind::Node, 1)?.is_granted());
                child.invoke_provider::<()>(EvaluationRole::Constraint, || Err(failure.clone()))
            },
        );
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::Malformed,
                ..
            }
        ));
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Constraint),
            1
        );
        assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 1);
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), 1);
    }

    #[test]
    fn ancestor_count_overflow_does_not_partially_charge_a_child() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        *context.state.counts = SolveCounts::from_own(
            EvaluationCounts::new().with_provider_count(EvaluationRole::Objective, usize::MAX),
        );
        let mut child_state = SolveState::new();
        crate::SolveScope::from(&mut context)
            .run(
                &mut child_state,
                identity(2),
                SolveBudgets::unbounded(),
                |child| {
                    assert!(matches!(
                        (child.record_provider(EvaluationRole::Objective))
                            .as_ref()
                            .unwrap_err()
                            .view(),
                        crate::SciOxideErrorView::SizeOverflow { .. }
                    ));
                    assert_eq!(child.counts().own().provider(EvaluationRole::Objective), 0);
                    assert_eq!(child.budget_report(BudgetKind::Evaluation).consumed(), 0);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 0);
    }

    #[test]
    fn residual_report_retains_raw_scaled_norm_and_rule() {
        let tolerance = Tolerance::new(0.5_f64, 0.25).unwrap();
        let threshold = tolerance.threshold(2.0).unwrap();
        let report = ResidualReport::new(
            0.5,
            NormSelection::L2,
            ResidualScaling::new(ResidualScalingRule::ToleranceThreshold, threshold).unwrap(),
        )
        .unwrap();
        assert_eq!(report.raw_norm(), 0.5);
        assert_eq!(report.scaled_norm(), 0.5);
        assert_eq!(report.norm(), NormSelection::L2);
        assert_eq!(
            report.scaling().rule(),
            ResidualScalingRule::ToleranceThreshold
        );
    }

    #[test]
    fn child_budget_conserves_parent_and_does_not_charge_unused_allowance() {
        let mut parent =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(7));
        {
            let allowance =
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(5));
            let mut parent_state = SolveState::new();
            let mut root = parent_state
                .begin_solve(ExecutionContext::sequential(), identity(1), &mut parent)
                .unwrap();
            let mut child_state = SolveState::new();
            crate::SolveScope::from(&mut root)
                .run(&mut child_state, identity(2), allowance, |child| {
                    assert!(
                        child
                            .try_consume(BudgetKind::Evaluation, 3)
                            .unwrap()
                            .is_granted()
                    );
                    assert!(
                        !child
                            .try_consume(BudgetKind::Evaluation, 3)
                            .unwrap()
                            .is_granted()
                    );
                    assert_eq!(child.budget_report(BudgetKind::Evaluation).consumed(), 3);
                    Ok(())
                })
                .unwrap();
            assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 3);
        }
        assert_eq!(parent.report(BudgetKind::Evaluation).remaining(), Some(4));
    }

    #[test]
    fn begin_resets_ephemeral_state_and_resume_validates_before_mutation() {
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(3));
        let first_identity = identity(1);
        let checkpoint = {
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), first_identity, &mut budgets)
                .unwrap();
            assert!(
                context
                    .record_provider(EvaluationRole::Function)
                    .unwrap()
                    .is_granted()
            );
            assert_eq!(context.record_proposal().unwrap(), Admission::Granted(()));
            context
                .commit_accepted(PartialResultTrust::ResidualChecked)
                .unwrap();
            context.checkpoint(11).unwrap()
        };

        let mismatch = state.resume(
            ExecutionContext::sequential(),
            identity(2),
            11,
            checkpoint,
            &mut budgets,
        );
        assert!(matches!(
            (mismatch).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));

        let context = state
            .resume(
                ExecutionContext::sequential(),
                first_identity,
                11,
                checkpoint,
                &mut budgets,
            )
            .unwrap();
        assert_eq!(context.counts().own().provider(EvaluationRole::Function), 1);
        assert_eq!(context.accepted_state().unwrap().revision(), 1);
    }

    #[test]
    fn provider_failures_remain_counted_and_cancellation_precedes_deadline() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(execution, identity(1), &mut budgets)
            .unwrap();
        assert_eq!(
            context.stop_reason().unwrap(),
            Some(ExecutionStopReason::Cancelled)
        );
        let provider_error = context.invoke_provider(EvaluationRole::Function, || {
            Err::<(), _>(SciOxideError::invalid_input(
                crate::argument!("callback"),
                InvalidInputReason::Malformed,
                "provider failure",
            ))
        });
        assert!(matches!(
            (provider_error).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(context.counts().own().provider(EvaluationRole::Function), 1);
    }

    #[test]
    fn exhausted_provider_budget_does_not_call_or_count_provider() {
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(1));
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(1), &mut budgets)
            .unwrap();
        let mut calls = 0;
        assert_eq!(
            context
                .invoke_provider(EvaluationRole::Function, || {
                    calls += 1;
                    Ok(7_u8)
                })
                .unwrap(),
            Invocation::Evaluated(7)
        );
        assert_eq!(
            context
                .invoke_provider(EvaluationRole::Function, || {
                    calls += 1;
                    Ok(9_u8)
                })
                .unwrap(),
            Invocation::BudgetExhausted(context.budget_report(BudgetKind::Evaluation))
        );
        assert_eq!(calls, 1);
        assert_eq!(context.counts().own().provider(EvaluationRole::Function), 1);
        assert_eq!(context.budget_report(BudgetKind::Evaluation).consumed(), 1);
    }
}
