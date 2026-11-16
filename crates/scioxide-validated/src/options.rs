//! Options shared by every verified plan.
use scioxide_core::{ExecutionPolicy, SolveBudgets};

/// Execution policy, work budgets and storage limit of a verified plan.
///
/// - The [execution policy](scioxide_core#execution-contexts) (default
///   `Sequential`) splits the independent proofs of a batch form between the
///   workers of the executor, in at most 64 partitions. A single proof always
///   runs on one thread. The results do not depend on the policy or the
///   number of workers.
/// - The [budgets](scioxide_core#solve-accounting) (default unbounded) limit the
///   iterations and callback evaluations of each call. A batch divides what
///   remains of them between its proofs by index. An exhausted budget makes a
///   proof `Inconclusive`; it is not an error.
/// - The byte limit (default unlimited) bounds the storage of workspaces and of
///   prepared data such as [`LuIntervalInverse`](crate::LuIntervalInverse).
///   Exceeding it is a `ResourceLimit` error.
///
/// ```
/// use scioxide_core::{Budget, BudgetKind, ExecutionPolicy, SolveBudgets};
/// use scioxide_validated::ValidatedOptions;
///
/// // At most 64 callback evaluations per call and 1 MiB of storage.
/// let budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(64)?);
/// let options = ValidatedOptions::new()
///     .with_budgets(budgets)
///     .with_byte_limit(1 << 20);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// assert_eq!(options.byte_limit(), 1 << 20);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct ValidatedOptions {
    policy: ExecutionPolicy,
    budgets: SolveBudgets,
    byte_limit: usize,
}

impl Default for ValidatedOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidatedOptions {
    /// Returns the defaults: sequential batches, unbounded budgets and no byte
    /// limit.
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
            budgets: SolveBudgets::unbounded(),
            byte_limit: usize::MAX,
        }
    }
    /// Sets the execution policy of batch forms.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy of batch forms.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Sets the budgets of each call.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the budgets of each call.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }
    /// Sets the storage limit of workspaces and prepared data, in bytes.
    pub const fn with_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = bytes;
        self
    }
    /// Returns the storage limit of workspaces and prepared data, in bytes.
    pub const fn byte_limit(&self) -> usize {
        self.byte_limit
    }
}

/// What a root or quadrature workspace must hold: a number of independent
/// proofs, or one refinement into an output of a given capacity.
///
/// - [`Self::proofs`] sizes a workspace for single verifications (`proofs(1)`)
///   and for batches of up to `count` certificates.
/// - [`Self::refinement`] sizes it for one isolation, cover or quadrature
///   generation whose output holds `capacity` tree nodes, disks or panels, and
///   for verifying such an output.
///
/// ```
/// use scioxide_validated::ProofRequest;
///
/// assert_eq!(ProofRequest::proofs(4).proof_count(), 4);
/// assert_eq!(ProofRequest::refinement(32).capacity(), 32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofRequest {
    proofs: usize,
    capacity: usize,
}

impl ProofRequest {
    /// Returns a request for batches of up to `count` independent proofs.
    pub const fn proofs(count: usize) -> Self {
        Self {
            proofs: count,
            capacity: 0,
        }
    }
    /// Returns a request for one refinement into an output of `capacity`
    /// entries.
    pub const fn refinement(capacity: usize) -> Self {
        Self {
            proofs: 1,
            capacity,
        }
    }
    /// Returns the number of independent proofs.
    pub const fn proof_count(&self) -> usize {
        self.proofs
    }
    /// Returns the output capacity of a refinement.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}
