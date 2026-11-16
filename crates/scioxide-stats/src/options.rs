//! Options shared by batch statistics.
use crate::{MergePolicy, MissingPolicy, MomentOrder, NonFinitePolicy};
use scioxide_core::ExecutionPolicy;

/// Options shared by the functions that take a [`BatchView`](crate::BatchView).
///
/// The options select how missing and non-finite observations are handled,
/// how work is split across workers, which moments streaming reductions
/// accumulate, and the largest workspace a requirement query accepts.
/// [`Self::new`] lists the defaults. The execution context passed with the
/// options supplies the executor and its worker budget; see
/// [Execution contexts](scioxide_core#execution-contexts).
///
/// ```
/// use scioxide_stats::{MissingPolicy, StatisticsOptions};
///
/// let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
/// assert_eq!(options.missing_policy(), MissingPolicy::Skip);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StatisticsOptions {
    missing: MissingPolicy,
    non_finite: NonFinitePolicy,
    execution: ExecutionPolicy,
    merge: MergePolicy,
    order: MomentOrder,
    max_bytes: usize,
}
impl Default for StatisticsOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl StatisticsOptions {
    /// Returns the default options.
    ///
    /// Missing and non-finite observations are rejected
    /// ([`MissingPolicy::Reject`], [`NonFinitePolicy::Reject`]), execution is
    /// sequential, moment reductions merge in input order
    /// ([`MergePolicy::InputOrder`]), streaming moments go up to
    /// [`MomentOrder::Second`], and there is no byte limit.
    pub const fn new() -> Self {
        Self {
            missing: MissingPolicy::Reject,
            non_finite: NonFinitePolicy::Reject,
            execution: ExecutionPolicy::Sequential,
            merge: MergePolicy::InputOrder,
            order: MomentOrder::Second,
            max_bytes: usize::MAX,
        }
    }
    /// Sets how observations marked missing by the validity mask are handled.
    pub const fn with_missing_policy(mut self, value: MissingPolicy) -> Self {
        self.missing = value;
        self
    }
    /// Sets how NaN and infinite values of present observations are handled.
    pub const fn with_non_finite_policy(mut self, value: NonFinitePolicy) -> Self {
        self.non_finite = value;
        self
    }
    /// Sets the execution policy. Independent batches may run on separate
    /// workers, up to what the execution context provides.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution = value;
        self
    }
    /// Sets the order in which moment reductions combine observations.
    ///
    /// [`MergePolicy::InputOrder`] reduces each batch sequentially.
    /// [`MergePolicy::StablePartitions`] lets a reduction run on several
    /// workers and merges the partial results in a fixed partition order.
    pub const fn with_merge_policy(mut self, value: MergePolicy) -> Self {
        self.merge = value;
        self
    }
    /// Sets the highest moment that streaming moment reductions accumulate.
    pub const fn with_moment_order(mut self, value: MomentOrder) -> Self {
        self.order = value;
        self
    }
    /// Sets the largest workspace size in bytes.
    ///
    /// Requirement queries and reservations return `ResourceLimit` when a
    /// workspace would exceed it. The limit includes storage that a workspace
    /// already holds from earlier reservations.
    pub const fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the missing-observation policy.
    pub const fn missing_policy(self) -> MissingPolicy {
        self.missing
    }
    /// Returns the non-finite-observation policy.
    pub const fn non_finite_policy(self) -> NonFinitePolicy {
        self.non_finite
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution
    }
    /// Returns the merge policy.
    pub const fn merge_policy(self) -> MergePolicy {
        self.merge
    }
    /// Returns the highest accumulated moment.
    pub const fn moment_order(self) -> MomentOrder {
        self.order
    }
    /// Returns the workspace byte limit (`usize::MAX` when unlimited).
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
    pub(crate) fn order(self, axis: crate::Axis) -> crate::OrderPolicy {
        crate::OrderPolicy {
            axis: match axis {
                crate::Axis::All => None,
                crate::Axis::Index(i) => Some(i),
            },
            missing: self.missing,
            non_finite: self.non_finite,
            execution: self.execution,
            workers: core::num::NonZeroUsize::MIN.saturating_add(63),
        }
    }
}
