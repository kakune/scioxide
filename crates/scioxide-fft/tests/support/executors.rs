//! Executors that violate the partition protocol, for transactional tests.

use scioxide_core::{
    ExecutionContext, ExecutionPlan, Executor, PartitionFn, SciOxideError, SequentialExecutor,
};
use std::num::NonZeroUsize;

/// A four-worker executor that skips every partition or runs each twice.
///
/// Parallel plans must detect either violation through their partition
/// claims, return an error and leave caller output unchanged.
#[derive(Clone, Copy, Debug)]
pub enum BrokenExecutor {
    /// Return success without running any partition.
    SkipPartitions,
    /// Run every partition twice.
    DuplicatePartitions,
}

impl BrokenExecutor {
    /// Both protocol violations.
    pub const ALL: [Self; 2] = [Self::SkipPartitions, Self::DuplicatePartitions];
}

impl Executor for BrokenExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN.saturating_add(3)
    }

    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::SkipPartitions => SequentialExecutor.execute(context, plan, &|_, (), _| Ok(())),
            Self::DuplicatePartitions => {
                SequentialExecutor.execute(context, plan, operation)?;
                SequentialExecutor.execute(context, plan, operation)
            }
        }
    }
}
