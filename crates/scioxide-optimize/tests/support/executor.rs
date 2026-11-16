//! Inject admitted partitions without enabling numerical std convenience types.
use core::num::NonZeroUsize;
use scioxide_core::{
    ExecutionContext, ExecutionPlan, Executor, PartitionFn, SciOxideError, SequentialExecutor,
};

pub struct PartitionExecutor(pub NonZeroUsize);

impl Executor for PartitionExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.0
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)
    }
}
