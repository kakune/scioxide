//! Scoped execution of disjoint mutable chunks.

use scioxide_core::{ExecutionContext, NumericScalar, SciOxideError};

pub(crate) fn execute_chunks<S: NumericScalar>(
    data: &mut [S],
    stride: usize,
    policy: scioxide_core::ExecutionPolicy,
    execution: ExecutionContext<'_>,
    work: impl Fn(usize, &mut [S]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::backend::PartitionSlot;
    if data.is_empty() {
        return Ok(());
    }
    let limit = NonZeroUsize::MIN.saturating_add(31);
    let plan = execution.plan(
        policy,
        data.len().div_ceil(stride),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: limit,
        },
    );
    if plan.worker_count() == 1 {
        return work(0, data);
    }
    // The data is split into disjoint `&mut` chunks up front, one per planned
    // partition, and parked in take-once slots. A (possibly untrusted)
    // executor can then run partitions concurrently without aliasing; a
    // foreign, duplicated or skipped partition is detected and reported
    // instead of silently leaving part of the data unprocessed.
    let mut slots: [PartitionSlot<&mut [S]>; 32] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = data;
    for part in plan.partitions() {
        let (chunk, rest) = crate::workspace::split_mut(
            remaining,
            (part.range().len() * stride).min(remaining.len()),
        )?;
        slots[part.worker_index()] = PartitionSlot::new(chunk);
        remaining = rest;
    }
    execution.execute_partitioned(plan, &slots[..plan.worker_count()], &|part, chunk, _| {
        work(part.range().start, chunk)
    })?;

    Ok(())
}
