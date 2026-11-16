//! Partitioned dispatch without allocation, and all-or-nothing column output.
use core::num::NonZeroUsize;
use scioxide_core::backend::{PartitionSlot, checked_mul};
use scioxide_core::{
    BlockVectorViewMut, ExecutionContext, ExecutionPlan, ExecutionPolicy, NumericScalar,
    PartitionFn, PlanLimits, SciOxideError,
};

/// Worker bound of a dispatch policy: one when sequential, otherwise at most 64.
pub(crate) const fn policy_workers(policy: ExecutionPolicy) -> NonZeroUsize {
    const MAXIMUM: NonZeroUsize = NonZeroUsize::MIN.saturating_add(63);
    match policy {
        ExecutionPolicy::Sequential => NonZeroUsize::MIN,
        ExecutionPolicy::Auto => MAXIMUM,
        ExecutionPolicy::Parallel(workers) if workers.get() > MAXIMUM.get() => MAXIMUM,
        ExecutionPolicy::Parallel(workers) => workers,
    }
}

/// Plans have at most 64 workers. Payloads follow partition order and own
/// disjoint caller-provided scratch.
/// No staged result may be written to an output until this returns successfully.
pub(crate) fn dispatch<I: Send>(
    context: ExecutionContext<'_>,
    plan: ExecutionPlan,
    mut inputs: impl Iterator<Item = I>,
    kernel: &PartitionFn<'_, I>,
) -> Result<(), SciOxideError> {
    context.check()?;
    let mut slots: [PartitionSlot<I>; 64] = core::array::from_fn(|_| PartitionSlot::empty());
    for partition in plan.partitions() {
        let input = inputs.next().ok_or_else(|| {
            SciOxideError::dimension_mismatch(plan.worker_count(), partition.worker_index())
        })?;
        slots[partition.worker_index()] = PartitionSlot::new(input);
    }
    if inputs.next().is_some() {
        return Err(SciOxideError::dimension_mismatch(
            plan.worker_count(),
            plan.worker_count() + 1,
        ));
    }
    context.execute_partitioned(plan, &slots[..plan.worker_count()], kernel)?;
    context.check()
}

/// Each stride starts with one candidate column, followed by kernel-local work.
/// Shapes and family-specific validity are checked by callers. Only the queried
/// scratch prefix is borrowed, and copying to the output cannot fail after dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn columns<S: NumericScalar>(
    output: &mut BlockVectorViewMut<'_, S>,
    scratch: &mut [S],
    stride: usize,
    policy: ExecutionPolicy,
    workers: NonZeroUsize,
    context: ExecutionContext<'_>,
    kernel: &(
         impl Fn(usize, &mut [S], &mut [S], ExecutionContext<'_>) -> Result<(), SciOxideError> + Sync
     ),
) -> Result<(), SciOxideError> {
    stage_columns(
        (output.rows(), output.right_hand_sides()),
        scratch,
        stride,
        policy,
        workers,
        context,
        kernel,
    )?;
    publish_columns(output, scratch, stride);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_columns<S: NumericScalar>(
    (rows, count): (usize, usize),
    scratch: &mut [S],
    stride: usize,
    policy: ExecutionPolicy,
    workers: NonZeroUsize,
    context: ExecutionContext<'_>,
    kernel: &(
         impl Fn(usize, &mut [S], &mut [S], ExecutionContext<'_>) -> Result<(), SciOxideError> + Sync
     ),
) -> Result<(), SciOxideError> {
    let needed = checked_mul(stride, count, scioxide_core::quantity!("block_scratch"))?;
    crate::helpers::minimum(scratch.len(), needed)?;
    context.check()?;
    if count == 0 || stride == 0 {
        return Ok(());
    }
    let plan = context.plan(
        policy,
        count,
        PlanLimits {
            work_per_item: NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    );
    let mut remaining = &mut scratch[..needed];
    let inputs = plan.partitions().map(|partition| {
        let (part, tail) =
            core::mem::take(&mut remaining).split_at_mut(partition.range().len() * stride);
        remaining = tail;
        part
    });
    dispatch(context, plan, inputs, &|partition, part, scope| {
        for (column, buffer) in partition.range().zip(part.chunks_exact_mut(stride)) {
            scope.check()?;
            let (candidate, work) = buffer.split_at_mut(rows);
            kernel(column, candidate, work, scope)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(crate) fn publish_columns<S: NumericScalar>(
    output: &mut BlockVectorViewMut<'_, S>,
    scratch: &[S],
    stride: usize,
) {
    let rows = output.rows();
    if rows != 0 && stride != 0 {
        for (column, buffer) in output
            .as_mut_slice()
            .chunks_exact_mut(rows)
            .zip(scratch.chunks_exact(stride))
        {
            column.copy_from_slice(&buffer[..rows]);
        }
    }
}

#[cfg(test)]
mod tests;
