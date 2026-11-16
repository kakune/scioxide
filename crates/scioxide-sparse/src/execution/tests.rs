#![allow(clippy::unwrap_used)]
use super::*;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{CancellationToken, ExecutionFailureReason, Executor, SciOxideErrorKind};

#[derive(Clone, Copy)]
enum Completion {
    Complete,
    OmitLast,
    FailAfterFirst,
    DuplicateFirst,
    CancelAfterAll,
}
struct TestExecutor<'a> {
    completion: Completion,
    cancellation: &'a CancellationToken,
    workers: NonZeroUsize,
}
impl Executor for TestExecutor<'_> {
    fn worker_budget(&self) -> NonZeroUsize {
        self.workers
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        kernel: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        for partition in plan.partitions() {
            if matches!(self.completion, Completion::OmitLast)
                && partition.worker_index() + 1 == plan.worker_count()
            {
                return Ok(());
            }
            kernel(partition.clone(), (), context)?;
            if matches!(self.completion, Completion::FailAfterFirst) {
                return Err(SciOxideError::execution_failure(
                    None,
                    ExecutionFailureReason::ProviderFailure,
                    "injected failure after a completed partition",
                ));
            }
            if matches!(self.completion, Completion::DuplicateFirst) {
                kernel(partition, (), context)?;
            }
        }
        if matches!(self.completion, Completion::CancelAfterAll) {
            self.cancellation.cancel();
        }
        Ok(())
    }
}

#[test]
fn executor_and_late_kernel_failures_preserve_every_column() {
    for (completion, fail_column, expected) in [
        (
            Completion::OmitLast,
            None,
            SciOxideErrorKind::ExecutionFailure,
        ),
        (
            Completion::FailAfterFirst,
            None,
            SciOxideErrorKind::ExecutionFailure,
        ),
        (
            Completion::DuplicateFirst,
            None,
            SciOxideErrorKind::ExecutionFailure,
        ),
        (
            Completion::CancelAfterAll,
            None,
            SciOxideErrorKind::ExecutionStopped,
        ),
        (
            Completion::Complete,
            Some(3),
            SciOxideErrorKind::InvalidInput,
        ),
    ] {
        let token = CancellationToken::new();
        let workers = NonZeroUsize::new(2).unwrap();
        let executor = TestExecutor {
            completion,
            cancellation: &token,
            workers,
        };
        let calls = AtomicUsize::new(0);
        let mut output = [19.; 10];
        let mut scratch = [23.; 17];
        let result = columns(
            &mut BlockVectorViewMut::new(&mut output, 2, 5).unwrap(),
            &mut scratch,
            3,
            ExecutionPolicy::Parallel(workers),
            workers,
            ExecutionContext::new(&executor).with_cancellation(&token),
            &|column, candidate, work, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                candidate.fill(2.);
                work.fill(3.);
                if fail_column == Some(column) {
                    return Err(crate::helpers::invalid(
                        scioxide_core::argument!("rhs"),
                        "injected column failure",
                    ));
                }
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err().kind(), expected);
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_eq!(output, [19.; 10]);
        assert_eq!(&scratch[15..], &[23.; 2]);
    }
}

#[test]
fn uneven_partitions_publish_in_column_order_without_touching_extra_scratch() {
    for worker_count in [1, 2, 3, 64] {
        let token = CancellationToken::new();
        let workers = NonZeroUsize::new(worker_count).unwrap();
        let executor = TestExecutor {
            completion: Completion::Complete,
            cancellation: &token,
            workers,
        };
        let mut output = [19.; 130];
        let mut scratch = [23.; 197];
        columns(
            &mut BlockVectorViewMut::new(&mut output, 2, 65).unwrap(),
            &mut scratch,
            3,
            ExecutionPolicy::Parallel(workers),
            workers,
            ExecutionContext::new(&executor),
            &|column, candidate, work, _| {
                assert_eq!(candidate.len(), 2);
                assert_eq!(work.len(), 1);
                candidate.fill(column as f64);
                work.fill(-1.);
                Ok(())
            },
        )
        .unwrap();
        for (column, actual) in output.chunks_exact(2).enumerate() {
            assert!(
                actual
                    .iter()
                    .all(|v| (v - column as f64).abs() <= f64::EPSILON)
            );
        }
        assert_eq!(&scratch[195..], &[23.; 2]);
    }
}

#[test]
fn empty_blocks_short_scratch_and_worker_boundaries() {
    let context = ExecutionContext::sequential();
    let workers = NonZeroUsize::MIN;
    let mut empty: [f64; 0] = [];
    columns(
        &mut BlockVectorViewMut::new(&mut empty, usize::MAX, 0).unwrap(),
        &mut [],
        0,
        ExecutionPolicy::Sequential,
        workers,
        context,
        &|_, _, _, _| {
            Err(crate::helpers::invalid(
                scioxide_core::argument!("rhs"),
                "unexpected kernel invocation",
            ))
        },
    )
    .unwrap();
    columns(
        &mut BlockVectorViewMut::new(&mut empty, 0, 5).unwrap(),
        &mut [],
        0,
        ExecutionPolicy::Sequential,
        workers,
        context,
        &|_, _, _, _| {
            Err(crate::helpers::invalid(
                scioxide_core::argument!("rhs"),
                "unexpected kernel invocation",
            ))
        },
    )
    .unwrap();
    let mut output = [19.; 2];
    let result = columns(
        &mut BlockVectorViewMut::new(&mut output, 2, 1).unwrap(),
        &mut [0.],
        2,
        ExecutionPolicy::Sequential,
        workers,
        context,
        &|_, _, _, _| {
            Err(crate::helpers::invalid(
                scioxide_core::argument!("rhs"),
                "unexpected kernel invocation",
            ))
        },
    );
    assert_eq!(
        result.unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [19.; 2]);
    for (policy, workers) in [
        (ExecutionPolicy::Sequential, 1),
        (ExecutionPolicy::Auto, 64),
        (ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap()), 3),
        (
            ExecutionPolicy::Parallel(NonZeroUsize::new(65).unwrap()),
            64,
        ),
    ] {
        assert_eq!(policy_workers(policy).get(), workers);
    }
}
