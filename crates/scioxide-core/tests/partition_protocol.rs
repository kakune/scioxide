//! Partition ownership and scoped-thread execution contracts.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    ExecutionContext, ExecutionFailureReason, ExecutionPlan, ExecutionPolicy, Executor,
    PartitionFn, PlanLimits, SciOxideError, SciOxideErrorView, backend::PartitionSlot,
};

fn plan(cx: ExecutionContext<'_>, n: usize) -> ExecutionPlan {
    let workers = NonZeroUsize::new(4).unwrap();
    cx.plan(
        ExecutionPolicy::Parallel(workers),
        n,
        PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    )
}
#[derive(Clone, Copy)]
enum Submission {
    Complete,
    Missing,
    Duplicate,
    Foreign,
}
struct TestExecutor(Submission);
impl Executor for TestExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        cx: ExecutionContext<'_>,
        schedule: ExecutionPlan,
        f: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        match self.0 {
            Submission::Missing => Ok(()),
            Submission::Complete => {
                for part in schedule.partitions() {
                    f(part, (), cx)?;
                }
                Ok(())
            }
            Submission::Duplicate => {
                if let Some(part) = schedule.partitions().next() {
                    f(part.clone(), (), cx)?;
                    f(part, (), cx)?;
                }
                Ok(())
            }
            Submission::Foreign => {
                if let Some(part) = plan(cx, 100).partitions().next() {
                    f(part, (), cx)?;
                }
                Ok(())
            }
        }
    }
}
#[test]
fn protocol_failures_do_not_allocate_or_repeat_callbacks() {
    for (submission, reason, expected_calls) in [
        (
            Submission::Missing,
            ExecutionFailureReason::MissingPartition,
            0,
        ),
        (
            Submission::Duplicate,
            ExecutionFailureReason::DuplicatePartition,
            1,
        ),
        (
            Submission::Foreign,
            ExecutionFailureReason::UnknownPartition,
            0,
        ),
    ] {
        let executor = TestExecutor(submission);
        let cx = ExecutionContext::new(&executor);
        let schedule = plan(cx, 4);
        let slots = core::array::from_fn::<_, 4, _>(|_| PartitionSlot::new(()));
        let calls = AtomicUsize::new(0);
        {
            let error = cx
                .execute_partitioned(schedule, &slots, &|_, (), child| {
                    assert_eq!(child.worker_budget(), NonZeroUsize::MIN);
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
                .unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::ExecutionFailure { reason: actual, .. } if actual == reason)
            );
        };
        assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
    }
}
#[test]
fn partition_inputs_are_moved_once_and_lengths_are_checked_before_work() {
    let executor = TestExecutor(Submission::Complete);
    let cx = ExecutionContext::new(&executor);
    let schedule = plan(cx, 4);
    let mut output = [0_usize; 4];
    let mut elements = output.iter_mut();
    let slots = core::array::from_fn::<_, 4, _>(|_| PartitionSlot::new(elements.next().unwrap()));
    cx.execute_partitioned(schedule, &slots, &|part, value, _| {
        *value = part.worker_index() + 1;
        Ok(())
    })
    .unwrap();
    assert!(slots.iter().all(PartitionSlot::was_claimed));
    let error = cx
        .execute_partitioned(schedule, &slots[..3], &|_, _, _| unreachable!())
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    let error = cx
        .execute_partitioned(schedule, &slots, &|_, _, _| unreachable!())
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionFailure {
            reason: ExecutionFailureReason::DuplicatePartition,
            ..
        }
    ));
    drop(slots);
    assert_eq!(output, [1, 2, 3, 4]);
    cx.execute_partitioned::<()>(plan(cx, 0), &[], &|_, (), _| unreachable!())
        .unwrap();
}
#[cfg(feature = "std")]
#[test]
fn scoped_threads_restrict_fresh_and_cross_executor_contexts() {
    use scioxide_core::ThreadExecutor;
    let workers = NonZeroUsize::new(4).unwrap();
    let first = ThreadExecutor::new(workers);
    let second = ThreadExecutor::new(workers);
    let cx = ExecutionContext::new(&first);
    let caller = std::thread::current().id();
    cx.execute(plan(cx, 4), &|part, (), child| {
        if part.worker_index() == 0 {
            assert_eq!(std::thread::current().id(), caller);
        }
        let current = std::thread::current().id();
        assert_eq!(child.worker_budget(), NonZeroUsize::MIN);
        for executor in [&first, &second] {
            let fresh = ExecutionContext::new(executor);
            assert_eq!(fresh.worker_budget(), NonZeroUsize::MIN);
            fresh.execute(plan(fresh, 4), &|_, (), nested| {
                assert_eq!(std::thread::current().id(), current);
                assert_eq!(nested.worker_budget(), NonZeroUsize::MIN);
                Ok(())
            })?;
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(cx.worker_budget(), workers);
}
#[cfg(feature = "std")]
#[test]
fn scoped_threads_join_all_workers_and_choose_the_lowest_partition_failure() {
    use scioxide_core::ThreadExecutor;
    let executor = ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
    let cx = ExecutionContext::new(&executor);
    let completed = AtomicUsize::new(0);
    let error = cx
        .execute(plan(cx, 4), &|part, (), _| {
            completed.fetch_add(1, Ordering::Relaxed);
            if part.worker_index() == 1 {
                std::panic::resume_unwind(Box::new("callback"));
            }
            if part.worker_index() == 0 {
                return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                    "sentinel"
                )));
            }
            Ok(())
        })
        .unwrap_err();
    assert_eq!(completed.load(Ordering::Relaxed), 4);
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::SizeOverflow);
    for n in [1, 4] {
        let error = cx
            .execute(plan(cx, n), &|_, (), _| {
                std::panic::resume_unwind(Box::new("callback"))
            })
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionFailure {
                reason: ExecutionFailureReason::CallbackPanicked,
                partition: Some(0),
                ..
            }
        ));
    }
}

#[test]
fn callback_boundary_keeps_parent_clock_and_checks_each_partition_once() {
    use scioxide_core::{CapabilityError, Clock, ClockBinding, Deadline, MonotonicInstant};
    struct TestClock(AtomicUsize);
    impl Clock for TestClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::ZERO,
            ))
        }
    }
    struct ForeignContextExecutor<'a>(&'a ClockBinding<'a>);
    impl Executor for ForeignContextExecutor<'_> {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _cx: ExecutionContext<'_>,
            plan: ExecutionPlan,
            f: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            let foreign = ExecutionContext::sequential().with_clock(self.0)?;
            for partition in plan.partitions() {
                f(partition, (), foreign)?;
            }
            Ok(())
        }
    }
    let clock = TestClock(AtomicUsize::new(0));
    let binding = ClockBinding::new(&clock);
    let other = ClockBinding::new(&clock);
    let executor = ForeignContextExecutor(&other);
    let cx = ExecutionContext::new(&executor)
        .with_deadline(Deadline::at(
            &binding,
            MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
        ))
        .unwrap();
    cx.execute(plan(cx, 4), &|_, (), child| {
        assert!(core::ptr::eq(child.clock().unwrap(), &binding));
        assert!(child.with_clock(&other).is_err());
        assert_eq!(child.worker_budget(), NonZeroUsize::MIN);
        #[cfg(feature = "std")]
        assert_eq!(
            ExecutionContext::new(&executor).worker_budget(),
            NonZeroUsize::MIN
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(clock.0.load(Ordering::Relaxed), 4);

    #[cfg(feature = "std")]
    {
        let threads = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
        let cx = ExecutionContext::new(&threads)
            .with_deadline(Deadline::at(
                &binding,
                MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
            ))
            .unwrap();
        for direct in [false, true] {
            clock.0.store(0, Ordering::Relaxed);
            let callback = |_: scioxide_core::WorkPartition, (), child: ExecutionContext<'_>| {
                assert!(core::ptr::eq(child.clock().unwrap(), &binding));
                assert_eq!(child.worker_budget(), NonZeroUsize::MIN);
                assert_eq!(
                    ExecutionContext::new(&threads).worker_budget(),
                    NonZeroUsize::MIN
                );
                Ok(())
            };
            if direct {
                threads.execute(cx, plan(cx, 4), &callback).unwrap();
            } else {
                cx.execute(plan(cx, 4), &callback).unwrap();
            }
            assert_eq!(clock.0.load(Ordering::Relaxed), 4);
        }
    }
}
