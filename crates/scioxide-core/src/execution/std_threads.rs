//! Executor on scoped standard-library threads.
#![forbid(unsafe_code)]
use super::{
    ExecutionContext, ExecutionMembership, ExecutionPlan, Executor, PartitionFn, WorkPartition,
    discard_panic_payload, invoke_partition,
};
use crate::{ExecutionFailureReason, SciOxideError};
use core::num::NonZeroUsize;

/// Executor that runs partitions on scoped threads, up to a fixed number of workers.
///
/// Each call joins all its threads before returning, and partition zero runs
/// on the calling thread. Work started from inside a callback runs on that
/// callback's thread. A panic in a callback becomes an `ExecutionFailure` with
/// reason `CallbackPanicked`; an aborting panic or a callback that never
/// returns cannot be recovered. Under `Auto`, a worker is started only for at
/// least 32,768 estimated work units per partition; `Parallel` ignores this
/// minimum.
///
/// Run `cargo run -p scioxide-core --example workspace_operations --features std` for a complete caller-owned storage example.
///
/// A context for a long computation adds native kernels, cancellation and a
/// deadline:
///
/// ```
/// use core::{num::NonZeroUsize, time::Duration};
/// use scioxide_core::{CancellationToken, ClockBinding, ExecutionContext, StdClock, ThreadExecutor};
///
/// let executor = ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
/// let clock = StdClock::new();
/// let binding = ClockBinding::new(&clock);
/// let cancel = CancellationToken::new();
/// let cx = ExecutionContext::native(&executor)
///     .with_cancellation(&cancel)
///     .with_clock(&binding)?
///     .with_time_limit(Duration::from_secs(10))?;
/// assert!(cx.check().is_ok());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ThreadExecutor {
    workers: NonZeroUsize,
}
impl ThreadExecutor {
    /// Creates an executor with at most `workers` workers.
    ///
    /// Threads are created only while parallel work runs.
    pub const fn new(workers: NonZeroUsize) -> Self {
        Self { workers }
    }
}

fn invoke_catching(
    cx: ExecutionContext<'_>,
    partition: WorkPartition,
    f: &PartitionFn<'_>,
) -> Result<(), SciOxideError> {
    let index = partition.worker_index();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        invoke_partition(cx, partition, f)
    })) {
        Ok(result) => result,
        Err(payload) => {
            discard_panic_payload(payload);
            Err(SciOxideError::execution_failure(
                Some(index),
                ExecutionFailureReason::CallbackPanicked,
                "executor callback panicked",
            ))
        }
    }
}
impl Executor for ThreadExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.workers
    }
    fn auto_min_work_per_worker(&self) -> NonZeroUsize {
        const { NonZeroUsize::new(32_768).expect("positive Auto work floor") }
    }
    fn execute(
        &self,
        cx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        f: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let run = |partition| invoke_catching(cx, partition, f);
        if plan.worker_count() <= 1
            || cx.worker_budget() == NonZeroUsize::MIN
            || ExecutionMembership::active()
        {
            let mut result = Ok(());
            for partition in plan.partitions() {
                result = result.and(run(partition));
            }
            return result;
        }
        let mut partitions = plan.partitions();
        let Some(first) = partitions.next() else {
            return Ok(());
        };
        // Reject impossible handle storage before scope allocates its bookkeeping.
        core::alloc::Layout::array::<(
            usize,
            std::thread::ScopedJoinHandle<'_, Result<(), SciOxideError>>,
        )>(partitions.len())
        .map_err(|_| {
            SciOxideError::allocation_failure(
                crate::quantity!("executor_workers"),
                partitions.len(),
            )
        })?;
        std::thread::scope(|scope| {
            let mut handles = alloc::vec::Vec::new();
            handles.try_reserve_exact(partitions.len()).map_err(|_| {
                SciOxideError::allocation_failure(
                    crate::quantity!("executor_workers"),
                    partitions.len(),
                )
            })?;
            let mut spawn_error = None;
            for partition in partitions {
                let index = partition.worker_index();
                match std::thread::Builder::new().spawn_scoped(scope, move || run(partition)) {
                    Ok(handle) => handles.push((index, handle)),
                    Err(error) => {
                        spawn_error = Some(SciOxideError::execution_failure(
                            Some(index),
                            ExecutionFailureReason::WorkerCreation {
                                os_error: error.raw_os_error(),
                            },
                            "could not create executor worker",
                        ));
                        break;
                    }
                }
            }
            let mut result = run(first);
            for (index, handle) in handles {
                let joined = match handle.join() {
                    Ok(result) => result,
                    Err(payload) => {
                        discard_panic_payload(payload);
                        Err(SciOxideError::execution_failure(
                            Some(index),
                            ExecutionFailureReason::CallbackPanicked,
                            "executor callback panicked",
                        ))
                    }
                };
                result = result.and(joined);
            }
            result.and(spawn_error.map_or(Ok(()), Err))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionPolicy;
    use alloc::{vec, vec::Vec};
    use core::sync::atomic::Ordering;
    use std::sync::Mutex;

    fn non_zero(value: usize) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(value).unwrap()
    }
    fn plan(context: ExecutionContext<'_>, workers: usize) -> ExecutionPlan {
        context.plan(
            ExecutionPolicy::Parallel(non_zero(workers)),
            workers,
            crate::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: non_zero(1),
                worker_limit: non_zero(workers),
            },
        )
    }
    #[test]
    fn concurrent_callers_keep_partition_workspaces_isolated() {
        let executor = ThreadExecutor::new(non_zero(4));
        std::thread::scope(|scope| {
            (0..16).for_each(|solve| {
                let executor = &executor;
                scope.spawn(move || {
                    let context = ExecutionContext::new(executor);
                    let plan = context.plan(
                        ExecutionPolicy::Parallel(non_zero(4)),
                        64,
                        crate::PlanLimits {
                            work_per_item: core::num::NonZeroUsize::MIN,
                            auto_threshold: non_zero(1),
                            worker_limit: non_zero(4),
                        },
                    );
                    let workspace = Mutex::new(vec![usize::MAX; 64]);
                    context
                        .execute(plan, &|partition, (), _| {
                            let mut workspace = workspace.lock().unwrap();
                            partition.range().for_each(|index| workspace[index] = solve);
                            Ok(())
                        })
                        .unwrap();
                    assert!(
                        workspace
                            .into_inner()
                            .unwrap()
                            .into_iter()
                            .all(|value| value == solve)
                    );
                });
            });
        });
    }
    #[test]
    fn concurrent_failures_report_the_lowest_partition() {
        let executor = ThreadExecutor::new(non_zero(4));
        let context = ExecutionContext::new(&executor);
        let plan = context.plan(
            ExecutionPolicy::Parallel(non_zero(4)),
            4,
            crate::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: non_zero(1),
                worker_limit: non_zero(4),
            },
        );
        let error = context
            .execute(plan, &|partition, (), _| {
                if matches!(partition.worker_index(), 1 | 3) {
                    return Err(SciOxideError::execution_failure(
                        Some(partition.worker_index()),
                        crate::ExecutionFailureReason::ProviderFailure,
                        "injected failure",
                    ));
                }
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(
            (error).view(),
            crate::SciOxideErrorView::ExecutionFailure {
                partition: Some(1),
                ..
            }
        ));
    }
    #[test]
    fn scoped_fresh_contexts_and_cross_executor_cycles_are_inline() {
        for workers in [1, 2, 4] {
            let a = ThreadExecutor::new(non_zero(workers));
            let b = ThreadExecutor::new(non_zero(workers));
            let context = ExecutionContext::new(&a);
            let outer = plan(context, workers);
            context
                .execute(outer, &|_, (), scope| {
                    let thread = std::thread::current().id();
                    for executor in [&a, &b, &a] {
                        let fresh = ExecutionContext::new(executor);
                        assert_eq!(fresh.worker_budget(), non_zero(1));
                        let inner = plan(fresh, workers);
                        assert_eq!(inner.worker_count(), 1);
                        fresh.execute(inner, &|_, (), _| {
                            assert_eq!(std::thread::current().id(), thread);
                            let return_to_a = ExecutionContext::new(&a);
                            return_to_a.execute(plan(return_to_a, workers), &|_, (), _| {
                                assert_eq!(std::thread::current().id(), thread);
                                Ok(())
                            })?;
                            Ok(())
                        })?;
                    }
                    assert_eq!(scope.worker_budget(), non_zero(1));
                    Ok(())
                })
                .unwrap();
            assert_eq!(context.worker_budget(), non_zero(workers));
        }
    }
    #[test]
    fn scoped_budgets_survive_transfer_to_another_thread_and_executor() {
        let a = ThreadExecutor::new(non_zero(2));
        let b = ThreadExecutor::new(non_zero(4));
        let foreign_plan = plan(ExecutionContext::new(&b), 4);
        let context = ExecutionContext::new(&a);
        context
            .execute(plan(context, 2), &|_, (), scope| {
                let fresh = ExecutionContext::new(&b);
                std::thread::scope(|threads| {
                    threads
                        .spawn(|| {
                            assert!(!ExecutionMembership::active());
                            let caller = std::thread::current().id();
                            for inherited in [scope, fresh] {
                                assert_eq!(inherited.worker_budget(), non_zero(1));
                                b.execute(inherited, foreign_plan, &|_, (), nested| {
                                    assert_eq!(std::thread::current().id(), caller);
                                    assert_eq!(nested.worker_budget(), non_zero(1));
                                    Ok(())
                                })
                                .unwrap();
                            }
                        })
                        .join()
                        .unwrap();
                });
                Ok(())
            })
            .unwrap();
    }
    #[test]
    fn scoped_errors_and_stops_reconcile_every_partition() {
        use crate::{
            CancellationToken, CapabilityError, Clock, ClockBinding, Deadline, ExecutionStopReason,
            MonotonicInstant,
        };
        struct TestClock(bool);
        impl Clock for TestClock {
            fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
                if self.0 {
                    Err(CapabilityError::new("provider detail"))
                } else {
                    Ok(MonotonicInstant::from_duration_since_epoch(
                        core::time::Duration::from_secs(1),
                    ))
                }
            }
        }
        let pool = ThreadExecutor::new(non_zero(4));
        let root = ExecutionContext::new(&pool);
        let schedule = plan(root, 4);
        for context in [root, ExecutionContext::sequential()] {
            let calls: [std::sync::atomic::AtomicUsize; 4] =
                core::array::from_fn(|_| std::sync::atomic::AtomicUsize::new(0));
            let error = pool
                .execute(context, schedule, &|partition, (), _| {
                    let index = partition.worker_index();
                    calls[index].fetch_add(1, Ordering::Relaxed);
                    match index {
                        1 => Err(crate::SciOxideError::singular(None)),
                        3 => std::panic::resume_unwind(Box::new("callback panic")),
                        _ => Ok(()),
                    }
                })
                .unwrap_err();
            assert!(matches!(
                (error).view(),
                crate::SciOxideErrorView::Singular { .. }
            ));
            assert!(calls.iter().all(|count| count.load(Ordering::Relaxed) == 1));
            for fail in [false, true] {
                let clock = TestClock(fail);
                let binding = ClockBinding::new(&clock);
                let timed = context
                    .with_deadline(Deadline::at(
                        &binding,
                        MonotonicInstant::from_duration_since_epoch(core::time::Duration::ZERO),
                    ))
                    .unwrap();
                let observed = std::sync::atomic::AtomicUsize::new(0);
                let error = pool
                    .execute(timed, schedule, &|_, (), _| {
                        observed.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    })
                    .unwrap_err();
                if fail {
                    assert!(
                        matches!((error).view(), crate :: SciOxideErrorView :: CapabilityFailure { capability : crate :: Capability :: Clock , detail , .. } if detail == "provider detail")
                    );
                } else {
                    assert!(matches!(
                        (error).view(),
                        crate::SciOxideErrorView::ExecutionStopped {
                            reason: ExecutionStopReason::DeadlineReached,
                            ..
                        }
                    ));
                }
                assert_eq!(observed.load(Ordering::Relaxed), 0);
                let cancellation = CancellationToken::new();
                cancellation.cancel();
                let error = pool
                    .execute(
                        timed.with_cancellation(&cancellation),
                        schedule,
                        &|_, (), _| {
                            observed.fetch_add(1, Ordering::Relaxed);
                            Ok(())
                        },
                    )
                    .unwrap_err();
                assert!(matches!(
                    (error).view(),
                    crate::SciOxideErrorView::ExecutionStopped {
                        reason: ExecutionStopReason::Cancelled,
                        ..
                    }
                ));
                assert_eq!(observed.load(Ordering::Relaxed), 0);
            }
            let completed = core::sync::atomic::AtomicUsize::new(0);
            pool.execute(context, schedule, &|_, (), _| {
                completed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
            assert_eq!(completed.load(Ordering::Relaxed), 4);
        }
    }

    struct DropPanic;
    impl Drop for DropPanic {
        fn drop(&mut self) {
            std::panic::resume_unwind(Box::new(DropPanic));
        }
    }
    #[test]
    fn scoped_mutable_cleanup_and_payload_drop_precede_return() {
        struct Cleanup<'a>(&'a mut usize);
        impl Drop for Cleanup<'_> {
            fn drop(&mut self) {
                *self.0 = 2;
            }
        }
        let executor = ThreadExecutor::new(non_zero(4));
        let context = ExecutionContext::new(&executor);
        for count in [1, 4] {
            let mut data = vec![0; count];
            let slots = data
                .iter_mut()
                .map(super::super::PartitionSlot::new)
                .collect::<Vec<_>>();
            let error = context
                .execute_partitioned(plan(context, count), &slots, &|partition, value, _| {
                    *value = 1;
                    let _cleanup = Cleanup(value);
                    if partition.worker_index() == 0 {
                        std::panic::resume_unwind(Box::new(DropPanic));
                    }
                    Ok(())
                })
                .unwrap_err();
            assert!(matches!(
                error.view(),
                crate::SciOxideErrorView::ExecutionFailure {
                    reason: ExecutionFailureReason::CallbackPanicked,
                    ..
                }
            ));
            drop(slots);
            assert_eq!(data, vec![2; count]);
            context
                .execute(plan(context, count), &|_, (), _| Ok(()))
                .unwrap();
        }
    }
    #[test]
    fn sequential_executor_propagates_panics() {
        let context = ExecutionContext::sequential();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.execute(plan(context, 1), &|_, (), _| {
                std::panic::resume_unwind(Box::new(()))
            })
        }));
        assert!(outcome.is_err());
    }
}
