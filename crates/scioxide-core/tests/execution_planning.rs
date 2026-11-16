//! Deterministic partition identities, malformed-result rejection and limits.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ExecutionContext, ExecutionFailureReason, ExecutionPlan, ExecutionPolicy, Executor,
    PartitionFn, PlanLimits, SciOxideError, SequentialExecutor,
};
use std::{cell::Cell, num::NonZeroUsize};

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}
struct Planner(NonZeroUsize);
impl Executor for Planner {
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
fn limits() -> PlanLimits {
    PlanLimits {
        work_per_item: core::num::NonZeroUsize::MIN,
        auto_threshold: nz(3),
        worker_limit: nz(4),
    }
}

#[test]
fn auto_work_floor_bounds_every_partition_without_overflow() {
    struct FloorPlanner(core::sync::atomic::AtomicUsize, NonZeroUsize);
    impl Executor for FloorPlanner {
        fn worker_budget(&self) -> NonZeroUsize {
            nz(8)
        }
        fn auto_min_work_per_worker(&self) -> NonZeroUsize {
            self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.1
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
    let executor = FloorPlanner(core::sync::atomic::AtomicUsize::new(0), nz(10));
    let context = ExecutionContext::new(&executor);
    for (units, count, expected) in [
        (1, 0, 0),
        (1, 1, 1),
        (1, 19, 1),
        (1, 20, 2),
        (1, 39, 3),
        (1, 40, 4),
        (3, 7, 1),
        (3, 8, 2),
        (10, 2, 1),
        (10, 3, 3),
        (usize::MAX, usize::MAX, 4),
    ] {
        let limits = PlanLimits {
            work_per_item: nz(units),
            ..limits()
        };
        let plan = context.plan(ExecutionPolicy::Auto, count, limits);
        assert_eq!(
            plan.worker_count(),
            expected,
            "units={units}, count={count}"
        );
        if expected > 1 {
            assert!(
                plan.partitions()
                    .all(|p| p.range().len() >= 10_usize.div_ceil(units))
            );
        }
    }
    executor.0.store(0, core::sync::atomic::Ordering::Relaxed);
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Parallel(nz(4)),
    ] {
        let plan = context.plan(policy, 4, limits());
        assert_eq!(
            plan.worker_count(),
            if policy == ExecutionPolicy::Sequential {
                1
            } else {
                4
            }
        );
    }
    assert_eq!(executor.0.load(core::sync::atomic::Ordering::Relaxed), 0);
    let extreme = FloorPlanner(core::sync::atomic::AtomicUsize::new(0), NonZeroUsize::MAX);
    for (units, expected) in [(1, 1), (2, 1), (3, 3), (usize::MAX, 4)] {
        let plan = ExecutionContext::new(&extreme).plan(
            ExecutionPolicy::Auto,
            usize::MAX,
            PlanLimits {
                work_per_item: nz(units),
                ..limits()
            },
        );
        assert_eq!(plan.worker_count(), expected);
    }
}
#[test]
fn plans_respect_each_named_limit_and_cover_extreme_ranges() {
    let executor = Planner(nz(8));
    let context = ExecutionContext::new(&executor);
    for count in [0, 1, 2, 3, 4, 17, usize::MAX] {
        let plan = context.plan(ExecutionPolicy::Auto, count, limits());
        assert_eq!(
            plan.worker_count(),
            if count < 3 {
                usize::from(count != 0)
            } else {
                count.min(4)
            }
        );
        let partitions = plan.partitions().collect::<Vec<_>>();
        assert_eq!(partitions.first().map(|p| p.range().start).unwrap_or(0), 0);
        assert_eq!(partitions.last().map(|p| p.range().end).unwrap_or(0), count);
        for pair in partitions.windows(2) {
            assert_eq!(pair[0].range().end, pair[1].range().start);
        }
        for (index, partition) in partitions.iter().enumerate() {
            assert_eq!(plan.partition(index).as_ref(), Some(partition));
        }
        assert!(plan.partition(plan.worker_count()).is_none());
        assert!(plan.partition(usize::MAX).is_none());
    }
    assert_eq!(
        context
            .plan(ExecutionPolicy::Parallel(nz(2)), 100, limits())
            .worker_count(),
        2
    );
    assert_eq!(
        ExecutionContext::sequential()
            .plan(ExecutionPolicy::Parallel(nz(8)), 100, limits())
            .worker_count(),
        1
    );
}
#[test]
fn reduction_checks_all_results_before_merging_and_uses_plan_order() {
    let executor = Planner(nz(8));
    let context = ExecutionContext::new(&executor);
    let plan = context.plan(ExecutionPolicy::Auto, 16, limits());
    let mut results = plan
        .partitions()
        .map(|p| {
            let index = p.worker_index();
            (p, index)
        })
        .collect::<Vec<_>>();
    results.reverse();
    let order = plan
        .reduce(results, Vec::new(), |mut order, index| {
            order.push(index);
            Ok(order)
        })
        .unwrap();
    assert_eq!(order, [0, 1, 2, 3]);
    let first = plan.partition(0).unwrap();
    let foreign = context.plan(
        ExecutionPolicy::Parallel(nz(8)),
        32,
        PlanLimits {
            worker_limit: nz(8),
            ..limits()
        },
    );
    let cases = [
        (
            vec![(first.clone(), 0), (first.clone(), 1)],
            ExecutionFailureReason::DuplicatePartition,
        ),
        (vec![(first, 0)], ExecutionFailureReason::MissingPartition),
        (
            vec![(foreign.partition(7).unwrap(), 0)],
            ExecutionFailureReason::UnknownPartition,
        ),
        (
            vec![(
                context
                    .plan(ExecutionPolicy::Sequential, 32, limits())
                    .partition(0)
                    .unwrap(),
                0,
            )],
            ExecutionFailureReason::UnknownPartition,
        ),
    ];
    for (results, expected) in cases {
        let calls = Cell::new(0);
        let result = plan.reduce(results, 0, |a, b| {
            calls.set(calls.get() + 1);
            Ok(a + b)
        });
        assert!(
            matches!((result).as_ref().unwrap_err().view(), scioxide_core :: SciOxideErrorView :: ExecutionFailure { reason , .. } if reason == expected)
        );
        assert_eq!(calls.get(), 0);
    }
    let error = plan
        .reduce(plan.partitions().map(|p| (p, ())), 0, |_, ()| {
            Err(scioxide_core::SciOxideError::singular(None))
        })
        .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
    let empty = context.plan(ExecutionPolicy::Auto, 0, limits());
    assert_eq!(
        empty
            .reduce(
                std::iter::empty::<(scioxide_core::WorkPartition, usize)>(),
                42,
                |a, b| Ok(a + b)
            )
            .unwrap(),
        42
    );
}
