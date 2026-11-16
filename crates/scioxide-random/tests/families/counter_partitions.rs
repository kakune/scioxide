//! Replay under injected schedulers, independent of completion order.
#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;
use scioxide_core::{
    ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, SciOxideError,
    SequentialExecutor,
};
use scioxide_random::{
    BitGenerator, BoundedSubstream, Distribution, Philox4x64, Rng, SeedableRng, SerializableState,
    SkipAhead, SplittableRng, StreamPartitioner, Threefry4x64, Uniform,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Schedule {
    Forward,
    Reverse,
    Concurrent,
}
struct Scheduler(Schedule);

impl Executor for Scheduler {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        if matches!(self.0, Schedule::Forward) {
            return SequentialExecutor.execute(context, plan, operation);
        }
        // Core derives restricted child contexts. Use its reference executor as
        // a scope issuer, admitting the user operation only for the selected
        // partition. Each partition's actual operation executes exactly once.
        let run = |index| {
            SequentialExecutor.execute(context, plan, &|partition, (), scope| {
                if partition.worker_index() == index {
                    operation(partition, (), scope)
                } else {
                    Ok(())
                }
            })
        };
        match self.0 {
            Schedule::Reverse => {
                for index in (0..plan.worker_count()).rev() {
                    run(index)?;
                }
            }
            Schedule::Concurrent => std::thread::scope(|scope| {
                // Launch every worker before joining; then join every worker
                // even when an earlier result failed. Neither loop may short-circuit.
                let mut handles = Vec::with_capacity(plan.worker_count());
                for index in 0..plan.worker_count() {
                    let run = &run;
                    handles.push(scope.spawn(move || run(index)));
                }
                let mut first = Ok(());
                for handle in handles {
                    let result = handle.join().unwrap().map(|_| ());
                    if first.is_ok() {
                        first = result;
                    }
                }
                first
            })?,
            Schedule::Forward => unreachable!(),
        }
        SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
    }
}

fn replay<B>()
where
    B: SeedableRng + SerializableState + SplittableRng + SkipAhead + Clone + Send + Sync,
    B: Rng<f32> + Rng<f64>,
{
    let scheduler = Scheduler(Schedule::Forward);
    let context = ExecutionContext::new(&scheduler);
    let workers = scheduler.worker_budget();
    let plan = context.plan(
        ExecutionPolicy::Parallel(workers),
        40,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    );
    let partitioner = StreamPartitioner::from_plan(plan, 0x529);
    let mut base = B::from_seed(42);
    let root = base.export_state().unwrap();
    let run = |base: &B, schedule| {
        let scheduler = Scheduler(schedule);
        let output = Mutex::new(vec![None; 4]);
        ExecutionContext::new(&scheduler)
            .execute(plan, &|partition, (), _| {
                let mut rng = partitioner.for_partition(base, &partition, 19)?;
                let mut bits = Vec::new();
                for _ in 0..6 {
                    let a: f32 = rng.next_float()?;
                    let b: f64 = rng.next_float()?;
                    bits.extend([u64::from(a.to_bits()), b.to_bits(), rng.next_u64()?]);
                }
                let value: f64 = Uniform::new(0.0, 1.0)?.sample(&mut rng)?;
                bits.push(value.to_bits());
                let error = rng.next_u64().unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
                assert_eq!(rng.remaining(), 0);
                output.lock().unwrap()[partition.worker_index()] = Some(bits);
                Ok(())
            })
            .unwrap();
        output.into_inner().unwrap()
    };
    let expected = run(&base, Schedule::Forward);
    assert_eq!(base.export_state().unwrap(), root);
    base.skip_ahead(700).unwrap();
    assert_eq!(expected, run(&base, Schedule::Reverse));
    assert_eq!(expected, run(&base, Schedule::Concurrent));
    let partitions: Vec<_> = plan.partitions().collect();
    let foreign = context
        .plan(
            ExecutionPolicy::Parallel(workers),
            41,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: workers,
            },
        )
        .partitions()
        .next()
        .unwrap();
    assert!(partitioner.for_partition(&base, &foreign, 1).is_err());
    assert!(partitioner.for_partition(&base, &partitions[0], 0).is_err());
    assert!(
        partitioner
            .for_partition(&base, &partitions[0], (1_u128 << 66) + 1)
            .is_err()
    );
    let selected = partitioner.for_partition(&base, &partitions[0], 1).unwrap();
    let other = StreamPartitioner::from_plan(plan, 0x530)
        .for_partition(&base, &partitions[0], 1)
        .unwrap();
    assert_ne!(selected.stream_identity(), other.stream_identity());

    let root = B::from_seed(42);
    let mut sequential = root.clone();
    for index in 0..4 {
        let mut chunk = BoundedSubstream::for_chunk(&root, index, 7).unwrap();
        for _ in 0..7 {
            assert_eq!(chunk.next_u64().unwrap(), sequential.next_u64().unwrap());
        }
        assert!(chunk.next_u64().is_err());
    }
}

#[test]
fn both_engines_replay_across_schedules() {
    replay::<Philox4x64>();
    replay::<Threefry4x64>();
}
