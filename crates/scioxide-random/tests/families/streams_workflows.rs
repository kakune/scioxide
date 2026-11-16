//! streams workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use core::num::NonZeroUsize;
use scioxide_core::{
    ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, SciOxideError,
};
use scioxide_random::*;

struct PlanningExecutor(NonZeroUsize);

impl Executor for PlanningExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.0
    }

    fn execute(
        &self,
        _context: ExecutionContext<'_>,
        _plan: ExecutionPlan,
        _operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("executor"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "planning-only executor cannot execute work",
        ))
    }
}

#[test]
fn test_stream_partitioner_non_overlap() {
    let base = Pcg64::from_seed(42);
    let workers = NonZeroUsize::new(4).unwrap();
    let executor = PlanningExecutor(workers);
    let plan = ExecutionContext::new(&executor).plan(
        ExecutionPolicy::Parallel(workers),
        400,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    );
    let partitioner = StreamPartitioner::from_plan(plan, 0x390);

    // Collect 100 values from each stable plan partition.
    let streams: Vec<Vec<f64>> = plan
        .partitions()
        .map(|partition| {
            let mut rng = partitioner.for_partition(&base, &partition, 100).unwrap();
            (0..100).map(|_| rng.next_float().unwrap()).collect()
        })
        .collect();

    // Verify streams differ
    for i in 0..streams.len() {
        for j in (i + 1)..streams.len() {
            assert_ne!(streams[i], streams[j], "streams {i} and {j} should differ");
        }
    }
}

#[test]
fn test_stream_partitioner_chunk_sequential() {
    let base = SplitMix64::from_seed(42);
    let chunk_len = 500_u64;

    // Chunk 0 then chunk 1 should match a single sequential run
    let mut rng0 = scioxide_random::BoundedSubstream::for_chunk(&base, 0, chunk_len).unwrap();
    let mut rng1 = scioxide_random::BoundedSubstream::for_chunk(&base, 1, chunk_len).unwrap();

    let mut sequential = base;
    for _ in 0..chunk_len {
        let a: f64 = rng0.next_float().unwrap();
        let b: f64 = sequential.next_float().unwrap();
        assert!((a - b).abs() < 1e-15);
    }
    for _ in 0..chunk_len {
        let a: f64 = rng1.next_float().unwrap();
        let b: f64 = sequential.next_float().unwrap();
        assert!((a - b).abs() < 1e-15);
    }
}
