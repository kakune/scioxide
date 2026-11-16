//! Stream partitioning demonstration for scioxide-random.
//!
//! Demonstrates parallel RNG stream partitioning:
//!
//! 1. **Planned streams** — stable `ExecutionPlan` partition identities
//! 2. **Chunk streams** — explicit contiguous `BoundedSubstream` ranges
//! 3. **Seed sequence** — deterministic seed generation for engine initialization
//! 4. **Determinism** — full pipeline reproducibility check
//!
//! Run: `cargo run --example stream_partitioning -p scioxide-random`

use core::num::NonZeroUsize;

use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError, ThreadExecutor};
use scioxide_random::{
    BitGenerator, BoundedSubstream, Pcg64, Rng, SeedableRng, SkipAhead, SplitMix64,
    StreamPartitioner, seed_sequence,
};

// ---------------------------------------------------------------------------
// Part 1: Worker streams
// ---------------------------------------------------------------------------

fn part1_worker_streams() -> Result<(), SciOxideError> {
    println!("=== Part 1: Worker streams ===");
    println!();

    let base = Pcg64::from_seed(42);
    let workers = NonZeroUsize::MIN.saturating_add(3);
    let executor = ThreadExecutor::new(workers);
    let plan = ExecutionContext::new(&executor).plan(
        ExecutionPolicy::Parallel(workers),
        4,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: workers,
        },
    );
    let partitioner = StreamPartitioner::from_plan(plan, 0x390);
    println!("  Partitioner: {} partitions", partitioner.num_partitions());

    for partition in plan.partitions() {
        let w = partition.worker_index();
        let mut stream = partitioner.for_partition(&base, &partition, 1)?;
        let val: f64 = stream.next_float()?;
        println!("  Worker {w}: first float = {val:.10}");
        assert!(
            (0.0..1.0).contains(&val),
            "Float out of range for worker {w}"
        );
    }

    // Different workers should produce different values
    let mut partitions = plan.partitions();
    let first = partitions.next().ok_or_else(missing_partition)?;
    let second = partitions.next().ok_or_else(missing_partition)?;
    let mut s0 = partitioner.for_partition(&base, &first, 1)?;
    let mut s1 = partitioner.for_partition(&base, &second, 1)?;
    let v0 = s0.next_u64()?;
    let v1 = s1.next_u64()?;
    assert_ne!(
        v0, v1,
        "Different workers should produce different sequences"
    );

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Chunk streams
// ---------------------------------------------------------------------------

fn part2_chunk_streams() -> Result<(), SciOxideError> {
    println!("=== Part 2: Chunk streams ===");
    println!();

    let base = SplitMix64::from_seed(42);
    let chunk_len = 1000_u64;

    // for_chunk should be equivalent to skipping chunk_index * chunk_len
    for chunk in 0..4 {
        let mut stream = BoundedSubstream::for_chunk(&base, chunk, chunk_len)?;
        let val = stream.next_u64()?;
        println!(
            "  Chunk {chunk} (offset {}): first u64 = {val}",
            chunk * chunk_len
        );
    }

    // Verify chunk 2 matches manual skip
    let mut manual = base.clone();
    manual.skip_ahead(u128::from(2 * chunk_len))?;
    let manual_val = manual.next_u64()?;

    let mut chunk_stream = BoundedSubstream::for_chunk(&base, 2, chunk_len)?;
    let chunk_val = chunk_stream.next_u64()?;

    println!("  Manual skip to {}: {manual_val}", 2 * chunk_len);
    println!("  Chunk 2 first:     {chunk_val}");
    assert_eq!(
        manual_val, chunk_val,
        "for_chunk must match manual skip_ahead"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Seed sequence
// ---------------------------------------------------------------------------

fn part3_seed_sequence() -> Result<(), SciOxideError> {
    println!("=== Part 3: Seed sequence ===");
    println!();

    let n = 8;
    let seeds = seed_sequence(42, n)?;
    println!("  seed_sequence(42, {n}): {seeds:?}");

    assert_eq!(seeds.len(), n, "Must produce exactly {n} seeds");

    // All seeds should be distinct
    let mut unique = seeds.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), n, "All seeds must be distinct");

    // Use seeds to initialize engines
    println!("  Engines initialized from seed sequence:");
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = SplitMix64::from_seed(seed);
        let val: f64 = rng.next_float()?;
        println!("    Engine {i} (seed={seed}): first float = {val:.10}");
    }

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Determinism
// ---------------------------------------------------------------------------

fn part4_determinism() -> Result<(), SciOxideError> {
    println!("=== Part 4: Determinism ===");
    println!();

    // Full pipeline: same setup must produce identical results
    let run = || -> Result<Vec<f64>, SciOxideError> {
        let base = Pcg64::from_seed(42);
        let workers = NonZeroUsize::MIN.saturating_add(1);
        let executor = ThreadExecutor::new(workers);
        let plan = ExecutionContext::new(&executor).plan(
            ExecutionPolicy::Parallel(workers),
            2,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: workers,
            },
        );
        let partitioner = StreamPartitioner::from_plan(plan, 0x390);

        let mut results = Vec::new();
        for partition in plan.partitions() {
            let mut stream = partitioner.for_partition(&base, &partition, 5)?;
            for _ in 0..5 {
                results.push(stream.next_float()?);
            }
        }
        Ok(results)
    };

    let run_a = run()?;
    let run_b = run()?;

    println!("  Run A: {:?}", &run_a[..5]);
    println!("  Run B: {:?}", &run_b[..5]);
    assert_eq!(
        run_a, run_b,
        "Identical setup must produce identical results"
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

fn missing_partition() -> SciOxideError {
    SciOxideError::external("execution plan did not contain its declared partitions")
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-random Stream Partitioning Examples");
    println!("==========================================");
    println!();

    part1_worker_streams()?;
    part2_chunk_streams()?;
    part3_seed_sequence()?;
    part4_determinism()?;

    println!("All examples completed successfully.");
    Ok(())
}
