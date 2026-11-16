//! PRNG engine demonstration for scioxide-random.
//!
//! Demonstrates the pseudo-random number generator engines:
//!
//! 1. **Seeding & generation** — create engines, generate floats and u64s
//! 2. **Reproducibility** — same seed produces same sequence
//! 3. **State serialization** — export/import roundtrip
//! 4. **Skip-ahead** — deterministic skip vs sequential generation
//! 5. **Jump & split** — jump on Xoshiro, split on Pcg64
//!
//! Run: `cargo run --example prng_engines -p scioxide-random`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::unnecessary_wraps
)]

use scioxide_core::SciOxideError;
use scioxide_random::{
    BitGenerator, JumpableRng, Lcg64, Mrg32k3a, Pcg64, Rng, SeedableRng, SerializableState,
    SkipAhead, SplitMix64, SplittableRng, Xoshiro128StarStar, Xoshiro256StarStar,
};

// ---------------------------------------------------------------------------
// Part 1: Seeding & generation
// ---------------------------------------------------------------------------

fn part1_seeding_generation() -> Result<(), SciOxideError> {
    println!("=== Part 1: Seeding & generation ===");
    println!();

    let mut sm = SplitMix64::from_seed(42);
    let mut xo256 = Xoshiro256StarStar::from_seed(42);
    let mut xo128 = Xoshiro128StarStar::from_seed(42);
    let mut pcg = Pcg64::from_seed(42);
    let mut lcg = Lcg64::from_seed(42);
    let mut mrg = Mrg32k3a::from_seed(42);

    // Generate u64 values
    let u_sm = sm.next_u64()?;
    let u_xo256 = xo256.next_u64()?;
    let u_xo128 = xo128.next_u64()?;
    let u_pcg = pcg.next_u64()?;
    let u_lcg = lcg.next_u64()?;
    let u_mrg = mrg.next_u64()?;
    println!("  SplitMix64  u64 = {u_sm}");
    println!("  Xoshiro256  u64 = {u_xo256}");
    println!("  Xoshiro128  u64 = {u_xo128}");
    println!("  Pcg64       u64 = {u_pcg}");
    println!("  Lcg64       u64 = {u_lcg}");
    println!("  Mrg32k3a    u64 = {u_mrg}");

    // Generate floats in [0, 1)
    let f_sm: f64 = sm.next_float()?;
    let f_xo256: f64 = xo256.next_float()?;
    let f_xo128: f64 = xo128.next_float()?;
    let f_pcg: f64 = pcg.next_float()?;
    let f_lcg: f64 = lcg.next_float()?;
    let f_mrg: f64 = mrg.next_float()?;
    println!("  SplitMix64  f64 = {f_sm}");
    println!("  Xoshiro256  f64 = {f_xo256}");
    println!("  Xoshiro128  f64 = {f_xo128}");
    println!("  Pcg64       f64 = {f_pcg}");
    println!("  Lcg64       f64 = {f_lcg}");
    println!("  Mrg32k3a    f64 = {f_mrg}");

    assert!((0.0..1.0).contains(&f_sm), "SplitMix64 float out of range");
    assert!(
        (0.0..1.0).contains(&f_xo256),
        "Xoshiro256 float out of range"
    );
    assert!(
        (0.0..1.0).contains(&f_xo128),
        "Xoshiro128 float out of range"
    );
    assert!((0.0..1.0).contains(&f_pcg), "Pcg64 float out of range");
    assert!((0.0..1.0).contains(&f_lcg), "Lcg64 float out of range");
    assert!((0.0..1.0).contains(&f_mrg), "Mrg32k3a float out of range");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Reproducibility
// ---------------------------------------------------------------------------

fn part2_reproducibility() -> Result<(), SciOxideError> {
    println!("=== Part 2: Reproducibility ===");
    println!();

    let mut rng_a = SplitMix64::from_seed(42);
    let mut rng_b = SplitMix64::from_seed(42);

    let seq_a: Vec<u64> = (0..10)
        .map(|_| rng_a.next_u64())
        .collect::<Result<_, _>>()?;
    let seq_b: Vec<u64> = (0..10)
        .map(|_| rng_b.next_u64())
        .collect::<Result<_, _>>()?;

    println!("  Sequence A (seed 42): {:?}", &seq_a[..5]);
    println!("  Sequence B (seed 42): {:?}", &seq_b[..5]);
    assert_eq!(seq_a, seq_b, "Same seed must produce identical sequences");

    // Different seed produces different sequence
    let mut rng_c = SplitMix64::from_seed(99);
    let seq_c: Vec<u64> = (0..10)
        .map(|_| rng_c.next_u64())
        .collect::<Result<_, _>>()?;
    println!("  Sequence C (seed 99): {:?}", &seq_c[..5]);
    assert_ne!(seq_a, seq_c, "Different seeds should differ");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: State serialization
// ---------------------------------------------------------------------------

fn part3_state_serialization() -> Result<(), SciOxideError> {
    println!("=== Part 3: State serialization ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);
    // Advance a few steps
    for _ in 0..5 {
        rng.next_u64()?;
    }

    // Export state
    let state = rng.export_state()?;
    let state_len = state.len();
    println!(
        "  Exported state ({state_len} bytes): {:?}",
        &state[..8.min(state.len())]
    );

    // Import state into a new engine
    let mut restored = SplitMix64::import_state(&state)?;

    // Both should produce identical output from this point
    let orig_vals: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect::<Result<_, _>>()?;
    let rest_vals: Vec<u64> = (0..5)
        .map(|_| restored.next_u64())
        .collect::<Result<_, _>>()?;

    println!("  Original next 5:  {orig_vals:?}");
    println!("  Restored next 5:  {rest_vals:?}");
    assert_eq!(orig_vals, rest_vals, "Restored state must match original");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Skip-ahead
// ---------------------------------------------------------------------------

fn part4_skip_ahead() -> Result<(), SciOxideError> {
    println!("=== Part 4: Skip-ahead ===");
    println!();

    // SplitMix64 skip-ahead
    let mut rng_seq = SplitMix64::from_seed(42);
    let mut rng_skip = SplitMix64::from_seed(42);

    // Advance sequentially by 1000 steps
    for _ in 0..1000 {
        rng_seq.next_u64()?;
    }

    // Skip-ahead by 1000
    rng_skip.skip_ahead(1000)?;

    let seq_val = rng_seq.next_u64()?;
    let skip_val = rng_skip.next_u64()?;
    println!("  SplitMix64 sequential 1000: {seq_val}");
    println!("  SplitMix64 skip-ahead 1000: {skip_val}");
    assert_eq!(
        seq_val, skip_val,
        "Skip-ahead must match sequential advance"
    );

    // Pcg64 skip-ahead
    let mut pcg_seq = Pcg64::from_seed(42);
    let mut pcg_skip = Pcg64::from_seed(42);

    for _ in 0..500 {
        pcg_seq.next_u64()?;
    }
    pcg_skip.skip_ahead(500)?;

    let pcg_seq_val = pcg_seq.next_u64()?;
    let pcg_skip_val = pcg_skip.next_u64()?;
    println!("  Pcg64 sequential 500: {pcg_seq_val}");
    println!("  Pcg64 skip-ahead 500: {pcg_skip_val}");
    assert_eq!(pcg_seq_val, pcg_skip_val, "Pcg64 skip-ahead must match");

    // Position tracking
    let mut rng = SplitMix64::from_seed(0);
    let pos0 = rng.position();
    rng.next_u64()?;
    let pos1 = rng.position();
    println!("  Position before: {pos0:?}, after one step: {pos1:?}");
    assert_eq!(pos0, Some(0));
    assert_eq!(pos1, Some(1), "Position should increment by 1");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Jump & split
// ---------------------------------------------------------------------------

fn part5_jump_and_split() -> Result<(), SciOxideError> {
    println!("=== Part 5: Jump & split ===");
    println!();

    // JumpableRng on Xoshiro256StarStar
    let mut rng = Xoshiro256StarStar::from_seed(42);
    let before_jump = rng.next_u64()?;
    rng.jump()?;
    let after_jump = rng.next_u64()?;
    println!("  Xoshiro256** before jump: {before_jump}");
    println!("  Xoshiro256** after jump:  {after_jump}");
    assert_ne!(
        before_jump, after_jump,
        "Jump should advance state significantly"
    );

    // SplittableRng on Pcg64
    let base = Pcg64::from_seed(42);
    let mut stream_0 = base.split(0)?;
    let mut stream_1 = base.split(1)?;

    let val_0 = stream_0.next_u64()?;
    let val_1 = stream_1.next_u64()?;
    println!("  Pcg64 stream 0: {val_0}");
    println!("  Pcg64 stream 1: {val_1}");
    assert_ne!(
        val_0, val_1,
        "Different streams should produce different values"
    );

    // Substream
    let mut sub_0_0 = base.substream(0, 0)?;
    let mut sub_0_1 = base.substream(0, 1)?;
    let sub_val_0 = sub_0_0.next_u64()?;
    let sub_val_1 = sub_0_1.next_u64()?;
    println!("  Pcg64 substream(0,0): {sub_val_0}");
    println!("  Pcg64 substream(0,1): {sub_val_1}");
    assert_ne!(sub_val_0, sub_val_1, "Different substreams should differ");

    println!();
    println!("  Part 5 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-random PRNG Engines Examples");
    println!("====================================");
    println!();

    part1_seeding_generation()?;
    part2_reproducibility()?;
    part3_state_serialization()?;
    part4_skip_ahead()?;
    part5_jump_and_split()?;

    println!("All examples completed successfully.");
    Ok(())
}
