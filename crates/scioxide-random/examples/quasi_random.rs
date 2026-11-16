//! Quasi-random sequence demonstration for scioxide-random.
//!
//! Demonstrates low-discrepancy sequence generators:
//!
//! 1. **Sobol basic** — 2D Sobol generation, verify range
//! 2. **Scrambling** — digital shift and linear matrix scramble
//! 3. **Halton** — 3D Halton sequence
//! 4. **Skip & batch** — deterministic reproduction via skip and batch
//! 5. **Brownian bridge** — Sobol-driven Brownian path construction
//!
//! Run: `cargo run --example quasi_random -p scioxide-random`

use scioxide_core::SciOxideError;
use scioxide_random::Quantile;
use scioxide_random::{BrownianBridgePlan, QmcPlan, QuasiRng, SamplingWorkspace};

// ---------------------------------------------------------------------------
// Part 1: Sobol basic
// ---------------------------------------------------------------------------

fn part1_sobol_basic() -> Result<(), SciOxideError> {
    println!("=== Part 1: Sobol basic ===");
    println!();

    let sobol_plan = QmcPlan::sobol(2)?;
    let mut sobol = sobol_plan.sequence::<f64>()?;
    println!("  Sobol dim = {}", sobol.dim());
    assert_eq!(sobol.dim(), 2);

    let n = 16;
    // One caller-owned row-major buffer is the allocation-free prepared path.
    let mut points = vec![0.0; n * sobol.dim()];
    let policy =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut workspace = policy.workspace(scioxide_random::SamplingRequest::rows(n, sobol.dim()))?;
    sobol.points_into(policy, &mut points, &mut workspace)?;
    println!("  Generated {n} points in 2D:");
    for (i, pt) in points.chunks_exact(2).enumerate().take(8) {
        println!("    [{i:2}] ({:.6}, {:.6})", pt[0], pt[1]);
    }

    // Verify all points in [0, 1)
    for pt in points.chunks_exact(2) {
        for &coord in pt {
            assert!(
                (0.0..1.0).contains(&coord),
                "Sobol point out of [0, 1) range: {coord}"
            );
        }
    }

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Scrambling
// ---------------------------------------------------------------------------

fn part2_scrambling() -> Result<(), SciOxideError> {
    println!("=== Part 2: Scrambling ===");
    println!();

    // Digital shift
    let sobol_ds_plan =
        QmcPlan::sobol(2)?.with_scramble(scioxide_random::QmcScramble::DigitalShift(42))?;
    let mut sobol_ds = sobol_ds_plan.sequence::<f64>()?;
    let pts_ds = sobol_ds.points(8)?;
    println!("  Digital-shift Sobol (seed=42), first 4 points:");
    for (i, pt) in pts_ds.chunks_exact(2).enumerate().take(4) {
        println!("    [{i}] ({:.6}, {:.6})", pt[0], pt[1]);
    }

    // linear matrix scramble
    let sobol_linear_plan = QmcPlan::sobol(2)?.with_randomization(
        scioxide_random::QmcRandomization::LinearMatrixShift,
        &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(42),
    )?;
    let mut sobol_linear = sobol_linear_plan.sequence::<f64>()?;
    let pts_linear = sobol_linear.points(8)?;
    println!("  Linear-scrambled Sobol (seed=42), first 4 points:");
    for (i, pt) in pts_linear.chunks_exact(2).enumerate().take(4) {
        println!("    [{i}] ({:.6}, {:.6})", pt[0], pt[1]);
    }

    // Both should still be in [0, 1)
    for pt in pts_ds.chunks_exact(2).chain(pts_linear.chunks_exact(2)) {
        for &coord in pt {
            assert!(
                (0.0..1.0).contains(&coord),
                "Scrambled point out of range: {coord}"
            );
        }
    }

    // Scrambled and unscrambled should differ
    let sobol_plain_plan = QmcPlan::sobol(2)?;
    let mut sobol_plain = sobol_plain_plan.sequence::<f64>()?;
    let pts_plain = sobol_plain.points(8)?;
    assert_ne!(
        &pts_plain[2..4],
        &pts_ds[2..4],
        "Digital-shift should differ from plain"
    );

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Halton
// ---------------------------------------------------------------------------

fn part3_halton() -> Result<(), SciOxideError> {
    println!("=== Part 3: Halton ===");
    println!();

    let halton_plan = QmcPlan::halton(3)?;
    let mut halton = halton_plan.sequence::<f64>()?;
    println!("  Halton dim = {}", halton.dim());
    assert_eq!(halton.dim(), 3);

    let n = 10;
    let points = halton.points(n)?;
    println!("  Generated {n} points in 3D:");
    for (i, pt) in points.chunks_exact(3).enumerate() {
        println!("    [{i}] ({:.6}, {:.6}, {:.6})", pt[0], pt[1], pt[2]);
    }

    // Verify all in [0, 1)
    for pt in points.chunks_exact(3) {
        for &coord in pt {
            assert!(
                (0.0..1.0).contains(&coord),
                "Halton point out of range: {coord}"
            );
        }
    }

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Skip & batch
// ---------------------------------------------------------------------------

fn part4_skip_and_batch() -> Result<(), SciOxideError> {
    println!("=== Part 4: Skip & batch ===");
    println!();

    // Generate sequentially, then compare with skip_to
    let sobol_seq_plan = QmcPlan::sobol(2)?;
    let mut sobol_seq = sobol_seq_plan.sequence::<f64>()?;
    let flat = sobol_seq.points(10)?; // one allocation, 10 row-major points
    assert_eq!(flat.len(), 20);
    let pt_10 = sobol_seq.next_point()?;

    let sobol_skip_plan = QmcPlan::sobol(2)?;
    let mut sobol_skip = sobol_skip_plan.sequence::<f64>()?;
    sobol_skip.skip_to(10)?;
    let pt_skip = sobol_skip.next_point()?;

    println!("  Sequential point 10: ({:.6}, {:.6})", pt_10[0], pt_10[1]);
    println!(
        "  Skip-to(10) point:   ({:.6}, {:.6})",
        pt_skip[0], pt_skip[1]
    );
    assert!(
        (pt_10[0] - pt_skip[0]).abs() < 1e-15 && (pt_10[1] - pt_skip[1]).abs() < 1e-15,
        "skip_to must reproduce sequential generation"
    );

    // skip() relative test
    let sobol_a_plan = QmcPlan::sobol(2)?;
    let mut sobol_a = sobol_a_plan.sequence::<f64>()?;
    sobol_a.skip(5)?;
    let pt_a = sobol_a.next_point()?;

    let sobol_b_plan = QmcPlan::sobol(2)?;
    let mut sobol_b = sobol_b_plan.sequence::<f64>()?;
    sobol_b.skip_to(5)?;
    let pt_b = sobol_b.next_point()?;

    println!("  skip(5):    ({:.6}, {:.6})", pt_a[0], pt_a[1]);
    println!("  skip_to(5): ({:.6}, {:.6})", pt_b[0], pt_b[1]);
    assert!(
        (pt_a[0] - pt_b[0]).abs() < 1e-15 && (pt_a[1] - pt_b[1]).abs() < 1e-15,
        "skip(5) must match skip_to(5)"
    );

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Brownian bridge
// ---------------------------------------------------------------------------

fn part5_brownian_bridge() -> Result<(), SciOxideError> {
    println!("=== Part 5: Brownian bridge ===");
    println!();

    // Generate one low-discrepancy point with one coordinate per bridge input.
    let n_steps = 8_usize;
    let sobol_plan = QmcPlan::sobol(n_steps)?;
    let mut sobol = sobol_plan.sequence::<f64>()?;
    sobol.skip_to(13)?;
    let raw_point = sobol.next_point()?;

    // Convert each Sobol coordinate from (0,1) to a standard normal.
    let normal = scioxide_random::Normal::new(0.0, 1.0)?;
    let z: Vec<f64> = raw_point
        .into_iter()
        .map(|u| normal.quantile(u))
        .collect::<Result<Vec<_>, _>>()?;

    // A bridge time grid includes t_0, so n steps require n+1 times.
    let times: Vec<f64> = (0..=n_steps)
        .map(|index| index as f64 / n_steps as f64)
        .collect();
    let plan = BrownianBridgePlan::new(&times)?;
    let mut path = vec![0.0; plan.output_len()];
    let mut workspace = SamplingWorkspace::new();

    // The same plan and output buffer can be reused for subsequent Sobol points.
    plan.apply_into(&z, &mut path, &mut workspace)?;

    println!("  Brownian bridge path ({n_steps} steps, T=1):");
    for (i, &w) in path.iter().enumerate() {
        println!("    t={:.3} : W = {w:+.6}", times[i + 1]);
    }
    assert_eq!(
        path.len(),
        n_steps,
        "Path length must match number of steps"
    );

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
    println!("scioxide-random Quasi-Random Examples");
    println!("====================================");
    println!();

    part1_sobol_basic()?;
    part2_scrambling()?;
    part3_halton()?;
    part4_skip_and_batch()?;
    part5_brownian_bridge()?;

    println!("All examples completed successfully.");
    Ok(())
}
