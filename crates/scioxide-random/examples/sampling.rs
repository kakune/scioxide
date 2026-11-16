//! Sampling utilities demonstration for scioxide-random.
//!
//! Demonstrates the sampling module:
//!
//! 1. **Shuffle & choose** — Fisher-Yates shuffle, random element selection
//! 2. **Weighted sampling** — prepared weighted selection
//! 3. **Sample indices** — Floyd's algorithm for distinct indices
//! 4. **Stratified uniform** — one point per stratum of [0, 1) via a one-dimensional Latin hypercube
//! 5. **Latin hypercube** — LHS in multiple dimensions
//!
//! Run: `cargo run --example sampling -p scioxide-random`

use scioxide_core::SciOxideError;
use scioxide_random::{
    SeedableRng, SplitMix64, WeightedSampler, choose, latin_hypercube, sample_indices, shuffle,
};

fn usize_to_f64(value: usize, name: &str) -> Result<f64, SciOxideError> {
    let as_u32 = u32::try_from(value).map_err(|_| {
        SciOxideError::external(format!("{name} is too large for precise f64 conversion"))
    })?;
    Ok(f64::from(as_u32))
}

// ---------------------------------------------------------------------------
// Part 1: Shuffle & choose
// ---------------------------------------------------------------------------

fn part1_shuffle_choose() -> Result<(), SciOxideError> {
    println!("=== Part 1: Shuffle & choose ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    // Shuffle
    let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let original = data.clone();
    println!("  Before shuffle: {data:?}");

    shuffle(&mut data, &mut rng)?;
    println!("  After shuffle:  {data:?}");

    // Same elements, just reordered
    let mut sorted = data.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, original, "Shuffle must preserve elements");

    // Choose
    let items = vec!["apple", "banana", "cherry", "date"];
    let chosen = choose(&items, &mut rng)?;
    println!("  Chosen from fruits: {chosen:?}");
    assert!(
        chosen.is_some(),
        "choose from non-empty slice should return Some"
    );
    if let Some(selected) = chosen {
        assert!(
            items.contains(selected),
            "Chosen item must be from the original slice"
        );
    }

    // Choose from empty slice
    let empty: Vec<i32> = vec![];
    let none = choose(&empty, &mut rng)?;
    assert!(none.is_none(), "choose from empty slice should return None");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Weighted sampling
// ---------------------------------------------------------------------------

fn part2_weighted() -> Result<(), SciOxideError> {
    println!("=== Part 2: Weighted sampling ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    let items = vec!["rare", "common", "very_common"];
    let weights = vec![0.01_f64, 0.29, 0.70];

    // Prepare immutable alias tables when the same weights are reused.
    let sampler = WeightedSampler::new(&weights)?;
    let chosen = sampler.sample(&mut rng, &items)?;
    println!("  Weighted choice: {chosen}");

    // Repeated draws are expected O(1) and allocate nothing.
    let n = 10_000_usize;
    let mut counts = [0usize; 3];
    for _ in 0..n {
        let chosen = sampler.sample(&mut rng, &items)?;
        if *chosen == "rare" {
            counts[0] += 1;
        } else if *chosen == "common" {
            counts[1] += 1;
        } else {
            counts[2] += 1;
        }
    }

    let n_f = usize_to_f64(n, "n")?;
    let freq_rare = usize_to_f64(counts[0], "counts[0]")? / n_f;
    let freq_common = usize_to_f64(counts[1], "counts[1]")? / n_f;
    let freq_very = usize_to_f64(counts[2], "counts[2]")? / n_f;
    println!("  Weighted sampling ({n} draws):");
    println!("    rare:        {freq_rare:.4} (weight 0.01)");
    println!("    common:      {freq_common:.4} (weight 0.29)");
    println!("    very_common: {freq_very:.4} (weight 0.70)");

    assert!(freq_rare < 0.05, "Rare item should appear infrequently");
    assert!(freq_very > 0.60, "Very common item should dominate");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Sample indices
// ---------------------------------------------------------------------------

fn part3_sample_indices() -> Result<(), SciOxideError> {
    println!("=== Part 3: Sample indices ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    let n = 100;
    let k = 10;
    let indices = sample_indices(n, k, &mut rng)?;

    println!("  sample_indices(n={n}, k={k}): {indices:?}");
    assert_eq!(indices.len(), k, "Must return exactly k indices");

    // All must be in range [0, n)
    for &idx in &indices {
        assert!(idx < n, "Index {idx} out of range [0, {n})");
    }

    // All must be distinct
    let mut unique = indices;
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), k, "All indices must be distinct");

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Stratified uniform
// ---------------------------------------------------------------------------

fn part4_stratified() -> Result<(), SciOxideError> {
    println!("=== Part 4: Stratified uniform ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    // A one-dimensional Latin hypercube places one point in each of n strata
    // in random order; sorting puts sample i in stratum i.
    let n = 10;
    let mut samples: Vec<f64> = latin_hypercube(n, 1, &mut rng)?;
    samples.sort_by(f64::total_cmp);

    println!("  Stratified uniform ({n} strata):");
    for (i, &s) in samples.iter().enumerate() {
        println!("    [{i}] {s:.6}");
    }

    assert_eq!(samples.len(), n, "Must return n samples");

    // All in [0, 1)
    for &s in &samples {
        assert!((0.0..1.0).contains(&s), "Sample {s} out of [0, 1)");
    }

    // Should be roughly sorted (each sample falls in its own stratum [i/n, (i+1)/n))
    let n_f = usize_to_f64(n, "n")?;
    for (i, &sample) in samples.iter().enumerate() {
        let i_f = usize_to_f64(i, "stratum index")?;
        let lo = i_f / n_f;
        let hi = (i_f + 1.0) / n_f;
        assert!(
            sample >= lo && sample < hi,
            "Sample {i} = {sample} not in [{lo}, {hi})",
        );
    }

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 5: Latin hypercube
// ---------------------------------------------------------------------------

fn part5_latin_hypercube() -> Result<(), SciOxideError> {
    println!("=== Part 5: Latin hypercube ===");
    println!();

    let mut rng = SplitMix64::from_seed(42);

    let n = 8;
    let dim = 3;
    let lhs: Vec<f64> = latin_hypercube(n, dim, &mut rng)?;

    println!("  Latin hypercube (n={n}, dim={dim}):");
    for (i, pt) in lhs.chunks_exact(dim).enumerate() {
        println!("    [{i}] ({:.4}, {:.4}, {:.4})", pt[0], pt[1], pt[2]);
    }

    assert_eq!(lhs.len(), n * dim, "Must return n points");

    // All coordinates in [0, 1)
    for pt in lhs.chunks_exact(dim) {
        assert_eq!(pt.len(), dim, "Each point must have {dim} dimensions");
        for &coord in pt {
            assert!(
                (0.0..1.0).contains(&coord),
                "Coordinate {coord} out of [0, 1)"
            );
        }
    }

    // In each dimension, each stratum [i/n, (i+1)/n) should contain exactly one sample
    for d in 0..dim {
        let mut strata = vec![false; n];
        let n_f = usize_to_f64(n, "n")?;
        let boundaries: Vec<f64> = (1..=n)
            .map(|idx| usize_to_f64(idx, "stratum boundary"))
            .collect::<Result<Vec<_>, SciOxideError>>()?
            .into_iter()
            .map(|idx_f| idx_f / n_f)
            .collect();
        for pt in lhs.chunks_exact(dim) {
            let stratum = boundaries.partition_point(|&upper| pt[d] >= upper);
            let stratum = stratum.min(n - 1);
            strata[stratum] = true;
        }
        assert!(
            strata.iter().all(|&s| s),
            "Dimension {d}: not all strata filled"
        );
    }

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
    println!("scioxide-random Sampling Examples");
    println!("================================");
    println!();

    part1_shuffle_choose()?;
    part2_weighted()?;
    part3_sample_indices()?;
    part4_stratified()?;
    part5_latin_hypercube()?;

    println!("All examples completed successfully.");
    Ok(())
}
