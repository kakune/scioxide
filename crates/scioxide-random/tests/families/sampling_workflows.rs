//! sampling workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn shuffle_keeps_the_multiset_and_draws_one_index_per_position() {
    let mut data: Vec<i32> = (0..100).map(|i| i % 7).collect();
    let mut rng = SplitMix64::from_seed(42);
    shuffle(&mut data, &mut rng).unwrap();
    let mut sorted = data.clone();
    sorted.sort_unstable();
    let mut expected: Vec<i32> = (0..100).map(|i| i % 7).collect();
    expected.sort_unstable();
    assert_eq!(sorted, expected, "shuffle must preserve all elements");
    // n − 1 draws of uniform_index, from upper n down to 2.
    let mut reference = SplitMix64::from_seed(42);
    for upper in (2..=100).rev() {
        uniform_index(upper, &mut reference).unwrap();
    }
    assert_eq!(rng.position(), reference.position());
    for len in [0, 1] {
        let mut short: Vec<i32> = (0..len).collect();
        let mut rng = SplitMix64::from_seed(42);
        shuffle(&mut short, &mut rng).unwrap();
        assert_eq!(rng.position(), Some(0));
    }
}

#[test]
fn shuffle_in_place_permutes_like_shuffle_under_every_policy() {
    let mut expected: Vec<u16> = (0..64).collect();
    shuffle(&mut expected, &mut SplitMix64::from_seed(2469)).unwrap();
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        let mut workspace = SamplingWorkspace::new();
        let mut data: Vec<u16> = (0..64).collect();
        let policy = SamplingPolicy::advance_on_error(output);
        shuffle_in_place(
            &mut data,
            &mut SplitMix64::from_seed(2469),
            policy,
            &mut workspace,
        )
        .unwrap();
        assert_eq!(data, expected);
        let mut data: Vec<u16> = (0..64).collect();
        let policy = SamplingPolicy::rollback(output);
        shuffle_in_place(
            &mut data,
            &mut SplitMix64::from_seed(2469),
            policy,
            &mut workspace,
        )
        .unwrap();
        assert_eq!(data, expected);
    }
}

#[test]
fn test_sample_indices_distinct() {
    let mut rng = SplitMix64::from_seed(42);
    let indices = sample_indices(1000, 50, &mut rng).unwrap();
    assert_eq!(indices.len(), 50);

    let mut sorted = indices;
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 50, "all indices must be distinct");
}

/// Lehmer-ranked permutations check positional dependence as well as marginals.
/// Seed 1335, 32768 draws; shared Pearson/DKW significance and power apply.
#[test]
fn shuffle_is_uniform_over_all_permutations() {
    let mut rng = SplitMix64::from_seed(1335);
    let mut counts = [0; 120];
    for _ in 0..statistics::DRAWS {
        let mut permutation = [0, 1, 2, 3, 4];
        shuffle(&mut permutation, &mut rng).unwrap();
        let rank = statistics::permutation_rank(&permutation);
        counts[rank] += 1;
    }
    statistics::assert_fit("shuffle/5", &counts, &[1.0 / 120.0; 120], 0.0);
}

/// `sample_indices(n, n)` is a uniformly random permutation of `0..n`, in order.
/// Seed 2469, 32768 draws; shared Pearson/DKW significance and power apply.
#[test]
fn full_index_samples_are_uniform_permutations() {
    let mut rng = SplitMix64::from_seed(2469);
    let mut counts = [0; 120];
    for _ in 0..statistics::DRAWS {
        let permutation = sample_indices(5, 5, &mut rng).unwrap();
        counts[statistics::permutation_rank(&permutation)] += 1;
    }
    statistics::assert_fit("sample_indices/5/5", &counts, &[1.0 / 120.0; 120], 0.0);
    for n in [1, 17, 1_000] {
        let mut all = sample_indices(n, n, &mut rng).unwrap();
        all.sort_unstable();
        assert!(all.into_iter().eq(0..n));
    }
}

/// Floyd promises uniformly distributed unordered subsets, not random order.
/// Enumerating all subsets exercises both collision and noncollision choices.
#[test]
fn floyd_is_uniform_over_subsets() {
    for k in [1, 2, 4, 5] {
        let subsets: Vec<u32> = (0_u32..64)
            .filter(|mask| mask.count_ones() == k as u32)
            .collect();
        let mut counts = vec![0; subsets.len()];
        let mut rng = SplitMix64::from_seed(1335);
        for _ in 0..statistics::DRAWS {
            let indices = sample_indices(6, k, &mut rng).unwrap();
            let mask = indices.iter().fold(0_u32, |mask, &i| mask | (1 << i));
            assert_eq!(mask.count_ones(), k as u32);
            counts[subsets.binary_search(&mask).unwrap()] += 1;
        }
        let probabilities = vec![1.0 / subsets.len() as f64; subsets.len()];
        statistics::assert_fit(&format!("Floyd/6/{k}"), &counts, &probabilities, 0.0);
    }
    let mut rng = SplitMix64::from_seed(1335);
    assert!(sample_indices(6, 0, &mut rng).unwrap().is_empty());
    let mut all = sample_indices(6, 6, &mut rng).unwrap();
    all.sort_unstable();
    assert_eq!(all, [0, 1, 2, 3, 4, 5]);
}

use crate::statistics;
