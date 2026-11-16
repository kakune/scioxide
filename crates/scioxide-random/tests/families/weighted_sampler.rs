//! Alias-table shapes at both widths. Seed is fixed; shared alpha/power bounds
//! apply. A million draws for the 1e-4 category also detects missing rare mass.
#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_random::{
    BitGenerator, OutputPolicy, SamplingPolicy, SamplingWorkspace, SeedableRng, SplitMix64,
    WeightedSampler,
};

fn frequencies<S: RealScalar>() {
    for weights in [
        &[1.0, 1.0, 1.0, 1.0][..],
        &[1.0, 3.0, 6.0],
        &[0.0, 2.0, 0.0, 8.0],
        &[0.0, 3.0, 0.0, 2.0],
        &[1.0, 9999.0],
        &[1e30, 3e30, 6e30],
        &[1e-30, 3e-30, 6e-30],
    ] {
        let weights: Vec<_> = weights
            .iter()
            .map(|&w| S::try_from_f64_lossy(w).unwrap())
            .collect();
        let sampler = WeightedSampler::new(&weights).unwrap();
        let mut rng = SplitMix64::from_seed(0x4752_4144_4941_0248);
        let draws = if weights.len() == 2 {
            1_000_000
        } else {
            statistics::DRAWS
        };
        let mut counts = vec![0; weights.len()];
        for _ in 0..draws {
            counts[sampler.sample_index(&mut rng).unwrap()] += 1;
        }
        for (&weight, &count) in weights.iter().zip(&counts) {
            if weight == S::ZERO {
                assert_eq!(count, 0);
            }
        }
        let total: f64 = weights.iter().map(|w| w.into_f64()).sum();
        let probabilities: Vec<_> = weights.iter().map(|w| w.into_f64() / total).collect();
        statistics::assert_fit(
            &format!("WeightedSampler/{weights:?}"),
            &counts,
            &probabilities,
            8.0 * S::EPSILON.into_f64(),
        );
    }
}
#[test]
fn alias_shapes_f32() {
    frequencies::<f32>();
}
#[test]
fn alias_shapes_f64() {
    frequencies::<f64>();
}

#[test]
fn batches_equal_single_index_draws() {
    let sampler = WeightedSampler::new(&[0.5_f64, 0.0, 2.0, 1.5, 4.0]).unwrap();
    let mut rng = SplitMix64::from_seed(2469);
    let expected: Vec<_> = (0..257)
        .map(|_| sampler.sample_index(&mut rng).unwrap())
        .collect();
    let mut owned_rng = SplitMix64::from_seed(2469);
    assert_eq!(sampler.sample_n(&mut owned_rng, 257).unwrap(), expected);
    assert_eq!(owned_rng.position(), rng.position());
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        let mut indices = [usize::MAX; 257];
        let policy = SamplingPolicy::rollback(output);
        let mut workspace = SamplingWorkspace::new();
        let mut batch_rng = SplitMix64::from_seed(2469);
        sampler
            .sample_n_into(&mut batch_rng, policy, &mut indices, &mut workspace)
            .unwrap();
        assert_eq!(indices[..], expected[..]);
    }
    let mut empty_rng = SplitMix64::from_seed(2469);
    assert!(sampler.sample_n(&mut empty_rng, 0).unwrap().is_empty());
    assert_eq!(empty_rng.position(), Some(0));
    let values = ["a", "b", "c", "d", "e"];
    let mut value_rng = SplitMix64::from_seed(2469);
    assert_eq!(
        *sampler.sample(&mut value_rng, &values).unwrap(),
        values[expected[0]]
    );
}

use crate::statistics;
