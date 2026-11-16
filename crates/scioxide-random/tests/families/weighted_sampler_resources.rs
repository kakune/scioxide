//! Resource contracts for weighted sampler.
#![allow(clippy::unwrap_used)]

use scioxide_random::{SeedableRng, SplitMix64, WeightedSampler};

#[test]
fn prepared_scalar_and_bulk_draws_execute() {
    let sampler = WeightedSampler::new(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut scalar_rng = SplitMix64::from_seed(31);
    {
        (0..1_024).for_each(|_| {
            let _ = sampler.sample_index(&mut scalar_rng).unwrap();
        });
    };

    let mut output = [usize::MAX; 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(output.len()))
        .unwrap();
    let mut bulk_rng = SplitMix64::from_seed(37);
    {
        sampler
            .sample_n_into(&mut bulk_rng, batch, &mut output, &mut workspace)
            .unwrap();
    };
    assert!(output.into_iter().all(|index| index < sampler.len()));
}
