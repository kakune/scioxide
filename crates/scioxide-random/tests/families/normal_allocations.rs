//! Steady-state allocation contracts for Normal sampling.

#![allow(clippy::unwrap_used)]

use scioxide_random::{Distribution, Normal, SeedableRng, SplitMix64};

#[test]
fn normal_f32_scalar_and_caller_buffer_execute() {
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    let mut scalar_rng = SplitMix64::from_seed(0x4e4f_524d);
    dist.sample(&mut scalar_rng).unwrap();

    let mut output = vec![0.0_f32; 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(output.len()))
        .unwrap();
    let mut bulk_rng = SplitMix64::from_seed(0x4e4f_524d);
    {
        dist.sample_n_into(&mut bulk_rng, batch, &mut output, &mut workspace)
            .unwrap();
    };
}

#[test]
fn normal_f64_scalar_and_caller_buffer_execute() {
    let dist = Normal::new(0.0_f64, 1.0).unwrap();
    let mut scalar_rng = SplitMix64::from_seed(0x4e4f_524d);
    dist.sample(&mut scalar_rng).unwrap();

    let mut output = vec![0.0_f64; 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(output.len()))
        .unwrap();
    let mut bulk_rng = SplitMix64::from_seed(0x4e4f_524d);
    {
        dist.sample_n_into(&mut bulk_rng, batch, &mut output, &mut workspace)
            .unwrap();
    };
}
