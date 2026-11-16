//! public api workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_full_workflow_splitmix_normal() {
    let mut rng = SplitMix64::from_seed(42);
    let dist = Normal::new(0.0_f64, 1.0).unwrap();

    let n = 5_000_usize;
    let mut samples = vec![0.0_f64; n];
    let mut workspace = SamplingWorkspace::new();
    dist.sample_n_into(
        &mut rng,
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        &mut samples,
        &mut workspace,
    )
    .unwrap();
    let sum = samples.iter().sum::<f64>();
    let mean = sum / n as f64;
    assert!(
        mean.abs() < 0.1,
        "standard normal sample mean should be near 0, got {mean}"
    );
}

#[test]
fn test_full_workflow_xoshiro_uniform() {
    let mut rng = Xoshiro256StarStar::from_seed(42);
    let dist = Uniform::new(2.0_f64, 5.0).unwrap();

    let n = 5_000;
    let mut sum = 0.0_f64;
    for _ in 0..n {
        let x = dist.sample(&mut rng).unwrap();
        assert!((2.0..5.0).contains(&x));
        sum += x;
    }
    let mean = sum / f64::from(n);
    assert!(
        (mean - 3.5).abs() < 0.2,
        "Uniform(2,5) mean should be ~3.5, got {mean}"
    );
}

#[test]
fn test_full_workflow_pcg_exponential() {
    let mut rng = Pcg64::from_seed(42);
    let dist = Exponential::new(2.0_f64).unwrap();

    let n = 5_000;
    let mut sum = 0.0_f64;
    for _ in 0..n {
        let x = dist.sample(&mut rng).unwrap();
        assert!(x >= 0.0);
        sum += x;
    }
    let mean = sum / f64::from(n);
    // Expected mean = 1/lambda = 0.5
    assert!(
        (mean - 0.5).abs() < 0.1,
        "Exp(2) mean should be ~0.5, got {mean}"
    );
}

#[test]
fn test_reproducibility_distribution() {
    let dist = Normal::new(5.0_f64, 2.0).unwrap();
    let values_a: Vec<f64> = {
        let mut rng = SplitMix64::from_seed(99);
        (0..50).map(|_| dist.sample(&mut rng).unwrap()).collect()
    };
    let values_b: Vec<f64> = {
        let mut rng = SplitMix64::from_seed(99);
        (0..50).map(|_| dist.sample(&mut rng).unwrap()).collect()
    };
    assert_eq!(values_a, values_b);
}
