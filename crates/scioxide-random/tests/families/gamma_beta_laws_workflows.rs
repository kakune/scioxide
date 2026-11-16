//! gamma beta laws workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_beta_underflow_has_no_midpoint_mass_f64() {
    let dist = Beta::new(0.0005_f64, 0.0005).unwrap();
    let mut rng = SplitMix64::from_seed(42);
    let samples = dist.sample_n(&mut rng, 10_000).unwrap();

    assert!(samples.iter().all(|sample| (0.0..=1.0).contains(sample)));
    assert!(
        samples.iter().any(|&sample| sample == 0.0 || sample == 1.0),
        "extreme f64 shapes should exercise underflow-prone boundary samples"
    );
    assert_eq!(
        samples.iter().filter(|&&sample| sample == 0.5).count(),
        0,
        "degenerate Gamma pairs must not create an f64 point mass at 0.5"
    );
}

#[test]
fn test_beta_underflow_has_no_midpoint_mass_f32() {
    let dist = Beta::new(0.005_f32, 0.005).unwrap();
    let mut rng = SplitMix64::from_seed(42);
    let samples = dist.sample_n(&mut rng, 10_000).unwrap();

    assert!(samples.iter().all(|sample| (0.0..=1.0).contains(sample)));
    assert!(
        samples.iter().any(|&sample| sample == 0.0 || sample == 1.0),
        "extreme f32 shapes should exercise underflow-prone boundary samples"
    );
    assert_eq!(
        samples.iter().filter(|&&sample| sample == 0.5).count(),
        0,
        "degenerate Gamma pairs must not create an f32 point mass at 0.5"
    );
}
