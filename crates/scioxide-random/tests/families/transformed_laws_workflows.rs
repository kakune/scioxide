//! transformed laws workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_log_normal_positive() {
    let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
    let mut rng = SplitMix64::from_seed(42);

    for _ in 0..1000 {
        let x = dist.sample(&mut rng).unwrap();
        assert!(x > 0.0, "LogNormal must be positive");
    }
}
