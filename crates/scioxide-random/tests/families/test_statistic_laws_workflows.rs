//! test statistic laws workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_student_t_symmetry() {
    let dist = StudentT::new(5.0_f64).unwrap();
    let mut rng = SplitMix64::from_seed(42);
    let n = 10_000;

    let mut sum = 0.0_f64;
    for _ in 0..n {
        sum += dist.sample(&mut rng).unwrap();
    }

    let mean = sum / n as f64;
    // Student-t with df=5: mean = 0
    assert!(
        mean.abs() < 0.2,
        "StudentT(5) mean should be ~0, got {mean}"
    );
}
