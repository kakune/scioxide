//! engines state workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_seed_sequence_deterministic() {
    let a = seed_sequence(42, 10).unwrap();
    let b = seed_sequence(42, 10).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 10);
}

#[test]
fn test_seed_sequence_different_seeds() {
    let a = seed_sequence(1, 5).unwrap();
    let b = seed_sequence(2, 5).unwrap();
    assert_ne!(a, b);
}

fn assert_f32_float_intervals<R: Rng<f32>>(mut rng: R) {
    for _ in 0..1000 {
        let half_open = rng.next_float().unwrap();
        assert!((0.0..1.0).contains(&half_open));

        let open = rng.next_open_float().unwrap();
        assert!(open > 0.0 && open < 1.0);
    }
}

fn assert_f64_float_intervals<R: Rng<f64>>(mut rng: R) {
    for _ in 0..1000 {
        let half_open = rng.next_float().unwrap();
        assert!((0.0..1.0).contains(&half_open));

        let open = rng.next_open_float().unwrap();
        assert!(open > 0.0 && open < 1.0);
    }
}

#[test]
fn test_all_prng_float_adapters_honor_f32_endpoints() {
    assert_f32_float_intervals(SplitMix64::from_seed(42));
    assert_f32_float_intervals(Xoshiro256StarStar::from_seed(42));
    assert_f32_float_intervals(Xoshiro128StarStar::from_seed(42));
    assert_f32_float_intervals(Pcg64::from_seed(42));
    assert_f32_float_intervals(Lcg64::from_seed(42));
    assert_f32_float_intervals(Mrg32k3a::from_seed(42));
}

#[test]
fn test_all_prng_float_adapters_honor_f64_endpoints() {
    assert_f64_float_intervals(SplitMix64::from_seed(42));
    assert_f64_float_intervals(Xoshiro256StarStar::from_seed(42));
    assert_f64_float_intervals(Xoshiro128StarStar::from_seed(42));
    assert_f64_float_intervals(Pcg64::from_seed(42));
    assert_f64_float_intervals(Lcg64::from_seed(42));
    assert_f64_float_intervals(Mrg32k3a::from_seed(42));
}
