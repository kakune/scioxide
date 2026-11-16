//! designs workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::*;

#[test]
fn test_latin_hypercube_coverage() {
    let mut rng = SplitMix64::from_seed(42);
    let lhs = latin_hypercube::<f64, _>(20, 3, &mut rng).unwrap();
    assert_eq!(lhs.len(), 60);

    for p in lhs.chunks_exact(3) {
        assert_eq!(p.len(), 3);
        for &v in p {
            assert!((0.0..1.0).contains(&v));
        }
    }
}
