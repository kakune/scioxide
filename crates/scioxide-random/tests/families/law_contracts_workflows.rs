//! law contracts workflows and statistical contracts.
#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
use scioxide_random::{Cdf, Quantile, Uniform};

#[test]
fn test_uniform_quantile_roundtrip() {
    let dist = Uniform::new(2.0_f64, 8.0).unwrap();
    for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!(
            (p - p_back).abs() < 1e-10,
            "quantile→CDF roundtrip failed at p={p}"
        );
    }
}
