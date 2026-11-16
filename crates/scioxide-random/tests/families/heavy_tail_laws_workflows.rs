//! Location/scale Cauchy sampling at both widths, beyond the standard-law sweep.
#![allow(clippy::unwrap_used)]
use crate::statistics;
use scioxide_core::RealScalar;
use scioxide_random::*;

fn shifted_cauchy<S: RealScalar>() {
    let dist = Cauchy::new(S::try_from_f64_lossy(3.0).unwrap(), S::ONE).unwrap();
    let mut rng = SplitMix64::from_seed(42);
    statistics::continuous("shifted Cauchy", &dist, || dist.sample(&mut rng).unwrap());
}
#[test]
fn shifted_cauchy_f32() {
    shifted_cauchy::<f32>();
}
#[test]
fn shifted_cauchy_f64() {
    shifted_cauchy::<f64>();
}
