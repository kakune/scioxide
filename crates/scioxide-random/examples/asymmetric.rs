//! Standardized skew laws, shared affine transforms and caller-owned RNG state.
use scioxide_core::SciOxideError;
use scioxide_random::{
    Affine, Distribution, GeneralizedHyperbolic, LogSurvival, Mean, Quantile, SeedableRng,
    SkewNormal, SplitMix64,
};
fn main() -> Result<(), SciOxideError> {
    let skew = SkewNormal::new(4.0_f64)?;
    let shifted = Affine::new(&skew, 10., 2.)?;
    println!(
        "skew mean={}, shifted 99% quantile={}",
        skew.mean()?,
        shifted.quantile(0.99)?
    );
    let heavy = GeneralizedHyperbolic::new(-3.0_f64, 1., 1.)?;
    let mut rng = SplitMix64::from_seed(539);
    println!(
        "boundary log tail={}, sample={}",
        heavy.log_sf(8.)?,
        heavy.sample(&mut rng)?
    );
    Ok(())
}
