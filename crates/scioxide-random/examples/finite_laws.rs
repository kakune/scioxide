//! Finite urns and explicit equivalent discrete-law parameterizations.
use scioxide_core::SciOxideError;
use scioxide_random::{
    BetaBinomial, Cdf, CountDistribution, DiscreteUniform, Geometric, LogSurvival, Mean,
    SeedableRng, SplitMix64, WalleniusNoncentralHypergeometric,
};
fn main() -> Result<(), SciOxideError> {
    let urn = WalleniusNoncentralHypergeometric::new(50, 20, 12, 2.0_f64)?;
    let mut rng = SplitMix64::from_seed(533);
    println!(
        "biased successes: {}; mean: {}",
        urn.sample(&mut rng)?,
        urn.mean()?
    );
    // Planck counts failures: shift the canonical geometric trial count by one.
    let geometric = Geometric::from_rate(1000.0_f64)?;
    println!(
        "Planck log P(X>0): {}; mean: {}",
        geometric.log_sf(1)?,
        geometric.mean_failures()?
    );
    // M=50, n=20 successes; stop after r=12 failures. beta=M-n-r+1=19.
    let stopping = BetaBinomial::new(20, 12.0_f64, 19.0)?;
    println!("stopping-urn P(X<=8): {}", stopping.cdf(8)?);
    let uniform = DiscreteUniform::<f64>::new(-10, 11)?;
    println!("signed uniform: {}", uniform.sample(&mut rng)?);
    Ok(())
}
