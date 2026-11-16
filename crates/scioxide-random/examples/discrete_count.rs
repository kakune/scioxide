//! Canonical count laws, prepared samplers, batches and signed support.
use scioxide_core::SciOxideError;
use scioxide_random::{
    BetaNegativeBinomial, Binomial, CountDistribution, LogSurvival, OutputPolicy, PoissonBinomial,
    Quantile, SamplingPolicy, SamplingRequest, SeedableRng, SignedCountSupport, Skellam,
    SplitMix64,
};
fn main() -> Result<(), SciOxideError> {
    let mut rng = SplitMix64::from_seed(532);
    let bernoulli = Binomial::new(1, 0.25_f64)?;
    let success = bernoulli.sample(&mut rng)?;
    let count = PoissonBinomial::new(&[0.1_f64, 0.5, 0.9])?;
    let failures = BetaNegativeBinomial::new(2.0_f64, 3.0, 2.0)?;
    let difference = Skellam::new(3.0_f64, 2.0)?;
    println!(
        "Bernoulli: {success}; count median: {}",
        count.quantile(0.5)?
    );
    println!("log P(failures > 40): {}", failures.log_sf(40)?);
    // Batches reuse one reserved workspace and publish atomically.
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let mut workspace = policy.workspace(SamplingRequest::scalars(4))?;
    let mut counts = [0; 4];
    count.sample_n_into(&mut rng, policy, &mut counts, &mut workspace)?;
    println!("Poisson-binomial batch: {counts:?}");
    println!(
        "signed support: {:?}; draw: {}",
        difference.signed_count_support(),
        difference.sample(&mut rng)?
    );
    Ok(())
}
