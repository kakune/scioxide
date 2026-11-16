//! Analytic inference without a probabilistic-programming runtime.
use scioxide_core::SciOxideError;
use scioxide_random::{InversionOptions, Mean};
use scioxide_stats::{
    BetaBinomialPrior, ConfidenceProbability, GammaPoissonPrior, NormalInverseGammaPrior,
    normal_credible_summary,
};
fn main() -> Result<(), SciOxideError> {
    let beta = BetaBinomialPrior::<f64>::new(1., 1.)?.update(7, 10)?;
    println!(
        "Binomial parameter posterior mean: {}",
        beta.posterior_law()?.mean()?
    );
    let gamma = GammaPoissonPrior::<f64>::new(2., 3.)?.update(5, 2.)?;
    println!(
        "Future count mean at exposure 1: {}",
        gamma.predictive(1.)?.mean()?
    );
    let prior = NormalInverseGammaPrior::<f64>::new(0., 1., 2., 1.)?;
    let posterior = prior.update_batch(&[1., 2., 3.])?;
    println!(
        "Normal posterior (location, precision, shape, scale): {:?}",
        posterior.parameters()?
    );
    let summary = normal_credible_summary(
        &[6., 9., 12., 7., 8., 8., 13.],
        ConfidenceProbability::new(0.9)?,
        &InversionOptions::default(),
    )?;
    println!(
        "Reference-posterior mean: {:?}; credible bounds: {:?}",
        summary.mean.mean,
        summary.mean.interval.bounds()
    );
    Ok(())
}
