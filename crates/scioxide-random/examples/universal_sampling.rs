//! An unnormalized Gaussian density with explicit mathematical guarantees.
use scioxide_core::{ExtendedInterval, SciOxideError};
use scioxide_random::{
    AdaptiveRejectionSampler, ConcaveDensity, DensityTransform, LogDensity, OutputPolicy,
    SamplingPolicy, SamplingRequest, SeedableRng, SplitMix64, Support, UserDensity,
};
struct Gaussian;
impl Support<f64> for Gaussian {
    fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
        ExtendedInterval::new(None, None)
    }
}
impl LogDensity<f64> for Gaussian {
    fn log_pdf(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(-0.5 * x * x)
    }
}
impl UserDensity<f64> for Gaussian {
    fn validate_density(&self) -> Result<(), SciOxideError> {
        // The integral is sqrt(2*pi): positive and finite. No normalization is needed.
        Ok(())
    }
}
impl ConcaveDensity<f64> for Gaussian {
    fn log_density_derivative(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(-x)
    }
    fn validate_concavity(&self, _: DensityTransform) -> Result<(), SciOxideError> {
        // log(f)'' = -1, and (-1/sqrt(f))'' = -(1/2+x²/4)*exp(x²/4).
        Ok(())
    }
}
fn main() -> Result<(), SciOxideError> {
    let sampler = AdaptiveRejectionSampler::new(Gaussian, &[-1.0, 0.0, 1.0])?;
    let mut adaptive = sampler.workspace()?;
    let mut output = [0.0; 16];
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let mut staging = policy.workspace(SamplingRequest::scalars(output.len()))?;
    let mut rng = SplitMix64::from_seed(543);
    sampler.sample_adaptive_into(&mut rng, policy, &mut output, &mut adaptive, &mut staging)?;
    assert!(output.iter().all(|x| x.is_finite()));
    Ok(())
}
