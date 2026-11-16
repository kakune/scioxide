//! Explicit noisy values/gradients, fixed minibatches and bounded adaptive replication.
use scioxide_core::{Budget, GradientTolerance, SciOxideError};
use scioxide_optimize::{
    ReplicationOptions, SampledGradient, SampledObjective, SgdOptions, SgdStep,
    StochasticGradientDescent, StochasticProblem, StochasticTrustOptions, StochasticTrustRegion,
};
struct Quadratic;
impl SampledObjective<f64> for Quadratic {
    fn dimension(&self) -> usize {
        1
    }
    fn objectives(&self) -> usize {
        1
    }
    fn sample_into(&self, x: &[f64], sample: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
        let noise = scioxide_random::decode_u64::<f64>(sample) - 0.5;
        out[0] = 0.5 * (x[0] - 2.).powi(2) + 0.01 * noise;
        Ok(())
    }
}
impl SampledGradient<f64> for Quadratic {
    fn gradient_sample_into(
        &self,
        x: &[f64],
        sample: u64,
        out: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] - 2. + 0.02 * (scioxide_random::decode_u64::<f64>(sample) - 0.5);
        Ok(())
    }
}
fn main() -> Result<(), SciOxideError> {
    let iterations = Budget::try_from(200)?;
    let tolerance = GradientTolerance::new(0.01, 0.)?;
    let sgd = StochasticGradientDescent::new(
        SgdOptions::new(
            SgdStep::Diminishing {
                initial: 0.5,
                decay: 0.01,
                power: 1.,
            },
            iterations,
            Budget::try_from(16)?,
        )
        .with_tolerance(tolerance),
    )?;
    let trust = StochasticTrustRegion::new(
        StochasticTrustOptions::new(
            iterations,
            ReplicationOptions::adaptive(Budget::try_from(8)?, Budget::try_from(128)?)?,
        )
        .with_tolerance(tolerance),
    )?;
    let problem = StochasticProblem::new(&Quadratic);
    for (name, result) in [
        ("SGD", sgd.minimize(&problem, &[0.])?),
        ("trust", trust.minimize(&problem, &[0.])?),
    ] {
        println!(
            "{name}: x={:?}, mean={:?}, standard error={:?}, stop={}, samples={}",
            result.x(),
            result.value(),
            result.standard_error(),
            result.termination(),
            result.sample_cursor()
        );
    }
    Ok(())
}
