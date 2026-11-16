//! Differentiate the mean of `sin(θ)·noise` over samples, written once.
use scioxide_ad::{
    BatchFunction, BatchOptions, BatchOutputs, BatchPlan, BatchReduction, Real, batch_mean_and_grad,
};
use scioxide_core::{ExecutionContext, SciOxideError};

/// The sample `sin(θ)·z` for a passive draw `z`.
struct Sample;

impl BatchFunction for Sample {
    type Shared<R: Real> = R;
    fn prepare<R: Real>(&self, theta: &[R]) -> Result<R, SciOxideError> {
        Ok(theta[0].sin())
    }
    fn item<R: Real>(&self, shared: &R, _: &[R], noise: &[R::Base]) -> Result<R, SciOxideError> {
        Ok(*shared * R::passive(noise[0]))
    }
}

fn main() -> Result<(), SciOxideError> {
    let noise: Vec<f64> = (0..1000).map(|k| ((k as f64) * 0.37).cos() + 0.5).collect();
    let plan = BatchPlan::<8>::new(
        noise.len(),
        1,
        0,
        1,
        BatchReduction::Mean,
        BatchOptions::new(),
    )?;
    let mut workspace = plan.workspace(1)?;
    let mut gradient = [0.0];
    let outputs = BatchOutputs::new().with_parameter_gradient(&mut gradient);
    let estimate = plan.evaluate_into(
        &Sample,
        &[0.4],
        &[],
        &noise,
        outputs,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let mean = noise.iter().sum::<f64>() / noise.len() as f64;
    assert!((estimate.value() - 0.4_f64.sin() * mean).abs() < 1e-14);
    assert!((gradient[0] - 0.4_f64.cos() * mean).abs() < 1e-14);
    // The one-shot form runs the same plan configuration: the same bits.
    let (once, once_gradient) = batch_mean_and_grad(&Sample, &[0.4], &noise, noise.len())?;
    assert_eq!(once.value().to_bits(), estimate.value().to_bits());
    assert_eq!(once_gradient.as_slice()[0].to_bits(), gradient[0].to_bits());
    println!("mean {} gradient {}", estimate.value(), gradient[0]);
    Ok(())
}
