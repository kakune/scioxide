//! Reconstruct samples and smooth missing observations with reusable workspaces.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{
    SignalExecutionOptions, SplineDegree, SplineFilter, SplineRequest, WhittakerHendersonPlan,
};
fn main() -> Result<(), SciOxideError> {
    let samples = [0., 1.1, 1.8, 8., 4.1, 5.];
    let spline = SplineFilter::<f64>::new(SplineDegree::Cubic, SignalExecutionOptions::new())?;
    let coefficients = spline.coefficients(&samples)?;
    let restored = spline.reconstruct(&coefficients)?;
    println!("reconstructed nodes: {restored:?}");
    let queries = [0.5, 1.5, 2.5];
    let mut values = [0.; 3];
    let mut scratch = spline.workspace(SplineRequest::Evaluate(queries.len()))?;
    spline.evaluate_into(
        &coefficients,
        &queries,
        0.,
        1.,
        &mut values,
        &mut scratch,
        ExecutionContext::sequential_native(),
    )?;
    println!("half-grid values: {values:?}");
    let weights = [1., 1., 1., 0., 1., 1.];
    let smoother = WhittakerHendersonPlan::<f64>::new(
        samples.len(),
        2,
        2.,
        Some(&weights),
        SignalExecutionOptions::new(),
    )?;
    let mut workspace = smoother.workspace(1)?;
    let mut smoothed = [0.; 6];
    let diagnostics = smoother.smooth_into(
        &samples,
        &mut smoothed,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("smoothed samples (fourth observation ignored): {smoothed:?}");
    println!("diagnostics: {diagnostics:?}");
    Ok(())
}
