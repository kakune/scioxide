//! Reconstruct a short signal using a periodic Hann window and explicit hop.
use scioxide_core::SciOxideError;
use scioxide_signal::{
    StftOptions, StftPacking, StftPlan, WindowKind, WindowNormalization, WindowSymmetry, window,
};
fn main() -> Result<(), SciOxideError> {
    let window = window::<f64>(
        WindowKind::Hann,
        64,
        WindowSymmetry::Periodic,
        WindowNormalization::Native,
    )?;
    let options = StftOptions::new()
        .with_packing(StftPacking::OneSided)
        .with_sample_rate(8000.);
    let plan = StftPlan::new(&window, 16, options)?;
    let input: Vec<_> = (0..1000)
        .map(|i| (2. * std::f64::consts::PI * 440. * i as f64 / 8000.).sin())
        .collect();
    let coefficients = plan.analyze(&input)?;
    let output = plan.synthesize(&coefficients, input.len())?;
    let error = input
        .iter()
        .zip(&output)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    println!(
        "{} frames, {} bins, maximum reconstruction error {error:e}",
        plan.frames(input.len())?.count(),
        plan.bins()
    );
    Ok(())
}
