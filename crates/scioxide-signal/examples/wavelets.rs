//! Explicit packing and original-length reconstruction of an odd finite record.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{
    SignalBoundary, Wavelet2dLayout, Wavelet2dPlan, WaveletFamily, WaveletOptions, WaveletPlan,
};
fn main() -> Result<(), SciOxideError> {
    let plan = WaveletPlan::<f64>::new(
        WaveletFamily::Daubechies(4),
        7,
        1,
        WaveletOptions::default().with_boundary(SignalBoundary::Reflect),
    )?;
    let signal = [1., 2., 3., 4., 3., 2., 1.];
    let coefficients = plan.analyze(&signal)?;
    let mut recovered = [0.; 7];
    let mut workspace = plan.workspace(1)?;
    plan.synthesize_into(
        &coefficients,
        &mut recovered,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "approximation: {:?}",
        &coefficients[..plan.approximation_len()]
    );
    println!("levels: {:?}; reconstructed: {recovered:?}", plan.levels());
    let image = Wavelet2dPlan::<f64>::new(
        WaveletFamily::BiorthogonalSpline(202),
        [3, 5],
        [2, 2],
        Wavelet2dLayout::Pyramid,
        WaveletOptions::default(),
    )?;
    let pixels: Vec<_> = (0..15).map(|i| i as f64).collect();
    let packed = image.analyze(&pixels)?;
    println!(
        "subbands: {:?}; image: {:?}",
        image.subbands(),
        image.synthesize(&packed)?
    );
    Ok(())
}
