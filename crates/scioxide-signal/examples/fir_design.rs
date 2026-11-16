//! FIR design and derivative smoothing through canonical signal APIs.
use scioxide_core::SciOxideError;
use scioxide_signal::{
    FirBand, FirSymmetry, RemezOptions, SavitzkyGolayBoundary, SavitzkyGolayOptions,
    SavitzkyGolayPlan, convolve, design_fir_remez_with_options,
};
fn main() -> Result<(), SciOxideError> {
    let bands = [
        FirBand::new(0., 0.3, 1., 1.),
        FirBand::new(0.45, 1., 0., 0.).with_weight(8.),
    ];
    let design =
        design_fir_remez_with_options(31, FirSymmetry::Even, &bands, RemezOptions::default())?;
    println!(
        "FIR {:?} (success {}), iterations {}, grid ripple {:?}",
        design.termination(),
        design.is_success(),
        design.iterations(),
        design.weighted_ripple()
    );
    let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).powi(2)).collect();
    let filtered = convolve(&input, design.coefficients())?;
    println!("Full filtered samples: {}", filtered.len());
    let derivative = SavitzkyGolayPlan::new(
        7,
        2,
        SavitzkyGolayOptions::new()
            .with_derivative(1)
            .with_spacing(0.1),
    )?;
    let output = derivative.filter(&input, SavitzkyGolayBoundary::Polynomial)?;
    println!(
        "Derivative at the endpoints: {} and {}",
        output[0], output[31]
    );
    Ok(())
}
