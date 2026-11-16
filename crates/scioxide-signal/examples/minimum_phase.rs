//! Factor an autocorrelation FIR and reuse its taps with the shared FIR plan.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_signal::{MinimumPhaseOptions, MinimumPhasePlan, PolyphasePlan};
fn main() -> Result<(), SciOxideError> {
    // Causal shift of the autocorrelation of [1, -0.5].
    let input = [-0.5_f64, 1.25, -0.5];
    let plan = MinimumPhasePlan::new(input.len(), MinimumPhaseOptions::default())?;
    let mut workspace = plan.workspace(())?;
    let mut taps = vec![0.; plan.output_len()];
    let report = plan.reconstruct_into(
        &input,
        &mut taps,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let filter = PolyphasePlan::fir(&taps)?;
    println!(
        "minimum-phase taps: {:?}; grid error: {}",
        filter.coefficients(),
        report.magnitude_error()
    );
    Ok(())
}
