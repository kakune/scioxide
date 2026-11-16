//! Periodic Hann metrics and reusable multitaper preparation.
use scioxide_signal::{
    DpssOptions, DpssTapers, WindowKind, WindowNormalization, WindowSymmetry, window,
    window_metrics,
};
fn main() -> Result<(), scioxide_core::SciOxideError> {
    let hann = window::<f64>(
        WindowKind::Hann,
        128,
        WindowSymmetry::Periodic,
        WindowNormalization::Native,
    )?;
    let metrics = window_metrics(&hann)?;
    println!(
        "Hann gain: {}, ENBW: {} bins",
        metrics.coherent_gain(),
        metrics.equivalent_noise_bandwidth()
    );
    let tapers = DpssTapers::<f64>::new(128, 3.5, 4, DpssOptions::default())?;
    println!(
        "DPSS tapers: {} x {}, concentration: {:?}",
        tapers.taper_count(),
        tapers.length(),
        tapers.concentration_ratios()
    );
    Ok(())
}
