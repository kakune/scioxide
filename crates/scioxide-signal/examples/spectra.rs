//! Explicit spectral units and irregular-sampling analysis.
use scioxide_core::SciOxideError;
use scioxide_signal::{lomb_scargle, welch};
fn main() -> Result<(), SciOxideError> {
    let sample_rate = 128.;
    let samples: Vec<_> = (0..512)
        .map(|i| (2. * std::f64::consts::PI * 12. * i as f64 / sample_rate).sin())
        .collect();
    // Segments of 64 samples: bin k lies at k·128/64 Hz.
    let density = welch(&samples, sample_rate, 64)?;
    println!(
        "Welch bin 6: {} Hz, {} sample²/Hz",
        6. * sample_rate / 64.,
        density[6]
    );
    let times: Vec<_> = (0..50)
        .map(|i| i as f64 * 0.17 + (i % 3) as f64 * 0.03)
        .collect();
    let values: Vec<_> = times.iter().map(|t| 2. + (1.3 * t).sin()).collect();
    let frequencies = [0.7, 1.3, 2.1];
    println!(
        "Lomb–Scargle explained fractions: {:?}",
        lomb_scargle(&times, &values, &frequencies)?
    );
    Ok(())
}
