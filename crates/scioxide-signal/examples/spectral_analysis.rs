//! Convolution, correlation, discrete spectra and analytic envelopes.
//!
//! 1. Linear and circular convolution, allocating and through a prepared plan.
//! 2. Correlation in ascending lag order.
//! 3. Discrete power and cross spectra, including the `X * conj(Y)` orientation.
//! 4. The analytic signal and the envelope of an amplitude-modulated carrier.
//!
//! Run: `cargo run -p scioxide-signal --example spectral_analysis`
use core::f64::consts::TAU;
use scioxide_core::{NdView, SciOxideError, StridedLayout};
use scioxide_signal::{
    ConvolutionNdPlan, ConvolutionOptions, convolve, convolve_circular, correlate, cross_spectrum,
    envelope, hilbert, power_spectrum,
};

/// Absolute tolerance for FFT-rounded results of order one.
const TOLERANCE: f64 = 1e-10;

fn close(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() < TOLERANCE)
}

/// Samples `f(2π k / n)` for `k` in `0..n`.
fn sampled(n: u32, f: impl Fn(f64) -> f64) -> Vec<f64> {
    (0..n)
        .map(|k| f(TAU * f64::from(k) / f64::from(n)))
        .collect()
}

fn convolution() -> Result<(), SciOxideError> {
    let boxcar = [1.0_f64; 3];
    let linear = convolve(&boxcar, &boxcar)?;
    assert!(close(&linear, &[1.0, 2.0, 3.0, 2.0, 1.0]));
    // A delayed impulse rotates the record by one sample.
    let circular = convolve_circular(&[1.0_f64, 2.0, 3.0, 4.0], &[0.0, 1.0, 0.0, 0.0])?;
    assert!(close(&circular, &[4.0, 1.0, 2.0, 3.0]));
    println!("linear {linear:?}, circular {circular:?}");

    // A prepared plan fixes the record shape and kernel for repeated calls.
    let signal = sampled(64, f64::sin);
    let kernel: Vec<f64> = (0..16).map(|i| (-0.25 * f64::from(i)).exp()).collect();
    let plan = ConvolutionNdPlan::new(
        &[signal.len()],
        NdView::new(&kernel, StridedLayout::new(&[kernel.len()], &[1])?)?,
        &[0],
        ConvolutionOptions::default(),
    )?;
    let prepared = plan.convolve(NdView::new(
        &signal,
        StridedLayout::new(&[signal.len()], &[1])?,
    )?)?;
    assert!(close(&prepared, &convolve(&signal, &kernel)?));
    println!("prepared convolution: {} samples", prepared.len());
    Ok(())
}

fn correlation() -> Result<(), SciOxideError> {
    // Full lags ascend from -(b.len() - 1) to a.len() - 1: here -2, -1, 0, 1.
    let ordered = correlate(&[1.0_f64, 2.0], &[3.0, 4.0, 5.0])?;
    assert!(close(&ordered, &[5.0, 14.0, 11.0, 6.0]));
    // An autocorrelation peaks at zero lag, the central sample.
    let pulse: Vec<f64> = (0..16).map(|k| if k < 4 { 1.0 } else { 0.0 }).collect();
    let auto = correlate(&pulse, &pulse)?;
    let peak = auto.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((auto[pulse.len() - 1] - peak).abs() < TOLERANCE);
    println!("lags -2..=1: {ordered:?}, zero-lag autocorrelation {peak}");
    Ok(())
}

fn spectra() -> Result<(), SciOxideError> {
    // `|DFT(x)|² / n`: a unit sine on bin 4 of 64 puts n/4 in bins 4 and 60.
    let sine = sampled(64, |t| (4.0 * t).sin());
    let power = power_spectrum(&sine)?;
    for (bin, &p) in power.iter().enumerate() {
        let expected = if bin == 4 || bin == 60 { 16.0 } else { 0.0 };
        assert!((p - expected).abs() < TOLERANCE);
    }
    // Parseval: the discrete power spectrum sums to the record energy.
    let energy: f64 = sine.iter().map(|x| x * x).sum();
    assert!((power.iter().sum::<f64>() - energy).abs() < TOLERANCE);

    // `DFT(a) * conj(DFT(b)) / n`: a cosine leads the sine by a quarter period.
    let cosine = sampled(64, |t| (4.0 * t).cos());
    let cross = cross_spectrum(&cosine, &sine)?;
    assert!(cross[4].re.abs() < TOLERANCE && (cross[4].im - 16.0).abs() < TOLERANCE);
    println!(
        "power[4] = {:.3}, cross[4] = {:.3} + {:.3}i",
        power[4],
        cross[4].re.abs(),
        cross[4].im
    );
    Ok(())
}

fn analytic() -> Result<(), SciOxideError> {
    // The analytic signal keeps the real record and adds its quadrature.
    let cosine = sampled(64, |t| (3.0 * t).cos());
    let analytic = hilbert(&cosine)?;
    let quadrature = sampled(64, |t| (3.0 * t).sin());
    for ((z, &c), &s) in analytic.iter().zip(&cosine).zip(&quadrature) {
        assert!((z.re - c).abs() < TOLERANCE && (z.im - s).abs() < TOLERANCE);
    }

    // A carrier on bin 8 modulated on bin 1 has envelope `1 + cos(t) / 2`.
    let modulation = |t: f64| 1.0 + 0.5 * t.cos();
    let modulated = sampled(64, |t| modulation(t) * (8.0 * t).cos());
    let detected = envelope(&modulated)?;
    assert!(close(&detected, &sampled(64, modulation)));
    println!(
        "envelope range: [{:.3}, {:.3}]",
        detected.iter().copied().fold(f64::INFINITY, f64::min),
        detected.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );
    Ok(())
}

fn main() -> Result<(), SciOxideError> {
    convolution()?;
    correlation()?;
    spectra()?;
    analytic()?;
    println!("All spectral-analysis checks passed.");
    Ok(())
}
