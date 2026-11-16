//! Canonical signal crate-root workflows.
use scioxide_core::{Complex, ExecutionContext, SciOxideError};

use scioxide_linalg::Matrix;
use scioxide_signal::{
    ConvolutionNdPlan, ConvolutionOptions, SignalLayout, SpectralEstimatePlan, convolve,
    convolve_circular, convolve2d, correlate, cross_spectrum, hilbert, power_spectrum,
};
#[test]
fn allocating_signal_operations() -> Result<(), SciOxideError> {
    let real = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
    assert_eq!(hilbert(real.as_slice())?.len(), 4);
    let signal = [1.0_f64, 2.0, 3.0];
    let kernel = [0.5_f64, -1.0];
    assert_eq!(convolve(&signal, &kernel)?.len(), 4);
    assert_eq!(correlate(&signal, &kernel)?.len(), 4);
    assert_eq!(power_spectrum(&signal)?.len(), signal.len());
    assert_eq!(cross_spectrum(&signal, &signal)?.len(), signal.len());
    assert_eq!(
        convolve_circular(&signal, &[0.5, -1.0, 0.25])?.len(),
        signal.len()
    );
    assert_eq!(
        convolve2d(real.view(), Matrix::from_vec(1, 1, vec![1.0_f64])?.view(),)?
            .as_slice()
            .len(),
        real.as_slice().len()
    );
    Ok(())
}
#[test]
fn prepared_signal_operations() -> Result<(), SciOxideError> {
    let input = [1.0_f64, -2.0, 0.5, 3.0];
    let spectral = SpectralEstimatePlan::discrete(input.len())?;
    let mut work = spectral.workspace(SignalLayout::Contiguous(input.len()))?;
    let mut power = vec![0.; 4];
    let mut cross = vec![Complex::new(0., 0.); 4];
    spectral.power_into(
        &input,
        &mut power,
        &mut work,
        ExecutionContext::sequential_native(),
    )?;
    spectral.csd_into(
        &input,
        &input,
        &mut cross,
        &mut work,
        ExecutionContext::sequential_native(),
    )?;
    let kernel = [0.5, -1.];
    let plan = ConvolutionNdPlan::new(
        &[4],
        scioxide_core::NdView::new(&kernel, scioxide_core::StridedLayout::new(&[2], &[1])?)?,
        &[0],
        ConvolutionOptions::default(),
    )?;
    let result = plan.convolve(scioxide_core::NdView::new(
        &input,
        scioxide_core::StridedLayout::new(&[4], &[1])?,
    )?)?;
    assert_eq!(result, convolve(&input, &kernel)?);
    Ok(())
}
