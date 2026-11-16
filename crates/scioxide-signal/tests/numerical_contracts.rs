//! Independent finite-sum definitions, not FFT round trips.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{Complex, ExecutionContext, SciOxideErrorKind};

use scioxide_linalg::Matrix;
use scioxide_signal::{
    ConvolutionNdPlan, ConvolutionOptions, SignalLayout, SignalWorkspace, SpectralEstimatePlan,
    convolve2d, hilbert,
};

fn dft(input: &[f64]) -> Vec<Complex<f64>> {
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let phase = -std::f64::consts::TAU * (j * k) as f64 / input.len() as f64;
                    Complex::new(phase.cos() * x, phase.sin() * x)
                })
                .sum()
        })
        .collect()
}

macro_rules! qualify {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            for n in [1, 2, 7, 16, 127] {
                let a: Vec<$scalar> = (0..n)
                    .map(|i| ((i as f64 * 0.71).cos() + 0.25) as $scalar)
                    .collect();
                let b: Vec<$scalar> = (0..n)
                    .map(|i| ((i as f64 * 0.31).sin() - 0.5) as $scalar)
                    .collect();
                let wide_a: Vec<f64> = a.iter().map(|&x| x as f64).collect();
                let wide_b: Vec<f64> = b.iter().map(|&x| x as f64).collect();
                let fa = dft(&wide_a);
                let fb = dft(&wide_b);
                let tolerance = 128.0 * <$scalar>::EPSILON as f64 * n as f64;
                {
                    let plan = SpectralEstimatePlan::discrete(n).unwrap();
                    let mut scratch = plan.workspace(SignalLayout::Contiguous(n)).unwrap();
                    let mut power = vec![0.0; n];
                    let mut cross = vec![Complex::new(0.0, 0.0); n];
                    plan.power_into(
                        &a,
                        &mut power,
                        &mut scratch,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                    plan.csd_into(
                        &a,
                        &b,
                        &mut cross,
                        &mut scratch,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                    for i in 0..n {
                        let expected = fa[i] * fb[i].conj() / n as f64;
                        assert!((power[i] as f64 - fa[i].norm_sqr() / n as f64).abs() <= tolerance);
                        assert!((cross[i].re as f64 - expected.re).abs() <= tolerance);
                        assert!((cross[i].im as f64 - expected.im).abs() <= tolerance);
                    }
                    // Undersized workspaces are rejected before any output commit.
                    let mut short = SignalWorkspace::new();
                    power.fill(37.0);
                    assert_eq!(
                        error_kind(plan.power_into(
                            &a,
                            &mut power,
                            &mut short,
                            ExecutionContext::sequential_native()
                        )),
                        SciOxideErrorKind::DimensionMismatch
                    );
                    assert_eq!(power, vec![37.0; n]);
                    cross.fill(Complex::new(37.0, 19.0));
                    assert_eq!(
                        error_kind(plan.csd_into(
                            &a,
                            &b,
                            &mut cross,
                            &mut SignalWorkspace::new(),
                            ExecutionContext::sequential_native()
                        )),
                        SciOxideErrorKind::DimensionMismatch
                    );
                    assert_eq!(cross, vec![Complex::new(37.0, 19.0); n]);
                }
                // Independently apply the analytic-signal filter and inverse finite sum.
                let filtered: Vec<_> = fa
                    .iter()
                    .enumerate()
                    .map(|(k, &z)| {
                        let weight = if k == 0 || (n % 2 == 0 && k == n / 2) {
                            1.0
                        } else if k <= (n - 1) / 2 {
                            2.0
                        } else {
                            0.0
                        };
                        z * weight
                    })
                    .collect();
                let analytic = hilbert(&a).unwrap();
                for (j, actual) in analytic.iter().enumerate() {
                    let expected: Complex<f64> = filtered
                        .iter()
                        .enumerate()
                        .map(|(k, &z)| {
                            let phase = std::f64::consts::TAU * (j * k) as f64 / n as f64;
                            z * Complex::new(phase.cos(), phase.sin()) / n as f64
                        })
                        .sum();
                    assert!((actual.re as f64 - expected.re).abs() <= tolerance);
                    assert!((actual.im as f64 - expected.im).abs() <= tolerance);
                }
            }
            let a =
                Matrix::from_vec(2, 3, vec![1.0 as $scalar, -2.0, 3.0, 0.5, 2.0, -1.0]).unwrap();
            let b = Matrix::from_vec(2, 2, vec![0.5 as $scalar, 1.0, -1.0, 2.0]).unwrap();
            let view = a.view().transpose();
            let result = convolve2d(view, b.view()).unwrap();
            for r in 0..result.rows() {
                for c in 0..result.cols() {
                    let mut expected = 0.0;
                    for i in 0..view.rows() {
                        for j in 0..view.cols() {
                            if let (Some(k), Some(l)) = (r.checked_sub(i), c.checked_sub(j)) {
                                if k < b.rows() && l < b.cols() {
                                    expected += view.get(i, j).unwrap() as f64
                                        * b.view().get(k, l).unwrap() as f64;
                                }
                            }
                        }
                    }
                    assert!(
                        (result.view().get(r, c).unwrap() as f64 - expected).abs()
                            <= 128.0 * <$scalar>::EPSILON as f64
                    );
                }
            }
        }
    };
}
qualify!(f32_direct_definitions, f32);
qualify!(f64_direct_definitions, f64);

#[test]
fn frequency_scratch_failure_preserves_output_and_prepared_plan_is_reusable() {
    let input = vec![0.5_f64; 128];
    let kernel = vec![0.25; 127];
    let plan = ConvolutionNdPlan::new(
        &[128],
        scioxide_core::NdView::new(
            &kernel,
            scioxide_core::StridedLayout::new(&[127], &[1]).unwrap(),
        )
        .unwrap(),
        &[0],
        ConvolutionOptions::default(),
    )
    .unwrap();
    let mut output = vec![19.; 254];
    let mut short = scioxide_signal::SignalWorkspace::new();
    let mut scratch = plan
        .workspace(scioxide_signal::ConvolutionKernel::Prepared)
        .unwrap();
    assert_eq!(
        error_kind(
            plan.convolve_into(
                scioxide_core::NdView::new(
                    &input,
                    scioxide_core::StridedLayout::new(&[128], &[1]).unwrap()
                )
                .unwrap(),
                scioxide_core::NdViewMut::new(
                    &mut output,
                    scioxide_core::StridedLayout::new(&[254], &[1]).unwrap()
                )
                .unwrap(),
                &mut short,
                scioxide_core::ExecutionContext::sequential_native()
            )
        ),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, vec![19.; 254]);
    plan.convolve_into(
        scioxide_core::NdView::new(
            &input,
            scioxide_core::StridedLayout::new(&[128], &[1]).unwrap(),
        )
        .unwrap(),
        scioxide_core::NdViewMut::new(
            &mut output,
            scioxide_core::StridedLayout::new(&[254], &[1]).unwrap(),
        )
        .unwrap(),
        &mut scratch,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    for (i, &value) in output.iter().enumerate() {
        let terms = (0..128)
            .filter(|&j| i.checked_sub(j).is_some_and(|k| k < 127))
            .count();
        assert!((value - terms as f64 * 0.125).abs() < 1e-11);
    }
}

#[test]
fn cross_spectra_restore_scales_after_products_and_skip_unrequested_auto_power() {
    for (large, small) in [(1e300_f64, 1e-300), (1e300, f64::from_bits(2))] {
        let x = [large; 8];
        let y = [small; 8];
        let expected = scioxide_core::checked_float_product_factors(&[8., large, small]).unwrap();
        let actual = scioxide_signal::cross_spectrum(&x, &y).unwrap();
        assert!((actual[0].re - expected).abs() <= 32. * f64::EPSILON * expected.abs());
        assert_eq!(actual[0].im, 0.);
        assert!(actual[1..].iter().all(|v| v.norm() == 0.));
    }
}

#[test]
fn weighted_cross_spectra_defer_exceptional_scales_for_every_packing() {
    use scioxide_signal::{
        SpectralEstimateOptions, SpectralEstimatePlan, SpectralScaling, StftPacking,
    };
    for packing in [
        StftPacking::Full,
        StftPacking::Centered,
        StftPacking::OneSided,
    ] {
        for scaling in [SpectralScaling::Density, SpectralScaling::Spectrum] {
            let options = SpectralEstimateOptions::new()
                .with_fft_len(Some(4))
                .with_sample_rate(1.0)
                .with_detrend(None)
                .with_packing(packing)
                .with_scaling(scaling);
            let plan = SpectralEstimatePlan::<f64>::new(&[1.0, 2.0, 1.0, 2.0], options).unwrap();
            for small in [1e-300, f64::from_bits(2)] {
                let result = plan.csd(&[1e300; 4], &[small; 4]).unwrap();
                let (dc, nyquist) = if scaling == SpectralScaling::Density {
                    (3.6, 0.4)
                } else {
                    (1.0, 1.0 / 9.0)
                };
                let factors: &[f64] = match packing {
                    StftPacking::Full => &[dc, 0.0, nyquist, 0.0],
                    StftPacking::Centered => &[nyquist, 0.0, dc, 0.0],
                    StftPacking::OneSided => &[dc, 0.0, nyquist],
                };
                for (value, &factor) in result.iter().zip(factors) {
                    let expected =
                        scioxide_core::checked_float_product_factors(&[factor, 1e300, small])
                            .unwrap();
                    assert!((value.re - expected).abs() <= 32.0 * f64::EPSILON * expected.abs());
                    assert_eq!(value.im, 0.0);
                }
            }
        }
    }
}

#[test]
fn inverse_window_product_does_not_underflow_before_scaling() {
    let options = scioxide_signal::StftOptions::new()
        .with_fft_len(Some(1))
        .with_scaling(scioxide_signal::StftScaling::Psd);
    let plan = scioxide_signal::StftPlan::new(&[1e300_f64], 1, options).unwrap();
    let output = plan.synthesize(&[Complex::new(1e-70, 0.0)], 1).unwrap();
    assert!((output[0] / 1e-70 - 1.0).abs() <= 16.0 * f64::EPSILON);
}
