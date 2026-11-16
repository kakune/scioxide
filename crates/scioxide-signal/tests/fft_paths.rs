//! Independent definitions and boundary cases for conditional FFT scaling.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, RealScalar, SciOxideErrorKind, StridedLayout,
};
use scioxide_signal::{
    AnalyticSignalNdPlan, ConvolutionKernel, ConvolutionMethod, ConvolutionNdPlan,
    ConvolutionOptions, FourierResamplePlan, OverlapPlan, RidgeOptions, RidgePeak, RidgePlan,
    SignalExecutionOptions, SignalWorkspace, StreamChunk,
};

fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

fn resample_definition(input: &[f64], m: usize) -> Vec<f64> {
    let n = input.len();
    let spectrum: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let phase = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                    Complex::new(phase.cos() * x, phase.sin() * x)
                })
                .sum()
        })
        .collect();
    let common = n.min(m);
    (0..m)
        .map(|j| {
            let mut value = spectrum[0].re;
            for (k, z) in spectrum.iter().enumerate().take(common.div_ceil(2)).skip(1) {
                let phase = std::f64::consts::TAU * (j * k) as f64 / m as f64;
                value += 2.0 * (z.re * phase.cos() - z.im * phase.sin());
            }
            if common.is_multiple_of(2) {
                let k = common / 2;
                let phase = std::f64::consts::TAU * (j * k) as f64 / m as f64;
                value += spectrum[k].re * phase.cos() * if n > m { 2.0 } else { 1.0 };
            }
            value / n as f64
        })
        .collect()
}

fn real_parities<S: RealScalar>() {
    for n in [1, 2, 3, 4, 7, 8, 15, 16] {
        let x: Vec<S> = (0..n)
            .map(|i| {
                S::try_from_f64_lossy((i as f64 * 0.71).cos() + (i as f64 * 0.29).sin()).unwrap()
            })
            .collect();
        let wide: Vec<_> = x.iter().map(|v| v.into_f64()).collect();
        for m in [1, 2, 3, 4, 7, 8, 15, 16] {
            let plan =
                FourierResamplePlan::<S>::new(&[n], 0, m, SignalExecutionOptions::new()).unwrap();
            let mut work = plan.workspace(()).unwrap();
            let mut out = vec![S::ZERO; m];
            plan.resample_into(
                NdView::new(&x, StridedLayout::new(&[n], &[1]).unwrap()).unwrap(),
                NdViewMut::new(&mut out, StridedLayout::new(&[m], &[1]).unwrap()).unwrap(),
                &mut work,
                native(),
            )
            .unwrap();
            let bound = 64.0 * S::EPSILON.into_f64() * (n.max(m) as f64).log2().max(1.0);
            for (actual, expected) in out.iter().zip(resample_definition(&wide, m)) {
                assert!(
                    (actual.into_f64() - expected).abs() <= bound,
                    "n={n}, m={m}: {} vs {expected}",
                    actual.into_f64()
                );
            }
        }
    }
}
#[test]
fn real_resampling_parities_against_finite_sums() {
    real_parities::<f32>();
    real_parities::<f64>();
}

#[test]
fn non_last_resampling_axis_and_padding_are_independent_batches() {
    let shape = [4, 3];
    let layout = StridedLayout::new(&shape, &[6, 2]).unwrap();
    let mut x = [f64::NAN; 24];
    for i in 0..4 {
        for j in 0..3 {
            x[i * 6 + j * 2] = (i * 3 + j) as f64;
        }
    }
    for m in [3, 4, 7, 8] {
        let plan = FourierResamplePlan::new(&shape, 0, m, SignalExecutionOptions::new()).unwrap();
        let mut work = plan.workspace(()).unwrap();
        let mut out = vec![0.0; m * 3];
        plan.resample_into(
            NdView::new(&x, layout).unwrap(),
            NdViewMut::new(&mut out, StridedLayout::new(&[m, 3], &[3, 1]).unwrap()).unwrap(),
            &mut work,
            native(),
        )
        .unwrap();
        for j in 0..3 {
            let lane: Vec<_> = (0..4).map(|i| x[i * 6 + j * 2]).collect();
            for (i, expected) in resample_definition(&lane, m).into_iter().enumerate() {
                assert!((out[i * 3 + j] - expected).abs() < 2e-12);
            }
        }
    }
}

#[test]
fn real_resampling_empty_batches_are_a_no_op() {
    let plan =
        FourierResamplePlan::<f64>::new(&[0, 4], 1, 7, SignalExecutionOptions::new()).unwrap();
    let mut work = plan.workspace(()).unwrap();
    plan.resample_into(
        NdView::new(&[], StridedLayout::new(&[0, 4], &[0, 0]).unwrap()).unwrap(),
        NdViewMut::new(&mut [], StridedLayout::new(&[0, 7], &[0, 0]).unwrap()).unwrap(),
        &mut work,
        native(),
    )
    .unwrap();
}

#[test]
fn convolution_reuses_one_plan_across_ordinary_and_exceptional_ranges() {
    for (x, taps) in [
        (
            [1e300, 1e-300, 0.0, -1e300, 0.0, 1e-30, 0.0, 0.0],
            [1.0, 0.5, 0.25],
        ),
        (
            [1e300, -1e300, 0.0, 1e-300, 0.0, 0.0, 0.0, 0.0],
            [1e-300, 0.5e-300, 0.25e-300],
        ),
        (
            [
                f64::from_bits(4),
                0.0,
                f64::from_bits(2),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.5, 1.0, 0.25],
        ),
    ] {
        let kernel = || NdView::new(&taps, StridedLayout::new(&[3], &[1]).unwrap()).unwrap();
        let plan = ConvolutionNdPlan::new(
            &[8],
            kernel(),
            &[0],
            ConvolutionOptions::new().with_method(ConvolutionMethod::Fft),
        )
        .unwrap();
        let direct = ConvolutionNdPlan::new(
            &[8],
            kernel(),
            &[0],
            ConvolutionOptions::new().with_method(ConvolutionMethod::Direct),
        )
        .unwrap();
        let mut work = plan.workspace(ConvolutionKernel::Prepared).unwrap();
        let mut out = [0.0; 10];
        for input in [[0.0; 8], [1.0; 8], x, [0.125; 8], [0.0; 8]] {
            let view = || NdView::new(&input, StridedLayout::new(&[8], &[1]).unwrap()).unwrap();
            let expected = direct.convolve(view()).unwrap();
            plan.convolve_into(
                view(),
                NdViewMut::new(&mut out, StridedLayout::new(&[10], &[1]).unwrap()).unwrap(),
                &mut work,
                native(),
            )
            .unwrap();
            let am = input.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let bm = taps.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let bound = scioxide_core::checked_float_product_factors(&[
                128.0 * f64::EPSILON * 4.0,
                3.0,
                am,
                bm,
            ])
            .unwrap()
            .max(f64::from_bits(16));
            for (actual, expected) in out.iter().zip(expected) {
                assert!((actual - expected).abs() <= bound);
            }
        }
    }
}

#[test]
fn scalar_scratch_admission_is_atomic_and_large_dc_is_representable() {
    let x = [f64::MAX * 0.125; 5];
    let input = || NdView::new(&x, StridedLayout::new(&[5], &[1]).unwrap()).unwrap();
    let analytic = AnalyticSignalNdPlan::new(&[5], &[0], SignalExecutionOptions::new()).unwrap();
    let mut short = AnalyticSignalNdPlan::<f64>::new(&[4], &[0], SignalExecutionOptions::new())
        .unwrap()
        .workspace(())
        .unwrap();
    let mut scratch = analytic.workspace(()).unwrap();
    let mut out = [Complex::new(42.0, 17.0); 5];
    assert_eq!(
        error_kind(analytic.analytic_into(
            input(),
            NdViewMut::new(&mut out, StridedLayout::new(&[5], &[1]).unwrap()).unwrap(),
            &mut short,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [Complex::new(42.0, 17.0); 5]);
    analytic
        .analytic_into(
            input(),
            NdViewMut::new(&mut out, StridedLayout::new(&[5], &[1]).unwrap()).unwrap(),
            &mut scratch,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    for (actual, &expected) in out.iter().zip(&x) {
        assert_eq!(actual.re, expected);
        assert!(actual.im.abs() <= 128.0 * f64::EPSILON * expected);
    }
    let resample = FourierResamplePlan::new(&[5], 0, 8, SignalExecutionOptions::new()).unwrap();
    let mut scratch = resample.workspace(()).unwrap();
    let mut out = [42.0; 8];
    assert_eq!(
        error_kind(resample.resample_into(
            input(),
            NdViewMut::new(&mut out, StridedLayout::new(&[8], &[1]).unwrap()).unwrap(),
            &mut SignalWorkspace::new(),
            native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [42.0; 8]);
    resample
        .resample_into(
            input(),
            NdViewMut::new(&mut out, StridedLayout::new(&[8], &[1]).unwrap()).unwrap(),
            &mut scratch,
            native(),
        )
        .unwrap();
    for actual in out {
        assert!((actual - x[0]).abs() <= 128.0 * f64::EPSILON * x[0]);
    }
}

#[test]
fn late_overlap_overflow_preserves_state_and_all_output() {
    let plan = OverlapPlan::new(&[1.0, 1.0], 8, SignalExecutionOptions::new()).unwrap();
    let mut stream = plan.stream().unwrap();
    let checkpoint = stream.clone();
    let mut input = [1.0; 16];
    input[8..].fill(f64::MAX);
    let mut out = [42.0; 17];
    let mut work = stream.workspace(16).unwrap();
    assert_eq!(
        error_kind(stream.process_into(&input, StreamChunk::Final, &mut out, &mut work, native())),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(out, [42.0; 17]);
    assert_eq!(stream.output_len(16, StreamChunk::Final).unwrap(), 17);
    stream.clone_from(&checkpoint);
    input.fill(0.125);
    stream
        .process_into(&input, StreamChunk::Final, &mut out, &mut work, native())
        .unwrap();
    assert!((out[0] - 0.125).abs() < 1e-14);
    assert!(out[1..16].iter().all(|v| (v - 0.25).abs() < 1e-14));
    assert!((out[16] - 0.125).abs() < 1e-14);
}

#[test]
fn prepared_ricker_coefficients_match_literal_kernel_and_reuse() {
    let options = RidgeOptions::default()
        .with_support_radius(3)
        .with_max_distance(2)
        .with_max_gap(0)
        .with_min_length(1)
        .with_min_strength(0.0);
    let plan = RidgePlan::new(&[1.0_f64, 2.0], options).unwrap();
    let input = [0.0, 0.25, 1.0, 0.25, 0.0, -0.125, 0.0];
    let mut peaks = [RidgePeak::default(); 3];
    let mut workspace = plan.workspace(7).unwrap();
    let count = plan
        .find_peaks_into(&input, &mut peaks, &mut workspace, native())
        .unwrap();
    assert!(count > 0);
    let before = peaks;
    // A ridge ending at the finest scale reports its literal unit-scale coefficient.
    let coefficient = |i: usize| -> f64 {
        (-3_isize..=3)
            .map(|offset| {
                let u = offset as f64;
                let kernel = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.sqrt().sqrt())
                    * (1.0 - u * u)
                    * (-0.5 * u * u).exp();
                let j = (i as isize + offset).clamp(0, 6) as usize;
                input[j] * kernel
            })
            .sum()
    };
    for peak in peaks[..count].iter().filter(|p| p.scale_index() == 0) {
        assert!((peak.strength() - coefficient(peak.index())).abs() <= 16.0 * f64::EPSILON);
    }
    assert_eq!(
        plan.find_peaks_into(&input, &mut peaks, &mut workspace, native())
            .unwrap(),
        count
    );
    assert_eq!(peaks, before);
    assert_eq!(
        error_kind(RidgePlan::new(&[2.0, 1.0], options)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(RidgePlan::new(&[1.0, f64::NAN], options)),
        SciOxideErrorKind::NonFinite
    );
}

fn extreme_real_transforms<S: RealScalar>(large: S, small: S) {
    let analytic = AnalyticSignalNdPlan::new(&[8], &[0], SignalExecutionOptions::new()).unwrap();
    let resample = FourierResamplePlan::new(&[8], 0, 16, SignalExecutionOptions::new()).unwrap();
    let mut analytic_work = analytic.workspace(()).unwrap();
    let mut resample_work = resample.workspace(()).unwrap();
    let mut z = [Complex::new(S::ZERO, S::ZERO); 8];
    let mut y = [S::ZERO; 16];
    let tiny = S::MIN_SUBNORMAL * S::try_from_f64_lossy(4.0).unwrap();
    for value in [
        S::ONE,
        large,
        small,
        tiny,
        S::MAX * S::try_from_f64_lossy(0.125).unwrap(),
        S::ZERO,
    ] {
        let x = [value; 8];
        let view = || NdView::new(&x, StridedLayout::new(&[8], &[1]).unwrap()).unwrap();
        analytic
            .analytic_into(
                view(),
                NdViewMut::new(&mut z, StridedLayout::new(&[8], &[1]).unwrap()).unwrap(),
                &mut analytic_work,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        resample
            .resample_into(
                view(),
                NdViewMut::new(&mut y, StridedLayout::new(&[16], &[1]).unwrap()).unwrap(),
                &mut resample_work,
                native(),
            )
            .unwrap();
        let bound = S::try_from_f64_lossy(128.0).unwrap() * S::EPSILON * value.abs()
            + S::try_from_f64_lossy(16.0).unwrap() * S::MIN_SUBNORMAL;
        for actual in z {
            assert_eq!(actual.re, value);
            assert!(actual.im.abs() <= bound);
        }
        for actual in y {
            assert!((actual - value).abs() <= bound);
        }
    }
}
#[test]
fn real_transforms_reuse_both_precision_frames_and_subnormals() {
    extreme_real_transforms(1e30_f32, 1e-30);
    extreme_real_transforms(1e300_f64, 1e-300);
}

#[test]
fn large_fft_paths_follow_independent_filter_and_tone_definitions() {
    let n = 4096;
    let x: Vec<_> = (0..n)
        .map(|i| (std::f64::consts::TAU * (i * 17) as f64 / n as f64).cos())
        .collect();
    let taps: Vec<_> = (0..17).map(|i| (i as f64 * 0.23).sin() / 17.0).collect();
    let plan = ConvolutionNdPlan::new(
        &[n],
        NdView::new(&taps, StridedLayout::new(&[17], &[1]).unwrap()).unwrap(),
        &[0],
        ConvolutionOptions::new().with_method(ConvolutionMethod::Fft),
    )
    .unwrap();
    let out = plan
        .convolve(NdView::new(&x, StridedLayout::new(&[n], &[1]).unwrap()).unwrap())
        .unwrap();
    let bound =
        128.0 * f64::EPSILON * (8192.0_f64).log2() * taps.iter().map(|v| v.abs()).sum::<f64>();
    for (i, actual) in out.iter().enumerate() {
        let expected: f64 = taps
            .iter()
            .enumerate()
            .filter_map(|(j, h)| i.checked_sub(j).filter(|&k| k < n).map(|k| x[k] * h))
            .sum();
        assert!((actual - expected).abs() <= bound);
    }
    let analytic = scioxide_signal::hilbert(&x).unwrap();
    for (i, actual) in analytic.iter().enumerate() {
        let phase = std::f64::consts::TAU * (i * 17) as f64 / n as f64;
        assert_eq!(actual.re, x[i]);
        assert!((actual.im - phase.sin()).abs() <= 128.0 * f64::EPSILON * 12.0);
    }
}
