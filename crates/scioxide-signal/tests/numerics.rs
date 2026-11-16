#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, NumericalIssue, SciOxideErrorView, StridedLayout,
};
use scioxide_linalg::{GeneralEigenvaluesResult, Matrix};
use scioxide_signal::*;

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

#[test]
fn lomb_scipy_and_rank_revealing_reference() {
    let times: Vec<_> = (0..64).map(|i| i as f64).collect();
    // The generator rounds samples to multiples of 2^-20 so that last-ulp libm
    // differences in sin/cos cannot change them.
    let scale = 2.0_f64.powi(20);
    let input: Vec<_> = times
        .iter()
        .map(|&t| {
            (((0.7 * t).sin() + 0.03 * (1.3 * t).cos() + 0.2) * scale).round_ties_even() / scale
        })
        .collect();
    let weights: Vec<_> = (0..64).map(|i| 1.0 + (i % 5) as f64 / 5.0).collect();
    let rows: Vec<_> = include_str!("fixtures/lomb_numerics.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(numbers)
        .collect();
    for group in rows.chunks_exact(8) {
        let frequencies: Vec<_> = group.iter().map(|r| r[2]).collect();
        let options = LombScargleOptions::default()
            .with_mean(if group[0][0] == 1.0 {
                LombScargleMean::Floating
            } else {
                LombScargleMean::Zero
            })
            .with_normalization(if group[0][1] == 1.0 {
                LombScargleNormalization::Normalized
            } else {
                LombScargleNormalization::Power
            });
        let power = LombScargle::new(&times, &frequencies, Some(&weights), options)
            .unwrap()
            .power(&input)
            .unwrap();
        for (actual, row) in power.iter().zip(group) {
            assert!(actual.is_finite());
            assert!(
                (actual - row[3]).abs() < 1e-12 * row[3].abs().max(1.0),
                "{row:?}: {actual}"
            );
            if row[4].is_finite() {
                assert!((actual - row[4]).abs() < 1e-12 * row[4].abs().max(1.0));
            }
            let isolated = LombScargle::new(&times, &row[2..3], Some(&weights), options)
                .unwrap()
                .power(&input)
                .unwrap();
            assert_eq!(*actual, isolated[0]);
        }
    }
}

#[test]
fn state_space_scipy_coefficients_and_root_accuracy() {
    let lines: Vec<_> = include_str!("fixtures/lti_numerics.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(9) {
        let header = numbers(case[0]);
        let n = header[0] as usize;
        let a = Matrix::from_vec(n, n, numbers(case[1])).unwrap();
        let b = Matrix::from_vec(n, 1, numbers(case[2])).unwrap();
        let c = Matrix::from_vec(1, n, numbers(case[3])).unwrap();
        let d = Matrix::from_vec(1, 1, vec![header[1]]).unwrap();
        let system =
            StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
        let tf = system.to_transfer(0, 0).unwrap();
        let mut numerator = vec![0.0; n + 1];
        numerator[n + 1 - tf.numerator().len()..].copy_from_slice(tf.numerator());
        for (actual, expected) in [
            (&numerator[..], numbers(case[4])),
            (tf.denominator(), numbers(case[5])),
        ] {
            let scale = expected.iter().copied().map(f64::abs).fold(0.0, f64::max);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                error <= 1e-12 * scale,
                "n={n} d={} coefficient error={}",
                header[1],
                error / scale
            );
        }
        let mut companion = Matrix::zeros(n, n).unwrap();
        for (out, &den) in companion.as_mut_slice()[..n]
            .iter_mut()
            .zip(&tf.denominator()[1..])
        {
            *out = -den;
        }
        for i in 1..n {
            companion.as_mut_slice()[i * n + i - 1] = 1.0;
        }
        let roots = GeneralEigenvaluesResult::compute(&companion).unwrap();
        let re = numbers(case[6]);
        let im = numbers(case[7]);
        let eigen: Vec<_> = re
            .iter()
            .zip(im)
            .map(|(&re, im)| Complex::new(re, im))
            .collect();
        let error = roots
            .eigenvalues()
            .iter()
            .map(|r| {
                eigen
                    .iter()
                    .map(|&e| (*r - e).norm())
                    .fold(f64::INFINITY, f64::min)
                    / r.norm()
            })
            .fold(0.0, f64::max);
        let reference: f64 = case[8].parse().unwrap();
        assert!(
            error <= 10.0 * reference.max(f64::EPSILON),
            "n={n} root error={error} scipy={reference}"
        );
    }
}

#[test]
fn sectioned_realization_preserves_stable_unstable_and_delayed_responses() {
    for domain in [IirDomain::Analog, IirDomain::Digital] {
        for pole in [-0.7, 1.2] {
            let poles: Vec<_> = (0..24)
                .map(|i| Complex::new(pole - i as f64 * 0.03, 0.0))
                .collect();
            let zeros = [Complex::new(-0.2, 0.1), Complex::new(-0.2, -0.1)];
            let zpk = PoleZeroGain::new(domain, &zeros, &poles, Complex::new(0.7, 0.2)).unwrap();
            let time = match domain {
                IirDomain::Analog => LtiTime::Continuous,
                IirDomain::Digital => LtiTime::Discrete { sample_period: 1.0 },
            };
            let system = zpk.to_state_space(time).unwrap();
            for frequency in [0.2, 0.6] {
                // Both representations share one frequency convention per domain.
                let expected = zpk.response(frequency).unwrap();
                let response = system.frequency_response(&[frequency]).unwrap();
                let actual = response.values()[0];
                assert!(
                    (actual - expected).norm() <= 1e-11 * expected.norm().max(1e-100),
                    "{domain:?} {pole}: {actual:?} {expected:?}"
                );
            }
        }
    }
}

#[test]
fn deconvolution_preserves_padding_and_zero_conventions() {
    assert_eq!(
        deconvolve(&[0.0, 1.0, 2.0], &[1.0, 1.0]).unwrap(),
        (vec![0.0, 1.0], vec![0.0, 0.0, 1.0])
    );
    assert_eq!(
        deconvolve(&[0.0, 0.0], &[2.0]).unwrap(),
        (vec![0.0, 0.0], vec![0.0, 0.0])
    );
    assert_eq!(deconvolve::<f64>(&[], &[1.0]).unwrap(), (vec![], vec![]));
}

#[test]
fn wide_range_convolution_accepts_tiny_components() {
    for large in [1e300, 1e-300, f64::from_bits(8)] {
        let mut input = vec![0.0; 64];
        input[0] = large;
        input[1] = if large > 1.0 {
            1e-300
        } else {
            f64::from_bits(1)
        };
        let h = [1.0, 0.5, 0.25];
        let run = |method| {
            let plan = ConvolutionNdPlan::new(
                &[64],
                NdView::new(&h, StridedLayout::new(&[3], &[1]).unwrap()).unwrap(),
                &[0],
                ConvolutionOptions::new().with_method(method),
            )
            .unwrap();
            plan.convolve(NdView::new(&input, StridedLayout::new(&[64], &[1]).unwrap()).unwrap())
                .unwrap()
        };
        let reference = run(ConvolutionMethod::Direct);
        for method in [ConvolutionMethod::Fft, ConvolutionMethod::Automatic] {
            for (actual, expected) in run(method).iter().zip(&reference) {
                assert!(
                    (actual - expected).abs() <= 256.0 * f64::EPSILON * large + f64::from_bits(8)
                );
            }
        }
    }
}

fn view<'a, T>(x: &'a [T], shape: &'a [usize], strides: &'a [usize]) -> NdView<'a, 'a, T> {
    NdView::new(x, StridedLayout::new(shape, strides).unwrap()).unwrap()
}

#[test]
fn wide_range_fft_adapters_and_complex_components() {
    let mut x = vec![0.0; 64];
    x[0] = 1e300;
    x[1] = 1e-300;
    let h = [1.0, 0.5, 0.25];
    let reference = convolve(&x, &h).unwrap();
    let close = |a: &[f64], b: &[f64]| {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() / 1e300 < 1e-12);
        }
    };
    {
        let p = OverlapPlan::new(&h, 64, SignalExecutionOptions::new()).unwrap();
        let mut stream = p.stream().unwrap();
        let mut out = vec![0.0; stream.output_len(x.len(), StreamChunk::Final).unwrap()];
        let mut w = stream.workspace(x.len()).unwrap();
        stream
            .process_into(
                &x,
                StreamChunk::Final,
                &mut out,
                &mut w,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        close(&out, &reference);
    }
    let p = SeparableFir2dPlan::new(
        [1, 64],
        &[1.0],
        &h,
        ConvolutionOptions::new().with_method(ConvolutionMethod::Fft),
    )
    .unwrap();
    let mut out = vec![0.0; 66];
    p.filter_into(
        view(&x, &[1, 64], &[64, 1]),
        NdViewMut::new(&mut out, StridedLayout::new(&[1, 66], &[66, 1]).unwrap()).unwrap(),
        &mut p.workspace(()).unwrap(),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    close(&out, &reference);
    let a = Matrix::from_vec(1, 64, x.clone()).unwrap();
    let b = Matrix::from_vec(1, 3, h.to_vec()).unwrap();
    close(
        convolve2d(a.view(), b.view()).unwrap().as_slice(),
        &reference,
    );

    let p = FourierResamplePlan::<f64>::new(&[64], 0, 64, SignalExecutionOptions::new()).unwrap();
    let mut out = vec![0.0; 64];
    p.resample_into(
        view(&x, &[64], &[1]),
        NdViewMut::new(&mut out, StridedLayout::new(&[64], &[1]).unwrap()).unwrap(),
        &mut p.workspace(()).unwrap(),
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    close(&out, &x);

    let z: Vec<_> = x.iter().map(|&v| Complex::new(v, 1e-300)).collect();
    let kernel = [Complex::new(1.0, 0.0), Complex::new(0.5, 1e-300)];
    let run = |method| {
        ConvolutionNdPlan::new(
            &[64],
            view(&kernel, &[2], &[1]),
            &[0],
            ConvolutionOptions::new().with_method(method),
        )
        .unwrap()
        .convolve(view(&z, &[64], &[1]))
        .unwrap()
    };
    for (a, b) in run(ConvolutionMethod::Direct)
        .iter()
        .zip(run(ConvolutionMethod::Fft))
    {
        assert!((*a - b).norm() / 1e300 < 1e-12);
    }
}

#[test]
fn fft_and_direct_reject_overflow_and_nonfinite_atomically() {
    for method in [
        ConvolutionMethod::Direct,
        ConvolutionMethod::Fft,
        ConvolutionMethod::Automatic,
    ] {
        let h = [2.0, 1.0];
        let p = ConvolutionNdPlan::new(
            &[64],
            view(&h, &[2], &[1]),
            &[0],
            ConvolutionOptions::new().with_method(method),
        )
        .unwrap();
        for bad in [f64::MAX, f64::NAN, f64::INFINITY] {
            let mut x = vec![0.0; 64];
            x[0] = bad;
            let mut output = vec![123.0; 65];
            let error = p
                .convolve_into(
                    view(&x, &[64], &[1]),
                    NdViewMut::new(&mut output, StridedLayout::new(&[65], &[1]).unwrap()).unwrap(),
                    &mut p.workspace(ConvolutionKernel::Prepared).unwrap(),
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err();
            if bad.is_finite() {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ));
            } else {
                assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
            }
            assert_eq!(output, vec![123.0; 65]);
        }
    }
}

#[test]
fn lomb_f32_degenerate_grid_retains_ordinary_bins() {
    let times: Vec<_> = (0..64).map(|i| i as f32).collect();
    let y: Vec<_> = times.iter().map(|t| (0.7 * t).sin()).collect();
    let frequencies = [0.7, core::f32::consts::PI, core::f32::consts::TAU, 1e-12];
    for floating_mean in [true, false] {
        let options = LombScargleOptions::default().with_mean(if floating_mean {
            LombScargleMean::Floating
        } else {
            LombScargleMean::Zero
        });
        let power = LombScargle::new(&times, &frequencies, None, options)
            .unwrap()
            .power(&y)
            .unwrap();
        assert!(
            power
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        );
        assert!((power[0] - 1.0).abs() < 64.0 * f32::EPSILON);
        if floating_mean {
            assert_eq!(power[3], 0.0);
        }
    }
}

#[test]
fn automatic_fft_and_prepared_kernels_share_wide_range_policy() {
    let mut wide = vec![0.0_f64; 64];
    wide[0] = 1e300;
    wide[1] = 1e-30;
    let mut taps = vec![0.0; 64];
    taps[..3].copy_from_slice(&[1.0, -0.5, 0.25]);
    for (input, kernel) in [(&wide, &taps), (&taps, &wide)] {
        let prepare = |method| {
            ConvolutionNdPlan::new(
                &[64],
                view(kernel, &[64], &[1]),
                &[0],
                ConvolutionOptions::new().with_method(method),
            )
            .unwrap()
        };
        let reference = prepare(ConvolutionMethod::Direct)
            .convolve(view(input, &[64], &[1]))
            .unwrap();
        let automatic = prepare(ConvolutionMethod::Automatic);
        assert_eq!(automatic.selection().method(), ConvolutionMethod::Fft);
        let actual = automatic.convolve(view(input, &[64], &[1])).unwrap();
        for (a, b) in actual.iter().zip(&reference) {
            assert!((a - b).abs() / 1e300 < 1e-12);
        }
    }
}
