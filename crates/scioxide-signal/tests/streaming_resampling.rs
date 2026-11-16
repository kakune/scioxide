#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, SciOxideErrorKind, StridedLayout,
};
use scioxide_signal::*;
fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
fn view<'a, T>(x: &'a [T], shape: &'a [usize], strides: &'a [usize]) -> NdView<'a, 'a, T> {
    NdView::new(x, StridedLayout::new(shape, strides).unwrap()).unwrap()
}
#[test]
fn asymmetric_nd_direct_and_automatic_match_scatter_reference() {
    let long: Vec<_> = (0..10_000).map(|i| (i % 11) as f64 - 5.0).collect();
    let short = [0.25, -0.5, 0.75];
    for (a, b) in [
        (long.as_slice(), short.as_slice()),
        (short.as_slice(), long.as_slice()),
    ] {
        for product in [SignalProduct::Convolution, SignalProduct::Correlation] {
            let mut expected = vec![0.0; a.len() + b.len() - 1];
            for (i, &x) in a.iter().enumerate() {
                for (j, &y) in b.iter().enumerate() {
                    let k = if product == SignalProduct::Correlation {
                        i + b.len() - 1 - j
                    } else {
                        i + j
                    };
                    expected[k] += x * y;
                }
            }
            for extent in [
                ConvolutionExtent::Full,
                ConvolutionExtent::Same,
                ConvolutionExtent::Valid,
            ] {
                if extent == ConvolutionExtent::Valid && a.len() < b.len() {
                    continue;
                }
                let (start, len) = match extent {
                    ConvolutionExtent::Full => (0, expected.len()),
                    ConvolutionExtent::Same => ((b.len() - 1) / 2, a.len()),
                    ConvolutionExtent::Valid => (b.len() - 1, a.len() - b.len() + 1),
                    ConvolutionExtent::Circular => (0, a.len()),
                };
                for method in [ConvolutionMethod::Direct, ConvolutionMethod::Automatic] {
                    let plan = ConvolutionNdPlan::new(
                        &[a.len()],
                        view(b, &[b.len()], &[1]),
                        &[0],
                        ConvolutionOptions::new()
                            .with_method(method)
                            .with_product(product)
                            .with_extent(extent),
                    )
                    .unwrap();
                    assert_eq!(plan.selection().method(), ConvolutionMethod::Direct);
                    let products = if extent == ConvolutionExtent::Full {
                        a.len() * b.len()
                    } else {
                        len * a.len().min(b.len())
                    };
                    assert_eq!(plan.selection().direct_products(), products);
                    let actual = plan.convolve(view(a, &[a.len()], &[1])).unwrap();
                    assert_eq!(actual.len(), len);
                    for (a, b) in actual.iter().zip(&expected[start..start + len]) {
                        assert!((a - b).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
fn run<T: NumericScalar>(
    x: &[T],
    h: &[T],
    method: ConvolutionMethod,
    product: SignalProduct,
) -> Vec<T> {
    let p = ConvolutionNdPlan::new(
        &[x.len()],
        view(h, &[h.len()], &[1]),
        &[0],
        ConvolutionOptions::new()
            .with_method(method)
            .with_product(product),
    )
    .unwrap();
    let mut out = vec![T::ZERO; p.output_shape()[0]];
    let mut work = p.workspace(ConvolutionKernel::Prepared).unwrap();
    let shape = [out.len()];
    p.convolve_into(
        view(x, &[x.len()], &[1]),
        NdViewMut::new(&mut out, StridedLayout::new(&shape, &[1]).unwrap()).unwrap(),
        &mut work,
        native(),
    )
    .unwrap();
    out
}
#[test]
fn direct_fft_real_complex_and_conventional_lags() {
    for method in [ConvolutionMethod::Direct, ConvolutionMethod::Fft] {
        let y = run(
            &[1.0_f64, 2.0],
            &[3.0, 4.0, 5.0],
            method,
            SignalProduct::Correlation,
        );
        for (a, b) in y.iter().zip([5.0, 14.0, 11.0, 6.0]) {
            assert!((a - b).abs() < 1e-12);
        }
        let a = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, -1.0)];
        let b = [Complex::new(2.0_f64, 1.0), Complex::new(-1.0, 2.0)];
        let expected = [
            a[0] * b[1].conj(),
            a[0] * b[0].conj() + a[1] * b[1].conj(),
            a[1] * b[0].conj(),
        ];
        for (actual, expected) in run(&a, &b, method, SignalProduct::Correlation)
            .iter()
            .zip(expected)
        {
            assert!((*actual - expected).abs() < 1e-12);
        }
        let y = run(
            &[1.0_f32, 2.0],
            &[3.0_f32, 4.0],
            method,
            SignalProduct::Convolution,
        );
        for (a, b) in y.iter().zip([3.0, 10.0, 8.0]) {
            assert!((a - b).abs() < 1e-5);
        }
        let a = [Complex::new(1.0_f32, 2.0), Complex::new(3.0, -1.0)];
        let y = run(&a, &a, method, SignalProduct::Convolution);
        for (a, b) in y
            .iter()
            .zip([a[0] * a[0], a[0] * a[1] + a[1] * a[0], a[1] * a[1]])
        {
            assert!((*a - b).abs() < 1e-5);
        }
    }
}
#[test]
fn nd_boundaries_and_strided_channels() {
    let x = [1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0];
    let k = [1.0, 2.0, 1.0];
    for boundary in [
        SignalBoundary::Zero,
        SignalBoundary::Reflect,
        SignalBoundary::Periodic,
    ] {
        let mut results = Vec::new();
        for method in [ConvolutionMethod::Direct, ConvolutionMethod::Fft] {
            let p = ConvolutionNdPlan::new(
                &[2, 3],
                view(&k, &[1, 3], &[3, 1]),
                &[1],
                ConvolutionOptions::new()
                    .with_method(method)
                    .with_boundary(boundary)
                    .with_extent(ConvolutionExtent::Same),
            )
            .unwrap();
            let mut out = [0.0; 6];
            let mut w = p.workspace(ConvolutionKernel::Prepared).unwrap();
            p.convolve_into(
                view(&x, &[2, 3], &[1, 2]),
                NdViewMut::new(&mut out, StridedLayout::new(&[2, 3], &[3, 1]).unwrap()).unwrap(),
                &mut w,
                native(),
            )
            .unwrap();
            results.push(out);
        }
        for (a, b) in results[0].iter().zip(results[1]) {
            assert!((a - b).abs() < 1e-10);
        }
        let first = match boundary {
            SignalBoundary::Zero => 4.0,
            SignalBoundary::Reflect => 6.0,
            SignalBoundary::Periodic => 7.0,
        };
        assert!((results[0][0] - first).abs() < 1e-12);
    }
}
fn oracle(x: &[f64], h: &[f64], up: usize, down: usize, phase: usize) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let last = (x.len() - 1) * up + h.len() - 1;
    (phase..=last)
        .step_by(down)
        .map(|tick| {
            x.iter()
                .enumerate()
                .filter_map(|(i, &v)| {
                    tick.checked_sub(i * up)
                        .and_then(|k| h.get(k).map(|h| v * h))
                })
                .sum()
        })
        .collect()
}
#[test]
fn polyphase_segmentation_phase_flush_and_checkpoint() {
    let x: Vec<_> = (0..79)
        .map(|i| ((i * 17 % 31) as f64 - 15.0) / 16.0)
        .collect();
    let h = [0.2, -0.3, 0.4, 0.8, 0.1, -0.2, 0.7];
    for (up, down) in [(1, 1), (2, 3), (3, 2), (6, 4), (1, 7), (7, 1)] {
        for phase in 0..down {
            let p = PolyphasePlan::new(&h, up, down, phase, ResampleAlignment::Causal).unwrap();
            let expected = oracle(&x, &h, up, down, phase);
            for chunk in [1, 2, 7, 19, 100] {
                let mut stream = p.stream().unwrap();
                let mut work = stream.workspace(chunk).unwrap();
                let mut actual = Vec::new();
                for part in x.chunks(chunk) {
                    let n = stream.output_len(part.len(), StreamChunk::Partial).unwrap();
                    let mut out = vec![0.0; n];
                    stream
                        .process_into(part, StreamChunk::Partial, &mut out, &mut work, native())
                        .unwrap();
                    actual.extend(out);
                }
                let checkpoint = stream.clone();
                let n = stream.output_len(0, StreamChunk::Final).unwrap();
                let mut tail = vec![0.0; n];
                stream
                    .process_into(&[], StreamChunk::Final, &mut tail, &mut work, native())
                    .unwrap();
                actual.extend_from_slice(&tail);
                stream.clone_from(&checkpoint);
                let mut replay = vec![0.0; n];
                stream
                    .process_into(&[], StreamChunk::Final, &mut replay, &mut work, native())
                    .unwrap();
                assert_eq!(tail, replay);
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "{up}/{down} phase {phase}: {a} != {b}"
                    );
                }
                assert_eq!(
                    error_kind(stream.output_len(0, StreamChunk::Final)),
                    SciOxideErrorKind::InvalidInput
                );
                stream.reset();
                assert_eq!(stream.output_len(0, StreamChunk::Final).unwrap(), 0);
            }
        }
    }
}
#[test]
fn default_filter_centered_and_transactional_errors() {
    let p = PolyphasePlan::<f64>::rational_default(6, 4, ResampleAlignment::Centered).unwrap();
    assert_eq!((p.up(), p.down()), (3, 2));
    assert_eq!(p.output_len(11).unwrap(), 17);
    assert!((p.coefficients().iter().sum::<f64>() - 3.0).abs() < 1e-12);
    let mut stream = p.stream().unwrap();
    let mut work = stream.workspace(11).unwrap();
    let mut out = vec![99.0; 17];
    assert_eq!(
        error_kind(stream.process_into(
            &[f64::NAN; 11],
            StreamChunk::Final,
            &mut out,
            &mut work,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, vec![99.0; 17]);
    assert_eq!(stream.output_len(11, StreamChunk::Final).unwrap(), 17);
    stream
        .process_into(
            &[1.0; 11],
            StreamChunk::Final,
            &mut out,
            &mut work,
            native(),
        )
        .unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}
#[test]
fn overlap_add_matches_independent_full_convolution() {
    let x: Vec<_> = (0..71).map(|i| (i as f64 * 0.3).sin()).collect();
    let h: Vec<_> = (0..23).map(|i| (i as f64 * 0.2).cos() / 23.0).collect();
    let expected = oracle(&x, &h, 1, 1, 0);
    {
        for block in [1, 7, 32] {
            for chunk in [1, 5, 64, 100] {
                let p = OverlapPlan::new(&h, block, SignalExecutionOptions::new()).unwrap();
                let mut s = p.stream().unwrap();
                let mut w = s.workspace(chunk).unwrap();
                let mut actual = Vec::new();
                for part in x.chunks(chunk) {
                    let mut out =
                        vec![0.0; s.output_len(part.len(), StreamChunk::Partial).unwrap()];
                    s.process_into(part, StreamChunk::Partial, &mut out, &mut w, native())
                        .unwrap();
                    actual.extend(out);
                }
                let checkpoint = s.clone();
                let mut out = vec![0.0; s.output_len(0, StreamChunk::Final).unwrap()];
                s.process_into(&[], StreamChunk::Final, &mut out, &mut w, native())
                    .unwrap();
                actual.extend_from_slice(&out);
                s.clone_from(&checkpoint);
                let mut replay = vec![0.0; out.len()];
                s.process_into(&[], StreamChunk::Final, &mut replay, &mut w, native())
                    .unwrap();
                assert_eq!(replay, out);
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    assert!((a - b).abs() < 1e-11, "{block} {chunk}: {a} {b}");
                }
            }
        }
    }
}
#[test]
fn fourier_parity_tones_and_strided_channels() {
    for n in [1, 2, 5, 6, 9, 10] {
        for m in [1, 2, 5, 6, 9, 10] {
            let x: Vec<_> = (0..n)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / n as f64;
                    Complex::new(2.0 + a.cos(), a.sin())
                })
                .collect();
            let p =
                FourierResamplePlan::<Complex<f64>>::new(&[n], 0, m, SignalExecutionOptions::new())
                    .unwrap();
            let mut out = vec![Complex::new(0.0, 0.0); m];
            let mut work = p.workspace(()).unwrap();
            p.resample_into(
                view(&x, &[n], &[1]),
                NdViewMut::new(&mut out, StridedLayout::new(&[m], &[1]).unwrap()).unwrap(),
                &mut work,
                native(),
            )
            .unwrap();
            // Independent direct DFT followed by explicit signed-frequency interpolation.
            let bins: Vec<_> = (0..n)
                .map(|k| {
                    x.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let a = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                            v * Complex::new(a.cos(), a.sin())
                        })
                        .sum::<Complex<f64>>()
                        / n as f64
                })
                .collect();
            for (j, &actual) in out.iter().enumerate() {
                let mut expected = bins[0];
                let common = n.min(m);
                for k in 1..=(common - 1) / 2 {
                    let a = std::f64::consts::TAU * (j * k) as f64 / m as f64;
                    expected += bins[k] * Complex::new(a.cos(), a.sin())
                        + bins[n - k] * Complex::new(a.cos(), -a.sin());
                }
                if common.is_multiple_of(2) {
                    let k = common / 2;
                    let a = std::f64::consts::TAU * (j * k) as f64 / m as f64;
                    expected += if n <= m {
                        bins[k] * a.cos()
                    } else {
                        (bins[k] + bins[n - k]) * a.cos()
                    };
                }
                assert!(
                    (actual - expected).abs() < 1e-11,
                    "{n}->{m}: {actual:?} != {expected:?}"
                );
            }
        }
    }
    let x = [1.0_f32, 2.0, 1.0, 2.0, 1.0, 2.0];
    let p = FourierResamplePlan::<f32>::new(&[2, 3], 1, 8, SignalExecutionOptions::new()).unwrap();
    let mut out = [0.0; 16];
    let mut work = p.workspace(()).unwrap();
    p.resample_into(
        view(&x, &[2, 3], &[1, 2]),
        NdViewMut::new(&mut out, StridedLayout::new(&[2, 8], &[8, 1]).unwrap()).unwrap(),
        &mut work,
        native(),
    )
    .unwrap();
    for (i, v) in out.iter().enumerate() {
        assert!((v - if i < 8 { 1.0 } else { 2.0 }).abs() < 1e-5);
    }
}
#[test]
fn separable_matches_outer_product_and_polyphase_channels() {
    let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rows = [0.25, 0.5, 0.25];
    let cols = [1.0, -1.0];
    let kernel: Vec<_> = rows
        .iter()
        .flat_map(|a| cols.iter().map(move |b| a * b))
        .collect();
    let p = SeparableFir2dPlan::new([2, 3], &rows, &cols, ConvolutionOptions::default()).unwrap();
    let q = ConvolutionNdPlan::new(
        &[2, 3],
        view(&kernel, &[3, 2], &[2, 1]),
        &[0, 1],
        ConvolutionOptions::default(),
    )
    .unwrap();
    let mut a = [0.0; 16];
    let mut b = [0.0; 16];
    let mut w = SignalWorkspace::new();
    w.try_reserve(
        &p.workspace_requirements(()).unwrap().max(
            q.workspace_requirements(ConvolutionKernel::Prepared)
                .unwrap(),
        ),
    )
    .unwrap();
    p.filter_into(
        view(&x, &[2, 3], &[3, 1]),
        NdViewMut::new(&mut a, StridedLayout::new(&[4, 4], &[4, 1]).unwrap()).unwrap(),
        &mut w,
        native(),
    )
    .unwrap();
    q.convolve_into(
        view(&x, &[2, 3], &[3, 1]),
        NdViewMut::new(&mut b, StridedLayout::new(&[4, 4], &[4, 1]).unwrap()).unwrap(),
        &mut w,
        native(),
    )
    .unwrap();
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-12);
    }
    let p = PolyphasePlan::fir(&[1.0, -1.0]).unwrap();
    let mut w = p
        .workspace(SignalLayout::Axis {
            shape: &[2, 3],
            axis: 1,
        })
        .unwrap();
    let mut out = [0.0; 8];
    p.resample_axis_into(
        view(&x, &[2, 3], &[3, 1]),
        1,
        NdViewMut::new(&mut out, StridedLayout::new(&[2, 4], &[4, 1]).unwrap()).unwrap(),
        &mut w,
        native(),
    )
    .unwrap();
    for (a, b) in out.iter().zip([1.0, 1.0, 1.0, -3.0, 4.0, 1.0, 1.0, -6.0]) {
        assert!((a - b).abs() < 1e-12);
    }
}
#[test]
fn multiple_axes_and_distinct_channel_kernels_match_literal_sums() {
    let x: Vec<_> = (0..12)
        .map(|i| Complex::new(i as f64 * 0.1 - 0.5, i as f64 * 0.07))
        .collect();
    let h: Vec<_> = (0..8)
        .map(|i| Complex::new(i as f64 * 0.13, 0.4 - i as f64 * 0.03))
        .collect();
    for product in [SignalProduct::Convolution, SignalProduct::Correlation] {
        let mut expected = vec![Complex::new(0.0, 0.0); 24];
        for r in 0..3 {
            for channel in 0..2 {
                for col in 0..4 {
                    for kr in 0..2 {
                        for kc in 0..2 {
                            let (sr, sc) = if product == SignalProduct::Convolution {
                                (r as isize - kr as isize, col as isize - kc as isize)
                            } else {
                                (r as isize - 1 + kr as isize, col as isize - 1 + kc as isize)
                            };
                            if (0..2).contains(&sr) && (0..3).contains(&sc) {
                                let k = h[(kr * 2 + channel) * 2 + kc];
                                expected[(r * 2 + channel) * 4 + col] += x
                                    [(sr as usize * 2 + channel) * 3 + sc as usize]
                                    * if product == SignalProduct::Correlation {
                                        k.conj()
                                    } else {
                                        k
                                    };
                            }
                        }
                    }
                }
            }
        }
        for method in [ConvolutionMethod::Direct, ConvolutionMethod::Fft] {
            let p = ConvolutionNdPlan::new(
                &[2, 2, 3],
                view(&h, &[2, 2, 2], &[4, 2, 1]),
                &[2, 0],
                ConvolutionOptions::new()
                    .with_method(method)
                    .with_product(product),
            )
            .unwrap();
            assert_eq!(p.selection().direct_products(), 48);
            let actual = p.convolve(view(&x, &[2, 2, 3], &[6, 3, 1])).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((*a - *b).abs() < 1e-11);
            }
        }
    }
}
#[test]
fn valid_even_same_and_empty_batch_extents() {
    for extent in [ConvolutionExtent::Same, ConvolutionExtent::Valid] {
        let p = ConvolutionNdPlan::new(
            &[3],
            view(&[1.0_f64, 2.0], &[2], &[1]),
            &[0],
            ConvolutionOptions::new().with_extent(extent),
        )
        .unwrap();
        let out = p.convolve(view(&[1.0, 2.0, 3.0], &[3], &[1])).unwrap();
        assert_eq!(
            out,
            if extent == ConvolutionExtent::Same {
                vec![1.0, 4.0, 7.0]
            } else {
                vec![4.0, 7.0]
            }
        );
    }
    for method in [ConvolutionMethod::Direct, ConvolutionMethod::Fft] {
        let p = ConvolutionNdPlan::new(
            &[0, 3],
            view(&[1.0_f64], &[1, 1], &[1, 1]),
            &[1],
            ConvolutionOptions::new().with_method(method),
        )
        .unwrap();
        assert!(p.convolve(view(&[], &[0, 3], &[3, 1])).unwrap().is_empty());
    }
}
#[test]
fn default_f32_audio_and_large_coprime_rates_preserve_dc_gain() {
    for (up, down) in [(160, 147), (1000, 1001)] {
        let plan =
            PolyphasePlan::<f32>::rational_default(up, down, ResampleAlignment::Centered).unwrap();
        let sum: f64 = plan.coefficients().iter().map(|&v| f64::from(v)).sum();
        assert!(
            (sum - up as f64).abs() <= up as f64 * f64::from(f32::EPSILON) * 4.0,
            "{up}/{down}: DC gain {sum}"
        );
    }
}
