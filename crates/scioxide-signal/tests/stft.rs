//! Independent finite DFT, overlap constraints and streaming coverage.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorKind, Tolerance,
};
use scioxide_signal::*;
fn z() -> Complex<f64> {
    Complex::new(0., 0.)
}
fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
#[test]
fn direct_spectral_reference_and_boundary_reconstruction() {
    for m in [3, 4, 7, 8] {
        for fft in [m, m + 3] {
            for hop in [1, 2, m] {
                let window: Vec<_> = (0..m).map(|i| 0.3 + 0.07 * i as f64).collect();
                for n in [0, 1, 2, 19] {
                    for boundary in [
                        SignalBoundary::Zero,
                        SignalBoundary::Periodic,
                        SignalBoundary::Reflect,
                    ] {
                        for packing in [
                            StftPacking::Full,
                            StftPacking::Centered,
                            StftPacking::OneSided,
                        ] {
                            let options = StftOptions::new()
                                .with_fft_len(Some(fft))
                                .with_boundary(boundary)
                                .with_packing(packing)
                                .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap());
                            let plan = StftPlan::new(&window, hop, options).unwrap();
                            let input: Vec<_> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
                            let actual = plan.analyze(&input).unwrap();
                            let frames = plan.frames(n).unwrap();
                            for (i, frame) in actual.chunks(plan.bins()).enumerate() {
                                let start =
                                    (frames.first() + i as i64) * hop as i64 - (m / 2) as i64;
                                for (k, &value) in frame.iter().enumerate() {
                                    let frequency = plan.frequency(k).unwrap();
                                    let mut expected = z();
                                    for (j, &w) in window.iter().enumerate() {
                                        let t = start + j as i64;
                                        let sample = if t >= 0 && t < n as i64 {
                                            input[t as usize]
                                        } else {
                                            match boundary {
                                                SignalBoundary::Zero => 0.,
                                                SignalBoundary::Periodic => {
                                                    input[t.rem_euclid(n as i64) as usize]
                                                }
                                                SignalBoundary::Reflect => {
                                                    let p = 2 * (n as i64 - 1);
                                                    let idx = if p == 0 {
                                                        0
                                                    } else {
                                                        let a = t.rem_euclid(p);
                                                        a.min(p - a) as usize
                                                    };
                                                    input[idx]
                                                }
                                            }
                                        };
                                        let phase =
                                            -2. * std::f64::consts::PI * frequency * j as f64;
                                        expected +=
                                            Complex::new(phase.cos(), phase.sin()) * (sample * w);
                                    }
                                    assert!(
                                        (value - expected).norm() < 3e-12,
                                        "m={m} fft={fft} h={hop} n={n} frame={i} bin={k}: {value:?} {expected:?}"
                                    );
                                }
                            }
                            let mut output = vec![0.; n];
                            let mut work = plan.workspace(StftRequest::Synthesis(n)).unwrap();
                            plan.synthesize_into(&actual, &mut output, &mut work, cx())
                                .unwrap();
                            for (a, b) in output.iter().zip(&input) {
                                assert!((a - b).abs() < 3e-12);
                            }
                        }
                    }
                }
            }
        }
    }
}
fn precisions<T: NumericScalar>()
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let scalar = |x: f64| T::Real::try_from_f64_lossy(x).unwrap();
    let value = |x: f64, y: f64| T::from_components(scalar(x), scalar(y));
    let hop = 3_usize;
    for m in [7_usize, 8, 31] {
        let window: Vec<_> = (0..m).map(|i| value(0.6 + 0.01 * i as f64, 0.03)).collect();
        let input: Vec<_> = (0..73)
            .map(|i| value((i as f64 * 0.3).sin(), (i as f64 * 0.17).cos()))
            .collect();
        let fft = m + 2;
        // Each sample overlap-adds `overlaps` frames. The canonical dual's unit
        // constraint sums `overlaps` products per hop residue; the round trip
        // adds a length-`fft` transform pair per frame. Envelopes: `overlaps`
        // and `overlaps * ceil(log2 fft)` unit roundoffs on unit-scale data
        // (observed: dual residual <= 1 eps, round trip <= 2.5 eps).
        let overlaps = m.div_ceil(hop) as f64;
        let eps = T::Real::EPSILON;
        let dual_tolerance = scalar(overlaps) * eps;
        let bound = scalar(overlaps * (fft as f64).log2().ceil()) * eps;
        let options = StftOptions::new()
            .with_fft_len(Some(fft))
            .with_tolerance(Tolerance::new(dual_tolerance, dual_tolerance).unwrap());
        let plan = StftPlan::new(&window, hop, options).unwrap();
        let a = plan.analyze(&input).unwrap();
        let mut out = vec![T::ZERO; input.len()];
        let mut work = plan.workspace(StftRequest::Synthesis(input.len())).unwrap();
        plan.synthesize_into(&a, &mut out, &mut work, cx()).unwrap();
        for (x, y) in out.iter().zip(input) {
            assert!((*x - y).abs() <= bound);
        }
    }
}
#[test]
fn real_complex_both_precisions() {
    precisions::<f32>();
    precisions::<f64>();
    precisions::<Complex<f32>>();
    precisions::<Complex<f64>>();
}
#[test]
fn stream_chunk_invariance() {
    for m in [1_usize, 3, 8, 17] {
        for hop in [1, m.div_ceil(2), m] {
            for n in [0, 1, 2, 37, 100] {
                let w = vec![1.; m];
                let plan =
                    StftPlan::new(&w, hop, StftOptions::new().with_fft_len(Some(m + 2))).unwrap();
                let input: Vec<_> = (0..n).map(|i| (i as f64).sin()).collect();
                let reference = plan.analyze(&input).unwrap();
                for chunk in [1, 3, 11, 101] {
                    let mut stream = plan.analysis_stream().unwrap();
                    // One reservation serves every chunk of at most `chunk` samples.
                    let mut work = SignalWorkspace::new();
                    work.try_reserve(&stream.workspace_requirements(chunk).unwrap())
                        .unwrap();
                    let mut coefficients = Vec::new();
                    for part in input.chunks(chunk) {
                        let mut out =
                            vec![z(); stream.output_len(part.len(), StreamChunk::Partial).unwrap()];
                        stream
                            .process_into(part, StreamChunk::Partial, &mut out, &mut work, cx())
                            .unwrap();
                        coefficients.extend(out);
                    }
                    let mut tail = vec![z(); stream.output_len(0, StreamChunk::Final).unwrap()];
                    stream
                        .process_into(&[], StreamChunk::Final, &mut tail, &mut work, cx())
                        .unwrap();
                    coefficients.extend(tail);
                    assert_eq!(
                        reference, coefficients,
                        "analysis m={m} h={hop} n={n} chunk={chunk}"
                    );
                    assert_eq!(
                        error_kind(stream.output_len(0, StreamChunk::Final)),
                        SciOxideErrorKind::InvalidInput
                    );
                    stream.reset();
                    let mut stream = plan.synthesis_stream(n).unwrap();
                    let mut work = SignalWorkspace::new();
                    work.try_reserve(&stream.workspace_requirements(plan.bins() * chunk).unwrap())
                        .unwrap();
                    let mut samples = Vec::new();
                    for part in coefficients.chunks(plan.bins() * chunk) {
                        let mut out =
                            vec![0.; stream.output_len(part.len(), StreamChunk::Partial).unwrap()];
                        stream
                            .process_into(part, StreamChunk::Partial, &mut out, &mut work, cx())
                            .unwrap();
                        samples.extend(out);
                    }
                    let mut out = vec![0.; stream.output_len(0, StreamChunk::Final).unwrap()];
                    stream
                        .process_into(&[], StreamChunk::Final, &mut out, &mut work, cx())
                        .unwrap();
                    samples.extend(out);
                    assert_eq!(samples.len(), n);
                    for (a, b) in samples.iter().zip(&input) {
                        assert!(
                            (a - b).abs() < 2e-12,
                            "synthesis m={m} h={hop} n={n} chunk={chunk} {a} {b}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn dual_constraints_and_optimality() {
    let tol = Tolerance::new(1e-12, 1e-12).unwrap();
    let w = [
        Complex::new(1., 0.2),
        Complex::new(2., -0.3),
        Complex::new(0.7, 0.1),
        Complex::new(1.1, 0.4),
    ];
    let target = [Complex::new(0.5, 0.1); 4];
    for hop in [1, 2, 3, 4] {
        let canonical = canonical_stft_dual(&w, hop, tol).unwrap();
        let closest = closest_stft_dual(&w, hop, &target, DualScale::Fixed, tol).unwrap();
        let optimized = closest_stft_dual(&w, hop, &target, DualScale::Optimized, tol).unwrap();
        let tight = tight_stft_window(&w, hop, tol).unwrap();
        for r in 0..hop {
            for (window, dual) in [
                (&w[..], &canonical[..]),
                (&w[..], closest.dual()),
                (&tight[..], &tight[..]),
            ] {
                let constraint: Complex<f64> = window
                    .iter()
                    .zip(dual)
                    .skip(r)
                    .step_by(hop)
                    .map(|(&a, &b)| a.conj() * b)
                    .sum();
                assert!((constraint - Complex::new(1., 0.)).norm() < 2e-12);
            }
            let residual: Vec<_> = closest
                .dual()
                .iter()
                .zip(target)
                .map(|(&a, b)| a - b)
                .collect();
            let i = r;
            let j = r + hop;
            if j < w.len() {
                let mut direction = [z(); 4];
                direction[i] = w[j].conj();
                direction[j] = -w[i].conj();
                let orth: Complex<f64> = direction
                    .iter()
                    .zip(&residual)
                    .map(|(a, b)| a.conj() * *b)
                    .sum();
                assert!(orth.norm() < 2e-12);
            }
            let constraint: Complex<f64> = w
                .iter()
                .zip(optimized.dual())
                .skip(r)
                .step_by(hop)
                .map(|(&a, &b)| (a * optimized.window_scale()).conj() * b)
                .sum();
            assert!((constraint - Complex::new(1., 0.)).norm() < 2e-12);
        }
    }
    let report = check_overlap(&[1., 2., 3., 4.], 2, tol).unwrap();
    assert!(!report.cola() && report.nola());
    assert!(!check_overlap(&[0., 1., 0., 1.], 2, tol).unwrap().nola());
    let singular = StftPlan::new(
        &[0., 1., 0., 1.],
        2,
        StftOptions::new().with_fft_len(Some(4)),
    )
    .unwrap();
    assert!(singular.analyze(&[1.; 8]).is_ok());
    assert!(singular.dual_window().is_none());
    assert_eq!(
        error_kind(singular.synthesize_into(&[], &mut [], &mut SignalWorkspace::new(), cx())),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(StftPlan::from_windows(
            &[1.; 4],
            &[1.; 4],
            2,
            StftOptions::new().with_fft_len(Some(4))
        )),
        SciOxideErrorKind::InvalidInput
    );
}
#[test]
fn scaling_parseval_and_spectrogram() {
    for m in [7, 8] {
        for scaling in [StftScaling::Raw, StftScaling::Magnitude, StftScaling::Psd] {
            let o = StftOptions::new()
                .with_fft_len(Some(m))
                .with_packing(StftPacking::OneSided)
                .with_scaling(scaling)
                .with_sample_rate(16.)
                .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap());
            let plan = StftPlan::<f64>::new(&vec![1.; m], 2, o).unwrap();
            let input = vec![1.; 33];
            let spectrum = plan.analyze(&input).unwrap();
            let mut power = vec![0.; spectrum.len()];
            let mut work = plan
                .workspace(StftRequest::Spectrogram(input.len()))
                .unwrap();
            plan.spectrogram_into(&input, &mut power, &mut work, cx())
                .unwrap();
            for (p, s) in power.iter().zip(&spectrum) {
                assert!((p - s.norm_sqr()).abs() < 1e-12);
            }
            if scaling == StftScaling::Psd {
                let frame = power
                    .chunks(plan.bins())
                    .find(|v| (v[0] - m as f64 / 16.).abs() < 1e-10)
                    .unwrap();
                assert!((frame.iter().sum::<f64>() * 16. / m as f64 - 1.).abs() < 1e-12);
            }
            let mut out = vec![0.; input.len()];
            let mut work = plan.workspace(StftRequest::Synthesis(input.len())).unwrap();
            plan.synthesize_into(&spectrum, &mut out, &mut work, cx())
                .unwrap();
            assert!(out.iter().all(|v| (v - 1.).abs() < 1e-12));
        }
    }
}
#[test]
fn geometry_and_extreme_values_follow_fft_scaling_policy() {
    let plan = StftPlan::new(&[1.0_f64], 1, StftOptions::new().with_fft_len(Some(1))).unwrap();
    for x in [f64::MAX, f64::from_bits(1)] {
        assert_eq!(
            plan.synthesize(&plan.analyze(&[x]).unwrap(), 1).unwrap(),
            [x]
        );
    }
    let plan = StftPlan::new(&[1.; 4], 2, StftOptions::new().with_fft_len(Some(4))).unwrap();
    assert_eq!(plan.frame_sample_range(-1).unwrap(), -4..0);
    assert_eq!(
        error_kind(plan.frame_sample_range(i64::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
    let mut output = vec![Complex::new(7., 1.); plan.output_len(4).unwrap()];
    let mut work = plan.workspace(StftRequest::Analysis(4)).unwrap();
    plan.analyze_into(
        &[f64::MAX, f64::from_bits(1), 0., 0.],
        &mut output,
        &mut work,
        cx(),
    )
    .unwrap();
    assert!(output.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    let expected = plan.analyze(&[f64::MAX, 0., 0., 0.]).unwrap();
    assert_eq!(output, expected);
    let before = output.clone();
    let error = plan
        .analyze_into(&[f64::NAN, 0., 0., 0.], &mut output, &mut work, cx())
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(output, before);
    let error = plan
        .analyze_into(&[f64::MAX; 4], &mut output, &mut work, cx())
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(output, before);
}
/// Synthesis through a stream fed `frames_per_call` frames at a time; the
/// first failing call's error, if any.
fn stream_synthesis<T: NumericScalar>(
    plan: &StftPlan<T>,
    coefficients: &[Complex<T::Real>],
    n: usize,
    frames_per_call: usize,
) -> Result<Vec<T>, scioxide_core::SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let mut stream = plan.synthesis_stream(n)?;
    let chunk = plan.bins() * frames_per_call;
    let mut ws = stream.workspace(chunk)?;
    let mut samples = Vec::new();
    for part in coefficients.chunks(chunk) {
        let mut out = vec![T::ZERO; stream.output_len(part.len(), StreamChunk::Partial)?];
        stream.process_into(part, StreamChunk::Partial, &mut out, &mut ws, cx())?;
        samples.extend(out);
    }
    let mut out = vec![T::ZERO; stream.output_len(0, StreamChunk::Final)?];
    stream.process_into(&[], StreamChunk::Final, &mut out, &mut ws, cx())?;
    samples.extend(out);
    Ok(samples)
}
fn real_reconstruction_scales<S: RealScalar>(accepted: &[i32], rejected: &[i32])
where
    Complex<S>: NumericScalar<Real = S>,
{
    let value = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let window: Vec<S> = (0..8)
        .map(|i| value((std::f64::consts::PI * (i as f64 + 0.5) / 8.).sin()))
        .collect();
    let x: Vec<S> = (0..32)
        .map(|i| value((i as f64 * 0.77).sin() + 0.3 * (i as f64 * 0.29).cos() + 0.1))
        .collect();
    let eps = S::EPSILON;
    let eta = S::MIN_POSITIVE * S::EPSILON;
    for packing in [
        StftPacking::Full,
        StftPacking::Centered,
        StftPacking::OneSided,
    ] {
        let plan = StftPlan::new(
            &window,
            2,
            StftOptions::new()
                .with_fft_len(Some(8))
                .with_packing(packing),
        )
        .unwrap();
        // Valid real coefficients are accepted down to the smallest subnormal.
        for &k in accepted {
            let scaled: Vec<S> = x.iter().map(|&v| v.scalbn(k)).collect();
            let maximum = scaled.iter().fold(S::ZERO, |m, v| m.max(v.abs()));
            let coefficients = plan.analyze(&scaled).unwrap();
            let restored = plan.synthesize(&coefficients, x.len()).unwrap();
            for (a, b) in restored.iter().zip(&scaled) {
                assert!(
                    (*a - *b).abs() <= value(64.) * eps * maximum + value(64.) * eta,
                    "{packing:?} 2^{k}: {a:?} {b:?}"
                );
            }
            if packing != StftPacking::OneSided {
                for frames in [1, 3] {
                    let streamed = stream_synthesis(&plan, &coefficients, x.len(), frames).unwrap();
                    assert_eq!(streamed, restored, "{packing:?} 2^{k} stream {frames}");
                }
            }
        }
        // Multiplying every coefficient by i reconstructs i*x, which is not
        // real, at every scale: the checks are dimensionless.
        for &k in rejected {
            let scaled: Vec<S> = x.iter().map(|&v| v.scalbn(k)).collect();
            let rotated: Vec<_> = plan
                .analyze(&scaled)
                .unwrap()
                .iter()
                .map(|z| Complex::new(-z.im, z.re))
                .collect();
            assert_eq!(
                error_kind(plan.synthesize(&rotated, x.len())),
                SciOxideErrorKind::DomainError,
                "{packing:?} 2^{k}"
            );
            if packing != StftPacking::OneSided {
                for frames in [1, 3] {
                    assert_eq!(
                        error_kind(stream_synthesis(&plan, &rotated, x.len(), frames)),
                        SciOxideErrorKind::DomainError,
                        "{packing:?} 2^{k} stream {frames}"
                    );
                }
            }
        }
    }
    // A chunk of quiet samples next to loud frames is judged by its
    // overlapping frames, not by the chunk maximum.
    let plan = StftPlan::new(&window, 2, StftOptions::new().with_fft_len(Some(8))).unwrap();
    let mixed: Vec<S> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| v.scalbn(if i < 16 { 40 } else { -40 }))
        .collect();
    let coefficients = plan.analyze(&mixed).unwrap();
    let offline = plan.synthesize(&coefficients, mixed.len()).unwrap();
    for frames in [1, 2, 3] {
        assert_eq!(
            stream_synthesis(&plan, &coefficients, mixed.len(), frames).unwrap(),
            offline
        );
    }
}
#[test]
fn real_reconstruction_checks_are_scale_invariant() {
    real_reconstruction_scales::<f64>(
        &[-1074, -1070, -1050, -1022, -600, -30, 0, 30, 600, 1018],
        &[-1000, -700, -300, -60, -20, 0, 20, 60, 300, 700, 1000],
    );
    real_reconstruction_scales::<f32>(
        &[-149, -145, -130, -126, -60, -10, 0, 10, 60, 120],
        &[-110, -60, -20, 0, 20, 60, 110],
    );
}
#[test]
fn spectrograms_restore_one_native_frame_product() {
    let window: Vec<f64> = (0..16)
        .map(|i| 0.51 - 0.49 * (std::f64::consts::TAU * i as f64 / 16.).cos())
        .collect();
    let options = StftOptions::new()
        .with_fft_len(Some(16))
        .with_packing(StftPacking::OneSided)
        .with_scaling(StftScaling::Psd)
        .with_sample_rate(3.);
    let plan = StftPlan::new(&window, 4, options).unwrap();
    let x: Vec<f64> = (0..97).map(|i| (i as f64 * 0.61).sin() + 0.2).collect();
    let y: Vec<f64> = (0..97).map(|i| (i as f64 * 1.13).cos() - 0.4).collect();
    let n = plan.output_len(x.len()).unwrap();
    let mut ws = plan
        .workspace(StftRequest::CrossSpectrogram(x.len()))
        .unwrap();
    let (a, b) = (plan.analyze(&x).unwrap(), plan.analyze(&y).unwrap());
    // Exceptional frames carry their binary exponents into the products and
    // restore once, so scaled records equal the scaled native products.
    for (kx, ky) in [
        (0, 0),
        (-540, 0),
        (-300, 200),
        (300, -300),
        (500, 0),
        (600, 0),
    ] {
        let xs: Vec<f64> = x.iter().map(|v| v.scalbn(kx)).collect();
        let ys: Vec<f64> = y.iter().map(|v| v.scalbn(ky)).collect();
        let mut power = vec![0.; n];
        let expected: Vec<f64> = a.iter().map(|z| z.norm_sqr().scalbn(2 * kx)).collect();
        let result = plan.spectrogram_into(&xs, &mut power, &mut ws, cx());
        if expected.iter().all(|p| p.is_finite()) {
            result.unwrap();
            let bits = |v: &[f64]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&power), bits(&expected), "spectrogram 2^{kx}");
        } else {
            assert_eq!(error_kind(result), SciOxideErrorKind::NumericalInstability);
        }
        let mut cross = vec![z(); n];
        plan.cross_spectrogram_into(&xs, &ys, &mut cross, &mut ws, cx())
            .unwrap();
        for (i, (&u, &v)) in a.iter().zip(&b).enumerate() {
            let product = u * v.conj();
            let restored = |v: f64| v.scalbn(kx + ky).to_bits();
            assert_eq!(
                (cross[i].re.to_bits(), cross[i].im.to_bits()),
                (restored(product.re), restored(product.im)),
                "cross 2^{kx} 2^{ky} at {i}"
            );
        }
    }
}
#[test]
fn default_options_take_the_fft_length_of_the_window() {
    let (default, new) = (StftOptions::<f64>::default(), StftOptions::<f64>::new());
    assert_eq!((default.fft_len(), new.fft_len()), (None, None));
    assert_eq!(default.sample_rate(), new.sample_rate());
    assert_eq!(
        (default.packing(), default.boundary()),
        (new.packing(), new.boundary())
    );
    assert_eq!(
        (default.scaling(), default.tolerance()),
        (new.scaling(), new.tolerance())
    );
    assert_eq!(default.execution_policy(), new.execution_policy());
    assert_eq!(
        (default.sample_rate(), default.packing()),
        (1., StftPacking::Full)
    );
    let window = [0.2, 0.5, 1., 0.7, 0.4];
    let input: Vec<_> = (0..23).map(|i| (i as f64 * 0.37).sin()).collect();
    let implicit = StftPlan::new(&window, 2, new).unwrap();
    let explicit = StftPlan::new(&window, 2, new.with_fft_len(Some(5))).unwrap();
    assert_eq!(
        (implicit.hop(), implicit.fft_len(), implicit.bins()),
        (2, 5, 5)
    );
    assert_eq!(implicit.options().fft_len(), None);
    assert_eq!(explicit.options().fft_len(), Some(5));
    assert_eq!(
        implicit.analyze(&input).unwrap(),
        explicit.analyze(&input).unwrap()
    );
    let padded = StftPlan::new(&window, 3, new.with_fft_len(Some(8))).unwrap();
    assert_eq!((padded.hop(), padded.fft_len(), padded.bins()), (3, 8, 8));
    let dual = implicit.dual_window().unwrap();
    let from_dual = StftPlan::from_dual(dual, 2, new).unwrap();
    assert_eq!((from_dual.hop(), from_dual.fft_len()), (2, 5));
    let restored = from_dual
        .synthesize(&from_dual.analyze(&input).unwrap(), input.len())
        .unwrap();
    assert!(
        restored
            .iter()
            .zip(&input)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
}
