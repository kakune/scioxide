//! Transform families preserve finite results and atomic failures at extreme scales.
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/owned.rs"]
mod owned;
use owned::ExecuteOwned;
use scioxide_core::{
    Complex, NdView, NdViewMut, RealScalar, SciOxideError, SciOxideErrorView, StridedLayout,
};
use scioxide_fft::*;

fn cx() -> scioxide_core::ExecutionContext<'static> {
    FftPlanningContext::<f64>::default().execution()
}

fn close(actual: f64, expected: f64, exponent: i32, n: usize) {
    // Compare in the input frame, avoiding overflow in error calculations.
    // Include the final subnormal grid, which can dominate relative error.
    let error = (actual.scalbn(-exponent) - expected).abs();
    let bound = 512.0 * f64::EPSILON * n as f64 * expected.abs().max(1.0)
        + 2.0 * f64::from_bits(1).scalbn(-exponent);
    numerics::assert_close(
        error,
        bound,
        format_args!("actual={actual:e}, expected={expected:e}, exponent={exponent}"),
    );
}

fn fourier(input: &[Complex<f64>], turns: impl Fn(usize, usize) -> f64) -> Vec<Complex<f64>> {
    (0..input.len())
        .map(|k| numerics::direct_sum(input, |j| (-core::f64::consts::TAU * turns(j, k), 1.0)))
        .collect()
}

#[test]
fn fourier_families_match_direct_definitions_at_extreme_scales() {
    for n in [7, 8, 11] {
        for exponent in [1000, -995, -1070] {
            let input: Vec<_> = (0..n)
                .map(|j| {
                    Complex::new(
                        ((j % 3) as f64 - 1.0).scalbn(exponent),
                        ((j % 5) as f64 * 0.25 - 0.5).scalbn(exponent),
                    )
                })
                .collect();
            let bounded: Vec<_> = input
                .iter()
                .map(|z| Complex::new(z.re.scalbn(-exponent), z.im.scalbn(-exponent)))
                .collect();
            let reference = fourier(&bounded, |j, k| (j * k) as f64 / n as f64);
            for normalization in [
                Normalization::Unscaled,
                Normalization::Forward,
                Normalization::Orthonormal,
            ] {
                let factor = match normalization {
                    Normalization::Forward => 1.0 / n as f64,
                    Normalization::Orthonormal => 1.0 / (n as f64).sqrt(),
                    _ => 1.0,
                };
                let actual = fft(&input, normalization).unwrap();
                for (z, r) in actual.iter().zip(&reference) {
                    close(z.re, r.re * factor, exponent, n);
                    close(z.im, r.im * factor, exponent, n);
                }
                let real: Vec<_> = input.iter().map(|z| z.re).collect();
                let real_bounded: Vec<_> =
                    bounded.iter().map(|z| Complex::new(z.re, 0.0)).collect();
                let reference = fourier(&real_bounded, |j, k| (j * k) as f64 / n as f64);
                let actual = rfft(&real, normalization).unwrap();
                for (z, r) in actual.bins().iter().zip(&reference) {
                    close(z.re, r.re * factor, exponent, n);
                    close(z.im, r.im * factor, exponent, n);
                }
                let recovered = irfft(&actual, Normalization::Inverse).unwrap();
                for (a, r) in recovered.iter().zip(real_bounded) {
                    close(*a, r.re * factor, exponent, n);
                }
            }
            let plan = CztPlan::new(CztPlanSpec::new(
                n,
                n,
                CztContour::new(0.0, 0.0, 0.125, 0.03125).unwrap(),
                Normalization::Unscaled,
            ))
            .unwrap();
            let reference = fourier(&bounded, |j, k| j as f64 * (0.125 + 0.03125 * k as f64));
            for (z, r) in plan.execute_owned(&input).unwrap().iter().zip(reference) {
                close(z.re, r.re, exponent, n);
                close(z.im, r.im, exponent, n);
            }
        }
    }
}

#[test]
fn cosine_and_sine_match_direct_definitions_at_extreme_scales() {
    let n = 8;
    for exponent in [1000, -995, -1070] {
        let input: Vec<_> = (0..n)
            .map(|j| ((j % 3) as f64 - 1.0).scalbn(exponent))
            .collect();
        let bounded: Vec<_> = input.iter().map(|v| v.scalbn(-exponent)).collect();
        for (ty, kind) in [
            (DctType::I, 1),
            (DctType::II, 2),
            (DctType::III, 3),
            (DctType::IV, 4),
        ] {
            let actual = dct(&input, ty, Normalization::Unscaled).unwrap();
            for (k, a) in actual.iter().enumerate() {
                let expected: f64 = bounded
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let (turns, weight) = match kind {
                            1 => (
                                (j * k) as f64 / (n - 1) as f64,
                                if j == 0 || j + 1 == n { 1.0 } else { 2.0 },
                            ),
                            2 => ((j as f64 + 0.5) * k as f64 / n as f64, 1.0),
                            3 => (
                                j as f64 * (k as f64 + 0.5) / n as f64,
                                if j == 0 { 1.0 } else { 2.0 },
                            ),
                            _ => ((j as f64 + 0.5) * (k as f64 + 0.5) / n as f64, 1.0),
                        };
                        v * weight * (core::f64::consts::PI * turns).cos()
                    })
                    .sum();
                close(*a, expected, exponent, n);
            }
        }
        for (ty, kind) in [
            (DstType::I, 1),
            (DstType::II, 2),
            (DstType::III, 3),
            (DstType::IV, 4),
        ] {
            let actual = dst(&input, ty, Normalization::Unscaled).unwrap();
            for (k, a) in actual.iter().enumerate() {
                let expected: f64 = bounded
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let (turns, weight) = match kind {
                            1 => (((j + 1) * (k + 1)) as f64 / (n + 1) as f64, 1.0),
                            2 => ((j as f64 + 0.5) * (k + 1) as f64 / n as f64, 1.0),
                            3 => (
                                (j + 1) as f64 * (k as f64 + 0.5) / n as f64,
                                if j + 1 == n { 1.0 } else { 2.0 },
                            ),
                            _ => ((j as f64 + 0.5) * (k as f64 + 0.5) / n as f64, 1.0),
                        };
                        v * weight * (core::f64::consts::PI * turns).sin()
                    })
                    .sum();
                close(*a, expected, exponent, n);
            }
        }
    }
}

#[test]
fn tensor_extreme_scales_match_direct_definition() {
    let shape = [2, 4];
    let layout = StridedLayout::new(&shape, &[4, 1]).unwrap();
    let plan = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[0, 1],
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        )
        .unwrap(),
    )
    .unwrap();
    for exponent in [1000, -995, -1070] {
        let input: Vec<_> = (0..8)
            .map(|j| Complex::new(((j % 3) as f64 - 1.0).scalbn(exponent), 0.0))
            .collect();
        let bounded: Vec<_> = input
            .iter()
            .map(|z| Complex::new(z.re.scalbn(-exponent), 0.0))
            .collect();
        let reference = fourier(&bounded, |j, k| {
            ((j / 4) * (k / 4)) as f64 / 2.0 + ((j % 4) * (k % 4)) as f64 / 4.0
        });
        let mut output = vec![Complex::new(0.0, 0.0); 8];
        let mut workspace = plan.workspace(()).unwrap();
        plan.execute_into(
            FftNdInput::Complex(NdView::new(&input, layout).unwrap()),
            FftNdOutput::Complex(NdViewMut::new(&mut output, layout).unwrap()),
            &mut workspace,
            cx(),
        )
        .unwrap();
        for (a, r) in output.iter().zip(reference) {
            close(a.re, r.re, exponent, 8);
            close(a.im, r.im, exponent, 8);
        }
    }
}

fn overflow(error: SciOxideError) {
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ),
        "{error:?}"
    );
}

#[expect(clippy::unwrap_used)]
fn specialized_overflows<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let scalar = |value| S::try_from_f64_lossy(value).unwrap();
    let input = [Complex::new(S::MAX, S::ZERO); 4];
    let sentinel = Complex::new(scalar(17.0), scalar(-23.0));
    let mut output = [sentinel; 4];
    let czt = CztPlan::new(CztPlanSpec::new(
        4,
        4,
        CztContour::dft(4).unwrap(),
        Normalization::Unscaled,
    ))
    .unwrap();
    overflow(
        czt.execute_into(
            &input[..],
            &mut output[..],
            &mut czt.workspace(()).unwrap(),
            cx(),
        )
        .unwrap_err(),
    );
    assert_eq!(output, [sentinel; 4]);

    let zeros: Vec<_> = (1..=5)
        .map(|j| scalar(j as f64 * core::f64::consts::PI))
        .collect();
    let tolerance = scalar(64.0) * S::EPSILON;
    let hankel = DiscreteHankelPlan::new(
        &zeros,
        DiscreteHankelPlanSpec::new(scalar(0.5), scalar(16.0), FftDirection::Forward)
            .with_tolerance(scioxide_core::Tolerance::new(tolerance, tolerance).unwrap()),
    )
    .unwrap();
    overflow(
        hankel
            .execute_into(
                &input,
                &mut output,
                &mut hankel.workspace(()).unwrap(),
                cx(),
            )
            .unwrap_err(),
    );
    assert_eq!(output, [sentinel; 4]);

    let points = [S::ZERO; 4];
    let layout = StridedLayout::new(&[4, 1], &[1, 1]).unwrap();
    for transform in [
        NufftTransform::Type1 {
            points: NdView::new(&points, layout).unwrap(),
            modes: &[4],
        },
        NufftTransform::Type2 {
            points: NdView::new(&points, layout).unwrap(),
            modes: &[4],
        },
        NufftTransform::Type3 {
            points: NdView::new(&points, layout).unwrap(),
            frequencies: NdView::new(&points, layout).unwrap(),
        },
    ] {
        let nufft = NufftPlan::new(
            transform,
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, scalar(1e-3)),
        )
        .unwrap();
        overflow(
            nufft
                .execute_into(
                    &input[..],
                    &mut output[..],
                    &mut nufft.workspace(()).unwrap(),
                    cx(),
                )
                .unwrap_err(),
        );
        assert_eq!(output, [sentinel; 4]);
    }

    let fftlog = FftLogPlan::new(FftLogPlanSpec::new(
        16,
        S::ONE,
        S::ZERO,
        S::ZERO,
        Complex::new(scalar(8.0), S::ZERO),
        FftDirection::Forward,
    ))
    .unwrap();
    let large = if S::EPSILON < scalar(1e-10) {
        1e270
    } else {
        1e22
    };
    let input = [Complex::new(scalar(large), S::ZERO); 16];
    let mut output = [sentinel; 16];
    overflow(
        fftlog
            .execute_into(
                &input,
                &mut output,
                &mut fftlog.workspace(()).unwrap(),
                cx(),
            )
            .unwrap_err(),
    );
    assert_eq!(output, [sentinel; 16]);
}

#[test]
fn specialized_overflows_preserve_output_in_both_precisions() {
    specialized_overflows::<f32>();
    specialized_overflows::<f64>();
}

#[test]
fn public_dc_obeys_fft_rounding_under_cancellation() {
    let tail = 1.0_f64.scalbn(-54);
    for n in [4, 7, 8, 11] {
        let mut input = vec![0.0; n];
        input[..4].copy_from_slice(&[1.0, -1.0, tail, tail]);
        let complex: Vec<_> = input.iter().map(|&x| Complex::new(x, -x)).collect();
        let actual = fft(&complex, Normalization::Unscaled).unwrap();
        close(actual[0].re, 2.0 * tail, 0, n);
        close(actual[0].im, -2.0 * tail, 0, n);
        let actual = rfft(&input, Normalization::Unscaled).unwrap().into_vec();
        close(actual[0].re, 2.0 * tail, 0, n);
        assert_eq!(actual[0].im.to_bits(), 0.0_f64.to_bits());

        let mut spectrum = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        spectrum[0].re = 1.0;
        spectrum[1].re = -0.5;
        spectrum[2].re = tail;
        let expected = if n == 4 { tail } else { 2.0 * tail };
        let actual = irfft(
            &HermitianSpectrum::from_vec(n, spectrum).unwrap(),
            Normalization::Unscaled,
        )
        .unwrap();
        close(actual[0], expected, 0, n);
    }
}

#[test]
fn overflowing_restores_preserve_every_output() {
    for n in [7, 8] {
        let input = vec![Complex::new(f64::MAX, f64::MAX); n];
        let mut output = vec![Complex::new(17.0, -23.0); n];
        let expected = output.clone();
        let plan = FftPlan::new(FftPlanSpec::new(
            n,
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        ))
        .unwrap();
        overflow(
            plan.execute_into(
                &input[..],
                &mut output[..],
                &mut plan.workspace(()).unwrap(),
                cx(),
            )
            .unwrap_err(),
        );
        assert_eq!(output, expected);

        let input = vec![f64::MAX; n];
        let mut bins = vec![Complex::new(17.0, -23.0); n / 2 + 1];
        let expected = bins.clone();
        let plan = FftPlan::new(FftPlanSpec::new(
            n,
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Real,
        ))
        .unwrap();
        overflow(
            plan.execute_into(
                &input[..],
                FftOutput::Hermitian {
                    logical_len: n,
                    bins: scioxide_linalg::VectorViewMut::from_slice(&mut bins),
                },
                &mut plan.workspace(()).unwrap(),
                cx(),
            )
            .unwrap_err(),
        );
        assert_eq!(bins, expected);

        let spectrum = vec![Complex::new(f64::MAX, 0.0); n / 2 + 1];
        let mut output = vec![17.0; n];
        let expected = output.clone();
        let plan = FftPlan::new(FftPlanSpec::new(
            n,
            FftDirection::Inverse,
            Normalization::Unscaled,
            FftLayout::Real,
        ))
        .unwrap();
        overflow(
            plan.execute_into(
                FftInput::Hermitian {
                    logical_len: n,
                    bins: scioxide_linalg::VectorView::from_slice(&spectrum),
                },
                &mut output[..],
                &mut plan.workspace(()).unwrap(),
                cx(),
            )
            .unwrap_err(),
        );
        assert_eq!(output, expected);
    }
    for n in [4, 7, 8] {
        for kind in [DctType::II.into(), DstType::II.into()] {
            let input = vec![f64::MAX; n];
            let mut output = vec![17.0; n];
            let expected = output.clone();
            let plan = RealTransformPlan::new(RealTransformPlanSpec::new(
                n,
                kind,
                FftDirection::Forward,
                Normalization::Unscaled,
            ))
            .unwrap();
            overflow(
                plan.execute_into(&input, &mut output, &mut plan.workspace(()).unwrap(), cx())
                    .unwrap_err(),
            );
            assert_eq!(output, expected);
        }
    }
    let shape = [2, 4];
    let layout = StridedLayout::new(&shape, &[4, 1]).unwrap();
    let input = [Complex::new(f64::MAX, f64::MAX); 8];
    let mut output = [Complex::new(17.0, -23.0); 8];
    let expected = output;
    let plan = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[0, 1],
            FftDirection::Forward,
            Normalization::Unscaled,
            FftLayout::Complex,
        )
        .unwrap(),
    )
    .unwrap();
    overflow(
        plan.execute_into(
            FftNdInput::Complex(NdView::new(&input, layout).unwrap()),
            FftNdOutput::Complex(NdViewMut::new(&mut output, layout).unwrap()),
            &mut plan.workspace(()).unwrap(),
            cx(),
        )
        .unwrap_err(),
    );
    assert_eq!(output, expected);
}

#[test]
fn fused_public_restoration_preserves_underflow_signs() {
    macro_rules! check {
        ($scalar:ty) => {
            let input = [-<$scalar>::from_bits(1), 0.0];
            let complex = input.map(|v| Complex::new(v, 0.0));
            let transformed = fft(&complex, Normalization::Forward).unwrap();
            let real = rfft(&input, Normalization::Forward).unwrap();
            let inverse = irfft(
                &HermitianSpectrum::from_vec(2, complex.to_vec()).unwrap(),
                Normalization::Inverse,
            )
            .unwrap();
            for value in transformed
                .iter()
                .map(|v| v.re)
                .chain(real.bins().iter().map(|v| v.re))
                .chain(inverse)
            {
                assert_eq!(value.to_bits(), (-0.0 as $scalar).to_bits());
            }
            let zero = fft(
                &[Complex::new(-0.0 as $scalar, -0.0); 2],
                Normalization::Unscaled,
            )
            .unwrap();
            for value in zero.iter().flat_map(|v| [v.re, v.im]) {
                assert_eq!(value.to_bits(), (0.0 as $scalar).to_bits());
            }
        };
    }
    check!(f32);
    check!(f64);
}
