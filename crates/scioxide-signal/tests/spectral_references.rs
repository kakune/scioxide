//! Independent SciPy fixtures for forward framing, detrended framing, synthesis
//! and spectral reductions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar};
use scioxide_signal::*;

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}
fn values<T: NumericScalar>(line: &str) -> Vec<T> {
    numbers(line)
        .chunks_exact(2)
        .map(|v| {
            T::from_components(
                T::Real::try_from_f64_lossy(v[0]).unwrap(),
                T::Real::try_from_f64_lossy(v[1]).unwrap(),
            )
        })
        .collect()
}
fn packing(value: f64) -> StftPacking {
    [
        StftPacking::Full,
        StftPacking::Centered,
        StftPacking::OneSided,
    ][value as usize]
}
fn close<T: NumericScalar>(actual: T, expected: Complex<f64>) {
    let actual = Complex::new(
        actual.real_component().into_f64(),
        actual.imaginary_component().into_f64(),
    );
    let tolerance = if T::Real::MANTISSA_DIGITS == 24 {
        128.0
    } else {
        4096.0
    } * T::Real::EPSILON.into_f64();
    assert!(
        (actual - expected).norm() <= tolerance * expected.norm().max(1.0),
        "{actual:?} != {expected:?}"
    );
}
fn spectral<T: NumericScalar>()
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let lines: Vec<_> = include_str!("fixtures/spectral_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(7) {
        let h = numbers(case[0]);
        if (h[7] != 0.0) == T::IS_REAL {
            continue;
        }
        let x = values::<T>(case[1]);
        let y = values::<T>(case[2]);
        let window: Vec<_> = numbers(case[3])
            .into_iter()
            .map(|v| T::Real::try_from_f64_lossy(v).unwrap())
            .collect();
        let options = SpectralEstimateOptions::new()
            .with_fft_len(Some(h[1] as usize))
            .with_sample_rate(T::Real::try_from_f64_exact(2.0).unwrap())
            .with_overlap(h[2] as usize)
            .with_packing(packing(h[3]))
            .with_scaling(if h[4] == 0.0 {
                SpectralScaling::Density
            } else {
                SpectralScaling::Spectrum
            })
            .with_detrend(
                [None, Some(DetrendKind::Constant), Some(DetrendKind::Linear)][h[5] as usize],
            )
            .with_average(if h[6] == 0.0 {
                SpectralAverage::Mean
            } else {
                SpectralAverage::Median
            });
        let plan = SpectralEstimatePlan::<T>::new(&window, options).unwrap();
        let power = plan.power(&x).unwrap();
        for (a, b) in power.into_iter().zip(numbers(case[4])) {
            close(a, Complex::new(b, 0.0));
        }
        for (a, b) in plan
            .csd(&x, &y)
            .unwrap()
            .into_iter()
            .zip(values::<Complex<f64>>(case[5]))
        {
            close(a, b);
        }
        if options.average() == SpectralAverage::Mean {
            for (a, b) in plan
                .coherence(&x, &y)
                .unwrap()
                .into_iter()
                .zip(numbers(case[6]))
            {
                close(a, Complex::new(b, 0.0));
            }
        }
    }
}
fn stft<T: NumericScalar>()
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let lines: Vec<_> = include_str!("fixtures/stft_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(5) {
        let h = numbers(case[0]);
        if (h[6] != 0.0) == T::IS_REAL {
            continue;
        }
        let x = values::<T>(case[1]);
        let window: Vec<_> = numbers(case[2])
            .into_iter()
            .map(|v| T::from_real(T::Real::try_from_f64_lossy(v).unwrap()))
            .collect();
        let options = StftOptions::new()
            .with_fft_len(Some(h[1] as usize))
            .with_sample_rate(T::Real::try_from_f64_exact(2.0).unwrap())
            .with_packing(packing(h[3]))
            .with_scaling(
                [StftScaling::Raw, StftScaling::Magnitude, StftScaling::Psd][h[4] as usize],
            )
            .with_boundary(
                [
                    SignalBoundary::Zero,
                    SignalBoundary::Periodic,
                    SignalBoundary::Reflect,
                ][h[5] as usize],
            );
        let plan = StftPlan::new(&window, h[2] as usize, options).unwrap();
        let frames = plan.frames(x.len()).unwrap();
        assert_eq!(frames.first(), h[7] as i64);
        assert_eq!(frames.count(), h[8] as usize);
        let expected = values::<Complex<f64>>(case[3]);
        let actual = plan.analyze(&x).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(&expected) {
            close(a, b);
        }
        // Invert independently generated coefficients, rather than relying on a round trip.
        let coefficients = values::<Complex<T::Real>>(case[3]);
        let mut output = vec![T::ZERO; x.len()];
        let mut scratch = plan.workspace(StftRequest::Synthesis(x.len())).unwrap();
        plan.synthesize_into(
            &coefficients,
            &mut output,
            &mut scratch,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        for (a, b) in output.into_iter().zip(values::<Complex<f64>>(case[4])) {
            close(a, b);
        }
    }
}
/// `ShortTimeFFT.stft_detrend` detrends each padded frame before windowing.
fn stft_detrended<T: NumericScalar>()
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let lines: Vec<_> = include_str!("fixtures/stft_detrend_scipy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .collect();
    for case in lines.chunks_exact(4) {
        let h = numbers(case[0]);
        if (h[6] != 0.0) == T::IS_REAL {
            continue;
        }
        let x = values::<T>(case[1]);
        let window: Vec<_> = numbers(case[2])
            .into_iter()
            .map(|v| T::from_real(T::Real::try_from_f64_lossy(v).unwrap()))
            .collect();
        let options = StftOptions::new()
            .with_fft_len(Some(h[1] as usize))
            .with_sample_rate(T::Real::try_from_f64_exact(2.0).unwrap())
            .with_packing(packing(h[3]))
            .with_scaling(
                [StftScaling::Raw, StftScaling::Magnitude, StftScaling::Psd][h[4] as usize],
            )
            .with_boundary(
                [
                    SignalBoundary::Zero,
                    SignalBoundary::Periodic,
                    SignalBoundary::Reflect,
                ][h[5] as usize],
            );
        let kind = [DetrendKind::Constant, DetrendKind::Linear][h[7] as usize];
        let plan = StftPlan::new(&window, h[2] as usize, options).unwrap();
        let frames = plan.frames(x.len()).unwrap();
        assert_eq!(frames.first(), h[8] as i64);
        assert_eq!(frames.count(), h[9] as usize);
        let expected = values::<Complex<f64>>(case[3]);
        let mut actual = vec![Complex::<T::Real>::ZERO; plan.output_len(x.len()).unwrap()];
        let mut scratch = plan
            .workspace(StftRequest::DetrendedAnalysis(x.len()))
            .unwrap();
        plan.analyze_detrended_into(
            &x,
            kind,
            &mut actual,
            &mut scratch,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(&expected) {
            close(a, b);
        }
    }
}
#[test]
fn scipy_spectral_all_scalar_domains() {
    spectral::<f64>();
    spectral::<f32>();
    spectral::<Complex<f64>>();
    spectral::<Complex<f32>>();
}
#[test]
fn scipy_stft_and_independent_istft_all_scalar_domains() {
    stft::<f64>();
    stft::<f32>();
    stft::<Complex<f64>>();
    stft::<Complex<f32>>();
}
#[test]
fn scipy_detrended_stft_all_scalar_domains() {
    stft_detrended::<f64>();
    stft_detrended::<f32>();
    stft_detrended::<Complex<f64>>();
    stft_detrended::<Complex<f32>>();
}
