//! Independent finite DFT and weighted least-squares tests for spectral estimators.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_signal::*;
fn direct(x: &[Complex<f64>], window: &[f64], k: usize, fft: usize) -> Complex<f64> {
    x.iter()
        .zip(window)
        .enumerate()
        .map(|(j, (&x, &w))| {
            let t = -2. * std::f64::consts::PI * k as f64 * j as f64 / fft as f64;
            x * Complex::new(t.cos(), t.sin()) * w
        })
        .sum()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1. + b.abs()), "{a} != {b}");
}
fn domains<T: NumericScalar>()
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let r = |v| T::Real::try_from_f64_lossy(v).unwrap();
    let tolerance = if T::Real::EPSILON.into_f64() > 1e-10 {
        2e-5
    } else {
        2e-12
    };
    let input: Vec<T> = (0..23)
        .map(|i| T::from_components(r((i as f64 * 0.7).sin() + 0.2), r((i as f64 * 0.4).cos())))
        .collect();
    let reference: Vec<_> = input
        .iter()
        .map(|x| {
            Complex::new(
                x.real_component().into_f64(),
                x.imaginary_component().into_f64(),
            )
        })
        .collect();
    let complex = T::from_components(r(0.), r(1.)).imaginary_component() != r(0.);
    for fft in [7, 8, 11] {
        for packing in [
            StftPacking::Full,
            StftPacking::Centered,
            StftPacking::OneSided,
        ] {
            if complex && packing == StftPacking::OneSided {
                continue;
            }
            for scaling in [SpectralScaling::Density, SpectralScaling::Spectrum] {
                for detrend in [None, Some(DetrendKind::Constant), Some(DetrendKind::Linear)] {
                    let w = [0.2, 0.5, 1., 0.8, 0.3];
                    let options = SpectralEstimateOptions::new()
                        .with_fft_len(Some(fft))
                        .with_sample_rate(r(2.5))
                        .with_overlap(2)
                        .with_packing(packing)
                        .with_scaling(scaling)
                        .with_detrend(detrend);
                    let plan = SpectralEstimatePlan::<T>::new(&w.map(r), options).unwrap();
                    let result = plan.power(&input).unwrap();
                    let count = 1 + (input.len() - w.len()) / 3;
                    assert_eq!(plan.segments(input.len()).unwrap(), count);
                    for (k, &actual) in result.iter().enumerate() {
                        let bin = if packing == StftPacking::Centered {
                            (k + fft.div_ceil(2)) % fft
                        } else {
                            k
                        };
                        let mut expected = 0.;
                        for p in 0..count {
                            let mut segment = reference[p * 3..p * 3 + 5].to_vec();
                            if let Some(kind) = detrend {
                                let mean: Complex<f64> =
                                    segment.iter().copied().sum::<Complex<f64>>() / 5.;
                                let slope = if matches!(kind, DetrendKind::Linear) {
                                    segment
                                        .iter()
                                        .enumerate()
                                        .map(|(j, &v)| v * (j as f64 - 2.))
                                        .sum::<Complex<f64>>()
                                        / 10.
                                } else {
                                    Complex::new(0., 0.)
                                };
                                for (j, v) in segment.iter_mut().enumerate() {
                                    *v = *v - mean - slope * (j as f64 - 2.);
                                }
                            }
                            let z = direct(&segment, &w, bin, fft);
                            let denom = if scaling == SpectralScaling::Density {
                                2.5 * w.iter().map(|x| x * x).sum::<f64>()
                            } else {
                                w.iter().sum::<f64>().powi(2)
                            };
                            expected += (z.re * z.re + z.im * z.im) / denom / count as f64;
                        }
                        if packing == StftPacking::OneSided
                            && k != 0
                            && !(fft % 2 == 0 && k == fft / 2)
                        {
                            expected *= 2.;
                        }
                        close(actual.into_f64(), expected, tolerance);
                    }
                }
            }
        }
    }
}
#[test]
fn all_scalar_domains_against_dft() {
    domains::<f32>();
    domains::<f64>();
    domains::<Complex<f32>>();
    domains::<Complex<f64>>();
}
#[test]
fn parseval_endpoints_and_periodogram() {
    for n in [7, 8] {
        let x: Vec<_> = (0..n).map(|j| (j as f64 * 0.7).cos()).collect();
        let w: Vec<_> = (0..n).map(|j| 0.2 + j as f64 / n as f64).collect();
        let o = SpectralEstimateOptions::new()
            .with_fft_len(Some(n * 2))
            .with_sample_rate(4.)
            .with_detrend(None)
            .with_packing(StftPacking::OneSided);
        let p = SpectralEstimatePlan::<f64>::new(&w, o).unwrap();
        // A window-length record is a single segment: the periodogram.
        let spectrum = p.power(&x).unwrap();
        let power = spectrum.iter().sum::<f64>() * 4. / (n * 2) as f64;
        let expected = x.iter().zip(&w).map(|(a, b)| (a * b).powi(2)).sum::<f64>()
            / w.iter().map(|v| v * v).sum::<f64>();
        close(power, expected, 2e-12);
    }
}
#[test]
fn median_bias_and_trailing_segments() {
    for values in [&[1., 2., 10.][..], &[1., 2., 3., 10.][..]] {
        let o = SpectralEstimateOptions::new()
            .with_fft_len(Some(1))
            .with_sample_rate(1.)
            .with_detrend(None)
            .with_average(SpectralAverage::Median);
        let p = SpectralEstimatePlan::<f64>::new(&[1.], o).unwrap();
        let expected = if values.len() == 3 { 4. } else { 6.5 } / (5. / 6.);
        close(p.power(values).unwrap()[0], expected, 1e-14);
    }
    let o = SpectralEstimateOptions::new()
        .with_fft_len(Some(4))
        .with_sample_rate(1.)
        .with_detrend(None);
    let p = SpectralEstimatePlan::<f64>::new(&[1.; 4], o).unwrap();
    assert_eq!(
        p.power(&[1., 2., 3., 4., 999.]).unwrap(),
        p.power(&[1., 2., 3., 4.]).unwrap()
    );
}
#[test]
fn complex_csd_orientation_and_coherence() {
    let x: Vec<_> = (0..47)
        .map(|i| Complex::new((i as f64 * 0.32).sin(), (i as f64 * 0.71).cos()))
        .collect();
    let gain = Complex::new(0.4, 0.7);
    let y: Vec<_> = x.iter().map(|&v| v * gain).collect();
    let o = SpectralEstimateOptions::new()
        .with_fft_len(Some(9))
        .with_sample_rate(2.)
        .with_overlap(3)
        .with_detrend(None);
    let p = SpectralEstimatePlan::<Complex<f64>>::new(&[1.; 7], o).unwrap();
    let power = p.power(&x).unwrap();
    let csd = p.csd(&x, &y).unwrap();
    let coherence = p.coherence(&x, &y).unwrap();
    for ((&a, &z), &c) in power.iter().zip(&csd).zip(&coherence) {
        close(z.re, a * gain.re, 2e-12);
        close(z.im, -a * gain.im, 2e-12);
        close(c, 1., 2e-12);
    }
    let reverse = p.csd(&y, &x).unwrap();
    for (a, b) in csd.iter().zip(reverse) {
        close(a.re, b.re, 2e-12);
        close(a.im, -b.im, 2e-12);
    }
}
/// Bitwise components of real values.
fn bits<S: RealScalar>(values: impl IntoIterator<Item = S>) -> Vec<u64> {
    values.into_iter().map(|v| v.into_f64().to_bits()).collect()
}
fn separately_scaled_records<S: RealScalar>(exponents: &[i32])
where
    Complex<S>: NumericScalar<Real = S>,
{
    let value = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let window: Vec<S> = (0..16)
        .map(|i| {
            value(
                (std::f64::consts::PI * (i as f64 + 0.5) / 16.)
                    .sin()
                    .powi(2),
            )
        })
        .collect();
    let x: Vec<S> = (0..64)
        .map(|i| value((i as f64 * 0.83).sin() + 0.4 * (i as f64 * 0.31).cos()))
        .collect();
    let y: Vec<S> = (0..64)
        .map(|i| value(0.6 * (i as f64 * 0.83).sin() + 0.5 * (i as f64 * 1.37).cos()))
        .collect();
    let scaled = |v: &[S], k: i32| v.iter().map(|&v| v.scalbn(k)).collect::<Vec<_>>();
    for (packing, scaling, detrend) in [
        (
            StftPacking::OneSided,
            SpectralScaling::Density,
            Some(DetrendKind::Constant),
        ),
        (
            StftPacking::Full,
            SpectralScaling::Spectrum,
            Some(DetrendKind::Linear),
        ),
        (StftPacking::Centered, SpectralScaling::Density, None),
    ] {
        let options = SpectralEstimateOptions::new()
            .with_fft_len(Some(16))
            .with_sample_rate(value(3.))
            .with_overlap(8)
            .with_packing(packing)
            .with_scaling(scaling)
            .with_detrend(detrend);
        let plan = SpectralEstimatePlan::<S>::new(&window, options).unwrap();
        let coherence = plan.coherence(&x, &y).unwrap();
        let identical = plan.coherence(&x, &x).unwrap();
        let power = plan.power(&x).unwrap();
        let csd = plan.csd(&x, &y).unwrap();
        for &kx in exponents {
            let xs = scaled(&x, kx);
            // Power and CSD are the unscaled estimates restored once per bin.
            let expected: Vec<_> = power.iter().map(|p| p.scalbn(2 * kx)).collect();
            if expected.iter().all(|p| p.is_finite()) {
                assert_eq!(
                    bits(plan.power(&xs).unwrap()),
                    bits(expected),
                    "power 2^{kx}"
                );
            } else {
                assert_eq!(
                    error_kind(plan.power(&xs)),
                    SciOxideErrorKind::NumericalInstability
                );
            }
            // Proportional records have unit coherence at every scale.
            assert_eq!(
                bits(plan.coherence(&x, &xs).unwrap()),
                bits(identical.iter().copied()),
                "{packing:?} coherence(x, x*2^{kx})"
            );
            for &ky in exponents {
                let ys = scaled(&y, ky);
                assert_eq!(
                    bits(plan.coherence(&xs, &ys).unwrap()),
                    bits(coherence.iter().copied()),
                    "{packing:?} coherence(x*2^{kx}, y*2^{ky})"
                );
                let expected: Vec<_> = csd
                    .iter()
                    .map(|z| Complex::new(z.re.scalbn(kx + ky), z.im.scalbn(kx + ky)))
                    .collect();
                if expected
                    .iter()
                    .all(|z| z.re.is_finite() && z.im.is_finite())
                {
                    let actual = plan.csd(&xs, &ys).unwrap();
                    assert_eq!(
                        bits(actual.iter().flat_map(|z| [z.re, z.im])),
                        bits(expected.iter().flat_map(|z| [z.re, z.im])),
                        "{packing:?} csd(x*2^{kx}, y*2^{ky})"
                    );
                }
            }
        }
        for &c in &identical {
            assert!((c - S::ONE).abs() <= value(64.) * S::EPSILON);
        }
        // Coherence is undefined where an auto-power is genuinely zero.
        assert_eq!(
            error_kind(plan.coherence(&x, &vec![S::ZERO; x.len()])),
            SciOxideErrorKind::DomainError
        );
    }
}
#[test]
fn coherence_is_invariant_under_separate_power_of_two_scaling() {
    separately_scaled_records::<f64>(&[
        -1000, -600, -540, -530, -515, -300, -60, 0, 60, 300, 505, 512, 600, 1000,
    ]);
    separately_scaled_records::<f32>(&[-100, -80, -75, -70, -65, -30, 0, 30, 60, 64, 100]);
}
// Independent normal-equation elimination used ONLY as an oracle on well-conditioned fixtures.
fn reference_lomb(
    times: &[f64],
    y: &[f64],
    weights: &[f64],
    omega: f64,
    floating: bool,
) -> (f64, f64) {
    let d = if floating { 3 } else { 2 };
    let mut a = [[0.; 4]; 3];
    for ((&t, &v), &w) in times.iter().zip(y).zip(weights) {
        let row = [(omega * t).cos(), (omega * t).sin(), 1.];
        for i in 0..d {
            for j in 0..d {
                a[i][j] += w * row[i] * row[j];
            }
            a[i][d] += w * row[i] * v;
        }
    }
    for k in 0..d {
        let pivot = a[k][k];
        for value in &mut a[k][k..=d] {
            *value /= pivot;
        }
        let pivot_row = a[k];
        for (i, row) in a.iter_mut().enumerate().take(d) {
            if i != k {
                let factor = row[k];
                for (value, &pivot_value) in row[k..=d].iter_mut().zip(&pivot_row[k..=d]) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }
    let mean = if floating {
        y.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / weights.iter().sum::<f64>()
    } else {
        0.
    };
    let base = y
        .iter()
        .zip(weights)
        .map(|(v, w)| w * (v - mean).powi(2))
        .sum::<f64>();
    let residual = times
        .iter()
        .zip(y)
        .zip(weights)
        .map(|((&t, &v), &w)| {
            w * (v
                - a[0][d] * (omega * t).cos()
                - a[1][d] * (omega * t).sin()
                - if floating { a[2][d] } else { 0. })
            .powi(2)
        })
        .sum::<f64>();
    (
        (base - residual) / base,
        (base - residual) / weights.iter().sum::<f64>() * y.len() as f64 / 2.,
    )
}
fn lomb_precision<S: RealScalar>() {
    let r = |v| S::try_from_f64_lossy(v).unwrap();
    let t: Vec<_> = (0..37)
        .map(|i| r(i as f64 * 0.31 + (i % 3) as f64 * 0.07))
        .collect();
    let t64: Vec<_> = t.iter().map(|v| v.into_f64()).collect();
    let y: Vec<_> = t64
        .iter()
        .map(|t| r(1.2 + (1.7 * t).cos() + 0.3 * (0.8 * t).sin()))
        .collect();
    let y64: Vec<_> = y.iter().map(|v| v.into_f64()).collect();
    let w: Vec<_> = (0..t.len()).map(|i| r((i % 4) as f64)).collect();
    let w64: Vec<_> = w.iter().map(|v| v.into_f64()).collect();
    let f = [0.3, 0.8, 1.7, 3.2].map(r);
    let tol = if S::EPSILON.into_f64() > 1e-10 {
        2e-5
    } else {
        2e-12
    };
    for floating_mean in [false, true] {
        for normalization in [
            LombScargleNormalization::Normalized,
            LombScargleNormalization::Power,
        ] {
            let options = LombScargleOptions::default()
                .with_mean(if floating_mean {
                    LombScargleMean::Floating
                } else {
                    LombScargleMean::Zero
                })
                .with_normalization(normalization);
            let p = LombScargle::new(&t, &f, Some(&w), options).unwrap();
            for (&v, &omega) in p.power(&y).unwrap().iter().zip(&f) {
                let (norm, power) =
                    reference_lomb(&t64, &y64, &w64, omega.into_f64(), floating_mean);
                close(
                    v.into_f64(),
                    if normalization == LombScargleNormalization::Normalized {
                        norm
                    } else {
                        power
                    },
                    tol,
                );
            }
        }
    }
}
#[test]
fn lomb_weighted_independent_least_squares() {
    lomb_precision::<f32>();
    lomb_precision::<f64>();
}
#[test]
fn lomb_irregular_recovery_aliases_and_time_translation() {
    let t: Vec<_> = (0..50)
        .map(|i| i as f64 * 0.23 + (i % 3) as f64 * 0.04)
        .collect();
    let f = [0.7, 1.3, 2.1, 7.];
    let y: Vec<_> = t
        .iter()
        .map(|t| 3. + 2. * (1.3 * t).cos() - 0.8 * (1.3 * t).sin())
        .collect();
    let p = LombScargle::new(&t, &f, None, Default::default()).unwrap();
    let a = p.power(&y).unwrap();
    close(a[1], 1., 1e-13);
    assert!(a[0] < 0.5 && a[2] < 0.5);
    let shifted: Vec<_> = t.iter().map(|v| v + 1024.).collect();
    let b = LombScargle::new(&shifted, &f, None, Default::default())
        .unwrap()
        .power(&y)
        .unwrap();
    for (a, b) in a.iter().zip(b) {
        close(*a, b, 2e-12);
    }
    let regular: Vec<_> = (0..50).map(|i| i as f64).collect();
    let y: Vec<_> = regular.iter().map(|t| (0.7 * t).cos()).collect();
    let aliases = [0.7, 2. * std::f64::consts::PI - 0.7];
    let powers = LombScargle::new(&regular, &aliases, None, Default::default())
        .unwrap()
        .power(&y)
        .unwrap();
    for p in powers {
        close(p, 1., 1e-12);
    }
}
#[test]
fn lomb_constant_rank_and_relative_weight_conventions() {
    let t = [0., 0.3, 1.2, 2.7, 4.1];
    let f = [0.7, 1.4];
    let w = [1., 2., 3., 4., 5.];
    let p = LombScargle::new(&t, &f, Some(&w), Default::default()).unwrap();
    // A constant record has zero energy after the floating mean is removed.
    assert_eq!(
        error_kind(p.power(&[3.; 5])),
        SciOxideErrorKind::DomainError
    );
    let y = [1., 3., -2., 0.5, 4.];
    let a = p.power(&y).unwrap();
    let scaled = w.map(|v| v * 1e200);
    let b = LombScargle::new(&t, &f, Some(&scaled), Default::default())
        .unwrap()
        .power(&y)
        .unwrap();
    for (a, b) in a.iter().zip(b) {
        close(*a, b, 2e-12);
    }
    let options = LombScargleOptions::default().with_normalization(LombScargleNormalization::Power);
    assert_eq!(
        LombScargle::new(&t, &f, None, options)
            .unwrap()
            .power(&[0.; 5])
            .unwrap(),
        [0., 0.]
    );
}
#[test]
fn default_options_take_the_fft_length_of_the_window() {
    let (default, new) = (
        SpectralEstimateOptions::<f64>::default(),
        SpectralEstimateOptions::<f64>::new(),
    );
    assert_eq!((default.fft_len(), new.fft_len()), (None, None));
    assert_eq!((default.overlap(), default.sample_rate()), (0, 1.));
    assert_eq!((new.overlap(), new.sample_rate()), (0, 1.));
    assert_eq!(
        (default.packing(), default.scaling()),
        (new.packing(), new.scaling())
    );
    assert_eq!(
        (default.packing(), default.scaling()),
        (StftPacking::Full, SpectralScaling::Density)
    );
    assert!(matches!(default.detrend(), Some(DetrendKind::Constant)));
    assert!(matches!(new.detrend(), Some(DetrendKind::Constant)));
    assert_eq!(
        (default.average(), new.average()),
        (SpectralAverage::Mean, SpectralAverage::Mean)
    );
    assert_eq!(default.execution_policy(), new.execution_policy());
    let window = [0.2, 0.5, 1., 0.8, 0.3];
    let x: Vec<_> = (0..17).map(|i| (i as f64 * 0.41).cos()).collect();
    let options = new.with_overlap(2).with_sample_rate(4.);
    let implicit = SpectralEstimatePlan::<f64>::new(&window, options).unwrap();
    let explicit =
        SpectralEstimatePlan::<f64>::new(&window, options.with_fft_len(Some(5))).unwrap();
    assert_eq!((implicit.fft_len(), implicit.bins()), (5, 5));
    assert_eq!(implicit.options().fft_len(), None);
    assert_eq!(implicit.power(&x).unwrap(), explicit.power(&x).unwrap());
    assert!((implicit.frequency(1).unwrap() - 0.8).abs() < 1e-15);
    let padded = SpectralEstimatePlan::<f64>::new(&window, options.with_fft_len(Some(9))).unwrap();
    assert_eq!((padded.fft_len(), padded.bins()), (9, 9));
    assert_eq!(
        SpectralEstimatePlan::<f64>::discrete(6).unwrap().fft_len(),
        6
    );
}
