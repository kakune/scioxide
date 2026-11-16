#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{Complex, RealScalar, SciOxideErrorKind};
use scioxide_signal::*;
fn s<S: RealScalar>(v: f64) -> S {
    S::try_from_f64_lossy(v).unwrap()
}
fn band<S: RealScalar>(lo: f64, hi: f64, a: f64, b: f64, w: f64) -> FirBand<S> {
    FirBand::new(s(lo), s(hi), s(a), s(b)).with_weight(s(w))
}
fn response(h: &[impl RealScalar], f: f64) -> (f64, f64) {
    h.iter().enumerate().fold((0., 0.), |(re, im), (i, h)| {
        let x = std::f64::consts::PI * f * i as f64;
        (re + h.into_f64() * x.cos(), im - h.into_f64() * x.sin())
    })
}
fn amplitude(h: &[impl RealScalar], f: f64, odd: bool) -> f64 {
    let (re, im) = response(h, f);
    let x = std::f64::consts::PI * f * (h.len() - 1) as f64 / 2.;
    if odd {
        re * x.sin() + im * x.cos()
    } else {
        re * x.cos() - im * x.sin()
    }
}
fn direct<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let tol = if core::mem::size_of::<S>() == 4 {
        3e-5
    } else {
        2e-12
    };
    for (n, sym) in [
        (15, FirSymmetry::Even),
        (16, FirSymmetry::Even),
        (15, FirSymmetry::Odd),
        (16, FirSymmetry::Odd),
    ] {
        let bands = [band::<S>(0.1, 0.4, 1., 1., 1.)];
        let d =
            design_fir_window(n, sym, &bands, WindowKind::Boxcar, FirNormalization::None).unwrap();
        for (i, &v) in d.coefficients().iter().enumerate() {
            let k = (n - 1) as f64 / 2. - i as f64;
            let expected = if k == 0. {
                if sym == FirSymmetry::Even { 0.3 } else { 0. }
            } else if sym == FirSymmetry::Even {
                ((std::f64::consts::PI * k * 0.4).sin() - (std::f64::consts::PI * k * 0.1).sin())
                    / (std::f64::consts::PI * k)
            } else {
                ((std::f64::consts::PI * k * 0.1).cos() - (std::f64::consts::PI * k * 0.4).cos())
                    / (std::f64::consts::PI * k)
            };
            assert!((v.into_f64() - expected).abs() < tol, "{n} {sym:?} {i}");
            assert_eq!(
                v,
                if sym == FirSymmetry::Even {
                    d.coefficients()[n - 1 - i]
                } else {
                    -d.coefficients()[n - 1 - i]
                }
            );
        }
        for f in [0., 0.13, 0.5, 1.] {
            let r = d.response(s(f)).unwrap();
            let (re, im) = response(d.coefficients(), f);
            assert!((r.re.into_f64() - re).abs() < tol);
            assert!((r.im.into_f64() - im).abs() < tol);
        }
    }
    let d = design_fir_window(
        63,
        FirSymmetry::Even,
        &[band::<S>(0., 0.3, 1., 1., 1.)],
        WindowKind::Hamming,
        FirNormalization::At {
            frequency: s(0.),
            gain: s(1.),
        },
    )
    .unwrap();
    assert!((amplitude(d.coefficients(), 0., false) - 1.).abs() < tol);
    assert!(amplitude(d.coefficients(), 0.6, false).abs() < 0.003);
    let p = d.to_polyphase::<Complex<S>>().unwrap();
    assert_eq!(p.coefficients()[0].re, d.coefficients()[0]);
    assert_eq!(p.coefficients()[0].im, s(0.));
}
fn sampling<S: RealScalar>() {
    let tol = if core::mem::size_of::<S>() == 4 {
        2e-5
    } else {
        2e-12
    };
    // A constant response gives a centered unit impulse for Type I.
    let d = design_fir_frequency_sampling(
        15,
        FirSymmetry::Even,
        &[s::<S>(0.), s(1.)],
        &[s(1.), s(1.)],
        128,
        None,
        FirNormalization::None,
    )
    .unwrap();
    for (i, h) in d.coefficients().iter().enumerate() {
        assert!((h.into_f64() - if i == 7 { 1. } else { 0. }).abs() < tol);
    }
    // Fine knot interpolation of an exactly representable sine/cosine response.
    for (n, sym, k) in [
        (15, FirSymmetry::Even, 2.),
        (16, FirSymmetry::Even, 1.5),
        (15, FirSymmetry::Odd, 2.),
        (16, FirSymmetry::Odd, 1.5),
    ] {
        let f: Vec<S> = (0..=512).map(|i| s(i as f64 / 512.)).collect();
        let a: Vec<S> = (0..=512)
            .map(|i| {
                let x = std::f64::consts::PI * k * i as f64 / 512.;
                let v = if sym == FirSymmetry::Even {
                    x.cos()
                } else {
                    x.sin()
                };
                s(
                    if (i == 0 && sym == FirSymmetry::Odd)
                        || (i == 512
                            && ((n % 2 == 0 && sym == FirSymmetry::Even)
                                || (n % 2 == 1 && sym == FirSymmetry::Odd)))
                    {
                        0.
                    } else {
                        v
                    },
                )
            })
            .collect();
        let d = design_fir_frequency_sampling(n, sym, &f, &a, 1024, None, FirNormalization::None)
            .unwrap();
        for i in 1..99 {
            let f = i as f64 / 100.;
            let expected = if sym == FirSymmetry::Even {
                (std::f64::consts::PI * k * f).cos()
            } else {
                (std::f64::consts::PI * k * f).sin()
            };
            assert!(
                (amplitude(d.coefficients(), f, sym == FirSymmetry::Odd) - expected).abs() < tol,
                "sampling {n} {sym:?}"
            );
        }
    }
}
fn least_squares<S: RealScalar>() {
    let tol = if core::mem::size_of::<S>() == 4 {
        2e-5
    } else {
        2e-8
    };
    for (n, sym) in [
        (9, FirSymmetry::Even),
        (10, FirSymmetry::Even),
        (9, FirSymmetry::Odd),
        (10, FirSymmetry::Odd),
    ] {
        let bands = [
            band::<S>(0.05, 0.35, 0.2, 1., 2.),
            band(0.5, 0.95, 0., 0., 5.),
        ];
        let d = design_fir_least_squares(n, sym, &bands).unwrap();
        // Independent midpoint integration of each first-order optimality condition.
        for j in 0..n / 2 + n % 2 * usize::from(sym == FirSymmetry::Even) {
            let k = j as f64
                + if n % 2 == 0 {
                    0.5
                } else if sym == FirSymmetry::Odd {
                    1.
                } else {
                    0.
                };
            let mut gradient = 0.;
            for b in bands {
                let lo = b.lower().into_f64();
                let hi = b.upper().into_f64();
                let step = (hi - lo) / 20000.;
                for i in 0..20000 {
                    let f = lo + (i as f64 + 0.5) * step;
                    let desired = b.desired_lower().into_f64()
                        + (b.desired_upper().into_f64() - b.desired_lower().into_f64()) * (f - lo)
                            / (hi - lo);
                    let basis = if sym == FirSymmetry::Even {
                        (std::f64::consts::PI * k * f).cos()
                    } else {
                        (std::f64::consts::PI * k * f).sin()
                    };
                    gradient += (amplitude(d.coefficients(), f, sym == FirSymmetry::Odd) - desired)
                        * basis
                        * b.weight().into_f64()
                        * step;
                }
            }
            assert!(gradient.abs() < tol, "LS {n} {sym:?} {j}: {gradient}");
        }
    }
}
fn remez<S: RealScalar>() {
    for (n, sym) in [
        (31, FirSymmetry::Even),
        (32, FirSymmetry::Even),
        (31, FirSymmetry::Odd),
        (32, FirSymmetry::Odd),
    ] {
        let bands = if sym == FirSymmetry::Even {
            [band::<S>(0., 0.3, 1., 1., 1.), band(0.45, 1., 0., 0., 8.)]
        } else {
            [
                band::<S>(0.05, 0.3, 1., 1., 1.),
                band(0.45, 0.95, 0., 0., 8.),
            ]
        };
        let d = design_fir_remez_with_options(n, sym, &bands, RemezOptions::default()).unwrap();
        assert_eq!(
            d.termination(),
            FirTermination::Converged,
            "{n} {sym:?} {:?}",
            d.weighted_ripple()
        );
        let ripple = d.weighted_ripple().unwrap().into_f64();
        assert!(ripple < 0.15, "{n} {sym:?} {ripple}");
        let mut actual: f64 = 0.;
        for b in bands {
            for i in 0..=10000 {
                let f = b.lower().into_f64()
                    + (b.upper().into_f64() - b.lower().into_f64()) * i as f64 / 10000.;
                actual = actual.max(
                    (amplitude(d.coefficients(), f, sym == FirSymmetry::Odd)
                        - b.desired_lower().into_f64())
                    .abs()
                        * b.weight().into_f64(),
                );
            }
        }
        assert!(
            actual <= ripple * 1.005 + 2e-5,
            "offgrid {actual} reported {ripple}"
        );
    }
}
#[test]
fn window_f64() {
    direct::<f64>();
}
#[test]
fn window_f32() {
    direct::<f32>();
}
#[test]
fn sampling_f64() {
    sampling::<f64>();
}
#[test]
fn sampling_f32() {
    sampling::<f32>();
}
#[test]
fn least_squares_f64() {
    least_squares::<f64>();
}
#[test]
fn least_squares_f32() {
    least_squares::<f32>();
}
#[test]
fn remez_f64() {
    remez::<f64>();
}
#[test]
fn remez_f32() {
    remez::<f32>();
}
#[test]
fn invalid_and_unconverged() {
    let bands = [band::<f64>(0., 0.3, 1., 1., 1.), band(0.4, 1., 0., 0., 1.)];
    let d = design_fir_remez_with_options(
        31,
        FirSymmetry::Even,
        &bands,
        RemezOptions::default().with_max_iterations(1),
    )
    .unwrap();
    assert_eq!(d.termination(), FirTermination::IterationLimit);
    assert!(!d.is_success());
    assert_eq!(
        error_kind(design_fir_remez_with_options(
            0,
            FirSymmetry::Even,
            &bands,
            RemezOptions::default()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(design_fir_least_squares(10, FirSymmetry::Odd, &bands)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(design_fir_least_squares(
            9,
            FirSymmetry::Even,
            &[band::<f64>(0.3, 0.3000000001, 1., 1., 1.)]
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(
        error_kind(design_fir_window(
            5,
            FirSymmetry::Even,
            &[band::<f64>(0., 0.5, 0., 1., 1.)],
            WindowKind::Hann,
            FirNormalization::None
        )),
        SciOxideErrorKind::InvalidInput
    );
}
// Direct constructions complete without a tolerance claim; only the exchange
// converges. Success derives from the termination alone.
#[test]
fn termination_distinguishes_completion_from_convergence() {
    let bands = [band::<f64>(0., 0.3, 1., 1., 1.), band(0.4, 1., 0., 0., 1.)];
    let direct = [
        design_fir_window(
            15,
            FirSymmetry::Even,
            &bands[..1],
            WindowKind::Hann,
            FirNormalization::None,
        )
        .unwrap(),
        design_fir_frequency_sampling(
            15,
            FirSymmetry::Even,
            &[0., 0.3, 0.4, 1.],
            &[1., 1., 0., 0.],
            64,
            None,
            FirNormalization::None,
        )
        .unwrap(),
        design_fir_least_squares(15, FirSymmetry::Even, &bands).unwrap(),
    ];
    for design in &direct {
        assert_eq!(design.termination(), FirTermination::Completed);
        assert!(design.is_success() && design.termination().is_success());
        assert_eq!(design.iterations(), 0);
    }
    let exchange =
        design_fir_remez_with_options(15, FirSymmetry::Even, &bands, RemezOptions::default())
            .unwrap();
    assert_eq!(exchange.termination(), FirTermination::Converged);
    assert!(exchange.is_success() && exchange.iterations() > 0);
    for unsuccessful in [
        FirTermination::IterationLimit,
        FirTermination::ExchangeStalled,
    ] {
        assert!(!unsuccessful.is_success());
    }
}
#[test]
fn wide_transition_f32_converges() {
    let d = design_fir_remez_with_options(
        31,
        FirSymmetry::Even,
        &[band::<f32>(0., 0.3, 1., 1., 1.), band(0.6, 1., 0., 0., 4.)],
        RemezOptions::default(),
    )
    .unwrap();
    assert_eq!(d.termination(), FirTermination::Converged);
    assert!(d.weighted_ripple().unwrap() < 0.01);
}

// Direct inverse DFT oracle exercises the packed-real path with nonzero
// self-conjugate endpoints and meshes that are not powers of two.
fn packed_sampling<S: RealScalar>() {
    let tolerance = 64.0 * S::EPSILON.into_f64();
    for (n, symmetry) in [
        (1, FirSymmetry::Even),
        (7, FirSymmetry::Even),
        (8, FirSymmetry::Even),
        (7, FirSymmetry::Odd),
        (8, FirSymmetry::Odd),
    ] {
        let mesh = 6 * n;
        let frequencies = [0.0, 0.3, 0.7, 1.0];
        let amplitudes = [
            if symmetry == FirSymmetry::Odd {
                0.0
            } else {
                1.0
            },
            0.8,
            0.2,
            if (symmetry == FirSymmetry::Even && n % 2 == 0)
                || (symmetry == FirSymmetry::Odd && n % 2 == 1)
            {
                0.0
            } else {
                0.4
            },
        ];
        let f = frequencies.map(s::<S>);
        let a = amplitudes.map(s::<S>);
        let actual = design_fir_frequency_sampling(
            n,
            symmetry,
            &f,
            &a,
            mesh,
            Some(WindowKind::Hamming),
            FirNormalization::None,
        )
        .unwrap();
        let center = (n - 1) as f64 / 2.0;
        let raw = |i: usize| {
            let mut sum = 0.0;
            for k in 0..=mesh / 2 {
                let f = k as f64 / (mesh / 2) as f64;
                let segment = if f <= 0.3 {
                    0
                } else if f <= 0.7 {
                    1
                } else {
                    2
                };
                let t =
                    (f - frequencies[segment]) / (frequencies[segment + 1] - frequencies[segment]);
                let a = amplitudes[segment] * (1.0 - t) + amplitudes[segment + 1] * t;
                let phase = std::f64::consts::PI * f * center;
                let (re, im) = if symmetry == FirSymmetry::Even {
                    (a * phase.cos(), -a * phase.sin())
                } else {
                    (a * phase.sin(), a * phase.cos())
                };
                let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / mesh as f64;
                sum += if k == 0 || k == mesh / 2 {
                    re * angle.cos()
                } else {
                    2.0 * (re * angle.cos() - im * angle.sin())
                };
            }
            sum / mesh as f64
        };
        for (i, &value) in actual.coefficients().iter().enumerate() {
            let reflected = raw(n - 1 - i)
                * if symmetry == FirSymmetry::Even {
                    1.0
                } else {
                    -1.0
                };
            let weight = if n == 1 {
                1.0
            } else {
                0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
            };
            let expected = (raw(i) + reflected) * 0.5 * weight;
            assert!(
                (value.into_f64() - expected).abs() <= tolerance,
                "n={n}, {symmetry:?}, i={i}"
            );
        }
    }
}

#[test]
fn packed_real_sampling_matches_independent_inverse_dft() {
    packed_sampling::<f32>();
    packed_sampling::<f64>();
}
