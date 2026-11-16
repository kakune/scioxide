#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::{error, error_kind};
use scioxide_core::{
    Budget, CancellationToken, ExecutionContext, RealScalar, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_signal::{
    DpssOptions, DpssTapers, Window, WindowKind as W, WindowNormalization as Norm,
    WindowSymmetry as Sym, window, window_metrics,
};

fn families() -> Vec<W<'static>> {
    vec![
        W::Boxcar,
        W::Triang,
        W::Parzen,
        W::Bohman,
        W::Blackman,
        W::Nuttall,
        W::BlackmanHarris,
        W::FlatTop,
        W::Bartlett,
        W::Barthann,
        W::Hamming,
        W::Hann,
        W::Kaiser { beta: 14.0 },
        W::Gaussian { sigma: 3.0 },
        W::GeneralCosine(&[0.5, 0.4, 0.1]),
        W::GeneralGaussian {
            power: 1.5,
            sigma: 3.0,
        },
        W::GeneralHamming { alpha: 0.6 },
        W::Chebyshev { attenuation: 80.0 },
        W::Cosine,
        W::Exponential {
            tau: 2.0,
            center: None,
        },
        W::Tukey { alpha: 0.4 },
        W::Taylor {
            nbar: 4,
            sidelobe_level: 60.0,
            normalize: true,
        },
        W::Lanczos,
    ]
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:.17e} != {b:.17e} (tol {tol})");
}
fn geometry<S: RealScalar>() {
    let tol = 64.0 * S::EPSILON.into_f64();
    for kind in families() {
        assert!(
            window::<S>(kind, 0, Sym::Symmetric, Norm::Native)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            window::<S>(kind, 1, Sym::Periodic, Norm::Native).unwrap(),
            [S::ONE]
        );
        for n in [2, 3, 8, 9, 32, 33] {
            let symmetric = window::<S>(kind, n, Sym::Symmetric, Norm::Native).unwrap();
            for (&a, &b) in symmetric.iter().zip(symmetric.iter().rev()) {
                close(a.into_f64(), b.into_f64(), tol);
            }
            let periodic = window::<S>(kind, n, Sym::Periodic, Norm::Native).unwrap();
            let extended = window::<S>(kind, n + 1, Sym::Symmetric, Norm::Native).unwrap();
            assert_eq!(periodic, extended[..n]);
            if symmetric.iter().any(|x| *x != S::ZERO) {
                let unit = window::<S>(kind, n, Sym::Symmetric, Norm::UnitEnergy).unwrap();
                close(unit.iter().map(|x| x.into_f64().powi(2)).sum(), 1.0, tol);
                let peak = window::<S>(kind, n, Sym::Symmetric, Norm::UnitPeak).unwrap();
                close(
                    peak.iter().map(|x| x.into_f64().abs()).fold(0.0, f64::max),
                    1.0,
                    tol,
                );
            }
        }
    }
}
#[test]
fn geometry_f32() {
    geometry::<f32>();
}
#[test]
fn geometry_f64() {
    geometry::<f64>();
}

#[test]
fn normalization_extreme_ranges_and_periodic_lengths() {
    for magnitude in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
        for n in [2, 9, 4096] {
            for symmetry in [Sym::Symmetric, Sym::Periodic] {
                let coefficients = [magnitude];
                let kind = W::GeneralCosine(&coefficients);
                let peak = window::<f64>(kind, n, symmetry, Norm::UnitPeak).unwrap();
                assert!(peak.iter().all(|&x| x == 1.0));
                let energy = window::<f64>(kind, n, symmetry, Norm::UnitEnergy).unwrap();
                for value in energy {
                    close(value, 1.0 / (n as f64).sqrt(), 2e-15);
                }
            }
        }
    }
    assert_eq!(
        error_kind(window::<f64>(
            W::GeneralCosine(&[0.0]),
            8,
            Sym::Periodic,
            Norm::UnitEnergy
        )),
        SciOxideErrorKind::DomainError
    );
}

#[test]
fn large_windows_preserve_formulas_and_reflection() {
    for n in [4096, 65536] {
        for kind in [
            W::Hann,
            W::Gaussian {
                sigma: n as f64 / 8.0,
            },
            W::Tukey { alpha: 0.5 },
        ] {
            let samples = window::<f64>(kind, n, Sym::Symmetric, Norm::Native).unwrap();
            for i in [0, 1, n / 8, n / 4, n / 2 - 1] {
                let x = i as f64 / (n - 1) as f64;
                let expected = match kind {
                    W::Hann => 0.5 - 0.5 * (2.0 * std::f64::consts::PI * x).cos(),
                    W::Gaussian { sigma } => {
                        (-0.5 * ((i as f64 - (n - 1) as f64 / 2.0) / sigma).powi(2)).exp()
                    }
                    _ => {
                        if x >= 0.25 {
                            1.0
                        } else {
                            0.5 * (1.0 - (4.0 * std::f64::consts::PI * x).cos())
                        }
                    }
                };
                close(samples[i], expected, 4e-15);
                assert_eq!(samples[i], samples[n - 1 - i]);
            }
        }
    }
}

#[test]
fn tiny_tukey_taper_keeps_exact_endpoints() {
    for alpha in [f64::from_bits(1), f64::MIN_POSITIVE, 1e-100] {
        for n in [2, 9] {
            let samples =
                window::<f64>(W::Tukey { alpha }, n, Sym::Symmetric, Norm::Native).unwrap();
            assert_eq!(samples[0], 0.0);
            assert_eq!(samples[n - 1], 0.0);
            assert!(samples[1..n - 1].iter().all(|&value| value == 1.0));
        }
    }
}

#[test]
fn direct_high_harmonics_match_high_precision_references() {
    let coefficients: Vec<_> = (0..32)
        .map(|j| (if j % 2 == 0 { 1.0 } else { -1.0 }) / (j + 1) as f64)
        .collect();
    let samples = window::<f64>(
        W::GeneralCosine(&coefficients),
        257,
        Sym::Symmetric,
        Norm::Native,
    )
    .unwrap();
    // mpmath 1.3.0, 90 digits, using the represented binary64 coefficients.
    for (i, expected) in [
        (0, 4.058_495_195_436_52),
        (1, 3.921_039_789_583_822),
        (31, 0.997_724_868_767_883_3),
        (63, 0.767_213_694_682_138_3),
        (128, 0.677_766_202_207_526_9),
    ] {
        close(samples[i], expected, 8e-14);
    }
}
#[test]
fn endpoint_metrics_and_exact_small_cases() {
    for (kind, expected) in [
        (W::Boxcar, [1.0, 1.0, 1.0, 1.0, 1.0]),
        (W::Bartlett, [0.0, 0.5, 1.0, 0.5, 0.0]),
        (W::Hann, [0.0, 0.5, 1.0, 0.5, 0.0]),
        (W::Triang, [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0]),
    ] {
        for (a, b) in window::<f64>(kind, 5, Sym::Symmetric, Norm::Native)
            .unwrap()
            .iter()
            .zip(expected)
        {
            close(*a, b, 2e-15);
        }
    }
    let hann = window::<f64>(W::Hann, 64, Sym::Periodic, Norm::Native).unwrap();
    let metrics = window_metrics(&hann).unwrap();
    close(metrics.coherent_gain(), 0.5, 2e-15);
    close(metrics.equivalent_noise_bandwidth(), 1.5, 2e-15);
    let huge = window_metrics(&[f64::MAX, f64::MAX]).unwrap();
    assert_eq!(huge.coherent_gain(), f64::MAX);
    close(huge.equivalent_noise_bandwidth(), 1.0, 0.0);
    let tiny = window_metrics(&[f64::from_bits(1), f64::from_bits(1)]).unwrap();
    assert_eq!(tiny.coherent_gain(), f64::from_bits(1));
    close(tiny.equivalent_noise_bandwidth(), 1.0, 0.0);
    assert_eq!(
        error_kind(window_metrics::<f64>(&[])),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(window_metrics(&[1.0, -1.0])),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        error_kind(window_metrics(&[0.0, 0.0])),
        SciOxideErrorKind::DomainError
    );
    assert_eq!(
        error_kind(window_metrics(&[f64::NAN])),
        SciOxideErrorKind::NonFinite
    );
}
#[test]
fn kaiser_bessel_complementary_energy() {
    for n in [2, 8, 32, 128] {
        for beta in [0.0, 14.0, 1000.0] {
            let w = window::<f64>(
                W::KaiserBesselDerived { beta },
                n,
                Sym::Symmetric,
                Norm::Native,
            )
            .unwrap();
            for i in 0..n / 2 {
                close(w[i] * w[i] + w[i + n / 2] * w[i + n / 2], 1.0, 3e-14);
                close(w[i], w[n - 1 - i], 0.0);
            }
        }
    }
    assert!(
        window::<f64>(
            W::KaiserBesselDerived { beta: 1.0 },
            0,
            Sym::Symmetric,
            Norm::Native
        )
        .unwrap()
        .is_empty()
    );
}
#[test]
fn validation_and_atomic_output() {
    // Parameters are validated before the (empty) length is accepted.
    for (kind, error) in [
        (W::Gaussian { sigma: 0.0 }, SciOxideErrorKind::InvalidInput),
        (W::Kaiser { beta: -1.0 }, SciOxideErrorKind::InvalidInput),
        (
            W::GeneralGaussian {
                power: 0.0,
                sigma: 1.0,
            },
            SciOxideErrorKind::InvalidInput,
        ),
        (
            W::Chebyshev {
                attenuation: f64::INFINITY,
            },
            SciOxideErrorKind::NonFinite,
        ),
        (W::GeneralCosine(&[]), SciOxideErrorKind::InvalidInput),
        (
            W::Taylor {
                nbar: 0,
                sidelobe_level: 60.0,
                normalize: true,
            },
            SciOxideErrorKind::InvalidInput,
        ),
        (
            W::Exponential {
                tau: 1.0,
                center: Some(0.0),
            },
            SciOxideErrorKind::InvalidInput,
        ),
    ] {
        assert_eq!(
            error_kind(Window::<f64>::new(kind, 0, Sym::Symmetric, Norm::Native)),
            error
        );
    }
    for (n, sym) in [(3, Sym::Symmetric), (4, Sym::Periodic)] {
        assert_eq!(
            error_kind(window::<f64>(
                W::KaiserBesselDerived { beta: 1.0 },
                n,
                sym,
                Norm::Native
            )),
            SciOxideErrorKind::InvalidInput
        );
    }
    assert_eq!(
        error_kind(window::<f64>(
            W::Hann,
            usize::MAX,
            Sym::Periodic,
            Norm::Native
        )),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        error_kind(window::<f64>(
            W::GeneralCosine(&[0.0]),
            4,
            Sym::Symmetric,
            Norm::UnitPeak
        )),
        SciOxideErrorKind::DomainError
    );
    let plan = Window::<f64>::new(W::Hann, 8, Sym::Symmetric, Norm::Native).unwrap();
    let built = Window::<f64>::build(
        W::Hann,
        8,
        Sym::Symmetric,
        Norm::Native,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(plan.samples(), built.samples());
    assert_eq!(plan.clone().into_samples(), plan.samples());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancellation);
    for kind in [W::Hann, W::Chebyshev { attenuation: 60.0 }] {
        assert_eq!(
            error_kind(Window::<f64>::build(
                kind,
                8,
                Sym::Symmetric,
                Norm::Native,
                context
            )),
            SciOxideErrorKind::ExecutionStopped
        );
    }
    assert_eq!(
        error_kind(DpssTapers::<f64>::build(
            8,
            2.0,
            3,
            DpssOptions::default(),
            context
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    let built = DpssTapers::<f64>::build(
        8,
        2.0,
        3,
        DpssOptions::default(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let plan = DpssTapers::<f64>::new(8, 2.0, 3, DpssOptions::default()).unwrap();
    assert_eq!(plan.samples(), built.samples());
}
fn dpss<S: RealScalar>() {
    let tol = if S::EPSILON.into_f64() > 1e-10 {
        3e-6
    } else {
        2e-12
    };
    for (n, nw, k) in [(2, 0.4, 2), (16, 2.5, 8), (65, 8.0, 20), (128, 20.0, 40)] {
        let plan = DpssTapers::<S>::new(n, nw, k, DpssOptions::default()).unwrap();
        let nf = f64::from(u32::try_from(n).unwrap());
        for order in 0..k {
            let v = plan.taper(order).unwrap();
            for (i, &x) in v.iter().enumerate() {
                close(
                    x.into_f64(),
                    v[n - 1 - i].into_f64() * if order % 2 == 0 { 1.0 } else { -1.0 },
                    tol,
                );
            }
            for j in 0..=order {
                let inner: f64 = v
                    .iter()
                    .zip(plan.taper(j).unwrap())
                    .map(|(a, b)| a.into_f64() * b.into_f64())
                    .sum();
                close(inner, if j == order { 1.0 } else { 0.0 }, tol);
            }
            // Independent dense sinc-kernel quadratic form, unlike production's lag autocorrelation.
            let mut concentration = 0.0;
            for (i, &a) in v.iter().enumerate() {
                for (j, &b) in v.iter().enumerate() {
                    let lag = f64::from(i32::try_from(i).unwrap() - i32::try_from(j).unwrap());
                    let kernel = if lag == 0.0 {
                        2.0 * nw / nf
                    } else {
                        (2.0 * std::f64::consts::PI * nw / nf * lag).sin()
                            / (std::f64::consts::PI * lag)
                    };
                    concentration += a.into_f64() * b.into_f64() * kernel;
                }
            }
            close(concentration, plan.concentration_ratios()[order], tol * 4.0);
            // Rayleigh residual for the commuting tridiagonal matrix.
            let applied: Vec<f64> = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let t = f64::from(u32::try_from(i).unwrap());
                    let d = ((nf - 1.0) / 2.0 - t).powi(2)
                        * (2.0 * std::f64::consts::PI * nw / nf).cos();
                    let left = if i > 0 {
                        t * (nf - t) / 2.0 * v[i - 1].into_f64()
                    } else {
                        0.0
                    };
                    let right = if i + 1 < n {
                        (t + 1.0) * (nf - t - 1.0) / 2.0 * v[i + 1].into_f64()
                    } else {
                        0.0
                    };
                    (d * x.into_f64() + left + right) / (nf * nf)
                })
                .collect();
            let lambda: f64 = applied.iter().zip(v).map(|(a, b)| a * b.into_f64()).sum();
            let residual = applied
                .iter()
                .zip(v)
                .map(|(a, b)| (a - lambda * b.into_f64()).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < tol, "{residual}");
        }
        assert!(
            plan.concentration_ratios()
                .windows(2)
                .all(|pair| pair[0] + tol >= pair[1])
        );
        assert_eq!(plan.samples().len(), n * k);
        assert_eq!(error_kind(plan.taper(k)), SciOxideErrorKind::InvalidInput);
    }
}
#[test]
fn dpss_f32() {
    dpss::<f32>();
}
#[test]
fn dpss_f64() {
    dpss::<f64>();
}

#[test]
fn selected_dpss_large_lengths_keep_residuals_orthogonality_and_concentration() {
    for n in [1024, 4096] {
        let plan = DpssTapers::<f64>::new(n, 4.0, 4, DpssOptions::default()).unwrap();
        assert!(plan.report().residual().scaled_norm() < 2e-12);
        assert!(plan.report().counts().total().unwrap().iterations() > 0);
        for k in 0..4 {
            let taper = plan.taper(k).unwrap();
            assert!((0.999..=1.0).contains(&plan.concentration_ratios()[k]));
            for j in 0..=k {
                let dot: f64 = taper
                    .iter()
                    .zip(plan.taper(j).unwrap())
                    .map(|(a, b)| a * b)
                    .sum();
                close(dot, if j == k { 1.0 } else { 0.0 }, 2e-12);
            }
            for (&a, &b) in taper.iter().zip(taper.iter().rev()) {
                close(a, if k % 2 == 0 { b } else { -b }, 2e-9);
            }
        }
    }
}

#[test]
fn dpss_overrides_spectral_selection_for_symmetric_and_periodic_jobs() {
    for symmetry in [Sym::Symmetric, Sym::Periodic] {
        for k in [1, 9] {
            let options = DpssOptions::new().with_symmetry(symmetry);
            let expected = DpssTapers::<f64>::new(9, 2.0, k, options).unwrap();
            let actual = DpssTapers::<f64>::new(
                9,
                2.0,
                k,
                options.with_eigen(
                    options
                        .eigen()
                        .with_index_range(scioxide_linalg::EigenIndexRange::new(100, 101).unwrap())
                        .with_ordering(scioxide_linalg::EigenvalueOrder::Ascending),
                ),
            )
            .unwrap();
            assert_eq!(actual.samples(), expected.samples());
            assert_eq!(
                actual.concentration_ratios(),
                expected.concentration_ratios()
            );
        }
    }
}
#[test]
fn dpss_small_exact_and_periodic_contract() {
    let plan = DpssTapers::<f64>::new(2, 0.4, 2, DpssOptions::default()).unwrap();
    let ratio = 0.4 + (0.4 * std::f64::consts::PI).sin() / std::f64::consts::PI;
    close(plan.concentration_ratios()[0], ratio, 1e-15);
    close(
        plan.taper(0).unwrap()[0],
        std::f64::consts::FRAC_1_SQRT_2,
        1e-15,
    );
    let options = DpssOptions::new()
        .with_symmetry(Sym::Periodic)
        .with_normalization(Norm::Native);
    assert_eq!(
        (options.symmetry(), options.normalization()),
        (Sym::Periodic, Norm::Native)
    );
    let periodic = DpssTapers::<f64>::new(16, 3.0, 4, options).unwrap();
    let symmetric = DpssTapers::<f64>::new(
        17,
        3.0,
        4,
        DpssOptions::new().with_normalization(Norm::Native),
    )
    .unwrap();
    assert_eq!(
        periodic.concentration_ratios(),
        symmetric.concentration_ratios()
    );
    for k in 0..4 {
        assert_eq!(
            periodic.taper(k).unwrap(),
            &symmetric.taper(k).unwrap()[..16]
        );
    }
    for (n, nw, k) in [
        (0, 1.0, 1),
        (4, 0.0, 1),
        (4, 2.0, 1),
        (4, 1.0, 0),
        (4, 1.0, 5),
    ] {
        assert_eq!(
            error_kind(DpssTapers::<f64>::new(n, nw, k, DpssOptions::default())),
            SciOxideErrorKind::InvalidInput
        );
    }
    let eigen = DpssOptions::default().eigen();
    let exhausted = DpssOptions::new().with_eigen(
        eigen.with_execution_options(eigen.execution_options().with_iteration_budget(Budget::ONE)),
    );
    assert_eq!(
        error_kind(DpssTapers::<f64>::new(32, 3.0, 4, exhausted)),
        SciOxideErrorKind::ConvergenceFailure
    );
}
#[test]
fn high_order_and_spectral_leakage() {
    for kind in [
        W::Kaiser { beta: 1000.0 },
        W::Chebyshev { attenuation: 300.0 },
        W::Taylor {
            nbar: 32,
            sidelobe_level: 120.0,
            normalize: true,
        },
        W::GeneralGaussian {
            power: 100.0,
            sigma: 4.0,
        },
    ] {
        for n in [64, 65] {
            let a = window::<f64>(kind, n, Sym::Symmetric, Norm::UnitPeak).unwrap();
            let b = window::<f32>(kind, n, Sym::Symmetric, Norm::UnitPeak).unwrap();
            for (&a, &b) in a.iter().zip(&b) {
                close(a, f64::from(b), 6e-8);
            }
        }
    }
    // Harris' periodic Hann: half-bin-offset far sidelobes decay faster than rectangular.
    let n = 128;
    let hann = window::<f64>(W::Hann, n, Sym::Periodic, Norm::Native).unwrap();
    let boxcar = vec![1.0; n];
    let response = |w: &[f64], frequency: f64| {
        let (re, im) = w.iter().enumerate().fold((0.0, 0.0), |(re, im), (i, x)| {
            let a = 2.0 * std::f64::consts::PI * frequency * f64::from(u32::try_from(i).unwrap())
                / 128.0;
            (re + x * a.cos(), im + x * a.sin())
        });
        re.hypot(im) / w.iter().sum::<f64>()
    };
    for bin in [10.5, 20.5, 30.5] {
        assert!(response(&hann, bin) < response(&boxcar, bin) * 0.02);
    }
}

#[test]
fn supplementary_scipy_comparator_and_independent_high_precision_dpss() {
    let names = [
        "boxcar",
        "triang",
        "parzen",
        "bohman",
        "blackman",
        "nuttall",
        "blackmanharris",
        "flattop",
        "bartlett",
        "barthann",
        "hamming",
        "hann",
        "kaiser",
        "gaussian",
        "general_cosine",
        "general_gaussian",
        "general_hamming",
        "chebyshev",
        "cosine",
        "exponential",
        "tukey",
        "taylor",
        "lanczos",
    ];
    let kinds = families();
    for line in include_str!("fixtures/windows.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let mut fields = line.split(',');
        let name = fields.next().unwrap();
        let kind = if name == "kbd" {
            W::KaiserBesselDerived { beta: 14.0 }
        } else {
            kinds[names.iter().position(|&n| n == name).unwrap()]
        };
        let n = fields.next().unwrap().parse().unwrap();
        let symmetry = if fields.next().unwrap() == "1" {
            Sym::Symmetric
        } else {
            Sym::Periodic
        };
        let actual = window::<f64>(kind, n, symmetry, Norm::Native).unwrap();
        let narrow = window::<f32>(kind, n, symmetry, Norm::Native).unwrap();
        for ((a, b), expected) in actual.iter().zip(narrow).zip(fields) {
            let expected = expected.parse().unwrap();
            close(*a, expected, 3e-13);
            close(f64::from(b), expected, 8e-8);
        }
    }
    let wide = DpssTapers::<f64>::new(16, 2.5, 8, DpssOptions::default()).unwrap();
    let narrow = DpssTapers::<f32>::new(16, 2.5, 8, DpssOptions::default()).unwrap();
    for line in include_str!("fixtures/dpss.csv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let mut fields = line.split(',');
        let k: usize = fields.next().unwrap().parse().unwrap();
        close(
            wide.concentration_ratios()[k],
            fields.next().unwrap().parse().unwrap(),
            2e-13,
        );
        for ((a, b), expected) in wide
            .taper(k)
            .unwrap()
            .iter()
            .zip(narrow.taper(k).unwrap())
            .zip(fields)
        {
            let expected = expected.parse().unwrap();
            close(*a, expected, 3e-13);
            close(f64::from(*b), expected, 8e-8);
        }
    }
}

#[test]
fn nonfinite_parameters_and_metrics_report_input_origin() {
    use scioxide_core::NonFiniteOrigin;
    for kind in [
        W::Kaiser { beta: f64::NAN },
        W::Gaussian {
            sigma: f64::INFINITY,
        },
        W::GeneralCosine(&[f64::NAN]),
        W::GeneralHamming { alpha: f64::NAN },
        W::Exponential {
            tau: 1.0,
            center: Some(f64::NAN),
        },
    ] {
        assert!(matches!(
            error(Window::<f64>::new(kind, 8, Sym::Periodic, Norm::Native)).view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
    assert!(matches!(
        error(window_metrics(&[1.0, f64::INFINITY])).view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    assert!(matches!(
        error(DpssTapers::<f64>::new(
            8,
            f64::NAN,
            2,
            DpssOptions::default()
        ))
        .view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn single_sample_dpss_is_a_unit_impulse() {
    use scioxide_signal::DpssTermination;
    for symmetry in [Sym::Symmetric, Sym::Periodic] {
        for normalization in [Norm::UnitEnergy, Norm::UnitPeak, Norm::Native] {
            for nw in [0.1, 0.25, 0.49] {
                let options = DpssOptions::new()
                    .with_symmetry(symmetry)
                    .with_normalization(normalization);
                let plan = DpssTapers::<f32>::new(1, nw, 1, options).unwrap();
                assert_eq!(plan.samples(), [1.0]);
                // The flat spectrum of an impulse puts 2W = 2 NW of its energy
                // in [-W, W].
                assert_eq!(plan.concentration_ratios(), [2.0 * nw]);
                assert_eq!(plan.report().termination(), DpssTermination::Completed);
                assert_eq!(plan.report().residual().raw_norm(), 0.0);
                assert_eq!(plan.report().counts().total().unwrap().iterations(), 0);
            }
        }
    }
    // The documented bandwidth range 0 < NW < length/2 still applies.
    for nw in [0.0, 0.5] {
        assert_eq!(
            error_kind(DpssTapers::<f64>::new(1, nw, 1, DpssOptions::default())),
            SciOxideErrorKind::InvalidInput
        );
    }
}
