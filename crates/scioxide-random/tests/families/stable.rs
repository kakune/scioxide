//! Independent stable-law evaluations against retained mpmath references.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Budget, SciOxideErrorKind, Tolerance};
use scioxide_random::{Cdf, InverseSurvival, LevyStable, LogCdf, LogSurvival, Quantile, Survival};
use scioxide_random::{LawOptions, LogDensity};

#[test]
fn skew_transition_precision() {
    for tol in [1.4901161193847656e-8, 1e-10, 1e-12] {
        let options = LawOptions::default()
            .with_tolerance(Tolerance::new(0., tol).unwrap())
            .with_work_budget(Budget::try_from(65536).unwrap());
        let law = LevyStable::with_options(1.01_f64, -1., options).unwrap();
        assert!((law.log_cdf(-1.).unwrap() + 4.631_288_059_146_488).abs() < 8e-8);
    }
}

#[test]
fn policies_boundaries_and_both_precisions() {
    for (a, b) in [
        (0., 0.),
        (2.01, 0.),
        (1., 1.01),
        (f64::NAN, 0.),
        (1., f64::INFINITY),
    ] {
        assert!(LevyStable::new(a, b).is_err());
    }
    let tiny = LawOptions::default()
        .with_tolerance(Tolerance::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE);
    let law = LevyStable::with_options(0.99_f64, 0.5, tiny).unwrap();
    for result in [
        law.cdf(1.),
        law.sf(1.),
        law.log_pdf(1.),
        law.quantile(0.1),
        law.isf(0.1),
    ] {
        assert_eq!(
            result.unwrap_err().kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
    }
    for a in [0.5_f32, 0.9, 0.99, 1., 1.01, 1.1, 1.5, 1.9, 2.] {
        for b in [-1., -0.5, 0., 0.5, 1.] {
            let law = LevyStable::new(a, b).unwrap();
            let reference = LevyStable::new(f64::from(a), f64::from(b)).unwrap();
            for x in [-10., -1., 0., 1., 10.] {
                for (v, r) in [
                    (law.cdf(x).unwrap(), reference.cdf(f64::from(x)).unwrap()),
                    (law.sf(x).unwrap(), reference.sf(f64::from(x)).unwrap()),
                ] {
                    assert!(
                        (f64::from(v) - r).abs()
                            <= 8.
                                * f64::from(f32::EPSILON.sqrt())
                                * r.max(f64::from(f32::MIN_POSITIVE)),
                        "{a} {b} {x}: {v} {r}"
                    );
                }
            }
            for p in [0.05, 0.5, 0.95] {
                let x = law.quantile(p).unwrap();
                let expected = reference.quantile(f64::from(p)).unwrap();
                assert!(
                    (f64::from(x) - expected).abs()
                        <= 8. * f64::from(f32::EPSILON.sqrt()) * expected.abs().max(1.)
                );
                if b == 0. && p == 0.5 {
                    assert_eq!(x.to_bits(), 0);
                }
            }
            for p in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
                assert!(law.quantile(p).is_err());
                assert!(law.isf(p).is_err());
            }
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(law.cdf(x).is_err());
                assert!(law.log_sf(x).is_err());
            }
        }
    }
    for a in [0.5, 0.99, 1., 1.01, 1.5, 2.] {
        let law = LevyStable::new(a, 0.).unwrap();
        assert_eq!(law.quantile(0.).unwrap(), f64::NEG_INFINITY);
        assert_eq!(law.quantile(1.).unwrap(), f64::INFINITY);
        for p in [0.5 - 1e-12, 0.5 + 1e-12] {
            let x = law.quantile(p).unwrap();
            assert_eq!(x.is_sign_positive(), p > 0.5);
            assert!(x != 0.);
            assert_eq!(x, -law.isf(p).unwrap());
        }
    }
    for b in [-1., 1.] {
        let law = LevyStable::new(0.99, b).unwrap();
        assert_eq!(law.quantile(if b > 0. { 0. } else { 1. }).unwrap(), 0.);
        assert_eq!(law.cdf(-b).unwrap(), if b > 0. { 0. } else { 1. });
    }
}

#[test]
fn default_domain_grid() {
    let mut failures = Vec::new();
    for alpha in [0.5, 0.9, 0.99, 1., 1.01, 1.1, 1.5, 1.9, 2.] {
        for beta in [-1., -0.5, 0., 0.5, 1.] {
            let law = LevyStable::<f64>::new(alpha, beta).unwrap();
            let mut previous = 0.;
            for j in 0..41 {
                let x = -10. + f64::from(j) / 2.;
                match (law.cdf(x), law.sf(x)) {
                    (Ok(cdf), Ok(sf)) => {
                        assert!(cdf >= previous - 1e-14, "monotonicity {alpha} {beta} {x}");
                        assert!(
                            (cdf + sf - 1.).abs() < 1e-10,
                            "complement {alpha} {beta} {x}"
                        );
                        previous = cdf;
                    }
                    pair => failures.push(format!("tails {alpha} {beta} {x}: {pair:?}")),
                }
            }
            for i in 1..20 {
                let p = f64::from(i) / 20.;
                match law.quantile(p) {
                    Ok(x) => {
                        let actual = if p <= 0.5 {
                            law.log_cdf(x)
                        } else {
                            law.log_sf(x)
                        };
                        match actual {
                            Ok(actual) if (actual - p.min(1. - p).ln()).abs() < 8e-8 => (),
                            result => {
                                failures.push(format!("inverse {alpha} {beta} {p} {x}: {result:?}"))
                            }
                        }
                    }
                    Err(e) => failures.push(format!("quantile {alpha} {beta} {p}: {e}")),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn exact_symmetric_medians() {
    for alpha in [0.1, 0.5, 0.9, 0.99, 1., 1.01, 1.5, 1.9, 2.] {
        let law = LevyStable::new(alpha, 0.).unwrap();
        assert_eq!(law.quantile(0.5).unwrap(), 0.);
        assert_eq!(law.isf(0.5).unwrap(), 0.);
    }
}

#[test]
fn very_small_alpha_and_extreme_coordinates() {
    for a in [1e-8, 1e-20, f64::MIN_POSITIVE, f64::from_bits(1)] {
        for b in [-1., -0.5, 0., 0.5, 1.] {
            let law = LevyStable::new(a, b).unwrap();
            for x in [1e-100, 1., 1e100] {
                let expected = (1. - b) / 2. + (1. + b) / 2. * (-1_f64).exp();
                let actual = law.cdf(x).unwrap();
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "{a} {b} {x}: {actual} vs {expected}"
                );
                assert!((actual + law.sf(x).unwrap() - 1.).abs() < 1e-10);
            }
        }
    }
    let normal = LevyStable::new(2., 0.).unwrap();
    assert_eq!(normal.cdf(f64::MAX).unwrap(), 1.);
    assert_eq!(normal.sf(f64::MAX).unwrap(), 0.);
    assert!(normal.log_sf(f64::MAX).is_err());
    assert!(normal.log_pdf(f64::MAX).is_err());
    let tiny = LevyStable::new(0.01_f64, 0.5).unwrap();
    let x = tiny.quantile(0.999).unwrap();
    assert!(x.is_finite() && x > 1e250);
    assert!((tiny.log_sf(x).unwrap() - (1. - 0.999_f64).ln()).abs() < 8e-8);
    assert_eq!(
        tiny.quantile(0.999999).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(
        LevyStable::new(0.001_f64, 0.)
            .unwrap()
            .quantile(0.5001)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn skewness_boundaries() {
    for a in [0.1, 0.5, 0.99, 1., 1.01, 1.5, 1.9] {
        for b in [
            -1_f64.next_down(),
            -1. + 1e-12,
            -1e-12,
            1e-12,
            1. - 1e-12,
            1_f64.next_down(),
        ] {
            let law = LevyStable::new(a, b).unwrap();
            for x in [-100., -1., 0., 1., 100.] {
                let lc = law.log_cdf(x).unwrap();
                let ls = law.log_sf(x).unwrap();
                assert!(lc.is_finite() && ls.is_finite(), "{a} {b} {x}: {lc} {ls}");
                assert!((lc.exp() + ls.exp() - 1.).abs() < 1e-10);
            }
            for p in [0.05, 0.5, 0.95] {
                assert!(law.quantile(p).is_ok(), "{a} {b} {p}");
            }
        }
    }
}

#[test]
fn continuity_and_extremes() {
    let mut failures = Vec::new();
    for alpha in [
        0.01_f64,
        0.1,
        0.25,
        0.999,
        1. - 1e-6,
        1. - 1e-12,
        1_f64.next_down(),
        1_f64.next_up(),
        1. + 1e-12,
        1. + 1e-6,
        1.001,
        1.999999,
    ] {
        for beta in [-1., -0.5, 0., 0.5, 1.] {
            let law = LevyStable::<f64>::new(alpha, beta).unwrap();
            let shift = if alpha != 1. {
                -beta / ((alpha - 1.) * core::f64::consts::FRAC_PI_2).tan()
            } else {
                0.
            };
            for x in [
                -100.,
                -1.,
                -1e-12,
                0.,
                1e-12,
                1.,
                100.,
                shift - 1.,
                shift,
                shift + 1.,
            ] {
                let cdf = law.cdf(x).unwrap();
                let sf = law.sf(x).unwrap();
                assert!((cdf + sf - 1.).abs() < 1e-10);
                for (name, result) in [("cdf", law.log_cdf(x)), ("sf", law.log_sf(x))] {
                    match result {
                        Ok(v) if v <= 0. => (),
                        Err(e)
                            if alpha < 1.
                                && beta.abs() == 1.
                                && beta * x > 0.
                                && beta * x <= 100.
                                && alpha >= 0.999
                                && matches!(
                                    e.view(),
                                    scioxide_core::SciOxideErrorView::NumericalInstability {
                                        issue: scioxide_core::NumericalIssue::Overflow,
                                        ..
                                    }
                                ) => {}
                        other => failures.push(format!("{name} {alpha} {beta} {x}: {other:?}")),
                    }
                }
            }
            for p in [0.05, 0.5, 0.95] {
                if let Err(e) = law.quantile(p) {
                    failures.push(format!("quantile {alpha} {beta} {p}: {e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn independent_quantile_fixtures() {
    assert_eq!(
        include_str!("../data/stable_quantiles.tsv")
            .lines()
            .filter(|s| !s.starts_with('#'))
            .count(),
        45 * 19
    );
    for row in include_str!("../data/stable_quantiles.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let v: Vec<f64> = row.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let law = LevyStable::new(v[0], v[1]).unwrap();
        let actual = law.quantile(v[2]).unwrap();
        assert!(
            (actual - v[3]).abs() <= 8e-8 * v[3].abs().max(1.),
            "{row}: {actual}"
        );
        let reflected = LevyStable::new(v[0], -v[1]).unwrap().isf(v[2]).unwrap();
        assert!(
            (reflected + v[3]).abs() <= 8e-8 * v[3].abs().max(1.),
            "isf {row}: {reflected}"
        );
    }
}

#[test]
fn independent_fourier_near_one() {
    assert_eq!(
        include_str!("../data/stable_near_one.tsv")
            .lines()
            .filter(|s| !s.starts_with('#'))
            .count(),
        72
    );
    for row in include_str!("../data/stable_near_one.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let v: Vec<f64> = row.split_whitespace().map(|s| s.parse().unwrap()).collect();
        for sign in [-1., 1.] {
            let law = LevyStable::new(v[0], sign * v[1]).unwrap();
            let x = sign * v[2];
            let (lc, ls) = if sign > 0. {
                (v[3], v[4])
            } else {
                (v[4], v[3])
            };
            for (actual, expected) in [
                (law.log_cdf(x).unwrap(), lc),
                (law.log_sf(x).unwrap(), ls),
                (law.log_pdf(x).unwrap(), v[5]),
            ] {
                assert!(
                    (actual - expected).abs() < 8e-8,
                    "{row}, sign={sign}: {actual} vs {expected}"
                );
            }
            let q = law.quantile(lc.exp()).unwrap();
            let ulp = (x.next_up() - x).max(x - x.next_down());
            assert!(
                (q - x).abs() <= 4. * ulp + 8e-8 / v[5].exp(),
                "inverse {row}: {q}"
            );
        }
    }
}

#[test]
fn independent_tail_fixtures() {
    assert_eq!(
        include_str!("../data/stable_tails.tsv")
            .lines()
            .filter(|s| !s.starts_with('#'))
            .count(),
        45 * 41
    );
    assert_eq!(
        include_str!("../data/stable_boundaries.tsv")
            .lines()
            .filter(|s| !s.starts_with('#'))
            .count(),
        27
    );
    for row in include_str!("../data/stable_tails.tsv")
        .lines()
        .chain(include_str!("../data/stable_boundaries.tsv").lines())
        .filter(|s| !s.starts_with('#'))
    {
        let v: Vec<f64> = row.split_whitespace().map(|s| s.parse().unwrap()).collect();
        let law = LevyStable::new(v[0], v[1]).unwrap();
        for (actual, expected) in [
            (law.cdf(v[2]).unwrap(), v[3].exp()),
            (law.sf(v[2]).unwrap(), v[4].exp()),
        ] {
            assert!(
                (actual - expected).abs() <= 8e-8 * expected + 4. * f64::from_bits(1),
                "probability {row}: {actual} vs {expected}"
            );
        }
        for (actual, expected) in [
            (law.log_cdf(v[2]).unwrap(), v[3]),
            (law.log_sf(v[2]).unwrap(), v[4]),
        ] {
            if expected.is_infinite() {
                assert_eq!(actual, expected);
            } else {
                let allowance = if expected < -700. {
                    8e-8 * expected.abs()
                } else {
                    8e-8
                };
                assert!(
                    (actual - expected).abs() <= allowance,
                    "{row}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn heavy_tail_references() {
    for row in include_str!("../data/stable_heavy.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<&str> = row.split('\t').collect();
        let values: Vec<f64> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
        let (alpha, beta) = (values[0], values[1]);
        let law = LevyStable::new(alpha, beta).unwrap();
        let mirror = LevyStable::new(alpha, -beta).unwrap();
        if fields[0] == "tail" {
            let (x, log_sf) = (values[2], values[3]);
            for actual in [law.log_sf(x).unwrap(), mirror.log_cdf(-x).unwrap()] {
                assert!(
                    (actual - log_sf).abs() <= 8e-8 * log_sf.abs().max(1.),
                    "{alpha} {beta} {x}: {actual} {log_sf}"
                );
            }
            assert!(law.log_pdf(x).unwrap().is_finite());
        } else {
            // A relative tail error e moves the quantile by about e / alpha.
            let (q, x) = (values[2], values[3]);
            let tolerance = 8e-8 / alpha * x;
            assert!((law.isf(q).unwrap() - x).abs() <= tolerance, "{alpha} {q}");
            assert!(
                (mirror.quantile(q).unwrap() + x).abs() <= tolerance,
                "{alpha} {q}"
            );
        }
    }
}

/// All 45 independent fixture shape pairs: exact shortcuts and CMS regimes,
/// including skew boundaries and both sides of alpha=1. Reusing the 19 retained
/// quantiles avoids numerical inversion in the sampling test. Seed 1335+case,
/// 32768 draws/case; the shared DKW/Pearson bounds apply at both widths.
fn stable_sampling<S: scioxide_core::RealScalar>() {
    use scioxide_random::{Distribution, SeedableRng, SplitMix64};
    let rows: Vec<Vec<f64>> = include_str!("../data/stable_quantiles.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|row| row.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    for (index, rows) in rows.chunks_exact(19).enumerate() {
        let s = |x| S::try_from_f64_lossy(x).unwrap();
        let law = LevyStable::new(s(rows[0][0]), s(rows[0][1])).unwrap();
        let cuts: Vec<_> = rows.iter().map(|row| (s(row[3]), row[2])).collect();
        let mut rng = SplitMix64::from_seed(1335 + index as u64);
        statistics::cdf_fit(
            &format!("Stable/{:?}", &rows[0][..2]),
            &cuts,
            statistics::DRAWS,
            statistics::reference_error::<S>(),
            || {
                let x = law.sample(&mut rng).unwrap();
                assert!(x.is_finite());
                x
            },
        );
    }
}
#[test]
fn stable_sampling_f32() {
    stable_sampling::<f32>();
}
#[test]
fn stable_sampling_f64() {
    stable_sampling::<f64>();
}

use crate::statistics;
