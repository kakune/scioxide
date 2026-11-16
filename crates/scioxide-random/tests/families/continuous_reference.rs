//! Independent 80-digit defining-formula fixtures for all 53 pinned names.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_random::{Affine, SeedableRng, SplitMix64};
use std::boxed::Box;
use std::vec::Vec;

#[path = "../support/laws.rs"]
mod laws;
use laws::CheckedLaw;

fn make<S: RealScalar>(name: &str) -> Box<dyn CheckedLaw<S>> {
    let c = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let z = S::ZERO;
    let o = S::ONE;
    match name {
        "cauchy" => Box::new(scioxide_random::Cauchy::new(z, o).unwrap()),
        "cosine" => Box::new(scioxide_random::Cosine::<S>::new().unwrap()),
        "dgamma" => Box::new(scioxide_random::DoubleGamma::new(c(2.)).unwrap()),
        "dweibull" => Box::new(scioxide_random::DoubleWeibull::new(c(1.5)).unwrap()),
        "genlogistic" => Box::new(scioxide_random::GeneralizedLogistic::new(c(2.)).unwrap()),
        "gennorm" => Box::new(scioxide_random::GeneralizedNormal::new(c(2.)).unwrap()),
        "gumbel_r" => Box::new(scioxide_random::GumbelRight::<S>::new()),
        "gumbel_l" => {
            Box::new(Affine::new(scioxide_random::GumbelRight::<S>::new(), z, -o).unwrap())
        }
        "hypsecant" => Box::new(scioxide_random::HyperbolicSecant::<S>::new()),
        "laplace" => Box::new(scioxide_random::DoubleWeibull::new(o).unwrap()),
        "laplace_asymmetric" => Box::new(scioxide_random::AsymmetricLaplace::new(c(2.)).unwrap()),
        "logistic" => Box::new(scioxide_random::Logistic::new(z, o).unwrap()),
        "norm" => Box::new(scioxide_random::Normal::new(z, o).unwrap()),
        "t" => Box::new(scioxide_random::StudentT::new(c(5.)).unwrap()),
        "uniform" => Box::new(scioxide_random::Uniform::new(z, o).unwrap()),
        "alpha" => Box::new(scioxide_random::Alpha::new(c(2.)).unwrap()),
        "anglit" => Box::new(scioxide_random::Anglit::<S>::new()),
        "arcsine" => Box::new(scioxide_random::Beta::new(c(0.5), c(0.5)).unwrap()),
        "argus" => Box::new(scioxide_random::Argus::new(c(2.)).unwrap()),
        "beta" => Box::new(scioxide_random::Beta::new(c(2.), c(3.)).unwrap()),
        "betaprime" => Box::new(scioxide_random::BetaPrime::new(c(2.), c(3.)).unwrap()),
        "bradford" => Box::new(scioxide_random::Bradford::new(c(2.)).unwrap()),
        "gausshyper" => Box::new(
            scioxide_random::GaussHypergeometric::new(c(2.), c(3.), c(0.5), c(0.3)).unwrap(),
        ),
        "irwinhall" => Box::new(scioxide_random::IrwinHall::new(5).unwrap()),
        "johnsonsb" => Box::new(scioxide_random::JohnsonSB::new(c(0.5), c(2.)).unwrap()),
        "kappa3" => Box::new(
            Affine::new(
                scioxide_random::Mielke::new(o, c(2.)).unwrap(),
                z,
                c(2.).sqrt(),
            )
            .unwrap(),
        ),
        "kappa4" => Box::new(scioxide_random::Kappa4::new(c(0.5), c(0.3)).unwrap()),
        "mielke" => Box::new(scioxide_random::Mielke::new(c(2.), c(3.)).unwrap()),
        "powerlaw" => Box::new(scioxide_random::Beta::new(c(2.), o).unwrap()),
        "rdist" | "semicircular" => Box::new(
            Affine::new(
                scioxide_random::Beta::new(c(1.5), c(1.5)).unwrap(),
                -o,
                c(2.),
            )
            .unwrap(),
        ),
        "trapezoid" => Box::new(scioxide_random::Trapezoid::new(c(0.2), c(0.8)).unwrap()),
        "triang" => Box::new(scioxide_random::Trapezoid::new(c(0.5), c(0.5)).unwrap()),
        "tukeylambda" => Box::new(scioxide_random::TukeyLambda::new(c(0.5)).unwrap()),
        "chi" => Box::new(
            Affine::new(
                scioxide_random::GeneralizedGamma::new(c(2.5), c(2.)).unwrap(),
                z,
                c(2.).sqrt(),
            )
            .unwrap(),
        ),
        "chi2" => Box::new(scioxide_random::ChiSquared::new(c(5.)).unwrap()),
        "erlang" => Box::new(scioxide_random::Gamma::new(c(3.), o).unwrap()),
        "expon" => Box::new(scioxide_random::Exponential::new(o).unwrap()),
        "exponpow" => Box::new(scioxide_random::ExponentialPower::new(c(2.)).unwrap()),
        "fatiguelife" => Box::new(scioxide_random::FatigueLife::new(c(0.5)).unwrap()),
        "gamma" => Box::new(scioxide_random::Gamma::new(c(2.), o).unwrap()),
        "gengamma" => Box::new(scioxide_random::GeneralizedGamma::new(c(2.), c(1.5)).unwrap()),
        "gompertz" => Box::new(scioxide_random::Gompertz::new(c(2.)).unwrap()),
        "invgamma" => Box::new(scioxide_random::GeneralizedGamma::new(c(3.), -o).unwrap()),
        "invgauss" => Box::new(scioxide_random::InverseGaussian::new(o).unwrap()),
        "maxwell" => Box::new(
            Affine::new(
                scioxide_random::GeneralizedGamma::new(c(1.5), c(2.)).unwrap(),
                z,
                c(2.).sqrt(),
            )
            .unwrap(),
        ),
        "nakagami" => Box::new(
            Affine::new(
                scioxide_random::GeneralizedGamma::new(c(2.), c(2.)).unwrap(),
                z,
                o / c(2.).sqrt(),
            )
            .unwrap(),
        ),
        "pearson3" => Box::new(scioxide_random::Pearson3::new(o).unwrap()),
        "rayleigh" => Box::new(
            Affine::new(
                scioxide_random::WeibullMin::new(c(2.)).unwrap(),
                z,
                c(2.).sqrt(),
            )
            .unwrap(),
        ),
        "recipinvgauss" => Box::new(scioxide_random::ReciprocalInverseGaussian::new(o).unwrap()),
        "wald" => Box::new(scioxide_random::InverseGaussian::new(o).unwrap()),
        "weibull_min" => Box::new(scioxide_random::WeibullMin::new(c(2.)).unwrap()),
        "weibull_max" => {
            Box::new(Affine::new(scioxide_random::WeibullMin::new(c(2.)).unwrap(), z, -o).unwrap())
        }
        _ => unreachable!("unknown law fixture {name}"),
    }
}

fn make_extreme<S: RealScalar>(name: &str) -> Box<dyn CheckedLaw<S>> {
    let c = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let o = S::ONE;
    match name {
        "dgamma" => Box::new(scioxide_random::DoubleGamma::new(c(0.2)).unwrap()),
        "dweibull" => Box::new(scioxide_random::DoubleWeibull::new(c(0.2)).unwrap()),
        "genlogistic" => Box::new(scioxide_random::GeneralizedLogistic::new(c(10.)).unwrap()),
        "gennorm" => Box::new(scioxide_random::GeneralizedNormal::new(c(0.5)).unwrap()),
        "laplace_asymmetric" => Box::new(scioxide_random::AsymmetricLaplace::new(c(0.1)).unwrap()),
        "alpha" => Box::new(scioxide_random::Alpha::new(c(0.1)).unwrap()),
        "argus" => Box::new(scioxide_random::Argus::new(c(0.2)).unwrap()),
        "betaprime" => Box::new(scioxide_random::BetaPrime::new(c(0.3), c(0.4)).unwrap()),
        "bradford" => Box::new(scioxide_random::Bradford::new(c(0.001)).unwrap()),
        "gausshyper" => Box::new(
            scioxide_random::GaussHypergeometric::new(c(0.5), c(0.5), c(2.), c(-0.9)).unwrap(),
        ),
        "irwinhall" => Box::new(scioxide_random::IrwinHall::<S>::new(20).unwrap()),
        "johnsonsb" => Box::new(scioxide_random::JohnsonSB::new(c(5.), c(0.2)).unwrap()),
        "kappa4" => Box::new(scioxide_random::Kappa4::new(c(-0.5), c(0.2)).unwrap()),
        "mielke" => Box::new(scioxide_random::Mielke::new(c(0.5), c(0.8)).unwrap()),
        "trapezoid" => Box::new(scioxide_random::Trapezoid::new(S::ZERO, o).unwrap()),
        "tukeylambda" => Box::new(scioxide_random::TukeyLambda::new(c(-0.5)).unwrap()),
        "exponpow" => Box::new(scioxide_random::ExponentialPower::new(c(0.4)).unwrap()),
        "fatiguelife" => Box::new(scioxide_random::FatigueLife::new(c(3.)).unwrap()),
        "gengamma" => Box::new(scioxide_random::GeneralizedGamma::new(c(0.5), -o).unwrap()),
        "gompertz" => Box::new(scioxide_random::Gompertz::new(c(0.1)).unwrap()),
        "invgauss" => Box::new(scioxide_random::InverseGaussian::new(c(0.1)).unwrap()),
        "recipinvgauss" => {
            Box::new(scioxide_random::ReciprocalInverseGaussian::new(c(10.)).unwrap())
        }
        "pearson3" => Box::new(scioxide_random::Pearson3::new(c(-2.)).unwrap()),
        "weibull_min" => Box::new(scioxide_random::WeibullMin::new(c(0.3)).unwrap()),
        _ => unreachable!("unknown extreme fixture {name}"),
    }
}

fn fixtures<S: RealScalar>(tol: f64) {
    let data = include_str!("../data/continuous_laws.tsv");
    let mut rows = 0;
    for row in data.lines().filter(|line| !line.starts_with('#')).skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(fields.len(), 5);
        let name = fields[0];
        let value = |i: usize| fields[i].parse::<f64>().unwrap();
        let x = S::try_from_f64_lossy(value(1)).unwrap();
        let law = make::<S>(name);
        let got = [
            law.log_pdf(x).unwrap().into_f64(),
            law.log_cdf(x).unwrap().into_f64(),
            law.log_sf(x).unwrap().into_f64(),
        ];
        let expected = [value(2), value(3), value(4)];
        for (i, (actual, reference)) in got.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - reference).abs() <= tol * (1. + reference.abs()),
                "{name} column {i}: actual={actual} reference={reference}"
            );
        }
        let p = expected[1].exp();
        let q = expected[2].exp();
        let (inverse, probability) = if p < 0.9 { (false, p) } else { (true, q) };
        if probability
            > if S::MANTISSA_DIGITS == 24 {
                1e-30
            } else {
                1e-290
            }
        {
            let prob = S::try_from_f64_lossy(probability).unwrap();
            let reconstructed = if inverse {
                law.isf(prob)
            } else {
                law.quantile(prob)
            }
            .unwrap()
            .into_f64();
            assert!(
                (reconstructed - x.into_f64()).abs() < tol * 20. * (1. + x.into_f64().abs()),
                "{name} inverse: x={x:?} got={reconstructed}"
            );
        }
        rows += 1;
    }
    assert_eq!(rows, 106);
}

#[test]
fn defining_formula_fixtures_f64() {
    fixtures::<f64>(3e-6)
}
#[test]
fn defining_formula_fixtures_f32() {
    fixtures::<f32>(3e-3)
}

fn range_law<S: RealScalar>(name: &str, a: S, b: S) -> Box<dyn CheckedLaw<S>> {
    use scioxide_random::*;
    match name {
        "norm" => Box::new(Normal::new(a, b).unwrap()),
        "lognorm" => Box::new(LogNormal::new(a, b).unwrap()),
        "gamma" => Box::new(Gamma::new(a, b).unwrap()),
        "gengamma" => Box::new(GeneralizedGamma::new(a, b).unwrap()),
        "gennorm" => Box::new(GeneralizedNormal::new(a).unwrap()),
        "genlogistic" => Box::new(GeneralizedLogistic::new(a).unwrap()),
        "mielke" => Box::new(Mielke::new(a, b).unwrap()),
        "kappa4" => Box::new(Kappa4::new(a, b).unwrap()),
        "argus" => Box::new(Argus::new(a).unwrap()),
        "trapezoid" => Box::new(Trapezoid::new(a, b).unwrap()),
        "tukeylambda" => Box::new(TukeyLambda::new(a).unwrap()),
        "kolmogorov" => Box::new(Kolmogorov::new().unwrap()),
        "pearson3" => Box::new(Pearson3::new(a).unwrap()),
        _ => unreachable!("unknown range fixture {name}"),
    }
}
fn extreme_fixtures<S: RealScalar>(tol: f64) {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    let width = if S::MANTISSA_DIGITS == 24 {
        "f32"
    } else {
        "f64"
    };
    let c = |x: &str| S::try_from_f64_lossy(x.parse::<f64>().unwrap()).unwrap();
    let mut rows = 0;
    for row in include_str!("../data/continuous_extreme_laws.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
    {
        let f: Vec<_> = row.split('\t').collect();
        assert_eq!(f.len(), 9);
        if f[5] != "both" && f[5] != width {
            continue;
        }
        let law = if f[5] == "both" {
            make_extreme::<S>(f[0])
        } else {
            range_law::<S>(f[0], c(f[1]), c(f[2]))
        };
        let x = c(f[7]);
        let got = match f[6] {
            "pdf" => law.pdf(x),
            "cdf" => law.cdf(x),
            "sf" => law.sf(x),
            "log_pdf" => law.log_pdf(x),
            "log_cdf" => law.log_cdf(x),
            "log_sf" => law.log_sf(x),
            "quantile" => law.quantile(x),
            "isf" => law.isf(x),
            _ => unreachable!("unknown query"),
        };
        if f[8] == "error:Underflow" {
            assert!(
                matches!(
                    got.unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Underflow,
                        ..
                    }
                ),
                "{row}"
            );
        } else {
            assert!(got.is_ok(), "{row}: {got:?}");
            let value = got.unwrap();
            let actual = value.into_f64();
            let expected = f[8].parse::<f64>().unwrap();
            let tolerance = if f[5] == "both" {
                tol
            } else if width == "f64" {
                8e-8
            } else {
                8.0 * f64::from(f32::EPSILON).sqrt()
            };
            let scale = if f[6].starts_with("log_") {
                let log_subnormal = S::MIN_POSITIVE.into_f64().ln()
                    - (S::MANTISSA_DIGITS as f64 - 1.0) * core::f64::consts::LN_2;
                if f[5] != "both" && f[6] != "log_pdf" && expected >= log_subnormal {
                    1.0
                } else {
                    1.0 + expected.abs()
                }
            } else {
                expected.abs()
            };
            let boundary = if f[6] == "isf" && f[0] == "trapezoid" {
                4.0 * S::EPSILON.into_f64()
            } else {
                0.0
            };
            assert!(
                actual == expected || (actual - expected).abs() <= tolerance * scale + boundary,
                "{row}: got {actual:e} expected {expected:e}"
            );
            if f[6] == "quantile" || f[6] == "isf" {
                let (low, high) = law.support().unwrap().bounds().unwrap();
                assert!(
                    low != Some(value) && high != Some(value),
                    "{row}: interior endpoint"
                );
            }
        }
        rows += 1;
    }
    assert!(rows >= 110, "missing range fixtures: {rows}");
}
#[test]
fn extreme_parameter_defining_formulas() {
    extreme_fixtures::<f64>(3e-6);
    extreme_fixtures::<f32>(3e-3)
}

#[test]
fn seeded_probability_bins_all_53_f64() {
    seeded_bins::<f64>();
}
#[test]
fn seeded_probability_bins_all_53_f32() {
    seeded_bins::<f32>();
}
fn seeded_bins<S: RealScalar>() {
    let data = include_str!("../data/continuous_laws.tsv");
    let mut count = 0usize;
    for row in data
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .step_by(2)
    {
        let fields: Vec<_> = row.split('\t').collect();
        let name = fields[0];
        let law = make::<S>(name);
        let mut rng = SplitMix64::from_seed(534536 + count as u64);
        if name == "gausshyper" {
            // Nested quadrature inside every inverse makes default binary64
            // sampling dominate debug runtime. A 1e-4 quadrature target is
            // sufficient for this .03-power test; compare with default-accuracy
            // fixture-validated cuts and retain a separate .003 numerical band.
            // Default numerical accuracy is still exercised by the fixtures.
            let c = |x| S::try_from_f64_lossy(x).unwrap();
            let sampler = scioxide_random::GaussHypergeometric::with_options(
                c(2.0),
                c(3.0),
                c(0.5),
                c(0.3),
                scioxide_random::LawOptions::default()
                    .with_tolerance(scioxide_core::Tolerance::new(c(1e-4), c(1e-4)).unwrap()),
            )
            .unwrap();
            use scioxide_random::Distribution;
            statistics::continuous_with_error(name, &*law, 0.003, || {
                sampler.sample(&mut rng).unwrap()
            });
        } else {
            statistics::continuous(name, &*law, || law.draw(&mut rng).unwrap());
        }
        count += 1;
    }
    assert_eq!(count, 53);
}

#[test]
fn bounded_seeded_means_with_family_error_control() {
    bounded_means::<f64>();
    bounded_means::<f32>();
}
fn bounded_means<S: RealScalar>() {
    let rows = include_str!("../data/continuous_bounded_means.tsv");
    let n = 1024usize;
    // Complementary fixture mean checks: Hoeffding gives alpha<=5e-7 each.
    // Distributional power comes from the separate 32768-draw bin tests.
    let threshold = ((2.0 / 5e-7_f64).ln() / (2.0 * n as f64)).sqrt();
    let mut count = 0;
    for row in rows.lines().filter(|line| !line.starts_with('#')).skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(fields.len(), 4);
        let values: Vec<f64> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
        let (low, high, mean) = (values[0], values[1], values[2]);
        let law = make::<S>(fields[0]);
        let mut rng = SplitMix64::from_seed(535000 + count as u64);
        let mut total = 0.;
        for _ in 0..n {
            total += (law.draw(&mut rng).unwrap().into_f64() - low) / (high - low);
        }
        let observed = total / n as f64;
        let expected = (mean - low) / (high - low);
        assert!(
            (observed - expected).abs() <= threshold,
            "{} bounded mean {observed} versus {expected}; family threshold {threshold}",
            fields[0]
        );
        count += 1;
    }
    assert_eq!(count, 17);
}

#[test]
fn declared_moments_match_independent_means() {
    use scioxide_random::Mean;
    let argus = scioxide_random::Argus::<f64>::new(2.).unwrap();
    assert!((argus.mean().unwrap() - 0.7056585155030369).abs() < 1e-8);
    let johnson = scioxide_random::JohnsonSB::<f64>::new(0.5, 2.).unwrap();
    assert!((johnson.mean().unwrap() - 0.4412484512922977).abs() < 1e-8);
    let gh = scioxide_random::GaussHypergeometric::<f64>::new(2., 3., 0.5, 0.3).unwrap();
    assert!((gh.mean().unwrap() - 0.3946880274762808).abs() < 1e-8);
}

#[test]
fn independent_bounded_moment_fixtures() {
    moment_fixtures::<f64>(2e-6);
    moment_fixtures::<f32>(5e-3);
}
fn moment_fixtures<S: RealScalar>(tol: f64) {
    use scioxide_random::{Mean, Variance};
    let c = |x: f64| S::try_from_f64_lossy(x).unwrap();
    for row in include_str!("../data/continuous_moments.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
    {
        let f: Vec<_> = row.split('\t').collect();
        let expected = [f[1].parse::<f64>().unwrap(), f[2].parse::<f64>().unwrap()];
        let got = match f[0] {
            "exponpow" => {
                let law = scioxide_random::ExponentialPower::new(c(2.)).unwrap();
                [
                    law.mean().unwrap().into_f64(),
                    law.variance().unwrap().into_f64(),
                ]
            }
            "gompertz" => {
                let law = scioxide_random::Gompertz::new(c(2.)).unwrap();
                [
                    law.mean().unwrap().into_f64(),
                    law.variance().unwrap().into_f64(),
                ]
            }
            "johnsonsb" => {
                let law = scioxide_random::JohnsonSB::new(c(0.5), c(2.)).unwrap();
                [
                    law.mean().unwrap().into_f64(),
                    law.variance().unwrap().into_f64(),
                ]
            }
            "kappa4" => {
                let law = scioxide_random::Kappa4::new(c(0.5), c(0.3)).unwrap();
                [
                    law.mean().unwrap().into_f64(),
                    law.variance().unwrap().into_f64(),
                ]
            }
            "argus" => {
                let law = scioxide_random::Argus::new(c(2.)).unwrap();
                [
                    law.mean().unwrap().into_f64(),
                    law.variance().unwrap().into_f64(),
                ]
            }
            _ => unreachable!("unknown moment fixture"),
        };
        for i in 0..2 {
            assert!(
                (got[i] - expected[i]).abs() <= tol * (1. + expected[i].abs()),
                "{} moment {i}: got={} expected={}",
                f[0],
                got[i],
                expected[i]
            );
        }
    }
}

use crate::statistics;
