//! Independent numerical fixtures, support identities and calibrated seeded tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{Budget, RealScalar, Tolerance};
use scioxide_random::*;

fn check<S: RealScalar, X: Copy + core::fmt::Debug + PartialEq, L>(
    law: L,
    row: &[&str],
    parse: impl Fn(&str) -> X,
    shift: f64,
) where
    L: Mass<S, X>
        + LogMass<S, X>
        + Cdf<S, X>
        + Survival<S, X>
        + LogCdf<S, X>
        + LogSurvival<S, X>
        + Quantile<S, X>
        + InverseSurvival<S, X>
        + Mean<S>
        + Variance<S>,
{
    if row[0].starts_with("q:") {
        let p = S::try_from_f64_lossy(row[5].parse().unwrap()).unwrap();
        assert_eq!(law.quantile(p).unwrap(), parse(row[6]), "{row:?}");
        assert_eq!(law.isf(p).unwrap(), parse(row[7]), "{row:?}");
        return;
    }
    let k = parse(row[5]);
    let epsilon = S::EPSILON.into_f64();
    let tolerance = if epsilon > 1e-10 { 2e-6 } else { 5e-12 };
    let subnormal = if epsilon > 1e-10 {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    for (i, (log, prob)) in [law.log_pmf(k), law.log_cdf(k), law.log_sf(k)]
        .into_iter()
        .zip([law.pmf(k), law.cdf(k), law.sf(k)])
        .enumerate()
    {
        let expected: f64 = row[6 + i].parse().unwrap();
        let actual = log.unwrap_or_else(|e| panic!("{row:?}: {e:?}")).into_f64();
        if expected == f64::NEG_INFINITY {
            assert_eq!(actual, expected, "{row:?}");
        } else {
            assert!(
                (actual - expected).abs() <= tolerance + 2.0 * epsilon * expected.abs(),
                "{row:?}, log {i}: {actual} != {expected}"
            );
        }
        let actual = prob.unwrap().into_f64();
        let expected = expected.exp();
        assert!(
            (actual - expected).abs() <= tolerance * expected + subnormal,
            "{row:?}, prob {i}: {actual} != {expected}"
        );
    }
    for (i, actual) in [
        law.mean().unwrap().into_f64() - shift,
        law.variance().unwrap().into_f64(),
    ]
    .into_iter()
    .enumerate()
    {
        let expected: f64 = row[9 + i].parse().unwrap();
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{row:?}, moment {i}: {actual} != {expected}"
        );
    }
}
fn fixtures<S: RealScalar>() {
    for line in include_str!("../data/finite_laws.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let r: Vec<_> = line.split('\t').collect();
        let p = |i: usize| S::try_from_f64_lossy(r[i].parse().unwrap()).unwrap();
        let n = |i: usize| r[i].parse::<usize>().unwrap();
        let name = r[0].strip_prefix("q:").unwrap_or(r[0]);
        macro_rules! unsigned {
            ($law:expr) => {
                check::<S, usize, _>($law.unwrap(), &r, |s| s.parse().unwrap(), 0.0)
            };
        }
        macro_rules! signed {
            ($law:expr) => {
                check::<S, i64, _>($law.unwrap(), &r, |s| s.parse().unwrap(), 0.0)
            };
        }
        match name {
            "boltzmann" => unsigned!(Boltzmann::new(p(1), n(2))),
            "dlaplace" => signed!(DiscreteLaplace::new(p(1))),
            "hypergeom" => unsigned!(Hypergeometric::new(n(1), n(2), n(3))),
            "nchypergeom_fisher" => {
                unsigned!(FisherNoncentralHypergeometric::new(n(1), n(2), n(3), p(4)))
            }
            "nchypergeom_wallenius" => unsigned!(WalleniusNoncentralHypergeometric::new(
                n(1),
                n(2),
                n(3),
                p(4)
            )),
            "nhypergeom" if n(3) == 0 => unsigned!(Binomial::new(0, S::ZERO)),
            "nhypergeom" => unsigned!(BetaBinomial::new(
                n(2),
                p(3),
                S::try_from_f64_lossy((n(1) - n(2) - n(3) + 1) as f64).unwrap()
            )),
            "planck" => {
                let law = Geometric::from_rate(p(1)).unwrap();
                if !r[0].starts_with("q:") {
                    let expected = r[9].parse::<f64>().unwrap();
                    let actual = law.mean_failures().unwrap().into_f64();
                    let tol = if S::EPSILON.into_f64() > 1e-10 {
                        2e-6
                    } else {
                        5e-12
                    };
                    let floor = if S::EPSILON.into_f64() > 1e-10 {
                        f64::from(f32::from_bits(1))
                    } else {
                        f64::from_bits(1)
                    };
                    assert!(
                        (actual - expected).abs() <= tol * expected + floor,
                        "Planck mean {r:?}"
                    );
                }
                check::<S, usize, _>(law, &r, |s| s.parse::<usize>().unwrap() + 1, 1.0);
            }
            "randint" => signed!(DiscreteUniform::new(
                r[1].parse().unwrap(),
                r[2].parse().unwrap()
            )),
            _ => panic!("unknown fixture {name}"),
        }
    }
}
#[test]
fn independent_f64() {
    fixtures::<f64>();
}
#[test]
fn independent_f32() {
    fixtures::<f32>();
}

fn identities<S: RealScalar>() {
    for (m, n, d) in [
        (0, 0, 0),
        (20, 0, 6),
        (20, 20, 6),
        (20, 8, 20),
        (20, 8, 6),
        (20, 16, 18),
    ] {
        let h = Hypergeometric::<S>::new(m, n, d).unwrap();
        let f = FisherNoncentralHypergeometric::new(m, n, d, S::ONE).unwrap();
        let w = WalleniusNoncentralHypergeometric::new(m, n, d, S::ONE).unwrap();
        let (lo, hi) = h.count_support();
        let hi = hi.unwrap();
        assert_eq!(h.quantile(S::ZERO).unwrap(), lo);
        assert_eq!(h.isf(S::ZERO).unwrap(), hi);
        let mut total = 0.0;
        for k in 0..=m + 1 {
            let expected = h.log_pmf(k).unwrap().into_f64();
            for other in [f.log_pmf(k).unwrap(), w.log_pmf(k).unwrap()] {
                let actual = other.into_f64();
                assert!(
                    actual == expected
                        || (actual - expected).abs()
                            <= 256.0 * S::EPSILON.into_f64() * expected.abs().max(1.0),
                    "({m}, {n}, {d}), k={k}: {actual} vs {expected}"
                );
            }
            total += h.pmf(k).unwrap().into_f64();
        }
        assert!((total - 1.0).abs() < 1e-6);
    }
    let f =
        FisherNoncentralHypergeometric::new(20, 8, 6, S::try_from_f64_lossy(2.0).unwrap()).unwrap();
    let w = WalleniusNoncentralHypergeometric::new(20, 8, 6, S::try_from_f64_lossy(2.0).unwrap())
        .unwrap();
    assert!((f.mean().unwrap().into_f64() - w.mean().unwrap().into_f64()).abs() > 0.1);
}
#[test]
fn reductions_both_widths() {
    identities::<f32>();
    identities::<f64>();
}

#[test]
fn exact_signed_uniform_inversion_and_endpoints() {
    let law = DiscreteUniform::<f64>::new(i64::MIN, i64::MAX).unwrap();
    assert_eq!(law.quantile(0.5).unwrap(), -1);
    assert_eq!(law.isf(0.5).unwrap(), -1);
    assert_eq!(law.quantile(f64::from_bits(1)).unwrap(), i64::MIN);
    assert_eq!(law.isf(f64::from_bits(1)).unwrap(), i64::MAX - 1);
    for (lo, hi) in [
        (-9, 12),
        (i64::MIN, i64::MIN + 16),
        (i64::MAX - 16, i64::MAX),
    ] {
        let law = DiscreteUniform::<f64>::new(lo, hi).unwrap();
        assert_eq!(law.quantile(0.0).unwrap(), lo);
        assert_eq!(law.quantile(1.0).unwrap(), hi - 1);
        for numerator in 1..16 {
            let p = f64::from(numerator) / 16.0;
            let expected =
                i128::from(lo) + (i128::from(hi) - i128::from(lo)) * i128::from(numerator) / 16;
            let exact = (i128::from(hi) - i128::from(lo)) * i128::from(numerator) % 16 == 0;
            assert_eq!(
                law.quantile(p).unwrap() as i128,
                expected - i128::from(exact)
            );
        }
    }
}

#[test]
fn domains_budgets_and_rate_limits() {
    let options = LawOptions::default()
        .with_tolerance(Tolerance::<f64>::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE);
    let law = Hypergeometric::with_options(20, 8, 6, options).unwrap();
    let error = law.cdf(2).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let error =
        WalleniusNoncentralHypergeometric::with_options(20, 8, 6, 2.0, options).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert!(Hypergeometric::with_options(usize::MAX, usize::MAX, 6, options).is_ok());
    let error = Hypergeometric::<f64>::new(2, 3, 1).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    for x in [0.0, -1.0] {
        let error = Boltzmann::new(x, 2).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        let error = DiscreteLaplace::new(x).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        let error = Geometric::from_rate(x).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        let error = FisherNoncentralHypergeometric::new(20, 8, 6, x).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }
    for x in [f64::NAN, f64::INFINITY] {
        let error = WalleniusNoncentralHypergeometric::new(20, 8, 6, x).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        let error = DiscreteLaplace::new(x).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }
    assert!(Boltzmann::new(0.5, 0).is_err());
    assert!(DiscreteUniform::<f32>::new(1, 1).is_err());
    let g = Geometric::from_rate(1000.0).unwrap();
    assert_eq!(g.count_support(), (1, None));
    assert_eq!(g.log_sf(1).unwrap(), -1000.0);
    assert_eq!(g.log_pmf(2).unwrap(), -1000.0);
    assert!(g.quantile(1.0).is_err());
    assert!(DiscreteLaplace::new(0.5).unwrap().quantile(0.0).is_err());
}

// Seed 533; shared bounds use 32768 draws and explicit outside-tail bins.
fn seeded<S: RealScalar>() {
    let p = |v| S::try_from_f64_lossy(v).unwrap();
    macro_rules! run {
        ($law:expr,$lo:expr,$hi:expr,$signed:ty,$shift:expr) => {{
            let law = $law.unwrap();
            let sampler = &law;
            let mut rng = SplitMix64::from_seed(533);
            statistics::discrete(
                stringify!($law),
                $lo,
                $hi,
                |k| {
                    if k + $shift < 0 && $lo >= 0 {
                        0.0
                    } else {
                        law.pmf((k + $shift) as $signed).unwrap().into_f64()
                    }
                },
                |k| {
                    if k + $shift < 0 && $lo >= 0 {
                        0.0
                    } else {
                        law.cdf((k + $shift) as $signed).unwrap().into_f64()
                    }
                },
                statistics::reference_error::<S>(),
                || sampler.sample(&mut rng).unwrap() as i64 - $shift,
            );
        }};
    }
    run!(Boltzmann::new(p(0.5), 16), 0, 15, usize, 0);
    for (population, marked, draws) in [
        (50, 20, 12),
        (100, 7, 30),
        (100, 80, 60),
        (10, 0, 4),
        (10, 10, 4),
    ] {
        run!(
            Hypergeometric::<S>::new(population, marked, draws),
            0,
            draws as i64,
            usize,
            0
        );
        for odds in [0.2, 1.0, 5.0] {
            run!(
                FisherNoncentralHypergeometric::new(population, marked, draws, p(odds)),
                0,
                draws as i64,
                usize,
                0
            );
            run!(
                WalleniusNoncentralHypergeometric::new(population, marked, draws, p(odds)),
                0,
                draws as i64,
                usize,
                0
            );
        }
    }
    run!(
        FisherNoncentralHypergeometric::new(50, 20, 12, p(2.0)),
        0,
        12,
        usize,
        0
    );
    run!(
        WalleniusNoncentralHypergeometric::new(50, 20, 12, p(2.0)),
        0,
        12,
        usize,
        0
    );
    run!(BetaBinomial::new(20, p(12.0), p(19.0)), 0, 20, usize, 0);
    run!(Geometric::from_rate(p(0.5)), 0, 64, usize, 1);
    run!(DiscreteLaplace::new(p(0.5)), -64, 64, i64, 0);
    run!(DiscreteUniform::<S>::new(-10, 11), -10, 10, i64, 0);
}
#[test]
fn seeded_f64() {
    seeded::<f64>();
}
#[test]
fn seeded_f32() {
    seeded::<f32>();
}

#[test]
fn shifted_geometric_retains_small_log_probabilities_and_mean() {
    let rate = 64.0_f64;
    let g = Geometric::from_rate(rate).unwrap();
    let q = (-rate).exp();
    assert_eq!(g.mean().unwrap(), 1.0);
    assert!((g.mean_failures().unwrap() / q - 1.0).abs() < 5e-15);
    assert!((g.log_pmf(1).unwrap() / (-q) - 1.0).abs() < 5e-15);
    assert!((g.log_cdf(1).unwrap() / (-q) - 1.0).abs() < 5e-15);
    assert!(DiscreteLaplace::new(f64::MAX).unwrap().log_pmf(2).is_err());
    assert!(Boltzmann::new(f64::MAX, 4).unwrap().log_sf(1).is_err());
    assert!(Geometric::from_rate(f64::MAX).unwrap().log_sf(2).is_err());
}

#[test]
fn symmetric_urn_medians_choose_the_first_half_probability_crossing() {
    for population in [20, 50, 100] {
        for draws in [1, 3, 5, 11, 19] {
            let law = Hypergeometric::<f64>::new(population, population / 2, draws).unwrap();
            // Color reflection pairs every atom across the half-integer center.
            assert_eq!(
                law.quantile(0.5).unwrap(),
                draws / 2,
                "M={population}, N={draws}"
            );
            assert_eq!(
                law.isf(0.5).unwrap(),
                draws / 2,
                "M={population}, N={draws}"
            );
        }
    }
}

#[path = "../support/statistics.rs"]
mod statistics;
