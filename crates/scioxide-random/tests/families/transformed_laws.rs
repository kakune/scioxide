#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::RealScalar;
use scioxide_random::{
    Affine, Cauchy, DoubleParetoLogNormal, DoubleWeibull, Exponential, ExponentiallyModifiedNormal,
    ExponentiatedWeibull, Gamma, GeneralizedExponential, GeneralizedHalfLogistic,
    GeneralizedNormal, JohnsonSU, LogCdf, LogNormal, LogSurvival, Logistic, Normal, PowerNormal,
    SeedableRng, SplitMix64, Support, Transformed, Truncated, TruncatedPareto, Uniform, WeibullMin,
};

#[path = "../support/laws.rs"]
mod laws;
use laws::CheckedLaw;

fn law<S: RealScalar>(name: &str, a: S, b: S, c: S, d: S) -> Box<dyn CheckedLaw<S>> {
    let zero = S::ZERO;
    let one = S::ONE;
    match name {
        "dpareto_lognorm" => Box::new(DoubleParetoLogNormal::new(a, b, c, d).unwrap()),
        "exponnorm" => Box::new(ExponentiallyModifiedNormal::new(a).unwrap()),
        "exponweib" => Box::new(ExponentiatedWeibull::new(a, b).unwrap()),
        "foldcauchy" => Box::new(Transformed::abs(Cauchy::new(a, one).unwrap()).unwrap()),
        "foldnorm" => Box::new(Transformed::abs(Normal::new(a, one).unwrap()).unwrap()),
        "genexpon" => Box::new(GeneralizedExponential::new(a, b, c).unwrap()),
        "genhalflogistic" => Box::new(GeneralizedHalfLogistic::new(a).unwrap()),
        "gibrat" => Box::new(LogNormal::new(zero, one).unwrap()),
        "halfcauchy" => Box::new(Transformed::abs(Cauchy::new(zero, one).unwrap()).unwrap()),
        "halflogistic" => Box::new(Transformed::abs(Logistic::new(zero, one).unwrap()).unwrap()),
        "halfnorm" => Box::new(Transformed::abs(Normal::new(zero, one).unwrap()).unwrap()),
        "halfgennorm" => Box::new(Transformed::abs(GeneralizedNormal::new(a).unwrap()).unwrap()),
        "johnsonsu" => Box::new(JohnsonSU::new(a, b).unwrap()),
        "loggamma" => Box::new(Transformed::log(Gamma::new(a, one).unwrap()).unwrap()),
        "loglaplace" => Box::new(
            Transformed::exp(Affine::new(DoubleWeibull::new(one).unwrap(), zero, one / a).unwrap())
                .unwrap(),
        ),
        "lognorm" => Box::new(LogNormal::new(zero, a).unwrap()),
        "loguniform" => Box::new(Transformed::exp(Uniform::new(a.ln(), b.ln()).unwrap()).unwrap()),
        "powerlognorm" => Box::new(
            Transformed::exp(Affine::new(PowerNormal::new(a).unwrap(), zero, b).unwrap()).unwrap(),
        ),
        "powernorm" => Box::new(PowerNormal::new(a).unwrap()),
        "truncexpon" => Box::new(Truncated::new(Exponential::new(one).unwrap(), zero, a).unwrap()),
        "truncnorm" => Box::new(Truncated::new(Normal::new(zero, one).unwrap(), a, b).unwrap()),
        "truncpareto" => Box::new(TruncatedPareto::new(a, b).unwrap()),
        "truncweibull_min" => Box::new(Truncated::new(WeibullMin::new(a).unwrap(), b, c).unwrap()),
        _ => panic!("unknown law {name}"),
    }
}

fn fixture_rows<S: RealScalar>(tol: f64) {
    for row in include_str!("../data/transformed_laws.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let cols: Vec<_> = row.split_whitespace().collect();
        let nums: Vec<f64> = cols[1..].iter().map(|x| x.parse().unwrap()).collect();
        let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
        let d = law(cols[0], s(nums[0]), s(nums[1]), s(nums[2]), s(nums[3]));
        let x = s(nums[4]);
        for (label, got, want) in [
            ("pdf", d.pdf(x).unwrap().into_f64(), nums[5]),
            ("cdf", d.cdf(x).unwrap().into_f64(), nums[6]),
            ("sf", d.sf(x).unwrap().into_f64(), nums[7]),
        ] {
            assert!(
                (got - want).abs() <= tol * (1. + want.abs()),
                "{row}: {label} {got} != {want}"
            );
        }
        let q = d.quantile(s(nums[6])).unwrap().into_f64();
        let u = d.isf(s(nums[7])).unwrap().into_f64();
        assert!(
            (q - nums[4]).abs() <= tol * 30. * (1. + nums[4].abs()),
            "{row}: quantile {q}"
        );
        assert!(
            (u - nums[4]).abs() <= tol * 30. * (1. + nums[4].abs()),
            "{row}: isf {u}"
        );
        assert!(d.support().unwrap().contains(x));
    }
}
#[test]
fn independent_f64_fixtures() {
    fixture_rows::<f64>(3e-8);
}
#[test]
fn independent_f32_fixtures() {
    fixture_rows::<f32>(8e-4);
}

fn seeded_bins<S: RealScalar>() {
    for (index, row) in include_str!("../data/transformed_laws.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .step_by(2)
        .enumerate()
    {
        let cols: Vec<_> = row.split_whitespace().collect();
        let nums: Vec<f64> = cols[1..].iter().map(|x| x.parse().unwrap()).collect();
        let s = |x: f64| S::try_from_f64_lossy(x).unwrap();
        let d = law(cols[0], s(nums[0]), s(nums[1]), s(nums[2]), s(nums[3]));
        let mut rng = SplitMix64::from_seed(538 + index as u64);
        statistics::continuous(row, &*d, || d.draw(&mut rng).unwrap());
    }
}
#[test]
fn seeded_f64_bins() {
    seeded_bins::<f64>();
}
#[test]
fn seeded_f32_bins() {
    seeded_bins::<f32>();
}

#[test]
fn domains_and_moment_boundaries() {
    use scioxide_random::{Mean, Variance};
    assert!(DoubleParetoLogNormal::new(0., 1., 0., 1.).is_err());
    assert!(ExponentiatedWeibull::new(1., 0.).is_err());
    assert!(GeneralizedExponential::new(1., -1., 1.).is_err());
    assert!(GeneralizedHalfLogistic::new(0.).is_err());
    assert!(JohnsonSU::new(0., 0.).is_err());
    assert!(PowerNormal::new(-1.).is_err());
    assert!(TruncatedPareto::new(0., 2.).is_err());
    assert!(TruncatedPareto::new(1., 1.).is_err());
    assert!(
        DoubleParetoLogNormal::new(0., 1., 1., 2.)
            .unwrap()
            .mean()
            .is_err()
    );
    assert!(
        DoubleParetoLogNormal::new(0., 1., 2., 2.)
            .unwrap()
            .variance()
            .is_err()
    );
    assert!(
        DoubleParetoLogNormal::new(0., 1., 3., 2.)
            .unwrap()
            .variance()
            .unwrap()
            > 0.
    );
    assert_eq!(
        GeneralizedHalfLogistic::new(2.)
            .unwrap()
            .support()
            .unwrap()
            .bounds(),
        Some((Some(0.), Some(0.5)))
    );
    assert_eq!(
        TruncatedPareto::new(-1.5, 7.)
            .unwrap()
            .support()
            .unwrap()
            .bounds(),
        Some((Some(1.), Some(7.)))
    );
}

#[test]
fn extreme_tails_remain_ordered() {
    let laws: [Box<dyn CheckedLaw<f64>>; 4] = [
        Box::new(DoubleParetoLogNormal::new(0., 1., 1.5, 2.).unwrap()),
        Box::new(ExponentiallyModifiedNormal::new(0.01).unwrap()),
        Box::new(GeneralizedExponential::new(0.2, 5., 4.).unwrap()),
        Box::new(PowerNormal::new(0.2).unwrap()),
    ];
    for (index, d) in laws.iter().enumerate() {
        let xs: &[f64] = if index == 0 {
            &[1e-20, 1e-10, 1., 1e10, 1e20]
        } else if index == 2 {
            &[0., 1e-10, 1., 10., 100.]
        } else {
            &[-30., -10., 0., 10., 30.]
        };
        let mut previous = f64::NEG_INFINITY;
        for &x in xs {
            let lower = d.log_cdf(x).unwrap();
            let upper = d.log_sf(x).unwrap();
            assert!(lower >= previous && lower <= 0., "{index} {x}: {lower}");
            assert!(upper <= 0., "{index} {x}: {upper}");
            previous = lower;
        }
    }
}

#[test]
fn densities_normalize_over_central_probability_window() {
    for row in include_str!("../data/transformed_laws.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .step_by(2)
    {
        let cols: Vec<_> = row.split_whitespace().collect();
        let nums: Vec<f64> = cols[1..].iter().map(|x| x.parse().unwrap()).collect();
        let d = law(cols[0], nums[0], nums[1], nums[2], nums[3]);
        let low = d.quantile(0.01).unwrap();
        let high = d.quantile(0.99).unwrap();
        let n = 4096;
        let h = (high - low) / f64::from(n);
        let mut sum = d.pdf(low).unwrap() + d.pdf(high).unwrap();
        for i in 1..n {
            let x = (i as f64).mul_add(h, low);
            sum += if i % 2 == 0 { 2. } else { 4. } * d.pdf(x).unwrap();
        }
        let mass = sum * h / 3.;
        assert!((mass - 0.98).abs() < 0.003, "{row}: central mass {mass}");
    }
}

#[test]
fn double_pareto_extreme_parameters_keep_valid_tails() {
    for sigma in [0.1_f64, 1., 5.] {
        for alpha in [0.1_f64, 1., 10.] {
            for beta in [0.1_f64, 1., 10.] {
                let d = DoubleParetoLogNormal::new(-2., sigma, alpha, beta).unwrap();
                for offset in [-20.0_f64, -5., 0., 5., 20.] {
                    let x = (-2. + offset).exp();
                    let lower = d.log_cdf(x).unwrap();
                    let upper = d.log_sf(x).unwrap();
                    assert!(
                        lower <= 0. && upper <= 0.,
                        "{sigma} {alpha} {beta} {offset}"
                    );
                    assert!(
                        (lower.exp() + upper.exp() - 1.).abs() < 2e-8,
                        "{sigma} {alpha} {beta} {offset}: {lower} {upper}"
                    );
                }
            }
        }
    }
}

fn seeded_bounded_means<S: RealScalar>() {
    let cases: [(Box<dyn CheckedLaw<S>>, f64, f64); 2] = [
        (
            Box::new(GeneralizedHalfLogistic::new(S::ONE).unwrap()),
            2. * (1. - 2f64.ln()),
            0.05,
        ),
        (
            Box::new(
                TruncatedPareto::new(
                    S::try_from_f64_lossy(2.).unwrap(),
                    S::try_from_f64_lossy(4.).unwrap(),
                )
                .unwrap(),
            ),
            1.6,
            0.15,
        ),
    ];
    for (index, (law, expected, tolerance)) in cases.into_iter().enumerate() {
        let mut rng = SplitMix64::from_seed(8538 + index as u64);
        let mut sum = 0.;
        for _ in 0..8192 {
            sum += law.draw(&mut rng).unwrap().into_f64();
        }
        let mean = sum / 8192.;
        assert!(
            (mean - expected).abs() <= tolerance,
            "case {index}: {mean} != {expected}"
        );
    }
    // Hoeffding union bound across two laws and widths is below 1e-7.
}
#[test]
fn seeded_f64_bounded_means() {
    seeded_bounded_means::<f64>();
}
#[test]
fn seeded_f32_bounded_means() {
    seeded_bounded_means::<f32>();
}

#[path = "../support/statistics.rs"]
mod statistics;
