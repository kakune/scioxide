//! Extreme-shape tests against independent PMF sums and incomplete integrals.
#![allow(clippy::unwrap_used, clippy::panic, clippy::cast_precision_loss)]
use scioxide_core::{NumericScalar, NumericalIssue, RealScalar, SciOxideError, SciOxideErrorView};
use scioxide_linalg::Matrix;
use scioxide_random::*;

use super::statistics;

fn cuts<S: RealScalar>(kind: &str, name: &str) -> Vec<(f64, f64)> {
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let rows: Vec<_> = include_str!("../data/extreme_gamma_beta.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            (fields[0] == kind && fields[1] == width && fields[2] == name)
                .then(|| (fields[3].parse().unwrap(), fields[4].parse().unwrap()))
        })
        .collect();
    assert!(!rows.is_empty(), "missing fixture {kind}/{width}/{name}");
    rows
}

/// Include every attempt, with range failures in their corresponding tail.
fn log_outcome<S: RealScalar>(draw: Result<S, SciOxideError>) -> f64 {
    match draw {
        Ok(value) => {
            assert!(
                value.is_finite() && value > S::ZERO,
                "invalid draw {value:?}"
            );
            value.into_f64().ln()
        }
        Err(error) => match error.view() {
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            } => f64::NEG_INFINITY,
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            } => f64::INFINITY,
            _ => panic!("unexpected draw error: {error}"),
        },
    }
}

fn count_laws<S: RealScalar + NumericScalar<Real = S>>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    macro_rules! check {
        ($name:literal, $law:expr) => {{
            let law = $law;
            let mut rng = SplitMix64::from_seed(423);
            statistics::cdf_fit($name, &cuts::<S>("count", $name), 131_072, 3e-10, || {
                law.sample(&mut rng).unwrap() as f64
            });
        }};
    }
    check!("nb001", NegativeBinomial::new(s(0.001), s(0.5)).unwrap());
    check!("nb01", NegativeBinomial::new(s(0.01), s(0.5)).unwrap());
    check!(
        "bnb",
        BetaNegativeBinomial::new(s(0.001), s(2.0), s(2.0)).unwrap()
    );
    check!("bb", BetaBinomial::new(1000, s(0.001), s(0.001)).unwrap());

    let law = DirichletMultinomial::new(1000, &[s(0.001), s(0.001)]).unwrap();
    let mut rng = SplitMix64::from_seed(423);
    statistics::cdf_fit(
        "DirichletMultinomial",
        &cuts::<S>("count", "bb"),
        statistics::DRAWS,
        3e-10,
        || {
            let row = law.sample(&mut rng).unwrap();
            assert_eq!(row.iter().sum::<usize>(), 1000);
            row[0] as f64
        },
    );
}

fn continuous_laws<S: RealScalar + NumericScalar<Real = S>>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    macro_rules! positive {
        ($name:literal, $law:expr) => {{
            let law = $law;
            let mut rng = SplitMix64::from_seed(423);
            statistics::cdf_fit(
                $name,
                &cuts::<S>("positive", $name),
                statistics::DRAWS,
                3e-6,
                || log_outcome(law.sample(&mut rng)),
            );
        }};
    }
    positive!("gg1", GeneralizedGamma::new(s(0.001), S::ONE).unwrap());
    positive!("ggneg1", GeneralizedGamma::new(s(0.001), -S::ONE).unwrap());
    positive!("prime", BetaPrime::new(s(0.001), s(0.001)).unwrap());
    positive!("f", FisherF::new(s(0.001), s(0.001)).unwrap());

    let scale = Matrix::from_vec(1, 1, vec![S::ONE]).unwrap();
    let wishart = Wishart::new(s(0.001), &scale).unwrap();
    let mut rng = SplitMix64::from_seed(423);
    statistics::cdf_fit(
        "Wishart",
        &cuts::<S>("positive", "wishart"),
        statistics::DRAWS,
        3e-6,
        || log_outcome(wishart.sample(&mut rng).map(|row| row[0])),
    );

    for (beta, name) in [(100.0, "normal100"), (1000.0, "normal1000")] {
        let law = GeneralizedNormal::new(s(beta)).unwrap();
        let mut rng = SplitMix64::from_seed(423);
        statistics::cdf_fit(
            name,
            &cuts::<S>("normal", name),
            statistics::DRAWS,
            3e-6,
            || {
                let value = law.sample(&mut rng).unwrap();
                assert!(
                    value.is_finite() && value != S::ZERO,
                    "artificial zero/nonfinite draw"
                );
                value.into_f64()
            },
        );
    }
    // A power can recover a representable value from either Gamma tail.
    for c in [-1000.0, 1000.0] {
        let law = GeneralizedGamma::new(s(0.001), s(c)).unwrap();
        let mut rng = SplitMix64::from_seed(423);
        for _ in 0..statistics::DRAWS {
            let value = law.sample(&mut rng).unwrap();
            assert!(value.is_finite() && value > S::ZERO);
        }
    }
}

fn quantiles<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        2e-6
    } else {
        2e-13
    };
    for (beta, name) in [(100.0, "normal100"), (1000.0, "normal1000")] {
        let law = GeneralizedNormal::new(s(beta)).unwrap();
        let mut previous = f64::NEG_INFINITY;
        for (p, expected) in cuts::<S>("quantile", name) {
            let value = law.quantile(s(p)).unwrap().into_f64();
            assert!(
                (value - expected).abs() <= tolerance * expected.abs(),
                "beta={beta}, p={p}: {value} != {expected}"
            );
            assert!(value > previous);
            previous = value;
            assert_eq!(law.isf(s(p)).unwrap().into_f64(), -value);
            assert!((law.cdf(s(value)).unwrap().into_f64() - p).abs() < tolerance);
        }
        for (x, expected) in cuts::<S>("normal", name) {
            assert!((law.cdf(s(x)).unwrap().into_f64() - expected).abs() < tolerance);
        }
        assert_eq!(law.quantile(S::HALF).unwrap(), S::ZERO);
        assert_eq!(law.quantile(S::ZERO).unwrap(), S::NEG_INFINITY);
        assert_eq!(law.quantile(S::ONE).unwrap(), S::INFINITY);
    }
}

#[test]
fn extreme_counts_f64() {
    count_laws::<f64>();
}
#[test]
fn extreme_counts_f32() {
    count_laws::<f32>();
}
#[test]
fn extreme_continuous_f64() {
    continuous_laws::<f64>();
}
#[test]
fn extreme_continuous_f32() {
    continuous_laws::<f32>();
}
#[test]
fn generalized_normal_quantiles_f64() {
    quantiles::<f64>();
}
#[test]
fn generalized_normal_quantiles_f32() {
    quantiles::<f32>();
}

#[test]
fn wishart_scale_recovers_an_underflowing_unit_scale_draw() {
    let unit = Wishart::new(0.001, &Matrix::<f64>::identity(1).unwrap()).unwrap();
    let scaled = Wishart::new(0.001, &Matrix::from_vec(1, 1, vec![1e300]).unwrap()).unwrap();
    let mut unit_rng = SplitMix64::from_seed(423);
    let mut scaled_rng = SplitMix64::from_seed(423);
    let mut recovered = 0;
    for _ in 0..1024 {
        let ordinary = log_outcome(unit.sample(&mut unit_rng).map(|row| row[0]));
        let with_scale = log_outcome(scaled.sample(&mut scaled_rng).map(|row| row[0]));
        recovered += usize::from(ordinary == f64::NEG_INFINITY && with_scale.is_finite());
    }
    assert!(
        recovered > 0,
        "scale must be combined before final rounding"
    );
    assert_eq!(unit_rng.position(), scaled_rng.position());
}

#[test]
fn logarithmic_mixtures_preserve_word_budgets() {
    let options = LawOptions::<f64>::default().with_sample_word_budget(scioxide_core::Budget::ONE);
    let mut rng = SplitMix64::from_seed(423);
    let errors = [
        NegativeBinomial::with_options(0.001, 0.5, options)
            .unwrap()
            .sample(&mut rng)
            .unwrap_err(),
        BetaNegativeBinomial::with_options(0.001, 2.0, 2.0, options)
            .unwrap()
            .sample(&mut rng)
            .unwrap_err(),
        BetaBinomial::with_options(1000, 0.001, 0.001, options)
            .unwrap()
            .sample(&mut rng)
            .unwrap_err(),
        FisherF::with_options(0.001, 0.001, options)
            .unwrap()
            .sample(&mut rng)
            .unwrap_err(),
        Wishart::with_options(0.001, &Matrix::<f64>::identity(1).unwrap(), options)
            .unwrap()
            .sample(&mut rng)
            .unwrap_err(),
    ];
    assert!(
        errors
            .iter()
            .all(|error| error.kind() == scioxide_core::SciOxideErrorKind::ConvergenceFailure)
    );
}

struct FailedProvider;
fn provider_failure() -> SciOxideError {
    SciOxideError::not_implemented("provider exhausted")
}
impl BitGenerator for FailedProvider {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Err(provider_failure())
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Err(provider_failure())
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        Err(provider_failure())
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        Err(provider_failure())
    }
    fn fill_bytes(&mut self, _bytes: &mut [u8]) -> Result<(), SciOxideError> {
        Err(provider_failure())
    }
}
impl Rng<f64> for FailedProvider {
    fn next_float(&mut self) -> Result<f64, SciOxideError> {
        Err(provider_failure())
    }
}

#[test]
fn logarithmic_draws_propagate_provider_failures() {
    let mut rng = FailedProvider;
    for result in [
        GeneralizedNormal::new(1000.0).unwrap().sample(&mut rng),
        GeneralizedGamma::new(0.001, 1.0).unwrap().sample(&mut rng),
        BetaPrime::new(0.001, 0.001).unwrap().sample(&mut rng),
        FisherF::new(0.001, 0.001).unwrap().sample(&mut rng),
    ] {
        assert_eq!(
            result.unwrap_err().to_string(),
            provider_failure().to_string()
        );
    }
}
