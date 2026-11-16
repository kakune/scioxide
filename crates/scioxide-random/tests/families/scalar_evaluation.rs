//! Prepared normalization and direct probability paths.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, RealScalar, SciOxideErrorKind};
use scioxide_random::*;

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

fn relative(actual: f64, expected: f64, tolerance: f64) {
    relative_width::<f64>(actual, expected, tolerance);
}

fn relative_width<S: RealScalar>(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs()
            <= tolerance * expected.abs() + 8.0 * (S::MIN_POSITIVE * S::EPSILON).into_f64(),
        "actual={actual:e}, expected={expected:e}"
    );
}

fn prepared_densities<S: RealScalar>() {
    let tolerance = 64.0 * S::EPSILON.into_f64();
    let beta = Beta::new(scalar::<S>(2.0), scalar(3.0)).unwrap();
    let gamma = Gamma::new(scalar::<S>(2.0), scalar(0.5)).unwrap();
    let student = StudentT::new(S::ONE).unwrap();
    let fisher = FisherF::new(scalar::<S>(2.0), scalar(2.0)).unwrap();
    for value in [0.01, 0.125, 0.5, 0.75, 0.99] {
        let x = scalar::<S>(value);
        let value = x.into_f64();
        for (density, log, expected) in [
            (
                beta.pdf(x),
                beta.log_pdf(x),
                12.0 * value * (1.0 - value).powi(2),
            ),
            (
                gamma.pdf(x),
                gamma.log_pdf(x),
                4.0 * value * (-2.0 * value).exp(),
            ),
            (
                student.pdf(x),
                student.log_pdf(x),
                1.0 / (core::f64::consts::PI * (1.0 + value * value)),
            ),
            (
                fisher.pdf(x),
                fisher.log_pdf(x),
                1.0 / (1.0 + value).powi(2),
            ),
        ] {
            relative(density.unwrap().into_f64(), expected, tolerance);
            relative(log.unwrap().into_f64().exp(), expected, tolerance);
        }
    }
    assert_eq!(beta.pdf(S::ZERO).unwrap(), S::ZERO);
    assert_eq!(gamma.pdf(S::ZERO).unwrap(), S::ZERO);
    assert_eq!(fisher.pdf(S::ZERO).unwrap(), S::ONE);
    for x in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
        for result in [beta.pdf(x), gamma.pdf(x), student.pdf(x), fisher.pdf(x)] {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
        }
    }
}

#[test]
fn prepared_densities_both_widths() {
    prepared_densities::<f64>();
    prepared_densities::<f32>();
}

#[test]
fn alpha_closed_inverses_resolve_both_tails_and_large_shapes() {
    for shape in [0.1, 2.0, 30.0, 100.0] {
        let law = Alpha::new(shape).unwrap();
        for p in [1e-300_f64, 1e-20, 0.01, 0.25, 0.5, 0.9] {
            for (x, upper) in [
                (law.quantile(p).unwrap(), false),
                (law.isf(p).unwrap(), true),
            ] {
                let actual = if upper { law.log_sf(x) } else { law.log_cdf(x) }.unwrap();
                assert!(
                    (actual - p.ln()).abs() <= 8e-8 * (1.0 + p.ln().abs()),
                    "shape={shape}, p={p}, upper={upper}, x={x}, log={actual}"
                );
            }
        }
    }
}

#[test]
fn preparation_preserves_independent_extreme_operations() {
    let beta = Beta::new(f64::MAX, f64::MAX).unwrap();
    assert_eq!(beta.mean().unwrap(), 0.5);
    assert!(beta.variance().unwrap() > 0.0);
    assert_eq!(beta.pdf(0.0).unwrap(), 0.0);
    relative(
        beta.pdf(0.5).unwrap(),
        2.0 * (f64::MAX / core::f64::consts::PI).sqrt(),
        1e-12,
    );
    let gamma = Gamma::new(f64::MAX, 0.5).unwrap();
    assert_eq!(gamma.mean().unwrap(), f64::MAX * 0.5);
    assert_eq!(gamma.pdf(0.0).unwrap(), 0.0);
    assert!(gamma.pdf(1.0).is_err());
    let student = StudentT::new(1e14_f64).unwrap();
    relative(
        student.pdf(0.0).unwrap(),
        1.0 / (2.0 * core::f64::consts::PI).sqrt(),
        1e-13,
    );
}

#[test]
fn centered_beta_densities_match_independent_log_gamma_references() {
    for row in include_str!("../data/beta_density.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let columns: Vec<_> = row.split_whitespace().collect();
        let numbers: Vec<f64> = columns[1..]
            .iter()
            .map(|value| value.parse().unwrap())
            .collect();
        let actual = if columns[0] == "32" {
            f64::from(
                Beta::new(numbers[0] as f32, numbers[1] as f32)
                    .unwrap()
                    .log_pdf(numbers[2] as f32)
                    .unwrap(),
            )
        } else {
            Beta::new(numbers[0], numbers[1])
                .unwrap()
                .log_pdf(numbers[2])
                .unwrap()
        };
        let tolerance = if columns[0] == "32" { 2e-6 } else { 2e-12 };
        assert!(
            (actual - numbers[3]).abs() <= tolerance * numbers[3].abs().max(1.0),
            "{row}: {actual:e}"
        );
    }
}

fn negative_binomial_recurrence<S: RealScalar>() {
    let tolerance = (64.0 * S::EPSILON.into_f64()).max(3e-13);
    for r in [0.125, 1.0, 2.5, 15.5, 16.0, 16.5] {
        for p in [0.05, 0.3, 0.5, 0.9] {
            let p = scalar::<S>(p);
            let law = NegativeBinomial::new(scalar::<S>(r), p).unwrap();
            let p = p.into_f64();
            let mut mass = p.powf(r);
            let mut cumulative = 0.0;
            let mut previous = 0.0;
            for k in 0..=64 {
                let actual = law.pmf(k).unwrap().into_f64();
                relative_width::<S>(actual, mass, tolerance);
                cumulative += mass;
                let cdf = law.cdf(k).unwrap().into_f64();
                let sf = law.sf(k).unwrap().into_f64();
                assert!(cdf >= previous);
                relative(cdf, cumulative, tolerance);
                assert!((cdf - previous - mass).abs() <= tolerance);
                assert!((cdf + sf - 1.0).abs() <= tolerance);
                relative_width::<S>(sf, law.log_sf(k).unwrap().into_f64().exp(), tolerance * 8.0);
                previous = cdf;
                mass *= (r + k as f64) / (k + 1) as f64 * (1.0 - p);
            }
        }
    }
}

#[test]
fn negative_binomial_direct_tails_and_mass_recurrence() {
    negative_binomial_recurrence::<f64>();
    negative_binomial_recurrence::<f32>();
}

#[test]
fn negative_binomial_extremes_and_work_limits() {
    for (r, p, k) in [
        (2.5, 1e-20, 16),
        (1e-20, 0.5, 16),
        (1e20, 1.0 - f64::EPSILON, 16),
        (1e6, 0.5, 1_000_000),
    ] {
        let law = NegativeBinomial::new(r, p).unwrap();
        let count = k as f64;
        let reference = -scioxide_special::ln_beta(r, count + 1.0).unwrap() - (r + count).ln()
            + r * p.ln()
            + count * (-p).ln_1p();
        assert!((law.log_pmf(k).unwrap() - reference).abs() <= 3e-9 * reference.abs().max(1.0));
    }
    let geometric = NegativeBinomial::new(1.0_f64, 0.5).unwrap();
    assert_eq!(geometric.sf(2048).unwrap(), 0.0);
    relative(
        geometric.log_sf(2048).unwrap(),
        -2049.0 * core::f64::consts::LN_2,
        2e-14,
    );
    for p in [f64::from_bits(1), 1e-20, 1.0 - f64::EPSILON] {
        let law = NegativeBinomial::new(1.0_f64, p).unwrap();
        relative(law.cdf(0).unwrap(), p, 2e-14);
        relative(law.sf(0).unwrap(), 1.0 - p, 2e-14);
    }
    let options = LawOptions::default().with_work_budget(Budget::ONE);
    let bounded = NegativeBinomial::with_options(2.5_f64, 0.3, options).unwrap();
    assert_eq!(
        bounded.cdf(3).unwrap_err().kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert!(bounded.cdf(usize::MAX).is_err());
    let degenerate = NegativeBinomial::with_options(2.5_f64, 1.0, options).unwrap();
    assert_eq!(degenerate.cdf(usize::MAX).unwrap(), 1.0);
    assert_eq!(degenerate.sf(usize::MAX).unwrap(), 0.0);
}

#[test]
fn fisher_direct_and_logarithmic_paths_agree_at_boundaries() {
    let law = FisherF::new(2.0_f64, 2.0).unwrap();
    for x in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-200,
        0.5,
        1.0 - f64::EPSILON,
        1.0,
        1.0 + f64::EPSILON,
        2.0,
        1e200,
        f64::MAX,
    ] {
        relative(law.cdf(x).unwrap(), x / (1.0 + x), 3e-13);
        relative(law.sf(x).unwrap(), 1.0 / (1.0 + x), 3e-13);
    }
    for (a, b) in [(0.125, 30.0), (30.0, 0.125), (5.0, 10.0), (1000.0, 2000.0)] {
        let law = FisherF::new(a, b).unwrap();
        for x in [0.01, 0.5, 1.0, b / a, 2.0, 100.0] {
            relative(law.cdf(x).unwrap(), law.log_cdf(x).unwrap().exp(), 3e-12);
            relative(law.sf(x).unwrap(), law.log_sf(x).unwrap().exp(), 3e-12);
        }
    }
    let narrow = FisherF::new(2.0_f32, 2.0).unwrap();
    for x in [f32::MIN_POSITIVE, 0.5, 1.0, 2.0, 1e10] {
        relative(
            narrow.cdf(x).unwrap().into(),
            (x as f64) / (1.0 + x as f64),
            2e-6,
        );
    }
}

#[test]
fn elementary_direct_probabilities_retain_small_tails() {
    let logistic = Logistic::new(0.0_f64, 1.0).unwrap();
    for x in [-740.0, -36.0, -1.0, 0.0, 1.0, 36.0, 740.0] {
        relative(
            logistic.cdf(x).unwrap(),
            logistic.log_cdf(x).unwrap().exp(),
            2e-14,
        );
        relative(
            logistic.sf(x).unwrap(),
            logistic.log_sf(x).unwrap().exp(),
            2e-14,
        );
    }
    let narrow = Logistic::new(0.0_f64, f64::MIN_POSITIVE).unwrap();
    assert_eq!(narrow.cdf(f64::MAX).unwrap(), 1.0);
    assert_eq!(narrow.sf(f64::MAX).unwrap(), 0.0);
    assert_eq!(narrow.cdf(-f64::MAX).unwrap(), 0.0);
    let pareto = Pareto::new(1.0_f64).unwrap();
    assert_eq!(pareto.cdf(1.0).unwrap().to_bits(), 0.0_f64.to_bits());
    let x = 1.0 + f64::EPSILON;
    relative(pareto.cdf(x).unwrap(), (x - 1.0) / x, 2e-14);
    let weibull = WeibullMin::new(1.0_f64).unwrap();
    assert_eq!(weibull.cdf(0.0).unwrap().to_bits(), 0.0_f64.to_bits());
    for x in [f64::from_bits(1), 1e-30, 1e-10, 1.0, 1000.0] {
        relative(weibull.cdf(x).unwrap(), -(-x).exp_m1(), 2e-14);
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for result in [logistic.cdf(x), pareto.cdf(x), weibull.cdf(x)] {
            assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
        }
    }
}
