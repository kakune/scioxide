//! Independent inverse and tail references for numerically inverted laws.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_random::*;

trait InvertibleLaw: ContinuousLaw<f64> + Quantile<f64> + InverseSurvival<f64> {}
impl<D: ContinuousLaw<f64> + Quantile<f64> + InverseSurvival<f64>> InvertibleLaw for D {}

fn relative(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs(),
        "{actual:e} != {expected:e}"
    );
}

fn tail_tolerance(law: &dyn InvertibleLaw, x: f64, q: f64) -> f64 {
    // A coordinate ULP is amplified by density / tail near finite bounds.
    let quantum = (x.next_up() - x).max(x - x.next_down());
    8e-8 + 4.0 * quantum * (law.log_pdf(x).unwrap() - q.ln()).exp()
}

#[test]
fn student_t_median_references() {
    for row in include_str!("../data/student_t_quantiles.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let values: Vec<f64> = row.split('\t').map(|s| s.parse().unwrap()).collect();
        let law = StudentT::new(values[0]).unwrap();
        for (actual, expected) in [
            (law.quantile(values[1]).unwrap(), values[2]),
            (law.isf(values[1]).unwrap(), values[3]),
        ] {
            let ulps = actual.to_bits().abs_diff(expected.to_bits());
            assert!(
                ulps <= 4,
                "nu={} p={} actual={actual:e} expected={expected:e} ulps={ulps}",
                values[0],
                values[1]
            );
        }
    }
}

#[test]
fn student_t_boundaries_and_precision() {
    for nu in [0.5_f64, 1.0, 5.0, 1e6] {
        let law = StudentT::new(nu).unwrap();
        assert_eq!(law.quantile(0.5).unwrap(), 0.0);
        assert_eq!(law.isf(0.5).unwrap(), 0.0);
        assert_eq!(law.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(law.quantile(1.0).unwrap(), f64::INFINITY);
        assert_eq!(law.isf(0.0).unwrap(), f64::INFINITY);
        assert_eq!(law.isf(1.0).unwrap(), f64::NEG_INFINITY);
        for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(law.quantile(p).is_err());
            assert!(law.isf(p).is_err());
        }
        let narrow = StudentT::new(nu.min(1e30) as f32).unwrap();
        let mut previous = f64::NEG_INFINITY;
        for p in [
            1e-12,
            0.1,
            0.249999999,
            0.25,
            0.250000001,
            0.4999,
            0.5,
            0.5001,
            0.749999999,
            0.75,
            0.750000001,
            0.9,
            1.0 - 1e-12,
        ] {
            let x = law.quantile(p).unwrap();
            assert!(x > previous, "nu={nu}, p={p}");
            previous = x;
            assert_eq!(x, -law.isf(p).unwrap());
            let p32 = p as f32;
            if p32 > 0.0 && p32 < 1.0 && p32 != 0.5 {
                let reference = law.quantile(f64::from(p32)).unwrap() as f32;
                let actual = narrow.quantile(p32).unwrap();
                assert!(actual.to_bits().abs_diff(reference.to_bits()) <= 4);
            }
        }
    }
}

fn noncentral(name: &str) -> Box<dyn InvertibleLaw> {
    match name {
        "ncx2" => Box::new(
            NoncentralChiSquared::with_options(
                4.0,
                2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "ncf" => Box::new(
            NoncentralF::with_options(
                5.0,
                10.0,
                2.0,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "rice" => Box::new(
            Rice::with_options(
                1.5,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        _ => panic!("unknown reference law"),
    }
}

#[test]
fn noncentral_upper_references() {
    for row in include_str!("../data/upper_quantiles.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let value = |i: usize| fields[i].parse::<f64>().unwrap();
        let law = noncentral(fields[0]);
        let q = value(1);
        let x = law.isf(q).unwrap();
        assert!(
            x >= value(5) && x <= value(6),
            "{} q={q:e} x={x} reference={}",
            fields[0],
            value(2)
        );
        if 1.0 - q < 1.0 {
            relative(law.quantile(1.0 - q).unwrap(), x, 4e-14);
        } else {
            assert_eq!(law.quantile(1.0 - q).unwrap(), f64::INFINITY);
            let report = invert_log_probability_with_options(
                &*law,
                q.ln(),
                true,
                &InversionOptions::default()
                    .with_root_options(InversionOptions::default().root_options().clone())
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(1_048_575)),
            )
            .unwrap();
            assert!(report.root().is_success());
            relative(report.value(), x, 4e-8);
        }
        // The reference logs are evaluated at the rounded fixture coordinate.
        let reference_x = value(2);
        relative(law.log_cdf(reference_x).unwrap(), value(3), 2e-8);
        relative(law.log_sf(reference_x).unwrap(), value(4), 2e-8);
    }
}

fn generic_laws() -> Vec<(&'static str, Box<dyn InvertibleLaw>)> {
    vec![
        (
            "f",
            Box::new(
                FisherF::with_options(
                    5.0,
                    10.0,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        ("ncx2", noncentral("ncx2")),
        ("ncf", noncentral("ncf")),
        ("rice", noncentral("rice")),
        (
            "nct",
            Box::new(
                NoncentralStudentT::with_options(
                    5.0,
                    1.0,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        (
            "range",
            Box::new(
                StudentizedRange::with_options(
                    3.0,
                    10.0,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        (
            "ksone",
            Box::new(
                KolmogorovSmirnovOneSided::with_options(
                    5,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        (
            "kstwo",
            Box::new(
                KolmogorovSmirnovTwoSided::with_options(
                    5,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        (
            "kolmogorov",
            Box::new(
                Kolmogorov::with_options(
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        ("cosine", Box::new(Cosine::new().unwrap())),
        ("beta", Box::new(Beta::new(2.0, 3.0).unwrap())),
        ("alpha", Box::new(Alpha::new(2.0).unwrap())),
        ("invgauss", Box::new(InverseGaussian::new(1.5).unwrap())),
        (
            "exponnorm",
            Box::new(ExponentiallyModifiedNormal::new(1.5).unwrap()),
        ),
        (
            "exponweib",
            Box::new(ExponentiatedWeibull::new(2.0, 1.5).unwrap()),
        ),
        (
            "genexpon",
            Box::new(GeneralizedExponential::new(1.0, 2.0, 1.5).unwrap()),
        ),
        (
            "genhalflogistic",
            Box::new(GeneralizedHalfLogistic::new(0.5).unwrap()),
        ),
        ("powernorm", Box::new(PowerNormal::new(2.0).unwrap())),
        (
            "dpln",
            Box::new(DoubleParetoLogNormal::new(0.0, 1.0, 3.0, 2.0).unwrap()),
        ),
        (
            "breitwigner",
            Box::new(RelativisticBreitWigner::new(2.0).unwrap()),
        ),
        ("irwinhall", Box::new(IrwinHall::new(5).unwrap())),
        ("stable", Box::new(LevyStable::new(1.0, 0.0).unwrap())),
        (
            "gausshyper",
            Box::new(GaussHypergeometric::new(2.0, 3.0, 1.0, 0.5).unwrap()),
        ),
        ("skewnormal", Box::new(SkewNormal::new(2.0).unwrap())),
        (
            "gig",
            Box::new(GeneralizedInverseGaussian::new(0.5, 1.0).unwrap()),
        ),
        (
            "gh",
            Box::new(
                GeneralizedHyperbolic::with_options(
                    1.0,
                    2.0,
                    0.5,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        (
            "nig",
            Box::new(
                NormalInverseGaussian::with_options(
                    2.0,
                    0.5,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
            ),
        ),
        ("vonmises", Box::new(VonMisesLine::new(0.0, 2.0).unwrap())),
        (
            "vonmises_circle",
            Box::new(VonMises::new(0.0, 2.0).unwrap()),
        ),
        (
            "mixture",
            Box::new(
                Mixture::new(
                    vec![
                        Normal::new(-1.0, 1.0).unwrap(),
                        Normal::new(2.0, 1.5).unwrap(),
                    ],
                    &[0.3, 0.7],
                )
                .unwrap(),
            ),
        ),
        (
            "truncated",
            Box::new(Truncated::new(Normal::new(0.0, 1.0).unwrap(), -2.0, 3.0).unwrap()),
        ),
        (
            "order",
            Box::new(OrderStatistic::new(Normal::new(0.0, 1.0).unwrap(), 3, 8).unwrap()),
        ),
        (
            "absolute",
            Box::new(Transformed::abs(Normal::new(1.0, 1.5).unwrap()).unwrap()),
        ),
        (
            "exponential",
            Box::new(Transformed::exp(Normal::new(0.0, 1.0).unwrap()).unwrap()),
        ),
        (
            "logarithmic",
            Box::new(Transformed::log(Gamma::new(2.0, 1.0).unwrap()).unwrap()),
        ),
    ]
}

#[test]
fn generic_inverses_select_both_small_tails() {
    for (name, law) in generic_laws() {
        for q in [1e-3, 1e-10] {
            let p = 1.0 - q;
            let q = 1.0 - p; // identical represented tail for both entry points
            let upper = law
                .quantile(p)
                .unwrap_or_else(|e| panic!("{name} upper {q}: {e}"));
            let isf = law.isf(q).unwrap_or_else(|e| panic!("{name} isf {q}: {e}"));
            assert!(
                (upper - isf).abs() <= 4e-13 * isf.abs().max(1.0),
                "{name}: {upper} != {isf}"
            );
            let lower = law
                .quantile(q)
                .unwrap_or_else(|e| panic!("{name} lower {q}: {e}"));
            let reflected = law
                .isf(p)
                .unwrap_or_else(|e| panic!("{name} reflected {q}: {e}"));
            assert!(
                (lower - reflected).abs() <= 4e-13 * lower.abs().max(1.0),
                "{name}: {lower} != {reflected}"
            );
            // Compare probabilities relatively, including near-zero large-tail logs.
            let log_upper = law.log_sf(upper).unwrap();
            let upper_tolerance = tail_tolerance(&*law, upper, q);
            let lower_tolerance = tail_tolerance(&*law, lower, q);
            assert!(
                (log_upper - q.ln()).abs() <= upper_tolerance,
                "{name} upper log tail: {log_upper} != {}",
                q.ln()
            );
            let log_lower = law.log_cdf(lower).unwrap();
            assert!(
                (log_lower - q.ln()).abs() <= lower_tolerance,
                "{name} lower log tail: {log_lower} != {}",
                q.ln()
            );
            relative(law.log_cdf(upper).unwrap(), (-q).ln_1p(), upper_tolerance);
            relative(law.log_sf(lower).unwrap(), (-q).ln_1p(), lower_tolerance);
        }
    }
}

#[test]
fn generic_extreme_upper_tails() {
    for (name, law) in generic_laws() {
        let (_, bound) = law.support().unwrap().bounds().unwrap();
        if bound.is_some() {
            assert!(
                law.isf(1e-300).is_err(),
                "{name}: unresolved endpoint must be an error"
            );
            continue;
        }
        let x = law.isf(1e-300).unwrap_or_else(|e| panic!("{name}: {e}"));
        let tail = law.log_sf(x).unwrap();
        assert!((tail - 1e-300_f64.ln()).abs() <= 8e-8, "{name}: {tail}");
        relative(law.log_cdf(x).unwrap(), -1e-300, 8e-8);
    }
}

#[test]
fn generic_extreme_lower_tails() {
    for (name, law) in generic_laws() {
        let (bound, _) = law.support().unwrap().bounds().unwrap();
        let result = law.quantile(1e-300);
        if bound.is_some_and(|x| x != 0.0) {
            assert!(
                result.is_err(),
                "{name}: unresolved lower endpoint must be an error"
            );
            continue;
        }
        let x = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let tail = law.log_cdf(x).unwrap();
        assert!((tail - 1e-300_f64.ln()).abs() <= 8e-8, "{name}: {tail}");
        relative(
            law.log_sf(x).unwrap_or_else(|e| panic!("{name}: {e}")),
            -1e-300,
            8e-8,
        );
    }
}

#[test]
fn independent_generic_tail_references() {
    let laws = generic_laws();
    for row in include_str!("../data/generic_quantiles.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let number = |i: usize| fields[i].parse::<f64>().unwrap();
        let name = fields[0];
        let law = &laws.iter().find(|(n, _)| *n == name).unwrap().1;
        let q = number(1);
        let x = number(2);
        let upper = fields[3] == "1";
        let expected_log = number(4);
        let condition = number(5);
        // This independent forward oracle tests the true probability at an
        // inverse candidate, rather than trusting a SciOxide round trip.
        let quantum = (x.next_up() - x).max(x - x.next_down());
        let tolerance = 8e-8 + 4.0 * quantum / condition;
        let oracle_error = (expected_log - q.ln()).exp_m1().abs();
        assert!(
            oracle_error <= tolerance,
            "{name} q={q:e} upper={upper}: mpmath relative tail error {oracle_error:e}"
        );
        let actual_log = if upper { law.log_sf(x) } else { law.log_cdf(x) }.unwrap();
        assert!(
            (actual_log - expected_log).abs() <= tolerance,
            "{name} q={q:e} upper={upper}: log tail {actual_log} != {expected_log}"
        );
        let inverse = if upper { law.isf(q) } else { law.quantile(q) }.unwrap();
        assert!(
            (inverse - x).abs() <= 2e-8 * condition + 4.0 * quantum,
            "{name} q={q:e} upper={upper}: inverse {inverse} != candidate {x}"
        );
    }
}

#[test]
fn complementary_tails_share_one_evaluation() {
    // Each request evaluates the tail on its side of the approximate median
    // once and forms the other request from it.
    let law = NoncentralChiSquared::with_options(
        4.0_f64,
        2.0,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for x in [0.5, 66.697] {
        let upper = law.evaluate_tail(x, true).unwrap();
        let lower = law.evaluate_tail(x, false).unwrap();
        assert_eq!(upper.evaluations, lower.evaluations);
        relative(
            lower.log_probability.exp() + upper.log_probability.exp(),
            1.0,
            4e-14,
        );
        let (small, large) = if x > 1.0 {
            (upper, lower)
        } else {
            (lower, upper)
        };
        relative(
            large.log_probability.exp_m1(),
            -small.log_probability.exp(),
            4e-14,
        );
        // Nested kernels may select a cheaper convergent path when headroom
        // is smaller. Every successful path must still report work within it.
        for work in [small.evaluations, small.evaluations - 1] {
            let options = LawOptions::default()
                .with_work_budget(scioxide_core::Budget::try_from(work).unwrap());
            let bounded = NoncentralChiSquared::with_options(4.0, 2.0, options).unwrap();
            let upper = bounded.evaluate_tail(x, true);
            let lower = bounded.evaluate_tail(x, false);
            assert_eq!(upper.is_ok(), lower.is_ok());
            if work == small.evaluations {
                assert!(upper.is_ok());
            }
            for report in [upper, lower].into_iter().flatten() {
                assert!(report.evaluations <= work);
            }
        }
        let bounded = NoncentralChiSquared::with_options(
            4.0,
            2.0,
            LawOptions::default().with_work_budget(scioxide_core::Budget::ONE),
        )
        .unwrap();
        assert!(bounded.evaluate_tail(x, true).is_err());
    }
}

#[test]
fn student_t_underflowed_beta_coordinate() {
    let cauchy = Cauchy::new(0.0_f64, 1.0).unwrap();
    let t = StudentT::new(1.0_f64).unwrap();
    for p in [1e-100, 1e-200, 1e-300] {
        relative(t.quantile(p).unwrap(), cauchy.quantile(p).unwrap(), 2e-13);
        relative(t.isf(p).unwrap(), cauchy.isf(p).unwrap(), 2e-13);
    }
    assert!(StudentT::new(0.5_f64).unwrap().quantile(1e-300).is_err());
}

#[test]
fn student_t_large_degree_limit() {
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    for nu in [2048.0 / f64::EPSILON, 1e100, 1e308] {
        let law = StudentT::new(nu).unwrap();
        for p in [
            f64::from_bits(1),
            1e-300,
            0.5 - 1e-16,
            0.5,
            0.5 + 1e-16,
            1.0_f64.next_down(),
        ] {
            assert_eq!(law.quantile(p).unwrap(), normal.quantile(p).unwrap());
            assert_eq!(law.isf(p).unwrap(), normal.isf(p).unwrap());
        }
    }
}

#[test]
fn student_t_tail_references() {
    // Tails are conditioned by their logarithm: forming a beta coordinate or a
    // normal deviate from an exact x rounds once, amplified by |ln tail|.
    let tail_tolerance = |log_tail: f64| 8.0 * f64::EPSILON * (1.0 - log_tail);
    for row in include_str!("../data/student_t_tails.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let fields: Vec<&str> = row.split('\t').collect();
        let nu: f64 = fields[1].parse().unwrap();
        let law = StudentT::new(nu).unwrap();
        if fields[0] == "tail" {
            let x: f64 = fields[2].parse().unwrap();
            let log_cdf: f64 = fields[3].parse().unwrap();
            let log_sf: f64 = fields[4].parse().unwrap();
            let tolerance = tail_tolerance(log_sf);
            for (actual, expected) in [
                (law.log_sf(x).unwrap(), log_sf),
                (law.log_cdf(-x).unwrap(), log_sf),
            ] {
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "nu={nu} x={x}: {actual:e} != {expected:e}"
                );
            }
            // The large tail's logarithm is the small tail to first order.
            for actual in [law.log_cdf(x).unwrap(), law.log_sf(-x).unwrap()] {
                relative(actual, log_cdf, tolerance);
            }
            if log_sf > f64::MIN_POSITIVE.ln() {
                relative(law.sf(x).unwrap(), log_sf.exp(), tolerance);
                relative(law.cdf(-x).unwrap(), log_sf.exp(), tolerance);
                relative(law.cdf(x).unwrap(), log_cdf.exp(), tolerance);
            }
        } else {
            let p: f64 = fields[2].parse().unwrap();
            if fields[3] == "overflow" {
                assert!(law.quantile(p).is_err(), "nu={nu} p={p}");
                assert!(law.isf(p).is_err(), "nu={nu} p={p}");
                continue;
            }
            let expected: f64 = fields[3].parse().unwrap();
            let tolerance = tail_tolerance(p.ln());
            relative(law.quantile(p).unwrap(), expected, tolerance);
            relative(law.isf(p).unwrap(), -expected, tolerance);
            if 1.0 - p < 1.0 {
                let mirrored = 1.0 - (1.0 - p);
                assert_eq!(
                    law.quantile(1.0 - p).unwrap(),
                    -law.quantile(mirrored).unwrap()
                );
            }
        }
    }
}
