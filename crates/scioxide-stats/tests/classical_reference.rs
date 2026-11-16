//! Fraction/90-digit references and supplemental SciPy MGC comparisons.
//! Regenerated MGC low-bit differences are covered by the existing numerical
//! envelopes; generators are never needed at test time.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_linalg::Matrix;
use scioxide_stats::*;
fn fixture(name: &str) -> (f64, Option<f64>) {
    let line = include_str!("data/classical.tsv")
        .lines()
        .find(|l| l.starts_with(&format!("{name}\t")))
        .unwrap();
    let mut fields = line.split('\t').skip(1);
    let stat = fields.next().unwrap().parse().unwrap();
    let p = fields.next().unwrap();
    (
        stat,
        if p == "none" {
            None
        } else {
            Some(p.parse().unwrap())
        },
    )
}
fn check<S: RealScalar>(name: &str, result: ClassicalTestResult<S>, tol: f64) {
    let (stat, p) = fixture(name);
    assert!(
        (result.statistic().into_f64() - stat).abs() <= tol * (1. + stat.abs()),
        "{name}: {} != {stat}",
        result.statistic().into_f64()
    );
    if let Some(p) = p {
        assert!(
            (result.p_value() - p).abs() <= tol,
            "{name}: {:?} != {p}",
            result.p_value()
        );
    }
}
fn cases<S: RealScalar>(tol: f64) {
    let convert = |v| S::try_from_f64_lossy(v).unwrap();
    let x = [1., 1., 2., 4., 7.].map(convert);
    let y = [0., 2., 2., 5., 3.].map(convert);
    let defaults = ClassicalOptions::default();
    let asym = defaults.with_calibration(TestCalibration::Asymptotic);
    let upper = defaults.with_alternative(TestAlternative::Greater);
    check(
        "one",
        t_test_one_sample_with_options(&x, S::ZERO, defaults).unwrap(),
        tol,
    );
    check(
        "paired",
        t_test_paired_with_options(&x, &y, defaults).unwrap(),
        tol,
    );
    check(
        "welch",
        t_test_independent_with_options(&x, &y, TTestVariance::Welch, defaults).unwrap(),
        tol,
    );
    check(
        "pooled",
        t_test_independent_with_options(&x, &y, TTestVariance::Pooled, defaults).unwrap(),
        tol,
    );
    check(
        "pearson",
        pearson_test_with_options(&x, &y, defaults).unwrap(),
        tol,
    );
    check(
        "spearman",
        spearman_test_with_options(&x, &y, defaults).unwrap(),
        tol,
    );
    check(
        "kendall_b",
        kendall_test_with_options(&x, &y, KendallVariant::TauB, defaults).unwrap(),
        tol,
    );
    check(
        "kendall_c",
        kendall_test_with_options(&x, &y, KendallVariant::TauC, defaults).unwrap(),
        tol,
    );
    check(
        "mann_whitney",
        mann_whitney_test_with_options(&x, &y, defaults).unwrap(),
        tol,
    );
    let d = [0., 1., 1., -2., 3.].map(convert);
    for (name, zeros) in [
        ("discard", SignedRankZeros::Discard),
        ("pratt", SignedRankZeros::Pratt),
        ("split", SignedRankZeros::Split),
    ] {
        check(
            &format!("wilcoxon_{name}"),
            wilcoxon_test_with_options(&d, zeros, defaults).unwrap(),
            tol,
        );
    }
    check(
        "brunner_student",
        brunner_munzel_test_with_options(&x, &y, BrunnerMunzelCalibration::Student, defaults)
            .unwrap(),
        tol,
    );
    check(
        "brunner_normal",
        brunner_munzel_test_with_options(&x, &y, BrunnerMunzelCalibration::Normal, defaults)
            .unwrap(),
        tol,
    );
    check(
        "xi_exact",
        chatterjee_test_with_options(&x, &y, upper).unwrap(),
        tol,
    );
    check(
        "xi_normal",
        chatterjee_test_with_options(&x, &y, asym.with_alternative(TestAlternative::Greater))
            .unwrap(),
        tol,
    );
    check(
        "somers",
        somers_test_with_options(&x, &y, asym).unwrap(),
        tol,
    );
    let weights = [1., 0.5, 1. / 3., 0.25, 0.2];
    check_coefficient(
        "weighted_add",
        weighted_kendall_tau(&x, &y, Some(&weights), KendallWeightCombination::Additive).unwrap(),
        tol,
    );
    check_coefficient(
        "weighted_multiply",
        weighted_kendall_tau(
            &x,
            &y,
            Some(&weights),
            KendallWeightCombination::Multiplicative,
        )
        .unwrap(),
        tol,
    );
    check(
        "bws_two",
        bws_test_with_options(&x, &y, defaults).unwrap(),
        tol,
    );
    check(
        "bws_greater",
        bws_test_with_options(&x, &y, upper).unwrap(),
        tol,
    );
    for (name, counts, exposures, difference) in [
        ("poisson_zero", [0, 3], [100., 100.], 0.),
        ("poisson_nonzero", [3, 8], [10., 20.], 0.1),
        ("poisson_tiled", [90, 110], [10., 10.], 0.),
        ("poisson_tie", [3, 3], [10., 10.], 0.),
    ] {
        for (suffix, alternative) in [
            ("", TestAlternative::TwoSided),
            ("_less", TestAlternative::Less),
            ("_greater", TestAlternative::Greater),
        ] {
            check(
                &format!("{name}{suffix}"),
                poisson_means_test_with_options(
                    counts,
                    exposures.map(convert),
                    convert(difference),
                    defaults.with_alternative(alternative),
                )
                .unwrap(),
                if S::EPSILON.into_f64() < 1e-10 {
                    4e-13
                } else {
                    tol
                },
            );
        }
    }
    for n in [7, 16] {
        let x = (0..n)
            .map(|i| convert(i as f64 - (n as f64 / 2.).ceil()))
            .collect();
        let x = Matrix::from_vec(n, 1, x).unwrap();
        let y = Matrix::from_vec(n, 1, x.as_slice().iter().map(|&v| v * v).collect()).unwrap();
        let plan = MgcPlan::from_samples(
            x.view(),
            y.view(),
            scioxide_stats::ClassicalOptions::default(),
        )
        .unwrap();
        let mut scratch = plan.workspace().unwrap();
        let result = plan.statistic(None, &mut scratch).unwrap();
        let expected = fixture(&format!("mgc_{n}")).0;
        assert!(
            (result.correlation - expected).abs() < tol,
            "MGC {n}: {} != {expected}",
            result.correlation
        );
        let (a, b) = fixture(&format!("mgc_scale_{n}"));
        assert_eq!(result.scale, [a as usize, b.unwrap() as usize]);
        let [nx, ny] = plan.map_shape();
        for i in 0..nx {
            for j in 0..ny {
                let e = fixture(&format!("mgc_map_{n}_{i}_{j}")).0;
                assert!((scratch[i * ny + j] - e).abs() < tol, "MGC map {n}/{i}/{j}");
            }
        }
    }
}
#[test]
fn f64_independent_evidence() {
    cases::<f64>(2e-11);
}
#[test]
fn f32_independent_evidence() {
    cases::<f32>(2e-5);
}

fn check_coefficient<S: RealScalar>(name: &str, actual: S, tolerance: f64) {
    let expected = fixture(name).0;
    assert!((actual.into_f64() - expected).abs() <= tolerance * (1.0 + expected.abs()));
}
