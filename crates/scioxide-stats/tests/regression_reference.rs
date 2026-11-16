//! Defining-integral references plus supplemental Alexander–Govern/robust-slope
//! comparisons. Refined quadrature changes endpoints by less than 3e-13, inside
//! the existing reference envelopes; no tolerance adjustment is required.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Budget, RealScalar};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;
fn expected(family: &str, index: usize, metric: &str) -> f64 {
    include_str!("data/regression.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|line| {
            let v: Vec<_> = line.split('\t').collect();
            (v[0] == family && v[1].parse::<usize>().unwrap() == index && v[2] == metric)
                .then(|| v[3].parse().unwrap())
        })
        .unwrap()
}
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t, "{a} != {b}, tolerance {t}");
}
fn reference<S: RealScalar>() {
    let a = [24.5, 23.5, 26.4, 27.1, 29.9].map(|x| S::try_from_f64_lossy(x).unwrap());
    let b = [28.4, 34.2, 29.5, 32.2, 30.1, 33.0].map(|x| S::try_from_f64_lossy(x).unwrap());
    let c = [26.1, 28.3, 24.3, 26.2, 27.8, 25.0, 26.0].map(|x| S::try_from_f64_lossy(x).unwrap());
    let groups = PreparedGroups::new(&[&a, &b, &c], StatisticsOptions::new()).unwrap();
    let mut options =
        PostHocOptions::default().with_evaluations(Budget::try_from(40_000_000).unwrap());
    options = options.with_tolerance(scioxide_core::Tolerance::new(1e-9, 1e-9).unwrap());
    let tol = if core::mem::size_of::<S>() == 4 {
        2e-5
    } else {
        3e-7
    };
    for (name, method) in [
        ("tukey", PostHocMethod::TukeyKramer),
        ("games", PostHocMethod::GamesHowell),
    ] {
        let result = groups.pairwise_with_options(method, options).unwrap();
        for (i, pair) in result.comparisons().iter().enumerate() {
            close(
                pair.p_value().try_to_f64_exact().unwrap(),
                expected(name, i, "p"),
                tol,
            );
            close(
                pair.lower().unwrap().try_to_f64_exact().unwrap(),
                expected(name, i, "low"),
                tol,
            );
            close(
                pair.upper().unwrap().try_to_f64_exact().unwrap(),
                expected(name, i, "high"),
                tol,
            );
        }
    }
    for (name, alternative) in [
        ("dunnett_two", TestAlternative::TwoSided),
        ("dunnett_greater", TestAlternative::Greater),
        ("dunnett_less", TestAlternative::Less),
    ] {
        let result = groups
            .compare_control_with_options(0, alternative, options)
            .unwrap();
        for (i, pair) in result.comparisons().iter().enumerate() {
            close(
                pair.p_value().try_to_f64_exact().unwrap(),
                expected(name, i, "p"),
                tol,
            );
            if let Some(low) = pair.lower() {
                close(
                    low.try_to_f64_exact().unwrap(),
                    expected(name, i, "low"),
                    tol,
                );
            }
            if let Some(high) = pair.upper() {
                close(
                    high.try_to_f64_exact().unwrap(),
                    expected(name, i, "high"),
                    tol,
                );
            }
        }
    }
    for (name, result) in [
        ("welch", groups.anova(GroupTestMethod::Welch).unwrap()),
        ("alexander", groups.alexander_govern().unwrap()),
        (
            "kruskal",
            kruskal_wallis_test_with_options(&[&a, &b, &c], StatisticsOptions::new()).unwrap(),
        ),
    ] {
        close(
            result.statistic().try_to_f64_exact().unwrap(),
            expected(name, 0, "stat"),
            tol * 5.,
        );
        close(
            result.p_value().try_to_f64_exact().unwrap(),
            expected(name, 0, "p"),
            tol,
        );
    }
    let x = [0., 0., 1., 2., 3., 4., 5.].map(|v| S::try_from_f64_lossy(v).unwrap());
    let y = [1., 2., 2., 4., 4., 7., 8.].map(|v| S::try_from_f64_lossy(v).unwrap());
    for (family, method, intercept) in [
        (
            "theil_separate",
            RobustSlopeMethod::TheilSen,
            RobustIntercept::Separate,
        ),
        (
            "theil_joint",
            RobustSlopeMethod::TheilSen,
            RobustIntercept::Joint,
        ),
        (
            "siegel_separate",
            RobustSlopeMethod::Siegel,
            RobustIntercept::Separate,
        ),
        (
            "siegel_hierarchical",
            RobustSlopeMethod::Siegel,
            RobustIntercept::Joint,
        ),
    ] {
        let mut work = RobustSlopeWorkspace::new(7, method, StatisticsOptions::new()).unwrap();
        let result = work
            .evaluate(&x, &y, intercept, Default::default())
            .unwrap();
        close(
            result.slope().try_to_f64_exact().unwrap(),
            expected(family, 0, "slope"),
            tol,
        );
        close(
            result.intercept().try_to_f64_exact().unwrap(),
            expected(family, 0, "intercept"),
            tol,
        );
        if let Some(interval) = result.slope_interval() {
            close(
                interval.lower().try_to_f64_exact().unwrap(),
                expected(family, 0, "low"),
                tol,
            );
            close(
                interval.upper().try_to_f64_exact().unwrap(),
                expected(family, 0, "high"),
                tol,
            );
        }
    }
}
#[test]
fn independent_integrals_f64() {
    reference::<f64>();
}
#[test]
fn independent_integrals_f32() {
    reference::<f32>();
}
#[test]
fn centered_line_and_nested_models() {
    let x = [1e12, 1e12 + 1., 1e12 + 2., 1e12 + 3., 1e12 + 4.];
    let y = [1., 2., 4., 4., 5.];
    let result = linear_regression(&x, &y).unwrap();
    close(result.line().slope, 1., 1e-12);
    close(
        result
            .slope_inference(Default::default(), Default::default())
            .unwrap()
            .unwrap()
            .standard_error(),
        (0.8_f64 / 30.).sqrt(),
        1e-12,
    );
    let a = [1., 0., 1., 1., 1., 2., 1., 3., 1., 4.];
    let full = PreparedRegression::new(
        MatrixView::from_slice(5, 2, &a).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let reduced = PreparedRegression::new(
        MatrixView::from_slice(5, 1, &[1.; 5]).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let comparison = full.compare_nested(&reduced, &y).unwrap();
    close(comparison.statistic(), 37.5, 1e-11);
    close(comparison.p_value(), expected("student", 0, "p"), 1e-12);
}
#[test]
fn seeded_small_sample_coverage() {
    // Independent Box–Muller normal samples and closed-form standard errors.
    // 4096 draws: 95% coverage must lie inside a conservative six-sigma
    // binomial envelope fixed before execution (no seed/threshold tuning).
    let mut state = 0x549_2026_u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / 9007199254740992.
    };
    let x = [-2., -1., 0., 1., 2.];
    let critical = ConfidenceProbability::default()
        .student_t_critical_value_with_options(3., Default::default())
        .unwrap();
    let mut covered = 0;
    for _ in 0..4096 {
        let y = x.map(|v| {
            let u = uniform();
            let angle = core::f64::consts::TAU * uniform();
            1. + 2. * v + (-2. * u.ln()).sqrt() * angle.cos()
        });
        let mean = y.iter().sum::<f64>() / 5.;
        let slope = x.iter().zip(y).map(|(&a, b)| a * b).sum::<f64>() / 10.;
        let ss = x
            .iter()
            .zip(y)
            .map(|(&a, b)| (b - mean - slope * a).powi(2))
            .sum::<f64>();
        let se = (ss / 30.).sqrt();
        covered += usize::from((slope - 2.).abs() <= critical * se);
    }
    let rate = covered as f64 / 4096.;
    let bound = 6. * (0.95_f64 * 0.05 / 4096.).sqrt();
    assert!((rate - 0.95).abs() < bound, "coverage {rate}");
}

#[test]
fn calibrated_familywise_error() {
    let a = [-2., -1., 0., 0., 1., 2.];
    let b = [-1., 0., 1., 1., 2., 3.];
    let groups = PreparedGroups::new(&[&a, &b, &a], StatisticsOptions::new()).unwrap();
    let options = PostHocOptions::default().with_evaluations(Budget::try_from(20_000_000).unwrap());
    let range = groups
        .pairwise_with_options(PostHocMethod::TukeyKramer, options)
        .unwrap();
    let control = groups
        .compare_control_with_options(0, Default::default(), options)
        .unwrap();
    // Pooled variance is exactly 2 for each fixture group, with n=6.
    let q = (range.comparisons()[0].upper().unwrap() - range.comparisons()[0].difference())
        / (2.0_f64 / 6.).sqrt();
    let t = (control.comparisons()[0].upper().unwrap() - control.comparisons()[0].difference())
        / (4.0_f64 / 6.).sqrt();
    let mut seed = 0x549_ffff_u64;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / 9007199254740992.
    };
    let mut tukey = 0;
    let mut dunnett = 0;
    for _ in 0..8192 {
        let samples: [[f64; 6]; 3] = core::array::from_fn(|_| {
            core::array::from_fn(|_| {
                let u = uniform();
                (-2. * u.ln()).sqrt() * (core::f64::consts::TAU * uniform()).cos()
            })
        });
        let means = samples.map(|g| g.iter().sum::<f64>() / 6.);
        let ss = samples
            .iter()
            .zip(means)
            .flat_map(|(g, m)| g.iter().map(move |&v| (v - m).powi(2)))
            .sum::<f64>();
        let scale = (ss / 15.).sqrt();
        let low = means.iter().copied().fold(f64::INFINITY, f64::min);
        let high = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        tukey += usize::from((high - low) / (scale / 6.0_f64.sqrt()) > q);
        dunnett += usize::from(
            (means[1] - means[0]).abs().max((means[2] - means[0]).abs())
                / (scale * (2.0_f64 / 6.).sqrt())
                > t,
        );
    }
    let bound = 6. * (0.05_f64 * 0.95 / 8192.).sqrt();
    for count in [tukey, dunnett] {
        let rate = count as f64 / 8192.;
        assert!((rate - 0.05).abs() < bound, "familywise error {rate}");
    }
}

fn posthoc_families<S: RealScalar>(plan: &PreparedGroups<S>) -> Vec<PostHocResult<S>> {
    let options = PostHocOptions::default().with_evaluations(Budget::try_from(40_000_000).unwrap());
    let mut families = Vec::new();
    for method in [PostHocMethod::TukeyKramer, PostHocMethod::GamesHowell] {
        families.push(plan.pairwise_with_options(method, options).unwrap());
    }
    for alternative in [
        TestAlternative::TwoSided,
        TestAlternative::Greater,
        TestAlternative::Less,
    ] {
        families.push(
            plan.compare_control_with_options(0, alternative, options)
                .unwrap(),
        );
    }
    families
}

fn posthoc_scale_family<S: RealScalar>(exponents: &[i32], shift_exponent: i32, step_exponent: i32) {
    let a = [-2., 0., 2., 4.];
    let b = [0., 1., 2., 3., 4.];
    let c = [-2., 0., 2., 4., 6., 8.];
    let base = PreparedGroups::new(&[&a, &b, &c], StatisticsOptions::new()).unwrap();
    let expected = posthoc_families(&base);
    let mut frames = exponents
        .iter()
        .map(|&e| (S::ONE.scalbn(e), S::ZERO))
        .collect::<Vec<_>>();
    frames.push((S::ONE.scalbn(step_exponent), S::ONE.scalbn(shift_exponent)));
    let tolerance = if core::mem::size_of::<S>() == 4 {
        2e-5
    } else {
        2e-12
    };
    for (scale, shift) in frames {
        let transform = |v| shift + scale * S::try_from_f64_lossy(v).unwrap();
        let (x, y, z) = (a.map(transform), b.map(transform), c.map(transform));
        let plan = PreparedGroups::new(&[&x, &y, &z], StatisticsOptions::new()).unwrap();
        let actual = posthoc_families(&plan);
        for (actual, expected) in actual.iter().zip(&expected) {
            for (pair, reference) in actual.comparisons().iter().zip(expected.comparisons()) {
                close(pair.p_value().into_f64(), reference.p_value(), tolerance);
                close(
                    pair.difference().into_f64() / scale.into_f64(),
                    reference.difference(),
                    tolerance,
                );
                close(
                    pair.degrees_of_freedom(),
                    reference.degrees_of_freedom(),
                    tolerance,
                );
                close(
                    pair.probability_error().into_f64(),
                    reference.probability_error(),
                    tolerance,
                );
                for (endpoint, reference) in [
                    (pair.lower(), reference.lower()),
                    (pair.upper(), reference.upper()),
                ] {
                    assert_eq!(endpoint.is_some(), reference.is_some());
                    if let (Some(value), Some(expected)) = (endpoint, reference) {
                        close(value.into_f64() / scale.into_f64(), expected, tolerance);
                    }
                }
            }
        }
    }
}

#[test]
fn posthoc_queries_preserve_scale_and_compensated_translation() {
    posthoc_scale_family::<f64>(&[-600, 510], 1000, 950);
    posthoc_scale_family::<f32>(&[-80, 65], 100, 80);
}

#[test]
fn posthoc_final_endpoint_overflow_is_numerical() {
    let a = [-f64::MAX, -f64::MAX, f64::MAX, f64::MAX];
    let plan = PreparedGroups::new(&[&a, &a], StatisticsOptions::new()).unwrap();
    let options = PostHocOptions::default().with_evaluations(Budget::try_from(40_000_000).unwrap());
    for method in [PostHocMethod::TukeyKramer, PostHocMethod::GamesHowell] {
        let error = plan.pairwise_with_options(method, options).err().unwrap();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
    assert!(
        plan.compare_control_with_options(0, TestAlternative::TwoSided, options)
            .is_err()
    );
}
