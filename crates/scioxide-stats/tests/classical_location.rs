//! Independent closed-form checks for location and discrete inference.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_stats::*;
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b}");
}
fn location<S: RealScalar>(tol: f64) {
    let x = [1., 2., 3., 4., 5.].map(|x| S::try_from_f64_lossy(x).unwrap());
    let a =
        t_test_one_sample_with_options(&x, S::try_from_f64_lossy(3.).unwrap(), Default::default())
            .unwrap();
    close(a.statistic().into_f64(), 0., tol);
    close(a.p_value(), 1., tol);
    let y = x.map(|v| v + S::ONE);
    let b =
        t_test_independent_with_options(&x, &y, TTestVariance::Welch, Default::default()).unwrap();
    close(b.statistic().into_f64(), -1., tol);
    assert_eq!(
        b.method(),
        ClassicalMethod::StudentT {
            degrees_of_freedom: 8.
        }
    );
    let summary = |m| TTestSummary {
        mean: S::try_from_f64_lossy(m).unwrap(),
        variance: S::try_from_f64_lossy(2.5).unwrap(),
        count: 5,
    };
    let c = t_test_from_summaries_with_options(
        summary(3.),
        summary(4.),
        TTestVariance::Welch,
        Default::default(),
    )
    .unwrap();
    close(b.p_value(), c.p_value(), tol);
    let zero = [S::ZERO; 5];
    let d = t_test_paired_with_options(&x, &zero, Default::default()).unwrap();
    close(d.statistic().into_f64(), 18_f64.sqrt(), tol * 10.);
    let r = pearson_test_with_options(&x, &y, Default::default()).unwrap();
    close(r.statistic().into_f64(), 1., tol);
}
#[test]
fn both_precisions() {
    location::<f32>(2e-6);
    location::<f64>(2e-13);
}
#[test]
fn exact_binomial() {
    // P(K <= 1 or K >= 4), K~Bin(5,1/2) = 12/32.
    let a = binomial_test_with_options(1, 5, 0.5_f64, Default::default()).unwrap();
    close(a.p_value(), 0.375, 1e-13);
    let b = binomial_test_with_options(0, 10, 0.5_f32, Default::default()).unwrap();
    close(b.p_value(), 1. / 512., 1e-12);
    assert_eq!(
        binomial_test_with_options(0, 10, 0.0_f64, Default::default())
            .unwrap()
            .p_value(),
        1.
    );
    assert_eq!(
        binomial_test_with_options(1, 10, 0.0_f64, Default::default())
            .unwrap()
            .p_value(),
        0.
    );
}
#[test]
fn quantile_ties() {
    let x = [0., 0., 0., 1., 2.];
    let _work = [0.; 5];
    let options = ClassicalOptions::default().with_alternative(TestAlternative::Greater);
    let a = quantile_test_with_options(&x, 0., 0.5, options).unwrap();
    close(a.statistic(), 3., 0.);
    close(a.p_value(), 26. / 32., 1e-13);
    assert!(a.interval().is_none());
}
#[test]
fn poisson_reference_and_reflection() {
    let a = poisson_means_test_with_options([0, 3], [100., 100.], 0., Default::default()).unwrap();
    close(a.statistic(), -3_f64.sqrt(), 1e-13);
    close(
        a.p_value(),
        0.08837900945483522,
        a.omitted_null_mass() + 1e-14,
    );
    assert!(a.omitted_null_mass() < 5e-13);
    let a = poisson_means_test_with_options([3, 8], [10., 20.], 0.1, Default::default()).unwrap();
    let b = poisson_means_test_with_options([8, 3], [20., 10.], -0.1, Default::default()).unwrap();
    close(a.statistic(), -b.statistic(), 1e-13);
    close(a.p_value(), b.p_value(), 1e-13);
    assert!(
        poisson_means_test_with_options(
            [10, 20],
            [1., 1.],
            0.,
            ClassicalOptions::default().with_max_work(1)
        )
        .is_err()
    );
}

fn poisson_scaling<S: RealScalar>(scales: &[f64], tolerance: f64) {
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    for alternative in [
        TestAlternative::Less,
        TestAlternative::Greater,
        TestAlternative::TwoSided,
    ] {
        let options = ClassicalOptions::default()
            .with_alternative(alternative)
            .with_max_workspace_bytes(0);
        let baseline = poisson_means_test_with_options(
            [3, 8],
            [convert(10.), convert(20.)],
            convert(0.1),
            options,
        )
        .unwrap();
        for &scale in scales {
            let exposures = [10. * scale, 20. * scale].map(convert);
            let difference = convert(0.1 / scale);
            let out =
                poisson_means_test_with_options([3, 8], exposures, difference, options).unwrap();
            close(
                out.statistic().into_f64(),
                baseline.statistic().into_f64(),
                tolerance,
            );
            close(out.p_value(), baseline.p_value(), tolerance);
            assert!(out.omitted_null_mass() < 4e-13);
            let reverse = match alternative {
                TestAlternative::Less => TestAlternative::Greater,
                TestAlternative::Greater => TestAlternative::Less,
                other => other,
            };
            let reflected = poisson_means_test_with_options(
                [8, 3],
                [exposures[1], exposures[0]],
                -difference,
                options.with_alternative(reverse),
            )
            .unwrap();
            close(
                out.statistic().into_f64(),
                -reflected.statistic().into_f64(),
                tolerance,
            );
            close(out.p_value(), reflected.p_value(), tolerance);
            assert_eq!(reflected.retained(), [8, 3]);
        }
    }
}

#[test]
fn poisson_extreme_exposures_and_directional_reflection() {
    poisson_scaling::<f64>(&[1e-200, 1., 1e200], 2e-13);
    poisson_scaling::<f32>(&[1e-20, 1., 1e20], 2e-6);
    // A ratio whose square underflows requires scaled pair comparisons.
    let options = ClassicalOptions::default();
    let out = poisson_means_test_with_options([0, 3], [1e150, 1e-150], 0., options).unwrap();
    let reflected = poisson_means_test_with_options([3, 0], [1e-150, 1e150], 0., options).unwrap();
    close(out.statistic(), -3_f64.sqrt(), 1e-13);
    close(out.statistic(), -reflected.statistic(), 1e-13);
    close(out.p_value(), reflected.p_value(), 1e-13);
    // P(Poisson(3) >= 3), with only j=0 retained in the other margin.
    close(out.p_value(), 1. - 8.5 * (-3_f64).exp(), 4e-13);
}

#[test]
fn poisson_exact_joint_work_boundary_and_omitted_mass() {
    use scioxide_random::{Cdf, Poisson, Quantile, Survival};
    for counts in [[10, 20], [90, 110]] {
        let mean = (counts[0] + counts[1]) as f64 / 2.;
        let law = Poisson::new(mean).unwrap();
        let low = law.quantile(1e-13).unwrap();
        let high = law.quantile(1. - 1e-13).unwrap();
        let width = high - low + 1;
        if counts[0] == 90 {
            assert!(width > 128);
        }
        let work = width * width;
        let omitted = if low == 0 {
            0.
        } else {
            law.cdf(low - 1).unwrap()
        } + law.sf(high).unwrap();
        let options = ClassicalOptions::default()
            .with_max_work(work)
            .with_max_workspace_bytes(0);
        let out = poisson_means_test_with_options(counts, [1.; 2], 0., options).unwrap();
        close(
            out.omitted_null_mass(),
            2. * omitted - omitted * omitted,
            1e-27,
        );
        assert!(out.omitted_null_mass() < 4e-13);
        // Work allowances are budgets, not resource limits.
        let error =
            poisson_means_test_with_options(counts, [1.; 2], 0., options.with_max_work(work - 1))
                .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
    }
}

#[test]
fn poisson_zero_counts_shortcut_and_invalid_domains() {
    let options = ClassicalOptions::default()
        .with_max_work(0)
        .with_max_workspace_bytes(0);
    for (counts, exposures, difference) in [([0, 0], [1.; 2], 0.), ([1, 1], [1.; 2], 3.)] {
        let out = poisson_means_test_with_options(counts, exposures, difference, options).unwrap();
        assert_eq!(out.statistic(), 0.);
        assert_eq!(out.p_value(), 1.);
        assert_eq!(out.omitted_null_mass(), 0.);
    }
    for (exposures, difference) in [
        ([0., 1.], 0.),
        ([-1., 1.], 0.),
        ([f64::INFINITY, 1.], 0.),
        ([f64::NAN, 1.], 0.),
        ([1.; 2], f64::NAN),
        ([1.; 2], f64::INFINITY),
    ] {
        assert!(
            poisson_means_test_with_options([3, 8], exposures, difference, Default::default())
                .is_err()
        );
    }
    assert!(
        poisson_means_test_with_options(
            [3, 8],
            [1.; 2],
            0.,
            options.with_calibration(TestCalibration::Exact)
        )
        .is_err()
    );
    assert!(
        poisson_means_test_with_options(
            [usize::MAX; 2],
            [1.; 2],
            0.,
            options.with_max_work(usize::MAX)
        )
        .is_err()
    );
}
#[test]
fn invalid_domains() {
    assert!(t_test_one_sample_with_options(&[1.], 0., Default::default()).is_err());
    assert!(t_test_one_sample_with_options(&[1., 1.], 0., Default::default()).is_err());
    assert!(t_test_paired_with_options(&[1., 2.], &[1.], Default::default()).is_err());
    assert!(pearson_test_with_options(&[1., 1.], &[1., 2.], Default::default()).is_err());
    assert!(binomial_test_with_options(3, 2, 0.5, Default::default()).is_err());
    assert!(t_test_one_sample_with_options(&[1., f64::NAN], 0., Default::default()).is_err());
}
#[test]
fn scaled_student() {
    let a = t_test_one_sample_with_options(
        &[1e200, 2e200, 3e200, 4e200, 5e200],
        0.,
        Default::default(),
    )
    .unwrap();
    close(a.statistic(), 18_f64.sqrt(), 1e-12);
    let b = t_test_one_sample_with_options(
        &[1e-200, 2e-200, 3e-200, 4e-200, 5e-200],
        0.,
        Default::default(),
    )
    .unwrap();
    close(a.statistic(), b.statistic(), 1e-12);
}
