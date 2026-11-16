#![allow(clippy::unwrap_used)]
use scioxide_core::{
    NumericScalar, RealScalar, SciOxideErrorKind, SciOxideErrorView, StridedLayout,
};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

fn bits<S: RealScalar>(value: S) -> u64 {
    value.into_f64().to_bits()
}
fn summaries<S: RealScalar>() {
    let x = [2., 4., 4., 4., 5., 5., 7., 9.].map(|v| S::try_from_f64_lossy(v).unwrap());
    for (actual, request) in [
        (mean(&x).unwrap(), Statistic::mean()),
        (sample_variance(&x).unwrap(), Statistic::variance(1)),
        (
            sample_standard_deviation(&x).unwrap(),
            Statistic::standard_deviation(1),
        ),
        (
            median(&x).unwrap(),
            Statistic::quantile(
                QuantileProbability::new(0.5).unwrap(),
                QuantileRule::new(QuantileMethod::Linear),
            ),
        ),
        (minimum(&x).unwrap(), Statistic::minimum()),
        (maximum(&x).unwrap(), Statistic::maximum()),
    ] {
        assert_eq!(
            bits(actual),
            bits(
                statistic(&x, Unweighted, request)
                    .unwrap()
                    .scalar()
                    .unwrap()
            )
        );
    }
    assert!((mean(&x).unwrap().into_f64() - 5.).abs() < 1e-6);
    assert!((sample_variance(&x).unwrap().into_f64() - 32. / 7.).abs() < 1e-6);
    assert!((median(&x).unwrap().into_f64() - 4.5).abs() < 1e-6);
    assert_eq!(bits(quantile(&x, 0.).unwrap()), bits(minimum(&x).unwrap()));
    assert_eq!(bits(quantile(&x, 1.).unwrap()), bits(maximum(&x).unwrap()));
    assert_eq!(
        bits(quantile(&x, 0.25).unwrap()),
        bits(S::try_from_f64_lossy(4.).unwrap())
    );
    for values in [&[][..], &[S::NAN][..], &[S::ONE][..]] {
        for (actual, request) in [
            (mean(values), Statistic::mean()),
            (sample_variance(values), Statistic::variance(1)),
            (
                sample_standard_deviation(values),
                Statistic::standard_deviation(1),
            ),
        ] {
            let configured =
                statistic(values, Unweighted, request).and_then(StatisticValue::scalar);
            assert_eq!(
                actual.map(bits).map_err(|e| e.kind()),
                configured.map(bits).map_err(|e| e.kind())
            );
        }
    }
}
#[test]
fn summaries_preserve_both_scalar_widths() {
    summaries::<f64>();
    summaries::<f32>();
}

macro_rules! same {
    ($plain:expr, $configured:expr) => {{
        let a = $plain.unwrap();
        let b = $configured.unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"), "{}", stringify!($plain));
    }};
}
fn tests<S: RealScalar>() {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let x = [0.2, 1., 1.8, 2.5, 4., 5., 6.2, 8.].map(s);
    let y = [0.8, 4.1, 1.3, 7., 2.8, 8.2, 4.4, 6.].map(s);
    let groups = [&x[..], &y[..]];
    let o = ClassicalOptions::default();
    same!(
        t_test_one_sample(&x, s(3.)),
        t_test_one_sample_with_options(&x, s(3.), o)
    );
    same!(t_test_paired(&x, &y), t_test_paired_with_options(&x, &y, o));
    same!(
        t_test_independent(&x, &y),
        t_test_independent_with_options(&x, &y, TTestVariance::Welch, o)
    );
    same!(
        binomial_test(3, 8, s(0.5)),
        binomial_test_with_options(3, 8, s(0.5), o)
    );
    same!(
        quantile_test(&x, s(3.), 0.5),
        quantile_test_with_options(&x, s(3.), 0.5, o)
    );
    same!(
        poisson_means_test([3, 8], [s(10.), s(20.)], S::ZERO),
        poisson_means_test_with_options([3, 8], [s(10.), s(20.)], S::ZERO, o)
    );
    same!(pearson_test(&x, &y), pearson_test_with_options(&x, &y, o));
    same!(
        kendall_test(&x, &y, KendallVariant::TauB),
        kendall_test_with_options(&x, &y, KendallVariant::TauB, o)
    );
    same!(spearman_test(&x, &y), spearman_test_with_options(&x, &y, o));
    same!(
        chatterjee_test(&x, &y),
        chatterjee_test_with_options(&x, &y, o.with_alternative(TestAlternative::Greater))
    );
    same!(somers_test(&x, &y), somers_test_with_options(&x, &y, o));
    same!(
        mann_whitney_test(&x, &y),
        mann_whitney_test_with_options(&x, &y, o)
    );
    same!(rank_sum_test(&x, &y), rank_sum_test_with_options(&x, &y, o));
    same!(
        wilcoxon_test(&x),
        wilcoxon_test_with_options(&x, SignedRankZeros::Discard, o)
    );
    same!(
        brunner_munzel_test(&x, &y),
        brunner_munzel_test_with_options(&x, &y, BrunnerMunzelCalibration::Student, o)
    );
    same!(bws_test(&x, &y), bws_test_with_options(&x, &y, o));
    same!(
        anderson_k_sample_test(&groups),
        anderson_k_sample_test_with_options(&groups, AndersonKVariant::Midrank, o)
    );
    same!(
        median_test(&groups),
        median_test_with_options(&groups, MedianTies::Below, o)
    );
    same!(skew_test(&x), skew_test_with_options(&x, o));
    same!(kurtosis_test(&x), kurtosis_test_with_options(&x, o));
    same!(normality_test(&x), normality_test_with_options(&x, o));
    same!(jarque_bera_test(&x), jarque_bera_test_with_options(&x, o));
    let cdf = |v: S| Ok(v / s(10.));
    same!(
        ks_one_sample_statistic(&x, cdf),
        ks_one_sample_statistic_with_options(&x, cdf, o)
    );
    same!(
        ks_one_sample_test(&x, cdf),
        ks_one_sample_test_with_options(&x, cdf, o)
    );
    same!(
        cramer_von_mises_statistic(&x, cdf),
        cramer_von_mises_statistic_with_options(&x, cdf, o)
    );
    same!(
        cramer_von_mises_test(&x, cdf),
        cramer_von_mises_test_with_options(&x, cdf, o)
    );
    same!(
        anderson_darling_statistic(
            &x,
            |v: S| cdf(v).map(RealScalar::ln),
            |v: S| cdf(v).map(|p| (S::ONE - p).ln())
        ),
        anderson_darling_statistic_with_options(
            &x,
            |v: S| cdf(v).map(RealScalar::ln),
            |v: S| cdf(v).map(|p| (S::ONE - p).ln()),
            o
        )
    );
    same!(
        mood_scale_test(&x, &y),
        mood_scale_test_with_options(&x, &y, o)
    );
    same!(
        ansari_bradley_test(&x, &y),
        ansari_bradley_test_with_options(&x, &y, o)
    );
    same!(
        shapiro_wilk_statistic(&x),
        shapiro_wilk_statistic_with_options(&x, o)
    );
    same!(shapiro_wilk_test(&x), shapiro_wilk_test_with_options(&x, o));
    same!(
        ks_two_sample_test(&x, &y),
        ks_two_sample_test_with_options(&x, &y, o)
    );
    same!(
        cramer_von_mises_two_sample_test(&x, &y),
        cramer_von_mises_two_sample_test_with_options(&x, &y, o)
    );
    same!(
        bartlett_test(&groups),
        bartlett_test_with_options(&groups, o)
    );
    same!(
        levene_test(&groups),
        levene_test_with_options(&groups, VarianceCenter::Median, o)
    );
    same!(
        fligner_test(&groups),
        fligner_test_with_options(&groups, VarianceCenter::Median, o)
    );
    same!(
        kruskal_wallis_test(&groups),
        kruskal_wallis_test_with_options(&groups, Default::default())
    );
    same!(
        entropy(&x),
        entropy_with_options(&x, None, Default::default())
    );
    same!(
        relative_entropy(&x, &y),
        entropy_with_options(&x, Some(&y), Default::default())
    );
    same!(
        differential_entropy(&x),
        differential_entropy_with_options(&x, Default::default())
    );
    let empty: &[S] = &[];
    for (plain, configured) in [
        (
            t_test_one_sample(empty, S::ZERO),
            t_test_one_sample_with_options(empty, S::ZERO, o),
        ),
        (
            pearson_test(empty, empty),
            pearson_test_with_options(empty, empty, o),
        ),
    ] {
        assert_eq!(plain.unwrap_err(), configured.unwrap_err());
    }
}
#[test]
fn classical_defaults_match_both_scalar_widths() {
    tests::<f64>();
    tests::<f32>();
}

#[test]
fn scalar_and_model_answers_match_independent_values() {
    let x = [2., 4., 4., 4., 5., 5., 7., 9.];
    let interval = mean_confidence_interval(&x).unwrap();
    let margin = 2.364624 * (32.0_f64 / 56.).sqrt();
    assert!((interval.bounds().lower() - (5. - margin)).abs() < 1e-6);
    let error = quantile(&x, 1.5).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument.as_str()=="probability")
    );
    let error = shapiro_wilk_test(&vec![1.; 5001]).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument.as_str()=="x")
    );
    let table = ContingencyView::new(&[10, 20, 20, 10], 2, 2).unwrap();
    let chi = chi2_contingency_test(table).unwrap();
    assert!((chi.statistic() - 5.4).abs() < 1e-14);
    assert!((chi.p_value() - 0.0201368).abs() < 1e-7);
    assert!((fisher_exact_test(table).unwrap().p_value() - 0.0193832).abs() < 1e-7);
    let h = histogram(&[0., 0.1, 0.4, 0.6, 1.], 4).unwrap();
    assert_eq!(h.counts(), &[2, 1, 1, 1]);
    assert_eq!(
        histogram(&x, 0).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        histogram(&[1., 1.], 2).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let kde = GaussianKde::new(&x).unwrap();
    let grid = [3., 5., 7.];
    let density = kde.evaluate(&grid).unwrap();
    let h = 8.0_f64.powf(-0.2) * (32.0_f64 / 7.).sqrt();
    for (&point, &actual) in grid.iter().zip(&density) {
        assert_eq!(actual.to_bits(), kde.pdf(&[point]).unwrap().to_bits());
        let expected = x
            .iter()
            .map(|value| (-0.5 * ((point - value) / h).powi(2)).exp())
            .sum::<f64>()
            / (8. * h * (2. * core::f64::consts::PI).sqrt());
        assert!((actual - expected).abs() < 1e-14);
    }
    let design = MatrixView::from_slice(4, 2, &[1., 0., 1., 1., 1., 2., 1., 3.]).unwrap();
    let response = [1., 3., 7., 9.];
    let fit = ordinary_least_squares(design, &response).unwrap();
    assert!((fit.coefficients().as_slice()[0] - 0.8).abs() < 1e-13);
    assert!((fit.coefficients().as_slice()[1] - 2.8).abs() < 1e-13);
    let view = BatchView::new(
        &x,
        StridedLayout::new(&[2, 4], &[4, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    same!(
        statistics(view, Statistic::mean()),
        statistics_with_options(view, Statistic::mean(), Default::default())
    );
}

fn remaining_defaults<S: RealScalar>() {
    let s = |value| S::try_from_f64_lossy(value).unwrap();
    let x = [0.2, 1., 1.8, 2.5, 4., 5., 6.2, 8.].map(s);
    let y = [0.8, 4.1, 1.3, 7., 2.8, 8.2, 4.4, 6.].map(s);
    let o = ClassicalOptions::default();
    let a = TTestSummary {
        mean: s(3.),
        variance: s(2.5),
        count: 5,
    };
    let b = TTestSummary {
        mean: s(4.),
        variance: s(2.5),
        count: 5,
    };
    same!(
        t_test_from_summaries(a, b),
        t_test_from_summaries_with_options(a, b, TTestVariance::Welch, o)
    );
    let binary = [
        S::ZERO,
        S::ONE,
        S::ZERO,
        S::ONE,
        S::ZERO,
        S::ONE,
        S::ZERO,
        S::ONE,
    ];
    same!(
        point_biserial_test(&binary, &x),
        point_biserial_test_with_options(&binary, &x, o)
    );
    same!(
        epps_singleton_two_sample_test(&x, &y, [0.4, 0.8]),
        epps_singleton_two_sample_test_with_options(&x, &y, [0.4, 0.8], o)
    );
    let ranks = [1., 2., 3., 1., 2., 3.].map(s);
    let data = MatrixView::from_slice(2, 3, &ranks).unwrap();
    same!(
        page_trend_test(data, &[0, 1, 2]),
        page_trend_test_with_options(
            data,
            &[0, 1, 2],
            o.with_alternative(TestAlternative::Greater)
        )
    );
    same!(
        friedman_test(data),
        friedman_test_with_options(data, Default::default())
    );
    same!(
        distribution_distance(&x, &y, DistributionDistance::Wasserstein),
        distribution_distance_with_options(
            &x,
            Unweighted,
            &y,
            Unweighted,
            DistributionDistance::Wasserstein,
            Default::default()
        )
    );
    same!(
        isotonic_regression(&y),
        isotonic_regression_with_options(
            &y,
            Unweighted,
            IsotonicDirection::Increasing,
            Default::default()
        )
    );
    let view = BatchView::new(&x, StridedLayout::new(&[8], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let other =
        BatchView::new(&y, StridedLayout::new(&[8], &[1]).unwrap(), Axis::Index(0)).unwrap();
    same!(
        cdf(view, &[s(2.), s(4.)], CdfMethod::Empirical),
        cdf_with_options(
            view,
            &[s(2.), s(4.)],
            CdfMethod::Empirical,
            Default::default()
        )
    );
    same!(
        reduce_moments(view),
        reduce_moments_with_options(view, Default::default())
    );
    let limits = TrimLimits::new(None, None).unwrap();
    same!(
        reduce_trimmed_moments(view, limits),
        reduce_trimmed_moments_with_options(view, limits, Default::default())
    );
    same!(
        reduce_covariance(view, other),
        reduce_covariance_with_options(view, other, Default::default())
    );
    same!(
        reduce_moment(view, 2, None),
        reduce_moment_with_options(view, 2, None, Default::default())
    );
    let weighted = view
        .with_weights(Weights::<_, Frequency>::new(&[1u64; 8]))
        .unwrap();
    same!(
        reduce_weighted_moments(weighted),
        reduce_weighted_moments_with_options(weighted, Default::default())
    );
    same!(
        reduce_weighted_covariance(weighted, other),
        reduce_weighted_covariance_with_options(weighted, other, Default::default())
    );
    same!(
        time_series_summary(&x, TimeSeriesStatistic::Acf, 2),
        time_series_summary_with_options(
            &x,
            None,
            TimeSeriesStatistic::Acf,
            2,
            Default::default(),
            Default::default()
        )
    );
    let matrix = MatrixView::from_slice(8, 1, &x).unwrap();
    same!(
        sample_covariance(matrix),
        sample_covariance_with_options(matrix, Default::default())
    );
    same!(
        ledoit_wolf_covariance(matrix),
        ledoit_wolf_covariance_with_options(matrix, Default::default())
    );
    same!(
        frequency_covariance(matrix, Weights::<_, Frequency>::new(&[1u64; 8])),
        frequency_covariance_with_options(
            matrix,
            Weights::<_, Frequency>::new(&[1u64; 8]),
            Default::default()
        )
    );
    let observations =
        x.map(|time| SurvivalObservation::new(None, time, SurvivalOutcome::Event).unwrap());
    same!(
        kaplan_meier(&observations),
        kaplan_meier_with_options(&observations, Default::default(), Default::default())
    );
    same!(
        nelson_aalen(&observations),
        nelson_aalen_with_options(&observations, Default::default(), Default::default())
    );
    let second =
        y.map(|time| SurvivalObservation::new(None, time, SurvivalOutcome::Event).unwrap());
    same!(
        logrank_test(&observations, &second),
        logrank_test_with_options(&observations, &second, Default::default())
    );
    let snapshot = OnlineMoments::from_slice(&x, MomentOrder::Second)
        .unwrap()
        .snapshot();
    same!(
        snapshot.confidence_interval_of_mean(),
        snapshot.confidence_interval_of_mean_with_options(
            Default::default(),
            MeanIntervalMethod::StudentT,
            Default::default()
        )
    );
    let confidence = ConfidenceProbability::default();
    same!(
        confidence.student_t_critical_value(7.),
        confidence.student_t_critical_value_with_options(7., Default::default())
    );
    let chain_values: Vec<_> = x.iter().chain(&y).copied().collect();
    let chains = MatrixView::from_slice(2, 8, &chain_values).unwrap();
    same!(
        posterior_summary(chains),
        posterior_summary_with_options(chains, Default::default())
    );
    let snapshot = reduce_weighted_moments(weighted).unwrap()[0];
    same!(
        snapshot.confidence_interval_of_weighted_mean(),
        snapshot.confidence_interval_of_weighted_mean_with_options(
            Default::default(),
            MeanIntervalMethod::StudentT,
            Default::default()
        )
    );
    same!(
        probplot(&x, &DistributionFitModel::Normal, &[S::ZERO, S::ONE])
            .map(ProbabilityPlot::into_parts),
        probplot_with_options(
            &x,
            &DistributionFitModel::Normal,
            &[S::ZERO, S::ONE],
            Default::default()
        )
        .map(ProbabilityPlot::into_parts)
    );
    let plain = huber_location(&x).unwrap();
    let configured =
        huber_location_with_options(&x, HuberOptions::default().with_tuning(s(1.345))).unwrap();
    assert_eq!(
        plain.coefficients().as_slice(),
        configured.coefficients().as_slice()
    );
    let plain = linear_odr(matrix, &y, &[s(1.)], s(1.)).unwrap();
    let configured =
        linear_odr_with_options(matrix, &y, &[s(1.)], s(1.), Default::default()).unwrap();
    assert_eq!(plain.coefficients(), configured.coefficients());
}
#[test]
fn remaining_sample_defaults_match_both_widths() {
    remaining_defaults::<f64>();
    remaining_defaults::<f32>();
}
#[test]
fn contingency_defaults_match_configured_calls() {
    let table = ContingencyView::new(&[1, 3, 3, 1], 2, 2).unwrap();
    same!(
        fisher_exact_test(table),
        fisher_exact_test_with_options(table, Default::default())
    );
    same!(
        barnard_exact_test(table),
        barnard_exact_test_with_options(table, Default::default())
    );
    same!(
        boschloo_exact_test(table),
        boschloo_exact_test_with_options(table, Default::default())
    );
    same!(
        somers_contingency_test(table),
        somers_contingency_test_with_options(table, Default::default())
    );
    let method = PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg);
    same!(
        adjust_pvalues(&[0.01, 0.04, 0.2], method),
        adjust_pvalues_with_options(&[0.01, 0.04, 0.2], method, Default::default())
    );
    let error = adjust_pvalues(&[], method).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument.as_str() == "pvalues")
    );
}

#[test]
fn auto_ks_falls_back_when_the_exact_law_exhausts_its_work() {
    let x: Vec<f64> = (0..200).map(|i| (i as f64 + 0.5) / 200.0 * 0.9).collect();
    let auto = ks_one_sample_test::<f64>(&x, Ok).unwrap();
    assert_eq!(auto.method(), DistributionMethod::Asymptotic);
    assert!((auto.statistic() - 0.10225).abs() < 1e-12);
    assert!(auto.p_value() > 0.0 && auto.p_value() < 0.05);
    let exact = ClassicalOptions::default().with_calibration(TestCalibration::Exact);
    assert_eq!(
        ks_one_sample_test_with_options::<f64>(&x, Ok, exact)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let small: Vec<f64> = x.iter().take(20).map(|v| v * 10.0).collect();
    let small = ks_one_sample_test::<f64>(&small, Ok).unwrap();
    assert_eq!(small.method(), DistributionMethod::Exact);
}
