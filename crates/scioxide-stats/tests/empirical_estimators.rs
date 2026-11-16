#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, RealScalar, StridedLayout};
use scioxide_stats::*;
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn check<S: RealScalar>() {
    let edges = [s::<S>(0.), s(1.), s(3.)];
    let plan = BinPlan::new(&[&edges]).unwrap();
    assert_eq!(plan.locate(&[s(1.)]).unwrap(), BinLocation::Inside(1));
    assert_eq!(plan.locate(&[s(3.)]).unwrap(), BinLocation::Inside(1));
    assert_eq!(plan.locate(&[s(-1.)]).unwrap(), BinLocation::Underflow(0));
    let x = [s::<S>(-1.), s(0.), s(1.), s(2.), s(3.), s(4.)];
    let layout = StridedLayout::new(&[6], &[1]).unwrap();
    let view = BatchView::new(&x, layout, Axis::Index(0)).unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut w = BinnedWorkspace::default();
    w.try_reserve(
        &(BinnedWorkspace::workspace_requirements(
            &plan,
            view,
            BinnedStatistic::Density,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    let mut out = [BinEstimate::default(); 2];
    let mut report = [BinnedDiagnostics::default()];
    binned_statistics_into(
        &plan,
        view,
        None,
        BinnedStatistic::Density,
        options,
        &mut out,
        &mut report,
        &mut w,
        context,
    )
    .unwrap();
    assert_eq!(out[0].value, Some(s(0.25)));
    assert_eq!(out[1].value, Some(s(0.375)));
    assert_eq!(report[0].admitted_mass, Some(s(6.)));
    assert_eq!(report[0].retained_mass, Some(s(4.)));
    assert_eq!(report[0].underflow_mass, Some(s(1.)));
    assert_eq!(report[0].overflow_mass, Some(s(1.)));
    w.try_reserve(
        &(BinnedWorkspace::workspace_requirements(
            &plan,
            view,
            BinnedStatistic::Mean,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    binned_statistics_into(
        &plan,
        view,
        Some(&x),
        BinnedStatistic::Mean,
        options,
        &mut out,
        &mut report,
        &mut w,
        context,
    )
    .unwrap();
    assert_eq!(out[1].value, Some(s(2.)));
    let previous = out;
    assert!(
        binned_statistics_into(
            &plan,
            view,
            None,
            BinnedStatistic::Mean,
            options,
            &mut out,
            &mut report,
            &mut w,
            context
        )
        .is_err()
    );
    assert_eq!(previous, out);
    let d = EmpiricalDistribution::new(&[s::<S>(0.), s(1.), s(2.), s(3.)], Unweighted).unwrap();
    assert_eq!(d.cdf(s(1.)).unwrap(), s(0.5));
    assert_eq!(d.survival(s(1.)).unwrap(), s(0.5));
    assert_eq!(
        d.estimated_cdf(s(1.5), QuantileRule::new(QuantileMethod::Linear))
            .unwrap(),
        s(0.5)
    );
    let weighted = {
        let pairs = &[(s::<S>(1.), s(2.)), (s(2.), s(2.))];
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    assert_eq!(weighted.cdf(s(1.)).unwrap(), s(0.5));
    assert_eq!(weighted.survival(s(2.)).unwrap(), s(0.));
}
#[test]
fn both_precisions() {
    check::<f32>();
    check::<f64>();
}
#[test]
fn extremes_and_small_survival() {
    let d = {
        let pairs = &[(0., f64::MAX), (1., f64::MAX), (2., 1.)];
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    assert!(d.survival(1.).unwrap() > 0.);
    let d = EmpiricalDistribution::new(&[-f64::MAX, f64::MAX], Unweighted).unwrap();
    assert_eq!(
        d.estimated_cdf(0., QuantileRule::new(QuantileMethod::Linear))
            .unwrap(),
        0.5
    );
    assert!(d.cdf(f64::NAN).is_err());
}
#[test]
fn replication_and_cartesian_edges() {
    let plan = BinPlan::uniform(&[(0., 2., 2), (0., 2., 2)]).unwrap();
    assert_eq!(plan.locate(&[2., 2.]).unwrap(), BinLocation::Inside(3));
    let plan = BinPlan::uniform(&[(0., 3., 1)]).unwrap();
    let x = [0., 1., 2.];
    let values = [0., 10., 20.];
    let weights = [1, 2, 1];
    let view = BatchView::new(&x, StridedLayout::new(&[3], &[1]).unwrap(), Axis::Index(0))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&weights))
        .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut w = BinnedWorkspace::default();
    let mut out = [BinEstimate::default()];
    let mut report = [BinnedDiagnostics::default()];
    for statistic in [BinnedStatistic::Mean, BinnedStatistic::Median] {
        w.try_reserve(
            &(BinnedWorkspace::workspace_requirements(&plan, view, statistic, options, context)
                .unwrap()),
        )
        .unwrap();
        binned_statistics_into(
            &plan,
            view,
            Some(&values),
            statistic,
            options,
            &mut out,
            &mut report,
            &mut w,
            context,
        )
        .unwrap();
        assert_eq!(out[0].count, 4);
        assert_eq!(out[0].value, Some(10.));
    }
}

fn kde_checks<S: RealScalar>() {
    let x = [s::<S>(0.)];
    let sd = [s::<S>(1.)];
    let view = BatchView::new(&x, StridedLayout::new(&[1], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let options = KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&sd));
    let model = GaussianKde::build(view, 0, options, ExecutionContext::sequential()).unwrap();
    let _scratch = [0.];
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        2e-6
    } else {
        2e-14
    };
    assert!(
        (model.pdf(&[s(0.)]).unwrap().into_f64() - 1. / (2. * std::f64::consts::PI).sqrt()).abs()
            < tolerance
    );
    assert!(
        (model.log_pdf(&[s(20.)]).unwrap().into_f64()
            - (-200. - 0.5 * (2. * std::f64::consts::PI).ln()))
        .abs()
            < tolerance * 100.
    );
    assert_eq!(
        model
            .integrate_box(&[S::NEG_INFINITY], &[S::INFINITY])
            .unwrap(),
        s(1.)
    );
    assert!(
        (model.integrate_box(&[s(-1.)], &[s(1.)]).unwrap().into_f64() - 0.6826894921370859).abs()
            < tolerance
    );
    for (low, high) in [
        (S::NEG_INFINITY, S::ZERO),
        (S::ZERO, S::INFINITY),
        (-S::MAX, S::ZERO),
        (S::ZERO, S::MAX),
    ] {
        assert!((model.integrate_box(&[low], &[high]).unwrap().into_f64() - 0.5).abs() < tolerance);
    }
    // Finite bounds can overflow binary64 when standardized by a narrow kernel.
    let narrow_sd = [s::<S>(0.5)];
    let narrow = GaussianKde::build(
        view,
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&narrow_sd)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (low, high) in [(-S::MAX, S::ZERO), (S::ZERO, S::MAX)] {
        assert!(
            (narrow.integrate_box(&[low], &[high]).unwrap().into_f64() - 0.5).abs() < tolerance
        );
    }
    assert!(
        (model.integrate_kde(&model).unwrap().into_f64() - 1. / (2. * std::f64::consts::PI.sqrt()))
            .abs()
            < tolerance
    );
    let support = [scioxide_core::Interval::new(s(-1.), s(1.)).unwrap()];
    let bounded = GaussianKde::build(
        view,
        0,
        options.with_support(Some(&support)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(
        (bounded
            .integrate_box(&[s(-1.)], &[s(1.)])
            .unwrap()
            .into_f64()
            - 1.)
            .abs()
            < tolerance
    );
    assert_eq!(bounded.pdf(&[s(2.)]).unwrap(), s(0.));
    assert_eq!(bounded.log_pdf(&[s(2.)]).unwrap(), S::NEG_INFINITY);
    assert!((bounded.pdf(&[s(0.)]).unwrap().into_f64() - 0.5843685672568166).abs() < tolerance);
}
#[test]
fn analytic_gaussian_both_precisions() {
    kde_checks::<f32>();
    kde_checks::<f64>();
}

#[test]
fn fitted_covariance_and_frequency_replication() {
    let x = [0.0_f64, 1., 3.];
    let counts = [1, 3, 2];
    let expanded = [0.0_f64, 1., 1., 1., 3., 3.];
    let view = BatchView::new(&x, StridedLayout::new(&[3], &[1]).unwrap(), Axis::Index(0))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&counts))
        .unwrap();
    let expanded = BatchView::new(
        &expanded,
        StridedLayout::new(&[6], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    for bandwidth in [
        KdeBandwidth::Scott,
        KdeBandwidth::Silverman,
        KdeBandwidth::Factor(0.7),
    ] {
        let options = KdeOptions::default().with_bandwidth(bandwidth);
        let a = GaussianKde::build(view, 0, options, ExecutionContext::sequential()).unwrap();
        let b = GaussianKde::build(expanded, 0, options, ExecutionContext::sequential()).unwrap();
        for x in [-2., 0., 1., 2., 5.] {
            assert!((a.pdf(&[x]).unwrap() - b.pdf(&[x]).unwrap()).abs() < 2e-14);
        }
        assert_eq!(a.effective_size(), 6.);
    }
    let mean = [0., 0.];
    let cov = [1., 0.5, 0.5, 1.];
    let shape = [1, 2];
    let strides = [2, 1];
    let view = BatchView::new(
        &mean,
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(0),
    )
    .and_then(|view| view.with_coordinates(Axis::Index(1)))
    .unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::Covariance(&cov)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(
        (kde.pdf(&mean).unwrap() - 1. / (2. * std::f64::consts::PI * 0.75f64.sqrt())).abs() < 2e-14
    );
    assert!(
        (kde.integrate_gaussian(&mean, &cov).unwrap()
            - 1. / (4. * std::f64::consts::PI * 0.75f64.sqrt()))
        .abs()
            < 2e-14
    );
    assert!(kde.integrate_box(&[-1.; 2], &[1.; 2]).is_err());
}
#[test]
fn cdf_all_conventions_and_prepared_replication() {
    let x = [0.0_f64, 1., 2., 3.];
    let d = EmpiricalDistribution::new(&x, Unweighted).unwrap();
    let cases = [
        (QuantileMethod::NearestRank, 0.5),
        (QuantileMethod::AveragedInverseCdf, 0.375),
        (QuantileMethod::ClosestObservation, 0.625),
        (QuantileMethod::InterpolatedInverseCdf, 0.5625),
        (QuantileMethod::Hazen, 0.4375),
        (QuantileMethod::Weibull, 0.45),
        (QuantileMethod::Linear, 1.25 / 3.),
        (
            QuantileMethod::MedianUnbiased,
            1.9166666666666667 / 4.333333333333333,
        ),
        (QuantileMethod::NormalUnbiased, 1.875 / 4.25),
    ];
    for (method, expected) in cases {
        assert!(
            (d.estimated_cdf(1.25, QuantileRule::new(method)).unwrap() - expected).abs() < 2e-15,
            "{method:?}"
        );
    }
    let counts = [1, 2, 3, 4];
    let expanded = [0.0_f64, 1., 1., 2., 2., 2., 3., 3., 3., 3.];
    let distribution = EmpiricalDistribution::new(&expanded, Unweighted).unwrap();
    let view = BatchView::new(&x, StridedLayout::new(&[4], &[1]).unwrap(), Axis::Index(0))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&counts))
        .unwrap();
    let queries = [-1., 0., 1.25, 3., 4.];
    let mut w = CdfWorkspace::default();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    w.try_reserve(
        &(CdfWorkspace::workspace_requirements(view, queries.len(), options, context).unwrap()),
    )
    .unwrap();
    let mut out = [0.; 5];
    for (method, _) in cases {
        cdf_into(
            view,
            &queries,
            CdfMethod::Estimated(QuantileRule::new(method)),
            options,
            &mut out,
            &mut w,
            context,
        )
        .unwrap();
        for (q, y) in queries.iter().zip(out) {
            assert!(
                (y - distribution
                    .estimated_cdf(*q, QuantileRule::new(method))
                    .unwrap())
                .abs()
                    < 2e-15
            );
        }
    }
}

fn ordered_cdf_queries<S: RealScalar>() {
    let values = [s::<S>(3.), s(-0.), s(1.), s(1.), s(0.), s(7.)];
    let counts = [2, 1, 0, 3, 2, 1];
    let queries = [
        S::INFINITY,
        s(1.),
        s::<S>(1.).next_after(s(0.)),
        s(-0.),
        s(3.),
        s::<S>(1.).next_after(s(2.)),
        -S::INFINITY,
        s(1.),
        s(7.),
        s(-1.),
    ];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[6], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = CdfWorkspace::default();
    workspace
        .try_reserve(
            &(CdfWorkspace::workspace_requirements(view, queries.len(), options, context).unwrap()),
        )
        .unwrap();
    let conventions = [
        QuantileMethod::NearestRank,
        QuantileMethod::AveragedInverseCdf,
        QuantileMethod::ClosestObservation,
        QuantileMethod::InterpolatedInverseCdf,
        QuantileMethod::Hazen,
        QuantileMethod::Weibull,
        QuantileMethod::Linear,
        QuantileMethod::MedianUnbiased,
        QuantileMethod::NormalUnbiased,
    ];
    let mut output = [S::ZERO; 10];
    macro_rules! check_replication {
        ($input:expr, $weighted:expr) => {{
            let input = $input;
            let weighted = $weighted;
            let expanded: Vec<S> = values
                .iter()
                .zip(counts)
                .flat_map(|(&value, count)| {
                    core::iter::repeat_n(value, if weighted { count as usize } else { 1 })
                })
                .collect();
            let reference = EmpiricalDistribution::new(&expanded, Unweighted).unwrap();
            for method in [CdfMethod::Empirical, CdfMethod::Survival]
                .into_iter()
                .chain(conventions.map(|method| CdfMethod::Estimated(QuantileRule::new(method))))
            {
                cdf_into(
                    input,
                    &queries,
                    method,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (&query, &actual) in queries.iter().zip(&output) {
                    let expected = match method {
                        CdfMethod::Empirical => reference.cdf(query),
                        CdfMethod::Survival => reference.survival(query),
                        CdfMethod::Estimated(rule) => {
                            reference.estimated_cdf(query, QuantileRule::new(rule.method()))
                        }
                    }
                    .unwrap();
                    assert!((actual - expected).abs() <= s::<S>(4.) * S::EPSILON);
                }
            }
        }};
    }
    check_replication!(view, false);
    check_replication!(
        view.with_weights(Weights::<_, Frequency>::new(&counts))
            .unwrap(),
        true
    );
    for weights in [
        [s::<S>(2.), s(1.), s(0.), s(3.), s(2.), s(1.)],
        [s(20.), s(10.), s(0.), s(30.), s(20.), s(10.)],
    ] {
        let pairs: Vec<_> = values.iter().copied().zip(weights).collect();
        let reference = {
            let pairs = &pairs;
            let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
            let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
            EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
        }
        .unwrap();
        let input = view
            .with_weights(Weights::<_, Mass>::new(&weights))
            .unwrap();
        for method in [
            CdfMethod::Survival,
            CdfMethod::Empirical,
            CdfMethod::Estimated(QuantileRule::inverse_ecdf()),
        ] {
            cdf_into(
                input,
                &queries,
                method,
                options,
                &mut output,
                &mut workspace,
                context,
            )
            .unwrap();
            for (&query, &actual) in queries.iter().zip(&output) {
                let expected = if method == CdfMethod::Survival {
                    reference.survival(query)
                } else {
                    reference.cdf(query)
                }
                .unwrap();
                assert!((actual - expected).abs() <= s::<S>(4.) * S::EPSILON);
            }
        }
    }
}

#[test]
fn ordered_cdf_queries_preserve_both_precisions_and_conventions() {
    ordered_cdf_queries::<f32>();
    ordered_cdf_queries::<f64>();
}

fn cdf_small_tails<S: RealScalar>() {
    let values = [S::ZERO, S::ONE, S::TWO];
    let weights = [S::MAX, S::MAX, S::ONE];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[3], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_weights(Weights::<_, Mass>::new(&weights))
    .unwrap();
    let actual = cdf_with_options(
        view,
        &[S::ONE],
        CdfMethod::Survival,
        StatisticsOptions::new(),
    )
    .unwrap()[0];
    let expected = {
        let pairs = &[(S::ZERO, S::MAX), (S::ONE, S::MAX), (S::TWO, S::ONE)];
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap()
    .survival(S::ONE)
    .unwrap();
    assert!(actual > S::ZERO && actual < S::MIN_POSITIVE);
    assert!((actual - expected).abs() <= S::MIN_SUBNORMAL);
    // A one-point support still follows every estimated convention at its knot.
    let singleton = BatchView::new(
        &values,
        StridedLayout::new(&[1], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let reference = EmpiricalDistribution::new(&values[..1], Unweighted).unwrap();
    for method in [
        QuantileMethod::NearestRank,
        QuantileMethod::AveragedInverseCdf,
        QuantileMethod::ClosestObservation,
        QuantileMethod::InterpolatedInverseCdf,
        QuantileMethod::Hazen,
        QuantileMethod::Weibull,
        QuantileMethod::Linear,
        QuantileMethod::MedianUnbiased,
        QuantileMethod::NormalUnbiased,
    ] {
        let output = cdf_with_options(
            singleton,
            &[-S::ONE, S::ZERO, S::ONE],
            CdfMethod::Estimated(QuantileRule::new(method)),
            StatisticsOptions::new(),
        )
        .unwrap();
        for (actual, query) in output.into_iter().zip([-S::ONE, S::ZERO, S::ONE]) {
            assert!(
                (actual
                    - reference
                        .estimated_cdf(query, QuantileRule::new(method))
                        .unwrap())
                .abs()
                    <= S::EPSILON
            );
        }
    }
}

#[test]
fn cdf_tiny_tails_and_singletons_in_both_precisions() {
    cdf_small_tails::<f32>();
    cdf_small_tails::<f64>();
}

#[test]
fn prepared_cdf_extreme_masses_and_direct_tails() {
    let values = [0.0_f64, 1., 2., 3.];
    let weights = [f64::MAX, f64::MAX, 1., 0.];
    let pairs: Vec<_> = values.iter().copied().zip(weights).collect();
    let reference = {
        let pairs = &pairs;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[4], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_weights(Weights::<_, Mass>::new(&weights))
    .unwrap();
    let queries = [1., 2., -1., 0., f64::from_bits(1f64.to_bits() + 1), 1.];
    for method in [CdfMethod::Empirical, CdfMethod::Survival] {
        let output = cdf_with_options(view, &queries, method, StatisticsOptions::new()).unwrap();
        for (&query, &actual) in queries.iter().zip(&output) {
            let expected = if method == CdfMethod::Survival {
                reference.survival(query)
            } else {
                reference.cdf(query)
            }
            .unwrap();
            assert!((actual - expected).abs() <= f64::EPSILON * expected.abs());
        }
        if method == CdfMethod::Survival {
            assert!(output[0] > 0.);
            assert!(output[0] < f64::MIN_POSITIVE);
        }
    }
    let counts = [(usize::MAX - 1) as u64, 1, 0, 0];
    let output = cdf_with_options(
        view.with_weights(Weights::<_, Frequency>::new(&counts))
            .unwrap(),
        &[0.],
        CdfMethod::Survival,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!((output[0] * usize::MAX as f64 - 1.).abs() <= f64::EPSILON);
}
#[test]
fn narrow_support_and_range_scaling() {
    let x = [0.0_f64];
    let sd = [1.];
    let support = [scioxide_core::Interval::new(-1e-12, 1e-12).unwrap()];
    let view = BatchView::new(&x, StridedLayout::new(&[1], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default()
            .with_bandwidth(KdeBandwidth::StandardDeviations(&sd))
            .with_support(Some(&support)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((kde.pdf(&[0.]).unwrap() / 5e11 - 1.).abs() < 1e-14);
    assert!((kde.integrate_box(&[-1e-12], &[0.]).unwrap() - 0.5).abs() < 2e-15);
    let x = [-f64::MAX / 4., f64::MAX / 4.];
    let view = BatchView::new(&x, StridedLayout::new(&[2], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(kde.pdf(&[0.]).unwrap() > 0.);
}

#[test]
fn exact_neighbor_edges_and_invalid_models() {
    let plan = BinPlan::new(&[&[0.0_f64, 1., 2.]]).unwrap();
    assert_eq!(
        plan.locate(&[f64::from_bits(1f64.to_bits() - 1)]).unwrap(),
        BinLocation::Inside(0)
    );
    assert_eq!(
        plan.locate(&[f64::from_bits(1f64.to_bits() + 1)]).unwrap(),
        BinLocation::Inside(1)
    );
    assert_eq!(
        plan.locate(&[f64::from_bits(2f64.to_bits() + 1)]).unwrap(),
        BinLocation::Overflow(0)
    );
    assert!(BinPlan::new(&[&[0., 0.]]).is_err());
    assert!(BinPlan::uniform(&[(1., f64::from_bits(1f64.to_bits() + 1), 2)]).is_err());
    let x = [1.0, 1.];
    let view = BatchView::new(&x, StridedLayout::new(&[2], &[1]).unwrap(), Axis::Index(0)).unwrap();
    assert!(
        GaussianKde::build(
            view,
            0,
            KdeOptions::default(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        GaussianKde::build(
            view,
            0,
            KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&[0.])),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    let weights = [-1., 1.];
    let bad = view
        .with_weights(Weights::<_, Mass>::new(&weights))
        .unwrap();
    assert!(cdf_with_options(bad, &[1.], CdfMethod::Empirical, StatisticsOptions::new()).is_err());
    let values = [f64::NAN, 1.];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let options = StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip);
    assert_eq!(
        cdf_with_options(view, &[1.], CdfMethod::Empirical, options).unwrap(),
        [1.]
    );
}
#[test]
fn bounded_product_and_weight_rescaling() {
    let centers = [0.0_f64, 0., 1., 1.];
    let weights = [1., 3.];
    let scaled = [1e200, 3e200];
    let view = BatchView::new(
        &centers,
        StridedLayout::new(&[2, 2], &[2, 1]).unwrap(),
        Axis::Index(0),
    )
    .and_then(|view| view.with_coordinates(Axis::Index(1)))
    .unwrap();
    let support = [scioxide_core::Interval::new(-1., 2.).unwrap(); 2];
    let sd = [0.5, 1.];
    let options = KdeOptions::default()
        .with_bandwidth(KdeBandwidth::StandardDeviations(&sd))
        .with_support(Some(&support));
    let a = GaussianKde::build(
        view.with_weights(Weights::<_, Mass>::new(&weights))
            .unwrap(),
        0,
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let b = GaussianKde::build(
        view.with_weights(Weights::<_, Mass>::new(&scaled)).unwrap(),
        0,
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((a.integrate_box(&[-1.; 2], &[2.; 2]).unwrap() - 1.).abs() < 2e-14);
    assert!((a.pdf(&[0.5; 2]).unwrap() - b.pdf(&[0.5; 2]).unwrap()).abs() < 2e-14);
}

#[test]
fn bandwidth_uses_independent_rational_sample_covariance() {
    let x = [0.0_f64, 1., 3.];
    let counts = [1, 3, 2];
    let weights = [1., 3., 2.];
    let view = BatchView::new(&x, StridedLayout::new(&[3], &[1]).unwrap(), Axis::Index(0)).unwrap();
    macro_rules! check_model {
        ($weighted:expr, $frequency:expr) => {{
            let weighted = $weighted;
            let frequency = $frequency;
            let neff: f64 = if frequency { 6. } else { 18. / 7. };
            let variance: f64 = if frequency { 1.5 } else { 45. / 22. };
            for bandwidth in [
                KdeBandwidth::Scott,
                KdeBandwidth::Silverman,
                KdeBandwidth::Factor(0.7),
            ] {
                let factor = match bandwidth {
                    KdeBandwidth::Scott => neff.powf(-0.2),
                    KdeBandwidth::Silverman => (neff * 0.75).powf(-0.2),
                    _ => 0.7,
                };
                let sd = variance.sqrt() * factor;
                let expected = x
                    .iter()
                    .zip(weights)
                    .map(|(x, w)| {
                        let z = (0.5 - x) / sd;
                        w / 6. * (-0.5 * z * z).exp() / (sd * (2. * std::f64::consts::PI).sqrt())
                    })
                    .sum::<f64>();
                let kde = GaussianKde::build(
                    weighted,
                    0,
                    KdeOptions::default().with_bandwidth(bandwidth),
                    ExecutionContext::sequential(),
                )
                .unwrap();
                assert!((kde.effective_size() - neff).abs() < 2e-15);
                assert!((kde.pdf(&[0.5]).unwrap() - expected).abs() < 2e-14);
            }
        }};
    }
    check_model!(
        view.with_weights(Weights::<_, Frequency>::new(&counts))
            .unwrap(),
        true
    );
    check_model!(
        view.with_weights(Weights::<_, Mass>::new(&weights))
            .unwrap(),
        false
    );
}

#[test]
fn truncated_density_survives_unrepresentable_normalization_probability() {
    let x = [0.0_f64];
    let sd = [1e300];
    let support = [scioxide_core::Interval::new(-1e-300, 1e-300).unwrap()];
    let view = BatchView::new(&x, StridedLayout::new(&[1], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default()
            .with_bandwidth(KdeBandwidth::StandardDeviations(&sd))
            .with_support(Some(&support)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((kde.pdf(&[0.]).unwrap() / 5e299 - 1.).abs() < 3e-13);
    let half_mass = kde.integrate_box(&[-1e-300], &[0.]).unwrap();
    assert!((half_mass - 0.5).abs() < 1e-13, "{half_mass:e}");
    assert_eq!(kde.pdf(&[1e-300]).unwrap(), kde.pdf(&[0.]).unwrap());
    assert!((kde.integrate_box(&[-1e-300], &[1e-300]).unwrap() - 1.).abs() < 1e-13);
}

#[test]
fn negligible_gaussian_distance_retains_peak_density() {
    let x = [0.0_f64];
    let view = BatchView::new(&x, StridedLayout::new(&[1], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&[1.])),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(kde.pdf(&[1e-200]).unwrap(), kde.pdf(&[0.]).unwrap());
}

#[test]
fn gaussian_log_density_avoids_unnecessary_squared_distance_overflow() {
    let centers = [0.0_f64];
    let view = BatchView::new(
        &centers,
        StridedLayout::new(&[1], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let kde = GaussianKde::build(
        view,
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&[1.])),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let x = 1.5e154;
    let expected = -(x * 0.5) * x;
    let actual = kde.log_pdf(&[x]).unwrap();
    assert!((actual / expected - 1.).abs() < 2e-15);
    assert_eq!(kde.pdf(&[x]).unwrap(), 0.);
}

fn tiled_kde_mixtures<S: RealScalar>() {
    let centers = [s::<S>(-1.), s(0.5), s(2.), s(-1.), s(0.25), s(1.5)];
    let masses = [s::<S>(1.), s(2.), s(3.)];
    let counts = [1, 2, 3];
    let view = BatchView::new(
        &centers,
        StridedLayout::new(&[3, 2], &[2, 1]).unwrap(),
        Axis::Index(0),
    )
    .and_then(|view| view.with_coordinates(Axis::Index(1)))
    .unwrap();
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        4e-6
    } else {
        1e-13
    };
    for correlation in [0., 0.5] {
        let covariance = [s::<S>(1.), s(correlation), s(correlation), s(2.)];
        macro_rules! check_weights {
            ($input:expr) => {{
                let weights = $input;
                let model = GaussianKde::build(
                    view.with_weights(weights).unwrap(),
                    0,
                    KdeOptions::default().with_bandwidth(KdeBandwidth::Covariance(&covariance)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
                for n in [0, 1, 7, 8, 9, 17] {
                    // Coordinate-major, two interleaved logical batches. Include
                    // unordered/repeated points and exact triangular cancellation.
                    let points = [[0., 0.], [2., 1.], [-1., 2.], [0., 0.], [4., -3.]];
                    let values: Vec<S> = (0..2)
                        .flat_map(|batch| {
                            (0..2).flat_map(move |coordinate| {
                                (0..n)
                                    .map(move |i| s(points[(i + batch) % points.len()][coordinate]))
                            })
                        })
                        .collect();
                    let shape = [2, 2, n];
                    let strides = [2 * n, n, 1];
                    let queries = BatchView::new(
                        &values,
                        StridedLayout::new(&shape, &strides).unwrap(),
                        Axis::Index(2),
                    )
                    .and_then(|view| view.with_coordinates(Axis::Index(1)))
                    .unwrap();
                    let options = StatisticsOptions::new();
                    let context = ExecutionContext::sequential();
                    let mut workspace = KdeWorkspace::default();
                    workspace
                        .try_reserve(
                            &(KdeWorkspace::workspace_requirements(
                                &model, queries, options, context,
                            )
                            .unwrap()),
                        )
                        .unwrap();
                    let mut output = vec![S::NAN; 2 * n];
                    for log in [false, true] {
                        kde_into(
                            &model,
                            queries,
                            (if log {
                                scioxide_stats::KdeScale::LogDensity
                            } else {
                                scioxide_stats::KdeScale::Density
                            }),
                            options,
                            &mut output,
                            &mut workspace,
                            context,
                        )
                        .unwrap();
                        for (index, &actual) in output.iter().enumerate() {
                            let point = points[(index % n + index / n) % points.len()];
                            let determinant = 2. - correlation * correlation;
                            let pdf: f64 = centers
                                .chunks_exact(2)
                                .zip([1., 2., 3.])
                                .map(|(center, w)| {
                                    let x = point[0] - center[0].into_f64();
                                    let y = point[1] - center[1].into_f64();
                                    let q = (2. * x * x - 2. * correlation * x * y + y * y)
                                        / determinant;
                                    w / 6. * (-0.5 * q).exp()
                                        / (2. * core::f64::consts::PI * determinant.sqrt())
                                })
                                .sum();
                            let expected = if log { pdf.ln() } else { pdf };
                            assert!(
                                (actual.into_f64() - expected).abs()
                                    <= tolerance * expected.abs().max(1e-100)
                            );
                            let point = [s::<S>(point[0]), s(point[1])];
                            let scalar = if log {
                                model.log_pdf(&point)
                            } else {
                                model.pdf(&point)
                            }
                            .unwrap();
                            assert_eq!(actual, scalar);
                        }
                    }
                }
            }};
        }
        check_weights!(Weights::<_, Mass>::new(&masses));
        check_weights!(Weights::<_, Frequency>::new(&counts));
    }
}

#[test]
fn tiled_kde_matches_analytic_mixtures_in_both_precisions() {
    tiled_kde_mixtures::<f32>();
    tiled_kde_mixtures::<f64>();
}

fn offset_kde<S: RealScalar>() {
    let base = 2_f64.powi(if S::MANTISSA_DIGITS == 24 { 24 } else { 52 });
    let step = if S::MANTISSA_DIGITS == 24 { 2. } else { 1. };
    let centers = [s::<S>(base), s(base + 2. * step)];
    let sd = [s::<S>(step)];
    let model = GaussianKde::build(
        BatchView::new(
            &centers,
            StridedLayout::new(&[2], &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap(),
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&sd)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let expected = (-0.5_f64).exp() / (step * (2. * core::f64::consts::PI).sqrt());
    let actual = model.pdf(&[s(base + step)]).unwrap().into_f64();
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        2e-6
    } else {
        2e-14
    };
    assert!((actual / expected - 1.).abs() < tolerance);
    let support = [scioxide_core::Interval::new(s::<S>(-1.), s(1.)).unwrap(); 3];
    let center = [S::ZERO; 3];
    let sd = [S::ONE; 3];
    let bounded = GaussianKde::build(
        BatchView::new(
            &center,
            StridedLayout::new(&[1, 3], &[3, 1]).unwrap(),
            Axis::Index(0),
        )
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap(),
        0,
        KdeOptions::default()
            .with_bandwidth(KdeBandwidth::StandardDeviations(&sd))
            .with_support(Some(&support)),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let values: Vec<S> = (0..17)
        .flat_map(|i| [s(if i % 3 == 0 { 1.25 } else { 0.5 }), S::ZERO, S::ZERO])
        .collect();
    let queries = BatchView::new(
        &values,
        StridedLayout::new(&[17, 3], &[3, 1]).unwrap(),
        Axis::Index(0),
    )
    .and_then(|view| view.with_coordinates(Axis::Index(1)))
    .unwrap();
    let mut workspace = KdeWorkspace::default();
    workspace
        .try_reserve(
            &(KdeWorkspace::workspace_requirements(
                &bounded,
                queries,
                StatisticsOptions::new(),
                ExecutionContext::sequential(),
            )
            .unwrap()),
        )
        .unwrap();
    for log in [false, true] {
        let mut output = [S::NAN; 17];
        kde_into(
            &bounded,
            queries,
            if log {
                scioxide_stats::KdeScale::LogDensity
            } else {
                scioxide_stats::KdeScale::Density
            },
            StatisticsOptions::new(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let log_peak = -1.5 * (2. * core::f64::consts::PI).ln() - 3. * 0.6826894921370859_f64.ln();
        for (i, actual) in output.into_iter().enumerate() {
            if i % 3 == 0 {
                assert_eq!(actual, if log { S::NEG_INFINITY } else { S::ZERO });
            } else {
                let expected = if log {
                    log_peak - 0.125
                } else {
                    (log_peak - 0.125).exp()
                };
                assert!((actual.into_f64() - expected).abs() < tolerance);
            }
        }
    }
}

#[test]
fn kde_direct_differences_and_three_dimensional_truncation() {
    offset_kde::<f32>();
    offset_kde::<f64>();
}

#[test]
fn tiled_kde_preserves_range_rescued_densities_and_log_tails() {
    for (sd, support) in [
        (1e-300, None),
        (
            1e300,
            Some([scioxide_core::Interval::new(-1e-300, 1e-300).unwrap()]),
        ),
        (1., None),
    ] {
        let model = GaussianKde::build(
            BatchView::new(
                &[0.0_f64],
                StridedLayout::new(&[1], &[1]).unwrap(),
                Axis::Index(0),
            )
            .unwrap(),
            0,
            KdeOptions::default()
                .with_bandwidth(KdeBandwidth::StandardDeviations(&[sd]))
                .with_support(support.as_ref().map(|s| &s[..])),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let point = if support.is_some() {
            0.
        } else if sd == 1. {
            1.5e154
        } else {
            38. * sd
        };
        let values = [point; 17];
        let queries = BatchView::new(
            &values,
            StridedLayout::new(&[17], &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap();
        let mut workspace = KdeWorkspace::default();
        workspace
            .try_reserve(
                &(KdeWorkspace::workspace_requirements(
                    &model,
                    queries,
                    StatisticsOptions::new(),
                    ExecutionContext::sequential(),
                )
                .unwrap()),
            )
            .unwrap();
        let expected_log = if support.is_some() {
            (5e299_f64).ln()
        } else {
            let z = point / sd;
            -(0.5 * z) * z - sd.ln() - 0.5 * (2. * core::f64::consts::PI).ln()
        };
        for log in [false, true] {
            let mut output = [0.; 17];
            kde_into(
                &model,
                queries,
                if log {
                    scioxide_stats::KdeScale::LogDensity
                } else {
                    scioxide_stats::KdeScale::Density
                },
                StatisticsOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let expected = if log {
                expected_log
            } else {
                expected_log.exp()
            };
            for actual in output {
                assert!(
                    (actual - expected).abs() <= 5e-324 + 5e-13 * expected.abs(),
                    "{sd:e}/{log}: {actual:e} != {expected:e}"
                );
            }
        }
    }
}

#[test]
fn kde_normal_and_subnormal_tail_boundaries_and_extreme_weight_ratios() {
    let context = ExecutionContext::sequential();
    let options = StatisticsOptions::new();
    let model = GaussianKde::build(
        BatchView::new(
            &[0.0_f64],
            StridedLayout::new(&[1], &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap(),
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&[1.])),
        context,
    )
    .unwrap();
    let points = [37., 37.5, 37.6, 37.7, 38., 38.5, 40., 0., -37.6];
    let queries = BatchView::new(
        &points,
        StridedLayout::new(&[9], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let mut workspace = KdeWorkspace::default();
    workspace
        .try_reserve(
            &(KdeWorkspace::workspace_requirements(&model, queries, options, context).unwrap()),
        )
        .unwrap();
    for log in [false, true] {
        let mut output = [0.; 9];
        kde_into(
            &model,
            queries,
            if log {
                scioxide_stats::KdeScale::LogDensity
            } else {
                scioxide_stats::KdeScale::Density
            },
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        for (&actual, point) in output.iter().zip(points) {
            let expected_log = -0.5 * point * point - 0.5 * (2. * core::f64::consts::PI).ln();
            let expected = if log {
                expected_log
            } else {
                expected_log.exp()
            };
            let scalar = if log {
                model.log_pdf(&[point])
            } else {
                model.pdf(&[point])
            }
            .unwrap();
            assert!((actual - expected).abs() <= 5e-324 + 2e-13 * expected.abs());
            assert!((scalar - expected).abs() <= 5e-324 + 2e-13 * expected.abs());
        }
    }
    let model = GaussianKde::build(
        BatchView::new(
            &[0.0_f64, 1.],
            StridedLayout::new(&[2], &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap()
        .with_weights(Weights::<_, Mass>::new(&[1e-300, 1e300]))
        .unwrap(),
        0,
        KdeOptions::default().with_bandwidth(KdeBandwidth::StandardDeviations(&[1e-300])),
        context,
    )
    .unwrap();
    let points = [0., 1., 0., 1., 0., 1., 0., 1., 0.];
    let queries = BatchView::new(
        &points,
        StridedLayout::new(&[9], &[1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    for log in [false, true] {
        let mut output = [0.; 9];
        kde_into(
            &model,
            queries,
            if log {
                scioxide_stats::KdeScale::LogDensity
            } else {
                scioxide_stats::KdeScale::Density
            },
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        for (actual, point) in output.into_iter().zip(points) {
            let expected_log = if point == 0. {
                -300. * core::f64::consts::LN_10
            } else {
                300. * core::f64::consts::LN_10
            } - 0.5 * (2. * core::f64::consts::PI).ln();
            let expected = if log {
                expected_log
            } else {
                expected_log.exp()
            };
            assert!((actual / expected - 1.).abs() < 4e-13);
        }
    }
}

#[test]
fn three_dimensional_histogram_conserves_cube_mass() {
    fn cube<S: RealScalar>() {
        let points: Vec<S> = (0..8)
            .flat_map(|i| {
                [
                    s(((i >> 2) & 1) as f64),
                    s(((i >> 1) & 1) as f64),
                    s((i & 1) as f64),
                ]
            })
            .collect();
        let view = BatchView::new(
            &points,
            StridedLayout::new(&[8, 3], &[3, 1]).unwrap(),
            Axis::Index(0),
        )
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
        let plan = BinPlan::uniform(&[(s(0.), s(1.), 2); 3]).unwrap();
        let result = binned_statistics(
            &plan,
            view,
            None,
            BinnedStatistic::Density,
            StatisticsOptions::new(),
        )
        .unwrap();
        assert_eq!(result.bins.len(), 8);
        for bin in result.bins {
            assert_eq!(bin.count, 1);
            assert_eq!(bin.value, Some(s(1.)));
        }
        assert_eq!(result.diagnostics[0].retained_mass, Some(s(8.)));
    }
    cube::<f32>();
    cube::<f64>();
}

const BINNED_STATISTICS: [BinnedStatistic; 11] = [
    BinnedStatistic::Count,
    BinnedStatistic::Mass,
    BinnedStatistic::RelativeFrequency,
    BinnedStatistic::CumulativeFrequency,
    BinnedStatistic::Density,
    BinnedStatistic::Sum,
    BinnedStatistic::Mean,
    BinnedStatistic::StandardDeviation,
    BinnedStatistic::Minimum,
    BinnedStatistic::Maximum,
    BinnedStatistic::Median,
];

#[test]
fn direct_bins_match_independent_values_and_frequency_replication() {
    fn check<S: RealScalar>() {
        let edges = [s::<S>(0.), s(1.), s(2.), s(3.), s(4.)];
        let plan = BinPlan::new(&[&edges]).unwrap();
        let x = [-1., 0., 1., 0.5, 2., 1.5, 5., 2.5].map(s::<S>);
        let values = [99., 2., 4., 6., 8., 12., -99., 10.].map(s::<S>);
        let counts = [1u64, 1, 2, 3, 1, 1, 2, 2];
        let view =
            BatchView::new(&x, StridedLayout::new(&[8], &[1]).unwrap(), Axis::Index(0)).unwrap();
        let expanded_x: Vec<S> = x
            .iter()
            .zip(counts)
            .flat_map(|(&x, n)| core::iter::repeat_n(x, n as usize))
            .collect();
        let expanded_values: Vec<S> = values
            .iter()
            .zip(counts)
            .flat_map(|(&x, n)| core::iter::repeat_n(x, n as usize))
            .collect();
        let shape = [expanded_x.len()];
        let expanded = BatchView::new(
            &expanded_x,
            StridedLayout::new(&shape, &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap();
        let tolerance = 64. * S::EPSILON.into_f64();
        for statistic in BINNED_STATISTICS {
            let actual = binned_statistics(
                &plan,
                view,
                Some(&values),
                statistic,
                StatisticsOptions::new(),
            )
            .unwrap();
            let expected = match statistic {
                BinnedStatistic::Count | BinnedStatistic::Mass => [2., 2., 2.],
                BinnedStatistic::RelativeFrequency => [0.25; 3],
                BinnedStatistic::CumulativeFrequency => [2., 4., 6.],
                BinnedStatistic::Density => [1. / 3.; 3],
                BinnedStatistic::Sum => [8., 16., 18.],
                BinnedStatistic::Mean | BinnedStatistic::Median => [4., 8., 9.],
                BinnedStatistic::StandardDeviation => [2., 4., 1.],
                BinnedStatistic::Minimum => [2., 4., 8.],
                BinnedStatistic::Maximum => [6., 12., 10.],
            };
            for (bin, expected) in actual.bins.iter().zip(expected) {
                assert!(
                    (bin.value.unwrap().into_f64() - expected).abs()
                        <= tolerance * expected.abs().max(1.),
                    "{statistic:?}"
                );
                assert_eq!(bin.count, 2);
            }
            let empty = actual.bins[3].value;
            assert_eq!(
                empty,
                match statistic {
                    BinnedStatistic::Mean
                    | BinnedStatistic::StandardDeviation
                    | BinnedStatistic::Minimum
                    | BinnedStatistic::Maximum
                    | BinnedStatistic::Median => None,
                    BinnedStatistic::CumulativeFrequency => Some(s(6.)),
                    _ => Some(S::ZERO),
                }
            );
            assert_eq!(actual.diagnostics[0].underflow_mass, Some(S::ONE));
            assert_eq!(actual.diagnostics[0].overflow_mass, Some(S::ONE));
            let weighted = binned_statistics(
                &plan,
                view.with_weights(Weights::<_, Frequency>::new(&counts))
                    .unwrap(),
                Some(&values),
                statistic,
                StatisticsOptions::new(),
            )
            .unwrap();
            let replicated = binned_statistics(
                &plan,
                expanded,
                Some(&expanded_values),
                statistic,
                StatisticsOptions::new(),
            )
            .unwrap();
            for (a, b) in weighted.bins.iter().zip(replicated.bins) {
                assert_eq!(a.count, b.count);
                assert_eq!(a.mass, b.mass);
                match (a.value, b.value) {
                    (Some(a), Some(b)) => assert!(
                        (a.into_f64() - b.into_f64()).abs()
                            <= tolerance * b.into_f64().abs().max(1.),
                        "{statistic:?}"
                    ),
                    (a, b) => assert_eq!(a, b),
                }
            }
        }
        let weights = [1., 1., 2., 3., 1., 1., 2., 2.].map(s::<S>);
        let scaled_weights = weights.map(|w| w * s::<S>(8.));
        for statistic in [
            BinnedStatistic::Mean,
            BinnedStatistic::StandardDeviation,
            BinnedStatistic::RelativeFrequency,
            BinnedStatistic::Density,
            BinnedStatistic::Median,
        ] {
            let a = binned_statistics(
                &plan,
                view.with_weights(Weights::<_, Mass>::new(&weights))
                    .unwrap(),
                Some(&values),
                statistic,
                StatisticsOptions::new(),
            )
            .unwrap();
            let b = binned_statistics(
                &plan,
                view.with_weights(Weights::<_, Mass>::new(&scaled_weights))
                    .unwrap(),
                Some(&values),
                statistic,
                StatisticsOptions::new(),
            )
            .unwrap();
            for (a, b) in a.bins.iter().zip(b.bins) {
                assert_eq!(a.count, b.count);
                if let (Some(a), Some(b)) = (a.value, b.value) {
                    assert!(
                        (a.into_f64() - b.into_f64()).abs()
                            <= tolerance * b.into_f64().abs().max(1.)
                    );
                } else {
                    assert_eq!(a.value, b.value);
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn binned_means_preserve_cancellation_extreme_ranges_and_tiny_terms() {
    let plan = BinPlan::uniform(&[(0., 1., 1)]).unwrap();
    let x = [0.; 3];
    let view = BatchView::new(&x, StridedLayout::new(&[3], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let evaluate = |values: &[f64], weights: Option<&[f64]>, statistic| match weights {
        Some(weights) => binned_statistics(
            &plan,
            view.with_weights(Weights::<_, Mass>::new(weights)).unwrap(),
            Some(values),
            statistic,
            StatisticsOptions::new(),
        ),
        None => binned_statistics(
            &plan,
            view,
            Some(values),
            statistic,
            StatisticsOptions::new(),
        ),
    };
    for (values, expected) in [
        ([1e16, 1., -1e16], 1. / 3.),
        ([f64::MAX, 1., -f64::MAX], 1. / 3.),
        ([f64::MAX, f64::MAX, -f64::MAX], f64::MAX / 3.),
        ([1e16, 1e16 + 2., 1e16 + 4.], 1e16 + 2.),
        ([f64::from_bits(1); 3], f64::from_bits(1)),
    ] {
        let actual = evaluate(&values, None, BinnedStatistic::Mean).unwrap().bins[0]
            .value
            .unwrap();
        assert!(
            (actual - expected).abs() <= 8. * f64::EPSILON * expected.abs(),
            "{actual} != {expected}"
        );
    }
    let weights = [f64::MAX; 3];
    let actual = evaluate(&[2., 4., 6.], Some(&weights), BinnedStatistic::Mean).unwrap();
    assert_eq!(actual.bins[0].mass, None);
    assert_eq!(actual.diagnostics[0].admitted_mass, None);
    assert!((actual.bins[0].value.unwrap() - 4.).abs() <= 8. * f64::EPSILON);
    let actual = evaluate(
        &[0., 1., 0.],
        Some(&[1., f64::from_bits(1), 0.]),
        BinnedStatistic::Mean,
    )
    .unwrap();
    assert_eq!(actual.bins[0].value, Some(f64::from_bits(1)));
    assert_eq!(
        evaluate(&[f64::MAX; 3], None, BinnedStatistic::Minimum)
            .unwrap()
            .bins[0]
            .value,
        Some(f64::MAX)
    );
    assert!(evaluate(&[f64::MAX; 3], None, BinnedStatistic::Sum).is_err());
    let actual = evaluate(&[1e16, 1., -1e16], None, BinnedStatistic::Sum).unwrap();
    assert_eq!(actual.bins[0].value, Some(1.));
}

#[test]
fn cartesian_location_checks_all_axes_and_exact_neighbors() {
    let plan = BinPlan::new(&[&[0., 1., 2.], &[0., 1., 2.]]).unwrap();
    let below = f64::from_bits(1f64.to_bits() - 1);
    let above = f64::from_bits(1f64.to_bits() + 1);
    for (point, expected) in [
        ([below, 1.], BinLocation::Inside(1)),
        ([1., above], BinLocation::Inside(3)),
        ([2., 2.], BinLocation::Inside(3)),
        ([-1., 3.], BinLocation::Underflow(0)),
        ([3., -1.], BinLocation::Overflow(0)),
    ] {
        assert_eq!(plan.locate(&point).unwrap(), expected);
    }
    assert!(plan.locate(&[-1., f64::NAN]).is_err());
}

#[test]
fn bin_prediction_matches_edge_search_in_both_precisions() {
    fn check<S: RealScalar>(edges: &[S], points: &[S]) {
        let plan = BinPlan::new(&[edges]).unwrap();
        for &x in points {
            let expected = if x < edges[0] {
                BinLocation::Underflow(0)
            } else if x > edges[edges.len() - 1] {
                BinLocation::Overflow(0)
            } else {
                BinLocation::Inside(
                    edges
                        .partition_point(|edge| *edge <= x)
                        .saturating_sub(1)
                        .min(edges.len() - 2),
                )
            };
            assert_eq!(
                plan.locate(&[x]).unwrap(),
                expected,
                "x={x:?}, edges={edges:?}"
            );
        }
    }
    for edges in [
        vec![0., 0.1, 0.2, 0.3, 0.4, 0.5],
        vec![0., 1., 3., 100.],
        vec![-f64::MAX, 0., f64::MAX],
        vec![0., f64::from_bits(1), f64::from_bits(2), f64::from_bits(3)],
        // Predicted indices cross actual edges in this near-regular grid.
        vec![0., 1.4, 2., 3.],
    ] {
        let points: Vec<f64> = edges
            .iter()
            .flat_map(|&x| [x.next_down(), x, x.next_up()])
            .filter(|x| x.is_finite())
            .chain((0..1000).map(|i| i as f64 / 999. * edges[edges.len() - 1]))
            .filter(|x| x.is_finite())
            .collect();
        check(&edges, &points);
    }
    let edges: Vec<f32> = (0..=64).map(|i| i as f32 / 64.).collect();
    let points: Vec<f32> = edges
        .iter()
        .flat_map(|&x| [x.next_down(), x, x.next_up()])
        .collect();
    check(&edges, &points);
}
