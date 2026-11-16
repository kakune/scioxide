//! External coverage for every supported crate-root statistics contract.

#[path = "support/reductions.rs"]
mod reduction_test;
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

#[test]
fn every_supported_statistic_is_available_from_the_crate_root() -> Result<(), SciOxideError> {
    let confidence = ConfidenceProbability::new(0.95)?;
    assert_eq!(confidence.get(), 0.95);
    assert!(confidence.normal_critical_value::<f64>()?.is_finite());
    let _interval = scioxide_stats::MeanIntervalPlan::new(
        confidence,
        scioxide_stats::MeanIntervalMethod::NormalAsymptotic,
        None,
        Default::default(),
    )
    .and_then(|plan| plan.interval(2.0_f64, 0.5))?;

    let mut moments = OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    assert!(moments.is_empty());
    [1.0_f64, 2.0, 3.0]
        .into_iter()
        .try_for_each(|value| moments.observe(value))?;
    assert_eq!(moments.count(), 3);
    moments.merge(&OnlineMoments::new(scioxide_stats::MomentOrder::Second))?;
    let moments: MomentsSnapshot<f64> = moments.snapshot();
    assert_eq!(moments.count(), 3);
    assert_eq!(moments.mean().unwrap(), Some(2.0));
    assert!(moments.population_variance().unwrap().is_some());
    assert!(moments.sample_variance().unwrap().is_some());
    assert!(moments.variance_of_sample_mean().unwrap().is_some());
    assert!(moments.standard_error_of_sample_mean().unwrap().is_some());
    assert!(
        moments
            .confidence_interval_of_mean_with_options(
                confidence,
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default()
            )?
            .is_some()
    );

    let mut covariance = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    assert!(covariance.is_empty());
    [(1.0_f64, 2.0), (2.0, 4.0), (3.0, 6.0)]
        .into_iter()
        .try_for_each(|(x, y)| covariance.observe(x, y))?;
    assert_eq!(covariance.count(), 3);
    covariance.merge(&OnlineCovariance::new(scioxide_stats::MomentOrder::Second))?;
    let covariance: CovarianceSnapshot<f64> = covariance.snapshot();
    assert_eq!(covariance.count(), 3);
    assert_eq!(covariance.mean_x().unwrap(), Some(2.0));
    assert_eq!(covariance.mean_y().unwrap(), Some(4.0));
    assert!(covariance.population_covariance().unwrap().is_some());
    assert!(covariance.sample_covariance().unwrap().is_some());
    assert!(covariance.covariance_of_sample_means().unwrap().is_some());

    let mut weighted_moments = OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
        scioxide_stats::MomentOrder::Second,
    );
    assert!(weighted_moments.is_empty());
    [(1.0_f64, 1.0), (2.0, 2.0), (4.0, 1.0)]
        .into_iter()
        .try_for_each(|(value, weight)| weighted_moments.observe(value, weight))?;
    assert_eq!(weighted_moments.observation_count(), 3);
    weighted_moments.merge(
        &OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
            scioxide_stats::MomentOrder::Second,
        ),
    )?;
    let weighted_moments: WeightedMomentsSnapshot<f64, Reliability> = weighted_moments.snapshot();
    assert_eq!(weighted_moments.observation_count(), 3);
    assert_eq!(weighted_moments.total_weight()?, 4.0);
    assert!(weighted_moments.effective_sample_size().unwrap().is_some());
    assert!(weighted_moments.mean().unwrap().is_some());
    assert!(
        weighted_moments
            .weighted_population_variance()
            .unwrap()
            .is_some()
    );
    assert!(weighted_moments.sample_variance().unwrap().is_some());
    assert!(
        weighted_moments
            .variance_of_weighted_mean()
            .unwrap()
            .is_some()
    );
    assert!(
        weighted_moments
            .standard_error_of_weighted_mean()
            .unwrap()
            .is_some()
    );
    assert!(
        weighted_moments
            .confidence_interval_of_weighted_mean_with_options(
                confidence,
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default()
            )?
            .is_some()
    );

    let values = [1.0_f64, 2.0, 3.0, 4.0];
    assert_eq!(
        OnlineMoments::from_slice(&values, MomentOrder::Fourth)?.count(),
        4
    );
    assert_eq!(
        OnlineWeightedMoments::from_slices(
            &values,
            Weights::<_, Frequency>::new(&values),
            MomentOrder::Fourth
        )?
        .snapshot()
        .frequency_count(),
        Some(10)
    );
    let matrix = MatrixView::from_slice(2, 2, &values)?;
    let view_shape = [2, 2];
    let view_strides = [2, 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options_axis = Axis::Index(0);
    let options = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Reject)
        .with_non_finite_policy(NonFinitePolicy::Reject)
        .with_merge_policy(MergePolicy::InputOrder);
    let fourth = options.with_moment_order(MomentOrder::Fourth);
    let description = reduction_test::trimmed(
        view.with_axis(options_axis).unwrap(),
        TrimLimits::new(None, None)?,
        fourth,
        ExecutionContext::sequential(),
    )?[0];
    assert!(description.skewness(BiasCorrection::None)?.is_some());
    assert!(
        description
            .kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)?
            .is_some()
    );
    assert!(
        reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            8,
            Some(0.0),
            fourth,
            ExecutionContext::sequential()
        )?[0]
            .is_some()
    );
    let mut paired = OnlineWeightedCovariance::<_, Frequency>::new(MomentOrder::Fourth);
    paired.observe(1.0_f64, 2.0, 1.0)?;
    paired.observe(2.0, 4.0, 2.0)?;
    let paired: WeightedCovarianceSnapshot<f64, Frequency> = paired.snapshot();
    assert_eq!(paired.validity()?, CovarianceValidity::SampleDefined);
    assert_eq!(paired.correlation_validity(), CorrelationValidity::Defined);
    assert_eq!(
        reduction_test::weighted_covariance::<_, Frequency>(
            view.with_axis(options_axis).unwrap(),
            view.with_axis(options_axis).unwrap(),
            matrix,
            fourth,
            ExecutionContext::sequential()
        )?
        .len(),
        2
    );
    assert_eq!(
        reduction_test::moments(
            view.with_axis(options_axis).unwrap(),
            options,
            ExecutionContext::sequential()
        )?[0]
            .mean()
            .unwrap(),
        Some(2.0)
    );
    assert_eq!(
        reduction_test::covariance(
            view.with_axis(options_axis).unwrap(),
            view.with_axis(options_axis).unwrap(),
            options,
            ExecutionContext::sequential()
        )?[0]
            .count(),
        2
    );
    assert_eq!(
        reduction_test::weighted_moments::<_, Reliability>(
            view.with_axis(options_axis).unwrap(),
            matrix,
            options,
            ExecutionContext::sequential()
        )?[0]
            .observation_count(),
        2
    );

    let probability = QuantileProbability::new(0.5)?;
    assert_eq!(probability.get(), 0.5);
    let distribution = EmpiricalDistribution::new(&[4.0_f64, 1.0, 3.0, 2.0], Unweighted)?;
    assert_eq!(distribution.len(), 4);
    let quantile: QuantileResult<f64> =
        distribution.quantile(probability, QuantileRule::new(QuantileMethod::Linear))?;
    assert_eq!(quantile.value(), 2.5);
    assert_eq!(quantile.probability(), probability);
    assert_eq!(
        quantile.convention(),
        QuantileConvention::Unweighted(QuantileMethod::Linear)
    );
    let tail: TailMeanResult<f64> = distribution.tail_mean(probability, TailSide::Lower)?;
    assert_eq!(tail.mean(), 1.5);
    assert_eq!(tail.probability(), probability);
    assert_eq!(tail.side(), TailSide::Lower);
    assert_eq!(tail.convention(), TailMeanConvention::FractionalInverseEcdf);
    assert_eq!(tail.tail_mass(), Some(2.0));
    assert_eq!(tail.boundary_value(), 2.0);
    assert_eq!(tail.boundary_mass(), Some(1.0));
    assert_eq!(tail.boundary_contribution(), Some(2.0));

    let weighted_samples = [(4.0_f64, 1.0), (1.0, 1.0), (3.0, 1.0), (2.0, 1.0)];
    let weighted = {
        let pairs = &weighted_samples;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }?;
    assert_eq!(weighted.len(), 4);
    assert_eq!(weighted.total_weight(), Some(4.0));
    let weighted_quantile = weighted.quantile(probability, QuantileRule::inverse_ecdf())?;
    assert_eq!(weighted_quantile.value(), 2.0);
    assert_eq!(
        weighted_quantile.convention(),
        QuantileConvention::MassInverseEcdf
    );
    assert_eq!(
        weighted.tail_mean(probability, TailSide::Upper)?.mean(),
        3.5
    );

    let mut quantile_work = [4.0_f64, 1.0, 3.0, 2.0];
    assert_eq!(
        quantile_in_place(
            &mut quantile_work,
            probability,
            QuantileRule::new(QuantileMethod::NearestRank)
        )?
        .value(),
        2.0
    );
    let mut tail_work = [4.0_f64, 1.0, 3.0, 2.0];
    assert_eq!(
        tail_mean_in_place(&mut tail_work, probability, TailSide::Upper)?.mean(),
        3.5
    );
    let mut weighted_quantile_work = weighted_samples;
    assert_eq!(
        quantile_in_place(
            Weights::<_, Mass>::new(&mut weighted_quantile_work[..]),
            probability,
            QuantileRule::inverse_ecdf()
        )?
        .value(),
        2.0
    );
    let mut weighted_tail_work = weighted_samples;
    assert_eq!(
        tail_mean_in_place(
            Weights::<_, Mass>::new(&mut weighted_tail_work[..]),
            probability,
            TailSide::Lower
        )?
        .mean(),
        1.5
    );

    Ok(())
}
