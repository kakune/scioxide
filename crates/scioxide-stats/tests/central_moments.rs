//! Independent identities and public moment contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::SciOxideError;
use scioxide_stats::*;

/// Run a block once per sampling weight model, binding the model type and its label.
macro_rules! each_model {
    ($model:ident, $mode:ident => $body:block) => {{
        {
            type $model = Frequency;
            let $mode = WeightSemantics::Frequency;
            let _ = $mode;
            $body
        }
        {
            type $model = Reliability;
            let $mode = WeightSemantics::Reliability;
            let _ = $mode;
            $body
        }
        {
            type $model = Analytic;
            let $mode = WeightSemantics::Analytic;
            let _ = $mode;
            $body
        }
    }};
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}
#[test]
fn batch_center_retains_cancellation_before_normalizing_shifted_parts() {
    for order in [MomentOrder::Second, MomentOrder::Third, MomentOrder::Fourth] {
        let x = [1e16_f64, 1.0, -1e16];
        let plain = OnlineMoments::from_slice(&x, order).unwrap().snapshot();
        close(plain.mean().unwrap().unwrap(), 1.0 / 3.0);
        let weighted =
            OnlineWeightedMoments::from_slices(&x, Weights::<_, Frequency>::new(&[1.0; 3]), order)
                .unwrap()
                .snapshot();
        close(weighted.mean().unwrap().unwrap(), 1.0 / 3.0);
    }
}

#[test]
fn block_boundaries_match_independent_rational_centered_powers() {
    use scioxide_core::RealScalar;
    fn check<S: RealScalar>() {
        let (origin, step) = if S::MANTISSA_DIGITS == 24 {
            (16384.0, 1.0 / 64.0)
        } else {
            (1099511627776.0, 1.0 / 256.0)
        };
        for n in [1, 3, 4, 5, 255, 256, 257, 259, 513, 1025] {
            let integers = (0..n).map(|i| (i * 11 % 9) as i128 - 4).collect::<Vec<_>>();
            let values = integers
                .iter()
                .map(|&x| S::try_from_f64_exact(origin + step * x as f64).unwrap())
                .collect::<Vec<_>>();
            for weighted in [false, true] {
                let masses = (0..n)
                    .map(|i| {
                        if weighted {
                            if i % 7 == 6 { 0 } else { (i % 3 + 1) as i128 }
                        } else {
                            1
                        }
                    })
                    .collect::<Vec<_>>();
                let mass = masses.iter().sum::<i128>();
                let first = integers
                    .iter()
                    .zip(&masses)
                    .map(|(x, w)| x * w)
                    .sum::<i128>();
                let weights = masses
                    .iter()
                    .map(|&w| S::try_from_f64_exact(w as f64).unwrap())
                    .collect::<Vec<_>>();
                each_model!(M, _mode => {
                    let weighted_snapshot = OnlineWeightedMoments::from_slices(
                        &values,
                        Weights::<_, M>::new(&weights),
                        MomentOrder::Fourth,
                    )
                    .unwrap()
                    .snapshot();
                    let snapshot = OnlineMoments::from_slice(&values, MomentOrder::Fourth)
                        .unwrap()
                        .snapshot();
                    let mean = if weighted {
                        weighted_snapshot.mean()
                    } else {
                        snapshot.mean()
                    }
                    .unwrap()
                    .unwrap()
                    .into_f64();
                    assert!(
                        (mean - (origin + step * first as f64 / mass as f64)).abs()
                            <= S::EPSILON.into_f64() * origin
                    );
                    for order in 2..=4_u32 {
                        // Integer powers of n*x-sum(x), divided only once, are
                        // independent of the floating-point centering algorithm.
                        let numerator = integers
                            .iter()
                            .zip(&masses)
                            .map(|(x, w)| w * (mass * x - first).pow(order))
                            .sum::<i128>();
                        let expected = numerator as f64 / (mass as f64).powi(order as i32 + 1)
                            * step.powi(order as i32);
                        let actual = if weighted {
                            weighted_snapshot.central_moment(order as usize)
                        } else {
                            snapshot.central_moment(order as usize)
                        }
                        .unwrap()
                        .unwrap()
                        .into_f64();
                        let bound = 512.0 * S::EPSILON.into_f64() * (8.0 * step).powi(order as i32);
                        assert!(
                            (actual - expected).abs() <= bound,
                            "n={n}, order={order}, weighted={weighted}: {actual} != {expected}"
                        );
                    }
                });
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn late_block_failures_and_range_misses_replay_the_original_sample() {
    for index in [3, 255, 256, 258, 512] {
        let mut values = [1.0_f64; 513];
        values[index] = f64::NAN;
        let mut weights = [1.0; 513];
        weights[index] = 0.0;
        // Descriptive means omit zero mass; moment constructors still validate it.
        assert_eq!(
            scioxide_stats::statistic(
                &values,
                scioxide_stats::Weights::<_, Mass>::new(&weights),
                scioxide_stats::Statistic::mean()
            )
            .and_then(StatisticValue::scalar)
            .unwrap(),
            1.0
        );
        assert!(
            OnlineWeightedMoments::from_slices(
                &values,
                Weights::<_, Reliability>::new(&weights),
                MomentOrder::Fourth
            )
            .is_err()
        );
        for extreme in [
            f64::MAX,
            f64::from_bits(1),
            2.0_f64.powi(65),
            2.0_f64.powi(-65),
        ] {
            values[index] = extreme;
            let batch = OnlineMoments::from_slice(&values, MomentOrder::Fourth)
                .unwrap()
                .snapshot();
            let stream = sample(&values).snapshot();
            close(
                batch.mean().unwrap().unwrap(),
                stream.mean().unwrap().unwrap(),
            );
            close(
                batch.standard_deviation(0).unwrap().unwrap(),
                stream.standard_deviation(0).unwrap().unwrap(),
            );
            close(
                batch.skewness(BiasCorrection::None).unwrap().unwrap(),
                stream.skewness(BiasCorrection::None).unwrap().unwrap(),
            );
        }
    }
}

#[test]
fn block_extrema_keep_first_zero_signs_and_singleton_mean_signs() {
    for first in [-0.0_f64, 0.0] {
        let mut values = [first; 513];
        values[256] = -first;
        let batch = OnlineMoments::from_slice(&values, MomentOrder::Fourth)
            .unwrap()
            .snapshot();
        assert_eq!(batch.minimum().unwrap().to_bits(), first.to_bits());
        assert_eq!(batch.maximum().unwrap().to_bits(), first.to_bits());
        assert_eq!(batch.variance(0).unwrap().unwrap().to_bits(), 0);
        let singleton = OnlineMoments::from_slice(&values[..1], MomentOrder::Second)
            .unwrap()
            .snapshot();
        let streaming = sample(&values[..1]).snapshot();
        assert_eq!(
            singleton.mean().unwrap().unwrap().to_bits(),
            streaming.mean().unwrap().unwrap().to_bits()
        );
        let mut weights = [1.0; 513];
        weights[0] = 0.0;
        each_model!(M, semantics => {
            let weighted = OnlineWeightedMoments::from_slices(&values, Weights::<_, M>::new(&weights), MomentOrder::Fourth)
            .unwrap()
            .snapshot();
            assert_eq!(weighted.central_moment(2).unwrap().unwrap().to_bits(), 0);
        });
    }
}

fn sample(values: &[f64]) -> OnlineMoments<f64> {
    values
        .iter()
        .try_fold(OnlineMoments::new(MomentOrder::Fourth), |mut a, &x| {
            a.observe(x)?;
            Ok::<_, SciOxideError>(a)
        })
        .unwrap()
}

#[test]
fn exact_four_point_descriptive_statistics() {
    let s = sample(&[1.0, 2.0, 3.0, 4.0]).snapshot();
    assert_eq!(s.minimum(), Some(1.0));
    assert_eq!(s.maximum(), Some(4.0));
    for (order, expected) in [(0, 1.0), (1, 0.0), (2, 1.25), (3, 0.0), (4, 2.5625)] {
        close(s.central_moment(order).unwrap().unwrap(), expected);
    }
    close(s.skewness(BiasCorrection::Corrected).unwrap().unwrap(), 0.0);
    close(
        s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
            .unwrap()
            .unwrap(),
        1.64,
    );
    close(
        s.kurtosis(BiasCorrection::Corrected, KurtosisConvention::Fisher)
            .unwrap()
            .unwrap(),
        -1.2,
    );
    for (order, expected) in [(1, 2.5), (2, 5.0 / 3.0), (3, 0.0), (4, -10.0 / 3.0)] {
        close(s.k_statistic(order).unwrap().unwrap(), expected);
    }
    close(s.k_statistic_variance(1).unwrap().unwrap(), 5.0 / 12.0);
    close(s.k_statistic_variance(2).unwrap().unwrap(), 11.0 / 18.0);
    close(
        s.standard_deviation(1).unwrap().unwrap(),
        (5.0_f64 / 3.0).sqrt(),
    );
    close(
        s.standard_error(1).unwrap().unwrap(),
        (5.0_f64 / 12.0).sqrt(),
    );
    close(
        s.coefficient_of_variation(0).unwrap().unwrap(),
        1.25_f64.sqrt() / 2.5,
    );
}

#[test]
fn shifted_fourth_order_partition_merges_match_direct_centered_powers() {
    let values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let expected = sample(&values).snapshot();
    for split in 0..=values.len() {
        let mut a = sample(&values[..split]);
        a.merge(&sample(&values[split..])).unwrap();
        let s = a.snapshot();
        for order in 2..=4 {
            close(
                s.central_moment(order).unwrap().unwrap(),
                expected.central_moment(order).unwrap().unwrap(),
            );
        }
    }
}

#[test]
fn weight_models_have_distinct_degrees_and_mean_variance() {
    fn check<M: MomentWeights>(
        sample_variance: Option<f64>,
        mean_variance: f64,
        degrees: f64,
    ) -> WeightedMomentsSnapshot<f64, M> {
        let mut a = OnlineWeightedMoments::<f64, M>::new(MomentOrder::Fourth);
        a.observe(1.0_f64, 1.0).unwrap();
        a.observe(3.0, 3.0).unwrap();
        let s = a.snapshot();
        close(s.mean().unwrap().unwrap(), 2.5);
        assert_eq!(s.sample_variance().unwrap(), sample_variance);
        close(
            s.variance_of_weighted_mean().unwrap().unwrap(),
            mean_variance,
        );
        close(s.degrees_of_freedom().unwrap().unwrap(), degrees);
        s
    }
    let _ = check::<Frequency>(Some(1.0), 0.25, 3.0);
    let _ = check::<Reliability>(Some(2.0), 1.25, 0.6);
    let analytic = check::<Analytic>(None, 0.75, 1.0);
    close(analytic.noise_scale().unwrap().unwrap(), 3.0);
}

#[test]
fn frequency_moments_match_replication_and_zero_weights_are_identities() {
    let mut weighted = OnlineWeightedMoments::<_, Frequency>::new(MomentOrder::Fourth);
    for (x, w) in [(1.0, 1.0), (2.0, 2.0), (4.0, 3.0)] {
        weighted.observe(x, w).unwrap();
    }
    let before = weighted;
    weighted.observe(999.0, 0.0).unwrap();
    assert_eq!(before, weighted);
    assert!(weighted.observe(1.0, 0.5).is_err());
    assert_eq!(before, weighted);
    let a = weighted.snapshot();
    let b = sample(&[1.0, 2.0, 2.0, 4.0, 4.0, 4.0]).snapshot();
    for order in 2..=4 {
        close(
            a.central_moment(order).unwrap().unwrap(),
            b.central_moment(order).unwrap().unwrap(),
        );
        close(
            a.k_statistic(order).unwrap().unwrap(),
            b.k_statistic(order).unwrap().unwrap(),
        );
    }
    assert_eq!(a.frequency_count(), Some(6));
    assert_eq!(a.observation_count(), 3);
    for order in 1..=2 {
        close(
            a.k_statistic_variance(order).unwrap().unwrap(),
            b.k_statistic_variance(order).unwrap().unwrap(),
        );
    }
}

#[test]
fn incompatible_empty_merges_and_untracked_queries_are_errors() {
    let mut a = sample(&[1.0, 2.0]);
    let before = a;
    assert!(a.merge(&OnlineMoments::new(MomentOrder::Second)).is_err());
    assert_eq!(a, before);
    let mut weighted = OnlineWeightedMoments::<f64, Frequency>::new(MomentOrder::Fourth);
    assert!(
        weighted
            .merge(&OnlineWeightedMoments::new(MomentOrder::Second))
            .is_err()
    );
    assert!(
        OnlineMoments::<f64>::new(MomentOrder::Second)
            .snapshot()
            .central_moment(3)
            .is_err()
    );
    let constant = sample(&[2.0, 2.0, 2.0, 2.0]).snapshot();
    assert_eq!(constant.skewness(BiasCorrection::None).unwrap(), None);
    assert_eq!(
        sample(&[0.0, 0.0])
            .snapshot()
            .coefficient_of_variation(0)
            .unwrap(),
        None
    );
}

#[test]
fn covariance_marginals_and_correlation_are_independent_diagnostics() {
    let mut c = OnlineCovariance::new(MomentOrder::Fourth);
    for x in [1.0_f64, 2.0, 4.0, 8.0] {
        c.observe(x, -2.0 * x).unwrap();
    }
    let s = c.snapshot();
    assert_eq!(s.validity(), CovarianceValidity::SampleDefined);
    assert_eq!(s.correlation_validity(), CorrelationValidity::Defined);
    close(s.correlation().unwrap().unwrap(), -1.0);
    close(
        s.marginal_y().central_moment(4).unwrap().unwrap(),
        16.0 * s.marginal_x().central_moment(4).unwrap().unwrap(),
    );
}

#[test]
fn overflowing_fourth_moment_does_not_invalidate_variance_or_kurtosis() {
    let a = sample(&[-1e100, 1e100]).snapshot();
    assert!(a.central_moment(4).is_err());
    close(a.population_variance().unwrap().unwrap(), 1e200);
    close(
        a.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
            .unwrap()
            .unwrap(),
        1.0,
    );
}

#[test]
fn fourth_moments_across_binary_scale_boundaries_preserve_standardized_shape() {
    // Straddle native arithmetic bounds and cover subnormal source observations.
    for exponent in [-1020, -300, -65, -64, -63, 0, 63, 64, 65, 100, 500] {
        let scale = 2.0_f64.powi(exponent);
        let values = [-scale, scale, -scale, scale];
        let expected = sample(&values).snapshot();
        for split in 0..=values.len() {
            let mut a = sample(&values[..split]);
            a.merge(&sample(&values[split..])).unwrap();
            let s = a.snapshot();
            close(s.skewness(BiasCorrection::None).unwrap().unwrap(), 0.0);
            close(
                s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
                    .unwrap()
                    .unwrap(),
                1.0,
            );
            if let Ok(Some(moment)) = s.central_moment(4) {
                let reference = scale.powi(4);
                assert!(moment == reference || (moment / reference - 1.0).abs() < 1e-12);
            }
            close(
                s.population_variance().unwrap().unwrap(),
                expected.population_variance().unwrap().unwrap(),
            );
        }
    }
}

#[test]
fn unbiased_k_statistic_variance_is_not_clipped_to_zero() {
    let s = sample(&[-1.0, -1.0, 1.0, 1.0]).snapshot();
    close(s.k_statistic_variance(2).unwrap().unwrap(), -8.0 / 9.0);
}

#[test]
fn paired_slice_summaries_preserve_covariance_and_marginal_contracts() {
    fn check<S: scioxide_core::RealScalar>() {
        let scalar = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let x = [1.0, 3.0, 7.0, 9.0].map(scalar);
        let y = [9.0, 1.0, -1.0, 5.0].map(scalar);
        for order in [MomentOrder::Second, MomentOrder::Third, MomentOrder::Fourth] {
            let full = OnlineCovariance::from_slices(&x, &y, order).unwrap();
            let snapshot = full.snapshot();
            assert_eq!(
                snapshot.population_covariance().unwrap(),
                Some(scalar(-5.0))
            );
            assert_eq!(snapshot.marginal_x().mean().unwrap(), Some(scalar(5.0)));
            assert_eq!(snapshot.marginal_y().mean().unwrap(), Some(scalar(3.5)));
            for split in 0..=x.len() {
                let mut partial =
                    OnlineCovariance::from_slices(&x[..split], &y[..split], order).unwrap();
                partial
                    .merge(&OnlineCovariance::from_slices(&x[split..], &y[split..], order).unwrap())
                    .unwrap();
                partial.observe(scalar(11.0), scalar(2.0)).unwrap();
                let mut expected = full;
                expected.observe(scalar(11.0), scalar(2.0)).unwrap();
                let actual = partial.snapshot();
                let reference = expected.snapshot();
                assert!(
                    (actual.correlation().unwrap().unwrap()
                        - reference.correlation().unwrap().unwrap())
                    .abs()
                        <= scalar(128.0) * <S as scioxide_core::RealScalar>::EPSILON
                );
                for r in 2..=order.maximum() {
                    let a = actual.marginal_y().central_moment(r).unwrap().unwrap();
                    let b = reference.marginal_y().central_moment(r).unwrap().unwrap();
                    assert!(
                        (a - b).abs()
                            <= scalar(128.0) * <S as scioxide_core::RealScalar>::EPSILON * b.abs()
                    );
                }
            }
            let correlation = OnlineCovariance::from_slices(&x, &x, order)
                .unwrap()
                .snapshot()
                .correlation()
                .unwrap()
                .unwrap();
            assert!(
                (correlation - S::ONE).abs()
                    <= scalar(2.0) * <S as scioxide_core::RealScalar>::EPSILON
            );
            assert_eq!(
                OnlineCovariance::<S>::from_slices(&[], &[], order).unwrap(),
                OnlineCovariance::new(order)
            );
        }
    }
    check::<f32>();
    check::<f64>();
    let order = MomentOrder::Fourth;
    assert!(
        matches ! (& (OnlineCovariance :: from_slices (& [1.0_f64] , & [] , order)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: DimensionMismatch { .. }))
    );
    assert!(
        matches ! (& (OnlineCovariance :: from_slices (& [1.0] , & [f64 :: NAN] , order)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    let origin = 1e15_f64;
    let shifted = OnlineCovariance::from_slices(
        &[origin, origin + 0.25, origin + 0.25, origin + 0.25],
        &[0.0, 1.0, 1.0, 1.0],
        order,
    )
    .unwrap()
    .snapshot();
    assert_eq!(shifted.population_covariance().unwrap(), Some(3.0 / 64.0));
    close(shifted.correlation().unwrap().unwrap(), 1.0);
    let tiny = f64::from_bits(1);
    let subnormal =
        OnlineCovariance::from_slices(&[0.0, tiny, tiny, tiny], &[0.0, tiny, tiny, tiny], order)
            .unwrap()
            .snapshot();
    close(subnormal.correlation().unwrap().unwrap(), 1.0);
    let overflow = OnlineCovariance::from_slices(&[-2e19_f32, 2e19], &[-1e-19, 1e-19], order)
        .unwrap()
        .snapshot();
    assert!(overflow.marginal_x().population_variance().is_err());
    assert!((overflow.population_covariance().unwrap().unwrap() - 2.0).abs() < 1e-6);
    assert!((overflow.correlation().unwrap().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn slice_constructors_validate_inputs_and_remain_mergeable() {
    fn check<S: scioxide_core::RealScalar>() {
        let scalar = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let values = [1.0, 3.0, 7.0, 9.0].map(scalar);
        let weights = [1.0, 2.0, 3.0, 4.0].map(scalar);
        for order in [MomentOrder::Second, MomentOrder::Third, MomentOrder::Fourth] {
            each_model!(M, mode => {
                let all =
                    OnlineWeightedMoments::from_slices(&values, Weights::<_, M>::new(&weights), order).unwrap();
                assert_eq!(
                    OnlineWeightedMoments::from_slices(&values, Weights::<_, M>::new(&[S::ZERO; 4]), order)
                        .unwrap(),
                    OnlineWeightedMoments::<_, M>::new(order)
                );
                for split in 0..=values.len() {
                    let mut partial = OnlineWeightedMoments::from_slices(&values[..split], Weights::<_, M>::new(&weights[..split]), order)
                    .unwrap();
                    partial
                        .merge(
                            &OnlineWeightedMoments::from_slices(&values[split..], Weights::<_, M>::new(&weights[split..]), order)
                            .unwrap(),
                        )
                        .unwrap();
                    partial.observe(scalar(11.0), scalar(2.0)).unwrap();
                    let mut expected = all;
                    expected.observe(scalar(11.0), scalar(2.0)).unwrap();
                    for r in 2..=order.maximum() {
                        let actual = partial.snapshot().central_moment(r).unwrap().unwrap();
                        let reference = expected.snapshot().central_moment(r).unwrap().unwrap();
                        assert!(
                            ((actual - reference) / reference).abs()
                                <= scalar(128.0) * <S as scioxide_core::RealScalar>::EPSILON
                        );
                    }
                }
            });
        }
    }
    check::<f32>();
    check::<f64>();
    assert!(
        matches ! (& (OnlineWeightedMoments :: from_slices (& [1.0_f64] , Weights :: <_, Frequency> :: new (& []) , MomentOrder :: Fourth)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: DimensionMismatch { .. }))
    );
    assert!(
        matches ! (& (OnlineWeightedMoments :: from_slices (& [f64 :: NAN] , Weights :: <_, Frequency> :: new (& [0.0]) , MomentOrder :: Fourth)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    assert!(
        matches ! (& (OnlineWeightedMoments :: from_slices (& [1.0_f64] , Weights :: <_, Frequency> :: new (& [0.5]) , MomentOrder :: Fourth)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ScalarConversion { .. }))
    );
    if usize::BITS == 64 {
        assert!(
            matches ! (& (OnlineWeightedMoments :: from_slices (& [1.0_f64 , 2.0] , Weights :: <_, Frequency> :: new (& [2.0_f64 . powi (63) ; 2]) , MomentOrder :: Fourth)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
    }
    let extremes = OnlineMoments::from_slice(&[-1e100_f64, 1e100], MomentOrder::Fourth)
        .unwrap()
        .snapshot();
    assert!(extremes.central_moment(4).is_err());
    close(
        extremes
            .kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
            .unwrap()
            .unwrap(),
        1.0,
    );
    let tiny = OnlineMoments::from_slice(&[0.0_f64, f64::from_bits(2)], MomentOrder::Fourth)
        .unwrap()
        .snapshot();
    assert_eq!(
        tiny.standard_error_of_sample_mean().unwrap(),
        Some(f64::from_bits(1))
    );
}

#[test]
fn weighted_covariance_matches_exact_rational_references_and_all_partitions() {
    fn check<M: MomentWeights>(
        sample_covariance_with_options: Option<f64>,
        mean_covariance: f64,
    ) -> Vec<WeightedCovarianceSnapshot<f64, M>> {
        let values = [(1.0, 2.0, 1.0), (3.0, 9.0, 2.0), (7.0, 4.0, 3.0)];
        let build = |data: &[(f64, f64, f64)]| {
            data.iter()
                .try_fold(
                    OnlineWeightedCovariance::<f64, M>::new(MomentOrder::Fourth),
                    |mut a, &(x, y, w)| {
                        a.observe(x, y, w)?;
                        Ok::<_, SciOxideError>(a)
                    },
                )
                .unwrap()
        };
        (0..=values.len())
            .map(|split| {
                let mut a = build(&values[..split]);
                a.merge(&build(&values[split..])).unwrap();
                let s = a.snapshot();
                close(s.population_covariance().unwrap().unwrap(), -14.0 / 9.0);
                close(s.correlation().unwrap().unwrap(), -14.0 / 3445.0_f64.sqrt());
                close(
                    s.marginal_x()
                        .weighted_population_variance()
                        .unwrap()
                        .unwrap(),
                    53.0 / 9.0,
                );
                close(
                    s.marginal_y()
                        .weighted_population_variance()
                        .unwrap()
                        .unwrap(),
                    65.0 / 9.0,
                );
                close(
                    s.covariance_of_weighted_means().unwrap().unwrap(),
                    mean_covariance,
                );
                match sample_covariance_with_options {
                    Some(reference) => close(s.sample_covariance().unwrap().unwrap(), reference),
                    None => assert_eq!(s.sample_covariance().unwrap(), None),
                }
                s
            })
            .collect()
    }
    let _ = check::<Frequency>(Some(-28.0 / 15.0), -14.0 / 45.0);
    let _ = check::<Reliability>(Some(-28.0 / 11.0), -98.0 / 99.0);
    for s in check::<Analytic>(None, -7.0 / 9.0) {
        close(s.noise_covariance().unwrap().unwrap(), -14.0 / 3.0);
    }
}

#[test]
fn analytic_precision_rescaling_preserves_mean_inference_in_both_precisions() {
    fn check<S: scioxide_core::RealScalar>() {
        let scalar = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let build = |factor: f64| {
            [(1.0, 1.0), (3.0, 2.0), (7.0, 3.0)]
                .into_iter()
                .try_fold(
                    OnlineWeightedMoments::<_, Analytic>::new(MomentOrder::Fourth),
                    |mut a, (x, w)| {
                        a.observe(scalar(x), scalar(w * factor))?;
                        Ok::<_, SciOxideError>(a)
                    },
                )
                .unwrap()
                .snapshot()
        };
        let base = build(1.0);
        for factor in [2.0_f64.powi(-64), 2.0_f64.powi(64)] {
            let scaled = build(factor);
            for (a, b) in [
                (
                    scaled.central_moment(4).unwrap().unwrap(),
                    base.central_moment(4).unwrap().unwrap(),
                ),
                (
                    scaled.variance_of_weighted_mean().unwrap().unwrap(),
                    base.variance_of_weighted_mean().unwrap().unwrap(),
                ),
                (
                    scaled.noise_scale().unwrap().unwrap() / scalar(factor),
                    base.noise_scale().unwrap().unwrap(),
                ),
            ] {
                let relative = ((a - b) / b).abs();
                assert!(relative <= scalar(64.0) * <S as scioxide_core::RealScalar>::EPSILON);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn subnormal_observations_preserve_standardized_shape_and_variation() {
    fn check<S: scioxide_core::RealScalar>() {
        let unit = S::MIN_POSITIVE * <S as scioxide_core::RealScalar>::EPSILON;
        let values = [S::ZERO, unit, unit, unit];
        let mut streamed = OnlineMoments::new(MomentOrder::Fourth);
        let mut paired = OnlineCovariance::new(MomentOrder::Fourth);
        values
            .iter()
            .try_for_each(|&v| streamed.observe(v))
            .unwrap();
        for state in [
            streamed,
            OnlineMoments::from_slice(&values, MomentOrder::Fourth).unwrap(),
        ] {
            values
                .iter()
                .try_for_each(|&v| paired.observe(v, v + v))
                .unwrap();
            close(
                paired
                    .snapshot()
                    .correlation()
                    .unwrap()
                    .unwrap()
                    .try_to_f64_exact()
                    .unwrap(),
                1.0,
            );
            let s = state.snapshot();
            for (actual, reference) in [
                (
                    s.skewness(BiasCorrection::None).unwrap().unwrap(),
                    -2.0 / 3.0_f64.sqrt(),
                ),
                (
                    s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
                        .unwrap()
                        .unwrap(),
                    7.0 / 3.0,
                ),
                (
                    s.coefficient_of_variation(0).unwrap().unwrap(),
                    1.0 / 3.0_f64.sqrt(),
                ),
            ] {
                assert!(
                    (actual.try_to_f64_exact().unwrap() - reference).abs()
                        <= 128.0
                            * <S as scioxide_core::RealScalar>::EPSILON
                                .try_to_f64_exact()
                                .unwrap()
                );
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn tiny_partition_mass_preserves_its_representable_mean_contribution() {
    fn check<M: MomentWeights>() {
        for order in [MomentOrder::Second, MomentOrder::Fourth] {
            let values = [-1e100_f64, 0.0];
            let weights = [1e-200, 1.0];
            let mut streamed = OnlineWeightedMoments::<f64, M>::new(order);
            values
                .iter()
                .zip(&weights)
                .try_for_each(|(&x, &w)| streamed.observe(x, w))
                .unwrap();
            for state in [
                streamed,
                OnlineWeightedMoments::from_slices(&values, Weights::<_, M>::new(&weights), order)
                    .unwrap(),
            ] {
                let mean = state.snapshot().mean().unwrap().unwrap();
                assert!((mean / -1e-100 - 1.0).abs() < 128.0 * f64::EPSILON);
            }
        }
    }
    check::<Reliability>();
    check::<Analytic>();
}

#[test]
fn weighted_slices_with_an_outlying_first_observation_keep_the_exact_center() {
    let (mut values, mut weights) = (vec![1e15_f64], vec![1e-12_f64]);
    for i in 1..=1000_usize {
        values.push(((i * 7919) % 1009) as f64 / 1009.0 - 0.5);
        weights.push(0.1 + 0.001 * i as f64);
    }
    // Exact rational moments of the binary64 inputs, each rounded once. The
    // negligible first mass makes the mean well conditioned.
    let (mean, variance) = (1.665_307_321_533_341_2, 1_665_278_934_221_476.5);
    let check = |state: OnlineWeightedMoments<f64, Reliability>| {
        let snapshot = state.snapshot();
        let actual = snapshot.mean().unwrap().unwrap();
        assert!(
            (actual - mean).abs() <= 4.0 * f64::EPSILON * mean,
            "{actual}"
        );
        let actual = snapshot.weighted_population_variance().unwrap().unwrap();
        assert!((actual - variance).abs() <= 1e-13 * variance, "{actual}");
    };
    let order = MomentOrder::Second;
    let typed = |w| Weights::<_, Reliability>::new(w);
    check(OnlineWeightedMoments::from_slices(&values, typed(&weights[..]), order).unwrap());
    let mut streamed = OnlineWeightedMoments::new(order);
    for (&x, &w) in values.iter().zip(&weights) {
        streamed.observe(x, w).unwrap();
    }
    check(streamed);
    // The right partition of a merge starts with the outlier as its origin.
    let mut rotated_values = values[1..].to_vec();
    let mut rotated_weights = weights[1..].to_vec();
    rotated_values.insert(500, values[0]);
    rotated_weights.insert(500, weights[0]);
    let mut merged = OnlineWeightedMoments::from_slices(
        &rotated_values[..500],
        typed(&rotated_weights[..500]),
        order,
    )
    .unwrap();
    merged
        .merge(
            &OnlineWeightedMoments::from_slices(
                &rotated_values[500..],
                typed(&rotated_weights[500..]),
                order,
            )
            .unwrap(),
        )
        .unwrap();
    check(merged);
    check(
        OnlineWeightedMoments::from_slices(&rotated_values, typed(&rotated_weights[..]), order)
            .unwrap(),
    );
}

#[test]
fn replayed_and_streamed_moments_keep_two_part_center_contrasts() {
    fn paths(values: &[f64]) -> Vec<OnlineMoments<f64>> {
        let mut states = vec![OnlineMoments::from_slice(values, MomentOrder::Fourth).unwrap()];
        let mut streamed = OnlineMoments::new(MomentOrder::Fourth);
        for &x in values {
            streamed.observe(x).unwrap();
        }
        states.push(streamed);
        for split in 1..values.len() {
            let mut left =
                OnlineMoments::from_slice(&values[..split], MomentOrder::Fourth).unwrap();
            left.merge(&OnlineMoments::from_slice(&values[split..], MomentOrder::Fourth).unwrap())
                .unwrap();
            states.push(left);
        }
        states
    }
    // Exact cancellation of the large contrasts on both sides of the native
    // source frame, including the extended replay of `from_slice`. Far
    // outside it, a partition mean such as (4 - 2^200)/3 exceeds two-part
    // precision, so only the replay and the observation stream are exact.
    for exponent in [64, 65, 200] {
        let large = 2.0_f64.powi(exponent);
        let states = paths(&[large, 3.0, -large, 1.0]);
        let exact = if exponent > 65 { 2 } else { states.len() };
        for state in &states[..exact] {
            assert_eq!(state.snapshot().mean().unwrap(), Some(1.0), "2^{exponent}");
        }
    }
    // Exact rational means and population variances, each rounded once. The
    // compensated envelope is relative to the mean plus second order in the
    // absolute contrasts.
    for (exponent, variance) in [
        (64, 1.329_227_995_784_915_9e38),
        (65, 5.316_911_983_139_663_5e38),
    ] {
        let large = 2.0_f64.powi(exponent);
        let values = [
            large,
            3.0,
            -large,
            1.0,
            0.1,
            -0.75 * large,
            0.75 * large,
            0.2,
        ];
        let mass = values.iter().map(|x| (x - 0.5375).abs()).sum::<f64>();
        let envelope = 16.0 * (f64::EPSILON * 0.5375 + f64::EPSILON * f64::EPSILON * mass);
        for state in paths(&values) {
            let snapshot = state.snapshot();
            let mean = snapshot.mean().unwrap().unwrap();
            assert!((mean - 0.5375).abs() <= envelope, "2^{exponent}: {mean}");
            let actual = snapshot.population_variance().unwrap().unwrap();
            assert!(
                (actual - variance).abs() <= 1e-14 * variance,
                "2^{exponent}: {actual}"
            );
        }
    }
}

#[test]
fn adjacent_f64_observations_retain_the_half_ulp_center() {
    let base = 2.0_f64.powi(53);
    let values = [base, base + 2.0, base + 4.0];
    for split in 0..=values.len() {
        let mut a = sample(&values[..split]);
        a.merge(&sample(&values[split..])).unwrap();
        let s = a.snapshot();
        assert!((s.sample_variance().unwrap().unwrap() - 4.0).abs() < 32.0 * f64::EPSILON);
        assert!(s.central_moment(3).unwrap().unwrap().abs() < 128.0 * f64::EPSILON);
        assert!((s.central_moment(4).unwrap().unwrap() - 32.0 / 3.0).abs() < 256.0 * f64::EPSILON);
    }
}

#[test]
fn diagnostic_overflow_is_independent_of_snapshot_and_other_queries() {
    fn check<S: scioxide_core::RealScalar>(scale: S) {
        let order = MomentOrder::Fourth;
        let values = [-scale, scale];
        let state = OnlineMoments::from_slice(&values, order).unwrap();
        let s = state.snapshot();
        assert_eq!(s.mean().unwrap(), Some(S::ZERO));
        assert!(s.population_variance().unwrap().unwrap() > S::ZERO);
        assert!(s.sample_variance().is_err());
        assert!(s.standard_deviation(1).unwrap().unwrap().is_finite());
        assert!(s.variance_of_sample_mean().unwrap().unwrap().is_finite());
        assert_eq!(s.standard_error_of_sample_mean().unwrap(), Some(scale));
        assert_eq!(s.skewness(BiasCorrection::None).unwrap(), Some(S::ZERO));
        assert!(
            (s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
                .unwrap()
                .unwrap()
                - S::ONE)
                .abs()
                <= S::try_from_f64_lossy(16.0).unwrap() * <S as scioxide_core::RealScalar>::EPSILON
        );
        assert!(
            s.confidence_interval_of_mean_with_options(
                Default::default(),
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default()
            )
            .unwrap()
            .is_some()
        );
        assert_eq!(state.snapshot(), s);

        let c = OnlineCovariance::from_slices(&values, &[scale, -scale], order)
            .unwrap()
            .snapshot();
        assert!(c.population_covariance().unwrap().unwrap() < S::ZERO);
        assert!(c.sample_covariance().is_err());
        assert!(c.covariance_of_sample_means().unwrap().unwrap().is_finite());
        assert!(
            (c.correlation().unwrap().unwrap() + S::ONE).abs()
                <= S::try_from_f64_lossy(16.0).unwrap() * <S as scioxide_core::RealScalar>::EPSILON
        );
        assert_eq!(c.marginal_x().mean().unwrap(), Some(S::ZERO));
        each_model!(M, mode => {
            let mut a = OnlineWeightedMoments::<_, M>::new(order);
            let mut c = OnlineWeightedCovariance::<_, M>::new(order);
            for value in values {
                a.observe(value, S::ONE).unwrap();
                c.observe(value, -value, S::ONE).unwrap();
            }
            let a = a.snapshot();
            let c = c.snapshot();
            assert!(a.weighted_population_variance().unwrap().unwrap() > S::ZERO);
            assert_eq!(a.standard_error_of_weighted_mean().unwrap(), Some(scale));
            assert!(
                a.confidence_interval_of_weighted_mean_with_options(
                    Default::default(),
                    scioxide_stats::MeanIntervalMethod::StudentT,
                    scioxide_special::BetaOptions::default()
                )
                .unwrap()
                .is_some()
            );
            assert!(
                c.covariance_of_weighted_means()
                    .unwrap()
                    .unwrap()
                    .is_finite()
            );
            assert!(c.correlation().unwrap().is_some());
            if mode == WeightSemantics::Analytic {
                assert_eq!(a.sample_variance().unwrap(), None);
            } else {
                assert!(a.sample_variance().is_err());
                assert!(c.sample_covariance().is_err());
            }
        });
        let mut a = OnlineWeightedMoments::<_, Analytic>::new(order);
        let mut c = OnlineWeightedCovariance::<_, Analytic>::new(order);
        for value in values {
            a.observe(value, S::ONE).unwrap();
            c.observe(value, -value, S::ONE).unwrap();
        }
        assert!(a.snapshot().noise_scale().is_err());
        assert!(c.snapshot().noise_covariance().is_err());
    }
    check(1e154_f64);
    check(1.5e19_f32);
}

#[test]
fn underflowed_variances_preserve_shape_errors_and_correlation() {
    fn check<S: scioxide_core::RealScalar>(scale: S) {
        let order = MomentOrder::Fourth;
        let s = OnlineMoments::from_slice(&[-scale, scale], order)
            .unwrap()
            .snapshot();
        assert_eq!(s.population_variance().unwrap(), Some(S::ZERO));
        assert_eq!(s.sample_variance().unwrap(), Some(S::ZERO));
        assert_eq!(s.variance_of_sample_mean().unwrap(), Some(S::ZERO));
        assert_eq!(s.standard_error_of_sample_mean().unwrap(), Some(scale));
        assert!(s.standard_deviation(1).unwrap().unwrap() > S::ZERO);
        assert!(
            s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
                .unwrap()
                .unwrap()
                > S::ZERO
        );
        each_model!(M, mode => {
            let mut c = OnlineWeightedCovariance::<_, M>::new(order);
            for x in [-scale, scale] {
                c.observe(x, -x, S::ONE).unwrap();
            }
            let c = c.snapshot();
            let covariance = c.population_covariance().unwrap().unwrap();
            assert_eq!(covariance, S::ZERO);
            assert!(covariance.is_sign_negative());
            assert_eq!(
                c.marginal_x().standard_error_of_weighted_mean().unwrap(),
                Some(scale)
            );
            assert!(
                (c.correlation().unwrap().unwrap() + S::ONE).abs()
                    <= S::try_from_f64_lossy(16.0).unwrap()
                        * <S as scioxide_core::RealScalar>::EPSILON
            );
        });
    }
    check(2.0_f64.powi(-600));
    check(2.0_f32.powi(-100));
}

#[test]
fn standardized_moment_overflow_and_weight_overflow_are_query_local() {
    let mut a = OnlineWeightedMoments::<_, Reliability>::new(MomentOrder::Fourth);
    a.observe(1.0_f64, 1e-200).unwrap();
    a.observe(0.0, 1e200).unwrap();
    let s = a.snapshot();
    assert_eq!(s.weighted_population_variance().unwrap(), Some(0.0));
    assert!((s.skewness(BiasCorrection::None).unwrap().unwrap() / 1e200 - 1.0).abs() < 1e-13);
    assert!(
        s.kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)
            .is_err()
    );
    assert!(s.standard_error_of_weighted_mean().unwrap().is_some());

    let mut a = OnlineWeightedMoments::<_, Reliability>::new(MomentOrder::Fourth);
    a.observe(1.0_f32, f32::from_bits(1)).unwrap();
    a.observe(0.0, f32::MAX).unwrap();
    assert!(a.snapshot().skewness(BiasCorrection::None).is_err());
    assert_eq!(a.snapshot().mean().unwrap(), Some(0.0));
    assert!(
        a.snapshot()
            .standard_error_of_weighted_mean()
            .unwrap()
            .is_some()
    );

    let mut a = OnlineWeightedMoments::<_, Reliability>::new(MomentOrder::Second);
    for value in [1.0, 2.0, 3.0] {
        a.observe(value, f64::MAX).unwrap();
    }
    let s = a.snapshot();
    assert!(s.total_weight().is_err());
    assert!(s.variance_divisor().is_err());
    assert_eq!(s.mean().unwrap(), Some(2.0));
    close(s.sample_variance().unwrap().unwrap(), 1.0);
}

#[test]
fn overflowing_population_covariance_remains_mergeable_and_correlated() {
    fn check<S: scioxide_core::RealScalar>() {
        let mut left = OnlineCovariance::new(MomentOrder::Fourth);
        let mut right = OnlineCovariance::new(MomentOrder::Fourth);
        left.observe(-S::MAX, S::MAX).unwrap();
        right.observe(S::MAX, -S::MAX).unwrap();
        left.merge(&right).unwrap();
        let s = left.snapshot();
        assert!(s.population_covariance().is_err());
        assert!(s.sample_covariance().is_err());
        assert!(s.covariance_of_sample_means().is_err());
        assert_eq!(s.mean_x().unwrap(), Some(S::ZERO));
        assert_eq!(s.marginal_x().standard_error(1).unwrap(), Some(S::MAX));
        assert!(
            (s.correlation().unwrap().unwrap() + S::ONE).abs()
                <= S::try_from_f64_lossy(16.0).unwrap() * <S as scioxide_core::RealScalar>::EPSILON
        );
        each_model!(M, mode => {
            let mut c = OnlineWeightedCovariance::<_, M>::new(MomentOrder::Fourth);
            c.observe(-S::MAX, S::MAX, S::ONE).unwrap();
            let mut other = OnlineWeightedCovariance::<_, M>::new(MomentOrder::Fourth);
            other.observe(S::MAX, -S::MAX, S::ONE).unwrap();
            c.merge(&other).unwrap();
            let c = c.snapshot();
            assert!(c.population_covariance().is_err());
            assert!(c.marginal_x().weighted_population_variance().is_err());
            assert_eq!(
                c.marginal_x().standard_error_of_weighted_mean().unwrap(),
                Some(S::MAX)
            );
            assert!(
                (c.correlation().unwrap().unwrap() + S::ONE).abs()
                    <= S::try_from_f64_lossy(16.0).unwrap()
                        * <S as scioxide_core::RealScalar>::EPSILON
            );
        });
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn replicated_moment_queries_cross_the_native_count_boundary() {
    fn check<S: scioxide_core::RealScalar>() {
        let values = [S::ONE, S::TWO];
        let mut moments =
            scioxide_stats::OnlineMoments::from_slice(&values, scioxide_stats::MomentOrder::Second)
                .unwrap();
        for _ in 0..(usize::BITS - 2).min(54) {
            let snapshot = moments.snapshot();
            let n = snapshot.count() as f64;
            let expected = 0.25 * n / (n - 1.0);
            let variance = snapshot.sample_variance().unwrap().unwrap().into_f64();
            let deviation = snapshot.standard_deviation(1).unwrap().unwrap().into_f64();
            let error = snapshot
                .standard_error_of_sample_mean()
                .unwrap()
                .unwrap()
                .into_f64();
            let tolerance = 16.0 * S::EPSILON.into_f64();
            assert!((variance / expected - 1.0).abs() <= tolerance);
            assert!((deviation / expected.sqrt() - 1.0).abs() <= tolerance);
            assert!((error / (0.25 / (n - 1.0)).sqrt() - 1.0).abs() <= tolerance);
            let previous = moments;
            moments.merge(&previous).unwrap();
        }
    }
    check::<f32>();
    check::<f64>();
}
