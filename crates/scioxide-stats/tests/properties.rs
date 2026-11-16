//! Deterministic algebraic and merge properties for online statistics.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_stats::*;

const MERGE_BOUND_FACTOR: f64 = 128.0;

fn generated_values(seed: u64, count: usize, translation: f64) -> Vec<f64> {
    (0..count)
        .scan(seed, |state, _| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let centered = ((*state >> 32) as i32 % 2001) - 1000;
            Some(translation + f64::from(centered) / 17.0)
        })
        .collect()
}

fn accumulate_f64(values: &[f64]) -> OnlineMoments<f64> {
    values.iter().copied().fold(
        OnlineMoments::new(scioxide_stats::MomentOrder::Second),
        |mut moments, value| {
            moments.observe(value).unwrap();
            moments
        },
    )
}

fn accumulate_f32(values: &[f64]) -> OnlineMoments<f32> {
    values.iter().copied().fold(
        OnlineMoments::new(scioxide_stats::MomentOrder::Second),
        |mut moments, value| {
            moments.observe(value as f32).unwrap();
            moments
        },
    )
}

fn merge_tree_f64(values: &[f64], partition_size: usize) -> (OnlineMoments<f64>, usize) {
    let mut level = values
        .chunks(partition_size)
        .map(accumulate_f64)
        .collect::<Vec<_>>();
    let mut depth = 0;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                let mut merged = pair[0];
                if let Some(right) = pair.get(1) {
                    merged.merge(right).unwrap();
                }
                merged
            })
            .collect();
        depth += 1;
    }
    (level[0], depth)
}

fn merge_tree_f32(values: &[f64], partition_size: usize) -> (OnlineMoments<f32>, usize) {
    let mut level = values
        .chunks(partition_size)
        .map(accumulate_f32)
        .collect::<Vec<_>>();
    let mut depth = 0;
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                let mut merged = pair[0];
                if let Some(right) = pair.get(1) {
                    merged.merge(right).unwrap();
                }
                merged
            })
            .collect();
        depth += 1;
    }
    (level[0], depth)
}

fn scales(values: &[f64]) -> (f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_abs = values.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let range = max - min;
    (max_abs.max(1.0), (range * range).max(1.0))
}

#[test]
fn arbitrary_partitions_match_single_pass_with_documented_scale_bound() {
    for (case, partition_size) in [1_usize, 2, 3, 5, 8, 13, 21].into_iter().enumerate() {
        let values = generated_values(0x1760_0000 + case as u64, 257, 1.0e6);
        let (mean_scale, second_scale) = scales(&values);

        let single_f64 = accumulate_f64(&values).snapshot();
        let (merged_f64, depth) = merge_tree_f64(&values, partition_size);
        let merged_f64 = merged_f64.snapshot();
        let mean_bound = MERGE_BOUND_FACTOR * f64::EPSILON * (depth as f64 + 1.0) * mean_scale;
        let variance_bound =
            MERGE_BOUND_FACTOR * f64::EPSILON * (depth as f64 + 1.0) * second_scale;
        assert!(
            (single_f64.mean().unwrap().unwrap() - merged_f64.mean().unwrap().unwrap()).abs()
                <= mean_bound,
            "case={case}, partition={partition_size}, depth={depth}"
        );
        assert!(
            (single_f64.population_variance().unwrap().unwrap()
                - merged_f64.population_variance().unwrap().unwrap())
            .abs()
                <= variance_bound,
            "case={case}, partition={partition_size}, depth={depth}"
        );

        let single_f32 = accumulate_f32(&values).snapshot();
        let (merged_f32, depth) = merge_tree_f32(&values, partition_size);
        let merged_f32 = merged_f32.snapshot();
        let mean_bound =
            MERGE_BOUND_FACTOR * f64::from(f32::EPSILON) * (depth as f64 + 1.0) * mean_scale;
        let variance_bound =
            MERGE_BOUND_FACTOR * f64::from(f32::EPSILON) * (depth as f64 + 1.0) * second_scale;
        assert!(
            f64::from(
                (single_f32.mean().unwrap().unwrap() - merged_f32.mean().unwrap().unwrap()).abs()
            ) <= mean_bound,
            "case={case}, partition={partition_size}, depth={depth}"
        );
        assert!(
            f64::from(
                (single_f32.population_variance().unwrap().unwrap()
                    - merged_f32.population_variance().unwrap().unwrap())
                .abs()
            ) <= variance_bound,
            "case={case}, partition={partition_size}, depth={depth}"
        );
    }
}

#[test]
fn translation_and_positive_scale_laws_hold() {
    let values = generated_values(0x1760_1000, 193, 0.0);
    let translated = values
        .iter()
        .map(|value| value + 10_000.0)
        .collect::<Vec<_>>();
    let scaled = values.iter().map(|value| value * 3.5).collect::<Vec<_>>();
    let base = accumulate_f64(&values).snapshot();
    let translated = accumulate_f64(&translated).snapshot();
    let scaled = accumulate_f64(&scaled).snapshot();

    assert!(
        (translated.mean().unwrap().unwrap() - (base.mean().unwrap().unwrap() + 10_000.0)).abs()
            < 1.0e-11
    );
    assert!(
        (translated.population_variance().unwrap().unwrap()
            - base.population_variance().unwrap().unwrap())
        .abs()
            < 1.0e-10
    );
    assert!(
        (scaled.mean().unwrap().unwrap() - base.mean().unwrap().unwrap() * 3.5).abs() < 1.0e-12
    );
    assert!(
        (scaled.population_variance().unwrap().unwrap()
            - base.population_variance().unwrap().unwrap() * 3.5_f64.powi(2))
        .abs()
            < 1.0e-10
    );
}

#[test]
fn covariance_is_symmetric_and_scales_bilinearly() {
    let x = generated_values(0x1760_2000, 129, 100.0);
    let y = generated_values(0x1760_3000, 129, -50.0);
    let mut xy = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    let mut yx = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    let mut scaled = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    x.iter()
        .copied()
        .zip(y.iter().copied())
        .try_for_each(|(x_value, y_value)| {
            xy.observe(x_value, y_value)?;
            yx.observe(y_value, x_value)?;
            scaled.observe(2.0 * x_value, 4.0 * y_value)
        })
        .unwrap();
    let xy = xy.snapshot().sample_covariance().unwrap().unwrap();
    let yx = yx.snapshot().sample_covariance().unwrap().unwrap();
    let scaled = scaled.snapshot().sample_covariance().unwrap().unwrap();
    assert!((xy - yx).abs() < 1.0e-12);
    assert!((scaled - 8.0 * xy).abs() < 1.0e-10);
}

#[test]
fn weighted_and_covariance_partition_merges_match_single_pass() {
    let x = generated_values(0x1760_5000, 161, 10_000.0);
    let y = generated_values(0x1760_6000, 161, -20_000.0);
    let weights = generated_values(0x1760_7000, 161, 100.0)
        .into_iter()
        .map(|weight| weight.abs().max(0.25))
        .collect::<Vec<_>>();
    let (_, x_second_scale) = scales(&x);
    let (_, x_range_scale) = scales(&x);
    let (_, y_range_scale) = scales(&y);
    let covariance_scale = (x_range_scale.sqrt() * y_range_scale.sqrt()).max(1.0);

    let mut weighted_single = OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
        scioxide_stats::MomentOrder::Second,
    );
    let mut covariance_single = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    x.iter()
        .copied()
        .zip(y.iter().copied())
        .zip(weights.iter().copied())
        .try_for_each(|((x_value, y_value), weight)| {
            weighted_single.observe(x_value, weight)?;
            covariance_single.observe(x_value, y_value)
        })
        .unwrap();
    let weighted_single = weighted_single.snapshot();
    let covariance_single = covariance_single.snapshot();

    for partition_size in [1_usize, 3, 7, 16, 31] {
        let mut weighted_level = x
            .chunks(partition_size)
            .zip(weights.chunks(partition_size))
            .map(|(x_chunk, weight_chunk)| {
                let mut moments = OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
                    scioxide_stats::MomentOrder::Second,
                );
                x_chunk
                    .iter()
                    .copied()
                    .zip(weight_chunk.iter().copied())
                    .try_for_each(|(value, weight)| moments.observe(value, weight))
                    .unwrap();
                moments
            })
            .collect::<Vec<_>>();
        let mut covariance_level = x
            .chunks(partition_size)
            .zip(y.chunks(partition_size))
            .map(|(x_chunk, y_chunk)| {
                let mut covariance = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
                x_chunk
                    .iter()
                    .copied()
                    .zip(y_chunk.iter().copied())
                    .try_for_each(|(x_value, y_value)| covariance.observe(x_value, y_value))
                    .unwrap();
                covariance
            })
            .collect::<Vec<_>>();
        let mut depth = 0;
        while weighted_level.len() > 1 {
            weighted_level = weighted_level
                .chunks(2)
                .map(|pair| {
                    let mut merged = pair[0];
                    if let Some(right) = pair.get(1) {
                        merged.merge(right).unwrap();
                    }
                    merged
                })
                .collect();
            covariance_level = covariance_level
                .chunks(2)
                .map(|pair| {
                    let mut merged = pair[0];
                    if let Some(right) = pair.get(1) {
                        merged.merge(right).unwrap();
                    }
                    merged
                })
                .collect();
            depth += 1;
        }

        let weighted = weighted_level[0].snapshot();
        let covariance = covariance_level[0].snapshot();
        let multiplier = MERGE_BOUND_FACTOR * f64::EPSILON * (depth as f64 + 1.0);
        assert!(
            (weighted.weighted_population_variance().unwrap().unwrap()
                - weighted_single
                    .weighted_population_variance()
                    .unwrap()
                    .unwrap())
            .abs()
                <= multiplier * x_second_scale,
            "weighted partition={partition_size}, depth={depth}"
        );
        assert!(
            (covariance.sample_covariance().unwrap().unwrap()
                - covariance_single.sample_covariance().unwrap().unwrap())
            .abs()
                <= multiplier * covariance_scale,
            "covariance partition={partition_size}, depth={depth}"
        );
    }
}

#[test]
fn finite_inputs_produce_non_negative_finite_variances() {
    for seed in 0_u64..64 {
        let values = generated_values(0x1760_4000 + seed, 65, seed as f64 * 1.0e4);
        let snapshot = accumulate_f64(&values).snapshot();
        assert!(snapshot.population_variance().unwrap().unwrap().is_finite());
        assert!(snapshot.population_variance().unwrap().unwrap() >= 0.0);
        assert!(snapshot.sample_variance().unwrap().unwrap().is_finite());
        assert!(snapshot.sample_variance().unwrap().unwrap() >= 0.0);
    }
}

#[test]
fn reliability_weights_are_scale_invariant_and_match_repeated_population_moments() {
    let observations = [(2.0_f64, 3_u32), (5.0, 2), (11.0, 4)];
    let mut weighted = OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
        scioxide_stats::MomentOrder::Second,
    );
    let mut rescaled = OnlineWeightedMoments::<_, scioxide_stats::Reliability>::new(
        scioxide_stats::MomentOrder::Second,
    );
    let mut repeated = OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    observations
        .into_iter()
        .try_for_each(|(value, repetitions)| {
            weighted.observe(value, f64::from(repetitions))?;
            rescaled.observe(value, f64::from(repetitions) * 7.0)?;
            (0..repetitions).try_for_each(|_| repeated.observe(value))
        })
        .unwrap();
    let weighted = weighted.snapshot();
    let rescaled = rescaled.snapshot();
    let repeated = repeated.snapshot();
    assert!(
        (weighted.mean().unwrap().unwrap() - rescaled.mean().unwrap().unwrap()).abs() < 1.0e-14
    );
    assert!(
        (weighted.sample_variance().unwrap().unwrap()
            - rescaled.sample_variance().unwrap().unwrap())
        .abs()
            < 1.0e-13
    );
    assert!(
        (weighted.mean().unwrap().unwrap() - repeated.mean().unwrap().unwrap()).abs() < 1.0e-14
    );
    assert!(
        (weighted.weighted_population_variance().unwrap().unwrap()
            - repeated.population_variance().unwrap().unwrap())
        .abs()
            < 1.0e-13
    );
}

fn quantile_probability(value: f64) -> QuantileProbability {
    QuantileProbability::new(value).unwrap()
}

#[test]
fn empirical_queries_are_permutation_invariant() {
    let values = generated_values(0x1770_1000, 97, 0.0);
    let mut reversed = values.clone();
    reversed.reverse();
    let mut rotated = values.clone();
    rotated.rotate_left(31);
    let distributions = [&values, &reversed, &rotated]
        .map(|sample| EmpiricalDistribution::new(sample, Unweighted).unwrap());

    for method in [QuantileMethod::NearestRank, QuantileMethod::Linear] {
        for p in [0.0, 0.01, 0.1, 0.37, 0.5, 0.91, 1.0] {
            let expected = distributions[0]
                .quantile(quantile_probability(p), QuantileRule::new(method))
                .unwrap();
            assert_eq!(
                distributions[1]
                    .quantile(quantile_probability(p), QuantileRule::new(method))
                    .unwrap(),
                expected
            );
            assert_eq!(
                distributions[2]
                    .quantile(quantile_probability(p), QuantileRule::new(method))
                    .unwrap(),
                expected
            );
        }
    }

    for side in [TailSide::Lower, TailSide::Upper] {
        for p in [0.1, 0.37, 0.5, 0.91] {
            let expected = distributions[0]
                .tail_mean(quantile_probability(p), side)
                .unwrap();
            assert_eq!(
                distributions[1]
                    .tail_mean(quantile_probability(p), side)
                    .unwrap(),
                expected
            );
            assert_eq!(
                distributions[2]
                    .tail_mean(quantile_probability(p), side)
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn empirical_quantiles_are_monotone_and_bounded() {
    let values = generated_values(0x1770_2000, 129, 1.0e3);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let distribution = EmpiricalDistribution::new(&values, Unweighted).unwrap();
    let probabilities = (0_u16..=100)
        .map(|index| f64::from(index) / 100.0)
        .collect::<Vec<_>>();
    for method in [QuantileMethod::NearestRank, QuantileMethod::Linear] {
        let mut previous = f64::NEG_INFINITY;
        for p in &probabilities {
            let value = distribution
                .quantile(quantile_probability(*p), QuantileRule::new(method))
                .unwrap()
                .value();
            assert!(value >= previous, "method={method:?}, p={p}");
            assert!(value >= min && value <= max, "method={method:?}, p={p}");
            previous = value;
        }
    }

    let weighted_values = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, f64::from((index % 7 + 1) as u16) / 3.0))
        .collect::<Vec<_>>();
    let weighted = {
        let pairs = &weighted_values;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    let mut previous = f64::NEG_INFINITY;
    for p in probabilities {
        let value = weighted
            .quantile(quantile_probability(p), QuantileRule::inverse_ecdf())
            .unwrap()
            .value();
        assert!(value >= previous, "weighted p={p}");
        assert!(value >= min && value <= max, "weighted p={p}");
        previous = value;
    }
}

#[test]
fn empirical_queries_obey_positive_affine_transformation_laws() {
    let values = generated_values(0x1770_3000, 73, -20.0);
    let scale = 2.75_f64;
    let translation = -14.5_f64;
    let transformed = values
        .iter()
        .map(|value| scale.mul_add(*value, translation))
        .collect::<Vec<_>>();
    let base = EmpiricalDistribution::new(&values, Unweighted).unwrap();
    let affine = EmpiricalDistribution::new(&transformed, Unweighted).unwrap();

    for p in [0.05, 0.23, 0.5, 0.81, 0.97] {
        for method in [QuantileMethod::NearestRank, QuantileMethod::Linear] {
            let actual = affine
                .quantile(quantile_probability(p), QuantileRule::new(method))
                .unwrap()
                .value();
            let expected = scale.mul_add(
                base.quantile(quantile_probability(p), QuantileRule::new(method))
                    .unwrap()
                    .value(),
                translation,
            );
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "method={method:?}, p={p}"
            );
        }
        for side in [TailSide::Lower, TailSide::Upper] {
            let original = base.tail_mean(quantile_probability(p), side).unwrap();
            let actual = affine.tail_mean(quantile_probability(p), side).unwrap();
            assert!((actual.mean() - scale.mul_add(original.mean(), translation)).abs() < 1.0e-12);
            assert!(
                (actual.boundary_value() - scale.mul_add(original.boundary_value(), translation))
                    .abs()
                    < 1.0e-12
            );
            assert_eq!(actual.boundary_mass(), original.boundary_mass());
            let expected_contribution = scale.mul_add(
                original.boundary_contribution().unwrap(),
                translation * original.boundary_mass().unwrap(),
            );
            assert!(
                (actual.boundary_contribution().unwrap() - expected_contribution).abs() < 1.0e-11
            );
        }
    }

    let f32_values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let f32_transformed = f32_values
        .iter()
        .map(|value| 1.5_f32.mul_add(*value, 7.0))
        .collect::<Vec<_>>();
    let f32_base = EmpiricalDistribution::new(&f32_values, Unweighted).unwrap();
    let f32_affine = EmpiricalDistribution::new(&f32_transformed, Unweighted).unwrap();
    let original = f32_base
        .quantile(
            quantile_probability(0.43),
            QuantileRule::new(QuantileMethod::Linear),
        )
        .unwrap()
        .value();
    let actual = f32_affine
        .quantile(
            quantile_probability(0.43),
            QuantileRule::new(QuantileMethod::Linear),
        )
        .unwrap()
        .value();
    assert!((actual - 1.5_f32.mul_add(original, 7.0)).abs() < 2.0e-5);
}

#[test]
fn integer_weights_match_replicated_inverse_ecdf_and_tails() {
    let weighted_values = [(1.0_f64, 2_u16), (3.0, 1), (5.0, 4), (12.0, 3)];
    let pairs = weighted_values.map(|(value, repetitions)| (value, f64::from(repetitions)));
    let repeated = weighted_values
        .into_iter()
        .flat_map(|(value, repetitions)| std::iter::repeat_n(value, usize::from(repetitions)))
        .collect::<Vec<_>>();
    let weighted = {
        let pairs = &pairs;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    let unweighted = EmpiricalDistribution::new(&repeated, Unweighted).unwrap();

    for p in [0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0] {
        assert_eq!(
            weighted
                .quantile(quantile_probability(p), QuantileRule::inverse_ecdf())
                .unwrap()
                .value(),
            unweighted
                .quantile(
                    quantile_probability(p),
                    QuantileRule::new(QuantileMethod::NearestRank)
                )
                .unwrap()
                .value()
        );
        if p > 0.0 {
            let weighted_tail = weighted
                .tail_mean(quantile_probability(p), TailSide::Lower)
                .unwrap();
            let repeated_tail = unweighted
                .tail_mean(quantile_probability(p), TailSide::Lower)
                .unwrap();
            assert!((weighted_tail.mean() - repeated_tail.mean()).abs() < 1.0e-13);
            assert!(
                (weighted_tail.tail_mass().unwrap() - repeated_tail.tail_mass().unwrap()).abs()
                    < 1.0e-13
            );
        }
        if p < 1.0 {
            let weighted_tail = weighted
                .tail_mean(quantile_probability(p), TailSide::Upper)
                .unwrap();
            let repeated_tail = unweighted
                .tail_mean(quantile_probability(p), TailSide::Upper)
                .unwrap();
            assert!((weighted_tail.mean() - repeated_tail.mean()).abs() < 1.0e-13);
            assert!(
                (weighted_tail.tail_mass().unwrap() - repeated_tail.tail_mass().unwrap()).abs()
                    < 1.0e-13
            );
        }
    }
}

#[test]
fn tail_means_remain_between_the_boundary_and_sample_bounds() {
    let values = generated_values(0x1770_4000, 111, 50.0);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let distribution = EmpiricalDistribution::new(&values, Unweighted).unwrap();
    for p in [0.01, 0.1, 0.33, 0.5, 0.87, 0.99] {
        let lower = distribution
            .tail_mean(quantile_probability(p), TailSide::Lower)
            .unwrap();
        let upper = distribution
            .tail_mean(quantile_probability(p), TailSide::Upper)
            .unwrap();
        assert!(lower.mean() >= min && lower.mean() <= lower.boundary_value());
        assert!(upper.mean() >= upper.boundary_value() && upper.mean() <= max);
        assert!(lower.mean() <= upper.mean());
    }
}
