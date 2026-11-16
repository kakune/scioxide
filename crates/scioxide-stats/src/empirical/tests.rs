use super::position::scaled_probability;
use super::selection::WeightedSelectionMetrics;
use super::tail::unweighted_tail_plan;
use super::*;
use alloc::vec;
use alloc::vec::Vec;

#[test]
fn finite_total_order_preserves_primitive_boundaries_and_zero_signs() {
    fn check<S: RealScalar>() {
        let ordered = [
            -S::MAX,
            -S::ONE,
            -S::MIN_SUBNORMAL,
            S::NEG_ZERO,
            S::ZERO,
            S::MIN_SUBNORMAL,
            S::ONE,
            S::MAX,
        ];
        for (i, left) in ordered.iter().enumerate() {
            for (j, right) in ordered.iter().enumerate() {
                assert_eq!(finite_total_cmp(left, right), i.cmp(&j));
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn adjacent_selection_matches_sorted_queries_on_both_tails() {
    fn check<S: RealScalar>() {
        let values = [
            S::MAX,
            S::ZERO,
            -S::ONE,
            S::MIN_SUBNORMAL,
            -S::MAX,
            S::ONE,
            -S::MIN_SUBNORMAL,
            S::NEG_ZERO,
        ];
        let sorted = RawEmpiricalDistribution::new(&values).unwrap();
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
            for p in [0.0, 0.125, 0.25, 0.49, 0.5, 0.51, 0.75, 0.9, 1.0] {
                let expected = sorted.quantile(probability(p), method);
                let mut work = values;
                let actual = quantile_in_place(&mut work, probability(p), method);
                if let Ok(expected) = expected {
                    let actual = actual.unwrap();
                    assert_eq!(actual, expected);
                    assert_eq!(
                        actual.value().is_sign_negative(),
                        expected.value().is_sign_negative()
                    );
                } else {
                    assert!(actual.is_err());
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

fn probability(value: f64) -> QuantileProbability {
    QuantileProbability::new(value).unwrap()
}

fn weighted_bits(samples: &[(f64, f64)]) -> Vec<(u64, u64)> {
    samples
        .iter()
        .map(|(value, weight)| (value.to_bits(), weight.to_bits()))
        .collect()
}

fn assert_weighted_selection_matches_oracle<S: RealScalar>(
    samples: &[(S, S)],
    probabilities: &[f64],
) {
    let oracle = RawWeightedEmpiricalDistribution::new(samples).unwrap();
    for &requested_probability in probabilities {
        let probability = probability(requested_probability);
        let expected = oracle.weighted_inverse_ecdf_quantile(probability).unwrap();
        let mut work = samples.to_vec();
        let actual = weighted_inverse_ecdf_quantile_in_place(&mut work, probability).unwrap();
        assert_eq!(actual, expected);

        let mut expected_pairs = samples.to_vec();
        expected_pairs.sort_unstable_by(weighted_total_cmp);
        work.sort_unstable_by(weighted_total_cmp);
        assert_eq!(work, expected_pairs);
    }
}

#[test]
fn probability_requires_a_finite_closed_unit_interval_value() {
    for invalid in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        assert!(
            matches ! (& (QuantileProbability :: new (invalid)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
    }
    for invalid in [-0.1, 1.1] {
        assert!(
            matches ! (& (QuantileProbability :: new (invalid)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
    }
    assert_eq!(probability(0.0).get(), 0.0);
    assert_eq!(probability(1.0).get(), 1.0);
}

#[test]
fn construction_rejects_empty_and_non_finite_samples_without_mutating_input() {
    assert!(
        matches ! (& (RawEmpiricalDistribution :: < f64 > :: new (& [])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    let samples = [2.0_f64, f64::NAN, 1.0];
    assert!(
        matches ! (& (RawEmpiricalDistribution :: new (& samples)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    assert_eq!(samples[0], 2.0);
    assert!(samples[1].is_nan());
    assert_eq!(samples[2], 1.0);
}

#[test]
fn singleton_and_endpoints_are_exact_for_both_methods() {
    let singleton = RawEmpiricalDistribution::new(&[3.5_f64]).unwrap();
    let samples = RawEmpiricalDistribution::new(&[4.0_f64, 1.0, 3.0, 2.0]).unwrap();
    for method in [QuantileMethod::NearestRank, QuantileMethod::Linear] {
        assert_eq!(
            singleton
                .quantile(probability(0.37), method)
                .unwrap()
                .value(),
            3.5
        );
        assert_eq!(
            samples.quantile(probability(0.0), method).unwrap().value(),
            1.0
        );
        assert_eq!(
            samples.quantile(probability(1.0), method).unwrap().value(),
            4.0
        );
    }
}

#[test]
fn nearest_rank_and_linear_methods_are_explicit_and_distinct() {
    let distribution = RawEmpiricalDistribution::new(&[1.0_f64, 2.0, 8.0, 10.0]).unwrap();
    let nearest = distribution
        .quantile(probability(0.5), QuantileMethod::NearestRank)
        .unwrap();
    let linear = distribution
        .quantile(probability(0.5), QuantileMethod::Linear)
        .unwrap();
    assert_eq!(nearest.value(), 2.0);
    assert_eq!(linear.value(), 5.0);
    assert_eq!(
        nearest.convention(),
        QuantileConvention::Unweighted(QuantileMethod::NearestRank)
    );
    assert_eq!(linear.probability(), probability(0.5));
}

#[test]
fn f32_linear_quantile_retains_a_tiny_f64_interpolation_fraction() {
    let p = 2.0_f64.powi(-151);
    let expected = (f64::from(f32::MAX) * p) as f32;
    assert!(expected > 0.0);

    let distribution = RawEmpiricalDistribution::new(&[0.0_f32, f32::MAX]).unwrap();
    assert_eq!(
        distribution
            .quantile(probability(p), QuantileMethod::Linear)
            .unwrap()
            .value(),
        expected
    );

    let mut work = [0.0_f32, f32::MAX];
    assert_eq!(
        quantile_in_place(&mut work, probability(p), QuantileMethod::Linear)
            .unwrap()
            .value(),
        expected
    );
}

#[test]
fn f32_weighted_boundaries_are_decided_before_scalar_conversion() {
    let boundary = 0.5_f64;
    let above_boundary = f64::from_bits(boundary.to_bits() + 1);
    let distribution =
        RawWeightedEmpiricalDistribution::new(&[(0.0_f32, 1.0), (1.0, 1.0)]).unwrap();

    assert_eq!(
        distribution
            .weighted_inverse_ecdf_quantile(probability(boundary))
            .unwrap()
            .value(),
        0.0
    );
    assert_eq!(
        distribution
            .weighted_inverse_ecdf_quantile(probability(above_boundary))
            .unwrap()
            .value(),
        1.0
    );
    let tail = distribution
        .tail_mean(probability(above_boundary), TailSide::Lower)
        .unwrap();
    assert_eq!(tail.boundary_value(), 1.0);
    assert!(tail.mean() > 0.0);
}

#[test]
fn adjacent_probabilities_preserve_exact_discrete_rank_boundaries() {
    let below_one_third = 1.0_f64 / 3.0;
    let above_one_third = f64::from_bits(below_one_third.to_bits() + 1);
    let below = probability(below_one_third);
    let above = probability(above_one_third);
    let distribution = RawEmpiricalDistribution::new(&[0.0_f64, 1.0, 2.0]).unwrap();

    assert_eq!(
        distribution
            .quantile(below, QuantileMethod::NearestRank)
            .unwrap()
            .value(),
        0.0
    );
    assert_eq!(
        distribution
            .quantile(above, QuantileMethod::NearestRank)
            .unwrap()
            .value(),
        1.0
    );

    let mut in_place = [0.0_f64, 1.0, 2.0];
    assert_eq!(
        quantile_in_place(&mut in_place, above, QuantileMethod::NearestRank)
            .unwrap()
            .value(),
        1.0
    );

    let weighted =
        RawWeightedEmpiricalDistribution::new(&[(0.0_f64, 1.0), (1.0, 1.0), (2.0, 1.0)]).unwrap();
    assert_eq!(
        weighted
            .weighted_inverse_ecdf_quantile(above)
            .unwrap()
            .value(),
        1.0
    );

    let (_, boundary_fraction) = scaled_probability(3, above).unwrap();
    assert!(boundary_fraction > 0.0);
    let tail_plan = unweighted_tail_plan(3, above, TailSide::Lower).unwrap();
    assert_eq!(tail_plan.boundary_index, 1);
    assert_eq!(
        tail_plan.boundary_mass.to_optional_scalar(),
        Some(boundary_fraction)
    );
    let tail = distribution.tail_mean(above, TailSide::Lower).unwrap();
    assert_eq!(tail.boundary_value(), 1.0);
    assert_eq!(tail.boundary_mass(), Some(boundary_fraction));
    assert!(tail.mean() > 0.0);
}

#[test]
fn ties_and_signed_zero_use_deterministic_finite_total_order() {
    let distribution = RawEmpiricalDistribution::new(&[2.0_f64, -0.0, 2.0, 1.0, 0.0]).unwrap();
    let negative_zero = distribution
        .quantile(probability(0.0), QuantileMethod::NearestRank)
        .unwrap()
        .value();
    let positive_zero = distribution
        .quantile(probability(0.4), QuantileMethod::NearestRank)
        .unwrap()
        .value();
    assert!(negative_zero.is_sign_negative());
    assert!(!positive_zero.is_sign_negative());
    assert_eq!(
        distribution
            .quantile(probability(0.8), QuantileMethod::NearestRank)
            .unwrap()
            .value(),
        2.0
    );
}

#[test]
fn in_place_selection_matches_immutable_queries() {
    let samples = [9.0_f64, 1.0, 7.0, 3.0, 5.0, 5.0];
    let immutable = RawEmpiricalDistribution::new(&samples).unwrap();
    for method in [QuantileMethod::NearestRank, QuantileMethod::Linear] {
        for p in [0.0, 0.1, 0.5, 0.75, 1.0] {
            let mut work = samples;
            let expected = immutable.quantile(probability(p), method).unwrap();
            let actual = quantile_in_place(&mut work, probability(p), method).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn in_place_validation_errors_leave_the_buffer_unchanged() {
    let mut empty: [f64; 0] = [];
    assert!(
        matches ! (& (quantile_in_place (& mut empty , probability (0.5) , QuantileMethod :: NearestRank)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );

    let mut non_finite = [3.0_f64, f64::INFINITY, 1.0];
    let original = non_finite;
    assert!(
        matches ! (& (quantile_in_place (& mut non_finite , probability (0.5) , QuantileMethod :: Linear)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    assert_eq!(non_finite, original);
}

#[test]
fn checked_positions_support_maximum_usize_without_wrapping() {
    let nearest =
        quantile_position(usize::MAX, probability(1.0), QuantileMethod::NearestRank).unwrap();
    assert_eq!(nearest.lower, usize::MAX - 1);
    let linear = quantile_position(usize::MAX, probability(1.0), QuantileMethod::Linear).unwrap();
    assert_eq!(linear.lower, usize::MAX - 1);
    assert_eq!(linear.upper, usize::MAX - 1);
}

#[test]
fn convex_interpolation_avoids_overflowing_the_endpoint_difference() {
    let distribution = RawEmpiricalDistribution::new(&[-f64::MAX, f64::MAX]).unwrap();
    assert_eq!(
        distribution
            .quantile(probability(0.5), QuantileMethod::Linear)
            .unwrap()
            .value(),
        0.0
    );
}

#[test]
fn weighted_construction_validates_pairs_and_total_weight() {
    assert!(
        matches ! (& (RawWeightedEmpiricalDistribution :: < f64 > :: new (& [])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    for samples in [[(f64::NAN, 1.0)], [(1.0, f64::INFINITY)]] {
        assert!(
            matches ! (& (RawWeightedEmpiricalDistribution :: new (& samples)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
    }
    for samples in [[(1.0_f64, -1.0)], [(1.0, 0.0)]] {
        assert!(
            matches ! (& (RawWeightedEmpiricalDistribution :: new (& samples)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
    }

    let distribution =
        RawWeightedEmpiricalDistribution::new(&[(0.0_f64, 0.0), (2.0, 1.0)]).unwrap();
    assert_eq!(distribution.total_weight(), Some(1.0));
    assert_eq!(
        distribution
            .weighted_inverse_ecdf_quantile(probability(0.0))
            .unwrap()
            .value(),
        2.0
    );
}

#[test]
fn exact_weight_accumulation_retains_extreme_dynamic_range() {
    let large = 2.0_f64.powi(54);
    for samples in [
        [(0.0_f64, 1.0), (1.0, large)],
        [(1.0_f64, large), (0.0, 1.0)],
    ] {
        let distribution = RawWeightedEmpiricalDistribution::new(&samples).unwrap();
        assert_eq!(distribution.total_weight(), Some(large));
        let mut work = samples;
        assert_eq!(
            weighted_inverse_ecdf_quantile_in_place(&mut work, probability(2.0_f64.powi(-54)),)
                .unwrap()
                .value(),
            1.0
        );
    }
}

#[test]
fn exact_full_tails_retain_small_support_and_ignore_input_order() {
    let large = 2.0_f64.powi(54);
    let expected = 2.0_f64.powi(-54);
    let forward = RawWeightedEmpiricalDistribution::new(&[(0.0_f64, large), (1.0, 1.0)]).unwrap();
    let reverse = RawWeightedEmpiricalDistribution::new(&[(1.0_f64, 1.0), (0.0, large)]).unwrap();

    for distribution in [&forward, &reverse] {
        let lower = distribution
            .tail_mean(probability(1.0), TailSide::Lower)
            .unwrap();
        let upper = distribution
            .tail_mean(probability(0.0), TailSide::Upper)
            .unwrap();
        assert_eq!(lower.mean(), expected);
        assert_eq!(upper.mean(), expected);
        assert_eq!(lower.tail_mass(), Some(large));
        assert_eq!(upper.tail_mass(), Some(large));
    }
}

#[test]
fn unavailable_tail_metadata_does_not_shrink_the_mean_domain() {
    let distribution =
        RawWeightedEmpiricalDistribution::new(&[(1.0_f64, f64::MAX), (2.0, f64::MAX)]).unwrap();
    let full = distribution
        .tail_mean(probability(1.0), TailSide::Lower)
        .unwrap();
    assert_eq!(full.mean(), 1.5);
    assert_eq!(full.tail_mass(), None);
    assert_eq!(full.boundary_mass(), Some(f64::MAX));
    assert_eq!(full.boundary_contribution(), None);

    let minimum_subnormal = f32::from_bits(1);
    let subnormal = RawEmpiricalDistribution::new(&[minimum_subnormal]).unwrap();
    let partial = subnormal
        .tail_mean(probability(0.5), TailSide::Lower)
        .unwrap();
    assert_eq!(partial.mean(), minimum_subnormal);
    assert_eq!(partial.boundary_mass(), Some(0.5));
    assert_eq!(partial.boundary_contribution(), None);
}

#[test]
fn weighted_selection_matches_sorted_oracle_across_input_shapes() {
    let mut state = 0x4752_4144_4941_0215_u64;
    let random = (0..257)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 24) % 41) as f64 - 20.0;
            let weight = if index % 13 == 0 {
                0.0
            } else {
                f64::from((index % 9 + 1) as u8) / 8.0
            };
            (value, weight)
        })
        .collect::<Vec<_>>();
    let mut sorted = random.clone();
    sorted.sort_unstable_by(weighted_total_cmp);
    let mut reverse = sorted.clone();
    reverse.reverse();
    let duplicate_heavy = (0..193)
        .map(|index| ((index % 5) as f64, f64::from((index % 7 + 1) as u8)))
        .collect::<Vec<_>>();
    let dominant = vec![(4.0, 1.0), (-3.0, 1.0), (8.0, 2.0_f64.powi(40)), (1.0, 0.0)];
    let all_equal = (0..129)
        .map(|index| ((index % 11) as f64, 0.25))
        .collect::<Vec<_>>();
    for samples in [
        &random,
        &sorted,
        &reverse,
        &duplicate_heavy,
        &dominant,
        &all_equal,
    ] {
        assert_weighted_selection_matches_oracle(samples, &[0.0, 0.01, 0.2, 0.5, 0.73, 0.975, 1.0]);
    }

    assert_weighted_selection_matches_oracle(
        &[(3.0_f32, 0.0), (1.0, 0.25), (2.0, 0.5), (4.0, 0.25)],
        &[0.0, 0.25, 0.5, 0.75, 1.0],
    );
}

#[test]
fn weighted_selection_handles_subnormal_mass_boundaries_and_signed_zero() {
    let subnormal = f64::from_bits(1);
    assert_weighted_selection_matches_oracle(
        &[(2.0, subnormal), (0.0, subnormal), (1.0, subnormal)],
        &[0.0, 0.5, 1.0],
    );

    let one_third = 1.0_f64 / 3.0;
    let above_one_third = f64::from_bits(one_third.to_bits() + 1);
    let samples = [(-0.0_f64, 1.0), (0.0, 1.0), (1.0, 1.0)];
    let mut at_boundary = samples;
    let selected =
        weighted_inverse_ecdf_quantile_in_place(&mut at_boundary, probability(one_third))
            .unwrap()
            .value();
    assert_eq!(selected.to_bits(), (-0.0_f64).to_bits());

    let mut above_boundary = samples;
    let selected =
        weighted_inverse_ecdf_quantile_in_place(&mut above_boundary, probability(above_one_third))
            .unwrap()
            .value();
    assert_eq!(selected.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn weighted_selection_errors_are_bitwise_atomic() {
    let invalid_cases = [
        vec![(f64::NAN, 1.0), (-0.0, 2.0)],
        vec![(1.0, f64::INFINITY), (-0.0, 2.0)],
        vec![(1.0, -1.0), (-0.0, 2.0)],
        vec![(1.0, -0.0), (-0.0, 0.0)],
    ];
    for mut work in invalid_cases {
        let original = weighted_bits(&work);
        assert!(weighted_inverse_ecdf_quantile_in_place(&mut work, probability(0.5)).is_err());
        assert_eq!(weighted_bits(&work), original);
    }

    let mut exact_tiny_target = [(1.0_f32, 1.0_f32), (-0.0, 2.0)];
    let selected = weighted_inverse_ecdf_quantile_in_place(
        &mut exact_tiny_target,
        probability(f64::MIN_POSITIVE),
    )
    .unwrap()
    .value();
    assert_eq!(selected.to_bits(), (-0.0_f32).to_bits());
}

#[test]
fn weighted_selection_is_linear_and_uses_bounded_fallback() {
    let common = (0..4_096)
        .map(|index| {
            let scrambled = (index * 2_653 + 17) % 4_096;
            (scrambled as f64, f64::from((index % 11 + 1) as u8))
        })
        .collect::<Vec<_>>();
    let mut common_work = common.clone();
    let (common_result, common_metrics) = WeightedSelectionMetrics::record(|| {
        weighted_inverse_ecdf_quantile_in_place(&mut common_work, probability(0.73))
    });
    let _ = common_result.unwrap();
    assert!(common_metrics.work_units() <= 16 * common.len() as u64);

    let mut organ_pipe = (0..2_048)
        .chain((0..2_048).rev())
        .map(|value| (value as f64, 1.0))
        .collect::<Vec<_>>();
    let (result, adversarial_metrics) = WeightedSelectionMetrics::record(|| {
        weighted_inverse_ecdf_quantile_in_place(&mut organ_pipe, probability(0.9))
    });
    let result = result.unwrap();
    assert_eq!(result.value(), 1_843.0);
    assert!(adversarial_metrics.fallback_activations() > 0);
    assert!(adversarial_metrics.partitions() > 0);
    assert!(adversarial_metrics.work_units() <= 128 * organ_pipe.len() as u64);
}

#[test]
fn successful_weighted_selection_does_not_sort_the_buffer() {
    let samples = [
        (19.0_f64, 1.0),
        (2.0, 3.0),
        (17.0, 2.0),
        (4.0, 1.0),
        (15.0, 2.0),
        (6.0, 3.0),
        (13.0, 1.0),
        (8.0, 2.0),
        (11.0, 3.0),
        (10.0, 1.0),
        (9.0, 2.0),
        (12.0, 3.0),
        (7.0, 1.0),
        (14.0, 2.0),
        (5.0, 3.0),
        (16.0, 1.0),
        (3.0, 2.0),
        (18.0, 3.0),
        (1.0, 1.0),
    ];
    let mut work = samples;
    let _ = weighted_inverse_ecdf_quantile_in_place(&mut work, probability(0.73)).unwrap();
    assert!(
        work.windows(2)
            .any(|pair| { weighted_total_cmp(&pair[0], &pair[1]) == Ordering::Greater })
    );
    let mut sorted_original = samples;
    sorted_original.sort_unstable_by(weighted_total_cmp);
    work.sort_unstable_by(weighted_total_cmp);
    assert_eq!(work, sorted_original);
}

#[test]
fn fractional_weights_follow_the_named_inverse_ecdf() {
    let distribution =
        RawWeightedEmpiricalDistribution::new(&[(5.0_f64, 1.0), (1.0, 0.5), (2.0, 1.5)]).unwrap();
    let cases = [
        (0.0, 1.0),
        (1.0 / 6.0, 1.0),
        (0.2, 2.0),
        (0.75, 5.0),
        (1.0, 5.0),
    ];
    for (p, expected) in cases {
        let result = distribution
            .weighted_inverse_ecdf_quantile(probability(p))
            .unwrap();
        assert_eq!(result.value(), expected);
        assert_eq!(result.convention(), QuantileConvention::MassInverseEcdf);
    }
}

#[test]
fn equal_weights_share_nearest_rank_indices() {
    let values = [8.0_f64, 1.0, 3.0, 3.0, 12.0];
    let weighted_values = values.map(|value| (value, 2.5));
    let unweighted = RawEmpiricalDistribution::new(&values).unwrap();
    let weighted = RawWeightedEmpiricalDistribution::new(&weighted_values).unwrap();
    for p in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0] {
        assert_eq!(
            weighted
                .weighted_inverse_ecdf_quantile(probability(p))
                .unwrap()
                .value(),
            unweighted
                .quantile(probability(p), QuantileMethod::NearestRank)
                .unwrap()
                .value()
        );
    }
}

#[test]
fn weighted_in_place_queries_match_immutable_queries() {
    let samples = [(8.0_f64, 0.5), (1.0, 2.0), (4.0, 0.0), (3.0, 1.5)];
    let immutable = RawWeightedEmpiricalDistribution::new(&samples).unwrap();
    for p in [0.0, 0.2, 0.5, 0.9, 1.0] {
        let mut work = samples;
        assert_eq!(
            weighted_inverse_ecdf_quantile_in_place(&mut work, probability(p)).unwrap(),
            immutable
                .weighted_inverse_ecdf_quantile(probability(p))
                .unwrap()
        );
    }

    let mut invalid = [(2.0_f64, 1.0), (1.0, -0.5)];
    let original = invalid;
    assert!(
        matches ! (& (weighted_inverse_ecdf_quantile_in_place (& mut invalid , probability (0.5))) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    assert_eq!(invalid, original);
}

#[test]
fn unweighted_tail_means_include_fractional_boundary_mass() {
    let distribution = RawEmpiricalDistribution::new(&[4.0_f64, 1.0, 3.0, 2.0]).unwrap();
    let lower = distribution
        .tail_mean(probability(0.375), TailSide::Lower)
        .unwrap();
    assert!((lower.mean() - 4.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(lower.tail_mass(), Some(1.5));
    assert_eq!(lower.boundary_value(), 2.0);
    assert_eq!(lower.boundary_mass(), Some(0.5));
    assert_eq!(lower.boundary_contribution(), Some(1.0));
    assert_eq!(lower.side(), TailSide::Lower);
    assert_eq!(
        lower.convention(),
        TailMeanConvention::FractionalInverseEcdf
    );

    let upper = distribution
        .tail_mean(probability(0.625), TailSide::Upper)
        .unwrap();
    assert!((upper.mean() - 11.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(upper.tail_mass(), Some(1.5));
    assert_eq!(upper.boundary_value(), 3.0);
    assert_eq!(upper.boundary_mass(), Some(0.5));
    assert_eq!(upper.boundary_contribution(), Some(1.5));
}

#[test]
fn tied_unweighted_boundaries_report_aggregate_included_mass() {
    let distribution = RawEmpiricalDistribution::new(&[2.0_f64, 9.0, 2.0, 1.0, 2.0]).unwrap();
    let lower = distribution
        .tail_mean(probability(0.7), TailSide::Lower)
        .unwrap();
    assert!((lower.mean() - 6.0 / 3.5).abs() < 1.0e-14);
    assert_eq!(lower.boundary_value(), 2.0);
    assert_eq!(lower.boundary_mass(), Some(2.5));
    assert_eq!(lower.boundary_contribution(), Some(5.0));
}

#[test]
fn tail_endpoints_distinguish_whole_and_empty_tails() {
    let distribution = RawEmpiricalDistribution::new(&[1.0_f64, 2.0, 6.0]).unwrap();
    let lower = distribution
        .tail_mean(probability(1.0), TailSide::Lower)
        .unwrap();
    let upper = distribution
        .tail_mean(probability(0.0), TailSide::Upper)
        .unwrap();
    assert!((lower.mean() - 3.0).abs() < 1.0e-14);
    assert!((upper.mean() - 3.0).abs() < 1.0e-14);
    assert!(
        matches ! (& (distribution . tail_mean (probability (0.0) , TailSide :: Lower)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    assert!(
        matches ! (& (distribution . tail_mean (probability (1.0) , TailSide :: Upper)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
}

#[test]
fn weighted_tail_means_audit_fractional_tied_boundaries() {
    let samples = [(10.0_f64, 1.0), (2.0, 1.0), (1.0, 0.5), (2.0, 0.5)];
    let distribution = RawWeightedEmpiricalDistribution::new(&samples).unwrap();
    let lower = distribution
        .tail_mean(probability(0.5), TailSide::Lower)
        .unwrap();
    assert!((lower.mean() - 5.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(lower.tail_mass(), Some(1.5));
    assert_eq!(lower.boundary_value(), 2.0);
    assert_eq!(lower.boundary_mass(), Some(1.0));
    assert_eq!(lower.boundary_contribution(), Some(2.0));

    let upper = distribution
        .tail_mean(probability(0.5), TailSide::Upper)
        .unwrap();
    assert!((upper.mean() - 22.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(upper.boundary_value(), 2.0);
    assert_eq!(upper.boundary_mass(), Some(0.5));
    assert_eq!(upper.boundary_contribution(), Some(1.0));
}

#[test]
fn in_place_tail_queries_match_immutable_queries() {
    let values = [9.0_f32, 1.0, 4.0, 2.0, 7.0];
    let immutable = RawEmpiricalDistribution::new(&values).unwrap();
    let mut work = values;
    assert_eq!(
        tail_mean_in_place(&mut work, probability(0.6), TailSide::Lower).unwrap(),
        immutable
            .tail_mean(probability(0.6), TailSide::Lower)
            .unwrap()
    );

    let weighted_values = [(9.0_f32, 0.5), (1.0, 1.0), (4.0, 2.0), (2.0, 0.5)];
    let weighted = RawWeightedEmpiricalDistribution::new(&weighted_values).unwrap();
    let mut weighted_work = weighted_values;
    assert_eq!(
        weighted_tail_mean_in_place(&mut weighted_work, probability(0.4), TailSide::Upper).unwrap(),
        weighted
            .tail_mean(probability(0.4), TailSide::Upper)
            .unwrap()
    );
}
