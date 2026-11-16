//! Exact and published numerical references for online moments and covariance.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_stats::*;

fn exact_integer_moments(deviations: &[i64]) -> (f64, f64, f64) {
    let count = deviations.len() as i128;
    let sum = deviations.iter().copied().map(i128::from).sum::<i128>();
    let squared_sum = deviations
        .iter()
        .copied()
        .map(i128::from)
        .map(|value| value * value)
        .sum::<i128>();
    let centered_numerator = count * squared_sum - sum * sum;
    let mean = sum as f64 / count as f64;
    let population_variance = centered_numerator as f64 / (count * count) as f64;
    let sample_variance = centered_numerator as f64 / (count * (count - 1)) as f64;
    (mean, population_variance, sample_variance)
}

#[test]
fn large_offset_small_variance_matches_exact_two_pass_oracle() {
    let deviations = [-17_i64, -9, -4, -1, 0, 2, 7, 13, 21];
    let (deviation_mean, population_variance, sample_variance) = exact_integer_moments(&deviations);

    let f64_offset = 2.0_f64.powi(40);
    let mut f64_moments = OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    deviations
        .iter()
        .copied()
        .map(|deviation| f64_offset + deviation as f64)
        .try_for_each(|value| f64_moments.observe(value))
        .unwrap();
    let f64_snapshot = f64_moments.snapshot();
    assert!(
        (f64_snapshot.mean().unwrap().unwrap() - (f64_offset + deviation_mean)).abs() <= 5.0e-4
    );
    assert!(
        (f64_snapshot.population_variance().unwrap().unwrap() - population_variance).abs()
            <= 2.0e-3
    );
    assert!((f64_snapshot.sample_variance().unwrap().unwrap() - sample_variance).abs() <= 2.0e-3);

    let f32_offset = 2.0_f32.powi(16);
    let mut f32_moments = OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    deviations
        .iter()
        .copied()
        .map(|deviation| f32_offset + deviation as f32)
        .try_for_each(|value| f32_moments.observe(value))
        .unwrap();
    let f32_snapshot = f32_moments.snapshot();
    assert!(
        (f64::from(f32_snapshot.mean().unwrap().unwrap())
            - (f64::from(f32_offset) + deviation_mean))
            .abs()
            <= 5.0e-2
    );
    assert!(
        (f64::from(f32_snapshot.population_variance().unwrap().unwrap()) - population_variance)
            .abs()
            <= 2.0e-1
    );
    assert!(
        (f64::from(f32_snapshot.sample_variance().unwrap().unwrap()) - sample_variance).abs()
            <= 2.0e-1
    );
}

#[test]
fn published_moment_and_covariance_examples_match() {
    let mut moments = OnlineMoments::new(scioxide_stats::MomentOrder::Second);
    [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        .into_iter()
        .try_for_each(|value| moments.observe(value))
        .unwrap();
    let moments = moments.snapshot();
    assert_eq!(moments.mean().unwrap(), Some(5.0));
    assert_eq!(moments.population_variance().unwrap(), Some(4.0));

    let mut covariance = OnlineCovariance::new(scioxide_stats::MomentOrder::Second);
    [(2.1_f64, 8.0), (2.5, 12.0), (4.0, 14.0), (3.6, 10.0)]
        .into_iter()
        .try_for_each(|(x, y)| covariance.observe(x, y))
        .unwrap();
    let covariance = covariance.snapshot();
    assert!(
        (covariance.sample_covariance().unwrap().unwrap() - 1.533_333_333_333_333_4).abs()
            < 1.0e-14
    );
}

#[test]
fn empirical_quantiles_match_committed_r_type_1_and_type_7_vectors() {
    // Generated independently with R 4.5.3:
    // x <- c(8, -2, 1.5, 1.5, 4, 10)
    // p <- c(0, .1, .25, .5, .75, .9, 1)
    // quantile(x, p, type=1, names=FALSE)
    // quantile(x, p, type=7, names=FALSE)
    let samples = [8.0_f64, -2.0, 1.5, 1.5, 4.0, 10.0];
    let probabilities = [0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    let r_type_1 = [-2.0_f64, -2.0, 1.5, 1.5, 8.0, 10.0, 10.0];
    let r_type_7 = [-2.0_f64, -0.25, 1.5, 2.75, 7.0, 9.0, 10.0];
    let distribution = EmpiricalDistribution::new(&samples, Unweighted).unwrap();

    probabilities
        .into_iter()
        .zip(r_type_1)
        .zip(r_type_7)
        .for_each(|((p, expected_type_1), expected_type_7)| {
            let probability = QuantileProbability::new(p).unwrap();
            let type_1 = distribution
                .quantile(probability, QuantileRule::new(QuantileMethod::NearestRank))
                .unwrap()
                .value();
            let type_7 = distribution
                .quantile(probability, QuantileRule::new(QuantileMethod::Linear))
                .unwrap()
                .value();
            assert_eq!(type_1, expected_type_1, "R type 1 at p={p}");
            assert!(
                (type_7 - expected_type_7).abs() < 1.0e-14,
                "R type 7 at p={p}"
            );
        });
}

#[test]
fn weighted_inverse_ecdf_and_tail_means_match_hand_calculation() {
    let samples = [(10.0_f64, 1.0), (4.0, 0.5), (0.0, 0.25), (4.0, 0.25)];
    let distribution = {
        let pairs = &samples;
        let values: Vec<_> = pairs.iter().map(|&(x, _)| x).collect();
        let masses: Vec<_> = pairs.iter().map(|&(_, w)| w).collect();
        EmpiricalDistribution::new(&values, Weights::<_, Mass>::new(&masses))
    }
    .unwrap();
    for (p, expected) in [
        (0.0, 0.0),
        (0.125, 0.0),
        (0.5, 4.0),
        (0.500_1, 10.0),
        (1.0, 10.0),
    ] {
        let actual = distribution
            .quantile(
                QuantileProbability::new(p).unwrap(),
                QuantileRule::inverse_ecdf(),
            )
            .unwrap()
            .value();
        assert_eq!(actual, expected, "weighted inverse ECDF at p={p}");
    }

    let lower = distribution
        .tail_mean(QuantileProbability::new(0.375).unwrap(), TailSide::Lower)
        .unwrap();
    assert!((lower.mean() - 8.0 / 3.0).abs() < 1.0e-14);
    assert_eq!(lower.tail_mass(), Some(0.75));
    assert_eq!(lower.boundary_value(), 4.0);
    assert_eq!(lower.boundary_mass(), Some(0.5));
    assert_eq!(lower.boundary_contribution(), Some(2.0));

    let upper = distribution
        .tail_mean(QuantileProbability::new(0.625).unwrap(), TailSide::Upper)
        .unwrap();
    assert_eq!(upper.mean(), 10.0);
    assert_eq!(upper.tail_mass(), Some(0.75));
    assert_eq!(upper.boundary_value(), 10.0);
    assert_eq!(upper.boundary_mass(), Some(0.75));
    assert_eq!(upper.boundary_contribution(), Some(7.5));
}
