//! Recurrence and tail-conditioning tests for gamma-beta-cdfs work CDF kernels.

use scioxide_random::{Binomial, Cdf, Mass, Poisson};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
    );
}

#[test]
fn poisson_cdf_differences_match_pmf_recurrence() {
    let distribution = Poisson::new(5.0_f64).unwrap();
    let values = (0..=40)
        .map(|k| distribution.cdf(k).unwrap())
        .collect::<Vec<_>>();

    assert_close(values[0], distribution.pmf(0).unwrap(), 2e-15);
    values.windows(2).enumerate().for_each(|(k, pair)| {
        assert!(pair[0] <= pair[1]);
        assert_close(pair[1] - pair[0], distribution.pmf(k + 1).unwrap(), 3e-14);
    });
}

#[test]
fn binomial_cdf_differences_match_pmf_recurrence() {
    let distribution = Binomial::new(80, 0.3_f64).unwrap();
    let values = (0..=80)
        .map(|k| distribution.cdf(k).unwrap())
        .collect::<Vec<_>>();

    assert_close(values[0], distribution.pmf(0).unwrap(), 2e-15);
    values.windows(2).enumerate().for_each(|(k, pair)| {
        assert!(pair[0] <= pair[1]);
        assert_close(pair[1] - pair[0], distribution.pmf(k + 1).unwrap(), 2e-13);
    });
}

#[test]
fn listed_speed_cases_match_high_precision_oracles() {
    let poisson = Poisson::new(5.0_f64).unwrap();
    assert_close(poisson.cdf(5).unwrap(), 0.615_960_654_833_063_1, 2e-15);

    let binomial = Binomial::new(20, 0.3_f64).unwrap();
    assert_close(binomial.cdf(5).unwrap(), 0.416_370_829_447_481_2, 2e-15);
}

#[test]
fn f32_short_tail_paths_remain_monotone_and_normalized() {
    let poisson = Poisson::new(5.0_f32).unwrap();
    let poisson_values = (0..=40)
        .map(|k| poisson.cdf(k).unwrap())
        .collect::<Vec<_>>();
    assert!(poisson_values.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!((poisson_values[5] - 0.615_960_66).abs() <= 2e-6);

    let binomial = Binomial::new(20, 0.3_f32).unwrap();
    let binomial_values = (0..=20)
        .map(|k| binomial.cdf(k).unwrap())
        .collect::<Vec<_>>();
    assert!(binomial_values.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!((binomial_values[5] - 0.416_370_84).abs() <= 2e-6);
}
