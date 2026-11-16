//! Deterministic success and rejection tests for the statistical machinery.
#![allow(clippy::unwrap_used)]
use crate::statistics;

#[test]
fn exact_histograms_and_inclusive_boundaries() {
    let mut values = [-1.0, 0.0, 0.0, 0.5, 1.0, 2.0].into_iter();
    assert_eq!(
        statistics::histogram(&[0.0, 1.0], 6, || values.next().unwrap()),
        [3, 2, 1]
    );
    statistics::assert_fit("exact", &[8192, 0, 24576], &[0.25, 0.0, 0.75], 0.0);
    assert_eq!(statistics::pearson(&[10, 20], [10.0, 20.0]), 0.0);
    let mut i = 0;
    statistics::cdf_fit(
        "tied discrete quantiles",
        &[(0, 0.5), (0, 0.5), (1, 1.0)],
        statistics::DRAWS,
        0.0,
        || {
            i += 1;
            i % 2
        },
    );
    statistics::discrete(
        "point mass",
        0,
        0,
        |k| f64::from(k == 0),
        |k| f64::from(k >= 0),
        0.0,
        || 0,
    );
}
#[test]
fn rejects_bias_and_impossible_categories() {
    assert!(
        std::panic::catch_unwind(|| statistics::assert_fit(
            "biased",
            &[16384, 0, 16384],
            &[0.25, 0.5, 0.25],
            0.0
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| statistics::assert_fit(
            "impossible",
            &[1, 32767],
            &[0.0, 1.0],
            0.0
        ))
        .is_err()
    );
}
#[test]
fn rejects_invalid_reference_and_draws() {
    assert!(
        std::panic::catch_unwind(|| statistics::assert_fit("total", &[1, 1], &[0.5, 0.4], 0.0))
            .is_err()
    );
    assert!(std::panic::catch_unwind(|| statistics::histogram(&[0.0], 1, || f64::NAN)).is_err());
    assert!(std::panic::catch_unwind(|| statistics::histogram(&[f64::NAN], 1, || 0.0)).is_err());
    assert!(std::panic::catch_unwind(|| statistics::histogram(&[1.0, 0.0], 1, || 0.0)).is_err());
    let law = scioxide_random::Uniform::new(0.0, 1.0).unwrap();
    assert!(
        std::panic::catch_unwind(|| statistics::continuous("infinity", &law, || f64::INFINITY))
            .is_err()
    );
}
#[test]
fn power_bound_covers_three_percentage_points() {
    let n = statistics::DRAWS as f64;
    const {
        assert!(statistics::PEARSON_ALPHA + statistics::DKW_ALPHA <= 1e-6);
    }
    let band = ((2.0 / statistics::DKW_ALPHA).ln() / (2.0 * n)).sqrt();
    for reference_error in [
        statistics::reference_error::<f64>(),
        statistics::reference_error::<f32>(),
        statistics::reference_error::<f32>() + 1e-5,
    ] {
        let miss = 2.0 * (-2.0 * n * (0.03 - band - 2.0 * reference_error).powi(2)).exp();
        assert!(miss < 0.01, "miss probability {miss}");
    }
}

#[test]
fn detects_local_and_cumulative_bias_separately() {
    // Alternating errors have a small cumulative discrepancy but large Pearson
    // distance. A coherent half-to-half shift has the opposite behavior.
    let probabilities = [1.0 / 16.0; 16];
    let alternating: Vec<_> = (0..16)
        .map(|i| if i % 2 == 0 { 2048 + 350 } else { 2048 - 350 })
        .collect();
    let failure = std::panic::catch_unwind(|| {
        statistics::assert_fit("local", &alternating, &probabilities, 0.0)
    })
    .unwrap_err();
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("Pearson")
    );
    let shifted: Vec<_> = (0..16)
        .map(|i| if i < 8 { 2048 + 120 } else { 2048 - 120 })
        .collect();
    let failure = std::panic::catch_unwind(|| {
        statistics::assert_fit("cumulative", &shifted, &probabilities, 0.0)
    })
    .unwrap_err();
    assert!(failure.downcast_ref::<String>().unwrap().contains("CDF"));
    // Zero can be a rounded tail as well as exact absence. The explicit
    // reference allowance admits a few such observations without division by 0.
    statistics::assert_fit("rounded tail", &[1, 32767], &[0.0, 1.0], 1e-4);
}

#[test]
fn detects_a_missing_permutation() {
    // Missing one of 120 outcomes creates only .00834 CDF error. Individual
    // Bernstein bands reject it even when aggregate Pearson/DKW bands do not.
    let mut counts = [275; 120];
    counts[0] = 0;
    counts[119] = 318;
    let failure = std::panic::catch_unwind(|| {
        statistics::assert_fit("missing permutation", &counts, &[1.0 / 120.0; 120], 0.0)
    })
    .unwrap_err();
    assert!(
        failure
            .downcast_ref::<String>()
            .unwrap()
            .contains("Pearson bin 0")
    );
}
