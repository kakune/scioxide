//! Public detailed-result branch and iteration diagnostics.

use scioxide_special::{
    BetaOptions, GammaOptions, beta_inc_with_options, beta_with_options, gamma_log_p_with_options,
    gamma_p, gamma_p_inv_with_options, gamma_p_with_options, gamma_q_with_options,
    gamma_with_options,
};

#[test]
fn incomplete_gamma_regions_are_named_and_bounded() {
    let cases = [
        (
            gamma_p_with_options(2.3_f64, 0.4, GammaOptions::default()).unwrap(),
            "lower_compensated_series",
        ),
        (
            gamma_q_with_options(2.3_f64, 8.0, GammaOptions::default()).unwrap(),
            "upper_scaled_lentz",
        ),
        (
            gamma_p_with_options(100.0_f64, 100.0, GammaOptions::default()).unwrap(),
            "temme_transition",
        ),
        (
            gamma_q_with_options(4.5_f64, 6.0, GammaOptions::default()).unwrap(),
            "half_integer_upper_finite",
        ),
        (
            gamma_p_with_options(5.0_f64, 5.0, GammaOptions::default()).unwrap(),
            "integer_upper_finite",
        ),
    ];

    cases.into_iter().for_each(|(evaluation, expected_branch)| {
        assert_eq!(evaluation.algorithm().method(), expected_branch);
        assert!(evaluation.counts().iterations() <= 512);
    });
}

/// The fixed Temme band `|x−a| < 0.4a` (for `a >= 100`) keeps work bounded:
/// just outside it the series and fraction need O(1) terms even at `a = 1e10`.
#[test]
fn large_shape_incomplete_gamma_work_is_independent_of_the_shape() {
    for a in [1e3_f64, 1.6e8, 1e10] {
        for (sigma, branch) in [
            (-0.401, "lower_compensated_series"),
            (-0.399, "temme_transition"),
            (-1e-4, "temme_transition"),
            (0.399, "temme_transition"),
            (0.401, "upper_scaled_lentz"),
        ] {
            let evaluation =
                gamma_p_with_options(a, a * (1.0 + sigma), GammaOptions::default()).unwrap();
            assert_eq!(
                evaluation.algorithm().method(),
                branch,
                "a={a}, sigma={sigma}"
            );
            assert!(
                evaluation.counts().iterations() <= 512,
                "a={a}, sigma={sigma}"
            );
        }
    }
    // Inputs that previously exhausted the default budget (Issue #1501).
    for (a, x) in [(3e8, 299_920_325.662_851_8), (1_000_158_114.0, 1e9)] {
        let evaluation = gamma_q_with_options(a, x, GammaOptions::default()).unwrap();
        assert_eq!(evaluation.algorithm().method(), "temme_transition");
    }
    let deep = gamma_log_p_with_options(1e10_f64, 0.99e10, GammaOptions::default()).unwrap();
    assert!(deep.counts().iterations() <= 512);
    let quantile = gamma_p_inv_with_options(1e9_f64, 1e-6, GammaOptions::default()).unwrap();
    assert!(quantile.counts().iterations() <= 512);
    let x = quantile.try_value().unwrap();
    assert!((gamma_p(1e9_f64, x).unwrap() / 1e-6 - 1.0).abs() < 1e-9);
}

#[test]
fn incomplete_beta_regions_are_named_and_bounded() {
    let integer = beta_inc_with_options(5.0_f64, 4.0, 0.5, BetaOptions::default()).unwrap();
    assert_eq!(integer.algorithm().method(), "lower_integer_finite");
    assert_eq!(integer.counts().iterations(), 4);

    // Moderate shapes take the conditioned binary64 tail; larger shapes near
    // the symmetry threshold keep the compensated fraction.
    let conditioned = beta_inc_with_options(2.3_f64, 4.7, 0.2, BetaOptions::default()).unwrap();
    assert_eq!(conditioned.algorithm().method(), "conditioned_scalar_tail");
    assert!(conditioned.counts().iterations() <= 200);

    let lower = beta_inc_with_options(12.3_f64, 14.7, 0.45, BetaOptions::default()).unwrap();
    assert_eq!(lower.algorithm().method(), "lower_scaled_lentz");
    assert!(lower.counts().iterations() <= 200);

    let symmetric = beta_inc_with_options(2.3_f64, 4.7, 0.8, BetaOptions::default()).unwrap();
    assert_eq!(symmetric.algorithm().method(), "conditioned_scalar_tail");
    assert!(symmetric.counts().iterations() <= 200);

    // Beyond the moderate normalization range the endpoint series remains.
    let series = beta_inc_with_options(150.0_f64, 0.5, 0.3, BetaOptions::default()).unwrap();
    assert_eq!(series.algorithm().method(), "lower_power_series");
    assert!(series.counts().iterations() <= 200);
}

#[test]
fn qualified_gamma_dispatch_is_visible_without_changing_the_supported_api() {
    assert_eq!(
        gamma_with_options(5.0_f64, GammaOptions::default())
            .unwrap()
            .algorithm()
            .method(),
        "factorial_table"
    );
    assert_eq!(
        beta_with_options(5.0_f64, 5.0, BetaOptions::default())
            .unwrap()
            .algorithm()
            .method(),
        "small_lattice"
    );
}
