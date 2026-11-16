//! Unit tests for hypergeometric kernels.

use super::*;
use scioxide_core::RealScalar;

// =====================================================================
// hyp0f1 tests
// =====================================================================

#[test]
fn test_hyp0f1_zero_argument_f64() {
    // 0F1(;b;0) = 1 for any valid b
    let val = hyp0f1(2.0_f64, 0.0).unwrap();
    assert!((val - 1.0).abs() < 1e-15);
}

#[test]
fn test_hyp0f1_zero_argument_f32() {
    let val = hyp0f1(2.0_f32, 0.0).unwrap();
    assert_eq!(val.to_bits(), 1.0_f32.to_bits());
}

#[test]
fn test_hyp0f1_zero_argument_preserves_validation_precedence() {
    {
        let outcome = &(hyp0f1(0.0_f64, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(hyp0f1(f64::NAN, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    let delta = 1e-8_f64;
    assert!(hyp0f1(1.5_f64, -delta).unwrap() < 1.0);
    assert!(hyp0f1(1.5_f64, delta).unwrap() > 1.0);
}

#[test]
fn test_hyp0f1_bessel_j0_relation_f64() {
    // 0F1(;1;-x^2/4) = J_0(x) approximately
    // J_0(0) = 1, J_0(1) ~ 0.7651976865579666
    let x = 1.0_f64;
    let val = hyp0f1(1.0, -x * x / 4.0).unwrap();
    let expected = 0.765_197_686_557_966_6_f64;
    assert!(
        (val - expected).abs() < 1e-10,
        "hyp0f1 Bessel J0 relation: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp0f1_bessel_j0_relation_f32() {
    let x = 1.0_f32;
    let val = hyp0f1(1.0_f32, -x * x / 4.0).unwrap();
    let expected = 0.765_197_7_f32;
    assert!(
        (val - expected).abs() < 1e-5,
        "hyp0f1 Bessel J0 relation f32: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp0f1_bessel_j0_at_larger_x_f64() {
    // J_0(3) ~ -0.2600519549019334
    let x = 3.0_f64;
    let val = hyp0f1(1.0, -x * x / 4.0).unwrap();
    let expected = -0.260_051_954_901_933_4_f64;
    assert!(
        (val - expected).abs() < 1e-10,
        "hyp0f1 Bessel J0(3): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp0f1_positive_x_f64() {
    // 0F1(;2;1) — test with positive argument
    // Manual series: 1 + 1/(2*1!) + 1/(2*3 * 2!) + 1/(2*3*4 * 3!) + ...
    // = sinh(2)/2 via the identity 0F1(;3/2;x/4) relation, but let's just
    // check against a known numerical value.
    // 0F1(;2;1) = I_1(2) * 1 / (1)^{1/2} ... actually let's use a known value.
    // Using relation: 0F1(;n+1;-z^2/4) = (z/2)^{-n} * Gamma(n+1) * J_n(z)
    // For b=2: 0F1(;2;x) = Gamma(2) * I_0(2*sqrt(x)) / (sqrt(x))^0 if we
    // expand carefully. Let's just test numerical consistency.
    let val = hyp0f1(2.0_f64, 1.0).unwrap();
    // Known: 0F1(;2;1) = sum 1/((2)_k * k!) = 1 + 1/2 + 1/12 + 1/144 + ...
    // = 1 + 0.5 + 0.08333... + 0.006944... + ... ~ 1.59637...
    // Actually, sinh(2)/1 = 3.6268..., but that's a different relation.
    // Let's compute manually:
    // k=0: 1
    // k=1: 1/(2*1) = 0.5
    // k=2: 1/(2*3 * 2) = 1/12 = 0.08333
    // k=3: 1/(2*3*4 * 6) = 1/144 = 0.00694
    // k=4: 1/(2*3*4*5 * 24) = 1/2880 = 0.000347
    // sum ~ 1.59062
    // More precisely ~ 1.59063...
    assert!(val > 1.59);
    assert!(val < 1.60);
}

#[test]
fn test_hyp0f1_domain_error_nonpositive_integer_b() {
    let result = hyp0f1(0.0_f64, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    let result = hyp0f1(-1.0_f64, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    let result = hyp0f1(-2.0_f64, 0.5);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_hyp0f1_nonfinite_input() {
    let result = hyp0f1(f64::NAN, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    let result = hyp0f1(1.0_f64, f64::INFINITY);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}

// =====================================================================
// hyp1f1 tests
// =====================================================================

#[test]
fn test_hyp1f1_identity_exp_f64() {
    // 1F1(1, 1, x) = e^x
    for &x in &[0.0, 0.5, 1.0, 2.0, 5.0, -1.0, -3.0] {
        let val = hyp1f1(1.0_f64, 1.0, x).unwrap();
        let expected = x.exp();
        assert!(
            (val - expected).abs() < 1e-10 * expected.abs().max(1.0),
            "1F1(1,1,{x}): got {val}, expected {expected}"
        );
    }
}

#[test]
fn test_hyp1f1_identity_exp_f32() {
    let x = 1.0_f32;
    let val = hyp1f1(1.0_f32, 1.0, x).unwrap();
    let expected = x.exp();
    assert!(
        (val - expected).abs() < 1e-5,
        "1F1(1,1,{x}) f32: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp1f1_zero_argument() {
    // 1F1(a, b, 0) = 1
    let val = hyp1f1(2.5_f64, 3.5, 0.0).unwrap();
    assert!((val - 1.0).abs() < 1e-15);
}

#[test]
fn test_hyp1f1_exact_zero_terminating_term() {
    assert_eq!(hyp1f1(-1.0_f64, 2.0, 2.0).unwrap(), 0.0);
    assert_eq!(hyp1f1(-1.0_f32, 2.0, 2.0).unwrap(), 0.0);
}

#[test]
fn test_hyp1f1_scalar_native_identities_are_exact() {
    assert_eq!(
        hyp1f1(0.5_f32, 1.5, 0.0).unwrap().to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(
        hyp1f1(0.0_f32, 1.5, 5.0).unwrap().to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(
        hyp1f1(1.0_f32, 1.0, 1.0).unwrap().to_bits(),
        1.0_f32.exp().to_bits()
    );
}

#[test]
fn test_hyp1f1_identities_preserve_validation_and_neighborhoods() {
    {
        let outcome = &(hyp1f1(0.0_f64, 0.0, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(hyp1f1(f64::NAN, f64::NAN, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    let delta = 1e-8_f64;
    assert!(hyp1f1(0.5, 1.5, -delta).unwrap() < 1.0);
    assert!(hyp1f1(0.5, 1.5, delta).unwrap() > 1.0);
    let nearby = hyp1f1_with_options(
        1.0_f64.next_down(),
        1.0,
        1.0,
        HypergeometricOptions::default(),
    )
    .unwrap();
    assert_ne!(nearby.algorithm().method(), "exponential_identity");
    // Arb at 256 bits rounds this neighboring value to the same f64 as e.
    assert_eq!(nearby.try_value().unwrap(), 1.0_f64.exp());
    assert_ne!(hyp1f1(1.0_f64.next_up(), 1.0, 1.0).unwrap(), 1.0_f64.exp());
}

#[test]
fn test_hyp1f1_kummer_transformation_large_x_f64() {
    // For large x, Kummer transformation should kick in.
    // 1F1(1, 1, 50) = e^50
    let x = 50.0_f64;
    let val = hyp1f1(1.0, 1.0, x).unwrap();
    let expected = x.exp();
    // Use log-space comparison for large values
    let log_val = val.ln();
    let log_expected = expected.ln();
    assert!(
        (log_val - log_expected).abs() < 1e-8,
        "1F1(1,1,{x}) large: log got {log_val}, log expected {log_expected}"
    );
}

#[test]
fn test_hyp1f1_negative_x_f64() {
    // 1F1(1, 1, -2) = e^{-2}
    let val = hyp1f1(1.0_f64, 1.0, -2.0).unwrap();
    let expected = (-2.0_f64).exp();
    assert!(
        (val - expected).abs() < 1e-12,
        "1F1(1,1,-2): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp1f1_a_equals_b() {
    // 1F1(a, a, x) = e^x for any valid a
    let val = hyp1f1(3.5_f64, 3.5, 2.0).unwrap();
    let expected = 2.0_f64.exp();
    assert!(
        (val - expected).abs() < 1e-10,
        "1F1(3.5,3.5,2): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp1f1_domain_error_b() {
    let result = hyp1f1(1.0_f64, 0.0, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    let result = hyp1f1(1.0_f64, -1.0, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_hyp1f1_nonfinite_input() {
    let result = hyp1f1(f64::NAN, 1.0, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    let result = hyp1f1(1.0_f64, 1.0, f64::INFINITY);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}

#[test]
fn test_hyp1f1_known_value_f64() {
    // 1F1(0.5, 1.5, -1) = sqrt(pi)/2 * erf(1)
    // erf(1) ~ 0.8427007929497149
    // sqrt(pi)/2 ~ 0.8862269254527580
    // product ~ 0.7468241328...
    let val = hyp1f1(0.5_f64, 1.5, -1.0).unwrap();
    let expected = 0.746_824_132_812_427_f64;
    assert!(
        (val - expected).abs() < 1e-10,
        "1F1(0.5, 1.5, -1): got {val}, expected {expected}"
    );
}

// =====================================================================
// hyp2f1 tests
// =====================================================================

#[test]
fn test_hyp2f1_identity_power_f64() {
    // 2F1(a, b; b; x) = (1-x)^{-a}
    let a = 2.0_f64;
    let b_val = 3.0_f64;
    for &x in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, -0.5, -1.0] {
        if x >= 1.0 {
            continue;
        }
        let val = hyp2f1(a, b_val, b_val, x).unwrap();
        let expected = (1.0 - x).powf(-a);
        let rel_err = (val - expected).abs() / expected.abs().max(1.0);
        assert!(
            rel_err < 1e-8,
            "2F1({a},{b_val};{b_val};{x}): got {val}, expected {expected}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_hyp2f1_identity_power_f32() {
    let a = 2.0_f32;
    let b_val = 3.0_f32;
    let x = 0.3_f32;
    let val = hyp2f1(a, b_val, b_val, x).unwrap();
    let expected = (1.0_f32 - x).powf(-a);
    assert!(
        (val - expected).abs() < 1e-4,
        "2F1 power identity f32: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_zero_argument() {
    // 2F1(a, b; c; 0) = 1
    let val = hyp2f1(1.0_f64, 2.0, 3.0, 0.0).unwrap();
    assert!((val - 1.0).abs() < 1e-15);
}

#[test]
fn test_hyp2f1_special_cases_are_stable_and_symmetric() {
    let x = 0.3_f64;
    let cancel_a = hyp2f1(0.5, 0.75, 0.5, x).unwrap();
    let cancel_b = hyp2f1(0.75, 0.5, 0.5, x).unwrap();
    let expected = (-0.75 * (-x).ln_1p()).exp();
    let tolerance = 4.0 * f64::EPSILON * expected.abs();
    assert!((cancel_a - expected).abs() <= tolerance);
    assert!((cancel_b - expected).abs() <= tolerance);

    let logarithmic = hyp2f1(1.0, 1.0, 2.0, x).unwrap();
    let logarithmic_expected = -(-x).ln_1p() / x;
    assert!(
        (logarithmic - logarithmic_expected).abs()
            <= 4.0 * f64::EPSILON * logarithmic_expected.abs()
    );
    assert_eq!(
        hyp2f1(0.5_f32, 0.5, 1.5, 0.0).unwrap().to_bits(),
        1.0_f32.to_bits()
    );
}

#[test]
fn test_hyp2f1_special_cases_preserve_validation_precedence() {
    {
        let outcome = &(hyp2f1(0.0_f64, 1.0, 0.0, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(hyp2f1(0.5_f64, 0.75, 0.5, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(hyp2f1(1.0_f64, 1.0, 2.0, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(hyp2f1(f64::NAN, 1.0, 2.0, 0.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}

#[test]
fn test_hyp2f1_logarithmic_identity_has_two_sided_zero_limit() {
    for x in [-f64::EPSILON, f64::EPSILON] {
        let actual = hyp2f1(1.0_f64, 1.0, 2.0, x).unwrap();
        // Verify the shared scalar identity in either feature configuration.
        // The portable and host log1p leaves may differ by one ULP; the
        // independent zero-limit bound below applies to both.
        let expected = -RealScalar::ln_1p(-x) / x;
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert!((actual - 1.0).abs() <= f64::EPSILON);
    }
}

#[test]
fn test_hyp2f1_pfaff_transformation_region_f64() {
    // Test in the x > 0.5 region where Pfaff transformation is used
    // 2F1(1, 1; 2; x) = -ln(1-x) / x
    let x = 0.7_f64;
    let val = hyp2f1(1.0, 1.0, 2.0, x).unwrap();
    let expected = -(1.0 - x).ln() / x;
    assert!(
        (val - expected).abs() < 1e-10,
        "2F1(1,1;2;{x}): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_log_identity_f64() {
    // 2F1(1, 1; 2; x) = -ln(1-x)/x for various x
    for &x in &[0.1, 0.3, 0.5, 0.8, 0.95] {
        let val = hyp2f1(1.0_f64, 1.0, 2.0, x).unwrap();
        let expected = -(1.0 - x).ln() / x;
        let rel_err = (val - expected).abs() / expected.abs().max(1.0);
        assert!(
            rel_err < 1e-8,
            "2F1(1,1;2;{x}): got {val}, expected {expected}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_hyp2f1_negative_x_f64() {
    // 2F1(a, b; b; x) = (1-x)^{-a} also for negative x
    let a = 1.5_f64;
    let x = -0.8;
    let val = hyp2f1(a, 4.0, 4.0, x).unwrap();
    let expected = (1.0 - x).powf(-a);
    assert!(
        (val - expected).abs() < 1e-10,
        "2F1 negative x: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_domain_error_c_nonpositive_integer() {
    let result = hyp2f1(1.0_f64, 1.0, 0.0, 0.5);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    let result = hyp2f1(1.0_f64, 1.0, -1.0, 0.5);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_hyp2f1_domain_error_x_ge_1() {
    let result = hyp2f1(1.0_f64, 1.0, 2.0, 1.0);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    let result = hyp2f1(1.0_f64, 1.0, 2.0, 1.5);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_hyp2f1_nonfinite_input() {
    let result = hyp2f1(f64::NAN, 1.0, 2.0, 0.5);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}

#[test]
fn test_hyp2f1_terminating_series_f64() {
    // When a is a non-positive integer, the series terminates.
    // 2F1(-2, 1; 1; x) = 1 - 2x + x^2 = (1-x)^2
    let x = 0.7_f64;
    let val = hyp2f1(-2.0, 1.0, 1.0, x).unwrap();
    let expected = (1.0 - x) * (1.0 - x);
    assert!(
        (val - expected).abs() < 1e-12,
        "2F1(-2,1;1;{x}): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_reciprocal_gamma_zero_in_continuation() {
    let value = hyp2f1(2.5_f64, 0.25, 1.5, 0.75).unwrap();
    assert!((value - 2.121_320_343_559_642_4).abs() < 3e-13);
}

#[test]
fn test_hyp2f1_extreme_negative_transform_avoids_rounded_endpoint() {
    let value = hyp2f1(0.25_f64, 1.75, 2.5, -1.0e300).unwrap();
    let expected = 1.131_370_849_898_476e-75;
    assert!(value.is_finite() && value > 0.0);
    assert!(((value - expected) / expected).abs() < 3e-13);
}

#[test]
fn test_hyp2f1_integer_continuation_work_is_bounded() {
    let n = 70_000;
    let c = n as f64 + 1.0;
    let result = hyp2f1_with_options(
        0.5_f64,
        0.5,
        c,
        0.75,
        HypergeometricOptions::default().with_iteration_budget(scioxide_core::Budget::ONE),
    )
    .unwrap_err();
    let result_counts = crate::test_support::exhausted_counts(&result);
    assert_eq!(result_counts.iterations(), 1);
}

#[test]
fn large_integer_connections_and_shifts_use_the_options_allowance() {
    use crate::hypergeometric_family::{hypergeometric_complex, hypergeometric_real};
    use crate::support::wide::Wide;
    for limit in [1, 3] {
        let mut work = Work::new(limit);
        let real = hypergeometric_real::integer_reciprocal(
            Wide::from(0.5),
            Wide::from(70_000.5),
            Wide::from(1.25),
            -2.0,
            &mut work,
        )
        .err()
        .unwrap();
        assert_eq!(
            crate::test_support::exhausted_counts(&real).iterations(),
            limit
        );

        let mut work = Work::new(limit);
        let complex = hypergeometric_complex::near_one(
            Jet::from(0.5),
            Jet::from(0.5),
            Jet::from(70_001.0),
            Jet::constant(C::new(0.75, 0.01)),
            true,
            &mut work,
        )
        .err()
        .unwrap();
        assert_eq!(
            crate::test_support::exhausted_counts(&complex).iterations(),
            limit
        );

        let mut work = Work::new(limit);
        let shifted = kernel::hyperu(
            Jet::from(-70_000.5),
            Jet::from(1.25),
            Jet::constant(C::new(1.0, 0.25)),
            &mut work,
        )
        .err()
        .unwrap();
        assert_eq!(
            crate::test_support::exhausted_counts(&shifted).iterations(),
            limit
        );
    }
}

#[test]
fn integer_continuation_indices_reject_unrepresentable_conversions() {
    use crate::hypergeometric_family::{hypergeometric_complex, hypergeometric_real};
    use crate::support::wide::Wide;
    let real = hypergeometric_real::integer_reciprocal(
        Wide::from(0.5),
        Wide::from(1e100),
        Wide::from(1.25),
        -2.0,
        &mut Work::new(1),
    )
    .err()
    .unwrap();
    let complex = hypergeometric_complex::near_one(
        Jet::from(0.0),
        Jet::from(0.0),
        Jet::from(1e100),
        Jet::constant(C::new(0.75, 0.01)),
        true,
        &mut Work::new(1),
    )
    .err()
    .unwrap();
    let shifted = kernel::hyperu(
        Jet {
            v: C::new(-1e100, 0.0),
            d: C::new(1.0, 0.0),
        },
        Jet::from(1.25),
        Jet::constant(C::new(1.0, 0.25)),
        &mut Work::new(1),
    )
    .err()
    .unwrap();
    for error in [real, complex, shifted] {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ScalarConversion
        );
    }
}

#[test]
fn test_hyp2f1_terminating_b_negative_integer() {
    // 2F1(1, -1; 2; x) = 1 - x/2
    let x = 0.6_f64;
    let val = hyp2f1(1.0, -1.0, 2.0, x).unwrap();
    let expected = 1.0 - x / 2.0;
    assert!(
        (val - expected).abs() < 1e-12,
        "2F1(1,-1;2;{x}): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_gauss_sum_at_x_approaching_1_f64() {
    // 2F1(a, b; c; 1) = Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b))
    // when c-a-b > 0 (Gauss summation theorem).
    // Test at x very close to 1.
    let a = 0.5_f64;
    let b_val = 0.5_f64;
    let c = 2.0_f64;
    let x = 0.999_f64;
    let val = hyp2f1(a, b_val, c, x).unwrap();

    // Expected at x=1: Gamma(2)*Gamma(1) / (Gamma(1.5)*Gamma(1.5))
    //   = 1 / (Gamma(1.5))^2 = 1 / (sqrt(pi)/2)^2 = 4/pi
    let expected_at_1 = 4.0 / std::f64::consts::PI;
    // At x=0.999 should be close to 4/pi
    assert!(
        (val - expected_at_1).abs() < 0.01,
        "2F1 near 1: got {val}, expected ~{expected_at_1}"
    );
}

#[test]
fn test_hyp2f1_near_boundary_x_f64() {
    // Test at x = 0.5 (boundary between direct and Pfaff)
    // 2F1(1, 1; 2; 0.5) = -ln(0.5)/0.5 = 2*ln(2) ~ 1.38629...
    let val = hyp2f1(1.0_f64, 1.0, 2.0, 0.5).unwrap();
    let expected = 2.0 * 2.0_f64.ln();
    assert!(
        (val - expected).abs() < 1e-10,
        "2F1 at x=0.5: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_identity_power_f32_negative_x() {
    let a = 1.0_f32;
    let x = -0.5_f32;
    let val = hyp2f1(a, 2.0, 2.0, x).unwrap();
    let expected = (1.0_f32 - x).powf(-a);
    assert!(
        (val - expected).abs() < 1e-4,
        "2F1 f32 neg x: got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp2f1_extended_f32_oracle_cases() {
    const RELATIVE_BUDGET: f32 = 1.0e-3;
    const ABSOLUTE_BUDGET: f32 = 2.0e-5;
    let cases = [
        (
            [0x3f98_4b50, 0x3f98_8951, 0x4048_f231, 0xc051_24b9],
            0x3ee7_8521,
        ),
        (
            [0x3f59_36d8, 0x3f06_c28f, 0x4074_4ec3, 0xbefd_239a],
            0x3f72_fb6e,
        ),
        (
            [0x3f6d_63d7, 0x3fd8_8a3e, 0x4097_81d9, 0x3f54_2bb1],
            0x3fbc_d675,
        ),
        (
            [0x3f08_ed3a, 0x3f98_82c4, 0x401a_f5a3, 0xbe09_b3a6],
            0x3f77_7e2e,
        ),
    ];

    for (arguments, expected_bits) in cases {
        let [a, b, c, x] = arguments.map(f32::from_bits);
        let expected = f32::from_bits(expected_bits);
        let actual = hyp2f1(a, b, c, x).unwrap();
        let error = (actual - expected).abs();
        let allowed = ABSOLUTE_BUDGET.max(RELATIVE_BUDGET * expected.abs());
        assert!(
            error <= allowed,
            "2F1({a},{b};{c};{x}): got {actual}, expected {expected}, error={error}, allowed={allowed}"
        );
    }
}

// =====================================================================
// Cross-function and mathematical relation tests
// =====================================================================

#[test]
fn test_hyp0f1_small_b_f64() {
    // 0F1(;0.5; x) is related to cosh(2*sqrt(x)) for x > 0
    // 0F1(;1/2; x) = cosh(2*sqrt(x))
    let x = 1.0_f64;
    let val = hyp0f1(0.5, x).unwrap();
    let expected = (2.0 * x.sqrt()).cosh();
    assert!(
        (val - expected).abs() < 1e-10,
        "0F1(;0.5;1) = cosh(2): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp0f1_sinh_relation_f64() {
    // 0F1(;3/2; x) = sinh(2*sqrt(x)) / (2*sqrt(x)) for x > 0
    let x = 1.0_f64;
    let two_sqrt_x = 2.0 * x.sqrt();
    let val = hyp0f1(1.5, x).unwrap();
    let expected = two_sqrt_x.sinh() / two_sqrt_x;
    assert!(
        (val - expected).abs() < 1e-10,
        "0F1(;3/2;1): got {val}, expected {expected}"
    );
}

#[test]
fn test_hyp1f1_large_negative_x_f64() {
    // 1F1(1, 1, -10) = e^{-10}
    let val = hyp1f1(1.0_f64, 1.0, -10.0).unwrap();
    let expected = (-10.0_f64).exp();
    assert!(
        (val - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "1F1(1,1,-10): got {val}, expected {expected}"
    );
}
