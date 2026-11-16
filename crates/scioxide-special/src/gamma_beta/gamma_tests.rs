//! Unit tests for gamma kernels.

use super::*;
use crate::gamma_beta::incomplete_gamma::*;
use crate::gamma_p_inv;
use crate::{binomial, binomial_with_options, factorial, ln_factorial};

// ---- ln_gamma ----

#[test]
fn test_ln_gamma_positive_integer_f64() {
    // Γ(5) = 24, ln(24)
    let val = ln_gamma(5.0_f64).unwrap();
    assert!((val - 24.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn test_ln_gamma_positive_integer_f32() {
    let val = ln_gamma(5.0_f32).unwrap();
    assert!((val - 24.0_f32.ln()).abs() < 1e-5);
}

#[test]
fn test_ln_gamma_half_f64() {
    // Γ(0.5) = √π, ln(√π) = 0.5 * ln(π)
    let val = ln_gamma(0.5_f64).unwrap();
    let expected = 0.5 * std::f64::consts::PI.ln();
    assert!((val - expected).abs() < 1e-12);
}

#[test]
fn test_ln_gamma_negative_half_f64() {
    // Γ(-0.5) = -2√π, ln|Γ(-0.5)| = ln(2√π) = ln(2) + 0.5*ln(π)
    let val = ln_gamma(-0.5_f64).unwrap();
    let expected = (2.0 * std::f64::consts::PI.sqrt()).ln();
    assert!((val - expected).abs() < 1e-12);
}

#[test]
fn test_ln_gamma_pole_zero() {
    let result = ln_gamma(0.0_f64);
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
fn test_ln_gamma_pole_neg_integer() {
    let result = ln_gamma(-1.0_f64);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

// ---- gamma ----

#[test]
fn test_gamma_5_f64() {
    let val = gamma(5.0_f64).unwrap();
    assert!((val - 24.0).abs() < 1e-10);
}

#[test]
fn test_gamma_half_f64() {
    let val = gamma(0.5_f64).unwrap();
    let expected = std::f64::consts::PI.sqrt();
    assert!((val - expected).abs() < 1e-12);
}

#[test]
fn test_gamma_preserves_negative_sign() {
    let value = gamma(-0.5_f64).unwrap();
    let expected = -2.0 * std::f64::consts::PI.sqrt();
    assert!((value - expected).abs() < 1e-12);
}

#[test]
fn test_gamma_overflow_is_numerical_instability() {
    assert!(ln_gamma(172.0_f64).unwrap().is_finite());
    {
        let outcome = &(gamma(172.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}

#[test]
fn test_gamma_5_f32() {
    let val = gamma(5.0_f32).unwrap();
    assert!((val - 24.0_f32).abs() < 1e-4);
}

// ---- gamma logarithmic representation ----

#[test]
fn test_gamma_logarithmic_positive() {
    let report = crate::gamma_with_options(5.0_f64, GammaOptions::default()).unwrap();
    let value = report.logarithmic().unwrap().value();
    let ln_abs = value.log_magnitude().unwrap();
    let sign = value.sign();
    assert!((ln_abs - 24.0_f64.ln()).abs() < 1e-12);
    assert!((sign - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_gamma_logarithmic_negative_half() {
    // Γ(-0.5) = -2√π < 0, sign = -1
    let report = crate::gamma_with_options(-0.5_f64, GammaOptions::default()).unwrap();
    let value = report.logarithmic().unwrap().value();
    let ln_abs = value.log_magnitude().unwrap();
    let sign = value.sign();
    let expected = (2.0 * std::f64::consts::PI.sqrt()).ln();
    assert!((ln_abs - expected).abs() < 1e-12);
    assert!((sign - (-1.0)).abs() < f64::EPSILON);
}

// ---- digamma ----

#[test]
fn test_digamma_1_f64() {
    // ψ(1) = -γ (Euler-Mascheroni) ≈ -0.5772156649
    let val = digamma(1.0_f64).unwrap();
    assert!((val - (-0.577_215_664_901_532_9)).abs() < 1e-10);
}

#[test]
fn test_digamma_half_f64() {
    // ψ(0.5) = -γ - 2ln(2) ≈ -1.9635100260
    let val = digamma(0.5_f64).unwrap();
    let expected = (-2.0_f64).mul_add(2.0_f64.ln(), -0.577_215_664_901_532_9);
    assert!((val - expected).abs() < 1e-10);
}

#[test]
fn test_digamma_large_f64() {
    // ψ(10) ≈ 2.25175258906
    let val = digamma(10.0_f64).unwrap();
    assert!((val - 2.251_752_589_066_721).abs() < 1e-10);
}

#[test]
fn test_digamma_pole_zero() {
    let result = digamma(0.0_f64);
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
fn test_digamma_distinguishes_subnormal_from_pole() {
    let finite = digamma(f64::MIN_POSITIVE).unwrap();
    assert!(finite.is_finite() && finite < 0.0);
    {
        let outcome = &(digamma(f64::from_bits(1)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}

// ---- polygamma ----

#[test]
fn test_polygamma_0_is_digamma() {
    let psi0 = polygamma(0, 2.0_f64).unwrap();
    let psi = digamma(2.0_f64).unwrap();
    assert!((psi0 - psi).abs() < 1e-12);
}

#[test]
fn test_polygamma_1_f64() {
    // ψ'(1) = π²/6 ≈ 1.6449340668
    let val = polygamma(1, 1.0_f64).unwrap();
    let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    assert!((val - expected).abs() < 1e-12);
}

#[test]
fn polygamma_continues_negative_nonintegers_without_factorial_overflow() {
    let trigamma = polygamma(1, -0.5_f64).unwrap();
    let expected_trigamma = std::f64::consts::PI.powi(2) / 2.0 + 4.0;
    assert!((trigamma - expected_trigamma).abs() < 2.0e-12);

    let second = polygamma(2, -0.5_f64).unwrap();
    let expected_second = -0.828_796_644_234_319_9_f64;
    assert!((second - expected_second).abs() < 2.0e-12);
}

#[test]
fn polygamma_poles_are_machine_readable() {
    {
        let outcome = &(polygamma(3, -2.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Pole,
                ..
            }
        ));
    }
}

#[test]
fn detailed_gamma_retains_overflowing_values_in_scaled_and_log_forms() {
    let result = crate::gamma_with_options(172.0_f64, GammaOptions::default()).unwrap();
    assert_eq!(
        result.scalar_representation(),
        crate::ScalarRepresentation::Overflow
    );
    assert!(result.value().is_none());
    assert!(result.scaled().is_some());
    assert!(result.logarithmic().is_some());
}

// ---- factorial ----

#[test]
fn test_factorial_small_f64() {
    let val = factorial::<f64>(5).unwrap();
    assert!((val - 120.0).abs() < 1e-10);
}

#[test]
fn test_factorial_zero() {
    let val = factorial::<f64>(0).unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_factorial_20_f64() {
    let val = factorial::<f64>(20).unwrap();
    assert!((val - 2_432_902_008_176_640_000.0).abs() < 1e2);
}

#[test]
fn test_factorial_large() {
    // n = 25 should use gamma path
    let val = factorial::<f64>(25).unwrap();
    assert!(val > 1e25);
    assert!(val.is_finite());
}

#[test]
fn test_factorial_small_f32() {
    let val = factorial::<f32>(5).unwrap();
    assert!((val - 120.0_f32).abs() < 0.1);
}

// ---- ln_factorial ----

#[test]
fn test_ln_factorial_5() {
    let val = ln_factorial::<f64>(5).unwrap();
    assert!((val - 120.0_f64.ln()).abs() < 1e-12);
}

// ---- binomial ----

#[test]
fn test_binomial_10_3() {
    let val = binomial::<f64>(10, 3).unwrap();
    assert!((val - 120.0).abs() < 1e-8);
}

#[test]
fn test_binomial_edge_cases() {
    assert!((binomial::<f64>(5, 0).unwrap() - 1.0).abs() < 1e-10);
    assert!((binomial::<f64>(5, 5).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_binomial_k_greater_than_n() {
    {
        let outcome = &(binomial::<f64>(3, 5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }
}

#[test]
fn test_binomial_f32() {
    let val = binomial::<f32>(10, 3).unwrap();
    assert!((val - 120.0_f32).abs() < 0.1);
}

#[test]
fn test_binomial_avoids_representable_intermediate_overflow() {
    let central_f64 = binomial::<f64>(1024, 512).unwrap();
    let expected_f64 = 4.481_254_552_098_971e306_f64;
    assert!(((central_f64 - expected_f64) / expected_f64).abs() < 8e-14);

    let central_f32 = binomial::<f32>(128, 64).unwrap();
    let expected_f32 = 2.395_114_6e37_f32;
    assert!(((central_f32 - expected_f32) / expected_f32).abs() < 4e-6);
}

#[test]
fn test_binomial_large_indices_use_bounded_logarithmic_work() {
    let excessive = 20_000_usize;
    let report = binomial_with_options::<f64>(
        excessive.saturating_mul(2),
        excessive,
        crate::CombinatorialOptions::default(),
    )
    .unwrap();
    assert_eq!(report.counts().iterations(), 39);
    assert_eq!(
        report.scalar_representation(),
        crate::ScalarRepresentation::Overflow
    );
    assert!(report.logarithmic().is_some());
}

// ---- gamma_p / gamma_q ----

#[test]
fn test_gamma_p_plus_q_equals_one() {
    let a = 2.5_f64;
    let x = 3.0_f64;
    let p = gamma_p(a, x).unwrap();
    let q = gamma_q(a, x).unwrap();
    assert!((p + q - 1.0).abs() < 1e-12);
}

#[test]
fn test_gamma_p_known_value_f64() {
    // P(1, 1) = 1 - e^{-1} ≈ 0.6321205588
    let val = gamma_p(1.0_f64, 1.0).unwrap();
    let expected = 1.0 - (-1.0_f64).exp();
    assert!((val - expected).abs() < 1e-12);
}

#[test]
fn test_gamma_p_zero_x() {
    let val = gamma_p(2.0_f64, 0.0).unwrap();
    assert!(val.abs() < 1e-15);
}

#[test]
fn test_gamma_p_preserves_positive_subnormal_input() {
    let x = f64::from_bits(1);
    let value = gamma_p(0.5_f64, x).unwrap();
    assert!(value > 0.0);
    assert!(value.is_finite());
}

#[test]
fn test_gamma_q_zero_x() {
    let val = gamma_q(2.0_f64, 0.0).unwrap();
    assert!((val - 1.0).abs() < 1e-15);
}

#[test]
fn test_gamma_p_domain_error() {
    {
        let outcome = &(gamma_p(-1.0_f64, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(gamma_p(1.0_f64, -1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_gamma_p_large_x_f64() {
    // P(3, 20) should be close to 1
    let val = gamma_p(3.0_f64, 20.0).unwrap();
    assert!(val > 0.999);
}

#[test]
fn test_gamma_p_large_shape_converges() {
    let p = gamma_p(1_001.0_f64, 1_000.0).unwrap();
    let q = gamma_q(1_001.0_f64, 1_000.0).unwrap();
    assert!((0.0..=1.0).contains(&p));
    assert!((p + q - 1.0).abs() < 1e-12);
}

#[test]
fn test_gamma_p_f32() {
    let p = gamma_p(2.0_f32, 3.0).unwrap();
    let q = gamma_q(2.0_f32, 3.0).unwrap();
    assert!((p + q - 1.0_f32).abs() < 1e-5);
}

#[test]
fn test_incomplete_gamma_iteration_budget_failure() {
    // The lower series and the upper fraction both charge every term.
    for x in [0.4, 8.0] {
        let outcome = log_tail(
            2.3,
            x,
            &Normalization::wide(2.3),
            TailArithmetic::Mixed,
            &mut crate::support::Work::new(0),
        );
        let err = outcome.as_ref().err().expect("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}

// ---- gamma_lower / gamma_upper ----

#[test]
fn test_gamma_lower_plus_upper_equals_gamma() {
    let a = 3.0_f64;
    let x = 2.0_f64;
    let lower = gamma_lower(a, x).unwrap();
    let upper = gamma_upper(a, x).unwrap();
    let g = gamma(a).unwrap();
    assert!((lower + upper - g).abs() < 1e-10);
}

#[test]
fn test_unregularized_gamma_uses_direct_log_scale() {
    assert!(gamma(200.0_f64).is_err());

    let lower = gamma_lower(200.0_f64, 1.0).unwrap();
    let upper = gamma_upper(200.0_f64, 1000.0).unwrap();
    assert!(lower.is_finite() && lower > 0.0);
    assert!(upper.is_finite() && upper > 0.0);
}

// ---- gamma_p_inv ----

#[test]
fn test_gamma_p_inv_roundtrip_f64() {
    let a = 3.0_f64;
    let x = 2.5_f64;
    let p = gamma_p(a, x).unwrap();
    let x_inv = gamma_p_inv(a, p).unwrap();
    assert!((x_inv - x).abs() < 1e-8);
}

#[test]
fn test_gamma_p_inv_boundary() {
    let val = gamma_p_inv(2.0_f64, 0.0).unwrap();
    assert!(val.abs() < 1e-15);
}

#[test]
fn test_gamma_p_inv_domain_error() {
    {
        let outcome = &(gamma_p_inv(1.0_f64, -0.1));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(gamma_p_inv(1.0_f64, 1.5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

#[test]
fn test_gamma_p_inv_non_finite_input() {
    {
        let outcome = &(gamma_p_inv(f64::NAN, 0.5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }
    {
        let outcome = &(gamma_p_inv(1.0_f64, f64::INFINITY));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}
