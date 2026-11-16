//! Unit tests for elliptic kernels.

use super::*;
use crate::{elliptic_e, elliptic_e_inc, elliptic_f, elliptic_k, elliptic_pi};

#[test]
fn test_carlson_rf_reports_iteration_exhaustion() {
    {
        let outcome = &(carlson_rf_with_work(0.0_f64, 0.5, 1.0, &mut Work::new(0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}

#[test]
fn test_carlson_rc_reports_iteration_exhaustion() {
    {
        let outcome = &(carlson_rf_with_work(0.5_f64, 1.0, 1.0, &mut Work::new(0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}

#[test]
fn test_shared_carlson_reports_iteration_exhaustion() {
    {
        let outcome = &(carlson_rf_rd_with_work(0.0_f64, 0.5, 1.0, &mut Work::new(0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
    {
        let outcome =
            &(carlson_rf_rj_combination(0.0_f64, 0.5, 1.0, 0.25, 0.75, &mut Work::new(0)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
}

#[test]
fn test_complete_near_one_minimax_f64() {
    let m = 1.0_f64 - 2.0_f64.powi(-40);
    let k = elliptic_k_evaluate_with_work(m, &mut Work::new(0)).unwrap();
    let e = elliptic_e_evaluate_with_work(m, &mut Work::new(0)).unwrap();
    assert_eq!(k.branch_id.as_str(), "complete_minimax");
    assert_eq!(e.branch_id.as_str(), "complete_minimax");
    assert_eq!((k.iterations, e.iterations), (0, 0));
    assert!((k.value - 15.249_237_972_322_037).abs() < 1e-13);
    assert!((e.value - 1.000_000_000_006_707).abs() < 1e-14);
}

#[test]
fn test_complete_near_one_minimax_f32() {
    let m = 1.0_f32 - 2.0_f32.powi(-16);
    let k = elliptic_k(m).unwrap();
    let e = elliptic_e(m).unwrap();
    assert!((k - 6.931_494).abs() < 2e-5);
    assert!((e - 1.000_049_1).abs() < 2e-6);
}

#[test]
fn test_regime_oracles_f64() {
    let near_one = 1.0_f64 - 2.0_f64.powi(-40);
    for (actual, expected, relative_tolerance) in [
        (
            elliptic_f(0.7, 0.5).unwrap(),
            0.728_770_305_718_190_2,
            5e-14,
        ),
        (
            elliptic_f(1.4, near_one).unwrap(),
            2.457_995_590_365_781,
            5e-14,
        ),
        (
            elliptic_e_inc(0.7, 0.5).unwrap(),
            0.673_189_174_547_128_8,
            5e-14,
        ),
        (
            elliptic_e_inc(1.4, near_one).unwrap(),
            0.985_449_729_989_129_8,
            5e-14,
        ),
        (elliptic_pi(0.5, 0.5).unwrap(), 2.701_287_762_095_351, 5e-14),
        (
            elliptic_pi(0.5, near_one).unwrap(),
            29.252_025_464_375_894,
            5e-12,
        ),
        (
            elliptic_pi(near_one, 0.5).unwrap(),
            2_329_349.362_669_148,
            5e-12,
        ),
    ] {
        assert!((actual - expected).abs() <= relative_tolerance * expected.abs());
    }
}

#[test]
fn negative_complete_kernels_are_fixed_cost() {
    use crate::{EllipticOptions, elliptic_e_with_options, elliptic_k_with_options};
    let options = EllipticOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
    for m in [-f64::from_bits(1), -0.5, -1.0, -1e300, -1e308, -f64::MAX] {
        for report in [
            elliptic_k_with_options(m, options),
            elliptic_e_with_options(m, options),
        ] {
            let report = report.unwrap();
            assert!(report.termination().is_success());
            assert_eq!(report.counts().iterations(), 0);
            assert!(report.try_value().unwrap().is_finite());
        }
    }
}

#[test]
fn test_simultaneous_near_corner_uses_shared_carlson() {
    let n = 1.0_f64 - 2.0_f64.powi(-39);
    let m = 1.0_f64 - 2.0_f64.powi(-40);
    let evaluation = elliptic_pi_evaluate(n, m).unwrap();
    let expected = 685_243_398_264.924_7;
    assert!(
        (evaluation.value - expected).abs() <= 5e-11 * expected,
        "joint-corner result was {:.17e}",
        evaluation.value
    );
    assert_eq!(evaluation.branch_id.as_str(), "elliptic_pi.carlson_rf_rj");
    assert!(evaluation.iterations < MAX_ITER);
}

#[test]
fn test_amplitude_reduction_quadrants_and_periods() {
    let m = 0.5_f64;
    let quarter = std::f64::consts::FRAC_PI_4;
    let three_quarters = 3.0 * quarter;
    let five_halves = 5.0 * std::f64::consts::FRAC_PI_2;
    let k = elliptic_k(m).unwrap();
    let e = elliptic_e(m).unwrap();
    let f_quarter = elliptic_f(quarter, m).unwrap();
    let e_quarter = elliptic_e_inc(quarter, m).unwrap();

    assert!((elliptic_f(three_quarters, m).unwrap() - (2.0 * k - f_quarter)).abs() < 2e-13);
    assert!((elliptic_e_inc(three_quarters, m).unwrap() - (2.0 * e - e_quarter)).abs() < 2e-13);
    assert!((elliptic_f(five_halves, m).unwrap() - 5.0 * k).abs() < 2e-13);
    assert!((elliptic_e_inc(five_halves, m).unwrap() - 5.0 * e).abs() < 2e-13);
    assert!((elliptic_f(-five_halves, m).unwrap() + 5.0 * k).abs() < 2e-13);
    assert!((elliptic_e_inc(-five_halves, m).unwrap() + 5.0 * e).abs() < 2e-13);
}

#[test]
fn test_exact_pi_identities() {
    let n = 0.25_f64;
    let m = 0.5_f64;
    let pi_m_zero = std::f64::consts::FRAC_PI_2 / (1.0 - n).sqrt();
    assert!((elliptic_pi(n, 0.0).unwrap() - pi_m_zero).abs() < 2e-14);
    assert!((elliptic_pi(m, m).unwrap() - elliptic_e(m).unwrap() / (1.0 - m)).abs() < 2e-14);
}

#[test]
fn test_all_iterative_paths_are_bounded() {
    for evaluation in [
        elliptic_k_evaluate(0.5_f64).unwrap(),
        elliptic_e_evaluate(0.5).unwrap(),
        elliptic_f_evaluate(0.7, 0.5).unwrap(),
        elliptic_e_inc_evaluate(0.7, 0.5).unwrap(),
        elliptic_pi_evaluate(0.5, 0.25).unwrap(),
    ] {
        assert!(evaluation.iterations < MAX_ITER);
    }
}

// ---- elliptic_k ----

#[test]
fn test_elliptic_k_zero_f64() {
    // K(0) = pi/2
    let val = elliptic_k(0.0_f64).unwrap();
    assert!(
        (val - std::f64::consts::FRAC_PI_2).abs() < 1e-14,
        "K(0) should be pi/2, got {val}"
    );
}

#[test]
fn test_elliptic_k_zero_f32() {
    let val = elliptic_k(0.0_f32).unwrap();
    assert!(
        (val - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
        "K(0) should be pi/2 (f32), got {val}"
    );
}

#[test]
fn test_elliptic_k_half_f64() {
    // K(0.5) ~ 1.854074677301372
    let val = elliptic_k(0.5_f64).unwrap();
    let expected = 1.854_074_677_301_372;
    assert!(
        (val - expected).abs() < 1e-12,
        "K(0.5) expected {expected}, got {val}"
    );
}

#[test]
fn test_elliptic_k_small_f64() {
    // K(0.01) slightly larger than pi/2
    let val = elliptic_k(0.01_f64).unwrap();
    assert!(val > std::f64::consts::FRAC_PI_2);
    assert!(val < 1.60);
}

#[test]
fn test_elliptic_k_near_one_f64() {
    // K(0.99) is large (logarithmic singularity as m -> 1)
    let val = elliptic_k(0.99_f64).unwrap();
    assert!(val > 3.0, "K(0.99) should be > 3.0, got {val}");
}

#[test]
fn test_elliptic_k_negative_parameter_continuation() {
    assert!((elliptic_k(-1.0_f64).unwrap() - 1.311_028_777_146_059_8).abs() < 2.0e-14);
}

#[test]
fn test_elliptic_k_infinite_endpoint() {
    assert_eq!(elliptic_k(1.0_f64).unwrap(), f64::INFINITY);
}

#[test]
fn test_elliptic_k_domain_error_greater_than_one() {
    let result = elliptic_k(1.5_f64);
    {
        let outcome = &(result);
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}

// ---- elliptic_e ----

#[test]
fn test_elliptic_e_zero_f64() {
    // E(0) = pi/2
    let val = elliptic_e(0.0_f64).unwrap();
    assert!(
        (val - std::f64::consts::FRAC_PI_2).abs() < 1e-14,
        "E(0) should be pi/2, got {val}"
    );
}

#[test]
fn test_elliptic_e_zero_f32() {
    let val = elliptic_e(0.0_f32).unwrap();
    assert!(
        (val - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
        "E(0) should be pi/2 (f32), got {val}"
    );
}

#[test]
fn test_elliptic_e_one_f64() {
    // E(1) = 1
    let val = elliptic_e(1.0_f64).unwrap();
    assert!((val - 1.0).abs() < 1e-14, "E(1) should be 1.0, got {val}");
}

#[test]
fn test_elliptic_e_one_f32() {
    let val = elliptic_e(1.0_f32).unwrap();
    assert!(
        (val - 1.0_f32).abs() < 1e-6,
        "E(1) should be 1.0 (f32), got {val}"
    );
}

#[test]
fn test_elliptic_e_half_f64() {
    // E(0.5) ~ 1.3506438810476755
    let val = elliptic_e(0.5_f64).unwrap();
    let expected = 1.350_643_881_047_675_5;
    assert!(
        (val - expected).abs() < 1e-12,
        "E(0.5) expected {expected}, got {val}"
    );
}

#[test]
fn test_elliptic_e_negative_parameter_continuation() {
    assert!((elliptic_e(-1.0_f64).unwrap() - 1.910_098_894_513_856).abs() < 2.0e-14);
}

#[test]
fn test_elliptic_e_domain_error_greater_than_one() {
    let result = elliptic_e(1.5_f64);
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
fn test_elliptic_e_decreasing_in_m() {
    // E(m) is a decreasing function of m
    let e1 = elliptic_e(0.2_f64).unwrap();
    let e2 = elliptic_e(0.5_f64).unwrap();
    let e3 = elliptic_e(0.8_f64).unwrap();
    assert!(e1 > e2, "E(0.2) = {e1} should be > E(0.5) = {e2}");
    assert!(e2 > e3, "E(0.5) = {e2} should be > E(0.8) = {e3}");
}

// ---- elliptic_f (incomplete first kind) ----

#[test]
fn test_elliptic_f_zero_phi() {
    let val = elliptic_f(0.0_f64, 0.5).unwrap();
    assert!(val.abs() < 1e-15, "F(0|m) should be 0, got {val}");
}

#[test]
fn test_elliptic_f_zero_m() {
    // F(phi | 0) = phi
    let phi = 1.0_f64;
    let val = elliptic_f(phi, 0.0).unwrap();
    assert!(
        (val - phi).abs() < 1e-14,
        "F(phi|0) should be phi, got {val}"
    );
}

#[test]
fn test_elliptic_f_at_half_pi_equals_k_f64() {
    // F(pi/2 | m) = K(m)
    let m = 0.5_f64;
    let f_val = elliptic_f(std::f64::consts::FRAC_PI_2, m).unwrap();
    let k_val = elliptic_k(m).unwrap();
    assert!(
        (f_val - k_val).abs() < 1e-12,
        "F(pi/2|{m}) = {f_val}, K({m}) = {k_val}"
    );
}

#[test]
fn test_elliptic_f_at_half_pi_equals_k_f32() {
    let m = 0.5_f32;
    let f_val = elliptic_f(std::f32::consts::FRAC_PI_2, m).unwrap();
    let k_val = elliptic_k(m).unwrap();
    assert!(
        (f_val - k_val).abs() < 1e-4,
        "F(pi/2|{m}) f32 = {f_val}, K({m}) = {k_val}"
    );
}

#[test]
fn test_elliptic_f_negative_phi() {
    // F(-phi|m) = -F(phi|m)
    let phi = 0.8_f64;
    let m = 0.5;
    let f_pos = elliptic_f(phi, m).unwrap();
    let f_neg = elliptic_f(-phi, m).unwrap();
    assert!(
        (f_pos + f_neg).abs() < 1e-14,
        "F(-phi|m) should be -F(phi|m)"
    );
}

#[test]
fn test_elliptic_f_domain_error() {
    assert!(elliptic_f(1.0_f64, -0.1).unwrap().is_finite());

    let result = elliptic_f(1.0_f64, 1.5);
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
fn test_elliptic_f_m_one_limits_and_crossed_pole() {
    let half_pi = std::f64::consts::FRAC_PI_2;
    assert_eq!(elliptic_f(half_pi, 1.0).unwrap(), f64::INFINITY);
    assert_eq!(elliptic_f(-half_pi, 1.0).unwrap(), f64::NEG_INFINITY);
    assert!((elliptic_f(0.7_f64, 1.0).unwrap() - 0.7_f64.sin().atanh()).abs() < 2e-14);
    {
        let outcome = &(elliptic_f(half_pi + 0.25, 1.0));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        ));
    }
}

#[test]
fn test_elliptic_f_known_value_f64() {
    // F(1.0 | 0.5) ~ 1.08322...
    // Verified via Carlson R_F: sin(1) * R_F(cos^2(1), 1-0.5*sin^2(1), 1)
    let val = elliptic_f(1.0_f64, 0.5).unwrap();
    let expected = 1.083_216_772_845_2;
    assert!(
        (val - expected).abs() < 1e-10,
        "F(1.0|0.5) expected ~{expected}, got {val}"
    );
}

// ---- elliptic_e_inc (incomplete second kind) ----

#[test]
fn test_elliptic_e_inc_zero_phi() {
    let val = elliptic_e_inc(0.0_f64, 0.5).unwrap();
    assert!(val.abs() < 1e-15, "E(0|m) should be 0, got {val}");
}

#[test]
fn test_elliptic_e_inc_zero_m() {
    // E(phi | 0) = phi
    let phi = 1.0_f64;
    let val = elliptic_e_inc(phi, 0.0).unwrap();
    assert!(
        (val - phi).abs() < 1e-14,
        "E(phi|0) should be phi, got {val}"
    );
}

#[test]
fn test_elliptic_e_inc_at_half_pi_equals_e_f64() {
    // E(pi/2 | m) = E(m)
    let m = 0.5_f64;
    let e_inc = elliptic_e_inc(std::f64::consts::FRAC_PI_2, m).unwrap();
    let e_com = elliptic_e(m).unwrap();
    assert!(
        (e_inc - e_com).abs() < 1e-12,
        "E(pi/2|{m}) = {e_inc}, E({m}) = {e_com}"
    );
}

#[test]
fn test_elliptic_e_inc_at_half_pi_equals_e_f32() {
    let m = 0.5_f32;
    let incomplete = elliptic_e_inc(std::f32::consts::FRAC_PI_2, m).unwrap();
    assert!((incomplete - elliptic_e(m).unwrap()).abs() < 2e-5);
}

#[test]
fn test_elliptic_e_inc_m_one_f64() {
    // E(phi | 1) = sin(phi)
    let phi = 0.7_f64;
    let val = elliptic_e_inc(phi, 1.0).unwrap();
    let expected = phi.sin();
    assert!(
        (val - expected).abs() < 1e-12,
        "E({phi}|1) should be sin({phi}) = {expected}, got {val}"
    );
}

#[test]
fn test_elliptic_e_inc_negative_phi() {
    // E(-phi|m) = -E(phi|m)
    let phi = 0.8_f64;
    let m = 0.5;
    let e_pos = elliptic_e_inc(phi, m).unwrap();
    let e_neg = elliptic_e_inc(-phi, m).unwrap();
    assert!(
        (e_pos + e_neg).abs() < 1e-14,
        "E(-phi|m) should be -E(phi|m)"
    );
}

#[test]
fn test_elliptic_e_inc_domain_error() {
    assert!(elliptic_e_inc(1.0_f64, -0.1).unwrap().is_finite());

    let result = elliptic_e_inc(1.0_f64, 1.5);
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
fn test_elliptic_e_inc_known_value_f64() {
    // E(1.0 | 0.5) ~ 0.927329883624444 (verified via Simpson's rule, N=100000)
    let val = elliptic_e_inc(1.0_f64, 0.5).unwrap();
    let expected = 0.927_329_883_624_444;
    assert!(
        (val - expected).abs() < 1e-12,
        "E(1.0|0.5) expected ~{expected}, got {val}"
    );
}

// ---- elliptic_pi ----

#[test]
fn test_elliptic_pi_n_zero_equals_k_f64() {
    // Pi(0 | m) = K(m)
    let m = 0.5_f64;
    let pi_val = elliptic_pi(0.0, m).unwrap();
    let k_val = elliptic_k(m).unwrap();
    assert!(
        (pi_val - k_val).abs() < 1e-12,
        "Pi(0|{m}) = {pi_val}, K({m}) = {k_val}"
    );
}

#[test]
fn test_elliptic_pi_n_zero_equals_k_f32() {
    let m = 0.5_f32;
    let pi_val = elliptic_pi(0.0_f32, m).unwrap();
    let k_val = elliptic_k(m).unwrap();
    assert!(
        (pi_val - k_val).abs() < 1e-4,
        "Pi(0|{m}) f32 = {pi_val}, K({m}) = {k_val}"
    );
}

#[test]
fn test_elliptic_pi_infinite_n_endpoint() {
    assert_eq!(elliptic_pi(1.0_f64, 0.5).unwrap(), f64::INFINITY);
}

#[test]
fn test_elliptic_pi_negative_parameter_continuation() {
    assert!(elliptic_pi(0.5_f64, -0.1).unwrap().is_finite());
}

#[test]
fn test_elliptic_pi_preserves_validation_precedence() {
    {
        let outcome = &(elliptic_pi(f64::NAN, -0.5));
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
        let outcome = &(elliptic_pi(2.0, f64::NAN));
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
        let outcome = &(elliptic_pi(2.0, -0.5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        ));
    }
    {
        let outcome = &(elliptic_pi(2.0, 0.5));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError {
                reason: scioxide_core::DomainReason::Undefined,
                ..
            }
        ));
    }
}

#[test]
fn test_elliptic_pi_infinite_m_endpoint() {
    assert_eq!(elliptic_pi(0.5_f64, 1.0).unwrap(), f64::INFINITY);
}

#[test]
fn test_elliptic_pi_known_value_f64() {
    // Pi(0.5 | 0.5) ~ 3.234773...
    let val = elliptic_pi(0.5_f64, 0.5).unwrap();
    assert!(
        val > 2.5 && val < 4.0,
        "Pi(0.5|0.5) should be in [2.5, 4.0], got {val}"
    );
}

#[test]
fn test_elliptic_pi_negative_n_f64() {
    // Pi(n|m) with n < 0 should yield a finite positive result
    let val = elliptic_pi(-0.5_f64, 0.5).unwrap();
    assert!(val.is_finite());
    assert!(val > 0.0, "Pi(-0.5|0.5) should be positive, got {val}");
}

#[test]
fn test_elliptic_pi_extreme_negative_characteristic_is_scaled() {
    let value = elliptic_pi(-1.0e300_f64, 0.5).unwrap();
    let expected = core::f64::consts::FRAC_PI_2 * 1.0e-150;
    assert!(value.is_finite() && value > 0.0);
    assert!(((value - expected) / expected).abs() < 2e-14);
}

#[test]
fn test_incomplete_elliptic_integrals_preserve_signed_zero() {
    assert!(elliptic_f(-0.0_f64, 0.5).unwrap().is_sign_negative());
    assert!(elliptic_e_inc(-0.0_f64, 0.5).unwrap().is_sign_negative());
}

// ---- Legendre relation ----

#[test]
fn test_legendre_relation_f64() {
    // K(m) E(1-m) + E(m) K(1-m) - K(m) K(1-m) = pi/2
    let m = 0.3_f64;
    let mp = 1.0 - m;
    let k_m = elliptic_k(m).unwrap();
    let e_m = elliptic_e(m).unwrap();
    let k_mp = elliptic_k(mp).unwrap();
    let e_mp = elliptic_e(mp).unwrap();
    let lhs = k_m.mul_add(e_mp - k_mp, e_m * k_mp);
    assert!(
        (lhs - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
        "Legendre relation: got {lhs}, expected pi/2"
    );
}
