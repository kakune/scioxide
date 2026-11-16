//! Independent mpmath references for count-law tails and Hurwitz regressions.
#![allow(clippy::unwrap_used)]
use scioxide_core::Budget;
use scioxide_special::{
    ZetaOptions, hurwitz_zeta, hurwitz_zeta_log, hurwitz_zeta_log_with_options,
};
#[test]
fn relative_small_and_logarithmic_underflow_tails() {
    for (s, q, expected) in [
        (4.0_f64, 1001.0, -21.823_378_250_239_536),
        (200.0, 101.0, -922.873_473_182_593),
        (1.125, 1001.0, 1.215_909_641_573_343_5),
    ] {
        let actual = hurwitz_zeta_log(s, q).unwrap();
        assert!((actual - expected).abs() < 5e-12);
        if expected > -700.0 {
            assert!((hurwitz_zeta(s, q).unwrap() / expected.exp() - 1.0).abs() < 5e-12);
        }
        let actual = hurwitz_zeta_log(s as f32, q as f32).unwrap();
        assert!((f64::from(actual) - expected).abs() < 2e-6 * expected.abs().max(1.0));
    }
    {
        let outcome = &(hurwitz_zeta_log_with_options(
            4.0,
            1001.0,
            ZetaOptions::default().with_iteration_budget(Budget::ONE),
        ));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
    assert!(hurwitz_zeta_log(1.0, 1.0).is_err());
    assert!(hurwitz_zeta_log(2.0, 0.0).is_err());
    assert!(hurwitz_zeta_log(f64::NAN, 1.0).is_err());
}

#[test]
fn reported_hurwitz_regressions_and_range_errors() {
    for (s, a, expected) in [
        (
            28.538253901013725_f64,
            96.76342676815749,
            f64::from_bits(0x344b_2d43_05b5_6b4a),
        ),
        (
            28.927540120581234_f64,
            82.58493219631185,
            f64::from_bits(0x3488_3223_9a28_610d),
        ),
    ] {
        let result =
            scioxide_special::hurwitz_zeta_with_options(s, a, ZetaOptions::default()).unwrap();
        let value = result.value().unwrap();
        // The fixture provides exact references for both reported failures;
        // this loop separately exercises plain/log parity.
        let ordinary = hurwitz_zeta(s, a).unwrap();
        assert_eq!(ordinary, *value.value());
        assert!(
            (hurwitz_zeta_log(s, a).unwrap() - ordinary.ln()).abs()
                <= 8.0 * f64::EPSILON * ordinary.ln().abs()
        );
        assert!((ordinary - expected).abs() <= value.absolute_error());
    }
    // ζ(200,101) ≈ 101⁻²⁰⁰ rounds to zero; ζ(1000,0.1) overflows.
    let tail = scioxide_special::hurwitz_zeta_with_options(200.0_f64, 101., ZetaOptions::default())
        .unwrap();
    assert_eq!(hurwitz_zeta(200.0_f64, 101.).unwrap().to_bits(), 0);
    assert_eq!(tail.try_value().unwrap().to_bits(), 0);
    assert!(tail.value().unwrap().absolute_error() >= f64::from_bits(1));
    let err = hurwitz_zeta(1000.0_f64, 0.1).unwrap_err();
    assert!(matches!(
        err.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    for (s, a) in [(200.0_f64, 101.), (1000., 0.1)] {
        assert!(hurwitz_zeta_log(s, a).unwrap().is_finite());
    }
}

#[test]
fn unrepresentable_hurwitz_log_is_a_numerical_overflow() {
    let error = hurwitz_zeta_log(f64::MAX, f64::MAX).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    let error = hurwitz_zeta_log(f32::MAX, f32::MAX).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
}
