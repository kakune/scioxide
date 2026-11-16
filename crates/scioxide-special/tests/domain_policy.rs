//! Cross-family tests for the public domain and non-finite-value policy.

#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_core::{DomainReason, NonFiniteOrigin, NumericalIssue, SciOxideError};
use scioxide_special::*;

fn assert_non_finite_input<T>(result: Result<T, SciOxideError>) {
    {
        let outcome = &(result);
        let Err(err) = outcome else {
            panic!("expected error");
        };
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

fn assert_domain_reason<T>(result: Result<T, SciOxideError>, expected: DomainReason) {
    let Err(error) = result else {
        panic!("expected domain error");
    };
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DomainError { reason, .. } if reason == expected
    ));
}

#[test]
fn checked_apis_reject_non_finite_inputs_at_the_boundary() {
    let nan = f64::NAN;

    assert_non_finite_input(airy_ai(nan));
    assert_non_finite_input(airy_bi(nan));
    assert_non_finite_input(airy_ai_scaled(nan));
    assert_non_finite_input(airy_bi_scaled(nan));

    assert_non_finite_input(bessel_j(nan, 1.0));
    assert_non_finite_input(bessel_y(0.0, nan));
    assert_non_finite_input(bessel_i(nan, 1.0));
    assert_non_finite_input(bessel_i_scaled(0.0, nan));
    assert_non_finite_input(bessel_k(nan, 1.0));
    assert_non_finite_input(bessel_k_scaled(0.0, nan));
    assert_non_finite_input(hankel_1(0.0, nan));
    assert_non_finite_input(hankel_2(nan, 1.0));

    assert_non_finite_input(ln_beta(nan, 1.0));
    assert_non_finite_input(beta(1.0, nan));
    assert_non_finite_input(beta_inc(1.0, 1.0, nan));
    assert_non_finite_input(beta_inc_inv(1.0, nan, 0.5));

    assert_non_finite_input(elliptic_k(nan));
    assert_non_finite_input(elliptic_e(nan));
    assert_non_finite_input(elliptic_f(nan, 0.5));
    assert_non_finite_input(elliptic_e_inc(0.5, nan));
    assert_non_finite_input(elliptic_pi(nan, 0.5));

    assert_non_finite_input(erf(nan));
    assert_non_finite_input(erfc(f64::INFINITY));
    assert_non_finite_input(erfinv(nan));
    assert_non_finite_input(erfcinv(f64::NEG_INFINITY));

    assert_non_finite_input(ln_gamma(nan));
    assert_non_finite_input(gamma(f64::INFINITY));
    assert_non_finite_input(gamma_with_options(nan, GammaOptions::default()));
    assert_non_finite_input(digamma(nan));
    assert_non_finite_input(polygamma(1, nan));
    assert_non_finite_input(gamma_p(nan, 1.0));
    assert_non_finite_input(gamma_q(1.0, nan));
    assert_non_finite_input(gamma_lower(nan, 1.0));
    assert_non_finite_input(gamma_upper(1.0, nan));
    assert_non_finite_input(gamma_p_inv(nan, 0.5));

    assert_non_finite_input(hyp0f1(nan, 1.0));
    assert_non_finite_input(hyp1f1(1.0, nan, 1.0));
    assert_non_finite_input(hyp2f1(1.0, 1.0, 2.0, nan));

    assert_non_finite_input(legendre_p(1, nan));
    assert_non_finite_input(legendre_p_assoc(1, 0, nan));
    assert_non_finite_input(legendre_q(1, nan));

    assert_non_finite_input(riemann_zeta(nan));
    assert_non_finite_input(dirichlet_eta(nan));
    assert_non_finite_input(hurwitz_zeta(2.0, nan));
}

#[test]
fn poles_and_unsupported_real_regions_are_distinct() {
    assert_domain_reason(gamma(0.0_f64), DomainReason::Pole);
    assert_domain_reason(digamma(-1.0_f64), DomainReason::Pole);
    assert_domain_reason(beta(0.0_f64, 1.0), DomainReason::Pole);
    assert!((gamma_p(0.0_f64, 1.0).unwrap() - 1.0).abs() < f64::EPSILON);
    assert_domain_reason(bessel_y(0.0_f64, 0.0), DomainReason::Pole);
    assert_domain_reason(bessel_j(-0.5_f64, 0.0), DomainReason::Pole);
    assert_domain_reason(bessel_i(-0.5_f64, 0.0), DomainReason::Pole);
    assert_domain_reason(riemann_zeta(1.0_f64), DomainReason::Pole);
    assert_domain_reason(hyp0f1(0.0_f64, 1.0), DomainReason::Pole);
    assert_domain_reason(hyp1f1(1.0_f64, -1.0, 1.0), DomainReason::Pole);

    assert!(polygamma(1, -0.5_f64).unwrap().is_finite());
    assert_domain_reason(bessel_j(0.5_f64, -1.0), DomainReason::OutsideDomain);
    assert_domain_reason(hyp2f1(1.0_f64, 1.0, 2.0, 1.0), DomainReason::OutsideDomain);
    assert!(hurwitz_zeta(0.5_f64, 1.0).unwrap().is_finite());
}

#[test]
fn endpoint_infinities_are_values_but_finite_overflow_is_an_error() {
    assert_eq!(erfinv(1.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(erfinv(-1.0_f64).unwrap(), f64::NEG_INFINITY);
    assert_eq!(erfcinv(0.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(erfcinv(2.0_f64).unwrap(), f64::NEG_INFINITY);
    assert_eq!(gamma_p_inv(2.0_f64, 1.0).unwrap(), f64::INFINITY);
    assert_eq!(elliptic_k(1.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(elliptic_pi(1.0_f64, 0.5).unwrap(), f64::INFINITY);
    assert_eq!(legendre_q(0, 1.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(legendre_q(0, -1.0_f64).unwrap(), f64::NEG_INFINITY);
    assert_eq!(bessel_k(0.0_f64, 0.0).unwrap(), f64::INFINITY);

    {
        let outcome = &(gamma(172.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
    {
        let outcome = &(airy_bi(1_000.0_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
    assert!(airy_bi_scaled(1_000.0_f64).unwrap().is_finite());
    assert!(bessel_i_scaled(0.0_f64, 1_000.0).unwrap().is_finite());
}
