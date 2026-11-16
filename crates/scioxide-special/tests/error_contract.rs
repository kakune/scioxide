//! Typed errors and numerical range regressions.

#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Complex, DomainReason, NonFiniteOrigin, NumericalIssue, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView,
};
use scioxide_special::*;

fn assert_failure<T>(
    result: Result<T, SciOxideError>,
    case: &str,
    kind: SciOxideErrorKind,
) -> SciOxideError {
    let Some(error) = result.err() else {
        panic!("{case}: expected {kind:?}");
    };
    assert_eq!(error.kind(), kind, "{case}: {error}");
    error
}

macro_rules! nonfinite_cases {
    ($scalar:ty, $bad:ident; $($function:ident($($args:expr),*);)*) => {
        for $bad in [<$scalar>::NAN, <$scalar>::INFINITY, <$scalar>::NEG_INFINITY] {
            $(let error = assert_failure(
                $function::<$scalar>($($args),*),
                concat!("scioxide_special::", stringify!($function)),
                SciOxideErrorKind::NonFinite,
            );
            assert!(matches!(error.view(), SciOxideErrorView::NonFinite { origin: NonFiniteOrigin::Input, .. }));)*
        }
    }
}

macro_rules! inferred_nonfinite_cases {
    ($scalar:ty, $bad:ident; $($function:ident($($args:expr),*);)*) => {
        for $bad in [<$scalar>::NAN, <$scalar>::INFINITY, <$scalar>::NEG_INFINITY] {
            $(let error = assert_failure(
                $function($($args),*),
                concat!("scioxide_special::", stringify!($function)),
                SciOxideErrorKind::NonFinite,
            );
            assert!(matches!(error.view(), SciOxideErrorView::NonFinite { origin: NonFiniteOrigin::Input, .. }));)*
        }
    }
}

macro_rules! real_nonfinite_cases {
    ($scalar:ty) => {
        nonfinite_cases!($scalar, bad;
            spherical_harmonic(1, 0, bad, 1.0);
            spherical_harmonic(1, 0, 1.0, bad);
            spherical_harmonic_real(1, 0, bad, 1.0);
            spherical_harmonic_real(1, 0, 1.0, bad);
            debye_1(bad);
            debye_2(bad);
            debye_3(bad);
            debye_4(bad);
            debye_5(bad);
            debye_6(bad);
            fermi_dirac_m1(bad);
            fermi_dirac_0(bad);
            fermi_dirac_1(bad);
            fermi_dirac_2(bad);
            fermi_dirac_mhalf(bad);
            fermi_dirac_half(bad);
            fermi_dirac_3half(bad);
            fermi_dirac_int(1, bad);
            fermi_dirac_inc_0(bad, 1.0);
            fermi_dirac_inc_0(1.0, bad);
            expint_en(1, bad);
            expint_en_scaled(1, bad);
            expint_ei(bad);
            expint_ei_scaled(bad);
            expint_e1(bad);
            expint_e1_scaled(bad);
            expint_e2(bad);
            expint_e2_scaled(bad);
            si(bad);
            ci(bad);
            shi(bad);
            chi(bad);
            shi_scaled(bad);
            chi_scaled(bad);
            dawson(bad);
            fresnel_c(bad);
            fresnel_s(bad);
            atanint(bad);
            expint_3(bad);
            logarithmic_integral(bad);
            logarithmic_integral_offset(bad);
            hyp0f1(bad, 1.0);
            hyp0f1(1.0, bad);
            hyp1f1(bad, 1.0, 1.0);
            hyp1f1(1.0, bad, 1.0);
            hyp1f1(1.0, 1.0, bad);
            hyp2f1(bad, 1.0, 1.0, 1.0);
            hyp2f1(1.0, bad, 1.0, 1.0);
            hyp2f1(1.0, 1.0, bad, 1.0);
            hyp2f1(1.0, 1.0, 1.0, bad);
            hyp2f0(bad, 1.0, 1.0);
            hyp2f0(1.0, bad, 1.0);
            hyp2f0(1.0, 1.0, bad);
            hyperu(bad, 1.0, 1.0);
            hyperu(1.0, bad, 1.0);
            hyperu(1.0, 1.0, bad);
            hyp2f1_regularized(bad, 1.0, 1.0, 1.0);
            hyp2f1_regularized(1.0, bad, 1.0, 1.0);
            hyp2f1_regularized(1.0, 1.0, bad, 1.0);
            hyp2f1_regularized(1.0, 1.0, 1.0, bad);
            gamma_log_p(bad, 1.0);
            gamma_log_p(1.0, bad);
            gamma_log_q(bad, 1.0);
            gamma_log_q(1.0, bad);
            beta_inc_q(bad, 1.0, 1.0);
            beta_inc_q(1.0, bad, 1.0);
            beta_inc_q(1.0, 1.0, bad);
            beta_inc_log(bad, 1.0, 1.0);
            beta_inc_log(1.0, bad, 1.0);
            beta_inc_log(1.0, 1.0, bad);
            beta_inc_log_q(bad, 1.0, 1.0);
            beta_inc_log_q(1.0, bad, 1.0);
            beta_inc_log_q(1.0, 1.0, bad);
            gamma_q_inv(bad, 0.5);
            gamma_q_inv(1.0, bad);
            gamma_p_inv_log(bad, 0.5);
            gamma_p_inv_log(1.0, bad);
            gamma_q_inv_log(bad, 0.5);
            gamma_q_inv_log(1.0, bad);
            beta_inc_q_inv(bad, 1.0, 0.5);
            beta_inc_q_inv(1.0, bad, 0.5);
            beta_inc_q_inv(1.0, 1.0, bad);
            beta_inc_inv_log(bad, 1.0, 0.5);
            beta_inc_inv_log(1.0, bad, 0.5);
            beta_inc_inv_log(1.0, 1.0, bad);
            beta_inc_q_inv_log(bad, 1.0, 0.5);
            beta_inc_q_inv_log(1.0, bad, 0.5);
            beta_inc_q_inv_log(1.0, 1.0, bad);
            dilog(bad);
            spence(bad);
            polylog(bad, 1.0);
            polylog(1.0, bad);
            lerch_phi(bad, 1.0, 1.0);
            lerch_phi(1.0, bad, 1.0);
            lerch_phi(1.0, 1.0, bad);
            spherical_jn(0, bad);
            spherical_yn(0, bad);
            spherical_in(0, bad);
            spherical_kn(0, bad);
            spherical_in_scaled(0, bad);
            spherical_kn_scaled(0, bad);
            riccati_jn(0, bad);
            riccati_yn(0, bad);
            legendre_h3d(0, bad, 0.0);
            legendre_h3d(0, 1.0, bad);
            bessel_j0_integral(bad);
            bessel_y0_integral(bad);
            bessel_i0_integral(bad);
            bessel_k0_integral(bad);
            bessel_j0_complement_integral(bad);
            bessel_y0_tail_integral(bad);
            bessel_i0_remainder_integral(bad);
            bessel_k0_tail_integral(bad);
            struve_h0_integral(bad);
            struve_h0_tail_integral(bad);
            struve_l0_integral(bad);
            coulomb_wave(bad, 0.0, 1.0); coulomb_wave(0.0, bad, 1.0); coulomb_wave(0.0, 0.0, bad);
            coulomb_f(bad, 0.0, 1.0);
            coulomb_f(0.0, bad, 1.0);
            coulomb_f(0.0, 0.0, bad);
            coulomb_g(bad, 0.0, 1.0);
            coulomb_g(0.0, bad, 1.0);
            coulomb_g(0.0, 0.0, bad);
            coulomb_f_prime(bad, 0.0, 1.0);
            coulomb_f_prime(0.0, bad, 1.0);
            coulomb_f_prime(0.0, 0.0, bad);
            coulomb_g_prime(bad, 0.0, 1.0);
            coulomb_g_prime(0.0, bad, 1.0);
            coulomb_g_prime(0.0, 0.0, bad);
            coulomb_spherical_f(bad, 0.0, 1.0);
            coulomb_spherical_f(0.0, bad, 1.0);
            coulomb_spherical_f(0.0, 0.0, bad);
            coulomb_normalization(bad, 0.0);
            coulomb_normalization(0.0, bad);
            hydrogenic_radial(1, 0, bad, 1.0);
            hydrogenic_radial(1, 0, 1.0, bad);
            transport_2(bad);
            transport_3(bad);
            transport_4(bad);
            transport_5(bad);
            airy_ai(bad);
            airy_bi(bad);
            airy_ai_scaled(bad);
            airy_bi_scaled(bad);
            ber(bad);
            bei(bad);
            ker(bad);
            kei(bad);
            berp(bad);
            beip(bad);
            kerp(bad);
            keip(bad);
            kelvin(bad);
            lambert_w0(bad);
            lambert_wm1(bad);
            wright_omega(bad);
            ellipsoidal_harmonic(bad, 4.0, 1, 0, 1.0, scioxide_special::EllipsoidalSigns::default());
            ellipsoidal_harmonic(1.0, bad, 1, 0, 1.0, scioxide_special::EllipsoidalSigns::default());
            ellipsoidal_harmonic(1.0, 4.0, 1, 0, bad, scioxide_special::EllipsoidalSigns::default());
            ellipsoidal_harmonic_second_kind(bad, 4.0, 1, 0, 3.0);
            ellipsoidal_harmonic_second_kind(1.0, bad, 1, 0, 3.0);
            ellipsoidal_harmonic_second_kind(1.0, 4.0, 1, 0, bad);
            ellipsoidal_normalization(bad, 4.0, 1, 0);
            ellipsoidal_normalization(1.0, bad, 1, 0);
            mathieu_a(2, bad);
            mathieu_b(2, bad);
            mathieu_ce(2, bad, 0.5);
            mathieu_ce(2, 1.0, bad);
            mathieu_se(2, bad, 0.5);
            mathieu_se(2, 1.0, bad);
            mathieu_mc(2, bad, 0.5, RadialKind::First);
            mathieu_mc(2, 1.0, bad, RadialKind::First);
            mathieu_ms(2, bad, 0.5, RadialKind::First);
            mathieu_ms(2, 1.0, bad, RadialKind::First);
            spheroidal_characteristic_value(SpheroidalKind::Prolate, 0, 2, bad);
            spheroidal_angular(SpheroidalKind::Prolate, 0, 2, bad, 0.5);
            spheroidal_angular(SpheroidalKind::Prolate, 0, 2, 1.0, bad);
            spheroidal_radial(SpheroidalKind::Oblate, 0, 2, bad, 1.5, RadialKind::First);
            spheroidal_radial(SpheroidalKind::Oblate, 0, 2, 1.0, bad, RadialKind::First);
            ln_gamma(bad);
            gamma(bad);
            digamma(bad);
            polygamma(1, bad);
            gamma_p(bad, 1.0);
            gamma_p(1.0, bad);
            gamma_q(bad, 1.0);
            gamma_q(1.0, bad);
            gamma_lower(bad, 1.0);
            gamma_lower(1.0, bad);
            gamma_upper(bad, 1.0);
            gamma_upper(1.0, bad);
            gamma_p_inv(bad, 0.5);
            gamma_p_inv(1.0, bad);
            struve(bad, 1.0);
            struve(0.0, bad);
            modstruve(bad, 1.0);
            modstruve(0.0, bad);
            normal_cdf(bad);
            normal_sf(bad);
            normal_log_cdf(bad);
            normal_log_sf(bad);
            erfcx(bad);
            log_erfc(bad);
            normal_quantile(bad);
            normal_isf(bad);
            normal_inv_log_cdf(bad);
            normal_inv_log_sf(bad);
            erf(bad);
            erfc(bad);
            owens_t(bad, 1.0);
            owens_t(1.0, bad);
            voigt_complex(bad, 1.0, 1.0);
            voigt_complex(1.0, bad, 1.0);
            voigt_complex(1.0, 1.0, bad);
            voigt_profile(bad, 1.0, 1.0);
            voigt_profile(1.0, bad, 1.0);
            voigt_profile(1.0, 1.0, bad);
            erfi(bad);
            dawson(bad);
            synchrotron_1(bad);
            synchrotron_2(bad);
            synchrotron_1_scaled(bad);
            synchrotron_2_scaled(bad);
            clausen(bad);
            elliptic_k(bad);
            elliptic_e(bad);
            erfinv(bad);
            erfcinv(bad);
            riemann_zeta(bad);
            dirichlet_eta(bad);
            bessel_j(bad, 1.0);
            bessel_j(1.0, bad);
            bessel_y(bad, 1.0);
            bessel_y(1.0, bad);
            bessel_i(bad, 1.0);
            bessel_i(1.0, bad);
            bessel_i_scaled(bad, 1.0);
            bessel_i_scaled(1.0, bad);
            bessel_k(bad, 1.0);
            bessel_k(1.0, bad);
            bessel_k_scaled(bad, 1.0);
            bessel_k_scaled(1.0, bad);
            elliptic_f(bad, 1.0);
            elliptic_f(1.0, bad);
            elliptic_e_inc(bad, 1.0);
            elliptic_e_inc(1.0, bad);
            elliptic_pi(bad, 1.0);
            elliptic_pi(1.0, bad);
            hurwitz_zeta(bad, 1.0);
            hurwitz_zeta(1.0, bad);
            hankel_1(bad, 1.0);
            hankel_1(0.0, bad);
            hankel_2(bad, 1.0);
            hankel_2(0.0, bad);
            legendre_p(1, bad);
            legendre_p_assoc(1, 0, bad);
            legendre_q(1, bad);
            hurwitz_zeta_log(bad, 1.0);
            hurwitz_zeta_log(1.0, bad);
            pochhammer(bad, 1.0);
            pochhammer(1.0, bad);
            pochhammer_relative(bad, 1.0);
            pochhammer_relative(1.0, bad);
            pochhammer(bad, 1.0);
            gamma(bad + 1.0);
            generalized_binomial(bad, 1);
            ln_beta(bad, 1.0);
            ln_beta(1.0, bad);
            beta(bad, 1.0);
            beta(1.0, bad);
            beta_inc(bad, 1.0, 1.0);
            beta_inc(1.0, bad, 1.0);
            beta_inc(1.0, 1.0, bad);
            beta_inc_inv(bad, 1.0, 0.5);
            beta_inc_inv(1.0, bad, 0.5);
            beta_inc_inv(1.0, 1.0, bad);
            parabolic_d(bad, 1.0);
            parabolic_d(0.0, bad);
            parabolic_v(bad, 1.0);
            parabolic_v(0.0, bad);
            parabolic_w(bad, 1.0);
            parabolic_w(0.0, bad);
            jacobi_sn(bad, 0.5 as $scalar);
            jacobi_sn(1.0, bad);
            jacobi_cn(bad, 0.5 as $scalar);
            jacobi_cn(1.0, bad);
            jacobi_dn(bad, 0.5 as $scalar);
            jacobi_dn(1.0, bad);
            jacobi_amplitude(bad, 0.5 as $scalar);
            jacobi_amplitude(1.0, bad);
            hyp_pfq(&[bad], &[1.0], 0.5);
            hyp_pfq(&[1.0], &[bad], 0.5);
            hyp_pfq(&[1.0], &[1.0], bad);
            carlson_rf(bad, 1.0, 1.0);
            carlson_rf(1.0, bad, 1.0);
            carlson_rf(1.0, 1.0, bad);
            carlson_rd(bad, 1.0, 1.0);
            carlson_rd(1.0, bad, 1.0);
            carlson_rd(1.0, 1.0, bad);
            carlson_rj(bad, 1.0, 1.0, 1.0);
            carlson_rj(1.0, bad, 1.0, 1.0);
            carlson_rj(1.0, 1.0, bad, 1.0);
            carlson_rj(1.0, 1.0, 1.0, bad);
            carlson_rc(bad, 1.0);
            carlson_rc(1.0, bad);
            carlson_rg(bad, 1.0, 1.0);
            carlson_rg(1.0, bad, 1.0);
            carlson_rg(1.0, 1.0, bad);
            bessel_lambda(bad, 1.0);
            bessel_lambda(0.0, bad);
            bessel_k_log(bad, 1.0);
            bessel_k_log(0.0, bad);
            bessel_weighted_integral(bad, 1.0, 0.0);
            bessel_weighted_integral(1.0, bad, 0.0);
            bessel_weighted_integral(1.0, 1.0, bad);
            bessel_j_derivative(bad, 1.0, 1);
            bessel_j_derivative(0.0, bad, 1);
            bessel_y_derivative(bad, 1.0, 1);
            bessel_y_derivative(0.0, bad, 1);
            bessel_i_derivative(bad, 1.0, 1);
            bessel_i_derivative(0.0, bad, 1);
            bessel_k_derivative(bad, 1.0, 1);
            bessel_k_derivative(0.0, bad, 1);
            bessel_i_scaled_derivative(bad, 1.0, 1);
            bessel_i_scaled_derivative(0.0, bad, 1);
            bessel_k_scaled_derivative(bad, 1.0, 1);
            bessel_k_scaled_derivative(0.0, bad, 1);
            spherical_hankel_1(0, bad);
            spherical_hankel_2(0, bad);
            hankel_1_derivative(bad, 1.0, 1); hankel_1_derivative(0.0, bad, 1);
            hankel_2_derivative(bad, 1.0, 1); hankel_2_derivative(0.0, bad, 1);
        );
    };
}

#[test]
fn plain_real_nonfinite_errors_f64() {
    real_nonfinite_cases!(f64);
}

#[test]
fn plain_real_nonfinite_errors_f32() {
    real_nonfinite_cases!(f32);
}

fn assert_domain<T>(result: Result<T, SciOxideError>, name: &str, expected: DomainReason) {
    let error = assert_failure(result, name, SciOxideErrorKind::DomainError);
    assert!(
        matches!(error.view(), SciOxideErrorView::DomainError { reason, .. } if reason == expected),
        "{error}"
    );
}

fn assert_overflow<T>(result: Result<T, SciOxideError>, name: &str) {
    let error = assert_failure(result, name, SciOxideErrorKind::NumericalInstability);
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ),
        "{error}"
    );
}

macro_rules! domain_cases {
    ($scalar:ty; $reason:ident; $($function:ident($($args:expr),*);)*) => {
        $(assert_domain($function::<$scalar>($($args),*), concat!("scioxide_special::", stringify!($function)), DomainReason::$reason);)*
    }
}

macro_rules! real_domain_cases {
    ($scalar:ty) => {
        domain_cases!($scalar; Pole;
            gamma(0.0); ln_gamma(0.0); digamma(-1.0); polygamma(1, 0.0);
            beta(0.0, 1.0); ln_beta(0.0, 1.0);
            gamma_lower(0.0, 1.0);
            bessel_y(0.0, 0.0); bessel_j(-0.5, 0.0); bessel_i(-0.5, 0.0); bessel_i_scaled(-0.5, 0.0);
            bessel_k_log(0.0, 0.0); bessel_y_derivative(0.0, 0.0, 1);
            expint_e1(0.0); expint_e1_scaled(0.0); expint_ei(0.0); expint_ei_scaled(0.0);
            hyp0f1(0.0, 1.0); hyp1f1(1.0, -1.0, 1.0); riemann_zeta(1.0);
        );
        domain_cases!($scalar; OutsideDomain;
            gamma_p(-1.0, 1.0); gamma_q(-1.0, 1.0); gamma_lower(-1.0, 1.0); gamma_upper(-1.0, 1.0);
            gamma_p_inv(1.0, -0.5); beta(-1.0, 1.0); ln_beta(-1.0, 1.0);
            beta_inc(1.0, 1.0, -0.5); beta_inc_inv(1.0, 1.0, -0.5); beta_inc_inv(0.0, 1.0, 0.5);
            gamma_log_p(-1.0, 1.0); gamma_log_q(-1.0, 1.0);
            beta_inc_q(-1.0, 1.0, 0.5); beta_inc_log(-1.0, 1.0, 0.5); beta_inc_log_q(-1.0, 1.0, 0.5);
            gamma_q_inv(1.0, -0.5); gamma_p_inv_log(1.0, 0.5); gamma_q_inv_log(1.0, 0.5);
            beta_inc_q_inv(1.0, 1.0, -0.5); beta_inc_inv_log(1.0, 1.0, 0.5); beta_inc_q_inv_log(1.0, 1.0, 0.5);
            bessel_j(0.5, -1.0); bessel_y(1.0, -1.0); bessel_i(0.5, -1.0); bessel_i_scaled(0.5, -1.0);
            bessel_k(1.0, -1.0); bessel_k_scaled(1.0, -1.0); bessel_lambda(-1.0, 1.0);
            hankel_1(0.0, -1.0); hankel_2(0.0, -1.0);
            erfinv(2.0); erfcinv(-1.0); normal_quantile(-0.5); normal_isf(-0.5);
            normal_inv_log_cdf(0.5); normal_inv_log_sf(0.5);
            elliptic_k(2.0); elliptic_e(2.0); elliptic_f(1.0, 2.0); elliptic_e_inc(1.0, 2.0);

            legendre_p_assoc(1, 0, 2.0); legendre_q(1, 2.0);
            hurwitz_zeta(2.0, -0.5); hurwitz_zeta_log(2.0, -0.5);
            lambert_w0(-1.0); lambert_wm1(3.0); spence(-1.0); polylog(0.5, 2.0); lerch_phi(2.0, 0.5, 1.0);
            ci(-2.0); chi(-2.0); chi_scaled(-2.0);
            expint_e2(-1.0); expint_e2_scaled(-1.0); expint_en(2, -1.0); expint_en_scaled(2, -1.0);
            debye_1(-1.0); debye_2(-1.0); debye_3(-1.0); debye_4(-1.0); debye_5(-1.0); debye_6(-1.0);
            transport_2(-1.0); transport_3(-1.0); transport_4(-1.0); transport_5(-1.0);
            synchrotron_1(-1.0); synchrotron_2(-1.0); synchrotron_1_scaled(-1.0); synchrotron_2_scaled(-1.0);
            ker(-1.0); kei(-1.0); kerp(-1.0); keip(-1.0);
            struve(0.5, -1.0); modstruve(0.5, -1.0);
        );
    }
}

#[test]
fn plain_real_domain_errors_f64() {
    real_domain_cases!(f64);
}

#[test]
fn plain_real_domain_errors_f32() {
    real_domain_cases!(f32);
}

#[test]
fn plain_integer_errors_keep_typed_kinds() {
    macro_rules! failure {
        ($kind:ident; $($function:ident($($args:expr),*);)*) => {
            $(let _ = assert_failure($function::<f64>($($args),*), concat!("scioxide_special::", stringify!($function)), SciOxideErrorKind::$kind);)*
        }
    }
    failure!(InvalidInput;
        binomial(1, 2); ln_binomial(1, 2); multifactorial(3, 0); ln_multifactorial(3, 0);
    );
    assert_overflow(bernoulli::<f64>(100_000), "scioxide_special::bernoulli");
    assert_overflow(euler::<f64>(100_000), "scioxide_special::euler");
    failure!(ConvergenceFailure;
        legendre_p(100_000, 0.5); legendre_q(100_000, 0.5);
         legendre_p_assoc(100_000, 0, 0.5); stirling_first_kind(100_000, 2); stirling_second_kind(100_000, 2);
    );
    for (result, name) in [
        (
            checked_bernoulli(100).map(|_| ()),
            "scioxide_special::checked_bernoulli",
        ),
        (
            checked_euler(100).map(|_| ()),
            "scioxide_special::checked_euler",
        ),
    ] {
        let _ = assert_failure(result, name, SciOxideErrorKind::SizeOverflow);
    }
    assert_overflow(factorial::<f64>(172), "scioxide_special::factorial");
    assert_overflow(
        double_factorial::<f64>(400),
        "scioxide_special::double_factorial",
    );
    // Logarithmic forms remain representable after the scalar product overflows.
    assert!(ln_factorial::<f64>(172).unwrap().is_finite());
    assert!(ln_double_factorial::<f64>(400).unwrap().is_finite());
}

#[test]
fn plain_overflow_keeps_typed_provenance_and_tails_round_to_zero() {
    macro_rules! overflow {
        ($scalar:ty; $($function:ident($($args:expr),*);)*) => {
            $(assert_overflow($function::<$scalar>($($args),*), concat!("scioxide_special::", stringify!($function)));)*
        }
    }
    overflow!(f64;
        gamma(172.0); airy_bi(1000.0); bessel_i(0.0, 1000.0); bessel_k(200.0, 0.001);
        bessel_y(190.06189565406507, 0.021919563227472242); gamma_lower(172.0, 1000.0);
        gamma_upper(172.0, 1.0); erfi(30.0); shi(1000.0); chi(1000.0);
    );
    overflow!(f32;
        gamma(40.0); airy_bi(100.0); bessel_i(0.0, 100.0); bessel_k(40.0, 0.001);
        bessel_y(40.0, 0.001); factorial(35); double_factorial(80);
    );
    // B(1000,1000) ≈ 1.2e-603 is below the range: a rounded zero, not an error.
    assert_eq!(beta(1000.0_f64, 1000.0).unwrap().to_bits(), 0);
}

#[test]
fn bessel_y_overflow_sweep_f64() {
    for step in 0..32 {
        let order = 180.0 + f64::from(step) * 0.625;
        for index in 0..13 {
            let x = 0.001 * 10.0_f64.powf(f64::from(index) / 6.0);
            assert_overflow(bessel_y(order, x), "scioxide_special::bessel_y");
            assert_overflow(
                bessel_y_with_options(order, x, BesselOptions::default()),
                "scioxide_special::bessel_y_with_options",
            );
        }
    }
}

#[test]
fn bessel_y_overflow_sweep_f32() {
    for step in 0..32 {
        let order = 40.0 + step as f32 * 0.625;
        for index in 0..13 {
            let x = 0.001 * 10.0_f32.powf(index as f32 / 6.0);
            assert_overflow(bessel_y(order, x), "scioxide_special::bessel_y");
            assert_overflow(
                bessel_y_with_options(order, x, BesselOptions::default()),
                "scioxide_special::bessel_y_with_options",
            );
        }
    }
}

#[test]
fn range_boundaries_preserve_finite_values_and_documented_limits() {
    // The requested Y_1 is finite although Y_2 is outside the f64 range.
    let x = 1.0e-308;
    let y = bessel_y(1.0_f64, x).unwrap();
    assert!((y * x + 2.0 / core::f64::consts::PI).abs() < 8.0 * f64::EPSILON);
    assert_overflow(bessel_y(2.0_f64, x), "scioxide_special::bessel_y");
    assert!(bessel_y(0.0_f64, 1.0).unwrap().is_finite());
    assert!(bessel_y(0.0_f32, 1.0).unwrap().is_finite());
    assert_eq!(erfinv(1.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(erfcinv(0.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(elliptic_k(1.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(bessel_k(0.0_f64, 0.0).unwrap(), f64::INFINITY);
    assert_eq!(airy_ai(1000.0_f64).unwrap(), 0.0);
    assert_eq!(bessel_k(0.0_f64, 1000.0).unwrap(), 0.0);
    assert!(gamma_p(0.5_f64, f64::from_bits(1)).unwrap() > 0.0);
}

#[test]
fn angular_preparation_and_kernel_errors_keep_typed_kinds() {
    assert_eq!(
        AngularMomentum::new(-1).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        AngularState::new(0, 2).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let large = [2_000_000_000, 0, 0];
    let _ = assert_failure(
        wigner_3j::<f64>(large, [0; 3]),
        "scioxide_special::wigner_3j",
        SciOxideErrorKind::SizeOverflow,
    );
    let _ = assert_failure(
        clebsch_gordan::<f64>(large, [0; 3]),
        "scioxide_special::clebsch_gordan",
        SciOxideErrorKind::SizeOverflow,
    );
    let _ = assert_failure(
        gaunt::<f64>([1_000_000_000, 0, 0], [0; 3]),
        "scioxide_special::gaunt",
        SciOxideErrorKind::SizeOverflow,
    );
    let _ = assert_failure(
        gaunt::<f64>([1; 3], [2, 0, 0]),
        "scioxide_special::gaunt",
        SciOxideErrorKind::InvalidInput,
    );
    let _ = assert_failure(
        wigner_6j::<f64>([2_000_000_000; 6]),
        "scioxide_special::wigner_6j",
        SciOxideErrorKind::SizeOverflow,
    );
    let _ = assert_failure(
        wigner_9j::<f64>([2_000_000_000; 9]),
        "scioxide_special::wigner_9j",
        SciOxideErrorKind::SizeOverflow,
    );
    let _ = assert_failure(
        racah_w::<f64>([2_000_000_000; 6]),
        "scioxide_special::racah_w",
        SciOxideErrorKind::SizeOverflow,
    );
}

/// Returns the argument named by an `InvalidInput` or `DomainError`.
fn named_argument(error: SciOxideError) -> scioxide_core::Argument {
    match error.view() {
        SciOxideErrorView::InvalidInput { argument, .. }
        | SciOxideErrorView::DomainError { argument, .. } => argument,
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn coupling_plan_errors_name_the_quantum_numbers_of_the_call() {
    use scioxide_core::{ExecutionContext, argument};
    let plan =
        AngularPlan::new(AngularMomentum::new(2).unwrap(), AngularOptions::default()).unwrap();
    let mut scratch = plan.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    let (small, above) = (
        AngularState::new(2, 0).unwrap(),
        AngularState::new(4, 0).unwrap(),
    );
    let (j, high) = (AngularMomentum::new(2).unwrap(), above.momentum());
    let half = AngularState::new(1, 1).unwrap();
    for (result, expected) in [
        (
            plan.wigner_3j_scoped::<f64>([small, above, small], &mut scratch, scope()),
            argument!("q"),
        ),
        (plan.wigner_3j::<f64>([small, small, above]), argument!("q")),
        (
            plan.wigner_6j_scoped::<f64>([j, j, j, j, j, high], &mut scratch, scope()),
            argument!("j"),
        ),
        (plan.wigner_9j::<f64>([high; 9]), argument!("j")),
        (plan.racah_w::<f64>([j, high, j, j, j, j]), argument!("j")),
        (
            plan.clebsch_gordan_scoped::<f64>(small, above, small, &mut scratch, scope()),
            argument!("b"),
        ),
        (
            plan.clebsch_gordan::<f64>(above, small, above),
            argument!("a"),
        ),
        (
            plan.clebsch_gordan::<f64>(small, small, above),
            argument!("c"),
        ),
        (
            plan.gaunt_scoped::<f64>([half; 3], &mut scratch, scope()),
            argument!("q"),
        ),
        (plan.gaunt::<f64>([small, small, above]), argument!("q")),
    ] {
        assert_eq!(named_argument(result.unwrap_err()), expected);
    }
}

#[test]
fn coulomb_sequence_errors_name_the_length_argument_of_the_call() {
    use scioxide_core::{ExecutionContext, argument};
    // Consecutive orders above 2⁵³ round to the same `f64`.
    let owned = [
        coulomb_wave_sequence(1e17, 0.0, 1.0_f64, 2).map(drop),
        coulomb_sequence(CoulombComponent::F, 1e17, 0.0, 1.0_f64, 2).map(drop),
        coulomb_normalization_sequence(1e17, 0.0_f64, 2).map(drop),
    ];
    for result in owned {
        let error = result.unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DomainError {
                reason: DomainReason::OutsideDomain,
                ..
            }
        ));
        assert_eq!(named_argument(error), argument!("count"));
    }
    let options = CoulombOptions::default();
    let scope = ExecutionContext::sequential;
    let mut waves = [CoulombWaveResult::<f64>::zero(); 4];
    let (output, staging) = waves.split_at_mut(2);
    let mut values = [SpecialFunctionResult::<f64>::zero(); 4];
    let into = [
        coulomb_wave_sequence_into(1e17, 0.0, 1.0_f64, options, output, staging, scope()),
        {
            let (output, staging) = values.split_at_mut(2);
            coulomb_sequence_into(
                CoulombComponent::F,
                1e17,
                0.0,
                1.0_f64,
                options,
                output,
                staging,
                scope(),
            )
        },
        {
            let (output, staging) = values.split_at_mut(2);
            coulomb_normalization_sequence_into(1e17, 0.0_f64, options, output, staging, scope())
        },
    ];
    for result in into {
        assert_eq!(named_argument(result.unwrap_err()), argument!("output"));
    }
}

macro_rules! complex_nonfinite_cases {
    ($scalar:ty) => {{
        let c = |re: $scalar| Complex::new(re, 0.0);
        let one = c(1.0);
        inferred_nonfinite_cases!($scalar, bad;
            gamma(c(bad));
            ln_gamma(c(bad));
            digamma(c(bad));
            polygamma(1, c(bad));
            gamma_lower(one, c(bad));
            gamma_upper(one, c(bad));
            gamma_p(one, c(bad));
            gamma_q(one, c(bad));
            beta(one, c(bad));
            ln_beta(one, c(bad));
            beta_inc(one, one, c(bad));
            riemann_zeta(c(bad));
            dirichlet_eta(c(bad));
            hurwitz_zeta(c(bad), one);
            fresnel_c(c(bad));
            fresnel_s(c(bad));
            elliptic_k(c(bad));
            elliptic_e(c(bad));
            elliptic_f(c(bad), one);
            elliptic_e_inc(c(bad), one);
            elliptic_pi(c(bad), one);
            erf(c(bad));
            erfc(c(bad));
            erfcx(c(bad));
            erfi(c(bad));
            dawson(c(bad));
            log_erfc(c(bad));
            faddeeva(c(bad));
            airy_ai(c(bad));
            airy_bi(c(bad));
            bessel_j(one, c(bad));
            hankel_1(one, c(bad));
            spherical_jn(1, c(bad));
            dilog(c(bad));
            spence(c(bad));
            wright_omega(c(bad));
            lambert_w_complex(c(bad), 0);
            polylog(one, c(bad));
            lerch_phi(c(bad), one, one);
            mathieu_ce(2, 1.0, c(bad));
            mathieu_se(2, 1.0, c(bad));
            mathieu_mc(2, 1.0, c(bad), RadialKind::First);
            mathieu_ms(2, 1.0, c(bad), RadialKind::First);
            spheroidal_angular(SpheroidalKind::Prolate, 0, 2, 1.0, c(bad));
            spheroidal_radial(SpheroidalKind::Prolate, 0, 2, 1.0, c(bad), RadialKind::First);
        );
    }};
}

#[test]
fn plain_complex_errors_report_non_finite_inputs() {
    complex_nonfinite_cases!(f64);
    complex_nonfinite_cases!(f32);
}

#[test]
fn subnormal_bessel_arguments_preserve_finite_values_and_typed_overflow() {
    for x in [f64::from_bits(1), f64::from_bits(2), f64::MIN_POSITIVE] {
        let zero = bessel_y(0.0_f64, x).unwrap();
        let leading =
            core::f64::consts::FRAC_2_PI * (x.ln() - core::f64::consts::LN_2 + 0.5772156649015329);
        assert!((zero - leading).abs() <= 16.0 * f64::EPSILON * leading.abs());
        for order in [1.0, 2.0, 200.0] {
            let result = bessel_y(order, x);
            if let Err(error) = result {
                assert!(
                    matches!(
                        error.view(),
                        SciOxideErrorView::NumericalInstability {
                            issue: NumericalIssue::Overflow,
                            ..
                        }
                    ),
                    "{order}, {x}: {error}"
                );
            } else {
                assert_eq!(order, 1.0);
                assert_eq!(x, f64::MIN_POSITIVE);
            }
        }
    }
}

#[test]
fn publication_domain_classifications_match_across_spellings() {
    macro_rules! cases {
        ($s:ty) => {{
            let zero = 0.0 as $s;
            let one = 1.0 as $s;
            for (name, result) in [
                ("hurwitz s", hurwitz_zeta(one, one)),
                ("hurwitz q", hurwitz_zeta(2.0 as $s, zero)),
                ("ker", ker(zero)),
                ("kerp", kerp(zero)),
                ("coulomb G", coulomb_g(one, zero, zero)),
                ("coulomb G prime", coulomb_g_prime(one, zero, zero)),
                (
                    "coulomb zero-order derivative",
                    coulomb_g_prime(zero, one, zero),
                ),
            ] {
                assert_domain(result, name, DomainReason::Pole);
            }
            for result in [
                gamma_p(zero, zero),
                gamma_q(zero, zero),
                gamma_log_p(zero, zero),
                gamma_log_q(zero, zero),
                beta_inc(zero, one, zero),
                beta_inc_q(one, zero, one),
                beta_inc_log(zero, zero, 0.5 as $s),
                beta_inc_log_q(zero, zero, 0.5 as $s),
            ] {
                assert_domain(result, "ambiguous corner", DomainReason::Undefined);
            }
            for (result, expected) in [
                (gamma_p(zero, one), one),
                (gamma_q(zero, one), zero),
                (gamma_log_p(zero, one), zero),
                (gamma_log_q(zero, one), <$s>::NEG_INFINITY),
                (beta_inc(zero, one, 0.5 as $s), one),
                (beta_inc_q(zero, one, 0.5 as $s), zero),
                (beta_inc_log(zero, one, 0.5 as $s), zero),
                (beta_inc_log_q(zero, one, 0.5 as $s), <$s>::NEG_INFINITY),
                (beta_inc(one, zero, 0.5 as $s), zero),
                (beta_inc_q(one, zero, 0.5 as $s), one),
            ] {
                let value = result.unwrap();
                assert!(value == expected || (value - expected).abs() <= 8.0 as $s * <$s>::EPSILON);
            }
            for x in [0.01 as $s, one, 10.0 as $s] {
                let expected = expint_e1(x).unwrap();
                assert!(
                    (gamma_upper(zero, x).unwrap() - expected).abs()
                        <= 32.0 as $s * <$s>::EPSILON * expected.abs()
                );
                let c = Complex::new;
                for (real, complex) in [
                    (gamma_p(zero, x), gamma_p(c(zero, zero), c(x, zero))),
                    (gamma_q(zero, x), gamma_q(c(zero, zero), c(x, zero))),
                    (gamma_upper(zero, x), gamma_upper(c(zero, zero), c(x, zero))),
                ] {
                    let expected = real.unwrap();
                    let actual = complex.unwrap();
                    assert!(
                        (actual.re - expected).abs()
                            <= 256.0 as $s * <$s>::EPSILON * expected.abs().max(one)
                    );
                    assert!(actual.im.abs() <= 8.0 as $s * <$s>::EPSILON);
                }
            }
            for x in [-2.0 as $s, -0.25 as $s, 0.25 as $s, 2.0 as $s] {
                assert_eq!(
                    expint_e1(x).unwrap().to_bits(),
                    expint_en(1, x).unwrap().to_bits()
                );
                assert_eq!(
                    expint_e1_scaled(x).unwrap().to_bits(),
                    expint_en_scaled(1, x).unwrap().to_bits()
                );
            }
            for eta in [-1.0 as $s, zero, one] {
                let expected = one / coulomb_normalization(zero, eta).unwrap();
                assert!(
                    (coulomb_g(zero, eta, zero).unwrap() - expected).abs()
                        <= 128.0 as $s * <$s>::EPSILON * expected.abs()
                );
                let result = coulomb_g_with_options(zero, eta, zero, Default::default()).unwrap();
                assert_eq!(result.derivative().is_none(), eta != zero);
            }
            assert!(coulomb_g_prime(zero, zero, zero).unwrap().abs() <= <$s>::EPSILON);
            let c = Complex::new;
            for result in [
                hurwitz_zeta(c(one, zero), c(one, zero)),
                bessel_y(c(one, 0.5 as $s), c(zero, zero)),
                bessel_k(c(one, 0.5 as $s), c(zero, zero)),
            ] {
                assert_domain(result, "complex pole", DomainReason::Pole);
            }
            assert!(
                (beta_inc(c(zero, zero), c(one, zero), c(0.5 as $s, zero))
                    .unwrap()
                    .re
                    - one)
                    .abs()
                    <= <$s>::EPSILON
            );
        }};
    }
    cases!(f32);
    cases!(f64);
}

#[test]
fn discrete_parameter_violations_use_out_of_range() {
    let failures = [
        legendre_p_assoc(1, 2, 0.5_f64).map(|_| ()),
        spherical_harmonic(1, 2, 0.5_f64, 0.0).map(|_| ()),
        AngularState::new(2, 4).map(|_| ()),
        MathieuPlan::new(
            0,
            MathieuParity::Odd,
            1.0_f64,
            SpectralNormalization::Standard,
            Default::default(),
        )
        .map(|_| ()),
        SpheroidalPlan::new(
            SpheroidalKind::Prolate,
            2,
            1,
            1.0_f64,
            SpectralNormalization::Standard,
            Default::default(),
        )
        .map(|_| ()),
        EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 1, 0, Default::default()).map(|_| ()),
    ];
    for result in failures {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { reason: scioxide_core::InvalidInputReason::OutOfRange, detail, .. } if !detail.is_empty())
        );
    }
}

#[test]
fn spectral_truncation_bounds_name_the_parameter_the_caller_passed() {
    fn argument<T>(result: Result<T, SciOxideError>) -> &'static str {
        let Some(error) = result.err() else {
            panic!("expected an out-of-range argument");
        };
        match error.view() {
            SciOxideErrorView::InvalidInput {
                argument,
                reason: scioxide_core::InvalidInputReason::OutOfRange,
                ..
            } => argument.as_str(),
            other => panic!("expected an out-of-range argument, got {other:?}"),
        }
    }
    let prolate = SpheroidalKind::Prolate;
    // The default bounds keep a mode index below 248, or below 256 without a parameter.
    assert_eq!(argument(mathieu_a(496, 1.0_f64)), "order");
    assert_eq!(argument(mathieu_b(497, 1.0_f64)), "order");
    assert_eq!(argument(mathieu_ce(496, 1.0, 0.5_f64)), "order");
    assert_eq!(argument(mathieu_se(497, 1.0, 0.5_f64)), "order");
    assert_eq!(
        argument(mathieu_mc(496, 1.0, 0.5_f64, RadialKind::First)),
        "order"
    );
    assert_eq!(
        argument(mathieu_ms(497, 1.0, 0.5_f64, RadialKind::First)),
        "order"
    );
    assert_eq!(argument(mathieu_a(512, 0.0_f64)), "order");
    assert_eq!(argument(mathieu_b(513, 0.0_f64)), "order");
    assert!(mathieu_a(495, 1.0_f64).is_ok());
    assert!(mathieu_b(496, 1.0_f64).is_ok());
    assert!(mathieu_a(511, 0.0_f64).is_ok());
    assert_eq!(
        argument(MathieuPlan::new(
            496,
            MathieuParity::Even,
            1.0_f64,
            SpectralNormalization::Standard,
            Default::default(),
        )),
        "options.terms.maximum"
    );
    assert_eq!(
        argument(spheroidal_characteristic_value(prolate, 0, 496, 1.0_f64)),
        "n"
    );
    assert_eq!(
        argument(spheroidal_angular(prolate, 1, 497, 1.0, 0.5_f64)),
        "n"
    );
    assert_eq!(
        argument(spheroidal_radial(
            prolate,
            0,
            496,
            1.0,
            1.5_f64,
            RadialKind::First
        )),
        "n"
    );
    assert_eq!(
        argument(spheroidal_characteristic_value(prolate, 0, 512, 0.0_f64)),
        "n"
    );
    assert!(spheroidal_characteristic_value(prolate, 0, 511, 0.0_f64).is_ok());
    assert_eq!(
        argument(SpheroidalPlan::new(
            prolate,
            0,
            496,
            1.0_f64,
            SpectralNormalization::Standard,
            Default::default(),
        )),
        "options.terms.maximum"
    );
}

#[test]
fn details_describe_the_rejected_condition_and_scoped_domains_agree() {
    for result in [
        bessel_y(0.0_f64, -1.0),
        bessel_k(0.0, -1.0),
        bessel_k_scaled(0.0, -1.0),
    ] {
        match result.unwrap_err().view() {
            SciOxideErrorView::DomainError {
                argument,
                reason,
                detail,
            } => {
                assert_eq!(argument, scioxide_core::argument!("x"));
                assert_eq!(reason, DomainReason::OutsideDomain);
                assert!(
                    detail.contains("positive")
                        || detail.contains("non-negative")
                        || detail.contains(">0")
                        || detail.contains("> 0"),
                    "{detail}"
                );
                assert!(!detail.contains("must be finite"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
    for x in [-1.0_f64, 1.0] {
        let result = expint_en_scoped(
            1,
            x,
            IntegralOptions::default(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            result.try_value().unwrap().to_bits(),
            expint_e1(x).unwrap().to_bits()
        );
        let scaled = expint_en_scaled_scoped(
            1,
            x,
            IntegralOptions::default(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            scaled.try_value().unwrap().to_bits(),
            expint_e1_scaled(x).unwrap().to_bits()
        );
    }
    for result in [hurwitz_zeta_log(1.0_f64, 1.0), hurwitz_zeta_log(2.0, 0.0)] {
        assert_domain(result, "logarithmic Hurwitz pole", DomainReason::Pole);
    }
}

/// Asserts that `result` is a domain error at a pole that names `expected`.
fn assert_pole<T: core::fmt::Debug>(
    result: Result<T, SciOxideError>,
    expected: scioxide_core::Argument,
) {
    match result.unwrap_err().view() {
        SciOxideErrorView::DomainError {
            argument, reason, ..
        } => {
            assert_eq!(argument, expected);
            assert_eq!(reason, DomainReason::Pole);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn hypergeometric_poles_name_the_denominator_parameter_of_the_call() {
    let context = scioxide_core::ExecutionContext::sequential();
    let b = scioxide_core::argument!("b");
    assert_pole(hyp_pfq(&[1.0_f64], &[-2.0], 0.5), b);
    assert_pole(
        hyp_pfq_with_options(&[1.0_f64], &[-2.0], 0.5, Default::default()),
        b,
    );
    assert_pole(
        hyp_pfq_scoped(&[1.0_f64], &[-2.0], 0.5, Default::default(), context),
        b,
    );
    assert_pole(hyp0f1(-2.0_f64, 0.5), b);
    assert_pole(hyp1f1(1.0_f64, -2.0, 0.5), b);
    assert_pole(
        hyp1f1_with_options(1.0_f64, -2.0, 0.5, Default::default()),
        b,
    );
    assert_pole(
        hyp1f1(1.0, -2.0, scioxide_core::Complex::new(0.5_f64, 0.25)),
        b,
    );
    let c = scioxide_core::argument!("c");
    assert_pole(hyp2f1(1.0_f64, 1.0, -2.0, 0.5), c);
    assert_pole(
        hyp2f1_with_options(1.0_f64, 1.0, -2.0, 0.5, Default::default()),
        c,
    );
    assert_pole(
        HypergeometricFunction::pfq(&[1.0_f64], &[-2.0], false, Default::default()),
        scioxide_core::argument!("denominator"),
    );
}
