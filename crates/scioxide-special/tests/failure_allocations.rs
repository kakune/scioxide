//! Failed scalar evaluations retain typed, static diagnostics without allocating.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, Complex, ConvergenceReason, DomainReason, ExecutionContext, SciOxideError,
    SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_special::*;

fn detail(error: &SciOxideError) -> &str {
    match error.view() {
        SciOxideErrorView::DomainError { detail, .. }
        | SciOxideErrorView::NonFinite { detail, .. }
        | SciOxideErrorView::ConvergenceFailure { detail, .. }
        | SciOxideErrorView::InvalidInput { detail, .. }
        | SciOxideErrorView::NumericalInstability { detail, .. } => detail,
        _ => unreachable!(),
    }
}
macro_rules! failures {
    ($kind:ident; $($value:expr);+ $(;)?) => {$({
        let error = ($value).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::$kind, "{}: {error}", stringify!($value));
    })+};
}
macro_rules! equal_errors {
    ($first:expr; $($other:expr);+ $(;)?) => {{
        let expected = ($first).unwrap_err();
        $(let actual = ($other).unwrap_err();
        assert_eq!(actual.kind(), expected.kind());
        assert_eq!(detail(&actual), detail(&expected), "{}", stringify!($other));)+
    }};
}
#[test]
fn beta_shapes_are_validated_before_the_smallest_work_budget() {
    macro_rules! rejected {
        ($kind:expr; $($call:expr);+ $(;)?) => {$(
            let error = ($call).unwrap_err();
            assert_eq!(error.kind(), $kind, "{}", stringify!($call));
        )+};
    }
    macro_rules! width {
        ($s:ty) => {{
            let options = BetaOptions::default().with_iteration_budget(Budget::ONE);
            for (invalid, kind) in [
                (<$s>::NAN, SciOxideErrorKind::NonFinite),
                (<$s>::INFINITY, SciOxideErrorKind::NonFinite),
                (-1.0 as $s, SciOxideErrorKind::DomainError),
            ] {
                for (a, b) in [(invalid, 2.0 as $s), (2.0 as $s, invalid)] {
                    rejected!(kind;
                        beta_with_options(a, b, options);
                        ln_beta_with_options(a, b, options);
                        beta_inc_with_options(a, b, 0.25 as $s, options);
                        beta_inc_q_with_options(a, b, 0.25 as $s, options);
                        beta_inc_log_with_options(a, b, 0.25 as $s, options);
                        beta_inc_log_q_with_options(a, b, 0.25 as $s, options);
                        beta_inc_inv_with_options(a, b, 0.25 as $s, options);
                        beta_inc_q_inv_with_options(a, b, 0.25 as $s, options);
                        beta_inc_inv_log_with_options(a, b, -1.0 as $s, options);
                        beta_inc_q_inv_log_with_options(a, b, -1.0 as $s, options)
                    );
                }
            }
        }};
    }
    width!(f32);
    width!(f64);
    let options = BetaOptions::default().with_iteration_budget(Budget::ONE);
    for (invalid, kind) in [
        (Complex::new(f64::NAN, 0.0), SciOxideErrorKind::NonFinite),
        (Complex::new(-1.0, 0.0), SciOxideErrorKind::DomainError),
    ] {
        for (a, b) in [
            (invalid, Complex::new(2.0, 0.25)),
            (Complex::new(2.0, 0.25), invalid),
        ] {
            rejected!(kind;
                beta_with_options(a, b, options);
                ln_beta_with_options(a, b, options)
            );
        }
    }
}

#[test]
fn nan_inputs_across_families_and_precisions_do_not_allocate() {
    macro_rules! width {
        ($s:ty) => {{
            let nan = <$s>::NAN;
            let z = Complex::new(nan, 0.0);
            let one = Complex::new(1.0 as $s, 0.0);
            failures!(NonFinite;
                erf(nan); erfc(nan); erfcx(nan); log_erfc(nan); erfinv(nan); erfcinv(nan);
                normal_cdf(nan); normal_quantile(nan); gamma(nan); ln_gamma(nan); digamma(nan);
                polygamma(2, nan); gamma_p(1.0, nan); gamma_q(1.0, nan);
                gamma_lower(1.0, nan); gamma_upper(1.0, nan); beta(nan, 1.0); ln_beta(nan, 1.0);
                beta_inc(1.0, 1.0, nan); riemann_zeta(nan); dirichlet_eta(nan); hurwitz_zeta(2.0, nan);
                bessel_j(0.5, nan); bessel_i(0.5, nan); bessel_y(0.5, nan); bessel_k(0.5, nan);
                spherical_jn(3, nan); spherical_yn(3, nan); spherical_in(3, nan); spherical_kn(3, nan);
                airy_ai(nan); airy_bi(nan); elliptic_k(nan); elliptic_e(nan); elliptic_f(nan, 0.5);
                elliptic_e_inc(nan, 0.5); elliptic_pi(nan, 0.5); carlson_rf(nan, 1.0, 2.0);
                jacobi_sn(nan, 0.5 as $s); legendre_p(3, nan); legendre_q(3, nan);
                expint_e1(nan); si(nan); fresnel_c(nan); fresnel_s(nan); debye_3(nan);
                fermi_dirac_half(nan); transport_3(nan); synchrotron_1(nan); clausen(nan);
                coulomb_f(0.0, 0.0, nan); hyp1f1(0.5, 1.5, nan); lambert_w0(nan);
                polylog(2.0, nan); pochhammer(nan, 3.0); gamma(nan + 1.0);
                ber(nan); struve(0.5, nan); parabolic_d(0.5, nan);
                gamma(z); ln_gamma(z); digamma(z); beta(one, z); ln_beta(one, z);
                riemann_zeta(z); hurwitz_zeta(one, z); coulomb_f(0.0, 0.0, z);
                bessel_j(one, z); airy_ai(z); fresnel_c(z); elliptic_k(z); erf(z)
            );
        }};
    }
    width!(f32);
    width!(f64);
}
#[test]
fn poles_and_domains_have_static_consistent_details() {
    failures!(DomainError;
        gamma(-2.0_f64); ln_gamma(-2.0_f64); digamma(-2.0_f64); polygamma(2, -2.0_f64);
        gamma_p(1.0_f64, -1.0); beta_inc(1.0_f64, 1.0, 2.0); erfinv(2.0_f64); erfcinv(3.0_f64);
        bessel_j(-0.5_f64, -1.0); bessel_i(-0.5_f64, 0.0); elliptic_k(2.0_f64); legendre_q(1, 2.0_f64);
        expint_e1(0.0_f64); carlson_rf(-1.0_f64, 1.0, 2.0); hyp1f1(1.0_f64, 0.0, 0.5);
        lambert_w0(-1.0_f64); fermi_dirac_inc_0(1.0_f64, -1.0); debye_3(-1.0_f64);
        coulomb_f(-0.5_f64, 0.0, 1.0); riemann_zeta(1.0_f64); gamma(-2.0_f64 + 1.0)
    );
    equal_errors!(gamma(-2.0_f64); ln_gamma(-2.0_f64); digamma(-2.0_f64);
        gamma_with_options(-2.0_f64, GammaOptions::default());
        ln_gamma_with_options(-2.0_f64, GammaOptions::default()));
    equal_errors!(gamma_p(1.0_f64, -1.0);
        gamma_p_with_options(1.0_f64, -1.0, GammaOptions::default());
        gamma_p_scoped(1.0_f64, -1.0, GammaOptions::default(), ExecutionContext::sequential()));
    equal_errors!(elliptic_k(2.0_f64);
        elliptic_k_with_options(2.0_f64, EllipticOptions::default());
        elliptic_k_scoped(2.0_f64, EllipticOptions::default(), ExecutionContext::sequential()));
    for error in [
        gamma(-2.0_f64).unwrap_err(),
        ln_gamma(-2.0_f64).unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DomainError {
                reason: DomainReason::Pole,
                ..
            }
        ));
    }
}
#[test]
fn nonfinite_widening_names_the_correct_family() {
    let nan = Complex::new(f64::NAN, 0.0);
    for (error, family) in [
        (gamma(nan).unwrap_err(), "gamma"),
        (riemann_zeta(nan).unwrap_err(), "zeta"),
        (
            hurwitz_zeta(Complex::new(2.0, 0.0), nan).unwrap_err(),
            "zeta",
        ),
        (ln_beta(Complex::new(1.0, 0.0), nan).unwrap_err(), "beta"),
        (coulomb_f(0.0, 0.0, f64::NAN).unwrap_err(), "Coulomb"),
    ] {
        assert!(detail(&error).contains(family), "{error}");
        assert!(!detail(&error).contains("elliptic"));
    }
    equal_errors!(gamma(nan); gamma_with_options(nan, GammaOptions::default()));
    equal_errors!(riemann_zeta(nan);
        riemann_zeta_with_options(nan, ZetaOptions::default());
        riemann_zeta_scoped(nan, ZetaOptions::default(), ExecutionContext::sequential()));
}
#[test]
fn exhausted_reports_and_errors_do_not_allocate() {
    macro_rules! exhausted {
        ($($detailed:ident($($arg:expr),+; $options:ident));+ $(;)?) => {$({
            let error = $detailed($($arg,)+ $options::default().with_iteration_budget(Budget::ONE)).and_then(|report| report.try_value()).unwrap_err();
            assert!(matches!(error.view(), SciOxideErrorView::ConvergenceFailure { reason: ConvergenceReason::IterationLimit, .. }), "{error}");
            assert_eq!(detail(&error), "the special-function iteration budget was exhausted");
        })+};
    }
    exhausted!(
        gamma_with_options(3.25_f64; GammaOptions);
        beta_with_options(2.25_f64, 3.25; BetaOptions);
        gamma_p_with_options(2.25_f64, 1.25; GammaOptions);
        beta_inc_with_options(2.25_f64, 3.25, 0.3; BetaOptions);
        polygamma_with_options(2, 1.25_f64; GammaOptions);
        erf_with_options(Complex::new(0.25_f64, 0.5); ErrorFunctionOptions);
        gamma_p_inv_with_options(2.25_f64, 0.3; GammaOptions);
        bessel_j_with_options(0.3_f64, 1.25; BesselOptions);
        spherical_jn_with_options(3, 1.25_f64; SphericalBesselOptions);
        airy_ai_with_options(5.0_f64; AiryOptions);
        elliptic_f_with_options(1.25_f64, 0.3; EllipticOptions);
        carlson_rf_with_options(1.0_f64, 2.0, 3.0; EllipticOptions);
        jacobi_sn_with_options(1.25_f64, 0.3; EllipticOptions);
        legendre_p_with_options(30, 0.25_f64; LegendreOptions);
        expint_e1_with_options(2.0_f64; IntegralOptions);
        fresnel_c_with_options(1.25_f64; IntegralOptions);
        debye_3_with_options(1.0_f64; DebyeOptions);
        fermi_dirac_half_with_options(1.0_f64; FermiDiracOptions);
        transport_3_with_options(1.0_f64; TransportOptions);
        synchrotron_1_with_options(1.0_f64; SynchrotronOptions);
        clausen_with_options(1.0_f64; ClausenOptions);
        coulomb_f_with_options(2.0, 3.0, 1.0_f64; CoulombOptions);
        hyp1f1_with_options(0.5_f64, 1.5, 5.0; HypergeometricOptions);
        lambert_w0_with_options(2.0_f64; LambertOptions);
        polylog_with_options(2.25_f64, 0.75; PolylogOptions);
        gamma_with_options(3.25_f64; GammaOptions);
        struve_with_options(0.5_f64, 2.0; StruveOptions);
        parabolic_d_with_options(0.5_f64, 2.0; ParabolicCylinderOptions)
    );
}

#[test]
fn prepared_spectral_and_ellipsoidal_failures_do_not_allocate() {
    let mathieu = MathieuPlan::new(
        2,
        MathieuParity::Even,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let spheroidal = SpheroidalPlan::new(
        SpheroidalKind::Prolate,
        0,
        2,
        1.0_f64,
        scioxide_special::SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let ellipsoid =
        EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 4, 1, EllipsoidalOptions::default()).unwrap();
    failures!(NonFinite; mathieu.angular(f64::NAN); mathieu.radial(f64::NAN, RadialKind::First);
        spheroidal.angular(f64::NAN); ellipsoid.first_kind(f64::NAN, Default::default()).and_then(|report| report.try_value()); ellipsoid.second_kind(f64::NAN).and_then(|report| report.try_value()));
    failures!(DomainError; spheroidal.angular(2.0_f64); ellipsoid.second_kind(0.5_f64).and_then(|report| report.try_value()));
    let ellipsoid = EllipsoidalHarmonicPlan::new(
        0.5_f64,
        1.0,
        0,
        1,
        EllipsoidalOptions::default().with_iteration_budget(Budget::try_from(64).unwrap()),
    )
    .unwrap();
    let error = ellipsoid
        .second_kind_scoped(1.5_f64, ExecutionContext::sequential())
        .and_then(|report| report.try_value())
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
}
