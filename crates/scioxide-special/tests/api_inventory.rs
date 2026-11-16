//! Reviewed root API inventory, compiled as an external consumer.
#![allow(unused_imports)]
use scioxide_special::{
    AiryOptions, AngularMomentum, AngularOptions, AngularPlan, AngularState, AngularWorkspace,
    BesselIntegralOptions, BesselOptions, BetaOptions, ClausenOptions, CombinatorialOptions,
    CoulombComponent, CoulombOptions, CoulombWaveResult, DebyeOptions, EllipsoidalHarmonicPlan,
    EllipsoidalOptions, EllipsoidalSigns, EllipticOptions, ErrorFunctionOptions, FermiDiracOptions,
    GammaOptions, HypergeometricDerivative, HypergeometricFamily, HypergeometricFunction,
    HypergeometricOptions, HypergeometricWorkspace, IntegralOptions, KelvinOptions, LambertOptions,
    LameClass, LegendreOptions, LerchOptions, MathieuParity, MathieuPlan, ParabolicCylinderOptions,
    Parameter, PolylogOptions, ProbabilityOptions, RadialKind, ScalarRepresentation,
    SpecialFunctionResult, SpecialFunctionTermination, SpectralDiagnostics, SpectralNormalization,
    SpectralOptions, SpectralTerms, SphericalBesselOptions, SpheroidalKind, SpheroidalPlan,
    StruveOptions, SynchrotronOptions, TransportOptions, ValueWithError, ZetaOptions, airy_ai,
    airy_ai_scaled, airy_ai_scaled_scoped, airy_ai_scaled_with_options, airy_ai_scoped,
    airy_ai_with_options, airy_bi, airy_bi_scaled, airy_bi_scaled_scoped,
    airy_bi_scaled_with_options, airy_bi_scoped, airy_bi_with_options, atanint,
    atanint_with_options, bei, bei_scoped, bei_with_options, beip, beip_scoped, beip_with_options,
    ber, ber_scoped, ber_with_options, bernoulli, bernoulli_with_options, berp, berp_scoped,
    berp_with_options, bessel_i, bessel_i_derivative, bessel_i_derivative_scoped,
    bessel_i_derivative_with_options, bessel_i_scaled, bessel_i_scaled_derivative,
    bessel_i_scaled_derivative_scoped, bessel_i_scaled_derivative_with_options,
    bessel_i_scaled_scoped, bessel_i_scaled_with_options, bessel_i_scoped, bessel_i_with_options,
    bessel_i0_integral, bessel_i0_integral_scoped, bessel_i0_integral_with_options,
    bessel_i0_remainder_integral, bessel_i0_remainder_integral_scoped,
    bessel_i0_remainder_integral_with_options, bessel_j, bessel_j_derivative,
    bessel_j_derivative_scoped, bessel_j_derivative_with_options, bessel_j_scaled,
    bessel_j_scaled_scoped, bessel_j_scaled_with_options, bessel_j_scoped, bessel_j_with_options,
    bessel_j0_complement_integral, bessel_j0_complement_integral_scoped,
    bessel_j0_complement_integral_with_options, bessel_j0_integral, bessel_j0_integral_scoped,
    bessel_j0_integral_with_options, bessel_k, bessel_k_derivative, bessel_k_derivative_scoped,
    bessel_k_derivative_with_options, bessel_k_log, bessel_k_log_scoped, bessel_k_log_with_options,
    bessel_k_scaled, bessel_k_scaled_derivative, bessel_k_scaled_derivative_scoped,
    bessel_k_scaled_derivative_with_options, bessel_k_scaled_scoped, bessel_k_scaled_with_options,
    bessel_k_scoped, bessel_k_with_options, bessel_k0_integral, bessel_k0_integral_scoped,
    bessel_k0_integral_with_options, bessel_k0_tail_integral, bessel_k0_tail_integral_scoped,
    bessel_k0_tail_integral_with_options, bessel_lambda, bessel_lambda_scoped,
    bessel_lambda_with_options, bessel_weighted_integral, bessel_weighted_integral_scoped,
    bessel_weighted_integral_with_options, bessel_y, bessel_y_derivative,
    bessel_y_derivative_scoped, bessel_y_derivative_with_options, bessel_y_scaled,
    bessel_y_scaled_scoped, bessel_y_scaled_with_options, bessel_y_scoped, bessel_y_with_options,
    bessel_y0_integral, bessel_y0_integral_scoped, bessel_y0_integral_with_options,
    bessel_y0_tail_integral, bessel_y0_tail_integral_scoped, bessel_y0_tail_integral_with_options,
    beta, beta_inc, beta_inc_inv, beta_inc_inv_log, beta_inc_inv_log_scoped,
    beta_inc_inv_log_with_options, beta_inc_inv_scoped, beta_inc_inv_with_options, beta_inc_log,
    beta_inc_log_q, beta_inc_log_q_scoped, beta_inc_log_q_with_options, beta_inc_log_scoped,
    beta_inc_log_with_options, beta_inc_q, beta_inc_q_inv, beta_inc_q_inv_log,
    beta_inc_q_inv_log_scoped, beta_inc_q_inv_log_with_options, beta_inc_q_inv_scoped,
    beta_inc_q_inv_with_options, beta_inc_q_scoped, beta_inc_q_with_options, beta_inc_scoped,
    beta_inc_with_options, beta_with_options, binomial, binomial_with_options, carlson_rc,
    carlson_rc_scoped, carlson_rc_with_options, carlson_rd, carlson_rd_scoped,
    carlson_rd_with_options, carlson_rf, carlson_rf_scoped, carlson_rf_with_options, carlson_rg,
    carlson_rg_scoped, carlson_rg_with_options, carlson_rj, carlson_rj_scoped,
    carlson_rj_with_options, checked_bernoulli, checked_euler, chi, chi_scaled, chi_scaled_scoped,
    chi_scaled_with_options, chi_scoped, chi_with_options, ci, ci_scoped, ci_with_options, clausen,
    clausen_with_options, clebsch_gordan, coulomb_f, coulomb_f_prime, coulomb_f_prime_scoped,
    coulomb_f_prime_with_options, coulomb_f_scoped, coulomb_f_with_options, coulomb_g,
    coulomb_g_prime, coulomb_g_prime_scoped, coulomb_g_prime_with_options, coulomb_g_scoped,
    coulomb_g_with_options, coulomb_normalization, coulomb_normalization_sequence,
    coulomb_normalization_sequence_into, coulomb_normalization_with_options, coulomb_sequence,
    coulomb_sequence_into, coulomb_spherical_f, coulomb_spherical_f_scoped,
    coulomb_spherical_f_with_options, coulomb_wave, coulomb_wave_scoped, coulomb_wave_sequence,
    coulomb_wave_sequence_into, coulomb_wave_with_options, dawson, dawson_scoped,
    dawson_with_options, debye_1, debye_1_scoped, debye_1_with_options, debye_2, debye_2_scoped,
    debye_2_with_options, debye_3, debye_3_scoped, debye_3_with_options, debye_4, debye_4_scoped,
    debye_4_with_options, debye_5, debye_5_scoped, debye_5_with_options, debye_6, debye_6_scoped,
    debye_6_with_options, digamma, digamma_with_options, dilog, dilog_with_options, dirichlet_eta,
    dirichlet_eta_scoped, dirichlet_eta_with_options, double_factorial,
    double_factorial_with_options, ellipsoidal_harmonic, ellipsoidal_harmonic_second_kind,
    ellipsoidal_normalization, elliptic_e, elliptic_e_inc, elliptic_e_inc_scoped,
    elliptic_e_inc_with_options, elliptic_e_scoped, elliptic_e_with_options, elliptic_f,
    elliptic_f_scoped, elliptic_f_with_options, elliptic_k, elliptic_k_scoped,
    elliptic_k_with_options, elliptic_pi, elliptic_pi_scoped, elliptic_pi_with_options, erf,
    erf_scoped, erf_with_options, erfc, erfc_scoped, erfc_with_options, erfcinv,
    erfcinv_with_options, erfcx, erfcx_scoped, erfcx_with_options, erfi, erfi_scoped,
    erfi_with_options, erfinv, erfinv_with_options, euler, euler_with_options, expint_3,
    expint_3_with_options, expint_e1, expint_e1_scaled, expint_e1_scaled_with_options,
    expint_e1_with_options, expint_e2, expint_e2_scaled, expint_e2_scaled_scoped,
    expint_e2_scaled_with_options, expint_e2_scoped, expint_e2_with_options, expint_ei,
    expint_ei_scaled, expint_ei_scaled_with_options, expint_ei_with_options, expint_en,
    expint_en_scaled, expint_en_scaled_scoped, expint_en_scaled_with_options, expint_en_scoped,
    expint_en_with_options, factorial, factorial_with_options, faddeeva, faddeeva_scoped,
    faddeeva_with_options, fermi_dirac_0, fermi_dirac_0_with_options, fermi_dirac_1,
    fermi_dirac_1_scoped, fermi_dirac_1_with_options, fermi_dirac_2, fermi_dirac_2_scoped,
    fermi_dirac_2_with_options, fermi_dirac_3half, fermi_dirac_3half_scoped,
    fermi_dirac_3half_with_options, fermi_dirac_half, fermi_dirac_half_scoped,
    fermi_dirac_half_with_options, fermi_dirac_inc_0, fermi_dirac_inc_0_with_options,
    fermi_dirac_int, fermi_dirac_int_scoped, fermi_dirac_int_with_options, fermi_dirac_m1,
    fermi_dirac_m1_with_options, fermi_dirac_mhalf, fermi_dirac_mhalf_scoped,
    fermi_dirac_mhalf_with_options, fresnel_c, fresnel_c_scoped, fresnel_c_with_options, fresnel_s,
    fresnel_s_scoped, fresnel_s_with_options, gamma, gamma_log_p, gamma_log_p_scoped,
    gamma_log_p_with_options, gamma_log_q, gamma_log_q_scoped, gamma_log_q_with_options,
    gamma_lower, gamma_lower_scoped, gamma_lower_with_options, gamma_p, gamma_p_inv,
    gamma_p_inv_log, gamma_p_inv_log_scoped, gamma_p_inv_log_with_options, gamma_p_inv_scoped,
    gamma_p_inv_with_options, gamma_p_scoped, gamma_p_with_options, gamma_q, gamma_q_inv,
    gamma_q_inv_log, gamma_q_inv_log_scoped, gamma_q_inv_log_with_options, gamma_q_inv_scoped,
    gamma_q_inv_with_options, gamma_q_scoped, gamma_q_with_options, gamma_upper,
    gamma_upper_scoped, gamma_upper_with_options, gamma_with_options, gaunt, generalized_binomial,
    generalized_binomial_scoped, generalized_binomial_with_options, hankel_1, hankel_1_derivative,
    hankel_1_derivative_scoped, hankel_1_derivative_with_options, hankel_1_scaled,
    hankel_1_scaled_scoped, hankel_1_scaled_with_options, hankel_1_scoped, hankel_1_with_options,
    hankel_2, hankel_2_derivative, hankel_2_derivative_scoped, hankel_2_derivative_with_options,
    hankel_2_scaled, hankel_2_scaled_scoped, hankel_2_scaled_with_options, hankel_2_scoped,
    hankel_2_with_options, hurwitz_zeta, hurwitz_zeta_log, hurwitz_zeta_log_scoped,
    hurwitz_zeta_log_with_options, hurwitz_zeta_scoped, hurwitz_zeta_with_options,
    hydrogenic_radial, hydrogenic_radial_scoped, hydrogenic_radial_with_options, hyp_pfq,
    hyp_pfq_scoped, hyp_pfq_with_options, hyp0f1, hyp0f1_scoped, hyp0f1_with_options, hyp1f1,
    hyp1f1_scoped, hyp1f1_with_options, hyp2f0, hyp2f0_scoped, hyp2f0_with_options, hyp2f1,
    hyp2f1_regularized, hyp2f1_regularized_scoped, hyp2f1_regularized_with_options, hyp2f1_scoped,
    hyp2f1_with_options, hyperu, hyperu_scoped, hyperu_with_options, jacobi_amplitude,
    jacobi_amplitude_scoped, jacobi_amplitude_with_options, jacobi_cn, jacobi_cn_scoped,
    jacobi_cn_with_options, jacobi_dn, jacobi_dn_scoped, jacobi_dn_with_options, jacobi_sn,
    jacobi_sn_scoped, jacobi_sn_with_options, kei, kei_scoped, kei_with_options, keip, keip_scoped,
    keip_with_options, kelvin, kelvin_scoped, kelvin_with_options, ker, ker_scoped,
    ker_with_options, kerp, kerp_scoped, kerp_with_options, lambert_w_complex,
    lambert_w_complex_scoped, lambert_w_complex_with_options, lambert_w0, lambert_w0_scoped,
    lambert_w0_with_options, lambert_wm1, lambert_wm1_scoped, lambert_wm1_with_options,
    legendre_h3d, legendre_h3d_scoped, legendre_h3d_with_options, legendre_p, legendre_p_assoc,
    legendre_p_assoc_scoped, legendre_p_assoc_with_options, legendre_p_scoped,
    legendre_p_with_options, legendre_q, legendre_q_scoped, legendre_q_with_options, lerch_phi,
    lerch_phi_scoped, lerch_phi_with_options, ln_beta, ln_beta_with_options, ln_binomial,
    ln_binomial_with_options, ln_double_factorial, ln_double_factorial_with_options, ln_factorial,
    ln_factorial_with_options, ln_gamma, ln_gamma_with_options, ln_multifactorial,
    ln_multifactorial_with_options, log_erfc, log_erfc_scoped, log_erfc_with_options,
    logarithmic_integral, logarithmic_integral_offset, logarithmic_integral_offset_with_options,
    logarithmic_integral_with_options, modstruve, modstruve_scoped, modstruve_with_options,
    multifactorial, multifactorial_with_options, normal_cdf, normal_cdf_with_options,
    normal_inv_log_cdf, normal_inv_log_cdf_scoped, normal_inv_log_cdf_with_options,
    normal_inv_log_sf, normal_inv_log_sf_scoped, normal_inv_log_sf_with_options, normal_isf,
    normal_isf_with_options, normal_log_cdf, normal_log_cdf_with_options, normal_log_sf,
    normal_log_sf_with_options, normal_quantile, normal_quantile_with_options, normal_sf,
    normal_sf_with_options, owens_t, owens_t_with_options, parabolic_d, parabolic_d_complex,
    parabolic_d_complex_scoped, parabolic_d_complex_with_options, parabolic_d_scoped,
    parabolic_d_with_options, parabolic_v, parabolic_v_scoped, parabolic_v_with_options,
    parabolic_w, parabolic_w_scoped, parabolic_w_with_options, pochhammer, pochhammer_relative,
    pochhammer_relative_with_options, pochhammer_with_options, polygamma, polygamma_scoped,
    polygamma_with_options, polylog, polylog_scoped, polylog_with_options, racah_w, riccati_jn,
    riccati_jn_scoped, riccati_jn_with_options, riccati_yn, riccati_yn_scoped,
    riccati_yn_with_options, riemann_zeta, riemann_zeta_scoped, riemann_zeta_with_options, shi,
    shi_scaled, shi_scaled_scoped, shi_scaled_with_options, shi_scoped, shi_with_options, si,
    si_scoped, si_with_options, spence, spence_with_options, spherical_hankel_1,
    spherical_hankel_1_scaled, spherical_hankel_1_scaled_scoped,
    spherical_hankel_1_scaled_with_options, spherical_hankel_1_scoped,
    spherical_hankel_1_with_options, spherical_hankel_2, spherical_hankel_2_scaled,
    spherical_hankel_2_scaled_scoped, spherical_hankel_2_scaled_with_options,
    spherical_hankel_2_scoped, spherical_hankel_2_with_options, spherical_harmonic,
    spherical_harmonic_real, spherical_harmonic_real_scoped, spherical_harmonic_real_with_options,
    spherical_harmonic_scoped, spherical_harmonic_with_options, spherical_in, spherical_in_scaled,
    spherical_in_scaled_scoped, spherical_in_scaled_with_options, spherical_in_scoped,
    spherical_in_with_options, spherical_jn, spherical_jn_scoped, spherical_jn_with_options,
    spherical_kn, spherical_kn_scaled, spherical_kn_scaled_scoped,
    spherical_kn_scaled_with_options, spherical_kn_scoped, spherical_kn_with_options, spherical_yn,
    spherical_yn_scoped, spherical_yn_with_options, stirling_first_kind,
    stirling_first_kind_scoped, stirling_first_kind_with_options, stirling_second_kind,
    stirling_second_kind_scoped, stirling_second_kind_with_options, struve, struve_h0_integral,
    struve_h0_integral_scoped, struve_h0_integral_with_options, struve_h0_tail_integral,
    struve_h0_tail_integral_scoped, struve_h0_tail_integral_with_options, struve_l0_integral,
    struve_l0_integral_scoped, struve_l0_integral_with_options, struve_scoped, struve_with_options,
    synchrotron_1, synchrotron_1_scaled, synchrotron_1_scaled_with_options,
    synchrotron_1_with_options, synchrotron_2, synchrotron_2_scaled,
    synchrotron_2_scaled_with_options, synchrotron_2_with_options, transport_2, transport_2_scoped,
    transport_2_with_options, transport_3, transport_3_scoped, transport_3_with_options,
    transport_4, transport_4_scoped, transport_4_with_options, transport_5, transport_5_scoped,
    transport_5_with_options, voigt_complex, voigt_complex_scoped, voigt_complex_with_options,
    voigt_profile, voigt_profile_scoped, voigt_profile_with_options, wigner_3j, wigner_6j,
    wigner_9j, wright_omega, wright_omega_scoped, wright_omega_with_options,
};

use scioxide_special::{
    bei_scaled, bei_scaled_scoped, bei_scaled_with_options, beip_scaled, beip_scaled_scoped,
    beip_scaled_with_options, ber_scaled, ber_scaled_scoped, ber_scaled_with_options, berp_scaled,
    berp_scaled_scoped, berp_scaled_with_options, kei_scaled, kei_scaled_scoped,
    kei_scaled_with_options, keip_scaled, keip_scaled_scoped, keip_scaled_with_options, ker_scaled,
    ker_scaled_scoped, ker_scaled_with_options, kerp_scaled, kerp_scaled_scoped,
    kerp_scaled_with_options,
};

use scioxide_special::{
    mathieu_a, mathieu_b, mathieu_ce, mathieu_mc, mathieu_ms, mathieu_se, spheroidal_angular,
    spheroidal_characteristic_value, spheroidal_radial,
};

#[path = "support/domain_inventory.rs"]
#[allow(clippy::unwrap_used, clippy::panic)]
mod documentation_inventory;

#[test]
fn every_public_plain_function_has_a_runnable_example() {
    let documents = documentation_inventory::plain_documentation();
    assert!(!documents.is_empty());
    for complex in [false, true] {
        for name in documentation_inventory::functions(complex) {
            assert!(documents.contains_key(&name));
        }
    }
    let missing: Vec<_> = documents
        .into_iter()
        .filter_map(|(name, documentation)| {
            let example = documentation
                .split_once("# Examples")
                .map(|(_, example)| example);
            let valid = example.is_some_and(|example| {
                example.contains("```")
                    && (example.contains(&format!("{name}("))
                        || example.contains(&format!("{name}::<")))
            });
            (!valid).then_some(name)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "plain functions missing runnable examples: {missing:?}"
    );
}
