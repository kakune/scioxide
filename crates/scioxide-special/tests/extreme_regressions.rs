//! Allocation regressions for restored extreme-domain paths.

#![allow(clippy::unwrap_used)]

use scioxide_special::{
    AiryOptions, BesselOptions, ErrorFunctionOptions, GammaOptions, airy_ai, airy_ai_with_options,
    bessel_i, bessel_k, beta, beta_inc, binomial, elliptic_pi, erf_with_options, erfcinv,
    gamma_lower, gamma_p, gamma_upper, gamma_with_options, hankel_1_with_options, hyp1f1, hyp2f1,
    legendre_p, legendre_p_assoc, legendre_q,
};

#[test]
fn compensated_legendre_paths_are_allocation_free() {
    for degree in [2, 23, 100] {
        std::hint::black_box(legendre_p(degree, 0.805_f64)).unwrap();
        std::hint::black_box(legendre_q(degree, 0.475_f64)).unwrap();
    }
    std::hint::black_box(legendre_p_assoc(200, 200, 0.999999_f64)).unwrap();
    std::hint::black_box(legendre_p_assoc(100, -20, 0.25_f64)).unwrap();
}

#[test]
fn restored_extreme_success_paths_are_allocation_free() {
    std::hint::black_box(erfcinv(1.0e-20_f64)).unwrap();
    std::hint::black_box(scioxide_special::gamma_p_inv(
        0.0014109960028295145_f64,
        0.8670461346108965,
    ))
    .unwrap();
    std::hint::black_box(scioxide_special::beta_inc_inv(
        0.021813515576359375_f64,
        0.4295632308260261,
        0.4212025283717187,
    ))
    .unwrap();
    std::hint::black_box(scioxide_special::normal_sf(38_f64)).unwrap();
    std::hint::black_box(erfcinv(f64::from_bits(1))).unwrap();
    std::hint::black_box(gamma_p(0.5_f64, f64::from_bits(1))).unwrap();
    std::hint::black_box(gamma_lower(200.0_f64, 1.0)).unwrap();
    std::hint::black_box(gamma_upper(200.0_f64, 1_000.0)).unwrap();
    std::hint::black_box(beta(1.0e16_f64, 1.0)).unwrap();
    std::hint::black_box(beta_inc(1.0e300_f64, 1.0e300, 0.5)).unwrap();
    std::hint::black_box(binomial::<f64>(1_024, 512)).unwrap();
    std::hint::black_box(airy_ai(1_000.0_f64)).unwrap();
    std::hint::black_box(airy_ai(-1.0e300_f64)).unwrap();
    std::hint::black_box(bessel_i(100.0_f64, 10.0)).unwrap();
    std::hint::black_box(bessel_k(100.0_f64, 10.0)).unwrap();
    std::hint::black_box(bessel_k(0.0_f64, 1_000.0)).unwrap();
    std::hint::black_box(elliptic_pi(-1.0e300_f64, 0.5)).unwrap();
    std::hint::black_box(hyp1f1(-1.0_f64, 2.0, 2.0)).unwrap();
    std::hint::black_box(hyp2f1(0.25_f64, 1.75, 2.5, -1.0e300)).unwrap();
    std::hint::black_box(legendre_p_assoc(200, 200, 1.0_f64)).unwrap();
}

#[test]
fn restored_f32_success_paths_are_allocation_free() {
    std::hint::black_box(gamma_p(0.5_f32, f32::from_bits(1))).unwrap();
    std::hint::black_box(binomial::<f32>(128, 64)).unwrap();
    std::hint::black_box(hyp1f1(-1.0_f32, 2.0, 2.0)).unwrap();
    std::hint::black_box(hyp2f1(
        f32::from_bits(0x3f98_4b50),
        f32::from_bits(0x3f98_8951),
        f32::from_bits(0x4048_f231),
        f32::from_bits(0xc051_24b9),
    ))
    .unwrap();
}

#[test]
fn detailed_representations_are_allocation_free() {
    let _ = std::hint::black_box(gamma_with_options(172.0_f64, GammaOptions::default())).unwrap();
    let _ = std::hint::black_box(airy_ai_with_options(-f64::MAX, AiryOptions::default())).unwrap();
    let _ =
        std::hint::black_box(erf_with_options(0.75_f64, ErrorFunctionOptions::default())).unwrap();
    let _ = std::hint::black_box(hankel_1_with_options(
        2.5_f64,
        10.0,
        BesselOptions::default(),
    ))
    .unwrap();
}

#[test]
fn legendre_rounds_to_finite_maximum_before_reporting_overflow() {
    let x = f32::from_bits(0x4634_0e69);
    let wide = f64::from(x);
    // P9(x) = (12155 x^9 - 25740 x^7 + 18018 x^5 - 4620 x^3 + 315 x)/128.
    let square = wide * wide;
    let reference =
        ((((12155.0 * square - 25740.0) * square + 18018.0) * square - 4620.0) * square + 315.0)
            * wide
            / 128.0;
    let maximum = f64::from(f32::MAX);
    let spacing = maximum - f64::from(f32::from_bits(f32::MAX.to_bits() - 1));
    assert!(reference > maximum && reference < maximum + spacing * 0.5);
    for sign in [-1.0_f32, 1.0] {
        let value = scioxide_special::legendre_p(9, sign * x).unwrap();
        let report =
            scioxide_special::legendre_p_with_options(9, sign * x, Default::default()).unwrap();
        assert_eq!(value, sign * f32::MAX);
        assert_eq!(value, report.try_value().unwrap());
        let error =
            scioxide_special::legendre_p(9, sign * f32::from_bits(x.to_bits() + 1)).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}
