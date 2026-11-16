//! Integration tests for cross-module interactions in `scioxide-special`.
//!
//! These tests verify that different special function modules compose
//! correctly and that mathematical identities relating functions from
//! different modules hold to machine precision.

#![allow(clippy::unwrap_used, clippy::suboptimal_flops)]

use scioxide_special::{
    airy_ai, airy_bi, beta, beta_inc, dirichlet_eta, elliptic_e, elliptic_k, erf, erfc, erfinv,
    gamma, gamma_p, gamma_q, hyp2f1, legendre_p, ln_beta, ln_gamma, riemann_zeta,
};

// ---------------------------------------------------------------------------
// erf ↔ gamma_p connection
// ---------------------------------------------------------------------------

#[test]
fn test_erf_via_gamma_p_f64() {
    // erf(x) = sign(x) · P(0.5, x²)
    for &x in &[0.2_f64, 0.5, 1.0, 1.5, 2.0] {
        let from_erf = erf(x).unwrap();
        let from_gamma_p = gamma_p(0.5, x * x).unwrap();
        assert!(
            (from_erf - from_gamma_p).abs() < 1e-14,
            "erf({x}) = {from_erf}, P(0.5,{})={from_gamma_p}",
            x * x
        );
    }
}

#[test]
fn test_erfc_via_gamma_q_f64() {
    // erfc(x) = Q(0.5, x²) for x > 0
    for &x in &[0.5_f64, 1.0, 2.0, 3.0] {
        let from_erfc = erfc(x).unwrap();
        let from_gamma_q = gamma_q(0.5, x * x).unwrap();
        assert!(
            (from_erfc - from_gamma_q).abs() < 1e-14,
            "erfc({x})={from_erfc}, Q(0.5,{})={from_gamma_q}",
            x * x
        );
    }
}

// ---------------------------------------------------------------------------
// beta ↔ gamma connection
// ---------------------------------------------------------------------------

#[test]
fn test_beta_via_gamma_f64() {
    // B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    let a = 2.5_f64;
    let b = 3.5;
    let beta_val = beta(a, b).unwrap();
    let gamma_ratio = gamma(a).unwrap() * gamma(b).unwrap() / gamma(a + b).unwrap();
    assert!(
        (beta_val - gamma_ratio).abs() < 1e-12,
        "B({a},{b})={beta_val}, Γ ratio={gamma_ratio}"
    );
}

#[test]
fn test_ln_beta_via_ln_gamma_f64() {
    let a = 4.0_f64;
    let b = 7.0;
    let lb = ln_beta(a, b).unwrap();
    let lg_sum = ln_gamma(a).unwrap() + ln_gamma(b).unwrap() - ln_gamma(a + b).unwrap();
    assert!(
        (lb - lg_sum).abs() < 1e-12,
        "ln_beta({a},{b})={lb}, sum={lg_sum}"
    );
}

// ---------------------------------------------------------------------------
// Incomplete gamma identities
// ---------------------------------------------------------------------------

#[test]
fn test_gamma_p_plus_q_equals_one_f64() {
    for &(a, x) in &[(1.0_f64, 0.5), (2.5, 1.0), (0.5, 3.0), (5.0, 2.0)] {
        let p = gamma_p(a, x).unwrap();
        let q = gamma_q(a, x).unwrap();
        assert!(
            (p + q - 1.0).abs() < 1e-13,
            "P({a},{x})+Q({a},{x})={}, expected 1",
            p + q
        );
    }
}

// ---------------------------------------------------------------------------
// Incomplete beta symmetry
// ---------------------------------------------------------------------------

#[test]
fn test_beta_inc_symmetry_cross_module_f64() {
    // I_x(a,b) = 1 - I_{1-x}(b,a)
    let a = 3.0_f64;
    let b = 2.0;
    let x = 0.4;
    let lhs = beta_inc(a, b, x).unwrap();
    let rhs = 1.0 - beta_inc(b, a, 1.0 - x).unwrap();
    assert!((lhs - rhs).abs() < 1e-10, "symmetry failed: {lhs} vs {rhs}");
}

// ---------------------------------------------------------------------------
// erf/erfinv roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_erf_erfinv_roundtrip_f64() {
    for &x in &[0.1_f64, 0.3, 0.5, 0.7, 0.9, -0.2, -0.6, -0.95] {
        let t = erfinv(x).unwrap();
        let back = erf(t).unwrap();
        assert!(
            (back - x).abs() < 1e-13,
            "roundtrip failed for x={x}: erf(erfinv({x}))={back}"
        );
    }
}

// ---------------------------------------------------------------------------
// Elliptic integral known values
// ---------------------------------------------------------------------------

#[test]
fn test_elliptic_k_e_at_zero_f64() {
    let pi_half = std::f64::consts::FRAC_PI_2;
    let k0 = elliptic_k(0.0_f64).unwrap();
    let e0 = elliptic_e(0.0_f64).unwrap();
    assert!((k0 - pi_half).abs() < 1e-14, "K(0)={k0}");
    assert!((e0 - pi_half).abs() < 1e-14, "E(0)={e0}");
}

// ---------------------------------------------------------------------------
// Legendre P_l(1) = 1 and orthogonality spot-check
// ---------------------------------------------------------------------------

#[test]
fn test_legendre_p_boundary_f64() {
    for l in 0..=8 {
        let val = legendre_p(l, 1.0_f64).unwrap();
        assert!((val - 1.0).abs() < 1e-13, "P_{l}(1)={val}");
    }
}

// ---------------------------------------------------------------------------
// Zeta known values
// ---------------------------------------------------------------------------

#[test]
fn test_zeta_2_equals_pi_sq_over_6_f64() {
    let z2 = riemann_zeta(2.0_f64).unwrap();
    let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    assert!(
        (z2 - expected).abs() < 1e-10,
        "ζ(2)={z2}, expected {expected}"
    );
}

#[test]
fn test_zeta_4_equals_pi_4_over_90_f64() {
    let z4 = riemann_zeta(4.0_f64).unwrap();
    let pi = std::f64::consts::PI;
    let expected = pi * pi * pi * pi / 90.0;
    assert!(
        (z4 - expected).abs() < 1e-9,
        "ζ(4)={z4}, expected {expected}"
    );
}

#[test]
fn test_zeta_eta_relation_f64() {
    // η(s) = (1 − 2^{1−s}) ζ(s)
    let s = 3.0_f64;
    let zeta = riemann_zeta(s).unwrap();
    let eta = dirichlet_eta(s).unwrap();
    let expected_eta = (1.0 - 2.0_f64.powf(1.0 - s)) * zeta;
    assert!(
        (eta - expected_eta).abs() < 1e-10,
        "η({s})={eta}, expected {expected_eta}"
    );
}

// ---------------------------------------------------------------------------
// Hypergeometric reduction identity
// ---------------------------------------------------------------------------

#[test]
fn test_hyp2f1_reduction_f64() {
    // ₂F₁(a, b; b; x) = (1 − x)^{−a}
    let a = 2.5_f64;
    let b = 3.0;
    let x = 0.3;
    let val = hyp2f1(a, b, b, x).unwrap();
    let expected = (1.0 - x).powf(-a);
    assert!(
        (val - expected).abs() < 1e-10,
        "₂F₁({a},{b};{b};{x})={val}, expected {expected}"
    );
}

// ---------------------------------------------------------------------------
// Airy function identity
// ---------------------------------------------------------------------------

#[test]
fn test_airy_ai_bi_at_zero_f64() {
    // Ai(0) ≈ 1/(3^{2/3} Γ(2/3)), Bi(0) ≈ 1/(3^{1/6} Γ(2/3))
    let ai0 = airy_ai(0.0_f64).unwrap();
    let bi0 = airy_bi(0.0_f64).unwrap();
    assert!((ai0 - 0.355_028_053_887_817_2).abs() < 1e-10, "Ai(0)={ai0}");
    assert!((bi0 - 0.614_926_627_446_000_7).abs() < 1e-10, "Bi(0)={bi0}");
}

// ---------------------------------------------------------------------------
// Cross-precision: f32 smoke tests
// ---------------------------------------------------------------------------

#[test]
fn test_cross_module_f32() {
    // Verify that the main functions work with f32
    let g = gamma(5.0_f32).unwrap();
    assert!((g - 24.0_f32).abs() < 1e-4);

    let e = erf(1.0_f32).unwrap();
    assert!((e - 0.842_700_8_f32).abs() < 1e-5);

    let p2 = legendre_p(2_usize, 0.5_f32).unwrap();
    assert!((p2 - (-0.125_f32)).abs() < 1e-5);

    let z2 = riemann_zeta(2.0_f32).unwrap();
    let pi32 = std::f32::consts::PI;
    assert!((z2 - pi32 * pi32 / 6.0).abs() < 1e-3);
}
