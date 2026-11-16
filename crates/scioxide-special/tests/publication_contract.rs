//! Cross-family publication rules and scalar-range preservation.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{
    Budget, Complex, ConvergenceReason, NumericScalar, RealScalar, SciOxideError, SciOxideErrorView,
};
use scioxide_special::*;

fn exhausted<V: NumericScalar>(
    case: &str,
    result: Result<SpecialFunctionResult<V>, SciOxideError>,
) {
    let error = result.unwrap_err();
    match error.view() {
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => {
            assert_eq!(reason, ConvergenceReason::IterationLimit, "{case}");
            assert_eq!(
                diagnostics.counts().total().unwrap().iterations(),
                1,
                "{case}"
            );
        }
        other => panic!("{case}: {other:?}"),
    }
}

#[test]
fn no_estimate_exhaustion_has_one_error_channel() {
    for (case, result) in [
        (
            "combinatorial",
            double_factorial_with_options::<f64>(
                20,
                CombinatorialOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "hypergeometric",
            hyp1f1_with_options(
                0.5_f64,
                1.5,
                5.0,
                HypergeometricOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "legendre",
            legendre_p_assoc_with_options(
                10,
                2,
                0.5_f64,
                LegendreOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "zeta",
            riemann_zeta_with_options(
                2.0_f64,
                ZetaOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "hurwitz",
            hurwitz_zeta_with_options(
                3.0_f64,
                1.0,
                ZetaOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "hurwitz logarithm",
            hurwitz_zeta_log_with_options(
                3.0_f64,
                1.0,
                ZetaOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "elliptic",
            elliptic_f_with_options(
                0.5_f64,
                0.5,
                EllipticOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "gamma",
            gamma_p_with_options(
                2.25_f64,
                1.25,
                GammaOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "beta",
            beta_inc_with_options(
                2.25_f64,
                3.5,
                0.4,
                BetaOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "bessel",
            bessel_j_with_options(
                2.5_f64,
                2.0,
                BesselOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "airy",
            airy_ai_with_options(
                1.25_f64,
                AiryOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "kelvin",
            ber_with_options(
                2.0_f64,
                KelvinOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "coulomb",
            coulomb_g_with_options(
                0.0_f64,
                1.0,
                2.0,
                CoulombOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "parabolic",
            parabolic_d_with_options(
                0.5_f64,
                2.0,
                ParabolicCylinderOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
    ] {
        exhausted(case, result);
    }
    exhausted(
        "complex polylog",
        polylog_with_options(
            Complex::new(-200.0_f64, 0.0),
            Complex::new(0.5, 0.0),
            PolylogOptions::default().with_iteration_budget(Budget::ONE),
        ),
    );
}

#[test]
fn overflow_keeps_computable_extended_forms() {
    let gamma = gamma_with_options(200.0_f64, GammaOptions::default()).unwrap();
    let polylog = polylog_with_options(
        Complex::new(-200.0_f64, 0.0),
        Complex::new(0.5, 0.0),
        PolylogOptions::default(),
    )
    .unwrap();
    assert_eq!(
        gamma.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert_eq!(
        polylog.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(gamma.value().is_none() && polylog.value().is_none());
    assert!(gamma.scaled().is_some() && polylog.scaled().is_some());
    let logarithm = polylog
        .logarithmic()
        .unwrap()
        .value()
        .log_magnitude()
        .unwrap();
    // For this positive series Li_-n(exp(-mu)) agrees with n!/mu^(n+1)
    // to far below binary64 precision (the other poles are exponentially small).
    let expected = ln_gamma(201.0_f64).unwrap() - 201.0 * (2.0_f64.ln()).ln();
    assert!(
        (logarithm - expected).abs() < 2e-11,
        "{logarithm} {expected}"
    );
    assert!(polylog.try_value().is_err());
    let narrow = polylog_with_options(
        Complex::new(-35.0_f32, 0.0),
        Complex::new(0.5, 0.0),
        PolylogOptions::default(),
    )
    .unwrap();
    assert_eq!(
        narrow.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(narrow.scaled().is_some() && narrow.logarithmic().is_some());
}

#[test]
fn infinite_limits_are_distinct_from_overflow() {
    for result in [
        erfinv_with_options(1.0_f64, ErrorFunctionOptions::default()),
        erfinv_with_options(-1.0_f64, ErrorFunctionOptions::default()),
        gamma_log_q_with_options(0.0_f64, 1.0, GammaOptions::default()),
        beta_inc_log_with_options(1.0_f64, 0.0, 0.5, BetaOptions::default()),
    ] {
        let result = result.unwrap();
        assert_eq!(
            result.scalar_representation(),
            ScalarRepresentation::InfiniteLimit
        );
        assert_eq!(result.termination(), SpecialFunctionTermination::Limit);
        assert!(result.value().unwrap().value().is_infinite());
        assert!(result.try_value().unwrap().is_infinite());
    }
}

#[test]
fn useful_partial_estimates_remain_successful_result_values() {
    for (case, result) in [
        (
            "integral",
            expint_e1_with_options(2.0_f64, IntegralOptions::new(Budget::ONE)),
        ),
        (
            "probability inverse",
            normal_inv_log_cdf_with_options(
                -1000.0_f64,
                ProbabilityOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "Owen",
            owens_t_with_options(
                2.0_f64,
                0.5,
                ProbabilityOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
        (
            "omega",
            wright_omega_with_options(
                3.0_f64,
                LambertOptions::default().with_iteration_budget(Budget::ONE),
            ),
        ),
    ] {
        let result = result.unwrap();
        assert_eq!(
            result.termination(),
            SpecialFunctionTermination::IterationLimit,
            "{case}"
        );
        assert!(result.value().is_some(), "{case}");
        assert_eq!(result.counts().iterations(), 1, "{case}");
        let error = result.try_value().unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    ..
                }
            ),
            "{case}"
        );
    }
}

#[test]
fn rescaled_series_preserves_representable_values_and_plain_projection() {
    for order in [-80.0_f64, -120.0] {
        let report = polylog_with_options(order, 0.5, PolylogOptions::default()).unwrap();
        let value = report.value().unwrap().value();
        let expected_log = ln_gamma(1.0 - order).unwrap() - (1.0 - order) * 2.0_f64.ln().ln();
        assert!((value.ln() - expected_log).abs() < 2e-12);
        assert_eq!(value.to_bits(), polylog(order, 0.5).unwrap().to_bits());
        assert_eq!(report.scalar_representation(), ScalarRepresentation::Finite);
        let scaled = report.scaled().unwrap().value().try_value().unwrap();
        assert!((scaled - value).abs() <= report.value().unwrap().absolute_error());
    }
}

#[test]
fn every_family_publishes_a_finite_completed_estimate() {
    let mathieu = MathieuPlan::new(
        2,
        MathieuParity::Even,
        0.5_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let spheroidal = SpheroidalPlan::new(
        SpheroidalKind::Prolate,
        0,
        2,
        0.5_f64,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    let ellipsoidal =
        EllipsoidalHarmonicPlan::new(0.5_f64, 1.0, 0, 1, EllipsoidalOptions::default()).unwrap();
    for (family, result) in [
        (
            "gamma",
            gamma_with_options(2.5_f64, GammaOptions::default()),
        ),
        (
            "beta",
            beta_inc_with_options(2.0, 3.0, 0.5, BetaOptions::default()),
        ),
        (
            "error",
            erf_with_options(0.5, ErrorFunctionOptions::default()),
        ),
        (
            "probability",
            normal_log_cdf_with_options(-3.0, ProbabilityOptions::default()),
        ),
        (
            "bessel",
            bessel_j_with_options(2.5, 2.0, BesselOptions::default()),
        ),
        ("airy", airy_ai_with_options(1.0, AiryOptions::default())),
        (
            "spherical",
            spherical_jn_with_options(2, 1.0, SphericalBesselOptions::default()),
        ),
        (
            "struve",
            struve_with_options(0.0, 1.0, StruveOptions::default()),
        ),
        ("kelvin", ber_with_options(1.0, KelvinOptions::default())),
        (
            "parabolic",
            parabolic_d_with_options(0.5, 1.0, ParabolicCylinderOptions::default()),
        ),
        (
            "coulomb",
            coulomb_f_with_options(0.0, 0.5, 2.0, CoulombOptions::default()),
        ),
        (
            "elliptic",
            carlson_rf_with_options(1.0, 2.0, 3.0, EllipticOptions::default()),
        ),
        (
            "hypergeometric",
            hyp1f1_with_options(0.5, 1.5, 1.0, HypergeometricOptions::default()),
        ),
        (
            "zeta",
            riemann_zeta_with_options(2.0, ZetaOptions::default()),
        ),
        (
            "polylog",
            polylog_with_options(2.5, 0.5, PolylogOptions::default()),
        ),
        (
            "integrals",
            expint_e1_with_options(1.0, IntegralOptions::default()),
        ),
        ("debye", debye_3_with_options(1.0, DebyeOptions::default())),
        (
            "fermi",
            fermi_dirac_half_with_options(1.0, FermiDiracOptions::default()),
        ),
        (
            "transport",
            transport_2_with_options(1.0, TransportOptions::default()),
        ),
        (
            "synchrotron",
            synchrotron_1_with_options(1.0, SynchrotronOptions::default()),
        ),
        (
            "clausen",
            clausen_with_options(1.0, ClausenOptions::default()),
        ),
        (
            "combinatorial",
            double_factorial_with_options(20, CombinatorialOptions::default()),
        ),
        (
            "legendre",
            legendre_p_with_options(3, 0.5, LegendreOptions::default()),
        ),
        (
            "angular",
            spherical_harmonic_real_with_options(2, 1, 0.5, 0.25, LegendreOptions::default()),
        ),
        ("mathieu", mathieu.angular(0.5)),
        ("spheroidal", spheroidal.angular(0.5)),
        (
            "ellipsoidal",
            ellipsoidal.first_kind(1.5, Default::default()),
        ),
    ] {
        let report = result.unwrap_or_else(|error| panic!("{family}: {error}"));
        assert!(report.termination().is_success(), "{family}");
        assert_eq!(
            report.scalar_representation(),
            ScalarRepresentation::Finite,
            "{family}"
        );
        assert!(report.value().unwrap().value().is_finite(), "{family}");
        assert_eq!(
            report.try_value().unwrap().to_bits(),
            report.value().unwrap().value().to_bits(),
            "{family}"
        );
    }
}

/// A final result whose correctly rounded value is zero is an ordinary signed
/// zero, identical in the plain and detailed forms, with a bound covering the
/// nonzero mathematical magnitude `magnitude` (zero for an exact zero).
fn rounded_zero<V: NumericScalar>(
    case: &str,
    plain: Result<V, SciOxideError>,
    detailed: Result<SpecialFunctionResult<V>, SciOxideError>,
    negative: bool,
    magnitude: f64,
) {
    let plain = plain.unwrap_or_else(|error| panic!("{case}: {error:?}"));
    let detailed = detailed.unwrap_or_else(|error| panic!("{case}: {error:?}"));
    let re = plain.real_component().into_f64();
    assert_eq!(re, 0.0, "{case}");
    assert_eq!(plain.imaginary_component().into_f64(), 0.0, "{case}");
    assert_eq!(re.is_sign_negative(), negative, "{case}");
    assert_eq!(
        detailed.scalar_representation(),
        ScalarRepresentation::Finite,
        "{case}"
    );
    let value = detailed.value().unwrap();
    for (published, expected) in [
        (detailed.try_value().unwrap(), plain),
        (*value.value(), plain),
    ] {
        assert_eq!(
            published.real_component().into_f64().to_bits(),
            expected.real_component().into_f64().to_bits(),
            "{case}"
        );
        assert_eq!(
            published.imaginary_component().into_f64().to_bits(),
            expected.imaginary_component().into_f64().to_bits(),
            "{case}"
        );
    }
    assert!(
        value.absolute_error().into_f64() >= magnitude,
        "{case}: {:?} < {magnitude:e}",
        value.absolute_error()
    );
    // Only an exact zero claims a zero bound; any other zero covers at least
    // one quantum, which bounds every magnitude that rounds to zero.
    assert!(
        detailed.termination() == SpecialFunctionTermination::Exact
            || value.absolute_error().into_f64() > 0.0,
        "{case}: {detailed:?}"
    );
}

#[test]
fn rounded_zero_results_follow_one_rule_for_real_complex_and_binary32() {
    macro_rules! real {
        ($name:literal, $plain:ident, $detailed:ident, $options:expr, [$($arg:expr),+], $negative:expr, $magnitude:expr) => {{
            rounded_zero(
                concat!($name, " f64"),
                $plain($($arg as f64),+),
                $detailed($($arg as f64,)+ $options),
                $negative,
                $magnitude,
            );
            rounded_zero(
                concat!($name, " f32"),
                $plain($($arg as f32),+),
                $detailed($($arg as f32,)+ $options),
                $negative,
                $magnitude,
            );
        }};
    }
    macro_rules! numeric {
        ($name:literal, $plain:ident, $detailed:ident, $options:expr, [$($arg:expr),+], $negative:expr, $magnitude:expr) => {{
            real!($name, $plain, $detailed, $options, [$($arg),+], $negative, $magnitude);
            rounded_zero(
                concat!($name, " complex f64"),
                $plain($(Complex::new($arg as f64, 0.0)),+),
                $detailed($(Complex::new($arg as f64, 0.0),)+ $options),
                $negative,
                $magnitude,
            );
            rounded_zero(
                concat!($name, " complex f32"),
                $plain($(Complex::new($arg as f32, 0.0)),+),
                $detailed($(Complex::new($arg as f32, 0.0),)+ $options),
                $negative,
                $magnitude,
            );
        }};
    }
    // Ai(200) ≈ 1.9e-1639, E1(800) ≈ 2.4e-351, erfc(30) ≈ 2.6e-393,
    // K0(800) ≈ 1.6e-349, Γ(-200.5) ≈ -2.6e-376 (mpmath), Γ(1,800) = e^-800.
    numeric!(
        "Airy",
        airy_ai,
        airy_ai_with_options,
        AiryOptions::default(),
        [200.0],
        false,
        0.0
    );
    numeric!(
        "erfc",
        erfc,
        erfc_with_options,
        ErrorFunctionOptions::default(),
        [30.0],
        false,
        0.0
    );
    numeric!(
        "K",
        bessel_k,
        bessel_k_with_options,
        BesselOptions::default(),
        [0.0, 800.0],
        false,
        0.0
    );
    numeric!(
        "gamma",
        gamma,
        gamma_with_options,
        GammaOptions::default(),
        [-200.5],
        true,
        0.0
    );
    numeric!(
        "upper gamma",
        gamma_upper,
        gamma_upper_with_options,
        GammaOptions::default(),
        [1.0, 800.0],
        false,
        0.0
    );
    real!(
        "E1",
        expint_e1,
        expint_e1_with_options,
        IntegralOptions::default(),
        [800.0],
        false,
        0.0
    );
    real!(
        "Fermi",
        fermi_dirac_0,
        fermi_dirac_0_with_options,
        FermiDiracOptions::default(),
        [-800.0],
        false,
        0.0
    );
    real!(
        "synchrotron",
        synchrotron_1,
        synchrotron_1_with_options,
        SynchrotronOptions::default(),
        [800.0],
        false,
        0.0
    );
    real!(
        "transport origin",
        transport_3,
        transport_3_with_options,
        TransportOptions::default(),
        [0.0],
        false,
        0.0
    );
    rounded_zero(
        "Debye f64",
        debye_6(f64::MAX),
        debye_6_with_options(f64::MAX, DebyeOptions::default()),
        false,
        0.0,
    );
    rounded_zero(
        "Debye f32",
        debye_6(f32::MAX),
        debye_6_with_options(f32::MAX, DebyeOptions::default()),
        false,
        0.0,
    );
    // Values that are subnormal in binary64 round to zero only in binary32,
    // where the bound covers the binary64 magnitude.
    rounded_zero(
        "Airy f32 narrowing",
        airy_ai(40.0_f32),
        airy_ai_with_options(40.0_f32, AiryOptions::default()),
        false,
        airy_ai(40.0_f64).unwrap(),
    );
    rounded_zero(
        "Airy complex f32 narrowing",
        airy_ai(Complex::new(40.0_f32, 0.0)),
        airy_ai_with_options(Complex::new(40.0_f32, 0.0), AiryOptions::default()),
        false,
        airy_ai(40.0_f64).unwrap(),
    );
    rounded_zero(
        "gamma f32 narrowing",
        gamma(-40.5_f32),
        gamma_with_options(-40.5_f32, GammaOptions::default()),
        true,
        gamma(-40.5_f64).unwrap().abs(),
    );
    rounded_zero(
        "upper gamma f32 narrowing",
        gamma_upper(1.0_f32, 200.0),
        gamma_upper_with_options(1.0_f32, 200.0, GammaOptions::default()),
        false,
        (-200.0_f64).exp(),
    );
    // Relative kernel bounds vanish at zero; these tails, ζ(400,10) ≈ Φ(z,400,10)
    // ≈ 10⁻⁴⁰⁰, T(40,1) ≈ 10⁻³⁵⁰ and ψ⁽⁴⁾(10¹⁰⁰) ≈ -6·10⁻⁴⁰⁰, still cover a quantum.
    numeric!(
        "Hurwitz zeta",
        hurwitz_zeta,
        hurwitz_zeta_with_options,
        ZetaOptions::default(),
        [400.0, 10.0],
        false,
        0.0
    );
    numeric!(
        "Lerch series",
        lerch_phi,
        lerch_phi_with_options,
        LerchOptions::default(),
        [0.5, 400.0, 10.0],
        false,
        0.0
    );
    numeric!(
        "Lerch at z = 0",
        lerch_phi,
        lerch_phi_with_options,
        LerchOptions::default(),
        [0.0, 400.0, 10.0],
        false,
        0.0
    );
    real!(
        "Owen T",
        owens_t,
        owens_t_with_options,
        ProbabilityOptions::default(),
        [40.0, 1.0],
        false,
        0.0
    );
    // ψ⁽⁴⁾(10¹⁰⁰) < 0: the complex path on the real axis keeps the real sign.
    rounded_zero(
        "polygamma f64",
        polygamma(4, 1e100_f64),
        polygamma_with_options(4, 1e100_f64, GammaOptions::default()),
        true,
        0.0,
    );
    rounded_zero(
        "polygamma complex f64",
        polygamma(4, Complex::new(1e100_f64, 0.0)),
        polygamma_with_options(4, Complex::new(1e100_f64, 0.0), GammaOptions::default()),
        true,
        0.0,
    );
    // J₂₀₁(-1) = -J₂₀₁(1) and I₂₀₁(-1) = -I₂₀₁(1), both ≈ -10⁻⁴⁴⁰.
    numeric!(
        "J",
        bessel_j,
        bessel_j_with_options,
        BesselOptions::default(),
        [201.0, -1.0],
        true,
        0.0
    );
    numeric!(
        "I",
        bessel_i,
        bessel_i_with_options,
        BesselOptions::default(),
        [201.0, -1.0],
        true,
        0.0
    );
    // The logarithmic form keeps the magnitude of a rounded zero.
    let gamma = gamma_with_options(-200.5_f64, GammaOptions::default()).unwrap();
    assert!(gamma.scaled().is_some() && gamma.logarithmic().is_some());
    let upper = gamma_upper_with_options(1.0_f64, 800.0, GammaOptions::default()).unwrap();
    assert!(
        (upper
            .logarithmic()
            .unwrap()
            .value()
            .log_magnitude()
            .unwrap()
            + 800.0)
            .abs()
            < 1e-10
    );
}

#[test]
fn synchrotron_scaled_error_uses_scaled_coordinates() {
    for x in [80.0_f64, 800.0, 1000.0] {
        for report in [
            synchrotron_1_with_options(x, SynchrotronOptions::default()).unwrap(),
            synchrotron_2_with_options(x, SynchrotronOptions::default()).unwrap(),
        ] {
            let scaled = report.scaled().unwrap();
            let expected = scaled.relative_error().unwrap() * scaled.value().scaled_magnitude();
            assert!((scaled.absolute_error() - expected).abs() <= f64::EPSILON * expected);
        }
    }
}

#[test]
fn complex_log_bounds_include_destination_rounding() {
    // erfc(-100) differs from two by far less than one binary64 quantum.
    let expected = 10_000.0_f64 + core::f64::consts::LN_2;
    let report = erfcx_with_options(
        Complex::new(-100.0_f32, 0.0),
        ErrorFunctionOptions::default(),
    )
    .unwrap();
    let logarithmic = report.logarithmic().unwrap();
    let actual = f64::from(logarithmic.value().log_magnitude().unwrap());
    assert!(
        (actual - expected).abs() <= f64::from(logarithmic.absolute_error()),
        "destination log rounding: {} > {}",
        (actual - expected).abs(),
        logarithmic.absolute_error()
    );
}

#[test]
fn complete_elliptic_e_endpoint_is_exact() {
    fn check<S: RealScalar>() {
        let r = elliptic_e_with_options(S::ONE, EllipticOptions::default()).unwrap();
        assert_eq!(r.termination(), SpecialFunctionTermination::Exact);
        assert_eq!(r.try_value().unwrap(), S::ONE);
        assert_eq!(r.value().unwrap().absolute_error(), S::ZERO);
        assert_eq!(r.counts().iterations(), 0);
        let interior =
            elliptic_e_with_options(S::ONE.next_after(S::ZERO), EllipticOptions::default())
                .unwrap();
        assert_eq!(
            interior.termination(),
            SpecialFunctionTermination::Converged
        );
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn real_harmonic_error_uses_the_basis_normalization() {
    for order in [-1, 0, 1] {
        let options = LegendreOptions::default();
        let complex = spherical_harmonic_with_options(2, order, 0.5_f64, 0.25, options).unwrap();
        let real = spherical_harmonic_real_with_options(2, order, 0.5_f64, 0.25, options).unwrap();
        let factor = if order == 0 {
            1.0
        } else {
            core::f64::consts::SQRT_2
        };
        let expected_error = factor * complex.value().unwrap().absolute_error();
        if order == 0 {
            assert!(
                (real.value().unwrap().absolute_error() - expected_error).abs()
                    <= 32.0 * f64::EPSILON * expected_error
            );
        }
        let value = complex.try_value().unwrap();
        let expected = match order {
            -1 => -factor * value.im,
            0 => value.re,
            _ => -factor * value.re,
        };
        assert!(
            (real.try_value().unwrap() - expected).abs()
                <= real.value().unwrap().absolute_error() + expected_error
        );
    }
}

#[test]
fn real_polylog_family_budget_and_domain_contracts() {
    for limit in [13, 14] {
        let options =
            PolylogOptions::default().with_iteration_budget(Budget::try_from(limit).unwrap());
        for result in [
            dilog_with_options(0.3_f64, options),
            spence_with_options(0.7_f64, options),
        ] {
            if limit == 13 {
                assert!(matches!(
                    result.unwrap_err().view(),
                    SciOxideErrorView::ConvergenceFailure { .. }
                ));
            } else {
                let r = result.unwrap();
                assert_eq!(r.termination(), SpecialFunctionTermination::Converged);
                assert_eq!(r.counts().iterations(), 14);
            }
        }
    }
    for (order, expected) in [
        (2.5, scioxide_core::DomainReason::OutsideDomain),
        (2.0, scioxide_core::DomainReason::Pole),
    ] {
        let error = lerch_phi(0.5_f64, order, -1.0).unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DomainError { reason, .. } if reason == expected)
        );
    }
    assert!((lerch_phi(0.5_f64, 0.0, -1.0).unwrap() - 2.0).abs() < 1e-14);
    let error = lerch_phi(Complex::new(0.5_f64, 0.0), 2.5, -1.0).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DomainError {
            reason: scioxide_core::DomainReason::Pole,
            ..
        }
    ));
}
