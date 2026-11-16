//! Public real/complex entry forms and read-only result diagnostics.

#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, SciOxideErrorView};
use scioxide_special::{
    GammaOptions, ScalarRepresentation, SpecialFunctionTermination, airy_ai, bessel_j, beta_inc,
    beta_inc_q, gamma, gamma_with_options,
};

#[test]
fn one_name_accepts_real_and_complex_values() {
    assert!((gamma(5.0_f64).unwrap() - 24.0).abs() < 1e-12);
    assert!((gamma(5.0_f32).unwrap() - 24.0).abs() < 1e-4);
    let complex = gamma(Complex::new(5.0_f64, 0.0)).unwrap();
    assert!((complex.re - 24.0).abs() < 1e-12);
    assert!(complex.im.abs() < 1e-12);

    let airy = airy_ai(Complex::new(0.0_f64, 0.0)).unwrap();
    assert!((airy.re - airy_ai(0.0_f64).unwrap()).abs() < 1e-12);
    let bessel = bessel_j(Complex::new(0.0_f64, 0.0), Complex::new(2.0, 0.0)).unwrap();
    assert!((bessel.re - bessel_j(0.0_f64, 2.0).unwrap()).abs() < 1e-12);
}

#[test]
fn logarithmic_gamma_uses_the_detailed_result() {
    let report = gamma_with_options(172.0_f64, GammaOptions::default()).unwrap();
    assert_eq!(report.termination(), SpecialFunctionTermination::Converged);
    assert_eq!(
        report.scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(report.value().is_none());
    assert!(
        report
            .logarithmic()
            .unwrap()
            .value()
            .log_magnitude()
            .unwrap()
            > 700.0
    );
    assert!(matches!(
        gamma(172.0_f64).unwrap_err().view(),
        SciOxideErrorView::NumericalInstability { .. }
    ));
}

#[test]
fn upper_beta_tail_has_q_names() {
    let (a, b, x) = (2.5_f64, 3.5, 0.4);
    let lower = beta_inc(a, b, x).unwrap();
    let upper = beta_inc_q(a, b, x).unwrap();
    assert!((lower + upper - 1.0).abs() < 1e-13);
    let detailed = scioxide_special::beta_inc_q_with_options(
        a,
        b,
        x,
        scioxide_special::BetaOptions::default(),
    )
    .unwrap();
    assert_eq!(
        detailed.termination(),
        SpecialFunctionTermination::Converged
    );
}

#[test]
fn scaled_hankel_real_inputs_keep_the_unscaled_domain() {
    for result in [
        scioxide_special::hankel_1_scaled(0.0_f64, -1.0),
        scioxide_special::hankel_2_scaled(0.0_f64, -1.0),
    ] {
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
    let options = scioxide_special::BesselOptions::default();
    assert_eq!(
        scioxide_special::hankel_1_scaled_with_options(0.0_f64, -1.0, options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DomainError
    );
    let z = scioxide_core::Complex::new(-1.0_f64, 0.0);
    assert!(
        scioxide_special::hankel_1_scaled(scioxide_core::Complex::new(0.0, 0.0), z)
            .unwrap()
            .is_finite()
    );
    let actual = scioxide_special::hankel_1_scaled(0.0_f64, 1.0).unwrap();
    let expected = scioxide_special::hankel_1(0.0_f64, 1.0).unwrap()
        * scioxide_core::Complex::new(1.0_f64.cos(), -1.0_f64.sin());
    assert!((actual - expected).norm() < 1e-12);
}
