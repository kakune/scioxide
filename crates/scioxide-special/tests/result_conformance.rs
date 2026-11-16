//! Public detailed-result conformance across every special-function family.

#![allow(clippy::unwrap_used)]

#[path = "support/publication.rs"]
mod publication;
use scioxide_core::Budget;
use scioxide_special::*;

#[test]
fn scaled_log_derivative_and_complex_representations_agree() {
    let result = erf_with_options(0.75_f64, ErrorFunctionOptions::default()).unwrap();
    let value = *result.value().unwrap().value();
    assert_eq!(
        result
            .scaled()
            .unwrap()
            .value()
            .try_value()
            .unwrap()
            .to_bits(),
        value.to_bits()
    );
    assert!((result.logarithmic().unwrap().value().try_value().unwrap() - value).abs() < 2.0e-15);
    let expected_derivative = 2.0 / std::f64::consts::PI.sqrt() * (-0.75_f64.powi(2)).exp();
    assert!((*result.derivative().unwrap().value() - expected_derivative).abs() < 2.0e-15);

    let first = hankel_1_with_options(2.5_f64, 10.0, BesselOptions::default()).unwrap();
    let second = hankel_2_with_options(2.5_f64, 10.0, BesselOptions::default()).unwrap();
    let first_value = *first.value().unwrap().value();
    let second_value = *second.value().unwrap().value();
    assert_eq!(first_value.re.to_bits(), second_value.re.to_bits());
    assert_eq!(first_value.im.to_bits(), (-second_value.im).to_bits());
    assert_eq!(
        first.scaled().unwrap().value().try_value().unwrap(),
        first_value
    );
}

#[test]
fn budget_exhaustion_without_an_estimate_is_an_error() {
    for result in [
        riemann_zeta_with_options(
            2.0_f64,
            ZetaOptions::default().with_iteration_budget(Budget::ONE),
        ),
        elliptic_f_with_options(
            0.5,
            0.5,
            EllipticOptions::default().with_iteration_budget(Budget::ONE),
        ),
    ] {
        assert_eq!(
            publication::exhausted_counts(&result.unwrap_err()).iterations(),
            1
        );
    }
}

#[test]
fn f32_detailed_result_uses_the_same_contract() {
    let scalar = airy_ai(-16_777_216.0_f32).unwrap();
    let detailed = airy_ai_with_options(-16_777_216.0_f32, AiryOptions::default()).unwrap();
    assert_eq!(scalar.to_bits(), detailed.try_value().unwrap().to_bits());
}

#[test]
fn derivative_projection_returns_the_derivative_or_a_structured_error() {
    let result = erf_with_options(0.75_f64, ErrorFunctionOptions::default()).unwrap();
    assert_eq!(
        result.try_derivative().unwrap().to_bits(),
        result.derivative().unwrap().value().to_bits()
    );
    let ai = airy_ai_with_options(1.0_f64, AiryOptions::default()).unwrap();
    assert_eq!(
        ai.try_derivative().unwrap().to_bits(),
        ai.derivative().unwrap().value().to_bits()
    );

    // Bi(x) is finite at the last value below its overflow edge, but Bi′(x) is not.
    let last_finite = f64::from_bits(0x405a_1bea_c387_f62c);
    let bi = airy_bi_with_options(last_finite, AiryOptions::default()).unwrap();
    assert!(bi.try_value().is_ok());
    let error = bi.try_derivative().unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ),
        "{error}"
    );

    let tiny = ProbabilityOptions::default().with_iteration_budget(Budget::ONE);
    for report in [
        normal_inv_log_cdf_with_options(-1000., tiny).unwrap(),
        owens_t_with_options(2., 0.5, tiny).unwrap(),
    ] {
        assert_eq!(
            report.termination(),
            SpecialFunctionTermination::IterationLimit
        );
        let error = report.try_derivative().unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::IterationLimit,
                    ..
                }
            ),
            "{error}"
        );
    }
}
