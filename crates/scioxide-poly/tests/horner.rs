//! Shared monomial evaluation contracts at ordinary and exceptional ranges.
#![allow(clippy::unwrap_used)]

#[path = "support/tolerance.rs"]
mod tolerance;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorView};
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{
    ApproxPolynomial, PolynomialApproximationPolicy, PolynomialOperationPlan,
    PolynomialRootBasisRef, checked_basis_evaluation,
};
use tolerance::assert_close;

fn complex_case<R: RealScalar>() {
    let c = [
        Complex::new(R::ONE, R::TWO),
        Complex::new(R::TWO, -R::ONE),
        Complex::new(-R::ONE, R::HALF),
    ];
    let z = Complex::new(R::HALF, R::HALF);
    let expected = c[0] + c[1] * z + c[2] * z * z;
    let derivative = c[1] + (c[2] * z) * R::TWO;
    let bound = c[0].abs() + c[1].abs() * z.abs() + c[2].abs() * z.abs() * z.abs();
    let polynomial = ApproxPolynomial::monomial(c.to_vec()).unwrap();
    let approximate = polynomial.evaluate_with_derivative(z).unwrap();
    let checked = checked_basis_evaluation(&c, z, PolynomialRootBasisRef::Monomial).unwrap();
    assert_close(checked.value().unwrap().re, expected.re, 64);
    assert_close(checked.value().unwrap().im, expected.im, 64);
    assert_close(checked.derivative().unwrap().re, derivative.re, 64);
    assert_close(checked.derivative().unwrap().im, derivative.im, 64);
    assert_close(
        checked.second_derivative().unwrap().re,
        c[2].re * R::TWO,
        64,
    );
    assert_close(
        checked.second_derivative().unwrap().im,
        c[2].im * R::TWO,
        64,
    );
    assert_close(checked.normalized_residual(), expected.abs() / bound, 64);
    assert_close(approximate.value().re, checked.value().unwrap().re, 64);
    assert_close(approximate.value().im, checked.value().unwrap().im, 64);
    assert_close(
        approximate.derivative().re,
        checked.derivative().unwrap().re,
        64,
    );
    assert_close(
        approximate.derivative().im,
        checked.derivative().unwrap().im,
        64,
    );
    let direct = polynomial.root_condition(z).unwrap();
    let retained = checked.root_condition().unwrap();
    assert_close(
        direct.coefficient().value().unwrap(),
        retained.coefficient().value().unwrap(),
        64,
    );
    assert_close(
        direct.absolute().value().unwrap(),
        retained.absolute().value().unwrap(),
        64,
    );
    assert_close(
        direct.relative().value().unwrap(),
        retained.relative().value().unwrap(),
        64,
    );
    assert_close(
        approximate.condition().condition_estimate(),
        bound / expected.abs(),
        64,
    );
}

#[test]
fn complex_points_and_coefficients_in_both_widths() {
    complex_case::<f32>();
    complex_case::<f64>();
    let z = Complex::new(0.25_f64, 0.5);
    let checked = checked_basis_evaluation(
        &[1.0, -2.0, 3.0],
        z,
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    let expected = Complex::new(1.0, 0.0) - z * 2.0 + z * z * 3.0;
    assert_close(checked.value().unwrap().re, expected.re, 64);
    assert_close(checked.value().unwrap().im, expected.im, 64);
}

fn real_scale<R: RealScalar>(scale: R) {
    let coefficients = [scale, -scale, scale];
    let polynomial = ApproxPolynomial::monomial(coefficients.to_vec()).unwrap();
    let approximate = polynomial.evaluate_with_derivative(R::ONE).unwrap();
    let checked = checked_basis_evaluation(
        &coefficients,
        Complex::new(R::ONE, R::ZERO),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    // Divide before comparing so tiny-scale regressions cannot pass an absolute tolerance.
    assert_close(approximate.value() / scale, R::ONE, 64);
    assert_close(approximate.derivative() / scale, R::ONE, 64);
    assert_close(checked.value().unwrap().re / scale, R::ONE, 64);
    assert_close(checked.derivative().unwrap().re / scale, R::ONE, 64);
    assert_close(checked.second_derivative().unwrap().re / scale, R::TWO, 64);
    assert_close(
        checked.normalized_residual(),
        R::ONE / (R::TWO + R::ONE),
        64,
    );
    assert_close(
        approximate.condition().condition_estimate(),
        R::TWO + R::ONE,
        64,
    );
}

#[test]
fn extreme_scales_and_subnormal_components_survive_restart() {
    for scale in [2.0_f64.powi(-1000), 2.0_f64.powi(1000)] {
        real_scale(scale);
    }
    for scale in [2.0_f32.powi(-100), 2.0_f32.powi(100)] {
        real_scale(scale);
    }
    let tiny = f64::from_bits(1);
    let c = [Complex::new(f64::MAX * 0.5, -tiny)];
    let checked = checked_basis_evaluation(
        &c,
        Complex::new(0.0, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_eq!(checked.value().unwrap().im.to_bits(), (-tiny).to_bits());
    let checked = checked_basis_evaluation(
        &[-tiny, -tiny],
        Complex::new(tiny, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_eq!(checked.value().unwrap().re.to_bits(), (-tiny).to_bits());
    assert_eq!(
        checked.derivative().unwrap().re.to_bits(),
        (-tiny).to_bits()
    );
    let checked = checked_basis_evaluation(
        &[0.0, -tiny],
        Complex::new(0.5, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_eq!(checked.value().unwrap().re.to_bits(), (-0.0_f64).to_bits());
    // Final scalar underflow must not turn a nonzero wide residual into zero.
    assert_close(checked.normalized_residual(), 1.0, 64);
}

#[test]
fn native_overflow_can_leave_representable_checked_outputs() {
    let a = f64::MAX * 0.75;
    assert!((a * 1.5).is_infinite());
    let checked = checked_basis_evaluation(
        &[-a, a],
        Complex::new(1.5, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_close(checked.value().unwrap().re / a, 0.5, 64);
    assert_close(checked.derivative().unwrap().re / a, 1.0, 64);
    assert_close(checked.normalized_residual(), 0.2, 64);

    // p(x) = a*x^64*(x-2)^3. Its positive envelope overflows during
    // ordinary Horner, while p(2), p'(2) and p''(2) are all exactly zero.
    let a = 2.0_f64.powi(1018);
    let mut c = [0.0; 68];
    c[64..].copy_from_slice(&[-8.0 * a, 12.0 * a, -6.0 * a, a]);
    let checked = checked_basis_evaluation(
        &c,
        Complex::new(2.0, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_eq!(checked.value().unwrap().re, 0.0);
    assert_eq!(checked.derivative().unwrap().re, 0.0);
    assert_eq!(checked.second_derivative().unwrap().re, 0.0);
    assert_eq!(checked.normalized_residual(), 0.0);
}

#[test]
fn callers_project_only_their_requested_outputs() {
    let coefficients = [0.0, 0.0, f64::MAX];
    let polynomial = ApproxPolynomial::monomial(coefficients.to_vec()).unwrap();
    let evaluation = polynomial.evaluate_with_derivative(0.0).unwrap();
    assert_eq!(evaluation.value(), 0.0);
    assert_eq!(evaluation.derivative(), 0.0);
    let checked = checked_basis_evaluation(
        &coefficients,
        Complex::new(0.0, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap();
    assert_eq!(checked.value().unwrap().re, 0.0);
    assert!(checked.second_derivative().is_err());
    let plan = PolynomialOperationPlan::new(&[
        PolynomialOperationRequest::Compose {
            outer_len: 3,
            inner_len: 1,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 2,
            inner_len: 1,
        },
    ])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut output = [99.0];
    plan.compose_into(
        &coefficients,
        &[0.0],
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [0.0]);
    let a = f64::MAX * 0.75;
    plan.compose_into(
        &[-a, a],
        &[1.5],
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_close(output[0] / a, 0.5, 64);
    let before = output;
    assert!(
        plan.compose_into(
            &[0.0, f64::MAX],
            &[2.0],
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output, before);
}

#[test]
fn empty_nonfinite_constant_and_signed_zero_inputs() {
    assert!(matches!(
        checked_basis_evaluation::<f64>(
            &[],
            Complex::new(0.0, 0.0),
            scioxide_poly::PolynomialRootBasisRef::Monomial
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    for c in [[f64::NAN, 1.0], [0.0, f64::INFINITY]] {
        assert!(matches!(
            checked_basis_evaluation(
                &c,
                Complex::new(0.0, 0.0),
                scioxide_poly::PolynomialRootBasisRef::Monomial
            )
            .unwrap_err()
            .view(),
            SciOxideErrorView::NonFinite { .. }
        ));
    }
    for point in [
        Complex::new(f64::INFINITY, 0.0),
        Complex::new(0.0, f64::NAN),
    ] {
        assert!(
            checked_basis_evaluation(
                &[1.0],
                point,
                scioxide_poly::PolynomialRootBasisRef::Monomial
            )
            .is_err()
        );
    }
    for zero in [0.0_f64, -0.0_f64] {
        let c = [zero, -zero, zero];
        let checked = checked_basis_evaluation(
            &c,
            Complex::new(zero, -zero),
            scioxide_poly::PolynomialRootBasisRef::Monomial,
        )
        .unwrap();
        assert_eq!(checked.value().unwrap().re, 0.0);
        assert_eq!(checked.derivative().unwrap().re, 0.0);
        assert_eq!(checked.normalized_residual().to_bits(), 0.0_f64.to_bits());
        let constant = checked_basis_evaluation(
            &[3.0],
            Complex::new(zero, -zero),
            scioxide_poly::PolynomialRootBasisRef::Monomial,
        )
        .unwrap();
        assert_eq!(constant.value().unwrap(), Complex::new(3.0, 0.0));
        assert_eq!(constant.derivative().unwrap().re, 0.0);
        assert_eq!(constant.second_derivative().unwrap().re, 0.0);
    }
}

#[test]
fn immutable_coefficient_classification_and_preparation_errors() {
    let policy = PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::new(0.25, 0.0).unwrap(),
        scioxide_linalg::RankPolicy::Threshold(
            scioxide_core::RankThreshold::new(0.125, 0.0).unwrap(),
        ),
    );
    let polynomial =
        ApproxPolynomial::monomial_with_options(vec![1.0, 0.125, 0.5, 0.01], policy).unwrap();
    for x in [0.0, 1.0, -1.0] {
        let report = polynomial.classify_coefficients().unwrap();
        assert!(polynomial.evaluate_with_derivative(x).is_ok());
        assert_eq!(polynomial.policy(), report.policy());
        assert_eq!(report.effective_degree(), 2);
        assert_eq!(report.effective_rank(), 2);
        assert_eq!(report.coefficient_scale(), 1.0);
        assert_eq!(report.policy(), policy);
    }
    assert_close(
        polynomial
            .evaluate_with_derivative(0.0)
            .unwrap()
            .condition()
            .backward_error()
            / (4.0 * f64::EPSILON),
        1.0,
        64,
    );
    let overflow_policy = PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::new(0.0, 2.0).unwrap(),
        scioxide_linalg::RankPolicy::Exact,
    );
    assert!(matches!(
        ApproxPolynomial::monomial_with_options(vec![f64::MAX], overflow_policy)
            .unwrap_err()
            .view(),
        SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
}
