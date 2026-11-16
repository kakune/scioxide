#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, RankThreshold, Tolerance};
use scioxide_linalg::RankPolicy;
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{
    ApproxPolynomial, ApproxRationalFunction, PolynomialApproximationPolicy, approximate_gcd,
};

fn policy<R: scioxide_core::RealScalar>() -> PolynomialApproximationPolicy<R> {
    PolynomialApproximationPolicy::new(
        Tolerance::from_sqrt_epsilon(),
        RankPolicy::DimensionScaledEpsilon,
    )
}

macro_rules! scalar_contract {
    ($name:ident, $scalar:ty, $error:expr) => {
        #[test]
        fn $name() {
            let polynomial = ApproxPolynomial::from_roots_with_options(
                &[-2.0 as $scalar, 0.5, 3.0],
                policy::<$scalar>(),
            )
            .unwrap();
            for root in [-2.0 as $scalar, 0.5, 3.0] {
                let evaluation = polynomial.evaluate_with_derivative(root).unwrap();
                assert!(evaluation.value().abs() <= $error);
                assert_eq!(
                    polynomial
                        .classify_coefficients()
                        .unwrap()
                        .effective_degree(),
                    3
                );
            }
            let matrix = polynomial.root_matrix().unwrap();
            let roots =
                scioxide_linalg::GeneralEigenvaluesResult::<$scalar>::compute(&matrix).unwrap();
            let mut actual = roots.eigenvalues().to_vec();
            assert_eq!(actual.len(), 3);
            actual.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
            for (root, expected) in actual.iter().zip([-2.0 as $scalar, 0.5, 3.0]) {
                assert!((root.re - expected).abs() <= $error);
                assert!(root.im.abs() <= $error);
            }
        }
    };
}

scalar_contract!(prepared_polynomial_contract_f32, f32, 1.0e-4);
scalar_contract!(prepared_polynomial_contract_f64, f64, 1.0e-12);

#[test]
fn complex_basis_and_approximate_rational_are_closed() {
    let policy = policy::<f64>();
    let numerator = ApproxPolynomial::monomial_with_options(
        vec![Complex::new(1.0, 2.0), Complex::new(1.0, 0.0)],
        policy,
    )
    .unwrap();
    let denominator = ApproxPolynomial::monomial_with_options(
        vec![Complex::new(2.0, -1.0), Complex::new(1.0, 0.0)],
        policy,
    )
    .unwrap();
    let rational = ApproxRationalFunction::new(numerator, denominator).unwrap();
    // (1.5 + 2.25i)/(2.5 - 0.75i) = (33 + 108i)/109.
    let value = rational.evaluate(Complex::new(0.5, 0.25)).unwrap();
    assert!((value.re - 33.0 / 109.0).abs() < 1e-14);
    assert!((value.im - 108.0 / 109.0).abs() < 1e-14);
}

#[test]
fn approximate_gcd_reports_the_selected_rank_policy() {
    let policy = PolynomialApproximationPolicy::new(
        Tolerance::new(0.0_f64, 1.0e-8).unwrap(),
        RankPolicy::Threshold(RankThreshold::new(0.0, 1.0e-8).unwrap()),
    );
    let common = ApproxPolynomial::from_roots_with_options(&[1.0_f64, 1.0], policy).unwrap();
    let left = common
        .try_mul(&ApproxPolynomial::from_roots_with_options(&[-2.0], policy).unwrap())
        .unwrap();
    let right = common
        .try_mul(&ApproxPolynomial::from_roots_with_options(&[3.0], policy).unwrap())
        .unwrap();
    let result = approximate_gcd(&left, &right).unwrap();
    assert_eq!(result.report().effective_degree(), 2);
    assert_eq!(result.report().policy(), policy);
    assert!(result.report().backward_error() <= 1.0e-8);
}

#[test]
fn errors_report_rejected_arguments() {
    use scioxide_core::argument;
    use scioxide_poly::{OrthogonalFamily, PolynomialOperationPlan};

    let error = OrthogonalFamily::<f64>::ChebyshevT.zeros(0).unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _view_argument,
        reason: _view_reason,
        detail: _view_detail,
    } = error.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_view_argument, argument!("n"));

    // A convenience constructor reports the entry point it forwards to.
    let error =
        PolynomialOperationPlan::<f64>::new(&[PolynomialOperationRequest::TaylorShift { len: 0 }])
            .map(|_| ())
            .unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _view_argument,
        reason: _view_reason,
        detail: _view_detail,
    } = error.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_view_argument, argument!("len"));
}
