#![allow(clippy::unwrap_used)]
#[path = "support/exact_field.rs"]
mod exact_field;

use scioxide_poly::{
    ApproxPolynomial, ApproxRationalFunction, BarycentricBasis, BernsteinBasis, ExactCoefficient,
    ExactField, ExactPolynomial, ExactRationalFunction, MonomialBasis, NewtonBasis,
    OrthogonalBasis, PolynomialApproximationPolicy, PolynomialBasis, PolynomialConditionReport,
    PolynomialConversion, PolynomialEvaluation,
};

#[test]
fn basis_trait_borrows_stored_coordinates_without_conversion() {
    fn coordinates<B: PolynomialBasis<f64>>(p: &ApproxPolynomial<f64, B>) -> &[f64] {
        p.basis().coefficients()
    }
    let policy = PolynomialApproximationPolicy::default();
    let stored = vec![1.0, 3.0, 2.0];
    let monomial = ApproxPolynomial::monomial_with_options(stored.clone(), policy).unwrap();
    let newton =
        ApproxPolynomial::newton_with_options(stored.clone(), vec![2.0, 4.0], policy).unwrap();
    let bernstein =
        ApproxPolynomial::bernstein_with_options(stored.clone(), 2.0, 4.0, policy).unwrap();
    let barycentric = ApproxPolynomial::interpolate_barycentric_with_options(
        vec![2.0, 3.0, 4.0],
        stored.clone(),
        policy,
    )
    .unwrap();
    let spec = scioxide_poly::OrthogonalBasisSpec::new(
        scioxide_poly::OrthogonalFamily::Legendre,
        scioxide_poly::BasisNormalization::Classical,
        2.0,
        4.0,
    )
    .unwrap();
    let orthogonal = ApproxPolynomial::orthogonal_with_plan(
        stored.clone(),
        spec.prepare(stored.len() - 1, Default::default()).unwrap(),
        policy,
    )
    .unwrap();
    for actual in [
        coordinates(&monomial),
        coordinates(&newton),
        coordinates(&bernstein),
        coordinates(&barycentric),
        coordinates(&orthogonal),
    ] {
        assert_eq!(actual, stored);
    }
}

#[test]
fn public_surface_names_exact_and_approximate_domains_explicitly() {
    fn require_basis<C: scioxide_core::NumericScalar, B: PolynomialBasis<C>>() {}
    require_basis::<f64, MonomialBasis<f64>>();
    require_basis::<f64, NewtonBasis<f64>>();
    require_basis::<f64, BarycentricBasis<f64>>();
    require_basis::<f64, OrthogonalBasis<f64>>();
    require_basis::<f64, BernsteinBasis<f64>>();

    let policy = PolynomialApproximationPolicy::<f64>::default();
    let polynomial = ApproxPolynomial::monomial_with_options(vec![1.0, 2.0], policy).unwrap();
    let _: PolynomialEvaluation<f64> = polynomial.evaluate_with_derivative(0.5).unwrap();
    let conversion: PolynomialConversion<f64, MonomialBasis<f64>> =
        polynomial.to_monomial().unwrap();
    let _: PolynomialConditionReport<f64> = conversion.condition();
    let _: PolynomialConversion<f64, NewtonBasis<f64>> =
        polynomial.to_newton(vec![0.0, 1.0]).unwrap();
    let _: PolynomialConversion<f64, BarycentricBasis<f64>> =
        polynomial.to_barycentric(vec![0.0, 1.0]).unwrap();
    let _: PolynomialConversion<f64, OrthogonalBasis<f64>> = polynomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::ChebyshevT,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let _: PolynomialConversion<f64, OrthogonalBasis<f64>> = polynomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::Legendre,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let _: PolynomialConversion<f64, BernsteinBasis<f64>> =
        polynomial.to_bernstein(0.0, 1.0).unwrap();
    let matrix: scioxide_linalg::Matrix<f64> = polynomial.root_matrix().unwrap();
    assert_eq!(
        (matrix.rows(), matrix.cols()),
        (polynomial.degree(), polynomial.degree())
    );

    let rational = ApproxRationalFunction::new(
        ApproxPolynomial::monomial_with_options(vec![1.0], policy).unwrap(),
        ApproxPolynomial::monomial_with_options(vec![2.0, 1.0], policy).unwrap(),
    )
    .unwrap();
    assert!((rational.evaluate(0.0).unwrap() - 0.5_f64).abs() < 1e-14);

    fn exact_contract<C: ExactCoefficient, F: ExactField>() {
        assert!(core::mem::size_of::<ExactPolynomial<C>>() > 0);
        assert!(core::mem::size_of::<ExactRationalFunction<F>>() > 0);
    }
    exact_contract::<exact_field::Q, exact_field::Q>();
}
