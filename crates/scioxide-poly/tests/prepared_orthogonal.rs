//! Owned orthogonal expansions retain authoritative prepared configuration.
#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;
use scioxide_core::{ExecutionPolicy, SciOxideErrorKind};
use scioxide_poly::{
    ApproxPolynomial, BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily, PolynomialBasis,
    PolynomialExecutionOptions, PolynomialFitPlan,
};

fn options() -> PolynomialExecutionOptions {
    PolynomialExecutionOptions::new()
        .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()))
        .with_worker_bound(NonZeroUsize::new(8).unwrap())
}

fn spec() -> OrthogonalBasisSpec<f64> {
    OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0,
        1.0,
    )
    .unwrap()
}

#[test]
fn construction_retains_the_full_plan_for_shorter_expansions() {
    let plan = spec().prepare(6, options()).unwrap();
    let polynomial = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0_f64, 0.5, -0.25],
        plan.clone(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(polynomial.basis().plan(), &plan);
    assert_eq!(polynomial.degree(), 2);
    assert!((polynomial.evaluate(0.5).unwrap() - 1.375).abs() <= f64::EPSILON);
    let constant =
        ApproxPolynomial::orthogonal_with_plan(vec![2.0_f64], plan.clone(), Default::default())
            .unwrap();
    assert_eq!(constant.basis().plan(), &plan);
    assert_eq!(constant.degree(), 0);
}

/// Calculus and conversion of an expansion shorter than its prepared plan
/// agree with the same coefficients on an exactly sized plan.
#[test]
fn shorter_expansions_support_calculus_and_conversion() {
    let families = [
        OrthogonalFamily::ChebyshevT,
        OrthogonalFamily::Legendre,
        OrthogonalFamily::HermiteHe,
        OrthogonalFamily::GeneralizedLaguerre { alpha: 0.5 },
        OrthogonalFamily::Jacobi {
            alpha: 0.25,
            beta: -0.5,
        },
    ];
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    };
    for family in families {
        let spec =
            OrthogonalBasisSpec::new(family, BasisNormalization::Classical, 0.25, 1.5).unwrap();
        for coefficients in [vec![1.0_f64], vec![1.0, 2.0, 1.0]] {
            let degree = coefficients.len() - 1;
            let long = ApproxPolynomial::orthogonal_with_plan(
                coefficients.clone(),
                spec.prepare(degree + 4, options()).unwrap(),
                Default::default(),
            )
            .unwrap();
            let exact = ApproxPolynomial::orthogonal_with_plan(
                coefficients,
                spec.prepare(degree, options()).unwrap(),
                Default::default(),
            )
            .unwrap();
            let derivative = long.derivative().unwrap();
            assert_eq!(
                derivative.basis().coefficients(),
                exact.derivative().unwrap().basis().coefficients()
            );
            assert_eq!(derivative.basis().plan().options(), options());
            let primitive = long.antiderivative(0.5, 2.0).unwrap();
            assert_eq!(
                primitive.basis().coefficients(),
                exact
                    .antiderivative(0.5, 2.0)
                    .unwrap()
                    .basis()
                    .coefficients()
            );
            assert_eq!(primitive.basis().plan().options(), options());
            close(
                long.integral(-0.5, 1.0).unwrap(),
                exact.integral(-0.5, 1.0).unwrap(),
            );
            let monomial = long.to_monomial().unwrap();
            for point in [-0.5, 0.25, 1.0] {
                close(
                    monomial.polynomial().evaluate(point).unwrap(),
                    exact.evaluate(point).unwrap(),
                );
            }
        }
    }
}

#[test]
fn construction_rejects_coefficients_outside_the_prepared_contract() {
    let plan = spec().prepare(1, options()).unwrap();
    let empty =
        ApproxPolynomial::orthogonal_with_plan(Vec::<f64>::new(), plan.clone(), Default::default())
            .unwrap_err();
    assert!(matches!(
        empty.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Empty,
            ..
        }
    ));
    for coefficients in [vec![f64::NAN], vec![f64::INFINITY], vec![f64::NEG_INFINITY]] {
        assert_eq!(
            ApproxPolynomial::orthogonal_with_plan(coefficients, plan.clone(), Default::default())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert!(matches!(
        ApproxPolynomial::orthogonal_with_plan(vec![1.0_f64, 2.0, 3.0], plan, Default::default())
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Inconsistent,
            ..
        }
    ));
}

#[test]
fn repeated_orthogonal_fits_retain_degree_options_and_recurrence() {
    let basis = spec().prepare(2, options()).unwrap();
    let fit = PolynomialFitPlan::orthogonal(
        &[-1.0_f64, 0.0, 1.0],
        basis.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    for values in [[2.0, -1.0, 2.0], [1.0, 1.0, 1.0]] {
        let result = fit.fit(&values).unwrap();
        assert_eq!(result.polynomial().basis().plan(), &basis);
        assert_eq!(result.polynomial().degree(), 2);
        for (point, expected) in [-1.0, 0.0, 1.0].into_iter().zip(values) {
            let actual = result.polynomial().evaluate(point).unwrap();
            assert!((actual - expected).abs() <= 128.0 * f64::EPSILON);
        }
    }
}

#[test]
fn conversions_use_documented_default_or_inherited_options() {
    let monomial = ApproxPolynomial::monomial(vec![1.0_f64, 0.5, -0.25]).unwrap();
    let converted = monomial.to_orthogonal(spec()).unwrap();
    assert_eq!(
        converted.polynomial().basis().plan().options(),
        PolynomialExecutionOptions::new()
    );
    assert_eq!(converted.polynomial().basis().plan().degree(), 2);
    let bernstein = ApproxPolynomial::bernstein(vec![1.0_f64, 0.5, -0.25], -1.0, 1.0).unwrap();
    let converted = bernstein.to_orthogonal(spec()).unwrap();
    assert_eq!(
        converted.polynomial().basis().plan().options(),
        PolynomialExecutionOptions::new()
    );
    let source = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0_f64, 0.5, -0.25],
        spec().prepare(6, options()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let target = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Monic,
        0.5,
        2.0,
    )
    .unwrap();
    let converted = source.to_orthogonal(target).unwrap();
    assert_eq!(converted.polynomial().basis().plan().options(), options());
    assert_eq!(
        converted.polynomial().basis().plan().degree(),
        source.degree()
    );
    for point in [-1.0, 0.0, 0.5, 1.0] {
        let expected = source.evaluate(point).unwrap();
        let actual = converted.polynomial().evaluate(point).unwrap();
        assert!((actual - expected).abs() <= 32.0 * f64::EPSILON);
    }
}
