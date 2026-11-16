//! Cross-family integration behavior and option validation.
#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, Interval};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, CubaturePlan, FourierDomain, FourierTailOptions,
    FourierTailPlan, FourierWeight, GaussKronrod, GaussKronrodRule, GeometricOptions,
    ProductOptions, ProductRule, SimplexRule, SparseGridOptions, SparseGridPlan,
    SubdivisionOptions,
};

#[test]
fn cubature_one_dimension_matches_adaptive_and_product_shares_embedded_estimates() {
    let options = SubdivisionOptions::<f64>::default();
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let adaptive = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone()).unwrap();
    let cube = CubaturePlan::new(&bounds, options).unwrap();
    let (a, ar) = adaptive.integrate(|x: f64| Ok(x.exp()), bounds[0]).unwrap();
    let (b, br) = cube.integrate(|x: &[f64]| Ok(x[0].exp())).unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    assert_eq!(ar.evaluations(), br.evaluations());
    assert_eq!(ar.termination(), br.termination());
    assert_eq!(ar.error_estimate(), br.error_estimate());
    let product = ProductRule::new(
        GaussKronrod::new(GaussKronrodRule::G7K15).unwrap(),
        ProductOptions::default(),
    );
    let mut ws = product.workspace((1, 2)).unwrap();
    let mut out = [0.0; 2];
    let report = product
        .integrate_into(
            |x: &[f64]| Ok([x[0].exp(), 2.0 * x[0].exp()]),
            &bounds,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.error_estimate().is_some());
    assert_eq!(out[1], 2.0 * out[0]);
    assert!(ws.component_errors().is_some());
}

#[test]
fn options_are_validated_by_plan_construction() {
    assert!(
        SparseGridPlan::<f64>::new(2, SparseGridOptions::default().with_max_level(53)).is_err()
    );
    assert!(
        SimplexRule::<f64>::new(
            2,
            1,
            GeometricOptions::default().with_predicate_terms(scioxide_core::Budget::ONE)
        )
        .is_err()
    );
    assert!(
        FourierTailPlan::<f64>::new(
            FourierDomain::Upper { lower: 0.0 },
            1.0,
            FourierWeight::Sine,
            FourierTailOptions::default().with_max_cycles(scioxide_core::Budget::ONE)
        )
        .is_err()
    );
}
