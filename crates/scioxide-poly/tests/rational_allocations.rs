//! The first prepared execution and subsequent executions must not allocate.
#![allow(clippy::unwrap_used)]

use scioxide_poly::*;

#[test]
fn prepared_decomposition_has_no_hidden_growth() {
    let policy = PolynomialApproximationPolicy::default();
    let a = ApproxPolynomial::monomial_with_options(vec![-1.0_f64, 1.0], policy).unwrap();
    let b = ApproxPolynomial::monomial_with_options(vec![1.0, 0.0, 1.0], policy).unwrap();
    let d = a.try_mul(&a).unwrap().try_mul(&b).unwrap();
    let plan = ApproxPartialFractionPlan::new(&d, &[(&a, 2), (&b, 1)]).unwrap();
    let requirements = plan.workspace_requirements(8).unwrap();
    let mut output = ApproxPartialFractions::new();
    let mut workspace = PartialFractionWorkspace::new();
    output.try_reserve(&requirements).unwrap();
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    {
        plan.decompose_into(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
    {
        plan.decompose_into(
            &[2.0],
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    };
}
