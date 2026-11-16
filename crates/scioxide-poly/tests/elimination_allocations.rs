//! First and subsequent prepared executions must allocate no memory.
#![allow(clippy::unwrap_used)]

use scioxide_core::ExecutionContext;
use scioxide_poly::{PolynomialApproximationPolicy, SylvesterDiscriminant, SylvesterElimination};

#[test]
fn prepared_storage_is_sufficient() {
    let plan = SylvesterElimination::<f64>::new(3, 2, 0, PolynomialApproximationPolicy::default())
        .unwrap();
    let mut ws = plan.workspace().unwrap();
    for a in [[1., 2., 3., 1.], [2., -1., 1., 2.]] {
        {
            let result = plan
                .coefficient_scoped(
                    &a,
                    &[-2., 1., 1.],
                    0,
                    &mut ws,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(result.report().degrees(), [3, 2]);
        };
    }
}
#[test]
fn prepared_discriminant_has_no_hidden_growth() {
    let plan =
        SylvesterDiscriminant::<f64>::new(3, PolynomialApproximationPolicy::default()).unwrap();
    let mut ws = plan.workspace().unwrap();
    for a in [[1., 2., 3., 1.], [2., -1., 1., 2.]] {
        {
            let result = plan
                .compute_scoped(&a, &mut ws, ExecutionContext::sequential())
                .unwrap();
            assert_eq!(result.report().degrees(), [3, 2]);
        };
    }
}
