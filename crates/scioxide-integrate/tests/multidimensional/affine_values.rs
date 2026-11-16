//! Whole-value, preparation and allocation contracts across multidimensional families.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, ExecutionContext, Interval};
use scioxide_integrate::*;

macro_rules! sparse_grid_stop_evidence {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let zero = scioxide_core::ErrorTolerance::scalar(
                0.0 as $s,
                0.0 as $s,
                scioxide_core::NormSelection::Infinity,
            )
            .unwrap();
            let square = [Interval::new(0.0 as $s, 1.0 as $s).unwrap(); 2];
            let cube = [Interval::new(0.0 as $s, 1.0 as $s).unwrap(); 5];
            let kink = |x: &[$s]| Ok((x[0] - 0.01).abs());
            let step = |x: &[$s]| {
                Ok(if x[0] > 0.3 || x[1] > 0.6 {
                    0.0
                } else {
                    x.iter().sum::<$s>().exp()
                })
            };
            let cases: [(
                SparseGridOptions<$s>,
                &[Interval<$s>],
                IntegrationTermination,
            ); 3] = [
                // Level exhaustion around an unresolved kink.
                (
                    SparseGridOptions::default().with_tolerance(zero.clone()),
                    &square,
                    IntegrationTermination::RefinementLimit,
                ),
                // The evaluation limit inside a forward-neighbour activation.
                (
                    SparseGridOptions::default()
                        .with_tolerance(zero.clone())
                        .with_max_level(8)
                        .with_max_indices(Budget::try_from(100_000).unwrap())
                        .with_max_evaluations(Budget::try_from(5_000).unwrap()),
                    &cube,
                    IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation),
                ),
                // The index limit before any refinement can be accepted.
                (
                    SparseGridOptions::default()
                        .with_tolerance(zero)
                        .with_max_indices(Budget::try_from(2).unwrap()),
                    &square,
                    IntegrationTermination::RefinementLimit,
                ),
            ];
            for (index, (options, bounds, termination)) in cases.into_iter().enumerate() {
                let plan = SparseGridPlan::new(bounds.len(), options).unwrap();
                let mut ws = plan.workspace(1).unwrap();
                let mut value = 0.0 as $s;
                let f: &(dyn Fn(&[$s]) -> Result<$s, scioxide_core::SciOxideError> + Sync) =
                    if bounds.len() == 2 { &kink } else { &step };
                let report = plan
                    .integrate_into(
                        f,
                        bounds,
                        &mut value,
                        &mut ws,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert_eq!(report.termination(), termination, "case {index}");
                let estimate = report.error_estimate().unwrap();
                assert!(
                    estimate.is_finite() && estimate > 0.0,
                    "case {index}: {estimate:?}"
                );
                let evidence = ws.component_errors().unwrap();
                assert_eq!(evidence[0].total(), Some(estimate));
                assert!(ws.directional_indicators().iter().any(|&d| d > 0.0));
                if index == 0 {
                    // ∫|x-c| = c² - c + 1/2; the frontier covers the unresolved kink.
                    let exact = 0.01 * 0.01 - 0.01 + 0.5;
                    assert!(
                        ((value as f64) - exact).abs() <= estimate as f64,
                        "{value:?} {estimate:?}"
                    );
                }
            }
        }
    };
}
sparse_grid_stop_evidence!(
    sparse_grid_publishes_frontier_evidence_at_every_stop_f64,
    f64
);
sparse_grid_stop_evidence!(
    sparse_grid_publishes_frontier_evidence_at_every_stop_f32,
    f32
);
