//! Good-Broyden regressions on independently specified MINPACK systems.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
#[path = "support/mgh.rs"]
mod mgh;
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance, StepTolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Broyden, BroydenOptions, BroydenUpdate, SystemRootResult, SystemRootSolver,
};

#[test]
fn difficult_forward_models_keep_the_mgh_successes() {
    let mut failures = Vec::new();
    for (id, n, factor, strategy) in [
        (8, 10, 100, DerivativeStrategy::Forward),
        (8, 40, 1, DerivativeStrategy::Analytic),
        (12, 10, 100, DerivativeStrategy::Forward),
    ] {
        let row: Vec<_> = include_str!("data/mgh.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').collect::<Vec<_>>())
            .find(|row| {
                row[0] == id.to_string() && row[1] == n.to_string() && row[2] == factor.to_string()
            })
            .unwrap();
        let problem = mgh::Mgh::new(id, n);
        let point: Vec<f64> = row[9].split(',').map(|x| x.parse().unwrap()).collect();
        let solver = Broyden::new(
            BroydenUpdate::First,
            BroydenOptions::default()
                .with_line_search(scioxide_rootfind::BacktrackingLineSearch::default())
                .with_derivative_strategy(strategy)
                .with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0).unwrap())
                .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap())
                .with_max_iterations(Budget::try_from(1000).unwrap()),
        )
        .unwrap();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut result = SystemRootResult::empty();
        result.try_reserve(n).unwrap();
        problem.reset_counts();
        let outcome = solver.solve_into(
            &problem,
            &point,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        );
        if outcome.is_err() || !result.is_success() {
            failures.push(format!(
                "{id}/{n}/{factor}/{strategy:?}: {outcome:?} {result:?} accepted={:?}",
                workspace.last_accepted()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
