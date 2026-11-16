//! Explicit budgets, reusable workspaces, and caller-selected random streams.

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, ExecutionContext, Interval, NormSelection, ProblemToken,
    SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, MonteCarloIntegrator, MonteCarloOptions, RandomStreams,
    SubdivisionOptions,
};
use scioxide_random::{Pcg64, SeedableRng};

fn main() -> Result<(), SciOxideError> {
    // Budgets live in the options; the caller-owned workspace is reusable across calls.
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(
            1e-9_f64,
            1e-9,
            NormSelection::Infinity,
        )?)
        .with_max_evaluations(Some(Budget::try_from(1500)?));
    let solver = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options)?;
    let mut workspace = solver.workspace(1)?;
    let mut value = 0.0;
    let report = solver.integrate_into(
        |x: f64| Ok(x * x),
        Interval::new(0.0, 1.0)?,
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "quadrature: {value:?}, {:?}, {} calls",
        report.termination(),
        report.evaluations()
    );

    let monte_carlo = MonteCarloIntegrator::new(
        &[Interval::new(0.0, 1.0)?; 2],
        MonteCarloOptions::default().with_samples(Budget::try_from(1000)?),
    )?;
    let mut workspace = monte_carlo.workspace(1, ExecutionContext::sequential_native())?;
    let root = Pcg64::from_seed(42);
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Sample, Budget::try_from(600)?);
    let identity = SolveIdentity::new(
        ProblemToken::new(2),
        scioxide_core::algorithm_id!("integrate", "MonteCarlo", 1),
        2,
        1,
    );
    let mut state = SolveState::new();
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let mut value = 0.0;
    // The ancestor sample budget stops the single estimate early; it still
    // publishes the mean of the observed prefix with an unsuccessful termination.
    let report = monte_carlo.integrate_into(
        |x: &[f64]| Ok(x[0] + x[1]),
        RandomStreams::new(&root, 0),
        &mut value,
        &mut workspace,
        &mut context,
    )?;
    println!(
        "Monte Carlo: {value:?}, {:?}, {:?}",
        report.termination(),
        workspace.statistics()[0]
    );
    Ok(())
}
