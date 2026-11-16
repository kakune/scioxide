//! Terminal objective gradients using DP5 and an explicit memory/recompute budget.
#[allow(dead_code)]
#[path = "support/adjoint_models.rs"]
mod models;
use scioxide_core::SciOxideError;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::{AdjointMode, AdjointOptions, OdeAdjointSolver, OdeAdjointSystem, OdeSystem};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = models::Diagonal::<f64>::new(4, -0.7)?;
    let options = AdjointOptions::new(AdjointMode::Discrete, 0.125)?
        .with_tape_capacity(128)
        .with_max_replay_steps(20000)
        .with_max_reverse_steps(10000)
        .with_max_workspace_bytes(64 * 1024)
        .with_checkpoint_policy(CheckpointPolicy::MemoryBudget);
    let plan = OdeAdjointSolver::new(model.dimension(), model.parameter_dimension(), options)?;
    let result = plan
        .solve(&scioxide_ode::OdeAdjointProblem::new(
            &model,
            &models::Energy,
            &[1.; 4],
            (0., 0.5),
        ))
        .map_err(SciOxideError::from)?;
    let report = result.report();
    assert!(report.is_success());
    println!(
        "initial gradient: {:?}; parameter gradient: {:?}; recorded/replayed steps: {}/{}; reserved bound: {} bytes",
        result.initial_gradient(),
        result.parameter_gradient(),
        report.accepted_steps(),
        report.replayed_steps(),
        report.workspace_bytes()
    );
    Ok(())
}
