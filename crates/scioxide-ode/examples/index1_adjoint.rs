//! Constrained terminal-objective gradients through a regular index-1 residual.
#[allow(dead_code)]
#[path = "support/adjoint_models.rs"]
mod models;
use scioxide_ad::CheckpointSchedule;
use scioxide_core::SciOxideError;
use scioxide_ode::CheckpointPolicy;
use scioxide_ode::{
    AdjointMode, AdjointOptions, DaeAdjointSolver, DaeAdjointSystem, DaeNumericalOptions,
    DaePartition, OdeStageBackend,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = models::Pairs::<f64>::new(1, -0.7)?;
    let options = AdjointOptions::new(AdjointMode::Discrete, 0.0625)?
        .with_fixed_step(true)
        .with_tape_capacity(128)
        .with_max_replay_steps(20000)
        .with_max_reverse_steps(10000)
        .with_max_workspace_bytes(512 * 1024)
        .with_checkpoint_policy(CheckpointPolicy::Explicit(CheckpointSchedule::binomial(
            128, 2,
        )?));
    let plan = DaeAdjointSolver::new(
        DaePartition::new(&[true, false], &[true, false])?,
        model.parameter_dimension(),
        OdeStageBackend::Dense,
        scioxide_ode::DaeAdjointOptions::new(options, DaeNumericalOptions::new())?,
    )?;
    let result = plan
        .solve(&scioxide_ode::DaeAdjointProblem::new(
            &model,
            &models::Energy,
            &[1., 2.],
            &[-0.7, 0.],
            (0., 0.5),
        ))
        .map_err(SciOxideError::from)?;
    assert!(result.report().is_success());
    println!(
        "free initial gradient: {:?}; parameter gradient: {:?}; transpose defect: {:?}",
        result.initial_gradient(),
        result.parameter_gradient(),
        result.report().errors().linear_residual()
    );
    Ok(())
}
