//! Prepared real/complex collocation with independent exponential accuracy.
mod support {
    pub mod bvp_models;
}
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_ode::{BvpOptions, BvpPlan, BvpProblem, BvpSolution, BvpSystem};
fn main() -> Result<(), SciOxideError> {
    let model = support::bvp_models::Exponential { amplitude: 1f64 };
    let plan = BvpPlan::new(
        model.dimension(),
        model.parameter_dimension(),
        &[0., 0.5, 1.],
        BvpOptions::new()
            .with_max_nodes(128)
            .with_max_workspace_bytes(32 * 1024 * 1024),
    )?;
    let problem = BvpProblem::analytic(&model);
    let mut workspace = plan.workspace(&problem)?;
    let mut solution = BvpSolution::new();
    let guess = Matrix::from_vec(3, 2, vec![1.; 6])?;
    let report = plan.solve_into(
        &problem,
        guess.view(),
        &[],
        &mut solution,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    let mut y = [0.; 2];
    solution.evaluate_into(0.5, &mut y)?;
    println!(
        "status={:?}, nodes={}, y(0.5)={}, error={}, normalized residual={}",
        report.termination(),
        report.nodes(),
        y[0],
        (y[0] - 0.5f64.exp()).abs(),
        report.max_residual()
    );
    Ok(())
}
