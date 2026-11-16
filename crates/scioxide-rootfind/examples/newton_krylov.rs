//! Solve a sparse nonlinear system without assembling its Jacobian.
#[path = "support/newton_krylov.rs"]
mod support;
use scioxide_core::{ResidualTolerance, SciOxideError};
use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};
fn main() -> Result<(), SciOxideError> {
    let residual = support::Tridiagonal(128);
    let problem = NewtonKrylovProblem::with_jacobian(&residual).with_preconditioner(&residual);
    let options = NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-8, 0.)?)
        .with_history(true);
    let result = NewtonKrylovSolver::new(options)
        .and_then(|root_plan| root_plan.solve(&problem, &[0.4; 128]))?;
    assert!(result.is_success());
    println!(
        "{:?}: {} Newton steps; {} JVPs; residual {:?}",
        result.termination(),
        result.iterations(),
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::LinearOperator),
        result.residual_report()
    );
    Ok(())
}
