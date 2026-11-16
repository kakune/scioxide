//! Reverify a solver result against its original mathematical problem.
use scioxide_core::SciOxideError;
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    LinearConstraintSense, LinearConstraints, LpProblem, LpVerificationProblem,
    ProgrammingVerifier, SimplexSolver,
};
fn main() -> Result<(), SciOxideError> {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(1, 2, vec![1., 1.])?,
        vec![3.],
        vec![LinearConstraintSense::LessEqual],
    )?;
    let problem = LpProblem::new(vec![-1f64, -2.])?.with_constraints(constraints)?;
    let result = SimplexSolver::default().solve(&problem)?;
    let report = ProgrammingVerifier::default()
        .verify(&LpVerificationProblem::from_problem(&problem), &result)?;
    println!(
        "status={:?}, primal={:?}, gap={:?}, dual bound={:?}",
        report.status(),
        report.primal_objective(),
        report.objective_gap(),
        report.dual_bound()
    );
    assert!(report.is_accepted());
    Ok(())
}
