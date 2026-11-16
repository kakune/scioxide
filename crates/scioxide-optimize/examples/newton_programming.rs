//! Simplex basis resumes, prepared HSD restarts and an injected QP KKT provider.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    DenseKktProvider, InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem,
    ProgrammingResult, QpMethod, QpProblem, QpSolver, SimplexSolver,
};
fn main() -> Result<(), SciOxideError> {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(3, 2, vec![1., 1., 1., 0., 0., 1.])?,
        vec![4., 3., 2.],
        vec![LinearConstraintSense::LessEqual; 3],
    )?;
    let lp = LpProblem::new(vec![-1.0_f64, -2.])?.with_constraints(constraints)?;
    let simplex = SimplexSolver::default();
    let mut workspace = simplex.workspace(&lp)?;
    let mut first = ProgrammingResult::empty();
    simplex.solve_into(
        &lp,
        &mut first,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let basis = workspace.warm_start()?;
    let mut repeated = ProgrammingResult::empty();
    simplex.resume_into(
        &lp,
        &basis,
        &mut repeated,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!(
        "LP objective {}, repeated pivots {}, reused basis {}",
        first.fun(),
        repeated.iterations(),
        workspace.warm_start()?.reused_basis()
    );
    let hsd = InteriorPointLp::default();
    let mut workspace = hsd.workspace(&lp)?;
    let mut interior = ProgrammingResult::empty();
    hsd.solve_into(
        &lp,
        &mut interior,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!(
        "HSD {:?}, embedding {:?}",
        interior.termination(),
        workspace.embedding_scalars()
    );
    let qp = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.])?,
        vec![-2., -6.],
    )?;
    let solver = QpSolver::default().with_method(QpMethod::PrimalDual);
    let mut workspace = solver
        .workspace(&qp)?
        .with_kkt_provider(Box::new(DenseKktProvider::new(2)?));
    let mut quadratic = ProgrammingResult::empty();
    solver.solve_into(
        &qp,
        &mut quadratic,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!("QP {:?}: {:?}", quadratic.termination(), quadratic.x());
    Ok(())
}
