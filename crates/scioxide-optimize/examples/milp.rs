//! Bounded branch-and-cut with workspace-owned relaxation solver and search state.
use scioxide_core::Bounds;
use scioxide_core::{
    Budget, BudgetKind, ExecutionContext, ProblemToken, SciOxideError, SolveBudgets, SolveIdentity,
    SolveState,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    BranchAndBound, LinearConstraintSense, LinearConstraints, LpProblem, MipProblem, MipResult,
};

fn main() -> Result<(), SciOxideError> {
    let rows = LinearConstraints::new(
        Matrix::from_vec(1, 2, vec![2.0, 2.0])?,
        vec![3.0],
        vec![LinearConstraintSense::LessEqual],
    )?;
    let lp = LpProblem::new(vec![-1.0_f64, -1.0])?
        .with_bounds(Bounds::new(vec![(Some(0.0), Some(1.0)); 2])?)?
        .with_constraints(rows)?;
    let problem = MipProblem::new(lp, vec![0, 1])?;
    let solver = BranchAndBound::default();
    let mut workspace = solver.workspace(&problem)?;
    let mut result = MipResult::empty();
    // A parent epoch shares its node budget with the child search.
    let mut work = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::try_from(32)?);
    let mut context = work.begin_solve(
        ExecutionContext::sequential(),
        SolveIdentity::new(
            ProblemToken::new(606),
            scioxide_core::algorithm_id!("example", "milp", 1),
            0,
            0,
        ),
        &mut budgets,
    )?;
    solver.solve_into(&problem, &mut result, &mut workspace, &mut context)?;
    println!(
        "termination={}, incumbent={}, objective={}, lower={:?}, gap={:?}, cuts={}, reused={}, open={}",
        result.termination(),
        result.has_incumbent(),
        result.fun(),
        result.global_lower_bound(),
        result.absolute_gap(),
        result.statistics().cuts(),
        result.statistics().basis_reuses(),
        workspace.state().open_nodes().len()
    );
    assert!(result.is_success());
    assert_eq!(result.fun(), -1.0);
    Ok(())
}
