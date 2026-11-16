//! Cross-representation operator, solve, and workspace contracts.

#![allow(clippy::unwrap_used)]

use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Budget, BudgetKind, ClosureOperator, ExecutionContext,
    LinearOperator, PartialResultTrust, ProblemToken, ResidualScalingRule, ResidualTolerance,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgstabSolver, CgSolver, GmresSolver, IterativeLinearSolver, IterativeWorkspace,
    IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
};
use scioxide_sparse::ExplicitZeroPolicy;

fn solver() -> CgSolver<f64> {
    CgSolver::new(
        ResidualTolerance::new(1.0e-12, 0.0).unwrap(),
        Budget::try_from(20).unwrap(),
    )
}

#[test]
fn dense_sparse_and_matrix_free_solves_share_true_residual_contract() {
    let dense = Matrix::from_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let closure = ClosureOperator::new(2, 2, |input: &[f64], output: &mut [f64]| {
        output[0] = 4.0 * input[0] + input[1];
        output[1] = input[0] + 3.0 * input[1];
        Ok(())
    });
    let right_hand_side = scioxide_linalg::Vector::from_vec(vec![1.0, 2.0]);

    let results = [
        &dense.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &sparse,
        &closure,
    ]
    .into_iter()
    .map(|operator| {
        solver()
            .solve_with_options(operator, &right_hand_side, Preconditioning::None, None)
            .unwrap()
    })
    .collect::<Vec<_>>();

    results.iter().for_each(|result| {
        assert!(result.is_success());
        assert_eq!(result.report().trust(), PartialResultTrust::ResidualChecked);
        assert!(
            result
                .report()
                .true_residual()
                .is_some_and(|residual| residual.raw_norm() <= 1.0e-12)
        );
        assert_eq!(
            result.report().true_residual().unwrap().scaling().rule(),
            ResidualScalingRule::Reference
        );
    });
    for result in &results[1..] {
        result
            .solution()
            .as_slice()
            .iter()
            .zip(results[0].solution().as_slice())
            .for_each(|(actual, expected)| assert!((actual - expected).abs() <= 1.0e-12));
    }
}

#[test]
fn nearly_singular_and_singular_representations_share_backward_error_outcomes() {
    let dense = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0e-10]).unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let closure = ClosureOperator::new(2, 2, |input: &[f64], output: &mut [f64]| {
        output[0] = input[0];
        output[1] = 1.0e-10 * input[1];
        Ok(())
    });
    let right_hand_side = scioxide_linalg::Vector::from_vec(vec![1.0, 1.0e-10]);
    let near_solver = CgSolver::new(
        ResidualTolerance::new(1.0e-14, 0.0).unwrap(),
        Budget::try_from(10).unwrap(),
    );

    [
        &dense.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &sparse,
        &closure,
    ]
    .into_iter()
    .for_each(|operator| {
        let result = near_solver
            .solve_with_options(operator, &right_hand_side, Preconditioning::None, None)
            .unwrap();
        assert!(result.is_success());
        let x_norm = result
            .solution()
            .as_slice()
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        let residual = result.report().true_residual().unwrap().raw_norm();
        let backward_error = residual / (x_norm + 1.0);
        assert!(backward_error <= 1.0e-14);
    });

    let singular = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 0.0]).unwrap();
    let singular_sparse = singular.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let singular_closure = ClosureOperator::new(2, 2, |input: &[f64], output: &mut [f64]| {
        output[0] = input[0];
        output[1] = 0.0;
        Ok(())
    });
    let inconsistent_rhs = scioxide_linalg::Vector::from_vec(vec![1.0, 1.0]);
    [
        &singular.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &singular_sparse,
        &singular_closure,
    ]
    .into_iter()
    .for_each(|operator| {
        let result = solver()
            .solve_with_options(operator, &inconsistent_rhs, Preconditioning::None, None)
            .unwrap();
        assert!(matches!(
            result.report().termination(),
            scioxide_linalg::LinearSolveTermination::Breakdown(_)
        ));
        assert!(result.report().true_residual().is_some());
    });
}

#[test]
fn evaluation_budget_returns_only_the_last_true_residual_checked_solution() {
    let matrix = Matrix::from_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let right_hand_side = scioxide_linalg::Vector::from_vec(vec![1.0, 2.0]);
    let initial = scioxide_linalg::Vector::zeros(2).unwrap();
    let tolerance = ResidualTolerance::new(1.0e-16, 0.0).unwrap();

    let cg = CgSolver::new(tolerance, Budget::try_from(20).unwrap());
    let mut cg_workspace = cg
        .workspace(
            &IterativeWorkspaceRequest::operator(&matrix.view().try_finite().unwrap()).unwrap(),
        )
        .unwrap();
    let mut cg_solution = initial.clone();
    let cg_report = {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("tests", "operator_parent", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        cg.solve_into(
            &matrix.view().try_finite().unwrap(),
            &right_hand_side,
            Preconditioning::None,
            cg_solution.view_mut(),
            &mut cg_workspace,
            &mut context,
        )
        .unwrap()
    };
    assert_eq!(
        cg_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    assert_eq!(cg_solution, initial);
    assert!(cg_report.true_residual().is_some());

    let gmres =
        GmresSolver::new(tolerance, Budget::try_from(20).unwrap()).with_restart(Budget::ONE);
    let mut gmres_workspace = gmres
        .workspace(
            &IterativeWorkspaceRequest::operator(&matrix.view().try_finite().unwrap()).unwrap(),
        )
        .unwrap();
    let mut gmres_solution = initial.clone();
    let gmres_report = {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(2),
            scioxide_core::algorithm_id!("tests", "operator_parent", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        gmres
            .solve_into(
                &matrix.view().try_finite().unwrap(),
                &right_hand_side,
                Preconditioning::None,
                gmres_solution.view_mut(),
                &mut gmres_workspace,
                &mut context,
            )
            .unwrap()
    };
    assert_eq!(
        gmres_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    assert_eq!(gmres_solution, initial);
    assert!(gmres_report.true_residual().is_some());

    let bicgstab = BicgstabSolver::new(tolerance, Budget::try_from(20).unwrap());
    let mut bicgstab_workspace = bicgstab
        .workspace(
            &IterativeWorkspaceRequest::operator(&matrix.view().try_finite().unwrap()).unwrap(),
        )
        .unwrap();
    let mut bicgstab_solution = initial.clone();
    let bicgstab_report = {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(3),
            scioxide_core::algorithm_id!("tests", "operator_parent", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        bicgstab
            .solve_into(
                &matrix.view().try_finite().unwrap(),
                &right_hand_side,
                Preconditioning::None,
                bicgstab_solution.view_mut(),
                &mut bicgstab_workspace,
                &mut context,
            )
            .unwrap()
    };
    assert_eq!(
        bicgstab_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    assert_eq!(bicgstab_solution, initial);
    assert!(bicgstab_report.true_residual().is_some());
}

#[test]
fn iterative_block_rhs_is_transactional_and_column_major() {
    let dense = Matrix::from_vec(2, 2, vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let dense = dense.view().try_finite().unwrap();
    let right_hand_sides = [6.0, 7.0, 7.0, -1.0];
    let mut output_values = [9.0; 4];
    let mut workspace = IterativeWorkspace::new();
    // A non-converging column fails the whole block and leaves every column unchanged.
    let starved = CgSolver::new(ResidualTolerance::new(1.0e-15, 0.0).unwrap(), Budget::ONE);
    assert!(
        IterativeLinearSolver::solve_block_into(
            &starved,
            &dense,
            BlockVectorView::new(&right_hand_sides, 2, 2).unwrap(),
            Preconditioning::None,
            BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap(),
            &mut workspace,
            ExecutionContext::sequential().into(),
        )
        .is_err()
    );
    assert_eq!(output_values, [9.0; 4]);

    let solver = solver();
    let request = IterativeWorkspaceRequest::operator(&dense)
        .unwrap()
        .with_right_hand_sides(2);
    let mut workspace = IterativeLinearSolver::workspace(&solver, &request).unwrap();
    output_values = [0.0; 4];
    IterativeLinearSolver::solve_block_into(
        &solver,
        &dense,
        BlockVectorView::new(&right_hand_sides, 2, 2).unwrap(),
        Preconditioning::None,
        BlockVectorViewMut::new(&mut output_values, 2, 2).unwrap(),
        &mut workspace,
        ExecutionContext::sequential().into(),
    )
    .unwrap();
    output_values
        .iter()
        .zip([1.0, 2.0, 2.0, -1.0])
        .for_each(|(actual, expected)| assert!((actual - expected).abs() <= 1.0e-12));
}
