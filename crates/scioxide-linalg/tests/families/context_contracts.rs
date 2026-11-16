#![allow(clippy::expect_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, CancellationToken, EvaluationRole, ExecutionContext,
    PartialResultTrust, ProblemToken, ResidualTolerance, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgstabSolver, CgSolver, GmresSolver, IterativeSolveReport, IterativeWorkspace,
    LinearSolveTermination, Matrix, Vector,
};

fn budget(value: usize) -> Budget {
    Budget::try_from(value).expect("test budgets are positive")
}

fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(1.0e-12, 0.0).expect("test tolerance is valid")
}

fn parent() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(11),
        scioxide_core::algorithm_id!("tests", "context_parent", 1),
        0,
        0,
    )
}

/// Run one solver in a child epoch of a parent with a single operator evaluation.
fn evaluation_limited(
    solve: impl FnOnce(
        &mut Vector<f64>,
        &mut IterativeWorkspace<f64>,
        &mut scioxide_core::SolveContext<'_, '_>,
    ) -> IterativeSolveReport<f64>,
) -> (
    IterativeSolveReport<f64>,
    IterativeWorkspace<f64>,
    Vector<f64>,
) {
    let mut workspace = IterativeWorkspace::new();
    let mut output = Vector::from_vec(vec![7.0, 8.0]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(1));
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .expect("solve epoch starts");
    let report = solve(&mut output, &mut workspace, &mut context);
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    assert_eq!(context.counts().own().iterations(), 0);
    (report, workspace, output)
}

#[test]
fn iterative_solvers_return_accounted_partial_results_at_evaluation_limit() {
    let matrix =
        Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0]).expect("fixture dimensions match");
    let matrix = matrix.view().try_finite().expect("finite matrix fixture");
    let rhs = Vector::from_vec(vec![1.0, 2.0]);

    let cg = CgSolver::new(tolerance(), budget(8));
    let (cg_report, cg_workspace, cg_output) = evaluation_limited(|x, workspace, context| {
        cg.solve_into(
            &matrix,
            &rhs,
            Preconditioning::None,
            x.view_mut(),
            workspace,
            context,
        )
        .expect("evaluation exhaustion is a partial result")
    });
    assert_eq!(
        cg_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    let attempt = cg_workspace.last_attempt().expect("attempt is recorded");
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(
        attempt
            .counts()
            .own()
            .provider(EvaluationRole::Preconditioner),
        0
    );
    assert_eq!(cg_output.as_slice(), &[7.0, 8.0]);
    assert_eq!(
        attempt.progress().map(|progress| progress.trust()),
        Some(PartialResultTrust::Admissible)
    );

    let gmres = GmresSolver::new(tolerance(), budget(8)).with_restart(budget(2));
    let (gmres_report, gmres_workspace, gmres_output) =
        evaluation_limited(|x, workspace, context| {
            gmres
                .solve_into(
                    &matrix,
                    &rhs,
                    Preconditioning::None,
                    x.view_mut(),
                    workspace,
                    context,
                )
                .expect("evaluation exhaustion is a partial result")
        });
    assert_eq!(
        gmres_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    assert_eq!(
        gmres_workspace
            .last_attempt()
            .expect("attempt is recorded")
            .counts()
            .own()
            .iterations(),
        1
    );
    assert_eq!(gmres_output.as_slice(), &[7.0, 8.0]);

    let bicgstab = BicgstabSolver::new(tolerance(), budget(8));
    let (bicgstab_report, bicgstab_workspace, bicgstab_output) =
        evaluation_limited(|x, workspace, context| {
            bicgstab
                .solve_into(
                    &matrix,
                    &rhs,
                    Preconditioning::None,
                    x.view_mut(),
                    workspace,
                    context,
                )
                .expect("evaluation exhaustion is a partial result")
        });
    assert_eq!(
        bicgstab_report.termination(),
        LinearSolveTermination::EvaluationLimit
    );
    assert_eq!(
        bicgstab_workspace
            .last_attempt()
            .expect("attempt is recorded")
            .counts()
            .own()
            .iterations(),
        1
    );
    assert_eq!(bicgstab_output.as_slice(), &[7.0, 8.0]);
}

#[test]
fn cancelled_dense_operation_returns_typed_error_without_mutating_output() {
    let matrix =
        Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("fixture dimensions match");
    let input = Vector::from_vec(vec![5.0, 6.0]);
    let mut output = Vector::from_vec(vec![7.0, 8.0]);
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    let result = matrix.view().mul_vec_into(
        &input,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        output.view_mut(),
        &mut scioxide_linalg::LinalgWorkspace::new(),
        ExecutionContext::sequential().with_cancellation(&cancellation),
    );

    assert!(matches!(
        (result).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(output.as_slice(), &[7.0, 8.0]);
}

#[test]
fn preflight_rejection_leaves_output_and_parent_ledger_unchanged() {
    let matrix = Matrix::<f64>::identity(2).expect("fixture allocation succeeds");
    let rhs = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let solver = CgSolver::new(tolerance(), budget(4));
    let mut workspace = IterativeWorkspace::new();
    let mut output = Vector::from_vec(vec![7.0, 8.0]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .expect("solve epoch starts");

    let result = solver.solve_into(
        &matrix.view().try_finite().expect("finite matrix fixture"),
        &rhs,
        Preconditioning::None,
        output.view_mut(),
        &mut workspace,
        &mut context,
    );

    assert!(matches!(
        (result).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output.as_slice(), &[7.0, 8.0]);
    assert_eq!(context.counts(), scioxide_core::SolveCounts::new());
    let attempt = workspace.last_attempt().expect("attempt is recorded");
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
}
