//! BiCGSTAB shadow breakdown, recovery, and verified publication contracts.
#![allow(clippy::unwrap_used)]
use crate::support::{krylov_reference, providers::parent};

use scioxide_core::{
    Budget, BudgetKind, ExecutionContext, ExecutionStopReason, LinearOperator, ResidualTolerance,
    SciOxideError, SolveBudgets, SolveState,
};
use scioxide_linalg::{
    BicgstabSolver, IterativeSolveReport, IterativeWorkspaceRequest, LinearSolveBreakdown,
    LinearSolveTermination, Matrix, Preconditioning, Vector,
};
use std::cell::Cell;

fn limit(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn solver(iterations: usize) -> BicgstabSolver<f64> {
    BicgstabSolver::new(ResidualTolerance::new(0., 1e-8).unwrap(), limit(iterations))
        .with_history(true)
}

#[test]
fn matrix_free_poisson_128_converges_without_recovery() {
    let operator = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson" && f.order == 16384 && f.method == "bicgstab")
        .unwrap();
    let result = solver(4000)
        .solve_with_options(
            &operator,
            operator.rhs().view(),
            Preconditioning::None,
            None,
        )
        .unwrap();
    operator.check(&result);
    assert_eq!(result.report().krylov().unwrap().restarts(), 0);
}

#[test]
fn exact_shadow_breakdown_stops_after_one_retry_without_progress() {
    let matrix = Matrix::from_vec(2, 2, vec![0., -1., 1., 0.]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let rhs = Vector::from_vec(vec![1., 0.]);
    for recovery in [false, true] {
        let solver = if recovery {
            solver(20).with_breakdown_restarts(limit(3))
        } else {
            solver(20)
        };
        let result = solver
            .solve_with_options(&operator, &rhs, Preconditioning::None, None)
            .unwrap();
        assert_eq!(
            result.report().termination(),
            LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)
        );
        assert_eq!(result.solution().as_slice(), &[0., 0.]);
        assert_eq!(result.report().diagnostic_residual_norm(), Some(1.));
        assert_eq!(result.report().iterations(), if recovery { 2 } else { 1 });
        let diagnostics = result.report().krylov().unwrap();
        assert_eq!(diagnostics.restarts(), usize::from(recovery));
        assert!(diagnostics.rank_loss_detected());
        assert_eq!(result.history().unwrap(), &[1.]);
    }
}

#[test]
fn stabilization_breakdown_publishes_the_verified_half_step() {
    let matrix = Matrix::from_vec(2, 2, vec![1., 1., 1., 0.]).unwrap();
    let operator = matrix.view().try_finite().unwrap();
    let rhs = Vector::from_vec(vec![1., 0.]);
    for recovery in [false, true] {
        let solver = if recovery {
            solver(20).with_breakdown_restarts(limit(3))
        } else {
            solver(20)
        };
        let result = solver
            .solve_with_options(&operator, &rhs, Preconditioning::None, None)
            .unwrap();
        assert_eq!(
            result.report().termination(),
            LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)
        );
        assert_eq!(result.solution().as_slice(), &[1., 0.]);
        assert_eq!(result.report().diagnostic_residual_norm(), Some(1.));
        assert_eq!(
            result.report().krylov().unwrap().restarts(),
            usize::from(recovery)
        );
        assert_eq!(result.history().unwrap(), &[1., 1.]);
    }
}

// A = [[-1,0,-1],[0,-1,-1],[-1,-1,-1]], b = e₁. The second shadow
// divisor vanishes after one full step; a new shadow solves from that iterate.
struct Recoverable {
    calls: Cell<usize>,
    fail_at: usize,
    stop: bool,
}
impl Recoverable {
    fn new(fail_at: usize, stop: bool) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at,
            stop,
        }
    }
}
impl LinearOperator<f64> for Recoverable {
    fn rows(&self) -> usize {
        3
    }
    fn cols(&self) -> usize {
        3
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail_at && self.stop {
            return Err(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            ));
        }
        y.copy_from_slice(&[-x[0] - x[2], -x[1] - x[2], -x[0] - x[1] - x[2]]);
        if self.calls.get() == self.fail_at {
            y[0] = f64::NAN;
        }
        Ok(())
    }
}
fn check_residual(x: &Vector<f64>, report: IterativeSolveReport<f64>) {
    let [a, b, c] = [x.as_slice()[0], x.as_slice()[1], x.as_slice()[2]];
    let actual = (1. + a + c).hypot(b + c).hypot(a + b + c);
    assert!((report.diagnostic_residual_norm().unwrap() - actual).abs() < 1e-12);
}

#[test]
fn recovery_converges_and_reuses_reserved_storage() {
    let operator = Recoverable::new(usize::MAX, false);
    let rhs = Vector::from_vec(vec![1., 0., 0.]);
    let failed = solver(20)
        .solve_with_options(&operator, &rhs, Preconditioning::None, None)
        .unwrap();
    assert_eq!(
        failed.report().termination(),
        LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)
    );
    check_residual(failed.solution(), failed.report());
    let solver = solver(20).with_breakdown_restarts(limit(1));
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&operator).unwrap())
        .unwrap();
    let mut x = Vector::zeros(3).unwrap();
    for _ in 0..2 {
        x.as_mut_slice().fill(0.);
        let report = solver
            .solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), LinearSolveTermination::Converged);
        assert_eq!(report.krylov().unwrap().restarts(), 1);
        assert!(report.krylov().unwrap().rank_loss_detected());
        check_residual(&x, report);
        for (actual, expected) in x.as_slice().iter().zip([0., 1., -1.]) {
            assert!((actual - expected).abs() < 1e-8);
        }
        let accepted = workspace.last_accepted().unwrap();
        assert_eq!(accepted.progress().restarts(), 1);
        assert_eq!(
            accepted.history().unwrap().last().copied(),
            report.diagnostic_residual_norm()
        );
    }
}

#[test]
fn recovery_respects_iteration_and_evaluation_budgets() {
    let operator = Recoverable::new(usize::MAX, false);
    let rhs = Vector::from_vec(vec![1., 0., 0.]);
    let result = solver(2)
        .with_breakdown_restarts(limit(3))
        .solve_with_options(&operator, &rhs, Preconditioning::None, None)
        .unwrap();
    assert_eq!(
        result.report().termination(),
        LinearSolveTermination::IterationLimit
    );
    assert_eq!(result.report().iterations(), 2);
    check_residual(result.solution(), result.report());

    let solver = solver(20).with_breakdown_restarts(limit(3));
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&operator).unwrap())
        .unwrap();
    for count in 1..12 {
        operator.calls.set(0);
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit(count));
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
            .unwrap();
        let mut x = Vector::zeros(3).unwrap();
        let report = solver
            .solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert!(operator.calls.get() <= count);
        assert!(matches!(
            report.termination(),
            LinearSolveTermination::EvaluationLimit | LinearSolveTermination::Converged
        ));
        check_residual(&x, report);
        assert_eq!(
            workspace
                .last_accepted()
                .unwrap()
                .history()
                .unwrap()
                .last()
                .copied(),
            report.diagnostic_residual_norm()
        );
    }
}

#[test]
fn recovery_provider_faults_preserve_transactional_publication() {
    let rhs = Vector::from_vec(vec![1., 0., 0.]);
    let solver = solver(20).with_breakdown_restarts(limit(3));
    // Calls 5 and 6 verify the breakdown candidate and start the new recurrence.
    for call in [5, 6] {
        for stop in [false, true] {
            let operator = Recoverable::new(call, stop);
            let mut workspace = solver
                .workspace(&IterativeWorkspaceRequest::operator(&operator).unwrap())
                .unwrap();
            let mut x = Vector::zeros(3).unwrap();
            let result = solver.solve_into(
                &operator,
                &rhs,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            );
            if stop {
                let report = result.unwrap();
                assert_eq!(
                    report.termination(),
                    LinearSolveTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
                );
                check_residual(&x, report);
            } else {
                assert!(matches!(
                    result.unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
                assert_eq!(x.as_slice(), &[0., 0., 0.]);
                assert!(workspace.last_accepted().is_none());
            }
        }
    }
}
