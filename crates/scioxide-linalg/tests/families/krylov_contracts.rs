//! Resource, provider, and representation contracts for MINRES/LSQR/LSMR.
#![allow(clippy::unwrap_used)]
use crate::support::providers::{Provider, parent};

use scioxide_core::{
    AdjointOperator, AttemptOutcome, Budget, BudgetKind, CancellationToken, Capacity,
    EvaluationRole, ExecutionContext, GradientTolerance, LinearOperator, PartialResultTrust,
    Preconditioner, ResidualTolerance, SciOxideError, SolveBudgets, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, LsmrSolver, LsqrSolver,
    Matrix, MinresSolver, Vector,
};
use scioxide_sparse::ExplicitZeroPolicy;
use std::cell::Cell;
fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(1e-12, 0.).unwrap()
}
fn normal() -> GradientTolerance<f64> {
    GradientTolerance::new(1e-12, 0.).unwrap()
}
fn limit(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}

// Independent rectangular matrix-free diagonal operator, with nonzero scratch
// requirements and counters shared by forward and adjoint applications.
macro_rules! budget_contract {
    ($name:ident,$solver:ident $(, $inverse:expr)?) => {
        #[test]
        fn $name() {
            let a = Provider {
                rows: 3,
                diagonal: &[1., 2., 4.],
                calls: Cell::new(0),
                fail_at: usize::MAX,
                cancel: None,
            };
            let b = Vector::from_vec(vec![1., 1., 1.]);
            let solver = $solver::new(tolerance(), limit(20))
                .with_normal_tolerance(normal())
                .with_history(true);
            let mut w = IterativeWorkspace::new();
            for count in 1..14 {
                a.calls.set(0);
                let mut x = Vector::zeros(3).unwrap();
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit(count));
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                    .unwrap();
                let report = solver.solve_into(
                    &a,
                    &b,
                    $($inverse,)?
                    x.view_mut(),
                    &mut w,
                    &mut context,
                );
                assert!(a.calls.get() <= count);
                assert_eq!(
                    context
                        .counts()
                        .nested()
                        .provider(EvaluationRole::LinearOperator),
                    a.calls.get()
                );
                let report = report.unwrap();
                if count == 1 {
                    assert_eq!(
                        report.termination(),
                        LinearSolveTermination::EvaluationLimit
                    );
                    assert!(report.krylov().unwrap().normal_residual().is_none());
                    assert!(
                        (report.true_residual().unwrap().raw_norm() - 3_f64.sqrt()).abs() < 1e-12
                    );
                    assert_eq!(x.as_slice(), &[0.; 3]);
                    continue;
                }
                assert_eq!(
                    report.trust(),
                    if report.is_success() {
                        PartialResultTrust::ResidualChecked
                    } else {
                        PartialResultTrust::Admissible
                    }
                );
                let r: Vec<_> = x
                    .as_slice()
                    .iter()
                    .zip(a.diagonal)
                    .map(|(&x, &d)| 1. - d * x)
                    .collect();
                let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                let gn = r
                    .iter()
                    .zip(a.diagonal)
                    .map(|(&v, &d)| (d * v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((report.true_residual().unwrap().raw_norm() - rn).abs() < 1e-12);
                if let Some(normal) = report.krylov().unwrap().normal_residual() {
                    assert!((normal.raw_norm() - gn).abs() < 1e-12);
                } else {
                    assert_eq!(report.termination(), LinearSolveTermination::EvaluationLimit);
                }
                assert_eq!(
                    w.last_accepted()
                        .unwrap()
                        .history()
                        .unwrap()
                        .last()
                        .copied(),
                    Some(report.true_residual().unwrap().raw_norm())
                );
            }
            let mut x = Vector::zeros(3).unwrap();
            { let mut evaluate_case = || {
                solver
                    .solve_into(
                        &a,
                        &b,
                        $($inverse,)?
                        x.view_mut(),
                        &mut w,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .map(|r| r.is_success())
            }; evaluate_case() }
            .unwrap();
            assert!(
                solver
                    .solve_with_options(&a, &b, $($inverse,)? Some(x.view()))
                    .unwrap()
                    .is_success()
            );
            a.calls.set(0);
            let failure = Provider { fail_at: 5, ..a };
            let before = x.clone();
            assert!(
                solver
                    .solve_into(
                        &failure,
                        &b,
                        $($inverse,)?
                        x.view_mut(),
                        &mut w,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .is_ok()
            ); // exact initial solution needs two calls
            x.view_mut().fill(0.);
            let before_zero = x.clone();
            failure.calls.set(0);
            assert!(matches!(
                (solver.solve_into(
                    &failure,
                    &b,
                    $($inverse,)?
                    x.view_mut(),
                    &mut w,
                    scioxide_core::ExecutionContext::sequential(),
                ))
                .as_ref()
                .err()
                .unwrap()
                .view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
            assert_eq!(x.as_slice(), before_zero.as_slice());
            let _ = before;
        }
    };
}
budget_contract!(minres_budgets, MinresSolver, Preconditioning::None);
budget_contract!(lsqr_budgets, LsqrSolver);
budget_contract!(lsmr_budgets, LsmrSolver);

#[test]
fn representations_and_executor_agree() {
    let a = Matrix::from_vec(4, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 4., 0., 0., 0.]).unwrap();
    let csr = a.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let csc = csr.to_csc().unwrap();
    let view = a.view();
    let provider = Provider {
        rows: 4,
        diagonal: &[1., 2., 4.],
        calls: Cell::new(0),
        fail_at: usize::MAX,
        cancel: None,
    };
    let b = Vector::from_vec(vec![1., 2., 3., 4.]);
    for op in [
        &a.view().try_finite().unwrap() as &dyn AdjointOperator<f64>,
        &view.try_finite().unwrap(),
        &csr,
        &csc,
        &provider,
    ] {
        let lsqr = LsqrSolver::new(tolerance(), limit(30)).with_normal_tolerance(normal());
        let lsmr = LsmrSolver::new(tolerance(), limit(30)).with_normal_tolerance(normal());
        for result in [
            lsqr.solve_with_options(op, &b, None).unwrap(),
            lsmr.solve_with_options(op, &b, None).unwrap(),
        ] {
            assert_eq!(
                result.report().termination(),
                LinearSolveTermination::LeastSquaresConverged
            );
            for (&x, expected) in result.solution().as_slice().iter().zip([1., 1., 0.75]) {
                assert!((x - expected).abs() < 1e-11);
            }
        }
        #[cfg(feature = "std")]
        {
            let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::new(&pool), parent(), &mut budgets)
                .unwrap();
            let mut w = IterativeWorkspace::new();
            let mut x = Vector::zeros(3).unwrap();
            assert!(
                lsmr.solve_into(op, &b, x.view_mut(), &mut w, &mut context)
                    .unwrap()
                    .is_success()
            );
            assert!((x.as_slice()[2] - 0.75).abs() < 1e-11);
        }
    }
}

struct DiagonalPreconditioner(f64);
impl Preconditioner<f64> for DiagonalPreconditioner {
    fn dimension(&self) -> usize {
        3
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(1)
    }
    fn precondition_into(
        &self,
        r: &[f64],
        z: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(w.len(), 1);
        for (z, (&r, d)) in z.iter_mut().zip(r.iter().zip([1., 2., 4.])) {
            *z = self.0 * r / d;
        }
        Ok(())
    }
}
#[test]
fn minres_preconditioner_and_invalid_inputs() {
    let a = Matrix::from_vec(3, 3, vec![-1., 0., 0., 0., 2., 0., 0., 0., -4.]).unwrap();
    let b = Vector::from_vec(vec![1.; 3]);
    for sign in [1., -1., 0.] {
        let solver = MinresSolver::new(tolerance(), limit(30));
        let result = solver.solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::Inverse(&DiagonalPreconditioner(sign)),
            None,
        );
        if sign > 0. {
            assert!(result.unwrap().is_success());
        } else {
            assert!(matches!(
                result.unwrap().report().termination(),
                LinearSolveTermination::Breakdown(
                    scioxide_linalg::LinearSolveBreakdown::Preconditioner
                )
            ));
        }
    }
    let solver = LsqrSolver::new(tolerance(), limit(30));
    let mut w = IterativeWorkspace::new();
    let mut x = Vector::from_vec(vec![7.; 3]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    assert!(matches!(
        (solver.solve_into(
            &a.view().try_finite().unwrap(),
            &Vector::from_vec(vec![1.]),
            x.view_mut(),
            &mut w,
            &mut context,
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(x.as_slice(), &[7.; 3]);
    assert_eq!(context.counts(), scioxide_core::SolveCounts::new());
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let rectangular = Matrix::<f64>::zeros(2, 3).unwrap();
    assert!(
        MinresSolver::new(tolerance(), limit(30))
            .solve_with_options(
                &rectangular.view().try_finite().unwrap(),
                &Vector::zeros(2).unwrap(),
                Preconditioning::None,
                None,
            )
            .is_err()
    );
}

#[test]
fn cancellation_during_provider_restores_verified_solution() {
    let cancellation = CancellationToken::new();
    let a = Provider {
        rows: 3,
        diagonal: &[1., 2., 4.],
        calls: Cell::new(0),
        fail_at: usize::MAX,
        cancel: Some(&cancellation),
    };
    let b = Vector::from_vec(vec![1.; 3]);
    let solver = LsmrSolver::new(tolerance(), limit(30));
    let mut w = IterativeWorkspace::new();
    let mut x = Vector::zeros(3).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            parent(),
            &mut budgets,
        )
        .unwrap();
    let report = solver
        .solve_into(&a, &b, x.view_mut(), &mut w, &mut context)
        .unwrap();
    assert!(matches!(
        report.termination(),
        LinearSolveTermination::ExecutionStopped(_)
    ));
    assert_eq!(x.as_slice(), &[0.; 3]);
    assert!((report.true_residual().unwrap().raw_norm() - 3_f64.sqrt()).abs() < 1e-12);
}

#[test]
fn deadline_during_operator_returns_a_verified_partial_result() {
    use crate::support::clocks::TestClock;
    use scioxide_core::{Deadline, ExecutionStopReason, MonotonicInstant};
    use std::sync::atomic::{AtomicU64, Ordering};
    struct TimedOperator<'a> {
        inner: Provider<'a>,
        clock: &'a TestClock,
    }
    impl LinearOperator<f64> for TimedOperator<'_> {
        fn rows(&self) -> usize {
            3
        }
        fn cols(&self) -> usize {
            3
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(2)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
            self.inner.apply_into(x, y, w)?;
            if self.inner.calls.get() == 4 {
                self.clock.0.store(10, Ordering::Relaxed);
            }
            Ok(())
        }
    }
    let clock = TestClock(AtomicU64::new(0));
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let a = TimedOperator {
        inner: Provider {
            rows: 3,
            diagonal: &[1., 2., 4.],
            calls: Cell::new(0),
            fail_at: usize::MAX,
            cancel: None,
        },
        clock: &clock,
    };
    let solver = MinresSolver::new(tolerance(), limit(30));
    let b = Vector::from_vec(vec![1.; 3]);
    let mut x = Vector::zeros(3).unwrap();
    let mut w = IterativeWorkspace::new();
    let execution = ExecutionContext::sequential()
        .with_deadline(Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(std::time::Duration::from_secs(5)),
        ))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(execution, parent(), &mut budgets)
        .unwrap();
    let report = solver
        .solve_into(
            &a,
            &b,
            Preconditioning::None,
            x.view_mut(),
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        LinearSolveTermination::ExecutionStopped(ExecutionStopReason::DeadlineReached)
    );
    assert_eq!(x.as_slice(), &[0.; 3]);
}

#[test]
fn smaller_history_capacity_grows_before_providers() {
    let a = Provider {
        rows: 3,
        diagonal: &[1., 2., 4.],
        calls: Cell::new(0),
        fail_at: usize::MAX,
        cancel: None,
    };
    let request = IterativeWorkspaceRequest::adjoint(&a).unwrap();
    let small = LsqrSolver::new(tolerance(), limit(1)).with_history(true);
    let mut w = small.workspace(&request).unwrap();
    let solver = LsqrSolver::new(tolerance(), limit(30)).with_history(true);
    let requirements = solver.workspace_requirements(&request).unwrap();
    assert_eq!(w.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    w.try_reserve(&requirements).unwrap();
    assert_eq!(w.check(&requirements).unwrap(), Capacity::Sufficient);
    assert_eq!(requirements.history_capacity(), 31);
    assert_eq!(a.calls.get(), 0);
    let mut x = Vector::zeros(3).unwrap();
    let b = Vector::from_vec(vec![1.; 3]);
    let report = solver
        .solve_into(&a, &b, x.view_mut(), &mut w, ExecutionContext::sequential())
        .unwrap();
    assert!(report.is_success());
}

#[test]
fn initial_true_convergence_and_exhausted_budget_have_distinct_outcomes() {
    let a = Provider {
        rows: 3,
        diagonal: &[1., 1., 1.],
        calls: Cell::new(0),
        fail_at: usize::MAX,
        cancel: None,
    };
    let solver = LsqrSolver::new(tolerance(), limit(10));
    let b = Vector::from_vec(vec![1.; 3]);
    let mut w = IterativeWorkspace::new();
    for consumed in [false, true] {
        let mut x = b.clone();
        a.calls.set(0);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit(1));
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
            .unwrap();
        if consumed {
            assert!(
                context
                    .record_provider(EvaluationRole::Function)
                    .unwrap()
                    .is_granted()
            );
        }
        let result = solver.solve_into(&a, &b, x.view_mut(), &mut w, &mut context);
        if consumed {
            assert!(matches!(
                (result).as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
            assert_eq!(a.calls.get(), 0);
        } else {
            let report = result.unwrap();
            assert_eq!(report.termination(), LinearSolveTermination::Converged);
            assert_eq!(report.true_residual().unwrap().raw_norm(), 0.);
            assert!(report.krylov().unwrap().normal_residual().is_none());
            assert_eq!(a.calls.get(), 1);
        }
        assert_eq!(x.as_slice(), b.as_slice());
    }
}

use crate::support::scoped_adjoint;
macro_rules! scoped_adjoint_solver {
    ($test:ident, $solver:ident) => {
        #[test]
        fn $test() {
            use std::sync::atomic::Ordering;
            let a = scoped_adjoint::ScopedDiagonal::new();
            let b = Vector::from_vec(vec![2.0, 3.0, 4.0]);
            let solver = $solver::new(tolerance(), limit(20)).with_normal_tolerance(normal());
            let mut workspace = solver
                .workspace(&IterativeWorkspaceRequest::adjoint(&a).unwrap())
                .unwrap();
            let mut output = Vector::zeros(3).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                .unwrap();
            let report = solver
                .solve_into(&a, &b, output.view_mut(), &mut workspace, &mut context)
                .unwrap();
            assert!(report.is_success());
            assert!(output.as_slice().iter().all(|v| (*v - 1.0).abs() < 1e-10));
            let context = workspace.last_attempt().unwrap();
            let forward = a.forward.load(Ordering::Relaxed);
            let adjoint = a.adjoint.load(Ordering::Relaxed);
            assert!(forward > 0 && adjoint > 0);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::LinearOperator),
                forward + adjoint
            );
            assert_eq!(
                context.counts().nested().provider(EvaluationRole::Function),
                forward + adjoint
            );
            assert_eq!(
                context.counts().nested().iterations(),
                2 * (forward + adjoint)
            );
        }
    };
}
scoped_adjoint_solver!(lsqr_passes_scope_to_both_operator_orientations, LsqrSolver);
scoped_adjoint_solver!(lsmr_passes_scope_to_both_operator_orientations, LsmrSolver);

macro_rules! scoped_square_solver {
    ($test:ident, $solver:ident) => {
        #[test]
        fn $test() {
            use std::sync::atomic::Ordering;
            let a = scoped_adjoint::ScopedDiagonal::new();
            let b = Vector::from_vec(vec![2.0, 3.0, 4.0]);
            let solver = scioxide_linalg::$solver::new(tolerance(), limit(20));
            let mut workspace = solver
                .workspace(&IterativeWorkspaceRequest::operator(&a).unwrap())
                .unwrap();
            let mut output = Vector::zeros(3).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                .unwrap();
            let report = solver
                .solve_into(
                    &a,
                    &b,
                    Preconditioning::None,
                    output.view_mut(),
                    &mut workspace,
                    &mut context,
                )
                .unwrap();
            assert!(report.is_success());
            assert!(output.as_slice().iter().all(|v| (*v - 1.0).abs() < 1e-10));
            let context = workspace.last_attempt().unwrap();
            let calls = a.forward.load(Ordering::Relaxed);
            assert!(calls > 0);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::LinearOperator),
                calls
            );
            assert_eq!(
                context.counts().nested().provider(EvaluationRole::Function),
                calls
            );
            assert_eq!(context.counts().nested().iterations(), 2 * calls);
        }
    };
}
scoped_square_solver!(cg_scopes_recurrence_and_true_residual_calls, CgSolver);
scoped_square_solver!(
    bicgstab_scopes_recurrence_and_true_residual_calls,
    BicgstabSolver
);

use crate::support::krylov_reference;

#[test]
fn long_recurrences_match_unrestarted_reference_counts() {
    let tolerance = ResidualTolerance::new(0., 1e-8).unwrap();
    let limit = Budget::try_from(20000).unwrap();
    for fixture in krylov_reference::fixtures() {
        let b = fixture.rhs();
        let result = match fixture.method {
            "cg" => scioxide_linalg::CgSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            "bicgstab" => scioxide_linalg::BicgstabSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            "gmres" => MinresSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            _ => continue,
        };
        fixture.check(&result);
    }
}

use crate::support::iterative_events;
iterative_events::iterative_event_contract!(
    cg_execution_and_provider_contract,
    scioxide_linalg::CgSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    bicgstab_execution_and_provider_contract,
    scioxide_linalg::BicgstabSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    gmres_execution_and_provider_contract,
    scioxide_linalg::GmresSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    minres_execution_and_provider_contract,
    scioxide_linalg::MinresSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    lsqr_execution_and_provider_contract,
    scioxide_linalg::LsqrSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100))
);
iterative_events::iterative_event_contract!(
    lsmr_execution_and_provider_contract,
    scioxide_linalg::LsmrSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100))
);
iterative_events::iterative_event_contract!(
    bicg_execution_and_provider_contract,
    scioxide_linalg::BicgSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    cgs_execution_and_provider_contract,
    scioxide_linalg::CgsSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    qmr_execution_and_provider_contract,
    scioxide_linalg::QmrSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    tfqmr_execution_and_provider_contract,
    scioxide_linalg::TfqmrSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);

iterative_events::iterative_event_contract!(
    lgmres_execution_and_provider_contract,
    scioxide_linalg::LgmresSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    gcrot_execution_and_provider_contract,
    scioxide_linalg::GcrotSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100)),
    Preconditioning::None
);
iterative_events::iterative_event_contract!(
    richardson_execution_and_provider_contract,
    scioxide_linalg::RichardsonSolver::new(
        ResidualTolerance::new(0., 1e-10).unwrap(),
        limit(100),
        scioxide_linalg::RelaxationWeight::new(0.05).unwrap()
    ),
    Preconditioning::None
);

#[test]
fn cg_does_not_call_roundoff_level_residuals_converged() {
    let fixture = krylov_reference::fixtures()
        .find(|f| f.method == "cg")
        .unwrap();
    let b = fixture.rhs();
    let result =
        scioxide_linalg::CgSolver::new(ResidualTolerance::new(0., 1e-15).unwrap(), limit(512))
            .solve_with_options(&fixture, &b, Preconditioning::None, None)
            .unwrap();
    let mut ax = vec![0.; fixture.rows()];
    fixture
        .apply_into(result.solution().as_slice(), &mut ax, &mut [])
        .unwrap();
    let residual = ax.iter().fold(0_f64, |n, x| n.hypot(1. - x));
    if result.report().termination() == LinearSolveTermination::Converged {
        assert!(residual <= 1e-15 * 64.);
    } else {
        assert!(!result.is_success());
    }
}

#[test]
fn multiple_rhs_helper_preserves_provider_stop_and_atomic_output() {
    use scioxide_core::{
        BlockVectorView, BlockVectorViewMut, ExecutionStopReason, SciOxideErrorView,
    };
    use scioxide_linalg::IterativeLinearSolver;
    let provider = iterative_events::Provider {
        calls: Cell::new(0),
        at: 3,
        stop: true,
    };
    let b = [1.; 32];
    let mut x = [0.5; 32];
    let error =
        scioxide_linalg::CgSolver::new(ResidualTolerance::new(0., 1e-10).unwrap(), limit(100))
            .solve_block_into(
                &provider,
                BlockVectorView::new(&b, 16, 2).unwrap(),
                Preconditioning::None,
                BlockVectorViewMut::new(&mut x, 16, 2).unwrap(),
                &mut IterativeWorkspace::new(),
                ExecutionContext::sequential().into(),
            )
            .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::Cancelled
        }
    ));
    assert_eq!(x, [0.5; 32]);
}
