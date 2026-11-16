//! Exact resource, representation, recovery, and transaction contracts.
#![allow(clippy::unwrap_used)]
use crate::support::providers::{Provider, parent};

use scioxide_core::{
    AdjointOperator, AttemptOutcome, Budget, BudgetKind, CancellationToken, EvaluationRole,
    ExecutionContext, Preconditioner, ResidualTolerance, SciOxideError, SolveBudgets, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgSolver, CgsSolver, GaussSeidelSolver, IterativeWorkspace, IterativeWorkspaceRequest,
    JacobiSolver, LinearSolveTermination, Matrix, QmrSolver, RelaxationWeight, RichardsonSolver,
    SorSolver, SorWeight, StationaryMatrix, TfqmrSolver, Vector,
};
use scioxide_sparse::ExplicitZeroPolicy;
use std::cell::Cell;
fn limit(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn tol() -> ResidualTolerance<f64> {
    ResidualTolerance::new(1e-12, 0.).unwrap()
}
macro_rules! contracts {
    ($name:ident,$make:expr) => {
        #[test]
        fn $name() {
            let a = Provider {
                rows: 3,
                diagonal: &[1., 2., 4.],
                calls: Cell::new(0),
                fail_at: usize::MAX,
                cancel: None,
            };
            let make = $make;
            let solver = make(limit(100)).with_history(true);
            let b = Vector::from_vec(vec![1.; 3]);
            let mut w = solver
                .workspace(&IterativeWorkspaceRequest::adjoint(&a).unwrap())
                .unwrap();
            for maximum in 1..25 {
                a.calls.set(0);
                let mut x = Vector::zeros(3).unwrap();
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit(maximum));
                let mut c = state
                    .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                    .unwrap();
                let r = solver
                    .solve_into(&a, &b, Preconditioning::None, x.view_mut(), &mut w, &mut c)
                    .unwrap();
                assert!(a.calls.get() <= maximum);
                assert_eq!(
                    a.calls.get(),
                    c.counts().nested().provider(EvaluationRole::LinearOperator)
                );
                let actual = x
                    .as_slice()
                    .iter()
                    .zip(a.diagonal)
                    .fold(0_f64, |acc, (&x, &d)| acc.hypot(1. - x * d));
                assert!((r.true_residual().unwrap().raw_norm() - actual).abs() < 1e-12);
                let accepted = w.last_accepted().unwrap();
                assert_eq!(accepted.solution(), x.as_slice());
                assert_eq!(
                    accepted.history().unwrap().last().copied(),
                    Some(r.true_residual().unwrap().raw_norm())
                );
                if r.is_success() {
                    assert!(actual < 2e-12);
                }
            }
            for maximum in 1..8 {
                let solver = make(limit(maximum));
                let mut x = Vector::zeros(3).unwrap();
                let r = solver
                    .solve_into(
                        &a,
                        &b,
                        Preconditioning::None,
                        x.view_mut(),
                        &mut w,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(r.iterations() <= maximum);
                if !r.is_success() {
                    assert_eq!(r.iterations(), maximum);
                    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Stopped);
                }
            }
            // The first solve reserves storage for this solver's own request.
            let mut x = Vector::zeros(3).unwrap();
            let _ = solver
                .solve_into(
                    &a,
                    &b,
                    Preconditioning::None,
                    x.view_mut(),
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            x.view_mut().fill(0.);
            {
                let _ = solver
                    .solve_into(
                        &a,
                        &b,
                        Preconditioning::None,
                        x.view_mut(),
                        &mut w,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            };
            for fail_at in 2..8 {
                let failing = Provider {
                    rows: 3,
                    diagonal: &[1., 2., 4.],
                    calls: Cell::new(0),
                    fail_at,
                    cancel: None,
                };
                let mut x = Vector::from_vec(vec![7.; 3]);
                assert!(matches!(
                    (solver.solve_into(
                        &failing,
                        &b,
                        Preconditioning::None,
                        x.view_mut(),
                        &mut w,
                        ExecutionContext::sequential(),
                    ))
                    .as_ref()
                    .err()
                    .unwrap()
                    .view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
                assert_eq!(x.as_slice(), &[7.; 3]);
                let attempt = w.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
                assert!(attempt.epoch().is_some());
                assert!(w.last_accepted().is_none());
            }
            let cancellation = CancellationToken::new();
            let cancelling = Provider {
                rows: 3,
                diagonal: &[1., 2., 4.],
                calls: Cell::new(0),
                fail_at: usize::MAX,
                cancel: Some(&cancellation),
            };
            let mut x = Vector::zeros(3).unwrap();
            let r = solver
                .solve_into(
                    &cancelling,
                    &b,
                    Preconditioning::None,
                    x.view_mut(),
                    &mut w,
                    ExecutionContext::sequential().with_cancellation(&cancellation),
                )
                .unwrap();
            assert!(matches!(
                r.termination(),
                LinearSolveTermination::ExecutionStopped(_)
            ));
            let residual = x
                .as_slice()
                .iter()
                .zip(a.diagonal)
                .fold(0_f64, |sum, (&x, &d)| sum.hypot(1. - x * d));
            assert!((r.true_residual().unwrap().raw_norm() - residual).abs() < 1e-12);
        }
    };
}
contracts!(bicg_resources, |limit| BicgSolver::new(tol(), limit));
contracts!(cgs_resources, |limit| CgsSolver::new(tol(), limit));
contracts!(qmr_resources, |limit| QmrSolver::new(tol(), limit));
contracts!(tfqmr_resources, |limit| TfqmrSolver::new(tol(), limit));
contracts!(richardson_resources, |limit| RichardsonSolver::new(
    tol(),
    limit,
    RelaxationWeight::new(0.2).unwrap()
));

macro_rules! forms {
    ($name:ident,$make:expr) => {
        #[test]
        fn $name() {
            let a =
                Matrix::<f64>::from_vec(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 4.]).unwrap();
            let csr = a.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
            let csc = csr.to_csc().unwrap();
            let view = a.view().transpose();
            let p = Provider {
                rows: 3,
                diagonal: &[1., 2., 4.],
                calls: Cell::new(0),
                fail_at: usize::MAX,
                cancel: None,
            };
            let b = Vector::from_vec(vec![1., 2., 3.]);
            let solver = $make;
            for a in [
                &a.view().try_finite().unwrap() as &dyn AdjointOperator<f64>,
                &csr,
                &csc,
                &view.try_finite().unwrap(),
                &p,
            ] {
                let report = solver
                    .solve_with_options(a, &b, Preconditioning::None, None)
                    .unwrap();
                assert!(report.is_success());
                for (&x, y) in report.solution().as_slice().iter().zip([1., 1., 0.75]) {
                    assert!((x - y).abs() < 1e-11);
                }
                #[cfg(feature = "std")]
                {
                    let pool = scioxide_core::ThreadExecutor::new(
                        core::num::NonZeroUsize::new(2).unwrap(),
                    );
                    let mut w = IterativeWorkspace::new();
                    let mut x = Vector::zeros(3).unwrap();
                    let parallel = solver
                        .solve_into(
                            a,
                            &b,
                            Preconditioning::None,
                            x.view_mut(),
                            &mut w,
                            ExecutionContext::new(&pool),
                        )
                        .unwrap();
                    assert!(parallel.is_success());
                    assert!(
                        x.view()
                            .try_sub(report.solution())
                            .unwrap()
                            .view()
                            .norm2_scoped(
                                &scioxide_linalg::LinalgExecutionOptions::new(),
                                scioxide_core::ExecutionContext::sequential()
                            )
                            .unwrap()
                            < 1e-11
                    );
                }
            }
        }
    };
}
forms!(bicg_forms, BicgSolver::new(tol(), limit(100)));
forms!(cgs_forms, CgsSolver::new(tol(), limit(100)));
forms!(qmr_forms, QmrSolver::new(tol(), limit(100)));
forms!(tfqmr_forms, TfqmrSolver::new(tol(), limit(100)));
forms!(
    richardson_forms,
    RichardsonSolver::new(tol(), limit(200), RelaxationWeight::new(0.2).unwrap())
);

macro_rules! boundaries {
    ($name:ident,$solver:ident) => {
        #[test]
        fn $name() {
            let zero = Matrix::<f64>::from_vec(2, 2, vec![0.; 4]).unwrap();
            let b = Vector::from_vec(vec![1.; 2]);
            let solver = $solver::new(tol(), limit(20))
                .with_breakdown_restarts(limit(3))
                .with_history(true);
            let r = solver
                .solve_with_options(
                    &zero.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            assert!(matches!(
                r.report().termination(),
                LinearSolveTermination::Breakdown(_)
            ));
            assert_eq!(r.report().iterations(), 2);
            assert_eq!(r.report().krylov().unwrap().restarts(), 1);
            assert_eq!(r.solution().as_slice(), &[0.; 2]);
            assert_eq!(r.history().unwrap().len(), 1);
            let empty = Matrix::<f64>::zeros(0, 0).unwrap();
            let b = Vector::zeros(0).unwrap();
            let r = solver
                .solve_with_options(
                    &empty.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            assert!(r.is_success());
            assert_eq!(r.report().iterations(), 0);
            let eye = Matrix::<f64>::identity(2).unwrap();
            let b = Vector::from_vec(vec![1., 2.]);
            let r = solver
                .solve_with_options(
                    &eye.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    Some(b.view()),
                )
                .unwrap();
            assert!(r.is_success());
            assert_eq!(r.report().iterations(), 0);
            let r = solver
                .solve_with_options(
                    &eye.view().try_finite().unwrap(),
                    &Vector::zeros(2).unwrap(),
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            assert!(r.is_success());
            assert!(
                solver
                    .solve_with_options(
                        &eye.view().try_finite().unwrap(),
                        &Vector::from_vec(vec![f64::NAN, 1.]),
                        Preconditioning::None,
                        None,
                    )
                    .is_err()
            );
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut c = state
                .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                .unwrap();
            let mut w = IterativeWorkspace::new();
            let mut x = b.clone();
            assert!(matches!(
                (solver.solve_into(
                    &eye.view().try_finite().unwrap(),
                    &Vector::from_vec(vec![1.; 3]),
                    Preconditioning::None,
                    x.view_mut(),
                    &mut w,
                    &mut c,
                ))
                .as_ref()
                .err()
                .unwrap()
                .view(),
                scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
            ));
            assert_eq!(x, b);
            assert_eq!(c.counts(), scioxide_core::SolveCounts::new());
            let attempt = w.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
            assert!(attempt.epoch().is_none());
        }
    };
}
boundaries!(bicg_boundaries, BicgSolver);
boundaries!(cgs_boundaries, CgsSolver);
boundaries!(qmr_boundaries, QmrSolver);
boundaries!(tfqmr_boundaries, TfqmrSolver);

struct Inverse;
impl Preconditioner<f64> for Inverse {
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
            *z = r / d;
        }
        Ok(())
    }
}
#[test]
fn right_preconditioning() {
    let a = Matrix::<f64>::from_vec(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 4.]).unwrap();
    let inverse =
        scioxide_linalg::MatrixView::from_slice(3, 3, &[1., 0., 0., 0., 0.5, 0., 0., 0., 0.25])
            .unwrap()
            .try_finite()
            .unwrap();
    let b = Vector::from_vec(vec![1.; 3]);
    for r in [
        BicgSolver::new(tol(), limit(30))
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::InverseAdjoint(&inverse),
                None,
            )
            .unwrap(),
        QmrSolver::new(tol(), limit(30))
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::InverseAdjoint(&inverse),
                None,
            )
            .unwrap(),
        CgsSolver::new(tol(), limit(30))
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&Inverse),
                None,
            )
            .unwrap(),
        TfqmrSolver::new(tol(), limit(30))
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&Inverse),
                None,
            )
            .unwrap(),
        RichardsonSolver::new(tol(), limit(30), RelaxationWeight::one())
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&Inverse),
                None,
            )
            .unwrap(),
    ] {
        assert!(r.is_success());
        assert_eq!(r.report().iterations(), 1);
    }
}

macro_rules! stationary_resources {
    ($name:ident,$make:expr) => {
        #[test]
        fn $name() {
            let a = Matrix::<f64>::from_vec(3, 3, vec![4., -1., 0., -1., 4., -1., 0., -1., 4.])
                .unwrap();
            let csr = a.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
            let csc = csr.to_csc().unwrap();
            let b = Vector::from_vec(vec![1.; 3]);
            let solver = $make.with_history(true);
            let csr_request =
                IterativeWorkspaceRequest::matrix(StationaryMatrix::Csr(&csr)).unwrap();
            let csc_request =
                IterativeWorkspaceRequest::matrix(StationaryMatrix::Csc(&csc)).unwrap();
            assert_eq!(
                solver
                    .workspace_requirements(&csr_request)
                    .unwrap()
                    .required_bytes(),
                solver
                    .workspace_requirements(&csc_request)
                    .unwrap()
                    .required_bytes()
            );
            for input in [
                StationaryMatrix::Dense(a.view().try_finite().unwrap()),
                StationaryMatrix::Csr(&csr),
                StationaryMatrix::Csc(&csc),
            ] {
                let mut w = solver
                    .workspace(&IterativeWorkspaceRequest::matrix(input).unwrap())
                    .unwrap();
                let mut x = Vector::zeros(3).unwrap();
                {
                    assert!(
                        solver
                            .solve_into(
                                input,
                                &b,
                                x.view_mut(),
                                &mut w,
                                scioxide_core::ExecutionContext::sequential(),
                            )
                            .unwrap()
                            .is_success()
                    );
                };
                for maximum in 1..10 {
                    x.view_mut().fill(0.);
                    let mut state = SolveState::new();
                    let mut budgets = SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Evaluation, limit(maximum));
                    let mut c = state
                        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                        .unwrap();
                    let r = solver
                        .solve_into(input, &b, x.view_mut(), &mut w, &mut c)
                        .unwrap();
                    assert_eq!(r.termination(), LinearSolveTermination::EvaluationLimit);
                    assert_eq!(
                        c.counts().nested().provider(EvaluationRole::Preconditioner),
                        0
                    );
                    assert_eq!(
                        c.counts().nested().provider(EvaluationRole::LinearOperator),
                        maximum
                    );
                    let residual = b
                        .view()
                        .try_sub(&a.view().mul_vec(&x).unwrap())
                        .unwrap()
                        .view()
                        .norm2_scoped(
                            &scioxide_linalg::LinalgExecutionOptions::new(),
                            scioxide_core::ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert!((r.true_residual().unwrap().raw_norm() - residual).abs() < 1e-12);
                }
                #[cfg(feature = "std")]
                {
                    let pool = scioxide_core::ThreadExecutor::new(
                        core::num::NonZeroUsize::new(2).unwrap(),
                    );
                    x.view_mut().fill(0.);
                    assert!(
                        solver
                            .solve_into(
                                input,
                                &b,
                                x.view_mut(),
                                &mut w,
                                ExecutionContext::new(&pool),
                            )
                            .unwrap()
                            .is_success()
                    );
                }
            }
        }
    };
}
stationary_resources!(jacobi_resources, JacobiSolver::new(tol(), limit(100)));
stationary_resources!(gs_resources, GaussSeidelSolver::new(tol(), limit(100)));
stationary_resources!(
    sor_resources,
    SorSolver::new(tol(), limit(100), SorWeight::new(1.2).unwrap())
);

#[test]
fn exhausted_initial_budget_and_nonfinite_inverse_are_transactional() {
    let a = Provider {
        rows: 3,
        diagonal: &[1., 2., 4.],
        calls: Cell::new(0),
        fail_at: usize::MAX,
        cancel: None,
    };
    let b = Vector::from_vec(vec![1.; 3]);
    let solver = CgsSolver::new(tol(), limit(20));
    let mut w = IterativeWorkspace::new();
    let mut x = Vector::from_vec(vec![7.; 3]);
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit(1));
    let mut state = SolveState::new();
    let mut c = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    assert!(
        c.try_consume(BudgetKind::Evaluation, 1)
            .unwrap()
            .is_granted()
    );
    assert!(matches!(
        (solver.solve_into(&a, &b, Preconditioning::None, x.view_mut(), &mut w, &mut c))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(a.calls.get(), 0);
    assert_eq!(x.as_slice(), &[7.; 3]);
    struct BadInverse;
    impl Preconditioner<f64> for BadInverse {
        fn dimension(&self) -> usize {
            3
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out.fill(f64::NAN);
            Ok(())
        }
    }
    assert!(matches!(
        (solver.solve_into(
            &a,
            &b,
            Preconditioning::Inverse(&BadInverse),
            x.view_mut(),
            &mut w,
            scioxide_core::ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(
        a.calls.get(),
        1,
        "bad inverse output must be rejected before reaching A"
    );
    assert_eq!(x.as_slice(), &[7.; 3]);
}

use crate::support::scoped_adjoint;
macro_rules! scoped_biorthogonal {
    ($test:ident, $solver:ident) => {
        #[test]
        fn $test() {
            use std::sync::atomic::Ordering;
            for inverse in [false, true] {
                let a = scoped_adjoint::ScopedDiagonal::new();
                let b = Vector::from_vec(vec![2.0, 3.0, 4.0]);
                let solver = $solver::new(tol(), limit(30));
                let inverse_provider = scoped_adjoint::ScopedDiagonal::new();
                let preconditioning = if inverse {
                    Preconditioning::InverseAdjoint(&inverse_provider)
                } else {
                    Preconditioning::None
                };
                let mut workspace = solver
                    .workspace(
                        &IterativeWorkspaceRequest::adjoint(&a)
                            .unwrap()
                            .with_preconditioner(preconditioning)
                            .unwrap(),
                    )
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
                        preconditioning,
                        output.view_mut(),
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                let context = workspace.last_attempt().unwrap();
                assert!(report.is_success());
                assert!(output.as_slice().iter().all(|v| (*v - 1.0).abs() < 1e-10));
                let calls = a.forward.load(Ordering::Relaxed) + a.adjoint.load(Ordering::Relaxed);
                assert!(a.adjoint.load(Ordering::Relaxed) > 0);
                assert_eq!(
                    context
                        .counts()
                        .own()
                        .provider(EvaluationRole::LinearOperator),
                    calls
                );
                let precondition = context
                    .counts()
                    .own()
                    .provider(EvaluationRole::Preconditioner);
                assert_eq!(precondition > 0, inverse);
                assert_eq!(
                    context.counts().nested().provider(EvaluationRole::Function),
                    calls + precondition
                );
                assert_eq!(
                    context.counts().nested().iterations(),
                    2 * (calls + precondition)
                );
            }
        }
    };
}
scoped_biorthogonal!(bicg_scopes_operator_and_inverse_adjoint_calls, BicgSolver);
scoped_biorthogonal!(qmr_scopes_operator_and_inverse_adjoint_calls, QmrSolver);

use crate::support::krylov_reference;

#[test]
fn long_recurrences_match_unrestarted_reference_counts() {
    let tolerance = ResidualTolerance::new(0., 1e-8).unwrap();
    let limit = Budget::try_from(20000).unwrap();
    for fixture in krylov_reference::fixtures() {
        let b = fixture.rhs();
        let result = match fixture.method {
            "bicg" => scioxide_linalg::BicgSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            "cgs" => scioxide_linalg::CgsSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            "qmr" => scioxide_linalg::QmrSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            "tfqmr" => scioxide_linalg::TfqmrSolver::new(tolerance, limit)
                .solve_with_options(&fixture, &b, Preconditioning::None, None)
                .unwrap(),
            _ => continue,
        };
        fixture.check(&result);
    }
}

#[test]
fn cgs_reference_counts_hold_under_portable_and_native_dispatch() {
    let solver = CgsSolver::new(ResidualTolerance::new(0., 1e-8).unwrap(), limit(20000));
    let fixture = krylov_reference::fixtures()
        .find(|f| f.problem == "poisson" && f.order == 16384 && f.method == "cgs")
        .unwrap();
    let b = fixture.rhs();
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&fixture).unwrap())
        .unwrap();
    for context in [
        ExecutionContext::sequential(),
        ExecutionContext::sequential_native(),
    ] {
        let mut x = Vector::zeros(b.len()).unwrap();
        let report = solver
            .solve_into(
                &fixture,
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                context,
            )
            .unwrap();
        fixture.check_report(x.as_slice(), report);
    }
}
