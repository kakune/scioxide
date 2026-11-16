//! Lifecycle of iterative solvers: epochs, preflight, stops, failures
//! and reservation.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;

use scioxide_core::{
    AdjointOperator, AttemptOutcome, BlockVectorView, BlockVectorViewMut, Budget, BudgetKind,
    Capacity, ExecutionContext, LinearOperator, ProblemToken, ResidualTolerance, SciOxideError,
    SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    CgSolver, GmresSolver, IterativeLinearSolver, IterativeSolveReport, IterativeWorkspace,
    IterativeWorkspaceRequest, LinalgExecutionContext, LinearSolveTermination, LsqrSolver,
    MinresSolver, RelaxationWeight, SorWeight, VectorView, VectorViewMut,
};

const N: usize = 8;

/// Symmetric positive definite `tridiag(-1, 4, -1)` that counts applications
/// and can fail at a chosen call.
struct Tridiagonal {
    n: usize,
    fail_at: Option<usize>,
    calls: Cell<usize>,
}
impl Tridiagonal {
    fn new(n: usize) -> Self {
        Self {
            n,
            fail_at: None,
            calls: Cell::new(0),
        }
    }
    fn failing_at(call: usize) -> Self {
        Self {
            fail_at: Some(call),
            ..Self::new(N)
        }
    }
}
impl LinearOperator<f64> for Tridiagonal {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "injected failure",
            ));
        }
        for (i, out) in y.iter_mut().enumerate() {
            let left = if i > 0 { x[i - 1] } else { 0. };
            let right = x.get(i + 1).copied().unwrap_or(0.);
            *out = 4. * x[i] - left - right;
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Tridiagonal {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}

fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(0., 1e-10).unwrap()
}
fn budget(limit: usize) -> Budget {
    Budget::try_from(limit).unwrap()
}
fn rhs() -> [f64; N] {
    core::array::from_fn(|i| (i + 1) as f64)
}
fn residual(b: &[f64], x: &[f64]) -> f64 {
    let mut ax = [0.; N];
    Tridiagonal::new(N).apply_into(x, &mut ax, &mut []).unwrap();
    b.iter()
        .zip(ax)
        .map(|(b, ax)| (b - ax).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(945),
        scioxide_core::algorithm_id!("tests", "iterative-lifecycle", 1),
        0,
        0,
    )
}

macro_rules! lifecycle {
    ($name:ident, $solver:ty, $request:ident, |$s:ident, $a:ident, $b:ident, $x:ident, $w:ident, $cx:ident| $call:expr) => {
        mod $name {
            use super::*;

            fn run<'s, 'c>(
                $s: &$solver,
                $a: &Tridiagonal,
                b: &[f64],
                x: &mut [f64],
                $w: &mut IterativeWorkspace<f64>,
                $cx: impl Into<LinalgExecutionContext<'s, 'c>>,
            ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
                let ($b, $x) = (VectorView::from_slice(b), VectorViewMut::from_slice(x));
                $call
            }
            fn reserved(solver: &$solver, a: &Tridiagonal) -> IterativeWorkspace<f64> {
                solver
                    .workspace(&IterativeWorkspaceRequest::$request(a).unwrap())
                    .unwrap()
            }

            #[test]
            fn root_and_child_epochs_share_the_parent_budget() {
                let a = Tridiagonal::new(N);
                let solver = <$solver>::new(tolerance(), budget(50));
                let mut w = reserved(&solver, &a);
                let b = rhs();
                let mut x = [0.; N];
                let report = run(
                    &solver,
                    &a,
                    &b,
                    &mut x,
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
                assert!(report.is_success(), "{report:?}");
                assert!(residual(&b, &x) < 1e-8);
                let root = w.last_attempt().unwrap();
                assert_eq!(root.outcome(), AttemptOutcome::Succeeded);
                let root_epoch = root.epoch().unwrap();
                let full = root.counts().own().total_providers().unwrap();
                let accepted = w.last_accepted().unwrap();
                assert_eq!(accepted.solution(), &x);
                assert_eq!(accepted.progress().termination(), report.termination());
                assert_eq!(accepted.progress().iterations(), report.iterations());

                // Two children of one parent: the second exhausts the shared evaluations.
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(full + 2));
                let mut parent = state
                    .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                    .unwrap();
                let mut first = [0.; N];
                let report = run(&solver, &a, &b, &mut first, &mut w, &mut parent).unwrap();
                assert!(report.is_success(), "{report:?}");
                let first_epoch = w.last_attempt().unwrap().epoch().unwrap();
                assert_ne!(first_epoch, root_epoch);
                let mut second = [0.; N];
                let stopped = run(&solver, &a, &b, &mut second, &mut w, &mut parent).unwrap();
                assert_eq!(
                    stopped.termination(),
                    LinearSolveTermination::EvaluationLimit
                );
                let attempt = w.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
                assert_ne!(attempt.epoch().unwrap(), first_epoch);
                assert_eq!(w.last_accepted().unwrap().solution(), &second);
                let counts = parent.counts();
                assert_eq!(counts.nested().total_providers().unwrap(), full + 2);
                assert_eq!(counts.own().total_providers().unwrap(), 0);
            }

            #[test]
            fn preflight_rejection_leaves_x_and_the_parent_ledger_unchanged() {
                let a = Tridiagonal::new(N);
                let solver = <$solver>::new(tolerance(), budget(50));
                let mut w = reserved(&solver, &a);
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(100));
                let mut parent = state
                    .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                    .unwrap();
                let before = parent.counts();
                let mut x = [7.; N];
                let error =
                    run(&solver, &a, &rhs()[..N - 1], &mut x, &mut w, &mut parent).unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
                assert_eq!(x, [7.; N]);
                assert_eq!(parent.counts(), before);
                assert_eq!(a.calls.get(), 0);
                let attempt = w.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
                assert_eq!(attempt.epoch(), None);
                assert!(w.last_accepted().is_none());
            }

            #[test]
            fn iteration_limit_returns_the_accepted_iterate() {
                let a = Tridiagonal::new(N);
                let solver = <$solver>::new(tolerance(), budget(1));
                let mut w = reserved(&solver, &a);
                let b = rhs();
                let mut x = [0.; N];
                let report = run(
                    &solver,
                    &a,
                    &b,
                    &mut x,
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
                assert_eq!(report.termination(), LinearSolveTermination::IterationLimit);
                assert!(!report.is_success());
                assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Stopped);
                let accepted = w.last_accepted().unwrap();
                assert_eq!(accepted.solution(), &x);
                assert_eq!(accepted.progress().iterations(), report.iterations());
                assert!(residual(&b, &x) < residual(&b, &[0.; N]));
            }

            #[test]
            fn provider_failure_fails_the_attempt_and_keeps_x() {
                let a = Tridiagonal::failing_at(3);
                let solver = <$solver>::new(tolerance(), budget(50));
                let mut w = reserved(&solver, &a);
                let mut x = [7.; N];
                assert!(
                    run(
                        &solver,
                        &a,
                        &rhs(),
                        &mut x,
                        &mut w,
                        ExecutionContext::sequential()
                    )
                    .is_err()
                );
                assert_eq!(x, [7.; N]);
                let attempt = w.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
                assert!(attempt.epoch().is_some());
                assert!(w.last_accepted().is_none());
            }

            #[test]
            fn reserved_workspace_solves_without_allocation() {
                let a = Tridiagonal::new(N);
                let solver = <$solver>::new(tolerance(), budget(50)).with_history(true);
                let request = IterativeWorkspaceRequest::$request(&a).unwrap();
                let requirements = solver.workspace_requirements(&request).unwrap();
                let mut w = IterativeWorkspace::new();
                assert_eq!(w.check(&requirements).unwrap(), Capacity::NeedsGrowth);
                w.try_reserve(&requirements).unwrap();
                assert_eq!(w.check(&requirements).unwrap(), Capacity::Sufficient);
                let b = rhs();
                let mut x = [0.; N];
                let report = run(
                    &solver,
                    &a,
                    &b,
                    &mut x,
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
                assert!(report.is_success(), "{report:?}");
                assert!(!w.last_accepted().unwrap().history().unwrap().is_empty());
                let larger = IterativeWorkspaceRequest::$request(&Tridiagonal::new(N + 1)).unwrap();
                assert_eq!(
                    w.check(&solver.workspace_requirements(&larger).unwrap())
                        .unwrap(),
                    Capacity::NeedsGrowth
                );
            }
        }
    };
}

lifecycle!(cg, CgSolver<f64>, operator, |s, a, b, x, w, cx| s
    .solve_into(a, b, Preconditioning::None, x, w, cx));

lifecycle!(minres, MinresSolver<f64>, operator, |s, a, b, x, w, cx| s
    .solve_into(a, b, Preconditioning::None, x, w, cx));
lifecycle!(lsqr, LsqrSolver<f64>, adjoint, |s, a, b, x, w, cx| s
    .solve_into(a, b, x, w, cx));
lifecycle!(gmres, GmresSolver<f64>, operator, |s, a, b, x, w, cx| s
    .solve_into(a, b, Preconditioning::None, x, w, cx));

#[test]
fn block_solves_run_without_allocation_after_reservation() {
    let a = Tridiagonal::new(N);
    let solvers: [Box<dyn IterativeLinearSolver<f64>>; 3] = [
        Box::new(CgSolver::new(tolerance(), budget(50))),
        Box::new(GmresSolver::new(tolerance(), budget(50))),
        Box::new(MinresSolver::new(tolerance(), budget(50))),
    ];
    let b: Vec<f64> = rhs().into_iter().chain(rhs().map(|v| -v)).collect();
    for solver in &solvers {
        let request = IterativeWorkspaceRequest::operator(&a)
            .unwrap()
            .with_right_hand_sides(2);
        let mut w = solver.workspace(&request).unwrap();
        let mut x = vec![0.; 2 * N];
        {
            solver
                .solve_block_into(
                    &a,
                    BlockVectorView::new(&b, N, 2).unwrap(),
                    Preconditioning::None,
                    BlockVectorViewMut::new(&mut x, N, 2).unwrap(),
                    &mut w,
                    ExecutionContext::sequential().into(),
                )
                .unwrap();
        };
        for (b, x) in b.chunks_exact(N).zip(x.chunks_exact(N)) {
            assert!(residual(b, x) < 1e-8);
        }
    }
}

#[test]
fn relaxation_weights_are_validated_values() {
    let kind = |error: SciOxideError| error.kind();
    for (weight, expected) in [
        (0., SciOxideErrorKind::InvalidInput),
        (-1., SciOxideErrorKind::InvalidInput),
        (f64::NAN, SciOxideErrorKind::NonFinite),
        (f64::INFINITY, SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(kind(RelaxationWeight::new(weight).unwrap_err()), expected);
    }
    assert_eq!(RelaxationWeight::new(0.5).unwrap().get(), 0.5);
    assert_eq!(RelaxationWeight::<f64>::one().get(), 1.);
    for (weight, expected) in [
        (0., SciOxideErrorKind::InvalidInput),
        (2., SciOxideErrorKind::InvalidInput),
        (f64::NAN, SciOxideErrorKind::NonFinite),
        (f64::NEG_INFINITY, SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(kind(SorWeight::new(weight).unwrap_err()), expected);
    }
    let error = SorWeight::new(2.).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::OutOfRange,
            ..
        }
    ));
    assert_eq!(SorWeight::new(1.5).unwrap().get(), 1.5);
}

/// `tridiag(-1, 4, -1)` whose `fail_at`-th application reports the provider's
/// own `NumericalInstability(Breakdown)`.
struct BreakingDown {
    fail_at: usize,
    calls: Cell<usize>,
}
impl LinearOperator<f64> for BreakingDown {
    fn rows(&self) -> usize {
        N
    }
    fn cols(&self) -> usize {
        N
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "provider breakdown",
            ));
        }
        Tridiagonal::new(N).apply_into(x, y, &mut [])
    }
}
impl AdjointOperator<f64> for BreakingDown {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}

type Outcome = (
    Result<IterativeSolveReport<f64>, SciOxideError>,
    [f64; N],
    AttemptOutcome,
    usize,
);
type Case = fn(usize) -> Outcome;

macro_rules! breaking {
    ($solver:expr, $request:ident, |$s:ident, $a:ident, $b:ident, $x:ident, $w:ident| $call:expr) => {
        |fail_at: usize| -> Outcome {
            let $a = BreakingDown {
                fail_at,
                calls: Cell::new(0),
            };
            let $s = $solver;
            let mut $w = $s
                .workspace(&IterativeWorkspaceRequest::$request(&$a).unwrap())
                .unwrap();
            let b = rhs();
            let mut x = [7.; N];
            let result = {
                let ($b, $x) = (
                    VectorView::from_slice(&b),
                    VectorViewMut::from_slice(&mut x),
                );
                $call
            };
            let outcome = $w.last_attempt().unwrap().outcome();
            (result, x, outcome, $a.calls.get())
        }
    };
}

#[test]
fn provider_breakdown_is_a_provider_failure_not_recurrence_breakdown() {
    use scioxide_linalg::{
        BicgSolver, BicgstabSolver, CgsSolver, LsmrSolver, QmrSolver, RichardsonSolver, TfqmrSolver,
    };
    let cases: [(&str, Case); 15] = [
        (
            "cg",
            breaking!(
                CgSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "minres",
            breaking!(
                MinresSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "gmres",
            breaking!(
                GmresSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "lsqr",
            breaking!(
                LsqrSolver::new(tolerance(), budget(50)),
                adjoint,
                |s, a, b, x, w| s.solve_into(&a, b, x, &mut w, ExecutionContext::sequential())
            ),
        ),
        (
            "lsmr",
            breaking!(
                LsmrSolver::new(tolerance(), budget(50)),
                adjoint,
                |s, a, b, x, w| s.solve_into(&a, b, x, &mut w, ExecutionContext::sequential())
            ),
        ),
        (
            "bicgstab",
            breaking!(
                BicgstabSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "bicgstab restarts",
            breaking!(
                BicgstabSolver::new(tolerance(), budget(50)).with_breakdown_restarts(budget(5)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "bicg",
            breaking!(
                BicgSolver::new(tolerance(), budget(50)),
                adjoint,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "bicg restarts",
            breaking!(
                BicgSolver::new(tolerance(), budget(50)).with_breakdown_restarts(budget(5)),
                adjoint,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "cgs",
            breaking!(
                CgsSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "cgs restarts",
            breaking!(
                CgsSolver::new(tolerance(), budget(50)).with_breakdown_restarts(budget(5)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "qmr",
            breaking!(
                QmrSolver::new(tolerance(), budget(50)),
                adjoint,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "tfqmr",
            breaking!(
                TfqmrSolver::new(tolerance(), budget(50)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "tfqmr restarts",
            breaking!(
                TfqmrSolver::new(tolerance(), budget(50)).with_breakdown_restarts(budget(5)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "richardson",
            breaking!(
                RichardsonSolver::new(
                    tolerance(),
                    budget(100),
                    RelaxationWeight::new(0.25).unwrap()
                ),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
    ];
    for (name, run) in cases {
        for fail_at in 1..=6 {
            let (result, x, outcome, calls) = run(fail_at);
            assert!(calls >= fail_at, "{name}: converged before call {fail_at}");
            let error = result.expect_err(name);
            assert!(
                matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Breakdown,
                        ..
                    }
                ),
                "{name} at call {fail_at}: {error:?}"
            );
            assert_eq!(outcome, AttemptOutcome::Failed, "{name} at call {fail_at}");
            assert_eq!(x, [7.; N], "{name} at call {fail_at}");
            assert_eq!(calls, fail_at, "{name}: no application after the failure");
        }
    }
}

macro_rules! zero_rhs {
    ($solver:expr, $request:ident, |$s:ident, $a:ident, $b:ident, $x:ident, $w:ident| $call:expr) => {
        || -> (IterativeSolveReport<f64>, Vec<f64>) {
            let $a = Tridiagonal::new(30);
            let $s = $solver;
            let mut $w = $s
                .workspace(&IterativeWorkspaceRequest::$request(&$a).unwrap())
                .unwrap();
            let b = [0.; 30];
            let mut x = vec![1.; 30];
            let report = {
                let ($b, $x) = (
                    VectorView::from_slice(&b),
                    VectorViewMut::from_slice(&mut x),
                );
                $call.unwrap()
            };
            (report, x)
        }
    };
}

type ZeroCase = fn() -> (IterativeSolveReport<f64>, Vec<f64>);

fn zero_tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(0., 1e-8).unwrap()
}

#[test]
fn zero_right_hand_sides_converge_relative_to_one_in_every_family() {
    use scioxide_linalg::{
        BicgSolver, BicgstabSolver, CgsSolver, LsmrSolver, QmrSolver, RichardsonSolver, TfqmrSolver,
    };
    let cases: [(&str, ZeroCase); 11] = [
        (
            "cg",
            zero_rhs!(
                CgSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "minres",
            zero_rhs!(
                MinresSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "gmres",
            zero_rhs!(
                GmresSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "lsqr",
            zero_rhs!(
                LsqrSolver::new(zero_tolerance(), budget(200)),
                adjoint,
                |s, a, b, x, w| s.solve_into(&a, b, x, &mut w, ExecutionContext::sequential())
            ),
        ),
        (
            "lsmr",
            zero_rhs!(
                LsmrSolver::new(zero_tolerance(), budget(200)),
                adjoint,
                |s, a, b, x, w| s.solve_into(&a, b, x, &mut w, ExecutionContext::sequential())
            ),
        ),
        (
            "bicgstab",
            zero_rhs!(
                BicgstabSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "bicg",
            zero_rhs!(
                BicgSolver::new(zero_tolerance(), budget(200)),
                adjoint,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "cgs",
            zero_rhs!(
                CgsSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "qmr",
            zero_rhs!(
                QmrSolver::new(zero_tolerance(), budget(200)),
                adjoint,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "tfqmr",
            zero_rhs!(
                TfqmrSolver::new(zero_tolerance(), budget(200)),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
        (
            "richardson",
            zero_rhs!(
                RichardsonSolver::new(
                    zero_tolerance(),
                    budget(200),
                    RelaxationWeight::new(0.25).unwrap()
                ),
                operator,
                |s, a, b, x, w| s.solve_into(
                    &a,
                    b,
                    Preconditioning::None,
                    x,
                    &mut w,
                    ExecutionContext::sequential()
                )
            ),
        ),
    ];
    for (name, run) in cases {
        let (report, x) = run();
        assert_eq!(
            report.termination(),
            LinearSolveTermination::Converged,
            "{name}: {report:?}"
        );
        let residual = report.true_residual().unwrap();
        // The reference of a zero right-hand side is one.
        assert_eq!(residual.scaling().denominator(), 1., "{name}");
        assert!(residual.scaled_norm() <= 1e-8, "{name}: {residual:?}");
        let mut ax = [0.; 30];
        Tridiagonal::new(30)
            .apply_into(&x, &mut ax, &mut [])
            .unwrap();
        let actual = ax.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(actual <= 1e-8, "{name}: {actual}");
    }
}

#[test]
fn block_preflight_failures_replace_the_previous_record_and_name_the_argument() {
    let a = Tridiagonal::new(N);
    let solver = CgSolver::new(tolerance(), budget(50));
    let request = IterativeWorkspaceRequest::operator(&a)
        .unwrap()
        .with_right_hand_sides(2);
    let mut w = solver.workspace(&request).unwrap();
    let b: Vec<f64> = rhs().into_iter().chain(rhs().map(|v| -v)).collect();
    let mut x = vec![0.; 2 * N];
    fn solve(
        solver: &CgSolver<f64>,
        b: BlockVectorView<'_, f64>,
        x: &mut [f64],
        w: &mut IterativeWorkspace<f64>,
    ) -> Result<(), SciOxideError> {
        solver.solve_block_into(
            &Tridiagonal::new(N),
            b,
            Preconditioning::None,
            BlockVectorViewMut::new(x, N, 2).unwrap(),
            w,
            ExecutionContext::sequential().into(),
        )
    }
    let succeeded = |w: &IterativeWorkspace<f64>| {
        assert_eq!(
            w.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
        assert!(w.last_accepted().is_some());
    };
    let rejected = |w: &IterativeWorkspace<f64>| {
        let attempt = w.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert_eq!(attempt.epoch(), None);
        assert!(w.last_accepted().is_none());
    };
    solve(
        &solver,
        BlockVectorView::new(&b, N, 2).unwrap(),
        &mut x,
        &mut w,
    )
    .unwrap();
    succeeded(&w);
    // A byte limit below the requirement fails before any column starts.
    let limited = CgSolver::new(tolerance(), budget(50))
        .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(1));
    let mut y = vec![7.; 2 * N];
    let error = solve(
        &limited,
        BlockVectorView::new(&b, N, 2).unwrap(),
        &mut y,
        &mut w,
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(y, [7.; 2 * N]);
    rejected(&w);
    // A short right-hand side is reported with its own dimensions.
    solve(
        &solver,
        BlockVectorView::new(&b, N, 2).unwrap(),
        &mut x,
        &mut w,
    )
    .unwrap();
    succeeded(&w);
    let short = BlockVectorView::new(&b[..2 * (N - 1)], N - 1, 2).unwrap();
    let error = solve(&solver, short, &mut y, &mut w).unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { expected, actual }
                if expected == [N, 2] && actual == [N - 1, 2]
        ),
        "{error:?}"
    );
    assert_eq!(y, [7.; 2 * N]);
    rejected(&w);
}
