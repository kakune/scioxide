//! Resource, epoch, representation, and transaction contracts for Arnoldi solvers.
#![allow(clippy::unwrap_used)]
use crate::support::providers::{FlexibleProvider as Provider, parent};
use core::cell::Cell;

use scioxide_core::ProblemToken;
use scioxide_core::{
    BlockVectorView, BlockVectorViewMut, Budget, BudgetKind, CancellationToken, EvaluationRole,
    ExecutionContext, LinearOperator, RankThreshold, ResidualTolerance, SciOxideError,
    SolveBudgets, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BlockGcrotSolver, BlockGmresSolver, BlockLgmresSolver, GcrotSolver, GmresSolver,
    IterativeWorkspace, IterativeWorkspaceRequest, KrylovRecycleSpace, LgmresSolver,
    LinearSolveTermination, Matrix, Vector,
};
use scioxide_sparse::ExplicitZeroPolicy;
fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(1e-11, 0.).unwrap()
}
fn provider() -> Provider<'static> {
    Provider {
        calls: Cell::new(0),
        fail: usize::MAX,
        cancel: None,
        scale: 1.,
    }
}
fn residual(x: &[f64], b: &[f64], scale: f64) -> f64 {
    x.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (&x, &b))| (b - scale * (i + 1) as f64 * x).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn every_scalar_budget_boundary_returns_verified_output_with_reused_storage() {
    let a = provider();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    macro_rules! check {
        ($solver:ident) => {{
            let solver = $solver::new(tolerance(), budget(12))
                .with_restart(budget(3))
                .with_history(true);
            let mut workspace = solver
                .workspace(&IterativeWorkspaceRequest::operator(&a).unwrap())
                .unwrap();
            for count in 1..10 {
                a.calls.set(0);
                let mut x = Vector::zeros(3).unwrap();
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(count));
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                    .unwrap();
                let report = solver
                    .solve_into(
                        &a,
                        &b,
                        Preconditioning::None,
                        x.view_mut(),
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                assert!(a.calls.get() <= count);
                assert_eq!(
                    context
                        .counts()
                        .nested()
                        .provider(EvaluationRole::LinearOperator),
                    a.calls.get()
                );
                assert_eq!(
                    context
                        .counts()
                        .nested()
                        .provider(EvaluationRole::Preconditioner),
                    0
                );
                let actual = residual(x.as_slice(), b.as_slice(), 1.);
                assert!((report.true_residual().unwrap().raw_norm() - actual).abs() < 1e-12);
                assert_eq!(
                    workspace
                        .last_accepted()
                        .unwrap()
                        .history()
                        .unwrap()
                        .last()
                        .copied(),
                    Some(report.true_residual().unwrap().raw_norm())
                );
            }
        }};
    }
    check!(GmresSolver);
    check!(LgmresSolver);
    check!(GcrotSolver);
}

#[test]
fn block_budget_interruptions_commit_only_complete_verified_blocks() {
    let a = provider();
    let b = [1., 1., 1., 2., 0., 3.];
    macro_rules! check {
        ($solver:ident) => {{
            let solver = $solver::new(tolerance(), budget(8)).with_history(true);
            let mut workspace = IterativeWorkspace::new();
            for count in 1..15 {
                a.calls.set(0);
                let mut x = [0.; 6];
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(count));
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                    .unwrap();
                let result = if count == 1 {
                    solver.solve_into(
                        &a,
                        BlockVectorView::new(&b, 3, 2).unwrap(),
                        Preconditioning::None,
                        BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                        &mut workspace,
                        &mut context,
                    )
                } else {
                    solver.solve_into(
                        &a,
                        BlockVectorView::new(&b, 3, 2).unwrap(),
                        Preconditioning::None,
                        BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                        &mut workspace,
                        &mut context,
                    )
                };
                assert!(a.calls.get() <= count);
                if count == 1 {
                    assert!(result.is_err());
                    assert_eq!(x, [0.; 6]);
                    continue;
                }
                let _report = result.unwrap();
                let accepted = workspace.last_accepted().unwrap();
                let columns = accepted.columns().unwrap();
                assert_eq!(accepted.solution(), &x);
                for col in 0..2 {
                    assert!(
                        (columns.residuals()[col]
                            - residual(&x[col * 3..col * 3 + 3], &b[col * 3..col * 3 + 3], 1.))
                        .abs()
                            < 1e-12
                    );
                }
            }
        }};
    }
    check!(BlockGmresSolver);
    check!(BlockLgmresSolver);
    check!(BlockGcrotSolver);
}

#[test]
fn recycle_epochs_refresh_and_repeated_rhs_are_explicit() {
    let a = provider();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    let solver = GcrotSolver::new(tolerance(), budget(12)).with_restart(budget(3));
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&a).unwrap())
        .unwrap();
    let mut recycle = KrylovRecycleSpace::new(3, 2, ProblemToken::new(10)).unwrap();
    workspace
        .try_reserve(
            &solver
                .workspace_requirements(
                    &IterativeWorkspaceRequest::operator(&a)
                        .unwrap()
                        .with_recycle_space(&recycle)
                        .unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    for solve in 0..2 {
        let mut x = Vector::zeros(3).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
            .unwrap();
        let report = solver
            .solve_recycled_into(
                &a,
                &b,
                Preconditioning::None,
                ProblemToken::new(10),
                &mut recycle,
                x.view_mut(),
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert!(report.is_success());
        if solve == 1 {
            assert_eq!(report.iterations(), 0);
        }
        assert!(residual(x.as_slice(), b.as_slice(), 1.) < 1e-11);
    }
    assert!(!recycle.is_empty());
    for index in 0..recycle.len() {
        let (u, c) = recycle.pair(index).unwrap();
        assert!(
            u.iter()
                .zip(c)
                .enumerate()
                .all(|(i, (&u, &c))| ((i + 1) as f64 * u - c).abs() < 1e-12)
        );
    }
    let changed = Provider {
        scale: 2.,
        ..provider()
    };
    let mut x = Vector::from_vec(vec![7., 8., 9.]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    assert!(
        solver
            .solve_recycled_into(
                &changed,
                &b,
                Preconditioning::None,
                ProblemToken::new(11),
                &mut recycle,
                x.view_mut(),
                &mut workspace,
                &mut context
            )
            .is_err()
    );
    assert_eq!(changed.calls.get(), 0);
    assert_eq!(x.as_slice(), [7., 8., 9.]);
    let mut refresh = recycle
        .workspace(
            &scioxide_linalg::KrylovRecycleWorkspaceRequest::new(
                &changed,
                &scioxide_linalg::LinalgExecutionOptions::new(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(
        recycle
            .refresh(
                &changed,
                ProblemToken::new(11),
                RankThreshold::new(0., 1e-12).unwrap(),
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut refresh,
                &mut context
            )
            .unwrap()
    );
    assert_eq!(recycle.operator_epoch(), ProblemToken::new(11));
    x.view_mut().fill(0.);
    let result = solver
        .solve_recycled_into(
            &changed,
            &b,
            Preconditioning::None,
            ProblemToken::new(11),
            &mut recycle,
            x.view_mut(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.iterations(), 0);
    assert!(residual(x.as_slice(), b.as_slice(), 2.) < 1e-11);
    recycle.clear(ProblemToken::new(12));
    assert!(recycle.is_empty());
}

#[test]
fn provider_errors_preserve_solution_and_retained_space() {
    let a = provider();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    let solver = GcrotSolver::new(tolerance(), budget(12));
    let mut recycle = KrylovRecycleSpace::new(3, 2, ProblemToken::new(0)).unwrap();
    let mut workspace = solver
        .workspace(&IterativeWorkspaceRequest::operator(&a).unwrap())
        .unwrap();
    for failure in [2, 3, 4, 5] {
        let bad = Provider {
            fail: failure,
            ..provider()
        };
        let mut x = Vector::from_vec(vec![7., 8., 9.]);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
            .unwrap();
        assert!(
            solver
                .solve_recycled_into(
                    &bad,
                    &b,
                    Preconditioning::None,
                    ProblemToken::new(0),
                    &mut recycle,
                    x.view_mut(),
                    &mut workspace,
                    &mut context
                )
                .is_err()
        );
        assert_eq!(x.as_slice(), [7., 8., 9.]);
        assert!(recycle.is_empty());
    }
}

#[test]
fn cancellation_during_a_provider_restores_verified_state() {
    let cancel = CancellationToken::new();
    let a = Provider {
        cancel: Some(&cancel),
        ..provider()
    };
    let solver = GmresSolver::new(tolerance(), budget(8)).with_history(true);
    let mut workspace = IterativeWorkspace::new();
    let mut x = Vector::zeros(3).unwrap();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancel),
            parent(),
            &mut budgets,
        )
        .unwrap();
    let report = solver
        .solve_into(
            &a,
            &b,
            Preconditioning::None,
            x.view_mut(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(matches!(
        report.termination(),
        LinearSolveTermination::ExecutionStopped(_)
    ));
    assert!(
        (report.true_residual().unwrap().raw_norm() - residual(x.as_slice(), b.as_slice(), 1.))
            .abs()
            < 1e-12
    );
}

#[test]
fn dense_sparse_views_and_operators_share_results() {
    let a = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 3.]).unwrap();
    let csr = a.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let csc = csr.to_csc().unwrap();
    let csr_view = csr.view();
    let csc_view = csc.view();
    let view = a.view();
    let p = provider();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    for op in [
        &a.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &view.try_finite().unwrap(),
        &csr,
        &csc,
        &csr_view,
        &csc_view,
        &p,
    ] {
        let solver = GmresSolver::new(tolerance(), budget(12));
        let result = solver
            .solve_with_options(op, &b, Preconditioning::None, None)
            .unwrap();
        assert!(result.is_success());
        assert!(residual(result.solution().as_slice(), b.as_slice(), 1.) < 1e-11);
        #[cfg(feature = "std")]
        {
            let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
            let mut workspace = IterativeWorkspace::new();
            let mut x = Vector::zeros(3).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::new(&pool), parent(), &mut budgets)
                .unwrap();
            let report = solver
                .solve_into(
                    op,
                    &b,
                    Preconditioning::None,
                    x.view_mut(),
                    &mut workspace,
                    &mut context,
                )
                .unwrap();
            assert!(report.is_success());
            assert!(residual(x.as_slice(), b.as_slice(), 1.) < 1e-11);
        }
    }
}

#[test]
fn empty_shapes_and_invalid_inputs_are_explicit() {
    let solver = BlockGmresSolver::<f64>::new(tolerance(), budget(3));
    let empty = Matrix::zeros(0, 0).unwrap();
    for cols in [0, 2] {
        let result = solver
            .solve(
                &empty.view().try_finite().unwrap(),
                BlockVectorView::new(&[], 0, cols).unwrap(),
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert_eq!(
            result.report().termination(),
            LinearSolveTermination::Converged
        );
    }
    let a = provider();
    let result = solver
        .solve(
            &a,
            BlockVectorView::new(&[], 3, 0).unwrap(),
            Preconditioning::None,
            None,
        )
        .unwrap();
    assert_eq!(result.report().iterations(), 0);
    assert_eq!(a.calls.get(), 0);
    let mut workspace = IterativeWorkspace::new();
    let mut x = [7.; 3];
    assert!(
        solver
            .solve_into(
                &a,
                BlockVectorView::new(&[f64::NAN, 0., 0.], 3, 1).unwrap(),
                Preconditioning::None,
                BlockVectorViewMut::new(&mut x, 3, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(x, [7.; 3]);
    assert_eq!(a.calls.get(), 0);
}

#[test]
fn retained_block_methods_reuse_and_refresh_transactionally() {
    let a = provider();
    let rhs = [1., 1., 1., 2., 2., 2., 1., 0., 1.];
    macro_rules! check {
        ($solver:ident) => {{
            let solver = $solver::new(tolerance(), budget(40))
                .with_restart(budget(2))
                .with_history(true);
            let mut workspace = solver
                .workspace(
                    &IterativeWorkspaceRequest::operator(&a)
                        .unwrap()
                        .with_right_hand_sides(3),
                )
                .unwrap();
            let mut recycle = KrylovRecycleSpace::new(3, 3, ProblemToken::new(8)).unwrap();
            workspace
                .try_reserve(
                    &solver
                        .workspace_requirements(
                            &IterativeWorkspaceRequest::operator(&a)
                                .unwrap()
                                .with_right_hand_sides(3)
                                .with_recycle_space(&recycle)
                                .unwrap(),
                        )
                        .unwrap(),
                )
                .unwrap();
            for _ in 0..2 {
                let mut x = [0.; 9];
                let mut state = SolveState::new();
                let mut budgets = SolveBudgets::unbounded();
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                    .unwrap();
                let report = solver
                    .solve_recycled_into(
                        &a,
                        BlockVectorView::new(&rhs, 3, 3).unwrap(),
                        Preconditioning::None,
                        ProblemToken::new(8),
                        &mut recycle,
                        BlockVectorViewMut::new(&mut x, 3, 3).unwrap(),
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                assert!(report.is_success());
                assert!(report.retained() > 0);
                assert_eq!(
                    workspace
                        .last_accepted()
                        .unwrap()
                        .history()
                        .unwrap()
                        .last()
                        .copied(),
                    report.max_residual()
                );
                for col in 0..3 {
                    assert!(
                        residual(&x[col * 3..col * 3 + 3], &rhs[col * 3..col * 3 + 3], 1.) < 1e-11
                    );
                }
            }
            let saved: Vec<_> = (0..recycle.len())
                .map(|i| {
                    let (u, c) = recycle.pair(i).unwrap();
                    (u.to_vec(), c.to_vec())
                })
                .collect();
            let bad = Provider {
                fail: 1,
                ..provider()
            };
            let mut refresh = recycle
                .workspace(
                    &scioxide_linalg::KrylovRecycleWorkspaceRequest::new(
                        &bad,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    )
                    .unwrap(),
                )
                .unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
                .unwrap();
            assert!(
                recycle
                    .refresh(
                        &bad,
                        ProblemToken::new(9),
                        RankThreshold::new(0., 1e-12).unwrap(),
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        &mut refresh,
                        &mut context
                    )
                    .is_err()
            );
            assert_eq!(recycle.operator_epoch(), ProblemToken::new(8));
            for (i, (u, c)) in saved.iter().enumerate() {
                assert_eq!(recycle.pair(i).unwrap(), (u.as_slice(), c.as_slice()));
            }
            let cancel = CancellationToken::new();
            cancel.cancel();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential().with_cancellation(&cancel),
                    parent(),
                    &mut budgets,
                )
                .unwrap();
            assert!(
                !recycle
                    .refresh(
                        &a,
                        ProblemToken::new(9),
                        RankThreshold::new(0., 1e-12).unwrap(),
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                        &mut refresh,
                        &mut context
                    )
                    .unwrap()
            );
            assert_eq!(recycle.operator_epoch(), ProblemToken::new(8));
        }};
    }
    check!(BlockLgmresSolver);
    check!(BlockGcrotSolver);
}

#[test]
fn restart_sensitivity_and_lgmres_raw_corrections_are_observable() {
    let a = provider();
    let b = Vector::from_vec(vec![1., 1., 1.]);
    let short = GmresSolver::new(tolerance(), budget(100))
        .with_restart(budget(1))
        .solve_with_options(&a, &b, Preconditioning::None, None)
        .unwrap();
    let full = GmresSolver::new(tolerance(), budget(100))
        .with_restart(budget(3))
        .solve_with_options(&a, &b, Preconditioning::None, None)
        .unwrap();
    assert!(short.is_success() && full.is_success());
    assert!(short.report().iterations() > full.report().iterations());
    let solver = LgmresSolver::new(tolerance(), budget(3)).with_restart(budget(3));
    let mut workspace = IterativeWorkspace::new();
    let mut space = KrylovRecycleSpace::new(3, 1, ProblemToken::new(1)).unwrap();
    let mut x = Vector::zeros(3).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    let result = solver
        .solve_recycled_into(
            &a,
            &b,
            Preconditioning::None,
            ProblemToken::new(1),
            &mut space,
            x.view_mut(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(result.is_success());
    let (u, image) = space.pair(0).unwrap();
    let scale = x.as_slice()[0] / u[0];
    for (&direction, &solution) in u.iter().zip(x.as_slice()) {
        assert!((direction * scale - solution).abs() < 1e-12);
    }
    assert!((image.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-12);
}

#[test]
fn context_aware_operator_charges_nested_work_once() {
    struct Nested;
    impl LinearOperator<f64> for Nested {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("context"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "context-aware entrypoint required",
            ))
        }
        fn apply_into_scoped(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
            context: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
        ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
            assert!(
                context
                    .record_provider(EvaluationRole::Function)?
                    .is_granted()
            );
            y[0] = 2.0 * x[0];
            Ok(scioxide_core::Invocation::Evaluated(()))
        }
    }
    let solver = GmresSolver::new(tolerance(), budget(4));
    let mut workspace = IterativeWorkspace::new();
    let mut x = Vector::zeros(1).unwrap();
    let b = Vector::from_vec(vec![4.0]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    let report = solver
        .solve_into(
            &Nested,
            &b,
            Preconditioning::None,
            x.view_mut(),
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(report.is_success());
    let counts = workspace.last_attempt().unwrap().counts();
    assert_eq!(
        counts.own().provider(EvaluationRole::Function),
        counts.own().provider(EvaluationRole::LinearOperator)
    );
    assert_eq!(
        context.counts().nested().provider(EvaluationRole::Function),
        counts.own().provider(EvaluationRole::Function)
    );
    assert!((x.as_slice()[0] - 2.0).abs() < 1e-12);
}

#[test]
fn pre_cancelled_blocks_preserve_input_without_stale_residuals() {
    use scioxide_linalg::BlockGmresSolver;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let b = [1., 1., 1., 2., 0., 3.];
    macro_rules! check {
        ($solver:ident) => {{
            let a = provider();
            let solver = $solver::new(tolerance(), budget(8)).with_history(true);
            let mut workspace = IterativeWorkspace::new();
            let mut x = [0.; 6];
            assert!(
                solver
                    .solve_into(
                        &a,
                        BlockVectorView::new(&b, 3, 2).unwrap(),
                        Preconditioning::None,
                        BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                        &mut workspace,
                        ExecutionContext::sequential()
                    )
                    .unwrap()
                    .is_success()
            );
            x.fill(7.);
            a.calls.set(0);
            let report = solver
                .solve_into(
                    &a,
                    BlockVectorView::new(&b, 3, 2).unwrap(),
                    Preconditioning::None,
                    BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                    &mut workspace,
                    ExecutionContext::sequential().with_cancellation(&cancellation),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                LinearSolveTermination::ExecutionStopped(
                    scioxide_core::ExecutionStopReason::Cancelled
                )
            );
            assert_eq!(report.iterations(), 0);
            assert!(report.max_residual().is_none());
            assert_eq!(x, [7.; 6]);
            assert_eq!(a.calls.get(), 0);
            let accepted = workspace.last_accepted().unwrap();
            assert!(accepted.history().is_none());
            assert!(accepted.columns().is_none());
        }};
    }
    check!(BlockGmresSolver);
    check!(BlockGmresSolver);
    check!(BlockLgmresSolver);
    check!(BlockGcrotSolver);
}

#[test]
fn recycle_space_alone_controls_scalar_and_block_capacity() {
    let a = provider();
    let epoch = ProblemToken::new(1218);
    let b = [1., 1., 1., -1., 2., -3.];
    macro_rules! check {
        ($scalar:ident, $block:ident) => {{
            let scalar = $scalar::new(tolerance(), budget(40)).with_restart(budget(3));
            let block = $block::new(tolerance(), budget(40)).with_restart(budget(3));
            let mut workspace = IterativeWorkspace::new();
            for capacity in [3, 0, 1, 2] {
                let mut space = KrylovRecycleSpace::new(3, capacity, epoch).unwrap();
                let request = IterativeWorkspaceRequest::operator(&a).unwrap();
                workspace
                    .try_reserve(
                        &scalar
                            .workspace_requirements(&request.with_recycle_space(&space).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
                for _ in 0..2 {
                    let mut x = [0.; 3];
                    let report = scalar
                        .solve_recycled_into(
                            &a,
                            &b[..3],
                            Preconditioning::None,
                            epoch,
                            &mut space,
                            scioxide_linalg::VectorViewMut::from_slice(&mut x),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert!(report.is_success());
                    assert!(residual(&x, &b[..3], 1.) < 1e-11);
                    assert!(space.len() <= capacity);
                    assert_eq!(space.is_empty(), capacity == 0);
                }
                space.clear(epoch);
                workspace
                    .try_reserve(
                        &block
                            .workspace_requirements(
                                &request
                                    .with_right_hand_sides(2)
                                    .with_recycle_space(&space)
                                    .unwrap(),
                            )
                            .unwrap(),
                    )
                    .unwrap();
                for _ in 0..2 {
                    let mut x = [0.; 6];
                    let report = block
                        .solve_recycled_into(
                            &a,
                            BlockVectorView::new(&b, 3, 2).unwrap(),
                            Preconditioning::None,
                            epoch,
                            &mut space,
                            BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert!(report.is_success());
                    assert!(
                        x.chunks_exact(3)
                            .zip(b.chunks_exact(3))
                            .all(|(x, b)| residual(x, b, 1.) < 1e-11)
                    );
                    assert_eq!(report.retained(), space.len());
                    assert!(space.len() <= capacity);
                    assert_eq!(space.is_empty(), capacity == 0);
                }
                let wrong_shape = KrylovRecycleSpace::<f64>::new(2, capacity, epoch).unwrap();
                assert!(request.with_recycle_space(&wrong_shape).is_err());
            }
        }};
    }
    check!(LgmresSolver, BlockLgmresSolver);
    check!(GcrotSolver, BlockGcrotSolver);
}

#[test]
fn block_shape_errors_name_the_mismatched_argument() {
    let solver = BlockGmresSolver::<f64>::new(tolerance(), budget(3));
    let a = provider();
    let mut workspace = IterativeWorkspace::new();
    let mismatch = |error: SciOxideError, expected: [usize; 2], actual: [usize; 2]| {
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::DimensionMismatch { expected: e, actual: a }
                    if e == expected && a == actual
            ),
            "{error:?}"
        );
    };
    let mut x = [7.; 6];
    let short = [1.; 4];
    mismatch(
        solver
            .solve_into(
                &a,
                BlockVectorView::new(&short, 2, 2).unwrap(),
                Preconditioning::None,
                BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err(),
        [3, 2],
        [2, 2],
    );
    let b = [1.; 6];
    mismatch(
        solver
            .solve_into(
                &a,
                BlockVectorView::new(&b, 3, 2).unwrap(),
                Preconditioning::None,
                BlockVectorViewMut::new(&mut x[..4], 2, 2).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err(),
        [3, 2],
        [2, 2],
    );
    assert_eq!(x, [7.; 6]);
    assert_eq!(a.calls.get(), 0);
}
