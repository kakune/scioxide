//! Prepared programming and verification lifecycles: success, certificates,
//! preflight rejection, option and parent-scope budget stops, injected
//! providers, simplex resumes and plan thread-safety.
#![allow(clippy::unwrap_used)]
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    AttemptKind, AttemptOutcome, Bounds, Budget, BudgetKind, CancellationToken, ExecutionContext,
    ProblemToken, SciOxideError, SciOxideErrorView, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_optimize::{
    BasisMatrix, BasisProvider, BranchAndBound, BranchAndBoundWorkspace, DenseBasisProvider,
    DenseKktProvider, InteriorPointLp, KktProvider, LinearConstraintSense as Sense,
    LinearConstraints, LpProblem, LpVerificationProblem, MipProblem, MipResult,
    OptimizeTermination as Stop, ProgrammingCertificate, ProgrammingResult, ProgrammingVerifier,
    QapDirection, QapMethod, QapOptions, QapPlan, QapProblemView, QapResult, QpMethod, QpProblem,
    QpSolver, QpVerificationProblem, SimplexSolver, VerificationOptions, VerificationReport,
    VerificationStatus,
};

fn lp(c: Vec<f64>, a: Vec<f64>, b: Vec<f64>, sense: Vec<Sense>) -> LpProblem<f64> {
    let n = c.len();
    let m = b.len();
    LpProblem::new(c)
        .unwrap()
        .with_constraints(
            LinearConstraints::new(Matrix::from_vec(m, n, a).unwrap(), b, sense).unwrap(),
        )
        .unwrap()
}
/// min -x - 2y, x + y <= 4, x <= 3, y <= 2: optimum -6 at (2, 2).
fn feasible() -> LpProblem<f64> {
    lp(
        vec![-1., -2.],
        vec![1., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    )
}
/// x + y <= 1 and x + y >= 3.
fn infeasible() -> LpProblem<f64> {
    lp(
        vec![1., 1.],
        vec![1., 1., -1., -1.],
        vec![1., -3.],
        vec![Sense::LessEqual; 2],
    )
}
/// min -x with x - y = 1, x, y >= 0.
fn unbounded() -> LpProblem<f64> {
    lp(vec![-1., 0.], vec![1., -1.], vec![1.], vec![Sense::Equal])
}
fn cancelled() -> CancellationToken {
    let token = CancellationToken::new();
    token.cancel();
    token
}
fn parent(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "programming-parent", 1),
        0,
        0,
    )
}
fn stopped_execution(error: &SciOxideError) -> bool {
    matches!(error.view(), SciOxideErrorView::ExecutionStopped { .. })
}
fn outcome<P>(record: Option<&scioxide_core::AttemptRecord<P>>) -> AttemptOutcome {
    record.unwrap().outcome()
}

/// Dense basis factorization that counts provider calls.
struct CountingBasis {
    inner: DenseBasisProvider<f64>,
    factors: Arc<AtomicUsize>,
}
impl BasisProvider<f64> for CountingBasis {
    fn factor(
        &mut self,
        basis: BasisMatrix<'_, f64>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.factors.fetch_add(1, Ordering::Relaxed);
        self.inner.factor(basis, execution)
    }
    fn solve_into(
        &mut self,
        rhs: &[f64],
        output: &mut [f64],
        transpose: bool,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.inner.solve_into(rhs, output, transpose, execution)
    }
}

/// Dense KKT factorization that counts provider calls.
struct CountingKkt {
    inner: DenseKktProvider<f64>,
    factors: Arc<AtomicUsize>,
}
impl KktProvider<f64> for CountingKkt {
    fn factor(
        &mut self,
        matrix: MatrixView<'_, f64>,
        execution: ExecutionContext<'_>,
    ) -> Result<scioxide_sparse::LdlInertia, SciOxideError> {
        self.factors.fetch_add(1, Ordering::Relaxed);
        self.inner.factor(matrix, execution)
    }
    fn solve_into(
        &mut self,
        rhs: &[f64],
        output: &mut [f64],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.inner.solve_into(rhs, output, execution)
    }
}

#[test]
fn simplex_publishes_certificates_records_attempts_and_rejects_in_preflight() {
    let solver = SimplexSolver::default();
    let p = feasible();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert!(out.is_success());
    assert!((out.fun() + 6.).abs() < 1e-10);
    assert!(out.verification_report().unwrap().is_accepted());
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Succeeded);
    let accepted = workspace.last_accepted().unwrap();
    assert_eq!(accepted.point(), out.x());
    assert_eq!(accepted.progress().objective(), out.fun());
    for (problem, stop) in [
        (infeasible(), Stop::Infeasible),
        (unbounded(), Stop::Unbounded),
    ] {
        let mut out = ProgrammingResult::empty();
        solver
            .solve_into(
                &problem,
                &mut out,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(out.termination(), stop, "{out:?}");
        assert!(out.verification_report().unwrap().is_accepted());
        assert!(out.certificate().is_some());
    }
    let previous = out.clone();
    let error = solver
        .solve_into(
            &p,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancelled()),
        )
        .unwrap_err();
    assert!(stopped_execution(&error));
    assert_eq!(out.x(), previous.x());
    assert_eq!(out.termination(), previous.termination());
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    assert!(workspace.last_accepted().is_none());
}

#[test]
fn simplex_budget_stops_through_options_and_parent_scope() {
    let p = feasible();
    let solver = SimplexSolver::default().with_max_iterations(Budget::ONE);
    let mut workspace = solver.workspace(&p).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(out.termination(), Stop::IterationLimit);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    assert!(workspace.last_accepted().is_some());
    // Evaluations are shared: a parent allowance stops the child epoch.
    let solver = SimplexSolver::default();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(1), &mut budgets)
        .unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(out.termination(), Stop::EvaluationLimit);
    assert_eq!(out.counts().total().unwrap(), context.counts().nested());
}

#[test]
fn simplex_injected_basis_provider_serves_cold_and_resumed_calls() {
    let factors = Arc::new(AtomicUsize::new(0));
    let solver = SimplexSolver::default();
    let p = feasible();
    let mut workspace =
        solver
            .workspace(&p)
            .unwrap()
            .with_basis_provider(Box::new(CountingBasis {
                inner: DenseBasisProvider::new(8).unwrap(),
                factors: Arc::clone(&factors),
            }));
    let mut cold = ProgrammingResult::empty();
    solver
        .solve_into(
            &p,
            &mut cold,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(cold.is_success());
    let after_cold = factors.load(Ordering::Relaxed);
    assert!(after_cold > 0);
    let reference = SimplexSolver::default().solve(&p).unwrap();
    assert_eq!(cold.x(), reference.x());
    // A resume with a changed matrix restarts cold through the same provider.
    let warm = workspace.warm_start().unwrap();
    let changed = lp(
        vec![-1., -2.],
        vec![2., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    );
    let mut resumed = ProgrammingResult::empty();
    solver
        .resume_into(
            &changed,
            &warm,
            &mut resumed,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(resumed.is_success());
    assert!(!workspace.warm_start().unwrap().reused_basis());
    assert!(factors.load(Ordering::Relaxed) > after_cold);
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        AttemptKind::Resume
    );
}

#[test]
fn simplex_resume_reuses_bases_for_rhs_changes_and_appended_rows() {
    let solver = SimplexSolver::default();
    let p = feasible();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    let warm = workspace.warm_start().unwrap();
    let tighter = lp(
        vec![-1., -2.],
        vec![1., 1., 1., 0., 0., 1., 1., 0.],
        vec![4., 3., 2., 1.],
        vec![Sense::LessEqual; 4],
    );
    let mut resumed = ProgrammingResult::empty();
    solver
        .resume_into(
            &tighter,
            &warm,
            &mut resumed,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let cold = solver.solve(&tighter).unwrap();
    assert!(resumed.is_success(), "{resumed:?}");
    assert!(workspace.warm_start().unwrap().reused_basis());
    assert!((resumed.fun() - cold.fun()).abs() < 1e-12);
    // The snapshot itself is immutable.
    assert!(!warm.reused_basis());
}

#[test]
fn interior_point_lp_lifecycle_provider_and_shape_checks() {
    let solver = InteriorPointLp::default();
    let p = feasible();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert!(out.is_success());
    assert!((out.fun() + 6.).abs() < 1e-7);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Succeeded);
    for (problem, stop) in [
        (infeasible(), Stop::Infeasible),
        (unbounded(), Stop::Unbounded),
    ] {
        let r = solver.solve(&problem).unwrap();
        assert_eq!(r.termination(), stop, "{r:?}");
        assert!(r.verification_report().unwrap().is_accepted());
    }
    // A different standard-form shape is rejected before factorization.
    let previous = out.fun();
    assert!(matches!(
        solver
            .solve_into(
                &unbounded(),
                &mut out,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(out.fun(), previous);
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    let error = solver
        .solve_into(
            &p,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancelled()),
        )
        .unwrap_err();
    assert!(stopped_execution(&error));
    assert_eq!(out.fun(), previous);
    let limited = InteriorPointLp::default().with_max_iterations(Budget::ONE);
    limited
        .solve_into(&p, &mut out, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert_eq!(out.termination(), Stop::IterationLimit);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    let factors = Arc::new(AtomicUsize::new(0));
    let mut workspace = solver
        .workspace(&p)
        .unwrap()
        .with_kkt_provider(Box::new(CountingKkt {
            // An augmented (columns + rows) factorization exceeds this cap.
            inner: DenseKktProvider::new(3).unwrap(),
            factors: Arc::clone(&factors),
        }));
    let mut injected = ProgrammingResult::empty();
    solver
        .solve_into(
            &p,
            &mut injected,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(injected.is_success());
    assert!((injected.fun() + 6.).abs() < 1e-7);
    assert!(factors.load(Ordering::Relaxed) > 0);
}

#[test]
fn qp_lifecycle_certificates_budgets_and_injected_kkt() {
    let qp = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(1, 2, vec![1., 1.]).unwrap(),
            vec![2.],
            vec![Sense::LessEqual],
        )
        .unwrap(),
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), None); 2]).unwrap())
    .unwrap();
    let solver = QpSolver::default();
    let mut workspace = solver.workspace(&qp).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(
            &qp,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(out.is_success(), "{out:?}");
    assert!(out.x()[0].abs() < 1e-8 && (out.x()[1] - 2.).abs() < 1e-8);
    assert_eq!(workspace.last_accepted().unwrap().point(), out.x());
    let contradictory = QpProblem::<f64>::new(Matrix::identity(2).unwrap(), vec![0., 0.])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(2, 2, vec![1., 1., 1., 1.]).unwrap(),
                vec![1., 2.],
                vec![Sense::Equal; 2],
            )
            .unwrap(),
        )
        .unwrap();
    let r = solver.solve(&contradictory).unwrap();
    assert!(matches!(
        r.termination(),
        Stop::Infeasible | Stop::SuspectedInfeasible
    ));
    let ray = QpProblem::<f64>::new(Matrix::from_vec(1, 1, vec![0.]).unwrap(), vec![-1.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), None)]).unwrap())
        .unwrap();
    let r = QpSolver::default()
        .with_method(QpMethod::ActiveSet)
        .solve(&ray)
        .unwrap();
    assert_eq!(r.termination(), Stop::Unbounded);
    let previous = out.x().to_vec();
    assert!(stopped_execution(
        &solver
            .solve_into(
                &qp,
                &mut out,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancelled()),
            )
            .unwrap_err()
    ));
    assert_eq!(out.x(), previous);
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    // Shared evaluations stop the child through the parent scope.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(2), &mut budgets)
        .unwrap();
    let bounded = QpProblem::<f64>::new(Matrix::from_vec(1, 1, vec![2.]).unwrap(), vec![-4.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(1.))]).unwrap())
        .unwrap();
    let mut workspace = solver.workspace(&bounded).unwrap();
    let mut stopped = ProgrammingResult::empty();
    solver
        .solve_into(&bounded, &mut stopped, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(stopped.termination(), Stop::EvaluationLimit);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    let factors = Arc::new(AtomicUsize::new(0));
    let primal_dual = QpSolver::default().with_method(QpMethod::PrimalDual);
    let mut workspace = primal_dual
        .workspace(&qp)
        .unwrap()
        .with_kkt_provider(Box::new(CountingKkt {
            inner: DenseKktProvider::new(16).unwrap(),
            factors: Arc::clone(&factors),
        }));
    let mut injected = ProgrammingResult::empty();
    primal_dual
        .solve_into(
            &qp,
            &mut injected,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(injected.is_success(), "{injected:?}");
    assert!(factors.load(Ordering::Relaxed) > 0);
}

fn knapsack() -> MipProblem<f64> {
    let p = LpProblem::new(vec![-3., -2., -4.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(1.)); 3]).unwrap())
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(1, 3, vec![2., 2., 3.]).unwrap(),
                vec![4.],
                vec![Sense::LessEqual],
            )
            .unwrap(),
        )
        .unwrap();
    MipProblem::new(p, vec![0, 1, 2]).unwrap()
}

#[test]
fn branch_and_bound_lifecycle_limits_and_custom_relaxation() {
    let problem = knapsack();
    let solver = BranchAndBound::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut out = MipResult::empty();
    solver
        .solve_into(
            &problem,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(out.is_success(), "{}", out.message());
    assert_eq!(out.fun(), -5.);
    assert_eq!(workspace.last_accepted().unwrap().point(), out.x());
    assert_eq!(workspace.state().incumbent(), Some(out.x()));
    let infeasible = MipProblem::new(infeasible(), vec![0]).unwrap();
    let r = BranchAndBound::default().solve(&infeasible).unwrap();
    assert_eq!(r.termination(), Stop::Infeasible);
    assert!(!r.has_incumbent());
    let previous = out.fun();
    assert!(stopped_execution(
        &solver
            .solve_into(
                &problem,
                &mut out,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancelled()),
            )
            .unwrap_err()
    ));
    assert_eq!(out.fun(), previous);
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    // Node limits through options and through a parent's shared node budget.
    let plain = || {
        BranchAndBound::default()
            .with_presolve(false)
            .with_branching(scioxide_optimize::MipBranching::MostFractional)
            .with_limits(
                scioxide_optimize::MipLimits::default()
                    .with_cut_rounds(0)
                    .with_repair(false),
            )
    };
    let limited = plain().with_max_nodes(Budget::ONE);
    limited
        .solve_into(
            &problem,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(out.termination(), Stop::NodeLimit);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(3), &mut budgets)
        .unwrap();
    plain()
        .solve_into(&problem, &mut out, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(out.termination(), Stop::NodeLimit);
    assert_eq!(context.budget_report(BudgetKind::Node).consumed(), 1);
    // A custom relaxation is owned by its workspace.
    let mut custom = BranchAndBoundWorkspace::new(
        SimplexSolver::default().with_tolerances(solver.relaxation_tolerances()),
    );
    let mut r = MipResult::empty();
    solver
        .solve_into(
            &problem,
            &mut r,
            &mut custom,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(r.fun(), -5.);
    assert!(r.statistics().relaxations() > 0);
}

fn qap() -> QapPlan<f64> {
    let a = [0., 3., 1., 8., 0., 2., 4., 6., 0.];
    let b = [0., 5., 7., 2., 0., 9., 3., 1., 0.];
    QapProblemView::new(
        MatrixView::from_slice(3, 3, &a).unwrap(),
        MatrixView::from_slice(3, 3, &b).unwrap(),
        &[],
    )
    .unwrap()
    .prepare()
    .unwrap()
}

#[test]
fn quadratic_assignment_lifecycle_and_termination_mapping() {
    let plan = qap();
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000)
        .with_restarts(2);
    let mut workspace = plan.workspace(&options).unwrap();
    let mut out = QapResult::empty(3).unwrap();
    plan.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out.termination(), Stop::Converged);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Succeeded);
    let owned = plan
        .solve_with_options(QapDirection::Minimize, None, &options)
        .unwrap();
    assert_eq!(owned.permutation(), out.permutation());
    let previous = out.objective();
    assert!(stopped_execution(
        &plan
            .solve_into(
                QapDirection::Minimize,
                None,
                &options,
                &mut out,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancelled())
            )
            .unwrap_err()
    ));
    assert_eq!(out.objective(), previous);
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    // Method limits map to IterationLimit and keep a checked permutation.
    let limited = options.clone().with_max_fw_iterations(Budget::ONE);
    plan.solve_into(
        QapDirection::Minimize,
        None,
        &limited,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out.termination(), Stop::IterationLimit);
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    assert!(out.objective().is_some());
}

#[test]
fn verifier_lifecycle_budgets_and_result_claims() {
    let problem = feasible();
    let view = LpVerificationProblem::from_problem(&problem);
    let result = SimplexSolver::default().solve(&problem).unwrap();
    let verifier = ProgrammingVerifier::default();
    let mut workspace = verifier.workspace(&view).unwrap();
    let mut report = VerificationReport::empty();
    verifier
        .verify_into(
            &view,
            &result,
            &mut report,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_accepted());
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Succeeded);
    // The bare certificate and the result claim agree.
    let bare = verifier
        .verify(&view, result.certificate().unwrap())
        .unwrap();
    assert_eq!(bare.status(), report.status());
    assert!(stopped_execution(
        &verifier
            .verify_into(
                &view,
                &result,
                &mut report,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancelled()),
            )
            .unwrap_err()
    ));
    assert!(report.is_accepted());
    assert_eq!(
        outcome(workspace.last_attempt()),
        AttemptOutcome::PreflightRejected
    );
    // An evaluation budget in options interrupts operator applications.
    let limited =
        ProgrammingVerifier::new(VerificationOptions::default().with_max_evaluations(Budget::ONE));
    let mut interrupted = VerificationReport::empty();
    limited
        .verify_into(
            &view,
            &result,
            &mut interrupted,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(interrupted.status(), VerificationStatus::Incomplete);
    assert_eq!(interrupted.termination(), Some(Stop::EvaluationLimit));
    assert_eq!(outcome(workspace.last_attempt()), AttemptOutcome::Stopped);
    // The same budget through a parent scope.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(4), &mut budgets)
        .unwrap();
    verifier
        .verify_into(
            &view,
            &result,
            &mut interrupted,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(interrupted.status(), VerificationStatus::Incomplete);
    // Malformed evidence is rejected before any operator work.
    let malformed = ProgrammingCertificate::Unboundedness {
        point: &[0.],
        direction: &[1.],
    };
    assert!(verifier.verify(&view, malformed).is_err());
    // QP claims use the same plan.
    let qp = QpProblem::<f64>::new(Matrix::identity(1).unwrap(), vec![-1.]).unwrap();
    let qp_result = QpSolver::default().solve(&qp).unwrap();
    let qp_view = QpVerificationProblem::from_problem(&qp);
    assert!(verifier.verify(&qp_view, &qp_result).unwrap().is_accepted());
}
