//! Typed linear outcomes, structural gauge preflight, and scoped injected solvers.
#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use mesh_fixture::uniform;
use scioxide_core::{
    Budget, BudgetKind, ExecutionContext, InvalidInputReason, LinearOperator, NormSelection,
    PartialResultTrust, ResidualReport, ResidualTolerance, SciOxideError, SciOxideErrorView,
    SolveBudgets,
};
use scioxide_linalg::{
    CgSolver, GmresSolver, IterativeLinearSolver, IterativeSolveReport, IterativeWorkspace,
    LinalgExecutionContext, LinearSolveBreakdown, LinearSolveTermination, Preconditioning,
    VectorView, VectorViewMut,
};
use scioxide_pde::{
    EllipticGauge, EllipticOptions, EllipticProblem, FieldData, PdeSolveReport, PdeTermination,
    SpatialDiscretization,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use support::{Discretization, EllipticPde, End, Ends, Plan1d, Steady, value};

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}
fn count(calls: &AtomicUsize) -> usize {
    calls.load(Ordering::Relaxed)
}
#[derive(Clone)]
struct Problem {
    calls: Arc<AtomicUsize>,
    reaction: f64,
}
impl EllipticPde<f64> for Problem {
    fn diffusion(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    }
    fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.reaction)
    }
    fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok((core::f64::consts::PI * x[0]).sin() + x[0] * x[0])
    }
}
/// Homogeneous Dirichlet or `k du/dn + u = 0` ends whose data counts calls.
fn boundaries(robin: bool, calls: &Arc<AtomicUsize>) -> Ends<f64> {
    let end = || {
        let calls = Arc::clone(calls);
        let data = value(move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(0.0)
        });
        if robin {
            End::robin(1.0, data)
        } else {
            End::dirichlet(data)
        }
    };
    Ends::new(end(), end())
}
/// A prepared plan and its source data.
struct Case {
    plan: Plan1d<f64>,
    source: FieldData<f64, f64, 1>,
}
fn case<P: EllipticPde<f64> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    problem: P,
    ends: &Ends<f64>,
) -> Case {
    let mesh = uniform(0.0, 1.0, 33);
    let problem = Steady(problem);
    Case {
        plan: discretization.plan(&mesh, &problem, ends).unwrap(),
        source: support::source(&problem),
    }
}
fn default_case(discretization: Discretization, reaction: f64) -> Case {
    case(
        discretization,
        Problem {
            calls: counter(),
            reaction,
        },
        &boundaries(false, &counter()),
    )
}
type Solver = Option<Box<dyn IterativeLinearSolver<f64> + Send>>;
/// Solve `case` into caller storage; every error must leave that storage unchanged.
fn run(
    options: EllipticOptions<f64>,
    solver: Solver,
    case: &Case,
    execution: ExecutionContext<'_>,
) -> Result<PdeSolveReport<f64>, SciOxideError> {
    let elliptic = options;
    let mut workspace = scioxide_pde::EllipticWorkspace::new();
    if let Some(solver) = solver {
        workspace = workspace.with_linear_solver(solver);
    }
    const SENTINEL: f64 = -7.25;
    let mut out = vec![SENTINEL; with_plan!(&case.plan, |p| p.unknowns())];
    let result = with_plan!(&case.plan, |p| EllipticProblem::new(p, &case.source)
        .solve_into(&elliptic, &mut out, &mut workspace, execution));
    if result.is_err() {
        assert!(
            out.iter().all(|&u| u == SENTINEL),
            "failed solve wrote output"
        );
    }
    result
}
fn solve(
    options: EllipticOptions<f64>,
    solver: Solver,
    case: &Case,
) -> Result<PdeSolveReport<f64>, SciOxideError> {
    run(options, solver, case, ExecutionContext::sequential())
}
struct Outcome(LinearSolveTermination);
impl IterativeLinearSolver<f64> for Outcome {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "outcome-regression", 1)
    }
    fn solve_into(
        &self,
        _: &dyn LinearOperator<f64>,
        _: VectorView<'_, f64>,
        _: Preconditioning<'_, f64>,
        _x: VectorViewMut<'_, f64>,
        _: &mut IterativeWorkspace<f64>,
        _: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        Ok(IterativeSolveReport::new(
            self.0,
            1,
            Some(ResidualReport::unscaled(0.25, NormSelection::L2)?),
            PartialResultTrust::DiagnosticOnly,
        ))
    }
}
fn cg(tolerance: f64, budget: usize) -> Box<CgSolver<f64>> {
    Box::new(CgSolver::new(
        ResidualTolerance::new(tolerance, tolerance).unwrap(),
        Budget::try_from(budget).unwrap(),
    ))
}
fn contracts(discretization: Discretization) {
    for robin in [false, true] {
        for gauge in [EllipticGauge::Orthogonal, EllipticGauge::WeightedMean(2.0)] {
            let calls = counter();
            let case = case(
                discretization,
                Problem {
                    calls: Arc::clone(&calls),
                    reaction: 0.0,
                },
                &boundaries(robin, &calls),
            );
            // Coefficients are sampled when the plan is prepared; the gauge
            // preflight precedes every source and boundary-data callback.
            let prepared = count(&calls);
            let failure = solve(EllipticOptions::new().with_gauge(gauge), None, &case).unwrap_err();
            assert!(matches!(
                failure.view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
            assert_eq!(count(&calls), prepared);
        }
    }
    for (inner, expected) in [
        (
            LinearSolveTermination::ConditionLimit,
            PdeTermination::LinearConditionLimit,
        ),
        (
            LinearSolveTermination::Stagnation,
            PdeTermination::LinearStagnation,
        ),
        (
            LinearSolveTermination::Breakdown(LinearSolveBreakdown::NonPositiveCurvature),
            PdeTermination::LinearBreakdown(LinearSolveBreakdown::NonPositiveCurvature),
        ),
        (
            LinearSolveTermination::LeastSquaresConverged,
            PdeTermination::LinearLeastSquaresConverged,
        ),
    ] {
        let report = solve(
            EllipticOptions::new(),
            Some(Box::new(Outcome(inner))),
            &default_case(discretization, 0.0),
        )
        .unwrap();
        assert_eq!(report.termination(), expected);
        assert!(!report.is_success());
        assert_eq!(report.trust(), PartialResultTrust::DiagnosticOnly);
    }
    for (local, shared) in [(100, 1), (1, 100)] {
        let report = solve(
            EllipticOptions::new()
                .with_max_linear_iterations(Budget::try_from(local).unwrap())
                .with_budgets(
                    SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Iteration, Budget::try_from(shared).unwrap()),
                ),
            Some(cg(1e-12, 1000)),
            &default_case(discretization, 0.0),
        )
        .unwrap();
        assert_eq!(report.termination(), PdeTermination::LinearIterationLimit);
        assert_eq!(report.diagnostics().num_linear_iterations(), 1);
    }
    let boundary_calls = counter();
    let case = case(
        discretization,
        Problem {
            calls: counter(),
            reaction: 0.0,
        },
        &boundaries(false, &boundary_calls),
    );
    // The direct solve evaluates each endpoint value once (FEM preparation
    // additionally records the prescribed values at time zero).
    let prepared = count(&boundary_calls);
    let report = solve(EllipticOptions::new(), None, &case).unwrap();
    assert!(report.is_success());
    assert_eq!(count(&boundary_calls) - prepared, 2);
}
#[test]
fn fdm_contracts() {
    contracts(Discretization::Fdm);
}
#[test]
fn fem_contracts() {
    contracts(Discretization::Fem);
}
#[test]
fn fvm_contracts() {
    contracts(Discretization::Fvm);
}
#[test]
fn indefinite_cg_preserves_nonpositive_curvature() {
    let report = solve(
        EllipticOptions::new(),
        Some(cg(1e-10, 100)),
        &default_case(Discretization::Fdm, -1000.0),
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        PdeTermination::LinearBreakdown(LinearSolveBreakdown::NonPositiveCurvature)
    );
}
struct Cancel(Arc<scioxide_core::CancellationToken>);
impl IterativeLinearSolver<f64> for Cancel {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "cancel-delegate", 1)
    }
    fn solve_into(
        &self,
        a: &dyn LinearOperator<f64>,
        b: VectorView<'_, f64>,
        p: Preconditioning<'_, f64>,
        x: VectorViewMut<'_, f64>,
        w: &mut IterativeWorkspace<f64>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        self.0.cancel();
        cg(1e-10, 100).solve_into(a, b, p, x, w, context)
    }
}
#[test]
fn injected_solver_observes_cancellation() {
    let token = Arc::new(scioxide_core::CancellationToken::new());
    let failure = run(
        EllipticOptions::new(),
        Some(Box::new(Cancel(Arc::clone(&token)))),
        &default_case(Discretization::Fdm, 0.0),
        ExecutionContext::sequential().with_cancellation(&*token),
    )
    .unwrap_err();
    assert!(matches!(
        failure.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
}
#[derive(Clone)]
struct Single;
impl EllipticPde<f32> for Single {
    fn diffusion(&self, _: &[f32]) -> Result<f32, SciOxideError> {
        Ok(1.0)
    }
    fn reaction(&self, _: &[f32]) -> Result<f32, SciOxideError> {
        Ok(0.0)
    }
    fn source(&self, x: &[f32]) -> Result<f32, SciOxideError> {
        Ok(1.0 + x[0] * x[0])
    }
}
#[test]
fn strict_f32_iterative_solve_reports_stagnation() {
    let mesh = uniform(0.0_f32, 1.0, 33);
    let plan = Discretization::Fdm
        .plan(&mesh, &Steady(Single), &support::homogeneous_dirichlet())
        .unwrap();
    let elliptic = EllipticOptions::new();
    let mut workspace = scioxide_pde::EllipticWorkspace::new().with_linear_solver(Box::new(
        GmresSolver::new(
            ResidualTolerance::new(1e-12, 1e-12).unwrap(),
            Budget::try_from(1000).unwrap(),
        )
        .with_restart(Budget::try_from(31).unwrap()),
    ));
    let source = support::source(&Steady(Single));
    let mut out = vec![0.0_f32; with_plan!(&plan, |p| p.unknowns())];
    let report = with_plan!(&plan, |p| EllipticProblem::new(p, &source).solve_into(
        &elliptic,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential()
    ))
    .unwrap();
    assert_eq!(report.termination(), PdeTermination::LinearStagnation);
    // The unconverged iterate is published for diagnosis only.
    assert_eq!(report.trust(), PartialResultTrust::DiagnosticOnly);
    assert!(out.iter().all(|u| u.is_finite()));
}

#[derive(Clone)]
struct Singular;
impl EllipticPde<f64> for Singular {
    fn diffusion(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.0)
    }
    fn source(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}
#[test]
fn undeclared_singular_problem_never_completes() {
    let case = case(
        Discretization::Fdm,
        Singular,
        &boundaries(false, &counter()),
    );
    assert!(solve(EllipticOptions::new(), None, &case).is_err());
    let result = solve(
        EllipticOptions::new(),
        Some(Box::new(scioxide_linalg::MinresSolver::new(
            ResidualTolerance::new(1e-12, 1e-12).unwrap(),
            Budget::try_from(100).unwrap(),
        ))),
        &case,
    );
    if let Ok(report) = result {
        assert!(!report.is_success());
        assert_ne!(report.termination(), PdeTermination::Completed);
    }
}
#[test]
fn injected_solver_observes_ancestor_evaluation_limit() {
    // One evaluation admits load preparation; the second gives the inner
    // solver a residual to report before its next operator call is denied.
    let report = solve(
        EllipticOptions::new().with_budgets(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::ONE.saturating_add(1)),
        ),
        Some(cg(1e-12, 100)),
        &default_case(Discretization::Fdm, 0.0),
    )
    .unwrap();
    assert_eq!(report.termination(), PdeTermination::EvaluationLimit);
    // Without the second admission no residual exists, so the solve returns
    // a structured failure and preserves the caller's initial output.
    let error = solve(
        EllipticOptions::new().with_budgets(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        ),
        Some(cg(1e-12, 100)),
        &default_case(Discretization::Fdm, 0.0),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
}

struct TestClock(Arc<AtomicU64>);
impl scioxide_core::Clock for TestClock {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(self.0.load(Ordering::Relaxed)),
        ))
    }
}
struct Expire(Arc<AtomicU64>);
impl IterativeLinearSolver<f64> for Expire {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "deadline-delegate", 1)
    }
    fn solve_into(
        &self,
        a: &dyn LinearOperator<f64>,
        b: VectorView<'_, f64>,
        p: Preconditioning<'_, f64>,
        x: VectorViewMut<'_, f64>,
        w: &mut IterativeWorkspace<f64>,
        context: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        self.0.store(1, Ordering::Relaxed);
        cg(1e-10, 100).solve_into(a, b, p, x, w, context)
    }
}
#[test]
fn injected_solver_observes_the_callers_deadline() {
    let time = Arc::new(AtomicU64::new(0));
    let clock = TestClock(Arc::clone(&time));
    let binding = scioxide_core::ClockBinding::new(&clock);
    let deadline =
        scioxide_core::Deadline::after(&binding, core::time::Duration::from_secs(1)).unwrap();
    let execution = ExecutionContext::sequential()
        .with_deadline(deadline)
        .unwrap();
    let failure = run(
        EllipticOptions::new(),
        Some(Box::new(Expire(time))),
        &default_case(Discretization::Fdm, 0.0),
        execution,
    )
    .unwrap_err();
    assert!(matches!(
        failure.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
}
