//! Prepared lifecycle contracts of the constrained solvers: success, preflight
//! rejection with unchanged output, budget stops with an accepted state,
//! nested budgets, allocation-free prepared calls, workspace-owned providers,
//! early problem validation and shareable plans.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    AttemptOutcome, Bounds, Budget, BudgetKind, ExecutionContext, PartialResultTrust, ProblemToken,
    SciOxideError, SciOxideErrorKind, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::MatrixView;
use scioxide_optimize::{
    AugmentedLagrangian, AugmentedLagrangianOptions, BoundedObjective, Cobyla, CobylaOptions,
    Cobyqa, CobyqaOptions, ConstrainedProblem, ConstraintFunction, Constraints, DenseKktProvider,
    DenseSqpSubproblem, InteriorPoint, InteriorPointOptions, KktProvider, ObjectiveFunction,
    OptimizeResult, OptimizeTermination, Optimizer, PenaltyMethod, PenaltyOptions, Sqp, SqpOptions,
    SqpState, SqpSubproblem, SqpSubproblemProvider, SqpSubproblemReport,
};
use scioxide_sparse::LdlInertia;

/// `(x - 1)² + (y - 2)²` with an analytic gradient and an objective-call count.
#[derive(Default)]
struct Bowl {
    calls: AtomicUsize,
}

impl ObjectiveFunction<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * (x[0] - 1.0);
        output[1] = 2.0 * (x[1] - 2.0);
        Ok(true)
    }
}

/// `x + y - 2 <= 0` with an exact, allocation-free Jacobian.
struct Halfplane;

impl ConstraintFunction<f64> for Halfplane {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 2.0)
    }
    fn jacobian_into(&self, _: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.fill(1.0);
        Ok(true)
    }
}

/// The halfplane inside an inactive box; the minimizer is `(0.5, 1.5)`.
fn constraints() -> Constraints<f64> {
    Constraints::new(2)
        .unwrap()
        .inequality(Halfplane)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(-1.0), Some(3.0)), (Some(-1.0), Some(3.0))]).unwrap())
        .unwrap()
}

const START: [f64; 2] = [2.5, 2.5];

fn assert_unchanged(before: &OptimizeResult<f64>, after: &OptimizeResult<f64>) {
    assert_eq!(before.x(), after.x());
    assert_eq!(before.fun(), after.fun());
    assert_eq!(before.termination(), after.termination());
    assert_eq!(before.counts(), after.counts());
}

fn parent_identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1988),
        scioxide_core::algorithm_id!("test", "ConstrainedParent", 1),
        2,
        0,
    )
}

/// Exercise one constrained solver's prepared lifecycle. Every prepared call,
/// including the first, is allocation-free.
macro_rules! constrained_lifecycle {
    ($test:ident, $solver:expr, $limited:expr, $timed:expr, $tolerance:expr) => {
        #[test]
        fn $test() {
            let objective = Bowl::default();
            let constraints = constraints();
            let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
            let solver = $solver;
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut output = OptimizeResult::<f64>::empty();
            output.try_reserve(2).unwrap();
            let sequential = ExecutionContext::sequential;

            // Success with prepared storage does not allocate.
            for _ in 0..3 {
                solver
                    .minimize_into(&problem, &START, &mut output, &mut workspace, sequential())
                    .unwrap();
                assert!(output.is_success(), "{output:?}");
                assert!((output.x()[0] - 0.5).abs() < $tolerance, "{output:?}");
                assert!((output.x()[1] - 1.5).abs() < $tolerance, "{output:?}");
                let attempt = workspace.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
                assert!(attempt.epoch().is_some());
                assert_eq!(attempt.counts(), output.counts());
                let accepted = workspace.last_accepted().unwrap();
                assert_eq!(accepted.point(), output.x());
                assert_eq!(accepted.progress().value(), output.fun());
                assert_eq!(
                    accepted.progress().trust(),
                    PartialResultTrust::ResidualChecked
                );
            }
            let owned = solver.minimize(&problem, &START).unwrap();
            assert!(owned.is_success(), "{owned:?}");
            assert!((owned.x()[0] - 0.5).abs() < $tolerance, "{owned:?}");

            // Preflight rejection changes neither the output nor any accepted state.
            let before = output.clone();
            let calls = objective.calls.load(Ordering::Relaxed);
            let error = solver
                .minimize_into(&problem, &[1.0], &mut output, &mut workspace, sequential())
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
            assert!(attempt.epoch().is_none());
            assert!(workspace.last_accepted().is_none());
            assert_unchanged(&before, &output);
            assert_eq!(objective.calls.load(Ordering::Relaxed), calls);

            // A time limit without a clock is rejected before any callback.
            let timed = $timed;
            let mut timed_workspace = timed.workspace(&problem).unwrap();
            let error = timed
                .minimize_into(
                    &problem,
                    &START,
                    &mut output,
                    &mut timed_workspace,
                    sequential(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
            assert_eq!(
                timed_workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::PreflightRejected
            );
            assert_unchanged(&before, &output);
            assert_eq!(objective.calls.load(Ordering::Relaxed), calls);

            // A budget stop publishes the accepted state as an unsuccessful result.
            let limited = $limited;
            let mut limited_workspace = limited.workspace(&problem).unwrap();
            limited
                .minimize_into(
                    &problem,
                    &START,
                    &mut output,
                    &mut limited_workspace,
                    sequential(),
                )
                .unwrap();
            assert!(!output.is_success(), "{output:?}");
            assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
            assert_eq!(
                limited_workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Stopped
            );
            let accepted = limited_workspace.last_accepted().unwrap();
            assert_eq!(accepted.point(), output.x());
            assert_eq!(accepted.progress().trust(), PartialResultTrust::Admissible);

            // A child epoch of a caller's solve charges its work to the ancestor.
            let objective = Bowl::default();
            let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(6).unwrap());
            let mut parent: SolveContext<'_, '_> = state
                .begin_solve(sequential(), parent_identity(), &mut budgets)
                .unwrap();
            let before = output.clone();
            let nested =
                solver.minimize_into(&problem, &START, &mut output, &mut workspace, &mut parent);
            let attempt = workspace.last_attempt().unwrap();
            let child = attempt.counts().total().unwrap();
            let charged = parent.counts();
            assert_eq!(charged.own().total_providers().unwrap(), 0);
            assert_eq!(
                charged.total().unwrap().total_providers().unwrap(),
                child.total_providers().unwrap()
            );
            assert_eq!(
                charged
                    .total()
                    .unwrap()
                    .provider(scioxide_core::EvaluationRole::Objective),
                objective.calls.load(Ordering::Relaxed)
            );
            assert!(objective.calls.load(Ordering::Relaxed) > 0);
            match nested {
                Ok(()) => {
                    assert!(!output.is_success(), "{output:?}");
                    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
                    assert_eq!(output.counts(), attempt.counts());
                }
                Err(_) => {
                    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
                    assert_unchanged(&before, &output);
                }
            }
        }
    };
}

constrained_lifecycle!(
    penalty,
    PenaltyMethod::<f64>::default(),
    PenaltyMethod::new(PenaltyOptions::default().with_max_outer_iterations(Budget::ONE)),
    PenaltyMethod::new(
        PenaltyOptions::default().with_time_limit(core::time::Duration::from_secs(1))
    ),
    1e-3
);
constrained_lifecycle!(
    augmented_lagrangian,
    AugmentedLagrangian::<f64>::default(),
    AugmentedLagrangian::new(
        AugmentedLagrangianOptions::default().with_max_outer_iterations(Budget::ONE)
    ),
    AugmentedLagrangian::new(
        AugmentedLagrangianOptions::default().with_time_limit(core::time::Duration::from_secs(1))
    ),
    1e-3
);
constrained_lifecycle!(
    sqp,
    Sqp::<f64>::default(),
    Sqp::new(SqpOptions::default().with_max_iterations(Budget::ONE)),
    Sqp::new(SqpOptions::default().with_time_limit(core::time::Duration::from_secs(1))),
    1e-6
);
constrained_lifecycle!(
    interior_point,
    InteriorPoint::<f64>::default(),
    InteriorPoint::new(InteriorPointOptions::default().with_max_iterations(Budget::ONE)),
    InteriorPoint::new(
        InteriorPointOptions::default().with_time_limit(core::time::Duration::from_secs(1))
    ),
    1e-5
);
constrained_lifecycle!(
    cobyla,
    Cobyla::<f64>::new(CobylaOptions::default().with_radii(0.5, 1e-8)).unwrap(),
    Cobyla::new(CobylaOptions::default().with_max_iterations(Budget::ONE)).unwrap(),
    Cobyla::new(CobylaOptions::default().with_time_limit(core::time::Duration::from_secs(1)))
        .unwrap(),
    1e-5
);
constrained_lifecycle!(
    cobyqa,
    Cobyqa::<f64>::new(CobyqaOptions::default().with_radii(0.5, 1e-8)).unwrap(),
    Cobyqa::new(CobyqaOptions::default().with_max_iterations(Budget::ONE)).unwrap(),
    Cobyqa::new(CobyqaOptions::default().with_time_limit(core::time::Duration::from_secs(1)))
        .unwrap(),
    1e-5
);

/// Dense SQP subproblems that count their calls.
#[derive(Debug)]
struct CountingSubproblem {
    inner: DenseSqpSubproblem<f64>,
    calls: Arc<AtomicUsize>,
}

impl SqpSubproblemProvider<f64> for CountingSubproblem {
    fn solve_into(
        &mut self,
        problem: SqpSubproblem<'_, f64>,
        state: &mut SqpState<f64>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<SqpSubproblemReport, SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.solve_into(problem, state, context)
    }
}

/// Dense KKT factors that count their factorizations.
struct CountingKkt {
    inner: DenseKktProvider<f64>,
    factors: Arc<AtomicUsize>,
}

impl KktProvider<f64> for CountingKkt {
    fn factor(
        &mut self,
        matrix: MatrixView<'_, f64>,
        execution: ExecutionContext<'_>,
    ) -> Result<LdlInertia, SciOxideError> {
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
fn providers_are_injected_through_the_workspace() {
    let objective = Bowl::default();
    let constraints = constraints();
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();

    let sqp = Sqp::<f64>::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut workspace =
        sqp.workspace(&problem)
            .unwrap()
            .with_subproblem_provider(CountingSubproblem {
                inner: DenseSqpSubproblem::new(2, 1, 1).unwrap(),
                calls: Arc::clone(&calls),
            });
    let mut output = OptimizeResult::empty();
    sqp.minimize_into(
        &problem,
        &START,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(output.is_success(), "{output:?}");
    assert!((output.x()[0] - 0.5).abs() < 1e-6);
    let first = calls.load(Ordering::Relaxed);
    assert!(first > 0);
    // The provider stays installed for later calls.
    sqp.minimize_into(
        &problem,
        &START,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2 * first);
    assert_eq!(workspace.state().multipliers().len(), 1);

    let interior = InteriorPoint::<f64>::default();
    let requirements = interior.workspace_requirements(&problem).unwrap();
    let order = requirements.dimension() + requirements.constraints() + requirements.bound_sides();
    assert_eq!(order, 2 + 1 + 4);
    let factors = Arc::new(AtomicUsize::new(0));
    let mut workspace = interior
        .workspace(&problem)
        .unwrap()
        .with_kkt_provider(CountingKkt {
            inner: DenseKktProvider::new(order).unwrap(),
            factors: Arc::clone(&factors),
        });
    let mut output = OptimizeResult::empty();
    interior
        .minimize_into(
            &problem,
            &START,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.is_success(), "{output:?}");
    assert!((output.x()[1] - 1.5).abs() < 1e-5);
    assert!(factors.load(Ordering::Relaxed) > 0);
    assert_eq!(workspace.state().slacks().len(), 1 + 4);
}

#[test]
fn problems_reject_mismatched_dimensions_disjoint_boxes_and_objective_bounds() {
    let objective = Bowl::default();
    let wrong = Constraints::<f64>::new(3).unwrap();
    assert_eq!(
        ConstrainedProblem::new(&objective, &wrong)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );

    let first = Bounds::new(vec![(Some(0.0), Some(1.0)), (None, None)]).unwrap();
    let disjoint = Bounds::new(vec![(Some(2.0), None), (None, None)]).unwrap();
    let error = Constraints::new(2)
        .unwrap()
        .bounds(first.clone())
        .unwrap()
        .bounds(disjoint)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(
        Constraints::<f64>::new(2)
            .unwrap()
            .bounds(Bounds::new(vec![(None, None); 3]).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    // Touching boxes fix a coordinate rather than empty the intersection.
    let touching = Bounds::new(vec![(Some(1.0), None), (None, None)]).unwrap();
    assert!(
        Constraints::<f64>::new(2)
            .unwrap()
            .bounds(first.clone())
            .unwrap()
            .bounds(touching)
            .is_ok()
    );

    let bounded = BoundedObjective::new(&objective, &first).unwrap();
    let constraints = Constraints::new(2).unwrap();
    assert_eq!(
        ConstrainedProblem::new(&bounded, &constraints)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );

    for (initial, last) in [(1e-4, 1.0), (f64::NAN, 1e-4), (1.0, 0.0)] {
        assert!(Cobyla::new(CobylaOptions::default().with_radii(initial, last)).is_err());
        assert!(Cobyqa::new(CobyqaOptions::default().with_radii(initial, last)).is_err());
    }
    assert_eq!(objective.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn one_plan_serves_problems_of_different_shapes() {
    let objective = Bowl::default();
    let constrained = constraints();
    let free = Constraints::new(2).unwrap();
    let solver = Sqp::<f64>::default();
    let mut output = OptimizeResult::empty();
    let mut workspace = scioxide_optimize::SqpWorkspace::new();
    for (constraints, expected) in [(&constrained, [0.5, 1.5]), (&free, [1.0, 2.0])] {
        let problem = ConstrainedProblem::new(&objective, constraints).unwrap();
        solver
            .minimize_into(
                &problem,
                &START,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success(), "{output:?}");
        assert!((output.x()[0] - expected[0]).abs() < 1e-6, "{output:?}");
        assert!((output.x()[1] - expected[1]).abs() < 1e-6, "{output:?}");
    }
}

#[test]
fn constrained_solvers_share_one_optimizer_interface() {
    let objective = Bowl::default();
    let constraints = constraints();
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    type ConstrainedOptimizer = dyn for<'a> Optimizer<f64, ConstrainedProblem<'a, f64>>;
    let solvers: [Box<ConstrainedOptimizer>; 6] = [
        Box::new(PenaltyMethod::default()),
        Box::new(AugmentedLagrangian::default()),
        Box::new(Sqp::default()),
        Box::new(InteriorPoint::default()),
        Box::new(Cobyla::default()),
        Box::new(Cobyqa::default()),
    ];
    for solver in &solvers {
        let result = solver.minimize(&problem, &START).unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.fun() - 0.5).abs() < 1e-2, "{result:?}");
    }
}

#[test]
fn plans_are_send_sync_and_clone() {
    fn plan<T: Send + Sync + Clone>(_: &T) {}
    plan(&PenaltyMethod::<f64>::default());
    plan(&AugmentedLagrangian::<f64>::default());
    plan(&Sqp::<f64>::default());
    plan(&InteriorPoint::<f64>::default());
    plan(&Cobyla::<f64>::default());
    plan(&Cobyqa::<f64>::default());
    plan(&PenaltyOptions::<f64>::default());
    plan(&SqpOptions::<f64>::default());
    plan(&CobylaOptions::<f64>::default());
    fn workspace<T: Send>(_: &T) {}
    let objective = Bowl::default();
    let constraints = constraints();
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    workspace(&Sqp::<f64>::default().workspace(&problem).unwrap());
    workspace(&InteriorPoint::<f64>::default().workspace(&problem).unwrap());
    workspace(&PenaltyMethod::<f64>::default().workspace(&problem).unwrap());
    workspace(&Cobyqa::<f64>::default().workspace(&problem).unwrap());
}
