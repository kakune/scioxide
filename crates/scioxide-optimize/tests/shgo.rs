#![allow(clippy::unwrap_used, clippy::expect_used)]

use scioxide_core::{
    Bounds, Budget, BudgetKind, ExecutionContext, ExecutionPolicy, ProblemToken, RealScalar,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    BorrowedStartDesign, BoundedObjective, ObjectiveFn, OptimizeTermination, PowellLocalFactory,
    Shgo, ShgoOptions, ShgoResult, UniformStartDesign,
};
use scioxide_spatial::{GeometryLimits, GeometryOptions};

#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;

fn geometry() -> GeometryOptions {
    GeometryOptions::new().with_limits(
        GeometryLimits::new()
            .with_max_dimension(3)
            .with_max_points(64)
            .with_max_facets(512)
            .with_max_predicate_terms(20_000_000)
            .with_max_storage_bytes(64 * 1024 * 1024),
    )
}

fn parent(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        0,
        0,
    )
}

fn boxed<S: RealScalar>(lower: f64, upper: f64, dimension: usize) -> Bounds<S> {
    let n = |x| S::try_from_f64_lossy(x).unwrap();
    Bounds::new(vec![(Some(n(lower)), Some(n(upper))); dimension]).unwrap()
}

fn multimodal<S: RealScalar>() {
    let n = |x| S::try_from_f64_lossy(x).unwrap();
    let bounds = boxed::<S>(-2.0, 2.0, 2);
    let samples = [
        n(-1.0),
        n(-1.0),
        n(0.0),
        n(-1.0),
        n(1.0),
        n(-1.0),
        n(-1.0),
        n(0.0),
        n(0.0),
        n(0.0),
        n(1.0),
        n(0.0),
        n(-1.0),
        n(1.0),
        n(0.0),
        n(1.0),
        n(1.0),
        n(1.0),
    ];
    let options = ShgoOptions::new(Budget::try_from(9).unwrap(), geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&samples, 2).unwrap(),
        PowellLocalFactory::default(),
    );
    let f = ObjectiveFn::new(2, |x: &[S]| {
        let a = x[0] * x[0] - S::ONE;
        let b = x[1] * x[1] - S::ONE;
        Ok(a * a + b * b)
    });
    let result = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert_eq!(result.values().len(), 13);
    assert!(result.simplex_count() > 0);
    assert!(result.best_sample().unwrap().1.abs() <= n(1e-6));
    assert!(result.candidates().len() >= 2);
    assert!(result.best_refined().unwrap().fun() < n(1e-5));
}

#[test]
fn bounded_multimodal_both_precisions() {
    multimodal::<f32>();
    multimodal::<f64>();
}

#[test]
fn deterministic_reuse_and_duplicate_plateau() {
    let bounds = boxed::<f64>(0.0, 1.0, 2);
    let options = ShgoOptions::new(Budget::try_from(3).unwrap(), geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&[0.5, 0.5, 0.5, 0.5, 0.0, 0.0], 2).unwrap(),
        PowellLocalFactory::default(),
    );
    let f = ObjectiveFn::new(2, |_x: &[f64]| Ok(1.0));
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut output = ShgoResult::empty();
    let mut work = plan.workspace(2).unwrap();
    plan.minimize_into(
        &problem,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output.candidates().len(), 1);
    assert!(work.complex().is_some());
    let first = output.candidates().to_vec();
    plan.minimize_into(
        &problem,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(first, output.candidates());
}

#[test]
fn rotated_ill_conditioned_and_boundary_minima() {
    let options = ShgoOptions::new(Budget::ONE, geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&[0.0, 0.0], 2).unwrap(),
        PowellLocalFactory::default(),
    );
    let bounds = boxed::<f64>(-1.0, 1.0, 2);
    let rotated = ObjectiveFn::new(2, |x: &[f64]| {
        Ok(1e6 * (x[0] + x[1]).powi(2) + (x[0] - x[1]).powi(2))
    });
    let result = plan
        .minimize(&BoundedObjective::new(&rotated, &bounds).unwrap())
        .unwrap();
    assert!(result.best_sample().unwrap().1.abs() <= 1e-12);
    assert!(result.best_refined().unwrap().fun() < 1e-8);
    let boundary = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] + x[1]));
    let result = plan
        .minimize(&BoundedObjective::new(&boundary, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.best_sample().unwrap().0.len(), 2);
    assert!(
        result
            .best_sample()
            .unwrap()
            .0
            .iter()
            .all(|x| (x + 1.0).abs() <= 1e-12)
    );
}

#[test]
fn shared_evaluation_budget_publishes_partial_samples() {
    let options = ShgoOptions::new(Budget::ONE, geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&[0.5], 1).unwrap(),
        PowellLocalFactory::default(),
    );
    let mut output = ShgoResult::empty();
    let mut work = plan.workspace(1).unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(0.0, 1.0, 1);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(4), &mut budgets)
        .unwrap();
    plan.minimize_into(
        &BoundedObjective::new(&f, &bounds).unwrap(),
        &mut output,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(output.values().len(), 1);
    assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
}

#[test]
fn geometry_node_budget_stops_after_sampling() {
    let options = ShgoOptions::new(Budget::ONE, geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&[0.5, 0.5], 2).unwrap(),
        PowellLocalFactory::default(),
    );
    let mut output = ShgoResult::empty();
    let mut work = plan.workspace(2).unwrap();
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    let bounds = boxed::<f64>(0.0, 1.0, 2);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(6), &mut budgets)
        .unwrap();
    plan.minimize_into(
        &BoundedObjective::new(&f, &bounds).unwrap(),
        &mut output,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(output.values().len(), 5);
    assert_eq!(output.termination(), OptimizeTermination::NodeLimit);
}

#[test]
fn provider_error_keeps_previous_result() {
    let options = ShgoOptions::new(Budget::ONE, geometry());
    let plan = Shgo::new(
        options,
        BorrowedStartDesign::new(&[0.5], 1).unwrap(),
        PowellLocalFactory::default(),
    );
    let mut output = ShgoResult::empty();
    let mut work = plan.workspace(1).unwrap();
    let bounds = boxed::<f64>(0.0, 1.0, 1);
    let good = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    plan.minimize_into(
        &BoundedObjective::new(&good, &bounds).unwrap(),
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let before = output.values().to_vec();
    let bad = ObjectiveFn::new(1, |_x: &[f64]| Ok(f64::NAN));
    assert!(
        plan.minimize_into(
            &BoundedObjective::new(&bad, &bounds).unwrap(),
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.values(), before);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Failed
    );
}

#[test]
fn seeded_one_dimensional_sampling_and_refinement_cap_replay() {
    let options = ShgoOptions::new(Budget::try_from(5).unwrap(), geometry())
        .with_max_refinements(Budget::ONE)
        .with_seed(19);
    let plan = Shgo::new(options, UniformStartDesign, PowellLocalFactory::default());
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 0.25).powi(2)));
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut output = ShgoResult::empty();
    let mut work = plan.workspace(1).unwrap();
    plan.minimize_into(
        &problem,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let points = output.points().to_vec();
    assert_eq!(output.values().len(), 7);
    assert_eq!(output.refinements().len(), 1);
    assert!(output.best_observed().unwrap().1 < 1e-7);
    plan.minimize_into(
        &problem,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output.points(), points);
    let accepted = work.last_accepted().unwrap();
    assert_eq!(
        Some((accepted.point(), accepted.progress().value())),
        output.best_observed()
    );
}

#[test]
fn geometry_point_limit_is_structured() {
    let geometry = geometry().with_limits(geometry().limits().with_max_points(4));
    let plan = Shgo::new(
        ShgoOptions::new(Budget::ONE, geometry),
        UniformStartDesign,
        PowellLocalFactory::<f64>::default(),
    );
    let error = plan.workspace_requirements(2).unwrap_err();
    assert!(matches!(
        error,
        error if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_points"))
    ));
    assert!(plan.workspace_requirements(1).is_ok());
}

#[test]
fn partitioned_policy_matches_sequential_with_unbounded_budget() {
    let samples = [-1.0, -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0];
    let plan = |execution| {
        Shgo::new(
            ShgoOptions::new(Budget::try_from(5).unwrap(), geometry()).with_execution(execution),
            BorrowedStartDesign::new(&samples, 2).unwrap(),
            PowellLocalFactory::default(),
        )
    };
    let f = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] * x[0] - 1.0).powi(2) + (x[1] * x[1] - 1.0).powi(2))
    });
    let bounds = boxed::<f64>(-2.0, 2.0, 2);
    let sequential = plan(ExecutionPolicy::Sequential)
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    let parallel = plan(ExecutionPolicy::Parallel(
        core::num::NonZeroUsize::new(2).unwrap(),
    ));
    let mut output = ShgoResult::empty();
    let mut work = parallel.workspace(2).unwrap();
    #[cfg(feature = "std")]
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(core::num::NonZeroUsize::new(2).unwrap());
    parallel
        .minimize_into(
            &BoundedObjective::new_sync(&f, &bounds).unwrap(),
            &mut output,
            &mut work,
            ExecutionContext::new(&pool),
        )
        .unwrap();
    assert_eq!(sequential.candidates(), output.candidates());
    assert_eq!(
        sequential.best_refined().unwrap().fun(),
        output.best_refined().unwrap().fun()
    );
    assert_eq!(sequential.counts(), output.counts());
}
