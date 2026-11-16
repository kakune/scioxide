#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Bounds, Budget, BudgetKind, ExecutionContext, ExecutionPolicy, ProblemToken,
    RealScalar, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    BoundedObjective, Direct, DirectOptions, DirectResult, DirectVariant, ObjectiveFn,
    OptimizeTermination,
};

fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}

fn plan<S: RealScalar + scioxide_core::NumericScalar<Real = S>>(
    variant: DirectVariant,
    evaluations: usize,
) -> Direct<S> {
    Direct::new(
        DirectOptions::new(
            budget(evaluations),
            budget(evaluations * 2 + 1),
            budget(10_000_000),
        )
        .with_variant(variant),
    )
    .unwrap()
}

fn multimodal<S: RealScalar + scioxide_core::NumericScalar<Real = S>>() {
    let n = |v| S::try_from_f64_lossy(v).unwrap();
    let f = ObjectiveFn::new(2, |x: &[S]| {
        let a = x[0] * x[0] - S::ONE;
        let b = x[1] * x[1] - S::ONE;
        Ok(a * a + b * b)
    });
    let bounds = Bounds::new(vec![(Some(n(-2.0)), Some(n(2.0))); 2]).unwrap();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    for variant in [DirectVariant::Original, DirectVariant::LocallyBiased] {
        let solver = plan(variant, 300);
        let result = solver.minimize(&problem).unwrap();
        let (x, value) = result.best_sample().unwrap();
        assert!(value.into_f64() < 0.02, "{variant:?}: {}", value.into_f64());
        assert!(x.iter().all(|&v| (v.into_f64().abs() - 1.0).abs() < 0.15));
        assert!(result.evaluations() <= 300);
        assert!(result.rectangles() <= 601);
        assert!(!result.termination().is_success());
    }
}

#[test]
fn bounded_multimodal_both_precisions() {
    multimodal::<f32>();
    multimodal::<f64>();
}

#[test]
fn first_two_dimensional_split_matches_direct_stencil() {
    let solver = Direct::new(
        DirectOptions::new(budget(10), budget(20), budget(100_000)).with_max_sweeps(Budget::ONE),
    )
    .unwrap();
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] + 2.0 * x[1]));
    let bounds = Bounds::new(vec![(Some(0.0_f64), Some(1.0)); 2]).unwrap();
    let result = solver
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.evaluations(), 5);
    assert_eq!(result.rectangles(), 5);
    assert_eq!(result.sweeps(), 1);
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    let (x, value) = result.best_sample().unwrap();
    assert_eq!(x[0], 0.5);
    assert!((x[1] - 1.0 / 6.0).abs() < 1e-14);
    assert!((value - 5.0 / 6.0).abs() < 1e-14);
}

#[test]
fn fixed_and_representationally_stalled_coordinates() {
    let solver = plan(DirectVariant::Original, 12);
    let f = ObjectiveFn::new(2, |x: &[f64]| {
        assert_eq!(x[0], 2.0);
        Ok(x[1])
    });
    let bounds = Bounds::new(vec![(Some(2.0_f64), Some(2.0)), (Some(0.0), Some(1.0))]).unwrap();
    let result = solver
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert!(result.best_sample().unwrap().1 < 0.2);
    assert!(result.evaluations() <= 12);

    let singleton = plan(DirectVariant::Original, 8);
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0]));
    let fixed = Bounds::new(vec![(Some(2.0_f64), Some(2.0))]).unwrap();
    let result = singleton
        .minimize(&BoundedObjective::new(&f, &fixed).unwrap())
        .unwrap();
    assert_eq!(result.evaluations(), 1);
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );

    let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
    let narrow = plan(DirectVariant::Original, 8);
    let bounds = Bounds::new(vec![(Some(1.0_f64), Some(adjacent))]).unwrap();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let result = narrow.minimize(&problem).unwrap();
    assert_eq!(result.evaluations(), 1);
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert_eq!(result.stalled_rectangles(), 1);
    let mut output = DirectResult::empty();
    let mut work = narrow.workspace(1).unwrap();
    for _ in 0..2 {
        narrow
            .minimize_into(
                &problem,
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
    }
    assert_eq!(output.stalled_rectangles(), 1);
}

#[test]
fn declared_and_shared_limits_preserve_finite_partial_result() {
    let bounds = Bounds::new(vec![(Some(0.0_f64), Some(1.0))]).unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 0.8).powi(2)));
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let options = DirectOptions::new(budget(30), Budget::ONE, budget(10_000));
    let solver = Direct::new(options).unwrap();
    let result = solver.minimize(&problem).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::MemoryLimit);
    assert_eq!(result.evaluations(), 1);

    let solver = plan(DirectVariant::Original, 30);
    let mut output = DirectResult::empty();
    let mut workspace = solver.workspace(1).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(8),
                scioxide_core::algorithm_id!("test", "Parent", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(&problem, &mut output, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(output.evaluations(), 1);
    assert_eq!(output.best_sample().unwrap().0, &[0.5]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert_eq!(workspace.last_accepted().unwrap().point(), &[0.5]);
}

#[test]
fn invalid_bounds_and_provider_error_are_structured_and_transactional() {
    let solver = plan(DirectVariant::Original, 30);
    let okay = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0]));
    let unbounded = Bounds::new(vec![(None::<f64>, Some(1.0))]).unwrap();
    assert!(
        solver
            .minimize(&BoundedObjective::new(&okay, &unbounded).unwrap())
            .is_err()
    );
    let bounds = Bounds::new(vec![(Some(0.0_f64), Some(1.0))]).unwrap();
    let failure = ObjectiveFn::new(1, |_x: &[f64]| Ok(f64::NAN));
    let mut output = DirectResult::empty();
    let mut workspace = solver.workspace(1).unwrap();
    solver
        .minimize_into(
            &BoundedObjective::new(&okay, &bounds).unwrap(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let previous = output.best_sample().unwrap().1;
    assert!(
        solver
            .minimize_into(
                &BoundedObjective::new(&failure, &bounds).unwrap(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(output.best_sample().unwrap().1, previous);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
}

#[test]
fn prepared_replay_is_sequential_and_rejects_parallel_policies() {
    let solver = plan(DirectVariant::LocallyBiased, 40);
    let f = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 0.3).powi(2) + (x[1] + 0.4).powi(2))
    });
    let bounds = Bounds::new(vec![(Some(-1.0_f64), Some(1.0)); 2]).unwrap();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut output = DirectResult::empty();
    let mut workspace = solver.workspace(2).unwrap();
    solver
        .minimize_into(
            &problem,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let first = output.best_sample().unwrap().0.to_vec();
    let count = output.evaluations();
    solver
        .minimize_into(
            &problem,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.best_sample().unwrap().0, first);
    assert_eq!(output.evaluations(), count);
    assert_eq!(
        solver.minimize(&problem).unwrap().best_sample().unwrap().0,
        first
    );

    for policy in [
        ExecutionPolicy::Auto,
        ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap()),
    ] {
        assert!(
            Direct::<f64>::new(
                DirectOptions::new(budget(40), budget(81), budget(10_000_000))
                    .with_execution(policy)
            )
            .is_err()
        );
    }
}

#[test]
fn storage_cap_below_requirement_is_a_resource_limit() {
    let solver =
        Direct::<f64>::new(DirectOptions::new(budget(10), budget(20), budget(64))).unwrap();
    assert_eq!(
        solver.workspace(2).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn quality_per_evaluation_beats_equal_budget_uniform_grid() {
    let target = 0.821_f64;
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - target).powi(2)));
    let bounds = Bounds::new(vec![(Some(0.0_f64), Some(1.0))]).unwrap();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let uniform = (0..21)
        .map(|i| ((i as f64) / 20.0 - target).powi(2))
        .fold(f64::INFINITY, f64::min);
    for variant in [DirectVariant::Original, DirectVariant::LocallyBiased] {
        let solver = plan(variant, 21);
        let result = solver.minimize(&problem).unwrap();
        assert!(result.best_sample().unwrap().1 < uniform, "{variant:?}");
        assert!(result.evaluations() <= 21);
    }
}

#[test]
fn himmelblau_multimodal_box_finds_low_value() {
    let f = ObjectiveFn::new(2, |x: &[f64]| {
        let a = x[0] * x[0] + x[1] - 11.0;
        let b = x[0] + x[1] * x[1] - 7.0;
        Ok(a * a + b * b)
    });
    let bounds = Bounds::new(vec![(Some(-5.0_f64), Some(5.0)); 2]).unwrap();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    for variant in [DirectVariant::Original, DirectVariant::LocallyBiased] {
        let solver = plan(variant, 700);
        let result = solver.minimize(&problem).unwrap();
        assert!(result.best_sample().unwrap().1 < 0.5, "{variant:?}");
        assert!(result.evaluations() <= 700);
    }
}
