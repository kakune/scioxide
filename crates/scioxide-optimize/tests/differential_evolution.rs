//! Independent DE policy, replay, and failure tests.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AttemptOutcome, Bounds, Budget, ExecutionContext, ExecutionPolicy, ProblemToken, SciOxideError,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    BoundedObjective, DifferentialEvolution, DifferentialEvolutionBoundary as Boundary,
    DifferentialEvolutionOptions, ObjectiveFn, OptimizeResult, UpdatingMode,
};

fn parent() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        2,
        0,
    )
}

fn options(updating: UpdatingMode, boundary: Boundary) -> DifferentialEvolutionOptions<f64> {
    DifferentialEvolutionOptions::default()
        .with_population_size(Budget::try_from(12).unwrap())
        .with_max_generations(Budget::try_from(80).unwrap())
        .with_updating(updating)
        .with_boundary(boundary)
}

fn solver(updating: UpdatingMode, boundary: Boundary) -> DifferentialEvolution<f64> {
    DifferentialEvolution::new(options(updating, boundary)).unwrap()
}

fn five() -> Bounds<f64> {
    Bounds::new(vec![(Some(-5.), Some(5.)); 2]).unwrap()
}

#[test]
fn policies_find_analytic_minimum_and_report_complete_generations() {
    let bounds = five();
    for update in [UpdatingMode::Immediate, UpdatingMode::Deferred] {
        for boundary in [Boundary::Clip, Boundary::Reflect] {
            let solver = solver(update, boundary);
            let f = ObjectiveFn::new(2, |x: &[f64]| {
                assert!(x.iter().all(|v| (-5. ..=5.).contains(v)));
                Ok(x.iter().map(|v| v * v).sum())
            });
            let problem = BoundedObjective::new(&f, &bounds).unwrap();
            let mut work = solver.workspace(2).unwrap();
            let mut out = OptimizeResult::empty();
            solver
                .minimize_into(
                    &problem,
                    &[2., 3.],
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(out.x().iter().map(|v| v * v).sum::<f64>() < 1e-5);
            assert!(!out.is_success());
            assert_eq!(work.generations().len(), out.iterations());
            for (index, report) in work.generations().iter().enumerate() {
                assert_eq!(report.generation(), index + 1);
                assert_eq!(report.accepted() + report.rejected(), 12);
                assert!(report.best_fitness() <= report.worst_fitness());
            }
            let first = out.x().to_vec();
            solver
                .minimize_into(
                    &problem,
                    &[2., 3.],
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(first, out.x());
            assert_eq!(
                work.last_attempt().unwrap().counts(),
                out.counts(),
                "work is read from the attempt record"
            );
        }
    }
}

#[test]
fn sequential_policy_options_match_the_owned_form() {
    let solver = solver(UpdatingMode::Deferred, Boundary::Reflect);
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let expected = solver.minimize(&problem, &[2., 3.]).unwrap();
    let explicit = DifferentialEvolution::new(
        options(UpdatingMode::Deferred, Boundary::Reflect)
            .with_execution(ExecutionPolicy::Sequential),
    )
    .unwrap();
    let mut work = explicit.workspace(2).unwrap();
    let mut out = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
        .unwrap();
    explicit
        .minimize_into(&problem, &[2., 3.], &mut out, &mut work, &mut context)
        .unwrap();
    assert_eq!(out.x(), expected.x());
    assert_eq!(out.fun_evals(), expected.fun_evals());
    assert!(
        context
            .budget_report(scioxide_core::BudgetKind::Sample)
            .consumed()
            > 0
    );
}

#[test]
fn callback_failure_preserves_output_and_generations() {
    let solver = solver(UpdatingMode::Deferred, Boundary::Clip);
    let good = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let mut out = solver
        .minimize(&BoundedObjective::new(&good, &bounds).unwrap(), &[2., 3.])
        .unwrap();
    let before = out.x().to_vec();
    let calls = core::cell::Cell::new(0);
    let failing = ObjectiveFn::new(2, |_: &[f64]| {
        calls.set(calls.get() + 1);
        if calls.get() == 15 {
            Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Output,
                "injected",
            ))
        } else {
            Ok(calls.get() as f64)
        }
    });
    let mut work = solver.workspace(2).unwrap();
    assert!(
        solver
            .minimize_into(
                &BoundedObjective::new(&failing, &bounds).unwrap(),
                &[2., 3.],
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(out.x(), before);
    assert!(work.generations().is_empty());
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
}

#[test]
fn deferred_reflection_supports_f32() {
    let options = DifferentialEvolutionOptions::default()
        .with_updating(UpdatingMode::Deferred)
        .with_boundary(Boundary::Reflect);
    let f = ObjectiveFn::new(2, |x: &[f32]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = Bounds::new(vec![(Some(-5_f32), Some(5_f32)); 2]).unwrap();
    let out = DifferentialEvolution::new(options)
        .unwrap()
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap(), &[2., 3.])
        .unwrap();
    assert!(out.x().iter().map(|v| v * v).sum::<f32>() < 1e-4);
}

#[test]
fn f32_all_policies_meet_explicit_objective_accuracy() {
    let bounds = Bounds::new(vec![(Some(-5_f32), Some(5_f32)); 2]).unwrap();
    for update in [UpdatingMode::Immediate, UpdatingMode::Deferred] {
        for boundary in [Boundary::Clip, Boundary::Reflect] {
            let options = DifferentialEvolutionOptions::default()
                .with_population_size(Budget::try_from(12).unwrap())
                .with_max_generations(Budget::try_from(80).unwrap())
                .with_fitness_tolerance(
                    scioxide_core::ResidualTolerance::new(1e-10, 1e-10).unwrap(),
                )
                .with_updating(update)
                .with_boundary(boundary);
            let f = ObjectiveFn::new(2, |x: &[f32]| Ok(x.iter().map(|v| v * v).sum()));
            let output = DifferentialEvolution::new(options)
                .unwrap()
                .minimize(&BoundedObjective::new(&f, &bounds).unwrap(), &[2., 3.])
                .unwrap();
            assert!(output.x().iter().map(|v| v * v).sum::<f32>() < 1e-5);
        }
    }
}

#[test]
fn prepared_execution_allocates_nothing() {
    let solver = solver(UpdatingMode::Deferred, Boundary::Reflect);
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    {
        solver
            .minimize_into(
                &problem,
                &[2., 3.],
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
    };
}

#[cfg(feature = "std")]
#[test]
fn concurrent_executor_replays_deferred_trials() {
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let expected = solver(UpdatingMode::Deferred, Boundary::Reflect)
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap(), &[2., 3.])
        .unwrap();
    let solver = DifferentialEvolution::new(
        options(UpdatingMode::Deferred, Boundary::Reflect).with_execution(ExecutionPolicy::Auto),
    )
    .unwrap();
    let mut output = OptimizeResult::empty();
    let mut work = solver.workspace(2).unwrap();
    solver
        .minimize_into(
            &BoundedObjective::new_sync(&f, &bounds).unwrap(),
            &[2., 3.],
            &mut output,
            &mut work,
            ExecutionContext::new(&pool),
        )
        .unwrap();
    assert_eq!(expected.x(), output.x());
    assert_eq!(expected.fun_evals(), output.fun_evals());
}

#[test]
fn sample_limits_never_overdraw_and_keep_valid_partial_results() {
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    for limit in [1, 21, 22, 23, 80] {
        let solver = solver(UpdatingMode::Deferred, Boundary::Reflect);
        let mut output = OptimizeResult::empty();
        let mut work = solver.workspace(2).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Sample,
            Budget::try_from(limit).unwrap(),
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(), &mut budgets)
            .unwrap();
        solver
            .minimize_into(&problem, &[2., 3.], &mut output, &mut work, &mut context)
            .unwrap();
        assert_eq!(
            context
                .budget_report(scioxide_core::BudgetKind::Sample)
                .consumed(),
            limit
        );
        assert_eq!(
            output.termination(),
            scioxide_optimize::OptimizeTermination::SampleLimit
        );
        assert!(output.fun() <= 13.);
        assert!(work.generations().is_empty());
        assert_eq!(
            work.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    }
}

struct FailingExecutor;

#[test]
fn unaffordable_deferred_batch_does_not_charge_unused_evaluations() {
    let solver = DifferentialEvolution::new(
        options(UpdatingMode::Deferred, Boundary::Reflect).with_execution(ExecutionPolicy::Auto),
    )
    .unwrap();
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let problem = BoundedObjective::new_sync(&f, &bounds).unwrap();
    let mut output = OptimizeResult::empty();
    let mut work = solver.workspace(2).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(
        scioxide_core::BudgetKind::Evaluation,
        Budget::try_from(13).unwrap(),
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::new(&FailingExecutor),
            parent(),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(&problem, &[2., 3.], &mut output, &mut work, &mut context)
        .unwrap();
    assert_eq!(
        output.termination(),
        scioxide_optimize::OptimizeTermination::EvaluationLimit
    );
    assert_eq!(output.fun_evals(), 12);
    assert_eq!(
        context
            .budget_report(scioxide_core::BudgetKind::Evaluation)
            .remaining(),
        Some(1)
    );
    assert!(work.generations().is_empty());
}

impl scioxide_core::Executor for FailingExecutor {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }
    fn execute(
        &self,
        _: ExecutionContext<'_>,
        _: scioxide_core::ExecutionPlan,
        _: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        Err(SciOxideError::execution_failure(
            None,
            scioxide_core::ExecutionFailureReason::ProviderFailure,
            "injected executor failure",
        ))
    }
}

#[test]
fn executor_failure_and_invalid_mode_do_not_publish_output() {
    assert!(
        DifferentialEvolution::new(
            options(UpdatingMode::Immediate, Boundary::Reflect)
                .with_execution(ExecutionPolicy::Auto)
        )
        .is_err()
    );
    let solver = DifferentialEvolution::new(
        options(UpdatingMode::Deferred, Boundary::Reflect).with_execution(ExecutionPolicy::Auto),
    )
    .unwrap();
    let calls = core::sync::atomic::AtomicUsize::new(0);
    let f = ObjectiveFn::new(2, |x: &[f64]| {
        calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(x.iter().map(|v| v * v).sum())
    });
    let bounds = five();
    let mut output = OptimizeResult::empty();
    let mut work = solver.workspace(2).unwrap();
    // A partitioned policy needs a thread-safe problem before any callback.
    assert!(
        solver
            .minimize_into(
                &BoundedObjective::new(&f, &bounds).unwrap(),
                &[2., 3.],
                &mut output,
                &mut work,
                ExecutionContext::new(&FailingExecutor),
            )
            .is_err()
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(
        solver
            .minimize_into(
                &BoundedObjective::new_sync(&f, &bounds).unwrap(),
                &[2., 3.],
                &mut output,
                &mut work,
                ExecutionContext::new(&FailingExecutor),
            )
            .is_err()
    );
    assert!(output.x().is_empty());
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 12);
    assert!(work.generations().is_empty());
}

#[test]
fn seeds_select_distinct_replayable_streams() {
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = five();
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let run = |seed| {
        DifferentialEvolution::new(
            options(UpdatingMode::Immediate, Boundary::Clip)
                .with_max_generations(Budget::try_from(3).unwrap())
                .with_seed(seed),
        )
        .unwrap()
        .minimize(&problem, &[2., 3.])
        .unwrap()
        .x()
        .to_vec()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}
