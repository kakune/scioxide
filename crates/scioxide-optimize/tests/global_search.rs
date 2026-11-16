#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{
    Bounds, Budget, BudgetKind, Capacity, ExecutionContext, ProblemToken, RealScalar, SolveBudgets,
    SolveIdentity, SolveState,
};
use scioxide_optimize::{
    BorrowedStartDesign, BoundedObjective, GlobalSearchOptions, GlobalSearchResult, MultiStart,
    ObjectiveFn, OptimizeTermination, PowellLocalFactory, UniformStartDesign,
};

fn parent(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        0,
        0,
    )
}

fn boxed<S: RealScalar>(lower: f64, upper: f64, dimension: usize) -> Bounds<S> {
    let num = |x| S::try_from_f64_lossy(x).unwrap();
    Bounds::new(vec![(Some(num(lower)), Some(num(upper))); dimension]).unwrap()
}

fn quadratic<S: RealScalar>() {
    let num = |x| S::try_from_f64_lossy(x).unwrap();
    let bounds = boxed::<S>(-4.0, 4.0, 2);
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(4).unwrap());
    let plan = MultiStart::new(options, UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let f = ObjectiveFn::new(2, |x: &[S]| {
        Ok(x.iter()
            .map(|&v| (v - num(0.5)) * (v - num(0.5)))
            .fold(S::ZERO, |a, b| a + b))
    });
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut work = plan.workspace(2).unwrap();
    let mut result = GlobalSearchResult::empty();
    plan.minimize_into(
        &problem,
        &mut result,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(result.starts().len(), 4);
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert_eq!(result.failures(), 0);
    assert!(result.best().unwrap().fun() < num(1e-4));
    let starts: Vec<_> = result
        .starts()
        .iter()
        .map(|r| (r.seed(), r.start().to_vec()))
        .collect();
    let counts = result.counts();
    plan.minimize_into(
        &problem,
        &mut result,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(counts, result.counts());
    assert_eq!(
        starts,
        result
            .starts()
            .iter()
            .map(|r| (r.seed(), r.start().to_vec()))
            .collect::<Vec<_>>()
    );
}
#[test]
fn prepared_replay_both_precisions() {
    quadratic::<f32>();
    quadratic::<f64>();
}

#[test]
fn borrowed_design_keeps_boundary_starts() {
    let bounds = boxed::<f64>(0.0, 1.0, 1);
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(2).unwrap());
    let plan = MultiStart::new(
        options,
        BorrowedStartDesign::new(&[0.0, 1.0], 1).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let result = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.starts()[0].start(), &[0.0]);
    assert_eq!(result.starts()[1].start(), &[1.0]);
    assert!(result.best().unwrap().fun() < 1e-10);
}

#[test]
fn child_budget_does_not_stop_restarts() {
    let options = GlobalSearchOptions::default()
        .with_starts(Budget::try_from(3).unwrap())
        .with_local_budgets(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE),
        );
    let plan = MultiStart::new(options, UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let result = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.starts().len(), 3);
    assert_eq!(
        result.counts().total().unwrap().total_providers().unwrap(),
        3
    );
    assert_eq!(result.successes(), 0);
}

fn multimodal<S: RealScalar>() {
    use scioxide_optimize::{
        BasinHopping, BasinHoppingOptions, DualAnnealing, DualAnnealingOptions,
        UniformBasinProposal,
    };
    let num = |x| S::try_from_f64_lossy(x).unwrap();
    // Nonnegative double well: two analytically known global minima at +/-1.
    let f = ObjectiveFn::new(1, |x: &[S]| {
        let y = x[0] * x[0] - S::ONE;
        Ok(y * y)
    });
    let bounds = boxed::<S>(-3.0, 3.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(24).unwrap());
    let basin = BasinHopping::new(
        BasinHoppingOptions::new(options.clone(), num(0.1)).unwrap(),
        UniformBasinProposal::new(num(0.5)).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let result = basin.minimize(&problem, &[num(2.0)]).unwrap();
    assert_eq!(result.starts().len(), 24);
    assert!(result.best().unwrap().fun() < num(1e-5), "{result:?}");
    assert!(result.successes() > 0);
    let anneal = DualAnnealing::new(
        DualAnnealingOptions::new(options),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let result = anneal.minimize(&problem, &[num(2.0)]).unwrap();
    assert_eq!(result.starts().len(), 24);
    assert!(result.best().unwrap().fun() < num(1e-5), "{result:?}");
    assert_eq!(
        result.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
}
#[test]
fn multimodal_chains_both_precisions() {
    multimodal::<f32>();
    multimodal::<f64>();
}

#[test]
fn prepared_paths_execute() {
    use scioxide_optimize::{
        BasinHopping, BasinHoppingOptions, DualAnnealing, DualAnnealingOptions,
        UniformBasinProposal,
    };
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(4).unwrap());
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let bounds = boxed::<f64>(-3.0, 3.0, 2);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let sequential = ExecutionContext::sequential;
    let multi = MultiStart::new(
        options.clone(),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut work = multi.workspace(2).unwrap();
    let mut out = GlobalSearchResult::empty();
    // The first publish reserves caller output; the prepared call that follows allocates nothing.
    multi
        .minimize_into(&problem, &mut out, &mut work, sequential())
        .unwrap();
    multi
        .minimize_into(&problem, &mut out, &mut work, sequential())
        .unwrap();
    let basin = BasinHopping::new(
        BasinHoppingOptions::new(options.clone(), 1.0).unwrap(),
        UniformBasinProposal::new(0.2).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut work = basin.workspace(2).unwrap();
    basin
        .minimize_into(&problem, &[1.0, 2.0], &mut out, &mut work, sequential())
        .unwrap();
    let anneal = DualAnnealing::new(
        DualAnnealingOptions::new(options),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut work = anneal.workspace(2).unwrap();
    anneal
        .minimize_into(&problem, &[1.0, 2.0], &mut out, &mut work, sequential())
        .unwrap();
}

#[test]
fn parent_budget_retains_first_observation() {
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(3).unwrap());
    let multi = MultiStart::new(
        options,
        BorrowedStartDesign::new(&[2.0, 1.0, 0.0], 1).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let calls = core::cell::Cell::new(0);
    let f = ObjectiveFn::new(1, |x: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(x[0] * x[0])
    });
    let bounds = boxed::<f64>(-3.0, 3.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut work = multi.workspace(1).unwrap();
    let mut out = GlobalSearchResult::empty();
    let mut state = SolveState::new();
    let mut budget = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), parent(1), &mut budget)
        .unwrap();
    multi
        .minimize_into(&problem, &mut out, &mut work, &mut ctx)
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(out.best().unwrap().x(), &[2.0]);
    assert_eq!(out.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(out.counts().total().unwrap(), ctx.counts().total().unwrap());
}

#[test]
fn qmc_starts_replay_exact_indices() {
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(3).unwrap());
    let plan = MultiStart::new(
        options,
        scioxide_random::QmcPlan::halton(1).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(0.0, 1.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    for _ in 0..2 {
        let result = plan.minimize(&problem).unwrap();
        assert_eq!(
            result
                .starts()
                .iter()
                .map(|r| r.start()[0])
                .collect::<Vec<_>>(),
            vec![0.0, 0.5, 0.25]
        );
    }
}

#[derive(Debug)]
struct SometimesFails;
struct FailingLocal {
    inner: scioxide_optimize::PowellSearchLocal<f64>,
}
impl scioxide_optimize::LocalSolverFactory<f64> for SometimesFails {
    type Solver = FailingLocal;
    fn prepare(&self, dimension: usize) -> Result<Self::Solver, scioxide_core::SciOxideError> {
        Ok(FailingLocal {
            inner: scioxide_optimize::LocalSolverFactory::prepare(
                &PowellLocalFactory::default(),
                dimension,
            )?,
        })
    }
}
impl scioxide_optimize::SearchLocalSolver<f64> for FailingLocal {
    fn minimize_into(
        &mut self,
        problem: &BoundedObjective<'_, f64>,
        x: &[f64],
        seed: u64,
        out: &mut scioxide_optimize::OptimizeResult<f64>,
        ctx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        if x[0] < 0.0 {
            return Err(scioxide_core::SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::IterationLimit,
                scioxide_core::ConvergenceDiagnostics::new(),
                "fixture",
            ));
        }
        self.inner.minimize_into(problem, x, seed, out, ctx)
    }
}
#[test]
fn individual_failures_and_all_failed_search_are_truthful() {
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(3).unwrap());
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(-3.0, 3.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    for (points, failures) in [([-2.0, 1.0, -1.0], 2), ([-2.0, -1.0, -0.5], 3)] {
        let plan = MultiStart::new(
            options.clone(),
            BorrowedStartDesign::new(&points, 1).unwrap(),
            SometimesFails,
        )
        .unwrap();
        let result = plan.minimize(&problem).unwrap();
        assert_eq!(result.failures(), failures);
        assert_eq!(result.starts().len(), 3);
        assert_eq!(result.best().is_none(), failures == 3);
        assert_eq!(
            result.termination(),
            OptimizeTermination::SearchBudgetCompleted
        );
    }
}

#[test]
#[cfg(feature = "std")]
fn executor_replay_conserves_finite_ancestor_budget() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::ExecutionPolicy;
    let mut expected = None;
    let bounds = boxed::<f64>(-3.0, 3.0, 1);
    for workers in [1, 2, 4] {
        let count = core::num::NonZeroUsize::new(workers).unwrap();
        let plan = MultiStart::new(
            GlobalSearchOptions::default()
                .with_starts(Budget::try_from(4).unwrap())
                .with_execution(ExecutionPolicy::Parallel(count)),
            UniformStartDesign,
            PowellLocalFactory::default(),
        )
        .unwrap();
        let pool = scioxide_core::ThreadExecutor::new(count);
        let calls = AtomicUsize::new(0);
        let f = ObjectiveFn::new(1, |x: &[f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(x[0] * x[0])
        });
        let problem = BoundedObjective::new_sync(&f, &bounds).unwrap();
        let mut work = plan.workspace(1).unwrap();
        let mut out = GlobalSearchResult::empty();
        let mut state = SolveState::new();
        let mut budget = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(40).unwrap());
        let mut ctx = state
            .begin_solve(ExecutionContext::new(&pool), parent(7), &mut budget)
            .unwrap();
        plan.minimize_into(&problem, &mut out, &mut work, &mut ctx)
            .unwrap();
        assert!(calls.load(Ordering::Relaxed) <= 40);
        assert_eq!(
            out.counts().total().unwrap().total_providers().unwrap(),
            calls.load(Ordering::Relaxed)
        );
        assert_eq!(ctx.counts().total().unwrap(), out.counts().total().unwrap());
        let values = out
            .starts()
            .iter()
            .map(|r| (r.seed(), r.candidate().map(|c| c.fun()), r.counts()))
            .collect::<Vec<_>>();
        if let Some(expected) = &expected {
            assert_eq!(&values, expected);
        } else {
            expected = Some(values);
        }
    }
}

#[test]
fn partitioned_policy_requires_a_thread_safe_problem() {
    let plan = MultiStart::new(
        GlobalSearchOptions::default()
            .with_starts(Budget::try_from(2).unwrap())
            .with_execution(scioxide_core::ExecutionPolicy::Auto),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(-1.0, 1.0, 1);
    let error = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    let result = plan
        .minimize(&BoundedObjective::new_sync(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.starts().len(), 2);
}

#[test]
fn seeds_replay_and_errors_and_sample_stops_are_transactional() {
    use scioxide_core::{AttemptOutcome, SciOxideError};
    let plan_for = |seed| {
        MultiStart::new(
            GlobalSearchOptions::default()
                .with_starts(Budget::try_from(3).unwrap())
                .with_seed(seed),
            UniformStartDesign,
            PowellLocalFactory::default(),
        )
        .unwrap()
    };
    let plan = plan_for(27);
    let bounds = boxed::<f64>(-2.0, 2.0, 2);
    let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let starts = |result: &GlobalSearchResult<f64>| {
        result
            .starts()
            .iter()
            .map(|r| r.start().to_vec())
            .collect::<Vec<_>>()
    };
    let mut work = plan.workspace(2).unwrap();
    let mut out = GlobalSearchResult::empty();
    plan.minimize_into(
        &problem,
        &mut out,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let first = starts(&out);
    assert_eq!(first, starts(&plan_for(27).minimize(&problem).unwrap()));
    assert_ne!(first, starts(&plan_for(28).minimize(&problem).unwrap()));
    let saved = out.best().unwrap().x().to_vec();

    let failing = ObjectiveFn::new(2, |_: &[f64]| {
        Err(SciOxideError::execution_failure(
            None,
            scioxide_core::ExecutionFailureReason::ProtocolViolation,
            "objective failed",
        ))
    });
    let failing = BoundedObjective::new(&failing, &bounds).unwrap();
    let mut state = SolveState::new();
    let mut budget = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), parent(3), &mut budget)
        .unwrap();
    assert!(
        plan.minimize_into(&failing, &mut out, &mut work, &mut ctx)
            .is_err()
    );
    assert_eq!(out.best().unwrap().x(), saved);
    assert_eq!(starts(&out), first);
    assert_eq!(ctx.counts().total().unwrap().total_providers().unwrap(), 1);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );

    // A sample allowance of one admits the first draw and refuses the second,
    // so the search stops before any start is complete.
    let mut budget = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, Budget::ONE);
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), parent(3), &mut budget)
        .unwrap();
    plan.minimize_into(&problem, &mut out, &mut work, &mut ctx)
        .unwrap();
    assert_eq!(out.termination(), OptimizeTermination::SampleLimit);
    assert!(out.best().is_none());
    assert_eq!(ctx.budget_report(BudgetKind::Sample).consumed(), 1);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert!(work.last_accepted().is_none());
}

#[test]
fn annealing_restarts_and_disabled_refinement_are_reported() {
    use scioxide_optimize::{DualAnnealing, DualAnnealingOptions};
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(8).unwrap());
    let anneal = DualAnnealingOptions::new(options)
        .with_schedule(1.0, 2.0, -5.0, 0.9)
        .with_local_search(false);
    let plan = DualAnnealing::new(anneal, PowellLocalFactory::default()).unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let result = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap(), &[0.0])
        .unwrap();
    assert!(result.starts().iter().any(|r| r.reannealed()));
    assert!(result.starts().iter().all(|r| r.local_result().is_none()));
    for start in result.starts() {
        let raw = start.candidate().unwrap();
        assert_eq!(
            raw.counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Objective),
            1
        );
        assert_eq!(raw.algorithm().unwrap().method(), "DualAnnealing");
        assert_eq!(raw.trust(), scioxide_core::PartialResultTrust::Admissible);
    }
    assert_eq!(
        result.counts().total().unwrap().total_providers().unwrap(),
        8
    );
    assert!(result.best().unwrap().fun().abs() <= 1e-12);
    assert_eq!(result.successes(), 0);
    assert!(result.best_successful().is_none());
}

#[test]
fn basin_clustering_keeps_distinct_analytic_minima() {
    let options = GlobalSearchOptions::default()
        .with_starts(Budget::try_from(4).unwrap())
        .with_basin_tolerances(1e-4, 1e-8);
    let plan = MultiStart::new(
        options,
        BorrowedStartDesign::new(&[-1.0, -1.0, 1.0, 1.0], 1).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] * x[0] - 1.0).powi(2)));
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let result = plan
        .minimize(&BoundedObjective::new(&f, &bounds).unwrap())
        .unwrap();
    assert_eq!(result.successes(), 4);
    assert_eq!(result.basins(), 2, "{result:?}");
    assert!(
        result
            .starts()
            .iter()
            .all(|r| r.local_result().unwrap().fun() < 1e-8)
    );
    let invalid = GlobalSearchOptions::<f64>::default().with_basin_tolerances(0.0, 1e-8);
    assert!(MultiStart::new(invalid, UniformStartDesign, PowellLocalFactory::default()).is_err());
}

#[test]
fn cooperative_clock_preserves_observation() {
    use core::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};
    struct ClockFixture(AtomicUsize);
    impl Clock for ClockFixture {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(self.0.load(Ordering::Relaxed) as u64),
            ))
        }
    }
    let clock = ClockFixture(AtomicUsize::new(0));
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let bounds = boxed::<f64>(-3.0, 3.0, 1);
    let options = GlobalSearchOptions::default().with_time_limit(Duration::from_secs(1));
    let plan = MultiStart::new(options, UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| {
        clock.0.store(2, Ordering::Relaxed);
        Ok(x[0] * x[0])
    });
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut work = plan.workspace(1).unwrap();
    let mut out = GlobalSearchResult::empty();
    // A time limit needs a clock; the owned form has none and rejects before any callback.
    assert_eq!(
        plan.minimize(&problem).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::CapabilityUnavailable
    );
    plan.minimize_into(
        &problem,
        &mut out,
        &mut work,
        ExecutionContext::sequential()
            .with_clock(&clock_binding)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(out.termination(), OptimizeTermination::TimeLimit);
    assert_eq!(out.counts().total().unwrap().total_providers().unwrap(), 1);
    assert!(out.best().is_some());
    // A fresh deadline on each child must not shorten the unbounded parent's search.
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(3).unwrap());
    let plan = MultiStart::new(
        options,
        UniformStartDesign,
        PowellLocalFactory::new(
            scioxide_optimize::PowellOptions::default().with_time_limit(Duration::from_secs(1)),
        ),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| {
        clock.0.fetch_add(2, Ordering::Relaxed);
        Ok(x[0] * x[0])
    });
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    let mut out = GlobalSearchResult::empty();
    let mut work = plan.workspace(1).unwrap();
    plan.minimize_into(
        &problem,
        &mut out,
        &mut work,
        ExecutionContext::sequential()
            .with_clock(&clock_binding)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(out.starts().len(), 3);
    assert_eq!(
        out.termination(),
        OptimizeTermination::SearchBudgetCompleted
    );
    assert!(
        out.starts()
            .iter()
            .all(|r| r.local_result().unwrap().termination() == OptimizeTermination::TimeLimit)
    );
}

#[test]
fn executor_error_after_work_preserves_output_and_counts() {
    use scioxide_core::{
        ExecutionPlan, ExecutionPolicy, Executor, PartitionFn, SciOxideError, SequentialExecutor,
    };
    struct FailAfter;
    impl Executor for FailAfter {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::MIN
        }
        fn execute(
            &self,
            ctx: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            SequentialExecutor.execute(ctx, plan, operation)?;
            Err(SciOxideError::execution_failure(
                None,
                scioxide_core::ExecutionFailureReason::ProtocolViolation,
                "failed after work",
            ))
        }
    }
    let options = GlobalSearchOptions::default()
        .with_starts(Budget::try_from(2).unwrap())
        .with_execution(ExecutionPolicy::Auto);
    let plan = MultiStart::new(options, UniformStartDesign, PowellLocalFactory::default()).unwrap();
    let calls = core::sync::atomic::AtomicUsize::new(0);
    let f = ObjectiveFn::new(1, |x: &[f64]| {
        calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(x[0] * x[0])
    });
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let problem = BoundedObjective::new_sync(&f, &bounds).unwrap();
    let mut work = plan.workspace(1).unwrap();
    let mut out = plan.minimize(&problem).unwrap();
    let previous = out.counts();
    calls.store(0, core::sync::atomic::Ordering::Relaxed);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::new(&FailAfter), parent(1), &mut budgets)
        .unwrap();
    assert!(
        plan.minimize_into(&problem, &mut out, &mut work, &mut ctx)
            .is_err()
    );
    assert_eq!(out.counts(), previous);
    assert_eq!(
        ctx.counts().total().unwrap().total_providers().unwrap(),
        calls.load(core::sync::atomic::Ordering::Relaxed)
    );
    assert!(calls.load(core::sync::atomic::Ordering::Relaxed) > 0);
}

#[test]
fn changed_factory_prepares_its_own_local_workspace() {
    let options = GlobalSearchOptions::default().with_starts(Budget::ONE);
    let original = MultiStart::new(
        options.clone(),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )
    .unwrap();
    let mut work = original.workspace(1).unwrap();
    let mut out = GlobalSearchResult::empty();
    let changed = MultiStart::new(
        options,
        UniformStartDesign,
        PowellLocalFactory::new(
            scioxide_optimize::PowellOptions::default().with_max_evaluations(Budget::ONE),
        ),
    )
    .unwrap();
    assert_eq!(
        work.check(&changed.workspace_requirements(1).unwrap())
            .unwrap(),
        Capacity::NeedsGrowth
    );
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    let bounds = boxed::<f64>(-2.0, 2.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    changed
        .minimize_into(
            &problem,
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        work.check(&changed.workspace_requirements(1).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
    assert_eq!(
        work.check(&original.workspace_requirements(1).unwrap())
            .unwrap(),
        Capacity::NeedsGrowth
    );
    let local = out.starts()[0].local_result().unwrap();
    assert_eq!(local.fun_evals(), 1);
    assert_eq!(local.termination(), OptimizeTermination::EvaluationLimit);
}

#[test]
fn unbounded_or_degenerate_boxes_and_outside_starts_are_rejected() {
    use scioxide_optimize::{BasinHopping, BasinHoppingOptions, UniformBasinProposal};
    let plan = MultiStart::new(
        GlobalSearchOptions::default(),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )
    .unwrap();
    let f = ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
    for bounds in [
        Bounds::new(vec![(None, Some(1.0))]).unwrap(),
        Bounds::new(vec![(Some(1.0), Some(1.0))]).unwrap(),
    ] {
        assert_eq!(
            plan.minimize(&BoundedObjective::new(&f, &bounds).unwrap())
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }
    let basin = BasinHopping::new(
        BasinHoppingOptions::new(GlobalSearchOptions::default(), 1.0).unwrap(),
        UniformBasinProposal::new(0.2).unwrap(),
        PowellLocalFactory::default(),
    )
    .unwrap();
    let bounds = boxed::<f64>(-1.0, 1.0, 1);
    let problem = BoundedObjective::new(&f, &bounds).unwrap();
    assert!(basin.minimize(&problem, &[2.0]).is_err());
    assert!(
        BasinHopping::new(
            BasinHoppingOptions::new(
                GlobalSearchOptions::default().with_execution(scioxide_core::ExecutionPolicy::Auto),
                1.0
            )
            .unwrap(),
            UniformBasinProposal::new(0.2).unwrap(),
            PowellLocalFactory::default(),
        )
        .is_err()
    );
}

#[test]
fn annealing_schedule_is_validated_when_the_method_is_created() {
    use scioxide_optimize::{DualAnnealing, DualAnnealingOptions};
    for schedule in [
        [0.0, 2.62, -5.0, 2e-5],
        [-1.0, 2.62, -5.0, 2e-5],
        [5230.0, 1.0, -5.0, 2e-5],
        [5230.0, 3.0, -5.0, 2e-5],
        [5230.0, 2.62, 1.0, 2e-5],
        [5230.0, 2.62, -5.0, 0.0],
        [5230.0, 2.62, -5.0, 1.0],
        [f64::NAN, 2.62, -5.0, 2e-5],
        [5230.0, f64::INFINITY, -5.0, 2e-5],
        [5230.0, 2.62, f64::NEG_INFINITY, 2e-5],
        [5230.0, 2.62, -5.0, f64::NAN],
    ] {
        let [t, v, a, r] = schedule;
        let options = DualAnnealingOptions::default().with_schedule(t, v, a, r);
        let error = DualAnnealing::new(options, PowellLocalFactory::default()).unwrap_err();
        assert!(
            matches!(error.view(),scioxide_core::SciOxideErrorView::InvalidInput {argument,..} if argument==scioxide_core::argument!("temperature"))
        );
    }
}
