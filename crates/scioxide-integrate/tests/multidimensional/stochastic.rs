//! Independent analytic, determinism and resource tests for QMC, VEGAS and MISER.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;

use scioxide_core::*;
use scioxide_integrate::*;
use scioxide_random::{
    BitGenerator, Pcg64, Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan, Rng,
    SeedableRng, SplittableRng,
};
use scioxide_stats::MeanIntervalMethod;

fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}
fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(594),
        scioxide_core::algorithm_id!("integrate", "adaptive-mc-test", 1),
        0,
        0,
    );
    f(&mut state.begin_solve(execution, id, &mut budgets).unwrap())
}
fn sequential<T>(f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    run(ExecutionContext::sequential(), SolveBudgets::unbounded(), f)
}
fn provider(context: &SolveContext<'_, '_>, role: EvaluationRole) -> usize {
    context.counts().total().unwrap().provider(role)
}
fn vegas<S: RealScalar>(bounds: &[Interval<S>]) -> VegasTrainer<S> {
    VegasTrainer::new(
        bounds,
        VegasOptions::default()
            .with_bins(16)
            .with_iterations(budget(4))
            .with_samples_per_iteration(budget(2048)),
    )
    .unwrap()
}
fn miser_options() -> MiserOptions {
    MiserOptions::default()
        .with_evaluation(MonteCarloOptions::default().with_samples(budget(8192)))
        .with_max_leaves(budget(8))
        .with_max_depth(4)
}
fn miser<S: RealScalar>(bounds: &[Interval<S>]) -> MiserTrainer<S> {
    MiserTrainer::new(bounds, miser_options()).unwrap()
}
fn qmc(dimension: usize, replicates: usize, points: usize, seed: u64) -> QmcReplicatePlan {
    QmcReplicatePlan::new(
        &QmcPlan::sobol(dimension).unwrap(),
        &Philox4x64::from_seed(seed),
        QmcReplicateOptions::default()
            .with_replicates(replicates)
            .with_points_per_replicate(points),
    )
    .unwrap()
}
fn qmc_options(replicates: usize, points: usize) -> MonteCarloOptions {
    MonteCarloOptions::default()
        .with_samples(budget(points))
        .with_replicates(replicates)
}
fn failure() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("test_callback"),
        InvalidInputReason::Inconsistent,
        "injected",
    )
}
fn is_callback_non_finite(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    )
}

#[test]
fn qmc_analytic_complex_values_and_replicate_reduction() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let plans = qmc(2, 8, 1024, 594);
            let bounds = [Interval::new(0.0 as $s, 1.0 as $s).unwrap(); 2];
            let plan = MonteCarloIntegrator::new(&bounds, qmc_options(8, 1024)).unwrap();
            let mut ws = plan
                .workspace(2, ExecutionContext::sequential_native())
                .unwrap();
            let mut out = Complex::new(0.0 as $s, 0.0);
            let f = |x: &[$s]| Ok(Complex::new(x[0] * x[1], -2.0 * x[0] * x[1]));
            let report =
                sequential(|ctx| plan.integrate_into(&f, &plans, &mut out, &mut ws, ctx)).unwrap();
            assert_eq!(report.evaluations(), 8192);
            assert_eq!(
                report.termination(),
                IntegrationTermination::SampleBudgetCompleted
            );
            assert!((out.re - 0.25).abs() < $tol);
            assert!((out.im + 0.5).abs() < 2.0 * $tol);
            let means: Vec<_> = ws
                .children()
                .map(|c| c.values().unwrap()[0] as f64)
                .collect();
            let mean = means.iter().sum::<f64>() / 8.0;
            let variance = means.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (7.0 * 8.0);
            let mut s = ws.statistics()[0].clone().unwrap();
            assert!((out.re as f64 - mean).abs() < $tol as f64);
            assert!(
                (s.estimator_variance().round().unwrap() as f64 - variance).abs()
                    < $tol as f64 * 1e-4
            );
            assert_eq!(s.basis(), VarianceBasis::Replicates);
            assert_eq!((s.units(), s.samples()), (8, 8192));
            assert_eq!(s.moments().unwrap().count(), 8);
            assert_eq!(
                s.interval_report().unwrap().method(),
                MeanIntervalMethod::StudentT
            );
            assert_eq!(s.interval_report().unwrap().degrees_of_freedom(), Some(7.0));
            assert_eq!(s.effective_sample_size(), None);
            let child = ws.children().next().unwrap();
            assert_eq!(child.effective_sample_size(), Some(1024.0 as $s));
            assert!(child.statistics().iter().all(Option::is_none));
            let previous = out;
            let _ =
                sequential(|ctx| plan.integrate_into(&f, &plans, &mut out, &mut ws, ctx)).unwrap();
            assert_eq!(out, previous);
        }};
    }
    check!(f32, 2e-4);
    check!(f64, 2e-4);
}

#[test]
fn vegas_normalization_support_and_exact_constant_weight_evaluation() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let bounds = [Interval::new(0.0 as $s, 1.0 as $s).unwrap()];
            let plan = vegas(&bounds);
            let root = Philox4x64::from_seed(594);
            let (map, report) = plan
                .train(
                    |x: &[$s]| Ok((30.0 * x[0]).exp()),
                    RandomStreams::new(&root, 0),
                )
                .unwrap();
            assert_eq!(report.evaluations(), 8192);
            assert_eq!(report.updates(), 4);
            assert!(report.is_success());
            let edges = map.boundaries();
            assert_eq!(edges[0], 0.0);
            assert_eq!(*edges.last().unwrap(), 1.0);
            assert!(edges.windows(2).all(|p| p[1] > p[0]));
            assert!(edges[8] > 0.8); // adaptation concentrates bins at the exponential peak
            let b = 16.0 as $s;
            let mut mass = 0.0 as $s;
            for (i, pair) in edges.windows(2).enumerate() {
                let mut point = [0.0 as $s];
                let q = map
                    .sample(
                        ProposalUniforms::new(&[(i as $s + 0.5) / b]).unwrap(),
                        &mut point,
                    )
                    .unwrap();
                assert!(point[0] >= pair[0] && point[0] < pair[1]);
                mass += q * (pair[1] - pair[0]);
            }
            assert!((mass - 1.0).abs() < $tol);
            // q(x) has integral one, so q(x)/q(x) is independently known exactly.
            let callback = |x: &[$s]| {
                let bin = edges
                    .windows(2)
                    .position(|p| x[0] >= p[0] && x[0] < p[1])
                    .unwrap();
                let q = 1.0 / (b * (edges[bin + 1] - edges[bin]));
                Ok([q, -2.0 * q])
            };
            let importance = MonteCarloIntegrator::importance(
                &bounds,
                map.clone(),
                MonteCarloOptions::default().with_samples(budget(1024)),
            )
            .unwrap();
            let mut out = [0.0 as $s; 2];
            let mut ws = importance
                .workspace(2, ExecutionContext::sequential_native())
                .unwrap();
            let _ = sequential(|ctx| {
                importance.integrate_into(
                    &callback,
                    RandomStreams::new(&root, 1),
                    &mut out,
                    &mut ws,
                    ctx,
                )
            })
            .unwrap();
            assert!((out[0] - 1.0).abs() < $tol);
            assert!((out[1] + 2.0).abs() < 2.0 * $tol);
            let statistics = ws.statistics()[0].clone().unwrap();
            assert!(statistics.standard_error() < $tol);
            assert!(statistics.effective_sample_size().unwrap() > 1.0);
            let plans = qmc(1, 4, 256, 594);
            let replicated =
                MonteCarloIntegrator::importance(&bounds, map.clone(), qmc_options(4, 256))
                    .unwrap();
            let mut ws = replicated
                .workspace(2, ExecutionContext::sequential_native())
                .unwrap();
            let _ = sequential(|ctx| {
                replicated.integrate_into(&callback, &plans, &mut out, &mut ws, ctx)
            })
            .unwrap();
            assert!((out[0] - 1.0).abs() < $tol);
            assert!((out[1] + 2.0).abs() < 2.0 * $tol);
        }};
    }
    check!(f32, 2e-6);
    check!(f64, 2e-14);
}

#[test]
fn miser_partition_coverage_and_independent_welch_satterthwaite_reference() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let bounds = [Interval::new(-1.0 as $s, 2.0 as $s).unwrap()];
            let root = Philox4x64::from_seed(8);
            let (strata, training) = miser(&bounds)
                .train(|x: &[$s]| Ok(x[0].abs()), RandomStreams::new(&root, 0))
                .unwrap();
            assert!(training.updates() > 0);
            assert_eq!(strata.samples().iter().sum::<usize>(), 8192);
            assert!(strata.samples().iter().all(|n| *n >= 2));
            let mut sorted: Vec<_> = strata.strata().map(|b| b[0]).collect();
            sorted.sort_by(|a, b| a.lower().partial_cmp(&b.lower()).unwrap());
            assert_eq!(sorted[0].lower(), -1.0);
            assert_eq!(sorted.last().unwrap().upper(), 2.0);
            assert!(sorted.windows(2).all(|p| p[0].upper() == p[1].lower()));
            let plan = MonteCarloIntegrator::stratified(strata.clone()).unwrap();
            let mut ws = plan
                .workspace(2, ExecutionContext::sequential_native())
                .unwrap();
            let mut out = [0.0 as $s; 2];
            let report = sequential(|ctx| {
                plan.integrate_into(
                    |x: &[$s]| Ok([x[0], 3.0]),
                    RandomStreams::new(&root, 1),
                    &mut out,
                    &mut ws,
                    ctx,
                )
            })
            .unwrap();
            assert_eq!(report.evaluations(), 8192);
            assert_eq!(report.subdivisions(), strata.len() - 1);
            assert!((out[0] - 1.5).abs() < $tol);
            assert!((out[1] - 9.0).abs() < $tol);
            assert!(
                ws.statistics()[1]
                    .clone()
                    .unwrap()
                    .estimator_variance()
                    .is_zero()
            );
            // Independent reproduction of each stratum from its documented blocks.
            let mut expected_mean = 0.0;
            let mut expected_variance = 0.0;
            let mut denominator = 0.0;
            let mut first_block = 0_u64;
            for (b, &n) in strata.strata().zip(strata.samples()) {
                let width = b[0].upper() as f64 - b[0].lower() as f64;
                let mut data = Vec::new();
                let mut rng = root.substream(1, first_block).unwrap();
                for k in 0..n {
                    if k > 0 && k % SAMPLE_BLOCK == 0 {
                        rng = root
                            .substream(1, first_block + (k / SAMPLE_BLOCK) as u64)
                            .unwrap();
                    }
                    let u: $s = rng.next_float().unwrap();
                    data.push((b[0].lower() as f64 + u as f64 * width) * width);
                }
                first_block += n.div_ceil(SAMPLE_BLOCK) as u64;
                let mean = data.iter().sum::<f64>() / n as f64;
                let variance =
                    data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((n - 1) * n) as f64;
                expected_mean += mean;
                expected_variance += variance;
                denominator += variance * variance / (n - 1) as f64;
            }
            assert!((out[0] as f64 - expected_mean).abs() < $tol as f64 * 0.01);
            let mut s = ws.statistics()[0].clone().unwrap();
            assert_eq!(s.basis(), VarianceBasis::Strata);
            assert_eq!((s.units(), s.samples()), (strata.len(), 8192));
            assert!(s.moments().is_none());
            assert!(
                (s.estimator_variance().round().unwrap() as f64 - expected_variance).abs()
                    < $tol as f64 * 1e-5
            );
            let degrees = s.degrees_of_freedom().unwrap() as f64;
            assert!(
                (degrees - expected_variance.powi(2) / denominator).abs() < $tol as f64 * 100.0
            );
            assert_eq!(
                s.interval_report().unwrap().degrees_of_freedom(),
                Some(degrees)
            );
            assert_eq!(ws.children().len(), strata.len());
            assert_eq!(root.position(), Some(0));
        }};
    }
    check!(f32, 0.03);
    check!(f64, 0.03);
}

#[test]
fn training_stops_publish_only_completed_updates() {
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let plan = VegasTrainer::new(
        &bounds,
        VegasOptions::default()
            .with_bins(8)
            .with_iterations(budget(2))
            .with_samples_per_iteration(budget(32)),
    )
    .unwrap();
    let root = Philox4x64::from_seed(594);
    let mut map = plan.uniform_map().unwrap();
    let uniform = map.clone();
    let mut ws = plan
        .workspace(ExecutionContext::sequential_native())
        .unwrap();
    let train = |ws: &mut VegasWorkspace<f64>, map: &mut VegasMap<f64>, samples| {
        run(
            ExecutionContext::sequential(),
            SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(samples)),
            |ctx| {
                plan.train_into(
                    |x: &[f64]| Ok(x[0]),
                    RandomStreams::new(&root, 0),
                    map,
                    ws,
                    ctx,
                )
            },
        )
    };
    let error = train(&mut ws, &mut map, 31).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(map, uniform);
    assert_eq!(ws.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    let stopped = train(&mut ws, &mut map, 40).unwrap();
    assert_eq!(stopped.evaluations(), 40);
    assert_eq!(stopped.updates(), 1);
    assert_eq!(
        stopped.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Sample)
    );
    assert_ne!(map, uniform);
    assert_eq!(
        ws.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let (one_pass, _) = VegasTrainer::new(
        &bounds,
        VegasOptions::default()
            .with_bins(8)
            .with_iterations(budget(1))
            .with_samples_per_iteration(budget(32)),
    )
    .unwrap()
    .train(|x: &[f64]| Ok(x[0]), RandomStreams::new(&root, 0))
    .unwrap();
    assert_eq!(map, one_pass);

    let tree = miser(&bounds);
    let mut strata = tree.initial_strata().unwrap();
    let initial = strata.clone();
    let mut ws = tree
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let train = |ws: &mut MiserWorkspace<f64>, strata: &mut MiserStrata<f64>, samples| {
        run(
            ExecutionContext::sequential(),
            SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(samples)),
            |ctx| {
                tree.train_into(
                    |x: &[f64]| Ok(x[0]),
                    RandomStreams::new(&root, 4),
                    strata,
                    ws,
                    ctx,
                )
            },
        )
    };
    assert!(train(&mut ws, &mut strata, 255).is_err());
    assert_eq!(strata, initial);
    let stopped = train(&mut ws, &mut strata, 256).unwrap();
    assert_eq!(stopped.updates(), 1);
    assert_eq!(strata.len(), 2);
    assert_eq!(strata.samples().iter().sum::<usize>(), 8192);
    let stopped = run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Subdivision, budget(2)),
        |ctx| {
            tree.train_into(
                |x: &[f64]| Ok(x[0]),
                RandomStreams::new(&root, 4),
                &mut strata,
                &mut ws,
                ctx,
            )
        },
    )
    .unwrap();
    assert_eq!(
        stopped.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
    );
    assert_eq!((stopped.updates(), strata.len()), (2, 3));
}

#[test]
fn aggregate_stops_and_failures_publish_nothing_but_keep_child_progress() {
    let plans = qmc(1, 4, 16, 1);
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let plan = MonteCarloIntegrator::new(&bounds, qmc_options(4, 16)).unwrap();
    let mut ws = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let function = |x: &[f64]| Ok(x[0]);
    let mut out = 91.0;
    run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Sample, budget(63)),
        |ctx| {
            let error = plan
                .integrate_into(&function, &plans, &mut out, &mut ws, &mut *ctx)
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::ConvergenceFailure { .. }
            ));
            assert_eq!(provider(ctx, EvaluationRole::Function), 63);
        },
    );
    assert_eq!(out, 91.0);
    assert!(ws.statistics()[0].is_none());
    assert_eq!(
        ws.children()
            .map(|c| c.report().unwrap().evaluations())
            .collect::<Vec<_>>(),
        [16, 16, 16, 15]
    );
    assert_eq!(
        ws.children()
            .last()
            .unwrap()
            .report()
            .unwrap()
            .termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Sample)
    );
    sequential(|ctx| {
        assert!(
            plan.integrate_into(
                |_: &[f64]| -> Result<f64, _> { Err(failure()) },
                &plans,
                &mut out,
                &mut ws,
                ctx
            )
            .is_err()
        )
    });
    assert_eq!(out, 91.0);
    let stratified =
        MonteCarloIntegrator::stratified(miser(&bounds).initial_strata().unwrap()).unwrap();
    let mut ws = stratified
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let root = Philox4x64::from_seed(1);
    run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(3)),
        |ctx| {
            assert!(
                stratified
                    .integrate_into(
                        &function,
                        RandomStreams::new(&root, 0),
                        &mut out,
                        &mut ws,
                        &mut *ctx
                    )
                    .is_err()
            );
            assert_eq!(provider(ctx, EvaluationRole::Function), 1);
        },
    );
    assert_eq!(out, 91.0);
    let child = ws.children().next().unwrap().report().unwrap();
    assert_eq!(child.evaluations(), 1);
    assert_eq!(
        child.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
    );
    let error = sequential(|ctx| {
        stratified.integrate_into(
            |_: &[f64]| Ok(f64::NAN),
            RandomStreams::new(&root, 0),
            &mut out,
            &mut ws,
            ctx,
        )
    })
    .unwrap_err();
    assert!(is_callback_non_finite(&error));
    assert_eq!(out, 91.0);
}

struct Reordered {
    workers: usize,
    threaded: bool,
    omit: bool,
    duplicate: bool,
}
impl Executor for Reordered {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.workers).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let mut partitions: Vec<_> = plan.partitions().collect();
        partitions.reverse();
        if self.omit {
            partitions.pop();
        }
        let dispatch = |target| {
            SequentialExecutor.execute(ctx, plan, &|p, (), scope| {
                if &p == target {
                    op(p, (), scope)
                } else {
                    Ok(())
                }
            })
        };
        if self.threaded {
            std::thread::scope(|scope| {
                let handles: Vec<_> = partitions
                    .iter()
                    .map(|p| scope.spawn(move || dispatch(p)))
                    .collect();
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok::<_, SciOxideError>(())
            })?;
        } else {
            for p in &partitions {
                dispatch(p)?;
            }
        }
        if self.duplicate {
            dispatch(&partitions[0])?;
        }
        SequentialExecutor.execute(ctx, plan, &|_, (), _| Ok(()))
    }
}

/// Value, statistics, report and provider counts of one evaluation.
type Evaluation = (
    [f64; 2],
    Vec<Option<MonteCarloStatistics<f64>>>,
    IntegrationReport<f64>,
    usize,
    usize,
);

fn evaluate<Src: UniformSource<f64> + Copy, P: MonteCarloProposal<f64> + Sync>(
    plan: &MonteCarloIntegrator<f64, P>,
    source: Src,
    execution: ExecutionContext<'_>,
) -> Evaluation {
    let f = |x: &[f64]| Ok([x[0] * x[1], (3.0 * x[0]).exp() - x[1]]);
    let mut ws = plan.workspace(2, execution).unwrap();
    let mut out = [0.0; 2];
    run(execution, SolveBudgets::unbounded(), |ctx| {
        let report = plan
            .integrate_into(&f, source, &mut out, &mut ws, &mut *ctx)
            .unwrap();
        (
            out,
            ws.statistics().to_vec(),
            report,
            provider(ctx, EvaluationRole::Sample),
            provider(ctx, EvaluationRole::Function),
        )
    })
}

#[test]
fn sequential_and_executor_runs_are_bit_identical_for_every_plan_kind() {
    let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
    let root = Philox4x64::from_seed(594);
    let plans = qmc(2, 8, 128, 594);
    let pilot = |x: &[f64]| Ok([x[0] * x[1], -x[0] * x[1]]);
    // MISER pilots and evaluation share the evaluation policy; 3000 pilot samples
    // span three blocks, so executor policies dispatch pilot waves too.
    let strata = |policy: ExecutionPolicy, execution: ExecutionContext<'_>| {
        let tree = MiserTrainer::new(
            &bounds,
            miser_options()
                .with_pilot_samples(budget(3000))
                .with_evaluation(
                    MonteCarloOptions::default()
                        .with_samples(budget(5000))
                        .with_execution_policy(policy),
                ),
        )
        .unwrap();
        let mut strata = tree.initial_strata().unwrap();
        let mut ws = tree.workspace(2, execution).unwrap();
        let _ = run(execution, SolveBudgets::unbounded(), |ctx| {
            tree.train_into(
                &pilot,
                RandomStreams::new(&root, 0),
                &mut strata,
                &mut ws,
                ctx,
            )
        })
        .unwrap();
        strata
    };
    let (map, _) = vegas(&bounds)
        .train(
            |x: &[f64]| Ok((3.0 * x[0]).exp()),
            RandomStreams::new(&root, 1),
        )
        .unwrap();
    let independent = MonteCarloOptions::default().with_samples(budget(5000));
    let replicates = independent.with_samples(budget(1500)).with_replicates(5);
    let evaluate_all = |policy: ExecutionPolicy, execution: ExecutionContext<'_>| {
        let trained = strata(policy, execution);
        let stratified = MonteCarloIntegrator::stratified(trained.clone()).unwrap();
        (
            trained,
            [
                evaluate(
                    &MonteCarloIntegrator::new(&bounds, independent.with_execution_policy(policy))
                        .unwrap(),
                    RandomStreams::new(&root, 2),
                    execution,
                ),
                evaluate(
                    &MonteCarloIntegrator::importance(
                        &bounds,
                        map.clone(),
                        replicates.with_execution_policy(policy),
                    )
                    .unwrap(),
                    RandomStreams::new(&root, 3),
                    execution,
                ),
                evaluate(
                    &MonteCarloIntegrator::new(
                        &bounds,
                        qmc_options(8, 128).with_execution_policy(policy),
                    )
                    .unwrap(),
                    &plans,
                    execution,
                ),
                evaluate(&stratified, RandomStreams::new(&root, 4), execution),
            ],
        )
    };
    let (reference_strata, reference) =
        evaluate_all(ExecutionPolicy::Sequential, ExecutionContext::sequential());
    assert!(reference_strata.len() > 1);
    assert_eq!(reference[0].2.evaluations(), 5000);
    assert_eq!(reference[1].2.evaluations(), 7500);
    assert_eq!(reference[2].2.evaluations(), 1024);
    assert_eq!(reference[3].2.evaluations(), 5000);
    let f = |x: &[f64]| Ok((3.0 * x[0]).exp() * x[1]);
    let training = |policy| {
        VegasTrainer::new(
            &bounds,
            VegasOptions::default()
                .with_bins(16)
                .with_iterations(budget(2))
                .with_samples_per_iteration(budget(3000))
                .with_execution_policy(policy),
        )
        .unwrap()
    };
    let (reference_map, _) = training(ExecutionPolicy::Sequential)
        .train(&f, RandomStreams::new(&root, 5))
        .unwrap();
    for workers in [1, 2, 3, 4] {
        for threaded in [false, true] {
            let executor = Reordered {
                workers,
                threaded,
                omit: false,
                duplicate: false,
            };
            let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(workers).unwrap());
            let execution = ExecutionContext::new(&executor);
            let (trained, actual) = evaluate_all(policy, execution);
            assert_eq!(trained.samples(), reference_strata.samples());
            assert!(trained.strata().eq(reference_strata.strata()));
            for (actual, expected) in actual.iter().zip(&reference) {
                assert_eq!(actual.0.map(f64::to_bits), expected.0.map(f64::to_bits));
                assert_eq!(actual.1, expected.1);
                assert_eq!(actual.2, expected.2);
                assert_eq!((actual.3, actual.4), (expected.3, expected.4));
            }
            let parallel = training(policy);
            let mut trained = parallel.uniform_map().unwrap();
            let mut ws = parallel.workspace(execution).unwrap();
            let _ = run(execution, SolveBudgets::unbounded(), |ctx| {
                parallel.train_into(&f, RandomStreams::new(&root, 5), &mut trained, &mut ws, ctx)
            })
            .unwrap();
            assert_eq!(trained, reference_map);
        }
    }
}

#[test]
fn malformed_executor_dispatch_is_an_error_without_publication() {
    let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
    let root = Philox4x64::from_seed(8);
    let plans = qmc(2, 8, 128, 594);
    let policy = ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap());
    let replicated =
        MonteCarloIntegrator::new(&bounds, qmc_options(8, 128).with_execution_policy(policy))
            .unwrap();
    let independent = MonteCarloIntegrator::new(
        &bounds,
        MonteCarloOptions::default()
            .with_samples(budget(4096))
            .with_execution_policy(policy),
    )
    .unwrap();
    for duplicate in [false, true] {
        let executor = Reordered {
            workers: 2,
            threaded: false,
            omit: !duplicate,
            duplicate,
        };
        let f = |x: &[f64]| Ok(x[0]);
        let mut out = 91.0;
        let mut ws = replicated
            .workspace(1, ExecutionContext::new(&executor))
            .unwrap();
        run(
            ExecutionContext::new(&executor),
            SolveBudgets::unbounded(),
            |ctx| {
                assert!(
                    replicated
                        .integrate_into(&f, &plans, &mut out, &mut ws, ctx)
                        .is_err()
                );
            },
        );
        assert_eq!(out, 91.0);
        let mut ws = independent
            .workspace(1, ExecutionContext::new(&executor))
            .unwrap();
        run(
            ExecutionContext::new(&executor),
            SolveBudgets::unbounded(),
            |ctx| {
                assert!(
                    independent
                        .integrate_into(&f, RandomStreams::new(&root, 0), &mut out, &mut ws, ctx)
                        .is_err()
                );
            },
        );
        assert_eq!(out, 91.0);
        assert!(ws.statistics()[0].is_none());
    }
}

#[test]
fn invalid_training_inputs_and_provider_failures_preserve_products() {
    #[derive(Clone)]
    struct Bad;
    impl BitGenerator for Bad {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            Err(failure())
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            Err(failure())
        }
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
            Err(failure())
        }
    }
    impl Rng<f64> for Bad {
        fn next_float(&mut self) -> Result<f64, SciOxideError> {
            Err(failure())
        }
    }
    impl SplittableRng for Bad {
        fn split(&self, _: u64) -> Result<Self, SciOxideError> {
            Ok(Bad)
        }
        fn substream(&self, _: u64, _: u64) -> Result<Self, SciOxideError> {
            Ok(Bad)
        }
    }
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    assert!(VegasTrainer::new(&bounds, VegasOptions::default().with_bins(1)).is_err());
    assert!(VegasTrainer::new(&bounds, VegasOptions::default().with_damping(f64::NAN)).is_err());
    assert!(VegasTrainer::new(&bounds, VegasOptions::default().with_damping(0.0)).is_err());
    assert!(VegasTrainer::<f32>::new(&[], VegasOptions::default().with_bins(1 << 25)).is_err());
    assert!(MiserTrainer::new(&bounds, miser_options().with_pilot_samples(budget(3))).is_err());
    assert!(
        MiserTrainer::new(
            &bounds,
            miser_options().with_evaluation(MonteCarloOptions::default().with_replicates(2))
        )
        .is_err()
    );
    let root = Philox4x64::from_seed(1);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let plan = vegas(&bounds);
        let mut map = plan.uniform_map().unwrap();
        let old = map.clone();
        let mut ws = plan
            .workspace(ExecutionContext::sequential_native())
            .unwrap();
        let error = sequential(|ctx| {
            plan.train_into(
                |_: &[f64]| Ok(invalid),
                RandomStreams::new(&root, 0),
                &mut map,
                &mut ws,
                ctx,
            )
        })
        .unwrap_err();
        assert!(is_callback_non_finite(&error));
        assert_eq!(map, old);
        let plan = miser(&bounds);
        let mut strata = plan.initial_strata().unwrap();
        let old = strata.clone();
        let mut ws = plan
            .workspace(2, ExecutionContext::sequential_native())
            .unwrap();
        let error = sequential(|ctx| {
            plan.train_into(
                |x: &[f64]| Ok([x[0], invalid]),
                RandomStreams::new(&root, 0),
                &mut strata,
                &mut ws,
                ctx,
            )
        })
        .unwrap_err();
        assert!(is_callback_non_finite(&error));
        assert_eq!(strata, old);
    }
    // Negative values are valid: the importance is the component norm.
    let (map, report) = vegas(&bounds)
        .train(|x: &[f64]| Ok(-x[0]), RandomStreams::new(&root, 0))
        .unwrap();
    let (same, _) = vegas(&bounds)
        .train(|x: &[f64]| Ok(x[0]), RandomStreams::new(&root, 0))
        .unwrap();
    assert!(report.is_success());
    assert_eq!(map, same);
    let plan = vegas(&bounds);
    let mut map = plan.uniform_map().unwrap();
    let old = map.clone();
    let mut ws = plan
        .workspace(ExecutionContext::sequential_native())
        .unwrap();
    sequential(|ctx| {
        assert!(
            plan.train_into(
                |_: &[f64]| Ok(1.0),
                RandomStreams::new(&Bad, 0),
                &mut map,
                &mut ws,
                &mut *ctx
            )
            .is_err()
        );
        assert_eq!(provider(ctx, EvaluationRole::Sample), 1);
        assert_eq!(provider(ctx, EvaluationRole::Function), 0);
    });
    assert_eq!(map, old);
    let plan = miser(&bounds);
    let mut strata = plan.initial_strata().unwrap();
    let old = strata.clone();
    let mut ws = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    sequential(|ctx| {
        assert!(
            plan.train_into(
                |_: &[f64]| -> Result<f64, _> { Err(failure()) },
                RandomStreams::new(&root, 0),
                &mut strata,
                &mut ws,
                ctx
            )
            .is_err()
        )
    });
    assert_eq!(strata, old);
    let other = VegasTrainer::new(&[bounds[0]; 2], VegasOptions::default().with_bins(16)).unwrap();
    let mut wrong = other.uniform_map().unwrap();
    let mut ws = vegas(&bounds)
        .workspace(ExecutionContext::sequential_native())
        .unwrap();
    assert!(
        vegas(&bounds)
            .train_into(
                |_: &[f64]| Ok(1.0),
                RandomStreams::new(&root, 0),
                &mut wrong,
                &mut ws,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        ws.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn training_products_follow_the_caller_stream() {
    let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
    let f = |x: &[f64]| Ok((-10.0 * (x[0] - 0.3).powi(2)).exp() * x[1]);
    let root = Pcg64::from_seed(7);
    let plan = vegas(&bounds);
    let (first, _) = plan.train(&f, RandomStreams::new(&root, 0)).unwrap();
    let (again, _) = plan.train(&f, RandomStreams::new(&root, 0)).unwrap();
    let (other, _) = plan.train(&f, RandomStreams::new(&root, 1)).unwrap();
    assert_eq!(first, again);
    assert_ne!(first, other);
    let plan = miser(&bounds);
    let (first, _) = plan.train(&f, RandomStreams::new(&root, 0)).unwrap();
    let (again, _) = plan.train(&f, RandomStreams::new(&root, 0)).unwrap();
    let (other, _) = plan.train(&f, RandomStreams::new(&root, 1)).unwrap();
    assert_eq!(first, again);
    assert_ne!(first, other);
}

#[test]
fn rank_zero_and_zero_measure_never_train_or_sample() {
    let root = Philox4x64::from_seed(1);
    for bounds in [vec![], vec![Interval::new(3.0, 3.0).unwrap()]] {
        let mut out = 91.0;
        let plan = MonteCarloIntegrator::new(&bounds, MonteCarloOptions::default()).unwrap();
        let mut ws = plan
            .workspace(1, ExecutionContext::sequential_native())
            .unwrap();
        sequential(|ctx| {
            let report = plan
                .integrate_into(
                    |_: &[f64]| Ok(7.0),
                    RandomStreams::new(&root, 0),
                    &mut out,
                    &mut ws,
                    &mut *ctx,
                )
                .unwrap();
            assert_eq!(out, if bounds.is_empty() { 7.0 } else { 0.0 });
            assert_eq!(
                report.termination(),
                if bounds.is_empty() {
                    IntegrationTermination::FixedRuleCompleted
                } else {
                    IntegrationTermination::DegenerateDomain
                }
            );
            assert_eq!(provider(ctx, EvaluationRole::Sample), 0);
        });
        assert!(ws.statistics()[0].is_none());
        let tree = miser(&bounds);
        let mut strata = tree.initial_strata().unwrap();
        let mut ws = tree
            .workspace(1, ExecutionContext::sequential_native())
            .unwrap();
        sequential(|ctx| {
            let trained = tree
                .train_into(
                    |_: &[f64]| -> Result<f64, _> { Err(failure()) },
                    RandomStreams::new(&root, 0),
                    &mut strata,
                    &mut ws,
                    &mut *ctx,
                )
                .unwrap();
            assert_eq!(trained.evaluations(), 0);
            assert_eq!(
                trained.termination(),
                if bounds.is_empty() {
                    IntegrationTermination::FixedRuleCompleted
                } else {
                    IntegrationTermination::DegenerateDomain
                }
            );
            let plan = MonteCarloIntegrator::stratified(strata.clone()).unwrap();
            let _ = plan
                .integrate_into(
                    |_: &[f64]| Ok(7.0),
                    RandomStreams::new(&root, 0),
                    &mut out,
                    &mut plan
                        .workspace(1, ExecutionContext::sequential_native())
                        .unwrap(),
                    &mut *ctx,
                )
                .unwrap();
            assert_eq!(out, if bounds.is_empty() { 7.0 } else { 0.0 });
            assert_eq!(provider(ctx, EvaluationRole::Sample), 0);
        });
        let map_plan = VegasTrainer::new(&bounds, VegasOptions::default()).unwrap();
        let (map, report) = map_plan
            .train(
                |_: &[f64]| -> Result<f64, _> { Err(failure()) },
                RandomStreams::new(&root, 0),
            )
            .unwrap();
        assert_eq!(report.evaluations(), 0);
        assert_eq!(map, map_plan.uniform_map().unwrap());
    }
    // Randomized QMC points always have at least one coordinate.
    let plans = qmc(1, 2, 8, 1);
    let plan = MonteCarloIntegrator::new(&[], qmc_options(2, 8)).unwrap();
    let mut out = 91.0;
    assert!(
        plan.integrate_into(
            |_: &[f64]| Ok(7.0),
            &plans,
            &mut out,
            &mut plan
                .workspace(1, ExecutionContext::sequential_native())
                .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, 91.0);
}

#[test]
fn cancellation_stops_between_observations_without_aggregate() {
    let cancel = CancellationToken::new();
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let plans = qmc(1, 4, 16, 1);
    let plan = MonteCarloIntegrator::new(&bounds, qmc_options(4, 16)).unwrap();
    let mut ws = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut out = 91.0;
    run(
        ExecutionContext::sequential().with_cancellation(&cancel),
        SolveBudgets::unbounded(),
        |ctx| {
            let error = plan
                .integrate_into(
                    |x: &[f64]| {
                        cancel.cancel();
                        Ok(x[0])
                    },
                    &plans,
                    &mut out,
                    &mut ws,
                    &mut *ctx,
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::ExecutionStopped { .. }
            ));
            assert_eq!(provider(ctx, EvaluationRole::Function), 1);
        },
    );
    assert_eq!(out, 91.0);
    let first = ws.children().next().unwrap().report().unwrap();
    assert_eq!(first.evaluations(), 1);
    assert!(matches!(
        first.termination(),
        IntegrationTermination::ExecutionStopped(_)
    ));
    assert!(ws.children().skip(1).all(|c| c.report().is_none()));
    let map_plan = vegas(&bounds);
    let mut map = map_plan.uniform_map().unwrap();
    let original = map.clone();
    let mut ws = map_plan
        .workspace(ExecutionContext::sequential_native())
        .unwrap();
    let root = Philox4x64::from_seed(1);
    run(
        ExecutionContext::sequential().with_cancellation(&cancel),
        SolveBudgets::unbounded(),
        |ctx| {
            let error = map_plan
                .train_into(
                    |_: &[f64]| -> Result<f64, _> { Err(failure()) },
                    RandomStreams::new(&root, 0),
                    &mut map,
                    &mut ws,
                    &mut *ctx,
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::ExecutionStopped { .. }
            ));
            assert_eq!(provider(ctx, EvaluationRole::Function), 0);
        },
    );
    assert_eq!(map, original);
}

#[test]
fn qmc_never_requires_representable_pointwise_iid_variance() {
    // Every balanced scrambled digital block cancels exactly, while the pointwise
    // squared values overflow. Only replicate dispersion is relevant to QMC.
    macro_rules! check {
        ($s:ty,$amplitude:expr) => {{
            let plans = qmc(1, 4, 16, 594);
            let plan = MonteCarloIntegrator::new(
                &[Interval::new(0.0 as $s, 1.0 as $s).unwrap()],
                qmc_options(4, 16),
            )
            .unwrap();
            let mut ws = plan
                .workspace(1, ExecutionContext::sequential_native())
                .unwrap();
            let mut out = 1.0 as $s;
            let r = sequential(|ctx| {
                plan.integrate_into(
                    |x: &[$s]| Ok(if x[0] < 0.5 { $amplitude } else { -$amplitude }),
                    &plans,
                    &mut out,
                    &mut ws,
                    ctx,
                )
            })
            .unwrap();
            assert!(r.is_success());
            assert_eq!(out, 0.0);
            let s = ws.statistics()[0].clone().unwrap();
            assert_eq!(
                s.moments().unwrap().variance_of_sample_mean().unwrap(),
                Some(0.0)
            );
            assert_eq!(s.standard_error(), 0.0);
            assert!(ws.children().all(|c| c.statistics()[0].is_none()));
        }};
    }
    check!(f32, 1e30f32);
    check!(f64, 1e200f64);
}

/// Publish a representable estimate and standard error even though their
/// estimator variance leaves the IEEE range; only an explicit projection fails.
fn assert_unrepresentable_variance<S: RealScalar>(
    value: S,
    exact: S,
    statistics: &Option<MonteCarloStatistics<S>>,
    report: &IntegrationReport<S>,
) {
    assert_eq!(
        report.termination(),
        IntegrationTermination::SampleBudgetCompleted
    );
    let mut statistics = statistics.clone().unwrap();
    let error = statistics.standard_error();
    assert!(error.is_finite() && error > S::ZERO);
    assert!(statistics.estimator_variance().round().is_err());
    let ratio = statistics
        .estimator_variance()
        .div(Scaled::from_value(error).unwrap().square().unwrap())
        .unwrap()
        .round()
        .unwrap();
    assert!((ratio - S::ONE).abs() <= S::try_from_usize_exact(8).unwrap() * S::EPSILON);
    assert!(
        (value.into_f64() - exact.into_f64()).abs() <= 6.0 * error.into_f64(),
        "{value:?} {exact:?} {error:?} {:?}",
        statistics.basis()
    );
    let interval = statistics.confidence_interval().unwrap();
    assert!(interval.lower() < value && value < interval.upper());
}

#[test]
fn independent_estimators_publish_when_their_variance_leaves_the_ieee_range() {
    macro_rules! check {
        ($s:ty, $scale:expr, $boxes:expr) => {{
            let scale: $s = $scale;
            let f = move |x: &[$s]| Ok(scale * x[0]);
            let exact = scale / 2.0;
            let bounds = [Interval::new(0.0 as $s, 1.0 as $s).unwrap()];
            let root = Philox4x64::from_seed(161);
            let mut value = 0.0 as $s;
            for (stream, options) in [
                (0, MonteCarloOptions::default()),
                (
                    1,
                    MonteCarloOptions::default()
                        .with_samples(budget(512))
                        .with_replicates(8),
                ),
            ] {
                let plan = MonteCarloIntegrator::new(&bounds, options).unwrap();
                let mut ws = plan
                    .workspace(1, ExecutionContext::sequential_native())
                    .unwrap();
                let report = sequential(|ctx| {
                    plan.integrate_into(
                        &f,
                        RandomStreams::new(&root, stream),
                        &mut value,
                        &mut ws,
                        ctx,
                    )
                })
                .unwrap();
                assert_unrepresentable_variance(value, exact, &ws.statistics()[0], &report);
            }
            let (strata, _) = miser(&bounds)
                .train(&f, RandomStreams::new(&root, 2))
                .unwrap();
            let stratified = MonteCarloIntegrator::stratified(strata).unwrap();
            let mut ws = stratified
                .workspace(1, ExecutionContext::sequential_native())
                .unwrap();
            let report = sequential(|ctx| {
                stratified.integrate_into(
                    &f,
                    RandomStreams::new(&root, 3),
                    &mut value,
                    &mut ws,
                    ctx,
                )
            })
            .unwrap();
            assert_unrepresentable_variance(value, exact, &ws.statistics()[0], &report);
            let (map, _) = vegas(&bounds)
                .train(&f, RandomStreams::new(&root, 4))
                .unwrap();
            let importance =
                MonteCarloIntegrator::importance(&bounds, map, MonteCarloOptions::default())
                    .unwrap();
            let mut ws = importance
                .workspace(1, ExecutionContext::sequential_native())
                .unwrap();
            let report = sequential(|ctx| {
                importance.integrate_into(
                    &f,
                    RandomStreams::new(&root, 5),
                    &mut value,
                    &mut ws,
                    ctx,
                )
            })
            .unwrap();
            assert_unrepresentable_variance(value, exact, &ws.statistics()[0], &report);
            // A constant integrand over an extreme but representable box: the
            // trained proposal's weights alone carry the out-of-range variance.
            let boxes: [($s, $s, $s); 2] = $boxes;
            for (stream, (lower, upper, scale)) in (6..).zip(boxes) {
                let bounds = [Interval::new(lower, upper).unwrap(); 2];
                let (center, half) = (lower / 2.0 + upper / 2.0, upper / 2.0 - lower / 2.0);
                let (map, _) = VegasTrainer::new(
                    &bounds,
                    VegasOptions::default()
                        .with_bins(8)
                        .with_iterations(budget(2))
                        .with_samples_per_iteration(budget(512)),
                )
                .unwrap()
                .train(
                    |x: &[$s]| Ok(scale * (1.0 + ((x[0] - center) / half).abs())),
                    RandomStreams::new(&root, stream),
                )
                .unwrap();
                let importance = MonteCarloIntegrator::importance(
                    &bounds,
                    map,
                    MonteCarloOptions::default().with_samples(budget(512)),
                )
                .unwrap();
                let mut ws = importance
                    .workspace(1, ExecutionContext::sequential_native())
                    .unwrap();
                let report = sequential(|ctx| {
                    importance.integrate_into(
                        |_: &[$s]| Ok(scale),
                        RandomStreams::new(&root, stream + 10),
                        &mut value,
                        &mut ws,
                        ctx,
                    )
                })
                .unwrap();
                let exact = half * scale * 4.0 * half;
                assert_unrepresentable_variance(value, exact, &ws.statistics()[0], &report);
            }
        }};
    }
    check!(
        f32,
        1e25,
        [(1e30, 1e30 + 1e28, 1e-30), (-1e37, 1e37, 1e-37)]
    );
    check!(
        f64,
        1e165,
        [(1e300, 1e300 + 1e290, 1e-300), (-1.7e308, 1.7e308, 1e-310)]
    );
}

#[test]
fn unrepresentable_damped_scores_retain_the_previous_map() {
    macro_rules! check {
        ($s:ty) => {{
            let options = VegasOptions::default()
                .with_bins(8)
                .with_iterations(budget(1))
                .with_samples_per_iteration(budget(256))
                .with_damping(<$s>::MAX);
            let plan =
                VegasTrainer::<$s>::new(&[Interval::new(0.0 as $s, 1.0 as $s).unwrap()], options)
                    .unwrap();
            let root = Philox4x64::from_seed(594);
            let (map, report) = plan
                .train(|x: &[$s]| Ok(x[0]), RandomStreams::new(&root, 0))
                .unwrap();
            assert_eq!(report.updates(), 1);
            assert_eq!(map, plan.uniform_map().unwrap());
        }};
    }
    check!(f32);
    check!(f64);
}

macro_rules! signed_miser {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let bounds = [Interval::new(0.0 as $s, 1.0).unwrap(); 2];
            let root = Philox4x64::from_seed(42);
            let plan = MiserTrainer::new(
                &bounds,
                MiserOptions::default()
                    .with_max_leaves(budget(2))
                    .with_max_depth(1)
                    .with_evaluation(MonteCarloOptions::default().with_samples(budget(8192))),
            )
            .unwrap();
            let signed = |x: &[$s]| {
                Ok(if x[0] < 0.5 {
                    -(1.0 + 0.1 * x[1])
                } else {
                    1.0 + 0.1 * x[1]
                })
            };
            let (strata, _) = plan.train(&signed, RandomStreams::new(&root, 0)).unwrap();
            assert!(
                strata
                    .strata()
                    .all(|b| b[0].upper() <= 0.5 || b[0].lower() >= 0.5)
            );
            // Removing the sign reproduces the old objective and isolates its y split.
            let (magnitude_strata, _) = plan
                .train(
                    |x: &[$s]| Ok(1.0 + 0.1 * x[1]),
                    RandomStreams::new(&root, 0),
                )
                .unwrap();
            assert!(
                magnitude_strata
                    .strata()
                    .all(|b| b[1].upper() <= 0.5 || b[1].lower() >= 0.5)
            );
            let evaluate = |strata| {
                let plan = MonteCarloIntegrator::stratified(strata).unwrap();
                let mut ws = plan
                    .workspace(1, ExecutionContext::sequential_native())
                    .unwrap();
                let mut out = 0.0;
                let _ = plan
                    .integrate_into(
                        &signed,
                        RandomStreams::new(&root, 1),
                        &mut out,
                        &mut ws,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                (out, ws.statistics()[0].clone().unwrap().standard_error())
            };
            let (value, standard_error) = evaluate(strata);
            let (_, magnitude_error) = evaluate(magnitude_strata);
            assert!(
                standard_error * 10.0 <= magnitude_error,
                "{standard_error} vs {magnitude_error}"
            );
            assert!(value.abs() < 5.0 * standard_error);
        }
    };
}
signed_miser!(miser_signed_dispersion_f64, f64);
signed_miser!(miser_signed_dispersion_f32, f32);

#[test]
fn miser_spends_refinements_in_the_high_variance_right_half() {
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let root = Philox4x64::from_seed(42);
    let plan = MiserTrainer::new(
        &bounds,
        MiserOptions::default()
            .with_max_leaves(budget(5))
            .with_max_depth(4),
    )
    .unwrap();
    let (strata, report) = plan
        .train(
            |x: &[f64]| Ok(if x[0] < 0.5 { 0.01 * x[0] } else { x[0] }),
            RandomStreams::new(&root, 0),
        )
        .unwrap();
    assert_eq!(report.updates(), 4);
    assert_eq!(strata.strata().filter(|b| b[0].upper() <= 0.5).count(), 1);
    assert_eq!(strata.strata().filter(|b| b[0].lower() >= 0.5).count(), 4);
}

#[test]
fn miser_training_keeps_large_f32_sample_budgets_exact() {
    let total = 50_000_001;
    let plan = MiserTrainer::new(
        &[Interval::new(0.0_f32, 1.0).unwrap()],
        MiserOptions::default()
            .with_max_leaves(budget(2))
            .with_max_depth(1)
            .with_evaluation(MonteCarloOptions::default().with_samples(budget(total))),
    )
    .unwrap();
    let root = Philox4x64::from_seed(42);
    let (strata, report) = plan
        .train(|x: &[f32]| Ok(x[0]), RandomStreams::new(&root, 0))
        .unwrap();
    assert_eq!(report.updates(), 1);
    assert_eq!(strata.samples().iter().sum::<usize>(), total);
    assert!(strata.samples().iter().all(|&samples| samples >= 2));
}

#[test]
fn miser_owned_training_infers_scalar_and_fixed_array_components() {
    fn check<V: scioxide_core::AffineValue<Scale = f64> + Copy + Default>(
        f: impl Fn(&[f64]) -> Result<V, scioxide_core::SciOxideError> + Sync,
    ) {
        let root = Philox4x64::from_seed(42);
        let trainer = MiserTrainer::new(
            &[Interval::new(0.0, 1.0).unwrap()],
            MiserOptions::default().with_max_leaves(budget(2)),
        )
        .unwrap();
        let (owned, report) = trainer.train(&f, RandomStreams::new(&root, 0)).unwrap();
        let mut output = trainer.initial_strata().unwrap();
        let mut ws = trainer
            .workspace(
                V::default().dimension().unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        let expected = trainer
            .train_into(
                &f,
                RandomStreams::new(&root, 0),
                &mut output,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(owned, output);
        assert_eq!(report, expected);
    }
    check(|x| Ok(x[0]));
    check(|x| Ok([x[0], x[0] * x[0]]));
    check(|_| Ok([0.0_f64; 0]));
}
