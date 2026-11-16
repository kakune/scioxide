#![allow(clippy::unwrap_used)]

use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{Budget, ExecutionContext, ExecutionPolicy, Interval, Scaled, SciOxideError};
use scioxide_random::{Philox4x64, Rng, SeedableRng, SplittableRng};
use scioxide_stats::ConfidenceProbability;

use super::kernel::lanes;
use super::statistics::welch_satterthwaite;
use super::training::importance;
use super::*;
use crate::IntegrationTermination;

fn plan(width: f64, samples: usize) -> MonteCarloIntegrator<f64> {
    MonteCarloIntegrator::new(
        &[Interval::new(0.0, width).unwrap()],
        MonteCarloOptions::default().with_samples(Budget::try_from(samples).unwrap()),
    )
    .unwrap()
}

fn integrate(
    plan: &MonteCarloIntegrator<f64>,
    stream: u64,
    f: &(dyn Fn(&[f64]) -> Result<f64, SciOxideError> + Sync),
) -> Result<(f64, Option<MonteCarloStatistics<f64>>, usize), SciOxideError> {
    let root = Philox4x64::from_seed(42);
    let mut workspace = plan.workspace(1, ExecutionContext::sequential_native())?;
    let mut value = 0.0;
    let report = plan.integrate_into(
        f,
        RandomStreams::new(&root, stream),
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((
        value,
        workspace.statistics()[0].clone(),
        report.evaluations(),
    ))
}

#[test]
fn consumes_exact_sample_budget_and_is_reproducible_per_stream() {
    let plan = plan(1.0, 128);
    let first = integrate(&plan, 3, &|x| Ok(x[0])).unwrap();
    let second = integrate(&plan, 3, &|x| Ok(x[0])).unwrap();
    let other = integrate(&plan, 4, &|x| Ok(x[0])).unwrap();
    assert_eq!(first.0, second.0);
    assert_ne!(first.0, other.0);
    assert_eq!(first.2, 128);
    let statistics = first.1.unwrap();
    assert_eq!(statistics.samples(), 128);
    let moments = statistics.moments().unwrap();
    let variance = statistics.estimator_variance().round().unwrap();
    let reference = moments.variance_of_sample_mean().unwrap().unwrap();
    assert!(moments.sample_variance().unwrap().unwrap() > variance);
    assert!((variance - reference).abs() <= 4.0 * f64::EPSILON * reference);
    assert!(statistics.standard_error() > 0.0);
}

#[test]
fn observations_follow_the_documented_block_streams() {
    let samples = 2 * SAMPLE_BLOCK + 300;
    let (value, statistics, _) = integrate(&plan(1.0, samples), 9, &|x| Ok(x[0])).unwrap();
    let root = Philox4x64::from_seed(42);
    let mut sum = 0.0;
    for block in 0..3 {
        let mut rng = root.substream(9, block).unwrap();
        for _ in 0..SAMPLE_BLOCK.min(samples - block as usize * SAMPLE_BLOCK) {
            let u: f64 = rng.next_float().unwrap();
            sum += u;
        }
    }
    assert!((value - sum / samples as f64).abs() < 1e-14);
    assert_eq!(statistics.unwrap().units(), samples);
}

#[test]
fn constant_integrand_has_zero_estimator_variance() {
    let (value, statistics, _) = integrate(&plan(2.0, 16), 7, &|_| Ok(1.0)).unwrap();
    let mut statistics = statistics.unwrap();
    assert_eq!(value, 2.0);
    assert!(statistics.estimator_variance().is_zero());
    assert_eq!(statistics.standard_error(), 0.0);
    assert_eq!(statistics.confidence_interval().unwrap().lower(), 2.0);
    assert_eq!(statistics.confidence_interval().unwrap().upper(), 2.0);
}

#[test]
fn estimator_variance_keeps_extended_range_beyond_ieee_projection() {
    // (volume, integrand, IEEE projection of the variance): representable,
    // underflowing to zero, and overflowing while the estimate is representable.
    type Integrand = fn(&[f64]) -> Result<f64, SciOxideError>;
    let cases: [(f64, Integrand, Option<bool>); 3] = [
        (1.0e200, |x| Ok((x[0] / 1.0e200) * 1.0e-150), Some(true)),
        (1.0e-200, |x| Ok(x[0] / 1.0e-200), Some(false)),
        (1.0, |x| Ok(1.0e160 * x[0]), None),
    ];
    for (width, f, projection) in cases {
        let (value, statistics, _) = integrate(&plan(width, 64), 13, &f).unwrap();
        let mut statistics = statistics.unwrap();
        let error = statistics.standard_error();
        assert!(value.is_finite());
        assert!(error.is_finite() && error > 0.0);
        let variance = statistics.estimator_variance();
        let ratio = variance
            .div(Scaled::from_value(error).unwrap().square().unwrap())
            .unwrap()
            .round()
            .unwrap();
        assert!((ratio - 1.0).abs() <= 8.0 * f64::EPSILON);
        match projection {
            Some(normal) => assert_eq!(variance.round().unwrap() > 0.0, normal),
            None => assert!(variance.round().is_err()),
        }
        let interval = statistics.confidence_interval().unwrap();
        assert!(interval.lower() < value && value < interval.upper());
    }
}

#[test]
fn higher_confidence_produces_wider_interval() {
    let interval = |probability| {
        let options = MonteCarloOptions::default()
            .with_samples(Budget::try_from(256).unwrap())
            .with_confidence_probability(ConfidenceProbability::new(probability).unwrap());
        let plan = MonteCarloIntegrator::new(&[Interval::new(0.0, 1.0).unwrap()], options).unwrap();
        let mut statistics = integrate(&plan, 11, &|x| Ok(x[0])).unwrap().1.unwrap();
        assert_eq!(statistics.confidence_probability().get(), probability);
        statistics.confidence_interval().unwrap().width().unwrap()
    };
    assert!(interval(0.975) > interval(0.80));
}

#[test]
fn zero_width_domain_does_not_evaluate() {
    let calls = AtomicUsize::new(0);
    let (value, statistics, evaluations) = integrate(&plan(0.0, 8), 1, &|_| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    })
    .unwrap();
    assert_eq!((value, evaluations), (0.0, 0));
    assert!(statistics.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn non_finite_samples_are_callback_faults() {
    let error = integrate(&plan(1.0, 8), 1, &|_| Ok(f64::NAN)).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Callback,
            ..
        }
    ));
}

#[test]
fn importance_is_exact_for_scalars_and_a_norm_for_components() {
    assert_eq!(importance(&-0.1_f64).unwrap().round().unwrap(), 0.1);
    assert_eq!(
        importance(&3.0e-310_f64).unwrap().round().unwrap(),
        3.0e-310
    );
    assert_eq!(importance(&[3.0_f64, -4.0]).unwrap().round().unwrap(), 5.0);
    assert_eq!(
        importance(&[1.0e300_f64, 1.0e300])
            .unwrap()
            .round()
            .unwrap(),
        1.0e300 * core::f64::consts::SQRT_2
    );
    assert!(importance(&[1.0_f64, f64::INFINITY]).is_err());
}

#[test]
fn welch_satterthwaite_reduces_to_the_stratum_degrees_and_falls_back() {
    let variance = |v: f64| Ok((Scaled::from_value(v).unwrap(), 11));
    let (total, degrees) = welch_satterthwaite([variance(2.0)], 0).unwrap();
    assert_eq!((total.round().unwrap(), degrees), (2.0, 10.0));
    let (total, degrees) = welch_satterthwaite([variance(1.0), variance(1.0)], 0).unwrap();
    assert_eq!((total.round().unwrap(), degrees), (2.0, 20.0));
    let (_, degrees) = welch_satterthwaite::<f64>([variance(0.0), variance(0.0)], 20).unwrap();
    assert_eq!(degrees, 20.0);
}

#[test]
fn single_runs_sample_waves_of_policy_lanes() {
    let execution = ExecutionContext::sequential_native();
    assert_eq!(lanes(ExecutionPolicy::Sequential, 100, execution), 1);
    assert_eq!(lanes(ExecutionPolicy::Auto, 100, execution), 1);
    assert_eq!(
        lanes(
            ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(1000).unwrap()),
            100,
            execution
        ),
        1
    );
    let policy = ExecutionPolicy::Auto;
    let plan = MonteCarloIntegrator::new(
        &[Interval::new(0.0, 1.0).unwrap()],
        MonteCarloOptions::default().with_execution_policy(policy),
    )
    .unwrap();
    assert_eq!(
        plan.workspace_requirements(1, ExecutionContext::sequential_native())
            .unwrap()
            .units(),
        [1, 1]
    );
    let replicated = MonteCarloIntegrator::new(
        &[Interval::new(0.0, 1.0).unwrap()],
        MonteCarloOptions::default().with_replicates(5),
    )
    .unwrap();
    assert_eq!(
        replicated
            .workspace_requirements(1, ExecutionContext::sequential_native())
            .unwrap()
            .units(),
        [5, 1]
    );
    assert_eq!(
        replicated.algorithm_id(),
        scioxide_core::algorithm_id!("integrate", "MonteCarloReplicates", 1)
    );
}

#[cfg(feature = "std")]
#[test]
fn worker_storage_is_bounded_and_growth_precedes_sampling() {
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    let execution = ExecutionContext::new(&executor);
    let policy = ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(8).unwrap());
    let options = MonteCarloOptions::default()
        .with_samples(Budget::try_from(5 * SAMPLE_BLOCK + 1).unwrap())
        .with_execution_policy(policy);
    let bounds = [Interval::new(0.0_f64, 1.0).unwrap()];
    let plan = MonteCarloIntegrator::new(&bounds, options).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    assert_eq!(workspace.lanes.len(), 1);
    let calls = AtomicUsize::new(0);
    let root = Philox4x64::from_seed(42);
    let f = |x: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x[0])
    };
    let mut value = 91.0;
    assert!(
        plan.integrate_into(
            &f,
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            execution
        )
        .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(value, 91.0);
    workspace
        .try_reserve(&plan.workspace_requirements(1, execution).unwrap())
        .unwrap();
    assert_eq!(workspace.lanes.len(), 3);
    let report = plan
        .integrate_into(
            &f,
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            execution,
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(calls.load(Ordering::Relaxed), 5 * SAMPLE_BLOCK + 1);
    assert_eq!(value, integrate(&plan, 0, &|x| Ok(x[0])).unwrap().0);

    let replicated = MonteCarloIntegrator::new(&bounds, options.with_replicates(9)).unwrap();
    let workspace = replicated.workspace(1, execution).unwrap();
    assert_eq!(workspace.children.len(), 9);
    assert_eq!(workspace.lanes.len(), 3);
    let auto = MonteCarloIntegrator::new(
        &bounds,
        options.with_execution_policy(ExecutionPolicy::Auto),
    )
    .unwrap();
    assert_eq!(
        auto.workspace(1, ExecutionContext::sequential_native())
            .unwrap()
            .lanes
            .len(),
        1
    );
}

#[test]
fn single_run_budget_stop_publishes_with_its_termination() {
    let plan = plan(1.0, 64);
    let root = Philox4x64::from_seed(42);
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut value = 91.0;
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded().with_limit(
        scioxide_core::BudgetKind::Sample,
        Budget::try_from(5).unwrap(),
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            crate::workspace::identity(plan.algorithm_id(), 0),
            &mut budgets,
        )
        .unwrap();
    let report = plan
        .integrate_into(
            |x: &[f64]| Ok(x[0]),
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Sample)
    );
    assert_eq!(report.evaluations(), 5);
    assert_ne!(value, 91.0);
    assert_eq!(workspace.statistics()[0].clone().unwrap().samples(), 5);
}

#[test]
fn block_prefixes_charge_exact_attempts_for_every_budget_and_failure() {
    use super::kernel::{Job, Lane, Schedule, Summary, sample};
    use scioxide_core::{BudgetKind, EvaluationRole, SolveBudgets, SolveState};
    #[derive(Default)]
    struct Count(usize);
    impl Summary for Count {
        fn clear(&mut self) {
            self.0 = 0;
        }
    }
    let root = Philox4x64::from_seed(42);
    let source = RandomStreams::new(&root, 0);
    let proposal = UniformProposal::new(2);
    for prefix in [SAMPLE_BLOCK - 1, SAMPLE_BLOCK, SAMPLE_BLOCK + 1] {
        for kind in [
            BudgetKind::Sample,
            BudgetKind::Evaluation,
            BudgetKind::Iteration,
            BudgetKind::Proposal,
        ] {
            for fail in [false, true] {
                for proposal in [None, Some(&proposal)] {
                    let attempts = AtomicUsize::new(0);
                    let function = |_: &[f64]| {
                        let index = attempts.fetch_add(1, Ordering::Relaxed);
                        if fail && index == prefix - 1 {
                            Err(SciOxideError::invalid_input(
                                scioxide_core::argument!("test"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "scripted callback failure",
                            ))
                        } else {
                            Ok(1.0)
                        }
                    };
                    let job = Job {
                        source: &source,
                        proposal,
                        function: &function,
                        dimension: 2,
                        uniforms: 2,
                    };
                    let mut lanes = [Lane::allocate(2, Count::default()).unwrap()];
                    let mut state = SolveState::new();
                    let amount = if kind == BudgetKind::Evaluation {
                        prefix * 3 + 2
                    } else {
                        prefix
                    };
                    let mut budgets = SolveBudgets::unbounded()
                        .with_limit(kind, Budget::try_from(amount).unwrap());
                    let mut context = state
                        .begin_solve(
                            ExecutionContext::sequential(),
                            crate::workspace::identity(
                                scioxide_core::algorithm_id!("integrate", "batch_test", 1),
                                0,
                            ),
                            &mut budgets,
                        )
                        .unwrap();
                    let result = sample(
                        &job,
                        Schedule::uniform(0, 2 * SAMPLE_BLOCK).unwrap(),
                        &mut lanes,
                        ExecutionPolicy::Sequential,
                        &mut context,
                        &|count: &mut Count, _, _: &f64| {
                            count.0 += 1;
                            Ok(())
                        },
                        &mut |_| Ok(()),
                    );
                    if fail {
                        assert!(result.is_err());
                    } else {
                        let progress = result.unwrap();
                        assert_eq!(progress.observations, prefix);
                        assert_eq!(
                            progress.stop,
                            Some(IntegrationTermination::BudgetLimit(kind))
                        );
                    }
                    let counts = context.counts().own();
                    assert_eq!(counts.iterations(), prefix);
                    assert_eq!(counts.proposals(), prefix);
                    assert_eq!(counts.provider(EvaluationRole::Sample), prefix * 2);
                    assert_eq!(counts.provider(EvaluationRole::Function), prefix);
                    assert_eq!(context.budget_report(BudgetKind::Sample).consumed(), prefix);
                }
            }
        }
    }
}

#[test]
fn prepared_half_open_mapping_covers_native_and_extended_boundaries() {
    fn check<S: scioxide_core::RealScalar>() {
        use crate::geometry::IntervalGeometry;
        let tiny = S::MIN_POSITIVE * S::EPSILON;
        let odd = S::ONE.next_after(S::TWO);
        for (lower, upper) in [
            (S::ZERO, S::ONE),
            (S::ONE, S::ONE.next_after(S::TWO)),
            (odd, odd.next_after(S::TWO)),
            (S::ZERO, tiny),
            (-S::MAX, S::MAX),
        ] {
            let geometry =
                IntervalGeometry::prepare_half_open(Interval::new(lower, upper).unwrap())
                    .unwrap()
                    .unwrap();
            for u in [S::ZERO, S::HALF, S::ONE.next_after(S::ZERO)] {
                let x = geometry.half_open_point(u).unwrap();
                assert!(x.is_finite() && x >= lower && x < upper);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn deadline_between_blocks_publishes_only_the_completed_prefix() {
    use core::time::Duration;
    use scioxide_core::{Clock, ClockBinding, Deadline, MonotonicInstant};
    struct ClockState(AtomicUsize);
    impl Clock for ClockState {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(self.0.load(Ordering::Relaxed) as u64),
            ))
        }
    }
    let clock = ClockState(AtomicUsize::new(0));
    let binding = ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &binding,
        MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
    );
    let calls = AtomicUsize::new(0);
    let root = Philox4x64::from_seed(42);
    let plan = plan(1.0, 2 * SAMPLE_BLOCK);
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut out = 91.0;
    let report = plan
        .integrate_into(
            |_: &[f64]| {
                if calls.fetch_add(1, Ordering::Relaxed) + 1 == SAMPLE_BLOCK {
                    clock.0.store(1, Ordering::Relaxed);
                }
                Ok(1.0)
            },
            RandomStreams::new(&root, 0),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential()
                .with_deadline(deadline)
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        report.termination(),
        IntegrationTermination::ExecutionStopped(_)
    ));
    assert_eq!(report.evaluations(), SAMPLE_BLOCK);
    assert_eq!(
        workspace.statistics()[0].clone().unwrap().samples(),
        SAMPLE_BLOCK
    );
    assert_eq!(out, 1.0);
}

#[test]
fn failing_uniform_in_a_later_block_is_counted_without_a_callback() {
    use super::kernel::{Job, Lane, Schedule, sample};
    use super::workspace::Block;
    use scioxide_core::{EvaluationRole, SolveBudgets, SolveState};
    struct Broken;
    impl Source<f64> for Broken {
        type Block = ();
        const INDEPENDENT: bool = true;
        fn check(&self, _: usize, _: usize, _: usize) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn open(&self, _: usize, _: u64) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn fill<'a>(
            &self,
            _: &mut (),
            _: usize,
            index: usize,
            uniforms: &'a mut [f64],
            draws: &mut usize,
        ) -> Result<ProposalUniforms<'a, f64>, SciOxideError> {
            for (axis, uniform) in uniforms.iter_mut().enumerate() {
                *draws += 1;
                if index == SAMPLE_BLOCK && axis == 1 {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Callback,
                        "scripted draw failure",
                    ));
                }
                *uniform = 0.25;
            }
            ProposalUniforms::new(uniforms)
        }
    }
    let proposal = UniformProposal::new(2);
    let f = |_: &[f64]| Ok(1.0);
    let job = Job {
        source: &Broken,
        proposal: Some(&proposal),
        function: &f,
        dimension: 2,
        uniforms: 2,
    };
    let mut lanes = [Lane::allocate(2, Block::allocate(1, true).unwrap()).unwrap()];
    let mut total = Block::allocate(1, false).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            crate::workspace::identity(
                scioxide_core::algorithm_id!("integrate", "batch_failure", 1),
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    assert!(
        sample(
            &job,
            Schedule::uniform(0, SAMPLE_BLOCK + 5).unwrap(),
            &mut lanes,
            ExecutionPolicy::Sequential,
            &mut context,
            &|block: &mut Block<f64>, draw, value: &f64| block.observe::<Broken, _>(
                &draw,
                value,
                Scaled::from_value(1.0)?,
                Some(1.0),
            ),
            &mut |block| total.merge(block)
        )
        .is_err()
    );
    let counts = context.counts().own();
    assert_eq!(counts.iterations(), SAMPLE_BLOCK + 1);
    assert_eq!(counts.proposals(), SAMPLE_BLOCK + 1);
    assert_eq!(
        counts.provider(EvaluationRole::Sample),
        2 * SAMPLE_BLOCK + 2
    );
    assert_eq!(counts.provider(EvaluationRole::Function), SAMPLE_BLOCK);
    assert_eq!(total.observations(), SAMPLE_BLOCK);
}

#[test]
fn miser_pilot_capacity_is_checked_before_work_and_larger_storage_is_reusable() {
    use scioxide_core::AttemptOutcome;
    let make = |pilot| {
        MiserTrainer::new(
            &[Interval::new(0.0_f64, 1.0).unwrap()],
            MiserOptions::default()
                .with_pilot_samples(Budget::try_from(pilot).unwrap())
                .with_max_leaves(Budget::try_from(2).unwrap())
                .with_evaluation(
                    MonteCarloOptions::default().with_samples(Budget::try_from(128).unwrap()),
                ),
        )
        .unwrap()
    };
    let small = make(64);
    let large = make(128);
    let root = Philox4x64::from_seed(42);
    let calls = AtomicUsize::new(0);
    let f = |x: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x[0])
    };
    let mut workspace = small
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut strata = large.initial_strata().unwrap();
    let original = strata.clone();
    assert!(
        large
            .train_into(
                &f,
                RandomStreams::new(&root, 0),
                &mut strata,
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(strata, original);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let mut workspace = large
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let _ = small
        .train_into(
            &f,
            RandomStreams::new(&root, 0),
            &mut strata,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let (expected, _) = small.train(&f, RandomStreams::new(&root, 0)).unwrap();
    assert_eq!(strata, expected);
}

#[test]
fn welch_native_promotion_preserves_extreme_variances_and_degrees() {
    fn check<S: scioxide_core::RealScalar>() {
        for value in [
            Scaled::from_value(S::MAX).unwrap(),
            Scaled::from_value(S::MIN_POSITIVE)
                .unwrap()
                .shifted(-i64::from(S::MANTISSA_DIGITS))
                .unwrap(),
        ] {
            let (total, degrees) =
                welch_satterthwaite([Ok((value, 11)), Ok((value, 11))], 20).unwrap();
            assert!(
                (total.div(value).unwrap().round().unwrap() - S::TWO).abs()
                    <= S::EPSILON * S::try_from_f64_lossy(8.0).unwrap()
            );
            let expected = S::try_from_f64_lossy(20.0).unwrap();
            assert!(
                (degrees - expected).abs()
                    <= expected * S::EPSILON * S::try_from_f64_lossy(8.0).unwrap()
            );
        }
    }
    check::<f32>();
    check::<f64>();
}
