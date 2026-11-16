//! Analytic importance tests and public proposal, source and replicate contracts.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;

use scioxide_core::*;
use scioxide_integrate::*;
use scioxide_random::{BitGenerator, Philox4x64, Rng, SeedableRng, SplittableRng};
use scioxide_stats::MeanIntervalMethod;

fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(590),
        scioxide_core::algorithm_id!("integrate", "proposal-test", 1),
        0,
        0,
    );
    f(&mut state.begin_solve(execution, id, &mut budgets).unwrap())
}
fn options(samples: usize) -> MonteCarloOptions {
    MonteCarloOptions::default().with_samples(Budget::try_from(samples).unwrap())
}
fn unit<S: RealScalar>() -> [Interval<S>; 1] {
    [Interval::new(S::ZERO, S::ONE).unwrap()]
}

#[test]
fn admitted_uniforms_cover_both_precisions_and_reject_provider_faults() {
    fn check<S: RealScalar>() {
        let valid = [S::ZERO, -S::ZERO, S::HALF, S::ONE.next_after(S::ZERO)];
        let admitted = ProposalUniforms::new(&valid).unwrap();
        assert_eq!(admitted.as_slice(), valid);
        let mut point = [S::NAN; 4];
        assert_eq!(
            UniformProposal::new(4)
                .sample(admitted, &mut point)
                .unwrap(),
            S::ONE
        );
        assert_eq!(point, valid);
        for invalid in [-S::EPSILON, S::ONE, S::INFINITY, S::NAN] {
            assert!(ProposalUniforms::new(&[invalid]).is_err());
        }
    }
    check::<f32>();
    check::<f64>();
}

fn provider(context: &SolveContext<'_, '_>, role: EvaluationRole) -> usize {
    context.counts().total().unwrap().provider(role)
}

// q(x) = (1+2x)/2, Q^-1(u) = 4u/(sqrt(1+8u)+1).
// Positive support everywhere; stable inverse avoids cancellation near zero.
struct LinearProposal;
impl<S: RealScalar> MonteCarloProposal<S> for LinearProposal {
    fn dimension(&self) -> usize {
        1
    }
    fn uniform_count(&self) -> usize {
        1
    }
    fn sample(&self, u: ProposalUniforms<'_, S>, x: &mut [S]) -> Result<S, SciOxideError> {
        let u = u.as_slice();
        let two = S::TWO;
        let four = two + two;
        x[0] = four * u[0] / ((S::ONE + four * two * u[0]).sqrt() + S::ONE);
        Ok((S::ONE + two * x[0]) / two)
    }
}

/// Every substream replays the same scripted uniforms from its start.
#[derive(Clone)]
struct ScriptedRng {
    points: [f64; 4],
    index: usize,
    fail_at: Option<usize>,
}
fn scripted(points: [f64; 4]) -> ScriptedRng {
    ScriptedRng {
        points,
        index: 0,
        fail_at: None,
    }
}
impl BitGenerator for ScriptedRng {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(0)
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Ok(0)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        output.fill(0);
        Ok(())
    }
}
impl Rng<f64> for ScriptedRng {
    fn next_float(&mut self) -> Result<f64, SciOxideError> {
        if self.fail_at == Some(self.index) {
            return Err(SciOxideError::external("scripted RNG failure"));
        }
        let x = self.points[self.index % 4];
        self.index += 1;
        Ok(x)
    }
}
impl SplittableRng for ScriptedRng {
    fn split(&self, _: u64) -> Result<Self, SciOxideError> {
        Ok(Self {
            index: 0,
            ..self.clone()
        })
    }
    fn substream(&self, _: u64, _: u64) -> Result<Self, SciOxideError> {
        Ok(Self {
            index: 0,
            ..self.clone()
        })
    }
}

#[test]
fn source_admission_finishes_the_point_and_preserves_rng_failure_precedence() {
    let plan = MonteCarloIntegrator::new(&[unit::<f64>()[0]; 2], options(2)).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    for (invalid, fail_at) in [(f64::NAN, None), (1.0, None), (f64::NAN, Some(1))] {
        let mut root = scripted([invalid, 0.5, 0.0, 0.0]);
        root.fail_at = fail_at;
        let mut value = 91.0;
        run(
            ExecutionContext::sequential_native(),
            SolveBudgets::unbounded(),
            |ctx| {
                let error = plan
                    .integrate_into(
                        |_: &[f64]| Ok(1.0),
                        RandomStreams::new(&root, 0),
                        &mut value,
                        &mut workspace,
                        &mut *ctx,
                    )
                    .unwrap_err();
                if fail_at.is_some() {
                    assert_eq!(error, SciOxideError::external("scripted RNG failure"));
                }
                assert_eq!(provider(ctx, EvaluationRole::Sample), 2);
                assert_eq!(provider(ctx, EvaluationRole::Function), 0);
            },
        );
        assert_eq!(value, 91.0);
    }
}

#[test]
fn unit_box_mapping_preserves_each_lower_endpoint_zero_sign() {
    for lower in [[0.0_f64, 0.0], [-0.0, 0.0], [0.0, -0.0]] {
        let bounds = lower.map(|x| Interval::new(x, 1.0).unwrap());
        let plan = MonteCarloIntegrator::new(&bounds, options(2)).unwrap();
        let mut workspace = plan
            .workspace(1, ExecutionContext::sequential_native())
            .unwrap();
        let root = scripted([-0.0; 4]);
        let mut value = 0.0;
        let report = plan
            .integrate_into(
                |x: &[f64]| {
                    for (&actual, expected) in x.iter().zip(lower) {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                    Ok(1.0)
                },
                RandomStreams::new(&root, 0),
                &mut value,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(value, 1.0);
    }
}

#[test]
fn importance_integrals_with_complex_and_vector_values_in_both_precisions() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let plan =
                MonteCarloIntegrator::importance(&unit::<$s>(), LinearProposal, options(128))
                    .unwrap();
            let root = Philox4x64::from_seed(42);
            // q-proportional integrands make every observation the exact integral.
            let mut complex = Complex::new(0.0 as $s, 0.0);
            let mut workspace = plan
                .workspace(2, ExecutionContext::sequential_native())
                .unwrap();
            let report = plan
                .integrate_into(
                    |x: &[$s]| {
                        let q = (1.0 + 2.0 * x[0]) / 2.0;
                        Ok(Complex::new(q, -2.0 * q))
                    },
                    RandomStreams::new(&root, 0),
                    &mut complex,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                IntegrationTermination::SampleBudgetCompleted
            );
            assert!((complex.re - 1.0).abs() < $tol);
            assert!((complex.im + 2.0).abs() < $tol);
            let mut statistics = workspace.statistics()[0].clone().unwrap();
            assert!(statistics.standard_error() < $tol);
            assert_eq!(statistics.basis(), VarianceBasis::Observations);
            let ess = statistics.effective_sample_size().unwrap();
            assert!(ess > 90.0 && ess < 128.0);
            assert_eq!(
                statistics.interval_report().unwrap().method(),
                MeanIntervalMethod::StudentT
            );
            assert_eq!(
                statistics.interval_report().unwrap().degrees_of_freedom(),
                Some(127.0)
            );
            assert_eq!(statistics.degrees_of_freedom(), Some(127.0));
            let mut vector = [0.0 as $s; 3];
            let mut workspace = plan
                .workspace(3, ExecutionContext::sequential_native())
                .unwrap();
            let _ = plan
                .integrate_into(
                    |x: &[$s]| {
                        let q = (1.0 + 2.0 * x[0]) / 2.0;
                        Ok([q, 3.0 * q, -q])
                    },
                    RandomStreams::new(&root, 0),
                    &mut vector,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(vector[0], complex.re);
            assert!((vector[1] - 3.0).abs() < 3.0 * $tol);
            assert!((vector[2] + 1.0).abs() < $tol);
        }};
    }
    check!(f32, 5e-7);
    check!(f64, 2e-15);
}

#[test]
fn independent_weight_moments_ess_and_student_reference() {
    let root = scripted([0.1, 0.3, 0.6, 0.9]);
    let values: Vec<_> = root
        .points
        .iter()
        .map(|u| {
            let x = 4.0 * u / ((1.0 + 8.0 * u).sqrt() + 1.0);
            (x * x / ((1.0 + 2.0 * x) / 2.0), 2.0 / (1.0 + 2.0 * x))
        })
        .collect();
    let mean = values.iter().map(|(y, _)| y).sum::<f64>() / 4.0;
    let variance = values.iter().map(|(y, _)| (y - mean).powi(2)).sum::<f64>() / 12.0;
    let ess = values.iter().map(|(_, w)| w).sum::<f64>().powi(2)
        / values.iter().map(|(_, w)| w * w).sum::<f64>();
    let plan = MonteCarloIntegrator::importance(&unit(), LinearProposal, options(4)).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut value = 0.0;
    let _ = plan
        .integrate_into(
            |x: &[f64]| Ok(x[0] * x[0]),
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let mut s = workspace.statistics()[0].clone().unwrap();
    assert!((value - mean).abs() < 2e-15);
    assert!((s.estimator_variance().round().unwrap() - variance).abs() < 2e-15);
    assert!((s.effective_sample_size().unwrap() - ess).abs() < 2e-14);
    // Independent tabulated t_.975,3 = 3.182446305284263 (NIST/R standard table).
    assert!(
        (s.confidence_interval().unwrap().upper() - mean - 3.182446305284263 * variance.sqrt())
            .abs()
            < 2e-10
    );
    let child = workspace.children().next().unwrap();
    assert_eq!(child.values(), Some(&[value][..]));
    assert_eq!(child.report().unwrap().evaluations(), 4);
    assert_eq!(child.effective_sample_size(), s.effective_sample_size());
}

struct InvalidProposal {
    density: f64,
    coordinate: f64,
}
impl MonteCarloProposal<f64> for InvalidProposal {
    fn dimension(&self) -> usize {
        1
    }
    fn uniform_count(&self) -> usize {
        1
    }
    fn sample(&self, _: ProposalUniforms<'_, f64>, x: &mut [f64]) -> Result<f64, SciOxideError> {
        x[0] = self.coordinate;
        Ok(self.density)
    }
}
#[test]
fn invalid_support_density_and_points_are_transactional_and_charge_draws() {
    let root = scripted([0.5; 4]);
    for (density, coordinate) in [
        (0.0, 0.5),
        (-1.0, 0.5),
        (f64::NAN, 0.5),
        (f64::INFINITY, 0.5),
        (1.0, -0.1),
        (1.0, 1.0),
        (1.0, f64::NAN),
    ] {
        let plan = MonteCarloIntegrator::importance(
            &unit(),
            InvalidProposal {
                density,
                coordinate,
            },
            options(4),
        )
        .unwrap();
        let mut workspace = plan
            .workspace(1, ExecutionContext::sequential_native())
            .unwrap();
        let mut output = 91.0;
        run(
            ExecutionContext::sequential(),
            SolveBudgets::unbounded(),
            |ctx| {
                let result = plan.integrate_into(
                    |_: &[f64]| -> Result<f64, SciOxideError> { unreachable!() },
                    RandomStreams::new(&root, 0),
                    &mut output,
                    &mut workspace,
                    &mut *ctx,
                );
                assert!(result.is_err());
                assert_eq!(provider(ctx, EvaluationRole::Sample), 1);
                assert_eq!(provider(ctx, EvaluationRole::Function), 0);
            },
        );
        assert_eq!(output, 91.0);
        assert!(workspace.statistics()[0].is_none());
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
}

struct ReorderedExecutor {
    workers: usize,
    threaded: bool,
    omit: bool,
}
impl Executor for ReorderedExecutor {
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
                for handle in handles {
                    handle.join().unwrap()?;
                }
                Ok::<(), SciOxideError>(())
            })?;
        } else {
            for p in &partitions {
                dispatch(p)?;
            }
        }
        // An omitted callback with a nominally complete report is deliberately malformed.
        SequentialExecutor.execute(ctx, plan, &|_, (), _| Ok(()))
    }
}

#[test]
fn replicate_streams_are_worker_and_dispatch_independent() {
    let root = Philox4x64::from_seed(590);
    let f = |x: &[f64]| Ok([x[0] * x[0], -x[0]]);
    let reference_plan =
        MonteCarloIntegrator::importance(&unit(), LinearProposal, options(16).with_replicates(4))
            .unwrap();
    let mut reference = [0.0; 2];
    let mut workspace = reference_plan
        .workspace(2, ExecutionContext::sequential_native())
        .unwrap();
    let _ = reference_plan
        .integrate_into(
            &f,
            RandomStreams::new(&root, 3),
            &mut reference,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let reference_statistics = workspace.statistics()[0].clone().unwrap();
    assert_eq!(reference_statistics.basis(), VarianceBasis::Replicates);
    assert_eq!(reference_statistics.units(), 4);
    assert_eq!(reference_statistics.samples(), 64);
    assert_eq!(reference_statistics.degrees_of_freedom(), Some(3.0));
    let replicates: Vec<f64> = workspace
        .children()
        .map(|c| c.values().unwrap()[0])
        .collect();
    assert!((replicates.iter().sum::<f64>() / 4.0 - reference[0]).abs() < 1e-15);
    assert_ne!(replicates[0], replicates[1]);
    for workers in [1, 2, 4] {
        for threaded in [false, true] {
            let executor = ReorderedExecutor {
                workers,
                threaded,
                omit: false,
            };
            let plan =
                MonteCarloIntegrator::importance(
                    &unit(),
                    LinearProposal,
                    options(16).with_replicates(4).with_execution_policy(
                        ExecutionPolicy::Parallel(NonZeroUsize::new(workers).unwrap()),
                    ),
                )
                .unwrap();
            let mut workspace = plan.workspace(2, ExecutionContext::new(&executor)).unwrap();
            let mut output = [0.0; 2];
            run(
                ExecutionContext::new(&executor),
                SolveBudgets::unbounded(),
                |ctx| {
                    let _ = plan
                        .integrate_into(
                            &f,
                            RandomStreams::new(&root, 3),
                            &mut output,
                            &mut workspace,
                            &mut *ctx,
                        )
                        .unwrap();
                    assert_eq!(provider(ctx, EvaluationRole::Sample), 64);
                    assert_eq!(provider(ctx, EvaluationRole::Function), 64);
                },
            );
            assert_eq!(output, reference);
            assert_eq!(
                workspace.statistics()[0].clone().unwrap(),
                reference_statistics
            );
            // Shared ancestor budgets are divided by replicate index; the aggregate
            // then stops, while every replicate's progress stays readable.
            let mut stopped = [91.0; 2];
            run(
                ExecutionContext::new(&executor),
                SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Sample, Budget::try_from(31).unwrap()),
                |ctx| {
                    let error = plan
                        .integrate_into(
                            &f,
                            RandomStreams::new(&root, 3),
                            &mut stopped,
                            &mut workspace,
                            &mut *ctx,
                        )
                        .unwrap_err();
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::ConvergenceFailure { .. }
                    ));
                    assert_eq!(provider(ctx, EvaluationRole::Function), 31);
                },
            );
            assert_eq!(stopped, [91.0; 2]);
            assert!(workspace.statistics().iter().all(Option::is_none));
            assert_eq!(
                workspace
                    .children()
                    .map(|c| c.report().unwrap().evaluations())
                    .collect::<Vec<_>>(),
                [8, 8, 8, 7]
            );
            assert!(
                workspace
                    .children()
                    .all(|c| c.report().unwrap().termination()
                        == IntegrationTermination::BudgetLimit(BudgetKind::Sample))
            );
        }
    }
    assert_eq!(root.position(), Some(0));
}

#[test]
fn replicate_omission_preserves_output_and_reconciles_work() {
    let executor = ReorderedExecutor {
        workers: 2,
        threaded: false,
        omit: true,
    };
    let plan = MonteCarloIntegrator::new(
        &unit(),
        options(4)
            .with_replicates(2)
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap())),
    )
    .unwrap();
    let mut workspace = plan.workspace(1, ExecutionContext::new(&executor)).unwrap();
    let mut output = 91.0;
    let root = Philox4x64::from_seed(590);
    run(
        ExecutionContext::new(&executor),
        SolveBudgets::unbounded(),
        |ctx| {
            assert!(
                plan.integrate_into(
                    |x: &[f64]| Ok(x[0]),
                    RandomStreams::new(&root, 0),
                    &mut output,
                    &mut workspace,
                    &mut *ctx,
                )
                .is_err()
            );
            assert_eq!(provider(ctx, EvaluationRole::Function), 4);
        },
    );
    assert_eq!(output, 91.0);
    assert!(workspace.statistics()[0].is_none());
}

#[test]
fn cancellation_publishes_observed_prefix_and_failures_publish_nothing() {
    let cancellation = CancellationToken::new();
    let plan = MonteCarloIntegrator::new(&unit(), options(5)).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let root = scripted([0.5; 4]);
    let mut output = 91.0;
    run(
        ExecutionContext::sequential().with_cancellation(&cancellation),
        SolveBudgets::unbounded(),
        |ctx| {
            let report = plan
                .integrate_into(
                    |x: &[f64]| {
                        cancellation.cancel();
                        Ok(x[0])
                    },
                    RandomStreams::new(&root, 0),
                    &mut output,
                    &mut workspace,
                    ctx,
                )
                .unwrap();
            assert_eq!(report.evaluations(), 1);
            assert_eq!(
                report.termination(),
                IntegrationTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
            );
        },
    );
    assert_eq!(output, 0.5);
    assert!(workspace.statistics()[0].is_none());
    run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded(),
        |ctx| {
            assert!(
                plan.integrate_into(
                    |_: &[f64]| -> Result<f64, SciOxideError> {
                        Err(SciOxideError::execution_failure(
                            None,
                            scioxide_core::ExecutionFailureReason::ProviderFailure,
                            "callback failure",
                        ))
                    },
                    RandomStreams::new(&root, 0),
                    &mut output,
                    &mut workspace,
                    &mut *ctx,
                )
                .is_err()
            );
            assert_eq!(provider(ctx, EvaluationRole::Function), 1);
        },
    );
    assert_eq!(output, 0.5);
}

#[test]
fn extreme_importance_weights_use_scaled_arithmetic() {
    // A proposal can have tiny density locally and compensate elsewhere. These
    // prescribed points exercise arithmetic independently of its global sampler.
    let plan = MonteCarloIntegrator::importance(
        &[Interval::new(0.0, 1e-100).unwrap()],
        InvalidProposal {
            density: 1e-300,
            coordinate: 0.5,
        },
        options(4),
    )
    .unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut value = 0.0;
    let _ = plan
        .integrate_into(
            |_: &[f64]| Ok(1e-300),
            RandomStreams::new(&scripted([0.5; 4]), 0),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!((value / 1e-100 - 1.0).abs() < 1e-14);
    let mut stats = workspace.statistics()[0].clone().unwrap();
    assert!((stats.effective_sample_size().unwrap() - 4.0).abs() < 1e-14);
    assert!(stats.interval_report().unwrap().is_degenerate());
}

#[test]
fn invalid_options_are_rejected_by_plan_construction() {
    for options in [
        options(1),
        options(4).with_replicates(0),
        options(4).with_interval_method(MeanIntervalMethod::NormalKnownVariance),
    ] {
        assert!(matches!(
            MonteCarloIntegrator::new(&unit::<f64>(), options)
                .unwrap_err()
                .view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
    }
}
