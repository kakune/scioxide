//! Enumerated mathematical oracles, transaction failures and reproducible execution.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::*;
use scioxide_random::{
    BitGenerator, OutputPolicy, Philox4x64, Rng, SamplingPolicy, SeedableRng, SplitMix64,
};
use scioxide_stats::*;
use std::{
    num::NonZeroUsize,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(550),
        scioxide_core::algorithm_id!("stats", "resampling-test", 1),
        0,
        0,
    );
    f(&mut state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap())
}
fn sequential<T>(f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    run(ExecutionContext::sequential(), SolveBudgets::unbounded(), f)
}
fn boot(design: BootstrapDesign, method: BootstrapMethod) -> ResamplingMethod {
    ResamplingMethod::Bootstrap {
        design,
        method,
        confidence: ConfidenceProbability::new(0.8).unwrap(),
    }
}
fn close<S: RealScalar>(a: S, b: f64, tol: f64) {
    assert!(
        (a.try_to_f64_exact().unwrap() - b).abs() <= tol * b.abs().max(1.0),
        "{a:?} != {b}"
    );
}
fn average<S: RealScalar>(data: &[S], sample: ResampledIndices<'_>) -> Result<S, SciOxideError> {
    let sum = sample.indices().iter().fold(S::ZERO, |a, &i| a + data[i]);
    Ok(sum / S::try_from_usize_exact(sample.indices().len())?)
}
fn test_result<S: RealScalar>(result: &ResamplingResult<S>) -> ResamplingTest {
    let Some(ResamplingInference::Test(t)) = result.inference() else {
        panic!("missing test: {result:?}")
    };
    t
}
fn interval<S: RealScalar>(result: &ResamplingResult<S>) -> BootstrapInterval<S> {
    let Some(ResamplingInference::Bootstrap(t)) = result.inference() else {
        panic!("missing interval: {result:?}")
    };
    t
}
struct Scripted {
    words: Vec<u64>,
    position: usize,
}
impl Scripted {
    fn indices(indices: impl IntoIterator<Item = usize>, n: usize) -> Self {
        Self {
            words: indices
                .into_iter()
                .map(|i| (((2 * i + 1) as u128 * (1_u128 << 63)) / n as u128) as u64)
                .collect(),
            position: 0,
        }
    }
}
impl BitGenerator for Scripted {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        let word = self.words.get(self.position).copied().ok_or_else(|| {
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("script"),
                self.position,
                self.words.len(),
            )
        })?;
        self.position += 1;
        Ok(word)
    }
    // The high half keeps the scripted index midpoint, so 32-bit index draws
    // decode the same index as 64-bit ones.
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok((self.next_u64()? >> 32) as u32)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        for c in out.chunks_mut(8) {
            let v = self.next_u64()?.to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
        Ok(())
    }
}
#[test]
fn enumerated_bootstrap_and_analytic_jackknife_both_precisions() {
    fn check<S: RealScalar>(tol: f64) {
        let data: Vec<S> = [1.0, 2.0, 4.0]
            .map(|x| S::try_from_f64_lossy(x).unwrap())
            .into();
        let all: Vec<_> = (0..3)
            .flat_map(|a| (0..3).flat_map(move |b| (0..3).map(move |c| [a, b, c])))
            .collect();
        let mut oracle: Vec<_> = all
            .iter()
            .map(|r| r.iter().map(|&i| [1.0, 2.0, 4.0][i]).sum::<f64>() / 3.0)
            .collect();
        let center = oracle.iter().sum::<f64>() / 27.0;
        let variance = oracle.iter().map(|x| (x - center).powi(2)).sum::<f64>() / 26.0;
        oracle.sort_by(f64::total_cmp);
        let lower = oracle[2] * 0.4 + oracle[3] * 0.6;
        let upper = oracle[23] * 0.6 + oracle[24] * 0.4;
        for method in [BootstrapMethod::Percentile, BootstrapMethod::Basic] {
            let plan = ResamplingPlan::new(
                &[3],
                boot(BootstrapDesign::Independent, method),
                ResamplingOptions::default()
                    .with_replicates(27)
                    .with_batch_size(4),
            )
            .unwrap();
            let mut rng = Scripted::indices(all.iter().flatten().copied(), 3);
            let output = sequential(|ctx| {
                evaluate(
                    &plan,
                    scioxide_stats::ResamplingProblem::Indexed(&|s| average(&data, s)),
                    scioxide_stats::ReplicateOptions::sequential(
                        &mut rng,
                        SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                    ),
                    ctx,
                )
            })
            .unwrap();
            let r = interval(output.report());
            let (lo, hi) = if method == BootstrapMethod::Basic {
                (2.0 * 7.0 / 3.0 - upper, 2.0 * 7.0 / 3.0 - lower)
            } else {
                (lower, upper)
            };
            close(r.bounds.lower(), lo, tol);
            close(r.bounds.upper(), hi, tol);
            close(r.standard_error, variance.sqrt(), tol);
            assert_eq!(rng.position, 81);
            assert_eq!(output.distribution().len(), 27);
        }
        let plan = ResamplingPlan::new(
            &[3],
            ResamplingMethod::Jackknife {
                design: BootstrapDesign::Independent,
            },
            ResamplingOptions::default()
                .with_replicates(3)
                .with_batch_size(2),
        )
        .unwrap();
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|s| average(&data, s)),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut scioxide_random::SplitMix64::from_seed(0),
                    scioxide_random::SamplingPolicy::advance_on_error(
                        scioxide_random::OutputPolicy::Incremental,
                    ),
                ),
                ctx,
            )
        })
        .unwrap();
        let Some(ResamplingInference::Jackknife(j)) = output.report().inference() else {
            panic!()
        };
        close(j.bias, 0.0, tol);
        close(j.bias_corrected, 7.0 / 3.0, tol);
        close(j.standard_error, (7.0_f64 / 9.0).sqrt(), tol);
        for (x, y) in output.distribution().iter().zip([3.0, 2.5, 1.5]) {
            close(*x, y, tol);
        }
    }
    check::<f64>(2e-14);
    check::<f32>(8e-7);
}
#[test]
fn exact_permutation_spaces_and_finite_sample_correction() {
    for (kind, expected) in [
        (PermutationKind::Independent, 6),
        (PermutationKind::PairedLabels, 4),
        (PermutationKind::Pairings, 4),
    ] {
        let seen = Mutex::new(Vec::new());
        let data = [1.0, 2.0, 3.0, 4.0];
        let statistic = |s: ResampledIndices<'_>| {
            seen.lock().unwrap().push(s.indices().to_vec());
            Ok(s.group(0)?.iter().map(|&i| data[i]).sum::<f64>() / 2.0
                - s.group(1)?.iter().map(|&i| data[i]).sum::<f64>() / 2.0)
        };
        let method = ResamplingMethod::Permutation {
            kind,
            mode: PermutationMode::Exact,
            alternative: TestAlternative::TwoSided,
        };
        let plan = ResamplingPlan::new(
            &[2, 2],
            method,
            ResamplingOptions::default()
                .with_replicates(expected)
                .with_batch_size(2),
        )
        .unwrap();
        let mut rng = Scripted::indices([], 2);
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&statistic),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        let t = test_result(output.report());
        assert!(t.exact());
        assert!(t.tail_uncertainty().is_none());
        assert_eq!(output.distribution().len(), expected);
        assert_eq!(rng.position, 0);
        let seen = seen.into_inner().unwrap();
        let distinct: std::collections::BTreeSet<_> = seen[1..].iter().collect();
        assert_eq!(distinct.len(), expected);
        assert!(distinct.contains(&seen[0]));
        for selection in &seen[1..] {
            let mut indices = selection.clone();
            indices.sort_unstable();
            assert_eq!(indices, [0, 1, 2, 3]);
        }
        if kind == PermutationKind::Independent {
            close(t.p_value(), 1.0 / 3.0, 1e-15);
        }
    }
    let plan = ResamplingPlan::new(
        &[2, 2],
        ResamplingMethod::Permutation {
            kind: PermutationKind::Independent,
            mode: PermutationMode::Randomized,
            alternative: TestAlternative::Less,
        },
        ResamplingOptions::default()
            .with_replicates(7)
            .with_batch_size(2),
    )
    .unwrap();
    let calls = AtomicUsize::new(0);
    let output = sequential(|ctx| {
        evaluate(
            &plan,
            scioxide_stats::ResamplingProblem::Indexed(&|_| {
                Ok(if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    -1.0
                } else {
                    1.0
                })
            }),
            scioxide_stats::ReplicateOptions::sequential(
                &mut SplitMix64::from_seed(2),
                SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
            ),
            ctx,
        )
    })
    .unwrap();
    let t = test_result(output.report());
    assert_eq!(t.p_value(), 1.0 / 8.0);
    assert_eq!(t.less_equal(), 0);
    assert!(t.tail_uncertainty().unwrap()[0].upper() > 0.0);
}
#[test]
fn paired_stratified_and_block_indices_preserve_the_design() {
    for design in [BootstrapDesign::Paired, BootstrapDesign::Stratified] {
        let plan = ResamplingPlan::<f64>::new(
            &[4, 4],
            boot(design, BootstrapMethod::Percentile),
            ResamplingOptions::default()
                .with_replicates(64)
                .with_batch_size(3),
        )
        .unwrap();
        let statistic = |s: ResampledIndices<'_>| {
            assert_eq!(s.group(0)?.len(), 4);
            assert_eq!(s.group(1)?.len(), 4);
            for (&a, &b) in s.group(0)?.iter().zip(s.group(1)?) {
                assert!(a < 4 && (4..8).contains(&b));
                if design == BootstrapDesign::Paired {
                    assert_eq!(a + 4, b);
                }
            }
            Ok(1.0)
        };
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&statistic),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(5),
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        assert!(interval(output.report()).degenerate);
    }
    for design in [
        BootstrapDesign::MovingBlock { length: 3 },
        BootstrapDesign::CircularBlock { length: 3 },
    ] {
        let plan = ResamplingPlan::<f64>::new(
            &[8],
            boot(design, BootstrapMethod::Basic),
            ResamplingOptions::default()
                .with_replicates(32)
                .with_batch_size(4),
        )
        .unwrap();
        let calls = AtomicUsize::new(0);
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|s| {
                    if calls.fetch_add(1, Ordering::Relaxed) > 0 {
                        for block in s.indices().chunks(3) {
                            for pair in block.windows(2) {
                                assert_eq!(pair[1], (pair[0] + 1) % 8);
                            }
                            if matches!(design, BootstrapDesign::MovingBlock { .. }) {
                                assert!(block[0] <= 5);
                            }
                        }
                    }
                    Ok(1.0)
                }),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(9),
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        assert_eq!(output.distribution().len(), 32);
    }
}
#[test]
fn bca_degeneracy_validation_and_nonfinite_callbacks() {
    for design in [
        BootstrapDesign::Independent,
        BootstrapDesign::Paired,
        BootstrapDesign::Stratified,
    ] {
        let plan = ResamplingPlan::<f64>::new(
            &[3, 3],
            boot(design, BootstrapMethod::Bca),
            ResamplingOptions::default()
                .with_replicates(20)
                .with_batch_size(2),
        )
        .unwrap();
        let err = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|_| Ok(4.0)),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(3),
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap_err();
        assert!(
            matches ! (& (err) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { .. }))
        );
    }
    assert!(
        ResamplingPlan::<f64>::new(
            &[3],
            boot(
                BootstrapDesign::MovingBlock { length: 2 },
                BootstrapMethod::Bca
            ),
            ResamplingOptions::default()
                .with_replicates(20)
                .with_batch_size(2)
        )
        .is_err()
    );
    assert!(
        ResamplingPlan::<f64>::new(
            &[3, 2],
            boot(BootstrapDesign::Paired, BootstrapMethod::Basic),
            ResamplingOptions::default()
                .with_replicates(20)
                .with_batch_size(2)
        )
        .is_err()
    );
    assert!(
        ResamplingPlan::<f64>::new(
            &[0],
            boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
            ResamplingOptions::default()
                .with_replicates(20)
                .with_batch_size(2)
        )
        .is_err()
    );
    assert!(
        ResamplingPlan::<f64>::new(
            &[100],
            ResamplingMethod::Permutation {
                kind: PermutationKind::Pairings,
                mode: PermutationMode::Exact,
                alternative: TestAlternative::Less
            },
            ResamplingOptions::default()
                .with_replicates(usize::MAX)
                .with_batch_size(1)
        )
        .is_err()
    );
    let plan = ResamplingPlan::<f64>::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(4)
            .with_batch_size(2),
    )
    .unwrap();
    let mut rng = SplitMix64::from_seed(7);
    let mut control = rng.clone();
    let err = sequential(|ctx| {
        evaluate(
            &plan,
            scioxide_stats::ResamplingProblem::Indexed(&|_| Ok(f64::NAN)),
            scioxide_stats::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::rollback(OutputPolicy::Incremental),
            ),
            ctx,
        )
    })
    .unwrap_err();
    assert!(
        matches ! (& (err) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    assert_eq!(rng.next_u64().unwrap(), control.next_u64().unwrap());
}
#[test]
fn output_and_rng_transactions_at_every_callback_boundary() {
    let plan = ResamplingPlan::<f64>::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    for fail_at in 0..=5 {
        for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            for rollback in [false, true] {
                let mut workspace = plan.workspace().unwrap();
                let mut out = [-7.0; 5];
                let mut rng = SplitMix64::from_seed(11);
                let mut original = rng.clone();
                let calls = AtomicUsize::new(0);
                let statistic = |s: ResampledIndices<'_>| {
                    let count = calls.fetch_add(1, Ordering::Relaxed);
                    if count == fail_at {
                        Err(SciOxideError::out_of_bounds(
                            scioxide_core::quantity!("injected"),
                            count,
                            5,
                        ))
                    } else {
                        Ok(s.indices().iter().sum::<usize>() as f64)
                    }
                };
                let result = sequential(|ctx| {
                    if rollback {
                        plan.evaluate_into(
                            ResamplingProblem::Indexed(&statistic),
                            scioxide_stats::ReplicateOptions::sequential(
                                &mut rng,
                                SamplingPolicy::rollback(output_policy),
                            ),
                            &mut out,
                            &mut workspace,
                            ctx,
                        )
                    } else {
                        plan.evaluate_into(
                            ResamplingProblem::Indexed(&statistic),
                            scioxide_stats::ReplicateOptions::sequential(
                                &mut rng,
                                SamplingPolicy::advance_on_error(output_policy),
                            ),
                            &mut out,
                            &mut workspace,
                            ctx,
                        )
                    }
                });
                assert!(result.is_err());
                assert_eq!(calls.load(Ordering::Relaxed), fail_at + 1);
                let published = if output_policy == OutputPolicy::Atomic {
                    0
                } else {
                    fail_at.saturating_sub(1)
                };
                assert!(out[..published].iter().all(|&v| v >= 0.0));
                assert!(out[published..].iter().all(|&v| v == -7.0));
                if rollback || fail_at == 0 {
                    assert_eq!(rng.next_u64().unwrap(), original.next_u64().unwrap());
                } else {
                    assert_ne!(rng.next_u64().unwrap(), original.next_u64().unwrap());
                }
            }
        }
    }
}
#[test]
fn budget_boundaries_stop_without_false_inference() {
    let plan = ResamplingPlan::<f64>::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Percentile),
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    for limit in 1..=11 {
        let calls = AtomicUsize::new(0);
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let output = run(ExecutionContext::sequential(), budgets, |ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(1.0)
                }),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(7),
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        assert_eq!(output.report().completed(), (limit - 1) / 2);
        assert_eq!(calls.load(Ordering::Relaxed), 1 + (limit - 1) / 2);
        assert_eq!(output.report().inference().is_some(), limit == 11);
        assert_eq!(output.distribution().len(), (limit - 1) / 2);
    }
    let mut options = ResamplingOptions::<f64>::default()
        .with_replicates(5)
        .with_batch_size(2);
    options = options.with_max_rng_calls(2);
    let plan = ResamplingPlan::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        options,
    )
    .unwrap();
    let output = sequential(|ctx| {
        evaluate(
            &plan,
            scioxide_stats::ResamplingProblem::Indexed(&|_| Ok(1.0)),
            scioxide_stats::ReplicateOptions::sequential(
                &mut SplitMix64::from_seed(3),
                SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
            ),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(
        output.report().termination(),
        StatsTermination::BudgetExhausted
    );
    assert_eq!(output.report().completed(), 0);
    options = options.with_max_workspace_bytes(plan.workspace_bytes().unwrap() - 1);
    assert!(
        ResamplingPlan::new(&[3], plan.method(), options)
            .unwrap()
            .workspace()
            .is_err()
    );
    options = options.with_max_workspace_bytes(options.max_workspace_bytes() + 1);
    assert!(
        ResamplingPlan::new(&[3], plan.method(), options)
            .unwrap()
            .workspace()
            .is_ok()
    );
}
struct ReverseExecutor {
    omit: bool,
}
impl Executor for ReverseExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(3).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let mut parts: Vec<_> = plan.partitions().collect();
        parts.reverse();
        if self.omit {
            parts.pop();
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = parts
                .iter()
                .map(|target| {
                    scope.spawn(move || {
                        SequentialExecutor.execute(ctx, plan, &|p, (), s| {
                            if &p == target { op(p, (), s) } else { Ok(()) }
                        })
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap()?;
            }
            Ok::<(), SciOxideError>(())
        })?;
        SequentialExecutor.execute(ctx, plan, &|_, (), _| Ok(()))
    }
}
#[test]
fn indexed_streams_match_across_batches_workers_and_budgets() {
    let root = Philox4x64::from_seed(550);
    let data = [1.0_f64, 2.0, 3.0, 8.0];
    for budget in [None, Some(45)] {
        let mut reference = None;
        for batch in [1, 3, 8] {
            for parallel in [false, true] {
                let executor = ReverseExecutor { omit: false };
                let execution = ExecutionContext::new(&executor);
                let plan = ResamplingPlan::new(
                    &[4],
                    boot(BootstrapDesign::Independent, BootstrapMethod::Bca),
                    ResamplingOptions::default()
                        .with_replicates(32)
                        .with_batch_size(batch),
                )
                .unwrap();
                let mut ws = plan.workspace().unwrap();
                let mut out = vec![-5.0; 32];
                let mut budgets = SolveBudgets::unbounded();
                if let Some(limit) = budget {
                    budgets = budgets
                        .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
                }
                let report = run(execution, budgets, |ctx| {
                    plan.evaluate_into(
                        ResamplingProblem::Indexed(&|s| average(&data, s)),
                        scioxide_stats::ReplicateOptions::indexed(
                            &root,
                            9,
                            if parallel {
                                ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap())
                            } else {
                                ExecutionPolicy::Sequential
                            },
                            OutputPolicy::Incremental,
                        ),
                        &mut out,
                        &mut ws,
                        ctx,
                    )
                })
                .unwrap();
                if let Some((expected, values)) = &reference {
                    assert_eq!(&report, expected);
                    assert_eq!(&out, values);
                } else {
                    reference = Some((report, out));
                }
            }
        }
    }
    let executor = ReverseExecutor { omit: true };
    let plan = ResamplingPlan::new(
        &[4],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(8)
            .with_batch_size(3),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut out = [-5.0; 8];
    let result = run(
        ExecutionContext::new(&executor),
        SolveBudgets::unbounded(),
        |ctx| {
            plan.evaluate_into(
                ResamplingProblem::Indexed(&|s| average(&data, s)),
                scioxide_stats::ReplicateOptions::indexed(
                    &root,
                    0,
                    ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap()),
                    OutputPolicy::Atomic,
                ),
                &mut out,
                &mut ws,
                ctx,
            )
        },
    );
    assert!(result.is_err());
    assert_eq!(out, [-5.0; 8]);
}
#[test]
fn monte_carlo_uses_same_statistic_and_checks_complete_sample_writes() {
    fn check<S: RealScalar>(tol: f64) {
        let observed = [S::ZERO; 2];
        let statistic = |data: &[S]| Ok((data[0] + data[1]) / (S::TWO));
        let sampler = |rng: &mut dyn Rng<S>, out: &mut [S]| {
            for x in out {
                *x = rng.next_float()?;
            }
            Ok(())
        };
        let plan = ResamplingPlan::new(
            &[2],
            ResamplingMethod::MonteCarlo {
                alternative: TestAlternative::Less,
            },
            ResamplingOptions::default()
                .with_replicates(99)
                .with_batch_size(4),
        )
        .unwrap();
        let result = sequential(|ctx| {
            evaluate(
                &plan,
                ResamplingProblem::MonteCarlo {
                    observed: &observed,
                    sampler: &sampler,
                    statistic: &statistic,
                },
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(3),
                    SamplingPolicy::rollback(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        close(test_result(result.report()).p_value(), 0.01, tol);
        assert!(
            result
                .distribution()
                .iter()
                .all(|&x| x > S::ZERO && x < S::ONE)
        );
        let err = sequential(|ctx| {
            evaluate(
                &plan,
                ResamplingProblem::MonteCarlo {
                    observed: &observed,
                    sampler: &|_, out| {
                        out[0] = S::ZERO;
                        Ok(())
                    },
                    statistic: &statistic,
                },
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(3),
                    SamplingPolicy::rollback(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap_err();
        assert!(
            matches ! (& (err) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
    }
    check::<f64>(1e-15);
    check::<f32>(1e-7);
}

#[test]
fn independent_high_precision_bca_fixture_and_multisample_acceleration() {
    let reference: Vec<f64> = include_str!("data/resampling.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(|l| l.split_whitespace().map(|x| x.parse().unwrap()))
        .collect();
    fn check<S: RealScalar>(reference: &[f64], tol: f64) {
        let data: Vec<S> = [1.0, 2.0, 3.0, 9.0]
            .map(|v| S::try_from_f64_lossy(v).unwrap())
            .into();
        let all: Vec<_> = (0..4)
            .flat_map(|a| {
                (0..4)
                    .flat_map(move |b| (0..4).flat_map(move |c| (0..4).map(move |d| [a, b, c, d])))
            })
            .collect();
        let plan = ResamplingPlan::new(
            &[4],
            boot(BootstrapDesign::Independent, BootstrapMethod::Bca),
            ResamplingOptions::default()
                .with_replicates(256)
                .with_batch_size(7),
        )
        .unwrap();
        let mut rng = Scripted::indices(all.iter().flatten().copied(), 4);
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|s| average(&data, s)),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        let ci = interval(output.report());
        close(ci.bias_correction.unwrap(), reference[0], tol);
        close(ci.acceleration.unwrap(), reference[1], tol);
        close(ci.bounds.lower(), reference[2], tol);
        close(ci.bounds.upper(), reference[3], tol);
    }
    check::<f64>(&reference, 2e-12);
    check::<f32>(&reference, 1e-6);
    // Equal paired columns: deleting one pair recovers the ordinary four-unit jackknife.
    // Separate strata: fixed weights yield variance sum w_j^2 s_j^2/n_j.
    let data = [1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 9.0];
    for design in [
        BootstrapDesign::Paired,
        BootstrapDesign::Stratified,
        BootstrapDesign::Independent,
    ] {
        let statistic = |s: ResampledIndices<'_>| {
            let m = |g| -> Result<f64, SciOxideError> {
                let group = s.group(g)?;
                Ok(group.iter().map(|&i| data[i]).sum::<f64>() / group.len() as f64)
            };
            Ok(0.5 * m(0)? + 0.5 * m(1)?)
        };
        let plan = ResamplingPlan::new(
            &[4, 4],
            ResamplingMethod::Jackknife { design },
            ResamplingOptions::default()
                .with_replicates(8)
                .with_batch_size(3),
        )
        .unwrap();
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&statistic),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut scioxide_random::SplitMix64::from_seed(0),
                    scioxide_random::SamplingPolicy::advance_on_error(
                        scioxide_random::OutputPolicy::Incremental,
                    ),
                ),
                ctx,
            )
        })
        .unwrap();
        let Some(ResamplingInference::Jackknife(j)) = output.report().inference() else {
            panic!()
        };
        close(j.bias, 0.0, 1e-14);
        let variance = 38.75 / 12.0
            * if design == BootstrapDesign::Paired {
                1.0
            } else {
                0.5
            };
        close(j.standard_error, variance.sqrt(), 1e-14);
    }
}
#[test]
fn complex_strided_observations_and_extreme_finite_statistics() {
    let data = [
        Complex::new(1.0_f64, 2.0),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 4.0),
        Complex::new(0.0, 0.0),
        Complex::new(5.0, 6.0),
    ];
    let plan = ResamplingPlan::new(
        &[3],
        ResamplingMethod::Jackknife {
            design: BootstrapDesign::Independent,
        },
        ResamplingOptions::default()
            .with_replicates(3)
            .with_batch_size(1),
    )
    .unwrap();
    let output = sequential(|ctx| {
        evaluate(
            &plan,
            scioxide_stats::ResamplingProblem::Indexed(&|s| {
                Ok(s.indices().iter().map(|&i| data[2 * i].re).sum::<f64>()
                    / s.indices().len() as f64)
            }),
            scioxide_stats::ReplicateOptions::sequential(
                &mut scioxide_random::SplitMix64::from_seed(0),
                scioxide_random::SamplingPolicy::advance_on_error(
                    scioxide_random::OutputPolicy::Incremental,
                ),
            ),
            ctx,
        )
    })
    .unwrap();
    assert_eq!(output.report().observed(), Some(3.0));
    for scale in [1e-250, 1e250] {
        let plan = ResamplingPlan::new(
            &[3],
            boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
            ResamplingOptions::default()
                .with_replicates(27)
                .with_batch_size(1),
        )
        .unwrap();
        let all: Vec<_> = (0..3)
            .flat_map(|a| (0..3).flat_map(move |b| (0..3).flat_map(move |c| [a, b, c])))
            .collect();
        let mut rng = Scripted::indices(all, 3);
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|s| {
                    Ok(scale * s.indices().iter().map(|&i| i as f64 + 1.0).sum::<f64>() / 3.0)
                }),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        let ci = interval(output.report());
        assert!(ci.standard_error > 0.0 && ci.standard_error.is_finite());
        assert!(ci.bounds.lower() < 2.0 * scale && ci.bounds.upper() > 2.0 * scale);
    }
}
#[test]
fn inclusive_tolerance_and_provider_failure_do_not_publish_atomic_output() {
    let method = ResamplingMethod::MonteCarlo {
        alternative: TestAlternative::Less,
    };
    for tolerance in [None, Some(Tolerance::new(0.0, 0.0).unwrap())] {
        let mut options = ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2);
        options = options.with_tie_tolerance(tolerance);
        let plan = ResamplingPlan::new(&[1], method, options).unwrap();
        let problem = ResamplingProblem::MonteCarlo {
            observed: &[1.0],
            sampler: &|_, out| {
                out[0] = 1.0 + f64::EPSILON;
                Ok(())
            },
            statistic: &|x| Ok(x[0]),
        };
        let result = sequential(|ctx| {
            evaluate(
                &plan,
                problem,
                scioxide_stats::ReplicateOptions::sequential(
                    &mut SplitMix64::from_seed(1),
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        assert_eq!(
            test_result(result.report()).p_value(),
            if tolerance.is_none() { 1.0 } else { 1.0 / 6.0 }
        );
    }
    let plan = ResamplingPlan::new(
        &[2],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(4)
            .with_batch_size(2),
    )
    .unwrap();
    let mut rng = Scripted::indices([0, 1, 0], 2);
    let mut out = [-1.0; 4];
    let mut ws = plan.workspace().unwrap();
    let result = sequential(|ctx| {
        plan.evaluate_into(
            ResamplingProblem::Indexed(&|_| Ok(1.0)),
            scioxide_stats::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            ),
            &mut out,
            &mut ws,
            ctx,
        )
    });
    assert!(
        matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
    );
    assert_eq!(out, [-1.0; 4]);
    assert_eq!(rng.position, 3);
}

#[test]
fn cancellation_preserves_atomic_output_and_counts_real_callbacks() {
    let token = CancellationToken::new();
    let plan = ResamplingPlan::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut output = [-9.0; 5];
    let mut rng = SplitMix64::from_seed(8);
    let calls = AtomicUsize::new(0);
    let statistic = |_: ResampledIndices<'_>| {
        if calls.fetch_add(1, Ordering::Relaxed) == 1 {
            token.cancel();
        }
        Ok(1.0)
    };
    let report = run(
        ExecutionContext::sequential().with_cancellation(&token),
        SolveBudgets::unbounded(),
        |ctx| {
            plan.evaluate_into(
                ResamplingProblem::Indexed(&statistic),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::rollback(OutputPolicy::Atomic),
                ),
                &mut output,
                &mut workspace,
                ctx,
            )
        },
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        StatsTermination::Stopped(ExecutionStopReason::Cancelled)
    );
    assert_eq!(report.completed(), 1);
    assert!(report.inference().is_none());
    assert_eq!(output, [-9.0; 5]);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn exhaustive_small_block_distributions() {
    for circular in [false, true] {
        let starts = if circular { 4 } else { 3 };
        let design = if circular {
            BootstrapDesign::CircularBlock { length: 2 }
        } else {
            BootstrapDesign::MovingBlock { length: 2 }
        };
        let all: Vec<_> = (0..starts)
            .flat_map(|a| (0..starts).map(move |b| [a, b]))
            .collect();
        let plan = ResamplingPlan::new(
            &[4],
            boot(design, BootstrapMethod::Percentile),
            ResamplingOptions::default()
                .with_replicates(starts * starts)
                .with_batch_size(2),
        )
        .unwrap();
        let data = [1.0, 2.0, 4.0, 8.0];
        let mut rng = Scripted::indices(all.iter().flatten().copied(), starts);
        let output = sequential(|ctx| {
            evaluate(
                &plan,
                scioxide_stats::ResamplingProblem::Indexed(&|s| average(&data, s)),
                scioxide_stats::ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
                ),
                ctx,
            )
        })
        .unwrap();
        for (value, [a, b]) in output.distribution().iter().zip(all) {
            assert_eq!(
                *value,
                (data[a] + data[(a + 1) % 4] + data[b] + data[(b + 1) % 4]) / 4.0
            );
        }
    }
}
struct SnapshotFailure(SplitMix64);
impl BitGenerator for SnapshotFailure {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.0.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.0.next_u32()
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        self.0.fill_bytes(out)
    }
}
impl scioxide_random::SnapshotGenerator for SnapshotFailure {
    type Snapshot = ();
    fn snapshot(&self) -> Result<(), SciOxideError> {
        Err(SciOxideError::allocation_failure(
            scioxide_core::quantity!("snapshot_injection"),
            1,
        ))
    }
    fn restore_snapshot(&mut self, _: ()) {
        panic!("failed capture must not restore")
    }
}
#[test]
fn snapshot_failure_precedes_all_callbacks_and_draws() {
    let plan = ResamplingPlan::new(
        &[3],
        boot(BootstrapDesign::Independent, BootstrapMethod::Basic),
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut out = [-1.0; 5];
    let mut rng = SnapshotFailure(SplitMix64::from_seed(10));
    let mut control = rng.0.clone();
    let calls = AtomicUsize::new(0);
    let err = sequential(|ctx| {
        plan.evaluate_into(
            ResamplingProblem::Indexed(&|_| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(1.0)
            }),
            scioxide_stats::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::rollback(OutputPolicy::Atomic),
            ),
            &mut out,
            &mut workspace,
            ctx,
        )
    })
    .unwrap_err();
    assert!(
        matches ! (& (err) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: AllocationFailure { .. }))
    );
    assert_eq!(out, [-1.0; 5]);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(rng.next_u64().unwrap(), control.next_u64().unwrap());
}

#[test]
fn workspace_sample_storage_must_match_the_declared_method() {
    let monte = ResamplingPlan::<f64>::new(
        &[3],
        ResamplingMethod::MonteCarlo {
            alternative: TestAlternative::Less,
        },
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    let permutation = ResamplingPlan::new(
        &[3],
        ResamplingMethod::Permutation {
            kind: PermutationKind::Pairings,
            mode: PermutationMode::Randomized,
            alternative: TestAlternative::Less,
        },
        ResamplingOptions::default()
            .with_replicates(5)
            .with_batch_size(2),
    )
    .unwrap();
    let mut workspace = monte.workspace().unwrap();
    let mut output = [-1.0; 5];
    let mut rng = SplitMix64::from_seed(8);
    let calls = AtomicUsize::new(0);
    let result = sequential(|ctx| {
        permutation.evaluate_into(
            ResamplingProblem::Indexed(&|_| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(1.0)
            }),
            scioxide_stats::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            ),
            &mut output,
            &mut workspace,
            ctx,
        )
    });
    assert!(
        matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(output, [-1.0; 5]);
}

#[derive(Debug)]
struct Evaluated<S: scioxide_core::RealScalar> {
    report: ResamplingResult<S>,
    distribution: Vec<S>,
}
impl<S: scioxide_core::RealScalar> Evaluated<S> {
    fn report(&self) -> &ResamplingResult<S> {
        &self.report
    }
    fn distribution(&self) -> &[S] {
        &self.distribution
    }
}
// Unpublished output uses incremental publication to retain completed replicates on stops.
fn evaluate<'a, 'b, S: scioxide_core::RealScalar>(
    plan: &ResamplingPlan<S>,
    problem: ResamplingProblem<'_, S>,
    replicates: impl ReplicateExecution<S>,
    scope: impl Into<scioxide_core::SolveScope<'a, 'b>>,
) -> Result<Evaluated<S>, SciOxideError> {
    let mut workspace = plan.workspace()?;
    let mut distribution = vec![S::ZERO; plan.replicates()];
    let report = plan.evaluate_into(
        problem,
        replicates,
        &mut distribution,
        &mut workspace,
        scope,
    )?;
    distribution.truncate(report.completed());
    Ok(Evaluated {
        report,
        distribution,
    })
}
