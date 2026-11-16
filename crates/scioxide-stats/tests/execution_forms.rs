//! Reserved execution, publication and resource accounting for migrated families.
#![allow(clippy::unwrap_used)]

use scioxide_core::{AttemptOutcome, Budget, ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_linalg::Matrix;
use scioxide_stats::*;

#[test]
fn child_glm_keeps_parent_iteration_and_acceptance_state_local() {
    let data = Matrix::from_vec(4, 1, vec![1.0_f64; 4]).unwrap();
    let plan = PreparedGlm::new(
        data.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        GlmStorageOptions::new(),
    )
    .unwrap();
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded()
        .with_limit(scioxide_core::BudgetKind::Iteration, Budget::ONE);
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(1),
        scioxide_core::algorithm_id!("stats-test", "parent", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert!(parent.record_iteration().unwrap().is_granted());
    let result = plan
        .fit_scoped(
            GlmInputs::new(&[1., 2., 3., 4.], GlmOptions::default()),
            &mut parent,
        )
        .unwrap();
    assert_eq!(result.report().termination, StatsTermination::Converged);
    assert!(result.report().counts.own().iterations() > 1);
    assert_eq!(parent.counts().own().iterations(), 1);
    assert_eq!(parent.counts().own().accepted(), 0);
    assert_eq!(
        parent.counts().nested(),
        result.report().counts.total().unwrap()
    );
}

struct FailingIntercept {
    calls: core::sync::atomic::AtomicUsize,
    fail_at: core::sync::atomic::AtomicUsize,
}
impl FailingIntercept {
    fn admit(&self) -> Result<(), SciOxideError> {
        use core::sync::atomic::Ordering;
        if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_at.load(Ordering::Relaxed) {
            Err(SciOxideError::external("injected operator failure"))
        } else {
            Ok(())
        }
    }
}
impl scioxide_core::LinearOperator<f64> for FailingIntercept {
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn rows(&self) -> usize {
        4
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.admit()?;
        output.fill(input[0]);
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for FailingIntercept {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.admit()?;
        output[0] = input.iter().sum();
        Ok(())
    }
}

#[test]
fn a_later_operator_failure_keeps_checked_workspace_progress_and_caller_output() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    let operator = FailingIntercept {
        calls: AtomicUsize::new(0),
        fail_at: AtomicUsize::new(0),
    };
    let plan = PreparedOperatorGlm::new(
        &operator,
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        GlmStorageOptions::new(),
    )
    .unwrap();
    let mut output = plan.output().unwrap();
    let mut work = plan.workspace(GlmOptions::default()).unwrap();
    let mut observed = false;
    for failure in 0..128 {
        operator.calls.store(0, Ordering::Relaxed);
        operator.fail_at.store(failure, Ordering::Relaxed);
        let before = output.fit().map(|fit| fit.coefficients().as_slice()[0]);
        let result = plan.fit_into(
            GlmInputs::new(&[1., 2., 3., 4.], GlmOptions::default()),
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        );
        if result.is_err()
            && let Some(accepted) = work.last_accepted()
        {
            assert!(accepted.accepted_iterations() > 0);
            assert!(accepted.coefficients().as_slice()[0].is_finite());
            assert!(
                accepted
                    .fitted()
                    .as_slice()
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
            );
            assert_eq!(
                output.fit().map(|fit| fit.coefficients().as_slice()[0]),
                before
            );
            assert_eq!(
                work.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
            observed = true;
            break;
        }
    }
    assert!(observed);
}

#[test]
fn batch_limits_include_worker_and_staging_storage_before_execution() {
    let design = Matrix::from_vec(4, 1, vec![1.0_f64; 4]).unwrap();
    let plan = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        GlmStorageOptions::new(),
    )
    .unwrap();
    assert_eq!(
        plan.batch_workspace(2, 1, GlmOptions::default().with_max_bytes(1))
            .err()
            .unwrap()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut workspace = plan.batch_workspace(2, 1, GlmOptions::default()).unwrap();
    let mut output = [
        GlmBatchOutput {
            output: plan.output().unwrap(),
            report: None,
        },
        GlmBatchOutput {
            output: plan.output().unwrap(),
            report: None,
        },
    ];
    let error = glm_batch_into(
        GlmBatchInputs {
            model: &plan,
            responses: &[&[1., 2., 3., 4.], &[2., 3., 4., 5.]],
            options: GlmOptions::default(),
            batch_options: StatisticsOptions::new().with_max_bytes(1),
        },
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert!(
        output
            .iter()
            .all(|slot| slot.output.fit().is_none() && slot.report.is_none())
    );
}

#[test]
fn cancellation_inside_a_later_huber_factor_keeps_the_previous_checked_iterate() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter {
        polls: AtomicUsize,
        limit: usize,
    }
    impl scioxide_core::Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.polls.fetch_add(1, Ordering::Relaxed) >= self.limit
        }
    }
    let design =
        Matrix::from_vec(8, 2, (0..8).flat_map(|i| [1.0_f64, f64::from(i)]).collect()).unwrap();
    let response = [1.1, 2.8, 5.2, 6.7, 9.1, 11.3, 12.8, 40.0];
    let plan = PreparedHuber::new(
        design.view(),
        HuberOptions::default().with_iterations(Budget::ONE.saturating_add(99)),
    )
    .unwrap();
    let mut work = plan.workspace().unwrap();
    let mut output = plan.output().unwrap();
    let mut observed = false;
    for limit in 0..512 {
        let stop = StopAfter {
            polls: AtomicUsize::new(0),
            limit,
        };
        let before = output.fit().map(|fit| fit.coefficients().as_slice()[0]);
        let result = plan.fit_into(
            &response,
            &mut output,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&stop),
        );
        if result.is_err()
            && let Some(accepted) = work.last_accepted()
        {
            assert!(accepted.iterations() > 0);
            assert!(accepted.scale().is_finite() && accepted.scale() > 0.0);
            assert!(
                accepted
                    .coefficients()
                    .as_slice()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert_eq!(
                output.fit().map(|fit| fit.coefficients().as_slice()[0]),
                before
            );
            assert_eq!(
                work.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
            observed = true;
            break;
        }
    }
    assert!(observed);
}

mod one_shots {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use scioxide_core::*;
    use scioxide_random::{OutputPolicy, SamplingPolicy, SeedableRng, SplitMix64};
    use scioxide_stats::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn resampling_slice_adapters_match_prepared_streams() {
        let data = [2., 4., 4., 4., 5., 5., 7., 9.];
        let plan = ResamplingPlan::new(
            &[data.len()],
            ResamplingMethod::Bootstrap {
                design: BootstrapDesign::Independent,
                method: BootstrapMethod::Bca,
                confidence: ConfidenceProbability::default(),
            },
            ResamplingOptions::default(),
        )
        .unwrap()
        .with_statistic_scratch(data.len())
        .unwrap();
        let adapter = |indices: ResampledIndices<'_>, scratch: &mut [f64]| {
            for (out, &index) in scratch.iter_mut().zip(indices.indices()) {
                *out = data[index];
            }
            mean(&scratch[..indices.indices().len()])
        };
        let mut a = SplitMix64::from_seed(7);
        let mut b = a.clone();
        let plain = bootstrap(mean, &data, &mut a).unwrap();
        let owned = plan
            .evaluate(
                ResamplingProblem::IndexedWorkspace(&adapter),
                ReplicateOptions::sequential(
                    &mut b,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                ),
            )
            .unwrap();
        let Some(ResamplingInference::Bootstrap(configured)) = owned.report().inference() else {
            panic!("bootstrap inference missing")
        };
        assert_eq!(plain, configured);
        assert!((plain.standard_error - 2. / 8.0_f64.sqrt()).abs() < 0.03 * 2. / 8.0_f64.sqrt());
        assert!(plain.bounds.contains(5.));
        let j = jackknife(mean, &data).unwrap();
        assert!(j.bias.abs() < 1e-14);
        assert!((j.standard_error - (32.0_f64 / 56.).sqrt()).abs() < 1e-14);
        let error = bootstrap(mean, &[1.], &mut a).unwrap_err();
        assert!(
            matches!(error.view(),SciOxideErrorView::InvalidInput {argument,..} if argument.as_str()=="sample")
        );
        assert_eq!(
            bootstrap(mean, &[1., 1., 1.], &mut a).unwrap_err().kind(),
            SciOxideErrorKind::NumericalInstability
        );
        let callback = SciOxideError::invalid_input(
            argument!("options"),
            InvalidInputReason::Inconsistent,
            "caller-defined callback error",
        );
        assert_eq!(
            bootstrap(|_: &[f64]| Err(callback.clone()), &data, &mut a).unwrap_err(),
            callback
        );
    }

    fn parent_identity() -> SolveIdentity {
        SolveIdentity::new(
            ProblemToken::new(42),
            algorithm_id!("test", "stats-parent", 1),
            0,
            0,
        )
    }
    #[test]
    fn child_replicates_keep_parent_iteration_and_proposal_allowances_local() {
        let plan = ResamplingPlan::new(
            &[3],
            ResamplingMethod::Jackknife {
                design: BootstrapDesign::Independent,
            },
            ResamplingOptions::<f64>::default().with_replicates(3),
        )
        .unwrap();
        let statistic = |indices: ResampledIndices<'_>| {
            Ok(indices
                .indices()
                .iter()
                .map(|&i| [1., 2., 4.][i])
                .sum::<f64>()
                / indices.indices().len() as f64)
        };
        let mut workspace = plan.workspace().unwrap();
        let mut output = [0.; 3];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Proposal, Budget::ONE)
            .with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        let own = parent.counts().own();
        let accepted = parent.accepted_state();
        let mut rng = SplitMix64::from_seed(1);
        let report = plan
            .evaluate_into(
                ResamplingProblem::Indexed(&statistic),
                ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                ),
                &mut output,
                &mut workspace,
                &mut parent,
            )
            .unwrap();
        assert_eq!(report.termination(), StatsTermination::Completed);
        assert_eq!(parent.counts().own(), own);
        assert_eq!(parent.accepted_state(), accepted);
        assert_eq!(
            parent.counts().nested(),
            workspace.last_attempt().unwrap().counts().total().unwrap()
        );
        let before = parent.counts();
        assert!(
            plan.evaluate_into(
                ResamplingProblem::Indexed(&statistic),
                ReplicateOptions::sequential(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic)
                ),
                &mut [],
                &mut workspace,
                &mut parent
            )
            .is_err()
        );
        assert_eq!(parent.counts(), before);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
    #[test]
    fn sobol_shared_evaluation_limit_precedes_model_callbacks() {
        let calls = AtomicUsize::new(0);
        let root = scioxide_random::Pcg64::from_seed(1);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
        let mut parent = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                parent_identity(),
                &mut budgets,
            )
            .unwrap();
        let result = sobol_indices_monte_carlo_scoped(
            1,
            8,
            &root,
            |_, _| Ok(0.5_f64),
            |x| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(x[0])
            },
            StatisticsOptions::default(),
            &mut parent,
        );
        assert!(matches!(
            result.unwrap_err().view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn randomized_test_adapters_match_prepared_streams() {
        let x = [1., 2., 4.];
        let y = [3., 5., 7.];
        let mut a = SplitMix64::from_seed(12);
        let mut b = a.clone();
        let actual = permutation_test(|x, y| Ok(mean(x)? - mean(y)?), &x, &y, &mut a).unwrap();
        let plan = ResamplingPlan::new(
            &[3, 3],
            ResamplingMethod::Permutation {
                kind: PermutationKind::Independent,
                mode: PermutationMode::Randomized,
                alternative: TestAlternative::TwoSided,
            },
            ResamplingOptions::default(),
        )
        .unwrap()
        .with_statistic_scratch(6)
        .unwrap();
        let adapter = |indices: ResampledIndices<'_>, scratch: &mut [f64]| {
            for (out, &index) in scratch.iter_mut().zip(indices.indices()) {
                *out = if index < 3 { x[index] } else { y[index - 3] };
            }
            Ok(mean(&scratch[..3])? - mean(&scratch[3..6])?)
        };
        let expected = plan
            .evaluate(
                ResamplingProblem::IndexedWorkspace(&adapter),
                ReplicateOptions::sequential(
                    &mut b,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                ),
            )
            .unwrap();
        assert_eq!(
            expected.report().inference(),
            Some(ResamplingInference::Test(actual))
        );
        let sampler = |rng: &mut dyn scioxide_random::Rng<f64>, sample: &mut [f64]| {
            for x in sample {
                *x = rng.next_float()?;
            }
            Ok(())
        };
        let observed = [0.2, 0.5, 0.8];
        let actual = monte_carlo_test(mean, sampler, &observed, &mut a).unwrap();
        let plan = ResamplingPlan::new(
            &[3],
            ResamplingMethod::MonteCarlo {
                alternative: TestAlternative::TwoSided,
            },
            ResamplingOptions::default(),
        )
        .unwrap();
        let expected = plan
            .evaluate(
                ResamplingProblem::MonteCarlo {
                    observed: &observed,
                    sampler: &sampler,
                    statistic: &mean,
                },
                ReplicateOptions::sequential(
                    &mut b,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                ),
            )
            .unwrap();
        assert_eq!(
            expected.report().inference(),
            Some(ResamplingInference::Test(actual))
        );
    }

    #[test]
    fn glm_owned_forms_and_separation_are_authoritative() {
        let x = [
            1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        let design = scioxide_linalg::MatrixView::from_slice(8, 2, &x).unwrap();
        let response = [1., 0., 0., 0., 0., 1., 1., 1.];
        let fit = glm_fit(design, GlmFamily::Binomial, &response).unwrap();
        assert!(fit.coefficients().as_slice()[0].abs() < 1e-8);
        assert!((fit.coefficients().as_slice()[1] - 3.0_f64.ln()).abs() < 1e-8);
        let model = PreparedGlm::new(
            design,
            GlmFamily::Binomial,
            Unweighted,
            None,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        let owned = model
            .fit(GlmInputs::new(&response, Default::default()))
            .unwrap();
        let scoped = model
            .fit_scoped(
                GlmInputs::new(&response, Default::default()),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(owned.report(), scoped.report());
        assert_eq!(
            owned.fit().unwrap().coefficients().as_slice(),
            scoped.fit().unwrap().coefficients().as_slice()
        );
        let separated = glm_fit(
            design,
            GlmFamily::Binomial,
            &[0., 0., 0., 0., 1., 1., 1., 1.],
        )
        .err()
        .unwrap();
        assert!(matches!(
            separated.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::Divergence,
                ..
            }
        ));
    }
}
