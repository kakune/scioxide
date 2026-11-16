#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AdjointOperator, Budget, BudgetKind, ExecutionContext, SolveBudgets, SolveState,
};
use scioxide_linalg::Matrix;
use scioxide_stats::{
    ConfidenceProbability, Frequency, GlmCovariance, GlmFamily, GlmOptions, GlmResult, PreparedGlm,
    PreparedOperatorGlm, StatsTermination, Unweighted, Weights, glm_batch_into,
};

fn fit_owned<S: scioxide_core::RealScalar>(
    model: &PreparedGlm<S>,
    response: &[S],
    options: GlmOptions<S>,
) -> Result<GlmResult<S>, scioxide_core::SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state.begin_solve(
        ExecutionContext::sequential(),
        test_identity()?,
        &mut budgets,
    )?;
    model.fit_scoped(
        scioxide_stats::GlmInputs::new(response, options),
        &mut context,
    )
}

fn intercept(n: usize) -> Matrix<f64> {
    let mut design = Matrix::zeros(n, 1).unwrap();
    for i in 0..n {
        design.view_mut().set(i, 0, 1.0).unwrap();
    }
    design
}

#[test]
fn owned_rank_loss_retains_the_attempt_without_fabricating_a_fit() {
    let design = Matrix::from_vec(3, 2, vec![1.0; 6]).unwrap();
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let result = fit_owned(
        &model,
        &[1.0, 2.0, 3.0],
        GlmOptions::default().with_iterations(Budget::ONE),
    )
    .unwrap();
    assert!(result.fit().is_none());
    assert_eq!(result.report().termination, StatsTermination::RankDeficient);
    assert_eq!(result.report().accepted_iterations, 0);
    assert_eq!(result.report().counts.own().iterations(), 1);
    assert_eq!(
        result
            .report()
            .counts
            .own()
            .provider(scioxide_core::EvaluationRole::Factorization),
        1
    );
    assert_eq!(
        result
            .report()
            .counts
            .own()
            .provider(scioxide_core::EvaluationRole::LinearSolve),
        0
    );
}

fn resource_outcomes<S: scioxide_core::RealScalar>() {
    use scioxide_core::{CancellationToken, EvaluationRole};
    let design = Matrix::from_vec(4, 1, vec![S::ONE; 4]).unwrap();
    let response = [1.0, 2.0, 3.0, 4.0].map(|x| S::try_from_f64_lossy(x).unwrap());
    let dense = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let view = design.view().try_finite().unwrap();
    let operator = PreparedOperatorGlm::new(
        &view,
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let options = GlmOptions::default().with_iterations(Budget::try_from(40).unwrap());
    let mut output =
        scioxide_stats::GlmOutput::new(4, 1, scioxide_stats::GlmStorageOptions::new()).unwrap();
    let mut dw = dense
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let mut ow = operator
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    dense
        .fit_into(
            scioxide_stats::GlmInputs::new(&response, options),
            &mut output,
            &mut dw,
            ExecutionContext::sequential(),
        )
        .unwrap();
    for is_operator in [false, true] {
        let identity = test_identity().unwrap();
        for mode in 0..3 {
            let token = CancellationToken::new();
            if mode == 2 {
                token.cancel();
            }
            for owned in [false, true] {
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
                let mut context = state
                    .begin_solve(
                        ExecutionContext::sequential().with_cancellation(&token),
                        identity,
                        &mut budgets,
                    )
                    .unwrap();
                if mode == 0 {
                    assert!(
                        context
                            .record_provider(EvaluationRole::Function)
                            .unwrap()
                            .is_granted()
                    );
                }
                if mode == 1 {
                    assert!(
                        context
                            .record_provider(EvaluationRole::Function)
                            .unwrap()
                            .is_granted()
                    );
                }
                let before = output.fit().unwrap().coefficients().as_slice().to_vec();
                let fitted_before = output.fit().unwrap().fitted().as_slice().to_vec();
                let report = if owned {
                    let result = if is_operator {
                        operator.fit_scoped(
                            scioxide_stats::GlmInputs::new(&response, options),
                            &mut context,
                        )
                    } else {
                        dense.fit_scoped(
                            scioxide_stats::GlmInputs::new(&response, options),
                            &mut context,
                        )
                    }
                    .unwrap();
                    assert!(result.fit().is_none());
                    *result.report()
                } else if is_operator {
                    operator
                        .fit_into(
                            scioxide_stats::GlmInputs::new(&response, options),
                            &mut output,
                            &mut ow,
                            &mut context,
                        )
                        .unwrap()
                } else {
                    dense
                        .fit_into(
                            scioxide_stats::GlmInputs::new(&response, options),
                            &mut output,
                            &mut dw,
                            &mut context,
                        )
                        .unwrap()
                };
                assert_eq!(
                    report.termination,
                    if mode == 2 {
                        StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
                    } else {
                        StatsTermination::BudgetExhausted
                    }
                );
                assert_eq!(report.accepted_iterations, 0);
                assert_eq!(report.counts.total().unwrap(), context.counts().nested());
                assert_eq!(output.fit().unwrap().coefficients().as_slice(), before);
                assert_eq!(output.fit().unwrap().fitted().as_slice(), fitted_before);
            }
        }
        // Reuse both the numerical output and the previously exhausted workspace.
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let report = if is_operator {
            operator.fit_into(
                scioxide_stats::GlmInputs::new(&response, options),
                &mut output,
                &mut ow,
                &mut context,
            )
        } else {
            dense.fit_into(
                scioxide_stats::GlmInputs::new(&response, options),
                &mut output,
                &mut dw,
                &mut context,
            )
        }
        .unwrap();
        assert_eq!(report.termination, StatsTermination::Converged);
        assert!(report.accepted_iterations > 0);
        assert_eq!(report.counts.own().accepted(), report.accepted_iterations);
        assert_eq!(
            output.fit().unwrap().rank(),
            if is_operator { None } else { Some(1) }
        );
        assert!((output.fit().unwrap().fitted().as_slice()[0].into_f64() - 2.5).abs() < 2e-6);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let short = GlmOptions::default().with_iterations(Budget::ONE);
        let result = if is_operator {
            operator.fit_scoped(
                scioxide_stats::GlmInputs::new(&response, short),
                &mut context,
            )
        } else {
            dense.fit_scoped(
                scioxide_stats::GlmInputs::new(&response, short),
                &mut context,
            )
        }
        .unwrap();
        assert_eq!(
            result.report().termination,
            StatsTermination::BudgetExhausted
        );
        assert_eq!(result.report().accepted_iterations, 1);
        assert!(result.fit().is_some());
    }
}

#[test]
fn all_glm_forms_report_empty_stops_partial_fits_and_successful_reuse() {
    resource_outcomes::<f64>();
    resource_outcomes::<f32>();
}

#[test]
fn cancellation_at_solver_and_publication_boundaries_retains_accepted_state() {
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
    let design = intercept(4);
    let dense = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let view = design.view().try_finite().unwrap();
    let operator = PreparedOperatorGlm::new(
        &view,
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let mut dw = dense
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let mut ow = operator
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let options = GlmOptions::default().with_iterations(Budget::try_from(40).unwrap());
    for is_operator in [false, true] {
        let identity = test_identity().unwrap();
        let mut stopped_empty = false;
        let mut stopped_partial = false;
        let mut converged = false;
        for limit in 0..512 {
            let stop = StopAfter {
                polls: AtomicUsize::new(0),
                limit,
            };
            let mut output =
                scioxide_stats::GlmOutput::new(4, 1, scioxide_stats::GlmStorageOptions::new())
                    .unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential().with_cancellation(&stop),
                    identity,
                    &mut budgets,
                )
                .unwrap();
            let report = if is_operator {
                operator.fit_into(
                    scioxide_stats::GlmInputs::new(&[1.0, 2.0, 3.0, 4.0], options),
                    &mut output,
                    &mut ow,
                    &mut context,
                )
            } else {
                dense.fit_into(
                    scioxide_stats::GlmInputs::new(&[1.0, 2.0, 3.0, 4.0], options),
                    &mut output,
                    &mut dw,
                    &mut context,
                )
            }
            .unwrap();
            assert_eq!(report.counts.total().unwrap(), context.counts().nested());
            if report.accepted_iterations == 0 {
                assert!(output.fit().is_none());
                stopped_empty = true;
            } else {
                assert!((output.fit().unwrap().fitted().as_slice()[0] - 2.5).abs() < 1e-12);
                assert!((output.fit().unwrap().deviance() - 5.0).abs() < 1e-12);
                stopped_partial |= report.termination
                    == StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled);
            }
            if report.termination == StatsTermination::Converged {
                converged = true;
                break;
            }
            assert_eq!(
                report.termination,
                StatsTermination::Stopped(scioxide_core::ExecutionStopReason::Cancelled)
            );
        }
        assert!(stopped_empty && stopped_partial && converged);
    }
}

#[test]
fn canonical_intercept_fits() {
    let cases = [
        (GlmFamily::Gaussian, [1.0, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::Binomial, [0.0, 1.0, 0.0, 1.0], 0.5),
        (GlmFamily::Poisson, [1.0, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::Gamma, [1.0, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::InverseGaussian, [1.0, 2.0, 3.0, 4.0], 2.5),
    ];
    for (family, response, expected) in cases {
        let design = intercept(response.len());
        let model = PreparedGlm::new(
            design.view(),
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let attempt = fit_owned(
            &model,
            &response,
            GlmOptions::default().with_iterations(Budget::try_from(100).unwrap()),
        )
        .unwrap();
        let fit = attempt.fit().unwrap();
        assert_eq!(
            attempt.report().termination,
            StatsTermination::Converged,
            "{family:?}"
        );
        for &mean in fit.fitted().as_slice() {
            assert!((mean - expected).abs() < 1e-8, "{family:?}: {mean}");
        }
    }
}

#[test]
fn canonical_intercept_fits_f32() {
    let cases = [
        (GlmFamily::Gaussian, [1.0_f32, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::Binomial, [0.0, 1.0, 0.0, 1.0], 0.5),
        (GlmFamily::Poisson, [1.0, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::Gamma, [1.0, 2.0, 3.0, 4.0], 2.5),
        (GlmFamily::InverseGaussian, [1.0, 2.0, 3.0, 4.0], 2.5),
    ];
    let mut design = Matrix::zeros(4, 1).unwrap();
    for i in 0..4 {
        design.view_mut().set(i, 0, 1.0_f32).unwrap();
    }
    for (family, response, expected) in cases {
        let model = PreparedGlm::new(
            design.view(),
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let attempt = fit_owned(
            &model,
            &response,
            GlmOptions::default().with_iterations(Budget::try_from(60).unwrap()),
        )
        .unwrap();
        let fit = attempt.fit().unwrap();
        assert_eq!(
            attempt.report().termination,
            StatsTermination::Converged,
            "{family:?}"
        );
        assert!(
            (fit.fitted().view().get(0).unwrap() - expected).abs() < 1e-4,
            "{family:?}"
        );
    }
}

fn two_groups() -> Matrix<f64> {
    let mut design = Matrix::zeros(8, 2).unwrap();
    for i in 0..8 {
        design.view_mut().set(i, 0, 1.0).unwrap();
        design
            .view_mut()
            .set(i, 1, if i < 4 { 0.0 } else { 1.0 })
            .unwrap();
    }
    design
}

#[test]
fn canonical_group_means_and_scores() {
    let cases = [
        (
            GlmFamily::Gaussian,
            [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            2.0,
            4.0,
        ),
        (
            GlmFamily::Binomial,
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            0.25,
            0.75,
        ),
        (
            GlmFamily::Poisson,
            [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            2.0,
            4.0,
        ),
        (
            GlmFamily::Gamma,
            [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            2.0,
            4.0,
        ),
        (
            GlmFamily::InverseGaussian,
            [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            2.0,
            4.0,
        ),
    ];
    let design = two_groups();
    for (family, response, left, right) in cases {
        let model = PreparedGlm::new(
            design.view(),
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let attempt = fit_owned(
            &model,
            &response,
            GlmOptions::default().with_iterations(Budget::try_from(100).unwrap()),
        )
        .unwrap();
        let fit = attempt.fit().unwrap();
        assert_eq!(
            attempt.report().termination,
            StatsTermination::Converged,
            "{family:?}"
        );
        assert!(
            (fit.fitted().view().get(0).unwrap() - left).abs() < 1e-7,
            "{family:?}"
        );
        assert!(
            (fit.fitted().view().get(7).unwrap() - right).abs() < 1e-7,
            "{family:?}"
        );
        assert!(fit.score_norm() < 1e-6, "{family:?}: {}", fit.score_norm());
    }
}

#[test]
fn grouped_binomial_and_exposure() {
    let design = intercept(4);
    let response = [0.25, 0.75, 0.25, 0.75];
    let trials = [4, 4, 4, 4];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Binomial,
        Unweighted,
        Some(&trials),
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &model,
        &response,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    assert_eq!(attempt.report().termination, StatsTermination::Converged);
    assert!((fit.fitted().view().get(0).unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(fit.degrees_of_freedom(), Some(15.0));

    let exposure = [1.0, 2.0, 3.0, 4.0];
    let counts = [2.0, 4.0, 6.0, 8.0];
    let poisson = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        Some(&exposure),
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &poisson,
        &counts,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    assert_eq!(attempt.report().termination, StatsTermination::Converged);
    assert!((fit.coefficients().view().get(0).unwrap() - 2.0_f64.ln()).abs() < 1e-8);
    assert!(fit.deviance() < 1e-10);
}

#[test]
fn dispersion_and_covariance_reference() {
    let design = intercept(4);
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let y = [1.0, 2.0, 3.0, 4.0];
    let attempt = fit_owned(
        &model,
        &y,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    assert!((fit.pearson_chi_square() - 2.0).abs() < 1e-10);
    assert!((fit.dispersion().unwrap() - 2.0 / 3.0).abs() < 1e-10);
    let classical = model.covariance(&y, &fit, GlmCovariance::Model).unwrap();
    assert!((classical.view().get(0, 0).unwrap() - 0.1).abs() < 1e-10);
    let hc0 = model.covariance(&y, &fit, GlmCovariance::Hc0).unwrap();
    let hc3 = model.covariance(&y, &fit, GlmCovariance::Hc3).unwrap();
    assert!(hc0.view().get(0, 0).unwrap() > 0.0);
    assert!(hc3.view().get(0, 0).unwrap() > hc0.view().get(0, 0).unwrap());
    let inference = model
        .coefficient_inference(
            &y,
            &fit,
            GlmCovariance::Model,
            ConfidenceProbability::new(0.95).unwrap(),
        )
        .unwrap();
    assert_eq!(inference.len(), 1);
    assert!((inference[0].standard_error() - 0.1_f64.sqrt()).abs() < 1e-10);
    // A fit from different observations cannot be paired with this response.
    let other = fit_owned(&model, &[2.0, 3.0, 4.0, 5.0], GlmOptions::default()).unwrap();
    let fit = other.fit().unwrap();
    assert!(model.covariance(&y, &fit, GlmCovariance::Model).is_err());
}

#[test]
fn binomial_sandwich_matches_hand_calculation() {
    let design = intercept(4);
    let y = [0.0, 1.0, 0.0, 1.0];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Binomial,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &model,
        &y,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    let model_variance = model
        .covariance(&y, &fit, GlmCovariance::Model)
        .unwrap()
        .view()
        .get(0, 0)
        .unwrap();
    let hc0 = model
        .covariance(&y, &fit, GlmCovariance::Hc0)
        .unwrap()
        .view()
        .get(0, 0)
        .unwrap();
    let hc3 = model
        .covariance(&y, &fit, GlmCovariance::Hc3)
        .unwrap()
        .view()
        .get(0, 0)
        .unwrap();
    assert!((model_variance - 1.0).abs() < 1e-10);
    assert!((hc0 - 1.0).abs() < 1e-10);
    assert!((hc3 - 16.0 / 9.0).abs() < 1e-10);
}

#[test]
fn rank_loss_and_transactional_budget() {
    let mut design = Matrix::zeros(4, 2).unwrap();
    for i in 0..4 {
        design.view_mut().set(i, 0, 1.0).unwrap();
        design.view_mut().set(i, 1, 1.0).unwrap();
    }
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let mut fit =
        scioxide_stats::GlmOutput::new(4, 2, scioxide_stats::GlmStorageOptions::new()).unwrap();
    assert!(fit.fit().is_none());
    let mut work = model
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            test_identity().unwrap(),
            &mut budgets,
        )
        .unwrap();
    let status = model
        .fit_into(
            scioxide_stats::GlmInputs::new(
                &[1.0, 2.0, 3.0, 4.0],
                GlmOptions::default().with_iterations(Budget::try_from(10).unwrap()),
            ),
            &mut fit,
            &mut work,
            &mut context,
        )
        .unwrap();
    assert_eq!(status.termination, StatsTermination::RankDeficient);
    assert!(fit.fit().is_none());
}

#[test]
fn detects_binomial_separation() {
    let design = two_groups();
    let y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Binomial,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &model,
        &y,
        GlmOptions::default().with_iterations(Budget::try_from(100).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    assert_eq!(attempt.report().termination, StatsTermination::Separated);
    assert!(model.covariance(&y, &fit, GlmCovariance::Model).is_err());
}

#[test]
fn operator_fit_matches_dense_poisson() {
    let design = two_groups();
    let response = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
    let dense = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &dense,
        &response,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let expected = attempt.fit().unwrap();
    let sparse = design
        .to_sparse_csr(scioxide_sparse::ExplicitZeroPolicy::Preserve)
        .unwrap();
    let view = design.view();
    for source in [
        &design.view().try_finite().unwrap() as &dyn AdjointOperator<f64>,
        &sparse,
        &view.try_finite().unwrap(),
    ] {
        let operator = PreparedOperatorGlm::new(
            source,
            GlmFamily::Poisson,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let mut output =
            scioxide_stats::GlmOutput::new(8, 2, scioxide_stats::GlmStorageOptions::new()).unwrap();
        let mut work = operator
            .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
            .unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                test_identity().unwrap(),
                &mut budgets,
            )
            .unwrap();
        let status = operator
            .fit_into(
                scioxide_stats::GlmInputs::new(
                    &response,
                    GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
                ),
                &mut output,
                &mut work,
                &mut context,
            )
            .unwrap();
        assert_eq!(status.termination, StatsTermination::Converged);
        for (&a, &b) in output
            .fit()
            .unwrap()
            .coefficients()
            .as_slice()
            .iter()
            .zip(expected.coefficients().as_slice())
        {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(output.fit().unwrap().degrees_of_freedom(), None);

        let before = output.fit().unwrap().coefficients().as_slice().to_vec();
        let mut limited = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
        let mut state = SolveState::new();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                test_identity().unwrap(),
                &mut limited,
            )
            .unwrap();
        let status = operator
            .fit_into(
                scioxide_stats::GlmInputs::new(
                    &response,
                    GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
                ),
                &mut output,
                &mut work,
                &mut context,
            )
            .unwrap();
        assert_eq!(status.termination, StatsTermination::BudgetExhausted);
        assert_eq!(output.fit().unwrap().coefficients().as_slice(), before);
    }
}

#[test]
fn f32_family_smoke_and_weight_equivalence() {
    let mut design = Matrix::zeros(4, 1).unwrap();
    for i in 0..4 {
        design.view_mut().set(i, 0, 1.0_f32).unwrap();
    }
    let y = [1.0_f32, 2.0, 3.0, 4.0];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Weights::<_, Frequency>::new(&[2.0, 1.0, 1.0, 1.0]),
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &model,
        &y,
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let fit = attempt.fit().unwrap();
    assert_eq!(attempt.report().termination, StatsTermination::Converged);
    assert!((fit.fitted().view().get(0).unwrap() - 2.2).abs() < 1e-4);

    let expanded = intercept(5);
    let expanded_model = PreparedGlm::new(
        expanded.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let other = fit_owned(
        &expanded_model,
        &[1.0, 1.0, 2.0, 3.0, 4.0],
        GlmOptions::default().with_iterations(Budget::try_from(40).unwrap()),
    )
    .unwrap()
    .into_parts()
    .0
    .unwrap();
    assert!(
        (fit.coefficients().view().get(0).unwrap() as f64
            - other.coefficients().view().get(0).unwrap())
        .abs()
            < 1e-4
    );
    assert!(model.covariance(&y, &fit, GlmCovariance::Hc0).is_err());
}

#[test]
fn quasi_separation_and_invalid_domains() {
    let mut design = Matrix::zeros(6, 2).unwrap();
    let x = [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
    for (i, &value) in x.iter().enumerate() {
        design.view_mut().set(i, 0, 1.0).unwrap();
        design.view_mut().set(i, 1, value).unwrap();
    }
    let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Binomial,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let attempt = fit_owned(
        &model,
        &y,
        GlmOptions::default().with_iterations(Budget::try_from(100).unwrap()),
    )
    .unwrap();
    assert!(attempt.fit().is_some());
    assert_eq!(attempt.report().termination, StatsTermination::Separated);
    assert!(
        fit_owned(
            &model,
            &[0.0, 0.0, 0.0, 1.0, 1.0, 2.0],
            GlmOptions::default().with_iterations(Budget::try_from(10).unwrap())
        )
        .is_err()
    );
    assert!(
        PreparedGlm::new(
            design.view(),
            GlmFamily::Poisson,
            Unweighted,
            None,
            None,
            Some(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX)
        )
        .is_err()
    );
}

#[test]
fn dense_batch_commits_only_after_all_responses() {
    let design = intercept(4);
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let first = [1.0, 2.0, 3.0, 4.0];
    let second = [2.0, 3.0, 4.0, 5.0];
    let bad = [2.0, -1.0, 4.0, 5.0];
    let mut output = [
        scioxide_stats::GlmBatchOutput {
            output: model.output().unwrap(),
            report: None,
        },
        scioxide_stats::GlmBatchOutput {
            output: model.output().unwrap(),
            report: None,
        },
    ];
    let mut workspace = model.batch_workspace(2, 1, GlmOptions::default()).unwrap();
    let options = GlmOptions::default().with_iterations(Budget::try_from(40).unwrap());
    glm_batch_into(
        scioxide_stats::GlmBatchInputs {
            model: &model,
            responses: &[&first, &second],
            options,
            batch_options: Default::default(),
        },
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(
        (output[0]
            .output
            .fit()
            .unwrap()
            .fitted()
            .view()
            .get(0)
            .unwrap()
            - 2.5)
            .abs()
            < 1e-9
    );
    assert!(
        (output[1]
            .output
            .fit()
            .unwrap()
            .fitted()
            .view()
            .get(0)
            .unwrap()
            - 3.5)
            .abs()
            < 1e-9
    );
    let prior = output[0]
        .output
        .fit()
        .unwrap()
        .coefficients()
        .view()
        .get(0)
        .unwrap();
    let prior_reports = [output[0].report, output[1].report];
    assert!(
        glm_batch_into(
            scioxide_stats::GlmBatchInputs {
                model: &model,
                responses: &[&second, &bad],
                options,
                batch_options: Default::default()
            },
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(
        output[0]
            .output
            .fit()
            .unwrap()
            .coefficients()
            .view()
            .get(0)
            .unwrap(),
        prior
    );
    assert_eq!([output[0].report, output[1].report], prior_reports);
    let zero = Matrix::zeros(4, 1).unwrap();
    let deficient = PreparedGlm::new(
        zero.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    glm_batch_into(
        scioxide_stats::GlmBatchInputs {
            model: &deficient,
            responses: &[&first, &second],
            options,
            batch_options: Default::default(),
        },
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(
        output[0]
            .output
            .fit()
            .unwrap()
            .coefficients()
            .view()
            .get(0)
            .unwrap(),
        prior
    );
    for report in output.iter().map(|slot| slot.report) {
        let report = report.unwrap();
        assert_eq!(report.termination, StatsTermination::RankDeficient);
        assert_eq!(report.accepted_iterations, 0);
    }
}

#[test]
fn dense_operator_and_injected_deviances_agree_for_all_families() {
    use scioxide_stats::{FitObjective, GlmObjective};
    let design = intercept(4);
    let view = design.view().try_finite().unwrap();
    for (family, y) in [
        (GlmFamily::Gaussian, [1., 2., 3., 4.]),
        (GlmFamily::Binomial, [0., 1., 0., 1.]),
        (GlmFamily::Poisson, [1., 2., 3., 4.]),
        (GlmFamily::Gamma, [1., 2., 3., 4.]),
        (GlmFamily::InverseGaussian, [1., 2., 3., 4.]),
    ] {
        let dense = PreparedGlm::new(
            design.view(),
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let operator = PreparedOperatorGlm::new(
            &view,
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
        )
        .unwrap();
        let options = GlmOptions::default().with_iterations(Budget::try_from(100).unwrap());
        let attempt = fit_owned(&dense, &y, options).unwrap();
        let _fit_report = attempt.report();
        let fit = attempt.fit().unwrap();
        let objective = GlmObjective::new(&dense, &y).unwrap();
        let value = objective
            .value(fit.coefficients().as_slice(), &mut ())
            .unwrap();
        assert!((fit.deviance() - 2. * value).abs() < 1e-12, "{family:?}");
        let mut output =
            scioxide_stats::GlmOutput::new(4, 1, scioxide_stats::GlmStorageOptions::new()).unwrap();
        let mut work = operator
            .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
            .unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                test_identity().unwrap(),
                &mut budgets,
            )
            .unwrap();
        let status = operator
            .fit_into(
                scioxide_stats::GlmInputs::new(&y, options),
                &mut output,
                &mut work,
                &mut context,
            )
            .unwrap();
        assert_eq!(
            status.termination,
            StatsTermination::Converged,
            "{family:?}"
        );
        assert!(
            (fit.deviance() - output.fit().unwrap().deviance()).abs() < 1e-10,
            "{family:?}"
        );
    }
}

#[test]
fn numerical_failures_preserve_every_dense_and_operator_fit_field() {
    let design = intercept(4);
    let dense = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let view = design.view().try_finite().unwrap();
    let operator = PreparedOperatorGlm::new(
        &view,
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(usize::MAX),
    )
    .unwrap();
    let options = GlmOptions::default().with_iterations(Budget::try_from(40).unwrap());
    let mut output = dense.output().unwrap();
    let mut seed = dense.workspace(options).unwrap();
    dense
        .fit_into(
            scioxide_stats::GlmInputs::new(&[1., 2., 3., 4.], options),
            &mut output,
            &mut seed,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let snapshot = |fit: &scioxide_stats::GlmFitView<'_, f64>| {
        (
            [
                fit.coefficients().as_slice(),
                fit.fitted().as_slice(),
                fit.residuals().as_slice(),
                fit.linear_predictor().as_slice(),
            ]
            .concat(),
            [fit.deviance(), fit.pearson_chi_square(), fit.score_norm()],
            [fit.dispersion(), fit.degrees_of_freedom()],
            fit.rank(),
        )
    };
    let before = snapshot(&output.fit().unwrap());
    let y = [1e155, -1e155, 1e155, -1e155];
    let mut work = dense
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            test_identity().unwrap(),
            &mut budgets,
        )
        .unwrap();
    assert!(
        dense
            .fit_into(
                scioxide_stats::GlmInputs::new(&y, options),
                &mut output,
                &mut work,
                &mut context
            )
            .is_err()
    );
    assert_eq!(snapshot(&output.fit().unwrap()), before);
    let mut work = operator
        .workspace(scioxide_stats::GlmOptions::default().with_max_bytes(usize::MAX))
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            test_identity().unwrap(),
            &mut budgets,
        )
        .unwrap();
    assert!(
        operator
            .fit_into(
                scioxide_stats::GlmInputs::new(&y, options),
                &mut output,
                &mut work,
                &mut context
            )
            .is_err()
    );
    assert_eq!(snapshot(&output.fit().unwrap()), before);
}

fn test_identity() -> Result<scioxide_core::SolveIdentity, scioxide_core::SciOxideError> {
    Ok(scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(1),
        scioxide_core::algorithm_id!("stats-test", "parent", 1),
        0,
        0,
    ))
}
