#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_random::{Philox4x64, SeedableRng};
use scioxide_stats::*;

#[test]
fn sobol_bootstrap_uses_the_shared_open_probability_domain() {
    let a: Vec<f64> = (0..32).map(|i| i as f64 / 32.0).collect();
    let b: Vec<f64> = (0..32).map(|i| ((i * 13) % 32) as f64 / 32.0).collect();
    let evaluations = SobolEvaluations {
        a: &a,
        b: &b,
        hybrids: &b,
        dimensions: 1,
    };
    let root = Philox4x64::from_seed(1791);
    let run = |level| -> Result<SobolBootstrap, SciOxideError> {
        sobol_bootstrap_scoped(
            evaluations,
            &root,
            32,
            ConfidenceProbability::new(level)?,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(8192),
            ExecutionContext::sequential(),
        )
    };
    for level in [0.0, -0.0, 1.0, -0.1, 1.1] {
        assert_eq!(
            run(level).unwrap_err().kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    for level in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(run(level).unwrap_err().kind(), SciOxideErrorKind::NonFinite);
    }
    let result = run(0.9).unwrap();
    assert_eq!(
        result.confidence(),
        ConfidenceProbability::new(0.9).unwrap()
    );
    assert!(result.first_order()[0].0 < result.first_order()[0].1);
    assert!(result.total_order()[0].0 <= result.total_order()[0].1);
}

#[test]
fn cressie_read_presets_and_validation_share_the_same_kernel() {
    for lambda in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            CressieReadParameter::new(lambda).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    let observed = [10.0_f64, 20.0];
    let expected = [15.0, 15.0];
    let pearson =
        power_divergence_test(&observed, Some(&expected), CressieReadParameter::PEARSON, 0)
            .unwrap();
    assert!((pearson.statistic() - 10.0 / 3.0).abs() < 1e-12);
    let g =
        power_divergence_test(&observed, Some(&expected), CressieReadParameter::G_TEST, 0).unwrap();
    let reference = 2.0 * (10.0 * (2.0_f64 / 3.0).ln() + 20.0 * (4.0_f64 / 3.0).ln());
    assert!((g.statistic() - reference).abs() < 1e-12);
}

#[test]
fn one_mean_method_retains_degrees_and_rejects_inconsistent_models() {
    let confidence = ConfidenceProbability::default();
    let interval = scioxide_stats::MeanIntervalPlan::new(
        confidence,
        MeanIntervalMethod::StudentT,
        Some(4.0),
        Default::default(),
    )
    .and_then(|plan| plan.interval(2.0_f64, 0.5))
    .unwrap();
    assert_eq!(interval.method(), MeanIntervalMethod::StudentT);
    assert_eq!(interval.degrees_of_freedom(), Some(4.0));
    for degrees in [None, Some(0.0), Some(-1.0), Some(f64::NAN)] {
        assert!(
            scioxide_stats::MeanIntervalPlan::new(
                confidence,
                MeanIntervalMethod::StudentT,
                degrees,
                Default::default()
            )
            .and_then(|plan| plan.interval(2.0, 0.5))
            .is_err()
        );
    }
    assert!(
        scioxide_stats::MeanIntervalPlan::new(
            confidence,
            MeanIntervalMethod::NormalAsymptotic,
            Some(4.0),
            Default::default()
        )
        .and_then(|plan| plan.interval(2.0, 0.5))
        .is_err()
    );
    let moments = OnlineMoments::from_slice(&[1.0_f64, 2.0, 3.0], MomentOrder::Second).unwrap();
    assert!(
        moments
            .snapshot()
            .confidence_interval_of_mean_with_options(
                confidence,
                MeanIntervalMethod::NormalKnownVariance,
                Default::default()
            )
            .is_err()
    );
}

#[test]
fn classical_interval_and_coverage_are_published_together() {
    let confidence = ConfidenceProbability::new(0.9).unwrap();
    let result = t_test_one_sample_with_options(
        &[1.0_f64, 2.0, 3.0, 4.0],
        0.0,
        ClassicalOptions::default().with_confidence(confidence),
    )
    .unwrap();
    assert!(result.interval().is_some());
    assert_eq!(result.confidence(), Some(confidence));
    assert_eq!(
        result.into_parts().3.map(|(_, level)| level),
        Some(confidence)
    );
}

#[test]
fn injected_fit_rejects_a_replicate_completion_status_without_publication() {
    struct Completed;
    impl FitSolver<f64> for Completed {
        fn capabilities(&self) -> FitSolverCapabilities {
            FitSolverCapabilities::default()
        }
        fn solve(
            &mut self,
            _request: FitRequest<'_, f64>,
            _candidate: &mut [f64],
            _workspace: &mut (),
            _context: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<StatsTermination, SciOxideError> {
            Ok(StatsTermination::Completed)
        }
    }
    let objective = PreparedPowerFit::new(
        &[1.0, 2.0, 4.0],
        PowerTransform::BoxCox,
        PowerFitMethod::Likelihood,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )
    .unwrap();
    let bounds = scioxide_core::Bounds::new(vec![(None, None)]).unwrap();
    let mut workspace = FitWorkspace::for_objective(
        &objective,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(128),
    )
    .unwrap();
    let mut output = [99.0];
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(1791),
        scioxide_core::algorithm_id!("stats", "contract-test", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let error = fit_into(
        FitRequest {
            objective: &objective,
            derivatives: None,
            bounds: &bounds,
            integrality: &[false],
            initial: &[0.0],
        },
        &mut Completed,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(output, [99.0]);
}
