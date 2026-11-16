//! Progressive checkpoint, atomic publication and prepared-allocation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptKind, AttemptOutcome, Budget, ErrorTolerance, EvaluationRole, ExecutionContext,
    NormSelection, SciOxideError, SciOxideErrorView,
};
use scioxide_integrate::{
    SeriesMethod, SeriesOptions, SeriesProblem, SeriesSummation, SeriesTermination,
};
fn plan(quota: usize, limit: Option<Budget>) -> SeriesSummation<f64> {
    SeriesSummation::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(12).unwrap())
            .with_order(Budget::try_from(6).unwrap())
            .with_terms_per_advance(Budget::try_from(quota).unwrap())
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
            .with_max_evaluations(limit),
    )
    .unwrap()
}
#[test]
fn resume_keeps_epoch_and_cumulative_allowance() {
    let p = plan(2, Some(Budget::try_from(3).unwrap()));
    let provider = |n: usize, row: &mut [f64]| {
        row[0] = 0.5_f64.powi(n as i32);
        Ok(())
    };
    let mut state = p.state(1).unwrap();
    let mut ws = p.workspace(1).unwrap();
    let first = p
        .advance(
            SeriesProblem::Terms(&provider),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(first.terms(), 2);
    let epoch = ws.last_attempt().unwrap().epoch();
    assert_eq!(ws.last_attempt().unwrap().kind(), AttemptKind::Call);
    let second = p
        .advance(
            SeriesProblem::Terms(&provider),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(second.terms(), 3);
    assert_eq!(
        second.termination(),
        SeriesTermination::ContextBudgetLimit(scioxide_core::BudgetKind::Evaluation)
    );
    let attempt = ws.last_attempt().unwrap();
    assert_eq!(attempt.kind(), AttemptKind::Resume);
    assert_eq!(attempt.epoch(), epoch);
    assert_eq!(
        attempt
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        3
    );
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    let mut wrong = p.workspace(2).unwrap();
    let before = state.value().unwrap()[0];
    assert!(
        p.advance(
            SeriesProblem::Terms(&provider),
            &mut state,
            |_| Err(SciOxideError::external("rejected resume emitted a sample")),
            &mut wrong,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(state.value().unwrap()[0], before);
    assert_eq!(state.terms(), 3);
    assert_eq!(wrong.last_attempt().unwrap().kind(), AttemptKind::Resume);
    assert_eq!(
        wrong.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}
#[test]
fn provider_failure_keeps_accepted_prefix_and_atomic_output() {
    let p = plan(3, None);
    let mut state = p.state(1).unwrap();
    let mut ws = p.workspace(1).unwrap();
    let mut emitted = 0;
    let provider = |n: usize, row: &mut [f64]| {
        if n == 1 {
            return Err(SciOxideError::external("sentinel"));
        }
        row[0] = 1.0;
        Ok(())
    };
    let error = p
        .advance(
            SeriesProblem::Terms(&provider),
            &mut state,
            |_| {
                emitted += 1;
                Ok(())
            },
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::External { .. }));
    assert_eq!(state.terms(), 1);
    assert_eq!(emitted, 1);
    assert_eq!(state.value().unwrap(), &[1.0]);
    let epoch = ws.last_attempt().unwrap().epoch();
    assert_eq!(ws.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    let good = |n: usize, row: &mut [f64]| {
        row[0] = 0.5_f64.powi(n as i32);
        Ok(())
    };
    p.advance(
        SeriesProblem::Terms(&good),
        &mut state,
        |_| Ok(()),
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(state.terms(), 4);
    assert_eq!(ws.last_attempt().unwrap().kind(), AttemptKind::Resume);
    assert_eq!(ws.last_attempt().unwrap().epoch(), epoch);
    assert_eq!(
        ws.last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        5
    );
    let mut out = [99.0];
    assert!(
        p.sum_into(
            SeriesProblem::Terms(&provider),
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(out, [99.0]);
}

#[test]
fn failed_first_provider_does_not_reset_the_cumulative_budget() {
    let p = plan(2, Some(Budget::ONE));
    let mut state = p.state(1).unwrap();
    let mut ws = p.workspace(1).unwrap();
    let bad = |_: usize, _: &mut [f64]| -> Result<(), SciOxideError> {
        Err(SciOxideError::external("first provider failed"))
    };
    assert!(
        p.advance(
            SeriesProblem::Terms(&bad),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(state.terms(), 0);
    let epoch = ws.last_attempt().unwrap().epoch();
    let good = |_: usize, row: &mut [f64]| {
        row[0] = 1.0;
        Ok(())
    };
    let error = p
        .advance(
            SeriesProblem::Terms(&good),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
    assert_eq!(ws.last_attempt().unwrap().kind(), AttemptKind::Resume);
    assert_eq!(ws.last_attempt().unwrap().epoch(), epoch);
    assert_eq!(
        ws.last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        1
    );
    assert!(state.value().is_none());
    state.reset();
    let report = p
        .advance(
            SeriesProblem::Terms(&good),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.terms(), 1);
    assert_ne!(ws.last_attempt().unwrap().epoch(), epoch);
}

fn policy_probe<T: scioxide_core::NumericScalar>(
    policy: scioxide_core::ExecutionPolicy,
    allowance: usize,
    inherited: Option<scioxide_core::BudgetKind>,
    geometric: bool,
) -> (T, SeriesTermination, usize, usize) {
    use scioxide_core::{AlgebraicScalar, RealScalar};
    let tolerance = T::Real::try_from_f64_lossy(if geometric { 1e-3 } else { 1e-14 }).unwrap();
    let p = SeriesSummation::<T>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(64).unwrap())
            .with_order(Budget::try_from(10).unwrap())
            .with_execution_policy(policy)
            .with_max_evaluations(
                inherited
                    .is_none()
                    .then(|| Budget::try_from(allowance).unwrap()),
            )
            .with_tolerance(
                ErrorTolerance::scalar(tolerance, T::Real::ZERO, NormSelection::Infinity).unwrap(),
            ),
    )
    .unwrap();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let provider = |n: usize, row: &mut [T]| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let value = if geometric {
            0.5_f64.powi(i32::try_from(n).unwrap())
        } else {
            let x = f64::from(u32::try_from(n + 1).unwrap());
            1.0 / (x * x)
        };
        let real = T::Real::try_from_f64_lossy(value).unwrap();
        row[0] = T::from_components(real, real * T::Real::HALF);
        Ok(())
    };
    #[cfg(feature = "std")]
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(4).unwrap());
    #[cfg(not(feature = "std"))]
    let executor = scioxide_core::SequentialExecutor;
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    if let Some(kind) = inherited {
        budgets = budgets.with_limit(kind, Budget::try_from(allowance).unwrap());
    }
    let mut lifecycle = scioxide_core::SolveState::new();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(2171),
        scioxide_core::algorithm_id!("integrate", "series-policy-probe", 1),
        0,
        0,
    );
    let mut parent = lifecycle
        .begin_solve(ExecutionContext::new(&executor), identity, &mut budgets)
        .unwrap();
    let mut ws = p.workspace(1).unwrap();
    let mut output = [T::ZERO];
    let report = p
        .sum_into(
            SeriesProblem::Terms(&provider),
            &mut output,
            &mut ws,
            &mut parent,
        )
        .unwrap();
    let actual = calls.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(report.provider_calls(), actual);
    assert_eq!(
        ws.last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        actual
    );
    (output[0], report.termination(), report.terms(), actual)
}

#[test]
fn bounded_prefetch_preserves_budget_results_across_policies_and_precisions() {
    use scioxide_core::{BudgetKind, Complex, ExecutionPolicy, NumericScalar, RealScalar};
    fn check<T: NumericScalar>() {
        for inherited in [
            None,
            Some(BudgetKind::Evaluation),
            Some(BudgetKind::Iteration),
            Some(BudgetKind::Proposal),
        ] {
            for allowance in [1, 7, 31, 32, 33, 40, 63, 64] {
                let expected =
                    policy_probe::<T>(ExecutionPolicy::Sequential, allowance, inherited, false);
                for policy in
                    core::iter::once(ExecutionPolicy::Auto).chain((1..=4).map(|workers| {
                        ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(workers).unwrap())
                    }))
                {
                    let actual = policy_probe::<T>(policy, allowance, inherited, false);
                    assert_eq!(actual, expected, "{policy:?}, {inherited:?}, {allowance}");
                    for (a, b) in [
                        (actual.0.real_component(), expected.0.real_component()),
                        (
                            actual.0.imaginary_component(),
                            expected.0.imaginary_component(),
                        ),
                    ] {
                        assert_eq!(a.into_f64().to_bits(), b.into_f64().to_bits());
                    }
                    if allowance == 40
                        && inherited.is_none_or(|kind| kind == BudgetKind::Evaluation)
                    {
                        assert_eq!(actual.2, 40);
                        assert_eq!(actual.3, 40);
                    }
                }
            }
        }
    }
    check::<f64>();
    check::<Complex<f32>>();
}

#[test]
fn early_convergence_discards_less_than_one_bounded_batch() {
    use scioxide_core::{Complex, ExecutionPolicy, NumericScalar};
    fn check<T: NumericScalar>() {
        let expected = policy_probe::<T>(ExecutionPolicy::Sequential, 64, None, true);
        assert_eq!(expected.1, SeriesTermination::HeuristicToleranceReached);
        for policy in core::iter::once(ExecutionPolicy::Auto).chain((1..=4).map(|workers| {
            ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(workers).unwrap())
        })) {
            let actual = policy_probe::<T>(policy, 64, None, true);
            assert_eq!(
                (actual.0, actual.1, actual.2),
                (expected.0, expected.1, expected.2)
            );
            assert!(actual.3 >= expected.3 && actual.3 <= expected.3 + 31);
        }
    }
    check::<f64>();
    check::<Complex<f32>>();
}

#[test]
fn failing_prefetch_accepts_the_prefix_before_the_failure_and_preserves_atomic_output() {
    let p = SeriesSummation::<f64>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(64).unwrap())
            .with_order(Budget::try_from(10).unwrap())
            .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(
                core::num::NonZeroUsize::new(2).unwrap(),
            ))
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap()),
    )
    .unwrap();
    let provider = |n: usize, row: &mut [f64]| {
        if n == 40 {
            return Err(SciOxideError::external("second batch failure"));
        }
        let x = f64::from(u32::try_from(n + 1).unwrap());
        row[0] = 1.0 / (x * x);
        Ok(())
    };
    let mut state = p.state(1).unwrap();
    let mut ws = p.workspace(1).unwrap();
    let mut emitted = 0;
    let executor = scioxide_core::SequentialExecutor;
    assert!(
        p.advance(
            SeriesProblem::Terms(&provider),
            &mut state,
            |_| {
                emitted += 1;
                Ok(())
            },
            &mut ws,
            ExecutionContext::new(&executor)
        )
        .is_err()
    );
    // The second batch fails at index 40; its rows 32..40 are accepted first, as they
    // would be sequentially. The sequential executor stops after the failing partition.
    assert_eq!(state.terms(), 40);
    assert_eq!(emitted, 40);
    assert_eq!(
        ws.last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        41
    );
    let good = |n: usize, row: &mut [f64]| {
        let x = f64::from(u32::try_from(n + 1).unwrap());
        row[0] = 1.0 / (x * x);
        Ok(())
    };
    let epoch = ws.last_attempt().unwrap().epoch();
    let resumed = p
        .advance(
            SeriesProblem::Terms(&good),
            &mut state,
            |_| Ok(()),
            &mut ws,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert_eq!(resumed.terms(), 64);
    assert_eq!(ws.last_attempt().unwrap().epoch(), epoch);
    assert_eq!(
        ws.last_attempt()
            .unwrap()
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        65
    );
    let mut out = [99.0];
    assert!(
        p.sum_into(
            SeriesProblem::Terms(&provider),
            &mut out,
            &mut ws,
            ExecutionContext::new(&executor)
        )
        .is_err()
    );
    assert_eq!(out, [99.0]);
}

#[test]
fn convergence_before_a_prefetched_failure_reports_actual_provider_calls() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{ExecutionPolicy, SequentialExecutor};

    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        let plan = SeriesSummation::<f64>::new(
            method,
            0,
            SeriesOptions::default().with_execution_policy(ExecutionPolicy::Auto),
        )
        .unwrap();
        let calls = AtomicUsize::new(0);
        let terms = |n: usize, row: &mut [f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            if n == 24 {
                return Err(SciOxideError::external("failure after convergence"));
            }
            row[0] = 0.5_f64.powi(n as i32);
            Ok(())
        };
        let mut workspace = plan.workspace(1).unwrap();
        let mut output = [99.0];
        let report = plan
            .sum_into(
                SeriesProblem::Terms(&terms),
                &mut output,
                &mut workspace,
                ExecutionContext::new(&SequentialExecutor),
            )
            .unwrap();
        assert!(report.termination().is_success());
        assert!((output[0] - 2.0).abs() < 1e-12);
        let actual = calls.load(Ordering::Relaxed);
        assert_eq!(actual, 25);
        assert_eq!(report.provider_calls(), actual);
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            actual
        );
    }
}

fn windowed(method: SeriesMethod, terms: usize, order: usize) -> SeriesSummation<f64> {
    SeriesSummation::new(
        method,
        1,
        SeriesOptions::default()
            .with_max_terms(Budget::try_from(terms).unwrap())
            .with_order(Budget::try_from(order).unwrap())
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap()),
    )
    .unwrap()
}

#[test]
fn resumes_longer_than_the_window_match_atomic_sums_bit_for_bit() {
    let provider = |n: usize, row: &mut [f64]| {
        let x = n as f64;
        row.copy_from_slice(&[
            0.97_f64.powi(n as i32),
            if n.is_multiple_of(2) { 1.0 } else { -1.0 } / (x + 1.0),
        ]);
        Ok(())
    };
    for method in [SeriesMethod::LevinU, SeriesMethod::WynnEpsilon] {
        let atomic = windowed(method, 200, 4);
        let mut ws = atomic.workspace(2).unwrap();
        let mut expected = [0.0; 2];
        let report = atomic
            .sum_into(
                SeriesProblem::Terms(&provider),
                &mut expected,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.terms(), 200);
        assert_eq!(report.termination(), SeriesTermination::TermLimit);
        let expected_errors = ws.component_errors().to_vec();
        let progressive = SeriesSummation::new(
            method,
            1,
            atomic
                .options()
                .clone()
                .with_terms_per_advance(Budget::try_from(3).unwrap()),
        )
        .unwrap();
        let mut state = progressive.state(2).unwrap();
        let mut advances = 0;
        let report = loop {
            let report = progressive
                .advance(
                    SeriesProblem::Terms(&provider),
                    &mut state,
                    |_| Ok(()),
                    &mut ws,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            advances += 1;
            if report.termination() != SeriesTermination::CallLimit {
                break report;
            }
        };
        assert_eq!(advances, 67);
        assert_eq!(report.termination(), SeriesTermination::TermLimit);
        assert_eq!(state.terms(), 200);
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(state.value().unwrap()), bits(&expected));
        for (actual, expected) in state.errors().unwrap().iter().zip(&expected_errors) {
            assert_eq!(actual.acceleration(), expected.acceleration());
            assert_eq!(actual.roundoff().to_bits(), expected.roundoff().to_bits());
        }
    }
}

#[test]
fn prefetched_failures_accept_the_same_prefix_as_sequential_execution() {
    use scioxide_core::ExecutionPolicy;
    let alternating = |n: usize| {
        let magnitude = 1.0 / (n as f64 + 1.0);
        if n.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    };
    let failing = |n: usize, row: &mut [f64]| {
        if n == 5 {
            return Err(SciOxideError::external("fail at 5"));
        }
        row[0] = alternating(n);
        Ok(())
    };
    let nonfinite = |n: usize, row: &mut [f64]| {
        row[0] = if n == 7 { f64::NAN } else { alternating(n) };
        Ok(())
    };
    #[cfg(feature = "std")]
    let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(4).unwrap());
    #[cfg(not(feature = "std"))]
    let executor = scioxide_core::SequentialExecutor;
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Auto,
        ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(4).unwrap()),
    ] {
        let p = SeriesSummation::<f64>::new(
            SeriesMethod::WynnEpsilon,
            0,
            SeriesOptions::default()
                .with_max_terms(Budget::try_from(40).unwrap())
                .with_terms_per_advance(Budget::try_from(40).unwrap())
                .with_execution_policy(policy)
                .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap()),
        )
        .unwrap();
        let mut ws = p.workspace(1).unwrap();
        for (provider, accepted) in [
            (&failing as &dyn scioxide_integrate::SeriesTerms<f64>, 5),
            (&nonfinite, 7),
        ] {
            let mut state = p.state(1).unwrap();
            let mut emitted = 0;
            let error = p
                .advance(
                    SeriesProblem::Terms(provider),
                    &mut state,
                    |_| {
                        emitted += 1;
                        Ok(())
                    },
                    &mut ws,
                    ExecutionContext::new(&executor),
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::External { .. } | SciOxideErrorView::NonFinite { .. }
            ));
            assert_eq!((state.terms(), emitted), (accepted, accepted), "{policy:?}");
            assert_eq!(ws.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
        }
    }
}

#[test]
fn termination_displays_variant_name() {
    assert_eq!(
        SeriesTermination::HeuristicToleranceReached.to_string(),
        "HeuristicToleranceReached"
    );
    assert_eq!(
        SeriesTermination::ContextBudgetLimit(scioxide_core::BudgetKind::Evaluation).to_string(),
        "ContextBudgetLimit(Evaluation)"
    );
}
