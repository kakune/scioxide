//! Public no_std-compatible numerical contracts, exercised with injected capabilities.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, CancellationToken, ConvergenceReason, ErrorTolerance,
    EvaluationRole, ExecutionContext, ExecutionStopReason, Interval, NormSelection, ProblemToken,
    SciOxideErrorView, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_integrate::*;
use scioxide_random::{Philox4x64, SeedableRng};
use std::sync::atomic::{AtomicUsize, Ordering};

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("integrate", "contract-test", 1),
        0,
        0,
    )
}
fn run<T>(mut budgets: SolveBudgets, operation: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    operation(&mut context)
}
fn limit(kind: BudgetKind, amount: usize) -> SolveBudgets {
    SolveBudgets::unbounded().with_limit(kind, Budget::try_from(amount).unwrap())
}
fn zero_options() -> SubdivisionOptions<f64> {
    SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
}

#[test]
fn whole_rules_check_ancestor_allowances_without_charging_unused_work() {
    let calls = AtomicUsize::new(0);
    let f = |x: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0 / (1.0 + 100.0 * (x - 0.5) * (x - 0.5)))
    };
    let interval = Interval::new(0.0, 1.0).unwrap();
    let integrator = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, zero_options()).unwrap();
    let mut workspace = integrator.workspace(1).unwrap();
    run(limit(BudgetKind::Evaluation, 44), |parent| {
        let mut value = 7.0;
        let report = integrator
            .integrate_into(&f, interval, &mut value, &mut workspace, &mut *parent)
            .unwrap();
        assert_eq!(report.evaluations(), 15);
        assert_eq!(calls.load(Ordering::Relaxed), 15);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(value.is_finite() && value != 7.0);
        assert_eq!(parent.counts().own().provider(EvaluationRole::Function), 0);
        assert_eq!(
            parent.counts().nested().provider(EvaluationRole::Function),
            15
        );
        assert_eq!(
            parent.budget_report(BudgetKind::Evaluation).remaining(),
            Some(29)
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    });
    calls.store(0, Ordering::Relaxed);
    let integrator =
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default()).unwrap();
    run(limit(BudgetKind::Evaluation, 14), |parent| {
        let mut value = 7.0;
        let error = integrator
            .integrate_into(&f, interval, &mut value, &mut workspace, &mut *parent)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
        assert_eq!(value, 7.0);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            parent
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            0
        );
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        assert!(attempt.progress().is_none());
    });
}

#[test]
fn fixed_and_refinement_methods_respect_shared_limits() {
    let interval = Interval::new(0.0_f32, 1.0).unwrap();
    run(limit(BudgetKind::Evaluation, 2), |context| {
        let plan = GaussianRulePlan::gauss_legendre(3).unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        let mut output = 5.0_f32;
        let error = plan
            .integrate_into(
                |x: f32| Ok(x * x),
                interval,
                &mut output,
                &mut workspace,
                &mut *context,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
        assert_eq!(output, 5.0);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            0
        );
    });
    run(limit(BudgetKind::Evaluation, 2), |context| {
        let romberg = Romberg::new(LevelOptions::default()).unwrap();
        let mut workspace = romberg.workspace(1).unwrap();
        let mut output = 5.0_f32;
        let report = romberg
            .integrate_into(
                |x: f32| Ok(x * x),
                interval,
                &mut output,
                &mut workspace,
                &mut *context,
            )
            .unwrap();
        assert_eq!(report.evaluations(), 2);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        // The published estimate is the initial trapezoid.
        assert!((output - 0.5).abs() <= f32::EPSILON);
    });
    run(limit(BudgetKind::Subdivision, 1), |context| {
        let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, zero_options()).unwrap();
        let mut workspace = simpson.workspace(1).unwrap();
        let mut output = 0.0;
        let report = simpson
            .integrate_into(
                |x: f64| Ok(x.exp()),
                Interval::new(0.0, 1.0).unwrap(),
                &mut output,
                &mut workspace,
                &mut *context,
            )
            .unwrap();
        assert_eq!(report.evaluations(), 18);
        assert_eq!(context.budget_report(BudgetKind::Subdivision).consumed(), 1);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Subdivision)
        );
    });
}

#[test]
fn cancellation_preserves_the_last_complete_domain_estimate() {
    let cancellation = CancellationToken::new();
    let calls = AtomicUsize::new(0);
    let f = |x: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        cancellation.cancel();
        Ok(1.0 / (1.0 + 100.0 * (x - 0.5) * (x - 0.5)))
    };
    let integrator = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, zero_options()).unwrap();
    let mut workspace = integrator.workspace(1).unwrap();
    let mut value = 0.0;
    let report = integrator
        .integrate_into(
            &f,
            Interval::new(0.0, 1.0).unwrap(),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 15);
    assert_eq!(
        report.termination(),
        IntegrationTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    let (initial, _) = GaussKronrod::new(GaussKronrodRule::G7K15)
        .unwrap()
        .integrate(
            |x: f64| Ok(1.0 / (1.0 + 100.0 * (x - 0.5) * (x - 0.5))),
            Interval::new(0.0, 1.0).unwrap(),
        )
        .unwrap();
    assert!((value - initial).abs() <= 4.0 * f64::EPSILON * initial);
}

#[test]
fn failed_callback_attempts_are_counted_and_workspace_can_be_reused() {
    let solver =
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default()).unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    let unit = Interval::new(0.0, 1.0).unwrap();
    let mut value = 7.0;
    run(SolveBudgets::unbounded(), |context| {
        let error = solver
            .integrate_into(
                |_: f64| Ok(f64::NAN),
                unit,
                &mut value,
                &mut workspace,
                &mut *context,
            )
            .unwrap_err();
        assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
        let counts = context.counts().total().unwrap();
        assert_eq!(counts.provider(EvaluationRole::Function), 1);
        assert_eq!(counts.accepted(), 0);
    });
    assert_eq!(value, 7.0);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    let report = solver
        .integrate_into(
            |x: f64| Ok(x * x),
            unit,
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((value - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn scalar_error_policy_rejects_incompatible_component_shapes() {
    let wrong = ErrorTolerance::try_vector(&[1e-8, 1e-8], 1e-8, NormSelection::Infinity).unwrap();
    let solver = AdaptiveIntegrator::new(
        AdaptiveRule::Simpson,
        SubdivisionOptions::default().with_tolerance(wrong),
    )
    .unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    let calls = AtomicUsize::new(0);
    let mut value = 7.0;
    let error = solver
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(x * x)
            },
            Interval::new(0.0, 1.0).unwrap(),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(value, 7.0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-6_f32, 1e-6, NormSelection::Infinity).unwrap());
    let (value, _) = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options)
        .unwrap()
        .integrate(|x: f32| Ok(x * x), Interval::new(0.0, 1.0).unwrap())
        .unwrap();
    assert!((value - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn deadline_stops_before_the_first_provider() {
    struct FixedClock;
    impl scioxide_core::Clock for FixedClock {
        fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(1),
            ))
        }
    }
    let clock = FixedClock;
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let deadline = scioxide_core::Deadline::at(
        &clock_binding,
        scioxide_core::MonotonicInstant::from_duration_since_epoch(core::time::Duration::ZERO),
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential()
                .with_deadline(deadline)
                .unwrap(),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let plan = GaussianRulePlan::gauss_legendre(3).unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    let mut output = 5.0_f32;
    let error = plan
        .integrate_into(
            |x: f32| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(x)
            },
            Interval::new(0.0, 1.0).unwrap(),
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::DeadlineReached
        }
    ));
    assert_eq!(output, 5.0);
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn monte_carlo_callback_failure_counts_attempts_and_never_advances_the_root() {
    let bounds = [Interval::new(0.0_f32, 1.0).unwrap(); 2];
    let plan = MonteCarloIntegrator::new(
        &bounds,
        MonteCarloOptions::default().with_samples(Budget::try_from(10).unwrap()),
    )
    .unwrap();
    let root = Philox4x64::from_seed(42);
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let mut value = 5.0_f32;
    run(SolveBudgets::unbounded(), |context| {
        let result = plan.integrate_into(
            |_: &[f32]| Ok(f32::NAN),
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            &mut *context,
        );
        assert!(matches!(
            result.as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Callback,
                ..
            }
        ));
        let total = context.counts().total().unwrap();
        assert_eq!(total.provider(EvaluationRole::Sample), 2);
        assert_eq!(total.provider(EvaluationRole::Function), 1);
        assert_eq!(context.counts().own().accepted(), 0);
    });
    assert_eq!(value, 5.0);
    assert_eq!(scioxide_random::BitGenerator::position(&root), Some(0));
}
#[test]
fn scalar_width_and_changed_domains_do_not_leave_stale_workspace_state() {
    let f = |x: &[f32]| Ok(x[0] * x[0] + x[1]);
    let initial = [Interval::new(0.0, 1.0).unwrap(); 2];
    let mut ws = CubaturePlan::new(&initial, SubdivisionOptions::default())
        .unwrap()
        .workspace(1)
        .unwrap();
    for width in [1.0_f32, 2.0, 0.0, 1.0] {
        let bounds = [Interval::new(0.0, width).unwrap(); 2];
        let plan = CubaturePlan::new(&bounds, SubdivisionOptions::default()).unwrap();
        let expected = plan.integrate(&f).unwrap();
        let mut value = 99.0;
        let report = run(SolveBudgets::unbounded(), |parent| {
            plan.integrate_into(&f, &mut value, &mut ws, parent)
        })
        .unwrap();
        assert_eq!((value, report), expected);
    }
}

#[test]
fn invalid_random_provider_output_is_counted_before_validation() {
    #[derive(Clone)]
    struct InvalidUniform(f32);
    impl scioxide_random::BitGenerator for InvalidUniform {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }
        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }
    impl scioxide_random::Rng<f32> for InvalidUniform {
        fn next_float(&mut self) -> Result<f32, scioxide_core::SciOxideError> {
            Ok(self.0)
        }
    }
    impl scioxide_random::SplittableRng for InvalidUniform {
        fn split(&self, _: u64) -> Result<Self, scioxide_core::SciOxideError> {
            Ok(self.clone())
        }
        fn substream(&self, _: u64, _: u64) -> Result<Self, scioxide_core::SciOxideError> {
            Ok(self.clone())
        }
    }
    let plan = MonteCarloIntegrator::new(
        &[Interval::new(0.0, 1.0).unwrap()],
        MonteCarloOptions::default().with_samples(Budget::try_from(2).unwrap()),
    )
    .unwrap();
    for value in [f32::NAN, 1.0] {
        run(SolveBudgets::unbounded(), |context| {
            let root = InvalidUniform(value);
            let result = plan.integrate_into(
                |x: &[f32]| Ok(x[0]),
                RandomStreams::new(&root, 0),
                &mut 0.0,
                &mut plan
                    .workspace(1, ExecutionContext::sequential_native())
                    .unwrap(),
                &mut *context,
            );
            if value.is_nan() {
                assert!(matches!(
                    result.as_ref().err().unwrap().view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
            } else {
                assert!(matches!(
                    result.as_ref().err().unwrap().view(),
                    scioxide_core::SciOxideErrorView::InvalidInput { .. }
                ));
            }
            let total = context.counts().total().unwrap();
            assert_eq!(total.provider(EvaluationRole::Sample), 1);
            assert_eq!(total.provider(EvaluationRole::Function), 0);
        });
    }
}
