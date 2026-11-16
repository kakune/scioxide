//! Analytic fixed-rule affine integration, budget and allocation coverage.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Complex, ExecutionContext, Interval, ProblemToken,
    SciOxideErrorView, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_integrate::{
    FixedRule, GaussianRuleOptions, GaussianRulePlan, GaussianRuleVariant, GaussianSupport,
    IntegrationTermination,
};
use scioxide_poly::OrthogonalFamily;
use std::sync::atomic::{AtomicUsize, Ordering};

fn run<T>(mut budgets: SolveBudgets, operation: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(466),
        scioxide_core::algorithm_id!("integrate", "affine-test", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    operation(&mut context)
}

#[test]
fn analytic_complex_integrals_in_both_precisions() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let rule = GaussianRulePlan::<$s>::gauss_legendre(3).unwrap();
            let mut workspace = rule.workspace(2).unwrap();
            let mut output = Complex::new(9.0 as $s, 9.0 as $s);
            let report = rule
                .integrate_into(
                    |x: $s| Ok(Complex::new(x * x, 1.0 as $s + 2.0 as $s * x)),
                    Interval::new(-1.0 as $s, 1.0 as $s).unwrap(),
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                IntegrationTermination::FixedRuleCompleted
            );
            assert_eq!(report.evaluations(), 3);
            assert!(report.error_estimate().is_none());
            assert!((output.re - 2.0 as $s / 3.0 as $s).abs() < $tol);
            assert!((output.im - 2.0 as $s).abs() < $tol);
        }};
    }
    check!(f32, 3e-6);
    check!(f64, 3e-14);
}

#[test]
fn preflight_budget_shape_and_callback_failures_never_publish_output() {
    let rule = GaussianRulePlan::<f64>::gauss_legendre(3).unwrap();
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    let mut output = [7.0, 8.0];
    let mut workspace = rule.workspace(2).unwrap();
    let calls = AtomicUsize::new(0);
    let function = |x: f64| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok([x, x * x])
    };
    let budget =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
    run(budget, |context| {
        let accepted = context.accepted_state();
        let error = rule
            .integrate_into(
                &function,
                interval,
                &mut output,
                &mut workspace,
                &mut *context,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ConvergenceFailure { .. }
        ));
        assert_eq!(context.accepted_state(), accepted);
    });
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(output, [7.0, 8.0]);

    // A callback whose value has the wrong shape fails inside the rule.
    let broken = |_: f64| Ok(vec![1.0]);
    let error = rule
        .integrate_into(
            &broken,
            interval,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, [7.0, 8.0]);

    // A wrong output shape is a preflight rejection.
    let error = rule
        .integrate_into(
            &function,
            interval,
            &mut output[..1],
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let report = rule
        .integrate_into(
            &function,
            interval,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((output[0] - 0.5).abs() < 1e-14);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn zero_width_has_no_callbacks_and_empty_values_are_supported() {
    let rule = GaussianRulePlan::<f32>::gauss_legendre(2).unwrap();
    let calls = AtomicUsize::new(0);
    let callback = |x: f32| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x)
    };
    let (value, report) = rule
        .integrate(&callback, Interval::new(1.0, 1.0).unwrap())
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::DegenerateDomain
    );
    assert_eq!(value, 0.0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let rule = GaussianRulePlan::<f64>::gauss_legendre(2).unwrap();
    let mut workspace = rule.workspace(0).unwrap();
    let report = rule
        .integrate_into(
            |_: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok([0.0_f64; 0])
            },
            Interval::new(0.0, 1.0).unwrap(),
            &mut [] as &mut [f64],
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(report.evaluations(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn cancellation_before_the_rule_preserves_output_and_accepted_state() {
    let cancellation = scioxide_core::CancellationToken::new();
    let rule = GaussianRulePlan::<f64>::gauss_legendre(3).unwrap();
    let mut workspace = rule.workspace(1).unwrap();
    let mut output = 7.0;
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(466),
        scioxide_core::algorithm_id!("integrate", "affine-cancel", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity,
            &mut budgets,
        )
        .unwrap();
    let accepted = context.accepted_state();
    cancellation.cancel();
    let calls = AtomicUsize::new(0);
    let error = rule
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
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
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(output, 7.0);
    assert_eq!(context.accepted_state(), accepted);
}

#[test]
fn endpoint_and_weighted_rules_preserve_mapped_moments() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            for (family, variant, count, mass) in [
                (
                    OrthogonalFamily::Legendre,
                    GaussianRuleVariant::RadauLeft,
                    1,
                    4.0 as $s,
                ),
                (
                    OrthogonalFamily::Legendre,
                    GaussianRuleVariant::RadauRight,
                    1,
                    4.0 as $s,
                ),
                (
                    OrthogonalFamily::Legendre,
                    GaussianRuleVariant::Lobatto,
                    2,
                    4.0 as $s,
                ),
                (
                    OrthogonalFamily::ChebyshevT,
                    GaussianRuleVariant::Gauss,
                    2,
                    core::f64::consts::PI as $s,
                ),
            ] {
                let rule = GaussianRulePlan::with_options(
                    family,
                    count,
                    variant,
                    GaussianRuleOptions::default(),
                )
                .unwrap();
                let mut workspace = rule.workspace(2).unwrap();
                let mut output = [9.0 as $s; 2];
                let report = rule
                    .integrate_weighted_into(
                        |x: $s| Ok([1.0 as $s, x]),
                        GaussianSupport::Interval(Interval::new(2.0 as $s, 6.0 as $s).unwrap()),
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert!(report.is_success());
                assert_eq!(report.evaluations(), count);
                assert!((output[0] - mass).abs() < $tol);
                // One-node Radau has degree zero; its selected endpoint is exact.
                let mean = match variant {
                    GaussianRuleVariant::RadauLeft => 2.0 as $s,
                    GaussianRuleVariant::RadauRight => 6.0 as $s,
                    _ => 4.0 as $s,
                };
                assert!((output[1] - mass * mean).abs() < $tol);
            }
        }};
    }
    check!(f32, 2e-5);
    check!(f64, 2e-13);
}
