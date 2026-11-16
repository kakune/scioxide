//! FFTLog preparation accounting, scalar-width operators and transactional execution.
#[path = "support/execution.rs"]
mod execution;
use execution::FixedClock;

use scioxide_core::{
    AdjointOperator, Budget, BudgetKind, Complex, ExecutionContext, ExecutionPolicy,
    LinearOperator, NdView, NdViewMut, SciOxideErrorKind, SolveBudgets, SolveState, StridedLayout,
};
use scioxide_fft::{FftLogLimits, FftLogPlan, FftLogPlanSpec};
#[path = "support/numerics.rs"]
mod numerics;
#[path = "support/provider.rs"]
mod provider;

fn identity() -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(512),
        scioxide_core::algorithm_id!("fftlog", "context-test", 1),
        0,
        0,
    )
}
fn spec() -> FftLogPlanSpec<f64> {
    FftLogPlanSpec::new(
        8,
        0.125,
        0.0,
        0.0,
        Complex::new(0.25, 0.125),
        scioxide_fft::FftDirection::Forward,
    )
    .with_execution_policy(ExecutionPolicy::Sequential)
    .with_limits(FftLogLimits::default())
}
#[test]
fn preparation_accounts_for_all_nested_special_work() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let before = context.counts().nested().iterations();
    let p = FftLogPlan::build(spec(), context.execution(), &mut context).unwrap();
    let used = context.counts().nested().iterations() - before;
    let input = [Complex::new(1.0, 0.25); 8];
    let mut out = [Complex::new(0.0, 0.0); 8];
    p.execute_into(
        &input,
        &mut out,
        &mut p.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    assert_eq!(used, p.diagnostics().preparation_work);
    let ordinary = FftLogPlan::new(spec()).unwrap();
    let mut expected = out;
    ordinary
        .execute_into(
            &input,
            &mut expected,
            &mut ordinary.workspace(()).unwrap(),
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    assert_eq!(out, expected);
}
#[test]
fn preparation_uses_its_local_iteration_allowance() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut child = SolveState::new();
    let result = scioxide_core::SolveScope::from(&mut context).run(
        &mut child,
        identity(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
        |context| {
            {
                FftLogPlan::build(
                    spec().with_limits(FftLogLimits {
                        preparation_work: Budget::ONE,
                        ..FftLogLimits::default()
                    }),
                    context.execution(),
                    context,
                )
                .map(|_| ())
            }
        },
    );
    let error = result.unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let scioxide_core::SciOxideErrorView::ConvergenceFailure {
        reason: _failure_reason,
        diagnostics: _failure_diagnostics,
        detail: _failure_detail,
    } = error.view()
    else {
        unreachable!("unexpected error kind")
    };
    assert_eq!(
        _failure_reason,
        scioxide_core::ConvergenceReason::IterationLimit
    );
    assert!(_failure_diagnostics.counts().own().iterations() >= 1);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 1);
}
#[test]
fn execution_and_operator_admission_leave_output_atomic() {
    let f = FftLogPlan::new(spec()).unwrap();
    let limited_f =
        FftLogPlan::new(spec().with_iteration_budget(Budget::try_from(3).unwrap())).unwrap();
    let input = [Complex::new(1.0, 0.25); 8];
    let initial = [Complex::new(123.0, 456.0); 8];

    let mut scratch = { f.workspace(()).unwrap() };
    let mut output = initial;
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(3).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let result = {
        limited_f
            .apply_into_scoped(
                &input,
                &mut output,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context),
            )
            .and_then(scioxide_core::Invocation::require)
    };
    assert!(result.is_err());
    assert_eq!(output, initial);
    assert_eq!(context.counts().nested().iterations(), 0);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    {
        f.apply_into_scoped(
            &input,
            &mut output,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
    }
    .unwrap();
    assert_eq!(context.counts().nested().iterations(), 8);
}
#[test]
fn f32_operator_uses_native_caller_workspace() {
    let p = FftLogPlan::new(
        FftLogPlanSpec::new(
            16,
            0.125,
            0.0_f32,
            0.0,
            Complex::new(0.25, 0.125),
            scioxide_fft::FftDirection::Forward,
        )
        .with_execution_policy(ExecutionPolicy::Sequential)
        .with_limits(FftLogLimits::default()),
    )
    .unwrap();
    let input: Vec<_> = (0..16)
        .map(|j| Complex::new((j as f32 * 0.3).sin(), (j as f32 * 0.2).cos()))
        .collect();
    let probe: Vec<_> = (0..16)
        .map(|j| Complex::new((j as f32 * 0.5).cos(), (j as f32 * 0.1).sin()))
        .collect();
    let mut expected = vec![Complex::new(0.0, 0.0); 16];
    let mut workspace = p.workspace(()).unwrap();
    p.execute_into(
        &input,
        &mut expected,
        &mut workspace,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    let mut actual = expected.clone();
    let mut adjoint = expected.clone();
    let op = &p;
    assert_eq!(op.apply_workspace_len().unwrap(), workspace.len());
    op.apply_into(&input, &mut actual, &mut workspace).unwrap();
    assert_eq!(actual, expected);
    op.apply_adjoint_into(&probe, &mut adjoint, &mut workspace)
        .unwrap();
    let dot = |a: &[Complex<f32>], b: &[Complex<f32>]| {
        a.iter()
            .zip(b)
            .fold(Complex::new(0.0, 0.0), |s, (x, y)| s + x.conj() * *y)
    };
    assert!((dot(&actual, &probe) - dot(&input, &adjoint)).norm() < 2e-5);
    let before = actual.clone();
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    assert!(
        op.apply_into_scoped(
            &input,
            &mut actual,
            &mut workspace,
            &mut scioxide_core::OperatorContext::new(&mut context)
        )
        .and_then(scioxide_core::Invocation::require)
        .is_err()
    );
    assert_eq!(actual, before);
}
#[test]
fn axis_lanes_share_one_parent_and_commit_together() {
    let p = FftLogPlan::new(spec()).unwrap();
    let input = [1.0_f64; 16];
    let initial = [Complex::new(123.0, 0.0); 16];
    let mut output = initial;
    let shape = [2, 8];
    let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
    let mut scratch = scioxide_fft::FftWorkspace::from_vec(vec![
        Complex::new(0.0, 0.0);
        p.workspace_requirements(
            scioxide_fft::AxisWorkspaceRequest::Axis {
                shape: &shape,
                axis: 1
            }
        )
        .map(|r| r.complex_len())
        .unwrap()
    ]);
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(35).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut bad = input;
    bad[15] = f64::NAN;
    assert!(
        p.execute_axis_into(
            NdView::new(&bad, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, layout).unwrap(),
            &mut scratch,
            &mut context,
        )
        .is_err()
    );
    assert_eq!(output, initial);
    assert_eq!(context.counts().own().iterations(), 0);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    p.execute_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(&mut output, layout).unwrap(),
        &mut scratch,
        &mut context,
    )
    .unwrap();
    assert_eq!(context.counts().nested().iterations(), 2 * 8);
}

#[test]
fn fftlog_provider_limits_errors_and_evaluation_admission() {
    use scioxide_fft::{FftPlanningContext, FftProviderPolicy};
    use std::sync::atomic::Ordering;
    let input = [Complex::new(1.0_f64, 0.25); 8];
    let initial = [Complex::new(123.0, 456.0); 8];
    let cpu = FftLogPlan::new(spec()).unwrap();
    let mut expected = initial;
    cpu.execute_into(
        &input,
        &mut expected,
        &mut cpu.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    for mode in [
        provider::Mode::Direct,
        provider::Mode::ExecutionFailure,
        provider::Mode::NonFinite,
    ] {
        let provider = provider::DirectProvider::new(mode);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut config = spec();
        config = config.with_limits(FftLogLimits {
            preparation_bytes: 100_000,
            ..config.limits()
        });
        config = config.with_limits(FftLogLimits {
            workspace_bytes: 5000,
            ..config.limits()
        });
        let planning = FftPlanningContext::with_provider(
            context.execution(),
            &provider,
            FftProviderPolicy::Required,
        );
        let p = FftLogPlan::build(config, planning, &mut context).unwrap();
        assert!(
            provider.preparation_limit.load(Ordering::Relaxed) < config.limits().preparation_bytes
        );
        assert!(provider.workspace_limit.load(Ordering::Relaxed) < config.limits().workspace_bytes);
        let mut actual = initial;
        let mut scratch = p.workspace(()).unwrap();
        let result = p.execute_into(&input, &mut actual, &mut scratch, &mut context);
        if matches!(mode, provider::Mode::Direct) {
            result.unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (*a - b).norm() < 1e-12)
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), 2);
        } else {
            assert!(result.is_err());
            assert_eq!(actual, initial);
        }
    }
    let provider = provider::DirectProvider::new(provider::Mode::Direct);
    let p = FftLogPlan::build(
        spec(),
        FftPlanningContext::with_provider(
            ExecutionContext::sequential(),
            &provider,
            FftProviderPolicy::Required,
        ),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let mut scratch = p.workspace(()).unwrap();
    let mut actual = initial;
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    p.execute_into(&input, &mut actual, &mut scratch, &mut context)
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed), 2);
    let admitted = actual;
    let error = p
        .execute_into(&input, &mut actual, &mut scratch, &mut context)
        .unwrap_err();
    let scioxide_core::SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        ..
    } = error.view()
    else {
        unreachable!("provider allowance must produce a convergence failure");
    };
    assert_eq!(reason, scioxide_core::ConvergenceReason::EvaluationLimit);
    let report = diagnostics.budget().unwrap();
    assert_eq!(report.kind(), BudgetKind::Evaluation);
    assert_eq!(report.consumed(), 1);
    assert_eq!(report.remaining(), Some(0));
    assert_eq!(diagnostics.counts().total().unwrap().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 8);
    assert_eq!(
        diagnostics
            .counts()
            .own()
            .provider(scioxide_core::EvaluationRole::LinearOperator),
        0
    );
    assert_eq!(actual, admitted);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn scoped_fftlog_adjoints_inherit_stops_and_account_for_nested_work() {
    use scioxide_core::{ClockBinding, Deadline, ExecutionStopReason, MonotonicInstant};
    let f = FftLogPlan::new(spec()).unwrap();
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let clock = FixedClock;
    let binding = ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &binding,
        MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
    );
    let input = [Complex::new(1.0, 0.25); 8];
    let initial = [Complex::new(99.0, 77.0); 8];
    for (op, iterations) in [(&f as &dyn AdjointOperator<Complex<f64>>, 8)] {
        let mut scratch = vec![Complex::new(0.0, 0.0); op.apply_adjoint_workspace_len().unwrap()];
        let mut reference = initial;
        op.apply_adjoint_into(&input, &mut reference, &mut scratch)
            .unwrap();
        for mode in 0..3 {
            let execution = match mode {
                1 => ExecutionContext::sequential().with_cancellation(&token),
                2 => ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                _ => ExecutionContext::sequential(),
            };
            let mut state = SolveState::new();
            let mut budgets =
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
            let mut context = state
                .begin_solve(execution, identity(), &mut budgets)
                .unwrap();
            assert!(context.record_iteration().unwrap().is_granted());
            let mut output = initial;
            let result = op
                .apply_adjoint_into_scoped(
                    &input,
                    &mut output,
                    &mut scratch,
                    &mut scioxide_core::OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require);
            if mode == 0 {
                result.unwrap();
                assert_eq!(output, reference);
                assert_eq!(context.counts().nested().iterations(), iterations);
            } else {
                assert!(
                    matches!(result.unwrap_err().view(), scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. }
                    if reason == if mode == 1 { ExecutionStopReason::Cancelled } else { ExecutionStopReason::DeadlineReached })
                );
                assert_eq!(output, initial);
                assert_eq!(context.counts().nested().iterations(), 0);
            }
            assert_eq!(context.budget_report(BudgetKind::Iteration).consumed(), 1);
        }
    }
}

#[test]
fn fftlog_bulk_admission_is_exact_for_zero_and_nonzero_lanes() {
    for n in [1, 7, 8] {
        for zero in [false, true] {
            let config = FftLogPlanSpec::new(
                n,
                0.125,
                0.0,
                0.0,
                Complex::new(0.0, 0.0),
                scioxide_fft::FftDirection::Forward,
            );
            let p = FftLogPlan::new(config.with_iteration_budget(Budget::try_from(n).unwrap()))
                .unwrap();
            let input = vec![Complex::new(if zero { 0.0 } else { 1.0 }, 0.0); n];
            let initial = vec![Complex::new(123.0, 0.0); n];
            let mut output = initial.clone();
            let mut work = p.workspace(()).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            p.execute_into(&input, &mut output, &mut work, &mut context)
                .unwrap();
            assert_eq!(context.counts().nested().iterations(), n);
            if n > 1 {
                let limited =
                    FftLogPlan::new(config.with_iteration_budget(Budget::try_from(n - 1).unwrap()))
                        .unwrap();
                output.clone_from(&initial);
                assert!(
                    limited
                        .execute_into(&input, &mut output, &mut work, &mut context)
                        .is_err()
                );
                assert_eq!(output, initial);
                assert_eq!(context.counts().nested().iterations(), n);
            }
        }
    }
}

#[test]
fn f32_fftlog_provider_uses_native_workspace() {
    use scioxide_fft::{FftPlanningContext, FftProviderPolicy, FftWorkspace};
    let provider = provider::DirectProvider::new(provider::Mode::Direct);
    let config = FftLogPlanSpec::new(
        8,
        0.125,
        0.0_f32,
        0.0,
        Complex::new(0.25, 0.125),
        scioxide_fft::FftDirection::Forward,
    );
    let plan = FftLogPlan::build(
        config,
        FftPlanningContext::with_provider(
            ExecutionContext::sequential(),
            &provider,
            FftProviderPolicy::Required,
        ),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let native = FftLogPlan::new(config).unwrap();
    let input: Vec<_> = (0..8)
        .map(|j| Complex::new(j as f32 / 8.0, -0.25))
        .collect();
    let mut actual = input.clone();
    let mut expected = input.clone();
    let mut workspace: FftWorkspace<f32> = plan.workspace(()).unwrap();
    assert_eq!(
        plan.workspace_requirements(()).unwrap().bytes(),
        workspace.len() * std::mem::size_of::<Complex<f32>>()
    );
    plan.apply_into(&input, &mut actual, &mut workspace)
        .unwrap();
    native
        .apply_into(&input, &mut expected, &mut native.workspace(()).unwrap())
        .unwrap();
    for (a, b) in actual.iter().zip(expected) {
        assert!((*a - b).norm() <= 2e-5);
    }
}

#[path = "support/operator_accounting.rs"]
mod operator_accounting;

#[test]
fn operator_evaluations_are_one_per_application_or_nonempty_axis() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::AxisWorkspaceRequest;
    use scioxide_fft::{FftPlanningContext, FftProviderPolicy};
    let provider = provider::DirectProvider::new(provider::Mode::Direct);
    for supplied in [false, true] {
        let planning = if supplied {
            FftPlanningContext::with_provider(
                scioxide_core::ExecutionContext::sequential_native(),
                &provider,
                FftProviderPolicy::Required,
            )
        } else {
            FftPlanningContext::default()
        };
        let plan = FftLogPlan::build(
            spec(),
            planning,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        for value in [0.0, 1.0] {
            let input = [Complex::new(value, 0.0); 8];
            let mut work = plan.workspace(()).unwrap();
            operator_accounting::admission(8, |output, context| {
                plan.execute_into(&input, output, &mut work, context)
            });
            operator_accounting::operators(&plan, &input);
            let shape = [2, 8];
            let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
            let mut work = plan
                .workspace(AxisWorkspaceRequest::Axis {
                    shape: &shape,
                    axis: 1,
                })
                .unwrap();
            operator_accounting::admission(16, |output, context| {
                plan.execute_axis_into(
                    NdView::new(&[value; 16], layout).unwrap(),
                    1,
                    NdViewMut::new(output, layout).unwrap(),
                    &mut work,
                    context,
                )
            });
            let shape = [0, 8];
            let layout = StridedLayout::new(&shape, &[8, 1]).unwrap();
            let mut work = plan
                .workspace(AxisWorkspaceRequest::Axis {
                    shape: &shape,
                    axis: 1,
                })
                .unwrap();
            operator_accounting::empty_axis(|context| {
                plan.execute_axis_into(
                    NdView::new(&[] as &[f64], layout).unwrap(),
                    1,
                    NdViewMut::new(&mut [], layout).unwrap(),
                    &mut work,
                    context,
                )
            });
        }
    }
}

#[test]
fn fftlog_honors_solve_cancellation_independently_of_provider_capabilities() {
    use scioxide_fft::{
        FftLogPlan, FftLogPlanSpec, FftPlanningContext, FftProvider, FftProviderLimits,
        FftProviderPolicy,
    };
    struct Cancelling<'a> {
        token: &'a scioxide_core::CancellationToken,
        inner: provider::DirectProvider,
    }
    impl FftProvider<f64> for Cancelling<'_> {
        fn prepare(
            &self,
            n: usize,
            policy: scioxide_core::ExecutionPolicy,
            limits: FftProviderLimits,
            cx: ExecutionContext<'_>,
        ) -> Result<
            Option<std::sync::Arc<dyn scioxide_fft::FftProviderPlan<f64>>>,
            scioxide_core::SciOxideError,
        > {
            self.token.cancel();
            self.inner.prepare(n, policy, limits, cx)
        }
    }
    let token = scioxide_core::CancellationToken::new();
    let provider = Cancelling {
        token: &token,
        inner: provider::DirectProvider::new(provider::Mode::Direct),
    };
    let spec = FftLogPlanSpec::new(
        8,
        0.125,
        0.0,
        0.0,
        Complex::new(0.0, 0.0),
        scioxide_fft::FftDirection::Forward,
    );
    let planning = FftPlanningContext::with_provider(
        ExecutionContext::sequential(),
        &provider,
        FftProviderPolicy::Required,
    );
    let scope = ExecutionContext::sequential().with_cancellation(&token);
    assert_eq!(
        FftLogPlan::build(spec, planning, scope).unwrap_err().kind(),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(
        provider
            .inner
            .preparations
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    {
        assert_eq!(
            FftLogPlan::build(spec, planning, scope).unwrap_err().kind(),
            SciOxideErrorKind::ExecutionStopped
        );
    };
    assert_eq!(
        provider
            .inner
            .preparations
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}
