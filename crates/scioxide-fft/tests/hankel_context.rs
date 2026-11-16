//! Hankel preparation accounting, scalar-width operators and transactional execution.
#![expect(clippy::unwrap_used)]
#[path = "support/execution.rs"]
mod execution;
use execution::FixedClock;
#[path = "support/hankel_zeros.rs"]
mod hankel_zeros;

use scioxide_core::{
    AdjointOperator, Budget, BudgetKind, Complex, ExecutionContext, ExecutionPolicy,
    LinearOperator, SciOxideErrorKind, SolveBudgets, SolveContext, SolveState, Tolerance,
};
use scioxide_fft::{DiscreteHankelPlan, FftWorkspace, HankelLimits};
#[cfg(feature = "std")]
#[path = "support/executors.rs"]
mod executors;
#[path = "support/numerics.rs"]
mod numerics;
#[cfg(feature = "std")]
use executors::BrokenExecutor;
#[path = "support/provider.rs"]
mod provider;

fn identity() -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(512),
        scioxide_core::algorithm_id!("hankel", "context-test", 1),
        0,
        0,
    )
}
fn dht(
    context: &mut SolveContext<'_, '_>,
) -> Result<DiscreteHankelPlan<f64>, scioxide_core::SciOxideError> {
    DiscreteHankelPlan::build(
        &hankel_zeros::zeros(0.0)[..(8) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_limits(HankelLimits::default()),
        context,
    )
}

#[test]
fn f32_context_and_operators_use_native_caller_workspace() {
    use scioxide_core::{EvaluationRole, OperatorContext};
    use scioxide_fft::{DiscreteHankelPlanSpec, FftDirection, FftWorkspace};
    let n = 16;
    let zeros = hankel_zeros::zeros(0.5_f32);
    let input: Vec<_> = (0..n)
        .map(|j| Complex::new((j as f32 * 0.3).sin(), (j as f32 * 0.2).cos()))
        .collect();
    let probe: Vec<_> = (0..n)
        .map(|j| Complex::new((j as f32 * 0.5).cos(), (j as f32 * 0.1).sin()))
        .collect();
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let tolerance = 32.0 * f32::EPSILON;
        let spec = DiscreteHankelPlanSpec::new(0.5, 4.0, direction)
            .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap())
            .with_execution_policy(ExecutionPolicy::Sequential);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential_native(),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let plan = DiscreteHankelPlan::build(&zeros[..n + 1], spec, &mut context).unwrap();
        let preparation = context.counts().nested().iterations();
        assert_eq!(preparation, plan.diagnostics().preparation_work);
        assert_eq!(context.counts().own().iterations(), 0);
        let ordinary = DiscreteHankelPlan::new(&zeros[..n + 1], spec).unwrap();
        assert_eq!(plan.nodes(), ordinary.nodes());
        assert_eq!(plan.frequencies(), ordinary.frequencies());
        let mut workspace: FftWorkspace<f32> = plan.workspace(()).unwrap();
        assert_eq!(
            plan.workspace_requirements(()).unwrap().bytes(),
            workspace.len() * size_of::<Complex<f32>>()
        );
        assert_eq!(plan.apply_workspace_len().unwrap(), workspace.len());
        assert_eq!(plan.apply_adjoint_workspace_len().unwrap(), workspace.len());
        let mut expected = vec![Complex::new(0.0, 0.0); n];
        ordinary
            .execute_into(
                &input,
                &mut expected,
                &mut ordinary.workspace(()).unwrap(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        let mut actual = expected.clone();
        plan.execute_into(&input, &mut actual, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            1
        );
        assert_eq!(context.counts().nested().iterations(), preparation + n);
        plan.apply_into(&input, &mut actual, &mut workspace)
            .unwrap();
        assert_eq!(actual, expected);
        let mut adjoint = expected.clone();
        plan.apply_adjoint_into_scoped(
            &probe,
            &mut adjoint,
            &mut workspace,
            &mut OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
        .unwrap();
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            2
        );
        let lhs = numerics::dot(&actual, &probe);
        let rhs = numerics::dot(&input, &adjoint);
        let scale = numerics::norm(&actual) * numerics::norm(&probe)
            + numerics::norm(&input) * numerics::norm(&adjoint);
        numerics::assert_close(
            (lhs - rhs).norm(),
            64.0 * f64::from(f32::EPSILON) * scale,
            "f32 Hankel adjoint identity",
        );
    }
}

#[test]
fn f32_operator_stops_preserve_output_and_accounting() {
    use scioxide_core::{
        CancellationToken, ConvergenceReason, EvaluationRole, OperatorContext, SciOxideErrorKind,
        SciOxideErrorView,
    };
    let tolerance = 32.0 * f32::EPSILON;
    let spec = scioxide_fft::DiscreteHankelPlanSpec::new(
        0.0_f32,
        6.0,
        scioxide_fft::FftDirection::Forward,
    )
    .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap())
    .with_iteration_budget(Budget::ONE);
    let plan = DiscreteHankelPlan::new(&hankel_zeros::zeros(0.0_f32)[..9], spec).unwrap();
    let input = [Complex::new(1.0_f32, 0.25); 8];
    let initial = [Complex::new(123.0_f32, 456.0); 8];
    let mut workspace = plan.workspace(()).unwrap();
    for adjoint in [false, true] {
        for cancelled in [false, true] {
            let token = CancellationToken::new();
            if cancelled {
                token.cancel();
            }
            let mut state = SolveState::new();
            let mut budgets =
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential_native().with_cancellation(&token),
                    identity(),
                    &mut budgets,
                )
                .unwrap();
            let mut output = initial;
            let error = if adjoint {
                plan.apply_adjoint_into_scoped(
                    &input,
                    &mut output,
                    &mut workspace,
                    &mut OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require)
            } else {
                plan.apply_into_scoped(
                    &input,
                    &mut output,
                    &mut workspace,
                    &mut OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require)
            }
            .unwrap_err();
            assert_eq!(
                error.kind(),
                if cancelled {
                    SciOxideErrorKind::ExecutionStopped
                } else {
                    SciOxideErrorKind::ConvergenceFailure
                }
            );
            assert_eq!(output, initial);
            assert_eq!(context.counts().nested().iterations(), 0);
            assert_eq!(context.budget_report(BudgetKind::Iteration).consumed(), 0);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::LinearOperator),
                0
            );
            assert_eq!(
                context
                    .counts()
                    .nested()
                    .provider(EvaluationRole::LinearOperator),
                1
            );
            if !cancelled {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::ConvergenceFailure {
                        reason: ConvergenceReason::IterationLimit,
                        ..
                    }
                ));
            }
        }
    }
}

#[test]
fn preparation_accounts_for_all_nested_special_work() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let a = dht(&mut context).unwrap();
    assert_eq!(
        a.diagnostics().preparation_work,
        context.counts().nested().iterations()
    );
    let reference = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0)[..(8) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_limits(HankelLimits::default()),
    )
    .unwrap();
    assert_eq!(a.nodes(), reference.nodes());
    assert_eq!(a.frequencies(), reference.frequencies());
    assert_eq!(
        a.diagnostics().preparation_work,
        reference.diagnostics().preparation_work
    );
}

#[test]
fn general_order_root_isolation_obeys_the_exact_preparation_budget() {
    let spec = scioxide_fft::DiscreteHankelPlanSpec::new(
        2.25_f64,
        4.0,
        scioxide_fft::FftDirection::Inverse,
    )
    .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap());
    let zeros = hankel_zeros::zeros(2.25);
    let work = DiscreteHankelPlan::new(&zeros[..9], spec)
        .unwrap()
        .diagnostics()
        .preparation_work;
    let make = |units| {
        DiscreteHankelPlan::new(
            &zeros[..9],
            spec.with_limits(HankelLimits {
                preparation_work: Budget::try_from(units).unwrap(),
                ..HankelLimits::default()
            }),
        )
    };
    assert_eq!(make(work).unwrap().diagnostics().preparation_work, work);
    assert!(matches!(
        make(work - 1).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit,
            ..
        }
    ));
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
                DiscreteHankelPlan::build(
                    &hankel_zeros::zeros(0.0)[..9],
                    scioxide_fft::DiscreteHankelPlanSpec::new(
                        0.0,
                        6.0,
                        scioxide_fft::FftDirection::Forward,
                    )
                    .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
                    .with_limits(HankelLimits {
                        preparation_work: Budget::ONE,
                        ..HankelLimits::default()
                    }),
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
    let d = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..(8) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            0.0_f64,
            6.0,
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
        .with_limits(HankelLimits::default()),
    )
    .unwrap();
    let limited_d = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..9],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_iteration_budget(Budget::try_from(3).unwrap()),
    )
    .unwrap();
    let input = [Complex::new(1.0, 0.25); 8];
    let initial = [Complex::new(123.0, 456.0); 8];

    let mut scratch = { d.workspace(()).unwrap() };
    let mut output = initial;
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(3).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let result = {
        limited_d
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
        d.apply_into_scoped(
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
#[cfg(feature = "std")]
#[test]
fn dht_executor_rows_match_scalar_order_and_admission() {
    let workers = core::num::NonZeroUsize::new(3).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let p = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..(32) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            0.0_f64,
            8.0,
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
        .with_limits(HankelLimits::default()),
    )
    .unwrap();
    let input: Vec<_> = p
        .nodes()
        .iter()
        .map(|&r| Complex::new((-r * r).exp(), (r * 0.2).sin()))
        .collect();
    let mut expected = vec![Complex::new(0.0, 0.0); 32];
    let mut actual = expected.clone();
    let mut scratch = p.workspace(()).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    p.execute_into(&input, &mut expected, &mut scratch, &mut context)
        .unwrap();
    let counts = context.counts();
    let p = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64),
        scioxide_fft::DiscreteHankelPlanSpec::new(
            p.spec().order(),
            p.spec().extent(),
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
        .with_execution_policy(ExecutionPolicy::Parallel(workers)),
    )
    .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::new(&pool), identity(), &mut budgets)
        .unwrap();
    p.execute_into(&input, &mut actual, &mut scratch, &mut context)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(context.counts(), counts);
}

#[test]
fn scoped_hankel_adjoints_inherit_stops_and_account_for_nested_work() {
    use scioxide_core::{ClockBinding, Deadline, ExecutionStopReason, MonotonicInstant};
    let d = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..9],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap()),
    )
    .unwrap();
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
    for (op, iterations) in [(&d as &dyn AdjointOperator<Complex<f64>>, 8)] {
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
fn hankel_operator_errors_preserve_output_and_accounting() {
    let plan = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..9],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap()),
    )
    .unwrap();
    let input = [Complex::new(1.0, 0.0); 8];
    let initial = [Complex::new(99.0, 0.0); 8];
    for mode in [0, 1, 2, 3] {
        let mut output = initial;
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let result = match mode {
            0 => plan.apply_into(&input, &mut output, &mut []),
            1 => plan
                .apply_into_scoped(
                    &input,
                    &mut output,
                    &mut [],
                    &mut scioxide_core::OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require),
            2 => plan.apply_adjoint_into(&input, &mut output, &mut []),
            _ => plan
                .apply_adjoint_into_scoped(
                    &input,
                    &mut output,
                    &mut [],
                    &mut scioxide_core::OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require),
        };
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(output, initial);
        assert_eq!(context.counts().nested().iterations(), 0);
    }
}

#[cfg(feature = "std")]
#[test]
fn scoped_dht_adjoint_dispatch_is_deterministic_and_transactional() {
    use scioxide_core::Executor;
    let input = [Complex::new(1.0, 0.25); 32];
    let initial = [Complex::new(99.0, 0.0); 32];
    let mut expected = None;
    for count in [1, 2, 4] {
        let workers = core::num::NonZeroUsize::new(count).unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        let p = DiscreteHankelPlan::new(
            &hankel_zeros::zeros(0.0_f64)[..33],
            scioxide_fft::DiscreteHankelPlanSpec::new(
                0.0,
                8.0,
                scioxide_fft::FftDirection::Forward,
            )
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_execution_policy(ExecutionPolicy::Parallel(workers)),
        )
        .unwrap();
        let mut scratch = p.workspace(()).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
            .unwrap();
        let mut output = initial;
        p.apply_adjoint_into_scoped(
            &input,
            &mut output,
            &mut scratch,
            &mut scioxide_core::OperatorContext::new(&mut context),
        )
        .and_then(scioxide_core::Invocation::require)
        .unwrap();
        assert_eq!(context.counts().nested().iterations(), 32);
        let bits: Vec<_> = output
            .iter()
            .map(|z| (z.re.to_bits(), z.im.to_bits()))
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&bits, expected);
        } else {
            expected = Some(bits);
        }
    }
    for broken in BrokenExecutor::ALL {
        let p = DiscreteHankelPlan::new(
            &hankel_zeros::zeros(0.0_f64)[..33],
            scioxide_fft::DiscreteHankelPlanSpec::new(
                0.0,
                8.0,
                scioxide_fft::FftDirection::Forward,
            )
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_execution_policy(ExecutionPolicy::Parallel(broken.worker_budget())),
        )
        .unwrap();
        let mut scratch = p.workspace(()).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&broken), identity(), &mut budgets)
            .unwrap();
        let mut output = initial;
        assert!(
            p.apply_adjoint_into_scoped(
                &input,
                &mut output,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context)
            )
            .and_then(scioxide_core::Invocation::require)
            .is_err()
        );
        assert_eq!(output, initial);
    }
}

#[test]
fn insufficient_hankel_bulk_admission_keeps_all_counts_and_output_unchanged() {
    let plan = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..9],
        scioxide_fft::DiscreteHankelPlanSpec::new(0.0, 6.0, scioxide_fft::FftDirection::Forward)
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_iteration_budget(Budget::try_from(3).unwrap()),
    )
    .unwrap();
    let input = [Complex::new(1.0, 0.0); 8];
    let initial = [Complex::new(99.0, 0.0); 8];
    let mut scratch = plan.workspace(()).unwrap();
    for adjoint in [false, true] {
        let mut output = initial;
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let result = if adjoint {
            plan.apply_adjoint_into_scoped(
                &input,
                &mut output,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context),
            )
            .and_then(scioxide_core::Invocation::require)
        } else {
            plan.apply_into_scoped(
                &input,
                &mut output,
                &mut scratch,
                &mut scioxide_core::OperatorContext::new(&mut context),
            )
            .and_then(scioxide_core::Invocation::require)
        };

        assert!(
            matches!(result.unwrap_err().view(), scioxide_core::SciOxideErrorView::ConvergenceFailure {  reason: scioxide_core::ConvergenceReason::IterationLimit, diagnostics, .. }
            if diagnostics.counts().own().iterations() == 0 && diagnostics.budget().unwrap().consumed() == 0)
        );
        assert_eq!(output, initial);
        assert_eq!(context.counts().total().unwrap().iterations(), 0);
    }
}

#[test]
fn dht_triangular_work_matches_real_bessel_reports_and_exact_budget() {
    let n = 8;
    let zeros: Vec<_> = (1..=n + 1)
        .map(|j| j as f64 * std::f64::consts::PI)
        .collect();
    let cost = |order, x| {
        1 + scioxide_special::bessel_j_with_options(
            order,
            x,
            scioxide_special::BesselOptions::default(),
        )
        .unwrap()
        .counts()
        .iterations()
    };
    let verification: usize = zeros.iter().map(|&x| cost(0.5, x) + cost(1.5, x)).sum();
    let weights: usize = zeros[..n].iter().map(|&x| cost(1.5, x)).sum();
    let kernel: usize = zeros[..n]
        .iter()
        .enumerate()
        .map(|(i, a)| {
            zeros[..=i]
                .iter()
                .map(|b| cost(0.5, (a / zeros[n]) * b))
                .sum::<usize>()
        })
        .sum();
    let expected = verification + weights + kernel;
    let make = |units| {
        DiscreteHankelPlan::new(
            &zeros,
            scioxide_fft::DiscreteHankelPlanSpec::new(
                0.5,
                4.0,
                scioxide_fft::FftDirection::Forward,
            )
            .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap())
            .with_limits(HankelLimits {
                preparation_work: Budget::try_from(units).unwrap(),
                ..HankelLimits::default()
            }),
        )
    };
    assert_eq!(
        make(expected).unwrap().diagnostics().preparation_work,
        expected
    );
    assert!(make(expected - 1).is_err());
}

#[path = "support/operator_accounting.rs"]
mod operator_accounting;

#[test]
fn operator_evaluations_are_one_per_application_or_nonempty_axis() {
    hankel_operator_evaluations::<f64>();
}

#[test]
fn f32_operator_evaluations_are_one_per_application_or_nonempty_axis() {
    hankel_operator_evaluations::<f32>();
}

fn hankel_operator_evaluations<S: scioxide_core::RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::AxisWorkspaceRequest;
    let scalar = numerics::scalar::<S>;
    let tolerance = scalar(1e-12).max(S::EPSILON * scalar(32.0));
    let plan = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(S::ZERO)[..9],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            S::ZERO,
            scalar(6.0),
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap()),
    )
    .unwrap();
    for value in [S::ZERO, S::ONE] {
        let input = [Complex::new(value, S::ZERO); 8];
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
                NdView::new(&[] as &[S], layout).unwrap(),
                1,
                NdViewMut::new(&mut [], layout).unwrap(),
                &mut work,
                context,
            )
        });
    }
}

#[test]
fn hankel_child_epochs_and_local_limits_are_independent_of_parent_iterations() {
    use scioxide_core::{
        Budget, BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState, Tolerance,
    };
    use scioxide_fft::{DiscreteHankelPlan, DiscreteHankelPlanSpec, HankelLimits};
    let zeros: Vec<_> = (1..=6).map(|j| j as f64 * core::f64::consts::PI).collect();
    let spec = DiscreteHankelPlanSpec::new(0.5, 4.0, scioxide_fft::FftDirection::Forward)
        .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap());
    for (limits, quantity) in [
        (
            HankelLimits {
                preparation_bytes: 1,
                ..HankelLimits::default()
            },
            scioxide_core::quantity!("preparation_bytes"),
        ),
        (
            HankelLimits {
                workspace_bytes: 1,
                ..HankelLimits::default()
            },
            scioxide_core::quantity!("workspace_bytes"),
        ),
    ] {
        let error = DiscreteHankelPlan::new(&zeros, spec.with_limits(limits))
            .err()
            .unwrap();
        assert_eq!(
            match error.view() {
                scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } => Some(resource),
                _ => None,
            }
            .unwrap(),
            quantity
        );
    }
    let plan = DiscreteHankelPlan::new(&zeros, spec).unwrap();
    let mut work = plan.workspace(()).unwrap();
    let input = [Complex::new(1.0, 0.0); 5];
    let mut output = input;
    plan.execute_into(
        &input,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let _ = plan
        .execute_into(
            &input[..4],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap_err();

    let bounded = DiscreteHankelPlan::new(
        &zeros,
        spec.with_limits(HankelLimits {
            workspace_bytes: 10 * size_of::<Complex<f64>>(),
            ..HankelLimits::default()
        }),
    )
    .unwrap();
    let mut retained = Vec::with_capacity(11);
    retained.resize(10, Complex::new(0.0, 0.0));
    let mut retained = FftWorkspace::from_vec(retained);
    let inverse_error = bounded
        .execute_into(
            &input,
            &mut output,
            &mut retained,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(inverse_error.kind(), SciOxideErrorKind::ResourceLimit);

    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(7),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let accepted = parent.accepted_state();
    plan.execute_into(&input, &mut output, &mut work, &mut parent)
        .unwrap();
    assert_eq!(parent.counts().own().iterations(), 0);
    assert_eq!(parent.counts().nested().iterations(), 5);
    assert_eq!(parent.accepted_state(), accepted);
    let limited = DiscreteHankelPlan::new(&zeros, spec.with_iteration_budget(Budget::ONE)).unwrap();
    let before = output;
    let error = limited
        .execute_into(&input, &mut output, &mut work, &mut parent)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
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
    let diagnostics = _failure_diagnostics;
    assert_eq!(diagnostics.counts().own().iterations(), 0);
    assert_eq!(diagnostics.budget().unwrap().consumed(), 0);
    assert_eq!(output, before);
    assert_eq!(parent.counts().nested().iterations(), 5);
}
