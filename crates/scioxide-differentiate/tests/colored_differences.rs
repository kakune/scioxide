//! Independent sparse analytic, replay, layout and resource tests for colored differences.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use core::num::NonZeroUsize;

#[path = "support/adaptive.rs"]
#[allow(dead_code)]
mod prepared_cases;
use prepared_cases::{TestAdaptivePlan, TestOperation};
#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{
    Bounds, Budget, BudgetKind, CancellationToken, CapabilityError, Clock, ColoringKind,
    ColoringSchedule, Complex, Deadline, ErrorTolerance, EvaluationRole, ExecutionContext,
    ExecutionPlan, ExecutionPolicy, Executor, InvalidInputReason, MonotonicInstant, NdView,
    NdViewMut, NonFiniteOrigin, NormSelection, NumericScalar, PartitionFn, ProblemToken,
    RealScalar, SciOxideError, SciOxideErrorView, SequentialExecutor, SolveBudgets, SolveIdentity,
    SolveState, StridedLayout, StructuralPattern,
};
use scioxide_differentiate::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn test_executor() -> Box<dyn Executor> {
    #[cfg(feature = "std")]
    {
        Box::new(ThreadExecutor::new(NonZeroUsize::new(4).unwrap()))
    }
    #[cfg(not(feature = "std"))]
    {
        Box::new(SequentialExecutor)
    }
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(470),
        scioxide_core::algorithm_id!("test", "colored", 1),
        0,
        0,
    )
}
fn diagonal(n: usize) -> ColoringSchedule {
    ColoringSchedule::new(
        StructuralPattern::new(n, n, (0..=n).collect(), (0..n).collect()).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0; n],
    )
    .unwrap()
}
fn options<S: RealScalar>() -> AdaptiveDifference<S> {
    let tol = S::SQRT_EPSILON * S::try_from_f64_exact(16.0).unwrap();
    AdaptiveDifference::default()
        .with_tolerance(ErrorTolerance::scalar(tol, tol, NormSelection::Infinity).unwrap())
}
fn evidence<T: NumericScalar>() {
    type R<T> = <T as NumericScalar>::Real;
    let c = |v| R::<T>::try_from_f64_lossy(v).unwrap();
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(3, 4, vec![0, 2, 4, 6], vec![0, 1, 1, 2, 2, 3]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 1, 0, 1],
    )
    .unwrap();
    let x = [c(0.5), c(0.75), c(1.0), c(1.25)];
    let factor = T::from_components(c(1.0), c(0.5));
    let function = |x: &[R<T>], y: &mut [T]| {
        for row in 0..3 {
            y[row] = factor * T::from_real(x[row].powi(3) + x[row + 1].sin());
        }
        Ok(())
    };
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        let options = options().with_method(method);
        let plan = ColoredJacobianPlan::new(
            &schedule,
            (options.clone()).with_execution(scioxide_core::ExecutionPolicy::Auto),
        )
        .unwrap();
        let serial = plan.evaluate(function, &x, None).unwrap();
        assert!(serial.report().is_success(), "{:?}", serial.report());
        let values = serial.values().unwrap();
        for row in 0..3 {
            assert!(
                (values[2 * row] - factor * T::from_real(c(3.0) * x[row] * x[row])).abs()
                    < c(0.002)
            );
            assert!(
                (values[2 * row + 1] - factor * T::from_real(x[row + 1].cos())).abs() < c(0.002)
            );
        }
        let dense = TestAdaptivePlan::new(TestOperation::Jacobian, 4, 3, options).unwrap();
        let expected = dense.evaluate(&x, None, None, function).unwrap();
        assert!(expected.report().is_success());
        let dense_values = expected.values().unwrap();
        for (row, span) in schedule.pattern().row_ptr().windows(2).enumerate() {
            for (entry, &col) in schedule
                .pattern()
                .col_indices()
                .iter()
                .enumerate()
                .take(span[1])
                .skip(span[0])
            {
                assert!((values[entry] - dense_values[row * 4 + col]).abs() < c(0.002));
            }
        }
        let mut workspace = plan.workspace(()).unwrap();
        let mut output = vec![T::ZERO; 6];
        let mut errors = vec![c(0.0); 6];
        let pool = test_executor();
        for executor in [&SequentialExecutor as &dyn Executor, pool.as_ref()] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut ctx = state
                .begin_solve(ExecutionContext::new(executor), identity(), &mut budgets)
                .unwrap();
            let report = plan
                .evaluate_into(
                    scioxide_differentiate::Parallel(function),
                    &x,
                    None,
                    &mut output,
                    &mut errors,
                    &mut workspace,
                    &mut ctx,
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(output, values);
            assert_eq!(errors, serial.errors().unwrap());
            assert_eq!(report.evaluations(), serial.report().evaluations());
            assert_eq!(
                ctx.counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Function),
                report.evaluations()
            );
        }
    }
}
#[test]
fn real_f32() {
    evidence::<f32>();
}
#[test]
fn real_f64() {
    evidence::<f64>();
}
#[test]
fn complex_f32() {
    evidence::<Complex<f32>>();
}
#[test]
fn complex_f64() {
    evidence::<Complex<f64>>();
}

#[test]
fn opposite_boundaries_unequal_steps_and_vector_tolerances() {
    let schedule = diagonal(3);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        (options()
            .with_tolerance(
                ErrorTolerance::try_vector(&[1e-7, 1e-8, 1e-9], 1e-8, NormSelection::Infinity)
                    .unwrap(),
            )
            .with_step(DifferenceStep::absolute_coordinates(&[0.2, 0.7, 0.05]).unwrap()))
        .with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    let bounds = Bounds::new(vec![
        (Some(0.0), Some(4.0)),
        (Some(-4.0), Some(0.0)),
        (None, None),
    ])
    .unwrap();
    let mut w = plan.workspace(()).unwrap();
    let mut values = [99.0; 3];
    let mut errors = [99.0; 3];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = plan
        .evaluate_into(
            |x: &[f64], y: &mut [f64]| {
                assert!(x[0] >= 0.0 && x[1] <= 0.0);
                for (out, &v) in y.iter_mut().zip(x) {
                    *out = v.exp();
                }
                Ok(())
            },
            &[0.0, 0.0, 2.0],
            Some(&bounds),
            &mut values,
            &mut errors,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert!(report.is_success(), "{report:?}");
    for (&value, expected) in values.iter().zip([1.0, 1.0, 2.0_f64.exp()]) {
        assert!((value - expected).abs() < 1e-7);
    }
}

#[test]
fn shared_color_saves_evaluations_and_prepared_paths_execute() {
    let schedule = diagonal(12);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    let function = |x: &[f64], y: &mut [f64]| {
        for (y, &x) in y.iter_mut().zip(x) {
            *y = x * x;
        }
        Ok(())
    };
    let x = [1.0; 12];
    let serial = plan.evaluate(function, &x, None).unwrap();
    let dense = TestAdaptivePlan::new(TestOperation::Jacobian, 12, 12, options()).unwrap();
    let dense_result = dense.evaluate(&x, None, None, function).unwrap();
    let dense_report = dense_result.report();
    assert_eq!(
        dense_report.evaluations() - 1,
        12 * (serial.report().evaluations() - 1)
    );
    let mut w = plan.workspace(()).unwrap();
    let mut values = [0.0; 12];
    let mut errors = [0.0; 12];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    plan.evaluate_into(
        scioxide_differentiate::Parallel(function),
        &x,
        None,
        &mut values,
        &mut errors,
        &mut w,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(values, [2.0; 12]);
}

#[test]
fn budgets_failures_and_reuse_are_transactional() {
    let schedule = diagonal(2);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    let mut w = plan.workspace(()).unwrap();
    for limit in 1..15 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut child = SolveState::new();
        let calls = AtomicUsize::new(0);
        let mut values = [99.0; 2];
        let mut errors = [99.0; 2];
        let report = scioxide_core::SolveScope::from(&mut ctx)
            .run(&mut child, identity(), SolveBudgets::unbounded(), |ctx| {
                plan.evaluate_into(
                    scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        y[0] = x[0] * x[0];
                        y[1] = x[1] * x[1];
                        Ok(())
                    }),
                    &[1.0, 1.0],
                    None,
                    &mut values,
                    &mut errors,
                    &mut w,
                    ctx,
                )
            })
            .unwrap();
        assert_eq!(report.evaluations(), calls.load(Ordering::Relaxed));
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            report.evaluations()
        );
        assert!(report.evaluations() <= limit);
        if !report.output_available() {
            assert_eq!(values, [99.0; 2]);
            assert_eq!(errors, [99.0; 2]);
        }
    }
    for fail_at in 1..8 {
        let calls = AtomicUsize::new(0);
        let mut values = [99.0; 2];
        let mut errors = [99.0; 2];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let outcome = plan.evaluate_into(
            scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                let call = calls.fetch_add(1, Ordering::Relaxed) + 1;
                if call == fail_at {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Callback,
                        "fixture",
                    ));
                }
                y[0] = x[0] * x[0];
                y[1] = x[1] * x[1];
                Ok(())
            }),
            &[1.0, 1.0],
            None,
            &mut values,
            &mut errors,
            &mut w,
            &mut ctx,
        );
        assert!(
            matches!((outcome).as_ref().unwrap_err().view(), scioxide_core :: SciOxideErrorView :: DifferentiationFailure { frames, .. } if frames[0].evaluations == calls . load (Ordering :: Relaxed))
        );
        assert_eq!(values, [99.0; 2]);
        assert_eq!(errors, [99.0; 2]);
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            calls.load(Ordering::Relaxed)
        );
    }
    assert!(
        plan.evaluate(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0];
                y[1] = x[1];
                Ok(())
            },
            &[1.0, 1.0],
            None,
        )
        .unwrap()
        .report()
        .is_success()
    );
}

struct FailingExecutor {
    after: bool,
}
impl Executor for FailingExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        if self.after {
            SequentialExecutor.execute(ctx, plan, op)?;
        }
        Err(SciOxideError::execution_failure(
            None,
            scioxide_core::ExecutionFailureReason::ProtocolViolation,
            "provider failed",
        ))
    }
}
#[test]
fn executor_failure_counts_only_started_callbacks() {
    let plan = TestAdaptivePlan::new(TestOperation::First, 1, 1, options::<f64>()).unwrap();
    for after in [false, true] {
        let provider = FailingExecutor { after };
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(ExecutionContext::new(&provider), identity(), &mut budgets)
            .unwrap();
        let mut w = plan.workspace().unwrap();
        let mut values = [99.0];
        let mut errors = [99.0];
        let calls = AtomicUsize::new(0);
        assert!(
            plan.evaluate_parallel(
                &[1.0],
                None,
                None,
                |x, y| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    y[0] = x[0];
                    Ok(())
                },
                ExecutionPolicy::Parallel(NonZeroUsize::MIN),
                &mut values,
                &mut errors,
                &mut w,
                &mut ctx
            )
            .is_err()
        );
        assert_eq!(calls.load(Ordering::Relaxed), usize::from(after));
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            usize::from(after)
        );
        assert_eq!(values, [99.0]);
        assert_eq!(errors, [99.0]);
    }
}

fn assert_invalid_input(
    error: &SciOxideError,
    expected_argument: scioxide_core::Argument,
    expected_reason: InvalidInputReason,
) {
    let SciOxideErrorView::InvalidInput {
        argument, reason, ..
    } = error.view()
    else {
        unreachable!("expected InvalidInput: {error:?}");
    };
    assert_eq!(argument, expected_argument);
    assert_eq!(reason, expected_reason);
}

#[test]
fn invalid_schedules_shapes_empty_identities_and_callback_writes() {
    let p = StructuralPattern::new(1, 2, vec![0, 2], vec![0, 1]).unwrap();
    assert!(ColoringSchedule::new(p, ColoringKind::DerivativeColumns, vec![0, 0]).is_err());
    let p = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
    let wrong = ColoringSchedule::new(p, ColoringKind::DistanceOne, vec![0, 0]).unwrap();
    assert_invalid_input(
        &ColoredJacobianPlan::new(
            &wrong,
            options::<f64>().with_execution(ExecutionPolicy::Auto),
        )
        .unwrap_err(),
        scioxide_core::argument!("schedule"),
        InvalidInputReason::UnsupportedConfiguration,
    );
    let schedule = diagonal(2);
    let l2 = ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::L2).unwrap();
    for (method, reason) in [
        (
            options::<f64>().with_max_levels(2),
            InvalidInputReason::OutOfRange,
        ),
        (
            options().with_max_levels(65),
            InvalidInputReason::OutOfRange,
        ),
        (options().with_tolerance(l2), InvalidInputReason::OutOfRange),
        (
            options().with_zero_direction(ZeroDirectionPolicy::ValidateBase),
            InvalidInputReason::Inconsistent,
        ),
    ] {
        assert_invalid_input(
            &ColoredJacobianPlan::new(&schedule, method).unwrap_err(),
            scioxide_core::argument!("method"),
            reason,
        );
    }
    for (rows, cols) in [(0, 0), (3, 0), (0, 3), (3, 3)] {
        let schedule = ColoringSchedule::new(
            StructuralPattern::new(rows, cols, vec![0; rows + 1], vec![]).unwrap(),
            ColoringKind::DerivativeColumns,
            vec![0; cols],
        )
        .unwrap();
        let plan = ColoredJacobianPlan::new(
            &schedule,
            (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
        )
        .unwrap();
        let result = plan
            .evaluate::<f64, _>(
                |_: &[f64], _: &mut [f64]| unreachable!("empty pattern"),
                &vec![0.0; cols],
                None,
            )
            .unwrap();
        assert!(result.report().is_success());
        assert_eq!(result.report().evaluations(), 0);
    }
    let plan = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    assert!(
        plan.evaluate::<f64, _>(
            |_: &[f64], _: &mut [f64]| unreachable!("bad shape"),
            &[1.0],
            None,
        )
        .is_err()
    );
    assert!(
        plan.evaluate::<f64, _>(
            |_: &[f64], y: &mut [f64]| {
                y[0] = 0.0;
                Ok(())
            },
            &[1.0, 2.0],
            None,
        )
        .is_err()
    );
}

#[test]
fn packed_strided_inputs_and_scattered_outputs_preserve_logical_entries() {
    let schedule = diagonal(2);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        options::<f64>().with_execution(ExecutionPolicy::Auto),
    )
    .unwrap();
    let mut w = plan.workspace(()).unwrap();
    let input = [1.0, 99.0, 2.0];
    let layout = StridedLayout::new(&[2], &[2]).unwrap();
    let packed: Vec<_> = NdView::new(&input, layout)
        .unwrap()
        .iter()
        .copied()
        .collect();
    let mut values = [99.0; 3];
    let mut errors = [99.0; 3];
    let mut dense_values = [99.0; 2];
    let mut dense_errors = [99.0; 2];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = plan
        .evaluate_into(
            Parallel(|x: &[f64], y: &mut [f64]| {
                y[0] = x[0] * x[0];
                y[1] = x[1] * x[1];
                Ok(())
            }),
            &packed,
            None,
            &mut dense_values,
            &mut dense_errors,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert!(report.is_success());
    if report.output_available() {
        let mut v = dense_values.into_iter();
        NdViewMut::new(&mut values, layout)
            .unwrap()
            .for_each_mut(|x| *x = v.next().unwrap());
        let mut e = dense_errors.into_iter();
        NdViewMut::new(&mut errors, layout)
            .unwrap()
            .for_each_mut(|x| *x = e.next().unwrap());
    }
    assert_eq!(values, [2.0, 99.0, 4.0]);
    assert_eq!(errors[1], 99.0);
    let dense = TestAdaptivePlan::new(TestOperation::Jacobian, 2, 2, options()).unwrap();
    let mut w = dense.workspace().unwrap();
    let mut flat = [99.0; 4];
    let mut estimates = [99.0; 4];
    let report = dense
        .evaluate_parallel(
            &packed,
            None,
            None,
            |x, y| {
                y[0] = x[0] * x[0] + 3.0 * x[1];
                y[1] = x[1] * x[1];
                Ok(())
            },
            ExecutionPolicy::Auto,
            &mut flat,
            &mut estimates,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert!(report.is_success());
    let mut values = [99.0; 4];
    let layout_out = StridedLayout::new(&[2, 2], &[1, 2]).unwrap();
    if report.output_available() {
        let mut v = flat.into_iter();
        NdViewMut::new(&mut values, layout_out)
            .unwrap()
            .for_each_mut(|x| *x = v.next().unwrap());
    }
    for (&a, b) in values.iter().zip([2.0, 0.0, 3.0, 4.0]) {
        assert!((a - b).abs() < 1e-8);
    }
}

fn all_operations<T: NumericScalar>() {
    let c = |v| T::Real::try_from_f64_lossy(v).unwrap();
    let pool = test_executor();
    for operation in [
        TestOperation::First,
        TestOperation::Second,
        TestOperation::Gradient,
        TestOperation::Jacobian,
        TestOperation::Hessian,
        TestOperation::Jvp(ZeroDirectionPolicy::MathematicalZero),
    ] {
        let n = if matches!(operation, TestOperation::First | TestOperation::Second) {
            1
        } else {
            2
        };
        let m = if matches!(operation, TestOperation::Jacobian | TestOperation::Jvp(_)) {
            2
        } else {
            1
        };
        let plan = TestAdaptivePlan::new(operation, n, m, options()).unwrap();
        let x = [c(1.0), c(2.0)];
        let d = [c(1.0), c(-0.5)];
        let direction = matches!(operation, TestOperation::Jvp(_)).then_some(&d[..]);
        let function = |x: &[T::Real], y: &mut [T]| {
            let sum = x.iter().fold(c(0.0), |s, &x| s + x.powi(3));
            for (i, y) in y.iter_mut().enumerate() {
                *y = T::from_components(sum, c(0.25) * sum) * T::from_real(c((i + 1) as f64));
            }
            Ok(())
        };
        let expected = plan.evaluate(&x[..n], None, direction, function).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let mut values = vec![T::ZERO; plan.output_len()];
        let mut errors = vec![c(0.0); plan.output_len()];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(
                ExecutionContext::new(pool.as_ref()),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let report = plan
            .evaluate_parallel(
                &x[..n],
                None,
                direction,
                function,
                ExecutionPolicy::Auto,
                &mut values,
                &mut errors,
                &mut workspace,
                &mut ctx,
            )
            .unwrap();
        assert!(report.is_success(), "{operation:?} {report:?}");
        assert_eq!(values, expected.values().unwrap());
        assert_eq!(errors, expected.errors().unwrap());
        assert_eq!(report.evaluations(), expected.report().evaluations());
        assert_eq!(report.iterations(), expected.report().iterations());
    }
}
#[test]
fn executor_operations_real32() {
    all_operations::<f32>();
}
#[test]
fn executor_operations_real64() {
    all_operations::<f64>();
}
#[test]
fn executor_operations_complex32() {
    all_operations::<Complex<f32>>();
}
#[test]
fn executor_operations_complex64() {
    all_operations::<Complex<f64>>();
}

#[test]
fn cancellation_and_iteration_limits_leave_incomplete_outputs_unchanged() {
    let schedule = diagonal(2);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    let token = CancellationToken::new();
    let calls = AtomicUsize::new(0);
    let mut w = plan.workspace(()).unwrap();
    let mut values = [99.0; 2];
    let mut errors = [99.0; 2];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let execution = ExecutionContext::sequential().with_cancellation(&token);
    let mut ctx = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    let report = plan
        .evaluate_into(
            scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                if calls.fetch_add(1, Ordering::Relaxed) == 1 {
                    token.cancel();
                }
                y[0] = x[0];
                y[1] = x[1];
                Ok(())
            }),
            &[1.0, 2.0],
            None,
            &mut values,
            &mut errors,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert_eq!(report.evaluations(), 2);
    assert_eq!(values, [99.0; 2]);
    assert_eq!(errors, [99.0; 2]);
    token.reset();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        options::<f64>()
            .with_execution(ExecutionPolicy::Auto)
            .with_budgets(budgets),
    )
    .unwrap();
    let mut ctx = state
        .begin_solve(execution, identity(), &mut budgets)
        .unwrap();
    let report = plan
        .evaluate_into(
            |x: &[f64], y: &mut [f64]| {
                y.copy_from_slice(x);
                Ok(())
            },
            &[1.0, 2.0],
            None,
            &mut values,
            &mut errors,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::IterationLimit
    );
    assert_eq!(report.iterations(), 1);
    assert!(!report.output_available());
    assert_eq!(values, [99.0; 2]);
}

struct ReorderedExecutor;
impl Executor for ReorderedExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, &|partition, (), scope| {
            let reversed = plan
                .partitions()
                .nth(plan.worker_count() - 1 - partition.worker_index())
                .unwrap();
            operation(reversed, (), scope)
        })
    }
}
#[test]
fn reversed_dispatch_replays_values_errors_counts_and_budget_prefix() {
    let schedule = diagonal(2);
    let p = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    for limit in [1, 2, 3, 6, 50] {
        let mut observations = Vec::new();
        for executor in [&SequentialExecutor as &dyn Executor, &ReorderedExecutor] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let mut ctx = state
                .begin_solve(ExecutionContext::new(executor), identity(), &mut budgets)
                .unwrap();
            let mut w = p.workspace(()).unwrap();
            let mut y = [99.0; 2];
            let mut errors = [99.0; 2];
            let report = p
                .evaluate_into(
                    scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                        y[0] = x[0].exp();
                        y[1] = x[1].sin();
                        Ok(())
                    }),
                    &[1.0, 2.0],
                    None,
                    &mut y,
                    &mut errors,
                    &mut w,
                    &mut ctx,
                )
                .unwrap();
            observations.push((
                y,
                errors,
                report.termination(),
                report.evaluations(),
                report.iterations(),
                report.completed_entries(),
            ));
        }
        assert_eq!(observations[0], observations[1]);
    }
}
struct BrokenDispatch {
    calls: AtomicUsize,
    duplicate: bool,
}
impl Executor for BrokenDispatch {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let dispatch = self.calls.fetch_add(1, Ordering::Relaxed);
        SequentialExecutor.execute(ctx, plan, &|partition, (), scope| {
            if dispatch > 0 && partition.worker_index() == 1 {
                return Err(SciOxideError::execution_failure(
                    None,
                    scioxide_core::ExecutionFailureReason::ProtocolViolation,
                    "partial dispatch",
                ));
            }
            if self.duplicate {
                op(partition.clone(), (), scope)?;
            }
            op(partition, (), scope)
        })
    }
}
#[test]
fn duplicate_and_partial_executor_failure_account_exactly() {
    let p = TestAdaptivePlan::new(TestOperation::First, 1, 1, options::<f64>()).unwrap();
    for duplicate in [false, true] {
        let executor = BrokenDispatch {
            calls: AtomicUsize::new(0),
            duplicate,
        };
        let calls = AtomicUsize::new(0);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
            .unwrap();
        let mut w = p.workspace().unwrap();
        let mut y = [99.0];
        let mut errors = [99.0];
        let result = p.evaluate_parallel(
            &[1.0],
            None,
            None,
            |x, y| {
                calls.fetch_add(1, Ordering::Relaxed);
                y[0] = x[0] * x[0];
                Ok(())
            },
            ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()),
            &mut y,
            &mut errors,
            &mut w,
            &mut ctx,
        );
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ExecutionFailure { .. }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), if duplicate { 1 } else { 2 });
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            calls.load(Ordering::Relaxed)
        );
        assert_eq!(y, [99.0]);
        assert_eq!(errors, [99.0]);
    }
}
#[test]
fn empty_validate_base_jvp_executor_is_callback_free() {
    let p = TestAdaptivePlan::new(
        TestOperation::Jvp(ZeroDirectionPolicy::ValidateBase),
        0,
        0,
        options::<f64>(),
    )
    .unwrap();
    let mut w = p.workspace::<f64>().unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = p
        .evaluate_parallel(
            &[],
            None,
            Some(&[]),
            |_, _| unreachable!("empty identity must not invoke callback"),
            ExecutionPolicy::Auto,
            &mut [],
            &mut [],
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(report.evaluations(), 0);
    assert!(report.is_success());
}

struct TestClock {
    fail: bool,
}
impl Clock for TestClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        if self.fail {
            Err(CapabilityError::new("fixture clock failure"))
        } else {
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(2),
            ))
        }
    }
}
#[test]
fn deadline_and_clock_failure_precede_callbacks() {
    let schedule = diagonal(1);
    let p = ColoredJacobianPlan::new(
        &schedule,
        (options::<f64>()).with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    for fail in [false, true] {
        let clock = TestClock { fail };
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let deadline = Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut ctx = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let mut w = p.workspace::<f64>(()).unwrap();
        let mut y = [99.0];
        let mut errors = [99.0];
        let result = p.evaluate_into(
            scioxide_differentiate::Parallel(|_: &[f64], _: &mut [f64]| {
                unreachable!("expired deadline")
            }),
            &[1.0],
            None,
            &mut y,
            &mut errors,
            &mut w,
            &mut ctx,
        );
        if fail {
            assert!(result.is_err());
        } else {
            assert_eq!(
                result.unwrap().termination(),
                DifferentiationTermination::DeadlineReached
            );
        }
        assert_eq!(
            ctx.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            0
        );
        assert_eq!(y, [99.0]);
        assert_eq!(errors, [99.0]);
    }
}
