//! Independent analytic and resource-contract tests for adaptive differences.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/adaptive.rs"]
#[allow(dead_code)]
mod prepared_cases;
use prepared_cases::{TestAdaptivePlan, TestOperation};
use scioxide_core::{
    Bounds, Budget, BudgetKind, CancellationToken, CapabilityError, Clock, Complex, Deadline,
    DomainReason, ErrorTolerance, EvaluationRole, ExecutionContext, InvalidInputReason,
    MonotonicInstant, NormSelection, NumericScalar, ProblemToken, RealScalar, SciOxideError,
    SciOxideErrorView, SequentialExecutor, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::*;

fn plan<S: RealScalar>(
    operation: TestOperation,
    inputs: usize,
    outputs: usize,
    method: AdaptiveDifferenceMethod,
    tolerance: S,
) -> TestAdaptivePlan<S> {
    TestAdaptivePlan::new(
        operation,
        inputs,
        outputs,
        AdaptiveDifference::default()
            .with_method(method)
            .with_tolerance(
                ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap(),
            ),
    )
    .unwrap()
}

fn real_evidence<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let tolerance = S::SQRT_EPSILON * c(8.0);
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        for (operation, expected) in [
            (TestOperation::First, c(4.0)),
            (TestOperation::Second, c(12.0)),
        ] {
            let p = plan(operation, 1, 1, method, tolerance);
            let result = p
                .evaluate(&[c(1.0)], None, None, |x, y| {
                    y[0] = x[0].powi(4);
                    Ok(())
                })
                .unwrap();
            assert!(
                result.report().is_success(),
                "{method:?} {operation:?}: {:?}",
                result.report()
            );
            assert!((result.values().unwrap()[0] - expected).abs() <= tolerance * c(16.0));
        }
        let p = plan(TestOperation::Hessian, 2, 1, method, tolerance);
        let result = p
            .evaluate(&[c(0.5), c(2.0)], None, None, |x, y| {
                y[0] = x[0] * x[0] * x[1] + x[1].powi(3);
                Ok(())
            })
            .unwrap();
        assert!(result.report().is_success(), "{:?}", result.report());
        let actual = result.values().unwrap();
        for (actual, expected) in actual.iter().zip([4.0, 1.0, 1.0, 12.0]) {
            assert!((*actual - c(expected)).abs() <= tolerance * c(16.0));
        }
        assert_eq!(actual[1], actual[2]);
        let p = plan(TestOperation::Gradient, 2, 1, method, tolerance);
        let result = p
            .evaluate(&[c(0.25), c(2.0)], None, None, |x, y| {
                y[0] = x[0].sin() + x[1].exp();
                Ok(())
            })
            .unwrap();
        assert!(
            result.report().is_success(),
            "{method:?}: {:?}",
            result.report()
        );
        for (actual, expected) in result
            .values()
            .unwrap()
            .iter()
            .zip([c(0.25).cos(), c(2.0).exp()])
        {
            assert!((*actual - expected).abs() <= tolerance * c(16.0));
        }
    }
}

#[test]
fn analytic_real_f32() {
    real_evidence::<f32>();
}
#[test]
fn analytic_real_f64() {
    real_evidence::<f64>();
}

fn complex_evidence<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        let tolerance = S::SQRT_EPSILON * c(8.0);
        let p = plan(TestOperation::Jacobian, 2, 2, method, tolerance);
        let callback = |x: &[S], y: &mut [Complex<S>]| {
            y[0] = Complex::new(x[0] * x[0], x[1] * x[1]);
            y[1] = Complex::new(x[0] * x[1], x[0] + x[1]);
            Ok(())
        };
        let result = p.evaluate(&[c(2.0), c(3.0)], None, None, callback).unwrap();
        assert!(result.report().is_success(), "{:?}", result.report());
        for (actual, (re, im)) in
            result
                .values()
                .unwrap()
                .iter()
                .zip([(4.0, 0.0), (0.0, 6.0), (3.0, 1.0), (2.0, 1.0)])
        {
            assert!((actual.real_component() - c(re)).abs() <= tolerance * c(8.0));
            assert!((actual.imaginary_component() - c(im)).abs() <= tolerance * c(8.0));
        }
        let p = plan(
            TestOperation::Jvp(ZeroDirectionPolicy::MathematicalZero),
            2,
            2,
            method,
            tolerance,
        );
        let result = p
            .evaluate(&[c(2.0), c(3.0)], None, Some(&[c(1.0), c(-2.0)]), callback)
            .unwrap();
        assert!(result.report().is_success(), "{:?}", result.report());
        for (actual, (re, im)) in result
            .values()
            .unwrap()
            .iter()
            .zip([(4.0, -12.0), (-1.0, -1.0)])
        {
            assert!((actual.real_component() - c(re)).abs() <= tolerance * c(16.0));
            assert!((actual.imaginary_component() - c(im)).abs() <= tolerance * c(16.0));
        }
        for (operation, inputs, expected) in [
            (TestOperation::First, 1, &[(2.0, 3.0)][..]),
            (TestOperation::Second, 1, &[(2.0, 6.0)][..]),
            (TestOperation::Gradient, 2, &[(2.0, 3.0), (4.0, 12.0)][..]),
            (
                TestOperation::Hessian,
                2,
                &[(2.0, 6.0), (0.0, 0.0), (0.0, 0.0), (2.0, 12.0)][..],
            ),
        ] {
            let p = plan(operation, inputs, 1, method, tolerance);
            let result = p
                .evaluate(&[c(1.0), c(2.0)][..inputs], None, None, |x, y| {
                    y[0] = Complex::new(
                        x.iter().copied().map(|v| v * v).sum(),
                        x.iter().copied().map(|v| v.powi(3)).sum(),
                    );
                    Ok(())
                })
                .unwrap();
            assert!(
                result.report().is_success(),
                "{operation:?}: {:?}",
                result.report()
            );
            for (actual, &(re, im)) in result.values().unwrap().iter().zip(expected) {
                assert!((actual.real_component() - c(re)).abs() <= tolerance * c(16.0));
                assert!((actual.imaginary_component() - c(im)).abs() <= tolerance * c(16.0));
            }
        }
    }
}
#[test]
fn complex_f32() {
    complex_evidence::<f32>();
}
#[test]
fn complex_f64() {
    complex_evidence::<f64>();
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(468),
        scioxide_core::algorithm_id!("test", "adaptive", 1),
        0,
        0,
    )
}

#[test]
fn boundary_stencils_remain_in_domain() {
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        for x in [0.0, 1.0, 1e-3, 1.0 - 1e-3] {
            for operation in [TestOperation::First, TestOperation::Second] {
                let p = plan(operation, 1, 1, method, 1e-7);
                let result = p
                    .evaluate(&[x], Some(&bounds), None, |x, y| {
                        assert!((0.0..=1.0).contains(&x[0]));
                        y[0] = x[0].exp();
                        Ok(())
                    })
                    .unwrap();
                assert!(
                    result.report().is_success(),
                    "{method:?} {operation:?} x={x}: {:?}",
                    result.report()
                );
                assert!((result.values().unwrap()[0] - x.exp()).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn shared_limits_count_every_attempt_and_keep_output_transactional() {
    let p = plan(
        TestOperation::First,
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-10,
    );
    let mut w = p.workspace::<f64>().unwrap();
    for limit in 1..12 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut output = [99.0];
        let mut errors = [77.0];
        let mut calls = 0;
        let report = p
            .evaluate_into(
                &[1.0],
                None,
                None,
                |x, y| {
                    calls += 1;
                    y[0] = x[0].exp();
                    Ok(())
                },
                &mut output,
                &mut errors,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.evaluations(), calls);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            calls
        );
        assert!(calls <= limit);
        if !report.output_available() {
            assert_eq!(output, [99.0]);
            assert_eq!(errors, [77.0]);
        }
        if calls == limit && limit < 5 {
            assert!(!report.is_success());
        }
    }
}

#[test]
fn failure_provenance_and_workspace_reuse() {
    let p = plan(
        TestOperation::Hessian,
        2,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    let mut w = p.workspace::<f64>().unwrap();
    for fail_at in 1..20 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut values = [99.0; 4];
        let mut errors = [77.0; 4];
        let mut calls = 0;
        let result = p.evaluate_into(
            &[1.0, 2.0],
            None,
            None,
            |x, y| {
                calls += 1;
                if calls == fail_at {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("x"),
                        DomainReason::OutsideDomain,
                        "injected",
                    ));
                }
                y[0] = x[0].exp() * x[1];
                Ok(())
            },
            &mut values,
            &mut errors,
            &mut w,
            &mut context,
        );
        assert!(
            matches!((result).as_ref().unwrap_err().view(), scioxide_core :: SciOxideErrorView :: DifferentiationFailure { frames, .. } if frames[0].evaluations == fail_at)
        );
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            fail_at
        );
        assert_eq!(values, [99.0; 4]);
        assert_eq!(errors, [77.0; 4]);
        let report = p
            .evaluate_into(
                &[1.0, 2.0],
                None,
                None,
                |x, y| {
                    y[0] = x[0] * x[1];
                    Ok(())
                },
                &mut values,
                &mut errors,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert!(report.is_success(), "{report:?}");
    }
}

#[test]
fn cancelling_and_noisy_callbacks_never_claim_precise_convergence() {
    let p = plan(
        TestOperation::First,
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-12,
    );
    for noise in [false, true] {
        let mut calls = 0;
        let result = p
            .evaluate(&[1.0], None, None, |x, y| {
                calls += 1;
                y[0] = if noise {
                    x[0].sin() + if calls % 2 == 0 { 1e-3 } else { -1e-3 }
                } else {
                    1e20 + x[0]
                };
                Ok(())
            })
            .unwrap();
        assert!(!result.report().is_success(), "{:?}", result.report());
    }
}

#[test]
fn unequal_scales_and_explicit_steps() {
    let p = TestAdaptivePlan::new(
        TestOperation::Gradient,
        2,
        1,
        AdaptiveDifference::default()
            .with_tolerance(ErrorTolerance::scalar(1e-7, 1e-7, NormSelection::Infinity).unwrap())
            .with_step(DifferenceStep::absolute_coordinates(&[1e-11, 1e9]).unwrap()),
    )
    .unwrap();
    let mut w = p.workspace::<f64>().unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut values = [0.0; 2];
    let mut errors = [0.0; 2];
    let report = p
        .evaluate_into(
            &[1e-10, 1e10],
            None,
            None,
            |x, y| {
                y[0] = (x[0] * 1e10).sin() + (x[1] * 1e-10).sin();
                Ok(())
            },
            &mut values,
            &mut errors,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert!(report.is_success(), "{report:?}");
    assert!((values[0] / 1e10 - 1.0_f64.cos()).abs() < 1e-7);
    assert!((values[1] / 1e-10 - 1.0_f64.cos()).abs() < 1e-5);
}

#[test]
fn empty_zero_direction_and_fixed_coordinates() {
    for op in [
        TestOperation::Gradient,
        TestOperation::Hessian,
        TestOperation::Jacobian,
    ] {
        let result = plan(op, 0, 1, AdaptiveDifferenceMethod::Ridders, 1e-8)
            .evaluate::<f64, _>(&[], None, None, |_, _| unreachable!())
            .unwrap();
        assert!(result.report().is_success());
        assert_eq!(result.report().evaluations(), 0);
    }
    let p = plan(
        TestOperation::Jvp(ZeroDirectionPolicy::MathematicalZero),
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    let result = p
        .evaluate::<f64, _>(&[1.0], None, Some(&[0.0]), |_, _| unreachable!())
        .unwrap();
    assert_eq!(result.values().unwrap(), [0.0]);
    assert_eq!(result.report().evaluations(), 0);
    let p = plan(
        TestOperation::Jvp(ZeroDirectionPolicy::ValidateBase),
        0,
        0,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    let mut calls = 0;
    let result = p
        .evaluate::<f64, _>(&[], None, Some(&[]), |_, output| {
            calls += 1;
            assert!(output.is_empty());
            Ok(())
        })
        .unwrap();
    assert!(result.report().is_success());
    // An empty derivative output is an exact, callback-free identity,
    // including ValidateBase. Nonempty ValidateBase products still validate once.
    assert_eq!(calls, 0);
    assert_eq!(result.report().evaluations(), 0);
    let bounds = Bounds::new(vec![(Some(1.0), Some(1.0))]).unwrap();
    let p = plan(
        TestOperation::First,
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    let result = p
        .evaluate(&[1.0], Some(&bounds), None, |_, y| {
            y[0] = 1.0;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::StepSizeLimit
    );
    assert!(result.values().is_none());
}

fn exponent_evidence<S: RealScalar + NumericScalar<Real = S>>(exponents: &[i32]) {
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    for &exponent in exponents {
        let scale = c(1.0).scalbn(exponent);
        for operation in [TestOperation::First, TestOperation::Second] {
            let order = if operation == TestOperation::First {
                1
            } else {
                2
            };
            let expected = c(if order == 1 { 3.0 } else { 6.0 }).scalbn(-order * exponent);
            let tolerance = ErrorTolerance::scalar(
                S::SQRT_EPSILON * expected.abs(),
                S::SQRT_EPSILON,
                NormSelection::Infinity,
            )
            .unwrap();
            let p = TestAdaptivePlan::new(
                operation,
                1,
                1,
                AdaptiveDifference::default()
                    .with_tolerance(tolerance)
                    .with_step(DifferenceStep::absolute(scale * c(0.5)).unwrap()),
            )
            .unwrap();
            let mut w = p.workspace().unwrap();
            let mut values = [c(0.0)];
            let mut errors = [c(0.0)];
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            let report = p
                .evaluate_into(
                    &[scale],
                    None,
                    None,
                    |x, y| {
                        y[0] = (x[0] / scale).powi(3);
                        Ok(())
                    },
                    &mut values,
                    &mut errors,
                    &mut w,
                    &mut context,
                )
                .unwrap();
            assert!(
                report.is_success(),
                "exponent={exponent}, order={order}: {report:?}, error={errors:?}"
            );
            assert!((values[0] / expected - c(1.0)).abs() < S::SQRT_EPSILON * c(16.0));
        }
    }
}

#[test]
fn exponent_scales_f32() {
    exponent_evidence::<f32>(&[-60, -20, 0, 20, 60]);
}
#[test]
fn exponent_scales_f64() {
    exponent_evidence::<f64>(&[-500, -250, 0, 250, 500]);
}

#[test]
fn jvp_direction_scaling_is_not_a_zero_shortcut() {
    for exponent in [-1000, -400, 0, 400, 1000] {
        let direction = 1.0_f64.scalbn(exponent);
        let p = plan(
            TestOperation::Jvp(ZeroDirectionPolicy::MathematicalZero),
            1,
            1,
            AdaptiveDifferenceMethod::Ridders,
            1e-8,
        );
        let result = p
            .evaluate(&[1.0], None, Some(&[direction]), |x, y| {
                y[0] = x[0];
                Ok(())
            })
            .unwrap();
        assert!(
            result.report().is_success(),
            "{exponent}: {:?}",
            result.report()
        );
        assert!(result.report().evaluations() > 0);
        assert!((result.values().unwrap()[0] / direction - 1.0).abs() < 1e-10);
    }
}

#[test]
fn prepared_operations_execute_and_share_parent_budgets() {
    for operation in [
        TestOperation::First,
        TestOperation::Second,
        TestOperation::Gradient,
        TestOperation::Jacobian,
        TestOperation::Hessian,
        TestOperation::Jvp(ZeroDirectionPolicy::MathematicalZero),
    ] {
        let p = plan(operation, 1, 1, AdaptiveDifferenceMethod::Ridders, 1e-8);
        let mut w = p.workspace().unwrap();
        let mut values = [0.0];
        let mut errors = [0.0];
        let mut parent_state = SolveState::new();
        let mut child_state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(7).unwrap());
        let mut parent = parent_state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(
            parent
                .record_provider(EvaluationRole::Function)
                .unwrap()
                .is_granted()
        );
        let report = scioxide_core::SolveScope::from(&mut parent)
            .run(
                &mut child_state,
                identity(),
                SolveBudgets::unbounded(),
                |child| {
                    p.evaluate_into(
                        &[1.0],
                        None,
                        matches!(operation, TestOperation::Jvp(_)).then_some(&[1.0][..]),
                        |x, y| {
                            y[0] = x[0].exp();
                            Ok(())
                        },
                        &mut values,
                        &mut errors,
                        &mut w,
                        child,
                    )
                },
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            DifferentiationTermination::EvaluationLimit
        );
        assert_eq!(report.evaluations(), 6);
        assert_eq!(
            parent
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            7
        );
    }
}

struct FixedClock;
impl Clock for FixedClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(2),
        ))
    }
}

#[test]
fn cancellation_deadlines_iteration_and_refinement_limits_are_distinct() {
    let p = plan(
        TestOperation::First,
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-12,
    );
    let mut w = p.workspace::<f64>().unwrap();
    let mut values = [99.0];
    let mut errors = [77.0];
    let token = CancellationToken::new();
    token.cancel();
    let clock = FixedClock;
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &clock_binding,
        MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
    );
    // Explicitly injected executor contexts use the same sequential refinement contract.
    let executor = SequentialExecutor;
    let execution = ExecutionContext::new(&executor)
        .with_cancellation(&token)
        .with_deadline(deadline)
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    for reason in [
        DifferentiationTermination::Cancelled,
        DifferentiationTermination::DeadlineReached,
    ] {
        let mut context = state
            .begin_solve(execution, identity(), &mut budgets)
            .unwrap();
        let report = p
            .evaluate_into(
                &[1.0],
                None,
                None,
                |_, _| unreachable!(),
                &mut values,
                &mut errors,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.termination(), reason);
        assert_eq!(report.evaluations(), 0);
        assert_eq!(values, [99.0]);
        token.reset();
    }
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let p = p.with_budgets(budgets).unwrap();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = p
        .evaluate_into(
            &[1.0],
            None,
            None,
            |x, y| {
                y[0] = x[0].exp();
                Ok(())
            },
            &mut values,
            &mut errors,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::IterationLimit
    );
    assert_eq!(report.iterations(), 1);
    assert_eq!(values, [99.0]);
    let p = TestAdaptivePlan::new(
        TestOperation::First,
        1,
        1,
        AdaptiveDifference::default().with_max_levels(3),
    )
    .unwrap();
    let result = p
        .evaluate(&[1.0], None, None, |x, y| {
            y[0] = (x[0] * 5.0).exp();
            Ok(())
        })
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::RefinementLimit
    );
}

#[test]
fn missing_callback_components_and_bad_configuration_fail_before_publication() {
    let p = plan(
        TestOperation::Jacobian,
        2,
        2,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    assert!(matches!(
        (p.evaluate(&[1.0, 2.0], None, None, |_, y: &mut [f64]| {
            y[0] = 0.0;
            Ok(())
        }))
        .as_ref()
        .unwrap_err()
        .view(),
        scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, .. } if frames[0].evaluations == 1
    ));
    let steps = DifferenceStep::absolute_coordinates(&[0.1, 0.1]).unwrap();
    assert!(
        AdaptiveDifference::default()
            .with_step(steps)
            .jacobian::<f64, _>(|_: &[f64], _: &mut [f64]| unreachable!(), &[1.0], 2, None)
            .is_err()
    );
    assert!(
        p.evaluate::<f64, _>(&[1.0, 2.0], None, Some(&[1.0, 1.0]), |_, _| unreachable!())
            .is_err()
    );
    assert!(DifferenceStep::<f32>::relative(f32::NAN).is_err());
    assert_rejected_settings(
        AdaptiveDifference::<f32>::default()
            .with_max_levels(2)
            .derivative::<f32, _>(|_| unreachable!(), 1.0),
        InvalidInputReason::OutOfRange,
    );
}

fn assert_rejected_settings<T>(result: Result<T, SciOxideError>, expected: InvalidInputReason) {
    let Err(error) = result else {
        unreachable!("rejected settings must fail the call");
    };
    let SciOxideErrorView::InvalidInput {
        argument, reason, ..
    } = error.view()
    else {
        unreachable!("rejected settings must be InvalidInput: {error:?}");
    };
    assert_eq!(argument, scioxide_core::argument!("self"));
    assert_eq!(reason, expected);
}

fn rejected_settings_fail_each_call<S: RealScalar + NumericScalar<Real = S>>() {
    let x = [S::ONE; 2];
    let direction = [S::ONE; 2];
    let never = |_: &[S], _: &mut [S]| -> Result<(), SciOxideError> { unreachable!("rejected") };
    let never_scalar = |_: S| -> Result<S, SciOxideError> { unreachable!("rejected") };
    let l2 = ErrorTolerance::scalar(S::SQRT_EPSILON, S::SQRT_EPSILON, NormSelection::L2).unwrap();
    let sentinel = S::try_from_f64_lossy(99.0).unwrap();
    let valid = AdaptiveDifference::default();
    let workspace = |shape| {
        AdaptiveDifferenceWorkspace::<S>::workspace(DerivativeRequest::new(&valid, shape)).unwrap()
    };
    let validate_base =
        AdaptiveDifference::default().with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    for (method, reason, rejects_jvp) in [
        (
            AdaptiveDifference::default().with_max_levels(2),
            InvalidInputReason::OutOfRange,
            true,
        ),
        (
            AdaptiveDifference::default().with_max_levels(65),
            InvalidInputReason::OutOfRange,
            true,
        ),
        (
            AdaptiveDifference::default().with_tolerance(l2),
            InvalidInputReason::OutOfRange,
            true,
        ),
        (validate_base, InvalidInputReason::Inconsistent, false),
    ] {
        let mut out = [sentinel; 4];
        let mut errors = [sentinel; 4];
        let cx = ExecutionContext::sequential;
        assert_rejected_settings(method.derivative(never_scalar, S::ONE), reason);
        assert_rejected_settings(method.second_derivative(never_scalar, S::ONE), reason);
        assert_rejected_settings(method.gradient(never, &x, None), reason);
        assert_rejected_settings(method.hessian(never, &x, None), reason);
        assert_rejected_settings(method.jacobian(never, &x, 2, None), reason);
        let (out2, errors2) = (&mut out[..2], &mut errors[..2]);
        let mut w = workspace(DerivativeShape::gradient(2));
        assert_rejected_settings(
            method.gradient_into(never, &x, None, out2, errors2, &mut w, cx()),
            reason,
        );
        let mut w = workspace(DerivativeShape::hessian(2));
        assert_rejected_settings(
            method.hessian_into(never, &x, None, &mut out, &mut errors, &mut w, cx()),
            reason,
        );
        let mut w = workspace(DerivativeShape::jacobian(2, 2));
        assert_rejected_settings(
            method.jacobian_into(never, &x, 2, None, &mut out, &mut errors, &mut w, cx()),
            reason,
        );
        if rejects_jvp {
            assert_rejected_settings(method.jvp(never, &x, &direction, 2, None), reason);
            let (out2, errors2) = (&mut out[..2], &mut errors[..2]);
            let mut w = workspace(DerivativeShape::jvp(2, 2));
            assert_rejected_settings(
                method.jvp_into(never, &x, &direction, None, out2, errors2, &mut w, cx()),
                reason,
            );
        }
        assert!(out.iter().chain(&errors).all(|&v| v == sentinel));
    }
}
#[test]
fn rejected_settings_fail_each_call_f32() {
    rejected_settings_fail_each_call::<f32>();
}
#[test]
fn rejected_settings_fail_each_call_f64() {
    rejected_settings_fail_each_call::<f64>();
}

#[test]
fn cancellation_on_the_final_level_is_reported_before_publication() {
    let p = TestAdaptivePlan::new(
        TestOperation::First,
        1,
        1,
        AdaptiveDifference::default().with_max_levels(3),
    )
    .unwrap();
    let token = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let mut w = p.workspace::<f64>().unwrap();
    let mut values = [0.0];
    let mut errors = [0.0];
    let mut calls = 0;
    let report = p
        .evaluate_into(
            &[1.0],
            None,
            None,
            |x, y| {
                calls += 1;
                y[0] = (5.0 * x[0]).exp();
                if calls == 7 {
                    token.cancel();
                }
                Ok(())
            },
            &mut values,
            &mut errors,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert_eq!(calls, 7);
}

#[test]
fn subnormal_output_and_adjacent_bounds_do_not_falsely_converge() {
    let p = plan(
        TestOperation::First,
        1,
        1,
        AdaptiveDifferenceMethod::Ridders,
        1e-8,
    );
    let result = p
        .evaluate(&[1.0], None, None, |x, y| {
            y[0] = x[0] * f64::from_bits(1024);
            Ok(())
        })
        .unwrap();
    assert!(result.report().is_success());
    assert!((result.values().unwrap()[0] / f64::from_bits(1024) - 1.0).abs() < 0.01);
    let bounds = Bounds::new(vec![(
        Some(1.0),
        Some(f64::from_bits(1.0_f64.to_bits() + 1)),
    )])
    .unwrap();
    let result = p
        .evaluate(&[1.0], Some(&bounds), None, |x, y| {
            y[0] = x[0];
            Ok(())
        })
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::StepSizeLimit
    );
    assert!(!result.report().is_success());
}

type DenseFunction<S> = fn(&[S], &mut [S]) -> Result<(), SciOxideError>;

fn unavailable_entry_stops_all_remaining_work<S: RealScalar + NumericScalar<Real = S>>() {
    for operation in [
        DerivativeOperation::Gradient,
        DerivativeOperation::Jacobian,
        DerivativeOperation::Hessian,
    ] {
        let shape = match operation {
            DerivativeOperation::Gradient => DerivativeShape::gradient(3),
            DerivativeOperation::Jacobian => DerivativeShape::jacobian(3, 1),
            _ => DerivativeShape::hessian(3),
        };
        for allowance in [7, 64] {
            let method = AdaptiveDifference::default()
                .with_method(AdaptiveDifferenceMethod::Richardson)
                .with_budgets(
                    SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Evaluation, Budget::try_from(allowance).unwrap()),
                );
            let evaluate =
                |function: DenseFunction<S>, x: &[S], bounds: Option<&Bounds<S>>| match operation {
                    DerivativeOperation::Gradient => method.gradient(function, x, bounds),
                    DerivativeOperation::Jacobian => method.jacobian(function, x, 1, bounds),
                    _ => method.hessian(function, x, bounds),
                };
            let evaluate_into =
                |function: DenseFunction<S>,
                 x: &[S],
                 bounds: Option<&Bounds<S>>,
                 values: &mut [S],
                 errors: &mut [S],
                 workspace: &mut AdaptiveDifferenceWorkspace<S>| {
                    let cx = ExecutionContext::sequential();
                    match operation {
                        DerivativeOperation::Gradient => {
                            method.gradient_into(function, x, bounds, values, errors, workspace, cx)
                        }
                        DerivativeOperation::Jacobian => method
                            .jacobian_into(function, x, 1, bounds, values, errors, workspace, cx),
                        _ => {
                            method.hessian_into(function, x, bounds, values, errors, workspace, cx)
                        }
                    }
                };
            let bounds = Bounds::new(vec![
                (None, None),
                (Some(S::ONE), Some(S::ONE)),
                (None, None),
            ])
            .unwrap();
            let point = [S::ONE; 3];
            let function: DenseFunction<S> = |x, y| {
                // Neither a fixed second axis nor any later axis may be sampled.
                assert_eq!(x[1], S::ONE);
                assert_eq!(x[2], S::ONE);
                y[0] = x[0] * x[0] + x[2] * x[2];
                Ok(())
            };
            let owned = evaluate(function, &point, Some(&bounds)).unwrap();
            assert_eq!(
                owned.report().termination(),
                DifferentiationTermination::StepSizeLimit
            );
            assert_eq!(owned.report().completed_entries(), 1);
            assert_eq!(owned.report().evaluations(), 7);
            assert!(owned.values().is_none());
            assert!(owned.errors().is_none());
            let sentinel = S::try_from_f64_lossy(99.0).unwrap();
            let mut values = vec![sentinel; shape.output_len().unwrap()];
            let mut errors = values.clone();
            let mut workspace =
                AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape))
                    .unwrap();
            let report = evaluate_into(
                function,
                &point,
                Some(&bounds),
                &mut values,
                &mut errors,
                &mut workspace,
            )
            .unwrap();
            assert_eq!(
                report.termination(),
                DifferentiationTermination::StepSizeLimit
            );
            assert_eq!(report.completed_entries(), 1);
            assert_eq!(report.evaluations(), 7);
            assert_eq!(report.iterations(), 4);
            assert!(!report.output_available());
            assert!(values.iter().chain(&errors).all(|&v| v == sentinel));
            if allowance == 64 {
                let retry: DenseFunction<S> = |x, y| {
                    y[0] = x.iter().fold(S::ZERO, |sum, &v| sum + S::TWO * v * v);
                    Ok(())
                };
                let report = evaluate_into(
                    retry,
                    &[S::ZERO; 3],
                    None,
                    &mut values,
                    &mut errors,
                    &mut workspace,
                )
                .unwrap();
                assert_eq!(report.termination(), DifferentiationTermination::Converged);
                assert_eq!(report.completed_entries(), values.len());
                assert!(report.output_available());
                for (entry, &value) in values.iter().enumerate() {
                    let expected =
                        if operation == DerivativeOperation::Hessian && entry / 3 == entry % 3 {
                            S::TWO * S::TWO
                        } else {
                            S::ZERO
                        };
                    assert!((value - expected).abs() <= S::SQRT_EPSILON);
                }
                assert!(
                    errors
                        .iter()
                        .all(|&error| error.is_finite() && error >= S::ZERO && error != sentinel)
                );
            }
        }
    }
}
#[test]
fn unavailable_entry_stops_remaining_work_f32() {
    unavailable_entry_stops_all_remaining_work::<f32>();
}
#[test]
fn unavailable_entry_stops_remaining_work_f64() {
    unavailable_entry_stops_all_remaining_work::<f64>();
}

#[test]
fn available_refinement_limit_estimates_allow_later_entries() {
    let result = AdaptiveDifference::default()
        .with_max_levels(3)
        .with_tolerance(ErrorTolerance::scalar(1e-30, 0.0, NormSelection::Infinity).unwrap())
        .gradient(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0].exp() + x[1].exp() + x[2].exp();
                Ok(())
            },
            &[1.0_f64; 3],
            None,
        )
        .unwrap();
    assert_eq!(
        result.report().termination(),
        DifferentiationTermination::RefinementLimit
    );
    assert_eq!(result.report().completed_entries(), 3);
    assert!(result.report().output_available());
    assert!(
        result
            .values()
            .unwrap()
            .iter()
            .all(|v| (*v - core::f64::consts::E).abs() < 1e-4)
    );
}
