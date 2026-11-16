//! Analytic polynomial and resource tests independent of generated weights.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, CapabilityError, Clock, Complex, Complex64, Deadline,
    EvaluationRole, ExecutionContext, InvalidInputReason, MonotonicInstant, NumericScalar,
    NumericalIssue, ProblemToken, RealScalar, SciOxideError, SequentialExecutor, SolveBudgets,
    SolveIdentity, SolveState,
};
use scioxide_differentiate::*;

fn polynomial_evidence<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    for raw in [
        &[-1.0, -0.4, 0.0, 0.3, 0.8, 1.0][..],
        &[0.0, 0.03, 0.2, 0.4, 0.7, 1.0][..],
        &[-1.0, -0.2, -0.001, 0.001, 0.2, 1.0][..],
    ] {
        let nodes: Vec<_> = raw.iter().copied().map(c).collect();
        for order in 0..nodes.len() {
            let stencil = FiniteDifferenceStencil::new(c(0.0), &nodes, order).unwrap();
            for degree in 0..nodes.len() {
                let samples: Vec<_> = nodes.iter().map(|v| v.powi(degree as i32)).collect();
                let expected = if degree == order {
                    (1..=order).fold(S::ONE, |v, i| v * c(i as f64))
                } else {
                    S::ZERO
                };
                let got = stencil.apply(&samples).unwrap();
                let tolerance = c(24.0) * S::EPSILON * stencil.weight_amplification().max(S::ONE);
                assert!(
                    (got - expected).abs() <= tolerance,
                    "order={order} degree={degree}: {got:?} vs {expected:?}"
                );
            }
        }
    }
    // Convergence in h for an independently known analytic derivative.
    for count in [3, 5, 7] {
        let mut errors = Vec::new();
        for h in [0.4, 0.2] {
            let nodes: Vec<_> = (0..count)
                .map(|i| c(h * (i as f64 - (count - 1) as f64 / 2.0)))
                .collect();
            let stencil = FiniteDifferenceStencil::new(S::ZERO, &nodes, 1).unwrap();
            let values: Vec<_> = nodes.iter().map(|v| v.exp()).collect();
            errors.push((stencil.apply(&values).unwrap() - S::ONE).abs());
        }
        assert!(errors[1] <= errors[0] / c(2.0) + c(8.0) * S::EPSILON);
    }
}
#[test]
fn polynomial_and_order_sweeps_f32() {
    polynomial_evidence::<f32>();
}
#[test]
fn polynomial_and_order_sweeps_f64() {
    polynomial_evidence::<f64>();
}

#[test]
fn scale_translation_permutation_and_underflow() {
    let nodes = [0.0, 0.25, 0.5, 1.0];
    let p = FiniteDifferenceStencil::new(0.25, &nodes, 1).unwrap();
    let samples = nodes.map(|x| x * x);
    assert!((p.apply(&samples).unwrap() - 0.5).abs() < 1e-14);
    let reversed = FiniteDifferenceStencil::new(0.25, &[1.0, 0.5, 0.25, 0.0], 1).unwrap();
    assert!((reversed.apply(&[1.0, 0.25, 0.0625, 0.0]).unwrap() - 0.5).abs() < 1e-14);
    for scale in [1e-300_f64, 1e300] {
        let p = FiniteDifferenceStencil::new(0.0, &[0.0, scale, 2.0 * scale], 1).unwrap();
        assert!((p.apply(&[0.0, scale, 2.0 * scale]).unwrap() - 1.0).abs() < 1e-14);
    }
    let sub = f64::from_bits(1);
    let p = FiniteDifferenceStencil::new(0.0, &[0.0, sub], 1).unwrap();
    assert_eq!(p.apply(&[0.0, sub]).unwrap(), 1.0);
    let p = FiniteDifferenceStencil::new(1e10, &[1e10, 1e10 + 1.0, 1e10 + 2.0], 1).unwrap();
    assert!((p.apply(&[0.0, 1.0, 4.0]).unwrap()).abs() < 1e-14);
    assert!(FiniteDifferenceStencil::new(0.0, &[0.0, 0.0], 1).is_err());
    assert!(FiniteDifferenceStencil::new(0.0, &[0.0, 1.0], 2).is_err());
    assert!(FiniteDifferenceStencil::new(f64::NAN, &[0.0, 1.0], 1).is_err());
    assert!(p.apply(&[1.0_f64, f64::NAN, 3.0]).is_err());
}

fn formula_weights<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    // The central difference (f(1) − f(−1))/2 has the weights −1/2, 0, 1/2.
    let central = FiniteDifferenceStencil::new(S::ZERO, &[-S::ONE, S::ZERO, S::ONE], 1).unwrap();
    let weights = central.weights().unwrap();
    let half = c(0.5);
    assert!((weights[0] + half).abs() <= half * S::EPSILON);
    assert_eq!(weights[1], S::ZERO);
    assert!((weights[2] - half).abs() <= half * S::EPSILON);
    for length in [2, 4] {
        let mut out = vec![c(7.0); length];
        assert!(matches!(
            central.weights_into(&mut out).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(out, vec![c(7.0); length]);
    }
    // Σⱼ weights[j]·f(xⱼ) differs from apply by the rounding of a sum of n
    // products, a small multiple of n·ε·Σⱼ |weights[j]|·maxⱼ |f(xⱼ)|.
    let f = |x: S| x.cos() + x * x * x / c(8.0);
    for scale in [1e-3, 1.0, 1e3] {
        for (coordinate, offsets, order) in [
            (0.0, &[-1.0, 0.0, 1.0][..], 1),
            (0.5, &[-2.0, -1.0, 0.0, 1.0, 2.0][..], 2),
            (0.0, &[0.0, 0.1, 0.25, 0.6, 1.0][..], 1),
            (-1.0, &[0.0, -1.0, -2.0, -3.0][..], 2),
        ] {
            let x = c(coordinate);
            let nodes: Vec<S> = offsets.iter().map(|&o| x + c(o * scale)).collect();
            let stencil = FiniteDifferenceStencil::new(x, &nodes, order).unwrap();
            let values: Vec<S> = nodes.iter().map(|&node| f(node)).collect();
            let weights = stencil.weights().unwrap();
            let mut out = vec![S::NAN; nodes.len()];
            stencil.weights_into(&mut out).unwrap();
            assert_eq!(out, weights);
            let sum = weights
                .iter()
                .zip(&values)
                .fold(S::ZERO, |sum, (&w, &v)| sum + w * v);
            let magnitude = weights.iter().fold(S::ZERO, |sum, w| sum + w.abs())
                * values.iter().fold(S::ZERO, |max, v| max.max(v.abs()));
            let tolerance = c(4.0 * nodes.len() as f64) * S::EPSILON * magnitude;
            let applied = stencil.apply(&values).unwrap();
            assert!(
                (sum - applied).abs() <= tolerance,
                "scale {scale}, nodes {nodes:?}: {sum:?} vs {applied:?}"
            );
        }
    }
}
#[test]
fn formula_weights_f32() {
    formula_weights::<f32>();
}
#[test]
fn formula_weights_f64() {
    formula_weights::<f64>();
}

fn multidimensional<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    let point = [c(0.5), c(2.0)];
    let x = [c(0.0), c(0.3), c(0.5), c(0.8), c(1.0)];
    let y = [c(1.0), c(1.4), c(2.0), c(2.2), c(3.0)];
    let p = PhysicalDifferencePlan::new(
        DerivativeOperation::Hessian,
        &point,
        &[&x, &y],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let result = p
        .evaluate(
            |x: &[S], y: &mut [Complex<S>]| {
                let v = x[0] * x[0] * x[1] + x[1].powi(3);
                y[0] = Complex::new(v, v + v);
                Ok(())
            },
            None,
        )
        .unwrap();
    assert!(result.report().output_available());
    for (got, expected) in result.values().unwrap().iter().zip([4.0, 1.0, 1.0, 12.0]) {
        assert!((got.real_component() - c(expected)).abs() < c(200.0) * S::EPSILON);
        assert!((got.imaginary_component() - c(2.0 * expected)).abs() < c(400.0) * S::EPSILON);
    }
    for (op, outputs, expected) in [
        (DerivativeOperation::Gradient, 1, vec![2.0, 12.25]),
        (DerivativeOperation::Jacobian, 2, vec![2.0, 12.25, 1.0, 1.0]),
    ] {
        let p = PhysicalDifferencePlan::new(
            op,
            &point,
            &[&x, &y],
            outputs,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let got = p
            .evaluate(
                |x: &[S], y: &mut [S]| {
                    y[0] = x[0] * x[0] * x[1] + x[1].powi(3);
                    if y.len() == 2 {
                        y[1] = x[0] + x[1];
                    }
                    Ok(())
                },
                None,
            )
            .unwrap()
            .into_parts()
            .1
            .unwrap();
        for (got, expected) in got.iter().zip(expected) {
            assert!((*got - c(expected)).abs() < c(200.0) * S::EPSILON);
        }
    }
}
#[test]
fn real_complex_consumers_f32() {
    multidimensional::<f32>();
}
#[test]
fn real_complex_consumers_f64() {
    multidimensional::<f64>();
}

fn complex_evidence<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    let point = [c(0.5), c(2.0)];
    let h = S::EPSILON * S::EPSILON;
    let extension = |z: &[Complex<S>], y: &mut [Complex<S>]| {
        y[0] = z[0] * z[0] * z[1] + z[1] * z[1] * z[1];
        Ok(())
    };
    let method = ComplexStep::default().with_step(h);
    let result = method.gradient(extension, &point).unwrap();
    assert_eq!(result.report().evaluations(), 3);
    for (got, expected) in result.values().unwrap().iter().zip([2.0, 12.25]) {
        assert!((*got - c(expected)).abs() < c(16.0) * S::EPSILON);
    }
    let x = [c(0.0), c(0.3), c(0.5), c(0.8), c(1.0)];
    let y = [c(1.0), c(1.4), c(2.0), c(2.2), c(3.0)];
    let p = ComplexStepPlan::new(&point, &[&x, &y], method).unwrap();
    let result = p.evaluate(extension, None).unwrap();
    // Both axes contain the real base: each imaginary-axis base observation
    // is sampled once, which leaves 21 - 2 = 19 distinct points.
    assert_eq!(result.report().evaluations(), 19);
    let got = result.values().unwrap();
    assert_eq!(got[1], got[2]);
    for (got, expected) in got.iter().zip([4.0, 1.0, 1.0, 12.0]) {
        assert!((*got - c(expected)).abs() < c(100.0) * S::EPSILON);
    }
    let method = method.with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let got = method
        .jvp(extension, &point, &[c(2.0), c(-1.0)], 1)
        .unwrap()
        .values()
        .unwrap()[0];
    assert!((got - c(-8.25)).abs() < c(16.0) * S::EPSILON);
    let result = method
        .jvp(extension, &point, &[S::ZERO, S::ZERO], 1)
        .unwrap();
    assert_eq!(result.report().evaluations(), 0);
    assert_eq!(result.values().unwrap(), vec![S::ZERO]);
}
#[test]
fn complex_step_f32() {
    complex_evidence::<f32>();
}
#[test]
fn complex_step_f64() {
    complex_evidence::<f64>();
}

#[test]
fn analytic_assumptions_are_not_inferred() {
    let p = ComplexStep::default().with_step(1e-200);
    for (kind, expected) in [(0, -1.0), (1, 0.0), (2, 0.0)] {
        let got = p
            .derivative(
                |z: Complex64| {
                    Ok(match kind {
                        0 => Complex::new(z.re, -z.im),
                        1 => Complex::new(z.norm(), 0.0),
                        _ => Complex::new(z.re, 0.0),
                    })
                },
                2.0,
            )
            .unwrap()
            .values()
            .unwrap()[0];
        assert_eq!(got, expected); // All disagree with the real derivative +1.
    }
    assert!(
        p.derivative(|_: Complex64| Ok(Complex::new(1.0, 1.0)), 2.0)
            .is_err()
    );
    // A callback can erase a tiny imaginary signal: no finite probe proves otherwise.
    let p = ComplexStep::default().with_step(f64::MIN_POSITIVE);
    let got = p
        .derivative(|z: Complex64| Ok(z * Complex::new(1e-20, 0.0)), 0.0)
        .unwrap()
        .values()
        .unwrap()[0];
    assert_eq!(got, 0.0);
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(469),
        scioxide_core::algorithm_id!("test", "prepared", 1),
        0,
        0,
    )
}
#[test]
fn budgets_transactions_and_allocation_free_reuse() {
    let nodes = [0.0, 0.25, 0.5, 0.75, 1.0];
    let real = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[0.5],
        &[&nodes],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let complex = ComplexStep::default().with_step(1e-100);
    let mut rw = real.workspace(()).unwrap();
    let mut cw = ComplexStepWorkspace::workspace(DerivativeRequest::new(
        &complex,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    for complex_mode in [false, true] {
        for limit in 1..=6 {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let executor = SequentialExecutor;
            let mut context = state
                .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
                .unwrap();
            let mut output = [99.0];
            let report = if complex_mode {
                complex
                    .gradient_into(
                        |z: &[Complex64], y: &mut [Complex64]| {
                            y[0] = z[0] * z[0];
                            Ok(())
                        },
                        &[0.5],
                        &mut output,
                        &mut cw,
                        &mut context,
                    )
                    .unwrap()
            } else {
                real.evaluate_into(
                    |x: &[f64], y: &mut [_]| {
                        y[0] = x[0] * x[0];
                        Ok(())
                    },
                    None,
                    &mut output,
                    &mut rw,
                    &mut context,
                )
                .unwrap()
            };
            let required = if complex_mode { 2 } else { 5 };
            assert_eq!(report.evaluations(), limit.min(required));
            assert_eq!(
                context
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Function),
                report.evaluations()
            );
            assert_eq!(report.output_available(), limit >= required);
            if limit < required {
                assert_eq!(output, [99.0]);
            } else {
                assert!((output[0] - 1.0).abs() < 1e-14);
            }
        }
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
        let mut output = [99.0];
        let mut calls = 0;
        let report = if complex_mode {
            complex
                .gradient_into(
                    |z: &[Complex64], y: &mut [Complex64]| {
                        y[0] = z[0] * z[0];
                        calls += 1;
                        if calls == 2 {
                            token.cancel();
                        }
                        Ok(())
                    },
                    &[0.5],
                    &mut output,
                    &mut cw,
                    &mut context,
                )
                .unwrap()
        } else {
            real.evaluate_into(
                |x: &[f64], y: &mut [_]| {
                    y[0] = x[0] * x[0];
                    calls += 1;
                    if calls == 5 {
                        token.cancel();
                    }
                    Ok(())
                },
                None,
                &mut output,
                &mut rw,
                &mut context,
            )
            .unwrap()
        };
        assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
        assert_eq!(output, [99.0]);
    }
}
#[test]
fn errors_empty_shapes_and_iteration_limits() {
    let p = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[0.0],
        &[&[0.0, 1.0, 2.0]],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut w = p.workspace::<f64>(()).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut output = [99.0];
    let error = p.evaluate_into(
        |_: &[f64], _: &mut [_]| Ok(()),
        None,
        &mut output,
        &mut w,
        &mut context,
    );
    assert!(matches!(
        (error).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, .. } if frames[0].evaluations == 1
    ));
    assert_eq!(output, [99.0]);
    let report = p
        .evaluate_into(
            |x: &[f64], y: &mut [_]| {
                y[0] = x[0];
                Ok(())
            },
            None,
            &mut output,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert!(report.output_available());
    assert!((output[0] - 1.0).abs() < 1e-14);
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert!(context.record_iteration().unwrap().is_granted());
    let limited = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[0.0, 0.0],
        &[&[0.0, 1.0, 2.0][..]; 2],
        1,
        NodeDifferenceOptions::default().with_budgets(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap()),
        ),
    )
    .unwrap();
    let mut work = limited.workspace::<f64>(()).unwrap();
    let mut output = [99.0; 2];
    let report = limited
        .evaluate_into(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0] + x[1];
                Ok(())
            },
            None,
            &mut output,
            &mut work,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::IterationLimit
    );
    assert_eq!(report.evaluations(), 3);
    assert_eq!(report.completed_entries(), 1);
    assert_eq!(output, [99.0; 2]);
    for op in [
        DerivativeOperation::Gradient,
        DerivativeOperation::Jacobian,
        DerivativeOperation::Hessian,
    ] {
        let p = PhysicalDifferencePlan::<f64>::new(
            op,
            &[],
            &[],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        assert_eq!(
            p.evaluate(|_: &[f64], _: &mut [f64]| unreachable!(), None)
                .unwrap()
                .values()
                .unwrap(),
            Vec::<f64>::new()
        );
        let method = ComplexStep::<f64>::default();
        let f =
            |_: &[Complex64], _: &mut [Complex64]| -> Result<(), SciOxideError> { unreachable!() };
        let result = match op {
            DerivativeOperation::Hessian => RelativeComplexStep::new(&[], method)
                .unwrap()
                .evaluate(f, &[], None),
            DerivativeOperation::Jacobian => method.jacobian(f, &[], 1),
            _ => method.gradient(f, &[]),
        };
        assert!(result.unwrap().values().unwrap().is_empty());
    }
}

struct ExpiredClock;
impl Clock for ExpiredClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(2),
        ))
    }
}
#[test]
fn parent_allowances_deadlines_and_complex_failures() {
    let p = ComplexStep::default().with_step(1e-100);
    let mut w =
        ComplexStepWorkspace::workspace(DerivativeRequest::new(&p, DerivativeShape::gradient(2)))
            .unwrap();
    let mut output = [99.0; 2];
    let mut parent_state = SolveState::new();
    let mut child_state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
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
                p.gradient_into(
                    |x: &[Complex64], y: &mut [Complex64]| {
                        y[0] = x[0] + x[1];
                        Ok(())
                    },
                    &[1.0, 2.0],
                    &mut output,
                    &mut w,
                    child,
                )
            },
        )
        .unwrap();
    assert_eq!(report.evaluations(), 1);
    assert_eq!(
        report.termination(),
        DifferentiationTermination::EvaluationLimit
    );
    assert_eq!(output, [99.0; 2]);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let clock = ExpiredClock;
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &clock_binding,
        MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_secs(1)),
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential()
                .with_deadline(deadline)
                .unwrap(),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let report = p
        .gradient_into(
            |_: &[Complex64], _: &mut [Complex64]| unreachable!(),
            &[1.0, 2.0],
            &mut output,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::DeadlineReached
    );
    assert_eq!(report.evaluations(), 0);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut calls = 0;
    let error = p.gradient_into(
        |x: &[Complex64], y: &mut [Complex64]| {
            calls += 1;
            if calls == 1 {
                y[0] = x[0] + x[1];
            }
            Ok(())
        },
        &[1.0, 2.0],
        &mut output,
        &mut w,
        &mut context,
    );
    assert!(matches!(
        (error).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, .. } if frames[0].evaluations == 2
    ));
    assert_eq!(output, [99.0; 2]);
    let mut bad_output = [];
    let error = p.gradient_into(
        |_: &[Complex64], _: &mut [Complex64]| unreachable!(),
        &[1.0, 2.0],
        &mut bad_output,
        &mut w,
        &mut context,
    );
    assert!(matches!(
        (error).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(
        context
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Function),
        2
    );
}
#[test]
fn scalar_second_and_complex_step_order() {
    let cubic = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] * x[0]
            + Complex::new(3.0, 0.0) * x[0] * x[0]
            + Complex::new(2.0, 0.0) * x[0];
        Ok(())
    };
    for op in [DerivativeOperation::First, DerivativeOperation::Second] {
        let nodes = [0.0, 0.1, 0.2, 0.4, 0.7];
        let real = PhysicalDifferencePlan::new(
            op,
            &[0.0],
            &[&nodes],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let expected = if op == DerivativeOperation::First {
            2.0
        } else {
            6.0
        };
        let result = real
            .evaluate(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0].powi(3) + 3.0 * x[0] * x[0] + 2.0 * x[0];
                    Ok(())
                },
                None,
            )
            .unwrap();
        assert!((result.values().unwrap()[0] - expected).abs() < 1e-12);
        let method = ComplexStep::default().with_step(1e-100);
        let result = if op == DerivativeOperation::Second {
            RelativeComplexStep::new(&[&nodes], method)
                .unwrap()
                .evaluate(cubic, &[0.0], None)
        } else {
            method.gradient(cubic, &[0.0])
        }
        .unwrap();
        assert!((result.values().unwrap()[0] - expected).abs() < 1e-12);
    }
    let mut errors = Vec::new();
    for h in [0.2, 0.1, 0.05] {
        let value = ComplexStep::default()
            .with_step(h)
            .derivative(|z: Complex64| Ok(z.exp()), 0.7)
            .unwrap()
            .values()
            .unwrap()[0];
        errors.push((value - 0.7_f64.exp()).abs());
    }
    for pair in errors.windows(2) {
        assert!(pair[0] / pair[1] > 3.9 && pair[0] / pair[1] < 4.1);
    }
    let error = ComplexStep::default()
        .with_step(0.0)
        .derivative(
            |_: Complex64| -> Result<Complex64, SciOxideError> { unreachable!() },
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            argument,
            reason: InvalidInputReason::OutOfRange,
            ..
        } if argument == scioxide_core::argument!("self")
    ));
    let result = ComplexStep::default()
        .with_step(1e-100)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
        .jvp(
            |_: &[Complex64], y: &mut [Complex64]| {
                y[0] = Complex::new(1.0, 0.0);
                Ok(())
            },
            &[],
            &[],
            1,
        )
        .unwrap();
    assert_eq!(result.report().evaluations(), 1);
    assert_eq!(result.values().unwrap(), vec![0.0]);
}

#[test]
fn stencil_failures_have_specific_error_kinds() {
    assert!(matches!(
        (FiniteDifferenceStencil::new(f64::NAN, &[0.0, 1.0], 1))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert!(matches!(
        (FiniteDifferenceStencil::new(0.0_f64, &[0.0, 0.0], 1))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: InvalidInputReason::Inconsistent,
            ..
        }
    ));
    let p = FiniteDifferenceStencil::new(0.0_f64, &[0.0, 2.0], 1).unwrap();
    assert!(matches!(
        (p.apply(&[1.0_f64])).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (p.apply(&[0.0, f64::from_bits(1)]))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
    // Entry points that share preparation kernels report their own parameter.
    let stencil = FiniteDifferenceStencil::new(0.0_f64, &[0.0, 0.0], 1).unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _stencil_argument,
        reason: _stencil_reason,
        detail: _stencil_detail,
    } = stencil.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_stencil_argument, scioxide_core::argument!("nodes"));
    let offsets = [0.0_f64, 0.0];
    let relative = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _relative_argument,
        reason: _relative_reason,
        detail: _relative_detail,
    } = relative.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_relative_argument, scioxide_core::argument!("offsets"));
    let method = ComplexStep::default();
    for (error, expected) in [
        (
            RelativeComplexStep::new(&[&offsets], method).unwrap_err(),
            scioxide_core::argument!("offsets"),
        ),
        (
            ComplexStepPlan::new(&[0.0], &[&offsets], method).unwrap_err(),
            scioxide_core::argument!("nodes"),
        ),
    ] {
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                argument,
                reason: InvalidInputReason::Inconsistent,
                ..
            } if argument == expected
        ));
    }
    let axis: &[f64] = &[-1.0, 1.0];
    let shape = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[0.0; 2],
        &[axis, axis],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: _shape_argument,
        reason: _shape_reason,
        detail: _shape_detail,
    } = shape.view()
    else {
        unreachable!("unexpected error kind")
    };

    assert_eq!(_shape_argument, scioxide_core::argument!("operation"));
}
