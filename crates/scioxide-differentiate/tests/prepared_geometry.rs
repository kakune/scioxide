//! Independent actual-node interpolation and prepared lifecycle coverage.
#![allow(clippy::unwrap_used, clippy::panic, clippy::float_cmp)]

use scioxide_core::{
    Bounds, Budget, BudgetKind, CancellationToken, CapabilityError, Clock, ClockBinding,
    ColoringKind, ColoringSchedule, Complex, Complex64, ConvergenceReason, Deadline,
    DifferentiationSite, EvaluationRole, ExecutionContext, InvalidInputReason, MonotonicInstant,
    NdView, NumericScalar, ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveIdentity,
    SolveState, StridedLayout, StructuralPattern,
};
use scioxide_differentiate::*;

// Independently expand each Lagrange cardinal polynomial in normalized physical
// displacements. No production recurrence or weight application is called here.
fn reference_weights(nodes: &[f64], coordinate: f64, scale: f64, order: usize) -> Vec<f64> {
    nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            let xi = (node - coordinate) / scale;
            let mut coefficients = vec![0.0; nodes.len()];
            coefficients[0] = 1.0;
            let mut degree = 0;
            let mut denominator = 1.0;
            for (j, &other) in nodes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let xj = (other - coordinate) / scale;
                for k in (1..=degree + 1).rev() {
                    coefficients[k] = coefficients[k - 1] - xj * coefficients[k];
                }
                coefficients[0] *= -xj;
                degree += 1;
                denominator *= xi - xj;
            }
            let factorial = (1..=order).product::<usize>() as f64;
            coefficients[order] * factorial / denominator
        })
        .collect()
}

fn actual_moments<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let v = |x| S::try_from_f64_lossy(x).unwrap();
    let epsilon = S::EPSILON;
    // Addition crosses a binade boundary: negative and positive spacing differ.
    let offsets = [
        v(-2.1) * epsilon,
        v(-0.8) * epsilon,
        S::ZERO,
        v(0.6) * epsilon,
        v(1.6) * epsilon,
    ];
    let points = [
        S::ONE.next_after(S::ZERO),
        S::ONE,
        S::ONE.next_after(v(2.0)),
    ];
    for (operation, order) in [
        (DerivativeOperation::First, 1),
        (DerivativeOperation::Second, 2),
    ] {
        let plan = RelativeDifference::new(
            operation,
            &[&offsets],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let mut workspace = plan.workspace::<Complex<S>>(()).unwrap();
        for coordinate in points {
            let nodes = offsets.map(|offset| (coordinate + offset).into_f64());
            let weights =
                reference_weights(&nodes, coordinate.into_f64(), epsilon.into_f64(), order);
            for power in 0..offsets.len() {
                let mut actual = [f64::NAN; 5];
                let mut count = 0;
                let mut output = [Complex::new(S::NAN, S::NAN)];
                let report = plan
                    .evaluate_into(
                        |x: &[S], y: &mut [_]| {
                            actual[count] = x[0].into_f64();
                            count += 1;
                            let t = (x[0] - coordinate) / epsilon;
                            let value = t.powi(power as i32);
                            y[0] = Complex::new(value, -value);
                            Ok(())
                        },
                        &[coordinate],
                        None,
                        &mut output,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(report.output_available());
                assert_eq!(actual, nodes);
                let expected = if power == order {
                    (1..=order).product::<usize>() as f64
                } else {
                    0.0
                };
                let reference: f64 = weights
                    .iter()
                    .zip(nodes)
                    .map(|(&weight, node)| {
                        weight
                            * ((node - coordinate.into_f64()) / epsilon.into_f64())
                                .powi(power as i32)
                    })
                    .sum();
                assert!((reference - expected).abs() < 1e-11);
                let condition: f64 = weights
                    .iter()
                    .zip(nodes)
                    .map(|(&weight, node)| {
                        (weight
                            * ((node - coordinate.into_f64()) / epsilon.into_f64())
                                .powi(power as i32))
                        .abs()
                    })
                    .sum();
                let tolerance = 64.0 * epsilon.into_f64() * condition.max(1.0);
                let real =
                    output[0].real_component().into_f64() * epsilon.into_f64().powi(order as i32);
                let imag = output[0].imaginary_component().into_f64()
                    * epsilon.into_f64().powi(order as i32);
                assert!(
                    (real - expected).abs() <= tolerance,
                    "{operation:?} power {power}: {real} vs {expected}"
                );
                assert!((imag + expected).abs() <= tolerance);
            }
        }
    }
}
#[test]
fn actual_rounded_moments_f32() {
    actual_moments::<f32>();
}
#[test]
fn actual_rounded_moments_f64() {
    actual_moments::<f64>();
}

fn changing_operations<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let v = |x| S::try_from_f64_lossy(x).unwrap();
    let offsets = [-0.5, -0.25, 0.0, 0.25, 0.5].map(v);
    let tolerance = 256.0 * S::EPSILON.into_f64();
    for operation in [
        DerivativeOperation::First,
        DerivativeOperation::Second,
        DerivativeOperation::Gradient,
        DerivativeOperation::Jacobian,
        DerivativeOperation::Hessian,
    ] {
        let scalar = matches!(
            operation,
            DerivativeOperation::First | DerivativeOperation::Second
        );
        let n = if scalar { 1 } else { 2 };
        let m = if operation == DerivativeOperation::Jacobian {
            2
        } else {
            1
        };
        let axes: [&[S]; 2] = [&offsets, &offsets];
        let plan = RelativeDifference::new(
            operation,
            &axes[..n],
            m,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let mut workspace = plan.workspace::<Complex<S>>(()).unwrap();
        let mut output = vec![Complex::new(S::NAN, S::NAN); plan.output_len()];
        for ordinal in 0..48 {
            let point = [
                v((ordinal as f64 - 24.0) / 16.0),
                v((ordinal as f64 + 1.0) / 16.0),
            ];
            let a = point[0].into_f64();
            let b = point[1].into_f64();
            let expected = match operation {
                DerivativeOperation::First => vec![2.0 * a],
                DerivativeOperation::Second => vec![2.0],
                DerivativeOperation::Gradient => vec![2.0 * a + b, a + 3.0 * b * b],
                DerivativeOperation::Jacobian => vec![2.0 * a + b, a + 3.0 * b * b, 1.0, 1.0],
                DerivativeOperation::Hessian => vec![2.0, 1.0, 1.0, 6.0 * b],
            };
            output.fill(Complex::new(S::NAN, S::NAN));
            let report = plan
                .evaluate_into(
                    |x: &[S], y: &mut [_]| {
                        let value = if scalar {
                            x[0] * x[0]
                        } else {
                            x[0] * x[0] + x[0] * x[1] + x[1].powi(3)
                        };
                        y[0] = Complex::new(value, -value);
                        if m == 2 {
                            y[1] = Complex::new(x[0] + x[1], -x[0] - x[1]);
                        }
                        Ok(())
                    },
                    &point[..n],
                    None,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.output_available());
            for (actual, expected) in output.iter().zip(expected) {
                assert!(
                    (actual.real_component().into_f64() - expected).abs()
                        <= tolerance * expected.abs().max(1.0)
                );
                assert!(
                    (actual.imaginary_component().into_f64() + expected).abs()
                        <= tolerance * expected.abs().max(1.0)
                );
            }
            // An unwritten lane poisons samples. Reuse must restore all state.
            let before = output.clone();
            assert!(
                plan.evaluate_into(
                    |_: &[S], _: &mut [_]| Ok(()),
                    &point[..n],
                    None,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
            );
            assert_eq!(output, before);
        }
    }
    let h = S::EPSILON * S::EPSILON;
    let method = ComplexStep::default().with_step(h);
    let second = RelativeComplexStep::new(&[&offsets], method).unwrap();
    let hessian = RelativeComplexStep::new(&[&offsets, &offsets], method).unwrap();
    let request = |shape: DerivativeShape| {
        ComplexStepWorkspace::workspace(DerivativeRequest::new(&method, shape))
    };
    for index in 0..5 {
        let scalar = index < 2;
        let n = if scalar { 1 } else { 2 };
        let (workspace, len) = match index {
            0 => (request(DerivativeShape::gradient(1)), 1),
            1 => (second.workspace(()), 1),
            2 => (request(DerivativeShape::gradient(2)), 2),
            3 => (request(DerivativeShape::jacobian(2, 2)), 4),
            _ => (hessian.workspace(()), 4),
        };
        let mut workspace = workspace.unwrap();
        let mut output = vec![S::NAN; len];
        for ordinal in 0..48 {
            let point = [
                v((ordinal as f64 - 24.0) / 16.0),
                v((ordinal as f64 + 1.0) / 16.0),
            ];
            let a = point[0].into_f64();
            let b = point[1].into_f64();
            let expected = match index {
                0 => vec![2.0 * a],
                1 => vec![2.0],
                2 => vec![2.0 * a + b, a + 3.0 * b * b],
                3 => vec![2.0 * a + b, a + 3.0 * b * b, 1.0, 1.0],
                _ => vec![2.0, 1.0, 1.0, 6.0 * b],
            };
            output.fill(S::NAN);
            let report = complex_operation(
                index,
                &method,
                [&second, &hessian],
                |x: &[Complex<S>], y: &mut [Complex<S>]| {
                    y[0] = if scalar {
                        x[0] * x[0]
                    } else {
                        x[0] * x[0] + x[0] * x[1] + x[1] * x[1] * x[1]
                    };
                    if y.len() == 2 {
                        y[1] = x[0] + x[1];
                    }
                    Ok(())
                },
                &point[..n],
                &mut output,
                &mut workspace,
            )
            .unwrap();
            assert!(report.output_available());
            for (&actual, expected) in output.iter().zip(expected) {
                assert!(
                    (actual.into_f64() - expected).abs() <= tolerance * expected.abs().max(1.0)
                );
            }
            let before = output.clone();
            assert!(
                complex_operation(
                    index,
                    &method,
                    [&second, &hessian],
                    |_: &[Complex<S>], _: &mut [Complex<S>]| Ok(()),
                    &point[..n],
                    &mut output,
                    &mut workspace,
                )
                .is_err()
            );
            assert_eq!(output, before);
        }
    }
    let directional = method.with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let mut workspace = ComplexStepWorkspace::workspace(DerivativeRequest::new(
        &directional,
        DerivativeShape::jvp(2, 1),
    ))
    .unwrap();
    for ordinal in 0..48 {
        let point = [v(ordinal as f64 / 16.0), v(0.5)];
        let mut output = [S::NAN];
        directional
            .jvp_into(
                |x: &[Complex<S>], y: &mut [Complex<S>]| {
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    Ok(())
                },
                &point,
                &[v(2.0), v(-1.0)],
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((output[0].into_f64() - (4.0 * point[0].into_f64() - 1.0)).abs() <= tolerance);
    }
}
// Runs operation `index` of the complex-step family: the derivative, the
// second derivative, the gradient, the Jacobian with two outputs or the Hessian.
fn complex_operation<S, F>(
    index: usize,
    method: &ComplexStep<S>,
    relative: [&RelativeComplexStep<S>; 2],
    function: F,
    x: &[S],
    out: &mut [S],
    workspace: &mut ComplexStepWorkspace<S>,
) -> Result<DifferentiationReport, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    F: SampleFunction<Complex<S>, Complex<S>>,
{
    let context = ExecutionContext::sequential();
    match index {
        0 | 2 => method.gradient_into(function, x, out, workspace, context),
        1 => relative[0].evaluate_into(function, x, None, out, workspace, context),
        3 => method.jacobian_into(function, x, 2, out, workspace, context),
        _ => relative[1].evaluate_into(function, x, None, out, workspace, context),
    }
}
#[test]
fn changing_points_all_operations_f32() {
    changing_operations::<f32>();
}
#[test]
fn changing_points_all_operations_f64() {
    changing_operations::<f64>();
}

fn invalid_geometry<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let v = |x| S::try_from_f64_lossy(x).unwrap();
    for offsets in [
        [S::ZERO, S::EPSILON, v(2.0) * S::EPSILON],
        [S::ZERO, v(0.6) * S::EPSILON, v(0.7) * S::EPSILON],
    ] {
        let plan = RelativeDifference::new(
            DerivativeOperation::First,
            &[&offsets],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let mut workspace = plan.workspace::<S>(()).unwrap();
        let mut output = [v(99.0)];
        let point = if offsets[2] == v(2.0) * S::EPSILON {
            v(1e10)
        } else {
            S::ONE
        };
        let error = plan
            .evaluate_into(
                |_: &[S], _: &mut [_]| panic!("callback on invalid geometry"),
                &[point],
                None,
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(
            invalid_input(&error),
            (
                scioxide_core::argument!("x"),
                InvalidInputReason::UnsupportedConfiguration
            )
        );
        assert_eq!(output, [v(99.0)]);
        {
            assert!(
                plan.evaluate_into(
                    |x: &[S], y: &mut [_]| {
                        y[0] = x[0];
                        Ok(())
                    },
                    &[S::ZERO],
                    None,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap()
                .output_available()
            );
        };
        assert!((output[0] - S::ONE).abs() <= v(32.0) * S::EPSILON);
    }
    let offsets = [v(0.0), v(0.25), v(0.5)];
    let bounds = Bounds::new(vec![(Some(S::ZERO), Some(S::ONE))]).unwrap();
    let real = RelativeDifference::new(
        DerivativeOperation::Second,
        &[&offsets],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let complex = RelativeComplexStep::new(
        &[&offsets],
        ComplexStep::default().with_step(S::EPSILON * S::EPSILON),
    )
    .unwrap();
    let mut rw = real.workspace::<S>(()).unwrap();
    let mut cw = complex.workspace(()).unwrap();
    let mut output = [v(99.0)];
    let outside = (
        scioxide_core::argument!("x"),
        InvalidInputReason::UnsupportedConfiguration,
    );
    let error = real
        .evaluate_into(
            |_: &[S], _: &mut [_]| panic!("callback outside bounds"),
            &[v(0.75)],
            Some(&bounds),
            &mut output,
            &mut rw,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(invalid_input(&error), outside);
    let error = complex
        .evaluate_into(
            |_: &[Complex<S>], _: &mut [Complex<S>]| panic!("complex callback outside bounds"),
            &[v(0.75)],
            Some(&bounds),
            &mut output,
            &mut cw,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(invalid_input(&error), outside);
    assert_eq!(output, [v(99.0)]);
    {
        real.evaluate_into(
            |x: &[S], y: &mut [_]| {
                assert!(x[0] >= S::ZERO && x[0] <= S::ONE);
                y[0] = x[0] * x[0];
                Ok(())
            },
            &[S::ZERO],
            Some(&bounds),
            &mut output,
            &mut rw,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    };
    assert!((output[0] - v(2.0)).abs() <= v(32.0) * S::EPSILON);
    {
        complex
            .evaluate_into(
                |x: &[Complex<S>], y: &mut [Complex<S>]| {
                    assert!(x[0].real_component() >= S::ZERO && x[0].real_component() <= S::ONE);
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &[S::ZERO],
                Some(&bounds),
                &mut output,
                &mut cw,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
    };
    assert!((output[0] - v(2.0)).abs() <= v(32.0) * S::EPSILON);
}
// Returns the argument and reason of an `InvalidInput` error.
fn invalid_input(error: &SciOxideError) -> (scioxide_core::Argument, InvalidInputReason) {
    match error.view() {
        scioxide_core::SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => (argument, reason),
        _ => panic!("expected InvalidInput, got {error:?}"),
    }
}
#[test]
fn geometry_fails_before_callbacks_f32() {
    invalid_geometry::<f32>();
}
#[test]
fn geometry_fails_before_callbacks_f64() {
    invalid_geometry::<f64>();
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "prepared-geometry", 1),
        0,
        0,
    )
}
#[test]
fn typed_base_direction_and_mixed_failure_locations() {
    let failure = || {
        SciOxideError::invalid_input(
            scioxide_core::argument!("callback"),
            InvalidInputReason::OutOfRange,
            "injected failure",
        )
    };
    let site = |error: SciOxideError| match error.view() {
        scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, .. } => {
            frames[0].location.site()
        }
        _ => panic!("expected DifferentiationFailure, got {error:?}"),
    };
    let method = FiniteDifference::default().with_scheme(DifferenceScheme::Central);
    let mut gradient = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    let mut out = [99.0; 2];
    let error = method
        .gradient_into(
            |_: &[f64], _: &mut [f64]| Err(failure()),
            &[1.0, 2.0],
            None,
            None,
            &mut out,
            &mut gradient,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(site(error), DifferentiationSite::BasePoint);
    let error = AdaptiveDifference::<f64>::default()
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
        .jvp(
            |_: &[f64], _: &mut [f64]| Err(failure()),
            &[1.0, 2.0],
            &[1.0, 0.0],
            1,
            None,
        )
        .unwrap_err();
    assert_eq!(site(error), DifferentiationSite::Direction);
    let offsets = [-0.25, 0.0, 0.25];
    let p = RelativeDifference::new(
        DerivativeOperation::Hessian,
        &[&offsets, &offsets],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let error = p
        .evaluate(
            |x: &[f64], y: &mut [f64]| {
                if x[0] != 1.0 && x[1] != 2.0 {
                    return Err(failure());
                }
                y[0] = x[0] * x[1];
                Ok(())
            },
            &[1.0, 2.0],
            None,
        )
        .unwrap_err();
    assert_eq!(site(error), DifferentiationSite::Variables(0, 1));
    let p = RelativeComplexStep::new(
        &[&offsets, &offsets],
        ComplexStep::default().with_step(1e-100),
    )
    .unwrap();
    let error = p
        .evaluate(
            |x: &[Complex64], y: &mut [Complex64]| {
                if x[0].imaginary_component() != 0.0 && x[1].real_component() != 2.0 {
                    return Err(failure());
                }
                y[0] = x[0] * x[1];
                Ok(())
            },
            &[1.0, 2.0],
            None,
        )
        .unwrap_err();
    assert_eq!(site(error), DifferentiationSite::Variables(0, 1));
}

#[test]
fn every_zero_direction_publication_polls_cancellation() {
    for policy in [
        ZeroDirectionPolicy::MathematicalZero,
        ZeroDirectionPolicy::ValidateBase,
    ] {
        let token = CancellationToken::new();
        if policy == ZeroDirectionPolicy::MathematicalZero {
            token.cancel();
        }
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
        let mut errors = [88.0];
        let p = AdaptiveDifference::<f64>::default().with_zero_direction(policy);
        let mut w = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
            &p,
            DerivativeShape::jvp(1, 1),
        ))
        .unwrap();
        let report = p
            .jvp_into(
                |_: &[f64], y: &mut [f64]| {
                    y[0] = 1.0;
                    token.cancel();
                    Ok(())
                },
                &[1.0],
                &[-0.0],
                None,
                &mut output,
                &mut errors,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
        assert!(!report.output_available());
        assert_eq!(output, [99.0]);
        assert_eq!(errors, [88.0]);
        let p = ComplexStep::default()
            .with_step(1e-100)
            .with_zero_direction(policy);
        let mut w =
            ComplexStepWorkspace::workspace(DerivativeRequest::new(&p, DerivativeShape::jvp(1, 1)))
                .unwrap();
        let report = p
            .jvp_into(
                |_: &[Complex64], y: &mut [Complex64]| {
                    y[0] = Complex64::new(1.0, 0.0);
                    Ok(())
                },
                &[1.0],
                &[-0.0],
                &mut output,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
        assert!(!report.output_available());
        assert_eq!(output, [99.0]);
        let p = FiniteDifference::default()
            .with_scheme(DifferenceScheme::Central)
            .with_zero_direction(policy);
        let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
            &p,
            DerivativeShape::jvp(1, 1),
        ))
        .unwrap();
        let report = p
            .jvp_into(
                |_: &[f64], _: &mut [f64]| panic!("cancelled callback"),
                &[1.0],
                &[-0.0],
                None,
                None,
                &mut output,
                &mut w,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
        assert_eq!(output, [99.0]);
    }
}

#[test]
fn fixed_context_budgets_and_final_callback_stop_are_transactional() {
    for limit in 0..=3 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::try_from(limit.max(1)).unwrap(),
        );
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        if limit == 0 {
            assert!(
                ctx.record_provider(EvaluationRole::Function)
                    .unwrap()
                    .is_granted()
            );
        }
        let method = FiniteDifference::default().with_scheme(DifferenceScheme::Central);
        let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
            &method,
            DerivativeShape::gradient(1),
        ))
        .unwrap();
        let mut out = [99.0];
        let mut calls = 0;
        let report = method
            .gradient_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &[1.0],
                None,
                None,
                &mut out,
                &mut w,
                &mut ctx,
            )
            .unwrap();
        assert_eq!(report.evaluations(), calls);
        assert_eq!(calls, limit.min(3));
        if limit < 3 {
            assert_eq!(out, [99.0]);
            assert_eq!(
                report.termination(),
                DifferentiationTermination::EvaluationLimit
            );
        } else {
            assert!((out[0] - 2.0).abs() < 1e-10);
        }
    }
    let token = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let method = FiniteDifference::default().with_scheme(DifferenceScheme::Central);
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    let mut out = [99.0];
    let mut calls = 0;
    let report = method
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                calls += 1;
                if calls == 3 {
                    token.cancel();
                }
                y[0] = x[0] * x[0];
                Ok(())
            },
            &[1.0],
            None,
            None,
            &mut out,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert!(!report.output_available());
    assert_eq!(out, [99.0]);
}

fn complex_rounded_moments<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let v = |x| S::try_from_f64_lossy(x).unwrap();
    let epsilon = S::EPSILON;
    let offsets = [
        v(-2.1) * epsilon,
        v(-0.8) * epsilon,
        S::ZERO,
        v(0.6) * epsilon,
        v(1.6) * epsilon,
    ];
    let p = RelativeComplexStep::new(
        &[&offsets],
        ComplexStep::default().with_step(epsilon * epsilon),
    )
    .unwrap();
    let mut w = p.workspace(()).unwrap();
    for coordinate in [
        S::ONE.next_after(S::ZERO),
        S::ONE,
        S::ONE.next_after(v(2.0)),
    ] {
        let nodes = offsets.map(|offset| (coordinate + offset).into_f64());
        let weights = reference_weights(&nodes, coordinate.into_f64(), epsilon.into_f64(), 1);
        for power in 0..5 {
            let expected = if power == 2 { 2.0 } else { 0.0 };
            let mut output = [S::NAN];
            let mut calls = 0;
            p.evaluate_into(
                |x: &[Complex<S>], y: &mut [Complex<S>]| {
                    if calls > 0 {
                        assert_eq!(x[0].real_component().into_f64(), nodes[calls - 1]);
                    }
                    calls += 1;
                    let t =
                        (x[0] - Complex::new(coordinate, S::ZERO)) / Complex::new(epsilon, S::ZERO);
                    y[0] = t.powi(power);
                    Ok(())
                },
                &[coordinate],
                None,
                &mut output,
                &mut w,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
            let derivative = |node: f64| {
                if power == 0 {
                    0.0
                } else {
                    power as f64
                        * ((node - coordinate.into_f64()) / epsilon.into_f64()).powi(power - 1)
                }
            };
            let reference: f64 = weights
                .iter()
                .zip(nodes)
                .map(|(&weight, node)| weight * derivative(node))
                .sum();
            assert!((reference - expected).abs() < 1e-10);
            let condition: f64 = weights
                .iter()
                .zip(nodes)
                .map(|(&weight, node)| (weight * derivative(node)).abs())
                .sum();
            let tolerance = 64.0 * epsilon.into_f64() * condition.max(1.0);
            let actual = output[0].into_f64() * epsilon.into_f64().powi(2);
            assert!(
                (actual - expected).abs() <= tolerance,
                "power {power}: {actual} vs {expected}"
            );
        }
    }
}
#[test]
fn complex_real_nodes_have_actual_moments_f32() {
    complex_rounded_moments::<f32>();
}
#[test]
fn complex_real_nodes_have_actual_moments_f64() {
    complex_rounded_moments::<f64>();
}

#[test]
fn zero_direction_base_callback_cancellation_suppresses_fixed_and_complex_output() {
    let token = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let p = ComplexStep::default()
        .with_step(1e-100)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    let mut w =
        ComplexStepWorkspace::workspace(DerivativeRequest::new(&p, DerivativeShape::jvp(1, 1)))
            .unwrap();
    let mut output = [99.0];
    let report = p
        .jvp_into(
            |_: &[Complex64], y: &mut [Complex64]| {
                y[0] = Complex64::new(1.0, 0.0);
                token.cancel();
                Ok(())
            },
            &[1.0],
            &[0.0],
            &mut output,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(report.evaluations(), 1);
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert_eq!(output, [99.0]);
    let token = CancellationToken::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(1, 1),
    ))
    .unwrap();
    let report = method
        .jvp_into(
            |_: &[f64], y: &mut [f64]| {
                y[0] = 1.0;
                token.cancel();
                Ok(())
            },
            &[1.0],
            &[0.0],
            None,
            None,
            &mut output,
            &mut w,
            &mut ctx,
        )
        .unwrap();
    assert_eq!(report.evaluations(), 1);
    assert_eq!(report.termination(), DifferentiationTermination::Cancelled);
    assert_eq!(output, [99.0]);
}

fn subnormal_geometry<S: RealScalar + NumericScalar<Real = S>>() {
    let smallest = S::MIN_POSITIVE * S::EPSILON;
    let offsets = [-smallest, S::ZERO, smallest];
    let p = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut w = p.workspace::<S>(()).unwrap();
    let mut out = [S::NAN];
    for point in [-S::ZERO, S::ZERO, smallest] {
        p.evaluate_into(
            |x: &[S], y: &mut [S]| {
                y[0] = x[0];
                Ok(())
            },
            &[point],
            None,
            &mut out,
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(out, [S::ONE]);
    }
}
#[test]
fn relative_subnormal_geometry_f32() {
    subnormal_geometry::<f32>();
}
#[test]
fn relative_subnormal_geometry_f64() {
    subnormal_geometry::<f64>();
}

#[test]
fn malformed_per_call_inputs_fail_without_allocating_or_invoking_callbacks() {
    let offsets = [-0.25, 0.0, 0.25];
    let p = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut w = p.workspace::<f64>(()).unwrap();
    let mut output = [99.0];
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    for point in [f64::NAN, f64::INFINITY, 2.0] {
        let error = p
            .evaluate_into(
                |_: &[f64], _: &mut [_]| panic!("invalid input callback"),
                &[point],
                Some(&bounds),
                &mut output,
                &mut w,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err();
        if point.is_finite() {
            assert_eq!(
                invalid_input(&error),
                (
                    scioxide_core::argument!("x"),
                    InvalidInputReason::OutOfRange
                )
            );
        } else {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
        assert_eq!(output, [99.0]);
    }
    let p = ComplexStep::default().with_step(1e-100);
    let mut w =
        ComplexStepWorkspace::workspace(DerivativeRequest::new(&p, DerivativeShape::gradient(1)))
            .unwrap();
    {
        assert!(
            p.gradient_into(
                |_: &[Complex64], _: &mut [Complex64]| panic!("invalid complex input callback"),
                &[f64::NAN],
                &mut output,
                &mut w,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
    };
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(1, 1),
    ))
    .unwrap();
    for (point, direction, base) in [
        ([f64::NAN], [1.0], None),
        ([1.0], [f64::NAN], None),
        ([1.0], [1.0], Some([f64::NAN])),
    ] {
        {
            assert!(
                method
                    .jvp_into(
                        |_: &[f64], _: &mut [f64]| panic!("invalid fixed input callback"),
                        &point,
                        &direction,
                        None,
                        base.as_ref().map(|v| &v[..]),
                        &mut output,
                        &mut w,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .is_err()
            );
        };
        assert_eq!(output, [99.0]);
    }
}

#[test]
fn empty_directional_validate_base_is_shared_across_families() {
    let mut calls = 0;
    let result = ComplexStep::default()
        .with_step(1e-100)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
        .jvp(
            |_: &[Complex64], y: &mut [Complex64]| {
                assert!(y.is_empty());
                calls += 1;
                Ok(())
            },
            &[],
            &[],
            0,
        )
        .unwrap();
    assert_eq!(calls, 0);
    assert_eq!(result.report().evaluations(), 0);
    assert_eq!(result.values(), Some(&[][..]));
}

#[test]
fn strided_publication_polls_after_numerical_staging() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct PublicationClock {
        calls: AtomicUsize,
        expires: usize,
    }
    impl Clock for PublicationClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            let calls = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(if calls >= self.expires { 2 } else { 0 }),
            ))
        }
    }
    let run = |expires, colored| {
        let clock = PublicationClock {
            calls: AtomicUsize::new(0),
            expires,
        };
        let binding = ClockBinding::new(&clock);
        let deadline = Deadline::at(
            &binding,
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
        let layout = StridedLayout::new(&[1], &[1]).unwrap();
        let mut output = [99.0];
        let mut errors = [88.0];
        let packed: Vec<_> = NdView::new(&[1.0_f64], layout)
            .unwrap()
            .iter()
            .copied()
            .collect();
        let report = if colored {
            let schedule = ColoringSchedule::new(
                StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
                ColoringKind::DerivativeColumns,
                vec![0],
            )
            .unwrap();
            let p =
                ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default()).unwrap();
            let mut w = p.workspace(()).unwrap();
            p.evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &packed,
                None,
                &mut output,
                &mut errors,
                &mut w,
                &mut ctx,
            )
            .unwrap()
        } else {
            let p = AdaptiveDifference::<f64>::default();
            let mut w = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
                &p,
                DerivativeShape::gradient(1),
            ))
            .unwrap();
            p.gradient_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &packed,
                None,
                &mut output,
                &mut errors,
                &mut w,
                &mut ctx,
            )
            .unwrap()
        };
        (clock.calls.load(Ordering::Relaxed), report, output, errors)
    };
    for colored in [false, true] {
        let (calls, report, output, _) = run(usize::MAX, colored);
        assert!(report.output_available());
        assert!((output[0] - 2.0).abs() < 1e-10);
        let (observed, report, output, errors) = run(calls, colored);
        assert_eq!(observed, calls);
        assert_eq!(report.completed_entries(), 1);
        assert_eq!(
            report.termination(),
            DifferentiationTermination::DeadlineReached
        );
        assert!(!report.output_available());
        assert_eq!(output, [99.0]);
        assert_eq!(errors, [88.0]);
    }
}

fn asymmetric_mixed_moments<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let v = |x| S::try_from_f64_lossy(x).unwrap();
    let epsilon = S::EPSILON;
    let ax = [-2.1, -0.8, 0.0, 0.6, 1.6].map(|x| v(x) * epsilon);
    let ay = [-1.7, -0.6, 0.0, 0.8, 2.1].map(|x| v(x) * epsilon);
    let real = RelativeDifference::new(
        DerivativeOperation::Hessian,
        &[&ax, &ay],
        1,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    let complex = RelativeComplexStep::new(
        &[&ax, &ay],
        ComplexStep::default().with_step(epsilon * epsilon),
    )
    .unwrap();
    let mut rw = real.workspace::<Complex<S>>(()).unwrap();
    let mut cw = complex.workspace(()).unwrap();
    for point in [
        [S::ONE, S::ONE.next_after(S::ZERO)],
        [S::ONE.next_after(v(2.0)), S::ONE],
    ] {
        let nx = ax.map(|offset| (point[0] + offset).into_f64());
        let ny = ay.map(|offset| (point[1] + offset).into_f64());
        let wx = reference_weights(&nx, point[0].into_f64(), epsilon.into_f64(), 1);
        let wy = reference_weights(&ny, point[1].into_f64(), epsilon.into_f64(), 1);
        for p in 0..5 {
            for q in 0..5 {
                let mut reference = 0.0;
                let mut condition = 0.0;
                for (i, &x) in nx.iter().enumerate() {
                    for (j, &y) in ny.iter().enumerate() {
                        let term = wx[i]
                            * wy[j]
                            * ((x - point[0].into_f64()) / epsilon.into_f64()).powi(p)
                            * ((y - point[1].into_f64()) / epsilon.into_f64()).powi(q);
                        reference += term;
                        condition += term.abs();
                    }
                }
                let expected = if p == 1 && q == 1 { 1.0 } else { 0.0 };
                assert!((reference - expected).abs() <= 64.0 * f64::EPSILON * condition.max(1.0));
                let tolerance = 64.0 * epsilon.into_f64() * condition.max(1.0);
                let mut ro = [Complex::new(S::NAN, S::NAN); 4];
                let mut co = [S::NAN; 4];
                real.evaluate_into(
                    |x: &[S], y: &mut [Complex<S>]| {
                        assert!(nx.contains(&x[0].into_f64()));
                        assert!(ny.contains(&x[1].into_f64()));
                        let value = ((x[0] - point[0]) / epsilon).powi(p)
                            * ((x[1] - point[1]) / epsilon).powi(q);
                        y[0] = Complex::new(value, -value);
                        Ok(())
                    },
                    &point,
                    None,
                    &mut ro,
                    &mut rw,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
                complex
                    .evaluate_into(
                        |x: &[Complex<S>], y: &mut [Complex<S>]| {
                            assert!(nx.contains(&x[0].real_component().into_f64()));
                            assert!(ny.contains(&x[1].real_component().into_f64()));
                            let a = (x[0] - Complex::new(point[0], S::ZERO))
                                / Complex::new(epsilon, S::ZERO);
                            let b = (x[1] - Complex::new(point[1], S::ZERO))
                                / Complex::new(epsilon, S::ZERO);
                            y[0] = a.powi(p) * b.powi(q);
                            Ok(())
                        },
                        &point,
                        None,
                        &mut co,
                        &mut cw,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                for index in [1, 2] {
                    let real = ro[index].real_component().into_f64() * epsilon.into_f64().powi(2);
                    let imag =
                        ro[index].imaginary_component().into_f64() * epsilon.into_f64().powi(2);
                    let complex = co[index].into_f64() * epsilon.into_f64().powi(2);
                    assert!(
                        (real - expected).abs() <= tolerance,
                        "real ({p},{q}): {real}"
                    );
                    assert!((imag + expected).abs() <= tolerance);
                    assert!(
                        (complex - expected).abs() <= tolerance,
                        "complex ({p},{q}): {complex}"
                    );
                }
            }
        }
    }
}
#[test]
fn asymmetric_mixed_actual_nodes_f32() {
    asymmetric_mixed_moments::<f32>();
}
#[test]
fn asymmetric_mixed_actual_nodes_f64() {
    asymmetric_mixed_moments::<f64>();
}

#[test]
fn complex_jvp_requirements_are_linear_and_allocation_free() {
    let method = ComplexStep::<f64>::default()
        .with_step(1e-100)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    for inputs in [0, 1, 4096] {
        let required = ComplexStepWorkspace::workspace_requirements(DerivativeRequest::new(
            &method,
            DerivativeShape::jvp(inputs, 1),
        ))
        .map(|required| required.bytes())
        .unwrap();
        assert_eq!(
            required,
            inputs * core::mem::size_of::<Complex64>()
                + core::mem::size_of::<Complex64>()
                + core::mem::size_of::<f64>()
        );
    }
}

fn rounded_directional_collision<S: RealScalar + NumericScalar<Real = S>>() {
    let x = S::ONE + S::EPSILON;
    let upper = S::ONE + S::EPSILON + S::EPSILON;
    let bounds = Bounds::new(vec![(Some(x), Some(upper))]).unwrap();
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(1, 1),
    ))
    .unwrap();
    let mut output = [S::try_from_f64_lossy(99.0).unwrap()];
    let error = method
        .jvp_into(
            |_: &[S], _: &mut [S]| panic!("callback on coincident directional nodes"),
            &[x],
            &[S::ONE],
            Some(&bounds),
            None,
            &mut output,
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
    let scioxide_core::SciOxideErrorView::DifferentiationFailure {
        frames: _failure_frames,
        cause: _failure_cause,
    } = error.view()
    else {
        unreachable!("unexpected error kind")
    };
    assert_eq!(
        _failure_frames[0].location.site(),
        DifferentiationSite::Direction
    );
    assert_eq!(
        match _failure_cause.view() {
            scioxide_core::SciOxideErrorView::ConvergenceFailure { reason, .. } => Some(reason),
            _ => None,
        }
        .unwrap(),
        ConvergenceReason::StepSizeLimit
    );
    assert_eq!(output[0], S::try_from_f64_lossy(99.0).unwrap());
    let report = method
        .jvp_into(
            |x: &[S], y: &mut [S]| {
                y[0] = x[0];
                Ok(())
            },
            &[S::ONE],
            &[S::ONE],
            None,
            None,
            &mut output,
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.output_available());
    assert!((output[0] - S::ONE).abs() <= S::SQRT_EPSILON);
}
#[test]
fn fixed_directional_actual_nodes_remain_distinct_f32() {
    rounded_directional_collision::<f32>();
}
#[test]
fn fixed_directional_actual_nodes_remain_distinct_f64() {
    rounded_directional_collision::<f64>();
}

#[test]
fn feasible_fixed_coordinates_can_change_when_reusing_workspace() {
    let method =
        FiniteDifference::default().with_fixed_coordinates(FixedCoordinatePolicy::FeasibleZero);
    let mut workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    for fixed in [0, 1, 0] {
        let mut intervals = vec![(None, None); 2];
        intervals[fixed] = (Some(1.0), Some(1.0));
        let bounds = Bounds::new(intervals).unwrap();
        let mut output = [99.0; 2];
        let report = method
            .gradient_into(
                |x: &[f64], y: &mut [f64]| {
                    assert_eq!(x[fixed], 1.0);
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    Ok(())
                },
                &[1.0, 1.0],
                Some(&bounds),
                Some(2.0),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(output[fixed], 0.0);
        assert!((output[1 - fixed] - 2.0).abs() < 1e-8);
    }
}

// Check the optimized per-sample ledger through the public gradient API. The
// callback nodes and polynomial moments are computed independently of stencils.
fn fixed_gradient_sample_admission<S: RealScalar + NumericScalar<Real = S>>() {
    let point = [S::ONE, -S::ONE];
    let step = S::try_from_f64_lossy(0.625).unwrap() * S::EPSILON;
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        let method = FiniteDifference::default()
            .with_scheme(scheme)
            .with_step(DifferenceStep::absolute(step).unwrap());
        let mut workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
            &method,
            DerivativeShape::gradient(2),
        ))
        .unwrap();
        for cached in [false, true] {
            let mut expected = [point; 5];
            let mut count = usize::from(!cached);
            for (variable, &coordinate) in point.iter().enumerate() {
                if scheme == DifferenceScheme::Central {
                    expected[count][variable] = coordinate - step;
                    count += 1;
                }
                expected[count][variable] = coordinate + step;
                count += 1;
            }
            // Every denied sample leaves the output intact; the next allowance
            // reuses the same workspace after its previous partial computation.
            for allowance in 0..=count {
                let mut state = SolveState::new();
                let mut budgets = SolveBudgets::unbounded().with_limit(
                    BudgetKind::Evaluation,
                    Budget::try_from(allowance.max(1)).unwrap(),
                );
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                    .unwrap();
                if allowance == 0 {
                    assert!(
                        context
                            .record_provider(EvaluationRole::Function)
                            .unwrap()
                            .is_granted()
                    );
                }
                let mut actual = [[S::NAN; 2]; 5];
                let mut calls = 0;
                let untouched = [S::try_from_f64_lossy(99.0).unwrap(); 2];
                let mut output = untouched;
                let report = method
                    .gradient_into(
                        |x: &[S], y: &mut [S]| {
                            actual[calls].copy_from_slice(x);
                            calls += 1;
                            let u = (x[0] - point[0]) / S::EPSILON;
                            let v = (x[1] - point[1]) / S::EPSILON;
                            y[0] = u
                                + S::TWO * v
                                + if scheme == DifferenceScheme::Central {
                                    u * u + v * v
                                } else {
                                    S::ZERO
                                };
                            Ok(())
                        },
                        &point,
                        None,
                        cached.then_some(S::ZERO),
                        &mut output,
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                assert_eq!(calls, allowance);
                assert_eq!(report.evaluations(), calls);
                assert_eq!(&actual[..calls], &expected[..calls]);
                assert_eq!(
                    context.counts().nested().provider(EvaluationRole::Function),
                    calls
                );
                assert_eq!(context.counts().nested().iterations(), report.iterations());
                assert_eq!(
                    context.budget_report(BudgetKind::Evaluation).consumed(),
                    calls + usize::from(allowance == 0)
                );
                if allowance < count {
                    assert_eq!(
                        report.termination(),
                        DifferentiationTermination::EvaluationLimit
                    );
                    assert!(!report.output_available());
                    assert_eq!(output, untouched);
                } else {
                    assert!(report.output_available());
                    assert_eq!(report.iterations(), 2);
                    for (value, derivative) in output.into_iter().zip([S::ONE, S::TWO]) {
                        assert!(
                            (value * S::EPSILON - derivative).abs()
                                <= S::try_from_f64_lossy(8.0).unwrap() * S::EPSILON
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_gradient_sample_admission_f32() {
    fixed_gradient_sample_admission::<f32>();
}

#[test]
fn fixed_gradient_sample_admission_f64() {
    fixed_gradient_sample_admission::<f64>();
}
