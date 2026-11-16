//! One-shot functions: accuracy, equality with their configured forms, empty
//! shapes, allocation and the errors of each parameter.
#![allow(clippy::unwrap_used, clippy::panic)]
use core::cell::Cell;
use scioxide_core::{
    ColoringKind, ColoringSchedule, Complex, ConvergenceReason, EvaluationRole, InvalidInputReason,
    NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, StructuralPattern, argument, quantity,
};
use scioxide_differentiate::*;

fn c<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}
fn close<S: RealScalar>(actual: S, expected: S, tolerance: S) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance * expected.abs().max(S::ONE),
        "{actual:?} != {expected:?} within {tolerance:?}"
    );
}
// Documented error scales: ε^(2/3) for first and √ε for second derivatives.
fn first_tolerance<S: RealScalar>() -> S {
    S::EPSILON.cbrt() * S::EPSILON.cbrt() * c(32.0)
}
fn second_tolerance<S: RealScalar>() -> S {
    S::SQRT_EPSILON * c(32.0)
}
fn cubic<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    Ok(x * x * x - c::<S>(2.0) * x)
}
fn field<S: RealScalar>(p: &[S]) -> Result<S, SciOxideError> {
    Ok(p[0] * p[0] * p[1] + p[1] * p[1] * p[1])
}
fn map<S: RealScalar>(p: &[S], y: &mut [S]) -> Result<(), SciOxideError> {
    y[0] = p[0] * p[1];
    y[1] = p[0] * p[0] - p[1];
    Ok(())
}
fn scalar_adapter<S: RealScalar>(
    f: impl Fn(&[S]) -> Result<S, SciOxideError>,
) -> impl Fn(&[S], &mut [S]) -> Result<(), SciOxideError> {
    move |p, y| {
        y[0] = f(p)?;
        Ok(())
    }
}
fn diagonal_schedule() -> ColoringSchedule {
    let pattern = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
    ColoringSchedule::new(pattern, ColoringKind::DerivativeColumns, vec![0, 0]).unwrap()
}

fn polynomial_accuracy<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let (x, p) = (c::<S>(0.7), [c::<S>(0.5), c::<S>(-1.2)]);
    let first = first_tolerance::<S>();
    let second = second_tolerance::<S>();
    close(
        derivative(cubic, x).unwrap(),
        c::<S>(3.0) * x * x - c(2.0),
        first,
    );
    close(
        second_derivative(cubic, x).unwrap(),
        c::<S>(6.0) * x,
        second,
    );
    let g = gradient(field, &p).unwrap();
    close(g[0], c::<S>(2.0) * p[0] * p[1], first);
    close(g[1], p[0] * p[0] + c::<S>(3.0) * p[1] * p[1], first);
    let h = hessian(field, &p).unwrap();
    let expected = [
        c::<S>(2.0) * p[1],
        c::<S>(2.0) * p[0],
        c::<S>(2.0) * p[0],
        c::<S>(6.0) * p[1],
    ];
    for (&value, expected) in h.iter().zip(expected) {
        close(value, expected, second);
    }
    assert_eq!(h[1], h[2]);
    let j = jacobian(map, &p, 2).unwrap();
    let expected = [p[1], p[0], c::<S>(2.0) * p[0], -S::ONE];
    for (&value, expected) in j.iter().zip(expected) {
        close(value, expected, first);
    }
    let v = [c::<S>(0.3), c::<S>(-0.5)];
    let jv = jvp(map, &p, &v, 2).unwrap();
    close(jv[0], p[1] * v[0] + p[0] * v[1], first);
    close(jv[1], c::<S>(2.0) * p[0] * v[0] - v[1], first);
    let schedule = diagonal_schedule();
    let sparse = sparse_jacobian(
        |q: &[S], y: &mut [S]| {
            y[0] = q[0] * q[0] * q[0];
            y[1] = q[1].sin();
            Ok(())
        },
        &p,
        &schedule,
    )
    .unwrap();
    close(
        sparse[0],
        c::<S>(3.0) * p[0] * p[0],
        S::SQRT_EPSILON * c(16.0),
    );
    close(sparse[1], p[1].cos(), S::SQRT_EPSILON * c(16.0));
    let nodes = [c::<S>(0.0), c::<S>(0.5), c::<S>(1.0)];
    let slope: S = physical_difference_derivative(|t: S| Ok(t * t), p[0], &nodes).unwrap();
    close(slope, S::ONE, S::EPSILON * c(8.0));
    let exact = complex_step_derivative(
        |z: Complex<S>| Ok(z * z * z - z * Complex::new(c(2.0), S::ZERO)),
        x,
    )
    .unwrap();
    close(exact, c::<S>(3.0) * x * x - c(2.0), S::EPSILON * c(8.0));
}

#[test]
fn polynomial_derivatives_meet_the_documented_accuracy_f32() {
    polynomial_accuracy::<f32>();
}
#[test]
fn polynomial_derivatives_meet_the_documented_accuracy_f64() {
    polynomial_accuracy::<f64>();
}

fn wraps<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    // The configured forms run the same kernel with the same arguments, so the
    // values agree exactly.
    let exact = |values: &[S]| values.to_vec();
    let (x, p) = (c::<S>(0.7), [c::<S>(0.5), c::<S>(-1.2)]);
    let fixed = FiniteDifference::<S>::default();
    let derivative_value = derivative(cubic, x).unwrap();
    assert_eq!(
        exact(&[derivative_value]),
        exact(&fixed.derivative(cubic, x).unwrap().into_values().unwrap())
    );
    // The stack and workspace paths run the same kernel.
    let workspace_path = fixed
        .gradient(scalar_adapter(|q: &[S]| cubic(q[0])), &[x], None, None)
        .unwrap()
        .into_values()
        .unwrap();
    assert_eq!(exact(&[derivative_value]), exact(&workspace_path));
    let curvature = second_derivative(cubic, x).unwrap();
    let hessian_path = fixed
        .hessian(scalar_adapter(|q: &[S]| cubic(q[0])), &[x], None, None)
        .unwrap()
        .into_values()
        .unwrap();
    assert_eq!(exact(&[curvature]), exact(&hessian_path));
    assert_eq!(
        exact(&gradient(field, &p).unwrap()),
        exact(
            &fixed
                .gradient(scalar_adapter(field), &p, None, None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    assert_eq!(
        exact(&hessian(field, &p).unwrap()),
        exact(
            &fixed
                .hessian(scalar_adapter(field), &p, None, None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    assert_eq!(
        exact(&jacobian(map, &p, 2).unwrap()),
        exact(
            &fixed
                .jacobian(map, &p, 2, None, None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    let v = [c::<S>(0.3), c::<S>(-0.5)];
    assert_eq!(
        exact(&jvp(map, &p, &v, 2).unwrap()),
        exact(
            &fixed
                .jvp(map, &p, &v, 2, None, None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    let schedule = diagonal_schedule();
    let diagonal = |q: &[S], y: &mut [S]| {
        y[0] = q[0].exp();
        y[1] = q[1] * q[1];
        Ok(())
    };
    let plan = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default()).unwrap();
    assert_eq!(
        exact(&sparse_jacobian(diagonal, &p, &schedule).unwrap()),
        exact(
            &plan
                .evaluate(diagonal, &p, None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    let nodes = [c::<S>(0.25), c::<S>(0.5), c::<S>(1.0)];
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[p[0]],
        &[&nodes],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let node_slope: S = physical_difference_derivative(|t: S| Ok(t.exp()), p[0], &nodes).unwrap();
    assert_eq!(
        exact(&[node_slope]),
        exact(
            &physical
                .evaluate(scalar_adapter(|q: &[S]| Ok(q[0].exp())), None)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
    let extension = |z: Complex<S>| Ok(z.exp());
    assert_eq!(
        exact(&[complex_step_derivative(extension, x).unwrap()]),
        exact(
            &ComplexStep::default()
                .derivative(extension, x)
                .unwrap()
                .into_values()
                .unwrap()
        )
    );
}

#[test]
fn one_shots_equal_their_configured_forms_bit_for_bit_f32() {
    wraps::<f32>();
}
#[test]
fn one_shots_equal_their_configured_forms_bit_for_bit_f64() {
    wraps::<f64>();
}

#[test]
fn derivative_allocates_nothing() {
    let value = derivative(|x: f64| Ok(x.sin()), 1.0).unwrap();
    close(value, 1.0_f64.cos(), first_tolerance());
}

#[test]
fn empty_shapes_return_empty_vectors_without_calls() {
    let calls = Cell::new(0);
    let counted = |p: &[f64]| {
        calls.set(calls.get() + 1);
        field(p)
    };
    let counted_map = |p: &[f64], y: &mut [f64]| {
        calls.set(calls.get() + 1);
        y.fill(p.iter().sum());
        Ok(())
    };
    assert!(gradient(counted, &[]).unwrap().is_empty());
    assert!(hessian(counted, &[]).unwrap().is_empty());
    assert!(jacobian(counted_map, &[], 3).unwrap().is_empty());
    assert!(jacobian(counted_map, &[1.0, 2.0], 0).unwrap().is_empty());
    assert!(
        jvp(counted_map, &[1.0, 2.0], &[1.0, -1.0], 0)
            .unwrap()
            .is_empty()
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn evaluation_counts_follow_the_documented_costs() {
    let calls = Cell::new(0);
    let counted = |p: &[f64]| {
        calls.set(calls.get() + 1);
        Ok(p.iter().map(|v| v.sin()).sum::<f64>())
    };
    let x = [0.3, -0.4, 0.5];
    let n = x.len();
    let _ = gradient(counted, &x).unwrap();
    assert_eq!(calls.replace(0), 2 * n + 1);
    let _ = hessian(counted, &x).unwrap();
    assert_eq!(calls.replace(0), 1 + 2 * n + 2 * n * (n - 1));
    let forward = FiniteDifference::default().with_scheme(DifferenceScheme::Forward);
    let _ = forward
        .hessian(scalar_adapter(counted), &x, None, None)
        .unwrap();
    assert_eq!(calls.replace(0), 1 + 2 * n + n * (n - 1) / 2);
    let _ = derivative(|t: f64| counted(&[t]), 0.2).unwrap();
    assert_eq!(calls.replace(0), 3);
    let _ = second_derivative(|t: f64| counted(&[t]), 0.2).unwrap();
    assert_eq!(calls.replace(0), 3);
    let counted_map = |p: &[f64], y: &mut [f64]| {
        calls.set(calls.get() + 1);
        map(p, y)
    };
    let _ = jvp(counted_map, &[0.5, 1.5], &[1.0, 2.0], 2).unwrap();
    assert_eq!(calls.replace(0), 3);
    let counted_extension = |z: Complex<f64>| {
        calls.set(calls.get() + 1);
        Ok(z * z)
    };
    let _ = complex_step_derivative(counted_extension, 0.5).unwrap();
    assert_eq!(calls.replace(0), 2);
}

fn invalid(error: &SciOxideError, name: scioxide_core::Argument, expected: InvalidInputReason) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason, .. }
            if argument == name && reason == expected),
        "{error:?}"
    );
}
fn non_finite_input(error: &SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}
fn caused_by(error: &SciOxideError, check: impl Fn(&SciOxideError) -> bool) {
    assert_eq!(error.kind(), SciOxideErrorKind::DifferentiationFailure);
    assert!(
        matches!(error.view(), SciOxideErrorView::DifferentiationFailure { cause, .. } if check(cause)),
        "{error:?}"
    );
}

#[test]
fn input_errors_name_the_parameters_of_each_one_shot() {
    non_finite_input(&derivative(cubic, f64::NAN).unwrap_err());
    non_finite_input(&second_derivative(cubic, f64::INFINITY).unwrap_err());
    non_finite_input(&gradient(field, &[1.0, f64::NAN]).unwrap_err());
    non_finite_input(&hessian(field, &[f64::NAN, 1.0]).unwrap_err());
    non_finite_input(&jacobian(map, &[1.0, f64::NAN], 2).unwrap_err());
    non_finite_input(&jvp(map, &[1.0, 2.0], &[f64::NAN, 1.0], 2).unwrap_err());
    non_finite_input(&complex_step_derivative(|z: Complex<f64>| Ok(z), f64::NAN).unwrap_err());
    let error = jvp(map, &[1.0, 2.0], &[1.0], 2).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
        if expected == [2] && actual == [1])
    );
    let error = jacobian(map, &[1.0, 2.0], usize::MAX).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::SizeOverflow { context }
        if context == quantity!("derivative_entries"))
    );
    let error = sparse_jacobian(map, &[1.0], &diagonal_schedule()).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
        if expected == [2] && actual == [1])
    );
    let pattern = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
    let distance_one =
        ColoringSchedule::new(pattern, ColoringKind::DistanceOne, vec![0, 0]).unwrap();
    let error = sparse_jacobian(map, &[1.0, 2.0], &distance_one).unwrap_err();
    invalid(
        &error,
        argument!("schedule"),
        InvalidInputReason::UnsupportedConfiguration,
    );
    let slope = |t: f64| Ok(t * t);
    let error = physical_difference_derivative(slope, 0.5, &[0.5]).unwrap_err();
    invalid(&error, argument!("nodes"), InvalidInputReason::OutOfRange);
    let error = physical_difference_derivative(slope, 0.5, &[0.0, 1.0, 0.0]).unwrap_err();
    invalid(&error, argument!("nodes"), InvalidInputReason::Inconsistent);
    non_finite_input(&physical_difference_derivative(slope, 0.5, &[0.0, f64::NAN]).unwrap_err());
    let error =
        complex_step_derivative(|z: Complex<f64>| Ok(z + Complex::new(0.0, 1.0)), 1.0).unwrap_err();
    invalid(
        &error,
        argument!("function"),
        InvalidInputReason::UnsupportedConfiguration,
    );
}

#[test]
fn callback_errors_and_non_finite_values_are_differentiation_failures() {
    let injected = || {
        SciOxideError::invalid_input(
            argument!("injected"),
            InvalidInputReason::OutOfRange,
            "injected callback failure",
        )
    };
    let is_injected = |cause: &SciOxideError| {
        matches!(cause.view(), SciOxideErrorView::InvalidInput { argument, .. }
            if argument == argument!("injected"))
    };
    let is_callback_nan = |cause: &SciOxideError| {
        matches!(
            cause.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        )
    };
    caused_by(
        &derivative(|_: f64| Err(injected()), 1.0).unwrap_err(),
        is_injected,
    );
    caused_by(
        &derivative(|_: f64| Ok(f64::NAN), 1.0).unwrap_err(),
        is_callback_nan,
    );
    caused_by(
        &second_derivative(|_: f64| Err(injected()), 1.0).unwrap_err(),
        is_injected,
    );
    caused_by(
        &gradient(|_: &[f64]| Ok(f64::INFINITY), &[1.0, 2.0]).unwrap_err(),
        is_callback_nan,
    );
    caused_by(
        &hessian(|_: &[f64]| Err(injected()), &[1.0, 2.0]).unwrap_err(),
        is_injected,
    );
    let failing_map = |_: &[f64], _: &mut [f64]| Err(injected());
    caused_by(&jacobian(failing_map, &[1.0], 1).unwrap_err(), is_injected);
    caused_by(
        &jvp(failing_map, &[1.0], &[1.0], 1).unwrap_err(),
        is_injected,
    );
    let nan_map = |_: &[f64], y: &mut [f64]| {
        y.fill(f64::NAN);
        Ok(())
    };
    caused_by(
        &sparse_jacobian(nan_map, &[1.0, 2.0], &diagonal_schedule()).unwrap_err(),
        is_callback_nan,
    );
    caused_by(
        &physical_difference_derivative(|_: f64| Err::<f64, _>(injected()), 0.5, &[0.0, 1.0])
            .unwrap_err(),
        is_injected,
    );
    caused_by(
        &complex_step_derivative(|_: Complex<f64>| Err(injected()), 1.0).unwrap_err(),
        is_injected,
    );
}

#[test]
fn sparse_jacobian_maps_an_unconverged_column_to_a_convergence_failure() {
    // A jump at x₀ = 0: the central differences grow like 1/h and never converge.
    let jump = |x: &[f64], y: &mut [f64]| {
        y[0] = if x[0] > 0.0 { 1.0 } else { 0.0 };
        y[1] = x[1] * x[1];
        Ok(())
    };
    let schedule = diagonal_schedule();
    let x = [0.0, 1.0];
    let report = *ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default())
        .unwrap()
        .evaluate(jump, &x, None)
        .unwrap()
        .report();
    assert!(!report.is_success());
    let expected = report.require_success().unwrap_err();
    let error = sparse_jacobian(jump, &x, &schedule).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    let reason_and_count = |error: &SciOxideError| match error.view() {
        SciOxideErrorView::ConvergenceFailure {
            reason,
            diagnostics,
            ..
        } => (
            reason,
            diagnostics
                .counts()
                .own()
                .provider(EvaluationRole::Function),
        ),
        _ => panic!("{error:?}"),
    };
    let (reason, count) = reason_and_count(&error);
    assert_eq!((reason, count), reason_and_count(&expected));
    assert_eq!(count, report.evaluations());
    assert_ne!(reason, ConvergenceReason::EvaluationLimit);
}
