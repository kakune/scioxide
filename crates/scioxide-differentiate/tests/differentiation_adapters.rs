//! API equivalence is separate from independent polynomial derivative checks.
#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::too_many_lines)]
use core::cell::Cell;
use core::error::Error as _;
use scioxide_core::{
    Bounds, Budget, BudgetKind, ColoringKind, ColoringSchedule, Complex, Complex64, DomainReason,
    ExecutionContext, ExtendedInterval, NumericScalar, ProblemToken, RealScalar, SciOxideError,
    SciOxideErrorView, SolveBudgets, SolveIdentity, SolveState, StructuralPattern,
    StructuralPatternView, algorithm_id, argument,
};
use scioxide_differentiate::*;

fn bits<S: RealScalar>(a: &[S], b: &[S]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert_eq!(
            a.try_to_f64_exact().unwrap().to_bits(),
            b.try_to_f64_exact().unwrap().to_bits()
        );
    }
}
fn exact<S: RealScalar>(actual: &[S], expected: &[S]) {
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            (a - b).abs()
                <= S::SQRT_EPSILON * S::try_from_f64_lossy(32.0).unwrap() * b.abs().max(S::ONE),
            "actual={}, expected={}",
            a.try_to_f64_exact().unwrap(),
            b.try_to_f64_exact().unwrap()
        );
    }
    assert_eq!(actual.len(), expected.len());
}
fn same_report(a: &DifferentiationReport, b: &DifferentiationReport) {
    assert_eq!(a.source(), b.source());
    assert_eq!(a.termination(), b.termination());
    assert_eq!(a.evaluations(), b.evaluations());
    assert_eq!(a.iterations(), b.iterations());
    assert_eq!(a.completed_entries(), b.completed_entries());
    assert_eq!(a.output_available(), b.output_available());
}
fn prepared_equal<S: RealScalar>(a: &DifferentiationResult<S>, b: &DifferentiationResult<S>) {
    bits(a.values().unwrap(), b.values().unwrap());
    same_report(a.report(), b.report());
}
fn adaptive_equal<S: RealScalar + NumericScalar<Real = S>>(
    a: &DifferentiationResult<S>,
    b: &DifferentiationResult<S>,
) {
    prepared_equal(a, b);
    bits(a.errors().unwrap(), b.errors().unwrap());
}
// Compares an owned result with the output and report of the `_into` form.
fn into_equal<S: RealScalar>(
    owned: &DifferentiationResult<S>,
    report: &DifferentiationReport,
    out: &[S],
) {
    bits(owned.values().unwrap(), out);
    same_report(owned.report(), report);
}
fn adaptive_into_equal<S: RealScalar + NumericScalar<Real = S>>(
    owned: &DifferentiationResult<S>,
    report: &DifferentiationReport,
    out: &[S],
    errors: &[S],
) {
    into_equal(owned, report, out);
    bits(owned.errors().unwrap(), errors);
}
fn scalar<S: RealScalar>(x: &[S]) -> Result<S, SciOxideError> {
    Ok(x[0] * x[0] + x[0] * x[1] + (S::TWO) * x[1] * x[1])
}
fn sample<S: RealScalar>(x: &[S], y: &mut [S]) -> Result<(), SciOxideError> {
    y[0] = scalar(x)?;
    Ok(())
}
fn vector<S: RealScalar>(x: &[S], y: &mut [S]) -> Result<(), SciOxideError> {
    y[0] = scalar(x)?;
    y[1] = x[0] + x[1];
    Ok(())
}
fn cubic<S: RealScalar>(x: &[S], y: &mut [S]) -> Result<(), SciOxideError> {
    y[0] = x[0] * x[0] * x[0];
    Ok(())
}

fn fixed<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let point = [c(0.25), c(0.75)];
    let direction = [c(1.0), c(-2.0)];
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        for (selection, step) in [
            DifferenceStep::default(),
            DifferenceStep::absolute(c(0.03125)).unwrap(),
            DifferenceStep::relative(c(0.03125)).unwrap(),
            DifferenceStep::absolute_coordinates(&[c(0.03125), c(0.0625)]).unwrap(),
            DifferenceStep::relative_coordinates(&[c(0.03125), c(0.0625)]).unwrap(),
        ]
        .into_iter()
        .enumerate()
        {
            let method = FiniteDifference::default()
                .with_scheme(scheme)
                .with_step(step);
            let workspace = |shape| {
                FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape))
                    .unwrap()
            };
            let owned = method.gradient(sample, &point, None, None).unwrap();
            let mut output = [c(99.0); 2];
            let report = method
                .gradient_into(
                    sample,
                    &point,
                    None,
                    None,
                    &mut output,
                    &mut workspace(DerivativeShape::gradient(2)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            into_equal(&owned, &report, &output);
            let owned = method.jacobian(vector, &point, 2, None, None).unwrap();
            let mut output = [c(99.0); 4];
            let report = method
                .jacobian_into(
                    vector,
                    &point,
                    2,
                    None,
                    None,
                    &mut output,
                    &mut workspace(DerivativeShape::jacobian(2, 2)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            into_equal(&owned, &report, &output);
            let owned = method.hessian(sample, &point, None, None).unwrap();
            let mut output = [c(99.0); 4];
            let report = method
                .hessian_into(
                    sample,
                    &point,
                    None,
                    None,
                    &mut output,
                    &mut workspace(DerivativeShape::hessian(2)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            into_equal(&owned, &report, &output);
            if scheme == DifferenceScheme::Forward && selection == 0 {
                // A quadratic has no truncation error. Forward second differences amplify
                // sample rounding by h^-2; the default h is epsilon^(1/3), not epsilon^(1/4).
                let h = S::CBRT_EPSILON;
                let rounding_bound =
                    c(8.0) * S::EPSILON / h / h * (scalar(&point).unwrap().abs() + c(1.0));
                for (actual, expected) in output.iter().zip([c(2.0), c(1.0), c(1.0), c(4.0)]) {
                    assert!((*actual - expected).abs() <= rounding_bound);
                }
            } else {
                exact(&output, &[c(2.0), c(1.0), c(1.0), c(4.0)]);
            }
            let owned = method
                .jvp(vector, &point, &direction, 2, None, None)
                .unwrap();
            let mut output = [c(99.0); 2];
            let report = method
                .jvp_into(
                    vector,
                    &point,
                    &direction,
                    None,
                    None,
                    &mut output,
                    &mut workspace(DerivativeShape::jvp(2, 2)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            into_equal(&owned, &report, &output);
        }
    }
    // The scalar forms are the one-coordinate gradient and Hessian.
    let x = c(0.25);
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        for step in [
            DifferenceStep::default(),
            DifferenceStep::absolute(c(0.03125)).unwrap(),
            DifferenceStep::relative(c(0.03125)).unwrap(),
            DifferenceStep::absolute_coordinates(&[c(0.03125)]).unwrap(),
            DifferenceStep::relative_coordinates(&[c(0.03125)]).unwrap(),
        ] {
            let method = FiniteDifference::default()
                .with_scheme(scheme)
                .with_step(step);
            let owned = method.derivative(|t: S| Ok(t * t * t), x).unwrap();
            prepared_equal(&owned, &method.gradient(cubic, &[x], None, None).unwrap());
            let mut value = c(99.0);
            let report = method
                .derivative_into(
                    |t: S| Ok(t * t * t),
                    x,
                    ExtendedInterval::entire(),
                    None,
                    &mut value,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            into_equal(&owned, &report, &[value]);
            prepared_equal(
                &method.second_derivative(|t: S| Ok(t * t * t), x).unwrap(),
                &method.hessian(cubic, &[x], None, None).unwrap(),
            );
        }
    }
    let method = FiniteDifference::default();
    exact(
        method
            .gradient(sample, &point, None, None)
            .unwrap()
            .values()
            .unwrap(),
        &[c(1.25), c(3.25)],
    );
    exact(
        method
            .jacobian(vector, &point, 2, None, None)
            .unwrap()
            .values()
            .unwrap(),
        &[c(1.25), c(3.25), c(1.0), c(1.0)],
    );
    exact(
        method
            .hessian(sample, &point, None, None)
            .unwrap()
            .values()
            .unwrap(),
        &[c(2.0), c(1.0), c(1.0), c(4.0)],
    );
    exact(
        method
            .jvp(vector, &point, &direction, 2, None, None)
            .unwrap()
            .values()
            .unwrap(),
        &[c(-5.25), c(-1.0)],
    );
}

fn adaptive<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let point = [c(0.25), c(0.75)];
    let direction = [c(1.0), c(-2.0)];
    let steps = DifferenceStep::absolute_coordinates(&[c(0.5), c(0.25)]).unwrap();
    for method in [
        AdaptiveDifference::default(),
        AdaptiveDifference::default()
            .with_method(AdaptiveDifferenceMethod::Richardson)
            .with_step(steps.clone()),
        AdaptiveDifference::default()
            .with_method(AdaptiveDifferenceMethod::Ridders)
            .with_step(steps),
    ] {
        let workspace = |shape| {
            AdaptiveDifferenceWorkspace::<S>::workspace(DerivativeRequest::new(&method, shape))
                .unwrap()
        };
        let owned = method.gradient(sample, &point, None).unwrap();
        let (mut output, mut errors) = ([c(99.0); 2], [c(99.0); 2]);
        let report = method
            .gradient_into(
                sample,
                &point,
                None,
                &mut output,
                &mut errors,
                &mut workspace(DerivativeShape::gradient(2)),
                ExecutionContext::sequential(),
            )
            .unwrap();
        adaptive_into_equal(&owned, &report, &output, &errors);
        exact(owned.values().unwrap(), &[c(1.25), c(3.25)]);
        let owned = method.jacobian(vector, &point, 2, None).unwrap();
        let (mut output, mut errors) = ([c(99.0); 4], [c(99.0); 4]);
        let report = method
            .jacobian_into(
                vector,
                &point,
                2,
                None,
                &mut output,
                &mut errors,
                &mut workspace(DerivativeShape::jacobian(2, 2)),
                ExecutionContext::sequential(),
            )
            .unwrap();
        adaptive_into_equal(&owned, &report, &output, &errors);
        exact(owned.values().unwrap(), &[c(1.25), c(3.25), c(1.0), c(1.0)]);
        let owned = method.hessian(sample, &point, None).unwrap();
        let (mut output, mut errors) = ([c(99.0); 4], [c(99.0); 4]);
        let report = method
            .hessian_into(
                sample,
                &point,
                None,
                &mut output,
                &mut errors,
                &mut workspace(DerivativeShape::hessian(2)),
                ExecutionContext::sequential(),
            )
            .unwrap();
        adaptive_into_equal(&owned, &report, &output, &errors);
        exact(owned.values().unwrap(), &[c(2.0), c(1.0), c(1.0), c(4.0)]);
        let owned = method.jvp(vector, &point, &direction, 2, None).unwrap();
        let (mut output, mut errors) = ([c(99.0); 2], [c(99.0); 2]);
        let report = method
            .jvp_into(
                vector,
                &point,
                &direction,
                None,
                &mut output,
                &mut errors,
                &mut workspace(DerivativeShape::jvp(2, 2)),
                ExecutionContext::sequential(),
            )
            .unwrap();
        adaptive_into_equal(&owned, &report, &output, &errors);
        adaptive_equal(
            &owned,
            &method
                .clone()
                .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
                .jvp(vector, &point, &direction, 2, None)
                .unwrap(),
        );
        exact(owned.values().unwrap(), &[c(-5.25), c(-1.0)]);
    }
    // The scalar forms are the one-coordinate gradient and Hessian.
    let x = c(0.25);
    let method = AdaptiveDifference::default();
    let workspace = |shape| {
        AdaptiveDifferenceWorkspace::<S>::workspace(DerivativeRequest::new(&method, shape)).unwrap()
    };
    let owned = method.derivative(|t: S| Ok(t * t * t), x).unwrap();
    let (mut output, mut errors) = ([c(99.0)], [c(99.0)]);
    let report = method
        .gradient_into(
            cubic,
            &[x],
            None,
            &mut output,
            &mut errors,
            &mut workspace(DerivativeShape::gradient(1)),
            ExecutionContext::sequential(),
        )
        .unwrap();
    adaptive_into_equal(&owned, &report, &output, &errors);
    exact(owned.values().unwrap(), &[c(0.1875)]);
    let owned = method.second_derivative(|t: S| Ok(t * t * t), x).unwrap();
    let (mut output, mut errors) = ([c(99.0)], [c(99.0)]);
    let report = method
        .hessian_into(
            cubic,
            &[x],
            None,
            &mut output,
            &mut errors,
            &mut workspace(DerivativeShape::hessian(1)),
            ExecutionContext::sequential(),
        )
        .unwrap();
    adaptive_into_equal(&owned, &report, &output, &errors);
    exact(owned.values().unwrap(), &[c(1.5)]);
}

fn geometry<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let point = [c(0.25), c(0.75)];
    let offsets = [c(-0.25), c(-0.125), c(0.0), c(0.375), c(0.5)];
    let axes = [offsets.as_slice(); 2];
    let nodes0 = offsets.map(|o| point[0] + o);
    let nodes1 = offsets.map(|o| point[1] + o);
    let nodes = [nodes0.as_slice(), nodes1.as_slice()];
    for (op, n) in [
        (DerivativeOperation::Gradient, 2),
        (DerivativeOperation::Jacobian, 2),
        (DerivativeOperation::Hessian, 2),
        (DerivativeOperation::First, 1),
        (DerivativeOperation::Second, 1),
    ] {
        let outputs = if op == DerivativeOperation::Jacobian {
            2
        } else {
            1
        };
        let callback = |x: &[S], y: &mut [S]| -> Result<(), SciOxideError> {
            match op {
                DerivativeOperation::Jacobian => vector(x, y),
                DerivativeOperation::First | DerivativeOperation::Second => cubic(x, y),
                _ => sample(x, y),
            }
        };
        let physical = PhysicalDifferencePlan::new(
            op,
            &point[..n],
            &nodes[..n],
            outputs,
            NodeDifferenceOptions::default(),
        )
        .unwrap();
        let owned = physical.evaluate(callback, None).unwrap();
        let mut output = vec![c(99.0); physical.output_len()];
        let report = physical
            .evaluate_into(
                callback,
                None,
                &mut output,
                &mut physical.workspace::<S>(()).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        into_equal(&owned, &report, &output);
        let relative =
            RelativeDifference::new(op, &axes[..n], outputs, NodeDifferenceOptions::default())
                .unwrap();
        let actual = relative.evaluate(callback, &point[..n], None).unwrap();
        let mut output = vec![c(99.0); relative.output_len()];
        let report = relative
            .evaluate_into(
                callback,
                &point[..n],
                None,
                &mut output,
                &mut relative.workspace::<S>(()).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        into_equal(&actual, &report, &output);
        match op {
            DerivativeOperation::Gradient => exact(actual.values().unwrap(), &[c(1.25), c(3.25)]),
            DerivativeOperation::Jacobian => exact(
                actual.values().unwrap(),
                &[c(1.25), c(3.25), c(1.0), c(1.0)],
            ),
            DerivativeOperation::Hessian => {
                exact(actual.values().unwrap(), &[c(2.0), c(1.0), c(1.0), c(4.0)]);
            }
            DerivativeOperation::First => {
                // The one-shot on the nodes `x + offsets` equals both plans bit for bit.
                let value: S =
                    physical_difference_derivative(|t: S| Ok(t * t * t), point[0], &nodes0)
                        .unwrap();
                bits(&[value], owned.values().unwrap());
                bits(&[value], actual.values().unwrap());
                exact(actual.values().unwrap(), &[c(0.1875)]);
            }
            DerivativeOperation::Second => exact(actual.values().unwrap(), &[c(1.5)]),
        }
    }
}

#[test]
fn fixed_f32() {
    fixed::<f32>();
}
#[test]
fn fixed_f64() {
    fixed::<f64>();
}
#[test]
fn adaptive_f32() {
    adaptive::<f32>();
}
#[test]
fn adaptive_f64() {
    adaptive::<f64>();
}
#[test]
fn geometry_f32() {
    geometry::<f32>();
}
#[test]
fn geometry_f64() {
    geometry::<f64>();
}

fn complex<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let point = [c(0.25), c(0.75)];
    let direction = [c(1.0), c(-2.0)];
    // Keep the cubic response at zero normal, including in f32.
    let step = S::EPSILON;
    let method = ComplexStep::default().with_step(step);
    let offsets = [c(-0.25), c(-0.125), c(0.0), c(0.375), c(0.5)];
    let axes = [offsets.as_slice(); 2];
    let scalar = |x: &[Complex<S>]| {
        Ok::<_, SciOxideError>(
            x[0] * x[0] + x[0] * x[1] + Complex::new(c(2.0), c(0.0)) * x[1] * x[1],
        )
    };
    let sample = |x: &[Complex<S>], y: &mut [Complex<S>]| -> Result<(), SciOxideError> {
        y[0] = scalar(x)?;
        Ok(())
    };
    let vector = |x: &[Complex<S>], y: &mut [Complex<S>]| -> Result<(), SciOxideError> {
        y[0] = scalar(x)?;
        y[1] = x[0] + x[1];
        Ok(())
    };
    let cubic = |x: &[Complex<S>], y: &mut [Complex<S>]| -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0] * x[0];
        Ok(())
    };
    let workspace =
        |shape| ComplexStepWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap();
    let owned = method.gradient(sample, &point).unwrap();
    let mut output = [c(99.0); 2];
    let report = method
        .gradient_into(
            sample,
            &point,
            &mut output,
            &mut workspace(DerivativeShape::gradient(2)),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(owned.values().unwrap(), &[c(1.25), c(3.25)]);
    let owned = method.jacobian(vector, &point, 2).unwrap();
    let mut output = [c(99.0); 4];
    let report = method
        .jacobian_into(
            vector,
            &point,
            2,
            &mut output,
            &mut workspace(DerivativeShape::jacobian(2, 2)),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(owned.values().unwrap(), &[c(1.25), c(3.25), c(1.0), c(1.0)]);
    let relative = RelativeComplexStep::new(&axes, method).unwrap();
    let owned = relative.evaluate(sample, &point, None).unwrap();
    let mut output = [c(99.0); 4];
    let report = relative
        .evaluate_into(
            sample,
            &point,
            None,
            &mut output,
            &mut relative.workspace(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(owned.values().unwrap(), &[c(2.0), c(1.0), c(1.0), c(4.0)]);
    // The scalar derivative is the one-coordinate gradient.
    let owned = method
        .derivative(|z: Complex<S>| Ok(z * z * z), point[0])
        .unwrap();
    prepared_equal(&owned, &method.gradient(cubic, &point[..1]).unwrap());
    exact(owned.values().unwrap(), &[c(0.1875)]);
    let value = complex_step_derivative(|z: Complex<S>| Ok(z * z * z), point[0]).unwrap();
    bits(
        &[value],
        ComplexStep::default()
            .derivative(|z: Complex<S>| Ok(z * z * z), point[0])
            .unwrap()
            .values()
            .unwrap(),
    );
    exact(&[value], &[c(0.1875)]);
    // The second derivative is the one-coordinate Hessian on relative offsets.
    let relative = RelativeComplexStep::new(&[&offsets], method).unwrap();
    let owned = relative.evaluate(cubic, &point[..1], None).unwrap();
    let mut output = [c(99.0)];
    let report = relative
        .evaluate_into(
            cubic,
            &point[..1],
            None,
            &mut output,
            &mut relative.workspace(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(owned.values().unwrap(), &[c(1.5)]);
    let nodes0 = offsets.map(|offset| point[0] + offset);
    let nodes1 = offsets.map(|offset| point[1] + offset);
    let nodes = [nodes0.as_slice(), nodes1.as_slice()];
    let physical = ComplexStepPlan::new(&point, &nodes, method).unwrap();
    let owned = physical.evaluate(sample, None).unwrap();
    let mut output = [c(99.0); 4];
    let report = physical
        .evaluate_into(
            sample,
            None,
            &mut output,
            &mut physical.workspace(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(owned.values().unwrap(), &[c(2.0), c(1.0), c(1.0), c(4.0)]);
    let physical = ComplexStepPlan::new(&point[..1], &nodes[..1], method).unwrap();
    let owned = physical.evaluate(cubic, None).unwrap();
    let mut output = [c(99.0)];
    let report = physical
        .evaluate_into(
            cubic,
            None,
            &mut output,
            &mut physical.workspace(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    exact(&output, &[c(1.5)]);
    let owned = method.jvp(vector, &point, &direction, 2).unwrap();
    let mut output = [c(99.0); 2];
    let report = method
        .jvp_into(
            vector,
            &point,
            &direction,
            &mut output,
            &mut workspace(DerivativeShape::jvp(2, 2)),
            ExecutionContext::sequential(),
        )
        .unwrap();
    into_equal(&owned, &report, &output);
    prepared_equal(
        &owned,
        &method
            .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
            .jvp(vector, &point, &direction, 2)
            .unwrap(),
    );
    exact(owned.values().unwrap(), &[c(-5.25), c(-1.0)]);
}
#[test]
fn complex_f32() {
    complex::<f32>();
}
#[test]
fn complex_f64() {
    complex::<f64>();
}

#[test]
fn invalid_options_and_shapes_precede_callbacks_and_publication() {
    for value in [
        0.0_f64,
        -0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(DifferenceStep::absolute(value).is_err());
        assert!(DifferenceStep::relative(value).is_err());
        assert!(DifferenceStep::absolute_coordinates(&[1.0, value]).is_err());
        assert!(DifferenceStep::relative_coordinates(&[value]).is_err());
    }
    let calls = Cell::new(0);
    let count = |_: &[f64], y: &mut [f64]| -> Result<(), SciOxideError> {
        calls.set(calls.get() + 1);
        y.fill(0.0);
        Ok(())
    };
    let count_complex = |_: &[Complex64], y: &mut [Complex64]| -> Result<(), SciOxideError> {
        calls.set(calls.get() + 1);
        y.fill(Complex64::new(0.0, 0.0));
        Ok(())
    };
    let bad_step = DifferenceStep::absolute_coordinates(&[0.125]).unwrap();
    let method = FiniteDifference::default().with_step(bad_step.clone());
    let mut output = [99.0; 2];
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    assert!(
        method
            .gradient_into(
                count,
                &[1.0, 2.0],
                None,
                None,
                &mut output,
                &mut w,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert_eq!(output, [99.0; 2]);
    let method = AdaptiveDifference::default().with_step(bad_step);
    let mut errors = [99.0; 2];
    let mut w = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    assert!(method.gradient(count, &[1.0, 2.0], None).is_err());
    assert!(
        method
            .gradient_into(
                count,
                &[1.0, 2.0],
                None,
                &mut output,
                &mut errors,
                &mut w,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(output, [99.0; 2]);
    assert_eq!(errors, [99.0; 2]);
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    assert!(
        FiniteDifference::default()
            .gradient(count, &[2.0], Some(&bounds), None)
            .is_err()
    );
    assert!(
        RelativeDifference::new(
            DerivativeOperation::Gradient,
            &[&[-0.5, 0.0, 0.5]],
            1,
            NodeDifferenceOptions::default(),
        )
        .unwrap()
        .evaluate(count, &[1.0, 2.0], None)
        .is_err()
    );
    assert!(
        ComplexStep::default()
            .with_step(-1e-20)
            .gradient(count_complex, &[1.0, 2.0])
            .is_err()
    );
    assert!(
        FiniteDifference::default()
            .jvp(count, &[1.0, 2.0], &[1.0], 1, None, None)
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    // A malformed output fails even when a supplied base would avoid the first callback.
    let mut out = [99.0];
    let method = FiniteDifference::default();
    let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jacobian(2, 2),
    ))
    .unwrap();
    assert!(
        method
            .jacobian_into(
                count,
                &[1.0, 2.0],
                2,
                None,
                Some(&[0.0, 0.0]),
                &mut out,
                &mut w,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert_eq!(out, [99.0]);
}

#[test]
fn empty_outputs_and_zero_directions_are_callback_free_identities() {
    let calls = Cell::new(0);
    let count = |_: &[f64], _: &mut [f64]| -> Result<(), SciOxideError> {
        calls.set(calls.get() + 1);
        Ok(())
    };
    let method = FiniteDifference::<f64>::default();
    assert!(
        method
            .gradient(count, &[], None, None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert!(
        method
            .hessian(count, &[], None, None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    let fixed = Bounds::new(vec![(Some(1.0), Some(1.0))]).unwrap();
    assert!(
        method
            .jacobian(count, &[1.0], 0, Some(&fixed), None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert!(
        method
            .clone()
            .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
            .jvp(count, &[1.0], &[1.0], 0, Some(&fixed), None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert!(
        AdaptiveDifference::default()
            .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
            .jvp(count, &[1.0], &[0.0], 0, None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        method
            .jvp(count, &[1.0], &[-0.0], 1, None, None)
            .unwrap()
            .values()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        AdaptiveDifference::default()
            .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
            .jvp(count, &[1.0], &[0.0], 1, None)
            .unwrap()
            .values(),
        Some(&[0.0][..])
    );
    assert_eq!(
        ComplexStep::default()
            .with_step(1e-100)
            .jvp(
                |_: &[Complex64], _: &mut [Complex64]| {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
                &[1.0],
                &[-0.0],
                1,
            )
            .unwrap()
            .values(),
        Some(&[0.0][..])
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn absolute_relative_scaling_and_replacement_are_explicit() {
    let run = |step| {
        let mut seen = Vec::new();
        let _ = FiniteDifference::default()
            .with_scheme(DifferenceScheme::Forward)
            .with_step(step)
            .derivative(
                |x: f64| {
                    seen.push(x);
                    Ok(x * x)
                },
                4.0,
            )
            .unwrap();
        seen
    };
    assert_eq!(
        run(DifferenceStep::absolute(0.125).unwrap()),
        vec![4.0, 4.125]
    );
    assert_eq!(
        run(DifferenceStep::relative(0.125).unwrap()),
        vec![4.0, 4.5]
    );
    let method = FiniteDifference::default()
        .with_step(DifferenceStep::absolute(0.125).unwrap())
        .with_step(DifferenceStep::relative(0.125).unwrap());
    let mut seen = Vec::new();
    let _ = method
        .derivative(
            |x: f64| {
                seen.push(x);
                Ok(x * x)
            },
            4.0,
        )
        .unwrap();
    assert_eq!(seen, vec![4.0, 3.5, 4.5]);
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    let mut nodes = Vec::new();
    let result = FiniteDifference::default()
        .with_step(DifferenceStep::absolute(0.125).unwrap())
        .gradient(
            |p: &[f64], y: &mut [f64]| {
                nodes.push(p[0]);
                y[0] = p[0] * p[0];
                Ok(())
            },
            &[0.0],
            Some(&bounds),
            None,
        )
        .unwrap();
    assert_eq!(nodes, vec![0.0, 0.125, 0.25]);
    assert_eq!(result.values().unwrap()[0], 0.0);
    let tiny = f64::from_bits(1);
    let mut calls = 0;
    assert!(
        FiniteDifference::default()
            .with_step(DifferenceStep::absolute(tiny).unwrap())
            .derivative(
                |_: f64| {
                    calls += 1;
                    Ok(0.0)
                },
                1.0,
            )
            .is_err()
    );
    assert_eq!(calls, 0);
}

fn colored<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    let pattern = StructuralPatternView::new(2, 2, &[0, 1, 2], &[0, 1]).unwrap();
    let schedule = ColoringSchedule::new(
        pattern.to_owned().unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0],
    )
    .unwrap();
    let point = [c(0.25), c(0.75)];
    let callback = |x: &[S], y: &mut [S]| -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0];
        y[1] = x[1] * x[1];
        Ok(())
    };
    let method = AdaptiveDifference::default()
        .with_step(DifferenceStep::relative_coordinates(&[c(0.25), c(0.5)]).unwrap());
    let plan = ColoredJacobianPlan::new(&schedule, method.clone()).unwrap();
    let owned = plan.evaluate(callback, &point, None).unwrap();
    let (mut output, mut errors) = ([c(99.0); 2], [c(99.0); 2]);
    let report = plan
        .evaluate_into(
            callback,
            &point,
            None,
            &mut output,
            &mut errors,
            &mut plan.workspace::<S>(()).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    adaptive_into_equal(&owned, &report, &output, &errors);
    exact(owned.values().unwrap(), &[c(0.5), c(1.5)]);
    // A schedule rebuilt from the same pattern gives the same plan.
    let rebuilt = pattern
        .to_owned()
        .and_then(|p| ColoringSchedule::new(p, ColoringKind::DerivativeColumns, vec![0, 0]))
        .unwrap();
    rebuilt.validate_pattern(pattern).unwrap();
    adaptive_equal(
        &owned,
        &ColoredJacobianPlan::new(&rebuilt, method)
            .unwrap()
            .evaluate(callback, &point, None)
            .unwrap(),
    );
    let one_shot = sparse_jacobian(callback, &point, &schedule).unwrap();
    let configured = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default())
        .unwrap()
        .evaluate(callback, &point, None)
        .unwrap();
    bits(&one_shot, configured.values().unwrap());
}
#[test]
fn colored_f32() {
    colored::<f32>();
}
#[test]
fn colored_f64() {
    colored::<f64>();
}

#[test]
fn injected_coloring_identity_and_kind_are_checked_before_numerical_callbacks() {
    let pattern = StructuralPatternView::new(1, 1, &[0, 1], &[0]).unwrap();
    let calls = Cell::new(0);
    let evaluate = |schedule: Result<ColoringSchedule, SciOxideError>,
                    view: StructuralPatternView<'_>,
                    x: &[f64]| {
        schedule.and_then(|schedule| {
            schedule.validate_pattern(view)?;
            ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default())?.evaluate(
                |_: &[f64], _: &mut [f64]| {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
                x,
                None,
            )
        })
    };
    let foreign = StructuralPattern::new(1, 1, vec![0, 0], vec![])
        .and_then(|p| ColoringSchedule::new(p, ColoringKind::DerivativeColumns, vec![0]));
    assert!(evaluate(foreign, pattern, &[1.0]).is_err());
    assert_eq!(calls.get(), 0);
    let distance_one = pattern
        .to_owned()
        .and_then(|p| ColoringSchedule::new(p, ColoringKind::DistanceOne, vec![0]));
    assert!(evaluate(distance_one, pattern, &[1.0]).is_err());
    assert_eq!(calls.get(), 0);
    let empty = StructuralPatternView::new(0, 2, &[0], &[]).unwrap();
    let schedule = empty
        .to_owned()
        .and_then(|p| ColoringSchedule::new(p, ColoringKind::DerivativeColumns, vec![0, 0]));
    assert!(
        evaluate(schedule, empty, &[1.0, 2.0])
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn recorded_scipy_conventions_have_independent_quadratic_references() {
    let capture = include_str!("data/scipy_conventions.tsv");
    assert!(capture.starts_with("# SciPy 1.17.1 NumPy 2.4.4"));
    let mut count = 0;
    for row in capture.lines().filter(|line| line.starts_with("float")) {
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(fields.len(), 10);
        let x: f64 = fields[3].parse().unwrap();
        let lower: f64 = fields[5].parse().unwrap();
        let upper: f64 = fields[6].parse().unwrap();
        let estimate: f64 = fields[7].parse().unwrap();
        let analytic: f64 = fields[8].parse().unwrap();
        let nodes: Vec<f64> = fields[9]
            .split(',')
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(analytic, 2.0 * x);
        assert!(nodes.iter().all(|&node| lower <= node && node <= upper));
        assert_eq!(nodes[0], x);
        let h = nodes[1] - x;
        assert!(h != 0.0);
        // Exact algebra supplies the forward finite-step reference; central and
        // second-order one-sided formulas differentiate a quadratic exactly.
        let reference = if fields[1] == "2-point" {
            2.0 * x + h
        } else {
            2.0 * x
        };
        let epsilon = if fields[0] == "float32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        let roundoff = 32.0 * epsilon * (1.0 + x * x) / h.abs();
        assert!((estimate - reference).abs() <= roundoff, "{row}");
        if fields[2] == "relative" && x.abs() > 0.0 {
            let relative: f64 = fields[4].parse().unwrap();
            assert_eq!(h.abs(), relative * x.abs());
        }
        if fields[2] == "absolute" {
            assert_eq!(h.abs(), 0.125);
        }
        count += 1;
    }
    assert_eq!(count, 32);
}

#[test]
fn configured_owned_resources_return_authoritative_unpublished_reports() {
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        algorithm_id!("test", "adapter-resource", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut calls = 0;
    let method = RelativeDifference::new(
        DerivativeOperation::First,
        &[&[-0.5, 0.0, 0.5]],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut workspace = method.workspace::<f64>(()).unwrap();
    let mut output = [99.0];
    let report = method
        .evaluate_into(
            |_: &[f64], value: &mut [f64]| {
                calls += 1;
                value[0] = 0.0;
                Ok(())
            },
            &[1.0],
            None,
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(!report.output_available());
    assert_eq!(output, [99.0]);
    assert_eq!(
        report.termination(),
        DifferentiationTermination::EvaluationLimit
    );
    assert_eq!(calls, 1);
    assert_eq!(report.evaluations(), 1);
}

#[test]
fn fixed_convenience_preserves_publication_and_exact_counts_in_both_precisions() {
    fn check<S: RealScalar>() {
        for x in [S::ZERO, S::ONE, -S::ONE] {
            let mut calls = 0;
            let result: DifferentiationResult<S> = FiniteDifference::default()
                .derivative(
                    |point| {
                        calls += 1;
                        Ok(point * point)
                    },
                    x,
                )
                .unwrap();
            let report = result.report();
            assert_eq!(
                report.source(),
                DerivativeSource::FiniteDifference(DifferenceScheme::Central)
            );
            assert_eq!(report.termination(), DifferentiationTermination::Completed);
            assert!(report.output_available());
            assert_eq!(report.output_available(), result.values().is_some());
            assert_eq!(report.evaluations(), calls);
            assert_eq!(report.completed_entries(), 1);
            assert!(result.errors().is_none());
            exact(result.values().unwrap(), &[S::TWO * x]);
            let one_shot_calls = Cell::new(0);
            let value = derivative(
                |point: S| {
                    one_shot_calls.set(one_shot_calls.get() + 1);
                    Ok(point * point)
                },
                x,
            )
            .unwrap();
            assert_eq!(one_shot_calls.get(), calls);
            bits(&[value], result.values().unwrap());
        }
    }
    check::<f32>();
    check::<f64>();
}

fn callback_failure() -> SciOxideError {
    SciOxideError::domain_error(
        argument!("x"),
        DomainReason::OutsideDomain,
        "callback failed",
    )
}
// The callback error is the cause, `source()`, of a `DifferentiationFailure` whose
// frame records the evaluations made.
fn assert_callback_cause(error: &SciOxideError, calls: usize) {
    assert!(
        matches!(error.view(), SciOxideErrorView::DifferentiationFailure { cause, frames }
        if frames[0].evaluations == calls
            && matches!(cause.view(), SciOxideErrorView::DomainError {
             reason: DomainReason::OutsideDomain, ..
        }))
    );
    let source = error
        .source()
        .and_then(|source| source.downcast_ref::<SciOxideError>())
        .unwrap();
    assert!(matches!(
        source.view(),
        SciOxideErrorView::DomainError {
            reason: DomainReason::OutsideDomain,
            ..
        }
    ));
}

#[test]
fn fixed_convenience_rejects_non_finite_values_and_preserves_callback_errors() {
    let calls = Cell::new(0);
    let error = derivative(
        |_: f64| {
            calls.set(calls.get() + 1);
            Err::<f64, _>(callback_failure())
        },
        1.0_f64,
    )
    .unwrap_err();
    assert_eq!(calls.get(), 1);
    assert_callback_cause(&error, calls.get());

    for point in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        calls.set(0);
        let error = derivative(
            |x: f64| {
                calls.set(calls.get() + 1);
                Ok(x)
            },
            point,
        )
        .unwrap_err();
        assert_eq!(calls.get(), 0);
        assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        calls.set(0);
        let error = derivative(
            |_: f64| {
                calls.set(calls.get() + 1);
                Ok(value)
            },
            1.0,
        )
        .unwrap_err();
        assert_eq!(calls.get(), 1);
        assert!(
            matches!(error.view(), SciOxideErrorView::DifferentiationFailure { cause, frames }
            if frames[0].evaluations == calls.get()
                && matches!(cause.view(), SciOxideErrorView::NonFinite {  .. }))
        );
    }
}

#[test]
fn callback_errors_are_the_cause_of_a_differentiation_failure_for_every_method_and_plan() {
    let calls = Cell::new(0);
    let fail = |_: &[f64], _: &mut [f64]| -> Result<(), SciOxideError> {
        calls.set(calls.get() + 1);
        Err(callback_failure())
    };
    let fail_complex = |_: &[Complex64], _: &mut [Complex64]| -> Result<(), SciOxideError> {
        calls.set(calls.get() + 1);
        Err(callback_failure())
    };
    let check = |error: SciOxideError| {
        assert_eq!(calls.get(), 1);
        assert_callback_cause(&error, calls.get());
        calls.set(0);
    };
    let x = [1.0, 2.0];
    check(
        FiniteDifference::default()
            .gradient(fail, &x, None, None)
            .unwrap_err(),
    );
    check(
        AdaptiveDifference::default()
            .jacobian(fail, &x, 2, None)
            .unwrap_err(),
    );
    check(
        ComplexStep::default()
            .gradient(fail_complex, &x)
            .unwrap_err(),
    );
    let pattern = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
    let schedule =
        ColoringSchedule::new(pattern, ColoringKind::DerivativeColumns, vec![0, 0]).unwrap();
    check(
        ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default())
            .unwrap()
            .evaluate(fail, &x, None)
            .unwrap_err(),
    );
    let nodes = [0.5, 1.0, 1.5];
    check(
        PhysicalDifferencePlan::new(
            DerivativeOperation::Gradient,
            &[1.0],
            &[&nodes],
            1,
            NodeDifferenceOptions::default(),
        )
        .unwrap()
        .evaluate(fail, None)
        .unwrap_err(),
    );
    check(
        ComplexStepPlan::new(&[1.0], &[&nodes], ComplexStep::default())
            .unwrap()
            .evaluate(fail_complex, None)
            .unwrap_err(),
    );
    let offsets = [-0.5, 0.0, 0.5];
    check(
        RelativeDifference::new(
            DerivativeOperation::Gradient,
            &[&offsets],
            1,
            NodeDifferenceOptions::default(),
        )
        .unwrap()
        .evaluate(fail, &[1.0], None)
        .unwrap_err(),
    );
    check(
        RelativeComplexStep::new(&[&offsets], ComplexStep::default())
            .unwrap()
            .evaluate(fail_complex, &[1.0], None)
            .unwrap_err(),
    );
}
