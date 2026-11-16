//! Public shape/nonfinite classification and transactional rejection.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;
use scioxide_core::{
    Argument, Bounds, ColoringKind, ColoringSchedule, Complex, ErrorTolerance, ExecutionContext,
    ExtendedInterval, InvalidInputReason, NonFiniteOrigin, NormSelection, SciOxideError,
    SciOxideErrorKind, SciOxideErrorView, StructuralPattern,
};
use scioxide_differentiate::*;

fn assert_dimensions(error: &SciOxideError, expected: usize, actual: usize) {
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch {
         expected: wanted, actual: found,
    } if wanted == [expected] && found == [actual])
    );
}

fn assert_nonfinite(error: &SciOxideError) {
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

fn assert_invalid(error: &SciOxideError, argument: Argument, reason: InvalidInputReason) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput {
            argument: found, reason: why, ..
        } if found == argument && why == reason),
        "{error:?}"
    );
}

/// How an evaluation treats one of the point, output and error-output lengths.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Length {
    /// The method or plan fixes the length 2, so another length is rejected.
    Fixed,
    /// The argument sets the shape, so the other arguments of length 2 are
    /// rejected against its length.
    Sets,
    /// The form has no such argument.
    Absent,
}

fn check_buffers(
    lengths: [Length; 3],
    calls: &Cell<usize>,
    mut evaluate: impl FnMut(
        &[f64],
        &mut [f64],
        &mut [f64],
    ) -> Result<DifferentiationReport, SciOxideError>,
) {
    // Each argument disagrees independently while all other arguments have length 2.
    for (argument, length) in lengths.into_iter().enumerate() {
        if length == Length::Absent {
            continue;
        }
        for actual in [0, 1, 3] {
            let mut sizes = [2; 3];
            sizes[argument] = actual;
            let mut output = [77.0; 3];
            let mut errors = [88.0; 3];
            let error = evaluate(
                &[1.0, 2.0, 3.0][..sizes[0]],
                &mut output[..sizes[1]],
                &mut errors[..sizes[2]],
            )
            .unwrap_err();
            if length == Length::Fixed {
                assert_dimensions(&error, 2, actual);
            } else {
                assert_dimensions(&error, actual, 2);
            }
            assert_eq!(calls.get(), 0);
            assert_eq!(output, [77.0; 3]);
            assert_eq!(errors, [88.0; 3]);
        }
    }
    if lengths[0] == Length::Absent {
        return;
    }
    // Correctly shaped nonfinite points are a separate input failure.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut output = [77.0; 2];
        let mut errors = [88.0; 2];
        let error = evaluate(&[value, 2.0], &mut output, &mut errors).unwrap_err();
        assert_nonfinite(&error);
        assert_eq!(calls.get(), 0);
        assert_eq!(output, [77.0; 2]);
        assert_eq!(errors, [88.0; 2]);
    }
}

fn diagonal() -> ColoringSchedule {
    ColoringSchedule::new(
        StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0],
    )
    .unwrap()
}

#[test]
fn adaptive_lengths_are_typed_and_do_not_publish() {
    let method = AdaptiveDifference::<f64>::default();
    // The point sets `n`; the workspace also serves three coordinates.
    let mut workspace = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(3),
    ))
    .unwrap();
    let calls = Cell::new(0);
    let lengths = [Length::Sets, Length::Fixed, Length::Fixed];
    check_buffers(lengths, &calls, |point, output, errors| {
        method.gradient_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            point,
            None,
            output,
            errors,
            &mut workspace,
            ExecutionContext::sequential(),
        )
    });
    let result = method
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x.iter().map(|v| v * v).sum();
                Ok(())
            },
            &[1.0, 2.0],
            None,
            &mut [0.0; 2],
            &mut [0.0; 2],
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.is_success());
}

#[test]
fn adaptive_jvp_checks_all_lengths_and_direction_values() {
    let method = AdaptiveDifference::<f64>::default();
    // The point sets `n` and the output sets `m`; the workspace serves both up to three.
    let mut workspace = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(3, 3),
    ))
    .unwrap();
    let calls = Cell::new(0);
    let lengths = [Length::Sets, Length::Sets, Length::Fixed];
    check_buffers(lengths, &calls, |point, output, errors| {
        method.jvp_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            point,
            &[1.0, 1.0],
            None,
            output,
            errors,
            &mut workspace,
            ExecutionContext::sequential(),
        )
    });
    for actual in [0, 1, 3] {
        let mut output = [77.0; 2];
        let mut errors = [88.0; 2];
        let error = method
            .jvp_into(
                |_: &[f64], _: &mut [f64]| {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
                &[1.0, 2.0],
                &[1.0; 3][..actual],
                None,
                &mut output,
                &mut errors,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_dimensions(&error, 2, actual);
        assert_eq!(calls.get(), 0);
        assert_eq!(output, [77.0; 2]);
        assert_eq!(errors, [88.0; 2]);
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut output = [77.0; 2];
        let mut errors = [88.0; 2];
        let error = method
            .jvp_into(
                |_: &[f64], _: &mut [f64]| {
                    calls.set(calls.get() + 1);
                    Ok(())
                },
                &[1.0, 2.0],
                &[1.0, value],
                None,
                &mut output,
                &mut errors,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_nonfinite(&error);
        assert_eq!(calls.get(), 0);
        assert_eq!(output, [77.0; 2]);
        assert_eq!(errors, [88.0; 2]);
    }
}

#[test]
fn colored_lengths_are_typed_and_do_not_publish() {
    let schedule = diagonal();
    let plan = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default()).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let calls = Cell::new(0);
    let lengths = [Length::Fixed; 3];
    check_buffers(lengths, &calls, |point, output, errors| {
        plan.evaluate_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            point,
            None,
            output,
            errors,
            &mut workspace,
            ExecutionContext::sequential(),
        )
    });
}

#[test]
fn physical_and_relative_lengths_are_typed_and_do_not_publish() {
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[1.0, 2.0],
        &[&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0]],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let relative = RelativeDifference::new(
        DerivativeOperation::Gradient,
        &[&[-1.0, 0.0, 1.0][..]; 2],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut physical_workspace = physical.workspace(()).unwrap();
    let mut relative_workspace = relative.workspace(()).unwrap();
    let calls = Cell::new(0);
    let lengths = [Length::Absent, Length::Fixed, Length::Absent];
    check_buffers(lengths, &calls, |_, output, _| {
        physical.evaluate_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            None,
            output,
            &mut physical_workspace,
            ExecutionContext::sequential(),
        )
    });
    let lengths = [Length::Fixed, Length::Fixed, Length::Absent];
    check_buffers(lengths, &calls, |point, output, _| {
        relative.evaluate_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            point,
            None,
            output,
            &mut relative_workspace,
            ExecutionContext::sequential(),
        )
    });
    // The node 0 of the first coordinate lies below the lower bound 0.5.
    let bounds = Bounds::new(vec![(Some(0.5), None), (None, None)]).unwrap();
    let mut output = [77.0; 2];
    let error = physical
        .evaluate_into(
            |_: &[f64], _: &mut [f64]| {
                calls.set(calls.get() + 1);
                Ok(())
            },
            Some(&bounds),
            &mut output,
            &mut physical_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let reason = InvalidInputReason::UnsupportedConfiguration;
    assert_invalid(&error, scioxide_core::argument!("bounds"), reason);
    assert_eq!(calls.get(), 0);
    assert_eq!(output, [77.0; 2]);
}

#[test]
fn tolerance_and_node_lengths_and_nonfinite_points_are_distinct() {
    let ignore = |_: &[f64], _: &mut [f64]| Ok(());
    for actual in [0, 1, 3] {
        let tolerance =
            ErrorTolerance::try_vector(&[1e-6; 3][..actual], 1e-6, NormSelection::Infinity)
                .unwrap();
        let method = AdaptiveDifference::<f64>::default().with_tolerance(tolerance);
        let error = method.gradient(ignore, &[1.0, 2.0], None).unwrap_err();
        assert_dimensions(&error, 2, actual);
        let error = method
            .jvp(ignore, &[1.0, 2.0], &[1.0, 1.0], 2, None)
            .unwrap_err();
        assert_dimensions(&error, 2, actual);
        let schedule = diagonal();
        let error = ColoredJacobianPlan::new(&schedule, method).unwrap_err();
        assert_dimensions(&error, 2, actual);
        let nodes = [&[0.0, 1.0, 2.0][..]; 3];
        let error = PhysicalDifferencePlan::new(
            DerivativeOperation::Gradient,
            &[1.0, 2.0],
            &nodes[..actual],
            1,
            NodeDifferenceOptions::default(),
        )
        .unwrap_err();
        assert_dimensions(&error, 2, actual);
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = PhysicalDifferencePlan::new(
            DerivativeOperation::First,
            &[value],
            &[&[0.0, 1.0, 2.0]],
            1,
            NodeDifferenceOptions::default(),
        )
        .unwrap_err();
        assert_nonfinite(&error);
    }
}

/// The operations of the method values; the `_into` forms write into sentinel
/// buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Derivative,
    SecondDerivative,
    DerivativeInto,
    Gradient,
    GradientInto,
    Hessian,
    HessianInto,
    Jacobian,
    JacobianInto,
    Jvp,
    JvpInto,
}

const FIXED: [Operation; 11] = [
    Operation::Derivative,
    Operation::SecondDerivative,
    Operation::DerivativeInto,
    Operation::Gradient,
    Operation::GradientInto,
    Operation::Hessian,
    Operation::HessianInto,
    Operation::Jacobian,
    Operation::JacobianInto,
    Operation::Jvp,
    Operation::JvpInto,
];
const ADAPTIVE: [Operation; 10] = [
    Operation::Derivative,
    Operation::SecondDerivative,
    Operation::Gradient,
    Operation::GradientInto,
    Operation::Hessian,
    Operation::HessianInto,
    Operation::Jacobian,
    Operation::JacobianInto,
    Operation::Jvp,
    Operation::JvpInto,
];
const COMPLEX: [Operation; 7] = [
    Operation::Derivative,
    Operation::Gradient,
    Operation::GradientInto,
    Operation::Jacobian,
    Operation::JacobianInto,
    Operation::Jvp,
    Operation::JvpInto,
];
const X: [f64; 2] = [1.0, 2.0];
const DIRECTION: [f64; 2] = [1.0, -1.0];

impl Operation {
    const fn is_jvp(self) -> bool {
        matches!(self, Self::Jvp | Self::JvpInto)
    }
    /// The number of coordinates the operation differentiates.
    const fn inputs(self) -> usize {
        match self {
            Self::Derivative | Self::SecondDerivative | Self::DerivativeInto => 1,
            _ => 2,
        }
    }
}

/// Runs `operation`, which must fail before any callback and leave its
/// outputs unchanged, and returns the error.
fn rejected(
    operation: Operation,
    run: impl FnOnce(&Cell<usize>, &mut [f64; 4], &mut [f64; 4]) -> Option<SciOxideError>,
) -> SciOxideError {
    let calls = Cell::new(0);
    let (mut out, mut errors) = ([77.0; 4], [88.0; 4]);
    let error = run(&calls, &mut out, &mut errors);
    assert_eq!(calls.get(), 0, "{operation:?}");
    assert_eq!(out, [77.0; 4], "{operation:?}");
    assert_eq!(errors, [88.0; 4], "{operation:?}");
    error.unwrap_or_else(|| unreachable!("{operation:?} accepted the setting"))
}

fn fixed(method: &FiniteDifference<f64>, operation: Operation) -> SciOxideError {
    rejected(operation, |calls, out, _| {
        let scalar = |t: f64| -> Result<f64, SciOxideError> {
            calls.set(calls.get() + 1);
            Ok(t * t)
        };
        let vector = |x: &[f64], y: &mut [f64]| -> Result<(), SciOxideError> {
            calls.set(calls.get() + 1);
            y.fill(x[0] * x[1]);
            Ok(())
        };
        let workspace = |shape| {
            let method = FiniteDifference::default();
            FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap()
        };
        let context = ExecutionContext::sequential;
        match operation {
            Operation::Derivative => method.derivative(scalar, 1.0).err(),
            Operation::SecondDerivative => method.second_derivative(scalar, 1.0).err(),
            Operation::DerivativeInto => {
                let bounds = ExtendedInterval::entire();
                let out = &mut out[0];
                method
                    .derivative_into(scalar, 1.0, bounds, None, out, context())
                    .err()
            }
            Operation::Gradient => method.gradient(vector, &X, None, None).err(),
            Operation::GradientInto => {
                let mut workspace = workspace(DerivativeShape::gradient(2));
                let out = &mut out[..2];
                method
                    .gradient_into(vector, &X, None, None, out, &mut workspace, context())
                    .err()
            }
            Operation::Hessian => method.hessian(vector, &X, None, None).err(),
            Operation::HessianInto => {
                let mut workspace = workspace(DerivativeShape::hessian(2));
                method
                    .hessian_into(vector, &X, None, None, out, &mut workspace, context())
                    .err()
            }
            Operation::Jacobian => method.jacobian(vector, &X, 2, None, None).err(),
            Operation::JacobianInto => {
                let mut workspace = workspace(DerivativeShape::jacobian(2, 2));
                method
                    .jacobian_into(vector, &X, 2, None, None, out, &mut workspace, context())
                    .err()
            }
            Operation::Jvp => method.jvp(vector, &X, &DIRECTION, 2, None, None).err(),
            Operation::JvpInto => {
                let mut workspace = workspace(DerivativeShape::jvp(2, 2));
                let out = &mut out[..2];
                method
                    .jvp_into(
                        vector,
                        &X,
                        &DIRECTION,
                        None,
                        None,
                        out,
                        &mut workspace,
                        context(),
                    )
                    .err()
            }
        }
    })
}

fn adaptive(method: &AdaptiveDifference<f64>, operation: Operation) -> SciOxideError {
    rejected(operation, |calls, out, errors| {
        let scalar = |t: f64| -> Result<f64, SciOxideError> {
            calls.set(calls.get() + 1);
            Ok(t * t)
        };
        let vector = |x: &[f64], y: &mut [f64]| -> Result<(), SciOxideError> {
            calls.set(calls.get() + 1);
            y.fill(x[0] * x[1]);
            Ok(())
        };
        let workspace = |shape| {
            let method = AdaptiveDifference::default();
            AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap()
        };
        let context = ExecutionContext::sequential;
        match operation {
            Operation::Derivative => method.derivative(scalar, 1.0).err(),
            Operation::SecondDerivative => method.second_derivative(scalar, 1.0).err(),
            Operation::Gradient => method.gradient(vector, &X, None).err(),
            Operation::GradientInto => {
                let mut workspace = workspace(DerivativeShape::gradient(2));
                let (out, errors) = (&mut out[..2], &mut errors[..2]);
                method
                    .gradient_into(vector, &X, None, out, errors, &mut workspace, context())
                    .err()
            }
            Operation::Hessian => method.hessian(vector, &X, None).err(),
            Operation::HessianInto => {
                let mut workspace = workspace(DerivativeShape::hessian(2));
                method
                    .hessian_into(vector, &X, None, out, errors, &mut workspace, context())
                    .err()
            }
            Operation::Jacobian => method.jacobian(vector, &X, 2, None).err(),
            Operation::JacobianInto => {
                let mut workspace = workspace(DerivativeShape::jacobian(2, 2));
                method
                    .jacobian_into(vector, &X, 2, None, out, errors, &mut workspace, context())
                    .err()
            }
            Operation::Jvp => method.jvp(vector, &X, &DIRECTION, 2, None).err(),
            Operation::JvpInto => {
                let mut workspace = workspace(DerivativeShape::jvp(2, 2));
                let (out, errors) = (&mut out[..2], &mut errors[..2]);
                method
                    .jvp_into(
                        vector,
                        &X,
                        &DIRECTION,
                        None,
                        out,
                        errors,
                        &mut workspace,
                        context(),
                    )
                    .err()
            }
            Operation::DerivativeInto => unreachable!("adaptive differences have no scalar form"),
        }
    })
}

fn complex(method: &ComplexStep<f64>, operation: Operation) -> SciOxideError {
    rejected(operation, |calls, out, _| {
        let scalar = |z: Complex<f64>| -> Result<Complex<f64>, SciOxideError> {
            calls.set(calls.get() + 1);
            Ok(z * z)
        };
        let vector = |z: &[Complex<f64>], y: &mut [Complex<f64>]| -> Result<(), SciOxideError> {
            calls.set(calls.get() + 1);
            y.fill(z[0] * z[1]);
            Ok(())
        };
        let workspace = |shape| {
            let method = ComplexStep::default();
            ComplexStepWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap()
        };
        let context = ExecutionContext::sequential;
        match operation {
            Operation::Derivative => method.derivative(scalar, 1.0).err(),
            Operation::Gradient => method.gradient(vector, &X).err(),
            Operation::GradientInto => {
                let mut workspace = workspace(DerivativeShape::gradient(2));
                let out = &mut out[..2];
                method
                    .gradient_into(vector, &X, out, &mut workspace, context())
                    .err()
            }
            Operation::Jacobian => method.jacobian(vector, &X, 2).err(),
            Operation::JacobianInto => {
                let mut workspace = workspace(DerivativeShape::jacobian(2, 2));
                method
                    .jacobian_into(vector, &X, 2, out, &mut workspace, context())
                    .err()
            }
            Operation::Jvp => method.jvp(vector, &X, &DIRECTION, 2).err(),
            Operation::JvpInto => {
                let mut workspace = workspace(DerivativeShape::jvp(2, 2));
                let out = &mut out[..2];
                method
                    .jvp_into(vector, &X, &DIRECTION, out, &mut workspace, context())
                    .err()
            }
            _ => unreachable!("the complex step has no {operation:?}"),
        }
    })
}

#[test]
fn method_settings_are_rejected_by_every_operation_before_callbacks() {
    let this = scioxide_core::argument!("self");
    let steps = || DifferenceStep::absolute_coordinates(&[0.1; 3]).unwrap();
    let validate = ZeroDirectionPolicy::ValidateBase;
    let l2 = ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::L2).unwrap();
    for operation in FIXED {
        let error = fixed(&FiniteDifference::default().with_step(steps()), operation);
        assert_dimensions(&error, operation.inputs(), 3);
        if !operation.is_jvp() {
            let method = FiniteDifference::default().with_zero_direction(validate);
            let error = fixed(&method, operation);
            assert_invalid(&error, this, InvalidInputReason::Inconsistent);
        }
        // Gradients and scalar first derivatives accept feasible-zero coordinates.
        if !matches!(
            operation,
            Operation::Derivative
                | Operation::DerivativeInto
                | Operation::Gradient
                | Operation::GradientInto
        ) {
            let feasible = FixedCoordinatePolicy::FeasibleZero;
            let method = FiniteDifference::default().with_fixed_coordinates(feasible);
            let error = fixed(&method, operation);
            assert_invalid(&error, this, InvalidInputReason::Inconsistent);
        }
    }
    for operation in ADAPTIVE {
        let default = AdaptiveDifference::default;
        for method in [
            default().with_max_levels(2),
            default().with_max_levels(65),
            default().with_tolerance(l2.clone()),
        ] {
            let error = adaptive(&method, operation);
            assert_invalid(&error, this, InvalidInputReason::OutOfRange);
        }
        let error = adaptive(&default().with_step(steps()), operation);
        assert_dimensions(&error, operation.inputs(), 3);
        if !operation.is_jvp() {
            let error = adaptive(&default().with_zero_direction(validate), operation);
            assert_invalid(&error, this, InvalidInputReason::Inconsistent);
        }
    }
    for operation in COMPLEX {
        for step in [0.0, -0.0, -1.0, 1e-320] {
            let error = complex(&ComplexStep::default().with_step(step), operation);
            assert_invalid(&error, this, InvalidInputReason::OutOfRange);
        }
        for step in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_nonfinite(&complex(&ComplexStep::default().with_step(step), operation));
        }
        if !operation.is_jvp() {
            let error = complex(
                &ComplexStep::default().with_zero_direction(validate),
                operation,
            );
            assert_invalid(&error, this, InvalidInputReason::Inconsistent);
        }
    }
}

#[test]
fn plan_constructors_name_the_method_for_its_settings() {
    let method = scioxide_core::argument!("method");
    let validate = ZeroDirectionPolicy::ValidateBase;
    let schedule = diagonal();
    let default = AdaptiveDifference::<f64>::default;
    let l2 = ErrorTolerance::scalar(1e-8, 1e-8, NormSelection::L2).unwrap();
    for settings in [
        default().with_max_levels(2),
        default().with_max_levels(65),
        default().with_tolerance(l2),
    ] {
        let error = ColoredJacobianPlan::new(&schedule, settings).unwrap_err();
        assert_invalid(&error, method, InvalidInputReason::OutOfRange);
    }
    let error = ColoredJacobianPlan::new(&schedule, default().with_zero_direction(validate));
    assert_invalid(
        &error.unwrap_err(),
        method,
        InvalidInputReason::Inconsistent,
    );
    let steps = DifferenceStep::absolute_coordinates(&[0.1; 3]).unwrap();
    let error = ColoredJacobianPlan::new(&schedule, default().with_step(steps)).unwrap_err();
    assert_dimensions(&error, 2, 3);

    let physical = |settings| ComplexStepPlan::new(&[1.0], &[&[0.0, 1.0, 2.0]], settings);
    let relative = |settings| RelativeComplexStep::new(&[&[-1.0, 0.0, 1.0]], settings);
    for step in [0.0, -0.0, -1.0, 1e-320] {
        let settings = ComplexStep::default().with_step(step);
        let reason = InvalidInputReason::OutOfRange;
        assert_invalid(&physical(settings).unwrap_err(), method, reason);
        assert_invalid(&relative(settings).unwrap_err(), method, reason);
    }
    for step in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let settings = ComplexStep::default().with_step(step);
        assert_nonfinite(&physical(settings).unwrap_err());
        assert_nonfinite(&relative(settings).unwrap_err());
    }
    let settings = ComplexStep::default().with_zero_direction(validate);
    let reason = InvalidInputReason::Inconsistent;
    assert_invalid(&physical(settings).unwrap_err(), method, reason);
    assert_invalid(&relative(settings).unwrap_err(), method, reason);
}

#[test]
fn supported_empty_results_are_valid_without_callbacks() {
    let calls = Cell::new(0);
    let mut callback = |_: &[f64], _: &mut [f64]| {
        calls.set(calls.get() + 1);
        Ok(())
    };
    let adaptive = AdaptiveDifference::<f64>::default();
    assert!(
        adaptive
            .gradient(&mut callback, &[], None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert!(
        adaptive
            .jvp(&mut callback, &[], &[], 0, None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[],
        &[],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    assert!(
        physical
            .evaluate(&mut callback, None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    let relative = RelativeDifference::new(
        DerivativeOperation::Gradient,
        &[],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    assert!(
        relative
            .evaluate(&mut callback, &[], None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(0, 0, vec![0], vec![]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![],
    )
    .unwrap();
    let colored =
        ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default()).unwrap();
    assert!(
        colored
            .evaluate(&mut callback, &[], None)
            .unwrap()
            .values()
            .unwrap()
            .is_empty()
    );
    assert_eq!(calls.get(), 0);
}
