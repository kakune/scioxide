//! Prepared step exhaustion, transactional publication and workspace reuse.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Bounds, ColoringKind, ColoringSchedule, DifferentiationLocation, ExecutionContext,
    ExtendedInterval, InvalidInputReason, NumericScalar, RealScalar, SciOxideError,
    SciOxideErrorKind, SciOxideErrorView, StructuralPattern,
};
use scioxide_differentiate::*;

type Callback<'a, S> = dyn FnMut(&[S], &mut [S]) -> Result<(), SciOxideError> + 'a;

fn exhaustion_options<S: RealScalar>(after_estimate: bool) -> AdaptiveDifference<S> {
    if after_estimate {
        AdaptiveDifference::default()
            .with_method(AdaptiveDifferenceMethod::Richardson)
            .with_step(DifferenceStep::absolute(S::TWO * S::EPSILON).unwrap())
    } else {
        AdaptiveDifference::default()
    }
}

// The adapter only dispatches public APIs; preparation and reservation happen
// before this harness measures evaluation and retries on the same workspace.
fn exhaustion_and_reuse<S: RealScalar + NumericScalar<Real = S>>(
    after_estimate: bool,
    base_location: DifferentiationLocation,
    refinement_location: DifferentiationLocation,
    mut evaluate: impl FnMut(
        &[S],
        Option<&Bounds<S>>,
        &mut Callback<'_, S>,
        &mut [S],
        &mut [S],
    ) -> Result<DifferentiationReport, SciOxideError>,
) {
    let four = S::TWO * S::TWO;
    let room = if after_estimate { four } else { S::ONE } * S::EPSILON;
    let bounds = Bounds::new(vec![(Some(S::ONE), Some(S::ONE + room))]).unwrap();
    let value_sentinel = S::try_from_f64_exact(99.0).unwrap();
    let error_sentinel = S::try_from_f64_exact(77.0).unwrap();
    let mut values = [value_sentinel];
    let mut errors = [error_sentinel];
    let mut calls = 0;
    let report = evaluate(
        &[S::ONE],
        Some(&bounds),
        &mut |x, y| {
            calls += 1;
            y[0] = x[0];
            Ok(())
        },
        &mut values,
        &mut errors,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::StepSizeLimit
    );
    assert!(!report.is_success());
    assert_eq!(report.evaluations(), if after_estimate { 5 } else { 1 });
    assert_eq!(calls, report.evaluations());
    assert_eq!(report.iterations(), if after_estimate { 3 } else { 1 });
    assert_eq!(report.completed_entries(), usize::from(after_estimate));
    assert_eq!(report.output_available(), after_estimate);
    if after_estimate {
        assert!((values[0] - S::ONE).abs() <= S::EPSILON * four);
        assert!(errors[0].is_finite() && errors[0] >= S::ZERO);
        assert_ne!(errors[0], error_sentinel);
    } else {
        assert_eq!(values, [value_sentinel]);
        assert_eq!(errors, [error_sentinel]);
    }

    // A different derivative makes stale publication observable after either
    // kind of exhaustion, rather than merely repeating the previous callback.
    let mut calls = 0;
    let report = evaluate(
        &[S::ZERO],
        None,
        &mut |x, y| {
            calls += 1;
            y[0] = -S::TWO * x[0];
            Ok(())
        },
        &mut values,
        &mut errors,
    )
    .unwrap();
    assert_retry(report, calls, &values, &errors, -S::TWO);

    for fail_at in [1, 6] {
        values.fill(value_sentinel);
        errors.fill(error_sentinel);
        // Nested error diagnostics are allowed to allocate. Test provenance
        // separately, then allocation-track the successful retry.
        let mut injected = Some(SciOxideError::invalid_input(
            scioxide_core::argument!("callback_input"),
            InvalidInputReason::Inconsistent,
            "unrelated callback rejection",
        ));
        let mut calls = 0;
        let error = evaluate(
            &[S::ZERO],
            None,
            &mut |x, y| {
                calls += 1;
                if calls == fail_at {
                    return Err(injected.take().unwrap());
                }
                y[0] = x[0];
                Ok(())
            },
            &mut values,
            &mut errors,
        )
        .unwrap_err();
        assert_eq!(calls, fail_at);
        let SciOxideErrorView::DifferentiationFailure { frames, cause } = error.view() else {
            unreachable!("callback rejection must retain its differentiation frame");
        };
        assert_eq!(frames.len(), 1);

        assert_eq!(frames[0].evaluations, fail_at);
        assert_eq!(
            frames[0].location,
            if fail_at == 1 {
                base_location
            } else {
                refinement_location
            }
        );
        assert_eq!(cause.kind(), SciOxideErrorKind::InvalidInput);
        assert!(matches!(cause.view(), SciOxideErrorView::InvalidInput {

            argument,
            reason: InvalidInputReason::Inconsistent,
            detail: "unrelated callback rejection",
        } if argument == scioxide_core::argument!("callback_input")));
        assert_eq!(values, [value_sentinel]);
        assert_eq!(errors, [error_sentinel]);

        let mut calls = 0;
        let report = evaluate(
            &[S::ZERO],
            None,
            &mut |x, y| {
                calls += 1;
                y[0] = four * x[0];
                Ok(())
            },
            &mut values,
            &mut errors,
        )
        .unwrap();
        assert_retry(report, calls, &values, &errors, four);
    }
}

fn assert_retry<S: RealScalar>(
    report: DifferentiationReport,
    calls: usize,
    values: &[S],
    errors: &[S],
    expected: S,
) {
    assert_eq!(report.termination(), DifferentiationTermination::Converged);
    assert_eq!(report.evaluations(), 7);
    assert_eq!(calls, 7);
    assert_eq!(report.iterations(), 3);
    assert_eq!(report.completed_entries(), 1);
    assert!(report.output_available());
    assert!((values[0] - expected).abs() <= S::EPSILON * expected.abs());
    let allowance = S::EPSILON * expected.abs() * S::try_from_f64_exact(16.0).unwrap();
    assert!(
        errors
            .iter()
            .all(|&error| error.is_finite() && error >= S::ZERO && error <= allowance)
    );
}

fn named_plans<S: RealScalar + NumericScalar<Real = S>>() {
    for after_estimate in [false, true] {
        let method = exhaustion_options(after_estimate);
        // The scalar first derivative through a workspace is the one-coordinate gradient.
        let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
        let mut workspace = AdaptiveDifferenceWorkspace::workspace(request).unwrap();
        exhaustion_and_reuse(
            after_estimate,
            DifferentiationLocation::base_point(),
            DifferentiationLocation::variable(0),
            |point, bounds, callback, values, errors| {
                method.gradient_into(
                    callback,
                    point,
                    bounds,
                    values,
                    errors,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
            },
        );
        let request = DerivativeRequest::new(&method, DerivativeShape::jvp(1, 1));
        let mut workspace = AdaptiveDifferenceWorkspace::workspace(request).unwrap();
        exhaustion_and_reuse(
            after_estimate,
            DifferentiationLocation::direction(),
            DifferentiationLocation::direction(),
            |point, bounds, callback, values, errors| {
                method.jvp_into(
                    callback,
                    point,
                    &[S::ONE],
                    bounds,
                    values,
                    errors,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
            },
        );
        let schedule = ColoringSchedule::new(
            StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
            ColoringKind::DerivativeColumns,
            vec![0],
        )
        .unwrap();
        let plan = ColoredJacobianPlan::new(&schedule, method).unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        exhaustion_and_reuse(
            after_estimate,
            DifferentiationLocation::base_point(),
            DifferentiationLocation::variable(0),
            |point, bounds, callback, values, errors| {
                plan.evaluate_into(
                    callback,
                    point,
                    bounds,
                    values,
                    errors,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
            },
        );
    }
}

#[test]
fn named_plans_exhaust_and_retry_with_reused_storage_f32() {
    named_plans::<f32>();
}

#[test]
fn named_plans_exhaust_and_retry_with_reused_storage_f64() {
    named_plans::<f64>();
}

fn colored_unavailable_entry<S: RealScalar + NumericScalar<Real = S>>() {
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 1],
    )
    .unwrap();
    let plan = ColoredJacobianPlan::new(
        &schedule,
        AdaptiveDifference::default().with_method(AdaptiveDifferenceMethod::Richardson),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let bounds = Bounds::new(vec![(None, None), (Some(S::ONE), Some(S::ONE))]).unwrap();
    let value_sentinel = S::try_from_f64_exact(99.0).unwrap();
    let error_sentinel = S::try_from_f64_exact(77.0).unwrap();
    let mut values = [value_sentinel; 2];
    let mut errors = [error_sentinel; 2];
    let mut calls = 0;
    let report = plan
        .evaluate_into(
            |x: &[S], y: &mut [S]| {
                calls += 1;
                assert_eq!(x[1], S::ONE);
                y[0] = x[0] * x[0];
                y[1] = x[1] * x[1];
                Ok(())
            },
            &[S::ONE; 2],
            Some(&bounds),
            &mut values,
            &mut errors,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::StepSizeLimit
    );
    assert_eq!(report.evaluations(), 7);
    assert_eq!(calls, 7);
    assert_eq!(report.iterations(), 4);
    assert_eq!(report.completed_entries(), 1);
    assert!(!report.output_available());
    assert_eq!(values, [value_sentinel; 2]);
    assert_eq!(errors, [error_sentinel; 2]);
    let report = plan
        .evaluate_into(
            |x: &[S], y: &mut [S]| {
                y[0] = S::TWO * x[0];
                y[1] = -S::TWO * x[1];
                Ok(())
            },
            &[S::ZERO; 2],
            None,
            &mut values,
            &mut errors,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Converged);
    assert_eq!(report.evaluations(), 13);
    assert_eq!(report.iterations(), 6);
    assert_eq!(report.completed_entries(), 2);
    assert!(report.output_available());
    assert_eq!(values, [S::TWO, -S::TWO]);
    let allowance = S::EPSILON * S::try_from_f64_exact(16.0).unwrap();
    assert!(
        errors
            .iter()
            .all(|&error| error.is_finite() && error >= S::ZERO && error <= allowance)
    );
}

#[test]
fn colored_unavailable_entry_and_retry_are_allocation_free_f32() {
    colored_unavailable_entry::<f32>();
}

#[test]
fn colored_unavailable_entry_and_retry_are_allocation_free_f64() {
    colored_unavailable_entry::<f64>();
}

/// Runs `operation` in its `_into` form and, where it takes the inputs, its
/// owned form, checks that neither calls the function nor writes the output,
/// and returns their errors. `First` and `Second` are the one-coordinate
/// gradient and Hessian, and `First` also runs the scalar forms.
fn stencil_errors<S: RealScalar>(
    method: &FiniteDifference<S>,
    operation: DerivativeOperation,
    point: &[S],
    bounds: Option<&Bounds<S>>,
) -> Vec<SciOxideError> {
    let n = point.len();
    let shape = match operation {
        DerivativeOperation::First | DerivativeOperation::Gradient => DerivativeShape::gradient(n),
        DerivativeOperation::Jacobian => DerivativeShape::jacobian(n, 1),
        DerivativeOperation::Second | DerivativeOperation::Hessian => DerivativeShape::hessian(n),
    };
    let request = DerivativeRequest::new(method, shape);
    let mut workspace = FiniteDifferenceWorkspace::workspace(request).unwrap();
    let sentinel = S::try_from_f64_exact(99.0).unwrap();
    let mut values = vec![sentinel; shape.output_len().unwrap()];
    let mut calls = 0;
    let mut function = |_: &[S], y: &mut [S]| {
        calls += 1;
        y.fill(S::ZERO);
        Ok(())
    };
    let context = ExecutionContext::sequential;
    let (into, owned) = match operation {
        DerivativeOperation::First | DerivativeOperation::Gradient => (
            method.gradient_into(
                &mut function,
                point,
                bounds,
                None,
                &mut values,
                &mut workspace,
                context(),
            ),
            method.gradient(&mut function, point, bounds, None),
        ),
        DerivativeOperation::Jacobian => (
            method.jacobian_into(
                &mut function,
                point,
                1,
                bounds,
                None,
                &mut values,
                &mut workspace,
                context(),
            ),
            method.jacobian(&mut function, point, 1, bounds, None),
        ),
        DerivativeOperation::Second | DerivativeOperation::Hessian => (
            method.hessian_into(
                &mut function,
                point,
                bounds,
                None,
                &mut values,
                &mut workspace,
                context(),
            ),
            method.hessian(&mut function, point, bounds, None),
        ),
    };
    let mut errors = vec![into.unwrap_err(), owned.unwrap_err()];
    if operation == DerivativeOperation::First {
        let mut scalar = |_: S| {
            calls += 1;
            Ok(S::ZERO)
        };
        let interval = bounds.map_or_else(ExtendedInterval::entire, |bounds| {
            let (lower, upper) = bounds.as_slice()[0];
            ExtendedInterval::new(lower, upper).unwrap()
        });
        let x = point[0];
        let out = &mut values[0];
        let error = method.derivative_into(&mut scalar, x, interval, None, out, context());
        errors.push(error.unwrap_err());
        if bounds.is_none() {
            errors.push(method.derivative(&mut scalar, x).unwrap_err());
        }
    }
    assert_eq!(calls, 0);
    assert!(values.iter().all(|&value| value == sentinel));
    errors
}

const OPERATIONS: [DerivativeOperation; 5] = [
    DerivativeOperation::First,
    DerivativeOperation::Second,
    DerivativeOperation::Gradient,
    DerivativeOperation::Jacobian,
    DerivativeOperation::Hessian,
];

fn fixed_error_compatibility<S: RealScalar>() {
    for operation in OPERATIONS {
        let inputs = if matches!(
            operation,
            DerivativeOperation::First | DerivativeOperation::Second
        ) {
            1
        } else {
            2
        };
        let variable = inputs - 1;
        for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
            let method = FiniteDifference::default().with_scheme(scheme);
            let point = vec![S::ONE; inputs];
            let mut intervals = vec![(None, None); inputs];
            intervals[variable] = (Some(S::ONE), Some(S::ONE));
            let bounds = Bounds::new(intervals).unwrap();
            for error in stencil_errors(&method, operation, &point, Some(&bounds)) {
                assert_fixed_error(&error, scioxide_core::argument!("bounds"), variable);
            }
        }
    }
}

// A step that rounds away at `x = 1` leaves no stencil although no bound
// restricts the coordinate, so the error names the point.
fn rounded_step_error<S: RealScalar>(step: S) {
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        let method = FiniteDifference::default()
            .with_scheme(scheme)
            .with_step(DifferenceStep::absolute(step).unwrap());
        for operation in OPERATIONS {
            for error in stencil_errors(&method, operation, &[S::ONE], None) {
                assert_fixed_error(&error, scioxide_core::argument!("x"), 0);
            }
        }
    }
}

fn assert_fixed_error(error: &SciOxideError, expected: scioxide_core::Argument, variable: usize) {
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    let SciOxideErrorView::InvalidInput {
        argument,
        reason,
        detail,
    } = error.view()
    else {
        unreachable!("fixed-step unavailable geometry must remain InvalidInput");
    };

    assert_eq!(argument, expected);
    assert_eq!(reason, InvalidInputReason::UnsupportedConfiguration);
    assert_eq!(
        detail,
        format!("no representable differentiation stencil exists for variable {variable}")
    );
}

#[test]
fn fixed_step_unavailable_stencil_errors_are_unchanged_f32() {
    fixed_error_compatibility::<f32>();
}

#[test]
fn fixed_step_unavailable_stencil_errors_are_unchanged_f64() {
    fixed_error_compatibility::<f64>();
}

#[test]
fn rounded_away_steps_without_bounds_name_the_point_f32() {
    rounded_step_error(1e-30_f32);
}

#[test]
fn rounded_away_steps_without_bounds_name_the_point_f64() {
    rounded_step_error(1e-300_f64);
}
