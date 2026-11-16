//! Retained-capacity limits and reuse across prepared adaptive shapes.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Capacity, ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_differentiate::{
    AdaptiveDifference, AdaptiveDifferenceWorkspace, DerivativeOperation, DerivativeRequest,
    DerivativeShape,
};

#[test]
fn cached_adaptive_requirements_follow_operation_levels_and_limits() {
    let mut reused = AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(
        &AdaptiveDifference::default().with_max_levels(16),
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    // A Hessian also keeps a sample cache, which a gradient workspace lacks.
    reused
        .try_reserve(
            &AdaptiveDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(
                &AdaptiveDifference::default().with_max_levels(3),
                DerivativeShape::hessian(1),
            ))
            .unwrap(),
        )
        .unwrap();
    for (shape, levels) in [
        (DerivativeShape::hessian(1), 3),
        (DerivativeShape::gradient(1), 12),
        (DerivativeShape::gradient(1), 6),
    ] {
        let method = AdaptiveDifference::default().with_max_levels(levels);
        let mut fresh =
            AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap();
        let evaluate = |workspace: &mut AdaptiveDifferenceWorkspace<f64>| {
            let mut output = [99.0];
            let mut error = [99.0];
            let function = |x: &[f64], y: &mut [f64]| {
                y[0] = x[0].sin();
                Ok(())
            };
            let report = if shape == DerivativeShape::hessian(1) {
                method.hessian_into(
                    function,
                    &[1.0],
                    None,
                    &mut output,
                    &mut error,
                    workspace,
                    ExecutionContext::sequential(),
                )
            } else {
                method.gradient_into(
                    function,
                    &[1.0],
                    None,
                    &mut output,
                    &mut error,
                    workspace,
                    ExecutionContext::sequential(),
                )
            }
            .unwrap();
            (report, output[0], error[0])
        };
        let expected = evaluate(&mut fresh);
        let actual = evaluate(&mut reused);
        assert_eq!(actual.0.termination(), expected.0.termination());
        assert_eq!(actual.0.evaluations(), expected.0.evaluations());
        assert_eq!(actual.0.iterations(), expected.0.iterations());
        assert!((actual.1 - expected.1).abs() <= f64::EPSILON * expected.1.abs());
        assert!((actual.2 - expected.2).abs() <= f64::EPSILON * expected.2.abs());
    }
    let capped = AdaptiveDifference::<f64>::default()
        .with_max_levels(6)
        .with_max_workspace_bytes(0);
    let mut output = [99.0];
    let mut error = [99.0];
    let mut calls = 0;
    let failure = capped
        .gradient_into(
            |_: &[f64], _: &mut [f64]| {
                calls += 1;
                Ok(())
            },
            &[1.0],
            None,
            &mut output,
            &mut error,
            &mut reused,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(failure.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(calls, 0);
    assert_eq!((output, error), ([99.0], [99.0]));
}

#[test]
fn adaptive_reuse_checks_retained_capacity_and_does_not_grow_during_evaluation() {
    let method = AdaptiveDifference::<f64>::default();
    let required = AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(
        DerivativeRequest::new(&method, DerivativeShape::gradient(3)),
    )
    .unwrap();
    let mut work = AdaptiveDifferenceWorkspace::default();
    assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
    work.try_reserve(&required).unwrap();
    assert_eq!(work.check(&required).unwrap(), Capacity::Sufficient);
    let small_required = AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(
        DerivativeRequest::new(&method, DerivativeShape::gradient(2)),
    )
    .unwrap();
    assert_eq!(work.check(&small_required).unwrap(), Capacity::Sufficient);
    let mut values = [77.0; 2];
    let mut errors = [88.0; 2];
    {
        let report = method
            .gradient_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    Ok(())
                },
                &[1.0, 2.0],
                None,
                &mut values,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(report.completed_entries(), 2);
    };
    for (actual, expected) in values.into_iter().zip([2.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-8);
    }
    let limited = method.with_max_workspace_bytes(small_required.bytes());
    let limited_required = AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(
        DerivativeRequest::new(&limited, DerivativeShape::gradient(2)),
    )
    .unwrap();
    {
        assert_eq!(
            work.check(&limited_required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            work.try_reserve(&limited_required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
    let capped = AdaptiveDifference::<f64>::default().with_max_workspace_bytes(0);
    {
        assert_eq!(
            AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(DerivativeRequest::new(
                &capped,
                DerivativeShape::hessian(4),
            ))
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
}

#[test]
fn fixed_workspace_reuses_capacity_and_cached_base_without_sampling_again() {
    use scioxide_differentiate::{FiniteDifference, FiniteDifferenceWorkspace};
    let method = FiniteDifference::<f64>::default();
    let requirements = |method: &FiniteDifference<f64>, inputs| {
        FiniteDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(
            method,
            DerivativeShape::gradient(inputs),
        ))
    };
    let mut work = FiniteDifferenceWorkspace::default();
    let required = requirements(&method, 3).unwrap();
    assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
    work.try_reserve(&required).unwrap();
    assert_eq!(
        work.check(&requirements(&method, 2).unwrap()).unwrap(),
        Capacity::Sufficient
    );
    let mut values = [77.0; 2];
    let mut calls = 0;
    {
        let report = method
            .gradient_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x[0] + x[1];
                    Ok(())
                },
                &[1.0, 2.0],
                None,
                Some(3.0),
                &mut values,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.evaluations(), 4);
        assert_eq!(report.completed_entries(), 2);
        assert!(report.output_available());
    };
    assert_eq!(calls, 4);
    for actual in values {
        assert!((actual - 1.0).abs() < 1e-8);
    }
    let bytes = requirements(&method, 2).unwrap().bytes();
    let capped = FiniteDifference::<f64>::default().with_max_workspace_bytes(bytes);
    {
        let required = requirements(&capped, 2).unwrap();
        assert_eq!(
            work.check(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            work.try_reserve(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{actual:?} != {expected:?}");
    }
}

// f(x) = x₀²x₁ has gradient (2x₀x₁, x₀²) and Hessian [[2x₁, 2x₀], [2x₀, 0]];
// F(x) = (x₀²x₁, x₀ + x₁²) has Jacobian [[2x₀x₁, x₀²], [1, 2x₁]].
const X: [f64; 2] = [1.0, 2.0];
const DIRECTION: [f64; 2] = [1.0, -1.0];
const GRADIENT: [f64; 2] = [4.0, 1.0];
const HESSIAN: [f64; 4] = [4.0, 2.0, 2.0, 0.0];
const JACOBIAN: [f64; 4] = [4.0, 1.0, 1.0, 4.0];
const JVP: [f64; 2] = [3.0, -3.0];

fn scalar_field(x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    y[0] = x[0] * x[0] * x[1];
    Ok(())
}
fn vector_field(x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    y[0] = x[0] * x[0] * x[1];
    y[1] = x[0] + x[1] * x[1];
    Ok(())
}

#[test]
fn fixed_into_forms_execute_with_shared_and_per_coordinate_steps() {
    use scioxide_core::ExtendedInterval;
    use scioxide_differentiate::{DifferenceStep, FiniteDifference, FiniteDifferenceWorkspace};
    for steps in [None, Some([1e-4, 2e-4])] {
        let method = |dimension: usize| {
            let method = FiniteDifference::<f64>::default();
            match steps {
                Some(steps) => method
                    .with_step(DifferenceStep::absolute_coordinates(&steps[..dimension]).unwrap()),
                None => method,
            }
        };
        let scalar = method(1);
        let mut slope = 99.0;
        let report = scalar
            .derivative_into(
                |t: f64| Ok(t * t * t),
                1.0,
                ExtendedInterval::entire(),
                None,
                &mut slope,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert!((slope - 3.0).abs() <= 1e-6);
        let method = method(2);
        let workspace = |shape| {
            FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap()
        };
        let mut out = [99.0; 4];
        let mut work = workspace(DerivativeShape::gradient(2));
        let report = method
            .gradient_into(
                scalar_field,
                &X,
                None,
                None,
                &mut out[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out[..2], &GRADIENT, 1e-6);
        let mut work = workspace(DerivativeShape::hessian(2));
        let report = method
            .hessian_into(
                scalar_field,
                &X,
                None,
                None,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out, &HESSIAN, 1e-6);
        let mut work = workspace(DerivativeShape::jacobian(2, 2));
        let report = method
            .jacobian_into(
                vector_field,
                &X,
                2,
                None,
                None,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out, &JACOBIAN, 1e-6);
        let mut work = workspace(DerivativeShape::jvp(2, 2));
        let report = method
            .jvp_into(
                vector_field,
                &X,
                &DIRECTION,
                None,
                None,
                &mut out[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out[..2], &JVP, 1e-6);
    }
}

#[test]
fn adaptive_into_forms_execute_with_shared_and_per_coordinate_steps() {
    use scioxide_differentiate::DifferenceStep;
    for steps in [None, Some([0.25, 0.5])] {
        let method = AdaptiveDifference::<f64>::default();
        let method = match steps {
            Some(steps) => method.with_step(DifferenceStep::absolute_coordinates(&steps).unwrap()),
            None => method,
        };
        let workspace = |shape| {
            AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(&method, shape))
                .unwrap()
        };
        let mut out = [99.0; 4];
        let mut errors = [99.0; 4];
        let mut work = workspace(DerivativeShape::gradient(2));
        let report = method
            .gradient_into(
                scalar_field,
                &X,
                None,
                &mut out[..2],
                &mut errors[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_close(&out[..2], &GRADIENT, 1e-6);
        let mut work = workspace(DerivativeShape::hessian(2));
        let report = method
            .hessian_into(
                scalar_field,
                &X,
                None,
                &mut out,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_close(&out, &HESSIAN, 1e-6);
        let mut work = workspace(DerivativeShape::jacobian(2, 2));
        let report = method
            .jacobian_into(
                vector_field,
                &X,
                2,
                None,
                &mut out,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_close(&out, &JACOBIAN, 1e-6);
        let mut work = workspace(DerivativeShape::jvp(2, 2));
        let report = method
            .jvp_into(
                vector_field,
                &X,
                &DIRECTION,
                None,
                &mut out[..2],
                &mut errors[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_close(&out[..2], &JVP, 1e-6);
    }
}

#[test]
fn complex_step_into_forms_execute() {
    use scioxide_core::Complex64;
    use scioxide_differentiate::{ComplexStep, ComplexStepWorkspace};
    let scalar = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] * x[1];
        Ok(())
    };
    let vector = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] * x[1];
        y[1] = x[0] + x[1] * x[1];
        Ok(())
    };
    for method in [
        ComplexStep::default(),
        ComplexStep::default().with_step(1e-100),
    ] {
        let workspace = |shape| {
            ComplexStepWorkspace::workspace(DerivativeRequest::new(&method, shape)).unwrap()
        };
        let mut out = [99.0; 4];
        let mut work = workspace(DerivativeShape::gradient(2));
        let report = method
            .gradient_into(
                scalar,
                &X,
                &mut out[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out[..2], &GRADIENT, 1e-12);
        let mut work = workspace(DerivativeShape::jacobian(2, 2));
        let report = method
            .jacobian_into(
                vector,
                &X,
                2,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out, &JACOBIAN, 1e-12);
        let mut work = workspace(DerivativeShape::jvp(2, 2));
        let report = method
            .jvp_into(
                vector,
                &X,
                &DIRECTION,
                &mut out[..2],
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_close(&out[..2], &JVP, 1e-12);
    }
}

#[test]
fn one_workspace_serves_every_reserved_shape_of_its_method_family() {
    use scioxide_core::Complex64;
    use scioxide_differentiate::{
        ComplexStep, ComplexStepWorkspace, FiniteDifference, FiniteDifferenceWorkspace,
    };
    let shapes = [
        DerivativeShape::gradient(2),
        DerivativeShape::jacobian(2, 2),
        DerivativeShape::hessian(2),
        DerivativeShape::jvp(2, 2),
    ];
    let mut out = [99.0; 4];
    let mut errors = [99.0; 4];

    let fixed = FiniteDifference::<f64>::default();
    let required = shapes.map(|shape| {
        FiniteDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(&fixed, shape))
            .unwrap()
    });
    let mut work = FiniteDifferenceWorkspace::default();
    for r in &required {
        work.try_reserve(r).unwrap();
    }
    for r in &required {
        assert_eq!(work.check(r).unwrap(), Capacity::Sufficient);
    }
    for _ in 0..2 {
        let cx = || ExecutionContext::sequential();
        fixed
            .gradient_into(scalar_field, &X, None, None, &mut out[..2], &mut work, cx())
            .unwrap();
        assert_close(&out[..2], &GRADIENT, 1e-6);
        fixed
            .jacobian_into(vector_field, &X, 2, None, None, &mut out, &mut work, cx())
            .unwrap();
        assert_close(&out, &JACOBIAN, 1e-6);
        fixed
            .hessian_into(scalar_field, &X, None, None, &mut out, &mut work, cx())
            .unwrap();
        assert_close(&out, &HESSIAN, 1e-6);
        fixed
            .jvp_into(
                vector_field,
                &X,
                &DIRECTION,
                None,
                None,
                &mut out[..2],
                &mut work,
                cx(),
            )
            .unwrap();
        assert_close(&out[..2], &JVP, 1e-6);
    }

    let adaptive = AdaptiveDifference::<f64>::default();
    let required = shapes.map(|shape| {
        AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(DerivativeRequest::new(
            &adaptive, shape,
        ))
        .unwrap()
    });
    let mut work = AdaptiveDifferenceWorkspace::default();
    for r in &required {
        work.try_reserve(r).unwrap();
    }
    for r in &required {
        assert_eq!(work.check(r).unwrap(), Capacity::Sufficient);
    }
    for _ in 0..2 {
        let cx = || ExecutionContext::sequential();
        adaptive
            .gradient_into(
                scalar_field,
                &X,
                None,
                &mut out[..2],
                &mut errors[..2],
                &mut work,
                cx(),
            )
            .unwrap();
        assert_close(&out[..2], &GRADIENT, 1e-6);
        adaptive
            .jacobian_into(
                vector_field,
                &X,
                2,
                None,
                &mut out,
                &mut errors,
                &mut work,
                cx(),
            )
            .unwrap();
        assert_close(&out, &JACOBIAN, 1e-6);
        adaptive
            .hessian_into(
                scalar_field,
                &X,
                None,
                &mut out,
                &mut errors,
                &mut work,
                cx(),
            )
            .unwrap();
        assert_close(&out, &HESSIAN, 1e-6);
        adaptive
            .jvp_into(
                vector_field,
                &X,
                &DIRECTION,
                None,
                &mut out[..2],
                &mut errors[..2],
                &mut work,
                cx(),
            )
            .unwrap();
        assert_close(&out[..2], &JVP, 1e-6);
    }

    let complex = ComplexStep::<f64>::default();
    let scalar = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] * x[1];
        Ok(())
    };
    let vector = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] * x[1];
        y[1] = x[0] + x[1] * x[1];
        Ok(())
    };
    let shapes = [shapes[0], shapes[1], shapes[3]];
    let required = shapes.map(|shape| {
        ComplexStepWorkspace::workspace_requirements(DerivativeRequest::new(&complex, shape))
            .unwrap()
    });
    let mut work = ComplexStepWorkspace::default();
    for r in &required {
        work.try_reserve(r).unwrap();
    }
    for r in &required {
        assert_eq!(work.check(r).unwrap(), Capacity::Sufficient);
    }
    for _ in 0..2 {
        let cx = || ExecutionContext::sequential();
        complex
            .gradient_into(scalar, &X, &mut out[..2], &mut work, cx())
            .unwrap();
        assert_close(&out[..2], &GRADIENT, 1e-12);
        complex
            .jacobian_into(vector, &X, 2, &mut out, &mut work, cx())
            .unwrap();
        assert_close(&out, &JACOBIAN, 1e-12);
        complex
            .jvp_into(vector, &X, &DIRECTION, &mut out[..2], &mut work, cx())
            .unwrap();
        assert_close(&out[..2], &JVP, 1e-12);
    }
}

#[test]
fn complex_step_workspace_rejects_a_hessian_request() {
    use scioxide_core::{InvalidInputReason, SciOxideErrorView};
    use scioxide_differentiate::{ComplexStep, ComplexStepWorkspace};
    let method = ComplexStep::<f64>::default();
    let request = DerivativeRequest::new(&method, DerivativeShape::hessian(2));
    let errors = [
        ComplexStepWorkspace::workspace_requirements(request).unwrap_err(),
        ComplexStepWorkspace::workspace(request).unwrap_err(),
    ];
    for error in errors {
        assert!(
            matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason, .. }
                if argument == scioxide_core::argument!("request")
                    && reason == InvalidInputReason::UnsupportedConfiguration),
            "{error:?}"
        );
    }
}

#[test]
fn node_workspaces_reuse_geometry_and_enforce_aggregate_retained_limits() {
    use scioxide_differentiate::{
        NodeDifferenceOptions, PhysicalDifferencePlan, RelativeDifference,
    };
    let nodes = [-0.5, 0.0, 0.5];
    let large = PhysicalDifferencePlan::new(
        DerivativeOperation::Hessian,
        &[0.0; 3],
        &[nodes.as_slice(); 3],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let small = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[0.0; 2],
        &[nodes.as_slice(); 2],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut physical = large.workspace::<f64>(()).unwrap();
    let small_required = small.workspace_requirements::<f64>(()).unwrap();
    assert_eq!(
        physical.check(&small_required).unwrap(),
        Capacity::Sufficient
    );
    let mut output = [99.0; 2];
    {
        let report = small
            .evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] + 2.0 * x[1];
                    Ok(())
                },
                None,
                &mut output,
                &mut physical,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_eq!(report.completed_entries(), 2);
    };
    assert_eq!(output, [1.0, 2.0]);
    let capped = PhysicalDifferencePlan::new(
        DerivativeOperation::Gradient,
        &[0.0; 2],
        &[nodes.as_slice(); 2],
        1,
        NodeDifferenceOptions::default().with_max_workspace_bytes(small_required.bytes()),
    )
    .unwrap();
    let required = capped.workspace_requirements::<f64>(()).unwrap();
    {
        assert_eq!(
            physical.check(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            physical.try_reserve(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
    let large = RelativeDifference::new(
        DerivativeOperation::Hessian,
        &[nodes.as_slice(); 3],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let small_nodes = [-0.25, 0.0, 0.25];
    let small = RelativeDifference::new(
        DerivativeOperation::Gradient,
        &[small_nodes.as_slice(); 2],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut relative = large.workspace::<f64>(()).unwrap();
    let small_required = small.workspace_requirements::<f64>(()).unwrap();
    assert_eq!(
        relative.check(&small_required).unwrap(),
        Capacity::Sufficient
    );
    {
        let report = small
            .evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0] + x[1] * x[1];
                    Ok(())
                },
                &[1.0, 2.0],
                None,
                &mut output,
                &mut relative,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
    };
    assert_eq!(output, [2.0, 4.0]);
    let capped = RelativeDifference::new(
        DerivativeOperation::Gradient,
        &[small_nodes.as_slice(); 2],
        1,
        NodeDifferenceOptions::default().with_max_workspace_bytes(small_required.bytes()),
    )
    .unwrap();
    let required = capped.workspace_requirements::<f64>(()).unwrap();
    {
        assert_eq!(
            relative.check(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            relative.try_reserve(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
}

#[test]
fn node_contextual_callbacks_share_the_live_scope_and_validate_before_sampling() {
    use scioxide_core::{Bounds, EvaluationRole, SolveContext};
    use scioxide_differentiate::{
        Contextual, NodeDifferenceOptions, PhysicalDifferencePlan, RelativeDifference,
    };
    let nodes = [-0.5, 0.0, 0.5];
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[0.0],
        &[&nodes],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut work = physical.workspace::<f64>(()).unwrap();
    let mut out = [99.0];
    let mut calls = 0;
    let callback = Contextual(|x: &[f64], y: &mut [f64], cx: &mut SolveContext<'_, '_>| {
        calls += 1;
        assert!(
            cx.invoke_provider(EvaluationRole::Residual, || Ok(()))?
                .is_evaluated()
        );
        assert_eq!(cx.counts().own().provider(EvaluationRole::Residual), calls);
        y[0] = x[0];
        Ok(())
    });
    let report = physical
        .evaluate_into(
            callback,
            None,
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 3);
    assert_eq!(calls, 3);
    assert_eq!(out, [1.0]);
    let before = out;
    let mut invalid_calls = 0;
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    assert!(
        physical
            .evaluate_into(
                |_: &[f64], value: &mut [f64]| {
                    invalid_calls += 1;
                    value[0] = 1.0;
                    Ok(())
                },
                Some(&bounds),
                &mut out,
                &mut work,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(out, before);
    assert_eq!(invalid_calls, 0);
    let relative = RelativeDifference::new(
        DerivativeOperation::First,
        &[&nodes],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let mut work = relative.workspace::<f64>(()).unwrap();
    let report = relative
        .evaluate_into(
            Contextual(|x: &[f64], y: &mut [f64], cx: &mut SolveContext<'_, '_>| {
                assert!(cx.counts().own().provider(EvaluationRole::Function) > 0);
                y[0] = x[0] * x[0];
                Ok(())
            }),
            &[2.0],
            None,
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 3);
    assert_eq!(out, [4.0]);
}

#[test]
fn complex_workspace_reuses_relative_and_physical_geometry_without_growth() {
    use scioxide_core::Complex64;
    use scioxide_differentiate::{
        ComplexStep, ComplexStepPlan, ComplexStepWorkspace, RelativeComplexStep,
    };
    let nodes = [-0.5, 0.0, 0.5];
    let method = ComplexStep::default().with_step(1e-100);
    let large = RelativeComplexStep::new(&[nodes.as_slice(); 3], method).unwrap();
    let mut work = ComplexStepWorkspace::default();
    let large_required = large.workspace_requirements(()).unwrap();
    assert_eq!(work.check(&large_required).unwrap(), Capacity::NeedsGrowth);
    work.try_reserve(&large_required).unwrap();
    let small = ComplexStepPlan::new(&[0.0], &[&nodes], method).unwrap();
    let small_required = small.workspace_requirements(()).unwrap();
    assert_eq!(work.check(&small_required).unwrap(), Capacity::Sufficient);
    let mut out = [99.0];
    {
        let report = small
            .evaluate_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                None,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        assert_eq!(report.evaluations(), 4);
        assert_eq!(report.completed_entries(), 1);
    };
    assert_eq!(out, [2.0]);
    let capped = ComplexStepPlan::new(
        &[0.0],
        &[&nodes],
        method.with_max_workspace_bytes(small_required.bytes()),
    )
    .unwrap();
    let required = capped.workspace_requirements(()).unwrap();
    {
        assert_eq!(
            work.check(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            work.try_reserve(&required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    };
    let relative = RelativeComplexStep::new(&[&nodes], method).unwrap();
    {
        let report = relative
            .evaluate_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &[1.0],
                None,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
    };
    assert_eq!(out, [2.0]);
}

#[test]
fn complex_contextual_callback_retains_analytic_contract_and_scope_accounting() {
    use scioxide_core::{Complex64, EvaluationRole, SolveContext};
    use scioxide_differentiate::{ComplexStep, ComplexStepWorkspace, Contextual};
    let method = ComplexStep::default().with_step(1e-100);
    let mut work = ComplexStepWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    let mut out = [99.0];
    let mut calls = 0;
    let report = method
        .gradient_into(
            Contextual(
                |x: &[Complex64], y: &mut [Complex64], cx: &mut SolveContext<'_, '_>| {
                    calls += 1;
                    assert!(
                        cx.invoke_provider(EvaluationRole::Residual, || Ok(()))?
                            .is_evaluated()
                    );
                    assert_eq!(cx.counts().own().provider(EvaluationRole::Residual), calls);
                    y[0] = x[0] * x[0];
                    Ok(())
                },
            ),
            &[2.0],
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 2);
    assert_eq!(calls, 2);
    assert_eq!(out, [4.0]);
    let error = method
        .gradient_into(
            Contextual(
                |_: &[Complex64], y: &mut [Complex64], _: &mut SolveContext<'_, '_>| {
                    y[0] = Complex64::new(1.0, 1.0);
                    Ok(())
                },
            ),
            &[2.0],
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(out, [4.0]);
}

#[test]
fn colored_reuse_charges_unused_children_and_rejects_before_publication() {
    use scioxide_core::{ColoringKind, ColoringSchedule, StructuralPattern};
    use scioxide_differentiate::{ColoredJacobianPlan, ColoredJacobianWorkspace};
    let large_schedule = ColoringSchedule::new(
        StructuralPattern::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0, 0],
    )
    .unwrap();
    let small_schedule = ColoringSchedule::new(
        StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0],
    )
    .unwrap();
    let large =
        ColoredJacobianPlan::new(&large_schedule, AdaptiveDifference::<f64>::default()).unwrap();
    let small = ColoredJacobianPlan::new(
        &small_schedule,
        AdaptiveDifference::<f64>::default().with_max_levels(3),
    )
    .unwrap();
    let mut work = large.workspace::<f64>(()).unwrap();
    let required = small.workspace_requirements::<f64>(()).unwrap();
    let mut output = [99.0];
    let mut errors = [88.0];
    {
        work.try_reserve(&required).unwrap();
        assert_eq!(work.check(&required).unwrap(), Capacity::Sufficient);
        let report = small
            .evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                },
                &[1.0],
                None,
                &mut output,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
    };
    assert!((output[0] - 2.0).abs() < f64::EPSILON.sqrt());
    let capped = ColoredJacobianPlan::new(
        &small_schedule,
        AdaptiveDifference::<f64>::default()
            .with_max_levels(3)
            .with_max_workspace_bytes(required.bytes()),
    )
    .unwrap();
    let capped_required = capped.workspace_requirements::<f64>(()).unwrap();
    output = [99.0];
    errors = [88.0];
    let mut calls = 0;
    {
        assert_eq!(
            work.check(&capped_required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            work.try_reserve(&capped_required).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let error = capped
            .evaluate_into(
                |_: &[f64], _: &mut [f64]| {
                    calls += 1;
                    Ok(())
                },
                &[1.0],
                None,
                &mut output,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
    };
    assert_eq!(calls, 0);
    assert_eq!(output, [99.0]);
    assert_eq!(errors, [88.0]);
    let mut fresh = ColoredJacobianWorkspace::<f64>::default();
    fresh.try_reserve(&capped_required).unwrap();
    assert_eq!(fresh.check(&capped_required).unwrap(), Capacity::Sufficient);
    {
        assert_eq!(
            work.check(&large.workspace_requirements::<f64>(()).unwrap())
                .unwrap(),
            Capacity::Sufficient
        );
        let mut values = [99.0; 3];
        let mut errors = [88.0; 3];
        let report = large
            .evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    for (x, y) in x.iter().zip(y) {
                        *y = x * x;
                    }
                    Ok(())
                },
                &[1.0, 2.0, 3.0],
                None,
                &mut values,
                &mut errors,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.output_available());
        for (actual, expected) in values.into_iter().zip([2.0, 4.0, 6.0]) {
            assert!((actual - expected).abs() < f64::EPSILON.sqrt());
        }
    };
}

#[test]
fn restarted_table_reuses_prepared_storage_without_allocating() {
    let method = AdaptiveDifference::default();
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
    let requirements = AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(request).unwrap();
    let mut workspace = AdaptiveDifferenceWorkspace::<f64>::workspace(request).unwrap();
    assert_eq!(workspace.retained_bytes().unwrap(), requirements.bytes());
    let bounds = scioxide_core::Bounds::new(vec![(Some(0.0), None)]).unwrap();
    for x in [0.05, 1.0, 0.001, 0.05] {
        let mut values = [0.0];
        let mut errors = [0.0];
        let report = method
            .gradient_into(
                |point: &[f64], y: &mut [f64]| {
                    y[0] = point[0].exp();
                    Ok(())
                },
                &[x],
                Some(&bounds),
                &mut values,
                &mut errors,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((values[0] - x.exp()).abs() < 1e-10);
    }
}
