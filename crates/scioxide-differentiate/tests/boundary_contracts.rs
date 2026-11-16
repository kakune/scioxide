//! Boundary accuracy, sample counts and complex-step range contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Bounds, Budget, BudgetKind, ColoringKind, ColoringSchedule, Complex64, ExecutionContext,
    InvalidInputReason, NumericalIssue, SciOxideError, SciOxideErrorView, SolveBudgets,
    StructuralPattern,
};
use scioxide_differentiate::*;

fn exp(x: &[f64], y: &mut [f64]) -> Result<(), SciOxideError> {
    y[0] = x[0].exp();
    Ok(())
}
fn bound(sign: f64) -> Bounds<f64> {
    Bounds::new(vec![if sign > 0.0 {
        (Some(0.0), None)
    } else {
        (None, Some(0.0))
    }])
    .unwrap()
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected}, tolerance {tolerance}"
    );
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
fn exponential_derivatives_on_and_near_both_bounds() {
    for sign in [-1.0, 1.0] {
        let bounds = bound(sign);
        for d in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4] {
            let x = sign * d;
            let sample = |point: &[f64], y: &mut [f64]| {
                assert!(sign * point[0] >= 0.0);
                exp(point, y)
            };
            for method in [
                AdaptiveDifferenceMethod::Ridders,
                AdaptiveDifferenceMethod::Richardson,
            ] {
                let adaptive = AdaptiveDifference::default().with_method(method);
                let g = adaptive.gradient(sample, &[x], Some(&bounds)).unwrap();
                let h = adaptive.hessian(sample, &[x], Some(&bounds)).unwrap();
                assert!(g.report().is_success(), "x={x}, {:?}", g.report());
                assert!(h.report().is_success(), "x={x}, {:?}", h.report());
                if method == AdaptiveDifferenceMethod::Ridders {
                    assert_eq!(g.report().evaluations(), 1 + 2 * g.report().iterations());
                    assert_eq!(h.report().evaluations(), 1 + 3 * h.report().iterations());
                }
                close(
                    g.values().unwrap()[0],
                    x.exp(),
                    if method == AdaptiveDifferenceMethod::Ridders {
                        1e-10
                    } else {
                        3e-8
                    },
                );
                close(
                    h.values().unwrap()[0],
                    x.exp(),
                    if method == AdaptiveDifferenceMethod::Ridders {
                        1e-8
                    } else {
                        3e-8
                    },
                );
            }
            for scheme in [DifferenceScheme::Central, DifferenceScheme::Forward] {
                let fixed = FiniteDifference::default().with_scheme(scheme);
                let g = fixed.gradient(sample, &[x], Some(&bounds), None).unwrap();
                let h = fixed.hessian(sample, &[x], Some(&bounds), None).unwrap();
                close(
                    g.values().unwrap()[0],
                    x.exp(),
                    if scheme == DifferenceScheme::Central {
                        1e-9
                    } else {
                        5e-8
                    },
                );
                close(
                    h.values().unwrap()[0],
                    x.exp(),
                    if scheme == DifferenceScheme::Central {
                        1e-7
                    } else {
                        2e-5
                    },
                );
            }
        }
    }
}
#[test]
fn bounded_jvp_and_mixed_hessian() {
    for sign in [-1.0, 1.0] {
        let bounds = bound(sign);
        let x = sign * 1e-12;
        for direction in [-1.0, 1.0] {
            let sample = |point: &[f64], y: &mut [f64]| {
                assert!(sign * point[0] >= 0.0);
                exp(point, y)
            };
            let result = AdaptiveDifference::default()
                .jvp(sample, &[x], &[direction], 1, Some(&bounds))
                .unwrap();
            assert!(result.report().is_success());
            close(result.values().unwrap()[0], direction * x.exp(), 1e-10);
            for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
                let result = FiniteDifference::default()
                    .with_scheme(scheme)
                    .jvp(sample, &[x], &[direction], 1, Some(&bounds), None)
                    .unwrap();
                close(result.values().unwrap()[0], direction * x.exp(), 5e-8);
            }
        }
        let bounds = Bounds::new(vec![bounds.as_slice()[0], (None, None)]).unwrap();
        let sample = |point: &[f64], y: &mut [f64]| {
            assert!(sign * point[0] >= 0.0);
            y[0] = (point[0] + point[1]).exp();
            Ok(())
        };
        for scheme in [DifferenceScheme::Central, DifferenceScheme::Forward] {
            // A larger explicit step keeps roundoff in the mixed partial below truncation.
            let result = FiniteDifference::default()
                .with_scheme(scheme)
                .with_step(DifferenceStep::absolute(0.001).unwrap())
                .hessian(sample, &[x, 0.25], Some(&bounds), None)
                .unwrap();
            close(result.values().unwrap()[1], (x + 0.25).exp(), 0.003);
        }
        let result = AdaptiveDifference::default()
            .hessian(sample, &[x, 0.25], Some(&bounds))
            .unwrap();
        assert!(result.report().is_success());
        close(result.values().unwrap()[1], (x + 0.25).exp(), 1e-8);
    }
}
#[test]
fn colored_mixed_geometries_fit_two_evaluations_per_level() {
    let schedule = diagonal();
    let sample = |x: &[f64], y: &mut [f64]| {
        y[0] = x[0].exp();
        y[1] = x[1].sin();
        Ok(())
    };
    for sign in [-1.0, 1.0] {
        let bounds = Bounds::new(vec![bound(sign).as_slice()[0], (None, None)]).unwrap();
        for d in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4] {
            let point = [sign * d, 1.0];
            let plan = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default()).unwrap();
            let result = plan.evaluate(sample, &point, Some(&bounds)).unwrap();
            assert!(result.report().is_success());
            close(result.values().unwrap()[0], point[0].exp(), 1e-10);
            close(result.values().unwrap()[1], 1.0_f64.cos(), 1e-10);
            for bounds in [None, Some(&bounds)] {
                let budgets = SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, Budget::try_from(9).unwrap());
                let plan = ColoredJacobianPlan::new(
                    &schedule,
                    AdaptiveDifference::default()
                        .with_max_levels(4)
                        .with_budgets(budgets),
                )
                .unwrap();
                let result = plan.evaluate(sample, &point, bounds).unwrap();
                assert_eq!(result.report().evaluations(), 9);
                assert_eq!(
                    result.report().termination(),
                    DifferentiationTermination::RefinementLimit
                );
                assert!(result.report().output_available());
            }
        }
    }
}
#[test]
fn fixed_stencils_keep_nominal_inward_spacing_and_order() {
    for sign in [-1.0, 1.0] {
        let bounds = bound(sign);
        let x = sign * 1e-12;
        let fixed = FiniteDifference::default().with_step(DifferenceStep::absolute(0.125).unwrap());
        let points = std::cell::RefCell::new(Vec::new());
        let sample = |point: &[f64], y: &mut [f64]| {
            points.borrow_mut().push(point[0]);
            exp(point, y)
        };
        let result = fixed.hessian(sample, &[x], Some(&bounds), None).unwrap();
        assert_eq!(result.report().evaluations(), 4);
        for (i, point) in points.borrow().iter().enumerate() {
            close(*point, x + sign * i as f64 * 0.125, 1e-15);
        }
    }
}
fn out_of_range(error: SciOxideError) {
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::OutOfRange,
                ..
            }
        ),
        "{error:?}"
    );
}
fn underflow(error: SciOxideError) {
    assert!(
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Underflow,
                ..
            }
        ),
        "{error:?}"
    );
}
#[test]
fn complex_step_rejects_subnormal_steps_before_callbacks() {
    let method = ComplexStep::default().with_step(1e-320);
    let scalar = |_: Complex64| -> Result<Complex64, SciOxideError> { unreachable!() };
    let vector =
        |_: &[Complex64], _: &mut [Complex64]| -> Result<(), SciOxideError> { unreachable!() };
    out_of_range(method.derivative(scalar, 1.0).unwrap_err());
    out_of_range(method.gradient(vector, &[1.0]).unwrap_err());
    out_of_range(method.jacobian(vector, &[1.0], 1).unwrap_err());
    out_of_range(method.jvp(vector, &[1.0], &[1.0], 1).unwrap_err());
    out_of_range(ComplexStepPlan::new(&[1.0], &[&[0.5, 1.0, 1.5]], method).unwrap_err());
    out_of_range(RelativeComplexStep::new(&[&[-0.5, 0.0, 0.5]], method).unwrap_err());
    let result = ComplexStep::default()
        .with_step(f64::MIN_POSITIVE)
        .derivative(|z: Complex64| Ok(z), 1.0)
        .unwrap();
    close(result.values().unwrap()[0], 1.0, 1e-15);
}
#[test]
fn subnormal_responses_fail_without_publishing_and_zero_is_valid() {
    let method = ComplexStep::default().with_step(1e-300);
    underflow(
        method
            .derivative(|z: Complex64| Ok(z * 1e-20), 1.0)
            .unwrap_err(),
    );
    underflow(complex_step_derivative(|z: Complex64| Ok(z * 1e-300), 1.0).unwrap_err());
    let result = method
        .derivative(|_: Complex64| Ok(Complex64::new(2.0, 0.0)), 1.0)
        .unwrap();
    close(result.values().unwrap()[0], 0.0, f64::EPSILON);
    let mut workspace = ComplexStepWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    let mut output = [99.0];
    underflow(
        method
            .gradient_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    y[0] = x[0] * 1e-20;
                    Ok(())
                },
                &[1.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err(),
    );
    assert_eq!(output, [99.0]);
}
#[test]
fn termination_display_uses_variant_names() {
    use DifferentiationTermination::*;
    for (status, name) in [
        (Completed, "Completed"),
        (Converged, "Converged"),
        (RoundoffLimit, "RoundoffLimit"),
        (RefinementLimit, "RefinementLimit"),
        (StepSizeLimit, "StepSizeLimit"),
        (EvaluationLimit, "EvaluationLimit"),
        (IterationLimit, "IterationLimit"),
        (Cancelled, "Cancelled"),
        (DeadlineReached, "DeadlineReached"),
    ] {
        assert_eq!(status.to_string(), name);
    }
}

#[test]
fn workspace_bytes_include_capacity_retained_after_smaller_requests() {
    assert_eq!(
        AdaptiveDifferenceWorkspace::<f64>::default()
            .retained_bytes()
            .unwrap(),
        0
    );
    assert_eq!(
        ColoredJacobianWorkspace::<f64>::default()
            .retained_bytes()
            .unwrap(),
        0
    );
    assert_eq!(
        ComplexStepWorkspace::<f64>::default()
            .retained_bytes()
            .unwrap(),
        0
    );
    assert_eq!(
        PhysicalDifferenceWorkspace::<f64>::default()
            .retained_bytes()
            .unwrap(),
        0
    );
    assert_eq!(
        RelativeDifferenceWorkspace::<f64>::default()
            .retained_bytes()
            .unwrap(),
        0
    );
    let method = AdaptiveDifference::default();
    let mut work = AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::hessian(3),
    ))
    .unwrap();
    let retained = work.retained_bytes().unwrap();
    let small = AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    assert!(retained > small.bytes());
    work.try_reserve(&small).unwrap();
    assert_eq!(work.retained_bytes().unwrap(), retained);
}

#[test]
fn f32_step_range_and_boundary_accuracy() {
    let method = ComplexStep::default().with_step(f32::from_bits(1));
    out_of_range(
        method
            .derivative(|z: scioxide_core::Complex32| Ok(z), 1.0)
            .unwrap_err(),
    );
    underflow(
        ComplexStep::default()
            .with_step(f32::MIN_POSITIVE)
            .derivative(|z: scioxide_core::Complex32| Ok(z * 0.5), 1.0)
            .unwrap_err(),
    );
    let result = ComplexStep::default()
        .with_step(f32::MIN_POSITIVE)
        .derivative(|z: scioxide_core::Complex32| Ok(z), 1.0)
        .unwrap();
    assert!((result.values().unwrap()[0] - 1.0).abs() < f32::EPSILON);
    for sign in [-1.0_f32, 1.0] {
        let bounds = Bounds::new(vec![if sign > 0.0 {
            (Some(0.0), None)
        } else {
            (None, Some(0.0))
        }])
        .unwrap();
        let x = sign * 1e-8;
        let sample = |x: &[f32], y: &mut [f32]| {
            assert!(sign * x[0] >= 0.0);
            y[0] = x[0].exp();
            Ok(())
        };
        let result = FiniteDifference::default()
            .gradient(sample, &[x], Some(&bounds), None)
            .unwrap();
        assert!((result.values().unwrap()[0] - x.exp()).abs() < 8.0 * f32::EPSILON.powf(2.0 / 3.0));
        let result = AdaptiveDifference::default()
            .hessian(sample, &[x], Some(&bounds))
            .unwrap();
        // In f32 the rounding floor may prevent confirmation even when the
        // best value is accurate; the authoritative status must expose it.
        assert!(matches!(
            result.report().termination(),
            DifferentiationTermination::Converged | DifferentiationTermination::RoundoffLimit
        ));
        assert!(result.errors().unwrap()[0] >= (result.values().unwrap()[0] - x.exp()).abs());
        assert!((result.values().unwrap()[0] - x.exp()).abs() < 8.0 * f32::EPSILON.sqrt());
    }
}

#[test]
fn narrow_interval_uses_distinct_symmetric_nodes_when_one_sided_cannot_fit() {
    let h = f64::EPSILON;
    let bounds = Bounds::new(vec![(Some(1.0), Some(1.0 + 2.0 * h))]).unwrap();
    let x = 1.0 + h;
    let result = FiniteDifference::default()
        .gradient(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0];
                Ok(())
            },
            &[x],
            Some(&bounds),
            None,
        )
        .unwrap();
    close(result.values().unwrap()[0], 1.0, 1e-15);
    let result = FiniteDifference::default()
        .jvp(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0];
                Ok(())
            },
            &[x],
            &[1.0],
            1,
            Some(&bounds),
            None,
        )
        .unwrap();
    close(result.values().unwrap()[0], 1.0, 1e-15);
}

#[test]
fn adaptive_exponential_converges_across_stencil_switch_grid() {
    let method = AdaptiveDifference::default();
    let schedule = diagonal();
    let colored = ColoredJacobianPlan::new(&schedule, method.clone()).unwrap();
    for sign in [-1.0, 1.0] {
        let bounds = bound(sign);
        let colored_bounds = Bounds::new(vec![bounds.as_slice()[0], (None, None)]).unwrap();
        for i in 1..=1000 {
            let x = sign * f64::from(i) / 1000.0;
            let sample = |point: &[f64], y: &mut [f64]| {
                assert!(sign * point[0] >= 0.0);
                exp(point, y)
            };
            let gradient = method.gradient(sample, &[x], Some(&bounds)).unwrap();
            let hessian = method.hessian(sample, &[x], Some(&bounds)).unwrap();
            let unbounded_g = method.gradient(exp, &[x], None).unwrap();
            let unbounded_h = method.hessian(exp, &[x], None).unwrap();
            for (bounded, unbounded, tolerance) in [
                (&gradient, &unbounded_g, 1e-10),
                (&hessian, &unbounded_h, 1e-8),
            ] {
                assert!(
                    bounded.report().is_success(),
                    "x={x}: {:?}",
                    bounded.report()
                );
                bounded.report().require_success().unwrap();
                close(bounded.values().unwrap()[0], x.exp(), tolerance);
                close(
                    bounded.values().unwrap()[0],
                    unbounded.values().unwrap()[0],
                    tolerance,
                );
            }
            let sample = |point: &[f64], y: &mut [f64]| {
                assert!(sign * point[0] >= 0.0);
                y[0] = point[0].exp();
                y[1] = point[1].sin();
                Ok(())
            };
            let result = colored
                .evaluate(sample, &[x, 1.0], Some(&colored_bounds))
                .unwrap();
            assert!(
                result.report().is_success(),
                "colored x={x}: {:?}",
                result.report()
            );
            close(result.values().unwrap()[0], x.exp(), 1e-10);
            close(result.values().unwrap()[1], 1.0_f64.cos(), 1e-10);
        }
    }
}

#[test]
fn restarted_table_retains_best_estimates_on_refinement_limit() {
    let bounds = bound(1.0);
    let mut previous_error = f64::MAX;
    for levels in 3..=16 {
        let result = AdaptiveDifference::default()
            .with_max_levels(levels)
            .gradient(exp, &[0.05], Some(&bounds))
            .unwrap();
        assert!(result.report().output_available());
        let error = result.errors().unwrap()[0];
        assert!(
            error <= previous_error,
            "levels={levels}: {error} > {previous_error}"
        );
        close(result.values().unwrap()[0], 0.05_f64.exp(), error + 1e-14);
        previous_error = error;
    }
}

#[test]
fn mixed_and_directional_derivatives_cross_geometry_switches() {
    for method in [
        AdaptiveDifferenceMethod::Ridders,
        AdaptiveDifferenceMethod::Richardson,
    ] {
        let adaptive = AdaptiveDifference::default().with_method(method);
        for sign in [-1.0, 1.0] {
            let bounds = Bounds::new(vec![bound(sign).as_slice()[0], (None, None)]).unwrap();
            let point = [sign * 0.05, 0.25];
            let sample = |x: &[f64], y: &mut [f64]| {
                assert!(sign * x[0] >= 0.0);
                y[0] = (x[0] + x[1]).exp();
                Ok(())
            };
            let hessian = adaptive.hessian(sample, &point, Some(&bounds)).unwrap();
            assert!(
                hessian.report().is_success(),
                "{method:?}: {:?}",
                hessian.report()
            );
            for &value in hessian.values().unwrap() {
                close(value, (point[0] + point[1]).exp(), 3e-8);
            }
            let jvp = adaptive
                .jvp(sample, &point, &[1.0, 0.0], 1, Some(&bounds))
                .unwrap();
            assert!(jvp.report().is_success(), "{method:?}: {:?}", jvp.report());
            close(jvp.values().unwrap()[0], (point[0] + point[1]).exp(), 3e-8);
        }
    }
}
