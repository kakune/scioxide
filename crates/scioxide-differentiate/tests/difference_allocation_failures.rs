//! Fallible preparation and allocation-free reuse for batched/colored differences.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{
    Bounds, ColoringKind, ColoringSchedule, Complex64, ExecutionContext, InvalidInputReason,
    ProblemToken, SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
    StructuralPattern, StructuralPatternView,
};
use scioxide_differentiate::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! { static REMAINING: Cell<usize> = const { Cell::new(usize::MAX) }; }
struct FaultAllocator;
fn admitted() -> bool {
    REMAINING.with(|remaining| match remaining.get() {
        usize::MAX => true,
        0 => false,
        count => {
            remaining.set(count - 1);
            true
        }
    })
}
// SAFETY: Successful requests delegate unchanged layouts/pointers to System;
// rejected requests return null as required by GlobalAlloc. Fault state is thread-local.
unsafe impl GlobalAlloc for FaultAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.alloc_zeroed(layout) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if admitted() {
            // SAFETY: delegates the unchanged request to `System`.
            unsafe { System.realloc(pointer, layout, size) }
        } else {
            std::ptr::null_mut()
        }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from `System` through this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: FaultAllocator = FaultAllocator;
fn fail_after<T>(count: usize, call: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REMAINING.with(|state| state.set(usize::MAX));
        }
    }
    REMAINING.with(|state| state.set(count));
    let _reset = Reset;
    call()
}

fn until_success<T>(mut prepare: impl FnMut() -> Result<T, SciOxideError>) -> T {
    for limit in 0..1024 {
        match fail_after(limit, &mut prepare) {
            Ok(result) => return result,
            Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::AllocationFailure => {}
            Err(error) => panic!("unexpected preparation error: {error}"),
        }
    }
    panic!("preparation did not succeed");
}

#[test]
fn owned_evaluation_allocation_failures_keep_the_invoking_operation() {
    let fixed = FiniteDifference::<f64>::default();
    let adaptive = AdaptiveDifference::<f64>::default();
    for call in 0..4 {
        let mut completed = false;
        for after in 0..128 {
            let result = fail_after(after, || {
                let scalar = |x: f64| Ok(x * x);
                let callback = |x: &[f64], y: &mut [f64]| {
                    y[0] = x[0] * x[0];
                    Ok(())
                };
                match call {
                    0 => fixed.derivative(scalar, 1.0),
                    1 => fixed.gradient(callback, &[1.0], None, None),
                    2 => adaptive.derivative(scalar, 1.0),
                    _ => adaptive.gradient(callback, &[1.0], None),
                }
            });
            match result {
                Ok(_) => {
                    assert!(after > 0);
                    completed = true;
                    break;
                }
                Err(error) => assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::AllocationFailure { .. }
                )),
            }
        }
        assert!(completed);
    }
}

#[test]
fn owned_adapters_and_coordinate_step_construction_reserve_fallibly() {
    let offsets = [-0.5_f64, -0.25, 0.0, 0.25, 0.5];
    let axes = [offsets.as_slice(); 2];
    let point = [0.0, 0.0];
    let direction = [1.0, 1.0];
    let scalar = |x: &[f64]| Ok::<_, SciOxideError>(x[0] * x[0] + x[1] * x[1]);
    let vector = |x: &[f64], y: &mut [f64]| {
        y[0] = x[0] * x[0] + x[1] * x[1];
        Ok(())
    };
    let square = |x: f64| Ok::<_, SciOxideError>(x * x);
    let _ = until_success(|| DifferenceStep::absolute_coordinates(&[0.125, 0.25]));
    let _ = until_success(|| DifferenceStep::relative_coordinates(&[0.125, 0.25]));
    let _ = until_success(|| derivative(square, 1.0));
    let _ = until_success(|| second_derivative(square, 1.0));
    let _ = until_success(|| gradient(scalar, &point));
    let _ = until_success(|| hessian(scalar, &point));
    let _ = until_success(|| jacobian(vector, &point, 1));
    let _ = until_success(|| jvp(vector, &point, &direction, 1));
    let fixed = FiniteDifference::<f64>::default();
    let _ = until_success(|| fixed.derivative(square, 1.0));
    let _ = until_success(|| fixed.second_derivative(square, 1.0));
    let _ = until_success(|| fixed.gradient(vector, &point, None, None));
    let _ = until_success(|| fixed.jacobian(vector, &point, 1, None, None));
    let _ = until_success(|| fixed.hessian(vector, &point, None, None));
    let _ = until_success(|| fixed.jvp(vector, &point, &direction, 1, None, None));
    let adaptive = AdaptiveDifference::<f64>::default()
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let _ = until_success(|| adaptive.derivative(square, 1.0));
    let _ = until_success(|| adaptive.second_derivative(square, 1.0));
    let _ = until_success(|| adaptive.gradient(vector, &point, None));
    let _ = until_success(|| adaptive.jacobian(vector, &point, 1, None));
    let _ = until_success(|| adaptive.hessian(vector, &point, None));
    let _ = until_success(|| adaptive.jvp(vector, &point, &direction, 1, None));
    let _ = until_success(|| physical_difference_derivative(square, 0.0, &offsets));
    let _ = until_success(|| {
        RelativeDifference::new(
            DerivativeOperation::Second,
            &[&offsets],
            1,
            NodeDifferenceOptions::default(),
        )
        .and_then(|plan| {
            plan.evaluate(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = square(x[0])?;
                    Ok(())
                },
                &[0.0],
                None,
            )
        })
    });
    let _ = until_success(|| {
        PhysicalDifferencePlan::new(
            DerivativeOperation::Hessian,
            &point,
            &axes,
            1,
            NodeDifferenceOptions::default(),
        )
        .and_then(|plan| plan.evaluate(vector, None))
    });
    let _ = until_success(|| {
        RelativeDifference::new(
            DerivativeOperation::Hessian,
            &axes,
            1,
            NodeDifferenceOptions::default(),
        )
        .and_then(|plan| plan.evaluate(vector, &point, None))
    });
    let complex = |x: &[Complex64], y: &mut [Complex64]| {
        y[0] = x[0] * x[0] + x[1] * x[1];
        Ok(())
    };
    let complex_square = |x: Complex64| Ok(x * x);
    let step = ComplexStep::default().with_step(1e-100);
    let _ = until_success(|| complex_step_derivative(complex_square, 1.0));
    let _ = until_success(|| step.derivative(complex_square, 1.0));
    let _ = until_success(|| step.gradient(complex, &point));
    let _ = until_success(|| step.jacobian(complex, &point, 1));
    let _ = until_success(|| step.jvp(complex, &point, &direction, 1));
    let _ = until_success(|| {
        RelativeComplexStep::new(&axes, step).and_then(|plan| plan.evaluate(complex, &point, None))
    });
    let _ = until_success(|| {
        ComplexStepPlan::new(&point, &axes, step).and_then(|plan| plan.evaluate(complex, None))
    });
    let pattern = StructuralPatternView::new(1, 2, &[0, 2], &[0, 1]).unwrap();
    let schedule = |p: StructuralPatternView<'_>| {
        ColoringSchedule::new(
            p.to_owned()?,
            ColoringKind::DerivativeColumns,
            crate::try_colors()?,
        )
    };
    let _ = until_success(|| {
        schedule(pattern).and_then(|schedule| {
            schedule.validate_pattern(pattern)?;
            ColoredJacobianPlan::new(&schedule, AdaptiveDifference::default())
                .and_then(|plan| plan.evaluate(vector, &point, None))
        })
    });
    let _ = until_success(|| {
        schedule(pattern).and_then(|schedule| sparse_jacobian(vector, &point, &schedule))
    });
}

fn try_colors() -> Result<Vec<usize>, SciOxideError> {
    // Inject a known schedule without an infallible allocation in the provider.
    let mut colors = Vec::new();
    colors.try_reserve_exact(2).map_err(|_| {
        SciOxideError::allocation_failure(
            scioxide_core::quantity!("test_coloring"),
            2 * core::mem::size_of::<usize>(),
        )
    })?;
    colors.extend_from_slice(&[0, 1]);
    Ok(colors)
}
#[test]
fn every_preparation_allocation_is_fallible_and_reuse_needs_none() {
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0],
    )
    .unwrap();
    let plan = until_success(|| {
        ColoredJacobianPlan::new(
            &schedule,
            AdaptiveDifference::<f64>::default()
                .with_execution(scioxide_core::ExecutionPolicy::Auto),
        )
    });
    let mut workspace = until_success(|| plan.workspace::<f64>(()));
    let dense =
        AdaptiveDifference::<f64>::default().with_execution(scioxide_core::ExecutionPolicy::Auto);
    let _dense_workspace = until_success(|| {
        AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(
            &dense,
            DerivativeShape::jacobian(2, 2),
        ))
    });
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(470),
        scioxide_core::algorithm_id!("test", "allocation", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut values = [0.0; 2];
    let mut errors = [0.0; 2];
    let result = fail_after(0, || {
        plan.evaluate_into(
            scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                y[0] = x[0] * x[0];
                y[1] = x[1] * x[1];
                Ok(())
            }),
            &[1.0, 2.0],
            None,
            &mut values,
            &mut errors,
            &mut workspace,
            &mut ctx,
        )
    })
    .unwrap();
    assert!(result.is_success());
    assert_eq!(values, [2.0, 4.0]);
}

#[test]
fn changing_geometry_preparation_is_fallible_and_every_rebinding_needs_no_allocation() {
    let offsets = [-0.5_f64, -0.25, 0.0, 0.25, 0.5];
    let axes: [&[f64]; 2] = [&offsets, &offsets];
    let physical = until_success(|| {
        PhysicalDifferencePlan::new(
            DerivativeOperation::Hessian,
            &[1.0, 2.0],
            &axes,
            1,
            NodeDifferenceOptions::default(),
        )
    });
    let _physical_work = until_success(|| physical.workspace::<Complex64>(()));
    let relative = until_success(|| {
        RelativeDifference::new(
            DerivativeOperation::Hessian,
            &axes,
            1,
            NodeDifferenceOptions::default(),
        )
    });
    let mut rw = until_success(|| relative.workspace::<Complex64>(()));
    let step = ComplexStep::default().with_step(1e-100);
    let complex = until_success(|| RelativeComplexStep::new(&axes, step));
    let mut cw = until_success(|| complex.workspace(()));
    let physical_complex = until_success(|| ComplexStepPlan::new(&[1.0, 2.0], &axes, step));
    let _physical_complex_work = until_success(|| physical_complex.workspace(()));
    let directional = step.with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let mut jw = until_success(|| {
        ComplexStepWorkspace::workspace(DerivativeRequest::new(
            &directional,
            DerivativeShape::jvp(2, 1),
        ))
    });
    let adaptive_jvp = AdaptiveDifference::<f64>::default()
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let _adaptive_jvp_work = until_success(|| {
        AdaptiveDifferenceWorkspace::<Complex64>::workspace(DerivativeRequest::new(
            &adaptive_jvp,
            DerivativeShape::jvp(2, 1),
        ))
    });
    let mut real_output = [99.0; 4];
    let mut complex_output = [Complex64::new(99.0, -99.0); 4];
    for ordinal in 0..64 {
        let point = [ordinal as f64 / 16.0, ordinal as f64 / 32.0];
        fail_after(0, || {
            assert!(
                relative
                    .evaluate_into(
                        |x: &[f64], y: &mut [_]| {
                            y[0] = Complex64::new(x[0] * x[1], -x[0] * x[1]);
                            Ok(())
                        },
                        &point,
                        None,
                        &mut complex_output,
                        &mut rw,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap()
                    .output_available()
            );
            assert!(
                complex
                    .evaluate_into(
                        |x: &[Complex64], y: &mut [Complex64]| {
                            y[0] = x[0] * x[1];
                            Ok(())
                        },
                        &point,
                        None,
                        &mut real_output,
                        &mut cw,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap()
                    .output_available()
            );
            let mut product = [99.0];
            assert!(
                directional
                    .jvp_into(
                        |x: &[Complex64], y: &mut [Complex64]| {
                            y[0] = x[0] * x[1];
                            Ok(())
                        },
                        &point,
                        &[1.0, -1.0],
                        &mut product,
                        &mut jw,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap()
                    .output_available()
            );
            assert!((product[0] - (point[1] - point[0])).abs() < 1e-12);
        });
    }
    let before = complex_output;
    let error = {
        relative.evaluate_into(
            |_: &[f64], _: &mut [_]| {
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("injected"),
                    InvalidInputReason::Inconsistent,
                    "callback failure",
                ))
            },
            &[1.0, 2.0],
            None,
            &mut complex_output,
            &mut rw,
            scioxide_core::ExecutionContext::sequential(),
        )
    }
    .unwrap_err();
    assert_eq!(complex_output, before);
    assert!(
        matches!(error.view(), scioxide_core::SciOxideErrorView::DifferentiationFailure { cause, .. } if cause.kind() == SciOxideErrorKind::InvalidInput)
    );
    fail_after(0, || {
        assert!(
            relative
                .evaluate_into(
                    |_: &[f64], _: &mut [_]| panic!("callback on collapsed nodes"),
                    &[1e100, 2.0],
                    None,
                    &mut complex_output,
                    &mut rw,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
    });
    assert_eq!(complex_output, before);
}

#[test]
fn invalid_owned_inputs_precede_allocation_and_callbacks() {
    let fixed = FiniteDifference::<f64>::default();
    let physical = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[1.0],
        &[&[0.0, 1.0, 2.0]],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let complex = ComplexStep::default().with_step(1e-100);
    let wide = Bounds::new(vec![(None, None); 2]).unwrap();
    let narrow = Bounds::new(vec![(Some(0.5), None)]).unwrap();
    let mut calls = 0;
    let mut callback = |_: &[f64], _: &mut [f64]| {
        calls += 1;
        Ok(())
    };
    for (point, bounds, base, expected) in [
        (&[f64::NAN][..], None, None, SciOxideErrorKind::NonFinite),
        (
            &[1.0][..],
            Some(&wide),
            None,
            SciOxideErrorKind::DimensionMismatch,
        ),
        (
            &[1.0][..],
            None,
            Some(&[f64::INFINITY][..]),
            SciOxideErrorKind::NonFinite,
        ),
        (
            &[1.0][..],
            None,
            Some(&[][..]),
            SciOxideErrorKind::DimensionMismatch,
        ),
    ] {
        let error =
            fail_after(0, || fixed.jacobian(&mut callback, point, 1, bounds, base)).unwrap_err();
        assert_eq!(error.kind(), expected);
    }
    for (direction, expected) in [
        (&[][..], SciOxideErrorKind::DimensionMismatch),
        (&[f64::NAN][..], SciOxideErrorKind::NonFinite),
    ] {
        let error = fail_after(0, || {
            fixed.jvp(&mut callback, &[1.0], direction, 1, None, None)
        })
        .unwrap_err();
        assert_eq!(error.kind(), expected);
    }
    let error = fail_after(0, || physical.evaluate(&mut callback, Some(&narrow))).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    let error = fail_after(0, || {
        complex.gradient(
            |_: &[Complex64], _: &mut [Complex64]| {
                calls += 1;
                Ok(())
            },
            &[f64::NAN],
        )
    })
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    assert_eq!(calls, 0);
}

#[test]
fn affected_owned_boundaries_reject_inputs_before_allocation() {
    use scioxide_core::{NonFiniteOrigin, SciOxideErrorView};
    let adaptive = AdaptiveDifference::<f64>::default()
        .with_step(DifferenceStep::absolute_coordinates(&[0.5, 0.5]).unwrap());
    let directional = AdaptiveDifference::<f64>::default();
    let schedule = ColoringSchedule::new(
        StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0],
    )
    .unwrap();
    let colored =
        ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default()).unwrap();
    let relative = RelativeDifference::new(
        DerivativeOperation::Gradient,
        &[&[-1.0, 0.0, 1.0][..]; 2],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let calls = Cell::new(0);
    let mut callback = |_: &[f64], _: &mut [f64]| {
        calls.set(calls.get() + 1);
        Ok(())
    };
    let analytic = |_: &[Complex64], _: &mut [Complex64]| {
        calls.set(calls.get() + 1);
        Ok(())
    };
    let assert_shape = |error: SciOxideError, wanted: usize, given: usize| {
        assert!(
            matches!(error.view(), SciOxideErrorView::DimensionMismatch {
             expected, actual,
        } if expected == [wanted] && actual == [given])
        );
    };
    let assert_nonfinite = |error: SciOxideError| {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    };
    for actual in [0, 1, 3] {
        let point = &[1.0, 2.0, 3.0][..actual];
        // The point fixes the dimension; per-coordinate steps and the
        // direction must match it.
        assert_shape(
            fail_after(0, || adaptive.gradient(&mut callback, point, None)).unwrap_err(),
            actual,
            2,
        );
        assert_shape(
            fail_after(0, || {
                directional.jvp(&mut callback, point, &[1.0, 1.0], 2, None)
            })
            .unwrap_err(),
            actual,
            2,
        );
        assert_shape(
            fail_after(0, || colored.evaluate(&mut callback, point, None)).unwrap_err(),
            2,
            actual,
        );
        assert_shape(
            fail_after(0, || relative.evaluate(&mut callback, point, None)).unwrap_err(),
            2,
            actual,
        );
        assert_shape(
            fail_after(0, || {
                directional.jvp(&mut callback, &[1.0, 2.0], &[1.0; 3][..actual], 2, None)
            })
            .unwrap_err(),
            2,
            actual,
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let point = [value, 2.0];
        assert_nonfinite(
            fail_after(0, || adaptive.gradient(&mut callback, &point, None)).unwrap_err(),
        );
        assert_nonfinite(
            fail_after(0, || {
                directional.jvp(&mut callback, &point, &[1.0, 1.0], 2, None)
            })
            .unwrap_err(),
        );
        assert_nonfinite(
            fail_after(0, || colored.evaluate(&mut callback, &point, None)).unwrap_err(),
        );
        assert_nonfinite(
            fail_after(0, || relative.evaluate(&mut callback, &point, None)).unwrap_err(),
        );
        assert_nonfinite(
            fail_after(0, || {
                directional.jvp(&mut callback, &[1.0, 2.0], &[value, 1.0], 2, None)
            })
            .unwrap_err(),
        );
        assert_nonfinite(
            fail_after(0, || {
                PhysicalDifferencePlan::new(
                    DerivativeOperation::First,
                    &[value],
                    &[&[0.0, 1.0, 2.0]],
                    1,
                    NodeDifferenceOptions::default(),
                )
            })
            .unwrap_err(),
        );
        let step = ComplexStep::default().with_step(value);
        assert_nonfinite(fail_after(0, || step.gradient(analytic, &[1.0])).unwrap_err());
        assert_nonfinite(fail_after(0, || step.jvp(analytic, &[1.0], &[1.0], 1)).unwrap_err());
    }
    assert_eq!(calls.get(), 0);
}
// Each trial begins with a nonempty, successfully reserved workspace. After an
// injected growth failure, inspection, rejection and retry use that exact object.

fn recovery_trials<W, R>(
    initial: impl Fn() -> W,
    required: &R,
    capped_previous: &R,
    reserve: impl Fn(&mut W, &R) -> Result<(), SciOxideError>,
    check: impl Fn(&W, &R) -> Result<scioxide_core::Capacity, SciOxideError>,
    evaluate: impl Fn(&mut W) -> Result<(), SciOxideError>,
) {
    use scioxide_core::{Capacity, SciOxideErrorView};
    for ordinal in 0..256 {
        let mut workspace = initial();
        assert_eq!(check(&workspace, required).unwrap(), Capacity::NeedsGrowth);
        let result = fail_after(ordinal, || reserve(&mut workspace, required));
        match result {
            Ok(()) => {
                assert!(ordinal > 1, "fixture must exercise successive allocations");
                assert_eq!(check(&workspace, required).unwrap(), Capacity::Sufficient);
                fail_after(0, || evaluate(&mut workspace)).unwrap();
                return;
            }
            Err(error) => assert!(matches!(
                error.view(),
                SciOxideErrorView::AllocationFailure { .. }
            )),
        }
        assert_eq!(
            fail_after(0, || check(&workspace, required)).unwrap(),
            Capacity::NeedsGrowth
        );
        let retained = fail_after(0, || check(&workspace, capped_previous));
        if ordinal == 0 {
            assert_eq!(retained.unwrap(), Capacity::Sufficient);
        } else {
            assert!(
                matches!(retained.unwrap_err().view(), SciOxideErrorView::ResourceLimit { required, limit, .. }
                if required > limit),
                "partially grown storage must remain charged"
            );
        }
        let error = fail_after(0, || evaluate(&mut workspace)).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DimensionMismatch { .. }
        ));
        reserve(&mut workspace, required).unwrap();
        assert_eq!(check(&workspace, required).unwrap(), Capacity::Sufficient);
        fail_after(0, || evaluate(&mut workspace)).unwrap();
    }
    panic!("reserved-workspace growth never succeeded");
}

fn verify_recovery_evaluation(
    result: Result<DifferentiationReport, SciOxideError>,
    calls: usize,
    output: &[f64],
    expected: &[f64],
    errors: Option<&[f64]>,
) -> Result<(), SciOxideError> {
    match result {
        Ok(report) => {
            assert!(report.output_available());
            assert!(calls > 0);
            for (&actual, &expected) in output.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 8.0 * f64::EPSILON.sqrt(),
                    "{actual} != {expected}"
                );
            }
            if let Some(errors) = errors {
                assert!(
                    errors
                        .iter()
                        .all(|error| error.is_finite() && *error >= 0.0)
                );
            }
            Ok(())
        }
        Err(error) => {
            assert_eq!(calls, 0);
            assert!(output.iter().all(|value| *value == 99.0));
            if let Some(errors) = errors {
                assert!(errors.iter().all(|value| *value == 88.0));
            }
            Err(error)
        }
    }
}

#[test]
fn reserved_fixed_and_jvp_workspaces_recover_at_each_growth_allocation() {
    let method = FiniteDifference::<f64>::default();
    let requirements = |method: &FiniteDifference<f64>, shape| {
        FiniteDifferenceWorkspace::workspace_requirements(DerivativeRequest::new(method, shape))
            .unwrap()
    };
    let small = DerivativeShape::gradient(1);
    let small_bytes = requirements(&method, small).bytes();
    let capped = requirements(&method.clone().with_max_workspace_bytes(small_bytes), small);
    let initial =
        || FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(&method, small)).unwrap();
    for shape in [DerivativeShape::jacobian(3, 1), DerivativeShape::hessian(3)] {
        let hessian = shape == DerivativeShape::hessian(3);
        let required = requirements(&method, shape);
        recovery_trials(
            initial,
            &required,
            &capped,
            FiniteDifferenceWorkspace::try_reserve,
            FiniteDifferenceWorkspace::check,
            |workspace| {
                let mut calls = 0;
                let mut output = [99.0; 9];
                let len = shape.output_len().unwrap();
                let function = |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| x * x).sum();
                    Ok(())
                };
                let result = if hessian {
                    method.hessian_into(
                        function,
                        &[1.0; 3],
                        None,
                        None,
                        &mut output[..len],
                        workspace,
                        ExecutionContext::sequential(),
                    )
                } else {
                    method.jacobian_into(
                        function,
                        &[1.0; 3],
                        1,
                        None,
                        None,
                        &mut output[..len],
                        workspace,
                        ExecutionContext::sequential(),
                    )
                };
                let expected = if hessian {
                    [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
                } else {
                    [2.0; 9]
                };
                verify_recovery_evaluation(result, calls, &output[..len], &expected[..len], None)
            },
        );
    }
    let required = requirements(&method, DerivativeShape::jvp(3, 2));
    recovery_trials(
        initial,
        &required,
        &capped,
        FiniteDifferenceWorkspace::try_reserve,
        FiniteDifferenceWorkspace::check,
        |workspace| {
            let mut calls = 0;
            let mut output = [99.0; 2];
            let result = method.jvp_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y.fill(x.iter().map(|x| x * x).sum());
                    Ok(())
                },
                &[1.0; 3],
                &[1.0; 3],
                None,
                None,
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &[6.0; 2], None)
        },
    );
}

#[test]
fn reserved_adaptive_and_jvp_workspaces_recover_at_each_growth_allocation() {
    let requirements = |method: &AdaptiveDifference<f64>, shape| {
        AdaptiveDifferenceWorkspace::<f64>::workspace_requirements(DerivativeRequest::new(
            method, shape,
        ))
        .unwrap()
    };
    let levels = AdaptiveDifference::<f64>::default().with_max_levels(3);
    let small = DerivativeShape::gradient(1);
    let bytes = requirements(&levels, small).bytes();
    let capped = requirements(&levels.clone().with_max_workspace_bytes(bytes), small);
    let initial = || {
        AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(&levels, small))
            .unwrap()
    };
    let method = AdaptiveDifference::<f64>::default();
    for shape in [DerivativeShape::jacobian(3, 1), DerivativeShape::hessian(3)] {
        let hessian = shape == DerivativeShape::hessian(3);
        let required = requirements(&method, shape);
        recovery_trials(
            initial,
            &required,
            &capped,
            AdaptiveDifferenceWorkspace::try_reserve,
            AdaptiveDifferenceWorkspace::check,
            |workspace| {
                let mut calls = 0;
                let mut output = [99.0; 9];
                let mut errors = [88.0; 9];
                let len = shape.output_len().unwrap();
                let function = |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| x * x).sum();
                    Ok(())
                };
                let result = if hessian {
                    method.hessian_into(
                        function,
                        &[1.0; 3],
                        None,
                        &mut output[..len],
                        &mut errors[..len],
                        workspace,
                        ExecutionContext::sequential(),
                    )
                } else {
                    method.jacobian_into(
                        function,
                        &[1.0; 3],
                        1,
                        None,
                        &mut output[..len],
                        &mut errors[..len],
                        workspace,
                        ExecutionContext::sequential(),
                    )
                };
                let expected = if hessian {
                    [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
                } else {
                    [2.0; 9]
                };
                verify_recovery_evaluation(
                    result,
                    calls,
                    &output[..len],
                    &expected[..len],
                    Some(&errors[..len]),
                )
            },
        );
    }
    let required = requirements(&method, DerivativeShape::jvp(3, 2));
    recovery_trials(
        initial,
        &required,
        &capped,
        AdaptiveDifferenceWorkspace::try_reserve,
        AdaptiveDifferenceWorkspace::check,
        |workspace| {
            let mut calls = 0;
            let mut output = [99.0; 2];
            let mut errors = [88.0; 2];
            let result = method.jvp_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y.fill(x.iter().map(|x| x * x).sum());
                    Ok(())
                },
                &[1.0; 3],
                &[1.0; 3],
                None,
                &mut output,
                &mut errors,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &[6.0; 2], Some(&errors))
        },
    );
}

#[test]
fn reserved_node_workspaces_recover_at_each_growth_allocation() {
    let nodes = [0.5_f64, 1.0, 1.5];
    let small = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[1.0],
        &[&nodes],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let bytes = small.workspace_requirements::<f64>(()).unwrap().bytes();
    let capped = PhysicalDifferencePlan::new(
        DerivativeOperation::First,
        &[1.0],
        &[&nodes],
        1,
        NodeDifferenceOptions::default().with_max_workspace_bytes(bytes),
    )
    .unwrap();
    let capped = capped.workspace_requirements::<f64>(()).unwrap();
    let large = PhysicalDifferencePlan::new(
        DerivativeOperation::Hessian,
        &[1.0; 3],
        &[nodes.as_slice(); 3],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let required = large.workspace_requirements::<f64>(()).unwrap();
    recovery_trials(
        || small.workspace::<f64>(()).unwrap(),
        &required,
        &capped,
        PhysicalDifferenceWorkspace::try_reserve,
        PhysicalDifferenceWorkspace::check,
        |workspace| {
            let mut output = [99.0; 9];
            let mut calls = 0;
            let result = large.evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| x * x).sum();
                    Ok(())
                },
                None,
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(
                result,
                calls,
                &output,
                &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
                None,
            )
        },
    );

    let offsets = [-0.5_f64, 0.0, 0.5];
    let wider = [-0.5_f64, -0.25, 0.0, 0.25, 0.5];
    let small = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let bytes = small.workspace_requirements::<f64>(()).unwrap().bytes();
    let capped = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        NodeDifferenceOptions::default().with_max_workspace_bytes(bytes),
    )
    .unwrap();
    let capped = capped.workspace_requirements::<f64>(()).unwrap();
    let large = RelativeDifference::new(
        DerivativeOperation::Hessian,
        &[wider.as_slice(); 3],
        1,
        NodeDifferenceOptions::default(),
    )
    .unwrap();
    let required = large.workspace_requirements::<f64>(()).unwrap();
    recovery_trials(
        || small.workspace::<f64>(()).unwrap(),
        &required,
        &capped,
        RelativeDifferenceWorkspace::try_reserve,
        RelativeDifferenceWorkspace::check,
        |workspace| {
            let mut output = [99.0; 9];
            let mut calls = 0;
            let result = large.evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| x * x).sum();
                    Ok(())
                },
                &[1.0; 3],
                None,
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(
                result,
                calls,
                &output,
                &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
                None,
            )
        },
    );
}

#[test]
fn reserved_complex_and_jvp_workspaces_recover_at_each_growth_allocation() {
    let method = ComplexStep::<f64>::default().with_step(1e-100);
    let small = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
    let bytes = ComplexStepWorkspace::workspace_requirements(small)
        .unwrap()
        .bytes();
    let capped = method.with_max_workspace_bytes(bytes);
    let capped = ComplexStepWorkspace::workspace_requirements(DerivativeRequest::new(
        &capped,
        DerivativeShape::gradient(1),
    ))
    .unwrap();
    let initial = || ComplexStepWorkspace::workspace(small).unwrap();
    let hessian = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
    let offsets = [-0.5, 0.0, 0.5];
    let physical = [0.5, 1.0, 1.5];
    let axes = [offsets.as_slice(); 3];
    let physical_axes = [physical.as_slice(); 3];

    let required = ComplexStepWorkspace::workspace_requirements(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(3),
    ))
    .unwrap();
    recovery_trials(
        initial,
        &required,
        &capped,
        ComplexStepWorkspace::try_reserve,
        ComplexStepWorkspace::check,
        |workspace| {
            let mut output = [99.0; 3];
            let mut calls = 0;
            let result = method.gradient_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| *x * *x).sum();
                    Ok(())
                },
                &[1.0; 3],
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &[2.0; 3], None)
        },
    );
    let relative = RelativeComplexStep::new(&axes, method).unwrap();
    let required = relative.workspace_requirements(()).unwrap();
    recovery_trials(
        initial,
        &required,
        &capped,
        ComplexStepWorkspace::try_reserve,
        ComplexStepWorkspace::check,
        |workspace| {
            let mut output = [99.0; 9];
            let mut calls = 0;
            let result = relative.evaluate_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| *x * *x).sum();
                    Ok(())
                },
                &[1.0; 3],
                None,
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &hessian, None)
        },
    );
    let plan = ComplexStepPlan::new(&[1.0; 3], &physical_axes, method).unwrap();
    let required = plan.workspace_requirements(()).unwrap();
    recovery_trials(
        initial,
        &required,
        &capped,
        ComplexStepWorkspace::try_reserve,
        ComplexStepWorkspace::check,
        |workspace| {
            let mut output = [99.0; 9];
            let mut calls = 0;
            let result = plan.evaluate_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    calls += 1;
                    y[0] = x.iter().map(|x| *x * *x).sum();
                    Ok(())
                },
                None,
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &hessian, None)
        },
    );
    let required = ComplexStepWorkspace::workspace_requirements(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(3, 2),
    ))
    .unwrap();
    recovery_trials(
        initial,
        &required,
        &capped,
        ComplexStepWorkspace::try_reserve,
        ComplexStepWorkspace::check,
        |workspace| {
            let mut output = [99.0; 2];
            let mut calls = 0;
            let result = method.jvp_into(
                |x: &[Complex64], y: &mut [Complex64]| {
                    calls += 1;
                    y.fill(x.iter().map(|x| *x * *x).sum());
                    Ok(())
                },
                &[1.0; 3],
                &[1.0; 3],
                &mut output,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &[6.0; 2], None)
        },
    );
}

#[test]
fn reserved_colored_workspace_recovers_at_each_child_and_shared_growth_allocation() {
    let small_schedule = ColoringSchedule::new(
        StructuralPattern::new(1, 1, vec![0, 1], vec![0]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0],
    )
    .unwrap();
    let large_schedule = ColoringSchedule::new(
        StructuralPattern::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2]).unwrap(),
        ColoringKind::DerivativeColumns,
        vec![0, 0, 0],
    )
    .unwrap();
    let small = ColoredJacobianPlan::new(
        &small_schedule,
        AdaptiveDifference::<f64>::default().with_max_levels(3),
    )
    .unwrap();
    let bytes = small.workspace_requirements::<f64>(()).unwrap().bytes();
    let capped = ColoredJacobianPlan::new(
        &small_schedule,
        AdaptiveDifference::<f64>::default()
            .with_max_levels(3)
            .with_max_workspace_bytes(bytes),
    )
    .unwrap();
    let capped = capped.workspace_requirements::<f64>(()).unwrap();
    let large =
        ColoredJacobianPlan::new(&large_schedule, AdaptiveDifference::<f64>::default()).unwrap();
    let required = large.workspace_requirements::<f64>(()).unwrap();
    recovery_trials(
        || small.workspace::<f64>(()).unwrap(),
        &required,
        &capped,
        ColoredJacobianWorkspace::try_reserve,
        ColoredJacobianWorkspace::check,
        |workspace| {
            let mut output = [99.0; 3];
            let mut errors = [88.0; 3];
            let mut calls = 0;
            let result = large.evaluate_into(
                |x: &[f64], y: &mut [f64]| {
                    calls += 1;
                    for (x, y) in x.iter().zip(y) {
                        *y = x * x;
                    }
                    Ok(())
                },
                &[1.0; 3],
                None,
                &mut output,
                &mut errors,
                workspace,
                ExecutionContext::sequential(),
            );
            verify_recovery_evaluation(result, calls, &output, &[2.0; 3], Some(&errors))
        },
    );
}

#[test]
fn scalar_fixed_differences_execute() {
    let bounds = scioxide_core::ExtendedInterval::new(Some(0.0), Some(1.0)).unwrap();
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        let method = FiniteDifference::default().with_scheme(scheme);
        let mut slope = f64::NAN;
        let report = fail_after(0, || {
            method.derivative_into(
                |x: f64| Ok(x * x),
                1.0,
                bounds,
                None,
                &mut slope,
                ExecutionContext::sequential(),
            )
        })
        .unwrap();
        assert!(report.output_available());
        assert!((slope - 2.0).abs() < 1e-6);
    }
}
