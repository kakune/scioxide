use super::*;
use crate::{DerivativeShape, DifferenceScheme, ZeroDirectionPolicy};
use alloc::vec::Vec;
use scioxide_core::{Bounds, ConvergenceReason, NonFiniteOrigin, SciOxideError};

use alloc::vec;

mod cached {
    use crate as scioxide_differentiate;
    include!("../../tests/support/fixed.rs");
}

fn quadratic(x: &[f64]) -> Result<f64, SciOxideError> {
    Ok(x[0] * x[0] + 3.0 * x[1] * x[1])
}

fn workspace_for<S: RealScalar>(shape: DerivativeShape) -> FiniteDifferenceWorkspace<S> {
    FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &FiniteDifference::default(),
        shape,
    ))
    .unwrap()
}

fn scalar<S: RealScalar>(
    mut callback: impl FnMut(&[S]) -> Result<S, SciOxideError>,
) -> impl FnMut(&[S], &mut [S]) -> Result<(), SciOxideError> {
    move |p: &[S], y: &mut [S]| {
        y[0] = callback(p)?;
        Ok(())
    }
}

fn injected_failure() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("point"),
        scioxide_core::DomainReason::OutsideDomain,
        "injected callback failure",
    )
}

#[test]
fn feasible_gradient_skips_fixed_axis_and_uses_narrow_box_stencil() {
    let bounds = Bounds::new(vec![(Some(2.0), Some(2.0)), (Some(1.0), Some(1.0 + 1e-8))]).unwrap();
    let mut workspace = workspace_for(DerivativeShape::gradient(2));
    let mut output = [f64::NAN; 2];
    let mut calls = 0;
    let feasible = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_fixed_coordinates(FixedCoordinatePolicy::FeasibleZero);
    let stats = feasible
        .gradient_into(
            scalar(|point: &[f64]| {
                assert_eq!(point[0], 2.0);
                assert!(point[1] >= 1.0 && point[1] <= 1.0 + 1e-8);
                calls += 1;
                Ok(point[1] * point[1])
            }),
            &[2.0, 1.0],
            Some(&bounds),
            Some(1.0),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output[0], 0.0);
    assert!((output[1] - 2.0).abs() < 1e-5);
    assert_eq!(calls, stats.evaluations());

    // The feasible and ordinary gradients share a kernel but keep distinct identities.
    let error = feasible
        .gradient_into(
            scalar(|_: &[f64]| Ok(0.0)),
            &[2.0, 1.0],
            None,
            None,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let scioxide_core::SciOxideErrorView::InvalidInput { argument, .. } = error.view() else {
        unreachable!("unexpected error kind")
    };
    assert_eq!(argument, scioxide_core::argument!("bounds"));
    let error = FiniteDifference::default()
        .gradient_into(
            scalar(|_: &[f64]| Ok(0.0)),
            &[2.0, f64::NAN],
            None,
            None,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn gradients_cover_zero_mixed_scales_f32_and_f64() {
    let report = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Forward)
        .gradient(
            scalar(|x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1])),
            &[0.0_f64, 1.0e6],
            None,
            None,
        )
        .unwrap();
    assert!(report.values().unwrap()[0].abs() < 1.0e-6);
    assert!((report.values().unwrap()[1] - 2.0e6).abs() < 1.0);
    assert_eq!(report.report().evaluations(), 3);

    let report = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .gradient(
            scalar(|x: &[f32]| Ok(x[0] * x[0] * x[0])),
            &[0.25_f32],
            None,
            None,
        )
        .unwrap();
    assert!((report.values().unwrap()[0] - 0.1875).abs() < 2.0e-4);
    assert_eq!(report.report().evaluations(), 3);
}

#[test]
fn bounded_central_gradient_never_leaves_domain() {
    let bounds = Bounds::new(vec![(Some(0.0_f64), Some(1.0))]).unwrap();
    let report = cached::gradient(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Central),
        Some(&bounds),
        &[0.0],
        Some(0.0),
        |x| {
            assert!((0.0..=1.0).contains(&x[0]));
            Ok(x[0] * x[0])
        },
    )
    .unwrap();
    assert!(f64::abs(report.values().unwrap()[0]) < 1.0e-8);
    assert_eq!(report.report().evaluations(), 2);
}

#[test]
fn jacobian_hessian_and_jvp_match_analytic_values() {
    let scheme = DifferenceScheme::Central;
    let x = [2.0_f64, 3.0];
    let base = [6.0, 7.0];
    let jacobian = cached::jacobian(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        &x,
        2,
        Some(&base),
        |point, output| {
            output[0] = point[0] * point[1];
            output[1] = point[0] * point[0] + point[1];
            Ok(())
        },
    )
    .unwrap();
    for (actual, expected) in jacobian.values().unwrap().iter().zip([3.0, 2.0, 4.0, 1.0]) {
        assert!((*actual - expected).abs() < 1.0e-8);
    }

    let hessian = FiniteDifference::default()
        .with_scheme(scheme)
        .hessian(scalar(quadratic), &[1.0, 2.0], None, None)
        .unwrap();
    for (actual, expected) in hessian.values().unwrap().iter().zip([2.0, 0.0, 0.0, 6.0]) {
        assert!((*actual - expected).abs() < 1.0e-5);
    }

    let jvp = cached::jvp(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        ZeroDirectionPolicy::MathematicalZero,
        &x,
        &[1.0, -1.0],
        2,
        Some(&base),
        |point, output| {
            output[0] = point[0] * point[1];
            output[1] = point[0] * point[0] + point[1];
            Ok(())
        },
    )
    .unwrap();
    assert!((jvp.values().unwrap()[0] - 1.0).abs() < 1.0e-8);
    assert!((jvp.values().unwrap()[1] - 3.0).abs() < 1.0e-8);
}

#[test]
fn zero_jvp_is_a_value_and_structural_mismatch_is_an_error() {
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Forward)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let mut calls = 0;
    let report = method
        .jvp(
            |_point: &[f64], output: &mut [f64]| {
                calls += 1;
                output[0] = 1.0;
                Ok(())
            },
            &[1.0_f64, 2.0],
            &[0.0, 0.0],
            1,
            None,
            None,
        )
        .unwrap();
    assert_eq!(report.values().unwrap(), &[0.0]);
    assert_eq!(report.report().evaluations(), 0);
    assert_eq!(calls, 0);

    let mismatch = method.jvp(
        |_point: &[f64], output: &mut [f64]| {
            output[0] = 1.0;
            Ok(())
        },
        &[1.0_f64, 2.0],
        &[1.0],
        1,
        None,
        None,
    );
    assert!(matches!(
        mismatch.as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
}

#[test]
fn strict_zero_direction_validates_the_base_callback() {
    let mut workspace = workspace_for(DerivativeShape::jvp(1, 1));
    let mut output = [7.0_f64];
    let mut calls = 0;
    let stats = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Forward)
        .with_zero_direction(ZeroDirectionPolicy::ValidateBase)
        .jvp_into(
            |point: &[f64], value: &mut [f64]| {
                calls += 1;
                value[0] = point[0] * point[0];
                Ok(())
            },
            &[1.0_f64],
            &[0.0],
            None,
            None,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [0.0]);
    assert_eq!(calls, 1);
    assert_eq!(stats.evaluations(), 1);
}

#[test]
fn tiny_nonzero_directions_never_take_the_zero_fast_path() {
    let minimum64 = f64::from_bits(1);
    let report64 = cached::jvp(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Forward),
        None,
        ZeroDirectionPolicy::MathematicalZero,
        &[0.0_f64],
        &[minimum64],
        1,
        Some(&[0.0]),
        |point, output| {
            output[0] = 2.0 * point[0];
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report64.values().unwrap(), &[2.0 * minimum64]);
    assert!(report64.report().evaluations() > 0);

    let minimum32 = f32::from_bits(1);
    let report32 = cached::jvp(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Forward),
        None,
        ZeroDirectionPolicy::MathematicalZero,
        &[0.0_f32],
        &[minimum32],
        1,
        Some(&[0.0]),
        |point, output| {
            output[0] = 2.0 * point[0];
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report32.values().unwrap(), &[2.0 * minimum32]);
    assert!(report32.report().evaluations() > 0);
}

#[test]
fn jvp_scales_with_power_of_two_direction_changes() {
    let evaluate = |direction| {
        cached::jvp(
            &FiniteDifference::default().with_scheme(DifferenceScheme::Central),
            None,
            ZeroDirectionPolicy::MathematicalZero,
            &[0.5_f64],
            &[direction],
            1,
            Some(&[0.25]),
            |point, output| {
                output[0] = point[0] * point[0];
                Ok(())
            },
        )
        .unwrap()
    };
    let base = evaluate(1.0);
    let scaled = evaluate(1.0_f64.scalbn(400));
    assert!((base.values().unwrap()[0] - 1.0).abs() < 1.0e-8);
    assert!((scaled.values().unwrap()[0].scalbn(-400) - base.values().unwrap()[0]).abs() < 1.0e-8);
}

#[test]
fn directional_bounds_choose_inward_stencils_and_detect_fixed_points() {
    let interval = Bounds::new(vec![(Some(0.0_f64), Some(1.0))]).unwrap();
    let report = cached::jvp(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Forward),
        Some(&interval),
        ZeroDirectionPolicy::MathematicalZero,
        &[1.0_f64],
        &[1.0_f64],
        1,
        Some(&[1.0_f64]),
        |point, output| {
            assert!((0.0..=1.0).contains(&point[0]));
            output[0] = point[0];
            Ok(())
        },
    )
    .unwrap();
    assert!((report.values().unwrap()[0] - 1.0).abs() < 1.0e-8);

    let fixed = Bounds::new(vec![(Some(1.0_f64), Some(1.0))]).unwrap();
    let error = cached::jvp(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Forward),
        Some(&fixed),
        ZeroDirectionPolicy::MathematicalZero,
        &[1.0],
        &[1.0],
        1,
        Some(&[1.0]),
        |_, output| {
            output[0] = 1.0;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(
        matches!((error).view(), scioxide_core::SciOxideErrorView :: DifferentiationFailure { cause: source, .. } if matches!(source.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { reason: ConvergenceReason::StepSizeLimit, .. }))
    );
}

#[test]
fn adjacent_boundary_uses_the_only_representable_inward_step() {
    let adjacent = scioxide_core::checked_next_after(1.0_f64, 0.0).unwrap();
    let bounds = Bounds::new(vec![(Some(adjacent), Some(1.0))]).unwrap();
    let mut observed = Vec::new();
    let report = cached::jvp(
        &FiniteDifference::default().with_scheme(DifferenceScheme::Forward),
        Some(&bounds),
        ZeroDirectionPolicy::MathematicalZero,
        &[1.0],
        &[1.0],
        1,
        Some(&[1.0]),
        |point, output| {
            observed.push(point[0]);
            output[0] = point[0];
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(observed, vec![adjacent]);
    assert_eq!(report.values().unwrap(), &[1.0]);
}

#[test]
fn jvp_workspace_is_reusable_after_each_callback_failure() {
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    for failure_call in 1..=3 {
        let mut workspace = workspace_for(DerivativeShape::jvp(1, 1));
        let mut output = [9.0_f64];
        let mut calls = 0;
        let error = method
            .jvp_into(
                |point: &[f64], value: &mut [f64]| {
                    calls += 1;
                    if calls == failure_call {
                        return Err(injected_failure());
                    }
                    value[0] = point[0];
                    Ok(())
                },
                &[1.0],
                &[1.0],
                None,
                None,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(
            matches!((error).view(), scioxide_core::SciOxideErrorView :: DifferentiationFailure { frames, .. } if frames[0].evaluations == failure_call)
        );
        assert_eq!(output, [9.0]);

        let stats = method
            .jvp_into(
                |point: &[f64], value: &mut [f64]| {
                    value[0] = point[0];
                    Ok(())
                },
                &[1.0],
                &[1.0],
                None,
                Some(&[1.0]),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(stats.evaluations(), 2);
        assert!((output[0] - 1.0).abs() < 1.0e-8);
    }
}

#[test]
fn every_operation_workspace_is_reusable_after_each_callback_failure() {
    let method = FiniteDifference::default().with_scheme(DifferenceScheme::Central);
    let failing = |failure_call: usize| {
        let mut calls = 0;
        move |point: &[f64], value: &mut [f64]| {
            calls += 1;
            if calls == failure_call {
                return Err(injected_failure());
            }
            value[0] = point[0] * point[0];
            Ok(())
        }
    };
    let square = |point: &[f64], value: &mut [f64]| {
        value[0] = point[0] * point[0];
        Ok(())
    };
    let failed = |error: SciOxideError, failure_call: usize| {
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DifferentiationFailure { frames, .. }
                if frames[0].evaluations == failure_call
        )
    };
    let context = ExecutionContext::sequential;
    for failure_call in 1..=3 {
        let mut gradient_workspace = workspace_for(DerivativeShape::gradient(1));
        let mut gradient = [9.0_f64];
        let error = method
            .gradient_into(
                failing(failure_call),
                &[1.0],
                None,
                None,
                &mut gradient,
                &mut gradient_workspace,
                context(),
            )
            .unwrap_err();
        assert!(failed(error, failure_call));
        assert_eq!(gradient, [9.0]);
        method
            .gradient_into(
                square,
                &[1.0],
                None,
                Some(1.0),
                &mut gradient,
                &mut gradient_workspace,
                context(),
            )
            .unwrap();
        assert!((gradient[0] - 2.0).abs() < 1.0e-8);

        let mut jacobian_workspace = workspace_for(DerivativeShape::jacobian(1, 1));
        let mut jacobian = [9.0_f64];
        let error = method
            .jacobian_into(
                failing(failure_call),
                &[1.0],
                1,
                None,
                None,
                &mut jacobian,
                &mut jacobian_workspace,
                context(),
            )
            .unwrap_err();
        assert!(failed(error, failure_call));
        assert_eq!(jacobian, [9.0]);
        method
            .jacobian_into(
                square,
                &[1.0],
                1,
                None,
                Some(&[1.0]),
                &mut jacobian,
                &mut jacobian_workspace,
                context(),
            )
            .unwrap();
        assert!((jacobian[0] - 2.0).abs() < 1.0e-8);

        let mut hessian_workspace = workspace_for(DerivativeShape::hessian(1));
        let mut hessian = [9.0_f64];
        let error = method
            .hessian_into(
                failing(failure_call),
                &[1.0],
                None,
                None,
                &mut hessian,
                &mut hessian_workspace,
                context(),
            )
            .unwrap_err();
        assert!(failed(error, failure_call));
        assert_eq!(hessian, [9.0]);
        method
            .hessian_into(
                square,
                &[1.0],
                None,
                Some(1.0),
                &mut hessian,
                &mut hessian_workspace,
                context(),
            )
            .unwrap();
        assert!((hessian[0] - 2.0).abs() < 1.0e-6);
    }
}

#[test]
fn supplied_bases_and_directions_keep_input_provenance() {
    let scheme = DifferenceScheme::Forward;
    let mut scalar_calls = 0;
    let scalar_error = cached::gradient(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        &[1.0_f64],
        Some(f64::NAN),
        |_| {
            scalar_calls += 1;
            Ok(1.0)
        },
    )
    .unwrap_err();
    assert!(matches!(
        (scalar_error).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    assert_eq!(scalar_calls, 0);

    let mut vector_calls = 0;
    let vector_error = cached::jacobian(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        &[1.0_f64],
        2,
        Some(&[1.0, f64::NAN]),
        |_, _| {
            vector_calls += 1;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        (vector_error).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    assert_eq!(vector_calls, 0);

    let mut jvp_calls = 0;
    let invalid_base = cached::jvp(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        ZeroDirectionPolicy::MathematicalZero,
        &[1.0_f64],
        &[0.0],
        1,
        Some(&[f64::NAN]),
        |_, _| {
            jvp_calls += 1;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        (invalid_base).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
    assert_eq!(jvp_calls, 0);

    let invalid_direction = FiniteDifference::default()
        .with_scheme(scheme)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
        .jvp(
            |_: &[f64], _: &mut [f64]| Ok(()),
            &[1.0_f64],
            &[f64::NAN],
            1,
            None,
            None,
        )
        .unwrap_err();
    assert!(matches!(
        (invalid_direction).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
#[expect(
    clippy::panic,
    reason = "test callback deliberately unwinds to verify containment and rollback"
)]
fn unavailable_extreme_perturbation_is_a_typed_step_limit() {
    let mut workspace = workspace_for(DerivativeShape::jvp(2, 1));
    let mut output = [9.0_f64];
    let mut calls = 0;

    let error = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Forward)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
        .jvp_into(
            |_: &[f64], value: &mut [f64]| {
                calls += 1;
                value[0] = 0.0;
                Ok(())
            },
            &[f64::MAX, -f64::MAX],
            &[f64::MAX, f64::MAX],
            None,
            Some(&[0.0]),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();

    let scioxide_core::SciOxideErrorView::DifferentiationFailure {
        frames,
        cause: source,
    } = (error).view()
    else {
        panic!("unexpected error variant");
    };

    assert_eq!(frames[0].evaluations, 0);

    assert!(matches!(
        (source).view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::StepSizeLimit,
            ..
        }
    ));
    assert_eq!(calls, 0);
    assert_eq!(output, [9.0]);
}

#[test]
#[expect(
    clippy::panic,
    reason = "test callback deliberately unwinds to verify containment and rollback"
)]
fn vector_callbacks_must_write_every_finite_lane() {
    let scheme = DifferenceScheme::Forward;
    let error = cached::jacobian(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        &[1.0_f64],
        2,
        Some(&[1.0, 2.0]),
        |point, output| {
            output[0] = point[0];
            Ok(())
        },
    )
    .unwrap_err();

    let scioxide_core::SciOxideErrorView::DifferentiationFailure {
        frames,
        cause: source,
    } = (error).view()
    else {
        panic!("unexpected error variant");
    };
    let location = frames[0].location;
    assert_eq!(location.component_index(), Some(1));
    assert!(matches!(
        (source).view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
}

#[test]
fn scaled_stencils_recover_finite_extreme_derivatives() {
    let config64 = DifferenceScheme::Central;
    let base64 = f64::MAX / 2.0;
    let slope64 = f64::MAX / 4.0;
    let gradient64 = cached::gradient(
        &FiniteDifference::default().with_scheme(config64),
        None,
        &[0.0],
        Some(base64),
        |point| Ok(base64 + slope64 * point[0]),
    )
    .unwrap();
    assert!((gradient64.values().unwrap()[0] - slope64).abs() / slope64 < 1.0e-8);

    let center64 = f64::MAX / 4.0;
    let hessian64 = cached::hessian(
        &FiniteDifference::default().with_scheme(config64),
        None,
        &[center64],
        Some(0.0),
        |point| {
            let relative = (point[0] - center64) / center64;
            Ok(relative * relative * center64)
        },
    )
    .unwrap();
    let expected64 = 2.0 / center64;
    assert!((hessian64.values().unwrap()[0] - expected64).abs() / expected64 < 1.0e-6);

    let config32 = DifferenceScheme::Central;
    let base32 = f32::MAX / 2.0;
    let slope32 = f32::MAX / 4.0;
    let gradient32 = cached::gradient(
        &FiniteDifference::default().with_scheme(config32),
        None,
        &[0.0],
        Some(base32),
        |point| Ok(base32 + slope32 * point[0]),
    )
    .unwrap();
    assert!((gradient32.values().unwrap()[0] - slope32).abs() / slope32 < 2.0e-4);

    let center32 = f32::MAX / 4.0;
    let hessian32 = cached::hessian(
        &FiniteDifference::default().with_scheme(config32),
        None,
        &[center32],
        Some(0.0),
        |point| {
            let relative = (point[0] - center32) / center32;
            Ok(relative * relative * center32)
        },
    )
    .unwrap();
    let expected32 = 2.0 / center32;
    assert!((hessian32.values().unwrap()[0] - expected32).abs() / expected32 < 2.0e-3);
}

#[test]
#[expect(
    clippy::panic,
    reason = "test callback deliberately unwinds to verify containment and rollback"
)]
fn callback_errors_retain_variable_component_and_count() {
    let scheme = DifferenceScheme::Forward;
    let error = cached::jacobian(
        &FiniteDifference::default().with_scheme(scheme),
        None,
        &[1.0_f64, 2.0],
        2,
        Some(&[3.0, 4.0]),
        |_point: &[f64], output: &mut [f64]| {
            output[0] = 1.0;
            output[1] = f64::NAN;
            Ok(())
        },
    )
    .unwrap_err();
    let scioxide_core::SciOxideErrorView::DifferentiationFailure {
        frames,
        cause: source,
    } = (error).view()
    else {
        panic!("unexpected error variant");
    };
    let location = frames[0].location;
    let evaluations = frames[0].evaluations;
    assert_eq!(
        location.site(),
        scioxide_core::DifferentiationSite::Variable(0)
    );
    assert_eq!(location.component_index(), Some(1));
    assert_eq!(evaluations, 1);
    assert!(matches!(
        (source).view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
}

#[test]
fn into_output_is_unchanged_on_failure() {
    let method = FiniteDifference::default().with_scheme(DifferenceScheme::Forward);
    let mut workspace = workspace_for(DerivativeShape::gradient(2));
    let mut output = [9.0_f64, 9.0];
    let result = method.gradient_into(
        scalar(|_point: &[f64]| {
            Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::OutsideDomain,
                "outside callback domain",
            ))
        }),
        &[1.0, 2.0],
        None,
        Some(5.0),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    );
    assert!(matches!(
        result.as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DifferentiationFailure { .. }
    ));
    assert_eq!(output, [9.0, 9.0]);

    method
        .gradient_into(
            scalar(|point: &[f64]| Ok(point[0] * point[0] + point[1] * point[1])),
            &[1.0, 2.0],
            None,
            Some(5.0),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
    assert_ne!(output, [9.0, 9.0]);
}

#[test]
fn coordinate_step_shape_errors_keep_their_public_precedence() {
    let method = FiniteDifference::default()
        .with_step(DifferenceStep::absolute_coordinates(&[0.1, 0.2]).unwrap());
    let mut workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    let mut output = [99.0; 2];
    let mut calls = 0;
    let mut function = |_: &[f64], _: &mut [f64]| {
        calls += 1;
        Ok(())
    };
    let owned = method
        .gradient(&mut function, &[f64::NAN], None, None)
        .unwrap_err();
    let borrowed = method
        .gradient_into(
            &mut function,
            &[f64::NAN],
            None,
            None,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    for error in [owned, borrowed] {
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::DimensionMismatch {
             expected, actual, ..
        } if expected == [1] && actual == [2])
        );
    }
    assert_eq!(calls, 0);
    assert_eq!(output, [99.0; 2]);
}

#[test]
fn every_fixed_buffer_is_required_independently_of_aggregate_spare_bytes() {
    let method = FiniteDifference::<f64>::default();
    for (shape, kind, outputs) in [
        (
            DerivativeShape::jacobian(2, 2),
            DifferenceShape::Jacobian,
            2,
        ),
        (DerivativeShape::hessian(2), DifferenceShape::Hessian, 1),
    ] {
        let request = DerivativeRequest::new(&method, shape);
        let plan = method.plan(kind, 2, outputs).unwrap();
        let required = FiniteDifferenceWorkspace::workspace_requirements(request).unwrap();
        for i in 0..8 {
            let n = if i < 5 {
                required.lengths[i]
            } else {
                required.stencils[i - 5]
            };
            if n == 0 {
                continue;
            }
            let mut work = FiniteDifferenceWorkspace::workspace(request).unwrap();
            work.data.point.reserve(1024);
            work.data.derivative.reserve(1024);
            if i < 5 {
                *work.data.values_mut()[i] = Vec::with_capacity(n - 1);
            } else {
                *work.data.geometries_mut()[i - 5] = Vec::with_capacity(n - 1);
            }
            assert!(work.data.bytes(None).unwrap() > required.bytes());
            assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
            assert_eq!(
                work.prepare(&plan).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
        }
        let mut work = FiniteDifferenceWorkspace::workspace(request).unwrap();
        work.data.point.reserve(1024);
        work.data.reduction = Vec::new();
        assert_eq!(work.check(&required).unwrap(), Capacity::NeedsGrowth);
    }
}

#[test]
fn scalar_evaluation_matches_the_workspace_form_for_both_schemes() {
    for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
        for (x, lower, upper) in [
            (0.5_f64, None, None),
            (1.0, Some(0.0), Some(1.0)),
            (-2.0, Some(-2.0), None),
        ] {
            let method = FiniteDifference::default().with_scheme(scheme);
            let bounds = scioxide_core::ExtendedInterval::new(lower, upper).unwrap();
            let function = |t: f64| {
                assert!(bounds.contains(t));
                Ok(t.sin() * t)
            };
            let mut scalar = f64::NAN;
            let report = method
                .derivative_into(
                    function,
                    x,
                    bounds,
                    Some(x.sin() * x),
                    &mut scalar,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            let mut vector = [f64::NAN];
            let reference = method
                .gradient_into(
                    |p: &[f64], y: &mut [f64]| {
                        y[0] = function(p[0])?;
                        Ok(())
                    },
                    &[x],
                    Some(&Bounds::new(vec![(lower, upper)]).unwrap()),
                    Some(x.sin() * x),
                    &mut vector,
                    &mut workspace_for(DerivativeShape::gradient(1)),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.output_available());
            assert_eq!(report.evaluations(), reference.evaluations());
            assert_eq!(scalar.to_bits(), vector[0].to_bits());
            assert!((scalar - (x.sin() + x * x.cos())).abs() < 1e-6);
        }
    }
}

#[test]
fn scalar_evaluation_rejects_steps_policies_and_inputs_before_callbacks() {
    let calls = core::cell::Cell::new(0);
    let function = |t: f64| {
        calls.set(calls.get() + 1);
        Ok(t)
    };
    let entire = scioxide_core::ExtendedInterval::entire();
    let run = |method: &FiniteDifference<f64>, x, bounds, base| {
        let mut out = 7.0;
        let error = method
            .derivative_into(
                function,
                x,
                bounds,
                base,
                &mut out,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(out, 7.0);
        error.kind()
    };
    let first = FiniteDifference::default();
    let wide = FiniteDifference::default()
        .with_step(DifferenceStep::absolute_coordinates(&[0.1, 0.2]).unwrap());
    let validated =
        FiniteDifference::default().with_zero_direction(ZeroDirectionPolicy::ValidateBase);
    use scioxide_core::SciOxideErrorKind::{DimensionMismatch, InvalidInput, NonFinite};
    assert_eq!(run(&wide, 0.0, entire, None), DimensionMismatch);
    assert_eq!(run(&validated, 0.0, entire, None), InvalidInput);
    assert_eq!(run(&first, f64::NAN, entire, None), NonFinite);
    assert_eq!(run(&first, 0.0, entire, Some(f64::INFINITY)), NonFinite);
    let outside = scioxide_core::ExtendedInterval::new(Some(1.0), Some(2.0)).unwrap();
    assert_eq!(run(&first, 0.0, outside, None), InvalidInput);
    assert_eq!(
        run(&first, 0.0, scioxide_core::ExtendedInterval::empty(), None),
        InvalidInput
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn scalar_evaluation_stops_on_its_options_budget() {
    let method = FiniteDifference::default().with_budgets(
        scioxide_core::SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Evaluation,
            scioxide_core::Budget::ONE,
        ),
    );
    let mut out = 7.0_f64;
    let report = method
        .derivative_into(
            |t: f64| Ok(t),
            1.0,
            scioxide_core::ExtendedInterval::entire(),
            Some(1.0),
            &mut out,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(!report.output_available());
    assert_eq!(out, 7.0);
}

fn mixed_two_point_jacobian<S: RealScalar>() {
    let tiny = S::MIN_SUBNORMAL;
    for sign in [S::ONE, -S::ONE] {
        let options = FiniteDifference::default()
            .with_scheme(DifferenceScheme::Forward)
            .with_step(crate::DifferenceStep::absolute(S::TWO).unwrap());
        let bounds = Bounds::new(vec![if sign > S::ZERO {
            (Some(S::ZERO), None)
        } else {
            (None, Some(S::ZERO))
        }])
        .unwrap();
        let base = [-sign * S::MAX, tiny, S::ZERO, S::ZERO];
        let result = cached::jacobian(
            &options,
            Some(&bounds),
            &[S::ZERO],
            4,
            Some(&base),
            |point, out| {
                assert_eq!(point[0], sign * S::TWO);
                out[0] = (point[0] - sign) * S::MAX;
                out[1] = tiny;
                out[2] = point[0];
                out[3] = tiny * point[0];
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result.report().evaluations(), 1);
        for (&actual, expected) in
            result
                .values()
                .unwrap()
                .iter()
                .zip([S::MAX, S::ZERO, S::ONE, tiny])
        {
            assert_eq!(actual, expected);
        }
    }
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Forward)
        .with_step(crate::DifferenceStep::absolute(S::HALF).unwrap());
    let mut workspace = workspace_for(DerivativeShape::jacobian(1, 4));
    let sentinel = S::TWO;
    let mut output = [sentinel; 4];
    let outcome = method.jacobian_into(
        |x: &[S], out: &mut [S]| {
            out[0] = ((S::TWO * S::TWO) * x[0] - S::ONE) * S::MAX;
            out[1] = tiny;
            out[2] = x[0];
            out[3] = tiny * x[0];
            Ok(())
        },
        &[S::ZERO],
        4,
        None,
        Some(&[-S::MAX, tiny, S::ZERO, S::ZERO]),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    );
    assert!(outcome.is_err());
    assert_eq!(output, [sentinel; 4]);
}
#[test]
fn two_point_jacobian_replays_individual_ranges_and_preserves_failure_f64() {
    mixed_two_point_jacobian::<f64>();
}
#[test]
fn two_point_jacobian_replays_individual_ranges_and_preserves_failure_f32() {
    mixed_two_point_jacobian::<f32>();
}

fn rounded_domain_callbacks<S: RealScalar>() {
    let tiny = S::EPSILON / S::try_from_usize_exact(16).unwrap();
    for sign in [S::ONE, -S::ONE] {
        let x = -sign;
        let (lower, upper) = if sign > S::ZERO {
            (-S::TWO, -tiny)
        } else {
            (tiny, S::TWO)
        };
        let bounds = Bounds::new(vec![(Some(lower), Some(upper))]).unwrap();
        let callback = |p: &[S], y: &mut [S]| {
            assert!(p[0] >= lower && p[0] <= upper);
            y[0] = p[0] * p[0];
            Ok(())
        };
        let step = crate::DifferenceStep::absolute(S::ONE).unwrap();
        let method = FiniteDifference::default()
            .with_scheme(DifferenceScheme::Central)
            .with_step(step.clone());
        let gradient = method
            .gradient(callback, &[x], Some(&bounds), None)
            .unwrap()
            .into_values()
            .unwrap();
        let hessian = method
            .hessian(callback, &[x], Some(&bounds), None)
            .unwrap()
            .into_values()
            .unwrap();
        let tolerance = S::try_from_usize_exact(64).unwrap() * S::EPSILON.sqrt();
        assert!((gradient[0] - S::TWO * x).abs() <= tolerance);
        assert!((hessian[0] - S::TWO).abs() <= tolerance);
        let adaptive = crate::AdaptiveDifference::default().with_step(step);
        let gradient = adaptive
            .gradient(callback, &[x], Some(&bounds))
            .unwrap()
            .into_values()
            .unwrap();
        let hessian = adaptive
            .hessian(callback, &[x], Some(&bounds))
            .unwrap()
            .into_values()
            .unwrap();
        assert!((gradient[0] - S::TWO * x).abs() <= tolerance);
        assert!((hessian[0] - S::TWO).abs() <= tolerance);
    }
}
#[test]
fn rounded_domain_callbacks_f32() {
    rounded_domain_callbacks::<f32>();
}
#[test]
fn rounded_domain_callbacks_f64() {
    rounded_domain_callbacks::<f64>();
}
