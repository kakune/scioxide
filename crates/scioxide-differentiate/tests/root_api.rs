//! Public root-path coverage for direct finite-difference lifecycle APIs.

#![allow(clippy::suboptimal_flops, clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, ExtendedInterval, SciOxideError};
use scioxide_differentiate::{
    DerivativeRequest, DerivativeShape, DerivativeSource, DifferenceScheme, DifferentiationReport,
    FiniteDifference, FiniteDifferenceWorkspace, ZeroDirectionPolicy,
};

fn square(point: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
    output[0] = point[0] * point[0];
    Ok(())
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance);
}

fn assert_central(stats: DifferentiationReport) {
    assert_eq!(
        stats.source(),
        DerivativeSource::FiniteDifference(DifferenceScheme::Central)
    );
    assert!(stats.evaluations() > 0);
}

#[test]
fn every_direct_finite_difference_lifecycle_is_available_at_the_root() -> Result<(), SciOxideError>
{
    let method = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let point = [2.0_f64];
    let base = [4.0_f64];

    let derivative = method.derivative(|x: f64| Ok(x * x), point[0])?;
    close(derivative.values().unwrap()[0], 4.0, 1.0e-8);
    let second = method.second_derivative(|x: f64| Ok(x * x), point[0])?;
    close(second.values().unwrap()[0], 2.0, 1.0e-6);
    let gradient = method.gradient(square, &point, None, None)?;
    close(gradient.values().unwrap()[0], 4.0, 1.0e-8);
    let jacobian = method.jacobian(square, &point, 1, None, None)?;
    close(jacobian.values().unwrap()[0], 4.0, 1.0e-8);
    let hessian = method.hessian(square, &point, None, None)?;
    close(hessian.values().unwrap()[0], 2.0, 1.0e-6);
    let jvp = method.jvp(square, &point, &[3.0], 1, None, None)?;
    close(jvp.values().unwrap()[0], 12.0, 1.0e-8);

    let mut output = [0.0_f64];
    let stats = method.derivative_into(
        |x: f64| Ok(x * x),
        point[0],
        ExtendedInterval::entire(),
        Some(base[0]),
        &mut output[0],
        ExecutionContext::sequential(),
    )?;
    assert_central(stats);
    close(output[0], 4.0, 1.0e-8);

    let mut gradient_workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::gradient(1),
    ))?;
    let stats = method.gradient_into(
        square,
        &point,
        None,
        Some(base[0]),
        &mut output,
        &mut gradient_workspace,
        ExecutionContext::sequential(),
    )?;
    assert_central(stats);
    close(output[0], 4.0, 1.0e-8);

    let mut jacobian_workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jacobian(1, 1),
    ))?;
    let stats = method.jacobian_into(
        square,
        &point,
        1,
        None,
        Some(&base),
        &mut output,
        &mut jacobian_workspace,
        ExecutionContext::sequential(),
    )?;
    assert_central(stats);
    close(output[0], 4.0, 1.0e-8);

    let mut hessian_workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::hessian(1),
    ))?;
    let stats = method.hessian_into(
        square,
        &point,
        None,
        Some(base[0]),
        &mut output,
        &mut hessian_workspace,
        ExecutionContext::sequential(),
    )?;
    assert_central(stats);
    close(output[0], 2.0, 1.0e-6);

    let mut jvp_workspace = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::jvp(1, 1),
    ))?;
    let stats = method.jvp_into(
        square,
        &point,
        &[3.0],
        None,
        Some(&base),
        &mut output,
        &mut jvp_workspace,
        ExecutionContext::sequential(),
    )?;
    assert_central(stats);
    close(output[0], 12.0, 1.0e-8);
    Ok(())
}
