//! Shared input, output and derivative-shape validation.

use crate::options::DifferenceShape;
use scioxide_core::backend::checked_mul;
use scioxide_core::{
    Bounds, DifferentiationLocation, Dimensions, InvalidInputReason, NonFiniteOrigin,
    NumericalIssue, RealScalar, SciOxideError,
};

/// Validate a public vector length without allocating diagnostic storage.
pub(super) fn length(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(actual),
        ))
    }
}
pub(super) fn differentiation_arithmetic_failure(
    location: DifferentiationLocation,
    evaluations: usize,
    error: SciOxideError,
) -> SciOxideError {
    SciOxideError::differentiation_failure(location, evaluations, error)
}

pub(super) fn validate_problem<S: RealScalar>(
    x: &[S],
    bounds: Option<&Bounds<S>>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if x.iter().any(|value| !value.is_finite()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "input coordinates must be finite",
        ));
    }
    if let Some(bounds) = bounds {
        if bounds.len() != x.len() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(x.len()),
                Dimensions::one(bounds.len()),
            ));
        }
        for (&value, &(lower, upper)) in x.iter().zip(bounds.as_slice()) {
            if lower.is_some_and(|bound| value < bound) || upper.is_some_and(|bound| value > bound)
            {
                return Err(SciOxideError::invalid_input(
                    argument,
                    InvalidInputReason::OutOfRange,
                    "input coordinates must lie within their bounds",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_vector_base<S: RealScalar>(
    base: &[S],
    output_dimension: usize,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if base.len() != output_dimension {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(output_dimension),
            Dimensions::one(base.len()),
        ));
    }
    if base.iter().any(|value| !value.is_finite()) {
        return Err(SciOxideError::non_finite(
            origin,
            "supplied base components must be finite",
        ));
    }
    Ok(())
}

pub(super) fn validate_generated_point<S: RealScalar>(
    point: &[S],
    location: DifferentiationLocation,
    evaluations: usize,
) -> Result<(), SciOxideError> {
    if let Some(variable) = point.iter().position(|value| !value.is_finite()) {
        return Err(SciOxideError::differentiation_failure(
            location,
            evaluations,
            SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                alloc::format!("generated point variable {variable} is outside the finite range"),
            ),
        ));
    }
    Ok(())
}

pub(super) fn validate_output<S: RealScalar>(
    values: &[S],
    detail: &'static str,
) -> Result<(), SciOxideError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(NonFiniteOrigin::Output, detail))
    }
}

// Resource stops are control flow, never fabricated callback errors.
// Private helpers report failures that
// prepared or evaluates the plan; they never mint their own identity.
pub(super) fn invalid(argument: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        InvalidInputReason::UnsupportedConfiguration,
        detail,
    )
}
pub(super) fn breakdown(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Breakdown, detail)
}
pub(super) fn shape(
    kind: DifferenceShape,
    inputs: usize,
    outputs: usize,
) -> Result<usize, SciOxideError> {
    use DifferenceShape::*;
    if matches!(kind, First | Second) && inputs != 1 {
        return Err(invalid(
            scioxide_core::argument!("operation"),
            "scalar derivatives require one coordinate",
        ));
    }
    if matches!(kind, First | Second | Gradient | Hessian) && outputs != 1 {
        return Err(invalid(
            scioxide_core::argument!("operation"),
            "scalar derivatives, gradients and Hessians require one output",
        ));
    }
    match kind {
        First | Second => Ok(1),
        Gradient => Ok(inputs),
        Jacobian => checked_mul(
            inputs,
            outputs,
            scioxide_core::quantity!("derivative_entries"),
        ),
        Hessian => checked_mul(
            inputs,
            inputs,
            scioxide_core::quantity!("derivative_entries"),
        ),
        Jvp(_) => Ok(outputs),
    }
}

pub(super) fn validate_direction<S: RealScalar>(
    x: &[S],
    direction: &[S],
) -> Result<(), SciOxideError> {
    length(x.len(), direction.len())?;
    if direction.iter().any(|v| !v.is_finite()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "direction components must be finite",
        ));
    }
    Ok(())
}
