//! Validation representation and operations.
use super::*;

/// Monomial coefficients without exact trailing zeros; an empty vector
/// is attributed to `argument`, the caller parameter that supplied it.
pub(super) fn canonical_coefficients<C: NumericScalar>(
    mut coefficients: Vec<C>,
    argument: Argument,
) -> Result<Vec<C>, SciOxideError> {
    let len = crate::util::canonical_coefficients(&coefficients, argument)?.len();
    coefficients.truncate(len);
    // Canonical owned zero storage does not retain a negative zero.
    if coefficients.len() == 1 && coefficients[0].abs() == C::Real::ZERO {
        coefficients[0] = C::ZERO;
    }
    Ok(coefficients)
}

pub(super) fn validate_interpolation_data<'a, C: NumericScalar>(
    nodes: &'a [C],
    values: &[C],
) -> Result<ValidatedNodes<'a, C>, SciOxideError> {
    if nodes.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("nodes"),
            InvalidInputReason::Empty,
            "at least one sample is required",
        ));
    }
    if nodes.len() != values.len() {
        return Err(layout_error(nodes.len(), values.len()));
    }
    validate_slice(values)?;
    ValidatedNodes::new(nodes)
}

pub(super) fn validate_conversion_nodes<C: NumericScalar>(
    nodes: &[C],
    expected: usize,
) -> Result<ValidatedNodes<'_, C>, SciOxideError> {
    if nodes.len() != expected {
        return Err(layout_error(expected, nodes.len()));
    }
    ValidatedNodes::new(nodes)
}

/// Finite, strictly increasing `[lower, upper]`: nonfinite endpoints are
/// `NonFinite(Input)`, unordered or equal ones `OutOfRange` on `upper`.
pub(crate) fn interval<R: RealScalar>(lower: R, upper: R) -> Result<(), SciOxideError> {
    crate::util::finite_input(lower, "interval endpoints must be finite")?;
    crate::util::finite_input(upper, "interval endpoints must be finite")?;
    if lower >= upper {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("upper"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "the interval must be strictly increasing",
        ));
    }
    Ok(())
}

pub(super) fn layout_error(expected: usize, actual: usize) -> SciOxideError {
    SciOxideError::dimension_mismatch(
        scioxide_core::Dimensions::one(expected),
        scioxide_core::Dimensions::one(actual),
    )
}
