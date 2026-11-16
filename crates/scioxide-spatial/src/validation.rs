//! Shared input classifications and checked slice access.
use scioxide_core::{
    Argument, InvalidInputReason, NonFiniteOrigin, Quantity, RealScalar, SciOxideError,
};

pub(crate) fn out_of_range(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::OutOfRange, detail)
}
pub(crate) fn inconsistent(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}
pub(crate) fn get<T>(values: &[T], index: usize, name: Quantity) -> Result<&T, SciOxideError> {
    values
        .get(index)
        .ok_or_else(|| SciOxideError::out_of_bounds(name, index, values.len()))
}

// Reject a non-finite intermediate or result.
pub(crate) fn finite<S: RealScalar, const N: usize>(v: [S; N]) -> Result<[S; N], SciOxideError> {
    let Some(value) = v.iter().find(|x| !x.is_finite()) else {
        return Ok(v);
    };
    if value.is_infinite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "finite arithmetic produced an unrepresentable result",
        ));
    }
    Err(SciOxideError::non_finite(
        NonFiniteOrigin::Intermediate,
        "non-finite coordinate or result",
    ))
}
// Reject non-finite input.
pub(crate) fn input<S: RealScalar, const N: usize>(v: [S; N]) -> Result<[S; N], SciOxideError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(v)
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "coordinates and angles must be finite",
        ))
    }
}
