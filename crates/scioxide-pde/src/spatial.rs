//! Pure checked coordinate and coefficient helpers shared by spatial plans.

use scioxide_core::checked_difference;
pub(crate) use scioxide_core::checked_midpoint as midpoint;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// Positive `right - left`; `argument` names the caller's coordinate source.
#[inline]
pub(crate) fn positive_width<S: RealScalar>(
    right: S,
    left: S,
    argument: scioxide_core::Argument,
) -> Result<S, SciOxideError> {
    let width = checked_difference(right, left)?;
    if width <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "derived width must be positive",
        ));
    }
    Ok(width)
}

pub(crate) fn finite_callback<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Callback,
            "callback value is non-finite",
        ));
    }
    Ok(value)
}

pub(crate) fn nonnegative_coefficient<S: RealScalar + NumericScalar<Real = S>>(
    value: S,
    argument: scioxide_core::Argument,
) -> Result<S, SciOxideError> {
    let value = finite_callback(value)?;
    if value < S::ZERO {
        return Err(SciOxideError::domain_error(
            argument,
            scioxide_core::DomainReason::OutsideDomain,
            "coefficient must be nonnegative",
        ));
    }
    Ok(value)
}
