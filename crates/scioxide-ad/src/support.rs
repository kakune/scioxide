//! Shared checks of borrowed storage.

use scioxide_core::SciOxideError;

/// Checks an exact length, reporting expected then actual dimensions.
#[inline]
pub(crate) fn check_len(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}

/// Rejects a coloring that does not allow direct recovery of derivatives.
pub(crate) fn check_coloring(
    coloring: &scioxide_core::ColoringSchedule,
) -> Result<(), scioxide_core::SciOxideError> {
    if coloring.kind() == scioxide_core::ColoringKind::DerivativeColumns {
        return Ok(());
    }
    Err(scioxide_core::SciOxideError::invalid_input(
        scioxide_core::argument!("coloring"),
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        "direct recovery needs a derivative-column coloring",
    ))
}

/// Returns the lanes whose value and every derivative component are zero.
/// Multiplication by a passive NaN is finite exactly in those lanes.
#[inline]
pub(crate) fn is_zero<S: scioxide_core::RealScalar, V: crate::Real<S>>(value: V) -> V::Mask {
    if V::LEVEL == 0 {
        value.ge(V::zero()) & value.le(V::zero())
    } else {
        value.azmul(V::constant(S::NAN)).is_finite()
    }
}
