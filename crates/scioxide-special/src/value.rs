//! Special-function values with their error estimates.

use scioxide_core::{EstimateReliability, InvalidInputReason, RealScalar, SciOxideError};

/// Stores a value with an absolute and an optional relative error estimate.
///
/// [`reliability`](Self::reliability) says how the estimate was obtained:
/// `ModelBased` for a bound derived from the truncation and rounding of the
/// evaluation, `StatisticallyCalibrated` for an envelope in ulps measured against
/// high-precision reference values, and `Heuristic` otherwise. No estimate is a
/// guaranteed enclosure. See [error estimates](crate#error-estimates).
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ValueWithError<V, R: RealScalar> {
    value: V,
    absolute_error: R,
    relative_error: Option<R>,
    reliability: EstimateReliability,
}

impl<V, R: RealScalar> ValueWithError<V, R> {
    /// Creates a value with nonnegative finite error estimates.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` (`OutOfRange`) if either estimate is negative or
    /// not finite.
    #[inline]
    pub(crate) fn new(
        value: V,
        absolute_error: R,
        relative_error: Option<R>,
        reliability: EstimateReliability,
    ) -> Result<Self, SciOxideError> {
        validate_error(absolute_error, scioxide_core::argument!("absolute_error"))?;
        if let Some(relative) = relative_error {
            validate_error(relative, scioxide_core::argument!("relative_error"))?;
        }
        Ok(Self {
            value,
            absolute_error,
            relative_error,
            reliability,
        })
    }

    /// Maps the value and keeps the error estimates.
    #[inline]
    pub(crate) fn try_map<W>(
        self,
        map: impl FnOnce(V) -> Result<W, SciOxideError>,
    ) -> Result<ValueWithError<W, R>, SciOxideError> {
        Ok(ValueWithError {
            value: map(self.value)?,
            absolute_error: self.absolute_error,
            relative_error: self.relative_error,
            reliability: self.reliability,
        })
    }

    /// Returns the value.
    pub const fn value(&self) -> &V {
        &self.value
    }

    /// Returns the value, consuming `self`.
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns how the absolute and relative error estimates were obtained.
    pub const fn reliability(&self) -> EstimateReliability {
        self.reliability
    }

    /// Returns the absolute error estimate, in the units of the value.
    pub const fn absolute_error(&self) -> R {
        self.absolute_error
    }

    /// Returns the relative error estimate, or `None` when it is not meaningful,
    /// for example for a zero value.
    pub const fn relative_error(&self) -> Option<R> {
        self.relative_error
    }
}

impl<V: scioxide_core::NumericScalar<Real = R>, R: RealScalar> ValueWithError<V, R> {
    pub(crate) fn exact_zero() -> Self {
        Self {
            value: V::ZERO,
            absolute_error: R::ZERO,
            relative_error: None,
            reliability: EstimateReliability::ModelBased,
        }
    }
}

fn validate_error<R: RealScalar>(
    value: R,
    name: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !value.is_finite() || value < R::ZERO {
        return Err(SciOxideError::invalid_input(
            name,
            InvalidInputReason::OutOfRange,
            "an error estimate must be finite and nonnegative",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn value_errors_must_be_nonnegative_and_finite() {
        assert!(
            ValueWithError::new(1.0_f64, 0.25, Some(0.125), EstimateReliability::ModelBased)
                .is_ok()
        );
        assert!(ValueWithError::new(1.0_f64, -0.25, None, EstimateReliability::Heuristic).is_err());
    }
}
