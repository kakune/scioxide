//! Complex and real values with one exponent shared by both components.

use crate::{AlgebraicScalar, NumericScalar, NumericalIssue, Scaled, ScaledValue, SciOxideError};

/// A value `s · 2ᵉ` whose components share one binary exponent `e`.
///
/// The largest nonzero component of `s` has magnitude in `[1, 2)`; signed zero
/// components are kept. Unlike [`ScaledValue`], this form cannot hold every pair
/// of finite components, because the smaller component may need more bits than
/// the significand has. Build it with [`ScaledValue::try_projective`], which
/// fails instead of dropping bits. No arithmetic is defined on this type.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex64, ScaledValue};
///
/// // 3 + 4i = (0.75 + 1i) · 2²
/// let value = ScaledValue::from_value(Complex64::new(3.0, 4.0))?;
/// let frame = value.try_projective()?;
/// assert_eq!(frame.exponent(), 2);
/// assert_eq!(frame.significand(), Complex64::new(0.75, 1.0));
/// assert_eq!(frame.into_scaled()?.try_value()?, Complex64::new(3.0, 4.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ProjectiveValue<V: NumericScalar> {
    significand: V,
    exponent: i64,
}

impl<V: NumericScalar> ProjectiveValue<V> {
    pub(crate) fn from_scaled(value: ScaledValue<V>) -> Result<Self, SciOxideError> {
        let exponent = value.scale_exponent();
        let project = |component: Scaled<V::Real>| -> Result<V::Real, SciOxideError> {
            let value = component.round_shifted(-i128::from(exponent));
            if !component.is_zero() && value.is_zero() {
                return Err(SciOxideError::numerical_instability(
                    NumericalIssue::Underflow,
                    "shared scaling would erase a nonzero component",
                ));
            }
            if Scaled::from_parts(value, exponent)? != component {
                return Err(SciOxideError::numerical_instability(
                    NumericalIssue::LossOfSignificance,
                    "shared scaling would discard component significand bits",
                ));
            }
            Ok(value)
        };
        Ok(ProjectiveValue {
            significand: V::from_components(project(value.real())?, project(value.imaginary())?),
            exponent,
        })
    }

    /// Returns the significand `s`; no nonzero component has been rounded to zero.
    pub const fn significand(self) -> V {
        self.significand
    }
    /// Returns the shared exponent `e`; zero has exponent zero.
    pub const fn exponent(self) -> i64 {
        self.exponent
    }
    /// Converts back to a [`ScaledValue`], exactly.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if a component exponent leaves
    ///   the `i64` range. This cannot happen for a value built by
    ///   [`ScaledValue::try_projective`].
    pub fn into_scaled(self) -> Result<ScaledValue<V>, SciOxideError> {
        ScaledValue::from_parts(self.significand, self.exponent)
    }
}
