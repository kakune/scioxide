//! Euclidean norms by scaled sums of squares.

use super::{checked_final_rounding, checked_float_product, validate_finite};
use crate::{InvalidInputReason, RealScalar, SciOxideError};

/// Running Euclidean norm `‖x‖₂` kept as a scaled sum of squares.
///
/// The state is a pair `(scale, ssq)` with `‖x‖₂² = scale² · ssq`, where `scale`
/// is the largest magnitude seen so far and `ssq ≥ 1` (or both are zero). Only
/// ratios `|xᵢ| / scale ≤ 1` are squared, so intermediate squares do not
/// overflow, and their underflow is covered by the error bound below. States can be
/// merged, and merging partial states in a fixed order gives the same result
/// for sequential and parallel runs. Subnormal inputs are accepted, and a zero
/// norm is `+0`. A failed update or merge leaves the state unchanged.
///
/// # Accuracy
///
/// The error depends on the order of updates and is `O(n ε)`, not a fixed
/// number of ulps. Let `m` be the number of nonzero updates and nontrivial
/// merges, `u = ε/2` the unit roundoff and `k = 8m + 8`. If `k u < 1`,
///
/// ```text
/// |computed − ‖x‖₂| ≤ γ_k ‖x‖₂ + η/2,   γ_k = k u / (1 − k u),
/// ```
///
/// where `η` is the smallest positive subnormal. The bound covers rounding in
/// ratios, squares, additions, the square root and the final product. For
/// `k u ≥ 1` there is no useful relative bound. A state built with
/// [`Self::from_parts`] has the bound `γ₂ ‖x‖₂ + η/2` for the norm it
/// represents; accuracy relative to an earlier vector depends on how the parts
/// were computed.
///
/// # Examples
///
/// ```
/// use scioxide_core::ScaledNorm;
///
/// let mut norm = ScaledNorm::<f64>::new();
/// norm.add(3.0)?;
/// norm.add(4.0)?;
/// assert_eq!(norm.finish()?, 5.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ScaledNorm<S: RealScalar> {
    scale: S,
    sum_squares: S,
}

impl<S: RealScalar> Default for ScaledNorm<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> ScaledNorm<S> {
    /// Returns the state of the zero norm.
    pub fn new() -> Self {
        Self {
            scale: S::ZERO,
            sum_squares: S::ZERO,
        }
    }

    /// Creates a state from `scale` and the normalized sum of squares `ssq`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either part is NaN
    ///   or infinite.
    /// - [`InvalidInput`](crate::SciOxideErrorKind::InvalidInput)
    ///   (`Inconsistent`) if a part is negative, `scale = 0` with `ssq ≠ 0`, or
    ///   `scale > 0` with `ssq < 1`.
    pub fn from_parts(scale: S, sum_squares: S) -> Result<Self, SciOxideError> {
        validate_finite(scale, "norm scale must be finite")?;
        validate_finite(sum_squares, "normalized sum of squares must be finite")?;
        let invalid_zero = scale == S::ZERO && sum_squares != S::ZERO;
        let invalid_nonzero = scale > S::ZERO && sum_squares < S::ONE;
        if scale < S::ZERO || sum_squares < S::ZERO || invalid_zero || invalid_nonzero {
            return Err(SciOxideError::invalid_input(
                if scale < S::ZERO {
                    crate::argument!("scale")
                } else {
                    crate::argument!("sum_squares")
                },
                InvalidInputReason::Inconsistent,
                "scale and normalized sum of squares violate the LASSQ invariant",
            ));
        }
        Ok(if scale == S::ZERO {
            Self::new()
        } else {
            Self { scale, sum_squares }
        })
    }

    /// Returns the scale, the largest magnitude added so far.
    pub const fn scale(self) -> S {
        self.scale
    }

    /// Returns the normalized sum of squares `ssq ≥ 1` (or zero).
    pub const fn sum_squares(self) -> S {
        self.sum_squares
    }

    /// Adds `magnitude²` to the sum of squares.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `magnitude` is NaN
    ///   or infinite.
    /// - [`InvalidInput`](crate::SciOxideErrorKind::InvalidInput) (`OutOfRange`)
    ///   if `magnitude` is negative.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the normalized
    ///   sum of squares overflows.
    pub fn add_magnitude(&mut self, magnitude: S) -> Result<(), SciOxideError> {
        validate_finite(magnitude, "norm magnitude must be finite")?;
        if magnitude < S::ZERO {
            return Err(SciOxideError::invalid_input(
                crate::argument!("magnitude"),
                InvalidInputReason::OutOfRange,
                "norm magnitude must be nonnegative",
            ));
        }
        if magnitude == S::ZERO {
            return Ok(());
        }
        // LASSQ invariant: norm^2 = scale^2 * sum_squares with scale = max |x_i| so
        // far and sum_squares >= 1. Rescaling to the new maximum keeps every
        // squared ratio <= 1, so no square can overflow, and a ratio small enough
        // to underflow is negligible next to sum_squares >= 1.
        let candidate = if self.scale < magnitude {
            let ratio = if self.scale == S::ZERO {
                S::ZERO
            } else {
                self.scale / magnitude
            };
            Self {
                scale: magnitude,
                sum_squares: S::ONE + self.sum_squares * ratio * ratio,
            }
        } else {
            let ratio = magnitude / self.scale;
            Self {
                scale: self.scale,
                sum_squares: self.sum_squares + ratio * ratio,
            }
        };
        checked_final_rounding(
            candidate.sum_squares,
            "normalized sum of squares is outside the finite scalar range",
        )?;
        *self = candidate;
        Ok(())
    }

    /// Adds `value²` to the sum of squares.
    ///
    /// # Errors
    ///
    /// The same as [`Self::add_magnitude`], except that negative values are
    /// accepted.
    pub fn add(&mut self, value: S) -> Result<(), SciOxideError> {
        self.add_magnitude(value.abs())
    }

    /// Adds the sum of squares of `other` to this state.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the merged
    /// normalized sum of squares overflows, which needs many repeated merges.
    pub fn merge(&mut self, other: Self) -> Result<(), SciOxideError> {
        if other.scale == S::ZERO {
            return Ok(());
        }
        if self.scale == S::ZERO {
            *self = other;
            return Ok(());
        }
        let candidate = if self.scale < other.scale {
            let ratio = self.scale / other.scale;
            Self {
                scale: other.scale,
                sum_squares: other.sum_squares + self.sum_squares * ratio * ratio,
            }
        } else {
            let ratio = other.scale / self.scale;
            Self {
                scale: self.scale,
                sum_squares: self.sum_squares + other.sum_squares * ratio * ratio,
            }
        };
        checked_final_rounding(
            candidate.sum_squares,
            "normalized sum of squares is outside the finite scalar range",
        )?;
        *self = candidate;
        Ok(())
    }

    /// Returns the Euclidean norm `scale · √ssq`.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the norm is
    /// outside the finite range of `S`.
    pub fn finish(self) -> Result<S, SciOxideError> {
        if self.scale == S::ZERO {
            return Ok(S::ZERO);
        }
        checked_float_product(self.scale, self.sum_squares.sqrt())
    }
}

/// Computes the Euclidean norm `‖x‖₂ = √(Σ xᵢ²)` without overflow in
/// intermediate squares.
///
/// Uses a [`ScaledNorm`] in `O(n)` time. The error bound is the one on
/// [`ScaledNorm`], with `m` the number of nonzero entries.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an entry is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the norm is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_scaled_norm;
///
/// assert_eq!(checked_scaled_norm(&[3.0_f64, 4.0])?, 5.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_scaled_norm<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    let mut state = ScaledNorm::new();
    values.iter().try_for_each(|&value| state.add(value))?;
    state.finish()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn scaled_norm_reuses_lassq_invariant() {
        let norm = checked_scaled_norm(&[f64::MAX / 2.0, f64::MAX / 2.0]).unwrap();
        assert!(norm.is_finite());
        assert!(norm > f64::MAX / 2.0);
        assert_eq!(
            checked_scaled_norm(&[f64::from_bits(1)]).unwrap(),
            f64::from_bits(1)
        );
        assert!(ScaledNorm::from_parts(0.0_f64, 1.0).is_err());
        assert!(ScaledNorm::from_parts(-1.0_f64, 1.0).is_err());
    }
}
