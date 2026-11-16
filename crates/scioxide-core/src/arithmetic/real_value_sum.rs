//! Exact running sums that switch from floating-point to integer storage when needed.

use super::{RealProductSum, checked_final_rounding, validate_finite};
use crate::{RealScalar, SciOxideError};

// The inline integer fallback is required by the allocation-free contract.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
#[repr(C)]
enum Storage<S: RealScalar> {
    Floating([S; 2]),
    Integer(RealProductSum<S>),
}

/// Exact running sum of finite real values.
///
/// While the exact sum fits in two floating-point numbers, updates use
/// error-free additions. When it does not, or an intermediate overflows, the sum
/// moves to exact integer storage without rounding. [`Self::finish`] returns the
/// current sum correctly rounded; an exact zero is `+0`. A sum that overflows
/// on `finish` can still be updated and cancel back into range.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::RealValueSum;
///
/// let mut sum = RealValueSum::<f64>::new();
/// for value in [1e16, 1.0, -1e16] {
///     sum.add_value(value)?;
/// }
/// assert_eq!(sum.finish()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RealValueSum<S: RealScalar> {
    storage: Storage<S>,
}

impl<S: RealScalar> Default for RealValueSum<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> RealValueSum<S> {
    /// Returns an empty sum, whose value is `+0`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            storage: Storage::Floating([S::ZERO; 2]),
        }
    }

    /// Resets the sum to zero.
    #[inline]
    pub fn clear(&mut self) {
        // Clearing floating storage writes its two parts in place rather than
        // staging a whole enum value.
        if let Storage::Floating(parts) = &mut self.storage {
            *parts = [S::ZERO; 2];
        } else {
            self.storage = Storage::Floating([S::ZERO; 2]);
        }
    }

    /// Adds a finite value exactly.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    ///   infinite; the sum is unchanged.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the integer storage capacity is exceeded.
    #[inline(always)]
    pub fn add_value(&mut self, value: S) -> Result<(), SciOxideError> {
        validate_finite(value, "sum addend must be finite")?;
        if value == S::ZERO {
            return Ok(());
        }
        let original = match &mut self.storage {
            Storage::Integer(sum) => return sum.add_value(value),
            Storage::Floating(parts) => {
                let [leading, trailing] = *parts;
                let next = leading + value;
                if next.is_finite() {
                    let residual = addition_residual(leading, value, next);
                    if residual == S::ZERO {
                        parts[0] = next;
                        return Ok(());
                    }
                    if trailing == S::ZERO {
                        *parts = [next, residual];
                        return Ok(());
                    }
                    let correction = trailing + residual;
                    if correction.is_finite()
                        && addition_residual(trailing, residual, correction) == S::ZERO
                    {
                        *parts = [next, correction];
                        return Ok(());
                    }
                }
                *parts
            }
        };
        self.promote(original, value)
    }

    // Keep integer construction and its large stack frame outside native
    // updates. Promotion uses original, unrounded operands exactly once.
    #[cold]
    #[inline(never)]
    fn promote(&mut self, original: [S; 2], value: S) -> Result<(), SciOxideError> {
        let mut exact = RealProductSum::new();
        exact.add_value(original[0])?;
        exact.add_value(original[1])?;
        exact.add_value(value)?;
        self.storage = Storage::Integer(exact);
        Ok(())
    }

    /// Returns the current sum, correctly rounded; the result may be subnormal.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the rounded sum is
    /// outside the finite range. The exact sum is kept and can be updated.
    #[inline]
    pub fn finish(&mut self) -> Result<S, SciOxideError> {
        match &mut self.storage {
            Storage::Integer(sum) => sum.finish(),
            Storage::Floating([leading, trailing]) => {
                let value = *leading + *trailing;
                checked_final_rounding(
                    if value == S::ZERO { S::ZERO } else { value },
                    "finite mathematical result is outside the scalar range",
                )
            }
        }
    }
}

// Neumaier's exact addition residual. The operands and rounded sum are finite;
// subtracting the dominant operand cannot overflow, including with subnormals.
#[inline]
fn addition_residual<S: RealScalar>(left: S, right: S, sum: S) -> S {
    if left.abs() >= right.abs() {
        (left - sum) + right
    } else {
        (right - sum) + left
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::RangeAccumulator;

    #[test]
    fn every_prefix_matches_full_range_rounding_across_promotion_and_reuse() {
        fn check<S: RealScalar>() {
            let large = S::try_from_f64_lossy(2.0_f64.powi(100)).unwrap();
            let small = S::try_from_f64_lossy(2.0_f64.powi(-100)).unwrap();
            let mut sum = RealValueSum::new();
            for values in [
                [
                    S::ONE,
                    S::EPSILON * S::HALF,
                    S::EPSILON * S::EPSILON,
                    -S::ONE,
                    -S::EPSILON * S::HALF,
                    -(S::EPSILON * S::EPSILON),
                ],
                [large, S::ONE, small, -large, -S::ONE, -small],
                [S::MAX, S::MAX, -S::MAX, -S::MAX, S::MIN_SUBNORMAL, -S::ZERO],
            ] {
                sum.clear();
                let mut reference = RangeAccumulator::new();
                for value in values {
                    sum.add_value(value).unwrap();
                    reference.add(value).unwrap();
                    match (sum.finish(), reference.finish()) {
                        (Ok(actual), Ok(expected)) => {
                            assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits())
                        }
                        (actual, expected) => assert_eq!(actual.is_err(), expected.is_err()),
                    }
                }
                let before = sum.finish().unwrap();
                for value in [S::NAN, S::INFINITY, -S::INFINITY] {
                    assert!(sum.add_value(value).is_err());
                    assert_eq!(
                        sum.finish().unwrap().into_f64().to_bits(),
                        before.into_f64().to_bits()
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
