//! Compensated sums with an extended exponent range.

use super::real::Wide;
use crate::{RealScalar, Scaled, SciOxideError};
use core::cmp::Ordering;
use core::ops::Neg;

/// A compensated sum of [`Scaled`](crate::Scaled) values that cannot overflow.
///
/// The sum is stored as a running sum and a Neumaier correction, both with an
/// `i64` exponent. The correction recovers low-order terms that a larger running
/// sum would absorb, so cancellation later in the sum can expose them. Products
/// are rounded before they are added, and the correction has the precision of
/// `S`, so the result is neither exactly rounded nor guaranteed to have the
/// exact sign. For an exactly rounded sum, use
/// [`RangeAccumulator`](crate::RangeAccumulator). A failed update leaves the sum
/// unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Scaled, ScaledAccumulator};
///
/// // f64::MAX + 1 − f64::MAX = 1, although f64::MAX + 1 rounds to f64::MAX.
/// let large = Scaled::from_value(f64::MAX)?;
/// let mut sum = ScaledAccumulator::from_scaled(large);
/// sum.add_value(1.0)?;
/// sum.sub_scaled(large)?;
/// assert_eq!(sum.round()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledAccumulator<S: RealScalar> {
    sum: Scaled<S>,
    correction: Scaled<S>,
}

impl<S: RealScalar> ScaledAccumulator<S> {
    fn checked_parts(sum: Wide<S>, correction: Wide<S>) -> Result<Self, SciOxideError> {
        if let (Ok(sum), Ok(correction)) = (sum.finish(), correction.finish()) {
            return Ok(Self { sum, correction });
        }
        // A supplied decomposition can have large cancelling parts. Recompose
        // them before rejecting storage exhaustion; retain the TwoSum residual.
        let total = sum.add(correction)?;
        let error = if sum.abs_cmp(correction) != Ordering::Less {
            sum.sub(total)?.add(correction)?
        } else {
            correction.sub(total)?.add(sum)?
        };
        Ok(Self {
            sum: total.finish()?,
            correction: error.finish()?,
        })
    }
    /// Positive zero.
    pub const ZERO: Self = Self {
        sum: Scaled::ZERO,
        correction: Scaled::ZERO,
    };

    /// Starts a sum from one value.
    pub fn from_scaled(value: Scaled<S>) -> Self {
        Self {
            sum: value,
            correction: Scaled::ZERO,
        }
    }

    /// Builds a sum from a running sum and a correction.
    ///
    /// The parts need not be ordered by magnitude.
    pub fn from_parts(sum: Scaled<S>, correction: Scaled<S>) -> Self {
        Self { sum, correction }
    }

    /// Returns the running sum and the correction; the value is their exact sum.
    pub fn parts(self) -> [Scaled<S>; 2] {
        [self.sum, self.correction]
    }

    /// Returns whether adding a value with exponent at most `exponent` changes nothing.
    ///
    /// This holds when both parts are nonzero and their exponents exceed `exponent`
    /// by more than `MANTISSA_DIGITS + 2`. A `false` result does not mean that the
    /// value would change the sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Scaled, ScaledAccumulator};
    ///
    /// let sum = ScaledAccumulator::from_parts(
    ///     Scaled::from_parts(0.75_f64, 1000)?,
    ///     Scaled::from_parts(-0.5, 900)?,
    /// );
    /// assert!(sum.absorbs(800));
    /// assert!(!sum.absorbs(900));
    /// let mut updated = sum;
    /// updated.add_scaled(Scaled::from_parts(0.5, 800)?)?;
    /// assert_eq!(updated.parts(), sum.parts());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn absorbs(self, exponent: i64) -> bool {
        let gap = i128::from(S::MANTISSA_DIGITS + 2);
        !self.sum.is_zero()
            && !self.correction.is_zero()
            && i128::from(self.sum.exponent()) - i128::from(exponent) > gap
            && i128::from(self.correction.exponent()) - i128::from(exponent) > gap
    }

    /// Returns the sum multiplied by `2^exponent`, exactly.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn shifted(self, exponent: i64) -> Result<Self, SciOxideError> {
        Self::checked_parts(
            Wide::from(self.sum).shifted(i128::from(exponent)),
            Wide::from(self.correction).shifted(i128::from(exponent)),
        )
    }

    /// Starts a sum from a finite value, keeping signed zeros and subnormals.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    pub fn from_value(value: S) -> Result<Self, SciOxideError> {
        Ok(Self::from_scaled(Scaled::from_value(value)?))
    }

    /// Adds `value` to the sum.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn add_scaled(&mut self, value: Scaled<S>) -> Result<(), SciOxideError> {
        self.add_wide(value.into())
    }

    fn add_wide(&mut self, value: Wide<S>) -> Result<(), SciOxideError> {
        // Neumaier's TwoSum: with |a| >= |b|, (a - fl(a + b)) + b recovers the
        // rounding error of the addition. It is accumulated separately so that
        // later cancellation of `sum` can expose low-order terms it hid.
        let sum = Wide::from(self.sum);
        let next = sum.add(value)?;
        let error = if sum.abs_cmp(value) != Ordering::Less {
            sum.sub(next)?.add(value)?
        } else {
            value.sub(next)?.add(sum)?
        };
        let candidate = Self::checked_parts(next, Wide::from(self.correction).add(error)?)?;
        *self = candidate;
        Ok(())
    }

    /// Adds another compensated sum to this one.
    ///
    /// The other running sum is added before its correction, so a low part exposed
    /// by cancellation in the other sum is not rounded away.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Scaled, ScaledAccumulator};
    ///
    /// let large = Scaled::from_value(f64::MAX)?;
    /// let mut left = ScaledAccumulator::from_scaled(large);
    /// let mut right = ScaledAccumulator::from_value(1.0)?;
    /// right.sub_scaled(large)?;
    /// // f64::MAX + (1 − f64::MAX) = 1
    /// left.merge(right)?;
    /// assert_eq!(left.round()?, 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn merge(&mut self, other: Self) -> Result<(), SciOxideError> {
        let mut merged = *self;
        merged.add_scaled(other.sum)?;
        merged.add_scaled(other.correction)?;
        *self = merged;
        Ok(())
    }

    /// Subtracts `value` from the sum.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn sub_scaled(&mut self, value: Scaled<S>) -> Result<(), SciOxideError> {
        self.add_scaled(value.neg())
    }

    /// Adds a finite value to the sum.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn add_value(&mut self, value: S) -> Result<(), SciOxideError> {
        self.add_scaled(Scaled::from_value(value)?)
    }

    /// Adds the product `left · right`, rounded once, to the sum.
    ///
    /// The exponent range is checked after cancellation with the sum.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn add_product(&mut self, left: Scaled<S>, right: Scaled<S>) -> Result<(), SciOxideError> {
        self.add_wide(Wide::from(left).mul(right.into())?)
    }

    /// Returns the sum with both parts multiplied by `factor`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn multiplied(self, factor: Scaled<S>) -> Result<Self, SciOxideError> {
        // Scale both parts separately so that the correction term survives the
        // multiplication (compensated Horner steps depend on this).
        let a = Wide::from(self.sum).mul(factor.into())?;
        let b = Wide::from(self.correction).mul(factor.into())?;
        let sum = a.add(b)?;
        let error = if a.abs_cmp(b) != Ordering::Less {
            a.sub(sum)?.add(b)?
        } else {
            b.sub(sum)?.add(a)?
        };
        Self::checked_parts(sum, error)
    }

    /// Returns the running sum plus the correction, rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn total(self) -> Result<Scaled<S>, SciOxideError> {
        self.sum.add(self.correction)
    }

    /// Converts the total to `S`, allowing gradual underflow to a signed zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the rounded value is infinite, or an exponent
    ///   leaves the `i64` range.
    pub fn round(self) -> Result<S, SciOxideError> {
        self.total()?.round()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn absorbed<S: RealScalar>() {
        let s = |value| S::try_from_f64_lossy(value).unwrap();
        let gap = i64::from(S::MANTISSA_DIGITS + 2);
        for exponent in [i64::MIN + gap + 1, -1000, 0, 1000, i64::MAX] {
            for (a, b) in [(1.5, 1.0), (-1.5, 1.0), (1.5, -1.0)] {
                let state = ScaledAccumulator::from_parts(
                    Scaled::from_parts(s(a), exponent).unwrap(),
                    Scaled::from_parts(s(b), exponent).unwrap(),
                );
                assert!(!state.absorbs(exponent - gap));
                assert!(state.absorbs(exponent - gap - 1));
                for c in [-1.999, -1.0, 1.0, 1.999] {
                    let mut updated = state;
                    updated
                        .add_scaled(Scaled::from_parts(s(c), exponent - gap - 1).unwrap())
                        .unwrap();
                    for (a, b) in updated.parts().into_iter().zip(state.parts()) {
                        assert_eq!(a.exponent(), b.exponent());
                        assert_eq!(a.significand().decompose(), b.significand().decompose());
                    }
                }
            }
        }
        assert!(!ScaledAccumulator::<S>::ZERO.absorbs(i64::MIN));
        assert!(!ScaledAccumulator::from_scaled(Scaled::<S>::ONE).absorbs(i64::MIN));
        assert!(
            ScaledAccumulator::from_parts(Scaled::<S>::ONE, Scaled::<S>::ONE).absorbs(i64::MIN)
        );
    }

    #[test]
    fn absorption_preserves_both_parts_bitwise() {
        absorbed::<f32>();
        absorbed::<f64>();
    }

    fn merged<S: RealScalar>() {
        let large = Scaled::from_parts(S::ONE, i64::MAX / 4).unwrap();
        let mut left = ScaledAccumulator::from_scaled(large);
        let mut right = ScaledAccumulator::from_scaled(Scaled::<S>::ONE);
        right.sub_scaled(large).unwrap();
        let mut sequential = left;
        for part in right.parts() {
            sequential.add_scaled(part).unwrap();
        }
        left.merge(right).unwrap();
        assert_eq!(left, sequential);
        assert_eq!(left.total().unwrap(), Scaled::ONE);
        // Exponent exhaustion in the second part rolls back the first.
        let top = Scaled::from_parts(S::ONE + S::HALF, i64::MAX).unwrap();
        let mut state = ScaledAccumulator::from_scaled(top);
        let before = state;
        let other = ScaledAccumulator::from_parts(Scaled::ONE, top);
        assert!(state.merge(other).is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn merges_add_both_parts_transactionally() {
        merged::<f32>();
        merged::<f64>();
    }
}
