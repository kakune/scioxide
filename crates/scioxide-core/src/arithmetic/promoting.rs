//! Compensated sums and products that switch to an extended exponent range
//! only when needed.
//!
//! They work in the precision of the scalar type and are not exact. A
//! [`CompensatedSum`] or plain scalar switches once to [`ScaledAccumulator`] or
//! [`Scaled`] when an operand, product or partial sum leaves the normal range.
//! The switch keeps both parts exactly, so no term is lost or added twice.
use super::{CompensatedSum, compensated_reduce, four_lane_sums, non_finite_input, two_sum};
use crate::{RealScalar, Scaled, ScaledAccumulator, SciOxideError};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Product<S: RealScalar> {
    Native(S),
    Extended(Scaled<S>),
}

/// Running product of finite reals that switches to [`Scaled`] when it leaves
/// the normal range.
///
/// Factors are multiplied as plain scalars while every partial product is
/// normal or zero. The first subnormal or overflowing partial product switches
/// the product to [`Scaled`], where it stays. Each multiplication rounds once in
/// either form, so the product is not exact.
///
/// # Examples
///
/// ```
/// use scioxide_core::PromotingProduct;
///
/// // MIN_POSITIVE / 2 is subnormal, so the product switches to `Scaled`.
/// let tiny = PromotingProduct::ONE.multiplied(f64::MIN_POSITIVE)?.multiplied(0.5)?;
/// assert!(tiny.native().is_none());
/// assert_eq!(tiny.round()?, f64::MIN_POSITIVE * 0.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotingProduct<S: RealScalar> {
    value: Product<S>,
}

impl<S: RealScalar> PromotingProduct<S> {
    /// The empty product, `1`.
    pub const ONE: Self = Self {
        value: Product::Native(S::ONE),
    };

    /// Creates a product equal to `value`, in [`Scaled`] form if `value` is
    /// subnormal.
    ///
    /// # Errors
    ///
    /// [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    /// infinite.
    #[inline]
    pub fn from_value(value: S) -> Result<Self, SciOxideError> {
        if value.is_normal() || value == S::ZERO {
            Ok(Self {
                value: Product::Native(value),
            })
        } else if value.is_finite() {
            Ok(Self::from_scaled(Scaled::from_value(value)?))
        } else {
            Err(non_finite_input("promoting product must be finite"))
        }
    }

    /// Creates a product in [`Scaled`] form; it stays in that form.
    #[inline]
    pub const fn from_scaled(value: Scaled<S>) -> Self {
        Self {
            value: Product::Extended(value),
        }
    }

    /// Returns the product as a plain scalar, or `None` once it is in [`Scaled`] form.
    #[inline]
    pub const fn native(self) -> Option<S> {
        match self.value {
            Product::Native(value) => Some(value),
            Product::Extended(_) => None,
        }
    }

    /// Returns the product as a [`Scaled`] value.
    ///
    /// # Errors
    ///
    /// Returns an error if the plain value is not finite.
    #[inline]
    pub fn scaled(self) -> Result<Scaled<S>, SciOxideError> {
        match self.value {
            Product::Native(value) => Scaled::from_value(value),
            Product::Extended(value) => Ok(value),
        }
    }

    /// Returns the product multiplied by `factor`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `factor` is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    #[inline]
    pub fn multiplied(self, factor: S) -> Result<Self, SciOxideError> {
        if let Product::Native(value) = self.value {
            let product = value * factor;
            if product.is_normal() || value == S::ZERO || factor == S::ZERO {
                return Ok(Self {
                    value: Product::Native(product),
                });
            }
        }
        self.extended_product(Scaled::from_value(factor)?)
    }

    /// Returns the product of two products, in [`Scaled`] form if either is.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a plain value is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    #[inline]
    pub fn multiplied_by(self, other: Self) -> Result<Self, SciOxideError> {
        match other.value {
            Product::Native(factor) => self.multiplied(factor),
            Product::Extended(factor) => self.extended_product(factor),
        }
    }

    #[cold]
    #[inline(never)]
    fn extended_product(self, factor: Scaled<S>) -> Result<Self, SciOxideError> {
        Ok(Self::from_scaled(self.scaled()?.mul(factor)?))
    }

    /// Returns the absolute value, in the same form.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            value: match self.value {
                Product::Native(value) => Product::Native(value.abs()),
                Product::Extended(value) => Product::Extended(value.abs()),
            },
        }
    }

    /// Returns the product rounded to `S`; a plain value is returned unchanged.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if a [`Scaled`] value is
    /// outside the finite range of `S`.
    #[inline]
    pub fn round(self) -> Result<S, SciOxideError> {
        match self.value {
            Product::Native(value) => Ok(value),
            Product::Extended(value) => value.round(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Sum<S: RealScalar> {
    Native(CompensatedSum<S>),
    Extended(ScaledAccumulator<S>),
}

/// Compensated sum that switches to a [`ScaledAccumulator`] when it leaves the
/// range of `S`.
///
/// The sum is a Neumaier [`CompensatedSum`] while every addend and rounded
/// product is normal or zero and both parts stay finite. The first subnormal
/// product or overflow switches it once to a [`ScaledAccumulator`]: both parts
/// are converted exactly, and a subnormal or overflowing product is recomputed
/// from its factors. No term is lost or added twice. The sum is compensated, not
/// exact: neither its sign nor its rounding is guaranteed to be exact.
///
/// The bulk constructors sum up to 64 terms of one sign in four plain lanes
/// (addition depth at most 17) and terms of mixed sign as in
/// [`compensated_reduce`](crate::compensated_reduce).
///
/// # Examples
///
/// ```
/// use scioxide_core::PromotingCompensatedSum;
///
/// // MAX + MAX overflows `f64`, so the sum switches to extended range.
/// let mut sum = PromotingCompensatedSum::<f64>::ZERO;
/// sum.add_value(f64::MAX)?;
/// sum.add_value(f64::MAX)?;
/// sum.add_value(-f64::MAX)?;
/// assert!(sum.native().is_none());
/// assert_eq!(sum.round()?, f64::MAX);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotingCompensatedSum<S: RealScalar> {
    sum: Sum<S>,
}

impl<S: RealScalar> PromotingCompensatedSum<S> {
    /// The empty sum; both parts are `+0`.
    pub const ZERO: Self = Self::from_compensated(CompensatedSum::ZERO);

    /// Creates a sum that continues `sum`.
    #[inline]
    pub const fn from_compensated(sum: CompensatedSum<S>) -> Self {
        Self {
            sum: Sum::Native(sum),
        }
    }

    /// Returns the compensated sum `Σ aᵢ bᵢ` of finite products.
    ///
    /// If a product is subnormal or the plain sum overflows, the cloned iterator
    /// is walked again and the products are recomputed in extended range.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    pub fn from_products(
        mut terms: impl Iterator<Item = (S, S)> + Clone,
    ) -> Result<Self, SciOxideError> {
        let mut normal = true;
        let reduced = native_values(terms.clone().map(|(a, b)| {
            let product = a * b;
            normal &= product.abs() >= S::MIN_POSITIVE || a == S::ZERO || b == S::ZERO;
            product
        }));
        if normal && let Some(sum) = reduced {
            return Ok(Self::from_compensated(sum));
        }
        terms.try_fold(Self::ZERO, |mut sum, (a, b)| {
            sum.add_product(a, b)?;
            Ok(sum)
        })
    }

    /// Returns the compensated sum of finite values; subnormal values are kept.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a value is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    pub fn from_values(mut values: impl Iterator<Item = S> + Clone) -> Result<Self, SciOxideError> {
        if let Some(sum) = native_values(values.clone()) {
            return Ok(Self::from_compensated(sum));
        }
        values.try_fold(Self::ZERO, |mut sum, value| {
            sum.add_value(value)?;
            Ok(sum)
        })
    }

    /// Adds both parts of `other` to this sum, switching to extended range if
    /// either sum is extended or the plain merge overflows.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if the
    /// extended exponent range is exceeded.
    #[inline]
    pub fn merge(&mut self, other: Self) -> Result<(), SciOxideError> {
        if let (Sum::Native(left), Sum::Native(right)) = (self.sum, other.sum) {
            let mut left = left;
            let [hi, lo] = right.parts();
            if left.add(hi).is_ok() && left.add(lo).is_ok() {
                self.sum = Sum::Native(left);
                return Ok(());
            }
        }
        match other.sum {
            Sum::Native(right) => {
                let [hi, lo] = right.parts();
                self.extended(|left| {
                    left.add_value(hi)?;
                    left.add_value(lo)
                })
            }
            Sum::Extended(right) => self.extended(|left| left.merge(right)),
        }
    }

    /// Adds a finite value, switching to extended range on overflow.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    #[inline]
    pub fn add_value(&mut self, value: S) -> Result<(), SciOxideError> {
        if let Sum::Native(sum) = &mut self.sum
            && sum.add(value).is_ok()
        {
            return Ok(());
        }
        self.extended(|sum| sum.add_value(value))
    }

    /// Adds `a · b`.
    ///
    /// A rounded normal product has full relative precision; a subnormal or
    /// overflowing product is recomputed in extended range from the factors.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    #[inline]
    pub fn add_product(&mut self, a: S, b: S) -> Result<(), SciOxideError> {
        if let Sum::Native(sum) = &mut self.sum {
            let product = a * b;
            if (product.is_normal() || a == S::ZERO || b == S::ZERO) && sum.add(product).is_ok() {
                return Ok(());
            }
        }
        self.extended(|sum| sum.add_product(Scaled::from_value(a)?, Scaled::from_value(b)?))
    }

    /// Adds `weight · value`; a [`Scaled`]-form weight switches the sum to
    /// extended range.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if
    ///   the extended exponent range is exceeded.
    #[inline]
    pub fn add_weighted(
        &mut self,
        weight: PromotingProduct<S>,
        value: S,
    ) -> Result<(), SciOxideError> {
        match weight.value {
            Product::Native(weight) => self.add_product(weight, value),
            Product::Extended(weight) => {
                self.extended(|sum| sum.add_product(weight, Scaled::from_value(value)?))
            }
        }
    }

    /// Adds a [`Scaled`] value, switching the sum to extended range first.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if the
    /// extended exponent range is exceeded.
    #[inline]
    pub fn add_scaled(&mut self, value: Scaled<S>) -> Result<(), SciOxideError> {
        self.extended(|sum| sum.add_scaled(value))
    }

    /// Promote once and apply an all-or-nothing extended update; a failed update
    /// leaves the promoted parts unchanged.
    fn extended(
        &mut self,
        update: impl FnOnce(&mut ScaledAccumulator<S>) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let mut sum = match self.sum {
            Sum::Native(sum) => {
                let [hi, lo] = sum.parts();
                ScaledAccumulator::from_parts(Scaled::from_value(hi)?, Scaled::from_value(lo)?)
            }
            Sum::Extended(sum) => sum,
        };
        let result = update(&mut sum);
        self.sum = Sum::Extended(sum);
        result
    }

    /// Returns the plain compensated sum, or `None` once the sum is in extended range.
    #[inline]
    pub const fn native(self) -> Option<CompensatedSum<S>> {
        match self.sum {
            Sum::Native(sum) => Some(sum),
            Sum::Extended(_) => None,
        }
    }

    /// Returns the sum rounded to one [`Scaled`] value.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`ExponentOutOfRange`](crate::NumericalIssue::ExponentOutOfRange)) if the
    /// extended exponent range is exceeded.
    #[inline]
    pub fn total(self) -> Result<Scaled<S>, SciOxideError> {
        match self.sum {
            Sum::Native(sum) => {
                let [hi, lo] = sum.parts();
                ScaledAccumulator::from_parts(Scaled::from_value(hi)?, Scaled::from_value(lo)?)
                    .total()
            }
            Sum::Extended(sum) => sum.total(),
        }
    }

    /// Returns the sum rounded to `S`.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the rounded sum is
    /// outside the finite range of `S`.
    #[inline(always)]
    pub fn round(self) -> Result<S, SciOxideError> {
        match self.sum {
            Sum::Native(sum) => sum.total(),
            Sum::Extended(sum) => sum.round(),
        }
    }

    /// Returns `self − other`, computed from both unrounded sums and rounded
    /// once; an exact zero difference is `+0`.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the rounded
    /// difference is outside the finite range of `S`.
    #[inline]
    pub fn difference(self, other: Self) -> Result<S, SciOxideError> {
        if let (Sum::Native(left), Sum::Native(right)) = (self.sum, other.sum)
            && let Some(difference) = native_difference(left.parts(), right.parts())
        {
            return Ok(difference);
        }
        self.total()?.sub(other.total()?)?.round()
    }
}

/// A short same-sign block has addition depth at most 17. Cancelling
/// observations use the shared compensated order; failures reject the
/// candidate so the operands can be replayed in extended range.
fn native_values<S: RealScalar>(values: impl Iterator<Item = S>) -> Option<CompensatedSum<S>> {
    if values.size_hint().1.is_some_and(|n| n <= 64) {
        let mut block = [S::ZERO; 64];
        let mut length = 0;
        let mut signs = 0_u8;
        for (slot, value) in block.iter_mut().zip(values) {
            signs |= u8::from(value.is_sign_negative()) + 1;
            *slot = value;
            length += 1;
        }
        if signs != 3 {
            let [total] = four_lane_sums(length, |i| [block[i]]);
            return CompensatedSum::from_parts(total, S::ZERO).ok();
        }
        let [sum] = compensated_reduce(block[..length].iter().map(|&value| [value])).ok()?;
        Some(sum)
    } else {
        let [sum] = compensated_reduce(values.map(|value| [value])).ok()?;
        Some(sum)
    }
}

/// Subtract two stored native representations without intermediate error
/// construction. A failed native candidate replays the original scaled parts.
#[inline(always)]
fn native_difference<S: RealScalar>(left: [S; 2], right: [S; 2]) -> Option<S> {
    let [left_hi, left_lo] = left;
    let [hi, lo] = right;
    if left_lo == S::ZERO && lo == S::ZERO {
        // A single IEEE subtraction already rounds this complete
        // difference once; an empty correction cannot change it.
        let difference = left_hi - hi;
        if difference.is_finite() {
            return Some(if difference == S::ZERO {
                S::ZERO
            } else {
                difference
            });
        }
    }
    // TwoSum retains the same exact residual as the checked Neumaier
    // update. Nonfinite parts are absorbing, so check the completed
    // candidate once and replay original parts on range breakdown.
    let (sum, error) = two_sum(left_hi, -hi);
    let correction = left_lo + error;
    let (sum, correction) = if lo == S::ZERO {
        (sum, correction)
    } else {
        let (sum, error) = two_sum(sum, -lo);
        (sum, correction + error)
    };
    let difference = sum + correction;
    difference.is_finite().then_some(if difference == S::ZERO {
        S::ZERO
    } else {
        difference
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn native<S: RealScalar>(hi: S, lo: S) -> PromotingCompensatedSum<S> {
        PromotingCompensatedSum::from_compensated(CompensatedSum::from_parts(hi, lo).unwrap())
    }

    #[test]
    fn empty_corrections_round_differences_once_and_retained_parts_survive() {
        fn check<S: RealScalar>() {
            let large = S::ONE / S::EPSILON;
            let zero = PromotingCompensatedSum::ZERO;
            assert_eq!(
                native(large, S::HALF)
                    .difference(native(large, S::ZERO))
                    .unwrap(),
                S::HALF
            );
            assert_eq!(
                native(S::MIN_SUBNORMAL, S::ZERO).difference(zero).unwrap(),
                S::MIN_SUBNORMAL
            );
            let difference = native(-S::ZERO, S::ZERO).difference(zero).unwrap();
            assert_eq!(difference, S::ZERO);
            assert!(!difference.is_sign_negative());
            assert!(
                native(S::MAX, S::ZERO)
                    .difference(native(-S::MAX, S::ZERO))
                    .is_err()
            );
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn short_product_blocks_fit_the_same_sign_allowance() {
        fn check<S: RealScalar>() {
            let sum = PromotingCompensatedSum::from_products(core::iter::repeat_n(
                (S::ONE, S::EPSILON),
                64,
            ))
            .unwrap();
            assert!(sum.native().is_some());
            assert!(
                (sum.round().unwrap() - S::try_from_usize_exact(64).unwrap() * S::EPSILON).abs()
                    <= S::EPSILON
            );
        }
        check::<f32>();
        check::<f64>();
    }

    fn promotion<S: RealScalar>() {
        let small = S::MIN_POSITIVE * S::EPSILON;
        let mut tiny = PromotingCompensatedSum::ZERO;
        tiny.add_product(small, S::HALF).unwrap();
        assert!(tiny.native().is_none());
        tiny.add_product(small, S::HALF).unwrap();
        assert_eq!(tiny.round().unwrap(), small);

        let mut large = PromotingCompensatedSum::ZERO;
        large.add_value(S::MAX).unwrap();
        large.add_value(S::MAX).unwrap();
        assert!(large.native().is_none());
        large.add_value(-S::MAX).unwrap();
        assert_eq!(large.round().unwrap(), S::MAX);

        let mut compensated = PromotingCompensatedSum::ZERO;
        compensated.add_value(S::ONE / S::EPSILON).unwrap();
        compensated.add_value(S::HALF).unwrap();
        assert!(compensated.native().is_some());
        // Promotion must retain the half hidden in the native correction.
        compensated.add_product(small, S::HALF).unwrap();
        compensated.add_value(-S::ONE / S::EPSILON).unwrap();
        assert_eq!(compensated.round().unwrap(), S::HALF);

        let weight = PromotingProduct::ONE
            .multiplied(small)
            .unwrap()
            .multiplied(S::HALF)
            .unwrap();
        assert!(weight.native().is_none());
        let mut recovered = PromotingCompensatedSum::ZERO;
        recovered.add_weighted(weight, S::TWO).unwrap();
        assert_eq!(recovered.round().unwrap(), small);
        let mut negative = PromotingCompensatedSum::ZERO;
        negative.add_weighted(weight.abs(), -S::TWO).unwrap();
        assert_eq!(negative.round().unwrap(), -small);

        let mut scaled = PromotingCompensatedSum::ZERO;
        scaled.add_value(S::ONE).unwrap();
        scaled
            .add_scaled(Scaled::from_value(S::ONE).unwrap())
            .unwrap();
        assert!(scaled.native().is_none());
        assert_eq!(scaled.round().unwrap(), S::TWO);
    }

    #[test]
    fn native_and_extended_boundaries_preserve_both_precisions() {
        promotion::<f32>();
        promotion::<f64>();
    }

    #[test]
    fn products_keep_native_zero_and_promote_only_out_of_range() {
        fn check<S: RealScalar>() {
            let zero = PromotingProduct::ONE.multiplied(S::ZERO).unwrap();
            assert_eq!(zero.native(), Some(S::ZERO));
            assert_eq!(
                zero.multiplied(S::MIN_SUBNORMAL).unwrap().native(),
                Some(S::ZERO)
            );
            assert_eq!(
                PromotingProduct::from_value(S::MIN_SUBNORMAL)
                    .unwrap()
                    .native(),
                None
            );
            assert!(PromotingProduct::from_value(S::NAN).is_err());
            let half = PromotingProduct::from_value(S::HALF).unwrap();
            let tiny = PromotingProduct::from_scaled(Scaled::from_value(S::MIN_POSITIVE).unwrap());
            let product = half.multiplied_by(tiny).unwrap();
            assert!(product.native().is_none());
            assert_eq!(product.round().unwrap(), S::MIN_POSITIVE * S::HALF);
            assert_eq!(
                half.multiplied_by(half).unwrap().native(),
                Some(S::HALF * S::HALF)
            );
            assert_eq!(
                PromotingProduct::from_value(-S::TWO)
                    .unwrap()
                    .abs()
                    .native(),
                Some(S::TWO)
            );
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn invalid_products_and_true_overflow_are_errors() {
        let mut sum = PromotingCompensatedSum::<f64>::ZERO;
        assert!(sum.add_product(f64::NAN, 0.0).is_err());
        assert!(sum.add_value(f64::INFINITY).is_err());
        sum.add_product(f64::MAX, f64::MAX).unwrap();
        assert!(sum.round().is_err());
    }

    #[test]
    fn retained_product_blocks_recover_range_and_cancellation() {
        fn check<S: RealScalar>() {
            let tiny = S::MIN_POSITIVE * S::EPSILON;
            let terms = [(tiny, S::HALF), (tiny, S::HALF)];
            assert_eq!(
                PromotingCompensatedSum::from_products(terms.into_iter())
                    .unwrap()
                    .round()
                    .unwrap(),
                tiny
            );
            let terms = [(S::MAX, S::MAX), (S::ONE, S::ONE), (-S::MAX, S::MAX)];
            assert_eq!(
                PromotingCompensatedSum::from_products(terms.into_iter())
                    .unwrap()
                    .round()
                    .unwrap(),
                S::ONE
            );
            assert!(
                PromotingCompensatedSum::from_products([(S::NAN, S::ZERO)].into_iter()).is_err()
            );
            // A native merge that overflows promotes both sides and keeps cancelling.
            let mut merged = PromotingCompensatedSum::from_values([S::MAX].into_iter()).unwrap();
            merged
                .merge(PromotingCompensatedSum::from_values([S::MAX].into_iter()).unwrap())
                .unwrap();
            assert!(merged.native().is_none());
            merged.add_value(-S::MAX).unwrap();
            assert_eq!(merged.round().unwrap(), S::MAX);
        }
        check::<f32>();
        check::<f64>();
    }
}
