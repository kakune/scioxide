//! Exact running sums of real products, held in a 192-bit integer while possible.
use super::accumulator::ExactTerm;
use super::{RangeAccumulator, checked_final_rounding};
use crate::scalar::{FiniteParts, finite_parts};
use crate::scaled::rounding::rounded_words_at_exponent;
use crate::{RealScalar, SciOxideError};
use core::marker::PhantomData;

/// Finite real factor, checked and split into significand and exponent once,
/// for reuse in many exact products.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::{ProductFactor, RealProductSum};
///
/// // 0.5 · (1 + 2) = 1.5
/// let weight = ProductFactor::new(0.5_f64)?;
/// let mut sum = RealProductSum::new();
/// sum.add_weighted_pair_factors(weight, ProductFactor::new(1.0)?, ProductFactor::new(2.0)?)?;
/// assert_eq!(sum.finish()?, 1.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ProductFactor<S: RealScalar> {
    parts: FiniteParts,
    marker: PhantomData<S>,
}
impl<S: RealScalar> ProductFactor<S> {
    /// Creates a factor equal to `value`.
    ///
    /// # Errors
    ///
    /// [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    /// infinite.
    #[inline]
    pub fn new(value: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            parts: finite_parts(value, "product factor must be finite")?,
            marker: PhantomData,
        })
    }

    /// Returns the same value with trailing zero bits removed from the
    /// significand, which can make later products cheaper when the factor is
    /// reused many times.
    #[inline]
    pub fn normalized(self) -> Self {
        Self {
            parts: self.parts.normalized(),
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Bin {
    low: u128,
    high: u64,
}
impl Bin {
    #[inline]
    fn words(self) -> [u64; 3] {
        [self.low as u64, (self.low >> 64) as u64, self.high]
    }

    // No bits may be discarded when moving the shared binary quantum down.
    #[inline]
    fn shifted(self, shift: u32) -> Option<Self> {
        let width = if self.high == 0 {
            128 - self.low.leading_zeros()
        } else {
            192 - self.high.leading_zeros()
        };
        if width == 0 {
            return Some(self);
        }
        if shift >= 192 || width + shift > 192 {
            return None;
        }
        Some(if shift == 0 {
            self
        } else if shift < 128 {
            Self {
                low: self.low << shift,
                high: if shift < 64 { self.high << shift } else { 0 }
                    | (self.low >> (128 - shift)) as u64,
            }
        } else {
            Self {
                low: 0,
                high: (self.low as u64) << (shift - 128),
            }
        })
    }

    #[inline]
    fn signed(self, negative: bool) -> Option<Self> {
        if self.high > i64::MAX as u64 {
            return None;
        }
        let mask = 0_u128.wrapping_sub(u128::from(negative));
        let low = (self.low ^ mask).wrapping_add(u128::from(negative));
        let high = if negative {
            -(self.high as i64) - i64::from(self.low != 0)
        } else {
            self.high as i64
        };
        Some(Self {
            low,
            high: high as u64,
        })
    }

    #[inline]
    fn magnitude(self) -> (Self, bool) {
        let negative = (self.high as i64) < 0;
        if negative {
            (
                Self {
                    low: self.low.wrapping_neg(),
                    high: (self.high as i64).unsigned_abs() - u64::from(self.low != 0),
                },
                true,
            )
        } else {
            (self, false)
        }
    }

    #[inline]
    fn added_signed(self, other: Self) -> Option<Self> {
        let (low, carry) = self.low.overflowing_add(other.low);
        let high = (self.high as i64)
            .checked_add(other.high as i64)?
            .checked_add(i64::from(carry))?;
        Some(Self {
            low,
            high: high as u64,
        })
    }
}

/// Exact running sum of real products whose partial sums can be rounded
/// cheaply.
///
/// While the exponents of the terms span a small enough range, the sum is held
/// in one signed 192-bit integer. Otherwise it moves, without rounding, to a
/// [`RangeAccumulator`](crate::RangeAccumulator). Every rounded partial sum is
/// bit-identical to that of a [`RangeAccumulator`](crate::RangeAccumulator),
/// including ties, subnormals and cancellation. No FMA is needed.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::RealProductSum;
///
/// let mut sum = RealProductSum::<f64>::new();
/// sum.add_product(1e100, 1.0)?;
/// sum.add_product(1.0, 1.0)?;
/// sum.add_product(-1e100, 1.0)?;
/// assert_eq!(sum.finish()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct RealProductSum<S: RealScalar> {
    integer: Bin,
    exponent: i32,
    wide: Option<RangeAccumulator<S>>,
    frame_shift: u32,
    frame_scale: f64,
    frame_mask: u64,
    frame_valid: bool,
}
impl<S: RealScalar> Default for RealProductSum<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> RealProductSum<S> {
    /// Returns an empty sum, whose value is `+0`.
    pub const fn new() -> Self {
        Self {
            integer: Bin { low: 0, high: 0 },
            exponent: 0,
            wide: None,
            frame_shift: 0,
            frame_scale: 0.0,
            frame_mask: 0,
            frame_valid: false,
        }
    }

    /// Resets the sum to zero.
    pub fn clear(&mut self) {
        self.integer = Bin::default();
        self.exponent = 0;
        self.wide = None;
        self.frame_valid = false;
    }

    /// Returns the sign of the exact sum, without rounding.
    ///
    /// The sign is exact even when the sum is smaller than the smallest
    /// subnormal or would overflow when rounded.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use scioxide_core::backend::RealProductSum;
    ///
    /// // The exact sum 2⁻¹⁰⁷⁵ rounds to zero but is positive.
    /// let mut sum = RealProductSum::<f64>::new();
    /// sum.add_product(f64::from_bits(1), 0.5)?;
    /// assert_eq!(sum.finish()?, 0.0);
    /// assert_eq!(sum.sign(), Ordering::Greater);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn sign(&self) -> core::cmp::Ordering {
        if let Some(wide) = &self.wide {
            wide.sign()
        } else if (self.integer.high as i64) < 0 {
            core::cmp::Ordering::Less
        } else if self.integer.low == 0 && self.integer.high == 0 {
            core::cmp::Ordering::Equal
        } else {
            core::cmp::Ordering::Greater
        }
    }

    /// Adds a finite value exactly; cheaper than `add_product(value, 1.0)`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    ///   infinite; the sum is unchanged.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the accumulator capacity is exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::RealProductSum;
    ///
    /// let mut sum = RealProductSum::<f64>::new();
    /// for value in [1e100, 1.0, -1e100] {
    ///     sum.add_value(value)?;
    /// }
    /// assert_eq!(sum.finish()?, 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn add_value(&mut self, value: S) -> Result<(), SciOxideError> {
        let value = finite_parts(value, "sum addend must be finite")?;
        if value.significand() == 0 {
            return Ok(());
        }
        // Low zero bits reserve alignment room for smaller following terms
        // and guard bits for cached exact rounding. A 53-bit significand plus
        // 32 zero bits still fits the signed narrow frame without rescaling.
        if self.wide.is_none() && self.integer.low == 0 && self.integer.high == 0 {
            let significand = i128::from(value.significand()) << 32;
            let integer = if value.negative() {
                -significand
            } else {
                significand
            };
            self.integer = Bin {
                low: integer as u128,
                high: (integer >> 127) as u64,
            };
            self.exponent = i32::from(value.exponent()) - 32;
            self.frame_valid = false;
            return Ok(());
        }
        self.add_decoded(
            u128::from(value.significand()),
            i32::from(value.exponent()),
            value.negative(),
        )
    }

    /// Adds the exact product `left · right`, which need not be representable
    /// in `S`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
    ///   infinite, even when the other is zero; the sum is unchanged.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the accumulator capacity is exceeded.
    #[inline(always)]
    pub fn add_product(&mut self, left: S, right: S) -> Result<(), SciOxideError> {
        let left = finite_parts(left, "left product factor must be finite")?;
        let right = finite_parts(right, "right product factor must be finite")?;
        let (product, exponent, negative) = left.product(right);
        self.add_decoded(product, exponent, negative)
    }

    /// Adds `weight · (left + right)` exactly, without rounding the pair sum or
    /// the products.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if any argument is NaN
    ///   or infinite, even when `weight` is zero; the sum is unchanged.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the accumulator capacity is exceeded.
    #[inline(always)]
    pub fn add_weighted_pair(&mut self, weight: S, left: S, right: S) -> Result<(), SciOxideError> {
        self.add_weighted_pair_factors(
            ProductFactor::new(weight)?.normalized(),
            ProductFactor::new(left)?,
            ProductFactor::new(right)?,
        )
    }

    /// Adds `weight · (left + right)` exactly, for factors that are already
    /// checked.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability) if
    /// the accumulator capacity is exceeded.
    #[inline(always)]
    pub fn add_weighted_pair_factors(
        &mut self,
        weight: ProductFactor<S>,
        left: ProductFactor<S>,
        right: ProductFactor<S>,
    ) -> Result<(), SciOxideError> {
        let (w, a, b) = (weight.parts, left.parts, right.parts);
        if let Some((product, exponent, negative)) = Self::weighted_pair_term(w, a, b) {
            self.add_decoded(product, exponent, negative)
        } else {
            self.add_weighted_parts(w, a, b)
        }
    }

    #[inline(always)]
    fn weighted_pair_term(
        w: FiniteParts,
        a: FiniteParts,
        b: FiniteParts,
    ) -> Option<(u128, i32, bool)> {
        let exponent = a.exponent().min(b.exponent());
        let ashift = (a.exponent() - exponent) as u32;
        let bshift = (b.exponent() - exponent) as u32;
        if ashift <= 9 && bshift <= 9 {
            // 53 significand bits, at most nine alignment bits, and one
            // carry fit a signed i64. Its product with the weight fits u128.
            let ai = (a.significand() as i64) << ashift;
            let bi = (b.significand() as i64) << bshift;
            let integer =
                (if a.negative() { -ai } else { ai }) + (if b.negative() { -bi } else { bi });
            return Some((
                u128::from(integer.unsigned_abs()) * u128::from(w.significand()),
                i32::from(exponent) + i32::from(w.exponent()),
                (integer < 0) ^ w.negative(),
            ));
        }
        None
    }

    /// Adds `weight · (left + right)` exactly and returns the new sum,
    /// correctly rounded.
    ///
    /// If the rounded sum overflows, the exact sum is still updated, so a later
    /// update can cancel it back into range.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability) if
    /// the rounded sum is outside the finite range or the accumulator capacity
    /// is exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::{ProductFactor, RealProductSum};
    ///
    /// let one = ProductFactor::new(1.0_f64)?;
    /// let mut sum = RealProductSum::new();
    /// assert_eq!(sum.add_weighted_pair_prefix(one, one, one)?, 2.0);
    /// assert_eq!(sum.add_weighted_pair_prefix(one, one, one)?, 4.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn add_weighted_pair_prefix(
        &mut self,
        weight: ProductFactor<S>,
        left: ProductFactor<S>,
        right: ProductFactor<S>,
    ) -> Result<S, SciOxideError> {
        let (w, a, b) = (weight.parts, left.parts, right.parts);
        if let Some((product, exponent, negative)) = Self::weighted_pair_term(w, a, b) {
            let shift = exponent - self.exponent;
            if product != 0 && self.wide.is_none() && (0..=64).contains(&shift) {
                let term = Self::shifted_product(product, shift as u32);
                if let Some(integer) = self.commit_narrow(term, negative) {
                    return if let Some(value) = self.try_finish_narrow(integer) {
                        Ok(value)
                    } else {
                        self.finish()
                    };
                }
            }
            self.add_decoded(product, exponent, negative)?;
        } else {
            self.add_weighted_parts(w, a, b)?;
        }
        self.finish()
    }

    #[cold]
    fn add_weighted_parts(
        &mut self,
        w: FiniteParts,
        a: FiniteParts,
        b: FiniteParts,
    ) -> Result<(), SciOxideError> {
        let exponent = a.exponent().min(b.exponent());
        let ashift = (a.exponent() - exponent) as u32;
        let bshift = (b.exponent() - exponent) as u32;
        // Each significand has at most 53 bits. Reserving a sign and a carry
        // bit proves both shifted terms and their signed sum fit i128.
        if ashift <= 73 && bshift <= 73 {
            let ai = i128::from(a.significand()) << ashift;
            let bi = i128::from(b.significand()) << bshift;
            let integer =
                (if a.negative() { -ai } else { ai }) + (if b.negative() { -bi } else { bi });
            let zeros = w.significand().trailing_zeros().min(63);
            if let Some(product) = integer
                .unsigned_abs()
                .checked_mul(u128::from(w.significand() >> zeros))
            {
                return self.add_decoded(
                    product,
                    i32::from(exponent) + i32::from(w.exponent()) + zeros as i32,
                    (integer < 0) ^ w.negative(),
                );
            }
        }
        let (product, exponent, negative) = a.product(w);
        self.add_decoded(product, exponent, negative)?;
        let (product, exponent, negative) = b.product(w);
        self.add_decoded(product, exponent, negative)
    }

    #[inline(always)]
    fn add_decoded(
        &mut self,
        product: u128,
        exponent: i32,
        negative: bool,
    ) -> Result<(), SciOxideError> {
        if product == 0 {
            return Ok(());
        }
        if self.wide.is_none() && self.add_compact(product, exponent, negative).is_some() {
            return Ok(());
        }
        self.add_wide(product, exponent, negative)
    }

    #[cold]
    fn add_wide(
        &mut self,
        product: u128,
        exponent: i32,
        negative: bool,
    ) -> Result<(), SciOxideError> {
        let term = [ExactTerm::new(
            [product as u64, (product >> 64) as u64, 0],
            exponent,
            negative,
        )];
        if let Some(wide) = &mut self.wide {
            wide.check_terms(&term)?;
            wide.commit_terms(&term);
        } else {
            let mut wide = RangeAccumulator::new();
            let (magnitude, negative) = self.integer.magnitude();
            let terms = [ExactTerm::new(magnitude.words(), self.exponent, negative)];
            wide.check_terms(&terms)?;
            wide.commit_terms(&terms);
            wide.check_terms(&term)?;
            wide.commit_terms(&term);
            self.wide = Some(wide);
        }
        Ok(())
    }

    #[inline(always)]
    fn add_compact(&mut self, product: u128, exponent: i32, negative: bool) -> Option<()> {
        // The common frame needs no rescaling of the running integer. The
        // 64-bit shift guard admits every u128 product with no lost bits.
        let shift = exponent - self.exponent;
        if (0..=64).contains(&shift) {
            let term = Self::shifted_product(product, shift as u32);
            if self.commit_narrow(term, negative).is_some() {
                return Some(());
            }
            self.integer = self.integer.added_signed(term.signed(negative)?)?;
            return Some(());
        }
        self.add_compact_rescaled(product, exponent, negative)
    }

    #[inline(always)]
    fn shifted_product(product: u128, shift: u32) -> Bin {
        Bin {
            low: product << shift,
            high: if shift == 0 {
                0
            } else {
                (product >> (128 - shift)) as u64
            },
        }
    }

    #[inline(always)]
    fn commit_narrow(&mut self, term: Bin, negative: bool) -> Option<i128> {
        if self.integer.high as i64 != ((self.integer.low as i128) >> 127) as i64
            || term.high != 0
            || term.low > i128::MAX as u128
        {
            return None;
        }
        let magnitude = term.low as i128;
        let signed = if negative { -magnitude } else { magnitude };
        let sum = (self.integer.low as i128).checked_add(signed)?;
        self.integer = Bin {
            low: sum as u128,
            high: (sum >> 127) as u64,
        };
        Some(sum)
    }

    #[cold]
    fn add_compact_rescaled(&mut self, product: u128, exponent: i32, negative: bool) -> Option<()> {
        let empty = self.integer.low == 0 && self.integer.high == 0;
        let base = if empty {
            exponent
        } else {
            self.exponent.min(exponent)
        };
        let term = Bin {
            low: product,
            high: 0,
        }
        .shifted((exponent - base) as u32)?
        .signed(negative)?;
        let (magnitude, sign) = self.integer.magnitude();
        let previous = magnitude
            .shifted((self.exponent - base).max(0) as u32)?
            .signed(sign)?;
        let integer = previous.added_signed(term)?;
        self.integer = integer;
        if self.exponent != base {
            self.frame_valid = false;
        }
        self.exponent = base;
        Some(())
    }

    pub(super) fn add_block(&mut self, pairs: &[(S, S)]) -> Result<(), SciOxideError> {
        if let Some(wide) = &mut self.wide {
            return wide.add_product_block(pairs);
        }
        for &(left, right) in pairs {
            self.add_product(left, right)?;
        }
        Ok(())
    }

    /// Returns the exact sum rounded once to nearest, ties to even.
    ///
    /// An exact zero is `+0`; a nonzero sum that underflows keeps its sign.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the rounded sum is
    /// outside the finite range. The exact sum is kept and can be updated.
    #[inline(always)]
    pub fn finish(&mut self) -> Result<S, SciOxideError> {
        if let Some(wide) = &self.wide {
            return wide.finish();
        }
        if self.integer.low == 0 && self.integer.high == 0 {
            return Ok(S::ZERO);
        }
        if self.integer.high as i64 == ((self.integer.low as i128) >> 127) as i64
            && let Some(value) = self.try_finish_narrow(self.integer.low as i128)
        {
            return Ok(value);
        }
        let (magnitude, negative) = self.integer.magnitude();
        let words = magnitude.words();
        checked_final_rounding(
            rounded_words_at_exponent::<S>(&words, negative, self.exponent, false),
            "finite mathematical result is outside the scalar range",
        )
    }
    #[inline(always)]
    fn try_finish_narrow(&mut self, integer: i128) -> Option<S> {
        let minimum = 1_u64 << (S::MANTISSA_DIGITS + 1);
        let mut leading = integer >> self.frame_shift;
        if !self.frame_valid
            || leading != i128::from(leading as i64)
            || (leading as i64).unsigned_abs() <= minimum
        {
            self.prepare_rounding_frame(integer.unsigned_abs());
            leading = integer >> self.frame_shift;
        }
        if self.frame_valid
            && leading == i128::from(leading as i64)
            && (leading as i64).unsigned_abs() > minimum
        {
            // Arithmetic shifting and an odd sticky bit retain signed
            // rounding cells with at least two guard bits. The frame's
            // binary scale is exact throughout the admitted normal range.
            let sticky = integer as u64 & self.frame_mask != 0;
            let retained = leading as i64 | i64::from(sticky);
            let rounded = if S::MANTISSA_DIGITS == 24 {
                f64::from(retained as f32)
            } else {
                retained as f64
            };
            return Some(S::from_f64_ieee(rounded * self.frame_scale));
        }
        None
    }

    // A frame is valid for every retained integer in [2^(p+1), 2^63).
    // Both ends scale strictly inside the normal destination range, so the
    // final multiplication/conversion is exact even after an upward carry.
    #[cold]
    fn prepare_rounding_frame(&mut self, magnitude: u128) {
        let shift = (128 - magnitude.leading_zeros()).saturating_sub(63);
        let exponent = self.exponent + shift as i32;
        let min_normal = crate::scaled::rounding::minimum_subnormal_exponent::<S>()
            + S::MANTISSA_DIGITS as i32
            - 1;
        let max_normal = crate::scaled::rounding::maximum_normal_exponent::<S>();
        self.frame_valid = shift <= 64
            && (-1022..=1023).contains(&exponent)
            && exponent + S::MANTISSA_DIGITS as i32 + 1 > min_normal
            && exponent + 63 < max_normal;
        self.frame_shift = shift;
        if self.frame_valid {
            self.frame_scale = f64::from_bits(((exponent + 1023) as u64) << 52);
            self.frame_mask = if shift == 64 {
                u64::MAX
            } else {
                (1_u64 << shift) - 1
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefixes<S: RealScalar>(pairs: &[(S, S)]) {
        let mut compact = RealProductSum::new();
        let mut reference = RangeAccumulator::new();
        let mut paired = RealProductSum::new();
        for &(left, right) in pairs {
            compact.add_product(left, right).unwrap();
            reference.add_product(left, right).unwrap();
            paired.add_weighted_pair(left, right, -right).unwrap();
            assert_eq!(paired.finish().unwrap(), S::ZERO);
            assert_eq!(paired.sign(), core::cmp::Ordering::Equal);
            assert_eq!(compact.sign(), reference.sign());
            match (compact.finish(), reference.finish()) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits())
                }
                (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
                (actual, expected) => assert_eq!(actual, expected),
            }
        }
    }

    fn cases<S: RealScalar>() {
        let tiny = S::MIN_SUBNORMAL;
        let next = S::ONE.next_after(S::TWO);
        for pairs in [
            [
                (S::ZERO, -S::ZERO),
                (-tiny, S::HALF),
                (tiny, S::HALF),
                (tiny, tiny),
            ],
            [
                (S::MAX, S::TWO),
                (-S::MAX, S::TWO),
                (next, next),
                (-S::ONE, S::ONE),
            ],
            [
                (S::ONE, S::ONE),
                (S::EPSILON, S::HALF),
                (S::EPSILON, S::EPSILON),
                (-S::ONE, S::ONE),
            ],
        ] {
            prefixes(&pairs);
        }
        for exponent in [0, 1, 63, 64, 65, 127, 128, 129, 190, 191, 192, 193, 500] {
            let high = S::ONE.scalbn(exponent);
            if high.is_finite() {
                for sign in [S::ONE, -S::ONE] {
                    prefixes(&[
                        (sign * high, S::ONE),
                        (S::ONE, S::ONE),
                        (-sign * high, S::ONE),
                        (tiny, S::ONE),
                    ]);
                    prefixes(&[
                        (S::ONE, S::ONE),
                        (sign * high, S::ONE),
                        (-sign * high, S::ONE),
                        (tiny, S::ONE),
                    ]);
                }
            }
        }
        let pairs: alloc::vec::Vec<_> = (0..4097)
            .map(|i| {
                let left = S::from_f64_ieee((i * 7919 % 65537) as f64 / 32768.0 - 1.0);
                let right = S::from_f64_ieee((i * 104729 % 65537) as f64 / 32768.0 - 1.0);
                (left, right)
            })
            .collect();
        prefixes(&pairs);
    }

    #[test]
    fn every_prefix_matches_exact_accumulation() {
        cases::<f32>();
        cases::<f64>();
    }

    #[test]
    fn promotion_invalid_input_and_clear_preserve_state() {
        let mut sum = RealProductSum::<f64>::new();
        sum.add_product(1.0, 1.0).unwrap();
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(sum.add_product(0.0, invalid).is_err());
            assert!(sum.add_product(invalid, 0.0).is_err());
            assert!(sum.add_weighted_pair(0.0, 0.0, invalid).is_err());
            assert_eq!(sum.finish().unwrap(), 1.0);
        }
        assert!(sum.wide.is_none());
        sum.add_product(f64::MIN_SUBNORMAL, f64::MIN_SUBNORMAL)
            .unwrap();
        assert!(sum.wide.is_some());
        assert_eq!(sum.finish().unwrap(), 1.0);
        sum.clear();
        assert!(sum.wide.is_none());
        assert_eq!(sum.sign(), core::cmp::Ordering::Equal);
        assert_eq!(sum.finish().unwrap().to_bits(), 0);
        sum.add_product(-f64::MIN_SUBNORMAL, 0.5).unwrap();
        assert_eq!(sum.finish().unwrap().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(sum.sign(), core::cmp::Ordering::Less);
    }

    #[test]
    fn weighted_pairs_match_separate_exact_products() {
        for weight in [0.0, 0.25, 1.5, -1.0, f64::MIN_SUBNORMAL, f64::MAX] {
            let mut sum = RealProductSum::new();
            let mut exact = RangeAccumulator::new();
            for (a, b) in [
                (1.5, 2.25),
                (1e30, -1e30),
                (f64::MAX, -f64::MAX),
                (1.0, f64::MIN_SUBNORMAL),
                (-f64::MIN_SUBNORMAL, -f64::MIN_SUBNORMAL),
                (0.0, -0.0),
            ] {
                sum.add_weighted_pair(weight, a, b).unwrap();
                exact.add_product(weight, a).unwrap();
                exact.add_product(weight, b).unwrap();
                match (sum.finish(), exact.finish()) {
                    (Ok(actual), Ok(expected)) => assert_eq!(actual.to_bits(), expected.to_bits()),
                    (actual, expected) => assert_eq!(actual.is_err(), expected.is_err()),
                }
            }
        }
    }

    #[test]
    fn fused_prefix_overflow_retains_terms_for_cancellation() {
        let one = ProductFactor::new(1.0_f64).unwrap().normalized();
        let positive = ProductFactor::new(f64::MAX).unwrap();
        let negative = ProductFactor::new(-f64::MAX).unwrap();
        let mut sum = RealProductSum::new();
        assert!(
            sum.add_weighted_pair_prefix(one, positive, positive)
                .is_err()
        );
        assert_eq!(
            sum.add_weighted_pair_prefix(one, negative, negative)
                .unwrap()
                .to_bits(),
            0
        );
    }

    fn rounding_patterns<S: RealScalar>() {
        let integers = [
            1,
            (1_u128 << 24) - 1,
            (1_u128 << 53) + 1,
            (1_u128 << 63) + 1,
            (1_u128 << 64) + 1,
            (1_u128 << 100) + (1_u128 << 47),
            (1_u128 << 100) + (1_u128 << 47) + 1,
            u128::MAX,
        ];
        for integer in integers {
            for exponent in [
                -2200, -1074, -1073, -1022, -1021, -200, -149, -126, -125, -53, 0, 64, 127, 971,
                1023,
            ] {
                for negative in [false, true] {
                    let mut sum = RealProductSum::<S>::new();
                    sum.integer = Bin {
                        low: integer,
                        high: 0,
                    }
                    .signed(negative)
                    .unwrap();
                    sum.exponent = exponent;
                    let mut exact = RangeAccumulator::<S>::new();
                    exact.commit_terms(&[ExactTerm::new(
                        [integer as u64, (integer >> 64) as u64, 0],
                        exponent,
                        negative,
                    )]);
                    match (sum.finish(), exact.finish()) {
                        (Ok(actual), Ok(expected)) => {
                            assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits())
                        }
                        (actual, expected) => assert_eq!(actual.is_err(), expected.is_err()),
                    }
                }
            }
        }
    }

    #[test]
    fn native_prefix_rounding_matches_integer_midpoints_and_range_boundaries() {
        rounding_patterns::<f32>();
        rounding_patterns::<f64>();
    }

    #[test]
    fn ordinary_weighted_prefixes_retain_the_compact_frame() {
        let mut sum = RealProductSum::<f64>::new();
        let mut wide = 0;
        let mut high = 0;
        let mut slow_pair = 0;
        let mut previous = ProductFactor::new(-1.0).unwrap();
        for i in 1..65536 {
            let weight =
                ProductFactor::new((i as f64 / 65535.0 - (i - 1) as f64 / 65535.0) * 0.5).unwrap();
            let next = ProductFactor::new((i * 7919 % 65537) as f64 / 32768.0 - 1.0).unwrap();
            slow_pair += usize::from((next.parts.exponent() - previous.parts.exponent()).abs() > 9);
            sum.add_weighted_pair_factors(weight, previous, next)
                .unwrap();
            wide += usize::from(sum.wide.is_some());
            high += usize::from(sum.integer.magnitude().0.high != 0);
            assert!(sum.finish().is_ok());
            previous = next;
        }
        assert_eq!(wide, 0);
        assert_eq!(high, 0);
        assert!(slow_pair < 1024);
    }

    #[test]
    fn carries_and_alignment_do_not_discard_bits() {
        let mut sum = RealProductSum::<f64>::new();
        sum.integer = Bin {
            low: u128::MAX,
            high: i64::MAX as u64,
        };
        assert!(sum.add_compact(1, 0, false).is_none());
        assert_eq!(sum.integer.words(), [u64::MAX, u64::MAX, i64::MAX as u64]);
        sum.add_product(1.0, 1.0).unwrap();
        assert!(sum.wide.is_some());
        // 2^191 - 1 plus one is 2^191, represented exactly after promotion.
        assert_eq!(sum.finish().unwrap(), 2.0_f64.powi(191));
        for shift in [0, 1, 63, 64, 127, 128, 191] {
            let bin = Bin { low: 1, high: 0 }.shifted(shift).unwrap();
            assert_eq!(
                rounded_words_at_exponent::<f64>(&bin.words(), false, 0, false),
                2.0_f64.powi(shift as i32)
            );
        }
        assert!(Bin { low: 1, high: 0 }.shifted(192).is_none());
    }

    #[test]
    fn scalar_rounding_cells_preserve_halfway_carries_and_range_promotion() {
        fn check<S: RealScalar>() {
            let large = S::try_from_f64_lossy(2.0_f64.powi(100)).unwrap();
            let tiny = S::MIN_SUBNORMAL;
            for terms in [
                [
                    (S::ONE, None),
                    (S::EPSILON, Some(S::HALF)),
                    (S::EPSILON * S::EPSILON, None),
                    (-S::ONE, None),
                    (-S::EPSILON, Some(S::HALF)),
                    (-(S::EPSILON * S::EPSILON), None),
                ],
                [
                    (large, None),
                    (S::ONE, None),
                    (-large, None),
                    (tiny, None),
                    (-S::ONE, None),
                    (-tiny, None),
                ],
            ] {
                let mut sum = RealProductSum::<S>::new();
                let mut exact = RangeAccumulator::<S>::new();
                for (value, factor) in terms {
                    if let Some(factor) = factor {
                        sum.add_product(value, factor).unwrap();
                        exact.add_product(value, factor).unwrap();
                    } else {
                        sum.add_value(value).unwrap();
                        exact.add(value).unwrap();
                    }
                    assert_eq!(
                        sum.finish().unwrap().into_f64().to_bits(),
                        exact.finish().unwrap().into_f64().to_bits()
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn scalar_prefixes_and_mixed_products_match_full_range_rounding() {
        fn check<S: RealScalar>() {
            let tiny = S::MIN_POSITIVE * S::EPSILON;
            let mut sum = RealProductSum::<S>::new();
            let mut exact = RangeAccumulator::<S>::new();
            for value in [
                S::MAX,
                S::MAX,
                -S::MAX,
                -S::MAX,
                tiny,
                tiny,
                S::EPSILON,
                -S::EPSILON,
                S::ONE,
                S::MAX,
                -S::MAX,
            ] {
                sum.add_value(value).unwrap();
                exact.add(value).unwrap();
                match (sum.finish(), exact.finish()) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits())
                    }
                    (actual, expected) => assert_eq!(actual.is_err(), expected.is_err()),
                }
            }
            for _ in 0..2 {
                sum.add_product(tiny, S::HALF).unwrap();
                exact.add_product(tiny, S::HALF).unwrap();
                assert_eq!(
                    sum.finish().unwrap().into_f64().to_bits(),
                    exact.finish().unwrap().into_f64().to_bits()
                );
            }
            let before = sum.finish().unwrap();
            assert!(sum.add_value(S::NAN).is_err());
            assert_eq!(sum.finish().unwrap(), before);
        }
        check::<f32>();
        check::<f64>();
    }
}
