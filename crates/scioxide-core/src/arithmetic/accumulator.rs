//! Exact real accumulation and portable fused multiply-add.

use super::{checked_final_rounding, overflow};
use crate::scalar::{FiniteParts, finite_parts};
use crate::scaled::rounding::{
    highest_bit, rounded_integer, rounded_words_at_exponent, shifted_low_u64,
};
use crate::{DomainReason, RealScalar, Scaled, SciOxideError};
#[cfg(doc)]
use crate::{NonFiniteOrigin, NumericalIssue};
use core::cmp::Ordering;
use core::marker::PhantomData;

// The least possible binary64 triple product is 2^-3222. One hundred limbs
// cover the complete triple-product range plus enough carry bits for every
// collection whose length is representable by supported usize targets.
pub(super) const ACCUMULATOR_MIN_EXPONENT: i32 = -3_222;
pub(super) const ACCUMULATOR_LIMBS: usize = 100;

// A decoded scalar/product spans at most four limbs, including its bit shift.
#[derive(Clone, Copy)]
pub(super) struct ExactTerm {
    words: [u64; 4],
    start: usize,
    negative: bool,
}

impl ExactTerm {
    #[inline]
    pub(super) fn new(words: [u64; 3], exponent: i32, negative: bool) -> Self {
        let offset = (exponent - ACCUMULATOR_MIN_EXPONENT) as usize;
        let shift = offset % 64;
        let mut shifted = [0; 4];
        for (index, word) in words.into_iter().enumerate() {
            shifted[index] |= word << shift;
            if shift != 0 {
                shifted[index + 1] |= word >> (64 - shift);
            }
        }
        Self {
            words: shifted,
            start: offset / 64,
            negative,
        }
    }

    #[cfg(all(
        not(feature = "std"),
        not(any(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "fma"
            ),
            all(target_arch = "aarch64", target_feature = "neon")
        ))
    ))]
    fn scalar(value: FiniteParts) -> Self {
        Self::new(
            [value.significand(), 0, 0],
            i32::from(value.exponent()),
            value.negative(),
        )
    }

    pub(super) fn pair(a: FiniteParts, b: FiniteParts, subtract: bool) -> Self {
        let (product, exponent, negative) = a.product(b);
        Self::new(
            [product as u64, (product >> 64) as u64, 0],
            exponent,
            negative ^ subtract,
        )
    }

    pub(super) fn triple(a: FiniteParts, b: FiniteParts, c: FiniteParts, subtract: bool) -> Self {
        let (pair, exponent, negative) = a.product(b);
        let low = u128::from(pair as u64) * u128::from(c.significand());
        let high = (pair >> 64) * u128::from(c.significand()) + (low >> 64);
        Self::new(
            [low as u64, high as u64, (high >> 64) as u64],
            exponent + i32::from(c.exponent()),
            negative ^ c.negative() ^ subtract,
        )
    }

    #[inline]
    fn end(self) -> usize {
        self.words
            .iter()
            .rposition(|word| *word != 0)
            .map_or(0, |last| self.start + last + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LimbBin {
    pub(super) words: [u64; ACCUMULATOR_LIMBS],
    pub(super) lo: usize,
    pub(super) hi: usize,
}

impl LimbBin {
    const fn new() -> Self {
        Self {
            words: [0; ACCUMULATOR_LIMBS],
            lo: ACCUMULATOR_LIMBS,
            hi: 0,
        }
    }

    // Ordinary finite products leave more than a full limb of carry headroom.
    // Only artificially enlarged states require the full carry preflight.
    fn can_add(&self, terms: &[ExactTerm], negative: bool) -> bool {
        let end = terms
            .iter()
            .filter(|term| term.negative == negative)
            .map(|term| term.end())
            .max()
            .unwrap_or(0);
        if end == 0 {
            return true;
        }
        if end > ACCUMULATOR_LIMBS {
            return false;
        }
        if self.hi < ACCUMULATOR_LIMBS && end < ACCUMULATOR_LIMBS {
            return true;
        }
        let start = terms
            .iter()
            .filter(|term| term.negative == negative && term.end() != 0)
            .map(|term| term.start)
            .min()
            .unwrap_or(0);
        let mut carry = 0_u128;
        for index in start..ACCUMULATOR_LIMBS {
            let sum = terms.iter().filter(|term| term.negative == negative).fold(
                u128::from(self.words[index]) + carry,
                |sum, term| {
                    sum + index
                        .checked_sub(term.start)
                        .and_then(|offset| term.words.get(offset))
                        .map_or(0, |word| u128::from(*word))
                },
            );
            carry = sum >> 64;
        }
        carry == 0
    }

    #[inline]
    fn add(&mut self, term: ExactTerm) {
        let end = term.end();
        if end == 0 {
            return;
        }
        self.lo = self.lo.min(term.start);
        let mut index = term.start;
        let mut carry = 0_u128;
        while index < end || carry != 0 {
            let word = term.words.get(index - term.start).copied().unwrap_or(0);
            let sum = u128::from(self.words[index]) + u128::from(word) + carry;
            self.words[index] = sum as u64;
            carry = sum >> 64;
            index += 1;
        }
        self.hi = self.hi.max(index);
    }

    fn can_merge(&self, other: &Self) -> bool {
        if self.hi.max(other.hi) < ACCUMULATOR_LIMBS {
            return true;
        }
        let mut carry = 0_u128;
        for index in self.lo.min(other.lo)..self.hi.max(other.hi) {
            carry = (u128::from(self.words[index]) + u128::from(other.words[index]) + carry) >> 64;
        }
        carry == 0
    }

    fn merge(&mut self, other: &Self) {
        if other.hi == 0 {
            return;
        }
        let mut carry = 0_u128;
        let mut index = other.lo;
        while index < other.hi || carry != 0 {
            let sum = u128::from(self.words[index]) + u128::from(other.words[index]) + carry;
            self.words[index] = sum as u64;
            carry = sum >> 64;
            index += 1;
        }
        self.lo = self.lo.min(other.lo);
        self.hi = self.hi.max(index);
    }
}

fn compare_words(left: &[u64], right: &[u64]) -> Ordering {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .find_map(|(left, right)| (left != right).then(|| left.cmp(right)))
        .unwrap_or(Ordering::Equal)
}

fn subtract_words(
    greater: &[u64; ACCUMULATOR_LIMBS],
    lesser: &[u64; ACCUMULATOR_LIMBS],
    lo: usize,
    hi: usize,
) -> [u64; ACCUMULATOR_LIMBS] {
    // signed_magnitude compares the bins first, so the final borrow is zero.
    let mut output = [0_u64; ACCUMULATOR_LIMBS];
    let mut borrow = false;
    for (output, (&greater, &lesser)) in output[lo..hi]
        .iter_mut()
        .zip(greater[lo..hi].iter().zip(&lesser[lo..hi]))
    {
        let (difference, first_borrow) = greater.overflowing_sub(lesser);
        let (difference, second_borrow) = difference.overflowing_sub(u64::from(borrow));
        *output = difference;
        borrow = first_borrow || second_borrow;
    }
    output
}

// A single finite product plus addend fits the accumulator by construction.
#[cfg(all(
    not(feature = "std"),
    not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
pub(crate) fn finite_fused_mul_add<S: RealScalar>(
    left: FiniteParts,
    right: FiniteParts,
    addend: FiniteParts,
) -> S {
    let mut state = RangeAccumulator::<S>::new();
    state.commit_terms(&[
        ExactTerm::pair(left, right, false),
        ExactTerm::scalar(addend),
    ]);
    let (sign, magnitude) = state.signed_magnitude();
    rounded_words_at_exponent::<S>(
        &magnitude,
        sign == Ordering::Less,
        ACCUMULATOR_MIN_EXPONENT,
        false,
    )
}

/// Exact accumulator for sums and sums of products of finite real scalars.
///
/// Every addend and product is added without rounding into a fixed-size
/// superaccumulator (two 100-limb integer bins), covering the whole exponent
/// range of the scalar type. [`Self::finish`] rounds the exact value once, to
/// nearest with ties to even, so the result is correctly rounded. Exact
/// cancellation gives `+0`; a nonzero value that underflows keeps its sign.
///
/// An update costs `O(1)`, and [`Self::merge`] is exact and `O(1)` in the input
/// length, so partial sums from parallel partitions can be combined in any fixed
/// order without changing the result. A failed update or merge leaves the state
/// unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_core::RangeAccumulator;
///
/// // The exact sum is 1; naive summation returns 0.
/// let mut sum = RangeAccumulator::<f64>::new();
/// sum.add(1.0e100)?;
/// sum.add(1.0)?;
/// sum.add(-1.0e100)?;
/// assert_eq!(sum.finish()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct RangeAccumulator<S: RealScalar> {
    pub(super) positive: LimbBin,
    negative: LimbBin,
    marker: PhantomData<S>,
}

impl<S: RealScalar> Default for RangeAccumulator<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> RangeAccumulator<S> {
    /// Resets the accumulator to zero.
    pub fn clear(&mut self) {
        for bin in [&mut self.positive, &mut self.negative] {
            if bin.hi != 0 {
                bin.words[bin.lo..bin.hi].fill(0);
            }
            bin.lo = ACCUMULATOR_LIMBS;
            bin.hi = 0;
        }
    }

    /// Returns an empty accumulator, whose value is `+0`.
    pub const fn new() -> Self {
        Self {
            positive: LimbBin::new(),
            negative: LimbBin::new(),
            marker: PhantomData,
        }
    }

    fn compare_bins(&self) -> Ordering {
        let lo = self.positive.lo.min(self.negative.lo);
        let hi = self.positive.hi.max(self.negative.hi);
        if hi == 0 {
            return Ordering::Equal;
        }
        compare_words(&self.positive.words[lo..hi], &self.negative.words[lo..hi])
    }

    pub(super) fn check_terms(&self, terms: &[ExactTerm]) -> Result<(), SciOxideError> {
        if self.positive.can_add(terms, false) && self.negative.can_add(terms, true) {
            Ok(())
        } else {
            Err(overflow("exact accumulator capacity was exceeded"))
        }
    }

    pub(super) fn commit_terms(&mut self, terms: &[ExactTerm]) {
        for &term in terms {
            if term.negative {
                self.negative.add(term);
            } else {
                self.positive.add(term);
            }
        }
    }

    #[inline]
    fn add_parts(
        &mut self,
        significand: u128,
        exponent: i32,
        negative: bool,
    ) -> Result<(), SciOxideError> {
        let term = ExactTerm::new(
            [significand as u64, (significand >> 64) as u64, 0],
            exponent,
            negative,
        );
        let destination = if negative {
            &mut self.negative
        } else {
            &mut self.positive
        };
        let end = term.end();
        if (destination.hi == ACCUMULATOR_LIMBS || end >= ACCUMULATOR_LIMBS)
            && !destination.can_add(&[term], negative)
        {
            return Err(overflow("exact accumulator capacity was exceeded"));
        }
        destination.add(term);
        Ok(())
    }

    /// Adds a finite scalar exactly.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) with
    ///   [`NonFiniteOrigin::Input`] if `value` is NaN or infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the accumulator capacity is exceeded.
    #[inline]
    pub fn add(&mut self, value: S) -> Result<(), SciOxideError> {
        let value = finite_parts(value, "sum input must be finite")?;
        self.add_finite(value)
    }

    fn add_finite(&mut self, value: FiniteParts) -> Result<(), SciOxideError> {
        if value.significand() != 0 {
            self.add_parts(
                value.significand().into(),
                i32::from(value.exponent()),
                value.negative(),
            )?;
        }
        Ok(())
    }

    pub(super) fn add_integer_product(
        &mut self,
        value: S,
        weight: usize,
    ) -> Result<(), SciOxideError> {
        let value = finite_parts(value, "weighted sum input must be finite")?;
        self.add_parts(
            u128::from(value.significand()) * weight as u128,
            i32::from(value.exponent()),
            value.negative(),
        )
    }

    /// Adds `weight · Σ values` exactly, without rounding the sum or the products.
    ///
    /// On error the accumulator is unchanged. Whether the final value fits the
    /// scalar type is checked by [`Self::finish`], not here.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `weight` or a value
    ///   is NaN or infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the accumulator capacity is exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::RangeAccumulator;
    /// let mut sum = RangeAccumulator::<f64>::new();
    /// sum.add_weighted_values(0.5, &[1e100, 2.0, -1e100])?;
    /// assert_eq!(sum.finish()?, 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn add_weighted_values(&mut self, weight: S, values: &[S]) -> Result<(), SciOxideError> {
        let weight = finite_parts(weight, "sum weight must be finite")?;
        // A short bounded block produces one exact term before any state
        // mutation. Admit and commit that term directly, avoiding a copy of
        // both complete limb bins for every caller-owned batch chunk.
        if values.len() <= 64
            && let Some(term) = Self::weighted_term(weight, values)?
        {
            self.check_terms(&[term])?;
            self.commit_terms(&[term]);
            return Ok(());
        }
        let mut candidate = self.clone();
        for chunk in values.chunks(64) {
            if let Some(term) = Self::weighted_term(weight, chunk)? {
                candidate.check_terms(&[term])?;
                candidate.commit_terms(&[term]);
            } else {
                for &value in chunk {
                    candidate.update_finite_product(
                        finite_parts(value, "sum input must be finite")?,
                        weight,
                        false,
                    )?;
                }
            }
        }
        *self = candidate;
        Ok(())
    }

    /// Decode a bounded block without touching the accumulator. The same
    /// exponent/carry proof applies to direct and all-or-nothing multi-block use.
    fn weighted_term(
        weight: FiniteParts,
        values: &[S],
    ) -> Result<Option<ExactTerm>, SciOxideError> {
        debug_assert!(values.len() <= 64);
        let mut minimum = i32::MAX;
        let mut maximum = i32::MIN;
        for &value in values {
            let value = finite_parts(value, "sum input must be finite")?;
            if value.significand() != 0 {
                let exponent = i32::from(value.exponent());
                minimum = minimum.min(exponent);
                maximum = maximum.max(exponent + (64 - value.significand().leading_zeros()) as i32);
            }
        }
        if minimum == i32::MAX {
            return Ok(Some(ExactTerm::new([0; 3], 0, false)));
        }
        let carry = values.len().next_power_of_two().ilog2() as i32;
        if maximum - minimum + carry > 127 {
            return Ok(None);
        }
        {
            let integer =
                values
                    .iter()
                    .try_fold(0_i128, |sum, &value| -> Result<i128, SciOxideError> {
                        let value = finite_parts(value, "sum input must be finite")?;
                        if value.significand() == 0 {
                            return Ok(sum);
                        }
                        // The block bound leaves the sign bit free even when every
                        // term has the maximum magnitude and the same sign.
                        let term = i128::from(value.significand())
                            << (i32::from(value.exponent()) - minimum);
                        // Apply the sign without a data-dependent branch. The
                        // span/carry bound proves both the signed term and the
                        // complete block sum fit i128.
                        let sign = -i128::from(value.negative());
                        Ok(sum + ((term ^ sign) - sign))
                    })?;
            let magnitude = integer.unsigned_abs();
            if weight.significand().is_power_of_two() {
                // Dyadic weights need only an exponent adjustment; avoid
                // multiplying by the decoder's implicit significand bit.
                return Ok(Some(ExactTerm::new(
                    [magnitude as u64, (magnitude >> 64) as u64, 0],
                    minimum
                        + i32::from(weight.exponent())
                        + weight.significand().trailing_zeros() as i32,
                    (integer < 0) ^ weight.negative(),
                )));
            }
            let low = u128::from(magnitude as u64) * u128::from(weight.significand());
            let high = (magnitude >> 64) * u128::from(weight.significand()) + (low >> 64);
            let term = ExactTerm::new(
                [low as u64, high as u64, (high >> 64) as u64],
                minimum + i32::from(weight.exponent()),
                (integer < 0) ^ weight.negative(),
            );
            Ok(Some(term))
        }
    }

    /// Adds the exact product `left · right`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) with
    ///   [`NonFiniteOrigin::Input`] if either factor is NaN or infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the accumulator capacity is exceeded.
    #[inline]
    pub fn add_product(&mut self, left: S, right: S) -> Result<(), SciOxideError> {
        self.update_product(left, right, false)
    }

    #[inline]
    pub(super) fn update_product(
        &mut self,
        left: S,
        right: S,
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        let left = finite_parts(left, "left product factor must be finite")?;
        let right = finite_parts(right, "right product factor must be finite")?;
        self.update_finite_product(left, right, subtract)
    }

    #[inline]
    pub(super) fn update_finite_product(
        &mut self,
        left: FiniteParts,
        right: FiniteParts,
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        let (product, exponent, negative) = left.product(right);
        if product != 0 {
            self.add_parts(product, exponent, negative ^ subtract)?;
        }
        Ok(())
    }

    /// Reduce at most 64 decoded products in a signed 192-bit integer when
    /// their exponent span leaves carry headroom. Wider spans use scalar
    /// exact updates; no partial block is rounded on either path.
    pub(super) fn add_product_block(&mut self, pairs: &[(S, S)]) -> Result<(), SciOxideError> {
        debug_assert!(pairs.len() <= 64);
        if let Some(&(weight, _)) = pairs.first()
            && pairs.iter().all(|&(a, _)| a == weight)
        {
            let mut values = [S::ZERO; 64];
            for (value, &(_, b)) in values.iter_mut().zip(pairs) {
                *value = b;
            }
            return self.add_weighted_values(weight, &values[..pairs.len()]);
        }
        let mut products = [(0_u128, 0_i32, false); 64];
        let mut minimum = i32::MAX;
        let mut maximum = i32::MIN;
        for (slot, &(a, b)) in products.iter_mut().zip(pairs) {
            let a = finite_parts(a, "left product factor must be finite")?;
            let b = finite_parts(b, "right product factor must be finite")?;
            let (product, exponent, negative) = a.product(b);
            if product == 0 {
                continue;
            }
            minimum = minimum.min(exponent);
            maximum = maximum.max(exponent + (128 - product.leading_zeros()) as i32);
            *slot = (product, exponent, negative);
        }
        if minimum == i32::MAX {
            return Ok(());
        }
        let carry = pairs.len().next_power_of_two().ilog2() as i32;
        let span = maximum - minimum + carry;
        if span <= 127 {
            let integer = products[..pairs.len()].iter().fold(
                0_i128,
                |sum, &(product, exponent, negative)| {
                    if product == 0 {
                        return sum;
                    }
                    let term = (product << (exponent - minimum)) as i128;
                    sum + if negative { -term } else { term }
                },
            );
            let magnitude = integer.unsigned_abs();
            if magnitude != 0 {
                self.add_parts(magnitude, minimum, integer < 0)?;
            }
        } else if span <= 191 {
            let mut low = 0_u128;
            let mut high = 0_i64;
            for &(product, exponent, negative) in &products[..pairs.len()] {
                if product == 0 {
                    continue;
                }
                let shift = (exponent - minimum) as u32;
                let (lo, hi) = if shift >= 128 {
                    (0, (product as u64) << (shift - 128))
                } else if shift == 0 {
                    (product, 0)
                } else {
                    (product << shift, (product >> (128 - shift)) as u64)
                };
                // The span bound reserves the sign bit even if all 64 terms
                // have the same sign. The signed upper limb also absorbs the
                // carry/borrow of the unsigned lower two limbs exactly.
                let signed_lo = if negative { lo.wrapping_neg() } else { lo };
                let signed_hi = if negative {
                    -(hi as i64) - i64::from(lo != 0)
                } else {
                    hi as i64
                };
                let (next, carry) = low.overflowing_add(signed_lo);
                low = next;
                high += signed_hi + i64::from(carry);
            }
            let negative = high < 0;
            let (low, high) = if negative {
                let magnitude = low.wrapping_neg();
                (magnitude, high.unsigned_abs() - u64::from(magnitude != 0))
            } else {
                (low, high as u64)
            };
            let term = ExactTerm::new([low as u64, (low >> 64) as u64, high], minimum, negative);
            self.check_terms(&[term])?;
            self.commit_terms(&[term]);
        } else {
            for &(product, exponent, negative) in &products[..pairs.len()] {
                if product != 0 {
                    self.add_parts(product, exponent, negative)?;
                }
            }
        }
        Ok(())
    }

    /// Adds `left · right · 2^exponent` exactly, without forming `2^exponent`.
    ///
    /// This recombines exponent-split products, for example in matrix products.
    /// On error the accumulator is unchanged.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either factor is NaN
    ///   or infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the term is outside the accumulator's
    ///   exponent range.
    pub fn add_scaled_product(
        &mut self,
        left: S,
        right: S,
        exponent: i32,
    ) -> Result<(), SciOxideError> {
        let left = finite_parts(left, "scaled product factor must be finite")?;
        let right = finite_parts(right, "scaled product factor must be finite")?;
        let (product, decoded_exponent, negative) = left.product(right);
        if product == 0 {
            return Ok(());
        }
        let zeros = product.trailing_zeros();
        let product = product >> zeros;
        let exponent = exponent
            .checked_add(decoded_exponent + zeros as i32)
            .ok_or_else(|| overflow("scaled product exponent exceeds accumulator capacity"))?;
        let top = i64::from(exponent) + i64::from(128 - product.leading_zeros());
        if exponent < ACCUMULATOR_MIN_EXPONENT
            || top > i64::from(ACCUMULATOR_MIN_EXPONENT) + (64 * ACCUMULATOR_LIMBS) as i64
        {
            return Err(overflow(
                "scaled product exceeds accumulator exponent capacity",
            ));
        }
        self.add_parts(product, exponent, negative)
    }

    /// Adds each triple product in a stream exactly.
    ///
    /// Unlike the single-term updates, this is not all-or-nothing: terms added
    /// before a failing term stay in the accumulator.
    ///
    /// # Errors
    ///
    /// Returns the first error of [`Self::add_triple_product`].
    pub fn add_triple_products(
        &mut self,
        scale: S,
        terms: impl IntoIterator<Item = (S, S)>,
    ) -> Result<(), SciOxideError> {
        let scale = finite_parts(scale, "product scale must be finite")?;
        let (weight_detail, value_detail) = if scale.is_one() {
            (
                "left product factor must be finite",
                "right product factor must be finite",
            )
        } else {
            (
                "second product factor must be finite",
                "third product factor must be finite",
            )
        };
        terms.into_iter().try_for_each(|(weight, value)| {
            let weight = finite_parts(weight, weight_detail)?;
            let value = finite_parts(value, value_detail)?;
            self.update_finite_triple_product(scale, weight, value, false)
        })
    }

    /// Subtracts the exact product `left · right`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::add_product`].
    #[inline]
    pub fn subtract_product(&mut self, left: S, right: S) -> Result<(), SciOxideError> {
        self.update_product(left, right, true)
    }

    pub(super) fn update_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        let first = finite_parts(first, "first product factor must be finite")?;
        let second = finite_parts(second, "second product factor must be finite")?;
        let third = finite_parts(third, "third product factor must be finite")?;
        self.update_finite_triple_product(first, second, third, subtract)
    }

    fn update_finite_triple_product(
        &mut self,
        first: FiniteParts,
        second: FiniteParts,
        third: FiniteParts,
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        if first.is_one() {
            return self.update_finite_product(second, third, subtract);
        }
        let term = ExactTerm::triple(first, second, third, subtract);
        self.check_terms(&[term])?;
        self.commit_terms(&[term]);
        Ok(())
    }

    /// Adds the exact product `first · second · third`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the accumulator capacity is exceeded.
    pub fn add_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
    ) -> Result<(), SciOxideError> {
        self.update_triple_product(first, second, third, false)
    }

    /// Subtracts the exact product `first · second · third`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::add_triple_product`].
    pub fn subtract_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
    ) -> Result<(), SciOxideError> {
        self.update_triple_product(first, second, third, true)
    }

    /// Adds the exact value of `other` to this accumulator.
    ///
    /// On error this accumulator is unchanged.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`NumericalIssue::Overflow`]) if repeated merges exceed the accumulator
    /// capacity. A reduction over any slice that fits in memory does not.
    pub fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        self.check_merge(other)?;
        self.commit_merge(other);
        Ok(())
    }

    pub(super) fn commit_merge(&mut self, other: &Self) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }

    pub(super) fn check_merge(&self, other: &Self) -> Result<(), SciOxideError> {
        if self.positive.can_merge(&other.positive) && self.negative.can_merge(&other.negative) {
            Ok(())
        } else {
            Err(overflow(
                "exact accumulator capacity was exceeded while merging",
            ))
        }
    }

    pub(super) fn signed_magnitude(&self) -> (Ordering, [u64; ACCUMULATOR_LIMBS]) {
        match self.compare_bins() {
            Ordering::Greater => (
                Ordering::Greater,
                subtract_words(
                    &self.positive.words,
                    &self.negative.words,
                    self.positive.lo.min(self.negative.lo),
                    self.positive.hi.max(self.negative.hi),
                ),
            ),
            Ordering::Less => (
                Ordering::Less,
                subtract_words(
                    &self.negative.words,
                    &self.positive.words,
                    self.positive.lo.min(self.negative.lo),
                    self.positive.hi.max(self.negative.hi),
                ),
            ),
            Ordering::Equal => (Ordering::Equal, [0; ACCUMULATOR_LIMBS]),
        }
    }

    /// Returns the exact sign of the accumulated value.
    pub fn sign(&self) -> Ordering {
        self.compare_bins()
    }

    /// Rounds the exact sum to a [`Scaled`] value with an extended exponent.
    ///
    /// The type parameter `R` selects the significand precision. An exact zero
    /// returns [`Scaled::ZERO`].
    ///
    /// # Errors
    ///
    /// Returns an error if the significand or exponent cannot be represented
    /// in [`Scaled`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::RangeAccumulator;
    ///
    /// let mut sum = RangeAccumulator::<f64>::new();
    /// sum.add(2.0)?;
    /// assert_eq!(sum.finish_scaled::<f64>()?.round()?, 2.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn finish_scaled<R: RealScalar>(&self) -> Result<Scaled<R>, SciOxideError> {
        let (sign, words) = self.signed_magnitude();
        let Some(top) = highest_bit(&words) else {
            return Ok(Scaled::ZERO);
        };
        let shift = (top + 1).saturating_sub(R::MANTISSA_DIGITS as usize);
        let integer = rounded_integer(&words, shift, false);
        let value = R::try_from_f64_exact(integer as f64)?;
        Scaled::from_parts(
            if sign == Ordering::Less {
                -value
            } else {
                value
            },
            i64::from(ACCUMULATOR_MIN_EXPONENT) + shift as i64,
        )
    }

    // Extract four nonoverlapping components before scalar rounding. Removing
    // each leading chunk and searching again retains widely separated tails,
    // e.g. 1 + 2^-1000. The remaining absolute tail is < 2^-211 in this frame.
    pub(crate) fn expansion(&self) -> ([f64; 4], i32) {
        let (sign, mut words) = self.signed_magnitude();
        let Some(top) = highest_bit(&words) else {
            return ([0.0; 4], 0);
        };
        let mut parts = [0.0; 4];
        for part in &mut parts {
            let Some(bit) = highest_bit(&words) else {
                break;
            };
            let width = (bit + 1).min(53);
            let shift = bit + 1 - width;
            let mask = (1_u64 << width) - 1;
            let integer = shifted_low_u64(&words, shift) & mask;
            let value = RealScalar::scalbn(integer as f64, shift as i32 - top as i32);
            *part = if sign == Ordering::Less {
                -value
            } else {
                value
            };
            let offset = shift % 64;
            words[shift / 64] &= !(mask << offset);
            if offset + width > 64 {
                words[shift / 64 + 1] &= !(mask >> (64 - offset));
            }
        }
        (parts, ACCUMULATOR_MIN_EXPONENT + top as i32)
    }

    // Destination-grid projection for private extended reconstructions. In
    // particular, do not round through binary64 at a binary32 midpoint.
    pub(crate) fn finish_to<R: RealScalar>(&self) -> Result<R, SciOxideError> {
        let (sign, words) = self.signed_magnitude();
        let value =
            rounded_words_at_exponent::<R>(&words, sign.is_lt(), ACCUMULATOR_MIN_EXPONENT, false);
        checked_final_rounding(value, "logarithmic result exceeds scalar range")
    }

    /// Returns the exact sum rounded once to nearest, ties to even.
    ///
    /// Exact cancellation returns `+0`.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`NumericalIssue::Overflow`]) if the rounded value is outside the finite
    /// range of `S`.
    pub fn finish(&self) -> Result<S, SciOxideError> {
        if self.positive.hi == 0 || self.negative.hi == 0 {
            let negative = self.negative.hi != 0;
            let bin = if negative {
                &self.negative
            } else {
                &self.positive
            };
            if bin.hi == 0 {
                return Ok(S::ZERO);
            }
            let value = rounded_words_at_exponent::<S>(
                &bin.words[bin.lo..bin.hi],
                negative,
                ACCUMULATOR_MIN_EXPONENT + (bin.lo * 64) as i32,
                false,
            );
            return if value.is_finite() {
                Ok(value)
            } else {
                Err(overflow(
                    "finite mathematical result is outside the scalar range",
                ))
            };
        }
        let (sign, magnitude) = self.signed_magnitude();
        if sign == Ordering::Equal {
            return Ok(S::ZERO);
        }
        let lo = self.positive.lo.min(self.negative.lo);
        let hi = self.positive.hi.max(self.negative.hi);
        let value = rounded_words_at_exponent::<S>(
            &magnitude[lo..hi],
            sign == Ordering::Less,
            ACCUMULATOR_MIN_EXPONENT + (lo * 64) as i32,
            false,
        );
        if value.is_finite() {
            Ok(value)
        } else {
            Err(overflow(
                "finite mathematical result is outside the scalar range",
            ))
        }
    }

    /// Returns the exact sum divided by `denominator`, rounded once to nearest,
    /// ties to even.
    ///
    /// Neither the sum nor its terms need to be representable in `S`; only the
    /// quotient must be. This serves normalized recurrences whose numerator is
    /// outside the range of `S` although the normalized value is finite.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `denominator` is
    ///   NaN or infinite.
    /// - [`DomainError`](crate::SciOxideErrorKind::DomainError) if `denominator`
    ///   is zero.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`NumericalIssue::Overflow`]) if the quotient is outside the finite
    ///   range of `S`.
    pub fn finish_quotient(&self, denominator: S) -> Result<S, SciOxideError> {
        let denominator_parts = finite_parts(denominator, "denominator must be finite")?;
        if denominator == S::ZERO {
            return Err(SciOxideError::domain_error(
                crate::argument!("denominator"),
                DomainReason::Undefined,
                "denominator must be nonzero",
            ));
        }
        self.divide_integer(
            denominator_parts.significand(),
            i32::from(denominator_parts.exponent()),
            denominator.is_sign_negative(),
        )
    }

    // The caller validates the nonzero denominator. Supported pointer widths
    // are at most 64 bits, so remainder << 64 fits the u128 division window.
    pub(super) fn finish_integer_quotient(&self, denominator: usize) -> Result<S, SciOxideError> {
        self.divide_integer(denominator as u64, 0, false)
    }

    fn divide_integer(
        &self,
        divisor: u64,
        exponent: i32,
        negative: bool,
    ) -> Result<S, SciOxideError> {
        let (sign, mut quotient) = self.signed_magnitude();
        if sign == Ordering::Equal {
            return Ok(S::ZERO);
        }
        let divisor = u128::from(divisor);
        let mut remainder = 0_u128;
        // Base-2^64 long division leaves an exact integer quotient and one
        // remainder. The remainder is below the quotient's lowest bit, so it
        // contributes only a sticky bit when rounding the destination scalar.
        // Even after division by the smallest scalar, this lowest bit lies
        // below the destination subnormal range (the accumulator covers triple
        // products). A zero integer quotient therefore necessarily underflows.
        // Restrict division to the live numerator window plus two low guard
        // limbs. The divisor has at most 64 bits: 128 guard bits therefore
        // extend below every destination rounding bit even if division shifts
        // the leading bit down by 64. The remaining tail is only a sticky bit.
        let lo = quotient
            .iter()
            .position(|word| *word != 0)
            .unwrap_or(0)
            .saturating_sub(2);
        let hi = quotient.iter().rposition(|word| *word != 0).unwrap_or(0);
        for word in quotient[lo..=hi].iter_mut().rev() {
            let numerator = (remainder << 64) | u128::from(*word);
            *word = (numerator / divisor) as u64;
            remainder = numerator % divisor;
        }
        let candidate = rounded_words_at_exponent::<S>(
            &quotient,
            (sign == Ordering::Less) ^ negative,
            ACCUMULATOR_MIN_EXPONENT - exponent,
            remainder != 0,
        );
        if !candidate.is_finite() {
            return Err(overflow(
                "final accumulated quotient is outside the scalar range",
            ));
        }
        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn weighted_blocks_match_individual_exact_updates() {
        macro_rules! check {
            ($s:ty) => {{
                for length in [0, 1, 3, 63, 64, 65, 127, 129] {
                    for spread in [0, 1, 32, 67, 68, 69, 90] {
                        for weight in [0.0 as $s, -0.0, 0.5, -1.25, <$s>::MIN_POSITIVE, <$s>::MAX] {
                            let values: alloc::vec::Vec<$s> = (0..length)
                                .map(|i| {
                                    let value = (1.0 + (i % 17) as $s * <$s>::EPSILON)
                                        * (2.0 as $s).powi(if i % 3 == 0 { spread } else { 0 });
                                    if i % 2 == 0 { value } else { -value }
                                })
                                .collect();
                            let mut expected = super::RangeAccumulator::<$s>::new();
                            let mut actual = super::RangeAccumulator::<$s>::new();
                            expected.add(0.25).unwrap();
                            actual.add(0.25).unwrap();
                            for &value in &values {
                                expected.add_product(value, weight).unwrap();
                            }
                            actual.add_weighted_values(weight, &values).unwrap();
                            match (actual.finish(), expected.finish()) {
                                (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                                (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind()),
                                (a, b) => panic!("weighted block mismatch: {a:?}, {b:?}"),
                            }
                        }
                    }
                }
                let tiny = <$s>::from_bits(1);
                for values in [
                    [tiny, tiny, -tiny],
                    [<$s>::MAX, -<$s>::MAX, tiny],
                    [-0.0, 0.0, -0.0],
                ] {
                    let mut expected = super::RangeAccumulator::<$s>::new();
                    let mut actual = super::RangeAccumulator::<$s>::new();
                    for value in values {
                        expected.add_product(value, 0.5).unwrap();
                    }
                    actual.add_weighted_values(0.5, &values).unwrap();
                    assert_eq!(
                        actual.finish().unwrap().to_bits(),
                        expected.finish().unwrap().to_bits()
                    );
                }
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn weighted_blocks_reject_late_nonfinite_values_transactionally() {
        let mut state = super::RangeAccumulator::<f64>::new();
        state.add(3.0).unwrap();
        let mut values = [1.0; 129];
        values[128] = f64::NAN;
        assert!(state.add_weighted_values(1.0, &values).is_err());
        assert_eq!(state.finish().unwrap(), 3.0);
        assert!(state.add_weighted_values(f64::INFINITY, &[1.0]).is_err());
        assert_eq!(state.finish().unwrap(), 3.0);
    }

    use super::super::ProductSum;
    use super::*;
    use crate::Complex;

    #[test]
    fn live_windows_preserve_carries_cancellation_and_subnormal_rounding() {
        let mut sum = RangeAccumulator::<f64>::new();
        sum.add_parts(u128::from(u64::MAX), -22, false).unwrap();
        sum.add_parts(1, -22, false).unwrap();
        assert_eq!(sum.positive.hi, 52);
        assert_eq!(sum.finish().unwrap(), 2.0_f64.powi(42));
        sum.add(-2.0_f64.powi(42)).unwrap();
        assert_eq!(sum.sign(), Ordering::Equal);
        sum.add_triple_product(f64::from_bits(1), 0.5, 1.0).unwrap();
        assert_eq!(sum.sign(), Ordering::Greater);
        assert_eq!(sum.finish().unwrap().to_bits(), 0);
        sum.add_triple_product(f64::from_bits(1), 0.5, 1.0).unwrap();
        assert_eq!(sum.finish().unwrap().to_bits(), 1);
        sum.clear();
        assert_eq!(sum, RangeAccumulator::new());
        sum.add(-f64::MAX).unwrap();
        assert_eq!(sum.finish().unwrap(), -f64::MAX);
    }

    #[test]
    fn exponent_tagged_partials_do_not_materialize_overflow() {
        let mut sum = RangeAccumulator::<f64>::new();
        sum.add_scaled_product(1.0, 0.5, 1025).unwrap();
        sum.add_scaled_product(-1.0, 1.0, 1023).unwrap();
        assert_eq!(sum.finish().unwrap(), 2.0_f64.powi(1023));
        let previous = sum.clone();
        assert!(sum.add_scaled_product(1.0, 1.0, i32::MAX).is_err());
        assert_eq!(sum, previous);
        assert!(sum.add_scaled_product(1.0, 1.0, i32::MIN).is_err());
        assert_eq!(sum, previous);
    }

    #[test]
    fn triple_products_delay_rounding_until_the_affine_result() {
        let mut real = RangeAccumulator::new();
        real.add_triple_product(0.25_f64, f64::MAX, 2.0).unwrap();
        assert_eq!(real.finish().unwrap(), f64::MAX / 2.0);
        let mut extremes = RangeAccumulator::new();
        extremes
            .add_triple_product(f64::MAX, f64::MAX, f64::MAX)
            .unwrap();
        extremes
            .subtract_triple_product(f64::MAX, f64::MAX, f64::MAX)
            .unwrap();
        extremes
            .add_triple_product(f64::from_bits(1), f64::from_bits(1), f64::from_bits(1))
            .unwrap();
        assert_eq!(extremes.finish().unwrap(), 0.0);

        let first = Complex::new(2.0_f64, -3.0);
        let second = Complex::new(-5.0, 7.0);
        let third = Complex::new(11.0, 13.0);
        let mut complex = ProductSum::new();
        complex.add_triple_product(first, second, third).unwrap();
        assert_eq!(complex.finish().unwrap(), first * second * third);

        complex
            .subtract_triple_product(first, second, third)
            .unwrap();
        assert_eq!(complex.finish().unwrap(), Complex::new(0.0, 0.0));
    }

    #[test]
    fn accumulated_quotients_round_once_without_projecting_the_numerator() {
        macro_rules! check {
            ($scalar:ty) => {{
                let tiny = <$scalar>::from_bits(1);
                let mut sum = RangeAccumulator::<$scalar>::new();
                sum.add_product(<$scalar>::MAX, 2.0).unwrap();
                assert!(sum.finish().is_err());
                assert_eq!(sum.finish_quotient(<$scalar>::MAX).unwrap(), 2.0);
                assert_eq!(sum.finish_quotient(-<$scalar>::MAX).unwrap(), -2.0);
                assert!(sum.finish_quotient(0.0).is_err());
                assert!(sum.finish_quotient(<$scalar>::INFINITY).is_err());
                assert!(sum.finish_quotient(0.5).is_err());

                let mut sum = RangeAccumulator::<$scalar>::new();
                sum.add_product(tiny, tiny).unwrap();
                assert_eq!(sum.finish().unwrap(), 0.0);
                assert_eq!(sum.finish_quotient(tiny).unwrap(), tiny);

                // The exact quotient is halfway between 1 and its successor.
                // Adding one minimum subnormal to the numerator changes only
                // the long division's discarded tail, but must round upward.
                let mut sum = RangeAccumulator::<$scalar>::new();
                sum.add(3.0).unwrap();
                sum.add_product(3.0, <$scalar>::EPSILON * 0.5).unwrap();
                assert_eq!(sum.finish_quotient(3.0).unwrap(), 1.0);
                sum.add(tiny).unwrap();
                assert_eq!(sum.finish_quotient(3.0).unwrap(), 1.0 + <$scalar>::EPSILON);

                let mut sum = RangeAccumulator::<$scalar>::new();
                sum.add(tiny).unwrap();
                assert_eq!(sum.finish_quotient(2.0).unwrap(), 0.0);
                assert!(sum.finish_quotient(-2.0).unwrap().is_sign_negative());
                sum.add_product(tiny, 2.0).unwrap();
                assert_eq!(sum.finish_quotient(2.0).unwrap(), tiny * 2.0);
                sum.subtract_product(tiny, tiny).unwrap();
                assert_eq!(sum.finish_quotient(2.0).unwrap(), tiny);
                let mut near_half = RangeAccumulator::<$scalar>::new();
                near_half.add(tiny).unwrap();
                near_half.add_product(tiny, tiny).unwrap();
                assert_eq!(near_half.finish_quotient(2.0).unwrap(), tiny);
            }};
        }
        check!(f32);
        check!(f64);
    }
}
