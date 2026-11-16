//! Exact fixed-width integer arithmetic on weights, for empirical quantiles and tails.
//!
//! [`ExactMass`] is a 54-limb unsigned integer in units of `2^-1075`. It holds
//! every finite non-negative `f64` weight, every sum of such weights over a
//! slice, and the product of such a sum with the significand of an `f64`
//! probability exactly. [`ExactTarget`] is such a product over a power-of-two
//! denominator, and [`ExactWeightedMean`] accumulates observations with exact
//! weights. Quantile positions, weighted selection and tail means need shifts,
//! ordered comparison and subtraction of targets, ceilings and division by
//! small integers on these values.
//!
//! [`scioxide_core::RangeAccumulator`] is a signed superaccumulator that rounds
//! once at the end. It has no shifts, no ordered comparison or subtraction of
//! two exact states, and no conversion of the sum to an integer.

use core::cmp::Ordering;

use scioxide_core::{NumericalIssue, RealScalar, Scaled, SciOxideError};

use super::QuantileProbability;
use crate::arithmetic::affine_with_fractions;

pub(super) const EXACT_MASS_MIN_EXPONENT: i32 = -1_075;
// A finite f64 weight spans at most 2,099 base-two mass bits. An exact sum can
// gain at most usize::BITS more, and multiplying by an f64 probability plus
// retaining its denominator needs another 1,127 bits. Keep the representation
// fixed-size and allocation-free while covering that complete domain.
const EXACT_MASS_LIMBS: usize = 54;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExactMass {
    limbs: [u64; EXACT_MASS_LIMBS],
}

impl ExactMass {
    pub(crate) const fn zero() -> Self {
        Self {
            limbs: [0; EXACT_MASS_LIMBS],
        }
    }

    pub(super) fn count(value: usize) -> Self {
        let mut unit = Self::zero();
        let unit_bit = (-EXACT_MASS_MIN_EXPONENT) as usize;
        unit.limbs[unit_bit / u64::BITS as usize] = 1_u64 << (unit_bit % u64::BITS as usize);
        unit.multiplied_by_u64(value as u64)
    }

    pub(crate) fn try_add_scalar<S: RealScalar>(&mut self, value: S) -> Result<(), SciOxideError> {
        if value == S::ZERO {
            return Ok(());
        }
        let scioxide_core::FloatDecomposition::Finite {
            significand: mantissa,
            exponent,
            negative,
            ..
        } = value.decompose()
        else {
            return Err(scioxide_core::SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "finite scalar required",
            ));
        };
        let sign = if negative { -1_i8 } else { 1_i8 };
        let shift = i32::from(exponent) - EXACT_MASS_MIN_EXPONENT;
        let significant_bits = u64::BITS - mantissa.leading_zeros();
        let capacity = u32::try_from(EXACT_MASS_LIMBS * u64::BITS as usize).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("exact_empirical_mass_capacity"))
        })?;
        if sign < 0
            || shift < 0
            || u32::try_from(shift)
                .ok()
                .and_then(|offset| offset.checked_add(significant_bits))
                .is_none_or(|end| end > capacity)
        {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "empirical weight cannot be represented by the fixed exact accumulator",
            ));
        }
        self.add_decoded(mantissa, shift as usize);
        Ok(())
    }

    pub(super) fn add_validated_scalar<S: RealScalar>(&mut self, value: S) {
        if value == S::ZERO {
            return;
        }
        let scioxide_core::FloatDecomposition::Finite {
            significand: mantissa,
            exponent,
            ..
        } = value.decompose()
        else {
            unreachable!("caller established a finite scalar before decomposition");
        };
        let shift = (i32::from(exponent) - EXACT_MASS_MIN_EXPONENT) as usize;
        self.add_decoded(mantissa, shift);
    }

    fn add_decoded(&mut self, mantissa: u64, shift: usize) {
        let limb = shift / u64::BITS as usize;
        let offset = shift % u64::BITS as usize;
        self.add_word(limb, mantissa << offset);
        if offset != 0 {
            self.add_word(limb + 1, mantissa >> (u64::BITS as usize - offset));
        }
    }

    fn add_word(&mut self, mut index: usize, mut word: u64) {
        while word != 0 {
            let (next, carry) = self.limbs[index].overflowing_add(word);
            self.limbs[index] = next;
            word = u64::from(carry);
            index += usize::from(carry);
        }
    }

    pub(super) fn add_assign(&mut self, other: Self) {
        let mut carry = false;
        for (left, right) in self.limbs.iter_mut().zip(other.limbs) {
            let (with_right, right_carry) = left.overflowing_add(right);
            let (with_carry, carry_carry) = with_right.overflowing_add(u64::from(carry));
            *left = with_carry;
            carry = right_carry || carry_carry;
        }
    }

    pub(super) fn multiplied_by_u64(mut self, mut factor: u64) -> Self {
        let mut result = Self::zero();
        while factor != 0 {
            if factor & 1 == 1 {
                result.add_assign(self);
            }
            factor >>= 1;
            if factor != 0 {
                self.double();
            }
        }
        result
    }

    /// Divides by a small nonzero integer, discarding the remainder.
    pub(super) fn divided_by_usize(mut self, divisor: usize) -> Self {
        let mut remainder = 0_u128;
        for limb in self.limbs.iter_mut().rev() {
            let word = (remainder << 64) | u128::from(*limb);
            *limb = (word / divisor as u128) as u64;
            remainder = word % divisor as u128;
        }
        self
    }

    pub(super) fn shifted_left(self, shift: usize) -> Self {
        if shift == 0 {
            return self;
        }
        let word_shift = shift / u64::BITS as usize;
        let bit_shift = shift % u64::BITS as usize;
        let mut result = Self::zero();
        for (source_index, word) in self.limbs.into_iter().enumerate() {
            if word == 0 {
                continue;
            }
            let target_index = source_index + word_shift;
            debug_assert!(target_index < EXACT_MASS_LIMBS);
            result.limbs[target_index] |= word << bit_shift;
            if bit_shift != 0 {
                debug_assert!(target_index + 1 < EXACT_MASS_LIMBS);
                result.limbs[target_index + 1] |= word >> (u64::BITS as usize - bit_shift);
            }
        }
        result
    }

    pub(super) fn shifted_right(self, shift: usize) -> Self {
        if shift == 0 {
            return self;
        }
        let word_shift = shift / u64::BITS as usize;
        let bit_shift = shift % u64::BITS as usize;
        let mut result = Self::zero();
        for target_index in 0..EXACT_MASS_LIMBS - word_shift {
            let source_index = target_index + word_shift;
            result.limbs[target_index] = self.limbs[source_index] >> bit_shift;
            if bit_shift != 0 && source_index + 1 < EXACT_MASS_LIMBS {
                result.limbs[target_index] |=
                    self.limbs[source_index + 1] << (u64::BITS as usize - bit_shift);
            }
        }
        result
    }

    fn increment(&mut self) {
        self.add_word(0, 1);
    }

    fn double(&mut self) {
        let mut carry = 0_u64;
        for limb in &mut self.limbs {
            let next_carry = *limb >> (u64::BITS - 1);
            *limb = (*limb << 1) | carry;
            carry = next_carry;
        }
    }

    pub(super) fn subtract_assign(&mut self, other: Self) {
        let mut borrow = false;
        for (left, right) in self.limbs.iter_mut().zip(other.limbs) {
            let (without_right, right_borrow) = left.overflowing_sub(right);
            let (without_borrow, borrow_borrow) = without_right.overflowing_sub(u64::from(borrow));
            *left = without_borrow;
            borrow = right_borrow || borrow_borrow;
        }
    }

    pub(super) fn compare(self, other: Self) -> Ordering {
        self.limbs
            .iter()
            .zip(other.limbs.iter())
            .rev()
            .find_map(|(left, right)| (left != right).then(|| left.cmp(right)))
            .unwrap_or(Ordering::Equal)
    }

    pub(super) fn highest_bit(self) -> Option<usize> {
        self.limbs.iter().rposition(|limb| *limb != 0).map(|index| {
            index * u64::BITS as usize
                + (u64::BITS - 1 - self.limbs[index].leading_zeros()) as usize
        })
    }

    fn bit(self, index: usize) -> bool {
        self.limbs[index / u64::BITS as usize] & (1_u64 << (index % u64::BITS as usize)) != 0
    }

    fn any_below(self, exclusive_end: usize) -> bool {
        let complete_limbs = exclusive_end / u64::BITS as usize;
        if self.limbs[..complete_limbs].iter().any(|limb| *limb != 0) {
            return true;
        }
        let trailing_bits = exclusive_end % u64::BITS as usize;
        trailing_bits != 0 && self.limbs[complete_limbs] & ((1_u64 << trailing_bits) - 1) != 0
    }

    fn bits(self, start: usize, end: usize) -> u64 {
        (start..end)
            .enumerate()
            .fold(0_u64, |bits, (offset, index)| {
                bits | (u64::from(self.bit(index)) << offset)
            })
    }

    pub(crate) fn to_scaled(self, binary_exponent: i32) -> Result<Scaled<f64>, SciOxideError> {
        let Some(highest_bit) = self.highest_bit() else {
            return Ok(Scaled::<f64>::ZERO);
        };
        const PRECISION: usize = f64::MANTISSA_DIGITS as usize;
        let retained_start = (highest_bit + 1).saturating_sub(PRECISION);
        let mut significand = self.bits(retained_start, highest_bit + 1);
        if retained_start != 0 {
            let guard_index = retained_start - 1;
            let round_up =
                self.bit(guard_index) && (self.any_below(guard_index) || significand & 1 == 1);
            if round_up {
                significand += 1;
                if significand == 1_u64 << PRECISION {
                    let exponent = EXACT_MASS_MIN_EXPONENT
                        .checked_add(binary_exponent)
                        .and_then(|value| value.checked_add(highest_bit as i32 + 1))
                        .ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "exact_empirical_mass_exponent"
                            ))
                        })?;
                    return Scaled::<f64>::from_parts(1.0, i64::from(exponent));
                }
            }
        }
        let retained_width = highest_bit + 1 - retained_start;
        let normalized = significand as f64 / 2.0_f64.powi((retained_width - 1) as i32);
        let exponent = EXACT_MASS_MIN_EXPONENT
            .checked_add(binary_exponent)
            .and_then(|value| value.checked_add(highest_bit as i32))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "exact_empirical_mass_exponent"
                ))
            })?;
        Scaled::<f64>::from_parts(normalized, i64::from(exponent))
    }

    pub(super) fn to_optional_scalar<S: RealScalar>(self) -> Option<S> {
        self.to_scaled(0).ok()?.try_value_to().ok()
    }

    pub(super) fn to_usize(self) -> Option<usize> {
        let value = self.limbs[0] as usize;
        (self.limbs[1..].iter().all(|limb| *limb == 0)
            && (usize::BITS == u64::BITS || value as u64 == self.limbs[0]))
            .then_some(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExactTarget {
    numerator: ExactMass,
    pub(super) denominator_shift: u32,
}

impl ExactTarget {
    pub(super) fn lower(total: ExactMass, probability: QuantileProbability) -> Self {
        let (significand, denominator_shift) = probability_parts(probability);
        Self {
            numerator: total.multiplied_by_u64(significand),
            denominator_shift,
        }
    }

    pub(super) fn upper(total: ExactMass, probability: QuantileProbability) -> Self {
        let lower = Self::lower(total, probability);
        let mut numerator = total.shifted_left(lower.denominator_shift as usize);
        numerator.subtract_assign(lower.numerator);
        Self {
            numerator,
            denominator_shift: lower.denominator_shift,
        }
    }

    pub(super) fn from_integer(mass: ExactMass, denominator_shift: u32) -> Self {
        Self {
            numerator: mass.shifted_left(denominator_shift as usize),
            denominator_shift,
        }
    }

    pub(super) fn is_zero(self) -> bool {
        self.numerator.highest_bit().is_none()
    }

    pub(super) fn compare_integer(self, mass: ExactMass) -> Ordering {
        self.numerator
            .compare(mass.shifted_left(self.denominator_shift as usize))
    }

    pub(super) fn add(self, other: Self) -> Self {
        debug_assert_eq!(self.denominator_shift, other.denominator_shift);
        let mut numerator = self.numerator;
        numerator.add_assign(other.numerator);
        Self {
            numerator,
            denominator_shift: self.denominator_shift,
        }
    }

    pub(super) fn subtract_integer(&mut self, mass: ExactMass) {
        self.numerator
            .subtract_assign(mass.shifted_left(self.denominator_shift as usize));
    }

    pub(super) fn ceil_mass(self) -> ExactMass {
        let mut mass = self
            .numerator
            .shifted_right(self.denominator_shift as usize);
        if self.numerator.any_below(self.denominator_shift as usize) {
            mass.increment();
        }
        mass
    }

    pub(super) fn ceil_count(self) -> Result<usize, SciOxideError> {
        let mass_shift = usize::try_from(-EXACT_MASS_MIN_EXPONENT).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("exact_count_mass_exponent"))
        })?;
        let shift = mass_shift
            .checked_add(self.denominator_shift as usize)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "exact_count_target_denominator"
                ))
            })?;
        let mut count = self
            .numerator
            .shifted_right(shift)
            .to_usize()
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "exact_empirical_tail_included_count"
                ))
            })?;
        if self.numerator.any_below(shift) {
            count = count.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "empirical_tail_included_count"
                ))
            })?;
        }
        Ok(count)
    }

    pub(super) fn to_scaled(self) -> Result<Scaled<f64>, SciOxideError> {
        let exponent = i32::try_from(self.denominator_shift).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("exact_target_denominator"))
        })?;
        self.numerator.to_scaled(-exponent)
    }

    pub(super) fn to_optional_scalar<S: RealScalar>(self) -> Option<S> {
        self.to_scaled().ok()?.try_value_to().ok()
    }
}

/// Returns `(significand, shift)` with `p = significand / 2^shift` exactly.
/// Every probability in `[0, 1]` has a nonpositive binary exponent, and zero
/// has significand zero.
pub(super) fn probability_parts(probability: QuantileProbability) -> (u64, u32) {
    match RealScalar::decompose(probability.get()) {
        scioxide_core::FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } => (significand, u32::from(exponent.unsigned_abs())),
        // A validated probability is finite.
        _ => (0, 1_074),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ExactWeightedMean<S: RealScalar> {
    mass: ExactTarget,
    pub(super) mean: S,
}

impl<S: RealScalar> ExactWeightedMean<S> {
    pub(super) fn zero(denominator_shift: u32) -> Self {
        Self {
            mass: ExactTarget {
                numerator: ExactMass::zero(),
                denominator_shift,
            },
            mean: S::ZERO,
        }
    }

    pub(super) fn add(self, value: S, portion: ExactTarget) -> Result<Self, SciOxideError> {
        if portion.is_zero() {
            return Ok(self);
        }
        let next_mass = self.mass.add(portion);
        let next_mean = if self.mass.is_zero() {
            value
        } else {
            let left_fraction = self.mass.to_scaled()?.div(next_mass.to_scaled()?)?;
            let right_fraction = portion.to_scaled()?.div(next_mass.to_scaled()?)?;
            affine_with_fractions(self.mean, value, left_fraction, right_fraction)?
        };
        Ok(Self {
            mass: next_mass,
            mean: next_mean,
        })
    }
}

pub(super) fn validated_exact_mass<S: RealScalar>(weights: impl Iterator<Item = S>) -> ExactMass {
    let mut mass = ExactMass::zero();
    for weight in weights {
        mass.add_validated_scalar(weight);
    }
    mass
}
