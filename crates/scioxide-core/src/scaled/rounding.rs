//! Integer-grid rounding shared by exact accumulation and scaled components.
use crate::RealScalar;

#[inline]
pub(crate) fn highest_bit(words: &[u64]) -> Option<usize> {
    words.iter().rposition(|word| *word != 0).map(|index| {
        index * u64::BITS as usize + (u64::BITS - 1 - words[index].leading_zeros()) as usize
    })
}

#[inline]
pub(crate) fn bit(words: &[u64], index: usize) -> bool {
    words.get(index / u64::BITS as usize).copied().unwrap_or(0)
        & (1_u64 << (index % u64::BITS as usize))
        != 0
}

#[inline]
pub(crate) fn any_bits_below(words: &[u64], limit: usize) -> bool {
    let complete_words = limit / u64::BITS as usize;
    if words[..complete_words.min(words.len())]
        .iter()
        .any(|word| *word != 0)
    {
        return true;
    }
    let remainder = limit % u64::BITS as usize;
    remainder != 0
        && words.get(complete_words).copied().unwrap_or(0) & ((1_u64 << remainder) - 1) != 0
}

#[inline]
pub(crate) fn shifted_low_u64(words: &[u64], shift: usize) -> u64 {
    let word = shift / u64::BITS as usize;
    let offset = shift % u64::BITS as usize;
    if offset == 0 {
        words.get(word).copied().unwrap_or(0)
    } else {
        words.get(word).copied().unwrap_or(0) >> offset
            | words.get(word + 1).copied().unwrap_or(0) << (64 - offset)
    }
}

pub(crate) fn minimum_subnormal_exponent<S: RealScalar>() -> i32 {
    // RealScalar is sealed to binary32 and binary64. Their minimum normal
    // exponents are -126 and -1022; subnormals extend another p - 1 bits.
    if S::MANTISSA_DIGITS == f32::MANTISSA_DIGITS {
        f32::MIN_EXP - f32::MANTISSA_DIGITS as i32
    } else {
        f64::MIN_EXP - f64::MANTISSA_DIGITS as i32
    }
}

// The sealed IEEE formats have maximum normal exponent one above the
// magnitude of their minimum normal exponent.
pub(crate) fn maximum_normal_exponent<S: RealScalar>() -> i32 {
    2 - S::MANTISSA_DIGITS as i32 - minimum_subnormal_exponent::<S>()
}

#[inline(always)]
pub(crate) fn rounded_words_at_exponent<S: RealScalar>(
    words: &[u64],
    negative: bool,
    exponent: i32,
    fractional_remainder: bool,
) -> S {
    let Some(top_bit) = highest_bit(words) else {
        return if negative { S::NEG_ZERO } else { S::ZERO };
    };
    // Discard the larger of two amounts: the bits beyond the destination
    // precision, and the bits below the smallest subnormal quantum. Rounding once
    // at that position gives correct results for normal and subnormal outputs
    // alike, with no double rounding. Discarded bits and any external remainder
    // act as sticky bits for round-half-even.
    let precision_shift = (top_bit + 1).saturating_sub(S::MANTISSA_DIGITS as usize);
    let subnormal_shift = (minimum_subnormal_exponent::<S>() - exponent).max(0) as usize;
    let shift = precision_shift.max(subnormal_shift);
    let significand = rounded_integer(words, shift, fractional_remainder);

    // The destination grid has already been rounded above. Assemble ordinary
    // results directly as binary64 bits instead of performing two scalbn calls.
    // Binary32 grid points (including its subnormals) are exact binary64 values,
    // so the final conversion introduces no second rounding. Exponent extremes
    // retain the original range-safe scaling below.
    if significand != 0 {
        let carry = significand == 1_u64 << S::MANTISSA_DIGITS;
        let normalized = if carry { significand >> 1 } else { significand };
        let width = 64 - normalized.leading_zeros();
        let top = exponent + shift as i32 + i32::from(carry) + width as i32 - 1;
        if (-1022..=1023).contains(&top) {
            let fraction = (normalized << (53 - width)) & ((1_u64 << 52) - 1);
            let bits = (u64::from(negative) << 63) | (((top + 1023) as u64) << 52) | fraction;
            return S::from_f64_ieee(f64::from_bits(bits));
        }
    }

    let precision = S::MANTISSA_DIGITS;
    // Rounding up can carry into bit `precision`, which the [0, 1) conversion
    // below cannot represent. Otherwise the integer significand is converted
    // exactly as a fraction and rescaled. The final `scalbn` is exact apart from
    // overflow to infinity, which callers detect.
    let integer = if significand == 1_u64 << precision {
        S::ONE.scalbn(precision as i32)
    } else {
        S::from_u64_half_open01(significand << (64 - precision)).scalbn(precision as i32)
    };
    let candidate = integer.scalbn(exponent + shift as i32);
    if negative { -candidate } else { candidate }
}

// Nearest-even integer rounding, shared by exact states and component projection.
#[inline]
pub(crate) fn rounded_integer(words: &[u64], shift: usize, sticky: bool) -> u64 {
    let mut significand = shifted_low_u64(words, shift);
    if shift != 0
        && bit(words, shift - 1)
        && (any_bits_below(words, shift - 1) || sticky || significand & 1 != 0)
    {
        significand += 1;
    }
    significand
}
