//! Prime bases and radical-inverse decoding for the Halton families.
use super::plan::invalid;
use crate::{BitGenerator, SeedableRng, SplitMix64};
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SciOxideError};

const PRIMES: [u32; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

pub(super) fn default_bases(dim: usize) -> Result<Vec<u32>, SciOxideError> {
    if dim == 0 || dim > PRIMES.len() {
        return Err(invalid(
            scioxide_core::argument!("dimensions"),
            "built-in Halton supports 1 through 100 dimensions",
        ));
    }
    let mut result =
        scioxide_core::backend::try_with_capacity(dim, scioxide_core::quantity!("halton_bases"))?;
    result.extend_from_slice(&PRIMES[..dim]);
    Ok(result)
}
fn prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut factor = 3;
    while factor <= n / factor {
        if n.is_multiple_of(factor) {
            return false;
        }
        factor += 2;
    }
    true
}
pub(super) fn validate_bases(bases: &[u32]) -> Result<(), SciOxideError> {
    if bases.is_empty() || bases.iter().any(|&base| !prime(base)) {
        return Err(invalid(
            scioxide_core::argument!("bases"),
            "Halton requires nonempty prime bases",
        ));
    }
    let mut sorted = scioxide_core::backend::try_with_capacity(
        bases.len(),
        scioxide_core::quantity!("halton_base_validation"),
    )?;
    sorted.extend_from_slice(bases);
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(
            scioxide_core::argument!("bases"),
            "Halton bases must be distinct",
        ));
    }
    Ok(())
}
pub(super) fn validate_permutations(
    bases: &[u32],
    permutations: &[Vec<u32>],
) -> Result<(), SciOxideError> {
    if permutations.len() != bases.len() {
        return Err(invalid(
            scioxide_core::argument!("permutations"),
            "one permutation per base is required",
        ));
    }
    for (&base, permutation) in bases.iter().zip(permutations) {
        if permutation.len() != base as usize || permutation.first() != Some(&0) {
            return Err(invalid(
                scioxide_core::argument!("permutations"),
                "permutation must have base entries and fix zero",
            ));
        }
        let mut seen = scioxide_core::backend::try_filled(
            base as usize,
            false,
            scioxide_core::quantity!("halton_permutation_validation"),
        )?;
        for &digit in permutation {
            if digit >= base || seen[digit as usize] {
                return Err(invalid(
                    scioxide_core::argument!("permutations"),
                    "permutation must be a bijection on base digits",
                ));
            }
            seen[digit as usize] = true;
        }
    }
    Ok(())
}
pub(super) fn seeded_permutations(
    bases: &[u32],
    seed: u64,
) -> Result<Vec<Vec<u32>>, SciOxideError> {
    permutations(bases, &mut SplitMix64::from_seed(seed))
}
fn permutations<R: BitGenerator>(
    bases: &[u32],
    rng: &mut R,
) -> Result<Vec<Vec<u32>>, SciOxideError> {
    let mut result = scioxide_core::backend::try_with_capacity(
        bases.len(),
        scioxide_core::quantity!("halton_permutations"),
    )?;
    for &base in bases {
        let mut permutation = scioxide_core::backend::try_with_capacity(
            base as usize,
            scioxide_core::quantity!("halton_digit_permutation"),
        )?;
        permutation.extend(0..base);
        // Fisher-Yates over digits 1..base only: digit 0 stays fixed so that the
        // infinitely many leading zeros of an index still map to zero and the
        // scrambled radical inverse remains a finite sum.
        for i in (2..base).rev() {
            // The shared index draw: 32-bit words below 2^32 and the
            // 1,024-candidate rejection guard.
            let j = 1 + crate::sampling::uniform_below(u64::from(i), rng)? as usize;
            permutation.swap(i as usize, j);
        }
        result.push(permutation);
    }
    Ok(result)
}

/// Exact division by a prepared prime, plus the largest index whose reversed
/// fraction fits u64. The reciprocal is integer data, never a floating estimate.
#[derive(Debug, Clone, Copy)]
pub(super) struct Base {
    value: u64,
    reciprocal: u64,
    narrow_limit: u64,
}
impl Base {
    pub(super) fn new(base: u32) -> Self {
        let value = u64::from(base);
        let mut power = value;
        while let Some(next) = power.checked_mul(value) {
            power = next;
        }
        Self {
            value,
            reciprocal: ((1_u128 << 64) / u128::from(value)) as u64,
            narrow_limit: power - 1,
        }
    }

    #[inline]
    fn div_rem(self, n: u64) -> (u64, u64) {
        // floor(2^64 / b) underestimates the reciprocal by less than 1/2^64.
        // For n < 2^64 the provisional quotient is exact or one too small.
        // Therefore q*b <= n, and at most one remainder correction is needed.
        let quotient = ((u128::from(n) * u128::from(self.reciprocal)) >> 64) as u64;
        let remainder = n - quotient * self.value;
        if remainder >= self.value {
            (quotient + 1, remainder - self.value)
        } else {
            (quotient, remainder)
        }
    }
}

#[inline]
fn permuted(digit: u64, base: u64, permutation: Option<&[u32]>, reverse: bool) -> u64 {
    if let Some(p) = permutation {
        u64::from(p[digit as usize])
    } else if reverse && digit != 0 {
        base - digit
    } else {
        digit
    }
}

pub(super) fn radical_inverse<S: RealScalar>(
    mut n: u64,
    base: &Base,
    permutation: Option<&[u32]>,
    reverse: bool,
) -> Result<S, SciOxideError> {
    let value = if base.value == 2 {
        // Every admitted base-two permutation is the identity. Reversing all
        // 64 bits only rescales the original integer fraction by a power of two;
        // integer-to-f64 rounding and the final scalar conversion are unchanged.
        (n.reverse_bits() as f64) * (1.0 / 18_446_744_073_709_551_616.0)
    } else if n <= base.narrow_limit {
        let mut numerator = 0_u64;
        let mut denominator = 1_u64;
        while n != 0 {
            let (quotient, digit) = base.div_rem(n);
            numerator = numerator * base.value + permuted(digit, base.value, permutation, reverse);
            denominator *= base.value;
            n = quotient;
        }
        numerator as f64 / denominator as f64
    } else {
        // base^digits <= base * 2^64 < 2^96: the full-range fraction fits u128.
        let mut numerator = 0_u128;
        let mut denominator = 1_u128;
        while n != 0 {
            let (quotient, digit) = base.div_rem(n);
            numerator = numerator * u128::from(base.value)
                + u128::from(permuted(digit, base.value, permutation, reverse));
            denominator *= u128::from(base.value);
            n = quotient;
        }
        numerator as f64 / denominator as f64
    };
    let rounded = S::try_from_f64_lossy(value)?;
    // Correct rounding to preserve the half-open output interval.
    Ok(if rounded >= S::ONE {
        S::from_u64_half_open01(u64::MAX)
    } else {
        rounded
    })
}

/// Fill one coordinate of a checked batch. Narrow reversed fractions advance
/// by integer digit carries: unlike a floating recurrence, this is the same
/// numerator/denominator as decoding each index independently, with no drift.
pub(super) fn fill_coordinate<S: RealScalar>(
    output: &mut [S],
    stride: usize,
    indices: core::ops::RangeInclusive<u64>,
    base: &Base,
    permutation: Option<&[u32]>,
    reverse: bool,
    shift: Option<S>,
) -> Result<(), SciOxideError> {
    let first = *indices.start();
    let last = *indices.end();
    if base.value == 2 || last > base.narrow_limit {
        for (index, value) in indices.zip(output.iter_mut().step_by(stride)) {
            *value = rotate(radical_inverse(index, base, permutation, reverse)?, shift)?;
        }
        return Ok(());
    }
    let mut remaining = first;
    let (mut numerator, mut denominator) = (0_u64, 1_u64);
    while remaining != 0 {
        let (quotient, digit) = base.div_rem(remaining);
        numerator = numerator * base.value + permuted(digit, base.value, permutation, reverse);
        denominator *= base.value;
        remaining = quotient;
    }
    let mut place = denominator / base.value;
    for (index, value) in indices.zip(output.iter_mut().step_by(stride)) {
        let rounded = S::try_from_f64_lossy(numerator as f64 / denominator as f64)?;
        *value = rotate(rounded.min(S::from_u64_half_open01(u64::MAX)), shift)?;
        if index == last {
            break;
        }
        if index + 1 == denominator {
            place = denominator;
            denominator *= base.value;
            numerator = permuted(1, base.value, permutation, reverse);
        } else {
            let mut remaining = index;
            let mut weight = place;
            loop {
                let (quotient, digit) = base.div_rem(remaining);
                numerator -= permuted(digit, base.value, permutation, reverse) * weight;
                if digit + 1 != base.value {
                    numerator += permuted(digit + 1, base.value, permutation, reverse) * weight;
                    break;
                }
                remaining = quotient;
                weight /= base.value;
            }
        }
    }
    Ok(())
}

/// Preserve scalar rounding before the Cranley-Patterson rotation and avoid
/// constructing a rounded upper endpoint when subtracting the wrap.
pub(super) fn rotate<S: RealScalar>(value: S, shift: Option<S>) -> Result<S, SciOxideError> {
    let Some(shift) = shift else {
        return Ok(value);
    };
    let (x, shift) = (value.into_f64(), shift.into_f64());
    let shifted = if x >= 1.0 - shift {
        x - (1.0 - shift)
    } else {
        x + shift
    };
    Ok(S::try_from_f64_lossy(shifted)?.min(S::from_u64_half_open01(u64::MAX)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    struct Zeros(usize);
    impl BitGenerator for Zeros {
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.0 += 1;
            Ok(0)
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0 += 1;
            Ok(0)
        }
        fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u32()
        }
        fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
            self.next_u64()
        }
        fn fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), SciOxideError> {
            bytes.fill(0);
            Ok(())
        }
    }
    #[test]
    fn permutation_rejection_exhaustion_is_a_work_limit() {
        let mut rng = Zeros(0);
        let error = permutations(&[5], &mut rng).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(rng.0, 1025); // One accepted word at bound 4, then 1024 rejections at 3.
    }
}
