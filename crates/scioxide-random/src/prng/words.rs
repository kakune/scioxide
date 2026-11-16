//! Exact conversion of restricted native ranges to full-width uniform words.

use scioxide_core::SciOxideError;

use crate::util::MAX_SAMPLING_ATTEMPTS;

/// Accumulate base-`radix` digits, first draw least significant, then reject the
/// incomplete upper interval before reduction modulo 2^bits. Every accepted
/// output has exactly floor(radix^digits / 2^bits) preimages. Native draws must
/// lie in 0..radix; callers subtract any nonzero native lower bound.
///
/// Only built-in ranges (2..=2^48) and widths (1..=64) reach this helper. The
/// accumulated range is below 2^112, so all arithmetic fits u128. No entropy is
/// cached: native checkpoints alone reproduce mixed native/uniform/float calls.
/// Callers pass constant widths and radices, so inlining folds the digit count
/// and acceptance limit away.
#[inline]
pub(super) fn uniform_word(
    bits: u32,
    radix: u64,
    mut draw: impl FnMut() -> Result<u64, SciOxideError>,
) -> Result<u64, SciOxideError> {
    debug_assert!((1..=64).contains(&bits) && (2..=1_u64 << 48).contains(&radix));
    let radix = u128::from(radix);
    let target = 1_u128 << bits;
    let mut range = radix;
    let mut digits = 1;
    while range < target {
        range *= radix;
        digits += 1;
    }
    let limit = range - range % target;
    for _ in 0..MAX_SAMPLING_ATTEMPTS {
        let mut value = 0;
        let mut place = 1;
        for _ in 0..digits {
            let digit = u128::from(draw()?);
            debug_assert!(digit < radix);
            value += digit * place;
            place *= radix;
        }
        if value < limit {
            // The mask restricts the result to at most 64 bits.
            #[allow(clippy::cast_possible_truncation)]
            return Ok((value & (target - 1)) as u64);
        }
    }
    Err(normalization_exhausted())
}

#[cold]
#[inline(never)]
fn normalization_exhausted() -> SciOxideError {
    crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn exhaustive_nonbinary_preimages_and_rejection() {
        for (radix, bits) in [(3_u64, 2), (5, 3), (7, 5)] {
            let target = 1_u64 << bits;
            let limit = radix * radix / target * target;
            let mut counts = vec![0; target as usize];
            for candidate in 0..radix * radix {
                let script = [candidate % radix, candidate / radix, 0, 0];
                let mut reads = 0;
                let actual = uniform_word(bits, radix, || {
                    let value = script[reads];
                    reads += 1;
                    Ok(value)
                })
                .unwrap();
                if candidate < limit {
                    assert_eq!(reads, 2);
                    counts[actual as usize] += 1;
                } else {
                    assert_eq!((reads, actual), (4, 0));
                }
            }
            assert!(counts.into_iter().all(|n| n == limit / target));
        }
    }

    #[test]
    fn dyadic_digits_pack_without_gaps() {
        let mut words = [0xab_cdef, 0x12_3456, 0x78_9abc].into_iter();
        assert_eq!(
            uniform_word(64, 1 << 24, || Ok(words.next().unwrap())).unwrap(),
            0x9abc_1234_56ab_cdef
        );
        let mut words = [0xab_cdef, 0x12_3456].into_iter();
        assert_eq!(
            uniform_word(32, 1 << 24, || Ok(words.next().unwrap())).unwrap(),
            0x56ab_cdef
        );
        let mut words = [0x1234_5678_9abc, 0xdef0_1234_5678].into_iter();
        assert_eq!(
            uniform_word(64, 1 << 48, || Ok(words.next().unwrap())).unwrap(),
            0x5678_1234_5678_9abc
        );
    }

    #[test]
    fn retries_are_bounded_and_provider_failures_propagate() {
        let mut reads = 0;
        let error = uniform_word(2, 3, || {
            reads += 1;
            Ok(2)
        })
        .unwrap_err();
        assert_eq!(reads, 2 * MAX_SAMPLING_ATTEMPTS);
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
        let mut reads = 0;
        let error = uniform_word(64, 1 << 24, || {
            reads += 1;
            if reads == 2 {
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("provider"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "injected failure",
                ))
            } else {
                Ok(0)
            }
        })
        .unwrap_err();
        assert_eq!(reads, 2);
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }
}
