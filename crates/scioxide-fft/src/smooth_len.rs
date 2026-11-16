//! Native smooth factorization and checked fast-length selection.
use scioxide_core::{Argument, SciOxideError};

pub(crate) fn fast_len(n: usize, next: bool, argument: Argument) -> Result<usize, SciOxideError> {
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "fast lengths must be positive",
        ));
    }
    // Depth-first enumeration of 2^a 3^b 5^c 7^d: `prime` indexes the factor
    // currently being raised, and index 4 means all exponents are fixed.
    // Raising a factor stops once the product reaches (next) or passes
    // (prev) `n`, since further powers cannot improve the candidate; checked
    // multiplication bounds the search at usize::MAX, so the number of visited
    // products stays polylogarithmic in `n`.
    fn visit(value: usize, prime: usize, n: usize, next: bool, best: &mut Option<usize>) {
        if prime == 4 {
            if (next && value >= n && best.is_none_or(|old| value < old))
                || (!next && value <= n && best.is_none_or(|old| value > old))
            {
                *best = Some(value);
            }
            return;
        }
        let mut value = value;
        loop {
            visit(value, prime + 1, n, next, best);
            if (next && value >= n) || (!next && value > n) {
                break;
            }
            let Some(product) = value.checked_mul(SMOOTH_PRIMES[prime]) else {
                break;
            };
            value = product;
        }
    }
    let mut best = None;
    visit(1, 0, n, next, &mut best);
    best.ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("next_fast_fft_length"))
    })
}

pub(crate) const SMOOTH_PRIMES: [usize; 4] = [2, 3, 5, 7];
/// Exponents of the native smooth factors and the unfactored remainder.
pub(crate) fn smooth_factors(mut n: usize) -> ([usize; 4], usize) {
    let exponents = SMOOTH_PRIMES.map(|prime| {
        let mut exponent = 0;
        while n > 1 && n.is_multiple_of(prime) {
            n /= prime;
            exponent += 1;
        }
        exponent
    });
    (exponents, n)
}
