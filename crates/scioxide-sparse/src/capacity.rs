//! Entry capacities of the one-shot factorizations.
//!
//! The entry limits of a factorization size its storage, so a constructor that
//! takes only the matrix chooses them: it starts from an estimate, doubles it
//! after each `ResourceLimit`, and stops at a bound that no factor of the
//! matrix exceeds. Kernels read entry limits only as capacities and in
//! comparisons, so a successful attempt returns the factor of any larger limits.
use scioxide_core::{SciOxideError, SciOxideErrorKind};

/// Runs `attempt(c)` for `c = min(initial, bound)`, then for `2c`, `4c`, …
/// capped at `bound`, while it fails with `ResourceLimit` below `bound`; any
/// other result returns at once. At most `⌈log₂(bound / initial)⌉ + 1`
/// attempts run.
pub(crate) fn with_capacity<T>(
    initial: usize,
    bound: usize,
    mut attempt: impl FnMut(usize) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut capacity = initial.min(bound);
    loop {
        match attempt(capacity) {
            Err(error) if error.kind() == SciOxideErrorKind::ResourceLimit && capacity < bound => {
                capacity = capacity.saturating_mul(2).max(1).min(bound);
            }
            result => return result,
        }
    }
}

/// Returns `n(n − 1)/2`, saturating at `usize::MAX`.
pub(crate) fn strict_triangle(n: usize) -> usize {
    halve_product(n, n.saturating_sub(1))
}

/// Returns `n(n + 1)/2`, saturating at `usize::MAX`.
pub(crate) fn triangle(n: usize) -> usize {
    halve_product(n, n.saturating_add(1))
}

// One of two consecutive integers is even, so halving it first keeps the
// product exact whenever it fits.
fn halve_product(a: usize, b: usize) -> usize {
    if a.is_multiple_of(2) {
        (a / 2).saturating_mul(b)
    } else {
        a.saturating_mul(b / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::{strict_triangle, triangle, with_capacity};
    use alloc::vec::Vec;
    use scioxide_core::{SciOxideError, SciOxideErrorKind};

    fn limit(required: usize, capacity: usize) -> Result<usize, SciOxideError> {
        if capacity < required {
            Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("entries"),
                required,
                capacity,
            ))
        } else {
            Ok(capacity)
        }
    }

    #[test]
    fn capacity_doubles_up_to_the_bound() {
        let mut seen = Vec::new();
        let result = with_capacity(3, 20, |c| {
            seen.push(c);
            limit(13, c)
        });
        assert_eq!(result.unwrap(), 20);
        assert_eq!(seen, [3, 6, 12, 20]);
        let mut seen = Vec::new();
        let error = with_capacity(3, 20, |c| {
            seen.push(c);
            limit(21, c)
        })
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
        assert_eq!(seen, [3, 6, 12, 20]);
    }

    #[test]
    fn zero_initial_capacity_grows_and_other_errors_return_at_once() {
        let mut seen = Vec::new();
        assert_eq!(
            with_capacity(0, 8, |c| {
                seen.push(c);
                limit(2, c)
            })
            .unwrap(),
            2
        );
        assert_eq!(seen, [0, 1, 2]);
        let mut calls = 0;
        let error = with_capacity(1, 8, |_| -> Result<(), SciOxideError> {
            calls += 1;
            Err(SciOxideError::singular(Some(0)))
        })
        .unwrap_err();
        assert_eq!((error.kind(), calls), (SciOxideErrorKind::Singular, 1));
        assert_eq!(with_capacity(9, 4, |c| limit(0, c)).unwrap(), 4);
    }

    #[test]
    fn triangle_counts_saturate() {
        assert_eq!((strict_triangle(0), triangle(0)), (0, 0));
        assert_eq!((strict_triangle(5), triangle(5)), (10, 15));
        assert_eq!(strict_triangle(usize::MAX), usize::MAX);
        assert_eq!(triangle(usize::MAX), usize::MAX);
    }
}
