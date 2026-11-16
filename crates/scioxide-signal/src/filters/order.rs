//! Sorted-window updates and the median absolute deviation for the Hampel filter.
use scioxide_core::{NumericalIssue, RealScalar, SciOxideError};

#[inline]
pub(super) fn compare<S: RealScalar>(a: &S, b: &S) -> core::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
}

// Remove one representation of the outgoing observation and insert the new
// value by moving only the interval between their positions. For the short
// windows used by local filters this also avoids two overlapping bulk moves.
#[inline]
pub(super) fn replace<S: RealScalar>(
    window: &mut [S],
    old: S,
    new: S,
) -> Result<(), SciOxideError> {
    if old == new && old.is_sign_negative() == new.is_sign_negative() {
        return Ok(());
    }
    let mut index = if window.len() <= 32 {
        window
            .iter()
            .position(|x| {
                *x == old && (old != S::ZERO || x.is_sign_negative() == old.is_sign_negative())
            })
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "sorted filter window lost an outgoing sample",
                )
            })?
    } else {
        window.partition_point(|x| *x < old)
    };
    while window[index].is_sign_negative() != old.is_sign_negative() {
        index += 1;
    }
    while index > 0 && new < window[index - 1] {
        window[index] = window[index - 1];
        index -= 1;
    }
    while index + 1 < window.len() && window[index + 1] < new {
        window[index] = window[index + 1];
        index += 1;
    }
    window[index] = new;
    Ok(())
}

#[inline]
fn normal_or_exact_zero<S: RealScalar>(value: S, zero: bool) -> bool {
    value.is_finite() && (value.abs() >= S::MIN_POSITIVE || (value == S::ZERO && zero))
}

// Every native subtraction/product must have the same rounded significand as
// Scaled's corresponding operation. Exclude underflow and overflow, including
// a product rounded to zero. Keeping the two products separate is intentional.
#[inline]
pub(super) fn hampel<S: RealScalar>(center: S, window: &[S], factor: S, threshold: S) -> Option<S> {
    let middle = window.len() / 2;
    let median = window[middle];
    let mut left = middle;
    let mut right = middle + 1;
    let mut mad = S::ZERO;
    for _ in 0..middle {
        let a = if left > 0 {
            let a = median - window[left - 1];
            if !normal_or_exact_zero(a, median == window[left - 1]) {
                return None;
            }
            a
        } else {
            S::INFINITY
        };
        let b = if right < window.len() {
            let b = window[right] - median;
            if !normal_or_exact_zero(b, window[right] == median) {
                return None;
            }
            b
        } else {
            S::INFINITY
        };
        if a <= b {
            mad = a;
            left -= 1;
        } else {
            mad = b;
            right += 1;
        }
    }
    let distance = (center - median).abs();
    let scaled_mad = mad * factor;
    let limit = scaled_mad * threshold;
    if normal_or_exact_zero(distance, center == median)
        && normal_or_exact_zero(scaled_mad, mad == S::ZERO)
        && normal_or_exact_zero(limit, scaled_mad == S::ZERO || threshold == S::ZERO)
    {
        Some(if distance > limit { median } else { center })
    } else {
        None
    }
}
