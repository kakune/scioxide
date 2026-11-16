//! Unchecked short sums for operands checked in advance against a safe range.
use crate::RealScalar;

/// Magnitude range in which short unchecked sums cannot overflow or underflow.
///
/// A value is in range if it is zero or its magnitude lies in
///
/// ```text
/// [√MIN_POSITIVE / ε,  √MAX / (8 L)]
/// ```
///
/// where `ε` is the machine epsilon of `S`, `MIN_POSITIVE` and `MAX` are the
/// smallest normal and largest finite values, and `L` is the reduction length
/// rounded up to a power of two. The upper bound keeps pair sums, deviations and
/// sums of products of at most `L` terms below overflow. The lower bound keeps
/// those products, and error bounds scaled by `ε`, normal. NaN, infinities and
/// subnormals are never in range. If every operand is in range, the caller can
/// use [`four_lane_sums`] and bound its rounding error itself; otherwise it
/// should use a checked or extended-range sum.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::ShortReductionRange;
///
/// let range = ShortReductionRange::<f64>::new(15);
/// assert!(range.admits(0.0) && range.admits(1.0));
/// assert!(!range.admits(f64::MIN_POSITIVE) && !range.admits(f64::MAX));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShortReductionRange<S> {
    lower: S,
    upper: S,
}

impl<S: RealScalar> ShortReductionRange<S> {
    /// Returns the range for sums of at most `length` terms.
    #[inline(always)]
    pub fn new(length: usize) -> Self {
        Self {
            lower: S::MIN_POSITIVE.sqrt() / S::EPSILON,
            upper: S::MAX.sqrt()
                / (S::from_f64_ieee(8.0)
                    * S::from_f64_ieee(length.max(1).next_power_of_two() as f64)),
        }
    }

    /// Returns whether `value` is zero or inside the magnitude range.
    #[inline(always)]
    pub fn admits(self, value: S) -> bool {
        value == S::ZERO || value.abs() >= self.lower && value.abs() <= self.upper
    }

    /// Returns whether every value is in range, like
    /// `values.iter().all(|&v| self.admits(v))`.
    ///
    /// The check is one branch-free pass that compilers can vectorize, which
    /// suits short blocks of operands checked together.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::backend::ShortReductionRange;
    ///
    /// let range = ShortReductionRange::<f64>::new(25);
    /// assert!(range.admits_all(&[0.0, -1.0, 2.0]));
    /// assert!(!range.admits_all(&[1.0, f64::MIN_POSITIVE]));
    /// ```
    #[inline(always)]
    pub fn admits_all(self, values: &[S]) -> bool {
        values.iter().fold(true, |admitted, &value| {
            let magnitude = value.abs();
            admitted
                & ((value == S::ZERO) | ((magnitude >= self.lower) & (magnitude <= self.upper)))
        })
    }
}

/// Sums `count` rows of `N` streams in four plain floating-point lanes.
///
/// `term(i)` returns row `i`; each row is requested once, in index order. Term
/// `i` goes to lane `i % 4`, and the lanes are combined as
/// `(l0 + l1) + (l2 + l3)`. For at most 64 terms the addition depth is at most
/// 17. The function does not check its inputs: the caller must ensure every
/// operand is in a [`ShortReductionRange`] and bound the rounding error itself.
/// Cost is `O(count · N)`.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::four_lane_sums;
///
/// let values = [1.0_f64, 2.0, 3.0];
/// let [sum, squares] = four_lane_sums(values.len(), |i| [values[i], values[i] * values[i]]);
/// assert_eq!((sum, squares), (6.0, 14.0));
/// ```
#[inline(always)]
pub fn four_lane_sums<S: RealScalar, const N: usize>(
    count: usize,
    mut term: impl FnMut(usize) -> [S; N],
) -> [S; N] {
    #[inline(always)]
    fn append<S: RealScalar, const N: usize>(lane: &mut [S; N], terms: [S; N]) {
        for (sum, term) in lane.iter_mut().zip(terms) {
            *sum += term;
        }
    }
    let mut lanes = [[S::ZERO; N]; 4];
    for block in 0..count / 4 {
        let base = block * 4;
        append(&mut lanes[0], term(base));
        append(&mut lanes[1], term(base + 1));
        append(&mut lanes[2], term(base + 2));
        append(&mut lanes[3], term(base + 3));
    }
    for index in count - count % 4..count {
        append(&mut lanes[index % 4], term(index));
    }
    core::array::from_fn(|i| (lanes[0][i] + lanes[1][i]) + (lanes[2][i] + lanes[3][i]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn reduction_range_admits_exact_boundaries_and_rejects_nonfinite_operands() {
        fn check<S: RealScalar>() {
            for length in [1, 9, 25, 51, 64] {
                let range = ShortReductionRange::<S>::new(length);
                let admitted = [
                    S::ZERO,
                    -S::ZERO,
                    range.lower,
                    -range.lower,
                    range.upper,
                    -range.upper,
                ];
                for value in admitted {
                    assert!(range.admits(value));
                }
                assert!(range.admits_all(&admitted) && range.admits_all(&[]));
                for value in [
                    S::MIN_SUBNORMAL,
                    range.lower * S::HALF,
                    range.upper * S::TWO,
                    S::NAN,
                    S::INFINITY,
                    -S::INFINITY,
                ] {
                    assert!(!range.admits(value));
                    let mut block = admitted;
                    block[length % admitted.len()] = value;
                    assert!(!range.admits_all(&block));
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn four_lane_sums_fit_the_short_rule_roundoff_allowance() {
        fn check<S: RealScalar>() {
            for count in [0, 1, 3, 4, 7, 25, 64] {
                let mut exact = [
                    crate::RangeAccumulator::new(),
                    crate::RangeAccumulator::new(),
                ];
                let result = four_lane_sums(count, |i| {
                    let terms = [
                        if i % 2 == 0 {
                            S::ONE
                        } else {
                            S::EPSILON * S::HALF
                        },
                        S::MAX / S::try_from_usize_exact(256).unwrap(),
                    ];
                    for (sum, &term) in exact.iter_mut().zip(&terms) {
                        sum.add(term).unwrap();
                    }
                    terms
                });
                for (actual, exact) in result.into_iter().zip(exact) {
                    let expected = exact.finish().unwrap();
                    assert!(actual.is_finite() && actual >= S::ZERO);
                    assert!(
                        (actual - expected).abs()
                            <= (S::try_from_usize_exact(17).unwrap() * S::EPSILON) * expected
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
