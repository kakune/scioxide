//! Compensated summation in working precision.
use super::{non_finite_input, overflow, two_sum, validate_finite};
use crate::{RealScalar, SciOxideError};

/// Computes `N` compensated sums in one pass over an iterator of `[S; N]` rows.
///
/// Row `i` adds `row[j]` to stream `j`. Each stream is summed in four
/// interleaved lanes with Knuth's TwoSum, which keeps every addition's rounding
/// error; every 256 rows the lanes are merged in a fixed order. As with
/// Neumaier summation, the result carries first-order rounding errors in a
/// correction term: it is accurate in working precision, but neither exact nor
/// correctly rounded. The fixed order makes the result deterministic, though it
/// can round differently from calling [`CompensatedSum::add`] row by row.
///
/// The iterator is consumed once, up to its first `None`. Cost is `O(n N)` time
/// and a stack block of at most `256 N` scalars. Empty streams give
/// [`CompensatedSum::ZERO`], and `N = 0` consumes the iterator and returns `[]`.
/// Subnormal values are allowed.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an addend is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if a lane or merge part
///   overflows, even when another summation order would not. Use
///   [`ScaledAccumulator`](crate::ScaledAccumulator) for sums beyond the range
///   of `S`.
///
/// The final total is checked separately by [`CompensatedSum::total`].
///
/// # Examples
///
/// ```
/// use scioxide_core::compensated_reduce;
///
/// // Σx and Σx² over 1, 2, 3 in one pass.
/// let [sum, squares] = compensated_reduce([1.0_f64, 2.0, 3.0].map(|x| [x, x * x]))?;
/// assert_eq!(sum.total()?, 6.0);
/// assert_eq!(squares.total()?, 14.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn compensated_reduce<S: RealScalar, const N: usize>(
    values: impl IntoIterator<Item = [S; N]>,
) -> Result<[CompensatedSum<S>; N], SciOxideError> {
    let values = values.into_iter().fuse();
    if values.size_hint().1.is_some_and(|length| length <= 64) {
        // A complete short iterator has the same four-lane reduction order,
        // without clearing a full staging block for a few observations.
        reduce_block::<S, N, 64>(values)
    } else {
        reduce_block::<S, N, 256>(values)
    }
}

/// Merges four compensated lanes per stream, in lane order, into one
/// [`CompensatedSum`] per stream.
///
/// Each lane's sum and correction are merged without first rounding them to
/// one scalar. The function does not check its inputs, which the caller must
/// have checked to be finite. It returns `None` if a lane part or merged part is
/// not finite; the caller should then redo the sum with an extended-range
/// accumulator.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::finish_compensated_lanes;
///
/// let [sum] = finish_compensated_lanes(&[[1e16_f64, 1.0, -1e16, 0.0]], &[[0.0; 4]]).unwrap();
/// assert_eq!(sum.total()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn finish_compensated_lanes<S: RealScalar, const N: usize>(
    sums: &[[S; 4]; N],
    corrections: &[[S; 4]; N],
) -> Option<[CompensatedSum<S>; N]> {
    let mut result = [CompensatedSum::ZERO; N];
    for ((result, sums), corrections) in result.iter_mut().zip(sums).zip(corrections) {
        merge_lanes(result, *sums, *corrections).ok()?;
    }
    Some(result)
}

/// Computes the same sums as [`compensated_reduce`] over at most 64 rows given
/// by index.
///
/// `term(i)` returns row `i` of the `N` streams. Each row is requested once, in
/// order, and the arithmetic matches [`compensated_reduce`] exactly, so both
/// return identical parts. The caller must have checked its inputs. The function
/// returns `None` for more than 64 rows, or if a term, lane or merged part is not
/// finite; the caller should then redo the sum with an extended-range
/// accumulator.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::compensated_indexed;
/// let terms = [1e16_f64, 1.0, -1e16, 1.0, 1.0];
/// let [sum] = compensated_indexed(terms.len(), |i| [terms[i]]).unwrap();
/// assert_eq!(sum.total()?, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn compensated_indexed<S: RealScalar, const N: usize>(
    count: usize,
    mut term: impl FnMut(usize) -> [S; N],
) -> Option<[CompensatedSum<S>; N]> {
    if count > 64 {
        return None;
    }
    if count <= 4 {
        // Each lane holds one term and an exact zero residual, merged in lane order.
        let mut result = [CompensatedSum::ZERO; N];
        for index in 0..count {
            for (sum, value) in result.iter_mut().zip(term(index)) {
                sum.add(value).ok()?;
            }
        }
        return Some(result);
    }
    let initial = [term(0), term(1), term(2), term(3)];
    // The first addition to a lane is exact; adding positive zero retains the
    // reduction's canonical zero convention.
    let mut sums: [[S; 4]; N] = core::array::from_fn(|s| initial.map(|t| S::ZERO + t[s]));
    let mut corrections = [[S::ZERO; 4]; N];
    for base in (4..count).step_by(4) {
        let length = 4.min(count - base);
        let rows: [[S; N]; 4] = core::array::from_fn(|lane| {
            if lane < length {
                term(base + lane)
            } else {
                [S::ZERO; N]
            }
        });
        for (stream, (sums, corrections)) in sums.iter_mut().zip(&mut corrections).enumerate() {
            add_lanes(
                sums,
                corrections,
                rows[..length].iter().map(|row| row[stream]),
            );
        }
    }
    finish_compensated_lanes(&sums, &corrections)
}

/// Add one term to each leading lane, keeping its TwoSum residual. Callers
/// check the finished parts once; nonfinite values are absorbing.
#[inline(always)]
fn add_lanes<S: RealScalar>(
    sums: &mut [S; 4],
    corrections: &mut [S; 4],
    terms: impl Iterator<Item = S>,
) {
    for ((sum, correction), term) in sums.iter_mut().zip(corrections.iter_mut()).zip(terms) {
        let (next, error) = two_sum(*sum, term);
        *sum = next;
        *correction += error;
    }
}

/// Every lane part has already consumed finite inputs. IEEE nonfinite values
/// are absorbing under these additions, so one final check detects any lane
/// or merge overflow. TwoSum and Neumaier keep the same exact residual.
#[inline]
fn merge_lanes<S: RealScalar>(
    result: &mut CompensatedSum<S>,
    sums: [S; 4],
    corrections: [S; 4],
) -> Result<(), SciOxideError> {
    let mut sum = result.sum;
    let mut correction = result.correction;
    for value in sums
        .into_iter()
        .zip(corrections)
        .flat_map(|(sum, correction)| [sum, correction])
    {
        let (next, error) = two_sum(sum, value);
        sum = next;
        correction += error;
    }
    if !(sum.is_finite() & correction.is_finite()) {
        return Err(overflow("compensated block overflow"));
    }
    *result = CompensatedSum { sum, correction };
    Ok(())
}

#[inline]
fn reduce_block<S: RealScalar, const N: usize, const BLOCK: usize>(
    mut values: impl Iterator<Item = [S; N]>,
) -> Result<[CompensatedSum<S>; N], SciOxideError> {
    let mut result = [CompensatedSum::ZERO; N];
    // Materialize only one fixed stack block. Admission and iterator machinery
    // stay outside the arithmetic loop, which can use contiguous four-row loads.
    let mut block = [[S::ZERO; N]; BLOCK];
    loop {
        let mut length = 0;
        block
            .iter_mut()
            .zip(values.by_ref())
            .for_each(|(slot, value)| {
                *slot = value;
                length += 1;
            });
        if length == 0 {
            return Ok(result);
        }
        let block = &block[..length];
        if !block
            .as_flattened()
            .iter()
            .fold(true, |finite, &x| finite & x.is_finite())
        {
            return Err(non_finite_input("compensated block addends must be finite"));
        }
        for (stream, result) in result.iter_mut().enumerate() {
            if length <= 4 {
                // Each lane contains one term and an exact zero residual.
                // Merge those terms in lane order without adding empty parts.
                for row in block {
                    result.add(row[stream])?;
                }
                continue;
            }
            // The first addition to a lane is exact. Adding positive zero
            // explicitly retains the reduction's canonical zero convention.
            let mut sums = [
                S::ZERO + block[0][stream],
                S::ZERO + block[1][stream],
                S::ZERO + block[2][stream],
                S::ZERO + block[3][stream],
            ];
            let mut corrections = [S::ZERO; 4];
            let mut chunks = block[4..].chunks_exact(4);
            for rows in &mut chunks {
                add_lanes(
                    &mut sums,
                    &mut corrections,
                    rows.iter().map(|row| row[stream]),
                );
            }
            add_lanes(
                &mut sums,
                &mut corrections,
                chunks.remainder().iter().map(|row| row[stream]),
            );
            merge_lanes(result, sums, corrections)?;
        }
        if length < BLOCK {
            return Ok(result);
        }
    }
}

/// Running compensated sum (Neumaier summation) in the range of `S`.
///
/// The value is `sum + correction`, two finite numbers of type `S`. Each
/// addition adds its rounding error to the correction, which removes the
/// first-order rounding error of plain summation. It is not exact: use [`RangeAccumulator`](crate::RangeAccumulator) for a
/// correctly rounded sum and [`ScaledAccumulator`](crate::ScaledAccumulator)
/// beyond the range of `S`. Subnormal values are allowed. A part that would
/// overflow is an error, and a failed update leaves the sum unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_core::CompensatedSum;
///
/// // The exact sum is 1; plain summation returns 0.
/// let mut sum = CompensatedSum::<f64>::ZERO;
/// for x in [1e16, 1.0, -1e16] {
///     sum.add(x)?;
/// }
/// assert_eq!(sum.total()?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompensatedSum<S: RealScalar> {
    sum: S,
    correction: S,
}

impl<S: RealScalar> CompensatedSum<S> {
    /// The empty sum; both parts are `+0`.
    pub const ZERO: Self = Self {
        sum: S::ZERO,
        correction: S::ZERO,
    };

    /// Creates a sum whose value is `sum + correction`, without rounding.
    ///
    /// # Errors
    ///
    /// [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either part is NaN or
    /// infinite.
    #[inline(always)]
    pub fn from_parts(sum: S, correction: S) -> Result<Self, SciOxideError> {
        validate_finite(sum, "compensated sum part must be finite")?;
        validate_finite(correction, "compensated correction must be finite")?;
        Ok(Self { sum, correction })
    }

    /// Returns `[sum, correction]`; the value of the sum is their exact sum.
    #[inline]
    pub fn parts(self) -> [S; 2] {
        [self.sum, self.correction]
    }

    /// Adds a finite value.
    ///
    /// On error the sum is unchanged.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
    ///   infinite.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   ([`Overflow`](crate::NumericalIssue::Overflow)) if a part overflows.
    #[inline]
    pub fn add(&mut self, value: S) -> Result<(), SciOxideError> {
        validate_finite(value, "compensated addend must be finite")?;
        let sum = self.sum + value;
        if !sum.is_finite() {
            return Err(overflow("compensated sum overflow"));
        }
        let error = if self.sum.abs() >= value.abs() {
            (self.sum - sum) + value
        } else {
            (value - sum) + self.sum
        };
        let correction = self.correction + error;
        if !correction.is_finite() {
            return Err(overflow("compensated correction overflow"));
        }
        *self = Self { sum, correction };
        Ok(())
    }

    /// Adds another compensated sum to this one.
    ///
    /// The other sum's running sum is added first, then its correction, so its
    /// rounding error is not lost before it meets this sum. Merging
    /// [`Self::ZERO`] leaves the value unchanged. On error this sum is unchanged.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if a part overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::CompensatedSum;
    ///
    /// let mut left = CompensatedSum::<f64>::ZERO;
    /// left.add(1e16)?;
    /// let mut right = CompensatedSum::<f64>::ZERO;
    /// for x in [1.0, -1e16] {
    ///     right.add(x)?;
    /// }
    /// left.merge(right)?;
    /// assert_eq!(left.total()?, 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline]
    pub fn merge(&mut self, other: Self) -> Result<(), SciOxideError> {
        let mut merged = *self;
        merged.add(other.sum)?;
        merged.add(other.correction)?;
        *self = merged;
        Ok(())
    }

    /// Returns `sum + correction` rounded to `S`; the result may be subnormal.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if the total is not
    /// finite.
    #[inline]
    pub fn total(self) -> Result<S, SciOxideError> {
        let value = self.sum + self.correction;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(overflow("compensated total overflow"))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    fn check<S: RealScalar>() {
        let mut sum = CompensatedSum::<S>::ZERO;
        for value in [S::MAX, S::ONE, -S::MAX] {
            sum.add(value).unwrap();
        }
        assert_eq!(sum.total().unwrap(), S::ONE);
        let before = sum;
        assert!(sum.add(S::NAN).is_err());
        assert_eq!(sum, before);
        let mut large = CompensatedSum::from_parts(S::MAX, S::ZERO).unwrap();
        let before = large;
        assert!(large.add(S::MAX).is_err());
        assert_eq!(large, before);
        let parts = CompensatedSum::from_parts(S::MAX, S::MAX).unwrap();
        assert!(parts.total().is_err());
    }
    #[test]
    fn compensated_cancellation_and_atomic_errors() {
        check::<f32>();
        check::<f64>();
    }

    fn merges<S: RealScalar>() {
        // Both parts of the merged stream survive: the right correction holds
        // the one that its own leading cancellation exposed.
        let mut left = CompensatedSum::from_parts(S::MAX, S::ZERO).unwrap();
        let mut right = CompensatedSum::<S>::ZERO;
        for value in [S::ONE, -S::MAX] {
            right.add(value).unwrap();
        }
        let [sum, correction] = right.parts();
        let mut sequential = left;
        sequential.add(sum).unwrap();
        sequential.add(correction).unwrap();
        left.merge(right).unwrap();
        assert_eq!(left, sequential);
        assert_eq!(left.total().unwrap(), S::ONE);
        let before = left;
        left.merge(CompensatedSum::ZERO).unwrap();
        assert_eq!(left.total().unwrap(), before.total().unwrap());
        // A failed second-part update rolls back the first part as well.
        let mut large = CompensatedSum::from_parts(S::MAX, S::ZERO).unwrap();
        let before = large;
        let overflowing = CompensatedSum::from_parts(S::ONE, S::MAX).unwrap();
        assert!(large.merge(overflowing).is_err());
        assert_eq!(large, before);
        let mut empty = CompensatedSum::<S>::ZERO;
        assert!(
            empty
                .merge(CompensatedSum::from_parts(S::MAX, S::MAX).unwrap())
                .is_err()
        );
        assert_eq!(empty, CompensatedSum::ZERO);
    }

    #[test]
    fn merges_add_both_parts_transactionally() {
        merges::<f32>();
        merges::<f64>();
    }

    fn blocks<S: RealScalar>() {
        let large = S::try_from_f64_exact((1_u64 << S::MANTISSA_DIGITS) as f64).unwrap();
        for n in [0, 1, 2, 3, 4, 5, 255, 256, 257, 258, 511, 512, 513, 1025] {
            let mut exact = crate::RangeAccumulator::<S>::new();
            let input = (0..n).map(|i| {
                let x = [large, S::ONE, -large][i % 3];
                exact.add(x).unwrap();
                [x, -x]
            });
            let [sum, negative] = compensated_reduce(input).unwrap();
            let expected = exact.finish().unwrap();
            assert_eq!(sum.total().unwrap(), expected);
            assert_eq!(negative.total().unwrap(), -expected);
        }
        let [zero] = compensated_reduce([[-S::ZERO]; 257]).unwrap();
        assert_eq!(zero.parts(), [S::ZERO; 2]);
        assert_eq!(zero.total().unwrap().into_f64().to_bits(), 0);
        for position in [0, 1, 3, 4, 254, 255, 256, 258] {
            for value in [S::NAN, S::INFINITY, -S::INFINITY] {
                let result = compensated_reduce(
                    (0..259).map(|i| [if i == position { value } else { S::ONE }]),
                );
                assert!(matches!(
                    result.unwrap_err().view(),
                    crate::SciOxideErrorView::NonFinite { .. }
                ));
            }
        }
        let result =
            compensated_reduce((0..8).map(|i| [if i % 4 == 0 { S::MAX } else { S::ZERO }]));
        assert!(matches!(
            result.unwrap_err().view(),
            crate::SciOxideErrorView::NumericalInstability { .. }
        ));
        // Exact cancellation at the native endpoints remains admissible.
        assert!(compensated_reduce([[-S::MAX], [S::ZERO], [S::ZERO], [S::ZERO], [S::MAX]]).is_ok());
    }

    #[test]
    fn block_reductions_preserve_residuals_and_check_boundaries() {
        blocks::<f32>();
        blocks::<f64>();
    }

    #[test]
    fn short_lane_initialization_retains_unrounded_parts_and_positive_zero() {
        fn check<S: RealScalar>() {
            let large = S::try_from_f64_exact((1_u64 << S::MANTISSA_DIGITS) as f64).unwrap();
            let [single] = compensated_reduce([[-S::ZERO]]).unwrap();
            assert_eq!(single.parts().map(|v| v.into_f64().to_bits()), [0, 0]);
            let [three] = compensated_reduce([[large], [S::ONE], [-large]]).unwrap();
            assert_eq!(three.parts(), [S::ZERO, S::ONE]);
            let [four] = compensated_reduce([[large], [S::ONE], [-large], [S::ONE]]).unwrap();
            assert_eq!(four.parts(), [S::ONE, S::ONE]);
            let [five] =
                compensated_reduce([[large], [S::ONE], [-large], [S::ONE], [S::ONE]]).unwrap();
            assert_eq!(five.parts(), [S::ONE, S::TWO]);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn lane_finalization_retains_corrections_and_rejects_failed_candidates() {
        fn check<S: RealScalar>() {
            let large = S::try_from_f64_exact((1_u64 << S::MANTISSA_DIGITS) as f64).unwrap();
            let [sum] = finish_compensated_lanes(
                &[[large, -large, S::ZERO, S::ZERO]],
                &[[S::ONE, S::ZERO, S::ZERO, S::ZERO]],
            )
            .unwrap();
            assert_eq!(sum.total().unwrap(), S::ONE);
            assert!(
                finish_compensated_lanes(&[[S::MAX, S::MAX, -S::MAX, S::ZERO]], &[[S::ZERO; 4]])
                    .is_none()
            );
            assert!(
                finish_compensated_lanes(&[[S::ONE; 4], [S::NAN; 4]], &[[S::ZERO; 4]; 2]).is_none()
            );
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn short_storage_preserves_full_block_parts_and_iterator_order() {
        fn check<S: RealScalar>() {
            let large = S::try_from_f64_exact((1_u64 << S::MANTISSA_DIGITS) as f64).unwrap();
            for length in [0, 1, 7, 16, 31, 32, 33, 63, 64, 65, 127, 255] {
                let input: alloc::vec::Vec<_> = (0..length)
                    .map(|i| {
                        let x = [large, S::ONE, -large][i % 3];
                        [x, -x]
                    })
                    .collect();
                let reference = reduce_block::<S, 2, 256>(input.iter().copied()).unwrap();
                let short = compensated_reduce(input.iter().copied()).unwrap();
                let mut iterator = input.iter().copied();
                let unknown = compensated_reduce(core::iter::from_fn(|| iterator.next())).unwrap();
                for ((reference, short), unknown) in reference.into_iter().zip(short).zip(unknown) {
                    let bits = |sum: CompensatedSum<S>| sum.parts().map(|x| x.into_f64().to_bits());
                    assert_eq!(bits(reference), bits(short));
                    assert_eq!(bits(reference), bits(unknown));
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn indexed_short_contractions_preserve_shared_parts_and_reject_breakdown() {
        fn check<S: RealScalar>() {
            let large = S::ONE / S::EPSILON;
            for length in 0..=64 {
                let term = |i: usize| {
                    let value = [large, S::ONE, -large][i % 3];
                    [value, -value]
                };
                let mut visited = 0;
                let actual = compensated_indexed(length, |index| {
                    assert_eq!(index, visited);
                    visited += 1;
                    term(index)
                })
                .unwrap();
                assert_eq!(visited, length);
                let expected = compensated_reduce((0..length).map(term)).unwrap();
                assert_eq!(
                    actual.map(CompensatedSum::parts),
                    expected.map(CompensatedSum::parts)
                );
            }
            for bad in [S::NAN, S::INFINITY, -S::INFINITY] {
                for position in 0..9 {
                    assert!(
                        compensated_indexed(9, |i| [if i == position { bad } else { S::ONE }])
                            .is_none()
                    );
                }
            }
            assert!(compensated_indexed(8, |_| [S::MAX]).is_none());
            assert!(compensated_indexed(65, |_| [S::ONE]).is_none());
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn block_reduction_stops_at_the_first_iterator_end() {
        let mut calls = 0;
        let input = core::iter::from_fn(|| {
            calls += 1;
            if calls == 2 { None } else { Some([1.0_f64]) }
        });
        let [sum] = compensated_reduce(input).unwrap();
        assert_eq!(sum.total().unwrap(), 1.0);
        assert_eq!(calls, 2);
        let result: [CompensatedSum<f64>; 0] = compensated_reduce([[]; 259]).unwrap();
        assert_eq!(result, []);
    }
}
