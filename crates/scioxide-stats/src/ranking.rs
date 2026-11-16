//! Sorting, midranks and tie runs of samples without NaN.
//!
//! Values tie when they are numerically equal, so the two signed zeros tie.
//! Callers reject NaN before ordering. Finite data and the infinite endpoints
//! of CDF queries use the same numerical order.

use crate::Rank;
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{Quantity, RealScalar, SciOxideError};

/// Numerical order of validated (never NaN) values, in which both zeros tie.
#[inline]
pub(crate) fn finite_cmp<S: RealScalar>(left: &S, right: &S) -> Ordering {
    // `partial_cmp` is total on admitted values, and it lowers to branch-free
    // code in sort comparators; the fallback only keeps the order total.
    left.partial_cmp(right).unwrap_or(Ordering::Equal)
}

/// [`finite_cmp`] refined by the sign of zero. Direct selection returns input
/// members, so it orders `-0` before `+0` instead of treating them as a tie.
pub(crate) fn finite_total_cmp<S: RealScalar>(left: &S, right: &S) -> Ordering {
    // Explicit branches keep selection partitions and extreme scans fast.
    if left < right {
        Ordering::Less
    } else if left > right {
        Ordering::Greater
    } else {
        match (left.is_sign_negative(), right.is_sign_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        }
    }
}

/// Sort validated finite values by [`finite_cmp`].
pub(crate) fn sort_finite<S: RealScalar>(values: &mut [S]) {
    values.sort_unstable_by(finite_cmp);
}

/// A maximal run `start..end` of tied values in sorted order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TieRun {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl TieRun {
    /// Number of tied observations.
    pub(crate) const fn len(self) -> usize {
        self.end - self.start
    }
    /// One-based average-rank bounds shared by the run.
    pub(crate) const fn rank(self) -> Rank {
        Rank::from_bounds(self.start + 1, self.end)
    }
    /// Average one-based rank, converted exactly.
    /// # Errors
    /// Rejects a half-integer rank that `f64` cannot represent.
    pub(crate) fn midrank(self) -> Result<f64, SciOxideError> {
        // Doubled ranks below 2^53 are exact; larger ones take the checked path.
        let doubled = self.start + self.end + 1;
        if (doubled as u64) < (1_u64 << 53) {
            Ok(doubled as f64 * 0.5)
        } else {
            self.rank().to_f64()
        }
    }
}

/// Maximal runs of `sorted` whose neighbours satisfy `tied`, with their members.
pub(crate) fn tie_runs<T>(
    sorted: &[T],
    tied: impl FnMut(&T, &T) -> bool,
) -> impl Iterator<Item = (TieRun, &[T])> {
    sorted.chunk_by(tied).scan(0, |start, members| {
        let run = TieRun {
            start: *start,
            end: *start + members.len(),
        };
        *start = run.end;
        Some((run, members))
    })
}

/// Tie-size polynomials of the rank-variance corrections, each summed over runs
/// of length `t`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TieSums {
    /// Σ t(t−1): tied pairs (Kendall τ-b).
    pub(crate) pairs: f64,
    /// Σ t(t−1)(t−2): tied triples (Kendall variance).
    pub(crate) triples: f64,
    /// Σ t(t−1)(2t+5): Kendall and Sen slope variance.
    pub(crate) kendall: f64,
    /// Σ t(t²−1): rank-sum and Kruskal–Wallis corrections.
    pub(crate) cubes: f64,
}

impl TieSums {
    /// Accumulate one run of `length` tied observations.
    pub(crate) fn add(&mut self, length: usize) {
        let t = length as f64;
        self.pairs += t * (t - 1.0);
        self.triples += t * (t - 1.0) * (t - 2.0);
        self.kendall += t * (t - 1.0) * (2.0 * t + 5.0);
        self.cubes += t * (t * t - 1.0);
    }
    /// Sums over the tie runs of validated finite values, sorting them in place.
    pub(crate) fn of_values<S: RealScalar>(values: &mut [S]) -> Self {
        sort_finite(values);
        tie_runs(values, |a, b| a == b)
            .map(|(run, _)| run.len())
            .collect()
    }
}

impl FromIterator<usize> for TieSums {
    /// Accumulate runs of the given tie lengths.
    fn from_iter<I: IntoIterator<Item = usize>>(lengths: I) -> Self {
        lengths
            .into_iter()
            .fold(Self::default(), |mut sums, length| {
                sums.add(length);
                sums
            })
    }
}

/// Labelled finite observations sorted once by [`finite_cmp`]. A label records
/// either sample membership or the original pooled position.
pub(crate) struct RankPool<S, L> {
    entries: Vec<(S, L)>,
}

impl<S: RealScalar> RankPool<S, usize> {
    /// Pool the samples in order, labelling each value by its concatenated position.
    /// # Errors
    /// Returns size overflow or allocation failure.
    pub(crate) fn positions(samples: &[&[S]], quantity: Quantity) -> Result<Self, SciOxideError> {
        Self::pooled(samples, quantity, |_, position| position)
    }
    /// Pool the samples, labelling each value by the index of its sample.
    /// # Errors
    /// Returns size overflow or allocation failure.
    pub(crate) fn samples(samples: &[&[S]], quantity: Quantity) -> Result<Self, SciOxideError> {
        Self::pooled(samples, quantity, |sample, _| sample)
    }
    fn pooled(
        samples: &[&[S]],
        quantity: Quantity,
        label: impl Fn(usize, usize) -> usize,
    ) -> Result<Self, SciOxideError> {
        let total = samples.iter().try_fold(0_usize, |total, sample| {
            total.checked_add(sample.len()).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
        })?;
        let mut entries = try_with_capacity(total, quantity)?;
        for (sample, values) in samples.iter().enumerate() {
            let start = entries.len();
            entries.extend(
                values
                    .iter()
                    .enumerate()
                    .map(|(offset, &value)| (value, label(sample, start + offset))),
            );
        }
        Ok(Self::sorted(entries))
    }
    /// Midranks in pooled input order, and whether any value is tied.
    /// # Errors
    /// Returns allocation failure.
    pub(crate) fn midranks(&self, quantity: Quantity) -> Result<(Vec<f64>, bool), SciOxideError> {
        let mut ranks = try_filled(self.entries.len(), 0.0, quantity)?;
        let mut tied = false;
        for (run, members) in self.runs() {
            tied |= run.len() > 1;
            let rank = run.midrank()?;
            for &(_, position) in members {
                ranks[position] = rank;
            }
        }
        Ok((ranks, tied))
    }
}

impl<S: RealScalar, L> RankPool<S, L> {
    /// Sorts labelled finite observations.
    pub(crate) fn sorted(entries: Vec<(S, L)>) -> Self {
        Self::sorted_by(entries, finite_cmp)
    }
}

impl<S, L> RankPool<S, L> {
    /// Sorts keys, including extended-range values, in numerical order. The
    /// comparator's equality must agree with the keys' tie equality.
    pub(crate) fn sorted_by(
        mut entries: Vec<(S, L)>,
        compare: impl Fn(&S, &S) -> Ordering,
    ) -> Self {
        entries.sort_unstable_by(|a, b| compare(&a.0, &b.0));
        Self { entries }
    }
    /// Labelled observations in sorted order.
    pub(crate) fn entries(&self) -> &[(S, L)] {
        &self.entries
    }
    /// Number of pooled observations.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<S: PartialEq, L> RankPool<S, L> {
    /// Maximal runs of tied values in sorted order, with their members.
    pub(crate) fn runs(&self) -> impl Iterator<Item = (TieRun, &[(S, L)])> {
        tie_runs(&self.entries, |a, b| a.0 == b.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn signed_zeros_tie_and_total_order_refines_them() {
        assert_eq!(finite_cmp(&-0.0_f64, &0.0), Ordering::Equal);
        assert_eq!(finite_total_cmp(&-0.0_f64, &0.0), Ordering::Less);
        assert_eq!(finite_total_cmp(&0.0_f64, &-0.0), Ordering::Greater);
        assert_eq!(finite_total_cmp(&1.0_f64, &-1.0), Ordering::Greater);
        let pool = RankPool::positions(
            &[&[0.0_f64, 1.0], &[-0.0, -1.0]],
            scioxide_core::quantity!("pool"),
        )
        .unwrap();
        let (ranks, tied) = pool.midranks(scioxide_core::quantity!("ranks")).unwrap();
        assert_eq!(ranks, [2.5, 4.0, 2.5, 1.0]);
        assert!(tied);
        let runs: Vec<_> = pool.runs().map(|(run, _)| run).collect();
        assert_eq!(
            runs.iter().map(|run| run.len()).collect::<Vec<_>>(),
            [1, 2, 1]
        );
        assert_eq!(runs[1].rank().bounds(), (2, 3));
    }

    #[test]
    fn tie_sums_match_run_polynomials() {
        let mut values = [2.0_f64, 1.0, 2.0, -0.0, 2.0, 0.0];
        let sums = TieSums::of_values(&mut values);
        // Runs of lengths 2 (both zeros), 1 and 3.
        assert_eq!(sums.pairs, 2.0 + 6.0);
        assert_eq!(sums.triples, 6.0);
        assert_eq!(sums.kendall, 2.0 * 9.0 + 6.0 * 11.0);
        assert_eq!(sums.cubes, 6.0 + 24.0);
    }
}
