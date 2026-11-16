//! Resumable summation state.

use super::{SeriesError, SeriesMethod};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{NumericScalar, RangeAccumulator, RealScalar, SciOxideError};

/// One estimate of one component with its error estimates.
#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate<T: NumericScalar> {
    pub(super) value: T,
    /// Change from the previous new candidate.
    pub(super) change: T::Real,
    /// Change estimate before the geometric-tail correction; infinite if unknown.
    pub(super) evidence: T::Real,
    /// Error used to rank estimates and test the tolerance; infinite if unknown.
    pub(super) total: T::Real,
    /// Smallest error the candidate can have: its rounding error or, for a
    /// partial sum, its remainder estimate.
    pub(super) floor: T::Real,
    pub(super) errors: SeriesError<T::Real>,
    /// Whether the candidate exists; the first one has no change.
    pub(super) seen: bool,
}

impl<T: NumericScalar> Candidate<T> {
    pub(super) fn unknown() -> Self {
        Self {
            value: T::ZERO,
            change: T::Real::INFINITY,
            evidence: T::Real::INFINITY,
            total: T::Real::INFINITY,
            floor: T::Real::INFINITY,
            errors: SeriesError::default(),
            seen: false,
        }
    }
}

/// Partial sum and recent terms of a series being summed in several calls to
/// [`SeriesSummation::advance`](crate::SeriesSummation::advance).
///
/// Create a state with [`SeriesSummation::state`](crate::SeriesSummation::state). It
/// holds the exact running sum of the terms added so far, the last `order + 1`
/// terms and partial sums, and the current estimate with its error estimates.
/// Its size does not depend on `max_terms`. A state belongs to one method,
/// first index, order and width. It does not record the provider: resuming it
/// with another series gives a meaningless result.
///
/// For each component, the state keeps the best estimate so far: the one with
/// the smallest error estimate, the newest on ties. For Levin and Wynn, the
/// partial sum also competes, ranked by its remainder estimate. The current
/// estimate is the newest one, unless its rounding error, or for a partial sum
/// its remainder estimate, already exceeds the error estimate of the best
/// estimate; then the best estimate is used. So a clearly less accurate later
/// estimate never replaces a better one, while a slowly improving transform
/// is still followed.
pub struct SeriesState<T: NumericScalar> {
    pub(super) checkpoint: Option<scioxide_core::SolveCheckpointMetadata>,
    pub(super) method: SeriesMethod,
    pub(super) first: usize,
    pub(super) order: usize,
    pub(super) width: usize,
    pub(super) count: usize,
    /// Consecutive steps with a new estimate whose error met the tolerance.
    pub(super) stable: usize,
    /// Ring of the last `order + 1` term rows; row `n` lives at `n % window`.
    pub(super) terms: Vec<T>,
    /// Ring of the matching correctly rounded partial sums.
    pub(super) partials: Vec<T>,
    pub(super) accumulators: Vec<[RangeAccumulator<T::Real>; 2]>,
    pub(super) latest: Vec<Candidate<T>>,
    pub(super) best: Vec<Candidate<T>>,
    /// Current estimates and their error estimates.
    pub(super) estimate: Vec<T>,
    pub(super) errors: Vec<SeriesError<T::Real>>,
}

impl<T: NumericScalar> core::fmt::Debug for SeriesState<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SeriesState")
            .field("method", &self.method)
            .field("first", &self.first)
            .field("order", &self.order)
            .field("width", &self.width)
            .field("terms", &self.count)
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> SeriesState<T> {
    pub(super) fn allocate(
        method: SeriesMethod,
        first: usize,
        order: usize,
        width: usize,
    ) -> Result<Self, SciOxideError> {
        let ring = checked_mul(order + 1, width, scioxide_core::quantity!("series_window"))?;
        Ok(Self {
            checkpoint: None,
            method,
            first,
            order,
            width,
            count: 0,
            stable: 0,
            terms: try_filled(ring, T::ZERO, scioxide_core::quantity!("terms"))?,
            partials: try_filled(ring, T::ZERO, scioxide_core::quantity!("partial_sums"))?,
            accumulators: try_filled(
                width,
                [RangeAccumulator::new(), RangeAccumulator::new()],
                scioxide_core::quantity!("component_accumulators"),
            )?,
            latest: try_filled(
                width,
                Candidate::unknown(),
                scioxide_core::quantity!("latest_candidates"),
            )?,
            best: try_filled(
                width,
                Candidate::unknown(),
                scioxide_core::quantity!("best_candidates"),
            )?,
            estimate: try_filled(width, T::ZERO, scioxide_core::quantity!("estimates"))?,
            errors: try_filled(
                width,
                SeriesError::default(),
                scioxide_core::quantity!("component_errors"),
            )?,
        })
    }

    /// Rows stored in the rings: `order + 1`.
    pub(super) const fn window(&self) -> usize {
        self.order + 1
    }

    /// First scalar of row `row` in the term and partial-sum rings.
    pub(super) const fn offset(&self, row: usize) -> usize {
        (row % self.window()) * self.width
    }

    /// First scalars of `count <= window` consecutive rows from `row`, in row
    /// order, wrapping around the rings with one division in total.
    pub(super) fn offsets(&self, row: usize, count: usize) -> impl Iterator<Item = usize> {
        let window = self.window();
        let width = self.width;
        let first = row % window;
        (first..first + count).map(move |slot| {
            if slot >= window {
                (slot - window) * width
            } else {
                slot * width
            }
        })
    }

    /// Returns the number of terms added to the sum so far.
    pub const fn terms(&self) -> usize {
        self.count
    }

    /// Clears the state, so that the next call starts again from the first
    /// term. The storage is kept.
    pub fn reset(&mut self) {
        self.checkpoint = None;
        self.count = 0;
        self.stable = 0;
        for sum in self.accumulators.iter_mut().flatten() {
            sum.clear();
        }
        self.latest.fill(Candidate::unknown());
        self.estimate.fill(T::ZERO);
        self.errors.fill(SeriesError::default());
        self.best.fill(Candidate::unknown());
    }

    /// Resets the state for another configuration whose window fits the
    /// stored rings; used by `sum_into`.
    pub(super) fn reset_as(&mut self, method: SeriesMethod, first: usize, order: usize) {
        self.reset();
        self.method = method;
        self.first = first;
        self.order = order;
    }

    /// Returns the current estimate of each component, or `None` before the
    /// first term.
    ///
    /// For terms with zero components, it returns an empty slice.
    pub fn value(&self) -> Option<&[T]> {
        (self.count > 0 || self.width == 0).then_some(&self.estimate)
    }

    /// Returns the error estimates of [`value`](Self::value), or `None` before
    /// the first term.
    pub fn errors(&self) -> Option<&[SeriesError<T::Real>]> {
        (self.count > 0 || self.width == 0).then_some(&self.errors)
    }
}
