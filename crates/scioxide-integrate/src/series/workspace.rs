//! Series workspace and its requirements.

use super::double::Wide;
use super::state::Candidate;
use super::{SeriesError, SeriesMethod, SeriesReport, SeriesState, TERM_BATCH};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{AlgebraicScalar, NumericScalar, SciOxideError};

/// Storage requirements of a [`SeriesWorkspace`], from
/// [`SeriesSummation::workspace_requirements`](crate::SeriesSummation::workspace_requirements).
///
/// They depend on the method, first index, width and order, but not on
/// `max_terms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeriesRequirements {
    pub(super) method: SeriesMethod,
    pub(super) first: usize,
    pub(super) width: usize,
    pub(super) order: usize,
}

impl SeriesRequirements {
    /// Returns the number of components of one term.
    pub const fn components(&self) -> usize {
        self.width
    }

    /// Whether storage prepared for `self` serves `requested` without growth:
    /// the same width and an order at least as large. Fails if the method or
    /// first index differ.
    pub(super) fn serves(&self, requested: &Self) -> Result<bool, SciOxideError> {
        if self.method != requested.method || self.first != requested.first {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("requirements"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "series preparation differs",
            ));
        }
        Ok(self.width == requested.width && self.order >= requested.order)
    }
}

/// Scratch buffers of one step. Everything derived from a new term, except its
/// exact sums, is computed here and copied to the state only after the whole
/// step succeeds. History, points and tables have `order + 1` entries.
pub(super) struct SeriesScratch<T: NumericScalar> {
    /// New candidates; `fresh` marks components that have one.
    pub(super) staged: Vec<Candidate<T>>,
    pub(super) fresh: Vec<bool>,
    pub(super) errors: Vec<SeriesError<T::Real>>,
    /// Remainder estimate and ranking error of each new partial sum.
    pub(super) remainders: Vec<SeriesError<T::Real>>,
    pub(super) remainder_totals: Vec<T::Real>,
    pub(super) norm_errors: Vec<T::Real>,
    pub(super) norm_scales: Vec<T::Real>,
    pub(super) history: Vec<T>,
    pub(super) term_history: Vec<T>,
    pub(super) points: Vec<usize>,
    /// Exact sums of the terms that Levin's transform leaves out before each
    /// row it uses, and the magnitudes of its weights.
    pub(super) gaps: Vec<Wide<T>>,
    pub(super) magnitudes: Vec<T::Real>,
    pub(super) tableau_a: Vec<T>,
    pub(super) tableau_b: Vec<T>,
    pub(super) derivative: Vec<T>,
    pub(super) provider_error: Vec<T::Real>,
    pub(super) l1_bound: Vec<T::Real>,
}

/// Attempt records, step scratch and output buffers used by both `sum_into`
/// and `advance`.
pub(super) struct SeriesStorage<T: NumericScalar> {
    pub(super) requirements: SeriesRequirements,
    pub(super) attempts: crate::workspace::Attempts<SeriesReport>,
    pub(super) scratch: SeriesScratch<T>,
    /// At most 32 term rows evaluated ahead under a parallel policy.
    pub(super) batch: Vec<T>,
    pub(super) output: Vec<T>,
    pub(super) errors: Vec<SeriesError<T::Real>>,
}

/// Scratch storage of a [`SeriesSummation`](crate::SeriesSummation), with the record
/// of its last call.
///
/// Create it with [`SeriesSummation::workspace`](crate::SeriesSummation::workspace). It
/// stores `O(width × (order + 32))` values, independent of `max_terms`, and
/// contains its own state for [`SeriesSummation::sum_into`](crate::SeriesSummation::sum_into).
/// A workspace serves any plan with the same method, first index and width and
/// an order no larger than the one it was created for. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct SeriesWorkspace<T: NumericScalar> {
    pub(super) atomic: SeriesState<T>,
    pub(super) storage: SeriesStorage<T>,
}

impl<T: NumericScalar> core::fmt::Debug for SeriesWorkspace<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SeriesWorkspace")
            .field("requirements", &self.storage.requirements)
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> SeriesWorkspace<T> {
    pub(super) fn allocate(r: &SeriesRequirements) -> Result<Self, SciOxideError> {
        let width = r.width;
        let window = r.order + 1;
        let real = |quantity| try_filled(width, T::Real::ZERO, quantity);
        let values = |quantity| try_filled(width, T::ZERO, quantity);
        let history = |quantity| try_filled(window, T::ZERO, quantity);
        Ok(Self {
            atomic: SeriesState::allocate(r.method, r.first, r.order, width)?,
            storage: SeriesStorage {
                requirements: *r,
                attempts: crate::workspace::Attempts::default(),
                scratch: SeriesScratch {
                    staged: try_filled(
                        width,
                        Candidate::unknown(),
                        scioxide_core::quantity!("staged_estimates"),
                    )?,
                    fresh: try_filled(width, false, scioxide_core::quantity!("fresh_components"))?,
                    errors: try_filled(
                        width,
                        SeriesError::default(),
                        scioxide_core::quantity!("staged_errors"),
                    )?,
                    remainders: try_filled(
                        width,
                        SeriesError::default(),
                        scioxide_core::quantity!("remainder_errors"),
                    )?,
                    remainder_totals: real(scioxide_core::quantity!("remainder_totals"))?,
                    norm_errors: real(scioxide_core::quantity!("norm_errors"))?,
                    norm_scales: real(scioxide_core::quantity!("norm_scales"))?,
                    history: history(scioxide_core::quantity!("partial_history"))?,
                    term_history: history(scioxide_core::quantity!("term_history"))?,
                    points: try_filled(window, 0, scioxide_core::quantity!("term_points"))?,
                    gaps: try_filled(window, Wide::ZERO, scioxide_core::quantity!("omitted_sums"))?,
                    magnitudes: try_filled(
                        window,
                        T::Real::ZERO,
                        scioxide_core::quantity!("levin_weights"),
                    )?,
                    tableau_a: history(scioxide_core::quantity!("acceleration_tableau"))?,
                    tableau_b: history(scioxide_core::quantity!("acceleration_tableau"))?,
                    derivative: values(scioxide_core::quantity!("derivatives"))?,
                    provider_error: real(scioxide_core::quantity!("provider_errors"))?,
                    l1_bound: real(scioxide_core::quantity!("l1_bounds"))?,
                },
                batch: try_filled(
                    checked_mul(TERM_BATCH, width, scioxide_core::quantity!("term_batch"))?,
                    T::ZERO,
                    scioxide_core::quantity!("term_batch"),
                )?,
                output: values(scioxide_core::quantity!("series_output"))?,
                errors: try_filled(
                    width,
                    SeriesError::default(),
                    scioxide_core::quantity!("series_errors"),
                )?,
            },
        })
    }

    /// Returns whether the workspace serves the requirements `r` without
    /// reallocation.
    ///
    /// It is `Sufficient` for the same width and an order no larger than the
    /// one the workspace was created for. Another width or a larger order
    /// needs growth.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `r` is for another method or first index.
    pub fn check(&self, r: &SeriesRequirements) -> Result<scioxide_core::Capacity, SciOxideError> {
        Ok(if self.storage.requirements.serves(r)? {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }

    /// Reallocates the workspace, if needed, so that it serves `r`.
    ///
    /// The new workspace keeps the larger of the two orders and the record of
    /// the last call. On error, the workspace is unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `r` is for another method or first index.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn try_reserve(&mut self, r: &SeriesRequirements) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::NeedsGrowth {
            let grown = SeriesRequirements {
                order: r.order.max(self.storage.requirements.order),
                ..*r
            };
            let mut grown = Self::allocate(&grown)?;
            core::mem::swap(&mut self.storage.attempts, &mut grown.storage.attempts);
            *self = grown;
        }
        Ok(())
    }

    /// Returns a record of the last call with this workspace: its outcome, its
    /// evaluation counts and its report, or `None` before the first call.
    pub fn last_attempt(&self) -> Option<scioxide_core::AttemptRecord<SeriesReport>> {
        self.storage.attempts.last_attempt()
    }

    /// Returns the error estimates of the last estimate computed by
    /// [`SeriesSummation::sum_into`](crate::SeriesSummation::sum_into) or
    /// [`SeriesSummation::advance`](crate::SeriesSummation::advance) with this workspace.
    pub fn component_errors(&self) -> &[SeriesError<T::Real>] {
        &self.storage.errors
    }
}
