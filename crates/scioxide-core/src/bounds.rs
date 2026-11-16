//! Lower and upper bounds on the variables of a vector problem.

use alloc::vec::Vec;

use crate::size::try_filled;
use crate::{RealScalar, SciOxideError};

/// Lower and upper bounds on each variable of a vector, such as `l ≤ x ≤ u`.
///
/// Each entry is `(lower, upper)` for one variable; `None` means that side is
/// unbounded. Bounds are closed and finite, and there is at least one variable.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Bounds<S: RealScalar> {
    entries: Vec<(Option<S>, Option<S>)>,
}

impl<S: RealScalar> Bounds<S> {
    /// Creates bounds from one `(lower, upper)` pair per variable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `entries` is empty.
    /// - `NonFinite` if a bound is NaN or infinite; use `None` for an unbounded side.
    /// - `InvalidInput` (`Inconsistent`) if a lower bound exceeds its upper bound.
    pub fn new(entries: Vec<(Option<S>, Option<S>)>) -> Result<Self, SciOxideError> {
        Self::validated(entries, crate::argument!("entries"))
    }

    // Shared validation; `argument` names the parameter that supplied the positional entries.
    fn validated(
        entries: Vec<(Option<S>, Option<S>)>,
        argument: crate::Argument,
    ) -> Result<Self, SciOxideError> {
        if entries.is_empty() {
            return Err(SciOxideError::invalid_input(
                argument,
                crate::InvalidInputReason::Empty,
                "bounds must describe at least one variable",
            ));
        }
        entries.iter().try_for_each(|&(lower, upper)| {
            crate::ExtendedInterval::new(lower, upper).map(|_| ())
        })?;
        Ok(Self { entries })
    }

    /// Creates bounds for `dimension` variables from `(index, lower, upper)`
    /// entries; variables without an entry are unbounded.
    ///
    /// Use this form when only a few variables are bounded. Each index may
    /// appear at most once.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `dimension` is zero.
    /// - `OutOfBounds` if an index is not less than `dimension`.
    /// - `InvalidInput` (`Inconsistent`) if an index appears twice.
    /// - `NonFinite` or `InvalidInput` for an invalid pair, as in [`Self::new`].
    /// - `AllocationFailure` if the storage for `dimension` entries cannot be
    ///   reserved.
    pub fn from_indexed(
        dimension: usize,
        indexed: Vec<(usize, Option<S>, Option<S>)>,
    ) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(SciOxideError::invalid_input(
                crate::argument!("dimension"),
                crate::InvalidInputReason::Empty,
                "bounds must describe at least one variable",
            ));
        }

        let mut entries = try_filled(
            dimension,
            (None, None),
            crate::quantity!("indexed_bounds_entries"),
        )?;
        // Duplicates cannot be detected from the entries, because (None, None) is
        // both the default and a legal explicit assignment; a separate flag
        // records each assignment.
        let mut assigned = try_filled(
            dimension,
            false,
            crate::quantity!("indexed_bounds_assignment_flags"),
        )?;
        for (index, lower, upper) in indexed {
            if index >= dimension {
                return Err(SciOxideError::out_of_bounds(
                    crate::quantity!("bounds_variable"),
                    index,
                    dimension,
                ));
            }
            if assigned[index] {
                return Err(SciOxideError::invalid_input(
                    crate::argument!("indexed"),
                    crate::InvalidInputReason::Inconsistent,
                    alloc::format!("bounds for variable {index} were specified more than once"),
                ));
            }
            entries[index] = (lower, upper);
            assigned[index] = true;
        }
        Self::validated(entries, crate::argument!("indexed"))
    }

    /// Returns the number of variables.
    #[expect(
        clippy::len_without_is_empty,
        reason = "validated Bounds always describes at least one variable"
    )]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the `(lower, upper)` pairs, one per variable.
    pub fn as_slice(&self) -> &[(Option<S>, Option<S>)] {
        &self.entries
    }

    /// Replaces all bounds with `entries`, without allocating.
    ///
    /// On error the previous bounds are unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `entries` has a different length.
    /// - `NonFinite` or `InvalidInput` for an invalid pair, as in [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Bounds;
    ///
    /// let mut bounds = Bounds::new(vec![(Some(0.0), None)]).unwrap();
    /// bounds.copy_from_slice(&[(Some(1.0), Some(2.0))]).unwrap();
    /// assert_eq!(bounds.as_slice(), &[(Some(1.0), Some(2.0))]);
    /// ```
    pub fn copy_from_slice(
        &mut self,
        entries: &[(Option<S>, Option<S>)],
    ) -> Result<(), SciOxideError> {
        if entries.len() != self.entries.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.entries.len()],
                [entries.len()],
            ));
        }
        entries.iter().try_for_each(|&(lower, upper)| {
            crate::ExtendedInterval::new(lower, upper).map(|_| ())
        })?;
        self.entries.copy_from_slice(entries);
        Ok(())
    }

    /// Returns whether `lower < upper` for every variable bounded on both sides.
    pub fn has_strict_interior(&self) -> bool {
        self.entries.iter().all(|&(lower, upper)| {
            !matches!((lower, upper), (Some(lower), Some(upper)) if lower >= upper)
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn validates_bounds_and_exposes_entries() {
        let bounds = Bounds::new(vec![(Some(-1.0_f64), Some(1.0)), (None, Some(2.0))]).unwrap();
        assert_eq!(bounds.len(), 2);
        assert!(bounds.has_strict_interior());
        assert_eq!(bounds.as_slice()[1], (None, Some(2.0)));
    }

    #[test]
    fn copying_endpoints_validates_before_publication() {
        let mut bounds = Bounds::new(vec![(Some(0.0), Some(1.0)), (None, None)]).unwrap();
        let original = bounds.clone();
        for invalid in [
            vec![(Some(2.), Some(3.))],
            vec![(Some(2.), Some(3.)), (Some(1.), Some(0.))],
            vec![(Some(2.), Some(3.)), (None, Some(f64::NAN))],
        ] {
            assert!(bounds.copy_from_slice(&invalid).is_err());
            assert_eq!(bounds, original);
        }
        bounds
            .copy_from_slice(&[(Some(2.), Some(2.)), (None, Some(4.))])
            .unwrap();
        assert_eq!(bounds.as_slice(), &[(Some(2.), Some(2.)), (None, Some(4.))]);
    }

    #[test]
    fn rejects_empty_non_finite_and_reversed_bounds() {
        assert!(matches!(
            (Bounds::<f64>::new(Vec::new()))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (Bounds::new(vec![(Some(f64::NAN), None)]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
        assert!(matches!(
            (Bounds::new(vec![(None, Some(f64::INFINITY))]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
        assert!(matches!(
            (Bounds::new(vec![(Some(2.0_f64), Some(1.0))]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn indexed_bounds_fill_unbounded_variables_and_reject_duplicates() {
        let bounds =
            Bounds::from_indexed(3, vec![(0, Some(0.0_f64), None), (2, None, Some(4.0))]).unwrap();
        assert_eq!(
            bounds.as_slice(),
            &[(Some(0.0), None), (None, None), (None, Some(4.0))]
        );

        assert!(matches!(
            (Bounds::from_indexed(2, vec![(1, Some(0.0_f64), None), (1, None, Some(2.0))]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput {
                reason: crate::InvalidInputReason::Inconsistent,
                ..
            }
        ));
        assert!(matches!(
            (Bounds::from_indexed(2, vec![(2, Some(0.0_f64), None)]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::OutOfBounds { .. }
        ));
    }
}
