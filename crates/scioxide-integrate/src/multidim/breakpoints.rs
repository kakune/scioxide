//! Per-axis breakpoints for adaptive cubature and iterated domains.
//!
//! Breakpoints are fractions strictly inside `(0, 1)` of an axis interval, so one
//! validated partition applies to every interval the axis takes. Rounding can
//! still collapse a mapped piece of a narrow interval. That is a geometry failure
//! of the call, not an input error.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_mul, try_with_capacity};
use scioxide_core::{Interval, RealScalar, SciOxideError};

use crate::geometry::IntervalGeometry;

/// Validated breakpoint fractions of every coordinate axis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AxisBreakpoints<S: RealScalar> {
    axes: Vec<Vec<S>>,
}

impl<S: RealScalar> AxisBreakpoints<S> {
    /// Creates a partition with no breakpoints on `dimension` axes.
    pub(crate) fn new(dimension: usize) -> Result<Self, SciOxideError> {
        let mut axes = try_with_capacity(dimension, scioxide_core::quantity!("axis_breakpoints"))?;
        axes.resize_with(dimension, Vec::new);
        Ok(Self { axes })
    }

    /// Replaces the breakpoints of `axis`.
    ///
    /// # Errors
    ///
    /// An axis outside the dimension, or fractions that are non-finite, outside
    /// `(0, 1)` or not strictly increasing, are `InvalidInput`.
    pub(crate) fn with_axis(mut self, axis: usize, points: &[S]) -> Result<Self, SciOxideError> {
        let detail = "breakpoints must increase strictly inside (0,1) on an existing axis";
        if axis >= self.axes.len() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("axis"),
                scioxide_core::InvalidInputReason::OutOfRange,
                detail,
            ));
        }
        if points
            .iter()
            .any(|&point| !point.is_finite() || point <= S::ZERO || point >= S::ONE)
            || points.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("points"),
                scioxide_core::InvalidInputReason::Inconsistent,
                detail,
            ));
        }
        let mut owned =
            try_with_capacity(points.len(), scioxide_core::quantity!("axis_breakpoints"))?;
        owned.extend_from_slice(points);
        self.axes[axis] = owned;
        Ok(self)
    }

    /// Returns the number of coordinate axes.
    pub(crate) fn dimension(&self) -> usize {
        self.axes.len()
    }

    /// Returns the number of pieces of `axis`: its breakpoints plus one.
    pub(crate) fn pieces(&self, axis: usize) -> usize {
        self.axes[axis].len() + 1
    }

    /// Returns the largest piece count of any axis, at least one.
    pub(crate) fn max_pieces(&self) -> usize {
        self.axes
            .iter()
            .map(|cuts| cuts.len() + 1)
            .max()
            .unwrap_or(1)
    }

    /// Returns the number of cells of the full partition, the product of the
    /// piece counts of all axes.
    pub(crate) fn cells(&self) -> Result<usize, SciOxideError> {
        self.axes.iter().try_fold(1, |cells, cuts| {
            checked_mul(
                cells,
                cuts.len() + 1,
                scioxide_core::quantity!("partition_cells"),
            )
        })
    }

    /// Returns piece `index` of `interval` on `axis`, or `None` when rounding
    /// collapses it. `interval` must have positive width.
    pub(crate) fn piece(
        &self,
        axis: usize,
        interval: Interval<S>,
        index: usize,
    ) -> Result<Option<Interval<S>>, SciOxideError> {
        let cuts = &self.axes[axis];
        let geometry = IntervalGeometry::prepare(interval)?.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("interval"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "breakpoint pieces require a positive-width interval",
            )
        })?;
        let lower = match index {
            0 => interval.lower(),
            _ => geometry.point_at_fraction(cuts[index - 1])?,
        };
        let upper = if index == cuts.len() {
            interval.upper()
        } else {
            geometry.point_at_fraction(cuts[index])?
        };
        if lower >= upper {
            return Ok(None);
        }
        Interval::new(lower, upper).map(Some)
    }

    /// Returns the mixed-radix digit of `cell` on `axis`; axis zero varies fastest.
    pub(crate) fn digit(&self, cell: usize, axis: usize) -> usize {
        let stride = self.axes[..axis]
            .iter()
            .fold(1, |stride, cuts| stride * (cuts.len() + 1));
        (cell / stride) % (self.axes[axis].len() + 1)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn validation_and_pieces_share_one_vocabulary() {
        let breakpoints = AxisBreakpoints::<f64>::new(2).unwrap();
        for points in [&[0.0][..], &[1.0], &[0.3, 0.2], &[0.2, 0.2], &[f64::NAN]] {
            assert!(breakpoints.clone().with_axis(0, points).is_err());
        }
        assert!(breakpoints.clone().with_axis(2, &[]).is_err());
        let breakpoints = breakpoints.with_axis(1, &[0.25, 0.5]).unwrap();
        assert_eq!(
            (breakpoints.cells().unwrap(), breakpoints.max_pieces()),
            (3, 3)
        );
        let unit = Interval::new(0.0, 1.0).unwrap();
        let pieces: Vec<_> = (0..3)
            .map(|index| breakpoints.piece(1, unit, index).unwrap().unwrap())
            .collect();
        assert_eq!(
            pieces,
            [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)].map(|(a, b)| Interval::new(a, b).unwrap())
        );
        assert_eq!(
            (0..3)
                .map(|cell| breakpoints.digit(cell, 1))
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let narrow = Interval::new(1.0, 1.0 + f64::EPSILON).unwrap();
        assert!(breakpoints.piece(1, narrow, 1).unwrap().is_none());
    }
}
