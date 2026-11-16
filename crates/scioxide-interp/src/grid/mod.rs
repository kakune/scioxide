//! Rectilinear grid interpolation with affine values.

mod bicubic;
mod bilinear;
mod monotone;

pub use bicubic::BicubicInterpolator;
pub use bilinear::BilinearInterpolator;

use alloc::vec::Vec;

use scioxide_core::{Interval, RealScalar, SciOxideError};

use crate::checked::{coordinate, ordinary_nodes};
use crate::protocol::{
    ExtrapolationPolicy, NodeSeparation, domain_from_nodes, find_interval, resolve_coordinate,
};
use crate::validate::validate_nodes;

/// Validated axes of a rectilinear grid, with the row-major sample count and
/// cell location.
#[derive(Debug, Clone)]
pub(super) struct GridAxes<S: RealScalar> {
    pub(super) x: Vec<S>,
    pub(super) y: Vec<S>,
    pub(super) domains: [Interval<S>; 2],
    /// Both axes allow unit-cell coordinates in plain floating-point arithmetic.
    pub(super) native: bool,
}

impl<S: RealScalar> GridAxes<S> {
    /// Validates at least two strictly increasing, separated nodes per axis.
    pub(super) fn new(
        x: Vec<S>,
        y: Vec<S>,
        separation: NodeSeparation<S>,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(&x, 2, separation, scioxide_core::argument!("x"))?;
        validate_nodes(&y, 2, separation, scioxide_core::argument!("y"))?;
        Ok(Self {
            domains: [domain_from_nodes(&x)?, domain_from_nodes(&y)?],
            native: ordinary_nodes(&x) && ordinary_nodes(&y),
            x,
            y,
        })
    }

    /// Requires `len` row-major samples (`ny × nx`) and returns that count.
    pub(super) fn samples(&self, len: usize) -> Result<usize, SciOxideError> {
        let expected = scioxide_core::backend::checked_mul(
            self.x.len(),
            self.y.len(),
            scioxide_core::quantity!("grid_points"),
        )?;
        if len == expected {
            Ok(expected)
        } else {
            Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(self.y.len(), self.x.len()),
                scioxide_core::Dimensions::one(len),
            ))
        }
    }

    /// Resolves a query and returns its cell `(ix, iy)` and unit-cell coordinates.
    #[inline]
    pub(super) fn locate(
        &self,
        point: [S; 2],
        extrapolation: ExtrapolationPolicy<S>,
    ) -> Result<(usize, usize, S, S), SciOxideError> {
        let x = resolve_coordinate(&self.domains[0], point[0], extrapolation)?;
        let y = resolve_coordinate(&self.domains[1], point[1], extrapolation)?;
        let ix = find_interval(&self.x, x);
        let iy = find_interval(&self.y, y);
        Ok((
            ix,
            iy,
            coordinate(x, self.x[ix], self.x[ix + 1], self.native)?,
            coordinate(y, self.y[iy], self.y[iy + 1], self.native)?,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn axes_validate_count_samples_and_locate_cells() {
        let axes = GridAxes::new(
            vec![0.0_f64, 1.0, 3.0],
            vec![0.0, 2.0],
            NodeSeparation::Exact,
        )
        .unwrap();
        assert_eq!(axes.samples(6).unwrap(), 6);
        assert_eq!(
            axes.samples(5).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        let (ix, iy, t, u) = axes
            .locate([2.0, 0.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!((ix, iy, t, u), (1, 0, 0.5, 0.25));
        let (ix, _, t, _) = axes
            .locate([3.0, 2.0], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!((ix, t), (1, 1.0));
        assert_eq!(
            axes.locate([4.0, 0.0], ExtrapolationPolicy::Reject)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DomainError
        );
        assert!(GridAxes::new(vec![0.0_f64], vec![0.0, 1.0], NodeSeparation::Exact).is_err());
        assert!(GridAxes::new(vec![0.0_f64, 1.0], vec![1.0, 0.0], NodeSeparation::Exact).is_err());
    }
}
