//! Points, extents and multi-indices in one to three dimensions.

use scioxide_core::backend::{checked_add, checked_mul, checked_product};
use scioxide_core::{Argument, InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError};

// The length `D` of the array `argument` must be 1..=3.
fn validate_dimension<const D: usize>(argument: Argument) -> Result<(), SciOxideError> {
    if (1..=3).contains(&D) {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "spatial primitives support dimensions one through three",
        ))
    }
}

fn axis_value<const D: usize>(values: &[usize; D], axis: usize) -> Result<usize, SciOxideError> {
    values.get(axis).copied().ok_or_else(|| {
        SciOxideError::out_of_bounds(scioxide_core::quantity!("spatial_axis"), axis, D)
    })
}

/// A point with finite coordinates in one, two or three dimensions.
///
/// ```
/// use scioxide_spatial::Point;
/// let point = Point::new([1.0_f64, 2.0])?;
/// assert!((point.coordinate(1)? - 2.0).abs() < 1e-12);
/// assert!(Point::new([f64::NAN, 0.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Point<S: RealScalar, const D: usize> {
    coordinates: [S; D],
}

impl<S: RealScalar, const D: usize> Point<S, D> {
    /// Creates a point from its coordinates.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is not 1, 2 or 3.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    pub fn new(coordinates: [S; D]) -> Result<Self, SciOxideError> {
        validate_dimension::<D>(scioxide_core::argument!("coordinates"))?;
        if let Some(index) = coordinates
            .iter()
            .position(|coordinate| !coordinate.is_finite())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                alloc::format!("coordinate {index} must be finite"),
            ));
        }
        Ok(Self { coordinates })
    }

    /// Returns the coordinates in axis order.
    pub const fn as_array(&self) -> &[S; D] {
        &self.coordinates
    }

    /// Returns the coordinates in axis order, consuming the point.
    pub const fn into_array(self) -> [S; D] {
        self.coordinates
    }

    /// Returns the coordinate along `axis`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    pub fn coordinate(&self, axis: usize) -> Result<S, SciOxideError> {
        self.coordinates.get(axis).copied().ok_or_else(|| {
            SciOxideError::out_of_bounds(scioxide_core::quantity!("point_coordinate"), axis, D)
        })
    }
}

/// The number of entries along each axis of a one-, two- or three-dimensional
/// index space.
///
/// Positions are numbered with axis 0 fastest: in `[nx, ny, nz]`, index
/// `[x, y, z]` has linear position `x + nx (y + ny z)`. An axis of length zero
/// is valid and makes the extent empty.
///
/// ```
/// use scioxide_spatial::{Extent, Index};
/// let extent = Extent::new([2, 3])?;
/// let index = Index::new([1, 1])?;
/// assert_eq!(extent.element_count()?, 6);
/// assert_eq!(extent.linearize(index)?, 3);
/// assert_eq!(extent.delinearize(3)?, index);
/// assert!(extent.linearize(Index::new([2, 0])?).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct Extent<const D: usize> {
    axes: [usize; D],
}

impl<const D: usize> Extent<D> {
    /// Creates an extent from the length of each axis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is not 1, 2 or 3.
    pub fn new(axes: [usize; D]) -> Result<Self, SciOxideError> {
        validate_dimension::<D>(scioxide_core::argument!("axes"))?;
        Ok(Self { axes })
    }

    /// Returns the axis lengths in axis order.
    pub const fn as_array(&self) -> &[usize; D] {
        &self.axes
    }

    /// Returns the length of one axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    pub fn axis(&self, axis: usize) -> Result<usize, SciOxideError> {
        axis_value(&self.axes, axis)
    }

    /// Returns whether some axis has length zero.
    pub fn is_empty(&self) -> bool {
        self.axes.contains(&0)
    }

    /// Returns the number of entries, the product of the axis lengths.
    ///
    /// An empty extent returns zero, even if the product of the other axes
    /// would overflow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the product of a nonempty extent exceeds `usize`.
    pub fn element_count(&self) -> Result<usize, SciOxideError> {
        checked_product(
            &self.axes,
            scioxide_core::quantity!("spatial_extent_cardinality"),
        )
    }

    /// Returns the linear position of `index`, with axis 0 fastest.
    ///
    /// Only the position of `index` needs to fit in `usize`, not the total
    /// number of entries.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a component of `index` is not less than its axis
    ///   length.
    /// - `SizeOverflow` if the position exceeds `usize`.
    pub fn linearize(&self, index: Index<D>) -> Result<usize, SciOxideError> {
        for axis in 0..D {
            if index.axes[axis] >= self.axes[axis] {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("spatial_index"),
                    index.axes[axis],
                    self.axes[axis],
                ));
            }
        }
        (0..D).rev().try_fold(0, |linear, axis| {
            checked_add(
                checked_mul(
                    linear,
                    self.axes[axis],
                    scioxide_core::quantity!("spatial_linear_index"),
                )?,
                index.axes[axis],
                scioxide_core::quantity!("spatial_linear_index"),
            )
        })
    }

    /// Returns the index at a linear position, the inverse of
    /// [`Self::linearize`].
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the extent is empty or `linear` is not less than the
    ///   number of entries.
    /// - `SizeOverflow` if the number of entries exceeds `usize`.
    pub fn delinearize(&self, linear: usize) -> Result<Index<D>, SciOxideError> {
        let count = self.element_count()?;
        if linear >= count {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("spatial_linear_index"),
                linear,
                count,
            ));
        }
        // `linear < count` means every extent is nonzero, so the mixed-radix
        // digit extraction below never divides by zero.
        let mut remainder = linear;
        let mut axes = [0_usize; D];
        for (coordinate, &extent) in axes.iter_mut().zip(&self.axes) {
            *coordinate = remainder % extent;
            remainder /= extent;
        }
        Ok(Index { axes })
    }
}

/// A multi-index in one, two or three dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct Index<const D: usize> {
    axes: [usize; D],
}

impl<const D: usize> Index<D> {
    /// Creates an index. It is checked against an extent only when used.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is not 1, 2 or 3.
    pub fn new(axes: [usize; D]) -> Result<Self, SciOxideError> {
        validate_dimension::<D>(scioxide_core::argument!("axes"))?;
        Ok(Self { axes })
    }

    pub(crate) const fn from_validated(axes: [usize; D]) -> Self {
        Self { axes }
    }

    /// Returns the components in axis order.
    pub const fn as_array(&self) -> &[usize; D] {
        &self.axes
    }

    /// Returns the component along `axis`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    pub fn axis(&self, axis: usize) -> Result<usize, SciOxideError> {
        axis_value(&self.axes, axis)
    }
}

/// One-dimensional point.
pub type Point1<S> = Point<S, 1>;
/// Two-dimensional point.
pub type Point2<S> = Point<S, 2>;
/// Three-dimensional point.
pub type Point3<S> = Point<S, 3>;
/// One-dimensional extent.
pub type Extent1 = Extent<1>;
/// Two-dimensional extent.
pub type Extent2 = Extent<2>;
/// Three-dimensional extent.
pub type Extent3 = Extent<3>;
/// One-dimensional index.
pub type Index1 = Index<1>;
/// Two-dimensional index.
pub type Index2 = Index<2>;
/// Three-dimensional index.
pub type Index3 = Index<3>;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn linearize_only_requires_the_position_to_fit() {
        let extent = Extent::new([2, usize::MAX]).unwrap();
        assert!(extent.element_count().is_err());
        assert_eq!(extent.linearize(Index::new([1, 0]).unwrap()).unwrap(), 1);
        assert_eq!(
            extent
                .linearize(Index::new([1, usize::MAX / 2]).unwrap())
                .unwrap(),
            usize::MAX
        );
        assert!(
            extent
                .linearize(Index::new([0, usize::MAX - 1]).unwrap())
                .is_err()
        );
        let extent = Extent::new([usize::MAX, 2, 2]).unwrap();
        assert_eq!(extent.linearize(Index::new([1, 0, 0]).unwrap()).unwrap(), 1);
        assert!(extent.linearize(Index::new([0, 0, 1]).unwrap()).is_err());
        assert!(extent.linearize(Index::new([0, 2, 0]).unwrap()).is_err());
    }

    #[test]
    fn point_requires_supported_dimension_and_finite_coordinates() {
        assert!(Point::<f64, 1>::new([0.0]).is_ok());
        assert!(
            matches ! (& (Point :: < f64 , 0 > :: new ([])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
        assert!(
            matches ! (& (Point :: < f32 , 2 > :: new ([0.0 , f32 :: INFINITY])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
    }

    #[test]
    fn zero_extent_short_circuits_unrelated_overflow() {
        let extent = Extent3::new([usize::MAX, 0, usize::MAX]).unwrap();
        assert_eq!(extent.element_count().unwrap(), 0);
        assert!(extent.is_empty());
    }

    #[test]
    fn nonempty_extent_reports_overflow() {
        let extent = Extent2::new([usize::MAX, 2]).unwrap();
        assert!(
            matches!(extent.element_count(), Err(error) if match error.view() { scioxide_core::SciOxideErrorView::SizeOverflow {  context: _view_context } => _view_context == scioxide_core::quantity!("spatial_extent_cardinality"), _ => false })
        );
    }

    #[test]
    fn axis_zero_fast_index_round_trips() {
        let extent = Extent3::new([5, 4, 3]).unwrap();
        for linear in 0..extent.element_count().unwrap() {
            let index = extent.delinearize(linear).unwrap();
            assert_eq!(extent.linearize(index).unwrap(), linear);
        }
        assert_eq!(
            extent.linearize(Index3::new([2, 1, 1]).unwrap()).unwrap(),
            27
        );
    }

    #[test]
    fn index_bounds_are_checked_without_scalar_conversion() {
        let extent = Extent1::new([usize::MAX]).unwrap();
        let index = Index1::new([usize::MAX - 1]).unwrap();
        assert_eq!(extent.linearize(index).unwrap(), usize::MAX - 1);
        assert!(
            matches ! (& (extent . linearize (Index1 :: new ([usize :: MAX]) . unwrap ())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
        );
    }
}
