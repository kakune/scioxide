//! Coordinate axes and structured grids in one to three dimensions.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::array;

use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{
    Argument, DomainReason, InvalidInputReason, NonFiniteOrigin, RangeAccumulator, RealScalar,
    SciOxideError, checked_affine, checked_affine_ratio, checked_difference,
    checked_float_quotient,
};

use crate::{Extent, Index, MeshToken, Point};

/// Whether the coordinates of an axis increase or decrease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum AxisOrientation {
    /// Coordinates increase from the first site to the last.
    Increasing,
    /// Coordinates decrease from the first site to the last.
    Decreasing,
}

/// Whether a grid axis has two boundaries or wraps around.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum AxisTopology {
    /// The first and last sites are two separate boundaries.
    Bounded,
    /// The first and last sites are identified across a seam, with period
    /// `|end − start|`. Both stay separate vertices, paired with each other.
    Periodic,
}

/// One end of a coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum AxisSide {
    /// The first site, whether it is the smallest or largest coordinate.
    Start,
    /// The last site, whether it is the smallest or largest coordinate.
    End,
}

#[derive(Debug, Clone)]
enum AxisStorage<S> {
    Uniform { start: S, end: S, count: usize },
    Explicit(Vec<S>),
}

/// The sites of one grid axis: finite and strictly increasing or strictly
/// decreasing.
///
/// A uniform axis stores only its endpoints and number of sites; an explicit
/// axis stores every coordinate. Site access is O(1), and [`Self::locate`]
/// finds the interval of a coordinate by bisection in O(log n).
///
/// ```
/// use scioxide_spatial::{AxisOrientation, CoordinateAxis};
/// let axis = CoordinateAxis::explicit(vec![2.0_f64, 1.0, -1.0])?;
/// assert_eq!(axis.orientation(), AxisOrientation::Decreasing);
/// assert_eq!(axis.locate(0.0)?.cell(), 1);
/// assert!(CoordinateAxis::explicit(vec![0.0_f64, 0.0]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct CoordinateAxis<S: RealScalar> {
    storage: AxisStorage<S>,
    orientation: AxisOrientation,
}

impl<S: RealScalar> CoordinateAxis<S> {
    /// Creates an axis of `count` equally spaced sites from `start` to `end`.
    ///
    /// Site `i` is `start + (end − start) i / (count − 1)`, correctly rounded,
    /// and the endpoints are returned exactly as given. To guarantee that the
    /// rounded sites are strictly monotonic, the exact spacing must be at least
    /// `u`, the distance from the larger-magnitude endpoint to the next float
    /// toward zero: `|end − start| ≥ (count − 1) u`. The check costs O(1)
    /// for any `count`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an endpoint is NaN or infinite.
    /// - `InvalidInput` if the endpoints are equal, `count < 2`, or the
    ///   spacing is below `u`.
    pub fn uniform(start: S, end: S, count: usize) -> Result<Self, SciOxideError> {
        validate_endpoints(start, end, count)?;
        let orientation = orientation(start, end);
        let axis = Self {
            storage: AxisStorage::Uniform { start, end, count },
            orientation,
        };
        let magnitude = start.abs().max(end.abs());
        let ulp = magnitude - magnitude.next_after(S::ZERO);
        let mut spacing = RangeAccumulator::new();
        spacing.add(start.max(end))?;
        spacing.add(-start.min(end))?;
        // Split the integer into exact 16-bit digits. Triple products retain
        // the integer weight without overflowing or rounding it to S.
        for shift in (0..usize::BITS).step_by(16) {
            let digit = ((count - 1) >> shift) & 0xffff;
            spacing.subtract_triple_product(
                ulp,
                S::try_from_usize_exact(digit)?,
                S::ONE.scalbn(shift as i32),
            )?;
        }
        if spacing.sign().is_lt() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("count"),
                InvalidInputReason::Inconsistent,
                "uniform spacing must be at least the inward endpoint ulp",
            ));
        }
        Ok(axis)
    }

    /// Creates an axis from its coordinates, which must be strictly increasing
    /// or strictly decreasing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two sites, two equal sites, or
    ///   a change of direction.
    /// - `NonFinite` if a coordinate is NaN or infinite.
    pub fn explicit(coordinates: Vec<S>) -> Result<Self, SciOxideError> {
        const COORDINATES: Argument = scioxide_core::argument!("coordinates");
        if coordinates.len() < 2 {
            return Err(SciOxideError::invalid_input(
                COORDINATES,
                InvalidInputReason::OutOfRange,
                "a coordinate axis requires at least two sites",
            ));
        }
        if let Some(index) = coordinates.iter().position(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                alloc::format!("coordinate {index} must be finite"),
            ));
        }
        let orientation = orientation(coordinates[0], coordinates[1]);
        if coordinates[0] == coordinates[1]
            || coordinates
                .windows(2)
                .any(|pair| !strictly_after(orientation, pair[1], pair[0]))
        {
            return Err(SciOxideError::invalid_input(
                COORDINATES,
                InvalidInputReason::Inconsistent,
                "axis coordinates must be strictly monotonic without duplicates",
            ));
        }
        Ok(Self {
            storage: AxisStorage::Explicit(coordinates),
            orientation,
        })
    }

    /// Returns the number of sites, including both endpoints.
    pub fn len(&self) -> usize {
        match &self.storage {
            AxisStorage::Uniform { count, .. } => *count,
            AxisStorage::Explicit(coordinates) => coordinates.len(),
        }
    }

    /// Returns `false`: an axis has at least two sites.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns whether the coordinates increase or decrease.
    pub const fn orientation(&self) -> AxisOrientation {
        self.orientation
    }

    /// Returns the first site.
    pub fn start(&self) -> S {
        match &self.storage {
            AxisStorage::Uniform { start, .. } => *start,
            AxisStorage::Explicit(coordinates) => coordinates[0],
        }
    }

    /// Returns the last site.
    pub fn end(&self) -> S {
        match &self.storage {
            AxisStorage::Uniform { end, .. } => *end,
            AxisStorage::Explicit(coordinates) => coordinates[coordinates.len() - 1],
        }
    }

    /// Returns site `index`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.len()`.
    pub fn coordinate(&self, index: usize) -> Result<S, SciOxideError> {
        if index >= self.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("coordinate_axis"),
                index,
                self.len(),
            ));
        }
        self.coordinate_unchecked(index)
    }

    /// Returns an iterator over all sites, computing uniform sites on the fly.
    /// Each item is the `Result` of [`Self::coordinate`].
    pub fn coordinates(
        &self,
    ) -> impl ExactSizeIterator<Item = Result<S, SciOxideError>> + Clone + core::fmt::Debug + '_
    {
        (0..self.len()).map(|index| self.coordinate_unchecked(index))
    }

    /// Finds the interval between consecutive sites that contains `coordinate`.
    ///
    /// Intervals are half-open in site order: a site belongs to the interval
    /// that starts there. The last site belongs to the last interval, with
    /// fraction one.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `coordinate` is NaN or infinite.
    /// - `DomainError` if `coordinate` lies outside the closed span of the
    ///   axis.
    pub fn locate(&self, coordinate: S) -> Result<IntervalLocation<S>, SciOxideError> {
        validate_query(self, coordinate, scioxide_core::argument!("coordinate"))?;
        // Intervals are half-open [site, next). Only the closing endpoint
        // needs a special case, which maps it into the last real interval.
        let last = self.len() - 1;
        if coordinate == self.end() {
            return Ok(IntervalLocation {
                cell: last - 1,
                fraction: S::ONE,
            });
        }

        let cell = match &self.storage {
            AxisStorage::Explicit(coordinates) => match self.orientation {
                AxisOrientation::Increasing => coordinates
                    .partition_point(|candidate| *candidate <= coordinate)
                    .saturating_sub(1)
                    .min(last - 1),
                AxisOrientation::Decreasing => coordinates
                    .partition_point(|candidate| *candidate >= coordinate)
                    .saturating_sub(1)
                    .min(last - 1),
            },
            AxisStorage::Uniform { .. } => self.locate_uniform(coordinate)?,
        };
        let left = self.coordinate_unchecked(cell)?;
        let right = self.coordinate_unchecked(cell + 1)?;
        Ok(IntervalLocation {
            cell,
            fraction: interval_fraction(left, right, coordinate)?,
        })
    }

    // Bisect over the evaluated (rounded) sites rather than computing
    // floor((x - start) / h). A closed-form index can disagree with
    // `coordinate_unchecked` by one near a site, while bisection keeps
    // coordinate(lower) <= x < coordinate(upper), in axis order, for the
    // exact sites callers observe. That preserves the site round trip.
    fn locate_uniform(&self, coordinate: S) -> Result<usize, SciOxideError> {
        let mut lower = 0_usize;
        let mut upper = self.len() - 1;
        while upper - lower > 1 {
            let midpoint = lower + (upper - lower) / 2;
            let candidate = self.coordinate_unchecked(midpoint)?;
            let begins_after = match self.orientation {
                AxisOrientation::Increasing => candidate > coordinate,
                AxisOrientation::Decreasing => candidate < coordinate,
            };
            if begins_after {
                upper = midpoint;
            } else {
                lower = midpoint;
            }
        }
        Ok(lower)
    }

    // Site `index`; affine evaluation
    fn coordinate_unchecked(&self, index: usize) -> Result<S, SciOxideError> {
        match &self.storage {
            AxisStorage::Explicit(coordinates) => Ok(coordinates[index]),
            // Endpoints are returned as stored. Affine evaluation at t = 1
            // need not reproduce `end` bit for bit, and seams and boundaries
            // rely on exact endpoint identity.
            AxisStorage::Uniform { start, end, count } => {
                if index == 0 {
                    return Ok(*start);
                }
                if index + 1 == *count {
                    return Ok(*end);
                }
                checked_affine_ratio(*start, *end, index, count - 1)
            }
        }
    }

    // Axis refinement for the `factors` argument.
    fn refined(&self, factor: usize) -> Result<Self, SciOxideError> {
        if factor == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("factors"),
                InvalidInputReason::OutOfRange,
                "refinement factors must be positive",
            ));
        }
        if factor == 1 {
            return Ok(self.clone());
        }
        let refined_intervals = checked_mul(
            self.len() - 1,
            factor,
            scioxide_core::quantity!("refined_axis_intervals"),
        )?;
        let refined_count = checked_add(
            refined_intervals,
            1,
            scioxide_core::quantity!("refined_axis_site_count"),
        )?;
        match &self.storage {
            AxisStorage::Uniform { start, end, .. } => Self::uniform(*start, *end, refined_count),
            AxisStorage::Explicit(coordinates) => {
                // Original sites are copied verbatim, so the refined axis
                // nests the coarse one exactly. `explicit` re-validates
                // that the rounded subdivision sites stay strictly monotonic.
                let mut refined = try_with_capacity(
                    refined_count,
                    scioxide_core::quantity!("refined_axis_coordinates"),
                )?;
                refined.push(coordinates[0]);
                for pair in coordinates.windows(2) {
                    for subinterval in 1..factor {
                        refined.push(checked_affine_ratio(pair[0], pair[1], subinterval, factor)?);
                    }
                    refined.push(pair[1]);
                }
                Self::explicit(refined)
            }
        }
    }
}

/// An interval of an axis and the relative position in it, as returned by
/// [`CoordinateAxis::locate`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct IntervalLocation<S: RealScalar> {
    cell: usize,
    fraction: S,
}

impl<S: RealScalar> IntervalLocation<S> {
    /// Returns the interval index `i`, for the interval from site `i` to site
    /// `i + 1`.
    pub const fn cell(self) -> usize {
        self.cell
    }

    /// Returns `(x − xᵢ) / (xᵢ₊₁ − xᵢ)` for the located coordinate `x`. It lies
    /// in `[0, 1)`, except at the last site of the axis, where it is exactly
    /// one.
    pub const fn fraction(self) -> S {
        self.fraction
    }
}

#[derive(Debug)]
struct StructuredGridStorage<S: RealScalar, const D: usize> {
    axes: [CoordinateAxis<S>; D],
    topology: [AxisTopology; D],
    vertex_extent: Extent<D>,
    cell_extent: Extent<D>,
    token: MeshToken,
}

/// Tensor-product grid in one, two or three dimensions.
///
/// The vertices are all combinations of the axis sites, numbered with axis 0
/// fastest; cells are numbered the same way. Each axis is bounded or periodic.
/// Cloning shares the storage and the [`MeshToken`] in O(1).
///
/// ```
/// use scioxide_spatial::{AxisTopology, CoordinateAxis, Grid2d, Index, Point};
/// let axis = CoordinateAxis::uniform(0.0_f64, 1.0, 3)?;
/// let grid = Grid2d::new([axis.clone(), axis], [AxisTopology::Bounded; 2])?;
/// assert_eq!(
///     grid.locate(Point::new([0.25, 0.75])?)?.cell(),
///     Index::new([0, 1])?
/// );
/// assert!(grid.locate(Point::new([2.0, 0.0])?).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct StructuredGrid<S: RealScalar, const D: usize> {
    storage: Arc<StructuredGridStorage<S, D>>,
}

impl<S: RealScalar, const D: usize> StructuredGrid<S, D> {
    /// Creates a bounded grid with `counts[i]` uniformly spaced sites from `start[i]` to `end[i]`.
    /// Builds axes in order in `O(D)` time with compact axis storage using [`CoordinateAxis::uniform`].
    ///
    /// # Errors
    /// As [`CoordinateAxis::uniform`] for each axis, then [`Self::new`].
    ///
    /// ```
    /// let grid = scioxide_spatial::Grid2d::uniform([0.0_f64; 2], [1.0, 2.0], [11, 21])?;
    /// assert_eq!(grid.vertex_extent().as_array(), &[11, 21]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn uniform(start: [S; D], end: [S; D], counts: [usize; D]) -> Result<Self, SciOxideError> {
        let mut axes = [const { None }; D];
        for (i, axis) in axes.iter_mut().enumerate() {
            *axis = Some(CoordinateAxis::uniform(start[i], end[i], counts[i])?);
        }
        // Every slot was initialized in order; collect through a fixed array without heap storage.
        let axes = axes.map(|axis| match axis {
            Some(axis) => axis,
            None => unreachable!(),
        });
        Self::new(axes, [AxisTopology::Bounded; D])
    }

    /// Creates a grid from one axis and one [`AxisTopology`] per dimension.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is not 1, 2 or 3.
    /// - `SizeOverflow` if the number of vertices or cells exceeds `usize`.
    pub fn new(
        axes: [CoordinateAxis<S>; D],
        topology: [AxisTopology; D],
    ) -> Result<Self, SciOxideError> {
        Self::from_parts(axes, topology, MeshToken::fresh())
    }

    fn from_parts(
        axes: [CoordinateAxis<S>; D],
        topology: [AxisTopology; D],
        token: MeshToken,
    ) -> Result<Self, SciOxideError> {
        let vertex_extent = Extent::new(array::from_fn(|axis| axes[axis].len()))?;
        let cell_extent = Extent::new(array::from_fn(|axis| axes[axis].len() - 1))?;
        // Cardinalities are checked once here, so later linearization and
        // field allocation can rely on representable counts.
        vertex_extent.element_count()?;
        cell_extent.element_count()?;
        Ok(Self {
            storage: Arc::new(StructuredGridStorage {
                axes,
                topology,
                vertex_extent,
                cell_extent,
                token,
            }),
        })
    }

    /// Returns the mesh token of this grid revision.
    pub fn mesh_token(&self) -> &MeshToken {
        &self.storage.token
    }

    /// Returns the number of vertices along each axis.
    pub fn vertex_extent(&self) -> Extent<D> {
        self.storage.vertex_extent
    }

    /// Returns the number of cells along each axis, one less than the number
    /// of vertices.
    pub fn cell_extent(&self) -> Extent<D> {
        self.storage.cell_extent
    }

    /// Returns one coordinate axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_spatial::{AxisTopology, CoordinateAxis, Grid1d};
    ///
    /// let grid = Grid1d::new(
    ///     [CoordinateAxis::uniform(0.0_f64, 1.0, 5)?],
    ///     [AxisTopology::Bounded],
    /// )?;
    /// let axis: &CoordinateAxis<f64> = grid.axis(0)?;
    /// assert_eq!(axis.len(), 5);
    /// assert_eq!(axis.coordinate(2)?, 0.5);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn axis(&self, axis: usize) -> Result<&CoordinateAxis<S>, SciOxideError> {
        self.storage.axes.get(axis).ok_or_else(|| {
            SciOxideError::out_of_bounds(scioxide_core::quantity!("structured_grid_axis"), axis, D)
        })
    }

    /// Returns whether one axis is bounded or periodic.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    pub fn axis_topology(&self, axis: usize) -> Result<AxisTopology, SciOxideError> {
        self.storage.topology.get(axis).copied().ok_or_else(|| {
            SciOxideError::out_of_bounds(scioxide_core::quantity!("structured_grid_axis"), axis, D)
        })
    }

    /// Returns the coordinates of a vertex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a component of `index` is outside the vertex extent.
    pub fn point(&self, index: Index<D>) -> Result<Point<S, D>, SciOxideError> {
        self.storage.vertex_extent.linearize(index)?;
        let mut coordinates = [S::ZERO; D];
        for (axis, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate = self.storage.axes[axis].coordinate_unchecked(index.as_array()[axis])?;
        }
        Point::new(coordinates)
    }

    /// Finds the cell that contains `point` and the relative position in it,
    /// applying [`CoordinateAxis::locate`] on each axis.
    ///
    /// Periodic axes are not wrapped; use [`Self::locate_wrapped`] for that.
    ///
    /// # Errors
    ///
    /// - `DomainError` if a coordinate lies outside the span of its axis.
    pub fn locate(&self, point: Point<S, D>) -> Result<CellLocation<S, D>, SciOxideError> {
        let mut cells = [0_usize; D];
        let mut fractions = [S::ZERO; D];
        for axis in 0..D {
            let location = self.storage.axes[axis].locate(point.as_array()[axis])?;
            cells[axis] = location.cell();
            fractions[axis] = location.fraction();
        }
        Ok(CellLocation {
            cell: Index::new(cells)?,
            fractions,
        })
    }

    /// Maps each periodic coordinate into the span of its axis, modulo the
    /// period.
    ///
    /// Coordinates already in the closed span, including both seam endpoints,
    /// are returned unchanged.
    ///
    /// # Errors
    ///
    /// - `DomainError` if a coordinate on a bounded axis lies outside its span.
    /// - `NonFinite` or `NumericalInstability` if wrapping does not give a
    ///   finite coordinate.
    pub fn wrap_point(&self, point: Point<S, D>) -> Result<Point<S, D>, SciOxideError> {
        let mut coordinates = [S::ZERO; D];
        for (axis, wrapped) in coordinates.iter_mut().enumerate() {
            let coordinate = point.as_array()[axis];
            *wrapped = match self.storage.topology[axis] {
                AxisTopology::Bounded => {
                    validate_query(
                        &self.storage.axes[axis],
                        coordinate,
                        scioxide_core::argument!("point"),
                    )?;
                    coordinate
                }
                AxisTopology::Periodic => wrap_coordinate(&self.storage.axes[axis], coordinate)?,
            };
        }
        Point::new(coordinates)
    }

    /// Wraps periodic coordinates with [`Self::wrap_point`], then calls
    /// [`Self::locate`].
    ///
    /// # Errors
    ///
    /// As [`Self::wrap_point`] and [`Self::locate`].
    pub fn locate_wrapped(&self, point: Point<S, D>) -> Result<CellLocation<S, D>, SciOxideError> {
        self.locate(self.wrap_point(point)?)
    }

    /// Returns the cells that share a face with `cell`, ordered by axis and,
    /// within an axis, the start side before the end side.
    ///
    /// On a periodic axis the first and last cells are neighbors. A periodic
    /// axis with a single cell does not make the cell its own neighbor.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell` is outside the cell extent.
    pub fn cell_neighbors(
        &self,
        cell: Index<D>,
    ) -> Result<
        impl ExactSizeIterator<Item = Index<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        self.storage.cell_extent.linearize(cell)?;
        // At most two neighbors per axis. With two periodic cells, the
        // wrapped lower and upper neighbor are the same cell, so
        // shared index storage deduplicates. `extent > 1` removes the
        // self-neighbor on a single-cell periodic axis.
        let mut neighbors = crate::indices::DistinctIndices::<D, 6>::new();
        for axis in 0..D {
            let extent = self.storage.cell_extent.as_array()[axis];
            if cell.as_array()[axis] > 0 {
                let mut candidate = *cell.as_array();
                candidate[axis] -= 1;
                neighbors.push(Index::from_validated(candidate));
            } else if self.storage.topology[axis] == AxisTopology::Periodic && extent > 1 {
                let mut candidate = *cell.as_array();
                candidate[axis] = extent - 1;
                neighbors.push(Index::from_validated(candidate));
            }
            if cell.as_array()[axis] + 1 < extent {
                let mut candidate = *cell.as_array();
                candidate[axis] += 1;
                neighbors.push(Index::from_validated(candidate));
            } else if self.storage.topology[axis] == AxisTopology::Periodic && extent > 1 {
                let mut candidate = *cell.as_array();
                candidate[axis] = 0;
                neighbors.push(Index::from_validated(candidate));
            }
        }
        Ok(neighbors.into_iter())
    }

    /// Returns `Decreasing` if an odd number of axes decrease, so that the
    /// grid has the opposite orientation of the standard basis, and
    /// `Increasing` otherwise.
    pub fn orientation(&self) -> AxisOrientation {
        let reversals = self
            .storage
            .axes
            .iter()
            .filter(|axis| axis.orientation() == AxisOrientation::Decreasing)
            .count();
        if reversals % 2 == 0 {
            AxisOrientation::Increasing
        } else {
            AxisOrientation::Decreasing
        }
    }

    /// Returns a grid with every interval of axis `i` split into `factors[i]`
    /// equal parts.
    ///
    /// The new grid keeps the mesh family and has the next generation. If every
    /// factor is one, the grid is returned unchanged with the same token.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a factor is zero, or if the refined sites would not
    ///   be strictly monotonic after rounding.
    /// - `SizeOverflow` if the new size or generation exceeds its type.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn refined(&self, factors: [usize; D]) -> Result<Self, SciOxideError> {
        if factors.contains(&0) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("factors"),
                InvalidInputReason::OutOfRange,
                "refinement factors must be positive",
            ));
        }
        if factors.iter().all(|factor| *factor == 1) {
            return Ok(self.clone());
        }
        let token = self.storage.token.next_generation()?;
        let mut axes = self.storage.axes.clone();
        for axis in 0..D {
            axes[axis] = self.storage.axes[axis].refined(factors[axis])?;
        }
        Self::from_parts(axes, self.storage.topology, token)
    }

    /// Returns a grid with one axis replaced. The new grid keeps the mesh
    /// family and has the next generation.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis >= D`.
    /// - `SizeOverflow` as for [`Self::new`], or if the generation overflows.
    pub fn with_axis(
        &self,
        axis: usize,
        replacement: CoordinateAxis<S>,
    ) -> Result<Self, SciOxideError> {
        if axis >= D {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_grid_axis"),
                axis,
                D,
            ));
        }
        let token = self.storage.token.next_generation()?;
        let axes = array::from_fn(|candidate| {
            if candidate == axis {
                replacement.clone()
            } else {
                self.storage.axes[candidate].clone()
            }
        });
        Self::from_parts(axes, self.storage.topology, token)
    }
}

/// A grid cell and the relative position in it along each axis, as returned
/// by [`StructuredGrid::locate`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CellLocation<S: RealScalar, const D: usize> {
    cell: Index<D>,
    fractions: [S; D],
}

impl<S: RealScalar, const D: usize> CellLocation<S, D> {
    /// Returns the cell index.
    pub const fn cell(self) -> Index<D> {
        self.cell
    }

    /// Returns the relative position along each axis, as defined by
    /// [`IntervalLocation::fraction`].
    pub const fn fractions(self) -> [S; D] {
        self.fractions
    }
}

/// One-dimensional structured grid.
pub type Grid1d<S> = StructuredGrid<S, 1>;
/// Two-dimensional structured grid.
pub type Grid2d<S> = StructuredGrid<S, 2>;
/// Three-dimensional structured grid.
pub type Grid3d<S> = StructuredGrid<S, 3>;

// Validates the `start`, `end` and `count` arguments.
fn validate_endpoints<S: RealScalar>(start: S, end: S, count: usize) -> Result<(), SciOxideError> {
    if !start.is_finite() || !end.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "uniform axis endpoints must be finite",
        ));
    }
    if start == end {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("end"),
            InvalidInputReason::Inconsistent,
            "uniform axis endpoints must be distinct",
        ));
    }
    if count < 2 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("count"),
            InvalidInputReason::OutOfRange,
            "a coordinate axis requires at least two sites",
        ));
    }
    Ok(())
}

fn orientation<S: RealScalar>(start: S, end: S) -> AxisOrientation {
    if end > start {
        AxisOrientation::Increasing
    } else {
        AxisOrientation::Decreasing
    }
}

fn strictly_after<S: RealScalar>(orientation: AxisOrientation, next: S, previous: S) -> bool {
    match orientation {
        AxisOrientation::Increasing => next > previous,
        AxisOrientation::Decreasing => next < previous,
    }
}

// Validates a query `coordinate` from `argument`.
fn validate_query<S: RealScalar>(
    axis: &CoordinateAxis<S>,
    coordinate: S,
    argument: Argument,
) -> Result<(), SciOxideError> {
    if !coordinate.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "query coordinate must be finite",
        ));
    }
    let outside = match axis.orientation {
        AxisOrientation::Increasing => coordinate < axis.start() || coordinate > axis.end(),
        AxisOrientation::Decreasing => coordinate > axis.start() || coordinate < axis.end(),
    };
    if outside {
        return Err(SciOxideError::domain_error(
            argument,
            DomainReason::OutsideDomain,
            "query coordinate lies outside the closed axis span",
        ));
    }
    Ok(())
}

fn interval_fraction<S: RealScalar>(left: S, right: S, value: S) -> Result<S, SciOxideError> {
    if value == left {
        return Ok(S::ZERO);
    }
    if value == right {
        return Ok(S::ONE);
    }
    // Exact endpoint matches return exact 0 and 1. The direct quotient is
    // used whenever the width is finite. If the width overflows (for
    // example, sites near -MAX and MAX), every operand is first divided by a
    // common scale >= 1 so the differences fit; the ratio is scale-invariant.
    // Intervals are half-open: a rounded quotient of one for an interior value
    // is clamped to the predecessor of one, which only fraction one marks as
    // the closing endpoint.
    let below_one = S::ONE - S::EPSILON * S::HALF;
    if let Ok(width) = checked_difference(right, left) {
        return Ok(checked_float_quotient(checked_difference(value, left)?, width)?.min(below_one));
    }
    let scale = left.abs().max(right.abs()).max(value.abs()).max(S::ONE);
    let normalized_left = checked_float_quotient(left, scale)?;
    let normalized_right = checked_float_quotient(right, scale)?;
    let normalized_value = checked_float_quotient(value, scale)?;
    Ok(checked_float_quotient(
        checked_difference(normalized_value, normalized_left)?,
        checked_difference(normalized_right, normalized_left)?,
    )?
    .min(below_one))
}

fn wrap_coordinate<S: RealScalar>(
    axis: &CoordinateAxis<S>,
    coordinate: S,
) -> Result<S, SciOxideError> {
    if !coordinate.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "query coordinate must be finite",
        ));
    }
    let low = axis.start().min(axis.end());
    let high = axis.start().max(axis.end());
    if coordinate >= low && coordinate <= high {
        return Ok(coordinate);
    }
    // `coordinate - low` can overflow for a huge query even when the period
    // is finite. IEEE remainder is exact, so reducing each operand modulo the
    // width first gives bounded phases in (-1, 1) that subtract safely. The
    // difference is reduced again into [0, 1) and mapped back affinely.
    if let Ok(width) = checked_difference(high, low) {
        let coordinate_phase = checked_float_quotient(coordinate % width, width)?;
        let low_phase = checked_float_quotient(low % width, width)?;
        let mut fraction = checked_difference(coordinate_phase, low_phase)? % S::ONE;
        if fraction < S::ZERO {
            fraction += S::ONE;
        }
        return checked_affine(low, high, fraction);
    }
    // The period itself overflows. Normalize by a common scale so the width
    // and offset are finite, then wrap in normalized units.
    let scale = low.abs().max(high.abs()).max(coordinate.abs()).max(S::ONE);
    let normalized_low = checked_float_quotient(low, scale)?;
    let normalized_high = checked_float_quotient(high, scale)?;
    let normalized_coordinate = checked_float_quotient(coordinate, scale)?;
    let width = checked_difference(normalized_high, normalized_low)?;
    let mut offset = checked_difference(normalized_coordinate, normalized_low)? % width;
    if offset < S::ZERO {
        offset += width;
    }
    checked_affine(low, high, checked_float_quotient(offset, width)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Index1, Point1};
    use alloc::vec;

    fn assert_uniform_round_trip<S: RealScalar>(start: S, end: S, count: usize) {
        let axis = CoordinateAxis::uniform(start, end, count).unwrap();
        assert_eq!(axis.coordinate(0).unwrap(), start);
        assert_eq!(axis.coordinate(count - 1).unwrap(), end);
        for index in 0..count {
            let coordinate = axis.coordinate(index).unwrap();
            let location = axis.locate(coordinate).unwrap();
            if index + 1 == count {
                assert_eq!(location.cell(), count - 2);
                assert_eq!(location.fraction(), S::ONE);
            } else {
                assert_eq!(location.cell(), index);
                assert_eq!(location.fraction(), S::ZERO);
            }
        }
    }

    #[test]
    fn uniform_axes_round_trip_for_both_scalars_and_orientations() {
        assert_uniform_round_trip(0.0_f64, 1.0, 33);
        assert_uniform_round_trip(1.0_f64, -1.0, 33);
        assert_uniform_round_trip(0.0_f32, 1.0, 33);
        assert_uniform_round_trip(1.0_f32, -1.0, 33);
    }

    #[test]
    fn compact_uniform_axis_accepts_an_inexact_integer_count() {
        let count = (1_usize << 24) + 1;
        let axis = CoordinateAxis::<f32>::uniform(0.0, 1.0, count).unwrap();
        assert_eq!(axis.len(), count);
        assert!(axis.coordinate(1).unwrap() > 0.0);
        assert!(axis.coordinate(count - 2).unwrap() < 1.0);
    }

    fn spacing_boundaries<S: RealScalar>() {
        let one = S::ONE;
        let ulp = S::EPSILON;
        let end = one + S::try_from_usize_exact(8).unwrap() * ulp;
        let tiny = S::MIN_SUBNORMAL;
        for (start, end) in [
            (one, end),
            (-end, -one),
            (
                -tiny * S::try_from_usize_exact(4).unwrap(),
                tiny * S::try_from_usize_exact(4).unwrap(),
            ),
        ] {
            for (start, end) in [(start, end), (end, start)] {
                for count in [8, 9] {
                    let axis = CoordinateAxis::uniform(start, end, count).unwrap();
                    let values: Vec<_> = (0..count).map(|i| axis.coordinate(i).unwrap()).collect();
                    assert!(values.windows(2).all(|pair| strictly_after(
                        axis.orientation(),
                        pair[1],
                        pair[0]
                    )));
                }
                assert!(CoordinateAxis::uniform(start, end, 10).is_err());
            }
        }
        // Crossing a binade uses the inward ulp, not the larger outward ulp.
        let below = one - S::try_from_usize_exact(4).unwrap() * ulp;
        assert!(CoordinateAxis::uniform(below, one, 9).is_ok());
        assert!(CoordinateAxis::uniform(below, one, 10).is_err());
    }

    #[test]
    fn uniform_spacing_boundaries_cover_both_widths_and_orientations() {
        spacing_boundaries::<f32>();
        spacing_boundaries::<f64>();
        assert!(CoordinateAxis::<f32>::uniform(0., 1., 1 << 25).is_err());
        assert!(CoordinateAxis::<f32>::uniform(0., 1., (1 << 24) + 2).is_err());
        #[cfg(target_pointer_width = "64")]
        assert!(CoordinateAxis::<f64>::uniform(0., 1., 3_usize << 52).is_err());
    }

    #[test]
    fn sign_crossing_uniform_sites_do_not_round_the_fraction_first() {
        let axis = CoordinateAxis::<f32>::uniform(-1., 1., (1 << 25) + 1).unwrap();
        let index = (1 << 24) + 1;
        assert_eq!(axis.coordinate(index).unwrap(), f32::EPSILON / 2.);
        assert!(axis.coordinate(index - 1).unwrap() < axis.coordinate(index).unwrap());
        assert!(axis.coordinate(index).unwrap() < axis.coordinate(index + 1).unwrap());
    }

    #[test]
    fn explicit_axis_rejects_duplicates_and_mixed_orientation() {
        assert!(
            matches ! (& (CoordinateAxis :: explicit (vec ! [0.0_f64 , 1.0 , 1.0])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
        assert!(
            matches ! (& (CoordinateAxis :: explicit (vec ! [0.0_f64 , 2.0 , 1.0])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
        assert!(CoordinateAxis::explicit(vec![2.0_f64, 0.5, -4.0]).is_ok());
    }

    #[test]
    fn extreme_endpoints_are_preserved_and_located() {
        let axis = CoordinateAxis::uniform(-f64::MAX, f64::MAX, 3).unwrap();
        assert_eq!(axis.coordinate(0).unwrap(), -f64::MAX);
        assert_eq!(axis.coordinate(1).unwrap(), 0.0);
        assert_eq!(axis.coordinate(2).unwrap(), f64::MAX);
        assert_eq!(axis.locate(0.0).unwrap().cell(), 1);
    }

    // Both rounded differences equal one, but the interval is half-open.
    #[test]
    fn interior_fractions_stay_below_one() {
        let tiny = 2f64.powi(-54);
        let below = 1.0 - 2f64.powi(-53);
        for axis in [
            CoordinateAxis::explicit(vec![-tiny, 1.0, 2.0]).unwrap(),
            CoordinateAxis::explicit(vec![2.0, 1.0, -tiny]).unwrap(),
        ] {
            let interior = axis.locate(below).unwrap();
            assert_eq!(
                interior.cell(),
                if axis.coordinate(0).unwrap() < 0.0 {
                    0
                } else {
                    1
                }
            );
            assert!(interior.fraction() < 1.0);
            assert_eq!(axis.locate(1.0).unwrap().fraction(), 0.0);
        }
        let last = CoordinateAxis::explicit(vec![-tiny, 1.0])
            .unwrap()
            .locate(1.0)
            .unwrap();
        assert_eq!((last.cell(), last.fraction()), (0, 1.0));
    }

    #[test]
    fn structured_grid_borrows_uniform_and_explicit_axes() {
        let grid = Grid2d::new(
            [
                CoordinateAxis::uniform(0.0_f64, 1.0, 3).unwrap(),
                CoordinateAxis::explicit(vec![3.0, 1.0, -2.0]).unwrap(),
            ],
            [AxisTopology::Bounded; 2],
        )
        .unwrap();
        let shared = grid.clone();
        for (index, expected, orientation) in [
            (0, [0.0, 0.5, 1.0], AxisOrientation::Increasing),
            (1, [3.0, 1.0, -2.0], AxisOrientation::Decreasing),
        ] {
            let axis: &CoordinateAxis<f64> = grid.axis(index).unwrap();
            assert!(core::ptr::eq(axis, shared.axis(index).unwrap()));
            assert_eq!(axis.orientation(), orientation);
            assert_eq!(axis.len(), expected.len());
            assert!(!axis.is_empty());
            assert!((axis.start() - expected[0]).abs() < 1e-14);
            assert!((axis.end() - expected[2]).abs() < 1e-14);
            for (site, (coordinate, expected)) in axis.coordinates().zip(expected).enumerate() {
                assert!((coordinate.unwrap() - expected).abs() < 1e-14);
                assert!((axis.coordinate(site).unwrap() - expected).abs() < 1e-14);
            }
            let location = axis.locate(expected[1]).unwrap();
            assert_eq!(location.cell(), 1);
            assert!(location.fraction().abs() < 1e-14);
        }
    }

    #[test]
    fn structured_grid_axis_rejects_out_of_bounds_indices() {
        let grid = Grid1d::new(
            [CoordinateAxis::uniform(0.0_f64, 1.0, 2).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap();
        for index in [1, usize::MAX] {
            let error = grid.axis(index).unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::OutOfBounds {
                    collection,
                    index: rejected,
                    size: 1,
                } if collection == scioxide_core::quantity!("structured_grid_axis") && rejected == index
            ));
        }
    }

    #[test]
    fn structured_index_and_coordinate_round_trip() {
        let grid = Grid2d::new(
            [
                CoordinateAxis::uniform(-2.0_f64, 2.0, 5).unwrap(),
                CoordinateAxis::explicit(vec![3.0, 1.0, -2.0]).unwrap(),
            ],
            [AxisTopology::Bounded; 2],
        )
        .unwrap();
        for linear in 0..grid.vertex_extent().element_count().unwrap() {
            let index = grid.vertex_extent().delinearize(linear).unwrap();
            let point = grid.point(index).unwrap();
            let location = grid.locate(point).unwrap();
            for axis in 0..2 {
                let site = index.as_array()[axis];
                let last = grid.vertex_extent().as_array()[axis] - 1;
                assert_eq!(
                    location.cell().as_array()[axis],
                    if site == last { last - 1 } else { site }
                );
            }
        }
        assert_eq!(grid.orientation(), AxisOrientation::Decreasing);
    }

    #[test]
    fn periodic_cell_adjacency_is_symmetric_and_wraps() {
        let grid = Grid1d::new(
            [CoordinateAxis::uniform(0.0_f64, 4.0, 5).unwrap()],
            [AxisTopology::Periodic],
        )
        .unwrap();
        let first = Index1::new([0]).unwrap();
        let neighbors = grid
            .cell_neighbors(first)
            .unwrap()
            .map(|index| index.as_array()[0])
            .collect::<Vec<_>>();
        assert_eq!(neighbors, [3, 1]);
        for cell in 0..4 {
            let source = Index1::new([cell]).unwrap();
            for neighbor in grid.cell_neighbors(source).unwrap() {
                assert!(
                    grid.cell_neighbors(neighbor)
                        .unwrap()
                        .any(|candidate| candidate == source)
                );
            }
        }
    }

    #[test]
    fn periodic_wrapping_is_explicit_and_preserves_seam_endpoints() {
        let grid = Grid1d::new(
            [CoordinateAxis::uniform(-1.0_f64, 1.0, 5).unwrap()],
            [AxisTopology::Periodic],
        )
        .unwrap();
        assert!(grid.locate(Point1::new([3.5]).unwrap()).is_err());
        let wrapped = grid.wrap_point(Point1::new([3.5]).unwrap()).unwrap();
        assert!((wrapped.as_array()[0] + 0.5).abs() < 1e-14);
        assert_eq!(
            grid.wrap_point(Point1::new([1.0]).unwrap())
                .unwrap()
                .as_array()[0],
            1.0
        );
    }

    #[test]
    fn refinement_preserves_family_and_advances_generation() {
        let grid = Grid1d::new(
            [CoordinateAxis::uniform(0.0_f64, 1.0, 3).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap();
        let same = grid.refined([1]).unwrap();
        let refined = grid.refined([2]).unwrap();
        assert!(grid.mesh_token().is_compatible_with(same.mesh_token()));
        assert!(grid.mesh_token().same_family(refined.mesh_token()));
        assert!(!grid.mesh_token().is_compatible_with(refined.mesh_token()));
        assert_eq!(refined.vertex_extent().as_array(), &[5]);
    }

    #[test]
    fn grid_cardinality_overflow_precedes_publication() {
        let result = Grid3d::new(
            [
                CoordinateAxis::uniform(0.0_f64, 1.0, 3_000_001).unwrap(),
                CoordinateAxis::uniform(0.0_f64, 1.0, 3_000_001).unwrap(),
                CoordinateAxis::uniform(0.0_f64, 1.0, 3_000_001).unwrap(),
            ],
            [AxisTopology::Bounded; 3],
        );
        assert!(
            matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
    }
}
