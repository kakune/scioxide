//! Convex hulls, Delaunay and Voronoi tessellations and halfspace intersections.
mod canonical;
mod cloud;
mod delaunay;
mod engine;
mod halfspace;
mod hull;
mod incidence;
mod linear;
mod numeric;
mod preparation;
mod spherical;
mod storage;
mod topology;
mod voronoi;
mod work;
use crate::validation::inconsistent as invalid;
pub use storage::GeometryConstructionRequirements;
use storage::retained;
#[cfg(test)]
mod tests;

use crate::{PredicateSign, predicates::nd::ExactWorkspace};
use alloc::vec::Vec;
pub use delaunay::{Delaunay, SimplexLocation};
pub use halfspace::{HalfspaceIntersection, IntersectionKind};
pub use hull::ConvexHull;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{Argument, Capacity, Quantity, RealScalar, SciOxideError};
pub use spherical::SphericalVoronoi;
pub use voronoi::{Voronoi, VoronoiRay, VoronoiRidge};

/// Selects how a tessellation treats duplicate sites and ties.
///
/// Neither policy moves coordinates. Both resolve exact zero signs by symbolic
/// perturbation in sorted-site order, so they build the same tessellation
/// whenever `Strict` accepts the input.
///
/// ```
/// use scioxide_spatial::{Delaunay, GeometryDegeneracy, GeometryOptions, PointSet};
/// let strict = GeometryOptions::new().with_degeneracy(GeometryDegeneracy::Strict);
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let corners = PointSet::from_rows(&sites[..8], 2)?;
/// let fan = PointSet::from_rows(&sites, 2)?;
/// // Four cocircular corners have two equally valid triangulations.
/// assert!(Delaunay::with_options(&corners, strict).is_err());
/// assert_eq!(
///     Delaunay::new(&corners)?.simplex_count(),
///     2
/// );
/// // The interior site makes the triangulation unique.
/// assert_eq!(Delaunay::with_options(&fan, strict)?.simplex_count(), 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryDegeneracy {
    /// Merges duplicate sites and resolves ties by symbolic perturbation. The
    /// result is the same for every input order.
    Deterministic,
    /// Returns `InvalidInput` for duplicate sites, and for a Delaunay or
    /// Voronoi input whose triangulation is not unique: two adjacent output
    /// simplices that share an empty circumsphere. Ties that do not affect
    /// the output simplices are accepted.
    Strict,
}
/// Sets resource limits for tessellation construction and queries.
///
/// The defaults allow 8 dimensions, 1,000,000 sites, 1,000,000 facets,
/// 10⁹ predicate work units, 10⁹ combinatorial work units and 512 MiB of
/// storage. Each limit is checked on its own; an input within the site limit
/// can still exceed another limit. A construction that exceeds a limit
/// returns `ResourceLimit`. Large limits do not reserve storage.
///
/// ```
/// use scioxide_spatial::{GeometryLimits, GeometryOptions};
/// let limits = GeometryLimits::new()
///     .with_max_points(4096)
///     .with_max_storage_bytes(1 << 24);
/// assert_eq!((limits.max_points(), limits.max_dimension()), (4096, 8));
/// let options = GeometryOptions::new().with_limits(limits);
/// assert_eq!(options.limits().max_storage_bytes(), 1 << 24);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeometryLimits {
    max_dimension: usize,
    max_points: usize,
    max_facets: usize,
    max_predicate_terms: usize,
    max_canonical_work: usize,
    max_storage_bytes: usize,
}
impl Default for GeometryLimits {
    fn default() -> Self {
        Self::new()
    }
}
impl GeometryLimits {
    /// Returns the default limits.
    pub const fn new() -> Self {
        Self {
            max_dimension: 8,
            max_points: 1000000,
            max_facets: 1000000,
            max_predicate_terms: 1000000000,
            max_canonical_work: 1000000000,
            max_storage_bytes: 536870912,
        }
    }
    /// Returns the largest accepted coordinate dimension.
    pub const fn max_dimension(self) -> usize {
        self.max_dimension
    }
    /// Sets the largest accepted coordinate dimension.
    pub const fn with_max_dimension(mut self, value: usize) -> Self {
        self.max_dimension = value;
        self
    }
    /// Returns the largest accepted number of input sites or halfspaces.
    pub const fn max_points(self) -> usize {
        self.max_points
    }
    /// Sets the largest accepted number of input sites or halfspaces.
    pub const fn with_max_points(mut self, value: usize) -> Self {
        self.max_points = value;
        self
    }
    /// Returns the largest number of hull facets present at once during
    /// construction, including the facets of the lifted hull for Delaunay.
    pub const fn max_facets(self) -> usize {
        self.max_facets
    }
    /// Sets the largest number of facets present at once during construction.
    pub const fn with_max_facets(mut self, value: usize) -> Self {
        self.max_facets = value;
        self
    }
    /// Returns the predicate work limit for one construction or query.
    ///
    /// Each sign evaluation costs one unit. When the floating-point filter
    /// cannot decide the sign, the exact evaluation adds the number of terms
    /// in the Leibniz expansion of the determinant: `(d+1)!` for an
    /// orientation test in dimension `d`, `d (d+2)!` for an in-sphere test, and
    /// `n!` for an order-`n` determinant in homogeneous coordinates. The count
    /// is deterministic, so repeated calls on the same input use the same
    /// number of units. Exceeding the limit returns `ResourceLimit`.
    pub const fn max_predicate_terms(self) -> usize {
        self.max_predicate_terms
    }
    /// Sets the predicate work limit.
    pub const fn with_max_predicate_terms(mut self, value: usize) -> Self {
        self.max_predicate_terms = value;
        self
    }
    /// Returns the combinatorial work limit for one construction.
    ///
    /// This counts the work of grouping coplanar facets into faces and
    /// subdividing them: incidence visits, vertex steps and sorting
    /// comparisons. Determinant evaluations count toward
    /// [`Self::max_predicate_terms`] instead. Exceeding the limit returns
    /// `ResourceLimit` with quantity `geometry_canonical_work`. In `build`, both
    /// counts also count toward the node limit of the solve context.
    pub const fn max_canonical_work(self) -> usize {
        self.max_canonical_work
    }
    /// Sets the combinatorial work limit; zero allows no such work.
    pub const fn with_max_canonical_work(mut self, value: usize) -> Self {
        self.max_canonical_work = value;
        self
    }
    /// Returns the storage limit in bytes for a construction, or for the
    /// scratch held by a [`GeometryWorkspace`].
    pub const fn max_storage_bytes(self) -> usize {
        self.max_storage_bytes
    }
    /// Sets the storage limit in bytes.
    pub const fn with_max_storage_bytes(mut self, value: usize) -> Self {
        self.max_storage_bytes = value;
        self
    }
}
/// Options for tessellation construction: resource limits and the degeneracy
/// policy.
///
/// A tessellation keeps its options and applies the same limits to its
/// queries.
///
/// ```
/// use scioxide_spatial::{GeometryLimits, GeometryOptions};
/// let options = GeometryOptions::new().with_limits(
///     GeometryLimits::new()
///         .with_max_storage_bytes(16 * 1024 * 1024)
///         .with_max_canonical_work(1_000_000),
/// );
/// assert_eq!(options.limits().max_dimension(), 8);
/// assert_eq!(options.limits().max_canonical_work(), 1_000_000);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeometryOptions {
    limits: GeometryLimits,
    degeneracy: GeometryDegeneracy,
}
impl Default for GeometryOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl GeometryOptions {
    /// Returns the default limits with [`GeometryDegeneracy::Deterministic`].
    pub const fn new() -> Self {
        Self {
            limits: GeometryLimits::new(),
            degeneracy: GeometryDegeneracy::Deterministic,
        }
    }
    /// Sets the resource limits.
    pub const fn with_limits(mut self, limits: GeometryLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Sets the degeneracy policy.
    pub const fn with_degeneracy(mut self, degeneracy: GeometryDegeneracy) -> Self {
        self.degeneracy = degeneracy;
        self
    }
    /// Returns the resource limits.
    pub const fn limits(self) -> GeometryLimits {
        self.limits
    }
    /// Returns the degeneracy policy.
    pub const fn degeneracy(self) -> GeometryDegeneracy {
        self.degeneracy
    }
}
type Result<T> = core::result::Result<T, SciOxideError>;
fn unstable(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::IllConditioned, detail)
}
fn degenerate() -> SciOxideError {
    invalid(
        scioxide_core::argument!("points"),
        "strict geometry policy rejected duplicate or cospherical sites",
    )
}
fn limit(resource: Quantity, required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(SciOxideError::resource_limit(resource, required, limit))
    } else {
        Ok(())
    }
}
fn sign_value(sign: PredicateSign) -> i8 {
    match sign {
        PredicateSign::Negative => -1,
        PredicateSign::Zero => 0,
        PredicateSign::Positive => 1,
    }
}
/// Scratch storage needed by the queries of one tessellation.
///
/// Obtain it from `workspace_requirements` on a tessellation and pass it to
/// [`GeometryWorkspace::try_reserve`].
///
/// ```
/// use scioxide_core::{Capacity, };
/// use scioxide_spatial::{ConvexHull, GeometryOptions, GeometryWorkspace, PointSet};
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let points = PointSet::from_rows(&sites, 2)?;
/// let hull = ConvexHull::new(&points)?;
/// let requirements = hull.workspace_requirements(())?;
/// let mut workspace = GeometryWorkspace::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert!(hull.contains_scoped(
///     &[0.25, 0.25],
///     &mut workspace,
///     scioxide_core::ExecutionContext::sequential()
/// )?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeometryWorkspaceRequirements {
    dimension: usize,
    bytes: usize,
    maximum: usize,
}
impl GeometryWorkspaceRequirements {
    fn checked(dimension: usize, options: GeometryOptions, source: Argument) -> Result<Self> {
        if dimension == 0 {
            return Err(invalid(source, "geometry requires positive dimension"));
        }
        limit(
            scioxide_core::quantity!("geometry_dimension"),
            dimension,
            options.limits.max_dimension,
        )?;
        let order = checked_add(dimension, 2, scioxide_core::quantity!("geometry_order"))?;
        let rows = checked_mul(order, order, scioxide_core::quantity!("geometry_rows"))?;
        let bytes = checked_add(
            ExactWorkspace::bytes(order)?,
            checked_add(
                checked_mul(
                    rows,
                    size_of::<f64>(),
                    scioxide_core::quantity!("geometry_rows"),
                )?,
                checked_mul(
                    dimension + 1,
                    size_of::<usize>(),
                    scioxide_core::quantity!("location_face"),
                )?,
                scioxide_core::quantity!("geometry_workspace"),
            )?,
            scioxide_core::quantity!("geometry_workspace"),
        )?;
        limit(
            scioxide_core::quantity!("geometry_storage_bytes"),
            bytes,
            options.limits.max_storage_bytes,
        )?;
        Ok(Self {
            dimension,
            bytes,
            maximum: options.limits.max_storage_bytes,
        })
    }
    /// Returns the scratch size in bytes. The tessellation itself is not
    /// included.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}
/// Scratch storage for point location and containment queries.
///
/// One workspace serves any tessellation whose requirements it meets. It also
/// remembers the simplex found by the last [`Delaunay::find_simplex`] call, which
/// speeds up nearby queries without changing their results. Queries with a
/// reserved workspace do not allocate.
///
/// ```
/// use scioxide_core::{Capacity, };
/// use scioxide_spatial::{Delaunay, GeometryOptions, GeometryWorkspace, PointSet};
/// let values = [0.0_f64, 0., 1., 0., 0., 1.];
/// let points = PointSet::from_rows(&values, 2)?;
/// let geometry = Delaunay::new(&points)?;
/// let requirements = geometry.workspace_requirements(())?;
/// let mut workspace = GeometryWorkspace::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// assert!(geometry.construction_requirements().bytes() >= requirements.bytes());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct GeometryWorkspace {
    exact: ExactWorkspace,
    rows: Vec<f64>,
    face_vertices: Vec<usize>,
    dimension: usize,
    terms: usize,
    // Work units admitted since the last stop poll.
    unpolled: usize,
    location: Option<(crate::MeshToken, usize)>,
}
impl Default for GeometryWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
impl GeometryWorkspace {
    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            exact: ExactWorkspace::empty(),
            rows: Vec::new(),
            face_vertices: Vec::new(),
            dimension: 0,
            terms: 0,
            unpolled: 0,
            location: None,
        }
    }
    fn retained_bytes(&self) -> usize {
        self.exact.retained_bytes() + retained(&self.rows) + retained(&self.face_vertices)
    }
    /// Reports whether the workspace meets `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the capacity already held exceeds the storage limit
    ///   of the options that produced `requirements`.
    pub fn check(&self, requirements: &GeometryWorkspaceRequirements) -> Result<Capacity> {
        limit(
            scioxide_core::quantity!("geometry_storage_bytes"),
            self.retained_bytes(),
            requirements.maximum,
        )?;
        Ok(if self.dimension >= requirements.dimension {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to meet `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the resulting capacity would exceed the storage
    ///   limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved. The workspace may then keep part of the new capacity.
    pub fn try_reserve(&mut self, requirements: &GeometryWorkspaceRequirements) -> Result<()> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let d = requirements.dimension;
        let projected = self.exact.projected_bytes(d + 2)?
            + self.rows.capacity().max((d + 2) * (d + 2)) * size_of::<f64>()
            + self.face_vertices.capacity().max(d + 1) * size_of::<usize>();
        limit(
            scioxide_core::quantity!("geometry_storage_bytes"),
            projected,
            requirements.maximum,
        )?;
        self.exact.try_reserve(d + 2)?;
        crate::workspace::grow(&mut self.rows, (d + 2) * (d + 2), 0.)?;
        crate::workspace::grow(&mut self.face_vertices, d + 1, 0)?;
        self.face_vertices.clear();
        limit(
            scioxide_core::quantity!("geometry_storage_bytes"),
            self.retained_bytes(),
            requirements.maximum,
        )?;
        self.dimension = d;
        self.location = None;
        Ok(())
    }
    fn prepared(requirements: GeometryWorkspaceRequirements) -> Result<Self> {
        let mut workspace = Self::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    /// Returns the predicate work units used by the last query with this
    /// workspace, counted as in [`GeometryLimits::max_predicate_terms`].
    pub const fn predicate_terms(&self) -> usize {
        self.terms
    }
}

/// Input counts and work used by a tessellation construction.
///
/// ```
/// use scioxide_spatial::{ConvexHull, GeometryOptions, PointSet};
/// // The last row repeats the first and is merged.
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let points = PointSet::from_rows(&sites, 2)?;
/// let counts = ConvexHull::new(&points)?.provenance();
/// assert_eq!((counts.input_points, counts.unique_points), (4, 3));
/// assert!(counts.predicate_terms > 0 && !counts.resolved_ties);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeometryProvenance {
    /// Number of input sites or inequalities.
    pub input_points: usize,
    /// Number of sites after merging exact duplicates. Inequalities are never
    /// merged.
    pub unique_points: usize,
    /// Predicate work units, counted as in
    /// [`GeometryLimits::max_predicate_terms`].
    pub predicate_terms: usize,
    /// Combinatorial work units, counted as in
    /// [`GeometryLimits::max_canonical_work`].
    pub canonical_work: usize,
    /// Whether symbolic perturbation decided at least one exact zero sign,
    /// including ties that do not affect the output. The value does not depend
    /// on the input order.
    pub resolved_ties: bool,
}

macro_rules! query_workspace {
    ($ty:ident) => {
        impl<S: RealScalar> $ty<S> {
            /// Returns the options used for construction, which also apply to
            /// queries.
            pub const fn options(&self) -> GeometryOptions {
                self.options
            }
            /// Returns the scratch storage that queries on this tessellation
            /// need.
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` if the scratch exceeds the storage limit.
            /// - `SizeOverflow` if its size is not representable.
            pub fn workspace_requirements(&self, (): ()) -> Result<GeometryWorkspaceRequirements> {
                GeometryWorkspaceRequirements::checked(
                    self.dimension(),
                    self.options,
                    scioxide_core::argument!("self"),
                )
            }
            /// Allocates a [`GeometryWorkspace`] that meets
            /// `workspace_requirements`.
            ///
            /// # Errors
            ///
            /// As `workspace_requirements`, and `AllocationFailure` if the
            /// storage cannot be reserved.
            pub fn workspace(&self, (): ()) -> Result<GeometryWorkspace> {
                GeometryWorkspace::prepared(self.workspace_requirements(())?)
            }
        }
    };
}
query_workspace!(ConvexHull);
query_workspace!(Delaunay);
