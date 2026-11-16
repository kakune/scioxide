//! Convex hulls in any dimension.
use crate::PointSet;
use crate::tessellation::GeometryOptions;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::engine::hull;
use crate::tessellation::numeric::validate_query;
use crate::tessellation::preparation::{prepare, scoped};
use crate::tessellation::storage::Storage;
use crate::tessellation::topology::Topology;
use crate::tessellation::work::Work;
use crate::tessellation::work::provenance;
use crate::tessellation::{
    GeometryConstructionRequirements, GeometryProvenance, GeometryWorkspace, Result,
};
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SolveScope};
/// Convex hull of a set of sites in `d` dimensions, as a triangulated boundary.
///
/// The boundary consists of `(d − 1)`-simplices called facets. A face of the
/// hull that is not a simplex, such as a square face of a cube, is split into
/// several facets, and [`Self::coplanar_faces`] groups them. The hull is built
/// by randomized incremental insertion with a fixed seed, using exact
/// orientation tests, so it is exact for the given coordinates. See the
/// [crate documentation](crate#tessellations) for site order and limits.
///
/// ```
/// use scioxide_core::{ExecutionContext, };
/// use scioxide_spatial::{ConvexHull, GeometryOptions, PointSet};
/// let values = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
/// let points = PointSet::from_rows(&values, 2)?;
/// let hull = ConvexHull::new(&points)?;
/// let mut workspace = hull.workspace(())?;
/// assert!(hull.contains_scoped(
///     &[0.25, 0.25],
///     &mut workspace,
///     ExecutionContext::sequential()
/// )?);
/// assert!(
///     hull.contains_scoped(&[0.0], &mut workspace, ExecutionContext::sequential())
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct ConvexHull<S: RealScalar> {
    pub(super) options: GeometryOptions,
    requirements: GeometryConstructionRequirements,
    pub(super) cloud: Cloud<S>,
    pub(super) topology: Topology,
    provenance: GeometryProvenance,
    coplanar_faces: Vec<usize>,
}
impl<S: RealScalar> ConvexHull<S> {
    /// Constructs a ConvexHull with default geometry options.
    ///
    /// Uses the finite default geometry limits and the construction method and cost of [`Self::with_options`].
    ///
    /// # Errors
    /// As [`Self::with_options`].
    ///
    /// ```
    /// use scioxide_spatial::{ConvexHull, PointSet};
    /// let sites = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
    /// let model = ConvexHull::new(&PointSet::from_points(&sites)?)?;
    /// assert!(model.contains(&[0.2, 0.2])?);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(points: &PointSet<'_, S>) -> Result<Self> {
        Self::with_options(points, GeometryOptions::default())
    }

    /// Computes the convex hull of `points`, one site per row.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the points have dimension zero, or if
    ///   [`GeometryDegeneracy::Strict`](crate::GeometryDegeneracy::Strict)
    ///   rejects duplicate sites.
    /// - `Singular` if the sites do not span the space, for example when there
    ///   are fewer than `d + 1` distinct sites.
    /// - `ResourceLimit` if the construction exceeds a
    ///   [`GeometryLimits`](crate::GeometryLimits) value.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn with_options(points: &PointSet<'_, S>, options: GeometryOptions) -> Result<Self> {
        Self::prepare(points, options, None)
    }
    /// Computes the convex hull under an execution or solve context.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `ExecutionStopped` or `ConvergenceFailure` as for
    /// [`Delaunay::build`](crate::Delaunay::build).
    pub fn build<'scope, 'capability>(
        points: &PointSet<'_, S>,
        options: GeometryOptions,
        cx: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        Self::prepare(points, options, Some(cx.into()))
    }
    fn prepare<'scope, 'capability>(
        points: &PointSet<'_, S>,
        options: GeometryOptions,
        scope: Option<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        prepare(
            points.dimension(),
            points.len(),
            scioxide_core::argument!("points"),
            options,
            scope,
            |work| Self::construct(points, work),
        )
    }
    /// Returns the peak storage used during construction.
    pub const fn construction_requirements(&self) -> GeometryConstructionRequirements {
        self.requirements
    }
    fn construct(points: &PointSet<'_, S>, work: &mut Work<'_, '_, '_, '_>) -> Result<Self> {
        let cloud = Cloud::new(points, work.options, work.storage)?;
        let topology = hull(&cloud, false, work)?;
        Self::from_topology(cloud, topology, work)
    }
    pub(super) fn from_topology(
        cloud: Cloud<S>,
        topology: Topology,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let (topology, coplanar_faces) = super::canonical::hull(&cloud, topology, work)?;
        let provenance = provenance(&cloud, work);
        Ok(Self {
            options: work.options,
            requirements: work.storage.requirements(),
            cloud,
            topology,
            provenance,
            coplanar_faces,
        })
    }
    /// Returns the coordinate dimension `d`.
    pub const fn dimension(&self) -> usize {
        self.cloud.dimension
    }
    /// Returns the site coordinates in sorted-site order, `d` values per site.
    pub fn points(&self) -> &[S] {
        &self.cloud.points
    }
    /// Returns the site index of each input row.
    pub fn input_mapping(&self) -> &[usize] {
        &self.cloud.mapping
    }
    /// Returns the first input row of each site.
    pub fn representatives(&self) -> &[usize] {
        &self.cloud.representatives
    }
    /// Returns the number of facets.
    pub fn facet_count(&self) -> usize {
        self.topology.facets.len()
    }
    /// Returns the `d` site indices of a facet in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.facet_count()`.
    pub fn facet(&self, index: usize) -> Result<&[usize]> {
        self.topology.vertices(index)
    }
    /// Returns, for each vertex of a facet, the facet that shares the
    /// opposite ridge.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.facet_count()`.
    pub fn neighbors(&self, index: usize) -> Result<&[Option<usize>]> {
        self.topology.adjacent(index)
    }
    /// Returns, for each facet, the index of the hull face it belongs to.
    /// Facets on the same supporting hyperplane share an index.
    pub fn coplanar_faces(&self) -> &[usize] {
        &self.coplanar_faces
    }
    /// Returns the input counts and work of the construction.
    pub const fn provenance(&self) -> GeometryProvenance {
        self.provenance
    }
    /// Returns whether `point` lies in the closed convex hull.
    ///
    /// The test evaluates one exact orientation per facet, so it costs
    /// O(facets). Queries with a reserved workspace do not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point.len()` is not the dimension or
    ///   `workspace` does not meet `workspace_requirements`.
    /// - `NonFinite` if `point` contains NaN or infinity.
    /// - `ResourceLimit` if the query exceeds
    ///   [`GeometryLimits::max_predicate_terms`](crate::GeometryLimits::max_predicate_terms).
    /// - `ExecutionStopped` on cancellation or deadline, and
    ///   `ConvergenceFailure` if the node limit of a solve context is
    ///   exhausted.
    pub fn contains_scoped<'scope, 'capability>(
        &self,
        point: &[S],
        workspace: &mut GeometryWorkspace,
        cx: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<bool> {
        let storage = Storage::new(self.options.limits.max_storage_bytes, 0)?;
        scoped(cx, |context| {
            validate_query(point, self.dimension())?;
            let mut work = Work::new(
                workspace,
                self.dimension(),
                self.options,
                Some(context),
                scioxide_core::argument!("self"),
                &storage,
            )?;
            for facet in self.topology.facets.iter() {
                fill_query(&self.cloud, facet.vertices, point, &mut work)?;
                let sign = work.raw(self.dimension() + 1, self.dimension(), false)?;
                if sign != 0 && sign != facet.inside {
                    return Ok(false);
                }
            }
            Ok(true)
        })
    }
}
fn fill_query<S: RealScalar>(
    cloud: &Cloud<S>,
    vertices: &[usize],
    point: &[S],
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<()> {
    let d = cloud.dimension;
    for (row, &index) in vertices.iter().enumerate() {
        for (axis, &value) in cloud.point(index).iter().enumerate() {
            work.workspace.rows[row * d + axis] = value.try_to_f64_exact()?;
        }
    }
    for (axis, &value) in point.iter().enumerate() {
        work.workspace.rows[vertices.len() * d + axis] = value.try_to_f64_exact()?;
    }
    Ok(())
}

impl<S: RealScalar> ConvexHull<S> {
    /// Tests hull containment.
    ///
    /// Allocates a workspace and runs [`Self::contains_scoped`] with sequential native execution; it has the same cost and accuracy.
    ///
    /// # Errors
    /// As [`Self::contains_scoped`], plus workspace allocation failure.
    pub fn contains(&self, point: &[S]) -> Result<bool> {
        self.contains_scoped(
            point,
            &mut self.workspace(())?,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}

impl<S: RealScalar> ConvexHull<S> {
    /// Returns the flat site indices, `d` per row for dimension `d`.
    ///
    /// Borrows the stored topology in `O(1)` time without allocation.
    ///
    /// # Errors
    /// This accessor is infallible.
    pub fn facets(&self) -> &[usize] {
        self.topology.facets.vertices()
    }
}
