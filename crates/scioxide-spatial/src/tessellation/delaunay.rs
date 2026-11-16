//! Delaunay triangulations, point location and barycentric coordinates.
use crate::PointSet;
use crate::incidence::Incidence;
use crate::tessellation::GeometryOptions;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::engine::hull;
use crate::tessellation::numeric::validate_query;
use crate::tessellation::preparation::prepare;
use crate::tessellation::preparation::scoped;
use crate::tessellation::storage::{Storage, retained};
use crate::tessellation::topology::{Facets, Topology};
use crate::tessellation::work::Work;
use crate::tessellation::work::provenance;
use crate::tessellation::{
    GeometryConstructionRequirements, GeometryDegeneracy, GeometryProvenance, GeometryWorkspace,
    Result, degenerate, unstable,
};
use crate::validation::inconsistent as invalid;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{RealScalar, SciOxideError, SolveScope};
mod location;
use super::linear::LinearFactor;
use crate::{MeshToken, Point, TetrahedronMesh, TriangleMesh};
use scioxide_linalg::{Matrix, Vector};
/// The simplex that contains a query point, as returned by
/// [`Delaunay::find_simplex`].
///
/// ```
/// use scioxide_core::{ExecutionContext};
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let points = scioxide_spatial::PointSet::from_rows(&sites, 2)?;
/// let delaunay = scioxide_spatial::Delaunay::with_options(&points, scioxide_spatial::GeometryOptions::new())?;
/// let mut workspace = delaunay.workspace(())?;
/// // A site lies on the boundary of every incident cell.
/// let at_site = delaunay.find_simplex_scoped(&[1.0, 0.0], &mut workspace, ExecutionContext::sequential())?;
/// assert!(at_site.is_some_and(|location| location.boundary));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplexLocation {
    /// Index of the simplex.
    pub simplex: usize,
    /// Whether the point lies exactly on a face, edge or vertex of the
    /// simplex, that is, whether at least one barycentric coordinate is exactly
    /// zero.
    pub boundary: bool,
}
/// Delaunay triangulation of a set of sites in `d` dimensions.
///
/// The triangulation consists of `d`-simplices whose circumspheres contain no
/// other site. It is computed as the lower convex hull of the sites lifted to
/// the paraboloid `x ↦ (x, |x|²)` in `d + 1` dimensions, built by randomized
/// incremental insertion with a fixed seed. All decisions use exact
/// predicates, so the topology is exact for the given coordinates. Ties
/// follow the [degeneracy policy](GeometryDegeneracy).
///
/// Barycentric coordinates and circumcenters come from a rounded affine map
/// stored for each simplex. If that map is too ill-conditioned for a simplex,
/// those queries return `NumericalInstability` for that simplex only; its
/// topology and point location stay usable. See the
/// [crate documentation](crate#tessellations) for site order and limits.
///
/// ```
/// use scioxide_core::{ExecutionContext};
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let points = scioxide_spatial::PointSet::from_rows(&sites, 2)?;
/// let delaunay = scioxide_spatial::Delaunay::with_options(&points, scioxide_spatial::GeometryOptions::new())?;
/// // The interior site makes the cocircular square corners a unique fan.
/// assert_eq!(delaunay.simplex_count(), 4);
/// let mut workspace = delaunay.workspace(())?;
/// let query = [0.5, 0.2];
/// let location = delaunay
///     .find_simplex_scoped(&query, &mut workspace, ExecutionContext::sequential())?
///     .expect("inside the hull");
/// let (mut weights, mut scratch) = ([0.0; 3], [0.0; 3]);
/// delaunay.barycentric_into(location.simplex, &query, &mut weights, &mut scratch)?;
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct Delaunay<S: RealScalar> {
    pub(super) options: GeometryOptions,
    requirements: GeometryConstructionRequirements,
    pub(super) cloud: Cloud<S>,
    pub(super) topology: Topology,
    pub(super) transforms: AffineTransforms,
    seeds: location::SeedIndex,
    pub(super) vertex_simplices: Incidence,
    provenance: GeometryProvenance,
    token: MeshToken,
}
// Flat per-simplex frames: origin[d], inverse[d*d], local center[d],
// scale, edge_scale. Physical x maps to ((x / scale) - origin) / edge_scale.
// Exact topology never depends on these approximate coordinates.
#[derive(Debug)]
pub(super) struct AffineTransforms {
    dimension: usize,
    // Checked frame length: origin, inverse, circumcenter and two scales.
    stride: usize,
    data: Vec<f64>,
    status: Vec<ApproximationStatus>,
}
/// Whether the affine map of a simplex is usable.
#[derive(Debug, Clone, Copy)]
pub(super) enum ApproximationStatus {
    Ready,
    Failed,
}
impl ApproximationStatus {
    pub(super) fn check(self) -> Result<()> {
        match self {
            Self::Ready => Ok(()),
            Self::Failed => Err(unstable(
                "simplex approximation is ill-conditioned or unrepresentable",
            )),
        }
    }
}
pub(super) struct AffineTransform<'a> {
    pub(super) origin: &'a [f64],
    pub(super) inverse: &'a [f64],
    center: &'a [f64],
    pub(super) scale: f64,
    pub(super) edge_scale: f64,
}
struct AffineWorkspace {
    matrix: Matrix<f64>,
    factor: LinearFactor,
    rhs: Vector<f64>,
    solution: Vector<f64>,
}
impl AffineTransforms {
    fn new<S: RealScalar>(
        cloud: &Cloud<S>,
        topology: &Topology,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let storage = work.storage;
        let frame = storage.frame();

        let d = cloud.dimension;

        let quantity = scioxide_core::quantity!("affine_transforms");
        let stride = checked_add(
            checked_mul(d, d, quantity)?,
            checked_add(checked_mul(2, d, quantity)?, 2, quantity)?,
            quantity,
        )?;
        let mut data = storage.filled(
            checked_mul(topology.facets.len(), stride, quantity)?,
            0.,
            quantity,
        )?;
        let mut status =
            storage.filled(topology.facets.len(), ApproximationStatus::Ready, quantity)?;
        storage.charge(checked_mul(
            checked_add(d * d, 2 * d, quantity)?,
            size_of::<f64>(),
            quantity,
        )?)?;
        let mut scratch = AffineWorkspace {
            matrix: Matrix::zeros(d, d)?,
            factor: LinearFactor::new(d, storage)?,
            rhs: Vector::zeros(d)?,
            solution: Vector::zeros(d)?,
        };
        for ((frame, facet), status) in data
            .chunks_exact_mut(stride)
            .zip(topology.facets.iter())
            .zip(&mut status)
        {
            work.check()?;
            if let Err(error) = Self::prepare(cloud, facet.vertices, frame, &mut scratch, storage) {
                match error.kind() {
                    scioxide_core::SciOxideErrorKind::NumericalInstability
                    | scioxide_core::SciOxideErrorKind::Singular => {
                        *status = ApproximationStatus::Failed;
                    }
                    _ => return Err(error),
                }
            }
        }
        let result = Self {
            dimension: d,
            stride,
            data,
            status,
        };
        frame.keep(0, result.retained_bytes());
        Ok(result)
    }
    fn retained_bytes(&self) -> usize {
        retained(&self.data) + retained(&self.status)
    }
    pub(super) fn at(&self, index: usize) -> Result<AffineTransform<'_>> {
        self.status[index].check()?;
        let d = self.dimension;
        let frame = &self.data[index * self.stride..(index + 1) * self.stride];
        let (origin, rest) = frame.split_at(d);
        let (inverse, rest) = rest.split_at(d * d);
        let (center, scales) = rest.split_at(d);
        Ok(AffineTransform {
            origin,
            inverse,
            center,
            scale: scales[0],
            edge_scale: scales[1],
        })
    }
    fn prepare<S: RealScalar>(
        cloud: &Cloud<S>,
        vertices: &[usize],
        frame: &mut [f64],
        scratch: &mut AffineWorkspace,
        storage: &Storage,
    ) -> Result<()> {
        let d = cloud.dimension;
        let (origin, rest) = frame.split_at_mut(d);
        let (inverse, rest) = rest.split_at_mut(d * d);
        let (center, scales) = rest.split_at_mut(d);
        let magnitude = vertices
            .iter()
            .flat_map(|&i| cloud.point(i))
            .try_fold(0_f64, |s, x| {
                Ok::<_, SciOxideError>(s.max(x.try_to_f64_exact()?.abs()))
            })?;
        if magnitude == 0. {
            return Err(SciOxideError::singular(None));
        }
        let scioxide_core::FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } = magnitude.decompose()
        else {
            return Err(scioxide_core::SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "finite scalar required",
            ));
        };
        let scale = 1_f64.scalbn(i32::from(exponent) + 63 - significand.leading_zeros() as i32);
        for (value, &x) in origin.iter_mut().zip(cloud.point(vertices[0])) {
            *value = x.try_to_f64_exact()? / scale;
        }
        let entries = scratch.matrix.as_mut_slice();
        for row in 0..d {
            for column in 0..d {
                entries[row * d + column] =
                    cloud.point(vertices[column + 1])[row].try_to_f64_exact()? / scale
                        - origin[row];
            }
        }
        let edge_scale = entries.iter().fold(0_f64, |s, x| s.max(x.abs()));
        if edge_scale == 0. {
            return Err(unstable("unrepresentable simplex differences"));
        }
        for value in entries.iter_mut() {
            *value /= edge_scale;
        }
        scratch.factor.factor(&scratch.matrix, storage)?;
        for column in 0..d {
            scratch.rhs.as_mut_slice().fill(0.);
            scratch.rhs.as_mut_slice()[column] = 1.;
            scratch.factor.solve(
                &scratch.matrix,
                &scratch.rhs,
                &mut scratch.solution,
                storage,
            )?;
            for (row, &value) in scratch.solution.as_slice().iter().enumerate() {
                inverse[row * d + column] = value;
            }
        }
        // Conditioning guard. Once the local coordinates are rounded, the
        // barycentric error scales like ||inverse|| * eps * d^2. The simplex
        // approximation is unavailable once that could exceed sqrt(eps).
        // This never rejects exact topology.
        let norm = inverse.iter().fold(0_f64, |s, x| s.max(x.abs()));
        if !norm.is_finite() || norm * f64::EPSILON * (d * d) as f64 > f64::SQRT_EPSILON {
            return Err(unstable("simplex affine coordinates are ill-conditioned"));
        }
        // With v_0 at the origin, the circumcenter c satisfies
        // e_j . c = |e_j|^2 / 2 for every edge e_j. So E^T c = s and
        // c = inverse^T s, with s_j = |e_j|^2 / 2.
        center.fill(0.);
        for j in 0..d {
            let square = (0..d)
                .map(|axis| scratch.matrix.as_slice()[axis * d + j].powi(2))
                .sum::<f64>();
            for (axis, value) in center.iter_mut().enumerate() {
                *value += inverse[j * d + axis] * (0.5 * square);
            }
        }
        scales.copy_from_slice(&[scale, edge_scale]);
        Ok(())
    }
}
impl AffineTransform<'_> {
    pub(super) fn circumcenter(&self) -> impl Iterator<Item = Result<f64>> + '_ {
        self.center
            .iter()
            .zip(self.origin)
            .map(move |(&value, &origin)| {
                let value = (value * self.edge_scale + origin) * self.scale;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(unstable("unrepresentable circumcenter"))
                }
            })
    }
}
impl<S: RealScalar> Delaunay<S> {
    /// Constructs a Delaunay with default geometry options.
    ///
    /// Uses the finite default geometry limits and the construction method and cost of [`Self::with_options`].
    ///
    /// # Errors
    /// As [`Self::with_options`].
    ///
    /// ```
    /// use scioxide_spatial::{Delaunay, PointSet};
    /// let sites = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
    /// let model = Delaunay::new(&PointSet::from_points(&sites)?)?;
    /// let found = model.find_simplex(&[0.2, 0.2])?.unwrap();
    /// let weights = model.barycentric(found.simplex, &[0.2, 0.2])?;
    /// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(points: &PointSet<'_, S>) -> Result<Self> {
        Self::with_options(points, GeometryOptions::default())
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.cloud.retained_bytes()
            + self.topology.retained_bytes()
            + self.transforms.retained_bytes()
            + self.seeds.retained_bytes()
            + self.vertex_simplices.retained_bytes()
    }

    /// Computes the Delaunay triangulation of `points`, one site per row.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the points have dimension zero, or if
    ///   [`GeometryDegeneracy::Strict`](crate::GeometryDegeneracy::Strict)
    ///   rejects duplicate sites or a triangulation that is not unique.
    /// - `Singular` if the sites do not span the space, for example when there
    ///   are fewer than `d + 1` distinct sites.
    /// - `ResourceLimit` if the construction exceeds a
    ///   [`GeometryLimits`](crate::GeometryLimits) value.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn with_options(points: &PointSet<'_, S>, options: GeometryOptions) -> Result<Self> {
        Self::prepare(points, options, None)
    }
    /// Computes the Delaunay triangulation under an execution or solve
    /// context.
    ///
    /// Work counts toward the node limit of an enclosing solve, as described
    /// in [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and:
    ///
    /// - `ExecutionStopped` on cancellation or deadline.
    /// - `ConvergenceFailure` if the node limit of the solve context is
    ///   exhausted.
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
    pub(super) fn construct(
        points: &PointSet<'_, S>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let cloud = Cloud::new(points, work.options, work.storage)?;
        Self::from_cloud(cloud, work)
    }
    fn from_cloud(cloud: Cloud<S>, work: &mut Work<'_, '_, '_, '_>) -> Result<Self> {
        let storage = work.storage;
        let before = cloud.retained_bytes();
        let frame = storage.frame();

        let options = work.options;
        let d = cloud.dimension;
        if cloud.len() < d + 1 {
            return Err(SciOxideError::singular(None));
        }
        let facets = if cloud.len() == d + 1 {
            let mut vertices =
                storage.capacity(d + 1, scioxide_core::quantity!("single_simplex"))?;
            vertices.extend(0..=d);
            let inside = work.sign(&cloud, &vertices, false, false)?;
            if inside == 0 {
                return Err(SciOxideError::singular(None));
            }
            if options.limits.max_facets == 0 {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    1,
                    options.limits.max_facets,
                ));
            }
            let mut facets = Facets::new(d + 1, 1, storage)?;
            facets.push(&vertices, inside, storage)?;
            (facets, None)
        } else {
            // Lifting map: the Delaunay simplices are the lower facets of the
            // hull of (x, |x|^2). A facet is lower when its interior witness
            // lies on the side opposite its projected orientation. The
            // projected d-dimensional orientation then becomes the
            // reference sign used by location queries.
            let lifted = hull(&cloud, true, work)?;
            let quantity = scioxide_core::quantity!("lower_facets");
            let mut lower = storage.filled(lifted.facets.len(), usize::MAX, quantity)?;
            let mut facets = Facets::new(d + 1, lifted.facets.len(), storage)?;
            for (index, facet) in lifted.facets.iter().enumerate() {
                let orientation = work.sign(&cloud, facet.vertices, false, false)?;
                if orientation != 0 && facet.inside == -orientation {
                    lower[index] = facets.len();
                    facets.push(facet.vertices, orientation, storage)?;
                }
            }
            // Lower facets keep the lifted vertex order, so lifted adjacency
            // carries over; upper neighbors become the convex boundary.
            let mut neighbors = storage.filled(
                checked_mul(facets.len(), d + 1, quantity)?,
                None,
                scioxide_core::quantity!("facet_adjacency"),
            )?;
            for (index, &simplex) in lower.iter().enumerate() {
                if simplex == usize::MAX {
                    continue;
                }
                for (slot, neighbor) in neighbors[simplex * (d + 1)..(simplex + 1) * (d + 1)]
                    .iter_mut()
                    .zip(lifted.adjacent(index)?)
                {
                    *slot = neighbor
                        .map(|neighbor| lower[neighbor])
                        .filter(|&neighbor| neighbor != usize::MAX);
                }
            }
            storage.release(lifted.retained_bytes());
            (facets, Some(neighbors))
        };
        let (facets, neighbors) = facets;
        if facets.is_empty() {
            return Err(SciOxideError::singular(None));
        }
        let topology = match neighbors {
            Some(neighbors) => Topology::from_parts(facets, neighbors),
            None => Topology::new(facets, storage)?,
        };
        if options.degeneracy == GeometryDegeneracy::Strict {
            // Adjacent cells with a zero unperturbed lifted determinant share an
            // empty sphere, so the triangulation is not unique. Ties elsewhere
            // (upper lifted facets, non-empty spheres) do not affect uniqueness.
            let groups = super::canonical::groups(&cloud, &topology, true, work)?;
            if groups.iter().enumerate().any(|(i, &group)| group != i) {
                return Err(degenerate());
            }
            storage.release(retained(&groups));
        }
        let transforms = AffineTransforms::new(&cloud, &topology, work)?;
        let seeds = location::SeedIndex::new(&cloud, &topology, work)?;
        let vertex_simplices = Incidence::from_pairs(
            cloud.len(),
            topology
                .facets
                .iter()
                .enumerate()
                .flat_map(|(simplex, facet)| {
                    facet.vertices.iter().map(move |&vertex| (vertex, simplex))
                }),
            work.context.as_deref(),
            storage,
        )?;
        let provenance = provenance(&cloud, work);
        let result = Self {
            options: work.options,
            requirements: work.storage.requirements(),
            cloud,
            topology,
            transforms,
            seeds,
            vertex_simplices,
            provenance,
            token: MeshToken::fresh(),
        };
        frame.keep(before, result.retained_bytes());
        Ok(result)
    }

    /// Returns the mesh token of this triangulation. Each construction has a
    /// new mesh family.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }
    /// Returns the coordinate dimension `d`.
    pub const fn dimension(&self) -> usize {
        self.cloud.dimension
    }
    /// Returns the site coordinates in sorted-site order, `d` values per site.
    pub fn points(&self) -> &[S] {
        &self.cloud.points
    }
    /// Returns the sites as a [`PointSet`] that borrows this triangulation,
    /// without copying or allocating.
    ///
    /// Row `i` is site `i`, so the result can index a [`KdTree`](crate::KdTree)
    /// whose neighbor indices are site indices of this triangulation.
    ///
    /// Returns the validated contiguous storage in `O(1)` time without rescanning.
    ///
    /// ```
    /// use scioxide_spatial::{Delaunay, GeometryOptions, KdTree, KdTreeOptions, PointSet, TreeMetric};
    /// let input = PointSet::from_points(&[[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]])?;
    /// let mesh = Delaunay::new(&input)?;
    /// let sites = mesh.point_set();
    /// assert_eq!(sites.coordinate(0, 0)?, 0.0);
    /// let index = KdTree::new(&sites)?;
    /// assert_eq!(index.len(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn point_set(&self) -> PointSet<'_, S> {
        PointSet::from_validated(self.points(), self.cloud.len(), self.dimension())
    }
    /// Returns the site index of each input row.
    pub fn input_mapping(&self) -> &[usize] {
        &self.cloud.mapping
    }
    /// Returns the first input row of each site.
    pub fn representatives(&self) -> &[usize] {
        &self.cloud.representatives
    }
    /// Returns the number of `d`-simplices.
    pub fn simplex_count(&self) -> usize {
        self.topology.facets.len()
    }
    /// Returns the `d + 1` site indices of a simplex in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.simplex_count()`.
    pub fn simplex(&self, index: usize) -> Result<&[usize]> {
        self.topology.vertices(index)
    }
    /// Returns, for each vertex of a simplex, the simplex across the opposite
    /// face, or `None` where that face lies on the convex hull.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.simplex_count()`.
    pub fn neighbors(&self, index: usize) -> Result<&[Option<usize>]> {
        self.topology.adjacent(index)
    }
    /// Returns the input counts and work of the construction.
    pub const fn provenance(&self) -> GeometryProvenance {
        self.provenance
    }
    /// Returns the simplices that contain a site, in ascending order.
    ///
    /// The incidence lists are built during construction, so the cost is
    /// proportional to the number of simplices returned.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex` is not a site index.
    pub fn incident_simplices(&self, vertex: usize) -> Result<impl Iterator<Item = usize> + '_> {
        if vertex >= self.cloud.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("delaunay_site"),
                vertex,
                self.cloud.len(),
            ));
        }
        Ok(self.vertex_simplices.at(vertex).iter().copied())
    }
    /// Finds a closed simplex that contains `point`.
    ///
    /// Returns `None` if the point lies outside the convex hull of the sites.
    /// A point on a face, edge or vertex shared by several simplices returns
    /// the one with the lowest index, with `boundary` set. The result does not
    /// depend on earlier queries.
    ///
    /// The search is a remembering visibility walk with exact orientation
    /// tests. It starts from the simplex found by the previous query with the
    /// same `workspace`, or from a simplex whose centroid is near the point.
    /// It visits at most `simplex_count()` simplices and then scans the rest.
    /// Queries with a reserved workspace do not allocate.
    ///
    /// ```
    /// use scioxide_spatial::{Delaunay, GeometryOptions, PointSet};
    /// let data = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
    /// let points = PointSet::from_rows(&data, 2)?;
    /// let options = GeometryOptions::new();
    /// let geometry = Delaunay::with_options(&points, options)?;
    /// let mut workspace = geometry.workspace(())?;
    /// assert!(
    ///     geometry
    ///         .find_simplex_scoped(
    ///             &[0.25, 0.25],
    ///             &mut workspace,
    ///             scioxide_core::ExecutionContext::sequential()
    ///         )?
    ///         .is_some()
    /// );
    /// assert!(
    ///     geometry
    ///         .find_simplex_scoped(
    ///             &[2.0, 2.0],
    ///             &mut workspace,
    ///             scioxide_core::ExecutionContext::sequential()
    ///         )?
    ///         .is_none()
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    pub fn find_simplex_scoped<'scope, 'capability>(
        &self,
        point: &[S],
        workspace: &mut GeometryWorkspace,
        cx: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Option<SimplexLocation>> {
        let storage = Storage::new(self.options.limits.max_storage_bytes, 0)?;
        scoped(cx, |context| {
            validate_query(point, self.dimension())?;
            let d = self.dimension();
            let mut work = Work::new(
                workspace,
                d,
                self.options,
                Some(context),
                scioxide_core::argument!("self"),
                &storage,
            )?;
            let hint = work
                .workspace
                .location
                .as_ref()
                .and_then(|(token, simplex)| (token == &self.token).then_some(*simplex));
            let seed = self.seeds.seed(point, hint, &mut work.workspace.rows)?;
            let result = self.walk(point, seed, self.simplex_count(), &mut work)?;
            match result {
                Some(location) => {
                    if let Some((token, simplex)) = &mut work.workspace.location
                        && token == &self.token
                    {
                        *simplex = location.simplex;
                    } else {
                        work.workspace.location = Some((self.token.clone(), location.simplex));
                    }
                }
                None => work.workspace.location = None,
            }
            Ok(result)
        })
    }
    /// Computes the barycentric coordinates of `point` with respect to a
    /// simplex.
    ///
    /// `output[i]` belongs to the `i`-th vertex of [`Self::simplex`], and the
    /// coordinates sum to one. `output` must have length `d + 1` and `scratch`
    /// at least `d + 1`. The point may lie outside the simplex; use
    /// [`Self::find_simplex`] to decide containment.
    ///
    /// The coordinates are rounded. They come from the inverse `A⁻¹` of the
    /// scaled edge matrix `A` of the simplex, computed in `f64` during
    /// construction. A simplex has no usable map when
    /// `max |A⁻¹ᵢⱼ| · ε · d² > √ε`, with `ε` the `f64` machine epsilon. This
    /// screens out ill-conditioned simplices but does not bound the error for
    /// points far outside the simplex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `simplex >= self.simplex_count()`.
    /// - `DimensionMismatch` if `point` or `output` has the wrong length, or
    ///   `scratch` is too short.
    /// - `NonFinite` if `point` contains NaN or infinity.
    /// - `NumericalInstability` if the simplex has no usable affine map or a
    ///   coordinate is not representable.
    pub fn barycentric_into(
        &self,
        simplex: usize,
        point: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<()> {
        validate_query(point, self.dimension())?;
        self.topology.vertices(simplex)?;
        let d = self.dimension();
        if output.len() != d + 1 {
            return Err(SciOxideError::dimension_mismatch([d + 1], [output.len()]));
        }
        if scratch.len() < d + 1 {
            return Err(SciOxideError::dimension_mismatch([d + 1], [scratch.len()]));
        }
        // lambda_1..d = inverse * local(x), and lambda_0 = 1 - sum.
        let map = self.transforms.at(simplex)?;
        let mut sum = 0.;
        for row in 0..d {
            let mut value = 0.;
            for (axis, x) in point.iter().enumerate() {
                value += map.inverse[row * d + axis]
                    * ((x.try_to_f64_exact()? / map.scale - map.origin[axis]) / map.edge_scale);
            }
            if !value.is_finite() {
                return Err(unstable("unrepresentable barycentric coordinate"));
            }
            scratch[row + 1] = S::try_from_f64_lossy(value)?;
            sum += value;
        }
        scratch[0] = S::try_from_f64_lossy(1. - sum)?;
        if scratch[..=d].iter().any(|v| !v.is_finite()) {
            return Err(unstable("unrepresentable barycentric output"));
        }
        output.copy_from_slice(&scratch[..=d]);
        Ok(())
    }

    /// Computes the gradients `∇λᵢ` of the barycentric coordinates of a
    /// simplex.
    ///
    /// The gradients are constant on the simplex. `output` holds `d + 1` rows of
    /// `d` values, one row per vertex in [`Self::simplex`] order; `scratch`
    /// needs at least `(d + 1) d` values.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `simplex >= self.simplex_count()`.
    /// - `DimensionMismatch` if `output` has the wrong length or `scratch` is too short.
    /// - `NumericalInstability` if the simplex has no usable affine map, as
    ///   described for [`Self::barycentric_into`], or a value is not
    ///   representable.
    pub fn barycentric_gradients_into(
        &self,
        simplex: usize,
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<()> {
        self.topology.vertices(simplex)?;
        let d = self.dimension();
        let count = checked_mul(d + 1, d, scioxide_core::quantity!("barycentric_gradients"))?;
        if output.len() != count {
            return Err(SciOxideError::dimension_mismatch([count], [output.len()]));
        }
        if scratch.len() < count {
            return Err(SciOxideError::dimension_mismatch([count], [scratch.len()]));
        }
        // Barycentric coordinates are affine, so their gradients are
        // constant: row j of `inverse`, divided by the combined physical
        // scale, for lambda_{j+1}. lambda_0's gradient is minus their sum.
        let map = self.transforms.at(simplex)?;
        let physical_scale = map.scale * map.edge_scale;
        if !physical_scale.is_finite() || physical_scale == 0. {
            return Err(unstable("unrepresentable barycentric gradient scale"));
        }
        for axis in 0..d {
            let mut origin = 0.;
            for row in 0..d {
                let value = map.inverse[row * d + axis] / physical_scale;
                scratch[(row + 1) * d + axis] = S::try_from_f64_lossy(value)?;
                origin -= value;
            }
            scratch[axis] = S::try_from_f64_lossy(origin)?;
        }
        if scratch[..count].iter().any(|value| !value.is_finite()) {
            return Err(unstable("unrepresentable barycentric gradient"));
        }
        output.copy_from_slice(&scratch[..count]);
        Ok(())
    }

    /// Writes the circumcenter of a simplex to `output`, which must have
    /// length `d`.
    ///
    /// The circumcenter is computed once during construction and rounded.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `simplex >= self.simplex_count()`.
    /// - `DimensionMismatch` if `output.len()` is not the dimension.
    /// - `NumericalInstability` if the simplex has no usable affine map or the
    ///   circumcenter is not representable.
    pub fn circumcenter_into(&self, simplex: usize, output: &mut [S]) -> Result<()> {
        self.topology.vertices(simplex)?;
        if output.len() != self.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                [self.dimension()],
                [output.len()],
            ));
        }
        let map = self.transforms.at(simplex)?;
        // Validate before publishing; repeating these scalar conversions needs no scratch.
        for value in map.circumcenter() {
            S::try_from_f64_lossy(value?)?;
        }
        for (slot, value) in output.iter_mut().zip(map.circumcenter()) {
            *slot = S::try_from_f64_lossy(value?)?;
        }
        Ok(())
    }
    /// Converts a planar triangulation to a [`TriangleMesh`].
    ///
    /// Vertices are the sites in sorted-site order and cells follow simplex
    /// order, with counterclockwise vertices. The mesh keeps the token of this
    /// triangulation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the dimension is not 2.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn triangle_mesh(&self) -> Result<TriangleMesh<S>> {
        if self.dimension() != 2 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "triangle conversion requires dimension two",
            ));
        }
        let mut points = try_with_capacity(
            self.cloud.len(),
            scioxide_core::quantity!("triangle_points"),
        )?;
        for point in self.points().chunks_exact(2) {
            points.push(Point::new([point[0], point[1]])?);
        }
        let mut cells = try_with_capacity(
            self.simplex_count(),
            scioxide_core::quantity!("triangle_cells"),
        )?;
        for facet in self.topology.facets.iter() {
            // `inside` is the orientation of the sorted tuple. One swap
            // gives the counterclockwise order that `TriangleMesh` requires.
            let mut cell = [facet.vertices[0], facet.vertices[1], facet.vertices[2]];
            if facet.inside < 0 {
                cell.swap(0, 1);
            }
            cells.push(cell);
        }
        Ok(TriangleMesh::new(points, cells)?.with_mesh_token(self.token.clone()))
    }
    /// Converts a three-dimensional triangulation to a [`TetrahedronMesh`].
    ///
    /// Vertices are the sites in sorted-site order and cells follow simplex
    /// order, positively oriented as [`TetrahedronMesh`] requires. The mesh
    /// keeps the token of this triangulation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the dimension is not 3.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn tetrahedron_mesh(&self) -> Result<TetrahedronMesh<S>> {
        if self.dimension() != 3 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "tetrahedron conversion requires dimension three",
            ));
        }
        let mut points = try_with_capacity(
            self.cloud.len(),
            scioxide_core::quantity!("tetrahedron_points"),
        )?;
        for point in self.points().chunks_exact(3) {
            points.push(Point::new([point[0], point[1], point[2]])?);
        }
        let mut cells = try_with_capacity(
            self.simplex_count(),
            scioxide_core::quantity!("tetrahedron_cells"),
        )?;
        for facet in self.topology.facets.iter() {
            let mut cell = [
                facet.vertices[0],
                facet.vertices[1],
                facet.vertices[2],
                facet.vertices[3],
            ];
            // `TetrahedronMesh` requires a negative orient3d (positive edge
            // determinant), which is the opposite sign convention to 2D.
            if facet.inside > 0 {
                cell.swap(0, 1);
            }
            cells.push(cell);
        }
        Ok(TetrahedronMesh::new(points, cells)?.with_mesh_token(self.token.clone()))
    }
}

impl<S: RealScalar> Delaunay<S> {
    /// Locates a point; returns `None` outside the hull.
    ///
    /// Allocates a workspace and runs [`Self::find_simplex_scoped`] with sequential native execution; it has the same cost and accuracy.
    ///
    /// # Errors
    /// As [`Self::find_simplex_scoped`], plus workspace allocation failure.
    pub fn find_simplex(&self, point: &[S]) -> Result<Option<SimplexLocation>> {
        self.find_simplex_scoped(
            point,
            &mut self.workspace(())?,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}

impl<S: RealScalar> Delaunay<S> {
    /// Returns the flat site indices, `d + 1` per row for dimension `d`.
    ///
    /// Borrows the stored topology in `O(1)` time without allocation.
    ///
    /// # Errors
    /// This accessor is infallible.
    pub fn simplices(&self) -> &[usize] {
        self.topology.facets.vertices()
    }
}

impl<S: RealScalar> Delaunay<S> {
    /// Computes the `d + 1` barycentric weights of `point` in `simplex`.
    /// Uses [`Self::barycentric_into`] with `2 (d + 1)` allocated scalars and identical accuracy.
    ///
    /// # Errors
    /// As [`Self::barycentric_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn barycentric(&self, simplex: usize, point: &[S]) -> Result<Vec<S>> {
        let n = scioxide_core::backend::checked_add(
            self.dimension(),
            1,
            scioxide_core::quantity!("barycentric_output"),
        )?;
        let len = scioxide_core::backend::checked_mul(
            n,
            2,
            scioxide_core::quantity!("barycentric_output"),
        )?;
        let mut values = scioxide_core::backend::try_filled(
            len,
            S::ZERO,
            scioxide_core::quantity!("barycentric_output"),
        )?;
        let (out, scratch) = values.split_at_mut(n);
        self.barycentric_into(simplex, point, out, scratch)?;
        values.truncate(n);
        Ok(values)
    }
}
