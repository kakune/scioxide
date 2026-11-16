//! Intersections of halfspaces.
use crate::PointSet;
use crate::incidence::Incidence;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::engine::hull;
use crate::tessellation::numeric::validate_query;
use crate::tessellation::preparation::prepare;
use crate::tessellation::work::Work;
use crate::tessellation::{GeometryConstructionRequirements, GeometryProvenance};
use crate::tessellation::{GeometryOptions, Result};
use crate::validation::inconsistent as invalid;
use alloc::vec::Vec;
use scioxide_core::RealScalar;
use scioxide_core::backend::checked_add;
use scioxide_core::backend::checked_mul;
use scioxide_core::{Argument, SciOxideError, SolveScope};
mod vertex;
/// Whether a [`HalfspaceIntersection`] is bounded.
///
/// An unbounded intersection has no vertex list; no bounding box is added to
/// close it.
///
/// ```
/// use scioxide_spatial::{GeometryOptions, HalfspaceIntersection, IntersectionKind, PointSet};
/// // The quadrant x >= 0, y >= 0 has no bounding inequality.
/// let rows = [-1.0_f64, 0.0, 0.0, 0.0, -1.0, 0.0];
/// let planes = PointSet::from_rows(&rows, 3)?;
/// let quadrant = HalfspaceIntersection::new(&planes, &[1.0, 1.0])?;
/// assert_eq!(quadrant.kind(), IntersectionKind::Unbounded);
/// assert!(quadrant.vertices().is_empty());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionKind {
    /// The intersection is a bounded polytope; its vertices and their active
    /// halfspaces are available.
    Bounded,
    /// The intersection is unbounded; no vertices are returned.
    Unbounded,
}
/// Intersection of halfspaces `a·x + b ≤ 0` in `d` dimensions, each given as
/// a row `[a₁, …, a_d, b]`.
///
/// The intersection is computed as the convex hull of the polar dual about a
/// point strictly inside every halfspace. Which halfspaces meet at each vertex,
/// and which vertices are adjacent, is decided exactly. Vertex coordinates are
/// rounded, and each is checked against its defining halfspaces: a vertex `x`
/// violates every inequality, active or not, by at most
/// `128 ε (d + 1) (|b| + Σᵢ |aᵢ xᵢ|)`, where `ε` is the machine epsilon of the
/// scalar type. In low dimensions the expected cost is `O(n log n)` for `n`
/// halfspaces, plus work proportional to the output.
///
/// ```
/// use scioxide_spatial::{GeometryOptions, HalfspaceIntersection, PointSet};
/// let rows = [1.0_f64, 0., -1., -1., 0., 0., 0., 1., -1., 0., -1., 0.];
/// let planes = PointSet::from_rows(&rows, 3)?;
/// let options = GeometryOptions::new();
/// let square = HalfspaceIntersection::with_options(&planes, &[0.5, 0.5], options)?;
/// assert_eq!(square.vertices().len(), 8);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct HalfspaceIntersection<S: RealScalar> {
    requirements: GeometryConstructionRequirements,
    dimension: usize,
    kind: IntersectionKind,
    vertices: Vec<S>,
    active: Vec<usize>,
    active_offsets: Vec<usize>,
    adjacency: Incidence,
    provenance: GeometryProvenance,
}
impl<S: RealScalar> HalfspaceIntersection<S> {
    /// Constructs a HalfspaceIntersection with default geometry options.
    ///
    /// Uses the finite default geometry limits and the construction method and cost of [`Self::with_options`].
    ///
    /// # Errors
    /// As [`Self::with_options`].
    pub fn new(halfspaces: &PointSet<'_, S>, interior: &[S]) -> Result<Self> {
        Self::with_options(halfspaces, interior, GeometryOptions::default())
    }

    /// Computes the intersection of `halfspaces`, one row `[a, b]` per
    /// halfspace, given a point `interior` strictly inside all of them.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the rows do not have `interior.len() + 1` entries.
    /// - `InvalidInput` if `interior` is not strictly inside every halfspace.
    /// - `NonFinite` if `interior` contains NaN or infinity.
    /// - `ResourceLimit` if the construction exceeds a
    ///   [`GeometryLimits`](crate::GeometryLimits) value.
    /// - `NumericalInstability` if a vertex fails its error check.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn with_options(
        halfspaces: &PointSet<'_, S>,
        interior: &[S],
        options: GeometryOptions,
    ) -> Result<Self> {
        Self::prepare(halfspaces, interior, options, None)
    }
    /// Computes the intersection under an execution or solve context.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `ExecutionStopped` or `ConvergenceFailure` as for
    /// [`Delaunay::build`](crate::Delaunay::build).
    pub fn build<'scope, 'capability>(
        halfspaces: &PointSet<'_, S>,
        interior: &[S],
        options: GeometryOptions,
        cx: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        Self::prepare(halfspaces, interior, options, Some(cx.into()))
    }
    fn prepare<'scope, 'capability>(
        halfspaces: &PointSet<'_, S>,
        interior: &[S],
        options: GeometryOptions,
        scope: Option<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        let coefficients = checked_add(
            interior.len(),
            1,
            scioxide_core::quantity!("halfspace_coefficients"),
        )?;
        if halfspaces.dimension() != coefficients {
            return Err(SciOxideError::dimension_mismatch(
                [coefficients],
                [halfspaces.dimension()],
            ));
        }
        prepare(
            interior.len(),
            halfspaces.len(),
            scioxide_core::argument!("interior"),
            options,
            scope,
            |work| Self::construct(halfspaces, interior, work),
        )
    }
    /// Returns the peak storage used during construction.
    pub const fn construction_requirements(&self) -> GeometryConstructionRequirements {
        self.requirements
    }
    fn construct(
        halfspaces: &PointSet<'_, S>,
        interior: &[S],
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        const HALFSPACES: Argument = scioxide_core::argument!("halfspaces");
        const INTERIOR: Argument = scioxide_core::argument!("interior");
        let d = interior.len();
        validate_query(interior, d)?;
        let storage = work.storage;
        // Each row [a | b] is the halfspace a . x + b <= 0. The bordered
        // determinant det([[I, p], [-a, b]]) equals a . p + b, so its exact
        // sign certifies that the interior point p satisfies every
        // inequality strictly, with no rounded residual. Each halfspace
        // becomes the homogeneous dual point [a, -b]. Polar duality about p
        // turns intersection vertices into facets of the dual hull.
        let n = halfspaces.len();
        let mut homogeneous = storage.capacity(
            checked_mul(
                n,
                d + 1,
                scioxide_core::quantity!("dual_homogeneous_points"),
            )?,
            scioxide_core::quantity!("dual_homogeneous_points"),
        )?;
        for i in 0..n {
            work.iteration()?;
            work.workspace.rows[..(d + 1) * (d + 1)].fill(0.);
            for (axis, coordinate) in interior.iter().enumerate() {
                work.workspace.rows[axis * (d + 1) + axis] = 1.;
                work.workspace.rows[axis * (d + 1) + d] = coordinate.try_to_f64_exact()?;
                work.workspace.rows[d * (d + 1) + axis] =
                    -halfspaces.coordinate(i, axis)?.try_to_f64_exact()?;
            }
            work.workspace.rows[d * (d + 1) + d] =
                halfspaces.coordinate(i, d)?.try_to_f64_exact()?;
            if work.determinant(d + 1, d + 1, false, true)? >= 0 {
                return Err(invalid(INTERIOR, "supplied point is not strictly interior"));
            }
            for axis in 0..d {
                homogeneous.push(halfspaces.coordinate(i, axis)?.try_to_f64_exact()?);
            }
            homogeneous.push(-halfspaces.coordinate(i, d)?.try_to_f64_exact()?);
        }
        let mut mapping = storage.capacity(n, scioxide_core::quantity!("polar_mapping"))?;
        mapping.extend(0..n);
        let cloud = Cloud {
            dimension: d,
            filter_envelope: homogeneous.chunks_exact(d + 1).all(|row| {
                row[..d]
                    .iter()
                    .copied()
                    .all(crate::predicates::fixed_in_range)
            }),
            // Homogeneous mode classifies original coefficients directly; no
            // rounded polar coordinates are constructed or consumed.
            points: Vec::<f64>::new(),
            representatives: storage
                .copy(&mapping, scioxide_core::quantity!("polar_representatives"))?,
            mapping,
            homogeneous: Some(homogeneous),
        };
        let unbounded = |terms| Self {
            requirements: storage.requirements(),
            dimension: d,
            kind: IntersectionKind::Unbounded,
            vertices: Vec::new(),
            active: Vec::new(),
            active_offsets: Vec::new(),
            adjacency: Incidence::default(),
            provenance: GeometryProvenance {
                input_points: n,
                unique_points: n,
                predicate_terms: terms,
                canonical_work: 0,
                resolved_ties: false,
            },
        };
        let topology = match hull(&cloud, false, work) {
            Ok(value) => value,
            Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular => {
                return Ok(unbounded(work.workspace.terms));
            }
            Err(error) => return Err(error),
        };
        // Positive row denominators preserve determinant sign. The homogeneous
        // dual rows [a,-b] are exact even when the chosen interior is nonzero:
        // replacing -b by -b-a·interior is a determinant-preserving column operation.
        // The intersection is bounded only if the dual hull strictly contains
        // the dual origin, the homogeneous point [0, ..., 0, 1]. Otherwise
        // some facet has the origin on or beyond it, and the corresponding
        // primal "vertex" is at infinity.
        let homogeneous = cloud
            .homogeneous
            .as_ref()
            .ok_or_else(|| invalid(HALFSPACES, "missing dual coefficients"))?;
        for facet in topology.facets.iter() {
            for (row, &plane) in facet.vertices.iter().enumerate() {
                work.workspace.rows[row * (d + 1)..(row + 1) * (d + 1)]
                    .copy_from_slice(&homogeneous[plane * (d + 1)..(plane + 1) * (d + 1)]);
            }
            work.workspace.rows[d * (d + 1)..(d + 1) * (d + 1)].fill(0.);
            work.workspace.rows[(d + 1) * (d + 1) - 1] = 1.;
            let sign = work.determinant(d + 1, d + 1, false, true)?;
            if sign == 0 || sign != facet.inside {
                return Ok(unbounded(work.workspace.terms));
            }
        }
        let mut vertices = storage.capacity(
            checked_mul(
                topology.facets.len(),
                d,
                scioxide_core::quantity!("intersection_vertices"),
            )?,
            scioxide_core::quantity!("intersection_vertices"),
        )?;
        let mut active = storage.capacity(
            topology.boundary_points.len(),
            scioxide_core::quantity!("intersection_incidences"),
        )?;
        let mut active_offsets = storage.capacity(
            checked_add(
                topology.facets.len(),
                1,
                scioxide_core::quantity!("intersection_offsets"),
            )?,
            scioxide_core::quantity!("intersection_offsets"),
        )?;
        active_offsets.push(0);
        let mut facet_vertices = storage.filled(
            topology.facets.len(),
            0,
            scioxide_core::quantity!("intersection_dual_map"),
        )?;
        let groups = super::canonical::groups(&cloud, &topology, false, work)?;
        let mut incidence = storage.capacity(
            topology.boundary_points.len(),
            scioxide_core::quantity!("active_planes"),
        )?;
        incidence.extend(
            topology
                .boundary_points
                .iter()
                .map(|&(facet, plane)| (groups[facet], plane)),
        );
        incidence.sort_unstable();
        incidence.dedup();
        let mut vertex_workspace = vertex::VertexWorkspace::new(d, storage)?;
        let mut offset = 0;
        for (facet_index, facet) in topology.facets.iter().enumerate() {
            work.check()?;
            if groups[facet_index] != facet_index {
                facet_vertices[facet_index] = facet_vertices[groups[facet_index]];
                continue;
            }
            let start = offset;
            while offset < incidence.len() && incidence[offset].0 == facet_index {
                offset += 1;
            }
            facet_vertices[facet_index] = active_offsets.len() - 1;
            vertex_workspace.solve(homogeneous, facet.vertices, &mut vertices, work)?;
            // Exact incidence identifies every tight original plane. The
            // componentwise vertex certificate already bounds their residuals,
            // together with all non-active inequalities; no rounded recheck is
            // needed to publish these indices.
            for &(_, plane) in &incidence[start..offset] {
                work.check()?;
                active.push(plane);
            }
            active_offsets.push(active.len());
        }
        let dual_vertices = &facet_vertices;
        let adjacency = Incidence::from_pairs(
            vertices.len() / d,
            facet_vertices
                .iter()
                .zip(topology.neighbors.chunks_exact(d))
                .flat_map(|(&vertex, neighbors)| {
                    neighbors.iter().flatten().filter_map(move |&neighbor| {
                        let other = dual_vertices[neighbor];
                        (vertex != other).then_some((vertex, other))
                    })
                }),
            work.context.as_deref(),
            storage,
        )?;
        Ok(Self {
            requirements: storage.requirements(),
            dimension: d,
            kind: IntersectionKind::Bounded,
            vertices,
            active,
            active_offsets,
            adjacency,
            provenance: GeometryProvenance {
                input_points: n,
                unique_points: n,
                predicate_terms: work.workspace.terms,
                canonical_work: work.canonical_work,
                resolved_ties: false,
            },
        })
    }
    /// Returns the input counts and work of the construction. Redundant and
    /// repeated inequalities are counted separately.
    pub const fn provenance(&self) -> GeometryProvenance {
        self.provenance
    }
    /// Returns the vertices joined to `vertex` by an edge, in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex` is not a vertex index.
    pub fn neighbors(&self, vertex: usize) -> Result<impl Iterator<Item = usize> + '_> {
        self.adjacency
            .neighbors(vertex, scioxide_core::quantity!("geometry_vertex"))
    }
    /// Returns the coordinate dimension `d`.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns whether the intersection is bounded.
    pub const fn kind(&self) -> IntersectionKind {
        self.kind
    }
    /// Returns the rounded vertex coordinates, `d` values per vertex. Empty
    /// for an unbounded intersection.
    pub fn vertices(&self) -> &[S] {
        &self.vertices
    }
    /// Returns the input rows of the halfspaces whose boundaries pass through
    /// a vertex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex` is not a vertex index.
    pub fn active_halfspaces(&self, vertex: usize) -> Result<&[usize]> {
        self.active_offsets
            .windows(2)
            .nth(vertex)
            .map(|range| &self.active[range[0]..range[1]])
            .ok_or_else(|| {
                SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("intersection_vertex"),
                    vertex,
                    self.vertices.len() / self.dimension,
                )
            })
    }
}
