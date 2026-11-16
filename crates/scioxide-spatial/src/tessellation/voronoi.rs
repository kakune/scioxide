//! Euclidean Voronoi diagrams: vertices, ridges, rays and regions.
use super::delaunay::ApproximationStatus;
use crate::PointSet;
use crate::incidence::Incidence;
use crate::tessellation::GeometryOptions;
use crate::tessellation::delaunay::Delaunay;
use crate::tessellation::numeric::{convert, normalize};
use crate::tessellation::preparation::prepare;
use crate::tessellation::storage::retained;
use crate::tessellation::topology::{Facets, Topology};
use crate::tessellation::work::Work;
use crate::tessellation::work::provenance;
use crate::tessellation::{GeometryConstructionRequirements, GeometryProvenance, Result, unstable};
use alloc::vec::Vec;
use scioxide_core::backend::checked_add;
use scioxide_core::backend::checked_mul;
use scioxide_core::{RealScalar, SciOxideError, SolveScope};
/// A ridge of a Voronoi diagram: the `(d − 1)`-dimensional face shared by the
/// regions of two sites.
///
/// Two sites have a ridge when their Delaunay edge is not a diagonal inside a
/// group of cospherical simplices.
///
/// ```
/// use scioxide_core::{ExecutionContext, };
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let points = scioxide_spatial::PointSet::from_rows(&sites, 2)?;
/// let diagram = scioxide_spatial::Voronoi::with_options(&points, scioxide_spatial::GeometryOptions::new())?;
/// // Four interior-site ridges plus four hull ridges; no square diagonal.
/// assert_eq!(diagram.ridges().len(), 8);
/// assert_eq!(
///     diagram
///         .ridges()
///         .filter(|ridge| ridge.is_unbounded())
///         .count(),
///     4
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VoronoiRidge<'a> {
    sites: [usize; 2],
    vertices: &'a [usize],
    unbounded: bool,
}
impl<'a> VoronoiRidge<'a> {
    /// Returns the two sites whose regions share this ridge, in ascending
    /// order.
    pub const fn sites(&self) -> [usize; 2] {
        self.sites
    }
    /// Returns the Voronoi vertices of this ridge.
    pub fn vertices(&self) -> &'a [usize] {
        self.vertices
    }
    /// Returns whether the ridge is unbounded.
    pub const fn is_unbounded(&self) -> bool {
        self.unbounded
    }
}
/// An unbounded ray of a Voronoi diagram, starting at a Voronoi vertex.
///
/// Each ray is dual to a face of the convex hull of the sites and points away
/// from the hull. Hull facets that lie on one exact hyperplane and share a
/// Voronoi vertex give a single ray.
///
/// ```
/// use scioxide_core::{ExecutionContext, };
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let points = scioxide_spatial::PointSet::from_rows(&sites, 2)?;
/// let diagram = scioxide_spatial::Voronoi::with_options(&points, scioxide_spatial::GeometryOptions::new())?;
/// // One ray per hull edge, each separating that edge's two sites.
/// assert_eq!(diagram.rays().len(), 4);
/// for ray in diagram.rays() {
///     assert_eq!(ray.sites().len(), 2);
///     let direction = ray.direction()?;
///     assert!((direction[0].hypot(direction[1]) - 1.0).abs() < 1e-15);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VoronoiRay<'a, S: RealScalar> {
    vertex: usize,
    sites: &'a [usize],
    direction: &'a [S],
    status: ApproximationStatus,
}
impl<'a, S: RealScalar> VoronoiRay<'a, S> {
    /// Returns the Voronoi vertex where the ray starts.
    pub const fn vertex(&self) -> usize {
        self.vertex
    }
    /// Returns, in ascending order, the sites that are equidistant from every
    /// point of the ray: the at least `d` sites of its hull face.
    pub fn sites(&self) -> &'a [usize] {
        self.sites
    }
    /// Returns the rounded outward unit direction of the ray.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the direction could not be computed
    ///   reliably for this ray.
    pub fn direction(&self) -> Result<&'a [S]> {
        self.status.check()?;
        Ok(self.direction)
    }
}
/// Euclidean Voronoi diagram of a set of sites in `d` dimensions.
///
/// The region of a site is the set of points at least as close to it as to any
/// other site. The diagram is computed as the dual of the [`Delaunay`]
/// triangulation, which it keeps. Each Voronoi vertex is the circumcenter of a
/// group of cospherical Delaunay simplices, so four cocircular sites in the
/// plane give one vertex, not two.
///
/// Vertices, ridges, rays and regions are exact. Vertex coordinates and ray
/// directions are rounded; if one cannot be computed reliably, only the
/// methods that return it fail, and [`Self::vertex_count`] and the incidence
/// queries stay usable.
///
/// ```
/// use scioxide_core::{ExecutionContext, };
/// let sites = [0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.4];
/// let points = scioxide_spatial::PointSet::from_rows(&sites, 2)?;
/// let diagram = scioxide_spatial::Voronoi::with_options(&points, scioxide_spatial::GeometryOptions::new())?;
/// assert_eq!(diagram.vertex_count(), 4);
/// // Row 4, the interior site, has a bounded region; hull sites do not.
/// let interior = diagram.delaunay().input_mapping()[4];
/// assert!(!diagram.region_is_unbounded(interior)?);
/// assert_eq!(diagram.region_vertices(interior)?.count(), 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct Voronoi<S: RealScalar> {
    requirements: GeometryConstructionRequirements,
    delaunay: Delaunay<S>,
    vertices: Vec<S>,
    vertex_status: Vec<ApproximationStatus>,
    // First unavailable vertex, recorded once for bulk access.
    unavailable_vertex: Option<usize>,
    simplex_vertices: Vec<usize>,
    region_vertices: Incidence,
    unbounded_regions: Vec<bool>,
    ridges: Vec<Ridge>,
    ridge_vertices: Vec<usize>,
    ray_vertices: Vec<usize>,
    ray_site_offsets: Vec<usize>,
    ray_sites: Vec<usize>,
    ray_directions: Vec<S>,
    ray_status: Vec<ApproximationStatus>,
    provenance: GeometryProvenance,
}
#[derive(Debug)]
struct Ridge {
    sites: [usize; 2],
    vertices: core::ops::Range<usize>,
    unbounded: bool,
}
impl<S: RealScalar> Voronoi<S> {
    /// Constructs a Voronoi with default geometry options.
    ///
    /// Uses the finite default geometry limits and the construction method and cost of [`Self::with_options`].
    ///
    /// # Errors
    /// As [`Self::with_options`].
    pub fn new(points: &PointSet<'_, S>) -> Result<Self> {
        Self::with_options(points, GeometryOptions::default())
    }

    fn retained_bytes(&self) -> usize {
        self.delaunay.retained_bytes()
            + self.region_vertices.retained_bytes()
            + retained(&self.vertices)
            + retained(&self.vertex_status)
            + retained(&self.simplex_vertices)
            + retained(&self.unbounded_regions)
            + retained(&self.ridges)
            + retained(&self.ridge_vertices)
            + retained(&self.ray_vertices)
            + retained(&self.ray_site_offsets)
            + retained(&self.ray_sites)
            + retained(&self.ray_directions)
            + retained(&self.ray_status)
    }

    /// Computes the Voronoi diagram of `points`, one site per row.
    ///
    /// # Errors
    ///
    /// As [`Delaunay::new`].
    pub fn with_options(points: &PointSet<'_, S>, options: GeometryOptions) -> Result<Self> {
        Self::prepare(points, options, None)
    }
    /// Computes the Voronoi diagram under an execution or solve context.
    ///
    /// # Errors
    ///
    /// As [`Delaunay::build`].
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
        let delaunay = Delaunay::construct(points, work)?;
        Self::from_delaunay(delaunay, work)
    }
    fn from_delaunay(delaunay: Delaunay<S>, work: &mut Work<'_, '_, '_, '_>) -> Result<Self> {
        let storage = work.storage;
        let before = delaunay.retained_bytes();
        let frame = storage.frame();

        let d = delaunay.dimension();
        let count = delaunay.simplex_count();
        // Adjacent simplices that are exactly cospherical (unperturbed
        // insphere zero) share one circumcenter, and so one Voronoi vertex.
        // They are merged into groups labeled by the smallest simplex id.
        // Perturbation made the triangulation unique; this undoes the
        // arbitrary split for the dual.
        let groups = super::canonical::groups(&delaunay.cloud, &delaunay.topology, true, work)?;
        let mut vertices = storage.capacity(
            checked_mul(count, d, scioxide_core::quantity!("voronoi_coordinates"))?,
            scioxide_core::quantity!("voronoi_coordinates"),
        )?;
        let mut simplex_vertices =
            storage.filled(count, 0, scioxide_core::quantity!("dual_vertex_map"))?;
        let mut vertex_status =
            storage.capacity(count, scioxide_core::quantity!("dual_vertex_status"))?;
        for i in 0..count {
            if groups[i] == i {
                simplex_vertices[i] = vertices.len() / d;
                vertices.extend(core::iter::repeat_n(S::ZERO, d));
                vertex_status.push(ApproximationStatus::Failed);
            } else {
                simplex_vertices[i] = simplex_vertices[groups[i]];
            }
            let vertex = simplex_vertices[i];
            if matches!(vertex_status[vertex], ApproximationStatus::Failed)
                && let Ok(map) = delaunay.transforms.at(i)
                && map
                    .circumcenter()
                    .all(|value| value.and_then(convert::<S>).is_ok())
            {
                for (slot, value) in vertices[vertex * d..(vertex + 1) * d]
                    .iter_mut()
                    .zip(map.circumcenter())
                {
                    *slot = convert(value?)?;
                }
                vertex_status[vertex] = ApproximationStatus::Ready;
            }
        }
        let region_vertices = Incidence::from_pairs(
            delaunay.cloud.len(),
            delaunay
                .topology
                .facets
                .iter()
                .enumerate()
                .flat_map(|(simplex, facet)| {
                    let vertex = simplex_vertices[simplex];
                    facet.vertices.iter().map(move |&site| (site, vertex))
                }),
            work.context.as_deref(),
            storage,
        )?;
        let mut unbounded_regions = storage.filled(
            delaunay.cloud.len(),
            false,
            scioxide_core::quantity!("unbounded_regions"),
        )?;
        // Boundary facets, in simplex and opposite-vertex order, and their exact
        // coplanar hull-face groups. Facets of one hull face whose cells share
        // a grouped Voronoi vertex have the same start and outward normal, so
        // they describe one ray; its equidistant sites are their union.
        let (boundary, boundary_groups, origins) = boundary_faces(&delaunay, work)?;
        let rays_quantity = scioxide_core::quantity!("voronoi_rays");
        let mut keys = storage.capacity(origins.len(), rays_quantity)?;
        for (face, &(simplex, _)) in origins.iter().enumerate() {
            keys.push((boundary_groups[face], simplex_vertices[simplex], face));
        }
        keys.sort_unstable();
        let mut runs = storage.capacity(keys.len(), rays_quantity)?;
        let mut begin = 0;
        while begin < keys.len() {
            let mut end = begin + 1;
            while end < keys.len() && keys[end].0 == keys[begin].0 && keys[end].1 == keys[begin].1 {
                end += 1;
            }
            runs.push((keys[begin].2, begin..end));
            begin = end;
        }
        runs.sort_unstable_by_key(|(first, _)| *first);
        let mut ray_vertices = storage.capacity(runs.len(), rays_quantity)?;
        let mut ray_site_offsets = storage.capacity(runs.len() + 1, rays_quantity)?;
        ray_site_offsets.push(0);
        let mut ray_sites = Vec::new();
        let mut ray_directions = Vec::new();
        let mut ray_status = storage.capacity(runs.len(), rays_quantity)?;
        let mut direction = storage.filled(d, 0., scioxide_core::quantity!("dual_direction"))?;
        for (_, run) in &runs {
            work.check()?;
            let start = ray_sites.len();
            for &(_, _, face) in &keys[run.clone()] {
                let sites = boundary.facets.at(face).vertices;
                storage.reserve(&mut ray_sites, sites.len())?;
                ray_sites.extend_from_slice(sites);
            }
            ray_sites[start..].sort_unstable();
            let mut write = start;
            for read in start..ray_sites.len() {
                if write == start || ray_sites[read] != ray_sites[write - 1] {
                    ray_sites[write] = ray_sites[read];
                    unbounded_regions[ray_sites[read]] = true;
                    write += 1;
                }
            }
            ray_sites.truncate(write);
            ray_site_offsets.push(write);
            // A boundary facet opposite `omitted` has outward direction
            // -grad(lambda_omitted), which is row (omitted - 1) of `inverse`,
            // negated. For omitted = 0 it is the sum of all rows, since
            // lambda_0 = 1 - sum. The local frame differs from physical
            // coordinates only by positive scales, so the normalized direction
            // carries over. The lowest merged facet with a usable frame decides.
            let mut status = ApproximationStatus::Failed;
            for &(_, _, face) in &keys[run.clone()] {
                let (simplex, omitted) = origins[face];
                if let Ok(map) = delaunay.transforms.at(simplex) {
                    for (axis, value) in direction.iter_mut().enumerate() {
                        *value = if omitted == 0 {
                            (0..d).map(|row| map.inverse[row * d + axis]).sum()
                        } else {
                            -map.inverse[(omitted - 1) * d + axis]
                        };
                    }
                    if normalize(&mut direction).is_ok()
                        && direction.iter().all(|&x| convert::<S>(x).is_ok())
                    {
                        status = ApproximationStatus::Ready;
                        break;
                    }
                }
            }
            if matches!(status, ApproximationStatus::Failed) {
                direction.fill(0.);
            }
            ray_status.push(status);
            storage.reserve(&mut ray_directions, d)?;
            for &x in &direction {
                ray_directions.push(convert(x)?);
            }
            ray_vertices.push(keys[run.start].1);
        }
        // Voronoi ridges are dual to Delaunay edges. The ridge between two
        // sites is spanned by the (grouped) vertices of every simplex
        // containing both. It is unbounded exactly when some boundary ray
        // also separates both sites, meaning the edge lies on the hull.
        let mut edges = storage.capacity(
            checked_mul(
                checked_mul(count, d, scioxide_core::quantity!("dual_edges"))?,
                d + 1,
                scioxide_core::quantity!("dual_edges"),
            )? / 2,
            scioxide_core::quantity!("dual_edges"),
        )?;
        for (simplex, facet) in delaunay.topology.facets.iter().enumerate() {
            for (i, &a) in facet.vertices.iter().enumerate() {
                for &b in &facet.vertices[i + 1..] {
                    edges.push(([a, b], simplex_vertices[simplex]));
                }
            }
        }
        edges.sort_unstable();
        let mut unbounded_pairs = storage.capacity(
            checked_mul(
                checked_mul(
                    boundary.facets.len(),
                    d,
                    scioxide_core::quantity!("boundary_edges"),
                )?,
                d,
                scioxide_core::quantity!("boundary_edges"),
            )?,
            scioxide_core::quantity!("boundary_edges"),
        )?;
        for facet in boundary.facets.iter() {
            for (i, &a) in facet.vertices.iter().enumerate() {
                for &b in &facet.vertices[i + 1..] {
                    unbounded_pairs.push([a, b]);
                }
            }
        }
        unbounded_pairs.sort_unstable();
        unbounded_pairs.dedup();
        let geometric_edges =
            geometric_edges(&delaunay, &groups, &boundary, &boundary_groups, work)?;
        let mut ridges =
            storage.capacity(edges.len(), scioxide_core::quantity!("voronoi_ridges"))?;
        let mut ridge_vertices =
            storage.capacity(edges.len(), scioxide_core::quantity!("ridge_vertices"))?;
        let mut begin = 0;
        while begin < edges.len() {
            work.check()?;
            let sites = edges[begin].0;
            let mut end = begin + 1;
            while end < edges.len() && edges[end].0 == sites {
                end += 1;
            }
            if geometric_edges.binary_search(&sites).is_err() {
                begin = end;
                continue;
            }
            let start = ridge_vertices.len();
            for &(_, vertex) in &edges[begin..end] {
                if ridge_vertices.len() == start || ridge_vertices.last() != Some(&vertex) {
                    ridge_vertices.push(vertex);
                }
            }
            ridges.push(Ridge {
                sites,
                vertices: start..ridge_vertices.len(),
                unbounded: unbounded_pairs.binary_search(&sites).is_ok(),
            });
            begin = end;
        }
        let provenance = provenance(&delaunay.cloud, work);
        let unavailable_vertex = vertex_status
            .iter()
            .position(|status| matches!(status, ApproximationStatus::Failed));
        let result = Self {
            requirements: storage.requirements(),
            delaunay,
            vertices,
            vertex_status,
            unavailable_vertex,
            simplex_vertices,
            region_vertices,
            unbounded_regions,
            ridges,
            ridge_vertices,
            ray_vertices,
            ray_site_offsets,
            ray_sites,
            ray_directions,
            ray_status,
            provenance,
        };
        frame.keep(before, result.retained_bytes());
        Ok(result)
    }
    /// Returns the Delaunay triangulation the diagram was computed from.
    pub const fn delaunay(&self) -> &Delaunay<S> {
        &self.delaunay
    }
    /// Returns the input counts and work of the whole construction. The
    /// [`Delaunay::provenance`] of [`Self::delaunay`] covers only the
    /// triangulation.
    pub const fn provenance(&self) -> GeometryProvenance {
        self.provenance
    }
    /// Returns the number of Voronoi vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertex_status.len()
    }
    /// Returns the coordinates of all Voronoi vertices, `d` values per vertex.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if any vertex could not be computed. Use
    ///   [`Self::vertex`] to read the others.
    pub fn vertices(&self) -> Result<&[S]> {
        if let Some(vertex) = self.unavailable_vertex {
            self.vertex_status[vertex].check()?;
        }
        Ok(&self.vertices)
    }
    /// Returns the rounded coordinates of one Voronoi vertex.
    ///
    /// The vertex is the circumcenter of the lowest-indexed simplex in its
    /// cospherical group that has a usable affine map.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.vertex_count()`.
    /// - `NumericalInstability` if no simplex of the group has a usable
    ///   circumcenter.
    pub fn vertex(&self, index: usize) -> Result<&[S]> {
        if index >= self.vertex_count() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("voronoi_vertex"),
                index,
                self.vertex_count(),
            ));
        }
        self.vertex_status[index].check()?;
        let d = self.delaunay.dimension();
        Ok(&self.vertices[index * d..(index + 1) * d])
    }
    /// Returns the Voronoi vertex of each Delaunay simplex. Cospherical
    /// simplices share a vertex.
    pub fn simplex_vertices(&self) -> &[usize] {
        &self.simplex_vertices
    }
    /// Returns the ridges, ordered by their pair of sites.
    ///
    /// ```
    /// use scioxide_spatial::{GeometryOptions, PointSet, Voronoi};
    /// let values = [0., 0., 1., 0., 0., 1.];
    /// let points = PointSet::from_rows(&values, 2)?;
    /// let options = GeometryOptions::new();
    /// let diagram = Voronoi::with_options(&points, options)?;
    /// assert_eq!(diagram.vertex_count(), 1);
    /// assert_eq!(diagram.vertex(0)?.len(), 2);
    /// assert_eq!(diagram.ridges().len(), 3);
    /// for ridge in diagram.ridges() {
    ///     assert_eq!(ridge.vertices(), &[0]);
    ///     assert!(ridge.is_unbounded());
    /// }
    /// for ray in diagram.rays() {
    ///     assert_eq!(ray.direction()?.len(), 2);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn ridges(
        &self,
    ) -> impl ExactSizeIterator<Item = VoronoiRidge<'_>> + DoubleEndedIterator + '_ {
        self.ridges.iter().map(|ridge| VoronoiRidge {
            sites: ridge.sites,
            vertices: &self.ridge_vertices[ridge.vertices.clone()],
            unbounded: ridge.unbounded,
        })
    }
    /// Returns the unbounded rays, ordered by the first Delaunay simplex and
    /// opposite vertex of their hull face. See [`Self::ridges`] for an example.
    pub fn rays(
        &self,
    ) -> impl ExactSizeIterator<Item = VoronoiRay<'_, S>> + DoubleEndedIterator + '_ {
        let d = self.delaunay.dimension();
        self.ray_vertices
            .iter()
            .enumerate()
            .map(move |(i, &vertex)| VoronoiRay {
                vertex,
                sites: &self.ray_sites[self.ray_site_offsets[i]..self.ray_site_offsets[i + 1]],
                direction: &self.ray_directions[i * d..(i + 1) * d],
                status: self.ray_status[i],
            })
    }
    /// Returns the Voronoi vertices of a site's region in ascending index
    /// order, not in angular order.
    ///
    /// The cost is proportional to the number of vertices returned. Two
    /// regions that share a vertex need not share a ridge.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `site` is not a site index.
    pub fn region_vertices(&self, site: usize) -> Result<impl Iterator<Item = usize> + '_> {
        self.validate_site(site)?;
        Ok(self.region_vertices.at(site).iter().copied())
    }
    /// Returns whether a site's region is unbounded, which holds exactly for
    /// sites on the boundary of the convex hull. The cost is O(1).
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `site` is not a site index.
    pub fn region_is_unbounded(&self, site: usize) -> Result<bool> {
        self.validate_site(site)?;
        Ok(self.unbounded_regions[site])
    }
    fn validate_site(&self, site: usize) -> Result<()> {
        if site >= self.delaunay.cloud.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("voronoi_site"),
                site,
                self.delaunay.cloud.len(),
            ));
        }
        Ok(())
    }
}

// Boundary facets in simplex and opposite-vertex order, with their origins and
// exact coplanar supporting-face groups.
#[allow(clippy::type_complexity)]
fn boundary_faces<S: RealScalar>(
    delaunay: &Delaunay<S>,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<(Topology, Vec<usize>, Vec<(usize, usize)>)> {
    let storage = work.storage;
    let frame = storage.frame();
    let d = delaunay.dimension();
    let quantity = scioxide_core::quantity!("boundary_faces");
    let mut boundary = Facets::new(d, 0, storage)?;
    let mut origins = Vec::new();
    let mut face = storage.capacity(d, quantity)?;
    for (simplex, facet) in delaunay.topology.facets.iter().enumerate() {
        work.check()?;
        for (omitted, neighbor) in delaunay.topology.adjacent(simplex)?.iter().enumerate() {
            if neighbor.is_none() {
                face.clear();
                face.extend(
                    facet
                        .vertices
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &site)| (i != omitted).then_some(site)),
                );
                boundary.push(&face, 0, storage)?;
                storage.reserve(&mut origins, 1)?;
                origins.push((simplex, omitted));
            }
        }
    }
    let boundary = Topology::new(boundary, storage)?;
    let groups = super::canonical::groups(&delaunay.cloud, &boundary, false, work)?;
    frame.keep(
        0,
        boundary.retained_bytes() + retained(&groups) + retained(&origins),
    );
    Ok((boundary, groups, origins))
}

// The smallest face containing a pair is the intersection of its incident
// maximal Delaunay cells and outer supporting faces. It is an edge exactly
// when its vertex set is the pair. Supporting planes, rather than boundary
// triangles, are essential: a cube face diagonal belongs to two triangles
// but only one geometric supporting face.
fn geometric_edges<S: RealScalar>(
    delaunay: &Delaunay<S>,
    groups: &[usize],
    boundary: &Topology,
    boundary_groups: &[usize],
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<Vec<[usize; 2]>> {
    let storage = work.storage;
    let frame = storage.frame();
    let quantity = scioxide_core::quantity!("geometric_dual_edges");
    let count = delaunay.simplex_count();
    let membership = Incidence::from_pairs(
        checked_add(count, boundary.facets.len(), quantity)?,
        delaunay
            .topology
            .facets
            .iter()
            .enumerate()
            .flat_map(|(i, facet)| facet.vertices.iter().map(move |&site| (groups[i], site)))
            .chain(boundary.facets.iter().enumerate().flat_map(|(i, facet)| {
                let group = count + boundary_groups[i];
                facet.vertices.iter().map(move |&site| (group, site))
            })),
        work.context.as_deref(),
        storage,
    )?;
    let mut records = Vec::new();
    for (i, facet) in delaunay.topology.facets.iter().enumerate().chain(
        boundary
            .facets
            .iter()
            .enumerate()
            .map(|(i, facet)| (count + i, facet)),
    ) {
        work.check()?;
        let group = if i < count {
            groups[i]
        } else {
            count + boundary_groups[i - count]
        };
        for (j, &a) in facet.vertices.iter().enumerate() {
            for &b in &facet.vertices[j + 1..] {
                work.charge_canonical(1)?;
                storage.reserve(&mut records, 1)?;
                records.push(([a, b], group));
            }
        }
    }
    records.sort_unstable();
    records.dedup();
    let mut ranges = storage.capacity(records.len(), quantity)?;
    let mut begin = 0;
    while begin < records.len() {
        let mut end = begin + 1;
        while end < records.len() && records[end].0 == records[begin].0 {
            end += 1;
        }
        ranges.push(begin..end);
        begin = end;
    }
    // Equal incidence signatures describe the same smallest face. Evaluate
    // their intersection once, including large cospherical face diagonals.
    let signature = |i: usize| records[ranges[i].clone()].iter().map(|&(_, group)| group);
    let mut order = storage.capacity(ranges.len(), quantity)?;
    order.extend(0..ranges.len());
    order.sort_unstable_by(|&a, &b| signature(a).cmp(signature(b)));
    let mut accepted = storage.filled(ranges.len(), false, quantity)?;
    let mut intersection = storage.capacity(delaunay.cloud.len(), quantity)?;
    let mut previous = None;
    let mut is_edge = false;
    for i in order {
        work.check()?;
        if previous.is_none_or(|p| !signature(i).eq(signature(p))) {
            let smallest = signature(i)
                .map(|group| membership.at(group))
                .min_by_key(|sites| sites.len())
                .ok_or_else(|| unstable("empty dual edge incidence"))?;
            if smallest.len() == 2 {
                is_edge = true;
            } else if ranges[i].len() == 1 {
                is_edge = false;
            } else {
                intersection.clear();
                work.charge_canonical(smallest.len())?;
                intersection.extend_from_slice(smallest);
                for group in signature(i) {
                    let sites = membership.at(group);
                    work.charge_canonical(intersection.len())?;
                    intersection.retain(|site| sites.binary_search(site).is_ok());
                    if intersection.len() == 2 {
                        break;
                    }
                }
                is_edge = intersection.len() == 2;
            }
            previous = Some(i);
        }
        accepted[i] = is_edge;
    }
    let mut edges = storage.capacity(ranges.len(), quantity)?;
    for (range, accepted) in ranges.iter().zip(accepted) {
        if accepted {
            edges.push(records[range.start].0);
        }
    }
    frame.keep(0, retained(&edges));
    Ok(edges)
}
