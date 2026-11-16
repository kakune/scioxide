//! Voronoi diagrams on a sphere.
use crate::PointSet;
use crate::incidence::Incidence;
use crate::math3::{cross, dot, sub as subtract};
use crate::tessellation::GeometryOptions;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::hull::ConvexHull;
use crate::tessellation::numeric::{convert, normalize, validate_query};
use crate::tessellation::preparation::prepare;
use crate::tessellation::work::Work;
use crate::tessellation::{GeometryConstructionRequirements, GeometryProvenance, Result, unstable};
use crate::validation::inconsistent as invalid;
use alloc::vec::Vec;
use scioxide_core::Tolerance;
use scioxide_core::backend::checked_add;
use scioxide_core::backend::checked_mul;
use scioxide_core::{Argument, RealScalar, SciOxideError, SolveScope};
/// Voronoi diagram of sites on a sphere in three dimensions, with geodesic
/// distance.
///
/// For sites on a sphere, the convex hull of the sites is their spherical
/// Delaunay triangulation. Each hull face is dual to a Voronoi vertex: the
/// point of the sphere along the outward normal of the face. Region vertices
/// are listed in cyclic order around the site. Region areas are sums of the
/// solid angles of spherical triangles, each computed with the formula of
/// Van Oosterom and Strackee, scaled by `r²`.
///
/// ```
/// use scioxide_core::{Tolerance};
/// use scioxide_spatial::{GeometryOptions, PointSet, SphericalVoronoi};
/// let values = [
///     1.0_f64, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
/// ];
/// let points = PointSet::from_rows(&values, 3)?;
/// let diagram = SphericalVoronoi::with_options(
///     &points,
///     [0.0; 3],
///     1.0,
///     Tolerance::new(1e-12, 1e-12)?,
///     GeometryOptions::new(),
/// )?;
/// assert_eq!(diagram.region(0)?.len(), 4);
/// assert!((diagram.areas().iter().sum::<f64>() - 4.0 * core::f64::consts::PI).abs() < 1e-10);
/// assert!(diagram.region(6).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SphericalVoronoi<S: RealScalar> {
    requirements: GeometryConstructionRequirements,
    vertices: Vec<S>,
    regions: Vec<usize>,
    region_offsets: Vec<usize>,
    areas: Vec<S>,
    adjacency: Incidence,
    center: [S; 3],
    radius: S,
    provenance: GeometryProvenance,
    input_mapping: Vec<usize>,
}
impl<S: RealScalar> SphericalVoronoi<S> {
    /// Constructs a SphericalVoronoi with default geometry options.
    ///
    /// Uses the finite default geometry limits and the construction method and cost of [`Self::with_options`].
    ///
    /// # Errors
    /// As [`Self::with_options`].
    pub fn new(
        points: &PointSet<'_, S>,
        center: [S; 3],
        radius: S,
        tolerance: scioxide_core::Tolerance<S>,
    ) -> Result<Self> {
        Self::with_options(
            points,
            center,
            radius,
            tolerance,
            GeometryOptions::default(),
        )
    }

    /// Computes the spherical Voronoi diagram of `points`, one site per row,
    /// on the sphere with the given `center` and `radius`.
    ///
    /// Each site must satisfy `||p − c| − r| ≤ τ`, where `τ` is the threshold
    /// of `tolerance` at `r`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `radius` is not finite and positive, a site is not
    ///   on the sphere, a region has fewer than three vertices, or
    ///   [`GeometryDegeneracy::Strict`](crate::GeometryDegeneracy::Strict)
    ///   rejects duplicate sites.
    /// - `DimensionMismatch` if the input points do not have three coordinates.
    /// - `Singular` if the sites do not span three dimensions.
    /// - `NumericalInstability` if a region or its area cannot be computed
    ///   reliably.
    /// - `ResourceLimit` if the construction exceeds a
    ///   [`GeometryLimits`](crate::GeometryLimits) value.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn with_options(
        points: &PointSet<'_, S>,
        center: [S; 3],
        radius: S,
        tolerance: scioxide_core::Tolerance<S>,
        options: GeometryOptions,
    ) -> Result<Self> {
        Self::prepare(points, center, radius, tolerance, options, None)
    }
    /// Computes the diagram under an execution or solve context.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `ExecutionStopped` or `ConvergenceFailure` as for
    /// [`Delaunay::build`](crate::Delaunay::build).
    pub fn build<'scope, 'capability>(
        points: &PointSet<'_, S>,
        center: [S; 3],
        radius: S,
        tolerance: scioxide_core::Tolerance<S>,
        options: GeometryOptions,
        cx: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        Self::prepare(points, center, radius, tolerance, options, Some(cx.into()))
    }
    fn prepare<'scope, 'capability>(
        points: &PointSet<'_, S>,
        center: [S; 3],
        radius: S,
        tolerance: Tolerance<S>,
        options: GeometryOptions,
        scope: Option<SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        prepare(
            3,
            points.len(),
            scioxide_core::argument!("points"),
            options,
            scope,
            |work| Self::construct(points, center, radius, tolerance, work),
        )
    }
    /// Returns the peak storage used during construction.
    pub const fn construction_requirements(&self) -> GeometryConstructionRequirements {
        self.requirements
    }
    fn construct(
        points: &PointSet<'_, S>,
        center: [S; 3],
        radius: S,
        radial_tolerance: Tolerance<S>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let storage = work.storage;
        let options = work.options;

        const POINTS: Argument = scioxide_core::argument!("points");
        if points.dimension() != 3 {
            return Err(SciOxideError::dimension_mismatch([3], [points.dimension()]));
        }
        validate_query(&center, 3)?;
        if !radius.is_finite() || radius <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("radius"),
                "sphere radius must be finite and positive",
            ));
        }
        let r = radius.try_to_f64_exact()?;
        let c = [
            center[0].try_to_f64_exact()?,
            center[1].try_to_f64_exact()?,
            center[2].try_to_f64_exact()?,
        ];
        // Translate before dividing, retaining differences outside native range.
        let unit_coordinate = |point: f64, center: f64| -> Result<f64> {
            scioxide_core::Scaled::from_value(point)?
                .sub(scioxide_core::Scaled::from_value(center)?)?
                .div(scioxide_core::Scaled::from_value(r)?)?
                .round()
        };
        let tolerance = radial_tolerance.threshold(radius)?.try_to_f64_exact()? / r;
        for i in 0..points.len() {
            work.check()?;
            let mut square = 0.;
            for (axis, center) in c.iter().enumerate() {
                let value =
                    unit_coordinate(points.coordinate(i, axis)?.try_to_f64_exact()?, *center)?;
                square += value * value;
            }
            if !square.is_finite() || (square.sqrt() - 1.).abs() > tolerance {
                return Err(invalid(POINTS, "site is not on the requested sphere"));
            }
        }
        // For sites on a sphere, the convex hull is the spherical Delaunay
        // triangulation. Each hull face (coplanar facets merged) is dual to
        // the Voronoi vertex center + r * n, where n is its outward unit
        // normal. That point is on the sphere and equidistant from the
        // face's sites.
        let cloud = Cloud::new(points, options, storage)?;
        let topology = super::engine::hull(&cloud, false, work)?;
        let hull = ConvexHull::from_topology(cloud, topology, work)?;
        // A canonical site on the unit sphere, scaled as in the radial check.
        let unit = |index: usize| -> Result<[f64; 3]> {
            let p = hull.cloud.point(index);
            Ok([
                unit_coordinate(p[0].try_to_f64_exact()?, c[0])?,
                unit_coordinate(p[1].try_to_f64_exact()?, c[1])?,
                unit_coordinate(p[2].try_to_f64_exact()?, c[2])?,
            ])
        };
        let mut group_vertex = storage.filled(
            hull.facet_count(),
            0,
            scioxide_core::quantity!("spherical_face_map"),
        )?;
        let mut normals: Vec<[f64; 3]> = storage.capacity(
            hull.facet_count(),
            scioxide_core::quantity!("spherical_normals"),
        )?;
        for (i, facet) in hull.topology.facets.iter().enumerate() {
            work.check()?;
            let group = hull.coplanar_faces()[i];
            if group != i {
                group_vertex[i] = group_vertex[group];
                continue;
            }
            let a = unit(facet.vertices[0])?;
            let b = unit(facet.vertices[1])?;
            let c = unit(facet.vertices[2])?;
            let mut normal = cross(subtract(b, a), subtract(c, a));
            normalize(&mut normal)?;
            if facet.inside < 0 {
                normal = normal.map(|x| -x);
            }
            group_vertex[i] = normals.len();
            normals.push(normal);
        }
        let mut vertices = storage.capacity(
            checked_mul(
                normals.len(),
                3,
                scioxide_core::quantity!("spherical_vertices"),
            )?,
            scioxide_core::quantity!("spherical_vertices"),
        )?;
        for normal in &normals {
            work.check()?;
            for axis in 0..3 {
                vertices.push(convert(c[axis] + r * normal[axis])?);
            }
        }
        let mut regions = storage.capacity(
            checked_mul(
                hull.facet_count(),
                3,
                scioxide_core::quantity!("spherical_regions"),
            )?,
            scioxide_core::quantity!("spherical_regions"),
        )?;
        let mut region_offsets = storage.capacity(
            checked_add(
                hull.cloud.len(),
                1,
                scioxide_core::quantity!("spherical_offsets"),
            )?,
            scioxide_core::quantity!("spherical_offsets"),
        )?;
        region_offsets.push(0);
        let mut region = Vec::new();
        let mut areas = storage.capacity(
            hull.cloud.len(),
            scioxide_core::quantity!("spherical_areas"),
        )?;
        let mut starts = storage.filled(
            hull.cloud.len(),
            None,
            scioxide_core::quantity!("spherical_starts"),
        )?;
        let mut degrees = storage.filled(
            hull.cloud.len(),
            0,
            scioxide_core::quantity!("spherical_degrees"),
        )?;
        for (facet, cell) in hull.topology.facets.iter().enumerate() {
            work.check()?;
            for &site in cell.vertices {
                starts[site].get_or_insert(facet);
                degrees[site] += 1;
            }
        }
        for site in 0..hull.cloud.len() {
            work.check()?;
            let start = starts[site].ok_or_else(region_error)?;
            // Walk the star of `site` in cyclic order. From each facet, step
            // across a ridge that still contains `site` (the omitted vertex is
            // not the site), never straight back. Consecutive facets in one
            // coplanar group map to the same vertex and are collapsed. The
            // walk is bounded by the facet count, so a corrupt adjacency
            // fails instead of looping.
            region.clear();
            storage.reserve(&mut region, degrees[site])?;
            let mut previous = None;
            let mut current = start;
            for _ in 0..=hull.facet_count() {
                work.check()?;
                if region.last() != Some(&group_vertex[current]) {
                    region.push(group_vertex[current]);
                }
                let facet = &hull.topology.facets.at(current);
                let next = hull
                    .topology
                    .adjacent(current)?
                    .iter()
                    .enumerate()
                    .filter(|(omitted, _)| facet.vertices[*omitted] != site)
                    .filter_map(|(_, n)| *n)
                    .find(|n| Some(*n) != previous)
                    .ok_or_else(|| unstable("open spherical adjacency"))?;
                previous = Some(current);
                current = next;
                if current == start {
                    break;
                }
            }
            if current != start {
                return Err(unstable("spherical adjacency did not close"));
            }
            if region.len() > 1 && region.first() == region.last() {
                region.pop();
            }
            if region.len() < 3 {
                return Err(region_error());
            }
            let mut generator = unit(site)?;
            normalize(&mut generator)?;
            // Region area is a fan of spherical triangles (generator, v_i,
            // v_{i+1}) on the unit sphere. Each solid angle uses the
            // Van Oosterom-Strackee formula
            // tan(omega / 2) = |g . (a x b)| / (1 + g . a + a . b + b . g),
            // which atan2 keeps accurate for small and near-hemispherical
            // triangles. 0/negative is the undefined antipodal case.
            let mut area = 0.;
            for i in 0..region.len() {
                work.check()?;
                let a = normals[region[i]];
                let b = normals[region[(i + 1) % region.len()]];
                let numerator = dot(generator, cross(a, b)).abs();
                let denominator = 1. + dot(generator, a) + dot(a, b) + dot(b, generator);
                if numerator == 0. && denominator <= 0. {
                    return Err(unstable("ambiguous antipodal spherical edge"));
                }
                area += 2. * numerator.atan2(denominator);
            }
            if !area.is_finite() || area <= 0. {
                return Err(unstable(
                    "spherical region has no representable positive solid angle",
                ));
            }
            let output = convert((area * r) * r)?;
            if area > 0. && output == S::ZERO {
                return Err(unstable("spherical area underflows the output scalar"));
            }
            areas.push(output);
            regions.extend_from_slice(&region);
            region_offsets.push(regions.len());
        }
        // Sites are Voronoi neighbors when they share a hull edge between two
        // different face groups. An edge inside a coplanar group is a
        // triangulation artifact with no Voronoi edge.
        let groups = &group_vertex;
        let adjacency = Incidence::from_pairs(
            region_offsets.len() - 1,
            hull.topology
                .facets
                .iter()
                .zip(hull.topology.neighbors.chunks_exact(3))
                .enumerate()
                .flat_map(|(i, (facet, neighbors))| {
                    neighbors
                        .iter()
                        .enumerate()
                        .filter_map(move |(omitted, neighbor)| {
                            neighbor
                                .filter(|&n| groups[n] != groups[i])
                                .map(|_| match omitted {
                                    0 => [facet.vertices[1], facet.vertices[2]],
                                    1 => [facet.vertices[0], facet.vertices[2]],
                                    _ => [facet.vertices[0], facet.vertices[1]],
                                })
                        })
                })
                .flat_map(|[a, b]| [(a, b), (b, a)]),
            work.context.as_deref(),
            storage,
        )?;
        let input_mapping = storage.copy(
            hull.input_mapping(),
            scioxide_core::quantity!("input_mapping"),
        )?;
        Ok(Self {
            requirements: storage.requirements(),
            vertices,
            regions,
            region_offsets,
            areas,
            adjacency,
            center,
            radius,
            provenance: hull.provenance(),
            input_mapping,
        })
    }
    /// Returns the center of the sphere.
    pub const fn center(&self) -> [S; 3] {
        self.center
    }
    /// Returns the radius of the sphere.
    pub const fn radius(&self) -> S {
        self.radius
    }
    /// Returns the input counts and work of the construction.
    pub const fn provenance(&self) -> GeometryProvenance {
        self.provenance
    }
    /// Returns, in ascending order, the sites whose regions share an arc with
    /// the region of `site`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `site` is not a site index.
    pub fn neighbors(&self, site: usize) -> Result<impl Iterator<Item = usize> + '_> {
        self.adjacency
            .neighbors(site, scioxide_core::quantity!("sphere_region"))
    }
    /// Returns the rounded Voronoi vertex coordinates, three values per vertex.
    pub fn vertices(&self) -> &[S] {
        &self.vertices
    }
    /// Returns the vertices of a site's region in cyclic order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `site` is not a site index.
    pub fn region(&self, site: usize) -> Result<&[usize]> {
        if site >= self.region_offsets.len() - 1 {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("sphere_region"),
                site,
                self.region_offsets.len() - 1,
            ));
        }
        Ok(&self.regions[self.region_offsets[site]..self.region_offsets[site + 1]])
    }
    /// Returns the area of each region, indexed by site, in squared
    /// coordinate units. The areas sum to `4π r²` up to rounding.
    pub fn areas(&self) -> &[S] {
        &self.areas
    }
    /// Returns the site index of each input row.
    pub fn input_mapping(&self) -> &[usize] {
        &self.input_mapping
    }
}

fn region_error() -> SciOxideError {
    invalid(
        scioxide_core::argument!("points"),
        "spherical Voronoi region requires at least three distinct vertices",
    )
}
