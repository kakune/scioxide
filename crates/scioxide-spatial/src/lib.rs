//! Computational geometry: meshes, exact predicates, rotations, distances,
//! nearest-neighbor search and tessellations.
//!
//! The crate builds structured and simplicial meshes and attaches typed fields
//! to them, refines meshes locally and partitions them for parallel solvers. It
//! decides orientation and in-sphere tests exactly, computes convex hulls,
//! Delaunay triangulations, Voronoi diagrams and halfspace intersections in any
//! dimension up to a configurable limit, and locates points in them. It also
//! represents and interpolates 3-D rotations and rigid motions, computes
//! distance matrices for 17 metrics plus standardized Euclidean and
//! Mahalanobis distances, and answers nearest-neighbor and radius queries
//! with KD trees. Coordinates are `f32` or `f64`.
//!
//! # Quick start
//!
//! ```
//! use scioxide_spatial::{Delaunay, PointSet};
//! let sites = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.3, 0.4]];
//! let points = PointSet::from_points(&sites)?;
//! let mesh = Delaunay::new(&points)?;
//! let query = [0.5, 0.2];
//! let found = mesh.find_simplex(&query)?.expect("inside the hull");
//! let weights = mesh.barycentric(found.simplex, &query)?;
//! assert_eq!(mesh.simplices().chunks_exact(3).len(), 4);
//! assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Entry point | Method |
//! |---|---|---|
//! | Orientation, in-circle and in-sphere tests | [`orient2d`], [`orient3d`], [`incircle`], [`insphere`], [`SimplexPredicateWorkspace`] | floating-point filter with exact fallback |
//! | Tensor-product grids and meshes | [`CoordinateAxis`], [`StructuredGrid::uniform`], [`CellMesh::uniform`], [`BoundarySet`] | implicit connectivity from integer indices |
//! | Interval, triangle and tetrahedron meshes | [`IntervalMesh`], [`TriangleMesh`], [`TetrahedronMesh`] | explicit connectivity, checked orientation and manifold links |
//! | Values on vertices, cells, faces or edges | [`Field`], [`FieldView`], [`VertexField`], [`CellField`] | checked against the mesh revision |
//! | Local refinement and coarsening | [`AdaptiveMesh`], [`TopologyMap`] | one level per call, 2:1 face balance |
//! | Distributing a mesh over processes | [`MeshPartition`], [`RepartitionMap`] | cell ownership, one layer of ghost cells |
//! | Rotations and rigid motions | [`Rotation`], [`RigidTransform`] | unit quaternions |
//! | Interpolating orientations | [`Slerp`], [`RotationSpline`] | spherical linear interpolation; cubic spline with continuous angular acceleration |
//! | Best-fit rigid motion between matched points | [`estimate_rigid_transform`] | weighted Kabsch algorithm |
//! | Distance matrices | [`distance`], [`pairwise`], [`condensed`] (Euclidean), [`PreparedDistance`] (other metrics) | blocked pair loops |
//! | Directed Hausdorff distance | [`directed_hausdorff`] | random order with early break (Taha–Hanbury) |
//! | Nearest-neighbor and radius queries | [`KdTree`] | KD tree split at the median of the widest axis |
//! | Convex hull | [`ConvexHull`] | randomized incremental construction |
//! | Delaunay triangulation and point location | [`Delaunay`] | lower convex hull of lifted sites; visibility walk |
//! | Voronoi diagram | [`Voronoi`], [`SphericalVoronoi`] | dual of the Delaunay triangulation; dual of the 3-D hull |
//! | Intersection of halfspaces | [`HalfspaceIntersection`] | convex hull of the polar dual |
//!
//! # Coordinates and conventions
//!
//! A [`PointSet`] borrows an array of points, a flat slice with a dimension,
//! or a rank-two strided [`NdView`](scioxide_core::NdView). It checks once that every coordinate is
//! finite. Distances, searches, rigid-motion batches and tessellations take
//! point sets. Grids and simplex meshes use [`Point`], [`Extent`] and [`Index`]
//! in one to three dimensions, numbered with axis 0 fastest: in an extent
//! `[nx, ny, nz]`, index `[x, y, z]` has linear position `x + nx (y + ny z)`.
//!
//! Rotations are active and right-handed. Angles are in radians. Quaternions
//! are stored scalar-last, `[x, y, z, w]`. Matrices are row-major and act on
//! column vectors.
//!
//! # Exact predicates and approximate constructions
//!
//! Every combinatorial decision uses the exact sign of a determinant of the
//! input coordinates as given. This covers the predicates, convex hull and
//! Delaunay facets, point location, and the orientation checks of simplex
//! meshes. A floating-point filter certifies most signs; when it cannot, the
//! determinant is evaluated in exact arithmetic. The result is therefore
//! correct for all finite inputs, including degenerate ones.
//!
//! Constructed coordinates are rounded: circumcenters, Voronoi vertices and
//! ray directions, halfspace vertices, barycentric coordinates and measures.
//! An unreliable Delaunay or Voronoi construction makes the query for that
//! item return `NumericalInstability`; the remaining items stay usable.
//! An uncertified halfspace vertex instead fails the entire
//! [`HalfspaceIntersection`] construction with `NumericalInstability`.
//!
//! # Tessellations
//!
//! [`ConvexHull`] and [`Delaunay`] sort sites lexicographically and merge exact
//! duplicates. Their `input_mapping` methods give the site of each input row;
//! `representatives` gives the first input row of each site. [`Voronoi`]
//! exposes these mappings through [`Voronoi::delaunay`]. [`SphericalVoronoi`]
//! also sorts and merges sites, but exposes only `input_mapping`. Per-site
//! results use canonical site indices, not input-row indices. These site-based
//! constructions depend on the set of sites, not their input order; sites must
//! span the full space or construction returns `Singular`.
//!
//! [`HalfspaceIntersection`] keeps inequality rows in input order, including
//! duplicates, and exposes no site mapping methods. Active halfspace indices
//! refer to the original rows.
//!
//! Ties, such as four cocircular sites in the plane, are resolved by symbolic
//! perturbation in sorted-site order under
//! [`GeometryDegeneracy::Deterministic`], the default.
//! [`GeometryDegeneracy::Strict`] rejects such input instead.
//! [`GeometryLimits`] bounds the dimension, site and facet counts, predicate
//! work and storage, and [`GeometryProvenance`] reports what a construction
//! used. `new` uses the finite [`GeometryLimits`] defaults; `with_options`
//! accepts configured limits. Both use default capabilities; `build` takes an
//! [execution context](scioxide_core#execution-contexts) or a solve context.
//!
//! # Mesh revisions and fields
//!
//! Each mesh carries a [`MeshToken`]: a mesh family and a generation number.
//! Methods that return a changed mesh, such as
//! [`StructuredGrid::refined`], [`SimplexMesh::with_points`] or
//! [`AdaptiveMesh::adapt`], keep the family and increase the generation.
//! Fields, boundary sets, boundary tags and topology maps store the token of
//! the mesh they were built for, and operations that combine them check it in
//! O(1) time.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: invalid options, mesh connectivity that fails validation,
//!   a field or map used with another mesh revision, a rotation matrix that is
//!   not orthogonal, an interpolation time outside the knots, sites rejected by
//!   [`GeometryDegeneracy::Strict`], or a point that is not strictly inside
//!   every halfspace.
//! - `DimensionMismatch`: coordinate dimensions or lengths disagree, or an
//!   output, scratch slice or workspace is too short, including radius matches
//!   that do not fit the output of [`KdTree::radius_batch_into`].
//! - `OutOfBounds`: an index outside its mesh, tessellation or point set.
//! - `NonFinite`: a coordinate, weight or query is NaN or infinite.
//! - `DomainError`: a point outside the coordinate span of a grid, or a
//!   distance that is undefined for its input, such as the cosine distance of
//!   a zero vector.
//! - `Singular`: tessellation sites that do not span the space.
//! - `NumericalInstability`: a constructed coordinate, measure or distance is
//!   ill-conditioned or not representable.
//! - `ResourceLimit`: a construction exceeds [`GeometryLimits`] or
//!   [`AdaptationLimits`], a workspace exceeds its byte limit, or
//!   [`KdTree::radius_into`] finds more neighbors than the output holds.
//! - `ExecutionStopped` and `ConvergenceFailure`: cancellation, a deadline or
//!   an exhausted solve budget in an operation that takes a context.
//!
//! An unbounded halfspace intersection is reported by [`IntersectionKind`], and
//! [`RotationSpline::with_options`] and [`RotationSpline::build`] report
//! non-convergence through [`RotationSplineTermination`]. [`RotationSpline::new`]
//! reports iteration exhaustion as `ConvergenceFailure`. On error, outputs are
//! unchanged, as described in [Output on error](scioxide_core#output-on-error).
//! Functions that write a variable number of results change only the prefix
//! they report.
//!
//! # Accuracy
//!
//! Predicates and every combinatorial result are exact. Distances use `f64`
//! intermediates for both scalar types, and Euclidean norms rescale to avoid
//! overflow and underflow. A vertex `x` of a [`HalfspaceIntersection`] in
//! dimension `d` violates each inequality `a·x + b ≤ 0` by at most
//! `128 ε (d + 1) (|b| + Σᵢ |aᵢ xᵢ|)`, where `ε` is the machine epsilon of the
//! scalar type. Other constructed coordinates have no proven bound; their
//! methods state the checks they apply.
//!
//! # Reusing plans and workspaces
//!
//! Trees, tessellations, prepared distances and splines are immutable plans
//! that follow the [shared rules](scioxide_core#plans-and-workspaces). Owned query
//! forms allocate their result and workspace per call; `_into` and `_scoped`
//! forms reuse caller storage. Batched
//! operations use a [`DistanceWorkspace`], [`SearchWorkspace`],
//! [`TransformWorkspace`] or [`GeometryWorkspace`]. Reserve it once from
//! `workspace_requirements`, and later calls up to that size do not allocate.
//! The `max_workspace_bytes` option bounds all capacity the workspace holds,
//! including capacity left from earlier, larger reservations. A failed
//! `try_reserve` may keep part of the new capacity; reserve again before use.
//!
//! # Execution
//!
//! Batch operations take an [`ExecutionPolicy`](scioxide_core::ExecutionPolicy)
//! in their options and an [execution context](scioxide_core#execution-contexts).
//! With `Auto`, distance, search and transform batches consider parallel
//! execution from 256 outputs or queries. Cancellation and deadlines are
//! checked at least every 256 pairs, points or queries, and again before the
//! output is written.
//!
//! # Feature flags
//!
//! - `std` (default): enables the standard-library parts of `scioxide-core` and
//!   `scioxide-linalg`.
//!
//! Without default features the crate is `no_std` and needs `alloc`; the API is
//! the same.
//!
//! # References
//!
//! - J. R. Shewchuk, "Adaptive precision floating-point arithmetic and fast
//!   robust geometric predicates", *Discrete Comput. Geom.* 18, 1997.
//! - K. L. Clarkson and P. W. Shor, "Applications of random sampling in
//!   computational geometry, II", *Discrete Comput. Geom.* 4, 1989.
//! - K. Q. Brown, "Voronoi diagrams from convex hulls", *Inf. Process. Lett.*
//!   9(5), 1979.
//! - O. Devillers, S. Pion and M. Teillaud, "Walking in a triangulation",
//!   *Int. J. Found. Comput. Sci.* 13(2), 2002.
//! - W. Kabsch, "A solution for the best rotation to relate two sets of
//!   vectors", *Acta Cryst.* A32, 1976.
//! - K. Shoemake, "Animating rotation with quaternion curves", *SIGGRAPH* 1985.
//! - J. H. Friedman, J. L. Bentley and R. A. Finkel, "An algorithm for finding
//!   best matches in logarithmic expected time", *ACM Trans. Math. Softw.*
//!   3(3), 1977.
//! - A. A. Taha and A. Hanbury, "An efficient algorithm for calculating the
//!   exact Hausdorff distance", *IEEE Trans. Pattern Anal. Mach. Intell.*
//!   37(11), 2015.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod adaptive;
mod batch;
mod context;
mod incidence;
mod math3;
mod search_workspace;
mod validation;
mod workspace;
pub use adaptive::{
    AdaptationLimits, AdaptationRequest, AdaptationResult, AdaptiveCellKind, AdaptiveInterface,
    AdaptiveInterfaceSide, AdaptiveMesh, EntityRelation, HangingEntity, HangingEntityKind,
    MeshPartition, RepartitionMap, RepartitionOptions, TopologyMap,
};
pub use context::SpatialExecutionContext;
pub use search_workspace::{
    SearchOptions, SearchWorkspace, SearchWorkspaceRequest, SearchWorkspaceRequirements,
};
pub use workspace::{
    DistanceOptions, DistanceWorkspace, DistanceWorkspaceRequest, DistanceWorkspaceRequirements,
};
mod distance;
mod field;
mod geometry;
mod grid;
mod identity;
mod indices;
mod predicates;
mod rigid;
mod rotation;
mod rotation_spline;
mod transform_workspace;
pub use transform_workspace::{
    TransformOptions, TransformWorkspace, TransformWorkspaceRequest, TransformWorkspaceRequirements,
};
mod search;
mod simplex;
mod topology;

pub use simplex::{
    IntervalMesh, OrientedEntity, SimplexBoundaryTags, SimplexMesh, TetrahedronMesh, TriangleMesh,
};

pub use predicates::{
    PredicateSign, SimplexPredicateWorkspace, incircle, insphere, orient2d, orient3d,
};

pub use rigid::{
    RigidEstimationOptions, RigidEstimationResult, RigidTransform, estimate_rigid_transform,
    estimate_rigid_transform_scoped, estimate_rigid_transform_with_options,
};
pub use rotation::{EulerAngles, EulerFrame, EulerSequence, Rotation, RotationAxis};
pub use rotation_spline::{
    RotationSpline, RotationSplineOptions, RotationSplineResult, RotationSplineSample,
    RotationSplineTermination, Slerp,
};

pub use distance::{
    DistanceMetric, HausdorffResult, PointSet, PreparedDistance, condensed, condensed_index,
    condensed_len, condensed_to_square, condensed_to_square_into, directed_hausdorff, distance,
    pairwise, square_to_condensed, square_to_condensed_into, validate_condensed,
    validate_distance_matrix,
};
pub use field::{
    Cell, CellField, Edge, EdgeField, Face, FaceField, Field, FieldLocation, FieldView,
    MeshEntities, UnstructuredEdge, UnstructuredEdgeField, UnstructuredFace, UnstructuredFaceField,
    Vertex, VertexField,
};

pub use geometry::{
    Extent, Extent1, Extent2, Extent3, Index, Index1, Index2, Index3, Point, Point1, Point2, Point3,
};
pub use grid::{
    AxisOrientation, AxisSide, AxisTopology, CellLocation, CoordinateAxis, Grid1d, Grid2d, Grid3d,
    IntervalLocation, StructuredGrid,
};
pub use identity::{MeshGeneration, MeshId, MeshToken};
pub use search::{KdTree, KdTreeOptions, Neighbor, TreeMetric};
pub use topology::{
    BoundarySet, CellKind, CellMesh, FaceIndex, FaceTopology, HexMesh, QuadMesh, SegmentMesh,
    TopologySizes,
};

mod tessellation;
pub use tessellation::{
    ConvexHull, Delaunay, GeometryConstructionRequirements, GeometryDegeneracy, GeometryLimits,
    GeometryOptions, GeometryProvenance, GeometryWorkspace, GeometryWorkspaceRequirements,
    HalfspaceIntersection, IntersectionKind, SimplexLocation, SphericalVoronoi, Voronoi,
    VoronoiRay, VoronoiRidge,
};
