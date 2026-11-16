//! Oriented interval, triangle and tetrahedron meshes.

use crate::incidence::Incidence;
use crate::validation::{get, inconsistent as invalid};
use crate::{MeshToken, Point, PredicateSign, orient2d, orient3d};
use alloc::{sync::Arc, vec::Vec};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{Argument, RealScalar, SciOxideError};

pub(crate) mod measure;

/// A face or edge index with an orientation sign.
///
/// Faces and edges are stored with their vertices in ascending order. The sign
/// is `1` when the orientation induced by the cell agrees with that order and
/// `-1` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrientedEntity {
    index: usize,
    sign: i8,
}
impl OrientedEntity {
    /// Returns the face or edge index.
    pub const fn index(self) -> usize {
        self.index
    }
    /// Returns the orientation sign, `1` or `-1`.
    pub const fn sign(self) -> i8 {
        self.sign
    }
}

// Cells and faces use fixed-width arrays sized for 3D. Intervals and triangles
// fill the first D + 1 (respectively D) slots and leave the rest unused, so one
// const-generic layout and one code path serve every dimension. The flat
// `cell_faces` and `cell_edges` arrays have a fixed stride per cell, and
// `face_edges` a fixed stride per face.
#[derive(Debug)]
struct Storage<S: RealScalar, const D: usize> {
    points: Vec<Point<S, D>>,
    cells: Vec<[usize; 4]>,
    faces: Vec<[usize; 3]>,
    edges: Vec<[usize; 2]>,
    cell_faces: Vec<OrientedEntity>,
    cell_edges: Vec<OrientedEntity>,
    face_edges: Vec<OrientedEntity>,
    face_cells: Incidence,
    edge_cells: Incidence,
    vertex_cells: Incidence,
}

/// Mesh of positively oriented simplices: intervals, triangles or tetrahedra.
///
/// Create one with [`IntervalMesh::new`], [`TriangleMesh::new`] or
/// [`TetrahedronMesh::new`]. Construction checks that each cell has distinct
/// vertices and positive orientation by an exact predicate, that no cell
/// repeats, that two cells sharing a face orient it oppositely, and that the
/// link of every vertex is a manifold. It then numbers the faces and edges and
/// stores their incidence, so the topology queries cost O(1) per result.
///
/// The global embedding is not checked: a mesh that folds over itself, for
/// example a fan wound twice around an interior vertex, is accepted and its
/// cell measures add up. In one dimension a face is a vertex, with measure
/// one, and each interval is its own edge. Cloning shares the storage.
#[derive(Debug, Clone)]
pub struct SimplexMesh<S: RealScalar, const D: usize> {
    storage: Arc<Storage<S, D>>,
    token: MeshToken,
}
/// Mesh of intervals on the real line, each with increasing endpoints.
///
/// ```
/// use scioxide_spatial::{IntervalMesh, Point};
/// let mesh = IntervalMesh::new(
///     vec![
///         Point::new([0.0_f64])?,
///         Point::new([0.25])?,
///         Point::new([1.0])?,
///     ],
///     vec![[0, 1], [1, 2]],
/// )?;
/// assert!((mesh.cell_measure(1)? - 0.75).abs() < 1e-15);
/// assert_eq!(mesh.boundary_faces().count(), 2);
/// assert_eq!(mesh.edge_count(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub type IntervalMesh<S> = SimplexMesh<S, 1>;
/// Mesh of counterclockwise triangles in the plane.
///
/// ```
/// use scioxide_spatial::{Point, SimplexBoundaryTags, TriangleMesh};
/// let mesh = TriangleMesh::new(
///     vec![
///         Point::new([0.0_f64, 0.0])?,
///         Point::new([1.0, 0.0])?,
///         Point::new([0.0, 1.0])?,
///     ],
///     vec![[0, 1, 2]],
/// )?;
/// assert!((mesh.cell_measure(0)? - 0.5).abs() < 1e-12);
/// let tags = SimplexBoundaryTags::new(&mesh, vec![(0, 7)])?;
/// assert_eq!(tags.tag(&mesh, 0)?, Some(7));
/// assert!(mesh.cell_vertices(1).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub type TriangleMesh<S> = SimplexMesh<S, 2>;
/// Mesh of tetrahedra `[a, b, c, d]` with `det(b − a, c − a, d − a) > 0`.
///
/// ```
/// use scioxide_spatial::{Point, TetrahedronMesh};
/// let mesh = TetrahedronMesh::new(
///     vec![
///         Point::new([0.0_f64; 3])?,
///         Point::new([1.0, 0.0, 0.0])?,
///         Point::new([0.0, 1.0, 0.0])?,
///         Point::new([0.0, 0.0, 1.0])?,
///     ],
///     vec![[0, 1, 2, 3]],
/// )?;
/// assert!((mesh.cell_measure(0)? - 1.0 / 6.0).abs() < 1e-12);
/// assert_eq!(mesh.boundary_faces().count(), 4);
/// assert!(mesh.cell_measure(1).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub type TetrahedronMesh<S> = SimplexMesh<S, 3>;

impl<S: RealScalar> IntervalMesh<S> {
    /// Creates an interval mesh. Vertices and cells keep their input order.
    ///
    /// Every cell `[a, b]` must satisfy `x[a] < x[b]`, and a vertex may end at
    /// most one interval and start at most one.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a cell refers to a missing vertex.
    /// - `InvalidInput` if a cell is inverted, degenerate or repeated, a
    ///   vertex is unused, or a vertex is shared by intervals in another way.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn new(points: Vec<Point<S, 1>>, cells: Vec<[usize; 2]>) -> Result<Self, SciOxideError> {
        let mut padded =
            try_with_capacity(cells.len(), scioxide_core::quantity!("interval_cells"))?;
        padded.extend(cells.into_iter().map(|[a, b]| [a, b, 0, 0]));
        Self::build(points, padded, scioxide_core::argument!("cells"))
    }
    /// Returns a mesh with the same points and new cells.
    ///
    /// Unchanged cells return a clone with the same token; otherwise the mesh
    /// keeps its family and has the next generation.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `SizeOverflow` if the generation overflows.
    pub fn with_cells(&self, cells: Vec<[usize; 2]>) -> Result<Self, SciOxideError> {
        if cells.iter().map(|c| &c[..]).eq(self.cells()) {
            return Ok(self.clone());
        }
        let mut next = Self::new(self.copy_points()?, cells)?;
        next.token = self.token.next_generation()?;
        Ok(next)
    }
}
impl<S: RealScalar> TriangleMesh<S> {
    /// Creates a triangle mesh. Vertices and cells keep their input order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a cell refers to a missing vertex.
    /// - `InvalidInput` if a cell is clockwise, degenerate or repeated, a
    ///   vertex is unused, a vertex link is not a manifold, or two cells
    ///   orient a shared edge the same way.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn new(points: Vec<Point<S, 2>>, cells: Vec<[usize; 3]>) -> Result<Self, SciOxideError> {
        let mut padded =
            try_with_capacity(cells.len(), scioxide_core::quantity!("triangle_cells"))?;
        padded.extend(cells.into_iter().map(|[a, b, c]| [a, b, c, 0]));
        Self::build(points, padded, scioxide_core::argument!("cells"))
    }
    /// Returns a mesh with the same points and new cells.
    ///
    /// Unchanged cells return a clone with the same token; otherwise the mesh
    /// keeps its family and has the next generation.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `SizeOverflow` if the generation overflows.
    pub fn with_cells(&self, cells: Vec<[usize; 3]>) -> Result<Self, SciOxideError> {
        if cells.iter().map(|c| &c[..]).eq(self.cells()) {
            return Ok(self.clone());
        }
        let mut next = Self::new(self.copy_points()?, cells)?;
        next.token = self.token.next_generation()?;
        Ok(next)
    }
}
impl<S: RealScalar> TetrahedronMesh<S> {
    /// Creates a tetrahedron mesh. Vertices and cells keep their input order.
    ///
    /// A cell `[a, b, c, d]` must have `det(b − a, c − a, d − a) > 0`, which is
    /// a negative [`orient3d`](crate::orient3d)`(a, b, c, d)`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a cell refers to a missing vertex.
    /// - `InvalidInput` if a cell is inverted, degenerate or repeated, a
    ///   vertex is unused, a vertex link is not a manifold, or two cells
    ///   orient a shared face the same way.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn new(points: Vec<Point<S, 3>>, cells: Vec<[usize; 4]>) -> Result<Self, SciOxideError> {
        Self::build(points, cells, scioxide_core::argument!("cells"))
    }
    /// Returns a mesh with the same points and new cells.
    ///
    /// Unchanged cells return a clone with the same token; otherwise the mesh
    /// keeps its family and has the next generation.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `SizeOverflow` if the generation overflows.
    pub fn with_cells(&self, cells: Vec<[usize; 4]>) -> Result<Self, SciOxideError> {
        if cells == self.storage.cells {
            return Ok(self.clone());
        }
        let mut next = Self::new(self.copy_points()?, cells)?;
        next.token = self.token.next_generation()?;
        Ok(next)
    }
}

impl<S: RealScalar, const D: usize> SimplexMesh<S, D> {
    // Validation and storage shared by the public constructors.inconsistent input names
    // `argument` of that entry point.
    fn build(
        points: Vec<Point<S, D>>,
        cells: Vec<[usize; 4]>,
        argument: Argument,
    ) -> Result<Self, SciOxideError> {
        validate_cells(&points, &cells, argument)?;
        let storage = build_storage(points, cells, argument)?;
        validate_links(&storage, argument)?;
        Ok(Self {
            storage: Arc::new(storage),
            token: MeshToken::fresh(),
        })
    }

    pub(crate) fn with_mesh_token(mut self, token: MeshToken) -> Self {
        self.token = token;
        self
    }
    fn copy_points(&self) -> Result<Vec<Point<S, D>>, SciOxideError> {
        let mut points = try_with_capacity(
            self.vertex_count(),
            scioxide_core::quantity!("simplex_coordinates"),
        )?;
        points.extend_from_slice(self.points());
        Ok(points)
    }
    /// Returns a mesh with the same connectivity and new vertex coordinates.
    ///
    /// Only the orientation of each cell is checked again. Unchanged points
    /// return a clone with the same token; otherwise the mesh keeps its family
    /// and has the next generation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of points changes.
    /// - `InvalidInput` if a cell becomes degenerate or inverted.
    /// - `AllocationFailure` if storage cannot be reserved, and `SizeOverflow`
    ///   if the generation overflows.
    pub fn with_points(&self, points: Vec<Point<S, D>>) -> Result<Self, SciOxideError> {
        if points == self.storage.points {
            return Ok(self.clone());
        }
        if points.len() != self.vertex_count() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.vertex_count()),
                scioxide_core::Dimensions::one(points.len()),
            ));
        }
        // Connectivity, links and incidence are unchanged; only the exact
        // orientation of every cell depends on the new coordinates.
        for cell in &self.storage.cells {
            validate_orientation(&points, cell, scioxide_core::argument!("points"))?;
        }
        let topology = &self.storage;
        let storage = Storage {
            points,
            cells: copy(&topology.cells, scioxide_core::quantity!("simplex_cells"))?,
            faces: copy(&topology.faces, scioxide_core::quantity!("simplex_faces"))?,
            edges: copy(&topology.edges, scioxide_core::quantity!("simplex_edges"))?,
            cell_faces: copy(
                &topology.cell_faces,
                scioxide_core::quantity!("simplex_cell_faces"),
            )?,
            cell_edges: copy(
                &topology.cell_edges,
                scioxide_core::quantity!("simplex_cell_edges"),
            )?,
            face_edges: copy(
                &topology.face_edges,
                scioxide_core::quantity!("simplex_face_edges"),
            )?,
            face_cells: topology.face_cells.clone(),
            edge_cells: topology.edge_cells.clone(),
            vertex_cells: topology.vertex_cells.clone(),
        };
        Ok(Self {
            storage: Arc::new(storage),
            token: self.token.next_generation()?,
        })
    }
    /// Returns the mesh token of this revision.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }
    /// Returns the vertex coordinates in input order.
    pub fn points(&self) -> &[Point<S, D>] {
        &self.storage.points
    }
    /// Returns the cells in input order, each as `D + 1` vertex indices.
    pub fn cells(&self) -> impl ExactSizeIterator<Item = &[usize]> + '_ {
        self.storage.cells.iter().map(|cell| &cell[..D + 1])
    }
    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.storage.points.len()
    }
    /// Returns the number of cells.
    pub fn cell_count(&self) -> usize {
        self.storage.cells.len()
    }
    /// Returns the number of faces.
    pub fn face_count(&self) -> usize {
        self.storage.faces.len()
    }
    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.storage.edges.len()
    }
    /// Returns whether the mesh has no cells, and hence no other entities.
    pub fn is_empty(&self) -> bool {
        self.storage.cells.is_empty()
    }
    /// Returns the vertices of a cell in their input order, which is
    /// positively oriented.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    pub fn cell_vertices(&self, cell: usize) -> Result<&[usize], SciOxideError> {
        Ok(&get(
            &self.storage.cells,
            cell,
            scioxide_core::quantity!("simplex_cell"),
        )?[..D + 1])
    }
    /// Returns the vertices of a face in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    pub fn face_vertices(&self, face: usize) -> Result<&[usize], SciOxideError> {
        Ok(&get(
            &self.storage.faces,
            face,
            scioxide_core::quantity!("simplex_face"),
        )?[..D])
    }
    /// Returns the endpoints of an edge in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `edge >= self.edge_count()`.
    pub fn edge_vertices(&self, edge: usize) -> Result<&[usize; 2], SciOxideError> {
        get(
            &self.storage.edges,
            edge,
            scioxide_core::quantity!("simplex_edge"),
        )
    }
    /// Returns the faces of a cell with their orientation signs. Face `i` is
    /// the face opposite local vertex `i`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    pub fn cell_faces(&self, cell: usize) -> Result<&[OrientedEntity], SciOxideError> {
        self.cell_vertices(cell)?;
        Ok(&self.storage.cell_faces[cell * (D + 1)..(cell + 1) * (D + 1)])
    }
    /// Returns the edges of a cell with their orientation signs, ordered by
    /// local vertex pair `(0, 1), (0, 2), …`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    pub fn cell_edges(&self, cell: usize) -> Result<&[OrientedEntity], SciOxideError> {
        self.cell_vertices(cell)?;
        let count = D * (D + 1) / 2;
        Ok(&self.storage.cell_edges[cell * count..(cell + 1) * count])
    }
    /// Returns the edges of a face with their orientation signs.
    ///
    /// In three dimensions a face is a triangle with three edges. In two
    /// dimensions a face is an edge, so the result is that edge with sign `1`.
    /// In one dimension a face is a vertex and has no edges.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    pub fn face_edges(&self, face: usize) -> Result<&[OrientedEntity], SciOxideError> {
        self.face_vertices(face)?;
        let count = face_edge_count::<D>();
        Ok(&self.storage.face_edges[face * count..(face + 1) * count])
    }
    /// Returns the one or two cells that contain a face, in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    pub fn face_cells(&self, face: usize) -> Result<&[usize], SciOxideError> {
        self.face_vertices(face)?;
        Ok(self.storage.face_cells.at(face))
    }
    /// Returns the cells that contain an edge, in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `edge >= self.edge_count()`.
    pub fn edge_cells(&self, edge: usize) -> Result<&[usize], SciOxideError> {
        self.edge_vertices(edge)?;
        Ok(self.storage.edge_cells.at(edge))
    }
    /// Returns the cells that contain a vertex, in ascending order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex >= self.vertex_count()`.
    pub fn vertex_cells(&self, vertex: usize) -> Result<&[usize], SciOxideError> {
        get(
            self.points(),
            vertex,
            scioxide_core::quantity!("simplex_vertex"),
        )?;
        Ok(self.storage.vertex_cells.at(vertex))
    }
    /// Returns the cells that share a face with `cell`, in the order of its
    /// faces. Boundary faces contribute nothing.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    pub fn cell_neighbors(
        &self,
        cell: usize,
    ) -> Result<impl Iterator<Item = usize> + '_, SciOxideError> {
        Ok(self
            .cell_faces(cell)?
            .iter()
            .flat_map(move |face| self.storage.face_cells.at(face.index).iter().copied())
            .filter(move |&other| other != cell))
    }
    /// Returns the boundary faces, those in only one cell, in ascending order.
    pub fn boundary_faces(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.face_count()).filter(|&face| self.storage.face_cells.at(face).len() == 1)
    }
    /// Returns whether a face is a boundary face.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    pub fn is_boundary_face(&self, face: usize) -> Result<bool, SciOxideError> {
        Ok(self.face_cells(face)?.len() == 1)
    }
    /// Returns the length, area or volume of a cell.
    ///
    /// The determinant of the edge vectors is accumulated exactly before it
    /// is rounded.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    /// - `NumericalInstability` if the measure is not representable.
    pub fn cell_measure(&self, cell: usize) -> Result<S, SciOxideError> {
        measure::cell(self.points(), self.cell_vertices(cell)?)
    }
    /// Returns the length of an edge, rescaled to avoid overflow and underflow.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `edge >= self.edge_count()`.
    /// - `NumericalInstability` if the length is not representable.
    pub fn edge_length(&self, edge: usize) -> Result<S, SciOxideError> {
        measure::length(self.points(), self.edge_vertices(edge)?)
    }
    /// Returns the measure of a face: one for a vertex in one dimension, the
    /// length of an edge in two, the area of a triangle in three.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    /// - `NumericalInstability` if the measure is not representable.
    pub fn face_measure(&self, face: usize) -> Result<S, SciOxideError> {
        measure::face(self.points(), self.face_vertices(face)?)
    }
}

// Edges bounding one face: none for a vertex, the face itself for a segment
// and three for a triangle.
const fn face_edge_count<const D: usize>() -> usize {
    D * (D - 1) / 2
}

// Sort the first `len` entries ascending, and return the sign of the
// permutation applied. The ascending tuple is the canonical entity key; the
// sign relates the caller's orientation to it.
fn canonical<const N: usize>(mut tuple: [usize; N], len: usize) -> ([usize; N], i8) {
    let sign = permutation_sign(&tuple[..len]);
    tuple[..len].sort_unstable();
    (tuple, sign)
}

// Sign of the permutation that sorts `values`: (-1)^inversions. Shared with
// adaptive simplex faces, whose orientation follows the same convention.
pub(crate) fn permutation_sign(values: &[usize]) -> i8 {
    let inversions = (0..values.len())
        .map(|i| values[i + 1..].iter().filter(|&&v| values[i] > v).count())
        .sum::<usize>();
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

fn validate_cells<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    cells: &[[usize; 4]],
    argument: Argument,
) -> Result<(), SciOxideError> {
    let mut keys = try_with_capacity(
        cells.len(),
        scioxide_core::quantity!("duplicate_cell_validation"),
    )?;
    for cell in cells {
        for &vertex in &cell[..D + 1] {
            get(
                points,
                vertex,
                scioxide_core::quantity!("simplex_cell_vertex"),
            )?;
        }
        let (key, _) = canonical(*cell, D + 1);
        if key[..D + 1].windows(2).any(|w| w[0] == w[1]) {
            return Err(invalid(argument, "cell vertices must be distinct"));
        }
        validate_orientation(points, cell, argument)?;
        keys.push(key);
    }
    // Keys ignore vertex order, so one vertex set listed with different
    // orientations or rotations also counts as a duplicate.
    keys.sort_unstable();
    if keys.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid(argument, "duplicate cells"));
    }
    Ok(())
}

// Fallible copy of fixed-width topology rows.
fn copy<T: Copy>(values: &[T], quantity: scioxide_core::Quantity) -> Result<Vec<T>, SciOxideError> {
    let mut copy = try_with_capacity(values.len(), quantity)?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn validate_orientation<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    cell: &[usize; 4],
    argument: Argument,
) -> Result<(), SciOxideError> {
    let p = |i: usize, axis: usize| points[cell[i]].as_array()[axis];
    let sign = if D == 1 {
        // Comparison of two coordinates is exact.
        match p(1, 0).partial_cmp(&p(0, 0)) {
            Some(core::cmp::Ordering::Greater) => PredicateSign::Positive,
            Some(core::cmp::Ordering::Less) => PredicateSign::Negative,
            _ => PredicateSign::Zero,
        }
    } else if D == 2 {
        orient2d(
            &[p(0, 0), p(0, 1)],
            &[p(1, 0), p(1, 1)],
            &[p(2, 0), p(2, 1)],
        )?
    } else {
        orient3d(
            &[p(0, 0), p(0, 1), p(0, 2)],
            &[p(1, 0), p(1, 1), p(1, 2)],
            &[p(2, 0), p(2, 1), p(2, 2)],
            &[p(3, 0), p(3, 1), p(3, 2)],
        )?
    };
    // Degeneracy and orientation use exact predicates, so no rounded
    // measure can admit a flat or inverted cell. orient3d is positive
    // when d lies below abc, which is the opposite of det(b-a, c-a, d-a).
    // Positive tetrahedra therefore require a Negative result.
    if sign == PredicateSign::Zero {
        return Err(invalid(argument, "cell is exactly degenerate"));
    }
    if sign
        != if D <= 2 {
            PredicateSign::Positive
        } else {
            PredicateSign::Negative
        }
    {
        return Err(invalid(argument, "cell has negative orientation"));
    }
    Ok(())
}

fn build_storage<S: RealScalar, const D: usize>(
    points: Vec<Point<S, D>>,
    cells: Vec<[usize; 4]>,
    argument: Argument,
) -> Result<Storage<S, D>, SciOxideError> {
    let face_slots = checked_mul(
        cells.len(),
        D + 1,
        scioxide_core::quantity!("simplex_face_incidences"),
    )?;
    let edge_slots = checked_mul(
        cells.len(),
        D * (D + 1) / 2,
        scioxide_core::quantity!("simplex_edge_incidences"),
    )?;
    let mut face_records =
        try_with_capacity(face_slots, scioxide_core::quantity!("simplex_face_records"))?;
    let mut edge_records =
        try_with_capacity(edge_slots, scioxide_core::quantity!("simplex_edge_records"))?;
    let mut vertex_pairs = try_with_capacity(
        face_slots,
        scioxide_core::quantity!("simplex_vertex_incidences"),
    )?;
    for (cell_id, cell) in cells.iter().enumerate() {
        for (local, &vertex) in cell[..D + 1].iter().enumerate() {
            vertex_pairs.push((vertex, cell_id));
            let mut face = [0; 3];
            for (out, &v) in face.iter_mut().zip(
                cell[..D + 1]
                    .iter()
                    .enumerate()
                    .filter_map(|(i, v)| (i != local).then_some(v)),
            ) {
                *out = v;
            }
            // Simplicial boundary: the face opposite local vertex i carries
            // (-1)^i, times the parity that maps it to its ascending key.
            let (key, parity) = canonical(face, D);
            face_records.push((
                key,
                cell_id * (D + 1) + local,
                if local % 2 == 0 { parity } else { -parity },
            ));
        }
        let mut local = 0;
        for i in 0..D + 1 {
            for j in i + 1..D + 1 {
                let (key, sign) = canonical([cell[i], cell[j]], 2);
                edge_records.push((key, cell_id * (D * (D + 1) / 2) + local, sign));
                local += 1;
            }
        }
    }
    // Sorting by key groups every occurrence of an entity. Canonical ids are
    // then assigned in ascending lexicographic key order, which the later
    // `binary_search` over `edges` relies on. `slot` is the flat per-cell
    // position, and dividing it by the stride recovers the cell.
    face_records.sort_unstable();
    edge_records.sort_unstable();
    let mut faces = try_with_capacity(face_slots, scioxide_core::quantity!("simplex_faces"))?;
    let mut edges = try_with_capacity(edge_slots, scioxide_core::quantity!("simplex_edges"))?;
    let empty = OrientedEntity { index: 0, sign: 1 };
    let mut cell_faces = try_filled(
        face_slots,
        empty,
        scioxide_core::quantity!("simplex_face_incidences"),
    )?;
    let mut cell_edges = try_filled(
        edge_slots,
        empty,
        scioxide_core::quantity!("simplex_edge_incidences"),
    )?;
    let mut face_pairs =
        try_with_capacity(face_slots, scioxide_core::quantity!("face_cell_pairs"))?;
    let mut edge_pairs =
        try_with_capacity(edge_slots, scioxide_core::quantity!("edge_cell_pairs"))?;
    let mut face_degree = 0;
    let mut first_sign = 0;
    for (key, slot, sign) in face_records {
        // In a consistently oriented manifold, an interior face is shared by
        // exactly two cells that induce opposite boundary orientations on it.
        if faces.last() != Some(&key) {
            faces.push(key);
            face_degree = 0;
            first_sign = sign;
        } else if sign == first_sign {
            return Err(invalid(argument, "shared face orientations must oppose"));
        }
        face_degree += 1;
        if face_degree > 2 {
            return Err(invalid(argument, "nonmanifold face"));
        }
        let face = faces.len() - 1;
        cell_faces[slot] = OrientedEntity { index: face, sign };
        face_pairs.push((face, slot / (D + 1)));
    }
    for (key, slot, sign) in edge_records {
        if edges.last() != Some(&key) {
            edges.push(key);
        }
        let edge = edges.len() - 1;
        cell_edges[slot] = OrientedEntity { index: edge, sign };
        edge_pairs.push((edge, slot / (D * (D + 1) / 2)));
    }
    let mut face_edges = try_with_capacity(
        checked_mul(
            faces.len(),
            face_edge_count::<D>(),
            scioxide_core::quantity!("face_edge_incidences"),
        )?,
        scioxide_core::quantity!("face_edge_incidences"),
    )?;
    // Boundary of an ascending triangle [a, b, c]: [b, c] - [a, c] + [a, b].
    // Every face edge is also a cell edge, so the lookup cannot fail on
    // validated input; the error is defensive.
    for face in &faces {
        let pairs = if D == 2 {
            [[face[0], face[1]], [0, 0], [0, 0]]
        } else {
            [[face[1], face[2]], [face[0], face[2]], [face[0], face[1]]]
        };
        for (local, pair) in pairs[..face_edge_count::<D>()].iter().enumerate() {
            let index = edges
                .binary_search(pair)
                .map_err(|_| invalid(argument, "missing face edge"))?;
            face_edges.push(OrientedEntity {
                index,
                sign: if local == 1 { -1 } else { 1 },
            });
        }
    }
    let face_cells = Incidence::new(faces.len(), face_pairs)?;
    let edge_cells = Incidence::new(edges.len(), edge_pairs)?;
    let vertex_cells = Incidence::new(points.len(), vertex_pairs)?;
    if (0..points.len()).any(|v| vertex_cells.at(v).is_empty()) {
        return Err(invalid(
            argument,
            "unused vertices are not a pure simplex mesh",
        ));
    }
    Ok(Storage {
        points,
        cells,
        faces,
        edges,
        cell_faces,
        cell_edges,
        face_edges,
        face_cells,
        edge_cells,
        vertex_cells,
    })
}

// Traverse the dual of one link. Scratch is reserved once and reused across
// all stars, so the work is proportional to total incidence, not V * C.
//
// Moving between cells only across faces that contain every vertex in
// `vertices` keeps the walk inside the star. Those faces are the link's
// codimension-one adjacencies. A face with one cell is a link boundary
// element. Only the `incident` entries of `visited` are cleared afterwards,
// which restores the all-false invariant without an O(C) reset.
fn connected_link<S: RealScalar, const D: usize>(
    s: &Storage<S, D>,
    vertices: &[usize],
    incident: &[usize],
    visited: &mut [bool],
    stack: &mut Vec<usize>,
    argument: Argument,
) -> Result<usize, SciOxideError> {
    stack.clear();
    stack.push(incident[0]);
    visited[incident[0]] = true;
    let mut count = 0;
    let mut boundary = 0;
    while let Some(cell) = stack.pop() {
        count += 1;
        for face in &s.cell_faces[cell * (D + 1)..(cell + 1) * (D + 1)] {
            if !vertices
                .iter()
                .all(|v| s.faces[face.index][..D].contains(v))
            {
                continue;
            }
            let neighbors = s.face_cells.at(face.index);
            if neighbors.len() == 1 {
                boundary += 1;
            }
            for &other in neighbors {
                if !visited[other] {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }
    }
    for &cell in incident {
        visited[cell] = false;
    }
    if count != incident.len() {
        return Err(invalid(argument, "disconnected simplex link"));
    }
    Ok(boundary)
}
fn validate_links<S: RealScalar, const D: usize>(
    s: &Storage<S, D>,
    argument: Argument,
) -> Result<(), SciOxideError> {
    let mut visited = try_filled(
        s.cells.len(),
        false,
        scioxide_core::quantity!("simplex_link_visited"),
    )?;
    let mut stack = try_with_capacity(
        s.cells.len(),
        scioxide_core::quantity!("simplex_link_traversal"),
    )?;
    let mut vertex_edges = try_filled(
        s.points.len(),
        0_usize,
        scioxide_core::quantity!("vertex_edge_counts"),
    )?;
    let mut vertex_faces = try_filled(
        s.points.len(),
        0_usize,
        scioxide_core::quantity!("vertex_face_counts"),
    )?;
    for edge in &s.edges {
        for &vertex in edge {
            vertex_edges[vertex] += 1;
        }
    }
    for face in &s.faces {
        for &vertex in &face[..D] {
            vertex_faces[vertex] += 1;
        }
    }
    // An interior edge of a tetrahedral manifold has a cycle of tetrahedra
    // around it, with no boundary faces in its star. A boundary edge has a
    // path with exactly two. Anything else is a nonmanifold pinch.
    if D == 3 {
        for (edge, vertices) in s.edges.iter().enumerate() {
            let boundary = connected_link(
                s,
                vertices,
                s.edge_cells.at(edge),
                &mut visited,
                &mut stack,
                argument,
            )?;
            if boundary != 0 && boundary != 2 {
                return Err(invalid(argument, "edge link must be a circle or interval"));
            }
        }
    }
    for vertex in 0..s.points.len() {
        let incident = s.vertex_cells.at(vertex);
        let boundary = connected_link(s, &[vertex], incident, &mut visited, &mut stack, argument)?;
        if D == 1 {
            // The face is the vertex itself: one closing and one opening
            // interval (interior), or a single interval (boundary).
            if boundary > 1 {
                return Err(invalid(
                    argument,
                    "vertex star must be one or two intervals",
                ));
            }
        } else if D == 2 {
            if boundary != 0 && boundary != 2 {
                return Err(invalid(
                    argument,
                    "vertex link must be a circle or interval",
                ));
            }
        } else {
            // A connected orientable manifold link is a sphere or disk exactly
            // when its Euler characteristic is 2 or 1 respectively.
            // The link's vertices, edges and triangles are the incident
            // edges, faces and cells, so chi = E_v - F_v + C_v. The check is
            // rearranged to avoid signed arithmetic.
            let lhs = checked_add(
                vertex_edges[vertex],
                incident.len(),
                scioxide_core::quantity!("vertex_link_euler_count"),
            )?;
            let rhs = checked_add(
                vertex_faces[vertex],
                if boundary == 0 { 2 } else { 1 },
                scioxide_core::quantity!("vertex_link_euler_count"),
            )?;
            if lhs != rhs {
                return Err(invalid(argument, "vertex link must be a sphere or disk"));
            }
        }
    }
    Ok(())
}

/// Labels on boundary faces of a [`SimplexMesh`], for example to select
/// boundary conditions.
///
/// Labels are arbitrary `u32` values; zero has no special meaning. Faces
/// without a label are allowed. The tags store the mesh token.
#[derive(Debug, Clone)]
pub struct SimplexBoundaryTags<const D: usize> {
    token: MeshToken,
    entries: Arc<Vec<(usize, u32)>>,
}
impl<const D: usize> SimplexBoundaryTags<D> {
    /// Creates tags from `(face, tag)` pairs in any order and sorts them by
    /// face. A tag may label many faces.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a face index is invalid for `mesh`.
    /// - `InvalidInput` if a face is interior or appears twice.
    pub fn new<S: RealScalar>(
        mesh: &SimplexMesh<S, D>,
        mut entries: Vec<(usize, u32)>,
    ) -> Result<Self, SciOxideError> {
        const ENTRIES: Argument = scioxide_core::argument!("entries");
        for &(face, _) in &entries {
            if !mesh.is_boundary_face(face)? {
                return Err(invalid(
                    ENTRIES,
                    "only exterior faces may carry boundary tags",
                ));
            }
        }
        entries.sort_unstable_by_key(|&(face, _)| face);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid(ENTRIES, "duplicate boundary tag face"));
        }
        Ok(Self {
            token: mesh.mesh_token().clone(),
            entries: Arc::new(entries),
        })
    }
    /// Returns the token of the mesh the tags were built for.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }
    /// Returns the `(face, tag)` pairs in ascending face order.
    pub fn entries(&self) -> &[(usize, u32)] {
        &self.entries
    }
    /// Returns the number of tagged faces.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Returns whether no face is tagged.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Returns the tag of a face, or `None` if it has none. Interior faces
    /// return `None`. The lookup costs O(log t) for `t` tags.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the tags were built for another revision of
    ///   `mesh`.
    /// - `OutOfBounds` if `face >= mesh.face_count()`.
    pub fn tag<S: RealScalar>(
        &self,
        mesh: &SimplexMesh<S, D>,
        face: usize,
    ) -> Result<Option<u32>, SciOxideError> {
        self.token.ensure_compatible(mesh.mesh_token())?;
        mesh.face_vertices(face)?;
        Ok(self
            .entries
            .binary_search_by_key(&face, |&(id, _)| id)
            .ok()
            .map(|index| self.entries[index].1))
    }
    /// Returns whether the tags were built for this revision of `mesh`.
    pub fn is_compatible_with<S: RealScalar>(&self, mesh: &SimplexMesh<S, D>) -> bool {
        self.token.is_compatible_with(mesh.mesh_token())
    }
}

impl<S: RealScalar> IntervalMesh<S> {
    /// Creates a mesh from plain coordinates and oriented connectivity.
    /// Allocates and validates `n` points in `O(n)` time, then uses [`Self::new`].
    ///
    /// # Errors
    /// Returns `AllocationFailure` for point storage, `NonFinite` for coordinates,
    /// then the connectivity and geometry errors of [`Self::new`].
    pub fn from_coordinates(
        coordinates: &[[S; 1]],
        cells: Vec<[usize; 2]>,
    ) -> Result<Self, SciOxideError> {
        let mut points = try_with_capacity(
            coordinates.len(),
            scioxide_core::quantity!("simplex_coordinates"),
        )?;
        for &coordinate in coordinates {
            points.push(Point::new(coordinate)?);
        }
        Self::new(points, cells)
    }
}

impl<S: RealScalar> TriangleMesh<S> {
    /// Creates a mesh from plain coordinates and oriented connectivity.
    /// Allocates and validates `n` points in `O(n)` time, then uses [`Self::new`].
    ///
    /// # Errors
    /// Returns `AllocationFailure` for point storage, `NonFinite` for coordinates,
    /// then the connectivity and geometry errors of [`Self::new`].
    pub fn from_coordinates(
        coordinates: &[[S; 2]],
        cells: Vec<[usize; 3]>,
    ) -> Result<Self, SciOxideError> {
        let mut points = try_with_capacity(
            coordinates.len(),
            scioxide_core::quantity!("simplex_coordinates"),
        )?;
        for &coordinate in coordinates {
            points.push(Point::new(coordinate)?);
        }
        Self::new(points, cells)
    }
}

impl<S: RealScalar> TetrahedronMesh<S> {
    /// Creates a mesh from plain coordinates and oriented connectivity.
    /// Allocates and validates `n` points in `O(n)` time, then uses [`Self::new`].
    ///
    /// # Errors
    /// Returns `AllocationFailure` for point storage, `NonFinite` for coordinates,
    /// then the connectivity and geometry errors of [`Self::new`].
    pub fn from_coordinates(
        coordinates: &[[S; 3]],
        cells: Vec<[usize; 4]>,
    ) -> Result<Self, SciOxideError> {
        let mut points = try_with_capacity(
            coordinates.len(),
            scioxide_core::quantity!("simplex_coordinates"),
        )?;
        for &coordinate in coordinates {
            points.push(Point::new(coordinate)?);
        }
        Self::new(points, cells)
    }
}
