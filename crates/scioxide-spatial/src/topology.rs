//! Tensor-product cell meshes: segments, quadrilaterals and hexahedra.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, checked_product, try_with_capacity};
use scioxide_core::{InvalidInputReason, RealScalar, SciOxideError};

use crate::{AxisSide, AxisTopology, Extent, Index, MeshToken, Point, StructuredGrid};

/// Cell shape of a [`CellMesh`], determined by its dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum CellKind {
    /// One-dimensional segment.
    Segment,
    /// Two-dimensional quadrilateral.
    Quadrilateral,
    /// Three-dimensional hexahedron.
    Hexahedron,
}

/// Entity and incidence counts of a [`CellMesh`].
///
/// Use these counts to size arrays before building explicit connectivity.
/// The mesh itself stores no connectivity tables; it computes incidence from
/// indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct TopologySizes<const D: usize> {
    vertices: usize,
    cells: usize,
    faces_by_normal: [usize; D],
    edges_by_direction: [usize; D],
    faces: usize,
    edges: usize,
    cell_vertex_incidences: usize,
    cell_face_incidences: usize,
}

impl<const D: usize> TopologySizes<D> {
    // Checked tensor-entity counts.
    fn new<S: RealScalar>(grid: &StructuredGrid<S, D>) -> Result<Self, SciOxideError> {
        let vertex_extent = grid.vertex_extent();
        let cell_extent = grid.cell_extent();
        let vertices = vertex_extent.element_count()?;
        let cells = cell_extent.element_count()?;
        let mut faces_by_normal = [0_usize; D];
        let mut edges_by_direction = [0_usize; D];

        // Faces normal to axis a lie at vertex positions along a and at cell
        // positions on the other axes. Edges along a are the dual pattern.
        // Periodic seams keep both endpoint planes as distinct entities,
        // paired by `face_topology`, so the counts do not depend on topology.
        for normal_axis in 0..D {
            let face_extent: [usize; D] = core::array::from_fn(|axis| {
                if axis == normal_axis {
                    vertex_extent.as_array()[axis]
                } else {
                    cell_extent.as_array()[axis]
                }
            });
            faces_by_normal[normal_axis] = checked_product(
                &face_extent,
                scioxide_core::quantity!("structured_face_cardinality"),
            )?;

            let edge_extent: [usize; D] = core::array::from_fn(|axis| {
                if axis == normal_axis {
                    cell_extent.as_array()[axis]
                } else {
                    vertex_extent.as_array()[axis]
                }
            });
            edges_by_direction[normal_axis] = checked_product(
                &edge_extent,
                scioxide_core::quantity!("structured_edge_cardinality"),
            )?;
        }

        let faces = checked_sum(
            &faces_by_normal,
            scioxide_core::quantity!("structured_face_total"),
        )?;
        let edges = checked_sum(
            &edges_by_direction,
            scioxide_core::quantity!("structured_edge_total"),
        )?;
        // StructuredGrid admits only dimensions one through three.
        let vertices_per_cell = 1_usize << D;
        let faces_per_cell = checked_mul(D, 2, scioxide_core::quantity!("cell_face_cardinality"))?;
        let cell_vertex_incidences = checked_mul(
            cells,
            vertices_per_cell,
            scioxide_core::quantity!("cell_to_vertex_incidence_cardinality"),
        )?;
        let cell_face_incidences = checked_mul(
            cells,
            faces_per_cell,
            scioxide_core::quantity!("cell_to_face_incidence_cardinality"),
        )?;

        Ok(Self {
            vertices,
            cells,
            faces_by_normal,
            edges_by_direction,
            faces,
            edges,
            cell_vertex_incidences,
            cell_face_incidences,
        })
    }

    /// Returns the number of vertices.
    pub const fn vertices(self) -> usize {
        self.vertices
    }

    /// Returns the number of cells.
    pub const fn cells(self) -> usize {
        self.cells
    }

    /// Returns the number of faces normal to one axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `normal_axis >= D`.
    pub fn faces_normal_to(self, normal_axis: usize) -> Result<usize, SciOxideError> {
        self.faces_by_normal
            .get(normal_axis)
            .copied()
            .ok_or_else(|| {
                SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("face_normal_axis"),
                    normal_axis,
                    D,
                )
            })
    }

    /// Returns the number of edges parallel to one axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `direction >= D`.
    pub fn edges_along(self, direction: usize) -> Result<usize, SciOxideError> {
        self.edges_by_direction
            .get(direction)
            .copied()
            .ok_or_else(|| {
                SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("edge_direction"),
                    direction,
                    D,
                )
            })
    }

    /// Returns the total number of faces.
    pub const fn faces(self) -> usize {
        self.faces
    }

    /// Returns the total number of edges.
    pub const fn edges(self) -> usize {
        self.edges
    }

    /// Returns the length of a flat cell-to-vertex array: `2^D` per cell.
    pub const fn cell_vertex_incidences(self) -> usize {
        self.cell_vertex_incidences
    }

    /// Returns the length of a flat cell-to-face array: `2D` per cell.
    pub const fn cell_face_incidences(self) -> usize {
        self.cell_face_incidences
    }
}

/// Mesh of segments, quadrilaterals or hexahedra on a [`StructuredGrid`].
///
/// Vertices and cells are identified by their [`Index`]; faces by a
/// [`FaceIndex`]. Incidence is computed from indices on demand, and the
/// iterators do not allocate. On a periodic axis the first and last vertices
/// and faces stay separate entities, paired across the seam. Cloning is O(1).
///
/// ```
/// use scioxide_spatial::{
///     AxisSide, AxisTopology, BoundarySet, CellMesh, CoordinateAxis, Grid2d, Index,
/// };
/// let axis = CoordinateAxis::uniform(0.0_f64, 1.0, 3)?;
/// let mesh = CellMesh::new(Grid2d::new(
///     [axis.clone(), axis],
///     [AxisTopology::Bounded; 2],
/// )?)?;
/// assert_eq!(mesh.cell_vertices(Index::new([0, 0])?)?.count(), 4);
/// let boundary = BoundarySet::axis_side(&mesh, 0, AxisSide::Start)?;
/// assert_eq!(boundary.len(), 2);
/// assert!(mesh.cell_vertices(Index::new([2, 0])?).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct CellMesh<S: RealScalar, const D: usize> {
    grid: StructuredGrid<S, D>,
    sizes: TopologySizes<D>,
}

impl<S: RealScalar, const D: usize> CellMesh<S, D> {
    /// Creates the cell mesh of a grid.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a face, edge or incidence count exceeds `usize`.
    pub fn new(grid: StructuredGrid<S, D>) -> Result<Self, SciOxideError> {
        let sizes = TopologySizes::new(&grid)?;
        Ok(Self { grid, sizes })
    }

    /// Returns the grid.
    pub const fn grid(&self) -> &StructuredGrid<S, D> {
        &self.grid
    }

    /// Returns the grid, consuming the mesh.
    pub fn into_grid(self) -> StructuredGrid<S, D> {
        self.grid
    }

    /// Returns the mesh token, which is the token of the grid.
    pub fn mesh_token(&self) -> &MeshToken {
        self.grid.mesh_token()
    }

    /// Returns the cell shape.
    pub const fn cell_kind(&self) -> CellKind {
        cell_kind(D)
    }

    /// Returns the entity and incidence counts.
    pub const fn topology_sizes(&self) -> TopologySizes<D> {
        self.sizes
    }

    /// Returns the number of vertices along each axis.
    pub fn vertex_extent(&self) -> Extent<D> {
        self.grid.vertex_extent()
    }

    /// Returns the number of cells along each axis.
    pub fn cell_extent(&self) -> Extent<D> {
        self.grid.cell_extent()
    }

    /// Returns the coordinates of a vertex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex` is outside the vertex extent.
    pub fn point(&self, vertex: Index<D>) -> Result<Point<S, D>, SciOxideError> {
        self.grid.point(vertex)
    }

    /// Returns the `2^D` vertices of a cell.
    ///
    /// Corner `k` takes the upper vertex along axis `i` when bit `i` of `k` is
    /// set.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell` is outside the cell extent.
    pub fn cell_vertices(
        &self,
        cell: Index<D>,
    ) -> Result<
        impl ExactSizeIterator<Item = Index<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        self.cell_extent().linearize(cell)?;
        Ok((0..1_usize << D).map(move |corner| {
            Index::from_validated(core::array::from_fn(|axis| {
                cell.as_array()[axis] + ((corner >> axis) & 1)
            }))
        }))
    }

    /// Returns the distinct cells that contain a vertex, with axis 0 varying
    /// fastest.
    ///
    /// A vertex on a periodic seam sees the cells on both sides of the seam,
    /// although the two seam vertices remain distinct entities.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `vertex` is outside the vertex extent.
    pub fn vertex_cells(
        &self,
        vertex: Index<D>,
    ) -> Result<
        impl ExactSizeIterator<Item = Index<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        self.vertex_extent().linearize(vertex)?;
        let mut choices = [[0_usize; 2]; D];
        let mut choice_lengths = [0_usize; D];
        for axis in 0..D {
            let coordinate = vertex.as_array()[axis];
            let cells = self.cell_extent().as_array()[axis];
            let topology = self.grid.axis_topology(axis)?;
            let (values, len) = incident_axis_cells(coordinate, cells, topology);
            choices[axis] = values;
            choice_lengths[axis] = len;
        }

        // Incident cells are the tensor product of per-axis choices, with at
        // most two per axis, so at most 8 in 3D. An odometer with axis zero
        // fastest enumerates them in a fixed order without allocating.
        let len = checked_product(
            &choice_lengths,
            scioxide_core::quantity!("vertex_to_cell_incidence_cardinality"),
        )?;
        let mut cells = crate::indices::DistinctIndices::<D, 8>::new();
        let mut cursors = [0_usize; D];
        for _ in 0..len {
            let index = core::array::from_fn(|axis| choices[axis][cursors[axis]]);
            cells.push(Index::from_validated(index));
            advance_odometer(&mut cursors, &choice_lengths);
        }
        Ok(cells.into_iter())
    }

    /// Returns the `2D` faces of a cell, ordered by normal axis and, within an
    /// axis, the start face before the end face.
    ///
    /// The end face of a cell is the start face of the next cell along that
    /// axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell` is outside the cell extent.
    pub fn cell_faces(
        &self,
        cell: Index<D>,
    ) -> Result<
        impl ExactSizeIterator<Item = FaceIndex<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        self.cell_extent().linearize(cell)?;
        Ok((0..2 * D).map(move |face| {
            let axis = face / 2;
            let mut index = *cell.as_array();
            index[axis] += face % 2;
            FaceIndex::from_validated(axis, Index::from_validated(index))
        }))
    }

    /// Returns the distinct cells adjacent to a face, the start-side cell first.
    ///
    /// A seam face of a periodic axis has the cells on both sides of the
    /// seam; a periodic axis with one cell gives that cell once.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the normal axis or a component of the face index is
    ///   invalid.
    pub fn face_cells(
        &self,
        face: FaceIndex<D>,
    ) -> Result<
        impl ExactSizeIterator<Item = Index<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        validate_face(self, face)?;
        let axis = face.normal_axis;
        let position = face.index.as_array()[axis];
        let cell_count = self.cell_extent().as_array()[axis];
        let topology = self.grid.axis_topology(axis)?;
        // Face position p along the normal separates cells p - 1 and p. On
        // a periodic axis, the two seam planes (p = 0 and p = cell_count)
        // both separate the last and first cell. With a single cell, both
        // sides are the same cell, and deduplication leaves one incidence.
        let mut cells = crate::indices::DistinctIndices::<D, 2>::new();

        if position > 0 {
            let mut index = *face.index.as_array();
            index[axis] -= 1;
            cells.push(Index::from_validated(index));
        } else if topology == AxisTopology::Periodic {
            let mut index = *face.index.as_array();
            index[axis] = cell_count - 1;
            cells.push(Index::from_validated(index));
        }
        if position < cell_count {
            let mut index = *face.index.as_array();
            index[axis] = position;
            cells.push(Index::from_validated(index));
        } else if topology == AxisTopology::Periodic {
            let mut index = *face.index.as_array();
            index[axis] = 0;
            cells.push(Index::from_validated(index));
        }

        Ok(cells.into_iter())
    }

    /// Returns the cells that share a face with `cell`, as
    /// [`StructuredGrid::cell_neighbors`] does.
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
        self.grid.cell_neighbors(cell)
    }

    /// Returns whether a face is interior, on an exterior boundary, or on a
    /// periodic seam.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the face index is invalid.
    pub fn face_topology(&self, face: FaceIndex<D>) -> Result<FaceTopology<D>, SciOxideError> {
        validate_face(self, face)?;
        let axis = face.normal_axis;
        let position = face.index.as_array()[axis];
        let final_position = self.cell_extent().as_array()[axis];
        let side = if position == 0 {
            Some(AxisSide::Start)
        } else if position == final_position {
            Some(AxisSide::End)
        } else {
            None
        };
        let Some(side) = side else {
            return Ok(FaceTopology::Interior);
        };
        if self.grid.axis_topology(axis)? == AxisTopology::Bounded {
            Ok(FaceTopology::Exterior { side })
        } else {
            let mut paired = *face.index.as_array();
            paired[axis] = match side {
                AxisSide::Start => final_position,
                AxisSide::End => 0,
            };
            Ok(FaceTopology::Periodic {
                side,
                paired: FaceIndex::from_validated(axis, Index::from_validated(paired)),
            })
        }
    }

    /// Returns the number of faces on one end of an axis, the product of the
    /// cell counts of the other axes.
    ///
    /// The count is the same for bounded and periodic axes.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `normal_axis >= D`.
    pub fn axis_boundary_len(&self, normal_axis: usize) -> Result<usize, SciOxideError> {
        if normal_axis >= D {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("boundary_normal_axis"),
                normal_axis,
                D,
            ));
        }
        let extent: [usize; D] = core::array::from_fn(|axis| {
            if axis == normal_axis {
                1
            } else {
                self.cell_extent().as_array()[axis]
            }
        });
        checked_product(
            &extent,
            scioxide_core::quantity!("axis_boundary_cardinality"),
        )
    }

    /// Returns the faces on one end of an axis, with the other axes numbered
    /// axis 0 fastest.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `normal_axis >= D`.
    pub fn axis_boundary_faces(
        &self,
        normal_axis: usize,
        side: AxisSide,
    ) -> Result<
        impl ExactSizeIterator<Item = FaceIndex<D>> + Clone + core::fmt::Debug + use<S, D>,
        SciOxideError,
    > {
        let len = self.axis_boundary_len(normal_axis)?;
        let cell_extent = self.cell_extent();
        Ok((0..len).map(move |position| {
            let mut remainder = position;
            let index = core::array::from_fn(|axis| {
                if axis == normal_axis {
                    match side {
                        AxisSide::Start => 0,
                        AxisSide::End => cell_extent.as_array()[axis],
                    }
                } else {
                    let extent = cell_extent.as_array()[axis];
                    let coordinate = remainder % extent;
                    remainder /= extent;
                    coordinate
                }
            });
            FaceIndex::from_validated(normal_axis, Index::from_validated(index))
        }))
    }
}

/// One-dimensional segment mesh.
pub type SegmentMesh<S> = CellMesh<S, 1>;
/// Two-dimensional quadrilateral mesh.
pub type QuadMesh<S> = CellMesh<S, 2>;
/// Three-dimensional hexahedral mesh.
pub type HexMesh<S> = CellMesh<S, 3>;

/// Index of a face of a [`CellMesh`]: its normal axis and position.
///
/// The face normal to axis `a` with index `[i₀, …]` lies on the vertex plane
/// `iₐ` of axis `a`; along the other axes, the components are cell indices.
/// Face indices order by normal axis first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct FaceIndex<const D: usize> {
    normal_axis: usize,
    index: Index<D>,
}

impl<const D: usize> FaceIndex<D> {
    /// Creates a face index. Mesh methods check it against the mesh.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `normal_axis >= D`.
    pub fn new(normal_axis: usize, index: Index<D>) -> Result<Self, SciOxideError> {
        if normal_axis >= D {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("face_normal_axis"),
                normal_axis,
                D,
            ));
        }
        Ok(Self { normal_axis, index })
    }

    const fn from_validated(normal_axis: usize, index: Index<D>) -> Self {
        Self { normal_axis, index }
    }

    /// Returns the axis normal to the face.
    pub const fn normal_axis(self) -> usize {
        self.normal_axis
    }

    /// Returns the position of the face.
    pub const fn index(self) -> Index<D> {
        self.index
    }
}

/// Whether a face of a [`CellMesh`] is interior, exterior or on a periodic seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum FaceTopology<const D: usize> {
    /// The face separates two cells inside the grid.
    Interior,
    /// The face lies on one end of a bounded axis.
    Exterior {
        /// End of the axis.
        side: AxisSide,
    },
    /// The face lies on one end of a periodic axis.
    Periodic {
        /// End of the axis.
        side: AxisSide,
        /// The face on the other end that this face is identified with.
        paired: FaceIndex<D>,
    },
}

/// A set of boundary faces of a [`CellMesh`], for example where a boundary
/// condition applies.
///
/// The faces are exterior or periodic-seam faces, sorted, without duplicates.
/// The set stores the mesh token, so its compatibility with a mesh is
/// checked in O(1).
#[derive(Debug, Clone)]
#[must_use]
pub struct BoundarySet<const D: usize> {
    token: MeshToken,
    faces: Vec<FaceIndex<D>>,
}

impl<const D: usize> BoundarySet<D> {
    /// Creates a set from faces in any order and sorts them.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if a face index is invalid for `mesh`.
    /// - `InvalidInput` if a face is interior or appears twice.
    pub fn new<S: RealScalar>(
        mesh: &CellMesh<S, D>,
        mut faces: Vec<FaceIndex<D>>,
    ) -> Result<Self, SciOxideError> {
        const FACES: scioxide_core::Argument = scioxide_core::argument!("faces");
        for &face in &faces {
            if mesh.face_topology(face)? == FaceTopology::Interior {
                return Err(SciOxideError::invalid_input(
                    FACES,
                    InvalidInputReason::Inconsistent,
                    "boundary sets cannot contain interior faces",
                ));
            }
        }
        faces.sort_unstable();
        if faces.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(SciOxideError::invalid_input(
                FACES,
                InvalidInputReason::Inconsistent,
                "boundary sets cannot contain duplicate faces",
            ));
        }
        Ok(Self {
            token: mesh.mesh_token().clone(),
            faces,
        })
    }

    /// Creates the set of all faces on one end of an axis.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `normal_axis >= D`.
    /// - `SizeOverflow` or `AllocationFailure` if the set cannot be stored.
    pub fn axis_side<S: RealScalar>(
        mesh: &CellMesh<S, D>,
        normal_axis: usize,
        side: AxisSide,
    ) -> Result<Self, SciOxideError> {
        let count = mesh.axis_boundary_len(normal_axis)?;
        let mut faces = try_with_capacity(count, scioxide_core::quantity!("boundary_face_set"))?;
        faces.extend(mesh.axis_boundary_faces(normal_axis, side)?);
        Self::new(mesh, faces)
    }

    /// Returns the token of the mesh the set was built for.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }

    /// Returns the faces in ascending order.
    pub fn faces(&self) -> &[FaceIndex<D>] {
        &self.faces
    }

    /// Returns the number of faces.
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    /// Returns whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// Returns whether the set was built for this revision of `mesh`.
    pub fn is_compatible_with<S: RealScalar>(&self, mesh: &CellMesh<S, D>) -> bool {
        self.token.is_compatible_with(mesh.mesh_token())
    }
}

fn checked_sum(values: &[usize], context: scioxide_core::Quantity) -> Result<usize, SciOxideError> {
    values
        .iter()
        .try_fold(0_usize, |sum, &value| checked_add(sum, value, context))
}

// StructuredGrid construction proves D is one, two or three.
const fn cell_kind(dimension: usize) -> CellKind {
    match dimension {
        1 => CellKind::Segment,
        2 => CellKind::Quadrilateral,
        _ => CellKind::Hexahedron,
    }
}

fn incident_axis_cells(
    coordinate: usize,
    cell_count: usize,
    topology: AxisTopology,
) -> ([usize; 2], usize) {
    // Both retained seam vertices touch the last and first cells, so their
    // incidence stays symmetric with `cell_vertices`. The choices are listed
    // as (last, first) to match the wrapped lower/upper order.
    let mut choices = [0_usize; 2];
    let mut len = 0;
    if topology == AxisTopology::Periodic && (coordinate == 0 || coordinate == cell_count) {
        choices[len] = cell_count - 1;
        len += 1;
        if cell_count > 1 {
            choices[len] = 0;
            len += 1;
        }
    } else {
        if coordinate > 0 {
            choices[len] = coordinate - 1;
            len += 1;
        }
        if coordinate < cell_count {
            choices[len] = coordinate;
            len += 1;
        }
    }
    (choices, len)
}

fn advance_odometer<const D: usize>(cursors: &mut [usize; D], limits: &[usize; D]) {
    for (cursor, &limit) in cursors.iter_mut().zip(limits) {
        *cursor += 1;
        if *cursor < limit {
            return;
        }
        *cursor = 0;
    }
}

// Bounds-checks `face` against `mesh`.
fn validate_face<S: RealScalar, const D: usize>(
    mesh: &CellMesh<S, D>,
    face: FaceIndex<D>,
) -> Result<(), SciOxideError> {
    if face.normal_axis >= D {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("face_normal_axis"),
            face.normal_axis,
            D,
        ));
    }
    for (axis, &coordinate) in face.index.as_array().iter().enumerate() {
        // Along the normal, a face ranges over vertex positions (cells + 1),
        // including both seam planes of a periodic axis.
        let size = if axis == face.normal_axis {
            checked_add(
                mesh.cell_extent().as_array()[axis],
                1,
                scioxide_core::quantity!("face_normal_extent"),
            )?
        } else {
            mesh.cell_extent().as_array()[axis]
        };
        if coordinate >= size {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_face_index"),
                coordinate,
                size,
            ));
        }
    }
    Ok(())
}

impl<S: RealScalar, const D: usize> CellMesh<S, D> {
    /// Creates the cell topology of a bounded uniform grid.
    /// Uses [`StructuredGrid::uniform`] followed by [`Self::new`], with their storage and cost.
    ///
    /// # Errors
    /// As [`StructuredGrid::uniform`] and [`Self::new`].
    pub fn uniform(start: [S; D], end: [S; D], counts: [usize; D]) -> Result<Self, SciOxideError> {
        Self::new(StructuredGrid::uniform(start, end, counts)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{CoordinateAxis, Grid1d, Grid2d, Grid3d, Index1, Index2, Index3};
    use alloc::vec;

    fn quad_mesh(topology: [AxisTopology; 2]) -> QuadMesh<f64> {
        CellMesh::new(
            Grid2d::new(
                [
                    CoordinateAxis::uniform(-2.0, 2.0, 5).unwrap(),
                    CoordinateAxis::uniform(10.0, 13.0, 4).unwrap(),
                ],
                topology,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn topology_sizes_cover_tensor_entities_and_incidences() {
        let mesh = quad_mesh([AxisTopology::Bounded; 2]);
        let sizes = mesh.topology_sizes();
        assert_eq!(mesh.cell_kind(), CellKind::Quadrilateral);
        assert_eq!(sizes.vertices(), 20);
        assert_eq!(sizes.cells(), 12);
        assert_eq!(sizes.faces_normal_to(0).unwrap(), 15);
        assert_eq!(sizes.faces_normal_to(1).unwrap(), 16);
        assert_eq!(sizes.faces(), 31);
        assert_eq!(sizes.edges(), 31);
        assert_eq!(sizes.cell_vertex_incidences(), 48);
        assert_eq!(sizes.cell_face_incidences(), 48);
    }

    #[test]
    fn cell_vertex_and_face_incidence_round_trips() {
        let mesh = quad_mesh([AxisTopology::Bounded; 2]);
        for linear in 0..mesh.cell_extent().element_count().unwrap() {
            let cell = mesh.cell_extent().delinearize(linear).unwrap();
            let vertices = mesh.cell_vertices(cell).unwrap().collect::<Vec<_>>();
            assert_eq!(vertices.len(), 4);
            assert!(vertices.iter().all(|&vertex| {
                mesh.vertex_cells(vertex)
                    .unwrap()
                    .any(|incident| incident == cell)
            }));
            let faces = mesh.cell_faces(cell).unwrap().collect::<Vec<_>>();
            assert_eq!(faces.len(), 4);
            assert!(faces.iter().all(|&face| {
                mesh.face_cells(face)
                    .unwrap()
                    .any(|incident| incident == cell)
            }));
        }
    }

    #[test]
    fn all_supported_cell_kinds_preserve_endpoint_coordinates() {
        let segment = SegmentMesh::new(
            Grid1d::new(
                [CoordinateAxis::uniform(-f64::MAX, f64::MAX, 3).unwrap()],
                [AxisTopology::Bounded],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(segment.cell_kind(), CellKind::Segment);
        assert_eq!(
            segment.point(Index1::new([2]).unwrap()).unwrap().as_array(),
            &[f64::MAX]
        );

        let hex = HexMesh::new(
            Grid3d::new(
                [
                    CoordinateAxis::uniform(0.0, 1.0, 2).unwrap(),
                    CoordinateAxis::uniform(0.0, 1.0, 2).unwrap(),
                    CoordinateAxis::uniform(0.0, 1.0, 2).unwrap(),
                ],
                [AxisTopology::Bounded; 3],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(hex.cell_kind(), CellKind::Hexahedron);
        assert_eq!(
            hex.cell_vertices(Index3::new([0; 3]).unwrap())
                .unwrap()
                .count(),
            8
        );
    }

    #[test]
    fn periodic_endpoint_faces_are_exactly_paired() {
        let mesh = quad_mesh([AxisTopology::Periodic, AxisTopology::Bounded]);
        let start = FaceIndex::new(0, Index2::new([0, 1]).unwrap()).unwrap();
        let end = FaceIndex::new(0, Index2::new([4, 1]).unwrap()).unwrap();
        assert_eq!(
            mesh.face_topology(start).unwrap(),
            FaceTopology::Periodic {
                side: AxisSide::Start,
                paired: end,
            }
        );
        assert_eq!(
            mesh.face_cells(start)
                .unwrap()
                .map(|cell| *cell.as_array())
                .collect::<Vec<_>>(),
            [[3, 1], [0, 1]]
        );
        assert_eq!(mesh.face_cells(end).unwrap().count(), 2);
    }

    #[test]
    fn periodic_vertex_incidence_is_symmetric_at_retained_endpoints() {
        let mesh = SegmentMesh::new(
            Grid1d::new(
                [CoordinateAxis::uniform(0.0, 3.0, 4).unwrap()],
                [AxisTopology::Periodic],
            )
            .unwrap(),
        )
        .unwrap();
        let start = mesh
            .vertex_cells(Index1::new([0]).unwrap())
            .unwrap()
            .collect::<Vec<_>>();
        let end = mesh
            .vertex_cells(Index1::new([3]).unwrap())
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(start, end);
        assert_eq!(
            start,
            [Index1::new([2]).unwrap(), Index1::new([0]).unwrap()]
        );
    }

    #[test]
    fn boundary_sets_validate_entities_and_generation() {
        let mesh = quad_mesh([AxisTopology::Periodic, AxisTopology::Bounded]);
        let boundary = BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap();
        assert_eq!(boundary.len(), 3);
        assert!(boundary.is_compatible_with(&mesh));
        assert!(boundary.faces().iter().all(|&face| matches!(
            mesh.face_topology(face).unwrap(),
            FaceTopology::Periodic { .. }
        )));

        let refined = CellMesh::new(mesh.grid().refined([2, 1]).unwrap()).unwrap();
        assert!(!boundary.is_compatible_with(&refined));

        let interior = FaceIndex::new(0, Index2::new([1, 0]).unwrap()).unwrap();
        assert!(
            matches ! (& (BoundarySet :: new (& mesh , vec ! [interior])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
    }
}
