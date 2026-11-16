use super::*;
/// One piece of the interface between two active cells, or between an active
/// cell and the exterior.
///
/// Where a coarse cell meets finer cells, its face is split into one piece
/// per finer face. A piece takes the orientation of its `piece_face`, and the
/// signs say whether the outward normal of each cell agrees with it. Cell and
/// face indices are the compact indices of the current revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveInterface {
    /// The active cell on the left side.
    pub left_cell: usize,
    /// The cell and face on the right side; `None` on the exterior boundary.
    pub right: Option<AdaptiveInterfaceSide>,
    /// The face of the left cell that contains this piece; a coarse face can
    /// contain several pieces.
    pub left_face: usize,
    /// The face that is exactly this piece: the finer of the two faces. On a
    /// periodic seam it lies on one side of the seam.
    pub piece_face: usize,
    /// `1` if the outward normal of the left cell agrees with the orientation
    /// of the piece, `-1` otherwise.
    pub left_sign: i8,
    /// Whether the piece joins the two sides of a periodic seam.
    pub periodic: bool,
}
/// The right side of an [`AdaptiveInterface`]: a cell and its face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveInterfaceSide {
    /// Index of the active cell.
    pub cell: usize,
    /// Index of the face of that cell that contains the piece.
    pub face: usize,
}

impl AdaptiveInterface {
    /// Returns the sign of the outward normal of the right cell relative to
    /// the piece, `-left_sign`, or `None` on the exterior boundary.
    ///
    /// ```
    /// use scioxide_spatial::{AdaptiveInterface, AdaptiveInterfaceSide};
    /// let interface = AdaptiveInterface {
    ///     left_cell: 0,
    ///     left_face: 0,
    ///     piece_face: 0,
    ///     left_sign: 1,
    ///     right: Some(AdaptiveInterfaceSide { cell: 1, face: 2 }),
    ///     periodic: false,
    /// };
    /// assert_eq!(interface.right_sign(), Some(-1));
    /// assert_eq!(
    ///     AdaptiveInterface {
    ///         right: None,
    ///         ..interface
    ///     }
    ///     .right_sign(),
    ///     None
    /// );
    /// ```
    pub fn right_sign(self) -> Option<i8> {
        self.right.map(|_| -self.left_sign)
    }
}

/// Kind of a [`HangingEntity`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HangingEntityKind {
    /// A vertex in the interior of a coarse edge or face.
    Vertex,
    /// An edge that lies in a coarse face but is not an edge of the coarse
    /// cell.
    Edge,
    /// A fine face that lies in a coarse face.
    Face,
}
/// A hanging vertex, edge or face: an entity of a finer cell that lies inside
/// a face of a coarser neighbor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HangingEntity {
    /// Whether the entity is a vertex, edge or face.
    pub kind: HangingEntityKind,
    /// Index of the vertex, edge or face.
    pub entity: usize,
    /// Index of the coarse face that contains it.
    pub coarse_face: usize,
}
/// Relation between an entity of the source revision and one of the
/// destination revision. It states topology only, not interpolation weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityRelation {
    /// The same entity, possibly with a different index.
    Unchanged {
        /// Source index.
        source: usize,
        /// Destination index.
        destination: usize,
    },
    /// The destination is a proper part of the source.
    Refined {
        /// Source index.
        source: usize,
        /// Destination index.
        destination: usize,
        /// `1` if the two have the same orientation, `-1` otherwise; always
        /// `1` for cells.
        orientation: i8,
    },
    /// The source is a proper part of the destination.
    Coarsened {
        /// Source index.
        source: usize,
        /// Destination index.
        destination: usize,
        /// `1` if the two have the same orientation, `-1` otherwise; always
        /// `1` for cells.
        orientation: i8,
    },
    /// A new entity not contained in any source entity of the same dimension.
    Created {
        /// Destination index.
        destination: usize,
    },
    /// A source entity not contained in any destination entity of the same
    /// dimension.
    Removed {
        /// Source index.
        source: usize,
    },
}
/// Relations between the cells, vertices, faces and edges of two revisions of
/// an [`AdaptiveMesh`](crate::AdaptiveMesh), for transferring field values.
///
/// Each list is sorted by source index, then destination index. A source with
/// no relation has one `Removed` entry, and destinations with no source follow
/// as `Created` entries in destination order. Face relations follow the cell
/// hierarchy and do not identify the two sides of a periodic seam. Edge
/// relations follow exact containment.
#[derive(Debug)]
pub struct TopologyMap {
    source: MeshToken,
    destination: MeshToken,
    cells: Vec<EntityRelation>,
    vertices: Vec<EntityRelation>,
    faces: Vec<EntityRelation>,
    edges: Vec<EntityRelation>,
}
impl TopologyMap {
    /// Returns the token of the source revision.
    pub fn source(&self) -> &MeshToken {
        &self.source
    }
    /// Returns the token of the destination revision.
    pub fn destination(&self) -> &MeshToken {
        &self.destination
    }
    /// Checks that the map goes from `source` to `destination`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if either token differs.
    pub fn validate(&self, source: &MeshToken, destination: &MeshToken) -> Result<()> {
        self.source.ensure_compatible(source)?;
        self.destination.ensure_compatible(destination)
    }
    /// Returns the cell relations.
    pub fn cells(&self) -> &[EntityRelation] {
        &self.cells
    }
    /// Returns the vertex relations: `Unchanged`, `Created` or `Removed`. For
    /// the position of a new vertex, use
    /// [`AdaptiveMesh::vertex_provenance`](crate::AdaptiveMesh::vertex_provenance).
    pub fn vertices(&self) -> &[EntityRelation] {
        &self.vertices
    }
    /// Returns the face relations.
    pub fn faces(&self) -> &[EntityRelation] {
        &self.faces
    }
    /// Returns the edge relations.
    pub fn edges(&self) -> &[EntityRelation] {
        &self.edges
    }
}
#[cfg(test)]
mod reference;
mod relations;

impl<S: RealScalar, const D: usize> Storage<S, D> {
    fn collect_active(&mut self, id: usize, work: &mut Work<'_, '_, '_>) -> Result<()> {
        work.tick()?;
        let node = self.nodes[id];
        if node.active {
            work.push(&mut self.active, id)?;
        } else {
            for &child in &node.children[..usize::from(node.child_count)] {
                self.collect_active(child, work)?;
            }
        }
        Ok(())
    }
    // Periodic normalization only on the face-normal endpoint plane; tangential
    // endpoint coordinates remain distinct even on a one-cell periodic root.
    //
    // A face lies on the last plane of a periodic axis when every root
    // contributing to its vertices sits at axis index extent - 1. Roots are
    // numbered axis-zero-fastest, so subtracting (extent - 1) * stride from a
    // root id moves it to the matching root on plane 0. That lets seam
    // partners compare equal as keys.
    fn face_seam(&self, id: usize) -> Option<(usize, usize)> {
        let axis = self.faces[id].axis.filter(|&axis| self.periodic[axis])?;
        let extent = self.root_extent[axis];
        self.face_on_root_plane(id, axis, extent - 1)
            .then(|| (self.root_extent[..axis].iter().product::<usize>(), extent))
    }
    // Whether every root of every face vertex lies on one root plane of `axis`.
    pub(super) fn face_on_root_plane(&self, id: usize, axis: usize, plane: usize) -> bool {
        let face = self.faces[id];
        let stride = self.root_extent[..axis].iter().product::<usize>();
        let extent = self.root_extent[axis];
        face.vertices[..usize::from(face.len)].iter().all(|&v| {
            let key = self.vertices[v].key;
            key.terms[..usize::from(key.len)]
                .iter()
                .all(|&(root, _)| (root / stride) % extent == plane)
        })
    }
    pub(super) fn face_vertex_key(&self, face: usize, vertex: usize) -> Key {
        let mut key = self.vertices[vertex].key;
        if let Some((stride, extent)) = self.face_seam(face) {
            for term in &mut key.terms[..usize::from(key.len)] {
                term.0 -= (extent - 1) * stride;
            }
            key.terms[..usize::from(key.len)].sort_unstable();
        }
        key
    }
    fn face_key(&self, id: usize) -> ([Key; 4], bool) {
        let face = self.faces[id];
        let mut keys = [Key::root(0); 4];
        for (out, &v) in keys.iter_mut().zip(&face.vertices[..usize::from(face.len)]) {
            *out = self.face_vertex_key(id, v);
        }
        keys[..usize::from(face.len)].sort_unstable();
        (keys, self.face_seam(id).is_some())
    }
    pub(super) fn rebuild(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        #[cfg(test)]
        if self.exhaustive {
            return self.rebuild_exhaustive(work);
        }
        self.active.clear();
        self.active_vertices.clear();
        self.active_faces.clear();
        self.edges.clear();
        self.interfaces.clear();
        self.hanging.clear();
        for id in 0..self.nodes.len() {
            if self.nodes[id].parent.is_none() {
                self.collect_active(id, work)?;
            }
        }
        for &id in &self.active {
            let node = self.nodes[id];
            for &v in &node.vertices[..self.vertex_arity()] {
                work.push(&mut self.active_vertices, v)?;
            }
            for &f in &node.faces[..self.face_arity()] {
                work.push(&mut self.active_faces, f)?;
            }
            for i in 0..self.vertex_arity() {
                for j in i + 1..self.vertex_arity() {
                    // Every simplex vertex pair is an edge. Tensor corners in
                    // bit order form an edge only when their indices differ
                    // in exactly one bit; other pairs are diagonals.
                    if self.simplex || (i ^ j).is_power_of_two() {
                        let mut edge = [node.vertices[i], node.vertices[j]];
                        edge.sort_unstable();
                        work.push(&mut self.edges, edge)?;
                    }
                }
            }
        }
        work.sort(&mut self.active_vertices)?;
        self.active_vertices.dedup();
        work.sort(&mut self.active_faces)?;
        self.active_faces.dedup();
        work.sort(&mut self.edges)?;
        self.edges.dedup();
        if self.edges.len() > work.limits.max_entities {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_edges"),
                self.edges.len(),
                work.limits.max_entities,
            ));
        }
        self.rebuild_interfaces(work)
    }
}
mod interfaces;
impl<S: RealScalar, const D: usize> AdaptiveMesh<S, D> {
    /// Returns the vertices of a face.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.face_count()`.
    pub fn face_vertices(&self, index: usize) -> Result<impl Iterator<Item = usize> + '_> {
        let face = &self.storage.faces[*at(&self.storage.active_faces, index)?];
        Ok(face.ordered[..usize::from(face.len)]
            .iter()
            .map(|id| self.storage.active_vertices.partition_point(|v| v < id)))
    }
    /// Returns the two vertices of an edge.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.edge_count()`.
    pub fn edge_vertices(&self, index: usize) -> Result<[usize; 2]> {
        Ok(at(&self.storage.edges, index)?
            .map(|id| self.storage.active_vertices.partition_point(|&v| v < id)))
    }
    /// Returns a vertex as an exact combination of root-mesh vertices.
    ///
    /// The result `(terms, k)` means `v = Σ wᵣ vᵣ / 2ᵏ` over the pairs
    /// `(r, wᵣ)` in `terms`, where `vᵣ` is vertex `r` of the imported mesh.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.vertex_count()`.
    pub fn vertex_provenance(&self, index: usize) -> Result<(&[(usize, u64)], u8)> {
        let key = &self.storage.vertices[*at(&self.storage.active_vertices, index)?].key;
        Ok((&key.terms[..usize::from(key.len)], key.exponent))
    }
    /// Returns the faces of an active cell, each with a sign: `1` when the
    /// normal given by the right-hand rule on the face's vertex order points
    /// out of the cell, `-1` otherwise. This holds for every axis orientation
    /// of the source grid.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.cell_count()`.
    pub fn cell_faces(&self, index: usize) -> Result<impl Iterator<Item = (usize, i8)> + '_> {
        let node = &self.storage.nodes[*at(&self.storage.active, index)?];
        Ok(node.faces[..self.storage.face_arity()]
            .iter()
            .zip(node.signs)
            .map(|(&id, sign)| (self.storage.active_faces.partition_point(|&v| v < id), sign)))
    }
}

// Exact affine containment in provenance space, independent of rounded physical coordinates.
//
// Tests whether point = a + t (b - a) for some t in [0, 1]. The first
// root coordinate where b - a is nonzero fixes t = numerator / denominator.
// Every other coordinate is then checked by cross-multiplication, with no
// division. Aligned weights are at most 2^exponent, so the products fit
// easily in i128. A degenerate edge (a == b) contains only a itself.
pub(super) fn on_edge(point: Key, a: Key, b: Key) -> bool {
    let exponent = point.exponent.max(a.exponent).max(b.exponent);
    let weight = |key: Key, id: usize| -> i128 {
        key.terms[..usize::from(key.len)]
            .iter()
            .find(|t| t.0 == id)
            .map_or(0, |t| i128::from(t.1) << u32::from(exponent - key.exponent))
    };
    let mut pivot = None;
    for &(id, _) in a.terms[..usize::from(a.len)]
        .iter()
        .chain(&b.terms[..usize::from(b.len)])
    {
        let delta = weight(b, id) - weight(a, id);
        if delta != 0 {
            pivot = Some((weight(point, id) - weight(a, id), delta));
            break;
        }
    }
    let Some((numerator, denominator)) = pivot else {
        return point == a;
    };
    if (denominator > 0 && (numerator < 0 || numerator > denominator))
        || (denominator < 0 && (numerator > 0 || numerator < denominator))
    {
        return false;
    }
    point.terms[..usize::from(point.len)]
        .iter()
        .chain(&a.terms[..usize::from(a.len)])
        .chain(&b.terms[..usize::from(b.len)])
        .all(|&(id, _)| {
            (weight(point, id) - weight(a, id)) * denominator
                == (weight(b, id) - weight(a, id)) * numerator
        })
}
