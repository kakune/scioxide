use super::*;
use crate::SpatialExecutionContext;
use scioxide_core::ExecutionPolicy;
/// Distribution of the active cells of an [`AdaptiveMesh`] over partitions,
/// with one layer of ghost cells.
///
/// Each active cell belongs to one partition. The ghost cells of a partition
/// are the cells of other partitions that share a face with one of its cells.
/// A vertex, face or edge shared by several partitions belongs to the one with
/// the lowest index; entities related across an interface piece or a periodic
/// seam count as shared. Construction costs `O(N log N)` for `N` active
/// entities and interfaces and counts toward the [`AdaptationLimits`] of the
/// mesh.
///
/// ```
/// use scioxide_spatial::{
///     AdaptationLimits, AdaptiveMesh, AxisTopology, CellMesh, CoordinateAxis, Grid1d, MeshPartition,
/// };
/// let grid = Grid1d::new(
///     [CoordinateAxis::uniform(0., 1., 5)?],
///     [AxisTopology::Bounded],
/// )?;
/// let limits = AdaptationLimits::new()
///     .with_max_depth(4)
///     .with_max_cells(1000);
/// let mesh = AdaptiveMesh::from_structured(&CellMesh::new(grid)?, limits)?;
/// let partition = MeshPartition::balanced(&mesh, 2)?;
/// assert_eq!(partition.owned_cells(0)?, &[0, 1]);
/// assert_eq!(partition.ghost_cells(0)?.collect::<Vec<_>>(), vec![2]);
/// assert_eq!(partition.vertex_owner(2)?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct MeshPartition {
    mesh: MeshToken,
    limits: AdaptationLimits,
    identity: Arc<()>,
    parts: usize,
    owners: Vec<usize>,
    order: Vec<usize>,
    ghosts: Vec<(usize, usize)>,
    interfaces: Vec<(usize, usize)>,
    vertex_owners: Vec<usize>,
    face_owners: Vec<usize>,
    edge_owners: Vec<usize>,
}
impl MeshPartition {
    /// Returns the resource limits, taken from the mesh.
    pub const fn limits(&self) -> AdaptationLimits {
        self.limits
    }

    /// Splits the active cells, in the order of
    /// [`AdaptiveMesh::active_cells`], into `parts` contiguous runs of equal
    /// length. When the count does not divide evenly, the lower partitions get
    /// one cell more. Partitions may be empty.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `parts` is zero.
    /// - `ResourceLimit` if the construction exceeds the limits of the mesh.
    pub fn balanced<S: RealScalar, const D: usize>(
        mesh: &AdaptiveMesh<S, D>,
        parts: usize,
    ) -> Result<Self> {
        if parts == 0 {
            return Err(invalid(
                scioxide_core::argument!("parts"),
                "partition count must be positive",
            ));
        }
        let mut work = Work::new(mesh.limits, None, scioxide_core::argument!("mesh"))?;
        // Contiguous blocks of the canonical active order (root, then child
        // traversal), so each part gets a hierarchy-local run of cells. The
        // first `cell_count % parts` parts take one extra cell, and block
        // sizes differ by at most one.
        let mut owners = Vec::new();
        work.reserve(&mut owners, mesh.cell_count())?;
        for part in 0..parts.min(mesh.cell_count()) {
            for _ in 0..mesh.cell_count() / parts + usize::from(part < mesh.cell_count() % parts) {
                work.tick()?;
                owners.push(part);
            }
        }
        Self::build(mesh, parts, &owners, &mut work)
    }
    /// Creates a partition in which active cell `i` belongs to partition
    /// `owners[i]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `parts` is zero, `owners` does not have one entry
    ///   per active cell, or an owner is not less than `parts`.
    /// - `ResourceLimit` if the construction exceeds the limits of the mesh.
    pub fn new<S: RealScalar, const D: usize>(
        mesh: &AdaptiveMesh<S, D>,
        parts: usize,
        owners: &[usize],
    ) -> Result<Self> {
        Self::build(
            mesh,
            parts,
            owners,
            &mut Work::new(mesh.limits, None, scioxide_core::argument!("owners"))?,
        )
    }
    fn build<S: RealScalar, const D: usize>(
        mesh: &AdaptiveMesh<S, D>,
        parts: usize,
        owners: &[usize],
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        if parts == 0 || owners.len() != mesh.cell_count() {
            let argument = if parts == 0 {
                scioxide_core::argument!("parts")
            } else {
                work.source
            };
            return Err(invalid(
                argument,
                "partition assignments must cover every cell with an in-range owner",
            ));
        }
        let mut order = Vec::new();
        work.reserve(&mut order, owners.len())?;
        for (cell, &owner) in owners.iter().enumerate() {
            work.tick()?;
            if owner >= parts {
                return Err(work.invalid("partition assignments must have an in-range owner"));
            }
            order.push(cell);
        }
        // Grouped by owner, ascending within a part. `owned_cells` and
        // `local_cell` then work by binary search, and a part's local
        // numbering follows global order.
        work.sort_by(&mut order, |&a, &b| (owners[a], a).cmp(&(owners[b], b)))?;
        let mut ghosts = Vec::new();
        let mut interfaces = Vec::new();
        for (index, interface) in mesh.interfaces().iter().enumerate() {
            work.tick()?;
            if let Some(right_side) = interface.right {
                let right = right_side.cell;
                let a = owners[interface.left_cell];
                let b = owners[right];
                if a != b {
                    work.push(&mut ghosts, (a, right))?;
                    work.push(&mut ghosts, (b, interface.left_cell))?;
                    work.push(&mut interfaces, (a, index))?;
                    work.push(&mut interfaces, (b, index))?;
                }
            }
        }
        work.sort(&mut ghosts)?;
        ghosts.dedup();
        work.sort(&mut interfaces)?;
        interfaces.dedup();
        // A shared entity belongs to the lowest-numbered part among its
        // incident cells. `parts` is the "unassigned" start value that every
        // real owner undercuts.
        let mut vertex_owners = Vec::new();
        let mut face_owners = Vec::new();
        let mut edge_owners = Vec::new();
        for (out, count) in [
            (&mut vertex_owners, mesh.vertex_count()),
            (&mut face_owners, mesh.face_count()),
            (&mut edge_owners, mesh.edge_count()),
        ] {
            work.reserve(out, count)?;
            for _ in 0..count {
                work.tick()?;
                out.push(parts);
            }
        }
        let storage = &mesh.storage;
        for (cell, &owner) in owners.iter().enumerate() {
            work.tick()?;
            let node = storage.nodes[storage.active[cell]];
            let vertices = &node.vertices[..storage.vertex_arity()];
            for id in vertices {
                let vertex = work
                    .search(&storage.active_vertices, id)?
                    .map_err(|_| work.invalid("missing active vertex"))?;
                vertex_owners[vertex] = vertex_owners[vertex].min(owner);
            }
            for id in &node.faces[..storage.face_arity()] {
                let face = work
                    .search(&storage.active_faces, id)?
                    .map_err(|_| work.invalid("missing active face"))?;
                face_owners[face] = face_owners[face].min(owner);
            }
            for index in incident_edges(vertices, &storage.edges, work)? {
                edge_owners[index] = edge_owners[index].min(owner);
            }
        }
        // Coarse/fine interface pieces belong to both incident cells, even if
        // the coarse cell does not list the fine face in its whole-face array.
        for interface in mesh.interfaces() {
            work.tick()?;
            if let Some(right_side) = interface.right {
                let right = right_side.cell;
                let owner = owners[interface.left_cell].min(owners[right]);
                for face in [interface.left_face, interface.piece_face, right_side.face] {
                    work.tick()?;
                    face_owners[face] = face_owners[face].min(owner);
                }
                let piece = storage.active_faces[interface.piece_face];
                let fine = storage.faces[piece];
                let fine_vertices = &fine.vertices[..usize::from(fine.len)];
                let fine_edges = incident_edges(fine_vertices, &storage.edges, work)?;
                for side in [interface.left_face, right_side.face] {
                    let side_id = storage.active_faces[side];
                    let face = storage.faces[side_id];
                    for &v in &face.vertices[..usize::from(face.len)] {
                        work.tick()?;
                        if fine_vertices.iter().any(|&f| {
                            storage.face_vertex_key(side_id, v) == storage.face_vertex_key(piece, f)
                        }) {
                            let index = work
                                .search(&storage.active_vertices, &v)?
                                .map_err(|_| work.invalid("missing interface vertex"))?;
                            vertex_owners[index] = vertex_owners[index].min(owner);
                        }
                    }
                    for index in incident_edges(
                        &face.vertices[..usize::from(face.len)],
                        &storage.edges,
                        work,
                    )? {
                        let endpoints =
                            storage.edges[index].map(|v| storage.face_vertex_key(side_id, v));
                        for fine_index in fine_edges.clone() {
                            work.tick()?;
                            if storage.edges[fine_index].iter().all(|&v| {
                                super::topology::on_edge(
                                    storage.face_vertex_key(piece, v),
                                    endpoints[0],
                                    endpoints[1],
                                )
                            }) {
                                edge_owners[index] = edge_owners[index].min(owner);
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            mesh: mesh.token.clone(),
            limits: mesh.limits,
            identity: Arc::new(()),
            parts,
            owners: work.copy(owners)?,
            order,
            ghosts,
            interfaces,
            vertex_owners,
            face_owners,
            edge_owners,
        })
    }
    /// Returns the token of the partitioned mesh revision.
    pub fn mesh_token(&self) -> &MeshToken {
        &self.mesh
    }
    /// Returns the number of partitions, including empty ones.
    pub const fn partition_count(&self) -> usize {
        self.parts
    }
    /// Returns the partition of each active cell.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }
    /// Returns the active cells grouped by partition: the owned cells of
    /// partition 0, then those of partition 1, and so on.
    pub fn cell_order(&self) -> &[usize] {
        &self.order
    }
    /// Returns the cells of a partition in increasing order.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `part >= self.partition_count()`.
    pub fn owned_cells(&self, part: usize) -> Result<&[usize]> {
        self.check_part(part)?;
        let start = self.order.partition_point(|&c| self.owners[c] < part);
        let end = self.order.partition_point(|&c| self.owners[c] <= part);
        Ok(&self.order[start..end])
    }
    /// Returns the ghost cells of a partition. Finding them costs `O(log G)`
    /// for `G` stored ghost entries, plus the number returned.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `part >= self.partition_count()`.
    pub fn ghost_cells(&self, part: usize) -> Result<impl Iterator<Item = usize> + '_> {
        self.check_part(part)?;
        Ok(sorted_range(&self.ghosts, part, |&(p, _)| p)
            .iter()
            .map(|&(_, c)| c))
    }
    /// Returns the indices, into [`AdaptiveMesh::interfaces`], of the
    /// interfaces between this partition and others. Finding them costs
    /// `O(log G)` for `G` stored entries, plus the number returned.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `part >= self.partition_count()`.
    pub fn interfaces(&self, part: usize) -> Result<impl Iterator<Item = usize> + '_> {
        self.check_part(part)?;
        Ok(sorted_range(&self.interfaces, part, |&(p, _)| p)
            .iter()
            .map(|&(_, i)| i))
    }
    /// Returns the partition that owns a vertex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index` is not a vertex index.
    pub fn vertex_owner(&self, index: usize) -> Result<usize> {
        at(&self.vertex_owners, index).copied()
    }
    /// Returns the partition that owns a face.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index` is not a face index.
    pub fn face_owner(&self, index: usize) -> Result<usize> {
        at(&self.face_owners, index).copied()
    }
    /// Returns the partition that owns an edge.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index` is not an edge index.
    pub fn edge_owner(&self, index: usize) -> Result<usize> {
        at(&self.edge_owners, index).copied()
    }
    /// Returns the partition of an active cell and the cell's position in
    /// that partition's [`Self::owned_cells`].
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `global` is not an active cell index.
    pub fn local_cell(&self, global: usize) -> Result<(usize, usize)> {
        let owner = *at(&self.owners, global)?;
        let local = self.owned_cells(owner)?.partition_point(|&c| c < global);
        Ok((owner, local))
    }
    fn check_part(&self, part: usize) -> Result<()> {
        if part >= self.parts {
            Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("partition"),
                part,
                self.parts,
            ))
        } else {
            Ok(())
        }
    }
}

// Rows are sorted by key. Borrow only the matching contiguous group, including
// an empty slice for absent keys, without scanning unrelated parts.
fn sorted_range<T, K: Ord>(rows: &[T], key: K, key_of: impl Fn(&T) -> K) -> &[T] {
    let start = rows.partition_point(|row| key_of(row) < key);
    let end = start + rows[start..].partition_point(|row| key_of(row) == key);
    &rows[start..end]
}

// A cell has at most eight vertices (28 pairs); a face has at most four.
// Looking up every pair preserves endpoint-membership incidence, including
// rejecting tensor diagonals, without scanning any unrelated active edges.
fn incident_edges(
    vertices: &[usize],
    edges: &[[usize; 2]],
    work: &mut Work<'_, '_, '_>,
) -> Result<impl Iterator<Item = usize> + Clone + use<>> {
    let mut found = [0; 28];
    let mut len = 0;
    for (i, &a) in vertices.iter().enumerate() {
        for &b in &vertices[..i] {
            work.tick()?;
            if let Ok(index) = work.search(edges, &[a.min(b), a.max(b)])? {
                found[len] = index;
                len += 1;
            }
        }
    }
    Ok(found.into_iter().take(len))
}

#[cfg(test)]
mod tests;
/// Options for a [`RepartitionMap`]: the execution policy of
/// [`RepartitionMap::gather_into`].
///
/// ```
/// use scioxide_core::ExecutionPolicy;
/// use scioxide_spatial::RepartitionOptions;
/// let options = RepartitionOptions::new().with_execution_policy(ExecutionPolicy::Auto);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Auto);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RepartitionOptions {
    policy: ExecutionPolicy,
}
impl Default for RepartitionOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl RepartitionOptions {
    /// Returns sequential execution.
    pub const fn new() -> Self {
        Self {
            policy: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the execution policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
}

/// Moves cell values from one partition of a mesh to another partition of the
/// same mesh revision.
///
/// Cell values are stored grouped by partition, in the order of
/// [`MeshPartition::cell_order`]. The map holds the permutation from the
/// source order to the destination order and the list of cells each pair of
/// partitions exchanges.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_spatial::{AdaptationLimits, AdaptiveMesh, SegmentMesh, MeshPartition, RepartitionMap};
/// let mesh = AdaptiveMesh::from_structured(
///     &SegmentMesh::uniform([0.0_f64], [1.0], [3])?, AdaptationLimits::new())?;
/// let source = MeshPartition::balanced(&mesh, 2)?;
/// let destination = MeshPartition::new(&mesh, 2, &[1, 0])?;
/// let map = RepartitionMap::new(&source, &destination)?;
/// let (mut output, mut staging) = ([0; 2], [0; 2]);
/// map.gather_into(&[10, 20], &mut output, &mut staging, ExecutionContext::sequential())?;
/// assert_eq!(output, [20, 10]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct RepartitionMap {
    options: RepartitionOptions,
    mesh: MeshToken,
    source: Arc<()>,
    destination: Arc<()>,
    source_parts: usize,
    destination_parts: usize,
    permutation: Vec<usize>,
    moves: Vec<(usize, usize, usize)>,
}
impl RepartitionMap {
    /// Creates a sequential redistribution map between two partitions.
    /// Allocates and computes the same permutation as [`Self::with_options`] with default options.
    ///
    /// # Errors
    /// As [`Self::with_options`].
    pub fn new(source: &MeshPartition, destination: &MeshPartition) -> Result<Self> {
        Self::with_options(source, destination, RepartitionOptions::default())
    }

    /// Returns the options the map was created with.
    pub const fn options(&self) -> RepartitionOptions {
        self.options
    }

    /// Creates the map from `source` to `destination`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the two partitions belong to different mesh
    ///   revisions.
    /// - `ResourceLimit` if the construction exceeds the limits of the mesh.
    pub fn with_options(
        source: &MeshPartition,
        destination: &MeshPartition,
        options: RepartitionOptions,
    ) -> Result<Self> {
        source.mesh.ensure_compatible(&destination.mesh)?;
        let mut work = Work::new(source.limits, None, scioxide_core::argument!("source"))?;
        // `permutation[k]` is the source-local position of the cell at
        // destination-local position k. A field stored in source partition
        // order is gathered into destination order with no global renumbering.
        let mut inverse = Vec::new();
        work.reserve(&mut inverse, source.order.len())?;
        inverse.resize(source.order.len(), 0);
        for (local, &global) in source.order.iter().enumerate() {
            inverse[global] = local;
        }
        let mut permutation = Vec::new();
        let mut moves = Vec::new();
        for &global in &destination.order {
            work.tick()?;
            work.push(&mut permutation, inverse[global])?;
            if source.owners[global] != destination.owners[global] {
                work.push(
                    &mut moves,
                    (source.owners[global], destination.owners[global], global),
                )?;
            }
        }
        work.sort(&mut moves)?;
        Ok(Self {
            options,
            mesh: source.mesh.clone(),
            source: Arc::clone(&source.identity),
            destination: Arc::clone(&destination.identity),
            source_parts: source.parts,
            destination_parts: destination.parts,
            permutation,
            moves,
        })
    }
    /// Checks that the map was created from these two partitions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if either partition is not the one the map was
    ///   created from. A partition rebuilt with the same owners does not
    ///   match.
    pub fn validate(&self, source: &MeshPartition, destination: &MeshPartition) -> Result<()> {
        self.mesh.ensure_compatible(&source.mesh)?;
        self.mesh.ensure_compatible(&destination.mesh)?;
        // Partitions of one mesh epoch share its token, so the per-partition
        // anchor is what binds this map to the exact pair it was built from.
        if !Arc::ptr_eq(&self.source, &source.identity)
            || !Arc::ptr_eq(&self.destination, &destination.identity)
        {
            let argument = if Arc::ptr_eq(&self.source, &source.identity) {
                scioxide_core::argument!("destination")
            } else {
                scioxide_core::argument!("source")
            };
            return Err(invalid(argument, "repartition endpoint identity mismatch"));
        }
        Ok(())
    }
    /// Returns, for each position in the destination order, the position of
    /// the same cell in the source order.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    /// Returns the cells that move from partition `source` to partition
    /// `destination`, which is also the list `destination` receives. Finding
    /// them costs `O(log G)` for `G` stored entries, plus the number returned.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `source` or `destination` is not a partition index.
    pub fn migration_cells(
        &self,
        source: usize,
        destination: usize,
    ) -> Result<impl Iterator<Item = usize> + '_> {
        if source >= self.source_parts {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("source_partition"),
                source,
                self.source_parts,
            ));
        }
        if destination >= self.destination_parts {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("destination_partition"),
                destination,
                self.destination_parts,
            ));
        }
        Ok(
            sorted_range(&self.moves, (source, destination), |&(a, b, _)| (a, b))
                .iter()
                .map(|&(_, _, c)| c),
        )
    }
    /// Reorders cell values from the source order to the destination order:
    /// `output[i] = input[permutation()[i]]`.
    ///
    /// `input` and `output` must have one entry per active cell, and
    /// `staging` at least as many; entries of `staging` beyond that are not
    /// changed. The call does not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input`, `output` or `staging` has the wrong
    ///   length.
    /// - `ExecutionStopped` on cancellation or deadline, and errors of the
    ///   executor.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_into<'scope, 'capability, T: Copy + Send + Sync>(
        &self,
        input: &[T],
        output: &mut [T],
        staging: &mut [T],
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<()> {
        self.check_shapes(input.len(), output.len(), staging.len())?;
        let context = context.into();
        let execution = context.execution();
        let staging = &mut staging[..self.permutation.len()];
        crate::batch::dispatch(
            staging,
            self.options.policy,
            execution,
            crate::batch::work(1),
            |i, value| {
                *value = input[self.permutation[i]];
                Ok(())
            },
        )?;
        execution.check()?;
        output.copy_from_slice(staging);
        Ok(())
    }
    fn check_shapes(&self, input: usize, output: usize, staging: usize) -> Result<()> {
        let required = self.permutation.len();
        for actual in [input, output] {
            if actual != required {
                return Err(SciOxideError::dimension_mismatch([required], [actual]));
            }
        }
        if staging < required {
            return Err(SciOxideError::dimension_mismatch([required], [staging]));
        }
        Ok(())
    }
}
