//! Adaptive meshes: local refinement and coarsening of structured and simplex
//! meshes, with 2:1 balance and topology maps between revisions.
//!
//! Entity identity is decided by exact dyadic reference coordinates, never by
//! rounded coordinates. Transferring field values between revisions is left
//! to the caller.
use crate::validation::inconsistent as invalid;
use crate::{AxisTopology, CellMesh, MeshToken, Point, SimplexMesh};
use alloc::{sync::Arc, vec::Vec};
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    Argument, BudgetKind, RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveScope,
};
mod balance;
mod geometry;
mod index;
use index::Index;
mod partition;
mod refine;
#[cfg(test)]
mod tests;
mod topology;
pub use geometry::AdaptiveCellKind;
pub use partition::{MeshPartition, RepartitionMap, RepartitionOptions};
pub use topology::{
    AdaptiveInterface, AdaptiveInterfaceSide, EntityRelation, HangingEntity, HangingEntityKind,
    TopologyMap,
};

fn at<T>(values: &[T], index: usize) -> Result<&T> {
    crate::validation::get(values, index, scioxide_core::quantity!("adaptive_entity"))
}

type Result<T> = core::result::Result<T, SciOxideError>;
// Preserve the child's counts and the exhausted ancestor's allowance.
fn budget_exhausted(context: &SolveContext<'_, '_>, kind: BudgetKind) -> SciOxideError {
    SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::ResourceLimit(kind),
        scioxide_core::ConvergenceDiagnostics::new()
            .with_counts(context.counts())
            .with_budget(context.effective_budget_report(kind)),
        "adaptive topology allowance exhausted",
    )
}
/// Resource limits of an [`AdaptiveMesh`], kept by all its revisions and
/// partitions.
///
/// Each operation counts from zero. The storage count is an upper bound on
/// the bytes allocated during the operation, without the inputs and earlier
/// revisions. A zero limit is valid and rejects every operation that needs
/// that resource. Exceeding a limit returns `ResourceLimit`. The setters do
/// not check their values; importing a mesh does.
///
/// ```
/// use scioxide_spatial::AdaptationLimits;
/// let limits = AdaptationLimits::new()
///     .with_max_depth(4)
///     .with_max_cells(1000);
/// assert_eq!(limits.max_depth(), 4);
/// assert_eq!(limits.max_cells(), 1000);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptationLimits {
    max_depth: u8,
    max_cells: usize,
    max_entities: usize,
    max_work: usize,
    max_storage_bytes: usize,
}
impl Default for AdaptationLimits {
    fn default() -> Self {
        Self::new()
    }
}
impl AdaptationLimits {
    /// Returns the defaults: depth 20, 1,000,000 cells, 8,000,000 vertices,
    /// faces and edges each, 10⁹ work units and 256 MiB of storage.
    pub const fn new() -> Self {
        Self {
            max_depth: 20,
            max_cells: 1_000_000,
            max_entities: 8_000_000,
            max_work: 1_000_000_000,
            max_storage_bytes: 256 * 1024 * 1024,
        }
    }
    /// Returns the largest refinement level. It is at most 20, so that
    /// reference coordinates stay exact.
    pub const fn max_depth(self) -> u8 {
        self.max_depth
    }
    /// Sets the largest refinement level. Import rejects values above 20.
    pub const fn with_max_depth(mut self, value: u8) -> Self {
        self.max_depth = value;
        self
    }
    /// Returns the largest number of cells in the hierarchy, including
    /// inactive ancestors and children.
    pub const fn max_cells(self) -> usize {
        self.max_cells
    }
    /// Sets the largest number of cells in the hierarchy.
    pub const fn with_max_cells(mut self, value: usize) -> Self {
        self.max_cells = value;
        self
    }
    /// Returns the largest number of vertices, of faces and of edges, each
    /// counted separately.
    pub const fn max_entities(self) -> usize {
        self.max_entities
    }
    /// Sets the largest number of vertices, faces and edges.
    pub const fn with_max_entities(mut self, value: usize) -> Self {
        self.max_entities = value;
        self
    }
    /// Returns the work limit per operation. Copies, comparisons, adjacency
    /// and ancestry visits and emitted relations each count one unit. Under a
    /// solve context, each unit also counts toward its node limit.
    pub const fn max_work(self) -> usize {
        self.max_work
    }
    /// Sets the work limit per operation.
    pub const fn with_max_work(mut self, value: usize) -> Self {
        self.max_work = value;
        self
    }
    /// Returns the storage limit per operation in bytes, without allocator
    /// overhead.
    pub const fn max_storage_bytes(self) -> usize {
        self.max_storage_bytes
    }
    /// Sets the storage limit per operation.
    pub const fn with_max_storage_bytes(mut self, value: usize) -> Self {
        self.max_storage_bytes = value;
        self
    }
}
// Per-call accounting.`source` names its argument whose geometry is imported or adapted.
struct Work<'a, 'b, 'c> {
    limits: AdaptationLimits,
    bytes: usize,
    operations: usize,
    context: Option<&'a mut SolveContext<'b, 'c>>,

    source: Argument,
}
impl<'a, 'b, 'c> Work<'a, 'b, 'c> {
    fn new(
        limits: AdaptationLimits,
        context: Option<&'a mut SolveContext<'b, 'c>>,
        source: Argument,
    ) -> Result<Self> {
        // Depth 20 with at most 3 dyadic exponent bits per level keeps every
        // provenance weight (see `Key::mean`) below 2^64.
        if limits.max_depth > 20 {
            return Err(invalid(
                scioxide_core::argument!("limits.max_depth"),
                "maximum depth exceeds 20",
            ));
        }
        Ok(Self {
            limits,
            bytes: 0,
            operations: 0,
            context,

            source,
        })
    }
    // Inconsistent source geometry of the owning entry point.
    fn invalid(&self, message: &'static str) -> SciOxideError {
        invalid(self.source, message)
    }
    fn tick(&mut self) -> Result<()> {
        self.operations = checked_add(
            self.operations,
            1,
            scioxide_core::quantity!("adaptive_work"),
        )?;
        if self.operations > self.limits.max_work {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_work"),
                self.operations,
                self.limits.max_work,
            ));
        }
        if let Some(context) = self.context.as_deref_mut() {
            if let Some(reason) = context.stop_reason()? {
                return Err(SciOxideError::execution_stopped(reason));
            }
            if !context.try_consume(BudgetKind::Node, 1)?.is_granted() {
                return Err(budget_exhausted(context, BudgetKind::Node));
            }
        }
        Ok(())
    }
    fn reserve<T>(&mut self, values: &mut Vec<T>, additional: usize) -> Result<()> {
        let len = checked_add(
            values.len(),
            additional,
            scioxide_core::quantity!("adaptive_allocation"),
        )?;
        if len > values.capacity() {
            // Charge the entire requested replacement, not just the growth:
            // allocator reallocations may temporarily retain the old payload.
            let preferred = values.capacity().saturating_mul(2).max(4).max(len);
            let available = self.limits.max_storage_bytes.saturating_sub(self.bytes);
            let target = if core::mem::size_of::<T>() == 0 {
                len
            } else {
                preferred.min(available / core::mem::size_of::<T>())
            };
            if target < len {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("adaptive_bytes"),
                    checked_add(
                        self.bytes,
                        checked_mul(
                            len,
                            core::mem::size_of::<T>(),
                            scioxide_core::quantity!("adaptive_bytes"),
                        )?,
                        scioxide_core::quantity!("adaptive_bytes"),
                    )?,
                    self.limits.max_storage_bytes,
                ));
            }
            self.bytes = checked_add(
                self.bytes,
                checked_mul(
                    target,
                    core::mem::size_of::<T>(),
                    scioxide_core::quantity!("adaptive_bytes"),
                )?,
                scioxide_core::quantity!("adaptive_bytes"),
            )?;
            if self.bytes > self.limits.max_storage_bytes {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("adaptive_bytes"),
                    self.bytes,
                    self.limits.max_storage_bytes,
                ));
            }
            values
                .try_reserve_exact(target - values.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("adaptive_storage"),
                        target,
                    )
                })?;
        }
        Ok(())
    }
    fn push<T>(&mut self, values: &mut Vec<T>, value: T) -> Result<()> {
        self.reserve(values, 1)?;
        values.push(value);
        Ok(())
    }
    fn copy<T: Clone>(&mut self, values: &[T]) -> Result<Vec<T>> {
        let mut result = Vec::new();
        self.reserve(&mut result, values.len())?;
        for value in values {
            self.tick()?;
            result.push(value.clone());
        }
        Ok(result)
    }
    fn share<T>(&mut self, value: T) -> Result<Arc<T>> {
        self.bytes = checked_add(
            self.bytes,
            core::mem::size_of::<T>(),
            scioxide_core::quantity!("adaptive_shared_storage"),
        )?;
        if self.bytes > self.limits.max_storage_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_bytes"),
                self.bytes,
                self.limits.max_storage_bytes,
            ));
        }
        Ok(Arc::new(value))
    }
}
// A normalized dyadic convex combination of at most eight original vertices.
// Keys decide vertex identity during refinement; see `Key::mean`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    terms: [(usize, u64); 8],
    len: u8,
    exponent: u8,
}
impl Key {
    fn root(index: usize) -> Self {
        let mut terms = [(0, 0); 8];
        terms[0] = (index, 1);
        Self {
            terms,
            len: 1,
            exponent: 0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Vertex<S: RealScalar, const D: usize> {
    point: Point<S, D>,
    key: Key,
}
// Fixed-width arrays sized for the largest cell (hexahedron: 8 vertices,
// 6 faces, 8 children). Simplices use the leading slots. `reference`
// stores this node's vertices in its parent's reference cell, in half
// units, which is what exact face-ancestry tests use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Node {
    vertices: [usize; 8],
    parent: Option<usize>,
    children: [usize; 8],
    child_count: u8,
    level: u8,
    active: bool,
    faces: [usize; 6],
    signs: [i8; 6],
    reference: [[u8; 3]; 8],
}
impl Node {
    fn root(vertices: [usize; 8]) -> Self {
        Self {
            vertices,
            parent: None,
            children: [0; 8],
            child_count: 0,
            level: 0,
            active: true,
            faces: [0; 6],
            signs: [1; 6],
            reference: [[0; 3]; 8],
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Face {
    vertices: [usize; 4],
    ordered: [usize; 4],
    len: u8,
    parent: Option<usize>,
    parent_sign: i8,
    axis: Option<usize>,
}
#[derive(Debug)]
struct Storage<S: RealScalar, const D: usize> {
    simplex: bool,
    vertices: Vec<Vertex<S, D>>,
    nodes: Vec<Node>,
    faces: Vec<Face>,
    vertex_index: Index<Key>,
    face_index: Index<(u8, [usize; 4])>,
    #[cfg(test)]
    exhaustive: bool,
    active: Vec<usize>,
    active_vertices: Vec<usize>,
    active_faces: Vec<usize>,
    edges: Vec<[usize; 2]>,
    interfaces: Vec<AdaptiveInterface>,
    hanging: Vec<HangingEntity>,
    root_extent: [usize; D],
    periodic: [bool; D],
    // Tensor roots from grids with an odd number of decreasing axes reverse
    // the physical orientation of every reference cell and face.
    reflected: bool,
}
impl<S: RealScalar, const D: usize> Storage<S, D> {
    fn empty(simplex: bool) -> Self {
        Self {
            simplex,
            vertices: Vec::new(),
            nodes: Vec::new(),
            faces: Vec::new(),
            vertex_index: Index::new(),
            face_index: Index::new(),
            #[cfg(test)]
            exhaustive: false,
            active: Vec::new(),
            active_vertices: Vec::new(),
            active_faces: Vec::new(),
            edges: Vec::new(),
            interfaces: Vec::new(),
            hanging: Vec::new(),
            root_extent: [0; D],
            periodic: [false; D],
            reflected: false,
        }
    }
    fn release_indexes(&mut self) {
        self.vertex_index = Index::new();
        self.face_index = Index::new();
    }
    fn vertex_arity(&self) -> usize {
        if self.simplex { D + 1 } else { 1 << D }
    }
    fn face_arity(&self) -> usize {
        if self.simplex { D + 1 } else { 2 * D }
    }
    // Only the append-only hierarchy (vertices, nodes, faces) is copied. The
    // active-entity views are derived data, and `rebuild` regenerates them
    // for the candidate. The source epoch is never mutated.
    fn copy(&self, work: &mut Work<'_, '_, '_>) -> Result<Self> {
        let mut result = Self {
            simplex: self.simplex,
            vertices: work.copy(&self.vertices)?,
            nodes: work.copy(&self.nodes)?,
            faces: work.copy(&self.faces)?,
            vertex_index: Index::new(),
            face_index: Index::new(),
            #[cfg(test)]
            exhaustive: self.exhaustive,
            active: Vec::new(),
            active_vertices: Vec::new(),
            active_faces: Vec::new(),
            edges: Vec::new(),
            interfaces: Vec::new(),
            hanging: Vec::new(),
            root_extent: self.root_extent,
            periodic: self.periodic,
            reflected: self.reflected,
        };
        for (id, vertex) in result.vertices.iter().enumerate() {
            result.vertex_index.insert(vertex.key, id, work)?;
        }
        for (id, face) in result.faces.iter().enumerate() {
            result
                .face_index
                .insert((face.len, face.vertices), id, work)?;
        }
        Ok(result)
    }
}
/// A locally refined mesh of tensor-product cells or simplices.
///
/// The mesh is a forest: each cell of the imported root mesh is the root of a
/// tree, and refining a cell splits it into `2^D` children by halving its
/// edges. The leaves of the forest are the active cells. Hierarchy nodes have
/// stable IDs that survive coarsening and later refinement; active cells,
/// vertices, faces and edges have compact indices `0..count` that are
/// renumbered in each revision.
///
/// The mesh is immutable. [`Self::adapt`] returns a new revision together
/// with a [`TopologyMap`] that relates the two. Import and adaptation cost
/// `O(N log N)` for `N` entities in the hierarchy, including inactive ones.
///
/// ```
/// use scioxide_spatial::{
///     AdaptationLimits, AdaptationRequest, AdaptiveMesh, AxisTopology, CellMesh, CoordinateAxis,
///     Grid2d,
/// };
/// let axis = CoordinateAxis::uniform(0., 1., 3)?;
/// let grid = Grid2d::new([axis.clone(), axis], [AxisTopology::Bounded; 2])?;
/// let limits = AdaptationLimits::new()
///     .with_max_depth(4)
///     .with_max_cells(1000);
/// let mesh = AdaptiveMesh::from_structured(&CellMesh::new(grid)?, limits)?;
/// let candidate = mesh.adapt_scoped(
///     AdaptationRequest {
///         source: mesh.mesh_token(),
///         refine: &[0],
///         coarsen: &[],
///     },
///     scioxide_core::ExecutionContext::sequential(),
/// )?;
/// candidate
///     .map
///     .validate(mesh.mesh_token(), candidate.mesh.mesh_token())?;
/// assert_eq!(candidate.mesh.cell_count(), 7);
/// assert!(
///     candidate
///         .map
///         .cells()
///         .iter()
///         .any(|relation| matches!(relation, scioxide_spatial::EntityRelation::Refined { .. }))
/// );
/// assert!(
///     candidate
///         .map
///         .validate(mesh.mesh_token(), mesh.mesh_token())
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveMesh<S: RealScalar, const D: usize> {
    storage: Arc<Storage<S, D>>,
    token: MeshToken,
    limits: AdaptationLimits,
}
/// Cells to refine and coarsen in one call of [`AdaptiveMesh::adapt`].
#[derive(Clone, Copy, Debug)]
pub struct AdaptationRequest<'a> {
    /// Token of the mesh revision the marks refer to.
    pub source: &'a MeshToken,
    /// Hierarchy IDs of active cells to split once.
    pub refine: &'a [usize],
    /// Hierarchy IDs of parents whose children are all active cells, to merge
    /// back into the parent.
    pub coarsen: &'a [usize],
}
/// Result of [`AdaptiveMesh::adapt`].
#[derive(Debug)]
pub struct AdaptationResult<S: RealScalar, const D: usize> {
    /// The new revision.
    pub mesh: AdaptiveMesh<S, D>,
    /// Relations between the entities of the source and the new revision.
    pub map: TopologyMap,
    /// Hierarchy IDs of cells split in addition to the request, to restore
    /// 2:1 balance across faces, in the order they were split. Each step
    /// splits the smallest ID that violates the balance, but a later step can
    /// add a smaller one, so the list need not be sorted.
    pub induced_refinements: Vec<usize>,
    /// Work units used, as counted for [`AdaptationLimits::max_work`].
    pub work: usize,
}
impl<S: RealScalar, const D: usize> AdaptiveMesh<S, D> {
    /// Creates an adaptive mesh whose roots are the cells of a [`CellMesh`].
    ///
    /// Periodic seams and the orientation of each axis are kept. Face
    /// incidence signs point out of each cell for every axis orientation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `limits.max_depth()` exceeds 20.
    /// - `ResourceLimit` if the mesh exceeds `limits`.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn from_structured(mesh: &CellMesh<S, D>, limits: AdaptationLimits) -> Result<Self> {
        let mut work = Work::new(limits, None, scioxide_core::argument!("mesh"))?;
        let mut storage = Storage::empty(false);
        storage.root_extent = *mesh.vertex_extent().as_array();
        storage.reflected = mesh.grid().orientation() == crate::AxisOrientation::Decreasing;
        for axis in 0..D {
            storage.periodic[axis] = mesh.grid().axis_topology(axis)? == AxisTopology::Periodic;
        }
        for i in 0..mesh.vertex_extent().element_count()? {
            storage.add_root(mesh.point(mesh.vertex_extent().delinearize(i)?)?, &mut work)?;
        }
        for i in 0..mesh.cell_extent().element_count()? {
            let mut vertices = [0; 8];
            for (j, index) in mesh
                .cell_vertices(mesh.cell_extent().delinearize(i)?)?
                .enumerate()
            {
                vertices[j] = mesh.vertex_extent().linearize(index)?;
            }
            storage.add_node(Node::root(vertices), &mut work)?;
        }
        storage.rebuild(&mut work)?;
        storage.release_indexes();
        Ok(Self {
            storage: work.share(storage)?,
            token: mesh.mesh_token().next_generation()?,
            limits,
        })
    }
    /// Creates an adaptive mesh whose roots are the cells of a triangle or
    /// tetrahedron mesh. The checks of [`SimplexMesh`] carry over. An interval
    /// mesh is rejected; build a 1-D adaptive mesh from a [`CellMesh`] with
    /// [`Self::from_structured`] instead.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` is 1 or `limits.max_depth()` exceeds 20.
    /// - `ResourceLimit` if the mesh exceeds `limits`.
    /// - `AllocationFailure` if storage cannot be reserved.
    pub fn from_simplex(mesh: &SimplexMesh<S, D>, limits: AdaptationLimits) -> Result<Self> {
        if D < 2 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("mesh"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "simplex adaptive meshes need triangles or tetrahedra",
            ));
        }
        let mut work = Work::new(limits, None, scioxide_core::argument!("mesh"))?;
        let mut storage = Storage::empty(true);
        for &point in mesh.points() {
            storage.add_root(point, &mut work)?;
        }
        for cell in mesh.cells() {
            let mut vertices = [0; 8];
            vertices[..cell.len()].copy_from_slice(cell);
            storage.add_node(Node::root(vertices), &mut work)?;
        }
        storage.rebuild(&mut work)?;
        storage.release_indexes();
        Ok(Self {
            storage: work.share(storage)?,
            token: mesh.mesh_token().next_generation()?,
            limits,
        })
    }
    /// Returns the mesh token of this revision.
    pub fn mesh_token(&self) -> &MeshToken {
        &self.token
    }
    /// Returns the number of active cells.
    pub fn cell_count(&self) -> usize {
        self.storage.active.len()
    }
    /// Returns the number of vertices. The two sides of a periodic seam count
    /// separately.
    pub fn vertex_count(&self) -> usize {
        self.storage.active_vertices.len()
    }
    /// Returns the number of faces of active cells. A coarse face next to
    /// finer cells counts once, as do each of the finer faces.
    pub fn face_count(&self) -> usize {
        self.storage.active_faces.len()
    }
    /// Returns the number of edges of active cells.
    pub fn edge_count(&self) -> usize {
        self.storage.edges.len()
    }
    /// Returns the number of hierarchy nodes, active or not.
    pub fn hierarchy_len(&self) -> usize {
        self.storage.nodes.len()
    }
    /// Returns the hierarchy ID of each active cell. Active cell `i` is
    /// `active_cells()[i]`; the order is a depth-first traversal from the
    /// roots in order, children in order.
    pub fn active_cells(&self) -> &[usize] {
        &self.storage.active
    }
    /// Returns the parent of a hierarchy node, or `None` for a root.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `id >= self.hierarchy_len()`.
    pub fn parent(&self, id: usize) -> Result<Option<usize>> {
        Ok(at(&self.storage.nodes, id)?.parent)
    }
    /// Returns the children of a hierarchy node. Children stay in the
    /// hierarchy after coarsening, so a node can have children without being
    /// refined.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `id >= self.hierarchy_len()`.
    pub fn children(&self, id: usize) -> Result<&[usize]> {
        let node = at(&self.storage.nodes, id)?;
        Ok(&node.children[..usize::from(node.child_count)])
    }
    /// Returns the refinement level of a hierarchy node; roots have level 0.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `id >= self.hierarchy_len()`.
    pub fn level(&self, id: usize) -> Result<u8> {
        Ok(at(&self.storage.nodes, id)?.level)
    }
    /// Returns the vertices of a child cell in the reference cell of its
    /// parent, in units of one half.
    ///
    /// The reference simplex has vertices at the origin and the unit vectors;
    /// the reference box `[0, 1]^D` numbers its vertices by bits, as
    /// [`CellMesh::cell_vertices`] does.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `id` is a root.
    /// - `OutOfBounds` if `id >= self.hierarchy_len()`.
    pub fn child_reference_vertices(&self, id: usize) -> Result<&[[u8; 3]]> {
        let node = at(&self.storage.nodes, id)?;
        if node.parent.is_none() {
            return Err(invalid(
                scioxide_core::argument!("id"),
                "root has no parent reference map",
            ));
        }
        Ok(&node.reference[..self.storage.vertex_arity()])
    }
    /// Returns the coordinates of a vertex.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.vertex_count()`.
    pub fn point(&self, index: usize) -> Result<Point<S, D>> {
        Ok(self.storage.vertices[*at(&self.storage.active_vertices, index)?].point)
    }
    /// Returns the vertices of an active cell.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index >= self.cell_count()`.
    pub fn cell_vertices(&self, index: usize) -> Result<impl Iterator<Item = usize> + '_> {
        let node = &self.storage.nodes[*at(&self.storage.active, index)?];
        // `active_vertices` is sorted by stable id, so a vertex's compact index
        // is its rank there.
        Ok(node.vertices[..self.storage.vertex_arity()]
            .iter()
            .map(|id| self.storage.active_vertices.partition_point(|v| v < id)))
    }
    /// Returns the interfaces between active cells, including exterior
    /// boundaries and periodic seams.
    pub fn interfaces(&self) -> &[AdaptiveInterface] {
        &self.storage.interfaces
    }
    /// Returns the hanging vertices, edges and faces, each with the coarse
    /// face that contains it.
    pub fn hanging_entities(&self) -> &[HangingEntity] {
        &self.storage.hanging
    }
    /// Returns the resource limits set at import.
    pub const fn limits(&self) -> AdaptationLimits {
        self.limits
    }

    /// Returns a new revision with the marked cells refined or coarsened by
    /// one level, plus the refinements needed for 2:1 balance across faces.
    ///
    /// The source mesh is not changed. With empty marks the result shares the
    /// storage and token of the source, and only the identity map is built.
    /// Under a solve context, each split counts toward its subdivision limit
    /// and each work unit toward its node limit, as described in
    /// [Solve accounting](scioxide_core#solve-accounting). Pass
    /// `ExecutionContext::sequential()` to run without one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `request.source` is not the token of this revision,
    ///   marks contain duplicates, a mark is not an active cell or a parent of
    ///   active cells, or rounding makes a new cell degenerate.
    /// - `OutOfBounds` if a refine or coarsen ID is at least [`Self::hierarchy_len`].
    /// - `ResourceLimit` if a cell would exceed the depth limit or the
    ///   operation exceeds another [`AdaptationLimits`] value.
    /// - `ExecutionStopped` on cancellation or deadline, and
    ///   `ConvergenceFailure` if a limit of the solve context is exhausted.
    pub fn adapt_scoped<'scope, 'capability>(
        &self,
        request: AdaptationRequest<'_>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdaptationResult<S, D>> {
        self.token.ensure_compatible(request.source)?;
        crate::context::scoped(
            scope,
            scioxide_core::algorithm_id!("spatial", "adaptive", 1),
            SolveBudgets::unbounded(),
            |context| {
                self.adapt_impl(
                    request,
                    &mut Work::new(
                        self.limits,
                        Some(context),
                        scioxide_core::argument!("request"),
                    )?,
                )
            },
        )
    }
    // Build a private candidate under the retained limits and local solve epoch.
    fn adapt_impl(
        &self,
        request: AdaptationRequest<'_>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<AdaptationResult<S, D>> {
        let (mesh, induced_refinements) = if request.refine.is_empty() && request.coarsen.is_empty()
        {
            (self.clone(), Vec::new())
        } else {
            self.adapt_candidate(request, work)?
        };
        let map = TopologyMap::build(self, &mesh, work)?;
        work.tick()?;
        if let Some(context) = work.context.as_deref_mut() {
            context.commit_accepted(scioxide_core::PartialResultTrust::Admissible)?;
        }
        Ok(AdaptationResult {
            mesh,
            map,
            induced_refinements,
            work: work.operations,
        })
    }

    fn adapt_candidate(
        &self,
        request: AdaptationRequest<'_>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<(Self, Vec<usize>)> {
        const REFINE: Argument = scioxide_core::argument!("request.refine");
        const COARSEN: Argument = scioxide_core::argument!("request.coarsen");

        let mut refine = work.copy(request.refine)?;
        work.sort(&mut refine)?;
        let mut coarsen = work.copy(request.coarsen)?;
        work.sort(&mut coarsen)?;
        if refine.windows(2).any(|p| p[0] == p[1]) || coarsen.windows(2).any(|p| p[0] == p[1]) {
            let argument = if refine.windows(2).any(|p| p[0] == p[1]) {
                REFINE
            } else {
                COARSEN
            };
            return Err(invalid(argument, "duplicate adaptation marks"));
        }
        for &id in &refine {
            if !at(&self.storage.nodes, id)?.active {
                return Err(invalid(REFINE, "refinement requires an active leaf"));
            }
        }
        for &id in &coarsen {
            let node = at(&self.storage.nodes, id)?;
            if node.active || node.child_count == 0 {
                return Err(invalid(
                    COARSEN,
                    "coarsening requires a complete active child family",
                ));
            }
            for &child in &node.children[..usize::from(node.child_count)] {
                if !self.storage.nodes[child].active || refine.binary_search(&child).is_ok() {
                    return Err(invalid(
                        COARSEN,
                        "coarsening requires unmarked active siblings",
                    ));
                }
            }
        }
        // All marks are validated against the source before anything is
        // copied. Every later failure only drops the private candidate, so
        // `self` and any published epoch stay valid (failure atomicity).
        work.tick()?;
        let mut storage = self.storage.copy(work)?;
        for &id in &coarsen {
            work.tick()?;
            let node = storage.nodes[id];
            for &child in &node.children[..usize::from(node.child_count)] {
                storage.nodes[child].active = false;
            }
            storage.nodes[id].active = true;
        }
        for &id in &refine {
            storage.split(id, work)?;
        }
        let induced = self.balance(&mut storage, &refine, &coarsen, work)?;
        storage.rebuild(work)?;
        storage.release_indexes();
        let mesh = Self {
            storage: work.share(storage)?,
            token: self.token.next_generation()?,
            limits: self.limits,
        };
        Ok((mesh, induced))
    }
}

impl<S: RealScalar, const D: usize> AdaptiveMesh<S, D> {
    /// Adapts the mesh using sequential native execution and the retained limits.
    /// Allocates the candidate mesh as [`Self::adapt_scoped`] does, with identical work and accuracy.
    ///
    /// # Errors
    /// As [`Self::adapt_scoped`].
    pub fn adapt(&self, request: AdaptationRequest<'_>) -> Result<AdaptationResult<S, D>> {
        self.adapt_scoped(
            request,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}
