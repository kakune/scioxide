//! Single-source and all-pairs shortest paths, and path reconstruction.
use super::execution::{Attempt, grow, require};
use super::*;
use scioxide_core::backend::{PartitionedSolve, with_partitioned_solves};
use scioxide_core::{ExecutionContext, NumericalIssue, SolveContext, SolveScope, SolveState};
/// Selects a shortest-path algorithm and, for a single-source method, its
/// source vertex.
///
/// The length of a path is the sum of its edge weights. No method is chosen
/// automatically by a plan; the function [`shortest_paths`] chooses between
/// Dijkstra and Bellman–Ford. Both all-pairs methods first search the whole
/// graph for a negative cycle with Bellman–Ford, at a cost of O(V E). In the
/// costs below, V is the number of vertices and E the number of stored edges.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPathMethod, GraphPlan};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![1], vec![1.0_f64])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// assert!(
///     graph
///         .shortest_paths(GraphPathMethod::BellmanFord { source: 0 })
///         .is_ok()
/// );
/// assert!(graph.shortest_paths(GraphPathMethod::Johnson).is_ok());
/// let error = graph
///     .shortest_paths(GraphPathMethod::Dijkstra { source: 2 })
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - E. W. Dijkstra, "A note on two problems in connexion with graphs",
///   *Numer. Math.* 1 (1959), 269–271.
/// - R. Bellman, "On a routing problem", *Quart. Appl. Math.* 16 (1958), 87–90.
/// - L. R. Ford, Jr., *Network Flow Theory*, RAND Corporation, P-923, 1956.
/// - R. W. Floyd, "Algorithm 97: Shortest path", *Comm. ACM* 5 (1962), 345.
/// - D. B. Johnson, "Efficient algorithms for shortest paths in sparse
///   networks", *J. ACM* 24 (1977), 1–13.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphPathMethod {
    /// Dijkstra's algorithm from one source with a binary heap,
    /// O((V + E) log V). Requires nonnegative weights.
    Dijkstra {
        /// Source vertex.
        source: usize,
    },
    /// Bellman–Ford from one source, O(V E). Accepts negative weights and
    /// detects a negative cycle reachable from the source.
    BellmanFord {
        /// Source vertex.
        source: usize,
    },
    /// Floyd–Warshall for all pairs, O(V³) time and O(V²) output.
    FloydWarshall,
    /// Johnson's algorithm for all pairs: potentials from Bellman–Ford make
    /// every edge weight nonnegative, then Dijkstra's algorithm runs from every
    /// source. O(V E + V (V + E) log V) time and O(V²) output.
    Johnson,
}
impl GraphPathMethod {
    fn all_pairs(self) -> bool {
        matches!(self, Self::FloydWarshall | Self::Johnson)
    }
    // The source of a single-source method.
    fn source(self) -> Option<usize> {
        match self {
            Self::Dijkstra { source } | Self::BellmanFord { source } => Some(source),
            Self::FloydWarshall | Self::Johnson => None,
        }
    }
}
/// Holds either shortest paths or a negative cycle.
///
/// Shortest paths are not defined when a path can run through a cycle of
/// negative total weight. The search then returns
/// [`NegativeCycle`](Self::NegativeCycle) and no distances. Single-source methods
/// report a negative cycle reachable from the source; all-pairs methods report
/// any negative cycle of the graph. In an undirected graph every negative edge
/// forms such a cycle, `u → v → u`. A negative cycle is an outcome of the plan
/// methods and an error of the function [`shortest_paths`], which returns only
/// distances.
///
/// ```
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPathMethod, GraphPlan, PathOutcome};
/// // 0 -> 1 (weight -2) and 1 -> 0 (weight 1) close a cycle of weight -1.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![-2.0_f64, 1.0])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// let outcome = graph.shortest_paths(GraphPathMethod::BellmanFord { source: 0 })?;
/// let PathOutcome::NegativeCycle(cycle) = outcome else {
///     unreachable!()
/// };
/// assert_eq!(cycle.vertices(), &[0, 1, 0]);
/// // Dijkstra rejects negative weights instead of searching for a cycle.
/// assert!(
///     graph
///         .shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
///         .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub enum PathOutcome<P, C = NegativeCycle> {
    /// Shortest paths to every reachable vertex.
    Paths(P),
    /// A cycle of negative total weight.
    NegativeCycle(C),
}
/// Represents a cycle of the graph whose edge weights sum to a negative number.
///
/// The weight of the cycle is recomputed from the graph before it is returned.
/// If rounding makes it nonnegative, the search fails with
/// `NumericalInstability` (`LossOfSignificance`) instead.
#[derive(Debug)]
pub struct NegativeCycle {
    vertices: Vec<usize>,
}
impl NegativeCycle {
    /// Returns the vertices along the cycle. The first vertex is repeated at the
    /// end, so a negative self-loop at `v` is `[v, v]`.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }
}
/// Borrows the distances and predecessors of a shortest-path result.
///
/// A single-source result holds V entries, one per target vertex. An all-pairs
/// result holds V² entries in row-major order: entry `i·V + j` belongs to the
/// pair `(i, j)`. Unreachable targets have distance `None`.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPathMethod, GraphPlan, PathOutcome};
/// // Directed chain 0 -> 1 -> 2 with unit weights.
/// let a = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![1, 2], vec![1.0_f64, 1.0])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// let outcome = graph.shortest_paths(GraphPathMethod::FloydWarshall)?;
/// let PathOutcome::Paths(result) = outcome else {
///     unreachable!()
/// };
/// let paths = result.as_view();
/// // All-pairs distances are row-major: pair (i, j) is entry i * 3 + j.
/// assert_eq!((paths.source(), paths.vertex_count()), (None, 3));
/// assert_eq!(
///     (paths.distances()[2], paths.distances()[2 * 3]),
///     (Some(2.0), None)
/// );
/// assert_eq!(
///     (paths.path(0, 2)?, paths.path(2, 0)?),
///     (Some(vec![0, 1, 2]), None)
/// );
/// assert_eq!(
///     paths.path(0, 3).unwrap_err().kind(),
///     SciOxideErrorKind::OutOfBounds
/// );
/// // A single-source result only reconstructs paths from its own source.
/// let single = scioxide_sparse::shortest_paths(a.view(), 0)?;
/// assert_eq!(single.as_view().path(0, 2)?, Some(vec![0, 1, 2]));
/// assert_eq!(
///     single.as_view().path(1, 2).unwrap_err().kind(),
///     SciOxideErrorKind::InvalidInput
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GraphPathView<'a, S: RealScalar> {
    n: usize,
    source: Option<usize>,
    distances: &'a [Option<S>],
    predecessors: &'a [Option<usize>],
}
impl<'a, S: RealScalar> GraphPathView<'a, S> {
    /// Returns the shortest distances; `None` where the target is unreachable.
    pub fn distances(self) -> &'a [Option<S>] {
        self.distances
    }
    /// Returns the predecessor of each target on a shortest path, in the layout
    /// of [`distances`](Self::distances); `None` for the source itself and for
    /// unreachable targets.
    pub fn predecessors(self) -> &'a [Option<usize>] {
        self.predecessors
    }
    /// Returns the source of a single-source result, or `None` for an all-pairs
    /// result.
    pub fn source(self) -> Option<usize> {
        self.source
    }
    /// Returns the number of vertices V.
    pub fn vertex_count(self) -> usize {
        self.n
    }
    /// Returns a shortest path from `source` to `target` as a list of vertices.
    ///
    /// The path is read backward from the predecessors in O(V) time. The result is
    /// `None` if `target` is unreachable and `[source]` if the two are equal.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `source` or `target` is not a vertex.
    /// - `InvalidInput` if this is a single-source result for a different
    ///   source, or if the predecessors do not lead back to `source`.
    /// - `AllocationFailure` if the path cannot be allocated.
    pub fn path(self, source: usize, target: usize) -> Result<Option<Vec<usize>>, SciOxideError> {
        const SELF: Argument = scioxide_core::argument!("self");
        for vertex in [source, target] {
            if vertex >= self.n {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("graph_vertices"),
                    vertex,
                    self.n,
                ));
            }
        }
        if self.source.is_some_and(|s| s != source) {
            return Err(invalid(
                scioxide_core::argument!("source"),
                "path source differs from the result's fixed source",
            ));
        }
        let offset = if self.source.is_none() {
            source * self.n
        } else {
            0
        };
        if self.distances[offset + target].is_none() {
            return Ok(None);
        }
        let mut path = try_with_capacity(self.n, scioxide_core::quantity!("path"))?;
        let mut v = target;
        for _ in 0..self.n {
            path.push(v);
            if v == source {
                path.reverse();
                return Ok(Some(path));
            }
            v = self.predecessors[offset + v]
                .filter(|&p| p < self.n)
                .ok_or_else(|| invalid(SELF, "broken predecessor chain"))?;
        }
        Err(invalid(SELF, "cyclic predecessor chain"))
    }
}
/// Holds a shortest-path result computed by [`GraphPlan::shortest_paths`] or
/// [`shortest_paths`].
#[derive(Debug)]
pub struct GraphPathResult<S: RealScalar> {
    n: usize,
    source: Option<usize>,
    distances: Vec<Option<S>>,
    predecessors: Vec<Option<usize>>,
}
impl<S: RealScalar> GraphPathResult<S> {
    /// Borrows the result as a [`GraphPathView`].
    pub fn as_view(&self) -> GraphPathView<'_, S> {
        GraphPathView {
            n: self.n,
            source: self.source,
            distances: &self.distances,
            predecessors: &self.predecessors,
        }
    }
}

/// Describes a shortest-path computation for which to size a [`GraphWorkspace`].
#[derive(Clone, Copy, Debug)]
pub struct GraphWorkspaceRequest {
    method: GraphPathMethod,
    options: GraphExecutionOptions,
}
impl GraphWorkspaceRequest {
    /// Creates a request for `method` under `options`.
    ///
    /// All-pairs methods need scratch for each worker the execution policy
    /// allows: at most 64, and at most the number of vertices. Single-source
    /// methods need no worker scratch.
    pub const fn new(method: GraphPathMethod, options: GraphExecutionOptions) -> Self {
        Self { method, options }
    }
}
/// Reports the storage a [`GraphWorkspace`] needs for one request.
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, GraphDirection, GraphExecutionOptions, GraphPathMethod, GraphPlan,
///     GraphWorkspaceRequest,
/// };
/// let a = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![1, 2], vec![1.0_f64, 1.0])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// let request = |method| GraphWorkspaceRequest::new(method, GraphExecutionOptions::new());
/// let single = graph.workspace_requirements(request(GraphPathMethod::Dijkstra { source: 0 }))?;
/// assert_eq!(
///     (single.vertices(), single.output_entries(), single.workers()),
///     (3, 3, 0)
/// );
/// let all = graph.workspace_requirements(request(GraphPathMethod::FloydWarshall))?;
/// assert_eq!(
///     (all.vertices(), all.output_entries(), all.workers()),
///     (3, 9, 1)
/// );
/// assert!(all.bytes::<f64>()? > single.bytes::<f64>()?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphWorkspaceRequirements {
    vertices: usize,
    output_entries: usize,
    cycle_entries: usize,
    workers: usize,
}
impl GraphWorkspaceRequirements {
    /// Returns the length of each per-vertex scratch vector.
    pub const fn vertices(self) -> usize {
        self.vertices
    }
    /// Returns the length of the distance and predecessor outputs: V for
    /// single-source methods and V² for all-pairs methods.
    pub const fn output_entries(self) -> usize {
        self.output_entries
    }
    /// Returns the number of worker scratch sets.
    pub const fn workers(self) -> usize {
        self.workers
    }
    /// Returns the number of bytes of workspace storage for weights of type `S`,
    /// not counting fixed-size fields, vector headers and allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let outputs = checked_mul(
            self.output_entries,
            core::mem::size_of::<Option<S>>() + core::mem::size_of::<Option<usize>>(),
            scioxide_core::quantity!("graph_workspace"),
        )?;
        let scratch = checked_mul(
            self.vertices,
            core::mem::size_of::<Option<S>>()
                + core::mem::size_of::<Option<usize>>()
                + core::mem::size_of::<S>(),
            scioxide_core::quantity!("graph_workspace"),
        )?;
        let workers = checked_mul(
            checked_mul(
                self.vertices,
                self.workers + 1,
                scioxide_core::quantity!("graph_workspace"),
            )?,
            2 * core::mem::size_of::<usize>(),
            scioxide_core::quantity!("graph_workspace"),
        )?;
        let cycles = checked_mul(
            self.cycle_entries,
            core::mem::size_of::<usize>(),
            scioxide_core::quantity!("graph_workspace"),
        )?;
        let ledgers = checked_mul(
            self.workers,
            core::mem::size_of::<PartitionedSolve>() + core::mem::size_of::<SolveState>(),
            scioxide_core::quantity!("graph_workspace"),
        )?;
        checked_add(
            checked_add(
                outputs,
                scratch,
                scioxide_core::quantity!("graph_workspace"),
            )?,
            checked_add(
                checked_add(workers, cycles, scioxide_core::quantity!("graph_workspace"))?,
                ledgers,
                scioxide_core::quantity!("graph_workspace"),
            )?,
            scioxide_core::quantity!("graph_workspace"),
        )
    }
}
#[derive(Debug, Default)]
struct WorkerScratch {
    heap: Vec<usize>,
    positions: Vec<usize>,
    state: SolveState,
}
impl WorkerScratch {
    fn reserve(&mut self, n: usize) -> Result<(), SciOxideError> {
        grow(&mut self.positions, n, usize::MAX)?;
        self.heap
            .try_reserve_exact(n.saturating_sub(self.heap.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("graph_heap"), n)
            })?;
        Ok(())
    }
}
/// Stores the outputs and scratch of [`GraphPlan::shortest_paths_scoped`] for reuse
/// across calls.
///
/// Size it with [`GraphPlan::workspace_requirements`] and
/// [`try_reserve`](Self::try_reserve), or create it with [`GraphPlan::workspace`].
/// A call rejects a workspace that is too small instead of growing it.
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, GraphDirection, GraphExecutionOptions, GraphPathMethod, GraphPlan,
///     GraphWorkspace, GraphWorkspaceRequest, PathOutcome,
/// };
/// // 0 -> 1 (weight 1), 0 -> 2 (weight 5), 1 -> 2 (weight 2).
/// let weights = vec![1.0_f64, 5.0, 2.0];
/// let a = CsrMatrix::from_raw(3, 3, vec![0, 2, 3, 3], vec![1, 2, 2], weights)?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// let (method, options) = (
///     GraphPathMethod::Dijkstra { source: 0 },
///     GraphExecutionOptions::new(),
/// );
/// let requirements = graph.workspace_requirements(GraphWorkspaceRequest::new(method, options))?;
/// let (mut workspace, c) = (GraphWorkspace::new(), ExecutionContext::sequential());
/// // Undersized storage is reported and rejected; execution never grows it.
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// let rejected = graph.shortest_paths_scoped(method, options, &mut workspace, c);
/// assert_eq!(
///     rejected.unwrap_err().kind(),
///     SciOxideErrorKind::DimensionMismatch
/// );
/// workspace.try_reserve(&requirements)?;
/// let outcome = graph.shortest_paths_scoped(method, options, &mut workspace, c)?;
/// let PathOutcome::Paths(paths) = outcome else {
///     unreachable!()
/// };
/// assert_eq!(paths.distances(), &[Some(0.0), Some(1.0), Some(3.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct GraphWorkspace<S: RealScalar> {
    distances: Vec<Option<S>>,
    predecessors: Vec<Option<usize>>,
    row: Vec<Option<S>>,
    parent: Vec<Option<usize>>,
    potentials: Vec<S>,
    cycle: Vec<usize>,
    primary: WorkerScratch,
    workers: Vec<WorkerScratch>,
    ledgers: Vec<PartitionedSolve>,
    attempt: Attempt,
    published: Option<(usize, Option<usize>, bool)>,
}
impl<S: RealScalar> Default for GraphWorkspace<S> {
    fn default() -> Self {
        Self {
            distances: Vec::new(),
            predecessors: Vec::new(),
            row: Vec::new(),
            parent: Vec::new(),
            potentials: Vec::new(),
            cycle: Vec::new(),
            primary: WorkerScratch::default(),
            workers: Vec::new(),
            ledgers: Vec::new(),
            attempt: Attempt::default(),
            published: None,
        }
    }
}
impl<S: RealScalar> GraphWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self::default()
    }
    /// Checks whether the workspace satisfies `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size of `requirements` overflows `usize`.
    pub fn check(
        &self,
        requirements: &GraphWorkspaceRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        match self.require(requirements) {
            Ok(()) => Ok(scioxide_core::Capacity::Sufficient),
            Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::DimensionMismatch => {
                Ok(scioxide_core::Capacity::NeedsGrowth)
            }
            Err(error) => Err(error),
        }
    }

    /// Grows the workspace to satisfy `requirements`.
    ///
    /// Storage never shrinks, so one workspace can serve several requests.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size of `requirements` overflows `usize`.
    /// - `AllocationFailure` if growth fails. The workspace may then be partly
    ///   grown, and stays usable.
    pub fn try_reserve(&mut self, r: &GraphWorkspaceRequirements) -> Result<(), SciOxideError> {
        r.bytes::<S>()?;
        grow(&mut self.distances, r.output_entries, None)?;
        grow(&mut self.predecessors, r.output_entries, None)?;
        grow(&mut self.row, r.vertices, None)?;
        grow(&mut self.parent, r.vertices, None)?;
        grow(&mut self.potentials, r.vertices, S::ZERO)?;
        self.cycle
            .try_reserve_exact(r.cycle_entries.saturating_sub(self.cycle.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("negative_cycle"),
                    r.cycle_entries,
                )
            })?;
        self.primary.reserve(r.vertices)?;
        self.workers
            .try_reserve_exact(r.workers.saturating_sub(self.workers.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("graph_workers"),
                    r.workers,
                )
            })?;
        self.ledgers
            .try_reserve_exact(r.workers.saturating_sub(self.ledgers.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("graph_ledgers"),
                    r.workers,
                )
            })?;
        while self.workers.len() < r.workers {
            self.workers.push(WorkerScratch::default());
        }
        while self.ledgers.len() < r.workers {
            self.ledgers.push(PartitionedSolve::default());
        }
        for worker in self.workers.iter_mut().take(r.workers) {
            worker.reserve(r.vertices)?;
        }
        Ok(())
    }
    fn require(&self, requirements: &GraphWorkspaceRequirements) -> Result<(), SciOxideError> {
        requirements.bytes::<S>()?;

        require([
            (self.distances.len(), requirements.output_entries),
            (self.predecessors.len(), requirements.output_entries),
            (self.row.len(), requirements.vertices),
            (self.parent.len(), requirements.vertices),
            (self.potentials.len(), requirements.vertices),
            (self.cycle.capacity(), requirements.cycle_entries),
            (self.workers.len(), requirements.workers),
            (self.ledgers.len(), requirements.workers),
            (self.primary.positions.len(), requirements.vertices),
            (self.primary.heap.capacity(), requirements.vertices),
        ])?;
        for worker in self.workers.iter().take(requirements.workers) {
            require([
                (worker.positions.len(), requirements.vertices),
                (worker.heap.capacity(), requirements.vertices),
            ])?;
        }
        Ok(())
    }
    /// Returns the record of the most recent call, including a call rejected
    /// before it started.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.attempt.record()
    }
    /// Returns the result of the most recent call if it succeeded, and `None`
    /// otherwise.
    pub fn last_accepted(&self) -> Option<PathOutcome<GraphPathView<'_, S>, &[usize]>> {
        self.published
            .filter(|_| self.attempt.succeeded())
            .map(|(n, source, negative)| self.view(n, source, negative))
    }
    fn view(
        &self,
        n: usize,
        source: Option<usize>,
        negative: bool,
    ) -> PathOutcome<GraphPathView<'_, S>, &[usize]> {
        if negative {
            PathOutcome::NegativeCycle(&self.cycle)
        } else {
            let count = if source.is_none() { n * n } else { n };
            PathOutcome::Paths(GraphPathView {
                n,
                source,
                distances: &self.distances[..count],
                predecessors: &self.predecessors[..count],
            })
        }
    }
}
type Tick<'a> = dyn FnMut() -> Result<(), SciOxideError> + 'a;
impl<S: RealScalar> GraphPlan<S> {
    /// Returns the storage that
    /// [`shortest_paths_scoped`](Self::shortest_paths_scoped) needs for
    /// `request`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if V² (for an all-pairs method) or V + 1 overflows
    ///   `usize`.
    pub fn workspace_requirements(
        &self,
        request: GraphWorkspaceRequest,
    ) -> Result<GraphWorkspaceRequirements, SciOxideError> {
        let n = self.vertex_count();
        Ok(GraphWorkspaceRequirements {
            vertices: n,
            output_entries: if request.method.all_pairs() {
                checked_mul(n, n, scioxide_core::quantity!("output_entries"))?
            } else {
                n
            },
            cycle_entries: checked_add(n, 1, scioxide_core::quantity!("cycle_entries"))?,
            workers: if request.method.all_pairs() {
                request.options.workers(n)
            } else {
                0
            },
        })
    }
    /// Creates a workspace sized for `request`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        &self,
        request: GraphWorkspaceRequest,
    ) -> Result<GraphWorkspace<S>, SciOxideError> {
        let mut workspace = GraphWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Computes shortest paths with `method` and returns an owned result.
    ///
    /// This is [`shortest_paths_scoped`](Self::shortest_paths_scoped) with
    /// [`GraphExecutionOptions::new`] in a fresh scope: the call allocates its
    /// workspace and runs on the calling thread with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPathMethod, GraphPlan, PathOutcome};
    ///
    /// // Edges 0 → 1 (1), 0 → 2 (5), 1 → 2 (2): the shortest path to 2 has length 1 + 2.
    /// let a = CsrMatrix::from_raw(
    ///     3,
    ///     3,
    ///     vec![0, 2, 3, 3],
    ///     vec![1, 2, 2],
    ///     vec![1.0_f64, 5.0, 2.0],
    /// )?;
    /// let graph = GraphPlan::new(a.view(), GraphDirection::Directed)?;
    /// let outcome = graph.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })?;
    /// let PathOutcome::Paths(paths) = outcome else {
    ///     unreachable!()
    /// };
    /// assert_eq!(
    ///     paths.as_view().distances(),
    ///     &[Some(0.0), Some(1.0), Some(3.0)]
    /// );
    /// assert_eq!(paths.as_view().path(0, 2)?, Some(vec![0, 1, 2]));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`shortest_paths_scoped`](Self::shortest_paths_scoped)
    /// other than `ConvergenceFailure` and `ExecutionStopped`, and
    /// `SizeOverflow` or `AllocationFailure` if the workspace cannot be created.
    pub fn shortest_paths(
        &self,
        method: GraphPathMethod,
    ) -> Result<PathOutcome<GraphPathResult<S>>, SciOxideError> {
        let options = GraphExecutionOptions::new();
        let mut workspace = self.workspace(GraphWorkspaceRequest::new(method, options))?;
        let negative = matches!(
            self.shortest_paths_scoped(
                method,
                options,
                &mut workspace,
                ExecutionContext::sequential_native()
            )?,
            PathOutcome::NegativeCycle(_)
        );
        if negative {
            Ok(PathOutcome::NegativeCycle(NegativeCycle {
                vertices: workspace.cycle,
            }))
        } else {
            Ok(PathOutcome::Paths(GraphPathResult {
                n: self.vertex_count(),
                source: method.source(),
                distances: workspace.distances,
                predecessors: workspace.predecessors,
            }))
        }
    }
    /// Computes shortest paths into a workspace, in a root or nested solve scope.
    ///
    /// The result borrows the workspace until the next call. One iteration is
    /// counted per Bellman–Ford pass, per vertex taken from the Dijkstra heap and
    /// per Floyd–Warshall pivot. Johnson's method runs the search from each source
    /// in a nested scope with the limits of `options`. The all-pairs methods split
    /// the sources between workers according to the execution policy; each row of
    /// the result is computed independently, so it does not depend on the number of
    /// workers. Single-source methods run sequentially.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if Dijkstra's method meets a negative weight.
    /// - `OutOfBounds` if the source of a single-source method is not a vertex.
    /// - `DimensionMismatch` if the workspace is smaller than
    ///   [`workspace_requirements`](Self::workspace_requirements) reports.
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `NumericalInstability`: `Overflow` if a distance overflows;
    ///   `LossOfSignificance` if rounding makes a Johnson-reweighted edge negative or a
    ///   detected negative cycle nonnegative; `Breakdown` if rounding lowers a
    ///   distance that Dijkstra's method has already settled.
    /// - `ConvergenceFailure` if the iteration limit is exhausted.
    /// - `ExecutionStopped` on cancellation or a passed deadline;
    ///   `CapabilityUnavailable`, `CapabilityFailure`, `ExecutionFailure` or
    ///   `AccountingInvariant` if the scope or its executor cannot run the call.
    pub fn shortest_paths_scoped<'w, 'scope, 'capability>(
        &self,
        method: GraphPathMethod,
        options: GraphExecutionOptions,
        workspace: &'w mut GraphWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PathOutcome<GraphPathView<'w, S>, &'w [usize]>, SciOxideError> {
        workspace.attempt.reject();
        workspace.published = None;
        let source = method.source();
        if let Some(source) = source {
            self.vertex(source)?;
        }
        if matches!(method, GraphPathMethod::Dijkstra { .. })
            && self.edges.iter().any(|e| e.2 < S::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "Dijkstra requires nonnegative weights",
            ));
        }
        workspace
            .require(&self.workspace_requirements(GraphWorkspaceRequest::new(method, options))?)?;
        let mut attempt = core::mem::take(&mut workspace.attempt);
        let result = attempt.run(
            scope,
            options,
            scioxide_core::algorithm_id!("sparse", "paths", 1),
            |context| self.run_paths(method, source, options, workspace, context),
        );
        workspace.attempt = attempt;
        let negative = result?;
        workspace.published = Some((self.vertex_count(), source, negative));
        Ok(workspace.view(self.vertex_count(), source, negative))
    }
    fn run_paths(
        &self,
        method: GraphPathMethod,
        source: Option<usize>,
        options: GraphExecutionOptions,
        workspace: &mut GraphWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = self.vertex_count();
        let count = if method.all_pairs() { n * n } else { n };
        let GraphWorkspace {
            distances,
            predecessors,
            row,
            parent,
            potentials,
            cycle,
            primary,
            workers,
            ledgers,
            ..
        } = workspace;
        let d = &mut distances[..count];
        let p = &mut predecessors[..count];
        let row = &mut row[..n];
        let parent = &mut parent[..n];
        let h = &mut potentials[..n];
        cycle.clear();
        if !method.all_pairs() {
            return match method {
                GraphPathMethod::Dijkstra { .. } => {
                    self.dijkstra(
                        source.ok_or_else(|| {
                            invalid(scioxide_core::argument!("source"), "missing source")
                        })?,
                        d,
                        p,
                        None,
                        &mut primary.heap,
                        &mut primary.positions[..n],
                        &mut || super::admit(context),
                    )?;
                    Ok(false)
                }
                _ => self.bellman(source, (d, p), cycle, &mut || super::admit(context)),
            };
        }
        if self.bellman(None, (row, parent), cycle, &mut || super::admit(context))? {
            return Ok(true);
        }
        d.fill(None);
        p.fill(None);
        let bound = options.workers(n);
        if method == GraphPathMethod::Johnson {
            for (h, value) in h.iter_mut().zip(row.iter()) {
                *h = value.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("workspace"),
                        "missing Johnson potential",
                    )
                })?;
            }
            for &(u, v, weight) in &self.edges {
                self.reweighted((u, v), weight, h)?;
            }
            dispatch_rows(
                n,
                d,
                p,
                &mut workers[..bound],
                &mut ledgers[..bound],
                options,
                context,
                &|source, d, p, worker, parent| {
                    let WorkerScratch {
                        heap,
                        positions,
                        state,
                    } = worker;
                    SolveScope::from(parent).run(
                        state,
                        super::execution::identity(scioxide_core::algorithm_id!(
                            "sparse", "dijkstra", 1
                        )),
                        options.budgets(),
                        |child| {
                            self.dijkstra(
                                source,
                                d,
                                p,
                                Some(h),
                                heap,
                                &mut positions[..n],
                                &mut || super::admit(child),
                            )?;
                            restore_distances(source, d, h)
                        },
                    )
                },
            )?;
        } else {
            for i in 0..n {
                d[i * n + i] = Some(S::ZERO);
            }
            for &(u, v, weight) in &self.edges {
                if d[u * n + v].is_none_or(|old| weight < old) {
                    d[u * n + v] = Some(weight);
                    p[u * n + v] = Some(u);
                }
            }
            for k in 0..n {
                super::admit(context)?;
                row.copy_from_slice(&d[k * n..(k + 1) * n]);
                parent.copy_from_slice(&p[k * n..(k + 1) * n]);
                dispatch_rows(
                    n,
                    d,
                    p,
                    &mut workers[..bound],
                    &mut ledgers[..bound],
                    options,
                    context,
                    &|_, d, p, _, _| floyd_row(k, (d, p), (row, parent)),
                )?;
            }
        }
        Ok(false)
    }
    // Bellman-Ford with a negative-cycle witness.
    fn bellman(
        &self,
        source: Option<usize>,
        (d, p): (&mut [Option<S>], &mut [Option<usize>]),
        cycle: &mut Vec<usize>,
        tick: &mut Tick<'_>,
    ) -> Result<bool, SciOxideError> {
        const SELF: Argument = scioxide_core::argument!("self");
        let n = self.vertex_count();
        d.fill(if source.is_none() {
            Some(S::ZERO)
        } else {
            None
        });
        p.fill(None);
        if let Some(s) = source {
            d[s] = Some(S::ZERO);
        }
        for pass in 0..n {
            tick()?;
            let mut changed = None;
            for &(u, v, weight) in &self.edges {
                if let Some(left) = d[u] {
                    let candidate = finite_candidate(left + weight)?;
                    if d[v].is_none_or(|old| candidate < old) {
                        d[v] = Some(candidate);
                        p[v] = Some(u);
                        changed = Some(v);
                    }
                }
            }
            let Some(mut v) = changed else {
                return Ok(false);
            };
            if pass + 1 == n {
                for _ in 0..n {
                    v = p[v].ok_or_else(|| invalid(SELF, "missing cycle predecessor"))?;
                }
                let start = v;
                cycle.clear();
                cycle.push(start);
                loop {
                    v = p[v].ok_or_else(|| invalid(SELF, "missing cycle predecessor"))?;
                    cycle.push(v);
                    if v == start {
                        break;
                    }
                    if cycle.len() > n {
                        return Err(invalid(SELF, "invalid negative cycle"));
                    }
                }
                cycle.reverse();
                let mut weight = S::ZERO;
                for pair in cycle.windows(2) {
                    weight = finite_candidate(
                        weight
                            + self
                                .weight(pair[0], pair[1])
                                .ok_or_else(|| invalid(SELF, "cycle edge absent"))?,
                    )?;
                }
                if weight >= S::ZERO {
                    return Err(SciOxideError::numerical_instability(
                        NumericalIssue::LossOfSignificance,
                        "negative-cycle witness is not negative in scalar arithmetic",
                    ));
                }
                return Ok(true);
            }
        }
        Ok(false)
    }
    // Johnson-reduced edge weight.
    fn reweighted(&self, (u, v): (usize, usize), weight: S, h: &[S]) -> Result<S, SciOxideError> {
        let value = finite_candidate(finite_candidate(weight + h[u])? - h[v])?;
        if value < S::ZERO {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::LossOfSignificance,
                "Johnson reduced edge is negative after rounding",
            ))
        } else {
            Ok(value)
        }
    }
    // Binary-heap Dijkstra.
    #[allow(clippy::too_many_arguments)]
    fn dijkstra(
        &self,
        source: usize,
        d: &mut [Option<S>],
        p: &mut [Option<usize>],
        h: Option<&[S]>,
        heap: &mut Vec<usize>,
        positions: &mut [usize],
        tick: &mut Tick<'_>,
    ) -> Result<(), SciOxideError> {
        const SELF: Argument = scioxide_core::argument!("self");
        let n = self.vertex_count();
        d.fill(None);
        p.fill(None);
        positions.fill(usize::MAX);
        heap.clear();
        d[source] = Some(S::ZERO);
        heap.push(source);
        positions[source] = 0;
        while !heap.is_empty() {
            tick()?;
            let u =
                super::heap::pop(heap, positions, d).ok_or_else(|| invalid(SELF, "empty heap"))?;
            let left = d[u].ok_or_else(|| invalid(SELF, "heap vertex is unreachable"))?;
            for &(_, v, weight) in self.row(u) {
                let weight = if let Some(h) = h {
                    self.reweighted((u, v), weight, h)?
                } else {
                    weight
                };
                let candidate = finite_candidate(left + weight)?;
                if d[v].is_none_or(|old| candidate < old) {
                    if positions[v] == n {
                        return Err(SciOxideError::numerical_instability(
                            NumericalIssue::Breakdown,
                            "settled Dijkstra distance decreased",
                        ));
                    }
                    d[v] = Some(candidate);
                    p[v] = Some(u);
                    super::heap::decrease(heap, positions, d, v);
                }
            }
        }
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
fn dispatch_rows<S: RealScalar>(
    n: usize,
    d: &mut [Option<S>],
    p: &mut [Option<usize>],
    workers: &mut [WorkerScratch],
    ledgers: &mut [PartitionedSolve],
    options: GraphExecutionOptions,
    context: &mut SolveContext<'_, '_>,
    compute: &(
         impl Fn(
        usize,
        &mut [Option<S>],
        &mut [Option<usize>],
        &mut WorkerScratch,
        &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
         + Sync
     ),
) -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    if n == 0 {
        return Ok(());
    }
    let plan = context.execution().plan(
        options.execution_policy(),
        n,
        scioxide_core::PlanLimits {
            work_per_item: NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::MIN.saturating_add(workers.len() - 1),
        },
    );
    with_partitioned_solves(
        context,
        &mut ledgers[..plan.worker_count()],
        scioxide_core::SolveBudgets::unbounded(),
        |execution, batch| {
            let mut rest_d = d;
            let mut rest_p = p;
            let inputs = plan.partitions().zip(workers.iter_mut()).zip(batch).map(
                |((partition, worker), permit)| {
                    let count = partition.range().len() * n;
                    let (d, tail) = core::mem::take(&mut rest_d).split_at_mut(count);
                    rest_d = tail;
                    let (p, tail) = core::mem::take(&mut rest_p).split_at_mut(count);
                    rest_p = tail;
                    (d, p, worker, permit)
                },
            );
            crate::execution::dispatch(
                execution,
                plan,
                inputs,
                &|partition, (d, p, worker, permit), partition_context| {
                    permit.run(
                        partition_context,
                        super::execution::identity(scioxide_core::algorithm_id!(
                            "sparse",
                            "path_rows",
                            1
                        )),
                        |parent| {
                            for ((source, d), p) in partition
                                .range()
                                .zip(d.chunks_exact_mut(n))
                                .zip(p.chunks_exact_mut(n))
                            {
                                parent.execution().check()?;
                                compute(source, d, p, worker, parent)?;
                            }
                            Ok(())
                        },
                    )
                },
            )
        },
    )
}
// Undoes Johnson reweighting for one source.
fn restore_distances<S: RealScalar>(
    source: usize,
    d: &mut [Option<S>],
    h: &[S],
) -> Result<(), SciOxideError> {
    for (v, value) in d.iter_mut().enumerate() {
        *value = value
            .map(|value| finite_candidate(finite_candidate(value - h[source])? + h[v]))
            .transpose()?;
    }
    d[source] = Some(S::ZERO);
    Ok(())
}
// One Floyd-Warshall row relaxation.
fn floyd_row<S: RealScalar>(
    k: usize,
    (d, p): (&mut [Option<S>], &mut [Option<usize>]),
    (row, parent): (&[Option<S>], &[Option<usize>]),
) -> Result<(), SciOxideError> {
    if let Some(left) = d[k] {
        for (((value, predecessor), right), via) in d.iter_mut().zip(p).zip(row).zip(parent) {
            if let Some(right) = right {
                let candidate = finite_candidate(left + *right)?;
                if value.is_none_or(|old| candidate < old) {
                    *value = Some(candidate);
                    *predecessor = *via;
                }
            }
        }
    }
    Ok(())
}

/// Computes the shortest distances from `source` in the directed graph of a
/// real square sparse matrix, and the predecessors on shortest paths.
///
/// Every stored entry `A[u, v]`, a stored zero included, is an edge from `u` to
/// `v` of weight `A[u, v]`, and the length of a path is the sum of its weights.
/// This is [`GraphPlan::new`] with [`GraphDirection::Directed`] followed by
/// [`GraphPlan::shortest_paths`] with
/// [`GraphPathMethod::Dijkstra`] when no stored weight is negative, in
/// O((V + E) log V), and [`GraphPathMethod::BellmanFord`] otherwise, in
/// O(V E). Unreachable vertices have distance `None`.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
///
/// // Edges 0 → 1 (1), 0 → 2 (5), 1 → 2 (2).
/// let a = CsrMatrix::from_triplets(3, 3, &[0, 0, 1], &[1, 2, 2], &[1.0_f64, 5.0, 2.0])?;
/// let paths = scioxide_sparse::shortest_paths(a.view(), 0)?;
/// assert_eq!(
///     paths.as_view().distances(),
///     &[Some(0.0), Some(1.0), Some(3.0)]
/// );
/// assert_eq!(paths.as_view().path(0, 2)?, Some(vec![0, 1, 2]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DomainError` with argument `matrix` if a negative cycle is reachable
///   from `source`: its distances do not exist. [`GraphPlan::shortest_paths`]
///   returns the cycle as [`PathOutcome::NegativeCycle`] instead.
/// - `DimensionMismatch` if the matrix is not square.
/// - `OutOfBounds` if `source` is not a vertex.
/// - `NumericalInstability` if a distance overflows, or rounding makes a
///   detected negative cycle nonnegative.
/// - `SizeOverflow` or `AllocationFailure` if the plan or the result cannot
///   be allocated.
pub fn shortest_paths<S: RealScalar>(
    matrix: SparseView<'_, S>,
    source: usize,
) -> Result<GraphPathResult<S>, SciOxideError> {
    let plan = GraphPlan::new(matrix, GraphDirection::Directed)?;
    let method = if plan.edges.iter().any(|e| e.2 < S::ZERO) {
        GraphPathMethod::BellmanFord { source }
    } else {
        GraphPathMethod::Dijkstra { source }
    };
    match plan.shortest_paths(method)? {
        PathOutcome::Paths(paths) => Ok(paths),
        PathOutcome::NegativeCycle(_) => Err(SciOxideError::domain_error(
            scioxide_core::argument!("matrix"),
            scioxide_core::DomainReason::Undefined,
            "a negative cycle is reachable from the source",
        )),
    }
}

#[cfg(test)]
mod reconstruction_tests {
    use super::*;
    #[test]
    fn rejects_out_of_range_and_cyclic_predecessors() {
        let distances = [Some(0.0_f64), Some(1.0), Some(2.0)];
        for predecessors in [
            [None, Some(3), Some(1)],
            [None, Some(2), Some(1)],
            [None, None, Some(1)],
        ] {
            let view = GraphPathView {
                n: 3,
                source: Some(0),
                distances: &distances,
                predecessors: &predecessors,
            };
            assert!(view.path(0, 2).is_err());
        }
    }
}
