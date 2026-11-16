//! Maximum flow and minimum cut in networks with integer capacities.
use super::execution::{Attempt, grow, require};
use super::{GraphExecutionOptions, invalid};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_copy_slice, try_filled};
use scioxide_core::{ExecutionContext, SciOxideError, SolveScope};
/// Represents a directed edge with an integer capacity.
///
/// Parallel edges between the same two vertices are allowed and kept separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityEdge {
    /// Tail vertex.
    pub source: usize,
    /// Head vertex.
    pub target: usize,
    /// Capacity of the edge.
    pub capacity: u64,
}
/// Holds a flow network for the maximum-flow problem, solved with Dinic's
/// algorithm.
///
/// A flow from a source `s` to a sink `t` assigns each edge a value between 0 and
/// its capacity, such that inflow equals outflow at every vertex other than `s`
/// and `t`. Its value is the net flow out of `s`. By the max-flow min-cut theorem,
/// the largest value equals the smallest capacity of an `s`–`t` cut, the total
/// capacity of the edges from a vertex set containing `s` to its complement,
/// which contains `t`. The result includes such a cut. All arithmetic is exact
/// integer arithmetic. The plan stores O(V + E) values: the edges and, for
/// every vertex, its residual arcs. The function [`maximum_flow`] builds a plan
/// for one call.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CapacityEdge, MaxFlowPlan};
/// let arc = |source, target, capacity| CapacityEdge {
///     source,
///     target,
///     capacity,
/// };
/// let edges = [
///     arc(0, 1, 3),
///     arc(0, 2, 2),
///     arc(1, 2, 1),
///     arc(1, 3, 2),
///     arc(2, 3, 3),
/// ];
/// let plan = MaxFlowPlan::new(4, &edges)?;
/// let flow = plan.maximum_flow(0, 3)?;
/// let view = flow.as_view();
/// // The saturated source arcs (capacity 3 + 2) form a minimum cut.
/// assert_eq!(
///     (view.value, view.source_side),
///     (5, &[true, false, false, false][..])
/// );
/// assert!(view.verify(&plan)?);
/// // Edge endpoints must be vertices.
/// let error = MaxFlowPlan::new(3, &edges).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - E. A. Dinic, "Algorithm for solution of a problem of maximum flow in a
///   network with power estimation", *Soviet Math. Dokl.* 11 (1970), 1277–1280.
/// - L. R. Ford, Jr. and D. R. Fulkerson, "Maximal flow through a network",
///   *Canad. J. Math.* 8 (1956), 399–404.
#[derive(Debug, Clone)]
pub struct MaxFlowPlan {
    vertices: usize,
    edges: Vec<CapacityEdge>,
    ptr: Vec<usize>,
    adjacency: Vec<usize>,
}
/// Reports the storage a [`FlowWorkspace`] needs for one plan.
#[derive(Debug, Clone, Copy)]
pub struct FlowWorkspaceRequirements {
    vertices: usize,
    edges: usize,
    arcs: usize,
}
/// Stores the residual capacities, scratch and outputs of
/// [`MaxFlowPlan::maximum_flow_scoped`] for reuse across calls.
///
/// A call rejects a workspace that is too small instead of growing it.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{CapacityEdge, FlowWorkspace, GraphExecutionOptions, MaxFlowPlan};
/// let edges = [CapacityEdge {
///     source: 0,
///     target: 1,
///     capacity: 7,
/// }];
/// let plan = MaxFlowPlan::new(2, &edges)?;
/// let (options, c) = (GraphExecutionOptions::new(), ExecutionContext::sequential());
/// let mut workspace = FlowWorkspace::new();
/// let error = plan
///     .maximum_flow_scoped(0, 1, options, &mut workspace, c)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// workspace.try_reserve(&plan.workspace_requirements(())?)?;
/// let flow = plan.maximum_flow_scoped(0, 1, options, &mut workspace, c)?;
/// assert_eq!((flow.value, flow.edge_flows), (7, &[7][..]));
/// // Source and sink must differ.
/// let error = plan
///     .maximum_flow_scoped(0, 0, options, &mut workspace, c)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Default)]
pub struct FlowWorkspace {
    attempt: Attempt,
    active: (usize, usize),
    published: Option<(usize, usize, u64)>,
    residual: Vec<u64>,
    flows: Vec<u64>,
    level: Vec<usize>,
    cursor: Vec<usize>,
    queue: Vec<usize>,
    path: Vec<usize>,
    cut: Vec<bool>,
}
impl FlowWorkspace {
    /// Creates an empty workspace. Reserve storage with
    /// [`try_reserve`](Self::try_reserve) before use.
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
        requirements: &FlowWorkspaceRequirements,
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
    /// Storage never shrinks, so one workspace can serve several plans.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size of `requirements` overflows `usize`.
    /// - `AllocationFailure` if growth fails. The workspace may then be partly
    ///   grown, and stays usable.
    pub fn try_reserve(
        &mut self,
        requirements: &FlowWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        requirements.bytes()?;
        grow(&mut self.residual, requirements.arcs, 0)?;
        grow(&mut self.flows, requirements.edges, 0)?;
        grow(&mut self.level, requirements.vertices, 0)?;
        grow(&mut self.cursor, requirements.vertices, 0)?;
        grow(&mut self.queue, requirements.vertices, 0)?;
        grow(&mut self.path, requirements.vertices, 0)?;
        grow(&mut self.cut, requirements.vertices, false)?;
        Ok(())
    }
    fn require(&self, requirements: &FlowWorkspaceRequirements) -> Result<(), SciOxideError> {
        requirements.bytes()?;

        require([
            (self.residual.len(), requirements.arcs),
            (self.flows.len(), requirements.edges),
            (self.level.len(), requirements.vertices),
            (self.cursor.len(), requirements.vertices),
            (self.queue.len(), requirements.vertices),
            (self.path.len(), requirements.vertices),
            (self.cut.len(), requirements.vertices),
        ])?;
        Ok(())
    }
    /// Returns the record of the most recent call, including a call rejected
    /// before it started.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.attempt.record()
    }
    /// Returns the flow of the most recent call if it succeeded, and `None`
    /// otherwise.
    pub fn last_accepted(&self) -> Option<FlowView<'_>> {
        self.published
            .filter(|_| self.attempt.succeeded())
            .map(|(source, sink, value)| FlowView {
                source,
                sink,
                value,
                edge_flows: &self.flows[..self.active.1],
                source_side: &self.cut[..self.active.0],
            })
    }
}
/// Borrows a flow and a minimum cut.
///
/// The fields are public, so [`verify`](Self::verify) can also check a flow
/// computed elsewhere.
#[derive(Clone, Copy, Debug)]
pub struct FlowView<'a> {
    /// Source terminal.
    pub source: usize,
    /// Sink terminal.
    pub sink: usize,
    /// Flow value: the net flow out of `source`.
    pub value: u64,
    /// Flow on each edge, in the order of the input edges.
    pub edge_flows: &'a [u64],
    /// `true` for the vertices on the source side of the cut. For a computed flow
    /// these are the vertices reachable from `source` in the residual network.
    pub source_side: &'a [bool],
}
impl FlowView<'_> {
    /// Checks that this is a maximum flow of the network of `plan` with a
    /// minimum cut.
    ///
    /// Returns `true` if `source` and `sink` are distinct vertices, no edge flow
    /// exceeds its capacity, flow is conserved at every other vertex, the net flow
    /// out of `source` and into `sink` equals `value`, the source side contains
    /// `source` but not `sink`, and the cut capacity equals `value`. Equal flow
    /// value and cut capacity prove both optimal. Sums use 128-bit integers. A flow
    /// that fails a check gives `Ok(false)`, not an error.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a sum of flows or capacities overflows `u128`.
    /// - `AllocationFailure` if the per-vertex sums cannot be allocated.
    pub fn verify(&self, plan: &MaxFlowPlan) -> Result<bool, SciOxideError> {
        let n = plan.vertices;
        if self.source >= n
            || self.sink >= n
            || self.source == self.sink
            || self.edge_flows.len() != plan.edges.len()
            || self.source_side.len() != n
        {
            return Ok(false);
        }
        if !self.source_side[self.source] || self.source_side[self.sink] {
            return Ok(false);
        }
        let mut incoming = try_filled(n, 0u128, scioxide_core::quantity!("incoming_flow"))?;
        let mut outgoing = try_filled(n, 0u128, scioxide_core::quantity!("outgoing_flow"))?;
        let mut cut = 0u128;
        for (edge, &flow) in plan.edges.iter().zip(self.edge_flows) {
            if flow > edge.capacity {
                return Ok(false);
            }
            incoming[edge.target] = incoming[edge.target]
                .checked_add(u128::from(flow))
                .ok_or_else(overflow)?;
            outgoing[edge.source] = outgoing[edge.source]
                .checked_add(u128::from(flow))
                .ok_or_else(overflow)?;
            if self.source_side[edge.source] && !self.source_side[edge.target] {
                cut = cut
                    .checked_add(u128::from(edge.capacity))
                    .ok_or_else(overflow)?;
            }
        }
        for u in 0..n {
            let value = u128::from(self.value);
            let valid = if u == self.source {
                outgoing[u].checked_sub(incoming[u]) == Some(value)
            } else if u == self.sink {
                incoming[u].checked_sub(outgoing[u]) == Some(value)
            } else {
                incoming[u] == outgoing[u]
            };
            if !valid {
                return Ok(false);
            }
        }
        Ok(cut == u128::from(self.value))
    }
}
// Flow accumulation overflow.
fn overflow() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!("flow_capacity"))
}
/// Holds a maximum flow and a minimum cut computed by
/// [`MaxFlowPlan::maximum_flow`] or [`maximum_flow`].
#[derive(Debug)]
pub struct MaximumFlow {
    workspace: FlowWorkspace,
    source: usize,
    sink: usize,
    value: u64,
}
impl MaximumFlow {
    /// Borrows the flow and its cut.
    pub fn as_view(&self) -> FlowView<'_> {
        FlowView {
            source: self.source,
            sink: self.sink,
            value: self.value,
            edge_flows: &self.workspace.flows,
            source_side: &self.workspace.cut,
        }
    }
}
impl MaxFlowPlan {
    /// Builds the flow network on the vertices `0..vertices` with the given
    /// edges, and a residual arc in each direction for every edge.
    ///
    /// Zero capacities, self-loops and parallel edges are allowed. The arcs
    /// leaving each vertex keep the order of `edges`. Costs O(V + E) time and
    /// memory.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if an endpoint is not less than `vertices`.
    /// - `SizeOverflow` if twice the number of edges overflows `usize`.
    /// - `AllocationFailure` if the plan cannot be allocated.
    pub fn new(vertices: usize, edges: &[CapacityEdge]) -> Result<Self, SciOxideError> {
        for edge in edges {
            for vertex in [edge.source, edge.target] {
                if vertex >= vertices {
                    return Err(SciOxideError::out_of_bounds(
                        scioxide_core::quantity!("vertices"),
                        vertex,
                        vertices,
                    ));
                }
            }
        }
        let count = checked_mul(edges.len(), 2, scioxide_core::quantity!("arcs"))?;
        let mut adjacency = try_filled(count, 0usize, scioxide_core::quantity!("adjacency"))?;
        // Arc 2i leaves edge i's tail and arc 2i + 1 leaves its head; the stable
        // transpose keeps each vertex's arcs in input order.
        let arcs = edges
            .iter()
            .enumerate()
            .flat_map(|(i, e)| [(e.source, 2 * i), (e.target, 2 * i + 1)]);
        let ptr = crate::helpers::counting_transpose(vertices, arcs, |slot, arc| {
            adjacency[slot] = arc;
        })?;
        Ok(Self {
            vertices,
            edges: try_copy_slice(edges, scioxide_core::quantity!("capacity_edges"))?,
            ptr,
            adjacency,
        })
    }
    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices
    }
    /// Returns the edges, in input order.
    pub fn edges(&self) -> &[CapacityEdge] {
        &self.edges
    }
    /// Returns the storage that [`maximum_flow_scoped`](Self::maximum_flow_scoped)
    /// needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if twice the number of edges overflows `usize`.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<FlowWorkspaceRequirements, SciOxideError> {
        Ok(FlowWorkspaceRequirements {
            vertices: self.vertices,
            edges: self.edges.len(),
            arcs: checked_mul(
                self.edges.len(),
                2,
                scioxide_core::quantity!("residual_capacities"),
            )?,
        })
    }
    /// Creates a workspace sized for this plan.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(&self, request: ()) -> Result<FlowWorkspace, SciOxideError> {
        let mut workspace = FlowWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    fn target(&self, arc: usize) -> usize {
        let e = self.edges[arc / 2];
        if arc.is_multiple_of(2) {
            e.target
        } else {
            e.source
        }
    }
    /// Computes a maximum flow from `source` to `sink` and a minimum cut, and
    /// returns an owned result.
    ///
    /// This is [`maximum_flow_scoped`](Self::maximum_flow_scoped) with
    /// [`GraphExecutionOptions::new`]: the call allocates its workspace and
    /// runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The errors of [`maximum_flow_scoped`](Self::maximum_flow_scoped) other
    /// than `ConvergenceFailure` and `ExecutionStopped`, and `SizeOverflow` or
    /// `AllocationFailure` if the workspace cannot be created.
    pub fn maximum_flow(&self, source: usize, sink: usize) -> Result<MaximumFlow, SciOxideError> {
        let mut workspace = self.workspace(())?;
        let value = self
            .maximum_flow_scoped(
                source,
                sink,
                GraphExecutionOptions::new(),
                &mut workspace,
                ExecutionContext::sequential_native(),
            )?
            .value;
        Ok(MaximumFlow {
            workspace,
            source,
            sink,
            value,
        })
    }
    /// Computes a maximum flow from `source` to `sink` and a minimum cut into a
    /// workspace, in a root or nested solve scope.
    ///
    /// Each phase of Dinic's algorithm builds the level graph of the residual
    /// network by breadth-first search, then saturates it with augmenting paths
    /// found by depth-first search. The cost is O(V² E) for V vertices and E edges.
    /// Arcs leaving a vertex are tried in input order, so the result is
    /// deterministic. One iteration is counted per level search and per augmenting
    /// path search. The call runs sequentially under every execution policy.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `source` or `sink` is not a vertex.
    /// - `InvalidInput` if `source` equals `sink`.
    /// - `DimensionMismatch` if the workspace is smaller than
    ///   [`workspace_requirements`](Self::workspace_requirements) reports.
    /// - `SizeOverflow` if the flow value or a residual capacity exceeds
    ///   `u64::MAX`, or the workspace size overflows `usize`.
    /// - `ConvergenceFailure` if the iteration limit is exhausted.
    /// - `ExecutionStopped` on cancellation or a passed deadline;
    ///   `CapabilityUnavailable`, `CapabilityFailure`, `ExecutionFailure` or
    ///   `AccountingInvariant` if the scope cannot run the call.
    pub fn maximum_flow_scoped<'a, 'scope, 'capability>(
        &self,
        source: usize,
        sink: usize,
        options: GraphExecutionOptions,
        workspace: &'a mut FlowWorkspace,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<FlowView<'a>, SciOxideError> {
        workspace.attempt.reject();
        workspace.published = None;
        for terminal in [source, sink] {
            if terminal >= self.vertices {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("vertices"),
                    terminal,
                    self.vertices,
                ));
            }
        }
        if source == sink {
            return Err(invalid(
                scioxide_core::argument!("sink"),
                "flow terminals must differ",
            ));
        }
        workspace.require(&self.workspace_requirements(())?)?;
        let mut attempt = core::mem::take(&mut workspace.attempt);
        let result = attempt.run(
            scope,
            options,
            scioxide_core::algorithm_id!("sparse", "flow", 1),
            |context| self.run((source, sink), workspace, &mut || super::admit(context)),
        );
        workspace.attempt = attempt;
        let value = result?;
        workspace.active = (self.vertices, self.edges.len());
        workspace.published = Some((source, sink, value));
        Ok(FlowView {
            source,
            sink,
            value,
            edge_flows: &workspace.flows[..self.edges.len()],
            source_side: &workspace.cut[..self.vertices],
        })
    }
    // Dinic max-flow with a min-cut witness.
    fn run(
        &self,
        (source, sink): (usize, usize),
        w: &mut FlowWorkspace,
        tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
    ) -> Result<u64, SciOxideError> {
        let residual = &mut w.residual[..2 * self.edges.len()];
        let flows = &mut w.flows[..self.edges.len()];
        let level = &mut w.level[..self.vertices];
        let cursor = &mut w.cursor[..self.vertices];
        let queue = &mut w.queue[..self.vertices];
        let path = &mut w.path[..self.vertices];
        let cut = &mut w.cut[..self.vertices];
        for (i, e) in self.edges.iter().enumerate() {
            residual[2 * i] = e.capacity;
            residual[2 * i + 1] = 0;
        }
        let mut value = 0u64;
        loop {
            tick()?;
            level.fill(usize::MAX);
            level[source] = 0;
            queue[0] = source;
            let mut head = 0;
            let mut tail = 1;
            while head < tail {
                let u = queue[head];
                head += 1;
                for &arc in &self.adjacency[self.ptr[u]..self.ptr[u + 1]] {
                    let v = self.target(arc);
                    if residual[arc] > 0 && level[v] == usize::MAX {
                        level[v] = level[u] + 1;
                        queue[tail] = v;
                        tail += 1;
                    }
                }
            }
            if level[sink] == usize::MAX {
                break;
            }
            cursor.copy_from_slice(&self.ptr[..self.vertices]);
            loop {
                tick()?;
                let mut u = source;
                let mut len = 0;
                loop {
                    if u == sink {
                        break;
                    }
                    while cursor[u] < self.ptr[u + 1] {
                        let arc = self.adjacency[cursor[u]];
                        let v = self.target(arc);
                        if residual[arc] > 0 && level[v] == level[u] + 1 {
                            break;
                        }
                        cursor[u] += 1;
                    }
                    if cursor[u] == self.ptr[u + 1] {
                        if len == 0 {
                            break;
                        }
                        len -= 1;
                        u = self.target(path[len] ^ 1);
                        cursor[u] += 1;
                    } else {
                        let arc = self.adjacency[cursor[u]];
                        path[len] = arc;
                        len += 1;
                        u = self.target(arc);
                    }
                }
                if u != sink {
                    break;
                }
                let amount = path[..len]
                    .iter()
                    .map(|&arc| residual[arc])
                    .min()
                    .ok_or_else(|| {
                        invalid(scioxide_core::argument!("self"), "empty augmenting path")
                    })?;
                value = value.checked_add(amount).ok_or_else(overflow)?;
                for &arc in &path[..len] {
                    residual[arc] -= amount;
                    residual[arc ^ 1] =
                        residual[arc ^ 1].checked_add(amount).ok_or_else(overflow)?;
                }
            }
        }
        for (i, e) in self.edges.iter().enumerate() {
            flows[i] = e.capacity - residual[2 * i];
        }
        for (side, &level) in cut.iter_mut().zip(level.iter()) {
            *side = level != usize::MAX;
        }
        Ok(value)
    }
}

impl FlowWorkspaceRequirements {
    /// Returns the number of bytes of workspace storage, not counting fixed-size
    /// fields, vector headers and allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn bytes(self) -> Result<usize, SciOxideError> {
        super::workspace_bytes(
            (
                self.vertices,
                4 * core::mem::size_of::<usize>() + core::mem::size_of::<bool>(),
            ),
            (self.edges, 3 * core::mem::size_of::<u64>()),
        )
    }
}

/// Computes a maximum flow from `source` to `sink` in the network on the
/// vertices `0..vertices` with the given integer-capacity edges, and a minimum
/// cut.
///
/// This is [`MaxFlowPlan::new`] followed by [`MaxFlowPlan::maximum_flow`]:
/// Dinic's algorithm in O(V² E) with exact integer arithmetic. Keep the plan to
/// compute flows between other terminals of the same network.
///
/// ```
/// use scioxide_sparse::CapacityEdge;
/// let arc = |source, target, capacity| CapacityEdge {
///     source,
///     target,
///     capacity,
/// };
/// let edges = [
///     arc(0, 1, 3),
///     arc(0, 2, 2),
///     arc(1, 2, 1),
///     arc(1, 3, 2),
///     arc(2, 3, 3),
/// ];
/// let flow = scioxide_sparse::maximum_flow(4, &edges, 0, 3)?;
/// assert_eq!(flow.as_view().value, 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `OutOfBounds` if an edge endpoint, `source` or `sink` is not a vertex.
/// - `InvalidInput` if `source` equals `sink`.
/// - `SizeOverflow` if the flow value or a residual capacity exceeds
///   `u64::MAX`, or a size overflows `usize`.
/// - `AllocationFailure` if the plan or workspace cannot be allocated.
pub fn maximum_flow(
    vertices: usize,
    edges: &[CapacityEdge],
    source: usize,
    sink: usize,
) -> Result<MaximumFlow, SciOxideError> {
    MaxFlowPlan::new(vertices, edges)?.maximum_flow(source, sink)
}
