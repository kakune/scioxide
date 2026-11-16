//! Graphs defined by sparsity patterns, and the algorithms on them.
//!
//! Graph weights and costs are finite on entry, so a non-finite distance, degree,
//! potential or weight sum is overflow from finite operands and is reported as
//! `NumericalInstability` with `NumericalIssue::Overflow`.
use crate::SparseView;
use crate::helpers::admit;
use crate::helpers::invalid;
use alloc::vec::Vec;
use scioxide_core::backend::standard::finite_candidate;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{Argument, Quantity, RealScalar, SciOxideError, StructuralPatternView};
mod execution;
pub(crate) mod heap;
pub use execution::GraphExecutionOptions;
mod paths;
mod traversal;
pub use paths::{
    GraphPathMethod, GraphPathResult, GraphPathView, GraphWorkspace, GraphWorkspaceRequest,
    GraphWorkspaceRequirements, NegativeCycle, PathOutcome, shortest_paths,
};
pub use traversal::{
    ForestEdge, GraphComponents, GraphTraversal, MinimumSpanningForest, connected_components,
};

/// Selects how the stored entries of a square matrix define graph edges.
///
/// Vertex `i` corresponds to row and column `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphDirection {
    /// A stored entry `(u, v)` is an edge from `u` to `v`.
    Directed,
    /// A stored entry `(u, v)` is an edge between `u` and `v`. If both `(u, v)`
    /// and `(v, u)` are stored, the edge takes the smaller of the two weights.
    Undirected,
}
/// Holds a graph in adjacency-list form for the graph algorithms.
///
/// [`new`](Self::new) reads a real square sparse matrix as a weighted graph:
/// every stored entry `A[u, v]`, a stored zero included, is an edge of weight
/// `A[u, v]`. [`from_pattern`](GraphPlan::from_pattern) reads a square
/// sparsity pattern as a graph whose edges all have weight 1. Vertex `i` is row
/// and column `i`. Sparse storage holds only finite values, so every weight is
/// finite; negative weights are allowed, and the algorithms that need
/// nonnegative weights say so.
///
/// Adjacency lists are sorted by vertex index and every algorithm visits
/// neighbors in increasing index order, so results are deterministic.
/// Preparation costs O(V + E log E) time and O(V + E) memory, where V is the
/// number of vertices and E the number of stored entries. The plan provides
/// traversal ([`breadth_first`](Self::breadth_first),
/// [`depth_first`](Self::depth_first)), connectivity
/// ([`weak_components`](Self::weak_components),
/// [`strong_components`](Self::strong_components),
/// [`minimum_spanning_forest`](Self::minimum_spanning_forest)), shortest paths
/// ([`shortest_paths`](Self::shortest_paths)) and the graph Laplacian
/// ([`laplacian`](Self::laplacian)). The functions
/// [`connected_components`](crate::connected_components),
/// [`shortest_paths`](crate::shortest_paths) and [`laplacian`](crate::laplacian)
/// build a plan for one call.
///
/// Weights are finite, so a distance, degree or weight sum that becomes infinite
/// is an overflow. It is reported as `NumericalInstability` with
/// [`NumericalIssue::Overflow`](scioxide_core::NumericalIssue::Overflow).
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPathMethod, GraphPlan, PathOutcome};
/// // Opposing entries (0,1) = 4 and (1,0) = 2 form one undirected edge of weight 2.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![4.0_f64, 2.0])?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Undirected)?;
/// assert_eq!((graph.vertex_count(), graph.edge_count()), (2, 2));
/// let outcome = graph.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })?;
/// let PathOutcome::Paths(paths) = outcome else {
///     unreachable!()
/// };
/// assert_eq!(paths.as_view().distances(), &[Some(0.0), Some(2.0)]);
/// // A rectangular matrix is not a graph.
/// let wide = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![1], vec![1.0_f64])?;
/// let error = GraphPlan::new(wide.view(), GraphDirection::Directed).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GraphPlan<S: RealScalar> {
    direction: GraphDirection,
    ptr: Vec<usize>,
    edges: Vec<(usize, usize, S)>,
    reverse_ptr: Vec<usize>,
    reverse: Vec<usize>,
}
impl GraphPlan<f64> {
    /// Builds a plan from a square sparsity pattern in which every stored entry,
    /// a stored zero included, is an edge of weight 1.
    ///
    /// The plan has weights of type `f64`, whatever the scalar type of the
    /// matrix the pattern belongs to. Costs O(V + E log E) time for V vertices
    /// and E stored entries.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPlan};
    /// // Stored entries (0,1) and (1,2); the stored zero is still an edge.
    /// let a = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![1, 2], vec![1.0_f64, 0.0])?;
    /// let graph = GraphPlan::from_pattern(a.structural_pattern(), GraphDirection::Directed)?;
    /// assert_eq!((graph.vertex_count(), graph.edge_count()), (3, 2));
    /// assert_eq!(graph.breadth_first(0)?.order(), &[0, 1, 2]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the pattern is not square.
    /// - `SizeOverflow` if the edge count overflows `usize` (an undirected graph
    ///   stores each edge in both directions).
    /// - `AllocationFailure` if the adjacency lists cannot be allocated.
    pub fn from_pattern(
        pattern: StructuralPatternView<'_>,
        direction: GraphDirection,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::exact(pattern.cols(), pattern.rows())?;
        let mut edges = try_with_capacity(
            pattern.col_indices().len(),
            scioxide_core::quantity!("graph_edges"),
        )?;
        for (u, bounds) in pattern.row_ptr().windows(2).enumerate() {
            edges.extend(
                pattern.col_indices()[bounds[0]..bounds[1]]
                    .iter()
                    .map(|&v| (u, v, 1.0)),
            );
        }
        Self::prepare(pattern.rows(), direction, edges)
    }
}
impl<S: RealScalar> GraphPlan<S> {
    /// Builds a plan from a real square sparse matrix whose stored entries are
    /// the weighted edges, with forward and reverse adjacency lists.
    ///
    /// Costs O(V + E log E) time for V vertices and E stored entries. See the
    /// type for an example.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the matrix is not square.
    /// - `SizeOverflow` if the edge count overflows `usize` (an undirected graph
    ///   stores each edge in both directions).
    /// - `AllocationFailure` if the adjacency lists cannot be allocated.
    pub fn new(
        matrix: SparseView<'_, S>,
        direction: GraphDirection,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::exact(matrix.cols(), matrix.rows())?;
        let mut edges = try_with_capacity(
            matrix.values().len(),
            scioxide_core::quantity!("graph_edges"),
        )?;
        edges.extend(matrix.iter());
        Self::prepare(matrix.rows(), direction, edges)
    }
    // Canonical adjacency preparation.
    fn prepare(
        n: usize,
        direction: GraphDirection,
        mut edges: Vec<(usize, usize, S)>,
    ) -> Result<Self, SciOxideError> {
        if direction == GraphDirection::Undirected {
            let count = edges.len();
            checked_mul(count, 2, scioxide_core::quantity!("graph_edges"))?;
            edges.try_reserve(count).map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("graph_edges"), count)
            })?;
            for i in 0..count {
                let (u, v, w) = edges[i];
                if u != v {
                    edges.push((v, u, w));
                }
            }
        }
        edges.sort_unstable_by(|a, b| {
            (a.0, a.1)
                .cmp(&(b.0, b.1))
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(core::cmp::Ordering::Equal))
        });
        edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        let ptr = crate::helpers::counting_transpose(
            n,
            edges.iter().map(|&(u, _, _)| (u, ())),
            |_, ()| {},
        )?;
        let mut reverse = try_filled(
            edges.len(),
            0usize,
            scioxide_core::quantity!("reverse_adjacency"),
        )?;
        let reverse_ptr = crate::helpers::counting_transpose(
            n,
            edges.iter().map(|&(u, v, _)| (v, u)),
            |slot, u| reverse[slot] = u,
        )?;
        Ok(Self {
            direction,
            ptr,
            edges,
            reverse_ptr,
            reverse,
        })
    }
    /// Returns the number of vertices, including isolated vertices.
    pub fn vertex_count(&self) -> usize {
        self.ptr.len() - 1
    }
    /// Returns the number of directed edges. In an undirected graph, an edge
    /// between distinct vertices counts twice and a self-loop once.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    /// Returns how the stored entries were read as edges.
    pub fn direction(&self) -> GraphDirection {
        self.direction
    }
    fn row(&self, u: usize) -> &[(usize, usize, S)] {
        &self.edges[self.ptr[u]..self.ptr[u + 1]]
    }
    fn incoming(&self, u: usize) -> &[usize] {
        &self.reverse[self.reverse_ptr[u]..self.reverse_ptr[u + 1]]
    }
    // Checks a vertex index.
    fn vertex(&self, u: usize) -> Result<(), SciOxideError> {
        if u < self.vertex_count() {
            Ok(())
        } else {
            Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("graph_vertices"),
                u,
                self.vertex_count(),
            ))
        }
    }
    fn weight(&self, u: usize, v: usize) -> Option<S> {
        let row = self.row(u);
        row.binary_search_by_key(&v, |e| e.1).ok().map(|i| row[i].2)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
    fn fixture(direction: GraphDirection) -> GraphPlan<f64> {
        let mut b = SparseBuilder::new(5, 5);
        for (u, v, w) in [(0, 1, 0.0), (1, 2, 2.0), (0, 2, 5.0), (3, 4, -1.0)] {
            b.push(u, v, w).unwrap();
        }
        let a = b
            .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        GraphPlan::new(a.view(), direction).unwrap()
    }
    #[test]
    fn traversal_and_components() {
        let g = fixture(GraphDirection::Directed);
        assert_eq!(g.breadth_first(0).unwrap().order(), &[0, 1, 2]);
        assert_eq!(g.depth_first(0).unwrap().order(), &[0, 1, 2]);
        assert_eq!(g.weak_components().unwrap().labels(), &[0, 0, 0, 1, 1]);
        assert_eq!(g.strong_components().unwrap().count(), 5);
    }
    #[test]
    fn paths_and_negative_cycle() {
        let g = fixture(GraphDirection::Directed);
        for method in [
            GraphPathMethod::BellmanFord { source: 0 },
            GraphPathMethod::FloydWarshall,
            GraphPathMethod::Johnson,
        ] {
            let PathOutcome::Paths(p) = g.shortest_paths(method).unwrap() else {
                panic!("unexpected cycle")
            };
            assert_eq!(p.as_view().distances()[2], Some(2.0));
            assert_eq!(p.as_view().path(0, 2).unwrap(), Some(alloc::vec![0, 1, 2]));
            assert_eq!(p.as_view().path(0, 4).unwrap(), None);
        }
        let g = fixture(GraphDirection::Undirected);
        assert!(matches!(
            g.shortest_paths(GraphPathMethod::Johnson).unwrap(),
            PathOutcome::NegativeCycle(_)
        ));
        assert!(matches!(
            g.shortest_paths(GraphPathMethod::BellmanFord { source: 0 })
                .unwrap(),
            PathOutcome::Paths(_)
        ));
        let f = g.minimum_spanning_forest().unwrap();
        assert_eq!(f.edges().len(), 3);
        assert_eq!(f.total_weight(), 1.0);
    }
}

mod matching;
pub use matching::{
    BipartiteMatching, MatchingView, MatchingWorkspace, MatchingWorkspaceRequirements,
    maximum_bipartite_matching, maximum_bipartite_matching_scoped,
    maximum_bipartite_matching_with_options,
};

mod flow;
pub use flow::{
    CapacityEdge, FlowView, FlowWorkspace, FlowWorkspaceRequirements, MaxFlowPlan, MaximumFlow,
    maximum_flow,
};

mod assignment;
pub use assignment::{
    AssignmentObjective, AssignmentOutcome, AssignmentPlan, AssignmentResiduals, AssignmentView,
    AssignmentWorkspace, AssignmentWorkspaceRequirements, FullAssignment, minimum_cost_assignment,
};

mod laplacian;
pub use laplacian::{LaplacianDegree, LaplacianNormalization, LaplacianPlan, laplacian};

// Workspace payload `n*a + m*b` in bytes.
fn workspace_bytes((n, a): (usize, usize), (m, b): (usize, usize)) -> Result<usize, SciOxideError> {
    const BYTES: Quantity = scioxide_core::quantity!("workspace_bytes");
    checked_add(checked_mul(n, a, BYTES)?, checked_mul(m, b, BYTES)?, BYTES)
}
