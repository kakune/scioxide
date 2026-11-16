//! Breadth-first and depth-first traversal, connected components and minimum spanning forests.
use super::*;
/// Holds the result of a breadth-first or depth-first traversal from one source
/// vertex.
///
/// Vertices are listed in the order they are discovered. Neighbors are visited
/// in increasing index order, so the order is deterministic. Vertices that are
/// not reachable from the source have neither a predecessor nor a depth.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPlan};
/// // Directed diamond 0 -> {1, 2} -> 3.
/// let a = CsrMatrix::from_raw(
///     4,
///     4,
///     vec![0, 2, 3, 4, 4],
///     vec![1, 2, 3, 3],
///     vec![1.0_f64; 4],
/// )?;
/// let graph = GraphPlan::from_pattern(a.structural_pattern(), GraphDirection::Directed)?;
/// let bfs = graph.breadth_first(0)?;
/// assert_eq!(bfs.order(), &[0, 1, 2, 3]);
/// assert_eq!(bfs.depths(), &[Some(0), Some(1), Some(1), Some(2)]);
/// assert_eq!(bfs.predecessors(), &[None, Some(0), Some(0), Some(1)]);
/// assert_eq!(graph.depth_first(0)?.order(), &[0, 1, 3, 2]);
/// let error = graph.breadth_first(4).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct GraphTraversal {
    order: Vec<usize>,
    predecessors: Vec<Option<usize>>,
    depths: Vec<Option<usize>>,
}
impl GraphTraversal {
    /// Returns the reachable vertices in discovery order, starting with the source.
    pub fn order(&self) -> &[usize] {
        &self.order
    }
    /// Returns the parent of each vertex in the search tree: `None` for the source
    /// and for unreachable vertices.
    pub fn predecessors(&self) -> &[Option<usize>] {
        &self.predecessors
    }
    /// Returns the depth of each vertex in the search tree: `None` for unreachable
    /// vertices. Breadth-first depths are the minimum numbers of edges from the
    /// source; depth-first depths are not.
    pub fn depths(&self) -> &[Option<usize>] {
        &self.depths
    }
}
/// Holds connected-component labels, one per vertex.
///
/// Labels run from 0 to `count() − 1` and are numbered in increasing order of
/// each component's smallest vertex.
///
/// ```
/// use scioxide_sparse::{CsrMatrix, GraphDirection, GraphPlan};
/// // 0 <-> 1 -> 2, and vertex 3 is isolated.
/// let a = CsrMatrix::from_raw(4, 4, vec![0, 1, 3, 3, 3], vec![1, 0, 2], vec![1.0_f64; 3])?;
/// let graph = GraphPlan::from_pattern(a.structural_pattern(), GraphDirection::Directed)?;
/// let weak = graph.weak_components()?;
/// assert_eq!((weak.labels(), weak.count()), (&[0, 0, 0, 1][..], 2));
/// let strong = graph.strong_components()?;
/// assert_eq!((strong.labels(), strong.count()), (&[0, 0, 1, 2][..], 3));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct GraphComponents {
    labels: Vec<usize>,
    count: usize,
}
impl GraphComponents {
    /// Returns the component label of each vertex.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    /// Returns the number of components; zero for an empty graph.
    pub fn count(&self) -> usize {
        self.count
    }
}
/// Represents one edge of a spanning forest, with endpoints `u < v`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForestEdge<S: RealScalar> {
    /// Smaller endpoint.
    pub u: usize,
    /// Larger endpoint.
    pub v: usize,
    /// Edge weight in the graph.
    pub weight: S,
}
/// Holds a minimum spanning forest: a minimum spanning tree of every connected
/// component.
///
/// An isolated vertex is a component of its own and contributes no edge.
/// Computed by [`GraphPlan::minimum_spanning_forest`].
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, ForestEdge, GraphDirection, GraphPlan};
/// // Triangle 0-1 (1), 0-2 (3), 1-2 (2) plus isolated vertex 3.
/// let weights = vec![1.0_f64, 3.0, 2.0];
/// let a = CsrMatrix::from_raw(4, 4, vec![0, 2, 3, 3, 3], vec![1, 2, 2], weights)?;
/// let graph = GraphPlan::new(a.view(), GraphDirection::Undirected)?;
/// let forest = graph.minimum_spanning_forest()?;
/// let first = ForestEdge {
///     u: 0,
///     v: 1,
///     weight: 1.0,
/// };
/// let second = ForestEdge {
///     u: 1,
///     v: 2,
///     weight: 2.0,
/// };
/// assert_eq!(forest.edges(), &[first, second]);
/// assert_eq!(
///     (forest.total_weight(), forest.components().count()),
///     (3.0, 2)
/// );
/// // Spanning forests are defined only for undirected graphs.
/// let directed = GraphPlan::new(a.view(), GraphDirection::Directed)?;
/// let error = directed.minimum_spanning_forest().unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct MinimumSpanningForest<S: RealScalar> {
    edges: Vec<ForestEdge<S>>,
    total: S,
    components: GraphComponents,
}
impl<S: RealScalar> MinimumSpanningForest<S> {
    /// Returns the forest edges in the order Kruskal's algorithm selects them: by
    /// weight, then by `(u, v)`.
    pub fn edges(&self) -> &[ForestEdge<S>] {
        &self.edges
    }
    /// Returns the exact sum of the selected edge weights, rounded once.
    pub fn total_weight(&self) -> S {
        self.total
    }
    /// Returns the connected components of the graph, including isolated
    /// vertices.
    pub fn components(&self) -> &GraphComponents {
        &self.components
    }
}
impl<S: RealScalar> GraphPlan<S> {
    /// Computes a breadth-first traversal from `source`.
    ///
    /// Costs O(V + E) time and O(V) memory for the result.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `source` is not a vertex.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # References
    ///
    /// T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to
    /// Algorithms*, 3rd ed., MIT Press, 2009, §22.2.
    pub fn breadth_first(&self, source: usize) -> Result<GraphTraversal, SciOxideError> {
        self.traverse(source, false)
    }
    /// Computes a depth-first traversal from `source`.
    ///
    /// The search keeps an explicit stack, so deep graphs cannot overflow the call
    /// stack. Costs O(V + E) time and O(V) memory.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `source` is not a vertex.
    /// - `AllocationFailure` if the result or the stack cannot be allocated.
    ///
    /// # References
    ///
    /// T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to
    /// Algorithms*, 3rd ed., MIT Press, 2009, §22.3.
    pub fn depth_first(&self, source: usize) -> Result<GraphTraversal, SciOxideError> {
        self.traverse(source, true)
    }
    // Shared BFS/DFS.
    fn traverse(&self, source: usize, dfs: bool) -> Result<GraphTraversal, SciOxideError> {
        self.vertex(source)?;
        let n = self.vertex_count();
        let mut order = try_with_capacity(n, scioxide_core::quantity!("traversal_order"))?;
        let mut predecessors = try_filled(n, None, scioxide_core::quantity!("predecessors"))?;
        let mut depths = try_filled(n, None, scioxide_core::quantity!("depths"))?;
        let mut stack = try_with_capacity(n, scioxide_core::quantity!("traversal_stack"))?;
        order.push(source);
        depths[source] = Some(0usize);
        if dfs {
            stack.push((source, self.ptr[source]));
            while let Some((u, edge)) = stack.last_mut() {
                if *edge == self.ptr[*u + 1] {
                    stack.pop();
                    continue;
                }
                let v = self.edges[*edge].1;
                *edge += 1;
                if depths[v].is_none() {
                    depths[v] = depths[*u].map(|d| d + 1);
                    predecessors[v] = Some(*u);
                    order.push(v);
                    stack.push((v, self.ptr[v]));
                }
            }
        } else {
            let mut head = 0;
            while head < order.len() {
                let u = order[head];
                head += 1;
                for &(_, v, _) in self.row(u) {
                    if depths[v].is_none() {
                        depths[v] = depths[u].map(|d| d + 1);
                        predecessors[v] = Some(u);
                        order.push(v);
                    }
                }
            }
        }
        Ok(GraphTraversal {
            order,
            predecessors,
            depths,
        })
    }
    /// Computes the weakly connected components: the components of the graph with
    /// edge directions ignored.
    ///
    /// For an undirected graph these are the connected components. Costs O(V + E)
    /// time and O(V) memory.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the labels or scratch cannot be allocated.
    pub fn weak_components(&self) -> Result<GraphComponents, SciOxideError> {
        self.components(false)
    }
    /// Computes the strongly connected components with Kosaraju's algorithm.
    ///
    /// Two vertices belong to the same component when each is reachable from the
    /// other. The algorithm orders vertices by a depth-first search and then
    /// searches the reverse graph in that order. Costs O(V + E) time and O(V)
    /// memory.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the labels or scratch cannot be allocated.
    ///
    /// # References
    ///
    /// M. Sharir, "A strong-connectivity algorithm and its applications in data
    /// flow analysis", *Comput. Math. Appl.* 7 (1981), 67–72.
    pub fn strong_components(&self) -> Result<GraphComponents, SciOxideError> {
        self.components(true)
    }
    // Shared weak/strong connectivity.
    fn components(&self, strong: bool) -> Result<GraphComponents, SciOxideError> {
        let n = self.vertex_count();
        let mut seen = try_filled(n, false, scioxide_core::quantity!("visited"))?;
        let mut order = try_with_capacity(n, scioxide_core::quantity!("finish_order"))?;
        let mut frames = try_with_capacity(n, scioxide_core::quantity!("dfs_frames"))?;
        if strong {
            for root in 0..n {
                if seen[root] {
                    continue;
                }
                seen[root] = true;
                frames.push((root, self.ptr[root]));
                while let Some((u, edge)) = frames.last_mut() {
                    if *edge == self.ptr[*u + 1] {
                        order.push(*u);
                        frames.pop();
                        continue;
                    }
                    let v = self.edges[*edge].1;
                    *edge += 1;
                    if !seen[v] {
                        seen[v] = true;
                        frames.push((v, self.ptr[v]));
                    }
                }
            }
            order.reverse();
        } else {
            order.extend(0..n);
        }
        let mut labels = try_filled(n, usize::MAX, scioxide_core::quantity!("component_labels"))?;
        let mut stack = try_with_capacity(n, scioxide_core::quantity!("component_stack"))?;
        let mut count = 0;
        for root in order {
            if labels[root] != usize::MAX {
                continue;
            }
            labels[root] = count;
            stack.push(root);
            while let Some(u) = stack.pop() {
                for v in self
                    .incoming(u)
                    .iter()
                    .copied()
                    .chain(self.row(u).iter().filter(|_| !strong).map(|e| e.1))
                {
                    if labels[v] == usize::MAX {
                        labels[v] = count;
                        stack.push(v);
                    }
                }
            }
            count += 1;
        }
        canonical_labels(&mut labels, count)?;
        Ok(GraphComponents { labels, count })
    }
    /// Computes a minimum spanning forest of an undirected graph with Kruskal's
    /// algorithm.
    ///
    /// Edges are taken in increasing order of weight, with ties broken by
    /// `(u, v)`, and an edge joins the forest unless it closes a cycle. Cycles are
    /// detected with a union–find structure. Negative weights are allowed. Costs
    /// O(E log E) time and O(V + E) memory.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the graph is directed.
    /// - `NumericalInstability` (`Overflow`) if the total weight overflows.
    /// - `AllocationFailure` if the result or scratch cannot be allocated.
    ///
    /// # References
    ///
    /// J. B. Kruskal, "On the shortest spanning subtree of a graph and the
    /// traveling salesman problem", *Proc. Amer. Math. Soc.* 7 (1956), 48–50.
    pub fn minimum_spanning_forest(&self) -> Result<MinimumSpanningForest<S>, SciOxideError> {
        if self.direction != GraphDirection::Undirected {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "forest requires undirected graph",
            ));
        }
        let n = self.vertex_count();
        let mut candidates = try_with_capacity(
            self.edges.len() / 2,
            scioxide_core::quantity!("forest_candidates"),
        )?;
        candidates.extend(
            self.edges
                .iter()
                .filter(|e| e.0 < e.1)
                .map(|&(u, v, weight)| ForestEdge { u, v, weight }),
        );
        candidates.sort_unstable_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.u, a.v).cmp(&(b.u, b.v)))
        });
        let mut parents = try_with_capacity(n, scioxide_core::quantity!("union_find_parents"))?;
        parents.extend(0..n);
        let mut sizes = try_filled(n, 1usize, scioxide_core::quantity!("union_find_sizes"))?;
        let mut edges = try_with_capacity(
            n.saturating_sub(1),
            scioxide_core::quantity!("forest_edges"),
        )?;
        let mut total = scioxide_core::RangeAccumulator::new();
        for edge in candidates {
            let mut a = root(&mut parents, edge.u);
            let mut b = root(&mut parents, edge.v);
            if a == b {
                continue;
            }
            if sizes[a] < sizes[b] {
                core::mem::swap(&mut a, &mut b);
            }
            parents[b] = a;
            sizes[a] += sizes[b];
            total.add(edge.weight)?;
            edges.push(edge);
        }
        let mut map = try_filled(n, usize::MAX, scioxide_core::quantity!("component_map"))?;
        let mut labels = try_filled(n, 0usize, scioxide_core::quantity!("component_labels"))?;
        let mut count = 0;
        for (u, label) in labels.iter_mut().enumerate() {
            let r = root(&mut parents, u);
            if map[r] == usize::MAX {
                map[r] = count;
                count += 1;
            }
            *label = map[r];
        }
        Ok(MinimumSpanningForest {
            edges,
            total: total.finish()?,
            components: GraphComponents { labels, count },
        })
    }
}
fn root(parents: &mut [usize], mut u: usize) -> usize {
    while parents[u] != u {
        parents[u] = parents[parents[u]];
        u = parents[u];
    }
    u
}
// Renumbers labels by first appearance.
fn canonical_labels(labels: &mut [usize], count: usize) -> Result<(), SciOxideError> {
    let mut map = try_filled(count, usize::MAX, scioxide_core::quantity!("label_map"))?;
    let mut next = 0;
    for label in labels {
        if map[*label] == usize::MAX {
            map[*label] = next;
            next += 1;
        }
        *label = map[*label];
    }
    Ok(())
}
/// Computes the connected components of the graph of a square sparsity
/// pattern, with edge directions ignored.
///
/// Every stored entry `(u, v)`, a stored zero included, joins vertices `u` and
/// `v`. This is [`GraphPlan::from_pattern`] with
/// [`GraphDirection::Directed`] followed by [`GraphPlan::weak_components`]:
/// for a directed pattern these are its weakly connected components. Labels
/// are numbered in increasing order of each component's smallest vertex.
/// Costs O(V + E log E) time for V vertices and E stored entries.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
/// // 0 -> 1 and 2 -> 3: two components.
/// let a = CsrMatrix::from_triplets(4, 4, &[0, 2], &[1, 3], &[1.0_f64, 1.0])?;
/// let components = scioxide_sparse::connected_components(a.structural_pattern())?;
/// assert_eq!(
///     (components.labels(), components.count()),
///     (&[0, 0, 1, 1][..], 2)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if the pattern is not square.
/// - `SizeOverflow` or `AllocationFailure` if the adjacency lists or labels
///   cannot be allocated.
pub fn connected_components(
    pattern: StructuralPatternView<'_>,
) -> Result<GraphComponents, SciOxideError> {
    GraphPlan::from_pattern(pattern, GraphDirection::Directed)?.weak_components()
}
