//! Fill-reducing and bandwidth-reducing orderings and graph colorings of sparsity patterns.
use crate::helpers::invalid;
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    ColoringKind, ColoringSchedule, Permutation, SciOxideError, StructuralPatternView,
};

mod amd;
mod compressed;

use compressed::Graph;
use scioxide_core::Triangle;

/// Selects the ordering that a sparse direct solver applies during symbolic
/// analysis, before any numerical pivoting.
///
/// A permutation `p` is stored in gather form, `p[new] = old`: entry `new` of the
/// permuted matrix is entry `p[new]` of the original. `FillReducing`, the
/// default, applies approximate minimum degree to the graph that suits each
/// solver: `A + Aᵀ` for [`SparseLu`](crate::SparseLu), the symmetric pattern
/// of `A` for [`SparseCholesky`](crate::SparseCholesky) and
/// [`SparseLdl`](crate::SparseLdl), and `AᵀA` for [`SparseQr`](crate::SparseQr). Pivoting in LU and `LDLᴴ` can
/// change the order further.
///
/// # Examples
///
/// ```
/// use scioxide_core::StructuralPattern;
/// use scioxide_sparse::{SparseLuLimits, SparseLuSymbolic, SparseOrdering};
/// let pattern = StructuralPattern::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1])?;
/// let plan = SparseLuSymbolic::analyze(
///     pattern.view(),
///     SparseOrdering::default(),
///     SparseLuLimits::new(4, 4096),
/// )?;
/// assert_eq!(plan.column_permutation().new_to_old().len(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SparseOrdering<'a> {
    /// Keeps the original order.
    Natural,
    /// Approximate minimum degree on the solver's graph; see
    /// [`approximate_minimum_degree`].
    #[default]
    FillReducing,
    /// Uses the given permutation, whose length must equal the number of
    /// columns.
    Supplied(&'a Permutation),
}

/// Graph semantics selected by the consuming factorization.
pub(crate) enum OrderingGraph {
    Symmetric(Option<Triangle>),
    Normal,
}

impl SparseOrdering<'_> {
    pub(crate) fn prepare(
        self,
        pattern: StructuralPatternView<'_>,
        graph: OrderingGraph,
        byte_limit: usize,
    ) -> Result<Permutation, SciOxideError> {
        match self {
            Self::Natural => Permutation::identity(pattern.cols()),
            Self::Supplied(p) => {
                crate::helpers::exact(p.new_to_old().len(), pattern.cols())?;
                Permutation::new(scioxide_core::backend::try_copy_slice(
                    p.new_to_old(),
                    scioxide_core::quantity!("ordering_permutation"),
                )?)
            }
            Self::FillReducing => {
                // A conservative payload bound covers graph construction, the
                // adjacency lists and temporary quotient-graph and heap arrays.
                // These temporaries are dropped before retained solver arrays are built.
                let q = scioxide_core::quantity!("ordering_preparation_bytes");
                use scioxide_core::backend::{checked_add as add, checked_mul as mul};
                let nodes = add(
                    pattern.cols(),
                    if matches!(graph, OrderingGraph::Normal) {
                        pattern.rows()
                    } else {
                        0
                    },
                    q,
                )?;
                let words = add(
                    mul(40, add(nodes, 1, q)?, q)?,
                    mul(8, pattern.col_indices().len(), q)?,
                    q,
                )?;
                let bytes = mul(words, core::mem::size_of::<usize>(), q)?;
                if bytes > byte_limit {
                    return Err(SciOxideError::resource_limit(q, bytes, byte_limit));
                }
                let graph = match graph {
                    OrderingGraph::Normal => compressed::incidence(pattern)?,
                    OrderingGraph::Symmetric(triangle) => compressed::union(pattern, triangle)?,
                };
                amd::order(graph, pattern.cols(), byte_limit)
            }
        }
    }
}

fn levels(
    root: usize,
    graph: &Graph,
    excluded: &[bool],
    distance: &mut [usize],
    queue: &mut Vec<usize>,
) -> (usize, usize) {
    for &vertex in queue.iter() {
        distance[vertex] = usize::MAX;
    }
    queue.clear();
    queue.push(root);
    distance[root] = 0;
    let mut cursor = 0;
    while cursor < queue.len() {
        let vertex = queue[cursor];
        cursor += 1;
        for &next in &graph[vertex] {
            if !excluded[next] && distance[next] == usize::MAX {
                distance[next] = distance[vertex] + 1;
                queue.push(next);
            }
        }
    }
    let mut candidate = root;
    for &vertex in queue.iter() {
        if distance[vertex] > distance[candidate]
            || (distance[vertex] == distance[candidate]
                && (graph[vertex].len(), vertex) < (graph[candidate].len(), candidate))
        {
            candidate = vertex;
        }
    }
    (candidate, distance[candidate])
}
/// Computes the reverse Cuthill–McKee ordering of a square sparsity pattern,
/// which reduces its bandwidth and profile.
///
/// The ordering works on the undirected graph of `A + Aᵀ`. In each connected
/// component, it starts a breadth-first search at a pseudo-peripheral vertex,
/// found by repeated level-structure searches from a vertex of minimum degree.
/// The search visits the neighbors of each vertex by increasing degree, with
/// the smaller index first among equal degrees, and the visiting order of each
/// component is reversed. The result is a permutation `p[new] = old`. The cost
/// is `O(n + nnz)` per search plus sorting of neighbor lists.
///
/// # Examples
///
/// ```
/// use scioxide_core::{SciOxideErrorKind, StructuralPattern};
/// use scioxide_sparse::reverse_cuthill_mckee;
/// // The path 0 - 2 - 1 becomes tridiagonal when gathered as [0, 2, 1].
/// let p = StructuralPattern::new(3, 3, vec![0, 2, 4, 7], vec![0, 2, 1, 2, 0, 1, 2])?;
/// let order = reverse_cuthill_mckee(p.view())?;
/// assert_eq!(order.new_to_old(), [0, 2, 1]);
/// let rectangular = StructuralPattern::new(1, 2, vec![0, 1], vec![0])?;
/// let e = reverse_cuthill_mckee(rectangular.view()).unwrap_err();
/// assert_eq!(e.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if the pattern is not square.
/// - `SizeOverflow`, `AllocationFailure` if the graph cannot be sized or
///   allocated.
///
/// # References
///
/// - E. Cuthill and J. McKee, "Reducing the bandwidth of sparse symmetric
///   matrices", *Proc. 24th ACM National Conference* (1969), 157–172.
/// - A. George and J. W. H. Liu, *Computer Solution of Large Sparse Positive
///   Definite Systems*, Prentice-Hall, 1981, chapter 4.
pub fn reverse_cuthill_mckee(
    pattern: StructuralPatternView<'_>,
) -> Result<Permutation, SciOxideError> {
    let adjacency = compressed::union(pattern, None)?;
    let n = adjacency.len();
    let mut seen = try_filled(n, false, scioxide_core::quantity!("rcm_visited"))?;
    let mut distance = try_filled(n, usize::MAX, scioxide_core::quantity!("rcm_distances"))?;
    let mut queue = try_with_capacity(n, scioxide_core::quantity!("rcm_queue"))?;
    let mut neighbors = try_with_capacity(n, scioxide_core::quantity!("rcm_neighbors"))?;
    let mut order = try_with_capacity(n, scioxide_core::quantity!("rcm_order"))?;
    let mut seeds = try_with_capacity(n, scioxide_core::quantity!("rcm_component_seeds"))?;
    seeds.extend(0..n);
    seeds.sort_unstable_by_key(|&i| (adjacency[i].len(), i));
    for mut root in seeds {
        if seen[root] {
            continue;
        }
        let mut depth = 0;
        loop {
            let (candidate, next_depth) =
                levels(root, &adjacency, &seen, &mut distance, &mut queue);
            if next_depth <= depth {
                break;
            }
            root = candidate;
            depth = next_depth;
        }
        queue.clear();
        queue.push(root);
        seen[root] = true;
        let start = order.len();
        let mut cursor = 0;
        while cursor < queue.len() {
            let vertex = queue[cursor];
            cursor += 1;
            order.push(vertex);
            neighbors.clear();
            neighbors.extend(adjacency[vertex].iter().copied().filter(|&v| !seen[v]));
            neighbors.sort_unstable_by_key(|&v| (adjacency[v].len(), v));
            for &next in &neighbors {
                seen[next] = true;
                queue.push(next);
            }
        }
        order[start..].reverse();
    }
    Permutation::new(order)
}
/// Computes an approximate minimum degree (AMD) ordering of a square sparsity
/// pattern, which reduces fill in a Cholesky or LU factorization.
///
/// The ordering works on the undirected graph of `A + Aᵀ`; diagonal entries are
/// ignored and stored zeros count as edges. It eliminates vertices on a
/// quotient graph, counting reachable external variables, with element absorption, mass
/// elimination and merging of indistinguishable vertices. Among vertices of
/// equal degree the one with the smallest original index goes first, and the
/// result is postordered by the assembly tree. Dense vertices, with degree above
/// `max(16, 10 ⌊√n⌋)`, are placed last. The result is a permutation
/// `p[new] = old`.
///
/// Building the graph costs `O(n + nnz)`, and storage is linear in the size of
/// the graph, not of the fill. Each degree update scans the adjacent elements
/// and their active variables; elimination work depends on these neighborhoods.
/// AMD is a heuristic: it does not minimize fill in
/// general. For QR, which needs the graph of `AᵀA`, use
/// [`SparseOrdering::FillReducing`] in [`SparseQrSymbolic`](crate::SparseQrSymbolic),
/// which orders that graph without forming `AᵀA`.
///
/// # Examples
///
/// ```
/// use scioxide_core::StructuralPattern;
/// use scioxide_sparse::approximate_minimum_degree;
/// let p = StructuralPattern::new(3, 3, vec![0, 1, 3, 4], vec![1, 0, 2, 1])?;
/// let order = approximate_minimum_degree(p.view())?;
/// assert_eq!(order.new_to_old().len(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if the pattern is not square.
/// - `SizeOverflow`, `AllocationFailure` if the graph cannot be sized or
///   allocated.
///
/// # References
///
/// P. R. Amestoy, T. A. Davis and I. S. Duff, "An approximate minimum degree
/// ordering algorithm", *SIAM J. Matrix Anal. Appl.* 17 (1996), 886–905.
pub fn approximate_minimum_degree(
    pattern: StructuralPatternView<'_>,
) -> Result<Permutation, SciOxideError> {
    crate::helpers::exact(pattern.rows(), pattern.cols())?;
    SparseOrdering::FillReducing.prepare(pattern, OrderingGraph::Symmetric(None), usize::MAX)
}
/// Computes a distance-1 coloring of the undirected graph of a square sparsity
/// pattern `A + Aᵀ`: adjacent vertices get different colors.
///
/// Vertices are colored greedily in increasing index order, each with the
/// smallest color not used by an earlier neighbor. Colors are numbered from 0.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ColoringKind, SciOxideErrorKind, StructuralPattern};
/// use scioxide_sparse::distance_one_coloring;
/// // Vertices 0 and 1 are both adjacent to 2 but not to each other.
/// let p = StructuralPattern::new(3, 3, vec![0, 2, 4, 7], vec![0, 2, 1, 2, 0, 1, 2])?;
/// let schedule = distance_one_coloring(p.view())?;
/// assert_eq!(schedule.kind(), ColoringKind::DistanceOne);
/// assert_eq!(schedule.colors(), [0, 0, 1]);
/// let rectangular = StructuralPattern::new(1, 2, vec![0, 1], vec![0])?;
/// let e = distance_one_coloring(rectangular.view()).unwrap_err();
/// assert_eq!(e.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if the pattern is not square.
/// - `AllocationFailure` if the coloring cannot be allocated.
///
/// # References
///
/// A. H. Gebremedhin, F. Manne and A. Pothen, "What color is your Jacobian?
/// Graph coloring for computing derivatives", *SIAM Rev.* 47 (2005), 629–705.
pub fn distance_one_coloring(
    pattern: StructuralPatternView<'_>,
) -> Result<ColoringSchedule, SciOxideError> {
    let adjacency = compressed::union(pattern, None)?;
    let mut colors = try_filled(
        pattern.cols(),
        usize::MAX,
        scioxide_core::quantity!("graph_colors"),
    )?;
    let mut marks = try_filled(
        pattern.cols(),
        usize::MAX,
        scioxide_core::quantity!("graph_color_marks"),
    )?;
    for (vertex, neighbors) in adjacency.iter().enumerate() {
        for &other in neighbors.iter().filter(|&&other| other < vertex) {
            marks[colors[other]] = vertex;
        }
        colors[vertex] = marks
            .iter()
            .position(|&mark| mark != vertex)
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("pattern"),
                    "no available graph color",
                )
            })?;
    }
    ColoringSchedule::new(pattern.to_owned()?, ColoringKind::DistanceOne, colors)
}
/// Groups the columns of a sparse Jacobian pattern so that no two columns in a
/// group have an entry in the same row (structurally orthogonal columns).
///
/// One directional derivative per group then recovers every entry of the
/// Jacobian. This is a distance-1 coloring of the column-intersection graph of
/// `AᵀA`, computed greedily in increasing column order without forming that
/// graph. The pattern may be rectangular. Colors are numbered from 0.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ColoringKind, StructuralPattern};
/// use scioxide_sparse::derivative_coloring;
/// // Jacobian pattern [[x, x, 0], [0, x, x]]: columns 0 and 2 share no row.
/// let p = StructuralPattern::new(2, 3, vec![0, 2, 4], vec![0, 1, 1, 2])?;
/// let schedule = derivative_coloring(p.view())?;
/// assert_eq!(schedule.kind(), ColoringKind::DerivativeColumns);
/// assert_eq!(schedule.colors(), [0, 1, 0]);
/// assert_eq!(schedule.groups().next(), Some(&[0, 2][..]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `SizeOverflow`, `AllocationFailure` if the coloring cannot be sized or
///   allocated.
///
/// # References
///
/// - A. R. Curtis, M. J. D. Powell and J. K. Reid, "On the estimation of sparse
///   Jacobian matrices", *IMA J. Appl. Math.* 13 (1974), 117–119.
/// - A. H. Gebremedhin, F. Manne and A. Pothen, "What color is your Jacobian?
///   Graph coloring for computing derivatives", *SIAM Rev.* 47 (2005), 629–705.
pub fn derivative_coloring(
    pattern: StructuralPatternView<'_>,
) -> Result<ColoringSchedule, SciOxideError> {
    let n = pattern.cols();
    let mut rows = try_filled(
        pattern.col_indices().len(),
        0usize,
        scioxide_core::quantity!("column_incidence_rows"),
    )?;
    let pointers = crate::helpers::counting_transpose(
        n,
        pattern
            .row_ptr()
            .windows(2)
            .enumerate()
            .flat_map(|(row, bounds)| {
                pattern.col_indices()[bounds[0]..bounds[1]]
                    .iter()
                    .map(move |&col| (col, row))
            }),
        |slot, row| rows[slot] = row,
    )?;
    let mut colors = try_filled(n, usize::MAX, scioxide_core::quantity!("derivative_colors"))?;
    let mut marks = try_filled(
        n,
        usize::MAX,
        scioxide_core::quantity!("derivative_color_marks"),
    )?;
    for col in 0..n {
        for &row in &rows[pointers[col]..pointers[col + 1]] {
            let bounds = &pattern.row_ptr()[row..row + 2];
            for &other in pattern.col_indices()[bounds[0]..bounds[1]]
                .iter()
                .take_while(|&&other| other < col)
            {
                marks[colors[other]] = col;
            }
        }
        colors[col] = marks.iter().position(|&mark| mark != col).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("pattern"),
                "no available derivative color",
            )
        })?;
    }
    ColoringSchedule::new(pattern.to_owned()?, ColoringKind::DerivativeColumns, colors)
}
