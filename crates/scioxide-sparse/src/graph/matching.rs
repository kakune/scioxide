//! Maximum bipartite matching with a minimum vertex cover.
use super::GraphExecutionOptions;
use super::execution::{Attempt, grow, require};
use alloc::vec::Vec;
use scioxide_core::{ExecutionContext, SciOxideError, SolveScope, StructuralPatternView};

/// Reports the storage a [`MatchingWorkspace`] needs for one pattern.
#[derive(Debug, Clone, Copy)]
pub struct MatchingWorkspaceRequirements {
    rows: usize,
    cols: usize,
}
/// Stores the outputs and scratch of [`maximum_bipartite_matching_scoped`] for
/// reuse across calls.
///
/// Matching has no plan, so the workspace type answers the sizing queries:
/// [`workspace_requirements`](Self::workspace_requirements) and
/// [`workspace`](Self::workspace) take the pattern. A call rejects a workspace
/// that is too small instead of growing it.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::maximum_bipartite_matching_scoped as matching;
/// use scioxide_sparse::{CsrMatrix, GraphExecutionOptions, MatchingWorkspace};
/// // Both rows compete for column 0.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0_f64; 2])?;
/// let (p, options, c) = (
///     a.structural_pattern(),
///     GraphExecutionOptions::new(),
///     ExecutionContext::sequential(),
/// );
/// let mut workspace = MatchingWorkspace::new();
/// let error = matching(p, options, &mut workspace, c).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// assert!(workspace.last_accepted().is_none());
/// workspace.try_reserve(&MatchingWorkspace::workspace_requirements(p)?)?;
/// assert_eq!(matching(p, options, &mut workspace, c)?.cardinality(), 1);
/// assert_eq!(workspace.last_accepted().map(|m| m.cardinality()), Some(1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Default)]
pub struct MatchingWorkspace {
    attempt: Attempt,
    active: (usize, usize),
    left: Vec<Option<usize>>,
    right: Vec<Option<usize>>,
    depth: Vec<usize>,
    cursor: Vec<usize>,
    queue: Vec<usize>,
    stack: Vec<usize>,
    left_cover: Vec<bool>,
    right_cover: Vec<bool>,
}
impl MatchingWorkspace {
    /// Creates an empty workspace. Reserve storage with
    /// [`try_reserve`](Self::try_reserve) before use.
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the storage that [`maximum_bipartite_matching_scoped`] needs for
    /// the bipartite graph of `request`: one entry per row and per column.
    ///
    /// # Errors
    ///
    /// This function does not fail; it returns a `Result` like the other
    /// requirement queries.
    pub fn workspace_requirements(
        request: StructuralPatternView<'_>,
    ) -> Result<MatchingWorkspaceRequirements, SciOxideError> {
        Ok(MatchingWorkspaceRequirements {
            rows: request.rows(),
            cols: request.cols(),
        })
    }
    /// Creates a workspace sized for the bipartite graph of `request`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(request: StructuralPatternView<'_>) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Checks whether the workspace satisfies `requirements`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size of `requirements` overflows `usize`.
    pub fn check(
        &self,
        requirements: &MatchingWorkspaceRequirements,
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
    /// Storage never shrinks, so one workspace can serve several patterns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size of `requirements` overflows `usize`.
    /// - `AllocationFailure` if growth fails. The workspace may then be partly
    ///   grown, and stays usable.
    pub fn try_reserve(
        &mut self,
        requirements: &MatchingWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        requirements.bytes()?;
        grow(&mut self.left, requirements.rows, None)?;
        grow(&mut self.right, requirements.cols, None)?;
        grow(&mut self.depth, requirements.rows, 0)?;
        grow(&mut self.cursor, requirements.rows, 0)?;
        grow(&mut self.queue, requirements.rows, 0)?;
        grow(&mut self.stack, requirements.rows, 0)?;
        grow(&mut self.left_cover, requirements.rows, false)?;
        grow(&mut self.right_cover, requirements.cols, false)?;
        Ok(())
    }
    fn require(&self, requirements: &MatchingWorkspaceRequirements) -> Result<(), SciOxideError> {
        requirements.bytes()?;

        require([
            (self.left.len(), requirements.rows),
            (self.right.len(), requirements.cols),
            (self.depth.len(), requirements.rows),
            (self.cursor.len(), requirements.rows),
            (self.queue.len(), requirements.rows),
            (self.stack.len(), requirements.rows),
            (self.left_cover.len(), requirements.rows),
            (self.right_cover.len(), requirements.cols),
        ])?;
        Ok(())
    }
    /// Returns the record of the most recent call, including a call rejected
    /// before it started.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.attempt.record()
    }
    /// Returns the matching of the most recent call if it succeeded, and `None`
    /// otherwise.
    pub fn last_accepted(&self) -> Option<MatchingView<'_>> {
        self.attempt.succeeded().then(|| self.view())
    }
}
/// Borrows a matching and a minimum vertex cover.
///
/// The fields are public, so [`verify`](Self::verify) can also check a matching
/// computed elsewhere.
#[derive(Clone, Copy, Debug)]
pub struct MatchingView<'a> {
    /// Matched column of each row, or `None`.
    pub left: &'a [Option<usize>],
    /// Matched row of each column, or `None`.
    pub right: &'a [Option<usize>],
    /// Whether each row belongs to the vertex cover.
    pub left_cover: &'a [bool],
    /// Whether each column belongs to the vertex cover.
    pub right_cover: &'a [bool],
}
impl MatchingView<'_> {
    /// Returns the number of matched edges.
    pub fn cardinality(&self) -> usize {
        self.left.iter().filter(|v| v.is_some()).count()
    }
    /// Checks that this is a maximum matching of the bipartite graph of
    /// `pattern`.
    ///
    /// Returns `true` if every matched pair is a stored entry of `pattern`,
    /// `left` and `right` agree, every entry has an endpoint in the cover, and
    /// the cover has as many vertices as the matching has edges. By Kőnig's
    /// theorem the matching is then maximum and the cover minimum.
    pub fn verify(&self, pattern: StructuralPatternView<'_>) -> bool {
        let graph = Bipartite(pattern);
        let (rows, cols) = (pattern.rows(), pattern.cols());
        if self.left.len() != rows
            || self.right.len() != cols
            || self.left_cover.len() != rows
            || self.right_cover.len() != cols
        {
            return false;
        }
        for (u, matched) in self.left.iter().enumerate() {
            if let Some(v) = *matched
                && (v >= cols
                    || self.right[v] != Some(u)
                    || graph.row(u).binary_search(&v).is_err())
            {
                return false;
            }
            if graph
                .row(u)
                .iter()
                .any(|&v| !self.left_cover[u] && !self.right_cover[v])
            {
                return false;
            }
        }
        if self
            .right
            .iter()
            .enumerate()
            .any(|(v, u)| u.is_some_and(|u| u >= rows || self.left[u] != Some(v)))
        {
            return false;
        }
        self.cardinality()
            == self
                .left_cover
                .iter()
                .chain(self.right_cover)
                .filter(|&&b| b)
                .count()
    }
}
/// Holds a matching computed by [`maximum_bipartite_matching`].
#[derive(Debug)]
pub struct BipartiteMatching {
    workspace: MatchingWorkspace,
}
impl BipartiteMatching {
    /// Borrows the matching and its vertex cover.
    pub fn as_view(&self) -> MatchingView<'_> {
        self.workspace.view()
    }
}
impl MatchingWorkspace {
    fn view(&self) -> MatchingView<'_> {
        MatchingView {
            left: &self.left[..self.active.0],
            right: &self.right[..self.active.1],
            left_cover: &self.left_cover[..self.active.0],
            right_cover: &self.right_cover[..self.active.1],
        }
    }
}
/// Computes a maximum-cardinality matching of the bipartite graph of a sparsity
/// pattern, and a minimum vertex cover.
///
/// Rows form one vertex set and columns the other, and every stored entry
/// `(i, j)`, a stored zero included, is an edge between row `i` and column `j`;
/// the pattern may be rectangular. A matching is a set of edges no two of which
/// share a vertex, and a vertex cover a set of vertices that touches every
/// edge. By Kőnig's theorem a maximum matching and a minimum cover have the
/// same size, and [`MatchingView::verify`] checks this certificate. This is
/// [`maximum_bipartite_matching_with_options`] with
/// [`GraphExecutionOptions::new`].
///
/// ```
/// use scioxide_sparse::CsrMatrix;
/// // Row 0 may take column 0 or 1; row 1 only column 0.
/// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 1, 0], vec![1.0_f64; 3])?;
/// let matching = scioxide_sparse::maximum_bipartite_matching(a.structural_pattern())?;
/// let view = matching.as_view();
/// assert_eq!(view.left, &[Some(1), Some(0)]);
/// assert_eq!(view.right, &[Some(1), Some(0), None]);
/// // A vertex cover of the same size certifies that the matching is maximum.
/// assert_eq!(
///     (view.cardinality(), view.left_cover),
///     (2, &[true, true][..])
/// );
/// assert!(view.verify(a.structural_pattern()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `SizeOverflow` or `AllocationFailure` if the workspace cannot be created.
///
/// # References
///
/// J. E. Hopcroft and R. M. Karp, "An n^(5/2) algorithm for maximum matchings in
/// bipartite graphs", *SIAM J. Comput.* 2 (1973), 225–231.
pub fn maximum_bipartite_matching(
    pattern: StructuralPatternView<'_>,
) -> Result<BipartiteMatching, SciOxideError> {
    maximum_bipartite_matching_with_options(pattern, GraphExecutionOptions::new())
}
/// Computes a maximum matching and a minimum vertex cover of the bipartite
/// graph of `pattern` under the iteration limits of `options`, and returns an
/// owned result.
///
/// The call allocates its workspace and runs
/// [`maximum_bipartite_matching_scoped`] with
/// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
///
/// # Errors
///
/// The errors of [`maximum_bipartite_matching_scoped`], and `SizeOverflow` or
/// `AllocationFailure` if the workspace cannot be created.
pub fn maximum_bipartite_matching_with_options(
    pattern: StructuralPatternView<'_>,
    options: GraphExecutionOptions,
) -> Result<BipartiteMatching, SciOxideError> {
    let mut workspace = MatchingWorkspace::workspace(pattern)?;
    maximum_bipartite_matching_scoped(
        pattern,
        options,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok(BipartiteMatching { workspace })
}
/// Computes a maximum matching and a minimum vertex cover of the bipartite
/// graph of `pattern` into a workspace, in a root or nested solve scope.
///
/// Each phase of the Hopcroft–Karp algorithm finds the length of the shortest
/// augmenting paths by breadth-first search, then augments along a maximal set
/// of vertex-disjoint paths of that length by depth-first search. There are
/// O(√V) phases, and the total cost is O((V + E) √V) for V vertices and E
/// edges. Rows and columns are scanned in increasing index order, so the result
/// is deterministic. The cover is read from the vertices reachable from
/// unmatched rows by alternating paths. One iteration is counted per phase.
/// The call runs sequentially under every execution policy.
///
/// # Errors
///
/// - `DimensionMismatch` if the workspace is smaller than
///   [`MatchingWorkspace::workspace_requirements`] reports.
/// - `SizeOverflow` if the workspace size overflows `usize`.
/// - `ConvergenceFailure` if the iteration limit is exhausted.
/// - `ExecutionStopped` on cancellation or a passed deadline;
///   `CapabilityUnavailable`, `CapabilityFailure`, `ExecutionFailure` or
///   `AccountingInvariant` if the scope cannot run the call.
pub fn maximum_bipartite_matching_scoped<'a, 'scope, 'capability>(
    pattern: StructuralPatternView<'_>,
    options: GraphExecutionOptions,
    workspace: &'a mut MatchingWorkspace,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<MatchingView<'a>, SciOxideError> {
    workspace.attempt.reject();
    workspace.require(&MatchingWorkspace::workspace_requirements(pattern)?)?;
    let graph = Bipartite(pattern);
    let mut attempt = core::mem::take(&mut workspace.attempt);
    let result = attempt.run(
        scope,
        options,
        scioxide_core::algorithm_id!("sparse", "matching", 1),
        |context| graph.run(workspace, &mut || super::admit(context)),
    );
    workspace.attempt = attempt;
    result?;
    workspace.active = (pattern.rows(), pattern.cols());
    Ok(workspace.view())
}
// Rows of a pattern as the left vertex set of a bipartite graph.
#[derive(Clone, Copy)]
struct Bipartite<'a>(StructuralPatternView<'a>);
impl Bipartite<'_> {
    fn row(&self, u: usize) -> &[usize] {
        let ptr = self.0.row_ptr();
        &self.0.col_indices()[ptr[u]..ptr[u + 1]]
    }
    // Hopcroft-Karp with a Konig witness.
    fn run(
        &self,
        w: &mut MatchingWorkspace,
        tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let (rows, cols) = (self.0.rows(), self.0.cols());
        let (ptr, indices) = (self.0.row_ptr(), self.0.col_indices());
        let left = &mut w.left[..rows];
        let right = &mut w.right[..cols];
        let depth = &mut w.depth[..rows];
        let cursor = &mut w.cursor[..rows];
        let queue = &mut w.queue[..rows];
        let stack = &mut w.stack[..rows];
        let left_cover = &mut w.left_cover[..rows];
        let right_cover = &mut w.right_cover[..cols];
        left.fill(None);
        right.fill(None);
        loop {
            tick()?;
            depth.fill(usize::MAX);
            let mut tail = 0;
            for (u, m) in left.iter().enumerate() {
                if m.is_none() {
                    depth[u] = 0;
                    queue[tail] = u;
                    tail += 1;
                }
            }
            let mut head = 0;
            let mut shortest = usize::MAX;
            while head < tail {
                let u = queue[head];
                head += 1;
                if depth[u] >= shortest {
                    continue;
                }
                for &v in self.row(u) {
                    if let Some(next) = right[v] {
                        if depth[next] == usize::MAX {
                            depth[next] = depth[u] + 1;
                            queue[tail] = next;
                            tail += 1;
                        }
                    } else {
                        shortest = depth[u] + 1;
                    }
                }
            }
            if shortest == usize::MAX {
                break;
            }
            cursor.copy_from_slice(&ptr[..rows]);
            for root in 0..rows {
                if left[root].is_some() {
                    continue;
                }
                let mut len = 1;
                stack[0] = root;
                while len > 0 {
                    let u = stack[len - 1];
                    let mut advanced = false;
                    while cursor[u] < ptr[u + 1] {
                        let v = indices[cursor[u]];
                        cursor[u] += 1;
                        if let Some(next) = right[v] {
                            if depth[next] == depth[u] + 1 && depth[next] < shortest {
                                stack[len] = next;
                                len += 1;
                                advanced = true;
                                break;
                            }
                        } else if depth[u] + 1 == shortest {
                            let mut target = Some(v);
                            while len > 0 {
                                len -= 1;
                                let row = stack[len];
                                let old = left[row];
                                left[row] = target;
                                if let Some(col) = target {
                                    right[col] = Some(row);
                                }
                                target = old;
                            }
                            advanced = true;
                            break;
                        }
                    }
                    if !advanced {
                        depth[u] = usize::MAX;
                        len -= 1;
                    }
                }
            }
        }
        // Alternating reachability from unmatched rows yields (L \ Z) union (R intersect Z).
        left_cover.fill(true);
        right_cover.fill(false);
        let mut tail = 0;
        for (u, m) in left.iter().enumerate() {
            if m.is_none() {
                left_cover[u] = false;
                queue[tail] = u;
                tail += 1;
            }
        }
        let mut head = 0;
        while head < tail {
            let u = queue[head];
            head += 1;
            for &v in self.row(u) {
                if left[u] != Some(v) && !right_cover[v] {
                    right_cover[v] = true;
                    if let Some(next) = right[v]
                        && left_cover[next]
                    {
                        left_cover[next] = false;
                        queue[tail] = next;
                        tail += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

impl MatchingWorkspaceRequirements {
    /// Returns the number of bytes of workspace storage, not counting fixed-size
    /// fields, vector headers and allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn bytes(self) -> Result<usize, SciOxideError> {
        let rows = core::mem::size_of::<Option<usize>>()
            + 4 * core::mem::size_of::<usize>()
            + core::mem::size_of::<bool>();
        let cols = core::mem::size_of::<Option<usize>>() + core::mem::size_of::<bool>();
        super::workspace_bytes((self.rows, rows), (self.cols, cols))
    }
}
