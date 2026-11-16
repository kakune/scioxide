//! Search options, search state and the relaxation solver interface.
use super::*;

/// Identifies the family of a cutting plane added by [`BranchAndBound`](crate::BranchAndBound).
///
/// Each cut is checked when it is derived and stays with the node that added it
/// and that node's descendants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipCutKind {
    /// A row with integer coefficients on integer variables, whose right-hand side
    /// is rounded to an integer.
    IntegerRow,
    /// A cover inequality derived from an inequality row whose variables are
    /// all binary, with coefficients of one sign.
    BinaryCover,
}

/// Selects the branching variable of [`BranchAndBound`](crate::BranchAndBound).
///
/// Every rule is deterministic; ties go to the variable with the smaller index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MipBranching {
    /// The variable whose fractional part is closest to one half.
    MostFractional,
    /// The variable with the best pseudocost score, the average objective change
    /// per unit change in each direction; most fractional while no pseudocost is known.
    Pseudocost,
    /// Pseudocost branching that runs strong-branching probes until each
    /// direction has [`MipLimits::reliability`] observations.
    #[default]
    Reliability,
}

/// Sets the storage limits and optional work of [`BranchAndBound`](crate::BranchAndBound).
///
/// Setting an optional-work limit (cuts, strong branching, repair) to zero
/// disables that work. [`Self::max_search_bytes`] covers the search tree;
/// storage of the relaxation solver is limited separately by
/// [`Self::max_relaxation_bytes`]. Reaching a storage limit ends the search with
/// a `MemoryLimit` termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLimits {
    pub(super) max_open_nodes: usize,
    pub(super) max_depth: usize,
    pub(super) max_cuts: usize,
    pub(super) max_cut_nonzeros: usize,
    pub(super) max_search_bytes: usize,
    pub(super) max_relaxation_bytes: usize,
    pub(super) strong_candidates: usize,
    pub(super) strong_iterations: usize,
    pub(super) reliability: usize,
    pub(super) cut_rounds: usize,
    pub(super) repair: bool,
}
impl Default for MipLimits {
    fn default() -> Self {
        Self {
            max_open_nodes: 1000,
            max_depth: 1000,
            max_cuts: 128,
            max_cut_nonzeros: 8192,
            max_search_bytes: 64 * 1024 * 1024,
            max_relaxation_bytes: 64 * 1024 * 1024,
            strong_candidates: 4,
            strong_iterations: 32,
            reliability: 2,
            cut_rounds: 2,
            repair: true,
        }
    }
}
impl MipLimits {
    /// Sets [`Self::max_open_nodes`].
    #[must_use]
    pub const fn with_max_open_nodes(mut self, value: usize) -> Self {
        self.max_open_nodes = value;
        self
    }
    /// Returns the maximum number of open nodes, not counting the active node (default 1000).
    pub const fn max_open_nodes(&self) -> usize {
        self.max_open_nodes
    }
    /// Sets [`Self::max_depth`].
    #[must_use]
    pub const fn with_max_depth(mut self, value: usize) -> Self {
        self.max_depth = value;
        self
    }
    /// Returns the maximum depth of the tree, with the root at depth zero (default 1000).
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }
    /// Sets [`Self::max_cuts`].
    #[must_use]
    pub const fn with_max_cuts(mut self, value: usize) -> Self {
        self.max_cuts = value;
        self
    }
    /// Returns the maximum number of cuts added in one solve (default 128).
    pub const fn max_cuts(&self) -> usize {
        self.max_cuts
    }
    /// Sets [`Self::max_cut_nonzeros`].
    #[must_use]
    pub const fn with_max_cut_nonzeros(mut self, value: usize) -> Self {
        self.max_cut_nonzeros = value;
        self
    }
    /// Returns the maximum total number of nonzeros in added cuts (default 8192).
    pub const fn max_cut_nonzeros(&self) -> usize {
        self.max_cut_nonzeros
    }
    /// Sets [`Self::max_search_bytes`].
    #[must_use]
    pub const fn with_max_search_bytes(mut self, value: usize) -> Self {
        self.max_search_bytes = value;
        self
    }
    /// Returns the storage limit in bytes for the search tree and its buffers (default 64 MiB).
    pub const fn max_search_bytes(&self) -> usize {
        self.max_search_bytes
    }
    /// Sets [`Self::max_relaxation_bytes`].
    #[must_use]
    pub const fn with_max_relaxation_bytes(mut self, value: usize) -> Self {
        self.max_relaxation_bytes = value;
        self
    }
    /// Returns the storage limit in bytes for each relaxation solve (default 64 MiB).
    pub const fn max_relaxation_bytes(&self) -> usize {
        self.max_relaxation_bytes
    }
    /// Sets [`Self::strong_candidates`].
    #[must_use]
    pub const fn with_strong_candidates(mut self, value: usize) -> Self {
        self.strong_candidates = value;
        self
    }
    /// Returns the maximum number of candidate variables probed by strong branching at one node (default 4).
    pub const fn strong_candidates(&self) -> usize {
        self.strong_candidates
    }
    /// Sets [`Self::strong_iterations`].
    #[must_use]
    pub const fn with_strong_iterations(mut self, value: usize) -> Self {
        self.strong_iterations = value;
        self
    }
    /// Returns the pivot limit of each strong-branching probe; zero disables probes (default 32).
    pub const fn strong_iterations(&self) -> usize {
        self.strong_iterations
    }
    /// Sets [`Self::reliability`].
    #[must_use]
    pub const fn with_reliability(mut self, value: usize) -> Self {
        self.reliability = value;
        self
    }
    /// Returns the number of observations after which a pseudocost is reliable (default 2).
    pub const fn reliability(&self) -> usize {
        self.reliability
    }
    /// Sets [`Self::cut_rounds`].
    #[must_use]
    pub const fn with_cut_rounds(mut self, value: usize) -> Self {
        self.cut_rounds = value;
        self
    }
    /// Returns the maximum number of rounds of cut generation and reoptimization per node (default 2).
    pub const fn cut_rounds(&self) -> usize {
        self.cut_rounds
    }
    /// Sets [`Self::repair`].
    #[must_use]
    pub const fn with_repair(mut self, value: bool) -> Self {
        self.repair = value;
        self
    }
    /// Returns whether a round-and-fix heuristic runs at each node (default true).
    pub const fn repair(&self) -> bool {
        self.repair
    }
}

/// Counts the work of one [`BranchAndBound`](crate::BranchAndBound) search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MipStatistics {
    pub(super) nodes: usize,
    pub(super) relaxations: usize,
    pub(super) iterations: usize,
    pub(super) probes: usize,
    pub(super) cuts: usize,
    pub(super) cut_nonzeros: usize,
    pub(super) repairs: usize,
    pub(super) basis_reuses: usize,
    pub(super) peak_search_bytes: usize,
}
impl MipStatistics {
    /// Returns the number of processed tree nodes, not counting strong-branching probes.
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    /// Returns the number of LP relaxation solves, including probes and repairs.
    pub const fn relaxations(&self) -> usize {
        self.relaxations
    }
    /// Returns the total number of simplex pivots and bound flips in all relaxation solves.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the number of strong-branching probes, one per direction.
    pub const fn probes(&self) -> usize {
        self.probes
    }
    /// Returns the number of cuts added.
    pub const fn cuts(&self) -> usize {
        self.cuts
    }
    /// Returns the total number of nonzeros in the added cuts.
    pub const fn cut_nonzeros(&self) -> usize {
        self.cut_nonzeros
    }
    /// Returns the number of round-and-fix attempts.
    pub const fn repairs(&self) -> usize {
        self.repairs
    }
    /// Returns the number of relaxation solves that started from a previous basis.
    pub const fn basis_reuses(&self) -> usize {
        self.basis_reuses
    }
    /// Returns the peak storage of the search tree in bytes, as counted against [`MipLimits::max_search_bytes`].
    pub const fn peak_search_bytes(&self) -> usize {
        self.peak_search_bytes
    }
}

/// Solves the LP relaxations of [`BranchAndBound`](crate::BranchAndBound).
///
/// [`SimplexSolver`](crate::SimplexSolver) implements this trait and is the
/// default. A custom implementation is installed with
/// [`BranchAndBoundWorkspace::new`](crate::BranchAndBoundWorkspace::new). The
/// objective and constraint rows of a workspace are fixed when it is prepared;
/// only the variable bounds change from solve to solve. When the cuts active
/// at a node change the rows, the search prepares a new workspace for the
/// extended problem. The search checks every
/// returned point, objective and certificate against the original problem, so
/// an incorrect relaxation result cannot produce a false optimality claim.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{LpProblem, MipRelaxationSolver, SimplexSolver};
///
/// let problem = LpProblem::new(vec![1.0_f64]).unwrap();
/// let solver = SimplexSolver::default();
/// let _workspace = MipRelaxationSolver::workspace(&solver, &problem).unwrap();
/// assert!(MipRelaxationSolver::workspace_bytes(&solver, &problem).unwrap() > 0);
/// ```
pub trait MipRelaxationSolver<S: RealScalar + NumericScalar<Real = S>> {
    /// Storage shared by the relaxation solves of all nodes.
    type Workspace;
    /// Returns an upper bound on the bytes a workspace for `p` and its solves need.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the size computation overflows, or an error if
    /// the shape of `p` is not supported.
    fn workspace_bytes(&self, p: &LpProblem<S>) -> Result<usize, SciOxideError>;
    /// Creates a workspace for the objective and constraint rows of `p`.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be allocated, or an
    /// error if the shape of `p` is not supported.
    fn workspace(&self, p: &LpProblem<S>) -> Result<Self::Workspace, SciOxideError>;
    /// Returns whether the previous solve in `workspace` started from a stored basis.
    fn reused_basis(&self, _: &Self::Workspace) -> bool {
        false
    }
    /// Solves `p`, whose variable bounds may differ from the previous call, into `output`.
    ///
    /// The solve may perform at most `iterations` pivots or iterations and use
    /// at most `max_workspace_bytes` bytes. `context` is the child solve context
    /// that the search opens for this relaxation.
    ///
    /// # Errors
    ///
    /// Returns an error if the solve fails numerically, storage cannot be
    /// allocated, or `context` stops the work.
    fn solve_into(
        &mut self,
        p: &LpProblem<S>,
        workspace: &mut Self::Workspace,
        output: &mut crate::ProgrammingResult<S>,
        iterations: Budget,
        max_workspace_bytes: usize,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}
impl<S: RealScalar + NumericScalar<Real = S>> MipRelaxationSolver<S> for crate::SimplexSolver<S> {
    type Workspace = crate::MipSimplexWorkspace<S>;
    fn workspace_bytes(&self, p: &LpProblem<S>) -> Result<usize, SciOxideError> {
        checked_add(
            relaxation_bytes(p)?,
            scioxide_core::backend::checked_product(
                &[
                    2,
                    p.constraints().map_or(0, |c| c.len()),
                    self.basis_update_limit(),
                    core::mem::size_of::<S>(),
                ],
                scioxide_core::quantity!("mip_basis_updates"),
            )?,
            scioxide_core::quantity!("mip_relaxation_storage"),
        )
    }
    fn workspace(&self, p: &LpProblem<S>) -> Result<Self::Workspace, SciOxideError> {
        crate::MipSimplexWorkspace::new(self, p)
    }
    fn reused_basis(&self, w: &Self::Workspace) -> bool {
        w.reused()
    }
    fn solve_into(
        &mut self,
        p: &LpProblem<S>,
        w: &mut Self::Workspace,
        output: &mut crate::ProgrammingResult<S>,
        iterations: Budget,
        max_workspace_bytes: usize,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.workspace_bytes(p)? > max_workspace_bytes {
            return Err(invalid(
                scioxide_core::argument!("max_workspace_bytes"),
                "LP workspace admission exceeded",
            ));
        }
        w.solve_into(self, p, output, iterations, context)
    }
}

pub(super) fn model_payload_bound<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpProblem<S>,
) -> Result<usize, SciOxideError> {
    let entries = p.constraints().map_or(0, |c| match c.coefficients() {
        crate::CoefficientMatrix::Dense(a) => a.as_slice().len(),
        crate::CoefficientMatrix::Sparse(a) => a.nnz(),
    });
    let rows = p.constraints().map_or(0, |c| c.len());
    let items = checked_add(
        checked_add(p.dimension(), rows, scioxide_core::quantity!("lp_payload"))?,
        checked_add(entries, 1, scioxide_core::quantity!("lp_payload"))?,
        scioxide_core::quantity!("lp_payload"),
    )?;
    scioxide_core::backend::checked_product(&[items, 256], scioxide_core::quantity!("lp_payload"))
}

/// Describes an open node of the branch-and-bound tree.
pub struct MipNode<S> {
    pub(super) id: usize,
    pub(super) depth: usize,
    pub(super) lower_bound: Option<S>,
    pub(super) rows: Option<usize>,
    pub(super) branch: Option<(usize, bool, S, S)>,
}
impl<S: Copy> MipNode<S> {
    /// Returns the node number, in order of creation.
    pub const fn id(&self) -> usize {
        self.id
    }
    /// Returns the depth of the node; the root has depth zero.
    pub const fn depth(&self) -> usize {
        self.depth
    }
    /// Returns the lower bound of the node, from its parent or its own relaxation, if known.
    pub const fn lower_bound(&self) -> Option<S> {
        self.lower_bound
    }
}
impl<S> MipNode<S> {
    pub(super) fn bytes(&self) -> usize {
        core::mem::size_of::<Self>()
    }
}

pub(super) struct RowLink<S> {
    pub parent: Option<usize>,
    pub row: model::Row<S>,
}

pub(super) fn growth_bytes<T>(values: &Vec<T>, additional: usize) -> Result<usize, SciOxideError> {
    let needed = checked_add(
        values.len(),
        additional,
        scioxide_core::quantity!("mip_capacity"),
    )?;
    if needed <= values.capacity() {
        return Ok(0);
    }
    let capacity = scioxide_core::backend::checked_next_power_of_two(
        needed,
        scioxide_core::quantity!("mip_capacity"),
    )?;
    scioxide_core::backend::checked_product(
        &[capacity, core::mem::size_of::<T>()],
        scioxide_core::quantity!("mip_capacity"),
    )
}
pub(super) fn reserve<T>(
    values: &mut Vec<T>,
    additional: usize,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    let needed = checked_add(values.len(), additional, quantity)?;
    if needed > values.capacity() {
        let capacity = scioxide_core::backend::checked_next_power_of_two(needed, quantity)?;
        scioxide_core::backend::try_reserve(values, capacity - values.len(), quantity)?;
    }
    Ok(())
}
#[derive(Clone)]
pub(super) struct NodeRows<'a, S> {
    arena: &'a [RowLink<S>],
    cursor: Option<usize>,
}
impl<'a, S> Iterator for NodeRows<'a, S> {
    type Item = &'a model::Row<S>;
    fn next(&mut self) -> Option<Self::Item> {
        let link = &self.arena[self.cursor?];
        self.cursor = link.parent;
        Some(&link.row)
    }
}

/// Holds the search tree of a [`BranchAndBoundWorkspace`](super::BranchAndBoundWorkspace).
///
/// Each solve starts a new search. After a call, including one stopped by a
/// limit, the open nodes, the incumbent and the presolve map can be inspected
/// until the next call.
pub struct MipSearchState<
    S: RealScalar + NumericScalar<Real = S>,
    W = crate::MipSimplexWorkspace<S>,
> {
    pub(super) closed_bound: Option<S>,
    pub(super) model: Option<model::Model<S>>,
    pub(super) queue: Vec<MipNode<S>>,
    pub(super) active: Option<MipNode<S>>,
    pub(super) incumbent: Option<Candidate<S>>,
    pub(super) partial: Option<Candidate<S>>,
    pub(super) candidate: Candidate<S>,
    pub(super) vars: Vec<usize>,
    pub(super) stats: MipStatistics,
    pub(super) costs: Vec<Pseudocost<S>>,
    pub(super) next_id: usize,
    pub(super) row_links: Vec<RowLink<S>>,
    pub(super) cut_workspace: cuts::CutWorkspace<S>,
    pub(super) row_payload_bytes: usize,
    pub(super) relaxation: Option<W>,
    pub(super) probe_output: crate::ProgrammingResult<S>,
    pub(super) next_cut_ids: Vec<usize>,
    pub(super) lp_output: crate::ProgrammingResult<S>,
    pub(super) verifier: Option<crate::VerificationWorkspace<S>>,
    pub(super) lp: Option<LpProblem<S>>,
    pub(super) cut_ids: Vec<usize>,
    pub(super) bounds_scratch: Vec<(Option<S>, Option<S>)>,
}
impl<S: RealScalar + NumericScalar<Real = S>, W> Default for MipSearchState<S, W> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, W> MipSearchState<S, W> {
    /// Creates an empty state without allocating.
    pub fn new() -> Self {
        Self {
            closed_bound: None,
            model: None,
            queue: Vec::new(),
            active: None,
            incumbent: None,
            partial: None,
            candidate: Candidate {
                x: Vec::new(),
                fun: <S as scioxide_core::AlgebraicScalar>::ZERO,
                primal: <S as scioxide_core::AlgebraicScalar>::ZERO,
                bounds: <S as scioxide_core::AlgebraicScalar>::ZERO,
                integrality: <S as scioxide_core::AlgebraicScalar>::ZERO,
            },
            vars: Vec::new(),
            stats: MipStatistics {
                nodes: 0,
                relaxations: 0,
                iterations: 0,
                probes: 0,
                cuts: 0,
                cut_nonzeros: 0,
                repairs: 0,
                basis_reuses: 0,
                peak_search_bytes: 0,
            },
            costs: Vec::new(),
            next_id: 0,
            row_links: Vec::new(),
            cut_workspace: cuts::CutWorkspace::default(),
            row_payload_bytes: 0,
            relaxation: None,
            lp_output: crate::ProgrammingResult::empty(),
            probe_output: crate::ProgrammingResult::empty(),
            next_cut_ids: Vec::new(),
            verifier: None,
            lp: None,
            cut_ids: Vec::new(),
            bounds_scratch: Vec::new(),
        }
    }
    /// Returns the open nodes, in storage order rather than processing order.
    pub fn open_nodes(&self) -> &[MipNode<S>] {
        &self.queue
    }
    /// Returns the incumbent in original coordinates, if one was found.
    pub fn incumbent(&self) -> Option<&[S]> {
        self.incumbent.as_ref().map(|c| c.x.as_slice())
    }
    /// Returns the objective of the incumbent, recomputed from the original problem.
    pub fn incumbent_objective(&self) -> Option<S> {
        self.incumbent.as_ref().map(|c| c.fun)
    }
    /// Returns the node that was being processed when a limit stopped the search.
    pub const fn active_node(&self) -> Option<&MipNode<S>> {
        self.active.as_ref()
    }
    /// Returns the presolve map, if presolve completed.
    pub fn presolve_map(&self) -> Option<&MipPresolveMap<S>> {
        self.model.as_ref().map(|m| &m.map)
    }
    /// Returns the work counters of the search.
    pub const fn statistics(&self) -> MipStatistics {
        self.stats
    }
    pub(super) fn rows(&self, cursor: Option<usize>) -> NodeRows<'_, S> {
        NodeRows {
            arena: &self.row_links,
            cursor,
        }
    }
    /// Returns the kind and source row of each cut that applies to `node`.
    ///
    /// Source row indices refer to the presolved model; use
    /// [`MipPresolveMap::original_rows`] to map them to original rows.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `node` is not an open or active node of this state.
    pub fn cut_provenance<'a>(
        &'a self,
        node: &'a MipNode<S>,
    ) -> Result<impl Iterator<Item = (MipCutKind, usize)> + 'a, SciOxideError> {
        if !self
            .queue
            .iter()
            .chain(self.active.iter())
            .any(|retained| core::ptr::eq(retained, node))
        {
            return Err(invalid(
                scioxide_core::argument!("node"),
                "node is not retained by this search state",
            ));
        }
        Ok(self.rows(node.rows).filter_map(|r| r.provenance))
    }
    pub(super) fn append_row(
        &mut self,
        parent: Option<usize>,
        row: model::Row<S>,
    ) -> Result<Option<usize>, SciOxideError> {
        reserve(
            &mut self.row_links,
            1,
            scioxide_core::quantity!("mip_path_arena"),
        )?;
        let payload = row.entries.capacity() * core::mem::size_of::<(usize, S)>();
        self.row_payload_bytes = checked_add(
            self.row_payload_bytes,
            payload,
            scioxide_core::quantity!("mip_path_payload"),
        )?;
        let index = self.row_links.len();
        self.row_links.push(RowLink { parent, row });
        Ok(Some(index))
    }
    pub(super) fn reset(&mut self) {
        self.model = None;
        self.row_links.clear();
        self.row_payload_bytes = 0;
        self.relaxation = None;
        self.verifier = None;
        self.lp = None;
        self.cut_ids.clear();
        self.queue.clear();
        self.costs.clear();
        self.active = None;
        self.incumbent = None;
        self.partial = None;
        self.closed_bound = None;
        self.stats = MipStatistics::default();
        self.next_id = 0;
    }
    pub(super) fn bytes(&self) -> usize {
        self.row_links.capacity() * core::mem::size_of::<RowLink<S>>()
            + self.row_payload_bytes
            + self.cut_workspace.bytes()
            + self.model.as_ref().map_or(0, model::Model::bytes)
            + self.queue.capacity() * core::mem::size_of::<MipNode<S>>()
            + self.active.as_ref().map_or(0, MipNode::bytes)
            + self.costs.capacity() * core::mem::size_of::<Pseudocost<S>>()
            + self.vars.capacity() * core::mem::size_of::<usize>()
            + self.bounds_scratch.capacity() * core::mem::size_of::<(Option<S>, Option<S>)>()
            + (self.cut_ids.capacity() + self.next_cut_ids.capacity())
                * core::mem::size_of::<usize>()
            + self.candidate.x.capacity() * core::mem::size_of::<S>()
            + self
                .incumbent
                .iter()
                .chain(self.partial.iter())
                .map(|c| c.x.capacity() * core::mem::size_of::<S>())
                .sum::<usize>()
    }
}

pub(super) struct Pseudocost<S> {
    pub sums: [S; 2],
    pub counts: [usize; 2],
}

// Conservative native/cold factor and certificate staging bound. Product-form
// update capacity is charged separately by the provider; it is caller configurable.
pub(super) fn relaxation_bytes<S: RealScalar + NumericScalar<Real = S>>(
    p: &LpProblem<S>,
) -> Result<usize, SciOxideError> {
    let n = p.dimension();
    let m = p.constraints().map_or(0, |c| c.len());
    let order = checked_add(
        checked_add(n, n, scioxide_core::quantity!("lp_storage"))?,
        m,
        scioxide_core::quantity!("lp_storage"),
    )?;
    let square = scioxide_core::backend::checked_product(
        &[order, order],
        scioxide_core::quantity!("lp_storage"),
    )?;
    scioxide_core::backend::checked_product(
        &[
            checked_add(
                square,
                checked_add(order, 1, scioxide_core::quantity!("lp_storage"))?,
                scioxide_core::quantity!("lp_storage"),
            )?,
            4 * core::mem::size_of::<S>() + 4 * core::mem::size_of::<usize>(),
        ],
        scioxide_core::quantity!("lp_storage"),
    )
}
