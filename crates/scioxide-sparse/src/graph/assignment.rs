//! Linear assignment on sparse or dense cost matrices by shortest augmenting paths.
use super::execution::{Attempt, grow, require};
use super::{GraphExecutionOptions, finite_candidate, invalid};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    ExecutionContext, RealScalar, SciOxideError, SolveScope, StructuralPatternView,
};
/// Selects whether an assignment minimizes or maximizes the total cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignmentObjective {
    /// Minimizes the sum of the selected costs.
    Minimize,
    /// Maximizes the sum of the selected costs.
    Maximize,
}
/// Holds the result of an assignment problem: a full assignment, or the finding
/// that none exists.
///
/// Infeasibility is a property of the sparsity pattern. It is an outcome of
/// [`AssignmentPlan::solve`] and an error of [`minimum_cost_assignment`], which
/// returns only full assignments.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{AssignmentObjective, AssignmentOutcome, AssignmentPlan, CsrMatrix};
/// // Both rows can only take column 0, so no full assignment exists.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 0], vec![1.0_f64, 2.0])?;
/// let plan = AssignmentPlan::new(a.structural_pattern())?;
/// let min = AssignmentObjective::Minimize;
/// let outcome = plan.solve(a.values(), min)?;
/// assert!(matches!(outcome, AssignmentOutcome::Infeasible));
/// // Costs must match the pattern's stored entries one to one.
/// let error = plan.solve(&[1.0_f64], min).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub enum AssignmentOutcome<T> {
    /// Every vertex of the smaller side is assigned.
    Full(T),
    /// The pattern has no matching that covers the smaller side.
    Infeasible,
}

impl AssignmentObjective {
    fn cost<S: RealScalar>(self, value: S) -> S {
        if self == Self::Maximize {
            -value
        } else {
            value
        }
    }
}
/// Holds the structure of a linear assignment problem: which rows may be
/// assigned to which columns.
///
/// Given costs `c(i, j)` on the allowed pairs, the problem is to choose pairs so
/// that every vertex of the smaller side (rows or columns) is used exactly once
/// and every vertex of the larger side at most once, minimizing or maximizing
/// the total cost `Σ c(i, j)` of the chosen pairs.
/// [`from_dense`](Self::from_dense) allows every pair of a square problem.
/// [`new`](Self::new) allows the stored entries of a sparsity pattern, including
/// stored zeros, and forbids the others. Costs are passed to each solve, so one
/// plan serves many cost matrices of the same structure.
///
/// The solver uses the shortest augmenting path method. It assigns one vertex of
/// the smaller side at a time along a shortest alternating path in the reduced
/// costs, found by Dijkstra's algorithm with a binary heap, and then updates the
/// dual potentials.
///
/// # Duals and rounding
///
/// Maximization negates the costs, so the solver always minimizes. It keeps
/// potentials `u` on the smaller side and `v ≤ 0` on the larger side. The reduced
/// cost of a pair is `c(i, j) − u_i − v_j`; in exact arithmetic it is nonnegative
/// and zero on the selected pairs. After each augmentation the solver recomputes
/// `u_i = c(i, σ(i)) − v_σ(i)` from the column `σ(i)` assigned to row `i`, so
/// selected reduced costs stay within one rounding of zero. Unselected reduced
/// costs gain one rounding per dual update. A negative reduced cost is therefore
/// treated as zero when it is at least
///
/// ```text
/// −(8ε · max(|c(i, j)|, |u_i|, |v_j|) + Δ),
/// ```
///
/// where `ε` is the machine epsilon. The allowance `Δ` starts at zero and grows by
/// `64ε · (D + M)` after each augmentation, where `D` is the reduced-cost length of
/// the augmenting path and `M` the largest potential that the augmentation reads
/// or writes. This bounds the worst case; typical drift is far smaller. A more
/// negative reduced cost is a `NumericalInstability` error (`Breakdown`). Costs
/// whose arithmetic is exact, such as small integers, incur no rounding.
/// [`verify`](Self::verify) recomputes the residuals of any result.
///
/// ```
/// use scioxide_sparse::{AssignmentObjective, AssignmentOutcome, AssignmentPlan};
///
/// // Costs [[4, 1], [2, 5]] in row-major order: pairs (0, 1) and (1, 0) cost 1 + 2.
/// let plan = AssignmentPlan::from_dense(2)?;
/// let outcome = plan.solve(&[4.0, 1.0, 2.0, 5.0], AssignmentObjective::Minimize)?;
/// let AssignmentOutcome::Full(result) = outcome else {
///     unreachable!()
/// };
/// assert_eq!(result.as_view().objective, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. Burkard, M. Dell'Amico and S. Martello, *Assignment Problems*, SIAM,
/// 2009, chapter 4.
#[derive(Debug, Clone)]
pub struct AssignmentPlan {
    rows: usize,
    cols: usize,
    small: usize,
    large: usize,
    ptr: Vec<usize>,
    edges: Vec<(usize, usize)>,
    dense: bool,
    cost_entries: usize,
}
/// Reports the storage an [`AssignmentWorkspace`] needs for one plan.
#[derive(Debug, Clone, Copy)]
pub struct AssignmentWorkspaceRequirements {
    rows: usize,
    cols: usize,
}
impl AssignmentWorkspaceRequirements {
    /// Returns the number of bytes of workspace storage for costs of type `S`, not
    /// counting fixed-size fields, vector headers and allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let small = self.rows.min(self.cols);
        let large = self.rows.max(self.cols);
        let arrays = super::workspace_bytes(
            (small, core::mem::size_of::<S>()),
            (
                large,
                core::mem::size_of::<S>()
                    + core::mem::size_of::<Option<S>>()
                    + 5 * core::mem::size_of::<usize>(),
            ),
        )?;
        checked_add(
            arrays,
            checked_mul(
                checked_add(
                    self.rows,
                    self.cols,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                core::mem::size_of::<Option<usize>>(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )
    }
}
/// Stores the heap, matching, dual potentials and result of
/// [`AssignmentPlan::solve_scoped`] for reuse across calls.
///
/// A call rejects a workspace that is too small instead of growing it.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     AssignmentObjective, AssignmentOutcome, AssignmentPlan, AssignmentWorkspace,
///     GraphExecutionOptions,
/// };
/// let (plan, costs) = (AssignmentPlan::from_dense(2)?, [4.0_f64, 1.0, 2.0, 5.0]);
/// let (min, options) = (AssignmentObjective::Minimize, GraphExecutionOptions::new());
/// let (mut workspace, c) = (AssignmentWorkspace::new(), ExecutionContext::sequential());
/// let error = plan
///     .solve_scoped(&costs, min, options, &mut workspace, c)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// workspace.try_reserve(&plan.workspace_requirements(())?)?;
/// let outcome = plan.solve_scoped(&costs, min, options, &mut workspace, c)?;
/// let AssignmentOutcome::Full(view) = outcome else {
///     unreachable!()
/// };
/// assert_eq!((view.left, view.objective), (&[Some(1), Some(0)][..], 3.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct AssignmentWorkspace<S: RealScalar> {
    u: Vec<S>,
    v: Vec<S>,
    distance: Vec<Option<S>>,
    matched: Vec<usize>,
    previous: Vec<usize>,
    positions: Vec<usize>,
    heap: Vec<usize>,
    touched: Vec<usize>,
    left: Vec<Option<usize>>,
    right: Vec<Option<usize>>,
    rows: usize,
    cols: usize,
    value: Option<S>,
    attempt: Attempt,
}
impl<S: RealScalar> Default for AssignmentWorkspace<S> {
    fn default() -> Self {
        Self {
            u: Vec::new(),
            v: Vec::new(),
            distance: Vec::new(),
            matched: Vec::new(),
            previous: Vec::new(),
            positions: Vec::new(),
            heap: Vec::new(),
            touched: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            rows: 0,
            cols: 0,
            value: None,
            attempt: Attempt::default(),
        }
    }
}
impl<S: RealScalar> AssignmentWorkspace<S> {
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
        requirements: &AssignmentWorkspaceRequirements,
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
        requirements: &AssignmentWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        requirements.bytes::<S>()?;
        let n = requirements.rows.min(requirements.cols);
        let m = requirements.rows.max(requirements.cols);
        grow(&mut self.u, n, S::ZERO)?;
        grow(&mut self.v, m, S::ZERO)?;
        grow(&mut self.distance, m, None)?;
        grow(&mut self.matched, m, usize::MAX)?;
        grow(&mut self.previous, m, usize::MAX)?;
        grow(&mut self.positions, m, usize::MAX)?;
        self.heap
            .try_reserve_exact(m.saturating_sub(self.heap.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("assignment_heap"), m)
            })?;
        self.touched
            .try_reserve_exact(m.saturating_sub(self.touched.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("assignment_touched"), m)
            })?;
        grow(&mut self.left, requirements.rows, None)?;
        grow(&mut self.right, requirements.cols, None)?;
        Ok(())
    }
    fn require(&self, requirements: &AssignmentWorkspaceRequirements) -> Result<(), SciOxideError> {
        requirements.bytes::<S>()?;
        let n = requirements.rows.min(requirements.cols);
        let m = requirements.rows.max(requirements.cols);
        require([
            (self.u.len(), n),
            (self.v.len(), m),
            (self.distance.len(), m),
            (self.matched.len(), m),
            (self.previous.len(), m),
            (self.positions.len(), m),
            (self.heap.capacity(), m),
            (self.touched.capacity(), m),
            (self.left.len(), requirements.rows),
            (self.right.len(), requirements.cols),
        ])?;
        Ok(())
    }
    /// Returns the record of the most recent call, including a call rejected
    /// before it started.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.attempt.record()
    }
    /// Returns the assignment of the most recent call if it found a full
    /// assignment, and `None` after an error or an infeasible problem.
    pub fn last_accepted(&self) -> Option<AssignmentView<'_, S>> {
        self.value
            .filter(|_| self.attempt.succeeded())
            .map(|value| self.view(value))
    }
}
/// Holds the residuals of the optimality conditions of an assignment, as
/// recomputed by [`AssignmentPlan::verify`].
///
/// For an optimal assignment with its potentials, all three residuals are zero in
/// exact arithmetic. In floating point they measure rounding and do not prove
/// optimality. For a result of the solver, negative reduced costs stay within the
/// allowance described on [`AssignmentPlan`], and the reduced costs of selected
/// pairs within one rounding of zero.
#[derive(Clone, Copy, Debug)]
pub struct AssignmentResiduals<S: RealScalar> {
    /// Largest violation of dual feasibility: the largest negative reduced cost
    /// negated, or the largest positive potential of the larger side.
    pub dual_violation: S,
    /// Largest absolute reduced cost of a selected pair, or absolute potential of
    /// an unassigned vertex of the larger side.
    pub complementary_slackness: S,
    /// Larger of `|primal − dual|`, with costs negated for maximization, and the
    /// difference between the recomputed and the reported objective.
    pub objective_gap: S,
}
/// Borrows an assignment and its dual potentials.
///
/// The potentials belong to the minimization of the costs, or of their negatives
/// for maximization. The fields are public, so [`AssignmentPlan::verify`] can
/// also check an assignment computed elsewhere.
#[derive(Clone, Copy, Debug)]
pub struct AssignmentView<'a, S: RealScalar> {
    /// Assigned column of each row, or `None`.
    pub left: &'a [Option<usize>],
    /// Assigned row of each column, or `None`.
    pub right: &'a [Option<usize>],
    /// Sum of the selected costs, not negated for maximization.
    pub objective: S,
    /// Potentials `u` of the smaller side (the rows if the problem is square).
    pub small_potentials: &'a [S],
    /// Potentials `v ≤ 0` of the larger side.
    pub large_potentials: &'a [S],
}
/// Holds an assignment computed by [`AssignmentPlan::solve`] or
/// [`minimum_cost_assignment`].
///
/// ```
/// use scioxide_sparse::{AssignmentObjective, AssignmentOutcome, AssignmentPlan, CsrMatrix};
/// // Row 1 can only take column 0, so row 0 must take column 1.
/// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 1, 0], vec![3.0_f64, 1.0, 2.0])?;
/// let plan = AssignmentPlan::new(a.structural_pattern())?;
/// let max = AssignmentObjective::Maximize;
/// let outcome = plan.solve(a.values(), max)?;
/// let AssignmentOutcome::Full(result) = outcome else {
///     unreachable!()
/// };
/// let view = result.as_view();
/// assert_eq!((view.left, view.objective), (&[Some(1), Some(0)][..], 3.0));
/// assert_eq!(view.right, &[Some(1), Some(0), None]);
/// // Recomputed residuals include rounding, so compare them with a tolerance.
/// let residuals = plan.verify(a.values(), max, view)?;
/// assert!(residuals.dual_violation <= 1e-12 && residuals.objective_gap <= 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct FullAssignment<S: RealScalar> {
    workspace: AssignmentWorkspace<S>,
    objective: S,
}
impl<S: RealScalar> FullAssignment<S> {
    /// Borrows the assignment and its dual potentials.
    pub fn as_view(&self) -> AssignmentView<'_, S> {
        self.workspace.view(self.objective)
    }
}
impl<S: RealScalar> AssignmentWorkspace<S> {
    fn view(&self, objective: S) -> AssignmentView<'_, S> {
        AssignmentView {
            left: &self.left[..self.rows],
            right: &self.right[..self.cols],
            objective,
            small_potentials: &self.u[..self.rows.min(self.cols)],
            large_potentials: &self.v[..self.rows.max(self.cols)],
        }
    }
}

impl AssignmentPlan {
    /// Creates a plan for a dense square problem of order `n`.
    ///
    /// Every pair is allowed, and costs are passed as an `n × n` array in
    /// row-major order.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn from_dense(n: usize) -> Result<Self, SciOxideError> {
        let cost_entries = checked_mul(n, n, scioxide_core::quantity!("assignment_costs"))?;
        Ok(Self {
            rows: n,
            cols: n,
            small: n,
            large: n,
            ptr: Vec::new(),
            edges: Vec::new(),
            dense: true,
            cost_entries,
        })
    }
    /// Creates a plan whose allowed pairs are the stored entries of `pattern`.
    ///
    /// Costs are passed in the order of the stored entries, such as the values of
    /// a CSR matrix with this pattern. The pattern may have more rows than
    /// columns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an index count overflows `usize`.
    /// - `AllocationFailure` if the plan cannot be allocated.
    pub fn new(pattern: StructuralPatternView<'_>) -> Result<Self, SciOxideError> {
        let rows = pattern.rows();
        let cols = pattern.cols();
        let small = rows.min(cols);
        let large = rows.max(cols);
        let count = pattern.col_indices().len();
        let mut edges = try_filled(
            count,
            (0usize, 0usize),
            scioxide_core::quantity!("assignment_edges"),
        )?;
        let entries = pattern
            .row_ptr()
            .windows(2)
            .enumerate()
            .flat_map(|(row, bounds)| {
                (bounds[0]..bounds[1]).map(move |position| {
                    let column = pattern.col_indices()[position];
                    let (i, j) = if rows <= cols {
                        (row, column)
                    } else {
                        (column, row)
                    };
                    (i, (j, position))
                })
            });
        let ptr =
            crate::helpers::counting_transpose(small, entries, |slot, edge| edges[slot] = edge)?;
        Ok(Self {
            rows,
            cols,
            small,
            large,
            ptr,
            edges,
            dense: false,
            cost_entries: count,
        })
    }
    /// Returns the storage that [`solve_scoped`](Self::solve_scoped) needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `rows + cols` overflows `usize`.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<AssignmentWorkspaceRequirements, SciOxideError> {
        checked_add(
            self.rows,
            self.cols,
            scioxide_core::quantity!("assignment_dimensions"),
        )?;
        Ok(AssignmentWorkspaceRequirements {
            rows: self.rows,
            cols: self.cols,
        })
    }
    /// Creates a workspace sized for this plan, usable with any costs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace<S: RealScalar>(
        &self,
        request: (),
    ) -> Result<AssignmentWorkspace<S>, SciOxideError> {
        let mut workspace = AssignmentWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    fn row(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let range = if self.dense {
            0..self.large
        } else {
            self.ptr[i]..self.ptr[i + 1]
        };
        range.map(move |k| {
            if self.dense {
                (k, i * self.large + k)
            } else {
                self.edges[k]
            }
        })
    }
    fn validate_costs<S: RealScalar>(&self, costs: &[S]) -> Result<(), SciOxideError> {
        crate::helpers::exact(costs.len(), self.cost_entries)?;
        crate::helpers::finite(costs)
    }
    /// Solves the assignment problem for `costs` and returns an owned result.
    ///
    /// This is [`solve_scoped`](Self::solve_scoped) with
    /// [`GraphExecutionOptions::new`]: the call allocates its workspace and
    /// runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The errors of [`solve_scoped`](Self::solve_scoped) other than
    /// `ConvergenceFailure` and `ExecutionStopped`, and `SizeOverflow` or
    /// `AllocationFailure` if the workspace cannot be created.
    pub fn solve<S: RealScalar>(
        &self,
        costs: &[S],
        objective: AssignmentObjective,
    ) -> Result<AssignmentOutcome<FullAssignment<S>>, SciOxideError> {
        let mut workspace = self.workspace(())?;
        match self.solve_scoped(
            costs,
            objective,
            GraphExecutionOptions::new(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )? {
            AssignmentOutcome::Full(view) => {
                let objective = view.objective;
                Ok(AssignmentOutcome::Full(FullAssignment {
                    workspace,
                    objective,
                }))
            }
            AssignmentOutcome::Infeasible => Ok(AssignmentOutcome::Infeasible),
        }
    }
    /// Solves the assignment problem for `costs` into a workspace, in a root or
    /// nested solve scope.
    ///
    /// `costs` holds one value per allowed pair, in the order described on
    /// [`from_dense`](Self::from_dense) and [`new`](Self::new). Each vertex of the
    /// smaller side is assigned by one shortest-path search, so the worst-case
    /// cost is O(k E log(l + 1) + k + l) for k vertices on the smaller side, l on
    /// the larger side and E allowed pairs. One iteration is counted each time a
    /// search adds a row of the smaller side to its tree. The call runs
    /// sequentially under every execution policy.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `costs` does not have one value per allowed pair,
    ///   or the workspace is smaller than
    ///   [`workspace_requirements`](Self::workspace_requirements) reports.
    /// - `NonFinite` if a cost is NaN or infinite.
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `NumericalInstability`: `Overflow` if a reduced cost, distance,
    ///   potential or the total cost overflows; `Breakdown` if a reduced cost is
    ///   negative beyond the rounding allowance described on [`AssignmentPlan`].
    /// - `ConvergenceFailure` if the iteration limit is exhausted.
    /// - `ExecutionStopped` on cancellation or a passed deadline;
    ///   `CapabilityUnavailable`, `CapabilityFailure`, `ExecutionFailure` or
    ///   `AccountingInvariant` if the scope cannot run the call.
    pub fn solve_scoped<'a, 'scope, 'capability, S: RealScalar>(
        &self,
        costs: &[S],
        objective: AssignmentObjective,
        options: GraphExecutionOptions,
        workspace: &'a mut AssignmentWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AssignmentOutcome<AssignmentView<'a, S>>, SciOxideError> {
        workspace.attempt.reject();
        workspace.value = None;
        self.validate_costs(costs)?;
        workspace.require(&self.workspace_requirements(())?)?;
        let mut attempt = core::mem::take(&mut workspace.attempt);
        let result = attempt.run(
            scope,
            options,
            scioxide_core::algorithm_id!("sparse", "assignment", 1),
            |context| self.run(costs, objective, workspace, &mut || super::admit(context)),
        );
        workspace.attempt = attempt;
        let value = result?;
        workspace.rows = self.rows;
        workspace.cols = self.cols;
        workspace.value = value;
        Ok(match value {
            Some(value) => AssignmentOutcome::Full(workspace.view(value)),
            None => AssignmentOutcome::Infeasible,
        })
    }
    fn run<S: RealScalar>(
        &self,
        costs: &[S],
        objective: AssignmentObjective,
        workspace: &mut AssignmentWorkspace<S>,
        tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
    ) -> Result<Option<S>, SciOxideError> {
        let w = workspace;
        w.v[..self.large].fill(S::ZERO);
        w.matched[..self.large].fill(usize::MAX);
        w.left[..self.rows].fill(None);
        w.right[..self.cols].fill(None);
        w.distance[..self.large].fill(None);
        w.positions[..self.large].fill(usize::MAX);
        w.heap.clear();
        w.touched.clear();
        for i in 0..self.small {
            let Some(minimum) = self
                .row(i)
                .map(|(_, position)| objective.cost(costs[position]))
                .reduce(|a, b| a.min(b))
            else {
                return Ok(None);
            };
            w.u[i] = minimum;
        }
        let mut envelope = Envelope::new();
        for root in 0..self.small {
            for j in w.touched.drain(..) {
                w.distance[j] = None;
                w.positions[j] = usize::MAX;
            }
            w.heap.clear();
            let mut col = self.large;
            let mut row = root;
            let mut base = S::ZERO;
            let (mut target, distance) = loop {
                tick()?;
                for (j, position) in self.row(row) {
                    if w.positions[j] == self.large {
                        continue;
                    }
                    let cost = objective.cost(costs[position]);
                    let reduced = finite_candidate(finite_candidate(cost - w.u[row])? - w.v[j])?;
                    let scale = cost.abs().max(w.u[row].abs()).max(w.v[j].abs());
                    let candidate = finite_candidate(base + envelope.admit(reduced, scale)?)?;
                    if w.distance[j].is_none_or(|old| candidate < old) {
                        if w.distance[j].is_none() {
                            w.touched.push(j);
                        }
                        w.distance[j] = Some(candidate);
                        w.previous[j] = col;
                        super::heap::decrease(
                            &mut w.heap,
                            &mut w.positions[..self.large],
                            &w.distance,
                            j,
                        );
                    }
                }
                let Some(next) =
                    super::heap::pop(&mut w.heap, &mut w.positions[..self.large], &w.distance)
                else {
                    return Ok(None);
                };
                let next_distance = w.distance[next].ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("workspace"),
                        "missing heap distance",
                    )
                })?;
                if w.matched[next] == usize::MAX {
                    break (next, next_distance);
                }
                col = next;
                row = w.matched[col];
                base = next_distance;
            };
            // Settled columns have distance at most `distance`; each absorbs its
            // offset (zero for the target). Column duals only decrease.
            let mut magnitude = distance;
            for &j in &w.touched {
                if w.positions[j] == self.large {
                    let d = w.distance[j].ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("workspace"),
                            "missing settled distance",
                        )
                    })?;
                    w.v[j] = finite_candidate(w.v[j] - finite_candidate(distance - d)?)?;
                    magnitude = magnitude.max(w.v[j].abs());
                }
            }
            loop {
                let previous = w.previous[target];
                w.matched[target] = if previous == self.large {
                    root
                } else {
                    w.matched[previous]
                };
                if previous == self.large {
                    break;
                }
                target = previous;
            }
            // Every settled column is now matched, and its rows are exactly the
            // tree rows. Re-deriving their duals from the matched edges keeps
            // matched reduced costs within one rounding of zero.
            for &j in &w.touched {
                if w.positions[j] == self.large {
                    let i = w.matched[j];
                    let previous = w.u[i];
                    let cost = objective.cost(self.selected(costs, i, j)?);
                    w.u[i] = finite_candidate(cost - w.v[j])?;
                    magnitude = magnitude.max(previous.abs()).max(w.u[i].abs());
                }
            }
            envelope.grow(distance, magnitude)?;
        }
        let mut total = S::ZERO;
        for (col, &row) in w.matched[..self.large].iter().enumerate() {
            if row == usize::MAX {
                continue;
            }
            let (r, c) = if self.rows <= self.cols {
                (row, col)
            } else {
                (col, row)
            };
            w.left[r] = Some(c);
            w.right[c] = Some(r);
            total = finite_candidate(total + self.selected(costs, row, col)?)?;
        }
        Ok(Some(total))
    }
    // Stored entry of edge (i, j) in smaller-partition orientation. Sparse rows
    // are sorted by larger-partition index (canonical CSR, or a stable transpose).
    fn position(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.small || j >= self.large {
            None
        } else if self.dense {
            Some(i * self.large + j)
        } else {
            let row = &self.edges[self.ptr[i]..self.ptr[i + 1]];
            row.binary_search_by_key(&j, |edge| edge.0)
                .ok()
                .map(|k| row[k].1)
        }
    }
    // Original cost of a matched edge, which is stored by construction.
    fn selected<S: RealScalar>(&self, costs: &[S], i: usize, j: usize) -> Result<S, SciOxideError> {
        self.position(i, j)
            .map(|position| costs[position])
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "selected assignment edge is absent",
                )
            })
    }
    /// Checks an assignment for `costs` and recomputes the residuals of its
    /// optimality conditions.
    ///
    /// The assignment must be full, use only allowed pairs, and have consistent
    /// `left` and `right` partners. The residuals come from the reduced costs
    /// `c(i, j) − u_i − v_j` of all allowed pairs. The check accepts assignments
    /// computed elsewhere.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `costs` does not have one value per allowed pair,
    ///   or a slice of `result` does not match the size of the plan.
    /// - `NonFinite` if a cost is NaN or infinite.
    /// - `InvalidInput` if a potential or the objective of `result` is not finite,
    ///   its partners disagree, a vertex of the smaller side is unassigned, or a
    ///   selected pair is not allowed.
    /// - `NumericalInstability` (`Overflow`) if a recomputed sum overflows.
    pub fn verify<S: RealScalar>(
        &self,
        costs: &[S],
        objective: AssignmentObjective,
        result: AssignmentView<'_, S>,
    ) -> Result<AssignmentResiduals<S>, SciOxideError> {
        self.validate_costs(costs)?;
        const RESULT: scioxide_core::Argument = scioxide_core::argument!("result");

        for (actual, expected) in [
            (result.left.len(), self.rows),
            (result.right.len(), self.cols),
            (result.small_potentials.len(), self.small),
            (result.large_potentials.len(), self.large),
        ] {
            crate::helpers::exact(actual, expected)?;
        }

        if result
            .small_potentials
            .iter()
            .chain(result.large_potentials)
            .any(|v| !v.is_finite())
            || !result.objective.is_finite()
        {
            return Err(invalid(RESULT, "non-finite assignment evidence"));
        }
        for (r, c) in result.left.iter().enumerate() {
            if c.is_some_and(|c| c >= self.cols || result.right[c] != Some(r)) {
                return Err(invalid(RESULT, "inconsistent assignment partners"));
            }
        }
        for (c, r) in result.right.iter().enumerate() {
            if r.is_some_and(|r| r >= self.rows || result.left[r] != Some(c)) {
                return Err(invalid(RESULT, "inconsistent assignment partners"));
            }
        }
        let (small, large) = if self.rows <= self.cols {
            (result.left, result.right)
        } else {
            (result.right, result.left)
        };
        let mut dual_violation = S::ZERO;
        let mut complementary_slackness = S::ZERO;
        let mut primal = S::ZERO;
        let mut dual = S::ZERO;
        for (i, partner) in small.iter().enumerate() {
            let j = partner.ok_or_else(|| invalid(RESULT, "assignment is not full"))?;
            let position = self
                .position(i, j)
                .ok_or_else(|| invalid(RESULT, "assignment edge is absent"))?;
            primal = finite_candidate(primal + costs[position])?;
            dual = finite_candidate(dual + result.small_potentials[i])?;
            for (j, position) in self.row(i) {
                let reduced = finite_candidate(
                    finite_candidate(objective.cost(costs[position]) - result.small_potentials[i])?
                        - result.large_potentials[j],
                )?;
                dual_violation = dual_violation.max(-reduced);
                if Some(j) == *partner {
                    complementary_slackness = complementary_slackness.max(reduced.abs());
                }
            }
        }
        for (j, &v) in result.large_potentials.iter().enumerate() {
            dual = finite_candidate(dual + v)?;
            dual_violation = dual_violation.max(v);
            if large[j].is_none() {
                complementary_slackness = complementary_slackness.max(v.abs());
            }
        }
        let objective_gap = finite_candidate(objective.cost(primal) - dual)?
            .abs()
            .max(finite_candidate(primal - result.objective)?.abs());
        Ok(AssignmentResiduals {
            dual_violation,
            complementary_slackness,
            objective_gap,
        })
    }
}
// Rounding envelope documented on `AssignmentPlan`: a local evaluation
// allowance plus the drift accumulated by completed dual updates.
struct Envelope<S> {
    local: S,
    growth: S,
    drift: S,
}
impl<S: RealScalar> Envelope<S> {
    fn new() -> Self {
        Self {
            local: S::EPSILON * S::TWO.powi(3),
            growth: S::EPSILON * S::TWO.powi(6),
            drift: S::ZERO,
        }
    }
    // Clamp a reduced cost evaluated from operands of magnitude at most `scale`;
    // a violation beyond the envelope means the duals are inconsistent.
    fn admit(&self, reduced: S, scale: S) -> Result<S, SciOxideError> {
        if reduced < -(scale * self.local + self.drift) {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "assignment reduced cost is negative beyond the rounding envelope",
            ))
        } else {
            Ok(reduced.max(S::ZERO))
        }
    }
    // Account for one augmentation of path length `distance` whose dual updates
    // read or wrote magnitudes at most `magnitude`.
    fn grow(&mut self, distance: S, magnitude: S) -> Result<(), SciOxideError> {
        self.drift =
            finite_candidate(self.drift + (distance * self.growth + magnitude * self.growth))?;
        Ok(())
    }
}

/// Solves the linear assignment problem whose allowed pairs and costs are the
/// stored entries of `costs`, minimizing the total cost, and returns the full
/// assignment.
///
/// Every vertex of the smaller side (rows or columns) is assigned exactly once
/// and every vertex of the larger side at most once, minimizing `Σ c(i, j)` over
/// the selected pairs; a stored zero is an allowed pair of cost zero. This is
/// [`AssignmentPlan::new`] on the pattern of `costs` followed by
/// [`AssignmentPlan::solve`] with [`AssignmentObjective::Minimize`]. Keep the
/// plan to maximize, or to solve for other costs on the same pattern.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
/// // Costs [[4, 1], [2, 5]]: pairs (0, 1) and (1, 0) cost 1 + 2.
/// let c = CsrMatrix::from_triplets(2, 2, &[0, 0, 1, 1], &[0, 1, 0, 1], &[4.0_f64, 1., 2., 5.])?;
/// let assignment = scioxide_sparse::minimum_cost_assignment(&c)?;
/// let view = assignment.as_view();
/// assert_eq!((view.left, view.objective), (&[Some(1), Some(0)][..], 3.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DomainError` with argument `costs` if the stored entries admit no full
///   assignment. [`AssignmentPlan::solve`] returns
///   [`AssignmentOutcome::Infeasible`] instead.
/// - `NumericalInstability`: `Overflow` if a reduced cost, distance, potential
///   or the total cost overflows; `Breakdown` if a reduced cost is negative
///   beyond the rounding allowance described on [`AssignmentPlan`].
/// - `SizeOverflow` or `AllocationFailure` if the plan or workspace cannot be
///   allocated.
pub fn minimum_cost_assignment<S: RealScalar>(
    costs: &crate::CsrMatrix<S>,
) -> Result<FullAssignment<S>, SciOxideError> {
    match AssignmentPlan::new(costs.structural_pattern())?
        .solve(costs.values(), AssignmentObjective::Minimize)?
    {
        AssignmentOutcome::Full(assignment) => Ok(assignment),
        AssignmentOutcome::Infeasible => Err(SciOxideError::domain_error(
            scioxide_core::argument!("costs"),
            scioxide_core::DomainReason::Undefined,
            "the stored entries admit no full assignment",
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{NumericalIssue, SciOxideErrorKind, SciOxideErrorView};
    #[test]
    fn envelope_clamps_rounding_and_rejects_inconsistent_duals() {
        let mut envelope = Envelope::<f64>::new();
        // Within eight epsilon-scaled operand units: rounding, clamped to zero.
        let rounding = -4.0 * f64::EPSILON * 1000.0;
        assert_eq!(envelope.admit(rounding, 1000.0).unwrap(), 0.0);
        assert_eq!(envelope.admit(0.25, 1000.0).unwrap(), 0.25);
        let drifted = -16.0 * f64::EPSILON * 1000.0;
        let error = envelope.admit(drifted, 1000.0).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Breakdown,
                ..
            }
        ));
        // One augmentation of length 10 over duals of magnitude 1000 admits the
        // accumulated drift, but not a genuinely inconsistent dual.
        envelope.grow(10.0, 1000.0).unwrap();
        let drift = 64.0 * f64::EPSILON * 1010.0;
        assert!((envelope.drift - drift).abs() <= f64::EPSILON * drift);
        assert_eq!(envelope.admit(drifted, 1000.0).unwrap(), 0.0);
        let error = envelope.admit(-1e-9, 1000.0).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    }
}
