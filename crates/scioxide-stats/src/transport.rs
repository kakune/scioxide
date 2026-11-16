//! Wasserstein-1 distance in `ℝᵈ` by discrete optimal transport, with a solver you provide.
use crate::StatsTermination;
use crate::arithmetic::Compensated;
use crate::error_helpers::invalid;
use crate::{ObservationData, StatisticsOptions};
use alloc::vec::Vec;
use scioxide_core::Argument;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    EvaluationRole, FeasibilityTolerance, PartialResultTrust, RealScalar, ResidualTolerance,
    SciOxideError, SolveContext,
};

/// Discrete optimal transport problem passed to a [`TransportSolver`].
///
/// With source probabilities `a ∈ ℝⁿ`, target probabilities `b ∈ ℝᵐ` and costs
/// `cᵢⱼ`, the primal and dual linear programs are
///
/// ```text
/// minimize    Σᵢⱼ cᵢⱼ πᵢⱼ          subject to  Σⱼ πᵢⱼ = aᵢ,  Σᵢ πᵢⱼ = bⱼ,  π ≥ 0
/// maximize    Σᵢ aᵢ uᵢ + Σⱼ bⱼ vⱼ   subject to  uᵢ + vⱼ ≤ cᵢⱼ
/// ```
///
/// The cost is `cᵢⱼ = ‖xᵢ − yⱼ‖₂ / s`, where `s` is `cost_scale`. Pairs with
/// a zero probability have cost zero.
#[derive(Clone, Copy)]
pub struct TransportProblem<'a, S: RealScalar> {
    /// Source probabilities `a`, summing to 1 up to rounding.
    pub source: &'a [S],
    /// Target probabilities `b`, summing to 1 up to rounding.
    pub target: &'a [S],
    /// Costs `cᵢⱼ` in row-major order, `n · m` entries.
    pub costs: &'a [S],
    /// Scale `s > 0`: the largest absolute coordinate of a point with positive
    /// probability, or 1 if all are zero. The distance is `s` times the optimal cost.
    pub cost_scale: S,
}
/// Buffers into which a [`TransportSolver`] writes its primal and dual solution.
///
/// The buffers are zero when the solver is called. The dual potentials should
/// satisfy `uᵢ + vⱼ ≤ cᵢⱼ`.
pub struct TransportCandidate<'a, S: RealScalar> {
    /// Coupling `π ≥ 0` in row-major order, `n · m` entries.
    pub coupling: &'a mut [S],
    /// Source potentials `u`, `n` entries.
    pub source_dual: &'a mut [S],
    /// Target potentials `v`, `m` entries.
    pub target_dual: &'a mut [S],
}
/// Linear-programming solver for a [`TransportProblem`], used by
/// [`wasserstein_distance_nd`].
///
/// Implement it with any exact or approximate solver, for example a network
/// simplex method. The solver keeps its own scratch storage, and must respect
/// the iteration and evaluation limits and the cancellation of the
/// [`SolveContext`].
pub trait TransportSolver<S: RealScalar> {
    /// Solves `problem` and writes the coupling and dual potentials into `candidate`.
    ///
    /// Returns [`StatsTermination::Converged`], `BudgetExhausted` or `Stopped`.
    /// A `Converged` solution is checked against the tolerances in
    /// [`TransportOptions`].
    ///
    /// # Errors
    ///
    /// Returns the solver's own errors, which are passed on to the caller.
    fn solve(
        &mut self,
        problem: TransportProblem<'_, S>,
        candidate: TransportCandidate<'_, S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError>;
}
/// Size limits and verification tolerances for [`TransportWorkspace`].
///
/// A coupling is primal feasible if its marginal residual is at most the
/// feasibility threshold `atol + rtol`. A `Converged` solution must also have
/// a dual residual and a duality gap of at most the optimality threshold
/// `atol + rtol · max cᵢⱼ`, in normalized cost units.
#[derive(Clone, Copy, Debug)]
pub struct TransportOptions<S: RealScalar> {
    /// Largest accepted number of source-target pairs `n · m`.
    max_pairs: usize,
    /// Largest accepted size of the workspace in bytes, excluding solver storage.
    max_workspace_bytes: usize,
    /// Tolerance on the marginal residual, in probability units.
    feasibility: FeasibilityTolerance<S>,
    /// Tolerance on the dual residual and duality gap, in normalized cost units.
    optimality: ResidualTolerance<S>,
}

impl<S: RealScalar> Default for TransportOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> TransportOptions<S> {
    /// Returns options with no pair or byte limit and default feasibility and optimality tolerances.
    pub fn new() -> Self {
        Self {
            max_pairs: usize::MAX,
            max_workspace_bytes: usize::MAX,
            feasibility: FeasibilityTolerance::default(),
            optimality: ResidualTolerance::default(),
        }
    }
    /// Returns the largest accepted number of source-target pairs.
    pub const fn max_pairs(&self) -> usize {
        self.max_pairs
    }
    /// Sets the largest accepted number of source-target pairs.
    pub const fn with_max_pairs(mut self, value: usize) -> Self {
        self.max_pairs = value;
        self
    }
    /// Returns the byte limit of the workspace, excluding solver storage.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the byte limit of the workspace, excluding solver storage.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the tolerance on the marginal residual.
    pub const fn feasibility(&self) -> FeasibilityTolerance<S> {
        self.feasibility
    }
    /// Sets the tolerance on the marginal residual.
    pub const fn with_feasibility(mut self, value: FeasibilityTolerance<S>) -> Self {
        self.feasibility = value;
        self
    }
    /// Returns the tolerance on the dual residual and duality gap.
    pub const fn optimality(&self) -> ResidualTolerance<S> {
        self.optimality
    }
    /// Sets the tolerance on the dual residual and duality gap.
    pub const fn with_optimality(mut self, value: ResidualTolerance<S>) -> Self {
        self.optimality = value;
        self
    }
}
/// Result of [`wasserstein_distance_nd`].
///
/// The residuals are computed from the returned coupling and potentials in
/// compensated arithmetic. They are estimates, not bounds. A distance with a
/// status other than `Converged` is the cost of a coupling that meets the
/// marginals within tolerance, but not necessarily the minimum.
#[derive(Clone, Copy, Debug)]
pub struct TransportResult<S: RealScalar> {
    /// Scale `s`: original coordinate units per normalized cost unit.
    pub cost_scale: S,
    /// Status of the solve.
    pub termination: StatsTermination,
    /// Transport cost `s · Σ cᵢⱼ πᵢⱼ` in original units, or `None` if the
    /// coupling is not primal feasible or the solver was not called.
    pub distance: Option<S>,
    /// Marginal residual `max(Σᵢ |Σⱼ πᵢⱼ − aᵢ|, Σⱼ |Σᵢ πᵢⱼ − bⱼ|)`; 1 if the solver was not called.
    pub marginal_residual: S,
    /// Largest dual constraint violation `max(0, uᵢ + vⱼ − cᵢⱼ)`, in normalized cost units.
    pub dual_residual: S,
    /// Duality gap `|Σ cᵢⱼ πᵢⱼ − Σ aᵢ uᵢ − Σ bⱼ vⱼ|` in normalized cost units, or
    /// `None` if the solver was not called.
    pub normalized_duality_gap: Option<S>,
}
impl<S: RealScalar> TransportResult<S> {
    /// Returns the dual residual in original units, `s` times `dual_residual`.
    ///
    /// # Errors
    ///
    /// Fails if the product cannot be represented in `S`.
    pub fn original_dual_residual(self) -> Result<S, SciOxideError> {
        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(self.dual_residual))?
            .mul(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(self.cost_scale),
            )?)?
            .round_to()
    }
    /// Returns the duality gap in original units, `s` times `normalized_duality_gap`.
    ///
    /// # Errors
    ///
    /// Fails if the product cannot be represented in `S`.
    pub fn original_duality_gap(self) -> Result<Option<S>, SciOxideError> {
        self.normalized_duality_gap
            .map(|gap| {
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(gap))?
                    .mul(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(self.cost_scale),
                    )?)?
                    .round_to()
            })
            .transpose()
    }
}
/// Workspace for the Wasserstein-1 distance between two point clouds of fixed sizes.
///
/// It holds the cost matrix, the probabilities, the solution buffers and the
/// scratch for checking the solution: O(n · m) memory for `n` source and `m`
/// target points.
///
/// # References
///
/// - C. Villani, *Optimal Transport: Old and New*, Springer, 2009.
/// - G. Peyré and M. Cuturi, "Computational Optimal Transport", *Foundations
///   and Trends in Machine Learning*, 2019.
pub struct TransportWorkspace<S: RealScalar> {
    solve: crate::execution::SolveSlot<()>,
    rows: usize,
    cols: usize,
    costs: Vec<S>,
    coupling: Vec<S>,
    source: Vec<S>,
    target: Vec<S>,
    u: Vec<S>,
    v: Vec<S>,
    column_sums: Vec<Compensated<Scaled<f64>>>,
}
impl<S: RealScalar> Default for TransportWorkspace<S> {
    fn default() -> Self {
        Self {
            solve: Default::default(),
            rows: 0,
            cols: 0,
            costs: Vec::new(),
            coupling: Vec::new(),
            source: Vec::new(),
            target: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            column_sums: Vec::new(),
        }
    }
}
impl<S: RealScalar> TransportWorkspace<S> {
    /// Returns the heap bytes a workspace for `rows` source and `cols` target points needs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `cols` is zero.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
        if rows == 0 || cols == 0 {
            return Err(invalid(
                scioxide_core::argument!("rows"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "transport support must be nonempty",
            ));
        }
        let pairs = rows.checked_mul(cols).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("transport_pairs"))
        })?;
        let counts = pairs
            .checked_add(rows)
            .and_then(|n| n.checked_add(cols))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("transport_storage"))
            })?;
        crate::estimation::storage_bytes(&[
            (counts, core::mem::size_of::<S>()),
            (cols, core::mem::size_of::<Compensated<Scaled<f64>>>()),
        ])
    }
    /// Allocates the workspace for `rows` source and `cols` target points.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `cols` is zero.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if `rows · cols` exceeds the pair limit or the size
    ///   exceeds the byte limit in `options`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn reserve(
        &mut self,
        rows: usize,
        cols: usize,
        options: TransportOptions<S>,
    ) -> Result<(), SciOxideError> {
        let bytes = Self::required_bytes(rows, cols)?;
        crate::error_helpers::limit(
            scioxide_core::backend::checked_mul(
                rows,
                cols,
                scioxide_core::quantity!("transport_pairs"),
            )?,
            options.max_pairs,
            scioxide_core::quantity!("transport_pairs"),
        )?;
        crate::error_helpers::limit(
            bytes,
            options.max_workspace_bytes,
            scioxide_core::quantity!("transport_bytes"),
        )?;
        let mut next = Self {
            solve: Default::default(),
            rows,
            cols,
            costs: try_filled(
                rows * cols,
                S::ZERO,
                scioxide_core::quantity!("transport_costs"),
            )?,
            coupling: try_filled(
                rows * cols,
                S::ZERO,
                scioxide_core::quantity!("transport_coupling"),
            )?,
            source: try_filled(rows, S::ZERO, scioxide_core::quantity!("transport_source"))?,
            target: try_filled(cols, S::ZERO, scioxide_core::quantity!("transport_target"))?,
            u: try_filled(rows, S::ZERO, scioxide_core::quantity!("transport_dual_u"))?,
            v: try_filled(cols, S::ZERO, scioxide_core::quantity!("transport_dual_v"))?,
            column_sums: try_filled(
                cols,
                Compensated::zero(),
                scioxide_core::quantity!("transport_column_sums"),
            )?,
        };
        next.solve = core::mem::take(&mut self.solve);
        *self = next;
        Ok(())
    }

    fn verify(
        &mut self,
        termination: StatsTermination,
        scale: S,
        options: TransportOptions<S>,
    ) -> Result<TransportResult<S>, SciOxideError> {
        if self
            .coupling
            .iter()
            .chain(&self.u)
            .chain(&self.v)
            .any(|x| !x.is_finite())
            || self.coupling.iter().any(|&x| x < S::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("solver"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "transport provider returned nonfinite or negative coupling",
            ));
        }
        self.column_sums.fill(Compensated::zero());
        let mut objective = Compensated::<Scaled<f64>>::zero();
        let mut dual = Compensated::<Scaled<f64>>::zero();
        let mut row_residual = Compensated::<Scaled<f64>>::zero();
        let mut column_residual = Compensated::<Scaled<f64>>::zero();
        let mut dual_residual = Scaled::<f64>::ZERO;
        let mut max_cost = S::ZERO;
        for (i, (&a, &u)) in self.source.iter().zip(&self.u).enumerate() {
            let mut row = Compensated::<Scaled<f64>>::zero();
            dual = dual.added(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(a))?.mul(
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(u))?,
                )?,
            )?;
            for (j, (&v, col)) in self.v.iter().zip(&mut self.column_sums).enumerate() {
                let k = i * self.cols + j;
                let p = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                    self.coupling[k],
                ))?;
                row = row.added(p)?;
                *col = col.added(p)?;
                objective = objective.added(p.mul(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(self.costs[k]),
                )?)?)?;
                let violation = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(u))?
                    .add(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(v),
                    )?)?
                    .sub(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(self.costs[k]),
                    )?)?;
                if !violation.is_sign_negative() && violation.abs_cmp(dual_residual).is_gt() {
                    dual_residual = violation;
                }
                max_cost = max_cost.max(self.costs[k]);
            }
            let residual = row
                .total()?
                .sub(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(a),
                )?)?
                .abs();
            row_residual = row_residual.added(residual)?;
        }
        for ((sum, &b), &v) in self.column_sums.iter().zip(&self.target).zip(&self.v) {
            let residual = sum
                .total()?
                .sub(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(b),
                )?)?
                .abs();
            column_residual = column_residual.added(residual)?;
            dual = dual.added(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(b))?.mul(
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(v))?,
                )?,
            )?;
        }
        let row_residual = row_residual.total()?;
        let column_residual = column_residual.total()?;
        let primal_residual = if row_residual.abs_cmp(column_residual).is_gt() {
            row_residual
        } else {
            column_residual
        };
        let objective = objective.total()?;
        let gap = objective.sub(dual.total()?)?.abs();
        let feasible = primal_residual
            .abs_cmp(Scaled::<f64>::from_value(
                scioxide_core::RealScalar::into_f64(options.feasibility.threshold(S::ONE)?),
            )?)
            .is_le();
        let optimal_threshold = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
            options.optimality.threshold(max_cost)?,
        ))?;
        if termination == StatsTermination::Converged
            && (!feasible
                || dual_residual.abs_cmp(optimal_threshold).is_gt()
                || gap.abs_cmp(optimal_threshold).is_gt())
        {
            return Err(invalid(
                scioxide_core::argument!("solver"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "transport solver convergence failed independent verification",
            ));
        }
        Ok(TransportResult {
            cost_scale: scale,
            termination,
            distance: if feasible {
                Some(
                    objective
                        .mul(Scaled::<f64>::from_value(
                            scioxide_core::RealScalar::into_f64(scale),
                        )?)?
                        .round_to()?,
                )
            } else {
                None
            },
            marginal_residual: primal_residual.round_to()?,
            dual_residual: dual_residual.round_to()?,
            normalized_duality_gap: Some(gap.round_to()?),
        })
    }
}
fn normalize<S: RealScalar>(
    view: ObservationData<'_, S>,
    options: StatisticsOptions,
    output: &mut [S],
    argument: Argument,
) -> Result<(), SciOxideError> {
    let mut total = Compensated::<Scaled<f64>>::zero();
    for i in 0..view.samples() {
        if let Some((w, _)) = view.admit(0, i, options, argument)? {
            total = total.added(w)?;
        }
    }
    let total = total.total()?;
    if total.is_zero() {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass transport support",
        ));
    }
    for (i, slot) in output.iter_mut().enumerate() {
        *slot = if let Some((w, _)) = view.admit(0, i, options, argument)? {
            let p: S = w.div(total)?.round_to()?;
            if p == S::ZERO {
                return Err(invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "positive transport probability underflows solver scalar",
                ));
            }
            p
        } else {
            S::ZERO
        };
    }
    Ok(())
}

/// Computes the Wasserstein-1 distance between two weighted point clouds in
/// `ℝᵈ` with the Euclidean ground distance.
///
/// ```text
/// W₁(μ, ν) = min over couplings π of μ and ν of  Σᵢⱼ ‖xᵢ − yⱼ‖₂ πᵢⱼ
/// ```
///
/// `left` and `right` each hold one batch of points `xᵢ` and `yⱼ` with the
/// sizes the workspace was reserved for. Their weights are divided by their
/// sums. The [`StatisticsOptions`] set the missing-value and non-finite
/// policies. The method builds the cost matrix in O(n·m·d), calls `solver`
/// once, and checks the returned coupling and potentials in O(n·m); see
/// [`TransportResult`]. The solver call is recorded in `context` as one
/// linear-solve evaluation. If the budget does not allow it, or the solve is
/// stopped before the call, the result is `BudgetExhausted` or `Stopped`
/// without a distance.
///
/// # Errors
///
/// - `InvalidInput` if a view has more than one batch, the dimensions
///   differ, or the sizes differ from the reserved ones.
/// - `ResourceLimit` if the pair or byte limit in `options` is exceeded.
/// - `InvalidInput` if a side has zero total weight, or a positive
///   probability rounds to zero in `S`.
/// - `InvalidInput` for an invalid weight, or a missing or non-finite point
///   that the [`StatisticsOptions`] policies reject.
/// - `InvalidInput` if the solver returns a status other than `Converged`,
///   `BudgetExhausted` or `Stopped`.
/// - `InvalidInput` if the coupling is negative or not finite, or a
///   potential is not finite.
/// - `InvalidInput` if the solver returns `Converged` but the solution fails
///   the feasibility or optimality check.
/// - `ExecutionStopped` on cancellation or deadline while the costs are built.
/// - Errors returned by `solver`.
#[allow(clippy::too_many_arguments)]
pub fn wasserstein_distance_nd_scoped<
    'scope,
    'capability,
    S: RealScalar,
    L: crate::EmpiricalWeights,
    R: crate::EmpiricalWeights,
>(
    left: crate::BatchView<'_, S, L>,
    right: crate::BatchView<'_, S, R>,
    admission: StatisticsOptions,
    options: TransportOptions<S>,
    solver: &mut dyn TransportSolver<S>,
    workspace: &mut TransportWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<TransportResult<S>, SciOxideError> {
    workspace.solve.reject();

    let left = ObservationData::from_batch(left);
    let right = ObservationData::from_batch(right);
    if left.batches() != 1
        || right.batches() != 1
        || left.dimensions() != right.dimensions()
        || left.samples() != workspace.rows
        || right.samples() != workspace.cols
    {
        return Err(invalid(
            scioxide_core::argument!("right"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "transport requires matching reserved single-batch geometry",
        ));
    }
    crate::error_helpers::limit(
        scioxide_core::backend::checked_mul(
            workspace.rows,
            workspace.cols,
            scioxide_core::quantity!("transport_pairs"),
        )?,
        options.max_pairs,
        scioxide_core::quantity!("transport_pairs"),
    )?;
    crate::error_helpers::limit(
        TransportWorkspace::<S>::required_bytes(workspace.rows, workspace.cols)?,
        options.max_workspace_bytes,
        scioxide_core::quantity!("transport_bytes"),
    )?;
    normalize(
        left,
        admission,
        &mut workspace.source,
        scioxide_core::argument!("left"),
    )?;
    normalize(
        right,
        admission,
        &mut workspace.target,
        scioxide_core::argument!("right"),
    )?;
    let mut slot = core::mem::take(&mut workspace.solve);
    let result = slot.run(
        scope.into(),
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "transport", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            context.execution().check()?;
            let mut scale = S::ZERO;
            for (view, weights) in [
                (left, workspace.source.as_slice()),
                (right, workspace.target.as_slice()),
            ] {
                for (i, &w) in weights.iter().enumerate() {
                    if w != S::ZERO {
                        for d in 0..view.dimensions() {
                            scale = scale.max(view.value(0, i, d).abs());
                        }
                    }
                }
            }
            if scale == S::ZERO {
                scale = S::ONE;
            }
            for i in 0..workspace.rows {
                context.execution().check()?;
                for j in 0..workspace.cols {
                    let mut norm = Compensated::<Scaled<f64>>::zero();
                    if workspace.source[i] != S::ZERO && workspace.target[j] != S::ZERO {
                        for d in 0..left.dimensions() {
                            norm = norm.added(
                                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                                    left.value(0, i, d),
                                ))?
                                .sub(Scaled::<f64>::from_value(
                                    scioxide_core::RealScalar::into_f64(right.value(0, j, d)),
                                )?)?
                                .square()?,
                            )?;
                        }
                    }
                    workspace.costs[i * workspace.cols + j] = norm
                        .total()?
                        .sqrt()?
                        .div(Scaled::<f64>::from_value(
                            scioxide_core::RealScalar::into_f64(scale),
                        )?)?
                        .round_to()?;
                }
            }
            let empty = |termination| TransportResult {
                cost_scale: scale,
                termination,
                distance: None,
                marginal_residual: S::ONE,
                dual_residual: S::ZERO,
                normalized_duality_gap: None,
            };
            if let Some(stop) = context.stop_reason()? {
                return Ok(empty(StatsTermination::Stopped(stop)));
            }
            if !context
                .record_provider(EvaluationRole::LinearSolve)?
                .is_granted()
            {
                return Ok(empty(StatsTermination::BudgetExhausted));
            }
            workspace.coupling.fill(S::ZERO);
            workspace.u.fill(S::ZERO);
            workspace.v.fill(S::ZERO);
            let termination = solver.solve(
                TransportProblem {
                    source: &workspace.source,
                    target: &workspace.target,
                    costs: &workspace.costs,
                    cost_scale: scale,
                },
                TransportCandidate {
                    coupling: &mut workspace.coupling,
                    source_dual: &mut workspace.u,
                    target_dual: &mut workspace.v,
                },
                context,
            )?;
            if !matches!(
                termination,
                StatsTermination::Converged
                    | StatsTermination::BudgetExhausted
                    | StatsTermination::Stopped(_)
            ) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("termination"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "injected transport solver returned an inapplicable termination",
                ));
            }
            let result = workspace.verify(termination, scale, options)?;
            if result.distance.is_some() {
                context.commit_accepted(if termination == StatsTermination::Converged {
                    PartialResultTrust::ResidualChecked
                } else {
                    PartialResultTrust::Admissible
                })?;
            }
            Ok(result)
        },
        |result| {
            (
                match result {
                    Ok(report) if report.termination.is_success() => {
                        scioxide_core::AttemptOutcome::Succeeded
                    }
                    Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                    Err(_) => scioxide_core::AttemptOutcome::Failed,
                },
                None,
            )
        },
    );
    workspace.solve = slot;
    result
}
impl<S: RealScalar> TransportWorkspace<S> {
    /// Returns diagnostics of the last attempted solve.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.solve.last_attempt()
    }
}
/// Computes the Wasserstein distance with owned workspace.
///
/// # Errors
///
/// Returns allocation and scoped transport errors.
pub fn wasserstein_distance_nd<
    S: RealScalar,
    L: crate::EmpiricalWeights,
    R: crate::EmpiricalWeights,
>(
    left: crate::BatchView<'_, S, L>,
    right: crate::BatchView<'_, S, R>,
    admission: StatisticsOptions,
    options: TransportOptions<S>,
    solver: &mut dyn TransportSolver<S>,
) -> Result<TransportResult<S>, SciOxideError> {
    let mut workspace = TransportWorkspace::default();
    workspace.reserve(left.samples(), right.samples(), options)?;
    wasserstein_distance_nd_scoped(
        left,
        right,
        admission,
        options,
        solver,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
