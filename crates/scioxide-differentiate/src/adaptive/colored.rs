//! Sparse Jacobians from a column coloring, refined column by column.

use super::{
    AdaptiveBuffers, AdaptivePlan, RefinementProgress, adaptive_axis, initial_step, invalid,
};
use crate::options::DifferenceShape;
use crate::sampling::stop;
use crate::stencil::{Stencil, bound_for};
use crate::validation::validate_problem;
use crate::workspace;
use crate::{
    AdaptiveDifference, AdaptiveDifferenceMethod, DerivativeSource, DifferenceStep,
    DifferentiationReport, DifferentiationResult, DifferentiationTermination, SampleFunction,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AlgebraicScalar, Bounds, ColoringKind, ColoringSchedule, DifferentiationLocation,
    ErrorTolerance, ExecutionPolicy, NormSelection, NumericScalar, RealScalar, SciOxideError,
    SolveBudgets, SolveContext,
};
/// Computes a sparse Jacobian and error estimates from a column coloring.
///
/// Columns of the same color have no nonzero row in common, so one
/// perturbation of all columns of a color yields each of their entries. Each
/// column is refined by the extrapolation of
/// [`AdaptiveDifference`], and the stored entries of the pattern are recovered
/// from the samples. Values and error estimates follow the order of the stored
/// entries, row by row. The pattern must contain every entry that can be
/// nonzero; the plan does not check the callback against it.
/// `scioxide_sparse::derivative_coloring` computes a coloring from a pattern.
///
/// Construction precomputes, for each column, the rows of its stored entries,
/// their positions in the output and their absolute tolerances, in `O(n + nnz)`
/// time and storage for `n` columns and `nnz` stored entries. Each level of a
/// color group costs two evaluations, so a Jacobian with `c` colors costs at
/// most `1 + 2cL` evaluations for `L` levels instead of `1 + 2nL`, including
/// groups mixing central and inward one-sided stencils near bounds.
///
/// With a [`Parallel`](crate::Parallel) callback, independent samples run on
/// the executor and their results are combined in sample order, so a
/// deterministic callback gives the same values for any number of workers.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ColoringKind, ColoringSchedule, StructuralPattern};
/// use scioxide_differentiate::{AdaptiveDifference, ColoredJacobianPlan};
///
/// // f(x) = (x₀², x₁²) has a diagonal Jacobian, so both columns share color 0.
/// let pattern = StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1])?;
/// let schedule = ColoringSchedule::new(pattern, ColoringKind::DerivativeColumns, vec![0, 0])?;
/// let plan = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default())?;
/// let f = |x: &[f64], y: &mut [f64]| {
///     y[0] = x[0] * x[0];
///     y[1] = x[1] * x[1];
///     Ok(())
/// };
/// let result = plan.evaluate(f, &[1.0, 2.0], None)?;
/// let values = result.values().unwrap(); // J₀₀ = 2x₀ = 2, J₁₁ = 2x₁ = 4
/// assert!((values[0] - 2.0).abs() < 1e-8 && (values[1] - 4.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. R. Curtis, M. J. D. Powell and J. K. Reid, "On the estimation of sparse
/// Jacobian matrices", *J. Inst. Math. Appl.* 13, 1974, 117–119.
#[derive(Debug, Clone)]
pub struct ColoredJacobianPlan<'a, S: RealScalar> {
    schedule: &'a ColoringSchedule,
    columns: Vec<ColumnPlan<S>>,
    levels: usize,
    step: DifferenceStep<S>,
    ratio: S,
    execution: ExecutionPolicy,
    max_workspace_bytes: usize,
    budgets: SolveBudgets,
}
#[derive(Debug, Clone)]
struct ColumnPlan<S: RealScalar> {
    options: AdaptiveDifference<S>,
    rows: Vec<usize>,
    entries: Vec<usize>,
}
impl<S: RealScalar> ColumnPlan<S> {
    // Each column is a one-input Jacobian over the rows of its stored entries.
    fn plan(&self) -> AdaptivePlan<'_, S> {
        AdaptivePlan {
            operation: DifferenceShape::Jacobian,
            inputs: 1,
            outputs: self.rows.len(),
            len: self.rows.len(),
            options: &self.options,
        }
    }
}
#[derive(Debug, Clone)]
struct ColumnWork<T: NumericScalar> {
    workspace: AdaptiveBuffers<T>,
    progress: RefinementProgress<T::Real>,
    stencil: Option<Stencil<T::Real>>,
    contraction: T::Real,
    step: T::Real,
    termination: Option<DifferentiationTermination>,
}
/// Scratch for a colored Jacobian: `O(n + m + nnz·L)` elements, allocated at
/// reservation. At most three perturbed points and callback outputs are held at
/// once, for any number of colors or workers.
#[derive(Debug, Clone)]
struct ColoredBuffers<T: NumericScalar> {
    columns: Vec<ColumnWork<T>>,
    points: Vec<T::Real>,
    samples: Vec<T>,
    base: Vec<T>,
    values: Vec<T>,
    errors: Vec<T::Real>,
}
impl<'a, S: RealScalar> ColoredJacobianPlan<'a, S> {
    /// Prepares a plan for the pattern and coloring in `schedule`, refined
    /// with the settings of `method`.
    ///
    /// A vector absolute tolerance has one value per stored entry. An empty
    /// pattern gives an empty result without callback calls.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`schedule`, `UnsupportedConfiguration`) if the
    ///   schedule is not a [`ColoringKind::DerivativeColumns`] coloring.
    /// - `InvalidInput` (`method`, `OutOfRange`) if the levels are outside 3
    ///   to 64 or the tolerance norm is not the maximum norm.
    /// - `InvalidInput` (`method`, `Inconsistent`) if the zero-direction policy
    ///   is not the default.
    /// - `DimensionMismatch` if per-coordinate steps or a vector absolute
    ///   tolerance have the wrong length.
    /// - `SizeOverflow` or `AllocationFailure` if the plan cannot be stored.
    pub fn new(
        schedule: &'a ColoringSchedule,
        method: AdaptiveDifference<S>,
    ) -> Result<Self, SciOxideError> {
        let options = method;
        if schedule.kind() != ColoringKind::DerivativeColumns {
            return Err(crate::validation::invalid(
                scioxide_core::argument!("schedule"),
                "Jacobian recovery requires derivative-column coloring",
            ));
        }
        let pattern = schedule.pattern();
        let len = pattern.col_indices().len();
        options.validate_shape(pattern.cols(), len, scioxide_core::argument!("method"))?;
        options.require_mathematical_zero(scioxide_core::argument!("method"))?;
        let mut counts = try_filled(
            pattern.cols(),
            0usize,
            scioxide_core::quantity!("column_entry_counts"),
        )?;
        for &col in pattern.col_indices() {
            counts[col] += 1;
        }
        // Every column is refined as an independent one-input Jacobian over the
        // rows where it has stored entries. The unit tolerance is a placeholder;
        // it is replaced below once each column's CSR entries are known, so the
        // componentwise tolerances follow the caller's stored-entry order.
        let mut columns =
            try_with_capacity(pattern.cols(), scioxide_core::quantity!("column_plans"))?;
        for &count in &counts {
            columns.push(ColumnPlan {
                options: AdaptiveDifference {
                    execution: options.execution,
                    max_workspace_bytes: options.max_workspace_bytes,
                    budgets: options.budgets,
                    method: options.method,
                    levels: options.levels,
                    step: DifferenceStep::default(),
                    tolerance: ErrorTolerance::scalar(S::ONE, S::ZERO, NormSelection::Infinity)?,
                    zero_direction: options.zero_direction,
                },
                rows: try_with_capacity(count, scioxide_core::quantity!("column_rows"))?,
                entries: try_with_capacity(count, scioxide_core::quantity!("column_entries"))?,
            });
        }
        for (row, span) in pattern.row_ptr().windows(2).enumerate() {
            for entry in span[0]..span[1] {
                let column = &mut columns[pattern.col_indices()[entry]];
                column.rows.push(row);
                column.entries.push(entry);
            }
        }
        for column in &mut columns {
            let mut absolute = try_with_capacity(
                column.entries.len(),
                scioxide_core::quantity!("column_absolute_tolerances"),
            )?;
            for &entry in &column.entries {
                absolute.push(options.tolerance.absolute().component(entry, len)?);
            }
            column.options.tolerance = ErrorTolerance::try_vector(
                &absolute,
                options.tolerance.relative(),
                NormSelection::Infinity,
            )?;
        }
        Ok(Self {
            schedule,
            columns,
            execution: options.execution,
            max_workspace_bytes: options.max_workspace_bytes,
            budgets: options.budgets,
            levels: options.levels,
            step: options.step,
            ratio: match options.method {
                AdaptiveDifferenceMethod::Richardson => S::TWO,
                AdaptiveDifferenceMethod::Ridders => S::try_from_f64_lossy(1.4)?,
            },
        })
    }
    /// Returns the coloring schedule the plan was prepared with.
    pub const fn schedule(&self) -> &'a ColoringSchedule {
        self.schedule
    }
    /// Returns the number of stored entries of the pattern.
    pub fn output_len(&self) -> usize {
        self.schedule.pattern().col_indices().len()
    }
    /// Computes the stored Jacobian entries and their error estimates at `x`.
    ///
    /// The callback must write all `m` rows, including rows without stored
    /// entries; its values may be real or complex. `bounds` restrict every
    /// sample point. The evaluation runs with
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native)
    /// and allocates a workspace and the result.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `bounds` does not have one entry per
    ///   column of the pattern.
    /// - `NonFinite` (origin `Input`) if `x` contains NaN or infinity.
    /// - `InvalidInput` (`x`, `OutOfRange`) if `x` lies outside `bounds`.
    /// - `DifferentiationFailure` whose `source()` is the error returned by
    ///   `function`, `NonFinite` (origin `Callback`) for a NaN or infinite value
    ///   of `function`, or `NumericalInstability` for a sample point or sum
    ///   outside the finite range; it records the sample location and the
    ///   evaluation count.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    pub fn evaluate<T, F>(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
    ) -> Result<DifferentiationResult<T>, SciOxideError>
    where
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    {
        self.validate_inputs(x, bounds)?;
        let mut workspace = self.workspace(())?;
        let mut values = try_filled(
            self.output_len(),
            T::ZERO,
            scioxide_core::quantity!("derivative_output"),
        )?;
        let mut errors = try_filled(
            self.output_len(),
            S::ZERO,
            scioxide_core::quantity!("error_output"),
        )?;
        let report = self.evaluate_validated(
            x,
            bounds,
            function,
            &mut values,
            &mut errors,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(DifferentiationResult::from_estimates(
            report, values, errors,
        ))
    }
    /// Computes the stored Jacobian entries at `x` into `out` and their error
    /// estimates into `errors`, using `workspace` for scratch.
    ///
    /// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) or a
    /// child [`SolveScope`](scioxide_core::SolveScope). With a sufficient
    /// workspace the evaluation does not allocate. `out` and `errors` are
    /// written together, only if the report has output available; the
    /// workspace stays usable after an error.
    ///
    /// # Errors
    ///
    /// - The errors of [`evaluate`](Self::evaluate), except allocation.
    /// - `DimensionMismatch` if `out` or `errors` does not have
    ///   [`output_len`](Self::output_len) entries, or `workspace` is too small.
    /// - `ResourceLimit` if the workspace exceeds `max_workspace_bytes`.
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    pub fn evaluate_into<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        function: F,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        out: &mut [T],
        errors: &mut [S],
        workspace: &mut ColoredJacobianWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        self.validate_inputs(x, bounds)?;
        self.evaluate_validated(x, bounds, function, out, errors, workspace, scope)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "validated problem data, caller buffers and execution stay explicit"
    )]
    fn evaluate_validated<
        'scope,
        'capability,
        T: NumericScalar<Real = S>,
        F: SampleFunction<S, T>,
    >(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        mut function: F,
        output: &mut [T],
        errors: &mut [S],
        workspace: &mut ColoredJacobianWorkspace<T>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        workspace.prepare(&self.workspace_requirements(())?)?;
        self.validate_workspace(output.len(), errors.len(), &workspace.data)?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("differentiate", "colored-jacobian", 1),
            self.output_len() as u128,
            0,
        );
        scope
            .into()
            .run(&mut workspace.state, identity, self.budgets, |context| {
                self.evaluate_samples(
                    x,
                    bounds,
                    &mut function,
                    output,
                    errors,
                    &mut workspace.data,
                    context,
                )
            })
    }
    fn validate_workspace<T: NumericScalar<Real = S>>(
        &self,
        output: usize,
        error_output: usize,
        w: &ColoredBuffers<T>,
    ) -> Result<(), SciOxideError> {
        super::length(self.output_len(), output)?;
        super::length(self.output_len(), error_output)?;
        let n = self.columns.len();
        let m = self.schedule.pattern().rows();
        if w.columns.len() < n
            || w.base.len() != m
            || w.values.len() != self.output_len()
            || self.columns.iter().zip(&w.columns).any(|(p, c)| {
                c.workspace.outputs != p.rows.len() || c.workspace.levels != self.levels
            })
        {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "colored workspace state does not match plan",
            ));
        }
        Ok(())
    }
    fn validate_inputs(&self, x: &[S], bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        super::length(self.columns.len(), x.len())?;
        validate_problem(x, bounds, scioxide_core::argument!("x"))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "problem data, caller buffers and execution resources remain explicit"
    )]
    fn evaluate_samples<T: NumericScalar<Real = S>, F: SampleFunction<S, T>>(
        &self,
        x: &[S],
        bounds: Option<&Bounds<S>>,
        function: &mut F,
        output: &mut [T],
        error_output: &mut [S],
        w: &mut ColoredBuffers<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<DifferentiationReport, SciOxideError> {
        // The public boundary validates inputs, outputs and workspace before the solve.
        let n = self.columns.len();
        let m = self.schedule.pattern().rows();
        let mut report = DifferentiationReport {
            termination: DifferentiationTermination::Converged,
            source: DerivativeSource::ColoredDifference,
            evaluations: 0,
            iterations: 0,
            completed_entries: 0,
            output_len: self.output_len(),
        };
        if self.output_len() == 0 {
            if let Some(reason) = stop(context)? {
                report.termination = reason;
            }
            return Ok(report);
        }
        if let Some(reason) = function.evaluate(
            x,
            &mut w.base,
            &[true],
            n,
            m,
            context,
            &mut report,
            DifferentiationLocation::base_point(),
            self.execution,
        )? {
            report.termination = reason;
            return Ok(report);
        }
        for (col, (column, work)) in self.columns.iter().zip(&mut w.columns).enumerate() {
            work.progress = RefinementProgress::new();
            work.stencil = None;
            work.step = initial_step(x[col], col, &self.step);
            work.termination = column
                .rows
                .is_empty()
                .then_some(DifferentiationTermination::Converged);
            for (base, &row) in work.workspace.base.iter_mut().zip(&column.rows) {
                *base = w.base[row];
            }
        }
        // Columns in one color group share no stored rows, so a single perturbed
        // point can move all of them at once: each row's change comes from its
        // unique column in the group. Every column keeps its own step, stencil
        // geometry and tableau; the two samples pack each column's nonzero offsets,
        // which is valid even when their geometries differ, because no column
        // reads rows owned by another. Finished columns stay at x in later levels.
        'colors: for group in self.schedule.groups() {
            for _ in 0..self.levels {
                if group.iter().all(|&c| w.columns[c].termination.is_some()) {
                    break;
                }
                if let Some(reason) = stop(context)? {
                    report.termination = reason;
                    break 'colors;
                }
                for &col in group {
                    let work = &mut w.columns[col];
                    work.stencil = None;
                    if work.termination.is_some() {
                        continue;
                    }
                    if !context.record_iteration()?.is_granted() {
                        report.termination = DifferentiationTermination::IterationLimit;
                        break 'colors;
                    }
                    report.iterations = checked_add(
                        report.iterations,
                        1,
                        scioxide_core::quantity!("differentiation_iterations"),
                    )?;
                    let stencil = adaptive_axis(x[col], work.step, 1, bound_for(bounds, col))?;
                    if let Some(a) = stencil
                        && let Some(contraction) = work.progress.prepare(a, None, self.ratio)
                    {
                        work.stencil = Some(a);
                        work.contraction = contraction;
                    } else {
                        work.termination = Some(DifferentiationTermination::StepSizeLimit);
                    }
                }
                if group.iter().all(|&c| w.columns[c].stencil.is_none()) {
                    break;
                }
                for sample in 0..2 {
                    let point = &mut w.points[sample * n..(sample + 1) * n];
                    point.copy_from_slice(x);
                    for &col in group {
                        if let Some(a) = w.columns[col].stencil {
                            let slot = if a.offsets[0] == S::ZERO {
                                sample + 1
                            } else {
                                sample * 2
                            };
                            point[col] = x[col] + a.offsets[slot];
                        }
                    }
                }
                if let Some(reason) = function.evaluate(
                    &w.points,
                    &mut w.samples,
                    &[true; 2],
                    n,
                    m,
                    context,
                    &mut report,
                    DifferentiationLocation::variable(group[0]),
                    self.execution,
                )? {
                    report.termination = reason;
                    break 'colors;
                }
                for &col in group {
                    let work = &mut w.columns[col];
                    let Some(a) = work.stencil else {
                        continue;
                    };
                    let column = &self.columns[col];
                    let lanes = column.rows.len();
                    // Gather this column's rows into the dense lane layout that the shared
                    // tableau update expects.
                    for sample in 0_usize..3 {
                        for (lane, &row) in column.rows.iter().enumerate() {
                            let slot = if a.offsets[0] == S::ZERO {
                                sample.checked_sub(1)
                            } else {
                                match sample {
                                    0 => Some(0),
                                    2 => Some(1),
                                    _ => None,
                                }
                            };
                            work.workspace.samples[sample * lanes + lane] =
                                slot.map_or(w.base[row], |slot| w.samples[slot * m + row]);
                        }
                    }
                    let (satisfied, improved) = column.plan().update_table(
                        0,
                        0,
                        a,
                        None,
                        work.progress.depth,
                        work.contraction,
                        work.progress.available,
                        &mut work.workspace,
                    )?;
                    work.termination = work.progress.observe(satisfied, improved);
                    work.step = a.scale
                        / if a.offsets[0] == S::ZERO {
                            S::try_from_usize_exact(a.len - 1)?
                        } else {
                            S::ONE
                        }
                        / self.ratio;
                }
            }
        }
        // Assemble only complete estimates, in canonical stored-entry order.
        for (column, work) in self.columns.iter().zip(&w.columns) {
            if work.progress.available {
                for (lane, &entry) in column.entries.iter().enumerate() {
                    w.values[entry] = work.workspace.best[lane];
                    w.errors[entry] = work.workspace.best_error[lane];
                    report.completed_entries += 1;
                }
            }
            // A column that never terminated used up its levels. The first
            // nonconverged column decides the report status.
            if report.termination == DifferentiationTermination::Converged {
                report.termination = work
                    .termination
                    .unwrap_or(DifferentiationTermination::RefinementLimit);
            }
        }
        if let Some(reason) = stop(context)? {
            report.termination = reason;
        }
        if report.output_available() {
            output.copy_from_slice(&w.values);
            error_output.copy_from_slice(&w.errors);
        }
        Ok(report)
    }
}

/// Workspace size of a [`ColoredJacobianPlan`] for callback values of type `T`,
/// returned by [`ColoredJacobianPlan::workspace_requirements`]. It borrows the
/// plan.
#[derive(Debug)]
pub struct ColoredJacobianRequirements<'plan, 'schedule, T: NumericScalar> {
    plan: &'plan ColoredJacobianPlan<'schedule, T::Real>,
    bytes: usize,
}
impl<T: NumericScalar> ColoredJacobianRequirements<'_, '_, T> {
    /// Returns the heap size in bytes of a new workspace for the plan.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}
/// Scratch storage for a [`ColoredJacobianPlan`] with callback values of type `T`.
///
/// It holds `O(n + m + nnz·L)` elements for `n` columns, `m` rows, `nnz`
/// stored entries and `L` levels. Create it with `plan.workspace(())`, or with
/// `Default` followed by [`try_reserve`](Self::try_reserve). Evaluation does
/// not grow it.
#[derive(Debug, Clone)]
pub struct ColoredJacobianWorkspace<T: NumericScalar> {
    data: ColoredBuffers<T>,
    state: scioxide_core::SolveState,
}
impl<T: NumericScalar> Default for ColoredJacobianWorkspace<T> {
    fn default() -> Self {
        Self {
            data: ColoredBuffers {
                columns: Vec::new(),
                points: Vec::new(),
                samples: Vec::new(),
                base: Vec::new(),
                values: Vec::new(),
                errors: Vec::new(),
            },
            state: scioxide_core::SolveState::new(),
        }
    }
}
impl<T: NumericScalar> ColoredJacobianWorkspace<T> {
    /// Returns the heap size in bytes of the buffers the workspace holds.
    ///
    /// Includes retained capacity from earlier requests and nested buffers.
    /// Performs no allocation.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.data.bytes()
    }

    /// Reports whether the workspace is large enough for `required`, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace holds more than the plan's
    ///   `max_workspace_bytes`.
    pub fn check(
        &self,
        required: &ColoredJacobianRequirements<'_, '_, T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let plan = required.plan;
        let retained = self.data.bytes()?;
        workspace::check(retained, plan.max_workspace_bytes, || {
            let (points, samples, base, values) = plan.lengths()?;
            let enough = self.data.columns.len() >= plan.columns.len()
                && self.data.points.capacity() >= points
                && self.data.samples.capacity() >= samples
                && self.data.base.capacity() >= base
                && self.data.values.capacity() >= values
                && self.data.errors.capacity() >= values;
            if !enough {
                return Ok(false);
            }
            for (column, work) in plan.columns.iter().zip(&self.data.columns) {
                if work.workspace.check(&column.plan().requirements()?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Ok(false);
                }
            }
            Ok(true)
        })
    }

    /// Grows the workspace to meet `required`.
    ///
    /// The byte limit is checked before any buffer grows. If an allocation
    /// fails, buffers grown before it are kept; call [`check`](Self::check)
    /// again before evaluating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the workspace would exceed `max_workspace_bytes`.
    /// - `AllocationFailure` if a buffer cannot grow.
    pub fn try_reserve(
        &mut self,
        required: &ColoredJacobianRequirements<'_, '_, T>,
    ) -> Result<(), SciOxideError> {
        let plan = required.plan;
        let projected = self.data.projected_bytes(plan)?;
        workspace::reserve(
            &mut self.data,
            projected,
            plan.max_workspace_bytes,
            |data| {
                let extra = plan.columns.len().saturating_sub(data.columns.len());
                scioxide_core::backend::try_reserve(
                    &mut data.columns,
                    extra,
                    scioxide_core::quantity!("colored_columns"),
                )?;
                while data.columns.len() < plan.columns.len() {
                    data.columns.push(ColumnWork {
                        workspace: AdaptiveBuffers::empty(),
                        progress: RefinementProgress::new(),
                        stencil: None,
                        contraction: T::Real::ONE,
                        step: T::Real::ZERO,
                        termination: None,
                    });
                }
                for (column, work) in plan.columns.iter().zip(&mut data.columns) {
                    work.workspace.reserve(&column.plan().requirements()?)?;
                }
                let (points, samples, base, values) = plan.lengths()?;
                workspace::reserve_buffer(
                    &mut data.points,
                    points,
                    T::Real::ZERO,
                    scioxide_core::quantity!("workspace_buffer"),
                )?;
                workspace::reserve_buffer(
                    &mut data.samples,
                    samples,
                    T::ZERO,
                    scioxide_core::quantity!("workspace_buffer"),
                )?;
                workspace::reserve_buffer(
                    &mut data.base,
                    base,
                    T::ZERO,
                    scioxide_core::quantity!("workspace_buffer"),
                )?;
                workspace::reserve_buffer(
                    &mut data.values,
                    values,
                    T::ZERO,
                    scioxide_core::quantity!("workspace_buffer"),
                )?;
                workspace::reserve_buffer(
                    &mut data.errors,
                    values,
                    T::Real::ZERO,
                    scioxide_core::quantity!("workspace_buffer"),
                )?;
                Ok(())
            },
            |data| data.bytes(),
        )?;
        self.prepare(required)
    }

    fn prepare(
        &mut self,
        required: &ColoredJacobianRequirements<'_, '_, T>,
    ) -> Result<(), SciOxideError> {
        workspace::prepare(self.check(required)?, required.bytes, || self.data.bytes())?;
        let plan = required.plan;
        let (points, samples, base, values) = plan.lengths()?;
        self.data.points.resize(points, T::Real::ZERO);
        self.data.samples.resize(samples, T::ZERO);
        self.data.base.resize(base, T::ZERO);
        self.data.values.resize(values, T::ZERO);
        self.data.errors.resize(values, T::Real::ZERO);
        for (column, work) in plan.columns.iter().zip(&mut self.data.columns) {
            work.workspace.configure(&column.plan().requirements()?)?;
        }
        Ok(())
    }
}
impl<T: NumericScalar> ColoredBuffers<T> {
    fn bytes(&self) -> Result<usize, SciOxideError> {
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<ColumnWork<T>>(
                    [(self.columns.capacity(), 0)],
                    workspace::BYTES,
                )?,
                self.shared_bytes(None)?,
                workspace::try_sum_bytes(
                    self.columns
                        .iter()
                        .map(|col| col.workspace.retained_bytes()),
                    workspace::BYTES,
                )?,
            ],
            workspace::BYTES,
        )
    }
    fn projected_bytes(
        &self,
        plan: &ColoredJacobianPlan<'_, T::Real>,
    ) -> Result<usize, SciOxideError> {
        let shared = self.shared_bytes(Some(plan.lengths()?));
        let children = plan
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| {
                let required = col.plan().requirements::<T>()?;
                self.columns.get(i).map_or(Ok(required.bytes), |w| {
                    w.workspace.projected_bytes(&required)
                })
            })
            .chain(
                self.columns
                    .iter()
                    .skip(plan.columns.len())
                    .map(|col| col.workspace.retained_bytes()),
            );
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<ColumnWork<T>>(
                    [(self.columns.capacity(), plan.columns.len())],
                    workspace::BYTES,
                )?,
                shared?,
                workspace::try_sum_bytes(children, workspace::BYTES)?,
            ],
            workspace::BYTES,
        )
    }
    fn shared_bytes(
        &self,
        lengths: Option<(usize, usize, usize, usize)>,
    ) -> Result<usize, SciOxideError> {
        let (points, samples, base, values) = lengths.unwrap_or((0, 0, 0, 0));
        workspace::sum_bytes(
            [
                workspace::buffer_bytes::<T::Real>(
                    [
                        (self.points.capacity(), points),
                        (self.errors.capacity(), values),
                    ],
                    workspace::BYTES,
                )?,
                workspace::buffer_bytes::<T>(
                    [
                        (self.samples.capacity(), samples),
                        (self.base.capacity(), base),
                        (self.values.capacity(), values),
                    ],
                    workspace::BYTES,
                )?,
            ],
            workspace::BYTES,
        )
    }
}

impl<'schedule, S: RealScalar> ColoredJacobianPlan<'schedule, S> {
    fn lengths(&self) -> Result<(usize, usize, usize, usize), SciOxideError> {
        Ok((
            checked_mul(
                2,
                self.columns.len(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            checked_mul(
                2,
                self.schedule.pattern().rows(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            self.schedule.pattern().rows(),
            self.output_len(),
        ))
    }
    /// Returns the workspace size this plan needs for callback values of type
    /// `T`, without allocating.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds `max_workspace_bytes`.
    pub fn workspace_requirements<T: NumericScalar<Real = S>>(
        &self,
        (): (),
    ) -> Result<ColoredJacobianRequirements<'_, 'schedule, T>, SciOxideError> {
        let empty = ColoredJacobianWorkspace::<T>::default();
        let bytes = empty.data.projected_bytes(self)?;
        workspace::memory_limit(bytes, self.max_workspace_bytes)?;
        Ok(ColoredJacobianRequirements { plan: self, bytes })
    }

    /// Allocates a workspace for callback values of type `T`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if a buffer cannot be allocated.
    pub fn workspace<T: NumericScalar<Real = S>>(
        &self,
        request: (),
    ) -> Result<ColoredJacobianWorkspace<T>, SciOxideError> {
        let () = request;
        let required = self.workspace_requirements(())?;
        let mut work = ColoredJacobianWorkspace::default();
        work.try_reserve(&required)?;
        Ok(work)
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use alloc::vec;
    use scioxide_core::{Capacity, Complex64, SciOxideErrorKind, StructuralPattern};

    #[test]
    fn colored_shared_and_child_buffers_require_each_slot() {
        let schedule = ColoringSchedule::new(
            StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap(),
            ColoringKind::DerivativeColumns,
            vec![0, 0],
        )
        .unwrap();
        let plan =
            ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default()).unwrap();
        let r = plan.workspace_requirements::<Complex64>(()).unwrap();
        let (points, samples, base, values) = plan.lengths().unwrap();
        for i in 0..7 {
            let mut work = plan.workspace::<Complex64>(()).unwrap();
            work.data.points.reserve(1024);
            work.data.values.reserve(1024);
            match i {
                0 => work.data.points = Vec::with_capacity(points - 1),
                1 => work.data.samples = Vec::with_capacity(samples - 1),
                2 => work.data.base = Vec::with_capacity(base - 1),
                3 => work.data.values = Vec::with_capacity(values - 1),
                4 => work.data.errors = Vec::with_capacity(values - 1),
                5 => {
                    work.data.columns.pop();
                }
                _ => {
                    let child = plan.columns[0].plan().requirements::<Complex64>().unwrap();
                    work.data.columns[0].workspace.table = Vec::with_capacity(child.values[2] - 1);
                }
            }
            assert!(work.data.bytes().unwrap() > r.bytes());
            assert_eq!(work.check(&r).unwrap(), Capacity::NeedsGrowth);
            assert_eq!(
                work.prepare(&r).unwrap_err().kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        }
    }
}
