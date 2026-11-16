//! Explicit sparse matrix powers and matrix functions with fixed output patterns.
//!
//! Plans fix every pattern when they are created: the input pattern (sorted,
//! without duplicates), symbolic products, inverse and connected-block
//! patterns, and the output pattern. Evaluation scatters values into a
//! [`SparseMaterializationWorkspace`] and writes only output values, so
//! explicit zeros are kept. Entry and byte limits are `ResourceLimit` errors
//! raised before any numerical work.
use crate::functions::{
    FunctionControl, MatrixFunctionWorkspaceRequest, PreparedRef, Scratch, ValidatedFunction,
    candidate_report, complexify_schur, condition_candidate, evaluate_candidate, real_defect,
    real_eigenvalues, scratch_lengths,
};
use crate::workspace::take_mut;
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    Estimation, LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace, MatrixFunction, MatrixFunctionOptions, MatrixView, SchurOptions,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::size_of;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Capacity, Complex, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, Scaled,
    SciOxideError, SolveContext, StructuralPattern,
};
use scioxide_sparse::{
    CsrMatrix, ExplicitZeroPolicy, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseLuWorkspace,
    SparseOperation, SparseProductOptions, SparseProductPlan, SparseProductPlanOptions,
    SparseProductRequirements, SparseProductWorkspace, SparseView,
};

/// Sparse product arithmetic and dispatch of the family's execution options;
/// the policy bounds product workers.
fn product_options(execution: LinalgExecutionOptions) -> SparseProductOptions {
    SparseProductOptions::new()
        .with_accuracy(execution.accuracy())
        .with_execution_policy(execution.execution_policy())
}

fn fill_limit(required: usize, limit: usize) -> SciOxideError {
    SciOxideError::resource_limit(scioxide_core::quantity!("sparse_entries"), required, limit)
}
fn overflow() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!("sparse_workspace_bytes"))
}
/// `total + count * width` with checked arithmetic.
fn tally(total: usize, count: usize, width: usize) -> Result<usize, SciOxideError> {
    count
        .checked_mul(width)
        .and_then(|bytes| total.checked_add(bytes))
        .ok_or_else(overflow)
}
/// Payload of a sorted, duplicate-free CSR matrix: row pointers, column indices, values and
/// the cached column traversal (row indices, column pointers, column order).
fn csr_bytes<S: NumericScalar>(
    rows: usize,
    cols: usize,
    entries: usize,
) -> Result<usize, SciOxideError> {
    let pointers = rows
        .checked_add(cols)
        .and_then(|sum| sum.checked_add(2))
        .ok_or_else(overflow)?;
    let pointer_bytes = tally(0, pointers, size_of::<usize>())?;
    tally(
        pointer_bytes,
        entries,
        3 * size_of::<usize>() + size_of::<S>(),
    )
}

fn admit_fill(entries: usize, limit: usize) -> Result<(), SciOxideError> {
    if entries > limit {
        Err(fill_limit(entries, limit))
    } else {
        Ok(())
    }
}
fn admit_iterations(solve: &mut SolveContext<'_, '_>, amount: usize) -> Result<(), SciOxideError> {
    if solve.record_iterations(amount)?.is_granted() {
        Ok(())
    } else {
        Err(crate::functions::limit(
            "sparse power iteration budget exhausted",
            solve.counts(),
        ))
    }
}

/// Validate a square view and its structural fill, check the converted
/// matrix's storage against the byte limit, and copy it to sorted,
/// duplicate-free CSR with explicit zeros kept. The conversion emits that
/// order directly, so the converted matrix is its only storage.
fn canonical<S: NumericScalar>(
    input: SparseView<'_, S>,
    max_entries: usize,
    limit: Option<usize>,
) -> Result<CsrMatrix<S>, SciOxideError> {
    if input.rows() != input.cols() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(input.rows(), input.rows()),
            scioxide_core::Dimensions::two(input.rows(), input.cols()),
        ));
    }
    let count = input.iter().count();
    admit_fill(count, max_entries)?;
    crate::workspace::admit_bytes(
        csr_bytes::<S>(input.rows(), input.cols(), count)?,
        limit,
        scioxide_core::quantity!("sparse_workspace_bytes"),
    )?;
    input.to_csr(ExplicitZeroPolicy::Preserve)
}

fn owned_pattern<S: NumericScalar>(
    matrix: &CsrMatrix<S>,
) -> Result<StructuralPattern, SciOxideError> {
    matrix.structural_pattern().to_owned()
}

/// Allocate a CSR matrix with `pattern` and zero values.
fn csr_from_pattern<T: NumericScalar>(
    pattern: &StructuralPattern,
) -> Result<CsrMatrix<T>, SciOxideError> {
    let view = pattern.view();
    CsrMatrix::from_pattern(
        view.to_owned()?,
        try_filled(
            view.col_indices().len(),
            T::ZERO,
            scioxide_core::quantity!("values"),
        )?,
    )
}

/// Reject output whose shape or pattern differs from `pattern`.
fn check_output<T: NumericScalar>(
    output: &CsrMatrix<T>,
    pattern: &StructuralPattern,
) -> Result<(), SciOxideError> {
    let expected = pattern.view();
    if (output.rows(), output.cols()) != (expected.rows(), expected.cols()) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(expected.rows(), expected.cols()),
            scioxide_core::Dimensions::two(output.rows(), output.cols()),
        ));
    }
    if output.structural_pattern() != expected {
        return Err(crate::checked::inconsistent(
            scioxide_core::argument!("output"),
            "output pattern differs from the plan; use create_output",
        ));
    }
    Ok(())
}

/// Scatter a borrowed input into the prepared sorted pattern without
/// allocation, rejecting pattern changes.
fn scatter<S: NumericScalar>(
    input: SparseView<'_, S>,
    pattern: &StructuralPattern,
    values: &mut [S],
    seen: &mut [bool],
) -> Result<(), SciOxideError> {
    let pattern = pattern.view();
    if (input.rows(), input.cols()) != (pattern.rows(), pattern.cols()) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(pattern.rows(), pattern.cols()),
            scioxide_core::Dimensions::two(input.rows(), input.cols()),
        ));
    }
    let changed = || {
        crate::checked::inconsistent(
            scioxide_core::argument!("input"),
            "sparse input pattern differs from the prepared pattern",
        )
    };
    let entries = pattern.col_indices().len();
    let (values, seen) = (&mut values[..entries], &mut seen[..entries]);
    values.fill(S::ZERO);
    seen.fill(false);
    for (row, col, value) in input.iter() {
        let start = pattern.row_ptr()[row];
        let offset = pattern.col_indices()[start..pattern.row_ptr()[row + 1]]
            .binary_search(&col)
            .map_err(|_| changed())?;
        let position = start + offset;
        // Duplicate storage sums, as canonical conversion does.
        values[position] = if seen[position] {
            values[position] + value
        } else {
            value
        };
        seen[position] = true;
    }
    if seen.iter().all(|&seen| seen) {
        Ok(())
    } else {
        Err(changed())
    }
}

/// Connected components of the symmetrized pattern. Members of each block are
/// ascending, so a block's local order is also its sorted column order.
/// Components come from scioxide-sparse; full-block fill is checked against the
/// entry limit before the output pattern is allocated.
#[derive(Debug, Clone)]
struct ConnectedBlockLayout {
    offsets: Vec<usize>,
    members: Vec<usize>,
    group: Vec<usize>,
    local: Vec<usize>,
}
impl ConnectedBlockLayout {
    fn new<S: NumericScalar>(input: &CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let n = input.rows();
        let components = scioxide_sparse::GraphPlan::from_pattern(
            input.structural_pattern(),
            scioxide_sparse::GraphDirection::Directed,
        )?
        .weak_components()?;
        let mut group = try_with_capacity(n, scioxide_core::quantity!("group"))?;
        group.extend_from_slice(components.labels());
        let count = components.count();
        let mut offsets = try_filled(count + 1, 0usize, scioxide_core::quantity!("block_offsets"))?;
        for &g in &group {
            offsets[g + 1] += 1;
        }
        for g in 0..count {
            offsets[g + 1] += offsets[g];
        }
        let mut cursors = try_with_capacity(count, scioxide_core::quantity!("cursors"))?;
        cursors.extend_from_slice(&offsets[..count]);
        let mut members = try_filled(n, 0usize, scioxide_core::quantity!("members"))?;
        let mut local = try_filled(n, 0usize, scioxide_core::quantity!("local"))?;
        for (vertex, &g) in group.iter().enumerate() {
            members[cursors[g]] = vertex;
            local[vertex] = cursors[g] - offsets[g];
            cursors[g] += 1;
        }
        Ok(Self {
            offsets,
            members,
            group,
            local,
        })
    }
    fn iter(&self) -> impl Iterator<Item = &[usize]> {
        self.offsets
            .windows(2)
            .map(|bounds| &self.members[bounds[0]..bounds[1]])
    }
    /// Total full-block fill and largest block order.
    fn fill(&self) -> Result<(usize, usize), SciOxideError> {
        self.iter().try_fold((0usize, 0usize), |(sum, max), block| {
            let square = block
                .len()
                .checked_mul(block.len())
                .and_then(|square| sum.checked_add(square))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("block_fill"))
                })?;
            Ok((square, max.max(block.len())))
        })
    }
    /// Block-diagonal pattern with every connected block full.
    fn pattern(&self, entries: usize) -> Result<StructuralPattern, SciOxideError> {
        let n = self.group.len();
        let mut pointers =
            try_with_capacity(n + 1, scioxide_core::quantity!("block_row_pointers"))?;
        let mut columns =
            try_with_capacity(entries, scioxide_core::quantity!("block_column_indices"))?;
        pointers.push(0);
        for &g in &self.group {
            let block = &self.members[self.offsets[g]..self.offsets[g + 1]];
            columns.extend_from_slice(block);
            pointers.push(columns.len());
        }
        StructuralPattern::new(n, n, pointers, columns)
    }
}

/// Options of [`SparsePowerPlan`]: entry limit and execution options.
///
/// Defaults: no limit on stored entries or bytes, `Auto` execution, and no
/// iteration or time limit. The byte limit bounds the whole plan workspace,
/// including intermediate products and the sparse LU of a negative power. That
/// LU may store at most as many entries as the full connected blocks of `A⁻¹`.
/// Sparse products use one worker for `Sequential`, at most 64 for `Auto` and
/// `min(n, 64)` for `Parallel(n)`; the result does not depend on the number of
/// workers.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::SparsePowerOptions;
/// let options = SparsePowerOptions::new()
///     .with_max_entries(64)
///     .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(1 << 20));
/// assert_eq!(options.max_entries(), 64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SparsePowerOptions {
    execution: LinalgExecutionOptions,
    max_entries: usize,
}
impl Default for SparsePowerOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SparsePowerOptions {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    ///
    /// Toward the iteration limit, each sparse product and each LU column
    /// solve counts as one iteration, and the LU factorization as `n`.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the default options.
    pub const fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            max_entries: usize::MAX,
        }
    }
    /// Sets the maximum number of stored entries of the input and of every
    /// intermediate or final matrix (default unlimited).
    pub const fn with_max_entries(mut self, entries: usize) -> Self {
        self.max_entries = entries;
        self
    }
    /// Returns the maximum number of stored entries.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }
}

/// Options of [`SparseFunctionPlan`]: entry limit, function settings and error estimate.
///
/// Defaults: no limit on stored entries or bytes, `Auto` execution, default
/// [`MatrixFunctionSettings`](crate::MatrixFunctionSettings) and
/// [`Estimation::Skip`]. The function settings give the tolerance and series
/// limit of the dense block evaluations. All blocks share the iteration, time
/// and byte limits of the execution options.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Estimation, SparseFunctionOptions};
/// let options = SparseFunctionOptions::<f64>::new().with_error_estimate(Estimation::Estimate);
/// assert_eq!(options.error_estimate(), Estimation::Estimate);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SparseFunctionOptions<R: RealScalar> {
    execution: LinalgExecutionOptions,
    max_entries: usize,
    function: crate::MatrixFunctionSettings<R>,
    error_estimate: Estimation,
}
impl<R: RealScalar> Default for SparseFunctionOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> SparseFunctionOptions<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    ///
    /// The iteration limit bounds the work units of the dense block
    /// evaluations, summed over all blocks (default 1 000 000).
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            execution: LinalgExecutionOptions::new(),
            max_entries: usize::MAX,
            function: crate::MatrixFunctionSettings::new(),
            error_estimate: Estimation::Skip,
        }
    }
    /// Sets the maximum number of stored entries of the input and of the
    /// full-block output (default unlimited).
    pub const fn with_max_entries(mut self, entries: usize) -> Self {
        self.max_entries = entries;
        self
    }
    /// Sets the tolerance and series limit of the dense block evaluations.
    pub const fn with_function_settings(
        mut self,
        options: crate::MatrixFunctionSettings<R>,
    ) -> Self {
        self.function = options;
        self
    }
    /// Sets whether to compute a first-order forward-error estimate (default `Skip`).
    ///
    /// The estimate uses Fréchet condition numbers of the blocks; see
    /// [`SparseMaterializationReport::estimated_error`]. Only built-in
    /// functions support it.
    pub const fn with_error_estimate(mut self, estimation: Estimation) -> Self {
        self.error_estimate = estimation;
        self
    }
    /// Returns the maximum number of stored entries.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }
    /// Returns the settings of the dense block evaluations.
    pub const fn function_settings(self) -> crate::MatrixFunctionSettings<R> {
        self.function
    }
    /// Returns whether the forward-error estimate is computed.
    pub const fn error_estimate(self) -> Estimation {
        self.error_estimate
    }
}

/// Diagnostics of a sparse power or matrix function evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SparseMaterializationReport<R: RealScalar> {
    /// Stored entries of the prepared output pattern.
    pub(crate) structural_fill: usize,
    /// Written output values that are not exactly zero, after real projection.
    pub(crate) nonzero_fill: usize,
    /// Largest prepared structural matrix, including the input.
    pub(crate) peak_intermediate_fill: usize,
    /// Plan workspace bytes checked against the byte limit.
    pub(crate) admitted_workspace_bytes: usize,
    /// Dense function work, or sparse products plus LU columns and solves.
    pub(crate) work: usize,
    /// Optional first-order forward-error heuristic, not a certificate.
    /// `None` means no estimate was requested or supplied.
    pub(crate) estimated_error: Option<R>,
}

impl<R: RealScalar> SparseMaterializationReport<R> {
    /// Returns the number of stored entries of the output pattern.
    pub const fn structural_fill(&self) -> usize {
        self.structural_fill
    }

    /// Returns the number of output values that are not exactly zero.
    pub const fn nonzero_fill(&self) -> usize {
        self.nonzero_fill
    }

    /// Returns the largest number of stored entries of any matrix of the plan,
    /// including the input.
    pub const fn peak_intermediate_fill(&self) -> usize {
        self.peak_intermediate_fill
    }

    /// Returns the workspace bytes of the plan, which were checked against the byte limit.
    pub const fn admitted_workspace_bytes(&self) -> usize {
        self.admitted_workspace_bytes
    }

    /// Returns the work count of the evaluation.
    ///
    /// For a power it is the number of sparse products, plus `2n` for a
    /// negative power (the LU factorization and the `n` column solves). For a
    /// function it is the work units of the dense block evaluations.
    pub const fn work(&self) -> usize {
        self.work
    }

    /// Returns the forward-error estimate, or `None` if none was requested.
    ///
    /// It is `Σ_b κ(f, A_b) ‖A_b‖ ε` over the connected blocks `A_b`, where
    /// `κ(f, A_b)` is the absolute condition number of `f` at `A_b`, estimated
    /// from the Fréchet derivative, and `‖A_b‖` is the sum of the absolute
    /// values of its entries. It is a first-order estimate, not a bound.
    pub const fn estimated_error(&self) -> Option<R> {
        self.estimated_error
    }
}

/// Result of an evaluation with new storage: the output matrix and its report.
#[derive(Debug)]
pub struct SparseMaterialization<S: NumericScalar> {
    /// Sparse result in the plan's output pattern.
    pub(crate) value: CsrMatrix<S>,
    /// Fill, work, and error diagnostics.
    pub(crate) report: SparseMaterializationReport<S::Real>,
}

impl<S: NumericScalar> SparseMaterialization<S> {
    /// Returns the output matrix and the report.
    pub fn into_parts(self) -> (CsrMatrix<S>, SparseMaterializationReport<S::Real>) {
        (self.value, self.report)
    }

    /// Returns the output matrix, which has the plan's output pattern.
    pub const fn value(&self) -> &CsrMatrix<S> {
        &self.value
    }

    /// Returns the report.
    pub const fn report(&self) -> SparseMaterializationReport<S::Real> {
        self.report
    }
}

/// Storage plan shared by the sparse plans and their workspace.
#[derive(Debug, Clone)]
struct Layout {
    /// CSR buffers; the first is the input.
    patterns: Vec<StructuralPattern>,
    /// Product scratch values.
    scratch: SparseProductRequirements,
    /// Complex output candidate values of a function plan.
    candidate: usize,
    /// LU of the input, for negative powers.
    lu: Option<SparseLuLimits>,
    /// Dense block arenas.
    dense: LinalgRequirements,
}
impl Layout {
    fn input_entries(&self) -> usize {
        self.patterns
            .first()
            .map_or(0, |pattern| pattern.view().col_indices().len())
    }
}

/// Workspace storage and byte limit of a [`SparsePowerPlan`] or [`SparseFunctionPlan`].
///
/// The value borrows the plan's patterns. A workspace reserved with it fits
/// that plan's layout only.
#[derive(Debug, Clone, Copy)]
pub struct SparseMaterializationRequirements<'p, S: NumericScalar> {
    layout: &'p Layout,
    bytes: usize,
    byte_limit: Option<usize>,
    scalar: PhantomData<fn() -> S>,
}
impl<S: NumericScalar> SparseMaterializationRequirements<'_, S> {
    /// Returns the workspace bytes, excluding allocator overhead and the output.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the byte limit, if any.
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
}

#[derive(Debug)]
struct LuStorage<S: NumericScalar> {
    factor: Option<SparseLu<S>>,
    workspace: SparseLuWorkspace<S>,
    rhs: Vec<S>,
    column: Vec<S>,
    scratch: Vec<S>,
}
impl<S: NumericScalar> LuStorage<S> {
    fn new(pattern: &StructuralPattern, limits: SparseLuLimits) -> Result<Self, SciOxideError> {
        let n = pattern.view().rows();
        let symbolic = SparseLuSymbolic::analyze(
            pattern.view(),
            scioxide_sparse::SparseOrdering::default(),
            limits,
        )?;
        let workspace = SparseLuWorkspace::prepare(symbolic)?;
        let vector = |len| try_filled(len, S::ZERO, scioxide_core::quantity!("lu_vector"));
        Ok(Self {
            factor: None,
            workspace,
            rhs: vector(n)?,
            column: vector(n)?,
            scratch: vector(n.checked_mul(2).ok_or_else(overflow)?)?,
        })
    }
    fn symbolic(&self) -> Result<&SparseLuSymbolic, SciOxideError> {
        self.factor
            .as_ref()
            .map(SparseLu::symbolic)
            .or_else(|| self.workspace.initial_symbolic())
            .ok_or_else(|| {
                crate::checked::inconsistent(
                    scioxide_core::argument!("workspace"),
                    "missing sparse LU analysis",
                )
            })
    }
    fn matches(&self, pattern: &StructuralPattern, limits: SparseLuLimits) -> bool {
        self.symbolic().is_ok_and(|symbolic| {
            symbolic.pattern() == pattern.view() && symbolic.limits() == limits
        })
    }
}

/// LU payload of the input pattern plus the solve vectors.
fn lu_bytes<S: NumericScalar>(
    input: &CsrMatrix<S>,
    limits: SparseLuLimits,
) -> Result<usize, SciOxideError> {
    let symbolic = SparseLuSymbolic::analyze(
        input.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        limits,
    )?;
    tally(
        symbolic.workspace_requirements::<S>(())?.total_bytes(),
        input.rows().checked_mul(4).ok_or_else(overflow)?,
        size_of::<S>(),
    )
}

/// Workspace of [`SparsePowerPlan`] and [`SparseFunctionPlan`].
///
/// It holds the input values in the plan's pattern, the intermediate CSR
/// matrices, product scratch, the sparse LU of a negative power, and dense
/// storage for the connected blocks of a function. Its buffers fit the
/// patterns of one plan: [`Self::check`] returns `NeedsGrowth` for any other
/// plan, and [`Self::try_reserve`] then rebuilds them.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{SparseMaterializationWorkspace, SparsePowerOptions, SparsePowerPlan};
/// use scioxide_sparse::CsrMatrix;
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0])?;
/// let plan = SparsePowerPlan::new(a.view(), 2, &SparsePowerOptions::new())?;
/// let requirements = plan.workspace_requirements(())?;
/// let mut workspace = SparseMaterializationWorkspace::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseMaterializationWorkspace<S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    matrices: Vec<CsrMatrix<S>>,
    values: Vec<scioxide_sparse::SparseValueWorkspace<S>>,
    scratch: SparseProductWorkspace<S>,
    seen: Vec<bool>,
    candidate: Vec<Complex<S::Real>>,
    lu: Option<LuStorage<S>>,
    dense: LinalgWorkspace<'static, Complex<S::Real>>,
}
impl<S: NumericScalar> Default for SparseMaterializationWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> SparseMaterializationWorkspace<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Creates an empty workspace.
    pub const fn new() -> Self {
        Self {
            matrices: Vec::new(),
            values: Vec::new(),
            scratch: SparseProductWorkspace::new(),
            seen: Vec::new(),
            candidate: Vec::new(),
            lu: None,
            dense: LinalgWorkspace::new(),
        }
    }

    /// Returns whether the workspace fits the plan layout of `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the required bytes exceed the byte limit, or if the
    ///   storage the workspace holds (for the same layout) or would hold after
    ///   [`Self::try_reserve`] (for another layout) exceeds it.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn check(
        &self,
        requirements: &SparseMaterializationRequirements<'_, S>,
    ) -> Result<Capacity, SciOxideError> {
        crate::workspace::admit_bytes(
            requirements.bytes,
            requirements.byte_limit,
            scioxide_core::quantity!("sparse_workspace_bytes"),
        )?;
        let layout = requirements.layout;
        let (capacity, held) = if self.matches(layout)? {
            (Capacity::Sufficient, self.retained_bytes()?)
        } else {
            // Sparse, LU and scratch buffers are replaced, but dense capacity
            // is retained.
            let replaced = requirements
                .bytes
                .checked_sub(layout.dense.required_bytes::<Complex<S::Real>>()?)
                .ok_or_else(overflow)?;
            (
                Capacity::NeedsGrowth,
                replaced
                    .checked_add(self.dense.reservation_bytes(&layout.dense)?)
                    .ok_or_else(overflow)?,
            )
        };
        crate::workspace::admit_bytes(
            held,
            requirements.byte_limit,
            scioxide_core::quantity!("sparse_workspace_bytes"),
        )?;
        Ok(capacity)
    }

    /// Builds the buffers of the plan layout in `requirements`, replacing those of another layout.
    ///
    /// Sparse, LU and scratch buffers are replaced, while dense storage keeps
    /// its capacity. The byte limit is checked for the resulting total before
    /// anything is allocated, so the buffers of a larger previous layout do
    /// not count against it. If an allocation fails, the previous sparse
    /// buffers are kept.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage would exceed the byte limit.
    /// - `AllocationFailure` if an allocation fails.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    /// - Errors of the symbolic analysis of the sparse LU.
    pub fn try_reserve(
        &mut self,
        requirements: &SparseMaterializationRequirements<'_, S>,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let layout = requirements.layout;
        let mut matrices = try_with_capacity(
            layout.patterns.len(),
            scioxide_core::quantity!("sparse_buffers"),
        )?;
        let mut values = try_with_capacity(
            layout.patterns.len(),
            scioxide_core::quantity!("sparse_value_workspaces"),
        )?;
        for pattern in &layout.patterns {
            let matrix = csr_from_pattern(pattern)?;
            values.push(matrix.workspace(())?);
            matrices.push(matrix);
        }
        let mut scratch = SparseProductWorkspace::new();
        scratch.try_reserve(&layout.scratch)?;
        let seen = try_filled(
            layout.input_entries(),
            false,
            scioxide_core::quantity!("input_seen"),
        )?;
        let candidate = try_filled(
            layout.candidate,
            Complex::ZERO,
            scioxide_core::quantity!("output_candidate"),
        )?;
        let lu = match (layout.lu, layout.patterns.first()) {
            (Some(limits), Some(pattern)) => Some(LuStorage::new(pattern, limits)?),
            _ => None,
        };
        self.dense.try_reserve(&layout.dense)?;
        self.matrices = matrices;
        self.values = values;
        self.scratch = scratch;
        self.seen = seen;
        self.candidate = candidate;
        self.lu = lu;
        crate::workspace::admit_bytes(
            self.retained_bytes()?,
            requirements.byte_limit,
            scioxide_core::quantity!("sparse_workspace_bytes"),
        )
    }

    /// Returns the attempt record of the most recent evaluation, or `None` before the first.
    ///
    /// Failures before the numerical stage are recorded as `PreflightRejected`.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.dense.last_attempt()
    }

    fn matches(&self, layout: &Layout) -> Result<bool, SciOxideError> {
        if self.matrices.len() != layout.patterns.len()
            || self.scratch.check(&layout.scratch)? == Capacity::NeedsGrowth
            || self.seen.len() < layout.input_entries()
            || self.candidate.len() < layout.candidate
        {
            return Ok(false);
        }
        for (matrix, pattern) in self.matrices.iter().zip(&layout.patterns) {
            if matrix.structural_pattern() != pattern.view() {
                return Ok(false);
            }
        }
        let lu = match (layout.lu, layout.patterns.first(), &self.lu) {
            (Some(limits), Some(pattern), Some(lu)) => lu.matches(pattern, limits),
            (Some(_), _, _) => false,
            (None, _, _) => true,
        };
        Ok(lu && self.dense.check(&layout.dense)? == Capacity::Sufficient)
    }

    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let mut total = self.dense.retained_bytes()?;
        for matrix in &self.matrices {
            total = total
                .checked_add(csr_bytes::<S>(matrix.rows(), matrix.cols(), matrix.nnz())?)
                .ok_or_else(overflow)?;
        }
        for matrix in &self.matrices {
            total = total
                .checked_add(matrix.retained_value_bytes() - matrix.nnz() * size_of::<S>())
                .ok_or_else(overflow)?;
        }
        for values in &self.values {
            total = total
                .checked_add(values.retained_bytes())
                .ok_or_else(overflow)?;
        }
        total = total
            .checked_add(self.scratch.retained_bytes()?)
            .ok_or_else(overflow)?;
        total = tally(total, self.seen.capacity(), size_of::<bool>())?;
        total = tally(
            total,
            self.candidate.capacity(),
            size_of::<Complex<S::Real>>(),
        )?;
        if let Some(lu) = &self.lu {
            total = total
                .checked_add(
                    lu.symbolic()?
                        .workspace_requirements::<S>(())?
                        .total_bytes(),
                )
                .ok_or_else(overflow)?;
            for vector in [&lu.rhs, &lu.column, &lu.scratch] {
                total = tally(total, vector.capacity(), size_of::<S>())?;
            }
        }
        Ok(total)
    }
}

fn layout_bytes<S: NumericScalar>(layout: &Layout, lu: usize) -> Result<usize, SciOxideError>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let mut total = layout
        .dense
        .required_bytes::<Complex<S::Real>>()?
        .checked_add(lu)
        .ok_or_else(overflow)?;
    for pattern in &layout.patterns {
        let view = pattern.view();
        total = total
            .checked_add(csr_bytes::<S>(
                view.rows(),
                view.cols(),
                view.col_indices().len(),
            )?)
            .ok_or_else(overflow)?;
    }
    for pattern in &layout.patterns {
        total = tally(total, pattern.view().col_indices().len(), size_of::<S>())?;
    }
    total = total
        .checked_add(layout.scratch.required_bytes::<S>()?)
        .ok_or_else(overflow)?;
    total = tally(total, layout.input_entries(), size_of::<bool>())?;
    tally(total, layout.candidate, size_of::<Complex<S::Real>>())
}

fn materialization_identity(n: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "sparse_materialization", 1),
        n as u128,
        0,
    )
}

/// One prepared CSR product `buffer[output] = buffer[left] * buffer[right]`.
#[derive(Debug, Clone)]
struct Step {
    plan: SparseProductPlan,
    left: usize,
    right: usize,
    output: usize,
}

const INPUT: usize = 0;
const AUXILIARY: usize = 1;

/// Plan for the integer power `Aᵏ` of a sparse square matrix, with a fixed output pattern.
///
/// The plan stores patterns, not values, and computes every product pattern when it is created, at
/// the cost of the symbolic sparse products and, for a negative power, of the symbolic sparse LU
/// analysis. A positive power uses binary exponentiation (repeated squaring) with
/// `⌊log₂ k⌋ + popcount(k) − 1` sparse products. `k = 0` gives the identity and `k = 1` a copy of
/// `A`. A negative power factors `A` with sparse LU and computes `A⁻¹` column by column; then
/// `A⁻ᵏ = (A⁻¹)ᵏ`. `A⁻¹` is zero outside the connected blocks of `A` (the weakly connected
/// components of its graph) but can be dense within them, so its pattern holds those blocks in
/// full. Entries that cancel numerically stay in the output as explicit zeros.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{SparsePowerOptions, SparsePowerPlan};
/// use scioxide_sparse::CsrMatrix;
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 1.0, 2.0])?;
/// let plan = SparsePowerPlan::new(a.view(), 2, &SparsePowerOptions::new().with_max_entries(3))?;
/// let square = plan.evaluate(a.view())?;
/// // A = [[1, 1], [0, 2]], so A² = [[1, 3], [0, 4]].
/// assert_eq!(square.value().values(), &[1.0, 3.0, 4.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparsePowerPlan<S: NumericScalar> {
    options: SparsePowerOptions,
    layout: Layout,
    steps: Vec<Step>,
    output: usize,
    identity: bool,
    blocks: Option<ConnectedBlockLayout>,
    peak_fill: usize,
    bytes: usize,
    scalar: PhantomData<fn() -> S>,
}

impl<S: NumericScalar> SparsePowerPlan<S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Creates a plan for `Aᵏ`, `k = exponent`, from the pattern of `input`.
    ///
    /// Only the pattern of `input` is used; duplicate entries are merged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` is not square.
    /// - `ResourceLimit` if the input, an intermediate or the output has more
    ///   stored entries than the entry limit, or the workspace exceeds the
    ///   byte limit.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    /// - Errors of the symbolic sparse LU analysis (negative exponents) and of
    ///   the symbolic sparse products.
    pub fn new(
        input: SparseView<'_, S>,
        exponent: i64,
        options: &SparsePowerOptions,
    ) -> Result<Self, SciOxideError> {
        let limit = options.execution.byte_limit();
        let matrix = canonical(input, options.max_entries, limit)?;
        let n = matrix.rows();
        if exponent == 0 {
            admit_fill(n, options.max_entries)?;
        }
        // Seed the result with its first factor; identity is needed only for A^0.
        let power = exponent.unsigned_abs();
        let products = match power {
            0 => 0,
            _ => (power.count_ones() + u64::BITS - 2 - power.leading_zeros()) as usize,
        };
        let buffers = products + 1 + usize::from(exponent <= 0);
        let quantity = scioxide_core::quantity!("sparse_power_buffers");
        let mut patterns = try_with_capacity(buffers, quantity)?;
        let mut operands = try_with_capacity(buffers, quantity)?;
        let mut steps = try_with_capacity(products, quantity)?;
        patterns.push(owned_pattern(&matrix)?);
        operands.push(matrix);
        if exponent == 0 {
            let identity = CsrMatrix::<S>::identity(n)?;
            patterns.push(owned_pattern(&identity)?);
            operands.push(identity);
        }
        let mut lu = 0;
        let blocks = if exponent < 0 {
            let blocks = ConnectedBlockLayout::new(&operands[INPUT])?;
            let (fill, _) = blocks.fill()?;
            admit_fill(fill, options.max_entries)?;
            let pattern = blocks.pattern(fill)?;
            operands.push(csr_from_pattern(&pattern)?);
            patterns.push(pattern);
            let limits = SparseLuLimits::new(fill, limit.unwrap_or(usize::MAX));
            lu = lu_bytes(&operands[INPUT], limits)?;
            Some((blocks, limits))
        } else {
            None
        };
        let mut base = if blocks.is_some() { AUXILIARY } else { INPUT };
        let mut result = None;
        let mut power = power;
        let mut push = |left: usize, right: usize| -> Result<usize, SciOxideError> {
            let plan = SparseProductPlan::with_options(
                &operands[left],
                &operands[right],
                SparseProductPlanOptions::new().with_max_output_entries(options.max_entries),
            )?;
            let output = plan.create_output::<S>()?;
            patterns.push(owned_pattern(&output)?);
            operands.push(output);
            steps.push(Step {
                plan,
                left,
                right,
                output: operands.len() - 1,
            });
            Ok(operands.len() - 1)
        };
        while power > 0 {
            if power & 1 != 0 {
                result = Some(match result {
                    Some(previous) => push(previous, base)?,
                    None => base,
                });
            }
            power >>= 1;
            if power > 0 {
                base = push(base, base)?;
            }
        }
        let peak_fill = operands.iter().map(CsrMatrix::nnz).max().unwrap_or(0);
        let layout = Layout {
            patterns,
            scratch: steps
                .iter()
                .fold(SparseProductRequirements::default(), |r, step| {
                    r.union(
                        step.plan
                            .workspace_requirements(product_options(options.execution)),
                    )
                }),
            candidate: 0,
            lu: blocks.as_ref().map(|(_, limits)| *limits),
            dense: LinalgRequirements::default(),
        };
        let bytes = layout_bytes::<S>(&layout, lu)?;
        crate::workspace::admit_bytes(
            bytes,
            limit,
            scioxide_core::quantity!("sparse_workspace_bytes"),
        )?;
        Ok(Self {
            options: *options,
            layout,
            steps,
            output: result.unwrap_or(AUXILIARY),
            identity: exponent == 0,
            blocks: blocks.map(|(blocks, _)| blocks),
            peak_fill,
            bytes,
            scalar: PhantomData,
        })
    }

    /// Returns the number of stored entries of the output pattern.
    pub fn structural_fill(&self) -> usize {
        self.layout.patterns[self.output].view().col_indices().len()
    }

    /// Returns the workspace storage of this plan; the request is `()`.
    ///
    /// # Errors
    ///
    /// None; all sizes are fixed when the plan is created. The `Result` keeps
    /// the signature of the other plans.
    pub fn workspace_requirements(
        &self,
        request: (),
    ) -> Result<SparseMaterializationRequirements<'_, S>, SciOxideError> {
        let () = request;
        Ok(SparseMaterializationRequirements {
            layout: &self.layout,
            bytes: self.bytes,
            byte_limit: self.options.execution.byte_limit(),
            scalar: PhantomData,
        })
    }

    /// Allocates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// The errors of [`SparseMaterializationWorkspace::try_reserve`].
    pub fn workspace(
        &self,
        request: (),
    ) -> Result<SparseMaterializationWorkspace<S>, SciOxideError> {
        let mut workspace = SparseMaterializationWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Allocates an output matrix with the output pattern and zero values.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if an allocation fails.
    pub fn create_output(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        csr_from_pattern(&self.layout.patterns[self.output])
    }

    /// Computes `Aᵏ` for the values of `input` with a new output and workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::create_output`] and [`Self::evaluate_into`].
    pub fn evaluate(
        &self,
        input: SparseView<'_, S>,
    ) -> Result<SparseMaterialization<S>, SciOxideError> {
        let mut value = self.create_output()?;
        let mut workspace = SparseMaterializationWorkspace::new();
        let report = self.evaluate_into(
            input,
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(SparseMaterialization { value, report })
    }

    /// Computes `Aᵏ` for the values of `input` and writes it to `output`.
    ///
    /// `input` must have the pattern the plan was created with; duplicate
    /// entries are summed. `output` must have the output pattern (see
    /// [`Self::create_output`]); only its values are written. The workspace
    /// grows on first use, and later calls reuse it, also for the LU
    /// refactorization and solves of a negative power. The iteration limit set
    /// with [`LinalgExecutionOptions::with_iteration_budget`] bounds the
    /// products and LU work, as described at
    /// [`SparsePowerOptions::with_execution_options`].
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has the wrong shape.
    /// - `InvalidInput` (`Inconsistent`) if the pattern of `input` or `output`
    ///   differs from the plan's.
    /// - `NonFinite` if `input` has a non-finite value.
    /// - The errors of [`SparseMaterializationWorkspace::try_reserve`].
    /// - `ConvergenceFailure` if the iteration limit is reached.
    /// - `ExecutionStopped` if cancellation or a deadline stops the evaluation.
    /// - Errors of the sparse LU factorization and solves, such as `Singular`,
    ///   and of the sparse products.
    pub fn evaluate_into<'scope, 'capability>(
        &self,
        input: SparseView<'_, S>,
        output: &mut CsrMatrix<S>,
        workspace: &mut SparseMaterializationWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<SparseMaterializationReport<S::Real>, SciOxideError> {
        workspace.dense.begin_attempt();
        let scope = context.into().with_options(&self.options.execution)?;
        check_output(output, &self.layout.patterns[self.output])?;
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        let SparseMaterializationWorkspace {
            matrices,
            values,
            scratch,
            seen,
            lu,
            dense,
            ..
        } = workspace;
        let n = self.layout.patterns[INPUT].view().rows();
        // Input validation precedes the epoch, so rejection stays a preflight attempt.
        matrices[INPUT].update_values(
            |_, candidate| scatter(input, &self.layout.patterns[INPUT], candidate, seen),
            &mut values[INPUT],
        )?;
        dense.run_scope(
            scope,
            materialization_identity(n),
            self.options.execution.solve_budgets(),
            |_, _, solve| {
                let execution = solve.execution();
                execution.check()?;
                if self.identity {
                    matrices[AUXILIARY].update_values(
                        |_, candidate| {
                            candidate.fill(S::ONE);
                            Ok::<_, SciOxideError>(())
                        },
                        &mut values[AUXILIARY],
                    )?;
                }
                let mut work = 0;
                if let Some(blocks) = &self.blocks {
                    // Reservation builds the LU whenever the layout declares one.
                    let lu = lu.as_mut().ok_or_else(|| {
                        crate::checked::inconsistent(
                            scioxide_core::argument!("workspace"),
                            "workspace lacks the plan's sparse LU",
                        )
                    })?;
                    let (head, tail) = crate::workspace::split_mut(matrices, AUXILIARY)?;
                    invert(
                        &head[INPUT],
                        &mut tail[0],
                        &self.layout.patterns[AUXILIARY],
                        blocks,
                        (lu, &mut values[AUXILIARY]),
                        self.options.execution.accuracy(),
                        solve,
                    )?;
                    work = n * 2;
                }
                let options = product_options(self.options.execution);
                for step in &self.steps {
                    admit_iterations(solve, 1)?;
                    let (head, tail) = crate::workspace::split_mut(matrices, step.output)?;
                    step.plan.evaluate_into(
                        &head[step.left],
                        &head[step.right],
                        options,
                        &mut tail[0],
                        scratch,
                        execution,
                    )?;
                    work += 1;
                }
                execution.check()?;
                let values = matrices[self.output].values();
                output.copy_values_from(values)?;
                Ok(SparseMaterializationReport {
                    structural_fill: values.len(),
                    nonzero_fill: values.iter().filter(|&&value| value != S::ZERO).count(),
                    peak_intermediate_fill: self.peak_fill,
                    admitted_workspace_bytes: self.bytes,
                    work,
                    estimated_error: None,
                })
            },
        )
    }
}

/// Materialize `A^{-1}` column by column into its full connected blocks.
fn invert<S: NumericScalar>(
    input: &CsrMatrix<S>,
    inverse: &mut CsrMatrix<S>,
    pattern: &StructuralPattern,
    blocks: &ConnectedBlockLayout,
    (lu, value_workspace): (
        &mut LuStorage<S>,
        &mut scioxide_sparse::SparseValueWorkspace<S>,
    ),
    accuracy: scioxide_core::Accuracy,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = input.rows();
    let options = scioxide_sparse::SparseSolveOptions::new().with_accuracy(accuracy);
    admit_iterations(solve, n)?;
    if let Some(factor) = &mut lu.factor {
        factor.refactor(input.view(), options, &mut lu.workspace, &mut *solve)?;
    } else {
        lu.factor = Some(
            lu.workspace
                .factor_scoped(input.view(), options, solve.execution())?,
        );
    }
    let factor = lu
        .factor
        .as_ref()
        .ok_or_else(|| SciOxideError::singular(None))?;
    let pointers = pattern.view().row_ptr();
    inverse.update_values(
        |_, values| {
            lu.rhs.fill(S::ZERO);
            for j in 0..n {
                admit_iterations(solve, 1)?;
                lu.rhs[j] = S::ONE;
                factor.solve_into(
                    &lu.rhs,
                    SparseOperation::Normal,
                    options,
                    &mut lu.column,
                    &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut lu.scratch),
                    &mut *solve,
                )?;
                lu.rhs[j] = S::ZERO;
                let group = blocks.group[j];
                for (i, &value) in lu.column.iter().enumerate() {
                    if blocks.group[i] == group {
                        values[pointers[i] + blocks.local[j]] = value;
                    } else if value != S::ZERO {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Overflow,
                            "inverse fill escaped its connected block",
                        ));
                    }
                }
            }
            Ok::<_, SciOxideError>(())
        },
        value_workspace,
    )
}

/// Plan for a matrix function `f(A)` of a sparse square matrix, with a fixed output pattern.
///
/// `f(A)` is zero outside the connected blocks of `A` (the weakly connected
/// components of its graph) but can be dense within them. The output pattern
/// therefore holds every block in full, and values that are numerically zero
/// stay as explicit zeros. Creation finds the blocks and builds the output
/// pattern, in time linear in the stored entries of `A` and of the output.
/// Each block of order `b` is evaluated as a dense matrix at O(b³) cost, so the
/// method suits matrices whose blocks are small.
/// The exponential uses scaling and squaring with Padé approximants. The
/// other functions start from a Schur decomposition: for real `A`, the
/// logarithm and square root use the real Schur form, and the other functions,
/// or a square root with a negative real eigenvalue, use the complex Schur form.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{MatrixFunction, SparseFunctionOptions, SparseFunctionPlan};
/// use scioxide_sparse::CsrMatrix;
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![0.0_f64, 1.0])?;
/// let plan = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &SparseFunctionOptions::new())?;
/// let exp = plan.evaluate(a.view())?;
/// // exp(diag(0, 1)) = diag(1, e).
/// assert!((exp.value().values()[1] - 1.0_f64.exp()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseFunctionPlan<'f, S: NumericScalar>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    function: ValidatedFunction<'f, S::Real>,
    options: SparseFunctionOptions<S::Real>,
    layout: Layout,
    blocks: ConnectedBlockLayout,
    output: StructuralPattern,
    request: MatrixFunctionWorkspaceRequest,
    bytes: usize,
    scalar: PhantomData<fn() -> S>,
}

impl<'f, S: NumericScalar> SparseFunctionPlan<'f, S>
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    /// Creates a plan for `f(A)`, `f = function`, from the pattern of `input`.
    ///
    /// Only the pattern of `input` is used; duplicate entries are merged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` is not square.
    /// - `DomainError` if `function` is a power with a non-finite exponent.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if an error estimate is
    ///   requested for [`MatrixFunction::Analytic`].
    /// - `ResourceLimit` if the input or the full-block output has more stored
    ///   entries than the entry limit, or the workspace exceeds the byte limit.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(
        input: SparseView<'_, S>,
        function: MatrixFunction<'f, S::Real>,
        options: &SparseFunctionOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let function = ValidatedFunction::new(function)?;
        let estimate = options.error_estimate == Estimation::Estimate;
        if estimate && matches!(function.function(), MatrixFunction::Analytic(_)) {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("options"),
                "error estimates require a built-in function",
            ));
        }
        let limit = options.execution.byte_limit();
        let matrix = canonical(input, options.max_entries, limit)?;
        let blocks = ConnectedBlockLayout::new(&matrix)?;
        let (fill, order) = blocks.fill()?;
        admit_fill(fill, options.max_entries)?;
        let request = if estimate {
            MatrixFunctionWorkspaceRequest::Condition
        } else {
            MatrixFunctionWorkspaceRequest::Evaluate
        };
        // Exponentials stage dense input directly; the remaining functions
        // stage T and Q and reuse the shared Schur function kernels.
        let scratch = scratch_lengths::<S::Real>(order, request)?;
        let exp = matches!(function.function(), MatrixFunction::Exp);
        let square = order.checked_mul(order).ok_or_else(overflow)?;
        let factors = square
            .checked_mul(if exp { 1 } else { 2 })
            .ok_or_else(overflow)?;
        let schur = if exp {
            None
        } else if S::IS_REAL {
            Some(crate::eigen::schur_kernel_lengths::<S::Real>(
                order,
                ExecutionPolicy::Sequential,
            )?)
        } else {
            Some(crate::eigen::schur_kernel_lengths::<Complex<S::Real>>(
                order,
                ExecutionPolicy::Sequential,
            )?)
        };
        let mut dense = LinalgRequirements::default();
        let complex_factors = if S::IS_REAL && exp { 0 } else { factors };
        let complex_work = if S::IS_REAL {
            0
        } else {
            schur.as_ref().map_or(0, |s| s.scalars)
        };
        *dense.scalar_mut(ScalarRegion::Direct) = complex_factors
            .checked_add(scratch.scalars.max(complex_work))
            .ok_or_else(overflow)?;
        *dense.real_mut(RealRegion::Direct) = if S::IS_REAL {
            let schur_work = schur
                .as_ref()
                .map_or(Some(0), |s| s.scalars.checked_add(s.reals))
                .ok_or_else(overflow)?;
            factors
                .checked_add(scratch.scalars.max(schur_work))
                .ok_or_else(overflow)?
        } else {
            schur.as_ref().map_or(0, |s| s.reals)
        };
        dense.extended = scratch.extended;
        dense.indices = scratch.indices;
        let mut patterns = try_with_capacity(1, scioxide_core::quantity!("patterns"))?;
        patterns.push(owned_pattern(&matrix)?);
        let layout = Layout {
            patterns,
            scratch: SparseProductRequirements::default(),
            candidate: fill,
            lu: None,
            dense,
        };
        let bytes = layout_bytes::<S>(&layout, 0)?;
        crate::workspace::admit_bytes(
            bytes,
            limit,
            scioxide_core::quantity!("sparse_workspace_bytes"),
        )?;
        Ok(Self {
            function,
            options: *options,
            output: blocks.pattern(fill)?,
            layout,
            blocks,
            request,
            bytes,
            scalar: PhantomData,
        })
    }

    /// Returns the number of stored entries of the full-block output pattern.
    pub fn structural_fill(&self) -> usize {
        self.output.view().col_indices().len()
    }

    /// Returns the workspace storage of this plan; the request is `()`.
    ///
    /// # Errors
    ///
    /// None; all sizes are fixed when the plan is created. The `Result` keeps
    /// the signature of the other plans.
    pub fn workspace_requirements(
        &self,
        request: (),
    ) -> Result<SparseMaterializationRequirements<'_, S>, SciOxideError> {
        let () = request;
        Ok(SparseMaterializationRequirements {
            layout: &self.layout,
            bytes: self.bytes,
            byte_limit: self.options.execution.byte_limit(),
            scalar: PhantomData,
        })
    }

    /// Allocates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// The errors of [`SparseMaterializationWorkspace::try_reserve`].
    pub fn workspace(
        &self,
        request: (),
    ) -> Result<SparseMaterializationWorkspace<S>, SciOxideError> {
        let mut workspace = SparseMaterializationWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Allocates a real or complex output matrix with the full-block pattern and zero values.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if an allocation fails.
    pub fn create_output<O: NumericScalar<Real = S::Real>>(
        &self,
    ) -> Result<CsrMatrix<O>, SciOxideError> {
        csr_from_pattern(&self.output)
    }

    /// Computes `f(A)` for the values of `input` with a new output and workspace.
    ///
    /// The output has the scalar type of `input`. Use [`Self::evaluate_into`]
    /// with complex output when `f(A)` of a real matrix can be complex.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::create_output`] and [`Self::evaluate_into`].
    pub fn evaluate(
        &self,
        input: SparseView<'_, S>,
    ) -> Result<SparseMaterialization<S>, SciOxideError> {
        let mut value = self.create_output()?;
        let mut workspace = SparseMaterializationWorkspace::new();
        let report = self.evaluate_into(
            input,
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(SparseMaterialization { value, report })
    }

    /// Computes `f(A)` for the values of `input` and writes it to `output`.
    ///
    /// `input` must have the pattern the plan was created with, and `output`
    /// the full-block pattern (see [`Self::create_output`]). The workspace grows
    /// on first use, and later calls reuse it. With [`Estimation::Estimate`] the
    /// report carries [`SparseMaterializationReport::estimated_error`].
    ///
    /// With a real output type the result must be real: the largest imaginary
    /// part may not exceed the tolerance of the function settings, relative to
    /// the sum of the absolute values of the result. The square root of a real
    /// matrix with a negative real eigenvalue therefore needs complex output.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has the wrong shape.
    /// - `InvalidInput` (`Inconsistent`) if the pattern of `input` or `output`
    ///   differs from the plan's.
    /// - `NonFinite` if `input` has a non-finite value or the result is not finite.
    /// - The errors of [`SparseMaterializationWorkspace::try_reserve`].
    /// - `DomainError` if a block has an eigenvalue on a branch cut of `f`, or
    ///   if `output` is real and the result is not.
    /// - `ConvergenceFailure` if a Schur decomposition does not converge or the
    ///   work limit is reached.
    /// - `ExecutionStopped` if cancellation or a deadline stops the evaluation.
    /// - Errors returned by an analytic `f`.
    pub fn evaluate_into<'scope, 'capability, O: NumericScalar<Real = S::Real>>(
        &self,
        input: SparseView<'_, S>,
        output: &mut CsrMatrix<O>,
        workspace: &mut SparseMaterializationWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<SparseMaterializationReport<S::Real>, SciOxideError> {
        workspace.dense.begin_attempt();
        let scope = context.into().with_options(&self.options.execution)?;
        check_output(output, &self.output)?;
        workspace.try_reserve(&self.workspace_requirements(())?)?;
        let SparseMaterializationWorkspace {
            matrices,
            values,
            seen,
            candidate,
            dense,
            ..
        } = workspace;
        let n = self.output.view().rows();
        // Input validation precedes the epoch, so rejection stays a preflight attempt.
        matrices[INPUT].update_values(
            |_, candidate| scatter(input, &self.layout.patterns[INPUT], candidate, seen),
            &mut values[INPUT],
        )?;
        dense.run_scope(
            scope,
            materialization_identity(n),
            scioxide_core::SolveBudgets::unbounded(),
            |dense, kernels, solve| {
                solve.execution().check()?;
                let (work, estimated_error) =
                    dense.with_direct(|direct, reals, indices, child| {
                        self.evaluate_blocks::<O>(
                            &matrices[INPUT],
                            candidate,
                            BlockScratch {
                                direct,
                                reals,
                                extended: child.extended_mut(),
                                indices,
                            },
                            kernels,
                            solve,
                        )
                    })?;
                let values = &candidate[..self.structural_fill()];
                if O::IS_REAL {
                    real_defect::<S::Real, O>(
                        candidate_report(values)?,
                        &MatrixFunctionOptions::new()
                            .with_function_settings(self.options.function)
                            .with_execution_options(self.options.execution),
                    )?;
                }
                solve.execution().check()?;
                if values
                    .iter()
                    .any(|v| !O::from_components(v.re, v.im).has_finite_components())
                {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Output,
                        "sparse materialization output is not finite",
                    ));
                }
                for (index, &value) in values.iter().enumerate() {
                    output.set_value(index, O::from_components(value.re, value.im))?;
                }
                Ok(SparseMaterializationReport {
                    structural_fill: values.len(),
                    nonzero_fill: output
                        .values()
                        .iter()
                        .filter(|&&value| value != O::ZERO)
                        .count(),
                    peak_intermediate_fill: values.len().max(self.layout.input_entries()),
                    admitted_workspace_bytes: self.bytes,
                    work,
                    estimated_error,
                })
            },
        )
    }

    fn evaluate_blocks<O: NumericScalar<Real = S::Real>>(
        &self,
        input: &CsrMatrix<S>,
        candidate: &mut [Complex<S::Real>],
        scratch: BlockScratch<'_, S::Real>,
        kernels: crate::KernelContext<'_>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<(usize, Option<S::Real>), SciOxideError> {
        let BlockScratch {
            direct,
            reals,
            extended,
            indices,
        } = scratch;
        let options = &MatrixFunctionOptions::new()
            .with_function_settings(self.options.function)
            .with_execution_options(self.options.execution);
        let estimate = self.request == MatrixFunctionWorkspaceRequest::Condition;
        let pointers = self.output.view().row_ptr();
        let limit = options.work_limit().get();
        let mut work = 0;
        let mut error = S::Real::ZERO;
        for block in self.blocks.iter() {
            let m = block.len();
            let mut rest = &mut direct[..];
            let mut real_rest = &mut reals[..];
            let exp = matches!(self.function.function(), MatrixFunction::Exp);
            let t = take_mut(&mut rest, if S::IS_REAL && exp { 0 } else { m * m })?;
            let q = take_mut(&mut rest, if exp { 0 } else { m * m })?;
            let rt = take_mut(&mut real_rest, if S::IS_REAL { m * m } else { 0 })?;
            let rq = take_mut(&mut real_rest, if S::IS_REAL && !exp { m * m } else { 0 })?;
            t.fill(Complex::ZERO);
            rt.fill(S::Real::ZERO);
            let mut norm = Scaled::ZERO;
            for (a, &row) in block.iter().enumerate() {
                for position in input.row_ptr()[row]..input.row_ptr()[row + 1] {
                    let value = input.values()[position];
                    let index = a * m + self.blocks.local[input.col_indices()[position]];
                    if S::IS_REAL {
                        rt[index] = value.real_component();
                    } else {
                        t[index] = crate::eigen::into_complex(value);
                    }
                    norm = norm.add(scioxide_core::ScaledValue::from_value(value)?.magnitude()?)?;
                }
            }
            let factors = if S::IS_REAL {
                if exp {
                    PreparedRef::DenseReal { a: rt, n: m }
                } else {
                    let lengths = crate::eigen::schur_kernel_lengths::<S::Real>(
                        m,
                        ExecutionPolicy::Sequential,
                    )?;
                    let (scalars, real) = crate::workspace::split_mut(real_rest, lengths.scalars)?;
                    crate::eigen::schur_kernel(
                        rt,
                        rq,
                        m,
                        scalars,
                        real,
                        &SchurOptions::new().with_execution_options(
                            self.options
                                .execution
                                .with_execution_policy(ExecutionPolicy::Sequential),
                        ),
                        solve,
                    )?;
                    match self.function {
                        ValidatedFunction::Principal(function)
                            if !matches!(self.function.function(), MatrixFunction::Sqrt)
                                || !real_eigenvalues(rt, m)
                                    .any(|z| z.im == S::Real::ZERO && z.re < S::Real::ZERO) =>
                        {
                            PreparedRef::RealSchur {
                                t: rt,
                                q: rq,
                                n: m,
                                function,
                            }
                        }
                        _ => {
                            complexify_schur(rt, rq, t, q, m)?;
                            PreparedRef::ComplexSchur {
                                t,
                                q,
                                n: m,
                                function: self.function,
                            }
                        }
                    }
                }
            } else if exp {
                PreparedRef::DenseComplex { a: t, n: m }
            } else {
                crate::eigen::schur_kernel(
                    t,
                    q,
                    m,
                    &mut rest[..],
                    &mut real_rest[..],
                    &SchurOptions::new().with_execution_options(
                        self.options
                            .execution
                            .with_execution_policy(ExecutionPolicy::Sequential),
                    ),
                    solve,
                )?;
                PreparedRef::ComplexSchur {
                    t,
                    q,
                    n: m,
                    function: self.function,
                }
            };
            factors.branch(estimate, O::IS_REAL)?;
            let mut scratch = Scratch::carve(
                m,
                self.request,
                rest,
                &mut real_rest[..],
                &mut extended[..],
                &mut indices[..],
            )?;

            let mut control = FunctionControl::new(kernels, Some(&mut *solve), limit - work);
            let value: &[Complex<S::Real>] = if estimate {
                let (_, absolute) =
                    condition_candidate(factors, &mut scratch, options, &mut control)?;
                let term = Scaled::from_value(absolute)?
                    .mul(Scaled::from_value(norm.round()?)?)?
                    .mul(Scaled::from_value(S::Real::EPSILON)?)?
                    .round()?;
                error = Scaled::from_value(error)?
                    .add(Scaled::from_value(term)?)?
                    .round()?;
                scratch.value
            } else {
                evaluate_candidate(
                    factors,
                    None::<MatrixView<'_, Complex<S::Real>>>,
                    &mut scratch.buffers,
                    options,
                    &mut control,
                )?;
                scratch.buffers.candidate
            };
            work += control.work;
            for (a, &row) in block.iter().enumerate() {
                candidate[pointers[row]..pointers[row] + m]
                    .copy_from_slice(&value[a * m..(a + 1) * m]);
            }
        }
        Ok((work, estimate.then_some(error)))
    }
}

/// Dense arenas lent to connected-block evaluation.
struct BlockScratch<'w, R: RealScalar> {
    direct: &'w mut [Complex<R>],
    reals: &'w mut [R],
    extended: &'w mut [scioxide_core::ScaledValue<scioxide_core::Complex<R>>],
    indices: &'w mut [usize],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_layout_includes_structural_zeros_and_isolated_vertices() {
        let matrix = CsrMatrix::from_raw(
            6,
            6,
            alloc::vec![0, 1, 1, 2, 3, 3, 3],
            alloc::vec![4, 1, 3],
            alloc::vec![0.0_f64, 2.0, 1.0],
        )
        .unwrap();
        let layout = ConnectedBlockLayout::new(&matrix).unwrap();
        assert_eq!(layout.offsets, [0, 2, 4, 5, 6]);
        assert_eq!(layout.members, [0, 4, 1, 2, 3, 5]);
        assert_eq!(layout.group, [0, 1, 1, 2, 0, 3]);
        assert_eq!(layout.local, [0, 0, 1, 0, 1, 0]);
        assert_eq!(layout.fill().unwrap(), (10, 2));
    }

    #[test]
    fn arrowhead_components_preserve_sorted_block_layout() {
        let n = 100_000;
        let mut pointers = alloc::vec![0, n];
        let mut columns: Vec<_> = (0..n).collect();
        for row in 1..n {
            columns.extend([0, row]);
            pointers.push(columns.len());
        }
        let matrix = CsrMatrix::from_raw(n, n, pointers, columns, alloc::vec![1.0; 3*n-2]).unwrap();
        let blocks = ConnectedBlockLayout::new(&matrix).unwrap();
        assert_eq!(blocks.offsets, [0, n]);
        assert!(blocks.members.iter().copied().eq(0..n));
        assert!(blocks.local.iter().copied().eq(0..n));
    }
}
