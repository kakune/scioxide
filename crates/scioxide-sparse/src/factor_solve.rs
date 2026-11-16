//! Solve options and workspaces, and the solve and apply methods that factors
//! and preconditioners share.
// Macros generate the inherent entry points without a public trait or blanket
// operator implementation. Families retain only numerical kernels.
use scioxide_core::{
    Accuracy, AttemptKind, AttemptOutcome, AttemptRecord, BlockVectorView, BlockVectorViewMut,
    ExecutionContext, ExecutionPolicy, ProblemToken, SciOxideError, SolveBudgets, SolveContext,
    SolveIdentity, SolveScope, SolveState,
};
/// Sets the accuracy, parallelism and iteration limits of sparse
/// factorizations, solves and preconditioner applications.
///
/// The default is [`Accuracy::Standard`], sequential execution and no limits.
/// With a parallel [`ExecutionPolicy`], block solves and applications split
/// their right-hand sides among at most 64 workers of the executor.
/// Factorization is always sequential. The limits apply to each call, and a call
/// made inside an enclosing solver also counts toward that solver's limits, as
/// described in [Solve accounting](scioxide_core#solve-accounting). The
/// [`LinearOperator`](scioxide_core::LinearOperator) and
/// [`Preconditioner`](scioxide_core::Preconditioner) implementations of factors
/// and preconditioners use the default options.
///
/// # Accuracy
///
/// With `Standard` accuracy, each real update `a − b·c` satisfies the error
/// bound of two floating-point operations. A complex component counts its real
/// products and additions separately. When an intermediate value would
/// overflow or underflow, the update is recomputed from its operands with exact
/// or rescaled arithmetic. Substitutions combine these bounds over the stored
/// entries of the triangular factors. The error relative to the original system
/// also depends on its condition number and on the growth of the factors, and
/// incomplete factors add the error of the entries they drop.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Accuracy, Budget, BudgetKind, SolveBudgets};
/// use scioxide_sparse::SparseSolveOptions;
/// let options = SparseSolveOptions::new()
///     .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
/// assert_eq!(
///     options.budgets().report(BudgetKind::Iteration).consumed(),
///     0
/// );
/// assert_eq!(
///     options.with_accuracy(Accuracy::Exact).accuracy(),
///     Accuracy::Exact
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SparseSolveOptions {
    accuracy: Accuracy,
    execution: ExecutionPolicy,
    budgets: SolveBudgets,
}
impl Default for SparseSolveOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseSolveOptions {
    /// Creates options with `Standard` accuracy, sequential execution and no
    /// limits.
    pub const fn new() -> Self {
        Self {
            accuracy: Accuracy::Standard,
            execution: ExecutionPolicy::Sequential,
            budgets: SolveBudgets::unbounded(),
        }
    }
    /// Sets the accuracy of LU, Cholesky, `LDLᴴ` and incomplete factorizations,
    /// their solves and preconditioner applications. QR always uses rescaled
    /// arithmetic, whatever this setting.
    pub const fn with_accuracy(mut self, accuracy: Accuracy) -> Self {
        self.accuracy = accuracy;
        self
    }
    /// Returns the selected accuracy.
    pub const fn accuracy(self) -> Accuracy {
        self.accuracy
    }
    /// Sets the execution policy. Workers come from the executor of the
    /// execution context; this setting creates none.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution = value;
        self
    }
    /// Sets the iteration and evaluation limits of each call.
    pub const fn with_budgets(mut self, value: SolveBudgets) -> Self {
        self.budgets = value;
        self
    }
    /// Returns the selected execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution
    }
    /// Returns the selected limits.
    pub const fn budgets(self) -> SolveBudgets {
        self.budgets
    }
    pub(crate) fn workers(self) -> core::num::NonZeroUsize {
        crate::execution::policy_workers(self.execution)
    }
}
#[derive(Debug, Default)]
pub(crate) struct Epoch {
    state: SolveState,
    record: Option<AttemptRecord<()>>,
}
impl Epoch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
    pub(crate) fn reject(&mut self) {
        self.record = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
    }
    pub(crate) fn record(&self) -> Option<&AttemptRecord<()>> {
        self.record.as_ref()
    }
    pub(crate) fn run<'s, 'c, T>(
        &mut self,
        scope: impl Into<SolveScope<'s, 'c>>,
        options: SparseSolveOptions,
        run: impl FnOnce(&mut SolveContext<'_, 'c>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut entered = false;
        let result = scope.into().run(
            &mut self.state,
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("sparse", "factor", 1),
                0,
                0,
            ),
            options.budgets(),
            |context| {
                entered = true;
                context.execution().check()?;
                let value = run(context)?;
                context.execution().check()?;
                Ok(value)
            },
        );
        if !entered {
            self.reject();
            return result;
        }
        self.record = Some(AttemptRecord::new(
            self.state.epoch(),
            AttemptKind::Call,
            if result.is_ok() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Failed
            },
            self.state.counts(),
            result.as_ref().ok().map(|_| ()),
        ));
        result
    }
}

/// Holds the scratch size of a vector or block solve, or of a preconditioner
/// application.
///
/// ```
/// use scioxide_core::{DiagonalKind, Triangle};
/// use scioxide_sparse::{CsrMatrix, SparseTriangular};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 4.0])?;
/// let factor = SparseTriangular::new(a, Triangle::Lower, DiagonalKind::Explicit)?;
/// // Three right-hand sides, each with 2 scalars of output and 2 of work.
/// let requirements = factor.workspace_requirements(3)?;
/// assert_eq!(requirements.scalar_len(), 12);
/// assert_eq!(
///     requirements.required_bytes(),
///     12 * core::mem::size_of::<f64>()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SparseSolveRequirements {
    pub(crate) scalars: usize,
    bytes: usize,
    base: usize,
    limit: usize,
}
impl SparseSolveRequirements {
    // `stride` staged scalars per right-hand side: one candidate column plus kernel work.
    pub(crate) fn new<S>(
        stride: usize,
        rhs: usize,
        base: usize,
        limit: usize,
    ) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("solve_workspace");
        let scalars = scioxide_core::backend::checked_mul(stride, rhs, q)?;
        let bytes = scioxide_core::backend::checked_mul(scalars, core::mem::size_of::<S>(), q)?;
        scioxide_core::backend::checked_add(base, bytes, q)?;
        Ok(Self {
            scalars,
            bytes,
            base,
            limit,
        })
    }
    /// Returns the number of scalars of scratch needed. A longer borrowed
    /// slice is accepted, and its extra entries are not touched.
    pub const fn scalar_len(self) -> usize {
        self.scalars
    }
    /// Returns the bytes of scratch needed, without the factor or plan.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
}
/// Holds the scratch storage of solves and preconditioner applications.
///
/// The workspace either owns its storage or borrows a slice you provide.
/// Create an owned one with the `workspace(rhs)` method of a factor or
/// preconditioner, or with [`new`](Self::new) and
/// [`try_reserve`](Self::try_reserve); wrap a slice with
/// [`borrowed`](Self::borrowed). Owned storage counts toward the byte limit of
/// the factor, while a borrowed slice is checked only for its length. The
/// workspace records the most recent call in
/// [`last_attempt`](Self::last_attempt).
///
/// ```
/// use scioxide_core::{DiagonalKind, ExecutionContext, SciOxideErrorKind, Triangle};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseSolveOptions, SparseSolveWorkspace, SparseTriangular,
/// };
/// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
/// let factor = SparseTriangular::new(a, Triangle::Lower, DiagonalKind::Explicit)?;
/// let mut workspace = factor.workspace(1)?;
/// let mut output = [0.];
/// factor.solve_into(
///     &[4.],
///     SparseOperation::Normal,
///     SparseSolveOptions::new(),
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(output, [2.]);
/// // A borrowed slice shorter than `scalar_len` is rejected; the output is unchanged.
/// let mut short = [0.; 1];
/// let e = factor
///     .solve_into(
///         &[6.],
///         SparseOperation::Normal,
///         SparseSolveOptions::new(),
///         &mut output,
///         &mut SparseSolveWorkspace::borrowed(&mut short),
///         ExecutionContext::sequential(),
///     )
///     .unwrap_err();
/// assert_eq!(e.kind(), SciOxideErrorKind::DimensionMismatch);
/// assert_eq!(output, [2.]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseSolveWorkspace<'a, S: scioxide_core::NumericScalar> {
    pub(crate) scratch: SolveScratch<'a, S>,
    pub(crate) epoch: Epoch,
}
impl<S: scioxide_core::NumericScalar> Default for SparseSolveWorkspace<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a, S: scioxide_core::NumericScalar> SparseSolveWorkspace<'a, S> {
    /// Creates an empty workspace that owns its storage. Reserve it with
    /// [`try_reserve`](Self::try_reserve) before use.
    pub fn new() -> Self {
        Self {
            scratch: SolveScratch::Owned(alloc::vec::Vec::new()),
            epoch: Epoch::new(),
        }
    }
    /// Creates a workspace that uses `scratch` as its storage.
    ///
    /// Only the first `scalar_len` entries are used, and the byte limit does not
    /// apply.
    pub fn borrowed(scratch: &'a mut [S]) -> Self {
        Self {
            scratch: SolveScratch::Borrowed(scratch),
            epoch: Epoch::new(),
        }
    }
    /// Returns the bytes of storage this workspace owns, including unused
    /// capacity. A borrowed slice counts as zero.
    pub fn retained_bytes(&self) -> usize {
        self.scratch.capacity() * core::mem::size_of::<S>()
    }
    /// Returns the record of the most recent call that used this workspace,
    /// including a call rejected before it started.
    pub fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.epoch.record()
    }
    /// Checks whether this workspace has room for `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the owned storage, together with the storage of the
    ///   factor or plan, exceeds the byte limit in `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(
        &self,
        r: &SparseSolveRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = scioxide_core::backend::checked_add(
            r.base,
            self.retained_bytes(),
            scioxide_core::quantity!("solve_workspace"),
        )?;
        if bytes > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("solve_workspace"),
                bytes,
                r.limit,
            ));
        }
        Ok(if self.scratch.len() >= r.scalars {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    /// Grows owned storage to fit `r`. A borrowed slice is only checked.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a borrowed slice is shorter than
    ///   `r.scalar_len()`.
    /// - `ResourceLimit` if the owned storage would exceed the byte limit in `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn try_reserve(&mut self, r: &SparseSolveRequirements) -> Result<(), SciOxideError> {
        let _ = self.check(r)?;
        if let SolveScratch::Borrowed(scratch) = &self.scratch {
            return crate::helpers::minimum(scratch.len(), r.scalars);
        }
        let bytes = scioxide_core::backend::checked_add(
            r.base,
            self.retained_bytes().max(r.bytes),
            scioxide_core::quantity!("solve_workspace"),
        )?;
        if bytes > r.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("solve_workspace"),
                bytes,
                r.limit,
            ));
        }
        match &mut self.scratch {
            SolveScratch::Owned(scratch) => {
                scratch
                    .try_reserve_exact(r.scalars.saturating_sub(scratch.len()))
                    .map_err(|_| {
                        SciOxideError::allocation_failure(
                            scioxide_core::quantity!("solve_workspace"),
                            r.scalars,
                        )
                    })?;
                scratch.resize(r.scalars.max(scratch.len()), S::ZERO);
            }
            SolveScratch::Borrowed(scratch) => crate::helpers::minimum(scratch.len(), r.scalars)?,
        }
        let _ = self.check(r)?;
        Ok(())
    }
    pub(crate) fn require(&self, r: &SparseSolveRequirements) -> Result<(), SciOxideError> {
        if self.check(r)? == scioxide_core::Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                r.scalars,
                self.scratch.len(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub(crate) enum SolveScratch<'a, S> {
    Owned(alloc::vec::Vec<S>),
    Borrowed(&'a mut [S]),
}
impl<S> SolveScratch<'_, S> {
    fn capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed(_) => 0,
        }
    }
}
impl<S> core::ops::Deref for SolveScratch<'_, S> {
    type Target = [S];
    fn deref(&self) -> &[S] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
        }
    }
}
impl<S> core::ops::DerefMut for SolveScratch<'_, S> {
    fn deref_mut(&mut self) -> &mut [S] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
        }
    }
}

/// One prepared column application `candidate = op(rhs)` behind the shared
/// vector, block and protocol forms.
///
/// Configuration is validated at preparation. Kernels receive finite,
/// dimension-checked input, a candidate column and exactly `work_len` slots;
/// callers copy the candidate to the output only after the kernel and scope
/// succeed.
pub(crate) trait Application<S: scioxide_core::NumericScalar>: Sync {
    /// Square dimension.
    fn dimension(&self) -> usize;
    /// Kernel scratch after each staged candidate column.
    fn work_len(&self) -> Result<usize, SciOxideError>;
    /// Base bytes already held and the byte limit charged to owned staging.
    fn staging_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((0, usize::MAX))
    }
    /// Work counted for one right-hand side before any column executes.
    fn admit(&self, context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
        crate::helpers::admit(context)
    }
    /// Apply one column; `options` selects arithmetic and in-column dispatch.
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        work: &mut [S],
        options: SparseSolveOptions,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}

/// A factor solve of one fixed operation.
pub(crate) struct Oriented<'a, F> {
    pub(crate) factor: &'a F,
    pub(crate) operation: crate::SparseOperation,
}

pub(crate) fn requirements<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    rhs: usize,
) -> Result<SparseSolveRequirements, SciOxideError> {
    let (base, limit) = application.staging_limits()?;
    SparseSolveRequirements::new::<S>(stride(application)?, rhs, base, limit)
}

pub(crate) fn workspace<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    rhs: usize,
) -> Result<SparseSolveWorkspace<'static, S>, SciOxideError> {
    let mut workspace = SparseSolveWorkspace::new();
    workspace.try_reserve(&requirements(application, rhs)?)?;
    Ok(workspace)
}

pub(crate) fn apply_into<'s, 'c, S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    rhs: &[S],
    options: SparseSolveOptions,
    output: &mut [S],
    workspace: &mut SparseSolveWorkspace<'_, S>,
    scope: impl Into<SolveScope<'s, 'c>>,
) -> Result<(), SciOxideError> {
    workspace.epoch.reject();
    let n = application.dimension();
    crate::helpers::exact(rhs.len(), n)?;
    crate::helpers::exact(output.len(), n)?;
    crate::helpers::finite(rhs)?;
    let r = requirements(application, 1)?;
    workspace.require(&r)?;
    let SparseSolveWorkspace { scratch, epoch } = workspace;
    epoch.run(scope, options, |cx| {
        application.admit(cx)?;
        let (candidate, work) = scratch[..r.scalar_len()].split_at_mut(n);
        application.apply_column(rhs, candidate, work, options, cx.execution())
    })?;
    output.copy_from_slice(&scratch[..n]);
    Ok(())
}

pub(crate) fn apply_block_into<
    's,
    'c,
    S: scioxide_core::NumericScalar,
    A: Application<S> + ?Sized,
>(
    application: &A,
    rhs: BlockVectorView<'_, S>,
    options: SparseSolveOptions,
    output: &mut BlockVectorViewMut<'_, S>,
    workspace: &mut SparseSolveWorkspace<'_, S>,
    scope: impl Into<SolveScope<'s, 'c>>,
) -> Result<(), SciOxideError> {
    workspace.epoch.reject();
    let count = validate_block(application, rhs, output)?;
    let r = requirements(application, count)?;
    workspace.require(&r)?;
    let stride = stride(application)?;
    let SparseSolveWorkspace { scratch, epoch } = workspace;
    epoch.run(scope, options, |cx| {
        for _ in 0..count {
            application.admit(cx)?;
        }
        stage_block(
            application,
            rhs,
            options,
            &mut scratch[..r.scalar_len()],
            cx.execution(),
        )
    })?;
    crate::execution::publish_columns(output, scratch, stride);
    Ok(())
}

/// Unaccounted protocol application with borrowed scalar scratch and Standard options.
pub(crate) fn apply_slices<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    input: &[S],
    output: &mut [S],
    scratch: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = application.dimension();
    crate::helpers::exact(input.len(), n)?;
    crate::helpers::exact(output.len(), n)?;
    crate::helpers::finite(input)?;
    let r = requirements(application, 1)?;
    crate::helpers::minimum(scratch.len(), r.scalar_len())?;
    execution.check()?;
    let (candidate, work) = scratch[..r.scalar_len()].split_at_mut(n);
    application.apply_column(input, candidate, work, SparseSolveOptions::new(), execution)?;
    execution.check()?;
    output.copy_from_slice(candidate);
    Ok(())
}

/// Unaccounted sequential block protocol application with borrowed scratch.
pub(crate) fn apply_block_slices<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    input: BlockVectorView<'_, S>,
    output: &mut BlockVectorViewMut<'_, S>,
    scratch: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let count = validate_block(application, input, output)?;
    let r = requirements(application, count)?;
    crate::helpers::minimum(scratch.len(), r.scalar_len())?;
    stage_block(
        application,
        input,
        SparseSolveOptions::new(),
        &mut scratch[..r.scalar_len()],
        execution,
    )?;
    crate::execution::publish_columns(output, scratch, stride(application)?);
    Ok(())
}

fn stride<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_add(
        application.dimension(),
        application.work_len()?,
        scioxide_core::quantity!("solve_workspace"),
    )
}

// Exact block shapes and finite input; returns the right-hand-side count.
fn validate_block<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    rhs: BlockVectorView<'_, S>,
    output: &BlockVectorViewMut<'_, S>,
) -> Result<usize, SciOxideError> {
    crate::helpers::exact(rhs.rows(), application.dimension())?;
    crate::helpers::exact(output.rows(), application.dimension())?;
    crate::helpers::exact(rhs.right_hand_sides(), output.right_hand_sides())?;
    crate::helpers::finite(rhs.as_slice())?;
    Ok(rhs.right_hand_sides())
}

// Stage every column under the selected policy; each column runs sequentially.
fn stage_block<S: scioxide_core::NumericScalar, A: Application<S> + ?Sized>(
    application: &A,
    rhs: BlockVectorView<'_, S>,
    options: SparseSolveOptions,
    scratch: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let column_options = options.with_execution_policy(ExecutionPolicy::Sequential);
    crate::execution::stage_columns(
        (application.dimension(), rhs.right_hand_sides()),
        scratch,
        stride(application)?,
        options.execution_policy(),
        options.workers(),
        execution,
        &|column, candidate, work, execution| {
            application.apply_column(
                rhs.column(column)?,
                candidate,
                work,
                column_options,
                execution,
            )
        },
    )
}

/// Generates a factor family's solve forms and inverse-operator protocols from
/// its private `dimension`, `solve_limits` and `solve_column`.
macro_rules! impl_solves {
    ($name:ident) => {
        impl<S: scioxide_core::NumericScalar> crate::factor_solve::Application<S>
            for crate::factor_solve::Oriented<'_, $name<S>>
        {
            fn dimension(&self) -> usize {
                self.factor.dimension()
            }
            fn work_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(self.factor.dimension())
            }
            fn staging_limits(&self) -> Result<(usize, usize), scioxide_core::SciOxideError> {
                self.factor.solve_limits()
            }
            fn apply_column(
                &self,
                rhs: &[S],
                candidate: &mut [S],
                work: &mut [S],
                options: crate::SparseSolveOptions,
                execution: scioxide_core::ExecutionContext<'_>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                self.factor.solve_column(
                    rhs,
                    candidate,
                    work,
                    self.operation,
                    crate::factor_arithmetic::Arithmetic(options.accuracy()),
                    execution,
                )
            }
        }
        impl<S: scioxide_core::NumericScalar> $name<S> {
            fn oriented(
                &self,
                operation: crate::SparseOperation,
            ) -> crate::factor_solve::Oriented<'_, Self> {
                crate::factor_solve::Oriented {
                    factor: self,
                    operation,
                }
            }
            /// Returns the scratch size of solves with `rhs` right-hand sides.
            ///
            /// Owned scratch counts toward the byte limit of the analysis, if
            /// the factor has one, together with the factor and its workspace.
            /// A borrowed slice is checked only for its length.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if the size overflows `usize`.
            pub fn workspace_requirements(
                &self,
                rhs: usize,
            ) -> Result<crate::SparseSolveRequirements, scioxide_core::SciOxideError> {
                crate::factor_solve::requirements(
                    &self.oriented(crate::SparseOperation::Normal),
                    rhs,
                )
            }
            /// Creates a workspace that owns scratch for `rhs` right-hand sides.
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` if the scratch exceeds the byte limit of the
            ///   analysis.
            /// - `SizeOverflow` if the size overflows `usize`.
            /// - `AllocationFailure` if memory cannot be allocated.
            pub fn workspace(
                &self,
                rhs: usize,
            ) -> Result<crate::SparseSolveWorkspace<'static, S>, scioxide_core::SciOxideError> {
                crate::factor_solve::workspace(&self.oriented(crate::SparseOperation::Normal), rhs)
            }
            /// Solves `op(A) x = b` for one right-hand side `b = rhs` and writes
            /// `x` to `output`.
            ///
            /// `operation` selects `op(A)` as `A`, `Aᵀ` or `Aᴴ`. The solve counts
            /// as one iteration toward the limits of `options` and `scope`, where
            /// `scope` is an execution context or the solve context of an
            /// enclosing solver (see [Execution](crate#execution)).
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `rhs` or `output` does not have length `n`,
            ///   or the scratch in `workspace` is too short.
            /// - `NonFinite` if `rhs` contains NaN or infinity.
            /// - `Singular` if the factor has a zero pivot, which only an `LDLᴴ`
            ///   factor can have.
            /// - `NumericalInstability` if an intermediate value overflows.
            /// - `ResourceLimit` if owned scratch exceeds the byte limit.
            /// - `ConvergenceFailure` if the iteration limit has been reached.
            /// - `ExecutionStopped` or `CapabilityUnavailable` from the
            ///   [execution context](scioxide_core#execution-contexts).
            pub fn solve_into<'s, 'c>(
                &self,
                rhs: &[S],
                operation: crate::SparseOperation,
                options: crate::SparseSolveOptions,
                output: &mut [S],
                workspace: &mut crate::SparseSolveWorkspace<'_, S>,
                scope: impl Into<scioxide_core::SolveScope<'s, 'c>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_into(
                    &self.oriented(operation),
                    rhs,
                    options,
                    output,
                    workspace,
                    scope,
                )
            }
            // Owned solve with the default options in a fresh root scope.
            fn solve_owned(
                &self,
                rhs: &[S],
                operation: crate::SparseOperation,
            ) -> Result<alloc::vec::Vec<S>, scioxide_core::SciOxideError> {
                let mut output = scioxide_core::backend::try_filled(
                    self.dimension(),
                    S::ZERO,
                    scioxide_core::quantity!("solution"),
                )?;
                self.solve_into(
                    rhs,
                    operation,
                    crate::SparseSolveOptions::new(),
                    &mut output,
                    &mut self.workspace(1)?,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            /// Solves `A x = b` for `b = rhs` and returns `x` in a new vector.
            ///
            /// This is [`solve_into`](Self::solve_into) with
            /// [`SparseOperation::Normal`](crate::SparseOperation::Normal) and
            /// the default [`SparseSolveOptions`](crate::SparseSolveOptions),
            /// run in a fresh scope (see [Execution](crate#execution)).
            ///
            /// # Errors
            ///
            /// - The errors of [`solve_into`](Self::solve_into).
            /// - `AllocationFailure` if `x` or the scratch cannot be allocated.
            pub fn solve(
                &self,
                rhs: &[S],
            ) -> Result<alloc::vec::Vec<S>, scioxide_core::SciOxideError> {
                self.solve_owned(rhs, crate::SparseOperation::Normal)
            }
            /// Solves `Aᵀ x = b` (transpose without conjugation) and returns `x`
            /// in a new vector, like [`solve`](Self::solve).
            ///
            /// # Errors
            ///
            /// - The errors of [`solve`](Self::solve).
            pub fn solve_transpose(
                &self,
                rhs: &[S],
            ) -> Result<alloc::vec::Vec<S>, scioxide_core::SciOxideError> {
                self.solve_owned(rhs, crate::SparseOperation::Transpose)
            }
            /// Solves `Aᴴ x = b` (conjugate transpose) and returns `x` in a new
            /// vector, like [`solve`](Self::solve).
            ///
            /// # Errors
            ///
            /// - The errors of [`solve`](Self::solve).
            pub fn solve_adjoint(
                &self,
                rhs: &[S],
            ) -> Result<alloc::vec::Vec<S>, scioxide_core::SciOxideError> {
                self.solve_owned(rhs, crate::SparseOperation::Adjoint)
            }
            /// Solves `op(A) X = B` for the columns of a column-major block `B`
            /// and writes `X` to `output`.
            ///
            /// Each column counts as one iteration. Under a parallel execution
            /// policy in `options`, columns are solved in parallel. `output` is
            /// written only after every column has been solved.
            ///
            /// # Errors
            ///
            /// - The errors of [`solve_into`](Self::solve_into).
            /// - `DimensionMismatch` also if `rhs` and `output` have different
            ///   numbers of columns.
            pub fn solve_block_into<'s, 'c>(
                &self,
                rhs: scioxide_core::BlockVectorView<'_, S>,
                operation: crate::SparseOperation,
                options: crate::SparseSolveOptions,
                output: &mut scioxide_core::BlockVectorViewMut<'_, S>,
                workspace: &mut crate::SparseSolveWorkspace<'_, S>,
                scope: impl Into<scioxide_core::SolveScope<'s, 'c>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_block_into(
                    &self.oriented(operation),
                    rhs,
                    options,
                    output,
                    workspace,
                    scope,
                )
            }
        }
        impl<S: scioxide_core::NumericScalar> scioxide_core::LinearOperator<S> for $name<S> {
            fn rows(&self) -> usize {
                self.dimension()
            }
            fn cols(&self) -> usize {
                self.dimension()
            }
            fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(self.workspace_requirements(1)?.scalar_len())
            }
            fn apply_into(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_slices(
                    &self.oriented(crate::SparseOperation::Normal),
                    input,
                    output,
                    workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            }
        }
        impl<S: scioxide_core::NumericScalar> scioxide_core::AdjointOperator<S> for $name<S> {
            fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(self.workspace_requirements(1)?.scalar_len())
            }
            fn apply_adjoint_into(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_slices(
                    &self.oriented(crate::SparseOperation::Adjoint),
                    input,
                    output,
                    workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            }
        }
        impl<S: scioxide_core::NumericScalar> scioxide_core::BlockOperator<S> for $name<S> {
            fn apply_block_workspace_len(
                &self,
                rhs: usize,
            ) -> Result<usize, scioxide_core::SciOxideError> {
                Ok(self.workspace_requirements(rhs)?.scalar_len())
            }
            fn apply_block_into(
                &self,
                input: scioxide_core::BlockVectorView<'_, S>,
                output: &mut scioxide_core::BlockVectorViewMut<'_, S>,
                workspace: &mut [S],
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_block_slices(
                    &self.oriented(crate::SparseOperation::Normal),
                    input,
                    output,
                    workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            }
        }
    };
}
pub(crate) use impl_solves;

/// Generates a preconditioner's inherent workspace and application forms and
/// the core `Preconditioner` protocol from its `Application` implementation.
macro_rules! impl_applications {
    ($name:ty) => {
        impl<S: scioxide_core::NumericScalar> $name {
            /// Returns the scratch size of applications to `rhs` right-hand sides.
            ///
            /// Owned scratch counts toward the byte limit of the
            /// preconditioner, if it has one. A borrowed slice is checked only
            /// for its length.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if the size overflows `usize`.
            pub fn workspace_requirements(
                &self,
                rhs: usize,
            ) -> Result<crate::SparseSolveRequirements, scioxide_core::SciOxideError> {
                crate::factor_solve::requirements(self, rhs)
            }
            /// Creates a workspace that owns scratch for `rhs` right-hand sides.
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` if the scratch exceeds the byte limit of the
            ///   preconditioner.
            /// - `SizeOverflow` if the size overflows `usize`.
            /// - `AllocationFailure` if memory cannot be allocated.
            pub fn workspace(
                &self,
                rhs: usize,
            ) -> Result<crate::SparseSolveWorkspace<'static, S>, scioxide_core::SciOxideError> {
                crate::factor_solve::workspace(self, rhs)
            }
            /// Applies the preconditioner to `rhs` and returns the result in a
            /// new vector.
            ///
            /// This is [`apply_into`](Self::apply_into) with the default
            /// [`SparseSolveOptions`](crate::SparseSolveOptions), run in a fresh
            /// scope (see [Execution](crate#execution)).
            ///
            /// # Errors
            ///
            /// - The errors of [`apply_into`](Self::apply_into).
            /// - `AllocationFailure` if the result or the scratch cannot be
            ///   allocated.
            pub fn apply(
                &self,
                rhs: &[S],
            ) -> Result<alloc::vec::Vec<S>, scioxide_core::SciOxideError> {
                let mut output = scioxide_core::backend::try_filled(
                    crate::factor_solve::Application::<S>::dimension(self),
                    S::ZERO,
                    scioxide_core::quantity!("preconditioned"),
                )?;
                self.apply_into(
                    rhs,
                    crate::SparseSolveOptions::new(),
                    &mut output,
                    &mut self.workspace(1)?,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            /// Applies the preconditioner to `rhs` and writes the result to
            /// `output`.
            ///
            /// `options` selects the accuracy, the execution policy and the
            /// iteration limits. The application counts toward the limits of
            /// `options` and `scope`, where `scope` is an execution context or the
            /// solve context of an enclosing solver (see
            /// [Execution](crate#execution)).
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `rhs` or `output` has the wrong length, or
            ///   the scratch in `workspace` is too short.
            /// - `NonFinite` if `rhs` contains NaN or infinity.
            /// - `NumericalInstability` if an intermediate value overflows.
            /// - `ResourceLimit` if owned scratch exceeds the byte limit.
            /// - `ConvergenceFailure` if the iteration limit has been reached.
            /// - `ExecutionStopped` or `CapabilityUnavailable` from the
            ///   [execution context](scioxide_core#execution-contexts).
            /// - Any error returned by a solver or preconditioner that you
            ///   supplied to this preconditioner.
            pub fn apply_into<'s, 'c>(
                &self,
                rhs: &[S],
                options: crate::SparseSolveOptions,
                output: &mut [S],
                workspace: &mut crate::SparseSolveWorkspace<'_, S>,
                scope: impl Into<scioxide_core::SolveScope<'s, 'c>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_into(self, rhs, options, output, workspace, scope)
            }
            /// Applies the preconditioner to each column of a column-major block
            /// `rhs` and writes the results to `output`.
            ///
            /// Under a parallel execution policy in `options`, columns are
            /// processed in parallel. `output` is written only after every column
            /// is done.
            ///
            /// # Errors
            ///
            /// - The errors of [`apply_into`](Self::apply_into).
            /// - `DimensionMismatch` also if `rhs` and `output` have different
            ///   numbers of columns.
            pub fn apply_block_into<'s, 'c>(
                &self,
                rhs: scioxide_core::BlockVectorView<'_, S>,
                options: crate::SparseSolveOptions,
                output: &mut scioxide_core::BlockVectorViewMut<'_, S>,
                workspace: &mut crate::SparseSolveWorkspace<'_, S>,
                scope: impl Into<scioxide_core::SolveScope<'s, 'c>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                crate::factor_solve::apply_block_into(self, rhs, options, output, workspace, scope)
            }
        }
        crate::factor_solve::impl_preconditioner!($name, |this| this);
    };
}
pub(crate) use impl_applications;

/// Generates the core `Preconditioner` protocol from an `Application` borrowed
/// from `self` by `|this| application`. Preconditioners apply themselves; a
/// factor family whose solve forms come from `impl_solves` applies its normal
/// oriented solve.
macro_rules! impl_preconditioner {
    ($name:ty, |$this:ident| $application:expr) => {
        impl<S: scioxide_core::NumericScalar> scioxide_core::Preconditioner<S> for $name {
            fn dimension(&self) -> usize {
                let $this = self;
                crate::factor_solve::Application::<S>::dimension($application)
            }
            fn precondition_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                let $this = self;
                Ok(crate::factor_solve::requirements($application, 1)?.scalar_len())
            }
            fn precondition_into(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
            ) -> Result<(), scioxide_core::SciOxideError> {
                let $this = self;
                crate::factor_solve::apply_slices(
                    $application,
                    input,
                    output,
                    workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            }
            fn precondition_into_scoped(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                let $this = self;
                crate::factor_solve::apply_slices(
                    $application,
                    input,
                    output,
                    workspace,
                    context.execution(),
                )
            }
        }
    };
}
pub(crate) use impl_preconditioner;
