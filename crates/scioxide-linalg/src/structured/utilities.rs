//! Exact norms, dense materialization and multi-column application of linear operators.
use super::direct::{admit, identity, narrow, size, wide};
use crate::kernel::CANDIDATE;
use crate::{
    LinalgExecutionContext, LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{LinearOperator, NumericScalar, RealScalar, ScaledNorm, SciOxideError};
/// Matrix norm computed by [`MatrixView::norm`] and [`operator_norm`].
#[derive(Debug, Clone, Copy)]
pub enum MatrixNorm {
    /// `‖A‖₁ = maxⱼ Σᵢ |aᵢⱼ|`, the largest column sum.
    One,
    /// `‖A‖∞ = maxᵢ Σⱼ |aᵢⱼ|`, the largest row sum.
    Infinity,
    /// `‖A‖_F = (Σᵢⱼ |aᵢⱼ|²)^(1/2)`, computed as a scaled sum of squares
    /// that does not overflow.
    Frobenius,
}

/// Describes an operator job, for sizing a [`LinalgWorkspace`].
///
/// The jobs are an exact norm, dense materialization, multi-column
/// application, and Kronecker or block diagonal solves. Pass the request to
/// [`LinalgWorkspace::workspace_requirements`] or
/// [`LinalgWorkspace::workspace`]. A request stores sizes and the byte limit
/// of `options`, not the operator.
///
/// ```
/// use scioxide_linalg::{
///     LinalgExecutionOptions, LinalgWorkspace, Matrix, OperatorWorkspaceRequest,
/// };
/// let matrix = Matrix::<f64>::identity(3)?;
/// let operator = matrix.view().try_finite()?;
/// let request = OperatorWorkspaceRequest::norm(&operator, &LinalgExecutionOptions::new())?;
/// let workspace = LinalgWorkspace::<f64>::workspace(&request)?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OperatorWorkspaceRequest<S: NumericScalar> {
    scalar: core::marker::PhantomData<S>,
    scalars: usize,
    options: crate::LinalgExecutionOptions,
}
#[derive(Clone, Copy)]
enum OperatorJob {
    Norm,
    Materialize,
    Block,
}
impl<S: NumericScalar> OperatorWorkspaceRequest<S> {
    /// Requests the workspace of [`operator_norm_scoped`] for `op`, without applying it.
    ///
    /// # Errors
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - Any error from the workspace query of `op`.
    pub fn norm(
        op: &dyn LinearOperator<S>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::new(op, 1, OperatorJob::Norm, options)
    }
    /// Requests the workspace of [`materialize_operator_into`] for `op`.
    ///
    /// # Errors
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - Any error from the workspace query of `op`.
    pub fn materialize(
        op: &dyn LinearOperator<S>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::new(op, 1, OperatorJob::Materialize, options)
    }
    /// Requests the workspace of [`apply_operator_matrix_into`] for `columns` input columns.
    ///
    /// # Errors
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - Any error from the workspace query of `op`.
    pub fn block(
        op: &dyn LinearOperator<S>,
        columns: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::new(op, columns, OperatorJob::Block, options)
    }
    /// Requests the workspace of a Kronecker solve with `inverses = [A⁻¹, B⁻¹]`.
    ///
    /// The solve is [`KroneckerOperator::solve_into`](crate::KroneckerOperator::solve_into).
    ///
    /// # Errors
    /// - `DimensionMismatch` if a factor is not square or an inverse does not
    ///   have the shape of its factor.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - Any error from the workspace queries of the operators.
    pub fn kronecker_solve(
        product: &crate::KroneckerOperator<'_, S>,
        inverses: [&dyn LinearOperator<S>; 2],
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            scalar: core::marker::PhantomData,
            scalars: product.solve_scratch(inverses)?,
            options: *options,
        })
    }
    /// Requests the workspace of a block diagonal solve with one inverse per block.
    ///
    /// The solve is
    /// [`BlockDiagonalOperator::solve_into`](crate::BlockDiagonalOperator::solve_into).
    ///
    /// # Errors
    /// - `DimensionMismatch` if the number of inverses differs from the number
    ///   of blocks, a block is not square, or an inverse does not have the
    ///   shape of its block.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - Any error from the workspace queries of the operators.
    pub fn block_diagonal_solve(
        operator: &crate::BlockDiagonalOperator<'_, S>,
        inverses: &[&dyn LinearOperator<S>],
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            scalar: core::marker::PhantomData,
            scalars: operator.solve_scratch(inverses)?,
            options: *options,
        })
    }
    fn new(
        op: &dyn LinearOperator<S>,
        columns: usize,
        job: OperatorJob,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let (m, n) = (op.rows(), op.cols());
        let scalars = if m == 0 || n == 0 || columns == 0 {
            0
        } else {
            let provider = op.apply_workspace_len()?;
            let base = add(add(n, m)?, provider)?;
            match job {
                OperatorJob::Norm => add(base, m)?,
                OperatorJob::Materialize => add(
                    base,
                    size(m, n, scioxide_core::quantity!("materialization"))?,
                )?,
                OperatorJob::Block => {
                    size(base, columns, scioxide_core::quantity!("block_workspace"))?
                }
            }
        };
        Ok(Self {
            scalar: core::marker::PhantomData,
            scalars,
            options: *options,
        })
    }
}
impl<S: NumericScalar> crate::workspace::Request<S> for OperatorWorkspaceRequest<S> {
    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        let mut requirements = LinalgRequirements::new(self.options.byte_limit());
        *requirements.scalar_mut(CANDIDATE) = self.scalars;
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
}
impl<S: NumericScalar> MatrixView<'_, S> {
    /// Computes the 1-, ∞- or Frobenius norm of the dense matrix `A`.
    ///
    /// `MatrixNorm::One` is the largest column sum `maxⱼ Σᵢ |aᵢⱼ|`,
    /// `MatrixNorm::Infinity` the largest row sum, which is the 1-norm of
    /// `Aᵀ`, and `MatrixNorm::Frobenius` is `(Σᵢⱼ |aᵢⱼ|²)^(1/2)`. `|aᵢⱼ|` is the
    /// complex modulus. Strides and the conjugation flag of the view are
    /// honored, and an empty matrix has norm zero.
    ///
    /// The 1- and ∞-norms check the entries for finiteness and then sum each
    /// column in working precision, summing a column again in extended range
    /// when its ordinary sum overflows. The Frobenius norm is the
    /// overflow-free Euclidean norm (see
    /// [`VectorView::norm2`](crate::VectorView::norm2)) of all entries when
    /// they are contiguous in memory; otherwise the norms of the rows or of the
    /// columns, whichever are fewer, are combined in a [`ScaledNorm`]. The cost
    /// is `O(m n)` for an `m × n` matrix, `O(1)` if it is empty, and nothing is
    /// allocated. Use [`operator_norm`] for a matrix-free operator.
    ///
    /// # Accuracy
    ///
    /// A 1- or ∞-norm has the error of a sum of `m` or `n` nonnegative terms
    /// in working precision. A Frobenius norm summed as `k` vectors of `p`
    /// real components each has relative error at most
    /// `γ_(2p+5) + γ_(8k+8) + γ_(2p+5) γ_(8k+8)` plus subnormal rounding, where
    /// `γ_j = j u / (1 − j u)` with the unit roundoff `u`. It can differ from
    /// [`operator_norm`] in the last bits because the order of accumulation
    /// differs.
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, MatrixNorm};
    /// let a = Matrix::from_rows(&[[1.0_f64, -2.0], [3.0, 4.0]])?;
    /// assert_eq!(a.view().norm(MatrixNorm::One)?, 6.0);
    /// assert_eq!(a.view().norm(MatrixNorm::Infinity)?, 7.0);
    /// assert!((a.view().norm(MatrixNorm::Frobenius)? - 30.0_f64.sqrt()).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if the norm exceeds the largest
    ///   finite value of the scalar type.
    pub fn norm(&self, kind: MatrixNorm) -> Result<S::Real, SciOxideError> {
        let one_norm = |view: MatrixView<'_, S>| {
            crate::diagnostics::matrix_one_norm(view.try_finite()?.as_view())?.round()
        };
        match kind {
            MatrixNorm::One => one_norm(*self),
            MatrixNorm::Infinity => one_norm(self.transpose()),
            MatrixNorm::Frobenius => {
                if self.is_empty() {
                    return Ok(S::Real::ZERO);
                }
                let context = crate::KernelContext::sequential_native();
                // Conjugation does not change a modulus.
                let stored = if self.is_conjugated() {
                    self.conjugate()
                } else {
                    *self
                };
                if let Some(entries) = stored.as_contiguous_slice() {
                    return crate::reduction::l2_norm_with_context(
                        crate::VectorView::from_slice(entries),
                        context,
                    );
                }
                let by_rows = self.rows() <= self.cols();
                let count = if by_rows { self.rows() } else { self.cols() };
                (0..count)
                    .try_fold(ScaledNorm::new(), |mut norm, index| {
                        let vector = if by_rows {
                            stored.row(index)?
                        } else {
                            stored.col(index)?
                        };
                        norm.add(crate::reduction::l2_norm_with_context(vector, context)?)?;
                        Ok::<_, SciOxideError>(norm)
                    })?
                    .finish()
            }
        }
    }
}

fn add(a: usize, b: usize) -> Result<usize, SciOxideError> {
    a.checked_add(b)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("operator_workspace")))
}

/// Computes the 1-, ∞- or Frobenius norm of a linear operator exactly.
///
/// The method is described on [`operator_norm_scoped`], which takes a
/// workspace and an execution context. To estimate the 1-norm of a large
/// square operator from a few products, use
/// [`OneNormEstimator`](crate::OneNormEstimator).
///
/// ```
/// use scioxide_linalg::{Matrix, MatrixNorm, operator_norm};
/// let a = Matrix::<f64>::identity(2)?;
/// assert_eq!(
///     operator_norm(
///         &a.view().try_finite()?,
///         MatrixNorm::One)?,
///     1.0
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// The same conditions as [`operator_norm_scoped`].
pub fn operator_norm<T: NumericScalar>(
    op: &dyn LinearOperator<T>,
    kind: MatrixNorm,
) -> Result<T::Real, SciOxideError> {
    operator_norm_with_options(op, kind, &crate::LinalgExecutionOptions::default())
}

/// Runs [`operator_norm`](operator_norm) with `options`, which set the product accuracy and the
/// limits described in [Execution](crate#execution).
///
/// # Errors
///
/// The errors of [`operator_norm`](operator_norm), and limit errors from `options` as described in
/// [Execution](crate#execution).
pub fn operator_norm_with_options<T: NumericScalar>(
    op: &dyn LinearOperator<T>,
    kind: MatrixNorm,
    options: &crate::LinalgExecutionOptions,
) -> Result<T::Real, SciOxideError> {
    operator_norm_scoped(
        op,
        kind,
        options,
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Computes the 1-, ∞- or Frobenius norm of a linear operator with a workspace and context.
///
/// The `m × n` operator `A` is applied to each unit vector `eⱼ`, and the
/// norm is accumulated from the columns `A eⱼ`. This takes `n` applications
/// of `op`, works for rectangular operators, and gives the exact norm up to
/// rounding. `op` must be linear and return the same result for the same
/// input. Each column counts as one iteration toward the iteration budget.
/// The workspace is sized by [`OperatorWorkspaceRequest::norm`]. An empty
/// operator has norm zero.
///
/// ```
/// use scioxide_linalg::{LinalgWorkspace, Matrix, MatrixNorm, operator_norm_scoped};
/// // Column sums of |A| for A = [1 −2; 3 4] are 4 and 6.
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, -2.0, 3.0, 4.0])?;
/// let finite = a.view().try_finite()?;
/// let norm = operator_norm_scoped(
///     &finite,
///     MatrixNorm::One,
///     &scioxide_linalg::LinalgExecutionOptions::new(),
///     &mut LinalgWorkspace::new(),
///     scioxide_core::ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(norm, 6.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - `NonFinite` if `op` returns NaN or an infinity.
/// - `NumericalInstability` if the norm overflows.
/// - `ConvergenceFailure` if the iteration budget runs out.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
/// - Any error returned by `op`.
/// - Workspace errors as described in
///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub fn operator_norm_scoped<'scope, 'capability, T: NumericScalar>(
    op: &dyn LinearOperator<T>,
    kind: MatrixNorm,
    options: &crate::LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, T>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<T::Real, SciOxideError> {
    workspace.begin_attempt();
    let context = context.into().with_options(options)?;
    context.execution().check()?;
    let requirements = LinalgWorkspace::<T>::workspace_requirements(
        &OperatorWorkspaceRequest::norm(op, options)?,
    )?;
    workspace.try_reserve(&requirements)?;
    workspace.run_scope(
        context,
        identity(
            scioxide_core::algorithm_id!("linalg", "operator_norm", 1),
            op.rows(),
        ),
        options.solve_budgets(),
        |workspace, kernels, scope| {
            if op.rows() == 0 || op.cols() == 0 {
                return Ok(T::Real::ZERO);
            }
            let workspace = &mut workspace.scalar_mut(CANDIDATE)[..requirements.scalar(CANDIDATE)];
            // Exact norms of a matrix-free operator by probing it with every unit
            // vector: column sums give the 1-norm, accumulated absolute row sums the
            // infinity norm, and a scaled sum of squares (real and imaginary parts
            // separately) the Frobenius norm. Costs `cols` applications.
            let (basis, work) = crate::workspace::split_mut(workspace, op.cols())?;
            let (column, work) = crate::workspace::split_mut(work, op.rows())?;
            let (rows, work) = crate::workspace::split_mut(work, op.rows())?;
            basis.fill(T::ZERO);
            rows.fill(T::ZERO);
            let mut one = T::Real::ZERO;
            let mut frobenius = ScaledNorm::new();
            for j in 0..op.cols() {
                basis[j] = T::ONE;
                admit(kernels.execution(), scope)?;
                op.apply_into_scoped(
                    basis,
                    column,
                    &mut work[..op.apply_workspace_len()?],
                    &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                )
                .and_then(scioxide_core::Invocation::require)?;
                basis[j] = T::ZERO;
                crate::checked::provider_output(column)?;
                let mut total = wide(T::ZERO)?;
                for (row, &v) in rows.iter_mut().zip(column.iter()) {
                    match kind {
                        MatrixNorm::One => {
                            total = total.add(wide(T::from_real(v.abs()))?)?;
                        }
                        MatrixNorm::Infinity => {
                            *row = narrow(wide(*row)?.add(wide(T::from_real(v.abs()))?)?)?;
                        }
                        MatrixNorm::Frobenius => {
                            frobenius.add(v.real_component())?;
                            frobenius.add(v.imaginary_component())?;
                        }
                    }
                }
                if matches!(kind, MatrixNorm::One) {
                    one = one.max(narrow::<T>(total)?.real_component());
                }
            }
            kernels.execution().check()?;
            match kind {
                MatrixNorm::One => Ok(one),
                MatrixNorm::Infinity => Ok(rows
                    .iter()
                    .fold(T::Real::ZERO, |v, x| v.max(x.real_component()))),
                MatrixNorm::Frobenius => frobenius.finish(),
            }
        },
    )
}

/// Writes the dense matrix of a linear operator into `output`.
///
/// Column `j` of `output` is `A eⱼ`, one application of `op` per column. The
/// workspace is sized by [`OperatorWorkspaceRequest::materialize`] and holds
/// the whole `m × n` result before it is copied to `output`. Each column
/// counts as one iteration toward the iteration budget.
///
/// ```
/// use scioxide_linalg::{LinalgWorkspace, Matrix, materialize_operator_into};
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let finite = a.view().try_finite()?;
/// let mut dense = Matrix::zeros(2, 2)?;
/// materialize_operator_into(
///     &finite,
///     &scioxide_linalg::LinalgExecutionOptions::new(),
///     dense.view_mut(),
///     &mut LinalgWorkspace::new(),
///     scioxide_core::ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!(dense, a);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - `DimensionMismatch` if `output` does not have the shape of `op`.
/// - `NonFinite` if `op` returns NaN or an infinity.
/// - `ConvergenceFailure` if the iteration budget runs out.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
/// - Any error returned by `op`.
/// - Workspace errors as described in
///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub fn materialize_operator_into<'scope, 'capability, T: NumericScalar>(
    op: &dyn LinearOperator<T>,
    options: &crate::LinalgExecutionOptions,
    mut output: MatrixViewMut<'_, T>,
    workspace: &mut LinalgWorkspace<'_, T>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    workspace.begin_attempt();
    let context = context.into().with_options(options)?;
    context.execution().check()?;
    if output.rows() != op.rows() || output.cols() != op.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [op.rows(), op.cols()],
            [output.rows(), output.cols()],
        ));
    }
    let requirements = LinalgWorkspace::<T>::workspace_requirements(
        &OperatorWorkspaceRequest::materialize(op, options)?,
    )?;
    workspace.try_reserve(&requirements)?;
    workspace.run_scope(
        context,
        identity(
            scioxide_core::algorithm_id!("linalg", "materialize_operator", 1),
            op.rows(),
        ),
        options.solve_budgets(),
        |workspace, kernels, scope| {
            if op.rows() == 0 || op.cols() == 0 {
                return Ok(());
            }
            let scratch = &mut workspace.scalar_mut(CANDIDATE)[..requirements.scalar(CANDIDATE)];
            let (candidate, w) = crate::workspace::split_mut(scratch, op.rows() * op.cols())?;
            let (basis, w) = crate::workspace::split_mut(w, op.cols())?;
            let (column, w) = crate::workspace::split_mut(w, op.rows())?;
            basis.fill(T::ZERO);
            for j in 0..op.cols() {
                basis[j] = T::ONE;
                admit(kernels.execution(), scope)?;
                op.apply_into_scoped(
                    basis,
                    column,
                    &mut w[..op.apply_workspace_len()?],
                    &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                )
                .and_then(scioxide_core::Invocation::require)?;
                basis[j] = T::ZERO;
                crate::checked::provider_output(column)?;
                for (i, &v) in column.iter().enumerate() {
                    candidate[i * op.cols() + j] = v;
                }
            }
            kernels.execution().check()?;
            for i in 0..op.rows() {
                for j in 0..op.cols() {
                    output.set_element(i, j, candidate[i * op.cols() + j]);
                }
            }
            Ok(())
        },
    )
}

/// Computes `Y = A X` column by column and writes it to `output`.
///
/// `op` is applied to each column of `input`. With a parallel execution
/// policy in `options`, columns are split among up to 32 workers. The
/// workspace is sized by
/// [`OperatorWorkspaceRequest::block`]. Each column counts as one iteration
/// toward the iteration budget; see
/// [Solve accounting](scioxide_core#solve-accounting). If `op` has no columns,
/// `output` is set to zero.
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # Errors
/// - `DimensionMismatch` if `input` does not have `n` rows or `output` is not
///   `m × k` for `k` input columns.
/// - `NonFinite` if `op` returns NaN or an infinity.
/// - `ConvergenceFailure` if the iteration budget runs out.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
/// - Any error returned by `op`.
/// - Workspace errors as described in
///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
pub fn apply_operator_matrix_into<'i, 'scope, 'capability, T: NumericScalar>(
    op: &(dyn LinearOperator<T> + Sync),
    input: impl Into<MatrixView<'i, T>>,
    options: &crate::LinalgExecutionOptions,
    mut output: MatrixViewMut<'_, T>,
    workspace: &mut LinalgWorkspace<'_, T>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    workspace.begin_attempt();
    let context = context.into().with_options(options)?;
    context.execution().check()?;
    let input = input.into();
    if input.rows() != op.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [op.cols(), input.cols()],
            [input.rows(), input.cols()],
        ));
    }
    if output.rows() != op.rows() || output.cols() != input.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [op.rows(), input.cols()],
            [output.rows(), output.cols()],
        ));
    }
    let columns = input.cols();
    let requirements = LinalgWorkspace::<T>::workspace_requirements(
        &OperatorWorkspaceRequest::block(op, columns, options)?,
    )?;
    let len = requirements.scalar(CANDIDATE);
    workspace.try_reserve(&requirements)?;
    workspace.run_scope(
        context,
        identity(
            scioxide_core::algorithm_id!("linalg", "operator_block_apply", 1),
            op.rows(),
        ),
        options.solve_budgets(),
        |workspace, kernels, scope| {
            // Empty products call no provider; a contracted operator maps to zero.
            if op.rows() == 0 || columns == 0 {
                return Ok(());
            }
            if op.cols() == 0 {
                for i in 0..output.rows() {
                    for j in 0..output.cols() {
                        output.set_element(i, j, T::ZERO);
                    }
                }
                return Ok(());
            }
            let execution = kernels.execution();
            for _ in 0..columns {
                admit(execution, scope)?;
            }
            let stride = len / columns;
            let scratch = &mut workspace.scalar_mut(CANDIDATE)[..len];
            use core::num::NonZeroUsize;
            use scioxide_core::backend::{
                PartitionSlot, PartitionedSolve, with_partitioned_solves,
            };
            let plan = execution.plan(
                options.execution_policy(),
                columns,
                scioxide_core::PlanLimits {
                    work_per_item: NonZeroUsize::MIN,
                    auto_threshold: NonZeroUsize::MIN,
                    worker_limit: NonZeroUsize::new(32).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("operator_workers"))
                    })?,
                },
            );
            let mut children: [PartitionedSolve; 32] =
                core::array::from_fn(|_| PartitionedSolve::default());
            with_partitioned_solves(
                scope,
                &mut children[..plan.worker_count()],
                scioxide_core::SolveBudgets::unbounded(),
                |execution, batch| {
                    let mut slots: [_; 32] = core::array::from_fn(|_| PartitionSlot::empty());
                    let mut remaining = &mut *scratch;
                    for (part, permit) in plan.partitions().zip(batch) {
                        let (chunk, rest) =
                            crate::workspace::split_mut(remaining, part.range().len() * stride)?;
                        slots[part.worker_index()] = PartitionSlot::new((chunk, permit));
                        remaining = rest;
                    }
                    execution.execute_partitioned(
                        plan,
                        &slots[..plan.worker_count()],
                        &|part, (chunk, permit), execution| {
                            permit.run(
                                execution,
                                identity(
                                    scioxide_core::algorithm_id!(
                                        "linalg",
                                        "operator_block_partition",
                                        1
                                    ),
                                    op.rows(),
                                ),
                                |solve| {
                                    for (offset, w) in chunk.chunks_exact_mut(stride).enumerate() {
                                        execution.check()?;
                                        let (x, w) = crate::workspace::split_mut(w, op.cols())?;
                                        let (y, w) = crate::workspace::split_mut(w, op.rows())?;
                                        for (i, v) in x.iter_mut().enumerate() {
                                            *v = input.get_element(i, part.range().start + offset);
                                        }
                                        op.apply_into_scoped(
                                            x,
                                            y,
                                            &mut w[..op.apply_workspace_len()?],
                                            &mut scioxide_core::OperatorContext::with_kernels(
                                                solve, &kernels,
                                            ),
                                        )
                                        .and_then(scioxide_core::Invocation::require)?;
                                        crate::checked::provider_output(y)?;
                                    }
                                    Ok(())
                                },
                            )
                        },
                    )
                },
            )?;
            execution.check()?;
            for (j, w) in scratch.chunks_exact(stride).enumerate() {
                for (i, &v) in w[op.cols()..op.cols() + op.rows()].iter().enumerate() {
                    output.set_element(i, j, v);
                }
            }
            Ok(())
        },
    )
}
