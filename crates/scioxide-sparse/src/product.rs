//! Sparse–sparse products with a prepared pattern, and sparse–dense products.
use crate::helpers::invalid;
use crate::{CsrMatrix, SparseView};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::backend::{
    checked_add, checked_mul, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{
    Accuracy, BlockVectorView, BlockVectorViewMut, Capacity, ExecutionContext, ExecutionPolicy,
    NumericScalar, ProductSum, SciOxideError,
};

/// Selects `op(A)`: the matrix `A`, its transpose `Aᵀ`, or its conjugate
/// transpose `Aᴴ`.
///
/// Products and solves take it to apply `op(A)` without forming the
/// transpose. For real scalars `Transpose` and `Adjoint` are the same.
///
/// ```
/// use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionContext};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseProductOptions, SparseProductRequirements,
///     SparseProductWorkspace, sparse_dense_into,
/// };
///
/// // A = [[1, 2]] is 1 × 2, so Aᵀ · [3] = [3, 6]ᵀ.
/// let a = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0])?;
/// let mut workspace = SparseProductWorkspace::new();
/// workspace.try_reserve(&SparseProductRequirements::sparse_dense(
///     a.view(),
///     SparseOperation::Transpose,
///     1,
/// )?)?;
/// let mut output = [0.0; 2];
/// sparse_dense_into(
///     a.view(),
///     SparseOperation::Transpose,
///     BlockVectorView::new(&[3.0], 1, 1)?,
///     SparseProductOptions::new(),
///     &mut BlockVectorViewMut::new(&mut output, 2, 1)?,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(output, [3.0, 6.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseOperation {
    /// `op(A) = A`.
    Normal,
    /// `op(A) = Aᵀ`, the transpose without conjugation.
    Transpose,
    /// `op(A) = Aᴴ`, the conjugate transpose.
    Adjoint,
}
/// Holds the accuracy and the execution policy of sparse–sparse and
/// sparse–dense products.
///
/// The [`ExecutionPolicy`] allows at most 64 workers. Workers split the rows
/// of a [`SparseProductPlan`] product or the columns of a
/// [`sparse_dense_into`] product; see [Execution](crate#execution).
///
/// ```
/// use scioxide_core::{Accuracy, ExecutionPolicy};
/// use scioxide_sparse::SparseProductOptions;
///
/// let options = SparseProductOptions::new().with_accuracy(Accuracy::Exact);
/// assert_eq!(options.accuracy(), Accuracy::Exact);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseProductOptions {
    accuracy: Accuracy,
    execution: ExecutionPolicy,
}
impl Default for SparseProductOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseProductOptions {
    /// Creates options with [`Accuracy::Standard`] and
    /// [`ExecutionPolicy::Sequential`].
    pub const fn new() -> Self {
        Self {
            accuracy: Accuracy::Standard,
            execution: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the accuracy. With [`Accuracy::Exact`] each output entry is
    /// accumulated exactly and rounded once.
    pub const fn with_accuracy(mut self, accuracy: Accuracy) -> Self {
        self.accuracy = accuracy;
        self
    }
    /// Returns the accuracy.
    pub const fn accuracy(self) -> Accuracy {
        self.accuracy
    }
    /// Sets the execution policy. Workers come from the execution context;
    /// none are created otherwise.
    pub const fn with_execution_policy(mut self, value: ExecutionPolicy) -> Self {
        self.execution = value;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution
    }
    const fn workers(self) -> NonZeroUsize {
        crate::execution::policy_workers(self.execution)
    }
}
/// Computes `Y = op(A) X` for a sparse matrix `A` and a dense block of
/// column vectors `X`, writing `Y` to `output`.
///
/// `A` may have any storage format. Each column of `Y` is a sequence of
/// sparse inner products with the accuracy in `options` (see
/// [Accuracy](crate#accuracy)), and columns may run in parallel. The
/// workspace must already hold [`SparseProductRequirements::sparse_dense`]; it is checked,
/// not grown, so the call does not allocate.
///
/// ```
/// use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseProductOptions, SparseProductRequirements,
///     SparseProductWorkspace, sparse_dense_into,
/// };
///
/// // A = [[1, 2], [0, 3]] times X = [1, 1]ᵀ gives [3, 3]ᵀ.
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0, 3.0])?;
/// let (b, mut output) = (BlockVectorView::new(&[1.0, 1.0], 2, 1)?, [0.0; 2]);
/// let mut workspace = SparseProductWorkspace::new();
/// let unreserved = sparse_dense_into(
///     a.view(),
///     SparseOperation::Normal,
///     b,
///     SparseProductOptions::new(),
///     &mut BlockVectorViewMut::new(&mut output, 2, 1)?,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// );
/// assert_eq!(unreserved.unwrap_err().kind(), SciOxideErrorKind::DimensionMismatch);
/// assert_eq!(output, [0.0; 2]);
/// workspace.try_reserve(&SparseProductRequirements::sparse_dense(
///     a.view(),
///     SparseOperation::Normal,
///     1,
/// )?)?;
/// sparse_dense_into(
///     a.view(),
///     SparseOperation::Normal,
///     b,
///     SparseProductOptions::new(),
///     &mut BlockVectorViewMut::new(&mut output, 2, 1)?,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(output, [3.0, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `input` does not have as many rows as `op(A)` has
///   columns, `output` is not `op(A)`'s rows by `input`'s columns, or the
///   workspace is smaller than the requirement.
/// - `NonFinite` if `input` has a NaN or infinite component.
/// - `NumericalInstability` if an output entry overflows.
/// - `ExecutionStopped`, `CapabilityUnavailable`, `CapabilityFailure` or
///   `ExecutionFailure` if the execution context cancels or fails the work.
pub fn sparse_dense_into<S: NumericScalar>(
    matrix: SparseView<'_, S>,
    operation: SparseOperation,
    input: BlockVectorView<'_, S>,
    options: SparseProductOptions,
    output: &mut BlockVectorViewMut<'_, S>,
    workspace: &mut SparseProductWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let transpose = operation != SparseOperation::Normal;
    let (rows, cols) = if transpose {
        (matrix.cols(), matrix.rows())
    } else {
        (matrix.rows(), matrix.cols())
    };
    crate::helpers::exact(input.rows(), cols)?;
    crate::helpers::shape(
        (output.rows(), output.right_hand_sides()),
        (rows, input.right_hand_sides()),
    )?;
    let requirements =
        SparseProductRequirements::sparse_dense(matrix, operation, input.right_hand_sides())?;
    workspace.require(&requirements)?;
    crate::helpers::finite(input.as_slice())?;
    context.check()?;
    if requirements.values == 0 {
        return Ok(());
    }
    crate::execution::columns(
        output,
        &mut workspace.values[..requirements.values],
        rows,
        options.execution_policy(),
        options.workers(),
        context,
        &|column, candidate, _, _| {
            matrix.apply_staged(
                input.column(column)?,
                candidate,
                transpose,
                operation == SparseOperation::Adjoint,
                options.accuracy(),
            )
        },
    )
}

/// Holds the workspace sizes of a sparse product.
///
/// A product computes its output in the workspace and copies it to the output
/// at the end. A [`SparseProductPlan`] product also needs, per worker, a
/// column-to-position map and accumulators for one output row of Gustavson's
/// algorithm. Requirements of several products can be combined with
/// [`union`](Self::union) so that one [`SparseProductWorkspace`] serves all of
/// them.
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseProductOptions, SparseProductPlan,
///     SparseProductRequirements,
/// };
///
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0, 3.0])?;
/// // `A X` for four dense columns needs 2 × 4 output values.
/// let dense = SparseProductRequirements::sparse_dense(a.view(), SparseOperation::Normal, 4)?;
/// assert_eq!((dense.values(), dense.workers()), (8, 0));
/// let plan = SparseProductPlan::new(&a, &a)?;
/// let product = plan.workspace_requirements(SparseProductOptions::new());
/// assert_eq!((product.values(), product.columns(), product.row_entries()), (3, 2, 2));
/// // One workspace reserved for the union serves both products.
/// let shared = dense.union(product);
/// assert_eq!((shared.values(), shared.columns(), shared.workers()), (8, 2, 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseProductRequirements {
    values: usize,
    columns: usize,
    row_entries: usize,
    workers: usize,
}
impl SparseProductRequirements {
    /// Returns the requirements of [`sparse_dense_into`] for `op(matrix) X`
    /// with `right_hand_sides` columns.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the output size overflows `usize`.
    pub fn sparse_dense<S: NumericScalar>(
        matrix: SparseView<'_, S>,
        operation: SparseOperation,
        right_hand_sides: usize,
    ) -> Result<Self, SciOxideError> {
        let rows = if operation == SparseOperation::Normal {
            matrix.rows()
        } else {
            matrix.cols()
        };
        Ok(Self {
            values: checked_mul(
                rows,
                right_hand_sides,
                scioxide_core::quantity!("sparse_dense_workspace"),
            )?,
            ..Self::default()
        })
    }
    /// Returns the number of output values computed in the workspace.
    pub const fn values(self) -> usize {
        self.values
    }
    /// Returns the length of the column-to-position map of each worker.
    pub const fn columns(self) -> usize {
        self.columns
    }
    /// Returns the number of accumulators of each worker: the largest number
    /// of entries in one output row.
    pub const fn row_entries(self) -> usize {
        self.row_entries
    }
    /// Returns the number of workers, at most 64.
    pub const fn workers(self) -> usize {
        self.workers
    }
    /// Returns the bytes these requirements take for scalar type `S`,
    /// including the exact accumulators used when a sum would overflow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn required_bytes<S: NumericScalar>(self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("product_workspace_bytes");
        let indices = checked_mul(self.columns, core::mem::size_of::<usize>(), q)?;
        let accumulators = checked_mul(
            self.row_entries,
            core::mem::size_of::<S>() + core::mem::size_of::<ProductSum<S>>(),
            q,
        )?;
        let row = checked_add(
            core::mem::size_of::<ProductRow<S>>(),
            checked_add(indices, accumulators, q)?,
            q,
        )?;
        checked_add(
            checked_mul(self.values, core::mem::size_of::<S>(), q)?,
            checked_mul(self.workers, row, q)?,
            q,
        )
    }

    /// Returns the componentwise maximum of two requirements, for a workspace
    /// shared by several products.
    pub fn union(self, other: Self) -> Self {
        Self {
            values: self.values.max(other.values),
            columns: self.columns.max(other.columns),
            row_entries: self.row_entries.max(other.row_entries),
            workers: self.workers.max(other.workers),
        }
    }
    const fn covers(self, other: Self) -> bool {
        self.values >= other.values
            && self.columns >= other.columns
            && self.row_entries >= other.row_entries
            && self.workers >= other.workers
    }
}
#[derive(Debug)]
struct ProductRow<S: NumericScalar> {
    positions: Vec<usize>,
    sums: Vec<S>,
    exact: Vec<ProductSum<S>>,
}
/// Holds the scratch storage of [`SparseProductPlan::evaluate_into`] and
/// [`sparse_dense_into`].
///
/// Reserve it for a [`SparseProductRequirements`] before use. Products then
/// do not allocate, for either accuracy.
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_sparse::{CsrMatrix, SparseProductOptions, SparseProductPlan, SparseProductWorkspace};
///
/// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
/// let plan = SparseProductPlan::new(&a, &a)?;
/// let requirements = plan.workspace_requirements(SparseProductOptions::new());
/// let mut workspace = SparseProductWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseProductWorkspace<S: NumericScalar> {
    values: Vec<S>,
    rows: Vec<ProductRow<S>>,
    requirements: SparseProductRequirements,
}
impl<S: NumericScalar> Default for SparseProductWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> SparseProductWorkspace<S> {
    /// Creates an empty workspace. Reserve it before use.
    pub const fn new() -> Self {
        Self {
            values: Vec::new(),
            rows: Vec::new(),
            requirements: SparseProductRequirements {
                values: 0,
                columns: 0,
                row_entries: 0,
                workers: 0,
            },
        }
    }
    /// Returns the bytes of reserved storage.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.requirements.required_bytes::<S>()
    }
    /// Reports whether the workspace covers the requirement `r` in every
    /// component.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte size of `r` overflows `usize`.
    pub fn check(&self, r: &SparseProductRequirements) -> Result<Capacity, SciOxideError> {
        r.required_bytes::<S>()?;
        Ok(if self.requirements.covers(*r) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to cover both its current size and `r`.
    ///
    /// If the reservation fails, the workspace keeps its previous storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `r` asks for more than 64 workers.
    /// - `SizeOverflow` if the byte size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn try_reserve(&mut self, r: &SparseProductRequirements) -> Result<(), SciOxideError> {
        if self.check(r)? == Capacity::Sufficient {
            return Ok(());
        }
        if r.workers > 64 {
            return Err(invalid(
                scioxide_core::argument!("requirements"),
                "product workers exceed 64",
            ));
        }
        let r = self.requirements.union(*r);
        r.required_bytes::<S>()?;
        let q = scioxide_core::quantity!("product_workspace");
        let values = try_filled(r.values, S::ZERO, q)?;
        let mut rows = try_with_capacity(r.workers, q)?;
        for _ in 0..r.workers {
            rows.push(ProductRow {
                positions: try_filled(r.columns, usize::MAX, q)?,
                sums: try_filled(r.row_entries, S::ZERO, q)?,
                exact: try_filled(r.row_entries, ProductSum::new(), q)?,
            });
        }
        *self = Self {
            values,
            rows,
            requirements: r,
        };
        Ok(())
    }
    // Undersized workspaces are rejected, never grown, by execution.
    fn require(&self, r: &SparseProductRequirements) -> Result<(), SciOxideError> {
        if self.check(r)? == Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                r.required_bytes::<S>()?,
                self.retained_bytes()?,
            ));
        }
        Ok(())
    }
}
/// Holds the output limit of a [`SparseProductPlan`].
///
/// The default has no limit. The limit is only compared with the number of
/// entries of the product pattern; storage is never sized by it.
///
/// ```
/// use scioxide_sparse::SparseProductPlanOptions;
///
/// let options = SparseProductPlanOptions::new().with_max_output_entries(100);
/// assert_eq!(options.max_output_entries(), 100);
/// assert_eq!(
///     SparseProductPlanOptions::default().max_output_entries(),
///     usize::MAX
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseProductPlanOptions {
    max_output_entries: usize,
}
impl Default for SparseProductPlanOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseProductPlanOptions {
    /// Creates options without an output limit (`usize::MAX`).
    pub const fn new() -> Self {
        Self {
            max_output_entries: usize::MAX,
        }
    }
    /// Sets the largest number of entries the product pattern may have.
    pub const fn with_max_output_entries(mut self, value: usize) -> Self {
        self.max_output_entries = value;
        self
    }
    /// Returns the largest number of entries the product pattern may have.
    pub const fn max_output_entries(self) -> usize {
        self.max_output_entries
    }
}
/// Computes the product `C = A B` of two CSR matrices whose patterns are known
/// in advance.
///
/// Creating the plan finds the pattern of `C` from the patterns of `A` and `B`
/// alone, row by row as in Gustavson's algorithm, in time proportional to the
/// number of multiplications `Σₖ nnz(A[:, k]) · nnz(B[k, :])`. Explicit zeros
/// in `A` and `B` count, and cancellation never removes an entry from the
/// pattern. [`evaluate_into`](Self::evaluate_into) then computes the values
/// for any `A` and `B` with these patterns. The plan stores only patterns.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseProductOptions, SparseProductPlan, SparseProductPlanOptions,
/// };
///
/// // A = [[1, 2], [0, 3]], so A · A = [[1, 8], [0, 9]].
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0, 3.0])?;
/// let plan = SparseProductPlan::new(&a, &a)?;
/// let options = SparseProductOptions::new();
/// let (mut output, mut workspace) = (plan.create_output()?, plan.workspace(options)?);
/// let context = ExecutionContext::sequential();
/// plan.evaluate_into(&a, &a, options, &mut output, &mut workspace, context)?;
/// assert_eq!(output.col_indices(), [0, 1, 1]);
/// assert_eq!(output.values(), [1.0, 8.0, 9.0]);
/// // The product has three structural entries, so a cap of two is rejected.
/// let capped = |max| SparseProductPlanOptions::new().with_max_output_entries(max);
/// assert!(SparseProductPlan::with_options(&a, &a, capped(3)).is_ok());
/// let e = SparseProductPlan::with_options(&a, &a, capped(2)).unwrap_err();
/// assert_eq!(e.kind(), SciOxideErrorKind::ResourceLimit);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SparseProductPlan {
    rows: usize,
    inner: usize,
    cols: usize,
    left_ptr: Vec<usize>,
    left_col: Vec<usize>,
    right_ptr: Vec<usize>,
    right_col: Vec<usize>,
    output_ptr: Vec<usize>,
    output_col: Vec<usize>,
}
impl SparseProductPlan {
    /// Creates a plan for `left · right` from the patterns of the two
    /// matrices, without an output limit.
    ///
    /// This is [`with_options`](Self::with_options) with
    /// [`SparseProductPlanOptions::new`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `right.rows()` differs from `left.cols()`.
    /// - `SizeOverflow` or `AllocationFailure` if the output pattern cannot be
    ///   allocated.
    pub fn new<S: NumericScalar>(
        left: &CsrMatrix<S>,
        right: &CsrMatrix<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(left, right, SparseProductPlanOptions::new())
    }
    /// Creates a plan like [`new`](Self::new), but fails if the product has
    /// more than `options.max_output_entries()` entries.
    ///
    /// The output pattern never takes more than that many entries of storage.
    /// The full count is still computed, so the error reports it.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `right.rows()` differs from `left.cols()`.
    /// - `ResourceLimit` if the product has more than the limit of entries;
    ///   the error carries the required count and the limit.
    /// - `SizeOverflow` or `AllocationFailure` if the output pattern cannot be
    ///   allocated.
    pub fn with_options<S: NumericScalar>(
        left: &CsrMatrix<S>,
        right: &CsrMatrix<S>,
        options: SparseProductPlanOptions,
    ) -> Result<Self, SciOxideError> {
        Self::new_inner(left, right, options.max_output_entries, &mut |_| Ok(()))
    }
    pub(crate) fn with_admission<S: NumericScalar>(
        left: &CsrMatrix<S>,
        right: &CsrMatrix<S>,
        admission: &mut dyn FnMut(Option<usize>) -> Result<(), SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        Self::new_inner(left, right, usize::MAX, admission)
    }
    // Shared symbolic product.
    fn new_inner<
        S: NumericScalar,
        F: FnMut(Option<usize>) -> Result<(), SciOxideError> + ?Sized,
    >(
        left: &CsrMatrix<S>,
        right: &CsrMatrix<S>,
        max_output_entries: usize,
        admission: &mut F,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::exact(right.rows(), left.cols())?;

        let mut ptr = try_with_capacity(
            checked_add(
                left.rows(),
                1,
                scioxide_core::quantity!("output_row_pointers"),
            )?,
            scioxide_core::quantity!("output_row_pointers"),
        )?;
        ptr.push(0);
        let mut columns = Vec::new();
        // Row stamps suppress duplicate paths without sorted insertion or a
        // temporary entry for every multiplication path.
        let mut marked = try_filled(
            right.cols(),
            usize::MAX,
            scioxide_core::quantity!("product_markers"),
        )?;
        let mut required = 0;
        let limit = max_output_entries;
        for row in 0..left.rows() {
            admission(None)?;
            let start = columns.len();
            for &k in &left.col_indices()[left.row_ptr()[row]..left.row_ptr()[row + 1]] {
                admission(Some(right.row_ptr()[k + 1] - right.row_ptr()[k]))?;
                for &col in &right.col_indices()[right.row_ptr()[k]..right.row_ptr()[k + 1]] {
                    if marked[col] == row {
                        continue;
                    }
                    marked[col] = row;
                    required =
                        checked_add(required, 1, scioxide_core::quantity!("sparse_entries"))?;
                    // Count the complete requirement without retaining excess entries.
                    if required <= limit {
                        columns.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                scioxide_core::quantity!("output_column_indices"),
                                required,
                            )
                        })?;
                        columns.push(col);
                    }
                }
            }
            columns[start..].sort_unstable();
            ptr.push(required);
        }
        if required > limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("sparse_entries"),
                required,
                limit,
            ));
        }
        Ok(Self {
            rows: left.rows(),
            inner: left.cols(),
            cols: right.cols(),
            left_ptr: try_copy_slice(
                left.row_ptr(),
                scioxide_core::quantity!("left_row_pointers"),
            )?,
            left_col: try_copy_slice(
                left.col_indices(),
                scioxide_core::quantity!("left_column_indices"),
            )?,
            right_ptr: try_copy_slice(
                right.row_ptr(),
                scioxide_core::quantity!("right_row_pointers"),
            )?,
            right_col: try_copy_slice(
                right.col_indices(),
                scioxide_core::quantity!("right_column_indices"),
            )?,
            output_ptr: ptr,
            output_col: columns,
        })
    }
    /// Returns the number of entries of the product pattern.
    pub fn output_len(&self) -> usize {
        self.output_col.len()
    }
    /// Returns the workspace that [`evaluate_into`](Self::evaluate_into) needs
    /// with the execution policy in `options`.
    pub fn workspace_requirements(
        &self,
        options: SparseProductOptions,
    ) -> SparseProductRequirements {
        SparseProductRequirements {
            values: self.output_len(),
            columns: if self.output_len() == 0 { 0 } else { self.cols },
            row_entries: self
                .output_ptr
                .windows(2)
                .map(|p| p[1] - p[0])
                .max()
                .unwrap_or(0),
            workers: options.workers().get().min(self.rows.max(1)),
        }
    }
    /// Creates a workspace that satisfies
    /// [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace<S: NumericScalar>(
        &self,
        options: SparseProductOptions,
    ) -> Result<SparseProductWorkspace<S>, SciOxideError> {
        let mut workspace = SparseProductWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(options))?;
        Ok(workspace)
    }
    /// Returns the row pointers of the product pattern.
    pub fn row_ptr(&self) -> &[usize] {
        &self.output_ptr
    }
    /// Returns the column indices of the product pattern, sorted within each
    /// row.
    pub fn col_indices(&self) -> &[usize] {
        &self.output_col
    }
    /// Creates a [`CsrMatrix`] with the product pattern and zero values, to
    /// pass as the output of [`evaluate_into`](Self::evaluate_into).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the matrix cannot be
    ///   allocated.
    pub fn create_output<S: NumericScalar>(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        CsrMatrix::from_raw(
            self.rows,
            self.cols,
            try_copy_slice(
                &self.output_ptr,
                scioxide_core::quantity!("output_row_pointers"),
            )?,
            try_copy_slice(
                &self.output_col,
                scioxide_core::quantity!("output_column_indices"),
            )?,
            try_filled(
                self.output_len(),
                S::ZERO,
                scioxide_core::quantity!("output_values"),
            )?,
        )
    }
    /// Computes the values of `left · right` and writes them to `output`.
    ///
    /// `left`, `right` and `output` must have exactly the patterns the plan
    /// was made for; their values may differ between calls. For a product with
    /// `Aᵀ` or `Aᴴ`, pass the matrix from [`CsrMatrix::transpose`] or
    /// [`CsrMatrix::adjoint`] converted to CSR. With [`Accuracy::Standard`]
    /// each row is accumulated in ordinary arithmetic and recomputed exactly if
    /// a product or sum overflows; with [`Accuracy::Exact`] every entry is
    /// accumulated exactly and rounded once. Rows may run in parallel. The
    /// workspace is checked, not grown, so the call does not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a shape differs from the plan, or the
    ///   workspace is smaller than the requirement.
    /// - `InvalidInput` if the pattern of `left`, `right` or `output` differs
    ///   from the plan.
    /// - `NumericalInstability` if an output entry overflows.
    /// - `ExecutionStopped`, `CapabilityUnavailable`, `CapabilityFailure` or
    ///   `ExecutionFailure` if the execution context cancels or fails the work.
    pub fn evaluate_into<S: NumericScalar>(
        &self,
        left: &CsrMatrix<S>,
        right: &CsrMatrix<S>,
        options: SparseProductOptions,
        output: &mut CsrMatrix<S>,
        workspace: &mut SparseProductWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::helpers::shape((left.rows(), left.cols()), (self.rows, self.inner))?;
        crate::helpers::shape((right.rows(), right.cols()), (self.inner, self.cols))?;
        crate::helpers::shape((output.rows(), output.cols()), (self.rows, self.cols))?;
        workspace.require(&self.workspace_requirements(options))?;
        let argument = if left.row_ptr() != self.left_ptr || left.col_indices() != self.left_col {
            Some(scioxide_core::argument!("left"))
        } else if right.row_ptr() != self.right_ptr || right.col_indices() != self.right_col {
            Some(scioxide_core::argument!("right"))
        } else if output.row_ptr() != self.output_ptr || output.col_indices() != self.output_col {
            Some(scioxide_core::argument!("output"))
        } else {
            None
        };
        if let Some(argument) = argument {
            return Err(invalid(
                argument,
                "prepared sparse product shape/pattern/workspace mismatch",
            ));
        }

        context.check()?;
        let plan = context.plan(
            options.execution_policy(),
            self.rows,
            scioxide_core::PlanLimits {
                work_per_item: NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: options.workers(),
            },
        );
        let mut values = &mut workspace.values[..self.output_len()];
        let inputs =
            plan.partitions()
                .zip(workspace.rows.iter_mut())
                .map(|(partition, row_work)| {
                    let range = partition.range();
                    let count = self.output_ptr[range.end] - self.output_ptr[range.start];
                    let (part, tail) = core::mem::take(&mut values).split_at_mut(count);
                    values = tail;
                    (part, row_work)
                });
        crate::execution::dispatch(
            context,
            plan,
            inputs,
            &|partition, (staged, work), scope| {
                let base = self.output_ptr[partition.range().start];
                for row in partition.range() {
                    scope.check()?;
                    let start = self.output_ptr[row];
                    let end = self.output_ptr[row + 1];
                    let width = end - start;
                    if width == 0 {
                        continue;
                    }
                    for (p, &col) in self.output_col[start..end].iter().enumerate() {
                        work.positions[col] = p;
                    }
                    work.sums[..width].fill(S::ZERO);
                    let mut retry = options.accuracy() == Accuracy::Exact;
                    if !retry {
                        for a in self.left_ptr[row]..self.left_ptr[row + 1] {
                            let k = self.left_col[a];
                            for b in self.right_ptr[k]..self.right_ptr[k + 1] {
                                let p = work.positions[self.right_col[b]];
                                let (av, bv) = (left.values()[a], right.values()[b]);
                                let product = av * bv;
                                retry |= scioxide_core::backend::standard::product_needs_retry(
                                    av, bv, product,
                                );
                                work.sums[p] = work.sums[p] + product;
                            }
                        }
                        retry |= work.sums[..width]
                            .iter()
                            .any(|s| !s.has_finite_components());
                    }
                    if retry {
                        work.exact[..width].fill(ProductSum::new());
                        for a in self.left_ptr[row]..self.left_ptr[row + 1] {
                            let k = self.left_col[a];
                            for b in self.right_ptr[k]..self.right_ptr[k + 1] {
                                let p = work.positions[self.right_col[b]];
                                work.exact[p].add_product(left.values()[a], right.values()[b])?;
                            }
                        }
                        for (slot, sum) in staged[start - base..end - base]
                            .iter_mut()
                            .zip(&work.exact[..width])
                        {
                            *slot = sum.finish()?;
                        }
                    } else {
                        staged[start - base..end - base].copy_from_slice(&work.sums[..width]);
                    }
                }
                Ok(())
            },
        )?;
        context.check()?;
        output.publish_finite(&workspace.values[..self.output_len()]);
        Ok(())
    }
}
impl<S: NumericScalar> CsrMatrix<S> {
    /// Returns the product `A B`, where `A` is `self` and `B` is `right`.
    ///
    /// The product runs sequentially through a [`SparseProductPlan`] with
    /// [`Accuracy::Standard`]. Its pattern is the structural product of the two
    /// patterns, so an entry that cancels to zero stays stored. For another
    /// accuracy, or several products with the same patterns, keep a
    /// [`SparseProductPlan`] instead.
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// // [[1, 1]] · [[1], [−1]] = [[0]]: a stored entry with value zero.
    /// let a = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 1.0])?;
    /// let b = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![1.0, -1.0])?;
    /// assert_eq!(a.multiply(&b)?.values(), [0.0]);
    /// // A 1 × 2 matrix cannot multiply another 1 × 2 matrix.
    /// let e = a.multiply(&a).unwrap_err();
    /// assert_eq!(e.kind(), SciOxideErrorKind::DimensionMismatch);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `right.rows()` differs from `self.cols()`.
    /// - `NumericalInstability` if an entry overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn multiply(&self, right: &Self) -> Result<Self, SciOxideError> {
        let plan = SparseProductPlan::new(self, right)?;
        let mut output = plan.create_output()?;
        let options = SparseProductOptions::new();
        plan.evaluate_into(
            self,
            right,
            options,
            &mut output,
            &mut plan.workspace(options)?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}
