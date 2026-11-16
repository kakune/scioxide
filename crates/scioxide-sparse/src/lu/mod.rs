//! Sparse LU factorization: symbolic analysis, storage limits and requirements.
use crate::helpers::invalid;
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{checked_add as add, checked_mul as mul, try_filled};
use scioxide_core::{
    Argument, NumericScalar, Permutation, Quantity, SciOxideError, StructuralPattern,
    StructuralPatternView,
};

mod numeric;
mod solve;
pub use numeric::{SparseLu, SparseLuWorkspace};

/// Sets the storage limits of a sparse LU factorization.
///
/// `max_factor_entries` bounds the number of stored entries of `L` and `U`
/// together, not counting the unit diagonal of `L`. `max_workspace_bytes` bounds
/// the bytes of the analysis, the factor and its refactorization workspace
/// together, and also the temporary storage of the analysis. Allocator overhead
/// is not counted, and a scratch slice you lend to a solve is checked only by
/// its length. Neither limit grows automatically: fill beyond the entry limit is
/// a `ResourceLimit` error.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseOrdering, SparseSolveOptions,
/// };
/// let limits = SparseLuLimits::new(1, 1 << 16);
/// assert_eq!(
///     (limits.max_factor_entries(), limits.max_workspace_bytes()),
///     (1, 1 << 16)
/// );
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 4.])?;
/// let factor = |limits| -> Result<SparseLu<f64>, scioxide_core::SciOxideError> {
///     let symbolic =
///         SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits)?;
///     let scope = ExecutionContext::sequential();
///     SparseLu::factor_scoped(symbolic, a.view(), SparseSolveOptions::new(), scope)
/// };
/// // The factor of diag(2, 4) stores two pivots, so a limit of one entry fails.
/// assert_eq!(
///     factor(limits).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// assert_eq!(
///     factor(limits.with_max_factor_entries(2))?.factor_entries(),
///     2
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseLuLimits {
    /// Stored entries of `L` and `U`, without the unit diagonal of `L`.
    pub(crate) max_factor_entries: usize,
    /// Bytes of the analysis, one factor and the workspace together.
    pub(crate) max_workspace_bytes: usize,
}

/// Holds the storage, in bytes, that a sparse LU analysis, its factor and its
/// workspace need.
///
/// The sizes count the arrays themselves, without allocator overhead or stack.
/// [`total_bytes`](Self::total_bytes) is compared with
/// [`SparseLuLimits::max_workspace_bytes`].
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, SparseLuLimits, SparseLuSymbolic, SparseOrdering};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 4.])?;
/// let limits = SparseLuLimits::new(2, 1 << 16);
/// let analyze =
///     |limits| SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits);
/// let r = analyze(limits)?.workspace_requirements::<f64>(())?;
/// assert_eq!(
///     r.total_bytes(),
///     r.symbolic_bytes() + r.factor_bytes() + r.workspace_bytes()
/// );
/// // Analysis fits one byte short of the total, but the reservation does not.
/// let tight = analyze(limits.with_max_workspace_bytes(r.total_bytes() - 1))?;
/// let error = tight.workspace::<f64>(()).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseLuRequirements {
    pub(crate) n: usize,
    pub(crate) z: usize,
    pub(crate) limits: SparseLuLimits,
    /// Symbolic analysis.
    pub(crate) symbolic_bytes: usize,
    /// One numeric factor at the selected entry capacity.
    pub(crate) factor_bytes: usize,
    /// Numeric scratch including a second candidate factor.
    pub(crate) workspace_bytes: usize,
    /// Sum of the three payloads above.
    pub(crate) total_bytes: usize,
}

/// Holds the symbolic analysis of a square sparsity pattern for [`SparseLu`].
///
/// The analysis stores the pattern and a column permutation `q` with
/// `q[new] = old`, so the matrix factored is `B[i, j] = A[i, q[j]]`. With
/// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing), the
/// default, `q` is an approximate minimum degree ordering of the pattern of
/// `A + Aᵀ`. The byte limit also bounds the storage used to compute that
/// ordering. Row pivoting during factorization decides the actual fill, so the
/// analysis does not predict the size of the factor.
///
/// An analysis depends only on the pattern. One analysis serves every matrix
/// with that pattern, whatever its values.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, SparseLuLimits, SparseLuSymbolic, SparseOrdering};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0_f64, 1., 4.])?;
/// let limits = SparseLuLimits::new(4, 1 << 16);
/// let symbolic =
///     SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::FillReducing, limits)?;
/// assert_eq!(symbolic.dimension(), 2);
/// assert_eq!(symbolic.column_permutation().new_to_old().len(), 2);
/// let wide = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![0], vec![1.0_f64])?;
/// let error = SparseLuSymbolic::analyze(wide.structural_pattern(), SparseOrdering::Natural, limits)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseLuSymbolic {
    capacity_n: usize,
    capacity_z: usize,
    pattern: StructuralPattern,
    permutation: Permutation,
    column_ptr: Vec<usize>,
    rows: Vec<usize>,
    positions: Vec<usize>,
    limits: SparseLuLimits,
}

impl SparseLuSymbolic {
    /// Copies the analysis without repeating it.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            capacity_n: self.dimension(),
            capacity_z: self.rows.len(),
            pattern: self.pattern.view().to_owned()?,
            permutation: Permutation::new(scioxide_core::backend::try_copy_slice(
                self.permutation.new_to_old(),
                scioxide_core::quantity!("symbolic_copy"),
            )?)?,
            column_ptr: scioxide_core::backend::try_copy_slice(
                &self.column_ptr,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            rows: scioxide_core::backend::try_copy_slice(
                &self.rows,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            positions: scioxide_core::backend::try_copy_slice(
                &self.positions,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            limits: self.limits,
        })
    }

    /// Analyzes the sparsity pattern of a square matrix and computes its column
    /// ordering.
    ///
    /// Explicitly stored zeros are part of the pattern.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the pattern is not square, or a supplied
    ///   ordering has a length other than the order of the pattern.
    /// - `ResourceLimit` if the analysis, including the temporary storage of the
    ///   ordering, exceeds `limits.max_workspace_bytes()`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn analyze(
        pattern: StructuralPatternView<'_>,
        ordering: crate::SparseOrdering<'_>,
        limits: SparseLuLimits,
    ) -> Result<Self, SciOxideError> {
        const PREPARATION: Quantity = scioxide_core::quantity!("symbolic_preparation_bytes");
        crate::helpers::exact(pattern.cols(), pattern.rows())?;
        if let crate::SparseOrdering::Supplied(order) = ordering {
            crate::helpers::exact(order.new_to_old().len(), pattern.cols())?;
        }
        let n = pattern.rows();

        let z = pattern.col_indices().len();
        let bytes = symbolic_bytes(n, z)?;
        // One n-entry cursor overlaps the final immutable arrays.
        if add(bytes, mul(n, size_of::<usize>(), PREPARATION)?, PREPARATION)?
            > limits.max_workspace_bytes
        {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                add(bytes, mul(n, size_of::<usize>(), PREPARATION)?, PREPARATION)?,
                limits.max_workspace_bytes,
            ));
        }
        let permutation = ordering.prepare(
            pattern,
            crate::ordering::OrderingGraph::Symmetric(None),
            limits.max_workspace_bytes,
        )?;
        let mut rows = try_filled(z, 0usize, scioxide_core::quantity!("column_rows"))?;
        let mut positions = try_filled(z, 0usize, scioxide_core::quantity!("column_positions"))?;
        let column_ptr = crate::helpers::counting_transpose(
            n,
            (0..n).flat_map(|row| {
                (pattern.row_ptr()[row]..pattern.row_ptr()[row + 1])
                    .map(move |pos| (pattern.col_indices()[pos], (row, pos)))
            }),
            |slot, (row, pos)| {
                rows[slot] = row;
                positions[slot] = pos;
            },
        )?;
        Ok(Self {
            capacity_n: n,
            capacity_z: z,
            pattern: pattern.to_owned()?,
            permutation,
            column_ptr,
            rows,
            positions,
            limits,
        })
    }
    // Rebuild natural column traversal while retaining structural capacity.
    fn reanalyze_natural(
        &mut self,
        pattern: StructuralPatternView<'_>,
    ) -> Result<(), SciOxideError> {
        crate::helpers::exact(pattern.rows(), pattern.cols())?;
        let n = pattern.rows();
        let z = pattern.col_indices().len();
        let capacity_n = n.max(self.capacity_n);
        let capacity_z = z.max(self.capacity_z);
        let bytes = symbolic_bytes(capacity_n, capacity_z)?;
        if bytes > self.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                self.limits.max_workspace_bytes,
            ));
        }
        self.capacity_n = capacity_n;
        self.capacity_z = capacity_z;
        crate::helpers::resize(&mut self.column_ptr, n + 1, 0)?;
        crate::helpers::resize(&mut self.rows, z, 0)?;
        crate::helpers::resize(&mut self.positions, z, 0)?;
        self.permutation.try_reset_identity(n)?;
        self.pattern.try_assign(pattern)?;
        crate::helpers::transpose_into(
            pattern,
            &mut self.column_ptr,
            &mut self.rows,
            &mut self.positions,
        );
        Ok(())
    }
    /// Returns the analyzed sparsity pattern.
    pub fn pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }
    /// Returns the column permutation `q`, with `q[new] = old`.
    pub fn column_permutation(&self) -> &Permutation {
        &self.permutation
    }
    /// Returns the order `n` of the matrix, which may be zero.
    pub fn dimension(&self) -> usize {
        self.pattern.view().rows()
    }
    /// Returns the storage limits of this analysis.
    pub fn limits(&self) -> SparseLuLimits {
        self.limits
    }
    /// Returns the storage that a factor with scalar type `S` and its
    /// refactorization workspace need.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements<S: NumericScalar>(
        &self,
        (): (),
    ) -> Result<SparseLuRequirements, SciOxideError> {
        SparseLuRequirements::for_shape::<S>(
            self.dimension(),
            self.rows.len(),
            symbolic_bytes(self.capacity_n, self.capacity_z)?,
            self.limits,
        )
    }
}
// Immutable symbolic payload in bytes.
impl SparseLuRequirements {
    fn for_shape<S: NumericScalar>(
        n: usize,
        z: usize,
        symbolic_bytes: usize,
        limits: SparseLuLimits,
    ) -> Result<Self, SciOxideError> {
        let factor = scioxide_core::quantity!("factor_bytes");
        let workspace = scioxide_core::quantity!("workspace_bytes");
        let total = scioxide_core::quantity!("total_bytes");
        let factor_bytes = add(
            mul(
                limits.max_factor_entries,
                size_of::<numeric::Entry<S>>(),
                factor,
            )?,
            mul(
                add(mul(4, n, factor)?, 1, factor)?,
                size_of::<usize>(),
                factor,
            )?,
            factor,
        )?;
        let scratch = add(
            add(
                mul(add(n, z, workspace)?, size_of::<S>(), workspace)?,
                mul(mul(3, n, workspace)?, size_of::<usize>(), workspace)?,
                workspace,
            )?,
            add(n, z, workspace)?,
            workspace,
        )?;
        let workspace_bytes = add(factor_bytes, scratch, workspace)?;
        let total_bytes = add(
            symbolic_bytes,
            add(factor_bytes, workspace_bytes, total)?,
            total,
        )?;
        Ok(SparseLuRequirements {
            n,
            z,
            limits,
            symbolic_bytes,
            factor_bytes,
            workspace_bytes,
            total_bytes,
        })
    }
}

fn symbolic_bytes(n: usize, z: usize) -> Result<usize, SciOxideError> {
    let bytes = scioxide_core::quantity!("symbolic_bytes");
    mul(
        add(add(mul(4, n, bytes)?, 2, bytes)?, mul(3, z, bytes)?, bytes)?,
        size_of::<usize>(),
        bytes,
    )
}

impl SparseLuLimits {
    /// Creates limits on the stored factor entries and on the total bytes.
    ///
    /// Analysis checks the byte limit against its own storage. The factor and
    /// the workspace are checked when they are created, and fill beyond
    /// `max_factor_entries` fails during factorization.
    pub const fn new(max_factor_entries: usize, max_workspace_bytes: usize) -> Self {
        Self {
            max_factor_entries,
            max_workspace_bytes,
        }
    }
    /// Returns the maximum number of stored entries of `L` and `U`.
    pub const fn max_factor_entries(self) -> usize {
        self.max_factor_entries
    }
    /// Sets the maximum number of stored entries of `L` and `U`.
    pub const fn with_max_factor_entries(mut self, value: usize) -> Self {
        self.max_factor_entries = value;
        self
    }
    /// Returns the maximum total bytes of the analysis, the factor and its
    /// workspace.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the maximum total bytes of the analysis, the factor and its
    /// workspace.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
}
impl SparseLuRequirements {
    /// Returns the bytes of the symbolic analysis.
    pub const fn symbolic_bytes(self) -> usize {
        self.symbolic_bytes
    }
    /// Returns the bytes of one numerical factor.
    pub const fn factor_bytes(self) -> usize {
        self.factor_bytes
    }
    /// Returns the bytes of the refactorization workspace, which holds a
    /// second factor.
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }
    /// Returns the sum of the analysis, factor and workspace bytes.
    pub const fn total_bytes(self) -> usize {
        self.total_bytes
    }
}

impl SparseLuSymbolic {
    /// Creates a refactorization workspace for this analysis.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the total bytes exceed the byte limit.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn workspace<S: NumericScalar>(
        &self,
        request: (),
    ) -> Result<SparseLuWorkspace<S>, SciOxideError> {
        let mut workspace = SparseLuWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}
