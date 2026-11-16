//! Sparse Cholesky and `LDLᴴ` factorizations: symbolic analysis, storage
//! limits and requirements.
use crate::helpers::invalid;
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{
    checked_add as add, checked_mul as mul, try_filled, try_with_capacity,
};
use scioxide_core::{
    Argument, NumericScalar, Permutation, SciOxideError, StructuralPattern, StructuralPatternView,
};

use crate::hermitian::{LdlInertia, LdlPivotBlock};
use scioxide_core::Triangle;
mod active;
mod numeric;
mod solve;
pub use numeric::{SparseCholesky, SparseLdl, SparseSymmetricWorkspace};
/// Sets the storage limits of a sparse Cholesky or `LDLᴴ` factorization.
///
/// `max_factor_entries` bounds the stored entries of `L` strictly below the
/// diagonal; the diagonal and the 2×2 blocks of `D` count only toward the byte
/// limit. `max_active_entries` bounds the nonzero entries of the lower triangle
/// of the Schur complement, diagonal included, during `LDLᴴ` factorization.
/// `max_workspace_bytes` bounds the bytes of the analysis, one factor and its
/// workspace together, and also the temporary storage of the analysis.
/// Inputs, outputs, scratch slices you lend to a solve, and allocator overhead
/// are not counted. No limit grows automatically, and there is no dense
/// fallback.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, SparseLdl, SparseOrdering, SparseSymmetricLimits, SparseSymmetricSymbolic,
/// };
/// let limits = SparseSymmetricLimits::new(1, 0, 1 << 16);
/// assert_eq!(limits.with_max_active_entries(1).max_active_entries(), 1);
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![2.0_f64, 2.])?;
/// let ldl = |limits| -> Result<SparseLdl<f64>, scioxide_core::SciOxideError> {
///     let plan = SparseSymmetricSymbolic::analyze(
///         a.structural_pattern(),
///         SparseOrdering::Natural,
///         None,
///         limits,
///     )?;
///     SparseLdl::factor(plan, a.view())
/// };
/// // The off-diagonal entry of [[0, 2], [2, 0]] needs one active entry.
/// assert_eq!(
///     ldl(limits).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// assert!(ldl(limits.with_max_active_entries(1)).is_ok());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseSymmetricLimits {
    /// Stored entries of `L` strictly below the diagonal.
    pub(crate) max_factor_entries: usize,
    /// Nonzero lower-triangle Schur complement entries during `LDLᴴ`.
    pub(crate) max_active_entries: usize,
    /// Bytes of the analysis, one factor and the workspace together.
    pub(crate) max_workspace_bytes: usize,
}
/// Holds the storage, in bytes, that a sparse Cholesky or `LDLᴴ` analysis, its
/// factor and its workspace need.
///
/// The sizes count the arrays themselves, without allocator overhead or stack.
/// [`total_bytes`](Self::total_bytes) is compared with
/// [`SparseSymmetricLimits::max_workspace_bytes`].
///
/// ```
/// use scioxide_core::{SciOxideErrorKind, Triangle};
/// use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseSymmetricLimits, SparseSymmetricSymbolic};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 9.])?;
/// let limits = SparseSymmetricLimits::new(1, 2, 1 << 16);
/// let analyze = |limits| {
///     SparseSymmetricSymbolic::analyze(
///         a.structural_pattern(),
///         SparseOrdering::Natural,
///         Some(Triangle::Lower),
///         limits,
///     )
/// };
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
pub struct SparseSymmetricRequirements {
    pub(crate) n: usize,
    pub(crate) z: usize,
    pub(crate) limits: SparseSymmetricLimits,
    /// Symbolic analysis, sized by the pattern and the Cholesky fill.
    pub(crate) symbolic_bytes: usize,
    /// One numeric factor.
    pub(crate) factor_bytes: usize,
    /// Scratch, active matrix and second factor.
    pub(crate) workspace_bytes: usize,
    /// Sum of symbolic, factor and workspace payloads.
    pub(crate) total_bytes: usize,
}
/// Holds the symbolic analysis of a Hermitian sparsity pattern for
/// [`SparseCholesky`] and [`SparseLdl`].
///
/// The analysis stores the pattern, a symmetric permutation `p` with
/// `p[new] = old`, so that the matrix factored is `B[i, j] = A[p[i], p[j]]`,
/// and the elimination tree of `B`. With
/// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing), the
/// default, `p` is an approximate minimum degree ordering of the pattern of the
/// triangle that is read. From the elimination tree the analysis computes the
/// pattern of the Cholesky factor, counting stored zeros as entries.
/// `LDLᴴ` pivoting can need more fill than this pattern. The byte limit also
/// bounds the storage used to compute the ordering.
///
/// # Examples
///
/// ```
/// use scioxide_core::{SciOxideErrorKind, Triangle};
/// use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseSymmetricLimits, SparseSymmetricSymbolic};
/// // Lower triangle of the tridiagonal [[4, 1, 0], [1, 4, 1], [0, 1, 4]].
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 1, 3, 5],
///     vec![0, 0, 1, 1, 2],
///     vec![4.0_f64, 1., 4., 1., 4.],
/// )?;
/// let limits = SparseSymmetricLimits::new(2, 8, 1 << 16);
/// let analyze = |limits| {
///     SparseSymmetricSymbolic::analyze(
///         a.structural_pattern(),
///         SparseOrdering::Natural,
///         Some(Triangle::Lower),
///         limits,
///     )
/// };
/// let plan = analyze(limits)?;
/// assert_eq!(plan.cholesky_entries(), 2);
/// assert!(plan.elimination_tree().eq([Some(1), Some(2), None]));
/// // Structural Cholesky fill above the factor-entry cap is rejected during analysis.
/// let error = analyze(limits.with_max_factor_entries(1)).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseSymmetricSymbolic {
    pattern: StructuralPattern,
    permutation: Permutation,
    triangle: Option<Triangle>,
    // Sorted (column,row) upper-triangle coordinates after permutation.
    upper: Vec<(usize, usize)>,
    upper_ptr: Vec<usize>,
    parent: Vec<usize>,
    lower_ptr: Vec<usize>,
    lower_rows: Vec<usize>,
    limits: SparseSymmetricLimits,
}
impl SparseSymmetricSymbolic {
    /// Copies the analysis without repeating it.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            pattern: self.pattern.view().to_owned()?,
            permutation: Permutation::new(scioxide_core::backend::try_copy_slice(
                self.permutation.new_to_old(),
                scioxide_core::quantity!("symbolic_copy"),
            )?)?,
            triangle: self.triangle,
            upper: scioxide_core::backend::try_copy_slice(
                &self.upper,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            upper_ptr: scioxide_core::backend::try_copy_slice(
                &self.upper_ptr,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            parent: scioxide_core::backend::try_copy_slice(
                &self.parent,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            lower_ptr: scioxide_core::backend::try_copy_slice(
                &self.lower_ptr,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            lower_rows: scioxide_core::backend::try_copy_slice(
                &self.lower_rows,
                scioxide_core::quantity!("symbolic_copy"),
            )?,
            limits: self.limits,
        })
    }

    /// Analyzes the sparsity pattern of a Hermitian matrix and computes its
    /// ordering and Cholesky fill.
    ///
    /// `triangle` selects the stored entries that define the matrix.
    /// `Some(Triangle::Lower)` reads the lower triangle and ignores entries above
    /// the diagonal, and `Some(Triangle::Upper)` the reverse. `None` reads both
    /// triangles, and each factorization then checks that the values are exactly
    /// Hermitian. Every stored entry, including those in an ignored triangle, is
    /// part of the pattern that later matrices must match.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the pattern is not square, or a supplied
    ///   ordering has a length other than the order of the pattern.
    /// - `ResourceLimit` if the Cholesky fill exceeds
    ///   `limits.max_factor_entries()`, or the analysis, including the temporary
    ///   storage of the ordering, exceeds `limits.max_workspace_bytes()`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn analyze(
        pattern: StructuralPatternView<'_>,
        ordering: crate::SparseOrdering<'_>,
        triangle: Option<Triangle>,
        limits: SparseSymmetricLimits,
    ) -> Result<Self, SciOxideError> {
        let preparation = scioxide_core::quantity!("symbolic_preparation_bytes");
        crate::helpers::exact(pattern.cols(), pattern.rows())?;
        if let crate::SparseOrdering::Supplied(order) = ordering {
            crate::helpers::exact(order.new_to_old().len(), pattern.cols())?;
        }
        let n = pattern.rows();
        let z = pattern.col_indices().len();

        // Three n-entry preparation arrays overlap the retained symbolic buffers.
        let scratch = mul(mul(3, n, preparation)?, size_of::<usize>(), preparation)?;
        let charge = |fill: usize| -> Result<(), SciOxideError> {
            let bytes = add(symbolic_bytes(n, z, fill)?, scratch, preparation)?;
            if bytes > limits.max_workspace_bytes {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("workspace_bytes"),
                    bytes,
                    limits.max_workspace_bytes,
                ));
            }
            Ok(())
        };
        // The structure-only payload is admitted before ordering, and the
        // structural Cholesky fill once it is known; the entry cap never sizes storage.
        charge(0)?;
        let permutation = ordering.prepare(
            pattern,
            crate::ordering::OrderingGraph::Symmetric(triangle),
            limits.max_workspace_bytes,
        )?;
        let mut upper = try_with_capacity(z, scioxide_core::quantity!("upper_coordinates"))?;
        for row in 0..n {
            for &col in &pattern.col_indices()[pattern.row_ptr()[row]..pattern.row_ptr()[row + 1]] {
                if !selected(triangle, row, col) {
                    continue;
                }
                let i = permutation.old_to_new()[row];
                let j = permutation.old_to_new()[col];
                upper.push((i.max(j), i.min(j)));
            }
        }
        upper.sort_unstable();
        upper.dedup();
        let mut upper_ptr = try_filled(
            add(n, 1, scioxide_core::quantity!("upper_pointers"))?,
            0usize,
            scioxide_core::quantity!("upper_pointers"),
        )?;
        for &(col, _) in &upper {
            upper_ptr[col + 1] += 1;
        }
        for k in 0..n {
            upper_ptr[k + 1] += upper_ptr[k];
        }
        let mut parent = try_filled(n, n, scioxide_core::quantity!("elimination_tree"))?;
        let mut mark = try_filled(n, n, scioxide_core::quantity!("reach_marks"))?;
        for k in 0..n {
            for &(_, root) in &upper[upper_ptr[k]..upper_ptr[k + 1]] {
                let mut i = root;
                while i < k {
                    let next = mark[i];
                    mark[i] = k;
                    if next == n {
                        parent[i] = k;
                    }
                    i = next;
                }
            }
        }
        let mut counts = try_filled(n, 0usize, scioxide_core::quantity!("column_counts"))?;
        mark.fill(n);
        for k in 0..n {
            visit_reach(
                k,
                &upper[upper_ptr[k]..upper_ptr[k + 1]],
                &parent,
                &mut mark,
                |i| {
                    counts[i] += 1;
                },
            );
        }
        let lower = scioxide_core::quantity!("lower_pointers");
        let mut lower_ptr = try_filled(
            add(n, 1, lower)?,
            0usize,
            scioxide_core::quantity!("lower_pointers"),
        )?;
        for (k, &count) in counts.iter().enumerate() {
            lower_ptr[k + 1] = add(lower_ptr[k], count, lower)?;
        }
        if lower_ptr[n] > limits.max_factor_entries {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("factor_entries"),
                lower_ptr[n],
                limits.max_factor_entries,
            ));
        }
        charge(lower_ptr[n])?;
        let mut lower_rows =
            try_filled(lower_ptr[n], 0usize, scioxide_core::quantity!("lower_rows"))?;
        counts.copy_from_slice(&lower_ptr[..n]);
        mark.fill(n);
        for k in 0..n {
            visit_reach(
                k,
                &upper[upper_ptr[k]..upper_ptr[k + 1]],
                &parent,
                &mut mark,
                |i| {
                    lower_rows[counts[i]] = k;
                    counts[i] += 1;
                },
            );
        }
        Ok(Self {
            pattern: pattern.to_owned()?,
            permutation,
            triangle,
            upper,
            upper_ptr,
            parent,
            lower_ptr,
            lower_rows,
            limits,
        })
    }
    /// Returns the analyzed sparsity pattern, including any ignored triangle.
    pub fn pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }
    /// Returns the order `n` of the matrix, which may be zero.
    pub fn dimension(&self) -> usize {
        self.parent.len()
    }
    /// Returns the symmetric permutation `p`, with `p[new] = old`.
    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }
    /// Returns the triangle that is read, or `None` if both triangles are read
    /// and checked for exact Hermitian symmetry.
    pub fn triangle(&self) -> Option<Triangle> {
        self.triangle
    }
    /// Returns the parent of each node in the elimination tree of the permuted
    /// matrix, or `None` for a root.
    pub fn elimination_tree(&self) -> impl ExactSizeIterator<Item = Option<usize>> + '_ {
        self.parent
            .iter()
            .map(|&p| (p < self.dimension()).then_some(p))
    }
    /// Returns the number of entries of the Cholesky factor `L` strictly below
    /// the diagonal, as predicted from the pattern. Numerical cancellation does
    /// not reduce it.
    pub fn cholesky_entries(&self) -> usize {
        self.lower_rows.len()
    }
    /// Returns the storage limits of this analysis.
    pub fn limits(&self) -> SparseSymmetricLimits {
        self.limits
    }
    /// Returns the storage that a factor with scalar type `S` and its
    /// refactorization workspace need.
    ///
    /// The workspace includes the storage of the `LDLᴴ` Schur complement, sized
    /// by the limits. Factorization never grows it, so pivoting that needs more
    /// fill fails with `ResourceLimit`. Solve scratch is separate.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements<S: NumericScalar>(
        &self,
        (): (),
    ) -> Result<SparseSymmetricRequirements, SciOxideError> {
        let factor = scioxide_core::quantity!("factor_bytes");
        let workspace = scioxide_core::quantity!("workspace_bytes");
        let total = scioxide_core::quantity!("total_bytes");
        let n = self.dimension();
        let z = self.pattern().col_indices().len();
        let symbolic_bytes = symbolic_bytes(n, z, self.cholesky_entries())?;
        let factor_bytes = add(
            add(
                mul(
                    self.limits.max_factor_entries,
                    size_of::<numeric::Entry<S>>(),
                    factor,
                )?,
                mul(mul(2, n, factor)?, size_of::<S>(), factor)?,
                factor,
            )?,
            add(
                mul(
                    add(mul(3, n, factor)?, 1, factor)?,
                    size_of::<usize>(),
                    factor,
                )?,
                mul(n, size_of::<LdlPivotBlock>(), factor)?,
                factor,
            )?,
            factor,
        )?;
        let workspace_bytes = add(
            factor_bytes,
            add(
                add(
                    mul(
                        add(z, mul(2, n, workspace)?, workspace)?,
                        size_of::<S>(),
                        workspace,
                    )?,
                    z,
                    workspace,
                )?,
                add(
                    mul(mul(3, n, workspace)?, size_of::<usize>(), workspace)?,
                    active::bytes::<S>(n, self.limits.max_active_entries)?,
                    workspace,
                )?,
                workspace,
            )?,
            workspace,
        )?;
        let total_bytes = add(
            symbolic_bytes,
            add(factor_bytes, workspace_bytes, total)?,
            total,
        )?;
        Ok(SparseSymmetricRequirements {
            n,
            z,
            limits: self.limits,
            symbolic_bytes,
            factor_bytes,
            workspace_bytes,
            total_bytes,
        })
    }
}
fn selected(triangle: Option<Triangle>, row: usize, col: usize) -> bool {
    match triangle {
        None => true,
        Some(Triangle::Lower) => row >= col,
        Some(Triangle::Upper) => row <= col,
    }
}
// Retained symbolic payload in bytes, with `f` structural strict-lower entries.
fn symbolic_bytes(n: usize, z: usize, f: usize) -> Result<usize, SciOxideError> {
    let bytes = scioxide_core::quantity!("symbolic_bytes");
    mul(
        add(
            add(add(mul(6, n, bytes)?, 3, bytes)?, mul(3, z, bytes)?, bytes)?,
            f,
            bytes,
        )?,
        size_of::<usize>(),
        bytes,
    )
}
fn visit_reach(
    k: usize,
    upper: &[(usize, usize)],
    parent: &[usize],
    mark: &mut [usize],
    mut visit: impl FnMut(usize),
) {
    for &(_, root) in upper {
        let mut i = root;
        while i < k && mark[i] != k {
            mark[i] = k;
            visit(i);
            i = parent[i];
        }
    }
}

impl SparseSymmetricLimits {
    /// Creates limits on the factor entries, the Schur complement entries and
    /// the total bytes.
    ///
    /// Analysis checks the Cholesky fill against `max_factor_entries` and its
    /// own storage against the byte limit. The factor and the workspace are
    /// checked when they are created. The fill and Schur complement entries that
    /// `LDLᴴ` pivoting creates are checked during factorization.
    pub const fn new(
        max_factor_entries: usize,
        max_active_entries: usize,
        max_workspace_bytes: usize,
    ) -> Self {
        Self {
            max_factor_entries,
            max_active_entries,
            max_workspace_bytes,
        }
    }
    /// Returns the maximum number of entries of `L` below the diagonal.
    pub const fn max_factor_entries(self) -> usize {
        self.max_factor_entries
    }
    /// Sets the maximum number of entries of `L` below the diagonal.
    pub const fn with_max_factor_entries(mut self, value: usize) -> Self {
        self.max_factor_entries = value;
        self
    }
    /// Returns the maximum number of nonzero entries in the lower triangle of
    /// the `LDLᴴ` Schur complement.
    pub const fn max_active_entries(self) -> usize {
        self.max_active_entries
    }
    /// Sets the maximum number of nonzero entries in the lower triangle of the
    /// `LDLᴴ` Schur complement.
    pub const fn with_max_active_entries(mut self, value: usize) -> Self {
        self.max_active_entries = value;
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
impl SparseSymmetricRequirements {
    /// Returns the bytes of the symbolic analysis.
    pub const fn symbolic_bytes(self) -> usize {
        self.symbolic_bytes
    }
    /// Returns the bytes of one numerical factor.
    pub const fn factor_bytes(self) -> usize {
        self.factor_bytes
    }
    /// Returns the bytes of the refactorization workspace, which holds a
    /// second factor and the `LDLᴴ` Schur complement.
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }
    /// Returns the sum of the analysis, factor and workspace bytes.
    pub const fn total_bytes(self) -> usize {
        self.total_bytes
    }
}

impl SparseSymmetricSymbolic {
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
    ) -> Result<SparseSymmetricWorkspace<S>, SciOxideError> {
        let mut workspace = SparseSymmetricWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}
