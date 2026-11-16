//! Scalar and block incomplete LU and Cholesky factorizations.
//!
//! Rows visit touched blocks and their elimination paths, with ordered sparse
//! reach discovery and sorting of kept columns. There are no dimension-wide
//! clears or scans per block row. Indexed row storage is O(n b²); ICT additionally
//! reserves O(n + f) column incidence for n block rows and f factor slots.
use crate::helpers::invalid;
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{checked_add as add, checked_mul as mul, try_filled};
use scioxide_core::{
    Argument, NumericScalar, Quantity, RealScalar, SciOxideError, StructuralPattern,
    StructuralPatternView,
};

mod accumulator;
mod block;
mod numeric;
mod solve;
pub use numeric::{IncompleteFactor, IncompleteWorkspace};
/// Selects the incomplete factorization and the rule that decides which entries
/// it keeps.
///
/// With block size `b > 1` every rule acts on `b × b` blocks instead of scalar
/// entries, and the magnitude of a block is its Frobenius norm.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
///     IncompleteSymbolic,
/// };
/// // Eliminating column 0 creates level-one fill at (1, 2) and (2, 1).
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 3, 5, 7],
///     vec![0, 1, 2, 0, 1, 0, 2],
///     vec![4.0_f64, 1., 1., 1., 4., 1., 4.],
/// )?;
/// let limits = IncompleteLimits::new(9, 4096);
/// for (level, blocks) in [(0, 7), (1, 9)] {
///     let plan = IncompleteSymbolic::analyze(
///         a.structural_pattern(),
///         IncompleteMethod::Ilu { level },
///         1,
///         limits,
///     )?;
///     let factor = IncompleteFactor::factor(plan, a.view(), IncompleteOptions::new())?;
///     assert_eq!(factor.diagnostics().factor_blocks(), blocks);
/// }
/// assert!(
///     IncompleteSymbolic::analyze(
///         a.structural_pattern(),
///         IncompleteMethod::Ilu { level: usize::MAX },
///         1,
///         limits
///     )
///     .is_err()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003,
/// §10.3–10.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompleteMethod {
    /// Incomplete LU by level of fill, ILU(k).
    ///
    /// Entries of `A` have level 0. Eliminating row `j` from row `i` gives a fill
    /// entry `(i, m)` the level `lev(i, j) + lev(j, m) + 1`, and the entry keeps
    /// the smallest level it receives. Fill entries above `level` are dropped;
    /// `Ilu { level: 0 }` keeps exactly the pattern of `A`.
    Ilu {
        /// Largest level of fill that is kept, `k`. Must be below `usize::MAX`.
        level: usize,
    },
    /// Incomplete LU with a drop tolerance and a fill limit per row, ILUT.
    ///
    /// The tolerance test of [`IncompleteOptions`] is applied to a lower block
    /// before it is divided by the diagonal pivot block, and to an upper block
    /// after elimination. The limit of
    /// [`IncompleteOptions::with_max_off_diagonal`] is applied after that.
    Ilut,
    /// Incomplete Cholesky with the pattern of the lower triangle of `A`, IC(0).
    /// The matrix must be exactly Hermitian.
    Ic0,
    /// Incomplete Cholesky with a drop tolerance and a fill limit per row, ICT.
    ///
    /// The tolerance test is applied to an updated lower block before the
    /// triangular solve with the diagonal block. The diagonal is then computed
    /// from the kept lower blocks only. The matrix must be exactly Hermitian.
    Ict,
}
impl IncompleteMethod {
    fn cholesky(self) -> bool {
        matches!(self, Self::Ic0 | Self::Ict)
    }
    fn threshold(self) -> bool {
        matches!(self, Self::Ilut | Self::Ict)
    }
}
/// Selects what an incomplete factorization does with a zero or small pivot.
///
/// A repaired pivot changes the preconditioner, not the input matrix.
/// [`IncompleteDiagnostics::repaired_pivots`] counts the repairs.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
///     IncompletePivotPolicy, IncompleteSymbolic,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![0.0_f64, 1., 1., 4.],
/// )?;
/// let plan = IncompleteSymbolic::analyze(
///     a.structural_pattern(),
///     IncompleteMethod::Ilu { level: 0 },
///     1,
///     IncompleteLimits::new(4, 4096),
/// )?;
/// // Strict pivots reject the zero diagonal entry.
/// let strict = IncompleteFactor::factor(plan.try_clone()?, a.view(), IncompleteOptions::new());
/// assert_eq!(strict.unwrap_err().kind(), SciOxideErrorKind::Singular);
/// // Repair floors the pivot at max(absolute, relative * row scale) = 1.
/// let repair = IncompletePivotPolicy::Repair {
///     absolute: 1.0,
///     relative: 0.0,
/// };
/// let factor =
///     IncompleteFactor::factor(plan, a.view(), IncompleteOptions::new().with_pivot(repair))?;
/// assert_eq!(factor.diagnostics().repaired_pivots(), 1);
/// assert_eq!(
///     factor.blocks().next().map(|(_, _, pivot)| pivot[0]),
///     Some(1.0)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IncompletePivotPolicy<R: RealScalar> {
    /// Fails with `Singular` on a zero LU pivot or a Cholesky pivot that is not
    /// positive.
    Strict,
    /// Replaces a pivot `d` with `|d| < τ`, or a bad pivot as above, by one of
    /// magnitude `τ = max(absolute, relative · s)`, where `s` is the largest
    /// Frobenius norm of a block in the same block row of `A`. An LU pivot keeps
    /// its sign or phase; a zero or Cholesky pivot becomes `τ`. Both floors must be
    /// finite and nonnegative, and at least one must be positive.
    Repair {
        /// Absolute floor.
        absolute: R,
        /// Floor relative to the block-row scale `s`.
        relative: R,
    },
}
/// Holds the dropping and pivot options of an incomplete factorization.
///
/// The options are checked when the factor is created and are reused by every
/// `refactor`. The matrix is not equilibrated.
///
/// For ILUT and ICT, a candidate off-diagonal block `B` in block row `i` is
/// dropped if
///
/// ```text
/// ‖B‖_F < drop_tolerance · s_i,
/// ```
///
/// where `s_i` is the largest Frobenius norm of a block in row `i` of `A`,
/// diagonal and upper blocks included. A block exactly at the threshold is
/// kept, and so is a zero block at zero tolerance. [`IncompleteMethod`] states
/// when each test is applied. After the row is eliminated, at most
/// `max_off_diagonal` blocks are kept in each triangle of the row: the largest
/// by norm, with the smaller column index first among equal norms. A lower
/// multiplier dropped by this limit has already been used to eliminate the
/// row; the updates it made stay in place.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
///     IncompleteSymbolic,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 1, 3],
///     vec![0, 0, 1],
///     vec![100.0_f64, 1.0, 4.0],
/// )?;
/// let plan = IncompleteSymbolic::analyze(
///     a.structural_pattern(),
///     IncompleteMethod::Ilut,
///     1,
///     IncompleteLimits::new(3, 4096),
/// )?;
/// let options = IncompleteOptions::new()
///     .with_drop_tolerance(0.2)
///     .with_max_off_diagonal(1);
/// let factor = IncompleteFactor::factor(plan.try_clone()?, a.view(), options)?;
/// // The lower residual 1 exceeds 0.2 * 4, so its divided multiplier survives.
/// let (_, _, lower) = factor.blocks().find(|&(r, c, _)| r == 1 && c == 0).unwrap();
/// assert!((lower[0] - 0.01).abs() < 1e-15);
/// // A negative tolerance is rejected before any numeric work.
/// let invalid = IncompleteOptions::new().with_drop_tolerance(-1.0);
/// assert!(IncompleteFactor::factor(plan, a.view(), invalid).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompleteOptions<R: RealScalar> {
    drop_tolerance: R,
    max_off_diagonal: usize,
    pivot: IncompletePivotPolicy<R>,
}
impl<R: RealScalar> Default for IncompleteOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> IncompleteOptions<R> {
    /// Returns options without dropping, without a fill limit and with
    /// [`IncompletePivotPolicy::Strict`].
    pub const fn new() -> Self {
        Self {
            drop_tolerance: R::ZERO,
            max_off_diagonal: usize::MAX,
            pivot: IncompletePivotPolicy::Strict,
        }
    }
    /// Sets the relative drop tolerance of ILUT and ICT; zero drops nothing by
    /// magnitude. It must be finite and nonnegative.
    pub const fn with_drop_tolerance(mut self, value: R) -> Self {
        self.drop_tolerance = value;
        self
    }
    /// Returns the drop tolerance.
    pub const fn drop_tolerance(self) -> R {
        self.drop_tolerance
    }
    /// Sets the largest number of off-diagonal blocks that ILUT and ICT keep in
    /// each triangle of a row.
    ///
    /// Diagonal blocks are always kept. Lower blocks are ranked after division by
    /// the pivot, upper blocks by their final values. Among equal norms the
    /// smaller column index wins.
    pub const fn with_max_off_diagonal(mut self, value: usize) -> Self {
        self.max_off_diagonal = value;
        self
    }
    /// Returns the limit on off-diagonal blocks per row and triangle.
    pub const fn max_off_diagonal(self) -> usize {
        self.max_off_diagonal
    }
    /// Sets the pivot policy.
    pub const fn with_pivot(mut self, value: IncompletePivotPolicy<R>) -> Self {
        self.pivot = value;
        self
    }
    /// Returns the pivot policy.
    pub const fn pivot(self) -> IncompletePivotPolicy<R> {
        self.pivot
    }
    // Checks the caller's `options`.
    fn validate(self) -> Result<Self, SciOxideError> {
        if !self.drop_tolerance.is_finite() || self.drop_tolerance < R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.drop_tolerance"),
                "drop tolerance must be finite and nonnegative",
            ));
        }
        if let IncompletePivotPolicy::Repair { absolute, relative } = self.pivot
            && (!absolute.is_finite()
                || !relative.is_finite()
                || absolute < R::ZERO
                || relative < R::ZERO
                || (absolute == R::ZERO && relative == R::ZERO))
        {
            return Err(invalid(
                scioxide_core::argument!("options.pivot"),
                "repair requires finite nonnegative floors, at least one positive",
            ));
        }
        Ok(self)
    }
}
/// Holds the storage limits of an incomplete factorization.
///
/// Storage is reserved from these limits before factorization and does not
/// grow afterwards.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::IncompleteLimits;
/// let limits = IncompleteLimits::new(16, 4096).with_max_factor_blocks(32);
/// assert_eq!(
///     (limits.max_factor_blocks(), limits.max_workspace_bytes()),
///     (32, 4096)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncompleteLimits {
    max_factor_blocks: usize,
    max_workspace_bytes: usize,
}
impl IncompleteLimits {
    /// Creates limits on the number of factor blocks and on storage bytes.
    ///
    /// Analysis checks only its own storage against the byte limit. The factor
    /// and workspace are checked when they are created, and a factorization that
    /// keeps more blocks than `max_factor_blocks` fails with `ResourceLimit`.
    pub const fn new(max_factor_blocks: usize, max_workspace_bytes: usize) -> Self {
        Self {
            max_factor_blocks,
            max_workspace_bytes,
        }
    }
    /// Returns the limit on stored blocks of `L` and `U` together, diagonal blocks
    /// included. For block size one, blocks are scalar entries.
    pub const fn max_factor_blocks(self) -> usize {
        self.max_factor_blocks
    }
    /// Sets the limit on stored blocks.
    pub const fn with_max_factor_blocks(mut self, value: usize) -> Self {
        self.max_factor_blocks = value;
        self
    }
    /// Returns the limit on bytes for the analysis, the factor and the workspace
    /// together, including the second factor that `refactor` builds and the
    /// solve scratch that the factor allocates itself.
    ///
    /// It also bounds the peak storage during analysis. Allocator overhead and
    /// the input matrix are not counted.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the limit on bytes.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
}
// Reserved shapes: block rows, block side, input entries, factor blocks and method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) rows: usize,
    pub(crate) block: usize,
    pub(crate) entries: usize,
    pub(crate) blocks: usize,
    pub(crate) method: IncompleteMethod,
}
/// Reports the storage, in bytes, that an incomplete factorization needs for a
/// scalar type.
///
/// Byte counts cover the stored data, not allocator overhead.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, SciOxideErrorKind};
/// use scioxide_sparse::{CsrMatrix, IncompleteLimits, IncompleteMethod, IncompleteSymbolic};
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![2.0_f64, -1., -1., 2.],
/// )?;
/// let analyze = |bytes| {
///     IncompleteSymbolic::analyze(
///         a.structural_pattern(),
///         IncompleteMethod::Ilut,
///         1,
///         IncompleteLimits::new(4, bytes),
///     )
/// };
/// let real = analyze(4096)?.workspace_requirements::<f64>(())?;
/// assert_eq!(
///     real.total_bytes(),
///     real.symbolic_bytes() + real.factor_bytes() + real.workspace_bytes()
/// );
/// // Wider scalars need more numeric storage; the analysis is the same size.
/// let complex = analyze(4096)?.workspace_requirements::<Complex<f64>>(())?;
/// assert!(complex.factor_bytes() > real.factor_bytes());
/// assert_eq!(complex.symbolic_bytes(), real.symbolic_bytes());
/// // A limit that fits only the analysis rejects the numeric workspace.
/// let tight = analyze(real.symbolic_bytes())?;
/// assert_eq!(
///     tight.workspace::<f64>(()).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncompleteRequirements {
    pub(crate) shape: Shape,
    pub(crate) limit: usize,
    symbolic_bytes: usize,
    factor_bytes: usize,
    workspace_bytes: usize,
    total_bytes: usize,
}
impl IncompleteRequirements {
    /// Returns the bytes of the analysis.
    pub const fn symbolic_bytes(self) -> usize {
        self.symbolic_bytes
    }
    /// Returns the bytes of one factor.
    pub const fn factor_bytes(self) -> usize {
        self.factor_bytes
    }
    /// Returns the bytes of the workspace, including the second factor used by
    /// `refactor`.
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }
    /// Returns the sum of the three byte counts above.
    pub const fn total_bytes(self) -> usize {
        self.total_bytes
    }
}
/// Reports what the last successful factorization dropped and repaired.
///
/// These counts describe the factor; they do not measure how well it
/// approximates `A`.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteDiagnostics, IncompleteFactor, IncompleteLimits, IncompleteMethod,
///     IncompleteOptions, IncompleteSymbolic,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![4.0_f64, 0.01, 0.01, 4.],
/// )?;
/// let plan = IncompleteSymbolic::analyze(
///     a.structural_pattern(),
///     IncompleteMethod::Ilut,
///     1,
///     IncompleteLimits::new(4, 4096),
/// )?;
/// let factor = IncompleteFactor::factor(
///     plan,
///     a.view(),
///     IncompleteOptions::new().with_drop_tolerance(0.1),
/// )?;
/// // Both off-diagonal entries fall below 0.1 times their row scale 4.
/// let d = factor.diagnostics();
/// assert_eq!(
///     (d.factor_blocks(), d.dropped_blocks(), d.repaired_pivots()),
///     (2, 2, 0)
/// );
/// assert_eq!(IncompleteDiagnostics::default().factor_blocks(), 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IncompleteDiagnostics {
    factor_blocks: usize,
    dropped_blocks: usize,
    repaired_pivots: usize,
}
impl IncompleteDiagnostics {
    /// Returns the number of stored blocks, diagonal blocks included.
    pub const fn factor_blocks(self) -> usize {
        self.factor_blocks
    }
    /// Returns the number of off-diagonal blocks dropped by the tolerance or the
    /// fill limit.
    pub const fn dropped_blocks(self) -> usize {
        self.dropped_blocks
    }
    /// Returns the number of scalar pivots replaced under
    /// [`IncompletePivotPolicy::Repair`].
    pub const fn repaired_pivots(self) -> usize {
        self.repaired_pivots
    }
}
/// Holds the symbolic analysis of an incomplete factorization: the sparsity
/// pattern of `A`, the method and the block size.
///
/// The matrix is split into square blocks of equal size. A block that contains
/// any stored entry is treated as dense, with zeros in its unstored positions.
/// The analysis keeps the scalar pattern, and `refactor` accepts only a matrix
/// with exactly that pattern.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, IncompleteLimits, IncompleteMethod, IncompleteSymbolic};
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![4.0_f64, -1., -1., 4., -1., -1., 4.],
/// )?;
/// let limits = IncompleteLimits::new(7, 4096);
/// let plan = IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ic0, 1, limits)?;
/// assert_eq!((plan.dimension(), plan.block_size()), (3, 1));
/// // Query the storage before reserving any workspace.
/// let requirements = plan.workspace_requirements::<f64>(())?;
/// assert!(requirements.total_bytes() <= limits.max_workspace_bytes());
/// // Block size 2 does not divide the dimension 3.
/// let e = IncompleteSymbolic::analyze(a.structural_pattern(), IncompleteMethod::Ic0, 2, limits)
///     .unwrap_err();
/// assert_eq!(e.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct IncompleteSymbolic {
    pattern: StructuralPattern,
    method: IncompleteMethod,
    block_size: usize,
    limits: IncompleteLimits,
}
impl IncompleteSymbolic {
    /// Returns a copy of the analysis.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            pattern: self.pattern.view().to_owned()?,
            method: self.method,
            block_size: self.block_size,
            limits: self.limits,
        })
    }

    /// Analyzes the pattern of a square matrix for an incomplete factorization.
    ///
    /// Use `block_size` 1 for a scalar factorization. A larger block size must
    /// divide the dimension. For a [`BsrMatrix`](crate::BsrMatrix), pass the
    /// pattern of its scalar entries. The work is proportional to the number of
    /// stored entries; the fill itself is found during factorization.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the pattern is not square.
    /// - `InvalidInput` if `block_size` is zero or does not divide the dimension,
    ///   or if `method` is `Ilu { level: usize::MAX }`.
    /// - `ResourceLimit` if the analysis exceeds `limits.max_workspace_bytes()`.
    /// - `SizeOverflow`, `AllocationFailure` if the storage cannot be sized or
    ///   allocated.
    pub fn analyze(
        pattern: StructuralPatternView<'_>,
        method: IncompleteMethod,
        block_size: usize,
        limits: IncompleteLimits,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::exact(pattern.cols(), pattern.rows())?;
        let symbolic = scioxide_core::quantity!("symbolic_bytes");
        if block_size == 0 || !pattern.rows().is_multiple_of(block_size) {
            let argument = scioxide_core::argument!("block_size");
            return Err(invalid(
                argument,
                "square shape and dividing positive block size required",
            ));
        }
        if matches!(method, IncompleteMethod::Ilu { level: usize::MAX }) {
            return Err(invalid(
                scioxide_core::argument!("method"),
                "fill level must be below usize::MAX",
            ));
        }
        let bytes = mul(
            add(
                add(pattern.rows(), 1, symbolic)?,
                pattern.col_indices().len(),
                symbolic,
            )?,
            size_of::<usize>(),
            symbolic,
        )?;
        if bytes > limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                limits.max_workspace_bytes,
            ));
        }
        Ok(Self {
            pattern: pattern.to_owned()?,
            method,
            block_size,
            limits,
        })
    }
    /// Returns the order of the matrix, in scalar entries.
    pub fn dimension(&self) -> usize {
        self.pattern.view().rows()
    }
    /// Returns the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    /// Returns the method.
    pub fn method(&self) -> IncompleteMethod {
        self.method
    }
    /// Returns the storage limits.
    pub fn limits(&self) -> IncompleteLimits {
        self.limits
    }
    /// Returns the scalar sparsity pattern of `A`.
    pub fn pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }
    fn rows(&self) -> usize {
        self.dimension() / self.block_size
    }
    pub(crate) fn shape(&self) -> Shape {
        Shape {
            rows: self.rows(),
            block: self.block_size,
            entries: self.pattern().col_indices().len(),
            blocks: self.limits.max_factor_blocks,
            method: self.method,
        }
    }
    /// Returns the storage that the factor and its workspace need for scalar
    /// type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn workspace_requirements<S: NumericScalar>(
        &self,
        (): (),
    ) -> Result<IncompleteRequirements, SciOxideError> {
        let symbolic = scioxide_core::quantity!("symbolic_bytes");
        let factor = scioxide_core::quantity!("factor_bytes");
        let workspace = scioxide_core::quantity!("workspace_bytes");
        let total = scioxide_core::quantity!("total_bytes");
        let n = self.rows();
        let b = self.block_size;
        let b2 = mul(b, b, scioxide_core::quantity!("block_area"))?;
        let f = self.limits.max_factor_blocks;
        let z = self.pattern().col_indices().len();
        let symbolic_bytes = mul(
            add(add(self.dimension(), 1, symbolic)?, z, symbolic)?,
            size_of::<usize>(),
            symbolic,
        )?;
        // factors: ptr, col, level, local pivot indices, blocks, decomposed diagonal blocks.
        let indices = add(
            add(add(n, 1, factor)?, mul(2, f, factor)?, factor)?,
            mul(n, b, factor)?,
            factor,
        )?;
        let factor_bytes = add(
            mul(indices, size_of::<usize>(), factor)?,
            mul(mul(add(f, n, factor)?, b2, factor)?, size_of::<S>(), factor)?,
            factor,
        )?;
        // Source, indexed block row and temporary block; row levels, touched columns,
        // heap and positions; seen/row membership flags; threshold norm ranking.
        let scalars = add(add(z, mul(n, b2, workspace)?, workspace)?, b2, workspace)?;
        let mut extra = add(
            add(
                mul(scalars, size_of::<S>(), workspace)?,
                mul(mul(4, n, workspace)?, size_of::<usize>(), workspace)?,
                workspace,
            )?,
            add(z, n, workspace)?,
            workspace,
        )?;
        if self.method.threshold() {
            extra = add(
                extra,
                mul(
                    n,
                    size_of::<(usize, scioxide_core::Scaled<S::Real>)>(),
                    workspace,
                )?,
                workspace,
            )?;
        }
        if matches!(self.method, IncompleteMethod::Ict) {
            extra = add(
                extra,
                mul(
                    add(n, mul(2, f, workspace)?, workspace)?,
                    size_of::<usize>(),
                    workspace,
                )?,
                workspace,
            )?;
        }
        let workspace_bytes = add(factor_bytes, extra, workspace)?;
        Ok(IncompleteRequirements {
            shape: self.shape(),
            limit: self.limits.max_workspace_bytes,
            symbolic_bytes,
            factor_bytes,
            workspace_bytes,
            total_bytes: add(
                add(symbolic_bytes, factor_bytes, total)?,
                workspace_bytes,
                total,
            )?,
        })
    }
    /// Creates a workspace for `refactor`, reserved for this analysis.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the total storage exceeds the byte limit.
    /// - `SizeOverflow`, `AllocationFailure` if the storage cannot be sized or
    ///   allocated.
    pub fn workspace<S: NumericScalar>(
        &self,
        request: (),
    ) -> Result<IncompleteWorkspace<S>, SciOxideError> {
        let mut workspace = IncompleteWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}
