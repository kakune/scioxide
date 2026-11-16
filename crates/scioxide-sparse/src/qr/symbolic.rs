use super::*;

/// Sets the storage limits of a sparse QR factorization.
///
/// `max_factor_entries` bounds the stored entries of the Householder vectors
/// and of `R` together, for each decomposition: the factor itself and each of
/// the two decompositions inside a [`SparseQrMinimumNormPlan`](crate::SparseQrMinimumNormPlan).
/// `max_active_entries` bounds the nonzero entries of the submatrix that is
/// still being reduced. `max_workspace_bytes` bounds the bytes of everything
/// held at the same time: the analysis, the factor, its workspace and a
/// minimum-norm plan. No limit grows automatically.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseQr, SparseQrLimits, SparseQrSymbolic};
/// let limits = SparseQrLimits::new(1, 8, 1 << 16);
/// assert_eq!(limits.with_max_factor_entries(8).max_factor_entries(), 8);
/// let a = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![3.0_f64, 4.])?;
/// let factor = |limits| {
///     SparseQrSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits)
///         .and_then(|symbolic| SparseQr::factor(symbolic, a.view()))
/// };
/// // The Householder vector has two entries and R has one, so one entry is too few.
/// assert_eq!(
///     factor(limits).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// assert!(factor(limits.with_max_factor_entries(8)).is_ok());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseQrLimits {
    pub(super) max_factor_entries: usize,
    pub(super) max_active_entries: usize,
    pub(super) max_workspace_bytes: usize,
}
impl SparseQrLimits {
    /// Creates limits on the factor entries, the active entries and the total
    /// bytes.
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
    /// Sets the maximum number of stored Householder vector and `R` entries
    /// per decomposition.
    pub const fn with_max_factor_entries(mut self, n: usize) -> Self {
        self.max_factor_entries = n;
        self
    }
    /// Sets the maximum number of nonzero entries in the submatrix being
    /// reduced.
    pub const fn with_max_active_entries(mut self, n: usize) -> Self {
        self.max_active_entries = n;
        self
    }
    /// Sets the maximum total bytes held at the same time.
    pub const fn with_max_workspace_bytes(mut self, n: usize) -> Self {
        self.max_workspace_bytes = n;
        self
    }
    /// Returns the maximum number of stored Householder vector and `R` entries
    /// per decomposition.
    pub const fn max_factor_entries(self) -> usize {
        self.max_factor_entries
    }
    /// Returns the maximum number of nonzero entries in the submatrix being
    /// reduced.
    pub const fn max_active_entries(self) -> usize {
        self.max_active_entries
    }
    /// Returns the maximum total bytes held at the same time.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
}
/// Holds upper bounds on the storage, in bytes, of a sparse QR analysis, its
/// factor, its workspace and a minimum-norm plan.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseQrLimits, SparseQrSymbolic};
/// let a = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![3.0_f64, 4.])?;
/// let limits = SparseQrLimits::new(8, 8, 1 << 16);
/// let analyze =
///     |limits| SparseQrSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits);
/// let r = analyze(limits)?.workspace_requirements::<f64>(())?;
/// assert!(
///     r.total_bytes()
///         >= r.symbolic_bytes()
///             + r.factor_bytes()
///             + r.factor_workspace_bytes()
///             + r.solve_workspace_bytes()
/// );
/// // Analysis fits one byte short of the total, but the reservation does not.
/// let tight = analyze(limits.with_max_workspace_bytes(r.total_bytes() - 1))?;
/// assert_eq!(
///     tight.workspace::<f64>(()).unwrap_err().kind(),
///     SciOxideErrorKind::ResourceLimit
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseQrRequirements {
    pub(super) symbolic_bytes: usize,
    pub(super) factor_bytes: usize,
    factor_workspace_bytes: usize,
    solve_workspace_bytes: usize,
    total_bytes: usize,
    // Analysis, raw factor and its reserved refactor workspace.
    pub(super) raw_bytes: usize,
    // Raw payload plus the retained compression and secondary factors.
    pub(super) prepared_bytes: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) source: usize,
    pub(super) limits: SparseQrLimits,
}
impl SparseQrRequirements {
    /// Returns the bytes of the symbolic analysis.
    pub const fn symbolic_bytes(self) -> usize {
        self.symbolic_bytes
    }
    /// Returns the bytes of one factor.
    pub const fn factor_bytes(self) -> usize {
        self.factor_bytes
    }
    /// Returns the bytes of the refactorization workspace: a second factor and
    /// the submatrix being reduced.
    pub const fn factor_workspace_bytes(self) -> usize {
        self.factor_workspace_bytes
    }
    /// Returns the bytes of a minimum-norm plan and the scratch of one
    /// right-hand side, without the factor.
    pub const fn solve_workspace_bytes(self) -> usize {
        self.solve_workspace_bytes
    }
    /// Returns the largest number of bytes held at the same time, including the
    /// temporary storage used to build a minimum-norm plan.
    pub const fn total_bytes(self) -> usize {
        self.total_bytes
    }
    // Admit the total payload against the configured byte limit.
    pub(super) fn admit(self) -> Result<(), SciOxideError> {
        if self.total_bytes > self.limits.max_workspace_bytes {
            Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                self.total_bytes,
                self.limits.max_workspace_bytes,
            ))
        } else {
            Ok(())
        }
    }
}

/// Holds the symbolic analysis of a rectangular sparsity pattern for
/// [`SparseQr`].
///
/// The analysis stores the pattern and a column permutation `q` with
/// `q[new] = old`. With
/// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing), the
/// default, `q` is an approximate minimum degree ordering of the pattern of
/// `AᵀA`, computed from the rows of `A` without forming `AᵀA`. Factorization
/// keeps this column order exactly; there is no column pivoting, so the
/// diagonal of `R` is not ordered by magnitude. Rank decisions are made by
/// [`SparseQrMinimumNormPlan`](crate::SparseQrMinimumNormPlan). Stored zeros are
/// part of the pattern. The byte limit also bounds the storage used to compute
/// the ordering.
///
/// ```
/// use scioxide_core::{SciOxideErrorKind, Permutation};
/// use scioxide_sparse::{CsrMatrix, SparseOrdering, SparseQrLimits, SparseQrSymbolic};
/// // Pattern of [[x, .], [., x], [x, x]].
/// let a = CsrMatrix::from_raw(
///     3,
///     2,
///     vec![0, 1, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![1.0_f64, 1., 1., 1.],
/// )?;
/// let limits = SparseQrLimits::new(16, 16, 1 << 16);
/// let symbolic =
///     SparseQrSymbolic::analyze(a.structural_pattern(), SparseOrdering::FillReducing, limits)?;
/// assert_eq!((symbolic.rows(), symbolic.cols()), (3, 2));
/// assert_eq!(symbolic.column_permutation().new_to_old().len(), 2);
/// // A supplied column order must match the column count.
/// let order = Permutation::new(vec![2, 0, 1])?;
/// let error = SparseQrSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Supplied(&order),
///     limits,
/// )
/// .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseQrSymbolic {
    pub(super) pattern: StructuralPattern,
    pub(super) ordering: Permutation,
    pub(super) col_ptr: Vec<usize>,
    pub(super) rows: Vec<usize>,
    pub(super) positions: Vec<usize>,
    pub(super) limits: SparseQrLimits,
}

impl SparseQrSymbolic {
    /// Copies the analysis without repeating it.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            pattern: self.pattern.view().to_owned()?,
            ordering: Permutation::new(scioxide_core::backend::try_copy_slice(
                self.ordering.new_to_old(),
                scioxide_core::quantity!("symbolic_copy"),
            )?)?,
            col_ptr: scioxide_core::backend::try_copy_slice(
                &self.col_ptr,
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

    /// Analyzes the sparsity pattern of an `m × n` matrix and computes its
    /// column ordering.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a supplied ordering has a length other than
    ///   `n`.
    /// - `ResourceLimit` if the analysis, including the temporary storage of the
    ///   ordering, exceeds `limits.max_workspace_bytes()`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn analyze(
        pattern: StructuralPatternView<'_>,
        ordering: crate::SparseOrdering<'_>,
        limits: SparseQrLimits,
    ) -> Result<Self, SciOxideError> {
        let preparation = scioxide_core::quantity!("symbolic_preparation_bytes");
        if let crate::SparseOrdering::Supplied(order) = ordering {
            crate::helpers::exact(order.new_to_old().len(), pattern.cols())?;
        }
        let (m, n) = (pattern.rows(), pattern.cols());

        let z = pattern.col_indices().len();
        let symbolic = mul(
            add(
                add(
                    mul(2, add(m, 1, preparation)?, preparation)?,
                    mul(3, z, preparation)?,
                    preparation,
                )?,
                mul(4, add(n, 1, preparation)?, preparation)?,
                preparation,
            )?,
            size_of::<usize>(),
            preparation,
        )?;
        if symbolic > limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                symbolic,
                limits.max_workspace_bytes,
            ));
        }
        let ordering = ordering.prepare(
            pattern,
            crate::ordering::OrderingGraph::Normal,
            limits.max_workspace_bytes,
        )?;
        let mut rows = try_filled(z, 0usize, scioxide_core::quantity!("column_rows"))?;
        let mut positions = try_filled(z, 0usize, scioxide_core::quantity!("column_positions"))?;
        let col_ptr = crate::helpers::counting_transpose(
            n,
            (0..m).flat_map(|row| {
                (pattern.row_ptr()[row]..pattern.row_ptr()[row + 1])
                    .map(move |pos| (pattern.col_indices()[pos], (row, pos)))
            }),
            |slot, (row, pos)| {
                rows[slot] = row;
                positions[slot] = pos;
            },
        )?;
        let result = Self {
            pattern: pattern.to_owned()?,
            ordering,
            col_ptr,
            rows,
            positions,
            limits,
        };
        Ok(result)
    }

    /// Returns the analyzed sparsity pattern.
    pub fn pattern(&self) -> StructuralPatternView<'_> {
        self.pattern.view()
    }
    /// Returns the column permutation `q`, with `q[new] = old`.
    pub fn column_permutation(&self) -> &Permutation {
        &self.ordering
    }
    /// Returns the number of rows `m`.
    pub fn rows(&self) -> usize {
        self.pattern.view().rows()
    }
    /// Returns the number of columns `n`.
    pub fn cols(&self) -> usize {
        self.pattern.view().cols()
    }
    /// Returns the storage limits of this analysis.
    pub const fn limits(&self) -> SparseQrLimits {
        self.limits
    }
    /// Returns upper bounds on the storage of a factor with scalar type `S`,
    /// its refactorization workspace and a minimum-norm plan.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn workspace_requirements<S: NumericScalar>(
        &self,
        (): (),
    ) -> Result<SparseQrRequirements, SciOxideError> {
        let q = scioxide_core::quantity!("qr_payload");
        let (m, n, z) = (self.rows(), self.cols(), self.rows.len());
        let k = m.min(n);
        let f = self.limits.max_factor_entries;
        let a = self.limits.max_active_entries;
        let symbolic_bytes = mul(
            add(
                add(mul(2, add(m, 1, q)?, q)?, mul(3, z, q)?, q)?,
                mul(4, add(n, 1, q)?, q)?,
                q,
            )?,
            size_of::<usize>(),
            q,
        )?;
        let factor_bytes = super::numeric::data_bytes::<S>(m, n, z, f)?;
        let factor_workspace_bytes = add(
            add(factor_bytes, super::numeric::kernel_bytes::<S>(m, n, a)?, q)?,
            z,
            q,
        )?;
        let extra = add(
            super::numeric::data_bytes::<S>(k, n, 0, f)?,
            super::numeric::data_bytes::<S>(n, k, 0, f)?,
            q,
        )?;
        let raw_bytes = add(
            add(symbolic_bytes, factor_bytes, q)?,
            factor_workspace_bytes,
            q,
        )?;
        // Keep charging the reserved refactor workspace while the solve plan owns
        // the raw factor. Callers may retain that workspace for a later epoch.
        let prepared_bytes = add(raw_bytes, extra, q)?;
        let scratch = mul(mul(6, m.max(n), q)?, size_of::<S>(), q)?;
        let solve_workspace_bytes = add(extra, scratch, q)?;
        let peak_kernel = super::numeric::kernel_bytes::<S>(k, n, a)?
            .max(super::numeric::kernel_bytes::<S>(n, k, a)?);
        let total_bytes = add(prepared_bytes, peak_kernel.max(scratch), q)?;
        Ok(SparseQrRequirements {
            symbolic_bytes,
            factor_bytes,
            factor_workspace_bytes,
            solve_workspace_bytes,
            total_bytes,
            raw_bytes,
            prepared_bytes,
            rows: m,
            cols: n,
            source: z,
            limits: self.limits,
        })
    }
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
    ) -> Result<SparseQrWorkspace<S>, SciOxideError> {
        SparseQrWorkspace::reserve(&self.workspace_requirements::<S>(request)?)
    }
}
