use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct Entry<S> {
    pub row: usize,
    pub value: S,
}
#[derive(Debug, Clone, Copy)]
pub(super) struct REntry<S> {
    pub row: usize,
    pub col: usize,
    pub value: S,
}

#[derive(Debug)]
pub(super) struct Data<S: NumericScalar> {
    pub v: Vec<Entry<S>>,
    pub v_ptr: Vec<usize>,
    pub beta: Vec<S::Real>,
    pub r: Vec<REntry<S>>,
    pub r_ptr: Vec<usize>,
    pub r_col_ptr: Vec<usize>,
    pub r_col_index: Vec<usize>,
    pub diagonal: Vec<S>,
    pub order: Vec<usize>,
    pub source: Vec<S>,
}
// A published factor's storage becomes refactor staging after the next epoch,
// so a copy keeps the reserved reflector and R entry capacity.
impl<S: NumericScalar> Clone for Data<S> {
    fn clone(&self) -> Self {
        fn reserved<T: Copy>(values: &Vec<T>) -> Vec<T> {
            let mut copy = Vec::with_capacity(values.capacity());
            copy.extend_from_slice(values);
            copy
        }
        Self {
            v: reserved(&self.v),
            v_ptr: self.v_ptr.clone(),
            beta: self.beta.clone(),
            r: reserved(&self.r),
            r_ptr: self.r_ptr.clone(),
            r_col_ptr: self.r_col_ptr.clone(),
            r_col_index: self.r_col_index.clone(),
            diagonal: self.diagonal.clone(),
            order: self.order.clone(),
            source: self.source.clone(),
        }
    }
}
impl<S: NumericScalar> Data<S> {
    pub(super) fn new(
        rows: usize,
        cols: usize,
        source_len: usize,
        max_entries: usize,
    ) -> Result<Self, SciOxideError> {
        let k = rows.min(cols);
        let q = scioxide_core::quantity!("qr_factor");
        Ok(Self {
            v: try_with_capacity(max_entries, q)?,
            v_ptr: try_filled(add(k, 1, q)?, 0, q)?,
            beta: try_filled(k, S::Real::ZERO, q)?,
            r: try_with_capacity(max_entries, q)?,
            r_ptr: try_filled(add(k, 1, q)?, 0, q)?,
            r_col_ptr: try_filled(add(cols, 1, q)?, 0, q)?,
            r_col_index: try_filled(max_entries, 0, q)?,
            diagonal: try_filled(k, S::ZERO, q)?,
            order: try_filled(cols, 0, q)?,
            source: try_filled(source_len, S::ZERO, q)?,
        })
    }
    fn check_entry(&self, limit: usize) -> Result<(), SciOxideError> {
        let required = self.v.len() + self.r.len() + 1;
        if required > limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("factor_entries"),
                required,
                limit,
            ));
        }
        Ok(())
    }
    fn push_v(&mut self, row: usize, value: S, limit: usize) -> Result<(), SciOxideError> {
        self.check_entry(limit)?;
        self.v.push(Entry { row, value });
        Ok(())
    }
    fn push_r(
        &mut self,
        (row, col): (usize, usize),
        value: S,
        limit: usize,
    ) -> Result<(), SciOxideError> {
        if value != S::ZERO {
            self.check_entry(limit)?;
            self.r.push(REntry { row, col, value });
        }
        Ok(())
    }
    fn index_r(&mut self, position: &[usize]) {
        self.r_ptr.fill(0);
        self.r_col_ptr.fill(0);
        self.diagonal.fill(S::ZERO);
        for e in &mut self.r {
            e.col = position[e.col];
        }
        self.r.sort_unstable_by_key(|e| (e.row, e.col));
        for e in &self.r {
            self.r_ptr[e.row + 1] += 1;
            self.r_col_ptr[e.col + 1] += 1;
            if e.row == e.col {
                self.diagonal[e.row] = e.value;
            }
        }
        for i in 1..self.r_ptr.len() {
            self.r_ptr[i] += self.r_ptr[i - 1];
        }
        for i in 1..self.r_col_ptr.len() {
            self.r_col_ptr[i] += self.r_col_ptr[i - 1];
        }
        // Row positions serve as column insertion cursors without an extra allocation.
        for (i, e) in self.r.iter().enumerate().rev() {
            self.r_col_ptr[e.col + 1] -= 1;
            self.r_col_index[self.r_col_ptr[e.col + 1]] = i;
        }
        for i in 0..self.order.len() {
            self.r_col_ptr[i] = self.r_col_ptr[i + 1];
        }
        self.r_col_ptr[self.order.len()] = self.r.len();
    }
}
const NONE: usize = usize::MAX;
#[derive(Debug, Clone, Copy)]
struct Slot<S> {
    row: usize,
    col: usize,
    value: S,
    rprev: usize,
    rnext: usize,
    cnext: usize,
}
/// Stable bounded slots; replacing a column only unlinks its own entries.
#[derive(Debug)]
pub(super) struct Kernel<S: NumericScalar> {
    slots: Vec<Slot<S>>,
    row_heads: Vec<usize>,
    col_heads: Vec<usize>,
    free: usize,
    len: usize,
    limit: usize,
    pub position: Vec<usize>,
    pub dense: Vec<S>,
    touched: Vec<usize>,
    marked: Vec<bool>,
    columns: Vec<usize>,
    col_marked: Vec<bool>,
    #[cfg(test)]
    pub visits: usize,
}
impl<S: NumericScalar> Kernel<S> {
    pub(super) fn new(rows: usize, cols: usize, limit: usize) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("qr_active_workspace");
        Ok(Self {
            slots: try_with_capacity(limit, q)?,
            row_heads: try_filled(rows, NONE, q)?,
            col_heads: try_filled(cols, NONE, q)?,
            free: NONE,
            len: 0,
            limit,
            position: try_filled(cols, 0, q)?,
            dense: try_filled(rows, S::ZERO, q)?,
            touched: try_with_capacity(rows, q)?,
            marked: try_filled(rows, false, q)?,
            columns: try_with_capacity(cols, q)?,
            col_marked: try_filled(cols, false, q)?,
            #[cfg(test)]
            visits: 0,
        })
    }
    // Empty the active matrix and cap it at the analysis' `limit`, which a
    // compatible reservation never exceeds.
    pub(super) fn clear(&mut self, limit: usize) {
        self.limit = limit;
        self.slots.clear();
        self.row_heads.fill(NONE);
        self.col_heads.fill(NONE);
        self.free = NONE;
        self.len = 0;
        self.clear_dense();
        self.columns.clear();
        self.col_marked.fill(false);
        #[cfg(test)]
        {
            self.visits = 0;
        }
    }
    pub(super) fn insert(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        if value == S::ZERO {
            return Ok(());
        }
        let value = finite_candidate(value)?;
        if self.len == self.limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("active_entries"),
                self.len + 1,
                self.limit,
            ));
        }
        let slot = Slot {
            row,
            col,
            value,
            rprev: NONE,
            rnext: self.row_heads[row],
            cnext: self.col_heads[col],
        };
        let id = if self.free == NONE {
            let id = self.slots.len();
            self.slots.push(slot);
            id
        } else {
            let id = self.free;
            self.free = self.slots[id].cnext;
            self.slots[id] = slot;
            id
        };
        if slot.rnext != NONE {
            self.slots[slot.rnext].rprev = id;
        }
        self.row_heads[row] = id;
        self.col_heads[col] = id;
        self.len += 1;
        Ok(())
    }
    fn remove_column(&mut self, col: usize) {
        let mut id = self.col_heads[col];
        self.col_heads[col] = NONE;
        while id != NONE {
            let slot = self.slots[id];
            if slot.rprev == NONE {
                self.row_heads[slot.row] = slot.rnext;
            } else {
                self.slots[slot.rprev].rnext = slot.rnext;
            }
            if slot.rnext != NONE {
                self.slots[slot.rnext].rprev = slot.rprev;
            }
            let next = slot.cnext;
            self.slots[id].cnext = self.free;
            self.free = id;
            self.len -= 1;
            id = next;
        }
    }
    fn clear_dense(&mut self) {
        for row in self.touched.drain(..) {
            self.marked[row] = false;
            self.dense[row] = S::ZERO;
        }
    }
    fn touch(&mut self, row: usize) {
        if !self.marked[row] {
            self.marked[row] = true;
            self.touched.push(row);
            self.dense[row] = S::ZERO;
        }
    }
    fn scatter(&mut self, col: usize) {
        self.clear_dense();
        let mut id = self.col_heads[col];
        while id != NONE {
            let e = self.slots[id];
            self.touch(e.row);
            self.dense[e.row] = e.value;
            id = e.cnext;
        }
    }
    fn norm(&self, col: usize) -> Result<S::Real, SciOxideError> {
        let mut norm = ScaledNorm::new();
        let mut id = self.col_heads[col];
        while id != NONE {
            let e = self.slots[id];
            norm.add_magnitude(e.value.abs())?;
            id = e.cnext;
        }
        norm.finish()
    }
    fn affected(&mut self, row: usize, pivot: usize, logical: usize) {
        let mut id = self.row_heads[row];
        while id != NONE {
            let e = self.slots[id];
            if e.col != pivot && self.position[e.col] > logical && !self.col_marked[e.col] {
                self.columns.push(e.col);
                self.col_marked[e.col] = true;
            }
            id = e.rnext;
        }
    }
}

/// Holds the storage for recomputing a sparse QR factorization without
/// allocation.
///
/// The workspace holds a second factor and the submatrix being reduced.
/// [`SparseQr::refactor`] computes into it and replaces the current factor only
/// on success. Create one with [`SparseQrSymbolic::workspace`].
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOrdering, SparseQr, SparseQrLimits, SparseQrSymbolic, SparseSolveOptions,
/// };
/// let analyze = |a: &CsrMatrix<f64>| {
///     SparseQrSymbolic::analyze(
///         a.structural_pattern(),
///         SparseOrdering::Natural,
///         SparseQrLimits::new(8, 8, 1 << 16),
///     )
/// };
/// let a = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![3.0_f64, 4.])?;
/// let symbolic = analyze(&a)?;
/// let mut workspace = symbolic.workspace::<f64>(())?;
/// let requirements = symbolic.workspace_requirements::<f64>(())?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let mut qr = SparseQr::factor_scoped(symbolic, a.view(), options, scope)?;
/// qr.refactor(a.view(), options, &mut workspace, scope)?;
/// assert!(workspace.last_attempt().is_some());
/// // A workspace reserved for another shape is rejected before any arithmetic.
/// let tall = CsrMatrix::from_raw(3, 1, vec![0, 1, 2, 3], vec![0, 0, 0], vec![1.0_f64, 2., 2.])?;
/// let mut other = analyze(&tall)?.workspace::<f64>(())?;
/// let error = qr
///     .refactor(a.view(), options, &mut other, scope)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseQrWorkspace<S: NumericScalar> {
    candidate: Data<S>,
    kernel: Kernel<S>,
    seen: Vec<bool>,
    epoch: crate::factor_solve::Epoch,
}
impl<S: NumericScalar> SparseQrWorkspace<S> {
    /// Returns the record of the most recent factorization that used this
    /// workspace.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.epoch.record()
    }

    /// Checks whether this workspace fits the requirements `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the bytes held by the workspace exceed the byte
    ///   limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(
        &self,
        r: &SparseQrRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        self.admit_retained(r)?;
        Ok(if self.shortfall(r).is_none() {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    // Rejects a retained byte overrun, then the first incompatible extent as
    // `DimensionMismatch` of the required and available quantities.
    fn require(&self, r: &SparseQrRequirements) -> Result<(), SciOxideError> {
        self.admit_retained(r)?;
        match self.shortfall(r) {
            Some((required, available)) => {
                Err(SciOxideError::dimension_mismatch(required, available))
            }
            None => Ok(()),
        }
    }
    // First `(required, available)` pair whose shape differs or whose reserved
    // entry capacity falls short.
    fn shortfall(&self, r: &SparseQrRequirements) -> Option<(usize, usize)> {
        [
            (r.cols, self.candidate.order.len()),
            (r.rows.min(r.cols), self.candidate.beta.len()),
            (r.source, self.seen.len()),
            (r.rows, self.kernel.dense.len()),
            (r.cols, self.kernel.col_heads.len()),
        ]
        .into_iter()
        .find(|(required, available)| required != available)
        .or_else(|| {
            [
                (r.limits.max_factor_entries, self.candidate.v.capacity()),
                (r.limits.max_factor_entries, self.candidate.r.capacity()),
                (r.limits.max_active_entries, self.kernel.slots.capacity()),
            ]
            .into_iter()
            .find(|(required, available)| available < required)
        })
    }
    fn admit_retained(&self, r: &SparseQrRequirements) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("qr_workspace_bytes");
        let retained = add(
            add(
                self.candidate.retained_bytes()?,
                self.kernel.retained_bytes()?,
                q,
            )?,
            self.seen.capacity(),
            q,
        )?;
        let bytes = add(add(r.symbolic_bytes, r.factor_bytes, q)?, retained, q)?;
        if bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        Ok(())
    }
    /// Grows or replaces the storage of this workspace to fit `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if `r.total_bytes()`, or the storage held while
    ///   growing, exceeds the byte limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn try_reserve(&mut self, r: &SparseQrRequirements) -> Result<(), SciOxideError> {
        r.admit()?;
        if self.check(r)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        // A failed later allocation may leave earlier replacements retained.
        // Admit each such intermediate layout before changing any buffer.
        let q = scioxide_core::quantity!("qr_workspace_bytes");
        let candidate = data_bytes::<S>(r.rows, r.cols, r.source, r.limits.max_factor_entries)?;
        let kernel = kernel_bytes::<S>(r.rows, r.cols, r.limits.max_active_entries)?;
        let first = add(self.kernel.retained_bytes()?, self.seen.capacity(), q)?;
        let later = add(kernel, self.seen.capacity().max(r.source), q)?;
        let projected = add(
            add(r.symbolic_bytes, r.factor_bytes, q)?,
            add(candidate, first.max(later), q)?,
            q,
        )?;
        if projected > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                projected,
                r.limits.max_workspace_bytes,
            ));
        }
        self.candidate = Data::new(r.rows, r.cols, r.source, r.limits.max_factor_entries)?;
        self.kernel = Kernel::new(r.rows, r.cols, r.limits.max_active_entries)?;
        self.seen = try_filled(r.source, false, scioxide_core::quantity!("source_seen"))?;
        let _ = self.check(r)?;
        Ok(())
    }
    // Fresh factorization buffers for an admitted request.
    pub(super) fn reserve(r: &SparseQrRequirements) -> Result<Self, SciOxideError> {
        r.admit()?;
        Ok(Self {
            candidate: Data::new(r.rows, r.cols, r.source, r.limits.max_factor_entries)?,
            kernel: Kernel::new(r.rows, r.cols, r.limits.max_active_entries)?,
            seen: try_filled(r.source, false, scioxide_core::quantity!("source_seen"))?,
            epoch: Default::default(),
        })
    }
}
/// Holds the sparse Householder QR factorization of an `m × n` matrix.
///
/// The factor satisfies
///
/// ```text
/// A[:, q] = Q R,
/// ```
///
/// where `q` is the column permutation of the [analysis](SparseQrSymbolic),
/// `Q = H₀ H₁ ⋯` is an `m × m` unitary product of Householder reflections
/// stored as sparse vectors, and `R` is `m × n` upper trapezoidal. Columns are
/// reduced in the order `q` with no column pivoting. Factorization is
/// sequential, and each column counts as one iteration toward the iteration
/// limit.
///
/// Apply `Q`, `Qᵀ` or `Qᴴ` with [`apply_q_into`](Self::apply_q_into). For
/// least-squares and minimum-norm solutions, pass the factor to
/// [`SparseQrMinimumNormPlan::build`](crate::SparseQrMinimumNormPlan::build).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseOrdering, SparseQr, SparseQrLimits, SparseQrSymbolic,
///     SparseSolveOptions,
/// };
/// // A = [3, 4]ᵀ, so Qᵀ A = R = [±5, 0]ᵀ with |R₀₀| = ‖A‖₂ = 5.
/// let a = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![3.0_f64, 4.])?;
/// let symbolic = SparseQrSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Natural,
///     SparseQrLimits::new(8, 8, 1 << 16),
/// )?;
/// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let mut qr = SparseQr::factor_scoped(symbolic, a.view(), options, scope)?;
/// let r: f64 = qr.r_entries().map(|(_, _, value)| value).sum();
/// let mut qt_a = [0.; 2];
/// let mut staging = qr.workspace(1)?;
/// qr.apply_q_into(
///     &[3., 4.],
///     SparseOperation::Transpose,
///     options,
///     &mut qt_a,
///     &mut staging,
///     scope,
/// )?;
/// assert!((r.abs() - 5.).abs() < 1e-14 && (qt_a[0] - r).abs() < 1e-14 && qt_a[1].abs() < 1e-14);
/// // Refactor reuses reserved storage; a different pattern keeps the previous factor.
/// let mut workspace = qr.symbolic().workspace::<f64>(())?;
/// let b = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![6.0_f64, 8.])?;
/// qr.refactor(b.view(), options, &mut workspace, scope)?;
/// let other = CsrMatrix::from_raw(2, 1, vec![0, 1, 1], vec![0], vec![1.0_f64])?;
/// let error = qr
///     .refactor(other.view(), options, &mut workspace, scope)
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
/// assert!(
///     qr.r_entries()
///         .all(|(_, _, value)| (value.abs() - 10.).abs() < 1e-13)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseQr<S: NumericScalar> {
    pub(super) symbolic: SparseQrSymbolic,
    pub(super) data: Data<S>,
}
impl<S: NumericScalar> SparseQr<S> {
    /// Computes the QR factorization of `matrix`, whose pattern `symbolic`
    /// analyzed, with the default [`SparseSolveOptions`] in a fresh scope.
    ///
    /// This is [`factor_scoped`](Self::factor_scoped) with
    /// [`SparseSolveOptions::new`] and [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`factor_scoped`](Self::factor_scoped).
    pub fn factor(
        symbolic: SparseQrSymbolic,
        matrix: SparseView<'_, S>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_scoped(
            symbolic,
            matrix,
            SparseSolveOptions::new(),
            ExecutionContext::sequential_native(),
        )
    }
    /// Computes the QR factorization of `matrix`, whose pattern `symbolic`
    /// analyzed, inside the caller's scope.
    ///
    /// This allocates the factor and a temporary workspace. To factor a
    /// sequence of matrices without allocation, use
    /// [`refactor`](Self::refactor).
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the factor and workspace exceed the byte limit, and
    ///   `SizeOverflow` or `AllocationFailure` if they cannot be allocated.
    /// - The errors of [`refactor`](Self::refactor).
    pub fn factor_scoped<'s, 'c>(
        symbolic: SparseQrSymbolic,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<Self, SciOxideError> {
        let mut workspace = SparseQrWorkspace::reserve(&symbolic.workspace_requirements::<S>(())?)?;
        let mut factor = Self {
            data: Data::new(
                symbolic.rows(),
                symbolic.cols(),
                symbolic.rows.len(),
                symbolic.limits.max_factor_entries,
            )?,
            symbolic,
        };
        factor.refactor(matrix, options, &mut workspace, scope)?;
        Ok(factor)
    }
    /// Returns the symbolic analysis.
    pub const fn symbolic(&self) -> &SparseQrSymbolic {
        &self.symbolic
    }
    /// Returns the column permutation `q`, with `q[new] = old`.
    pub fn column_permutation(&self) -> &[usize] {
        &self.data.order
    }
    /// Returns the number of stored entries of the Householder vectors and of
    /// `R`.
    pub fn factor_entries(&self) -> usize {
        self.data.v.len() + self.data.r.len()
    }
    /// Returns the stored entries `(i, j, R[i, j])` of `R`, with `j` indexing
    /// the permuted columns.
    pub fn r_entries(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        self.data
            .r
            .iter()
            .map(|entry| (entry.row, entry.col, entry.value))
    }
    /// Recomputes the factorization for new values with the analyzed pattern.
    ///
    /// `refactor` does not allocate. On error the previous factor is unchanged
    /// and stays usable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `matrix` does not have exactly the analyzed pattern.
    /// - `DimensionMismatch` if `matrix` has another shape, or `workspace` was
    ///   reserved for another analysis.
    /// - `ResourceLimit` if the factor needs more than `max_factor_entries`
    ///   entries, the submatrix being reduced needs more than
    ///   `max_active_entries`, or the workspace exceeds the byte limit.
    /// - `NumericalInstability` (`Overflow`) if a Householder vector or an update
    ///   overflows.
    /// - `ConvergenceFailure` if the iteration limit of `options` or `scope` is
    ///   reached before every column is reduced.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the
    ///   [execution context](scioxide_core#execution-contexts).
    pub fn refactor<'s, 'c>(
        &mut self,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        workspace: &mut SparseQrWorkspace<S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<(), SciOxideError> {
        let mut epoch = core::mem::take(&mut workspace.epoch);
        epoch.reject();
        // Pattern/finite checks run before any scope work is charged.
        let result = self.refactor_core(matrix, options, workspace, scope.into(), &mut epoch);
        workspace.epoch = epoch;
        result
    }
    // Transactional numeric epoch.
    fn refactor_core(
        &mut self,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        workspace: &mut SparseQrWorkspace<S>,
        scope: SolveScope<'_, '_>,
        epoch: &mut crate::factor_solve::Epoch,
    ) -> Result<(), SciOxideError> {
        const MATRIX: Argument = scioxide_core::argument!("matrix");
        let plan = &self.symbolic;
        workspace.require(&plan.workspace_requirements::<S>(())?)?;
        crate::helpers::shape((matrix.rows(), matrix.cols()), (plan.rows(), plan.cols()))?;

        workspace.seen.fill(false);
        for (row, col, value) in matrix.iter() {
            let start = plan.pattern.view().row_ptr()[row];
            let end = plan.pattern.view().row_ptr()[row + 1];
            let offset = plan.pattern.view().col_indices()[start..end]
                .binary_search(&col)
                .map_err(|_| invalid(MATRIX, "numeric pattern differs from analysis"))?;
            let pos = start + offset;
            if workspace.seen[pos] {
                return Err(invalid(MATRIX, "duplicate numeric structural entry"));
            }
            workspace.seen[pos] = true;
            workspace.candidate.source[pos] = value;
        }
        if workspace.seen.iter().any(|&seen| !seen) {
            return Err(invalid(MATRIX, "numeric pattern omits analyzed entries"));
        }
        workspace.kernel.clear(plan.limits.max_active_entries);
        for col in 0..plan.cols() {
            for pos in plan.col_ptr[col]..plan.col_ptr[col + 1] {
                workspace.kernel.insert(
                    plan.rows[pos],
                    col,
                    workspace.candidate.source[plan.positions[pos]],
                )?;
            }
        }
        workspace
            .candidate
            .order
            .copy_from_slice(plan.ordering.new_to_old());
        epoch.run(scope, options, |cx| {
            factor_kernel(
                &mut workspace.candidate,
                &mut workspace.kernel,
                (plan.rows(), plan.cols()),
                plan.limits,
                None,
                &mut || admit(cx),
            )
            .map(|_| ())
        })?;
        core::mem::swap(&mut self.data, &mut workspace.candidate);
        Ok(())
    }
}

// Fixed-order Householder QR. Without a cutoff no column norm is formed. With
// one, each pivot's fresh active-column norm is compared against it, and
// rejected tails do not consume a row.
pub(super) fn factor_kernel<S: NumericScalar>(
    data: &mut Data<S>,
    work: &mut Kernel<S>,
    (rows, cols): (usize, usize),
    limits: SparseQrLimits,
    cutoff: Option<S::Real>,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<usize, SciOxideError> {
    data.v.clear();
    data.r.clear();
    data.v_ptr.fill(0);
    data.beta.fill(S::Real::ZERO);
    for (logical, &physical) in data.order.iter().enumerate() {
        work.position[physical] = logical;
    }
    let mut k = 0;
    for logical in 0..cols {
        if k == rows.min(cols) {
            break;
        }
        tick()?;
        let pivot = data.order[logical];
        if let Some(cutoff) = cutoff
            && work.norm(pivot)? <= cutoff
        {
            work.remove_column(pivot);
            continue;
        }
        work.scatter(pivot);
        work.touch(k);
        let (diagonal, beta) = make_reflector(work, k)?;
        data.beta[k] = beta;
        data.v_ptr[k] = data.v.len();
        if beta != S::Real::ZERO {
            data.push_v(k, S::ONE, limits.max_factor_entries)?;
            work.touched.sort_unstable();
            for &row in &work.touched {
                if row > k && work.dense[row] != S::ZERO {
                    data.push_v(row, work.dense[row], limits.max_factor_entries)?;
                }
            }
        }
        data.v_ptr[k + 1] = data.v.len();
        data.push_r((k, pivot), diagonal, limits.max_factor_entries)?;
        work.columns.clear();
        work.affected(k, pivot, logical);
        for e in &data.v[data.v_ptr[k]..data.v_ptr[k + 1]] {
            work.affected(e.row, pivot, logical);
        }
        work.columns.sort_unstable();
        work.remove_column(pivot);
        for ordinal in 0..work.columns.len() {
            let col = work.columns[ordinal];
            work.col_marked[col] = false;
            #[cfg(test)]
            {
                work.visits += 1;
            }
            work.scatter(col);
            work.touch(k);
            if beta != S::Real::ZERO {
                let mut dot = S::ZERO;
                for e in &data.v[data.v_ptr[k]..data.v_ptr[k + 1]] {
                    dot = dot + e.value.conj() * work.dense[e.row];
                }
                let correction = S::from_real(beta) * dot;
                for e in &data.v[data.v_ptr[k]..data.v_ptr[k + 1]] {
                    work.touch(e.row);
                    work.dense[e.row] = work.dense[e.row] - e.value * correction;
                }
            }
            let r = finite_candidate(work.dense[k])?;
            data.push_r((k, col), r, limits.max_factor_entries)?;
            work.remove_column(col);
            for ordinal in 0..work.touched.len() {
                let row = work.touched[ordinal];
                if row > k {
                    work.insert(row, col, work.dense[row])?;
                }
            }
        }
        k += 1;
    }
    for ptr in &mut data.v_ptr[k + 1..] {
        *ptr = data.v.len();
    }
    data.index_r(&work.position);
    Ok(k)
}
fn make_reflector<S: NumericScalar>(
    work: &mut Kernel<S>,
    k: usize,
) -> Result<(S, S::Real), SciOxideError> {
    let scale = work
        .touched
        .iter()
        .map(|&row| work.dense[row].abs())
        .fold(S::Real::ZERO, |a, b| a.max(b));
    if scale == S::Real::ZERO {
        return Ok((S::ZERO, S::Real::ZERO));
    }
    let scale_scalar = S::from_real(scale);
    let mut tail = S::Real::ZERO;
    for &row in &work.touched {
        if row > k {
            tail += (work.dense[row] / scale_scalar).norm_sqr();
        }
    }
    if tail == S::Real::ZERO {
        return Ok((work.dense[k], S::Real::ZERO));
    }
    let first = work.dense[k] / scale_scalar;
    let first_abs = first.abs();
    let norm = (first.norm_sqr() + tail).sqrt();
    let leading = if first_abs == S::Real::ZERO {
        S::from_real(-norm)
    } else {
        first + first / S::from_real(first_abs) * S::from_real(norm)
    };
    let mut v_norm = S::Real::ONE;
    for &row in &work.touched {
        if row > k {
            work.dense[row] = finite_candidate((work.dense[row] / scale_scalar) / leading)?;
            v_norm += work.dense[row].norm_sqr();
        }
    }
    let beta = S::Real::TWO / v_norm;
    let diag = if first_abs == S::Real::ZERO {
        S::from_real(scale * norm)
    } else {
        -(first / S::from_real(first_abs)) * S::from_real(scale * norm)
    };
    Ok((finite_candidate(diag)?, finite_candidate(beta)?))
}
// Applies Q or Qᴴ in place.
pub(super) fn apply_q<S: NumericScalar>(
    data: &Data<S>,
    vector: &mut [S],
    adjoint: bool,
) -> Result<(), SciOxideError> {
    let kmax = data.beta.len();
    for ordinal in 0..kmax {
        let k = if adjoint { ordinal } else { kmax - 1 - ordinal };
        let beta = data.beta[k];
        if beta == S::Real::ZERO {
            continue;
        }
        let mut dot = S::ZERO;
        for entry in &data.v[data.v_ptr[k]..data.v_ptr[k + 1]] {
            dot = dot + entry.value.conj() * vector[entry.row];
        }
        let correction = S::from_real(beta) * dot;
        for entry in &data.v[data.v_ptr[k]..data.v_ptr[k + 1]] {
            vector[entry.row] = finite_candidate(vector[entry.row] - entry.value * correction)?;
        }
    }
    Ok(())
}

// Applies Q, Qᵀ or Qᴴ selected by `system`.
pub(super) fn apply_q_operation<S: NumericScalar>(
    data: &Data<S>,
    vector: &mut [S],
    system: SparseOperation,
) -> Result<(), SciOxideError> {
    match system {
        SparseOperation::Normal => apply_q(data, vector, false),
        SparseOperation::Adjoint => apply_q(data, vector, true),
        SparseOperation::Transpose => {
            for value in vector.iter_mut() {
                *value = value.conj();
            }
            apply_q(data, vector, true)?;
            for value in vector.iter_mut() {
                *value = value.conj();
            }
            Ok(())
        }
    }
}

pub(super) fn data_bytes<S: NumericScalar>(
    m: usize,
    n: usize,
    z: usize,
    f: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("qr_factor_bytes");
    let k = m.min(n);
    let index = size_of::<usize>();
    add(
        add(
            mul(
                f,
                add(
                    add(size_of::<Entry<S>>(), size_of::<REntry<S>>(), q)?,
                    index,
                    q,
                )?,
                q,
            )?,
            mul(
                add(mul(2, add(k, 1, q)?, q)?, add(mul(2, n, q)?, 1, q)?, q)?,
                index,
                q,
            )?,
            q,
        )?,
        add(
            mul(add(k, z, q)?, size_of::<S>(), q)?,
            mul(k, size_of::<S::Real>(), q)?,
            q,
        )?,
        q,
    )
}
pub(super) fn kernel_bytes<S: NumericScalar>(
    m: usize,
    n: usize,
    a: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("qr_kernel_bytes");
    let dims = add(m, n, q)?;
    add(
        mul(a, size_of::<Slot<S>>(), q)?,
        add(
            mul(
                dims,
                add(mul(3, size_of::<usize>(), q)?, size_of::<bool>(), q)?,
                q,
            )?,
            mul(m, size_of::<S>(), q)?,
            q,
        )?,
        q,
    )
}

#[cfg(test)]
impl<S: NumericScalar> SparseQrWorkspace<S> {
    pub(super) fn update_visits(&self) -> usize {
        self.kernel.visits
    }
}

fn vector_bytes<T>(values: &Vec<T>) -> Result<usize, SciOxideError> {
    mul(
        values.capacity(),
        size_of::<T>(),
        scioxide_core::quantity!("qr_retained_bytes"),
    )
}
fn sum_bytes<const N: usize>(
    values: [Result<usize, SciOxideError>; N],
) -> Result<usize, SciOxideError> {
    values.into_iter().try_fold(0, |sum, value| {
        add(sum, value?, scioxide_core::quantity!("qr_retained_bytes"))
    })
}
impl<S: NumericScalar> Data<S> {
    pub(super) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        sum_bytes([
            vector_bytes(&self.v),
            vector_bytes(&self.v_ptr),
            vector_bytes(&self.beta),
            vector_bytes(&self.r),
            vector_bytes(&self.r_ptr),
            vector_bytes(&self.r_col_ptr),
            vector_bytes(&self.r_col_index),
            vector_bytes(&self.diagonal),
            vector_bytes(&self.order),
            vector_bytes(&self.source),
        ])
    }
}
impl<S: NumericScalar> Kernel<S> {
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        sum_bytes([
            vector_bytes(&self.slots),
            vector_bytes(&self.row_heads),
            vector_bytes(&self.col_heads),
            vector_bytes(&self.position),
            vector_bytes(&self.dense),
            vector_bytes(&self.touched),
            vector_bytes(&self.marked),
            vector_bytes(&self.columns),
            vector_bytes(&self.col_marked),
        ])
    }
}
