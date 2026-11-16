use super::*;
use crate::SparseSolveOptions;
use crate::SparseView;
use crate::factor_arithmetic::Arithmetic;
use crate::helpers::admit;

#[derive(Debug, Clone, Copy)]
pub(super) struct Entry<S> {
    pub row: usize,
    pub value: S,
}

#[derive(Debug)]
pub(super) struct Factors<S> {
    pub entries: Vec<Entry<S>>,
    pub ptr: Vec<usize>,
    pub lower: Vec<usize>,
    pub p: Vec<usize>,
    pub inverse: Vec<usize>,
}
impl<S: NumericScalar> Factors<S> {
    // Reserves factor storage.
    fn reserve(plan: &SparseLuSymbolic) -> Result<Self, SciOxideError> {
        Self::reserve_shape(plan.dimension(), plan.limits.max_factor_entries)
    }
    fn reserve_shape(n: usize, entries: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            entries: try_filled(
                entries,
                Entry {
                    row: 0,
                    value: S::ZERO,
                },
                scioxide_core::quantity!("factor_entries"),
            )?,
            ptr: try_filled(
                add(n, 1, scioxide_core::quantity!("factor_column_pointers"))?,
                0,
                scioxide_core::quantity!("factor_column_pointers"),
            )?,
            lower: try_filled(n, 0, scioxide_core::quantity!("factor_lower_offsets"))?,
            p: try_filled(n, 0, scioxide_core::quantity!("row_permutation"))?,
            inverse: try_filled(
                n,
                usize::MAX,
                scioxide_core::quantity!("inverse_row_permutation"),
            )?,
        })
    }
    fn projected_bytes(&self, n: usize, entries: usize) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("workspace_bytes");
        let mut indices = self.ptr.capacity().max(add(n, 1, q)?);
        for capacity in [
            self.lower.capacity(),
            self.p.capacity(),
            self.inverse.capacity(),
        ] {
            indices = add(indices, capacity.max(n), q)?;
        }
        add(
            mul(
                self.entries.capacity().max(entries),
                size_of::<Entry<S>>(),
                q,
            )?,
            mul(indices, size_of::<usize>(), q)?,
            q,
        )
    }
    fn resize(&mut self, n: usize, entries: usize) -> Result<(), SciOxideError> {
        crate::helpers::resize(
            &mut self.entries,
            entries,
            Entry {
                row: 0,
                value: S::ZERO,
            },
        )?;
        crate::helpers::resize(&mut self.ptr, n + 1, 0)?;
        crate::helpers::resize(&mut self.lower, n, 0)?;
        crate::helpers::resize(&mut self.p, n, 0)?;
        crate::helpers::resize(&mut self.inverse, n, usize::MAX)
    }
    // Appends one candidate entry.
    fn push(&mut self, end: &mut usize, row: usize, value: S) -> Result<(), SciOxideError> {
        let capacity = self.entries.len();
        let slot = self.entries.get_mut(*end).ok_or_else(|| {
            SciOxideError::resource_limit(
                scioxide_core::quantity!("factor_entries"),
                end.saturating_add(1),
                capacity,
            )
        })?;
        *slot = Entry { row, value };
        *end += 1;
        Ok(())
    }
}

/// Holds the storage for recomputing a sparse LU factorization without
/// allocation.
///
/// The workspace holds a second factor. [`SparseLu::refactor`] computes into it
/// and replaces the current factor only on success. Create a workspace with
/// [`SparseLuSymbolic::workspace`], or with [`new`](Self::new) and
/// [`try_reserve`](Self::try_reserve). It fits any analysis with the same order,
/// the same number of pattern entries and the same `max_factor_entries`. After
/// an error its contents are unspecified, but it stays usable.
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext};
/// use scioxide_sparse::{
///     CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseLuWorkspace, SparseOrdering,
///     SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 4.])?;
/// let limits = SparseLuLimits::new(2, 1 << 16);
/// let symbolic =
///     SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits)?;
/// let requirements = symbolic.workspace_requirements::<f64>(())?;
/// let mut workspace = SparseLuWorkspace::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let mut lu = SparseLu::factor(symbolic, a.view())?;
/// let b = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 8.])?;
/// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// lu.refactor(b.view(), options, &mut workspace, scope)?;
/// assert!(workspace.last_attempt().is_some());
/// assert_eq!(lu.solve(&[4., 8.])?, vec![1., 1.]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseLuWorkspace<S: NumericScalar> {
    epoch: crate::factor_solve::Epoch,
    initial: Option<SparseLuSymbolic>,
    spare: Option<Factors<S>>,
    candidate: Factors<S>,
    x: Vec<S>,
    source: Vec<S>,
    seen: Vec<bool>,
    marked: Vec<bool>,
    reach: Vec<usize>,
    stack: Vec<usize>,
    cursor: Vec<usize>,
}
impl<S: NumericScalar> SparseLuWorkspace<S> {
    /// Creates a workspace that also holds storage for the first factor, and
    /// takes ownership of the analysis.
    ///
    /// Call [`factor_scoped`](Self::factor_scoped) on the result to compute the
    /// first factor without further allocation. If that factorization fails, the workspace
    /// keeps the analysis and its storage, so you can retry with other values.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_sparse::{
    ///     CsrMatrix, SparseLuLimits, SparseLuSymbolic, SparseLuWorkspace, SparseOrdering,
    ///     SparseSolveOptions,
    /// };
    /// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
    /// let limits = SparseLuLimits::new(1, 4096);
    /// let symbolic =
    ///     SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits)?;
    /// let mut workspace = SparseLuWorkspace::prepare(symbolic)?;
    /// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
    /// let factor = workspace.factor_scoped(a.view(), options, scope)?;
    /// assert!(workspace.initial_symbolic().is_none());
    /// assert_eq!(factor.solve(&[4.])?, vec![2.]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the analysis, two factors and the scratch exceed the
    ///   byte limit of the analysis.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn prepare(symbolic: SparseLuSymbolic) -> Result<Self, SciOxideError> {
        let mut workspace = symbolic.workspace(())?;
        workspace.spare = Some(Factors::reserve(&symbolic)?);
        workspace.initial = Some(symbolic);
        Ok(workspace)
    }
    /// Reuses this workspace for a new sparsity pattern in natural column
    /// order.
    ///
    /// If `factor` is given, the workspace takes back its analysis and storage,
    /// and its values are discarded. The new analysis uses the identity column
    /// ordering and the given `limits`; the byte limit covers all storage the
    /// workspace already holds. When that storage is large enough, neither this
    /// call nor the next [`factor_scoped`](Self::factor_scoped) allocates. After an error, call
    /// this method again before factoring.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_sparse::{
    ///     CsrMatrix, SparseLuLimits, SparseLuSymbolic, SparseLuWorkspace, SparseOrdering,
    ///     SparseSolveOptions,
    /// };
    /// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64; 2])?;
    /// let limits = SparseLuLimits::new(8, 65536);
    /// let p = SparseLuSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, limits)?;
    /// let mut w = SparseLuWorkspace::prepare(p)?;
    /// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
    /// let f = w.factor_scoped(a.view(), options, scope)?;
    /// let b = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![4.0_f64])?;
    /// w.reprepare_natural(Some(f), b.structural_pattern(), limits)?;
    /// let f = w.factor_scoped(b.view(), options, scope)?;
    /// assert_eq!(f.solve(&[8.0])?, vec![2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `factor` is `None` and the workspace holds no
    ///   analysis.
    /// - `DimensionMismatch` if `pattern` is not square.
    /// - `ResourceLimit` if the new analysis and the storage exceed
    ///   `limits.max_workspace_bytes()`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn reprepare_natural(
        &mut self,
        factor: Option<SparseLu<S>>,
        pattern: StructuralPatternView<'_>,
        limits: SparseLuLimits,
    ) -> Result<(), SciOxideError> {
        if let Some(factor) = factor {
            self.initial = Some(factor.symbolic);
            self.spare = Some(factor.data);
        }
        let symbolic = self.initial.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "initial analysis or a prior factor is required",
            )
        })?;
        let r = SparseLuRequirements::for_shape::<S>(
            pattern.rows(),
            pattern.col_indices().len(),
            symbolic_bytes(
                pattern.rows().max(symbolic.capacity_n),
                pattern.col_indices().len().max(symbolic.capacity_z),
            )?,
            limits,
        )?;
        self.admit_growth(&r)?;
        let mut symbolic = self.initial.take().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "missing initial analysis",
            )
        })?;
        // If structural preparation fails, no partial analysis can be factored.
        symbolic.limits = limits;
        symbolic.reanalyze_natural(pattern)?;
        self.initial = Some(symbolic);
        self.try_reserve(&r)?;
        self.epoch.reject();
        Ok(())
    }
    /// Returns the analysis that [`factor_scoped`](Self::factor_scoped) will
    /// use, or `None` once it has produced a factor.
    pub fn initial_symbolic(&self) -> Option<&SparseLuSymbolic> {
        self.initial.as_ref()
    }
    /// Computes the first factor in the storage reserved by
    /// [`prepare`](Self::prepare), without allocating.
    ///
    /// The returned factor owns the analysis. Compute later factorizations with
    /// [`SparseLu::refactor`] and this workspace. On error the workspace keeps
    /// the analysis and its storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the workspace holds no analysis, because it was not
    ///   created by [`prepare`](Self::prepare) or has already produced a factor.
    /// - The errors of [`SparseLu::refactor`].
    pub fn factor_scoped<'scope, 'capability>(
        &mut self,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<SparseLu<S>, SciOxideError> {
        self.epoch.reject();
        if self.spare.is_none() {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace has no initial factor reservation",
            ));
        }
        let symbolic = self.initial.take().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "workspace has no initial analysis",
            )
        })?;
        if let Err(error) = load(&symbolic, matrix, self) {
            self.initial = Some(symbolic);
            return Err(error);
        }
        let mut epoch = core::mem::take(&mut self.epoch);
        let result = epoch.run(scope, options, |context| {
            factor_epoch(&symbolic, self, Arithmetic(options.accuracy()), context)
        });
        self.epoch = epoch;
        if let Err(error) = result {
            self.initial = Some(symbolic);
            return Err(error);
        }
        let mut data = self.spare.take().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "workspace has no initial factor reservation",
            )
        })?;
        core::mem::swap(&mut data, &mut self.candidate);
        Ok(SparseLu { symbolic, data })
    }
}

/// Holds the sparse LU factorization with partial pivoting of a square matrix.
///
/// The factors satisfy
///
/// ```text
/// A[p[i], q[j]] = (L U)[i, j],
/// ```
///
/// where `L` is unit lower triangular, `U` is upper triangular, `q` is the
/// column permutation of the [analysis](SparseLuSymbolic) and `p` is the row
/// permutation chosen by pivoting. Both permutations map new to old indices.
///
/// The factorization is left-looking. Column `k` of `L` and `U` comes from a
/// sparse triangular solve with the first `k` columns of `L`, whose nonzero
/// pattern is found by a depth-first search (Gilbert and Peierls, 1988). The
/// pivot is the entry of largest magnitude among the rows not yet pivoted, with
/// ties broken by the smallest row index. Factorization is sequential, and each
/// column counts as one iteration toward the iteration limit.
///
/// Solve with `solve`, `solve_into` or `solve_block_into`, for `A`, `Aᵀ` or
/// `Aᴴ` as selected by [`SparseOperation`](crate::SparseOperation). The factor
/// also implements [`LinearOperator`](scioxide_core::LinearOperator) as the
/// action of `A⁻¹`. See [Direct solvers](crate#direct-solvers) for the
/// analyze–factor–solve sequence.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_sparse::{
///     CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic, SparseOrdering, SparseSolveOptions,
/// };
/// // A = [[2, 1], [0, 4]]
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0_f64, 1., 4.])?;
/// let symbolic = SparseLuSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Natural,
///     SparseLuLimits::new(4, 1 << 16),
/// )?;
/// let options = SparseSolveOptions::new();
/// let mut lu =
///     SparseLu::factor_scoped(symbolic, a.view(), options, ExecutionContext::sequential())?;
/// assert_eq!(lu.solve(&[4., 8.])?, vec![1., 2.]);
/// assert_eq!(lu.solve_transpose(&[2., 9.])?, vec![1., 2.]);
/// // Same pattern with an exact zero pivot: the previous factor stays usable.
/// let singular = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0_f64, 1., 0.])?;
/// let mut workspace = lu.symbolic().workspace::<f64>(())?;
/// let error = lu
///     .refactor(
///         singular.view(),
///         options,
///         &mut workspace,
///         ExecutionContext::sequential(),
///     )
///     .unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::Singular);
/// assert_eq!(lu.solve(&[4., 8.])?, vec![1., 2.]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time
///   proportional to arithmetic operations", *SIAM J. Sci. Stat. Comput.* 9
///   (1988), 862–874.
/// - T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006,
///   chapter 6.
#[derive(Debug)]
pub struct SparseLu<S: NumericScalar> {
    pub(super) symbolic: SparseLuSymbolic,
    pub(super) data: Factors<S>,
}
impl<S: NumericScalar> SparseLu<S> {
    /// Computes the LU factorization of the square matrix `A = matrix`, with a
    /// fill-reducing column ordering and storage chosen from the matrix.
    ///
    /// The analysis uses [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing)
    /// (approximate minimum degree) and no byte limit. The factor storage starts
    /// at `nnz(A) + n` entries and doubles after each `ResourceLimit`, up to the
    /// `n²` entries that no factor exceeds, so at most `⌈log₂ n⌉ + 1`
    /// factorizations run; each repeats the columns of the failed one. The
    /// result equals [`analyze`](SparseLuSymbolic::analyze) with sufficient
    /// limits followed by [`factor`](Self::factor), which fixes the limits
    /// instead. Rows are pivoted by partial pivoting as described on the type.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, SparseLu};
    ///
    /// // A = [[2, 1], [0, 4]]
    /// let a = CsrMatrix::from_triplets(2, 2, &[0, 0, 1], &[0, 1, 1], &[2.0_f64, 1.0, 4.0])?;
    /// let lu = SparseLu::new(&a)?;
    /// let (x, y) = (lu.solve(&[4.0, 8.0])?, lu.solve_transpose(&[2.0, 9.0])?);
    /// assert!(x.iter().zip([1.0, 2.0]).all(|(v, e)| (v - e).abs() < 1e-15));
    /// assert!(y.iter().zip([1.0, 2.0]).all(|(v, e)| (v - e).abs() < 1e-15));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `Singular` if a column has no nonzero pivot candidate.
    /// - `NumericalInstability` if an entry overflows or a nonzero multiplier
    ///   underflows to zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let n = matrix.rows();
        let symbolic = SparseLuSymbolic::analyze(
            matrix.structural_pattern(),
            crate::SparseOrdering::FillReducing,
            SparseLuLimits::new(usize::MAX, usize::MAX),
        )?;
        crate::capacity::with_capacity(
            matrix.nnz().saturating_add(n),
            n.saturating_mul(n),
            |entries| {
                let mut attempt = symbolic.try_clone()?;
                attempt.limits = SparseLuLimits::new(entries, usize::MAX);
                Self::factor(attempt, matrix.view())
            },
        )
    }
    /// Computes the LU factorization of `matrix`, whose pattern `symbolic`
    /// analyzed, with the default [`SparseSolveOptions`] in a fresh scope.
    ///
    /// This is [`factor_scoped`](Self::factor_scoped) with
    /// [`SparseSolveOptions::new`] and
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`factor_scoped`](Self::factor_scoped).
    pub fn factor(
        symbolic: SparseLuSymbolic,
        matrix: SparseView<'_, S>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_scoped(
            symbolic,
            matrix,
            SparseSolveOptions::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
    /// Computes the LU factorization of `matrix`, whose pattern `symbolic`
    /// analyzed, inside the caller's scope.
    ///
    /// This allocates the factor and a temporary workspace. To factor a sequence
    /// of matrices without allocation, use [`refactor`](Self::refactor).
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the factor and workspace exceed the byte limit, and
    ///   `SizeOverflow` or `AllocationFailure` if they cannot be allocated.
    /// - The errors of [`refactor`](Self::refactor).
    pub fn factor_scoped<'scope, 'capability>(
        symbolic: SparseLuSymbolic,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        SparseLuWorkspace::prepare(symbolic)?.factor_scoped(matrix, options, scope)
    }

    /// Returns the symbolic analysis.
    pub fn symbolic(&self) -> &SparseLuSymbolic {
        &self.symbolic
    }
    /// Returns the number of stored entries of `L` and `U`, without the unit
    /// diagonal of `L`.
    pub fn factor_entries(&self) -> usize {
        self.data.ptr[self.symbolic.dimension()]
    }
    /// Returns the row permutation `p` chosen by pivoting, with `p[new] = old`.
    pub fn row_permutation(&self) -> &[usize] {
        &self.data.p
    }
    /// Returns the entries `(i, j, L[i, j])` of `L` below the diagonal, column by
    /// column.
    pub fn lower_entries(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        (0..self.symbolic.dimension()).flat_map(move |col| {
            self.data.entries[self.data.lower[col]..self.data.ptr[col + 1]]
                .iter()
                .map(move |e| (self.data.inverse[e.row], col, e.value))
        })
    }
    /// Returns the entries `(i, j, U[i, j])` of `U`, including the diagonal,
    /// column by column.
    pub fn upper_entries(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        (0..self.symbolic.dimension()).flat_map(move |col| {
            self.data.entries[self.data.ptr[col]..self.data.lower[col]]
                .iter()
                .map(move |e| (e.row, col, e.value))
        })
    }
    /// Recomputes the factorization for new values with the analyzed pattern.
    ///
    /// The row pivots, and with them the fill, can differ from the previous
    /// factorization. `refactor` does not allocate. On error the previous factor
    /// is unchanged and stays usable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `matrix` does not have exactly the analyzed pattern.
    /// - `DimensionMismatch` if `matrix` has another shape, or `workspace` was
    ///   reserved for another order, pattern size or entry limit.
    /// - `ResourceLimit` if the factor needs more than `max_factor_entries`
    ///   entries, or the workspace exceeds the byte limit.
    /// - `Singular` if a column has no nonzero pivot candidate.
    /// - `NumericalInstability` if an entry overflows or a nonzero multiplier
    ///   underflows to zero.
    /// - `ConvergenceFailure` if the iteration limit of `options` or `scope` is
    ///   reached before every column is factored.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the
    ///   [execution context](scioxide_core#execution-contexts).
    pub fn refactor<'scope, 'capability>(
        &mut self,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        workspace: &mut SparseLuWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.epoch.reject();
        self.prepare(matrix, workspace)?;
        let mut epoch = core::mem::take(&mut workspace.epoch);
        let result = epoch.run(scope, options, |context| {
            factor_epoch(
                &self.symbolic,
                workspace,
                Arithmetic(options.accuracy()),
                context,
            )
        });
        workspace.epoch = epoch;
        result?;
        core::mem::swap(&mut self.data, &mut workspace.candidate);
        Ok(())
    }

    // Scatters `matrix` into a reserved candidate.
    fn prepare(
        &self,
        matrix: SparseView<'_, S>,
        w: &mut SparseLuWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        // A separate initial reservation is not retained alongside an existing factor.
        w.initial = None;
        w.spare = None;
        load(&self.symbolic, matrix, w)
    }
}
fn load<S: NumericScalar>(
    plan: &SparseLuSymbolic,
    matrix: SparseView<'_, S>,
    w: &mut SparseLuWorkspace<S>,
) -> Result<(), SciOxideError> {
    const MATRIX: Argument = scioxide_core::argument!("matrix");
    let n = plan.dimension();
    let pattern = plan.pattern();
    crate::helpers::shape((matrix.rows(), matrix.cols()), (n, n))?;
    // The same byte admission and reserved shape that `check` reports as sufficient.
    let requirements = plan.workspace_requirements::<S>(())?;
    let _ = w.check(&requirements)?;
    w.fits(&requirements)?;

    if let Some(values) = matrix.csr_values_on(pattern) {
        w.source.copy_from_slice(values);
    } else {
        w.seen.fill(false);
        for (row, col, value) in matrix.iter() {
            let start = pattern.row_ptr()[row];
            let offset = pattern.col_indices()[start..pattern.row_ptr()[row + 1]]
                .binary_search(&col)
                .map_err(|_| invalid(MATRIX, "numeric pattern differs from analysis"))?;
            let pos = start + offset;
            if w.seen[pos] {
                return Err(invalid(MATRIX, "duplicate numeric structural entry"));
            }
            w.seen[pos] = true;
            w.source[pos] = value;
        }
        if w.seen.iter().any(|&seen| !seen) {
            return Err(invalid(MATRIX, "numeric pattern omits analyzed entries"));
        }
    }
    w.candidate.inverse.fill(usize::MAX);
    w.candidate.ptr.fill(0);
    w.marked.fill(false);
    w.x.fill(S::ZERO);
    Ok(())
}

// No provider receives this context during numerical factorization. When the whole
// column count is admitted, retain the exact attempted prefix locally and publish
// it once, including on a numerical failure or execution stop. Restricted budgets
// and counter headroom retain per-column admission and its original stop boundary.
fn factor_epoch<S: NumericScalar>(
    plan: &SparseLuSymbolic,
    w: &mut SparseLuWorkspace<S>,
    arithmetic: Arithmetic,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let batched = context
        .can_record_iterations(plan.dimension())
        .is_ok_and(|a| a.is_granted());
    let mut attempts = 0;
    let result = factor_columns(plan, w, arithmetic, || {
        if batched {
            context.execution().check()?;
            attempts += 1;
            Ok(())
        } else {
            admit(context)
        }
    });
    if batched {
        context.record_iterations(attempts)?.require()?;
    }
    result
}

// Left-looking factorization into the candidate.
fn factor_columns<S: NumericScalar>(
    plan: &SparseLuSymbolic,
    w: &mut SparseLuWorkspace<S>,
    arithmetic: Arithmetic,
    mut tick: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let n = plan.dimension();
    let mut end = 0;
    for k in 0..n {
        tick()?;
        let col = plan.permutation.new_to_old()[k];
        let range = plan.column_ptr[col]..plan.column_ptr[col + 1];
        let mut top = n;
        // Iterative DFS computes the topological reach in the existing L columns.
        for &root in &plan.rows[range.clone()] {
            if w.marked[root] {
                continue;
            }
            let mut depth = 1;
            w.stack[0] = root;
            while depth != 0 {
                let row = w.stack[depth - 1];
                let pivot = w.candidate.inverse[row];
                if !w.marked[row] {
                    w.marked[row] = true;
                    w.cursor[depth - 1] = if pivot == usize::MAX {
                        0
                    } else {
                        w.candidate.lower[pivot]
                    };
                }
                let stop = if pivot == usize::MAX {
                    0
                } else {
                    w.candidate.ptr[pivot + 1]
                };
                let mut child = None;
                while w.cursor[depth - 1] < stop {
                    let next = w.candidate.entries[w.cursor[depth - 1]].row;
                    w.cursor[depth - 1] += 1;
                    if !w.marked[next] {
                        child = Some(next);
                        break;
                    }
                }
                if let Some(next) = child {
                    w.stack[depth] = next;
                    depth += 1;
                } else {
                    depth -= 1;
                    top -= 1;
                    w.reach[top] = row;
                }
            }
        }
        for &row in &w.reach[top..] {
            w.x[row] = S::ZERO;
        }
        for pos in range {
            w.x[plan.rows[pos]] = w.source[plan.positions[pos]];
        }
        for &row in &w.reach[top..] {
            let pivot = w.candidate.inverse[row];
            if pivot != usize::MAX {
                for e in &w.candidate.entries[w.candidate.lower[pivot]..w.candidate.ptr[pivot + 1]]
                {
                    w.x[e.row] = arithmetic.subtract_product(w.x[e.row], e.value, w.x[row])?;
                }
            }
        }
        let mut choice = None;
        let mut largest = S::ZERO;
        w.candidate.ptr[k] = end;
        for &row in &w.reach[top..] {
            let pivot = w.candidate.inverse[row];
            let value = w.x[row];
            if pivot != usize::MAX {
                if value != S::ZERO {
                    w.candidate.push(&mut end, pivot, value)?;
                }
            } else if value != S::ZERO {
                let comparison = arithmetic.compare_magnitude(value, largest)?;
                if choice.is_none()
                    || comparison.is_gt()
                    || (comparison.is_eq() && choice.is_some_and(|old| row < old))
                {
                    largest = value;
                    choice = Some(row);
                }
            }
        }
        let row = choice.ok_or_else(|| SciOxideError::singular(Some(k)))?;
        let diagonal = w.x[row];
        w.candidate.push(&mut end, k, diagonal)?;
        w.candidate.lower[k] = end;
        w.candidate.p[k] = row;
        w.candidate.inverse[row] = k;
        for &i in &w.reach[top..] {
            if w.candidate.inverse[i] == usize::MAX && w.x[i] != S::ZERO {
                let value =
                    arithmetic.divide(w.x[i], diagonal, crate::helpers::Projection::Gradual)?;
                if value == S::ZERO {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Underflow,
                        "nonzero LU multiplier is unrepresentable",
                    ));
                }
                w.candidate.push(&mut end, i, value)?;
            }
            w.marked[i] = false;
        }
        w.candidate.ptr[k + 1] = end;
    }
    Ok(())
}

impl<S: NumericScalar> Factors<S> {
    fn retained_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<Entry<S>>()
            + (self.ptr.capacity()
                + self.lower.capacity()
                + self.p.capacity()
                + self.inverse.capacity())
                * size_of::<usize>()
    }
}

impl<S: NumericScalar> Default for SparseLuWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> SparseLuWorkspace<S> {
    /// Creates an empty workspace. Reserve it with
    /// [`try_reserve`](Self::try_reserve) before use.
    pub fn new() -> Self {
        Self {
            epoch: crate::factor_solve::Epoch::new(),
            initial: None,
            spare: None,
            candidate: Factors {
                entries: Vec::new(),
                ptr: Vec::new(),
                lower: Vec::new(),
                p: Vec::new(),
                inverse: Vec::new(),
            },
            x: Vec::new(),
            source: Vec::new(),
            seen: Vec::new(),
            marked: Vec::new(),
            reach: Vec::new(),
            stack: Vec::new(),
            cursor: Vec::new(),
        }
    }
    /// Returns the bytes of scratch and second factor that this workspace
    /// holds, including unused capacity. The analysis and first factor held by
    /// a [prepared](Self::prepare) workspace are not included.
    pub fn retained_bytes(&self) -> usize {
        self.candidate.retained_bytes()
            + (self.x.capacity() + self.source.capacity()) * size_of::<S>()
            + self.seen.capacity()
            + self.marked.capacity()
            + (self.reach.capacity() + self.stack.capacity() + self.cursor.capacity())
                * size_of::<usize>()
    }
    /// Returns the record of the most recent factorization that used this
    /// workspace.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.epoch.record()
    }
    /// Checks whether this workspace fits the requirements `r`.
    ///
    /// Returns `Sufficient` when the order, the number of pattern entries and
    /// the entry limit match `r` exactly. Analyses that differ only in their
    /// byte limit share a workspace. A factorization rejects a workspace that
    /// needs growth with `DimensionMismatch`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the bytes held by the workspace, plus the analysis
    ///   and one factor, exceed the byte limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(
        &self,
        r: &SparseLuRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let symbolic_bytes = r.symbolic_bytes.max(match &self.initial {
            Some(plan) => plan.workspace_requirements::<S>(())?.symbolic_bytes,
            None => 0,
        });
        let factor_bytes = r
            .factor_bytes
            .max(self.spare.as_ref().map_or(0, Factors::retained_bytes));
        let bytes = add(
            add(
                symbolic_bytes,
                factor_bytes,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            self.retained_bytes(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        Ok(if self.fits(r).is_ok() {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    // Exact reserved shape and entry capacity of `r`; the byte ceiling is
    // admitted separately. A mismatch reports `DimensionMismatch` with the
    // requested extent first.
    fn fits(&self, r: &SparseLuRequirements) -> Result<(), SciOxideError> {
        crate::helpers::exact(self.x.len(), r.n)?;
        crate::helpers::exact(self.candidate.ptr.len(), r.n + 1)?;
        crate::helpers::exact(self.source.len(), r.z)?;
        crate::helpers::exact(self.candidate.entries.len(), r.limits.max_factor_entries)
    }
    fn admit_growth(&self, r: &SparseLuRequirements) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("workspace_bytes");
        let mut bytes = r.symbolic_bytes;
        bytes = add(
            bytes,
            self.spare.as_ref().map_or(Ok(r.factor_bytes), |s| {
                s.projected_bytes(r.n, r.limits.max_factor_entries)
            })?,
            q,
        )?;
        bytes = add(
            bytes,
            self.candidate
                .projected_bytes(r.n, r.limits.max_factor_entries)?,
            q,
        )?;
        for (capacity, requested, size) in [
            (self.x.capacity(), r.n, size_of::<S>()),
            (self.source.capacity(), r.z, size_of::<S>()),
            (self.seen.capacity(), r.z, size_of::<bool>()),
            (self.marked.capacity(), r.n, size_of::<bool>()),
            (self.reach.capacity(), r.n, size_of::<usize>()),
            (self.stack.capacity(), r.n, size_of::<usize>()),
            (self.cursor.capacity(), r.n, size_of::<usize>()),
        ] {
            bytes = add(bytes, mul(capacity.max(requested), size, q)?, q)?;
        }
        if bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                q,
                bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        Ok(())
    }
    /// Resizes this workspace to fit `r`, keeping its existing capacity.
    ///
    /// After an error the workspace may have grown without fitting `r`; call
    /// `try_reserve` again before use.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if `r.total_bytes()`, or the storage after growth,
    ///   exceeds the byte limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn try_reserve(&mut self, r: &SparseLuRequirements) -> Result<(), SciOxideError> {
        if r.total_bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                r.total_bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        self.admit_growth(r)?;
        if self.check(r)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        // A failed reservation must not expose an initial epoch with partial geometry.
        let initial = self.initial.take();
        // Admission uses retained capacity; resizing changes active lengths only.
        self.candidate.resize(r.n, r.limits.max_factor_entries)?;
        if let Some(spare) = &mut self.spare {
            spare.resize(r.n, r.limits.max_factor_entries)?;
        }
        crate::helpers::resize(&mut self.x, r.n, S::ZERO)?;
        crate::helpers::resize(&mut self.source, r.z, S::ZERO)?;
        crate::helpers::resize(&mut self.seen, r.z, false)?;
        crate::helpers::resize(&mut self.marked, r.n, false)?;
        crate::helpers::resize(&mut self.reach, r.n, 0)?;
        crate::helpers::resize(&mut self.stack, r.n, 0)?;
        crate::helpers::resize(&mut self.cursor, r.n, 0)?;
        self.initial = initial;
        Ok(())
    }
}
