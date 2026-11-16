use super::accumulator::{Incidence, Row};
use super::block::{norm, update};
use super::*;
use crate::SparseSolveOptions;
use crate::SparseView;
use crate::factor_arithmetic::Arithmetic;
use crate::helpers::admit;
use scioxide_core::Scaled;
use scioxide_core::{Capacity, ExecutionContext};

#[derive(Debug)]
pub(super) struct Factors<S: NumericScalar> {
    pub ptr: Vec<usize>,
    pub col: Vec<usize>,
    pub level: Vec<usize>,
    pub values: Vec<S>,
    pub diagonal: Vec<S>,
    pub pivots: Vec<usize>,
}
impl<S: NumericScalar> Factors<S> {
    // Reserves factor storage.
    fn new(p: Shape) -> Result<Self, SciOxideError> {
        const POINTERS: Quantity = scioxide_core::quantity!("factor_row_pointers");
        const VALUES: Quantity = scioxide_core::quantity!("factor_values");
        const DIAGONAL: Quantity = scioxide_core::quantity!("factor_diagonal");
        const PIVOTS: Quantity = scioxide_core::quantity!("diagonal_pivots");
        let n = p.rows;
        let b = p.block;
        let b2 = mul(b, b, scioxide_core::quantity!("block_area"))?;
        let f = p.blocks;
        Ok(Self {
            ptr: try_filled(add(n, 1, POINTERS)?, 0, POINTERS)?,
            col: try_filled(f, 0, scioxide_core::quantity!("factor_columns"))?,
            level: try_filled(f, 0, scioxide_core::quantity!("factor_levels"))?,
            values: try_filled(mul(f, b2, VALUES)?, S::ZERO, VALUES)?,
            diagonal: try_filled(mul(n, b2, DIAGONAL)?, S::ZERO, DIAGONAL)?,
            pivots: try_filled(mul(n, b, PIVOTS)?, 0, PIVOTS)?,
        })
    }
    const fn empty() -> Self {
        Self {
            ptr: Vec::new(),
            col: Vec::new(),
            level: Vec::new(),
            values: Vec::new(),
            diagonal: Vec::new(),
            pivots: Vec::new(),
        }
    }
    fn fits(&self, p: Shape) -> bool {
        let b2 = p.block * p.block;
        self.ptr.len() == p.rows + 1
            && self.col.len() == p.blocks
            && self.level.len() == p.blocks
            && self.values.len() == p.blocks * b2
            && self.diagonal.len() == p.rows * b2
            && self.pivots.len() == p.rows * p.block
    }
    fn retained_bytes(&self) -> usize {
        (self.ptr.capacity() + self.col.capacity() + self.level.capacity() + self.pivots.capacity())
            * size_of::<usize>()
            + (self.values.capacity() + self.diagonal.capacity()) * size_of::<S>()
    }
    pub(super) fn block(&self, pos: usize, b2: usize) -> &[S] {
        &self.values[pos * b2..(pos + 1) * b2]
    }
}
/// Holds the scratch storage of [`IncompleteFactor::refactor`].
///
/// The workspace holds a row accumulator and a second factor into which
/// `refactor` writes before it replaces the current one. Size it with
/// [`IncompleteSymbolic::workspace_requirements`], or create it with
/// [`IncompleteSymbolic::workspace`]. It also records the last attempt.
///
/// # Examples
///
/// ```
/// use scioxide_core::{AttemptOutcome, Capacity, ExecutionContext};
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
///     IncompleteSymbolic, IncompleteWorkspace, SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(
///     2,
///     2,
///     vec![0, 2, 4],
///     vec![0, 1, 0, 1],
///     vec![2.0_f64, -1., -1., 2.],
/// )?;
/// let plan = IncompleteSymbolic::analyze(
///     a.structural_pattern(),
///     IncompleteMethod::Ic0,
///     1,
///     IncompleteLimits::new(3, 4096),
/// )?;
/// let requirements = plan.workspace_requirements::<f64>(())?;
/// let mut workspace = IncompleteWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let (options, context) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let mut factor = IncompleteFactor::factor_scoped(
///     plan,
///     a.view(),
///     IncompleteOptions::new(),
///     options,
///     context,
/// )?;
/// factor.refactor(a.view(), options, &mut workspace, context)?;
/// let outcome = workspace.last_attempt().map(|attempt| attempt.outcome());
/// assert_eq!(outcome, Some(AttemptOutcome::Succeeded));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct IncompleteWorkspace<S: NumericScalar> {
    epoch: crate::factor_solve::Epoch,
    candidate: Factors<S>,
    source: Vec<S>,
    seen: Vec<bool>,
    row: Row<S>,
    temp: Vec<S>,
    incidence: Incidence,
}
impl<S: NumericScalar> Default for IncompleteWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> IncompleteWorkspace<S> {
    /// Creates an empty workspace. Call [`try_reserve`](Self::try_reserve) before
    /// use.
    pub fn new() -> Self {
        Self {
            epoch: crate::factor_solve::Epoch::new(),
            candidate: Factors::empty(),
            source: Vec::new(),
            seen: Vec::new(),
            row: Row::empty(),
            temp: Vec::new(),
            incidence: Incidence::empty(),
        }
    }
    // Reserve every buffer of `shape`.
    fn reserve(shape: Shape) -> Result<Self, SciOxideError> {
        const TEMP: Quantity = scioxide_core::quantity!("temporary_block");
        Ok(Self {
            epoch: crate::factor_solve::Epoch::new(),
            candidate: Factors::new(shape)?,
            source: try_filled(
                shape.entries,
                S::ZERO,
                scioxide_core::quantity!("source_values"),
            )?,
            seen: try_filled(
                shape.entries,
                false,
                scioxide_core::quantity!("source_seen"),
            )?,
            row: Row::new(shape)?,
            temp: try_filled(mul(shape.block, shape.block, TEMP)?, S::ZERO, TEMP)?,
            incidence: Incidence::new(shape)?,
        })
    }
    /// Returns the bytes this workspace holds, including its second factor.
    pub fn retained_bytes(&self) -> usize {
        self.candidate.retained_bytes()
            + (self.source.capacity() + self.temp.capacity()) * size_of::<S>()
            + self.seen.capacity()
            + self.row.retained_bytes()
            + self.incidence.retained_bytes()
    }
    /// Returns the record of the last factorization attempt, if any.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.epoch.record()
    }
    /// Reports whether the workspace already fits `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the analysis, one factor and this workspace together
    ///   exceed the byte limit of `r`.
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn check(&self, r: &IncompleteRequirements) -> Result<Capacity, SciOxideError> {
        let q = scioxide_core::quantity!("workspace_bytes");
        let bytes = add(
            add(r.symbolic_bytes(), r.factor_bytes(), q)?,
            self.retained_bytes(),
            q,
        )?;
        if bytes > r.limit {
            return Err(SciOxideError::resource_limit(q, bytes, r.limit));
        }
        let shape = r.shape;
        Ok(
            if self.candidate.fits(shape)
                && self.source.len() == shape.entries
                && self.seen.len() == shape.entries
                && self.temp.len() == shape.block * shape.block
                && self.row.fits(shape)
                && self.incidence.fits(shape)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to fit `r`. If growth fails, the workspace keeps its
    /// current storage.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the required storage, or the current storage plus the
    ///   new storage during growth, exceeds the byte limit of `r`.
    /// - `SizeOverflow`, `AllocationFailure` if the storage cannot be sized or
    ///   allocated.
    pub fn try_reserve(&mut self, r: &IncompleteRequirements) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("workspace_bytes");
        if r.total_bytes() > r.limit {
            return Err(SciOxideError::resource_limit(q, r.total_bytes(), r.limit));
        }
        if self.check(r)? == Capacity::Sufficient {
            return Ok(());
        }
        let projected = add(r.total_bytes(), self.retained_bytes(), q)?;
        if projected > r.limit {
            return Err(SciOxideError::resource_limit(q, projected, r.limit));
        }
        let replacement = Self::reserve(r.shape)?;
        let _ = replacement.check(r)?;
        *self = replacement;
        Ok(())
    }
}
/// Holds an incomplete LU or Cholesky factorization `A ≈ L U` or `A ≈ L Lᴴ`,
/// for use as a preconditioner.
///
/// The method, block size and dropping rules come from [`IncompleteSymbolic`]
/// and [`IncompleteOptions`]. `solve_into` applies the approximate inverse
/// `(L U)⁻¹`, and the factor also implements the core
/// [`Preconditioner`](scioxide_core::Preconditioner) trait for Krylov solvers. Each
/// diagonal block is factored densely, by LU with partial pivoting or by
/// Cholesky. A missing diagonal entry is treated as a stored zero.
///
/// A failed `refactor` keeps the previous factor and its diagnostics.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_sparse::{
///     CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
///     IncompleteSymbolic, SparseOperation, SparseSolveOptions,
/// };
/// let (ptr, cols) = (vec![0, 2, 5, 7], vec![0, 1, 0, 1, 2, 1, 2]);
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     ptr.clone(),
///     cols.clone(),
///     vec![4.0_f64, -1., -1., 4., -1., -1., 4.],
/// )?;
/// let plan = IncompleteSymbolic::analyze(
///     a.structural_pattern(),
///     IncompleteMethod::Ilu { level: 0 },
///     1,
///     IncompleteLimits::new(7, 4096),
/// )?;
/// let (options, context) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let mut factor = IncompleteFactor::factor_scoped(
///     plan,
///     a.view(),
///     IncompleteOptions::new(),
///     options,
///     context,
/// )?;
/// // Tridiagonal ILU(0) has no fill, so it is the exact LU factorization.
/// let (mut x, mut staging) = ([0.0; 3], factor.workspace(1)?);
/// factor.solve_into(
///     &[3., 2., 3.],
///     SparseOperation::Normal,
///     options,
///     &mut x,
///     &mut staging,
///     context,
/// )?;
/// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-14));
/// // Refactor new values on the same pattern with a reusable workspace.
/// let mut workspace = factor.symbolic().workspace::<f64>(())?;
/// let b = CsrMatrix::from_raw(3, 3, ptr, cols, vec![8.0_f64, -2., -2., 8., -2., -2., 8.])?;
/// factor.refactor(b.view(), options, &mut workspace, context)?;
/// // A different pattern is rejected and the previous factor still solves.
/// let diagonal = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3])?;
/// assert!(
///     factor
///         .refactor(diagonal.view(), options, &mut workspace, context)
///         .is_err()
/// );
/// factor.solve_into(
///     &[6., 4., 6.],
///     SparseOperation::Normal,
///     options,
///     &mut x,
///     &mut staging,
///     context,
/// )?;
/// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct IncompleteFactor<S: NumericScalar> {
    pub(super) symbolic: IncompleteSymbolic,
    pub(super) data: Factors<S>,
    options: IncompleteOptions<S::Real>,
    diagnostics: IncompleteDiagnostics,
}
impl<S: NumericScalar> IncompleteFactor<S> {
    /// Computes the ILU(0) factorization of the square matrix `A = matrix`:
    /// incomplete LU that keeps exactly the pattern of `A`, with the diagonal
    /// added where it is not stored.
    ///
    /// This analyzes `A` with [`IncompleteMethod::Ilu`] at `level: 0` in its
    /// natural order, block size 1, `nnz(A) + n` factor blocks (which level 0
    /// never exceeds) and no byte limit, and factors it with
    /// [`IncompleteOptions::new`]: no dropping and
    /// [`IncompletePivotPolicy::Strict`] pivots. Use it as a preconditioner;
    /// [`solve`](Self::solve) applies `(L U)⁻¹`.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, IncompleteFactor};
    ///
    /// // tridiag(−1, 4, −1) has no fill, so ILU(0) is its exact LU factorization.
    /// let (rows, cols) = ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]);
    /// let a = CsrMatrix::from_triplets(3, 3, &rows, &cols, &[4.0_f64, -1., -1., 4., -1., -1., 4.])?;
    /// let z = IncompleteFactor::ilu0(&a)?.solve(&[3., 2., 3.])?;
    /// assert!(z.iter().all(|v| (v - 1.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`factor`](Self::factor) other than `ResourceLimit`.
    pub fn ilu0(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        Self::level_zero(matrix, IncompleteMethod::Ilu { level: 0 })
    }
    /// Computes the IC(0) factorization `A ≈ L Lᴴ` of the Hermitian matrix
    /// `A = matrix`, which stores both triangles: incomplete Cholesky with the
    /// pattern of the lower triangle of `A`.
    ///
    /// This analyzes `A` with [`IncompleteMethod::Ic0`] as
    /// [`ilu0`](Self::ilu0) does and factors it with
    /// [`IncompleteOptions::new`]: no dropping and
    /// [`IncompletePivotPolicy::Strict`] pivots, so a pivot that is not
    /// positive is an error.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, IncompleteFactor};
    ///
    /// let (rows, cols) = ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]);
    /// let a = CsrMatrix::from_triplets(3, 3, &rows, &cols, &[4.0_f64, -1., -1., 4., -1., -1., 4.])?;
    /// let z = IncompleteFactor::ic0(&a)?.solve(&[3., 2., 3.])?;
    /// assert!(z.iter().all(|v| (v - 1.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`factor`](Self::factor) other than `ResourceLimit`.
    pub fn ic0(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        Self::level_zero(matrix, IncompleteMethod::Ic0)
    }
    // Level-0 analysis: the pattern of `A` and the diagonal fit `nnz + n` blocks.
    fn level_zero(
        matrix: &crate::CsrMatrix<S>,
        method: IncompleteMethod,
    ) -> Result<Self, SciOxideError> {
        let symbolic = IncompleteSymbolic::analyze(
            matrix.structural_pattern(),
            method,
            1,
            IncompleteLimits::new(matrix.nnz().saturating_add(matrix.rows()), usize::MAX),
        )?;
        Self::factor(symbolic, matrix.view(), IncompleteOptions::new())
    }
    /// Computes the incomplete factorization of `matrix` with the analysis
    /// `symbolic` and the dropping and pivot rules of `options`, with the
    /// default [`SparseSolveOptions`] in a fresh scope.
    ///
    /// This is [`factor_scoped`](Self::factor_scoped) with
    /// [`SparseSolveOptions::new`] and
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - The errors of [`factor_scoped`](Self::factor_scoped).
    pub fn factor(
        symbolic: IncompleteSymbolic,
        matrix: SparseView<'_, S>,
        options: IncompleteOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_scoped(
            symbolic,
            matrix,
            options,
            SparseSolveOptions::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
    /// Computes the incomplete factorization of `matrix` with the analysis
    /// `symbolic`, inside the caller's scope.
    ///
    /// `options` sets the dropping and pivot rules for this factor and every
    /// later `refactor`. `solve` sets the [`Accuracy`](scioxide_core::Accuracy) and
    /// the iteration limit, and each block row counts as one iteration. This
    /// form allocates its own workspace; use [`refactor`](Self::refactor) to
    /// reuse one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a drop tolerance or repair floor is invalid, if
    ///   `matrix` does not have the analyzed pattern, or if IC(0) or ICT gets a
    ///   matrix that is not exactly Hermitian.
    /// - `DimensionMismatch` if `matrix` has the wrong shape.
    /// - `Singular` if a pivot is zero, or not positive for Cholesky, under
    ///   [`IncompletePivotPolicy::Strict`].
    /// - `ResourceLimit` if the factor keeps more blocks than the analysis limit
    ///   or the storage exceeds the byte limit.
    /// - `NumericalInstability` if a value overflows or a nonzero result
    ///   underflows to zero.
    /// - `ConvergenceFailure` if the iteration limit is reached;
    ///   `ExecutionStopped` on cancellation or a deadline.
    /// - `SizeOverflow`, `AllocationFailure` if storage cannot be sized or
    ///   allocated.
    pub fn factor_scoped<'scope, 'capability>(
        symbolic: IncompleteSymbolic,
        matrix: SparseView<'_, S>,
        options: IncompleteOptions<S::Real>,
        solve: SparseSolveOptions,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let options = options.validate()?;
        let mut workspace = symbolic.workspace(())?;
        let mut result = Self {
            data: Factors::new(symbolic.shape())?,
            symbolic,
            options,
            diagnostics: IncompleteDiagnostics::default(),
        };
        result.refactor(matrix, solve, &mut workspace, scope)?;
        Ok(result)
    }
    /// Returns the symbolic analysis.
    pub fn symbolic(&self) -> &IncompleteSymbolic {
        &self.symbolic
    }
    /// Returns the dropping and pivot options.
    pub fn options(&self) -> IncompleteOptions<S::Real> {
        self.options
    }
    /// Returns the diagnostics of the current factor.
    pub fn diagnostics(&self) -> IncompleteDiagnostics {
        self.diagnostics
    }
    /// Iterates over the stored blocks as `(block row, block column, values)`,
    /// sorted by row, then column. Block values are row-major.
    ///
    /// For LU, off-diagonal blocks are the blocks of the unit lower factor `L`
    /// and the upper factor `U`, and a diagonal block holds its own dense LU
    /// factors (see [`diagonal_pivots`](Self::diagonal_pivots)). For Cholesky the
    /// blocks are those of `L`.
    pub fn blocks(&self) -> impl Iterator<Item = (usize, usize, &[S])> + '_ {
        let b2 = self.symbolic.block_size * self.symbolic.block_size;
        (0..self.symbolic.rows()).flat_map(move |r| {
            (self.data.ptr[r]..self.data.ptr[r + 1])
                .map(move |p| (r, self.data.col[p], self.data.block(p, b2)))
        })
    }
    /// Returns the row interchanges of the dense LU factorization of diagonal
    /// block `row`: step `k` swapped rows `k` and `pivots[k]`. Returns an empty
    /// slice for Cholesky.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row` is not a block row.
    pub fn diagonal_pivots(&self, row: usize) -> Result<&[usize], SciOxideError> {
        if row >= self.symbolic.rows() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("block_rows"),
                row,
                self.symbolic.rows(),
            ));
        }
        let b = self.symbolic.block_size;
        Ok(if self.symbolic.method.cholesky() {
            &[]
        } else {
            &self.data.pivots[row * b..(row + 1) * b]
        })
    }
    /// Recomputes the factor for new values of `matrix` on the analyzed pattern,
    /// with the stored [`IncompleteOptions`].
    ///
    /// `options` sets the [`Accuracy`](scioxide_core::Accuracy) and the iteration
    /// limit, and each block row counts as one iteration. The call does not
    /// allocate. On error the previous factor stays usable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `matrix` does not have the analyzed pattern, or if
    ///   IC(0) or ICT gets a matrix that is not exactly Hermitian.
    /// - `DimensionMismatch` if `matrix` has the wrong shape or `workspace` was
    ///   not reserved for this analysis.
    /// - `Singular` if a pivot is zero, or not positive for Cholesky, under
    ///   [`IncompletePivotPolicy::Strict`].
    /// - `ResourceLimit` if the factor keeps more blocks than the analysis limit.
    /// - `NumericalInstability` if a value overflows or a nonzero result
    ///   underflows to zero.
    /// - `ConvergenceFailure` if the iteration limit is reached;
    ///   `ExecutionStopped` on cancellation or a deadline.
    pub fn refactor<'scope, 'capability>(
        &mut self,
        matrix: SparseView<'_, S>,
        options: SparseSolveOptions,
        workspace: &mut IncompleteWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.epoch.reject();
        let requirements = self.symbolic.workspace_requirements::<S>(())?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            return Err(SciOxideError::dimension_mismatch(
                requirements.total_bytes(),
                workspace.retained_bytes(),
            ));
        }
        let mut epoch = core::mem::take(&mut workspace.epoch);
        let result = epoch.run(scope, options, |context| {
            self.run(
                matrix,
                workspace,
                Arithmetic(options.accuracy()),
                context.execution(),
                |_| admit(context),
            )
        });
        workspace.epoch = epoch;
        self.diagnostics = result?;
        core::mem::swap(&mut self.data, &mut workspace.candidate);
        Ok(())
    }
    // Transactional numeric epoch.
    fn run(
        &self,
        matrix: SparseView<'_, S>,
        w: &mut IncompleteWorkspace<S>,
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
        mut admission: impl FnMut(usize) -> Result<(), SciOxideError>,
    ) -> Result<IncompleteDiagnostics, SciOxideError> {
        let options = self.options;
        let p = &self.symbolic;

        context.check()?;
        load(p, matrix, w)?;
        let n = p.rows();
        let b = p.block_size;
        let b2 = b * b;
        let chol = p.method.cholesky();
        let mut report = IncompleteDiagnostics::default();
        let mut end = 0;
        w.candidate.ptr.fill(0);
        w.incidence.heads.fill(usize::MAX);
        for i in 0..n {
            admission(i)?;
            context.check()?;
            w.row.clear();
            for r in i * b..(i + 1) * b {
                for pos in p.pattern().row_ptr()[r]..p.pattern().row_ptr()[r + 1] {
                    let c = p.pattern().col_indices()[pos];
                    let j = c / b;
                    w.row.touch(j, i, b2);
                    w.row.values[j * b2 + (r % b) * b + c % b] = w.source[pos];
                    w.row.levels[j] = 0;
                }
            }
            // Scale is the maximum Frobenius norm among original row blocks.
            let mut scale = Scaled::ZERO;
            for &j in &w.row.columns {
                let v = norm(&w.row.values[j * b2..(j + 1) * b2])?;
                if v.abs_cmp(scale).is_gt() {
                    scale = v;
                }
            }
            let threshold = scale.mul(Scaled::from_value(options.drop_tolerance)?)?;
            let floor = match options.pivot {
                IncompletePivotPolicy::Strict => Scaled::ZERO,
                IncompletePivotPolicy::Repair { absolute, relative } => {
                    let a = Scaled::from_value(absolute)?;
                    let r = scale.mul(Scaled::from_value(relative)?)?;
                    if a.abs_cmp(r).is_gt() { a } else { r }
                }
            };
            w.row.touch(i, i, b2);
            w.row.levels[i] = 0; // Diagonal completion is explicit; strict policy still rejects zero pivots.
            if chol {
                cholesky_row(p, w, i, threshold, (context, arithmetic), &mut report)?;
            } else {
                lu_row(p, w, i, threshold, (context, arithmetic), &mut report)?;
            }
            w.row.columns.sort_unstable();
            if p.method.threshold() {
                truncate(w, i, (options.max_off_diagonal, chol, b2), &mut report)?;
            }
            if chol {
                // The diagonal must reflect only the retained lower row.
                w.temp.copy_from_slice(&w.row.values[i * b2..(i + 1) * b2]);
                for &j in &w.row.columns {
                    if j < i && w.row.levels[j] != usize::MAX {
                        update(
                            &mut w.temp,
                            &w.row.values[j * b2..(j + 1) * b2],
                            &w.row.values[j * b2..(j + 1) * b2],
                            b,
                            true,
                            arithmetic,
                        )?;
                    }
                }
                w.row.values[i * b2..(i + 1) * b2].copy_from_slice(&w.temp);
            }
            let diagonal = &mut w.candidate.diagonal[i * b2..(i + 1) * b2];
            diagonal.copy_from_slice(&w.row.values[i * b2..(i + 1) * b2]);
            block::factor(
                diagonal,
                &mut w.candidate.pivots[i * b..(i + 1) * b],
                b,
                chol,
                floor,
                i * b,
                &mut report.repaired_pivots,
                arithmetic,
            )?;
            w.row.values[i * b2..(i + 1) * b2].copy_from_slice(diagonal);
            w.candidate.ptr[i] = end;
            for &j in &w.row.columns {
                if (!chol || j <= i) && w.row.levels[j] != usize::MAX {
                    if end == p.limits.max_factor_blocks {
                        return Err(SciOxideError::resource_limit(
                            scioxide_core::quantity!("factor_blocks"),
                            end.saturating_add(1),
                            p.limits.max_factor_blocks,
                        ));
                    }
                    w.candidate.col[end] = j;
                    w.candidate.level[end] = w.row.levels[j];
                    w.candidate.values[end * b2..(end + 1) * b2]
                        .copy_from_slice(&w.row.values[j * b2..(j + 1) * b2]);
                    if matches!(p.method, IncompleteMethod::Ict) && j < i {
                        w.incidence.insert(i, j, end);
                    }
                    end += 1;
                }
            }
            w.candidate.ptr[i + 1] = end;
        }
        context.check()?;
        report.factor_blocks = end;
        Ok(report)
    }
}
// Scatters the caller's `matrix` into `w`.
fn load<S: NumericScalar>(
    p: &IncompleteSymbolic,
    m: SparseView<'_, S>,
    w: &mut IncompleteWorkspace<S>,
) -> Result<(), SciOxideError> {
    const MATRIX: Argument = scioxide_core::argument!("matrix");
    crate::helpers::shape((m.rows(), m.cols()), (p.dimension(), p.dimension()))?;

    w.seen.fill(false);
    let pattern = p.pattern();
    for (r, c, v) in m.iter() {
        let start = pattern.row_ptr()[r];
        let index = pattern.col_indices()[start..pattern.row_ptr()[r + 1]]
            .binary_search(&c)
            .map_err(|_| invalid(MATRIX, "matrix pattern differs"))?
            + start;
        if w.seen[index] {
            return Err(invalid(MATRIX, "duplicate input coordinate"));
        }
        w.source[index] = v;
        w.seen[index] = true;
    }
    if !w.seen.iter().all(|&v| v) {
        return Err(invalid(MATRIX, "matrix pattern differs"));
    }
    if p.method.cholesky() {
        for r in 0..p.dimension() {
            for pos in pattern.row_ptr()[r]..pattern.row_ptr()[r + 1] {
                let c = pattern.col_indices()[pos];
                let start = pattern.row_ptr()[c];
                let other = pattern.col_indices()[start..pattern.row_ptr()[c + 1]]
                    .binary_search(&r)
                    .ok()
                    .map_or(S::ZERO, |j| w.source[start + j]);
                if w.source[pos] != other.conj() {
                    return Err(invalid(MATRIX, "ICT/IC0 requires a Hermitian matrix"));
                }
            }
        }
    }
    Ok(())
}
// Eliminates block row `i`.
fn lu_row<S: NumericScalar>(
    p: &IncompleteSymbolic,
    w: &mut IncompleteWorkspace<S>,
    i: usize,
    threshold: Scaled<S::Real>,
    (context, arithmetic): (ExecutionContext<'_>, Arithmetic),
    report: &mut IncompleteDiagnostics,
) -> Result<(), SciOxideError> {
    let b = p.block_size;
    let b2 = b * b;
    while let Some(j) = w.row.next_lower() {
        context.check()?;
        if p.method.threshold()
            && norm(&w.row.values[j * b2..(j + 1) * b2])?
                .abs_cmp(threshold)
                .is_lt()
        {
            w.row.levels[j] = usize::MAX;
            w.row.values[j * b2..(j + 1) * b2].fill(S::ZERO);
            report.dropped_blocks += 1;
            continue;
        }
        block::right_solve(
            &mut w.row.values[j * b2..(j + 1) * b2],
            &w.candidate.diagonal[j * b2..(j + 1) * b2],
            &w.candidate.pivots[j * b..(j + 1) * b],
            b,
            false,
            arithmetic,
        )?;
        w.temp.copy_from_slice(&w.row.values[j * b2..(j + 1) * b2]);
        for pos in w.candidate.ptr[j]..w.candidate.ptr[j + 1] {
            let k = w.candidate.col[pos];
            if k <= j {
                continue;
            }
            let level = w.row.levels[j]
                .saturating_add(w.candidate.level[pos])
                .saturating_add(1);
            if let IncompleteMethod::Ilu { level: limit } = p.method
                && w.row.levels[k] == usize::MAX
                && level > limit
            {
                continue;
            }
            w.row.touch(k, i, b2);
            w.row.levels[k] = w.row.levels[k].min(level);
            update(
                &mut w.row.values[k * b2..(k + 1) * b2],
                &w.temp,
                w.candidate.block(pos, b2),
                b,
                false,
                arithmetic,
            )?;
        }
    }
    if p.method.threshold() {
        for &j in &w.row.columns {
            if j > i
                && w.row.levels[j] != usize::MAX
                && norm(&w.row.values[j * b2..(j + 1) * b2])?
                    .abs_cmp(threshold)
                    .is_lt()
            {
                w.row.levels[j] = usize::MAX;
                report.dropped_blocks += 1;
            }
        }
    }
    Ok(())
}
// Eliminates block row `i`.
fn cholesky_row<S: NumericScalar>(
    p: &IncompleteSymbolic,
    w: &mut IncompleteWorkspace<S>,
    i: usize,
    threshold: Scaled<S::Real>,
    (context, arithmetic): (ExecutionContext<'_>, Arithmetic),
    report: &mut IncompleteDiagnostics,
) -> Result<(), SciOxideError> {
    let b = p.block_size;
    let b2 = b * b;
    while let Some(j) = w.row.next_lower() {
        context.check()?;
        w.temp.copy_from_slice(&w.row.values[j * b2..(j + 1) * b2]);
        // The queue contains original lower blocks or ICT candidates reached
        // through a retained earlier column. Every scheduled block is active.
        for pos in w.candidate.ptr[j]..w.candidate.ptr[j + 1] {
            let k = w.candidate.col[pos];
            if k < j && w.row.levels[k] != usize::MAX {
                update(
                    &mut w.temp,
                    &w.row.values[k * b2..(k + 1) * b2],
                    w.candidate.block(pos, b2),
                    b,
                    true,
                    arithmetic,
                )?;
            }
        }
        if p.method.threshold() && norm(&w.temp)?.abs_cmp(threshold).is_lt() {
            w.row.levels[j] = usize::MAX;
            w.row.values[j * b2..(j + 1) * b2].fill(S::ZERO);
            report.dropped_blocks += 1;
            continue;
        }
        block::right_solve(
            &mut w.temp,
            &w.candidate.diagonal[j * b2..(j + 1) * b2],
            &[],
            b,
            true,
            arithmetic,
        )?;
        w.row.values[j * b2..(j + 1) * b2].copy_from_slice(&w.temp);
        w.row.levels[j] = 0;
        if matches!(p.method, IncompleteMethod::Ict) {
            let mut pos = w.incidence.heads[j];
            while pos != usize::MAX {
                let k = w.incidence.rows[pos];
                w.row.touch(k, i, b2);
                w.row.levels[k] = 0;
                pos = w.incidence.next[pos];
            }
        }
    }
    Ok(())
}
// Keeps the largest `cap` blocks per triangle.
fn truncate<S: NumericScalar>(
    w: &mut IncompleteWorkspace<S>,
    i: usize,
    (cap, chol, b2): (usize, bool, usize),
    report: &mut IncompleteDiagnostics,
) -> Result<(), SciOxideError> {
    for lower in [true, false] {
        if chol && !lower {
            break;
        }
        w.row.rank.clear();
        for &j in &w.row.columns {
            if (if lower { j < i } else { j > i }) && w.row.levels[j] != usize::MAX {
                w.row
                    .rank
                    .push((j, norm(&w.row.values[j * b2..(j + 1) * b2])?));
            }
        }
        w.row
            .rank
            .sort_unstable_by(|(a, x), (b, y)| y.abs_cmp(*x).then_with(|| a.cmp(b)));
        for &(j, _) in &w.row.rank[cap.min(w.row.rank.len())..] {
            w.row.levels[j] = usize::MAX;
            report.dropped_blocks += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn ilu_fill_keeps_original_and_elimination_levels() -> Result<(), SciOxideError> {
        let a = crate::CsrMatrix::from_raw(
            4,
            4,
            vec![0, 3, 6, 9, 12],
            vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
            vec![6.0_f64, 1., -1., 1., 5., 1., 1., 4., 1., -1., 1., 3.],
        )?;
        let p = IncompleteSymbolic::analyze(
            a.structural_pattern(),
            IncompleteMethod::Ilu { level: 1 },
            1,
            IncompleteLimits {
                max_factor_blocks: 16,
                max_workspace_bytes: 4096,
            },
        )?;
        let f = IncompleteFactor::factor(
            p.try_clone()?,
            a.view(),
            IncompleteOptions {
                drop_tolerance: 0.,
                max_off_diagonal: usize::MAX,
                pivot: IncompletePivotPolicy::Strict,
            },
        )?;
        for i in 0..4 {
            for pos in f.data.ptr[i]..f.data.ptr[i + 1] {
                let j = f.data.col[pos];
                let original = a.col_indices()[a.row_ptr()[i]..a.row_ptr()[i + 1]].contains(&j);
                assert_eq!(f.data.level[pos], usize::from(!original));
                assert!(original || (i, j) == (1, 3) || (i, j) == (3, 1));
            }
        }
        assert_eq!(f.diagnostics().factor_blocks, 14);
        Ok(())
    }
}
