use super::*;
use crate::factor_solve::{
    Application, Epoch, Oriented, SparseSolveRequirements, SparseSolveWorkspace,
};

/// Holds the rank and the residuals of a minimum-norm least-squares solve.
///
/// The residuals are computed from the original matrix `A`, not from its
/// truncated factorization.
///
/// ```
/// use scioxide_core::{ExecutionContext, RankThreshold};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOrdering, SparseQr, SparseQrLimits, SparseQrMinimumNormPlan,
///     SparseQrSymbolic, SparseSolveOptions,
/// };
/// // Least squares for A = [1, 1]ᵀ and b = [1, 3]: x = 2 with residual [1, -1].
/// let a = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![1.0_f64, 1.])?;
/// let symbolic = SparseQrSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Natural,
///     SparseQrLimits::new(32, 16, 100_000),
/// )?;
/// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let factor = SparseQr::factor_scoped(symbolic, a.view(), options, scope)?;
/// let threshold = RankThreshold::new(0., 1e-12)?;
/// let plan = SparseQrMinimumNormPlan::build(factor, threshold, options, scope)?;
/// let (x, report) = plan.solve(&[1., 3.])?;
/// assert_eq!((report.rank(), report.threshold()), (1, plan.threshold()));
/// assert!((x[0] - 2.).abs() < 1e-14);
/// assert!((report.residual().raw_norm() - 2_f64.sqrt()).abs() < 1e-14);
/// assert!(report.normal_residual().raw_norm() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseQrSolveReport<R: RealScalar> {
    rank: usize,
    threshold: R,
    residual: ResidualReport<R>,
    normal_residual: ResidualReport<R>,
}
impl<R: RealScalar> SparseQrSolveReport<R> {
    /// Returns the rank `r` of the truncated factorization used by the solve.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Returns the absolute cutoff `τ` that decided the rank.
    pub const fn threshold(&self) -> R {
        self.threshold
    }
    /// Returns the residual `‖op(A) x − b‖₂`.
    pub const fn residual(&self) -> ResidualReport<R> {
        self.residual
    }
    /// Returns the normal-equations residual `‖op(A)ᴴ (op(A) x − b)‖₂`, which
    /// is zero at an exact least-squares solution.
    pub const fn normal_residual(&self) -> ResidualReport<R> {
        self.normal_residual
    }
}
struct ResidualScratch<'a, S> {
    residual: &'a mut [S],
    normal: &'a mut [S],
}

// The square Q, Qᵀ or Qᴴ action of a raw factor; reflectors need no scratch
// beyond the staged candidate column.
impl<S: NumericScalar> Application<S> for Oriented<'_, SparseQr<S>> {
    fn dimension(&self) -> usize {
        self.factor.symbolic.rows()
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn staging_limits(&self) -> Result<(usize, usize), SciOxideError> {
        let symbolic = &self.factor.symbolic;
        Ok((
            symbolic.workspace_requirements::<S>(())?.raw_bytes,
            symbolic.limits.max_workspace_bytes,
        ))
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        _: &mut [S],
        _: SparseSolveOptions,
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        candidate.copy_from_slice(rhs);
        apply_q_operation(&self.factor.data, candidate, self.operation)
    }
}

impl<S: NumericScalar> SparseQr<S> {
    fn oriented(&self, operation: SparseOperation) -> Oriented<'_, Self> {
        Oriented {
            factor: self,
            operation,
        }
    }
    /// Returns the scratch size for applying `Q` to `rhs` vectors: `m` scalars
    /// per vector.
    ///
    /// Owned scratch counts toward the byte limit of the analysis, together
    /// with the factor and its workspace. A borrowed slice is checked only for
    /// its length.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        rhs: usize,
    ) -> Result<SparseSolveRequirements, SciOxideError> {
        crate::factor_solve::requirements(&self.oriented(SparseOperation::Normal), rhs)
    }
    /// Creates a workspace that owns scratch for applying `Q` to `rhs` vectors.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the analysis.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn workspace(&self, rhs: usize) -> Result<SparseSolveWorkspace<'static, S>, SciOxideError> {
        crate::factor_solve::workspace(&self.oriented(SparseOperation::Normal), rhs)
    }
    /// Applies `Q`, `Qᵀ` or `Qᴴ` to a vector of length `m` and writes the
    /// result to `output`.
    ///
    /// `operation` selects `Q` (`Normal`), `Qᵀ` or `Qᴴ`. The application counts
    /// as one iteration toward the limits of `options` and `scope` (see
    /// [Execution](crate#execution)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `output` does not have length `m`, or
    ///   the scratch in `workspace` is too short.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a reflection overflows.
    /// - `ResourceLimit` if owned scratch exceeds the byte limit.
    /// - `ConvergenceFailure` if the iteration limit has been reached.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the
    ///   [execution context](scioxide_core#execution-contexts).
    pub fn apply_q_into<'s, 'c>(
        &self,
        rhs: &[S],
        operation: SparseOperation,
        options: SparseSolveOptions,
        output: &mut [S],
        workspace: &mut SparseSolveWorkspace<'_, S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<(), SciOxideError> {
        crate::factor_solve::apply_into(
            &self.oriented(operation),
            rhs,
            options,
            output,
            workspace,
            scope,
        )
    }
    /// Applies `Q`, `Qᵀ` or `Qᴴ` to each column of a column-major block and
    /// writes the results to `output`.
    ///
    /// Each column counts as one iteration. Under a parallel execution policy
    /// in `options`, columns are processed in parallel. `output` is written only
    /// after every column is done.
    ///
    /// # Errors
    ///
    /// - The errors of [`apply_q_into`](Self::apply_q_into).
    /// - `DimensionMismatch` also if `rhs` and `output` have different numbers
    ///   of columns.
    pub fn apply_q_block_into<'s, 'c>(
        &self,
        rhs: BlockVectorView<'_, S>,
        operation: SparseOperation,
        options: SparseSolveOptions,
        output: &mut BlockVectorViewMut<'_, S>,
        workspace: &mut SparseSolveWorkspace<'_, S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<(), SciOxideError> {
        crate::factor_solve::apply_block_into(
            &self.oriented(operation),
            rhs,
            options,
            output,
            workspace,
            scope,
        )
    }

    // Original-system residual diagnostics for `system`.
    fn residual_report(
        &self,
        (rhs, solution): (&[S], &[S]),
        rank: usize,
        threshold: S::Real,
        system: SparseOperation,
        scratch: ResidualScratch<'_, S>,
    ) -> Result<SparseQrSolveReport<S::Real>, SciOxideError> {
        let ResidualScratch { residual, normal } = scratch;
        let (m, n) = (self.symbolic.rows(), self.symbolic.cols());
        let (out_len, in_len) = if system == SparseOperation::Normal {
            (m, n)
        } else {
            (n, m)
        };
        for (slot, &value) in residual[..out_len].iter_mut().zip(rhs) {
            *slot = -value;
        }
        normal[..in_len].fill(S::ZERO);
        let pattern = self.symbolic.pattern();
        for row in 0..m {
            for pos in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[pos];
                let value = self.data.source[pos];
                match system {
                    SparseOperation::Normal => {
                        residual[row] = residual[row] + value * solution[col]
                    }
                    SparseOperation::Adjoint => {
                        residual[col] = residual[col] + value.conj() * solution[row]
                    }
                    SparseOperation::Transpose => {
                        residual[col] = residual[col] + value * solution[row]
                    }
                }
            }
        }
        for row in 0..m {
            for pos in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[pos];
                let value = self.data.source[pos];
                match system {
                    SparseOperation::Normal => {
                        normal[col] = normal[col] + value.conj() * residual[row]
                    }
                    SparseOperation::Adjoint => normal[row] = normal[row] + value * residual[col],
                    SparseOperation::Transpose => {
                        normal[row] = normal[row] + value.conj() * residual[col]
                    }
                }
            }
        }
        let mut residual_norm = ScaledNorm::new();
        let mut normal_norm = ScaledNorm::new();
        for &value in &residual[..out_len] {
            residual_norm.add_magnitude(value.abs())?;
        }
        for &value in &normal[..in_len] {
            normal_norm.add_magnitude(value.abs())?;
        }
        Ok(SparseQrSolveReport {
            rank,
            threshold,
            residual: ResidualReport::unscaled(residual_norm.finish()?, NormSelection::L2)?,
            normal_residual: ResidualReport::unscaled(normal_norm.finish()?, NormSelection::L2)?,
        })
    }
}

/// Computes minimum-norm least-squares solutions from a sparse QR
/// factorization with a rank threshold.
///
/// For `op(A) = A`, a solve returns the vector `x` of smallest `‖x‖₂` among
/// the minimizers of `‖A x − b‖₂`; for `Aᵀ` and `Aᴴ` it does the same with
/// `op(A)` in place of `A`. Here `A` is the `m × n` matrix of the factor.
///
/// [`build`](Self::build) reduces `R` once more by Householder QR in the
/// column order of the factor. A column whose part below the current row has
/// 2-norm at most
///
/// ```text
/// τ = absolute + relative · max_j ‖A[:, j]‖₂
/// ```
///
/// gets no pivot, and the number of pivots is the rank `r`. The `r` accepted
/// rows are then factored by a Householder QR of their conjugate transpose,
/// which completes an orthogonal decomposition of rank `r`. The rank depends on
/// the column order. It is an estimate, not a bound derived from singular
/// values. Each solve applies Householder reflections and two triangular
/// substitutions, and reports residuals for the original matrix in a
/// [`SparseQrSolveReport`]. Solves need `6 · max(m, n)` scalars of scratch per
/// right-hand side.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, RankThreshold};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseOrdering, SparseQr, SparseQrLimits,
///     SparseQrMinimumNormPlan, SparseQrSymbolic, SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 1.0])?;
/// let symbolic = SparseQrSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Natural,
///     SparseQrLimits::new(32, 16, 100_000),
/// )?;
/// let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
/// let factor = SparseQr::factor_scoped(symbolic, a.view(), options, scope)?;
/// let plan =
///     SparseQrMinimumNormPlan::build(factor, RankThreshold::new(0., 1e-12)?, options, scope)?;
/// let mut workspace = plan.workspace(1)?;
/// let mut x = [0.; 2];
/// let report = plan.solve_into(
///     &[2.],
///     SparseOperation::Normal,
///     options,
///     &mut x,
///     &mut workspace,
///     scope,
/// )?;
/// assert_eq!(report.rank(), 1);
/// assert!((x[0] - 1.).abs() < 1e-14 && (x[1] - 1.).abs() < 1e-14);
/// // Scratch shorter than the requirement is rejected before the output changes.
/// let r = plan.workspace_requirements(1)?;
/// assert_eq!(r.scalar_len(), 6 * 2);
/// let mut short = [0.; 11];
/// let mut borrowed = scioxide_sparse::SparseSolveWorkspace::borrowed(&mut short);
/// assert!(
///     plan.solve_into(
///         &[4.],
///         SparseOperation::Normal,
///         options,
///         &mut x,
///         &mut borrowed,
///         scope
///     )
///     .is_err()
/// );
/// assert!((x[0] - 1.).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SparseQrMinimumNormPlan<S: NumericScalar> {
    factor: SparseQr<S>,
    compression: Data<S>,
    secondary: Data<S>,
    rank: usize,
    cutoff: S::Real,
}

impl<S: NumericScalar> SparseQrMinimumNormPlan<S> {
    /// Factors the `m × n` matrix `A = matrix` by Householder QR and builds the
    /// minimum-norm plan with the dimension-scaled rank cutoff.
    ///
    /// The analysis uses
    /// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing) and
    /// no byte limit. The rank threshold is
    /// [`RankThreshold::new(0, min(max(m, n)·ε, 1/2))`](RankThreshold::new),
    /// with `ε` the machine epsilon of `S::Real`, so the cutoff is
    ///
    /// ```text
    /// τ = max(m, n) · ε · max_j ‖A[:, j]‖₂.
    /// ```
    ///
    /// The entry limit `c` starts at `nnz(A) + max(m, n)` and doubles after each
    /// `ResourceLimit`; each attempt factors and builds with
    /// `max_factor_entries = min(c, k(m + n))` and
    /// `max_active_entries = min(c, m·n)`, `k = min(m, n)`, which no
    /// decomposition of the plan exceeds. The result equals
    /// [`analyze`](SparseQrSymbolic::analyze),
    /// [`SparseQr::factor`] and [`build`](Self::build) with sufficient limits
    /// and this threshold.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, SparseQrMinimumNormPlan};
    ///
    /// // x₀ + x₁ = 2 has the minimum-norm solution (1, 1).
    /// let a = CsrMatrix::from_triplets(1, 2, &[0, 0], &[0, 1], &[1.0_f64, 1.0])?;
    /// let plan = SparseQrMinimumNormPlan::new(&a)?;
    /// let (x, report) = plan.solve(&[2.0])?;
    /// assert_eq!(report.rank(), 1);
    /// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-15));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a column norm, the cutoff or a Householder
    ///   vector is not representable.
    /// - `Singular` if a diagonal entry of the second triangular factor is
    ///   exactly zero.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let (m, n) = (matrix.rows(), matrix.cols());
        let k = m.min(n);
        let symbolic = SparseQrSymbolic::analyze(
            matrix.structural_pattern(),
            crate::SparseOrdering::FillReducing,
            SparseQrLimits::new(usize::MAX, usize::MAX, usize::MAX),
        )?;
        let threshold = RankThreshold::new(S::Real::ZERO, dimension_epsilon::<S::Real>(m.max(n))?)?;
        let factor_bound = k.saturating_mul(m.saturating_add(n));
        let active_bound = m.saturating_mul(n);
        crate::capacity::with_capacity(
            matrix.nnz().saturating_add(m.max(n)),
            factor_bound.max(active_bound),
            |entries| {
                let mut attempt = symbolic.try_clone()?;
                attempt.limits = SparseQrLimits::new(
                    entries.min(factor_bound),
                    entries.min(active_bound),
                    usize::MAX,
                );
                let factor = SparseQr::factor(attempt, matrix.view())?;
                Self::build(
                    factor,
                    threshold,
                    SparseSolveOptions::new(),
                    ExecutionContext::sequential_native(),
                )
            },
        )
    }
    /// Computes the rank-`r` decomposition of `factor` for `threshold`, and
    /// takes ownership of the factor.
    ///
    /// The cutoff is `τ = threshold.threshold(s)`, where `s` is the largest
    /// 2-norm of a column of `A`. Building is sequential, and each pivot of the
    /// two decompositions counts as one iteration toward the limits of `options`
    /// and `scope`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if a decomposition needs more than `max_factor_entries`
    ///   or `max_active_entries`, or the plan exceeds the byte limit.
    /// - `NumericalInstability` if a column norm, the cutoff or a Householder
    ///   vector is not representable.
    /// - `Singular` if a diagonal entry of the second triangular factor is
    ///   exactly zero.
    /// - `ConvergenceFailure` if the iteration limit is reached.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the
    ///   [execution context](scioxide_core#execution-contexts).
    /// - `SizeOverflow` or `AllocationFailure` if the plan cannot be allocated.
    pub fn build<'s, 'c>(
        factor: SparseQr<S>,
        threshold: RankThreshold<S::Real>,
        options: SparseSolveOptions,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<Self, SciOxideError> {
        let p = &factor.symbolic;
        let (m, n) = (p.rows(), p.cols());
        let k = m.min(n);
        let limits = p.limits;
        p.workspace_requirements::<S>(())?.admit()?;
        let mut scale = S::Real::ZERO;
        for col in 0..n {
            let mut norm = ScaledNorm::new();
            for pos in p.col_ptr[col]..p.col_ptr[col + 1] {
                norm.add_magnitude(factor.data.source[p.positions[pos]].abs())?;
            }
            scale = scale.max(norm.finish()?);
        }
        let cutoff = threshold.threshold(scale)?;
        let mut compression = Data::new(k, n, 0, limits.max_factor_entries)?;
        for (i, col) in compression.order.iter_mut().enumerate() {
            *col = i;
        }
        let mut kernel = Kernel::new(k, n, limits.max_active_entries)?;
        for e in &factor.data.r {
            kernel.insert(e.row, e.col, e.value)?;
        }
        let (rank, secondary) = Epoch::new().run(scope, options, |cx| {
            let rank = factor_kernel(
                &mut compression,
                &mut kernel,
                (k, n),
                limits,
                Some(cutoff),
                &mut || admit(cx),
            )?;
            drop(kernel);
            let mut secondary = Data::new(n, rank, 0, limits.max_factor_entries)?;
            for (i, col) in secondary.order.iter_mut().enumerate() {
                *col = i;
            }
            let mut kernel = Kernel::new(n, rank, limits.max_active_entries)?;
            for e in &compression.r {
                kernel.insert(e.col, e.row, e.value.conj())?;
            }
            factor_kernel(
                &mut secondary,
                &mut kernel,
                (n, rank),
                limits,
                None,
                &mut || admit(cx),
            )?;
            for &diagonal in &secondary.diagonal {
                if diagonal == S::ZERO {
                    return Err(SciOxideError::singular(None));
                }
            }
            Ok((rank, secondary))
        })?;
        Ok(Self {
            factor,
            compression,
            secondary,
            rank,
            cutoff,
        })
    }
    /// Returns the QR factor that the plan was built from.
    pub const fn factor(&self) -> &SparseQr<S> {
        &self.factor
    }
    /// Returns the QR factor and discards the plan. After you refactor it,
    /// build a new plan.
    pub fn into_factor(self) -> SparseQr<S> {
        self.factor
    }
    /// Returns the rank `r` decided by [`build`](Self::build).
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Returns the absolute rank cutoff `τ`.
    pub const fn threshold(&self) -> S::Real {
        self.cutoff
    }
    /// Returns the scratch size of solves with `rhs` right-hand sides:
    /// `6 · max(m, n)` scalars each.
    ///
    /// Owned scratch counts toward the byte limit of the analysis, together
    /// with the factor and the plan. A borrowed slice is checked only for its
    /// length.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        rhs: usize,
    ) -> Result<SparseSolveRequirements, SciOxideError> {
        let symbolic = &self.factor.symbolic;
        SparseSolveRequirements::new::<S>(
            self.stride()?,
            rhs,
            symbolic.workspace_requirements::<S>(())?.prepared_bytes,
            symbolic.limits.max_workspace_bytes,
        )
    }
    /// Creates a workspace that owns scratch for `rhs` right-hand sides.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the scratch exceeds the byte limit of the analysis.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn workspace(&self, rhs: usize) -> Result<SparseSolveWorkspace<'static, S>, SciOxideError> {
        let mut workspace = SparseSolveWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(rhs)?)?;
        Ok(workspace)
    }
    // Longer matrix extent; each staged column holds a candidate and four work vectors.
    fn size(&self) -> usize {
        self.factor.symbolic.rows().max(self.factor.symbolic.cols())
    }
    fn stride(&self) -> Result<usize, SciOxideError> {
        mul(
            6,
            self.size(),
            scioxide_core::quantity!("qr_solve_workspace"),
        )
    }
    fn dimensions(&self, operation: SparseOperation) -> (usize, usize) {
        let p = &self.factor.symbolic;
        if operation == SparseOperation::Normal {
            (p.rows(), p.cols())
        } else {
            (p.cols(), p.rows())
        }
    }
    /// Computes the minimum-norm least-squares solution of `op(A) x ≈ b` for
    /// `b = rhs`, writes `x` to `output`, and returns the rank and residuals.
    ///
    /// For `op(A) = A`, `rhs` has length `m` and `output` length `n`; for `Aᵀ`
    /// and `Aᴴ` the lengths swap. The solve counts as one iteration toward the
    /// limits of `options` and `scope` (see [Execution](crate#execution)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `output` has the wrong length, or the
    ///   scratch in `workspace` is too short.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `NumericalInstability` if a substitution overflows or a residual norm
    ///   is not representable.
    /// - `ResourceLimit` if owned scratch exceeds the byte limit.
    /// - `ConvergenceFailure` if the iteration limit has been reached.
    /// - `ExecutionStopped` or `CapabilityUnavailable` from the
    ///   [execution context](scioxide_core#execution-contexts).
    pub fn solve_into<'s, 'c>(
        &self,
        rhs: &[S],
        operation: SparseOperation,
        options: SparseSolveOptions,
        output: &mut [S],
        workspace: &mut SparseSolveWorkspace<'_, S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<SparseQrSolveReport<S::Real>, SciOxideError> {
        workspace.epoch.reject();
        let (input, solution) = self.dimensions(operation);
        crate::helpers::exact(rhs.len(), input)?;
        crate::helpers::exact(output.len(), solution)?;
        crate::helpers::finite(rhs)?;
        let r = self.workspace_requirements(1)?;
        workspace.require(&r)?;
        let size = self.size();
        let SparseSolveWorkspace { scratch, epoch } = workspace;
        let report = epoch.run(scope, options, |cx| {
            admit(cx)?;
            let (candidate, work) = scratch[..r.scalar_len()].split_at_mut(size);
            let candidate = &mut candidate[..solution];
            self.solve_column(rhs, operation, candidate, work)?;
            let (residual, normal) = work.split_at_mut(size);
            self.factor.residual_report(
                (rhs, candidate),
                self.rank,
                self.cutoff,
                operation,
                ResidualScratch { residual, normal },
            )
        })?;
        output.copy_from_slice(&scratch[..solution]);
        Ok(report)
    }
    // Owned solve with the default options in a fresh root scope.
    fn solve_owned(
        &self,
        rhs: &[S],
        operation: SparseOperation,
    ) -> Result<(Vec<S>, SparseQrSolveReport<S::Real>), SciOxideError> {
        let mut output = try_filled(
            self.dimensions(operation).1,
            S::ZERO,
            scioxide_core::quantity!("solution"),
        )?;
        let report = self.solve_into(
            rhs,
            operation,
            SparseSolveOptions::new(),
            &mut output,
            &mut self.workspace(1)?,
            ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    /// Computes the minimum-norm least-squares solution of `A x ≈ b` for
    /// `b = rhs` and returns it in a new vector, with the rank and residuals.
    ///
    /// This is [`solve_into`](Self::solve_into) with
    /// [`SparseOperation::Normal`] and the default [`SparseSolveOptions`], run
    /// in a fresh scope.
    ///
    /// # Errors
    ///
    /// - The errors of [`solve_into`](Self::solve_into).
    /// - `AllocationFailure` if `x` or the scratch cannot be allocated.
    pub fn solve(
        &self,
        rhs: &[S],
    ) -> Result<(Vec<S>, SparseQrSolveReport<S::Real>), SciOxideError> {
        self.solve_owned(rhs, SparseOperation::Normal)
    }
    /// Computes the minimum-norm least-squares solution of `Aᵀ x ≈ b`
    /// (transpose without conjugation), like [`solve`](Self::solve).
    ///
    /// # Errors
    ///
    /// - The errors of [`solve`](Self::solve).
    pub fn solve_transpose(
        &self,
        rhs: &[S],
    ) -> Result<(Vec<S>, SparseQrSolveReport<S::Real>), SciOxideError> {
        self.solve_owned(rhs, SparseOperation::Transpose)
    }
    /// Computes the minimum-norm least-squares solution of `Aᴴ x ≈ b`
    /// (conjugate transpose), like [`solve`](Self::solve).
    ///
    /// # Errors
    ///
    /// - The errors of [`solve`](Self::solve).
    pub fn solve_adjoint(
        &self,
        rhs: &[S],
    ) -> Result<(Vec<S>, SparseQrSolveReport<S::Real>), SciOxideError> {
        self.solve_owned(rhs, SparseOperation::Adjoint)
    }
    /// Computes minimum-norm least-squares solutions for each column of a
    /// column-major block and writes them to `output`.
    ///
    /// Each column counts as one iteration. Under a parallel execution policy
    /// in `options`, columns are solved in parallel. `output` is written only
    /// after every column is done.
    ///
    /// # Errors
    ///
    /// - The errors of [`solve_into`](Self::solve_into).
    /// - `DimensionMismatch` also if `rhs` and `output` have different numbers
    ///   of columns.
    pub fn solve_block_into<'s, 'c>(
        &self,
        rhs: BlockVectorView<'_, S>,
        operation: SparseOperation,
        options: SparseSolveOptions,
        output: &mut BlockVectorViewMut<'_, S>,
        workspace: &mut SparseSolveWorkspace<'_, S>,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<(), SciOxideError> {
        workspace.epoch.reject();
        let (input, solution) = self.dimensions(operation);
        let count = rhs.right_hand_sides();
        crate::helpers::exact(rhs.rows(), input)?;
        crate::helpers::exact(output.rows(), solution)?;
        crate::helpers::exact(output.right_hand_sides(), count)?;
        crate::helpers::finite(rhs.as_slice())?;
        let r = self.workspace_requirements(count)?;
        workspace.require(&r)?;
        let stride = self.stride()?;
        let SparseSolveWorkspace { scratch, epoch } = workspace;
        epoch.run(scope, options, |cx| {
            for _ in 0..count {
                admit(cx)?;
            }
            crate::execution::stage_columns(
                (solution, count),
                &mut scratch[..r.scalar_len()],
                stride,
                options.execution_policy(),
                options.workers(),
                cx.execution(),
                &|column, candidate, work, _| {
                    self.solve_column(rhs.column(column)?, operation, candidate, work)
                },
            )
        })?;
        crate::execution::publish_columns(output, scratch, stride);
        Ok(())
    }
    fn solve_column(
        &self,
        rhs: &[S],
        operation: SparseOperation,
        out: &mut [S],
        work: &mut [S],
    ) -> Result<(), SciOxideError> {
        let p = &self.factor.symbolic;
        let (m, n) = (p.rows(), p.cols());
        let k = m.min(n);
        let size = m.max(n);
        let (a, rest) = work.split_at_mut(size);
        let (b, rest) = rest.split_at_mut(size);
        let (c, rest) = rest.split_at_mut(size);
        let (d, _) = rest.split_at_mut(size);
        if operation == SparseOperation::Normal {
            a[..m].copy_from_slice(rhs);
            apply_q(&self.factor.data, &mut a[..m], true)?;
            apply_q(&self.compression, &mut a[..k], true)?;
            for i in 0..self.rank {
                let mut sum = a[i];
                for &pos in &self.secondary.r_col_index
                    [self.secondary.r_col_ptr[i]..self.secondary.r_col_ptr[i + 1]]
                {
                    let e = self.secondary.r[pos];
                    if e.row < i {
                        sum = sum - e.value.conj() * c[e.row];
                    }
                }
                c[i] = finite_candidate(sum / self.secondary.diagonal[i].conj())?;
            }
            d[..n].fill(S::ZERO);
            d[..self.rank].copy_from_slice(&c[..self.rank]);
            apply_q(&self.secondary, &mut d[..n], false)?;
            for (j, &old) in self.factor.data.order.iter().enumerate() {
                out[old] = d[j];
            }
        } else {
            for (j, &old) in self.factor.data.order.iter().enumerate() {
                b[j] = if operation == SparseOperation::Transpose {
                    rhs[old].conj()
                } else {
                    rhs[old]
                };
            }
            apply_q(&self.secondary, &mut b[..n], true)?;
            for i in (0..self.rank).rev() {
                let mut sum = b[i];
                for e in &self.secondary.r[self.secondary.r_ptr[i]..self.secondary.r_ptr[i + 1]] {
                    if e.col > i {
                        sum = sum - e.value * c[e.col];
                    }
                }
                c[i] = finite_candidate(sum / self.secondary.diagonal[i])?;
            }
            d[..m].fill(S::ZERO);
            d[..self.rank].copy_from_slice(&c[..self.rank]);
            apply_q(&self.compression, &mut d[..k], false)?;
            apply_q(&self.factor.data, &mut d[..m], false)?;
            for (out, &value) in out.iter_mut().zip(&d[..m]) {
                *out = if operation == SparseOperation::Transpose {
                    value.conj()
                } else {
                    value
                };
            }
        }
        Ok(())
    }
}

/// Returns the dimension-scaled relative cutoff `min(d·ε, 1/2)`, with `d`
/// rounded to the nearest representable value of `R` and `ε` its machine
/// epsilon.
fn dimension_epsilon<R: RealScalar>(dimension: usize) -> Result<R, SciOxideError> {
    Ok((R::try_from_f64_lossy(dimension as f64)? * R::EPSILON).min(R::ONE / (R::ONE + R::ONE)))
}
