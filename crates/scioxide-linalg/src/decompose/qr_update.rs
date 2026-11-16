//! Explicit QR factors updated by plane rotations.
//!
//! Every edit reduces R with adjacent-plane rotations: a bottom-up chase of one
//! vector onto a coordinate, and a top-down sweep of Hessenberg fill. Rotations
//! act on R rows at once and are recorded for one row-wise pass over Q, so an
//! elementary edit costs `O(m² + mn)`. Rank is classified by incremental
//! condition estimation on R (Bischof 1990); singular values decide only when
//! the estimates are inconclusive.
use super::qr::QrRequest;
use super::util::reserve_vec;
use crate::kernel::standard;
use crate::rotation::{Rotation, divide_real};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix, ScratchVec};
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    HouseholderDecomposition, HouseholderOrientation, KernelContext, LinalgExecutionOptions,
    LinalgWorkspace, Matrix, MatrixView, RankPolicy, RankReport, SvdValues,
};
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, InvalidInputReason, NumericScalar, Quantity, RealScalar, SciOxideError,
    SolveContext,
};

/// Options for [`QrUpdateDecomposition`]: rank policy, column pivoting and execution options.
///
/// The rank policy only classifies `R` (see [`QrRankDecision`]). It never truncates the
/// factors. The execution options also bound the singular-value fallback of the rank
/// decision.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_linalg::{LinalgExecutionOptions, QrUpdateOptions, RankPolicy};
/// let execution = LinalgExecutionOptions::new()
///     .with_execution_policy(ExecutionPolicy::Sequential)
///     .with_byte_limit(1 << 20)
///     .with_iteration_budget(Budget::ONE.saturating_add(63));
/// let options = QrUpdateOptions::<f64>::new(RankPolicy::DimensionScaledEpsilon)
///     .with_execution_options(execution);
/// assert_eq!(options.execution_options(), execution);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QrUpdateOptions<R: RealScalar> {
    rank: RankPolicy<R>,
    execution: LinalgExecutionOptions,
    pivoting: crate::HouseholderPivoting,
}
impl<R: RealScalar> Default for QrUpdateOptions<R> {
    /// Returns options that classifies the rank with [`RankPolicy::Exact`] and column pivoting.
    fn default() -> Self {
        Self::new(RankPolicy::Exact)
    }
}

impl<R: RealScalar> QrUpdateOptions<R> {
    /// Returns options with rank policy `rank` and column pivoting.
    pub const fn new(rank: RankPolicy<R>) -> Self {
        Self {
            rank,
            execution: LinalgExecutionOptions::new(),
            pivoting: crate::HouseholderPivoting::Column,
        }
    }
    /// Sets the execution options, which also apply to nested stages.
    pub const fn with_execution_options(mut self, execution: LinalgExecutionOptions) -> Self {
        self.execution = execution;
        self
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> LinalgExecutionOptions {
        self.execution
    }
    /// Returns the rank policy.
    pub const fn rank_policy(self) -> RankPolicy<R> {
        self.rank
    }
    /// Sets the column pivoting of [`QrUpdateDecomposition::factor`] and
    /// [`QrUpdateDecomposition::factor_into`].
    ///
    /// The default is column pivoting. [`QrUpdateDecomposition::from_householder`] keeps
    /// the permutation of the given factor, and edits keep the current permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     HouseholderPivoting, Matrix, QrUpdateDecomposition, QrUpdateOptions, RankPolicy,
    /// };
    /// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 10.0])?;
    /// let options =
    ///     QrUpdateOptions::new(RankPolicy::Exact).with_pivoting(HouseholderPivoting::None);
    /// let factor = QrUpdateDecomposition::factor_with_options(a.view(), &options)?;
    /// assert_eq!(factor.permutation(), &[0, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn with_pivoting(mut self, pivoting: crate::HouseholderPivoting) -> Self {
        self.pivoting = pivoting;
        self
    }
    /// Returns the column pivoting used when factoring a new matrix.
    pub const fn pivoting(self) -> crate::HouseholderPivoting {
        self.pivoting
    }
}
/// Block edit of the matrix `A` of a [`QrUpdateDecomposition`], in the indices of `A`.
///
/// Column indices refer to the columns of `A`, not to the pivoted order of `R`. Each edit
/// is applied as a sequence of elementary edits: one per inserted or deleted row or
/// column, and one per column of `U` in a rank-`k` change.
#[derive(Debug, Clone, Copy)]
pub enum QrModification<'a, S: NumericScalar> {
    /// Inserts the rows of a `p × n` block before row `index` (`index = m` appends).
    InsertRows {
        /// Index before which the block is inserted.
        index: usize,
        /// The new entries.
        values: MatrixView<'a, S>,
    },
    /// Deletes the `count` rows `index..index + count`.
    DeleteRows {
        /// First index to remove.
        index: usize,
        /// Number of consecutive indices to remove; zero leaves the factor unchanged.
        count: usize,
    },
    /// Inserts the columns of an `m × p` block before column `index` (`index = n` appends).
    InsertColumns {
        /// Index before which the block is inserted.
        index: usize,
        /// The new entries.
        values: MatrixView<'a, S>,
    },
    /// Deletes the `count` columns `index..index + count`.
    DeleteColumns {
        /// First index to remove.
        index: usize,
        /// Number of consecutive indices to remove; zero leaves the factor unchanged.
        count: usize,
    },
    /// Replaces `A` by `A + U Vᴴ`, where `U` is `m × k` and `V` is `n × k`.
    RankUpdate {
        /// The `m × k` matrix `U`.
        u: MatrixView<'a, S>,
        /// The `n × k` matrix `V`, used as `Vᴴ`.
        v: MatrixView<'a, S>,
    },
    /// Replaces `A` by `A − U Vᴴ`. The result may lose rank; this is not an error.
    RankDowndate {
        /// The `m × k` matrix `U`.
        u: MatrixView<'a, S>,
        /// The `n × k` matrix `V`, used as `Vᴴ`.
        v: MatrixView<'a, S>,
    },
}
/// Method that decided the numerical rank of `R` in a [`QrRankDecision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrRankSource {
    /// Incremental condition estimation on `R` decided, clearing the threshold by a factor of 16.
    ///
    /// Write `R = [R₁₁ R₁₂; 0 R₂₂]` with `R₁₁` of order `k`. Then `σₖ₊₁ ≤ ‖R₂₂‖_F`, so a small
    /// trailing norm proves that at most `k` singular values exceed the threshold. That at
    /// least `k` exceed it rests on an estimate of `σ_min(R₁₁)`, which is a heuristic, not a
    /// bound. Under [`RankPolicy::Exact`], a nonzero diagonal proves full rank without
    /// estimates, and the scale is the largest entry magnitude of `R`.
    Estimator,
    /// The singular values of `R` decided, because the estimates were inconclusive.
    ///
    /// This happens when a bound lies within a factor of 16 of the threshold, or when a
    /// small leading block is not matched by small trailing rows of `R`.
    SingularValues,
}
/// Numerical rank of `R` under the rank policy, with the method that decided it.
///
/// For [`QrRankSource::Estimator`], [`RankReport::scale`] is a lower estimate of the
/// largest singular value, and [`RankReport::threshold`] is the policy threshold at that
/// scale. The rank holds for every scale up to `‖R‖_F`. For
/// [`QrRankSource::SingularValues`], both come from the singular values.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{
///     Matrix, QrRankSource, QrUpdateDecomposition, QrUpdateOptions, RankPolicy,
/// };
/// let a = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0])?;
/// let options = QrUpdateOptions::new(RankPolicy::DimensionScaledEpsilon);
/// let decision = QrUpdateDecomposition::factor_with_options(a.view(), &options)?.rank_decision();
/// assert_eq!(decision.report().rank(), 2);
/// assert_eq!(decision.source(), QrRankSource::Estimator);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct QrRankDecision<R: RealScalar> {
    report: RankReport<R>,
    source: QrRankSource,
}
impl<R: RealScalar> QrRankDecision<R> {
    /// Returns the rank, threshold and scale.
    pub const fn report(self) -> RankReport<R> {
        self.report
    }
    /// Returns the method that decided the rank.
    pub const fn source(self) -> QrRankSource {
        self.source
    }
}
/// Rank decisions before and after one edit of a [`QrUpdateDecomposition`], under the same policy.
#[derive(Debug, Clone, Copy)]
pub struct QrUpdateReport<R: RealScalar> {
    /// Rank before the edit.
    pub(crate) before: QrRankDecision<R>,
    /// Rank after the edit. No singular modes were discarded.
    pub(crate) after: QrRankDecision<R>,
}

impl<R: RealScalar> QrUpdateReport<R> {
    /// Returns the rank decision before the edit.
    pub const fn before(&self) -> QrRankDecision<R> {
        self.before
    }

    /// Returns the rank decision after the edit; the factors are not truncated.
    pub const fn after(&self) -> QrRankDecision<R> {
        self.after
    }
}

impl<R: RealScalar> QrUpdateReport<R> {
    /// Returns `true` if the numerical rank decreased.
    pub fn rank_lost(&self) -> bool {
        self.after.report.rank() < self.before.report.rank()
    }
}
/// Full QR factorization `A Π = Q R` with explicit factors that can be updated after edits of `A`.
///
/// `A` is `m × n`, `Q` is `m × m` unitary and `R` is `m × n` upper trapezoidal. The
/// permutation `Π` is stored as [`permutation`](Self::permutation): column `j` of `Q R` is
/// column `permutation()[j]` of `A`.
///
/// [`QrModification`] edits insert or delete rows or columns, or add a rank-`k` term
/// `± U Vᴴ`. Each elementary edit reduces `R` back to triangular form with adjacent plane
/// rotations and applies them to `Q`, at a cost of `O(m² + m n)`. A new factorization with
/// explicit `Q` costs `O(m² n)`. Edits keep the column permutation; they do not repivot.
///
/// After every factorization or edit, the numerical rank of `R` is classified under the
/// rank policy of [`QrUpdateOptions`] ([`rank_decision`](Self::rank_decision)). The rank
/// is decided by incremental condition estimation on `R`, in `O(m n)` time. When the
/// estimates are inconclusive, the singular values of `R` decide. The factors are never
/// truncated.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{
///     Matrix, QrModification, QrUpdateDecomposition, QrUpdateOptions, RankPolicy,
/// };
/// let a = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0])?;
/// let options = QrUpdateOptions::new(RankPolicy::DimensionScaledEpsilon);
/// let factor = QrUpdateDecomposition::factor_with_options(a.view(), &options)?;
/// // Deleting the second row leaves [[1, 0], [0, 0]], which has rank one.
/// let edit = QrModification::DeleteRows { index: 1, count: 1 };
/// let (updated, report) = factor.modify_with_options(edit, &options)?;
/// assert_eq!(updated.r_matrix().rows(), 2);
/// assert!(report.rank_lost());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins,
///   2013, §6.5.
/// - C. H. Bischof, "Incremental condition estimation", *SIAM J. Matrix Anal. Appl.*
///   11(2), 1990.
#[derive(Debug, Clone)]
pub struct QrUpdateDecomposition<S: NumericScalar> {
    q: Matrix<S>,
    r: Matrix<S>,
    permutation: Vec<usize>,
    rank: QrRankDecision<S::Real>,
    policy: RankPolicy<S::Real>,
    q_norm_bound: S::Real,
    requirements: Option<(QrUpdateWorkspaceRequest<S::Real>, crate::LinalgRequirements)>,
}
impl<S: NumericScalar> QrUpdateDecomposition<S> {
    /// Returns the bytes needed to store `Q`, `R` and the permutation for an `m × n` matrix.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn factor_storage_bytes(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("qr_factor_bytes");
        let elements = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_mul(rows, rows, q)?,
            scioxide_core::backend::checked_mul(rows, cols, q)?,
            q,
        )?;
        scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_mul(elements, core::mem::size_of::<S>(), q)?,
            scioxide_core::backend::checked_mul(cols, core::mem::size_of::<usize>(), q)?,
            q,
        )
    }
    /// Returns the heap capacity of `Q`, `R` and the permutation in bytes.
    ///
    /// The count includes capacity left from earlier, larger factors.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("qr_factor_bytes");
        scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_add(
                self.q.retained_bytes()?,
                self.r.retained_bytes()?,
                q,
            )?,
            scioxide_core::backend::checked_mul(
                self.permutation.capacity(),
                core::mem::size_of::<usize>(),
                q,
            )?,
            q,
        )
    }

    /// Returns the factorization of the `0 × 0` matrix, with rank zero.
    pub const fn empty() -> Self {
        Self {
            q: Matrix::empty(),
            r: Matrix::empty(),
            permutation: Vec::new(),
            rank: QrRankDecision {
                report: RankReport::from_factor(
                    0,
                    0,
                    <S::Real as AlgebraicScalar>::ZERO,
                    <S::Real as AlgebraicScalar>::ZERO,
                ),
                source: QrRankSource::Estimator,
            },
            policy: RankPolicy::Exact,
            q_norm_bound: S::Real::ZERO,
            requirements: None,
        }
    }
    /// Returns the unitary factor `Q` (`m × m`).
    pub fn q_matrix(&self) -> &Matrix<S> {
        &self.q
    }
    /// Returns the upper trapezoidal factor `R` (`m × n`).
    pub fn r_matrix(&self) -> &Matrix<S> {
        &self.r
    }
    /// Returns the column permutation: column `j` of `Q R` is column `permutation()[j]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    /// Returns the rank decision of the last factorization or edit.
    pub fn rank_decision(&self) -> QrRankDecision<S::Real> {
        self.rank
    }
    /// Computes the QR factorization of `a` and classifies its rank.
    ///
    /// # Errors
    ///
    /// The errors of [`factor_into`](Self::factor_into).
    pub fn factor<'m>(a: impl Into<MatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, &crate::QrUpdateOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the rank policy, the pivoting and
    /// the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        a: impl Into<MatrixView<'m, S>>,
        options: &QrUpdateOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            a,
            &options.with_execution_options(
                options
                    .execution
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Forms explicit factors from a Householder QR factorization, without refactoring `A`.
    ///
    /// # Errors
    ///
    /// The errors of [`from_householder_into`](Self::from_householder_into).
    pub fn from_householder(factor: &HouseholderDecomposition<S>) -> Result<Self, SciOxideError> {
        Self::from_householder_with_options(factor, &crate::QrUpdateOptions::default())
    }

    /// Runs [`from_householder`](Self::from_householder) with `options`, which set the rank policy,
    /// the pivoting and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`from_householder`](Self::from_householder), and limit errors from `options`
    /// as described in [Execution](crate#execution).
    pub fn from_householder_with_options(
        factor: &HouseholderDecomposition<S>,
        options: &QrUpdateOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::from_householder_into(
            factor,
            &options.with_execution_options(
                options
                    .execution
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes the QR factorization of `a` into `output` and classifies its rank.
    ///
    /// The factorization uses Householder QR with the pivoting of the options. Size the
    /// workspace with [`QrUpdateWorkspaceRequest`]. The call counts as one iteration
    /// toward the iteration limit of the options; the singular-value fallback of the rank
    /// decision is counted in a child scope.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry of `a` is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a column norm or an entry of `R`
    ///   overflows.
    /// - The errors of the rank policy.
    /// - `ConvergenceFailure` if the singular-value fallback exceeds the iteration limit.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        options: &QrUpdateOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(&options.execution)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let a = a.into();
        let (m, n) = (a.rows(), a.cols());
        let request = QrUpdateWorkspaceRequest::new(m, n, options).for_kernels(kernels);
        let requirements = output.requirements_for(request)?;
        let a = a.try_finite_dispatched(kernels.dispatch())?;
        workspace.try_reserve_prepared(&requirements)?;
        if (output.r.rows(), output.r.cols()) != (m, n) {
            output.reserve(m, n)?;
        }
        workspace.run_scope(
            context,
            identity(m, n),
            options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                with_scratch(workspace, request, m, n, |scratch| {
                    admit(solve)?;
                    super::qr::factor_explicit(
                        a,
                        options.pivoting,
                        scratch.candidate.q.try_view_mut()?,
                        scratch.candidate.r.try_view_mut()?,
                        Some(&mut scratch.candidate.permutation),
                        scratch.product,
                        crate::kernel::KernelExecution {
                            policy: options.execution.execution_policy(),
                            context: kernels,
                        },
                    )?;
                    let after = rank(
                        &scratch.candidate.r,
                        &mut scratch.estimator,
                        scratch.product,
                        *options,
                        kernels.scoped(solve),
                    )?;
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    publish(output, &scratch.candidate, after, options.rank, None)
                })
            },
        )?;
        output.requirements = Some((request, requirements));
        Ok(())
    }

    /// Forms explicit factors from a Householder QR factorization into `output`.
    ///
    /// `Q` is formed from the Householder reflectors, and `R` and the permutation are
    /// copied. The rank is then classified as in [`factor_into`](Self::factor_into).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `factor` has LQ orientation.
    /// - The errors of the rank policy.
    /// - `ConvergenceFailure` if the singular-value fallback exceeds the iteration limit.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn from_householder_into<'scope, 'capability>(
        factor: &HouseholderDecomposition<S>,
        options: &QrUpdateOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(&options.execution)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        if factor.orientation() != HouseholderOrientation::Qr {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("factor"),
                InvalidInputReason::UnsupportedConfiguration,
                "expected QR orientation",
            ));
        }
        let (m, n) = (factor.rows(), factor.cols());
        let request = QrUpdateWorkspaceRequest::new(m, n, options).for_kernels(kernels);
        let requirements = output.requirements_for(request)?;
        workspace.try_reserve_prepared(&requirements)?;
        if (output.r.rows(), output.r.cols()) != (m, n) {
            output.reserve(m, n)?;
        }
        workspace.run_scope(
            context,
            identity(m, n),
            options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                with_scratch(workspace, request, m, n, |scratch| {
                    admit(solve)?;
                    factor.q_matrix_into(
                        &options.execution,
                        scratch.candidate.q.try_view_mut()?,
                        scratch.product,
                        kernels.scoped(solve),
                    )?;
                    factor.triangular_matrix_into(scratch.candidate.r.try_view_mut()?)?;
                    scratch.candidate.permutation.assign(factor.permutation())?;
                    let after = rank(
                        &scratch.candidate.r,
                        &mut scratch.estimator,
                        scratch.product,
                        *options,
                        kernels.scoped(solve),
                    )?;
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    publish(output, &scratch.candidate, after, options.rank, None)
                })
            },
        )?;
        output.requirements = Some((request, requirements));
        Ok(())
    }

    /// Returns a copy with `edit` applied, and the rank decisions before and after the edit.
    ///
    /// # Errors
    ///
    /// The errors of [`modify_in_place`](Self::modify_in_place).
    pub fn modify(
        &self,
        edit: QrModification<'_, S>,
    ) -> Result<(Self, QrUpdateReport<S::Real>), SciOxideError> {
        self.modify_with_options(edit, &crate::QrUpdateOptions::default())
    }

    /// Runs [`modify`](Self::modify) with `options`, which set the rank policy, the pivoting and
    /// the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`modify`](Self::modify), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn modify_with_options(
        &self,
        edit: QrModification<'_, S>,
        options: &QrUpdateOptions<S::Real>,
    ) -> Result<(Self, QrUpdateReport<S::Real>), SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
        );
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options.execution)?;
        let mut output = Self::empty();
        output.reserve(self.r.rows(), self.r.cols())?;
        output.q.try_copy_from(&self.q)?;
        output.r.try_copy_from(&self.r)?;
        output.permutation.extend_from_slice(&self.permutation);
        output.rank = self.rank;
        output.policy = self.policy;
        output.requirements = self.requirements;
        output.q_norm_bound = self.q_norm_bound;
        let report =
            output.modify_in_place(edit, &options, &mut LinalgWorkspace::new(), context)?;
        Ok((output, report))
    }
    /// Applies `edit` to the factorization and returns the rank decisions before and after.
    ///
    /// Each elementary edit costs `O(m² + m n)` and counts as one iteration toward the
    /// iteration limit of the options. The singular-value fallback of the rank decision is
    /// counted in a child scope (see [`QrRankSource`]). The rank before the edit is reused
    /// when the last decision used the same policy. Size the workspace with a
    /// [`QrUpdateWorkspaceRequest`] that covers the shapes before and after the edit.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if an index range lies outside `A`.
    /// - `DimensionMismatch` if inserted rows do not have `n` columns, inserted columns do
    ///   not have `m` rows, or `U` and `V` are not `m × k` and `n × k`.
    /// - `SizeOverflow` if the new shape overflows `usize`.
    /// - `NonFinite` if an entry of the inserted block, `U` or `V` is NaN or infinite.
    /// - `NumericalInstability` if a rank-`k` change overflows.
    /// - The errors of the rank policy.
    /// - `ConvergenceFailure` if the elementary edits or the singular-value fallback exceed
    ///   the iteration limit.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    #[inline(always)]
    pub fn modify_in_place<'scope, 'capability>(
        &mut self,
        edit: QrModification<'_, S>,
        options: &QrUpdateOptions<S::Real>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<QrUpdateReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(&options.execution)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let (m, n) = (self.r.rows(), self.r.cols());
        let (new_m, new_n, mut count) = dimensions(edit, m, n)?;
        if matches!(
            edit,
            QrModification::RankUpdate { .. } | QrModification::RankDowndate { .. }
        ) && (m == 0 || n == 0)
        {
            count = 0;
        }
        let request =
            QrUpdateWorkspaceRequest::new(m.max(new_m), n.max(new_n), options).for_kernels(kernels);
        let requirements = self.requirements_for(request)?;
        if count != 0 {
            match edit {
                QrModification::InsertRows { values, .. }
                | QrModification::InsertColumns { values, .. } => {
                    let _ = values.try_finite_dispatched(kernels.dispatch())?;
                }
                QrModification::RankUpdate { u, v } | QrModification::RankDowndate { u, v } => {
                    let _ = u.try_finite_dispatched(kernels.dispatch())?;
                    let _ = v.try_finite_dispatched(kernels.dispatch())?;
                }
                _ => {}
            }
        }
        workspace.try_reserve_prepared(&requirements)?;
        if (m, n) != (new_m, new_n) {
            self.reserve(new_m, new_n)?;
        }
        let report = workspace.run_scope(
            context,
            identity(m, n),
            options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                with_scratch(workspace, request, m, n, |scratch| {
                    let before = if self.policy == options.rank {
                        self.rank
                    } else {
                        rank(
                            &self.r,
                            &mut scratch.estimator,
                            scratch.product,
                            *options,
                            kernels.scoped(solve),
                        )?
                    };
                    // The admitted scratch headers already have the factor's
                    // exact shape; no resize or repeated shape planning is needed.
                    scratch
                        .candidate
                        .q
                        .as_mut_slice()
                        .copy_from_slice(self.q.as_slice());
                    scratch
                        .candidate
                        .r
                        .as_mut_slice()
                        .copy_from_slice(self.r.as_slice());
                    scratch
                        .candidate
                        .permutation
                        .copy_from_slice(&self.permutation);
                    let mut q_bound = self.q_norm_bound;
                    for step in 0..count {
                        // Exact full-rank decisions already contain R's component maximum.
                        let r_bound = (step == 0
                            && options.rank == RankPolicy::Exact
                            && before.source == QrRankSource::Estimator
                            && before.report.is_full_rank())
                        .then_some(before.report.scale());
                        admit(solve)?;
                        kernels.execution().check()?;
                        match edit {
                            QrModification::InsertRows { index, values } => {
                                insert_row(scratch, index + step, values, step)?
                            }
                            QrModification::DeleteRows { index, .. } => delete_row(scratch, index)?,
                            QrModification::InsertColumns { index, values } => {
                                insert_column(scratch, index + step, values, step, kernels)?
                            }
                            QrModification::DeleteColumns { index, .. } => {
                                delete_column(scratch, index)?
                            }
                            QrModification::RankUpdate { u, v } => rank_one(
                                scratch,
                                u,
                                v,
                                step,
                                S::ONE,
                                &mut q_bound,
                                r_bound,
                                kernels,
                            )?,
                            QrModification::RankDowndate { u, v } => rank_one(
                                scratch,
                                u,
                                v,
                                step,
                                -S::ONE,
                                &mut q_bound,
                                r_bound,
                                kernels,
                            )?,
                        }
                    }
                    let after = if count == 0 {
                        before
                    } else {
                        rank(
                            &scratch.candidate.r,
                            &mut scratch.estimator,
                            scratch.product,
                            *options,
                            kernels.scoped(solve),
                        )?
                    };
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    let bound = matches!(
                        edit,
                        QrModification::RankUpdate { .. } | QrModification::RankDowndate { .. }
                    )
                    .then_some(q_bound);
                    publish(self, &scratch.candidate, after, options.rank, bound)?;
                    Ok(QrUpdateReport { before, after })
                })
            },
        )?;
        self.requirements = Some((request, requirements));
        Ok(report)
    }

    /// Cache only structural preparation. Capacity, stored limits and live
    /// execution capabilities are checked separately for every call.
    fn requirements_for(
        &self,
        request: QrUpdateWorkspaceRequest<S::Real>,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        if let Some((previous, requirements)) = self.requirements
            && previous.same_storage(request)
        {
            Ok(requirements)
        } else {
            Self::workspace_requirements(&request)
        }
    }

    fn reserve(&mut self, m: usize, n: usize) -> Result<(), SciOxideError> {
        self.q.try_reserve_shape(m, m)?;
        self.r.try_reserve_shape(m, n)?;
        reserve_vec(
            &mut self.permutation,
            n,
            scioxide_core::quantity!("permutation"),
        )
    }
}
impl<S: NumericScalar> Default for QrUpdateDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
/// Workspace request for a [`QrUpdateDecomposition`] of at most `rows × cols`.
///
/// One workspace covers factorization and every edit whose shapes before and after fit
/// these bounds.
#[derive(Debug, Clone, Copy)]
pub struct QrUpdateWorkspaceRequest<R: RealScalar> {
    rows: usize,
    cols: usize,
    options: QrUpdateOptions<R>,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
}
impl<R: RealScalar> QrUpdateWorkspaceRequest<R> {
    fn same_storage(self, other: Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.options.execution == other.options.execution
            && self.workers == other.workers
            && matches!(
                (self.packing, other.packing),
                (
                    crate::workspace::Packing::Native,
                    crate::workspace::Packing::Native
                ) | (
                    crate::workspace::Packing::Provider,
                    crate::workspace::Packing::Provider
                )
            )
    }

    /// Returns a request for factors of at most `rows × cols`.
    pub const fn new(rows: usize, cols: usize, options: &QrUpdateOptions<R>) -> Self {
        Self {
            rows,
            cols,
            options: *options,
            workers: core::num::NonZeroUsize::MAX,
            packing: crate::workspace::Packing::Native,
        }
    }
    /// Sizes the workspace for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: core::num::NonZeroUsize) -> Self {
        self.workers = workers;
        self
    }
    /// Adds packing space for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.packing = crate::workspace::Packing::Provider;
        self
    }
    /// Takes the worker bound and BLAS provider from `context`.
    ///
    /// Product accuracy still comes from the options of the request.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }
    /// Scalar, real and index prefix lengths: candidate Q and R, the projected
    /// vector, recorded rotations, and condition-estimation vectors.
    fn prefix(self) -> Result<[usize; 3], SciOxideError> {
        const QUANTITY: Quantity = scioxide_core::quantity!("qr_update_scratch");
        let (m, n) = (self.rows, self.cols);
        let order = m.min(n);
        let estimated = if order == 0 { 0 } else { n };
        let size = |shape: &[usize]| scioxide_core::backend::checked_product(shape, QUANTITY);
        let total = |counts: &[usize]| {
            counts.iter().try_fold(0usize, |sum, &count| {
                sum.checked_add(count)
                    .ok_or_else(|| SciOxideError::size_overflow(QUANTITY))
            })
        };
        let rotations = size(&[2, m])?;
        Ok([
            total(&[
                size(&[m, m])?,
                size(&[m, n])?,
                m,
                rotations,
                size(&[2, estimated])?,
            ])?,
            total(&[rotations, order, order])?,
            total(&[n, rotations])?,
        ])
    }
}
impl<S: NumericScalar> QrUpdateDecomposition<S> {
    /// Returns the workspace size for factorization and edits within the request bounds.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &QrUpdateWorkspaceRequest<S::Real>,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let options = &request.options.execution;
        let mut requirements = super::qr::requirements::<S>(
            QrRequest::Explicit {
                rows: request.rows,
                cols: request.cols,
            },
            options,
            request.workers,
            request.packing,
        )?;
        // The fallback runs on the leading `min(m, n)` rows of R.
        let mut svd = crate::SvdWorkspaceRequest::new(
            request.rows.min(request.cols),
            request.cols,
            &request.options.execution,
        )
        .with_worker_bound(request.workers);
        if matches!(request.packing, crate::workspace::Packing::Provider) {
            svd = svd.with_provider_packing();
        }
        requirements = requirements.max(&SvdValues::workspace_requirements::<S>(&svd)?);
        // `from_householder` forms Q from the prepared reflectors in the child workspace.
        let mut q = crate::HouseholderWorkspaceRequest::q_matrix(
            request.rows,
            request.cols,
            HouseholderOrientation::Qr,
            options,
        )
        .with_worker_bound(request.workers);
        if matches!(request.packing, crate::workspace::Packing::Provider) {
            q = q.with_provider_packing();
        }
        requirements =
            requirements.max(&HouseholderDecomposition::<S>::workspace_requirements(&q)?);
        let [scalars, reals, indices] = request.prefix()?;
        let grow = |current: usize, extra: usize| {
            current.checked_add(extra).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("qr_update_scratch"))
            })
        };
        *requirements.scalar_mut(ScalarRegion::Direct) =
            grow(requirements.scalar(ScalarRegion::Direct), scalars)?;
        *requirements.real_mut(RealRegion::Direct) =
            grow(requirements.real(RealRegion::Direct), reals)?;
        requirements.indices = grow(requirements.indices, indices)?;
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
    /// Allocates a workspace sized by [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     QrUpdateDecomposition, QrUpdateOptions, QrUpdateWorkspaceRequest, RankPolicy,
    /// };
    /// let options = QrUpdateOptions::new(RankPolicy::Exact);
    /// let request = QrUpdateWorkspaceRequest::new(5, 3, &options);
    /// let workspace = QrUpdateDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &QrUpdateWorkspaceRequest<S::Real>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

struct QrCandidate<'a, S: NumericScalar> {
    q: ScratchMatrix<'a, S>,
    r: ScratchMatrix<'a, S>,
    permutation: ScratchVec<'a, usize>,
}
/// Adjacent-plane rotations `(plane, plane + 1)` awaiting one pass over Q.
struct Rotations<'a, S: NumericScalar> {
    planes: ScratchVec<'a, usize>,
    cosines: ScratchVec<'a, S::Real>,
    sines: ScratchVec<'a, S>,
}
/// Incremental condition estimation state, sized for the declared bounds.
struct Estimator<'a, S: NumericScalar> {
    /// `xᴴ R[..k, :]` for the largest-value vector x.
    largest: &'a mut [S],
    /// `xᴴ R[..k, :]` for the smallest-value vector x.
    smallest: &'a mut [S],
    /// Running minimum of the smallest estimates of the leading blocks.
    minimum: &'a mut [S::Real],
    /// `trail[k] = ‖R[k.., :]‖_F`.
    trail: &'a mut [S::Real],
}

/// Workspace for estimating the numerical rank of an upper trapezoidal factor `R` you maintain.
///
/// [`estimate`](Self::estimate) uses the incremental condition estimation and the
/// factor-of-16 margin of [`QrRankSource::Estimator`]. It returns `None` when the estimates
/// are inconclusive; then use the singular values or a rank-revealing factorization. The
/// workspace is allocated once by [`new`](Self::new).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RankPolicy, TriangularRankWorkspace};
/// let r = Matrix::from_vec(3, 2, vec![2.0_f64, 1.0, 0.0, 3.0, 0.0, 0.0])?;
/// let mut workspace = TriangularRankWorkspace::new(3, 2)?;
/// let report = workspace.estimate(&r, RankPolicy::DimensionScaledEpsilon)?;
/// assert!(report.is_some_and(|report| report.is_full_rank()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct TriangularRankWorkspace<S: NumericScalar> {
    largest: Vec<S>,
    smallest: Vec<S>,
    minimum: Vec<S::Real>,
    trail: Vec<S::Real>,
}
impl<S: NumericScalar> TriangularRankWorkspace<S> {
    /// Returns a workspace for factors of at most `rows × cols`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn new(rows: usize, cols: usize) -> Result<Self, SciOxideError> {
        let q = scioxide_core::quantity!("triangular_rank_scratch");
        Ok(Self {
            largest: scioxide_core::backend::try_filled(cols, S::ZERO, q)?,
            smallest: scioxide_core::backend::try_filled(cols, S::ZERO, q)?,
            minimum: scioxide_core::backend::try_filled(rows.min(cols), S::Real::ZERO, q)?,
            trail: scioxide_core::backend::try_filled(rows.min(cols), S::Real::ZERO, q)?,
        })
    }

    /// Estimates the numerical rank of an upper trapezoidal factor `r` under `policy`.
    ///
    /// The shape of `r`, including zero trailing rows, sets the dimension used by
    /// dimension-scaled policies. Returns `None` if the estimates are inconclusive. An
    /// empty factor has rank zero. The cost is `O(m n)`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `r` is larger than the workspace.
    /// - `NonFinite` if an entry of `r` is NaN or infinite.
    /// - `InvalidInput` if `r` has a nonzero entry below the diagonal.
    /// - The errors of `policy` when its threshold is evaluated.
    pub fn estimate(
        &mut self,
        r: &Matrix<S>,
        policy: RankPolicy<S::Real>,
    ) -> Result<Option<RankReport<S::Real>>, SciOxideError> {
        let (m, n) = (r.rows(), r.cols());
        let order = m.min(n);
        if n > self.largest.len() || order > self.minimum.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.minimum.len(), self.largest.len()],
                [order, n],
            ));
        }
        if n != 0 {
            for (i, row) in r.as_slice().chunks_exact(n).enumerate() {
                for &value in row {
                    crate::checked::validate_input(value, "r")?;
                }
                if row[..i.min(n)].iter().any(|&v| v != S::ZERO) {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("r"),
                        InvalidInputReason::Inconsistent,
                        "expected an upper-trapezoidal factor",
                    ));
                }
            }
        }
        if order == 0 {
            return Ok(Some(RankReport::from_factor(
                0,
                0,
                policy.cutoff(S::Real::ZERO, m, n)?,
                S::Real::ZERO,
            )));
        }
        Estimator {
            largest: &mut self.largest,
            smallest: &mut self.smallest,
            minimum: &mut self.minimum,
            trail: &mut self.trail,
        }
        .rank(r, policy)
    }
}
struct QrUpdateScratch<'a, 'storage, S: NumericScalar> {
    candidate: QrCandidate<'a, S>,
    w: ScratchVec<'a, S>,
    rotations: Rotations<'a, S>,
    estimator: Estimator<'a, S>,
    product: &'a mut LinalgWorkspace<'storage, S>,
    policy: scioxide_core::ExecutionPolicy,
}
#[inline(always)]
fn with_scratch<S: NumericScalar, T>(
    workspace: &mut LinalgWorkspace<'_, S>,
    request: QrUpdateWorkspaceRequest<S::Real>,
    rows: usize,
    cols: usize,
    work: impl FnOnce(&mut QrUpdateScratch<'_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.with_prefixes(request.prefix()?, |scalars, reals, indices, child| {
        let (m, n) = (request.rows, request.cols);
        let order = m.min(n);
        let (q, rest) = crate::workspace::split_mut(scalars, m * m)?;
        let (r, rest) = crate::workspace::split_mut(rest, m * n)?;
        let (w, rest) = crate::workspace::split_mut(rest, m)?;
        let (sines, rest) = crate::workspace::split_mut(rest, 2 * m)?;
        let (largest, smallest) = crate::workspace::split_mut(rest, rest.len() / 2)?;
        let (cosines, rest) = crate::workspace::split_mut(reals, 2 * m)?;
        let (minimum, trail) = crate::workspace::split_mut(rest, order)?;
        let (permutation, planes) = crate::workspace::split_mut(indices, n)?;
        let mut scratch = QrUpdateScratch {
            candidate: QrCandidate {
                q: ScratchMatrix::new(q, rows, rows)?,
                r: ScratchMatrix::new(r, rows, cols)?,
                permutation: ScratchVec::new(permutation, cols)?,
            },
            w: ScratchVec::new(w, rows)?,
            rotations: Rotations {
                planes: ScratchVec::new(planes, 0)?,
                cosines: ScratchVec::new(cosines, 0)?,
                sines: ScratchVec::new(sines, 0)?,
            },
            estimator: Estimator {
                largest,
                smallest,
                minimum,
                trail,
            },
            product: child,
            policy: request.options.execution.execution_policy(),
        };
        work(&mut scratch)
    })
}

/// Estimates must clear the rank threshold by `2^MARGIN` to decide.
const MARGIN: i32 = 4;

#[inline]
fn rank<S: NumericScalar>(
    r: &impl MatrixStorage<S>,
    estimator: &mut Estimator<'_, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    options: QrUpdateOptions<S::Real>,
    context: crate::LinalgExecutionContext<'_, '_>,
) -> Result<QrRankDecision<S::Real>, SciOxideError> {
    if let Some(report) = estimator.rank(r, options.rank)? {
        return Ok(QrRankDecision {
            report,
            source: QrRankSource::Estimator,
        });
    }
    // Rows from `min(m, n)` on are zero, so the leading rows hold the spectrum.
    let (m, n) = (r.rows(), r.cols());
    let report = super::svd::values_candidate(
        crate::FiniteMatrixView::from_workspace(r.try_view()?.block(0, 0, m.min(n), n)?),
        options.execution,
        workspace,
        context,
        |values| super::svd::spectrum_rank(values, m, n, options.rank),
    )?;
    Ok(QrRankDecision {
        report,
        source: QrRankSource::SingularValues,
    })
}

impl<S: NumericScalar> Estimator<'_, S> {
    /// Rank of an upper-trapezoidal R, or `None` when the estimates are
    /// inconclusive. With `R = [R11 R12; 0 R22]` and `R11` k-by-k,
    /// `σ_{k+1} ≤ ‖R22‖_F` and `σ_k ≥ σ_min(R11)`. Rank k is accepted when
    /// `‖R22‖_F` is below the threshold at the lower scale bound and the
    /// smallest-value estimate of `R11` is above it at `‖R‖_F`, both by the
    /// margin. Norms are unscaled, so `Exact` sees every stored nonzero.
    #[inline]
    fn rank(
        &mut self,
        r: &impl MatrixStorage<S>,
        policy: RankPolicy<S::Real>,
    ) -> Result<Option<RankReport<S::Real>>, SciOxideError> {
        let (m, n) = (r.rows(), r.cols());
        let order = m.min(n);
        let data = r.as_slice();
        // An upper-trapezoidal factor with a nonzero leading diagonal has
        // structural full rank. Exact policy needs no condition estimate, even
        // when its smallest pivot is subnormal relative to the largest entry.
        if policy == RankPolicy::Exact && (0..order).all(|i| data[i * n + i] != S::ZERO) {
            let scale = data
                .iter()
                .copied()
                .map(standard::magnitude)
                .fold(S::Real::ZERO, S::Real::max);
            return Ok(Some(RankReport::from_factor(
                order,
                order,
                S::Real::ZERO,
                scale,
            )));
        }
        let mut below = S::Real::ZERO;
        let mut widest = S::Real::ZERO;
        for k in (0..order).rev() {
            let row = crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(
                &data[k * n + k..(k + 1) * n],
            ));
            let Ok(norm) = standard::norm_finite(row) else {
                return Ok(None);
            };
            widest = widest.max(norm);
            below = below.hypot(norm);
            self.trail[k] = below;
        }
        if !below.is_finite() {
            return Ok(None);
        }
        let exponent = standard::exponent(below)?;
        let largest = if below == S::Real::ZERO {
            S::Real::ZERO
        } else {
            self.condition(data, n, order, exponent)
                .scalbn(exponent)
                .min(below)
        };
        let scale = largest.max(widest);
        let threshold = policy.cutoff(scale, m, n)?;
        let Ok(ceiling) = policy.cutoff(below, m, n) else {
            return Ok(None);
        };
        let tolerance = threshold.scalbn(-MARGIN);
        let rank = (0..order)
            .find(|&k| self.trail[k] <= tolerance)
            .unwrap_or(order);
        let conclusive = rank == 0 || self.minimum[rank - 1].scalbn(exponent - MARGIN) > ceiling;
        Ok(conclusive.then(|| RankReport::from_factor(rank, order, threshold, scale)))
    }

    /// Incremental condition estimation over the leading blocks of R scaled
    /// by `2^-exponent`. Tracking `z = xᴴ R[..k, :]` instead of x gives
    /// `xᴴ R[..k, k] = z[k]` and streams R by rows. Fills `minimum[..order]`
    /// and returns the largest-value estimate of the leading triangle.
    fn condition(&mut self, data: &[S], n: usize, order: usize, exponent: i32) -> S::Real {
        let entry = |i: usize, j: usize| standard::scale(data[i * n + j], -exponent);
        let (largest, smallest) = (&mut self.largest[..n], &mut self.smallest[..n]);
        for (j, (high, low)) in largest.iter_mut().zip(smallest.iter_mut()).enumerate() {
            *high = entry(0, j);
            *low = *high;
        }
        let mut high = entry(0, 0).abs();
        let mut low = high;
        self.minimum[0] = low;
        for k in 1..order {
            let gamma = entry(k, k);
            let (estimate, s_high, c_high) =
                condition_step(Extreme::Largest, high, largest[k], gamma);
            high = estimate;
            let (estimate, s_low, c_low) =
                condition_step(Extreme::Smallest, low, smallest[k], gamma);
            low = estimate;
            let (s_high, c_high, s_low, c_low) =
                (s_high.conj(), c_high.conj(), s_low.conj(), c_low.conj());
            for j in k + 1..n {
                let value = entry(k, j);
                largest[j] = s_high * largest[j] + c_high * value;
                smallest[j] = s_low * smallest[j] + c_low * value;
            }
            self.minimum[k] = self.minimum[k - 1].min(low);
        }
        high
    }
}

#[derive(Clone, Copy)]
enum Extreme {
    Largest,
    Smallest,
}

/// One incremental condition estimation step. For unit x with
/// `‖xᴴT‖ = estimate` and `alpha = xᴴt`, the extremes of
/// `‖[s x; c]ᴴ [T t; 0 gamma]‖` over unit `(s, c)` are the square roots of the
/// eigenvalues of `diag(estimate², 0) + a aᴴ` with `a = [alpha; gamma]`, and
/// `(s, c)` is the eigenvector. Their product is `estimate |gamma|`, so the
/// smallest follows from the largest without cancellation.
fn condition_step<S: NumericScalar>(
    extreme: Extreme,
    estimate: S::Real,
    alpha: S,
    gamma: S,
) -> (S::Real, S, S) {
    let scale = estimate.max(alpha.abs()).max(gamma.abs());
    if scale == S::Real::ZERO {
        return (S::Real::ZERO, S::ONE, S::ZERO);
    }
    let sigma = estimate / scale;
    let a = divide_real(alpha, scale);
    let g = divide_real(gamma, scale);
    let p = sigma * sigma + a.norm_sqr();
    let r = g.norm_sqr();
    let q = a * g.conj();
    let half = (p - r) * S::Real::HALF;
    let radius = half.hypot(q.abs());
    // One entry of the normalized problem is 1, so the dominant value is >= 1.
    let dominant = ((p + r) * S::Real::HALF + radius).sqrt();
    let (v0, v1) = if half >= S::Real::ZERO {
        (S::from_real(half + radius), q.conj())
    } else {
        (q, S::from_real(radius - half))
    };
    let length = v0.abs().hypot(v1.abs());
    let (s, c) = if length == S::Real::ZERO {
        (S::ONE, S::ZERO)
    } else {
        (divide_real(v0, length), divide_real(v1, length))
    };
    match extreme {
        Extreme::Largest => (scale * dominant, s, c),
        Extreme::Smallest => (estimate * (g.abs() / dominant), -c.conj(), s.conj()),
    }
}

fn identity(rows: usize, cols: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "qr_update", 1),
        ((rows as u128) << 64) | cols as u128,
        0,
    )
}
fn admit(solve: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    crate::solve_support::admit_iteration(
        None,
        Some(solve),
        "no complete orthogonal-factor candidate",
    )
}
fn publish<S: NumericScalar>(
    output: &mut QrUpdateDecomposition<S>,
    candidate: &QrCandidate<'_, S>,
    rank: QrRankDecision<S::Real>,
    policy: RankPolicy<S::Real>,
    q_bound: Option<S::Real>,
) -> Result<(), SciOxideError> {
    let q_bound = match q_bound {
        Some(bound) => bound,
        None => unitary_bound(&candidate.q)?,
    };
    let (m, n) = (candidate.r.rows(), candidate.r.cols());
    output.q.try_resize_flat(m, m)?;
    output.r.try_resize_flat(m, n)?;
    output
        .q
        .as_mut_slice()
        .copy_from_slice(candidate.q.as_slice());
    output
        .r
        .as_mut_slice()
        .copy_from_slice(candidate.r.as_slice());
    output.permutation.clear();
    output.permutation.extend_from_slice(&candidate.permutation);
    output.rank = rank;
    output.policy = policy;
    output.q_norm_bound = q_bound;
    Ok(())
}
fn dimensions<S: NumericScalar>(
    edit: QrModification<'_, S>,
    m: usize,
    n: usize,
) -> Result<(usize, usize, usize), SciOxideError> {
    const ROWS: Quantity = scioxide_core::quantity!("factor_rows");
    const COLUMNS: Quantity = scioxide_core::quantity!("factor_columns");
    let insertion = |index: usize,
                     extent: usize,
                     count: usize,
                     quantity: Quantity|
     -> Result<usize, SciOxideError> {
        if index > extent {
            return Err(SciOxideError::out_of_bounds(
                quantity,
                index,
                extent.saturating_add(1),
            ));
        }
        extent
            .checked_add(count)
            .ok_or_else(|| SciOxideError::size_overflow(quantity))
    };
    let deletion = |index: usize,
                    extent: usize,
                    count: usize,
                    quantity: Quantity|
     -> Result<usize, SciOxideError> {
        if index > extent || count > extent - index {
            return Err(SciOxideError::out_of_bounds(
                quantity,
                index.saturating_add(count),
                extent.saturating_add(1),
            ));
        }
        Ok(extent - count)
    };
    match edit {
        QrModification::InsertRows { index, values } => {
            if values.cols() != n {
                return Err(SciOxideError::dimension_mismatch(
                    [values.rows(), n],
                    [values.rows(), values.cols()],
                ));
            }
            Ok((insertion(index, m, values.rows(), ROWS)?, n, values.rows()))
        }
        QrModification::InsertColumns { index, values } => {
            if values.rows() != m {
                return Err(SciOxideError::dimension_mismatch(
                    [m, values.cols()],
                    [values.rows(), values.cols()],
                ));
            }
            Ok((
                m,
                insertion(index, n, values.cols(), COLUMNS)?,
                values.cols(),
            ))
        }
        QrModification::DeleteRows { index, count } => {
            Ok((deletion(index, m, count, ROWS)?, n, count))
        }
        QrModification::DeleteColumns { index, count } => {
            Ok((m, deletion(index, n, count, COLUMNS)?, count))
        }
        QrModification::RankUpdate { u, v } | QrModification::RankDowndate { u, v } => {
            if u.rows() != m {
                return Err(SciOxideError::dimension_mismatch(
                    [m, u.cols()],
                    [u.rows(), u.cols()],
                ));
            }
            if v.rows() != n || v.cols() != u.cols() {
                return Err(SciOxideError::dimension_mismatch(
                    [n, u.cols()],
                    [v.rows(), v.cols()],
                ));
            }
            Ok((m, n, u.cols()))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rank_one<S: NumericScalar>(
    work: &mut QrUpdateScratch<'_, '_, S>,
    u: MatrixView<'_, S>,
    v: MatrixView<'_, S>,
    col: usize,
    sign: S,
    q_bound: &mut S::Real,
    r_bound: Option<S::Real>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // `A P +- u (P^T v)^H = Q (R +- w (P^T v)^H)` with `w = Q^H u`. Rotating w
    // to `||w|| e_0` from the bottom up turns R upper Hessenberg; the
    // rank-one term then only touches row 0, and the sweep restores the
    // triangle. v is indexed through the column permutation because R's
    // columns are those of `A P`.
    let QrUpdateScratch {
        candidate: f,
        w,
        rotations,
        policy,
        ..
    } = work;
    if f.r.rows() == 0 {
        return Ok(());
    }
    project(&f.q, u, col, w, Some(*q_bound), *policy, context)?;
    chase(w, 0, rotations)?;
    let maximum = match r_bound {
        Some(bound) => bound,
        None => standard::finite_slice_maximum(f.r.as_slice())?,
    };
    let plain = rotation_headroom::<S>(f.r.rows(), maximum);
    apply_chase(&mut f.r, rotations, plain)?;
    let w = sign * w[0];
    let mut row_bound = S::Real::ZERO;
    for j in 0..f.r.cols() {
        let previous = f.r.get_element(0, j);
        let update = v.get(f.permutation[j], col)?.conj();
        let product = w * update;
        let value = previous + product;
        if value.has_finite_components()
            && !scioxide_core::backend::standard::product_needs_retry(w, update, product)
        {
            row_bound = row_bound.max(standard::magnitude(value));
            f.r.set_element(0, j, value);
            continue;
        }
        let value = crate::checked::finite_candidate(
            standard::Affine::new(
                S::ONE,
                standard::magnitude(previous),
                w,
                standard::magnitude(update),
            )?
            .apply(previous, update),
        )?;
        row_bound = row_bound.max(standard::magnitude(value));
        f.r.set_element(0, j, value);
    }
    // The chase preserves each column norm within the headroom envelope.
    // 4m times the old component maximum bounds every unchanged entry; the
    // newly modified row is measured while it is written.
    let rows = S::Real::try_from_usize_exact(f.r.rows()).unwrap_or(S::Real::MAX);
    let grown = maximum * (rows * S::Real::TWO * S::Real::TWO);
    let plain = rotation_headroom::<S>(f.r.rows(), grown.max(row_bound));
    sweep(&mut f.r, 0, rotations, plain)?;
    let count = rotations.planes.len();
    rotations.flush(&mut f.q);
    *q_bound = transported_unitary_bound(&f.q, *q_bound, count)?;
    Ok(())
}
fn insert_row<S: NumericScalar>(
    work: &mut QrUpdateScratch<'_, '_, S>,
    index: usize,
    values: MatrixView<'_, S>,
    row: usize,
) -> Result<(), SciOxideError> {
    // The new row goes on top of R, which makes R upper Hessenberg, and Q
    // becomes `[0 Q]` with `e_0` inserted as row `index`, so A's row order is
    // kept. Both are laid out again in place, last row first.
    let QrUpdateScratch {
        candidate: f,
        rotations,
        ..
    } = work;
    let (m, n) = (f.r.rows(), f.r.cols());
    f.r.try_resize_flat(m + 1, n)?;
    let r = f.r.as_mut_slice();
    r.copy_within(..m * n, n);
    for (slot, &original) in r[..n].iter_mut().zip(f.permutation.iter()) {
        *slot = values.get(row, original)?;
    }
    f.q.try_resize_flat(m + 1, m + 1)?;
    let q = f.q.as_mut_slice();
    for target in (0..=m).rev() {
        let start = target * (m + 1);
        if target == index {
            q[start..start + m + 1].fill(S::ZERO);
            q[start] = S::ONE;
        } else {
            let source = if target < index { target } else { target - 1 };
            q.copy_within(source * m..(source + 1) * m, start + 1);
            q[start] = S::ZERO;
        }
    }
    sweep(&mut f.r, 0, rotations, false)?;
    rotations.flush(&mut f.q);
    Ok(())
}
fn delete_row<S: NumericScalar>(
    work: &mut QrUpdateScratch<'_, '_, S>,
    index: usize,
) -> Result<(), SciOxideError> {
    // Chasing `conj(Q[index, :])` onto `e_0` leaves Q's column 0 zero outside
    // row `index` and R upper Hessenberg. Dropping that row and column of Q
    // and R's first row (A's deleted row) leaves a triangular factor.
    let QrUpdateScratch {
        candidate: f,
        w,
        rotations,
        ..
    } = work;
    let (m, n) = (f.r.rows(), f.r.cols());
    w.resize(m, S::ZERO)?;
    for (j, x) in w.iter_mut().enumerate() {
        *x = f.q.get_element(index, j).conj();
    }
    chase(w, 0, rotations)?;
    apply_chase(&mut f.r, rotations, false)?;
    rotations.flush(&mut f.q);
    let q = f.q.as_mut_slice();
    for (target, source) in (0..m).filter(|&i| i != index).enumerate() {
        q.copy_within(source * m + 1..(source + 1) * m, target * (m - 1));
    }
    f.q.try_resize_flat(m - 1, m - 1)?;
    f.r.as_mut_slice().copy_within(n.., 0);
    f.r.try_resize_flat(m - 1, n)
}
fn insert_column<S: NumericScalar>(
    work: &mut QrUpdateScratch<'_, '_, S>,
    index: usize,
    values: MatrixView<'_, S>,
    col: usize,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // A new column `a` becomes the last column of R as `Q^H a`, chased onto
    // its diagonal; the permutation records its original position, so no
    // existing R column has to move.
    let QrUpdateScratch {
        candidate: f,
        w,
        rotations,
        policy,
        ..
    } = work;
    let (m, n) = (f.r.rows(), f.r.cols());
    project(&f.q, values, col, w, None, *policy, context)?;
    chase(w, n, rotations)?;
    f.r.try_resize_flat(m, n + 1)?;
    let r = f.r.as_mut_slice();
    for i in (0..m).rev() {
        r.copy_within(i * n..(i + 1) * n, i * (n + 1));
        r[i * (n + 1) + n] = w[i];
    }
    f.permutation
        .iter_mut()
        .filter(|p| **p >= index)
        .for_each(|p| *p += 1);
    f.permutation.push(index)?;
    rotations.flush(&mut f.q);
    Ok(())
}
fn delete_column<S: NumericScalar>(
    work: &mut QrUpdateScratch<'_, '_, S>,
    index: usize,
) -> Result<(), SciOxideError> {
    // Removing a column leaves R upper Hessenberg from that position on.
    let QrUpdateScratch {
        candidate: f,
        rotations,
        ..
    } = work;
    let (m, n) = (f.r.rows(), f.r.cols());
    let position = f
        .permutation
        .iter()
        .position(|&p| p == index)
        .ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "missing column permutation",
            )
        })?;
    let r = f.r.as_mut_slice();
    for i in 0..m {
        let (source, target) = (i * n, i * (n - 1));
        r.copy_within(source..source + position, target);
        r.copy_within(source + position + 1..source + n, target + position);
    }
    f.r.try_resize_flat(m, n - 1)?;
    f.permutation.remove(position);
    f.permutation
        .iter_mut()
        .filter(|p| **p > index)
        .for_each(|p| *p -= 1);
    sweep(&mut f.r, position, rotations, false)?;
    rotations.flush(&mut f.q);
    Ok(())
}

/// An upper row-norm bound also bounds every component. A binary factor
/// >= 2m covers real and complex row norms without rounding down the bound.
fn unitary_bound<S: NumericScalar>(q: &impl MatrixStorage<S>) -> Result<S::Real, SciOxideError> {
    let rows = scioxide_core::backend::checked_next_power_of_two(
        q.rows().max(1),
        scioxide_core::quantity!("qr_rows"),
    )?;
    let factor = S::Real::try_from_usize_exact(rows)? * S::Real::TWO;
    let bound = standard::finite_slice_maximum(q.as_slice())? * factor;
    if bound.is_finite() {
        Ok(bound)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "unitary norm bound is not finite",
        ))
    }
}

/// Plane rotations preserve row norms up to coefficient and application
/// rounding. The loose 64k*epsilon envelope covers both plus bound rounding
/// for k*epsilon <= 1/1024. Exceptional accumulated bounds are recomputed.
fn transported_unitary_bound<S: NumericScalar>(
    q: &impl MatrixStorage<S>,
    previous: S::Real,
    count: usize,
) -> Result<S::Real, SciOxideError> {
    if count == 0 {
        return Ok(previous);
    }
    if let Ok(count) = S::Real::try_from_usize_exact(count) {
        let sixty_four =
            S::Real::TWO * S::Real::TWO * S::Real::TWO * S::Real::TWO * S::Real::TWO * S::Real::TWO;
        let error = count * S::Real::EPSILON;
        let next = previous * (S::Real::ONE + sixty_four * error);
        if error
            <= S::Real::ONE
                / (sixty_four * S::Real::TWO * S::Real::TWO * S::Real::TWO * S::Real::TWO)
            && next.is_finite()
        {
            return Ok(next);
        }
    }
    unitary_bound(q)
}

fn project<S: NumericScalar>(
    q: &impl MatrixStorage<S>,
    values: MatrixView<'_, S>,
    col: usize,
    w: &mut ScratchVec<'_, S>,
    q_bound: Option<S::Real>,
    policy: scioxide_core::ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    w.resize(q.cols(), S::ZERO)?;
    let input = crate::FiniteVectorView::from_workspace(values.col(col)?);
    // Reuse the transported bound only to admit the ordinary frame. Range
    // replay retains the exact component maximum and its original reduction
    // order, so a loose bound never changes exceptional-range normalization.
    let ordinary_bound = q_bound.filter(|&bound| {
        input.maximum >= S::Real::MIN_POSITIVE.sqrt()
            && input.maximum <= S::Real::MAX.sqrt()
            && standard::product_scales(bound, input.maximum, q.cols())
                .is_ok_and(|frame| frame == (0, 0))
    });
    crate::kernel::level2::gemv_bounded(
        match ordinary_bound {
            Some(bound) => crate::FiniteMatrixView::from_validated(q.try_view()?.adjoint(), bound),
            None => crate::FiniteMatrixView::from_workspace(q.try_view()?.adjoint()),
        },
        input,
        w,
        policy,
        context.with_accuracy(scioxide_core::Accuracy::Standard),
    )
}

/// Rotate `w[lo..]` onto `w[lo]` from the bottom up, recording each plane.
fn chase<S: NumericScalar>(
    w: &mut [S],
    lo: usize,
    rotations: &mut Rotations<'_, S>,
) -> Result<(), SciOxideError> {
    for j in (lo + 1..w.len()).rev() {
        if w[j] == S::ZERO {
            continue;
        }
        let (rotation, r) = Rotation::generate(w[j - 1], w[j])?;
        w[j - 1] = r;
        w[j] = S::ZERO;
        rotations.push(j - 1, rotation)?;
    }
    Ok(())
}
/// A column's Euclidean norm is at most sqrt(2m) times its largest
/// component. At most m rotations change it by a modest rounding factor when
/// m*epsilon <= 1/64. MAX/(16m) therefore leaves headroom for every real or
/// complex rotation's component products and sums; exceptional columns keep
/// per-pair scaling. Recheck after adding the rank-one term.
fn rotation_headroom<S: NumericScalar>(rows: usize, maximum: S::Real) -> bool {
    let Ok(rows) = S::Real::try_from_usize_exact(rows.max(1)) else {
        return false;
    };
    let sixteen = S::Real::TWO * S::Real::TWO * S::Real::TWO * S::Real::TWO;
    let sixty_four = sixteen * S::Real::TWO * S::Real::TWO;
    rows * S::Real::EPSILON <= S::Real::ONE / sixty_four
        && maximum <= S::Real::MAX / (sixteen * rows)
}

/// Apply a recorded bottom-up chase to the triangular R. Rows `plane` and
/// `plane + 1` start at column `plane`, so each rotation adds one subdiagonal
/// entry; planes at or below R's last column touch only zero rows.
fn apply_chase<S: NumericScalar>(
    r: &mut ScratchMatrix<'_, S>,
    rotations: &Rotations<'_, S>,
    plain: bool,
) -> Result<(), SciOxideError> {
    for (plane, rotation) in rotations.iter() {
        if plane < r.cols() {
            let n = r.cols();
            if plain {
                rotation.rotate_rows(r.as_mut_slice(), n, plane, plane + 1, plane..n)?;
            } else {
                rotation.apply(r.try_view_mut()?, plane, plane + 1, plane..n)?;
            }
        }
    }
    Ok(())
}
/// Zero the subdiagonal of a Hessenberg R from column `from` on, top-down.
fn sweep<S: NumericScalar>(
    r: &mut ScratchMatrix<'_, S>,
    from: usize,
    rotations: &mut Rotations<'_, S>,
    plain: bool,
) -> Result<(), SciOxideError> {
    let (m, n) = (r.rows(), r.cols());
    for k in from..n.min(m.saturating_sub(1)) {
        let below = r.get_element(k + 1, k);
        if below == S::ZERO {
            continue;
        }
        let (rotation, diagonal) = Rotation::generate(r.get_element(k, k), below)?;
        r.set_element(k, k, diagonal);
        r.set_element(k + 1, k, S::ZERO);
        if plain {
            rotation.rotate_rows(r.as_mut_slice(), n, k, k + 1, k + 1..n)?;
        } else {
            rotation.apply(r.try_view_mut()?, k, k + 1, k + 1..n)?;
        }
        rotations.push(k, rotation)?;
    }
    Ok(())
}
impl<S: NumericScalar> Rotations<'_, S> {
    fn push(&mut self, plane: usize, rotation: Rotation<S>) -> Result<(), SciOxideError> {
        self.planes.push(plane)?;
        self.cosines.push(rotation.c)?;
        self.sines.push(rotation.s)
    }
    fn iter(&self) -> impl Iterator<Item = (usize, Rotation<S>)> + '_ {
        self.planes
            .iter()
            .zip(self.cosines.iter().zip(self.sines.iter()))
            .map(|(&plane, (&c, &s))| (plane, Rotation { c, s }))
    }
    /// `Q ← Q G_1^H G_2^H ...` in recording order, eight rows at a time so
    /// that independent rows overlap. Entries of the unitary Q and of the
    /// rotations are bounded by one, so plain arithmetic needs neither range
    /// scaling nor finiteness checks.
    fn flush(&mut self, q: &mut ScratchMatrix<'_, S>) {
        let m = q.cols();
        if m != 0 {
            let mut tiles = q.as_mut_slice().chunks_exact_mut(8 * m);
            for tile in &mut tiles {
                let (r0, tail) = tile.split_at_mut(m);
                let (r1, tail) = tail.split_at_mut(m);
                let (r2, tail) = tail.split_at_mut(m);
                let (r3, tail) = tail.split_at_mut(m);
                let (r4, tail) = tail.split_at_mut(m);
                let (r5, tail) = tail.split_at_mut(m);
                let (r6, r7) = tail.split_at_mut(m);
                // A fixed row count exposes independent rotation chains. Every
                // row still applies the rotations in their recorded order.
                let mut rows = [r0, r1, r2, r3, r4, r5, r6, r7];
                for (plane, rotation) in self.iter() {
                    let (c, s) = (S::from_real(rotation.c), rotation.s);
                    for row in &mut rows {
                        let (x, y) = (row[plane], row[plane + 1]);
                        row[plane] = c * x + s.conj() * y;
                        row[plane + 1] = if S::IS_REAL {
                            c * y + (-s) * x
                        } else {
                            c * y - s * x
                        };
                    }
                }
            }
            for row in tiles.into_remainder().chunks_exact_mut(m) {
                for (plane, rotation) in self.iter() {
                    let (c, s) = (S::from_real(rotation.c), rotation.s);
                    let (x, y) = (row[plane], row[plane + 1]);
                    row[plane] = c * x + s.conj() * y;
                    row[plane + 1] = if S::IS_REAL {
                        c * y + (-s) * x
                    } else {
                        c * y - s * x
                    };
                }
            }
        }
        self.planes.clear();
        self.cosines.clear();
        self.sines.clear();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    #[test]
    fn external_triangular_rank_estimates_validate_and_preserve_dimensions() {
        use crate::{Matrix, RankPolicy, TriangularRankWorkspace};
        let mut w = TriangularRankWorkspace::<f64>::new(1000, 3).unwrap();
        for scale in [1e-200, 1., 1e200] {
            let r = Matrix::from_vec(3, 2, alloc::vec![scale, scale, 0., scale, 0., 0.]).unwrap();
            assert!(
                w.estimate(&r, RankPolicy::DimensionScaledEpsilon)
                    .unwrap()
                    .unwrap()
                    .is_full_rank()
            );
        }
        let mut r = Matrix::zeros(1000, 2).unwrap();
        r.as_mut_slice()[0] = 1.;
        r.as_mut_slice()[3] = 1e-13;
        assert!(
            w.estimate(&r, RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            w.estimate(&r, RankPolicy::Exact).unwrap().unwrap().rank(),
            2
        );
        r.as_mut_slice()[2] = 1.;
        assert!(w.estimate(&r, RankPolicy::Exact).is_err());
        r.as_mut_slice()[2] = f64::NAN;
        assert!(w.estimate(&r, RankPolicy::Exact).is_err());
        assert!(
            w.estimate(&Matrix::zeros(1, 4).unwrap(), RankPolicy::Exact)
                .is_err()
        );
        assert_eq!(
            w.estimate(
                &Matrix::zeros(0, 3).unwrap(),
                RankPolicy::DimensionScaledEpsilon
            )
            .unwrap()
            .unwrap()
            .rank(),
            0
        );
        let zero = Matrix::zeros(2, 3).unwrap();
        assert_eq!(
            w.estimate(&zero, RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .unwrap()
                .rank(),
            0
        );
    }
    use super::*;
    use scioxide_core::Complex;

    fn triangle<S: NumericScalar>(n: usize, grading: f64, seed: usize) -> Matrix<S> {
        let mut t = Matrix::<S>::zeros(n, n).unwrap();
        for i in 0..n {
            for j in i..n {
                let re = ((i * 31 + j * 17 + seed * 7) % 23) as f64 / 11.0 - 1.0;
                let im = ((i * 13 + j * 5 + seed) % 19) as f64 / 9.0 - 1.0;
                let scale = grading.powi(j as i32);
                let value = if S::IS_REAL {
                    S::from_real(S::Real::try_from_f64_lossy(re * scale).unwrap())
                } else {
                    S::from_components(
                        S::Real::try_from_f64_lossy(re * scale).unwrap(),
                        S::Real::try_from_f64_lossy(im * scale).unwrap(),
                    )
                };
                t.view_mut().set(i, j, value).unwrap();
            }
        }
        t
    }

    fn estimate<S: NumericScalar>(
        r: &Matrix<S>,
        policy: RankPolicy<S::Real>,
    ) -> (Option<RankReport<S::Real>>, Vec<S::Real>) {
        let (m, n) = (r.rows(), r.cols());
        let order = m.min(n);
        let mut largest = vec![S::ZERO; n];
        let mut smallest = vec![S::ZERO; n];
        let mut minimum = vec![S::Real::ZERO; order];
        let mut trail = vec![S::Real::ZERO; order];
        let mut estimator = Estimator {
            largest: &mut largest,
            smallest: &mut smallest,
            minimum: &mut minimum,
            trail: &mut trail,
        };
        let report = estimator.rank(r, policy).unwrap();
        (report, minimum)
    }

    fn brackets<S: NumericScalar>() {
        for (grading, seed) in [(1.0, 1), (1.0, 2), (0.25, 3), (4.0, 4)] {
            let n = 9;
            let t = triangle::<S>(n, grading, seed);
            let values = SvdValues::compute(t.view()).unwrap();
            let sigma = values.singular_values();
            let (largest, smallest) = (sigma[0].into_f64(), sigma[n - 1].into_f64());
            let mut z = (vec![S::ZERO; n], vec![S::ZERO; n]);
            let mut minimum = vec![S::Real::ZERO; n];
            let mut trail = vec![S::Real::ZERO; n];
            let mut estimator = Estimator {
                largest: &mut z.0,
                smallest: &mut z.1,
                minimum: &mut minimum,
                trail: &mut trail,
            };
            let high = estimator.condition(t.as_slice(), n, n, 0).into_f64();
            let low = minimum[n - 1].into_f64();
            // Computed singular values are accurate to `eps * sigma_max`.
            let tol = 1e2 * S::Real::EPSILON.into_f64() * largest;
            // Lower bound on the largest, upper bound on the smallest, and
            // within a modest factor on these matrices.
            assert!(
                high <= largest + tol && high >= largest / 4.0,
                "{high} {largest}"
            );
            assert!(
                low >= smallest - tol && low <= 16.0 * smallest + tol,
                "{low} {smallest}"
            );
        }
    }

    #[test]
    fn condition_estimates_bracket_extreme_singular_values() {
        brackets::<f64>();
        brackets::<f32>();
        brackets::<Complex<f64>>();
        brackets::<Complex<f32>>();
    }

    #[test]
    fn trivial_and_exact_factors_decide_without_singular_values() {
        for (m, n) in [(0, 0), (0, 3), (3, 0), (3, 2), (2, 3)] {
            let zero = Matrix::<f64>::zeros(m, n).unwrap();
            for policy in [RankPolicy::Exact, RankPolicy::DimensionScaledEpsilon] {
                let (report, _) = estimate(&zero, policy);
                let report = report.unwrap();
                assert_eq!(report.rank(), 0);
                assert_eq!(report.maximum_rank(), m.min(n));
                assert_eq!((report.scale(), report.threshold()), (0.0, 0.0));
            }
        }
        // Subnormal entries are stored nonzeros under `Exact`.
        let tiny = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1e-310]).unwrap();
        let (report, _) = estimate(&tiny, RankPolicy::Exact);
        assert_eq!(report.unwrap().rank(), 2);
        let (report, _) = estimate(&tiny, RankPolicy::DimensionScaledEpsilon);
        assert_eq!(report.unwrap().rank(), 1);
        // An exactly zero trailing row is excluded; a hidden zero pivot is not
        // decided by the estimator.
        let trailing = Matrix::from_vec(2, 2, vec![2.0, 1.0, 0.0, 0.0]).unwrap();
        let (report, _) = estimate(&trailing, RankPolicy::Exact);
        assert_eq!(report.unwrap().rank(), 1);
        let leading = Matrix::from_vec(2, 2, vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(estimate(&leading, RankPolicy::Exact).0.is_none());
    }

    #[test]
    fn exact_diagonal_proof_survives_unrepresentable_condition_numbers() {
        fn check<S: NumericScalar>() {
            let tiny = S::from_real(S::Real::MIN_SUBNORMAL);
            let huge = S::from_components(S::Real::MAX, S::Real::MAX);
            for (m, n, values) in [
                (2, 2, vec![tiny, huge, S::ZERO, tiny]),
                (3, 2, vec![tiny, huge, S::ZERO, tiny, S::ZERO, S::ZERO]),
                (2, 3, vec![tiny, huge, S::ZERO, S::ZERO, tiny, huge]),
            ] {
                let r = Matrix::from_vec(m, n, values).unwrap();
                let report = estimate(&r, RankPolicy::Exact).0.unwrap();
                assert!(report.is_full_rank());
                assert_eq!(report.rank(), m.min(n));
                assert_eq!(report.threshold(), S::Real::ZERO);
                assert_eq!(report.scale(), S::Real::MAX);
            }
        }
        check::<f32>();
        check::<f64>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }

    #[test]
    fn prepared_requirements_still_admit_each_workspace_and_byte_limit() {
        let matrix = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
        let options = QrUpdateOptions::new(RankPolicy::Exact);
        let mut factor = QrUpdateDecomposition::factor(matrix.view()).unwrap();
        let before = factor.clone();
        let left = [0.1, 0.2];
        let right = [0.2, 0.1];
        let edit = QrModification::RankUpdate {
            u: MatrixView::from_slice(2, 1, &left).unwrap(),
            v: MatrixView::from_slice(2, 1, &right).unwrap(),
        };
        let mut workspace = LinalgWorkspace::new();
        let tight =
            options.with_execution_options(LinalgExecutionOptions::new().with_byte_limit(0));
        assert!(
            factor
                .modify_in_place(
                    edit,
                    &tight,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native()
                )
                .is_err()
        );
        assert_eq!(factor.q_matrix(), before.q_matrix());
        assert_eq!(factor.r_matrix(), before.r_matrix());
        assert_eq!(factor.rank_decision(), before.rank_decision());
        assert_eq!(factor.q_norm_bound, before.q_norm_bound);
        assert_eq!(
            factor.requirements.map(|(_, r)| r),
            before.requirements.map(|(_, r)| r)
        );
        factor
            .modify_in_place(
                edit,
                &options,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(workspace.retained_bytes().unwrap() > 0);
        assert!(factor.rank_decision().report().is_full_rank());
    }

    #[test]
    fn initialization_ordering_is_independent_of_cached_storage() {
        let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 10.0]).unwrap();
        let options = QrUpdateOptions::new(RankPolicy::Exact);
        let mut factor = QrUpdateDecomposition::factor(matrix.view()).unwrap();
        assert_eq!(factor.permutation(), &[1, 0]);
        let options = options.with_pivoting(crate::HouseholderPivoting::None);
        assert_eq!(options.pivoting(), crate::HouseholderPivoting::None);
        let mut workspace =
            QrUpdateDecomposition::<f64>::workspace(&QrUpdateWorkspaceRequest::new(2, 2, &options))
                .unwrap();
        QrUpdateDecomposition::factor_into(
            matrix.view(),
            &options,
            &mut factor,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(factor.permutation(), &[0, 1]);
    }

    #[test]
    fn large_rank_updates_retain_range_safe_rotations() {
        fn check<S: NumericScalar>() {
            let four = S::Real::TWO * S::Real::TWO;
            let scale = S::Real::MAX / (four * S::Real::TWO);
            let diagonal = S::from_real(scale * S::Real::TWO);
            let matrix =
                Matrix::from_vec(2, 2, vec![diagonal, S::ZERO, S::ZERO, diagonal]).unwrap();
            let options = QrUpdateOptions::new(RankPolicy::Exact)
                .with_pivoting(crate::HouseholderPivoting::None);
            let mut factor =
                QrUpdateDecomposition::factor_with_options(matrix.view(), &options).unwrap();
            let left = [
                S::from_components(scale, scale),
                S::from_components(-scale, scale),
            ];
            let right = [S::from_real(S::Real::ONE / four); 2];
            let mut workspace = QrUpdateDecomposition::<S>::workspace(
                &QrUpdateWorkspaceRequest::new(2, 2, &options),
            )
            .unwrap();
            factor
                .modify_in_place(
                    QrModification::RankUpdate {
                        u: MatrixView::from_slice(2, 1, &left).unwrap(),
                        v: MatrixView::from_slice(2, 1, &right).unwrap(),
                    },
                    &options,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            for (i, &u) in left.iter().enumerate() {
                for (j, &v) in right.iter().enumerate() {
                    let actual = (0..2).fold(S::ZERO, |sum, k| {
                        sum + factor.q.as_slice()[i * 2 + k] * factor.r.as_slice()[k * 2 + j]
                    });
                    let expected = matrix.as_slice()[i * 2 + j] + u * v.conj();
                    let error = (actual - expected).abs() / scale;
                    assert!(
                        error <= S::Real::try_from_usize_exact(128).unwrap() * S::Real::EPSILON,
                        "error={error:?}"
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }

    #[test]
    fn transported_projection_bounds_cover_updated_unitary_rows() {
        fn check<S: NumericScalar>() {
            for n in [1, 2, 7, 8, 16] {
                let mut matrix = Matrix::zeros(n, n).unwrap();
                for i in 0..n {
                    matrix.as_mut_slice()[i * n + i] = S::ONE;
                }
                let options = QrUpdateOptions::new(RankPolicy::Exact);
                let mut factor = QrUpdateDecomposition::factor(matrix.view()).unwrap();
                let mut workspace = QrUpdateDecomposition::<S>::workspace(
                    &QrUpdateWorkspaceRequest::new(n, n, &options),
                )
                .unwrap();
                let divisor = S::Real::try_from_usize_exact(512).unwrap();
                let left: Vec<_> = (0..n)
                    .map(|i| {
                        let x = S::Real::try_from_usize_exact(i + 1).unwrap() / divisor;
                        S::from_components(x, -x)
                    })
                    .collect();
                let right = vec![S::from_real(S::Real::ONE / divisor); n];
                for _ in 0..24 {
                    factor
                        .modify_in_place(
                            QrModification::RankUpdate {
                                u: MatrixView::from_slice(n, 1, &left).unwrap(),
                                v: MatrixView::from_slice(n, 1, &right).unwrap(),
                            },
                            &options,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential_native(),
                        )
                        .unwrap();
                    for row in factor.q.as_slice().chunks_exact(n) {
                        let norm = standard::norm_workspace(row.iter().copied()).unwrap();
                        assert!(norm <= factor.q_norm_bound);
                    }
                }
                // An accumulated bound that cannot be transported resets from Q.
                let restored = transported_unitary_bound(&factor.q, S::Real::MAX, 1).unwrap();
                assert!(restored.is_finite());
                assert!(standard::finite_slice_maximum(factor.q.as_slice()).unwrap() <= restored);
            }
        }
        check::<f32>();
        check::<f64>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }

    #[test]
    fn margin_sends_near_threshold_values_to_singular_values() {
        let policy = RankPolicy::Threshold(scioxide_core::RankThreshold::new(0.0, 1e-6).unwrap());
        for (delta, decided) in [(1e-3, Some(3)), (2e-6, None), (5e-7, None), (1e-9, Some(2))] {
            let r = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, delta])
                .unwrap();
            let (report, _) = estimate(&r, policy);
            assert_eq!(report.map(RankReport::rank), decided, "{delta}");
        }
    }
}
