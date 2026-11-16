//! Householder QR and LQ factorizations with optional pivoting.
//!
//! LQ is computed as the QR factorization of the adjoint, so both use the same
//! packed-reflector kernel.

mod trust_region;
use crate::workspace::ScalarRegion;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NormSelection, NumericScalar, RealScalar, ResidualReport,
    SciOxideError,
};

use crate::analysis::{
    ConditionEstimate, DirectSolveOptions, DirectSolveReport, DirectSolveResult,
    DirectSolveTermination, RankPolicy, RankReport,
};
use crate::dense::{Matrix, Vector};

use super::qr::{QrDecomposition, QrRequest};
use crate::{KernelContext, LinalgWorkspace};

/// Whether a [`HouseholderDecomposition`] is a QR or an LQ factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum HouseholderOrientation {
    /// `A[:, p] = Q R` for `m × n` `A`: `Q` is `m × m` unitary, `R` is `m × n`
    /// upper trapezoidal and `p` permutes the columns.
    Qr,
    /// `A[p, :] = L Q` for `m × n` `A`: `L` is `m × n` lower trapezoidal, `Q` is
    /// `n × n` unitary and `p` permutes the rows.
    Lq,
}

/// Pivoting of a Householder QR (columns) or LQ (rows) factorization.
///
/// [`HouseholderPivoting::Column`], the default, orders the diagonal of `R` (or
/// `L`) by decreasing magnitude, and the diagonal usually reveals the numerical
/// rank. This is not guaranteed: on the Kahan matrix the smallest diagonal entry
/// can exceed the smallest singular value by orders of magnitude, so the rank is
/// overestimated. Use the SVD when the rank must be certain. With
/// [`HouseholderPivoting::None`] the permutation is the identity, and the
/// diagonal does not reveal rank.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{
///     HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting, Matrix,
/// };
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let qr = HouseholderDecomposition::factor(
///     &a,
///     HouseholderOrientation::Qr,
///     HouseholderPivoting::None)?;
/// assert_eq!(qr.permutation(), &[0, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use]
pub enum HouseholderPivoting {
    /// Column pivoting: each step takes the column of the trailing matrix with
    /// the largest 2-norm (the row, for LQ).
    ///
    /// The column norms are updated after each step and recomputed when
    /// cancellation makes the update unreliable (Drmač & Bujanović, 2008).
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §5.4.
    /// - Z. Drmač and Z. Bujanović, On the failure of rank-revealing QR
    ///   factorization software – a case study, *ACM Trans. Math. Softw.* 35
    ///   (2008), article 12.
    #[default]
    Column,
    /// No pivoting; the permutation is the identity.
    None,
}

/// Scratch size request for a Householder factorization or one of its operations.
///
/// Pass it to [`HouseholderDecomposition::workspace_requirements`] or
/// [`HouseholderDecomposition::workspace`]. `rows × cols` is the shape of the
/// original matrix `A`.
#[derive(Debug, Clone, Copy)]
pub struct HouseholderWorkspaceRequest {
    rows: usize,
    cols: usize,
    orientation: HouseholderOrientation,
    job: HouseholderJob,
    options: crate::LinalgExecutionOptions,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
    residual: crate::Estimation,
}
#[derive(Debug, Clone, Copy)]
enum HouseholderJob {
    Factor,
    Vector,
    Matrix { rhs: usize },
    Damped,
    Q,
    ApplyQ { rhs: usize },
    Reconstruct,
}
impl HouseholderWorkspaceRequest {
    /// Requests scratch for [`HouseholderDecomposition::factor_into`].
    pub const fn factor(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            rows,
            cols,
            orientation,
            job: HouseholderJob::Factor,
            residual: crate::Estimation::Skip,
            options: *options,
            workers: core::num::NonZeroUsize::MAX,
            packing: crate::workspace::Packing::Native,
        }
    }
    /// Requests scratch for [`HouseholderDecomposition::solve_into`].
    ///
    /// The solve needs a QR factorization with `rows ≥ cols`; other geometries
    /// make [`HouseholderDecomposition::workspace_requirements`] fail.
    pub const fn solve(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::Vector,
            ..Self::factor(rows, cols, orientation, options)
        }
    }
    /// Requests scratch for [`HouseholderDecomposition::solve_matrix_into`] with
    /// `right_hand_sides` columns.
    pub const fn solve_matrix(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        right_hand_sides: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::Matrix {
                rhs: right_hand_sides,
            },
            ..Self::factor(rows, cols, orientation, options)
        }
    }
    /// Requests scratch for [`HouseholderDecomposition::solve_damped_into`] and
    /// [`HouseholderDecomposition::solve_trust_region_into`] (real QR, `rows ≥ cols`).
    pub const fn damped_solve(
        rows: usize,
        cols: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::Damped,
            ..Self::factor(rows, cols, HouseholderOrientation::Qr, options)
        }
    }
    /// Requests scratch for [`HouseholderDecomposition::q_matrix_into`].
    pub const fn q_matrix(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::Q,
            ..Self::factor(rows, cols, orientation, options)
        }
    }
    /// Requests scratch for [`HouseholderDecomposition::apply_q_into`] with
    /// `right_hand_sides` columns.
    pub const fn apply_q(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        right_hand_sides: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::ApplyQ {
                rhs: right_hand_sides,
            },
            ..Self::factor(rows, cols, orientation, options)
        }
    }

    /// Requests scratch for [`HouseholderDecomposition::reconstruct_into`].
    pub const fn reconstruct(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: HouseholderJob::Reconstruct,
            ..Self::factor(rows, cols, orientation, options)
        }
    }
    /// Adds the scratch for the residual diagnostic of a solve, if `residual`
    /// requests it.
    pub const fn with_residual(mut self, residual: crate::Estimation) -> Self {
        self.residual = residual;
        self
    }
    /// Sizes the per-worker scratch for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: core::num::NonZeroUsize) -> Self {
        self.workers = workers;
        self
    }
    /// Adds the packing buffers that an injected BLAS provider needs.
    pub const fn with_provider_packing(mut self) -> Self {
        self.packing = crate::workspace::Packing::Provider;
        self
    }
    /// Sets the worker bound and BLAS packing from `context`.
    ///
    /// Product accuracy still comes from the options passed to the constructor.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    fn for_kernels(mut self, context: crate::KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }
    fn oriented(self) -> (usize, usize) {
        match self.orientation {
            HouseholderOrientation::Qr => (self.rows, self.cols),
            HouseholderOrientation::Lq => (self.cols, self.rows),
        }
    }
    fn prefix(self) -> Result<usize, SciOxideError> {
        let size = |shape: &[usize]| {
            scioxide_core::backend::checked_product(
                shape,
                scioxide_core::quantity!("householder_scratch"),
            )
        };
        let parts = match self.job {
            HouseholderJob::Reconstruct => [size(&[2, self.rows, self.cols])?, 0, 0, 0],
            HouseholderJob::Damped => [
                size(&[self.cols, self.cols])?,
                self.rows,
                size(&[3, self.cols])?,
                0,
            ],
            _ => [0; 4],
        };
        parts.into_iter().try_fold(0usize, |total, n| {
            total.checked_add(n).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("householder_scratch"))
            })
        })
    }
}

impl<S: NumericScalar> HouseholderDecomposition<S> {
    /// Returns the workspace size that `request` needs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if a solve request is not
    ///   for QR with `rows ≥ cols`, or a damped-solve request has complex scalars.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        request: &HouseholderWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let strict = matches!(
            request.job,
            HouseholderJob::Vector | HouseholderJob::Matrix { .. } | HouseholderJob::Damped
        );
        if strict
            && (request.orientation != HouseholderOrientation::Qr || request.rows < request.cols)
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("request"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "strict solves require tall or square QR geometry",
            ));
        }
        if matches!(request.job, HouseholderJob::Damped) && !S::IS_REAL {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("request"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "damped Householder solves require a real scalar",
            ));
        }
        let (rows, cols) = request.oriented();
        let qr = |job| {
            super::qr::requirements::<S>(job, &request.options, request.workers, request.packing)
        };
        let mut requirements = match request.job {
            HouseholderJob::Factor => qr(QrRequest::Factor { rows, cols })?,
            HouseholderJob::ApplyQ { rhs } => qr(QrRequest::ApplyMatrix { rows, cols, rhs })?,
            HouseholderJob::Q => qr(QrRequest::ApplyMatrix {
                rows,
                cols,
                rhs: rows,
            })?,
            HouseholderJob::Reconstruct => qr(QrRequest::ApplyMatrix {
                rows,
                cols,
                rhs: cols,
            })?,
            HouseholderJob::Damped => crate::LinalgRequirements::default(),
            HouseholderJob::Vector => qr(QrRequest::Vector { rows })?,
            HouseholderJob::Matrix { rhs } => qr(QrRequest::SolveMatrix { rows, cols, rhs })?,
        };
        *requirements.scalar_mut(ScalarRegion::Direct) = requirements
            .scalar(ScalarRegion::Direct)
            .checked_add(request.prefix()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("householder_scratch"))
            })?;
        if strict && rows.min(cols) != 0 {
            requirements = requirements.max(&crate::estimate::condition_requirements(
                rows.max(cols),
                request.options.byte_limit(),
            )?);
        }
        if strict
            && request.residual == crate::Estimation::Estimate
            && !matches!(request.job, HouseholderJob::Damped)
        {
            let columns = match request.job {
                HouseholderJob::Matrix { rhs } => rhs,
                _ => 1,
            };
            if columns > 1 {
                requirements = requirements.max(&qr(QrRequest::ApplyMatrix {
                    rows,
                    cols,
                    rhs: columns,
                })?);
            }
            let prefix = scioxide_core::backend::checked_product(
                &[request.cols, columns],
                scioxide_core::quantity!("householder_scratch"),
            )?;
            *requirements.scalar_mut(ScalarRegion::Direct) = requirements
                .scalar(ScalarRegion::Direct)
                .checked_add(prefix)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("householder_scratch"))
                })?;
        }
        requirements.byte_limit = request.options.byte_limit();
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }

    /// Allocates a workspace large enough for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     HouseholderDecomposition, HouseholderOrientation, HouseholderWorkspaceRequest,
    ///     LinalgExecutionOptions,
    /// };
    /// let request = HouseholderWorkspaceRequest::factor(
    ///     5,
    ///     3,
    ///     HouseholderOrientation::Qr,
    ///     &LinalgExecutionOptions::new(),
    /// );
    /// let workspace = HouseholderDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &HouseholderWorkspaceRequest,
    ) -> Result<crate::LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = crate::LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Shape of the matrix that the internal QR engine factors for an orientation.
const fn oriented(rows: usize, cols: usize, orientation: HouseholderOrientation) -> (usize, usize) {
    match orientation {
        HouseholderOrientation::Qr => (rows, cols),
        HouseholderOrientation::Lq => (cols, rows),
    }
}

/// Householder QR factorization `A[:, p] = Q R` or LQ factorization
/// `A[p, :] = L Q` of a rectangular matrix, with optional pivoting.
///
/// See [`HouseholderOrientation`] for the shapes and [`HouseholderPivoting`]
/// for the pivoting. `A` may be tall, square or wide. `Q` is stored as a
/// product of Householder reflectors `I − τ v vᴴ` and is formed only on
/// request. The factorization is blocked, with reflector panels applied in
/// compact WY form through GEMM; QR of an `m × n` matrix costs about
/// `2mn² − 2n³/3` flops for `m ≥ n`. The factorization never fails on a
/// rank-deficient matrix; rank is decided when you solve.
///
/// [`Self::solve`] solves full-rank least-squares problems.
/// [`Self::into_minimum_norm`] prepares minimum-norm solutions of
/// rank-deficient or underdetermined problems.
///
/// # Accuracy
///
/// Householder QR is normwise backward stable: the computed factors are those
/// of `A + ΔA` with `‖ΔA‖_F ≤ c m n ε ‖A‖_F`, where `ε` is the machine epsilon
/// and `c` is a small constant (Higham, 2002, §19.3).
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §5.2.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Chapter 19.
#[derive(Debug, Clone)]
#[must_use]
pub struct HouseholderDecomposition<S: NumericScalar> {
    pub(super) qr: QrDecomposition<S>,
    source: crate::Matrix<S>,
    orientation: HouseholderOrientation,
    rows: usize,
    cols: usize,
    pivoting: HouseholderPivoting,
    pub(super) one_norm: scioxide_core::Scaled<S::Real>,
}

impl<S: NumericScalar> HouseholderDecomposition<S> {
    /// Returns the bytes of factor storage that a `rows × cols` factorization of
    /// the given orientation needs.
    ///
    /// The count excludes allocator overhead and the fixed-size fields. The
    /// workspace byte limit does not apply to this storage.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{HouseholderDecomposition, HouseholderOrientation};
    /// assert_eq!(
    ///     HouseholderDecomposition::<f64>::factor_storage_bytes(0, 0, HouseholderOrientation::Qr)?,
    ///     0
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn factor_storage_bytes(
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
    ) -> Result<usize, SciOxideError> {
        let elements = scioxide_core::backend::checked_product(
            &[rows, cols],
            scioxide_core::quantity!("factor_elements"),
        )?;
        let permutation = match orientation {
            HouseholderOrientation::Qr => cols,
            HouseholderOrientation::Lq => rows,
        };
        super::util::storage_bytes([
            (elements, core::mem::size_of::<S>()),
            (elements, core::mem::size_of::<S>()),
            (
                QrDecomposition::<S>::reflector_len(
                    match orientation {
                        HouseholderOrientation::Qr => rows,
                        HouseholderOrientation::Lq => cols,
                    },
                    rows.min(cols),
                )?,
                core::mem::size_of::<S>(),
            ),
            (rows.min(cols), core::mem::size_of::<S>()),
            (permutation, core::mem::size_of::<usize>()),
        ])
    }
    /// Returns the bytes of factor storage currently allocated.
    ///
    /// The count includes capacity left over from an earlier, larger
    /// factorization, and excludes allocator overhead and the fixed-size fields.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        self.qr
            .retained_bytes()?
            .checked_add(super::util::storage_bytes([(
                self.source.capacity(),
                core::mem::size_of::<S>(),
            )])?)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("factor_storage")))
    }

    /// Grows the factor storage to hold a `rows × cols` factorization of the
    /// given orientation.
    ///
    /// The current factorization is unchanged. Growth is exact: after reserving
    /// into [`Self::empty`], [`Self::retained_bytes`] equals
    /// [`Self::factor_storage_bytes`]. A later [`Self::factor_into`] of a matrix
    /// of this shape does not allocate factor storage. Capacity is never released.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if growth fails. The current factorization stays
    ///   valid, and part of the growth may remain.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{HouseholderDecomposition, HouseholderOrientation};
    /// let lq = HouseholderOrientation::Lq;
    /// let mut factor = HouseholderDecomposition::<f64>::empty();
    /// factor.try_reserve(5, 3, lq)?;
    /// assert_eq!(factor.check(5, 3, lq)?, Capacity::Sufficient);
    /// assert_eq!(
    ///     factor.retained_bytes()?,
    ///     HouseholderDecomposition::<f64>::factor_storage_bytes(5, 3, lq)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
    ) -> Result<(), SciOxideError> {
        self.source.try_reserve_shape(rows, cols)?;
        let (rows, cols) = oriented(rows, cols, orientation);
        self.qr.reserve_factor(rows, cols)
    }

    /// Reports whether the factor storage can hold a `rows × cols` factorization
    /// of the given orientation without growing.
    ///
    /// See [`Self::try_reserve`] for an example.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn check(
        &self,
        rows: usize,
        cols: usize,
        orientation: HouseholderOrientation,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let mut plan = crate::workspace::accounting::StoragePlan::new(0);
        plan.matrix(&self.source, rows, cols)?;
        let (rows, cols) = oriented(rows, cols, orientation);
        self.qr.action_storage(&mut plan, rows, cols)?;
        plan.check(None)
    }

    /// Returns an empty factorization of a `0 × 0` matrix, to pass to
    /// [`Self::factor_into`].
    pub const fn empty() -> Self {
        Self {
            qr: QrDecomposition::empty(),
            source: crate::Matrix::empty(),
            pivoting: HouseholderPivoting::Column,
            one_norm: scioxide_core::Scaled::ZERO,
            orientation: HouseholderOrientation::Qr,
            rows: 0,
            cols: 0,
        }
    }

    /// Computes the QR or LQ factorization of `matrix`.
    ///
    /// The factorization runs on the calling thread. Use [`Self::factor_into`]
    /// to reuse storage or to run the trailing updates in parallel.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `matrix` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a column (row, for LQ) norm or an
    ///   entry of `R` (`L`) overflows, for example a column of two entries near
    ///   the largest finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting, Matrix,
    /// };
    /// // Fit y = c₀ + c₁ t to (0, 1), (1, 3), (2, 5); the exact fit is c = (1, 2).
    /// let a = Matrix::from_rows(&[[1.0_f64, 0.0], [1.0, 1.0], [1.0, 2.0]])?;
    /// let qr = HouseholderDecomposition::factor(
    ///     &a,
    ///     HouseholderOrientation::Qr,
    ///     HouseholderPivoting::Column,
    /// )?;
    /// let c = qr.solve(&[1.0, 3.0, 5.0][..])?;
    /// let c = c.as_slice();
    /// assert!((c[0] - 1.0).abs() < 1e-12 && (c[1] - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn factor<'m>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        orientation: HouseholderOrientation,
        pivoting: HouseholderPivoting,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(
            matrix,
            orientation,
            pivoting,
            &crate::LinalgExecutionOptions::default(),
        )
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the product accuracy and the limits
    /// described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and:
    ///
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        orientation: HouseholderOrientation,
        pivoting: HouseholderPivoting,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            matrix,
            orientation,
            pivoting,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the factorization into `output`, the reusable form of [`Self::factor`].
    ///
    /// `workspace` holds the scratch and `context` supplies the executor, clock
    /// and cancellation. The trailing updates follow the execution policy in
    /// `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::factor`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn factor_into<'m, 'scope, 'capability>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        orientation: HouseholderOrientation,
        pivoting: HouseholderPivoting,
        options: &crate::LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        context.execution().check()?;
        let matrix = matrix.into();
        let request =
            HouseholderWorkspaceRequest::factor(matrix.rows(), matrix.cols(), orientation, options)
                .for_kernels(context);
        let requirements = Self::workspace_requirements(&request)?;
        let finite = matrix.try_finite_dispatched(context.dispatch())?;
        let original_norm = (orientation == HouseholderOrientation::Lq)
            .then(|| crate::diagnostics::matrix_one_norm(finite.as_view()))
            .transpose()?;
        let oriented = match orientation {
            HouseholderOrientation::Qr => finite,
            HouseholderOrientation::Lq => finite.adjoint(),
        };
        workspace.try_reserve(&requirements)?;
        output.try_reserve(matrix.rows(), matrix.cols(), orientation)?;
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "householder_factor", 1),
                matrix.rows().max(matrix.cols()) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                crate::solve_support::admit_iteration(
                    Some(context.execution()),
                    Some(solve),
                    "direct numerical work budget exhausted",
                )?;
                let oriented_norm = QrDecomposition::factor_with_pivoting(
                    oriented,
                    &mut output.qr,
                    workspace,
                    options.execution_policy(),
                    context,
                    pivoting,
                )?;
                output
                    .source
                    .try_resize_flat(matrix.rows(), matrix.cols())?;
                super::util::copy_rows(matrix, output.source.as_mut_slice());
                output.orientation = orientation;
                output.pivoting = pivoting;
                output.one_norm = original_norm.unwrap_or(oriented_norm);
                output.rows = matrix.rows();
                output.cols = matrix.cols();
                Ok(())
            },
        )
    }

    /// Returns the packed QR factors: `R` on and above the diagonal, and the
    /// Householder vectors `v` (with implicit leading 1) below it.
    ///
    /// For LQ these are the QR factors of `Aᴴ`, so `L = Rᴴ`.
    pub fn packed_factors(&self) -> crate::MatrixView<'_, S> {
        self.qr.packed_factors().as_view()
    }

    /// Returns whether this is a QR or an LQ factorization.
    pub const fn orientation(&self) -> HouseholderOrientation {
        self.orientation
    }

    /// Returns the number of rows `m` of the factored matrix.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns `n` of the factored matrix.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the permutation `p`: of columns for QR (`A[:, p] = Q R`), of rows
    /// for LQ (`A[p, :] = L Q`).
    pub fn permutation(&self) -> &[usize] {
        self.qr.permutation()
    }

    /// Classifies the numerical rank from the moduli of the diagonal of `R` (or `L`).
    ///
    /// This is reliable only with [`HouseholderPivoting::Column`]. See
    /// [`RankPolicy`] for the threshold.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a [`RankPolicy::Threshold`]
    ///   threshold exceeds the largest finite value.
    pub fn analyze_rank(
        &self,
        policy: RankPolicy<S::Real>,
    ) -> Result<RankReport<S::Real>, SciOxideError> {
        self.qr.analyze_rank(policy)
    }

    /// Returns the diagonal of `R` (QR) or `L` (LQ), of length `min(m, n)`, in
    /// pivoted order.
    ///
    /// The view borrows the stored factors.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting, Matrix,
    /// };
    /// let a = Matrix::from_vec(2, 2, vec![3.0_f64, 0.0, 0.0, 4.0])?;
    /// let qr = HouseholderDecomposition::factor(
    ///     &a,
    ///     HouseholderOrientation::Qr,
    ///     HouseholderPivoting::None)?;
    /// assert_eq!(qr.triangular_diagonal().iter().collect::<Vec<_>>(), vec![3.0, 4.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn triangular_diagonal(&self) -> crate::VectorView<'_, S> {
        let diagonal = self.qr.triangular_diagonal();
        match self.orientation {
            HouseholderOrientation::Qr => diagonal,
            HouseholderOrientation::Lq => diagonal.conjugate(),
        }
    }

    /// Returns `R` (QR) or `L` (LQ) as an `m × n` matrix.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the output cannot be allocated.
    pub fn to_triangular_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.rows, self.cols)?;
        self.triangular_matrix_into(output.view_mut())?;
        Ok(output)
    }

    /// Writes `R` (QR) or `L` (LQ) to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `m × n`.
    pub fn triangular_matrix_into(
        &self,
        mut output: crate::MatrixViewMut<'_, S>,
    ) -> Result<(), SciOxideError> {
        if output.rows() != self.rows || output.cols() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [output.rows(), output.cols()],
            ));
        }
        if output.is_empty() {
            return Ok(());
        }
        let packed = self.qr.packed_factors().as_view();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let value = match self.orientation {
                    HouseholderOrientation::Qr if row <= col => packed.get_element(row, col),
                    HouseholderOrientation::Lq if row >= col => packed.get_element(col, row).conj(),
                    _ => S::ZERO,
                };
                output.set_element(row, col, value);
            }
        }
        Ok(())
    }

    pub(crate) fn qr_triangular_into(&self, output: &mut Matrix<S>) -> Result<(), SciOxideError> {
        self.qr.r_matrix_into(output)
    }

    /// Returns the unitary factor `Q`: `m × m` for QR, `n × n` for LQ.
    ///
    /// Forming `Q` costs O(m²n) flops for QR. To multiply by `Q`, use
    /// [`Self::apply_q`] instead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the output or the scratch
    ///   cannot be allocated.
    pub fn to_q_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        self.to_q_matrix_with_options(&crate::LinalgExecutionOptions::default())
    }

    /// Runs [`to_q_matrix`](Self::to_q_matrix) with `options`, which set the product accuracy and
    /// the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`to_q_matrix`](Self::to_q_matrix), and:
    ///
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn to_q_matrix_with_options(
        &self,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let order = self.qr.packed_factors().rows();
        let mut output = Matrix::zeros(order, order)?;
        self.q_matrix_into(
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }

    /// Writes the unitary factor `Q` to `output`, the reusable form of [`Self::to_q_matrix`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `m × m` (QR) or `n × n` (LQ).
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn q_matrix_into<'scope, 'capability>(
        &self,
        options: &crate::LinalgExecutionOptions,
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        context.execution().check()?;
        let order = self.qr.packed_factors().rows();
        crate::checked::expect_shape([order, order], [output.rows(), output.cols()])?;
        let request =
            HouseholderWorkspaceRequest::q_matrix(self.rows, self.cols, self.orientation, options)
                .for_kernels(context);
        workspace.try_reserve(&Self::workspace_requirements(&request)?)?;
        self.run_unitary(
            workspace,
            scope_context,
            scioxide_core::algorithm_id!("linalg", "householder_q", 1),
            options,
            |workspace, kernels| {
                self.qr.q_view(
                    output,
                    self.orientation == HouseholderOrientation::Lq,
                    workspace,
                    options.execution_policy(),
                    kernels,
                )
            },
        )
    }

    /// Returns `op(Q) X`, where `op` is selected by `operation`.
    ///
    /// `X` has `m` rows for QR and `n` rows for LQ. The reflectors are applied
    /// directly, without forming `Q`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` has the wrong number of rows.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of a reflector update
    ///   overflows; `‖op(Q) x‖₂ = ‖x‖₂`, but an entry can exceed the largest
    ///   finite value when `‖x‖₂` does.
    pub fn apply_q<'m>(
        &self,
        input: impl Into<crate::MatrixView<'m, S>>,
        operation: crate::StructuredOperation,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.apply_q_with_options(input, operation, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`apply_q`](Self::apply_q) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`apply_q`](Self::apply_q), and:
    ///
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn apply_q_with_options<'m>(
        &self,
        input: impl Into<crate::MatrixView<'m, S>>,
        operation: crate::StructuredOperation,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let input = input.into();
        let mut output = Matrix::zeros(self.qr.packed_factors().rows(), input.cols())?;
        self.apply_q_into(
            input,
            operation,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }

    /// Writes `op(Q) X` to `output`, the reusable form of [`Self::apply_q`].
    ///
    /// For complex scalars, `Transpose` and `Adjoint` differ.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have the shape of `input`.
    /// - The errors of [`Self::apply_q`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn apply_q_into<'m, 'scope, 'capability>(
        &self,
        input: impl Into<crate::MatrixView<'m, S>>,
        transform: crate::StructuredOperation,
        options: &crate::LinalgExecutionOptions,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        context.execution().check()?;
        let input = input.into();
        let rows = self.qr.packed_factors().rows();
        if input.rows() != rows {
            return Err(SciOxideError::dimension_mismatch(
                [rows, input.cols()],
                [input.rows(), input.cols()],
            ));
        }
        if output.rows() != rows || input.cols() != output.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [rows, input.cols()],
                [output.rows(), output.cols()],
            ));
        }
        let request = HouseholderWorkspaceRequest::apply_q(
            self.rows,
            self.cols,
            self.orientation,
            input.cols(),
            options,
        )
        .for_kernels(context);
        let requirements = Self::workspace_requirements(&request)?;
        let input = input.try_finite_dispatched(context.dispatch())?;
        workspace.try_reserve(&requirements)?;
        if output.is_empty() {
            return Ok(());
        }
        let transposed = transform != crate::StructuredOperation::Normal;
        let adjoint = transposed != (self.orientation == HouseholderOrientation::Lq);
        let conjugate = transform == crate::StructuredOperation::Transpose && !S::IS_REAL;
        let input = if conjugate { input.conjugate() } else { input };
        self.run_unitary(
            workspace,
            scope_context,
            scioxide_core::algorithm_id!("linalg", "householder_apply_q", 1),
            options,
            |workspace, kernels| {
                self.qr.apply_view(
                    input,
                    output.reborrow(),
                    adjoint,
                    workspace,
                    options.execution_policy(),
                    kernels,
                )?;
                // This infallible conjugation is part of publication; no callbacks run here.
                if conjugate {
                    for row in 0..output.rows() {
                        for col in 0..output.cols() {
                            output.set_element(row, col, output.get_element(row, col).conj());
                        }
                    }
                }
                Ok(())
            },
        )
    }

    /// Returns `A` recomputed from the factors, with the permutation undone.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the output or the scratch
    ///   cannot be allocated.
    /// - `NumericalInstability` (`Overflow`) if an entry of the product
    ///   overflows.
    pub fn reconstruct(&self) -> Result<Matrix<S>, SciOxideError> {
        self.reconstruct_with_options(&crate::LinalgExecutionOptions::default())
    }

    /// Runs [`reconstruct`](Self::reconstruct) with `options`, which set the product accuracy and
    /// the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`reconstruct`](Self::reconstruct), and:
    ///
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn reconstruct_with_options(
        &self,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Matrix::zeros(self.rows, self.cols)?;
        self.reconstruct_into(
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }

    /// Writes `A` recomputed from the factors to `output`, the reusable form of
    /// [`Self::reconstruct`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `m × n`.
    /// - `NumericalInstability` (`Overflow`) if an entry of the product
    ///   overflows.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn reconstruct_into<'scope, 'capability>(
        &self,
        options: &crate::LinalgExecutionOptions,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        context.execution().check()?;
        if output.rows() != self.rows || output.cols() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [output.rows(), output.cols()],
            ));
        }
        let request = HouseholderWorkspaceRequest::reconstruct(
            self.rows,
            self.cols,
            self.orientation,
            options,
        )
        .for_kernels(context);
        workspace.try_reserve(&Self::workspace_requirements(&request)?)?;
        self.run_unitary(
            workspace,
            scope_context,
            scioxide_core::algorithm_id!("linalg", "householder_reconstruct", 1),
            options,
            |workspace, context| {
                if output.is_empty() {
                    return Ok(());
                }
                workspace.with_prefix(request.prefix()?, |buffers, child| {
                    let (matrix, temporary) =
                        crate::workspace::split_mut(buffers, self.rows * self.cols)?;
                    self.reconstruct_staged(
                        matrix,
                        temporary,
                        child,
                        options.execution_policy(),
                        context,
                    )?;
                    context.execution().check()?;
                    for row in 0..self.rows {
                        for col in 0..self.cols {
                            output.set_element(row, col, matrix[row * self.cols + col]);
                        }
                    }
                    Ok(())
                })
            },
        )
    }

    /// Run one unitary application or reconstruction in a fresh solve scope of
    /// the workspace, charging one iteration of the local allowance.
    fn run_unitary<'capability>(
        &self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: crate::LinalgExecutionContext<'_, 'capability>,
        algorithm: scioxide_core::AlgorithmId,
        options: &crate::LinalgExecutionOptions,
        work: impl FnOnce(
            &mut LinalgWorkspace<'_, S>,
            KernelContext<'capability>,
        ) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                algorithm,
                self.rows.max(self.cols) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, kernels, solve| {
                crate::solve_support::admit_iteration(
                    Some(kernels.execution()),
                    Some(solve),
                    "direct numerical work budget exhausted",
                )?;
                work(workspace, kernels)
            },
        )
    }

    pub(super) fn reconstruct_staged(
        &self,
        matrix: &mut [S],
        temporary: &mut [S],
        workspace: &mut LinalgWorkspace<'_, S>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        if matrix.is_empty() {
            return Ok(());
        }
        let packed = self.qr.packed_factors();
        let (rows, cols) = (packed.rows(), packed.cols());
        matrix.fill(S::ZERO);
        for row in 0..rows.min(cols) {
            for col in row..cols {
                matrix[row * cols + col] = packed.as_view().get_element(row, col);
            }
        }
        self.qr.apply_view(
            crate::FiniteMatrixView::from_validated(
                crate::MatrixView::from_slice(rows, cols, matrix)?,
                packed.maximum,
            ),
            crate::MatrixViewMut::from_slice(rows, cols, temporary)?,
            false,
            workspace,
            policy,
            context,
        )?;
        match self.orientation {
            HouseholderOrientation::Qr => {
                for row in 0..self.rows {
                    for (col, &original) in self.permutation().iter().enumerate() {
                        matrix[row * self.cols + original] = temporary[row * self.cols + col];
                    }
                }
            }
            HouseholderOrientation::Lq => {
                for (col, &original) in self.permutation().iter().enumerate() {
                    for row in 0..self.cols {
                        matrix[original * self.cols + row] =
                            temporary[row * self.rows + col].conj();
                    }
                }
            }
        }
        Ok(())
    }

    /// Prepares minimum-norm solutions `x = A⁺ b` of rank-deficient or
    /// underdetermined problems, with the rank fixed by `acceptance`.
    ///
    /// The factorization must use [`HouseholderPivoting::Column`]. The plan
    /// computes the complete orthogonal decomposition once; see
    /// [`HouseholderMinimumNormPlan`](crate::HouseholderMinimumNormPlan).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) without column pivoting.
    /// - The errors of the plan's preparation.
    pub fn into_minimum_norm(
        self,
        acceptance: &DirectSolveOptions<S::Real>,
    ) -> Result<super::minimum_norm::HouseholderMinimumNormPlan<S>, SciOxideError> {
        if self.pivoting != HouseholderPivoting::Column {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "minimum-norm preparation requires pivoted QR or LQ",
            ));
        }
        super::minimum_norm::HouseholderMinimumNormPlan::prepare(self, *acceptance)
    }

    /// Estimates the reciprocal condition number `1 / (‖A‖₁ ‖A⁺‖₁)`, where `A⁺`
    /// is the pseudoinverse (`A⁻¹` for square `A`).
    ///
    /// The method is that of
    /// [`LuDecomposition::condition_estimate`](crate::LuDecomposition::condition_estimate).
    /// A rank-deficient factor gives 0. For a wide QR or a tall LQ
    /// factorization, the estimate goes through a minimum-norm plan with
    /// [`RankPolicy::Exact`].
    ///
    /// # Errors
    ///
    /// - `DomainError` if the matrix is empty.
    /// - `InvalidInput` (`UnsupportedConfiguration`) for a wide QR or tall LQ
    ///   factorization without column pivoting.
    /// - `NumericalInstability` (`IllConditioned`) if the estimator cannot
    ///   complete.
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        let packed = self.qr.packed_factors();
        if packed.rows() < packed.cols() {
            return self
                .clone()
                .into_minimum_norm(&DirectSolveOptions::new(RankPolicy::Exact))?
                .condition_estimate();
        }
        self.condition_estimate_scoped(
            &crate::LinalgExecutionOptions::new(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Estimates the reciprocal condition number as [`Self::condition_estimate`]
    /// does, using `workspace` and `context`.
    ///
    /// Each solve counts toward the iteration budget in `options`. For a wide QR
    /// or tall LQ factorization, use the condition estimate of
    /// [`Self::into_minimum_norm`] instead.
    ///
    /// # Errors
    ///
    /// - `DomainError` if the matrix is empty.
    /// - `InvalidInput` (`UnsupportedConfiguration`) for a wide QR or tall LQ
    ///   factorization.
    /// - `NumericalInstability` (`IllConditioned`) if the estimator cannot
    ///   complete.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn condition_estimate_scoped<'scope, 'capability>(
        &self,
        options: &crate::LinalgExecutionOptions,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        let transposed = self.orientation == HouseholderOrientation::Lq;
        crate::estimate::scoped_condition(
            options,
            workspace,
            context.into(),
            || {
                if self.rows.min(self.cols) == 0 {
                    return Err(crate::analysis::empty_condition());
                }
                let packed = self.qr.packed_factors();
                if packed.rows() < packed.cols() {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("self"),
                        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                        "wide oriented factors require a prepared minimum-norm condition query",
                    ));
                }
                Ok(crate::estimate::ConditionQuery {
                    algorithm: scioxide_core::algorithm_id!("linalg", "householder_condition", 1),
                    order: self.rows.max(self.cols),
                    norm: self.one_norm,
                    singular: !self.analyze_rank(RankPolicy::Exact)?.is_full_rank(),
                })
            },
            |action, kernels| {
                let adjoint = action.adjoint ^ transposed;
                action.solve(
                    |values, temporary| {
                        self.qr
                            .inverse_ordinary(values, temporary, adjoint, kernels.dispatch())
                    },
                    |rhs, work| self.qr.inverse_scaled(rhs, work, adjoint),
                )
            },
        )
    }

    /// Solves the least-squares problem `min ‖A x − b‖₂` and returns `x`.
    ///
    /// The factorization must be QR with `m ≥ n`. The solution is
    /// `x = P R₁⁻¹ (Qᴴ b)₁`, where `R₁` is the leading `n × n` block of `R`. The solve
    /// uses the default [`DirectSolveOptions`](crate::DirectSolveOptions), so only an
    /// exactly zero diagonal entry of `R` rejects it. For rank-deficient or
    /// underdetermined problems use [`least_squares`](crate::least_squares) or
    /// [`Self::into_minimum_norm`]; for a numerical rank cutoff or a condition
    /// threshold use [`Self::solve_with_options`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) for LQ or for `m < n`.
    /// - `DimensionMismatch` if `rhs` does not have length `m`.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `Singular` if `R` has an exactly zero diagonal entry.
    /// - `NumericalInstability` (`Overflow`) if an entry of `Qᴴ b` or of `x`
    ///   overflows.
    ///
    /// # References
    ///
    /// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    /// Hopkins, 2013, §5.3.
    pub fn solve<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.solve_with_options(rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves the least-squares problem `min ‖A x − b‖₂`, checks the result
    /// against `acceptance` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The factorization must be QR with `m ≥ n`, and `A` must have full column
    /// rank under `acceptance.rank_policy()`; otherwise the solve returns `Ok`
    /// with [`DirectSolveTermination::RankDeficient`] and no solution. The
    /// condition check is as for
    /// [`LuDecomposition::solve_with_options`](crate::LuDecomposition::solve_with_options).
    /// A requested residual is `‖b − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)`; for an inconsistent
    /// system it does not tend to zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) for LQ or for `m < n`.
    /// - `DimensionMismatch` if `rhs` does not have length `m`.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `DomainError` if `A` is empty and a condition threshold is set.
    /// - `NumericalInstability` (`Overflow`) if an entry of `Qᴴ b`, of `x` or of
    ///   a requested residual overflows.
    /// - Limit errors from `acceptance` as described in [Execution](crate#execution).
    pub fn solve_with_options<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        acceptance: &DirectSolveOptions<S::Real>,
    ) -> Result<DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let acceptance = acceptance.with_execution_options(
            acceptance
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let acceptance = &acceptance;
        let mut output = Vector::zeros(self.cols)?;
        let report = self.solve_into(
            rhs,
            acceptance,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(DirectSolveResult::new(output, report))
    }

    /// Solves the least-squares problem into `output`, the reusable form of
    /// [`Self::solve`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have length `n`.
    /// - The errors of [`Self::solve`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        acceptance: &DirectSolveOptions<S::Real>,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        let options = acceptance.execution_options();
        let options = &options;
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let rhs = rhs.into();
        if rhs.len() != self.rows || output.len() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [rhs.len(), output.len()],
            ));
        }
        let request =
            HouseholderWorkspaceRequest::solve(self.rows, self.cols, self.orientation, options)
                .with_residual(acceptance.residual())
                .for_kernels(context.kernel());
        let requirements = Self::workspace_requirements(&request)?;
        let _ = workspace.check(&requirements)?;
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        workspace.try_reserve(&requirements)?;
        if self.rows.min(self.cols) == 0 && acceptance.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "householder_solve_into", 1),
                self.rows.max(self.cols) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                let mut report =
                    self.admit_solve(acceptance, workspace, context.scoped(&mut *solve))?;
                if report.termination() == DirectSolveTermination::Accepted {
                    crate::solve_support::admit_iteration(
                        Some(context.execution()),
                        Some(&mut *solve),
                        "Householder RHS budget exhausted",
                    )?;
                    if acceptance.residual() == crate::Estimation::Estimate {
                        let residual = self.solve_residual_columns(
                            crate::MatrixView::from(rhs.as_view()),
                            crate::MatrixViewMut::from(output),
                            workspace,
                            options,
                            context,
                        )?;
                        report = DirectSolveReport::new(
                            report.termination(),
                            report.rank(),
                            report.condition(),
                            residual,
                        );
                    } else {
                        self.qr.solve_view(rhs, output, workspace, context)?;
                    }
                }
                Ok(report)
            },
        )
    }

    /// Solves `min ‖A X − B‖_F` column by column, as [`Self::solve`] does, and
    /// returns `X`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `m` rows.
    /// - The other errors of [`Self::solve`].
    pub fn solve_matrix<'b>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.solve_matrix_with_options(rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `min ‖A X − B‖_F` column by column, checks the result against
    /// `acceptance` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The rank and condition checks of [`Self::solve_with_options`] are made once
    /// for all columns. A requested residual is the largest over the columns.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `m` rows.
    /// - The other errors of [`Self::solve_with_options`].
    pub fn solve_matrix_with_options<'b>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
        acceptance: &DirectSolveOptions<S::Real>,
    ) -> Result<DirectSolveResult<Matrix<S>, S::Real>, SciOxideError> {
        let acceptance = acceptance.with_execution_options(
            acceptance
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let acceptance = &acceptance;
        let rhs = rhs.into();
        let mut output = Matrix::zeros(self.cols, rhs.cols())?;
        let report = self.solve_matrix_into(
            rhs,
            acceptance,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(DirectSolveResult::new(output, report))
    }

    /// Solves `min ‖A X − B‖_F` into `output`, the reusable form of
    /// [`Self::solve_matrix`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `n × rhs.cols()`.
    /// - The errors of [`Self::solve_matrix`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn solve_matrix_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
        acceptance: &DirectSolveOptions<S::Real>,
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        let options = acceptance.execution_options();
        let options = &options;
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let rhs = rhs.into();
        if rhs.rows() != self.rows {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, rhs.cols()],
                [rhs.rows(), rhs.cols()],
            ));
        }
        if output.rows() != self.cols || rhs.cols() != output.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [self.cols, rhs.cols()],
                [output.rows(), output.cols()],
            ));
        }
        let request = HouseholderWorkspaceRequest::solve_matrix(
            self.rows,
            self.cols,
            self.orientation,
            rhs.cols(),
            options,
        )
        .with_residual(acceptance.residual())
        .for_kernels(context.kernel());
        let requirements = Self::workspace_requirements(&request)?;
        let _ = workspace.check(&requirements)?;
        let rhs = rhs.try_finite_dispatched(context.kernel().dispatch())?;
        workspace.try_reserve(&requirements)?;
        if self.rows.min(self.cols) == 0 && acceptance.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "householder_solve_matrix_into", 1),
                self.rows.max(self.cols) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                let mut report =
                    self.admit_solve(acceptance, workspace, context.scoped(&mut *solve))?;
                if rhs.cols() == 0 || (self.rows == 0 && self.cols == 0) {
                    return Ok(DirectSolveReport::new(
                        report.termination(),
                        report.rank(),
                        report.condition(),
                        crate::analysis::empty_residual(acceptance, rhs.cols())?,
                    ));
                }
                if report.termination() == DirectSolveTermination::Accepted {
                    for _ in 0..rhs.cols() {
                        crate::solve_support::admit_iteration(
                            Some(context.execution()),
                            Some(&mut *solve),
                            "Householder RHS budget exhausted",
                        )?;
                    }
                    let residual = (|| {
                        if acceptance.residual() == crate::Estimation::Estimate {
                            self.solve_residual_columns(
                                rhs.as_view(),
                                output,
                                workspace,
                                options,
                                context,
                            )
                        } else {
                            self.qr.solve_matrix_view(
                                rhs,
                                output,
                                workspace,
                                options.execution_policy(),
                                context,
                            )?;
                            Ok(None)
                        }
                    })()?;
                    report = DirectSolveReport::new(
                        report.termination(),
                        report.rank(),
                        report.condition(),
                        residual,
                    );
                }
                Ok(report)
            },
        )
    }

    fn solve_residual_columns(
        &self,
        rhs: crate::MatrixView<'_, S>,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        options: &crate::LinalgExecutionOptions,
        kernels: KernelContext<'_>,
    ) -> Result<Option<ResidualReport<S::Real>>, SciOxideError> {
        if rhs.cols() == 0 {
            return Ok(None);
        }
        workspace.with_prefix(self.cols * rhs.cols(), |candidate, child| {
            if rhs.cols() == 1 {
                self.qr.solve_view(
                    crate::FiniteVectorView::from_workspace(rhs.col(0)?),
                    crate::VectorViewMut::from_slice(candidate),
                    child,
                    kernels,
                )?;
            } else {
                self.qr.solve_matrix_view(
                    crate::FiniteMatrixView::from_workspace(rhs),
                    crate::MatrixViewMut::from_slice(self.cols, rhs.cols(), candidate)?,
                    child,
                    options.execution_policy(),
                    kernels,
                )?;
            }
            let candidate = crate::MatrixView::from_slice(self.cols, rhs.cols(), candidate)?;
            let residual = self.residual_columns(rhs, candidate, kernels)?;
            kernels.execution().check()?;
            crate::kernel::level2::copy_into(
                candidate,
                output.reborrow(),
                crate::CopyPolicy::Preserve,
            )?;
            Ok(residual)
        })
    }

    pub(super) fn residual_columns(
        &self,
        rhs: crate::MatrixView<'_, S>,
        solution: crate::MatrixView<'_, S>,
        kernels: KernelContext<'_>,
    ) -> Result<Option<ResidualReport<S::Real>>, SciOxideError> {
        if rhs.cols() == 0 {
            return Ok(None);
        }
        if self.rows == 0 && self.cols == 0 {
            return Ok(Some(ResidualReport::unscaled(
                S::Real::ZERO,
                NormSelection::Infinity,
            )?));
        }
        let mut largest = None;
        // The retained source is authoritative. Measure this diagnostic only
        // when requested instead of charging every factorization for it.
        let infinity_norm = crate::diagnostics::matrix_one_norm(self.source.view().transpose())?;
        for col in 0..rhs.cols() {
            crate::analysis::retain_largest_residual(
                &mut largest,
                crate::diagnostics::matrix_backward_error(
                    self.source.view(),
                    infinity_norm,
                    solution.col(col)?,
                    rhs.col(col)?,
                    kernels,
                )?,
            );
        }
        Ok(largest)
    }

    fn admit_solve(
        &self,
        acceptance: &DirectSolveOptions<S::Real>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: crate::LinalgExecutionContext<'_, '_>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        let rank = self.analyze_rank(acceptance.rank_policy())?;
        let condition = acceptance
            .minimum_reciprocal_condition()
            .map(|_| {
                self.condition_estimate_scoped(&acceptance.execution_options(), workspace, context)
            })
            .transpose()?;
        Ok(crate::analysis::acceptance(rank, condition, acceptance))
    }

    /// Solves the damped least-squares problem `min ‖A x − b‖₂² + λ ‖x‖₂²` with
    /// `λ = damping` and returns `x`.
    ///
    /// See [`Self::solve_damped_into`] for the method. The solve uses the default
    /// [`DirectSolveOptions`](crate::DirectSolveOptions).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_damped_into`], apart from workspace and
    ///   execution errors.
    /// - `Singular` if `R` has an exactly zero diagonal entry.
    pub fn solve_damped<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        damping: S,
    ) -> Result<Vector<S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        self.solve_damped_with_options(rhs, damping, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves the damped least-squares problem as [`Self::solve_damped`] does,
    /// checks the result against `acceptance` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// A solve that the checks reject returns `Ok` without a solution; see
    /// [Direct solves](crate#direct-solves).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::solve_damped_into`], apart from workspace and
    /// execution errors.
    pub fn solve_damped_with_options<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        damping: S,
        acceptance: &DirectSolveOptions<S>,
    ) -> Result<DirectSolveResult<Vector<S>, S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        let acceptance = acceptance.with_execution_options(
            acceptance
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let acceptance = &acceptance;
        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        );
        let mut output = Vector::zeros(self.cols)?;
        let report = self.solve_damped_into(
            rhs,
            damping,
            acceptance,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        let output = output;
        Ok(DirectSolveResult::new(output, report))
    }

    /// Solves `min ‖A x − b‖₂² + λ ‖x‖₂²` with `λ = damping ≥ 0` into `output`.
    ///
    /// This is the least-squares problem for `[A; √λ I] x ≈ [b; 0]`. The solve
    /// applies `Qᴴ` to `b`, then eliminates the rows `√λ I` against `R` with
    /// Givens rotations. The normal equations `(AᴴA + λI) x = Aᴴb`, whose
    /// condition number is the square of that of `A`, are never formed. It needs
    /// a real QR factorization with `m ≥ n` that passes the rank and condition
    /// checks of `acceptance`, as for [`Self::solve_with_options`]. O(mn + n³) time.
    /// `λ = 0` gives the ordinary least-squares solution.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) for LQ or for `m < n`.
    /// - `DimensionMismatch` if `rhs` does not have length `m` or `output`
    ///   does not have length `n`.
    /// - `NonFinite` if `damping` or `rhs` is not finite.
    /// - `InvalidInput` (`OutOfRange`) if `damping` is negative.
    /// - `DomainError` if `A` is empty and a condition threshold is set.
    /// - `NumericalInstability` (`Overflow`) if the regularized factor overflows.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # References
    ///
    /// J. J. Moré, The Levenberg–Marquardt algorithm: implementation and
    /// theory, in *Numerical Analysis*, Lecture Notes in Mathematics 630,
    /// Springer, 1978, 105–116.
    pub fn solve_damped_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        damping: S,
        acceptance: &DirectSolveOptions<S>,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        workspace.begin_attempt();
        let options = acceptance.execution_options();
        let options = &options;

        let context = context.into().with_options(options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let rhs = rhs.into();
        if self.orientation != HouseholderOrientation::Qr || self.rows < self.cols {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "damped solves require tall or square QR geometry",
            ));
        }
        crate::checked::expect_shape([self.rows], [rhs.len()])?;
        crate::checked::expect_shape([self.cols], [output.len()])?;
        crate::checked::validate_input(damping, "damping")?;
        if damping < <S as scioxide_core::AlgebraicScalar>::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("damping"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "damping must be nonnegative",
            ));
        }
        let request = HouseholderWorkspaceRequest::damped_solve(self.rows, self.cols, options)
            .with_residual(acceptance.residual())
            .for_kernels(kernels);
        let requirements = Self::workspace_requirements(&request)?;
        let _ = workspace.check(&requirements)?;
        let rhs = rhs.try_finite_dispatched(kernels.dispatch())?;
        workspace.try_reserve(&requirements)?;
        if self.rows.min(self.cols) == 0 && acceptance.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "householder_damped", 1),
                self.rows.max(self.cols) as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                let kernels = context;
                let report =
                    self.admit_solve(acceptance, workspace, kernels.scoped(&mut *solve))?;
                if report.termination() != DirectSolveTermination::Accepted {
                    return Ok(report);
                }
                let (rank, condition) = (report.rank(), report.condition());
                // Tikhonov least squares `min ||A x - b||^2 + d ||x||^2` is ordinary
                // least squares for the stacked system `[A; sqrt(d) I] x = [b; 0]`.
                // With `A P = Q R`, and because a permutation preserves `||x||`, this
                // reduces to `[R; sqrt(d) I] z = [Q^H b; 0]` with `x = P z`. The
                // damping rows are then folded into R one at a time with Givens
                // rotations, never forming `A^H A + d I`,
                // whose condition number would be the square of A's.
                crate::solve_support::admit_iteration(
                    Some(kernels.execution()),
                    Some(&mut *solve),
                    "damped solve budget exhausted",
                )?;
                workspace.with_prefix(
                    self.cols * self.cols + self.rows + 3 * self.cols,
                    |storage, _| {
                        let n = self.cols;
                        let (r, rest) = crate::workspace::split_mut(storage, n * n)?;
                        let (qtb, row) = crate::workspace::split_mut(rest, self.rows)?;
                        let (row, solution) = crate::workspace::split_mut(row, n)?;
                        let solution = &mut solution[..n];
                        qtb.iter_mut()
                            .zip(rhs.iter())
                            .for_each(|(out, value)| *out = value);
                        self.qr.apply_qh_vector_in_place(qtb)?;
                        trust_region::damped_step(
                            self.packed_factors(),
                            qtb,
                            None,
                            self.permutation(),
                            damping,
                            r,
                            solution,
                            row,
                            kernels.execution(),
                        )?;
                        qtb[..n].copy_from_slice(solution);
                        let residual = if acceptance.residual() == crate::Estimation::Estimate {
                            for (column, &original) in self.permutation().iter().enumerate() {
                                row[original] = qtb[column];
                            }
                            self.residual_columns(
                                crate::MatrixView::from(rhs.as_view()),
                                crate::MatrixView::from_slice(n, 1, row)?,
                                kernels,
                            )?
                        } else {
                            None
                        };
                        kernels.execution().check()?;
                        for (column, &original) in self.permutation().iter().enumerate() {
                            output.data[original * output.stride] = qtb[column];
                        }
                        Ok(DirectSolveReport::new(
                            DirectSolveTermination::Accepted,
                            rank,
                            condition,
                            residual,
                        ))
                    },
                )
            },
        )
    }
}

impl<S: NumericScalar> Default for HouseholderDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes for HouseholderDecomposition<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        self.retained_bytes()
    }
}
impl<S: NumericScalar> HouseholderDecomposition<S> {
    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        plan.matrix(&self.source, rows, cols)?;
        self.qr.action_storage(plan, rows, cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_reconstruction(matrix: &Matrix<f64>, orientation: HouseholderOrientation) {
        let factor = HouseholderDecomposition::factor(
            matrix,
            orientation,
            crate::HouseholderPivoting::Column,
        )
        .unwrap();
        let reconstructed = factor.reconstruct().unwrap();
        assert_eq!(
            (reconstructed.rows(), reconstructed.cols()),
            (matrix.rows(), matrix.cols())
        );
        for (&actual, &expected) in reconstructed.as_slice().iter().zip(matrix.as_slice()) {
            assert!((actual - expected).abs() < 1.0e-11);
        }
    }

    #[test]
    fn qr_and_lq_cover_tall_square_and_wide() {
        for (rows, cols) in [(4, 2), (3, 3), (2, 4)] {
            let matrix = Matrix::from_vec(
                rows,
                cols,
                (0..rows * cols)
                    .map(|index| (index * 7 % 11) as f64 - 3.0)
                    .collect(),
            )
            .unwrap();
            assert_reconstruction(&matrix, HouseholderOrientation::Qr);
            assert_reconstruction(&matrix, HouseholderOrientation::Lq);
        }
    }

    #[test]
    fn qr_and_lq_preserve_zero_extents() {
        for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
            let matrix = Matrix::zeros(rows, cols).unwrap();
            for orientation in [HouseholderOrientation::Qr, HouseholderOrientation::Lq] {
                assert_reconstruction(&matrix, orientation);
                let factor = HouseholderDecomposition::factor(
                    &matrix,
                    orientation,
                    crate::HouseholderPivoting::Column,
                )
                .unwrap();
                let rank = factor.analyze_rank(RankPolicy::Exact).unwrap();
                assert_eq!(rank.rank(), 0);
                assert_eq!(rank.maximum_rank(), 0);
            }
        }
    }

    #[test]
    fn wide_lq_returns_minimum_norm_solution() {
        let matrix = Matrix::from_vec(2, 3, vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let rhs = Vector::from_vec(vec![1.0, 1.0]);
        let factor = HouseholderDecomposition::factor(
            &matrix,
            HouseholderOrientation::Lq,
            crate::HouseholderPivoting::Column,
        )
        .unwrap();
        let result = factor
            .into_minimum_norm(
                &((crate::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                    .with_execution_options(crate::LinalgExecutionOptions::new()))
                .with_residual(crate::Estimation::Estimate),
            )
            .unwrap()
            .solve(&rhs)
            .unwrap();
        let solution = result.solution().unwrap();
        assert!((solution.view().get(0).unwrap() - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((solution.view().get(1).unwrap() - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((solution.view().get(2).unwrap() - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn rank_deficient_least_squares_keeps_rank_in_solve_metadata() {
        let matrix = Matrix::from_vec(3, 2, vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
        let rhs = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let factor = HouseholderDecomposition::factor(
            &matrix,
            HouseholderOrientation::Qr,
            crate::HouseholderPivoting::Column,
        )
        .unwrap();
        let result = factor
            .into_minimum_norm(
                &((crate::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                    .with_execution_options(crate::LinalgExecutionOptions::new()))
                .with_residual(crate::Estimation::Estimate),
            )
            .unwrap()
            .solve(&rhs)
            .unwrap();
        assert_eq!(result.report().rank().rank(), 1);
        assert_eq!(
            result.report().termination(),
            DirectSolveTermination::Accepted
        );
        assert!(result.solution().is_some());
        assert!(result.report().residual().unwrap().scaled_norm() < 1e-12);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod damped_tests {
    use super::*;
    #[test]
    fn regularized_pivoted_qr_matches_an_independent_exact_two_variable_solution() {
        let a = Matrix::from_vec(3, 2, alloc::vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let factor = HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            crate::HouseholderPivoting::Column,
        )
        .unwrap();
        let b = Vector::from_vec(alloc::vec![1.0, 2.0, 3.0]);
        let mut x = Vector::zeros(2).unwrap();
        let mut workspace = LinalgWorkspace::new();
        workspace
            .try_reserve(
                &crate::HouseholderDecomposition::<f64>::workspace_requirements(
                    &crate::HouseholderWorkspaceRequest::damped_solve(
                        3,
                        2,
                        &crate::LinalgExecutionOptions::new(),
                    ),
                )
                .unwrap(),
            )
            .unwrap();
        let report = factor
            .solve_damped_into(
                &b,
                1.0,
                &(crate::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon))
                    .with_execution_options(crate::LinalgExecutionOptions::new()),
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), DirectSolveTermination::Accepted);
        assert!((x.as_slice()[0] - 11.0 / 58.0).abs() < 1e-14);
        assert!((x.as_slice()[1] - 10.0 / 29.0).abs() < 1e-14);
    }
}
