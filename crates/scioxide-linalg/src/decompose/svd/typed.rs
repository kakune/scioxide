//! Values-only, thin and full SVD jobs and their reusable outputs.
//!
//! Thin and values jobs first use unpivoted blocked QR when the longer extent
//! is at least twice the nonzero shorter extent; wide inputs use the adjoint.
//! QR reflectors lift the thin vectors of R without forming a full Q.
//! Golub-Kahan reduction uses unblocked Householder updates through order 64
//! and blocked panels of width 32 above it, with Standard products. Contiguous
//! panel projections use native dots or four independent product lanes in scalar
//! execution. Values-only jobs use dqds without vector storage. Vector jobs use QR
//! through order 64 and bidiagonal divide and conquer above it (QR leaves of order
//! 32). Reflectors are applied one at a time on small cores and in compact-WY form
//! on larger ones. Full jobs also complete the orthogonal complement.
//! Input scaling and its restoration use exact powers of two. Results are staged
//! in the workspace and copied to the output only after the computation succeeds.

use super::bidiagonal_qr::bidiagonal_qr;
use crate::workspace::{RealRegion, ScalarRegion};
#[derive(Clone, Copy)]
struct PanelColumn<S> {
    start: usize,
    index: usize,
    panel_column: usize,
    coefficient: S,
}
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;

use core::cmp::Ordering;

use scioxide_core::backend::checked_mul;
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError,
};

use crate::dense::{Matrix, Vector};
use crate::kernel::{GemmConfig, GemmOp, LinalgWorkspace, gemm_workspace_with_context};
use crate::{KernelContext, LinalgExecutionOptions};

use super::{SvdDecomposition, spectrum_condition_estimate, spectrum_rank};
use super::{divide_conquer, dqds};
use crate::checked;
use crate::decompose::util::{copy_block, matrix_block, matrix_block_mut, reserve_vec};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::{ConditionEstimate, RankPolicy, RankReport};

// Default vector QR sweep budget per singular value. Typical inputs need
// only a few sweeps per value, so this acts as a safety net against
// non-convergence rather than a tuning parameter.
const DEFAULT_SWEEPS_PER_VALUE: usize = 30;

// xLABRD and compact-WY panel geometry is independent of scalar/ISA dispatch.
const PANEL_WIDTH: usize = 32;
// A direct QR solve avoids secular workspace and merging for small cores.
const VECTOR_QR_LIMIT: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reduction {
    Direct,
    QrFirst,
}
impl Reduction {
    fn for_job(rows: usize, cols: usize, job: SvdJob) -> Self {
        let order = rows.min(cols);
        if job != SvdJob::Full && order != 0 && rows.max(cols) / order >= 2 {
            Self::QrFirst
        } else {
            Self::Direct
        }
    }
}
// Back-transformation lends the xLABRD panels as compact-WY block storage.
const _: () = assert!(PANEL_WIDTH >= crate::reflector::BLOCK_WIDTH);

fn resolved_iteration_budget(
    options: crate::LinalgExecutionOptions,
    order: usize,
    vectors: bool,
) -> Result<usize, SciOxideError> {
    options.iteration_budget().map_or_else(
        || {
            let levels = if vectors && order > VECTOR_QR_LIMIT {
                1 + (usize::BITS - (order.div_ceil(divide_conquer::LEAF) - 1).leading_zeros())
                    as usize
            } else {
                1
            };
            checked_mul(
                checked_mul(
                    if vectors {
                        DEFAULT_SWEEPS_PER_VALUE
                    } else {
                        dqds::SWEEPS_PER_VALUE
                    },
                    levels,
                    scioxide_core::quantity!("svd_iteration_budget"),
                )?,
                order,
                scioxide_core::quantity!("svd_iteration_budget"),
            )
        },
        |budget| Ok(budget.get()),
    )
}

/// Singular values `σ₁ ≥ σ₂ ≥ … ≥ σₖ ≥ 0` of a dense `m × n` matrix, without singular vectors.
///
/// `k = min(m, n)`. The values follow the conventions of [`SvdDecomposition`]:
/// descending order, exact zeros stored as `+0.0`.
///
/// The matrix is scaled by a power of two and reduced to real bidiagonal form by
/// Householder bidiagonalization. When `max(m, n) ≥ 2 min(m, n)`, an unpivoted
/// Householder QR factorization comes first and only its triangular factor is
/// reduced. The bidiagonal singular values are then computed by the differential
/// qd algorithm with shifts (dqds), which needs `O(k)` scratch and no vectors.
/// The cost is `O(m n k)` time.
///
/// The reduction is normwise backward stable, so each singular value has an
/// absolute error of order `ε σ₁`. dqds computes the singular values of the
/// bidiagonal itself to high relative accuracy. The exception is values that are
/// subnormal after scaling, that is, below about `σ₁` times the smallest normal
/// number. They lose relative accuracy, and can round to zero, even when the
/// unscaled value is a normal number.
///
/// # References
///
/// - K. V. Fernando and B. N. Parlett, "Accurate singular values and differential
///   qd algorithms", *Numer. Math.* 67, 1994.
/// - B. N. Parlett and O. A. Marques, "An implementation of the dqds algorithm
///   (positive case)", *Linear Algebra Appl.* 309, 2000.
#[derive(Debug, Clone)]
#[must_use]
pub struct SvdValues<R: RealScalar> {
    singular_values: Vec<R>,
    rows: usize,
    cols: usize,
}

impl<R: RealScalar> SvdValues<R> {
    /// Returns the heap capacity of the singular-value storage in bytes.
    ///
    /// The count includes capacity left from earlier, larger results. It excludes the
    /// struct itself and allocator overhead.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn retained_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        scioxide_core::backend::checked_mul(
            self.singular_values.capacity(),
            core::mem::size_of::<R>(),
            scioxide_core::quantity!("retained_bytes"),
        )
    }

    /// Returns an empty result for use as the output of [`compute_into`](Self::compute_into).
    pub const fn empty() -> Self {
        Self {
            singular_values: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }

    /// Reserves storage for `count` singular values.
    ///
    /// The current values and shape are unchanged, also on failure.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be allocated.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use scioxide_linalg::SvdValues;
    /// let mut values = SvdValues::<f64>::empty();
    /// values.try_reserve(8)?;
    /// assert!(values.singular_values().is_empty());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(&mut self, count: usize) -> Result<(), SciOxideError> {
        reserve_vec(
            &mut self.singular_values,
            count,
            scioxide_core::quantity!("output.singular_values"),
        )
    }

    /// Returns the `min(m, n)` singular values in descending order.
    pub fn singular_values(&self) -> &[R] {
        &self.singular_values
    }

    /// Returns the number of rows `m` of the decomposed matrix.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns `n` of the decomposed matrix.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the numerical rank: the number of singular values above the policy threshold.
    ///
    /// # Errors
    ///
    /// The errors of `policy` when its threshold is evaluated.
    pub fn analyze_rank(&self, policy: RankPolicy<R>) -> Result<RankReport<R>, SciOxideError> {
        spectrum_rank(&self.singular_values, self.rows, self.cols, policy)
    }

    /// Returns the reciprocal two-norm condition number `σₖ / σ₁`.
    ///
    /// No rank threshold is applied: a singular matrix gives `0`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `min(m, n) = 0`, because the condition number is undefined.
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<R>, SciOxideError> {
        spectrum_condition_estimate(&self.singular_values)
    }
}

impl<R: RealScalar> Default for SvdValues<R> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Thin singular value decomposition `A = Uₖ Σₖ Vₖᴴ` of a dense `m × n` matrix.
///
/// With `k = min(m, n)`, `Uₖ` is `m × k` with orthonormal columns, `Σₖ` is the
/// `k × k` diagonal of singular values and `Vₖᴴ` is `k × n` with orthonormal rows.
/// They correspond to the first `k` columns of `U` and the first `k` rows of `Vᴴ` in
/// [`SvdDecomposition`], with the same ordering and phase conventions.
///
/// The method is that of [`SvdDecomposition`]. When `max(m, n) ≥ 2 min(m, n)`, an
/// unpivoted Householder QR factorization comes first, and only its `k × k`
/// triangular factor is bidiagonalized. The cost is `O(m n k)` time and `O(m n)`
/// memory, and the `m × m` or `n × n` complement is never formed.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, ThinSvdDecomposition};
///
/// let a = Matrix::from_vec(3, 2, vec![3.0_f64, 0.0, 0.0, 4.0, 0.0, 0.0])?;
/// let svd = ThinSvdDecomposition::compute(&a)?;
/// assert_eq!(svd.u_matrix().cols(), 2);
/// assert_eq!(svd.vt_matrix().rows(), 2);
/// // The singular values of this embedded diagonal matrix are 4 and 3.
/// assert!((svd.singular_values()[0] - 4.0).abs() < 1e-14);
/// assert!((svd.singular_values()[1] - 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ThinSvdDecomposition<S: NumericScalar> {
    u: Matrix<S>,
    singular_values: Vec<S::Real>,
    vt: Matrix<S>,
    rows: usize,
    cols: usize,
}

impl<S: NumericScalar> ThinSvdDecomposition<S> {
    /// Returns an empty decomposition to use as the output of [`compute_into`](Self::compute_into).
    pub const fn empty() -> Self {
        Self {
            u: Matrix::empty(),
            singular_values: Vec::new(),
            vt: Matrix::empty(),
            rows: 0,
            cols: 0,
        }
    }

    /// Reserves storage for the thin SVD of a `rows × cols` matrix.
    ///
    /// The current factors, values and shape are unchanged, also on failure. Storage
    /// that grew before a failure stays allocated.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a factor size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    ///
    /// # Examples
    ///
    ///
    /// ```
    /// use scioxide_linalg::ThinSvdDecomposition;
    /// let mut result = ThinSvdDecomposition::<f64>::empty();
    /// result.try_reserve(32, 4)?;
    /// assert_eq!(result.rows(), 0);
    /// assert!(result.singular_values().is_empty());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        let order = rows.min(cols);
        self.u.try_reserve_shape(rows, order)?;
        self.vt.try_reserve_shape(order, cols)?;
        reserve_vec(
            &mut self.singular_values,
            order,
            scioxide_core::quantity!("output.singular_values"),
        )
    }

    pub(crate) fn action_storage(
        &self,
        plan: &mut crate::workspace::accounting::StoragePlan,
        rows: usize,
        cols: usize,
    ) -> Result<(), SciOxideError> {
        let order = rows.min(cols);
        plan.matrix(&self.u, rows, order)?;
        plan.matrix(&self.vt, order, cols)?;
        plan.buffer(&self.singular_values, order)
    }

    pub(crate) fn reserve_result(&mut self, rows: usize, cols: usize) -> Result<(), SciOxideError> {
        let order = rows.min(cols);
        self.u.try_resize_flat(rows, order)?;
        self.vt.try_resize_flat(order, cols)?;
        let additional = order.saturating_sub(self.singular_values.len());
        scioxide_core::backend::try_reserve(
            &mut self.singular_values,
            additional,
            scioxide_core::quantity!("thin_svd_result"),
        )?;
        self.singular_values.resize(order, S::Real::ZERO);
        Ok(())
    }
    /// Returns the thin left singular vectors `Uₖ` (`m × k`, column `i` is `uᵢ`).
    pub const fn u_matrix(&self) -> &Matrix<S> {
        &self.u
    }

    /// Returns the `k = min(m, n)` singular values in descending order.
    pub fn singular_values(&self) -> &[S::Real] {
        &self.singular_values
    }

    /// Returns the thin right singular vectors as `Vₖᴴ` (`k × n`, row `i` is `vᵢᴴ`).
    pub const fn vt_matrix(&self) -> &Matrix<S> {
        &self.vt
    }

    /// Returns the number of rows `m` of the decomposed matrix.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns `n` of the decomposed matrix.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the numerical rank: the number of singular values above the policy threshold.
    ///
    /// # Errors
    ///
    /// The errors of `policy` when its threshold is evaluated.
    pub fn analyze_rank(
        &self,
        policy: RankPolicy<S::Real>,
    ) -> Result<RankReport<S::Real>, SciOxideError> {
        spectrum_rank(&self.singular_values, self.rows, self.cols, policy)
    }

    /// Returns the reciprocal two-norm condition number `σₖ / σ₁`.
    ///
    /// No rank threshold is applied: a singular matrix gives `0`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `min(m, n) = 0`, because the condition number is undefined.
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        spectrum_condition_estimate(&self.singular_values)
    }

    /// Solves the truncated, damped least-squares problem `min ‖A x − b‖₂² + λ ‖x‖₂²`.
    ///
    /// The solution and the threshold `τ` are as in [`SvdDecomposition::solve`], with
    /// `λ = damping`. With `λ = 0` this is the minimum-norm least-squares solution
    /// restricted to the singular vectors with `σᵢ > τ`.
    ///
    /// # Errors
    ///
    /// The errors of [`solve_into`](Self::solve_into).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{ThinSvdDecomposition, Matrix, RankPolicy};
    /// let a = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0])?;
    /// let factor = ThinSvdDecomposition::compute(&a)?;
    /// let x = factor.solve(&[2.0][..], RankPolicy::Exact, 0.0)?;
    /// // The minimum-norm solution of x₀ + x₁ = 2 is (1, 1).
    /// assert!(x.as_slice().iter().all(|value| (value - 1.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        rank_policy: RankPolicy<S::Real>,
        damping: S::Real,
    ) -> Result<Vector<S>, SciOxideError> {
        let mut output = Vector::zeros(self.cols)?;
        self.solve_into(
            rhs,
            rank_policy,
            damping,
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Solves the truncated, damped least-squares problem of [`solve`](Self::solve) into `output`.
    ///
    /// `rhs` has `rows` elements and `output` has `cols` elements. Both may be strided.
    /// Size the workspace with [`SvdWorkspaceRequest::solve`]. The solve counts as one
    /// iteration toward the iteration limit of `options`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs.len() != rows` or `output.len() != cols`.
    /// - `NonFinite` if `damping` or an entry of `rhs` is not finite.
    /// - `InvalidInput` if `damping < 0`, or the errors of `rank_policy`.
    /// - `NumericalInstability` if a filtered coefficient overflows, for example
    ///   `1/σᵢ` for a subnormal `σᵢ` kept by [`RankPolicy::Exact`].
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        rank_policy: RankPolicy<S::Real>,
        damping: S::Real,
        options: &crate::LinalgExecutionOptions,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.spectral_view()?.solve_into(
            rhs,
            rank_policy,
            damping,
            options,
            output,
            workspace,
            context,
        )
    }

    /// Returns the truncated Moore–Penrose pseudoinverse `A⁺ = Σ_{σᵢ > τ} σᵢ⁻¹ vᵢ uᵢᴴ` (`n × m`).
    ///
    /// The threshold `τ` comes from `rank_policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`pseudoinverse_into`](Self::pseudoinverse_into).
    pub fn pseudoinverse(
        &self,
        rank_policy: RankPolicy<S::Real>,
    ) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.cols, self.rows)?;
        self.pseudoinverse_into(
            rank_policy,
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Writes the truncated pseudoinverse of [`pseudoinverse`](Self::pseudoinverse) to `output`.
    ///
    /// `output` is `cols × rows`. Size the workspace with
    /// [`SvdWorkspaceRequest::pseudoinverse`]. Each of the `rows` output columns counts
    /// as one iteration toward the iteration limit of `options`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `cols × rows`.
    /// - The errors of `rank_policy`.
    /// - `NumericalInstability` if a reciprocal `1/σᵢ` overflows.
    /// - `ConvergenceFailure` if the iteration limit is smaller than `rows`.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn pseudoinverse_into<'scope, 'capability>(
        &self,
        rank_policy: RankPolicy<S::Real>,
        options: &crate::LinalgExecutionOptions,
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.spectral_view()?
            .pseudoinverse_into(rank_policy, options, output, workspace, context)
    }

    fn spectral_view(
        &self,
    ) -> Result<crate::decompose::svd::solve::SpectralView<'_, S>, SciOxideError> {
        Ok(crate::decompose::svd::solve::SpectralView {
            u: self.u.view(),
            values: &self.singular_values,
            vt: self
                .vt
                .view()
                .block(0, 0, self.singular_values.len(), self.cols)?,
        })
    }
}

impl<S: NumericScalar> Default for ThinSvdDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SvdJob {
    Values,
    Thin,
    Full,
}

/// Workspace request for an SVD, or for a solve or pseudoinverse from an existing SVD.
///
/// The request holds the matrix shape and the execution options. The job is chosen by
/// the type whose `workspace_requirements` receives the request: [`SvdValues`],
/// [`ThinSvdDecomposition`] or [`SvdDecomposition`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{LinalgExecutionOptions, SvdWorkspaceRequest, ThinSvdDecomposition};
/// let request = SvdWorkspaceRequest::new(5, 3, &LinalgExecutionOptions::new());
/// let requirements = ThinSvdDecomposition::<f64>::workspace_requirements(&request)?;
/// let workspace = ThinSvdDecomposition::<f64>::workspace(&request)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SvdWorkspaceRequest {
    rows: usize,
    cols: usize,
    options: LinalgExecutionOptions,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
    kind: SvdRequestKind,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SvdRequestKind {
    Compute,
    Solve,
    Pseudoinverse,
}

impl SvdWorkspaceRequest {
    /// Returns a request to compute the SVD of a `rows × cols` matrix.
    pub const fn new(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self {
            rows,
            cols,
            options: *options,
            workers: core::num::NonZeroUsize::MAX,
            packing: crate::workspace::Packing::Native,
            kind: SvdRequestKind::Compute,
        }
    }
    /// Returns a request for `solve_into` on the SVD of a `rows × cols` matrix.
    ///
    /// Only [`ThinSvdDecomposition`] and [`SvdDecomposition`] accept it; [`SvdValues`]
    /// returns `InvalidInput`.
    pub const fn solve(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self {
            kind: SvdRequestKind::Solve,
            ..Self::new(rows, cols, options)
        }
    }
    /// Returns a request for `pseudoinverse_into` on the SVD of a `rows × cols` matrix.
    ///
    /// Only [`ThinSvdDecomposition`] and [`SvdDecomposition`] accept it; [`SvdValues`]
    /// returns `InvalidInput`.
    pub const fn pseudoinverse(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self {
            kind: SvdRequestKind::Pseudoinverse,
            ..Self::new(rows, cols, options)
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
    pub(crate) fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }
    fn shapes(self, job: SvdJob) -> [[usize; 2]; 12] {
        let major = self.rows.max(self.cols);
        let order = self.rows.min(self.cols);
        let block = order.min(PANEL_WIDTH);
        let vectors = job != SvdJob::Values;
        let left = if job == SvdJob::Full { major } else { order };
        let wide = self.rows < self.cols && vectors;
        let qr = Reduction::for_job(self.rows, self.cols, job) == Reduction::QrFirst;
        [
            [if qr { order } else { major }, order],
            if vectors { [order, order] } else { [0, 0] },
            if vectors { [order, order] } else { [0, 0] },
            [major, block],
            [order, block],
            [major, block],
            [block, left],
            if vectors { [major, left] } else { [0, 0] },
            if vectors { [order, order] } else { [0, 0] },
            if wide { [order, order] } else { [0, 0] },
            if wide { [left, major] } else { [0, 0] },
            if qr { [major, order] } else { [0, 0] },
        ]
    }
    pub(super) fn requirements<S: NumericScalar>(
        self,
        job: SvdJob,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let order = self.rows.min(self.cols);
        let major = self.rows.max(self.cols);
        let block = order.min(PANEL_WIDTH);
        let left = if job == SvdJob::Full { major } else { order };
        if self.kind != SvdRequestKind::Compute {
            if job == SvdJob::Values {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("request"),
                    scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                    "a solve requires singular vectors",
                ));
            }
            let output = if self.kind == SvdRequestKind::Solve {
                if order == 0 { 0 } else { self.cols }
            } else {
                scioxide_core::backend::checked_product(
                    &[self.rows, self.cols],
                    scioxide_core::quantity!("svd_inverse"),
                )?
            };
            let column = if self.kind == SvdRequestKind::Pseudoinverse && output != 0 {
                self.cols
            } else {
                0
            };
            let mut requirements = crate::LinalgRequirements::new(self.options.byte_limit());
            *requirements.scalar_mut(ScalarRegion::Direct) = order
                .checked_add(output)
                .and_then(|n| n.checked_add(column))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("svd_solve_scratch"))
                })?;
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        resolved_iteration_budget(self.options, order, job != SvdJob::Values)?;
        let size = |shape: &[usize]| {
            scioxide_core::backend::checked_product(shape, scioxide_core::quantity!("svd_scratch"))
        };
        let add = |a: usize, b| {
            a.checked_add(b).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("svd_scratch"))
            })
        };
        let qr_tau = if Reduction::for_job(self.rows, self.cols, job) == Reduction::QrFirst {
            order
        } else {
            0
        };
        let mut requirements = crate::LinalgRequirements::new(self.options.byte_limit());
        *requirements.scalar_mut(ScalarRegion::Direct) = self.shapes(job).into_iter().try_fold(
            add(add(size(&[order, 2])?, size(&[block, 2])?)?, qr_tau)?,
            |total, shape| add(total, size(&shape)?),
        )?;
        *requirements.real_mut(RealRegion::Direct) = size(&[order, 3])?;
        if job == SvdJob::Values {
            *requirements.real_mut(RealRegion::Direct) =
                add(requirements.real(RealRegion::Direct), size(&[order, 4])?)?;
        }
        requirements.indices = order;
        if job != SvdJob::Values && order > VECTOR_QR_LIMIT {
            let dc = divide_conquer::layout::<S::Real>(
                order,
                &self.options,
                self.workers,
                self.packing,
            )?;
            *requirements.real_mut(RealRegion::Direct) =
                add(requirements.real(RealRegion::Direct), dc.reals)?;
            requirements.indices = add(requirements.indices, dc.indices)?;
        }

        for (m, k, n, structured) in [
            (major, block, order, false),
            (block, major, left, false),
            (major, block, left, false),
            (block, block, left, true),
        ] {
            let dense = if structured {
                crate::DenseWorkspaceRequest::structured(m, k, n, &self.options)
            } else {
                crate::DenseWorkspaceRequest::product(m, k, n, &self.options)
            };
            let mut dense = dense.with_worker_bound(self.workers);
            if matches!(self.packing, crate::workspace::Packing::Provider) {
                dense = dense.with_provider_packing();
            }
            requirements = requirements.max(&crate::kernel::product_requirements::<S>(&dense)?);
        }
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
}

impl<R: RealScalar> SvdValues<R> {
    /// Returns the workspace size for [`compute_into`](Self::compute_into) with scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` for a solve or pseudoinverse request, which needs singular vectors.
    /// - `SizeOverflow` if a size or the default iteration limit overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request options.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        request: &SvdWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        request.requirements::<S>(SvdJob::Values)
    }
    /// Allocates a workspace sized by [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// The errors of [`workspace_requirements`](Self::workspace_requirements), and
    /// workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn workspace<S: NumericScalar<Real = R>>(
        request: &SvdWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements::<S>(request)?)?;
        Ok(workspace)
    }
}
impl<S: NumericScalar> ThinSvdDecomposition<S> {
    /// Returns the workspace size for [`compute_into`](Self::compute_into), or for a solve or
    /// pseudoinverse request.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or the default iteration limit overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request options.
    pub fn workspace_requirements(
        request: &SvdWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        request.requirements::<S>(SvdJob::Thin)
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
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{LinalgExecutionOptions, SvdWorkspaceRequest, ThinSvdDecomposition};
    /// let request = SvdWorkspaceRequest::new(5, 3, &LinalgExecutionOptions::new());
    /// let requirements = ThinSvdDecomposition::<f64>::workspace_requirements(&request)?;
    /// let workspace = ThinSvdDecomposition::<f64>::workspace(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &SvdWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

pub(super) struct SvdScratch<'a, 'storage, S: NumericScalar> {
    dqds: &'a mut [S::Real],
    divide_conquer: Option<divide_conquer::Workspace<'a, S::Real>>,
    packed: ScratchMatrix<'a, S>,
    qr_packed: ScratchMatrix<'a, S>,
    qr_tau: &'a mut [S],
    tau_left: &'a mut [S],
    tau_right: &'a mut [S],
    diagonal: &'a mut [S::Real],
    superdiagonal: &'a mut [S::Real],
    core_u: ScratchMatrix<'a, S>,
    core_vt: ScratchMatrix<'a, S>,
    x_panel: ScratchMatrix<'a, S>,
    y_panel: ScratchMatrix<'a, S>,
    v_panel: ScratchMatrix<'a, S>,
    u_panel: ScratchMatrix<'a, S>,
    panel_temp_first: &'a mut [S],
    panel_temp_second: &'a mut [S],
    left_factor: ScratchMatrix<'a, S>,
    right_factor: ScratchMatrix<'a, S>,
    candidate_u: ScratchMatrix<'a, S>,
    candidate_vt: ScratchMatrix<'a, S>,
    candidate_values: &'a mut [S::Real],
    permutation: &'a mut [usize],
    gemm: &'a mut LinalgWorkspace<'storage, S>,
}

fn with_scratch<S: NumericScalar, T>(
    workspace: &mut LinalgWorkspace<'_, S>,
    request: SvdWorkspaceRequest,
    job: SvdJob,
    work: impl FnOnce(&mut SvdScratch<'_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.with_direct(|mut scalars, reals, indices, gemm| {
        let [
            packed,
            core_u,
            core_vt,
            x_panel,
            y_panel,
            v_panel,
            u_panel,
            left_factor,
            right_factor,
            candidate_u,
            candidate_vt,
            qr_packed,
        ] = request.shapes(job).map(|[rows, cols]| {
            let (data, rest) =
                crate::workspace::split_mut(core::mem::take(&mut scalars), rows * cols)?;
            scalars = rest;
            ScratchMatrix::new(data, rows, cols)
        });
        let order = request.rows.min(request.cols);
        let block = order.min(PANEL_WIDTH);
        let qr_len = if Reduction::for_job(request.rows, request.cols, job) == Reduction::QrFirst {
            order
        } else {
            0
        };
        let (qr_tau, scalars) = crate::workspace::split_mut(scalars, qr_len)?;
        let (tau_left, scalars) = crate::workspace::split_mut(scalars, order)?;
        let (tau_right, scalars) = crate::workspace::split_mut(scalars, order)?;
        let (panel_temp_first, scalars) = crate::workspace::split_mut(scalars, block)?;
        let (diagonal, reals) = crate::workspace::split_mut(reals, order)?;
        let (superdiagonal, reals) = crate::workspace::split_mut(reals, order)?;
        let (candidate_values, reals) = crate::workspace::split_mut(reals, order)?;
        let (dqds, reals) =
            crate::workspace::split_mut(reals, if job == SvdJob::Values { 4 * order } else { 0 })?;
        let (permutation, indices) = crate::workspace::split_mut(indices, order)?;
        let divide_conquer = if job != SvdJob::Values && order > VECTOR_QR_LIMIT {
            let layout = divide_conquer::layout::<S::Real>(
                order,
                &request.options,
                request.workers,
                request.packing,
            )?;
            Some(divide_conquer::Workspace::new(
                reals, indices, order, layout,
            )?)
        } else {
            None
        };
        let mut scratch = SvdScratch {
            dqds,
            divide_conquer,
            packed: packed?,
            qr_packed: qr_packed?,
            qr_tau,
            core_u: core_u?,
            core_vt: core_vt?,
            x_panel: x_panel?,
            y_panel: y_panel?,
            v_panel: v_panel?,
            u_panel: u_panel?,
            left_factor: left_factor?,
            right_factor: right_factor?,
            candidate_u: candidate_u?,
            candidate_vt: candidate_vt?,
            tau_left,
            tau_right,
            panel_temp_first,
            panel_temp_second: &mut scalars[..block],
            diagonal,
            superdiagonal,
            candidate_values,
            permutation,
            gemm,
        };
        work(&mut scratch)
    })
}

impl<R: RealScalar> SvdValues<R> {
    /// Computes the singular values of a dense matrix.
    ///
    /// The input may be any view, including strided and conjugated ones. dqds
    /// may use `100 k` transforms, including rejected shifts.
    ///
    /// # Errors
    ///
    /// The errors of [`compute_into`](Self::compute_into).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, SvdValues};
    /// let a = Matrix::from_rows(&[[3.0_f64, 0.0], [0.0, 1e-20]])?;
    /// let values = SvdValues::compute(&a)?;
    /// // A diagonal matrix has its absolute diagonal entries as singular values;
    /// // the tiny one keeps full relative accuracy.
    /// assert!((values.singular_values()[1] / 1e-20 - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compute<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::compute_with_options(matrix, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn compute_with_options<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        let mut workspace = LinalgWorkspace::new();
        Self::compute_into(
            matrix,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the singular values of a dense matrix into `output`.
    ///
    /// Size the workspace with [`SvdWorkspaceRequest::new`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry of `matrix` is NaN or infinite.
    /// - `ConvergenceFailure` if dqds exceeds the iteration limit.
    /// - `NumericalInstability` if undoing the input scaling overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn compute_into<'scope, 'm, 'a, S: NumericScalar<Real = R>>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(options)?;
        let context = scope.kernel();
        let matrix = matrix.into();
        let options = *options;

        let finite = prepare(matrix, workspace, SvdJob::Values, options, context)?;
        reserve_vec(
            &mut output.singular_values,
            matrix.rows().min(matrix.cols()),
            scioxide_core::quantity!("output.singular_values"),
        )?;
        compute_scoped::<S, false, false, _>(finite, workspace, options, scope, |scratch| {
            publish_values(output, scratch, matrix.rows(), matrix.cols())
        })?;
        Ok(())
    }
}

impl<S: NumericScalar> ThinSvdDecomposition<S> {
    /// Computes the thin SVD of a dense matrix.
    ///
    /// The input may be any view, including strided and conjugated ones.
    ///
    /// # Errors
    ///
    /// The errors of [`compute_into`](Self::compute_into).
    pub fn compute<'m>(matrix: impl Into<crate::MatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::compute_with_options(matrix, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn compute_with_options<'m>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        let mut workspace = LinalgWorkspace::new();
        Self::compute_into(
            matrix,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the thin SVD of a dense matrix into `output`.
    ///
    /// `output` grows to the shape of `matrix` as needed. Size the workspace with
    /// [`SvdWorkspaceRequest::new`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry of `matrix` is NaN or infinite.
    /// - `ConvergenceFailure` if bidiagonal QR or divide and conquer exceeds the
    ///   iteration limit.
    /// - `NumericalInstability` if undoing the input scaling overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn compute_into<'scope, 'm, 'a>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(options)?;
        let context = scope.kernel();
        let matrix = matrix.into();
        let options = *options;

        let rows = matrix.rows();
        let cols = matrix.cols();
        let order = rows.min(cols);
        let finite = prepare(matrix, workspace, SvdJob::Thin, options, context)?;
        output.u.try_reserve_shape(rows, order)?;
        output.vt.try_reserve_shape(order, cols)?;
        reserve_vec(
            &mut output.singular_values,
            order,
            scioxide_core::quantity!("output.singular_values"),
        )?;
        compute_scoped::<S, true, false, _>(finite, workspace, options, scope, |scratch| {
            publish_thin(output, scratch, rows, cols)
        })?;
        Ok(())
    }
}

fn publish_values<S: NumericScalar>(
    output: &mut SvdValues<S::Real>,
    workspace: &mut SvdScratch<'_, '_, S>,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    output.singular_values.clear();
    output
        .singular_values
        .extend_from_slice(workspace.candidate_values);
    output.rows = rows;
    output.cols = cols;
    Ok(())
}

fn publish_thin<S: NumericScalar>(
    output: &mut ThinSvdDecomposition<S>,
    workspace: &mut SvdScratch<'_, '_, S>,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    output
        .u
        .try_resize_flat(workspace.candidate_u.rows(), workspace.candidate_u.cols())?;
    output
        .vt
        .try_resize_flat(workspace.candidate_vt.rows(), workspace.candidate_vt.cols())?;
    output
        .u
        .as_mut_slice()
        .copy_from_slice(workspace.candidate_u.as_slice());
    output
        .vt
        .as_mut_slice()
        .copy_from_slice(workspace.candidate_vt.as_slice());
    output.singular_values.clear();
    output
        .singular_values
        .extend_from_slice(workspace.candidate_values);
    output.rows = rows;
    output.cols = cols;
    Ok(())
}

pub(super) fn publish_full<S: NumericScalar>(
    output: &mut SvdDecomposition<S>,
    workspace: &mut SvdScratch<'_, '_, S>,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    output
        .u
        .try_resize_flat(workspace.candidate_u.rows(), workspace.candidate_u.cols())?;
    output
        .vt
        .try_resize_flat(workspace.candidate_vt.rows(), workspace.candidate_vt.cols())?;
    output
        .u
        .as_mut_slice()
        .copy_from_slice(workspace.candidate_u.as_slice());
    output
        .vt
        .as_mut_slice()
        .copy_from_slice(workspace.candidate_vt.as_slice());
    output.singular_values.clear();
    output
        .singular_values
        .extend_from_slice(workspace.candidate_values);
    output.rows = rows;
    output.cols = cols;
    Ok(())
}

#[inline(always)]
pub(super) fn prepare<'a, S: NumericScalar>(
    matrix: crate::MatrixView<'a, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    job: SvdJob,
    options: LinalgExecutionOptions,
    context: KernelContext<'_>,
) -> Result<crate::FiniteMatrixView<'a, S>, SciOxideError> {
    context.execution().check()?;
    let request =
        SvdWorkspaceRequest::new(matrix.rows(), matrix.cols(), &options).for_kernels(context);
    let requirements = request.requirements::<S>(job)?;
    let finite = matrix.try_finite_dispatched(context.dispatch())?;
    workspace.try_reserve(&requirements)?;
    Ok(finite)
}

pub(super) fn compute_scoped<S: NumericScalar, const VECTORS: bool, const FULL: bool, T>(
    matrix: crate::FiniteMatrixView<'_, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    options: LinalgExecutionOptions,
    context: crate::LinalgExecutionContext<'_, '_>,
    publish: impl FnOnce(&mut SvdScratch<'_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "svd", 1),
        ((matrix.rows() as u128) << 64) | matrix.cols() as u128,
        0,
    );
    let job = if FULL {
        SvdJob::Full
    } else if VECTORS {
        SvdJob::Thin
    } else {
        SvdJob::Values
    };
    let request = SvdWorkspaceRequest::new(matrix.rows(), matrix.cols(), &options);
    workspace.run_scope(
        context,
        identity,
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, kernels, solve| {
            // Carve with the worker and packing bounds the reservation used.
            with_scratch(workspace, request.for_kernels(kernels), job, |scratch| {
                compute::<S, VECTORS, FULL>(matrix, scratch, options, kernels, solve)?;
                let _ = solve.counts().total()?;
                publish(scratch)
            })
        },
    )
}

pub(crate) fn values_candidate<S: NumericScalar, T>(
    matrix: crate::FiniteMatrixView<'_, S>,
    options: LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: crate::LinalgExecutionContext<'_, '_>,
    consume: impl FnOnce(&[S::Real]) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.begin_attempt();
    context.execution().check()?;
    let request = SvdWorkspaceRequest::new(matrix.rows(), matrix.cols(), &options)
        .for_kernels(context.kernel());
    workspace.try_reserve(&request.requirements::<S>(SvdJob::Values)?)?;
    compute_scoped::<S, false, false, T>(matrix, workspace, options, context, |scratch| {
        consume(scratch.candidate_values)
    })
}

pub(crate) fn thin_candidate<S: NumericScalar, T>(
    matrix: crate::FiniteMatrixView<'_, S>,
    options: LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: crate::LinalgExecutionContext<'_, '_>,
    consume: impl FnOnce(super::solve::SpectralView<'_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.begin_attempt();
    context.execution().check()?;
    let request = SvdWorkspaceRequest::new(matrix.rows(), matrix.cols(), &options)
        .for_kernels(context.kernel());
    workspace.try_reserve(&request.requirements::<S>(SvdJob::Thin)?)?;
    compute_scoped::<S, true, false, T>(matrix, workspace, options, context, |scratch| {
        consume(super::solve::SpectralView {
            u: scratch.candidate_u.try_view()?,
            values: scratch.candidate_values,
            vt: scratch.candidate_vt.try_view()?,
        })
    })
}

pub(crate) fn full_candidate<S: NumericScalar, T>(
    matrix: crate::FiniteMatrixView<'_, S>,
    options: LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: crate::LinalgExecutionContext<'_, '_>,
    consume: impl FnOnce(super::solve::SpectralView<'_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.begin_attempt();
    context.execution().check()?;
    let request = SvdWorkspaceRequest::new(matrix.rows(), matrix.cols(), &options)
        .for_kernels(context.kernel());
    workspace.try_reserve(&request.requirements::<S>(SvdJob::Full)?)?;
    compute_scoped::<S, true, true, T>(matrix, workspace, options, context, |scratch| {
        consume(super::solve::SpectralView {
            u: scratch.candidate_u.try_view()?,
            values: scratch.candidate_values,
            vt: scratch.candidate_vt.try_view()?,
        })
    })
}

fn compute<'capability, S: NumericScalar, const VECTORS: bool, const FULL: bool>(
    matrix: crate::FiniteMatrixView<'_, S>,
    workspace: &mut SvdScratch<'_, '_, S>,
    options: LinalgExecutionOptions,
    context: KernelContext<'capability>,
    solve: &mut scioxide_core::SolveContext<'_, 'capability>,
) -> Result<(), SciOxideError> {
    let context = context.with_accuracy(scioxide_core::Accuracy::Standard);
    let rows = matrix.rows();
    let cols = matrix.cols();
    let order = rows.min(cols);
    let qr = !workspace.qr_tau.is_empty();
    let scale = if qr {
        let scale = pack_oriented_scaled(matrix, &mut workspace.qr_packed)?;
        let maximum =
            crate::FiniteMatrixView::from_workspace(workspace.qr_packed.try_view()?).maximum;
        crate::decompose::qr::factor_unpivoted(
            crate::decompose::qr::UnpivotedQr {
                packed: &mut workspace.qr_packed,
                tau: workspace.qr_tau,
                panel: &mut workspace.x_panel,
                blocks: crate::reflector::CompactWy {
                    vectors: &mut workspace.v_panel,
                    triangular: &mut workspace.y_panel,
                    work: &mut workspace.u_panel,
                },
            },
            maximum,
            options.execution_policy(),
            workspace.gemm,
            context,
        )?;
        workspace.packed.try_resize_flat(order, order)?;
        for row in 0..order {
            for col in 0..order {
                workspace.packed.as_mut_slice()[row * order + col] = if row <= col {
                    workspace.qr_packed.get_element(row, col)
                } else {
                    S::ZERO
                };
            }
        }
        scale
    } else {
        pack_oriented_scaled(matrix, &mut workspace.packed)?
    };
    context.execution().check()?;
    if order <= VECTOR_QR_LIMIT {
        bidiagonalize_small(workspace, context)?;
    } else {
        bidiagonalize_compact(workspace, options.execution_policy(), context)?;
    }
    validate_bidiagonal(workspace.diagonal, workspace.superdiagonal)?;
    let budget = resolved_iteration_budget(options, order, VECTORS)?;
    if !VECTORS {
        dqds::solve(
            workspace.diagonal,
            workspace.superdiagonal,
            workspace.dqds,
            budget,
            solve,
        )?;
    } else if let Some(dc) = workspace.divide_conquer.as_mut() {
        dc.solve(
            workspace.diagonal,
            workspace.superdiagonal,
            budget,
            options.execution_policy(),
            context,
            solve,
        )?;
        for (target, &value) in workspace
            .core_u
            .as_mut_slice()
            .iter_mut()
            .zip(dc.u.as_slice())
        {
            *target = S::from_real(value);
        }
        for (target, &value) in workspace
            .core_vt
            .as_mut_slice()
            .iter_mut()
            .zip(dc.vt.as_slice())
        {
            *target = S::from_real(value);
        }
    } else {
        set_identity(&mut workspace.core_u);
        set_identity(&mut workspace.core_vt);
        bidiagonal_qr::<S>(
            workspace.diagonal,
            workspace.superdiagonal,
            &mut workspace.core_u,
            &mut workspace.core_vt,
            budget,
            solve,
        )?;
    }
    validate_bidiagonal(workspace.diagonal, workspace.superdiagonal)?;
    normalize_and_sort::<S>(
        workspace.diagonal,
        VECTORS.then_some(&mut workspace.core_vt),
        workspace.permutation,
    );
    workspace
        .candidate_values
        .iter_mut()
        .zip(workspace.permutation.iter().copied())
        .try_for_each(|(target, source)| {
            // Restoring the input scale of finite normalized values can only overflow.
            *target = crate::checked::finite_candidate(workspace.diagonal[source].scalbn(scale))?;
            Ok::<(), SciOxideError>(())
        })?;
    if VECTORS {
        let left_columns = if FULL { rows.max(cols) } else { order };
        form_vectors(
            matrix,
            workspace,
            left_columns,
            options.execution_policy(),
            context,
        )?;
        canonicalize_vectors(
            &mut workspace.candidate_u,
            &mut workspace.candidate_vt,
            &mut workspace.tau_left[..order],
            &mut workspace.diagonal[..order],
        );
        crate::checked::validate_values(
            workspace.candidate_u.as_slice(),
            NonFiniteOrigin::Output,
            "left singular-vector factor contains a non-finite component",
        )?;
        crate::checked::validate_values(
            workspace.candidate_vt.as_slice(),
            NonFiniteOrigin::Output,
            "right singular-vector factor contains a non-finite component",
        )?;
    }
    Ok(())
}

fn pack_oriented_scaled<S: NumericScalar>(
    matrix: crate::FiniteMatrixView<'_, S>,
    packed: &mut impl MatrixStorage<S>,
) -> Result<i32, SciOxideError> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let exponent = crate::kernel::standard::exponent(matrix.maximum)?;
    let oriented = if rows >= cols {
        matrix
    } else {
        matrix.adjoint()
    };
    packed.try_resize_flat(oriented.rows(), oriented.cols())?;
    if oriented.cols() == 0 {
        return Ok(exponent);
    }
    let multiplier = S::Real::ONE.scalbn(-exponent);
    for row in 0..oriented.rows() {
        for col in 0..oriented.cols() {
            let value = oriented.as_view().get_element(row, col);
            packed.as_mut_slice()[row * oriented.cols() + col] =
                crate::decompose::util::scale_with_power(value, multiplier, -exponent);
        }
    }
    Ok(exponent)
}

fn validate_bidiagonal<R: RealScalar>(
    diagonal: &[R],
    superdiagonal: &[R],
) -> Result<(), SciOxideError> {
    crate::checked::validate_values(
        diagonal,
        NonFiniteOrigin::Intermediate,
        "bidiagonal coefficient is not finite",
    )?;
    crate::checked::validate_values(
        superdiagonal,
        NonFiniteOrigin::Intermediate,
        "bidiagonal coefficient is not finite",
    )
}

/// Unblocked Householder bidiagonalization of a small scaled core. Small cores
/// skip the deferred panel updates of the blocked reduction. The packed
/// reflectors have the same representation as in the blocked reduction.
fn bidiagonalize_small<S: NumericScalar>(
    workspace: &mut SvdScratch<'_, '_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let SvdScratch {
        packed,
        tau_left,
        tau_right,
        diagonal,
        superdiagonal,
        v_panel: vector,
        u_panel: work,
        ..
    } = workspace;
    let (rows, cols) = (packed.rows(), packed.cols());
    superdiagonal.fill(S::Real::ZERO);
    if cols == 0 {
        return Ok(());
    }
    // Orthogonal transformations preserve the Frobenius norm. The entry
    // maximum times 2(m+n) bounds it, including an R core lifted from tall QR.
    let maximum = crate::FiniteMatrixView::from_workspace(packed.try_view()?).maximum
        * S::Real::TWO
        * (crate::checked::rounded_count::<S::Real>(rows)?
            + crate::checked::rounded_count::<S::Real>(cols)?);
    for index in 0..cols {
        context.execution().check()?;
        vector.try_resize_flat(rows - index, 1)?;
        for (value, row) in vector.as_mut_slice().iter_mut().zip(index..rows) {
            *value = packed.get_element(row, index);
        }
        let h =
            crate::reflector::generate(crate::VectorViewMut::from_slice(vector.as_mut_slice()))?;
        tau_left[index] = h.tau;
        diagonal[index] = h.beta;
        crate::reflector::apply_panel(
            crate::FiniteVectorView::from_validated(
                crate::VectorView::from_slice(vector.as_slice()),
                S::Real::TWO,
            ),
            h.tau.conj(),
            matrix_block_mut(packed, index..rows, index + 1..cols)?,
            maximum,
            work,
        )?;
        for (&value, row) in vector.as_slice().iter().zip(index..rows) {
            packed.as_mut_slice()[row * cols + index] = value;
        }
        packed.as_mut_slice()[index * (cols + 1)] = S::from_real(h.beta);
        tau_right[index] = S::ZERO;
        if index + 1 == cols {
            continue;
        }
        vector.try_resize_flat(cols - index - 1, 1)?;
        for (value, col) in vector.as_mut_slice().iter_mut().zip(index + 1..cols) {
            *value = packed.get_element(index, col).conj();
        }
        let h =
            crate::reflector::generate(crate::VectorViewMut::from_slice(vector.as_mut_slice()))?;
        tau_right[index] = h.tau;
        superdiagonal[index] = h.beta;
        // A is in the bounded binary frame and v has magnitude at most two.
        // Contiguous rows let the right rank-one update avoid range scans and
        // strided matrix-view lookups for each scalar.
        if h.tau != S::ZERO {
            for row in index + 1..rows {
                let values = &mut packed.as_mut_slice()[row * cols + index + 1..(row + 1) * cols];
                let v = vector.as_slice();
                let dot =
                    crate::kernel::arch::dot(values, v, context.dispatch())?.unwrap_or_else(|| {
                        values
                            .iter()
                            .zip(v)
                            .fold(S::ZERO, |sum, (&a, &b)| sum + a * b)
                    });
                let dot = h.tau * dot;
                values
                    .iter_mut()
                    .zip(v)
                    .for_each(|(value, &v)| *value = *value - dot * v.conj());
            }
        }
        vector
            .as_mut_slice()
            .iter_mut()
            .for_each(|value| *value = value.conj());
        for (&value, col) in vector.as_slice().iter().zip(index + 1..cols) {
            packed.as_mut_slice()[index * cols + col] = value;
        }
        packed.as_mut_slice()[index * cols + index + 1] = S::from_real(h.beta);
    }
    Ok(())
}

fn bidiagonalize_compact<S: NumericScalar>(
    workspace: &mut SvdScratch<'_, '_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let SvdScratch {
        packed,
        tau_left,
        tau_right,
        diagonal,
        superdiagonal,
        x_panel: x,
        y_panel: y,
        v_panel,
        u_panel,
        panel_temp_first,
        panel_temp_second,
        gemm: gemm_workspace,
        ..
    } = workspace;
    // Blocked Golub-Kahan bidiagonalization `A = Q_L B Q_R^H`.
    // Within a panel, each column and row reflector is
    // generated from entries that are brought up to date on the fly using
    // the panel's X and Y matrices; the trailing matrix is then updated once
    // per panel as `A22 -= V Y^H + X U^H` with two GEMMs. Reflector vectors
    // are stored in place below the diagonal (left) and right of the
    // superdiagonal (right), with their unit leading entries written into
    // the diagonal/superdiagonal slots while the panel is active; the real
    // bidiagonal values are restored into those slots after each panel.
    let rows = packed.rows();
    let cols = packed.cols();
    // The last slot has no reflector; consumers read it as the zero that
    // closes the bidiagonal.
    superdiagonal.fill(S::Real::ZERO);
    let mut start = 0;
    while start < cols {
        context.execution().check()?;
        let end = (start + PANEL_WIDTH).min(cols);
        let width = end - start;
        x.try_resize_flat(rows, width)?;
        y.try_resize_flat(cols, width)?;
        u_panel.try_resize_flat(width, cols)?;
        x.as_mut_slice().fill(S::ZERO);
        y.as_mut_slice().fill(S::ZERO);

        for index in start..end {
            let panel_column = index - start;
            update_panel_column(packed, x, y, start, index, panel_column);
            let (beta, tau) = {
                let columns = packed.cols();
                let rows = packed.rows();
                let h = crate::reflector::generate(crate::VectorViewMut::from_slice_with_stride(
                    &mut packed.as_mut_slice()[index * (columns + 1)..],
                    rows - index,
                    columns,
                )?)?;
                (h.beta, h.tau)
            };
            diagonal[index] = beta;
            tau_left[index] = tau;
            if index + 1 == cols {
                tau_right[index] = S::ZERO;
                continue;
            }

            compute_y_column(
                packed,
                x,
                y,
                PanelColumn {
                    start,
                    index,
                    panel_column,
                    coefficient: tau,
                },
                panel_temp_first,
                panel_temp_second,
                &mut u_panel.as_mut_slice()[..cols],
            );
            update_panel_row(packed, x, y, start, index, panel_column);
            let (beta, tau) = {
                let columns = packed.cols();
                let h = crate::reflector::generate(crate::VectorViewMut::from_slice(
                    &mut packed.as_mut_slice()[index * columns + index + 1..(index + 1) * columns],
                ))?;
                (h.beta, h.tau)
            };
            superdiagonal[index] = beta;
            tau_right[index] = tau;
            compute_x_column(
                packed,
                x,
                y,
                PanelColumn {
                    start,
                    index,
                    panel_column,
                    coefficient: tau,
                },
                panel_temp_first,
                panel_temp_second,
                context,
            )?;
            conjugate_row_tail(packed, index);
        }

        if end < cols {
            copy_block(matrix_block(packed, end..rows, start..end)?, v_panel)?;
            let first_update = GemmConfig {
                right_op: GemmOp::Adjoint,
                alpha: -S::ONE,
                beta: S::ONE,
                ..GemmConfig::ordinary()
            };
            gemm_workspace_with_context(
                crate::kernel::WorkspaceGemmProblem {
                    left: crate::FiniteMatrixView::from_workspace(v_panel.try_view()?),
                    right: crate::FiniteMatrixView::from_workspace(matrix_block(
                        y,
                        end..cols,
                        0..width,
                    )?),
                    output: matrix_block_mut(packed, end..rows, end..cols)?,
                    config: first_update,
                },
                policy,
                gemm_workspace,
                context,
            )?;
            copy_block(matrix_block(packed, start..end, end..cols)?, u_panel)?;
            let second_update = GemmConfig {
                alpha: -S::ONE,
                beta: S::ONE,
                ..GemmConfig::ordinary()
            };
            gemm_workspace_with_context(
                crate::kernel::WorkspaceGemmProblem {
                    left: crate::FiniteMatrixView::from_workspace(matrix_block(
                        x,
                        end..rows,
                        0..width,
                    )?),
                    right: crate::FiniteMatrixView::from_workspace(u_panel.try_view()?),
                    output: matrix_block_mut(packed, end..rows, end..cols)?,
                    config: second_update,
                },
                policy,
                gemm_workspace,
                context,
            )?;
        }

        for index in start..end {
            packed.as_mut_slice()[index * cols + index] = S::from_real(diagonal[index]);
            if index + 1 < cols {
                packed.as_mut_slice()[index * cols + index + 1] =
                    S::from_real(superdiagonal[index]);
            }
        }
        start = end;
    }
    Ok(())
}

fn update_panel_column<S: NumericScalar>(
    packed: &mut impl MatrixStorage<S>,
    x: &impl MatrixStorage<S>,
    y: &impl MatrixStorage<S>,
    start: usize,
    index: usize,
    panel_column: usize,
) {
    let rows = packed.rows();
    let cols = packed.cols();
    for row in index..rows {
        let correction = (0..panel_column).fold(S::ZERO, |sum, prior| {
            sum + packed.get_element(row, start + prior) * y.get_element(index, prior).conj()
                + x.get_element(row, prior) * packed.get_element(start + prior, index)
        });
        let offset = row * cols + index;
        packed.as_mut_slice()[offset] = packed.as_slice()[offset] - correction;
    }
}

fn compute_y_column<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    x: &impl MatrixStorage<S>,
    y: &mut impl MatrixStorage<S>,
    column: PanelColumn<S>,
    first_temporary: &mut [S],
    second_temporary: &mut [S],
    product: &mut [S],
) {
    let PanelColumn {
        start,
        index,
        panel_column,
        coefficient,
    } = column;

    let rows = packed.rows();
    let cols = packed.cols();
    first_temporary[..panel_column].fill(S::ZERO);
    second_temporary[..panel_column].fill(S::ZERO);
    product[index + 1..cols].fill(S::ZERO);
    // Accumulate every projection while its source row is in cache. Each
    // sum keeps ascending row order, and independent columns vectorize.
    for row in index..rows {
        let reflector = packed.get_element(row, index);
        let packed_row = &packed.as_slice()[row * cols..(row + 1) * cols];
        let x_row = &x.as_slice()[row * x.cols()..(row + 1) * x.cols()];
        for (sum, &value) in first_temporary[..panel_column]
            .iter_mut()
            .zip(&packed_row[start..start + panel_column])
        {
            *sum = *sum + value.conj() * reflector;
        }
        for (sum, &value) in second_temporary[..panel_column].iter_mut().zip(x_row) {
            *sum = *sum + value.conj() * reflector;
        }
        for (sum, &value) in product[index + 1..cols]
            .iter_mut()
            .zip(&packed_row[index + 1..cols])
        {
            *sum = *sum + value.conj() * reflector;
        }
    }
    for (column, &direct) in product.iter().enumerate().take(cols).skip(index + 1) {
        let first_correction = (0..panel_column).fold(S::ZERO, |sum, prior| {
            sum + y.get_element(column, prior) * first_temporary[prior]
        });
        let second_correction = (0..panel_column).fold(S::ZERO, |sum, prior| {
            sum + packed.get_element(start + prior, column).conj() * second_temporary[prior]
        });
        let offset = column * y.cols() + panel_column;
        y.as_mut_slice()[offset] = (direct - first_correction - second_correction) * coefficient;
    }
}

fn update_panel_row<S: NumericScalar>(
    packed: &mut impl MatrixStorage<S>,
    x: &impl MatrixStorage<S>,
    y: &impl MatrixStorage<S>,
    start: usize,
    index: usize,
    panel_column: usize,
) {
    let cols = packed.cols();
    for column in index + 1..cols {
        let first_correction = (0..=panel_column).fold(S::ZERO, |sum, prior| {
            let left = if prior == panel_column {
                S::ONE
            } else {
                packed.get_element(index, start + prior)
            };
            sum + y.get_element(column, prior) * left.conj()
        });
        let second_correction = (0..panel_column).fold(S::ZERO, |sum, prior| {
            sum + packed.get_element(start + prior, column).conj()
                * x.get_element(index, prior).conj()
        });
        let offset = index * cols + column;
        // The row is stored conjugated (xLACGV convention) so the row
        // reflector can be generated by the same column-style routine;
        // `conjugate_row_tail` conjugates the stored reflector once X has
        // been computed, leaving `conj(v)` in the row.
        packed.as_mut_slice()[offset] =
            packed.as_slice()[offset].conj() - first_correction - second_correction;
    }
}

fn compute_x_column<S: NumericScalar>(
    packed: &impl MatrixStorage<S>,
    x: &mut impl MatrixStorage<S>,
    y: &impl MatrixStorage<S>,
    column: PanelColumn<S>,
    first_temporary: &mut [S],
    second_temporary: &mut [S],
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let PanelColumn {
        start,
        index,
        panel_column,
        coefficient,
    } = column;

    let rows = packed.rows();
    let cols = packed.cols();
    first_temporary[..panel_column + 1].fill(S::ZERO);
    for column in index + 1..cols {
        let reflector = packed.get_element(index, column);
        let y_row = &y.as_slice()[column * y.cols()..(column + 1) * y.cols()];
        for (sum, &value) in first_temporary[..panel_column + 1].iter_mut().zip(y_row) {
            *sum = *sum + value.conj() * reflector;
        }
    }
    for (prior, temporary) in second_temporary.iter_mut().enumerate().take(panel_column) {
        *temporary = (index + 1..cols).fold(S::ZERO, |sum, column| {
            let reflector = if column == index + 1 {
                S::ONE
            } else {
                packed.get_element(index, column)
            };
            sum + packed.get_element(start + prior, column) * reflector
        });
    }
    for row in index + 1..rows {
        // The generated row reflector has an explicit unit leading entry.
        // Reuse the verified Standard dot leaf for these contiguous products.
        let input = &packed.as_slice()[row * cols + index + 1..(row + 1) * cols];
        let reflector = &packed.as_slice()[index * cols + index + 1..(index + 1) * cols];
        let direct = crate::kernel::arch::dot(input, reflector, context.dispatch())?
            .unwrap_or_else(|| {
                // Four independent ordinary-product lanes avoid a serial
                // dependency chain.
                // The panel's normalized range and Standard error bound also
                // cover this summation order.
                let mut left = input.chunks_exact(4);
                let mut right = reflector.chunks_exact(4);
                let sums = left
                    .by_ref()
                    .zip(right.by_ref())
                    .fold([S::ZERO; 4], |s, (a, b)| {
                        [
                            s[0] + a[0] * b[0],
                            s[1] + a[1] * b[1],
                            s[2] + a[2] * b[2],
                            s[3] + a[3] * b[3],
                        ]
                    });
                left.remainder().iter().zip(right.remainder()).fold(
                    (sums[0] + sums[1]) + (sums[2] + sums[3]),
                    |sum, (&a, &b)| sum + a * b,
                )
            });
        let direct = checked::finite_candidate(direct)?;
        let first_correction = (0..=panel_column).fold(S::ZERO, |sum, prior| {
            sum + packed.get_element(row, start + prior) * first_temporary[prior]
        });
        let second_correction = (0..panel_column).fold(S::ZERO, |sum, prior| {
            sum + x.get_element(row, prior) * second_temporary[prior]
        });
        let offset = row * x.cols() + panel_column;
        x.as_mut_slice()[offset] = (direct - first_correction - second_correction) * coefficient;
    }
    Ok(())
}

// Restore conjugated row packing after computing the panel update.
fn conjugate_row_tail<S: NumericScalar>(packed: &mut impl MatrixStorage<S>, index: usize) {
    let cols = packed.cols();
    for column in index + 1..cols {
        let offset = index * cols + column;
        packed.as_mut_slice()[offset] = packed.as_slice()[offset].conj();
    }
}

pub(super) fn set_identity<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>) {
    matrix.as_mut_slice().fill(S::ZERO);
    let columns = matrix.cols();
    for index in 0..matrix.rows().min(columns) {
        matrix.as_mut_slice()[index * columns + index] = S::ONE;
    }
}

fn normalize_and_sort<S: NumericScalar>(
    diagonal: &mut [S::Real],
    mut vt: Option<&mut ScratchMatrix<'_, S>>,
    permutation: &mut [usize],
) {
    for (index, value) in diagonal.iter_mut().enumerate() {
        if *value < S::Real::ZERO {
            *value = -*value;
            if let Some(matrix) = vt.as_deref_mut() {
                let columns = matrix.cols();
                for column in 0..columns {
                    matrix.as_mut_slice()[index * columns + column] =
                        -matrix.get_element(index, column);
                }
            }
        }
        // Comparisons do not distinguish the two IEEE zeros. Canonicalize
        // here for all jobs, including zeros returned by QR leaves.
        if *value == S::Real::ZERO {
            *value = S::Real::ZERO;
        }
    }
    // Descending order; ties fall back to the original index, so the order
    // (and hence which vector gets which slot) is deterministic even though
    // the sort itself is unstable.
    permutation
        .iter_mut()
        .enumerate()
        .for_each(|(index, target)| *target = index);
    permutation.sort_unstable_by(|&left, &right| {
        diagonal[right]
            .partial_cmp(&diagonal[left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
}

fn form_vectors<S: NumericScalar>(
    matrix: crate::FiniteMatrixView<'_, S>,
    workspace: &mut SvdScratch<'_, '_, S>,
    left_columns: usize,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let order = rows.min(cols);
    let major = rows.max(cols);
    // Left vectors of the packed (tall) matrix: `Q_L [U_B P; 0 I]`, i.e. the
    // sorted bidiagonal vectors padded with identity columns for a full job,
    // then multiplied by the left reflectors (last block first, as
    // `Q_L = H_1 ... H_k`). Right vectors are `Q_R V_B P`.
    workspace.left_factor.try_resize_flat(major, left_columns)?;
    workspace.left_factor.as_mut_slice().fill(S::ZERO);
    for row in 0..order {
        for column in 0..order {
            let source = workspace.permutation[column];
            workspace.left_factor.as_mut_slice()[row * left_columns + column] =
                workspace.core_u.get_element(row, source);
        }
    }
    for index in order..left_columns {
        workspace.left_factor.as_mut_slice()[index * left_columns + index] = S::ONE;
    }
    crate::reflector::apply_blocked(
        crate::reflector::Packed {
            factors: workspace.packed.try_view()?,
            tau: workspace.tau_left,
            offset: 0,
        },
        crate::reflector::Target::Matrix(crate::reflector::Product::Q),
        matrix_block_mut(
            &mut workspace.left_factor,
            0..workspace.packed.rows(),
            0..left_columns,
        )?,
        crate::reflector::CompactWy {
            vectors: &mut workspace.v_panel,
            triangular: &mut workspace.y_panel,
            work: &mut workspace.u_panel,
        },
        workspace.gemm,
        crate::kernel::KernelExecution { policy, context },
    )?;

    if !workspace.qr_tau.is_empty() {
        crate::reflector::apply_blocked(
            crate::reflector::Packed {
                factors: workspace.qr_packed.try_view()?,
                tau: workspace.qr_tau,
                offset: 0,
            },
            crate::reflector::Target::Padded(order),
            workspace.left_factor.try_view_mut()?,
            crate::reflector::CompactWy {
                vectors: &mut workspace.v_panel,
                triangular: &mut workspace.y_panel,
                work: &mut workspace.u_panel,
            },
            workspace.gemm,
            crate::kernel::KernelExecution { policy, context },
        )?;
    }

    form_internal_right_vectors(workspace, policy, context)?;
    copy_internal_vectors_to_output(matrix, workspace, order, left_columns)
}

fn form_internal_right_vectors<S: NumericScalar>(
    workspace: &mut SvdScratch<'_, '_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let order = workspace.diagonal.len();
    workspace.right_factor.try_resize_flat(order, order)?;
    workspace.right_factor.as_mut_slice().fill(S::ZERO);
    for row in 0..order {
        for column in 0..order {
            workspace.right_factor.as_mut_slice()[row * order + column] = workspace
                .core_vt
                .get_element(workspace.permutation[column], row)
                .conj();
        }
    }
    crate::reflector::apply_blocked(
        crate::reflector::Packed {
            factors: workspace.packed.try_view()?.adjoint(),
            tau: &workspace.tau_right[..order.saturating_sub(1)],
            offset: 1,
        },
        crate::reflector::Target::Matrix(crate::reflector::Product::Q),
        workspace.right_factor.try_view_mut()?,
        crate::reflector::CompactWy {
            vectors: &mut workspace.v_panel,
            triangular: &mut workspace.y_panel,
            work: &mut workspace.u_panel,
        },
        workspace.gemm,
        crate::kernel::KernelExecution { policy, context },
    )
}

fn copy_internal_vectors_to_output<S: NumericScalar>(
    matrix: crate::FiniteMatrixView<'_, S>,
    workspace: &mut SvdScratch<'_, '_, S>,
    order: usize,
    left_columns: usize,
) -> Result<(), SciOxideError> {
    // For a wide input the SVD was computed for `A^H = U' S V'^H`, so
    // `A = V' S U'^H`: the internal right factor becomes U and the adjoint
    // of the internal left factor becomes `V^H`.
    if matrix.rows() >= matrix.cols() {
        core::mem::swap(&mut workspace.candidate_u, &mut workspace.left_factor);
        adjoint_square_in_place(&mut workspace.right_factor);
        core::mem::swap(&mut workspace.candidate_vt, &mut workspace.right_factor);
    } else {
        workspace
            .candidate_u
            .try_resize_flat(matrix.rows(), order)?;
        workspace
            .candidate_u
            .as_mut_slice()
            .copy_from_slice(workspace.right_factor.as_slice());
        workspace
            .candidate_vt
            .try_resize_flat(left_columns, matrix.cols())?;
        for row in 0..left_columns {
            for column in 0..matrix.cols() {
                workspace.candidate_vt.as_mut_slice()[row * matrix.cols() + column] =
                    workspace.left_factor.get_element(column, row).conj();
            }
        }
    }
    Ok(())
}

fn adjoint_square_in_place<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>) {
    let order = matrix.rows();
    debug_assert_eq!(matrix.cols(), order);
    let data = matrix.as_mut_slice();
    for row in 0..order {
        data[row * order + row] = data[row * order + row].conj();
        for column in (row + 1)..order {
            let upper = data[row * order + column];
            let lower = data[column * order + row];
            data[row * order + column] = lower.conj();
            data[column * order + row] = upper.conj();
        }
    }
}

// Singular vector pairs are only defined up to a common unit phase
// (`u e^{i t}`, `e^{-i t} v^H`). Fixing the largest-magnitude entry of each
// U column to be real and positive (first such row on ties), and applying the
// compensating phase to the matching `V^H` row, makes the output
// deterministic without changing `U S V^H`.
fn canonicalize_vectors<S: NumericScalar>(
    u: &mut impl MatrixStorage<S>,
    vt: &mut impl MatrixStorage<S>,
    phases: &mut [S],
    magnitudes: &mut [S::Real],
) {
    if phases.is_empty() {
        return;
    }
    // The reduction scratch is dead after vector formation. Reuse it to
    // find every column pivot in one row-contiguous traversal, keeping the
    // first row on ties, then apply every phase in a second traversal.
    phases.fill(S::ONE);
    magnitudes.fill(S::Real::ZERO);
    for row in u.as_slice().chunks_exact(u.cols()) {
        for ((phase, magnitude), &value) in phases.iter_mut().zip(magnitudes.iter_mut()).zip(row) {
            let absolute = value.abs();
            if absolute > *magnitude {
                *magnitude = absolute;
                *phase = value;
            }
        }
    }
    for (phase, &magnitude) in phases.iter_mut().zip(magnitudes.iter()) {
        if magnitude != S::Real::ZERO {
            *phase = *phase / S::from_real(magnitude);
        }
    }
    // The pivot itself is stored as its magnitude, so it is exactly real; a
    // cleared magnitude marks a column whose pivot has been written.
    let columns = u.cols();
    for row in u.as_mut_slice().chunks_exact_mut(columns) {
        for ((value, phase), magnitude) in
            row.iter_mut().zip(phases.iter()).zip(magnitudes.iter_mut())
        {
            if *magnitude != S::Real::ZERO && value.abs() == *magnitude {
                *value = S::from_real(*magnitude);
                *magnitude = S::Real::ZERO;
            } else {
                *value = *value * phase.conj();
            }
        }
    }
    let columns = vt.cols();
    for (row, &phase) in vt
        .as_mut_slice()
        .chunks_exact_mut(columns)
        .zip(phases.iter())
    {
        row.iter_mut().for_each(|value| *value = *value * phase);
    }
}

impl<S: NumericScalar> crate::workspace::accounting::StorageBytes for ThinSvdDecomposition<S> {
    fn storage_bytes(&self) -> Result<usize, scioxide_core::SciOxideError> {
        crate::workspace::accounting::sum([
            self.u.storage_bytes()?,
            self.vt.storage_bytes()?,
            self.singular_values.storage_bytes()?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_reservation_preserves_values_and_shape_on_success_and_failure() {
        let matrix = Matrix::from_vec(2, 2, alloc::vec![3.0_f64, 0.0, 0.0, 2.0]).unwrap();
        let mut values = SvdValues::compute(&matrix).unwrap();
        for count in [0, 1, 8] {
            values.try_reserve(count).unwrap();
            assert_eq!((values.rows(), values.cols()), (2, 2));
            assert_eq!(values.singular_values(), &[3.0, 2.0]);
        }
        let error = values.try_reserve(usize::MAX).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::AllocationFailure { .. }
        ));
        assert_eq!((values.rows(), values.cols()), (2, 2));
        assert_eq!(values.singular_values(), &[3.0, 2.0]);
    }

    #[test]
    fn thin_reservation_preserves_factors_and_shape_on_success_and_failure() {
        let matrix = Matrix::from_vec(2, 2, alloc::vec![3.0_f64, 0.0, 0.0, 2.0]).unwrap();
        let mut result = ThinSvdDecomposition::compute(&matrix).unwrap();
        let original = result.clone();
        for (rows, cols) in [(0, 0), (1, 1), (8, 4), (4, 8)] {
            result.try_reserve(rows, cols).unwrap();
            assert_eq!((result.rows(), result.cols()), (2, 2));
            assert_eq!(result.singular_values(), original.singular_values());
            assert_eq!(result.u_matrix(), original.u_matrix());
            assert_eq!(result.vt_matrix(), original.vt_matrix());
        }
        assert!(result.try_reserve(usize::MAX, usize::MAX).is_err());
        assert_eq!((result.rows(), result.cols()), (2, 2));
        assert_eq!(result.singular_values(), original.singular_values());
        assert_eq!(result.u_matrix(), original.u_matrix());
        assert_eq!(result.vt_matrix(), original.vt_matrix());
    }

    fn assert_subnormal_normalization<R: RealScalar>(scale: R) {
        let exponent = crate::kernel::standard::exponent(scale).unwrap();
        let normalized = scale.scalbn(-exponent);
        assert!(normalized.is_finite());
        assert_eq!(normalized.scalbn(exponent), scale);
    }

    #[test]
    fn binary_normalization_preserves_smallest_subnormal_scales() {
        assert_subnormal_normalization(f32::from_bits(1));
        assert_subnormal_normalization(f64::from_bits(1));
    }

    fn reconstruct<S: NumericScalar>(
        u: &Matrix<S>,
        singular_values: &[S::Real],
        vt: &Matrix<S>,
    ) -> Matrix<S> {
        let mut result = Matrix::zeros(u.rows(), vt.cols()).unwrap();
        for row in 0..u.rows() {
            for column in 0..vt.cols() {
                result.as_mut_slice()[row * vt.cols() + column] = singular_values
                    .iter()
                    .enumerate()
                    .fold(S::ZERO, |sum, (index, &singular)| {
                        sum + u.get_element(row, index)
                            * S::from_real(singular)
                            * vt.get_element(index, column)
                    });
            }
        }
        result
    }

    #[test]
    fn typed_shapes_and_reconstruction() {
        let matrix = Matrix::from_vec(
            4,
            3,
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 10.0, 2.0, 3.0, 1.0,
            ],
        )
        .unwrap();
        let values = crate::SvdValues::compute(&matrix).unwrap();
        let thin = crate::ThinSvdDecomposition::compute(&matrix).unwrap();
        let full = crate::SvdDecomposition::compute(&matrix).unwrap();
        assert_eq!(values.singular_values().len(), 3);
        assert_eq!((thin.u_matrix().rows(), thin.u_matrix().cols()), (4, 3));
        assert_eq!((thin.vt_matrix().rows(), thin.vt_matrix().cols()), (3, 3));
        assert_eq!((full.u_matrix().rows(), full.u_matrix().cols()), (4, 4));
        let rebuilt = reconstruct(thin.u_matrix(), thin.singular_values(), thin.vt_matrix());
        for (&actual, &expected) in rebuilt.as_slice().iter().zip(matrix.as_slice()) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }

    #[test]
    fn wide_complex_reconstructs() {
        use scioxide_core::Complex;

        let matrix = Matrix::from_vec(
            2,
            4,
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, -1.0),
                Complex::new(0.0, 2.0),
                Complex::new(1.0, 0.0),
                Complex::new(-2.0, 1.0),
                Complex::new(0.5, 0.0),
                Complex::new(4.0, -3.0),
                Complex::new(2.0, 2.0),
            ],
        )
        .unwrap();
        let thin = crate::ThinSvdDecomposition::compute(&matrix).unwrap();
        let rebuilt = reconstruct(thin.u_matrix(), thin.singular_values(), thin.vt_matrix());
        for (&actual, &expected) in rebuilt.as_slice().iter().zip(matrix.as_slice()) {
            assert!((actual - expected).norm() < 1.0e-9);
        }
    }

    #[test]
    fn blocked_reduction_crosses_panel_boundaries() {
        let matrix = Matrix::from_vec(
            70,
            41,
            (0..70 * 41)
                .map(|index| {
                    let row = index / 41;
                    let column = index % 41;
                    ((row * 17 + column * 29 + 3) % 101) as f64 / 50.0 - 1.0
                })
                .collect(),
        )
        .unwrap();
        let thin = crate::ThinSvdDecomposition::compute(&matrix).unwrap();
        let rebuilt = reconstruct(thin.u_matrix(), thin.singular_values(), thin.vt_matrix());
        let error = rebuilt
            .as_slice()
            .iter()
            .zip(matrix.as_slice())
            .map(|(&actual, &expected)| (actual - expected) * (actual - expected))
            .sum::<f64>()
            .sqrt();
        let norm = matrix
            .as_slice()
            .iter()
            .map(|&value| value * value)
            .sum::<f64>()
            .sqrt();
        assert!(
            error / norm < 1.0e-11,
            "relative residual = {}",
            error / norm
        );
    }

    #[test]
    fn workspace_jobs_reserve_only_required_factor_shapes() {
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        let request = SvdWorkspaceRequest::new(256, 128, &options);
        let values = SvdValues::workspace_requirements::<f64>(&request).unwrap();
        let thin = ThinSvdDecomposition::<f64>::workspace_requirements(&request).unwrap();
        let full = SvdDecomposition::<f64>::workspace_requirements(&request).unwrap();
        assert!(values.required_bytes::<f64>().unwrap() < thin.required_bytes::<f64>().unwrap());
        assert!(thin.required_bytes::<f64>().unwrap() < full.required_bytes::<f64>().unwrap());
        let mut workspace = LinalgWorkspace::<f64>::new();
        for requirements in [values, thin, full] {
            workspace.try_reserve(&requirements).unwrap();
            assert_eq!(
                workspace.check(&requirements).unwrap(),
                scioxide_core::Capacity::Sufficient
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod reusable_solve_tests {
    use super::*;
    #[test]
    fn damped_and_minimum_norm_solutions_share_prepared_storage() {
        // A=[1 1], b=2. The minimum-norm solution is (1,1).
        // With lambda=2, solving the augmented least-squares model gives (.5,.5).
        let a = Matrix::from_vec(1, 2, alloc::vec![1.0_f64, 1.0]).unwrap();
        let factor = crate::ThinSvdDecomposition::compute(&a).unwrap();
        let b = Vector::from_vec(alloc::vec![2.0]);
        let mut x = Vector::zeros(2).unwrap();
        let mut scratch = LinalgWorkspace::new();
        for (damping, expected) in [(0.0, 1.0), (2.0, 0.5)] {
            factor
                .solve_into(
                    &b,
                    RankPolicy::DimensionScaledEpsilon,
                    damping,
                    &crate::LinalgExecutionOptions::new(),
                    x.view_mut(),
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(
                x.as_slice()
                    .iter()
                    .all(|&value| (value - expected).abs() < 1e-14)
            );
        }
        assert!(matches!(
            (factor.solve_into(
                &b,
                RankPolicy::DimensionScaledEpsilon,
                -1.0,
                &crate::LinalgExecutionOptions::new(),
                x.view_mut(),
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (factor.solve_into(
                &b,
                RankPolicy::DimensionScaledEpsilon,
                0.0,
                &crate::LinalgExecutionOptions::new(),
                Vector::empty().view_mut(),
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }
}
