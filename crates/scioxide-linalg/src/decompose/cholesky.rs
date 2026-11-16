//! Cholesky factorization `A = L Lᴴ`, with optional diagonal pivoting.
//!
//! Blocked right-looking Cholesky (Golub & Van Loan, §4.2): each panel of at
//! most 64 columns is gathered into a contiguous buffer, its diagonal block is
//! factored unblocked, the panel below it is solved with TRSM, and the trailing
//! lower triangle is updated with SYRK or HERK. About n³/3 flops, O(n²) memory.

use crate::MatrixSide;
use crate::workspace::{RealRegion, ScalarRegion};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::DiagonalKind;

use super::hermitian::{HermitianInput, HermitianProjectionReport};
use scioxide_core::RealScalar;
use scioxide_core::{ExecutionContext, ExecutionPolicy, NumericScalar, SciOxideError};

use crate::KernelContext;
use crate::dense::{Matrix, Vector};
use crate::kernel::level3::trsm_bounded;
use crate::kernel::{GemmConfig, GemmOp, LinalgWorkspace, Triangle, gemm_workspace_with_context};

use super::util::{copy_panel, reserve_vec, scale_slice, scale_with_power, transfer_panel};
use crate::kernel::TrailingBound;

/// Widest Cholesky panel, the cap of [`crate::kernel::panel_width`]; its
/// diagonal block is factored unblocked.
const MAX_PANEL: usize = 64;
use crate::analysis::analyze_magnitudes;
use crate::{RankPolicy, RankReport};

/// Scratch size request for a Cholesky factorization, solve or inverse.
///
/// Pass it to [`CholeskyDecomposition::workspace_requirements`] or
/// [`CholeskyDecomposition::workspace`]. The factor storage itself is separate;
/// see [`CholeskyDecomposition::try_reserve`].
#[derive(Debug, Clone, Copy)]
pub struct CholeskyWorkspaceRequest {
    job: CholeskyJob,
    kernel: crate::DenseWorkspaceRequest,
}

#[derive(Debug, Clone, Copy)]
enum CholeskyJob {
    Factor(usize),
    Vector(usize),
    Matrix { order: usize, rhs: usize },
}

impl CholeskyWorkspaceRequest {
    /// Requests scratch for [`CholeskyDecomposition::factor_into`] of an
    /// `order × order` matrix.
    pub const fn factor(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: CholeskyJob::Factor(order),
            kernel: crate::DenseWorkspaceRequest::product(order, order, order, options),
        }
    }

    /// Requests scratch for [`CholeskyDecomposition::solve_into`] with one
    /// right-hand side.
    pub const fn solve(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: CholeskyJob::Vector(order),
            kernel: crate::DenseWorkspaceRequest::product(order, order, 1, options),
        }
    }

    /// Requests scratch for [`CholeskyDecomposition::solve_matrix_into`] with
    /// `right_hand_sides` columns.
    ///
    /// Use `right_hand_sides == order` for [`CholeskyDecomposition::inverse_into`].
    pub const fn solve_matrix(
        order: usize,
        right_hand_sides: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: CholeskyJob::Matrix {
                order,
                rhs: right_hand_sides,
            },
            kernel: crate::DenseWorkspaceRequest::product(order, order, right_hand_sides, options),
        }
    }

    /// Sizes the per-worker scratch for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: core::num::NonZeroUsize) -> Self {
        self.kernel = self.kernel.with_worker_bound(workers);
        self
    }

    /// Adds the packing buffers that an injected BLAS provider needs.
    pub const fn with_provider_packing(mut self) -> Self {
        self.kernel = self.kernel.with_provider_packing();
        self
    }

    /// Sets the worker bound and BLAS packing from `context`.
    ///
    /// Product accuracy still comes from the options passed to the constructor.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self = self.with_worker_bound(context.execution().worker_budget());
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }
}

impl<S: NumericScalar> CholeskyDecomposition<S> {
    /// Returns the workspace size that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        request: &CholeskyWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let mut requirements = crate::kernel::product_requirements::<S>(&request.kernel)?;
        let product = |a, b| {
            scioxide_core::backend::checked_product(
                &[a, b],
                scioxide_core::quantity!("cholesky_scratch"),
            )
        };
        *requirements.scalar_mut(ScalarRegion::Direct) = match request.job {
            CholeskyJob::Factor(n) => {
                let block = n.min(MAX_PANEL);
                requirements.indices = n;
                requirements.exponents = n;
                *requirements.real_mut(RealRegion::Direct) = n;
                [
                    product(n, n)?,
                    product(n, n)?,
                    product(n, n)?,
                    product(n, block)?,
                ]
                .into_iter()
                .try_fold(0usize, |a, b| {
                    a.checked_add(b).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("cholesky_scratch"))
                    })
                })?
            }
            CholeskyJob::Vector(n) => {
                requirements.scaled = product(n, 2)?;
                product(n, 2)?
            }
            CholeskyJob::Matrix { order, rhs } => {
                requirements.scaled = product(order, 2)?;
                product(
                    order,
                    if order == 0 {
                        0
                    } else {
                        rhs.checked_add(2).ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "cholesky_scratch"
                            ))
                        })?
                    },
                )?
            }
        };
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }

    /// Allocates a workspace large enough for `request`.
    ///
    /// # Errors
    ///
    /// Workspace errors as described in
    /// [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{CholeskyDecomposition, CholeskyWorkspaceRequest, LinalgExecutionOptions};
    /// let request = CholeskyWorkspaceRequest::factor(4, &LinalgExecutionOptions::new());
    /// let requirements = CholeskyDecomposition::<f64>::workspace_requirements(&request)?;
    /// let workspace = CholeskyDecomposition::<f64>::workspace(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &CholeskyWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

struct CholeskyScratch<'a, 'storage, S: NumericScalar> {
    candidate: crate::MatrixViewMut<'a, S>,
    source: crate::MatrixViewMut<'a, S>,
    original: &'a mut [S],
    permutation: &'a mut [usize],
    scaling_exponents: &'a mut [i64],
    lower_panel: &'a mut [S],
    residual_diagonal: &'a mut [S::Real],
    gemm: &'a mut LinalgWorkspace<'storage, S>,
}

impl<'a, 'storage, S: NumericScalar> CholeskyScratch<'a, 'storage, S> {
    /// Carve the direct regions of [`CholeskyWorkspaceRequest::factor`].
    fn carve(
        order: usize,
        scalars: &'a mut [S],
        reals: &'a mut [S::Real],
        indices: &'a mut [usize],
        exponents: &'a mut [i64],
        gemm: &'a mut LinalgWorkspace<'storage, S>,
    ) -> Result<Self, SciOxideError> {
        let block = order.min(MAX_PANEL);
        let (candidate, scalars) = crate::workspace::split_mut(scalars, order * order)?;
        let (source, scalars) = crate::workspace::split_mut(scalars, order * order)?;
        let (original, scalars) = crate::workspace::split_mut(scalars, order * order)?;
        let permutation = &mut indices[..order];
        Ok(Self {
            candidate: crate::MatrixViewMut::from_slice(order, order, candidate)?,
            source: crate::MatrixViewMut::from_slice(order, order, source)?,
            original,
            permutation,
            scaling_exponents: &mut exponents[..order],
            lower_panel: &mut scalars[..order * block],
            residual_diagonal: &mut reals[..order],
            gemm,
        })
    }
}

/// Pivoting of a Cholesky factorization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum CholeskyPivoting<R: RealScalar> {
    /// No pivoting. The matrix must be positive definite.
    None,
    /// Diagonal pivoting for positive semidefinite matrices, with the rank
    /// decided by the given policy.
    ///
    /// Each step takes the largest remaining diagonal entry of the Schur
    /// complement as the pivot. The rank `r` is the number of steps before that
    /// entry drops to `max(τ, τ₀)`, where `τ` is the threshold of the policy
    /// and `τ₀ = min(n ε, 1/2) s`, both with scale `s = max_i a_ii`; columns
    /// `r..n` of `L` are zero. A pivot at or below `τ₀` is at the level of the
    /// rounding errors of the earlier steps, so `τ₀` ends the factorization
    /// under every policy, including [`RankPolicy::Exact`].
    /// [`RankPolicy::DimensionScaledEpsilon`] has `τ = τ₀`.
    ///
    /// The factorization returns `DomainError` if a Schur complement has a
    /// diagonal entry below `−δ`, or if the Schur complement left when the
    /// pivots reach `τ₀` has an entry of modulus above `τ₀ + δ`, where
    /// `δ = min(8 n ε, 1/2) s`. So a coarse `τ` cannot hide an indefinite matrix.
    ///
    /// See Higham (2002), §10.3, for the stopping criterion.
    Diagonal(RankPolicy<R>),
}

/// Rank of a Cholesky factor, as computed by the factorization and under a
/// solve's rank policy.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CholeskyDefinitenessReport<R: RealScalar> {
    factor_rank: RankReport<R>,
    effective_rank: RankReport<R>,
}

impl<R: RealScalar> CholeskyDefinitenessReport<R> {
    /// Returns the rank at which the factorization stopped, with the threshold
    /// `max(τ, τ₀)` of [`CholeskyPivoting::Diagonal`] (zero without pivoting).
    pub const fn factor_rank(self) -> RankReport<R> {
        self.factor_rank
    }

    /// Returns the rank under the requested policy, applied to the pivots
    /// `|L_kk|²`.
    pub const fn effective_rank(self) -> RankReport<R> {
        self.effective_rank
    }
}

/// Cholesky factorization `A[p, p] = L Lᴴ` of a Hermitian positive definite or
/// semidefinite matrix.
///
/// `L` is lower triangular with a real, nonnegative diagonal, and `p` is a
/// symmetric permutation. Without pivoting `p` is the identity. With
/// [`CholeskyPivoting::Diagonal`] the factor can have rank `r < n`; columns
/// `r..n` of `L` are then zero. For real matrices `Lᴴ = Lᵀ`.
///
/// Before elimination the matrix is scaled symmetrically, `D A D` with `D`
/// diagonal powers of two, so that each nonzero diagonal entry lies in
/// `[1/2, 2)`. The scaling is exact except for an off-diagonal entry `a_ij`
/// below about `μ √(a_ii a_jj)`, where `μ` is the smallest positive normal
/// number of the scalar type: its scaled value is subnormal and is rounded,
/// or flushed to zero. [`Self::l_matrix`] returns the factor of the unscaled
/// matrix.
///
/// # Accuracy
///
/// The computed factor is the exact factor of `A + ΔA` with
/// `‖ΔA‖∞ ≤ c n ε ‖|L| |Lᴴ|‖∞` in scaled coordinates, where `ε` is the machine
/// epsilon and `c` is a small constant (Higham, 2002, Chapter 10). A successful
/// factorization does not prove that `A` is positive definite.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, HermitianInput, Matrix, Vector};
///
/// // A = [4 2; 2 3] = L Lᵀ with L = [2 0; 1 √2], and A [1, 1] = [6, 5].
/// let a = Matrix::from_rows(&[[4.0_f64, 2.0], [2.0, 3.0]])?;
/// let chol = CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
/// assert!((chol.l_matrix().view().get(1, 1)? - 2.0_f64.sqrt()).abs() < 1e-15);
/// let x = chol.solve(&Vector::from_vec(vec![6.0, 5.0]))?;
/// assert!(x.as_slice().iter().all(|value| (value - 1.0).abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §4.2.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Chapter 10.
#[derive(Debug, Clone)]
#[must_use]
pub struct CholeskyDecomposition<S: NumericScalar> {
    /// Lower triangular factor L such that A = L * L^H
    l: Matrix<S>,
    source: Matrix<S>,
    maximum: S::Real,
    projection: Option<HermitianProjectionReport<S::Real>>,
    /// Dimension of the square matrix
    n: usize,
    /// Symmetric row/column permutation for diagonal pivoting.
    permutation: Vec<usize>,
    /// Rank at which the factorization stopped, with its threshold.
    factor_rank: RankReport<S::Real>,
    one_norm: scioxide_core::Scaled<S::Real>,
}

impl<S: NumericScalar> CholeskyDecomposition<S> {
    /// Returns the bytes of factor storage that an `order × order` factorization needs.
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
    /// assert_eq!(scioxide_linalg::CholeskyDecomposition::<f64>::factor_storage_bytes(0)?, 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn factor_storage_bytes(order: usize) -> Result<usize, SciOxideError> {
        let square = scioxide_core::backend::checked_product(
            &[order, order],
            scioxide_core::quantity!("factor_elements"),
        )?;
        super::util::storage_bytes([
            (square, core::mem::size_of::<S>()),
            (square, core::mem::size_of::<S>()),
            (order, core::mem::size_of::<usize>()),
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
        super::util::storage_bytes([
            (self.l.capacity(), core::mem::size_of::<S>()),
            (self.source.capacity(), core::mem::size_of::<S>()),
            (self.permutation.capacity(), core::mem::size_of::<usize>()),
        ])
    }

    /// Returns an empty factorization of order 0, to pass to [`Self::factor_into`].
    pub const fn empty() -> Self {
        Self {
            l: Matrix::empty(),
            source: Matrix::empty(),
            maximum: S::Real::ZERO,
            projection: None,
            n: 0,
            permutation: Vec::new(),
            factor_rank: RankReport::from_factor(0, 0, S::Real::ZERO, S::Real::ZERO),
            one_norm: scioxide_core::Scaled::ZERO,
        }
    }

    /// Grows the factor storage to hold an `order × order` factorization.
    ///
    /// The current factorization and projection report are unchanged. Growth is
    /// exact: after reserving into [`Self::empty`], [`Self::retained_bytes`]
    /// equals [`Self::factor_storage_bytes`]. A later [`Self::factor_into`] of a
    /// matrix of at most this order does not allocate factor storage. Capacity is
    /// never released.
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
    /// use scioxide_linalg::CholeskyDecomposition;
    /// let mut factor = CholeskyDecomposition::<f64>::empty();
    /// factor.try_reserve(3)?;
    /// assert_eq!(factor.check(3)?, Capacity::Sufficient);
    /// assert_eq!(factor.check(4)?, Capacity::NeedsGrowth);
    /// assert_eq!(
    ///     factor.retained_bytes()?,
    ///     CholeskyDecomposition::<f64>::factor_storage_bytes(3)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        self.l.try_reserve_shape(order, order)?;
        self.source.try_reserve_shape(order, order)?;
        reserve_vec(
            &mut self.permutation,
            order,
            scioxide_core::quantity!("permutation"),
        )
    }

    /// Reports whether the factor storage can hold an `order × order`
    /// factorization without growing.
    ///
    /// See [`Self::try_reserve`] for an example.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn check(&self, order: usize) -> Result<scioxide_core::Capacity, SciOxideError> {
        let mut plan = crate::workspace::accounting::StoragePlan::new(0);
        plan.matrix(&self.l, order, order)?;
        plan.matrix(&self.source, order, order)?;
        plan.buffer(&self.permutation, order)?;
        plan.check(None)
    }

    /// Computes the Cholesky factorization `A[p, p] = L Lᴴ` of the Hermitian
    /// matrix that `interpretation` defines from `a`.
    ///
    /// See [`HermitianInput`] for how `a` is read and [`CholeskyPivoting`] for
    /// the pivoting. Without pivoting the factorization costs about n³/3 flops,
    /// half the cost of [`LuDecomposition::factor`](crate::LuDecomposition::factor),
    /// and runs on the calling thread. Use [`Self::factor_into`] to reuse storage
    /// or to run the trailing updates in parallel. A projected input's change is
    /// reported by [`Self::projection_report`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square.
    /// - The input errors listed on [`HermitianInput`].
    /// - `DomainError` (`OutsideDomain`) without pivoting if a pivot is not
    ///   positive, that is, `A` is not numerically positive definite; with
    ///   [`CholeskyPivoting::Diagonal`] if `A` is not numerically positive
    ///   semidefinite.
    ///
    /// # References
    ///
    /// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    /// SIAM, 2002, §10.1 and §10.3.
    pub fn factor<'m>(
        a: impl Into<crate::MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        pivoting: CholeskyPivoting<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(
            a,
            interpretation,
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
        a: impl Into<crate::MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        pivoting: CholeskyPivoting<S::Real>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut factor = Self::empty();
        Self::factor_into(
            a,
            interpretation,
            pivoting,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(factor)
    }

    /// Computes the Cholesky factorization into `factor`, the reusable form of
    /// [`Self::factor`].
    ///
    /// `workspace` holds the scratch and `context` supplies the executor, clock
    /// and cancellation. The trailing updates follow the execution policy in
    /// `options`. On success the projection report of `factor` is replaced,
    /// and cleared for an exact or triangle input.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::factor`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<crate::MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        pivoting: CholeskyPivoting<S::Real>,
        options: &crate::LinalgExecutionOptions,
        factor: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let a = a.into();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        context.execution().check()?;
        let order = a.rows();
        if order != a.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [order, order],
                [order, a.cols()],
            ));
        }
        let requirements = Self::workspace_requirements(
            &CholeskyWorkspaceRequest::factor(order, options).for_kernels(context),
        )?;
        workspace.try_reserve(&requirements)?;
        let projection = workspace.with_direct(|scalars, _, _, _| {
            let (_, rest) = crate::workspace::split_mut(scalars, order * order)?;
            let (source, _) = crate::workspace::split_mut(rest, order * order)?;
            super::hermitian::prepare(
                a,
                interpretation,
                &mut crate::MatrixViewMut::from_slice(order, order, source)?,
                scioxide_core::argument!("a"),
            )
        })?;
        factor.try_reserve(order)?;
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "cholesky_factor", 1),
                order as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                crate::solve_support::admit_iteration(
                    Some(context.execution()),
                    Some(solve),
                    "direct numerical work budget exhausted",
                )?;
                workspace.with_scaled(|_, exponents, child| {
                    child.with_direct(|scalars, reals, indices, gemm| {
                        let mut scratch = CholeskyScratch::carve(
                            order, scalars, reals, indices, exponents, gemm,
                        )?;
                        Self::factor_prepared_into(
                            factor,
                            &mut scratch,
                            pivoting,
                            options.execution_policy(),
                            context,
                        )?;
                        factor.projection = projection;
                        Ok(())
                    })
                })
            },
        )
    }

    /// Returns the change made by [`HermitianInput::Projected`] in the last
    /// factorization, or `None` for an exact or triangle input.
    pub const fn projection_report(&self) -> Option<&HermitianProjectionReport<S::Real>> {
        self.projection.as_ref()
    }

    /// Factor an exactly Hermitian positive-definite view without pivoting and
    /// write `L` with `A = L Lᴴ` (row-major, `order × order`) into `lower`.
    ///
    /// Scratch follows [`CholeskyWorkspaceRequest::factor`] requirements in
    /// `workspace`'s direct region. No factor object is written, so reductions
    /// such as generalized Hermitian eigenproblems nest it inside their own arenas.
    pub(crate) fn factor_exact_lower(
        a: crate::MatrixView<'_, S>,
        argument: scioxide_core::Argument,
        lower: &mut [S],
        policy: ExecutionPolicy,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let order = a.rows();
        crate::checked::validate_exact_hermitian(a, argument)?;
        workspace.with_scaled(|_, exponents, child| {
            child.with_direct(|scalars, reals, indices, gemm| {
                let mut scratch =
                    CholeskyScratch::carve(order, scalars, reals, indices, exponents, gemm)?;
                for row in 0..order {
                    for col in 0..order {
                        scratch.source.data[row * order + col] = a.get_element(row, col);
                    }
                }
                // Name the caller's operand in an indefiniteness rejection.
                let _ =
                    Self::factor_candidate(&mut scratch, CholeskyPivoting::None, policy, context)
                        .map_err(|error| {
                        if error.kind() == scioxide_core::SciOxideErrorKind::DomainError {
                            not_positive_definite(argument)
                        } else {
                            error
                        }
                    })?;
                lower[..order * order].copy_from_slice(scratch.candidate.data);
                Ok(())
            })
        })
    }

    fn factor_prepared_into(
        factor: &mut Self,
        workspace: &mut CholeskyScratch<'_, '_, S>,
        pivoting: CholeskyPivoting<S::Real>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let order = workspace.source.rows();
        let one_norm = crate::diagnostics::matrix_one_norm(workspace.source.as_view())?;
        // Only diagonal pivoting stages its Schur complement in `source`.
        let pivoted = matches!(pivoting, CholeskyPivoting::Diagonal(_));
        if pivoted {
            workspace.original.copy_from_slice(workspace.source.data);
        }
        let (factor_rank, maximum) = Self::factor_candidate(workspace, pivoting, policy, context)?;
        context.execution().check()?;
        factor.source.try_resize_flat(order, order)?;
        factor.source.as_mut_slice().copy_from_slice(if pivoted {
            workspace.original
        } else {
            workspace.source.data
        });
        factor.l.try_resize_flat(order, order)?;
        factor
            .l
            .as_mut_slice()
            .copy_from_slice(workspace.candidate.data);
        factor.permutation.clear();
        factor.permutation.extend_from_slice(workspace.permutation);
        factor.maximum = maximum;
        factor.n = order;
        factor.factor_rank = factor_rank;
        factor.one_norm = one_norm;
        Ok(())
    }

    /// Equilibrate, factor and unscale the staged source into the candidate,
    /// returning the construction rank report and the candidate's largest magnitude.
    fn factor_candidate(
        workspace: &mut CholeskyScratch<'_, '_, S>,
        pivoting: CholeskyPivoting<S::Real>,
        policy: ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(RankReport<S::Real>, S::Real), SciOxideError> {
        let order = workspace.source.rows();
        let scale = (0..order).fold(S::Real::ZERO, |scale, i| {
            scale.max(workspace.source.get_element(i, i).real_component())
        });
        // Diagonal pivoting stops at the policy threshold `τ` and, whatever the
        // policy, at the rounding floor `min(n ε, 1/2) s`.
        let (threshold, floor) = match pivoting {
            CholeskyPivoting::None => (S::Real::ZERO, S::Real::ZERO),
            CholeskyPivoting::Diagonal(rank_policy) => (
                rank_policy.cutoff(scale, order, order)?,
                RankPolicy::DimensionScaledEpsilon.cutoff(scale, order, order)?,
            ),
        };

        // Symmetric power-of-two equilibration `D A D`, with `D_ii` chosen so
        // that each scaled diagonal is O(1). Scaling both sides by the same
        // D keeps the matrix Hermitian (so Cholesky still applies) and is
        // exact unless a scaled entry is subnormal. The factor of
        // `P^T D A D P` is unscaled row-wise at the end, giving the factor of
        // `P^T A P` directly. Exponents are construction scratch.
        for index in 0..order {
            workspace.scaling_exponents[index] =
                diagonal_equilibration_exponent(workspace.source.get_element(index, index).abs())?
                    as i64;
            workspace.residual_diagonal[index] =
                S::Real::ONE.scalbn(workspace.scaling_exponents[index] as i32);
        }
        // Powers are exact; when every product of two is a finite nonzero
        // multiplier, rows scale with one contiguous multiply per entry.
        let powers = &workspace.residual_diagonal[..order];
        let (smallest, largest) = powers.iter().fold(
            (S::Real::MAX, S::Real::ZERO),
            |(smallest, largest), &power| (smallest.min(power), largest.max(power)),
        );
        for row in 0..order {
            let factor = powers[row];
            let regular = (factor * largest).is_finite() && factor * smallest != S::Real::ZERO;
            for (col, &column_power) in powers.iter().enumerate().take(row + 1) {
                let index = row * order + col;
                let power = factor * column_power;
                let value = if regular {
                    let value = workspace.source.data[index];
                    S::from_components(
                        value.real_component() * power,
                        value.imaginary_component() * power,
                    )
                } else {
                    let exponent = (workspace.scaling_exponents[row] as i32)
                        + (workspace.scaling_exponents[col] as i32);
                    scale_with_power(workspace.source.data[index], power, exponent)
                };
                // A positive semidefinite matrix has `|a_ij| ≤ √(a_ii a_jj)`, so
                // its scaled entries have modulus at most 2. An entry that
                // overflows in scaled coordinates proves the matrix indefinite.
                if !value.has_finite_components() {
                    return Err(match pivoting {
                        CholeskyPivoting::None => {
                            not_positive_definite(scioxide_core::argument!("a"))
                        }
                        CholeskyPivoting::Diagonal(_) => not_positive_semidefinite(),
                    });
                }
                match pivoting {
                    CholeskyPivoting::None => workspace.candidate.data[index] = value,
                    CholeskyPivoting::Diagonal(_) => {
                        workspace.source.data[index] = value;
                        workspace.source.data[col * order + row] = value.conj();
                    }
                }
            }
            workspace.candidate.data[row * order + row + 1..(row + 1) * order].fill(S::ZERO);
        }
        workspace
            .permutation
            .iter_mut()
            .enumerate()
            .for_each(|(index, slot)| *slot = index);
        let completed_rank = match pivoting {
            CholeskyPivoting::None => {
                factor_unpivoted(workspace, policy, context)?;
                order
            }
            CholeskyPivoting::Diagonal(_) => {
                factor_pivoted(workspace, threshold, floor, scale, policy, context)?
            }
        };
        let mut maximum = S::Real::ZERO;
        for row in 0..order {
            let exponent = -(workspace.scaling_exponents[workspace.permutation[row]] as i32);
            let values = &mut workspace.candidate.data[row * order..row * order + row + 1];
            scale_slice(values, exponent);
            maximum = maximum.max(crate::kernel::standard::finite_slice_maximum(values)?);
        }
        Ok((
            RankReport::from_factor(completed_rank, order, threshold.max(floor), scale),
            maximum,
        ))
    }

    /// Solves `A X = B` for every column of `rhs` and returns `X`.
    ///
    /// The solve uses the default [`DirectSolveOptions`](crate::DirectSolveOptions),
    /// as [`Self::solve`] does.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have [`Self::order`] rows.
    /// - The other errors of [`Self::solve`].
    pub fn solve_matrix<'b>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.solve_matrix_with_options(rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `A X = B` for every column of `rhs`, checks the result against `options`
    /// and returns it with its [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The checks are those of [`Self::solve_with_options`], made once for all
    /// columns. A requested residual is the largest backward error over the columns.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have [`Self::order`] rows.
    /// - The other errors of [`Self::solve_with_options`].
    pub fn solve_matrix_with_options<'b>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
    ) -> Result<crate::DirectSolveResult<Matrix<S>, S::Real>, SciOxideError> {
        let rhs = rhs.into();
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let mut output = Matrix::zeros(self.order(), rhs.cols())?;
        let report = self.solve_matrix_into(
            rhs,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(crate::DirectSolveResult::new(output, report))
    }

    /// Solves `A X = B` into `output`, the reusable form of [`Self::solve_matrix`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not [`Self::order`] × `rhs.cols()`.
    /// - The errors of [`Self::solve_matrix`] and the workspace and execution
    ///   errors of [`Self::factor_into`].
    pub fn solve_matrix_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::MatrixView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&options.execution_options())?;
        context.execution().check()?;
        let rhs = rhs.into();
        let n = self.order();
        crate::checked::check_block_shape(n, rhs, &output)?;
        let rhs = rhs
            .try_finite_dispatched(context.kernel().dispatch())?
            .as_view();
        let rank = self
            .definiteness_report(options.rank_policy())?
            .effective_rank();
        let request =
            CholeskyWorkspaceRequest::solve_matrix(n, rhs.cols(), &options.execution_options())
                .for_kernels(context.kernel());
        let requirements = Self::workspace_requirements(&request)?;
        workspace.try_reserve(&requirements)?;
        if self.n == 0 && options.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "cholesky_solve_matrix", 1),
                n as u128,
                0,
            ),
            options.execution_options().solve_budgets(),
            |workspace, context, solve| {
                let condition = options
                    .minimum_reciprocal_condition()
                    .map(|_| {
                        self.condition_estimate_scoped(
                            &options.execution_options(),
                            workspace,
                            context.scoped(&mut *solve),
                        )
                    })
                    .transpose()?;
                let report = crate::analysis::acceptance(rank, condition, options);
                if report.termination() != crate::DirectSolveTermination::Accepted
                    || n == 0
                    || rhs.cols() == 0
                {
                    let residual = if n == 0 {
                        crate::analysis::empty_residual(options, rhs.cols())?
                    } else {
                        None
                    };
                    return Ok(crate::DirectSolveReport::new(
                        report.termination(),
                        rank,
                        condition,
                        residual,
                    ));
                }
                let kernels = context;
                for _ in 0..rhs.cols() {
                    crate::solve_support::admit_iteration(
                        Some(context.execution()),
                        Some(&mut *solve),
                        "direct RHS budget exhausted",
                    )?;
                }
                workspace.with_direct(|storage, _, _, gemm| {
                    let (candidate, rest) = crate::workspace::split_mut(storage, n * rhs.cols())?;
                    for (row, &source) in self.permutation.iter().enumerate() {
                        for col in 0..rhs.cols() {
                            candidate[row * rhs.cols() + col] = rhs.get_element(source, col);
                        }
                    }
                    self.solve_candidate(candidate, rhs.cols(), gemm, kernels)?;
                    crate::checked::validate_values(
                        candidate,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                        "Cholesky solution is not finite",
                    )?;
                    let mut residual = None;
                    if options.residual() == crate::Estimation::Estimate {
                        let (column, scratch) = crate::workspace::split_mut(rest, n)?;
                        for col in 0..rhs.cols() {
                            for row in 0..n {
                                column[row] = candidate[row * rhs.cols() + col];
                            }
                            crate::analysis::retain_largest_residual(
                                &mut residual,
                                self.backward_error(column, rhs.col(col)?, scratch, kernels)?,
                            );
                        }
                    }
                    kernels.execution().check()?;
                    for row in 0..n {
                        for col in 0..rhs.cols() {
                            output.set_element(
                                self.permutation[row],
                                col,
                                candidate[row * rhs.cols() + col],
                            );
                        }
                    }
                    Ok(crate::DirectSolveReport::new(
                        report.termination(),
                        rank,
                        condition,
                        residual,
                    ))
                })
            },
        )
    }

    /// Returns `A⁻¹`.
    ///
    /// Solves `A X = I` with the stored factor. Solving with [`Self::solve`] is
    /// cheaper and more accurate than multiplying by the inverse.
    ///
    /// # Errors
    ///
    /// - `Singular` if the factor has rank below [`Self::order`].
    /// - `NumericalInstability` (`Overflow`) or `NonFinite` if an entry of
    ///   `A⁻¹` overflows.
    pub fn inverse(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.n, self.n)?;
        self.inverse_into(
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Writes `A⁻¹` to `output`, the reusable form of [`Self::inverse`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not [`Self::order`] × [`Self::order`].
    /// - The errors of [`Self::inverse`] and the workspace and execution errors
    ///   of [`Self::factor_into`].
    pub fn inverse_into<'scope, 'capability>(
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
        if output.rows() != self.n || output.cols() != self.n {
            return Err(SciOxideError::dimension_mismatch(
                [self.n, self.n],
                [output.rows(), output.cols()],
            ));
        }
        workspace.try_reserve(&Self::workspace_requirements(
            &CholeskyWorkspaceRequest::solve_matrix(self.n, self.n, options).for_kernels(context),
        )?)?;
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "cholesky_inverse", 1),
                self.n as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                for _ in 0..self.n {
                    crate::solve_support::admit_iteration(
                        Some(context.execution()),
                        Some(&mut *solve),
                        "inverse RHS budget exhausted",
                    )?;
                }
                self.solve_staged(
                    self.n,
                    |row, col| if row == col { S::ONE } else { S::ZERO },
                    |row, col, value| output.set_element(row, col, value),
                    workspace,
                    context,
                )
            },
        )
    }

    fn solve_staged(
        &self,
        columns: usize,
        rhs: impl Fn(usize, usize) -> S,
        mut publish: impl FnMut(usize, usize, S),
        workspace: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.validate_diagonal()?;
        workspace.with_direct(|storage, _, _, gemm| {
            let candidate = &mut storage[..self.n * columns];
            for (row, &source) in self.permutation.iter().enumerate() {
                for col in 0..columns {
                    candidate[row * columns + col] = rhs(source, col);
                }
            }
            self.solve_candidate(candidate, columns, gemm, context)?;
            crate::checked::validate_values(
                candidate,
                scioxide_core::NonFiniteOrigin::Intermediate,
                "Cholesky solution is not finite",
            )?;
            context.execution().check()?;
            for (row, &destination) in self.permutation.iter().enumerate() {
                for col in 0..columns {
                    publish(destination, col, candidate[row * columns + col]);
                }
            }
            Ok(())
        })
    }

    /// Returns the lower triangular factor `L` of `A[p, p] = L Lᴴ`, of the
    /// unscaled matrix. The strictly upper part is zero.
    pub const fn l_matrix(&self) -> &Matrix<S> {
        &self.l
    }

    /// Returns the order `n` of the factored `n × n` matrix.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.n
    }

    /// Returns the symmetric permutation `p` with `A[p, p] = L Lᴴ`; the identity
    /// without pivoting.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Returns the sign of `det A` and `ln |det A| = 2 Σ ln L_kk`.
    ///
    /// The phase is always 1, since `det A > 0`. An empty matrix gives
    /// `ln |det A| = 0`.
    ///
    /// # Errors
    ///
    /// - `Singular` if the factor has rank below [`Self::order`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, HermitianInput, Matrix};
    /// // det A = 4 · 9 = 36
    /// let a = Matrix::from_vec(2, 2, vec![4.0_f64, 0.0, 0.0, 9.0])?;
    /// let factor = CholeskyDecomposition::factor(
    ///     &a,
    ///     HermitianInput::Exact,
    ///     CholeskyPivoting::None)?;
    /// assert!((factor.slogdet()?.log_abs() - 36.0_f64.ln()).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn slogdet(&self) -> Result<super::SignLogDeterminant<S>, SciOxideError> {
        self.validate_diagonal()?;
        let mut determinant = scioxide_core::ScaledValue::from_value(S::ONE)?;
        for i in 0..self.n {
            let diagonal = scioxide_core::ScaledValue::from_value(self.l.get_element(i, i))?;
            determinant = determinant.mul(diagonal)?.mul(diagonal)?;
        }
        super::SignLogDeterminant::from_scaled(determinant)
    }

    /// Returns the rank of the factorization and the rank of the pivots `|L_kk|²`
    /// under `policy`.
    ///
    /// The pivots `|L_kk|²` have the scale of the eigenvalues of `A`. Solves use
    /// this rank to accept or reject a solution.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a [`RankPolicy::Threshold`]
    ///   threshold exceeds the largest finite value.
    pub fn definiteness_report(
        &self,
        policy: RankPolicy<S::Real>,
    ) -> Result<CholeskyDefinitenessReport<S::Real>, SciOxideError> {
        if self.permutation.len() != self.n || self.factor_rank.rank() > self.n {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "factor metadata is incomplete",
            ));
        }
        // `|L_kk|^2` are the pivots of `A = L L^H` (the Schur-complement
        // diagonals), i.e. on the same scale as A's eigenvalues, so rank
        // policies are applied to them rather than to `|L_kk|`.
        let effective_rank = analyze_magnitudes(
            (0..self.n).map(|index| self.l.get_element(index, index).norm_sqr()),
            self.n,
            self.n,
            self.n,
            policy,
        )?;
        Ok(CholeskyDefinitenessReport {
            factor_rank: self.factor_rank,
            effective_rank,
        })
    }

    /// Estimates the reciprocal condition number `1 / (‖A‖₁ ‖A⁻¹‖₁)` of the original matrix.
    ///
    /// The method is that of
    /// [`LuDecomposition::condition_estimate`](crate::LuDecomposition::condition_estimate).
    /// A factor of rank below [`Self::order`] gives an estimate of 0.
    ///
    /// # Errors
    ///
    /// - `DomainError` if the matrix is empty.
    /// - `NumericalInstability` (`IllConditioned`) if the estimator cannot
    ///   complete.
    pub fn condition_estimate(&self) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
        self.condition_estimate_scoped(
            &crate::LinalgExecutionOptions::new(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Estimates the reciprocal condition number as [`Self::condition_estimate`] does,
    /// using `workspace` and `context`.
    ///
    /// Each solve with `A` counts toward the iteration budget in `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::condition_estimate`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - Limit errors from `options` and `ExecutionStopped`, as described in
    ///   [Execution](crate#execution).
    pub fn condition_estimate_scoped<'scope, 'capability>(
        &self,
        options: &crate::LinalgExecutionOptions,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
        crate::estimate::scoped_condition(
            options,
            workspace,
            context.into(),
            || {
                Ok(crate::estimate::ConditionQuery {
                    algorithm: scioxide_core::algorithm_id!("linalg", "cholesky_condition", 1),
                    order: self.n,
                    norm: self.one_norm,
                    singular: self.factor_rank.rank() < self.n,
                })
            },
            // The inverse is Hermitian, so adjoint actions coincide.
            |action, kernels| {
                action.solve(
                    |values, temporary| {
                        for (i, &source) in self.permutation.iter().enumerate() {
                            temporary[i] = values[source];
                        }
                        for op in [crate::kernel::GemmOp::None, crate::kernel::GemmOp::Adjoint] {
                            crate::kernel::triangular::solve(
                                crate::kernel::TriangularOperand {
                                    view: crate::kernel::triangular::Storage::Dense(self.l.view()),
                                    triangle: Triangle::Lower,
                                    op,
                                    diagonal: DiagonalKind::Explicit,
                                },
                                self.maximum,
                                &mut temporary[..self.n],
                                self.n.saturating_sub(1),
                                kernels.dispatch(),
                            )?;
                        }
                        for (i, &source) in self.permutation.iter().enumerate() {
                            values[source] = temporary[i];
                        }
                        Ok(())
                    },
                    |rhs, work| {
                        for (i, &original) in self.permutation.iter().enumerate() {
                            work[i] = rhs[original];
                        }
                        for adjoint in [false, true] {
                            crate::diagnostics::matrix_substitute(
                                self.l.view(),
                                scioxide_core::Triangle::Lower,
                                scioxide_core::DiagonalKind::Explicit,
                                adjoint,
                                work,
                            )?;
                        }
                        for (i, &original) in self.permutation.iter().enumerate() {
                            rhs[original] = work[i];
                        }
                        Ok(())
                    },
                )
            },
        )
    }

    /// Solves `A x = b` and returns `x`.
    ///
    /// The two triangular solves cost about 2n² flops. The solve uses the default
    /// [`DirectSolveOptions`](crate::DirectSolveOptions): with
    /// [`RankPolicy::Exact`](crate::RankPolicy::Exact) only an exactly zero pivot,
    /// which a pivoted factorization of a semidefinite matrix can have, rejects the
    /// solve. Use [`Self::solve_with_options`] for a numerical rank cutoff, a
    /// condition threshold or the residual of the solution.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `b` does not have length [`Self::order`].
    /// - `NonFinite` if `b` contains NaN or infinity.
    /// - `Singular` if the factor is rank deficient under `RankPolicy::Exact`.
    /// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, HermitianInput, Matrix};
    /// let a = Matrix::from_rows(&[[4.0_f64]])?;
    /// let factor =
    ///     CholeskyDecomposition::factor(&a, HermitianInput::Exact, CholeskyPivoting::None)?;
    /// let x = factor.solve(&[8.0][..])?;
    /// assert!((x.as_slice()[0] - 2.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.solve_with_options(rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `A x = b`, checks the result against `options` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The solve classifies the rank of the pivots `|L_kk|²` with `options.rank_policy()` (see
    /// [`Self::definiteness_report`]). If `options` sets a minimum reciprocal condition number, it
    /// then calls [`Self::condition_estimate`]. A rank-deficient or ill-conditioned system returns
    /// `Ok` with that [`DirectSolveTermination`](crate::DirectSolveTermination) and no solution;
    /// see [Direct solves](crate#direct-solves). A requested residual is the normwise backward
    /// error of the original system, `‖b − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have length [`Self::order`].
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `DomainError` if the matrix is empty and a condition threshold is set.
    /// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn solve_with_options<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
    ) -> Result<crate::DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let options = &options;
        let mut output = Vector::zeros(self.n)?;
        let report = self.solve_into(
            rhs,
            options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        let solution = output;
        Ok(crate::DirectSolveResult::new(solution, report))
    }

    /// Solves `A x = b` into `output`, the reusable form of [`Self::solve`].
    ///
    /// Reserve [`CholeskyWorkspaceRequest::solve`] in `workspace` beforehand to
    /// avoid allocation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have length [`Self::order`].
    /// - The errors of [`Self::solve`] and the workspace and execution errors of
    ///   [`Self::factor_into`].
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
        mut output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();

        let scope_context = context.into().with_options(&options.execution_options())?;
        let context = scope_context.kernel();
        context.execution().check()?;
        let rhs = rhs.into();
        crate::checked::expect_shape([self.n], [rhs.len()])?;
        crate::checked::expect_shape([self.n], [output.len()])?;
        let rhs = rhs.try_finite()?;
        let rank = self
            .definiteness_report(options.rank_policy())?
            .effective_rank();
        workspace.try_reserve(&Self::workspace_requirements(
            &CholeskyWorkspaceRequest::solve(self.n, &options.execution_options())
                .for_kernels(context),
        )?)?;
        if self.n == 0 && options.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "cholesky_solve", 1),
                self.n as u128,
                0,
            ),
            options.execution_options().solve_budgets(),
            |workspace, context, solve| {
                let condition = options
                    .minimum_reciprocal_condition()
                    .map(|_| {
                        self.condition_estimate_scoped(
                            &options.execution_options(),
                            workspace,
                            context.scoped(&mut *solve),
                        )
                    })
                    .transpose()?;
                let termination =
                    crate::analysis::acceptance(rank, condition, options).termination();
                if termination != crate::DirectSolveTermination::Accepted || self.n == 0 {
                    return Ok(crate::DirectSolveReport::new(
                        termination,
                        rank,
                        condition,
                        if self.n == 0 {
                            crate::analysis::empty_residual(options, 1)?
                        } else {
                            None
                        },
                    ));
                }
                crate::solve_support::admit_iteration(
                    Some(context.execution()),
                    Some(solve),
                    "direct solve budget exhausted",
                )?;
                workspace.with_direct(|storage, _, _, gemm| {
                    let (solution, scratch) = crate::workspace::split_mut(storage, self.n)?;
                    for (slot, &source) in solution.iter_mut().zip(self.permutation.iter()) {
                        *slot = rhs.as_view().get_element(source);
                    }
                    self.solve_candidate(solution, 1, gemm, context)?;
                    crate::checked::validate_values(
                        solution,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                        "Cholesky solution is not finite",
                    )?;
                    let residual = (options.residual() == crate::Estimation::Estimate)
                        .then(|| self.backward_error(solution, rhs.as_view(), scratch, context))
                        .transpose()?;
                    context.execution().check()?;
                    for (row, &destination) in self.permutation.iter().enumerate() {
                        output.set_element(destination, solution[row]);
                    }
                    Ok(crate::DirectSolveReport::new(
                        termination,
                        rank,
                        condition,
                        residual,
                    ))
                })
            },
        )
    }

    fn backward_error(
        &self,
        solution: &[S],
        rhs: crate::VectorView<'_, S>,
        scratch: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        for (row, &source) in self.permutation.iter().enumerate() {
            scratch[source] = solution[row];
        }
        crate::diagnostics::matrix_backward_error(
            self.source.view(),
            self.one_norm,
            crate::VectorView::from_slice(&scratch[..self.n]),
            rhs,
            context,
        )
    }

    fn validate_diagonal(&self) -> Result<(), SciOxideError> {
        // Successful preparation establishes finite factors and valid pivots;
        // only a deliberately retained semidefinite factor is unsolvable.
        if self.factor_rank.rank() != self.n {
            return Err(scioxide_core::SciOxideError::singular(None));
        }
        Ok(())
    }

    fn solve_candidate(
        &self,
        candidate: &mut [S],
        columns: usize,
        gemm: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        #[cfg(feature = "blas")]
        let native = context.blas().is_none();
        #[cfg(not(feature = "blas"))]
        let native = true;
        if columns == 1 && context.accuracy() == scioxide_core::Accuracy::Standard && native {
            for op in [GemmOp::None, GemmOp::Adjoint] {
                crate::kernel::triangular::solve(
                    crate::kernel::TriangularOperand {
                        view: crate::kernel::triangular::Storage::Dense(self.l.view()),
                        triangle: Triangle::Lower,
                        op,
                        diagonal: DiagonalKind::Explicit,
                    },
                    self.maximum,
                    candidate,
                    self.n.saturating_sub(1),
                    context.dispatch(),
                )?;
            }
            return Ok(());
        }
        let factor = crate::FiniteMatrixView::from_validated(self.l.view(), self.maximum);
        let mut output = crate::MatrixViewMut::from_slice(self.n, columns, candidate)?;
        trsm_bounded(
            crate::kernel::TriangularOperand {
                view: factor,
                triangle: Triangle::Lower,
                op: GemmOp::None,
                diagonal: DiagonalKind::Explicit,
            },
            output.reborrow(),
            MatrixSide::Left,
            S::ONE,
            gemm,
            context,
        )?;
        trsm_bounded(
            crate::kernel::TriangularOperand {
                view: factor,
                triangle: Triangle::Lower,
                op: GemmOp::Adjoint,
                diagonal: DiagonalKind::Explicit,
            },
            output,
            MatrixSide::Left,
            S::ONE,
            gemm,
            context,
        )
        .map(|_| ())
    }
}

fn diagonal_equilibration_exponent<R: RealScalar>(magnitude: R) -> Result<i32, SciOxideError> {
    if magnitude == R::ZERO {
        return Ok(0);
    }
    let (significand, exponent, _) = crate::checked::finite_parts(magnitude)?;
    let top_bit = (u64::BITS - 1 - significand.leading_zeros()) as i32;
    // The diagonal is scaled by `2^(2e)` (D appears on both sides), so the
    // exponent is half of `-floor(log2 |a_ii|)`, rounded toward -infinity.
    Ok((-(i32::from(exponent) + top_bit)).div_euclid(2))
}

fn factor_unpivoted<S: NumericScalar>(
    workspace: &mut CholeskyScratch<'_, '_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let order = workspace.candidate.rows();
    // Blocked right-looking Cholesky (xPOTRF). Each panel `start..end` and the
    // rows below it are gathered into a contiguous row-major buffer, where the
    // diagonal block is factored unblocked and `L21 = A21 L11^{-H}` is solved
    // (right-side TRSM). The rank-k update `A22 -= L21 L21^H` of the lower
    // trailing triangle reads L21 from the buffer, which is then scattered back.
    let mut bound = TrailingBound::new(crate::kernel::standard::component_maximum(
        workspace.candidate.as_view(),
        Some(Triangle::Lower),
    ));
    let mut start = 0;
    while start < order {
        context.execution().check()?;
        let end = (start + crate::kernel::panel_width::<S>(context, MAX_PANEL)).min(order);
        let (width, rows) = (end - start, order - start);
        let panel = &mut workspace.lower_panel[..rows * width];
        transfer_panel(workspace.candidate.data, order, start..end, panel, true);
        let (diagonal, below) = panel.split_at_mut(width * width);
        factor_diagonal_leaf(diagonal, width, 0, width)?;
        if end < order {
            let mut below = crate::MatrixViewMut::from_slice(rows - width, width, below)?;
            trsm_bounded(
                crate::kernel::TriangularOperand {
                    view: crate::FiniteMatrixView::from_workspace(crate::MatrixView::from_slice(
                        width, width, diagonal,
                    )?),
                    triangle: Triangle::Lower,
                    op: GemmOp::Adjoint,
                    diagonal: DiagonalKind::Explicit,
                },
                below.reborrow(),
                MatrixSide::Right,
                S::ONE,
                workspace.gemm,
                context,
            )?;
            let below = below.as_view();
            let below = crate::FiniteMatrixView::from_validated(
                below,
                crate::kernel::standard::finite_component_maximum(below)?,
            );
            update_trailing_lower(
                below,
                workspace
                    .candidate
                    .block_mut(end, end, order - end, order - end)?,
                bound.prior(),
                policy,
                workspace.gemm,
                context,
            )?;
            bound = bound.update(width, below.maximum, below.maximum);
        }
        transfer_panel(workspace.candidate.data, order, start..end, panel, false);
        start = end;
    }
    Ok(())
}

fn factor_pivoted<S: NumericScalar>(
    workspace: &mut CholeskyScratch<'_, '_, S>,
    threshold: S::Real,
    floor: S::Real,
    scale: S::Real,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<usize, SciOxideError> {
    // Blocked xPSTRF-style panels. Source holds the Schur complement at the
    // panel boundary; diagonals are updated once per accepted column.
    let order = workspace.source.rows();
    workspace.candidate.data.fill(S::ZERO);
    for (row, diagonal) in workspace.residual_diagonal.iter_mut().enumerate() {
        *diagonal = workspace.source.get_element(row, row).real_component();
    }
    let cutoff = scioxide_core::Scaled::from_value(threshold)?;
    // A Schur complement diagonal at or below `floor = min(n ε, 1/2) s` is
    // rounding noise of the earlier steps: eliminating it would divide noise
    // by its square root. Elimination ends there under every policy.
    let validation_cutoff = scioxide_core::Scaled::from_value(floor)?;
    let rounding =
        scioxide_core::Scaled::from_value(scale)?.mul(scioxide_core::Scaled::from_value(
            crate::checked::dimension_epsilon::<S::Real>(order.saturating_mul(8))?,
        )?)?;
    let mut stopped_rank = None;
    let mut start = 0;
    while start < order {
        let end = (start + crate::kernel::panel_width::<S>(context, MAX_PANEL)).min(order);
        for column in start..end {
            context.execution().check()?;
            // The working matrix is equilibrated; pivot order and rank policy
            // are evaluated after undoing each diagonal's congruence scaling.
            let original_diagonal = |row: usize| {
                scioxide_core::Scaled::from_value(workspace.residual_diagonal[row])?.shifted(
                    -2 * i64::from(workspace.scaling_exponents[workspace.permutation[row]] as i32),
                )
            };
            let mut pivot = column;
            let mut best = original_diagonal(column)?;
            for row in column..order {
                let value = original_diagonal(row)?;
                if value.is_sign_negative() && !value.abs_le(rounding) {
                    return Err(not_positive_semidefinite());
                }
                if value.cmp_value(best).is_gt() {
                    pivot = row;
                    best = value;
                }
            }
            if best.cmp_value(cutoff).is_le() && stopped_rank.is_none() {
                stopped_rank = Some(column);
            }
            // A coarse user cutoff cannot hide an indefinite trailing block.
            // Continue validation to rounding scale, then discard every column
            // at or beyond the requested stop. A finer cutoff stops here too.
            if best.cmp_value(validation_cutoff).is_le() {
                // Zero diagonals cannot hide a nonzero off-diagonal Schur block.
                let bound = validation_cutoff.add(rounding)?;
                for row in column..order {
                    for col in column..=row {
                        let mut value = scioxide_core::ScaledValue::from_value(
                            workspace.source.get_element(row, col),
                        )?;
                        for inner in start..column {
                            value = value.sub(
                                scioxide_core::ScaledValue::from_value(
                                    workspace.candidate.get_element(row, inner),
                                )?
                                .mul(
                                    scioxide_core::ScaledValue::from_value(
                                        workspace.candidate.get_element(col, inner).conj(),
                                    )?,
                                )?,
                            )?;
                        }
                        let exponent = -(i64::from(
                            workspace.scaling_exponents[workspace.permutation[row]] as i32,
                        ) + i64::from(
                            workspace.scaling_exponents[workspace.permutation[col]] as i32,
                        ));
                        if !value.shifted(exponent)?.magnitude()?.abs_le(bound) {
                            return Err(not_positive_semidefinite());
                        }
                    }
                }
                let rank = stopped_rank.unwrap_or(column);
                for row in 0..order {
                    workspace.candidate.data[row * order + rank..(row + 1) * order].fill(S::ZERO);
                }
                return Ok(rank);
            }
            let best = workspace.residual_diagonal[pivot];
            if pivot != column {
                workspace.permutation.swap(column, pivot);
                workspace.residual_diagonal.swap(column, pivot);
                for inner in 0..column {
                    workspace
                        .candidate
                        .data
                        .swap(column * order + inner, pivot * order + inner);
                }
                // Symmetric swap of the panel-boundary Schur complement.
                for inner in 0..order {
                    workspace
                        .source
                        .data
                        .swap(column * order + inner, pivot * order + inner);
                }
                for row in 0..order {
                    workspace
                        .source
                        .data
                        .swap(row * order + column, row * order + pivot);
                }
            }
            let diagonal = S::from_real(best.sqrt());
            workspace.candidate.data[column * order + column] = diagonal;
            for row in column + 1..order {
                let remainder = (start..column).fold(
                    workspace.source.get_element(row, column),
                    |value, inner| {
                        value
                            - workspace.candidate.get_element(row, inner)
                                * workspace.candidate.get_element(column, inner).conj()
                    },
                );
                let value = remainder / diagonal;
                workspace.candidate.data[row * order + column] = value;
                let real = value.real_component();
                let imaginary = value.imaginary_component();
                workspace.residual_diagonal[row] =
                    workspace.residual_diagonal[row] - real * real - imaginary * imaginary;
            }
        }
        if end < order {
            let panel = copy_panel(
                workspace
                    .candidate
                    .as_view()
                    .block(end, start, order - end, end - start)?,
                workspace.lower_panel,
            )?;
            update_trailing_lower(
                panel,
                workspace
                    .source
                    .block_mut(end, end, order - end, order - end)?,
                None,
                policy,
                workspace.gemm,
                context,
            )?;
            for row in end..order {
                for col in end..row {
                    workspace.source.data[col * order + row] =
                        workspace.source.get_element(row, col).conj();
                }
            }
        }
        start = end;
    }
    let rank = stopped_rank.unwrap_or(order);
    for row in 0..order {
        workspace.candidate.data[row * order + rank..(row + 1) * order].fill(S::ZERO);
    }
    Ok(rank)
}

fn not_positive_semidefinite() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("a"),
        scioxide_core::DomainReason::OutsideDomain,
        "matrix is not positive semidefinite",
    )
}

/// An unpivoted factorization met a nonpositive pivot.
fn not_positive_definite(argument: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::domain_error(
        argument,
        scioxide_core::DomainReason::OutsideDomain,
        "matrix is not numerically positive definite",
    )
}

fn factor_diagonal_leaf<S: NumericScalar>(
    matrix: &mut [S],
    order: usize,
    start: usize,
    end: usize,
) -> Result<(), SciOxideError> {
    // Unblocked Cholesky of the diagonal block `start..end`. Columns before
    // `start` were already folded into these entries by the trailing
    // updates, so only the in-block columns `start..column` are subtracted.
    // A non-real stored diagonal can only arise from rounding in those
    // updates and is reported as a breakdown.
    for column in start..end {
        let stored_diagonal = matrix[column * order + column];
        if stored_diagonal != stored_diagonal.conj() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "Cholesky pivot lost Hermitian structure",
            ));
        }
        let real = (start..column).fold(stored_diagonal.real_component(), |value, inner| {
            let entry = matrix[column * order + inner];
            value
                - entry.real_component() * entry.real_component()
                - entry.imaginary_component() * entry.imaginary_component()
        });
        if !real.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Intermediate,
                "Cholesky pivot is not finite",
            ));
        }
        if real <= S::Real::ZERO {
            return Err(not_positive_definite(scioxide_core::argument!("a")));
        }
        let factor_diagonal = S::from_real(real.sqrt());
        matrix[column * order + column] = factor_diagonal;
        for row in column + 1..end {
            let remainder = (start..column).fold(matrix[row * order + column], |value, inner| {
                value - matrix[row * order + inner] * matrix[column * order + inner].conj()
            });
            let value = crate::checked::finite_candidate(remainder / factor_diagonal)?;
            matrix[row * order + column] = value;
        }
    }
    Ok(())
}

fn update_trailing_lower<S: NumericScalar>(
    panel: crate::FiniteMatrixView<'_, S>,
    trailing: crate::dense::MatrixViewMut<'_, S>,
    prior: Option<S::Real>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // Triangle-masked GEMM is SYRK for real scalars and HERK for complex.
    let mut trailing = trailing;
    gemm_workspace_with_context(
        crate::kernel::WorkspaceGemmProblem {
            left: panel,
            right: panel,
            output: trailing.reborrow(),
            config: GemmConfig {
                right_op: if S::IS_REAL {
                    GemmOp::Transpose
                } else {
                    GemmOp::Adjoint
                },
                alpha: -S::ONE,
                beta: S::ONE,
                triangle: Some(Triangle::Lower),
                prior,
                ..GemmConfig::ordinary()
            },
        },
        policy,
        workspace,
        context,
    )?;
    for index in 0..trailing.rows() {
        let value = trailing.get_element(index, index).real_component();
        trailing.set_element(index, index, S::from_real(value));
    }
    Ok(())
}

/// The operator `x ↦ A⁻¹ x` of a [`CholeskyDecomposition`], returned by
/// [`CholeskyDecomposition::inverse_operator`].
///
/// It implements [`LinearOperator`](scioxide_core::LinearOperator),
/// [`AdjointOperator`](scioxide_core::AdjointOperator) (the same map, since `A⁻¹`
/// is Hermitian) and [`Preconditioner`](scioxide_core::Preconditioner). Each
/// application needs `n` scratch scalars. Calls without a context run on the
/// calling thread. Applying the inverse of a factor with rank below `n`
/// returns `Singular`.
#[derive(Debug, Clone, Copy)]
pub struct CholeskyInverseOperator<'a, S: NumericScalar>(&'a CholeskyDecomposition<S>);
impl<S: NumericScalar> CholeskyDecomposition<S> {
    /// Returns this factorization as the operator `x ↦ A⁻¹ x`, for example to use
    /// as a preconditioner.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::LinearOperator;
    /// use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting, HermitianInput, Matrix};
    /// let a = Matrix::from_vec(1, 1, vec![4.0_f64])?;
    /// let factor = CholeskyDecomposition::factor(
    ///     &a,
    ///     HermitianInput::Exact,
    ///     CholeskyPivoting::None)?;
    /// let mut x = [0.0];
    /// factor.inverse_operator().apply_into(&[8.0], &mut x, &mut [0.0])?;
    /// assert_eq!(x, [2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn inverse_operator(&self) -> CholeskyInverseOperator<'_, S> {
        CholeskyInverseOperator(self)
    }
}
impl<S: NumericScalar> CholeskyInverseOperator<'_, S> {
    fn apply(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        kernels.execution().check()?;
        let n = self.0.n;
        if input.len() != n || output.len() != n || scratch.len() < n {
            return Err(SciOxideError::dimension_mismatch(
                &[n, n, n],
                &[input.len(), output.len(), scratch.len()],
            ));
        }
        let input =
            crate::VectorView::from_slice(input).try_finite_dispatched(kernels.dispatch())?;
        self.0.validate_diagonal()?;
        let candidate = &mut scratch[..n];
        for (i, slot) in candidate.iter_mut().enumerate() {
            *slot = input.as_view().get_element(self.0.permutation[i]);
        }
        self.0
            .solve_candidate(candidate, 1, &mut LinalgWorkspace::new(), kernels)?;
        crate::checked::validate_values(
            candidate,
            scioxide_core::NonFiniteOrigin::Intermediate,
            "Cholesky inverse action is not finite",
        )?;
        kernels.execution().check()?;
        for (i, &value) in candidate.iter().enumerate() {
            output[self.0.permutation[i]] = value;
        }
        Ok(())
    }
}
impl<S: NumericScalar> scioxide_core::LinearOperator<S> for CholeskyInverseOperator<'_, S> {
    fn rows(&self) -> usize {
        self.0.n
    }
    fn cols(&self) -> usize {
        self.0.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.0.n)
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, scratch, KernelContext::sequential_native())
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(input, output, scratch, KernelContext::for_operator(context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: NumericScalar> scioxide_core::AdjointOperator<S> for CholeskyInverseOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.0.n)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, scratch, KernelContext::sequential_native())
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(input, output, scratch, KernelContext::for_operator(context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: NumericScalar> scioxide_core::Preconditioner<S> for CholeskyInverseOperator<'_, S> {
    fn dimension(&self) -> usize {
        self.0.n
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.0.n)
    }
    fn precondition_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, scratch, KernelContext::sequential_native())
    }
    fn precondition_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<(), SciOxideError> {
        self.apply(input, output, scratch, KernelContext::for_operator(context))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::HermitianProjectionPolicy;
    use scioxide_core::Complex;
    use scioxide_core::Tolerance;
    use scioxide_core::Triangle;

    fn factor<S: NumericScalar>(
        matrix: &Matrix<S>,
    ) -> Result<CholeskyDecomposition<S>, SciOxideError> {
        CholeskyDecomposition::factor(matrix, crate::HermitianInput::Exact, CholeskyPivoting::None)
    }

    // ---- Identity matrix ----

    #[test]
    fn test_decompose_identity() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let chol = factor(&a).unwrap();
        let l = chol.l_matrix();
        let eye = Matrix::<f64>::identity(3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (l.view().get(i, j).unwrap() - eye.view().get(i, j).unwrap()).abs() < 1e-12,
                    "L != I at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn blocked_equilibration_handles_unrepresentable_power_and_late_failure() {
        let order = 33;
        let scales = [f64::from_bits(1), f64::MIN_POSITIVE, 1.0, f64::MAX];
        let mut a = Matrix::zeros(order, order).unwrap();
        for i in 0..order {
            a.view_mut().set(i, i, scales[i % scales.len()]).unwrap();
        }
        let mut result = factor(&a).unwrap();
        for i in 0..order {
            let expected = scales[i % scales.len()].sqrt();
            assert!((result.l_matrix().view().get(i, i).unwrap() / expected - 1.0).abs() < 2e-15);
        }
        let before = result.l_matrix().clone();
        let options = crate::LinalgExecutionOptions::new();
        let mut workspace = LinalgWorkspace::new();
        a.view_mut().set(order - 1, order - 1, -1.0).unwrap();
        assert!(
            CholeskyDecomposition::factor_into(
                &a,
                crate::HermitianInput::Exact,
                CholeskyPivoting::None,
                &options,
                &mut result,
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
        );
        assert_eq!(result.l_matrix(), &before);
        a.view_mut().set(order - 1, order - 1, scales[0]).unwrap();
        CholeskyDecomposition::factor_into(
            &a,
            crate::HermitianInput::Exact,
            CholeskyPivoting::None,
            &options,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(result.l_matrix(), &before);
    }

    // ---- 2x2 SPD ----

    #[test]
    fn test_decompose_2x2_spd() {
        // A = [[4, 2], [2, 3]]
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let chol = factor(&a).unwrap();
        let l = chol.l_matrix();
        let lt = l.view().adjoint().try_to_matrix().unwrap();
        let llt = l.view().mul_mat(&lt).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llt.view().get(i, j).unwrap() - a.view().get(i, j).unwrap()).abs() < 1e-12,
                    "L*L^T != A at ({i}, {j})"
                );
            }
        }
    }

    // ---- 3x3 SPD ----

    #[test]
    fn test_decompose_3x3_spd() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // Known Cholesky: L = [[2,0,0],[6,1,0],[-8,5,3]]
        let a = Matrix::<f64>::from_vec(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        )
        .unwrap();
        let chol = factor(&a).unwrap();
        let l = chol.l_matrix();
        let lt = l.view().adjoint().try_to_matrix().unwrap();
        let llt = l.view().mul_mat(&lt).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (llt.view().get(i, j).unwrap() - a.view().get(i, j).unwrap()).abs() < 1e-12,
                    "L*L^T != A at ({i}, {j})"
                );
            }
        }

        // Verify known L values
        let expected_l = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]];
        for (i, row) in expected_l.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    (l.view().get(i, j).unwrap() - val).abs() < 1e-12,
                    "L[{i},{j}] = {}, expected {val}",
                    l.view().get(i, j).unwrap()
                );
            }
        }
    }

    // ---- Solve 2x2 ----

    #[test]
    fn test_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [8, 7] => x = [1, 1] (since 4*1+2*1=6? no)
        // Actually: A*[1,1] = [6,5], so use b = [6,5]
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let chol = factor(&a).unwrap();
        let b = Vector::from_vec(vec![6.0, 5.0]);
        let x = chol.solve(&b).unwrap();
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-12);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-12);
    }

    // ---- Solve identity ----

    #[test]
    fn test_solve_identity() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let chol = factor(&a).unwrap();
        let b = Vector::from_vec(vec![7.0, 8.0, 9.0]);
        let x = chol.solve(&b).unwrap();
        for i in 0..3 {
            assert!(
                (x.view().get(i).unwrap() - b.view().get(i).unwrap()).abs() < 1e-12,
                "x[{i}] = {}, expected {}",
                x.view().get(i).unwrap(),
                b.view().get(i).unwrap()
            );
        }
    }

    // ---- Inverse 2x2 ----

    #[test]
    fn test_inverse_2x2() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let chol = factor(&a).unwrap();
        let inv = chol.inverse().unwrap();
        let prod = a.view().mul_mat(&inv).unwrap();
        let eye = Matrix::<f64>::identity(2).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (prod.view().get(i, j).unwrap() - eye.view().get(i, j).unwrap()).abs() < 1e-12,
                    "A*A^-1 != I at ({i}, {j})"
                );
            }
        }
    }

    // ---- Error: not positive definite ----

    #[test]
    fn test_non_spd_error() {
        // Symmetric but not positive definite: eigenvalues of [[-1,0],[0,-1]] are negative
        let a = Matrix::<f64>::from_vec(2, 2, vec![-1.0, 0.0, 0.0, -1.0]).unwrap();
        let result = factor(&a);
        assert!(
            matches!(
                result.as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ),
            "expected DomainError, got {result:?}"
        );
    }

    // ---- Error: not symmetric ----

    #[test]
    fn test_non_symmetric_error() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 2.0, 3.0, 3.0]).unwrap();
        let result = factor(&a);
        assert!(
            matches!(
                result.as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ),
            "expected InvalidInput, got {result:?}"
        );
    }

    // ---- Error: not square ----

    #[test]
    fn test_non_square_error() {
        let a = Matrix::<f64>::zeros(2, 3).unwrap();
        let result = factor(&a);
        assert!(
            matches!(
                result.as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
            ),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    // ---- Error: solve dimension mismatch ----

    #[test]
    fn test_solve_dimension_mismatch() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let chol = factor(&a).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (chol.solve(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    // ---- f32 test ----

    #[test]
    fn test_decompose_f32() {
        let a = Matrix::<f32>::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let chol = factor(&a).unwrap();
        let l = chol.l_matrix();
        let lt = l.view().adjoint().try_to_matrix().unwrap();
        let llt = l.view().mul_mat(&lt).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llt.view().get(i, j).unwrap() - a.view().get(i, j).unwrap()).abs() < 1e-5,
                    "L*L^T != A at ({i}, {j}) for f32"
                );
            }
        }
    }

    // ---- L is lower triangular ----

    #[test]
    fn test_l_matrix_lower_triangular() {
        let a = Matrix::<f64>::from_vec(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        )
        .unwrap();
        let chol = factor(&a).unwrap();
        let l = chol.l_matrix();
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    l.view().get(i, j).unwrap().abs() < 1e-14,
                    "L[{i},{j}] = {} should be zero (upper triangle)",
                    l.view().get(i, j).unwrap()
                );
            }
        }
    }

    fn assert_scaled_spd_f64(scale: f64) {
        let a = Matrix::from_vec(
            2,
            2,
            vec![4.0 * scale, 2.0 * scale, 2.0 * scale, 3.0 * scale],
        )
        .unwrap();
        let rhs = Vector::from_vec(vec![6.0 * scale, 5.0 * scale]);
        let x = factor(&a).unwrap().solve(&rhs).unwrap();
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-12);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-12);
    }

    fn assert_scaled_spd_f32(scale: f32) {
        let a = Matrix::from_vec(
            2,
            2,
            vec![4.0 * scale, 2.0 * scale, 2.0 * scale, 3.0 * scale],
        )
        .unwrap();
        let rhs = Vector::from_vec(vec![6.0 * scale, 5.0 * scale]);
        let x = factor(&a).unwrap().solve(&rhs).unwrap();
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-5);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_aware_positive_definiteness_f64() {
        assert_scaled_spd_f64(1e-200);
        assert_scaled_spd_f64(1e200);

        let mixed = Matrix::from_vec(2, 2, vec![1e200, 0.0, 0.0, 1e-200]).unwrap();
        assert!(factor(&mixed).is_ok());
    }

    #[test]
    fn test_scale_aware_positive_definiteness_f32() {
        assert_scaled_spd_f32(1e-20);
        assert_scaled_spd_f32(1e20);

        let mixed = Matrix::from_vec(2, 2, vec![1e20_f32, 0.0, 0.0, 1e-20]).unwrap();
        assert!(factor(&mixed).is_ok());
    }

    #[test]
    fn near_semidefinite_construction_is_separate_from_rank_policy() {
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let a = Matrix::from_vec(
                2,
                2,
                vec![scale, scale, scale, scale * (1.0 + f64::EPSILON)],
            )
            .unwrap();
            let decomposition = factor(&a).unwrap();
            assert_eq!(
                decomposition
                    .definiteness_report(RankPolicy::Exact)
                    .unwrap()
                    .factor_rank()
                    .rank(),
                2
            );
        }

        for scale in [2.0_f32.powi(-60), 1.0, 2.0_f32.powi(60)] {
            let a = Matrix::from_vec(
                2,
                2,
                vec![scale, scale, scale, scale * (1.0 + f32::EPSILON)],
            )
            .unwrap();
            let decomposition = factor(&a).unwrap();
            assert_eq!(
                decomposition
                    .definiteness_report(RankPolicy::Exact)
                    .unwrap()
                    .factor_rank()
                    .rank(),
                2
            );
        }
    }

    #[test]
    fn exact_and_projected_symmetry_are_distinct() {
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let within_policy = Matrix::from_vec(
                2,
                2,
                vec![
                    4.0 * scale,
                    2.0 * scale,
                    2.0 * scale * (1.0 + f64::EPSILON),
                    3.0 * scale,
                ],
            )
            .unwrap();
            assert!(matches!(
                (factor(&within_policy)).as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
            let tolerance = Tolerance::new(0.0, 2.0 * f64::EPSILON).unwrap();
            let projected_factor = CholeskyDecomposition::factor(
                &within_policy,
                crate::HermitianInput::Projected(HermitianProjectionPolicy::Within(tolerance)),
                CholeskyPivoting::None,
            )
            .unwrap();
            assert!(
                projected_factor
                    .projection_report()
                    .unwrap()
                    .relative_frobenius()
                    <= 2.0 * f64::EPSILON
            );

            let asymmetric = Matrix::from_vec(
                2,
                2,
                vec![4.0 * scale, 2.0 * scale, 2.000_002 * scale, 3.0 * scale],
            )
            .unwrap();
            assert!(matches!(
                (factor(&asymmetric)).as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
        }
    }

    #[test]
    fn test_non_finite_input_is_structural_error() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, f64::INFINITY]).unwrap();
        assert!(matches!(
            (factor(&a)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn blocked_complex_factorization_reconstructs_and_solves_multiple_rhs() {
        for order in [15, 16, 17, 31, 32, 33, 65] {
            reconstruct_and_solve(order);
        }
    }

    fn reconstruct_and_solve(order: usize) {
        let lower = Matrix::from_vec(
            order,
            order,
            (0..order)
                .flat_map(|row| {
                    (0..order).map(move |col| {
                        if row == col {
                            Complex::new(3.0 + row as f64 / order as f64, 0.0)
                        } else if row > col {
                            Complex::new(
                                ((row * 7 + col * 3) % 11) as f64 / 40.0 - 0.1,
                                ((row * 5 + col * 13) % 7) as f64 / 50.0 - 0.06,
                            )
                        } else {
                            Complex::new(0.0, 0.0)
                        }
                    })
                })
                .collect(),
        )
        .unwrap();
        let mut a = Matrix::zeros(order, order).unwrap();
        lower
            .view()
            .mul_mat_into(
                &lower.view().adjoint().try_to_matrix().unwrap(),
                &crate::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential)
                    .with_accuracy(scioxide_core::Accuracy::Exact),
                a.view_mut(),
                &mut LinalgWorkspace::new(),
                crate::LinalgExecutionContext::unscoped(crate::KernelContext::sequential()),
            )
            .unwrap();
        let factor = factor(&a).unwrap();
        for (&actual, &expected) in factor.l_matrix().as_slice().iter().zip(lower.as_slice()) {
            assert!((actual - expected).norm() < 1e-10);
        }

        let rhs = Matrix::from_vec(
            order,
            3,
            (0..order * 3)
                .map(|index| Complex::new((index % 13) as f64 / 7.0, -0.25))
                .collect(),
        )
        .unwrap();
        let solution = factor.solve_matrix(&rhs).unwrap();
        let residual = a.view().mul_mat(&solution).unwrap();
        for (&actual, &expected) in residual.as_slice().iter().zip(rhs.as_slice()) {
            assert!((actual - expected).norm() < 1e-9);
        }
    }

    #[test]
    fn failed_reusable_factorization_preserves_previous_factor() {
        let valid = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let invalid = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let non_hermitian = Matrix::from_vec(2, 2, vec![4.0, 2.0, 3.0, 3.0]).unwrap();
        let mut factor = factor(&valid).unwrap();
        let before = factor.l_matrix().clone();
        let mut workspace = LinalgWorkspace::new();
        assert!(matches!(
            (CholeskyDecomposition::factor_into(
                &invalid,
                crate::HermitianInput::Exact,
                CholeskyPivoting::None,
                &crate::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut factor,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
        assert_eq!(factor.l_matrix(), &before);
        assert!(matches!(
            (CholeskyDecomposition::factor_into(
                &non_hermitian,
                crate::HermitianInput::Exact,
                CholeskyPivoting::None,
                &crate::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut factor,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(factor.l_matrix(), &before);

        let rhs = Vector::from_vec(vec![f64::INFINITY, 1.0]);
        let mut output = Vector::from_vec(vec![7.0, 9.0]);
        assert!(matches!(
            (factor
                .solve_into(
                    &rhs,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        assert_eq!(output.as_slice(), &[7.0, 9.0]);
    }

    #[test]
    fn triangle_interpretation_never_reads_the_opposite_triangle() {
        let conflicting = Matrix::from_vec(2, 2, vec![4.0_f64, 99.0, 2.0, 3.0]).unwrap();
        assert!(matches!(
            (factor(&conflicting)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        let lower = CholeskyDecomposition::factor(
            &conflicting,
            crate::HermitianInput::Triangle(Triangle::Lower),
            CholeskyPivoting::None,
        )
        .unwrap();
        let reconstructed = lower
            .l_matrix()
            .view()
            .mul_mat(&lower.l_matrix().view().adjoint().try_to_matrix().unwrap())
            .unwrap();
        assert_eq!(reconstructed.get_element(0, 1), 2.0);
        assert_eq!(reconstructed.get_element(1, 0), 2.0);
    }

    #[test]
    fn pivoted_panels_reconstruct_across_block_boundaries() {
        let block = crate::kernel::panel_width::<f64>(KernelContext::sequential(), MAX_PANEL);
        for order in [block - 1, block, block + 7] {
            let matrix = Matrix::from_vec(
                order,
                order,
                (0..order)
                    .flat_map(|row| {
                        (0..order).map(move |col| {
                            if row == col {
                                2.0 + (row % 3) as f64 * 0.25
                            } else {
                                0.125 / (row.abs_diff(col) as f64 + 1.0)
                            }
                        })
                    })
                    .collect(),
            )
            .unwrap();
            let factor = CholeskyDecomposition::factor(
                &matrix,
                crate::HermitianInput::Exact,
                CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
            )
            .unwrap();
            let product = factor
                .l_matrix()
                .view()
                .mul_mat(&factor.l_matrix().view().adjoint().try_to_matrix().unwrap())
                .unwrap();
            for row in 0..order {
                for col in 0..order {
                    let expected =
                        matrix.get_element(factor.permutation()[row], factor.permutation()[col]);
                    assert!((product.get_element(row, col) - expected).abs() < 1e-12);
                }
            }
            let mut semidefinite = Matrix::<f64>::identity(order).unwrap();
            semidefinite.as_mut_slice()[0] = 0.0;
            semidefinite.as_mut_slice()[(order - 1) * order + order - 1] = 0.0;
            let factor = CholeskyDecomposition::factor(
                &semidefinite,
                crate::HermitianInput::Exact,
                CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
            )
            .unwrap();
            assert_eq!(
                factor
                    .definiteness_report(RankPolicy::Exact)
                    .unwrap()
                    .factor_rank()
                    .rank(),
                order - 2
            );
        }
    }

    #[test]
    fn pivoted_cutoff_uses_original_diagonal_scale() {
        let a = Matrix::from_vec(
            2,
            2,
            vec![1.0_f64.scalbn(200), 0.0, 0.0, 1.0_f64.scalbn(-200)],
        )
        .unwrap();
        let truncated = CholeskyDecomposition::factor(
            &a,
            crate::HermitianInput::Exact,
            CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
        )
        .unwrap();
        let report = truncated
            .definiteness_report(RankPolicy::Exact)
            .unwrap()
            .factor_rank();
        assert_eq!(report.rank(), 1);
        assert!((report.threshold() / (2.0 * f64::EPSILON * a.as_slice()[0]) - 1.0).abs() < 1e-14);
        // The rounding floor `n ε s` ends the factorization under every policy.
        let exact = CholeskyDecomposition::factor(
            &a,
            crate::HermitianInput::Exact,
            CholeskyPivoting::Diagonal(RankPolicy::Exact),
        )
        .unwrap();
        let report = exact
            .definiteness_report(RankPolicy::Exact)
            .unwrap()
            .factor_rank();
        assert_eq!(report.rank(), 1);
        assert!((report.threshold() / (2.0 * f64::EPSILON * a.as_slice()[0]) - 1.0).abs() < 1e-14);
        let unpivoted = factor(&a).unwrap();
        assert_eq!(
            unpivoted
                .definiteness_report(RankPolicy::Exact)
                .unwrap()
                .factor_rank()
                .rank(),
            2
        );
    }

    #[test]
    fn diagonal_pivoting_returns_semidefinite_rank() {
        let semidefinite =
            Matrix::from_vec(3, 3, vec![1.0_f64, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            (factor(&semidefinite)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
        let pivoted = CholeskyDecomposition::factor(
            &semidefinite,
            crate::HermitianInput::Exact,
            CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
        )
        .unwrap();
        let report = pivoted.definiteness_report(RankPolicy::Exact).unwrap();
        assert_eq!(report.factor_rank().rank(), 1);
        assert_eq!(report.effective_rank().rank(), 1);
        assert!(
            pivoted
                .solve(&Vector::from_vec(vec![1.0, 1.0, 0.0]))
                .is_err()
        );
    }

    #[test]
    fn arithmetic_failure_preserves_all_solve_outputs() {
        let matrix = Matrix::from_vec(1, 1, vec![1.0e-320_f64]).unwrap();
        let factor = factor(&matrix).unwrap();
        let rhs = Vector::from_vec(vec![f64::MAX]);
        let mut vector_output = Vector::from_vec(vec![7.0]);
        let mut matrix_output = Matrix::from_vec(1, 1, vec![11.0]).unwrap();
        let mut inverse_output = Matrix::from_vec(1, 1, vec![13.0]).unwrap();
        let mut workspace = LinalgWorkspace::new();

        assert!(matches!(
            (factor
                .solve_into(
                    &rhs,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    vector_output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(vector_output.as_slice(), &[7.0]);

        let matrix_rhs = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
        assert!(matches!(
            (factor
                .solve_matrix_into(
                    &matrix_rhs,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    matrix_output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(matrix_output.as_slice(), &[11.0]);

        assert!(matches!(
            (factor.inverse_into(
                &crate::LinalgExecutionOptions::new(),
                inverse_output.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(inverse_output.as_slice(), &[13.0]);
    }
}
