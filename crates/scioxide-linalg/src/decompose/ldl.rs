//! Blocked `L D Lᴴ` factorization of Hermitian indefinite matrices with
//! bounded rook pivoting.
//!
//! The permutation maps factor indices to input indices: `A[p[i],p[j]] = (L D Lᴴ)[i,j]`.
//! Exact zero pivots are kept in `D`; rank is decided later by the caller's policy.

use crate::workspace::ScalarRegion;
struct FactorExecution<'a, 'scope, 'capability> {
    cx: KernelContext<'capability>,
    policy: scioxide_core::ExecutionPolicy,
    context: &'a mut SolveContext<'scope, 'capability>,
}
use super::util::resize_vec;
use crate::kernel::TrailingBound;
use crate::kernel::execution::execute_chunks;
use crate::kernel::standard;
use crate::kernel::{
    GemmConfig, GemmOp, LinalgWorkspace, Triangle, WorkspaceGemmProblem,
    gemm_workspace_with_context,
};
use crate::{
    FiniteMatrixView, KernelContext, Matrix, MatrixView, Vector, VectorView, VectorViewMut,
};

// Narrow rook panels limit sequential search work; the trailing update uses
// the shared level-3 kernel. Reservation does not depend on the processor.
const MAX_PANEL: usize = 32;
/// Rows of [`updated_column`] reduced together.
const COLUMN_LANES: usize = 4;
use super::hermitian::{HermitianInput, HermitianProjectionReport};
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AdjointOperator, Complex, ExecutionContext, LinearOperator, NumericScalar, Preconditioner,
    Quantity, RealScalar, SciOxideError, SolveContext,
};
use scioxide_core::{Scaled, ScaledValue};

// Extreme-frame multipliers and solution components use an extended-exponent
// representation and are rounded once. As in ordinary arithmetic, a value
// below the normal range rounds to a subnormal number or to zero.
type Wide<S> = ScaledValue<scioxide_core::Complex<<S as NumericScalar>::Real>>;
fn wide<S: NumericScalar>(x: S) -> Result<Wide<S>, SciOxideError> {
    ScaledValue::from_value(Complex::new(x.real_component(), x.imaginary_component()))
}
fn narrow<S: NumericScalar>(x: Wide<S>) -> Result<S, SciOxideError> {
    Ok(S::from_components(
        x.real().round()?,
        x.imaginary().round()?,
    ))
}
fn size(n: usize, m: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    n.checked_mul(m)
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}

use scioxide_sparse::{LdlInertia, LdlPivotBlock};

/// Factorization `Pᵀ A P = L D Lᴴ` of a Hermitian, possibly indefinite or
/// singular, matrix.
///
/// `L` is unit lower triangular, `P` is a symmetric permutation, and `D` is
/// block diagonal with real 1×1 blocks and Hermitian 2×2 blocks. The pivots are
/// chosen by bounded rook pivoting with `α = (1 + √17) / 8`, which bounds the
/// entries of `L`. The factorization is blocked in panels of at most 32
/// columns, costs about n³/3 flops and keeps O(n²) storage. If every real
/// and imaginary part of the entries of `A` is below 1 in modulus, elimination
/// runs on `2^σ A`, where the power of two `2^σ` brings the largest of them
/// into `[1, 2)`, so that tiny and subnormal entries keep full precision;
/// other matrices are not rescaled.
/// This scaling is exact, and [`Self::diagonal`] and [`Self::off_diagonal`]
/// return `D` of `A` itself, rounded once.
///
/// A singular matrix does not make the factorization fail: exact zero pivots
/// are kept in `D`. Rank and inertia are read from the eigenvalues of the blocks
/// of `D`, which by Sylvester's law of inertia have the signs of the
/// eigenvalues of `A`.
///
/// # Accuracy
///
/// The computed factors are the exact factors of `Pᵀ (A + ΔA) P` with
/// `‖ΔA‖∞ ≤ c n ε ‖|L| |D| |Lᴴ|‖∞`, where `ε` is the machine epsilon and `c`
/// is a small constant (Higham, 2002, Chapter 11).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{LdlDecomposition, Matrix};
/// // A has eigenvalues 2 and −2.
/// let a = Matrix::from_vec(2, 2, vec![0.0_f64, 2.0, 2.0, 0.0])?;
/// let factor = LdlDecomposition::factor(
///     &a,
///     scioxide_linalg::HermitianInput::Exact)?;
/// assert_eq!(factor.inertia(scioxide_linalg::RankPolicy::Exact)?.positive(), 1);
/// assert_eq!(factor.inertia(scioxide_linalg::RankPolicy::Exact)?.negative(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Chapter 11.
/// - C. Ashcraft, R. G. Grimes and J. G. Lewis, Accurate symmetric indefinite
///   linear equation solvers, *SIAM J. Matrix Anal. Appl.* 20 (1998), 513–561.
#[derive(Debug)]
pub struct LdlDecomposition<S: NumericScalar> {
    lower: Matrix<S>,
    maximum: S::Real,
    projection: Option<HermitianProjectionReport<S::Real>>,
    source: Matrix<S>,
    one_norm: Scaled<S::Real>,
    exact_inertia: LdlInertia,
    value_scale: Scaled<S::Real>,
    diagonal: Vec<S>,
    off_diagonal: Vec<S>,
    /// `2^σ D`, the block diagonal factor of the scaled matrix `2^σ A`.
    scaled_diagonal: Vec<S>,
    scaled_off_diagonal: Vec<S>,
    /// The exponent `σ ≥ 0` of the scaling applied before elimination.
    exponent: i32,
    blocks: Vec<LdlPivotBlock>,
    permutation: Vec<usize>,
}

impl<S: NumericScalar> Clone for LdlDecomposition<S> {
    fn clone(&self) -> Self {
        Self {
            lower: self.lower.clone(),
            maximum: self.maximum,
            projection: self.projection,
            source: self.source.clone(),
            one_norm: self.one_norm,
            exact_inertia: self.exact_inertia,
            value_scale: self.value_scale,
            diagonal: self.diagonal.clone(),
            off_diagonal: self.off_diagonal.clone(),
            scaled_diagonal: self.scaled_diagonal.clone(),
            scaled_off_diagonal: self.scaled_off_diagonal.clone(),
            exponent: self.exponent,
            // A clone retains preparation for this order even if all pivots
            // were paired: the next epoch can contain `order` single blocks.
            blocks: {
                let mut blocks = Vec::with_capacity(self.order());
                blocks.extend_from_slice(&self.blocks);
                blocks
            },
            permutation: self.permutation.clone(),
        }
    }
}
impl<S: NumericScalar> PartialEq for LdlDecomposition<S> {
    fn eq(&self, other: &Self) -> bool {
        self.projection == other.projection
            && self.lower == other.lower
            && self.source == other.source
            && self.diagonal == other.diagonal
            && self.off_diagonal == other.off_diagonal
            && self.scaled_diagonal == other.scaled_diagonal
            && self.scaled_off_diagonal == other.scaled_off_diagonal
            && self.exponent == other.exponent
            && self.blocks == other.blocks
            && self.permutation == other.permutation
    }
}

impl<S: NumericScalar> Default for LdlDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> LdlDecomposition<S> {
    /// Returns an empty factorization of order 0, to pass to [`Self::factor_into`].
    pub const fn empty() -> Self {
        Self {
            lower: Matrix::empty(),
            maximum: S::Real::ZERO,
            projection: None,
            source: Matrix::empty(),
            one_norm: Scaled::ZERO,
            exact_inertia: LdlInertia::ZERO,
            value_scale: Scaled::ZERO,
            diagonal: Vec::new(),
            off_diagonal: Vec::new(),
            scaled_diagonal: Vec::new(),
            scaled_off_diagonal: Vec::new(),
            exponent: 0,
            blocks: Vec::new(),
            permutation: Vec::new(),
        }
    }
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
    /// assert_eq!(scioxide_linalg::LdlDecomposition::<f64>::factor_storage_bytes(0)?, 0);
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
            (order, core::mem::size_of::<S>()),
            (order, core::mem::size_of::<S>()),
            (order, core::mem::size_of::<S>()),
            (order, core::mem::size_of::<S>()),
            (order, core::mem::size_of::<LdlPivotBlock>()),
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
            (self.lower.capacity(), core::mem::size_of::<S>()),
            (self.source.capacity(), core::mem::size_of::<S>()),
            (self.diagonal.capacity(), core::mem::size_of::<S>()),
            (self.off_diagonal.capacity(), core::mem::size_of::<S>()),
            (self.scaled_diagonal.capacity(), core::mem::size_of::<S>()),
            (
                self.scaled_off_diagonal.capacity(),
                core::mem::size_of::<S>(),
            ),
            (
                self.blocks.capacity(),
                core::mem::size_of::<LdlPivotBlock>(),
            ),
            (self.permutation.capacity(), core::mem::size_of::<usize>()),
        ])
    }
    /// Grows the factor storage to hold an `order × order` factorization.
    ///
    /// The current factorization is unchanged. See [`Self::check`] for an example.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if growth fails. The current factorization stays
    ///   valid.
    pub fn try_reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        self.lower.try_reserve_shape(order, order)?;
        self.source.try_reserve_shape(order, order)?;
        super::util::reserve_vec(
            &mut self.diagonal,
            order,
            scioxide_core::quantity!("diagonal"),
        )?;
        super::util::reserve_vec(
            &mut self.off_diagonal,
            order,
            scioxide_core::quantity!("off_diagonal"),
        )?;
        super::util::reserve_vec(
            &mut self.scaled_diagonal,
            order,
            scioxide_core::quantity!("diagonal"),
        )?;
        super::util::reserve_vec(
            &mut self.scaled_off_diagonal,
            order,
            scioxide_core::quantity!("off_diagonal"),
        )?;
        super::util::reserve_vec(&mut self.blocks, order, scioxide_core::quantity!("blocks"))?;
        super::util::reserve_vec(
            &mut self.permutation,
            order,
            scioxide_core::quantity!("permutation"),
        )?;
        Ok(())
    }
    /// Reports whether the factor storage can hold an `order × order`
    /// factorization without growing.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::LdlDecomposition;
    /// use scioxide_core::Capacity;
    /// let mut factor = LdlDecomposition::<f64>::empty();
    /// factor.try_reserve(4)?;
    /// assert_eq!(factor.check(4)?, Capacity::Sufficient);
    /// assert_eq!(factor.order(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(&self, order: usize) -> Result<scioxide_core::Capacity, SciOxideError> {
        let mut plan = crate::workspace::accounting::StoragePlan::new(0);
        plan.matrix(&self.lower, order, order)?;
        plan.matrix(&self.source, order, order)?;
        plan.buffer(&self.diagonal, order)?;
        plan.buffer(&self.off_diagonal, order)?;
        plan.buffer(&self.scaled_diagonal, order)?;
        plan.buffer(&self.scaled_off_diagonal, order)?;
        plan.buffer(&self.blocks, order)?;
        plan.buffer(&self.permutation, order)?;
        plan.check(None)
    }

    fn resize(&mut self, n: usize) -> Result<(), SciOxideError> {
        self.lower.try_resize_flat(n, n)?;
        self.source.try_resize_flat(n, n)?;
        resize_vec(
            &mut self.diagonal,
            n,
            S::ZERO,
            scioxide_core::quantity!("diagonal"),
        )?;
        resize_vec(
            &mut self.off_diagonal,
            n,
            S::ZERO,
            scioxide_core::quantity!("off_diagonal"),
        )?;
        resize_vec(
            &mut self.scaled_diagonal,
            n,
            S::ZERO,
            scioxide_core::quantity!("diagonal"),
        )?;
        resize_vec(
            &mut self.scaled_off_diagonal,
            n,
            S::ZERO,
            scioxide_core::quantity!("off_diagonal"),
        )?;
        self.blocks.clear();
        resize_vec(
            &mut self.permutation,
            n,
            0,
            scioxide_core::quantity!("permutation"),
        )
    }
    /// Returns the change made by [`HermitianInput::Projected`] in the last
    /// factorization, or `None` for an exact or triangle input.
    pub const fn projection_report(&self) -> Option<&HermitianProjectionReport<S::Real>> {
        self.projection.as_ref()
    }
    /// Returns the order `n` of the factored `n × n` matrix.
    pub fn order(&self) -> usize {
        self.diagonal.len()
    }
    /// Returns the permutation `p` with `A[p[i], p[j]] = (L D Lᴴ)[i, j]`.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    /// Returns the unit lower triangular factor `L`.
    pub fn l_matrix(&self) -> &Matrix<S> {
        &self.lower
    }
    /// Returns the diagonal entries `D_kk`, which are real.
    pub fn diagonal(&self) -> &[S] {
        &self.diagonal
    }
    /// Returns the subdiagonal entries of `D`: entry `k` is `D_(k+1),k` when a 2×2
    /// block starts at `k`, and zero otherwise.
    pub fn off_diagonal(&self) -> &[S] {
        &self.off_diagonal
    }
    /// Returns the 1×1 and 2×2 blocks of `D` in order.
    pub fn pivot_blocks(&self) -> impl Iterator<Item = LdlPivotBlock> + '_ {
        self.blocks.iter().copied()
    }
    /// Computes the factorization `Pᵀ A P = L D Lᴴ` of the Hermitian matrix that
    /// `interpretation` defines from `a`.
    ///
    /// See [`HermitianInput`] for how `a` is read. The factorization runs on
    /// the calling thread; use [`Self::factor_into`] to reuse storage or to run
    /// in parallel.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square.
    /// - The input errors listed on [`HermitianInput`].
    /// - `NumericalInstability` (`Overflow`) if element growth makes an entry
    ///   of `L` or `D` overflow.
    pub fn factor<'m>(
        a: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, interpretation, &crate::LinalgExecutionOptions::default())
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
        a: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            a,
            interpretation,
            &options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
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
    /// `options`. On success the projection report of `output` is replaced,
    /// and cleared for an exact or triangle input.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::factor`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &crate::LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let a = a.into();
        let n = a.rows();
        if a.cols() != n {
            return Err(SciOxideError::dimension_mismatch([n, n], [n, a.cols()]));
        }
        let request = LdlWorkspaceRequest::factor(n, options).for_kernels(kernels);
        workspace.try_reserve(&Self::workspace_requirements(&request)?)?;
        output.try_reserve(n)?;
        workspace.with_direct(|storage, _, indices, gemm| {
            let (lower, storage) = crate::workspace::split_mut(storage, n * n)?;
            let (source, storage) = crate::workspace::split_mut(storage, n * n)?;
            let (schur, storage) = crate::workspace::split_mut(storage, n * n)?;
            let (weighted, storage) = crate::workspace::split_mut(storage, n * n.min(MAX_PANEL))?;
            let (columns, storage) = crate::workspace::split_mut(storage, n * 2)?;
            let (diagonal, off_diagonal) = crate::workspace::split_mut(storage, n)?;
            let (permutation, blocks) = crate::workspace::split_mut(indices, n)?;
            let mut candidate = LdlCandidate {
                lower: crate::MatrixViewMut::from_slice(n, n, lower)?,
                source: crate::MatrixViewMut::from_slice(n, n, source)?,
                diagonal,
                off_diagonal: &mut off_diagonal[..n],
                permutation,
                blocks: &mut blocks[..n],
                maximum: S::Real::ZERO,
            };
            candidate.lower.data.fill(S::ZERO);
            candidate.off_diagonal.fill(S::ZERO);
            candidate.blocks.fill(0);
            let projection = super::hermitian::prepare(
                a,
                interpretation,
                &mut candidate.source,
                scioxide_core::argument!("a"),
            )?;
            // A matrix with every entry below 1 in modulus is factored as
            // `2^σ A`, whose largest component lies in [1, 2): tiny and
            // subnormal entries then keep full precision. The scaling is exact.
            let exponent = crate::refinement::equilibration_exponent(standard::component_maximum(
                candidate.source.as_view(),
                Some(Triangle::Lower),
            ))?
            .max(0);
            for i in 0..n {
                candidate.permutation[i] = i;
                candidate.lower.data[i * n + i] = S::ONE;
                for j in 0..=i {
                    schur[i * n + j] =
                        standard::scale(candidate.source.get_element(i, j), exponent);
                }
            }
            let one_norm = crate::diagnostics::matrix_one_norm(candidate.source.as_view())?;
            let mut schur = crate::MatrixViewMut::from_slice(n, n, schur)?;
            let mut weighted = crate::MatrixViewMut::from_slice(n, n.min(MAX_PANEL), weighted)?;
            gemm.run_scope(
                context,
                ldl_identity(n),
                options.solve_budgets(),
                |gemm, kernels, solve| {
                    factor_kernel(
                        &mut candidate,
                        &mut schur,
                        columns,
                        &mut weighted,
                        gemm,
                        FactorExecution {
                            cx: kernels,
                            policy: options.execution_policy(),
                            context: solve,
                        },
                    )?;
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    let mut exact_inertia = LdlInertia::ZERO;
                    let mut value_scale = Scaled::ZERO;
                    let mut retain = |value: Scaled<S::Real>| -> Result<(), SciOxideError> {
                        let value = value.shifted(-i64::from(exponent))?;
                        let sign = if value.is_zero() {
                            Ordering::Equal
                        } else if value.is_sign_negative() {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        };
                        exact_inertia = exact_inertia.checked_add(LdlInertia::from_sign(sign))?;
                        if !value.abs_le(value_scale) {
                            value_scale = value.abs();
                        }
                        Ok(())
                    };
                    for (k, &width) in candidate.blocks.iter().enumerate() {
                        if width == 1 {
                            retain(Scaled::from_value(candidate.diagonal[k].real_component())?)?;
                        } else if width == 2 {
                            for value in scioxide_sparse::backend::hermitian::block_values(
                                candidate.diagonal[k],
                                candidate.off_diagonal[k],
                                candidate.diagonal[k + 1],
                            )? {
                                retain(value)?;
                            }
                        }
                    }
                    output.resize(n)?;
                    output.exact_inertia = exact_inertia;
                    output.value_scale = value_scale;
                    output.one_norm = one_norm;
                    output
                        .lower
                        .as_mut_slice()
                        .copy_from_slice(candidate.lower.data);
                    output
                        .source
                        .as_mut_slice()
                        .copy_from_slice(candidate.source.data);
                    output.scaled_diagonal.copy_from_slice(candidate.diagonal);
                    output
                        .scaled_off_diagonal
                        .copy_from_slice(candidate.off_diagonal);
                    // Undoing the scaling only shrinks values: it cannot overflow.
                    for (target, &value) in output.diagonal.iter_mut().zip(&*candidate.diagonal) {
                        *target = standard::scale(value, -exponent);
                    }
                    for (target, &value) in
                        output.off_diagonal.iter_mut().zip(&*candidate.off_diagonal)
                    {
                        *target = standard::scale(value, -exponent);
                    }
                    output.exponent = exponent;
                    output
                        .blocks
                        .extend(candidate.blocks.iter().enumerate().filter_map(
                            |(start, &width)| match width {
                                1 => Some(LdlPivotBlock::One { start }),
                                2 => Some(LdlPivotBlock::Two { start }),
                                _ => None,
                            },
                        ));
                    output.permutation.copy_from_slice(candidate.permutation);
                    output.maximum = candidate.maximum;
                    output.projection = projection;
                    Ok(())
                },
            )
        })
    }
}

/// Scratch size request for an `L D Lᴴ` factorization, solve or inverse.
///
/// Pass it to [`LdlDecomposition::workspace_requirements`] or
/// [`LdlDecomposition::workspace`]. The factor storage itself is separate; see
/// [`LdlDecomposition::try_reserve`].
#[derive(Debug, Clone, Copy)]
pub struct LdlWorkspaceRequest {
    job: LdlJob,
    kernel: crate::DenseWorkspaceRequest,
}

#[derive(Debug, Clone, Copy)]
enum LdlJob {
    Factor(usize),
    Vector(usize),
    Matrix { order: usize, rhs: usize },
}

impl LdlWorkspaceRequest {
    /// Requests scratch for [`LdlDecomposition::factor_into`] of an `order × order` matrix.
    pub const fn factor(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: LdlJob::Factor(order),
            kernel: crate::DenseWorkspaceRequest::product(order, order, order, options),
        }
    }

    /// Requests scratch for [`LdlDecomposition::solve_into`] with one right-hand side.
    pub const fn solve(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: LdlJob::Vector(order),
            kernel: crate::DenseWorkspaceRequest::vector(0, options),
        }
    }

    /// Requests scratch for [`LdlDecomposition::solve_matrix_into`] with
    /// `right_hand_sides` columns.
    ///
    /// Use `right_hand_sides == order` for [`LdlDecomposition::inverse_into`].
    pub const fn solve_matrix(
        order: usize,
        right_hand_sides: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: LdlJob::Matrix {
                order,
                rhs: right_hand_sides,
            },
            kernel: crate::DenseWorkspaceRequest::vector(0, options),
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

impl<S: NumericScalar> LdlDecomposition<S> {
    /// Returns the workspace size that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        request: &LdlWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let mut requirements = crate::kernel::product_requirements::<S>(&request.kernel)?;
        let product = |a, b| {
            scioxide_core::backend::checked_product(
                &[a, b],
                scioxide_core::quantity!("ldl_scratch"),
            )
        };
        *requirements.scalar_mut(ScalarRegion::Direct) = match request.job {
            LdlJob::Factor(n) => {
                requirements.indices = product(n, 2)?;
                [
                    product(n, n)?,
                    product(n, n)?,
                    product(n, n)?,
                    product(n, n.min(MAX_PANEL))?,
                    product(n, 4)?,
                ]
                .into_iter()
                .try_fold(0usize, |a, b| {
                    a.checked_add(b).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("ldl_scratch"))
                    })
                })?
            }
            LdlJob::Vector(n) => {
                requirements.scaled = product(n, 2)?;
                product(n, 2)?
            }
            LdlJob::Matrix { order, rhs } => {
                requirements.scaled = product(order, 2)?;
                if order == 0 {
                    0
                } else {
                    product(
                        order,
                        rhs.checked_mul(2)
                            .map(|count| count.max(2))
                            .ok_or_else(|| {
                                SciOxideError::size_overflow(scioxide_core::quantity!(
                                    "ldl_scratch"
                                ))
                            })?,
                    )?
                }
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
    /// use scioxide_linalg::{LdlDecomposition, LdlWorkspaceRequest, LinalgExecutionOptions};
    /// let request = LdlWorkspaceRequest::factor(4, &LinalgExecutionOptions::new());
    /// let requirements = LdlDecomposition::<f64>::workspace_requirements(&request)?;
    /// let workspace = LdlDecomposition::<f64>::workspace(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &LdlWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

struct LdlCandidate<'a, S: NumericScalar> {
    lower: crate::MatrixViewMut<'a, S>,
    source: crate::MatrixViewMut<'a, S>,
    diagonal: &'a mut [S],
    off_diagonal: &'a mut [S],
    blocks: &'a mut [usize],
    permutation: &'a mut [usize],
    maximum: S::Real,
}

fn swap<S: NumericScalar>(
    f: &mut LdlCandidate<'_, S>,
    a: &mut crate::MatrixViewMut<'_, S>,
    p: usize,
    q: usize,
    k: usize,
) {
    if p == q {
        return;
    }
    // Symmetric interchange `P^T S P` in lower-triangle Schur storage,
    // plus the rows of the already computed L columns `0..k`; later L
    // columns do not exist yet and need no swap.
    let n = f.diagonal.len();
    let data = &mut *a.data;
    let (first, last) = (p.min(q), p.max(q));
    for i in 0..first {
        data.swap(first * n + i, last * n + i);
    }
    data.swap(first * n + first, last * n + last);
    for i in first + 1..last {
        let lower = data[i * n + first];
        data[i * n + first] = data[last * n + i].conj();
        data[last * n + i] = lower.conj();
    }
    data[last * n + first] = data[last * n + first].conj();
    for i in last + 1..n {
        data.swap(i * n + first, i * n + last);
    }
    for j in 0..k {
        f.lower.data.swap(p * n + j, q * n + j);
    }
    f.permutation.swap(p, q);
}
fn magnitude<S: NumericScalar>(x: S) -> Result<Scaled<S::Real>, SciOxideError> {
    if S::IS_REAL {
        Scaled::from_value(x.real_component().abs())
    } else {
        wide(x)?.magnitude()
    }
}
fn max_off<S: NumericScalar>(
    values: &[S],
    k: usize,
    col: usize,
) -> Result<(usize, Scaled<S::Real>), SciOxideError> {
    if S::IS_REAL {
        // updated_column supplies finite values. Compare real magnitudes
        // directly, converting only the winner to the extended exponent form.
        let (index, maximum) = values
            .iter()
            .enumerate()
            .skip(k)
            .filter(|&(row, _)| row != col)
            .fold((col, S::Real::ZERO), |(index, maximum), (row, &entry)| {
                let value = entry.real_component().abs();
                if value > maximum {
                    (row, value)
                } else {
                    (index, maximum)
                }
            });
        return Ok((index, Scaled::from_value(maximum)?));
    }
    values
        .iter()
        .enumerate()
        .skip(k)
        .filter(|&(row, _)| row != col)
        .try_fold((col, Scaled::ZERO), |(index, maximum), (row, &entry)| {
            let value = magnitude(entry)?;
            Ok(if value.abs_cmp(maximum) == Ordering::Greater {
                (row, value)
            } else {
                (index, maximum)
            })
        })
}
#[derive(Clone, Copy)]
enum PivotChoice {
    One(usize),
    Two(usize, usize),
}

fn choose_pivot<S: NumericScalar>(
    k: usize,
    columns: &mut [S],
    n: usize,
    mut column: impl FnMut(usize, &mut [S]) -> Result<(), SciOxideError>,
) -> Result<PivotChoice, SciOxideError> {
    // Bounded Bunch–Kaufman, with the existing extended-range magnitude and
    // tie conventions. Each rook-search column includes all earlier pivots
    // in this panel, even though the full Schur update is deferred.
    let alpha = Scaled::from_value(
        (S::Real::ONE + (S::Real::TEN + S::Real::TEN - S::Real::TWO - S::Real::ONE).sqrt())
            / (S::Real::TWO * S::Real::TWO * S::Real::TWO),
    )?;
    let (first, next) = crate::workspace::split_mut(&mut columns[..2 * n], n)?;
    column(k, first)?;
    let (mut imax, mut colmax) = max_off(first, k, k)?;
    if colmax.is_zero() || magnitude(first[k])?.abs_cmp(colmax.mul(alpha)?) != Ordering::Less {
        return Ok(PivotChoice::One(k));
    }
    let mut p = k;
    loop {
        column(imax, next)?;
        let (jmax, rowmax) = max_off(next, k, imax)?;
        if magnitude(next[imax])?.abs_cmp(rowmax.mul(alpha)?) != Ordering::Less {
            return Ok(PivotChoice::One(imax));
        }
        if jmax == p || rowmax.abs_cmp(colmax) != Ordering::Greater {
            return Ok(PivotChoice::Two(p, imax));
        }
        p = imax;
        imax = jmax;
        colmax = rowmax;
    }
}

/// Materialize one rook-search column of A - L Wᴴ. The bounds belong to
/// completed panel columns, and remain valid under symmetric interchanges.
/// Rows advance in groups of [`COLUMN_LANES`] independent dot products; each
/// row keeps its own sequential reduction order.
fn updated_column<S: NumericScalar>(
    a: &crate::MatrixViewMut<'_, S>,
    lower: FiniteMatrixView<'_, S>,
    weighted: FiniteMatrixView<'_, S>,
    k: usize,
    col: usize,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let depth = lower.cols();
    let l = lower.as_view();
    let w = weighted.as_view();
    let ae = standard::exponent(lower.maximum)?;
    let be = standard::exponent(weighted.maximum)?;
    let growth = usize::BITS as i32 - depth.max(1).leading_zeros() as i32 + 4;
    let prior = |row: usize| {
        if row >= col {
            a.get_element(row, col)
        } else {
            a.get_element(col, row).conj()
        }
    };
    // Panels start empty: zero-width views need not span any rows.
    let left = |row: usize| match depth {
        0 => &[][..],
        _ => &l.data[row * l.row_stride..row * l.row_stride + depth],
    };
    let right = match depth {
        0 => &[][..],
        _ => &w.data[col * w.row_stride..col * w.row_stride + depth],
    };
    let finish = |row: usize, prior: S, value: S| -> Result<S, SciOxideError> {
        let value = if value.has_finite_components() {
            value
        } else {
            let restore =
                (ae + be + growth).max(standard::exponent(standard::magnitude(prior))? + 2);
            let scaled = (0..depth).fold(standard::scale(prior, -restore), |value, inner| {
                value
                    - standard::scale(l.get_element(row, inner), -ae)
                        * standard::scale(
                            standard::scale(w.get_element(col, inner).conj(), -be),
                            ae + be - restore,
                        )
            });
            standard::restore(scaled, restore)?
        };
        Ok(if row == col {
            S::from_real(value.real_component())
        } else {
            value
        })
    };
    let grouped = k + (output.len() - k) / COLUMN_LANES * COLUMN_LANES;
    for first in (k..grouped).step_by(COLUMN_LANES) {
        let priors: [S; COLUMN_LANES] = core::array::from_fn(|lane| prior(first + lane));
        let lefts: [&[S]; COLUMN_LANES] = core::array::from_fn(|lane| left(first + lane));
        let mut values = priors;
        for (inner, &right) in right.iter().enumerate() {
            let right = right.conj();
            for (value, left) in values.iter_mut().zip(&lefts) {
                *value = *value - left[inner] * right;
            }
        }
        for lane in 0..COLUMN_LANES {
            output[first + lane] = finish(first + lane, priors[lane], values[lane])?;
        }
    }
    for (row, target) in output.iter_mut().enumerate().skip(grouped) {
        let prior = prior(row);
        let value = left(row)
            .iter()
            .zip(right)
            .fold(prior, |value, (&left, &right)| value - left * right.conj());
        *target = finish(row, prior, value)?;
    }
    Ok(())
}
fn block_solve<S: NumericScalar>(a: S, b: S, c: S, x: S, y: S) -> Result<(S, S), SciOxideError> {
    // A well-separated real determinant admits ordinary Cramer's rule. The
    // eight-epsilon guard bounds cancellation in its three products and sum;
    // extreme frames and tiny/zero candidates retain the strict wide path.
    if S::IS_REAL {
        let maximum = a.abs().max(b.abs()).max(c.abs());
        let rhs_maximum = x.abs().max(y.abs());
        if standard::product_scales(maximum, maximum, 2)? == (0, 0)
            && standard::product_scales(maximum, rhs_maximum, 2)? == (0, 0)
        {
            let ac = a.real_component() * c.real_component();
            let bb = b.real_component() * b.real_component();
            let determinant = ac - bb;
            if determinant.abs()
                > (ac.abs() + bb) * S::Real::EPSILON * (S::Real::TWO * S::Real::TWO * S::Real::TWO)
            {
                let determinant = S::from_real(determinant);
                let u = (c * x - b * y) / determinant;
                let v = (a * y - b * x) / determinant;
                if u.has_finite_components()
                    && v.has_finite_components()
                    && u.abs() >= S::Real::MIN_POSITIVE
                    && v.abs() >= S::Real::MIN_POSITIVE
                {
                    return Ok((u, v));
                }
            }
        }
    }
    let [u, v] = scioxide_sparse::backend::hermitian::solve_block(a, b, c, x, y)?;
    Ok((narrow(u)?, narrow(v)?))
}

fn factor_kernel<S: NumericScalar>(
    f: &mut LdlCandidate<'_, S>,
    a: &mut crate::MatrixViewMut<'_, S>,
    columns: &mut [S],
    weighted: &mut crate::MatrixViewMut<'_, S>,
    gemm: &mut LinalgWorkspace<'_, S>,
    execution: FactorExecution<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let FactorExecution {
        cx,
        policy,
        context,
    } = execution;

    let n = f.diagonal.len();
    let execution = cx.execution();
    f.maximum = if n == 0 { S::Real::ZERO } else { S::Real::ONE };
    let panel_width = MAX_PANEL;
    let stride = weighted.cols();
    // The Schur complement lives in the lower triangle of `a`.
    let mut bound = TrailingBound::new(standard::component_maximum(
        a.as_view(),
        Some(Triangle::Lower),
    ));
    let mut start = 0;
    while start < n {
        let limit = (start + panel_width).min(n);
        let mut k = start;
        let mut lmax = S::Real::ONE;
        let mut wmax = S::Real::ZERO;
        while k < limit {
            // Leave room for a complete 2x2 pivot at a non-final boundary.
            if k + 1 == limit && limit < n {
                break;
            }
            crate::solve_support::admit_iteration(
                Some(execution),
                Some(&mut *context),
                "LDL pivot/RHS iteration budget exhausted",
            )?;
            let lower = FiniteMatrixView::from_validated(
                f.lower.as_view().block(0, start, n, k - start)?,
                lmax,
            );
            let w = FiniteMatrixView::from_validated(
                weighted.as_view().block(0, 0, n, k - start)?,
                wmax,
            );
            let choice = choose_pivot(k, columns, n, |col, output| {
                updated_column(a, lower, w, k, col, output)
            })?;
            let (p, q, width) = match choice {
                PivotChoice::One(p) => (p, p, 1),
                PivotChoice::Two(p, q) => (p, q, 2),
            };
            swap(f, a, k, p, k);
            for col in 0..k - start {
                weighted.data.swap(k * stride + col, p * stride + col);
            }
            if width == 2 {
                let q = if q == k { p } else { q };
                swap(f, a, k + 1, q, k);
                for col in 0..k - start {
                    weighted.data.swap((k + 1) * stride + col, q * stride + col);
                }
            }
            let lower = FiniteMatrixView::from_validated(
                f.lower.as_view().block(0, start, n, k - start)?,
                lmax,
            );
            let w = FiniteMatrixView::from_validated(
                weighted.as_view().block(0, 0, n, k - start)?,
                wmax,
            );
            if width != 1 || p != k {
                for offset in 0..width {
                    updated_column(
                        a,
                        lower,
                        w,
                        k,
                        k + offset,
                        &mut columns[offset * n..(offset + 1) * n],
                    )?;
                }
            }
            f.diagonal[k] = columns[k];
            f.blocks[k] = width;
            if width == 2 {
                f.diagonal[k + 1] = columns[n + k + 1];
                f.off_diagonal[k] = columns[k + 1];
            }
            for row in k..n {
                for offset in 0..width {
                    let value = columns[offset * n + row];
                    weighted.data[row * stride + k - start + offset] = value;
                    wmax = wmax.max(standard::magnitude(value));
                }
            }
            // Retain execution checking even for matrices contained in one
            // panel. Only disjoint L rows are handed to executor partitions.
            let diagonal = f.diagonal[k];
            let off = f.off_diagonal[k];
            let second = if width == 2 {
                f.diagonal[k + 1]
            } else {
                S::ZERO
            };
            execute_chunks(
                &mut f.lower.data[(k + width) * n..],
                n,
                policy,
                execution,
                |offset, rows| {
                    for (local, row) in rows.chunks_mut(n).enumerate() {
                        let i = k + width + offset + local;
                        if width == 1 {
                            row[k] = if diagonal == S::ZERO {
                                S::ZERO
                            } else {
                                // Hermitian 1x1 pivots are real: component division
                                // avoids a general complex extended-range quotient.
                                let input = columns[i];
                                let d = diagonal.real_component();
                                let value = S::from_components(
                                    input.real_component() / d,
                                    input.imaginary_component() / d,
                                );
                                if value.has_finite_components()
                                    && (input.real_component() == S::Real::ZERO
                                        || value.real_component() != S::Real::ZERO)
                                    && (input.imaginary_component() == S::Real::ZERO
                                        || value.imaginary_component() != S::Real::ZERO)
                                {
                                    value
                                } else {
                                    narrow(wide(input)?.div(wide(diagonal)?)?)?
                                }
                            };
                        } else {
                            let (x, y) = block_solve(
                                diagonal,
                                off,
                                second,
                                columns[i].conj(),
                                columns[n + i].conj(),
                            )?;
                            row[k] = x.conj();
                            row[k + 1] = y.conj();
                        }
                    }
                    Ok(())
                },
            )?;
            // Rook pivoting bounds multipliers mathematically, but recording
            // the actual values also handles rounding and generic scalars.
            for row in k + width..n {
                for col in k..k + width {
                    lmax = lmax.max(standard::magnitude(f.lower.get_element(row, col)));
                }
            }
            k += width;
        }
        if k < n {
            gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: FiniteMatrixView::from_validated(
                        f.lower.as_view().block(k, start, n - k, k - start)?,
                        lmax,
                    ),
                    right: FiniteMatrixView::from_validated(
                        weighted.as_view().block(k, 0, n - k, k - start)?,
                        wmax,
                    ),
                    output: a.block_mut(k, k, n - k, n - k)?,
                    config: GemmConfig {
                        right_op: GemmOp::Adjoint,
                        alpha: -S::ONE,
                        beta: S::ONE,
                        triangle: Some(Triangle::Lower),
                        prior: bound.prior(),
                        ..GemmConfig::ordinary()
                    },
                },
                policy,
                gemm,
                cx,
            )?;
            bound = bound.update(k - start, lmax, wmax);
            for row in k..n {
                a.data[row * n + row] = S::from_real(a.get_element(row, row).real_component());
            }
        }
        f.maximum = f.maximum.max(lmax);
        start = k;
    }
    Ok(())
}

impl<S: NumericScalar> LdlDecomposition<S> {
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
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
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
    /// Each column counts toward the iteration budget in `options`.
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
        output: crate::MatrixViewMut<'_, S>,
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
        let rank = self.analyze_rank(options.rank_policy())?;
        let request =
            LdlWorkspaceRequest::solve_matrix(n, rhs.cols(), &options.execution_options())
                .for_kernels(context.kernel());
        let requirements = Self::workspace_requirements(&request)?;
        workspace.try_reserve(&requirements)?;
        if self.order() == 0 && options.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            context,
            ldl_identity(n),
            options.execution_options().solve_budgets(),
            |workspace, kernels, scope| {
                let condition = options
                    .minimum_reciprocal_condition()
                    .map(|_| {
                        self.condition_estimate_scoped(
                            &options.execution_options(),
                            workspace,
                            kernels.scoped(&mut *scope),
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
                let staged = self.solve_columns(
                    |i, j| rhs.get_element(i, j),
                    rhs.cols(),
                    workspace.scalar_mut(ScalarRegion::Direct),
                    kernels,
                    scope,
                )?;
                let mut residual = None;
                if options.residual() == crate::Estimation::Estimate {
                    for (j, column) in staged.chunks_exact(2 * n).enumerate() {
                        crate::analysis::retain_largest_residual(
                            &mut residual,
                            crate::diagnostics::matrix_backward_error(
                                self.source.view(),
                                self.one_norm,
                                crate::VectorView::from_slice(&column[..n]),
                                rhs.col(j)?,
                                kernels,
                            )?,
                        );
                    }
                }
                kernels.execution().check()?;
                publish_columns(staged, n, output);
                Ok(crate::DirectSolveReport::new(
                    report.termination(),
                    rank,
                    condition,
                    residual,
                ))
            },
        )
    }
    /// Returns `A⁻¹`.
    ///
    /// Solves `A X = I` with the stored factors. Solving with [`Self::solve`] is
    /// cheaper and more accurate than multiplying by the inverse.
    ///
    /// # Errors
    ///
    /// - `Singular` if `D` has an exactly zero eigenvalue.
    /// - `NumericalInstability` (`Overflow`) if an entry of `A⁻¹` overflows.
    pub fn inverse(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.order(), self.order())?;
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
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let n = self.order();
        if output.rows() != n || output.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                [n, n],
                [output.rows(), output.cols()],
            ));
        }
        workspace.try_reserve(&Self::workspace_requirements(
            &LdlWorkspaceRequest::solve_matrix(n, n, options),
        )?)?;
        run_ldl(
            workspace,
            context,
            n,
            options.solve_budgets(),
            |scratch, kernels, solve| {
                let staged = self.solve_columns(
                    |row, col| if row == col { S::ONE } else { S::ZERO },
                    n,
                    scratch,
                    kernels,
                    solve,
                )?;
                publish_columns(staged, n, output);
                Ok(())
            },
        )
    }

    /// Stage `columns` right-hand sides from `rhs` in `2n`-strided columns of
    /// `scratch`, charging one iteration per column before any dispatch, and
    /// solve them under the kernel policy. The solved values lead each column.
    fn solve_columns<'s>(
        &self,
        rhs: impl Fn(usize, usize) -> S,
        columns: usize,
        scratch: &'s mut [S],
        kernels: KernelContext<'_>,
        solve: &mut SolveContext<'_, '_>,
    ) -> Result<&'s mut [S], SciOxideError> {
        let n = self.order();
        let m = columns;
        let execution = kernels.execution();
        execution.check()?;
        const STAGED: Quantity = scioxide_core::quantity!("scratch");
        let staged = &mut scratch[..size(size(n, 2, STAGED)?, m, STAGED)?];
        if n == 0 {
            return Ok(staged);
        }
        // Reserve all RHS attempts in column order before dispatch. Budget failure
        // therefore admits no parallel jobs and never publishes partial columns.
        for j in 0..m {
            crate::solve_support::admit_iteration(
                Some(execution),
                Some(&mut *solve),
                "LDL RHS budget exhausted",
            )?;
            for i in 0..n {
                staged[j * 2 * n + i] = rhs(i, j);
            }
        }
        execute_chunks(staged, 2 * n, kernels.policy(), execution, |_, columns| {
            for column in columns.chunks_mut(2 * n) {
                execution.check()?;
                let (values, work) = crate::workspace::split_mut(column, n)?;
                self.solve_slice(values, work, kernels)?;
            }
            Ok(())
        })?;
        execution.check()?;
        Ok(staged)
    }
    /// Solve `A x = b` in place in `values`, with `work` of the factor order.
    fn solve_slice(
        &self,
        values: &mut [S],
        work: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        // `2^σ A = P L (2^σ D) Lᴴ Pᵀ`: gather `Pᵀ (2^σ b)`, forward-substitute
        // with unit L, solve each 1x1/2x2 block of `2^σ D`, back-substitute
        // with `Lᴴ`, and scatter through P. Singular D blocks are rejected
        // before `values` is touched.
        let n = self.order();
        if self.exact_inertia.zero() != 0 {
            return Err(SciOxideError::singular(None));
        }
        let work = &mut work[..n];
        for (i, value) in work.iter_mut().enumerate() {
            *value = crate::checked::finite_candidate(standard::scale(
                values[self.permutation[i]],
                self.exponent,
            ))?;
        }
        self.substitute(work, GemmOp::None, context)?;
        for block in self.pivot_blocks() {
            let k = block.checked_range(self.order())?.start;
            if block.size() == 1 {
                work[k] = crate::kernel::standard::finish_substitution(
                    work[k],
                    self.scaled_diagonal[k],
                    work[k],
                    core::iter::empty(),
                    None,
                )?;
            } else {
                (work[k], work[k + 1]) = block_solve(
                    self.scaled_diagonal[k],
                    self.scaled_off_diagonal[k],
                    self.scaled_diagonal[k + 1],
                    work[k],
                    work[k + 1],
                )?;
            }
        }
        self.substitute(work, GemmOp::Adjoint, context)?;
        for (i, &x) in work.iter().enumerate() {
            values[self.permutation[i]] = x;
        }
        Ok(())
    }

    /// Substitute with the unit lower triangular `L` (`op = None`) or `Lᴴ`.
    ///
    /// Under [`Accuracy::Exact`](scioxide_core::Accuracy::Exact) each row's
    /// inner product is accumulated exactly and rounded once, so the result
    /// does not depend on the kernel; otherwise the Standard kernel of
    /// `context` substitutes.
    fn substitute(
        &self,
        work: &mut [S],
        op: GemmOp,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        if context.accuracy() != scioxide_core::Accuracy::Exact {
            return crate::kernel::triangular::solve(
                crate::kernel::TriangularOperand {
                    view: crate::kernel::triangular::Storage::Dense(self.lower.view()),
                    triangle: Triangle::Lower,
                    op,
                    diagonal: scioxide_core::DiagonalKind::Unit,
                },
                self.maximum,
                work,
                n.saturating_sub(1),
                context.dispatch(),
            );
        }
        let lower = self.lower.view();
        let row = |i: usize, known: core::ops::Range<usize>, work: &mut [S]| {
            if i.is_multiple_of(64) {
                context.execution().check()?;
            }
            let mut sum = scioxide_core::ProductSum::new();
            sum.add_product(work[i], S::ONE)?;
            for j in known {
                let coefficient = if op == GemmOp::None {
                    lower.get_element(i, j)
                } else {
                    lower.get_element(j, i).conj()
                };
                sum.subtract_product(coefficient, work[j])?;
            }
            work[i] = sum.finish()?;
            Ok::<(), SciOxideError>(())
        };
        if op == GemmOp::None {
            (0..n).try_for_each(|i| row(i, 0..i, work))
        } else {
            (0..n).rev().try_for_each(|i| row(i, i + 1..n, work))
        }
    }
    /// Returns this factorization as the operator `x ↦ A⁻¹ x`.
    ///
    /// `A⁻¹` is indefinite in general, so it is not a valid preconditioner for
    /// methods that need a positive definite one, such as CG and MINRES.
    pub fn inverse_operator(&self) -> LdlInverseOperator<'_, S> {
        LdlInverseOperator { factor: self }
    }
}

/// The operator `x ↦ A⁻¹ x` of an [`LdlDecomposition`], returned by
/// [`LdlDecomposition::inverse_operator`].
///
/// It implements [`LinearOperator`], [`AdjointOperator`] (the same map, since
/// `A⁻¹` is Hermitian) and [`Preconditioner`]. Each application needs `2n`
/// scratch scalars and returns `Singular` if `D` has an exactly zero
/// eigenvalue. Calls without a context run on the calling thread.
#[derive(Debug, Clone, Copy)]
pub struct LdlInverseOperator<'a, S: NumericScalar> {
    factor: &'a LdlDecomposition<S>,
}
impl<S: NumericScalar> LdlInverseOperator<'_, S> {
    fn apply(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        kernels.execution().check()?;
        let n = self.rows();
        crate::checked::expect_shape([n], [input.len()])?;
        crate::checked::expect_shape([n], [output.len()])?;
        let required = self.apply_workspace_len()?;
        if workspace.len() < required {
            return Err(SciOxideError::dimension_mismatch(
                [required],
                [workspace.len()],
            ));
        }
        for &x in input {
            crate::checked::finite(x, scioxide_core::NonFiniteOrigin::Input, "nonfinite input")
                .map(|_| ())?;
        }
        let (staged, work) = crate::workspace::split_mut(workspace, n)?;
        staged.copy_from_slice(input);
        self.factor.solve_slice(staged, &mut work[..n], kernels)?;
        kernels.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
}
impl<S: NumericScalar> LinearOperator<S> for LdlInverseOperator<'_, S> {
    fn rows(&self) -> usize {
        self.factor.order()
    }
    fn cols(&self) -> usize {
        self.factor.order()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        size(self.rows(), 2, scioxide_core::quantity!("apply_workspace"))
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
impl<S: NumericScalar> AdjointOperator<S> for LdlInverseOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        size(self.rows(), 2, scioxide_core::quantity!("apply_workspace"))
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
impl<S: NumericScalar> Preconditioner<S> for LdlInverseOperator<'_, S> {
    fn dimension(&self) -> usize {
        self.factor.order()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        size(self.rows(), 2, scioxide_core::quantity!("apply_workspace"))
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

impl<S: NumericScalar> LdlDecomposition<S> {
    // Visits the eigenvalues of D (each 2x2 block contributes its two real
    // eigenvalues). By Sylvester's law of inertia, D and `A = P L D L^H P^T`
    // share signature, so these values drive inertia and rank decisions.
    fn visit_values(
        &self,
        mut visit: impl FnMut(Scaled<S::Real>) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let shift = -i64::from(self.exponent);
        for block in self.pivot_blocks() {
            let k = block.checked_range(self.order())?.start;
            if block.size() == 1 {
                visit(
                    Scaled::from_value(self.scaled_diagonal[k].real_component())?.shifted(shift)?,
                )?;
            } else {
                for value in scioxide_sparse::backend::hermitian::block_values(
                    self.scaled_diagonal[k],
                    self.scaled_off_diagonal[k],
                    self.scaled_diagonal[k + 1],
                )? {
                    visit(value.shifted(shift)?)?;
                }
            }
        }
        Ok(())
    }

    /// Returns the sign of `det A = det D` and `ln |det A|`.
    ///
    /// The symmetric permutation does not change the sign. An empty matrix gives
    /// sign 1 and `ln |det A| = 0`.
    ///
    /// # Errors
    ///
    /// - `Singular` if a block of `D` has zero determinant.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{HermitianInput, LdlDecomposition, Matrix};
    /// // det A = −4
    /// let a = Matrix::from_vec(2, 2, vec![0.0_f64, 2.0, 2.0, 0.0])?;
    /// let value = LdlDecomposition::factor(&a, HermitianInput::Exact)?
    ///     .slogdet()?;
    /// assert_eq!(value.phase(), -1.0);
    /// assert!((value.log_abs() - 4.0_f64.ln()).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn slogdet(&self) -> Result<super::SignLogDeterminant<S>, SciOxideError> {
        let mut determinant = Scaled::ONE;
        for block in self.pivot_blocks() {
            let k = block.checked_range(self.order())?.start;
            let value = if block.size() == 1 {
                Scaled::from_value(self.scaled_diagonal[k].real_component())?
            } else {
                scioxide_sparse::backend::hermitian::determinant(
                    self.scaled_diagonal[k],
                    self.scaled_off_diagonal[k],
                    self.scaled_diagonal[k + 1],
                )?
            };
            if value.is_zero() {
                return Err(SciOxideError::singular(Some(k)));
            }
            determinant = determinant.mul(value)?;
        }
        // `det A = 2^(−nσ) det(2^σ D)`.
        let shift = i64::try_from(self.order())
            .ok()
            .and_then(|order| order.checked_mul(-i64::from(self.exponent)))
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::ExponentOutOfRange,
                    "determinant exponent exceeds i64",
                )
            })?;
        super::SignLogDeterminant::from_scaled(ScaledValue::from_components(
            determinant.shifted(shift)?,
            Scaled::ZERO,
        ))
    }

    /// Returns the inertia: the numbers of positive, negative and zero eigenvalues
    /// of the blocks of `D`.
    ///
    /// With [`RankPolicy::Exact`](crate::RankPolicy::Exact) this is the exact
    /// inertia of the factored matrix, by Sylvester's law of inertia. Other
    /// policies count an eigenvalue of `D` as zero when its modulus is at most
    /// the threshold, with scale the largest such modulus. This is a threshold on
    /// `D`, not on the eigenvalues of `A`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the scale or the threshold
    ///   exceeds the largest finite value.
    pub fn inertia(&self, policy: crate::RankPolicy<S::Real>) -> Result<LdlInertia, SciOxideError> {
        if matches!(policy, crate::RankPolicy::Exact) {
            return Ok(self.exact_inertia);
        }
        let cutoff = Scaled::from_value(policy.cutoff(
            self.value_scale.round()?,
            self.order(),
            self.order(),
        )?)?;
        let mut result = LdlInertia::default();
        self.visit_values(|value| {
            let sign = if value.abs_cmp(cutoff) != Ordering::Greater {
                Ordering::Equal
            } else if value.is_sign_negative() {
                Ordering::Less
            } else {
                Ordering::Greater
            };
            result = result.checked_add(LdlInertia::from_sign(sign))?;
            Ok(())
        })?;
        Ok(result)
    }
    /// Classifies the numerical rank from the eigenvalues of the blocks of `D`.
    ///
    /// The rank is the number of nonzero eigenvalues counted by [`Self::inertia`].
    /// Under [`RankPolicy::Exact`](crate::RankPolicy::Exact), a scale beyond the
    /// scalar range is reported as the largest finite value.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::inertia`].
    pub fn analyze_rank(
        &self,
        policy: crate::RankPolicy<S::Real>,
    ) -> Result<crate::RankReport<S::Real>, SciOxideError> {
        let inertia = self.inertia(policy)?;
        let extended_scale = self.value_scale;
        // Exact rank depends only on signs, including blocks whose eigenvalue
        // magnitude exceeds the scalar range. Saturate only this diagnostic;
        // thresholded policies continue to require their actual finite scale.
        let scale = if matches!(policy, crate::RankPolicy::Exact)
            && !extended_scale.abs_le(Scaled::from_value(S::Real::MAX)?)
        {
            S::Real::MAX
        } else {
            extended_scale.round()?
        };
        Ok(crate::RankReport::from_factor(
            inertia.positive() + inertia.negative(),
            self.order(),
            policy.cutoff(scale, self.order(), self.order())?,
            scale,
        ))
    }
    /// Estimates the reciprocal condition number `1 / (‖A‖₁ ‖A⁻¹‖₁)`.
    ///
    /// The method is that of
    /// [`LuDecomposition::condition_estimate`](crate::LuDecomposition::condition_estimate).
    /// An exactly zero eigenvalue of `D` gives an estimate of 0. A positive
    /// estimate that underflows is reported as a tiny positive number, so 0
    /// always means exactly singular.
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
                    algorithm: scioxide_core::algorithm_id!("linalg", "ldl_condition", 1),
                    order: self.order(),
                    norm: self.one_norm,
                    singular: self.inertia(crate::RankPolicy::Exact)?.zero() != 0,
                })
            },
            // The inverse is Hermitian, so adjoint actions coincide.
            |action, kernels| {
                action.solve(
                    |values, temporary| self.solve_slice(values, temporary, kernels),
                    |rhs, work| self.solve_scaled(rhs, work),
                )
            },
        )
    }
    /// Apply the Hermitian inverse to `rhs` in extended range, using `work` of the factor order.
    fn solve_scaled(
        &self,
        rhs: &mut [ScaledValue<S>],
        work: &mut [ScaledValue<S>],
    ) -> Result<(), SciOxideError> {
        let n = self.order();
        // `A⁻¹ = 2^σ (2^σ A)⁻¹`, and the factors are those of `2^σ A`.
        for (i, &original) in self.permutation.iter().enumerate() {
            work[i] = rhs[original].shifted(i64::from(self.exponent))?;
        }
        crate::diagnostics::matrix_substitute(
            self.lower.view(),
            scioxide_core::Triangle::Lower,
            scioxide_core::DiagonalKind::Unit,
            false,
            work,
        )?;
        for block in self.pivot_blocks() {
            let k = block.checked_range(n)?.start;
            let a = ScaledValue::from_value(self.scaled_diagonal[k])?;
            if block.size() == 1 {
                work[k] = work[k].div(a)?;
            } else {
                let b = ScaledValue::from_value(self.scaled_off_diagonal[k])?;
                let bh = ScaledValue::from_value(self.scaled_off_diagonal[k].conj())?;
                let c = ScaledValue::from_value(self.scaled_diagonal[k + 1])?;
                let det = ScaledValue::from_components(
                    scioxide_sparse::backend::hermitian::determinant(
                        self.scaled_diagonal[k],
                        self.scaled_off_diagonal[k],
                        self.scaled_diagonal[k + 1],
                    )?,
                    Scaled::ZERO,
                );
                let x = work[k];
                let y = work[k + 1];
                work[k] = c.mul(x)?.sub(bh.mul(y)?)?.div(det)?;
                work[k + 1] = a.mul(y)?.sub(b.mul(x)?)?.div(det)?;
            }
        }
        crate::diagnostics::matrix_substitute(
            self.lower.view(),
            scioxide_core::Triangle::Lower,
            scioxide_core::DiagonalKind::Unit,
            true,
            work,
        )?;
        for (i, &original) in self.permutation.iter().enumerate() {
            rhs[original] = work[i];
        }
        Ok(())
    }
    /// Solves `A x = b` and returns `x`.
    ///
    /// The solve costs about 2n² flops. It uses the default
    /// [`DirectSolveOptions`](crate::DirectSolveOptions): with
    /// [`RankPolicy::Exact`](crate::RankPolicy::Exact) only an exactly singular
    /// block diagonal factor `D` rejects the solve. Use [`Self::solve_with_options`]
    /// for a numerical rank cutoff, a condition threshold or the residual of the
    /// solution.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have length [`Self::order`].
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    /// - `Singular` if `D` is rank deficient under `RankPolicy::Exact`.
    /// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
    pub fn solve<'b>(&self, rhs: impl Into<VectorView<'b, S>>) -> Result<Vector<S>, SciOxideError> {
        self.solve_with_options(rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `A x = b`, checks the result against `options` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The solve classifies the rank with [`Self::analyze_rank`] and `options.rank_policy()`. If
    /// `options` sets a minimum reciprocal condition number, it then calls
    /// [`Self::condition_estimate`]. A rank-deficient or ill-conditioned system returns `Ok` with
    /// that [`DirectSolveTermination`](crate::DirectSolveTermination) and no solution; see
    /// [Direct solves](crate#direct-solves). A requested residual is the normwise backward error of
    /// the original system, `‖b − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)`.
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
        rhs: impl Into<VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
    ) -> Result<crate::DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
        );
        let options = &options;
        let mut output = Vector::zeros(self.order())?;
        let mut workspace = LinalgWorkspace::new();
        let report = self.solve_into(
            rhs,
            options,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        let solution = output;
        Ok(crate::DirectSolveResult::new(solution, report))
    }

    /// Solves `A x = b` into `output`, the reusable form of [`Self::solve`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` does not have length [`Self::order`].
    /// - The errors of [`Self::solve`] and the workspace and execution errors of
    ///   [`Self::factor_into`].
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
        mut output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(&options.execution_options())?;
        let kernels = context.kernel();
        let execution = kernels.execution();
        execution.check()?;
        let rhs = rhs.into();
        let n = self.order();
        if rhs.len() != n || output.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                [n, n],
                [rhs.len(), output.len()],
            ));
        }
        let rhs = rhs.try_finite()?;
        workspace.try_reserve(&Self::workspace_requirements(&LdlWorkspaceRequest::solve(
            n,
            &options.execution_options(),
        ))?)?;
        if n == 0 && options.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        let rank = self.analyze_rank(options.rank_policy())?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "ldl_solve", 1),
            n as u128,
            0,
        );
        let result = workspace.run_scope(
            context,
            identity,
            options.execution_options().solve_budgets(),
            |workspace, kernels, solve| {
                let condition = options
                    .minimum_reciprocal_condition()
                    .map(|_| {
                        self.condition_estimate_scoped(
                            &options.execution_options(),
                            workspace,
                            kernels.scoped(&mut *solve),
                        )
                    })
                    .transpose()?;
                let termination =
                    crate::analysis::acceptance(rank, condition, options).termination();
                if n == 0 {
                    return Ok(crate::DirectSolveReport::new(
                        termination,
                        rank,
                        condition,
                        crate::analysis::empty_residual(options, 1)?,
                    ));
                }
                let residual = if termination == crate::DirectSolveTermination::Accepted {
                    crate::solve_support::admit_iteration(
                        Some(execution),
                        Some(solve),
                        "no accepted LDL solution",
                    )?;
                    let (solution, work) =
                        crate::workspace::split_mut(workspace.scalar_mut(ScalarRegion::Direct), n)?;
                    let work = &mut work[..n];
                    for (x, value) in solution.iter_mut().zip(rhs.as_view().iter()) {
                        *x = value;
                    }
                    self.solve_slice(solution, work, kernels)?;
                    (options.residual() == crate::Estimation::Estimate)
                        .then(|| {
                            crate::diagnostics::matrix_backward_error(
                                self.source.view(),
                                self.one_norm,
                                crate::VectorView::from_slice(solution),
                                rhs.as_view(),
                                kernels,
                            )
                        })
                        .transpose()?
                } else {
                    None
                };
                execution.check()?;
                Ok(crate::DirectSolveReport::new(
                    termination,
                    rank,
                    condition,
                    residual,
                ))
            },
        );
        let report = result?;
        if report.termination() == crate::DirectSolveTermination::Accepted {
            output
                .iter_mut()
                .zip(&workspace.scalar_mut(ScalarRegion::Direct)[..n])
                .for_each(|(out, &value)| *out = value);
        }
        Ok(report)
    }
}

// Fixed-size stack slots bound dispatch metadata independently of matrix order.
// Each partition owns disjoint rows/columns; kernels retain scalar operation order.

/// Write the leading `n` values of each `2n`-strided staged column to `output`.
fn publish_columns<S: NumericScalar>(
    staged: &[S],
    n: usize,
    mut output: crate::MatrixViewMut<'_, S>,
) {
    if n == 0 {
        return;
    }
    for (j, column) in staged.chunks_exact(2 * n).enumerate() {
        for (i, &value) in column[..n].iter().enumerate() {
            output.set_element(i, j, value);
        }
    }
}

fn ldl_identity(order: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "ldl", 1),
        order as u128,
        0,
    )
}

fn run_ldl<S: NumericScalar, T>(
    workspace: &mut LinalgWorkspace<'_, S>,
    context: crate::LinalgExecutionContext<'_, '_>,
    order: usize,
    budgets: scioxide_core::SolveBudgets,
    work: impl FnOnce(
        &mut [S],
        KernelContext<'_>,
        &mut SolveContext<'_, '_>,
    ) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    workspace.run_scope(
        context,
        ldl_identity(order),
        budgets,
        |workspace, kernels, solve| {
            work(workspace.scalar_mut(ScalarRegion::Direct), kernels, solve)
        },
    )
}
