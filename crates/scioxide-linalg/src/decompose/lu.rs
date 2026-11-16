//! LU factorization with partial pivoting, `P A = L U`.
//!
//! Blocked right-looking elimination (Golub & Van Loan, §3.4): each panel of at
//! most 64 columns is factored unblocked in a contiguous column-major buffer, its
//! row interchanges are applied to the rest of the matrix, and the trailing
//! matrix is updated with TRSM and GEMM. O(n³) time, O(n²) memory.

use crate::workspace::ScalarRegion;
/// Widest LU panel, the cap of [`crate::kernel::panel_width`]. Panels this
/// narrow factor fastest by unblocked elimination in a contiguous buffer.
const MAX_PANEL: usize = 64;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;

use scioxide_core::{ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError};

use crate::KernelContext;
use crate::MatrixSide;
use crate::dense::{Matrix, Vector};
use crate::kernel::level3::trsm_bounded;
use crate::kernel::standard::scale as scale_numeric;
use crate::kernel::{GemmConfig, LinalgWorkspace, Triangle, gemm_workspace_with_context};
use crate::refinement::equilibration_exponent;
use scioxide_core::DiagonalKind;

use super::util::{copy_rows, largest_abs, modulus_above, reserve_vec, scale_columns, scale_slice};
use crate::analysis::{
    DirectSolveOptions, DirectSolveReport, DirectSolveResult, DirectSolveTermination,
    EquilibrationMetadata, RankPolicy, RankReport, analyze_magnitudes,
};
use crate::kernel::TrailingBound;

/// Scratch size request for an LU factorization, solve or inverse.
///
/// Pass it to [`LuDecomposition::workspace_requirements`] or
/// [`LuDecomposition::workspace`]. The factor storage itself is separate; see
/// [`LuDecomposition::try_reserve`].
#[derive(Debug, Clone, Copy)]
pub struct LuWorkspaceRequest {
    job: LuJob,
    kernel: crate::DenseWorkspaceRequest,
}

#[derive(Debug, Clone, Copy)]
enum LuJob {
    Factor(usize),
    Vector(usize),
    Matrix { order: usize, rhs: usize },
}

impl LuWorkspaceRequest {
    /// Requests scratch for [`LuDecomposition::factor_into`] of an `order × order` matrix.
    pub const fn factor(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: LuJob::Factor(order),
            kernel: crate::DenseWorkspaceRequest::product(order, order, order, options),
        }
    }

    /// Requests scratch for [`LuDecomposition::solve_into`] with one right-hand side.
    ///
    /// The request includes the condition and residual diagnostics. With
    /// [`Accuracy::Exact`](scioxide_core::Accuracy::Exact) in `options` it also
    /// covers the matrix product scratch that exact substitution uses.
    pub const fn solve(order: usize, options: &crate::LinalgExecutionOptions) -> Self {
        Self {
            job: LuJob::Vector(order),
            kernel: if matches!(options.accuracy(), scioxide_core::Accuracy::Exact) {
                crate::DenseWorkspaceRequest::product(order, order, 1, options)
            } else {
                crate::DenseWorkspaceRequest::vector(0, options)
            },
        }
    }

    /// Requests scratch for [`LuDecomposition::solve_matrix_into`] with
    /// `right_hand_sides` columns.
    ///
    /// Use `right_hand_sides == order` for [`LuDecomposition::inverse_into`].
    pub const fn solve_matrix(
        order: usize,
        right_hand_sides: usize,
        options: &crate::LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: LuJob::Matrix {
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

impl<S: NumericScalar> LuDecomposition<S> {
    /// Returns the workspace size that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        request: &LuWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let mut requirements = crate::kernel::product_requirements::<S>(&request.kernel)?;
        let product = |a, b| {
            scioxide_core::backend::checked_product(&[a, b], scioxide_core::quantity!("lu_scratch"))
        };
        *requirements.scalar_mut(ScalarRegion::Direct) = match request.job {
            LuJob::Factor(n) => {
                let lengths = factor_kernel_lengths(n)?;
                requirements.indices = lengths.indices;
                lengths.scalars
            }
            LuJob::Vector(n) => {
                requirements.scaled = product(n, 2)?;
                product(n, 2)?
            }
            LuJob::Matrix { order, rhs } => {
                requirements.scaled = product(order, 2)?;
                product(
                    order,
                    if order == 0 {
                        0
                    } else {
                        rhs.checked_add(1)
                            .map(|count| count.max(2))
                            .ok_or_else(|| {
                                SciOxideError::size_overflow(scioxide_core::quantity!("lu_scratch"))
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
    /// use scioxide_linalg::{LinalgExecutionOptions, LuDecomposition, LuWorkspaceRequest};
    /// let request = LuWorkspaceRequest::factor(4, &LinalgExecutionOptions::new());
    /// let requirements = LuDecomposition::<f64>::workspace_requirements(&request)?;
    /// let workspace = LuDecomposition::<f64>::workspace(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &LuWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Determinant in the form `det A = phase · exp(log_abs)`.
///
/// For real scalars the phase is `±1`; for complex scalars it has modulus 1.
/// Returned by [`LuDecomposition::slogdet`] and other factorizations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SignLogDeterminant<S: NumericScalar> {
    phase: S,
    log_abs: S::Real,
}

impl<S: NumericScalar> SignLogDeterminant<S> {
    pub(crate) fn from_scaled(
        scaled: scioxide_core::ScaledValue<S>,
    ) -> Result<Self, SciOxideError> {
        if scaled.is_zero() {
            return Err(SciOxideError::singular(None));
        }
        let magnitude = scaled.magnitude()?;
        Ok(Self {
            phase: S::from_components(
                scaled.real().div(magnitude)?.round()?,
                scaled.imaginary().div(magnitude)?.round()?,
            ),
            log_abs: magnitude.ln_abs()?,
        })
    }

    /// Returns the sign (real) or unit-modulus phase (complex) of the determinant.
    ///
    /// Each component is rounded separately, so a component smaller than the
    /// smallest subnormal number may round to a signed zero.
    pub const fn phase(self) -> S {
        self.phase
    }

    /// Returns `ln |det A|`.
    pub const fn log_abs(self) -> S::Real {
        self.log_abs
    }
}

/// LU factorization `P A = L U` of a square matrix with partial pivoting.
///
/// `P` is a row permutation, `L` is unit lower triangular and `U` is upper
/// triangular. Before elimination, each row and then each column is scaled by
/// a power of two so that its largest entry has modulus in `[1, 2)`. The
/// scaling is exact except for an entry below about `μ` times the largest
/// modulus in its row, where `μ` is the smallest positive normal number of
/// the scalar type: its row-scaled value is subnormal and is rounded, or
/// flushed to zero. The stored factors are those of the scaled matrix
/// `P D_r A D_c`; [`Self::to_l_matrix`] and [`Self::to_u_matrix`] return the
/// factors of `P A`.
///
/// In each column the pivot is the first entry of largest modulus. The
/// factorization fails only on an exact zero pivot, or when pivot growth
/// carries an entry of `U` beyond the floating-point range. Rank and
/// condition are checked when you solve; see [Direct solves](crate#direct-solves).
///
/// # Accuracy
///
/// The computed factors are the exact factors of `P (A + ΔA)` with
/// `‖ΔA‖∞ ≤ c n ε ‖|L| |U|‖∞` in scaled coordinates, where `ε` is the machine
/// epsilon and `c` is a small constant (Higham, 2002, Chapter 9). Pivot growth
/// enters the bound through `|L| |U|`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{LuDecomposition, Matrix, Vector};
///
/// // 2x + y = 5 and 4x + 3y = 11 have the solution (2, 1).
/// let a = Matrix::from_rows(&[[2.0_f64, 1.0], [4.0, 3.0]])?;
/// let lu = LuDecomposition::factor(&a)?;
/// let x = lu.solve(&Vector::from_vec(vec![5.0, 11.0]))?;
/// assert!((x.as_slice()[0] - 2.0).abs() < 1e-14 && (x.as_slice()[1] - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §3.2–3.4.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Chapter 9.
#[derive(Debug, Clone)]
#[must_use]
pub struct LuDecomposition<S: NumericScalar> {
    /// Combined L\U storage: L below diagonal (unit diagonal implied),
    /// U on and above diagonal.
    lu: Matrix<S>,
    source: Matrix<S>,
    maximum: S::Real,
    /// Row permutation: `perm[i]` is the original row index for the
    /// `i`-th row in the factored form.
    perm: Vec<usize>,
    /// Sign of the permutation: +1 or -1 (for determinant).
    sign: i32,
    /// Dimension of the square matrix.
    n: usize,
    /// Reversible power-of-two input equilibration.
    equilibration: EquilibrationMetadata,
    one_norm: scioxide_core::Scaled<S::Real>,
    infinity_norm: scioxide_core::Scaled<S::Real>,
}

/// Stores equilibrated partial-pivoted LU factors for repeated backend solves.
///
/// Factors `P Dᵣ A D꜀ = L U`, where `P` permutes rows and `Dᵣ`, `D꜀` scale
/// by exact powers of two, and `n` is the matrix order.
/// Factorization costs `O(n³)` and a solve `O(n²)`;
/// storage costs `O(n²)`. No source matrix, condition estimate or execution
/// accounting is retained. The caller owns admission and publication.
/// Substitution uses range-safe replay. The usual componentwise backward-error
/// bound scales with `γ |L| |U|`, where `γ = 3n ε / (1 - 3n ε)` for
/// `3n ε < 1`, and `ε` is
/// scalar precision; for `n = 16` in binary64, `γ` is about `1.1e-14`.
/// Pivot growth enters the bound through `|L| |U|`.
/// A failed factorization invalidates the factor; solves preserve output on error.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::{backend::lu::LuFactor, MatrixView};
/// let execution = ExecutionContext::sequential();
/// let mut factor = LuFactor::new(2)?;
/// factor.factor(MatrixView::from_slice(2, 2, &[2.0_f64, 1.0, 4.0, 3.0])?, execution)?;
/// let mut x = [0.0; 2];
/// factor.solve_into(&[5.0, 11.0], &mut x, execution)?;
/// assert!((x[0] - 2.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Golub and Van Loan, *Matrix Computations*, 4th ed., §3.2–3.4.
/// - Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Chapter 9.
#[derive(Debug)]
pub struct LuFactor<S: NumericScalar> {
    scalars: Vec<S>,
    indices: Vec<usize>,
    scratch: Vec<S>,
    product: LinalgWorkspace<'static, S>,
    capacity: usize,
    order: Option<usize>,
    maximum: S::Real,
}

impl<S: NumericScalar> LuFactor<S> {
    /// Prepares capacity for matrices of order at most `order` in `O(order²)` space.
    ///
    /// # Errors
    ///
    /// Returns size, allocation or workspace resource errors when preparation fails.
    pub fn new(order: usize) -> Result<Self, SciOxideError> {
        let mut factor = Self {
            scalars: Vec::new(),
            indices: Vec::new(),
            scratch: Vec::new(),
            product: LinalgWorkspace::new(),
            capacity: 0,
            order: None,
            maximum: S::Real::ZERO,
        };
        factor.try_reserve(order)?;
        Ok(factor)
    }

    /// Grows capacity without releasing storage or changing the current factors.
    ///
    /// Repeated factorization and solves within this capacity allocate nothing.
    ///
    /// # Errors
    ///
    /// Returns size, allocation or workspace resource errors when growth fails.
    pub fn try_reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        if order <= self.capacity {
            return Ok(());
        }
        let lengths = factor_kernel_lengths(order)?;
        reserve_vec(
            &mut self.scalars,
            lengths.scalars,
            scioxide_core::quantity!("lu_scalars"),
        )?;
        reserve_vec(
            &mut self.indices,
            lengths.indices,
            scioxide_core::quantity!("lu_indices"),
        )?;
        reserve_vec(
            &mut self.scratch,
            order,
            scioxide_core::quantity!("lu_scratch"),
        )?;
        let options =
            crate::LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        self.product
            .try_reserve(&crate::kernel::product_requirements::<S>(
                &crate::DenseWorkspaceRequest::product(order, order, order, &options),
            )?)?;
        self.scalars.resize(lengths.scalars, S::ZERO);
        self.indices.resize(lengths.indices, 0);
        self.scratch.resize(order, S::ZERO);
        self.capacity = order;
        Ok(())
    }

    /// Returns retained heap bytes, excluding allocator overhead and fixed fields.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let storage = super::util::storage_bytes([
            (self.scalars.capacity(), core::mem::size_of::<S>()),
            (self.indices.capacity(), core::mem::size_of::<usize>()),
            (self.scratch.capacity(), core::mem::size_of::<S>()),
        ])?;
        scioxide_core::backend::checked_add(
            storage,
            self.product.retained_bytes()?,
            scioxide_core::quantity!("lu_storage"),
        )
    }

    /// Factors a finite square matrix within prepared capacity, without allocation.
    ///
    /// Uses the same equilibrated blocked elimination as [`LuDecomposition`].
    /// A failed attempt invalidates the previous factors. Cancellation is checked
    /// before work and between panels; there is no nested solve scope.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` for nonsquare input, `ResourceLimit` for
    /// insufficient capacity, `NonFinite` for invalid entries, `Singular` for
    /// exact zero pivots, and arithmetic or execution errors on breakdown or stops.
    pub fn factor(
        &mut self,
        matrix: crate::MatrixView<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.order = None;
        execution.check()?;
        let n = matrix.rows();
        crate::checked::expect_shape([n, n], [matrix.rows(), matrix.cols()])?;
        if n > self.capacity {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("lu_order"),
                n,
                self.capacity,
            ));
        }
        let candidate = factor_kernel(
            matrix,
            &mut self.scalars,
            &mut self.indices,
            &mut self.product,
            ExecutionPolicy::Sequential,
            KernelContext::new(execution),
        )?;
        execution.check()?;
        self.maximum = candidate.maximum;
        self.order = Some(n);
        Ok(())
    }

    /// Solves `A x = rhs` in prepared scratch and publishes finite output atomically.
    ///
    /// Performs no residual, condition or rank-policy diagnostics. Exact zero
    /// pivots were rejected during factorization. The cost is `O(n²)`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` without valid factors, `DimensionMismatch` for
    /// incompatible vectors, `NonFinite` for invalid input, and arithmetic or
    /// execution errors if substitution cannot produce a finite result.
    pub fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        execution.check()?;
        let n = self.order.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("factor"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "factor the matrix before solving",
            )
        })?;
        crate::checked::expect_shape([n, n], [rhs.len(), output.len()])?;
        crate::checked::validate_values(
            rhs,
            scioxide_core::NonFiniteOrigin::Input,
            "LU right-hand side must be finite",
        )?;
        let permutation = &self.indices[..n];
        let rows = &self.indices[n..2 * n];
        let columns = &self.indices[2 * n..3 * n];
        solve_vector::<S, true>(
            crate::FiniteMatrixView::from_validated(
                crate::MatrixView::from_slice(n, n, &self.scalars[..n * n])?,
                self.maximum,
            ),
            |row| {
                let source = permutation[row];
                scale_numeric(rhs[source], unpack_exponent(rows[source]))
            },
            |row| unpack_exponent(columns[row]),
            output,
            &mut self.scratch,
            KernelContext::new(execution),
        )
    }
}

impl<S: NumericScalar> LuDecomposition<S> {
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
    /// assert_eq!(scioxide_linalg::LuDecomposition::<f64>::factor_storage_bytes(0)?, 0);
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
            (order, core::mem::size_of::<i32>()),
            (order, core::mem::size_of::<i32>()),
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
            (self.lu.capacity(), core::mem::size_of::<S>()),
            (self.source.capacity(), core::mem::size_of::<S>()),
            (self.perm.capacity(), core::mem::size_of::<usize>()),
            (
                self.equilibration.row_exponents.capacity(),
                core::mem::size_of::<i32>(),
            ),
            (
                self.equilibration.column_exponents.capacity(),
                core::mem::size_of::<i32>(),
            ),
        ])
    }

    /// Grows the factor storage to hold an `order × order` factorization.
    ///
    /// The current factorization is unchanged. Growth is exact: after reserving
    /// into [`Self::empty`], [`Self::retained_bytes`] equals
    /// [`Self::factor_storage_bytes`]. A later [`Self::factor_into`] of a matrix
    /// of at most this order does not allocate factor storage. Capacity is never
    /// released.
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
    /// use scioxide_linalg::LuDecomposition;
    /// let mut factor = LuDecomposition::<f64>::empty();
    /// assert_eq!(factor.check(4)?, Capacity::NeedsGrowth);
    /// factor.try_reserve(4)?;
    /// assert_eq!(factor.check(4)?, Capacity::Sufficient);
    /// assert_eq!(
    ///     factor.retained_bytes()?,
    ///     LuDecomposition::<f64>::factor_storage_bytes(4)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(&mut self, order: usize) -> Result<(), SciOxideError> {
        self.lu.try_reserve_shape(order, order)?;
        self.source.try_reserve_shape(order, order)?;
        reserve_vec(
            &mut self.perm,
            order,
            scioxide_core::quantity!("factor.perm"),
        )?;
        for values in [
            &mut self.equilibration.row_exponents,
            &mut self.equilibration.column_exponents,
        ] {
            reserve_vec(values, order, scioxide_core::quantity!("factor.exponents"))?;
        }
        Ok(())
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
        plan.matrix(&self.lu, order, order)?;
        plan.matrix(&self.source, order, order)?;
        plan.buffer(&self.perm, order)?;
        plan.buffer(&self.equilibration.row_exponents, order)?;
        plan.buffer(&self.equilibration.column_exponents, order)?;
        plan.check(None)
    }

    /// Returns an empty factorization of order 0, to pass to [`Self::factor_into`].
    pub const fn empty() -> Self {
        Self {
            lu: Matrix::empty(),
            source: Matrix::empty(),
            maximum: S::Real::ZERO,
            perm: Vec::new(),
            sign: 1,
            n: 0,
            equilibration: EquilibrationMetadata::new(Vec::new(), Vec::new()),
            one_norm: scioxide_core::Scaled::ZERO,
            infinity_norm: scioxide_core::Scaled::ZERO,
        }
    }

    /// Computes the LU factorization `P A = L U` with partial pivoting.
    ///
    /// `L` is unit lower triangular, `U` is upper triangular and `P` is a row
    /// permutation. The factorization costs about 2n³/3 flops and runs on the
    /// calling thread. Use [`Self::factor_into`] to reuse storage or to run the
    /// trailing updates in parallel. For a Hermitian positive definite matrix,
    /// [`CholeskyDecomposition`](crate::CholeskyDecomposition) needs half the flops.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square.
    /// - `NonFinite` if `a` contains NaN or infinity.
    /// - `Singular` if a pivot is exactly zero.
    /// - `NumericalInstability` (`Overflow`) if pivot growth makes an entry of
    ///   `U` overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{LuDecomposition, Matrix, Vector};
    ///
    /// // 2x + y − z = 8, −3x − y + 2z = −11, −2x + y + 2z = −3 has the solution (2, 3, −1).
    /// let a = Matrix::from_rows(&[[2.0_f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])?;
    /// let lu = LuDecomposition::factor(&a)?;
    /// let x = lu.solve(&Vector::from_vec(vec![8.0, -11.0, -3.0]))?;
    /// for (computed, exact) in x.as_slice().iter().zip([2.0, 3.0, -1.0]) {
    ///     assert!((computed - exact).abs() < 1e-12);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    /// Hopkins, 2013, §3.2–3.4.
    pub fn factor<'m>(a: impl Into<crate::MatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, &crate::LinalgExecutionOptions::default())
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
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut factor = Self::empty();
        let mut workspace = LinalgWorkspace::new();
        Self::factor_into(
            a,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(factor)
    }

    /// Computes the LU factorization of `a` into `factor`.
    ///
    /// This is the reusable form of [`Self::factor`]. `workspace` holds the
    /// scratch, and `context` supplies the executor, clock and cancellation. The
    /// trailing-matrix updates follow the execution policy in `options`.
    /// Cancellation and the time limit are checked between panels.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::factor`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn factor_into<'m, 'scope, 'a>(
        a: impl Into<crate::MatrixView<'m, S>>,
        options: &crate::LinalgExecutionOptions,
        factor: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();
        let a = a.into();
        let policy = options.execution_policy();

        let context: crate::KernelContext<'_> = context;
        if a.rows() != a.cols() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(a.rows(), a.rows()),
                scioxide_core::Dimensions::two(a.rows(), a.cols()),
            ));
        }
        context.execution().check()?;
        let a = a.try_finite_dispatched(context.dispatch())?.as_view();
        let n = a.rows();
        let one_norm = crate::diagnostics::matrix_one_norm(a)?;
        let infinity_norm = crate::diagnostics::matrix_one_norm(a.transpose())?;
        let request = LuWorkspaceRequest::factor(n, options).for_kernels(context);
        workspace.try_reserve(&Self::workspace_requirements(&request)?)?;
        factor.try_reserve(n)?;
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "lu_factor", 1),
                n as u128,
                0,
            ),
            options.solve_budgets(),
            |workspace, context, solve| {
                crate::solve_support::admit_iteration(
                    Some(context.execution()),
                    Some(solve),
                    "direct numerical work budget exhausted",
                )?;
                workspace.with_direct(|storage, _, indices, gemm| {
                    let candidate = factor_kernel(a, storage, indices, gemm, policy, context)?;
                    context.execution().check()?;
                    // All growth and numerical work have succeeded; publish the candidate.
                    factor.source.try_resize_flat(n, n)?;
                    copy_rows(a, factor.source.as_mut_slice());
                    factor.lu.try_resize_flat(n, n)?;
                    factor.lu.as_mut_slice().copy_from_slice(candidate.lu);
                    factor.perm.clear();
                    factor.perm.extend_from_slice(candidate.permutation);
                    factor.equilibration.row_exponents.clear();
                    factor
                        .equilibration
                        .row_exponents
                        .extend(candidate.row_exponents.iter().map(|&e| unpack_exponent(e)));
                    factor.equilibration.column_exponents.clear();
                    factor.equilibration.column_exponents.extend(
                        candidate
                            .column_exponents
                            .iter()
                            .map(|&e| unpack_exponent(e)),
                    );
                    factor.maximum = candidate.maximum;
                    factor.sign = candidate.sign;
                    factor.n = n;
                    factor.one_norm = one_norm;
                    factor.infinity_norm = infinity_norm;
                    Ok(())
                })
            },
        )
    }

    fn solve_slices_with_scratch(
        &self,
        b: crate::FiniteVectorView<'_, S>,
        output: &mut [S],
        scratch: &mut [S],
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.execution().check()?;
        crate::checked::expect_shape([self.n], [b.len()])?;
        crate::checked::expect_shape([self.n], [output.len()])?;
        if scratch.len() < self.n {
            return Err(SciOxideError::dimension_mismatch([self.n], [scratch.len()]));
        }

        solve_vector::<S, false>(
            crate::FiniteMatrixView::from_validated(self.lu.view(), self.maximum),
            |row| {
                let source = self.perm[row];
                scale_numeric(
                    b.as_view().get_element(source),
                    self.equilibration.row_exponents[source],
                )
            },
            |row| self.equilibration.column_exponents[row],
            output,
            scratch,
            context,
        )
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
        let rank = self.analyze_rank(options.rank_policy())?;
        let request = LuWorkspaceRequest::solve_matrix(n, rhs.cols(), &options.execution_options())
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
                scioxide_core::algorithm_id!("linalg", "lu_solve_matrix", 1),
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
                    self.solve_matrix_candidate(rhs, candidate, gemm, kernels)?;
                    let mut residual = None;
                    if options.residual() == crate::Estimation::Estimate {
                        let (column, _) = crate::workspace::split_mut(rest, n)?;
                        for col in 0..rhs.cols() {
                            for row in 0..n {
                                column[row] = candidate[row * rhs.cols() + col];
                            }
                            crate::analysis::retain_largest_residual(
                                &mut residual,
                                crate::diagnostics::matrix_backward_error(
                                    self.source.view(),
                                    self.infinity_norm,
                                    crate::VectorView::from_slice(column),
                                    rhs.col(col)?,
                                    kernels,
                                )?,
                            );
                        }
                    }
                    kernels.execution().check()?;
                    for row in 0..n {
                        for col in 0..rhs.cols() {
                            output.set_element(row, col, candidate[row * rhs.cols() + col]);
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

    /// Returns `det A` with a separate binary exponent, so it cannot overflow or underflow.
    ///
    /// `det A = det(P) ∏ U_kk`, with the power-of-two scaling removed. The real
    /// and imaginary parts carry separate exponents. O(n) time.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the binary exponent
    ///   leaves the `i64` range.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{LuDecomposition, Matrix};
    ///
    /// // det A = 1e400 exceeds the `f64` range.
    /// let a = Matrix::from_vec(2, 2, vec![1e200_f64, 0.0, 0.0, 1e200])?;
    /// let determinant = LuDecomposition::factor(&a)?
    ///     .scaled_determinant()?;
    /// assert!(determinant.real().exponent() > 1023);
    /// assert!(determinant.try_value().is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn scaled_determinant(&self) -> Result<scioxide_core::ScaledValue<S>, SciOxideError> {
        let mut value = scioxide_core::ScaledValue::from_value(S::from_real(if self.sign > 0 {
            S::Real::ONE
        } else {
            -S::Real::ONE
        }))?;
        // det(A) = sign * prod(U_kk) / (prod D_r * prod D_c). Each component
        // retains its own exponent throughout the product and equilibration.
        for index in 0..self.n {
            let diagonal = self.lu.get_element(index, index);
            value = value.mul(scioxide_core::ScaledValue::from_value(diagonal)?)?;
        }
        let applied_exponent = self
            .equilibration
            .row_exponents
            .iter()
            .chain(&self.equilibration.column_exponents)
            .try_fold(0_i64, |sum, &exponent| sum.checked_sub(i64::from(exponent)))
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::ExponentOutOfRange,
                    "equilibration exponent accumulation exceeds i64",
                )
            })?;
        value.shifted(applied_exponent)
    }

    /// Returns the sign or phase of `det A` and `ln |det A|`.
    ///
    /// See [`SignLogDeterminant`] for the rounding of the phase. Use
    /// [`Self::scaled_determinant`] to keep each component exactly.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::scaled_determinant`].
    pub fn slogdet(&self) -> Result<SignLogDeterminant<S>, SciOxideError> {
        SignLogDeterminant::from_scaled(self.scaled_determinant()?)
    }

    /// Returns `det A`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if `|det A|` exceeds the largest
    ///   finite value, or (`Underflow`) if a nonzero component rounds to zero.
    ///   [`Self::slogdet`] and [`Self::scaled_determinant`] avoid both.
    pub fn determinant(&self) -> Result<S, SciOxideError> {
        self.scaled_determinant()?.try_value()
    }

    /// Returns `A⁻¹`.
    ///
    /// Solves `A X = I` column by column with the stored factors. Solving with
    /// [`Self::solve`] is cheaper and more accurate than multiplying by the inverse.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) or `NonFinite` if an entry of
    ///   `A⁻¹` overflows.
    pub fn inverse(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.n, self.n)?;
        let mut workspace = LinalgWorkspace::new();
        self.inverse_into(
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut workspace,
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
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn inverse_into<'scope, 'a>(
        &self,
        options: &crate::LinalgExecutionOptions,
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(options)?;
        let context = scope_context.kernel();

        let context: crate::KernelContext<'_> = context;
        if output.rows() != self.n || output.cols() != self.n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(self.n, self.n),
                scioxide_core::Dimensions::two(output.rows(), output.cols()),
            ));
        }
        context.execution().check()?;
        workspace.try_reserve(&Self::workspace_requirements(
            &LuWorkspaceRequest::solve_matrix(self.n, self.n, options).for_kernels(context),
        )?)?;
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "lu_inverse", 1),
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
                    |row, col| {
                        let source = self.perm[row];
                        if col == source {
                            S::from_real(
                                S::Real::ONE.scalbn(self.equilibration.row_exponents[source]),
                            )
                        } else {
                            S::ZERO
                        }
                    },
                    output,
                    workspace,
                    context,
                )
            },
        )
    }

    /// Returns the unit lower triangular factor `L` of `P A = L U`.
    ///
    /// The power-of-two scaling is removed, so `A[p, :] = L U` with `p` from
    /// [`Self::permutation`] and [`Self::to_u_matrix`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if removing the scaling overflows
    ///   an entry.
    /// - `SizeOverflow` or `AllocationFailure` if the output cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{LuDecomposition, Matrix};
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 4.0, 3.0])?;
    /// let factor = LuDecomposition::factor(&a)?;
    /// let (l, u) = (factor.to_l_matrix()?, factor.to_u_matrix()?);
    /// assert_eq!((l.rows(), u.cols()), (2, 2));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_l_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut l = Matrix::zeros(self.n, self.n)?;
        self.l_matrix_into(l.view_mut())?;
        Ok(l)
    }

    /// Writes the factor `L` of [`Self::to_l_matrix`] to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not [`Self::order`] × [`Self::order`].
    /// - `NumericalInstability` (`Overflow`) if removing the scaling overflows
    ///   an entry.
    pub fn l_matrix_into(
        &self,
        mut output: crate::MatrixViewMut<'_, S>,
    ) -> Result<(), SciOxideError> {
        crate::checked::validate_square_output(&output, self.n)?;
        // The stored factors satisfy `P D_r A D_c = L U`. The reported L and
        // U are those of `P A`: `L_true = D^{-1} L D` and
        // `U_true = D^{-1} U D_c^{-1}`, with `D` the permuted row scaling,
        // so L keeps its unit diagonal.
        let value = |row: usize, col: usize| {
            if row == col {
                S::ONE
            } else if col < row {
                scale_numeric(
                    self.lu.get_element(row, col),
                    self.equilibration.row_exponents[self.perm[col]]
                        - self.equilibration.row_exponents[self.perm[row]],
                )
            } else {
                S::ZERO
            }
        };
        for row in 0..self.n {
            for col in 0..self.n {
                crate::checked::finite_candidate(value(row, col))?;
            }
        }
        for row in 0..self.n {
            for col in 0..self.n {
                output.set_element(row, col, value(row, col));
            }
        }
        Ok(())
    }

    /// Returns the upper triangular factor `U` of `P A = L U`.
    ///
    /// See [`Self::to_l_matrix`] for the permutation convention and an example.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if removing the scaling overflows
    ///   an entry.
    /// - `SizeOverflow` or `AllocationFailure` if the output cannot be allocated.
    pub fn to_u_matrix(&self) -> Result<Matrix<S>, SciOxideError> {
        let mut u = Matrix::zeros(self.n, self.n)?;
        self.u_matrix_into(u.view_mut())?;
        Ok(u)
    }

    /// Writes the factor `U` of [`Self::to_u_matrix`] to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not [`Self::order`] × [`Self::order`].
    /// - `NumericalInstability` (`Overflow`) if removing the scaling overflows
    ///   an entry.
    pub fn u_matrix_into(
        &self,
        mut output: crate::MatrixViewMut<'_, S>,
    ) -> Result<(), SciOxideError> {
        crate::checked::validate_square_output(&output, self.n)?;
        let value = |row: usize, col: usize| {
            if row <= col {
                scale_numeric(
                    self.lu.get_element(row, col),
                    -self.equilibration.row_exponents[self.perm[row]]
                        - self.equilibration.column_exponents[col],
                )
            } else {
                S::ZERO
            }
        };
        for row in 0..self.n {
            for col in 0..self.n {
                crate::checked::finite_candidate(value(row, col))?;
            }
        }
        for row in 0..self.n {
            for col in 0..self.n {
                output.set_element(row, col, value(row, col));
            }
        }
        Ok(())
    }

    /// Returns the order `n` of the factored `n × n` matrix.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.n
    }

    /// Returns the row permutation `p`: row `i` of `P A` is row `p[i]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the stored factors of the scaled matrix, `P D_r A D_c = L U`.
    ///
    /// The strictly lower part holds `L` (its unit diagonal is not stored), and
    /// the upper part holds `U`. `D_r` and `D_c` are the powers of two in
    /// [`Self::equilibration`], and `P` is given by [`Self::permutation`]. These
    /// are the exact values used by the solves, for example to build an
    /// approximate inverse for a verified computation.
    pub fn packed_factors(&self) -> crate::MatrixView<'_, S> {
        self.lu.view()
    }

    /// Returns the power-of-two row and column scaling `D_r`, `D_c` applied before elimination.
    pub const fn equilibration(&self) -> &EquilibrationMetadata {
        &self.equilibration
    }

    /// Classifies the numerical rank from the moduli of the pivots `U_kk`.
    ///
    /// The pivots are those of the scaled matrix `P D_r A D_c`. See
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
        analyze_magnitudes(
            (0..self.n).map(|index| self.lu.get_element(index, index).abs()),
            self.n,
            self.n,
            self.n,
            policy,
        )
    }

    /// Estimates the reciprocal condition number `1 / (‖A‖₁ ‖A⁻¹‖₁)` of the original matrix.
    ///
    /// `‖A‖₁` is computed exactly when the matrix is factored. `‖A⁻¹‖₁` is
    /// estimated with Hager's method as refined by Higham, from a few solves with
    /// `A` and `Aᴴ`; for `n ≤ 4` it is computed exactly. The estimate is a lower
    /// bound on `‖A⁻¹‖₁` in exact arithmetic and is usually within a factor of
    /// 10, but it is not guaranteed. O(n²) time per solve.
    ///
    /// # Errors
    ///
    /// - `DomainError` if the matrix is empty.
    /// - `NumericalInstability` (`IllConditioned`) if the estimator cannot
    ///   complete.
    ///
    /// # References
    ///
    /// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    /// SIAM, 2002, Chapter 15.
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
    /// Each solve with `A` or `Aᴴ` counts toward the iteration budget in `options`.
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
                    algorithm: scioxide_core::algorithm_id!("linalg", "lu_condition", 1),
                    order: self.n,
                    norm: self.one_norm,
                    // Factorization rejects exact zero pivots.
                    singular: false,
                })
            },
            |action, kernels| {
                let adjoint = action.adjoint;
                action.solve(
                    |values, temporary| {
                        self.inverse_ordinary(values, temporary, adjoint, kernels.dispatch())
                    },
                    |rhs, work| self.inverse_scaled(rhs, work, adjoint),
                )
            },
        )
    }

    fn inverse_ordinary(
        &self,
        values: &mut [S],
        work: &mut [S],
        adjoint: bool,
        dispatch: scioxide_core::backend::KernelDispatch,
    ) -> Result<(), SciOxideError> {
        let work = &mut work[..self.n];
        let e = &self.equilibration;
        for (i, value) in work.iter_mut().enumerate() {
            let (source, exponent) = if adjoint {
                (i, e.column_exponents[i])
            } else {
                (self.perm[i], e.row_exponents[self.perm[i]])
            };
            *value = if exponent == 0 {
                values[source]
            } else {
                scioxide_core::ScaledValue::from_value(values[source])?
                    .shifted(i64::from(exponent))?
                    .try_value()?
            };
        }
        let triangles = if adjoint {
            [
                (Triangle::Upper, DiagonalKind::Explicit),
                (Triangle::Lower, DiagonalKind::Unit),
            ]
        } else {
            [
                (Triangle::Lower, DiagonalKind::Unit),
                (Triangle::Upper, DiagonalKind::Explicit),
            ]
        };
        for (triangle, diagonal) in triangles {
            crate::kernel::triangular::solve(
                crate::kernel::TriangularOperand {
                    view: crate::kernel::triangular::Storage::Dense(self.lu.view()),
                    triangle,
                    op: if adjoint {
                        crate::kernel::GemmOp::Adjoint
                    } else {
                        crate::kernel::GemmOp::None
                    },
                    diagonal,
                },
                self.maximum,
                work,
                self.n.saturating_sub(1),
                dispatch,
            )?;
        }
        for (i, &value) in work.iter().enumerate() {
            let (destination, exponent) = if adjoint {
                (self.perm[i], e.row_exponents[self.perm[i]])
            } else {
                (i, e.column_exponents[i])
            };
            values[destination] = if exponent == 0 {
                value
            } else {
                scioxide_core::ScaledValue::from_value(value)?
                    .shifted(i64::from(exponent))?
                    .try_value()?
            };
        }
        Ok(())
    }

    fn inverse_scaled(
        &self,
        rhs: &mut [scioxide_core::ScaledValue<S>],
        work: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        let e = &self.equilibration;
        if adjoint {
            for (i, x) in work.iter_mut().enumerate() {
                *x = rhs[i].shifted(i64::from(e.column_exponents[i]))?;
            }
            crate::diagnostics::matrix_substitute(
                self.lu.view(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                true,
                work,
            )?;
            crate::diagnostics::matrix_substitute(
                self.lu.view(),
                scioxide_core::Triangle::Lower,
                scioxide_core::DiagonalKind::Unit,
                true,
                work,
            )?;
            for (i, &source) in self.perm.iter().enumerate() {
                rhs[source] = work[i].shifted(i64::from(e.row_exponents[source]))?;
            }
        } else {
            for (i, &source) in self.perm.iter().enumerate() {
                work[i] = rhs[source].shifted(i64::from(e.row_exponents[source]))?;
            }
            crate::diagnostics::matrix_substitute(
                self.lu.view(),
                scioxide_core::Triangle::Lower,
                scioxide_core::DiagonalKind::Unit,
                false,
                work,
            )?;
            crate::diagnostics::matrix_substitute(
                self.lu.view(),
                scioxide_core::Triangle::Upper,
                scioxide_core::DiagonalKind::Explicit,
                false,
                work,
            )?;
            for (i, x) in rhs.iter_mut().enumerate() {
                *x = work[i].shifted(i64::from(e.column_exponents[i]))?;
            }
        }
        Ok(())
    }

    /// Solves `A x = b` and returns `x`.
    ///
    /// Forward and back substitution cost about 2n² flops. The solve uses the
    /// default [`DirectSolveOptions`](crate::DirectSolveOptions), whose
    /// [`RankPolicy::Exact`](crate::RankPolicy::Exact) counts every nonzero pivot of
    /// `U`, so the factor of a nonsingular matrix always passes. Use
    /// [`Self::solve_with_options`] for a numerical rank cutoff, a condition
    /// threshold or the residual of the solution.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `b` does not have length [`Self::order`].
    /// - `NonFinite` if `b` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
    ///
    /// ```
    /// use scioxide_linalg::{LuDecomposition, Matrix};
    /// let a = Matrix::from_rows(&[[4.0_f64, 3.0], [6.0, 3.0]])?;
    /// let lu = LuDecomposition::factor(&a)?;
    /// // Two right-hand sides reuse one factorization.
    /// let (x, y) = (lu.solve(&[10.0, 12.0][..])?, lu.solve(&[7.0, 9.0][..])?);
    /// assert!((x.as_slice()[0] - 1.0).abs() < 1e-14 && (x.as_slice()[1] - 2.0).abs() < 1e-14);
    /// assert!((y.as_slice()[0] - 1.0).abs() < 1e-14 && (y.as_slice()[1] - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'b>(
        &self,
        b: impl Into<crate::VectorView<'b, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.solve_with_options(b, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `A x = b`, checks the result against `options` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// The solve first classifies the rank of `U` with `options.rank_policy()`. If `options` sets a
    /// minimum reciprocal condition number, it then calls [`Self::condition_estimate`]. A
    /// rank-deficient or ill-conditioned system returns `Ok` with that
    /// [`DirectSolveTermination`](crate::DirectSolveTermination) and no solution; see
    /// [Direct solves](crate#direct-solves). A requested residual is the normwise backward error of
    /// the original system, `‖b − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `b` does not have length [`Self::order`].
    /// - `NonFinite` if `b` contains NaN or infinity.
    /// - `DomainError` if the matrix is empty and a condition threshold is set.
    /// - `NumericalInstability` (`Overflow`) if an entry of the solution overflows.
    /// - Limit errors from `options` as described in [Execution](crate#execution).
    pub fn solve_with_options<'b>(
        &self,
        b: impl Into<crate::VectorView<'b, S>>,
        options: &DirectSolveOptions<S::Real>,
    ) -> Result<DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let options = &options;
        let mut output = Vector::zeros(self.n)?;
        let mut workspace = LinalgWorkspace::new();
        let report = self.solve_into(
            b,
            options,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        let solution = output;
        Ok(DirectSolveResult::new(solution, report))
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
        b: impl Into<crate::VectorView<'b, S>>,
        options: &DirectSolveOptions<S::Real>,
        mut output: crate::VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let scope_context = context.into().with_options(&options.execution_options())?;
        let context = scope_context.kernel();
        let execution = context.execution();
        execution.check()?;
        let b = b.into();
        crate::checked::expect_shape([self.n], [b.len()])?;
        crate::checked::expect_shape([self.n], [output.len()])?;
        let finite_rhs = b.try_finite()?;
        let rank = self.analyze_rank(options.rank_policy())?;
        workspace.try_reserve(&Self::workspace_requirements(
            &LuWorkspaceRequest::solve(self.n, &options.execution_options()).for_kernels(context),
        )?)?;
        if self.n == 0 && options.minimum_reciprocal_condition().is_some() {
            return Err(crate::analysis::empty_condition());
        }
        workspace.run_scope(
            scope_context,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "lu_solve", 1),
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
                let report = crate::analysis::acceptance(rank, condition, options);
                if report.termination() != DirectSolveTermination::Accepted || self.n == 0 {
                    let residual = if self.n == 0 {
                        crate::analysis::empty_residual(options, 1)?
                    } else {
                        None
                    };
                    return Ok(DirectSolveReport::new(
                        report.termination(),
                        rank,
                        condition,
                        residual,
                    ));
                }
                crate::solve_support::admit_iteration(
                    Some(context.execution()),
                    Some(solve),
                    "direct solve budget exhausted",
                )?;
                workspace.with_direct(|storage, _, _, gemm| {
                    let (solution, rest) = crate::workspace::split_mut(storage, self.n)?;
                    let (scratch, _) = crate::workspace::split_mut(rest, self.n)?;
                    if context.accuracy() == scioxide_core::Accuracy::Exact {
                        self.solve_matrix_candidate(
                            crate::MatrixView::from(finite_rhs.as_view()),
                            solution,
                            gemm,
                            context,
                        )?;
                    } else {
                        self.solve_slices_with_scratch(finite_rhs, solution, scratch, context)?;
                    }
                    let residual = (options.residual() == crate::Estimation::Estimate)
                        .then(|| {
                            crate::diagnostics::matrix_backward_error(
                                self.source.view(),
                                self.infinity_norm,
                                crate::VectorView::from_slice(solution),
                                b,
                                context,
                            )
                        })
                        .transpose()?;
                    execution.check()?;
                    output
                        .iter_mut()
                        .zip(solution.iter())
                        .for_each(|(destination, &value)| *destination = value);
                    Ok(DirectSolveReport::new(
                        DirectSolveTermination::Accepted,
                        rank,
                        condition,
                        residual,
                    ))
                })
            },
        )
    }

    fn solve_staged(
        &self,
        columns: usize,
        rhs: impl FnMut(usize, usize) -> S,
        mut output: crate::MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        workspace.with_direct(|storage, _, _, gemm| {
            let candidate = &mut storage[..self.n * columns];
            solve_kernel(
                crate::FiniteMatrixView::from_validated(self.lu.view(), self.maximum),
                |row| self.equilibration.column_exponents[row],
                columns,
                rhs,
                candidate,
                gemm,
                context,
            )?;
            context.execution().check()?;
            for row in 0..self.n {
                for col in 0..columns {
                    output.set_element(row, col, candidate[row * columns + col]);
                }
            }
            Ok(())
        })
    }

    /// Stage `X = A^-1 B` for a finite `n x cols` view `B` in `candidate`
    /// (row-major), with `gemm` product scratch for the triangular solves.
    /// Plan families reuse a prepared factor with their own staging.
    pub(crate) fn solve_matrix_candidate(
        &self,
        rhs: crate::MatrixView<'_, S>,
        candidate: &mut [S],
        gemm: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        solve_kernel(
            crate::FiniteMatrixView::from_validated(self.lu.view(), self.maximum),
            |row| self.equilibration.column_exponents[row],
            rhs.cols(),
            self.permuted_rhs(rhs),
            candidate,
            gemm,
            context,
        )
    }

    // `(P D_r B)[row, col]`: the row-equilibrated right-hand side in pivot order.
    fn permuted_rhs<'r>(
        &'r self,
        rhs: crate::MatrixView<'r, S>,
    ) -> impl FnMut(usize, usize) -> S + 'r {
        move |row, col| {
            let source = self.perm[row];
            scale_numeric(
                rhs.get_element(source, col),
                self.equilibration.row_exponents[source],
            )
        }
    }
}

/// Direct scalar and index lengths of [`factor_kernel`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct FactorKernelLengths {
    pub(crate) scalars: usize,
    pub(crate) indices: usize,
}

/// Direct scratch of [`factor_kernel`] for order `n`: the packed candidate and
/// one column-major panel of scalars, then the permutation and two
/// exponent vectors of indices. Trailing updates charge the caller's product
/// regions, sized as an `n x n x n` product.
pub(crate) fn factor_kernel_lengths(n: usize) -> Result<FactorKernelLengths, SciOxideError> {
    let block = n.min(MAX_PANEL);
    let product = |a, b| {
        scioxide_core::backend::checked_product(&[a, b], scioxide_core::quantity!("lu_scratch"))
    };
    Ok(FactorKernelLengths {
        scalars: [product(n, n)?, product(n, block)?]
            .into_iter()
            .try_fold(0usize, |a, b| {
                a.checked_add(b).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("lu_scratch"))
                })
            })?,
        indices: product(n, 3)?,
    })
}

/// Packed equilibrated factors `P D_r A D_c = L U` staged in workspace slices
/// by [`factor_kernel`]. Exponents keep their `i32` bit patterns in index storage.
pub(crate) struct LuKernelFactor<'a, S: NumericScalar> {
    lu: &'a [S],
    permutation: &'a [usize],
    row_exponents: &'a [usize],
    column_exponents: &'a [usize],
    maximum: S::Real,
    sign: i32,
}

impl<S: NumericScalar> LuKernelFactor<'_, S> {
    /// Stage `X = A^-1 B` for a finite `n x cols` view `B` in `candidate`
    /// (row-major), with `gemm` product scratch for the triangular solves.
    pub(crate) fn solve_matrix(
        &self,
        rhs: crate::MatrixView<'_, S>,
        candidate: &mut [S],
        gemm: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.permutation.len();
        solve_kernel(
            crate::FiniteMatrixView::from_validated(
                crate::MatrixView::from_slice(n, n, self.lu)?,
                self.maximum,
            ),
            |row| unpack_exponent(self.column_exponents[row]),
            rhs.cols(),
            |row, col| {
                let source = self.permutation[row];
                scale_numeric(
                    rhs.get_element(source, col),
                    unpack_exponent(self.row_exponents[source]),
                )
            },
            candidate,
            gemm,
            context,
        )
    }
}

// Index storage keeps the `i32` exponent bit pattern.
fn pack_exponent(exponent: i32) -> usize {
    exponent as u32 as usize
}

fn unpack_exponent(exponent: usize) -> i32 {
    exponent as u32 as i32
}

/// Arena-level blocked LU with partial pivoting of a finite square view.
///
/// Rows and then columns are equilibrated by exact powers of two before
/// right-looking panel elimination. `scalars` and `indices` hold at least
/// [`factor_kernel_lengths`]; trailing updates use `gemm` product scratch.
/// Execution stops are observed between panels. On success the returned
/// factor borrows the packed candidate; on failure the slices are scratch.
pub(crate) fn factor_kernel<'a, S: NumericScalar>(
    a: crate::MatrixView<'_, S>,
    scalars: &'a mut [S],
    indices: &'a mut [usize],
    gemm: &mut LinalgWorkspace<'_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<LuKernelFactor<'a, S>, SciOxideError> {
    let n = a.rows();
    let lengths = factor_kernel_lengths(n)?;
    if scalars.len() < lengths.scalars || indices.len() < lengths.indices {
        return Err(SciOxideError::dimension_mismatch(
            [lengths.scalars, lengths.indices],
            [scalars.len(), indices.len()],
        ));
    }
    let block = n.min(MAX_PANEL);
    let (candidate, rest) = crate::workspace::split_mut(scalars, n * n)?;
    let lower = &mut rest[..n * block];
    let (permutation, rest) = crate::workspace::split_mut(indices, n)?;
    let (row_exponents, rest) = crate::workspace::split_mut(rest, n)?;
    let column_exponents = &mut rest[..n];
    // The panel buffer temporarily holds column maxima, then column powers.
    lower[..n].fill(S::ZERO);
    copy_rows(a, candidate);
    for (values, slot) in candidate
        .chunks_exact_mut(n.max(1))
        .zip(row_exponents.iter_mut())
    {
        let exponent = equilibration_exponent(largest_abs(values)?)?;
        *slot = pack_exponent(exponent);
        scale_slice(values, exponent);
        values
            .iter()
            .zip(&mut lower[..n])
            .for_each(|(value, maximum)| {
                let bound = maximum.real_component();
                if let Some(modulus) = modulus_above(*value, bound) {
                    *maximum = S::from_real(bound.max(modulus));
                }
            });
    }
    // Scaled column maxima bound the equilibrated components.
    let mut bound = S::Real::ZERO;
    for (maximum, slot) in lower[..n].iter_mut().zip(column_exponents.iter_mut()) {
        let exponent = equilibration_exponent(maximum.real_component())?;
        *slot = pack_exponent(exponent);
        let power = S::Real::ONE.scalbn(exponent);
        bound = bound.max(maximum.real_component() * power);
        *maximum = S::from_real(power);
    }
    let mut bound = TrailingBound::new(bound);
    // Every final entry is checked finite once: panels with their L21 and
    // U12 after its solve. Their maxima give the factor bound.
    let mut maximum = S::Real::ZERO;
    scale_columns(candidate, &lower[..n], |col| {
        unpack_exponent(column_exponents[col])
    });
    permutation.iter_mut().enumerate().for_each(|(i, p)| *p = i);
    let mut sign = 1;
    let mut start = 0;
    while start < n {
        context.execution().check()?;
        let end = (start + crate::kernel::panel_width::<S>(context, MAX_PANEL)).min(n);
        let execution = crate::kernel::KernelExecution { policy, context };
        let rows = n - start;
        let panel = &mut lower[..rows * (end - start)];
        let mut pivots = [0_usize; MAX_PANEL];
        let pivots = &mut pivots[..end - start];
        transfer_panel(candidate, n, start..end, panel, true);
        factor_lu_panel(panel, rows, pivots)?;
        transfer_panel(candidate, n, start..end, panel, false);
        for (column, &pivot) in pivots.iter().enumerate() {
            if pivot != column {
                swap_outside(candidate, n, start + column, start + pivot, start..end);
                permutation.swap(start + column, start + pivot);
                sign = -sign;
            }
        }
        if end < n {
            let (next, finished) =
                update_lu_trailing(candidate, n, start..end, panel, bound, gemm, execution)?;
            bound = next;
            maximum = maximum.max(finished);
        } else {
            maximum = maximum.max(finite_columns(panel, rows, end - start, rows)?.maximum);
        }
        start = end;
    }
    Ok(LuKernelFactor {
        lu: candidate,
        permutation,
        row_exponents,
        column_exponents,
        maximum,
        sign,
    })
}

fn solve_vector<S: NumericScalar, const PREPARED: bool>(
    factor: crate::FiniteMatrixView<'_, S>,
    rhs: impl Fn(usize) -> S,
    column_exponent: impl Fn(usize) -> i32,
    output: &mut [S],
    scratch: &mut [S],
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = factor.rows();
    // `(D_r A D_c) y = D_r b` is solved with the stored factors of
    // `P D_r A D_c`, and `x = D_c y`. The row scale belongs to the source
    // row, hence it is indexed through the permutation.
    let candidate = &mut scratch[..n];
    candidate
        .iter_mut()
        .enumerate()
        .for_each(|(row, slot)| *slot = rhs(row));
    candidate
        .iter()
        .try_for_each(|&value| crate::checked::finite_candidate(value).map(|_| ()))?;
    // Short rows do less work than the vector leaf's dispatch and reduction setup.
    if PREPARED && S::IS_REAL && n <= 32 {
        crate::kernel::triangular::solve_small_lu(
            factor.as_view(),
            factor.maximum,
            candidate,
            context.dispatch(),
        )?;
    } else {
        let triangular = crate::kernel::triangular::Storage::Dense(factor.as_view());
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Lower,
                op: crate::kernel::GemmOp::None,
                diagonal: DiagonalKind::Unit,
            },
            factor.maximum,
            candidate,
            n.saturating_sub(1),
            context.dispatch(),
        )?;
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: triangular,
                triangle: Triangle::Upper,
                op: crate::kernel::GemmOp::None,
                diagonal: DiagonalKind::Explicit,
            },
            factor.maximum,
            candidate,
            n.saturating_sub(1),
            context.dispatch(),
        )?;
    }
    // Substitution established finite y; restoring x can still overflow.
    for (row, value) in candidate.iter_mut().enumerate() {
        *value = crate::checked::finite_candidate(scale_numeric(*value, column_exponent(row)))?;
    }
    context.execution().check()?;
    output.copy_from_slice(candidate);
    Ok(())
}

/// Substitute `L U Y = rhs` in `candidate` and restore `X = D_c Y`.
///
/// `rhs(row, col)` supplies the permuted, row-equilibrated right-hand side.
fn solve_kernel<S: NumericScalar>(
    factor: crate::FiniteMatrixView<'_, S>,
    column_exponent: impl Fn(usize) -> i32,
    columns: usize,
    mut rhs: impl FnMut(usize, usize) -> S,
    candidate: &mut [S],
    gemm: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = factor.rows();
    let len = n * columns;
    if candidate.len() < len {
        return Err(SciOxideError::dimension_mismatch([len], [candidate.len()]));
    }
    let candidate = &mut candidate[..len];
    for row in 0..n {
        for col in 0..columns {
            candidate[row * columns + col] = rhs(row, col);
        }
    }
    for (triangle, diagonal) in [
        (Triangle::Lower, DiagonalKind::Unit),
        (Triangle::Upper, DiagonalKind::Explicit),
    ] {
        trsm_bounded(
            crate::kernel::TriangularOperand {
                view: factor,
                triangle,
                op: crate::kernel::GemmOp::None,
                diagonal,
            },
            crate::MatrixViewMut::from_slice(n, columns, candidate)?,
            MatrixSide::Left,
            S::ONE,
            gemm,
            context,
        )?;
    }
    // Substitution established finite Y; restoring X can still overflow.
    for row in 0..n {
        let exponent = column_exponent(row);
        for value in &mut candidate[row * columns..(row + 1) * columns] {
            *value = crate::checked::finite_candidate(scale_numeric(*value, exponent))?;
        }
    }
    Ok(())
}

/// `U12 := L11^-1 U12` and `A22 -= L21 U12` of the row-major matrix after its
/// panel `columns` has been factored in the column-major `panel`.
fn update_lu_trailing<S: NumericScalar>(
    matrix: &mut [S],
    order: usize,
    columns: core::ops::Range<usize>,
    panel: &[S],
    bound: TrailingBound<S::Real>,
    gemm: &mut LinalgWorkspace<'_, S>,
    execution: crate::kernel::KernelExecution<KernelContext<'_>>,
) -> Result<(TrailingBound<S::Real>, S::Real), SciOxideError> {
    let crate::kernel::KernelExecution { policy, context } = execution;
    let core::ops::Range { start, end } = columns;
    let (width, rows) = (end - start, order - start);
    let diagonal = finite_columns(panel, width, width, rows)?;
    let upper = trsm_bounded(
        crate::kernel::TriangularOperand {
            view: diagonal,
            triangle: Triangle::Lower,
            op: crate::kernel::GemmOp::None,
            diagonal: DiagonalKind::Unit,
        },
        crate::MatrixViewMut::from_slice(order, order, matrix)?.block_mut(
            start,
            end,
            width,
            order - end,
        )?,
        MatrixSide::Left,
        S::ONE,
        gemm,
        context,
    )?;
    let (top, bottom) = crate::workspace::split_mut(matrix, end * order)?;
    let right = crate::FiniteMatrixView::from_validated(
        crate::MatrixView::from_slice(end, order, top)?.block(start, end, width, order - end)?,
        upper,
    );
    let left = finite_columns(&panel[width..], rows - width, width, rows)?;
    gemm_workspace_with_context(
        crate::kernel::WorkspaceGemmProblem {
            left,
            right,
            output: crate::MatrixViewMut::from_slice(order - end, order, bottom)?.block_mut(
                0,
                end,
                order - end,
                order - end,
            )?,
            config: GemmConfig {
                alpha: -S::ONE,
                beta: S::ONE,
                prior: bound.prior(),
                ..GemmConfig::ordinary()
            },
        },
        policy,
        gemm,
        context,
    )?;
    Ok((
        bound.update(width, left.maximum, upper),
        diagonal.maximum.max(left.maximum).max(upper),
    ))
}

/// Finite column-major `rows x cols` view with column stride `stride`.
fn finite_columns<S: NumericScalar>(
    data: &[S],
    rows: usize,
    cols: usize,
    stride: usize,
) -> Result<crate::FiniteMatrixView<'_, S>, SciOxideError> {
    let view = crate::MatrixView::from_slice_with_strides(rows, cols, data, 1, stride)?;
    let maximum = crate::kernel::standard::finite_component_maximum(view)?;
    Ok(crate::FiniteMatrixView::from_validated(view, maximum))
}

/// Copy rows `columns.start..` of row-major `matrix` columns `columns` into a
/// column-major panel, or back with `gather == false`.
fn transfer_panel<S: NumericScalar>(
    matrix: &mut [S],
    order: usize,
    columns: core::ops::Range<usize>,
    panel: &mut [S],
    gather: bool,
) {
    let rows = order - columns.start;
    matrix[columns.start * order..]
        .chunks_exact_mut(order)
        .enumerate()
        .for_each(|(row, values)| {
            values[columns.clone()]
                .iter_mut()
                .zip(panel[row..].iter_mut().step_by(rows))
                .for_each(|(value, entry)| {
                    if gather {
                        *entry = *value;
                    } else {
                        *value = *entry;
                    }
                });
        });
}

/// Interchange rows `first < second` of a row-major matrix outside `columns`.
fn swap_outside<S: NumericScalar>(
    matrix: &mut [S],
    order: usize,
    first: usize,
    second: usize,
    columns: core::ops::Range<usize>,
) {
    let (upper, lower) = matrix.split_at_mut(second * order);
    let upper = &mut upper[first * order..(first + 1) * order];
    let lower = &mut lower[..order];
    upper[..columns.start].swap_with_slice(&mut lower[..columns.start]);
    upper[columns.end..].swap_with_slice(&mut lower[columns.end..]);
}

/// First row from `from` on of maximal `abs()` in a column.
fn column_pivot<S: NumericScalar>(column: &[S], from: usize) -> (usize, S::Real) {
    column[from..]
        .iter()
        .enumerate()
        .fold(
            (from, S::Real::ZERO),
            |best, (offset, &value)| match modulus_above(value, best.1) {
                Some(magnitude) if magnitude > best.1 => (from + offset, magnitude),
                _ => best,
            },
        )
}

/// Unblocked partial-pivoting elimination of a column-major panel:
/// column `j` occupies `panel[j * rows..(j + 1) * rows]`, so pivot searches,
/// swaps and eliminations stream contiguous columns. `pivots[j]` receives the
/// panel row interchanged with row `j`; the caller applies the interchanges to
/// the rest of the matrix in pivot order. The pivot is the first row of maximal
/// magnitude, so pivoting is deterministic.
fn factor_lu_panel<S: NumericScalar>(
    panel: &mut [S],
    rows: usize,
    pivots: &mut [usize],
) -> Result<(), SciOxideError> {
    let width = pivots.len();
    let mut pivot = column_pivot(&panel[..rows], 0);
    for column in 0..width {
        let (pivot_row, pivot_magnitude) = pivot;
        if pivot_magnitude == S::Real::ZERO {
            return Err(scioxide_core::SciOxideError::singular(None));
        }
        pivots[column] = pivot_row;
        if pivot_row != column {
            panel
                .chunks_exact_mut(rows)
                .for_each(|values| values.swap(column, pivot_row));
        }
        let next = column + 1;
        let (left, right) = panel.split_at_mut(next * rows);
        let pivot_column = &mut left[column * rows..];
        let diagonal = pivot_column[column];
        let multipliers = &mut pivot_column[next..];
        multipliers
            .iter_mut()
            .for_each(|multiplier| *multiplier = *multiplier / diagonal);
        let multipliers = &*multipliers;
        right[..(width - next) * rows]
            .chunks_exact_mut(rows)
            .for_each(|target| {
                let pivot_value = target[column];
                if pivot_value != S::ZERO {
                    target[next..]
                        .iter_mut()
                        .zip(multipliers)
                        .for_each(|(value, &multiplier)| {
                            *value = *value - multiplier * pivot_value
                        });
                }
            });
        if next < width {
            pivot = column_pivot(&right[..rows], next);
        }
    }
    Ok(())
}

/// The operator `x ↦ A⁻¹ x` of an [`LuDecomposition`], returned by
/// [`LuDecomposition::inverse_operator`].
///
/// It implements [`LinearOperator`](scioxide_core::LinearOperator),
/// [`AdjointOperator`](scioxide_core::AdjointOperator) (`x ↦ A⁻ᴴ x`) and
/// [`Preconditioner`](scioxide_core::Preconditioner). Each application solves
/// with the stored factors and needs `n` scratch scalars. Calls without a
/// context run on the calling thread.
pub struct LuInverseOperator<'a, S: NumericScalar>(&'a LuDecomposition<S>);
impl<S: NumericScalar> LuDecomposition<S> {
    /// Returns this factorization as the operator `x ↦ A⁻¹ x`, for example to use
    /// as a preconditioner or as a factor of a structured product.
    pub fn inverse_operator(&self) -> LuInverseOperator<'_, S> {
        LuInverseOperator(self)
    }
}
impl<S: NumericScalar> LuInverseOperator<'_, S> {
    fn apply(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let input =
            crate::VectorView::from_slice(input).try_finite_dispatched(kernels.dispatch())?;
        self.0
            .solve_slices_with_scratch(input, output, scratch, kernels)
    }
    fn apply_adjoint(
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
        let candidate = &mut scratch[..n];
        for (i, slot) in candidate.iter_mut().enumerate() {
            *slot = crate::checked::finite_candidate(scale_numeric(
                input.as_view().get_element(i),
                self.0.equilibration.column_exponents[i],
            ))?;
        }
        for (triangle, diagonal) in [
            (Triangle::Upper, DiagonalKind::Explicit),
            (Triangle::Lower, DiagonalKind::Unit),
        ] {
            crate::kernel::triangular::solve(
                crate::kernel::TriangularOperand {
                    view: crate::kernel::triangular::Storage::Dense(self.0.lu.view()),
                    triangle,
                    op: crate::kernel::GemmOp::Adjoint,
                    diagonal,
                },
                self.0.maximum,
                candidate,
                n.saturating_sub(1),
                kernels.dispatch(),
            )?;
        }
        for (i, slot) in candidate.iter_mut().enumerate() {
            *slot = crate::checked::finite_candidate(scale_numeric(
                *slot,
                self.0.equilibration.row_exponents[self.0.perm[i]],
            ))?;
        }
        kernels.execution().check()?;
        for (i, &value) in candidate.iter().enumerate() {
            output[self.0.perm[i]] = value;
        }
        Ok(())
    }
}
impl<S: NumericScalar> scioxide_core::LinearOperator<S> for LuInverseOperator<'_, S> {
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
impl<S: NumericScalar> scioxide_core::AdjointOperator<S> for LuInverseOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.0.n)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_adjoint(input, output, scratch, KernelContext::sequential_native())
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_adjoint(input, output, scratch, KernelContext::for_operator(context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: NumericScalar> scioxide_core::Preconditioner<S> for LuInverseOperator<'_, S> {
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

    fn lower_triangular_solve<T: NumericScalar>() {
        let value = |v| T::from_real(T::Real::try_from_f64_lossy(v).unwrap());
        let matrix = Matrix::from_vec(
            3,
            3,
            [4., 0., 0., 2., 3., 0., 0., 1., 2.]
                .into_iter()
                .map(value)
                .collect(),
        )
        .unwrap();
        let rhs = Vector::from_vec([4., -4., 4.].into_iter().map(value).collect());
        for execution in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            let mut factor = LuDecomposition::empty();
            let mut workspace = LinalgWorkspace::new();
            LuDecomposition::factor_into(
                matrix.view(),
                &crate::LinalgExecutionOptions::new(),
                &mut factor,
                &mut workspace,
                execution,
            )
            .unwrap();
            let mut output = Vector::zeros(3).unwrap();
            let report = factor
                .solve_into(
                    rhs.view(),
                    &DirectSolveOptions::new(RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    execution,
                )
                .unwrap();
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            for (&actual, expected) in output.as_slice().iter().zip([1., -2., 3.]) {
                assert!(
                    (actual - value(expected)).abs()
                        <= T::Real::try_from_f64_lossy(32.).unwrap() * T::Real::EPSILON
                );
            }
        }
    }

    #[test]
    fn zero_upper_panel_entries_preserve_lower_triangular_solves() {
        lower_triangular_solve::<f32>();
        lower_triangular_solve::<f64>();
        lower_triangular_solve::<scioxide_core::Complex<f32>>();
        lower_triangular_solve::<scioxide_core::Complex<f64>>();
    }

    #[test]
    fn screened_pivot_search_selects_the_first_maximal_modulus() {
        use scioxide_core::Complex;
        let base = [
            Complex::new(0.25, -0.5),
            Complex::new(-3.0, 4.0),
            Complex::new(4.0, -3.0),
            Complex::new(1e-200, 1e-200),
            Complex::new(5.0, 0.0),
            Complex::new(0.0, -5.0),
            Complex::new(1e200, 0.0),
            Complex::new(-1e200, 0.0),
            Complex::new(3.0, 3.999_999_999_999_999),
        ];
        for start in 0..base.len() {
            let column = &base[start..];
            for from in 0..column.len() {
                let expected = column[from..].iter().enumerate().fold(
                    (from, 0.0_f64),
                    |best, (offset, value)| {
                        let magnitude = value.abs();
                        if magnitude > best.1 {
                            (from + offset, magnitude)
                        } else {
                            best
                        }
                    },
                );
                assert_eq!(column_pivot(column, from), expected);
            }
        }
    }

    #[test]
    fn options_into_is_transactional_and_matches_owned() {
        use crate::ReciprocalConditionThreshold;
        let a = Matrix::from_vec(2, 2, vec![1.0_f64, 1.0, 0.0, 1e-8]).unwrap();
        let factor = LuDecomposition::factor(&a).unwrap();
        let rhs = Vector::from_vec(vec![1.0, 2.0]);
        let mut output = Vector::from_vec(vec![17.0, 19.0]);
        let sentinel = output.clone();
        let mut workspace = LinalgWorkspace::new();
        let rejected = [
            DirectSolveOptions::new(RankPolicy::Threshold(
                scioxide_core::RankThreshold::new(2.0, 0.0).unwrap(),
            )),
            DirectSolveOptions::new(RankPolicy::Exact)
                .with_minimum_reciprocal_condition(ReciprocalConditionThreshold::new(0.9).unwrap()),
        ];
        for options in rejected {
            let report = factor
                .solve_into(
                    &rhs,
                    &options,
                    output.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_ne!(report.termination(), DirectSolveTermination::Accepted);
            assert_eq!(output, sentinel);
        }
        let options =
            DirectSolveOptions::new(RankPolicy::Exact).with_residual(crate::Estimation::Estimate);
        let invalid = Vector::from_vec(vec![f64::NAN, 2.0]);
        assert!(
            factor
                .solve_into(
                    &invalid,
                    &options,
                    output.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native()
                )
                .is_err()
        );
        assert_eq!(output, sentinel);
        let owned = factor.solve_with_options(&rhs, &options).unwrap();
        let report = factor
            .solve_into(
                &rhs,
                &options,
                output.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report, *owned.report());
        assert_eq!(&output, owned.solution().unwrap());
    }

    #[test]
    fn test_decompose_identity() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        let det = lu.determinant().unwrap();
        assert!((det - 1.0).abs() < 1e-14);
    }

    #[test]
    fn blocked_equilibration_retains_extreme_exponents_and_exact_zero_failure() {
        let order = 33;
        let scales = [f64::from_bits(1), f64::MIN_POSITIVE, 1.0, f64::MAX];
        let mut a = Matrix::zeros(order, order).unwrap();
        for i in 0..order {
            a.view_mut().set(i, i, scales[i % scales.len()]).unwrap();
        }
        let options = crate::LinalgExecutionOptions::new();
        let mut factor = LuDecomposition::factor(&a).unwrap();
        assert_eq!(
            &factor.equilibration().row_exponents()[..4],
            &[1074, 1022, 0, -1023]
        );
        assert!(
            factor
                .equilibration()
                .column_exponents()
                .iter()
                .all(|&e| e == 0)
        );
        let before = factor.lu.clone();
        let mut workspace = LinalgWorkspace::new();
        a.view_mut().set(order - 1, order - 1, 0.0).unwrap();
        assert!(
            LuDecomposition::factor_into(
                &a,
                &options,
                &mut factor,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(factor.lu, before);
        a.view_mut().set(order - 1, order - 1, scales[0]).unwrap();
        LuDecomposition::factor_into(
            &a,
            &options,
            &mut factor,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(factor.lu, before);
    }

    #[test]
    fn test_decompose_2x2() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        // det(A) = 4*3 - 3*6 = -6
        let det = lu.determinant().unwrap();
        assert!((det - (-6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_decompose_3x3() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0])
            .unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        // Verify P*A = L*U
        let l = lu.to_l_matrix().unwrap();
        let u = lu.to_u_matrix().unwrap();
        let lu_prod = l.view().mul_mat(&u).unwrap();
        // Apply permutation to original matrix
        for i in 0..3 {
            let orig_row = lu.permutation()[i];
            for j in 0..3 {
                let expected = a.view().get(orig_row, j).unwrap();
                let got = lu_prod.view().get(i, j).unwrap();
                assert!(
                    (expected - got).abs() < 1e-12,
                    "PA != LU at ({i}, {j}): expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_solve_2x2() {
        // A = [[2,1],[4,3]], b = [5,11] => x = [2,1]
        let a = Matrix::<f64>::from_vec(2, 2, vec![2.0, 1.0, 4.0, 3.0]).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        let b = Vector::from_vec(vec![5.0, 11.0]);
        let x = lu.solve(&b).unwrap();
        assert!((x.view().get(0).unwrap() - 2.0).abs() < 1e-12);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_solve_dimension_mismatch() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            (lu.solve(&b)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_singular_matrix() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(matches!(
            (LuDecomposition::factor(&a)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
    }

    #[test]
    fn test_inverse_2x2() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        let inv = lu.inverse().unwrap();
        // A * A^-1 = I
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

    #[test]
    fn test_determinant_f32() {
        let a = Matrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let lu = LuDecomposition::factor(&a).unwrap();
        let det = lu.determinant().unwrap();
        // det = 1*4 - 2*3 = -2
        assert!((det - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn blocked_factorization_reconstructs_and_solves_multiple_rhs() {
        for order in [15, 16, 17, 31, 32, 33, 65, 129] {
            reconstruct_and_solve(order);
        }
    }

    fn reconstruct_and_solve(order: usize) {
        let data = (0..order)
            .flat_map(|row| {
                (0..order).map(move |col| {
                    // A cyclic row permutation forces swaps across panels,
                    // including rows of already formed L.
                    let row = (row + order / 2) % order;
                    if row == col {
                        8.0 + row as f64 / order as f64
                    } else {
                        ((row * 17 + col * 11 + 3) % 19) as f64 / 25.0 - 0.35
                    }
                })
            })
            .collect();
        let a = Matrix::<f64>::from_vec(order, order, data).unwrap();
        let factor = LuDecomposition::factor(&a).unwrap();
        let reconstructed = factor
            .to_l_matrix()
            .unwrap()
            .view()
            .mul_mat(&factor.to_u_matrix().unwrap())
            .unwrap();
        for row in 0..order {
            for col in 0..order {
                let expected = a.get_element(factor.permutation()[row], col);
                assert!((reconstructed.get_element(row, col) - expected).abs() < 1e-10);
            }
        }

        let rhs = Matrix::from_vec(
            order,
            3,
            (0..order * 3)
                .map(|index| (index % 13) as f64 / 7.0 - 0.6)
                .collect(),
        )
        .unwrap();
        let solution = factor.solve_matrix(&rhs).unwrap();
        let residual = a.view().mul_mat(&solution).unwrap();
        for (&actual, &expected) in residual.as_slice().iter().zip(rhs.as_slice()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn reusable_lu_preflight_and_singular_failures_preserve_outputs() {
        let valid = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let singular = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let mut factor = LuDecomposition::factor(&valid).unwrap();
        let before_l = factor.to_l_matrix().unwrap();
        let before_u = factor.to_u_matrix().unwrap();
        let before_perm = factor.permutation().to_vec();
        let mut workspace = LinalgWorkspace::new();
        assert!(matches!(
            (LuDecomposition::factor_into(
                &singular,
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
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
        assert_eq!(factor.to_l_matrix().unwrap(), before_l);
        assert_eq!(factor.to_u_matrix().unwrap(), before_u);
        assert_eq!(factor.permutation(), before_perm);

        let rhs = Vector::from_vec(vec![1.0, f64::NAN]);
        let mut output = Vector::from_vec(vec![7.0, 9.0]);
        assert!(matches!(
            (factor
                .solve_into(
                    &rhs,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
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
    fn slice_solve_rescaling_overflow_preserves_output() {
        fn check<S: RealScalar + NumericScalar<Real = S>>(tiny: S) {
            let one = <S as scioxide_core::AlgebraicScalar>::ONE;
            let eight = S::try_from_f64_lossy(8.0).unwrap();
            let matrix = Matrix::from_vec(2, 2, vec![tiny, one, tiny, -one]).unwrap();
            let factor = LuDecomposition::factor(&matrix).unwrap();
            let mut workspace = LinalgWorkspace::new();
            let mut output = [one, -one];
            let result = factor
                .solve_into(
                    crate::VectorView::from_slice(&[eight, eight]),
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    crate::VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                });
            assert!(matches!(
                (result).as_ref().err().unwrap().view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ));
            assert_eq!(output, [one, -one]);
            factor
                .solve_into(
                    crate::VectorView::from_slice(&[tiny, tiny]),
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    crate::VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                })
                .unwrap();
            assert_eq!(output, [one, <S as scioxide_core::AlgebraicScalar>::ZERO]);
        }
        check(f32::MIN_POSITIVE);
        check(f64::MIN_POSITIVE);
    }

    #[test]
    fn extreme_finite_lu_arithmetic_is_checked_and_transactional() {
        let extreme =
            Matrix::from_vec(2, 2, vec![f64::MAX, -f64::MAX, f64::MAX, f64::MAX]).unwrap();
        let extreme_factor = LuDecomposition::factor(&extreme).unwrap();
        let extreme_determinant = extreme_factor.scaled_determinant().unwrap();
        assert!(extreme_determinant.real().exponent() > 1023);
        assert!(matches!(
            (extreme_determinant.try_value())
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));

        let determinant_overflow = Matrix::from_vec(2, 2, vec![f64::MAX, 0.0, 0.0, 2.0]).unwrap();
        let factor = LuDecomposition::factor(&determinant_overflow).unwrap();
        assert!(matches!(
            (factor.determinant()).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));

        let tiny =
            Matrix::from_vec(2, 2, vec![f64::MIN_POSITIVE, 0.0, 0.0, f64::MIN_POSITIVE]).unwrap();
        let factor = LuDecomposition::factor(&tiny).unwrap();
        let rhs = Vector::from_vec(vec![f64::MAX, 1.0]);
        let mut output = Vector::from_vec(vec![17.0, 19.0]);
        let mut workspace = LinalgWorkspace::new();
        assert!(matches!(
            (factor
                .solve_into(
                    &rhs,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
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
        assert_eq!(output.as_slice(), &[17.0, 19.0]);

        let rhs_matrix = Matrix::from_vec(2, 1, vec![f64::MAX, 1.0]).unwrap();
        let mut matrix_output = Matrix::from_vec(2, 1, vec![23.0, 29.0]).unwrap();
        assert!(matches!(
            (factor
                .solve_matrix_into(
                    &rhs_matrix,
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    matrix_output.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
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
        assert_eq!(matrix_output.as_slice(), &[23.0, 29.0]);
    }

    #[test]
    fn test_non_square_error() {
        let a = Matrix::<f64>::zeros(2, 3).unwrap();
        assert!(matches!(
            (LuDecomposition::factor(&a)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    fn assert_scaled_f64(scale: f64) {
        let a = Matrix::from_vec(2, 2, vec![2.0 * scale, scale, scale, 3.0 * scale]).unwrap();
        let b = Vector::from_vec(vec![3.0 * scale, 4.0 * scale]);
        let x = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
        assert!(f64::abs(x.view().get(0).unwrap() - 1.0) < 1e-12);
        assert!(f64::abs(x.view().get(1).unwrap() - 1.0) < 1e-12);
    }

    fn assert_scaled_f32(scale: f32) {
        let a = Matrix::from_vec(2, 2, vec![2.0 * scale, scale, scale, 3.0 * scale]).unwrap();
        let b = Vector::from_vec(vec![3.0 * scale, 4.0 * scale]);
        let x = LuDecomposition::factor(&a).unwrap().solve(&b).unwrap();
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-5);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_aware_pivots_f64() {
        assert_scaled_f64(1e-200);
        assert_scaled_f64(1e200);

        let mixed = Matrix::from_vec(2, 2, vec![1e200, 0.0, 0.0, 1e-200]).unwrap();
        let rhs = Vector::from_vec(vec![1e200, 1e-200]);
        let x = LuDecomposition::factor(&mixed)
            .unwrap()
            .solve(&rhs)
            .unwrap();
        assert!(f64::abs(x.view().get(0).unwrap() - 1.0) < 1e-12);
        assert!(f64::abs(x.view().get(1).unwrap() - 1.0) < 1e-12);
    }

    #[test]
    fn test_scale_aware_pivots_f32() {
        assert_scaled_f32(1e-20);
        assert_scaled_f32(1e20);

        let mixed = Matrix::from_vec(2, 2, vec![1e20_f32, 0.0, 0.0, 1e-20]).unwrap();
        let rhs = Vector::from_vec(vec![1e20_f32, 1e-20]);
        let x = LuDecomposition::factor(&mixed)
            .unwrap()
            .solve(&rhs)
            .unwrap();
        assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-5);
        assert!((x.view().get(1).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_near_rank_deficiency_is_scale_invariant() {
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let a = Matrix::from_vec(
                2,
                2,
                vec![scale, scale, scale, scale * (1.0 + f64::EPSILON)],
            )
            .unwrap();
            let factor = LuDecomposition::factor(&a).unwrap();
            assert_eq!(
                factor
                    .analyze_rank(RankPolicy::DimensionScaledEpsilon)
                    .unwrap()
                    .rank(),
                1
            );
        }

        for scale in [2.0_f32.powi(-60), 1.0, 2.0_f32.powi(60)] {
            let a = Matrix::from_vec(
                2,
                2,
                vec![scale, scale, scale, scale * (1.0 + f32::EPSILON)],
            )
            .unwrap();
            let factor = LuDecomposition::factor(&a).unwrap();
            assert_eq!(
                factor
                    .analyze_rank(RankPolicy::DimensionScaledEpsilon)
                    .unwrap()
                    .rank(),
                1
            );
        }
    }

    #[test]
    fn test_non_finite_input_is_structural_error() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, f64::NAN]).unwrap();
        assert!(matches!(
            (LuDecomposition::factor(&a)).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
}
