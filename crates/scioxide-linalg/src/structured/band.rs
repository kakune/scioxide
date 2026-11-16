//! Band LU with partial pivoting and band Cholesky factorization.
use super::direct::{
    StructuredJob, StructuredOperation, StructuredWorkspaceRequest, identity, reserve_capacity,
    solve_methods, solve_requirements,
};
use super::storage::{BandedMatrixView, HermitianBandMatrixView};
use crate::kernel::{
    GemmOp,
    triangular::{self, Storage},
};
use crate::{
    KernelContext, LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace,
};
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_core::{DiagonalKind, Triangle};

pub(super) mod kernels;

/// LU factorization of a square band matrix with partial pivoting.
///
/// For an `n × n` matrix `A` with `kl` subdiagonals and `ku` superdiagonals,
/// Gaussian elimination with partial pivoting computes
///
/// ```text
/// U = Mₙ₋₁ Pₙ₋₁ ⋯ M₁ P₁ M₀ P₀ A
/// ```
///
/// where `Pₖ` interchanges rows `k` and [`pivots`](Self::pivots)`[k]` and
/// `Mₖ` is a unit lower triangular Gauss transform with at most `kl`
/// multipliers. At step `k` the pivot is the entry of largest modulus among
/// rows `k..=k + kl` of column `k`. The interchanges widen `U` to `kl + ku`
/// superdiagonals. The multipliers are stored in elimination order, as they
/// were computed, rather than permuted into a single unit lower triangular
/// `L` with `P A = L U`. See [`factors`](Self::factors) for the storage.
///
/// The factorization costs O(n kl (kl + ku)) operations and a solve
/// O(n (2 kl + ku)) per right-hand side. Bands with 16 or more subdiagonals
/// are factored in blocked panels with matrix–matrix updates. Substitution
/// with `U` uses the scaling of [`TriangularPlan`](crate::TriangularPlan) to
/// avoid overflow. Factorization fails only on an exactly zero pivot or when
/// an entry of `U` overflows; see [Direct solves](crate#direct-solves). A
/// multiplier below the normal range rounds to a subnormal number or to zero,
/// which keeps the factors normwise backward stable.
///
/// ```
/// use scioxide_linalg::{BandedLuDecomposition, BandedMatrix, StructuredOperation};
/// // A = [1 1; 1 2]. Each column stores the superdiagonal (padding in
/// // column 0), the diagonal and the subdiagonal (padding in column 1).
/// let a = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0, 1.0, 1.0, 1.0, 2.0, 0.0])?;
/// let factor = BandedLuDecomposition::factor(&a)?;
/// let x = factor
///     .solve(
///         StructuredOperation::Transpose,
///         &[2.0, 3.0][..])?;
/// // Aᵀ x = [2, 3] has the solution x = [1, 1].
/// assert!(x.as_slice().iter().all(|value| (value - 1.0_f64).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §4.3.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BandedLuDecomposition<S: NumericScalar> {
    order: usize,
    lower: usize,
    upper: usize,
    data: Vec<S>,
    pivots: Vec<usize>,
    maximum: S::Real,
    source: super::direct::Source<S>,
}
/// Cholesky factorization `A = L Lᴴ` of a Hermitian positive definite band matrix.
///
/// `A` is a [`HermitianBandMatrix`](crate::HermitianBandMatrix) of order `n`
/// and bandwidth `k`, in either triangle. `L` is lower triangular with
/// bandwidth `k` and a positive real diagonal. It is stored like a lower
/// Hermitian band, `L[i][j]` at position `j (k + 1) + i − j` for
/// `j ≤ i ≤ j + k`; [`lower`](Self::lower) returns it as a band matrix.
///
/// The factorization costs O(n k²) operations and a solve O(n k) per
/// right-hand side. Bands of bandwidth 16 or more are factored in blocked
/// panels with matrix–matrix updates. The substitutions with `L` and `Lᴴ`
/// rescale to avoid overflow, as in
/// [`TriangularPlan`](crate::TriangularPlan). The factorization fails if `A`
/// is not positive definite.
///
/// ```
/// use scioxide_core::Triangle;
/// use scioxide_linalg::{BandedCholeskyDecomposition, HermitianBandMatrix, StructuredOperation};
/// // A = [1 1; 1 2] in lower band storage, diagonal first in each column.
/// let a = HermitianBandMatrix::new(2, 1, Triangle::Lower, vec![1.0, 1.0, 2.0, 0.0])?;
/// let factor = BandedCholeskyDecomposition::factor(&a)?;
/// let x = factor
///     .solve(
///         StructuredOperation::Normal,
///         &[2.0, 3.0][..])?;
/// // A x = [2, 3] has the solution x = [1, 1].
/// assert!(x.as_slice().iter().all(|value| (value - 1.0_f64).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §4.3.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BandedCholeskyDecomposition<S: NumericScalar> {
    order: usize,
    bandwidth: usize,
    data: Vec<S>,
    maximum: S::Real,
    source: super::direct::Source<S>,
}
impl<S: NumericScalar> Default for BandedLuDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: NumericScalar> BandedLuDecomposition<S> {
    /// Returns the factorization of a `0 × 0` matrix.
    ///
    /// Use it as reusable output of [`factor_into`](Self::factor_into).
    pub const fn empty() -> Self {
        Self {
            order: 0,
            lower: 0,
            upper: 0,
            data: Vec::new(),
            pivots: Vec::new(),
            maximum: S::Real::ZERO,
            source: super::direct::Source::empty(),
        }
    }
    /// Returns the order `n`.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the pivot rows: step `k` interchanged rows `k` and `pivots()[k] ≥ k`.
    ///
    /// Indices are zero-based.
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    /// Returns the factors in column-major band storage.
    ///
    /// Each column `j` has `ld = 2 kl + ku + 1` elements. Both `U` and the
    /// multipliers of step `j` are stored at
    ///
    /// ```text
    /// U[i][j] = factors[j · ld + kl + ku + i − j],   j − kl − ku ≤ i ≤ j
    /// l[i][j] = factors[j · ld + kl + ku + i − j],   j < i ≤ j + kl
    /// ```
    ///
    /// so the diagonal of `U` is at position `kl + ku` of each column.
    /// Positions outside the matrix are zero.
    pub fn factors(&self) -> &[S] {
        &self.data
    }

    /// Returns the workspace size for a factor or solve `request`.
    ///
    /// A factor request needs `n (2 kl + ku + 1)` scalars and `n` indices,
    /// plus panel scratch for wide bands. A solve request needs `n m` scalars
    /// for `m` right-hand sides, and the scratch of a condition estimate. A
    /// solve workspace holds scalars only, so a slice you provide can serve as
    /// workspace through [`LinalgWorkspace::from_scalar_slice`].
    ///
    /// # Errors
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the workspace exceeds the byte limit of `request`.
    pub fn workspace_requirements(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        match request.job {
            StructuredJob::Factor {
                order,
                lower,
                upper,
            } => {
                let width = lower
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(upper))
                    .and_then(|v| v.checked_add(1))
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("band_width"))
                    })?;
                let scalars = scioxide_core::backend::checked_product(
                    &[order, width],
                    scioxide_core::quantity!("factor_storage"),
                )?;
                kernels::lu_requirements::<S>(scalars, order, lower, upper, request.byte_limit)
            }
            StructuredJob::Solve { order, columns } => {
                solve_requirements::<S>(order, columns, request.byte_limit)
            }
        }
    }

    /// Allocates a workspace for a factor or solve `request`.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{BandedLuDecomposition, LinalgExecutionOptions, StructuredWorkspaceRequest};
    /// let request = StructuredWorkspaceRequest::factor(16, 2, 1, &LinalgExecutionOptions::new());
    /// let workspace = BandedLuDecomposition::<f64>::workspace(&request)?;
    /// let requirements = BandedLuDecomposition::<f64>::workspace_requirements(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Computes the LU factorization of a square band matrix `a`.
    ///
    /// See [Execution](crate#execution) for how this form runs.
    ///
    /// # Errors
    /// The same conditions as [`factor_into`](Self::factor_into).
    pub fn factor<'m>(a: impl Into<BandedMatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the product accuracy and the limits
    /// described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        a: impl Into<BandedMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            a,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the LU factorization of `a` into `output`, reusing its storage.
    ///
    /// The scratch is described by a factor request
    /// ([`StructuredWorkspaceRequest::factor`]). The storage of `output` grows
    /// if needed. Each pivot column counts as one iteration toward the
    /// iteration budget in `options`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `a` is not square.
    /// - `NonFinite` if a stored entry of `a` is NaN or infinite.
    /// - `Singular` if a pivot is exactly zero.
    /// - `NumericalInstability` if an entry of `U` overflows.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces), and
    ///   `AllocationFailure` if the storage of `output` cannot grow.
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<BandedMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let a = a.into();
        if a.rows() != a.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [a.rows(), a.rows()],
                [a.rows(), a.cols()],
            ));
        }
        let (n, kl, ku) = (a.rows(), a.lower(), a.upper());
        let kernels = KernelContext::new(context.execution());
        let maximum = (0..n).try_fold(S::Real::ZERO, |maximum, j| {
            crate::checked::validated_maximum(a.column(j), "band coefficient", kernels)
                .map(|column| maximum.max(column))
        })?;
        let requirements =
            Self::workspace_requirements(&StructuredWorkspaceRequest::factor(n, kl, ku, options))?;
        let len = n * (2 * kl + ku + 1);
        workspace.try_reserve(&requirements)?;
        reserve_capacity(
            &mut output.data,
            len,
            scioxide_core::quantity!("factor.data"),
        )?;
        output.source.reserve(n, kl, ku)?;
        let norm = crate::diagnostics::column_one_norm(n, |j| a.column(j).iter().copied())?;
        reserve_capacity(
            &mut output.pivots,
            n,
            scioxide_core::quantity!("factor.pivots"),
        )?;
        workspace.run_scope(
            context,
            identity(scioxide_core::algorithm_id!("linalg", "banded_lu", 1), n),
            options.solve_budgets(),
            |workspace, kernels, scope| {
                workspace.with_direct(|storage, _, indices, child| {
                    let (data, scratch) = crate::workspace::split_mut(storage, len)?;
                    let pivots = &mut indices[..n];
                    let maximum = kernels::factor_lu(
                        a,
                        maximum,
                        data,
                        pivots,
                        scratch,
                        child,
                        kernels.execution(),
                        scope,
                    )?;
                    kernels.execution().check()?;
                    output.data.clear();
                    output.data.extend_from_slice(data);
                    output.pivots.clear();
                    output.pivots.extend_from_slice(pivots);
                    output
                        .source
                        .publish(n, kl, ku, norm, |j| a.column(j).iter().copied());
                    output.order = n;
                    output.lower = kl;
                    output.upper = ku;
                    output.maximum = maximum;
                    Ok(())
                })
            },
        )
    }
    solve_methods!(
        BandedLuDecomposition,
        order,
        solve_slice,
        "use scioxide_linalg::{BandedLuDecomposition, BandedMatrix, LinalgExecutionOptions, StructuredOperation};
let a = BandedMatrix::new(2, 2, 1, 1, 3, vec![0.0, 1.0, 1.0, 1.0, 2.0, 0.0])?;
let factor = BandedLuDecomposition::factor(&a)?;"
    );
}
impl<S: NumericScalar> Default for BandedCholeskyDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: NumericScalar> BandedCholeskyDecomposition<S> {
    /// Returns the factorization of a `0 × 0` matrix.
    ///
    /// Use it as reusable output of [`factor_into`](Self::factor_into).
    pub const fn empty() -> Self {
        Self {
            order: 0,
            bandwidth: 0,
            data: Vec::new(),
            maximum: S::Real::ZERO,
            source: super::direct::Source::empty(),
        }
    }
    /// Returns the order `n`.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Returns the workspace size for a factor or solve `request`.
    ///
    /// A factor request of bandwidth `k = max(lower, upper)` needs
    /// `n (k + 1)` scalars plus panel scratch for wide bands. A solve request
    /// needs `n m` scalars for `m` right-hand sides, and the scratch of a
    /// condition estimate.
    ///
    /// # Errors
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the workspace exceeds the byte limit of `request`.
    pub fn workspace_requirements(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        match request.job {
            StructuredJob::Factor {
                order,
                lower,
                upper,
            } => {
                let width = lower.max(upper).checked_add(1).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("band_width"))
                })?;
                let scalars = scioxide_core::backend::checked_product(
                    &[order, width],
                    scioxide_core::quantity!("factor_storage"),
                )?;
                kernels::cholesky_requirements::<S>(
                    scalars,
                    order,
                    lower.max(upper),
                    request.byte_limit,
                )
            }
            StructuredJob::Solve { order, columns } => {
                solve_requirements::<S>(order, columns, request.byte_limit)
            }
        }
    }

    /// Allocates a workspace for a factor or solve `request`.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{
    ///     BandedCholeskyDecomposition, LinalgExecutionOptions, StructuredWorkspaceRequest,
    /// };
    /// let request = StructuredWorkspaceRequest::factor(16, 2, 2, &LinalgExecutionOptions::new());
    /// let workspace = BandedCholeskyDecomposition::<f64>::workspace(&request)?;
    /// let requirements = BandedCholeskyDecomposition::<f64>::workspace_requirements(&request)?;
    /// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Computes the Cholesky factorization of a Hermitian positive definite band matrix `a`.
    ///
    /// See [Execution](crate#execution) for how this form runs.
    ///
    /// # Errors
    /// The same conditions as [`factor_into`](Self::factor_into).
    pub fn factor<'m>(a: impl Into<HermitianBandMatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the product accuracy and the limits
    /// described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        a: impl Into<HermitianBandMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let a = a.into();
        let options = options.with_execution_policy(ExecutionPolicy::Sequential);
        if a.bandwidth() < kernels::BLOCKED {
            return Self::factor_narrow_owned(a, &options);
        }
        let mut output = Self::empty();
        Self::factor_into(
            a,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    // A fresh owned result needs no separate proposal band: failure drops it.
    // Reusable factor_into keeps its transactional workspace-staging contract.
    fn factor_narrow_owned(
        a: HermitianBandMatrixView<'_, S>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let context = LinalgExecutionContext::from(ExecutionContext::sequential_native())
            .with_options(options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let (n, k) = (a.order(), a.bandwidth());
        let maximum = (0..n).try_fold(S::Real::ZERO, |maximum, j| {
            crate::checked::validated_maximum(
                a.stored_column(j),
                "Hermitian band coefficient",
                kernels,
            )
            .map(|column| maximum.max(column))
        })?;
        // Retain the public factor request's byte-limit admission, even though
        // the fresh owned result needs less transient storage than a refactor.
        Self::workspace_requirements(&StructuredWorkspaceRequest::factor(n, k, k, options))?
            .required_bytes::<S>()?;
        let norm = crate::diagnostics::column_one_norm(n, |j| a.column(j))?;
        let mut output = Self::empty();
        output.data = scioxide_core::backend::try_filled(
            n * (k + 1),
            S::ZERO,
            scioxide_core::quantity!("factor.data"),
        )?;
        output.source.reserve(n, k, k)?;
        context.into_scope().run(
            &mut scioxide_core::SolveState::new(),
            identity(
                scioxide_core::algorithm_id!("linalg", "banded_cholesky", 1),
                n,
            ),
            options.solve_budgets(),
            |scope| {
                kernels::factor_cholesky(
                    a,
                    maximum,
                    &mut output.data,
                    &mut [],
                    &mut LinalgWorkspace::new(),
                    kernels.execution(),
                    scope,
                )?;
                let maximum = kernels::maximum(&output.data, k + 1, k + 1, kernels)?;
                kernels.execution().check()?;
                output.source.publish(n, k, k, norm, |j| a.column(j));
                output.order = n;
                output.bandwidth = k;
                output.maximum = maximum;
                Ok(())
            },
        )?;
        Ok(output)
    }

    /// Computes the Cholesky factorization of `a` into `output`, reusing its storage.
    ///
    /// The scratch is described by a factor request with
    /// `lower = upper = bandwidth`. The storage of `output` grows if needed.
    /// Each column counts as one iteration toward the iteration budget in
    /// `options`.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// - `NonFinite` if a stored entry of `a` is NaN or infinite.
    /// - `DomainError` if `a` is not positive definite: a pivot is zero or
    ///   negative.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces), and
    ///   `AllocationFailure` if the storage of `output` cannot grow.
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<HermitianBandMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let a = a.into();
        let (n, k) = (a.order(), a.bandwidth());
        let kernels = KernelContext::new(context.execution());
        let maximum = (0..n).try_fold(S::Real::ZERO, |maximum, j| {
            crate::checked::validated_maximum(
                a.stored_column(j),
                "Hermitian band coefficient",
                kernels,
            )
            .map(|column| maximum.max(column))
        })?;
        let requirements =
            Self::workspace_requirements(&StructuredWorkspaceRequest::factor(n, k, k, options))?;
        let len = n * (k + 1);
        workspace.try_reserve(&requirements)?;
        reserve_capacity(
            &mut output.data,
            len,
            scioxide_core::quantity!("factor.data"),
        )?;
        output.source.reserve(n, k, k)?;
        let norm = crate::diagnostics::column_one_norm(n, |j| a.column(j))?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "banded_cholesky", 1),
                n,
            ),
            options.solve_budgets(),
            |workspace, kernels, scope| {
                workspace.with_direct(|storage, _, _, child| {
                    let (data, scratch) = crate::workspace::split_mut(storage, len)?;
                    kernels::factor_cholesky(
                        a,
                        maximum,
                        data,
                        scratch,
                        child,
                        kernels.execution(),
                        scope,
                    )?;
                    let maximum = kernels::maximum(data, k + 1, k + 1, kernels)?;
                    kernels.execution().check()?;
                    output.data.clear();
                    output.data.extend_from_slice(data);
                    output.source.publish(n, k, k, norm, |j| a.column(j));
                    output.order = n;
                    output.bandwidth = k;
                    output.maximum = maximum;
                    Ok(())
                })
            },
        )
    }
    solve_methods!(
        BandedCholeskyDecomposition,
        order,
        solve_slice,
        "use scioxide_linalg::{BandedCholeskyDecomposition, LinalgExecutionOptions, HermitianBandMatrix, StructuredOperation};
let a = HermitianBandMatrix::new(2, 1, scioxide_core::Triangle::Lower, vec![1.0, 1.0, 2.0, 0.0])?;
let factor = BandedCholeskyDecomposition::factor(&a)?;"
    );
}

impl<S: NumericScalar> BandedLuDecomposition<S> {
    fn upper_storage(&self) -> Storage<'_, S> {
        Storage::Band {
            data: &self.data,
            stride: 2 * self.lower + self.upper + 1,
            diagonal: self.lower + self.upper,
        }
    }
    fn element(&self, i: usize, j: usize) -> S {
        self.data[j * (2 * self.lower + self.upper + 1) + (self.lower + self.upper + i - j)]
    }
    pub(super) fn solve_slice(
        &self,
        x: &mut [S],
        op: StructuredOperation,
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.order;
        if op == StructuredOperation::Normal {
            for j in 0..n {
                x.swap(j, self.pivots[j]);
                for i in j + 1..j.saturating_add(self.lower).saturating_add(1).min(n) {
                    x[i] = crate::kernel::standard::substitute_row(
                        x[i],
                        S::ONE,
                        [(self.element(i, j), x[j])].into_iter(),
                    )?;
                }
            }
            triangular::solve(
                crate::kernel::TriangularOperand {
                    view: self.upper_storage(),
                    triangle: Triangle::Upper,
                    op: operation(op),
                    diagonal: DiagonalKind::Explicit,
                },
                self.maximum,
                x,
                self.lower + self.upper,
                kernels.dispatch(),
            )?;
        } else {
            triangular::solve(
                crate::kernel::TriangularOperand {
                    view: self.upper_storage(),
                    triangle: Triangle::Upper,
                    op: operation(op),
                    diagonal: DiagonalKind::Explicit,
                },
                self.maximum,
                x,
                self.lower + self.upper,
                kernels.dispatch(),
            )?;
            for j in (0..n).rev() {
                x[j] = crate::kernel::standard::substitute_row(
                    x[j],
                    S::ONE,
                    (j + 1..j.saturating_add(self.lower).saturating_add(1).min(n))
                        .map(|i| (op.value(self.element(i, j)), x[i])),
                )?;
                x.swap(j, self.pivots[j]);
            }
        }
        Ok(())
    }
}
impl<S: NumericScalar> BandedCholeskyDecomposition<S> {
    fn lower_storage(&self) -> Storage<'_, S> {
        Storage::Band {
            data: &self.data,
            stride: self.bandwidth + 1,
            diagonal: 0,
        }
    }
    /// Returns the Cholesky factor `L` as a band matrix with `lower = k` and `upper = 0`.
    ///
    /// # Errors
    /// Does not fail for a factor built by this type; the `Result` comes from
    /// the layout checks of [`BandedMatrixView::new`].
    pub fn lower(&self) -> Result<BandedMatrixView<'_, S>, SciOxideError> {
        BandedMatrixView::new(
            self.order,
            self.order,
            self.bandwidth,
            0,
            if self.order == 0 {
                0
            } else {
                self.bandwidth + 1
            },
            &self.data,
        )
    }
    pub(super) fn solve_slice(
        &self,
        x: &mut [S],
        op: StructuredOperation,
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        // A is Hermitian, so `A^T = conj(A)` and `A^T x = b` is solved as
        // `conj(A^{-1} conj(b))` with the same `L L^H` substitutions.
        if op == StructuredOperation::Transpose {
            for v in x.iter_mut() {
                *v = v.conj();
            }
        }
        if S::IS_REAL && self.bandwidth == 2 {
            kernels::solve_real_cholesky_two(&self.data, x)?;
        } else {
            for op in [GemmOp::None, GemmOp::Adjoint] {
                triangular::solve(
                    crate::kernel::TriangularOperand {
                        view: self.lower_storage(),
                        triangle: Triangle::Lower,
                        op,
                        diagonal: DiagonalKind::Explicit,
                    },
                    self.maximum,
                    x,
                    self.bandwidth,
                    kernels.dispatch(),
                )?;
            }
        }
        if op == StructuredOperation::Transpose {
            for v in x.iter_mut() {
                *v = v.conj();
            }
        }
        Ok(())
    }
}

fn operation(op: StructuredOperation) -> GemmOp {
    match op {
        StructuredOperation::Normal => GemmOp::None,
        StructuredOperation::Transpose => GemmOp::Transpose,
        StructuredOperation::Adjoint => GemmOp::Adjoint,
    }
}

impl<S: NumericScalar> BandedLuDecomposition<S> {
    pub(super) fn rank(
        &self,
        policy: crate::RankPolicy<S::Real>,
    ) -> Result<crate::RankReport<S::Real>, SciOxideError> {
        crate::analysis::analyze_scaled_magnitudes(
            (0..self.order)
                .map(|i| scioxide_core::ScaledValue::from_value(self.element(i, i))?.magnitude()),
            self.order,
            policy,
        )
    }
    fn diagnostic_norm(&self) -> Result<scioxide_core::Scaled<S::Real>, SciOxideError> {
        Ok(self.source.norm())
    }
    pub(super) fn residual(
        &self,
        op: StructuredOperation,
        x: &[S],
        b: crate::VectorView<'_, S>,
        context: crate::KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        self.source.residual(op, x, b, context)
    }
    fn solve_scaled(
        &self,
        x: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::ScaledValue;
        if !adjoint {
            for j in 0..self.order {
                x.swap(j, self.pivots[j]);
                for i in j + 1
                    ..j.saturating_add(self.lower)
                        .saturating_add(1)
                        .min(self.order)
                {
                    x[i] = x[i].sub(ScaledValue::from_value(self.element(i, j))?.mul(x[j])?)?;
                }
            }
        }
        crate::diagnostics::substitute(
            x,
            scioxide_core::Triangle::Upper,
            scioxide_core::DiagonalKind::Explicit,
            self.lower + self.upper,
            adjoint,
            |i, j| self.element(i, j),
        )?;
        if adjoint {
            for j in (0..self.order).rev() {
                for i in j + 1
                    ..j.saturating_add(self.lower)
                        .saturating_add(1)
                        .min(self.order)
                {
                    x[j] =
                        x[j].sub(ScaledValue::from_value(self.element(i, j).conj())?.mul(x[i])?)?;
                }
                x.swap(j, self.pivots[j]);
            }
        }
        Ok(())
    }
}
impl<S: NumericScalar> BandedCholeskyDecomposition<S> {
    fn rank(
        &self,
        policy: crate::RankPolicy<S::Real>,
    ) -> Result<crate::RankReport<S::Real>, SciOxideError> {
        if policy == crate::RankPolicy::Exact {
            // A published Cholesky factor has strictly positive finite real pivots.
            // Rank is structural; only the largest squared pivot needs projection.
            let largest = self
                .data
                .chunks_exact(self.bandwidth + 1)
                .fold(S::Real::ZERO, |a, column| a.max(column[0].real_component()));
            let squared = largest * largest;
            let scale = if squared.is_finite() {
                squared
            } else {
                S::Real::MAX
            };
            return Ok(crate::RankReport::from_factor(
                self.order,
                self.order,
                S::Real::ZERO,
                scale,
            ));
        }
        crate::analysis::analyze_scaled_magnitudes(
            (0..self.order).map(|i| {
                scioxide_core::ScaledValue::from_value(self.data[i * (self.bandwidth + 1)])?
                    .magnitude()
                    .and_then(|v| v.mul(v))
            }),
            self.order,
            policy,
        )
    }
    fn diagnostic_norm(&self) -> Result<scioxide_core::Scaled<S::Real>, SciOxideError> {
        Ok(self.source.norm())
    }
    fn residual(
        &self,
        op: StructuredOperation,
        x: &[S],
        b: crate::VectorView<'_, S>,
        context: crate::KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        self.source.residual(op, x, b, context)
    }
    fn solve_scaled(
        &self,
        x: &mut [scioxide_core::ScaledValue<S>],
        _: bool,
    ) -> Result<(), SciOxideError> {
        for adjoint in [false, true] {
            crate::diagnostics::substitute(
                x,
                scioxide_core::Triangle::Lower,
                scioxide_core::DiagonalKind::Explicit,
                self.bandwidth,
                adjoint,
                |i, j| self.data[j * (self.bandwidth + 1) + i - j],
            )?;
        }
        Ok(())
    }
}
