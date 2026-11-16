//! RQ factorization, computed from the unpivoted Householder QR of the
//! row- and column-reversed adjoint.

use crate::workspace::ScalarRegion;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::{KernelContext, LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView};
use scioxide_core::{ExecutionContext, ExecutionPolicy, NumericScalar, SciOxideError};

/// Scratch size request for an RQ factorization of a `rows × cols` matrix.
///
/// Pass it to [`RqDecomposition::workspace_requirements`] or
/// [`RqDecomposition::workspace`].
#[derive(Debug, Clone, Copy)]
pub struct RqWorkspaceRequest {
    rows: usize,
    cols: usize,
    options: LinalgExecutionOptions,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
}
impl RqWorkspaceRequest {
    /// Requests scratch for [`RqDecomposition::factor_into`] of a `rows × cols` matrix.
    pub const fn new(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self {
            rows,
            cols,
            options: *options,
            workers: core::num::NonZeroUsize::MAX,
            packing: crate::workspace::Packing::Native,
        }
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
    /// Product accuracy still comes from the options passed to [`Self::new`].
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
    fn prefix(self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("rq_scratch");
        let rectangular =
            scioxide_core::backend::checked_product(&[2, self.rows, self.cols], quantity)?;
        let square = scioxide_core::backend::checked_product(&[self.cols, self.cols], quantity)?;
        scioxide_core::backend::checked_add(rectangular, square, quantity)
    }
}

/// RQ factorization `A = R Q` of an `m × n` matrix.
///
/// `Q` is `n × n` unitary and `R` is `m × n` upper trapezoidal with its diagonal
/// ending at the bottom-right corner: `r_ij = 0` for `j < i + n − m`. Both
/// factors are stored explicitly. They are computed from the unpivoted
/// Householder QR factorization of `J_n Aᴴ J_m`, where `J_k` is the `k × k`
/// reversal permutation: if `J_n Aᴴ J_m = U T`, then `R = J_m Tᴴ J_n` and
/// `Q = J_n Uᴴ J_n`.
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §5.2.
#[derive(Debug, Clone)]
pub struct RqDecomposition<S: NumericScalar> {
    r: Matrix<S>,
    q: Matrix<S>,
}
impl<S: NumericScalar> RqDecomposition<S> {
    /// Returns an empty factorization of a `0 × 0` matrix, to pass to
    /// [`Self::factor_into`].
    pub const fn empty() -> Self {
        Self {
            r: Matrix::empty(),
            q: Matrix::empty(),
        }
    }
    /// Returns the `m × n` upper trapezoidal factor `R`.
    pub fn r_matrix(&self) -> &Matrix<S> {
        &self.r
    }
    /// Returns the `n × n` unitary factor `Q`.
    pub fn q_matrix(&self) -> &Matrix<S> {
        &self.q
    }

    /// Returns the workspace size that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        request: &RqWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        if request.cols == 0 {
            let requirements = crate::LinalgRequirements::new(request.options.byte_limit());
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        let mut requirements = super::qr::requirements::<S>(
            super::qr::QrRequest::Explicit {
                rows: request.cols,
                cols: request.rows,
            },
            &request.options,
            request.workers,
            request.packing,
        )?;
        *requirements.scalar_mut(ScalarRegion::Direct) = scioxide_core::backend::checked_add(
            requirements.scalar(ScalarRegion::Direct),
            request.prefix()?,
            scioxide_core::quantity!("rq_scratch"),
        )?;
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
    /// use scioxide_linalg::{LinalgExecutionOptions, RqDecomposition, RqWorkspaceRequest};
    /// let request = RqWorkspaceRequest::new(2, 3, &LinalgExecutionOptions::new());
    /// let workspace = RqDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &RqWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Computes the RQ factorization `A = R Q` of `a`.
    ///
    /// The factorization runs on the calling thread. Use [`Self::factor_into`]
    /// to reuse storage or to run in parallel.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of a factor overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, RqDecomposition};
    /// let a = Matrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// let rq = RqDecomposition::factor(&a)?;
    /// // For m = 2, n = 3, r_ij = 0 when j < i + 1.
    /// assert_eq!(rq.r_matrix().view().get(1, 0)?, 0.0);
    /// assert_eq!((rq.q_matrix().rows(), rq.q_matrix().cols()), (3, 3));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn factor<'m>(a: impl Into<MatrixView<'m, S>>) -> Result<Self, SciOxideError> {
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
        a: impl Into<MatrixView<'m, S>>,
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
    /// Computes the RQ factorization into `output`, the reusable form of [`Self::factor`].
    ///
    /// The factorization counts one unit toward the iteration budget in `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::factor`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let a = a.into();
        let (m, n) = (a.rows(), a.cols());
        let request = RqWorkspaceRequest::new(m, n, options).for_kernels(kernels);
        let requirements = Self::workspace_requirements(&request)?;
        let finite = a.try_finite_dispatched(kernels.dispatch())?;
        workspace.try_reserve(&requirements)?;
        output.r.try_reserve_shape(m, n)?;
        output.q.try_reserve_shape(n, n)?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "rq", 1),
            ((m as u128) << 64) | n as u128,
            0,
        );
        workspace.run_scope(
            context,
            identity,
            options.solve_budgets(),
            |workspace, kernels, solve| {
                if n == 0 {
                    crate::solve_support::admit_iteration(
                        Some(kernels.execution()),
                        Some(solve),
                        "no complete RQ candidate",
                    )?;
                    output.r.try_resize_flat(m, 0)?;
                    output.q.try_resize_flat(0, 0)?;
                    return Ok(());
                }
                workspace.with_prefixes([request.prefix()?, 0, 0], |scalars, _, _, child| {
                    crate::solve_support::admit_iteration(
                        Some(kernels.execution()),
                        Some(solve),
                        "no complete RQ candidate",
                    )?;
                    let (source, rest) = crate::workspace::split_mut(scalars, m * n)?;
                    let (r, q) = crate::workspace::split_mut(rest, m * n)?;
                    let mut source = ScratchMatrix::new(source, n, m)?;
                    let mut r = ScratchMatrix::new(r, n, m)?;
                    let mut q = ScratchMatrix::new(q, n, n)?;
                    // C = J_n A^H J_m = U T, hence
                    // R = J_m T^H J_n and Q = J_n U^H J_n.
                    for row in 0..n {
                        for col in 0..m {
                            source.as_mut_slice()[row * m + col] = finite
                                .as_view()
                                .get_element(m - 1 - col, n - 1 - row)
                                .conj();
                        }
                    }
                    super::qr::factor_explicit(
                        crate::FiniteMatrixView::from_workspace(source.try_view()?),
                        crate::HouseholderPivoting::None,
                        q.try_view_mut()?,
                        r.try_view_mut()?,
                        None,
                        child,
                        crate::kernel::KernelExecution {
                            policy: options.execution_policy(),
                            context: kernels,
                        },
                    )?;
                    for &value in r.as_slice().iter().chain(q.as_slice()) {
                        crate::checked::finite_candidate(value)?;
                    }
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.r.try_resize_flat(m, n)?;
                    output.q.try_resize_flat(n, n)?;
                    for row in 0..m {
                        for col in 0..n {
                            output.r.as_mut_slice()[row * n + col] =
                                r.get_element(n - 1 - col, m - 1 - row).conj();
                        }
                    }
                    for row in 0..n {
                        for col in 0..n {
                            output.q.as_mut_slice()[row * n + col] =
                                q.get_element(n - 1 - col, n - 1 - row).conj();
                        }
                    }
                    Ok(())
                })
            },
        )
    }
}
impl<S: NumericScalar> Default for RqDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
