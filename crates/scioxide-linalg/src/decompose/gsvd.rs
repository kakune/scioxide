//! Dense GSVD by a stacked SVD, two-block Jacobi CS iteration and RQ.
//!
//! No Gram matrix or inverse is formed. The stacked SVD selects the common
//! row space, and the SVD of its smaller row block followed by two-block Jacobi
//! rotations finds its cosine/sine coordinates.
//! Factors are written to the output only after the reconstruction check passes.

use super::cs_kernel::{PrerotationScratch, cs_sweep, identity, left_factor, norm, prerotate};
use super::svd::solve::SpectralView;
use super::util::reserve_vec;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix, ScratchVec};
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    ConditionEstimate, KernelContext, LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView,
    RankPolicy, RankReport, ThinSvdDecomposition,
};
use alloc::vec::Vec;
use core::ops::Neg;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::checked_add;
use scioxide_core::checked_float_quotient;
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, NumericScalar, NumericalIssue, RealScalar,
    SciOxideError, SolveContext,
};
use scioxide_core::{Scaled, ScaledAccumulator};

/// Options for [`GsvdDecomposition`]: the rank policy for `[A; B]` and the execution options.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_linalg::{LinalgExecutionOptions, GsvdOptions, RankPolicy};
/// let execution = LinalgExecutionOptions::new()
///     .with_execution_policy(ExecutionPolicy::Sequential)
///     .with_byte_limit(1 << 20)
///     .with_iteration_budget(Budget::ONE.saturating_add(63));
/// let options = GsvdOptions::<f64>::new(RankPolicy::DimensionScaledEpsilon)
///     .with_execution_options(execution);
/// assert_eq!(options.execution_options(), execution);
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GsvdOptions<R: RealScalar> {
    rank_policy: RankPolicy<R>,
    execution: LinalgExecutionOptions,
}
impl<R: RealScalar> Default for GsvdOptions<R> {
    /// Returns options that selects the rank of `[A; B] / s` with [`RankPolicy::Exact`].
    fn default() -> Self {
        Self::new(RankPolicy::Exact)
    }
}

impl<R: RealScalar> GsvdOptions<R> {
    /// Returns options that select the rank of `[A; B] / s` with `rank_policy`.
    ///
    /// `s` is the input scale of [`GsvdDecomposition::input_scale`].
    pub const fn new(rank_policy: RankPolicy<R>) -> Self {
        Self {
            rank_policy,
            execution: LinalgExecutionOptions::new(),
        }
    }
    /// Sets the execution options, which also apply to the nested SVD.
    pub const fn with_execution_options(mut self, execution: LinalgExecutionOptions) -> Self {
        self.execution = execution;
        self
    }
    /// Returns the execution options.
    pub const fn execution_options(self) -> LinalgExecutionOptions {
        self.execution
    }
    /// Returns the rank policy for `[A; B] / s`.
    pub const fn rank_policy(self) -> RankPolicy<R> {
        self.rank_policy
    }
}

/// Generalized singular value `σ = α / β`, stored as the pair `(α, β)`.
///
/// Both components are nonnegative and `α² + β² = 1` up to roundoff. `β = 0`
/// represents an infinite value and `α = 0` a zero value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedSingularValue<R: RealScalar> {
    /// Numerator; zero denotes a zero generalized value.
    alpha: R,
    /// Denominator; zero denotes an infinite generalized value.
    beta: R,
}
impl<R: RealScalar> GeneralizedSingularValue<R> {
    /// Returns the numerator `α ≥ 0`, the cosine; `α = 0` means `σ = 0`.
    pub const fn alpha(self) -> R {
        self.alpha
    }
    /// Returns the denominator `β ≥ 0`, the sine; `β = 0` means `σ = ∞`.
    pub const fn beta(self) -> R {
        self.beta
    }
    /// Returns `σ = α / β`, correctly rounded.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `β = 0`, that is, `σ = ∞`.
    /// - `NumericalInstability` if the quotient overflows.
    pub fn ratio(self) -> Result<R, SciOxideError> {
        checked_float_quotient(self.alpha, self.beta)
    }
}

/// Diagnostics of a [`GsvdDecomposition`], in units of the scaled inputs `A / s` and `B / s`.
///
/// The values are computed estimates, not verified bounds.
#[derive(Debug, Clone, Copy)]
pub struct GsvdReport<R: RealScalar> {
    /// Rank of `[A; B] / input_scale`, selected before CS iteration.
    pub(crate) rank: RankReport<R>,
    /// Threshold-independent condition of the raw stacked spectrum; absent when empty.
    pub(crate) condition: Option<ConditionEstimate<R>>,
    /// `σᵣ / σ₁` of the stacked matrix.
    pub(crate) retained_reciprocal_condition: R,
    /// Frobenius norm of discarded stacked singular values, in normalized units.
    pub(crate) discarded_norm: R,
    /// Frobenius reconstruction error for A, in normalized units.
    pub(crate) residual_a: R,
    /// Frobenius reconstruction error for B, in normalized units.
    pub(crate) residual_b: R,
    /// Number of completed CS sweeps.
    pub(crate) sweeps: usize,
}

impl<R: RealScalar> GsvdReport<R> {
    /// Returns the numerical rank `r` of `[A; B] / s` selected by the rank policy.
    pub const fn rank(&self) -> RankReport<R> {
        self.rank
    }

    /// Returns the reciprocal condition number `σ_min / σ_max` of `[A; B]`.
    ///
    /// No rank threshold is applied. Returns `None` if `[A; B]` has no singular values.
    pub const fn condition(&self) -> Option<ConditionEstimate<R>> {
        self.condition
    }

    /// Returns `σᵣ / σ₁` of `[A; B]`, the reciprocal condition of its rank-`r` part (`1` if `r = 0`).
    pub const fn retained_reciprocal_condition(&self) -> R {
        self.retained_reciprocal_condition
    }

    /// Returns `(Σ_{i > r} σᵢ²)^(1/2)`, the norm of the discarded singular values of `[A; B] / s`.
    pub const fn discarded_norm(&self) -> R {
        self.discarded_norm
    }

    /// Returns `‖A / s − U D₁ [0 R] Qᴴ‖_F`.
    pub const fn residual_a(&self) -> R {
        self.residual_a
    }

    /// Returns `‖B / s − V D₂ [0 R] Qᴴ‖_F`.
    pub const fn residual_b(&self) -> R {
        self.residual_b
    }

    /// Returns the number of Jacobi sweeps.
    pub const fn sweeps(&self) -> usize {
        self.sweeps
    }
}

/// Generalized singular value decomposition of a dense pair `A` (`m × n`) and `B` (`p × n`).
///
/// ```text
/// A = s · U D₁ [0 R] Qᴴ,    B = s · V D₂ [0 R] Qᴴ
/// ```
///
/// - `s` is the input scale: the largest entry magnitude of `A` and `B`, or `1` if both are
///   zero.
/// - `r` is the numerical rank of `[A; B] / s`, chosen by the rank policy of [`GsvdOptions`].
/// - `U` (`m × m`), `V` (`p × p`) and `Q` (`n × n`) are unitary.
/// - `R` is `r × r`, upper triangular and nonsingular. `[0 R]` is `r × n`.
/// - `D₁` is `m × r` and `D₂` is `p × r`. Column `j` of `D₁` has at most one nonzero entry,
///   equal to `αⱼ`, and column `j` of `D₂` at most one, equal to `βⱼ`. Each row also has at
///   most one nonzero entry, and `αⱼ² + βⱼ² = 1`, so `D₁ᴴ D₁ + D₂ᴴ D₂ = I`.
///
/// The generalized singular values are `σⱼ = αⱼ / βⱼ`, returned by
/// [`generalized_values`](Self::generalized_values) in the order of the columns of `D₁`
/// and `D₂` (not sorted). The first `n − r` columns of `Q` span the common null space of
/// `A` and `B` ([`common_null_space`](Self::common_null_space)). [`r_matrix`](Self::r_matrix)
/// is in units of `A / s`; multiply the reconstruction by [`input_scale`](Self::input_scale).
/// Entries smaller than `s` times the smallest subnormal number become zero after scaling.
///
/// The method computes a thin SVD `[A; B] / s = W Σ Zᴴ` and keeps the first `r` columns of
/// `W`. They are rotated by the right singular vectors of their row block with fewer rows,
/// which makes the columns of both blocks orthogonal up to roundoff, and two-block Jacobi
/// sweeps (usually one or two) refine them until the `A` and `B` blocks have orthogonal
/// columns, which gives `U D₁` and `V D₂`. An RQ factorization of the remaining
/// right factor gives `[0 R] Qᴴ`. No Gram matrix or inverse is formed. The factors are
/// accepted only if the reconstruction residual is at most the discarded singular-value
/// norm plus a roundoff allowance; [`report`](Self::report) gives both.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{GsvdDecomposition, Matrix};
/// let a = Matrix::from_vec(2, 2, vec![3.0_f64, 0.0, 0.0, 1.0])?;
/// let b = Matrix::from_vec(2, 2, vec![4.0_f64, 0.0, 0.0, 1.0])?;
/// let gsvd = GsvdDecomposition::factor(&a, &b)?;
/// // For diagonal A and B, the generalized singular values are aᵢᵢ / bᵢᵢ: 3/4 and 1.
/// let mut ratios = [0.0; 2];
/// for (ratio, value) in ratios.iter_mut().zip(gsvd.generalized_values()) {
///     *ratio = value.ratio()?;
/// }
/// ratios.sort_by(|x, y| x.total_cmp(y));
/// assert!((ratios[0] - 0.75).abs() < 1e-12 && (ratios[1] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. C. Paige and M. A. Saunders, "Towards a generalized singular value
///   decomposition", *SIAM J. Numer. Anal.* 18(3), 1981.
#[derive(Debug, Clone)]
#[must_use]
pub struct GsvdDecomposition<S: NumericScalar> {
    u: Matrix<S>,
    v: Matrix<S>,
    q: Matrix<S>,
    r: Matrix<S>,
    d1: Matrix<S>,
    d2: Matrix<S>,
    values: Vec<GeneralizedSingularValue<S::Real>>,
    scale: S::Real,
    report: Option<GsvdReport<S::Real>>,
}
impl<S: NumericScalar> GsvdDecomposition<S> {
    /// Returns an empty decomposition for use as the output of [`factor_into`](Self::factor_into).
    pub fn empty() -> Self {
        Self {
            u: Matrix::empty(),
            v: Matrix::empty(),
            q: Matrix::empty(),
            r: Matrix::empty(),
            d1: Matrix::empty(),
            d2: Matrix::empty(),
            values: Vec::new(),
            scale: S::Real::ONE,
            report: None,
        }
    }
    /// Returns the unitary factor `U` (`m × m`) of `A`.
    pub fn u_matrix(&self) -> &Matrix<S> {
        &self.u
    }
    /// Returns the unitary factor `V` (`p × p`) of `B`.
    pub fn v_matrix(&self) -> &Matrix<S> {
        &self.v
    }
    /// Returns the unitary factor `Q` (`n × n`); its first `n − r` columns span the common null space.
    pub fn q_matrix(&self) -> &Matrix<S> {
        &self.q
    }
    /// Returns the nonsingular upper triangular factor `R` (`r × r`), in units of `A / s`.
    pub fn r_matrix(&self) -> &Matrix<S> {
        &self.r
    }
    /// Returns `D₁` (`m × r`), whose column `j` holds `αⱼ` in at most one row.
    pub fn d1_matrix(&self) -> &Matrix<S> {
        &self.d1
    }
    /// Returns `D₂` (`p × r`), whose column `j` holds `βⱼ` in at most one row.
    pub fn d2_matrix(&self) -> &Matrix<S> {
        &self.d2
    }
    /// Returns the input scale `s > 0`, so that `A = s · U D₁ [0 R] Qᴴ`.
    pub fn input_scale(&self) -> S::Real {
        self.scale
    }
    /// Returns the `r` pairs `(αⱼ, βⱼ)` in the order of the columns of `D₁` and `D₂`.
    ///
    /// This is also the order of the rows of `R`.
    pub fn generalized_values(&self) -> &[GeneralizedSingularValue<S::Real>] {
        &self.values
    }
    /// Returns the diagnostics, or `None` for an [`empty`](Self::empty) result.
    pub fn report(&self) -> Option<&GsvdReport<S::Real>> {
        self.report.as_ref()
    }
    /// Returns the first `n − r` columns of `Q`, an orthonormal basis of the common null space.
    ///
    /// The basis spans the numerical null space of `[A; B]` under the rank policy.
    pub fn common_null_space(&self) -> MatrixView<'_, S> {
        self.q
            .view()
            .take_columns(self.q.cols().saturating_sub(self.r.rows()))
    }
    /// Computes the GSVD of the pair `(A, B)`.
    ///
    /// # Errors
    ///
    /// The errors of [`factor_into`](Self::factor_into).
    pub fn factor<'m>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, b, &crate::GsvdOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the rank policy for `[A; B]` and the
    /// execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        options: &GsvdOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            a,
            b,
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
    /// Computes the GSVD of the pair `(A, B)` into `output`.
    ///
    /// Size the workspace with [`GsvdWorkspaceRequest`]. The stacked SVD counts as one
    /// iteration and each Jacobi sweep as one more toward the iteration limit of the
    /// options. Without a limit, at most 40 sweeps run. The iterations of the two nested
    /// SVDs are counted in child scopes, and the iteration limit also bounds each of them.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `A` and `B` have different column counts.
    /// - `NonFinite` if an entry of `A` or `B` is NaN or infinite.
    /// - The errors of the rank policy.
    /// - `ConvergenceFailure` if a nested SVD or the Jacobi iteration exceeds the
    ///   iteration limit.
    /// - `NumericalInstability` if the reconstruction residual exceeds the discarded
    ///   singular-value norm plus a roundoff allowance, or if `R` has a zero pivot.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        options: &GsvdOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();

        let context = context.into().with_options(&options.execution)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let (a, b) = (a.into(), b.into());
        let (m, p, n) = (a.rows(), b.rows(), a.cols());
        if b.cols() != n {
            return Err(SciOxideError::dimension_mismatch([p, n], [p, b.cols()]));
        }
        let request = GsvdWorkspaceRequest::new(m, p, n, options).for_kernels(kernels);
        let requirements = Self::workspace_requirements(&request)?;
        let a = a.try_finite_dispatched(kernels.dispatch())?;
        let b = b.try_finite_dispatched(kernels.dispatch())?;
        let maximum = a.maximum.max(b.maximum);
        let scale = if maximum == S::Real::ZERO {
            S::Real::ONE
        } else {
            maximum
        };
        let rows = checked_add(m, p, scioxide_core::quantity!("stacked_rows"))?;
        if rows.min(n) != 0 {
            let _ = options.rank_policy.cutoff(S::Real::ONE, rows, n)?;
        }
        workspace.try_reserve(&requirements)?;
        output.reserve(m, p, n)?;
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "gsvd", 1),
            ((m as u128) << 64) | p as u128,
            n as u128,
        );
        let budgets = options.execution.solve_budgets();
        workspace.run_scope(context, identity, budgets, |workspace, kernels, solve| {
            workspace.with_prefixes(request.prefix()?, |scalars, reals, indices, child| {
                let mut work = GsvdScratch::new(scalars, reals, indices, &request)?;
                let report = compute(
                    GsvdProblem {
                        a: a.as_view(),
                        b: b.as_view(),
                    },
                    &mut work,
                    child,
                    *options,
                    scale,
                    kernels,
                    solve,
                )?;
                kernels.execution().check()?;
                let _ = solve.counts().total()?;
                let candidate = &work.candidate;
                for (out, value) in [
                    (&mut output.u, &candidate.u),
                    (&mut output.v, &candidate.v),
                    (&mut output.q, &candidate.q),
                    (&mut output.r, &candidate.r),
                    (&mut output.d1, &candidate.d1),
                    (&mut output.d2, &candidate.d2),
                ] {
                    out.try_resize_flat(value.rows(), value.cols())?;
                    out.as_mut_slice().copy_from_slice(value.as_slice());
                }
                output.values.clear();
                output.values.extend(
                    candidate
                        .alpha
                        .iter()
                        .zip(candidate.beta.iter())
                        .map(|(&alpha, &beta)| GeneralizedSingularValue { alpha, beta }),
                );
                output.scale = scale;
                output.report = Some(report);
                Ok(())
            })
        })
    }
    fn reserve(&mut self, m: usize, p: usize, n: usize) -> Result<(), SciOxideError> {
        let rank = n.min(checked_add(m, p, scioxide_core::quantity!("stacked_rows"))?);
        for (matrix, rows, cols) in [
            (&mut self.u, m, m),
            (&mut self.v, p, p),
            (&mut self.q, n, n),
            (&mut self.r, rank, rank),
            (&mut self.d1, m, rank),
            (&mut self.d2, p, rank),
        ] {
            matrix.try_reserve_shape(rows, cols)?;
        }
        reserve_vec(&mut self.values, rank, scioxide_core::quantity!("values"))
    }
}
impl<S: NumericScalar> Default for GsvdDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Workspace request for the GSVD of an `m × n` matrix `A` and a `p × n` matrix `B`.
#[derive(Debug, Clone, Copy)]
pub struct GsvdWorkspaceRequest<R: RealScalar> {
    m: usize,
    p: usize,
    n: usize,
    options: GsvdOptions<R>,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
}
impl<R: RealScalar> GsvdWorkspaceRequest<R> {
    /// Returns a request for `A` of shape `m × n` and `B` of shape `p × n`.
    pub const fn new(m: usize, p: usize, n: usize, options: &GsvdOptions<R>) -> Self {
        Self {
            m,
            p,
            n,
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
    fn geometry(self) -> Result<([usize; 13], usize), SciOxideError> {
        let Self { m, p, n, .. } = self;
        let rows = checked_add(m, p, scioxide_core::quantity!("stacked_rows"))?;
        let k = rows.min(n);
        let size = |r, c| {
            scioxide_core::backend::checked_product(
                &[r, c],
                scioxide_core::quantity!("gsvd_scratch"),
            )
        };
        Ok((
            [
                size(rows, n)?,
                size(rows, k)?,
                size(k, n)?,
                size(k, n)?,
                m.max(p),
                size(m, m)?,
                size(p, p)?,
                size(n, n)?,
                size(k, k)?,
                size(m, k)?,
                size(p, k)?,
                size(rows, k)?,
                size(k, n)?,
            ],
            k,
        ))
    }
    fn prefix(self) -> Result<[usize; 3], SciOxideError> {
        let (lengths, k) = self.geometry()?;
        let scalars = lengths.into_iter().try_fold(0, |a, b| {
            checked_add(a, b, scioxide_core::quantity!("gsvd_scratch"))
        })?;
        let reals = scioxide_core::backend::checked_product(
            &[5, k],
            scioxide_core::quantity!("gsvd_real_scratch"),
        )?;
        Ok([scalars, reals, k])
    }
}
impl<S: NumericScalar> GsvdDecomposition<S> {
    /// Returns the workspace size for [`factor_into`](Self::factor_into), including the nested SVD.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &GsvdWorkspaceRequest<S::Real>,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let rows = checked_add(
            request.m,
            request.p,
            scioxide_core::quantity!("stacked_rows"),
        )?;
        let mut svd = crate::SvdWorkspaceRequest::new(rows, request.n, &request.options.execution)
            .with_worker_bound(request.workers);
        if matches!(request.packing, crate::workspace::Packing::Provider) {
            svd = svd.with_provider_packing();
        }
        let mut requirements = ThinSvdDecomposition::<S>::workspace_requirements(&svd)?;
        // The SVD of the CS block with fewer rows reuses the nested arena.
        let (_, k) = request.geometry()?;
        let mut block = crate::SvdWorkspaceRequest::new(
            request.m.min(request.p),
            k,
            &request.options.execution,
        )
        .with_worker_bound(request.workers);
        if matches!(request.packing, crate::workspace::Packing::Provider) {
            block = block.with_provider_packing();
        }
        requirements = requirements
            .max(&ThinSvdDecomposition::<S>::workspace_requirements(&block)?)
            .max(&crate::SvdDecomposition::<S>::workspace_requirements(
                &block,
            )?);
        let [scalars, reals, indices] = request.prefix()?;
        *requirements.scalar_mut(ScalarRegion::Direct) = checked_add(
            requirements.scalar(ScalarRegion::Direct),
            scalars,
            scioxide_core::quantity!("gsvd_scratch"),
        )?;
        *requirements.real_mut(RealRegion::Direct) = checked_add(
            requirements.real(RealRegion::Direct),
            reals,
            scioxide_core::quantity!("gsvd_scratch"),
        )?;
        requirements.indices = checked_add(
            requirements.indices,
            indices,
            scioxide_core::quantity!("gsvd_scratch"),
        )?;
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
    /// use scioxide_linalg::{GsvdDecomposition, GsvdOptions, GsvdWorkspaceRequest, RankPolicy};
    /// let options = GsvdOptions::new(RankPolicy::DimensionScaledEpsilon);
    /// let request = GsvdWorkspaceRequest::new(3, 2, 2, &options);
    /// let work = GsvdDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &GsvdWorkspaceRequest<S::Real>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut work = LinalgWorkspace::new();
        work.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(work)
    }
}
struct GsvdCandidate<'a, S: NumericScalar> {
    u: ScratchMatrix<'a, S>,
    v: ScratchMatrix<'a, S>,
    q: ScratchMatrix<'a, S>,
    r: ScratchMatrix<'a, S>,
    d1: ScratchMatrix<'a, S>,
    d2: ScratchMatrix<'a, S>,
    alpha: ScratchVec<'a, S::Real>,
    beta: ScratchVec<'a, S::Real>,
}
struct GsvdScratch<'a, S: NumericScalar> {
    stack: ScratchMatrix<'a, S>,
    cs: ScratchMatrix<'a, S>,
    z: ScratchMatrix<'a, S>,
    right: ScratchMatrix<'a, S>,
    vector: ScratchVec<'a, S>,
    candidate: GsvdCandidate<'a, S>,
    norms_a: ScratchVec<'a, S::Real>,
    norms_b: ScratchVec<'a, S::Real>,
    order: ScratchVec<'a, usize>,
    spectral_u: ScratchMatrix<'a, S>,
    spectral_vt: ScratchMatrix<'a, S>,
    values: ScratchVec<'a, S::Real>,
}
impl<'a, S: NumericScalar> GsvdScratch<'a, S> {
    fn new(
        mut scalars: &'a mut [S],
        reals: &'a mut [S::Real],
        indices: &'a mut [usize],
        request: &GsvdWorkspaceRequest<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let (lengths, k) = request.geometry()?;
        let rows = request.m + request.p;
        let mut take = |length| crate::workspace::take_mut(&mut scalars, length);
        let stack = ScratchMatrix::new(take(lengths[0])?, 0, 0)?;
        let cs = ScratchMatrix::new(take(lengths[1])?, 0, 0)?;
        let z = ScratchMatrix::new(take(lengths[2])?, 0, 0)?;
        let right = ScratchMatrix::new(take(lengths[3])?, 0, 0)?;
        let vector = ScratchVec::new(take(lengths[4])?, request.m.max(request.p))?;
        let u = ScratchMatrix::new(take(lengths[5])?, 0, 0)?;
        let v = ScratchMatrix::new(take(lengths[6])?, 0, 0)?;
        let q = ScratchMatrix::new(take(lengths[7])?, 0, 0)?;
        let r = ScratchMatrix::new(take(lengths[8])?, 0, 0)?;
        let d1 = ScratchMatrix::new(take(lengths[9])?, 0, 0)?;
        let d2 = ScratchMatrix::new(take(lengths[10])?, 0, 0)?;
        let spectral_u = ScratchMatrix::new(take(lengths[11])?, rows, k)?;
        let spectral_vt = ScratchMatrix::new(take(lengths[12])?, k, request.n)?;
        let (norms_a, rest) = crate::workspace::split_mut(reals, k)?;
        let (norms_b, rest) = crate::workspace::split_mut(rest, k)?;
        let (alpha, rest) = crate::workspace::split_mut(rest, k)?;
        let (beta, values) = crate::workspace::split_mut(rest, k)?;
        Ok(Self {
            stack,
            cs,
            z,
            right,
            vector,
            spectral_u,
            spectral_vt,
            values: ScratchVec::new(values, k)?,
            norms_a: ScratchVec::new(norms_a, 0)?,
            norms_b: ScratchVec::new(norms_b, 0)?,
            order: ScratchVec::new(indices, 0)?,
            candidate: GsvdCandidate {
                u,
                v,
                q,
                r,
                d1,
                d2,
                alpha: ScratchVec::new(alpha, 0)?,
                beta: ScratchVec::new(beta, 0)?,
            },
        })
    }
}

fn failure(reason: ConvergenceReason, counts: scioxide_core::SolveCounts) -> SciOxideError {
    SciOxideError::convergence_failure(
        reason,
        ConvergenceDiagnostics::new().with_counts(counts),
        "no complete trustworthy GSVD candidate",
    )
}

fn rq<S: NumericScalar>(
    right: &mut impl MatrixStorage<S>,
    q: &mut impl MatrixStorage<S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // Givens RQ of the `r x n` common factor: working from the bottom row
    // up, every entry left of that row's target diagonal position
    // `n - r + i` is rotated into it (columns `j`, `k`), leaving the upper
    // triangle in the last r columns. Rows above are rotated along, and the
    // rotations accumulate into q so that `right_in = right_out q^H`. The
    // pair is pre-scaled so the rotation length cannot overflow.
    let n = right.cols();
    let r = right.rows();
    identity(q, n)?;
    for i in (0..r).rev() {
        context.execution().check()?;
        let k = n - r + i;
        for j in 0..k {
            let x = right.get_element(i, j);
            let y = right.get_element(i, k);
            let (rotation, _) = crate::rotation::Rotation::generate(y.conj(), x.conj())?;
            let rows = right.rows();
            rotation
                .conjugate()
                .apply(right.try_view_mut()?.transpose(), k, j, 0..rows)?;
            rotation
                .conjugate()
                .apply(q.try_view_mut()?.transpose(), k, j, 0..n)?;
            right.set_element(i, j, S::ZERO);
        }
    }
    Ok(())
}

fn compute<'capability, S: NumericScalar>(
    problem: GsvdProblem<'_, S>,
    work: &mut GsvdScratch<'_, S>,
    child: &mut LinalgWorkspace<'_, S>,
    options: GsvdOptions<S::Real>,
    scale: S::Real,
    context: KernelContext<'capability>,
    solve: &mut SolveContext<'_, 'capability>,
) -> Result<GsvdReport<S::Real>, SciOxideError> {
    let GsvdProblem { a, b } = problem;

    let NormalizedStack { m, p, n, rows } = prepare_stack(a, b, work, scale)?;
    // Algorithm (Paige-Saunders style via a stacked SVD):
    // 1. `[A; B] / scale = W S V^H`, truncated to the numerical rank r.
    // 2. The first r columns of W, split as `[W_A; W_B]`, have orthonormal
    //    columns; a Jacobi CS iteration finds unitary Z with
    //    `[W_A; W_B] Z = [U D1; V D2]` (orthogonal columns in each block).
    // 3. The common right factor `Z^H S V^H` is reduced by RQ to `[0 R] Q^H`.
    // The normalized-unit reconstruction is then checked against the
    // discarded singular mass plus a roundoff allowance before publication.
    crate::solve_support::admit_iteration(
        Some(context.execution()),
        Some(solve),
        "no complete trustworthy GSVD candidate",
    )?;
    super::svd::thin_candidate(
        crate::FiniteMatrixView::from_workspace(work.stack.try_view()?),
        options.execution,
        child,
        context.scoped(solve),
        |spectral| {
            work.values.copy_from_slice(spectral.values);
            for i in 0..rows {
                for j in 0..rows.min(n) {
                    work.spectral_u
                        .set_element(i, j, spectral.u.get_element(i, j));
                }
            }
            for i in 0..rows.min(n) {
                for j in 0..n {
                    work.spectral_vt
                        .set_element(i, j, spectral.vt.get_element(i, j));
                }
            }
            Ok(())
        },
    )?;
    let rank_report = super::svd::spectrum_rank(&work.values, rows, n, options.rank_policy)?;
    let rank = rank_report.rank();
    let values = &work.values;
    let discarded = crate::kernel::standard::norm_workspace(values.iter().skip(rank).copied())?;
    let condition = if values.is_empty() {
        None
    } else {
        Some(super::svd::spectrum_condition_estimate(values)?)
    };
    let retained = if rank == 0 {
        S::Real::ONE
    } else {
        values[rank - 1] / values[0]
    };
    refresh_range(
        &work.stack,
        &SpectralView {
            u: work.spectral_u.try_view()?,
            values,
            vt: work.spectral_vt.try_view()?,
        },
        rank,
        &mut work.cs,
        context,
    )?;
    // The left singular vectors are copied into `cs`, so their storage and
    // that of `D₁` are free until the left factors are formed.
    let (sweeps, tolerance) = orthogonalize_components(
        CsIteration {
            cs: &mut work.cs,
            z: &mut work.z,
            norms_a: &mut work.norms_a,
            norms_b: &mut work.norms_b,
            prerotation: PrerotationScratch {
                block: &mut work.candidate.d1,
                product: &mut work.spectral_u,
            },
            child,
        },
        (rows, m, n, rank),
        options,
        context,
        solve,
    )?;
    work.order.resize(rank, 0)?;
    work.vector.resize(m.max(p), S::ZERO)?;
    let candidate = &mut work.candidate;
    left_factor(
        &work.cs,
        0,
        m,
        crate::decompose::cs_kernel::LeftFactorScratch {
            norms: &work.norms_a,
            order: &mut work.order,
            vector: &mut work.vector,
        },
        &mut candidate.u,
        &mut candidate.d1,
        context,
    )?;
    left_factor(
        &work.cs,
        m,
        p,
        crate::decompose::cs_kernel::LeftFactorScratch {
            norms: &work.norms_b,
            order: &mut work.order,
            vector: &mut work.vector,
        },
        &mut candidate.v,
        &mut candidate.d2,
        context,
    )?;
    candidate.alpha.resize(rank, S::Real::ZERO)?;
    candidate.beta.resize(rank, S::Real::ZERO)?;
    // Structural rectangular complements are represented by explicit zero embeddings.
    // Column j of `[D1; D2]` has norm `hypot(alpha_j, beta_j) = length`
    // (1 up to roundoff); normalizing it gives `alpha^2 + beta^2 = 1`
    // exactly as reported, and the length is pushed into row j of the
    // common factor below so `[U D1; V D2] (common)` is unchanged.
    for j in 0..rank {
        let alpha = norm(&candidate.d1, j, 0, m)?;
        let beta = norm(&candidate.d2, j, 0, p)?;
        let length = alpha.hypot(beta);
        if length == S::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "both generalized components vanished",
            ));
        }
        candidate.alpha[j] = alpha / length;
        candidate.beta[j] = beta / length;
        for d in [&mut candidate.d1, &mut candidate.d2] {
            for i in 0..d.rows() {
                d.set_element(i, j, d.get_element(i, j) / S::from_real(length));
            }
        }
        // Move normalization to the common factor, retaining reconstruction.
        work.norms_a[j] = length;
    }
    work.right.try_resize_flat(rank, n)?;
    for i in 0..rank {
        for j in 0..n {
            let x = (0..rank).fold(S::ZERO, |sum, k| {
                sum + work.z.get_element(k, i).conj()
                    * S::from_real(values[k])
                    * work.spectral_vt.get_element(k, j)
            });
            work.right
                .set_element(i, j, x * S::from_real(work.norms_a[i]));
        }
    }
    rq(&mut work.right, &mut candidate.q, context)?;
    // Independently reconstruct the final common factor, including RQ roundoff.
    work.z.try_resize_flat(rank, n)?;
    for i in 0..rank {
        context.execution().check()?;
        for j in 0..n {
            let value = (0..rank).fold(S::ZERO, |sum, k| {
                sum + work.right.get_element(i, n - rank + k)
                    * candidate.q.get_element(j, n - rank + k).conj()
            });
            work.z.set_element(i, j, value);
        }
    }
    let (residual_a, residual_b) = verify_reconstruction(
        GsvdReconstruction {
            stack: &work.stack,
            candidate,
            common: &work.z,
            values,
            m,
            discarded,
            tolerance,
        },
        &mut work.order,
        context,
    )?;
    retain_triangular_factor(&work.right, candidate, (rank, n))?;
    Ok(GsvdReport {
        rank: rank_report,
        condition,
        retained_reciprocal_condition: retained,
        discarded_norm: discarded,
        residual_a,
        residual_b,
        sweeps,
    })
}
fn retain_triangular_factor<S: NumericScalar>(
    right: &impl MatrixStorage<S>,
    candidate: &mut GsvdCandidate<'_, S>,
    shape: (usize, usize),
) -> Result<(), SciOxideError> {
    let (rank, n) = shape;
    candidate.r.try_resize_flat(rank, rank)?;
    for i in 0..rank {
        if right.get_element(i, n - rank + i) == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "retained common factor has a zero triangular pivot",
            ));
        }
        for j in 0..rank {
            candidate
                .r
                .set_element(i, j, right.get_element(i, n - rank + j));
        }
    }
    Ok(())
}

struct CsIteration<'a, 'b, 'c, S: NumericScalar> {
    cs: &'a mut ScratchMatrix<'b, S>,
    z: &'a mut ScratchMatrix<'b, S>,
    norms_a: &'a mut ScratchVec<'b, S::Real>,
    norms_b: &'a mut ScratchVec<'b, S::Real>,
    prerotation: PrerotationScratch<'a, ScratchMatrix<'b, S>>,
    child: &'a mut LinalgWorkspace<'c, S>,
}
fn orthogonalize_components<'capability, S: NumericScalar>(
    buffers: CsIteration<'_, '_, '_, S>,
    shape: (usize, usize, usize, usize),
    options: GsvdOptions<S::Real>,
    context: KernelContext<'capability>,
    solve: &mut SolveContext<'_, 'capability>,
) -> Result<(usize, S::Real), SciOxideError> {
    let CsIteration {
        cs,
        z,
        norms_a,
        norms_b,
        prerotation,
        child,
    } = buffers;
    let (rows, m, n, rank) = shape;
    identity(z, rank)?;
    let tolerance = crate::checked::dimension_epsilon::<S::Real>(rows.max(n).max(1))?;
    let mut sweeps = 0;
    if rank > 0 {
        prerotate(
            cs,
            z,
            m,
            prerotation,
            options.execution,
            child,
            context,
            solve,
        )?;
        let limit = options
            .execution
            .iteration_budget()
            .map_or(40, |budget| budget.get().saturating_sub(1));
        let mut converged = false;
        for _ in 0..limit {
            crate::solve_support::admit_iteration(
                Some(context.execution()),
                Some(solve),
                "no complete trustworthy GSVD candidate",
            )?;
            sweeps += 1;
            if cs_sweep(cs, z, m, tolerance, context)? {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(failure(ConvergenceReason::IterationLimit, solve.counts()));
        }
    }
    norms_a.resize(rank, S::Real::ZERO)?;
    norms_b.resize(rank, S::Real::ZERO)?;
    for j in 0..rank {
        norms_a[j] = norm(cs, j, 0, m)?;
        norms_b[j] = norm(cs, j, m, rows)?;
    }

    Ok((sweeps, tolerance))
}

struct NormalizedStack {
    m: usize,
    p: usize,
    n: usize,
    rows: usize,
}
fn prepare_stack<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    work: &mut GsvdScratch<'_, S>,
    scale: S::Real,
) -> Result<NormalizedStack, SciOxideError> {
    let (m, p, n) = (a.rows(), b.rows(), a.cols());
    let rows = checked_add(m, p, scioxide_core::quantity!("stacked_rows"))?;
    work.stack.try_resize_flat(rows, n)?;
    // Components that flush under the common maximum-entry scale lie below
    // the smallest subnormal relative to it, far beneath eps * |[A; B]|.
    for (view, offset) in [(a, 0), (b, m)] {
        for i in 0..view.rows() {
            for j in 0..n {
                let x = view.get(i, j)?;
                work.stack.set_element(
                    offset + i,
                    j,
                    S::from_components(x.real_component() / scale, x.imaginary_component() / scale),
                );
            }
        }
    }
    Ok(NormalizedStack { m, p, n, rows })
}
struct GsvdReconstruction<'a, 'b, S: NumericScalar> {
    stack: &'a ScratchMatrix<'b, S>,
    candidate: &'a GsvdCandidate<'b, S>,
    common: &'a ScratchMatrix<'b, S>,
    values: &'a [S::Real],
    m: usize,
    discarded: S::Real,
    tolerance: S::Real,
}
fn verify_reconstruction<S: NumericScalar>(
    problem: GsvdReconstruction<'_, '_, S>,
    order: &mut ScratchVec<'_, usize>,
    context: KernelContext<'_>,
) -> Result<(S::Real, S::Real), SciOxideError> {
    let candidate = problem.candidate;
    let m = problem.m;
    let discarded = problem.discarded;
    let tolerance = problem.tolerance;
    let residual_a = residual(
        problem.stack,
        0,
        &candidate.u,
        &candidate.d1,
        problem.common,
        order,
        context,
    )?;
    let residual_b = residual(
        problem.stack,
        m,
        &candidate.v,
        &candidate.d2,
        problem.common,
        order,
        context,
    )?;
    let scale_norm = crate::kernel::finite::norm(problem.values)?;
    let bound = discarded + tolerance * S::Real::try_from_usize_exact(128)? * scale_norm;
    if residual_a.hypot(residual_b) > bound {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "GSVD reconstruction exceeds truncation plus roundoff allowance",
        ));
    }
    Ok((residual_a, residual_b))
}

fn residual<S: NumericScalar>(
    stack: &impl MatrixStorage<S>,
    offset: usize,
    u: &impl MatrixStorage<S>,
    d: &impl MatrixStorage<S>,
    h: &impl MatrixStorage<S>,
    order: &mut ScratchVec<'_, usize>,
    context: KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    order.resize(d.cols(), d.rows())?;
    for (j, row) in order.iter_mut().enumerate() {
        *row = (0..d.rows())
            .find(|&i| d.get_element(i, j) != S::ZERO)
            .unwrap_or(d.rows());
    }
    let order = &*order;
    // Frobenius norm of the reconstruction defect, one stop check per row.
    crate::kernel::standard::try_norm((0..u.rows()).map(|i| {
        context.execution().check()?;
        crate::kernel::standard::norm_workspace((0..h.cols()).map(|j| {
            let value = (0..d.cols()).fold(S::ZERO, |sum, k| {
                let left = if order[k] == d.rows() {
                    S::ZERO
                } else {
                    u.get_element(i, order[k]) * d.get_element(order[k], k)
                };
                sum + left * h.get_element(k, j)
            });
            stack.get_element(offset + i, j) - value
        }))
    }))
}

fn refresh_range<S: NumericScalar>(
    stack: &impl MatrixStorage<S>,
    svd: &SpectralView<'_, S>,
    rank: usize,
    cs: &mut impl MatrixStorage<S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    // Copies the retained left singular vectors into the CS work matrix.
    // Entries that came out subnormal (or flushed) are recomputed from
    // `(A V)_ij / sigma_j` in extended range, because the CS step derives
    // tiny sines/cosines from these blocks and would otherwise lose them.
    // Recomputed entries below the subnormal range are negligible in a unit
    // column and round to zero.
    let rows = stack.rows();
    let n = stack.cols();
    let values = svd.values;
    cs.try_resize_flat(rows, rank)?;
    for i in 0..rows {
        context.execution().check()?;
        for (j, &sigma) in values.iter().take(rank).enumerate() {
            let original = svd.u.get_element(i, j);
            if original.abs() >= S::Real::MIN_POSITIVE {
                cs.set_element(i, j, original);
                continue;
            }
            let mut real = ScaledAccumulator::ZERO;
            let mut imaginary = ScaledAccumulator::ZERO;
            for k in 0..n {
                let x = stack.get_element(i, k);
                let y = svd.vt.get_element(j, k).conj();
                let xr = Scaled::from_value(x.real_component())?;
                let xi = Scaled::from_value(x.imaginary_component())?;
                let yr = Scaled::from_value(y.real_component())?;
                let yi = Scaled::from_value(y.imaginary_component())?;
                real.add_scaled(xr.mul(yr)?)?;
                real.add_scaled(xi.mul(yi)?.neg())?;
                imaginary.add_scaled(xr.mul(yi)?)?;
                imaginary.add_scaled(xi.mul(yr)?)?;
            }
            // Only repair subnormal tails. Refreshing all of U as A V / sigma
            // would amplify conditioning and destroy its orthogonality.
            let divisor = Scaled::from_value(sigma)?;
            let x = S::from_components(
                real.total()?.div(divisor)?.round()?,
                imaginary.total()?.div(divisor)?.round()?,
            );
            cs.set_element(
                i,
                j,
                if x.abs() < S::Real::MIN_POSITIVE {
                    x
                } else {
                    original
                },
            );
        }
    }
    Ok(())
}

struct GsvdProblem<'a, S: NumericScalar> {
    a: MatrixView<'a, S>,
    b: MatrixView<'a, S>,
}
