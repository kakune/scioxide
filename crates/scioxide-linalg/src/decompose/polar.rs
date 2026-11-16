//! Rectangular polar decomposition from a thin SVD.
use crate::workspace::ScalarRegion;
struct PolarAssembly<R: RealScalar> {
    side: PolarSide,
    scale: R,
    policy: RankPolicy<R>,
}
use super::util::orthogonality;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};

use crate::{
    ConditionEstimate, KernelContext, LinalgExecutionOptions, LinalgWorkspace, Matrix, MatrixView,
    RankPolicy, RankReport, ThinSvdDecomposition,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, NumericalIssue, RealScalar, SciOxideError};

/// Side of the Hermitian positive semidefinite factor `H` in a polar decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolarSide {
    /// `A = U H`, with `H = (Aᴴ A)^(1/2)` of order `n` (the column count).
    #[default]
    Right,
    /// `A = H U`, with `H = (A Aᴴ)^(1/2)` of order `m` (the row count).
    Left,
}
/// Options for [`PolarDecomposition`]: a diagnostic rank policy and the execution options.
///
/// The rank policy only sets [`PolarReport::rank`] and
/// [`PolarReport::completion_nonunique`]. It never truncates the factors.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_linalg::{LinalgExecutionOptions, PolarOptions, RankPolicy};
/// let execution = LinalgExecutionOptions::new()
///     .with_execution_policy(ExecutionPolicy::Sequential)
///     .with_byte_limit(1 << 20)
///     .with_iteration_budget(Budget::ONE.saturating_add(63));
/// let options = PolarOptions::<f64>::new(RankPolicy::DimensionScaledEpsilon)
///     .with_execution_options(execution);
/// assert_eq!(options.execution_options(), execution);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PolarOptions<R: RealScalar> {
    rank: RankPolicy<R>,
    execution: LinalgExecutionOptions,
}
impl<R: RealScalar> Default for PolarOptions<R> {
    /// Returns options that classifies the rank with [`RankPolicy::Exact`].
    fn default() -> Self {
        Self::new(RankPolicy::Exact)
    }
}

impl<R: RealScalar> PolarOptions<R> {
    /// Returns options with the diagnostic rank policy `rank`.
    pub const fn new(rank: RankPolicy<R>) -> Self {
        Self {
            rank,
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
    /// Returns the diagnostic rank policy.
    pub const fn rank_policy(self) -> RankPolicy<R> {
        self.rank
    }
}
/// Diagnostics of a [`PolarDecomposition`], in units of the scaled input `A / s`.
///
/// The values are computed estimates, not verified bounds.
#[derive(Debug, Clone, Copy)]
pub struct PolarReport<R: RealScalar> {
    /// Rank of the input divided by `input_scale()` under the requested policy.
    pub(crate) rank: RankReport<R>,
    /// Raw normalized-spectrum condition estimate; absent for empty spectra.
    pub(crate) condition: Option<ConditionEstimate<R>>,
    /// Whether selected rank loss makes an orthonormal completion nonunique.
    pub(crate) completion_nonunique: bool,
    /// Frobenius reconstruction error divided by the input scale.
    pub(crate) residual: R,
    /// Frobenius error in the smaller of UᴴU and UUᴴ relative to identity.
    pub(crate) orthogonality_error: R,
}

impl<R: RealScalar> PolarReport<R> {
    /// Returns the numerical rank of `A / s` under the rank policy of the options.
    pub const fn rank(&self) -> RankReport<R> {
        self.rank
    }

    /// Returns the reciprocal condition number `σ_min / σ_max`, or `None` if `min(m, n) = 0`.
    pub const fn condition(&self) -> Option<ConditionEstimate<R>> {
        self.condition
    }

    /// Returns `true` if the rank is below `min(m, n)`, so that `U` is not unique.
    pub const fn completion_nonunique(&self) -> bool {
        self.completion_nonunique
    }

    /// Returns the reconstruction residual `‖A − U H‖_F / s` (or `‖A − H U‖_F / s`).
    pub const fn residual(&self) -> R {
        self.residual
    }

    /// Returns `‖Uᴴ U − I‖_F` if `m ≥ n`, or `‖U Uᴴ − I‖_F` if `m < n`.
    pub const fn orthogonality_error(&self) -> R {
        self.orthogonality_error
    }
}

/// Polar decomposition `A = U H` or `A = H U` of a dense `m × n` matrix.
///
/// With the thin SVD `A = Uₖ Σₖ Vₖᴴ`, `k = min(m, n)`:
///
/// ```text
/// U = Uₖ Vₖᴴ                         (m × n)
/// H = Vₖ Σₖ Vₖᴴ = (Aᴴ A)^(1/2)       (n × n)   for PolarSide::Right,  A = U H
/// H = Uₖ Σₖ Uₖᴴ = (A Aᴴ)^(1/2)       (m × m)   for PolarSide::Left,   A = H U
/// ```
///
/// `U` has orthonormal columns if `m ≥ n` and orthonormal rows if `m < n`. No matrix of
/// this kind is closer to `A` in the Frobenius norm. `H` is Hermitian positive
/// semidefinite: only its upper triangle is computed, and it is mirrored with a real
/// diagonal, so `H` is exactly Hermitian. `H` is unique. `U` is unique if and only if `A`
/// has rank `k`; [`PolarReport::completion_nonunique`] reports rank loss under the rank
/// policy. Empty dimensions give empty factors of the declared shapes.
///
/// The input is divided by its largest entry magnitude `s` ([`input_scale`](Self::input_scale))
/// before the SVD, and `H` is multiplied by `s` afterwards. Entries smaller than `s` times the
/// smallest subnormal number become zero. With `τ = 128 max(m, n) ε`, the factors are
/// accepted only if the residual of [`PolarReport::residual`] is at most `τ ‖A / s‖_F` and
/// the orthogonality error of `U` is at most `τ`. The cost is that of the thin SVD plus
/// `O(m n k + h² k + m n h)` for forming the factors and checking reconstruction,
/// where `h = n` for the right decomposition and `h = m` for the left one.
/// The factors occupy `O(m n + h²)` scalars, in addition to SVD workspace.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, PolarDecomposition, PolarOptions, PolarSide, RankPolicy};
/// let a = Matrix::from_vec(2, 2, vec![0.0_f64, -1.0, 2.0, 0.0])?;
/// let options = PolarOptions::new(RankPolicy::DimensionScaledEpsilon);
/// let polar = PolarDecomposition::factor_with_options(&a, PolarSide::Right, &options)?;
/// // H = (AᵀA)^(1/2) = diag(2, 1) and U = A H⁻¹ = [[0, −1], [1, 0]].
/// let (h, u) = (polar.h_matrix().as_slice(), polar.u_matrix().as_slice());
/// assert!((h[0] - 2.0).abs() < 1e-14 && (h[3] - 1.0).abs() < 1e-14);
/// assert!((u[1] + 1.0).abs() < 1e-14 && (u[2] - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008, Ch. 8.
#[derive(Debug, Clone)]
pub struct PolarDecomposition<S: NumericScalar> {
    u: Matrix<S>,
    h: Matrix<S>,
    side: PolarSide,
    scale: S::Real,
    report: Option<PolarReport<S::Real>>,
}
impl<S: NumericScalar> PolarDecomposition<S> {
    /// Returns an empty decomposition for use as the output of [`factor_into`](Self::factor_into).
    pub fn empty() -> Self {
        Self {
            u: Matrix::empty(),
            h: Matrix::empty(),
            side: PolarSide::Right,
            scale: S::Real::ONE,
            report: None,
        }
    }
    /// Returns `U` (`m × n`), with orthonormal columns if `m ≥ n` and orthonormal rows otherwise.
    pub fn u_matrix(&self) -> &Matrix<S> {
        &self.u
    }
    /// Returns the Hermitian positive semidefinite factor `H`, in the units of `A`.
    pub fn h_matrix(&self) -> &Matrix<S> {
        &self.h
    }
    /// Returns the side of `H` used for this decomposition.
    pub fn side(&self) -> PolarSide {
        self.side
    }
    /// Returns the input scale `s`: the largest entry magnitude of `A`, or `1` if `A = 0`.
    pub fn input_scale(&self) -> S::Real {
        self.scale
    }
    /// Returns the diagnostics, or `None` for an [`empty`](Self::empty) result.
    pub fn report(&self) -> Option<&PolarReport<S::Real>> {
        self.report.as_ref()
    }
    /// Computes the polar decomposition of `a` with `H` on the given side.
    ///
    /// # Errors
    ///
    /// The errors of [`factor_into`](Self::factor_into).
    pub fn factor<'m>(
        a: impl Into<MatrixView<'m, S>>,
        side: PolarSide,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(a, side, &crate::PolarOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the diagnostic rank policy and the
    /// execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        a: impl Into<MatrixView<'m, S>>,
        side: PolarSide,
        options: &PolarOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            a,
            side,
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
    /// Computes the polar decomposition of `a` into `output`.
    ///
    /// Size the workspace with [`PolarWorkspaceRequest`]. The call counts as one
    /// iteration toward the iteration limit of the options. The iterations of the
    /// nested SVD are counted in a child scope.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry of `a` is NaN or infinite.
    /// - The errors of the rank policy.
    /// - `ConvergenceFailure` if the nested SVD exceeds the iteration limit.
    /// - `NumericalInstability` if the residual or orthogonality check fails, or if
    ///   rescaling `H` overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn factor_into<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        side: PolarSide,
        options: &PolarOptions<S::Real>,
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
        let request = PolarWorkspaceRequest::new(m, n, side, options).for_kernels(kernels);
        let requirements = Self::workspace_requirements(&request)?;
        let a = a.try_finite_dispatched(kernels.dispatch())?;
        if m.min(n) != 0 {
            let _ = options.rank.cutoff(S::Real::ONE, m, n)?;
        }
        workspace.try_reserve(&requirements)?;
        output.reserve(m, n, side)?;
        let scale = if a.maximum == S::Real::ZERO {
            S::Real::ONE
        } else {
            a.maximum
        };
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("linalg", "polar", 1),
            ((m as u128) << 64) | n as u128,
            0,
        );
        workspace.run_scope(
            context,
            identity,
            options.execution.solve_budgets(),
            |workspace, kernels, solve| {
                workspace.with_prefix(request.prefix()?, |buffers, child| {
                    crate::solve_support::admit_iteration(
                        Some(kernels.execution()),
                        Some(solve),
                        "no complete polar candidate",
                    )?;
                    let (input, rest) = crate::workspace::split_mut(buffers, m * n)?;
                    let (u, h) = crate::workspace::split_mut(rest, m * n)?;
                    let h_order = if side == PolarSide::Right { n } else { m };
                    let mut u = ScratchMatrix::new(u, m, n)?;
                    let mut h = ScratchMatrix::new(h, h_order, h_order)?;
                    let mut input = ScratchMatrix::new(input, m, n)?;
                    let report = if m == 0 || n == 0 {
                        u.as_mut_slice().fill(S::ZERO);
                        h.as_mut_slice().fill(S::ZERO);
                        PolarReport {
                            rank: super::svd::spectrum_rank(&[], m, n, options.rank)?,
                            condition: None,
                            completion_nonunique: false,
                            residual: S::Real::ZERO,
                            orthogonality_error: S::Real::ZERO,
                        }
                    } else {
                        // Components that flush under the maximum-entry scale lie
                        // below the smallest subnormal relative to it, far beneath eps * |A|.
                        for row in 0..m {
                            kernels.execution().check()?;
                            for col in 0..n {
                                let x = a.as_view().get_element(row, col);
                                input.set_element(
                                    row,
                                    col,
                                    S::from_components(
                                        x.real_component() / scale,
                                        x.imaginary_component() / scale,
                                    ),
                                );
                            }
                        }
                        super::svd::thin_candidate(
                            crate::FiniteMatrixView::from_workspace(input.try_view()?),
                            options.execution,
                            child,
                            kernels.scoped(solve),
                            |spectral| {
                                assemble(
                                    &input,
                                    &spectral,
                                    &mut u,
                                    &mut h,
                                    PolarAssembly {
                                        side,
                                        scale,
                                        policy: options.rank,
                                    },
                                    kernels,
                                )
                            },
                        )?
                    };
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.u.try_resize_flat(m, n)?;
                    output.h.try_resize_flat(h_order, h_order)?;
                    output.u.as_mut_slice().copy_from_slice(u.as_slice());
                    output.h.as_mut_slice().copy_from_slice(h.as_slice());
                    output.side = side;
                    output.scale = scale;
                    output.report = Some(report);
                    Ok(())
                })
            },
        )
    }
    fn reserve(&mut self, m: usize, n: usize, side: PolarSide) -> Result<(), SciOxideError> {
        self.u.try_reserve_shape(m, n)?;
        let h = if side == PolarSide::Right { n } else { m };
        self.h.try_reserve_shape(h, h)
    }
}
impl<S: NumericScalar> Default for PolarDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}
/// Workspace request for the polar decomposition of a `rows × cols` matrix.
#[derive(Debug, Clone, Copy)]
pub struct PolarWorkspaceRequest<R: RealScalar> {
    rows: usize,
    cols: usize,
    side: PolarSide,
    options: PolarOptions<R>,
    workers: core::num::NonZeroUsize,
    packing: crate::workspace::Packing,
}
impl<R: RealScalar> PolarWorkspaceRequest<R> {
    /// Returns a request for a `rows × cols` matrix with `H` on `side`.
    pub const fn new(rows: usize, cols: usize, side: PolarSide, options: &PolarOptions<R>) -> Self {
        Self {
            rows,
            cols,
            side,
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
    fn prefix(self) -> Result<usize, SciOxideError> {
        let h = if self.side == PolarSide::Right {
            self.cols
        } else {
            self.rows
        };
        let size = |shape: &[usize]| {
            scioxide_core::backend::checked_product(
                shape,
                scioxide_core::quantity!("polar_scratch"),
            )
        };
        size(&[2, self.rows, self.cols])?
            .checked_add(size(&[h, h])?)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("polar_scratch")))
    }
}
impl<S: NumericScalar> PolarDecomposition<S> {
    /// Returns the workspace size for [`factor_into`](Self::factor_into), including the nested SVD.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the options.
    pub fn workspace_requirements(
        request: &PolarWorkspaceRequest<S::Real>,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        let mut svd =
            crate::SvdWorkspaceRequest::new(request.rows, request.cols, &request.options.execution)
                .with_worker_bound(request.workers);
        if matches!(request.packing, crate::workspace::Packing::Provider) {
            svd = svd.with_provider_packing();
        }
        let mut requirements = ThinSvdDecomposition::<S>::workspace_requirements(&svd)?;
        *requirements.scalar_mut(ScalarRegion::Direct) = requirements
            .scalar(ScalarRegion::Direct)
            .checked_add(request.prefix()?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("polar_scratch"))
            })?;
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
    ///     PolarDecomposition, PolarOptions, PolarSide, PolarWorkspaceRequest, RankPolicy,
    /// };
    /// let options = PolarOptions::new(RankPolicy::Exact);
    /// let request = PolarWorkspaceRequest::new(3, 2, PolarSide::Right, &options);
    /// let workspace = PolarDecomposition::<f64>::workspace(&request)?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(
        request: &PolarWorkspaceRequest<S::Real>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

fn assemble<S: NumericScalar>(
    input: &impl MatrixStorage<S>,
    spectral: &super::svd::solve::SpectralView<'_, S>,
    u: &mut impl MatrixStorage<S>,
    hermitian: &mut impl MatrixStorage<S>,
    options: PolarAssembly<S::Real>,
    context: KernelContext<'_>,
) -> Result<PolarReport<S::Real>, SciOxideError> {
    let PolarAssembly {
        side,
        scale,
        policy,
    } = options;

    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let rank = super::svd::spectrum_rank(spectral.values, m, n, policy)?;
    let condition = if k == 0 {
        None
    } else {
        Some(super::svd::spectrum_condition_estimate(spectral.values)?)
    };
    u.try_resize_flat(m, n)?;
    for i in 0..m {
        context.execution().check()?;
        for j in 0..n {
            u.set_element(
                i,
                j,
                (0..k).fold(S::ZERO, |sum, l| {
                    sum + spectral.u.get_element(i, l) * spectral.vt.get_element(l, j)
                }),
            );
        }
    }
    let h = if side == PolarSide::Right { n } else { m };
    hermitian.try_resize_flat(h, h)?;
    for i in 0..h {
        context.execution().check()?;
        for j in i..h {
            let x = (0..k).fold(S::ZERO, |sum, l| {
                let (u, v) = if side == PolarSide::Right {
                    (
                        spectral.vt.get_element(l, i).conj(),
                        spectral.vt.get_element(l, j),
                    )
                } else {
                    (
                        spectral.u.get_element(i, l),
                        spectral.u.get_element(j, l).conj(),
                    )
                };
                sum + u * S::from_real(spectral.values[l]) * v
            });
            // Only the upper triangle is computed and mirrored, and the
            // diagonal is forced real, so H is exactly Hermitian.
            let value = S::from_components(
                x.real_component(),
                if i == j {
                    S::Real::ZERO
                } else {
                    x.imaginary_component()
                },
            );
            hermitian.set_element(i, j, value);
            hermitian.set_element(j, i, value.conj());
        }
    }
    // Publication gate: the reconstruction residual (in normalized units)
    // and `||U^H U - I||_F` must both be within a `128 n eps` allowance.
    // When A is rank deficient, U's completion on the null space is not
    // unique; that is reported rather than treated as an error.
    let residual = crate::kernel::standard::try_norm((0..m).map(|i| {
        context.execution().check()?;
        crate::kernel::standard::norm_workspace((0..n).map(|j| {
            let value = (0..h).fold(S::ZERO, |sum, l| {
                let (u, h) = if side == PolarSide::Right {
                    (u.get_element(i, l), hermitian.get_element(l, j))
                } else {
                    (u.get_element(l, j), hermitian.get_element(i, l))
                };
                sum + u * h
            });
            input.get_element(i, j) - value
        }))
    }))?;
    let orthogonality_error = orthogonality(u.try_view()?, context)?;
    let eps = crate::checked::dimension_epsilon::<S::Real>(m.max(n).max(1).saturating_mul(128))?;
    let norm = crate::kernel::finite::norm(input.as_slice())?;
    if residual > eps * norm || orthogonality_error > eps {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "polar publication residual exceeds roundoff allowance",
        ));
    }
    // Restore input units only after the normalized gate; gradual underflow is
    // ordinary rounding of H, and overflow rejects the candidate.
    for value in hermitian.as_mut_slice() {
        *value = crate::checked::finite_candidate(S::from_components(
            value.real_component() * scale,
            value.imaginary_component() * scale,
        ))?;
    }
    Ok(PolarReport {
        rank,
        condition,
        completion_nonunique: rank.rank() < k,
        residual,
        orthogonality_error,
    })
}
