//! Column-space and null-space bases, principal angles and orthogonal Procrustes, all via the SVD.
//!
//! Output shapes depend on the numerical rank, so outputs grow to their largest
//! possible shape before any numerical work and are written only after every
//! stage has succeeded. Every SVD stage runs in a child scope of the operation
//! and counts as one iteration of the operation's own scope.
//!
//! Cross products, alignments and residuals are Standard dense products: unit
//! frame, with binary-scaled retry only for entries that overflow; final
//! gradual underflow is ordinary rounding.

use crate::decompose::{
    SpectralView, full_candidate, spectrum_condition_estimate, spectrum_rank, thin_candidate,
    values_candidate,
};
use crate::kernel::{GemmConfig, GemmOp, WorkspaceGemmProblem, gemm_workspace_with_context};
use crate::workspace::Packing;
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    Accuracy, ConditionEstimate, DenseWorkspaceRequest, FiniteMatrixView, KernelContext,
    LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace, Matrix,
    MatrixView, MatrixViewMut, RankPolicy, RankReport, SvdDecomposition, SvdValues,
    SvdWorkspaceRequest, ThinSvdDecomposition,
};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::backend::{checked_mul, try_reserve};
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, ExecutionPolicy, NumericScalar, Quantity, RealScalar,
    Scaled, ScaledAccumulator, ScaledNorm, SciOxideError, SolveContext,
};

/// Rows of `A` per Procrustes residual product, bounding its scratch.
const RESIDUAL_ROWS: usize = 256;

/// Numerical rank and condition number of the matrix whose subspace was computed.
#[derive(Debug, Clone, Copy)]
pub struct SubspaceReport<R: RealScalar> {
    /// Numerical rank under the rank policy.
    pub(crate) rank: RankReport<R>,
    /// Raw spectrum condition estimate; absent for an empty spectrum.
    pub(crate) condition: Option<ConditionEstimate<R>>,
}

impl<R: RealScalar> SubspaceReport<R> {
    /// Returns the numerical rank under the rank policy.
    pub const fn rank(&self) -> RankReport<R> {
        self.rank
    }

    /// Returns the reciprocal condition number `σ_min / σ_max`, or `None` without singular values.
    ///
    /// No rank threshold is applied.
    pub const fn condition(&self) -> Option<ConditionEstimate<R>> {
        self.condition
    }
}

/// Orthonormal basis of a subspace, returned by [`orth`] and [`null_space`], with its rank report.
///
/// The basis vectors are the columns of [`basis`](Self::basis). The subspace is determined
/// by the rank, but the basis vectors are not unique.
#[derive(Debug, Clone)]
pub struct SubspaceBasis<S: NumericScalar> {
    /// Basis vectors stored as columns.
    pub(crate) basis: Matrix<S>,
    /// Numerical interpretation of the original matrix.
    pub(crate) report: SubspaceReport<S::Real>,
}

impl<S: NumericScalar> SubspaceBasis<S> {
    /// Returns the basis and the report.
    pub fn into_parts(self) -> (Matrix<S>, SubspaceReport<S::Real>) {
        (self.basis, self.report)
    }
    /// Returns the basis, one vector per column.
    pub const fn basis(&self) -> &Matrix<S> {
        &self.basis
    }

    /// Returns the rank and condition number of the input matrix.
    pub const fn report(&self) -> SubspaceReport<S::Real> {
        self.report
    }
}

/// Subspace job, input shape and SVD options.
///
/// Requirements cover every nested SVD stage and Standard product of the job
/// at its largest possible rank, plus the staged results, so any numerical
/// rank fits.
#[derive(Debug, Clone, Copy)]
struct SubspaceRequest {
    job: SubspaceJob,
    options: LinalgExecutionOptions,
    workers: NonZeroUsize,
    packing: Packing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubspaceJob {
    Orth {
        rows: usize,
        cols: usize,
    },
    NullSpace {
        rows: usize,
        cols: usize,
    },
    Angles {
        rows: usize,
        first: usize,
        second: usize,
    },
    Procrustes {
        rows: usize,
        cols: usize,
    },
}

impl SubspaceRequest {
    const fn new(job: SubspaceJob, options: &LinalgExecutionOptions) -> Self {
        Self {
            job,
            options: *options,
            workers: NonZeroUsize::MAX,
            packing: Packing::Native,
        }
    }

    /// Declare a column-space basis of a `rows × cols` input ([`orth_into`]).
    const fn orth(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(SubspaceJob::Orth { rows, cols }, options)
    }

    /// Declare a complete right null-space basis of a `rows × cols` input
    /// ([`null_space_into`]).
    const fn null_space(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(SubspaceJob::NullSpace { rows, cols }, options)
    }

    /// Declare principal angles between `rows × first_cols` and
    /// `rows × second_cols` inputs ([`subspace_angles_into`]).
    const fn angles(
        rows: usize,
        first_cols: usize,
        second_cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self::new(
            SubspaceJob::Angles {
                rows,
                first: first_cols,
                second: second_cols,
            },
            options,
        )
    }

    /// Declare a Procrustes alignment of two `rows × cols` inputs
    /// ([`orthogonal_procrustes_into`]). Staged scratch is quadratic in `cols`;
    /// the residual is formed in bounded row blocks.
    const fn procrustes(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(SubspaceJob::Procrustes { rows, cols }, options)
    }

    fn for_kernels(mut self, context: KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self.packing = Packing::Provider;
        }
        self
    }

    fn svd(self, rows: usize, cols: usize) -> SvdWorkspaceRequest {
        let request =
            SvdWorkspaceRequest::new(rows, cols, &self.options).with_worker_bound(self.workers);
        match self.packing {
            Packing::Provider => request.with_provider_packing(),
            Packing::Native => request,
        }
    }

    // Standard `rows × shared` by `shared × cols` product scratch.
    fn product<S: NumericScalar>(
        self,
        rows: usize,
        shared: usize,
        cols: usize,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let request = DenseWorkspaceRequest::product(rows, shared, cols, &self.options)
            .with_product_accuracy(Accuracy::Standard)
            .with_worker_bound(self.workers);
        crate::kernel::product_requirements::<S>(&match self.packing {
            Packing::Provider => request.with_provider_packing(),
            Packing::Native => request,
        })
    }

    fn requirements<S: NumericScalar>(self) -> Result<LinalgRequirements, SciOxideError> {
        let thin =
            |rows, cols| ThinSvdDecomposition::<S>::workspace_requirements(&self.svd(rows, cols));
        let nested = match self.job {
            SubspaceJob::Orth { rows, cols } => thin(rows, cols)?,
            // A thin SVD of a tall or square input already has the complete V.
            SubspaceJob::NullSpace { rows, cols } if rows >= cols => thin(rows, cols)?,
            SubspaceJob::NullSpace { rows, cols } => {
                SvdDecomposition::<S>::workspace_requirements(&self.svd(rows, cols))?
            }
            SubspaceJob::Angles {
                rows,
                first,
                second,
            } => {
                // The cross product of the selected bases is at most
                // `large × small` and its complementary residual at most
                // `rows × small`; smaller ranks need less.
                let (ka, kb) = (rows.min(first), rows.min(second));
                let (large, small) = (ka.max(kb), ka.min(kb));
                thin(rows, first)?
                    .max(&thin(rows, second)?)
                    .max(&thin(large, small)?)
                    .max(&SvdValues::<S::Real>::workspace_requirements::<S>(
                        &self.svd(rows, small),
                    )?)
                    .max(&self.product::<S>(large, rows, small)?)
                    .max(&self.product::<S>(rows, large, small)?)
            }
            SubspaceJob::Procrustes { rows, cols } => thin(cols, cols)?
                .max(&self.product::<S>(cols, rows, cols)?)
                .max(&self.product::<S>(cols, cols, cols)?)
                .max(&self.product::<S>(RESIDUAL_ROWS.min(rows), cols, cols)?),
        };
        let stage = self.job.stage()?;
        let add = |a: usize, b: usize| {
            a.checked_add(b).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("subspace_scratch"))
            })
        };
        let mut requirements = nested;
        *requirements.scalar_mut(ScalarRegion::Direct) =
            add(requirements.scalar(ScalarRegion::Direct), stage.scalars)?;
        *requirements.real_mut(RealRegion::Direct) =
            add(requirements.real(RealRegion::Direct), stage.reals)?;
        requirements.byte_limit = self.options.byte_limit();
        requirements.required_bytes::<S>()?;
        Ok(requirements)
    }
}

/// Workspace request for [`orth_into`] or [`null_space_into`].
///
/// Pass it to [`LinalgWorkspace::workspace_requirements`] or [`LinalgWorkspace::workspace`].
/// The size covers the largest possible rank, so one workspace fits every input of the
/// declared shape.
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{LinalgExecutionOptions, LinalgWorkspace, SubspaceBasisWorkspaceRequest};
/// let request = SubspaceBasisWorkspaceRequest::orth(5, 3, &LinalgExecutionOptions::new());
/// let requirements = LinalgWorkspace::<f64>::workspace_requirements(&request)?;
/// let mut workspace = LinalgWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SubspaceBasisWorkspaceRequest(SubspaceRequest);
impl SubspaceBasisWorkspaceRequest {
    /// Returns a request for [`orth_into`] on a `rows × cols` matrix.
    pub const fn orth(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self(SubspaceRequest::orth(rows, cols, options))
    }
    /// Returns a request for [`null_space_into`] on a `rows × cols` matrix.
    pub const fn null_space(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self(SubspaceRequest::null_space(rows, cols, options))
    }
    /// Sizes the workspace for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.0.workers = workers;
        self
    }
    /// Adds packing space for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.0.packing = Packing::Provider;
        self
    }
    /// Takes the worker bound and BLAS provider from `context`.
    pub fn for_context(mut self, context: &LinalgExecutionContext<'_, '_>) -> Self {
        self.0 = self.0.for_kernels(context.kernel());
        self
    }
}
impl<S: NumericScalar> crate::workspace::Request<S> for SubspaceBasisWorkspaceRequest {
    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        self.0.requirements::<S>()
    }
}

/// Workspace request for [`subspace_angles_into`].
///
/// Pass it to [`LinalgWorkspace::workspace_requirements`] or [`LinalgWorkspace::workspace`].
#[derive(Debug, Clone, Copy)]
pub struct SubspaceAnglesWorkspaceRequest(SubspaceRequest);
impl SubspaceAnglesWorkspaceRequest {
    /// Returns a request for the angles between the column spaces of a `rows × first_cols`
    /// and a `rows × second_cols` matrix.
    pub const fn new(
        rows: usize,
        first_cols: usize,
        second_cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self(SubspaceRequest::angles(
            rows,
            first_cols,
            second_cols,
            options,
        ))
    }
    /// Sizes the workspace for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.0.workers = workers;
        self
    }
    /// Adds packing space for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.0.packing = Packing::Provider;
        self
    }
    /// Takes the worker bound and BLAS provider from `context`.
    pub fn for_context(mut self, context: &LinalgExecutionContext<'_, '_>) -> Self {
        self.0 = self.0.for_kernels(context.kernel());
        self
    }
}
impl<S: NumericScalar> crate::workspace::Request<S> for SubspaceAnglesWorkspaceRequest {
    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        self.0.requirements::<S>()
    }
}

/// Workspace request for [`orthogonal_procrustes_into`].
///
/// Pass it to [`LinalgWorkspace::workspace_requirements`] or [`LinalgWorkspace::workspace`].
/// The size is `O(cols²)`; the residual is formed in blocks of at most 256 rows.
#[derive(Debug, Clone, Copy)]
pub struct ProcrustesWorkspaceRequest(SubspaceRequest);
impl ProcrustesWorkspaceRequest {
    /// Returns a request for two `rows × cols` matrices.
    pub const fn new(rows: usize, cols: usize, options: &LinalgExecutionOptions) -> Self {
        Self(SubspaceRequest::procrustes(rows, cols, options))
    }
    /// Sizes the workspace for at most `workers` parallel workers.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.0.workers = workers;
        self
    }
    /// Adds packing space for an injected BLAS provider.
    pub const fn with_provider_packing(mut self) -> Self {
        self.0.packing = Packing::Provider;
        self
    }
    /// Takes the worker bound and BLAS provider from `context`.
    pub fn for_context(mut self, context: &LinalgExecutionContext<'_, '_>) -> Self {
        self.0 = self.0.for_kernels(context.kernel());
        self
    }
}
impl<S: NumericScalar> crate::workspace::Request<S> for ProcrustesWorkspaceRequest {
    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        self.0.requirements::<S>()
    }
}

/// Candidate storage lent to the subspace stage ahead of nested SVD scratch.
#[derive(Debug, Clone, Copy)]
struct Stage {
    scalars: usize,
    reals: usize,
}

impl SubspaceJob {
    fn stage(self) -> Result<Stage, SciOxideError> {
        let quantity = scioxide_core::quantity!("subspace_scratch");
        let mul = |a: usize, b: usize| checked_mul(a, b, quantity);
        let sum = |parts: &[usize]| {
            parts.iter().try_fold(0usize, |total, &part| {
                total
                    .checked_add(part)
                    .ok_or_else(|| SciOxideError::size_overflow(quantity))
            })
        };
        Ok(match self {
            Self::Orth { rows, cols } => Stage {
                scalars: mul(rows, rows.min(cols))?,
                reals: 0,
            },
            Self::NullSpace { cols, .. } => Stage {
                scalars: mul(cols, cols)?,
                reals: 0,
            },
            Self::Angles {
                rows,
                first,
                second,
            } => {
                let (ka, kb) = (rows.min(first), rows.min(second));
                let k = ka.min(kb);
                Stage {
                    scalars: sum(&[mul(rows, ka)?, mul(rows, kb)?, mul(ka, kb)?, mul(rows, k)?])?,
                    reals: mul(2, k)?,
                }
            }
            // Cross product (later the alignment), both singular-vector
            // factors and one residual row block.
            Self::Procrustes { rows, cols } => Stage {
                scalars: sum(&[
                    mul(3, mul(cols, cols)?)?,
                    mul(RESIDUAL_ROWS.min(rows), cols)?,
                ])?,
                reals: 0,
            },
        })
    }
}

// Rank and conditioning of a candidate spectrum of an original `rows × cols` input.
fn spectral_report<R: RealScalar>(
    values: &[R],
    rows: usize,
    cols: usize,
    policy: RankPolicy<R>,
) -> Result<SubspaceReport<R>, SciOxideError> {
    Ok(SubspaceReport {
        rank: spectrum_rank(values, rows, cols, policy)?,
        condition: if values.is_empty() {
            None
        } else {
            Some(spectrum_condition_estimate(values)?)
        },
    })
}

fn identity(
    method: scioxide_core::AlgorithmId,
    rows: usize,
    cols: usize,
) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        method,
        ((rows as u128) << 64) | cols as u128,
        0,
    )
}

// One iteration of the operation's own epoch admits each nested SVD stage.
fn admit_stage(
    kernels: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    crate::solve_support::admit_iteration(
        Some(kernels.execution()),
        Some(solve),
        "subspace factorization-stage budget exhausted",
    )
}

// Staged values are copied only after every fallible stage has succeeded;
// capacity was reserved before numerical work, so this cannot allocate.
fn publish<S: NumericScalar>(
    output: &mut Matrix<S>,
    rows: usize,
    cols: usize,
    values: &[S],
) -> Result<(), SciOxideError> {
    output.try_resize_flat(rows, cols)?;
    output.as_mut_slice().copy_from_slice(values);
    Ok(())
}

fn reserve_vec<T>(
    values: &mut Vec<T>,
    len: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.capacity() < len {
        try_reserve(values, len.saturating_sub(values.len()), quantity)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Basis {
    Range,
    Null,
}

/// Returns an orthonormal basis of the numerical column space (range) of `a`.
///
/// The basis is the first `r` left singular vectors `u₁, …, uᵣ` of a thin SVD of the
/// `m × n` matrix `a`, where `r` is the numerical rank under `policy`. The result is
/// `m × r`.
///
/// # Errors
///
/// The errors of [`orth_into`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RankPolicy, orth};
/// let a = Matrix::from_vec(3, 2, vec![1.0_f64, 2.0, 2.0, 4.0, 0.0, 0.0])?;
/// let range = orth(&a, RankPolicy::DimensionScaledEpsilon)?;
/// // The second column is twice the first, so the range has dimension one.
/// assert_eq!(range.basis().cols(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn orth<'m, S: NumericScalar>(
    a: impl Into<MatrixView<'m, S>>,
    policy: RankPolicy<S::Real>,
) -> Result<SubspaceBasis<S>, SciOxideError> {
    let mut basis = Matrix::empty();
    let report = orth_into(
        a,
        policy,
        &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut basis,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    Ok(SubspaceBasis { basis, report })
}

/// Writes an orthonormal basis of the numerical column space of `a` to `output`.
///
/// Returns the rank report. The basis is as in [`orth`]; `output` becomes `m × r`. Its storage grows to
/// `m × min(m, n)` before the computation. Size the workspace with
/// [`SubspaceBasisWorkspaceRequest::orth`]. The SVD counts toward the iteration limit of
/// `options` in a child scope.
///
/// # Errors
///
/// - `NonFinite` if an entry of `a` is NaN or infinite.
/// - The errors of `policy`.
/// - `ConvergenceFailure` if the SVD exceeds the iteration limit.
/// - `NumericalInstability` if a singular value overflows.
/// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
pub fn orth_into<'m, 'scope, 'capability, S: NumericScalar>(
    a: impl Into<MatrixView<'m, S>>,
    policy: RankPolicy<S::Real>,
    options: &LinalgExecutionOptions,
    output: &mut Matrix<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<SubspaceReport<S::Real>, SciOxideError> {
    basis_into(
        a.into(),
        policy,
        options,
        output,
        workspace,
        context.into(),
        Basis::Range,
    )
}

/// Returns an orthonormal basis of the numerical null space of `a`.
///
/// The basis is the last `n − r` right singular vectors `vᵣ₊₁, …, vₙ` of the `m × n`
/// matrix `a`, where `r` is the numerical rank under `policy`. The `n × (n − r)` result `Z`
/// satisfies `‖A Z‖₂ = σᵣ₊₁`, or `A Z = 0` up to rounding if `r = m`. For `m < n` this includes the `n − m` vectors that a thin
/// SVD does not compute. For a basis of the left null space (`Aᴴ Z ≈ 0`), pass
/// `a.adjoint()`.
///
/// # Errors
///
/// The errors of [`null_space_into`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RankPolicy, null_space};
/// let a = Matrix::from_vec(1, 3, vec![1.0_f64, 1.0, 1.0])?;
/// let kernel = null_space(&a, RankPolicy::DimensionScaledEpsilon)?;
/// // x₀ + x₁ + x₂ = 0 defines a plane in ℝ³.
/// assert_eq!((kernel.basis().rows(), kernel.basis().cols()), (3, 2));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn null_space<'m, S: NumericScalar>(
    a: impl Into<MatrixView<'m, S>>,
    policy: RankPolicy<S::Real>,
) -> Result<SubspaceBasis<S>, SciOxideError> {
    let mut basis = Matrix::empty();
    let report = null_space_into(
        a,
        policy,
        &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut basis,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    Ok(SubspaceBasis { basis, report })
}

/// Writes an orthonormal basis of the numerical null space of `a` to `output`.
///
/// Returns the rank report. The basis is as in [`null_space`]; `output` becomes `n × (n − r)`. It comes from a thin
/// SVD when `m ≥ n` and from a full SVD when `m < n`. The storage of `output` grows to
/// `n × n` before the computation. Size the workspace with
/// [`SubspaceBasisWorkspaceRequest::null_space`].
///
/// # Errors
///
/// The errors of [`orth_into`].
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
pub fn null_space_into<'m, 'scope, 'capability, S: NumericScalar>(
    a: impl Into<MatrixView<'m, S>>,
    policy: RankPolicy<S::Real>,
    options: &LinalgExecutionOptions,
    output: &mut Matrix<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<SubspaceReport<S::Real>, SciOxideError> {
    basis_into(
        a.into(),
        policy,
        options,
        output,
        workspace,
        context.into(),
        Basis::Null,
    )
}

fn basis_into<S: NumericScalar>(
    a: MatrixView<'_, S>,
    policy: RankPolicy<S::Real>,
    options: &LinalgExecutionOptions,
    output: &mut Matrix<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: LinalgExecutionContext<'_, '_>,
    basis: Basis,
) -> Result<SubspaceReport<S::Real>, SciOxideError> {
    workspace.begin_attempt();
    let scope = context.with_options(options)?;
    let kernels = scope.kernel();
    kernels.execution().check()?;
    let (rows, cols) = (a.rows(), a.cols());
    let (request, method, basis_rows, max_cols) = match basis {
        Basis::Range => (
            SubspaceRequest::orth(rows, cols, options),
            scioxide_core::algorithm_id!("linalg", "orth", 1),
            rows,
            rows.min(cols),
        ),
        Basis::Null => (
            SubspaceRequest::null_space(rows, cols, options),
            scioxide_core::algorithm_id!("linalg", "null_space", 1),
            cols,
            cols,
        ),
    };
    let request = request.for_kernels(kernels);
    let requirements = request.requirements::<S>()?;
    let a = a.try_finite_dispatched(kernels.dispatch())?;
    // Validate policy even when an empty matrix needs no numerical factorization.
    policy.cutoff(S::Real::ZERO, rows, cols)?;
    workspace.try_reserve(&requirements)?;
    output.try_reserve_shape(basis_rows, max_cols)?;
    let stage = request.job.stage()?;
    workspace.run_scope(
        scope,
        identity(method, rows, cols),
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, kernels, solve| {
            workspace.with_prefixes([stage.scalars, 0, 0], |staged, _, _, child| {
                // Column space: the leading `rank` left singular vectors of a
                // thin SVD. Null space: the trailing right singular vectors. A
                // thin SVD of a tall or square matrix has the complete V; a
                // wide matrix needs the full SVD's structural complement.
                let (report, basis_cols) = match basis {
                    Basis::Range => {
                        let report = RangeStage {
                            policy,
                            options: *options,
                            kernels,
                        }
                        .stage(a, staged, child, solve)?;
                        (report, report.rank.rank())
                    }
                    Basis::Null => {
                        admit_stage(kernels, solve)?;
                        let null = |spectral: SpectralView<'_, S>| {
                            let report = spectral_report(spectral.values, rows, cols, policy)?;
                            let rank = report.rank.rank();
                            let nullity = cols - rank;
                            for (index, slot) in staged[..cols * nullity].iter_mut().enumerate() {
                                *slot = spectral
                                    .vt
                                    .get_element(rank + index % nullity, index / nullity)
                                    .conj();
                            }
                            Ok((report, nullity))
                        };
                        let context = kernels.scoped(solve);
                        if rows >= cols {
                            thin_candidate(a, *options, child, context, null)?
                        } else {
                            full_candidate(a, *options, child, context, null)?
                        }
                    }
                };
                kernels.execution().check()?;
                let _ = solve.counts().total()?;
                publish(
                    output,
                    basis_rows,
                    basis_cols,
                    &staged[..basis_rows * basis_cols],
                )?;
                Ok(report)
            })
        },
    )
}

/// Rank reports of the two inputs of [`subspace_angles`].
#[derive(Debug, Clone, Copy)]
pub struct SubspaceAnglesReport<R: RealScalar> {
    /// First input interpretation.
    pub(crate) first: SubspaceReport<R>,
    /// Second input interpretation.
    pub(crate) second: SubspaceReport<R>,
}

impl<R: RealScalar> SubspaceAnglesReport<R> {
    /// Returns the rank report of `A`.
    pub const fn first(&self) -> SubspaceReport<R> {
        self.first
    }

    /// Returns the rank report of `B`.
    pub const fn second(&self) -> SubspaceReport<R> {
        self.second
    }
}

/// Principal angles between two column spaces, in radians, returned by [`subspace_angles`].
///
/// The angles are in descending order. The scalar parameter is that of the inputs; the
/// angles are real.
#[derive(Debug, Clone)]
pub struct SubspaceAngles<S: NumericScalar> {
    /// `min(rank(A), rank(B))` values in `[0, π/2]`; empty if either rank is zero.
    pub(crate) angles: Vec<S::Real>,
    /// Input rank and threshold-independent conditioning.
    pub(crate) report: SubspaceAnglesReport<S::Real>,
}

impl<S: NumericScalar> SubspaceAngles<S> {
    /// Returns the angles and the report.
    pub fn into_parts(self) -> (Vec<S::Real>, SubspaceAnglesReport<S::Real>) {
        (self.angles, self.report)
    }

    /// Returns the `min(rank(A), rank(B))` angles in `[0, π/2]`, largest first.
    ///
    /// The slice is empty if either rank is zero.
    pub fn angles(&self) -> &[S::Real] {
        &self.angles
    }

    /// Returns the rank reports of the two inputs.
    pub const fn report(&self) -> SubspaceAnglesReport<S::Real> {
        self.report
    }
}

/// Whether the minimizer of the orthogonal Procrustes problem is unique.
///
/// The minimizer `Q = U Vᴴ` is unique if and only if `Aᴴ B` is nonsingular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcrustesUniqueness {
    /// No singular value of `Aᴴ B` is zero, and all are above the rank threshold.
    Unique,
    /// No singular value of `Aᴴ B` is zero, but at least one is below the rank threshold.
    NumericallyUnresolved,
    /// A computed singular value of `Aᴴ B` is exactly zero, so the minimizer is not unique.
    Nonunique,
}
/// Diagnostics of the orthogonal Procrustes problem `min ‖A Q − B‖_F` subject to `Qᴴ Q = I`.
#[derive(Debug, Clone, Copy)]
pub struct ProcrustesReport<R: RealScalar> {
    /// Sum of cross-product singular values, in original input units.
    /// This is an objective diagnostic, not an applied scale factor.
    pub(crate) scale: R,
    /// Independently evaluated Frobenius residual in original input units.
    pub(crate) residual_norm: R,
    /// Cross-product rank and raw conditioning, without truncating the solution.
    pub(crate) cross_product: SubspaceReport<R>,
    /// Structural versus numerical ambiguity of the computed minimizer.
    pub(crate) uniqueness: ProcrustesUniqueness,
}

impl<R: RealScalar> ProcrustesReport<R> {
    /// Returns `Σᵢ σᵢ(Aᴴ B)`, the maximum of `Re tr(Qᴴ Aᴴ B)` over unitary `Q`.
    ///
    /// This is a diagnostic of the objective, not a scale factor applied to the result.
    pub const fn scale(&self) -> R {
        self.scale
    }

    /// Returns the residual `‖A Q − B‖_F`, evaluated from `A`, `B` and the computed `Q`.
    pub const fn residual_norm(&self) -> R {
        self.residual_norm
    }

    /// Returns the rank and condition number of `Aᴴ B`; the rank does not truncate `Q`.
    pub const fn cross_product(&self) -> SubspaceReport<R> {
        self.cross_product
    }

    /// Returns whether the minimizer is unique.
    pub const fn uniqueness(&self) -> ProcrustesUniqueness {
        self.uniqueness
    }
}

/// Solution `Q` of the orthogonal Procrustes problem, returned by [`orthogonal_procrustes`].
///
/// `Q` is orthogonal for real and unitary for complex inputs. It acts from the right:
/// `A Q` is the best alignment of `A` to `B`.
#[derive(Debug, Clone)]
pub struct ProcrustesAlignment<S: NumericScalar> {
    /// Right-acting minimizer: aligned data are `A * transformation`.
    pub(crate) transformation: Matrix<S>,
    /// Objective, conditioning and uniqueness diagnostics.
    pub(crate) report: ProcrustesReport<S::Real>,
}

impl<S: NumericScalar> ProcrustesAlignment<S> {
    /// Returns `Q` and the report.
    pub fn into_parts(self) -> (Matrix<S>, ProcrustesReport<S::Real>) {
        (self.transformation, self.report)
    }

    /// Returns the `n × n` minimizer `Q`; the aligned data are `A Q`.
    pub const fn transformation(&self) -> &Matrix<S> {
        &self.transformation
    }

    /// Returns the diagnostics.
    pub const fn report(&self) -> ProcrustesReport<S::Real> {
        self.report
    }
}

fn matching_rows<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    same_cols: bool,
) -> Result<(), SciOxideError> {
    if a.rows() != b.rows() || (same_cols && a.cols() != b.cols()) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(a.rows(), a.cols()),
            scioxide_core::Dimensions::two(b.rows(), b.cols()),
        ));
    }
    Ok(())
}
// One Standard product `output := alpha op(left) right + beta output` over
// reserved workspace; views own conjugation semantics.
fn standard_product<S: NumericScalar>(
    left: FiniteMatrixView<'_, S>,
    right: FiniteMatrixView<'_, S>,
    output: MatrixViewMut<'_, S>,
    config: GemmConfig<S>,
    options: &LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    kernels: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    gemm_workspace_with_context(
        WorkspaceGemmProblem {
            left,
            right,
            output,
            config,
        },
        options.execution_policy(),
        workspace,
        kernels.with_accuracy(Accuracy::Standard),
    )
}

// Staged row-major workspace values as a finite operand.
fn staged<S: NumericScalar>(
    rows: usize,
    cols: usize,
    values: &[S],
) -> Result<FiniteMatrixView<'_, S>, SciOxideError> {
    Ok(FiniteMatrixView::from_workspace(MatrixView::from_slice(
        rows, cols, values,
    )?))
}
// One input-range stage of the principal-angle computation.
struct RangeStage<'capability, R: RealScalar> {
    policy: RankPolicy<R>,
    options: LinalgExecutionOptions,
    kernels: KernelContext<'capability>,
}

impl<'capability, R: RealScalar> RangeStage<'capability, R> {
    // Stage the leading `rank` left singular vectors of `input` row-major.
    fn stage<S: NumericScalar<Real = R>>(
        &self,
        input: FiniteMatrixView<'_, S>,
        staged: &mut [S],
        child: &mut LinalgWorkspace<'_, S>,
        solve: &mut SolveContext<'_, 'capability>,
    ) -> Result<SubspaceReport<R>, SciOxideError> {
        admit_stage(self.kernels, solve)?;
        thin_candidate(
            input,
            self.options,
            child,
            self.kernels.scoped(solve),
            |spectral| {
                let report =
                    spectral_report(spectral.values, input.rows(), input.cols(), self.policy)?;
                let rank = report.rank.rank();
                crate::decompose::util::copy_rows(
                    spectral.u.block(0, 0, input.rows(), rank)?,
                    staged,
                );
                Ok(report)
            },
        )
    }
}

/// Returns the principal angles between the numerical column spaces of `a` and `b`.
///
/// Let `Q_A` (`m × p`) and `Q_B` (`m × q`) be orthonormal bases of the column spaces, with
/// the ranks `p` and `q` chosen by `policy`. The `k = min(p, q)` principal angles
/// `π/2 ≥ θ₁ ≥ … ≥ θₖ ≥ 0` have as cosines the singular values of `Q_Aᴴ Q_B`.
///
/// Small angles are inaccurate when computed from their cosines. When some angle is below
/// `π/4`, the sines are computed as the singular values of `(I − Q_L Q_Lᴴ) Q_S`, where `Q_L`
/// is the larger basis and `Q_S` the smaller. Each angle uses its sine if it is below `π/4`
/// and its cosine otherwise. Up to four SVDs run.
///
/// # Errors
///
/// The errors of [`subspace_angles_into`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RankPolicy, subspace_angles};
/// let a = Matrix::from_vec(2, 1, vec![1.0_f64, 0.0])?;
/// let b = Matrix::from_vec(2, 1, vec![1.0_f64, 1.0])?;
/// let result = subspace_angles(&a, &b, RankPolicy::DimensionScaledEpsilon)?;
/// // The vectors (1, 0) and (1, 1) are π/4 apart.
/// assert!((result.angles()[0] - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Å. Björck and G. H. Golub, "Numerical methods for computing angles between linear
///   subspaces", *Math. Comp.* 27(123), 1973.
/// - A. V. Knyazev and M. E. Argentati, "Principal angles between subspaces in an A-based
///   scalar product: algorithms and perturbation estimates", *SIAM J. Sci. Comput.*
///   23(6), 2002.
pub fn subspace_angles<'a, 'b, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<MatrixView<'b, S>>,
    policy: RankPolicy<S::Real>,
) -> Result<SubspaceAngles<S>, SciOxideError> {
    let mut angles = Vec::new();
    let report = subspace_angles_into(
        a,
        b,
        policy,
        &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut angles,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    Ok(SubspaceAngles { angles, report })
}

/// Writes the principal angles of [`subspace_angles`] to `output` and returns the rank reports.
///
/// The storage of `output` grows to `min(m, n_A, n_B)` values before the computation, and
/// `output` then holds the `min(rank(A), rank(B))` angles. Size the workspace with
/// [`SubspaceAnglesWorkspaceRequest::new`]. Each SVD counts toward the iteration limit of
/// `options` in a child scope.
///
/// # Errors
///
/// - `DimensionMismatch` if `a` and `b` have different row counts.
/// - `NonFinite` if an entry of `a` or `b` is NaN or infinite.
/// - The errors of `policy`.
/// - `ConvergenceFailure` if an SVD exceeds the iteration limit.
/// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::*;
/// let a = Matrix::<f64>::identity(2)?;
/// let opts = LinalgExecutionOptions::new();
/// let (mut out, mut work) = (Vec::new(), LinalgWorkspace::new());
/// let cx = ExecutionContext::sequential_native();
/// subspace_angles_into(&a, &a, RankPolicy::Exact, &opts, &mut out, &mut work, cx)?;
/// assert!(out.iter().all(|angle| angle.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn subspace_angles_into<'a, 'b, 'scope, 'capability, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<MatrixView<'b, S>>,
    policy: RankPolicy<S::Real>,
    options: &LinalgExecutionOptions,
    output: &mut Vec<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<SubspaceAnglesReport<S::Real>, SciOxideError> {
    workspace.begin_attempt();
    let scope = context.into().with_options(options)?;
    let kernels = scope.kernel();
    kernels.execution().check()?;
    let (a, b) = (a.into(), b.into());
    matching_rows(a, b, false)?;
    let (rows, first, second) = (a.rows(), a.cols(), b.cols());
    let request = SubspaceRequest::angles(rows, first, second, options).for_kernels(kernels);
    let requirements = request.requirements::<S>()?;
    let a = a.try_finite_dispatched(kernels.dispatch())?;
    let b = b.try_finite_dispatched(kernels.dispatch())?;
    policy.cutoff(S::Real::ZERO, rows, first.max(second))?;
    workspace.try_reserve(&requirements)?;
    let (ka, kb) = (rows.min(first), rows.min(second));
    reserve_vec(output, ka.min(kb), scioxide_core::quantity!("output"))?;
    let stage = request.job.stage()?;
    workspace.run_scope(
        scope,
        identity(
            scioxide_core::algorithm_id!("linalg", "subspace_angles", 1),
            rows,
            first.saturating_add(second),
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, kernels, solve| {
            workspace.with_prefixes(
                [stage.scalars, stage.reals, 0],
                |scalars, reals, _, child| {
                    let (qa, rest) = crate::workspace::split_mut(scalars, rows * ka)?;
                    let (qb, rest) = crate::workspace::split_mut(rest, rows * kb)?;
                    let (cross, residual) = crate::workspace::split_mut(rest, ka * kb)?;
                    let (cosines, angles) = crate::workspace::split_mut(reals, ka.min(kb))?;
                    let range = RangeStage {
                        policy,
                        options: *options,
                        kernels,
                    };
                    let report = SubspaceAnglesReport {
                        first: range.stage(a, qa, child, solve)?,
                        second: range.stage(b, qb, child, solve)?,
                    };
                    let p = report.first.rank.rank();
                    let q = report.second.rank.rank();
                    let k = p.min(q);
                    if k != 0 {
                        let order = p.max(q);
                        let (large, small) = if p >= q {
                            (&qa[..rows * p], &qb[..rows * q])
                        } else {
                            (&qb[..rows * q], &qa[..rows * p])
                        };
                        let large = staged(rows, order, large)?;
                        let cross = &mut cross[..order * k];
                        standard_product(
                            large,
                            staged(rows, k, small)?,
                            MatrixViewMut::from_slice(order, k, cross)?,
                            GemmConfig {
                                left_op: GemmOp::Adjoint,
                                ..GemmConfig::ordinary()
                            },
                            options,
                            child,
                            kernels,
                        )?;
                        let cross = staged(order, k, cross)?;
                        admit_stage(kernels, solve)?;
                        thin_candidate(
                            cross,
                            *options,
                            child,
                            kernels.scoped(solve),
                            |spectral| {
                                cosines[..k].copy_from_slice(&spectral.values[..k]);
                                Ok(())
                            },
                        )?;
                        // Principal angles (Björck-Golub with the Knyazev-Argentati
                        // fix): cosines are the singular values of
                        // `Q_large^H Q_small`, sines those of
                        // `(I - Q_large Q_large^H) Q_small`. `acos` is
                        // ill-conditioned for small angles and `asin` for angles
                        // near pi/2, so each angle uses the better formula,
                        // switching at 45 degrees; the sine stage runs only when
                        // some angle is below it. Both SVDs return descending
                        // singular values. Reversing the cosine indices pairs
                        // them with descending sines for the same angle.
                        let split = S::Real::try_from_f64_exact(0.5)?.sqrt();
                        let cosine =
                            |i: usize| cosines[k - 1 - i].clamp(S::Real::ZERO, S::Real::ONE);
                        if cosine(k - 1) > split {
                            let residual = &mut residual[..rows * k];
                            residual.copy_from_slice(small);
                            standard_product(
                                large,
                                cross,
                                MatrixViewMut::from_slice(rows, k, residual)?,
                                GemmConfig {
                                    alpha: -S::ONE,
                                    beta: S::ONE,
                                    ..GemmConfig::ordinary()
                                },
                                options,
                                child,
                                kernels,
                            )?;
                            admit_stage(kernels, solve)?;
                            values_candidate(
                                staged(rows, k, residual)?,
                                *options,
                                child,
                                kernels.scoped(solve),
                                |sines| {
                                    for (i, angle) in angles[..k].iter_mut().enumerate() {
                                        let sine = sines[i].clamp(S::Real::ZERO, S::Real::ONE);
                                        *angle = if cosine(i) > split {
                                            sine.asin()
                                        } else {
                                            cosine(i).acos()
                                        };
                                    }
                                    Ok(())
                                },
                            )?;
                        } else {
                            for (i, angle) in angles[..k].iter_mut().enumerate() {
                                *angle = cosine(i).acos();
                            }
                        }
                        // Rounding around the sine/cosine crossover may perturb the order.
                        angles[..k].sort_unstable_by(|x, y| {
                            y.partial_cmp(x).unwrap_or(core::cmp::Ordering::Equal)
                        });
                    }
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.clear();
                    output.extend_from_slice(&angles[..k]);
                    Ok(report)
                },
            )
        },
    )
}

/// Solves the orthogonal Procrustes problem `min ‖A Q − B‖_F` over unitary `Q`.
///
/// `A` and `B` are `m × n`, and `Q` is `n × n`. With the SVD `Aᴴ B = U Σ Vᴴ`, the
/// minimizer is `Q = U Vᴴ`, the unitary polar factor of `Aᴴ B`. It is unique if and only if
/// `Aᴴ B` is nonsingular ([`ProcrustesReport::uniqueness`]). The rank policy only
/// classifies `Aᴴ B`; it does not truncate `Q`. The cost is `O(m n²)` for the products and
/// `O(n³)` for the SVD.
///
/// # Errors
///
/// The errors of [`orthogonal_procrustes_into`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, RankPolicy, orthogonal_procrustes};
/// let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0])?;
/// let b = Matrix::from_vec(2, 2, vec![0.0_f64, 1.0, 1.0, 0.0])?;
/// let alignment = orthogonal_procrustes(&a, &b, RankPolicy::Exact)?;
/// // B is A with its columns swapped, so a permutation aligns them exactly.
/// assert!(alignment.report().residual_norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn orthogonal_procrustes<'a, 'b, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<MatrixView<'b, S>>,
    policy: RankPolicy<S::Real>,
) -> Result<ProcrustesAlignment<S>, SciOxideError> {
    let mut transformation = Matrix::empty();
    let report = orthogonal_procrustes_into(
        a,
        b,
        policy,
        &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
        &mut transformation,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    Ok(ProcrustesAlignment {
        transformation,
        report,
    })
}

/// Writes the solution `Q` of [`orthogonal_procrustes`] to `output` and returns the diagnostics.
///
/// `output` becomes `n × n`. The products `Aᴴ B`, `U Vᴴ` and the residual `A Q − B` use
/// Standard accuracy, and the residual is formed in blocks of at most 256 rows. Size the
/// workspace with [`ProcrustesWorkspaceRequest::new`]. The SVD counts toward the iteration
/// limit of `options` in a child scope.
///
/// # Errors
///
/// - `DimensionMismatch` if `a` and `b` have different shapes.
/// - `NonFinite` if an entry of `a` or `b` is NaN or infinite.
/// - The errors of `policy`.
/// - `NumericalInstability` if `Aᴴ B` or the sum of its singular values overflows.
/// - `ConvergenceFailure` if the SVD exceeds the iteration limit.
/// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
pub fn orthogonal_procrustes_into<'a, 'b, 'scope, 'capability, S: NumericScalar>(
    a: impl Into<MatrixView<'a, S>>,
    b: impl Into<MatrixView<'b, S>>,
    policy: RankPolicy<S::Real>,
    options: &LinalgExecutionOptions,
    output: &mut Matrix<S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<ProcrustesReport<S::Real>, SciOxideError> {
    workspace.begin_attempt();
    let scope = context.into().with_options(options)?;
    let kernels = scope.kernel();
    kernels.execution().check()?;
    let (a, b) = (a.into(), b.into());
    matching_rows(a, b, true)?;
    let (rows, cols) = (a.rows(), a.cols());
    let request = SubspaceRequest::procrustes(rows, cols, options).for_kernels(kernels);
    let requirements = request.requirements::<S>()?;
    let a = a.try_finite_dispatched(kernels.dispatch())?;
    let b = b.try_finite_dispatched(kernels.dispatch())?;
    policy.cutoff(S::Real::ZERO, cols, cols)?;
    workspace.try_reserve(&requirements)?;
    output.try_reserve_shape(cols, cols)?;
    let stage = request.job.stage()?;
    workspace.run_scope(
        scope,
        identity(
            scioxide_core::algorithm_id!("linalg", "orthogonal_procrustes", 1),
            rows,
            cols,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |workspace, kernels, solve| {
            workspace.with_prefixes([stage.scalars, 0, 0], |scalars, _, _, child| {
                let square = cols * cols;
                let (cross, rest) = crate::workspace::split_mut(scalars, square)?;
                let (u, rest) = crate::workspace::split_mut(rest, square)?;
                let (vt, block) = crate::workspace::split_mut(rest, square)?;
                // Orthogonal Procrustes: `min ||A Q - B||_F` over unitary Q is
                // solved by `Q = U V^H` from the SVD `A^H B = U S V^H`. Zero
                // singular values make Q nonunique.
                standard_product(
                    a,
                    b,
                    MatrixViewMut::from_slice(cols, cols, cross)?,
                    GemmConfig {
                        left_op: GemmOp::Adjoint,
                        ..GemmConfig::ordinary()
                    },
                    options,
                    child,
                    kernels,
                )?;
                admit_stage(kernels, solve)?;
                let (rank, condition, uniqueness, scale) = thin_candidate(
                    staged(cols, cols, cross)?,
                    *options,
                    child,
                    kernels.scoped(solve),
                    |spectral| {
                        crate::decompose::util::copy_rows(spectral.u, u);
                        crate::decompose::util::copy_rows(spectral.vt, vt);
                        let rank = crate::analysis::analyze_magnitudes(
                            spectral.values.iter().copied(),
                            cols,
                            cols,
                            cols,
                            policy,
                        )?;
                        let condition = if cols == 0 {
                            None
                        } else {
                            Some(spectrum_condition_estimate(spectral.values)?)
                        };
                        let uniqueness = if spectral.values.contains(&S::Real::ZERO) {
                            ProcrustesUniqueness::Nonunique
                        } else if rank.rank() < cols {
                            ProcrustesUniqueness::NumericallyUnresolved
                        } else {
                            ProcrustesUniqueness::Unique
                        };
                        let scale = spectral.values.iter().try_fold(
                            ScaledAccumulator::ZERO,
                            |mut total, &sigma| {
                                total.add_scaled(Scaled::from_value(sigma)?)?;
                                Ok::<_, SciOxideError>(total)
                            },
                        )?;
                        Ok((rank, condition, uniqueness, scale.total()?.round()?))
                    },
                )?;
                let transformation = &mut cross[..];
                standard_product(
                    staged(cols, cols, u)?,
                    staged(cols, cols, vt)?,
                    MatrixViewMut::from_slice(cols, cols, transformation)?,
                    GemmConfig::ordinary(),
                    options,
                    child,
                    kernels,
                )?;
                // The residual `A R - B` is formed one row block at a time and
                // reduced with binary-scaled squares.
                let alignment = staged(cols, cols, transformation)?;
                let mut residual = ScaledNorm::new();
                for start in (0..rows).step_by(RESIDUAL_ROWS) {
                    let height = RESIDUAL_ROWS.min(rows - start);
                    let block = &mut block[..height * cols];
                    crate::decompose::util::copy_rows(
                        b.as_view().block(start, 0, height, cols)?,
                        block,
                    );
                    standard_product(
                        a.block(start, 0, height, cols)?,
                        alignment,
                        MatrixViewMut::from_slice(height, cols, block)?,
                        GemmConfig {
                            beta: -S::ONE,
                            prior: Some(b.maximum),
                            ..GemmConfig::ordinary()
                        },
                        options,
                        child,
                        kernels,
                    )?;
                    residual.add(crate::kernel::standard::norm_workspace(
                        block.iter().copied(),
                    )?)?;
                }
                let report = ProcrustesReport {
                    scale,
                    residual_norm: residual.finish()?,
                    cross_product: SubspaceReport { rank, condition },
                    uniqueness,
                };
                kernels.execution().check()?;
                let _ = solve.counts().total()?;
                publish(output, cols, cols, transformation)?;
                Ok(report)
            })
        },
    )
}
