//! Level 2 rank-one and rank-two updates for dense and packed storage.

struct RankEntry<S> {
    xi: S,
    xj: S,
    yi: S,
    yj: S,
}
struct RankVectors<'a, S: NumericScalar> {
    alpha: S,
    x: VectorView<'a, S>,
    y: VectorView<'a, S>,
    kind: RankKind,
}

use scioxide_core::AlgebraicScalar;
use scioxide_core::{Dimensions, NumericScalar, ProductSum, SciOxideError};

use crate::kernel::triangle_contains;
use crate::{
    DenseWorkspaceRequest, KernelContext, LinalgExecutionOptions, LinalgWorkspace, MatrixViewMut,
    PackedKind, PackedMatrixViewMut, VectorView,
};
use scioxide_core::Triangle;

#[cfg(feature = "blas")]
use crate::{BlasMatrixMut, BlasOperation, BlasPackedMatrixMut, BlasStructuredMatrixMut};

#[derive(Clone, Copy)]
enum RankKind {
    General,
    GeneralConjugated,
    SymmetricOne,
    SymmetricTwo,
    HermitianOne,
    HermitianTwo,
}

impl RankKind {
    fn structured(self) -> bool {
        !matches!(self, Self::General | Self::GeneralConjugated)
    }
    fn hermitian(self) -> bool {
        matches!(self, Self::HermitianOne | Self::HermitianTwo)
    }
    #[cfg(feature = "blas")]
    fn blas_operation<S: NumericScalar>(self, packed: bool) -> BlasOperation {
        let complex = matches!(
            S::kind(),
            scioxide_core::backend::ScalarKind::Complex32
                | scioxide_core::backend::ScalarKind::Complex64
        );
        match (self, packed, complex) {
            (Self::General, _, _) => BlasOperation::Geru,
            (Self::GeneralConjugated, _, true) => BlasOperation::Gerc,
            (Self::GeneralConjugated, _, false) => BlasOperation::Geru,
            (Self::SymmetricOne, false, _) => BlasOperation::Syr,
            (Self::SymmetricTwo, false, _) => BlasOperation::Syr2,
            (Self::SymmetricOne, true, _) => BlasOperation::Spr,
            (Self::SymmetricTwo, true, _) => BlasOperation::Spr2,
            (Self::HermitianOne, false, true) => BlasOperation::Her,
            (Self::HermitianTwo, false, true) => BlasOperation::Her2,
            (Self::HermitianOne, true, true) => BlasOperation::Hpr,
            (Self::HermitianTwo, true, true) => BlasOperation::Hpr2,
            (Self::HermitianOne, false, false) => BlasOperation::Syr,
            (Self::HermitianTwo, false, false) => BlasOperation::Syr2,
            (Self::HermitianOne, true, false) => BlasOperation::Spr,
            (Self::HermitianTwo, true, false) => BlasOperation::Spr2,
        }
    }
}

fn candidate<S: NumericScalar>(
    prior: S,
    alpha: S,
    entry: RankEntry<S>,
    kind: RankKind,
    diagonal: bool,
) -> Result<S, SciOxideError> {
    let RankEntry { xi, xj, yi, yj } = entry;

    let mut sum = ProductSum::new();
    sum.add_product(prior, S::ONE)?;
    match kind {
        RankKind::General => sum.add_triple_product(alpha, xi, yj)?,
        RankKind::GeneralConjugated => sum.add_triple_product(alpha, xi, yj.conj())?,
        RankKind::SymmetricOne => sum.add_triple_product(alpha, xi, xj)?,
        RankKind::SymmetricTwo => {
            sum.add_triple_product(alpha, xi, yj)?;
            sum.add_triple_product(alpha, yi, xj)?;
        }
        RankKind::HermitianOne => sum.add_triple_product(alpha, xi, xj.conj())?,
        RankKind::HermitianTwo => {
            sum.add_triple_product(alpha, xi, yj.conj())?;
            sum.add_triple_product(alpha.conj(), yi, xj.conj())?;
        }
    }
    let result = sum.finish()?;
    Ok(if diagonal && kind.hermitian() {
        S::from_real(result.real_component())
    } else {
        result
    })
}

fn validate<S: NumericScalar>(
    rows: usize,
    cols: usize,
    x: VectorView<'_, S>,
    y: VectorView<'_, S>,
    alpha: S,
    kind: RankKind,
) -> Result<(), SciOxideError> {
    if kind.structured() && rows != cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(rows, rows),
            Dimensions::two(rows, cols),
        ));
    }
    let expected_y = if matches!(kind, RankKind::SymmetricOne | RankKind::HermitianOne) {
        rows
    } else {
        cols
    };
    if x.len() != rows || y.len() != expected_y {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(rows, expected_y),
            Dimensions::two(x.len(), y.len()),
        ));
    }
    crate::checked::validate_input(alpha, "alpha")?;
    if kind.hermitian()
        && matches!(kind, RankKind::HermitianOne)
        && alpha.imaginary_component() != S::Real::ZERO
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("alpha"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Hermitian rank-one scale must be real",
        ));
    }
    let _ = x.try_finite()?;
    let _ = y.try_finite()?;
    Ok(())
}

fn dense_update<S: NumericScalar>(
    matrix: &mut MatrixViewMut<'_, S>,
    triangle: Triangle,
    vectors: RankVectors<'_, S>,
    scratch: &mut [S],
    policy: scioxide_core::ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let RankVectors { alpha, x, y, kind } = vectors;

    #[cfg(not(feature = "blas"))]
    let _ = policy;
    let rows = matrix.rows();
    let cols = matrix.cols();
    validate(rows, cols, x, y, alpha, kind)?;
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("scratch")))?;
    if scratch.len() < len {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(len),
            Dimensions::one(scratch.len()),
        ));
    }
    context.execution().check()?;
    let selected = |row, col| !kind.structured() || triangle_contains(triangle, row, col);
    let prior = |matrix: crate::MatrixView<'_, S>, row, col| {
        let prior = matrix.get_element(row, col);
        crate::checked::validate_input(prior, "selected coefficient")?;
        if kind.hermitian() && row == col && prior.imaginary_component() != S::Real::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Hermitian diagonal must be real",
            ));
        }
        Ok(prior)
    };
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && len > 0
    {
        // SciOxide stages the candidate itself, so it uses the provider's layout.
        let capabilities = dispatch.capabilities()?;
        let layout = crate::blas::dispatch::preferred_layout(&capabilities);
        let index = |row, col| crate::blas::dispatch::matrix_index(layout, rows, cols, row, col);
        for row in 0..rows {
            for col in 0..cols {
                scratch[index(row, col)] = if selected(row, col) {
                    prior(matrix.as_view(), row, col)?
                } else {
                    S::ZERO
                };
            }
        }
        let route = crate::blas::dispatch::rank_update(crate::blas::dispatch::RankUpdateRequest {
            blas_operation: kind.blas_operation::<S>(false),
            triangle: (triangle),
            alpha,
            x,
            y: if matches!(kind, RankKind::SymmetricOne | RankKind::HermitianOne) {
                None
            } else {
                Some(y)
            },
            matrix: BlasStructuredMatrixMut::Dense(BlasMatrixMut::new(
                &mut scratch[..len],
                rows,
                cols,
                crate::blas::dispatch::leading_dimension(layout, rows, cols),
                layout,
            )),
            dispatch,
            capabilities: &capabilities,
            accuracy: context.accuracy(),
            workers: crate::kernel::provider_worker_bound::<S>(rows, 1, cols, policy, context)?,
        })?;
        if route == crate::blas::dispatch::BlasRoute::Provider {
            for row in 0..rows {
                for col in 0..cols {
                    if selected(row, col) {
                        matrix.set_element(row, col, scratch[index(row, col)]);
                    }
                }
            }
            return Ok(());
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            if !selected(row, col) {
                continue;
            }
            let prior = prior(matrix.as_view(), row, col)?;
            let xj = if kind.structured() {
                x.get_element(col)
            } else {
                S::ZERO
            };
            let yi = if matches!(kind, RankKind::SymmetricTwo | RankKind::HermitianTwo) {
                y.get_element(row)
            } else {
                S::ZERO
            };
            scratch[row * cols + col] = candidate(
                prior,
                alpha,
                RankEntry {
                    xi: x.get_element(row),
                    xj,
                    yi,
                    yj: y.get_element(col),
                },
                kind,
                row == col,
            )?;
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            if selected(row, col) {
                matrix.set_element(row, col, scratch[row * cols + col]);
            }
        }
    }
    Ok(())
}

impl<S: NumericScalar> MatrixViewMut<'_, S> {
    /// Computes the general rank-one update `A := α x yᵀ + A`.
    ///
    /// `output` is the `m × n` matrix `A`, `x` has length `m` and `y` has length `n`. No
    /// entry is conjugated (BLAS GERU).
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `m n` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `y` does not match the shape of `output`.
    /// - `NonFinite` if `alpha` or an entry of `x`, `y` or `output` is NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn rank_one_update_in_place<'v, 'scope, 'a>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let y = y.into();
        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            Triangle::Lower,
            RankVectors {
                alpha,
                x,
                y,
                kind: RankKind::General,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the general rank-one update `A := α x yᴴ + A`.
    ///
    /// `output` is the `m × n` matrix `A`, `x` has length `m` and `y` has length `n`. The
    /// entries of `y` are conjugated (BLAS GERC). For real scalars this equals
    /// [`rank_one_update_in_place`](Self::rank_one_update_in_place).
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `m n` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `y` does not match the shape of `output`.
    /// - `NonFinite` if `alpha` or an entry of `x`, `y` or `output` is NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn rank_one_adjoint_update_in_place<'v, 'scope, 'a>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let y = y.into();
        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            Triangle::Lower,
            RankVectors {
                alpha,
                x,
                y,
                kind: RankKind::GeneralConjugated,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the symmetric rank-one update `A := α x xᵀ + A` on one triangle of `A`.
    ///
    /// `output` is the `n × n` matrix `A`. Only its `triangle` is read and written; the other
    /// triangle is unchanged.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n²` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not square or `x` does not have length `n`.
    /// - `NonFinite` if `alpha`, an entry of `x` or an entry of the selected triangle is NaN
    ///   or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn symmetric_rank_one_update_in_place<'v, 'scope, 'a>(
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();

        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            triangle,
            RankVectors {
                alpha,
                x,
                y: x,
                kind: RankKind::SymmetricOne,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the Hermitian rank-one update `A := α x xᴴ + A` on one triangle of `A`.
    ///
    /// `output` is the `n × n` matrix `A`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `alpha` and the diagonal of `A` must be real, and the updated
    /// diagonal is stored with zero imaginary part.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n²` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not square or `x` does not have length `n`.
    /// - `InvalidInput` if `alpha` or a diagonal entry of `output` has a nonzero imaginary
    ///   part.
    /// - `NonFinite` if `alpha`, an entry of `x` or an entry of the selected triangle is NaN
    ///   or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn hermitian_rank_one_update_in_place<'v, 'scope, 'a>(
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();

        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            triangle,
            RankVectors {
                alpha,
                x,
                y: x,
                kind: RankKind::HermitianOne,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the symmetric rank-two update `A := α x yᵀ + α y xᵀ + A` on one triangle of `A`.
    ///
    /// `output` is the `n × n` matrix `A`. Only its `triangle` is read and written; the other
    /// triangle is unchanged.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n²` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not square or `x` or `y` does not have length `n`.
    /// - `NonFinite` if `alpha`, an entry of `x` or `y` or an entry of the selected triangle is
    ///   NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn symmetric_rank_two_update_in_place<'v, 'scope, 'a>(
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let y = y.into();
        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            triangle,
            RankVectors {
                alpha,
                x,
                y,
                kind: RankKind::SymmetricTwo,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the Hermitian rank-two update `A := α x yᴴ + conj(α) y xᴴ + A` on one triangle.
    ///
    /// `output` is the `n × n` matrix `A`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `alpha` may be complex. The diagonal of `A` must be real, and
    /// the updated diagonal is stored with zero imaginary part.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n²` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not square or `x` or `y` does not have length `n`.
    /// - `InvalidInput` if a diagonal entry of `output` has a nonzero imaginary part.
    /// - `NonFinite` if `alpha`, an entry of `x` or `y` or an entry of the selected triangle is
    ///   NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn hermitian_rank_two_update_in_place<'v, 'scope, 'a>(
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let y = y.into();
        let len = scioxide_core::backend::checked_product(
            &[output.rows(), output.cols()],
            scioxide_core::quantity!("rank_update_scratch"),
        )?;
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        dense_update(
            &mut output,
            triangle,
            RankVectors {
                alpha,
                x,
                y,
                kind: RankKind::HermitianTwo,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }
}

impl<S: NumericScalar> PackedMatrixViewMut<'_, S> {
    /// Computes the Hermitian rank-one update `A := α x xᴴ + A` in packed storage.
    ///
    /// `output` holds the stored triangle of the `n × n` Hermitian matrix `A` and must have
    /// [`PackedKind::Hermitian`]. `alpha` must be real. The updated diagonal is stored with
    /// zero imaginary part.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n (n + 1) / 2` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` is not [`PackedKind::Hermitian`] or `alpha` has a nonzero
    ///   imaginary part.
    /// - `DimensionMismatch` if `x` does not have length `n`.
    /// - `NonFinite` if `alpha` or an entry of `x` or `output` is NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn hermitian_rank_one_update_in_place<'v, 'scope, 'a>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();

        let len = output.as_view().as_slice().len();
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        output.rank_update(
            RankVectors {
                alpha,
                x,
                y: x,
                kind: RankKind::HermitianOne,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    /// Computes the Hermitian rank-two update `A := α x yᴴ + conj(α) y xᴴ + A` in packed storage.
    ///
    /// `output` holds the stored triangle of the `n × n` Hermitian matrix `A` and must have
    /// [`PackedKind::Hermitian`]. `alpha` may be complex. The updated diagonal is stored with
    /// zero imaginary part.
    /// Without a BLAS provider each updated entry is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `n (n + 1) / 2` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output` is not [`PackedKind::Hermitian`].
    /// - `DimensionMismatch` if `x` or `y` does not have length `n`.
    /// - `NonFinite` if `alpha` or an entry of `x`, `y` or `output` is NaN or infinite.
    /// - `NumericalInstability` if an updated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn hermitian_rank_two_update_in_place<'v, 'scope, 'a>(
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        y: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
        mut output: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let y = y.into();
        let len = output.as_view().as_slice().len();
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
            &DenseWorkspaceRequest::vector(len, options).for_kernels(context),
        )?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        output.rank_update(
            RankVectors {
                alpha,
                x,
                y,
                kind: RankKind::HermitianTwo,
            },
            scratch,
            options.execution_policy(),
            context,
        )
    }

    fn rank_update(
        &mut self,
        vectors: RankVectors<'_, S>,
        scratch: &mut [S],
        policy: scioxide_core::ExecutionPolicy,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        let RankVectors { alpha, x, y, kind } = vectors;

        #[cfg(not(feature = "blas"))]
        let _ = policy;
        if self.kind() != PackedKind::Hermitian {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "packed Hermitian interpretation required",
            ));
        }
        let n = self.order();
        validate(n, n, x, y, alpha, kind)?;
        let len =
            n.checked_mul(n.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("scratch"))
            })?)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("scratch")))?
                / 2;
        if scratch.len() < len {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(len),
                Dimensions::one(scratch.len()),
            ));
        }
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas()
            && n > 0
        {
            let mut next = 0;
            for col in 0..n {
                let rows = match self.triangle() {
                    Triangle::Upper => 0..col + 1,
                    Triangle::Lower => col..n,
                };
                for row in rows {
                    let prior = self.as_view().element(row, col);
                    crate::checked::validate_input(prior, "selected coefficient")?;
                    scratch[next] = prior;
                    next += 1;
                }
            }
            let route =
                crate::blas::dispatch::rank_update(crate::blas::dispatch::RankUpdateRequest {
                    blas_operation: kind.blas_operation::<S>(true),
                    triangle: (self.triangle()),
                    alpha,
                    x,
                    y: if matches!(kind, RankKind::HermitianOne) {
                        None
                    } else {
                        Some(y)
                    },
                    matrix: BlasStructuredMatrixMut::Packed(BlasPackedMatrixMut::new(
                        &mut scratch[..len],
                        n,
                        self.triangle(),
                    )),
                    dispatch,
                    capabilities: &dispatch.capabilities()?,
                    accuracy: context.accuracy(),
                    workers: crate::kernel::provider_worker_bound::<S>(n, 1, n, policy, context)?,
                })?;
            if route == crate::blas::dispatch::BlasRoute::Provider {
                let mut next = 0;
                for col in 0..n {
                    let rows = match self.triangle() {
                        Triangle::Upper => 0..col + 1,
                        Triangle::Lower => col..n,
                    };
                    for row in rows {
                        self.set_element(row, col, scratch[next]);
                        next += 1;
                    }
                }
                return Ok(());
            }
        }
        let mut next = 0;
        for col in 0..n {
            let rows = match self.triangle() {
                Triangle::Upper => 0..col + 1,
                Triangle::Lower => col..n,
            };
            for row in rows {
                let prior = self.as_view().element(row, col);
                crate::checked::validate_input(prior, "selected coefficient")?;
                scratch[next] = candidate(
                    prior,
                    alpha,
                    RankEntry {
                        xi: x.get_element(row),
                        xj: x.get_element(col),
                        yi: y.get_element(row),
                        yj: y.get_element(col),
                    },
                    kind,
                    row == col,
                )?;
                next += 1;
            }
        }
        next = 0;
        for col in 0..n {
            let rows = match self.triangle() {
                Triangle::Upper => 0..col + 1,
                Triangle::Lower => col..n,
            };
            for row in rows {
                self.set_element(row, col, scratch[next]);
                next += 1;
            }
        }
        Ok(())
    }
}
