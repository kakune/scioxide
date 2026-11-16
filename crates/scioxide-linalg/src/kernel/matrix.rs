//! Level 3 products, solves and rank-k updates with symmetric, Hermitian and triangular
//! matrices.

struct SelfAdjointProduct<S> {
    side: MatrixSide,
    triangle: Triangle,
    hermitian: bool,
    alpha: S,
    beta: S,
}
pub(crate) struct TriangularProduct<S> {
    pub(crate) side: MatrixSide,
    pub(crate) triangle: Triangle,
    pub(crate) diagonal: DiagonalKind,
    pub(crate) operation: StructuredOperation,
    pub(crate) alpha: S,
}

use scioxide_core::AlgebraicScalar;
use scioxide_core::{Dimensions, NumericScalar, SciOxideError};

use crate::kernel::{self, GemmOp};
use crate::{
    DenseWorkspaceRequest, KernelContext, LinalgExecutionOptions, LinalgWorkspace, Matrix,
    MatrixView, MatrixViewMut, StructuredOperation,
};
use scioxide_core::{DiagonalKind, Triangle};

#[cfg(feature = "blas")]
use crate::{BlasOperation, BlasTransform};

/// Side on which a structured matrix `A` multiplies a general matrix `B`.
///
/// Used by the symmetric, Hermitian and triangular Level 3 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixSide {
    /// `A B` for a product, `op(A) X = B` for a solve.
    Left,
    /// `B A` for a product, `X op(A) = B` for a solve.
    Right,
}

fn map_operation(operation: StructuredOperation) -> GemmOp {
    match operation {
        StructuredOperation::Normal => GemmOp::None,
        StructuredOperation::Transpose => GemmOp::Transpose,
        StructuredOperation::Adjoint => GemmOp::Adjoint,
    }
}

#[cfg(feature = "blas")]
fn blas_transform(operation: StructuredOperation) -> BlasTransform {
    match operation {
        StructuredOperation::Normal => BlasTransform::None,
        StructuredOperation::Transpose => BlasTransform::Transpose,
        StructuredOperation::Adjoint => BlasTransform::Adjoint,
    }
}

fn self_adjoint_get<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    triangle: Triangle,
    row: usize,
    col: usize,
    hermitian: bool,
) -> S {
    if kernel::triangle_contains(triangle, row, col) {
        matrix.get_element(row, col)
    } else {
        let value = matrix.get_element(col, row);
        if hermitian { value.conj() } else { value }
    }
}

/// Check finiteness of the selected triangle a provider reads. Shape and the
/// real Hermitian diagonal are checked before the route is chosen.
#[cfg(feature = "blas")]
fn validate_selected<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    triangle: Triangle,
) -> Result<(), SciOxideError> {
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            if kernel::triangle_contains(triangle, row, col) {
                crate::checked::validate_input(
                    matrix.get_element(row, col),
                    "selected coefficient",
                )?;
            }
        }
    }
    Ok(())
}

fn self_adjoint_product<S: NumericScalar>(
    structured: MatrixView<'_, S>,
    general: MatrixView<'_, S>,
    output: &mut MatrixViewMut<'_, S>,
    product: SelfAdjointProduct<S>,
    options: &LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let SelfAdjointProduct {
        side,
        triangle,
        hermitian,
        alpha,
        beta,
    } = product;

    let rows = output.rows();
    let cols = output.cols();
    let order = match side {
        MatrixSide::Left => rows,
        MatrixSide::Right => cols,
    };
    if structured.rows() != order || structured.cols() != order {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(order, order),
            Dimensions::two(structured.rows(), structured.cols()),
        ));
    }
    if general.rows() != rows || general.cols() != cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(rows, cols),
            Dimensions::two(general.rows(), general.cols()),
        ));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("scratch")))?;
    crate::checked::validate_input(alpha, "alpha")?;
    crate::checked::validate_input(beta, "beta")?;
    if len == 0 {
        return Ok(());
    }
    if alpha != S::ZERO && hermitian {
        for row in 0..order {
            let value = structured.get_element(row, row);
            crate::checked::validate_input(value, "Hermitian diagonal")?;
            if value.imaginary_component() != S::Real::ZERO {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Hermitian diagonal must be real",
                ));
            }
        }
    }
    context.execution().check()?;
    workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(
        &DenseWorkspaceRequest::structured(rows, order, cols, options).for_kernels(context),
    )?)?;
    #[cfg(feature = "blas")]
    if context.blas().is_some() {
        workspace.require_packed(
            scioxide_core::backend::checked_product(
                &[structured.rows(), structured.cols()],
                scioxide_core::quantity!("provider_structured_operand"),
            )?,
            len,
        )?;
    }
    kernel::structured_product::with_candidate(workspace, len, |scratch, workspace| {
        #[cfg(not(feature = "blas"))]
        let _ = scratch;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            let capabilities = dispatch.capabilities()?;
            let layout = crate::blas::dispatch::preferred_layout(&capabilities);
            if alpha != S::ZERO {
                validate_selected(structured, triangle)?;
                let _ = general.try_finite()?;
            }
            if beta != S::ZERO {
                let _ = output.as_view().try_finite()?;
            }
            for row in 0..rows {
                for col in 0..cols {
                    scratch[crate::blas::dispatch::matrix_index(layout, rows, cols, row, col)] =
                        if beta == S::ZERO {
                            S::ZERO
                        } else {
                            output.as_view().get_element(row, col)
                        };
                }
            }
            let crate::workspace::ProductParts {
                packed_a, packed_b, ..
            } = workspace.product_parts()?;
            let route = crate::blas::dispatch::level3(crate::blas::dispatch::Level3Request {
                kind: crate::blas::dispatch::Level3Kind::SelfAdjoint { side: (side) },
                blas_operation: if hermitian && !S::IS_REAL {
                    BlasOperation::Hemm
                } else {
                    BlasOperation::Symm
                },
                triangle: (triangle),
                alpha,
                a: structured,
                b: general,
                beta,
                candidate: &mut scratch[..len],
                packed_a,
                packed_b,
                rows,
                cols,
                dispatch,
                capabilities: &capabilities,
                accuracy: context.accuracy(),
                workers: kernel::provider_worker_bound::<S>(
                    rows,
                    order,
                    cols,
                    context.policy(),
                    context,
                )?,
            })?;
            if route == crate::blas::dispatch::BlasRoute::Provider {
                for row in 0..rows {
                    for col in 0..cols {
                        output.set_element(
                            row,
                            col,
                            scratch
                                [crate::blas::dispatch::matrix_index(layout, rows, cols, row, col)],
                        );
                    }
                }
                return Ok(());
            }
        }
        kernel::structured_product::product(
            crate::kernel::structured_product::ProductShape {
                rows,
                shared: order,
                cols,
            },
            |row, col| {
                if alpha == S::ZERO {
                    S::ZERO
                } else {
                    match side {
                        MatrixSide::Left => {
                            self_adjoint_get(structured, triangle, row, col, hermitian)
                        }
                        MatrixSide::Right => general.get_element(row, col),
                    }
                }
            },
            |row, col| {
                if alpha == S::ZERO {
                    S::ZERO
                } else {
                    match side {
                        MatrixSide::Left => general.get_element(row, col),
                        MatrixSide::Right => {
                            self_adjoint_get(structured, triangle, row, col, hermitian)
                        }
                    }
                }
            },
            output.reborrow(),
            kernel::GemmConfig {
                alpha,
                beta,
                ..kernel::GemmConfig::ordinary()
            },
            crate::kernel::KernelExecution {
                policy: options.execution_policy(),
                context,
            },
            workspace,
        )
    })
}

fn rank_two_k<S: NumericScalar>(
    problem: RankTwoProblem<'_, '_, '_, S>,
    policy: RankTwoPolicy<S>,
    options: &LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let len = scioxide_core::backend::checked_product(
        &[problem.output.rows(), problem.output.cols()],
        scioxide_core::quantity!("rank_two_candidate"),
    )?;
    let shared = match policy.system {
        StructuredOperation::Normal => problem.a.cols(),
        _ => problem.a.rows(),
    };
    workspace.try_reserve(&crate::LinalgWorkspace::<S>::workspace_requirements(
        &crate::DenseWorkspaceRequest::rank_two(problem.output.rows(), shared, options)
            .for_kernels(context),
    )?)?;
    kernel::structured_product::with_candidate(workspace, len, |scratch, workspace| {
        let RankTwoLayout {
            n,
            operands,
            #[cfg(feature = "blas")]
            len,
        } = validate_rank_two_k(&problem, &policy, scratch.len())?;
        #[cfg(feature = "blas")]
        let (a, b) = (problem.a, problem.b);
        let output = problem.output;
        let RankTwoPolicy {
            triangle,
            system,
            hermitian,
            alpha,
            beta,
        } = policy;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            let capabilities = dispatch.capabilities()?;
            let layout = crate::blas::dispatch::preferred_layout(&capabilities);
            for row in 0..n {
                for col in 0..n {
                    scratch[crate::blas::dispatch::matrix_index(layout, n, n, row, col)] =
                        if kernel::triangle_contains(triangle, row, col) && beta != S::ZERO {
                            output.as_view().get_element(row, col)
                        } else {
                            S::ZERO
                        };
                }
            }
            workspace.require_packed(
                scioxide_core::backend::checked_product(
                    &[a.rows(), a.cols()],
                    scioxide_core::quantity!("provider_rank_two_input"),
                )?,
                scioxide_core::backend::checked_product(
                    &[b.rows(), b.cols()],
                    scioxide_core::quantity!("provider_rank_two_input"),
                )?,
            )?;
            let crate::workspace::ProductParts {
                packed_a, packed_b, ..
            } = workspace.product_parts()?;
            let route = crate::blas::dispatch::level3(crate::blas::dispatch::Level3Request {
                kind: crate::blas::dispatch::Level3Kind::RankTwoK {
                    transform: blas_transform(system),
                },
                blas_operation: if hermitian && !S::IS_REAL {
                    BlasOperation::Her2k
                } else {
                    BlasOperation::Syr2k
                },
                triangle: (triangle),
                alpha,
                a,
                b,
                beta,
                candidate: &mut scratch[..len],
                packed_a,
                packed_b,
                rows: n,
                cols: n,
                dispatch,
                capabilities: &capabilities,
                accuracy: context.accuracy(),
                workers: kernel::provider_worker_bound::<S>(
                    n,
                    shared,
                    n,
                    context.policy(),
                    context,
                )?,
            })?;
            if route == crate::blas::dispatch::BlasRoute::Provider {
                for row in 0..n {
                    for col in 0..n {
                        if kernel::triangle_contains(triangle, row, col) {
                            output.set_element(
                                row,
                                col,
                                scratch
                                    [crate::blas::dispatch::matrix_index(layout, n, n, row, col)],
                            );
                        }
                    }
                }
                return Ok(());
            }
        }
        let transform = match system {
            StructuredOperation::Normal => GemmOp::None,
            _ if hermitian => GemmOp::Adjoint,
            _ => GemmOp::Transpose,
        };
        kernel::structured_product::rank_two(
            operands.map(|(a, b)| match transform {
                GemmOp::None => (a, b),
                GemmOp::Transpose => (a.transpose(), b.transpose()),
                GemmOp::Adjoint => (a.adjoint(), b.adjoint()),
            }),
            crate::kernel::structured_product::RankTwoUpdate {
                kind: if hermitian {
                    crate::kernel::RankKind::Hermitian
                } else {
                    crate::kernel::RankKind::Symmetric
                },
                alpha,
                beta,
                triangle,
            },
            output.as_view(),
            scratch,
            crate::kernel::KernelExecution {
                policy: options.execution_policy(),
                context,
            },
            workspace,
        )?;
        if hermitian {
            for index in 0..n {
                scratch[index * n + index] =
                    S::from_real(scratch[index * n + index].real_component());
            }
        }
        for row in 0..n {
            for col in 0..n {
                if kernel::triangle_contains(triangle, row, col) {
                    output.set_element(row, col, scratch[row * n + col]);
                }
            }
        }
        Ok(())
    })
}
struct RankTwoProblem<'a, 'b, 'c, S: NumericScalar> {
    a: MatrixView<'a, S>,
    b: MatrixView<'a, S>,
    output: &'b mut MatrixViewMut<'c, S>,
}
#[derive(Clone, Copy)]
struct RankTwoPolicy<S> {
    triangle: Triangle,
    system: StructuredOperation,
    hermitian: bool,
    alpha: S,
    beta: S,
}
struct RankTwoLayout<'a, S: NumericScalar> {
    n: usize,
    operands: Option<(
        crate::FiniteMatrixView<'a, S>,
        crate::FiniteMatrixView<'a, S>,
    )>,
    #[cfg(feature = "blas")]
    len: usize,
}
fn validate_rank_two_k<'a, S: NumericScalar>(
    problem: &RankTwoProblem<'a, '_, '_, S>,
    policy: &RankTwoPolicy<S>,
    scratch_len: usize,
) -> Result<RankTwoLayout<'a, S>, SciOxideError> {
    let a = problem.a;
    let b = problem.b;
    let output = &problem.output;
    let RankTwoPolicy {
        triangle,
        system,
        hermitian,
        alpha,
        beta,
    } = *policy;
    let n = output.rows();
    if output.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(n, n),
            Dimensions::two(output.rows(), output.cols()),
        ));
    }
    // `op(A)` is `n x k`; `b` shares the stored shape of `a`.
    let expected = if system == StructuredOperation::Normal {
        Dimensions::two(n, a.cols())
    } else {
        Dimensions::two(a.rows(), n)
    };
    if Dimensions::two(a.rows(), a.cols()) != expected {
        return Err(SciOxideError::dimension_mismatch(
            expected,
            Dimensions::two(a.rows(), a.cols()),
        ));
    }
    if (b.rows(), b.cols()) != (a.rows(), a.cols()) {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(a.rows(), a.cols()),
            Dimensions::two(b.rows(), b.cols()),
        ));
    }
    if hermitian && system == StructuredOperation::Transpose
        || !hermitian && system == StructuredOperation::Adjoint
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("operation"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "rank-2k transpose mode does not match symmetry",
        ));
    }
    if hermitian && beta.imaginary_component() != S::Real::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("beta"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Hermitian rank-2k beta must be real",
        ));
    }
    let len = n
        .checked_mul(n)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("scratch")))?;
    if scratch_len < len {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(len),
            Dimensions::one(scratch_len),
        ));
    }
    crate::checked::validate_input(alpha, "alpha")?;
    crate::checked::validate_input(beta, "beta")?;
    let operands = if alpha != S::ZERO {
        Some((a.try_finite()?, b.try_finite()?))
    } else {
        None
    };
    if beta != S::ZERO {
        for row in 0..n {
            for col in 0..n {
                if !kernel::triangle_contains(triangle, row, col) {
                    continue;
                }
                let prior = output.as_view().get_element(row, col);
                crate::checked::validate_input(prior, "selected output")?;
                if hermitian && row == col && prior.imaginary_component() != S::Real::ZERO {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("data"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "Hermitian diagonal must be real",
                    ));
                }
            }
        }
    }
    Ok(RankTwoLayout {
        n,
        operands,
        #[cfg(feature = "blas")]
        len,
    })
}

pub(crate) fn triangular_product<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    b: &mut MatrixViewMut<'_, S>,
    product: TriangularProduct<S>,
    options: &LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let TriangularProduct {
        side,
        triangle,
        diagonal,
        operation,
        alpha,
    } = product;

    let n = if side == MatrixSide::Left {
        b.rows()
    } else {
        b.cols()
    };
    let request = DenseWorkspaceRequest::triangular_product(b.rows(), n, b.cols(), options)
        .for_kernels(context);
    workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
    let len = scioxide_core::backend::checked_product(
        &[b.rows(), b.cols()],
        scioxide_core::quantity!("triangular_product_output"),
    )?;
    if len == 0 {
        return kernel::level3::triangular_with_context(
            kernel::level3::TriangularAction::Multiply,
            crate::kernel::TriangularOperand {
                view: matrix,
                triangle,
                op: map_operation(operation),
                diagonal,
            },
            b.reborrow(),
            crate::kernel::TriangularUpdate {
                side,
                alpha,
                policy: options.execution_policy(),
            },
            workspace,
            context,
        );
    }
    workspace.with_prefix(len, |scratch, child| {
        for row in 0..b.rows() {
            for col in 0..b.cols() {
                scratch[row * b.cols() + col] = b.get_element(row, col);
            }
        }
        kernel::level3::triangular_with_context(
            kernel::level3::TriangularAction::Multiply,
            crate::kernel::TriangularOperand {
                view: matrix,
                triangle,
                op: map_operation(operation),
                diagonal,
            },
            MatrixViewMut::from_slice(b.rows(), b.cols(), scratch)?,
            crate::kernel::TriangularUpdate {
                side,
                alpha,
                policy: options.execution_policy(),
            },
            child,
            context,
        )?;
        for row in 0..b.rows() {
            for col in 0..b.cols() {
                b.set_element(row, col, scratch[row * b.cols() + col]);
            }
        }
        Ok(())
    })
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Returns `A B` or `B A` for a symmetric matrix `A` stored in one triangle of `self`.
    ///
    /// `self` is the `n × n` matrix `A`. Only its `triangle` is read; the other triangle is
    /// taken as its transpose. `general` is `B`, and `side` selects `A B` or `B A`. This is
    /// [`symmetric_mul_mat_in_place`](Self::symmetric_mul_mat_in_place) with `α = 1` and
    /// `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `general.rows()` (`Left`) or
    ///   `general.cols()` (`Right`).
    /// - `NonFinite` if a read entry of `self` or an entry of `general` is NaN or infinite.
    /// - `NumericalInstability` if the product overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Triangle;
    /// use scioxide_linalg::{MatrixSide, MatrixView};
    /// // Only the lower triangle is read; the stored 9.0 is ignored.
    /// let a = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    /// let b = MatrixView::from_slice(2, 2, &[1.0, 0.0, 0.0, 1.0])?;
    /// let product = a.symmetric_mul_mat(b, MatrixSide::Left, Triangle::Lower)?;
    /// assert_eq!(product.as_slice(), &[2.0, 1.0, 1.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn symmetric_mul_mat<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.symmetric_mul_mat_with_options(
            general,
            side,
            triangle,
            &crate::LinalgExecutionOptions::default(),
        )
    }

    /// Runs [`symmetric_mul_mat`](Self::symmetric_mul_mat) with `options`, which set the product
    /// accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`symmetric_mul_mat`](Self::symmetric_mul_mat), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn symmetric_mul_mat_with_options<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        options: &LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let general = general.into();
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Matrix::zeros(general.rows(), general.cols())?;
        self.symmetric_mul_mat_in_place(
            general,
            side,
            triangle,
            S::ONE,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
    /// Computes `C := α A B + β C` or `C := α B A + β C` for a symmetric matrix `A`.
    ///
    /// `self` is the `n × n` matrix `A`, of which only `triangle` is read. `general` is `B`
    /// and `data` is `C`. Both are `m × p`, with `n = m` for [`MatrixSide::Left`] and `n = p`
    /// for [`MatrixSide::Right`]. When `alpha` is zero, `A` and `B` are not read.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not `n × n` or `general` and `data` differ in shape.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, if `alpha` is nonzero and a read
    ///   entry of `self` or an entry of `general` is, or if `beta` is nonzero and an entry of
    ///   `data` is.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn symmetric_mul_mat_in_place<'m, 'scope, 'capability>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        self_adjoint_product(
            *self,
            general.into(),
            &mut data,
            SelfAdjointProduct {
                side,
                triangle,
                hermitian: false,
                alpha,
                beta,
            },
            options,
            workspace,
            context,
        )
    }
    /// Returns `A B` or `B A` for a Hermitian matrix `A` stored in one triangle of `self`.
    ///
    /// `self` is the `n × n` matrix `A`. Only its `triangle` is read; the other triangle is
    /// taken as its conjugate transpose. The diagonal of `A` must be real. `general` is `B`,
    /// and `side` selects `A B` or `B A`. This is
    /// [`hermitian_mul_mat_in_place`](Self::hermitian_mul_mat_in_place) with `α = 1` and
    /// `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `general.rows()` (`Left`) or
    ///   `general.cols()` (`Right`).
    /// - `InvalidInput` if a diagonal entry of `self` has a nonzero imaginary part.
    /// - `NonFinite` if a read entry of `self` or an entry of `general` is NaN or infinite.
    /// - `NumericalInstability` if the product overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Complex, Triangle};
    /// use scioxide_linalg::{MatrixSide, MatrixView};
    /// let (zero, one) = (Complex::new(0.0_f64, 0.0), Complex::new(1.0, 0.0));
    /// let a = [Complex::new(2.0, 0.0), zero, Complex::new(1.0, 1.0), Complex::new(3.0, 0.0)];
    /// let a = MatrixView::from_slice(2, 2, &a)?;
    /// let identity = [one, zero, zero, one];
    /// let b = MatrixView::from_slice(2, 2, &identity)?;
    /// let product = a.hermitian_mul_mat(b, MatrixSide::Left, Triangle::Lower)?;
    /// // The upper entry is the conjugate of the stored lower entry.
    /// assert_eq!(product.as_slice()[1], Complex::new(1.0, -1.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn hermitian_mul_mat<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.hermitian_mul_mat_with_options(
            general,
            side,
            triangle,
            &crate::LinalgExecutionOptions::default(),
        )
    }

    /// Runs [`hermitian_mul_mat`](Self::hermitian_mul_mat) with `options`, which set the product
    /// accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`hermitian_mul_mat`](Self::hermitian_mul_mat), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn hermitian_mul_mat_with_options<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        options: &LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let general = general.into();
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Matrix::zeros(general.rows(), general.cols())?;
        self.hermitian_mul_mat_in_place(
            general,
            side,
            triangle,
            S::ONE,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
    /// Computes `C := α A B + β C` or `C := α B A + β C` for a Hermitian matrix `A`.
    ///
    /// `self` is the `n × n` matrix `A`, of which only `triangle` is read. Its diagonal must be
    /// real. `general` is `B` and `data` is `C`. Both are `m × p`, with `n = m` for
    /// [`MatrixSide::Left`] and `n = p` for [`MatrixSide::Right`]. When `alpha` is zero, `A`
    /// and `B` are not read.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not `n × n` or `general` and `data` differ in shape.
    /// - `InvalidInput` if `alpha` is nonzero and a diagonal entry of `self` has a nonzero
    ///   imaginary part.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, if `alpha` is nonzero and a read
    ///   entry of `self` or an entry of `general` is, or if `beta` is nonzero and an entry of
    ///   `data` is.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn hermitian_mul_mat_in_place<'m, 'scope, 'capability>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        self_adjoint_product(
            *self,
            general.into(),
            &mut data,
            SelfAdjointProduct {
                side,
                triangle,
                hermitian: true,
                alpha,
                beta,
            },
            options,
            workspace,
            context,
        )
    }
    /// Returns `α op(T) B` or `α B op(T)` for a triangular matrix `T` stored in `self`.
    ///
    /// `self` is the `n × n` matrix `T`. Only its `triangle` is read, and with
    /// [`DiagonalKind::Unit`](scioxide_core::DiagonalKind::Unit) its diagonal is taken as one.
    /// `operation` selects `op(T)` as `T`, `Tᵀ` or `Tᴴ`. `general` is `B`. When `alpha` is
    /// zero, the product is zero and neither `T` nor `B` is read. This is
    /// [`triangular_mul_mat_in_place`](Self::triangular_mul_mat_in_place) on a copy of `B`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `general.rows()` (`Left`) or
    ///   `general.cols()` (`Right`).
    /// - `NonFinite` if `alpha` is NaN or infinite, or if `alpha` is nonzero and a read entry
    ///   of `self` or an entry of `general` is.
    /// - `NumericalInstability` if the product overflows.
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{DiagonalKind, Triangle};
    /// use scioxide_linalg::{MatrixSide, MatrixView, StructuredOperation};
    /// // Lower triangle of T = [2 9; 1 3]; the stored 9.0 is ignored.
    /// let t = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    /// let b = MatrixView::from_slice(2, 1, &[1.0, 1.0])?;
    /// let product = t.triangular_mul_mat(
    ///     b,
    ///     MatrixSide::Left,
    ///     Triangle::Lower,
    ///     DiagonalKind::Explicit,
    ///     StructuredOperation::Normal,
    ///     1.0)?;
    /// assert_eq!(product.as_slice(), &[2.0, 4.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn triangular_mul_mat<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.triangular_mul_mat_with_options(
            general,
            side,
            triangle,
            diagonal,
            operation,
            alpha,
            &crate::LinalgExecutionOptions::default(),
        )
    }

    /// Runs [`triangular_mul_mat`](Self::triangular_mul_mat) with `options`, which set the product
    /// accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`triangular_mul_mat`](Self::triangular_mul_mat), and limit errors from
    /// `options` as described in [Execution](crate#execution).
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn triangular_mul_mat_with_options<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
        options: &LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = general.into().try_to_matrix()?;
        self.triangular_mul_mat_in_place(
            side,
            triangle,
            diagonal,
            operation,
            alpha,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
    /// Computes `B := α op(T) B` or `B := α B op(T)` for a triangular matrix `T`.
    ///
    /// `self` is the `n × n` matrix `T`, of which only `triangle` is read. With
    /// [`DiagonalKind::Unit`](scioxide_core::DiagonalKind::Unit) its diagonal is taken as one.
    /// `operation` selects `op(T)` as `T`, `Tᵀ` or `Tᴴ`. `data` holds `B` on entry and the
    /// product on return. When `alpha` is zero, `data` is set to zero and neither `T` nor `B`
    /// is read. The workspace size is given by
    /// [`DenseWorkspaceRequest::triangular_product`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `data.rows()` (`Left`) or
    ///   `data.cols()` (`Right`).
    /// - `NonFinite` if `alpha` is NaN or infinite, or if `alpha` is nonzero and a read entry
    ///   of `self` or an entry of `data` is.
    /// - `NumericalInstability` if the product overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn triangular_mul_mat_in_place<'scope, 'capability>(
        &self,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        triangular_product(
            *self,
            &mut data,
            TriangularProduct {
                side,
                triangle,
                diagonal,
                operation,
                alpha,
            },
            options,
            workspace,
            context,
        )
    }
    /// Solves `op(T) X = α B` or `X op(T) = α B` for a triangular matrix `T` stored in `self`.
    ///
    /// `self` is the `n × n` matrix `T`. Only its `triangle` is read, and with
    /// [`DiagonalKind::Unit`](scioxide_core::DiagonalKind::Unit) its diagonal is taken as one.
    /// `operation` selects `op(T)` as `T`, `Tᵀ` or `Tᴴ`. `general` is `B`. The method is
    /// substitution, about `n² p` flops for `p` right-hand sides. When `alpha` is zero, `X = 0`
    /// and neither `T` nor `B` is read. This is
    /// [`triangular_solve_mat_in_place`](Self::triangular_solve_mat_in_place) on a copy of `B`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `general.rows()` (`Left`) or
    ///   `general.cols()` (`Right`).
    /// - `Singular` if `alpha` is nonzero, the diagonal is explicit and a diagonal entry of
    ///   `T` is exactly zero.
    /// - `NonFinite` if `alpha` is NaN or infinite, or if `alpha` is nonzero and a read entry
    ///   of `self` or an entry of `general` is.
    /// - `NumericalInstability` if the solution overflows.
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{DiagonalKind, Triangle};
    /// use scioxide_linalg::{MatrixSide, MatrixView, StructuredOperation};
    /// let t = MatrixView::from_slice(2, 2, &[2.0_f64, 9.0, 1.0, 3.0])?;
    /// let b = MatrixView::from_slice(2, 1, &[2.0, 4.0])?;
    /// // Lower triangle of T = [2 9; 1 3]; T [1; 1] = [2; 4].
    /// let x = t.triangular_solve_mat(
    ///     b,
    ///     MatrixSide::Left,
    ///     Triangle::Lower,
    ///     DiagonalKind::Explicit,
    ///     StructuredOperation::Normal,
    ///     1.0)?;
    /// assert_eq!(x.as_slice(), &[1.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn triangular_solve_mat<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.triangular_solve_mat_with_options(
            general,
            side,
            triangle,
            diagonal,
            operation,
            alpha,
            &crate::LinalgExecutionOptions::default(),
        )
    }

    /// Runs [`triangular_solve_mat`](Self::triangular_solve_mat) with `options`, which set the
    /// product accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`triangular_solve_mat`](Self::triangular_solve_mat), and limit errors from
    /// `options` as described in [Execution](crate#execution).
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn triangular_solve_mat_with_options<'m>(
        &self,
        general: impl Into<MatrixView<'m, S>>,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
        options: &LinalgExecutionOptions,
    ) -> Result<Matrix<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = general.into().try_to_matrix()?;
        self.triangular_solve_mat_in_place(
            side,
            triangle,
            diagonal,
            operation,
            alpha,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
    /// Solves `op(T) X = α B` or `X op(T) = α B` in place for a triangular matrix `T`.
    ///
    /// `self` is the `n × n` matrix `T`, of which only `triangle` is read. With
    /// [`DiagonalKind::Unit`](scioxide_core::DiagonalKind::Unit) its diagonal is taken as one.
    /// `operation` selects `op(T)` as `T`, `Tᵀ` or `Tᴴ`. `data` holds `B` on entry and `X` on
    /// return. The workspace size is given by
    /// [`DenseWorkspaceRequest::triangular_solve`]. When `alpha` is zero, `X = 0` and neither
    /// `T` nor `B` is read.
    ///
    /// When the nonzero magnitudes of `T`, `B` and `alpha` lie within
    /// `[2^(−emax/4), 2^(emax/4)]`, where `2^emax` is the smallest power of two above the
    /// largest finite value, `Standard` [`Accuracy`](scioxide_core::Accuracy) runs blocked
    /// substitution in ordinary arithmetic. `Exact` instead accumulates each residual
    /// `α bᵢ − Σⱼ tᵢⱼ xⱼ` exactly, rounds it once and divides it by the diagonal entry, so its
    /// result is the same on every platform. Outside that range both use an unblocked
    /// substitution that rescales each row by powers of two, and the solve also restarts
    /// from `B` on that path when growth inside the substitution leaves the floating-point
    /// range. The rescaled path costs about the same `n² p` flops without blocking and
    /// rejects only solutions that do not fit in `S`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square of order `data.rows()` (`Left`) or
    ///   `data.cols()` (`Right`).
    /// - `Singular` if `alpha` is nonzero, the diagonal is explicit and a diagonal entry of
    ///   `T` is exactly zero.
    /// - `NonFinite` if `alpha` is NaN or infinite, or if `alpha` is nonzero and a read entry
    ///   of `self` or an entry of `data` is.
    /// - `NumericalInstability` if the solution overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn triangular_solve_mat_in_place<'scope, 'capability>(
        &self,
        side: MatrixSide,
        triangle: Triangle,
        diagonal: DiagonalKind,
        operation: StructuredOperation,
        alpha: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let order = if side == MatrixSide::Left {
            data.rows()
        } else {
            data.cols()
        };
        let request =
            DenseWorkspaceRequest::triangular_solve(data.rows(), order, data.cols(), options)
                .for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let len = scioxide_core::backend::checked_product(
            &[data.rows(), data.cols()],
            scioxide_core::quantity!("triangular_solve_output"),
        )?;
        if len == 0 {
            return kernel::level3::triangular_with_context(
                kernel::level3::TriangularAction::Solve,
                crate::kernel::TriangularOperand {
                    view: *self,
                    triangle,
                    op: map_operation(operation),
                    diagonal,
                },
                data,
                crate::kernel::TriangularUpdate {
                    side,
                    alpha,
                    policy: options.execution_policy(),
                },
                workspace,
                context,
            );
        }
        workspace.with_prefix(len, |scratch, child| {
            for row in 0..data.rows() {
                for col in 0..data.cols() {
                    scratch[row * data.cols() + col] = data.as_view().get_element(row, col);
                }
            }
            let staged = MatrixViewMut::from_slice(data.rows(), data.cols(), scratch)?;
            kernel::level3::triangular_solve_staged(
                crate::kernel::TriangularOperand {
                    view: *self,
                    triangle,
                    op: map_operation(operation),
                    diagonal,
                },
                staged,
                data.as_view(),
                crate::kernel::TriangularUpdate {
                    side,
                    alpha,
                    policy: options.execution_policy(),
                },
                child,
                context,
            )?;
            for row in 0..data.rows() {
                for col in 0..data.cols() {
                    data.set_element(row, col, scratch[row * data.cols() + col]);
                }
            }
            Ok(())
        })
    }
}

impl<S: NumericScalar> MatrixViewMut<'_, S> {
    /// Computes the symmetric rank-k update `C := α op(A) op(A)ᵀ + β C` on one triangle of `C`.
    ///
    /// `data` is the `n × n` matrix `C`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `operation` is `Normal` (`op(A) = A`, `A` is `n × k`) or
    /// `Transpose` (`op(A) = Aᵀ`, `A` is `k × n`).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `operation` is `Adjoint`.
    /// - `DimensionMismatch` if `data` is not `n × n` with `n` the row count of `op(A)`.
    /// - `NonFinite` if `alpha`, `beta`, an entry of `a`, or (for nonzero `beta`) an entry of
    ///   the selected triangle of `data` is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn symmetric_rank_k_in_place<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        triangle: Triangle,
        operation: StructuredOperation,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let a = a.into();
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let shared = if operation == StructuredOperation::Normal {
            a.cols()
        } else {
            a.rows()
        };
        let request = DenseWorkspaceRequest::product(data.rows(), shared, data.cols(), options)
            .for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        kernel::level3::syrk_with_context(
            a,
            data.reborrow(),
            crate::kernel::level3::RankUpdate {
                triangle,
                op: map_operation(operation),
                alpha,
                beta,
            },
            options.execution_policy(),
            workspace,
            context,
        )
    }
    /// Computes the symmetric rank-2k update `C := α op(A) op(B)ᵀ + α op(B) op(A)ᵀ + β C`.
    ///
    /// `data` is the `n × n` matrix `C`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `a` and `b` have the same shape. `operation` is `Normal`
    /// (`op(A) = A`, `n × k`) or `Transpose` (`op(A) = Aᵀ`, `A` is `k × n`).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `data` is not square, `op(A)` does not have `n` rows, or `a`
    ///   and `b` differ in shape.
    /// - `InvalidInput` if `operation` is `Adjoint`.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, if `alpha` is nonzero and an entry
    ///   of `a` or `b` is, or if `beta` is nonzero and an entry of the selected triangle of
    ///   `data` is.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn symmetric_rank_two_k_in_place<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        triangle: Triangle,
        operation: StructuredOperation,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let a = a.into();
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        rank_two_k(
            RankTwoProblem {
                a,
                b: b.into(),
                output: &mut data,
            },
            RankTwoPolicy {
                triangle,
                system: operation,
                hermitian: false,
                alpha,
                beta,
            },
            options,
            workspace,
            context,
        )
    }
    /// Computes the Hermitian rank-k update `C := α op(A) op(A)ᴴ + β C` on one triangle of `C`.
    ///
    /// `data` is the `n × n` matrix `C`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `alpha`, `beta` and the diagonal of `C` must be real, and the
    /// computed diagonal is stored with zero imaginary part. `operation` is `Normal`
    /// (`op(A) = A`, `A` is `n × k`) or `Adjoint` (`op(A) = Aᴴ`, `A` is `k × n`).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `operation` is `Transpose`, if `alpha` or `beta` has a nonzero
    ///   imaginary part, or if `beta` is nonzero and a diagonal entry of `data` is not real.
    /// - `DimensionMismatch` if `data` is not `n × n` with `n` the row count of `op(A)`.
    /// - `NonFinite` if `alpha`, `beta`, an entry of `a`, or (for nonzero `beta`) an entry of
    ///   the selected triangle of `data` is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn hermitian_rank_k_in_place<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        triangle: Triangle,
        operation: StructuredOperation,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let a = a.into();
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let shared = if operation == StructuredOperation::Normal {
            a.cols()
        } else {
            a.rows()
        };
        let request = DenseWorkspaceRequest::product(data.rows(), shared, data.cols(), options)
            .for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        if alpha.imaginary_component() != S::Real::ZERO
            || beta.imaginary_component() != S::Real::ZERO
        {
            return Err(SciOxideError::invalid_input(
                if alpha.imaginary_component() == S::Real::ZERO {
                    scioxide_core::argument!("beta")
                } else {
                    scioxide_core::argument!("alpha")
                },
                scioxide_core::InvalidInputReason::Inconsistent,
                "Hermitian rank-k scales must be real",
            ));
        }
        if data.rows() == data.cols() && beta != S::ZERO {
            for i in 0..data.rows() {
                let value = data.as_view().get_element(i, i);
                crate::checked::validate_input(value, "Hermitian diagonal")?;
                if value.imaginary_component() != S::Real::ZERO {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("data"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "Hermitian diagonal must be real",
                    ));
                }
            }
        }
        kernel::level3::herk_with_context(
            a,
            data.reborrow(),
            crate::kernel::level3::RankUpdate {
                triangle,
                op: map_operation(operation),
                alpha,
                beta,
            },
            options.execution_policy(),
            workspace,
            context,
        )
    }
    /// Computes the Hermitian rank-2k update `C := α op(A) op(B)ᴴ + conj(α) op(B) op(A)ᴴ + β C`.
    ///
    /// `data` is the `n × n` matrix `C`. Only its `triangle` is read and written; the other
    /// triangle is unchanged. `alpha` may be complex, but `beta` and the diagonal of `C` must
    /// be real. `a` and `b` have the same shape. `operation` is `Normal` (`op(A) = A`, `n × k`)
    /// or `Adjoint` (`op(A) = Aᴴ`, `A` is `k × n`).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `data` is not square, `op(A)` does not have `n` rows, or `a`
    ///   and `b` differ in shape.
    /// - `InvalidInput` if `operation` is `Transpose`, if `beta` has a nonzero imaginary part,
    ///   or if `beta` is nonzero and a diagonal entry of `data` is not real.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, if `alpha` is nonzero and an entry
    ///   of `a` or `b` is, or if `beta` is nonzero and an entry of the selected triangle of
    ///   `data` is.
    /// - `NumericalInstability` if the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn hermitian_rank_two_k_in_place<'m, 'scope, 'capability>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        triangle: Triangle,
        operation: StructuredOperation,
        alpha: S,
        beta: S,
        options: &LinalgExecutionOptions,
        mut data: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let a = a.into();
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        rank_two_k(
            RankTwoProblem {
                a,
                b: b.into(),
                output: &mut data,
            },
            RankTwoPolicy {
                triangle,
                system: operation,
                hermitian: true,
                alpha,
                beta,
            },
            options,
            workspace,
            context,
        )
    }
}
