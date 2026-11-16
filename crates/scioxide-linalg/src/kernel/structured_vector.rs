//! Level 2 matrix-vector products for band, packed, symmetric and Hermitian storage.
//!
//! `Standard` accuracy uses ordinary arithmetic with power-of-two rescaling.
//! `Exact` accumulates each entry exactly and rounds once. Owned forms use
//! `Standard`.

struct StructuredMatrix<B, E> {
    rows: usize,
    cols: usize,
    bounds: B,
    element: E,
}
struct VectorUpdate<'a, 'b, S: NumericScalar> {
    alpha: S,
    x: VectorView<'a, S>,
    beta: S,
    y: VectorViewMut<'b, S>,
}

use core::ops::Range;
use scioxide_core::AlgebraicScalar;

use crate::kernel::standard;
use scioxide_core::{Dimensions, NumericScalar, ProductSum, RealScalar, SciOxideError};

use crate::{
    BandedMatrixView, DenseWorkspaceRequest, HermitianBandMatrixView, KernelContext,
    LinalgExecutionOptions, LinalgWorkspace, MatrixView, PackedMatrixView, StructuredOperation,
    Vector, VectorView, VectorViewMut,
};
use scioxide_core::Triangle;

#[cfg(feature = "blas")]
use crate::{
    BlasBandMatrix, BlasLayout, BlasMatrix, BlasOperation, BlasPackedMatrix, BlasStructuredMatrix,
    BlasTransform, PackedKind,
};

#[cfg(feature = "blas")]
type ProviderOperand<'a, S> = crate::blas::dispatch::StructuredMatVec<'a, S>;
#[cfg(not(feature = "blas"))]
type ProviderOperand<'a, S> = core::marker::PhantomData<&'a S>;

#[cfg(feature = "blas")]
macro_rules! provider_operand {
    ($expression:expr) => {
        $expression
    };
}
#[cfg(not(feature = "blas"))]
macro_rules! provider_operand {
    ($expression:expr) => {
        None
    };
}

#[cfg(feature = "blas")]
fn blas_transform(operation: StructuredOperation) -> BlasTransform {
    match operation {
        StructuredOperation::Normal => BlasTransform::None,
        StructuredOperation::Transpose => BlasTransform::Transpose,
        StructuredOperation::Adjoint => BlasTransform::Adjoint,
    }
}

/// Direct provider encoding of a dense row- or column-major triangle, or
/// `None` when the view needs packing (strided or conjugated complex).
#[cfg(feature = "blas")]
pub(crate) fn dense_operand<'a, S: NumericScalar>(
    matrix: MatrixView<'a, S>,
    triangle: Triangle,
) -> Option<BlasStructuredMatrix<'a, S>> {
    if matrix.conjugated && !S::IS_REAL {
        return None;
    }
    let (rows, cols, leading_dimension, layout) =
        if matrix.col_stride == 1 && matrix.row_stride >= matrix.cols.max(1) {
            (
                matrix.rows,
                matrix.cols,
                matrix.row_stride.max(matrix.cols.max(1)),
                BlasLayout::RowMajor,
            )
        } else if matrix.row_stride == 1 && matrix.col_stride >= matrix.rows.max(1) {
            (
                matrix.rows,
                matrix.cols,
                matrix.col_stride.max(matrix.rows.max(1)),
                BlasLayout::ColumnMajor,
            )
        } else {
            return None;
        };
    Some(BlasStructuredMatrix::Dense {
        matrix: BlasMatrix::new(matrix.data, rows, cols, leading_dimension, layout),
        triangle: (triangle),
    })
}

#[cfg(feature = "blas")]
fn self_adjoint_operation<S: NumericScalar>(
    hermitian: bool,
    band: bool,
    packed: bool,
) -> BlasOperation {
    let complex = matches!(
        S::kind(),
        scioxide_core::backend::ScalarKind::Complex32
            | scioxide_core::backend::ScalarKind::Complex64
    );
    if packed {
        if hermitian && complex {
            BlasOperation::Hpmv
        } else {
            BlasOperation::Spmv
        }
    } else if band {
        if hermitian && complex {
            BlasOperation::Hbmv
        } else {
            BlasOperation::Sbmv
        }
    } else if hermitian && complex {
        BlasOperation::Hemv
    } else {
        BlasOperation::Symv
    }
}

fn matvec<S: NumericScalar>(
    matrix: StructuredMatrix<impl Fn(usize) -> Range<usize>, impl Fn(usize, usize) -> S>,
    update: VectorUpdate<'_, '_, S>,
    scratch: &mut [S],
    policy: scioxide_core::ExecutionPolicy,
    context: KernelContext<'_>,
    provider: Option<ProviderOperand<'_, S>>,
) -> Result<(), SciOxideError> {
    let StructuredMatrix {
        rows,
        cols,
        bounds,
        element,
    } = matrix;
    let VectorUpdate {
        alpha,
        x,
        beta,
        mut y,
    } = update;

    #[cfg(not(feature = "blas"))]
    let _ = policy;
    if x.len() != cols || y.len() != rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(cols, rows),
            Dimensions::two(x.len(), y.len()),
        ));
    }
    if scratch.len() < rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(rows),
            Dimensions::one(scratch.len()),
        ));
    }
    crate::checked::validate_input(alpha, "alpha")?;
    crate::checked::validate_input(beta, "beta")?;
    let input_maximum = if alpha != S::ZERO {
        x.try_finite()?.maximum
    } else {
        S::Real::ZERO
    };
    let prior_maximum = if beta != S::ZERO {
        y.as_view().try_finite()?.maximum
    } else {
        S::Real::ZERO
    };
    let mut matrix_maximum = S::Real::ZERO;
    let mut matrix_minimum = S::Real::ZERO;
    if alpha != S::ZERO {
        for row in 0..rows {
            for col in bounds(row) {
                let value = element(row, col);
                crate::checked::validate_input(value, "matrix coefficient")?;
                let magnitude = standard::magnitude(value);
                matrix_maximum = matrix_maximum.max(magnitude);
                if magnitude != S::Real::ZERO
                    && (matrix_minimum == S::Real::ZERO || magnitude < matrix_minimum)
                {
                    matrix_minimum = magnitude;
                }
            }
        }
    }
    context.execution().check()?;
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas() {
        for (row, slot) in scratch.iter_mut().enumerate().take(rows) {
            *slot = if beta == S::ZERO {
                S::ZERO
            } else {
                y.as_view().get_element(row)
            };
        }
        let route = if let Some(provider) = provider {
            crate::blas::dispatch::structured_matvec(crate::blas::dispatch::MatVecRequest {
                request: provider,
                alpha,
                x,
                beta,
                candidate: &mut scratch[..rows],
                dispatch,
                accuracy: context.accuracy(),
                workers: crate::kernel::provider_worker_bound::<S>(rows, cols, 1, policy, context)?,
            })?
        } else {
            crate::blas::dispatch::unsupported(dispatch)?
        };
        if route == crate::blas::dispatch::BlasRoute::Provider {
            for (output, candidate) in y.iter_mut().zip(scratch.iter().copied()) {
                *output = candidate;
            }
            return Ok(());
        }
    }
    #[cfg(not(feature = "blas"))]
    let _ = provider;
    if context.accuracy() == scioxide_core::Accuracy::Standard {
        let frame = standard::AffineProduct::new(
            alpha,
            (matrix_maximum, matrix_minimum),
            input_maximum,
            cols,
            beta,
            prior_maximum,
        )?;
        for (row, candidate) in scratch[..rows].iter_mut().enumerate() {
            let prior = if beta == S::ZERO {
                S::ZERO
            } else {
                y.as_view().get_element(row)
            };
            *candidate = frame.apply(
                prior,
                bounds(row)
                    .filter(|_| alpha != S::ZERO)
                    .map(|col| (element(row, col), x.get_element(col))),
            )?;
        }
        y.iter_mut()
            .zip(scratch.iter().copied())
            .for_each(|(out, value)| *out = value);
        return Ok(());
    }
    for (row, candidate) in scratch.iter_mut().enumerate().take(rows) {
        let mut sum = ProductSum::new();
        if alpha != S::ZERO {
            for col in bounds(row) {
                let value = element(row, col);
                sum.add_triple_product(alpha, value, x.get_element(col))?;
            }
        }
        if beta != S::ZERO {
            sum.add_product(beta, y.as_view().get_element(row))?;
        }
        *candidate = sum.finish()?;
    }
    for (output, candidate) in y.iter_mut().zip(scratch.iter().copied()) {
        *output = candidate;
    }
    Ok(())
}

fn upper_lower_window(row: usize, cols: usize, lower: usize, upper: usize) -> Range<usize> {
    row.saturating_sub(lower)..row.saturating_add(upper).saturating_add(1).min(cols)
}

impl<S: NumericScalar> BandedMatrixView<'_, S> {
    /// Computes `y := α op(A) x + β y` for a general band matrix `A`.
    ///
    /// `operation` selects `op(A)` as `A`, `Aᵀ` or `Aᴴ`. Entries outside the band are zero,
    /// and padding in the band storage is not read.
    /// When `alpha` is zero, `A` and `x` are not read. When `beta` is zero, the prior `y` is
    /// not read. Under `Accuracy::Exact` each entry of `y` is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `y.len()` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the length of `x` is not the column count of `op(A)` or the
    ///   length of `y` is not its row count.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, or if a read entry of `A`, `x` or
    ///   `y` is.
    /// - `NumericalInstability` if an entry of the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn mul_vec_in_place<'v, 'scope, 'a>(
        &self,
        operation: StructuredOperation,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        y: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let request = DenseWorkspaceRequest::vector(y.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        let (rows, cols, lower, upper) = match operation {
            StructuredOperation::Normal => (self.rows(), self.cols(), self.lower(), self.upper()),
            StructuredOperation::Transpose | StructuredOperation::Adjoint => {
                (self.cols(), self.rows(), self.upper(), self.lower())
            }
        };
        matvec(
            StructuredMatrix {
                rows,
                cols,
                bounds: |row| upper_lower_window(row, cols, lower, upper),
                element: |row, col| match operation {
                    StructuredOperation::Normal => self.element(row, col),
                    StructuredOperation::Transpose => self.element(col, row),
                    StructuredOperation::Adjoint => self.element(col, row).conj(),
                },
            },
            VectorUpdate { alpha, x, beta, y },
            scratch,
            options.execution_policy(),
            context,
            provider_operand!(Some(crate::blas::dispatch::StructuredMatVec {
                operation: BlasOperation::Gbmv,
                transform: blas_transform(operation),
                matrix: BlasStructuredMatrix::Band {
                    matrix: BlasBandMatrix::new(
                        self.as_slice(),
                        self.rows(),
                        self.cols(),
                        self.lower(),
                        self.upper(),
                        self.leading_dimension()
                    ),
                    triangle: None
                },
            })),
        )
    }
}

impl<S: NumericScalar> HermitianBandMatrixView<'_, S> {
    /// Computes `y := α A x + β y` for a Hermitian band matrix `A`.
    ///
    /// `A` is `n × n`, stored as one triangle with its band. The other triangle is the
    /// conjugate transpose.
    /// When `alpha` is zero, `A` and `x` are not read. When `beta` is zero, the prior `y` is
    /// not read. Under `Accuracy::Exact` each entry of `y` is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `y.len()` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `y` does not have length `n`.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, or if a read entry of `A`, `x` or
    ///   `y` is.
    /// - `NumericalInstability` if an entry of the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn mul_vec_in_place<'v, 'scope, 'a>(
        &self,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        y: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let request = DenseWorkspaceRequest::vector(y.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        let n = self.order();
        matvec(
            StructuredMatrix {
                rows: n,
                cols: n,
                bounds: |row| upper_lower_window(row, n, self.bandwidth(), self.bandwidth()),
                element: |row, col| self.element(row, col),
            },
            VectorUpdate { alpha, x, beta, y },
            scratch,
            options.execution_policy(),
            context,
            provider_operand!(Some(crate::blas::dispatch::StructuredMatVec {
                operation: self_adjoint_operation::<S>(true, true, false),
                transform: BlasTransform::None,
                matrix: BlasStructuredMatrix::Band {
                    matrix: BlasBandMatrix::new(
                        self.as_slice(),
                        n,
                        n,
                        if self.triangle() == Triangle::Lower {
                            self.bandwidth()
                        } else {
                            0
                        },
                        if self.triangle() == Triangle::Upper {
                            self.bandwidth()
                        } else {
                            0
                        },
                        self.bandwidth() + 1
                    ),
                    triangle: Some(self.triangle())
                },
            })),
        )
    }
}

impl<S: NumericScalar> PackedMatrixView<'_, S> {
    /// Computes `y := α op(A) x + β y` for a packed triangular or Hermitian matrix `A`.
    ///
    /// For [`PackedKind::Triangular`](crate::PackedKind::Triangular) the unstored triangle is
    /// zero, and a singular triangle is allowed. For
    /// [`PackedKind::Hermitian`](crate::PackedKind::Hermitian) it is the conjugate transpose of
    /// the stored triangle. `operation` selects `op(A)` as `A`, `Aᵀ` or `Aᴴ`.
    /// When `alpha` is zero, `A` and `x` are not read. When `beta` is zero, the prior `y` is
    /// not read. Under `Accuracy::Exact` each entry of `y` is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `y.len()` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` or `y` does not have length `n`.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, or if a read entry of `A`, `x` or
    ///   `y` is.
    /// - `NumericalInstability` if an entry of the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn mul_vec_in_place<'v, 'scope, 'a>(
        &self,
        operation: StructuredOperation,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        y: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let request = DenseWorkspaceRequest::vector(y.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);
        let n = self.order();
        matvec(
            StructuredMatrix {
                rows: n,
                cols: n,
                bounds: |_| 0..n,
                element: |row, col| match operation {
                    StructuredOperation::Normal => self.element(row, col),
                    StructuredOperation::Transpose => self.element(col, row),
                    StructuredOperation::Adjoint => self.element(col, row).conj(),
                },
            },
            VectorUpdate { alpha, x, beta, y },
            scratch,
            options.execution_policy(),
            context,
            provider_operand!(if self.kind() == PackedKind::Hermitian
                && operation != StructuredOperation::Transpose
            {
                Some(crate::blas::dispatch::StructuredMatVec {
                    operation: self_adjoint_operation::<S>(true, false, true),
                    transform: BlasTransform::None,
                    matrix: BlasStructuredMatrix::Packed(BlasPackedMatrix::new(
                        self.as_slice(),
                        n,
                        self.triangle(),
                    )),
                })
            } else {
                None
            }),
        )
    }
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Computes `y := α A x + β y` for a symmetric matrix `A` stored in one triangle of `self`.
    ///
    /// Only `triangle` is read; the other triangle may hold any value, including NaN.
    /// When `alpha` is zero, `A` and `x` are not read. When `beta` is zero, the prior `y` is
    /// not read. Under `Accuracy::Exact` each entry of `y` is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `y.len()` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square or `x` or `y` does not have length `n`.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, or if a read entry of `A`, `x` or
    ///   `y` is.
    /// - `NumericalInstability` if an entry of the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn symmetric_mul_vec_in_place<'v, 'scope, 'a>(
        &self,
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        y: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let request = DenseWorkspaceRequest::vector(y.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);

        if self.rows() != self.cols() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(self.rows(), self.rows()),
                Dimensions::two(self.rows(), self.cols()),
            ));
        }
        let n = self.rows();
        matvec(
            StructuredMatrix {
                rows: n,
                cols: n,
                bounds: |_| 0..n,
                element: |row, col| match triangle {
                    Triangle::Lower if row < col => self.get_element(col, row),
                    Triangle::Upper if row > col => self.get_element(col, row),
                    _ => self.get_element(row, col),
                },
            },
            VectorUpdate { alpha, x, beta, y },
            scratch,
            options.execution_policy(),
            context,
            provider_operand!(dense_operand(*self, triangle).map(|matrix| {
                crate::blas::dispatch::StructuredMatVec {
                    operation: self_adjoint_operation::<S>(false, false, false),
                    transform: BlasTransform::None,
                    matrix,
                }
            })),
        )
    }

    /// Computes `y := α A x + β y` for a Hermitian matrix `A` stored in one triangle of `self`.
    ///
    /// Only `triangle` is read; the other triangle may hold any value, including NaN. The
    /// diagonal must be real.
    /// When `alpha` is zero, `A` and `x` are not read. When `beta` is zero, the prior `y` is
    /// not read. Under `Accuracy::Exact` each entry of `y` is rounded once. Reserve
    /// [`DenseWorkspaceRequest::vector`] with length `y.len()` in `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square or `x` or `y` does not have length `n`.
    /// - `InvalidInput` if `alpha` is nonzero and a diagonal entry has a nonzero imaginary part.
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite, or if a read entry of `A`, `x` or
    ///   `y` is.
    /// - `NumericalInstability` if an entry of the result overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn hermitian_mul_vec_in_place<'v, 'scope, 'a>(
        &self,
        triangle: Triangle,
        alpha: S,
        x: impl Into<VectorView<'v, S>>,
        beta: S,
        options: &LinalgExecutionOptions,
        y: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let x = x.into();
        let request = DenseWorkspaceRequest::vector(y.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
        let scratch = workspace.scalar_mut(crate::kernel::CANDIDATE);

        if self.rows() != self.cols() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(self.rows(), self.rows()),
                Dimensions::two(self.rows(), self.cols()),
            ));
        }
        let n = self.rows();
        if alpha != S::ZERO {
            for row in 0..n {
                let diagonal = self.get_element(row, row);
                crate::checked::validate_input(diagonal, "Hermitian diagonal")?;
                if diagonal.imaginary_component() != S::Real::ZERO {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("self"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "Hermitian diagonal must be real",
                    ));
                }
            }
        }
        matvec(
            StructuredMatrix {
                rows: n,
                cols: n,
                bounds: |_| 0..n,
                element: |row, col| match triangle {
                    Triangle::Lower if row < col => self.get_element(col, row).conj(),
                    Triangle::Upper if row > col => self.get_element(col, row).conj(),
                    _ => self.get_element(row, col),
                },
            },
            VectorUpdate { alpha, x, beta, y },
            scratch,
            options.execution_policy(),
            context,
            provider_operand!(dense_operand(*self, triangle).map(|matrix| {
                crate::blas::dispatch::StructuredMatVec {
                    operation: self_adjoint_operation::<S>(true, false, false),
                    transform: BlasTransform::None,
                    matrix,
                }
            })),
        )
    }
}

impl<S: NumericScalar> BandedMatrixView<'_, S> {
    /// Returns `op(A) x` for a general band matrix `A`.
    ///
    /// This is [`mul_vec_in_place`](Self::mul_vec_in_place) with `α = 1` and `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the length of `x` is not the column count of `op(A)`.
    /// - `NonFinite` if a stored band entry or an entry of `x` is NaN or infinite.
    /// - `NumericalInstability` if an entry of the result overflows.
    pub fn mul_vec<'v>(
        &self,
        operation: StructuredOperation,
        x: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.mul_vec_with_options(operation, x, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`mul_vec`](Self::mul_vec) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`mul_vec`](Self::mul_vec), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn mul_vec_with_options<'v>(
        &self,
        operation: StructuredOperation,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Vector::zeros(if operation == StructuredOperation::Normal {
            self.rows()
        } else {
            self.cols()
        })?;
        self.mul_vec_in_place(
            operation,
            S::ONE,
            x,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
}

impl<S: NumericScalar> PackedMatrixView<'_, S> {
    /// Returns `op(A) x` for a packed triangular or Hermitian matrix `A`.
    ///
    /// This is [`mul_vec_in_place`](Self::mul_vec_in_place) with `α = 1` and `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` does not have length `n`.
    /// - `NonFinite` if a stored entry or an entry of `x` is NaN or infinite.
    /// - `NumericalInstability` if an entry of the result overflows.
    pub fn mul_vec<'v>(
        &self,
        operation: StructuredOperation,
        x: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.mul_vec_with_options(operation, x, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`mul_vec`](Self::mul_vec) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`mul_vec`](Self::mul_vec), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn mul_vec_with_options<'v>(
        &self,
        operation: StructuredOperation,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Vector::zeros(self.order())?;
        self.mul_vec_in_place(
            operation,
            S::ONE,
            x,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
}

impl<S: NumericScalar> HermitianBandMatrixView<'_, S> {
    /// Returns `A x` for a Hermitian band matrix `A`.
    ///
    /// This is [`mul_vec_in_place`](Self::mul_vec_in_place) with `α = 1` and `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` does not have length `n`.
    /// - `NonFinite` if a stored entry or an entry of `x` is NaN or infinite.
    /// - `NumericalInstability` if an entry of the result overflows.
    pub fn mul_vec<'v>(&self, x: impl Into<VectorView<'v, S>>) -> Result<Vector<S>, SciOxideError> {
        self.mul_vec_with_options(x, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`mul_vec`](Self::mul_vec) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`mul_vec`](Self::mul_vec), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn mul_vec_with_options<'v>(
        &self,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Vector::zeros(self.order())?;
        self.mul_vec_in_place(
            S::ONE,
            x,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Returns `A x` for a symmetric matrix `A` stored in one triangle of `self`.
    ///
    /// This is [`symmetric_mul_vec_in_place`](Self::symmetric_mul_vec_in_place) with `α = 1`
    /// and `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square or `x` does not have length `n`.
    /// - `NonFinite` if a read entry of `self` or an entry of `x` is NaN or infinite.
    /// - `NumericalInstability` if an entry of the result overflows.
    pub fn symmetric_mul_vec<'v>(
        &self,
        triangle: Triangle,
        x: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.symmetric_mul_vec_with_options(triangle, x, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`symmetric_mul_vec`](Self::symmetric_mul_vec) with `options`, which set the product
    /// accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`symmetric_mul_vec`](Self::symmetric_mul_vec), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn symmetric_mul_vec_with_options<'v>(
        &self,
        triangle: Triangle,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Vector::zeros(self.rows())?;
        self.symmetric_mul_vec_in_place(
            triangle,
            S::ONE,
            x,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
}

impl<S: NumericScalar> MatrixView<'_, S> {
    /// Returns `A x` for a Hermitian matrix `A` stored in one triangle of `self`.
    ///
    /// This is [`hermitian_mul_vec_in_place`](Self::hermitian_mul_vec_in_place) with `α = 1`
    /// and `β = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` is not square or `x` does not have length `n`.
    /// - `InvalidInput` if a diagonal entry of `self` has a nonzero imaginary part.
    /// - `NonFinite` if a read entry of `self` or an entry of `x` is NaN or infinite.
    /// - `NumericalInstability` if an entry of the result overflows.
    pub fn hermitian_mul_vec<'v>(
        &self,
        triangle: Triangle,
        x: impl Into<VectorView<'v, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.hermitian_mul_vec_with_options(triangle, x, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`hermitian_mul_vec`](Self::hermitian_mul_vec) with `options`, which set the product
    /// accuracy and the limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`hermitian_mul_vec`](Self::hermitian_mul_vec), and:
    ///
    /// - `ResourceLimit` or `AllocationFailure` if scratch exceeds the byte limit in `options`
    ///   or cannot be allocated.
    pub fn hermitian_mul_vec_with_options<'v>(
        &self,
        triangle: Triangle,
        x: impl Into<VectorView<'v, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Vector<S>, SciOxideError> {
        let options = options.with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);

        let context = crate::LinalgExecutionContext::new(
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .with_options(&options)?;
        let mut output = Vector::zeros(self.rows())?;
        self.hermitian_mul_vec_in_place(
            triangle,
            S::ONE,
            x,
            S::ZERO,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(output)
    }
}
