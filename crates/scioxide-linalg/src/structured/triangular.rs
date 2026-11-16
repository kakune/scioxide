//! Triangular solves for dense, band and packed storage.
use super::direct::{StructuredOperation, StructuredWorkspaceRequest, solve_methods};
use super::storage::{BandedMatrixView, PackedKind, PackedMatrixView};
use crate::{KernelContext, LinalgRequirements, LinalgWorkspace, MatrixView};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_core::{NumericScalar, SciOxideError};

use crate::kernel::triangular::Storage;
#[cfg(feature = "blas")]
use crate::{BlasBandMatrix, BlasOperation, BlasPackedMatrix, BlasStructuredMatrix, BlasTransform};
/// Triangular system `T x = b` prepared for repeated solves by substitution.
///
/// The plan borrows one triangle of a dense, band or packed matrix. Entries in
/// the other triangle are not read. With [`DiagonalKind::Unit`] the diagonal is
/// taken to be 1 and the stored diagonal is not read either. Preparation
/// reads each used entry once, in O(n²) operations for dense and packed
/// storage or O(n k) for a band of bandwidth `k`: it checks that the entries
/// are finite and that no diagonal entry is exactly zero, and records the
/// largest entry magnitude, which every solve uses to scale the substitution.
///
/// A solve is forward or back substitution in O(n²) operations for dense and
/// packed storage, or O(n k) for a band of bandwidth `k`. If the substitution
/// would overflow or lose accuracy to underflow before division by a small
/// pivot, it rescales by powers of two and repeats. Substitution is
/// componentwise backward stable (Higham, 2002, §8.1). With the `blas` feature,
/// single right-hand sides go to the attached BLAS provider (TRSV, TBSV or
/// TPSV) when it accepts the operand. Products with a triangular matrix,
/// including a singular one, are provided by the `mul_vec` methods of the
/// storage views and by [`MatrixView::triangular_mul_mat`].
///
/// ```
/// use scioxide_core::{DiagonalKind, Triangle};
/// use scioxide_linalg::{MatrixView, StructuredOperation, TriangularPlan, VectorView};
///
/// // L = [2 0; 1 4] and b = [2, 5] give x = [1, 1] exactly.
/// let a = MatrixView::from_slice(2, 2, &[2.0_f64, 0.0, 1.0, 4.0])?;
/// let plan = TriangularPlan::from_dense(a, Triangle::Lower, DiagonalKind::Explicit)?;
/// let x = plan.solve(StructuredOperation::Normal, VectorView::from_slice(&[2.0, 5.0]))?;
/// assert_eq!(x.as_slice(), &[1.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, Ch. 8.
#[derive(Debug, Clone, Copy)]
pub struct TriangularPlan<'a, S: NumericScalar> {
    storage: Storage<'a, S>,
    triangle: Triangle,
    diagonal: DiagonalKind,
    order: usize,
    bandwidth: usize,
    maximum: S::Real,
}
impl<'a, S: NumericScalar> TriangularPlan<'a, S> {
    #[cfg(feature = "blas")]
    fn provider_operand(&self) -> Option<BlasStructuredMatrix<'a, S>> {
        let triangle = self.triangle;
        match self.storage {
            Storage::Dense(matrix) => {
                crate::kernel::structured_vector::dense_operand(matrix, triangle)
            }
            Storage::Band {
                data,
                stride,
                diagonal,
            } => {
                let (values, lower, upper) = match self.triangle {
                    Triangle::Lower => (&data[diagonal..], self.bandwidth, 0),
                    Triangle::Upper => (data, 0, self.bandwidth),
                };
                Some(BlasStructuredMatrix::Band {
                    matrix: BlasBandMatrix::new(
                        values, self.order, self.order, lower, upper, stride,
                    ),
                    triangle: Some(triangle),
                })
            }
            Storage::Packed(matrix) => Some(BlasStructuredMatrix::Packed(BlasPackedMatrix::new(
                matrix.as_slice(),
                self.order,
                triangle,
            ))),
        }
    }

    #[cfg(feature = "blas")]
    fn provider_operation(&self) -> BlasOperation {
        match self.storage {
            Storage::Dense(_) => BlasOperation::Trsv,
            Storage::Band { .. } => BlasOperation::Tbsv,
            Storage::Packed(_) => BlasOperation::Tpsv,
        }
    }

    #[cfg(feature = "blas")]
    fn provider_transform(operation: StructuredOperation) -> BlasTransform {
        match operation {
            StructuredOperation::Normal => BlasTransform::None,
            StructuredOperation::Transpose => BlasTransform::Transpose,
            StructuredOperation::Adjoint => BlasTransform::Adjoint,
        }
    }

    #[cfg(feature = "blas")]
    fn provider_diagonal(&self) -> DiagonalKind {
        match self.diagonal {
            DiagonalKind::Unit => DiagonalKind::Unit,
            DiagonalKind::Explicit => DiagonalKind::Explicit,
        }
    }
    /// Prepares the `triangle` of a square dense matrix, without copying it.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if an entry of the triangle is NaN or infinite.
    /// - `Singular` if `diagonal` is [`DiagonalKind::Explicit`] and a diagonal
    ///   entry is exactly zero.
    pub fn from_dense(
        matrix: impl Into<MatrixView<'a, S>>,
        triangle: Triangle,
        diagonal: DiagonalKind,
    ) -> Result<Self, SciOxideError> {
        let matrix = matrix.into();
        validate_dense(matrix, triangle, diagonal)?;
        Ok(Self {
            storage: Storage::Dense(matrix),
            triangle,
            diagonal,
            order: matrix.rows(),
            bandwidth: matrix.rows().saturating_sub(1),
            maximum: Storage::Dense(matrix).maximum(
                matrix.rows(),
                triangle,
                diagonal,
                matrix.rows().saturating_sub(1),
            ),
        })
    }
    /// Prepares the `triangle` of a square band matrix, without copying it.
    ///
    /// The bandwidth of the triangular matrix is `matrix.lower()` for
    /// [`Triangle::Lower`] and `matrix.upper()` for [`Triangle::Upper`]; the
    /// band on the other side of the diagonal is not read.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if an entry of the triangle is NaN or infinite.
    /// - `Singular` if `diagonal` is [`DiagonalKind::Explicit`] and a diagonal
    ///   entry is exactly zero.
    pub fn from_band(
        matrix: BandedMatrixView<'a, S>,
        triangle: Triangle,
        diagonal: DiagonalKind,
    ) -> Result<Self, SciOxideError> {
        if matrix.rows() != matrix.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [matrix.rows(), matrix.rows()],
                [matrix.rows(), matrix.cols()],
            ));
        }
        Self::prepare(
            Storage::Band {
                data: matrix.as_slice(),
                stride: matrix.leading_dimension(),
                diagonal: matrix.upper(),
            },
            triangle,
            diagonal,
            matrix.rows(),
            match triangle {
                Triangle::Lower => matrix.lower(),
                Triangle::Upper => matrix.upper(),
            },
        )
    }
    /// Prepares a packed triangular matrix, without copying it.
    ///
    /// # Errors
    /// - `InvalidInput` if `matrix.kind()` is [`PackedKind::Hermitian`].
    /// - `NonFinite` if a stored entry is NaN or infinite.
    /// - `Singular` if `diagonal` is [`DiagonalKind::Explicit`] and a diagonal
    ///   entry is exactly zero.
    pub fn from_packed(
        matrix: PackedMatrixView<'a, S>,
        diagonal: DiagonalKind,
    ) -> Result<Self, SciOxideError> {
        if matrix.kind() != PackedKind::Triangular {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("matrix"),
                "packed triangular interpretation required",
            ));
        }
        Self::prepare(
            Storage::Packed(matrix),
            matrix.triangle(),
            diagonal,
            matrix.order(),
            matrix.order().saturating_sub(1),
        )
    }
    fn prepare(
        storage: Storage<'a, S>,
        triangle: Triangle,
        diagonal: DiagonalKind,
        order: usize,
        bandwidth: usize,
    ) -> Result<Self, SciOxideError> {
        let mut plan = Self {
            storage,
            triangle,
            diagonal,
            order,
            bandwidth,
            maximum: <S::Real as scioxide_core::AlgebraicScalar>::ZERO,
        };
        validate_coefficients(order, bandwidth, triangle, |i, j| plan.element(i, j))?;
        plan.maximum = storage.maximum(order, triangle, diagonal, bandwidth);
        Ok(plan)
    }
    /// Returns the order `n`.
    pub fn order(&self) -> usize {
        self.order
    }
    fn element(&self, i: usize, j: usize) -> S {
        if i == j && self.diagonal == DiagonalKind::Unit {
            return S::ONE;
        }
        match self.storage {
            Storage::Dense(a) => a.get_element(i, j),
            Storage::Band {
                data,
                stride,
                diagonal,
            } => data[j * stride + diagonal + i - j],
            Storage::Packed(a) => a.element(i, j),
        }
    }
    fn solve_slice(
        &self,
        x: &mut [S],
        op: StructuredOperation,
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::kernel::triangular::solve(
            crate::kernel::TriangularOperand {
                view: self.storage,
                triangle: self.triangle,
                op: match op {
                    StructuredOperation::Normal => crate::kernel::GemmOp::None,
                    StructuredOperation::Transpose => crate::kernel::GemmOp::Transpose,
                    StructuredOperation::Adjoint => crate::kernel::GemmOp::Adjoint,
                },
                diagonal: self.diagonal,
            },
            self.maximum,
            x,
            self.bandwidth,
            kernels.dispatch(),
        )
    }

    // A vector solve may route to an injected provider; a declined call falls
    // back to native substitution on the same, still unmodified, candidate.
    fn solve_vector(
        &self,
        x: &mut [S],
        op: StructuredOperation,
        kernels: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        #[cfg(feature = "blas")]
        if let Some(dispatch) = kernels.blas() {
            let route = if let Some(matrix) = self.provider_operand() {
                crate::blas::dispatch::triangular_vector(
                    crate::blas::dispatch::TriangularVectorRequest {
                        blas_operation: self.provider_operation(),
                        transform: Self::provider_transform(op),
                        diagonal: self.provider_diagonal(),
                        a: matrix,
                        candidate: x,
                        dispatch,
                        accuracy: kernels.accuracy(),
                        workers: crate::kernel::provider_policy_workers(kernels),
                    },
                )?
            } else {
                crate::blas::dispatch::unsupported(dispatch)?
            };
            if route == crate::blas::dispatch::BlasRoute::Provider {
                return Ok(());
            }
        }
        self.solve_slice(x, op, kernels)
    }

    /// Returns the workspace size for a solve `request`, including a condition estimate.
    ///
    /// The workspace needs scalars only, so a slice you provide can serve as
    /// workspace through [`LinalgWorkspace::from_scalar_slice`].
    ///
    /// ```
    /// use scioxide_core::{DiagonalKind, Triangle};
    /// use scioxide_linalg::{
    ///     LinalgExecutionOptions, MatrixView, StructuredWorkspaceRequest, TriangularPlan,
    /// };
    /// let a = MatrixView::from_slice(2, 2, &[2.0_f64, 0.0, 1.0, 4.0])?;
    /// let plan = TriangularPlan::from_dense(a, Triangle::Lower, DiagonalKind::Explicit)?;
    /// let request = StructuredWorkspaceRequest::solve_matrix(2, 3, &LinalgExecutionOptions::new());
    /// assert_eq!(plan.workspace_requirements(&request)?.scalar_len()?, 10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `InvalidInput` if `request` is a factor request.
    /// - `DimensionMismatch` if `request` is for another order.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the workspace exceeds the byte limit of `request`.
    pub fn workspace_requirements(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let (columns, byte_limit) = request.plan_solve(self.order, None)?;
        super::direct::solve_requirements::<S>(self.order, columns, byte_limit)
    }

    /// Allocates a workspace for a solve `request`.
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    solve_methods!(
        TriangularPlan,
        order,
        solve_vector,
        "use scioxide_linalg::{MatrixView, StructuredOperation, TriangularPlan};
use scioxide_core::{DiagonalKind, Triangle};
let a = MatrixView::from_slice(2, 2, &[2.0_f64, 0.0, 1.0, 2.0])?;
let factor = TriangularPlan::from_dense(a, Triangle::Lower, DiagonalKind::Explicit)?;"
    );
}
fn validate_coefficients<S: NumericScalar>(
    order: usize,
    bandwidth: usize,
    triangle: Triangle,
    element: impl Fn(usize, usize) -> S,
) -> Result<(), SciOxideError> {
    for i in 0..order {
        let range = match triangle {
            Triangle::Lower => i.saturating_sub(bandwidth)..i + 1,
            Triangle::Upper => i..i.saturating_add(bandwidth).saturating_add(1).min(order),
        };
        for j in range.clone() {
            crate::checked::validate_input(element(i, j), "triangular coefficient")?;
        }
        if element(i, i) == S::ZERO {
            return Err(SciOxideError::singular(Some(i)));
        }
    }
    Ok(())
}

pub(crate) fn validate_dense<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    triangle: Triangle,
    diagonal: DiagonalKind,
) -> Result<(), SciOxideError> {
    if matrix.rows() != matrix.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [matrix.rows(), matrix.rows()],
            [matrix.rows(), matrix.cols()],
        ));
    }
    validate_coefficients(
        matrix.rows(),
        matrix.rows().saturating_sub(1),
        triangle,
        |i, j| {
            if i == j && diagonal == DiagonalKind::Unit {
                S::ONE
            } else {
                matrix.get_element(i, j)
            }
        },
    )
}

impl<S: NumericScalar> TriangularPlan<'_, S> {
    fn bandwidths(&self) -> (usize, usize) {
        if self.triangle == Triangle::Lower {
            (self.bandwidth, 0)
        } else {
            (0, self.bandwidth)
        }
    }
    fn rank(
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
        let (lower, upper) = self.bandwidths();
        crate::diagnostics::one_norm(
            crate::diagnostics::Band::new(self.order, lower, upper),
            |i, j| self.element(i, j),
        )
    }
    fn residual(
        &self,
        op: StructuredOperation,
        x: &[S],
        b: crate::VectorView<'_, S>,
        context: crate::KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        let (lower, upper) = self.bandwidths();
        super::direct::residual(
            crate::diagnostics::Band::new(self.order, lower, upper),
            op,
            |i, j| self.element(i, j),
            x,
            b,
            context,
        )
    }
    fn solve_scaled(
        &self,
        x: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        crate::diagnostics::substitute(
            x,
            self.triangle,
            self.diagonal,
            self.bandwidth,
            adjoint,
            |i, j| self.element(i, j),
        )
    }
}
