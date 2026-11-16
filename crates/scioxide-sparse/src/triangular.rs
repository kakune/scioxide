use crate::factor_arithmetic::Arithmetic;
use crate::{CsrMatrix, SparseOperation};
use scioxide_core::{
    DiagonalKind, Dimensions, ExecutionContext, InvalidInputReason, NumericScalar, SciOxideError,
    Triangle,
};

/// Holds a sparse lower or upper triangular matrix `T` and solves triangular
/// systems with it.
///
/// Solves with `T`, `Tᵀ` and `Tᴴ` use forward or back substitution on the
/// same CSR storage; no transposed copy is made. With
/// [`DiagonalKind::Unit`] the diagonal is implicitly one and must not be
/// stored. Solve with `solve`, `solve_into` or `solve_block_into`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{DiagonalKind, ExecutionContext, Triangle};
/// use scioxide_sparse::{
///     CsrMatrix, SparseOperation, SparseSolveOptions, SparseSolveWorkspace, SparseTriangular,
/// };
/// // L = [[2, 0], [1, 1]]: solve L x = [2, 3], then Lᵀ x = [5, 1].
/// let l = CsrMatrix::from_raw(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![2.0_f64, 1.0, 1.0])?;
/// let factor = SparseTriangular::new(l, Triangle::Lower, DiagonalKind::Explicit)?;
/// let mut scratch = vec![0.0; factor.workspace_requirements(1)?.scalar_len()];
/// let mut workspace = SparseSolveWorkspace::borrowed(&mut scratch);
/// let (options, mut x) = (SparseSolveOptions::new(), [0.0; 2]);
/// factor.solve_into(
///     &[2.0, 3.0],
///     SparseOperation::Normal,
///     options,
///     &mut x,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(x, [1.0, 2.0]);
/// factor.solve_into(
///     &[5.0, 1.0],
///     SparseOperation::Transpose,
///     options,
///     &mut x,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(x, [2.0, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Construction rejects entries outside the selected triangle and zero or
/// missing explicit pivots:
///
/// ```
/// use scioxide_core::{DiagonalKind, SciOxideErrorKind, Triangle};
/// use scioxide_sparse::{CsrMatrix, SparseTriangular};
/// // [[1, 1], [0, 0]] stores (0, 1) above the diagonal and a zero pivot at (1, 1).
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 1.0, 0.0])?;
/// let lower = SparseTriangular::new(a.clone(), Triangle::Lower, DiagonalKind::Explicit);
/// assert_eq!(lower.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
/// let upper = SparseTriangular::new(a, Triangle::Upper, DiagonalKind::Explicit);
/// assert_eq!(upper.unwrap_err().kind(), SciOxideErrorKind::Singular);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SparseTriangular<S: NumericScalar> {
    matrix: CsrMatrix<S>,
    triangle: Triangle,
    diagonal: DiagonalKind,
}
crate::factor_solve::impl_solves!(SparseTriangular);
impl<S: NumericScalar> SparseTriangular<S> {
    /// Creates a triangular matrix from `matrix` after checking that every
    /// stored entry lies in `triangle`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `InvalidInput` if an entry lies outside `triangle`, or a diagonal entry
    ///   is stored although `diagonal` is [`DiagonalKind::Unit`].
    /// - `Singular` if `diagonal` is [`DiagonalKind::Explicit`] and a diagonal
    ///   entry is missing or zero.
    pub fn new(
        matrix: CsrMatrix<S>,
        triangle: Triangle,
        diagonal: DiagonalKind,
    ) -> Result<Self, SciOxideError> {
        if matrix.rows() != matrix.cols() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(matrix.rows(), matrix.rows()),
                Dimensions::two(matrix.rows(), matrix.cols()),
            ));
        }
        for row in 0..matrix.rows() {
            let bounds = matrix.row_ptr()[row]..matrix.row_ptr()[row + 1];
            let mut diagonal_value = None;
            for position in bounds {
                let col = matrix.col_indices()[position];
                let outside = match triangle {
                    Triangle::Lower => col > row,
                    Triangle::Upper => col < row,
                };
                if outside {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("matrix"),
                        InvalidInputReason::Malformed,
                        "stored entry lies outside the selected triangle",
                    ));
                }
                if col == row {
                    diagonal_value = Some(matrix.values()[position]);
                }
            }
            match (diagonal, diagonal_value) {
                (DiagonalKind::Explicit, None) => {
                    return Err(SciOxideError::singular(Some(row)));
                }
                (DiagonalKind::Explicit, Some(value)) if value == S::ZERO => {
                    return Err(SciOxideError::singular(Some(row)));
                }
                (DiagonalKind::Unit, Some(_)) => {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("matrix"),
                        InvalidInputReason::Inconsistent,
                        "a unit-diagonal matrix must not store diagonal entries",
                    ));
                }
                _ => {}
            }
        }
        Ok(Self {
            matrix,
            triangle,
            diagonal,
        })
    }
    /// Returns the order `n` of the matrix.
    pub fn dimension(&self) -> usize {
        self.matrix.rows()
    }
    /// Returns the triangle that holds the entries of the matrix.
    pub const fn triangle(&self) -> Triangle {
        self.triangle
    }
    /// Returns whether the diagonal is stored or implicitly one.
    pub const fn diagonal_kind(&self) -> DiagonalKind {
        self.diagonal
    }
    /// Returns the stored matrix.
    pub const fn matrix(&self) -> &CsrMatrix<S> {
        &self.matrix
    }
    fn solve_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((0, usize::MAX))
    }
    fn solve_column(
        &self,
        rhs: &[S],
        output: &mut [S],
        work: &mut [S],
        operation: SparseOperation,
        arithmetic: Arithmetic,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let transposed = operation != SparseOperation::Normal;
        let traversal = self.matrix.opposite_traversal();
        let (ptr, indices) = if transposed {
            (
                traversal.pattern().row_ptr(),
                traversal.pattern().col_indices(),
            )
        } else {
            (self.matrix.row_ptr(), self.matrix.col_indices())
        };
        let value = |pos: usize| {
            let v = self.matrix.values()[if transposed {
                traversal.positions()[pos]
            } else {
                pos
            }];
            if operation == SparseOperation::Adjoint {
                v.conj()
            } else {
                v
            }
        };
        let lower = (self.triangle == Triangle::Lower) != transposed;
        for step in 0..self.dimension() {
            cx.check()?;
            let row = if lower {
                step
            } else {
                self.dimension() - 1 - step
            };
            let range = ptr[row]..ptr[row + 1];
            let pivot = if self.diagonal == DiagonalKind::Unit {
                S::ONE
            } else {
                let offset = indices[range.clone()]
                    .binary_search(&row)
                    .map_err(|_| SciOxideError::singular(Some(row)))?;
                value(range.start + offset)
            };
            let terms = || {
                range.clone().map(|p| {
                    if indices[p] == row {
                        (S::ZERO, S::ZERO)
                    } else {
                        (value(p), work[indices[p]])
                    }
                })
            };
            work[row] = match arithmetic.0 {
                scioxide_core::Accuracy::Standard => {
                    scioxide_core::backend::standard::substitute_row(rhs[row], pivot, terms())?
                }
                scioxide_core::Accuracy::Exact => arithmetic.divide(
                    arithmetic.subtract_products(rhs[row], terms())?,
                    pivot,
                    crate::helpers::Projection::Gradual,
                )?,
            };
        }
        output.copy_from_slice(work);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{DiagonalKind, SparseTriangular, Triangle};
    use crate::CsrMatrix;

    #[test]
    fn lower_upper_adjoint_and_block_solves_are_reusable() {
        let lower = CsrMatrix::from_raw(
            3,
            3,
            vec![0, 1, 3, 6],
            vec![0, 0, 1, 0, 1, 2],
            vec![2.0_f64, 1.0, 3.0, 4.0, -2.0, 1.0],
        )
        .unwrap();
        let plan = SparseTriangular::new(lower, Triangle::Lower, DiagonalKind::Explicit).unwrap();
        let mut output = [0.0; 3];
        let mut workspace = [0.0; 6];
        plan.solve_into(
            &[2.0, 7.0, 3.0],
            crate::SparseOperation::Normal,
            crate::SparseSolveOptions::new(),
            &mut output,
            &mut crate::SparseSolveWorkspace::borrowed(&mut workspace),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0]);

        plan.solve_into(
            &[16.0, 0.0, 3.0],
            crate::SparseOperation::Adjoint,
            crate::SparseSolveOptions::new(),
            &mut output,
            &mut crate::SparseSolveWorkspace::borrowed(&mut workspace),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0]);

        let right_hand_sides = [2.0, 7.0, 3.0, 4.0, 11.0, 3.0];
        let mut block_output = [9.0; 6];
        let input = scioxide_core::BlockVectorView::new(&right_hand_sides, 3, 2).unwrap();
        let mut output = scioxide_core::BlockVectorViewMut::new(&mut block_output, 3, 2).unwrap();
        let mut block_workspace = vec![0.0; plan.workspace_requirements(2).unwrap().scalar_len()];
        plan.solve_block_into(
            input,
            crate::SparseOperation::Normal,
            crate::SparseSolveOptions::new(),
            &mut output,
            &mut crate::SparseSolveWorkspace::borrowed(&mut block_workspace),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(block_output, [1.0, 2.0, 3.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn undersized_workspace_is_transactional() {
        let matrix = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64]).unwrap();
        let plan = SparseTriangular::new(matrix, Triangle::Lower, DiagonalKind::Explicit).unwrap();
        let mut output = [7.0];
        assert!(
            plan.solve_into(
                &[2.0],
                crate::SparseOperation::Normal,
                crate::SparseSolveOptions::new(),
                &mut output,
                &mut crate::SparseSolveWorkspace::borrowed(&mut []),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [7.0]);
    }
}
