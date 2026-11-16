use alloc::vec::Vec;
use core::marker::PhantomData;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{Dimensions, ExecutionContext, NumericScalar, SciOxideError};

use crate::{CsrMatrix, SparseSolveOptions};

/// Applies the identity preconditioner `M⁻¹ = I`, which copies its input.
///
/// It is useful to run a Krylov solver without preconditioning through the same
/// interface. It has the same `apply_into` forms as the other preconditioners
/// and implements the core [`Preconditioner`](scioxide_core::Preconditioner)
/// trait. Each right-hand side counts as one iteration.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Preconditioner};
/// use scioxide_sparse::{IdentityPreconditioner, SparseSolveOptions};
/// let identity = IdentityPreconditioner::<f64>::new(2);
/// let mut workspace = identity.workspace(1)?;
/// let mut output = [0.0; 2];
/// identity.apply_into(
///     &[1.0, 2.0],
///     SparseSolveOptions::new(),
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(output, [1.0, 2.0]);
/// // A short input is rejected and the output keeps its previous value.
/// assert!(
///     identity
///         .precondition_into(&[1.0], &mut output, &mut [0.0; 2])
///         .is_err()
/// );
/// assert_eq!(output, [1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct IdentityPreconditioner<S: NumericScalar> {
    dimension: usize,
    scalar: PhantomData<fn() -> S>,
}

impl<S: NumericScalar> Clone for IdentityPreconditioner<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: NumericScalar> Copy for IdentityPreconditioner<S> {}
impl<S: NumericScalar> PartialEq for IdentityPreconditioner<S> {
    fn eq(&self, other: &Self) -> bool {
        self.dimension == other.dimension
    }
}
impl<S: NumericScalar> Eq for IdentityPreconditioner<S> {}

impl<S: NumericScalar> IdentityPreconditioner<S> {
    /// Creates the identity preconditioner of order `dimension`.
    pub const fn new(dimension: usize) -> Self {
        Self {
            dimension,
            scalar: PhantomData,
        }
    }
}

impl<S: NumericScalar> crate::factor_solve::Application<S> for IdentityPreconditioner<S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        _: &mut [S],
        _: SparseSolveOptions,
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        candidate.copy_from_slice(rhs);
        Ok(())
    }
}
crate::factor_solve::impl_applications!(IdentityPreconditioner<S>);

/// Applies the Jacobi (diagonal) preconditioner `M⁻¹ = D⁻¹`, where `D` is the
/// diagonal of `A`.
///
/// `D⁻¹` is computed once, when the preconditioner is created. Each application
/// multiplies the input entrywise by it, with the
/// [`Accuracy`](scioxide_core::Accuracy) of the [`SparseSolveOptions`] of the call;
/// the core [`Preconditioner`](scioxide_core::Preconditioner) trait uses
/// `Standard`. Each right-hand side counts as one iteration.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Accuracy, ExecutionContext};
/// use scioxide_sparse::{CsrMatrix, JacobiPreconditioner, SparseSolveOptions};
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 2.0])?;
/// let jacobi = JacobiPreconditioner::new(&a)?;
/// let mut workspace = jacobi.workspace(1)?;
/// let mut output = [0.0; 2];
/// let options = SparseSolveOptions::new().with_accuracy(Accuracy::Exact);
/// jacobi.apply_into(
///     &[4.0, 4.0],
///     options,
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert_eq!(output, [1.0, 2.0]);
/// // A diagonal entry that is not stored makes the matrix singular here.
/// let missing = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![1], vec![1.0_f64])?;
/// assert!(JacobiPreconditioner::new(&missing).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct JacobiPreconditioner<S: NumericScalar> {
    inverse_diagonal: Vec<S>,
}

impl<S: NumericScalar> JacobiPreconditioner<S> {
    /// Creates the Jacobi preconditioner from the diagonal of a square matrix.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `Singular` if a diagonal entry is zero or not stored.
    /// - `NumericalInstability` if a reciprocal `1/a_ii` overflows or a nonzero
    ///   part of it underflows to zero.
    /// - `AllocationFailure` if the diagonal cannot be allocated.
    pub fn new(matrix: &CsrMatrix<S>) -> Result<Self, SciOxideError> {
        if matrix.rows() != matrix.cols() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(matrix.rows(), matrix.rows()),
                Dimensions::two(matrix.rows(), matrix.cols()),
            ));
        }
        let mut inverse_diagonal = try_with_capacity(
            matrix.rows(),
            scioxide_core::quantity!("jacobi_inverse_diagonal"),
        )?;
        for row in 0..matrix.rows() {
            // A structurally missing diagonal reads as zero.
            let diagonal = matrix.get(row, row)?;
            if diagonal == S::ZERO {
                return Err(SciOxideError::singular(Some(row)));
            }
            inverse_diagonal.push(
                crate::factor_arithmetic::Arithmetic(scioxide_core::Accuracy::Standard).divide(
                    S::ONE,
                    diagonal,
                    crate::helpers::Projection::Strict,
                )?,
            );
        }
        Ok(Self { inverse_diagonal })
    }

    /// Returns the inverse diagonal `D⁻¹`.
    pub fn inverse_diagonal(&self) -> &[S] {
        &self.inverse_diagonal
    }
}

impl<S: NumericScalar> crate::factor_solve::Application<S> for JacobiPreconditioner<S> {
    fn dimension(&self) -> usize {
        self.inverse_diagonal.len()
    }
    fn work_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_column(
        &self,
        rhs: &[S],
        candidate: &mut [S],
        _: &mut [S],
        options: SparseSolveOptions,
        _: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        rhs.iter()
            .zip(&self.inverse_diagonal)
            .zip(candidate.iter_mut())
            .try_for_each(|((&value, &inverse), slot)| {
                *slot =
                    scioxide_core::backend::standard::product(value, inverse, options.accuracy())?;
                Ok(())
            })
    }
}
crate::factor_solve::impl_applications!(JacobiPreconditioner<S>);

#[cfg(test)]
mod tests {
    use alloc::vec;

    use scioxide_core::Preconditioner;

    use super::JacobiPreconditioner;
    use crate::CsrMatrix;

    #[test]
    fn sparse_preconditioners_use_explicit_caller_workspace() {
        let matrix = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0_f64, 1.0, 1.0, 3.0],
        )
        .unwrap();
        let jacobi = JacobiPreconditioner::new(&matrix).unwrap();
        let mut output = [0.0; 2];
        let mut jacobi_workspace = [0.0; 2];
        jacobi
            .precondition_into(&[4.0, 3.0], &mut output, &mut jacobi_workspace)
            .unwrap();
        assert_eq!(output, [1.0, 1.0]);
    }

    #[test]
    fn preconditioner_dimension_error_preserves_output() {
        let matrix = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64]).unwrap();
        let jacobi = JacobiPreconditioner::new(&matrix).unwrap();
        let mut output = [7.0];
        assert!(
            jacobi
                .precondition_into(&[2.0], &mut output, &mut [])
                .is_err()
        );
        assert_eq!(output, [7.0]);
    }

    #[test]
    fn jacobi_construction_classifies_diagonal_failures() {
        use scioxide_core::{Complex64, NumericalIssue, SciOxideErrorView};
        let diagonal = |values: alloc::vec::Vec<f64>| {
            let n = values.len();
            CsrMatrix::from_raw(n, n, (0..=n).collect(), (0..n).collect(), values).unwrap()
        };
        // A structurally missing and an explicitly zero diagonal are singular rows.
        let missing = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![0], vec![1.0_f64]).unwrap();
        let zero = diagonal(vec![1., 0.]);
        for matrix in [missing, zero] {
            let error = JacobiPreconditioner::new(&matrix).unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::Singular { pivot: Some(1) }
            ));
        }
        // The inverse of the smallest subnormal overflows.
        let error = JacobiPreconditioner::new(&diagonal(vec![f64::from_bits(1)])).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        // Extreme representable inverses are retained, including a subnormal one.
        let tiny = 2_f64.powi(-1000);
        let jacobi = JacobiPreconditioner::new(&diagonal(vec![f64::MAX, tiny])).unwrap();
        let inverse = jacobi.inverse_diagonal();
        assert!(inverse[0] > 0. && inverse[0] < f64::MIN_POSITIVE);
        assert!((inverse[0] - 1. / f64::MAX).abs() <= f64::from_bits(1));
        assert_eq!(inverse[1], 2_f64.powi(1000));
        // In-range complex inverses keep the ordinary quotient.
        let d = Complex64::new(3., 4.);
        let matrix = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![d]).unwrap();
        let inverse = JacobiPreconditioner::new(&matrix)
            .unwrap()
            .inverse_diagonal()[0];
        assert_eq!(inverse, Complex64::new(1., 0.) / d);
        assert!((inverse - Complex64::new(0.12, -0.16)).norm() <= 4. * f64::EPSILON);
    }
}
