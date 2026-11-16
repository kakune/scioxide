//! One-shot sparse linear solves.
use alloc::vec::Vec;

use scioxide_core::{NumericScalar, SciOxideError};

use crate::{CsrMatrix, SparseLu, SparseQrMinimumNormPlan};

/// Solves the square sparse system `A x = b` for `A = matrix`, `b = rhs`, and
/// returns `x`.
///
/// This is [`SparseLu::new`] followed by [`SparseLu::solve`]: a left-looking
/// LU factorization with partial pivoting after a fill-reducing column
/// ordering, then two triangular substitutions. To solve with the same matrix
/// for several right-hand sides, keep the [`SparseLu`]. The backward error
/// depends on the growth of the factors; the forward error also on the
/// condition number of `A`.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
///
/// // A = tridiag(−1, 2, −1) of order 3. For b = (1, 0, 1), x = (1, 1, 1).
/// let (rows, cols) = ([0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]);
/// let a =
///     CsrMatrix::from_triplets(3, 3, &rows, &cols, &[2.0_f64, -1., -1., 2., -1., -1., 2.])?;
/// let x = scioxide_sparse::solve(&a, &[1.0, 0.0, 1.0])?;
/// assert!(x.iter().all(|xi| (xi - 1.0).abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `matrix` is not square or `rhs.len()` differs from
///   its order.
/// - `NonFinite` if `rhs` has a NaN or infinite component.
/// - `Singular` if a column has no nonzero pivot candidate.
/// - `NumericalInstability` if an entry overflows or a nonzero multiplier
///   underflows to zero.
/// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
///
/// # References
///
/// - J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time
///   proportional to arithmetic operations", *SIAM J. Sci. Stat. Comput.* 9
///   (1988), 862–874.
pub fn solve<S: NumericScalar>(matrix: &CsrMatrix<S>, rhs: &[S]) -> Result<Vec<S>, SciOxideError> {
    SparseLu::new(matrix)?.solve(rhs)
}

/// Returns the minimum-norm minimizer `x` of `‖A x − b‖₂` for the `m × n`
/// matrix `A = matrix` and `b = rhs`.
///
/// This is [`SparseQrMinimumNormPlan::new`] followed by
/// [`SparseQrMinimumNormPlan::solve`], without the report: Householder QR
/// after a fill-reducing column ordering, and a complete orthogonal
/// decomposition of numerical rank `r` with the cutoff
/// `τ = max(m, n) · ε · max_j ‖A[:, j]‖₂`, `ε` the machine epsilon. It covers
/// overdetermined, underdetermined and rank-deficient systems; the rank
/// depends on the column order and is an estimate.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
///
/// // Fit y = c through the observations 1, 2 and 4: c = 7/3.
/// let a = CsrMatrix::from_triplets(3, 1, &[0, 1, 2], &[0, 0, 0], &[1.0_f64, 1.0, 1.0])?;
/// let x = scioxide_sparse::least_squares(&a, &[1.0, 2.0, 4.0])?;
/// assert!((x[0] - 7.0 / 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `rhs.len()` differs from the number of rows.
/// - `NonFinite` if `rhs` has a NaN or infinite component.
/// - `NumericalInstability` if a column norm, the cutoff, a Householder vector
///   or a solution entry is not representable.
/// - `Singular` if a diagonal entry of the second triangular factor is exactly
///   zero.
/// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
pub fn least_squares<S: NumericScalar>(
    matrix: &CsrMatrix<S>,
    rhs: &[S],
) -> Result<Vec<S>, SciOxideError> {
    SparseQrMinimumNormPlan::new(matrix)?
        .solve(rhs)
        .map(|(solution, _)| solution)
}
