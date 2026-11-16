//! Inner products and matrix products over any number type.

use scioxide_core::{RealScalar, SciOxideError};

use crate::Real;
use crate::real::sealed::Token;

fn product_len(a: usize, b: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(a, b, scioxide_core::quantity!("matrix"))
}

/// Returns the inner product `Σₖ leftₖ·rightₖ`, summed in index order.
///
/// On a tape it is one statement with `2n` arguments, however long the vectors.
/// The sum starts from `−0`, so a sum of `−0` products is `−0`; empty vectors give
/// `+0`.
///
/// # Accuracy
///
/// The value is the recursive sum of the products, so its error is at most
/// `γₙ·Σₖ |leftₖ·rightₖ|` with `γₙ = nε/(1 − nε)`; derivatives are sums of the same
/// form.
///
/// # Errors
///
/// `DimensionMismatch` if the lengths differ.
///
/// # References
///
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
/// 2002, §3.1.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{dot, grad};
/// let g = grad(|x| dot(&x[..2], &x[2..]), &[1.0, 2.0, 3.0, 4.0])?;
/// assert_eq!(g.as_slice(), [3.0, 4.0, 1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn dot<S: RealScalar, R: Real<S>>(left: &[R], right: &[R]) -> Result<R, SciOxideError> {
    crate::support::check_len(right.len(), left.len())?;
    Ok(R::dot(
        Token(()),
        left.iter().copied().zip(right.iter().copied()),
    ))
}

/// Writes `out = A·x` for a row-major `rows × cols` matrix, one [`dot`] per row.
///
/// Each entry has the accuracy of [`dot`] with `n = cols`.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length.
/// - `SizeOverflow` if `rows·cols` overflows `usize`.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Real, gemv_into};
/// // d(A·x)/dt along x = (1, 2) + t·(1, 0) is the first column of A.
/// let a = [1.0, 2.0, 3.0, 4.0].map(Dual::constant);
/// let x = [Dual::new(1.0, 1.0), Dual::new(2.0, 0.0)];
/// let mut y = [Dual::constant(0.0); 2];
/// gemv_into(&a, 2, 2, &x, &mut y)?;
/// assert_eq!(y.map(|v| v.value()), [5.0, 11.0]);
/// assert_eq!(y.map(|v| v.derivative()), [1.0, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn gemv_into<S: RealScalar, R: Real<S>>(
    matrix: &[R],
    rows: usize,
    cols: usize,
    vector: &[R],
    out: &mut [R],
) -> Result<(), SciOxideError> {
    crate::support::check_len(matrix.len(), product_len(rows, cols)?)?;
    crate::support::check_len(vector.len(), cols)?;
    crate::support::check_len(out.len(), rows)?;
    for (target, row) in out.iter_mut().zip(matrix.chunks_exact(cols.max(1))) {
        *target = R::dot(Token(()), row.iter().copied().zip(vector.iter().copied()));
    }
    if cols == 0 {
        out.fill(R::zero());
    }
    Ok(())
}

/// Writes `out = A·B` for row-major `rows × inner` and `inner × cols` matrices,
/// one [`dot`] per entry.
///
/// Each entry has the accuracy of [`dot`] with `n = inner`.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length.
/// - `SizeOverflow` if a matrix size overflows `usize`.
///
/// ```
/// use scioxide_ad::gemm_into;
/// use scioxide_core::SciOxideErrorKind;
/// let mut c = [0.0; 0];
/// let error = gemm_into(&[0.0_f64; 0], usize::MAX, 2, &[], 0, &mut c).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
/// ```
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, Var, gemm_into, grad};
/// // ∂/∂A of Σᵢⱼ (A·B)ᵢⱼ is the row sums of B, repeated in every row of A.
/// let b = [1.0, 2.0, 3.0, 4.0];
/// let g = grad(
///     |a| {
///         let right = b.map(Var::constant);
///         let mut c = [Var::constant(0.0); 4];
///         gemm_into(a, 2, 2, &right, 2, &mut c)?;
///         Ok(c.into_iter().sum())
///     },
///     &[0.5, -1.0, 2.0, 0.25],
/// )?;
/// assert_eq!(g.as_slice(), [3.0, 7.0, 3.0, 7.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn gemm_into<S: RealScalar, R: Real<S>>(
    left: &[R],
    rows: usize,
    inner: usize,
    right: &[R],
    cols: usize,
    out: &mut [R],
) -> Result<(), SciOxideError> {
    crate::support::check_len(left.len(), product_len(rows, inner)?)?;
    crate::support::check_len(right.len(), product_len(inner, cols)?)?;
    crate::support::check_len(out.len(), product_len(rows, cols)?)?;
    for i in 0..rows {
        for j in 0..cols {
            let pairs = (0..inner).map(|k| (left[i * inner + k], right[k * cols + j]));
            out[i * cols + j] = R::dot(Token(()), pairs);
        }
    }
    Ok(())
}
