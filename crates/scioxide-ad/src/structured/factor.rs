//! Cholesky, QR and least squares evaluated in the number type.
//!
//! These are the textbook algorithms written in `R` arithmetic, so every
//! derivative mode differentiates the evaluated operations; on a tape they
//! record `O(n³)` statements.

use alloc::vec::Vec;

use scioxide_core::{DomainReason, RealScalar, SciOxideError};

use crate::{Mask, Real};

fn len(a: usize, b: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(a, b, scioxide_core::quantity!("matrix"))
}

fn scratch<R: Copy>(len: usize, fill: R) -> Result<Vec<R>, SciOxideError> {
    scioxide_core::backend::try_filled(len, fill, scioxide_core::quantity!("factor_scratch"))
}

/// Writes the lower Cholesky factor `L` (`A = L·Lᵀ`, positive diagonal) of a
/// symmetric positive definite `order × order` row-major matrix to `out`; the
/// strict upper triangle of `out` is zero. Only the lower triangle of `matrix` is
/// read. Cost: about `n³/3` floating-point operations (`n³/6` multiply-adds). The
/// factor is built in an `order × order` scratch and copied to `out` on success.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length.
/// - `SizeOverflow` if `order²` overflows `usize`.
/// - `DomainError` (argument `matrix`, `OutsideDomain`) if a pivot is not positive
///   in some lane.
/// - `AllocationFailure` if the scratch cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::cholesky_into;
/// let mut l = [0.0; 4];
/// cholesky_into(&[4.0, 2.0, 2.0, 5.0], 2, &mut l)?;
/// assert_eq!(l, [2.0, 0.0, 1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §4.2.
pub fn cholesky_into<S: RealScalar, R: Real<S>>(
    matrix: &[R],
    order: usize,
    out: &mut [R],
) -> Result<(), SciOxideError> {
    let n = order;
    crate::support::check_len(matrix.len(), len(n, n)?)?;
    crate::support::check_len(out.len(), len(n, n)?)?;
    let mut l = scratch(n * n, R::zero())?;
    for j in 0..n {
        let mut d = matrix[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if !d.gt(R::zero()).all() {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("matrix"),
                DomainReason::OutsideDomain,
                "the matrix is not numerically positive definite",
            ));
        }
        let pivot = d.sqrt();
        l[j * n + j] = pivot;
        for i in j + 1..n {
            let mut s = matrix[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / pivot;
        }
    }
    out.copy_from_slice(&l);
    Ok(())
}

/// Writes the thin QR factorization `A = Q·R` of a row-major `rows × cols` matrix
/// with `rows ≥ cols`: `Q` (`rows × cols`, orthonormal columns) and `R`
/// (`cols × cols`, upper triangular, non-negative diagonal).
///
/// Householder reflections with signs chosen by `select` on base values, then the
/// diagonal of `R` made non-negative. Cost: `2·rows·cols² − ⅔cols³` floating-point
/// operations for `R` and as many for `Q`. Two `rows × cols` scratches are
/// allocated: the working matrix and the reflectors.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length or `rows < cols`.
/// - `SizeOverflow` if `rows·cols` overflows `usize`.
/// - `AllocationFailure` if a scratch cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::qr_into;
/// let (mut q, mut r) = ([0.0; 4], [0.0; 4]);
/// qr_into(&[3.0_f64, 1.0, 4.0, 2.0], 2, 2, &mut q, &mut r)?;
/// assert!((r[0] - 5.0).abs() < 1e-15 && r[2] == 0.0 && r[3] > 0.0);
/// assert!((q[0] * r[1] + q[1] * r[3] - 1.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §5.2.
pub fn qr_into<S: RealScalar, R: Real<S>>(
    matrix: &[R],
    rows: usize,
    cols: usize,
    q: &mut [R],
    r: &mut [R],
) -> Result<(), SciOxideError> {
    crate::support::check_len(matrix.len(), len(rows, cols)?)?;
    crate::support::check_len(q.len(), len(rows, cols)?)?;
    crate::support::check_len(r.len(), len(cols, cols)?)?;
    if rows < cols {
        return Err(SciOxideError::dimension_mismatch(
            [cols, cols],
            [rows, cols],
        ));
    }
    let mut work = scratch(rows * cols, R::zero())?;
    work.copy_from_slice(matrix);
    let mut reflectors = scratch(rows * cols, R::zero())?;
    for k in 0..cols {
        // v = x + sign(x₀)·‖x‖·e₁, normalized; H = I − 2vvᵀ.
        let norm = (k..rows)
            .map(|i| work[i * cols + k] * work[i * cols + k])
            .sum::<R>()
            .sqrt();
        let x0 = work[k * cols + k];
        let alpha = R::select(x0.lt(R::zero()), -norm, norm);
        for i in k..rows {
            reflectors[i * cols + k] = work[i * cols + k];
        }
        reflectors[k * cols + k] = x0 + alpha;
        let length = (k..rows)
            .map(|i| reflectors[i * cols + k] * reflectors[i * cols + k])
            .sum::<R>()
            .sqrt();
        let degenerate = length.le(R::zero());
        for i in k..rows {
            let v = reflectors[i * cols + k] / length;
            reflectors[i * cols + k] = R::select(degenerate, R::zero(), v);
        }
        for j in k..cols {
            let projection = (k..rows)
                .map(|i| reflectors[i * cols + k] * work[i * cols + j])
                .sum::<R>()
                * S::TWO;
            for i in k..rows {
                let update = reflectors[i * cols + k] * projection;
                work[i * cols + j] -= update;
            }
        }
    }
    // Q = H₀⋯H_{c−1}·[I; 0].
    for (index, target) in q.iter_mut().enumerate() {
        let (i, j) = (index / cols, index % cols);
        *target = if i == j { R::one() } else { R::zero() };
    }
    for k in (0..cols).rev() {
        for j in 0..cols {
            let projection = (k..rows)
                .map(|i| reflectors[i * cols + k] * q[i * cols + j])
                .sum::<R>()
                * S::TWO;
            for i in k..rows {
                let update = reflectors[i * cols + k] * projection;
                q[i * cols + j] -= update;
            }
        }
    }
    for i in 0..cols {
        let negative = work[i * cols + i].lt(R::zero());
        for j in 0..cols {
            let value = if j >= i {
                work[i * cols + j]
            } else {
                R::zero()
            };
            r[i * cols + j] = R::select(negative, -value, value);
        }
        for row in 0..rows {
            let value = q[row * cols + i];
            q[row * cols + i] = R::select(negative, -value, value);
        }
    }
    Ok(())
}

/// Writes the least-squares solution `X = argmin ‖A·X − B‖₂` for a row-major
/// `rows × cols` matrix `A` with `rows ≥ cols` and full column rank, and `columns`
/// right-hand sides, through the thin QR: `X = R⁻¹·Qᵀ·B`. The call allocates the
/// factors (`rows·cols + cols²` values), the two scratches of [`qr_into`] and the
/// `cols × columns` solution.
///
/// # Errors
///
/// - `DimensionMismatch` if a slice has the wrong length or `rows < cols`.
/// - `SizeOverflow` if a matrix size overflows `usize`.
/// - `Singular` if a diagonal entry of `R` is zero, NaN or infinite in some lane.
/// - `AllocationFailure` if a scratch cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::least_squares_into;
/// // Fit y = a + b·t through (0, 1), (1, 3), (2, 5).
/// let a = [1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0];
/// let mut x = [0.0; 2];
/// least_squares_into(&a, 3, 2, &[1.0, 3.0, 5.0], 1, &mut x)?;
/// assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §5.3.
pub fn least_squares_into<S: RealScalar, R: Real<S>>(
    matrix: &[R],
    rows: usize,
    cols: usize,
    rhs: &[R],
    columns: usize,
    out: &mut [R],
) -> Result<(), SciOxideError> {
    let size = len(rows, cols)?;
    crate::support::check_len(matrix.len(), size)?;
    if rows < cols {
        return Err(SciOxideError::dimension_mismatch(
            [cols, cols],
            [rows, cols],
        ));
    }
    crate::support::check_len(rhs.len(), len(rows, columns)?)?;
    crate::support::check_len(out.len(), len(cols, columns)?)?;
    let factors_len = size
        .checked_add(len(cols, cols)?)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("matrix")))?;
    let mut factors = scratch(factors_len, R::zero())?;
    let (q, r) = factors.split_at_mut(rows * cols);
    qr_into(matrix, rows, cols, q, r)?;
    // A zero, NaN or infinite diagonal entry of `R` in some lane.
    if let Some(i) = (0..cols).find(|&i| !r[i * cols + i].gt(R::zero()).all()) {
        return Err(SciOxideError::singular(Some(i)));
    }
    let mut x = scratch(cols * columns, R::zero())?;
    for i in 0..cols {
        for column in 0..columns {
            x[i * columns + column] = (0..rows)
                .map(|k| q[k * cols + i] * rhs[k * columns + column])
                .sum();
        }
    }
    for i in (0..cols).rev() {
        for column in 0..columns {
            let mut value = x[i * columns + column];
            for j in i + 1..cols {
                value -= r[i * cols + j] * x[j * columns + column];
            }
            x[i * columns + column] = value / r[i * cols + i];
        }
    }
    out.copy_from_slice(&x);
    Ok(())
}
