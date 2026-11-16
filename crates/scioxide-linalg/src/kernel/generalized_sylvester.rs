//! Coupled block Sylvester substitution for generalized deflating subspaces.
use super::frames::extended;
use super::small::{Singularity, small};
use crate::workspace::storage::MatrixStorage;
use core::ops::Neg;
use scioxide_core::NumericScalar;
use scioxide_core::ScaledValue;
use scioxide_core::{RealScalar, SciOxideError};

/// Assemble the local map (R,L) -> (A R - L D, B R - L E), or its adjoint.
pub(crate) fn coefficients<S: NumericScalar, M: MatrixStorage<S>>(
    a: &M,
    b: &M,
    r: usize,
    c: usize,
    p: usize,
    q: usize,
    adjoint: bool,
) -> Result<[ScaledValue<scioxide_core::Complex<S::Real>>; 64], SciOxideError> {
    // Kronecker matrix (order 2pq, stored with stride 8) of the coupled
    // block Sylvester operator `(R, L) -> (A11 R - L A22, B11 R - L B22)`
    // for diagonal blocks at rows `r` and columns `c`. The adjoint operator
    // is its conjugate transpose, used by the separation estimator.
    let count = p * q;
    let mut out = [ScaledValue::ZERO; 64];
    for i in 0..p {
        for j in 0..q {
            for u in 0..p {
                for v in 0..q {
                    let row = i * q + j;
                    let col = u * q + v;
                    for (offset, m) in [(0, a), (count, b)] {
                        if j == v {
                            out[(row + offset) * 8 + col] = extended(m.get_element(r + i, r + u))?;
                        }
                        if i == u {
                            out[(row + offset) * 8 + count + col] =
                                extended(m.get_element(c + v, c + j))?.neg();
                        }
                    }
                }
            }
        }
    }
    if adjoint {
        for i in 0..2 * count {
            for j in 0..=i {
                let x = conj(out[i * 8 + j]);
                let y = conj(out[j * 8 + i]);
                out[i * 8 + j] = y;
                out[j * 8 + i] = x;
            }
        }
    }
    Ok(out)
}
fn conj<R: RealScalar>(
    z: ScaledValue<scioxide_core::Complex<R>>,
) -> ScaledValue<scioxide_core::Complex<R>> {
    ScaledValue::from_components(z.real(), z.imaginary().neg())
}

/// In-place generalized Sylvester or Frobenius-adjoint solve. Storage contains
/// two row-major m×(n-m) arrays. Reverse=true interchanges the diagonal clusters.
///
pub(crate) fn solve<S: NumericScalar, M: MatrixStorage<S>>(
    a: &M,
    b: &M,
    blocks: &[[usize; 2]],
    m: usize,
    rhs: &mut [ScaledValue<scioxide_core::Complex<S::Real>>],
    adjoint: bool,
    reverse: bool,
) -> Result<bool, SciOxideError> {
    // Block substitution for the coupled generalized Sylvester equation over
    // the whole leading/trailing partition (`reverse` swaps their roles).
    // `rhs` holds R in its first `count` entries and L in the next `count`.
    // Blocks are visited in the order that makes all coupling terms known,
    // reversed for the adjoint operator, and each block pair is a small
    // Kronecker system solved by `small`.
    let n = a.rows();
    let split = blocks.iter().take_while(|&&[start, _]| start < m).count();
    let (rows, cols, rstart, cstart, rb, cb) = if reverse {
        (n - m, m, m, 0, &blocks[split..], &blocks[..split])
    } else {
        (m, n - m, 0, m, &blocks[..split], &blocks[split..])
    };
    let count = rows * cols;
    for ri in 0..rb.len() {
        let row = rb[if adjoint { ri } else { rb.len() - 1 - ri }];
        let [r, p] = row;
        for ci in 0..cb.len() {
            let col = cb[if adjoint { cb.len() - 1 - ci } else { ci }];
            let [c, q] = col;
            let mut local = [ScaledValue::ZERO; 8];
            let mut matrix = coefficients(a, b, r, c, p, q, adjoint)?;
            for i in 0..p {
                for j in 0..q {
                    let index = (r + i - rstart) * cols + c + j - cstart;
                    local[i * q + j] = rhs[index];
                    local[p * q + i * q + j] = rhs[count + index];
                    if !adjoint {
                        for (offset, t) in [(0, a), (p * q, b)] {
                            let mut x = local[offset + i * q + j];
                            for k in r + p..rstart + rows {
                                x = x.sub(
                                    extended(t.get_element(r + i, k))?
                                        .mul(rhs[(k - rstart) * cols + c + j - cstart])?,
                                )?;
                            }
                            for k in cstart..c {
                                x = x.add(
                                    rhs[count + (r + i - rstart) * cols + k - cstart]
                                        .mul(extended(t.get_element(k, c + j))?)?,
                                )?;
                            }
                            local[offset + i * q + j] = x;
                        }
                    } else {
                        let mut x = local[i * q + j];
                        let mut y = local[p * q + i * q + j];
                        for (offset, t) in [(0, a), (count, b)] {
                            for k in rstart..r {
                                x = x
                                    .sub(extended(t.get_element(k, r + i).conj())?.mul(
                                        rhs[offset + (k - rstart) * cols + c + j - cstart],
                                    )?)?;
                            }
                            for k in c + q..cstart + cols {
                                y = y.add(
                                    rhs[offset + (r + i - rstart) * cols + k - cstart]
                                        .mul(extended(t.get_element(c + j, k).conj())?)?,
                                )?;
                            }
                        }
                        local[i * q + j] = x;
                        local[p * q + i * q + j] = y;
                    }
                }
            }
            if !small(&mut matrix, &mut local, 2 * p * q, Singularity::Reject)? {
                return Ok(false);
            }
            for i in 0..p {
                for j in 0..q {
                    let index = (r + i - rstart) * cols + c + j - cstart;
                    rhs[index] = local[i * q + j];
                    rhs[count + index] = local[p * q + i * q + j];
                }
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::needless_range_loop)]
    use super::*;
    use crate::Matrix;
    use scioxide_core::AlgebraicScalar;
    use scioxide_core::Complex;

    fn exercise<S: NumericScalar>() {
        let n = 5;
        let m = 2;
        let blocks: alloc::vec::Vec<[usize; 2]> = if S::IS_REAL {
            alloc::vec![[0, 2], [2, 1], [3, 2]]
        } else {
            (0..n).map(|start| [start, 1]).collect()
        };
        let c = |x| S::from_real(S::Real::try_from_f64_lossy(x).unwrap());
        let a = Matrix::from_vec(
            n,
            n,
            (0..n * n)
                .map(|i| {
                    let (r, j) = (i / n, i % n);
                    if r == j {
                        c((r * 3 + 1) as f64)
                    } else if r < j
                        || S::IS_REAL && (r, j) == (1, 0)
                        || S::IS_REAL && (r, j) == (4, 3)
                    {
                        c(0.2 * (r + j + 1) as f64)
                    } else {
                        S::ZERO
                    }
                })
                .collect(),
        )
        .unwrap();
        let b = Matrix::from_vec(
            n,
            n,
            (0..n * n)
                .map(|i| {
                    let (r, j) = (i / n, i % n);
                    if r == j {
                        c(1. + r as f64 / 7.)
                    } else if r < j {
                        c(0.03 * (r + j + 1) as f64)
                    } else {
                        S::ZERO
                    }
                })
                .collect(),
        )
        .unwrap();
        for reverse in [false, true] {
            for adjoint in [false, true] {
                let (rows, cols, rs, cs) = if reverse {
                    (n - m, m, m, 0)
                } else {
                    (m, n - m, 0, m)
                };
                let count = rows * cols;
                let x: alloc::vec::Vec<_> = (0..2 * count)
                    .map(|i| {
                        S::from_components(
                            S::Real::try_from_f64_lossy((i + 1) as f64 / 13.).unwrap(),
                            if S::IS_REAL {
                                S::Real::ZERO
                            } else {
                                S::Real::try_from_f64_lossy((i + 3) as f64 / 17.).unwrap()
                            },
                        )
                    })
                    .collect();
                let mut rhs = alloc::vec![ScaledValue::ZERO;2*count];
                for i in 0..rows {
                    for j in 0..cols {
                        let at = i * cols + j;
                        if adjoint {
                            let mut u = S::ZERO;
                            let mut v = S::ZERO;
                            for (offset, t) in [(0, &a), (count, &b)] {
                                for k in 0..rows {
                                    u = u + t.get_element(rs + k, rs + i).conj()
                                        * x[offset + k * cols + j];
                                }
                                for k in 0..cols {
                                    v = v - x[offset + i * cols + k]
                                        * t.get_element(cs + j, cs + k).conj();
                                }
                            }
                            rhs[at] = extended(u).unwrap();
                            rhs[count + at] = extended(v).unwrap();
                        } else {
                            for (offset, t) in [(0, &a), (count, &b)] {
                                let u = (0..rows).fold(S::ZERO, |sum, k| {
                                    sum + t.get_element(rs + i, rs + k) * x[k * cols + j]
                                });
                                let v = (0..cols).fold(S::ZERO, |sum, k| {
                                    sum + x[count + i * cols + k] * t.get_element(cs + k, cs + j)
                                });
                                rhs[offset + at] = extended(u - v).unwrap();
                            }
                        }
                    }
                }
                assert!(solve(&a, &b, &blocks, m, &mut rhs, adjoint, reverse).unwrap());
                for (y, x) in rhs.iter().zip(&x) {
                    let y = crate::kernel::frames::scalar::<S>(*y).unwrap();
                    assert!(
                        (y - *x).abs()
                            < S::Real::try_from_usize_exact(512).unwrap() * S::Real::EPSILON
                    );
                }
            }
        }
    }
    #[test]
    fn real64() {
        exercise::<f64>();
    }
    #[test]
    fn real32() {
        exercise::<f32>();
    }
    #[test]
    fn complex64() {
        exercise::<Complex<f64>>();
    }
    #[test]
    fn complex32() {
        exercise::<Complex<f32>>();
    }
}
