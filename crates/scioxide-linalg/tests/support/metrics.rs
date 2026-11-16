//! Shared, overflow-safe metrics for decomposition tests.

#![allow(clippy::panic)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::Matrix;

fn entry<S: NumericScalar>(matrix: &Matrix<S>, row: usize, column: usize) -> S {
    matrix.as_slice()[row * matrix.cols() + column]
}

/// Residuals for an SVD vector job.
#[derive(Debug, Clone, Copy)]
pub struct SvdResiduals<R> {
    pub reconstruction: R,
    pub left_orthogonality: R,
    pub right_orthogonality: R,
}

impl<R: RealScalar> SvdResiduals<R> {
    #[must_use]
    pub fn maximum(self) -> R {
        self.reconstruction
            .max(self.left_orthogonality)
            .max(self.right_orthogonality)
    }
}

/// The shared baseline for well-conditioned decomposition fixtures.
#[must_use]
pub fn invariant_bound<R: RealScalar>(maximum_dimension: usize) -> R {
    let dimension = R::try_from_usize_exact(maximum_dimension.max(1)).unwrap_or(R::ONE);
    R::try_from_f64_exact(512.0).unwrap_or(R::ONE) * dimension * <R as RealScalar>::EPSILON
}

/// Compute a scaled sum-of-squares norm without squaring unscaled inputs.
#[must_use]
pub fn scaled_real_norm<R: RealScalar>(values: impl IntoIterator<Item = R>) -> R {
    let (scale, sum_squares) = values
        .into_iter()
        .fold((R::ZERO, R::ONE), |(scale, sum), value| {
            let magnitude = value.abs();
            if magnitude == R::ZERO {
                (scale, sum)
            } else if scale < magnitude {
                let ratio = scale / magnitude;
                (magnitude, R::ONE + sum * ratio * ratio)
            } else {
                let ratio = magnitude / scale;
                (scale, sum + ratio * ratio)
            }
        });
    if scale == R::ZERO {
        scale
    } else {
        scale * sum_squares.sqrt()
    }
}

/// Compute a scaled norm from the real and imaginary components of scalars.
#[must_use]
pub fn scaled_scalar_norm<S: NumericScalar>(values: impl IntoIterator<Item = S>) -> S::Real {
    scaled_real_norm(values.into_iter().flat_map(|value| {
        [
            value.real_component().abs(),
            value.imaginary_component().abs(),
        ]
    }))
}

#[must_use]
pub fn matrix_norm<S: NumericScalar>(matrix: &Matrix<S>) -> S::Real {
    scaled_scalar_norm(matrix.as_slice().iter().copied())
}

#[must_use]
pub fn matrix_difference_norm<S: NumericScalar>(left: &Matrix<S>, right: &Matrix<S>) -> S::Real {
    assert_eq!((left.rows(), left.cols()), (right.rows(), right.cols()));
    scaled_scalar_norm(
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(&left, &right)| left - right),
    )
}

#[must_use]
pub fn normalized<R: RealScalar>(numerator: R, denominator: R) -> R {
    if denominator == R::ZERO {
        numerator
    } else {
        numerator / denominator
    }
}

#[must_use]
pub fn relative_matrix_difference<S: NumericScalar>(
    actual: &Matrix<S>,
    expected: &Matrix<S>,
) -> S::Real {
    normalized(
        matrix_difference_norm(actual, expected),
        matrix_norm(expected),
    )
}

#[must_use]
pub fn column_orthogonality_residual<S: NumericScalar>(matrix: &Matrix<S>) -> S::Real {
    let columns = matrix.cols();
    let residual = scaled_scalar_norm((0..columns).flat_map(|left| {
        (0..columns).map(move |right| {
            let inner = (0..matrix.rows()).fold(S::ZERO, |sum, row| {
                sum + entry(matrix, row, left).conj() * entry(matrix, row, right)
            });
            inner - if left == right { S::ONE } else { S::ZERO }
        })
    }));
    normalized(
        residual,
        S::Real::try_from_usize_exact(columns.max(1))
            .unwrap_or(S::Real::ONE)
            .sqrt(),
    )
}

#[must_use]
pub fn row_orthogonality_residual<S: NumericScalar>(matrix: &Matrix<S>) -> S::Real {
    let rows = matrix.rows();
    let residual = scaled_scalar_norm((0..rows).flat_map(|upper| {
        (0..rows).map(move |lower| {
            let inner = (0..matrix.cols()).fold(S::ZERO, |sum, column| {
                sum + entry(matrix, upper, column) * entry(matrix, lower, column).conj()
            });
            inner - if upper == lower { S::ONE } else { S::ZERO }
        })
    }));
    normalized(
        residual,
        S::Real::try_from_usize_exact(rows.max(1))
            .unwrap_or(S::Real::ONE)
            .sqrt(),
    )
}

#[must_use]
pub fn column_normalization_residual<R: RealScalar>(matrix: &Matrix<Complex<R>>) -> R
where
    Complex<R>: NumericScalar<Real = R>,
{
    (0..matrix.cols()).fold(R::ZERO, |maximum, column| {
        let norm = scaled_scalar_norm((0..matrix.rows()).map(|row| entry(matrix, row, column)));
        maximum.max((norm - R::ONE).abs())
    })
}

#[must_use]
pub fn hermitian_eigen_residual<S: NumericScalar>(
    matrix: &Matrix<S>,
    eigenvalues: &[S::Real],
    eigenvectors: &Matrix<S>,
) -> S::Real {
    let order = matrix.rows();
    let residual = scaled_scalar_norm((0..order).flat_map(|row| {
        (0..order).map(move |column| {
            let applied = (0..order).fold(S::ZERO, |sum, inner| {
                sum + entry(matrix, row, inner) * entry(eigenvectors, inner, column)
            });
            applied - entry(eigenvectors, row, column) * S::from_real(eigenvalues[column])
        })
    }));
    normalized(residual, matrix_norm(matrix))
}

#[must_use]
pub fn general_eigen_residual<S, R>(
    matrix: &Matrix<S>,
    eigenvalues: &[Complex<R>],
    eigenvectors: &Matrix<Complex<R>>,
) -> R
where
    S: NumericScalar<Real = R>,
    R: RealScalar,
    Complex<R>: NumericScalar<Real = R>,
{
    let order = matrix.rows();
    let residual = scaled_scalar_norm((0..order).flat_map(|row| {
        (0..order).map(move |column| {
            let applied = (0..order).fold(Complex::new(R::ZERO, R::ZERO), |sum, inner| {
                sum + into_complex(entry(matrix, row, inner)) * entry(eigenvectors, inner, column)
            });
            applied - eigenvalues[column] * entry(eigenvectors, row, column)
        })
    }));
    normalized(residual, matrix_norm(matrix))
}

#[must_use]
pub fn general_left_eigen_residual<S, R>(
    matrix: &Matrix<S>,
    eigenvalues: &[Complex<R>],
    eigenvectors: &Matrix<Complex<R>>,
) -> R
where
    S: NumericScalar<Real = R>,
    R: RealScalar,
    Complex<R>: NumericScalar<Real = R>,
{
    let order = matrix.rows();
    let residual = scaled_scalar_norm((0..order).flat_map(|row| {
        (0..order).map(move |column| {
            let applied = (0..order).fold(Complex::new(R::ZERO, R::ZERO), |sum, inner| {
                sum + into_complex(entry(matrix, inner, row)).conj()
                    * entry(eigenvectors, inner, column)
            });
            applied - eigenvalues[column].conj() * entry(eigenvectors, row, column)
        })
    }));
    normalized(residual, matrix_norm(matrix))
}

pub fn reconstruct_svd<S: NumericScalar>(
    u: &Matrix<S>,
    singular_values: &[S::Real],
    vt: &Matrix<S>,
) -> Matrix<S> {
    let data = (0..u.rows() * vt.cols())
        .map(|index| {
            let row = index / vt.cols();
            let column = index % vt.cols();
            singular_values
                .iter()
                .enumerate()
                .fold(S::ZERO, |sum, (inner, &singular_value)| {
                    sum + entry(u, row, inner)
                        * S::from_real(singular_value)
                        * entry(vt, inner, column)
                })
        })
        .collect();
    Matrix::from_vec(u.rows(), vt.cols(), data).unwrap_or_else(|error| panic!("{error}"))
}

#[must_use]
pub fn svd_residuals<S: NumericScalar>(
    matrix: &Matrix<S>,
    u: &Matrix<S>,
    singular_values: &[S::Real],
    vt: &Matrix<S>,
) -> SvdResiduals<S::Real> {
    let reconstructed = reconstruct_svd(u, singular_values, vt);
    SvdResiduals {
        reconstruction: relative_matrix_difference(&reconstructed, matrix),
        left_orthogonality: column_orthogonality_residual(u),
        right_orthogonality: row_orthogonality_residual(vt),
    }
}

/// Both singular-triplet equations, normalized by the original matrix norm.
/// This remains meaningful at zero singular values and repeated spectra.
pub fn singular_triplet_residual<S: NumericScalar>(
    matrix: &Matrix<S>,
    u: &Matrix<S>,
    values: &[S::Real],
    vt: &Matrix<S>,
) -> S::Real {
    let forward = values.iter().enumerate().flat_map(|(k, &sigma)| {
        (0..matrix.rows()).map(move |i| {
            (0..matrix.cols()).fold(S::ZERO, |sum, j| {
                sum + entry(matrix, i, j) * entry(vt, k, j).conj()
            }) - entry(u, i, k) * S::from_real(sigma)
        })
    });
    let adjoint = values.iter().enumerate().flat_map(|(k, &sigma)| {
        (0..matrix.cols()).map(move |j| {
            (0..matrix.rows()).fold(S::ZERO, |sum, i| {
                sum + entry(matrix, i, j).conj() * entry(u, i, k)
            }) - entry(vt, k, j).conj() * S::from_real(sigma)
        })
    });
    normalized(
        scaled_scalar_norm(forward.chain(adjoint)),
        matrix_norm(matrix),
    )
}

#[must_use]
pub fn real_spectrum_difference<R: RealScalar>(left: &[R], right: &[R]) -> R {
    assert_eq!(left.len(), right.len());
    normalized(
        scaled_real_norm(left.iter().zip(right).map(|(&left, &right)| left - right)),
        scaled_real_norm(right.iter().copied()),
    )
}

#[must_use]
pub fn complex_spectrum_difference<R: RealScalar>(left: &[Complex<R>], right: &[Complex<R>]) -> R
where
    Complex<R>: NumericScalar<Real = R>,
{
    assert_eq!(left.len(), right.len());
    normalized(
        scaled_scalar_norm(left.iter().zip(right).map(|(&left, &right)| left - right)),
        scaled_scalar_norm(right.iter().copied()),
    )
}

#[must_use]
pub fn as_f64<R: RealScalar>(value: R) -> f64 {
    value.try_to_f64_exact().unwrap_or(f64::INFINITY)
}

fn into_complex<S: NumericScalar>(value: S) -> Complex<S::Real> {
    Complex::new(value.real_component(), value.imaginary_component())
}
