//! Deterministic fixtures and assertions shared by the decomposition suite.

#![allow(clippy::panic, clippy::unwrap_used)]

use scioxide_core::AlgebraicScalar;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{
    GeneralEigenDecomposition, HermitianEigenDecomposition, Matrix, SvdDecomposition, SvdValues,
    ThinSvdDecomposition,
};

use super::metrics::{
    column_normalization_residual, column_orthogonality_residual, complex_spectrum_difference,
    general_eigen_residual, general_left_eigen_residual, hermitian_eigen_residual, invariant_bound,
    real_spectrum_difference, svd_residuals,
};

/// Scalar construction and scale policy for deterministic invariant fixtures.
pub trait InvariantScalar: NumericScalar {
    const NAME: &'static str;

    fn from_parts(real: Self::Real, imaginary: Self::Real) -> Self;
    fn small_scale() -> Self::Real;
    fn large_scale() -> Self::Real;
}

macro_rules! impl_real_scalar {
    ($scalar:ty, $small:expr, $large:expr) => {
        impl InvariantScalar for $scalar {
            const NAME: &'static str = stringify!($scalar);

            fn from_parts(real: Self::Real, _imaginary: Self::Real) -> Self {
                real
            }

            fn small_scale() -> Self::Real {
                $small
            }

            fn large_scale() -> Self::Real {
                $large
            }
        }
    };
}

impl_real_scalar!(f32, 1.0e-20, 1.0e20);
impl_real_scalar!(f64, 1.0e-150, 1.0e150);

impl InvariantScalar for Complex<f32> {
    const NAME: &'static str = "complex f32";

    fn from_parts(real: Self::Real, imaginary: Self::Real) -> Self {
        Self::new(real, imaginary)
    }

    fn small_scale() -> Self::Real {
        1.0e-20
    }

    fn large_scale() -> Self::Real {
        1.0e20
    }
}

impl InvariantScalar for Complex<f64> {
    const NAME: &'static str = "complex f64";

    fn from_parts(real: Self::Real, imaginary: Self::Real) -> Self {
        Self::new(real, imaginary)
    }

    fn small_scale() -> Self::Real {
        1.0e-150
    }

    fn large_scale() -> Self::Real {
        1.0e150
    }
}

fn mix(mut word: u64) -> u64 {
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

fn sample<R: RealScalar>(seed: u64, row: usize, column: usize, lane: u64) -> R {
    let word = mix(seed
        ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (column as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
        ^ lane.wrapping_mul(0x94d0_49bb_1331_11eb));
    let mantissa = (word >> 11) as f64;
    let signed = mantissa.mul_add(2.0 / 9_007_199_254_740_992.0, -1.0);
    R::try_from_f64_lossy(signed).unwrap_or(R::ZERO)
}

fn generated_value<S: InvariantScalar>(seed: u64, row: usize, column: usize) -> S {
    S::from_parts(sample(seed, row, column, 0), sample(seed, row, column, 1))
}

/// Generate a reproducible dense matrix without a random-number dependency.
pub fn generated_matrix<S: InvariantScalar>(rows: usize, cols: usize, seed: u64) -> Matrix<S> {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|index| generated_value(seed, index / cols, index % cols))
            .collect(),
    )
    .unwrap()
}

/// Generate a small, diagonally shifted square matrix.
pub fn non_normal_matrix<S: InvariantScalar>(order: usize, seed: u64) -> Matrix<S> {
    let shift = S::Real::try_from_usize_exact(order.max(1) * 2).unwrap_or(S::Real::ONE);
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let row = index / order;
                let column = index % order;
                let value = generated_value(seed, row, column);
                if row == column {
                    value + S::from_real(shift)
                } else {
                    value
                }
            })
            .collect(),
    )
    .unwrap()
}

/// Generate a Hermitian matrix from a stateless upper-triangle fixture.
pub fn hermitian_matrix<S: InvariantScalar>(order: usize, seed: u64) -> Matrix<S> {
    let shift = S::Real::try_from_usize_exact(order.max(1) * 2).unwrap_or(S::Real::ONE);
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let row = index / order;
                let column = index % order;
                if row == column {
                    S::from_real(sample::<S::Real>(seed, row, column, 0).abs() + shift)
                } else if row < column {
                    generated_value(seed, row, column)
                } else {
                    generated_value::<S>(seed, column, row).conj()
                }
            })
            .collect(),
    )
    .unwrap()
}

/// Generate an upper-triangular non-normal matrix with controlled spectrum.
pub fn triangular_spectrum<S: InvariantScalar>(values: &[S::Real], seed: u64) -> Matrix<S> {
    let order = values.len();
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let row = index / order;
                let column = index % order;
                if row > column {
                    S::ZERO
                } else if row == column {
                    S::from_real(values[row])
                } else {
                    generated_value(seed, row, column)
                }
            })
            .collect(),
    )
    .unwrap()
}

/// Generate a rectangular diagonal matrix with a prescribed singular spectrum.
pub fn singular_spectrum<S: InvariantScalar>(
    rows: usize,
    cols: usize,
    values: &[S::Real],
) -> Matrix<S> {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let column = index % cols;
                if row == column && row < values.len() {
                    S::from_real(values[row])
                } else {
                    S::ZERO
                }
            })
            .collect(),
    )
    .unwrap()
}

pub fn scale_matrix<S: InvariantScalar>(matrix: &Matrix<S>, scale: S::Real) -> Matrix<S> {
    Matrix::from_vec(
        matrix.rows(),
        matrix.cols(),
        matrix
            .as_slice()
            .iter()
            .map(|&value| value * S::from_real(scale))
            .collect(),
    )
    .unwrap()
}

pub fn jordan_matrix<S: InvariantScalar>(order: usize, diagonal: S) -> Matrix<S> {
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let row = index / order;
                let column = index % order;
                if row == column {
                    diagonal
                } else if column == row + 1 {
                    S::ONE
                } else {
                    S::ZERO
                }
            })
            .collect(),
    )
    .unwrap()
}

pub fn assert_metric<R: RealScalar>(fixture: &str, metric: &str, value: R, bound: R) {
    assert!(
        value.is_finite() && value <= bound,
        "{fixture}: {metric} residual {value:?} exceeds {bound:?}"
    );
}

pub fn assert_hermitian_invariants<S: InvariantScalar>(
    matrix: &Matrix<S>,
    decomposition: &HermitianEigenDecomposition<S>,
    fixture: &str,
) {
    let order = matrix.rows();
    assert_eq!(decomposition.eigenvalues().len(), order);
    assert_eq!(
        (
            decomposition.eigenvectors().rows(),
            decomposition.eigenvectors().cols()
        ),
        (order, order)
    );
    assert!(
        decomposition
            .eigenvalues()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_real_spectrum_ascending(decomposition.eigenvalues());
    let bound = invariant_bound::<S::Real>(order);
    assert_metric(
        fixture,
        "eigenpair",
        hermitian_eigen_residual(
            matrix,
            decomposition.eigenvalues(),
            decomposition.eigenvectors(),
        ),
        bound,
    );
    assert_metric(
        fixture,
        "orthogonality",
        column_orthogonality_residual(decomposition.eigenvectors()),
        bound,
    );
}

pub fn assert_general_invariants<S: InvariantScalar>(
    matrix: &Matrix<S>,
    decomposition: &GeneralEigenDecomposition<S::Real>,
    fixture: &str,
) where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let order = matrix.rows();
    let right_eigenvectors = decomposition.right_eigenvectors().unwrap();
    assert_eq!(decomposition.eigenvalues().len(), order);
    assert_eq!(
        (right_eigenvectors.rows(), right_eigenvectors.cols()),
        (order, order)
    );
    assert!(
        decomposition
            .eigenvalues()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert_complex_spectrum_ascending(decomposition.eigenvalues());
    let bound = invariant_bound::<S::Real>(order);
    assert_metric(
        fixture,
        "eigenpair",
        general_eigen_residual(matrix, decomposition.eigenvalues(), right_eigenvectors),
        bound,
    );
    assert_metric(
        fixture,
        "vector normalization",
        column_normalization_residual(right_eigenvectors),
        bound,
    );
    if let Some(left_eigenvectors) = decomposition.left_eigenvectors() {
        assert_eq!(
            (left_eigenvectors.rows(), left_eigenvectors.cols()),
            (order, order)
        );
        assert_metric(
            fixture,
            "left eigenpair",
            general_left_eigen_residual(matrix, decomposition.eigenvalues(), left_eigenvectors),
            bound,
        );
        assert_metric(
            fixture,
            "left vector normalization",
            column_normalization_residual(left_eigenvectors),
            bound,
        );
        assert_eq!(decomposition.left_residuals().unwrap().len(), order);
    }
    assert_eq!(decomposition.right_residuals().unwrap().len(), order);
    assert_eq!(decomposition.eigenvector_condition_estimates().len(), order);
    assert_eq!(decomposition.balancing().scale_factors().len(), order);
    assert_eq!(decomposition.balancing().permutation().len(), order);
    if decomposition.left_eigenvectors().is_some() {
        assert_eq!(
            decomposition
                .eigenvalue_condition_estimates()
                .unwrap()
                .len(),
            order
        );
    }
}

pub fn assert_values_invariants<R: RealScalar>(values: &SvdValues<R>, rows: usize, cols: usize) {
    assert_eq!(values.singular_values().len(), rows.min(cols));
    assert_eq!((values.rows(), values.cols()), (rows, cols));
    assert!(
        values
            .singular_values()
            .iter()
            .all(|&value| { value.is_finite() && value >= R::ZERO })
    );
    assert!(
        values
            .singular_values()
            .windows(2)
            .all(|pair| pair[0] >= pair[1])
    );
}

pub fn assert_thin_svd_invariants<S: InvariantScalar>(
    matrix: &Matrix<S>,
    decomposition: &ThinSvdDecomposition<S>,
    fixture: &str,
) {
    let order = matrix.rows().min(matrix.cols());
    assert_eq!(decomposition.singular_values().len(), order);
    assert_eq!(
        (decomposition.rows(), decomposition.cols()),
        (matrix.rows(), matrix.cols())
    );
    assert_eq!(
        (
            decomposition.u_matrix().rows(),
            decomposition.u_matrix().cols()
        ),
        (matrix.rows(), order)
    );
    assert_eq!(
        (
            decomposition.vt_matrix().rows(),
            decomposition.vt_matrix().cols()
        ),
        (order, matrix.cols())
    );
    assert_svd_spectrum(decomposition.singular_values());
    let residuals = svd_residuals(
        matrix,
        decomposition.u_matrix(),
        decomposition.singular_values(),
        decomposition.vt_matrix(),
    );
    let bound = invariant_bound::<S::Real>(matrix.rows().max(matrix.cols()));
    assert_metric(fixture, "reconstruction", residuals.reconstruction, bound);
    assert_metric(
        fixture,
        "left orthogonality",
        residuals.left_orthogonality,
        bound,
    );
    assert_metric(
        fixture,
        "right orthogonality",
        residuals.right_orthogonality,
        bound,
    );
}

pub fn assert_full_svd_invariants<S: InvariantScalar>(
    matrix: &Matrix<S>,
    decomposition: &SvdDecomposition<S>,
    fixture: &str,
) {
    assert_eq!(
        decomposition.singular_values().len(),
        matrix.rows().min(matrix.cols())
    );
    assert_eq!(
        (decomposition.rows(), decomposition.cols()),
        (matrix.rows(), matrix.cols())
    );
    assert_eq!(
        (
            decomposition.u_matrix().rows(),
            decomposition.u_matrix().cols()
        ),
        (matrix.rows(), matrix.rows())
    );
    assert_eq!(
        (
            decomposition.vt_matrix().rows(),
            decomposition.vt_matrix().cols()
        ),
        (matrix.cols(), matrix.cols())
    );
    assert_svd_spectrum(decomposition.singular_values());
    let residuals = svd_residuals(
        matrix,
        decomposition.u_matrix(),
        decomposition.singular_values(),
        decomposition.vt_matrix(),
    );
    let bound = invariant_bound::<S::Real>(matrix.rows().max(matrix.cols()));
    assert_metric(fixture, "reconstruction", residuals.reconstruction, bound);
    assert_metric(
        fixture,
        "left orthogonality",
        residuals.left_orthogonality,
        bound,
    );
    assert_metric(
        fixture,
        "right orthogonality",
        residuals.right_orthogonality,
        bound,
    );
}

pub fn assert_real_spectra_agree<R: RealScalar>(fixture: &str, left: &[R], right: &[R], n: usize) {
    assert_metric(
        fixture,
        "spectrum agreement",
        real_spectrum_difference(left, right),
        invariant_bound::<R>(n),
    );
}

pub fn assert_complex_spectra_agree<R: RealScalar>(
    fixture: &str,
    left: &[Complex<R>],
    right: &[Complex<R>],
    n: usize,
) where
    Complex<R>: NumericScalar<Real = R>,
{
    assert_metric(
        fixture,
        "spectrum agreement",
        complex_spectrum_difference(left, right),
        invariant_bound::<R>(n),
    );
}

pub fn assert_real_spectrum_ascending<R: RealScalar>(values: &[R]) {
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
}

pub fn assert_complex_spectrum_ascending<R: RealScalar>(values: &[Complex<R>]) {
    assert!(values.windows(2).all(|pair| {
        pair[0].re < pair[1].re || (pair[0].re == pair[1].re && pair[0].im >= pair[1].im)
    }));
}

fn assert_svd_spectrum<R: RealScalar>(singular_values: &[R]) {
    assert!(
        singular_values
            .iter()
            .all(|&value| value.is_finite() && value >= R::ZERO)
    );
    assert!(singular_values.windows(2).all(|pair| pair[0] >= pair[1]));
}
