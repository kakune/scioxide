//! Cross-decomposition invariant matrix for EVD and SVD.

#![allow(clippy::panic, clippy::unwrap_used)]

use crate::support::decomposition as support;
use scioxide_core::AlgebraicScalar;
use scioxide_core::NumericScalar;

use scioxide_core::Complex;
use scioxide_core::{ExecutionContext, ExecutionPolicy, NumericalIssue, RealScalar, SciOxideError};
use scioxide_linalg::{
    EigenvalueOrder, GeneralEigenDecomposition, GeneralEigenOptions, GeneralEigenvaluesBuffer,
    GeneralEigenvectorSelection,
};
use scioxide_linalg::{
    HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenvaluesBuffer,
    LinalgExecutionOptions, LinalgWorkspace, Matrix, SvdDecomposition, SvdValues,
    ThinSvdDecomposition,
};
use support::{
    InvariantScalar, assert_complex_spectra_agree, assert_complex_spectrum_ascending,
    assert_full_svd_invariants, assert_general_invariants, assert_hermitian_invariants,
    assert_real_spectra_agree, assert_real_spectrum_ascending, assert_thin_svd_invariants,
    assert_values_invariants, generated_matrix, hermitian_matrix, jordan_matrix, non_normal_matrix,
    scale_matrix, singular_spectrum, triangular_spectrum,
};

const EIGEN_SEED: u64 = 0x4556_442d_0293_0001;
const SVD_SEED: u64 = 0x5356_442d_0293_0001;

fn real<R: RealScalar>(value: f64) -> R {
    R::try_from_f64_lossy(value).unwrap()
}

fn assert_hermitian_fixture<S>(fixture: &str, matrix: &Matrix<S>)
where
    S: InvariantScalar,
{
    let options = HermitianEigenOptions::default();

    let values = scioxide_linalg::HermitianEigenvaluesResult::<S::Real>::compute(
        matrix,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    let mut values_into = HermitianEigenvaluesBuffer::empty();
    let mut values_workspace = LinalgWorkspace::new();
    HermitianEigenvaluesBuffer::compute_into(
        matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut values_into,
        &mut values_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();

    let decomposition =
        HermitianEigenDecomposition::decompose(matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    let mut decomposition_into = HermitianEigenDecomposition::empty();
    let mut decomposition_workspace = LinalgWorkspace::new();
    HermitianEigenDecomposition::decompose_into(
        matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut decomposition_into,
        &mut decomposition_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();

    assert_real_spectrum_ascending(values.eigenvalues());
    for other in [
        values_into.eigenvalues(),
        decomposition.eigenvalues(),
        decomposition_into.eigenvalues(),
    ] {
        assert_real_spectrum_ascending(other);
        assert_real_spectra_agree(fixture, values.eigenvalues(), other, matrix.rows());
    }
    for candidate in [&decomposition, &decomposition_into] {
        assert_hermitian_invariants(matrix, candidate, fixture);
    }

    let descending = scioxide_linalg::HermitianEigenvaluesResult::<S::Real>::compute_with_options(
        matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options.with_ordering(EigenvalueOrder::Descending),
    )
    .unwrap();
    assert!(
        descending
            .eigenvalues()
            .windows(2)
            .all(|pair| pair[0] >= pair[1]),
        "{fixture}: Hermitian values are not descending"
    );
}

fn exercise_hermitian_evd<S: InvariantScalar>() {
    let epsilon = <S::Real as RealScalar>::EPSILON;
    let clustered = real::<S::Real>(3.0) + real::<S::Real>(32.0) * epsilon;
    let base = hermitian_matrix::<S>(4, EIGEN_SEED);
    let fixtures = [
        ("empty", Matrix::zeros(0, 0).unwrap()),
        ("singleton", singular_spectrum::<S>(1, 1, &[real(-3.0)])),
        ("odd generated", hermitian_matrix::<S>(3, EIGEN_SEED + 1)),
        ("even generated", base.clone()),
        (
            "repeated spectrum",
            singular_spectrum::<S>(4, 4, &[real(2.0), real(2.0), real(5.0), real(5.0)]),
        ),
        (
            "clustered spectrum",
            singular_spectrum::<S>(4, 4, &[real(1.0), real(2.0), real(3.0), clustered]),
        ),
        ("large finite scale", scale_matrix(&base, S::large_scale())),
        ("small finite scale", scale_matrix(&base, S::small_scale())),
    ];

    fixtures.iter().for_each(|(fixture, matrix)| {
        assert_hermitian_fixture(&format!("{fixture} ({})", S::NAME), matrix);
    });
}

#[test]
fn hermitian_evd_real_f32_invariant_matrix() {
    exercise_hermitian_evd::<f32>();
}

#[test]
fn hermitian_evd_real_f64_invariant_matrix() {
    exercise_hermitian_evd::<f64>();
}

#[test]
fn hermitian_evd_complex_f32_invariant_matrix() {
    exercise_hermitian_evd::<Complex<f32>>();
}

#[test]
fn hermitian_evd_complex_f64_invariant_matrix() {
    exercise_hermitian_evd::<Complex<f64>>();
}

fn assert_general_fixture<S>(fixture: &str, matrix: &Matrix<S>)
where
    S: InvariantScalar,
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let options =
        GeneralEigenOptions::default().with_eigenvectors(GeneralEigenvectorSelection::Both);

    let values = scioxide_linalg::GeneralEigenvaluesResult::<S::Real>::compute(matrix).unwrap();
    let values_with_options =
        scioxide_linalg::GeneralEigenvaluesResult::<S::Real>::compute_with_options(
            matrix, &options,
        )
        .unwrap();
    let mut values_into = GeneralEigenvaluesBuffer::empty();
    let mut values_workspace = LinalgWorkspace::new();
    GeneralEigenvaluesBuffer::compute_into(
        matrix,
        &options,
        &mut values_into,
        &mut values_workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();

    let decomposition = GeneralEigenDecomposition::<S::Real>::decompose(matrix).unwrap();
    let decomposition_with_options =
        GeneralEigenDecomposition::<S::Real>::decompose_with_options(matrix, &options).unwrap();
    let mut decomposition_into = GeneralEigenDecomposition::empty();
    let mut decomposition_workspace = LinalgWorkspace::new();
    GeneralEigenDecomposition::decompose_into(
        matrix,
        &options,
        &mut decomposition_into,
        &mut decomposition_workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();

    assert_complex_spectrum_ascending(values.eigenvalues());
    for other in [
        values_with_options.eigenvalues(),
        values_into.eigenvalues(),
        decomposition.eigenvalues(),
        decomposition_with_options.eigenvalues(),
        decomposition_into.eigenvalues(),
    ] {
        assert_complex_spectrum_ascending(other);
        assert_complex_spectra_agree(fixture, values.eigenvalues(), other, matrix.rows());
    }
    for candidate in [
        &decomposition,
        &decomposition_with_options,
        &decomposition_into,
    ] {
        assert_general_invariants(matrix, candidate, fixture);
    }
}

fn assert_general_values_fixture<S>(fixture: &str, matrix: &Matrix<S>)
where
    S: InvariantScalar,
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let options = GeneralEigenOptions::default();
    let values = scioxide_linalg::GeneralEigenvaluesResult::<S::Real>::compute(matrix).unwrap();
    let values_with_options =
        scioxide_linalg::GeneralEigenvaluesResult::<S::Real>::compute(matrix).unwrap();
    let mut values_into = GeneralEigenvaluesBuffer::empty();
    let mut workspace = LinalgWorkspace::new();
    GeneralEigenvaluesBuffer::compute_into(
        matrix,
        &options,
        &mut values_into,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(values.eigenvalues().len(), matrix.rows());
    assert!(
        values
            .eigenvalues()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    for candidate in [
        values.eigenvalues(),
        values_with_options.eigenvalues(),
        values_into.eigenvalues(),
    ] {
        assert_complex_spectrum_ascending(candidate);
    }
    assert_complex_spectra_agree(
        fixture,
        values.eigenvalues(),
        values_with_options.eigenvalues(),
        matrix.rows(),
    );
    assert_complex_spectra_agree(
        fixture,
        values.eigenvalues(),
        values_into.eigenvalues(),
        matrix.rows(),
    );
}

fn assert_defective_general_fixture<S>(fixture: &str, matrix: &Matrix<S>)
where
    S: InvariantScalar,
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let options = GeneralEigenOptions::default();
    assert_general_values_fixture(fixture, matrix);

    let assert_defective = |result: Result<GeneralEigenDecomposition<S::Real>, SciOxideError>| {
        assert!(matches!(
            (result).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::DefectiveEigenbasis,
                ..
            }
        ));
    };
    assert_defective(GeneralEigenDecomposition::<S::Real>::decompose(matrix));
    assert_defective(GeneralEigenDecomposition::<S::Real>::decompose(matrix));

    let mut result = GeneralEigenDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    let into_result = GeneralEigenDecomposition::decompose_into(
        matrix,
        &options,
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    );
    assert!(matches!(
        (into_result).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::DefectiveEigenbasis,
            ..
        }
    ));
    assert!(result.eigenvalues().is_empty());
}

fn exercise_general_evd<S>()
where
    S: InvariantScalar,
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let base = non_normal_matrix::<S>(4, EIGEN_SEED + 10);
    let successful = [
        ("empty", Matrix::zeros(0, 0).unwrap()),
        (
            "singleton",
            triangular_spectrum::<S>(&[real(-3.0)], EIGEN_SEED + 11),
        ),
        ("odd normal", hermitian_matrix::<S>(3, EIGEN_SEED + 12)),
        ("even non-normal", base.clone()),
        ("large finite scale", scale_matrix(&base, S::large_scale())),
        ("small finite scale", scale_matrix(&base, S::small_scale())),
    ];
    successful.iter().for_each(|(fixture, matrix)| {
        assert_general_fixture(&format!("{fixture} ({})", S::NAME), matrix);
    });

    let epsilon = <S::Real as RealScalar>::EPSILON;
    for (fixture, matrix) in [
        (
            "repeated values-only spectrum",
            triangular_spectrum::<S>(
                &[real(1.0), real(1.0), real(2.0), real(2.0)],
                EIGEN_SEED + 13,
            ),
        ),
        (
            "clustered values-only spectrum",
            triangular_spectrum::<S>(
                &[
                    real(1.0),
                    S::Real::ONE + real::<S::Real>(32.0) * epsilon,
                    real(2.0),
                    real(3.0),
                ],
                EIGEN_SEED + 14,
            ),
        ),
    ] {
        assert_general_values_fixture(&format!("{fixture} ({})", S::NAME), &matrix);
    }

    let diagonal = S::from_parts(real(1.0), real(0.5));
    let defective = jordan_matrix::<S>(3, diagonal);
    assert_defective_general_fixture("defective Jordan matrix", &defective);

    let near_defective =
        triangular_spectrum::<S>(&[S::Real::ONE, S::Real::ONE + epsilon], EIGEN_SEED + 15);
    assert_defective_general_fixture("numerically defective matrix", &near_defective);
}

#[test]
fn general_evd_real_f32_invariant_matrix() {
    exercise_general_evd::<f32>();
}

#[test]
fn general_evd_real_f64_invariant_matrix() {
    exercise_general_evd::<f64>();
}

#[test]
fn general_evd_complex_f32_invariant_matrix() {
    exercise_general_evd::<Complex<f32>>();
}

#[test]
fn general_evd_complex_f64_invariant_matrix() {
    exercise_general_evd::<Complex<f64>>();
}

fn assert_svd_fixture<S: InvariantScalar>(fixture: &str, matrix: &Matrix<S>) {
    let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
    let values = scioxide_linalg::SvdValues::compute(matrix)
        .unwrap_or_else(|error| panic!("{fixture}: values job failed: {error}"));
    let values_with_options = scioxide_linalg::SvdValues::compute_with_options(matrix, &options)
        .unwrap_or_else(|error| panic!("{fixture}: values-with-options job failed: {error}"));
    let mut values_into = SvdValues::empty();
    let mut values_workspace = LinalgWorkspace::new();
    scioxide_linalg::SvdValues::compute_into(
        matrix,
        &options,
        &mut values_into,
        &mut values_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_or_else(|error| panic!("{fixture}: values-into job failed: {error}"));

    let thin = scioxide_linalg::ThinSvdDecomposition::compute(matrix)
        .unwrap_or_else(|error| panic!("{fixture}: thin job failed: {error}"));
    let thin_with_options =
        scioxide_linalg::ThinSvdDecomposition::compute_with_options(matrix, &options)
            .unwrap_or_else(|error| panic!("{fixture}: thin-with-options job failed: {error}"));
    let mut thin_into = ThinSvdDecomposition::empty();
    let mut thin_workspace = LinalgWorkspace::new();
    scioxide_linalg::ThinSvdDecomposition::compute_into(
        matrix,
        &options,
        &mut thin_into,
        &mut thin_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_or_else(|error| panic!("{fixture}: thin-into job failed: {error}"));

    let full = scioxide_linalg::SvdDecomposition::compute(matrix)
        .unwrap_or_else(|error| panic!("{fixture}: full job failed: {error}"));
    let full_with_options =
        scioxide_linalg::SvdDecomposition::compute_with_options(matrix, &options)
            .unwrap_or_else(|error| panic!("{fixture}: full-with-options job failed: {error}"));
    let mut full_into = SvdDecomposition::empty();
    let mut full_workspace = LinalgWorkspace::new();
    scioxide_linalg::SvdDecomposition::compute_into(
        matrix,
        &options,
        &mut full_into,
        &mut full_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_or_else(|error| panic!("{fixture}: full-into job failed: {error}"));
    for candidate in [&values, &values_with_options, &values_into] {
        assert_values_invariants(candidate, matrix.rows(), matrix.cols());
        assert_real_spectra_agree(
            fixture,
            values.singular_values(),
            candidate.singular_values(),
            matrix.rows().max(matrix.cols()),
        );
    }
    for candidate in [&thin, &thin_with_options, &thin_into] {
        assert_thin_svd_invariants(matrix, candidate, fixture);
        assert_real_spectra_agree(
            fixture,
            values.singular_values(),
            candidate.singular_values(),
            matrix.rows().max(matrix.cols()),
        );
    }
    for candidate in [&full, &full_with_options, &full_into] {
        assert_full_svd_invariants(matrix, candidate, fixture);
        assert_real_spectra_agree(
            fixture,
            values.singular_values(),
            candidate.singular_values(),
            matrix.rows().max(matrix.cols()),
        );
    }
}

fn exercise_svd<S: InvariantScalar>() {
    let epsilon = <S::Real as RealScalar>::EPSILON;
    let generated = generated_matrix::<S>(3, 2, SVD_SEED);
    let fixtures = [
        ("empty square", Matrix::zeros(0, 0).unwrap()),
        ("empty tall", Matrix::zeros(3, 0).unwrap()),
        ("empty wide", Matrix::zeros(0, 3).unwrap()),
        ("singleton", singular_spectrum::<S>(1, 1, &[real(7.0)])),
        ("odd square", generated_matrix::<S>(3, 3, SVD_SEED + 1)),
        ("odd tall", generated_matrix::<S>(5, 3, SVD_SEED + 2)),
        ("even wide", generated_matrix::<S>(2, 4, SVD_SEED + 3)),
        (
            "rank deficient",
            singular_spectrum::<S>(4, 3, &[real(4.0), real(1.0), S::Real::ZERO]),
        ),
        (
            "repeated spectrum",
            singular_spectrum::<S>(3, 4, &[real(3.0), real(3.0), real(1.0)]),
        ),
        (
            "clustered spectrum",
            singular_spectrum::<S>(
                4,
                3,
                &[
                    real(3.0),
                    real::<S::Real>(3.0) - real::<S::Real>(32.0) * epsilon,
                    real(1.0),
                ],
            ),
        ),
        (
            "small finite scale",
            scale_matrix(&generated, S::small_scale()),
        ),
        (
            "large finite scale",
            scale_matrix(&generated, S::large_scale()),
        ),
    ];

    for (rows, cols) in [(31, 33), (32, 32), (33, 39), (39, 33), (65, 65)] {
        let matrix = generated_matrix::<S>(rows, cols, SVD_SEED + 8);
        assert_svd_fixture(&format!("crossover {rows}x{cols} ({})", S::NAME), &matrix);
    }

    let large = generated_matrix::<S>(39, 33, SVD_SEED + 9);
    for scale in [S::small_scale(), S::large_scale()] {
        assert_svd_fixture(
            &format!("scaled divide and conquer ({})", S::NAME),
            &scale_matrix(&large, scale),
        );
    }

    fixtures.iter().for_each(|(fixture, matrix)| {
        assert_svd_fixture(&format!("{fixture} ({})", S::NAME), matrix);
    });
}

#[test]
fn svd_real_f32_invariant_matrix() {
    exercise_svd::<f32>();
}

#[test]
fn svd_real_f64_invariant_matrix() {
    exercise_svd::<f64>();
}

#[test]
fn svd_complex_f32_invariant_matrix() {
    exercise_svd::<Complex<f32>>();
}

#[test]
fn svd_complex_f64_invariant_matrix() {
    exercise_svd::<Complex<f64>>();
}
