#![allow(clippy::unwrap_used)]

use crate::support::metrics as decomposition_metrics;

use scioxide_core::Complex;
use scioxide_core::{Budget, NumericalIssue};
use scioxide_linalg::{EigenIndexRange, EigenvalueAccuracy};
use scioxide_linalg::{
    EigenvalueOrder, GeneralEigenDecomposition, GeneralEigenOptions, GeneralEigenvectorSelection,
    LinalgWorkspace,
};
use scioxide_linalg::{HermitianEigenDecomposition, HermitianEigenOptions, Matrix};

#[test]
fn structured_symmetric_tridiagonal_matches_dense_decomposition() {
    let diagonal = [2.0_f64, 3.0, 5.0, 7.0];
    let off_diagonal = [-1.0_f64, 0.5, 2.0];
    let structured = HermitianEigenDecomposition::tridiagonal(&diagonal, &off_diagonal).unwrap();
    let dense = Matrix::from_vec(
        4,
        4,
        vec![
            2.0, -1.0, 0.0, 0.0, -1.0, 3.0, 0.5, 0.0, 0.0, 0.5, 5.0, 2.0, 0.0, 0.0, 2.0, 7.0,
        ],
    )
    .unwrap();
    let dense =
        HermitianEigenDecomposition::decompose(&dense, scioxide_linalg::HermitianInput::Exact)
            .unwrap();

    structured
        .eigenvalues()
        .iter()
        .zip(dense.eigenvalues())
        .for_each(|(left, right)| assert!((left - right).abs() <= 1.0e-12));
}

#[test]
fn structured_symmetric_tridiagonal_validates_layout_and_finiteness() {
    assert!(matches!(
        (HermitianEigenDecomposition::tridiagonal(&[1.0_f64, 2.0], &[]))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (HermitianEigenDecomposition::tridiagonal(&[1.0_f64, f64::NAN], &[0.5]))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
}

#[test]
fn empty_and_singleton_jobs_have_typed_outputs() {
    let empty = Matrix::<Complex<f64>>::zeros(0, 0).unwrap();
    assert!(
        scioxide_linalg::HermitianEigenvaluesResult::<f64>::compute(
            &empty,
            scioxide_linalg::HermitianInput::Exact
        )
        .unwrap()
        .eigenvalues()
        .is_empty()
    );
    assert_eq!(
        HermitianEigenDecomposition::decompose(&empty, scioxide_linalg::HermitianInput::Exact)
            .unwrap()
            .eigenvectors()
            .rows(),
        0
    );
    assert!(
        scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&empty)
            .unwrap()
            .eigenvalues()
            .is_empty()
    );

    let singleton = Matrix::from_vec(1, 1, vec![Complex::new(-3.0_f64, 2.0)]).unwrap();
    let general = GeneralEigenDecomposition::<f64>::decompose(&singleton).unwrap();
    assert_eq!(general.eigenvalues(), &[Complex::new(-3.0, 2.0)]);
    assert_eq!(
        general.right_eigenvectors().unwrap().as_slice(),
        &[Complex::new(1.0, 0.0)]
    );
}

#[test]
fn supported_f32_jobs_preserve_real_hermitian_values() {
    let matrix = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(2.0_f32, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, -1.0),
            Complex::new(2.0, 0.0),
        ],
    )
    .unwrap();
    let values = scioxide_linalg::HermitianEigenvaluesResult::<f32>::compute(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    assert_eq!(values.eigenvalues().len(), 2);
    assert!((values.eigenvalues()[0] - 1.0).abs() < 1.0e-5);
    assert!((values.eigenvalues()[1] - 3.0).abs() < 1.0e-5);
    let vectors =
        HermitianEigenDecomposition::decompose(&matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    let residual = decomposition_metrics::hermitian_eigen_residual(
        &matrix,
        vectors.eigenvalues(),
        vectors.eigenvectors(),
    );
    let bound = decomposition_metrics::invariant_bound::<f32>(matrix.rows());
    assert!(
        residual <= bound,
        "Hermitian f32 residual {residual} exceeds {bound}"
    );
}

#[test]
fn repeated_hermitian_spectrum_checks_subspace_not_basis() {
    let matrix = Matrix::from_vec(
        4,
        4,
        vec![
            2.0_f64, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ],
    )
    .unwrap();
    let decomposition =
        HermitianEigenDecomposition::decompose(&matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    assert_eq!(decomposition.eigenvalues(), &[2.0, 2.0, 5.0, 5.0]);
    let vectors = decomposition.eigenvectors();
    for column in 0..2 {
        let lower_subspace_energy = vectors.view().get(2, column).unwrap().powi(2)
            + vectors.view().get(3, column).unwrap().powi(2);
        assert!(lower_subspace_energy < 1.0e-24);
    }
}

#[test]
fn ordering_and_phase_conventions_are_deterministic() {
    let matrix =
        Matrix::from_vec(3, 3, vec![3.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let first =
        HermitianEigenDecomposition::decompose(&matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    let second =
        HermitianEigenDecomposition::decompose(&matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    assert_eq!(first.eigenvalues(), second.eigenvalues());
    assert_eq!(first.eigenvectors(), second.eigenvectors());
    for column in 0..3 {
        let pivot = (0..3)
            .max_by(|&left, &right| {
                first
                    .eigenvectors()
                    .view()
                    .get(left, column)
                    .unwrap()
                    .abs()
                    .partial_cmp(
                        &first
                            .eigenvectors()
                            .view()
                            .get(right, column)
                            .unwrap()
                            .abs(),
                    )
                    .unwrap()
            })
            .unwrap();
        assert!(first.eigenvectors().view().get(pivot, column).unwrap() >= 0.0);
    }

    let descending = HermitianEigenDecomposition::decompose_with_options(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        &HermitianEigenOptions::default().with_ordering(EigenvalueOrder::Descending),
    )
    .unwrap();
    assert!(
        descending
            .eigenvalues()
            .windows(2)
            .all(|pair| pair[0] >= pair[1])
    );
}

#[test]
fn general_real_conjugates_are_adjacent_positive_first() {
    let matrix = Matrix::from_vec(
        4,
        4,
        vec![
            0.0_f64, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0,
        ],
    )
    .unwrap();
    let values = scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
    assert!((values.eigenvalues()[0] - Complex::new(0.0, 2.0)).norm() < 1.0e-12);
    assert_eq!(values.eigenvalues()[1], values.eigenvalues()[0].conj());
    assert_eq!(values.eigenvalues()[2], Complex::new(3.0, 0.0));
    assert_eq!(values.eigenvalues()[3], Complex::new(4.0, 0.0));
}

#[test]
fn balancing_handles_widely_separated_scales() {
    let matrix = Matrix::from_vec(
        3,
        3,
        vec![
            1.0e-120_f64,
            1.0e80,
            0.0,
            1.0e-200,
            2.0,
            1.0e100,
            0.0,
            1.0e-100,
            3.0e120,
        ],
    )
    .unwrap();
    let values = scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
    assert_eq!(values.eigenvalues().len(), 3);
    assert!(values.eigenvalues().iter().all(|value| value.is_finite()));
}

#[test]
fn forced_nonconvergence_preserves_caller_outputs() {
    let matrix = Matrix::from_vec(
        4,
        4,
        vec![
            4.0_f64, 2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 0.0, 3.0, 1.0, 2.0, 4.0, 2.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let budget = Budget::ONE;

    let mut hermitian_result = HermitianEigenDecomposition::empty();
    let mut hermitian_workspace = LinalgWorkspace::new();
    let symmetric =
        Matrix::from_vec(3, 3, vec![3.0_f64, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 5.0]).unwrap();
    let error = HermitianEigenDecomposition::decompose_into(
        &symmetric,
        scioxide_linalg::HermitianInput::Exact,
        &HermitianEigenOptions::default().with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(budget),
        ),
        &mut hermitian_result,
        &mut hermitian_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert!(hermitian_result.eigenvalues().is_empty());

    let mut general_result = GeneralEigenDecomposition::empty();
    let mut general_workspace = LinalgWorkspace::new();
    let error = GeneralEigenDecomposition::decompose_into(
        &matrix,
        &GeneralEigenOptions::default().with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(budget),
        ),
        &mut general_result,
        &mut general_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert!(matches!(
        (error).view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert!(general_result.eigenvalues().is_empty());
}

#[test]
fn defective_vector_error_is_not_qr_nonconvergence() {
    let jordan =
        Matrix::from_vec(3, 3, vec![1.0_f64, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    assert!(scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&jordan).is_ok());
    assert!(matches!(
        (GeneralEigenDecomposition::<f64>::decompose(&jordan))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::DefectiveEigenbasis,
            ..
        }
    ));
}

#[test]
fn selected_eigenpairs_use_ascending_indices_before_output_order() {
    let n = 69;
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let row = index / n;
                let col = index % n;
                if row == col {
                    2.0_f64
                } else if row.abs_diff(col) == 1 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    let full =
        HermitianEigenDecomposition::decompose(&matrix, scioxide_linalg::HermitianInput::Exact)
            .unwrap();
    let options = HermitianEigenOptions::default()
        .with_index_range(EigenIndexRange::new(3, 8).unwrap())
        .with_ordering(EigenvalueOrder::Descending);
    let selected = HermitianEigenDecomposition::decompose_with_options(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options,
    )
    .unwrap();
    assert_eq!(
        (
            selected.eigenvectors().rows(),
            selected.eigenvectors().cols()
        ),
        (n, 5)
    );
    for (column, (&value, &expected)) in selected
        .eigenvalues()
        .iter()
        .zip(full.eigenvalues()[3..8].iter().rev())
        .enumerate()
    {
        assert!((value - expected).abs() <= 100.0 * n as f64 * f64::EPSILON);
        for row in 0..n {
            let v = selected.eigenvectors().as_slice()[row * 5 + column];
            let residual = (2.0 - value) * v
                - if row > 0 {
                    selected.eigenvectors().as_slice()[(row - 1) * 5 + column]
                } else {
                    0.0
                }
                - if row + 1 < n {
                    selected.eigenvectors().as_slice()[(row + 1) * 5 + column]
                } else {
                    0.0
                };
            assert!(residual.abs() <= 100.0 * n as f64 * f64::EPSILON);
        }
    }
    let values = scioxide_linalg::HermitianEigenvaluesResult::compute_with_options(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options,
    )
    .unwrap();
    assert_eq!(values.eigenvalues(), selected.eigenvalues());
    assert!(EigenIndexRange::new(9, 8).is_err());
    assert!(
        HermitianEigenDecomposition::decompose_with_options(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            &HermitianEigenOptions::default()
                .with_index_range(EigenIndexRange::new(0, n + 1).unwrap())
        )
        .is_err()
    );
}

#[test]
fn relative_bisection_preserves_small_diagonal_eigenvalues() {
    let diagonal = [1.0e-250_f64, 1.0e-100, 1.0e-20, 1.0];
    let result = HermitianEigenDecomposition::tridiagonal_with_options(
        &diagonal,
        &[0.0; 3],
        &HermitianEigenOptions::default().with_eigenvalue_accuracy(EigenvalueAccuracy::Relative),
    )
    .unwrap();
    for (&actual, &expected) in result.eigenvalues().iter().zip(&diagonal) {
        assert!(
            (actual / expected - 1.0).abs() <= 8.0 * f64::EPSILON,
            "{actual} vs {expected}"
        );
    }
}

/// Seeded entries in `[-1/2, 1/2)` for an `n x n` matrix.
fn random_matrix(n: usize, seed: u64) -> Matrix<f64> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    };
    Matrix::from_vec(n, n, (0..n * n).map(|_| next()).collect()).unwrap()
}

fn complex_copy(matrix: &Matrix<f64>) -> Matrix<Complex<f64>> {
    Matrix::from_vec(
        matrix.rows(),
        matrix.cols(),
        matrix
            .as_slice()
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect(),
    )
    .unwrap()
}

/// Right and left eigenpair residuals within the shared invariant bound.
fn assert_eigenpairs<S: scioxide_core::NumericScalar<Real = f64>>(
    matrix: &Matrix<S>,
    decomposition: &GeneralEigenDecomposition<f64>,
    fixture: &str,
) {
    let bound = decomposition_metrics::invariant_bound::<f64>(matrix.rows());
    let values = decomposition.eigenvalues();
    if let Some(right) = decomposition.right_eigenvectors() {
        let residual = decomposition_metrics::general_eigen_residual(matrix, values, right);
        assert!(residual <= bound, "{fixture}: right residual {residual}");
        let normalization = decomposition_metrics::column_normalization_residual(right);
        assert!(normalization <= bound, "{fixture}: right normalization");
    }
    if let Some(left) = decomposition.left_eigenvectors() {
        let residual = decomposition_metrics::general_left_eigen_residual(matrix, values, left);
        assert!(residual <= bound, "{fixture}: left residual {residual}");
        let normalization = decomposition_metrics::column_normalization_residual(left);
        assert!(normalization <= bound, "{fixture}: left normalization");
    }
}

#[test]
fn real_conjugate_pairs_stay_real_on_both_sides() {
    // Dense random matrices have many conjugate pairs; order 150 also runs
    // the blocked Hessenberg reduction and several back-transformation blocks.
    let options = GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
    for (n, seed) in [(6, 3), (150, 5)] {
        let matrix = random_matrix(n, seed);
        let decomposition =
            GeneralEigenDecomposition::decompose_with_options(&matrix, &options).unwrap();
        let fixture = format!("order {n}");
        assert_eigenpairs(&matrix, &decomposition, &fixture);
        let values = decomposition.eigenvalues();
        let pairs = (0..n).filter(|&k| values[k].im > 0.0).count();
        assert!(pairs > 0, "{fixture}: no conjugate pair");
        // Conjugate eigenvalues and vectors are exact conjugates, positive first.
        for k in (0..n).filter(|&k| values[k].im > 0.0) {
            assert_eq!(values[k + 1], values[k].conj(), "{fixture}");
            for vectors in [
                decomposition.right_eigenvectors().unwrap(),
                decomposition.left_eigenvectors().unwrap(),
            ] {
                for row in 0..n {
                    assert_eq!(
                        vectors.view().get(row, k + 1).unwrap(),
                        vectors.view().get(row, k).unwrap().conj()
                    );
                }
            }
        }
        let conditions = decomposition.eigenvalue_condition_estimates().unwrap();
        assert!(conditions.iter().all(|estimate| {
            (0.0..=1.0).contains(&estimate.reciprocal().value())
                && estimate.reciprocal().value() > 0.0
        }));
        // The complex-typed copy runs complex Schur QR on the same spectrum.
        let complex =
            GeneralEigenDecomposition::decompose_with_options(&complex_copy(&matrix), &options)
                .unwrap();
        assert_same_spectrum(complex.eigenvalues(), values, 1.0e-10);
    }

    let rotation = Matrix::from_vec(2, 2, vec![0.0_f32, -2.0, 1.0, 0.0]).unwrap();
    let single = GeneralEigenDecomposition::<f32>::decompose_with_options(
        &rotation,
        &GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both),
    )
    .unwrap();
    let values = single.eigenvalues();
    assert!((values[0] - Complex::new(0.0, 2.0_f32.sqrt())).norm() < 1.0e-6);
    assert_eq!(values[1], values[0].conj());
    let right = single.right_eigenvectors().unwrap();
    for row in 0..2 {
        assert_eq!(
            right.view().get(row, 1).unwrap(),
            right.view().get(row, 0).unwrap().conj()
        );
        let applied = (0..2).fold(Complex::new(0.0_f32, 0.0), |sum, k| {
            sum + Complex::new(rotation.view().get(row, k).unwrap(), 0.0)
                * right.view().get(k, 0).unwrap()
        });
        assert!((applied - values[0] * right.view().get(row, 0).unwrap()).norm() < 1.0e-5);
    }
}

#[test]
fn small_real_orders_publish_every_requested_output() {
    let options = GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
    let empty = GeneralEigenDecomposition::decompose_with_options(
        &Matrix::<f64>::zeros(0, 0).unwrap(),
        &options,
    )
    .unwrap();
    assert!(empty.eigenvalues().is_empty());
    assert_eq!(empty.left_eigenvectors().unwrap().rows(), 0);
    assert!(empty.right_residuals().unwrap().is_empty());

    let singleton = Matrix::from_vec(1, 1, vec![-2.5_f64]).unwrap();
    let one = GeneralEigenDecomposition::decompose_with_options(&singleton, &options).unwrap();
    assert_eq!(one.eigenvalues(), &[Complex::new(-2.5, 0.0)]);
    assert_eq!(
        one.right_eigenvectors().unwrap().as_slice(),
        &[Complex::new(1.0, 0.0)]
    );
    assert_eq!(
        one.left_eigenvectors().unwrap().as_slice(),
        &[Complex::new(1.0, 0.0)]
    );
    assert_eq!(
        one.eigenvalue_condition_estimates().unwrap()[0]
            .reciprocal()
            .value(),
        1.0
    );

    let pair = Matrix::from_vec(2, 2, vec![1.0_f64, -3.0, 2.0, 1.0]).unwrap();
    for selection in [
        GeneralEigenvectorSelection::Right,
        GeneralEigenvectorSelection::Left,
        GeneralEigenvectorSelection::Both,
    ] {
        let decomposition = GeneralEigenDecomposition::decompose_with_options(
            &pair,
            &GeneralEigenOptions::new().with_eigenvectors(selection),
        )
        .unwrap();
        assert_eigenpairs(&pair, &decomposition, &format!("{selection:?}"));
        assert!(
            (decomposition.eigenvalues()[0] - Complex::new(1.0, 6.0_f64.sqrt())).norm() < 1e-14
        );
    }
}

#[test]
fn balanced_vectors_return_to_input_coordinates_on_both_sides() {
    // `D M D^{-1}` with power-of-two grading spanning 2^12, and a first row
    // that permutation balancing isolates and moves to the bottom.
    let n = 6;
    let grading = [0, 4, -4, 8, -2, 0];
    let base = random_matrix(n, 11);
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| {
                let (row, col) = (index / n, index % n);
                if row == 0 {
                    if col == row { 0.75 } else { 0.0 }
                } else {
                    base.view().get(row, col).unwrap() * 2.0_f64.powi(grading[row] - grading[col])
                }
            })
            .collect(),
    )
    .unwrap();
    for selection in [
        GeneralEigenvectorSelection::Left,
        GeneralEigenvectorSelection::Both,
    ] {
        let decomposition = GeneralEigenDecomposition::decompose_with_options(
            &matrix,
            &GeneralEigenOptions::new().with_eigenvectors(selection),
        )
        .unwrap();
        let balancing = decomposition.balancing();
        assert!(balancing.scale_factors().iter().any(|&scale| scale != 1.0));
        assert_ne!(balancing.permutation(), &[0, 1, 2, 3, 4, 5]);
        let residuals = decomposition.left_residuals().unwrap();
        assert!(
            residuals
                .iter()
                .all(|&residual| residual <= 32.0 * n as f64 * f64::EPSILON)
        );
        // Residuals relative to each vector: the graded matrix norm hides
        // small-component errors, so check every left pair directly.
        let left = decomposition.left_eigenvectors().unwrap();
        for (column, &value) in decomposition.eigenvalues().iter().enumerate() {
            let scale = (0..n)
                .map(|row| left.view().get(row, column).unwrap().norm())
                .fold(0.0, f64::max);
            for col in 0..n {
                let applied = (0..n).fold(Complex::new(0.0, 0.0), |sum, row| {
                    sum + left.view().get(row, column).unwrap().conj()
                        * Complex::new(matrix.view().get(row, col).unwrap(), 0.0)
                });
                let expected = value * left.view().get(col, column).unwrap().conj();
                let row_norm = (0..n)
                    .map(|row| matrix.view().get(row, col).unwrap().abs())
                    .sum::<f64>();
                assert!(
                    (applied - expected).norm() <= 1e-12 * (row_norm + value.norm()) * scale,
                    "{selection:?}: column {column}"
                );
            }
        }
    }
}

#[test]
fn defective_bases_fail_on_every_side() {
    let defective = |matrix: &Matrix<f64>, selection| {
        matches!(
            GeneralEigenDecomposition::decompose_with_options(
                matrix,
                &GeneralEigenOptions::new().with_eigenvectors(selection)
            )
            .as_ref()
            .err()
            .map(|error| error.view()),
            Some(scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::DefectiveEigenbasis,
                ..
            })
        )
    };
    let fixtures = [
        (
            "2x2 Jordan block",
            Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, 1.0]),
        ),
        (
            "3x3 Jordan block",
            Matrix::from_vec(3, 3, vec![2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0]),
        ),
        // Coalescing real pair `1 +- 1e-10`: the unit diagonal dominates, so
        // balancing leaves the nearly parallel input basis unchanged.
        (
            "real pair",
            Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0e-20, 1.0]),
        ),
        // Coalescing conjugate pair `1 +- 1e-10 i` on the real 2x2 path.
        (
            "conjugate pair",
            Matrix::from_vec(2, 2, vec![1.0, 1.0, -1.0e-20, 1.0]),
        ),
        // A nearly defective conjugate pair coupled to a real eigenvalue.
        (
            "embedded pair",
            Matrix::from_vec(
                3,
                3,
                vec![1.0, 1.0, 0.5, -1.0e-20, 1.0, 0.25, 0.0, 0.0, 3.0],
            ),
        ),
    ];
    for (fixture, matrix) in fixtures {
        let matrix = matrix.unwrap();
        assert!(
            scioxide_linalg::GeneralEigenvaluesResult::compute(&matrix).is_ok(),
            "{fixture}: values"
        );
        for selection in [
            GeneralEigenvectorSelection::Right,
            GeneralEigenvectorSelection::Left,
            GeneralEigenvectorSelection::Both,
        ] {
            assert!(defective(&matrix, selection), "{fixture}: {selection:?}");
        }
    }
    // A well-separated pair of the same shape succeeds.
    let separated = Matrix::from_vec(2, 2, vec![1.0, 1.0, -1.0e-6, 1.0]).unwrap();
    assert!(!defective(&separated, GeneralEigenvectorSelection::Both));
}

#[test]
fn balanced_independence_accepts_graded_bases() {
    // Diagonal similarity exposes independent eigenvectors for these nearly
    // parallel input-coordinate bases, whose off-diagonal grading dominates.
    let (large, small) = (2.0_f64.powi(32), 2.0_f64.powi(-32));
    let fixtures = [
        (
            Matrix::from_vec(2, 2, vec![1.0, large, small, 1.0]).unwrap(),
            vec![Complex::new(0.0, 0.0), Complex::new(2.0, 0.0)],
        ),
        (
            Matrix::from_vec(2, 2, vec![1.0, large, -small, 1.0]).unwrap(),
            vec![Complex::new(1.0, 1.0), Complex::new(1.0, -1.0)],
        ),
        (
            Matrix::from_vec(
                3,
                3,
                vec![1.0, large, 0.5, -small, 1.0, 0.25, 0.0, 0.0, 3.0],
            )
            .unwrap(),
            vec![
                Complex::new(1.0, 1.0),
                Complex::new(1.0, -1.0),
                Complex::new(3.0, 0.0),
            ],
        ),
    ];
    for (matrix, expected) in fixtures {
        for selection in [
            GeneralEigenvectorSelection::Right,
            GeneralEigenvectorSelection::Left,
            GeneralEigenvectorSelection::Both,
        ] {
            let options = GeneralEigenOptions::new().with_eigenvectors(selection);
            let factor =
                GeneralEigenDecomposition::decompose_with_options(&matrix, &options).unwrap();
            assert_eigenpairs(&matrix, &factor, "balanced independent basis");
            assert_same_spectrum(factor.eigenvalues(), &expected, 1e-12);
            assert!(factor.balancing().scale_factors().iter().any(|&s| s != 1.0));
            assert!(matches!(
                GeneralEigenDecomposition::decompose_with_options(
                    &matrix,
                    &options.with_balancing(scioxide_linalg::GeneralEigenBalancing::None),
                )
                .unwrap_err()
                .view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::DefectiveEigenbasis,
                    ..
                }
            ));
        }
    }
}

fn exercise_extreme_balance_chain<S: scioxide_core::NumericScalar>(n: usize, subdiagonal: S::Real)
where
    Complex<S::Real>: scioxide_core::NumericScalar<Real = S::Real>,
{
    use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, Scaled};
    use scioxide_linalg::GeneralEigenvaluesResult;
    let real = |value| S::Real::try_from_usize_exact(value).unwrap();
    let root = subdiagonal.sqrt();
    let bound = decomposition_metrics::invariant_bound::<S::Real>(n);
    for transpose in [false, true] {
        for uniform_scale in [S::Real::ONE, root, S::Real::ONE / root] {
            let matrix = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| {
                        let (row, column) = if transpose {
                            (i % n, i / n)
                        } else {
                            (i / n, i % n)
                        };
                        S::from_real(
                            (if column == row + 1 {
                                S::Real::ONE
                            } else if row == column + 1 {
                                subdiagonal
                            } else {
                                S::Real::ZERO
                            }) * uniform_scale,
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let options =
                GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
            let factor =
                GeneralEigenDecomposition::decompose_with_options(&matrix, &options).unwrap();
            let values = GeneralEigenvaluesResult::compute_with_options(&matrix, &options).unwrap();
            for (j, &value) in factor.eigenvalues().iter().enumerate() {
                let theta = real(n - j) * S::Real::PI / real(n + 1);
                let eigen_scale = root * uniform_scale;
                let expected = S::Real::TWO * eigen_scale * theta.cos();
                for actual in [value, values.eigenvalues()[j]] {
                    assert!(
                        (actual.re - expected).abs() <= bound * eigen_scale,
                        "n={n}, transpose={transpose}, actual={actual:?}, expected={expected:?}"
                    );
                    assert!(actual.im.abs() <= bound * eigen_scale);
                }
                for (vectors, reverse) in [
                    (factor.right_eigenvectors().unwrap(), transpose),
                    (factor.left_eigenvectors().unwrap(), !transpose),
                ] {
                    let anchor = vectors.as_slice()[if reverse { (n - 1) * n + j } else { j }];
                    let mut grading = Scaled::ONE;
                    for i in 0..n {
                        let row = if reverse { n - 1 - i } else { i };
                        let actual = vectors.as_slice()[row * n + j] / anchor;
                        let expected = grading
                            .mul(
                                Scaled::from_value((real(i + 1) * theta).sin() / theta.sin())
                                    .unwrap(),
                            )
                            .unwrap()
                            .round()
                            .unwrap();
                        let allowance =
                            bound * grading.round().unwrap() + S::Real::MIN_SUBNORMAL * real(4);
                        assert!(
                            (actual.re - expected).abs() <= allowance,
                            "n={n}, transpose={transpose}, reverse={reverse}, row={row}, actual={actual:?}, expected={expected:?}, allowance={allowance:?}"
                        );
                        assert!(actual.im.abs() <= allowance);
                        grading = grading.mul(Scaled::from_value(root).unwrap()).unwrap();
                    }
                    assert!(decomposition_metrics::column_normalization_residual(vectors) <= bound);
                    assert!(
                        vectors
                            .as_slice()
                            .iter()
                            .all(|v| v.re.is_finite() && v.im.is_finite())
                    );
                }
            }
            assert!(
                factor
                    .balancing()
                    .scale_factors()
                    .iter()
                    .all(|&s| s.is_finite()
                        && s >= S::Real::MIN_POSITIVE
                        && (S::Real::ONE / s).is_finite())
            );
            assert!(
                decomposition_metrics::general_eigen_residual(
                    &matrix,
                    factor.eigenvalues(),
                    factor.right_eigenvectors().unwrap()
                ) <= bound
            );
            assert!(
                decomposition_metrics::general_left_eigen_residual(
                    &matrix,
                    factor.eigenvalues(),
                    factor.left_eigenvectors().unwrap()
                ) <= bound
            );
        }
    }
}

#[test]
fn extreme_balance_chains_f64() {
    exercise_extreme_balance_chain::<f64>(6, 1e-150);
    exercise_extreme_balance_chain::<Complex<f64>>(6, 1e-150);
}

#[test]
fn extreme_balance_chains_f32() {
    exercise_extreme_balance_chain::<f32>(5, 1e-25);
    exercise_extreme_balance_chain::<Complex<f32>>(5, 1e-25);
}

/// Lazy cyclic Markov chain `P_ii = 1 - p_i`, `P_i,i+1 = p_i` with graded
/// `p_i = 2^-(1 + grading*i)`. Every entry is dyadic, so rows sum to one
/// exactly and `1` is an exact eigenvalue.
fn graded_markov<S: scioxide_core::NumericScalar<Real = f64>>(n: usize, grading: i32) -> Matrix<S> {
    let mut values = vec![S::ZERO; n * n];
    for i in 0..n {
        let p = 2.0_f64.powi(-1 - grading * i as i32);
        values[i * n + i] = S::from_real(1.0 - p);
        values[i * n + (i + 1) % n] = S::from_real(p);
    }
    Matrix::from_vec(n, n, values).unwrap()
}

fn exercise_graded_markov<S: scioxide_core::NumericScalar<Real = f64>>() {
    for (n, grading) in [(8, 5), (10, 4)] {
        let matrix = graded_markov::<S>(n, grading);
        let options =
            GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
        let values =
            scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&matrix, &options)
                .unwrap();
        let factor = GeneralEigenDecomposition::decompose_with_options(&matrix, &options).unwrap();
        // Diagonal dominance keeps the chain unscaled: off-diagonal balancing
        // would trade its well-conditioned stationary eigenvalue for a graded
        // similarity whose QR backward error is amplified by many orders.
        assert!(factor.balancing().scale_factors().iter().all(|&s| s == 1.0));
        for spectrum in [values.eigenvalues(), factor.eigenvalues()] {
            let error = spectrum
                .iter()
                .map(|value| (*value - Complex::new(1.0, 0.0)).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(
                error <= 8.0 * n as f64 * f64::EPSILON,
                "n={n}, grading={grading}: |lambda - 1| = {error:e}"
            );
        }
        assert_eigenpairs(&matrix, &factor, "graded Markov chain");
    }
}

#[test]
fn graded_markov_chains_keep_the_stationary_eigenvalue() {
    exercise_graded_markov::<f64>();
    exercise_graded_markov::<Complex<f64>>();
}

/// Monic coefficients `c_0, ..., c_n` of Wilkinson's `prod_{k=1}^{n} (x - k)`,
/// formed exactly and rounded once (exact through `n = 18`).
fn wilkinson_coefficients(n: usize) -> Vec<f64> {
    let mut coefficients = vec![0_i128; n + 1];
    coefficients[0] = 1;
    for k in 1..=n as i128 {
        for i in (0..=n).rev() {
            let lower = if i == 0 { 0 } else { coefficients[i - 1] };
            coefficients[i] = lower - k * coefficients[i];
        }
    }
    coefficients.iter().map(|&c| c as f64).collect()
}

fn exercise_wilkinson_companion<S: scioxide_core::NumericScalar<Real = f64>>() {
    for n in [18, 20] {
        let coefficients = wilkinson_coefficients(n);
        // `scioxide_poly` layout: unit subdiagonal and a last column of `-c_i`.
        let mut values = vec![S::ZERO; n * n];
        for row in 0..n {
            if row > 0 {
                values[row * n + row - 1] = S::ONE;
            }
            values[row * n + n - 1] = S::from_real(-coefficients[row]);
        }
        let matrix = Matrix::from_vec(n, n, values).unwrap();
        let roots = scioxide_linalg::GeneralEigenvaluesResult::compute(&matrix).unwrap();
        // Coefficientwise backward error `|p(z)| / sum |c_j| |z|^j`, including
        // Horner rounding. Unbalanced or one-sidedly balanced companions lose
        // several orders of magnitude here at `n = 20`.
        for &root in roots.eigenvalues() {
            let (value, scale) = coefficients.iter().rev().fold(
                (Complex::new(0.0, 0.0), 0.0),
                |(value, scale), &c| {
                    (
                        value * root + Complex::new(c, 0.0),
                        scale * root.norm() + c.abs(),
                    )
                },
            );
            assert!(
                value.norm() <= 8.0 * n as f64 * f64::EPSILON * scale,
                "n={n}, root {root:?}: backward error {:e}",
                value.norm() / scale
            );
        }
    }
}

#[test]
fn wilkinson_companion_roots_are_coefficientwise_backward_stable() {
    exercise_wilkinson_companion::<f64>();
    exercise_wilkinson_companion::<Complex<f64>>();
}

/// Match a spectrum as a multiset: complex QR need not round conjugates to
/// identical real parts, so lexicographic positions need not agree.
fn assert_same_spectrum(actual: &[Complex<f64>], expected: &[Complex<f64>], bound: f64) {
    let mut unmatched = expected.to_vec();
    for value in actual {
        let (index, distance) = unmatched
            .iter()
            .enumerate()
            .map(|(i, reference)| (i, (*value - *reference).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            distance <= bound,
            "{value:?}, distance {distance}, bound {bound}"
        );
        unmatched.swap_remove(index);
    }
    assert!(unmatched.is_empty());
}

#[test]
fn values_only_deflation_matches_known_real_and_complex_blocks() {
    use scioxide_linalg::{GeneralEigenvaluesResult, MatrixView, SchurDecomposition};
    for n in [8, 16, 64] {
        let mut matrix = Matrix::<f64>::zeros(n, n).unwrap();
        let mut expected = Vec::new();
        for k in 0..n / 2 {
            let i = 2 * k;
            let diagonal = 3.0 * k as f64 + 1.0;
            let (b, c) = match k % 3 {
                0 => (-0.5, 0.25),
                1 => (0.5, 0.25),
                _ => (-1e-10, 1e-10),
            };
            let a = matrix.as_mut_slice();
            a[i * n + i] = diagonal;
            a[(i + 1) * n + i + 1] = diagonal;
            a[i * n + i + 1] = b;
            a[(i + 1) * n + i] = c;
            let gap = (b * c).abs().sqrt();
            if b * c < 0.0 {
                expected.extend([Complex::new(diagonal, gap), Complex::new(diagonal, -gap)]);
            } else {
                expected.extend([
                    Complex::new(diagonal - gap, 0.0),
                    Complex::new(diagonal + gap, 0.0),
                ]);
            }
            // Upper couplings must not affect either deflated block's values.
            for row in 0..i {
                a[row * n + i] = (row + i + 1) as f64 / n as f64;
                a[row * n + i + 1] = -0.25;
            }
        }
        let bound = 256.0 * n as f64 * f64::EPSILON;
        for scale in [f64::MIN_POSITIVE, 1.0, 1e200] {
            let scaled = matrix
                .as_slice()
                .iter()
                .map(|x| x * scale)
                .collect::<Vec<_>>();
            let view = MatrixView::from_slice(n, n, &scaled).unwrap();
            // The transpose follows a different reduction and layout path.
            for input in [view, view.transpose()] {
                let result = GeneralEigenvaluesResult::compute(input).unwrap();
                let restored = result
                    .eigenvalues()
                    .iter()
                    .map(|v| *v / scale)
                    .collect::<Vec<_>>();
                assert_same_spectrum(&restored, &expected, bound * n as f64);
            }
        }
        let vectors = GeneralEigenDecomposition::decompose(&matrix).unwrap();
        let schur = SchurDecomposition::decompose(&matrix).unwrap();
        assert_same_spectrum(vectors.eigenvalues(), &expected, bound * n as f64);
        assert_same_spectrum(schur.eigenvalues(), &expected, bound * n as f64);
    }
}

fn near_real_spectrum<S: scioxide_core::NumericScalar<Real = f64>>() {
    use scioxide_linalg::SchurDecomposition;
    for delta in [1e-6, 1e-10, 1e-14] {
        let a =
            Matrix::from_vec(2, 2, [1.0, -delta, delta, 1.0].map(S::from_real).to_vec()).unwrap();
        let options =
            GeneralEigenOptions::new().with_eigenvectors(GeneralEigenvectorSelection::Both);
        let values =
            scioxide_linalg::GeneralEigenvaluesResult::compute_with_options(&a, &options).unwrap();
        let vectors = GeneralEigenDecomposition::decompose_with_options(&a, &options).unwrap();
        let schur = SchurDecomposition::decompose(&a).unwrap();
        let expected = [Complex::new(1.0, delta), Complex::new(1.0, -delta)];
        for actual in [
            values.eigenvalues(),
            vectors.eigenvalues(),
            schur.eigenvalues(),
        ] {
            assert_same_spectrum(actual, &expected, 8.0 * f64::EPSILON);
            assert!(actual.iter().all(|value| value.im != 0.0));
        }
        assert_eigenpairs(&a, &vectors, "near-real normal pair");
    }
    let theta = 1e-9_f64;
    let a = Matrix::from_vec(
        3,
        3,
        [
            0.5,
            0.,
            0.,
            0.,
            theta.cos(),
            -theta.sin(),
            0.,
            theta.sin(),
            theta.cos(),
        ]
        .map(S::from_real)
        .to_vec(),
    )
    .unwrap();
    let values = scioxide_linalg::GeneralEigenvaluesResult::compute(&a).unwrap();
    assert_same_spectrum(
        values.eigenvalues(),
        &[
            Complex::new(0.5, 0.),
            Complex::new(theta.cos(), theta.sin()),
            Complex::new(theta.cos(), -theta.sin()),
        ],
        8.0 * f64::EPSILON,
    );
}

#[test]
fn near_real_pairs_keep_computed_imaginary_parts() {
    near_real_spectrum::<f64>();
    near_real_spectrum::<Complex<f64>>();
}

#[test]
fn general_vector_condition_estimates_are_scale_invariant() {
    for scale in [1e-10, 1.0, 1e10] {
        let a = Matrix::from_vec(
            3,
            3,
            [1., 1., 0., 0., 2., 1., 0., 0., 4.]
                .map(|x| x * scale)
                .to_vec(),
        )
        .unwrap();
        let matrix = complex_copy(&a);
        let result = GeneralEigenDecomposition::decompose(&matrix).unwrap();
        for (estimate, expected) in result
            .eigenvector_condition_estimates()
            .iter()
            .zip([0.25, 0.25, 0.5])
        {
            assert!((estimate.reciprocal().value() - expected).abs() <= 16.0 * f64::EPSILON);
        }
        let result = GeneralEigenDecomposition::decompose(&a).unwrap();
        for (estimate, expected) in result
            .eigenvector_condition_estimates()
            .iter()
            .zip([0.25, 0.25, 0.5])
        {
            assert!((estimate.reciprocal().value() - expected).abs() <= 16.0 * f64::EPSILON);
        }
    }
    for value in [0.0, 2.0] {
        let a = Matrix::from_vec(2, 2, vec![value, 0., 0., value]).unwrap();
        let result = GeneralEigenDecomposition::decompose(&a).unwrap();
        assert!(
            result
                .eigenvector_condition_estimates()
                .iter()
                .all(|estimate| estimate.reciprocal().value() == 0.0)
        );
    }
}
