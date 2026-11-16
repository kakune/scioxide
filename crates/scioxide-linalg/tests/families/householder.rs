//! Public Householder reconstruction, orthogonality, solve and pivoting contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy};
use scioxide_linalg::{
    DirectSolveOptions, DirectSolveTermination, HouseholderDecomposition, HouseholderOrientation,
    HouseholderPivoting, LinalgExecutionOptions, LinalgWorkspace, Matrix, RankPolicy, Vector,
};

/// Helper: check that two matrices are approximately equal.
fn assert_matrices_approx_eq(lhs: &Matrix<f64>, rhs: &Matrix<f64>, tol: f64) {
    assert_eq!(lhs.rows(), rhs.rows(), "row count mismatch");
    assert_eq!(lhs.cols(), rhs.cols(), "col count mismatch");
    for row in 0..lhs.rows() {
        for col in 0..lhs.cols() {
            let diff =
                (lhs.view().get(row, col).unwrap() - rhs.view().get(row, col).unwrap()).abs();
            assert!(
                diff < tol,
                "mismatch at ({row}, {col}): {} vs {}, diff = {diff}",
                lhs.view().get(row, col).unwrap(),
                rhs.view().get(row, col).unwrap()
            );
        }
    }
}

/// Helper: check that two matrices are approximately equal (f32).
fn assert_matrices_approx_eq_f32(lhs: &Matrix<f32>, rhs: &Matrix<f32>, tol: f32) {
    assert_eq!(lhs.rows(), rhs.rows(), "row count mismatch");
    assert_eq!(lhs.cols(), rhs.cols(), "col count mismatch");
    for row in 0..lhs.rows() {
        for col in 0..lhs.cols() {
            let diff =
                (lhs.view().get(row, col).unwrap() - rhs.view().get(row, col).unwrap()).abs();
            assert!(
                diff < tol,
                "mismatch at ({row}, {col}): {} vs {}, diff = {diff}",
                lhs.view().get(row, col).unwrap(),
                rhs.view().get(row, col).unwrap()
            );
        }
    }
}

fn assert_permuted_factorization(
    original: &Matrix<f64>,
    reconstructed: &Matrix<f64>,
    permutation: &[usize],
    tol: f64,
) {
    for row in 0..original.rows() {
        for (factor_column, &original_column) in permutation.iter().enumerate() {
            let expected = original.view().get(row, original_column).unwrap();
            let actual = reconstructed.view().get(row, factor_column).unwrap();
            assert!(
                (actual - expected).abs() < tol,
                "mismatch at ({row}, {factor_column}): {actual} vs {expected}"
            );
        }
    }
}

fn assert_permuted_factorization_f32(
    original: &Matrix<f32>,
    reconstructed: &Matrix<f32>,
    permutation: &[usize],
    tol: f32,
) {
    for row in 0..original.rows() {
        for (factor_column, &original_column) in permutation.iter().enumerate() {
            let expected = original.view().get(row, original_column).unwrap();
            let actual = reconstructed.view().get(row, factor_column).unwrap();
            assert!((actual - expected).abs() < tol);
        }
    }
}

// ---- A = Q * R roundtrip ----

#[test]
fn test_decompose_3x3_roundtrip() {
    let a = Matrix::<f64>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-10);
}

#[test]
fn test_decompose_identity_roundtrip() {
    let a = Matrix::<f64>::identity(4).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-14);
}

#[test]
fn test_decompose_2x2_roundtrip() {
    let a = Matrix::<f64>::from_vec(2, 2, vec![3.0, 1.0, 4.0, 1.5]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-12);
}

// ---- Q orthogonality: Q^T * Q = I ----

#[test]
fn test_q_orthogonality_3x3() {
    let a = Matrix::<f64>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let qt = q.view().adjoint().try_to_matrix().unwrap();
    let qtq = qt.view().mul_mat(&q).unwrap();
    let eye = Matrix::<f64>::identity(3).unwrap();
    assert_matrices_approx_eq(&qtq, &eye, 1e-12);
}

#[test]
fn test_q_orthogonality_4x4() {
    let a = Matrix::<f64>::from_vec(
        4,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0,
        ],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let qt = q.view().adjoint().try_to_matrix().unwrap();
    let qtq = qt.view().mul_mat(&q).unwrap();
    let eye = Matrix::<f64>::identity(4).unwrap();
    assert_matrices_approx_eq(&qtq, &eye, 1e-12);
}

// ---- R is upper triangular ----

#[test]
fn test_r_upper_triangular() {
    let a = Matrix::<f64>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    for row in 0..r.rows() {
        for col in 0..row.min(r.cols()) {
            assert!(
                r.view().get(row, col).unwrap().abs() < 1e-12,
                "R[{row},{col}] should be zero but is {}",
                r.view().get(row, col).unwrap()
            );
        }
    }
}

// ---- Solve square system ----

#[test]
fn test_solve_3x3() {
    // A = [[2,1,1],[4,3,3],[8,7,9]], b = [4,10,24], x = [1,1,1]
    let a =
        Matrix::<f64>::from_vec(3, 3, vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![4.0, 10.0, 24.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(x.len(), 3);
    for idx in 0..3 {
        assert!(
            (x.view().get(idx).unwrap() - 1.0).abs() < 1e-10,
            "x[{idx}] = {}, expected 1.0",
            x.view().get(idx).unwrap()
        );
    }
}

#[test]
fn test_solve_2x2() {
    // A = [[1,2],[3,4]], b = [5,11] => x = [1,2]
    let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![5.0, 11.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-12);
    assert!((x.view().get(1).unwrap() - 2.0).abs() < 1e-12);
}

// ---- Solve overdetermined (least-squares) ----

#[test]
fn test_solve_overdetermined_least_squares() {
    // A = [[1,1],[1,2],[1,3]], b = [1,2,2]
    // Least-squares solution: x = (A^T A)^{-1} A^T b
    // A^T A = [[3,6],[6,14]], A^T b = [5,11]
    // x = [2/3, 1/2]
    let a = Matrix::<f64>::from_vec(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![1.0, 2.0, 2.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert_eq!(x.len(), 2);
    assert!(
        (x.view().get(0).unwrap() - 2.0 / 3.0).abs() < 1e-10,
        "x[0] = {}, expected {}",
        x.view().get(0).unwrap(),
        2.0 / 3.0
    );
    assert!(
        (x.view().get(1).unwrap() - 0.5).abs() < 1e-10,
        "x[1] = {}, expected 0.5",
        x.view().get(1).unwrap()
    );
}

// ---- Rectangular matrix: m > n ----

#[test]
fn test_decompose_rectangular_4x3() {
    let a = Matrix::<f64>::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0,
        ],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    assert_eq!(qr.rows(), 4);
    assert_eq!(qr.cols(), 3);

    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();

    // Q is 4x4, R is 4x3
    assert_eq!(q.rows(), 4);
    assert_eq!(q.cols(), 4);
    assert_eq!(r.rows(), 4);
    assert_eq!(r.cols(), 3);

    // Q * R = A
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-10);

    // Q^T * Q = I
    let qt = q.view().adjoint().try_to_matrix().unwrap();
    let qtq = qt.view().mul_mat(&q).unwrap();
    let eye = Matrix::<f64>::identity(4).unwrap();
    assert_matrices_approx_eq(&qtq, &eye, 1e-12);
}

// ---- Rectangular matrix: m > n (5x2) ----

#[test]
fn test_decompose_rectangular_5x2() {
    let a = Matrix::<f64>::from_vec(5, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0])
        .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-10);
}

// ---- Wide QR ----

#[test]
fn test_decompose_more_cols_than_rows() {
    let a = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let reconstructed = qr
        .to_q_matrix()
        .unwrap()
        .view()
        .mul_mat(&qr.to_triangular_matrix().unwrap())
        .unwrap();
    assert_permuted_factorization(&a, &reconstructed, qr.permutation(), 1.0e-12);
}

#[test]
fn test_solve_dimension_mismatch() {
    let a = Matrix::<f64>::identity(3).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![1.0, 2.0]);
    assert!(matches!(
        (qr.solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new())
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
}

// ---- 1x1 matrix ----

#[test]
fn test_decompose_1x1() {
    let a = Matrix::<f64>::from_vec(1, 1, vec![5.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-14);
    // Q should be +/- 1
    assert!((q.view().get(0, 0).unwrap().abs() - 1.0).abs() < 1e-14);
}

#[test]
fn test_solve_1x1() {
    let a = Matrix::<f64>::from_vec(1, 1, vec![4.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![8.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((x.view().get(0).unwrap() - 2.0).abs() < 1e-14);
}

// ---- f32 tests ----

#[test]
fn test_decompose_f32_roundtrip() {
    let a = Matrix::<f32>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization_f32(&a, &qr_prod, qr.permutation(), 1e-4);
}

#[test]
fn test_q_orthogonality_f32() {
    let a = Matrix::<f32>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let qt = q.view().adjoint().try_to_matrix().unwrap();
    let qtq = qt.view().mul_mat(&q).unwrap();
    let eye = Matrix::<f32>::identity(3).unwrap();
    assert_matrices_approx_eq_f32(&qtq, &eye, 1e-5);
}

#[test]
fn test_solve_f32() {
    let a = Matrix::<f32>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![5.0_f32, 11.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((x.view().get(0).unwrap() - 1.0).abs() < 1e-5);
    assert!((x.view().get(1).unwrap() - 2.0).abs() < 1e-5);
}

// ---- Solve with identity ----

#[test]
fn test_solve_identity() {
    let a = Matrix::<f64>::identity(3).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let b = Vector::from_vec(vec![7.0, 8.0, 9.0]);
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    for idx in 0..3 {
        assert!(
            (x.view().get(idx).unwrap() - b.view().get(idx).unwrap()).abs() < 1e-14,
            "x[{idx}] = {}, expected {}",
            x.view().get(idx).unwrap(),
            b.view().get(idx).unwrap()
        );
    }
}

// ---- Verify A x = b after solve ----

#[test]
fn test_solve_verify_residual() {
    let a = Matrix::<f64>::from_vec(
        3,
        3,
        vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
    )
    .unwrap();
    let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let x = qr
        .solve_with_options(
            &b,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    // Compute A * x and compare with b
    let ax = a.view().mul_vec(&x).unwrap();
    for idx in 0..3 {
        assert!(
            (ax.view().get(idx).unwrap() - b.view().get(idx).unwrap()).abs() < 1e-10,
            "residual at {idx}: Ax[{idx}] = {}, b[{idx}] = {}",
            ax.view().get(idx).unwrap(),
            b.view().get(idx).unwrap()
        );
    }
}

#[test]
fn test_scaled_householder_roundtrip_f64() {
    for scale in [1e-150, 1e150] {
        let a = Matrix::from_vec(
            3,
            2,
            vec![
                scale,
                2.0 * scale,
                3.0 * scale,
                scale,
                2.0 * scale,
                4.0 * scale,
            ],
        )
        .unwrap();
        let qr = HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column,
        )
        .unwrap();
        assert!(
            qr.analyze_rank(RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .is_full_rank()
        );
        assert_eq!(
            qr.analyze_rank(RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .rank(),
            2
        );
        let reconstructed = qr
            .to_q_matrix()
            .unwrap()
            .view()
            .mul_mat(&qr.to_triangular_matrix().unwrap())
            .unwrap();
        assert_permuted_factorization(&a, &reconstructed, qr.permutation(), scale * 1e-12);
    }
}

#[test]
fn test_scaled_householder_roundtrip_f32() {
    for scale in [1e-20_f32, 1e20] {
        let a = Matrix::from_vec(
            3,
            2,
            vec![
                scale,
                2.0 * scale,
                3.0 * scale,
                scale,
                2.0 * scale,
                4.0 * scale,
            ],
        )
        .unwrap();
        let qr = HouseholderDecomposition::factor(
            &a,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column,
        )
        .unwrap();
        assert!(
            qr.analyze_rank(RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .is_full_rank()
        );
        assert_eq!(
            qr.analyze_rank(RankPolicy::DimensionScaledEpsilon)
                .unwrap()
                .rank(),
            2
        );
        let reconstructed = qr
            .to_q_matrix()
            .unwrap()
            .view()
            .mul_mat(&qr.to_triangular_matrix().unwrap())
            .unwrap();
        assert_permuted_factorization_f32(&a, &reconstructed, qr.permutation(), scale * 1e-5);
    }
}

#[test]
fn test_rank_deficient_qr_preserves_factors_and_reports_policy() {
    let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let rank = qr.analyze_rank(RankPolicy::DimensionScaledEpsilon).unwrap();
    assert_eq!(rank.rank(), 1);
    assert!(!rank.is_full_rank());

    let reconstructed = qr
        .to_q_matrix()
        .unwrap()
        .view()
        .mul_mat(&qr.to_triangular_matrix().unwrap())
        .unwrap();
    assert_permuted_factorization(&a, &reconstructed, qr.permutation(), 1e-12);
    assert_eq!(qr.analyze_rank(RankPolicy::Exact).unwrap().rank(), 2);
}

#[test]
fn column_pivoting_reveals_leading_zero_column_rank() {
    let matrix = Matrix::from_vec(3, 2, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let factor = HouseholderDecomposition::factor(
        &matrix,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    assert_eq!(factor.permutation(), &[1, 0]);
    let rank = factor
        .analyze_rank(RankPolicy::DimensionScaledEpsilon)
        .unwrap();
    assert_eq!(rank.rank(), 1);
    assert!(!rank.is_full_rank());
    let reconstructed = factor
        .to_q_matrix()
        .unwrap()
        .view()
        .mul_mat(&factor.to_triangular_matrix().unwrap())
        .unwrap();
    assert_permuted_factorization(&matrix, &reconstructed, factor.permutation(), 1.0e-14);
}

#[test]
fn checked_qr_solve_preserves_sentinel_on_overflow() {
    let matrix = Matrix::from_vec(2, 2, vec![f64::MIN_POSITIVE, 0.0, 0.0, 1.0]).unwrap();
    let factor = HouseholderDecomposition::factor(
        &matrix,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let rhs = Vector::from_vec(vec![f64::MAX, 1.0]);
    let mut output = Vector::from_vec(vec![31.0, 37.0]);
    let mut workspace = LinalgWorkspace::new();
    workspace
        .try_reserve(
            &HouseholderDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                    2,
                    2,
                    HouseholderOrientation::Qr,
                    &LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        (factor.solve_into(
            &rhs,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(output.as_slice(), &[31.0, 37.0]);
}

#[test]
fn test_local_column_scales_preserve_mixed_scale_rank() {
    let a = Matrix::from_vec(2, 2, vec![1e150, 0.0, 0.0, 1e-150]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    assert!(qr.analyze_rank(RankPolicy::Exact).unwrap().is_full_rank());
    assert_eq!(
        qr.analyze_rank(RankPolicy::DimensionScaledEpsilon)
            .unwrap()
            .rank(),
        1
    );
    let x = qr
        .solve_with_options(
            &Vector::from_vec(vec![1e150, 1e-150]),
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    assert!(f64::abs(x.view().get(0).unwrap() - 1.0) < 1e-12);
    assert!(f64::abs(x.view().get(1).unwrap() - 1.0) < 1e-12);
}

// ---- Larger system ----

#[test]
fn test_decompose_5x5_roundtrip() {
    let data = vec![
        2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        5.0, 1.0, 1.0, 0.0, 0.0, 1.0, 6.0,
    ];
    let a = Matrix::<f64>::from_vec(5, 5, data).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-10);

    let qt = q.view().adjoint().try_to_matrix().unwrap();
    let qtq = qt.view().mul_mat(&q).unwrap();
    let eye = Matrix::<f64>::identity(5).unwrap();
    assert_matrices_approx_eq(&qtq, &eye, 1e-12);
}

// ---- Matrix with zeros in first column ----

#[test]
fn test_decompose_zero_first_element() {
    let a =
        Matrix::<f64>::from_vec(3, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-10);
}

// ---- Diagonal matrix ----

#[test]
fn test_decompose_diagonal() {
    let a =
        Matrix::<f64>::from_vec(3, 3, vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0]).unwrap();
    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = qr.to_q_matrix().unwrap();
    let r = qr.to_triangular_matrix().unwrap();
    let qr_prod = q.view().mul_mat(&r).unwrap();
    assert_permuted_factorization(&a, &qr_prod, qr.permutation(), 1e-14);
}

#[test]
fn blocked_complex_qr_reconstructs_applies_and_solves_multiple_rhs() {
    let order = 69;
    let matrix = Matrix::from_vec(
        order,
        order,
        (0..order)
            .flat_map(|row| {
                (0..order).map(move |col| {
                    if row == col {
                        Complex::new(5.0 + row as f64 / order as f64, 0.2)
                    } else {
                        Complex::new(
                            ((row * 7 + col * 11 + 1) % 17) as f64 / 70.0 - 0.1,
                            ((row * 13 + col * 3 + 2) % 19) as f64 / 80.0 - 0.11,
                        )
                    }
                })
            })
            .collect(),
    )
    .unwrap();
    let factor = HouseholderDecomposition::factor(
        &matrix,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let q = factor.to_q_matrix().unwrap();
    let reconstructed = q
        .view()
        .mul_mat(&factor.to_triangular_matrix().unwrap())
        .unwrap();
    for row in 0..order {
        for (factor_column, &original_column) in factor.permutation().iter().enumerate() {
            let actual = reconstructed.view().get(row, factor_column).unwrap();
            let expected = matrix.view().get(row, original_column).unwrap();
            assert!((actual - expected).norm() < 1e-9);
        }
    }
    let orthogonality = q
        .view()
        .adjoint()
        .try_to_matrix()
        .unwrap()
        .view()
        .mul_mat(&q)
        .unwrap();
    for row in 0..order {
        for col in 0..order {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!(
                (orthogonality.view().get(row, col).unwrap() - Complex::new(expected, 0.0)).norm()
                    < 1e-9
            );
        }
    }

    let rhs = Matrix::from_vec(
        order,
        4,
        (0..order * 4)
            .map(|index| Complex::new((index % 13) as f64 / 7.0 - 0.4, (index % 7) as f64 / 9.0))
            .collect(),
    )
    .unwrap();
    let mut workspace = LinalgWorkspace::new();
    let mut transformed = Matrix::zeros(order, 4).unwrap();
    let mut restored = Matrix::zeros(order, 4).unwrap();
    factor
        .apply_q_into(
            &rhs,
            scioxide_linalg::StructuredOperation::Adjoint,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            transformed.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    factor
        .apply_q_into(
            &transformed,
            scioxide_linalg::StructuredOperation::Normal,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            restored.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    for (&actual, &expected) in restored.as_slice().iter().zip(rhs.as_slice()) {
        assert!((actual - expected).norm() < 1e-9);
    }

    let solution = factor
        .solve_matrix_with_options(
            &rhs,
            &(DirectSolveOptions::new(RankPolicy::Exact))
                .with_execution_options(LinalgExecutionOptions::new()),
        )
        .unwrap()
        .into_solution()
        .unwrap();
    let residual = matrix.view().mul_mat(&solution).unwrap();
    for (&actual, &expected) in residual.as_slice().iter().zip(rhs.as_slice()) {
        assert!((actual - expected).norm() < 1e-9);
    }
}

#[test]
fn reusable_qr_preflight_failures_preserve_outputs() {
    let valid = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap();
    let non_finite_factor =
        Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, f64::NAN]).unwrap();
    let mut factor = HouseholderDecomposition::factor(
        &valid,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let before_r = factor.to_triangular_matrix().unwrap();
    let mut workspace = LinalgWorkspace::new();
    assert!(matches!(
        (HouseholderDecomposition::factor_into(
            &non_finite_factor,
            HouseholderOrientation::Qr,
            HouseholderPivoting::Column,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(factor.to_triangular_matrix().unwrap(), before_r);

    let rank_deficient = HouseholderDecomposition::factor(
        &Matrix::from_vec(2, 2, vec![1.0, 0.0, 2.0, 0.0]).unwrap(),
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )
    .unwrap();
    let rhs = Vector::from_vec(vec![1.0, 2.0]);
    let mut output = Vector::from_vec(vec![7.0, 9.0]);
    workspace
        .try_reserve(
            &HouseholderDecomposition::<f64>::workspace_requirements(
                &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                    2,
                    2,
                    HouseholderOrientation::Qr,
                    &LinalgExecutionOptions::new(),
                ),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        rank_deficient
            .solve_into(
                &rhs,
                &(DirectSolveOptions::new(RankPolicy::Exact))
                    .with_execution_options(LinalgExecutionOptions::new()),
                output.view_mut(),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap()
            .termination(),
        DirectSolveTermination::RankDeficient
    );
    assert_eq!(output.as_slice(), &[7.0, 9.0]);

    let non_finite = Matrix::from_vec(2, 1, vec![1.0, f64::NAN]).unwrap();
    let mut applied = Matrix::from_vec(2, 1, vec![7.0, 9.0]).unwrap();
    assert!(matches!(
        (factor.apply_q_into(
            &non_finite,
            scioxide_linalg::StructuredOperation::Normal,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            applied.view_mut(),
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(applied.as_slice(), &[7.0, 9.0]);
}
