//! Plain forms equal their configured forms with default options, bit for bit.
#![allow(clippy::unwrap_used)]

use core::fmt::Debug;
use scioxide_core::{Complex, DiagonalKind, SciOxideError, SciOxideErrorKind, Triangle};
use scioxide_linalg::*;

/// Asserts that `plain` and `configured` are identical.
///
/// `Debug` prints every float as the shortest decimal that rounds back to it, so equal output
/// means equal bits for every non-NaN value, including the sign of zero.
fn identical<T: Debug>(plain: &T, configured: &T) {
    assert_eq!(format!("{plain:?}"), format!("{configured:?}"));
}

/// Asserts that a plain solve returns the accepted solution of the configured solve.
fn same_solution<T: Debug>(
    plain: Result<T, SciOxideError>,
    configured: Result<DirectSolveResult<T, f64>, SciOxideError>,
) -> Result<(), SciOxideError> {
    let configured = configured?;
    assert_eq!(
        configured.report().termination(),
        DirectSolveTermination::Accepted
    );
    identical(&plain?, &configured.into_accepted()?);
    Ok(())
}

fn general() -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_rows(&[[4.0, 1.0, -2.0], [1.0, 3.0, 0.5], [-1.0, 2.0, 5.0]])
}

fn positive_definite() -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_rows(&[[4.0, 1.0, 0.5], [1.0, 3.0, -1.0], [0.5, -1.0, 5.0]])
}

fn indefinite() -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_rows(&[[1.0, 2.0, 0.0], [2.0, 1.0, 3.0], [0.0, 3.0, -2.0]])
}

fn tall() -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_rows(&[
        [4.0, 1.0, -2.0],
        [1.0, 3.0, 0.5],
        [-1.0, 2.0, 5.0],
        [2.0, -1.0, 1.0],
    ])
}

const OPERATIONS: [StructuredOperation; 3] = [
    StructuredOperation::Normal,
    StructuredOperation::Transpose,
    StructuredOperation::Adjoint,
];

#[test]
fn dense_factor_plain_forms_match_default_options() -> Result<(), SciOxideError> {
    let execution = LinalgExecutionOptions::default();
    let (a, spd, indefinite, tall) = (general()?, positive_definite()?, indefinite()?, tall()?);
    identical(
        &LuDecomposition::factor(&a)?,
        &LuDecomposition::factor_with_options(&a, &execution)?,
    );
    for pivoting in [
        CholeskyPivoting::None,
        CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
    ] {
        identical(
            &CholeskyDecomposition::factor(&spd, HermitianInput::Exact, pivoting)?,
            &CholeskyDecomposition::factor_with_options(
                &spd,
                HermitianInput::Exact,
                pivoting,
                &execution,
            )?,
        );
    }
    let lower = HermitianInput::Triangle(Triangle::Lower);
    identical(
        &LdlDecomposition::factor(&indefinite, lower)?,
        &LdlDecomposition::factor_with_options(&indefinite, lower, &execution)?,
    );
    let input = Matrix::from_rows(&[[1.0, 0.25], [-2.0, 0.5], [0.3, 1.5], [0.7, -1.0]])?;
    for (orientation, matrix) in [
        (HouseholderOrientation::Qr, tall.view()),
        (HouseholderOrientation::Lq, tall.view().transpose()),
    ] {
        for pivoting in [HouseholderPivoting::Column, HouseholderPivoting::None] {
            let factor = HouseholderDecomposition::factor(matrix, orientation, pivoting)?;
            identical(
                &factor,
                &HouseholderDecomposition::factor_with_options(
                    matrix,
                    orientation,
                    pivoting,
                    &execution,
                )?,
            );
            identical(
                &factor.to_q_matrix()?,
                &factor.to_q_matrix_with_options(&execution)?,
            );
            for operation in OPERATIONS {
                identical(
                    &factor.apply_q(&input, operation)?,
                    &factor.apply_q_with_options(&input, operation, &execution)?,
                );
            }
            identical(
                &factor.reconstruct()?,
                &factor.reconstruct_with_options(&execution)?,
            );
        }
    }
    identical(
        &RqDecomposition::factor(&a)?,
        &RqDecomposition::factor_with_options(&a, &execution)?,
    );
    let update = QrUpdateOptions::default();
    let qr = QrUpdateDecomposition::factor(&tall)?;
    identical(
        &qr,
        &QrUpdateDecomposition::factor_with_options(&tall, &update)?,
    );
    let householder = HouseholderDecomposition::factor(
        &tall,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )?;
    identical(
        &QrUpdateDecomposition::from_householder(&householder)?,
        &QrUpdateDecomposition::from_householder_with_options(&householder, &update)?,
    );
    let row = Matrix::from_rows(&[[0.5, -1.5, 2.5]])?;
    for edit in [
        QrModification::DeleteRows { index: 1, count: 1 },
        QrModification::InsertRows {
            index: 2,
            values: row.view(),
        },
    ] {
        identical(&qr.modify(edit)?, &qr.modify_with_options(edit, &update)?);
    }
    Ok(())
}

#[test]
fn dense_solve_plain_forms_return_the_accepted_solution() -> Result<(), SciOxideError> {
    let accept = DirectSolveOptions::default();
    assert_eq!(accept, DirectSolveOptions::new(RankPolicy::Exact));
    let (a, spd, indefinite, tall) = (general()?, positive_definite()?, indefinite()?, tall()?);
    let b = [1.0, -2.0, 0.3];
    let rhs = Matrix::from_rows(&[[1.0, 0.1], [-2.0, 0.7], [0.3, -1.1]])?;

    let lu = LuDecomposition::factor(&a)?;
    same_solution(lu.solve(&b[..]), lu.solve_with_options(&b[..], &accept))?;
    same_solution(
        lu.solve_matrix(&rhs),
        lu.solve_matrix_with_options(&rhs, &accept),
    )?;

    let cholesky =
        CholeskyDecomposition::factor(&spd, HermitianInput::Exact, CholeskyPivoting::None)?;
    same_solution(
        cholesky.solve(&b[..]),
        cholesky.solve_with_options(&b[..], &accept),
    )?;
    same_solution(
        cholesky.solve_matrix(&rhs),
        cholesky.solve_matrix_with_options(&rhs, &accept),
    )?;

    let ldl = LdlDecomposition::factor(&indefinite, HermitianInput::Exact)?;
    same_solution(ldl.solve(&b[..]), ldl.solve_with_options(&b[..], &accept))?;
    same_solution(
        ldl.solve_matrix(&rhs),
        ldl.solve_matrix_with_options(&rhs, &accept),
    )?;

    let qr = HouseholderDecomposition::factor(
        &tall,
        HouseholderOrientation::Qr,
        HouseholderPivoting::Column,
    )?;
    let long = [1.0, -2.0, 0.3, 0.7];
    let long_rhs = Matrix::from_rows(&[[1.0, 0.1], [-2.0, 0.7], [0.3, -1.1], [0.7, 2.0]])?;
    same_solution(
        qr.solve(&long[..]),
        qr.solve_with_options(&long[..], &accept),
    )?;
    same_solution(
        qr.solve_matrix(&long_rhs),
        qr.solve_matrix_with_options(&long_rhs, &accept),
    )?;
    for damping in [0.0, 0.5] {
        same_solution(
            qr.solve_damped(&long[..], damping),
            qr.solve_damped_with_options(&long[..], damping, &accept),
        )?;
    }
    let lq = HouseholderDecomposition::factor(
        tall.view().transpose(),
        HouseholderOrientation::Lq,
        HouseholderPivoting::Column,
    )?;
    // Both forms reject an LQ factorization with the same error.
    identical(
        &lq.solve(&b[..]).unwrap_err(),
        &lq.solve_with_options(&b[..], &accept).unwrap_err(),
    );
    identical(
        &lq.solve_matrix(&rhs).unwrap_err(),
        &lq.solve_matrix_with_options(&rhs, &accept).unwrap_err(),
    );
    assert_eq!(
        lq.solve(&b[..]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    Ok(())
}

#[test]
fn rank_deficient_plain_solve_fails_where_the_configured_solve_reports() -> Result<(), SciOxideError>
{
    let singular = Matrix::from_rows(&[[4.0, 0.0], [0.0, 0.0]])?;
    let factor = CholeskyDecomposition::factor(
        &singular,
        HermitianInput::Exact,
        CholeskyPivoting::Diagonal(RankPolicy::DimensionScaledEpsilon),
    )?;
    let accept = DirectSolveOptions::default();
    let b = [8.0, 0.0];
    assert_eq!(
        factor.solve(&b[..]).unwrap_err().kind(),
        SciOxideErrorKind::Singular
    );
    let result = factor.solve_with_options(&b[..], &accept)?;
    assert_eq!(
        result.report().termination(),
        DirectSolveTermination::RankDeficient
    );
    assert!(result.solution().is_none());
    assert_eq!(
        result.into_accepted().unwrap_err().kind(),
        SciOxideErrorKind::Singular
    );

    let rhs = Matrix::from_rows(&[[8.0, 4.0], [0.0, 0.0]])?;
    assert_eq!(
        factor.solve_matrix(&rhs).unwrap_err().kind(),
        SciOxideErrorKind::Singular
    );
    assert_eq!(
        factor
            .solve_matrix_with_options(&rhs, &accept)?
            .report()
            .termination(),
        DirectSolveTermination::RankDeficient
    );
    Ok(())
}

#[test]
fn decomposition_plain_forms_match_default_options() -> Result<(), SciOxideError> {
    let execution = LinalgExecutionOptions::default();
    let (a, spd, tall) = (general()?, positive_definite()?, tall()?);
    identical(
        &SvdDecomposition::compute(&tall)?,
        &SvdDecomposition::compute_with_options(&tall, &execution)?,
    );
    identical(
        &SvdValues::compute(&tall)?,
        &SvdValues::compute_with_options(&tall, &execution)?,
    );
    identical(
        &ThinSvdDecomposition::compute(&tall)?,
        &ThinSvdDecomposition::compute_with_options(&tall, &execution)?,
    );
    identical(
        &GsvdDecomposition::factor(&tall, &spd)?,
        &GsvdDecomposition::factor_with_options(&tall, &spd, &GsvdOptions::default())?,
    );
    let orthogonal = HouseholderDecomposition::factor(
        &tall,
        HouseholderOrientation::Qr,
        HouseholderPivoting::None,
    )?
    .to_q_matrix()?;
    identical(
        &CsdDecomposition::factor(&orthogonal, 2, 1)?,
        &CsdDecomposition::factor_with_options(&orthogonal, 2, 1, &execution)?,
    );
    for (side, matrix) in [(PolarSide::Right, &tall), (PolarSide::Left, &a)] {
        identical(
            &PolarDecomposition::factor(matrix, side)?,
            &PolarDecomposition::factor_with_options(matrix, side, &PolarOptions::default())?,
        );
    }
    Ok(())
}

#[test]
fn eigen_plain_forms_match_default_options() -> Result<(), SciOxideError> {
    let hermitian = HermitianEigenOptions::default();
    let (a, spd, indefinite) = (general()?, positive_definite()?, indefinite()?);
    let exact = HermitianInput::Exact;
    identical(
        &HermitianEigenDecomposition::decompose(&spd, exact)?,
        &HermitianEigenDecomposition::decompose_with_options(&spd, exact, &hermitian)?,
    );
    identical(
        &HermitianEigenvaluesResult::compute(&spd, exact)?,
        &HermitianEigenvaluesResult::compute_with_options(&spd, exact, &hermitian)?,
    );
    let (diagonal, off_diagonal) = ([4.0, 3.0, 5.0], [1.0, -1.0]);
    identical(
        &HermitianEigenDecomposition::tridiagonal(&diagonal, &off_diagonal)?,
        &HermitianEigenDecomposition::tridiagonal_with_options(
            &diagonal,
            &off_diagonal,
            &hermitian,
        )?,
    );
    identical(
        &HermitianEigenvaluesResult::tridiagonal(&diagonal, &off_diagonal)?,
        &HermitianEigenvaluesResult::tridiagonal_with_options(
            &diagonal,
            &off_diagonal,
            &hermitian,
        )?,
    );
    identical(
        &HermitianEigenFirstComponents::tridiagonal(&diagonal, &off_diagonal)?,
        &HermitianEigenFirstComponents::tridiagonal_with_options(
            &diagonal,
            &off_diagonal,
            &hermitian,
        )?,
    );

    let general_options = GeneralEigenOptions::default();
    identical(
        &GeneralEigenDecomposition::decompose(&a)?,
        &GeneralEigenDecomposition::decompose_with_options(&a, &general_options)?,
    );
    identical(
        &GeneralEigenvaluesResult::compute(&a)?,
        &GeneralEigenvaluesResult::compute_with_options(&a, &general_options)?,
    );

    identical(
        &GeneralizedHermitianEigenDecomposition::decompose(&indefinite, &spd, exact, exact)?,
        &GeneralizedHermitianEigenDecomposition::decompose_with_options(
            &indefinite,
            &spd,
            exact,
            exact,
            &hermitian,
        )?,
    );
    identical(
        &GeneralizedHermitianEigenvaluesResult::compute(&indefinite, &spd, exact, exact)?,
        &GeneralizedHermitianEigenvaluesResult::compute_with_options(
            &indefinite,
            &spd,
            exact,
            exact,
            &hermitian,
        )?,
    );

    let schur_options = SchurOptions::default();
    let reorder = SchurReorderOptions::default();
    let schur = SchurDecomposition::decompose(&a)?;
    identical(
        &schur,
        &SchurDecomposition::decompose_with_options(&a, &schur_options)?,
    );
    // Selecting the trailing block by its real part keeps a complex conjugate pair together,
    // and the reorder moves it to the front.
    let trailing = schur.eigenvalues().last().unwrap().re;
    let selection: Vec<bool> = schur
        .eigenvalues()
        .iter()
        .map(|value| value.re == trailing)
        .collect();
    assert!(!selection[0]);
    identical(
        &schur.reorder(&selection)?,
        &schur.reorder_with_options(&selection, &reorder)?,
    );

    let qz = GeneralizedSchurDecomposition::decompose(&a, &spd)?;
    identical(
        &qz,
        &GeneralizedSchurDecomposition::decompose_with_options(&a, &spd, &schur_options)?,
    );
    let ratios = qz
        .eigenvalues()
        .iter()
        .map(|value| value.ratio().map(|ratio| ratio.re))
        .collect::<Result<Vec<_>, _>>()?;
    let trailing = *ratios.last().unwrap();
    let selection: Vec<bool> = ratios.iter().map(|&ratio| ratio == trailing).collect();
    assert!(!selection[0]);
    identical(
        &qz.reorder(&selection)?,
        &qz.reorder_with_options(&selection, &reorder)?,
    );
    Ok(())
}

#[test]
fn structured_plain_forms_match_default_options() -> Result<(), SciOxideError> {
    let execution = LinalgExecutionOptions::default();
    let accept = DirectSolveOptions::default();
    let b = [1.0, -2.0, 0.3];
    let rhs = Matrix::from_rows(&[[1.0, 0.1], [-2.0, 0.7], [0.3, -1.1]])?;

    // A = [4 −1 0; 2 5 1; 0 −3 6] with one subdiagonal and one superdiagonal.
    let banded = BandedMatrix::new(
        3,
        3,
        1,
        1,
        3,
        vec![0.0, 4.0, 2.0, -1.0, 5.0, -3.0, 1.0, 6.0, 0.0],
    )?;
    let banded_lu = BandedLuDecomposition::factor(&banded)?;
    identical(
        &banded_lu,
        &BandedLuDecomposition::factor_with_options(&banded, &execution)?,
    );
    // A = [4 1 0; 1 3 −1; 0 −1 5] in lower band storage.
    let band =
        HermitianBandMatrix::new(3, 1, Triangle::Lower, vec![4.0, 1.0, 3.0, -1.0, 5.0, 0.0])?;
    let banded_cholesky = BandedCholeskyDecomposition::factor(&band)?;
    identical(
        &banded_cholesky,
        &BandedCholeskyDecomposition::factor_with_options(&band, &execution)?,
    );
    let tridiagonal = TridiagonalMatrixView::new(&[2.0, -1.0], &[4.0, 5.0, 6.0], &[-1.0, 1.0])?;
    let tridiagonal_lu = TridiagonalLuDecomposition::factor(tridiagonal)?;
    identical(
        &tridiagonal_lu,
        &TridiagonalLuDecomposition::factor_with_options(tridiagonal, &execution)?,
    );
    let a = general()?;
    let triangular = TriangularPlan::from_dense(&a, Triangle::Lower, DiagonalKind::Explicit)?;
    let cyclic = CyclicTridiagonalPlan::new(tridiagonal, 0.5, -0.25, &execution)?;

    for operation in OPERATIONS {
        same_solution(
            banded_lu.solve(operation, &b[..]),
            banded_lu.solve_with_options(operation, &b[..], &accept),
        )?;
        same_solution(
            banded_lu.solve_matrix(operation, &rhs),
            banded_lu.solve_matrix_with_options(operation, &rhs, &accept),
        )?;
        same_solution(
            banded_cholesky.solve(operation, &b[..]),
            banded_cholesky.solve_with_options(operation, &b[..], &accept),
        )?;
        same_solution(
            banded_cholesky.solve_matrix(operation, &rhs),
            banded_cholesky.solve_matrix_with_options(operation, &rhs, &accept),
        )?;
        same_solution(
            tridiagonal_lu.solve(operation, &b[..]),
            tridiagonal_lu.solve_with_options(operation, &b[..], &accept),
        )?;
        same_solution(
            tridiagonal_lu.solve_matrix(operation, &rhs),
            tridiagonal_lu.solve_matrix_with_options(operation, &rhs, &accept),
        )?;
        same_solution(
            triangular.solve(operation, &b[..]),
            triangular.solve_with_options(operation, &b[..], &accept),
        )?;
        same_solution(
            triangular.solve_matrix(operation, &rhs),
            triangular.solve_matrix_with_options(operation, &rhs, &accept),
        )?;
        same_solution(
            cyclic.solve(operation, &b[..]),
            cyclic.solve_with_options(operation, &b[..], &accept),
        )?;
        same_solution(
            cyclic.solve_matrix(operation, &rhs),
            cyclic.solve_matrix_with_options(operation, &rhs, &accept),
        )?;
    }
    Ok(())
}

#[test]
fn product_plain_forms_match_default_options() -> Result<(), SciOxideError> {
    let execution = LinalgExecutionOptions::default();
    let (a, spd) = (general()?, positive_definite()?);
    let (a, spd) = (a.view(), spd.view());
    let x = [1.0, -2.0, 0.3];
    let rectangular = Matrix::from_rows(&[[1.0, 0.1], [-2.0, 0.7], [0.3, -1.1]])?;
    let square = Matrix::from_rows(&[[1.0, 0.1, -0.4], [-2.0, 0.7, 0.9], [0.3, -1.1, 1.3]])?;

    identical(
        &a.mul_mat(&rectangular)?,
        &a.mul_mat_with_options(&rectangular, &execution)?,
    );
    identical(
        &a.mul_vec(&x[..])?,
        &a.mul_vec_with_options(&x[..], &execution)?,
    );
    // A Hermitian matrix with complex off-diagonal entries.
    let hermitian = Matrix::from_rows(&[
        [Complex::new(2.0, 0.0), Complex::new(1.0, 0.5)],
        [Complex::new(1.0, -0.5), Complex::new(3.0, 0.0)],
    ])?;
    let hermitian = hermitian.view();
    let complex_general = Matrix::from_rows(&[
        [Complex::new(1.0, -0.3), Complex::new(0.2, 0.7)],
        [Complex::new(-1.5, 0.4), Complex::new(0.6, -0.1)],
    ])?;
    let complex_x = [Complex::new(0.3, -1.0), Complex::new(-0.7, 0.2)];
    for triangle in [Triangle::Lower, Triangle::Upper] {
        for side in [MatrixSide::Left, MatrixSide::Right] {
            identical(
                &spd.symmetric_mul_mat(&square, side, triangle)?,
                &spd.symmetric_mul_mat_with_options(&square, side, triangle, &execution)?,
            );
            identical(
                &hermitian.hermitian_mul_mat(&complex_general, side, triangle)?,
                &hermitian.hermitian_mul_mat_with_options(
                    &complex_general,
                    side,
                    triangle,
                    &execution,
                )?,
            );
            for operation in OPERATIONS {
                identical(
                    &a.triangular_mul_mat(
                        &square,
                        side,
                        triangle,
                        DiagonalKind::Explicit,
                        operation,
                        0.75,
                    )?,
                    &a.triangular_mul_mat_with_options(
                        &square,
                        side,
                        triangle,
                        DiagonalKind::Explicit,
                        operation,
                        0.75,
                        &execution,
                    )?,
                );
                identical(
                    &a.triangular_solve_mat(
                        &square,
                        side,
                        triangle,
                        DiagonalKind::Explicit,
                        operation,
                        0.75,
                    )?,
                    &a.triangular_solve_mat_with_options(
                        &square,
                        side,
                        triangle,
                        DiagonalKind::Explicit,
                        operation,
                        0.75,
                        &execution,
                    )?,
                );
            }
        }
        identical(
            &spd.symmetric_mul_vec(triangle, &x[..])?,
            &spd.symmetric_mul_vec_with_options(triangle, &x[..], &execution)?,
        );
        identical(
            &hermitian.hermitian_mul_vec(triangle, &complex_x[..])?,
            &hermitian.hermitian_mul_vec_with_options(triangle, &complex_x[..], &execution)?,
        );
    }

    let banded = BandedMatrix::new(
        3,
        3,
        1,
        1,
        3,
        vec![0.0, 4.0, 2.0, -1.0, 5.0, -3.0, 1.0, 6.0, 0.0],
    )?;
    // Upper packed storage of [4 1 0.5; 1 3 −1; 0.5 −1 5].
    let packed_data = [4.0, 1.0, 3.0, 0.5, -1.0, 5.0];
    for kind in [PackedKind::Hermitian, PackedKind::Triangular] {
        let packed = PackedMatrixView::new(3, Triangle::Upper, kind, &packed_data)?;
        for operation in OPERATIONS {
            identical(
                &packed.mul_vec(operation, &x[..])?,
                &packed.mul_vec_with_options(operation, &x[..], &execution)?,
            );
        }
    }
    for operation in OPERATIONS {
        identical(
            &banded.view().mul_vec(operation, &x[..])?,
            &banded
                .view()
                .mul_vec_with_options(operation, &x[..], &execution)?,
        );
    }
    let band =
        HermitianBandMatrix::new(3, 1, Triangle::Lower, vec![4.0, 1.0, 3.0, -1.0, 5.0, 0.0])?;
    identical(
        &band.view().mul_vec(&x[..])?,
        &band.view().mul_vec_with_options(&x[..], &execution)?,
    );

    let (column, row) = ([1.0, 0.5, -0.25], [1.0, 2.0, -3.0, 0.125]);
    let toeplitz = StructuredMatrixView::toeplitz(&column, &row)?;
    identical(
        &toeplitz.materialize()?,
        &toeplitz.materialize_with_options(&execution)?,
    );
    identical(
        &StructuredMatrix::from_view(toeplitz)?,
        &StructuredMatrix::from_view_with_options(toeplitz, &execution)?,
    );
    let wide = rectangular.view().transpose();
    identical(
        &khatri_rao(a, wide)?,
        &khatri_rao_with_options(a, wide, &execution)?,
    );
    let operator = a.try_finite()?;
    for kind in [MatrixNorm::One, MatrixNorm::Infinity, MatrixNorm::Frobenius] {
        assert_eq!(
            operator_norm(&operator, kind)?.to_bits(),
            operator_norm_with_options(&operator, kind, &execution)?.to_bits()
        );
    }
    Ok(())
}
