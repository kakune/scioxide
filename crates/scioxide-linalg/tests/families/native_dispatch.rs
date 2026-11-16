//! Owned results must match the same call with native sequential capabilities.

use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, GeneralEigenOptions, GeneralEigenvaluesBuffer,
    GeneralEigenvaluesResult, HermitianEigenDecomposition, HermitianEigenOptions,
    HermitianEigenvaluesBuffer, HermitianEigenvaluesResult, HermitianInput, LinalgExecutionOptions,
    LinalgWorkspace, LuDecomposition, Matrix, ThinSvdDecomposition,
};

const ORDER: usize = 256;

fn matrix(symmetric: bool) -> Result<Matrix<f64>, SciOxideError> {
    Matrix::from_vec(
        ORDER,
        ORDER,
        (0..ORDER * ORDER)
            .map(|index| {
                let (row, col) = (index / ORDER, index % ORDER);
                if row == col {
                    2.0 + row as f64 / ORDER as f64
                } else {
                    let (a, b) = if symmetric {
                        (row.min(col), row.max(col))
                    } else {
                        (row, col)
                    };
                    (((a * 17 + b * 31) % 103) as f64 / 103.0 - 0.5) / ORDER as f64
                }
            })
            .collect(),
    )
}

fn assert_bits(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(left.to_bits(), right.to_bits(), "component {index}");
    }
}

fn options() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential)
}

#[test]
fn owned_lu_matches_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(false)?;
    let owned = LuDecomposition::factor_with_options(&input, &options())?;
    let mut borrowed = LuDecomposition::empty();
    LuDecomposition::factor_into(
        &input,
        &options(),
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_bits(
        owned.to_l_matrix()?.as_slice(),
        borrowed.to_l_matrix()?.as_slice(),
    );
    assert_bits(
        owned.to_u_matrix()?.as_slice(),
        borrowed.to_u_matrix()?.as_slice(),
    );
    assert_eq!(owned.permutation(), borrowed.permutation());
    assert_eq!(owned.equilibration(), borrowed.equilibration());
    Ok(())
}

#[test]
fn owned_cholesky_matches_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(true)?;
    let owned = CholeskyDecomposition::factor_with_options(
        &input,
        HermitianInput::Exact,
        CholeskyPivoting::None,
        &options(),
    )?;
    let mut borrowed = CholeskyDecomposition::empty();
    CholeskyDecomposition::factor_into(
        &input,
        HermitianInput::Exact,
        CholeskyPivoting::None,
        &options(),
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_bits(owned.l_matrix().as_slice(), borrowed.l_matrix().as_slice());
    assert_eq!(owned.permutation(), borrowed.permutation());
    assert_eq!(owned.projection_report(), borrowed.projection_report());
    Ok(())
}

#[test]
fn owned_thin_svd_matches_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(false)?;
    let owned = ThinSvdDecomposition::compute_with_options(&input, &options())?;
    let mut borrowed = ThinSvdDecomposition::empty();
    ThinSvdDecomposition::compute_into(
        &input,
        &options(),
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_bits(owned.singular_values(), borrowed.singular_values());
    assert_bits(owned.u_matrix().as_slice(), borrowed.u_matrix().as_slice());
    assert_bits(
        owned.vt_matrix().as_slice(),
        borrowed.vt_matrix().as_slice(),
    );
    assert_eq!(
        (owned.rows(), owned.cols()),
        (borrowed.rows(), borrowed.cols())
    );
    Ok(())
}

#[test]
fn owned_hermitian_values_match_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(true)?;
    let options = HermitianEigenOptions::default().with_execution_options(options());
    let owned =
        HermitianEigenvaluesResult::compute_with_options(&input, HermitianInput::Exact, &options)?;
    let mut borrowed = HermitianEigenvaluesBuffer::empty();
    let report = HermitianEigenvaluesBuffer::compute_into(
        &input,
        HermitianInput::Exact,
        &options,
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_bits(owned.eigenvalues(), borrowed.eigenvalues());
    assert_eq!(owned.projection_report(), borrowed.projection_report());
    assert_eq!(owned.report().counts(), report.counts());
    Ok(())
}

#[test]
fn owned_hermitian_vectors_match_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(true)?;
    let options = HermitianEigenOptions::default().with_execution_options(options());
    let owned = HermitianEigenDecomposition::decompose_with_options(
        &input,
        HermitianInput::Exact,
        &options,
    )?;
    let mut borrowed = HermitianEigenDecomposition::empty();
    HermitianEigenDecomposition::decompose_into(
        &input,
        HermitianInput::Exact,
        &options,
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_bits(owned.eigenvalues(), borrowed.eigenvalues());
    assert_bits(
        owned.eigenvectors().as_slice(),
        borrowed.eigenvectors().as_slice(),
    );
    assert_eq!(owned.projection_report(), borrowed.projection_report());
    assert_eq!(
        owned.report().map(|r| r.counts()),
        borrowed.report().map(|r| r.counts())
    );
    Ok(())
}

#[test]
fn owned_general_values_match_native_borrowed_at_256() -> Result<(), SciOxideError> {
    let input = matrix(false)?;
    let options = GeneralEigenOptions::new().with_execution_options(options());
    let owned = GeneralEigenvaluesResult::compute_with_options(&input, &options)?;
    let mut borrowed = GeneralEigenvaluesBuffer::empty();
    let report = GeneralEigenvaluesBuffer::compute_into(
        &input,
        &options,
        &mut borrowed,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential_native(),
    )?;
    assert_eq!(owned.eigenvalues().len(), borrowed.eigenvalues().len());
    for (left, right) in owned.eigenvalues().iter().zip(borrowed.eigenvalues()) {
        assert_bits(&[left.re, left.im], &[right.re, right.im]);
    }
    assert_eq!(owned.report().counts(), report.counts());
    Ok(())
}
