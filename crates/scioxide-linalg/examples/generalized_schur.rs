//! Generalized definite eigenpairs and a selected real invariant subspace.
use scioxide_core::SciOxideError;
use scioxide_linalg::{
    Estimation, GeneralizedHermitianEigenDecomposition, Matrix, SchurDecomposition,
    SchurReorderOptions,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::from_vec(2, 2, vec![3., 1., 1., 3.])?;
    let b = Matrix::from_vec(2, 2, vec![2., 0., 0., 1.])?;
    let eigen = GeneralizedHermitianEigenDecomposition::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )?;
    println!("Generalized eigenvalues: {:?}", eigen.eigenvalues());
    println!("Original-pair residuals: {:?}", eigen.residuals());
    let general = Matrix::from_vec(3, 3, vec![1., 2., 3., 0., 4., 6., 0., 0., 9.])?;
    let schur = SchurDecomposition::<f64>::decompose(&general)?;
    let (schur, report) = schur.reorder_with_options(
        &[false, true, false],
        &SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate),
    )?;
    let basis = schur.invariant_subspace(report.selected_dimension())?;
    println!("Selected basis shape: {} x {}", basis.rows(), basis.cols());
    println!(
        "Conditioning: {:?}; separation: {:?}",
        report.cluster_condition(),
        report.separation()
    );
    Ok(())
}
