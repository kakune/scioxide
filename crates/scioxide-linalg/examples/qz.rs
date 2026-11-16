//! Homogeneous pencil eigenvalues and selected left/right deflating subspaces.
use scioxide_core::SciOxideError;
use scioxide_linalg::{Estimation, GeneralizedSchurDecomposition, Matrix, SchurReorderOptions};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 3, vec![2.0_f64, 1., 2., 0., -1., 1., 0., 0., 3.])?;
    let b = Matrix::from_vec(3, 3, vec![1., 0., 1., 0., 1., 0., 0., 0., 0.])?;
    let pencil = GeneralizedSchurDecomposition::decompose(&a, &b)?;
    for value in pencil.eigenvalues() {
        println!(
            "alpha={:?}, beta={}, infinite={}",
            value.alpha(),
            value.beta(),
            value.is_infinite()
        );
    }
    let selection: Vec<_> = pencil
        .eigenvalues()
        .iter()
        .map(|v| v.is_finite() && v.alpha().re < 0.)
        .collect();
    let (pencil, report) = pencil.reorder_with_options(
        &selection,
        &SchurReorderOptions::new().with_condition_estimates(Estimation::Estimate),
    )?;
    let left = pencil.left_deflating_subspace(report.selected_dimension())?;
    let right = pencil.right_deflating_subspace(report.selected_dimension())?;
    println!(
        "Selected {} columns: left {}x{}, right {}x{}",
        report.selected_dimension(),
        left.rows(),
        left.cols(),
        right.rows(),
        right.cols()
    );
    println!("Separation estimates: {:?}", report.separation());
    Ok(())
}
