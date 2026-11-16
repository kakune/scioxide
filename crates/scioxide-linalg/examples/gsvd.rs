//! Reconstruction factors and common-nullspace diagnostics for a rectangular pair.
use scioxide_core::SciOxideError;
use scioxide_linalg::{GsvdDecomposition, GsvdOptions, Matrix, RankPolicy};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(2, 3, vec![3.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])?;
    let b = Matrix::from_vec(2, 3, vec![4.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0])?;
    let factor = GsvdDecomposition::factor_with_options(
        a.view(),
        b.view(),
        &GsvdOptions::new(RankPolicy::DimensionScaledEpsilon),
    )?;
    println!("homogeneous values: {:?}", factor.generalized_values());
    println!("common nullity: {}", factor.common_null_space().cols());
    println!("input scale: {}", factor.input_scale());
    println!("diagnostics: {:?}", factor.report());
    // Reconstruct A / input_scale from the independently accessible factors.
    let r = factor.r_matrix();
    let mut right = Matrix::<f64>::zeros(r.rows(), a.cols())?;
    for i in 0..r.rows() {
        for j in 0..r.cols() {
            right
                .view_mut()
                .set(i, a.cols() - r.cols() + j, r.view().get(i, j)?)?;
        }
    }
    let normalized = factor
        .u_matrix()
        .view()
        .mul_mat(factor.d1_matrix())?
        .view()
        .mul_mat(&right)?
        .view()
        .mul_mat(&factor.q_matrix().view().adjoint().try_to_matrix()?)?;
    println!("normalized A: {:?}", normalized.as_slice());
    Ok(())
}
