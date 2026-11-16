//! Rectangular polar factors and a complete partitioned CS decomposition.
use scioxide_core::SciOxideError;
use scioxide_linalg::{CsdDecomposition, Matrix, PolarDecomposition, PolarOptions, RankPolicy};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 2, vec![3.0_f64, 0.0, 0.0, 4.0, 0.0, 0.0])?;
    let polar = PolarDecomposition::factor_with_options(
        a.view(),
        scioxide_linalg::PolarSide::Right,
        &PolarOptions::new(RankPolicy::DimensionScaledEpsilon),
    )?;
    println!(
        "A = U H: {:?}",
        polar.u_matrix().view().mul_mat(polar.h_matrix())?
    );
    let x = Matrix::from_vec(2, 2, vec![0.6, -0.8, 0.8, 0.6])?;
    let cs = CsdDecomposition::factor(x.view(), 1, 1)?;
    println!(
        "CS angles: {:?}; diagnostics: {:?}",
        cs.angles(),
        cs.report()
    );
    Ok(())
}
