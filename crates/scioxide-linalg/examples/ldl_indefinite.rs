//! Solve an indefinite KKT system and inspect its inertia.
use scioxide_core::SciOxideError;
use scioxide_linalg::{
    DirectSolveOptions, Estimation, LdlDecomposition, Matrix, RankPolicy, Vector,
};
pub fn main() -> Result<(), SciOxideError> {
    let kkt = Matrix::from_vec(3, 3, vec![2.0_f64, 0.0, 1.0, 0.0, 4.0, 1.0, 1.0, 1.0, 0.0])?;
    let factor = LdlDecomposition::factor(&kkt, scioxide_linalg::HermitianInput::Exact)?;
    println!(
        "inertia: {:?}; permutation: {:?}",
        factor.inertia(scioxide_linalg::RankPolicy::Exact)?,
        factor.permutation()
    );
    let rhs = Vector::from_vec(vec![0.0, 0.0, 1.0]);
    let options = DirectSolveOptions::new(RankPolicy::Exact).with_residual(Estimation::Estimate);
    let result = factor.solve_with_options(&rhs, &options)?;
    println!(
        "solution: {:?}; diagnostics: {:?}",
        result.solution(),
        result.report()
    );
    Ok(())
}
