//! Rank-selected bases, angles and right-acting alignment.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::{
    LinalgExecutionOptions, Matrix, RankPolicy, SubspaceBasisWorkspaceRequest, null_space,
    null_space_into, orth, orthogonal_procrustes, subspace_angles,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0])?;
    let b = Matrix::from_vec(3, 2, vec![0.0_f64, 1.0, 2.0, 0.0, 0.0, 0.0])?;
    let policy = RankPolicy::DimensionScaledEpsilon;
    let range = orth(&a, policy)?;
    let null = null_space(&a, policy)?;
    let angles = subspace_angles(&a, &b, policy)?;
    let alignment = orthogonal_procrustes(&a, &b, policy)?;
    println!(
        "rank={}, nullity={}",
        range.report().rank().rank(),
        null.basis().cols()
    );
    println!("angles={:?}", angles.angles());
    println!(
        "R={:?}, residual={}",
        alignment.transformation().as_slice(),
        alignment.report().residual_norm()
    );
    // Repeated bases reuse one requirement-sized workspace and output.
    let options = LinalgExecutionOptions::new();
    let request = SubspaceBasisWorkspaceRequest::null_space(2, 3, &options);
    let mut workspace = scioxide_linalg::LinalgWorkspace::<f64>::workspace(&request)?;
    let mut kernel = Matrix::empty();
    for scale in [1.0, 2.0] {
        let wide = Matrix::from_vec(2, 3, vec![scale, 0.0, 0.0, 0.0, scale, 0.0])?;
        null_space_into(
            &wide,
            policy,
            &options,
            &mut kernel,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!("kernel of scale {scale}: {:?}", kernel.as_slice());
    }
    Ok(())
}
