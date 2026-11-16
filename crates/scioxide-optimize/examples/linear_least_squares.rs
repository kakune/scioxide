//! Real nonnegative and box-constrained linear fits.
use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    BoundedLinearLeastSquares, LinearLeastSquaresProblem, LinearLeastSquaresResult, Nnls,
};
fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.])?;
    let b = [-1., 2., 1.];
    let problem = LinearLeastSquaresProblem::from_dense(&a, &b)?;
    let result = Nnls::<f64>::default().solve(&problem)?;
    println!("NNLS: {:?}, KKT {:?}", result.x(), result.kkt());
    let bounds = Bounds::new(vec![(Some(0.), Some(1.)); 2])?;
    let problem = problem.with_bounds(&bounds)?;
    let solver = BoundedLinearLeastSquares::<f64>::default();
    let mut workspace = solver.workspace(&problem)?;
    let mut result = LinearLeastSquaresResult::empty();
    result.try_reserve(3, 2)?;
    solver.solve_into(
        &problem,
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!("Bounded: {:?}, {:?}", result.x(), result.termination());
    Ok(())
}
