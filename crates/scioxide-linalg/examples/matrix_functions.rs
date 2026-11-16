//! Dense Padé exponential, real Schur functions and a noncommuting Frechet direction.
use scioxide_core::{Complex, ExecutionContext, SciOxideError};
use scioxide_linalg::{
    Matrix, MatrixFunction, MatrixFunctionOptions, MatrixFunctionPlan,
    MatrixFunctionWorkspaceRequest,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::from_vec(2, 2, vec![2., 0.5, 0., 2.])?;
    let options = MatrixFunctionOptions::default();
    let log = MatrixFunctionPlan::new(&a, MatrixFunction::Log, &options)?;
    let mut workspace = log.workspace(&MatrixFunctionWorkspaceRequest::Frechet)?;
    let mut output = Matrix::<f64>::zeros(2, 2)?;
    let report = log.evaluate_into(
        output.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("log(A) = {:?}; work = {}", output.as_slice(), report.work());
    let e = Matrix::<f64>::from_vec(2, 2, vec![0., 0., 1., 0.])?;
    MatrixFunctionPlan::new(&a, MatrixFunction::Exp, &options)?.frechet_into(
        &e,
        output.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("L_exp(A,E) = {:?}", output.as_slice());
    // Conjugate eigenvalues remain in real arithmetic for log and sqrt.
    let rotation = Matrix::from_vec(2, 2, vec![0.7_f64, -0.6, 0.6, 0.7])?;
    for (function, expected) in [
        (MatrixFunction::Exp, Complex::new(0.7_f64, 0.6).exp()),
        (MatrixFunction::Log, Complex::new(0.7_f64, 0.6).ln()),
        (MatrixFunction::Sqrt, Complex::new(0.7_f64, 0.6).sqrt()),
    ] {
        MatrixFunctionPlan::new(&rotation, function, &options)?.evaluate_into(
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        assert!((output.view().get(0, 0)? - expected.re).abs() < 1e-12);
        assert!((output.view().get(1, 0)? - expected.im).abs() < 1e-12);
    }
    let negative = Matrix::<f64>::from_vec(1, 1, vec![-1.])?;
    let mut root = Matrix::<Complex<f64>>::zeros(1, 1)?;
    MatrixFunctionPlan::new(&negative, MatrixFunction::Sqrt, &options)?.evaluate_into(
        root.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("principal sqrt(-1) = {:?}", root.as_slice());
    Ok(())
}
