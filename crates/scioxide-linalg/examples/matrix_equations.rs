//! Reuse one Schur preparation for multiple continuous Lyapunov right-hand sides.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::{LyapunovKind, LyapunovPlan, Matrix, MatrixEquationOptions};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::from_vec(2, 2, vec![-1., 0.25, 0., -2.])?;
    let plan = LyapunovPlan::new(&a, LyapunovKind::Continuous, MatrixEquationOptions::new())?;
    let mut workspace = plan.workspace(())?;
    let mut output = Matrix::<f64>::zeros(2, 2)?;
    for weight in [1., 2.] {
        let q = Matrix::<f64>::identity(2)?.view().try_scale(-weight)?;
        let report = plan.solve_into(
            &q,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "X = {:?}; {:?}; residual = {}",
            output.as_slice(),
            report.termination(),
            report.residual()
        );
    }
    Ok(())
}
