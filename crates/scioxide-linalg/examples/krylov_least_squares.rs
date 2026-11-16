//! Solve an inconsistent rectangular problem and an indefinite system.
use scioxide_core::{Budget, GradientTolerance, ResidualTolerance, SciOxideError};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{IterativeWorkspaceRequest, LsmrSolver, Matrix, MinresSolver, Vector};
pub fn main() -> Result<(), SciOxideError> {
    let tolerance = ResidualTolerance::new(1e-12, 0.)?;
    let budget = Budget::try_from(100)?;
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 0., 0.])?;
    let b = Vector::from_vec(vec![2., -1., 3.]);
    let solver = LsmrSolver::new(tolerance, budget)
        .with_normal_tolerance(GradientTolerance::new(1e-12, 0.)?);
    let mut workspace = solver.workspace(&IterativeWorkspaceRequest::adjoint(
        &a.view().try_finite()?,
    )?)?;
    let mut x = Vector::zeros(2)?;
    let report = solver.solve_into(
        &a.view().try_finite()?,
        &b,
        x.view_mut(),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!(
        "least squares: {:?}, x={:?}, diagnostics={:?}",
        report.termination(),
        x.as_slice(),
        report.krylov()
    );

    let indefinite = Matrix::from_vec(2, 2, vec![0., 1., 1., 0.])?;
    let result = MinresSolver::new(tolerance, budget).solve_with_options(
        &indefinite.view().try_finite()?,
        &Vector::from_vec(vec![1., 2.]),
        Preconditioning::None,
        None,
    )?;
    println!(
        "MINRES: {:?}, x={:?}",
        result.report().termination(),
        result.solution().as_slice()
    );
    Ok(())
}
