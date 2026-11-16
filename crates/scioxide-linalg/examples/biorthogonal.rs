//! Adjoint-capable and transpose-free Krylov solves, plus stationary sweeps.
use scioxide_core::{Budget, Complex, ResidualTolerance, SciOxideError};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgSolver, CgsSolver, GaussSeidelSolver, IterativeWorkspaceRequest, JacobiSolver, Matrix,
    QmrSolver, RelaxationWeight, RichardsonSolver, SorSolver, SorWeight, StationaryMatrix,
    TfqmrSolver, Vector,
};
pub fn main() -> Result<(), SciOxideError> {
    let tolerance = ResidualTolerance::new(0., 1e-11)?;
    let limit = Budget::try_from(200)?;
    let a = Matrix::from_vec(3, 3, vec![5., -2., 0., -1., 5., -2., 0., -1., 5.])?;
    let b = Vector::from_vec(vec![3., 2., 4.]);
    let bicg = BicgSolver::new(tolerance, limit).with_breakdown_restarts(Budget::try_from(2)?);
    for (algorithm, result) in [
        (
            scioxide_core::algorithm_id!("linear", "BiCG", 1),
            bicg.solve_with_options(&a.view().try_finite()?, &b, Preconditioning::None, None)?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "CGS", 1),
            CgsSolver::new(tolerance, limit).solve_with_options(
                &a.view().try_finite()?,
                &b,
                Preconditioning::None,
                None,
            )?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "QMR", 1),
            QmrSolver::new(tolerance, limit).solve_with_options(
                &a.view().try_finite()?,
                &b,
                Preconditioning::None,
                None,
            )?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "TFQMR", 1),
            TfqmrSolver::new(tolerance, limit).solve_with_options(
                &a.view().try_finite()?,
                &b,
                Preconditioning::None,
                None,
            )?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "Richardson", 1),
            RichardsonSolver::new(tolerance, limit, RelaxationWeight::new(0.15)?)
                .solve_with_options(&a.view().try_finite()?, &b, Preconditioning::None, None)?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "Jacobi", 1),
            JacobiSolver::new(tolerance, limit).solve_with_options(
                StationaryMatrix::Dense(a.view().try_finite()?),
                &b,
                None,
            )?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "Gauss-Seidel", 1),
            GaussSeidelSolver::new(tolerance, limit).solve_with_options(
                StationaryMatrix::Dense(a.view().try_finite()?),
                &b,
                None,
            )?,
        ),
        (
            scioxide_core::algorithm_id!("linear", "SOR", 1),
            SorSolver::new(tolerance, limit, SorWeight::new(1.1)?).solve_with_options(
                StationaryMatrix::Dense(a.view().try_finite()?),
                &b,
                None,
            )?,
        ),
    ] {
        println!(
            "{algorithm}: {:?}, iterations={}, residual={:?}",
            result.report().termination(),
            result.report().iterations(),
            result.report().diagnostic_residual_norm()
        );
        let _ = result.into_converged_solution()?;
    }
    // Complex QMR uses the conjugate adjoint and reusable caller storage.
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(3., 1.),
            Complex::new(1., -1.),
            Complex::new(0., 1.),
            Complex::new(4., -1.),
        ],
    )?;
    let b = Vector::from_vec(vec![Complex::new(1., 2.), Complex::new(3., -1.)]);
    let solver = QmrSolver::new(tolerance, limit).with_history(true);
    let mut workspace = solver.workspace(&IterativeWorkspaceRequest::adjoint(
        &a.view().try_finite()?,
    )?)?;
    let mut x = Vector::zeros(2)?;
    let report = solver.solve_into(
        &a.view().try_finite()?,
        &b,
        Preconditioning::None,
        x.view_mut(),
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!(
        "complex QMR: {:?}, x={:?}",
        report.termination(),
        x.as_slice()
    );
    Ok(())
}
