//! Compact structured solving and constituent-factor composition.
use scioxide_core::{ExecutionContext, LinearOperator, SciOxideError, Tolerance};
use scioxide_linalg::{
    KroneckerOperator, LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix,
    OperatorWorkspaceRequest, StructuredMatrixView, StructuredOperation, StructuredSolveOptions,
    StructuredSolvePlan, StructuredWorkspaceRequest, VectorViewMut,
};
pub fn main() -> Result<(), SciOxideError> {
    let column = [4.0_f64, 1., -1.];
    let row = [4., 2., 0.5];
    let matrix = StructuredMatrixView::toeplitz(&column, &row)?;
    let options = StructuredSolveOptions::new()
        .with_tolerance(Tolerance::new(0., 1e-12)?)
        .with_execution_options(LinalgExecutionOptions::new().with_byte_limit(4096));
    let plan = StructuredSolvePlan::new(matrix, &options)?;
    let mut scratch = plan.workspace(&StructuredWorkspaceRequest::solve(
        3,
        &scioxide_linalg::LinalgExecutionOptions::new(),
    ))?;
    let mut solution = [0.; 3];
    let report = plan.solve_into(
        StructuredOperation::Normal,
        &[6.5, 7., 4.][..],
        VectorViewMut::from_slice(&mut solution),
        &mut scratch,
        ExecutionContext::sequential_native(),
    )?;
    require_accepted(report.termination())?;
    println!("Toeplitz solution: {solution:?}; report {report:?}");
    // Defaults reserve GKO after Levinson, including for symmetric/Hermitian input.
    // Add .with_dense_fallback() to permit LU after GKO rejection.
    let symmetric =
        StructuredSolvePlan::new(StructuredMatrixView::toeplitz(&column, &column)?, &options)?;
    let mut scratch = symmetric.workspace(
        &StructuredWorkspaceRequest::solve(3, &scioxide_linalg::LinalgExecutionOptions::new())
            .with_dense_fallback(),
    )?;
    let report = symmetric.solve_into(
        StructuredOperation::Normal,
        &[4., 6., 4.][..],
        VectorViewMut::from_slice(&mut solution),
        &mut scratch,
        ExecutionContext::sequential_native(),
    )?;
    require_accepted(report.termination())?;
    println!("Symmetric Toeplitz solution: {solution:?}; report {report:?}");
    let a =
        matrix.materialize_with_options(&LinalgExecutionOptions::new().with_byte_limit(9 * 8))?;
    let b = Matrix::from_vec(2, 2, vec![2., 1., 0., 3.])?;
    let a_finite = a.view().try_finite()?;
    let b_finite = b.view().try_finite()?;
    let product = KroneckerOperator::new(&a_finite, &b_finite)?;
    let fa = LuDecomposition::factor(&a)?;
    let fb = LuDecomposition::factor(&b)?;
    let ai = fa.inverse_operator();
    let bi = fb.inverse_operator();
    let mut rhs = [0.; 6];
    let mut apply = vec![0.; product.apply_workspace_len()?];
    product.apply_into(&[1.; 6], &mut rhs, &mut apply)?;
    let mut out = [0.; 6];
    let mut workspace = LinalgWorkspace::workspace(&OperatorWorkspaceRequest::kronecker_solve(
        &product,
        [&ai, &bi],
        &LinalgExecutionOptions::new(),
    )?)?;
    let report = product.solve_into(
        [&ai, &bi],
        scioxide_linalg::VectorView::from_slice(&rhs),
        &scioxide_linalg::CompositeSolveOptions::new().with_tolerance(Tolerance::new(0., 1e-12)?),
        scioxide_linalg::VectorViewMut::from_slice(&mut out),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    require_accepted(report.termination())?;
    println!("Kronecker solution: {out:?}");
    Ok(())
}

fn require_accepted(
    termination: scioxide_linalg::DirectSolveTermination,
) -> Result<(), SciOxideError> {
    if termination.is_success() {
        Ok(())
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "structured residual was not accepted",
        ))
    }
}
