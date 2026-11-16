//! Compact band and packed triangular solves with explicit workspace.
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::{
    BandedLuDecomposition, BandedMatrix, LinalgExecutionOptions, Matrix, PackedKind, PackedMatrix,
    StructuredOperation, StructuredWorkspaceRequest, TriangularPlan, VectorView,
};
pub fn main() -> Result<(), SciOxideError> {
    // Column-major bands: padding, diagonal, lower for the first column.
    let a = BandedMatrix::new(
        3,
        3,
        1,
        1,
        3,
        vec![0.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 0.0],
    )?;
    let options = LinalgExecutionOptions::new();
    let factor = BandedLuDecomposition::factor(&a)?;
    let rhs = Matrix::from_vec(3, 2, vec![5.0, 10.0, 6.0, 12.0, 5.0, 10.0])?;
    let mut output = Matrix::zeros(3, 2)?;
    let mut workspace = BandedLuDecomposition::workspace(
        &StructuredWorkspaceRequest::solve_matrix(3, 2, &options),
    )?;
    factor
        .solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })?;
    println!("two RHS solutions: {:?}", output.as_slice());
    let packed = PackedMatrix::new(
        2,
        Triangle::Lower,
        PackedKind::Triangular,
        vec![2.0, 1.0, 3.0],
    )?;
    let plan = TriangularPlan::from_packed(packed.view(), DiagonalKind::Explicit)?;
    println!(
        "packed solution: {:?}",
        plan.solve(
            StructuredOperation::Normal,
            VectorView::from_slice(&[2.0, 4.0])
        )?
        .as_slice()
    );
    Ok(())
}
