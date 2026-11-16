//! Prepared scalar/block factors, explicit limits, and allocation-free value updates.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_sparse::{
    CsrMatrix, IncompleteFactor, IncompleteLimits, IncompleteMethod, IncompleteOptions,
    IncompleteSymbolic, SparseOperation,
};
fn main() -> Result<(), SciOxideError> {
    let matrix = CsrMatrix::from_raw(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![4., -1., -1., 4., -1., -1., 4., -1., -1., 4.],
    )?;
    for method in [
        IncompleteMethod::Ilu { level: 1 },
        IncompleteMethod::Ilut,
        IncompleteMethod::Ict,
    ] {
        for block_size in [1, 2] {
            let plan = IncompleteSymbolic::analyze(
                matrix.structural_pattern(),
                method,
                block_size,
                IncompleteLimits::new(16, 16_384),
            )?;
            let factor = IncompleteFactor::factor(
                plan,
                matrix.view(),
                IncompleteOptions::new()
                    .with_drop_tolerance(1e-4)
                    .with_max_off_diagonal(4),
            )?;
            let mut solution = [0.; 4];
            let mut workspace = factor.workspace(1)?;
            factor.solve_into(
                &[3., 2., 2., 3.],
                SparseOperation::Adjoint,
                scioxide_sparse::SparseSolveOptions::new(),
                &mut solution,
                &mut workspace,
                ExecutionContext::sequential(),
            )?;
            println!(
                "{method:?}, block={block_size}: {solution:?}; {:?}",
                factor.diagnostics()
            );
        }
    }
    Ok(())
}
