//! Reuse symbolic ordering and storage across changing numeric values.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_sparse::{
    DuplicatePolicy, ExplicitZeroPolicy, SparseLu, SparseLuLimits, SparseLuSymbolic,
    SparseOperation, approximate_minimum_degree,
};
fn main() -> Result<(), SciOxideError> {
    let mut matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4., 1., 2., 3.])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }?;
    let pattern = matrix.structural_pattern();
    let ordering = approximate_minimum_degree(pattern)?;
    let symbolic = SparseLuSymbolic::analyze(
        pattern,
        scioxide_sparse::SparseOrdering::Supplied(&ordering),
        SparseLuLimits::new(4, 4096),
    )?;
    let mut factor = SparseLu::factor(symbolic.clone(), matrix.view())?;
    let mut workspace = (symbolic).workspace(())?;
    let mut scratch = [0.; 4];
    let mut solution = [0.; 2];
    for diagonal in [4., 5.] {
        matrix.set_value(0, diagonal)?;
        factor.refactor(
            matrix.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        factor.solve_into(
            &[diagonal + 1., 5.],
            SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            &mut solution,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
            ExecutionContext::sequential(),
        )?;
        println!(
            "solution: {solution:?}; stored factors: {}",
            factor.factor_entries()
        );
    }
    Ok(())
}
