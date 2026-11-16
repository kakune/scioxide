//! Explicit storage expansion, reusable products, and numerical dropping.
use scioxide_core::{ExecutionContext, SciOxideError, Tolerance};
use scioxide_sparse::{
    BsrMatrix, DiaMatrix, SparseProductOptions, SparseProductPlan, SparsePruning, sparse_diagonals,
};
fn main() -> Result<(), SciOxideError> {
    let a = sparse_diagonals(4, 4, &[0, 1], &[&[4.0_f64; 4], &[-1.0; 3]])?;
    let requirements = BsrMatrix::conversion_requirements(a.view(), 2, 2)?;
    let blocks = BsrMatrix::from_sparse(a.view(), 2, 2)?;
    let diagonals = DiaMatrix::from_sparse(a.view())?;
    println!(
        "CSR entries: {}; BSR slots: {}; DIA offsets: {:?}",
        a.nnz(),
        requirements.values(),
        diagonals.offsets()
    );
    let plan = SparseProductPlan::new(&a, &a)?;
    let mut output = plan.create_output()?;
    let mut scratch = plan.workspace(SparseProductOptions::default())?;
    plan.evaluate_into(
        &a,
        &a,
        SparseProductOptions::default(),
        &mut output,
        &mut scratch,
        ExecutionContext::sequential(),
    )?;
    let policy = SparsePruning::new(Tolerance::new(1.0, 0.0)?, 0.0)?;
    let pruned = blocks.pruned(policy)?;
    println!(
        "A squared entries: {}; pruned blocks: {}",
        output.nnz(),
        pruned.block_count()
    );
    Ok(())
}
