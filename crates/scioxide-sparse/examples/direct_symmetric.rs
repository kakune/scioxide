//! Reusable SPD and indefinite factors over one sparse pattern.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_sparse::{
    DuplicatePolicy, ExplicitZeroPolicy, SparseCholesky, SparseLdl, SparseOperation,
    SparseSymmetricLimits, SparseSymmetricSymbolic,
};
fn main() -> Result<(), SciOxideError> {
    let mut a = {
        let mut builder = scioxide_sparse::SparseBuilder::new(2, 2);
        builder
            .extend(&[0, 0, 1, 1], &[0, 1, 0, 1], &[4., 1., 1., 3.])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }?;
    let plan = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        None,
        SparseSymmetricLimits::new(1, 3, 4096),
    )?;
    let context = ExecutionContext::sequential();
    let chol = SparseCholesky::factor(plan.clone(), a.view())?;
    println!(
        "tree={:?}, SPD solution={:?}",
        plan.elimination_tree().collect::<Vec<_>>(),
        chol.solve(&[5., 4.])?
    );
    let mut factor = SparseLdl::factor(plan.clone(), a.view())?;
    let mut workspace = (plan).workspace(())?;
    a.copy_values_from(&[0., 2., 2., 0.])?;
    factor.refactor(
        a.view(),
        scioxide_sparse::SparseSolveOptions::new(),
        &mut workspace,
        context,
    )?;
    let mut output = [0.; 2];
    let mut scratch = [0.; 4];
    factor.solve_into(
        &[2., 2.],
        SparseOperation::Normal,
        scioxide_sparse::SparseSolveOptions::new(),
        &mut output,
        &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut scratch),
        context,
    )?;
    println!(
        "inertia={:?}, permutation={:?}, solution={output:?}",
        factor.inertia()?,
        factor.permutation()
    );
    Ok(())
}
