//! Prepare once, then reuse one caller-owned workspace for multigrid corrections.
use scioxide_core::{BlockVectorView, ExecutionContext, LinearOperator, SciOxideError};
use scioxide_sparse::{
    AmgCycleOptions, AmgHierarchy, AmgLimits, AmgPreconditioner, AmgSetupOptions, DuplicatePolicy,
    ExplicitZeroPolicy, SparseBuilder, SparseSolveOptions, SparseSymmetricLimits,
};
fn main() -> Result<(), SciOxideError> {
    let n = 64;
    let mut builder = SparseBuilder::new(n, n);
    for i in 0..n {
        builder.push(i, i, 2.0_f64)?;
        if i > 0 {
            builder.push(i, i - 1, -1.)?;
        }
        if i + 1 < n {
            builder.push(i, i + 1, -1.)?;
        }
    }
    let a = builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Drop)?;
    let context = ExecutionContext::sequential();
    let options = SparseSolveOptions::new();
    let constant = vec![1.; n];
    let hierarchy = AmgHierarchy::build(
        a.view(),
        BlockVectorView::new(&constant, n, 1)?,
        AmgSetupOptions::new(),
        AmgLimits::new(16, 100_000, 64 * 1024 * 1024),
        options,
        context,
    )?;
    println!("Hierarchy: {:?}", hierarchy.diagnostics());
    let p = AmgPreconditioner::build(
        hierarchy,
        AmgCycleOptions::new(),
        SparseSymmetricLimits::new(512, 0, 8 * 1024 * 1024),
        options,
        context,
    )?;
    let mut workspace = p.workspace(1)?;
    let mut x = vec![0.; n];
    let mut residual = vec![1.; n];
    let mut correction = vec![0.; n];
    let mut aw = vec![0.; n];
    for _ in 0..30 {
        p.apply_into(&residual, options, &mut correction, &mut workspace, context)?;
        x.iter_mut().zip(&correction).for_each(|(x, &d)| *x += d);
        a.apply_into(&x, &mut residual, &mut aw)?;
        residual.iter_mut().for_each(|r| *r = 1. - *r);
    }
    println!(
        "Residual infinity norm: {:.3e}",
        residual.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    );
    Ok(())
}
