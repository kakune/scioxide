//! Solve an inconsistent tall system and its adjoint from one sparse QR factor.
use scioxide_core::{ExecutionContext, RankThreshold, SciOxideError};
use scioxide_sparse::{
    DuplicatePolicy, ExplicitZeroPolicy, SparseQr, SparseQrLimits, SparseQrMinimumNormPlan,
    SparseQrSymbolic, SparseSolveOptions,
};

fn main() -> Result<(), SciOxideError> {
    let matrix = {
        let mut builder = scioxide_sparse::SparseBuilder::new(3, 2);
        builder
            .extend(&[0, 1, 2, 2], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0])
            .and_then(|()| {
                builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            })
    }?;
    let symbolic = SparseQrSymbolic::analyze(
        matrix.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        SparseQrLimits::new(32, 16, 16 * 1024),
    )?;
    let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
    let factor = SparseQr::factor(symbolic, matrix.view())?;
    let rank = RankThreshold::new(0.0, 1e-12)?;
    let plan = SparseQrMinimumNormPlan::build(factor, rank, options, scope)?;
    let (solution, report) = plan.solve(&[1.0, 2.0, 4.0])?;
    let (adjoint, _) = plan.solve_adjoint(&[1.0, 2.0])?;
    println!(
        "least-squares x={solution:?}, rank={}, residual={}",
        report.rank(),
        report.residual().raw_norm()
    );
    println!("adjoint minimum-norm x={adjoint:?}");
    Ok(())
}
