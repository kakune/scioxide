//! Two overlapping local LU solves composed without geometry-specific partitioning.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_sparse::{
    CsrMatrix, SchwarzKind, SchwarzLocalSolveView, SchwarzOptions, SchwarzPlan,
    SchwarzSubdomainView, SparseLu, SparseLuLimits, SparseLuSymbolic,
};

fn main() -> Result<(), SciOxideError> {
    let local = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, -1.0, -1.0, 2.0],
    )?;
    let symbolic = SparseLuSymbolic::analyze(
        local.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        SparseLuLimits::new(4, 4096),
    )?;
    let inverse = SparseLu::factor(symbolic, local.view())?;
    let half = [1.0, 0.5];
    let other_half = [0.5, 1.0];
    let domains = [
        SchwarzSubdomainView {
            indices: &[0, 1],
            weights: Some(&half),
            solve: SchwarzLocalSolveView::InverseOperator(&inverse),
        },
        SchwarzSubdomainView {
            indices: &[1, 2],
            weights: Some(&other_half),
            solve: SchwarzLocalSolveView::InverseOperator(&inverse),
        },
    ];
    let plan = SchwarzPlan::with_options(
        3,
        SchwarzKind::Additive,
        &domains,
        None,
        SchwarzOptions::new()
            .with_max_plan_bytes(4096)
            .with_max_workspace_bytes(4096),
    )?;
    let mut workspace = plan.workspace(1)?;
    let mut correction = [0.; 3];
    plan.apply_into(
        &[1., 0., 0.],
        scioxide_sparse::SparseSolveOptions::new(),
        &mut correction,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!("weighted Schwarz correction: {correction:?}");
    Ok(())
}
