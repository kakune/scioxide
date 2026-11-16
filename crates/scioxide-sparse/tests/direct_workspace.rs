//! Requirements, scoped transactions and all triangular orientations.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AttemptOutcome, Capacity, ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_sparse::{
    SparseLu, SparseLuLimits, SparseLuSymbolic, SparseLuWorkspace, SparseOperation, SparseOrdering,
    SparseSolveOptions, SparseSolveWorkspace, SparseSymmetricLimits, SparseSymmetricSymbolic,
    SparseSymmetricWorkspace,
};

#[test]
fn numeric_and_solve_workspaces_follow_the_requirements_protocol() -> Result<(), SciOxideError> {
    let a =
        scioxide_sparse::CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 4.])?;
    let plan = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseLuLimits::new(2, 8192),
    )?;
    let req = plan.workspace_requirements::<f64>(())?;
    let mut numeric = SparseLuWorkspace::new();
    assert_eq!(numeric.check(&req)?, Capacity::NeedsGrowth);
    numeric.try_reserve(&req)?;
    assert_eq!(numeric.check(&req)?, Capacity::Sufficient);
    let mut factor = SparseLu::factor(plan, a.view())?;
    factor.refactor(
        a.view(),
        SparseSolveOptions::new(),
        &mut numeric,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(
        numeric.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let req = factor.workspace_requirements(2)?;
    assert_eq!(req.scalar_len(), 8);
    let mut solve = SparseSolveWorkspace::new();
    assert_eq!(solve.check(&req)?, Capacity::NeedsGrowth);
    solve.try_reserve(&req)?;
    let mut out = [19.; 2];
    factor.solve_into(
        &[4., 8.],
        SparseOperation::Normal,
        SparseSolveOptions::new(),
        &mut out,
        &mut solve,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(out, [2.; 2]);
    assert_eq!(
        solve.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let before = out;
    assert_eq!(
        factor
            .solve_into(
                &[f64::NAN, 8.],
                SparseOperation::Normal,
                SparseSolveOptions::new(),
                &mut out,
                &mut solve,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(out, before);
    assert_eq!(
        solve.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(
        factor
            .workspace_requirements(usize::MAX)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let symbolic = SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        None,
        SparseSymmetricLimits::new(2, 2, 8192),
    )?;
    let req = symbolic.workspace_requirements::<f64>(())?;
    let mut symmetric = SparseSymmetricWorkspace::<f64>::new();
    assert_eq!(symmetric.check(&req)?, Capacity::NeedsGrowth);
    symmetric.try_reserve(&req)?;
    assert_eq!(symmetric.check(&req)?, Capacity::Sufficient);
    Ok(())
}

#[test]
fn borrowed_solve_storage_does_not_consume_the_owned_byte_allowance() {
    let a = scioxide_sparse::CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64]).unwrap();
    let roomy = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseLuLimits::new(1, usize::MAX),
    )
    .unwrap();
    let limit = roomy
        .workspace_requirements::<f64>(())
        .unwrap()
        .total_bytes();
    let plan = SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseLuLimits::new(1, limit),
    )
    .unwrap();
    let factor = SparseLu::factor(plan, a.view()).unwrap();
    assert_eq!(
        factor.workspace(1).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut storage = [71.; 5];
    let mut workspace = SparseSolveWorkspace::borrowed(&mut storage);
    workspace
        .try_reserve(&factor.workspace_requirements(1).unwrap())
        .unwrap();
    let mut output = [0.];
    factor
        .solve_into(
            &[4.],
            SparseOperation::Normal,
            SparseSolveOptions::new(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [2.]);
    assert_eq!(&storage[2..], &[71.; 3]);
}
