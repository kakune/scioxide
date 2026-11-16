//! Sparse least-squares checks against a manufactured independent reference.
#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, RankThreshold};
use scioxide_sparse::{
    CsrMatrix, SparseOperation, SparseOrdering, SparseQr, SparseQrLimits, SparseQrMinimumNormPlan,
    SparseQrSymbolic, SparseSolveOptions,
};

#[test]
fn large_banded_least_squares_has_independent_residuals() {
    let (m, n) = (5000, 2000);
    let mut pointers = Vec::with_capacity(m + 1);
    let mut columns = Vec::with_capacity(2 * n);
    let mut values = Vec::with_capacity(2 * n);
    pointers.push(0);
    for row in 0..m {
        if row > 0 && row <= n {
            columns.push(row - 1);
            values.push(-1.);
        }
        if row < n {
            columns.push(row);
            values.push(1.);
        }
        pointers.push(columns.len());
    }
    let a = CsrMatrix::from_raw(m, n, pointers, columns, values).unwrap();
    let expected: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from(i % 17).unwrap()) / 17.)
        .collect();
    // A is the incidence matrix of a path with n+1 vertices. A constant
    // residual on those vertices is orthogonal to every column; other rows
    // are zero. Thus b=A*x+r has the unique least-squares solution x.
    let mut rhs = vec![0.5; m];
    for row in 0..=n {
        rhs[row] = 1. + if row < n { expected[row] } else { 0. }
            - if row > 0 { expected[row - 1] } else { 0. };
    }
    let symbolic = SparseQrSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseQrLimits::new(20 * n, 4 * n, 64 << 20),
    )
    .unwrap();
    let (options, scope) = (SparseSolveOptions::new(), ExecutionContext::sequential());
    let factor = SparseQr::factor(symbolic, a.view()).unwrap();
    let plan = SparseQrMinimumNormPlan::build(
        factor,
        RankThreshold::new(0., 1e-12).unwrap(),
        options,
        scope,
    )
    .unwrap();
    assert_eq!(plan.rank(), n);
    let mut workspace = plan.workspace(1).unwrap();
    let mut x = vec![0.; n];
    let report = plan
        .solve_into(
            &rhs,
            SparseOperation::Normal,
            options,
            &mut x,
            &mut workspace,
            scope,
        )
        .unwrap();
    for (actual, expected) in x.iter().zip(&expected) {
        assert!((actual - expected).abs() < 2e-8);
    }
    let residual: Vec<f64> = (0..m)
        .map(|row| {
            let product = if row < n { x[row] } else { 0. }
                - if row > 0 && row <= n { x[row - 1] } else { 0. };
            product - rhs[row]
        })
        .collect();
    let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    let normal_norm = (0..n)
        .map(|col| (residual[col] - residual[col + 1]).powi(2))
        .sum::<f64>()
        .sqrt();
    let reference = (f64::from(u32::try_from(n + 1).unwrap())
        + 0.25 * f64::from(u32::try_from(m - n - 1).unwrap()))
    .sqrt();
    assert!((report.residual().raw_norm() - reference).abs() < 1e-9);
    assert!((report.residual().raw_norm() - residual_norm).abs() < 1e-9);
    assert!((report.normal_residual().raw_norm() - normal_norm).abs() < 1e-10);
    assert!(normal_norm < 1e-9);
}
