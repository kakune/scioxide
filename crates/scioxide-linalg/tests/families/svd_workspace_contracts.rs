//! Public requirements, execution and transactional output contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError};
use scioxide_linalg::*;

fn sequential() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential)
}

/// `H₁ diag(σ) H₂` with Householder reflectors `H = I − 2 v vᵀ / vᵀv`.
fn reflected_spectrum(sigma: &[f64], seed: u64) -> Matrix<f64> {
    let n = sigma.len();
    let column = |lane: usize| -> Vec<f64> {
        crate::support::decomposition::generated_matrix::<f64>(n, 2, seed).as_slice()[lane..]
            .iter()
            .step_by(2)
            .copied()
            .collect()
    };
    let (left, right) = (column(0), column(1));
    let reflect = |v: &[f64], i: usize, j: usize| {
        let squares: f64 = v.iter().map(|x| x * x).sum();
        f64::from(u8::from(i == j)) - 2.0 * v[i] * v[j] / squares
    };
    let data = (0..n * n)
        .map(|index| {
            let (i, j) = (index / n, index % n);
            (0..n)
                .map(|k| reflect(&left, i, k) * sigma[k] * reflect(&right, k, j))
                .sum()
        })
        .collect();
    Matrix::from_vec(n, n, data).unwrap()
}

#[test]
fn values_of_blocked_bidiagonals_do_not_read_stale_workspace() -> Result<(), SciOxideError> {
    // Orders above the unblocked limit use the blocked reduction, whose last
    // superdiagonal slot has no reflector.
    let options = sequential();
    let dirty = crate::support::decomposition::scale_matrix(
        &crate::support::decomposition::generated_matrix::<f64>(100, 100, 1),
        1e300,
    );
    let mut sigma: Vec<f64> = (0..120).map(|i| 2.0 + f64::from(i)).collect();
    sigma[..3].copy_from_slice(&[1.0, 1.0 + 1e-9, 1.0 + 2e-9]);
    let matrix = reflected_spectrum(&sigma, 7);
    let mut workspace = LinalgWorkspace::new();
    let mut values = SvdValues::empty();
    SvdValues::compute_into(
        &dirty,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    SvdValues::compute_into(
        &matrix,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    sigma.sort_by(|a, b| b.total_cmp(a));
    for (computed, exact) in values.singular_values().iter().zip(&sigma) {
        assert!(
            (computed - exact).abs() <= 1e-12 * exact,
            "{computed} vs {exact}"
        );
    }
    Ok(())
}
