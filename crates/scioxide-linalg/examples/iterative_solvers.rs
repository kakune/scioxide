//! Iterative solver examples for scioxide-linalg.
//!
//! Demonstrates the three Krylov subspace solvers:
//!
//! 1. **CG** on a dense SPD system — basic usage
//! 2. **GMRES** on a nonsymmetric sparse system — restart parameter, convergence history
//! 3. **`BiCGSTAB`** as an alternative for nonsymmetric systems
//! 4. **Solver comparison** — same problem, different solvers
//!
//! Run: `cargo run --example iterative_solvers -p scioxide-linalg`

#![allow(clippy::cast_precision_loss)]

use scioxide_core::LinearOperator;
use scioxide_core::{Budget, ResidualTolerance, SciOxideError};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BicgstabSolver, CgSolver, GmresSolver, IterativeSolveResult, Matrix, Vector,
};
use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-8;

fn diagnostic_residual(result: &IterativeSolveResult<f64>) -> Result<f64, SciOxideError> {
    result
        .report()
        .diagnostic_residual_norm()
        .ok_or_else(|| SciOxideError::external("solver result omitted residual diagnostics"))
}

/// Build a dense SPD matrix of the given size (diagonally dominant).
fn build_spd_dense(n: usize) -> Result<Matrix<f64>, SciOxideError> {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = if i == j {
                (2 * n) as f64
            } else {
                1.0 / (1.0 + (i as f64 - j as f64).abs())
            };
        }
    }
    Matrix::from_vec(n, n, data)
}

/// Build a sparse nonsymmetric diagonally dominant system in CSR format.
fn build_nonsymmetric_sparse(n: usize) -> Result<scioxide_sparse::CsrMatrix<f64>, SciOxideError> {
    let mut coo = SparseBuilder::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 10.0)?;
        if i > 0 {
            coo.push(i, i - 1, -2.0)?;
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0)?; // asymmetric: -1 above, -2 below
        }
    }
    coo.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}

// ---------------------------------------------------------------------------
// Part 1: CG on a dense SPD system
// ---------------------------------------------------------------------------

fn part1_cg_dense() -> Result<(), SciOxideError> {
    println!("=== Part 1: CG on dense SPD system ===");
    println!();

    let n = 10;
    let a = build_spd_dense(n)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let solver = CgSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(500)?)
        .with_history(true);
    let result =
        solver.solve_with_options(&a.view().try_finite()?, &b, Preconditioning::None, None)?;

    println!("  Size: {n}x{n}");
    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual norm: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "CG did not converge");

    // Verify residual
    let ax = a.view().mul_vec(result.solution())?;
    let residual = ax.view().try_sub(&b)?;
    let rnorm = residual.view().norm2()?;
    println!("  Recomputed ||Ax - b|| = {rnorm:.2e}");
    assert!(rnorm < TOL);

    // History
    if let Some(history) = result.history() {
        println!("  History length: {} entries", history.len());
        assert!(history.len() <= result.report().iterations() + 1);
    }

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: GMRES on a nonsymmetric sparse system
// ---------------------------------------------------------------------------

fn part2_gmres_sparse() -> Result<(), SciOxideError> {
    println!("=== Part 2: GMRES on nonsymmetric sparse system ===");
    println!();

    let n = 50;
    let csr = build_nonsymmetric_sparse(n)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let solver = GmresSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(500)?)
        .with_restart(Budget::try_from(30)?)
        .with_history(true);

    let op: &dyn LinearOperator<f64> = &csr;
    let result = solver.solve_with_options(op, &b, Preconditioning::None, None)?;

    println!("  Size: {n}x{n}, nnz={}", csr.nnz());
    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual norm: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "GMRES did not converge");

    // Verify
    let mut ax = Vector::zeros(n)?;
    let mut operator_workspace = vec![0.0; csr.apply_workspace_len()?];
    csr.apply_into(
        result.solution().as_slice(),
        ax.as_mut_slice(),
        &mut operator_workspace,
    )?;
    let residual = ax.view().try_sub(&b)?;
    let rnorm = residual.view().norm2()?;
    println!("  Recomputed ||Ax - b|| = {rnorm:.2e}");
    assert!(rnorm < TOL);

    if let Some(history) = result.history() {
        println!(
            "  Residual history: first={:.2e}, last={:.2e}",
            history[0],
            history[history.len() - 1]
        );
    }

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: BiCGSTAB for nonsymmetric systems
// ---------------------------------------------------------------------------

fn part3_bicgstab() -> Result<(), SciOxideError> {
    println!("=== Part 3: BiCGSTAB on nonsymmetric sparse system ===");
    println!();

    let n = 50;
    let csr = build_nonsymmetric_sparse(n)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let solver = BicgstabSolver::new(ResidualTolerance::new(1e-8, 0.0)?, Budget::try_from(500)?)
        .with_history(true);
    let result = solver.solve_with_options(&csr, &b, Preconditioning::None, None)?;

    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual norm: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "BiCGSTAB did not converge");

    let mut ax = Vector::zeros(n)?;
    let mut operator_workspace = vec![0.0; csr.apply_workspace_len()?];
    csr.apply_into(
        result.solution().as_slice(),
        ax.as_mut_slice(),
        &mut operator_workspace,
    )?;
    let residual = ax.view().try_sub(&b)?;
    let rnorm = residual.view().norm2()?;
    println!("  Recomputed ||Ax - b|| = {rnorm:.2e}");
    assert!(rnorm < TOL);

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Solver comparison on the same problem
// ---------------------------------------------------------------------------

fn part4_comparison() -> Result<(), SciOxideError> {
    println!("=== Part 4: Solver comparison ===");
    println!();

    // Use a sparse SPD system so all three solvers can be applied
    let n = 30;
    let mut coo = SparseBuilder::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0)?;
        if i > 0 {
            coo.push(i, i - 1, -1.0)?;
            coo.push(i - 1, i, -1.0)?; // symmetric
        }
    }
    let csr = coo.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let op: &dyn LinearOperator<f64> = &csr;

    // CG
    let cg = CgSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(500)?);
    let cg_result = cg.solve_with_options(op, &b, Preconditioning::None, None)?;

    // GMRES
    let gmres = GmresSolver::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(500)?)
        .with_restart(Budget::try_from(30)?);
    let gmres_result = gmres.solve_with_options(op, &b, Preconditioning::None, None)?;

    // BiCGSTAB
    let bicgstab = BicgstabSolver::new(ResidualTolerance::new(1e-8, 0.0)?, Budget::try_from(500)?);
    let bicgstab_result = bicgstab.solve_with_options(op, &b, Preconditioning::None, None)?;

    println!("  Problem: {n}x{n} tridiagonal SPD");
    println!();
    println!(
        "  {:12} {:>10} {:>12} {:>10}",
        "Solver", "Converged", "Iterations", "Residual"
    );
    println!("  {:-<12} {:-<10} {:-<12} {:-<10}", "", "", "", "");
    println!(
        "  {:12} {:>10} {:>12} {:>10.2e}",
        "CG",
        cg_result.is_success(),
        cg_result.report().iterations(),
        diagnostic_residual(&cg_result)?
    );
    println!(
        "  {:12} {:>10} {:>12} {:>10.2e}",
        "GMRES",
        gmres_result.is_success(),
        gmres_result.report().iterations(),
        diagnostic_residual(&gmres_result)?
    );
    println!(
        "  {:12} {:>10} {:>12} {:>10.2e}",
        "BiCGSTAB",
        bicgstab_result.is_success(),
        bicgstab_result.report().iterations(),
        diagnostic_residual(&bicgstab_result)?
    );

    assert!(cg_result.is_success());
    assert!(gmres_result.is_success());
    assert!(bicgstab_result.is_success());

    // All solutions should be close to each other
    for i in 0..n {
        let cg_xi = cg_result.solution().view().get(i)?;
        let gmres_xi = gmres_result.solution().view().get(i)?;
        let bicgstab_xi = bicgstab_result.solution().view().get(i)?;
        assert!(
            (cg_xi - gmres_xi).abs() < 1e-6,
            "CG and GMRES disagree at [{i}]"
        );
        assert!(
            (cg_xi - bicgstab_xi).abs() < 1e-6,
            "CG and BiCGSTAB disagree at [{i}]"
        );
    }
    println!();
    println!("  All solvers agree on the solution.");

    println!();
    println!("  Part 4 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-linalg Iterative Solver Examples");
    println!("========================================");
    println!();

    part1_cg_dense()?;
    part2_gmres_sparse()?;
    part3_bicgstab()?;
    part4_comparison()?;

    println!("All examples completed successfully.");
    Ok(())
}
