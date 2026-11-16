//! Preconditioner examples for scioxide-linalg.
//!
//! Demonstrates all four preconditioners and their effect on convergence:
//!
//! 1. **Baseline** — CG with no preconditioner on a sparse SPD system
//! 2. **Jacobi** — diagonal scaling, reduces iteration count
//! 3. **IC(0)** — incomplete Cholesky, further improvement for SPD
//! 4. **ILU(0)** — incomplete LU, for general nonsymmetric systems with GMRES
//!
//! Run: `cargo run --example preconditioners -p scioxide-linalg`

use scioxide_core::LinearOperator;
use scioxide_core::{Budget, ResidualTolerance, SciOxideError};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{CgSolver, GmresSolver, IterativeSolveResult, Vector};
use scioxide_sparse::{
    CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, IncompleteFactor, IncompleteLimits,
    IncompleteMethod, IncompleteOptions, IncompleteSymbolic, JacobiPreconditioner, SparseBuilder,
};

fn diagnostic_residual(result: &IterativeSolveResult<f64>) -> Result<f64, SciOxideError> {
    result
        .report()
        .diagnostic_residual_norm()
        .ok_or_else(|| SciOxideError::external("solver result omitted residual diagnostics"))
}

/// Build a sparse SPD system: tridiagonal with dominant diagonal.
///
/// The matrix has the structure `diag(alpha) + off-diag(-1)` which is SPD
/// when `alpha > 2`.
fn build_sparse_spd(n: usize, alpha: f64) -> Result<CsrMatrix<f64>, SciOxideError> {
    let mut coo = SparseBuilder::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, alpha)?;
        if i > 0 {
            coo.push(i, i - 1, -1.0)?;
            coo.push(i - 1, i, -1.0)?;
        }
    }
    coo.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}

/// Build a sparse nonsymmetric diagonally dominant system.
fn build_sparse_nonsymmetric(n: usize) -> Result<CsrMatrix<f64>, SciOxideError> {
    let mut coo = SparseBuilder::<f64>::new(n, n);
    for i in 0..n {
        coo.push(i, i, 10.0)?;
        if i > 0 {
            coo.push(i, i - 1, -3.0)?;
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0)?;
        }
    }
    coo.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}

// ---------------------------------------------------------------------------
// Part 1: CG baseline (no preconditioner)
// ---------------------------------------------------------------------------

fn part1_baseline() -> Result<usize, SciOxideError> {
    println!("=== Part 1: CG baseline (no preconditioner) ===");
    println!();

    let n = 200;
    let csr = build_sparse_spd(n, 4.0)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let solver = CgSolver::new(ResidualTolerance::new(1e-10, 0.0)?, Budget::try_from(2000)?);
    let result = solver.solve_with_options(&csr, &b, Preconditioning::None, None)?;

    println!("  Size: {n}x{n}");
    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "CG baseline did not converge");

    println!();
    println!("  Part 1 passed.");
    println!();
    Ok(result.report().iterations())
}

// ---------------------------------------------------------------------------
// Part 2: CG + Jacobi preconditioner
// ---------------------------------------------------------------------------

fn part2_jacobi() -> Result<usize, SciOxideError> {
    println!("=== Part 2: CG + Jacobi preconditioner ===");
    println!();

    let n = 200;
    let csr = build_sparse_spd(n, 4.0)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let precond = JacobiPreconditioner::new(&csr)?;
    let solver = CgSolver::new(ResidualTolerance::new(1e-10, 0.0)?, Budget::try_from(2000)?);
    let result = solver.solve_with_options(&csr, &b, Preconditioning::Inverse(&precond), None)?;

    println!("  Size: {n}x{n}");
    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "CG + Jacobi did not converge");

    println!();
    println!("  Part 2 passed.");
    println!();
    Ok(result.report().iterations())
}

// ---------------------------------------------------------------------------
// Part 3: CG + IC(0) preconditioner
// ---------------------------------------------------------------------------

fn part3_ic0() -> Result<usize, SciOxideError> {
    println!("=== Part 3: CG + IC(0) preconditioner ===");
    println!();

    let n = 200;
    let csr = build_sparse_spd(n, 4.0)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    let precond = prepare(&csr, IncompleteMethod::Ic0)?;
    let solver = CgSolver::new(ResidualTolerance::new(1e-10, 0.0)?, Budget::try_from(2000)?);
    let result = solver.solve_with_options(&csr, &b, Preconditioning::Inverse(&precond), None)?;

    println!("  Size: {n}x{n}");
    println!("  Converged: {}", result.is_success());
    println!("  Iterations: {}", result.report().iterations());
    println!("  Residual: {:.2e}", diagnostic_residual(&result)?);

    assert!(result.is_success(), "CG + IC(0) did not converge");

    println!();
    println!("  Part 3 passed.");
    println!();
    Ok(result.report().iterations())
}

// ---------------------------------------------------------------------------
// Part 4: GMRES + ILU(0) on nonsymmetric system
// ---------------------------------------------------------------------------

fn part4_ilu0() -> Result<(), SciOxideError> {
    println!("=== Part 4: GMRES + ILU(0) on nonsymmetric system ===");
    println!();

    let n = 200;
    let csr = build_sparse_nonsymmetric(n)?;
    let b = Vector::<f64>::from_vec(vec![1.0; n]);

    // Without preconditioner
    let op: &dyn LinearOperator<f64> = &csr;
    let solver = GmresSolver::new(ResidualTolerance::new(1e-10, 0.0)?, Budget::try_from(2000)?)
        .with_restart(Budget::try_from(50)?);
    let result_base = solver.solve_with_options(op, &b, Preconditioning::None, None)?;

    // With ILU(0)
    let precond = prepare(&csr, IncompleteMethod::Ilu { level: 0 })?;
    let result_ilu = solver.solve_with_options(op, &b, Preconditioning::Inverse(&precond), None)?;

    println!("  Size: {n}x{n}");
    println!();
    println!(
        "  {:20} {:>10} {:>12} {:>12}",
        "Method", "Converged", "Iterations", "Residual"
    );
    println!("  {:-<20} {:-<10} {:-<12} {:-<12}", "", "", "", "");
    println!(
        "  {:20} {:>10} {:>12} {:>12.2e}",
        "GMRES (no precond)",
        result_base.is_success(),
        result_base.report().iterations(),
        diagnostic_residual(&result_base)?
    );
    println!(
        "  {:20} {:>10} {:>12} {:>12.2e}",
        "GMRES + ILU(0)",
        result_ilu.is_success(),
        result_ilu.report().iterations(),
        diagnostic_residual(&result_ilu)?
    );

    assert!(result_base.is_success(), "GMRES baseline did not converge");
    assert!(result_ilu.is_success(), "GMRES + ILU(0) did not converge");

    println!();
    println!("  Part 4 passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-linalg Preconditioner Examples");
    println!("======================================");
    println!();

    let baseline_iters = part1_baseline()?;
    let jacobi_iters = part2_jacobi()?;
    let ic0_iters = part3_ic0()?;
    part4_ilu0()?;

    println!("--- Summary (CG on SPD system) ---");
    println!("  No preconditioner: {baseline_iters} iterations");
    println!("  Jacobi:            {jacobi_iters} iterations");
    println!("  IC(0):             {ic0_iters} iterations");
    println!();

    println!("All examples completed successfully.");
    Ok(())
}

fn prepare(
    csr: &CsrMatrix<f64>,
    method: IncompleteMethod,
) -> Result<IncompleteFactor<f64>, SciOxideError> {
    let plan = IncompleteSymbolic::analyze(
        csr.structural_pattern(),
        method,
        1,
        IncompleteLimits::new(csr.nnz(), 1_000_000),
    )?;
    IncompleteFactor::factor(plan, csr.view(), IncompleteOptions::new())
}
