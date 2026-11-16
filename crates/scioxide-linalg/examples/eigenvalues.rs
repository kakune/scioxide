//! Eigenvalue decomposition examples for scioxide-linalg.
//!
//! Demonstrates symmetric and general eigenvalue computation:
//!
//! 1. **Symmetric eigenvalues** — real eigenvalues, orthogonal eigenvectors, A*v = λ*v
//! 2. **General eigenvalues** — complete complex eigenvalues from Schur form
//!
//! Generalized definite pairs and selected Schur subspaces are demonstrated in
//! the `generalized_schur` example.
//!
//! Run: `cargo run --example eigenvalues -p scioxide-linalg`

use scioxide_core::SciOxideError;
use scioxide_linalg::{HermitianEigenDecomposition, Matrix};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Part 1: Symmetric eigenvalues
// ---------------------------------------------------------------------------

fn part1_symmetric() -> Result<(), SciOxideError> {
    println!("=== Part 1: Symmetric eigenvalue decomposition ===");
    println!();

    // Symmetric 3×3 matrix with known eigenvalues
    let a = Matrix::<f64>::from_vec(3, 3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])?;

    let eig = HermitianEigenDecomposition::decompose(&a, scioxide_linalg::HermitianInput::Exact)?;
    let eigenvalues = eig.eigenvalues();

    println!("  A = [[2,1,0],[1,3,1],[0,1,2]]");
    println!("  Eigenvalues: {eigenvalues:?}");
    assert_eq!(eigenvalues.len(), 3);

    // Symmetric decompositions always contain eigenvectors.
    let q = eig.eigenvectors();

    println!("  Eigenvectors available: {}x{}", q.rows(), q.cols());

    // Verify A * v_i = λ_i * v_i for each eigenpair
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        let v_i = q.view().col(i)?;
        let av = a.view().mul_vec(v_i)?;

        for j in 0..3 {
            assert!(
                (av.view().get(j)? - v_i.get(j)? * lambda).abs() < TOL,
                "A*v != λ*v for eigenpair {i}, component {j}"
            );
        }
    }
    println!("  Verified: A * v_i = λ_i * v_i for all eigenpairs");

    // Verify eigenvectors are orthogonal: Q^T * Q ≈ I
    let qtq = q.view().adjoint().try_to_matrix()?.view().mul_mat(q)?;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq.view().get(i, j)? - expected).abs() < TOL,
                "Q^T*Q[{i},{j}] != I"
            );
        }
    }
    println!("  Verified: eigenvectors are orthonormal (Q^T Q ≈ I)");

    // Verify Q * diag(λ) * Q^T ≈ A
    let mut diag_lambda = Matrix::<f64>::zeros(3, 3)?;
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        diag_lambda.view_mut().set(i, i, lambda)?;
    }
    let reconstructed = q
        .view()
        .mul_mat(&diag_lambda)?
        .view()
        .mul_mat(&q.view().adjoint().try_to_matrix()?)?;
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (reconstructed.view().get(i, j)? - a.view().get(i, j)?).abs() < TOL,
                "Q*Λ*Q^T[{i},{j}] != A"
            );
        }
    }
    println!("  Verified: Q * Λ * Q^T ≈ A (spectral theorem)");

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: General eigenvalues
// ---------------------------------------------------------------------------

fn part2_general() -> Result<(), SciOxideError> {
    println!("=== Part 2: General eigenvalue decomposition ===");
    println!();

    // Non-symmetric matrix with real eigenvalues
    // Upper triangular — eigenvalues are the diagonal entries (1, 2, 3)
    let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 4.0, 2.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0])?;

    let eig = scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&a)?;
    let eigenvalues = eig.eigenvalues();

    println!("  A = [[1,4,2],[0,2,5],[0,0,3]]  (upper triangular)");
    println!("  Eigenvalues: {eigenvalues:?}");
    assert_eq!(eigenvalues.len(), 3);

    // Sort and verify
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal));
    assert!((sorted[0].re - 1.0).abs() < TOL && sorted[0].im.abs() < TOL);
    assert!((sorted[1].re - 2.0).abs() < TOL && sorted[1].im.abs() < TOL);
    assert!((sorted[2].re - 3.0).abs() < TOL && sorted[2].im.abs() < TOL);
    println!("  Verified: eigenvalues ≈ [1, 2, 3]");

    // A rotation has a normal complex spectrum rather than an error.
    let rotation = Matrix::from_vec(2, 2, vec![0.0_f64, -1.0, 1.0, 0.0])?;
    let mut rotation_values =
        scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&rotation)?.into_eigenvalues();
    rotation_values.sort_by(|a, b| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal));
    assert!(rotation_values.iter().all(|value| value.re.abs() < TOL));
    assert!((rotation_values[0].im + 1.0).abs() < TOL);
    assert!((rotation_values[1].im - 1.0).abs() < TOL);
    println!("  Verified: rotation eigenvalues are -i and +i");

    // Diagonal matrix — eigenvalues are the diagonal entries
    let diag = Matrix::<f64>::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])?;
    let eig_diag = scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&diag)?;
    let evals = eig_diag.eigenvalues();
    println!("  Diagonal matrix eigenvalues: {evals:?}");

    let mut sorted_diag = evals.to_vec();
    sorted_diag.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal));
    assert!((sorted_diag[0].re - 1.0).abs() < TOL);
    assert!((sorted_diag[1].re - 3.0).abs() < TOL);
    assert!((sorted_diag[2].re - 5.0).abs() < TOL);
    println!("  Verified: eigenvalues match diagonal entries");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main() -> Result<(), SciOxideError> {
    println!();
    println!("scioxide-linalg Eigenvalue Examples");
    println!("==================================");
    println!();

    part1_symmetric()?;
    part2_general()?;

    println!("All examples completed successfully.");
    Ok(())
}
