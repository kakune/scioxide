//! Direct solver examples for scioxide-linalg.
//!
//! Demonstrates the supported matrix decompositions for solving linear systems:
//!
//! 1. **LU** — general square system, range-safe determinant, inverse
//! 2. **Householder QR/LQ** — square, least-squares, and minimum-norm systems
//! 3. **Cholesky** — symmetric positive definite systems
//! 4. **SVD** — singular values, rank estimation, reciprocal condition number
//!
//! Run: `cargo run --example direct_solvers -p scioxide-linalg`

#![allow(clippy::many_single_char_names)]

use scioxide_core::SciOxideError;
use scioxide_linalg::{
    CholeskyDecomposition, HouseholderDecomposition, HouseholderOrientation, LuDecomposition,
    Matrix, RankPolicy, Vector,
};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Part 1: LU decomposition
// ---------------------------------------------------------------------------

fn part1_lu() -> Result<(), SciOxideError> {
    println!("=== Part 1: LU decomposition ===");
    println!();

    // Solve Ax = b where A = [[2,1,1],[4,3,3],[8,7,9]], b = [1,1,1]
    let a = Matrix::<f64>::from_vec(3, 3, vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0])?;
    let b = Vector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);

    let lu = LuDecomposition::factor(&a)?;

    // Solve
    let x = lu.solve(&b)?;
    println!("  A = [[2,1,1],[4,3,3],[8,7,9]]");
    println!("  b = [1, 1, 1]");
    println!("  x = {:?}", x.as_slice());

    // Verify: A*x ≈ b
    let ax = a.view().mul_vec(&x)?;
    for i in 0..3 {
        assert!(
            (ax.view().get(i)? - b.view().get(i)?).abs() < TOL,
            "A*x[{i}] != b[{i}]"
        );
    }
    println!("  Verified: A*x ≈ b");

    // Determinant: 2*(27-21) - 1*(36-24) + 1*(28-24) = 4
    let scaled_det = lu.scaled_determinant()?;
    let det = scaled_det.try_value()?;
    println!("  det(A) = {det}");
    assert!((det - 4.0).abs() < TOL);

    // Inverse
    let a_inv = lu.inverse()?;
    let should_be_identity = a.view().mul_mat(&a_inv)?;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (should_be_identity.view().get(i, j)? - expected).abs() < TOL,
                "A*A^{{-1}}[{i},{j}] != I"
            );
        }
    }
    println!("  Verified: A * A^{{-1}} ≈ I");

    // Extract L and U factors
    let l = lu.to_l_matrix()?;
    let u = lu.to_u_matrix()?;
    println!("  L rows={}, U rows={}", l.rows(), u.rows());

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: QR decomposition
// ---------------------------------------------------------------------------

fn part2_qr() -> Result<(), SciOxideError> {
    println!("=== Part 2: QR decomposition ===");
    println!();

    // Square system
    let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
    let b = Vector::<f64>::from_vec(vec![2.0, 2.0, 2.0]);

    let qr = HouseholderDecomposition::factor(
        &a,
        HouseholderOrientation::Qr,
        scioxide_linalg::HouseholderPivoting::Column,
    )?;
    let x = qr
        .solve_with_options(
            &b,
            &scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon),
        )?
        .into_accepted()?;
    println!("  Square system: x = {:?}", x.as_slice());

    // Verify A*x ≈ b
    let ax = a.view().mul_vec(&x)?;
    for i in 0..3 {
        assert!(
            (ax.view().get(i)? - b.view().get(i)?).abs() < TOL,
            "A*x[{i}] != b[{i}]"
        );
    }
    println!("  Verified: A*x ≈ b");

    // Q and R factors
    let q = qr.to_q_matrix()?;
    let r = qr.to_triangular_matrix()?;
    println!(
        "  Q is {}x{}, R is {}x{}",
        q.rows(),
        q.cols(),
        r.rows(),
        r.cols()
    );

    // Verify Q is orthogonal: Q^T * Q ≈ I
    let qtq = q.view().adjoint().try_to_matrix()?.view().mul_mat(&q)?;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq.view().get(i, j)? - expected).abs() < TOL,
                "Q^T*Q[{i},{j}] != I"
            );
        }
    }
    println!("  Verified: Q^T * Q ≈ I");

    // Overdetermined least-squares: 4x2 system
    let a_tall = Matrix::<f64>::from_vec(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])?;
    let b_tall = Vector::<f64>::from_vec(vec![2.0, 3.0, 5.0, 6.0]);

    let qr_tall = HouseholderDecomposition::factor(
        &a_tall,
        HouseholderOrientation::Qr,
        scioxide_linalg::HouseholderPivoting::Column,
    )?;
    let x_ls = qr_tall
        .solve_with_options(
            &b_tall,
            &scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon),
        )?
        .into_accepted()?;
    println!("  Least-squares solution: x = {:?}", x_ls.as_slice());

    let residual = a_tall.view().mul_vec(&x_ls)?;
    let orthogonality = a_tall
        .view()
        .adjoint()
        .try_to_matrix()?
        .view()
        .mul_vec(&residual.view().try_sub(&b_tall)?)?;
    assert!(orthogonality.view().norm2()? < TOL);
    println!("  Verified: least-squares residual is orthogonal to the column space");

    // Underdetermined minimum-norm solve through the same QR/LQ interface.
    let a_wide = Matrix::<f64>::from_vec(2, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
    let b_wide = Vector::<f64>::from_vec(vec![1.0, 1.0]);
    let lq = HouseholderDecomposition::factor(
        &a_wide,
        HouseholderOrientation::Lq,
        scioxide_linalg::HouseholderPivoting::Column,
    )?;
    let x_minimum_norm = lq
        .into_minimum_norm(
            &scioxide_linalg::DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon)
                .with_residual(scioxide_linalg::Estimation::Estimate),
        )?
        .solve(&b_wide)?
        .into_accepted()?;
    assert!((x_minimum_norm.view().get(0)? - 1.0 / 3.0).abs() < TOL);
    assert!((x_minimum_norm.view().get(1)? - 1.0 / 3.0).abs() < TOL);
    assert!((x_minimum_norm.view().get(2)? - 2.0 / 3.0).abs() < TOL);
    println!("  Verified: LQ returned the underdetermined minimum-norm solution");

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Cholesky decomposition
// ---------------------------------------------------------------------------

fn part3_cholesky() -> Result<(), SciOxideError> {
    println!("=== Part 3: Cholesky decomposition ===");
    println!();

    // Symmetric positive definite matrix
    let a = Matrix::<f64>::from_vec(3, 3, vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0])?;
    let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    let chol = CholeskyDecomposition::factor(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::CholeskyPivoting::None,
    )?;
    let x = chol.solve(&b)?;
    println!("  SPD system: x = {:?}", x.as_slice());

    // Verify A*x ≈ b
    let ax = a.view().mul_vec(&x)?;
    for i in 0..3 {
        assert!(
            (ax.view().get(i)? - b.view().get(i)?).abs() < TOL,
            "A*x[{i}] != b[{i}]"
        );
    }
    println!("  Verified: A*x ≈ b");

    // Extract L factor and verify A ≈ L * L^T
    let l = chol.l_matrix();
    let lt = l.view().adjoint().try_to_matrix()?;
    let llt = l.view().mul_mat(&lt)?;
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (llt.view().get(i, j)? - a.view().get(i, j)?).abs() < TOL,
                "L*L^T[{i},{j}] != A"
            );
        }
    }
    println!("  Verified: L * L^T ≈ A");
    println!("  L is {}x{}", l.rows(), l.cols());

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: SVD
// ---------------------------------------------------------------------------

fn part4_svd() -> Result<(), SciOxideError> {
    println!("=== Part 4: SVD ===");
    println!();

    let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;

    let svd = scioxide_linalg::SvdDecomposition::compute(&a)?;

    // Singular values (descending)
    let sigma = svd.singular_values();
    println!("  Singular values: {sigma:?}");
    assert!((sigma[0] - 3.0).abs() < TOL);
    assert!((sigma[1] - 2.0).abs() < TOL);
    assert!((sigma[2] - 1.0).abs() < TOL);

    // Rank
    let rank = svd
        .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
        .rank();
    println!("  Rank: {rank}");
    assert_eq!(rank, 3);

    let reciprocal_condition = svd.condition_estimate()?.reciprocal().value();
    println!("  Reciprocal condition number: {reciprocal_condition}");
    assert!((reciprocal_condition - 1.0 / 3.0).abs() < TOL);

    // U and V^T factors
    let u = svd.u_matrix();
    let vt = svd.vt_matrix();
    println!(
        "  U is {}x{}, V^T is {}x{}",
        u.rows(),
        u.cols(),
        vt.rows(),
        vt.cols()
    );

    // Rank-deficient matrix
    let b = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0])?;
    let svd_b = scioxide_linalg::SvdValues::compute(&b)?;
    let rank_b = svd_b
        .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
        .rank();
    println!("  Rank of rank-1 matrix: {rank_b}");
    assert_eq!(rank_b, 1);

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
    println!("scioxide-linalg Direct Solver Examples");
    println!("=====================================");
    println!();

    part1_lu()?;
    part2_qr()?;
    part3_cholesky()?;
    part4_svd()?;
    println!("All examples completed successfully.");
    Ok(())
}
