//! Dense vector and matrix basics for scioxide-linalg.
//!
//! Demonstrates core operations on [`Vector`] and [`Matrix`]:
//!
//! 1. **Vectors** — creation, element access, arithmetic, dot product, norms, cross product
//! 2. **Matrices** — creation, element access, row/column extraction, transpose
//! 3. **Matrix arithmetic** — addition, scalar multiplication, matrix-vector/matrix-matrix products
//! 4. **Determinant and rank** — computed via LU and SVD decompositions
//!
//! Run: `cargo run --example dense_basics -p scioxide-linalg`

use scioxide_core::SciOxideError;
use scioxide_linalg::{Matrix, Vector};

/// Tolerance for f64 floating-point comparisons.
const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Part 1: Vectors
// ---------------------------------------------------------------------------

fn part1_vectors() -> Result<(), SciOxideError> {
    println!("=== Part 1: Vector operations ===");
    println!();

    // Construction
    let v = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    let w = Vector::<f64>::from_vec(vec![4.0, 5.0, 6.0]);
    let z = Vector::<f64>::zeros(3)?;

    println!("  v = {:?}", v.as_slice());
    println!("  w = {:?}", w.as_slice());
    println!("  z = {:?}", z.as_slice());
    assert_eq!(v.len(), 3);
    assert_eq!(z.as_slice(), &[0.0, 0.0, 0.0]);

    // Element access
    let v0 = v.view().get(0)?;
    println!("  v[0] = {v0}");
    assert!((v0 - 1.0).abs() < TOL);

    // Arithmetic: v + w, v - w, scalar multiplication
    let sum = v.view().try_add(&w)?;
    let diff = v.view().try_sub(&w)?;
    let scaled = v.view().try_scale(2.0)?;
    println!("  v + w = {:?}", sum.as_slice());
    println!("  v - w = {:?}", diff.as_slice());
    println!("  2 * v = {:?}", scaled.as_slice());
    assert!((sum.view().get(0)? - 5.0).abs() < TOL);
    assert!((diff.view().get(0)? - (-3.0)).abs() < TOL);
    assert!((scaled.view().get(0)? - 2.0).abs() < TOL);

    // Dot product
    let dot = v.view().dotu(&w)?;
    // 1*4 + 2*5 + 3*6 = 32
    println!("  v · w = {dot}");
    assert!((dot - 32.0).abs() < TOL);

    // Norm and normalize
    let norm = v.view().norm2()?;
    // sqrt(1 + 4 + 9) = sqrt(14)
    let expected_norm = 14.0_f64.sqrt();
    println!("  ||v|| = {norm}");
    assert!((norm - expected_norm).abs() < TOL);

    let unit = v.view().normalize()?;
    let unit_norm = unit.view().norm2()?;
    println!("  ||normalize(v)|| = {unit_norm}");
    assert!((unit_norm - 1.0).abs() < TOL);

    // Cross product (3D)
    let cross = v.view().cross(&w)?;
    // [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    println!("  v × w = {:?}", cross.as_slice());
    assert!((cross.view().get(0)? - (-3.0)).abs() < TOL);
    assert!((cross.view().get(1)? - 6.0).abs() < TOL);
    assert!((cross.view().get(2)? - (-3.0)).abs() < TOL);

    println!();
    println!("  Part 1 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 2: Matrix construction and access
// ---------------------------------------------------------------------------

fn part2_matrices() -> Result<(), SciOxideError> {
    println!("=== Part 2: Matrix construction and access ===");
    println!();

    // Construction
    let m = Matrix::<f64>::zeros(2, 3)?; // 2×3 zero matrix
    let eye = Matrix::<f64>::identity(3)?; // 3×3 identity

    // Row-major data: [[1,2,3],[4,5,6]]
    let a = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    println!("  zero(2,3) data = {:?}", m.as_slice());
    println!(
        "  I(3) diag = {:?}",
        eye.view().try_diag()?.try_to_vector()?.as_slice()
    );
    println!("  A(2,3) data = {:?}", a.as_slice());

    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert!(eye.is_square());

    // Element access
    let a12 = a.view().get(0, 1)?;
    println!("  A[0,1] = {a12}");
    assert!((a12 - 2.0).abs() < TOL);

    // Row and column extraction
    let row0 = a.view().row(0)?;
    let col1 = a.view().col(1)?;
    println!("  A row 0 = {:?}", row0.as_contiguous_slice());
    println!("  A col 1 = {:?}", col1.iter().collect::<Vec<_>>());
    assert!((row0.get(2)? - 3.0).abs() < TOL);
    assert!((col1.get(1)? - 5.0).abs() < TOL);

    // Diagonal
    let sq = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;
    let diag = sq.view().try_diag()?;
    println!("  diag(3x3) = {:?}", diag.try_to_vector()?.as_slice());
    assert!((diag.get(0)? - 1.0).abs() < TOL);
    assert!((diag.get(1)? - 5.0).abs() < TOL);
    assert!((diag.get(2)? - 9.0).abs() < TOL);

    // Transpose
    let at = a.view().transpose().try_to_matrix()?;
    assert_eq!(at.rows(), 3);
    assert_eq!(at.cols(), 2);
    let at_01 = at.view().get(0, 1)?;
    // A^T[0,1] = A[1,0] = 4.0
    println!("  A^T[0,1] = {at_01}");
    assert!((at_01 - 4.0).abs() < TOL);

    // Conjugate transpose (same as transpose for real matrices)
    let ah = a.view().adjoint().try_to_matrix()?;
    assert_eq!(ah.rows(), 3);
    assert_eq!(ah.cols(), 2);

    println!();
    println!("  Part 2 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 3: Matrix arithmetic and products
// ---------------------------------------------------------------------------

fn part3_arithmetic() -> Result<(), SciOxideError> {
    println!("=== Part 3: Matrix arithmetic and products ===");
    println!();

    let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
    let b = Matrix::<f64>::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0])?;

    // Addition
    let sum = a.view().try_add(&b)?;
    println!("  A + B = {:?}", sum.as_slice());
    assert!((sum.view().get(0, 0)? - 6.0).abs() < TOL);

    // Scalar multiplication
    let scaled = a.view().try_scale(3.0)?;
    println!("  3A = {:?}", scaled.as_slice());
    assert!((scaled.view().get(0, 0)? - 3.0).abs() < TOL);
    assert!((scaled.view().get(1, 1)? - 12.0).abs() < TOL);

    // Matrix-vector product
    let x = Vector::<f64>::from_vec(vec![1.0, 2.0]);
    let ax = a.view().mul_vec(&x)?;
    // [1*1+2*2, 3*1+4*2] = [5, 11]
    println!("  A * x = {:?}", ax.as_slice());
    assert!((ax.view().get(0)? - 5.0).abs() < TOL);
    assert!((ax.view().get(1)? - 11.0).abs() < TOL);

    // Matrix-matrix product
    let ab = a.view().mul_mat(&b)?;
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    println!("  A * B = {:?}", ab.as_slice());
    assert!((ab.view().get(0, 0)? - 19.0).abs() < TOL);
    assert!((ab.view().get(0, 1)? - 22.0).abs() < TOL);
    assert!((ab.view().get(1, 0)? - 43.0).abs() < TOL);
    assert!((ab.view().get(1, 1)? - 50.0).abs() < TOL);

    println!();
    println!("  Part 3 assertions passed.");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Part 4: Determinant and rank
// ---------------------------------------------------------------------------

fn part4_det_rank() -> Result<(), SciOxideError> {
    println!("=== Part 4: Determinant and rank ===");
    println!();

    let a = Matrix::<f64>::from_vec(3, 3, vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0])?;

    // Determinant (via LU)
    let det = scioxide_linalg::LuDecomposition::factor(a.view())?.determinant()?;
    // By cofactor expansion along row 3:
    //   1*(1*2-1*3) - 0 + 0 = 1*(-1) = -1
    println!("  det(A) = {det}");
    assert!((det - (-1.0)).abs() < TOL);

    // Rank (via SVD)
    let rank = scioxide_linalg::SvdValues::compute(a.view())?
        .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
        .rank();
    println!("  rank(A) = {rank}");
    assert_eq!(rank, 3);

    // Singular matrix — rank deficient
    let singular = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0])?;
    let singular_rank = scioxide_linalg::SvdValues::compute(singular.view())?
        .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
        .rank();
    println!("  rank([[1,2],[2,4]]) = {singular_rank}");
    assert_eq!(singular_rank, 1);

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
    println!("scioxide-linalg Dense Basics Examples");
    println!("====================================");
    println!();

    part1_vectors()?;
    part2_matrices()?;
    part3_arithmetic()?;
    part4_det_rank()?;

    println!("All examples completed successfully.");
    Ok(())
}
