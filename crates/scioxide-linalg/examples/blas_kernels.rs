//! Checked dense and compact BLAS-level operations.
//! Run with `cargo run -p scioxide-linalg --example blas_kernels`.

use scioxide_core::{Complex64, SciOxideError};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{
    BandedMatrix, Matrix, MatrixSide, PackedKind, PackedMatrix, StructuredOperation,
    TriangularPlan, Vector, givens_rotation,
};

pub fn main() -> Result<(), SciOxideError> {
    let values = Vector::from_vec(vec![3.0_f64, -4.0, 2.0]);
    println!(
        "absolute sum={}, first maximum={:?}",
        values.view().asum()?,
        values.view().iamax()?
    );
    let generated = givens_rotation(3.0_f64, 4.0)?;
    println!(
        "Givens cosine={}, sine={}",
        generated.rotation().cosine(),
        generated.rotation().sine()
    );

    // Column-major general band, including padding: A = [[1, 2, 0], [3, 4, 5]].
    let band = BandedMatrix::new(
        2,
        3,
        1,
        1,
        3,
        vec![0.0, 1.0, 3.0, 2.0, 4.0, 0.0, 5.0, 0.0, 0.0],
    )?;
    let band_result = band.view().mul_vec(
        StructuredOperation::Normal,
        &Vector::from_vec(vec![7.0, 11.0, 13.0]),
    )?;
    println!("band product={:?}", band_result.as_slice());

    let symmetric = Matrix::from_vec(2, 2, vec![2.0, 0.0, 1.0, 3.0])?;
    let general = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
    let product =
        symmetric
            .view()
            .symmetric_mul_mat(&general, MatrixSide::Left, Triangle::Lower)?;
    println!("symmetric product={:?}", product.as_slice());

    let packed = PackedMatrix::new(
        2,
        Triangle::Lower,
        PackedKind::Triangular,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 0.0),
        ],
    )?;
    let plan = TriangularPlan::from_packed(packed.view(), DiagonalKind::Explicit)?;
    let rhs = Vector::from_vec(vec![Complex64::new(4.0, 0.0), Complex64::new(8.0, 2.0)]);
    let solution = plan.solve(StructuredOperation::Normal, &rhs)?;
    println!("packed triangular solution={:?}", solution.as_slice());
    Ok(())
}
