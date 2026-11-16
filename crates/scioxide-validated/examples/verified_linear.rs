//! Ordinary dense and sparse LU factors followed by independent interval proofs.
use scioxide_core::{Complex, ExecutionContext, SciOxideError};
use scioxide_linalg::{LuDecomposition, Matrix};
use scioxide_sparse::{CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic};
use scioxide_validated::{
    DenseIntervalOperator, LinearCertificate, LinearVerifier, LuIntervalInverse,
    SparseIntervalOperator, ValidatedOptions, point_enclosure,
};
fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(2, 2, vec![3_f64, 1., 1., 2.])?;
    let answer = scioxide_validated::solve(&a, &[1.0, 1.0])?;
    assert!((answer.view().center[0] - 0.2).abs() < 1e-15);

    let options = ValidatedOptions::new();
    let factor = LuDecomposition::factor(&a)?;
    let system = DenseIntervalOperator::point(a.view())?;
    let inverse = LuIntervalInverse::from_dense(&factor, options)?;
    let plan = LinearVerifier::new(&system, &inverse, options)?;
    let rhs = [point_enclosure(1_f64)?, point_enclosure(1_f64)?];
    // The owned form publishes a certificate only when the proof succeeds.
    let (report, certificate) = plan.solve(&rhs)?;
    assert!(report.is_verified());
    if let Some(certificate) = certificate {
        assert!(plan.verify(&rhs, certificate.view())?.is_verified());
        println!("dense: {:?}; {:?}", report, certificate.enclosure()?);
    }

    let sparse = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            Complex::new(3_f64, 1.),
            Complex::new(1., 0.),
            Complex::new(1., 0.),
            Complex::new(2., -1.),
        ],
    )?;
    let symbolic = SparseLuSymbolic::analyze(
        sparse.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        SparseLuLimits::new(4, 65536),
    )?;
    let factor = SparseLu::factor(symbolic, sparse.view())?;
    let inverse = LuIntervalInverse::from_sparse(&factor, options)?;
    let system = SparseIntervalOperator::point(&sparse)?;
    let plan = LinearVerifier::new(&system, &inverse, options)?;
    // A reserved workspace makes repeated proofs allocation-free.
    let mut workspace = plan.workspace(1)?;
    let mut certificate = LinearCertificate::new(2)?;
    let rhs = [point_enclosure(Complex::new(1_f64, 0.))?; 2];
    let report = plan.solve_into(
        &rhs,
        certificate.view_mut(),
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(report.is_verified());
    println!(
        "sparse complex: {:?}; {:?}",
        report,
        certificate.enclosure()?
    );
    Ok(())
}
