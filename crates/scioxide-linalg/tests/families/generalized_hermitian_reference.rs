#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_linalg::{GeneralizedHermitianEigenDecomposition, Matrix};
include!("../data/generalized_hermitian.rs");

fn definite<S: NumericScalar>() {
    let c = |x| S::Real::try_from_f64_lossy(x).unwrap();
    let off = if S::IS_REAL {
        S::ONE
    } else {
        S::from_components(c(0.), c(1.))
    };
    let a = Matrix::from_vec(
        2,
        2,
        vec![S::from_real(c(3.)), off, off.conj(), S::from_real(c(3.))],
    )
    .unwrap();
    let b = Matrix::from_vec(2, 2, vec![S::from_real(c(2.)), S::ZERO, S::ZERO, S::ONE]).unwrap();
    let result = GeneralizedHermitianEigenDecomposition::decompose(
        &a,
        &b,
        scioxide_linalg::HermitianInput::Exact,
        scioxide_linalg::HermitianInput::Exact,
    )
    .unwrap();
    for (actual, reference) in result.eigenvalues().iter().zip(DEFINITE_EIGENVALUES) {
        assert!((*actual - c(reference)).abs() < c(64.) * <S::Real as RealScalar>::EPSILON);
    }
}
#[test]
fn definite_real32() {
    definite::<f32>();
}
#[test]
fn definite_real64() {
    definite::<f64>();
}
#[test]
fn definite_complex32() {
    definite::<Complex<f32>>();
}
#[test]
fn definite_complex64() {
    definite::<Complex<f64>>();
}
