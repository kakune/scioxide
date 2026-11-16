#![allow(clippy::unwrap_used)]
use scioxide_core::{NumericScalar, RealScalar};
use scioxide_linalg::{Estimation, Matrix, SchurDecomposition, SchurReorderOptions};
include!("../data/schur_reordering.rs");

fn blocks<R: RealScalar + NumericScalar<Real = R>>() {
    let c = |x| R::try_from_f64_lossy(x).unwrap();
    let a = Matrix::from_vec(
        4,
        4,
        [
            1., 2., 1., 2., -1., 1., 3., 4., 0., 0., 5., 3., 0., 0., -2., 5.,
        ]
        .map(c)
        .to_vec(),
    )
    .unwrap();
    let s = SchurDecomposition::decompose(&a).unwrap();
    let options = SchurReorderOptions::default().with_condition_estimates(Estimation::Estimate);
    let (_, report) = s
        .reorder_with_options(&[true, true, false, false], &options)
        .unwrap();
    let tol = c(256.) * <R as RealScalar>::EPSILON;
    assert!(
        (report.cluster_condition().unwrap().reciprocal().value() - c(BLOCK_CLUSTER_CONDITION))
            .abs()
            < tol
    );
    assert!(
        (report.separation().unwrap().value() - c(BLOCK_INVERSE_ONE_NORM_RECIPROCAL)).abs() < tol
    );
}
#[test]
fn block32() {
    blocks::<f32>();
}
#[test]
fn block64() {
    blocks::<f64>();
}
