#![allow(dead_code, clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{BlockVectorView, ExecutionContext, NumericScalar, RealScalar};
use scioxide_sparse::*;
pub fn scalar<S: NumericScalar>(v: f64) -> S {
    S::from_real(S::Real::try_from_f64_lossy(v).unwrap())
}
pub fn grid<S: NumericScalar>(side: usize, dimensions: usize, anisotropy: f64) -> CsrMatrix<S> {
    assert!((1..=3).contains(&dimensions));
    let n = side.pow(dimensions as u32);
    let mut a = SparseBuilder::new(n, n);
    for i in 0..n {
        a.push(i, i, scalar(2. + 2. * anisotropy * (dimensions - 1) as f64))
            .unwrap();
        let mut stride = 1;
        for axis in 0..dimensions {
            let weight = if axis == 0 { 1. } else { anisotropy };
            let coordinate = (i / stride) % side;
            if coordinate > 0 {
                a.push(i, i - stride, scalar(-weight)).unwrap();
            }
            if coordinate + 1 < side {
                a.push(i, i + stride, scalar(-weight)).unwrap();
            }
            stride *= side;
        }
    }
    a.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
pub fn limits() -> AmgLimits {
    AmgLimits::new(16, 4_000_000, 2 * 1024 * 1024 * 1024)
}
pub fn coarse_limits() -> SparseSymmetricLimits {
    SparseSymmetricLimits::new(1024, 0, 128 * 1024 * 1024)
}
pub fn hierarchy<S: NumericScalar>(a: &CsrMatrix<S>) -> AmgHierarchy<S> {
    let modes = vec![S::ONE; a.rows()];
    AmgHierarchy::build(
        a.view(),
        BlockVectorView::new(&modes, a.rows(), 1).unwrap(),
        AmgSetupOptions::new().with_coarse_size(4),
        limits(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap()
}
pub fn preconditioner<S: NumericScalar>(
    a: &CsrMatrix<S>,
    kind: AmgCycleKind,
) -> AmgPreconditioner<S> {
    AmgPreconditioner::build(
        hierarchy(a),
        AmgCycleOptions::new().with_kind(kind),
        coarse_limits(),
        SparseSolveOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap()
}
pub fn close<S: NumericScalar>(a: &[S], b: &[S], factor: f64) {
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            (a - b).abs()
                <= S::Real::EPSILON
                    * S::Real::try_from_f64_lossy(factor).unwrap()
                    * (S::Real::ONE + b.abs()),
            "{a:?} != {b:?}"
        );
    }
}
