//! Independent Dirichlet eigenmode data shared by examples.
#![allow(dead_code, clippy::unwrap_used, clippy::cast_precision_loss)]
extern crate alloc;
use super::model::real;
use alloc::vec::Vec;
use scioxide_core::{
    AdjointOperator, LinearOperator, NumericScalar, SciOxideError, StructuralPattern,
};
use scioxide_sparse::CsrMatrix;
/// Independent uniform Dirichlet difference Laplacian in integrated units.
pub struct Laplacian<T: NumericScalar> {
    n: usize,
    h: T,
}
impl<T: NumericScalar> LinearOperator<T> for Laplacian<T> {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[T], out: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (T::from_real(real(2.)) * x[i]
                - if i > 0 { x[i - 1] } else { T::ZERO }
                - if i + 1 < self.n { x[i + 1] } else { T::ZERO })
                / self.h;
        }
        Ok(())
    }
}
impl<T: NumericScalar> AdjointOperator<T> for Laplacian<T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, out, scratch)
    }
}
pub type Fixture<T> = (
    Laplacian<T>,
    CsrMatrix<T>,
    Vec<<T as NumericScalar>::Real>,
    Vec<T>,
    Vec<T>,
    StructuralPattern,
);
pub fn fixture<T: NumericScalar>(n: usize) -> Fixture<T> {
    let h = 1. / (n + 1) as f64;
    let a = Laplacian {
        n,
        h: T::from_real(real(h)),
    };
    let mut coo = scioxide_sparse::SparseBuilder::new(n, n);
    for i in 0..n {
        coo.push(i, i, T::from_real(real(2. / h))).unwrap();
        if i > 0 {
            coo.push(i, i - 1, T::from_real(real(-1. / h))).unwrap();
        }
        if i + 1 < n {
            coo.push(i, i + 1, T::from_real(real(-1. / h))).unwrap();
        }
    }
    let csr = coo
        .finish_csr(
            scioxide_sparse::DuplicatePolicy::Sum,
            scioxide_sparse::ExplicitZeroPolicy::Drop,
        )
        .unwrap();
    let target: Vec<_> = (0..2 * n)
        .map(|k| {
            let q = (core::f64::consts::PI * (k % n + 1) as f64 * h).sin();
            T::from_components(real(q * if k < n { 0.8 } else { 0.5 }), real(q * 0.1))
        })
        .collect();
    let lambda = 4. * (core::f64::consts::PI * h / 2.).sin().powi(2) / (h * h);
    let laplace = target
        .iter()
        .map(|&q| q * T::from_real(real(lambda)))
        .collect();
    let pattern = block_pattern(&csr);
    (a, csr, vec![real(h); n], target, laplace, pattern)
}
pub fn block_pattern<T: NumericScalar>(a: &CsrMatrix<T>) -> StructuralPattern {
    let n = a.rows();
    let mut rows = vec![0];
    let mut columns = Vec::new();
    for row in 0..2 * n {
        let i = row % n;
        let mut block: Vec<_> = a.col_indices()[a.row_ptr()[i]..a.row_ptr()[i + 1]]
            .iter()
            .map(|&k| row / n * n + k)
            .collect();
        block.push((1 - row / n) * n + i);
        block.push(row);
        block.sort_unstable();
        block.dedup();
        columns.extend(block);
        rows.push(columns.len());
    }
    StructuralPattern::new(2 * n, 2 * n, rows, columns).unwrap()
}
