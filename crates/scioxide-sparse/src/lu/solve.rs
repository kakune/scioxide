use super::*;
use crate::SparseOperation;
use crate::factor_arithmetic::Arithmetic;
use scioxide_core::ExecutionContext;

crate::factor_solve::impl_solves!(SparseLu);

impl<S: NumericScalar> SparseLu<S> {
    // Solves a preflight-validated column of `system` into staged output.
    fn solve_column(
        &self,
        rhs: &[S],
        output: &mut [S],
        work: &mut [S],
        system: SparseOperation,
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.symbolic.dimension();
        let d = &self.data;
        let q = self.symbolic.permutation.new_to_old();
        let conjugate = |v: S| {
            if system == SparseOperation::Adjoint {
                v.conj()
            } else {
                v
            }
        };
        match system {
            SparseOperation::Normal => {
                for (i, x) in work.iter_mut().enumerate() {
                    *x = rhs[d.p[i]];
                }
                for k in 0..n {
                    context.check()?;
                    for e in &d.entries[d.lower[k]..d.ptr[k + 1]] {
                        let row = d.inverse[e.row];
                        work[row] = arithmetic.subtract_product(work[row], e.value, work[k])?;
                    }
                }
                for k in (0..n).rev() {
                    context.check()?;
                    let diagonal = d.lower[k] - 1;
                    work[k] = arithmetic.divide(
                        work[k],
                        d.entries[diagonal].value,
                        crate::helpers::Projection::Gradual,
                    )?;
                    for e in &d.entries[d.ptr[k]..diagonal] {
                        work[e.row] = arithmetic.subtract_product(work[e.row], e.value, work[k])?;
                    }
                }
                for (i, &old) in q.iter().enumerate() {
                    output[old] = work[i];
                }
            }
            SparseOperation::Transpose | SparseOperation::Adjoint => {
                for (i, x) in work.iter_mut().enumerate() {
                    *x = rhs[q[i]];
                }
                for k in 0..n {
                    context.check()?;
                    let diagonal = d.lower[k] - 1;
                    for e in &d.entries[d.ptr[k]..diagonal] {
                        work[k] = arithmetic.subtract_product(
                            work[k],
                            conjugate(e.value),
                            work[e.row],
                        )?;
                    }
                    work[k] = arithmetic.divide(
                        work[k],
                        conjugate(d.entries[diagonal].value),
                        crate::helpers::Projection::Gradual,
                    )?;
                }
                for k in (0..n).rev() {
                    context.check()?;
                    for e in &d.entries[d.lower[k]..d.ptr[k + 1]] {
                        work[k] = arithmetic.subtract_product(
                            work[k],
                            conjugate(e.value),
                            work[d.inverse[e.row]],
                        )?;
                    }
                }
                for (i, &old) in d.p.iter().enumerate() {
                    output[old] = work[i];
                }
            }
        }
        Ok(())
    }
}

impl<S: NumericScalar> SparseLu<S> {
    fn solve_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((
            self.symbolic.workspace_requirements::<S>(())?.total_bytes(),
            self.symbolic.limits.max_workspace_bytes,
        ))
    }
}

impl<S: NumericScalar> SparseLu<S> {
    /// Returns the order `n` of the factored matrix.
    pub fn dimension(&self) -> usize {
        self.symbolic.dimension()
    }
}
