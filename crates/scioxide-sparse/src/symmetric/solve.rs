use super::numeric::{Data, block_solve};
use super::*;
use crate::SparseOperation;
use crate::factor_arithmetic::Arithmetic;
use scioxide_core::ExecutionContext;

crate::factor_solve::impl_solves!(SparseCholesky);
impl<S: NumericScalar> SparseCholesky<S> {
    fn solve_column(
        &self,
        rhs: &[S],
        output: &mut [S],
        work: &mut [S],
        system: SparseOperation,
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        solve_column(
            &self.data,
            (rhs, output),
            work,
            system,
            context,
            (true, arithmetic),
        )
    }
}
crate::factor_solve::impl_solves!(SparseLdl);
impl<S: NumericScalar> SparseLdl<S> {
    fn solve_column(
        &self,
        rhs: &[S],
        output: &mut [S],
        work: &mut [S],
        system: SparseOperation,
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        solve_column(
            &self.data,
            (rhs, output),
            work,
            system,
            context,
            (false, arithmetic),
        )
    }
}

// Permuted triangular and block-diagonal sweeps for a preflight-validated RHS.
fn solve_column<S: NumericScalar>(
    d: &Data<S>,
    (rhs, output): (&[S], &mut [S]),
    work: &mut [S],
    system: SparseOperation,
    context: ExecutionContext<'_>,
    (cholesky, arithmetic): (bool, Arithmetic),
) -> Result<(), SciOxideError> {
    let n = d.p.len();
    let transform = |v: S| {
        if system == SparseOperation::Transpose {
            v.conj()
        } else {
            v
        }
    };
    for (i, x) in work.iter_mut().enumerate() {
        *x = transform(rhs[d.p[i]]);
    }
    for k in 0..n {
        context.check()?;
        if cholesky {
            work[k] =
                arithmetic.divide(work[k], d.diagonal[k], crate::helpers::Projection::Strict)?;
        }
        for e in &d.entries[d.ptr[k]..d.ptr[k + 1]] {
            let row = d.inverse[e.row];
            work[row] = arithmetic.subtract_product(work[row], e.value, work[k])?;
        }
    }
    if !cholesky {
        for block in &d.blocks {
            context.check()?;
            let k = block.checked_range(n)?.start;
            if matches!(block, crate::LdlPivotBlock::Two { .. }) {
                let (x, y) = block_solve(
                    d.diagonal[k],
                    d.off[k],
                    d.diagonal[k + 1],
                    work[k],
                    work[k + 1],
                    arithmetic,
                )?;
                work[k] = x;
                work[k + 1] = y;
            } else {
                work[k] = arithmetic.divide(
                    work[k],
                    d.diagonal[k],
                    crate::helpers::Projection::Strict,
                )?;
            }
        }
    }
    for k in (0..n).rev() {
        context.check()?;
        for e in &d.entries[d.ptr[k]..d.ptr[k + 1]] {
            work[k] =
                arithmetic.subtract_product(work[k], e.value.conj(), work[d.inverse[e.row]])?;
        }
        if cholesky {
            work[k] =
                arithmetic.divide(work[k], d.diagonal[k], crate::helpers::Projection::Strict)?;
        }
    }
    for (i, &old) in d.p.iter().enumerate() {
        output[old] = transform(work[i]);
    }
    Ok(())
}

impl<S: NumericScalar> SparseCholesky<S> {
    fn solve_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((
            self.symbolic.workspace_requirements::<S>(())?.total_bytes(),
            self.symbolic.limits.max_workspace_bytes,
        ))
    }
}

impl<S: NumericScalar> SparseLdl<S> {
    fn solve_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((
            self.symbolic.workspace_requirements::<S>(())?.total_bytes(),
            self.symbolic.limits.max_workspace_bytes,
        ))
    }
}

impl<S: NumericScalar> SparseCholesky<S> {
    /// Returns the order `n` of the factored matrix.
    pub fn dimension(&self) -> usize {
        self.symbolic.dimension()
    }
}

impl<S: NumericScalar> SparseLdl<S> {
    /// Returns the order `n` of the factored matrix.
    pub fn dimension(&self) -> usize {
        self.symbolic.dimension()
    }
}
