use super::*;
use crate::SparseOperation;
use crate::factor_arithmetic::Arithmetic;
use scioxide_core::ExecutionContext;

crate::factor_solve::impl_solves!(IncompleteFactor);

impl<S: NumericScalar> IncompleteFactor<S> {
    // One preflight-validated column solve of `system`.
    fn solve_column(
        &self,
        rhs: &[S],
        output: &mut [S],
        work: &mut [S],
        system: SparseOperation,
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        work.copy_from_slice(rhs);
        let chol = self.symbolic.method.cholesky();
        match (chol, system) {
            (false, SparseOperation::Normal) => {
                self.triangle(work, (true, false, false), arithmetic, context)?;
                self.triangle(work, (false, false, false), arithmetic, context)?;
            }
            (false, op) => {
                let conj = op == SparseOperation::Adjoint;
                self.triangle(work, (false, true, conj), arithmetic, context)?;
                self.triangle(work, (true, true, conj), arithmetic, context)?;
            }
            (true, op) => {
                let conj = op == SparseOperation::Transpose;
                self.triangle(work, (true, false, conj), arithmetic, context)?;
                self.triangle(work, (true, true, !conj), arithmetic, context)?;
            }
        }
        output.copy_from_slice(work);
        Ok(())
    }
    // One block-triangular sweep.
    fn triangle(
        &self,
        x: &mut [S],
        (lower, trans, conj): (bool, bool, bool),
        arithmetic: Arithmetic,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.symbolic.rows();
        let b = self.symbolic.block_size;
        let b2 = b * b;
        let chol = self.symbolic.method.cholesky();
        let d = &self.data;
        for step in 0..n {
            context.check()?;
            let i = if lower != trans { step } else { n - 1 - step };
            if !trans {
                for pos in d.ptr[i]..d.ptr[i + 1] {
                    let j = d.col[pos];
                    if (lower && j < i) || (!lower && j > i) {
                        let a = d.block(pos, b2);
                        for r in 0..b {
                            x[i * b + r] = arithmetic.subtract_products(
                                x[i * b + r],
                                (0..b).map(|c| {
                                    let v = a[r * b + c];
                                    (if conj { v.conj() } else { v }, x[j * b + c])
                                }),
                            )?;
                        }
                    }
                }
            }
            if chol || !lower {
                block::solve(
                    &d.diagonal[i * b2..(i + 1) * b2],
                    &d.pivots[i * b..(i + 1) * b],
                    &mut x[i * b..(i + 1) * b],
                    b,
                    chol,
                    trans,
                    conj,
                    arithmetic,
                )?;
            }
            if trans {
                for pos in d.ptr[i]..d.ptr[i + 1] {
                    let j = d.col[pos];
                    if (lower && j < i) || (!lower && j > i) {
                        let a = d.block(pos, b2);
                        for c in 0..b {
                            x[j * b + c] = arithmetic.subtract_products(
                                x[j * b + c],
                                (0..b).map(|r| {
                                    let v = a[r * b + c];
                                    (if conj { v.conj() } else { v }, x[i * b + r])
                                }),
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

crate::factor_solve::impl_preconditioner! {
    IncompleteFactor<S>, |factor| &factor.oriented(SparseOperation::Normal)
}

impl<S: NumericScalar> IncompleteFactor<S> {
    fn solve_limits(&self) -> Result<(usize, usize), SciOxideError> {
        Ok((
            self.symbolic.workspace_requirements::<S>(())?.total_bytes(),
            self.symbolic.limits.max_workspace_bytes,
        ))
    }
}

impl<S: NumericScalar> IncompleteFactor<S> {
    /// Scalar dimension of the factored system.
    pub fn dimension(&self) -> usize {
        self.symbolic.dimension()
    }
}
