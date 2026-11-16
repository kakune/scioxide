//! Adjacent block swaps from Sylvester invariant graphs and small Householder QR.
use super::swap::{apply_basis_left, apply_basis_right, graph_basis, rejected};
use crate::kernel::frames::extended;
use crate::kernel::sylvester::pair;
use crate::workspace::storage::MatrixStorage;
use core::ops::Neg;
use scioxide_core::{NumericScalar, ScaledValue, SciOxideError};
/// Work on a normalized Schur form. All trial matrices fit on the stack.
///
pub(crate) fn swap<S: NumericScalar, M: MatrixStorage<S>>(
    t: &mut M,
    qmat: &mut M,
    start: usize,
    p: usize,
    q: usize,
) -> Result<(), SciOxideError> {
    // Direct swap of adjacent diagonal blocks `[A11 A12; 0 A22]` (orders p, q
    // <= 2) by solving the small Sylvester
    // equation `A11 X - X A22 = -A12` (as a Kronecker system of order
    // p*q <= 4, in extended range). The columns of `[X; I]` span A22's
    // invariant subspace, so an orthonormal basis U for them, from a tiny
    // Householder QR, satisfies `U^H A U = [A22' *; 0 A11']`. A singular
    // system means the blocks share eigenvalues; that is only acceptable
    // when `A12 = 0` (already decoupled), in which case X = 0.
    let d = p + q;
    let mut rhs = [ScaledValue::ZERO; 4];
    for i in 0..p {
        for j in 0..q {
            rhs[i * q + j] = extended(t.get_element(start + i, start + p + j))?.neg();
        }
    }
    let uncoupled = rhs[..p * q].iter().all(|v| v.is_zero());
    if !pair(
        t.try_view()?.block(start, start, p, p)?,
        t.try_view()?.block(start + p, start + p, q, q)?,
        -S::ONE,
        &mut rhs[..p * q],
        false,
        || Ok(()),
    )? {
        if !uncoupled {
            return Err(rejected());
        }
        rhs.fill(ScaledValue::ZERO);
    }
    let u = graph_basis::<S>(&rhs, p, q)?;
    let trial = super::swap::accept(
        t.try_view()?.block(start, start, d, d)?,
        &u,
        &u,
        |i, j| i >= q && j < q,
        super::swap::SCHUR_ROUNDOFF,
    )?;
    apply_basis_right(t, start, d, &u)?;
    apply_basis_right(qmat, start, d, &u)?;
    apply_basis_left(t, start, d, &u)?;
    let n = t.rows();
    for i in 0..d {
        for j in 0..d {
            t.as_mut_slice()[(start + i) * n + start + j] = trial[i * 4 + j];
        }
    }
    Ok(())
}
