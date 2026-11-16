//! Backward-tested generalized adjacent-block exchanges.
use super::qz_kernel::Pencil;
use super::swap::{apply_basis_left, apply_basis_right, graph_basis, rejected};
use crate::kernel::frames::extended;
use crate::kernel::small::{Singularity, small};
use crate::workspace::storage::MatrixStorage;
use core::ops::Neg;
use scioxide_core::{NumericScalar, ScaledValue, SciOxideError};
pub(super) fn swap<S: NumericScalar>(
    pencil: &mut Pencil<'_, S>,
    infinite: &[usize],
    start: usize,
    p: usize,
    q: usize,
) -> Result<(), SciOxideError> {
    // Generalized analogue of the standard block swap: solve
    // the coupled Sylvester system `A11 R - L A22 = -A12`,
    // `B11 R - L B22 = -B12` (order 2pq <= 8). The graphs `[R; I]` and
    // `[L; I]` span the right and left deflating subspaces of the second
    // block, and orthonormal bases V and U for them give
    // `U^H (A, B) V` with the blocks exchanged. The same weak (zeroed
    // (2,1) blocks) and strong (reconstruction) backward-error tests as in
    // the standard case guard the exchange, and exact zero betas of
    // infinite eigenvalues are preserved explicitly.
    let d = p + q;
    let count = p * q;
    let mut coefficients = crate::kernel::generalized_sylvester::coefficients(
        &pencil.s,
        &pencil.t,
        start,
        start + p,
        p,
        q,
        false,
    )?;
    let mut rhs = [ScaledValue::ZERO; 8];
    for (offset, a) in [(0, &pencil.s), (count, &pencil.t)] {
        for i in 0..p {
            for j in 0..q {
                rhs[offset + i * q + j] = extended(a.get_element(start + i, start + p + j))?.neg();
            }
        }
    }
    let uncoupled = rhs[..2 * count].iter().all(|v| v.is_zero());
    if !small(&mut coefficients, &mut rhs, 2 * count, Singularity::Reject)? {
        if !uncoupled {
            return Err(rejected());
        }
        rhs.fill(ScaledValue::ZERO);
    }
    let v = graph_basis::<S>(&rhs[..count], p, q)?;
    let u = graph_basis::<S>(&rhs[count..2 * count], p, q)?;
    let mut trials = [[S::ZERO; 16]; 2];
    for (index, a) in [&pencil.s, &pencil.t].iter().enumerate() {
        trials[index] = super::swap::accept(
            a.try_view()?.block(start, start, d, d)?,
            &u,
            &v,
            |i, j| {
                let original = if i < q { start + p + i } else { start + i - q };
                (i >= q && j < q) || (index == 1 && i == j && infinite[original] != 0)
            },
            super::swap::QZ_ROUNDOFF,
        )?;
    }
    for a in [&mut pencil.s, &mut pencil.t] {
        apply_basis_right(a, start, d, &v)?;
        apply_basis_left(a, start, d, &u)?;
    }
    if let Some(left) = pencil.q.as_mut() {
        apply_basis_right(left, start, d, &u)?;
    }
    if let Some(right) = pencil.z.as_mut() {
        apply_basis_right(right, start, d, &v)?;
    }
    let n = pencil.s.rows();
    for (a, trial) in [&mut pencil.s, &mut pencil.t].into_iter().zip(trials) {
        for i in 0..d {
            for j in 0..d {
                a.as_mut_slice()[(start + i) * n + start + j] = trial[i * 4 + j];
            }
        }
    }
    // Each exchanged 2x2 diagonal pair needs its triangular B restored by QR.
    for (k, size) in [(start, q), (start + q, p)] {
        if size == 2 {
            let (g, _) = crate::rotation::Rotation::generate(
                pencil.t.get_element(k, k),
                pencil.t.get_element(k + 1, k),
            )?;
            // Rows `k, k + 1` of the quasi-triangular factors start at column `k`.
            super::qz_kernel::left(pencil, k, g, k..n, k..n)?;
            pencil.t.as_mut_slice()[(k + 1) * n + k] = S::ZERO;
        }
    }
    Ok(())
}
