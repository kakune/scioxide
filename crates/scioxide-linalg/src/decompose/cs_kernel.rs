//! Two-block Jacobi rotations and orthogonal basis completion.

pub(super) struct LeftFactorScratch<'a, S: NumericScalar> {
    pub(super) norms: &'a [S::Real],
    pub(super) order: &'a mut [usize],
    pub(super) vector: &'a mut [S],
}
use crate::workspace::storage::MatrixStorage;
use crate::{KernelContext, LinalgExecutionOptions, LinalgWorkspace};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{NumericScalar, NumericalIssue, RealScalar, SciOxideError, SolveContext};

/// Scratch of [`prerotate`]: room for the smaller row block of `[C; S]` and
/// for a copy of `[C; S]`.
pub(super) struct PrerotationScratch<'a, M> {
    pub(super) block: &'a mut M,
    pub(super) product: &'a mut M,
}

/// Rotates the columns of `[C; S]` (rows `0..split` and `split..`) and of `z`
/// by the right singular vectors `W` of the block with fewer rows.
///
/// Since `CᴴC + SᴴS = I`, `W` diagonalizes both Gram matrices, so `C W` and
/// `S W` have orthogonal columns up to roundoff, and the Jacobi sweeps only
/// polish them. Without this step the sweeps converge linearly when the
/// smaller block has fewer rows than columns: its Gram matrix then has a
/// multiple zero eigenvalue. The SVD runs in a child scope of `solve`.
#[allow(clippy::too_many_arguments)]
pub(super) fn prerotate<'capability, S: NumericScalar, M: MatrixStorage<S>>(
    cs: &mut impl MatrixStorage<S>,
    z: &mut impl MatrixStorage<S>,
    split: usize,
    scratch: PrerotationScratch<'_, M>,
    options: LinalgExecutionOptions,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'capability>,
    solve: &mut SolveContext<'_, 'capability>,
) -> Result<(), SciOxideError> {
    let PrerotationScratch { block, product } = scratch;
    let (rows, rank) = (cs.rows(), cs.cols());
    let (start, height) = if split <= rows - split {
        (0, split)
    } else {
        (split, rows - split)
    };
    if rank < 2 || height == 0 {
        return Ok(());
    }
    block.try_resize_flat(height, rank)?;
    for i in 0..height {
        for j in 0..rank {
            block.set_element(i, j, cs.get_element(start + i, j));
        }
    }
    let finite = crate::FiniteMatrixView::from_workspace(block.try_view()?);
    let rotate = |spectral: super::SpectralView<'_, S>| {
        multiply_adjoint(z, spectral.vt, product, context)?;
        multiply_adjoint(cs, spectral.vt, product, context)
    };
    // A tall block has all `rank` right singular vectors in the thin SVD.
    let scoped = context.scoped(solve);
    if height >= rank {
        super::svd::thin_candidate(finite, options, workspace, scoped, rotate)
    } else {
        super::svd::full_candidate(finite, options, workspace, scoped, rotate)
    }
}

/// Replaces `target` by `target · Yᴴ` for a square `Y`, through `product`.
fn multiply_adjoint<S: NumericScalar>(
    target: &mut impl MatrixStorage<S>,
    y: crate::MatrixView<'_, S>,
    product: &mut impl MatrixStorage<S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let (rows, order) = (target.rows(), target.cols());
    product.try_resize_flat(rows, order)?;
    for i in 0..rows {
        context.execution().check()?;
        for k in 0..order {
            let value = (0..order).fold(S::ZERO, |sum, j| {
                sum + target.get_element(i, j) * y.get_element(k, j).conj()
            });
            product.set_element(i, k, value);
        }
    }
    for i in 0..rows {
        for k in 0..order {
            target.set_element(i, k, product.get_element(i, k));
        }
    }
    Ok(())
}

pub(super) fn identity<S: NumericScalar>(
    a: &mut impl MatrixStorage<S>,
    n: usize,
) -> Result<(), SciOxideError> {
    a.try_resize_flat(n, n)?;
    a.as_mut_slice().fill(S::ZERO);
    for i in 0..n {
        a.set_element(i, i, S::ONE);
    }
    Ok(())
}
pub(super) fn norm<S: NumericScalar>(
    a: &impl MatrixStorage<S>,
    col: usize,
    start: usize,
    end: usize,
) -> Result<S::Real, SciOxideError> {
    let view = a.try_view()?.block(start, col, end - start, 1)?;
    crate::kernel::standard::norm_workspace(view.col(0)?.iter())
}

// Diagonalize the locally smaller CS block. Complementary norms, rather than
// sqrt(1-c*c), retain small sine/cosine values near the endpoints.
// This is one-sided (Hestenes) Jacobi on the stacked `[C; S]`: because its
// columns are orthonormal, the cross products of a column pair in the two
// blocks are negatives of each other, so one rotation (also applied to the
// right factor `z`) orthogonalizes the pair in both blocks. Measuring it in
// the block with the smaller norms keeps its absolute error smallest. The
// rotation angle is the classical `tan` formula in its cancellation-free
// `2|x| / (|delta| + hypot(delta, 2|x|))` form.
pub(super) fn cs_sweep<S: NumericScalar>(
    cs: &mut impl MatrixStorage<S>,
    z: &mut impl MatrixStorage<S>,
    m: usize,
    tolerance: S::Real,
    context: KernelContext<'_>,
) -> Result<bool, SciOxideError> {
    let two = S::Real::TWO;
    let mut converged = true;
    for i in 0..cs.cols() {
        context.execution().check()?;
        for j in i + 1..cs.cols() {
            let ai = norm(cs, i, 0, m)?;
            let aj = norm(cs, j, 0, m)?;
            let bi = norm(cs, i, m, cs.rows())?;
            let bj = norm(cs, j, m, cs.rows())?;
            let (start, end, x, y) = if ai.hypot(aj) <= bi.hypot(bj) {
                (0, m, ai, aj)
            } else {
                (m, cs.rows(), bi, bj)
            };
            let cross = (start..end).fold(S::ZERO, |sum, k| {
                sum + cs.get_element(k, i).conj() * cs.get_element(k, j)
            });
            let magnitude = cross.abs();
            if magnitude <= tolerance * (x * x + y * y) || magnitude == S::Real::ZERO {
                continue;
            }
            // At the absolute roundoff floor the two blocks need not agree.
            if magnitude <= tolerance * tolerance {
                continue;
            }
            converged = false;
            let delta = (y - x) * (y + x);
            let (c, s) = crate::rotation::jacobi(delta, two * magnitude);
            let phase = cross / S::from_real(magnitude);
            let ss = phase * S::from_real(s);
            let extent = cs.rows();
            crate::rotation::Rotation { c, s: -ss.conj() }.apply(
                cs.try_view_mut()?.transpose(),
                i,
                j,
                0..extent,
            )?;
            let extent = z.rows();
            crate::rotation::Rotation { c, s: -ss.conj() }.apply(
                z.try_view_mut()?.transpose(),
                i,
                j,
                0..extent,
            )?;
        }
    }
    Ok(converged)
}

// Complete each left basis in decreasing column-norm order. Reorthogonalizing
// small columns avoids magnifying absolute CS roundoff into nonunitary factors.

pub(super) fn left_factor<S: NumericScalar>(
    cs: &impl MatrixStorage<S>,
    start: usize,
    rows: usize,
    scratch: crate::decompose::cs_kernel::LeftFactorScratch<'_, S>,
    u: &mut impl MatrixStorage<S>,
    d: &mut impl MatrixStorage<S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let crate::decompose::cs_kernel::LeftFactorScratch {
        norms,
        order,
        vector,
    } = scratch;

    let rank = cs.cols();
    u.try_resize_flat(rows, rows)?;
    u.as_mut_slice().fill(S::ZERO);
    d.try_resize_flat(rows, rank)?;
    d.as_mut_slice().fill(S::ZERO);
    let order = &mut order[..rank];
    for (i, v) in order.iter_mut().enumerate() {
        *v = i;
    }
    order.sort_unstable_by(|&i, &j| {
        norms[j]
            .partial_cmp(&norms[i])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    let vector = &mut vector[..rows];
    let mut used = 0;
    let floor = crate::checked::dimension_epsilon::<S::Real>(rows.max(1))?;
    for &col in order.iter() {
        if used == rows || norms[col] == S::Real::ZERO {
            continue;
        }
        context.execution().check()?;
        for (i, v) in vector.iter_mut().enumerate() {
            *v = cs.get_element(start + i, col) / S::from_real(norms[col]);
        }
        orthogonalize(vector, u, used);
        let length = crate::kernel::finite::norm(vector)?;
        if length <= floor {
            continue;
        }
        for (i, &v) in vector.iter().enumerate() {
            u.set_element(i, used, v / S::from_real(length));
        }
        d.set_element(used, col, S::from_real(norms[col]));
        used += 1;
    }
    for axis in 0..rows {
        if used == rows {
            break;
        }
        vector.fill(S::ZERO);
        vector[axis] = S::ONE;
        orthogonalize(vector, u, used);
        let length = crate::kernel::finite::norm(vector)?;
        if length <= floor {
            continue;
        }
        for (i, &v) in vector.iter().enumerate() {
            u.set_element(i, used, v / S::from_real(length));
        }
        used += 1;
    }
    if used != rows {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "left unitary completion failed",
        ));
    }
    Ok(())
}
pub(super) fn orthogonalize<S: NumericScalar>(
    vector: &mut [S],
    u: &impl MatrixStorage<S>,
    used: usize,
) {
    for _ in 0..2 {
        for j in 0..used {
            let dot = vector
                .iter()
                .enumerate()
                .fold(S::ZERO, |sum, (i, &x)| sum + u.get_element(i, j).conj() * x);
            for (i, x) in vector.iter_mut().enumerate() {
                *x = *x - u.get_element(i, j) * dot;
            }
        }
    }
}
