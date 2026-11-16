//! Implicit-shift Golub-Kahan-Reinsch bidiagonal QR with vector rotations.

use crate::decompose::util::transpose_square_in_place;
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use scioxide_core::{
    AlgebraicScalar, ConvergenceDiagnostics, ConvergenceReason, NumericScalar, RealScalar,
    SciOxideError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BidiagonalCase {
    DeflateLast,
    Split,
    Sweep,
    Converged,
}

/// Direction in which a QR sweep chases its bulge through a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chase {
    /// From the top to the bottom, with the shift from the trailing 2 × 2.
    Down,
    /// From the bottom to the top, with the shift from the leading 2 × 2.
    Up,
}

#[expect(
    clippy::too_many_lines,
    reason = "the numerical recurrence keeps its coupled state updates together"
)]
pub(super) fn bidiagonal_qr<S: NumericScalar>(
    singular_values: &mut [S::Real],
    superdiagonal: &mut [S::Real],
    u: &mut ScratchMatrix<'_, S>,
    vt: &mut ScratchMatrix<'_, S>,
    iteration_budget: usize,
    solve: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    // Golub-Kahan-Reinsch implicit-shift QR on the real upper bidiagonal
    // `B(diagonal, superdiagonal)`, working on the trailing unreduced block
    // `start..active`. Each pass classifies the bottom of the matrix:
    // - `Converged`: the last superdiagonal is negligible, so `d[active-1]`
    //   is a singular value; it is made nonnegative and `active` shrinks.
    // - `DeflateLast` / `Split`: a negligible diagonal entry makes B
    //   singular; its superdiagonal neighbour is chased out with rotations
    //   (upwards through V, or rightwards through U) so B splits.
    // - `Sweep`: one shifted QR sweep on the unreduced block. A block
    //   disjoint from the previous swept one chases from its
    //   larger end diagonal entry towards the smaller one, so graded blocks
    //   converge at the small end in either orientation.
    // Negligibility is relative to the neighbouring entries
    // (`eps * (|d_i| + |d_{i+1}|)`), plus `tiny` so exact zeros and
    // underflowed values always deflate.
    // U and V start orthonormal. These bounded rotations cannot overflow;
    // callers validate complete candidates before publishing them.
    // Accumulate U^T so both vector factors receive contiguous row rotations.
    // DC leaves may supply a nonidentity U, so transpose at both boundaries.
    transpose_square_in_place(u);
    let epsilon = <S::Real as RealScalar>::EPSILON;
    let tiny = S::Real::MIN_POSITIVE;
    let mut active = singular_values.len();
    let mut iterations = 0usize;
    // The first and last index of the previous swept block, and its direction.
    let mut swept: Option<(usize, usize, Chase)> = None;

    while active > 0 {
        solve.execution().check()?;
        let mut split = active as isize - 2;
        while split >= 0 {
            let index = split as usize;
            let threshold =
                tiny + epsilon * (singular_values[index].abs() + singular_values[index + 1].abs());
            if superdiagonal[index].abs() <= threshold {
                superdiagonal[index] = S::Real::ZERO;
                break;
            }
            split -= 1;
        }

        let (case, start) = if split == active as isize - 2 {
            (BidiagonalCase::Converged, active - 1)
        } else {
            let lower = (split + 1) as usize;
            let negligible = (lower..active).rev().find(|&index| {
                let right = if index + 1 < active {
                    superdiagonal[index].abs()
                } else {
                    S::Real::ZERO
                };
                let left = if index > lower {
                    superdiagonal[index - 1].abs()
                } else {
                    S::Real::ZERO
                };
                singular_values[index].abs() <= tiny + epsilon * (left + right)
            });
            match negligible {
                None => (BidiagonalCase::Sweep, lower),
                Some(index) => {
                    singular_values[index] = S::Real::ZERO;
                    if index == active - 1 {
                        (BidiagonalCase::DeflateLast, lower)
                    } else {
                        // A zero at lower must be chased too: a QR sweep
                        // starts with g = d[lower] * e[lower] and stalls there.
                        (BidiagonalCase::Split, index + 1)
                    }
                }
            }
        };

        match case {
            BidiagonalCase::DeflateLast => {
                let mut bulge = superdiagonal[active - 2];
                superdiagonal[active - 2] = S::Real::ZERO;
                for index in (start..active - 1).rev() {
                    let (cosine, sine, norm) =
                        crate::rotation::Rotation::generate(singular_values[index], bulge)
                            .map(|(g, r)| (g.c, g.s, r))?;
                    singular_values[index] = norm;
                    if index != start {
                        bulge = -sine * superdiagonal[index - 1];
                        superdiagonal[index - 1] = cosine * superdiagonal[index - 1];
                    }
                    let extent = vt.cols();
                    crate::rotation::Rotation {
                        c: cosine,
                        s: S::from_real(sine),
                    }
                    .rotate_rows(
                        vt.as_mut_slice(),
                        extent,
                        index,
                        active - 1,
                        0..extent,
                    )?;
                }
            }
            BidiagonalCase::Split => {
                let anchor = start - 1;
                let mut bulge = superdiagonal[anchor];
                superdiagonal[anchor] = S::Real::ZERO;
                for index in start..active {
                    let (cosine, sine, norm) =
                        crate::rotation::Rotation::generate(singular_values[index], bulge)
                            .map(|(g, r)| (g.c, g.s, r))?;
                    singular_values[index] = norm;
                    bulge = -sine * superdiagonal[index];
                    superdiagonal[index] = cosine * superdiagonal[index];
                    let extent = u.rows();
                    crate::rotation::Rotation {
                        c: cosine,
                        s: -S::from_real(sine),
                    }
                    .rotate_rows(
                        u.as_mut_slice(),
                        extent,
                        anchor,
                        index,
                        0..extent,
                    )?;
                }
            }
            BidiagonalCase::Sweep => {
                if iterations >= iteration_budget {
                    let residual = superdiagonal[..active]
                        .iter()
                        .copied()
                        .map(scioxide_core::NumericScalar::abs)
                        .fold(S::Real::ZERO, RealScalar::max);
                    let residual_f64 = residual.try_to_f64_exact().unwrap_or(f64::INFINITY);
                    return Err(SciOxideError::convergence_failure(
                        ConvergenceReason::IterationLimit,
                        ConvergenceDiagnostics::new()
                            .with_counts(
                                (scioxide_core::EvaluationCounts::from_iterations(iterations))
                                    .into(),
                            )
                            .with_residual(scioxide_core::ResidualReport::unscaled(
                                residual_f64,
                                scioxide_core::NormSelection::Infinity,
                            )?),
                        "SVD bidiagonal QR iteration exhausted its budget",
                    ));
                }
                crate::solve_support::admit_iteration(
                    Some(solve.execution()),
                    Some(solve),
                    "no complete SVD candidate",
                )?;
                let last = active - 1;
                let chase = match swept {
                    Some((first, end, chase)) if start <= end && last >= first => chase,
                    _ if singular_values[start].abs() >= singular_values[last].abs() => Chase::Down,
                    _ => Chase::Up,
                };
                swept = Some((start, last, chase));
                qr_sweep(singular_values, superdiagonal, u, vt, start, active, chase)?;
                iterations += 1;
            }
            BidiagonalCase::Converged => {
                // Callers sort the values; here the sign only is fixed.
                if singular_values[start] < S::Real::ZERO {
                    singular_values[start] = -singular_values[start];
                    for column in 0..vt.cols() {
                        let offset = start * vt.cols() + column;
                        vt.as_mut_slice()[offset] = -vt.as_slice()[offset];
                    }
                }
                active -= 1;
            }
        }
    }
    transpose_square_in_place(u);
    Ok(())
}

/// One implicit-shift QR sweep on the block `start..active`.
///
/// [`Chase::Up`] runs the downward sweep on the reversed block
/// `B' = J Bᵀ J` (`d` and `e` reversed, `J` the exchange matrix): `B'` has the
/// singular vectors of `B` with left and right exchanged and their indices
/// reversed, so its right rotations act on rows of `U` and its left rotations on
/// rows of `Vᵀ`, at mirrored indices.
fn qr_sweep<S: NumericScalar>(
    singular_values: &mut [S::Real],
    superdiagonal: &mut [S::Real],
    u: &mut ScratchMatrix<'_, S>,
    vt: &mut ScratchMatrix<'_, S>,
    start: usize,
    active: usize,
    chase: Chase,
) -> Result<(), SciOxideError> {
    let reverse = |singular_values: &mut [S::Real], superdiagonal: &mut [S::Real]| {
        if chase == Chase::Up {
            singular_values[start..active].reverse();
            superdiagonal[start..active - 1].reverse();
        }
    };
    let (u_extent, vt_extent) = (u.cols(), vt.cols());
    let (u, vt) = (u.as_mut_slice(), vt.as_mut_slice());
    let (mut right, mut left) = match chase {
        Chase::Down => ((vt, vt_extent), (u, u_extent)),
        Chase::Up => ((u, u_extent), (vt, vt_extent)),
    };
    // Rotate rows `index` and `index + 1` of the block as swept, that is,
    // rows `m(index + 1)` and `m(index)` with `m(k) = start + active − 1 − k`
    // when the block is reversed; exchanging the rows negates the sine.
    let rotate = |(target, extent): &mut (&mut [S], usize),
                  index: usize,
                  cosine: S::Real,
                  sine: S::Real|
     -> Result<(), SciOxideError> {
        let (first, sine) = match chase {
            Chase::Down => (index, sine),
            Chase::Up => (start + active - 2 - index, -sine),
        };
        crate::rotation::Rotation {
            c: cosine,
            s: S::from_real(sine),
        }
        .rotate_rows(target, *extent, first, first + 1, 0..*extent)
    };
    reverse(singular_values, superdiagonal);
    let result = sweep_down(
        singular_values,
        superdiagonal,
        start,
        active,
        |side, index, c, s| match side {
            Side::Right => rotate(&mut right, index, c, s),
            Side::Left => rotate(&mut left, index, c, s),
        },
    );
    reverse(singular_values, superdiagonal);
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Right,
    Left,
}

fn sweep_down<R: RealScalar>(
    singular_values: &mut [R],
    superdiagonal: &mut [R],
    start: usize,
    active: usize,
    mut rotate: impl FnMut(Side, usize, R, R) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    // Wilkinson shift `mu = bottom^2 - shift`: the eigenvalue of the trailing
    // 2x2 block of `B B^T` closer to its bottom entry. `shift` is evaluated
    // as `off^2 / (middle + root)`, where `middle` and `root` have equal
    // signs, so there is no cancellation. The five entries involved are
    // divided by their maximum first so squaring cannot overflow or
    // underflow. `(f, g)` is the first column of `B^T B - mu I`; the sweep
    // chases the bulge it creates down the block with alternating right (V)
    // and left (U) rotations (implicit Q theorem).
    let scale = [
        singular_values[active - 1].abs(),
        singular_values[active - 2].abs(),
        superdiagonal[active - 2].abs(),
        singular_values[start].abs(),
        superdiagonal[start].abs(),
    ]
    .into_iter()
    .fold(R::ZERO, RealScalar::max);
    if scale == R::ZERO {
        superdiagonal[start] = R::ZERO;
        return Ok(());
    }

    let bottom = singular_values[active - 1] / scale;
    let prior = singular_values[active - 2] / scale;
    let bottom_superdiagonal = superdiagonal[active - 2] / scale;
    let top = singular_values[start] / scale;
    let top_superdiagonal = superdiagonal[start] / scale;
    let two = R::TWO;
    let middle =
        ((prior + bottom) * (prior - bottom) + bottom_superdiagonal * bottom_superdiagonal) / two;
    let product = bottom * bottom_superdiagonal;
    let square = product * product;
    let shift = if middle == R::ZERO && square == R::ZERO {
        R::ZERO
    } else {
        let root = middle.hypot(square.sqrt()).copysign(middle);
        square / (middle + root)
    };
    let mut f = (top + bottom) * (top - bottom) + shift;
    let mut g = top * top_superdiagonal;

    for index in start..active - 1 {
        let (right_cosine, right_sine, norm) =
            crate::rotation::Rotation::generate(f, g).map(|(g, r)| (g.c, g.s, r))?;
        if index != start {
            superdiagonal[index - 1] = norm;
        }
        f = right_cosine * singular_values[index] + right_sine * superdiagonal[index];
        superdiagonal[index] =
            right_cosine * superdiagonal[index] - right_sine * singular_values[index];
        g = right_sine * singular_values[index + 1];
        singular_values[index + 1] = right_cosine * singular_values[index + 1];
        rotate(Side::Right, index, right_cosine, right_sine)?;

        let (left_cosine, left_sine, norm) =
            crate::rotation::Rotation::generate(f, g).map(|(g, r)| (g.c, g.s, r))?;
        singular_values[index] = norm;
        f = left_cosine * superdiagonal[index] + left_sine * singular_values[index + 1];
        singular_values[index + 1] =
            -left_sine * superdiagonal[index] + left_cosine * singular_values[index + 1];
        g = left_sine * superdiagonal[index + 1];
        superdiagonal[index + 1] = left_cosine * superdiagonal[index + 1];
        rotate(Side::Left, index, left_cosine, left_sine)?;
    }
    superdiagonal[active - 2] = f;
    Ok(())
}
