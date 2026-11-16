use scioxide_core::AlgebraicScalar;
use scioxide_core::RealScalar;
use scioxide_core::{ConvergenceDiagnostics, ConvergenceReason, SciOxideError, Tolerance};

use crate::workspace::storage::MatrixStorage;

use scioxide_core::NumericScalar;

/// Reduce a complex upper-Hessenberg matrix to triangular Schur form, charging
/// every attempted QR step to `solve`.
///
/// The matrix and optional Schur-vector accumulator are operational scratch:
/// callers commit public results only after this function and their final
/// invariant checks succeed.
pub(super) fn schur_iteration<S: NumericScalar, M: MatrixStorage<S>>(
    hessenberg: &mut M,
    mut vectors: Option<&mut M>,
    tolerance: &Tolerance<S::Real>,
    iteration_budget: usize,
    solve: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    debug_assert!(!S::IS_REAL);
    let order = hessenberg.rows();
    if order < 2 {
        return Ok(());
    }

    // Complex single-shift QR with implicit Givens bulge chasing. Each outer
    // pass deflates negligible subdiagonals, isolates the trailing unreduced
    // block `start..=end`, and applies up to two steps using the two
    // eigenvalues of the trailing 2x2 block as shifts (the closer one to the
    // bottom entry first, i.e. a Wilkinson shift). `stagnant` counts passes
    // without deflation and triggers an exceptional shift every 20 of them.
    let mut end = order - 1;
    let mut steps = 0_usize;
    let mut stagnant = 0_usize;
    while end > 0 {
        solve.execution().check()?;
        early_deflation(hessenberg, end, tolerance);
        let previous_end = end;
        while end > 0 && hessenberg.get_element(end, end - 1) == S::ZERO {
            end -= 1;
        }
        if end == 0 {
            break;
        }

        stagnant = if end < previous_end { 0 } else { stagnant + 1 };
        let start = active_start(hessenberg, end);
        let (first_shift, second_shift) = trailing_shifts(hessenberg, end, stagnant)?;
        for shift in [first_shift, second_shift] {
            if steps >= iteration_budget {
                return Err(SciOxideError::convergence_failure(
                    ConvergenceReason::IterationLimit,
                    ConvergenceDiagnostics::new().with_counts(
                        (scioxide_core::EvaluationCounts::from_iterations(steps)).into(),
                    ),
                    "complex Schur iteration exhausted its budget",
                ));
            }
            crate::solve_support::admit_iteration(
                Some(solve.execution()),
                Some(&mut *solve),
                "solve iteration budget exhausted",
            )?;
            steps += 1;
            francis_shift_step(hessenberg, vectors.as_deref_mut(), start, end, shift)?;
            early_deflation(hessenberg, end, tolerance);
            if hessenberg.get_element(end, end - 1) == S::ZERO {
                break;
            }
        }
    }

    hessenberg
        .as_slice()
        .iter()
        .copied()
        .try_for_each(|value| {
            crate::checked::finite_candidate_with_detail(value, || {
                alloc::format!("{} exceeded the finite scalar range", "Schur form").into()
            })
            .map(|_| ())
        })?;
    if let Some(q) = vectors {
        q.as_slice().iter().copied().try_for_each(|value| {
            crate::checked::finite_candidate_with_detail(value, || {
                alloc::format!("{} exceeded the finite scalar range", "Schur vectors").into()
            })
            .map(|_| ())
        })?;
    }
    Ok(())
}

fn early_deflation<S: NumericScalar>(
    matrix: &mut impl MatrixStorage<S>,
    end: usize,
    tolerance: &Tolerance<S::Real>,
) {
    let order = matrix.cols();
    for row in 1..=end {
        let subdiagonal = matrix.get_element(row, row - 1).abs();
        let previous = matrix.get_element(row - 1, row - 1).abs();
        let current = matrix.get_element(row, row).abs();
        if should_deflate(subdiagonal, previous, current, tolerance) {
            matrix.as_mut_slice()[row * order + row - 1] = S::ZERO;
        }
    }
}

fn should_deflate<R: scioxide_core::RealScalar>(
    subdiagonal: R,
    previous: R,
    current: R,
    tolerance: &Tolerance<R>,
) -> bool {
    // Standard criterion `|h_{k,k-1}| <= abs + rel (|h_{k-1,k-1}| + |h_kk|)`,
    // rearranged so that every quantity is divided by the larger diagonal
    // magnitude first and the comparison cannot overflow.
    if subdiagonal <= tolerance.absolute() {
        return true;
    }
    let scale = previous.max(current);
    if scale == R::ZERO {
        return false;
    }
    let diagonal_sum_scaled = previous / scale + current / scale;
    let required_relative = ((subdiagonal - tolerance.absolute()) / scale) / diagonal_sum_scaled;
    tolerance.relative() >= required_relative
}

fn active_start<S: NumericScalar>(matrix: &impl MatrixStorage<S>, end: usize) -> usize {
    let mut start = end;
    while start > 0 && matrix.get_element(start, start - 1) != S::ZERO {
        start -= 1;
    }
    start
}

fn trailing_shifts<S: NumericScalar>(
    matrix: &impl MatrixStorage<S>,
    end: usize,
    stagnant: usize,
) -> Result<(S, S), SciOxideError> {
    let a = matrix.get_element(end - 1, end - 1);
    let b = matrix.get_element(end - 1, end);
    let c = matrix.get_element(end, end - 1);
    let d = matrix.get_element(end, end);
    let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
    if scale == S::Real::ZERO {
        return Ok((d, d));
    }

    let scale_scalar = S::from_real(scale);
    let a_scaled = crate::rotation::divide_real(a, scale);
    let b_scaled = crate::rotation::divide_real(b, scale);
    let c_scaled = crate::rotation::divide_real(c, scale);
    let d_scaled = crate::rotation::divide_real(d, scale);

    // Exceptional shifts `d +- (0.75 + 0.4375 i) |c|` (EISPACK-style
    // constants) perturb a stalled iteration off a cycle. Otherwise the 2x2
    // eigenvalues are computed as `half_trace +- sqrt(half_diff^2 + b c)` on
    // entries scaled by their maximum, so the squares cannot overflow.
    let (first_scaled, second_scaled) = if stagnant > 0 && stagnant.is_multiple_of(20) {
        let magnitude = c_scaled.abs();
        let real = magnitude * S::Real::try_from_f64_exact(0.75)?;
        let imaginary = magnitude * S::Real::try_from_f64_exact(0.4375)?;
        let exceptional = S::from_real(real) + imaginary_unit::<S>() * S::from_real(imaginary);
        (d_scaled + exceptional, d_scaled - exceptional)
    } else {
        let two = S::from_real(S::Real::TWO);
        let half_trace = (a_scaled + d_scaled) / two;
        let half_difference = (a_scaled - d_scaled) / two;
        let radicand = half_difference * half_difference + b_scaled * c_scaled;
        let root = principal_sqrt(radicand);
        crate::checked::finite_candidate_with_detail(root, || {
            alloc::format!(
                "{} exceeded the finite scalar range",
                "complex shift square root"
            )
            .into()
        })
        .map(|_| ())?;
        let first = half_trace + root;
        let second = half_trace - root;
        if (first - d_scaled).abs() <= (second - d_scaled).abs() {
            (first, second)
        } else {
            (second, first)
        }
    };

    let first = first_scaled * scale_scalar;
    let second = second_scaled * scale_scalar;
    crate::checked::finite_candidate_with_detail(first, || {
        alloc::format!("{} exceeded the finite scalar range", "first complex shift").into()
    })
    .map(|_| ())?;
    crate::checked::finite_candidate_with_detail(second, || {
        alloc::format!(
            "{} exceeded the finite scalar range",
            "second complex shift"
        )
        .into()
    })
    .map(|_| ())?;
    Ok((first, second))
}

fn principal_sqrt<S: NumericScalar>(value: S) -> S {
    if S::IS_REAL {
        S::from_real(value.real_component().sqrt())
    } else {
        let root = super::into_complex(value).sqrt();
        S::from_components(root.re, root.im)
    }
}

fn imaginary_unit<S: NumericScalar>() -> S {
    S::from_components(S::Real::ZERO, S::Real::ONE)
}

fn francis_shift_step<S: NumericScalar, M: MatrixStorage<S>>(
    matrix: &mut M,
    mut vectors: Option<&mut M>,
    start: usize,
    end: usize,
    shift: S,
) -> Result<(), SciOxideError> {
    // Implicit single-shift QR step: the first rotation is chosen from the
    // first column of `H - shift I`, which creates a bulge at
    // `(start + 2, start)`; each later rotation annihilates the bulge left
    // by the previous one. By the implicit Q theorem this equals an explicit
    // shifted QR step on the active block. Left rotations update rows from
    // the bulge column to the right edge and right rotations update rows
    // `0..=end`, so the step is a similarity on the whole matrix (keeping
    // the coupling to deflated blocks consistent for the full Schur form)
    // while skipping entries that are structurally zero.
    let order = matrix.rows();
    for column in start..end {
        let (top, bottom) = if column == start {
            (
                matrix.get_element(column, column) - shift,
                matrix.get_element(column + 1, column),
            )
        } else {
            (
                matrix.get_element(column, column - 1),
                matrix.get_element(column + 1, column - 1),
            )
        };
        let (rotation, _) = crate::rotation::Rotation::generate(top, bottom)?;
        let first_column = column.saturating_sub(1).max(start.saturating_sub(1));
        rotation.apply(
            matrix.try_view_mut()?,
            column,
            column + 1,
            first_column..order,
        )?;
        rotation.conjugate().apply(
            matrix.try_view_mut()?.transpose(),
            column,
            column + 1,
            0..end + 1,
        )?;
        if let Some(q) = vectors.as_deref_mut() {
            rotation.conjugate().apply(
                q.try_view_mut()?.transpose(),
                column,
                column + 1,
                0..order,
            )?;
        }
        if column > start {
            matrix.as_mut_slice()[(column + 1) * order + column - 1] = S::ZERO;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use scioxide_core::Complex;

    #[test]
    fn complex_givens_is_unitary_and_annihilates_the_second_entry() {
        let first = Complex::new(2.0_f64, -3.0);
        let second = Complex::new(-4.0, 5.0);
        let (cosine, sine) = crate::rotation::Rotation::generate(first, second)
            .map(|(g, _)| (g.c, g.s))
            .unwrap();
        let bottom = -sine.conj() * first + Complex::new(cosine, 0.0) * second;
        let norm = cosine * cosine + sine.norm_sqr();

        assert!(bottom.norm() <= 16.0 * f64::EPSILON);
        assert!((norm - 1.0).abs() <= 16.0 * f64::EPSILON);
    }
}
