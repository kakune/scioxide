//! Exact singleton solves, Sturm bisection and deterministic inverse iteration
//! for selected eigenpairs.
use super::*;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::RealBackend;
use scioxide_core::{NumericalIssue, checked_midpoint};

fn selected_step(
    used: &mut usize,
    budget: usize,
    context: ExecutionContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    if *used >= budget {
        return Err(SciOxideError::convergence_failure(
            ConvergenceReason::IterationLimit,
            ConvergenceDiagnostics::new()
                .with_counts((scioxide_core::EvaluationCounts::from_iterations(*used)).into()),
            "selected-eigenpair iteration budget exhausted",
        ));
    }
    context.check()?;
    crate::solve_support::admit_iteration(
        Some(solve.execution()),
        Some(solve),
        "solve iteration budget exhausted",
    )?;
    *used += 1;
    Ok(())
}

fn count<R: RealScalar>(d: &[R], e: &[R], x: R, minimum: R) -> usize {
    let mut pivot = d[0] - x;
    if pivot.abs() < minimum {
        pivot = -minimum;
    }
    let mut count = usize::from(pivot <= R::ZERO);
    for index in 1..d.len() {
        // Divide first to retain small squared couplings. Infinite quotients
        // have a defined Sturm sign and the following reciprocal is zero.
        pivot = d[index] - x - (e[index - 1] / pivot) * e[index - 1];
        if pivot.abs() < minimum {
            pivot = -minimum;
        }
        count += usize::from(pivot <= R::ZERO);
    }
    count
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn solve<S: NumericScalar, M: MatrixStorage<S>>(
    d: &[S::Real],
    e: &[S::Real],
    values: &mut [S::Real],
    mut vectors: Option<&mut M>,
    pivots: &mut [S],
    rhs: &mut [S],
    options: &HermitianEigenOptions<S::Real>,
    context: ExecutionContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let n = d.len();
    let range = options.resolve_indices(n)?;
    if let Some(vectors) = vectors.as_deref_mut() {
        vectors.as_mut_slice().fill(S::ZERO);
    }
    if range.is_empty() {
        return Ok(0);
    }
    if let [value] = d {
        context.check()?;
        values[0] = *value;
        if let Some(vectors) = vectors {
            vectors.as_mut_slice()[0] = S::ONE;
        }
        return Ok(1);
    }
    let budget = options.resolved_iteration_budget(n, vectors.is_some())?;
    let mut lower = S::Real::INFINITY;
    let mut upper = S::Real::NEG_INFINITY;
    let mut off_max = S::Real::ZERO;
    for index in 0..n {
        let radius = if index > 0 {
            e[index - 1].abs()
        } else {
            S::Real::ZERO
        } + if index + 1 < n {
            e[index].abs()
        } else {
            S::Real::ZERO
        };
        off_max = off_max.max(radius);
        lower = lower.min(d[index] - radius);
        upper = upper.max(d[index] + radius);
    }
    let scale = lower.abs().max(upper.abs());
    if scale == S::Real::ZERO {
        // `T = 0`: every eigenvalue is zero and `e_index` is an eigenvector.
        // Inverse iteration has no pivot scale here.
        context.check()?;
        for (destination, index) in range.clone().enumerate() {
            values[destination] = S::Real::ZERO;
            if let Some(vectors) = vectors.as_deref_mut() {
                vectors.as_mut_slice()[destination * n + index] = S::ONE;
            }
        }
        return Ok(range.len());
    }
    let epsilon = S::Real::EPSILON;
    let tiny = S::Real::ZERO.next_after(S::Real::ONE);
    let padding = (scale * epsilon * S::Real::try_from_f64_lossy(8.0)?).max(tiny);
    lower -= padding;
    upper += padding;
    let minimum = (S::Real::MIN_POSITIVE * off_max.min(S::Real::ONE)).max(tiny);
    let mut used = 0;
    for (destination, index) in range.clone().enumerate() {
        let (mut lo, mut hi) = (lower, upper);
        loop {
            selected_step(&mut used, budget, context, solve)?;
            let midpoint = checked_midpoint(lo, hi)?;
            if midpoint == lo || midpoint == hi {
                values[destination] = midpoint;
                break;
            }
            if count(d, e, midpoint, minimum) <= index {
                lo = midpoint;
            } else {
                hi = midpoint;
            }
            let error_scale = if options.eigenvalue_accuracy() == EigenvalueAccuracy::Relative {
                lo.abs().max(hi.abs())
            } else {
                scale
            };
            if hi - lo <= (epsilon * error_scale).max(tiny) {
                values[destination] = checked_midpoint(lo, hi)?;
                break;
            }
        }
        let Some(vectors) = vectors.as_deref_mut() else {
            continue;
        };
        let lambda = values[destination];
        let mut state = (index as u64).wrapping_add(0x475241444941);
        for value in rhs.iter_mut().take(n) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *value = S::from_real(S::Real::from_u64_half_open01(state) - S::Real::HALF);
        }
        let pivot_min = (epsilon * scale).max(tiny);
        let protected = |value: S::Real| {
            if value.abs() >= pivot_min {
                value
            } else if value < S::Real::ZERO {
                -pivot_min
            } else {
                pivot_min
            }
        };
        let mut converged = false;
        for iteration in 0..16 {
            selected_step(&mut used, budget, context, solve)?;
            pivots[0] = S::from_real(protected(d[0] - lambda));
            for row in 1..n {
                let multiplier = e[row - 1] / pivots[row - 1].real_component();
                pivots[row] = S::from_real(protected(d[row] - lambda - multiplier * e[row - 1]));
                rhs[row] = rhs[row] - S::from_real(multiplier) * rhs[row - 1];
            }
            rhs[n - 1] = rhs[n - 1] / pivots[n - 1];
            for row in (0..n - 1).rev() {
                rhs[row] = (rhs[row] - S::from_real(e[row]) * rhs[row + 1]) / pivots[row];
            }
            // Two reorthogonalization passes resolve multiplicities without
            // making basis selection depend on execution order.
            for _ in 0..2 {
                for previous in 0..destination {
                    let dot = (0..n).fold(S::ZERO, |sum, row| {
                        sum + vectors.get_element(previous, row).conj() * rhs[row]
                    });
                    for (row, value) in rhs[..n].iter_mut().enumerate() {
                        *value = *value - dot * vectors.get_element(previous, row);
                    }
                }
            }
            let norm = crate::kernel::standard::norm_finite(
                crate::FiniteVectorView::from_workspace(crate::VectorView::from_slice(&rhs[..n])),
            )?;
            if norm == S::Real::ZERO || !norm.is_finite() {
                return Err(SciOxideError::numerical_instability(
                    NumericalIssue::LossOfSignificance,
                    "inverse iteration lost its independent direction",
                ));
            }
            rhs[..n]
                .iter_mut()
                .for_each(|value| *value = *value / S::from_real(norm));
            let residual = (0..n).fold(S::Real::ZERO, |maximum, row| {
                let mut value = S::from_real(d[row] - lambda) * rhs[row];
                if row > 0 {
                    value = value + S::from_real(e[row - 1]) * rhs[row - 1];
                }
                if row + 1 < n {
                    value = value + S::from_real(e[row]) * rhs[row + 1];
                }
                maximum.max(value.abs())
            });
            // Refine once after the initial solve: a protected pivot can meet
            // the backward-error bound before the direction is fully resolved.
            if iteration > 0
                && residual
                    <= (scale * epsilon * S::Real::try_from_f64_lossy(32.0 * n as f64)?).max(tiny)
            {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::IterationLimit,
                ConvergenceDiagnostics::new()
                    .with_counts((scioxide_core::EvaluationCounts::from_iterations(used)).into()),
                "inverse iteration did not meet its residual bound",
            ));
        }
        for (row, &value) in rhs[..n].iter().enumerate() {
            vectors.as_mut_slice()[destination * n + row] = value;
        }
    }
    Ok(range.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_relative_budget_covers_a_large_values_job() {
        let n = 1024;
        let diagonal: alloc::vec::Vec<f64> = (0..n).map(|i| 2.0 + i as f64 / n as f64).collect();
        let off = alloc::vec![0.0; n];
        let mut values = alloc::vec![0.0; n];
        let options =
            HermitianEigenOptions::default().with_eigenvalue_accuracy(EigenvalueAccuracy::Relative);
        super::super::tests::with_solve(|context| {
            solve::<f64, crate::Matrix<f64>>(
                &diagonal,
                &off,
                &mut values,
                None,
                &mut [],
                &mut [],
                &options,
                ExecutionContext::sequential(),
                context,
            )
        })
        .unwrap();
        for (&actual, &expected) in values.iter().zip(&diagonal) {
            assert!((actual - expected).abs() <= 4.0 * f64::EPSILON * expected);
        }
    }
}
