//! Blockwise Pal–Walker–Kahan iteration for values-only tridiagonal jobs.
//!
//! Each unreduced block is scaled by a power of two before its couplings are
//! squared, iterated from its smaller endpoint, and finished with closed-form
//! 2×2 eigenvalues. Reversing a block implements QR with the same QL
//! recurrence. The supplied tolerance and one sweep budget cover all blocks.
use super::*;
use scioxide_core::{Budget, Tolerance};

pub(super) fn solve<R: RealScalar>(
    d: &mut [R],
    e: &mut [R],
    options: &HermitianEigenOptions<R>,
    context: ExecutionContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = d.len();
    let budget = options.resolved_iteration_budget(n, false)?;
    let mut first = 0;
    let mut used = 0;
    while first < n {
        context.check()?;
        let mut last = first;
        while last + 1 < n {
            let threshold = options
                .deflation_tolerance()
                .threshold(d[last].abs() + d[last + 1].abs())?;
            if e[last].abs() <= threshold {
                e[last] = R::ZERO;
                break;
            }
            last += 1;
        }
        if first == last {
            first += 1;
            continue;
        }
        let diagonal = &mut d[first..=last];
        let off = &mut e[first..=last];
        // The final off-diagonal slot is scratch for the guarded QL fallback.
        off[last - first] = R::ZERO;
        let maximum = diagonal
            .iter()
            .chain(&off[..last - first])
            .fold(R::ZERO, |bound, value| bound.max(value.abs()));
        let proposed_scale = block_scale(maximum)?;
        // A block with more dynamic range than any squared representation can
        // hold must retain its unsquared couplings, including during scaling.
        let loses_coupling = off[..last - first]
            .iter()
            .any(|value| *value != R::ZERO && value.scalbn(-proposed_scale) == R::ZERO);
        let exponent = if loses_coupling { 0 } else { proposed_scale };
        if exponent != 0 {
            diagonal
                .iter_mut()
                .chain(off.iter_mut())
                .for_each(|value| *value = value.scalbn(-exponent));
        }
        let tolerance = Tolerance::new(
            options.deflation_tolerance().absolute().scalbn(-exponent),
            options.deflation_tolerance().relative(),
        )?;
        // Scaling protects the block norm. Exceptionally graded blocks can
        // still contain a coupling whose square is zero; never discard it.
        if loses_coupling
            || off[..last - first]
                .iter()
                .any(|value| *value != R::ZERO && *value * *value == R::ZERO)
        {
            if used == budget {
                return Err(iteration_limit(used));
            }
            let remaining = budget - used;
            let fallback = options
                .with_deflation_tolerance(tolerance)
                .with_execution_options(
                    options
                        .execution_options()
                        .with_iteration_budget(Budget::try_from(remaining)?),
                );
            used += tridiagonal_ql::<R, R, crate::Matrix<R>>(
                diagonal, off, None, &fallback, context, solve,
            )
            .map_err(|error| {
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: ConvergenceReason::IterationLimit,
                        ..
                    }
                ) {
                    iteration_limit(used + remaining)
                } else {
                    error
                }
            })?;
        } else {
            if diagonal[diagonal.len() - 1].abs() < diagonal[0].abs() {
                diagonal.reverse();
                off[..last - first].reverse();
            }
            off[..last - first]
                .iter_mut()
                .for_each(|value| *value = *value * *value);
            iterate(diagonal, off, tolerance, budget, &mut used, context, solve)?;
        }
        for value in diagonal {
            *value = crate::checked::finite_candidate(value.scalbn(exponent))?;
        }
        first = last + 1;
    }
    Ok(())
}

/// Power-of-two scaling into an interval where squared entries neither overflow nor underflow.
fn block_scale<R: RealScalar>(maximum: R) -> Result<i32, SciOxideError> {
    let lower = R::MIN_POSITIVE.sqrt() / (<R as RealScalar>::EPSILON * <R as RealScalar>::EPSILON);
    let upper = (R::ONE / R::MIN_POSITIVE).sqrt() / (R::TWO + R::ONE);
    let exponent = crate::kernel::standard::exponent;
    Ok(if maximum < lower {
        exponent(maximum)? - exponent(lower)? - 1
    } else if maximum > upper {
        exponent(maximum)? - exponent(upper)? + 1
    } else {
        0
    })
}

fn iteration_limit(used: usize) -> SciOxideError {
    SciOxideError::convergence_failure(
        ConvergenceReason::IterationLimit,
        ConvergenceDiagnostics::new()
            .with_counts(scioxide_core::EvaluationCounts::from_iterations(used).into()),
        "Pal–Walker–Kahan iteration exhausted its budget",
    )
}

fn iterate<R: RealScalar>(
    d: &mut [R],
    e: &mut [R],
    tolerance: Tolerance<R>,
    budget: usize,
    used: &mut usize,
    context: ExecutionContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let n = d.len();
    let mut start = 0;
    while start < n {
        context.check()?;
        let mut end = start;
        while end + 1 < n {
            let threshold = tolerance.threshold(d[end].abs() + d[end + 1].abs())?;
            // Compare unsquared when the threshold's square underflows. This
            // also retains a caller's zero absolute/relative tolerance.
            let negligible = if threshold != R::ZERO && threshold * threshold == R::ZERO {
                e[end].sqrt() <= threshold
            } else {
                e[end] <= threshold * threshold
            };
            if negligible {
                e[end] = R::ZERO;
                break;
            }
            end += 1;
        }
        if end == start {
            start += 1;
            continue;
        }
        if end == start + 1 {
            let (first, second) = pair(d[start], e[start].sqrt(), d[end]);
            d[start] = first;
            d[end] = second;
            e[start] = R::ZERO;
            start += 2;
            continue;
        }
        if *used == budget {
            return Err(iteration_limit(*used));
        }
        crate::solve_support::admit_iteration(
            Some(solve.execution()),
            Some(&mut *solve),
            "solve iteration budget exhausted",
        )?;
        *used += 1;
        let root = e[start].sqrt();
        // Rationalized Wilkinson shift: unlike (d1-d0)/(2*e), this form
        // remains finite even when a surviving coupling is exceptionally tiny.
        let delta = (d[start + 1] - d[start]) * R::HALF;
        let radius = delta.hypot(root);
        let denominator = delta + if delta < R::ZERO { -radius } else { radius };
        let shift = d[start] - root * (root / denominator);
        let mut c = R::ONE;
        let mut s = R::ZERO;
        let mut gamma = d[end] - shift;
        let mut p = gamma * gamma;
        for index in (start..end).rev() {
            let b = e[index];
            let r = p + b;
            if index + 1 != end {
                e[index + 1] = s * r;
            }
            let old_c = c;
            if r == R::ZERO {
                c = R::ZERO;
                s = R::ONE;
            } else {
                c = p / r;
                s = b / r;
            }
            let old_gamma = gamma;
            let alpha = d[index];
            gamma = c * (alpha - shift) - s * old_gamma;
            d[index + 1] = old_gamma + (alpha - gamma);
            p = if c == R::ZERO {
                old_c * b
            } else {
                (gamma * gamma) / c
            };
        }
        e[start] = s * p;
        d[start] = shift + gamma;
        // A finite norm bound protects the recurrence's ordinary arithmetic;
        // check its endpoints before another deflation scan can hide breakdown.
        if !p.is_finite() || !gamma.is_finite() || !d[start].is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "Pal–Walker–Kahan recurrence is not finite",
            ));
        }
    }
    Ok(())
}

/// Eigenvalues of a symmetric 2×2 matrix; the small root is the determinant over
/// the large root, which avoids cancellation.
fn pair<R: RealScalar>(a: R, b: R, c: R) -> (R, R) {
    let sum = a + c;
    let radius = (a - c).hypot(R::TWO * b);
    if sum == R::ZERO {
        return (radius * R::HALF, -radius * R::HALF);
    }
    let large = (sum + if sum < R::ZERO { -radius } else { radius }) * R::HALF;
    let (big, small) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    (large, (big / large) * small - (b / large) * b)
}

#[cfg(test)]
mod tests {
    use super::super::tests::with_solve;
    use super::*;

    fn computed(d: &[f64], e: &[f64], tolerance: Tolerance<f64>) -> alloc::vec::Vec<f64> {
        let mut values = d.to_vec();
        let mut off = e.to_vec();
        off.resize(d.len(), 0.0);
        with_solve(|context| {
            solve(
                &mut values,
                &mut off,
                &HermitianEigenOptions::default().with_deflation_tolerance(tolerance),
                ExecutionContext::sequential(),
                context,
            )
        })
        .unwrap();
        values.sort_by(f64::total_cmp);
        values
    }

    #[test]
    fn empty_singleton_and_stable_pairs() {
        let tolerance = Tolerance::new(0.0, f64::EPSILON).unwrap();
        for (d, e, expected) in [
            (vec![], vec![], vec![]),
            (vec![3.0], vec![], vec![3.0]),
            (vec![2.0, 2.0], vec![-1.0], vec![1.0, 3.0]),
            (vec![0.0, 0.0], vec![1.0], vec![-1.0, 1.0]),
            (vec![1.0, 1e-20], vec![1e-12], vec![1e-20 - 1e-24, 1.0]),
        ] {
            for (actual, expected) in computed(&d, &e, tolerance).iter().zip(expected) {
                assert!(
                    (actual - expected).abs()
                        <= 8.0 * f64::EPSILON * expected.abs().max(f64::MIN_POSITIVE)
                );
            }
        }
    }

    #[test]
    fn binary_scaling_preserves_tiny_and_large_toeplitz_spectra() {
        for exponent in [-1070, -1000, -500, 0, 500, 1000] {
            let scale = 1.0_f64.scalbn(exponent);
            let values = computed(
                &[2.0 * scale; 7],
                &[-scale; 6],
                Tolerance::new(0.0, f64::EPSILON).unwrap(),
            );
            for (index, value) in values.iter().enumerate() {
                let expected = 2.0 - 2.0 * ((index + 1) as f64 * core::f64::consts::PI / 8.0).cos();
                let rounding = f64::from_bits(1) / scale;
                assert!(
                    (value / scale - expected).abs() <= 32.0 * f64::EPSILON + rounding,
                    "exponent={exponent}, actual={}, expected={expected}",
                    value / scale
                );
            }
        }
    }

    #[test]
    fn split_opposite_directions_clusters_and_grades_match_vector_solver() {
        for (d, e) in [
            (vec![1., 2., 4., 8., 6., 3.], vec![0.4, -0.3, 0., 0.2, 0.5]),
            (vec![1., 1., 1., 1., 1., 1.], vec![1e-8; 5]),
            (
                vec![1e-12, 1e-9, 1e-6, 1e-3, 1., 1e3],
                vec![1e-13, 1e-10, 1e-7, 1e-4, 0.1],
            ),
        ] {
            let options = HermitianEigenOptions::default();
            let reference = HermitianEigenDecomposition::tridiagonal(&d, &e).unwrap();
            for (actual, expected) in computed(&d, &e, options.deflation_tolerance())
                .iter()
                .zip(reference.eigenvalues())
            {
                assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs().max(1.0));
            }
        }
    }

    #[test]
    fn unsquared_fallback_preserves_nonzero_couplings() {
        let values = computed(
            &[0.0, 0.0, 1.0],
            &[1e-200, 1e-250],
            Tolerance::new(0.0, 0.0).unwrap(),
        );
        for (actual, expected) in values.iter().zip([-1e-200_f64, 1e-200, 1.0]) {
            assert!(
                (actual / expected - 1.0).abs() <= 16.0 * f64::EPSILON,
                "{values:?}"
            );
        }
    }

    #[test]
    fn unscalable_couplings_are_not_silently_replaced_with_zeros() {
        let mut d = [0.0, 0.0, 1e300_f64];
        let mut e = [1e-200, 1e-250, 0.0];
        let options = HermitianEigenOptions::default()
            .with_deflation_tolerance(Tolerance::new(0.0, 0.0).unwrap())
            .with_execution_options(
                crate::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
            );
        let error = with_solve(|context| {
            solve(
                &mut d,
                &mut e,
                &options,
                ExecutionContext::sequential(),
                context,
            )
        })
        .unwrap_err();
        // Squaring a scaled block would falsely report [0, 0, 1e300] as
        // converged. The unsquared iteration must honor the requested budget.
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                ..
            }
        ));
    }

    #[test]
    fn f32_scaled_blocks_match_analytic_values() {
        for scale in [1e-35_f32, 1.0, 1e35] {
            let mut d = [2.0 * scale; 3];
            let mut e = [scale, scale, 0.0];
            with_solve(|context| {
                solve(
                    &mut d,
                    &mut e,
                    &HermitianEigenOptions::default()
                        .with_deflation_tolerance(Tolerance::new(0.0, f32::EPSILON).unwrap()),
                    ExecutionContext::sequential(),
                    context,
                )
            })
            .unwrap();
            d.sort_by(f32::total_cmp);
            for (actual, expected) in
                d.iter()
                    .zip([2.0 - 2.0_f32.sqrt(), 2.0, 2.0 + 2.0_f32.sqrt()])
            {
                assert!((actual / scale - expected).abs() <= 16.0 * f32::EPSILON);
            }
        }
    }

    #[test]
    fn independent_blocks_share_one_iteration_budget() {
        let options = HermitianEigenOptions::default();
        let mut first = [1.0, 2.0, 4.0, 8.0];
        let mut off = [0.2, 0.3, 0.4, 0.0];
        let used = with_solve(|context| {
            solve(
                &mut first,
                &mut off,
                &options,
                ExecutionContext::sequential(),
                context,
            )?;
            Ok(context.counts().own().iterations())
        })
        .unwrap();
        assert!(used > 0);
        let mut d = [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0];
        let mut e = [0.2, 0.3, 0.4, 0.0, 0.2, 0.3, 0.4, 0.0];
        with_solve(|context| {
            let error = solve(
                &mut d,
                &mut e,
                &options.with_execution_options(
                    (options)
                        .execution_options()
                        .with_iteration_budget(Budget::try_from(used).unwrap()),
                ),
                ExecutionContext::sequential(),
                context,
            )
            .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: ConvergenceReason::IterationLimit,
                    ..
                }
            ));
            assert_eq!(context.counts().own().iterations(), used);
            Ok(())
        })
        .unwrap();
    }
}
