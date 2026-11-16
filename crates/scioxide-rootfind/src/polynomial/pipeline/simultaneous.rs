//! Aberth–Ehrlich and Durand–Kerner simultaneous iterations.
use super::*;
use scioxide_poly::backend::polynomial_iteration::{WeierstrassCorrection, weierstrass_correction};
pub(crate) fn simultaneous<S: RealScalar>(
    p: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
    method: PolynomialRootMethod,
    options: &PolynomialRootOptions<S>,
    stats: &mut PolynomialRootReport<S>,
    session: &crate::context::Session<'_, '_, '_>,
    seeded: bool,
) -> Result<bool, PolynomialFailure>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let n = p.degree;
    let mut best_error = S::INFINITY;
    w.best[..n].copy_from_slice(&w.current[..n]);
    for start in 0..options.restart_budget.get() {
        if start > 0 {
            stats.restarts =
                checked_add(stats.restarts, 1, scioxide_core::quantity!("root_restarts"))?;
            initialize(&mut w.current[..n], &w.seed_radii[..n], start)?;
        } else if !seeded {
            initialize(&mut w.current[..n], &w.seed_radii[..n], 0)?;
        }
        // This cache belongs to these coefficients, seeds and coordinates only.
        w.evaluations.fill(None);
        for _ in 0..options.max_iterations.get() {
            iteration(session)?;
            let mut max_error = S::ZERO;
            let mut converged = true;
            for (radius, root) in w.root_radii[..n].iter_mut().zip(&w.current[..n]) {
                *radius = root.norm();
            }
            let scaled_interactions = w.root_radii[..n]
                .iter()
                .any(|&radius| radius > S::MAX.sqrt() * S::HALF || radius < S::MIN_POSITIVE.sqrt());
            for i in 0..n {
                let root = w.current[i];
                let e = match w.evaluations[i] {
                    Some(e) => e,
                    None => invoke(session, scioxide_core::EvaluationRole::Function, || {
                        checked_basis_evaluation(&w.coefficients[..=n], root, p.basis)
                    })?,
                };
                let newton = e.newton_correction()?;
                let settled = match newton {
                    Some(step) if e.roundoff_limited() => options
                        .step_tolerance
                        .tolerance()
                        .is_residual_ok(step.norm(), S::ONE + root.norm())?,
                    _ => false,
                };
                let correction = if e.normalized_residual().is_zero() || settled {
                    Complex::new(S::ZERO, S::ZERO)
                } else if method == PolynomialRootMethod::DurandKerner {
                    durand_kerner_correction(
                        &w.current[..n],
                        i,
                        e.scaled_value()?
                            .div(scioxide_core::ScaledValue::from_value(w.coefficients[n])?)?,
                        start,
                        &mut stats.pair_interactions,
                    )?
                } else {
                    aberth_correction(
                        &w.current[..n],
                        &w.root_radii[..n],
                        scaled_interactions,
                        i,
                        newton,
                        start,
                        &mut stats.pair_interactions,
                    )?
                };
                let next = root - correction;
                finite_complex(next)?;
                let ne = if next == root {
                    e
                } else {
                    invoke(session, scioxide_core::EvaluationRole::Function, || {
                        checked_basis_evaluation(&w.coefficients[..=n], next, p.basis)
                    })?
                };
                w.evaluations[i] = Some(ne);
                max_error = max_error.max(ne.normalized_residual());
                converged &= residual_ok(ne.normalized_residual(), options)?
                    && options
                        .step_tolerance
                        .tolerance()
                        .is_residual_ok(correction.norm(), S::ONE + next.norm())?;
                w.next[i] = next;
                stats.active_root_updates = checked_add(
                    stats.active_root_updates,
                    1,
                    scioxide_core::quantity!("root_updates"),
                )?;
            }
            w.current[..n].copy_from_slice(&w.next[..n]);
            if max_error <= best_error {
                best_error = max_error;
                w.best[..n].copy_from_slice(&w.current[..n]);
                publish_candidates(p, w)?;
            }
            if converged {
                publish_candidates(p, w)?;
                return Ok(true);
            }
        }
    }
    w.current[..n].copy_from_slice(&w.best[..n]);
    w.evaluations.fill(None);
    publish_candidates(p, w)?;
    Ok(false)
}
/// Aberth–Ehrlich correction of one candidate against the others, counting
/// the pair interactions it uses.
pub(crate) fn aberth_correction<S: RealScalar>(
    roots: &[Complex<S>],
    radii: &[S],
    scaled: bool,
    index: usize,
    newton: Option<Complex<S>>,
    start: usize,
    interactions: &mut usize,
) -> Result<Complex<S>, SciOxideError> {
    let Some(newton) = newton else {
        return Ok(deterministic_perturbation(
            index,
            index,
            start,
            roots[index],
        ));
    };
    if scaled {
        return aberth_scaled(roots, index, newton, start, interactions);
    }
    let mut sum = Complex::new(real_zero::<S>(), real_zero::<S>());
    let mut compensation = Complex::new(real_zero::<S>(), real_zero::<S>());
    for (other_index, &other) in roots.iter().enumerate() {
        if index == other_index {
            continue;
        }
        *interactions = checked_add(
            *interactions,
            1,
            scioxide_core::quantity!("aberth_root_interaction_count"),
        )?;
        let mut difference = roots[index] - other;
        let collision = S::SQRT_EPSILON * radii[index].max(radii[other_index]);
        if difference.norm() <= collision {
            difference += deterministic_perturbation(index, other_index, start, roots[index]);
        }
        let term = Complex::new(real_one::<S>(), real_zero::<S>()) / difference;
        let adjusted = term - compensation;
        let next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
    }
    let denominator = Complex::new(real_one::<S>(), real_zero::<S>()) - newton * sum;
    let threshold = S::EPSILON * (S::ONE + newton.norm() * sum.norm());
    if denominator.norm() <= threshold {
        Ok(deterministic_perturbation(
            index,
            index,
            start,
            roots[index],
        ))
    } else {
        Ok(newton / denominator)
    }
}

/// Durand–Kerner correction from the Weierstrass kernel of `scioxide-poly`.
///
/// `value` is `p(z_i) / a_n`. Exactly coincident candidates, the kernel's only
/// undefined case, receive the deterministic restart perturbation instead;
/// near-equal candidates take their ordinary correction. Every call counts
/// `n - 1` pair interactions.
pub(crate) fn durand_kerner_correction<S: RealScalar>(
    roots: &[Complex<S>],
    index: usize,
    value: scioxide_core::ScaledValue<Complex<S>>,
    start: usize,
    interactions: &mut usize,
) -> Result<Complex<S>, SciOxideError> {
    *interactions = checked_add(
        *interactions,
        roots.len() - 1,
        scioxide_core::quantity!("durand_kerner_root_interaction_count"),
    )?;
    let others = roots
        .iter()
        .enumerate()
        .filter(|&(other, _)| other != index)
        .map(|(_, &root)| root);
    match weierstrass_correction(roots[index], others, value)? {
        WeierstrassCorrection::Defined { correction, .. } => correction.round(),
        WeierstrassCorrection::Coincident => Ok(deterministic_perturbation(
            index,
            index,
            start,
            roots[index],
        )),
    }
}

/// Tests whether two candidates nearly coincide, relative to their magnitude, in extended range.
fn separation<S: RealScalar>(
    root: Complex<S>,
    other: Complex<S>,
    index: usize,
    other_index: usize,
    start: usize,
) -> Result<scioxide_core::ScaledValue<Complex<S>>, SciOxideError> {
    let root_scaled = scioxide_core::ScaledValue::from_value(root)?;
    let other_scaled = scioxide_core::ScaledValue::from_value(other)?;
    let mut difference = root_scaled.sub(other_scaled)?;
    let a = root_scaled.magnitude()?;
    let b = other_scaled.magnitude()?;
    let scale = if a.abs_le(b) { b } else { a };
    let collision = scale.mul(Scaled::from_value(S::SQRT_EPSILON)?)?;
    if difference.magnitude()?.abs_le(collision) {
        difference = difference.add(scioxide_core::ScaledValue::from_value(
            deterministic_perturbation(index, other_index, start, root),
        )?)?;
    }
    Ok(difference)
}

fn aberth_scaled<S: RealScalar>(
    roots: &[Complex<S>],
    index: usize,
    newton: Complex<S>,
    start: usize,
    interactions: &mut usize,
) -> Result<Complex<S>, SciOxideError> {
    let one = scioxide_core::ScaledValue::from_value(Complex::new(S::ONE, S::ZERO))?;
    let mut sum = scioxide_core::ScaledValue::ZERO;
    for (j, &other) in roots.iter().enumerate() {
        if j == index {
            continue;
        }
        *interactions = checked_add(
            *interactions,
            1,
            scioxide_core::quantity!("aberth_root_interaction_count"),
        )?;
        sum = sum.add(one.div(separation(roots[index], other, index, j, start)?)?)?;
    }
    let newton = scioxide_core::ScaledValue::from_value(newton)?;
    let interaction = newton.mul(sum)?;
    let denominator = one.sub(interaction)?;
    let threshold = Scaled::from_value(S::EPSILON)?
        .mul(Scaled::from_value(S::ONE)?.add(interaction.magnitude()?)?)?;
    if denominator.magnitude()?.abs_le(threshold) {
        Ok(deterministic_perturbation(
            index,
            index,
            start,
            roots[index],
        ))
    } else {
        newton.div(denominator)?.round()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durand_kerner_uses_the_shared_kernel_and_perturbs_only_coincidences() {
        let distinct = [
            Complex::new(1.0_f64, 0.5),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 1.0),
        ];
        let value = scioxide_core::ScaledValue::from_value(Complex::new(0.5, -0.25)).unwrap();
        let correction = match weierstrass_correction(
            distinct[0],
            distinct[1..].iter().copied(),
            value,
        )
        .unwrap()
        {
            WeierstrassCorrection::Defined { correction, .. } => Some(correction),
            WeierstrassCorrection::Coincident => None,
        }
        .unwrap();
        let mut interactions = 0;
        let actual = durand_kerner_correction(&distinct, 0, value, 0, &mut interactions).unwrap();
        assert_eq!(actual, correction.round().unwrap());
        assert_eq!(interactions, 2);

        let coincident = [Complex::new(1.0_f64, 0.5); 3];
        for (index, start) in [(0, 0), (2, 1)] {
            let actual =
                durand_kerner_correction(&coincident, index, value, start, &mut interactions)
                    .unwrap();
            assert_eq!(
                actual,
                deterministic_perturbation(index, index, start, coincident[index])
            );
        }
        assert_eq!(interactions, 6);

        // Near-equal candidates are not coincident: no perturbation is applied.
        let near = [
            Complex::new(1.0_f64, 0.0),
            Complex::new(1.0 + f64::EPSILON, 0.0),
        ];
        let tiny = scioxide_core::ScaledValue::from_value(Complex::new(f64::EPSILON, 0.0)).unwrap();
        let actual = durand_kerner_correction(&near, 0, tiny, 0, &mut interactions).unwrap();
        assert!((actual - Complex::new(-1.0, 0.0)).norm() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn unchanged_seeded_roots_need_no_new_evaluation_in_the_sweep() {
        use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveState};
        let options = PolynomialRootOptions::default()
            .with_polish(false)
            .with_method(PolynomialRootMethod::Aberth);
        for method in [
            PolynomialRootMethod::Aberth,
            PolynomialRootMethod::DurandKerner,
        ] {
            let mut w = PolynomialRootWorkspace::workspace(4, &options).unwrap();
            w.coefficients.copy_from_slice(&[
                Complex::new(-1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ]);
            let roots = [
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(-1.0, 0.0),
                Complex::new(0.0, -1.0),
            ];
            w.current.copy_from_slice(&roots);
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let identity = scioxide_core::SolveIdentity::new(
                ProblemToken::new(2011),
                scioxide_core::algorithm_id!("test", "simultaneous-cache", 1),
                4,
                0,
            );
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                .unwrap();
            let session = crate::context::Session::new(&mut context);
            let mut report = PolynomialRootReport::new(
                method,
                options.backward_error_tolerance.tolerance(),
                1.0,
                4,
                session.algorithm().unwrap(),
            );
            let p = Prepared {
                zeros: 0,
                degree: 4,
                scale: 1.0,
                basis: PolynomialRootBasisRef::Monomial,
                real: true,
                companion_safe: true,
            };
            assert!(
                simultaneous(p, &mut w, method, &options, &mut report, &session, true).unwrap()
            );
            assert_eq!(
                session
                    .accounting()
                    .unwrap()
                    .own()
                    .provider(scioxide_core::EvaluationRole::Function),
                4
            );
            assert_eq!(report.active_root_updates, 4);
            assert_eq!(w.current, roots);
        }
    }
}
