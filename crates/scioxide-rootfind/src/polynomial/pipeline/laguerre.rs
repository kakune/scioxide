//! Candidates from Laguerre iteration with deflation.
use super::*;
pub(crate) fn laguerre_candidates<S: RealScalar>(
    p: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
    options: &PolynomialRootOptions<S>,
    stats: &mut PolynomialRootReport<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<bool, PolynomialFailure>
where
    Complex<S>: NumericScalar<Real = S>,
{
    w.deflated[..=p.degree].copy_from_slice(&w.coefficients[..=p.degree]);
    for found in 0..p.degree {
        let n = p.degree - found;
        if n <= 2 {
            analytic(&w.deflated[..=n], &mut w.current[found..p.degree])?;
            publish_candidates(p, w)?;
            return Ok(true);
        }
        // Use the origin when its first-order correction stays inside the seed
        // radius. Real arithmetic then stays real for real-root factors, without
        // maintaining a separate real-deflation implementation. Degenerate or
        // badly scaled origin derivatives retain the polygon start.
        let derivative = w.deflated[1].norm();
        let mut root = if !derivative.is_zero() && {
            let origin_step = w.deflated[0].norm() / derivative;
            origin_step.is_finite() && origin_step <= w.seed_radii[p.degree - 1]
        } {
            Complex::new(S::ZERO, S::ZERO)
        } else {
            w.current[found]
        };
        let mut success = false;
        for start in 0..options.restart_budget.get() {
            if start > 0 {
                stats.restarts = checked_add(
                    stats.restarts,
                    1,
                    scioxide_core::quantity!("laguerre_restarts"),
                )?;
                root = w.current[(found + start) % p.degree];
            }
            for _ in 0..options.max_iterations.get() {
                iteration(session)?;
                let e = invoke(session, scioxide_core::EvaluationRole::Function, || {
                    checked_basis_evaluation(
                        &w.deflated[..=n],
                        root,
                        PolynomialRootBasisRef::Monomial,
                    )
                })?;
                if residual_ok(e.normalized_residual(), options)? {
                    success = true;
                    break;
                }
                let value = e.scaled_value()?;
                let g = e.scaled_derivative()?.div(value)?;
                let h = g.mul(g)?.sub(e.scaled_second_derivative()?.div(value)?)?;
                let degree = scioxide_core::ScaledValue::from_value(Complex::new(
                    S::try_from_usize_exact(n)?,
                    S::ZERO,
                ))?;
                let one = scioxide_core::ScaledValue::from_value(Complex::new(S::ONE, S::ZERO))?;
                let d = degree
                    .sub(one)?
                    .mul(degree.mul(h)?.sub(g.mul(g)?)?)?
                    .sqrt()?;
                let plus = g.add(d)?;
                let minus = g.sub(d)?;
                let denominator = if plus.magnitude()?.abs_le(minus.magnitude()?) {
                    minus
                } else {
                    plus
                };
                let correction = if denominator.is_zero() {
                    deterministic_perturbation(found, start, start, root)
                } else {
                    degree.div(denominator)?.round()?
                };
                root -= correction;
                finite_complex(root)?;
                stats.active_root_updates = checked_add(
                    stats.active_root_updates,
                    1,
                    scioxide_core::quantity!("laguerre_updates"),
                )?;
            }
            if success {
                break;
            }
        }
        if !success {
            return Ok(false);
        }
        // Complex synthetic division is only a seeding operation. Never snap or
        // infer an uncomputed conjugate from a deflated polynomial.
        w.current[found] = root;
        let mut q = w.deflated[n];
        for j in (0..n).rev() {
            let old = w.deflated[j];
            w.deflated[j] = q;
            q = old + root * q;
        }
        finite_complex(q)?;
        publish_candidates(p, w)?;
    }
    Ok(true)
}
