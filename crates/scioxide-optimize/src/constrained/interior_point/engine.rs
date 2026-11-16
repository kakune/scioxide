//! Pure KKT/merit calculations surrounded by explicit observation and commit steps.
use super::super::sqp;
use super::*;
use crate::utils::{dot, norm2, trial_component};

use scioxide_core::{
    checked_float_product as product, checked_float_quotient as divide, checked_sum,
};

struct CorrectionParameters<S> {
    mu: S,
    penalty: S,
    current: S,
    alpha: S,
    slope: S,
}

fn fixed<S: RealScalar>(bounds: (Option<S>, Option<S>)) -> bool {
    bounds.0.is_some() && bounds.0 == bounds.1
}
fn identity<S: RealScalar + NumericScalar<Real = S>>(h: &mut Matrix<S>) {
    let n = h.rows();
    h.as_mut_slice()
        .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    for j in 0..n {
        h.as_mut_slice()[j * n + j] = <_ as scioxide_core::AlgebraicScalar>::ONE;
    }
}
fn interiorize<S: RealScalar + NumericScalar<Real = S>>(
    x: &mut [S],
    bounds: &[(Option<S>, Option<S>)],
) -> Result<(), SciOxideError> {
    for (v, &(lo, hi)) in x.iter_mut().zip(bounds) {
        if fixed((lo, hi)) {
            if let Some(l) = lo {
                *v = l;
            }
            continue;
        }
        let scale = lo
            .into_iter()
            .chain(hi)
            .fold(S::ONE, |s, b| s.max(NumericScalar::abs(b)));
        let mut push = number::<S>(0.01)? * scale;
        if let (Some(l), Some(u)) = (lo, hi) {
            if l > u {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "empty bound intersection",
                ));
            }
            push = push.min((u / number::<S>(2.)? - l / number::<S>(2.)?) * number::<S>(0.02)?);
        }
        if let Some(l) = lo {
            let inside = (l + push).max(l.next_after(S::INFINITY));
            if *v < inside {
                *v = inside;
            }
        }
        if let Some(u) = hi {
            let inside = (u - push).min(u.next_after(-S::INFINITY));
            if *v > inside {
                *v = inside;
            }
        }
        if !v.is_finite() || lo.is_some_and(|l| *v <= l) || hi.is_some_and(|u| *v >= u) {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "bound interval has no representable strict interior",
            ));
        }
    }
    Ok(())
}
fn value<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    row: usize,
) -> S {
    if row < o.values.len() {
        o.values[row]
    } else {
        let (j, sign, b) = sides[row - o.values.len()];
        sign * (o.x[j] - b)
    }
}
fn jac<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    row: usize,
    col: usize,
) -> S {
    if row < o.values.len() {
        o.jacobian.as_slice()[row * o.x.len() + col]
    } else {
        let (j, sign, _) = sides[row - o.values.len()];
        if j == col {
            sign
        } else {
            <_ as scioxide_core::AlgebraicScalar>::ZERO
        }
    }
}
fn initialize<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    s: &mut InteriorPointState<S>,
    mu: S,
) -> Result<(), SciOxideError> {
    s.lambda.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    s.slack.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    for i in s.equalities..s.lambda.len() {
        s.slack[i] = if i < o.values.len() {
            (-value(o, sides, i)).max(<_ as scioxide_core::AlgebraicScalar>::ONE)
        } else {
            -value(o, sides, i)
        };
        s.lambda[i] = divide(mu, s.slack[i])?;
    }
    Ok(())
}
fn map_duals<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    bounds: &[(Option<S>, Option<S>)],
    s: &mut InteriorPointState<S>,
    scratch: &mut [S],
) -> Result<(), SciOxideError> {
    let m = o.values.len();
    s.dual.multipliers.copy_from_slice(&s.lambda[..m]);
    s.dual
        .lower
        .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    s.dual
        .upper
        .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    for (&(j, sign, _), &l) in sides.iter().zip(&s.lambda[m..]) {
        if sign < <_ as scioxide_core::AlgebraicScalar>::ZERO {
            s.dual.lower[j] = l;
        } else {
            s.dual.upper[j] = l;
        }
    }
    sqp::lagrangian(o, &s.dual, scratch)?;
    for (j, &b) in bounds.iter().enumerate() {
        if fixed(b) {
            s.dual.lower[j] = scratch[j].max(<_ as scioxide_core::AlgebraicScalar>::ZERO);
            s.dual.upper[j] = (-scratch[j]).max(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        }
    }
    Ok(())
}
fn barrier_error<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    s: &InteriorPointState<S>,
    mu: S,
    stationarity: S,
) -> Result<S, SciOxideError> {
    let mut e = stationarity;
    for i in 0..s.lambda.len() {
        e = e.max(NumericScalar::abs(value(o, sides, i) + s.slack[i]));
        if i >= s.equalities {
            e = e.max(NumericScalar::abs(product(s.slack[i], s.lambda[i])? - mu));
        }
    }
    Ok(e)
}
fn merit<S: RealScalar + NumericScalar<Real = S>>(
    o: &Observation<S>,
    sides: &[(usize, S, S)],
    s: &InteriorPointState<S>,
    mu: S,
    penalty: S,
) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(
        o.value
            .ok_or_else(|| invalid(scioxide_core::argument!("f"), "missing merit observation"))?,
        <_ as scioxide_core::AlgebraicScalar>::ONE,
    )?;
    for i in 0..s.lambda.len() {
        sum.add_product(penalty, NumericScalar::abs(value(o, sides, i) + s.slack[i]))?;
        if i >= s.equalities {
            if s.slack[i] <= <_ as scioxide_core::AlgebraicScalar>::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("w"),
                    "nonpositive barrier slack",
                ));
            }
            sum.add_product(-mu, s.slack[i].ln())?;
        }
    }
    sum.finish()
}
fn slope<S: RealScalar + NumericScalar<Real = S>>(
    w: &InteriorPointStorage<S>,
    mu: S,
    penalty: S,
) -> Result<S, SciOxideError> {
    let n = w.accepted.x.len();
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(
        dot(&w.accepted.gradient, &w.direction[..n])?,
        <_ as scioxide_core::AlgebraicScalar>::ONE,
    )?;
    for i in 0..w.model_state.lambda.len() {
        let r = value(&w.accepted, &w.sides, i) + w.model_state.slack[i];
        let mut dr = w.slack_direction[i];
        for j in 0..n {
            dr = trial_component(dr, jac(&w.accepted, &w.sides, i, j), w.direction[j])?;
        }
        let d = crate::geometry::absolute_directional(r, dr);
        sum.add_product(penalty, d)?;
        if i >= w.model_state.equalities {
            sum.add_product(-mu, divide(w.slack_direction[i], w.model_state.slack[i])?)?;
        }
    }
    sum.finish()
}
fn fraction<S: RealScalar + NumericScalar<Real = S>>(
    w: &InteriorPointStorage<S>,
) -> Result<S, SciOxideError> {
    let n = w.accepted.x.len();
    let mut a = S::ONE;
    let tau = number::<S>(0.995)?;
    for i in w.model_state.equalities..w.model_state.lambda.len() {
        for (v, d) in [
            (w.model_state.slack[i], w.slack_direction[i]),
            (w.model_state.lambda[i], w.direction[n + i]),
        ] {
            if d < <_ as scioxide_core::AlgebraicScalar>::ZERO {
                a = a.min(divide(-tau * v, d)?);
            }
        }
    }
    for &(j, sign, b) in &w.sides {
        let d = sign * w.direction[j];
        if d > <_ as scioxide_core::AlgebraicScalar>::ZERO {
            a = a.min(divide(-tau * sign * (w.accepted.x[j] - b), d)?);
        }
    }
    Ok(a)
}
fn trial<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut InteriorPointStorage<S>,
    alpha: S,
) -> Result<bool, SciOxideError> {
    let n = w.accepted.x.len();
    w.trial.value = None;
    w.trial.derivatives = false;
    for j in 0..n {
        w.trial.x[j] = if fixed(w.bounds[j]) {
            w.accepted.x[j]
        } else {
            trial_component(w.accepted.x[j], alpha, w.direction[j])?
        };
    }
    for i in 0..w.model_state.lambda.len() {
        w.proposal.lambda[i] = trial_component(w.model_state.lambda[i], alpha, w.direction[n + i])?;
        w.proposal.slack[i] = trial_component(w.model_state.slack[i], alpha, w.slack_direction[i])?;
        if i >= w.model_state.equalities
            && (w.proposal.lambda[i] <= <_ as scioxide_core::AlgebraicScalar>::ZERO
                || w.proposal.slack[i] <= <_ as scioxide_core::AlgebraicScalar>::ZERO)
        {
            return Ok(false);
        }
    }
    for &(j, sign, b) in &w.sides {
        if sign * (w.trial.x[j] - b) >= <_ as scioxide_core::AlgebraicScalar>::ZERO {
            return Ok(false);
        }
    }
    Ok(true)
}
fn exact<S: RealScalar + NumericScalar<Real = S>>(
    h: &dyn LagrangianHessian<S>,
    o: &mut Observation<S>,
    s: &InteriorPointState<S>,
    out: &mut Matrix<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    // A reused buffer must not turn an incomplete provider write into valid
    // curvature from an earlier point.
    out.as_mut_slice().fill(S::NAN);
    for (i, &lambda) in s.dual.multipliers().iter().enumerate() {
        o.original_multipliers[i] = product(divide(lambda, o.row_scales[i])?, o.objective_scale)?;
    }
    session.invoke(EvaluationRole::Hessian, || {
        h.hessian_into(&o.x, &o.original_multipliers, out.view_mut())
    })?;
    crate::utils::validate_finite(out.as_slice())?;
    for v in out.as_mut_slice() {
        *v = divide(*v, o.objective_scale)?;
    }
    for i in 0..out.rows() {
        for j in 0..i {
            if out.as_slice()[i * out.cols() + j] != out.as_slice()[j * out.cols() + i] {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "Lagrangian Hessian must be symmetric",
                ));
            }
        }
    }
    Ok(())
}
impl<S: RealScalar + NumericScalar<Real = S>> InteriorPoint<S> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kernel(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        w: &mut InteriorPointStorage<S>,
        provider: &mut dyn KktProvider<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let (eq, m) = shape(problem.constraints());
        let h = problem.lagrangian_hessian();
        let n = x0.len();
        let tol = self.options.tolerances();
        w.accepted.objective_scale = <S as scioxide_core::AlgebraicScalar>::ZERO;
        w.accepted
            .row_scales
            .fill(<S as scioxide_core::AlgebraicScalar>::ONE);
        w.accepted.at(x0);
        w.trial.value = None;
        w.diagnostics = InteriorPointDiagnostics::default();
        w.optimality = OptimalityDiagnostics::unavailable();
        w.iterations = 0;
        interiorize(&mut w.accepted.x, &w.bounds)?;
        w.accepted
            .evaluate_start(problem.objective(), Some(problem.constraints()), session)?;
        w.accepted.derivatives(
            problem.objective(),
            Some(problem.constraints()),
            &mut w.differences,
            self.options.model.derivative_strategy(),
            session,
        )?;
        w.trial.inherit_scales(&w.accepted);
        w.restored.inherit_scales(&w.accepted);
        let mut mu = if w.model_state.lambda.len() > eq {
            number::<S>(0.1)?
        } else {
            <_ as scioxide_core::AlgebraicScalar>::ZERO
        };
        initialize(&w.accepted, &w.sides, &mut w.model_state, mu)?;
        map_duals(
            &w.accepted,
            &w.sides,
            &w.bounds,
            &mut w.model_state,
            &mut w.scratch,
        )?;
        identity(&mut w.hessian);
        let coordinate = w
            .accepted
            .x
            .iter()
            .fold(S::ONE, |a, &v| a.max(NumericScalar::abs(v)));
        for value in w.hessian.as_mut_slice() {
            *value /= coordinate;
        }
        if let Some(h) = h {
            exact(h, &mut w.accepted, &w.model_state, &mut w.hessian, session)?;
        }
        w.diagnostics.barrier_parameter = mu;
        w.optimality = sqp::diagnostics(
            &w.accepted,
            &w.model_state.dual,
            &w.bounds,
            eq,
            &mut w.scratch,
        )?;
        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
        let floor = tol
            .complementarity()
            .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?
            .min(
                tol.gradient()
                    .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?,
            )
            .min(
                tol.feasibility()
                    .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?,
            )
            * number::<S>(0.1)?;
        let mut penalty = S::TWO;
        // Phase I uses the same bounded feasibility model as later restoration.
        if sqp::violation(&w.accepted.values, eq)
            > tol
                .feasibility()
                .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?
        {
            if !self.restore(problem, w, session)? {
                return Ok(session.stopped().unwrap_or(OptimizeTermination::Stagnation));
            }
            self.commit_restoration(problem, w, mu, session)?;
        }
        for iteration in 0..self.options.model.max_iterations().get() {
            w.diagnostics.fixed_derivatives_available = w.accepted.fixed_derivatives;
            w.optimality = sqp::diagnostics(
                &w.accepted,
                &w.model_state.dual,
                &w.bounds,
                eq,
                &mut w.scratch,
            )?;
            if sqp::satisfies(w.optimality, tol, &w.accepted, &w.model_state.dual)? {
                return Ok(OptimizeTermination::Converged);
            }
            if let Some(stop) = session.trial_iteration()? {
                return Ok(stop);
            }
            let e = barrier_error(
                &w.accepted,
                &w.sides,
                &w.model_state,
                mu,
                w.optimality.stationarity().unwrap_or(S::INFINITY),
            )?;
            if mu > floor && e <= number::<S>(10.)? * mu {
                mu = (mu * number::<S>(0.2)?).max(floor);
            }
            w.diagnostics.barrier_parameter = mu;
            let ready = self.direction(w, provider, mu, session)?;
            penalty = penalty.max(
                w.model_state
                    .lambda
                    .iter()
                    .fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)))
                    + <_ as scioxide_core::AlgebraicScalar>::ONE,
            );
            let mut accepted = false;
            if ready {
                // At an already feasible minimizer the Newton step may only adjust
                // multipliers. Permit a verified perturbed-KKT reduction with
                // unchanged primal merit, including a central-path dual update.
                if norm2(&w.direction[..n])? <= S::SQRT_EPSILON * (S::ONE + norm2(&w.accepted.x)?) {
                    session.proposal()?;
                    if trial(w, fraction(w)?)?
                        && w.trial.evaluate(
                            problem.objective(),
                            Some(problem.constraints()),
                            session,
                        )?
                    {
                        w.trial.derivatives(
                            problem.objective(),
                            Some(problem.constraints()),
                            &mut w.differences,
                            self.options.model.derivative_strategy(),
                            session,
                        )?;
                        map_duals(
                            &w.trial,
                            &w.sides,
                            &w.bounds,
                            &mut w.proposal,
                            &mut w.scratch,
                        )?;
                        let next = sqp::diagnostics(
                            &w.trial,
                            &w.proposal.dual,
                            &w.bounds,
                            eq,
                            &mut w.scratch,
                        )?;
                        let previous_error = barrier_error(
                            &w.accepted,
                            &w.sides,
                            &w.model_state,
                            mu,
                            w.optimality.stationarity().unwrap_or(S::INFINITY),
                        )?;
                        let next_error = barrier_error(
                            &w.trial,
                            &w.sides,
                            &w.proposal,
                            mu,
                            next.stationarity().unwrap_or(S::INFINITY),
                        )?;
                        let current_merit =
                            merit(&w.accepted, &w.sides, &w.model_state, mu, penalty)?;
                        let roundoff = number::<S>(8.)?
                            * S::EPSILON
                            * (S::ONE + NumericScalar::abs(current_merit));
                        accepted = sqp::satisfies(next, tol, &w.trial, &w.proposal.dual)?
                            || (next_error <= number::<S>(0.5)? * previous_error
                                && next.primal_feasibility().unwrap_or(S::INFINITY)
                                    <= w.optimality
                                        .primal_feasibility()
                                        .unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO)
                                        .max(tol.feasibility().threshold(
                                            <_ as scioxide_core::AlgebraicScalar>::ONE,
                                        )?)
                                && merit(&w.trial, &w.sides, &w.proposal, mu, penalty)?
                                    <= current_merit + roundoff);
                    }
                    if !accepted {
                        session.reject()?;
                    }
                }
            }
            if ready && !accepted {
                // Regularization can perturb feasibility equations. Check the true merit slope.
                let mut derivative = slope(w, mu, penalty)?;
                for _ in 0..8 {
                    if derivative < <_ as scioxide_core::AlgebraicScalar>::ZERO {
                        break;
                    }
                    penalty = product(penalty, number::<S>(10.)?)?;
                    derivative = slope(w, mu, penalty)?;
                }
                if derivative < <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    let current = merit(&w.accepted, &w.sides, &w.model_state, mu, penalty)?;
                    let initial_alpha = fraction(w)?;
                    let _ = crate::globalization::backtrack(
                        initial_alpha,
                        S::EPSILON,
                        S::HALF,
                        crate::globalization::TRIALS,
                        |attempt, alpha| {
                            session.proposal()?;
                            if trial(w, alpha)?
                                && w.trial.evaluate(
                                    problem.objective(),
                                    Some(problem.constraints()),
                                    session,
                                )?
                                && crate::utils::armijo_satisfied(
                                    merit(&w.trial, &w.sides, &w.proposal, mu, penalty)?,
                                    current,
                                    crate::globalization::sufficient_decrease(),
                                    alpha,
                                    derivative,
                                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                                )?
                            {
                                accepted = true;
                                return Ok(Some(()));
                            }
                            session.reject()?;
                            if attempt == 0 && w.trial.value.is_some() && m > 0 {
                                // Correct nonlinear equality/slack defects with the same retained KKT factors.
                                w.diagnostics.second_order_corrections += 1;
                                if self.correct(
                                    problem,
                                    w,
                                    provider,
                                    CorrectionParameters {
                                        mu,
                                        penalty,
                                        current,
                                        alpha,
                                        slope: derivative,
                                    },
                                    session,
                                )? {
                                    w.diagnostics.accepted_corrections += 1;
                                    accepted = true;
                                    return Ok(Some(()));
                                }
                            }

                            Ok(None)
                        },
                    )?;
                }
            }
            if !accepted {
                if !self.restore(problem, w, session)? {
                    return Ok(session.stopped().unwrap_or(OptimizeTermination::Stagnation));
                }
                self.commit_restoration(problem, w, mu, session)?;
            } else {
                w.trial.derivatives(
                    problem.objective(),
                    Some(problem.constraints()),
                    &mut w.differences,
                    self.options.model.derivative_strategy(),
                    session,
                )?;
                map_duals(
                    &w.trial,
                    &w.sides,
                    &w.bounds,
                    &mut w.proposal,
                    &mut w.scratch,
                )?;
                for j in 0..n {
                    w.displacement[j] = w.trial.x[j] - w.accepted.x[j];
                }
                let next =
                    sqp::diagnostics(&w.trial, &w.proposal.dual, &w.bounds, eq, &mut w.scratch)?;
                let update = if let Some(h) = h {
                    exact(h, &mut w.trial, &w.proposal, &mut w.trial_hessian, session)?;
                    None
                } else {
                    sqp::lagrangian(&w.accepted, &w.proposal.dual, &mut w.old_gradient)?;
                    sqp::lagrangian(&w.trial, &w.proposal.dual, &mut w.secant)?;
                    for j in 0..n {
                        w.secant[j] = w.secant[j] - w.old_gradient[j];
                        if fixed(w.bounds[j]) {
                            w.secant[j] = <_ as scioxide_core::AlgebraicScalar>::ZERO;
                        }
                    }
                    Some(w.update.prepare(
                        w.hessian.view(),
                        &w.displacement,
                        &w.secant,
                        crate::HessianForm::Hessian,
                        crate::HessianUpdate::Bfgs(crate::CurvaturePolicy::default()),
                        session.execution()?,
                        Some(session),
                        crate::quasi_newton::UpdateSource::Verified,
                        scioxide_core::argument!("w"),
                    )?)
                };
                session.accept(scioxide_core::PartialResultTrust::Admissible)?;
                if let Some(u) = update {
                    w.update.commit(&mut w.hessian, u);
                } else {
                    w.hessian
                        .as_mut_slice()
                        .copy_from_slice(w.trial_hessian.as_slice());
                }
                w.accepted.copy_from(&w.trial);
                w.model_state.copy_from(&w.proposal);
                w.optimality = next;
            }
            w.iterations = iteration + 1;
        }
        Ok(
            if sqp::satisfies(w.optimality, tol, &w.accepted, &w.model_state.dual)? {
                OptimizeTermination::Converged
            } else {
                OptimizeTermination::IterationLimit
            },
        )
    }
    fn direction(
        &self,
        w: &mut InteriorPointStorage<S>,
        provider: &mut dyn KktProvider<S>,
        mu: S,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = w.accepted.x.len();
        let rows = w.model_state.lambda.len();
        let order = n + rows;
        let eq = w.model_state.equalities;
        sqp::lagrangian(&w.accepted, &w.model_state.dual, &mut w.scratch)?;
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut w.base_matrix, order)?;
        assembly.symmetric(0, n, |i, j| Ok(w.hessian.as_slice()[i * n + j]))?;
        assembly.coupling(n, 0, rows, n, |i, j| Ok(jac(&w.accepted, &w.sides, i, j)))?;
        assembly.diagonal(n, rows, |i| {
            if i < eq {
                Ok(<S as scioxide_core::AlgebraicScalar>::ZERO)
            } else {
                Ok(-divide(w.model_state.slack[i], w.model_state.lambda[i])?)
            }
        })?;
        for j in 0..n {
            w.rhs[j] = -w.scratch[j];
        }
        for i in 0..rows {
            let r = value(&w.accepted, &w.sides, i) + w.model_state.slack[i];
            w.rhs[n + i] = if i < eq {
                -r
            } else {
                checked_sum(&[
                    -r,
                    w.model_state.slack[i],
                    -divide(mu, w.model_state.lambda[i])?,
                ])?
            };
        }
        for j in 0..n {
            if fixed(w.bounds[j]) {
                assembly.fix(j)?;
                w.rhs[j] = <_ as scioxide_core::AlgebraicScalar>::ZERO;
            }
        }
        let scale = w
            .hessian
            .as_slice()
            .iter()
            .fold(S::ONE, |a, &v| a.max(NumericScalar::abs(v)));
        let mut primal = S::ZERO;
        let mut dual = S::ZERO;
        let execution = session.execution()?;
        for attempt in 0..16 {
            w.matrix
                .as_mut_slice()
                .copy_from_slice(w.base_matrix.as_slice());
            let mut assembly = crate::kkt::assembly::Assembly::retained(&mut w.matrix);
            for j in 0..n {
                if !fixed(w.bounds[j]) {
                    assembly.shift(j, primal)?;
                }
            }
            assembly.diagonal(n, eq, |_| Ok(-dual))?;
            crate::utils::validate_finite(w.matrix.as_slice())?;
            let inertia = session.invoke(EvaluationRole::Factorization, || {
                provider.factor(w.matrix.view(), execution)
            })?;
            crate::kkt::validate_inertia(inertia, order)?;
            w.diagnostics.inertia = Some(inertia);
            w.diagnostics.primal_regularization = primal;
            w.diagnostics.dual_regularization = dual;
            if crate::kkt::matches_inertia(inertia, n, rows)? {
                session.invoke(EvaluationRole::LinearSolve, || {
                    provider.solve_into(&w.rhs, &mut w.direction, execution)
                })?;
                crate::utils::validate_finite(&w.direction)?;
                self.verify(w)?;
                w.slack_direction
                    .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
                for i in eq..rows {
                    w.slack_direction[i] = checked_sum(&[
                        divide(mu, w.model_state.lambda[i])?,
                        -w.model_state.slack[i],
                        -product(
                            divide(w.model_state.slack[i], w.model_state.lambda[i])?,
                            w.direction[n + i],
                        )?,
                    ])?;
                }
                return Ok(true);
            }
            w.diagnostics.regularization_attempts += 1;
            // Equality regularization resolves dependent Jacobian rows, while a positive
            // primal shift supplies curvature. Both are removed from final KKT checks.
            if inertia.zero() > 0 && eq > 0 {
                dual = S::SQRT_EPSILON * mu.max(S::SQRT_EPSILON);
            }
            primal = if attempt == 0 {
                S::SQRT_EPSILON * scale
            } else {
                product(primal, number::<S>(10.)?)?
            };
        }
        Ok(false)
    }
    fn verify(&self, w: &InteriorPointStorage<S>) -> Result<(), SciOxideError> {
        let t = self.options.tolerances();
        crate::kkt::verify_residual_with_floor(
            w.matrix.view(),
            &w.rhs,
            &w.direction,
            t.gradient()
                .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?
                .min(
                    t.feasibility()
                        .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?,
                )
                * number::<S>(0.01)?,
        )
    }
    fn correct(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut InteriorPointStorage<S>,
        provider: &mut dyn KktProvider<S>,
        parameters: CorrectionParameters<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let CorrectionParameters {
            mu,
            penalty,
            current,
            alpha,
            slope,
        } = parameters;
        let n = w.accepted.x.len();
        let rows = w.model_state.lambda.len();
        let eq = w.model_state.equalities;
        w.rhs.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        for i in 0..rows {
            w.rhs[n + i] = -(value(&w.trial, &w.sides, i) + w.proposal.slack[i])
                + (S::ONE - alpha) * (value(&w.accepted, &w.sides, i) + w.model_state.slack[i]);
        }
        // Preserve the original direction across this solve so backtracking can resume.
        w.displacement.copy_from_slice(&w.direction[..n]);
        // The unregularized assembly is no longer needed during this search;
        // its first row holds a bounded backup of the full direction.
        let order = n + rows;
        for j in 0..order {
            w.base_matrix.as_mut_slice()[j] = w.direction[j];
        }
        let execution = session.execution()?;
        session.invoke(EvaluationRole::LinearSolve, || {
            provider.solve_into(&w.rhs, &mut w.direction, execution)
        })?;
        crate::utils::validate_finite(&w.direction)?;
        self.verify(w)?;
        let small = norm2(&w.direction[..n])? <= alpha * norm2(&w.displacement)?;
        let mut admissible = small;
        if small {
            for j in 0..n {
                if !fixed(w.bounds[j]) {
                    w.trial.x[j] = checked_sum(&[w.trial.x[j], w.direction[j]])?;
                }
            }
            for i in eq..rows {
                let ds = -product(
                    divide(w.model_state.slack[i], w.model_state.lambda[i])?,
                    w.direction[n + i],
                )?;
                w.proposal.slack[i] += ds;
                if w.proposal.slack[i] <= <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    admissible = false;
                }
            }
            for i in 0..rows {
                w.proposal.lambda[i] = w.proposal.lambda[i] + w.direction[n + i];
                if i >= eq && w.proposal.lambda[i] <= <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    admissible = false;
                }
            }
            for &(j, sign, b) in &w.sides {
                if sign * (w.trial.x[j] - b) >= <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    admissible = false;
                }
            }
        }
        for j in 0..order {
            w.direction[j] = w.base_matrix.as_slice()[j];
        }
        if admissible {
            session.proposal()?;
            if w.trial
                .evaluate(problem.objective(), Some(problem.constraints()), session)?
                && crate::utils::armijo_satisfied(
                    merit(&w.trial, &w.sides, &w.proposal, mu, penalty)?,
                    current,
                    crate::globalization::sufficient_decrease(),
                    alpha,
                    slope,
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                )?
            {
                return Ok(true);
            }
            session.reject()?;
        }
        Ok(false)
    }
    fn restore(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut InteriorPointStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let n = w.accepted.x.len();
        let eq = w.model_state.equalities;
        let initial = sqp::violation(&w.accepted.values, eq);
        let target = self
            .options
            .tolerances()
            .feasibility()
            .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
        if initial <= target {
            return Ok(false);
        }
        w.diagnostics.restoration_solves += 1;
        w.restored.copy_from(&w.accepted);
        for _ in 0..self.options.model.max_restoration_iterations().get() {
            if session.iteration()?.is_some() {
                return Ok(false);
            }
            for i in 0..w.restored.values.len() {
                let active =
                    i < eq || w.restored.values[i] > <_ as scioxide_core::AlgebraicScalar>::ZERO;
                w.restoration_target[i] = if active {
                    -w.restored.values[i]
                } else {
                    <_ as scioxide_core::AlgebraicScalar>::ZERO
                };
                for j in 0..n {
                    w.restoration_design.as_mut_slice()[i * n + j] = if active {
                        w.restored.jacobian.as_slice()[i * n + j]
                    } else {
                        <_ as scioxide_core::AlgebraicScalar>::ZERO
                    };
                }
            }
            sqp::step_bounds(&w.bounds, &w.restored.x, &mut w.step_bounds)?;
            // Feasibility steps ignore the barrier, so stop halfway to each bound.
            // Collapsing onto an active bound would leave tiny slacks and huge
            // multipliers that cut every later barrier step to the boundary.
            let tau = S::HALF;
            for (b, &original) in w.step_bounds.iter_mut().zip(&w.bounds) {
                if !fixed(original) {
                    b.0 = b.0.map(|v| v * tau);
                    b.1 = b.1.map(|v| v * tau);
                }
            }
            let model = crate::SqpSubproblem {
                objective: crate::SqpSubproblemObjective::LeastSquares {
                    design: w.restoration_design.view(),
                    target: &w.restoration_target,
                },
                jacobian: w.empty_jacobian.view(),
                values: &[],
                equalities: 0,
                bounds: &w.step_bounds,
                tolerances: self.options.tolerances(),
                max_iterations: self.options.model.max_subproblem_iterations(),
            };
            let restoration = w.restoration.as_mut().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    "unprepared restoration provider",
                )
            })?;
            let r = sqp::solve_child(restoration, model, &mut w.restoration_state, session)?;
            if r.termination != OptimizeTermination::Converged {
                return Ok(false);
            }
            let before = sqp::violation(&w.restored.values, eq);

            let searched = crate::globalization::backtrack(
                S::ONE,
                S::EPSILON,
                S::HALF,
                crate::globalization::TRIALS,
                |_attempt, alpha| {
                    session.proposal()?;
                    for j in 0..n {
                        w.trial.x[j] = if fixed(w.bounds[j]) {
                            w.restored.x[j]
                        } else {
                            trial_component(w.restored.x[j], alpha, w.restoration_state.step[j])?
                        };
                    }
                    if w.trial.evaluate(
                        problem.objective(),
                        Some(problem.constraints()),
                        session,
                    )? {
                        let v = sqp::violation(&w.trial.values, eq);
                        if v <= target || v <= initial * number::<S>(0.5)? {
                            return Ok(Some(true));
                        }
                        if v < before {
                            return Ok(Some(false));
                        }
                    }
                    session.reject()?;

                    Ok(None)
                },
            )?;
            if searched == Ok(true) {
                return Ok(true);
            }
            let progress = searched.is_ok();
            if !progress {
                return Ok(false);
            }
            w.trial.derivatives(
                problem.objective(),
                Some(problem.constraints()),
                &mut w.differences,
                self.options.model.derivative_strategy(),
                session,
            )?;
            w.restored.copy_from(&w.trial);
        }
        Ok(false)
    }
    fn commit_restoration(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut InteriorPointStorage<S>,
        mu: S,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let h = problem.lagrangian_hessian();
        w.trial.derivatives(
            problem.objective(),
            Some(problem.constraints()),
            &mut w.differences,
            self.options.model.derivative_strategy(),
            session,
        )?;
        initialize(&w.trial, &w.sides, &mut w.proposal, mu)?;
        map_duals(
            &w.trial,
            &w.sides,
            &w.bounds,
            &mut w.proposal,
            &mut w.scratch,
        )?;
        if let Some(h) = h {
            exact(h, &mut w.trial, &w.proposal, &mut w.trial_hessian, session)?;
        } else {
            identity(&mut w.trial_hessian);
        }
        let next = sqp::diagnostics(
            &w.trial,
            &w.proposal.dual,
            &w.bounds,
            w.model_state.equalities,
            &mut w.scratch,
        )?;
        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
        w.accepted.copy_from(&w.trial);
        w.model_state.copy_from(&w.proposal);
        w.hessian
            .as_mut_slice()
            .copy_from_slice(w.trial_hessian.as_slice());
        w.optimality = next;
        Ok(())
    }
}
