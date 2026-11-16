//! Feasible-step calculations and the active-set iteration.
use super::workspace::Storage;
use super::*;
use crate::context::SolveSession;
use crate::quasi_newton::scaled_dot as dot;

use scioxide_core::Scaled;
use scioxide_core::{EvaluationRole, PartialResultTrust};
pub(super) fn add<S: RealScalar>(x: S, a: S, d: S) -> Result<S, SciOxideError> {
    crate::utils::ordinary_trial_component(x, a, d)
}
fn norm<S: RealScalar>(x: &[S]) -> Result<S, SciOxideError> {
    dot(x, x)?.sqrt()?.round()
}
use crate::geometry::project_component as project;
pub(super) fn validate<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    nnls: bool,
) -> Result<(), SciOxideError> {
    if nnls && p.bounds.is_some() {
        return Err(invalid(
            scioxide_core::argument!("problem"),
            "NNLS uses implicit nonnegative bounds; use the bounded solver for explicit bounds",
        ));
    }
    Ok(())
}
fn evaluate<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    o: &LinearLeastSquaresOptions<S>,
    nnls: bool,
    r: &mut LinearLeastSquaresResult<S>,
    scratch: &mut [S],
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if p.rows() == 0 || p.cols() == 0 {
        r.residual.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    } else {
        session.invoke(EvaluationRole::LinearOperator, || {
            session.with_context(|context| {
                p.operator()
                    .apply_into_scoped(
                        &r.x,
                        &mut r.residual,
                        &mut scratch[..p.operator().apply_workspace_len()?],
                        &mut scioxide_core::OperatorContext::new(context),
                    )
                    .and_then(scioxide_core::Invocation::require)
            })?
        })?;
        finite_product(&r.residual)?;
    }
    for (v, &b) in r.residual.iter_mut().zip(p.rhs) {
        *v = add(*v, -S::ONE, b)?;
    }
    if p.rows() == 0 || p.cols() == 0 {
        r.gradient.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    } else {
        session.invoke(EvaluationRole::LinearOperator, || {
            session.with_context(|context| {
                p.operator()
                    .apply_adjoint_into_scoped(
                        &r.residual,
                        &mut r.gradient,
                        &mut scratch[..p.operator().apply_adjoint_workspace_len()?],
                        &mut scioxide_core::OperatorContext::new(context),
                    )
                    .and_then(scioxide_core::Invocation::require)
            })?
        })?;
        finite_product(&r.gradient)?;
    }
    r.residual_norm = norm(&r.residual)?;
    r.objective = dot(&r.residual, &r.residual)?
        .mul(Scaled::from_value(S::HALF)?)?
        .round()?;
    r.kkt = LinearLeastSquaresKkt {
        primal_feasibility: <_ as scioxide_core::AlgebraicScalar>::ZERO,
        stationarity: <_ as scioxide_core::AlgebraicScalar>::ZERO,
        dual_feasibility: <_ as scioxide_core::AlgebraicScalar>::ZERO,
        complementarity: <_ as scioxide_core::AlgebraicScalar>::ZERO,
    };
    let threshold = o
        .feasibility
        .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
    for (i, (&x, &g)) in r.x.iter().zip(&r.gradient).enumerate() {
        let (l, u) = p.bound(i, nnls);
        let dl = l.map(|l| add(x, -S::ONE, l)).transpose()?;
        let du = u.map(|u| add(u, -S::ONE, x)).transpose()?;
        let fixed = l.is_some() && l == u;
        r.activity[i] = if fixed {
            LinearLeastSquaresActivity::Fixed
        } else if dl.is_some_and(|d| d <= threshold) {
            LinearLeastSquaresActivity::Lower
        } else if du.is_some_and(|d| d <= threshold) {
            LinearLeastSquaresActivity::Upper
        } else {
            LinearLeastSquaresActivity::Free
        };
        r.lower[i] = if fixed || dl.is_some_and(|d| d <= threshold) {
            g.max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
        } else {
            <_ as scioxide_core::AlgebraicScalar>::ZERO
        };
        r.upper[i] = if fixed || du.is_some_and(|d| d <= threshold) {
            (-g).max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
        } else {
            <_ as scioxide_core::AlgebraicScalar>::ZERO
        };
        let stationarity = if r.lower[i] > S::ZERO || r.upper[i] > S::ZERO {
            S::ZERO
        } else {
            g
        };
        r.kkt.stationarity = r.kkt.stationarity.max(NumericScalar::abs(stationarity));
        for (distance, multiplier) in [(dl, r.lower[i]), (du, r.upper[i])] {
            if let Some(d) = distance {
                r.kkt.primal_feasibility = r.kkt.primal_feasibility.max(-d);
                let product = if d == S::ZERO || multiplier == S::ZERO {
                    S::ZERO
                } else {
                    Scaled::from_value(d)?
                        .mul(Scaled::from_value(multiplier)?)?
                        .abs()
                        .round()?
                };
                r.kkt.complementarity = r.kkt.complementarity.max(product);
            }
        }
    }
    Ok(())
}
fn gradient_scale<S: RealScalar>(r: &LinearLeastSquaresResult<S>) -> S {
    crate::utils::gradient_scale(r.objective, &r.x)
        .max(r.gradient.iter().fold(S::ONE, |a, &v| a.max(v.abs())))
        .max(r.lower.iter().fold(S::ONE, |a, &v| a.max(v.abs())))
        .max(r.upper.iter().fold(S::ONE, |a, &v| a.max(v.abs())))
}
fn converged<S: RealScalar>(
    r: &LinearLeastSquaresResult<S>,
    o: &LinearLeastSquaresOptions<S>,
) -> Result<bool, SciOxideError> {
    let primal = r.x.iter().fold(S::ONE, |a, &v| a.max(v.abs())).max(S::ONE);
    let gradient = gradient_scale(r);
    Ok(o.gradient.is_satisfied(r.kkt.stationarity, gradient)?
        && o.feasibility
            .is_satisfied(r.kkt.primal_feasibility, primal)?
        && o.complementarity
            .is_satisfied(r.kkt.complementarity, r.objective.abs().max(S::ONE))?)
}
fn accept<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut Storage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let iterations = w.accepted.iterations;
    core::mem::swap(&mut w.accepted, &mut w.trial);
    w.accepted.iterations = iterations;
    w.accepted.rank = w.rank;
    w.observed = true;
    session.accept(PartialResultTrust::Admissible)
}
fn trial<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    o: &LinearLeastSquaresOptions<S>,
    nnls: bool,
    alpha: Option<S>,
    w: &mut Storage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    for (i, v) in w.trial.x.iter_mut().enumerate() {
        *v = project(
            add(
                w.accepted.x[i],
                <_ as scioxide_core::AlgebraicScalar>::ONE,
                w.candidate[i],
            )?,
            p.bound(i, nnls),
        );
        if let Some(alpha) = alpha.filter(|&a| a < S::ONE) {
            let (l, u) = p.bound(i, nnls);
            let d = w.step[i];
            let end = if d < S::ZERO {
                l
            } else if d > S::ZERO {
                u
            } else {
                None
            };
            if let Some(end) = end {
                let ratio = Scaled::from_value(end)?
                    .sub(Scaled::from_value(w.accepted.x[i])?)?
                    .div(Scaled::from_value(d)?)?;
                // Snap all simultaneous hits before evaluating KKT, including
                // when rounding the limiting ratio leaves one endpoint short.
                let admitted = Scaled::from_value(alpha)?.mul(Scaled::from_value(
                    S::ONE + S::EPSILON * S::try_from_f64_exact(8.)?,
                )?)?;
                if ratio.cmp_value(admitted) != core::cmp::Ordering::Greater {
                    *v = end;
                }
            }
        }
    }
    evaluate(p, o, nnls, &mut w.trial, &mut w.provider, session)
}
fn boundary<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    nnls: bool,
    x: &[S],
    d: &[S],
) -> Result<S, SciOxideError> {
    let mut alpha = <_ as scioxide_core::AlgebraicScalar>::ONE;
    for (i, (&x, &d)) in x.iter().zip(d).enumerate() {
        let (l, u) = p.bound(i, nnls);
        let end = if d > <_ as scioxide_core::AlgebraicScalar>::ZERO {
            u
        } else if d < <_ as scioxide_core::AlgebraicScalar>::ZERO {
            l
        } else {
            None
        };
        if let Some(end) = end {
            let distance = Scaled::from_value(end)?.sub(Scaled::from_value(x)?)?;
            let ratio = distance.div(Scaled::from_value(d)?)?;
            if ratio.cmp_value(Scaled::from_value(alpha)?) == core::cmp::Ordering::Less {
                alpha = ratio.round()?.max(S::ZERO);
            }
        }
    }
    Ok(alpha)
}
fn active_set<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    o: &LinearLeastSquaresOptions<S>,
    nnls: bool,
    w: &mut Storage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<OptimizeTermination, SciOxideError> {
    // A feasible warm start from the clipped unconstrained solution avoids
    // admitting O(n) columns one at a time when many coordinates are positive.
    // Subsequent Lawson–Hanson/BVLS steps release one dual violation and retain
    // the updated factors. The projected zero remains the first accepted point.
    let mut initialize = true;
    let mut resolve = true;
    for (j, passive) in w.passive.iter_mut().enumerate() {
        let (l, u) = p.bound(j, nnls);
        *passive = l.is_none() || l != u;
    }
    for iteration in 0..o.iterations.get() {
        if converged(&w.accepted, o)? {
            return Ok(OptimizeTermination::Converged);
        }
        if let Some(stop) = session.trial_iteration()? {
            return Ok(stop);
        }
        w.accepted.iterations = iteration + 1;
        let resolving = resolve;
        let initializing = initialize;
        if !resolve {
            let mut entering = None;
            let mut best = S::ZERO;
            for (j, &g) in w.accepted.gradient.iter().enumerate() {
                if w.passive[j] || w.blocked[j] {
                    continue;
                }
                let (l, u) = p.bound(j, nnls);
                if l.is_some() && l == u {
                    continue;
                }
                let x = w.accepted.x[j];
                let violation = if l == Some(x) {
                    -g
                } else if u == Some(x) {
                    g
                } else {
                    NumericScalar::abs(g)
                };
                if violation > best {
                    best = violation;
                    entering = Some(j);
                }
            }
            let Some(j) = entering else {
                if w.dense.as_mut().is_some_and(|d| d.active.invalidate()) {
                    resolve = true;
                    session.reject()?;
                    continue;
                }
                return Ok(OptimizeTermination::Stagnation);
            };
            w.passive[j] = true;
        }
        super::linear::step(p, o, w, session)?;
        if let Some(stop) = session.termination()? {
            return Ok(stop);
        }
        let alpha = if initialize {
            S::ONE
        } else {
            boundary(p, nnls, &w.accepted.x, &w.step)?
        };
        if alpha == S::ONE {
            w.candidate.copy_from_slice(&w.step);
        } else if alpha == S::ZERO {
            w.candidate.fill(S::ZERO);
        } else {
            for (v, &d) in w.candidate.iter_mut().zip(&w.step) {
                *v = add(S::ZERO, alpha, d)?;
            }
        }
        trial(p, o, nnls, (!initialize).then_some(alpha), w, session)?;
        let changed = w.trial.x != w.accepted.x;
        let mut removed = false;
        for (j, passive) in w.passive.iter_mut().enumerate() {
            if !*passive {
                continue;
            }
            let (l, u) = p.bound(j, nnls);
            let x = w.trial.x[j];
            let hit =
                (w.step[j] < S::ZERO && l == Some(x)) || (w.step[j] > S::ZERO && u == Some(x));
            if hit {
                *passive = false;
                removed = true;
                // Clipping the warm start is not a post-pricing bounce.
                w.blocked[j] = !changed && !initializing;
            }
        }
        let gradient_limit = o.gradient.threshold(gradient_scale(&w.trial))?;
        resolve = removed
            || w.passive
                .iter()
                .zip(&w.trial.gradient)
                .any(|(&passive, &g)| passive && NumericScalar::abs(g) > gradient_limit);
        initialize = false;
        if !changed {
            if !removed && resolving && !resolve {
                // A solved passive subproblem (including an empty one) returns
                // to pricing, even when its solution did not move the point.
                session.reject()?;
                continue;
            }
            if !removed {
                if w.dense.as_mut().is_some_and(|d| d.active.invalidate()) {
                    resolve = true;
                    session.reject()?;
                    continue;
                }
                return Ok(OptimizeTermination::Stagnation);
            }
            session.reject()?;
            continue;
        }
        accept(w, session)?;
        w.blocked.fill(false);
    }
    Ok(if converged(&w.accepted, o)? {
        OptimizeTermination::Converged
    } else {
        OptimizeTermination::IterationLimit
    })
}
/// Reject a workspace prepared for another shape, mode or backend.
pub(super) fn check_workspace<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    nnls_mode: bool,
    w: &Storage<S>,
) -> Result<(), SciOxideError> {
    shape(p.cols(), w.accepted.x.len())?;
    shape(p.rows(), w.accepted.residual.len())?;
    if w.nnls != nnls_mode
        || (p.rows() > 0 && p.cols() > 0 && w.dense.is_some() != p.dense().is_some())
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            "workspace backend differs",
        ));
    }
    let needed = p
        .operator()
        .apply_workspace_len()?
        .max(p.operator().apply_adjoint_workspace_len()?);
    if needed > w.provider.len() {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            "operator scratch exceeds prepared workspace",
        ));
    }
    Ok(())
}
pub(super) fn solve<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    o: &LinearLeastSquaresOptions<S>,
    nnls_mode: bool,
    output: &mut LinearLeastSquaresResult<S>,
    w: &mut Storage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    validate(p, nnls_mode)?;
    check_workspace(p, nnls_mode, w)?;
    w.observed = false;
    w.rank = None;
    w.passive.fill(false);
    w.blocked.fill(false);
    w.accepted.iterations = 0;
    for (i, x) in w.trial.x.iter_mut().enumerate() {
        *x = project(
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
            p.bound(i, nnls_mode),
        );
    }
    session.check()?;
    evaluate(p, o, nnls_mode, &mut w.trial, &mut w.provider, session)?;
    accept(w, session)?;
    w.accepted.iterations = 0;
    if let (Some(d), Some(a)) = (w.dense.as_mut(), p.dense()) {
        d.active.reset(a, p.rhs)?;
    }
    let outcome = active_set(p, o, nnls_mode, w, session);
    let termination = match outcome {
        Ok(t) => t,
        Err(e) => {
            session.record_execution_stop(&e);
            if let Some(t) = session.stopped() {
                t
            } else {
                return Err(e);
            }
        }
    };
    let termination = session.termination()?.unwrap_or(termination);
    w.accepted.termination = termination;
    if termination == OptimizeTermination::Converged {
        session.assess(PartialResultTrust::ResidualChecked)?;
    }
    w.accepted.counts = session.accounting()?;
    w.accepted.algorithm = Some(session.algorithm()?);
    w.accepted.trust = crate::attempt::trust(termination.is_success());
    output.try_copy_from(&w.accepted)
}
