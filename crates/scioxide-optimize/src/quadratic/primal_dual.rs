//! Predictor/corrector QP with native bound elimination in the augmented system.
#![allow(clippy::needless_range_loop)]
use super::{
    common::*,
    model::{Model, Normal},
};
use crate::context::SolveSession;
use crate::utils::{dot, trial_component};
use crate::{OptimalityTolerances, OptimizeTermination as Stop, ProgrammingResult, QpProblem};
use scioxide_core::backend::{checked_add, try_copy_slice, try_filled};
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{checked_float_quotient as divide, checked_sum};
use scioxide_linalg::Matrix;

#[allow(clippy::too_many_arguments)]
pub(super) fn solve<S: RealScalar + NumericScalar<Real = S>>(
    problem: &QpProblem<S>,
    h: &[S],
    tol: OptimalityTolerances<S>,
    limit: Budget,
    warm: Option<&super::QpWarmStart<S>>,
    provider: Option<&mut (dyn crate::KktProvider<S> + Send)>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<ProgrammingResult<S>, SciOxideError> {
    let n = problem.dimension();
    let scale = h
        .iter()
        .chain(problem.linear_objective())
        .fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)));
    let scale = if scale == S::ZERO { S::ONE } else { scale };
    let mut model_h = try_copy_slice(h, scioxide_core::quantity!("qp_scaled_hessian"))?;
    let mut model_c = try_copy_slice(
        problem.linear_objective(),
        scioxide_core::quantity!("qp_scaled_linear_term"),
    )?;
    for v in model_h.iter_mut().chain(&mut model_c) {
        *v = divide(*v, scale)?;
    }
    let mut model = Model::new(problem)?;
    if !model.reduce_equalities(n, tol, session)? {
        let mut x = try_filled(
            n,
            S::ZERO,
            scioxide_core::quantity!("qp_inconsistent_point"),
        )?;
        if let Some(b) = problem.bounds() {
            project_to_bounds(&mut x, b.as_slice());
        }
        return stopped_result(problem, h, x, 0, Stop::SuspectedInfeasible);
    }
    let phase = model.initial(problem, tol, limit, warm, session)?;
    let mut x = try_copy_slice(
        &phase.x()[..n],
        scioxide_core::quantity!("qp_initial_point"),
    )?;
    if !phase.is_success() || !model.feasible(&x, tol)? {
        let mut result = stopped_result(
            problem,
            h,
            x,
            phase.iterations(),
            if phase.is_success() {
                Stop::SuspectedInfeasible
            } else {
                phase.termination()
            },
        )?;
        result.message = alloc::format!("QP Phase I terminated with {}", phase.termination());
        return Ok(result);
    }
    let m = model.rows.len();
    let mut general = scioxide_core::backend::try_with_capacity(
        model.general,
        scioxide_core::quantity!("qp_augmented_rows"),
    )?;
    general.extend((0..model.general).filter(|&i| model.rows[i].enabled));
    let order = checked_add(
        n,
        general.len(),
        scioxide_core::quantity!("qp_augmented_order"),
    )?;
    let mut slack = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_slack"))?;
    let mut dual = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_dual"))?;
    let mut inequalities = 0usize;
    for (i, row) in model.rows.iter().enumerate() {
        if row.enabled && !row.equality {
            slack[i] = (row.rhs - row.value(&x)?).max(S::ONE);
            dual[i] = S::ONE;
            inequalities += 1;
        }
    }
    let mut gradient = try_filled(n, S::ZERO, scioxide_core::quantity!("qp_gradient"))?;
    let mut residual = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_primal_residual"))?;
    let mut rhs = try_filled(order, S::ZERO, scioxide_core::quantity!("qp_augmented_rhs"))?;
    let mut direction = try_filled(order, S::ZERO, scioxide_core::quantity!("qp_direction"))?;
    let mut dy = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_dual_direction"))?;
    let mut ds = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_slack_direction"))?;
    let mut centering = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_centering"))?;
    let mut affine_s = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_affine_slack"))?;
    let mut affine_y = try_filled(m, S::ZERO, scioxide_core::quantity!("qp_affine_dual"))?;
    let mut trial = try_filled(n, S::ZERO, scioxide_core::quantity!("qp_trial_point"))?;
    let mut matrix = Matrix::zeros(order, order)?;
    let mut factor = crate::kkt::Factor::new(
        provider,
        order,
        Some(scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
    )?;
    for iteration in phase.iterations()..limit.get() {
        if let Some(stop) = session.trial_iteration()? {
            return stopped_result(problem, h, x, iteration, stop);
        }
        compute_gradient_into(&model_h, &model_c, &x, &mut gradient)?;
        let mut gradient_scale = gradient
            .iter()
            .fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)));
        let mut feasible = true;
        for (i, row) in model.rows.iter().enumerate() {
            if !row.enabled {
                continue;
            }
            residual[i] = checked_sum(&[row.value(&x)?, -row.rhs, slack[i]])?;
            feasible &= tol
                .feasibility()
                .is_satisfied(NumericScalar::abs(residual[i]), row.scale(&x)?)?;
            for (j, g) in gradient.iter_mut().enumerate() {
                let contribution = dual[i] * row.coefficient(j);
                gradient_scale = gradient_scale.max(NumericScalar::abs(contribution));
                *g = checked_sum(&[*g, contribution])?;
            }
        }
        let mu = if inequalities == 0 {
            S::ZERO
        } else {
            divide(dot(&slack, &dual)?, S::try_from_usize_exact(inequalities)?)?
        };
        let stationarity = gradient
            .iter()
            .fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)));
        let value = eval_objective(&model_h, &model_c, &x, n)?;
        if feasible
            && stationarity
                <= objective_threshold(tol.gradient().tolerance(), gradient_scale, scale)?
            && mu <= objective_threshold(tol.gap().tolerance(), NumericScalar::abs(value), scale)?
        {
            let assessment = model.assess(problem, &x, &dual, scale, tol, session)?;
            if assessment.accepted {
                return Ok(ProgrammingResult {
                    fun: eval_objective(h, problem.linear_objective(), &x, n)?,
                    x,
                    iterations: iteration,
                    termination: Stop::Converged,
                    diagnostics: assessment.diagnostics,
                    evidence: assessment.evidence,
                    verification: assessment.report,
                    ..ProgrammingResult::empty()
                });
            }
        }
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut matrix, order)?;
        assembly.symmetric(0, n, |i, j| Ok(model_h[i * n + j]))?;
        assembly.coupling(n, 0, general.len(), n, |i, j| {
            Ok(model.rows[general[i]].coefficient(j))
        })?;
        assembly.diagonal(n, general.len(), |i| {
            let row = general[i];
            if model.rows[row].equality {
                Ok(S::ZERO)
            } else {
                Ok(-divide(slack[row], dual[row])?)
            }
        })?;
        for (i, row) in model.rows.iter().enumerate().skip(model.general) {
            if row.enabled
                && let Normal::Coordinate { variable, .. } = row.normal
            {
                if row.equality {
                    assembly.fix(variable)?;
                } else {
                    assembly.shift(variable, divide(dual[i], slack[i])?)?;
                }
            }
        }
        factor.invalidate();
        let mut sigma = S::ZERO;
        for correction in [false, true] {
            for j in 0..n {
                rhs[j] = -gradient[j];
            }
            for (i, row) in model.rows.iter().enumerate() {
                if !row.enabled || row.equality {
                    centering[i] = S::ZERO;
                    continue;
                }
                centering[i] = checked_sum(&[
                    sigma * mu,
                    -slack[i] * dual[i],
                    if correction {
                        -affine_s[i] * affine_y[i]
                    } else {
                        S::ZERO
                    },
                ])?;
            }
            for (k, &i) in general.iter().enumerate() {
                rhs[n + k] = if model.rows[i].equality {
                    -residual[i]
                } else {
                    checked_sum(&[-residual[i], -divide(centering[i], dual[i])?])?
                };
            }
            for (i, row) in model.rows.iter().enumerate().skip(model.general) {
                if row.enabled
                    && let Normal::Coordinate { variable, sign } = row.normal
                    && !row.equality
                {
                    rhs[variable] = checked_sum(&[
                        rhs[variable],
                        -sign
                            * divide(
                                checked_sum(&[centering[i], dual[i] * residual[i]])?,
                                slack[i],
                            )?,
                    ])?;
                }
            }
            for (i, row) in model.rows.iter().enumerate().skip(model.general) {
                if row.enabled
                    && row.equality
                    && let Normal::Coordinate { variable, sign } = row.normal
                {
                    let step = -residual[i] / sign;
                    for j in 0..n {
                        if j != variable {
                            rhs[j] = checked_sum(&[rhs[j], -model_h[j * n + variable] * step])?;
                        }
                    }
                    for (k, &g) in general.iter().enumerate() {
                        rhs[n + k] = checked_sum(&[
                            rhs[n + k],
                            -model.rows[g].coefficient(variable) * step,
                        ])?;
                    }
                    rhs[variable] = step;
                }
            }
            match factor.solve(&matrix, &rhs, &mut direction, session) {
                Ok(()) => {}
                Err(_) if session.stopped().is_some() => {
                    return stopped_result(
                        problem,
                        h,
                        x,
                        iteration,
                        session.stopped().unwrap_or(Stop::Stagnation),
                    );
                }
                Err(error) => return Err(error),
            }
            dy.fill(S::ZERO);
            for (k, &i) in general.iter().enumerate() {
                dy[i] = direction[n + k];
            }
            for (i, row) in model.rows.iter().enumerate().skip(model.general) {
                if row.enabled && !row.equality {
                    dy[i] = divide(
                        checked_sum(&[
                            centering[i],
                            dual[i] * residual[i],
                            dual[i] * row.value(&direction[..n])?,
                        ])?,
                        slack[i],
                    )?;
                }
            }
            for (i, row) in model.rows.iter().enumerate().skip(model.general) {
                if row.enabled
                    && row.equality
                    && let Normal::Coordinate { variable, sign } = row.normal
                {
                    let mut sum = scioxide_core::RangeAccumulator::new();
                    sum.add(gradient[variable])?;
                    for j in 0..n {
                        sum.add_product(model_h[variable * n + j], direction[j])?;
                    }
                    for (k, other) in model.rows.iter().enumerate() {
                        if k != i {
                            sum.add_product(other.coefficient(variable), dy[k])?;
                        }
                    }
                    dy[i] = -sum.finish()? / sign;
                }
            }
            for (i, row) in model.rows.iter().enumerate() {
                ds[i] = if row.enabled && !row.equality {
                    divide(checked_sum(&[centering[i], -slack[i] * dy[i]])?, dual[i])?
                } else {
                    S::ZERO
                };
            }
            if !correction {
                let alpha = step_length(&model, &slack, &dual, &ds, &dy)?;
                let mut sum = scioxide_core::ProductSum::new();
                for (i, row) in model.rows.iter().enumerate() {
                    if row.enabled && !row.equality {
                        sum.add_product(
                            trial_component(slack[i], alpha, ds[i])?,
                            trial_component(dual[i], alpha, dy[i])?,
                        )?;
                    }
                }
                let ratio = if mu > S::ZERO {
                    divide(sum.finish()?, mu * S::try_from_usize_exact(inequalities)?)?
                        .max(S::ZERO)
                        .min(S::ONE)
                } else {
                    S::ZERO
                };
                sigma = ratio * ratio * ratio;
                affine_s.copy_from_slice(&ds);
                affine_y.copy_from_slice(&dy);
            }
        }
        let alpha = step_length(&model, &slack, &dual, &ds, &dy)? * S::try_from_f64_lossy(0.995)?;
        for j in 0..n {
            trial[j] = trial_component(x[j], alpha, direction[j])?;
        }
        for (i, row) in model.rows.iter().enumerate() {
            affine_s[i] = trial_component(slack[i], alpha, ds[i])?;
            affine_y[i] = trial_component(dual[i], alpha, dy[i])?;
            if row.enabled && !row.equality && (affine_s[i] <= S::ZERO || affine_y[i] <= S::ZERO) {
                return stopped_result(problem, h, x, iteration, Stop::Stagnation);
            }
        }
        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
        x.copy_from_slice(&trial);
        slack.copy_from_slice(&affine_s);
        dual.copy_from_slice(&affine_y);
    }
    stopped_result(problem, h, x, limit.get(), Stop::IterationLimit)
}
fn step_length<S: RealScalar + NumericScalar<Real = S>>(
    model: &Model<S>,
    s: &[S],
    y: &[S],
    ds: &[S],
    dy: &[S],
) -> Result<S, SciOxideError> {
    let mut alpha = S::ONE;
    for (i, row) in model.rows.iter().enumerate() {
        if row.enabled && !row.equality {
            if ds[i] < S::ZERO {
                alpha = alpha.min(divide(-s[i], ds[i])?);
            }
            if dy[i] < S::ZERO {
                alpha = alpha.min(divide(-y[i], dy[i])?);
            }
        }
    }
    Ok(alpha)
}
