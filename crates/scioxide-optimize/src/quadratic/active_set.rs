//! Feasible active-set iterations with updated Hessian-metric QR factors.
#![allow(clippy::needless_range_loop)]
use super::{
    QpSolver,
    active_factor::ActiveFactor,
    common::*,
    model::{Model, Normal},
};
use crate::context::SolveSession;
use crate::utils::{dot, norm2};
use crate::{
    KktProvider, OptimalityTolerances, OptimizeTermination as Stop, ProgrammingResult, QpProblem,
};
use alloc::vec::Vec;
use scioxide_core::backend::{
    checked_add, standard, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{EvaluationRole, NumericScalar, PartialResultTrust, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;

pub(super) struct Storage<S: RealScalar + NumericScalar<Real = S>> {
    factor: ActiveFactor<S>,
    h: Matrix<S>,
    gradient: Vec<S>,
    nonzero: Vec<(usize, S)>,
    pub x: Vec<S>,
    direction: Vec<S>,
    residual: Vec<S>,
    lambda: Vec<S>,
    pub dual: Vec<S>,
    working: Vec<usize>,
    general_positions: Vec<usize>,
    membership: Vec<bool>,
    kkt: Matrix<S>,
    rhs: Vec<S>,
    solution: Vec<S>,
    scale: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> Storage<S> {
    fn new(
        h: &[S],
        x: Vec<S>,
        m: usize,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<Self, SciOxideError> {
        let n = x.len();
        let scale = h.iter().fold(S::ZERO, |a, &v| a.max(NumericScalar::abs(v)));
        let scale = if scale == S::ZERO { S::ONE } else { scale };
        let mut matrix = Matrix::from_vec(
            n,
            n,
            try_copy_slice(h, scioxide_core::quantity!("qp_scaled_hessian"))?,
        )?;
        for v in matrix.as_mut_slice() {
            *v = scioxide_core::checked_float_quotient(*v, scale)?;
        }
        let factor = ActiveFactor::new(&matrix, session)?;
        let order = checked_add(n, n, scioxide_core::quantity!("qp_kkt_order"))?;
        Ok(Self {
            factor,
            h: matrix,
            x,
            gradient: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_gradient"))?,
            nonzero: try_with_capacity(n, scioxide_core::quantity!("qp_nonzero_coordinates"))?,
            direction: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_direction"))?,
            residual: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_null_direction"))?,
            lambda: try_filled(n, S::ZERO, scioxide_core::quantity!("qp_multipliers"))?,
            dual: try_filled(m, S::ZERO, scioxide_core::quantity!("qp_duals"))?,
            working: try_with_capacity(n, scioxide_core::quantity!("qp_working_set"))?,
            general_positions: try_with_capacity(
                n,
                scioxide_core::quantity!("qp_general_working_set"),
            )?,
            membership: try_filled(m, false, scioxide_core::quantity!("qp_membership"))?,
            kkt: Matrix::zeros(order, order)?,
            rhs: try_filled(order, S::ZERO, scioxide_core::quantity!("qp_kkt_rhs"))?,
            solution: try_filled(order, S::ZERO, scioxide_core::quantity!("qp_kkt_solution"))?,
            scale,
        })
    }
    fn insert(
        &mut self,
        index: usize,
        model: &Model<S>,
        t: OptimalityTolerances<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        if session.invoke(EvaluationRole::Factorization, || {
            self.factor.insert(&model.rows[index], t.rank())
        })? {
            self.working.push(index);
            self.membership[index] = true;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn delete(
        &mut self,
        position: usize,
        rank: scioxide_core::RankThreshold<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        session.invoke(EvaluationRole::Factorization, || {
            self.factor.delete(position, rank)
        })?;
        self.membership[self.working.remove(position)] = false;
        Ok(())
    }
    fn kkt_solve(
        &mut self,
        model: &Model<S>,
        factor: &mut crate::kkt::Factor<'_, S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.x.len();
        self.general_positions.clear();
        self.general_positions
            .extend(
                self.working
                    .iter()
                    .enumerate()
                    .filter_map(|(position, &i)| {
                        matches!(model.rows[i].normal, Normal::General(_)).then_some(position)
                    }),
            );
        let k = self.general_positions.len();
        let order = n + k;
        let mut assembly = crate::kkt::assembly::Assembly::new(&mut self.kkt, order)?;
        assembly.symmetric(0, n, |i, j| Ok(self.h.as_slice()[i * n + j]))?;
        assembly.coupling(n, 0, k, n, |i, j| {
            Ok(model.rows[self.working[self.general_positions[i]]].coefficient(j))
        })?;
        self.rhs[..order].fill(S::ZERO);
        for i in 0..n {
            self.rhs[i] = -self.gradient[i];
        }
        for &i in &self.working {
            if let Normal::Coordinate { variable, .. } = model.rows[i].normal {
                assembly.fix(variable)?;
                self.rhs[variable] = S::ZERO;
            }
        }
        factor.solve(
            &self.kkt,
            &self.rhs[..order],
            &mut self.solution[..order],
            session,
        )?;
        self.direction.copy_from_slice(&self.solution[..n]);
        for (j, &position) in self.general_positions.iter().enumerate() {
            self.lambda[position] = self.solution[n + j];
        }
        for (position, &index) in self.working.iter().enumerate() {
            if let Normal::Coordinate { variable, sign } = model.rows[index].normal {
                let mut sum = scioxide_core::RangeAccumulator::new();
                sum.add(self.gradient[variable])?;
                for j in 0..n {
                    sum.add_product(self.h.as_slice()[variable * n + j], self.direction[j])?;
                }
                for &p in &self.general_positions {
                    sum.add_product(
                        model.rows[self.working[p]].coefficient(variable),
                        self.lambda[p],
                    )?;
                }
                self.lambda[position] = -sum.finish()? / sign;
            }
        }
        Ok(())
    }
    fn residual_direction(&mut self, model: &Model<S>) -> Result<(), SciOxideError> {
        let n = self.x.len();
        for i in 0..n {
            let mut sum = scioxide_core::RangeAccumulator::new();
            sum.add(self.gradient[i])?;
            for j in 0..n {
                sum.add_product(self.h.as_slice()[i * n + j], self.direction[j])?;
            }
            for (&row, &y) in self.working.iter().zip(&self.lambda) {
                sum.add_product(model.rows[row].coefficient(i), y)?;
            }
            self.residual[i] = -sum.finish()?;
        }
        Ok(())
    }
    /// Everything called by this loop has been prepared before entry.
    fn iterate(
        &mut self,
        c: &[S],
        model: &Model<S>,
        t: OptimalityTolerances<S>,
        allowance: usize,
        fallback: &mut Option<crate::kkt::Factor<'_, S>>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(usize, Stop), SciOxideError> {
        let n = self.x.len();
        let mut nonprogress = 0usize;
        let mut bland = false;
        for iteration in 0..allowance {
            if let Some(stop) = session.trial_iteration()? {
                return Ok((iteration, stop));
            }
            // Standard arithmetic in the scaled model, with range-safe substitution
            // in triangular kernels and checked original-coordinate certification.
            self.nonzero.clear();
            self.nonzero.extend(
                self.x
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &x)| (x != S::ZERO).then_some((j, x))),
            );
            for i in 0..n {
                self.gradient[i] = scioxide_core::checked_sum(&[
                    scioxide_core::checked_float_quotient(c[i], self.scale)?,
                    if self.nonzero.len() < n / 2 {
                        standard::dot(
                            self.nonzero
                                .iter()
                                .map(|&(j, x)| (self.h.as_slice()[i * n + j], x)),
                        )
                    } else {
                        standard::dot(
                            self.h.as_slice()[i * n..(i + 1) * n]
                                .iter()
                                .zip(&self.x)
                                .map(|(&a, &x)| (a, x)),
                        )
                    },
                ])?;
            }
            if let Some(factor) = fallback {
                self.kkt_solve(model, factor, session)?;
            } else {
                session.invoke(EvaluationRole::LinearSolve, || {
                    self.factor
                        .solve(&self.gradient, &mut self.direction, &mut self.lambda)
                })?;
            }
            let gradient_scale = norm2(&self.gradient)?;
            if !self.factor.positive_definite() {
                self.residual_direction(model)?;
            }
            let recession = if !self.factor.positive_definite()
                && norm2(&self.residual)?
                    > objective_threshold(t.gradient().tolerance(), gradient_scale, self.scale)?
            {
                let dnorm = norm2(&self.residual)?;
                let mut null = true;
                for row in self.h.as_slice().chunks_exact(n) {
                    null &= NumericScalar::abs(dot(row, &self.residual)?)
                        <= t.rank().threshold(norm2(row)? * dnorm)?;
                }
                for &index in &self.working {
                    let row = &model.rows[index];
                    null &= NumericScalar::abs(row.value(&self.residual)?)
                        <= t.rank().threshold(row.norm()? * dnorm)?;
                }
                null && dot(&self.gradient, &self.residual)?
                    < -objective_threshold(
                        t.gradient().tolerance(),
                        gradient_scale * dnorm,
                        self.scale,
                    )?
            } else {
                false
            };
            if recession {
                self.direction.copy_from_slice(&self.residual);
            }

            let pnorm = norm2(&self.direction)?;
            // Test the projected gradient in gradient units, rather than comparing
            // a primal step with a gradient tolerance.
            let projected = NumericScalar::abs(dot(&self.gradient, &self.direction)?);
            let small = !recession
                && (pnorm == S::ZERO
                    || projected
                        <= objective_threshold(
                            t.gradient().tolerance(),
                            gradient_scale,
                            self.scale,
                        )? * pnorm);
            if small {
                let dual_scale = self.lambda[..self.working.len()]
                    .iter()
                    .fold(S::ZERO, |v, &y| v.max(NumericScalar::abs(y)));
                let cutoff =
                    objective_threshold(t.dual_feasibility().tolerance(), dual_scale, self.scale)?;
                let mut remove = None;
                for (position, (&index, &y)) in self.working.iter().zip(&self.lambda).enumerate() {
                    if !model.rows[index].equality
                        && y < -cutoff
                        && remove.is_none_or(|p| {
                            if bland {
                                index < self.working[p]
                            } else {
                                y < self.lambda[p]
                            }
                        })
                    {
                        remove = Some(position);
                    }
                }
                if let Some(position) = remove {
                    self.delete(position, t.rank(), session)?;
                    if let Some(factor) = fallback {
                        factor.invalidate();
                    }
                    nonprogress += 1;
                } else {
                    self.dual.fill(S::ZERO);
                    for (&i, &y) in self.working.iter().zip(&self.lambda) {
                        self.dual[i] = y;
                    }
                    return Ok((iteration + 1, Stop::Converged));
                }
            } else {
                let mut blocker: Option<usize> = None;
                let mut alpha = if recession { S::MAX } else { S::ONE };
                for (i, row) in model.rows.iter().enumerate() {
                    if row.equality || !row.enabled || self.membership[i] {
                        continue;
                    }
                    let slope = row.value(&self.direction)?;
                    if slope <= t.pivot().threshold(row.norm()? * pnorm)? {
                        continue;
                    }
                    let slack = scioxide_core::checked_difference(row.rhs, row.value(&self.x)?)?
                        .max(S::ZERO);
                    let distance = scioxide_core::checked_float_quotient(slack, slope)?;
                    let tie = t.pivot().threshold(alpha.min(S::ONE))?;
                    if distance < alpha
                        || (NumericScalar::abs(distance - alpha) <= tie
                            && blocker.is_some_and(|j| i < j))
                    {
                        alpha = distance;
                        blocker = Some(i);
                    }
                }
                if recession && blocker.is_none() {
                    return Ok((iteration, Stop::SuspectedUnbounded));
                }
                for i in 0..n {
                    self.x[i] = crate::utils::trial_component(self.x[i], alpha, self.direction[i])?;
                }
                session.accept(PartialResultTrust::Admissible)?;
                if alpha * pnorm <= t.pivot().threshold(norm2(&self.x)?)? {
                    nonprogress += 1;
                } else {
                    nonprogress = 0;
                }
                if let Some(i) = blocker {
                    if !self.insert(i, model, t, session)? {
                        // A blocking normal must enlarge the independent working set.
                        // Repeated dependent blockers are numerical stagnation.
                        return Ok((iteration + 1, Stop::Stagnation));
                    }
                    if let Some(factor) = fallback {
                        factor.invalidate();
                    }
                }
                // Native active coordinates are known exactly. Restoring them
                // prevents accumulated null-space roundoff from moving a bound.
                for &i in &self.working {
                    if let Normal::Coordinate { variable, sign } = model.rows[i].normal {
                        self.x[variable] = model.rows[i].rhs / sign;
                    }
                }
            }
            if nonprogress >= 8 {
                bland = true;
            }
        }
        Ok((allowance, Stop::IterationLimit))
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> QpSolver<S> {
    pub(super) fn solve_active_set(
        &self,
        problem: &QpProblem<S>,
        h: &[S],
        storage: &mut Option<Storage<S>>,
        session: &SolveSession<'_, '_, '_>,
        provider: Option<&mut (dyn KktProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let n = problem.dimension();
        let t = self.tolerances;
        let mut model = Model::new(problem)?;
        if !model.reduce_equalities(n, t, session)? {
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
        let phase = model.initial(
            problem,
            t,
            self.max_iterations,
            self.warm_start.as_ref(),
            session,
        )?;
        let x = try_copy_slice(
            &phase.x()[..n],
            scioxide_core::quantity!("qp_initial_point"),
        )?;
        if !phase.is_success() || !model.feasible(&x, t)? {
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
        *storage = Some(match Storage::new(h, x, model.rows.len(), session) {
            Ok(state) => state,
            Err(_) if session.stopped().is_some() => {
                return stopped_result(
                    problem,
                    h,
                    try_copy_slice(
                        &phase.x()[..n],
                        scioxide_core::quantity!("qp_stopped_point"),
                    )?,
                    phase.iterations(),
                    session.stopped().unwrap_or(Stop::Stagnation),
                );
            }
            Err(error) => return Err(error),
        });
        let state = storage
            .as_mut()
            .ok_or_else(|| SciOxideError::singular(None))?;
        let outcome = (|| {
            let needs_fallback = provider.is_some() || !state.factor.positive_definite();
            let mut fallback = if needs_fallback {
                let order = state.kkt.rows();
                let builtin = provider.is_none();
                let mut factor = crate::kkt::Factor::new(
                    provider,
                    order,
                    Some(scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
                )?
                .with_normwise_residual();
                // Admit the largest spectral fallback before any iteration. Later
                // rank changes and working-set orders reuse these capacities.
                if builtin {
                    factor.factor_lent(None, &state.kkt, session)?;
                }
                factor.invalidate();
                Some(factor)
            } else {
                None
            };
            let mut candidates = try_with_capacity(
                model.rows.len(),
                scioxide_core::quantity!("qp_initial_working_set"),
            )?;
            candidates.extend(
                model
                    .rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, row)| (row.enabled && row.equality).then_some(i)),
            );
            for (i, row) in model.rows.iter().enumerate() {
                if row.enabled
                    && !row.equality
                    && NumericScalar::abs(scioxide_core::checked_difference(
                        row.value(&state.x)?,
                        row.rhs,
                    )?) <= t.feasibility().threshold(row.scale(&state.x)?)?
                {
                    candidates.push(i);
                }
            }
            if state
                .factor
                .initialize(&model, &candidates, t.rank(), session)?
            {
                state.working.extend_from_slice(&candidates);
                for &i in &candidates {
                    state.membership[i] = true;
                }
            } else {
                // Equalities are permanent, and enter before active inequalities.
                for (i, row) in model.rows.iter().enumerate() {
                    if row.enabled && row.equality {
                        state.insert(i, &model, t, session)?;
                    }
                }
                for (i, row) in model.rows.iter().enumerate() {
                    if row.enabled
                        && !row.equality
                        && NumericScalar::abs(scioxide_core::checked_difference(
                            row.value(&state.x)?,
                            row.rhs,
                        )?) <= t.feasibility().threshold(row.scale(&state.x)?)?
                    {
                        state.insert(i, &model, t, session)?;
                    }
                }
            }
            state.iterate(
                problem.linear_objective(),
                &model,
                t,
                self.max_iterations.get().saturating_sub(phase.iterations()),
                &mut fallback,
                session,
            )
        })();
        let (iterations, mut termination) = match outcome {
            Ok(value) => value,
            Err(_) if session.stopped().is_some() => (
                session
                    .accounting()?
                    .total()?
                    .iterations()
                    .saturating_sub(phase.iterations()),
                session.stopped().unwrap_or(Stop::Stagnation),
            ),
            Err(error) => return Err(error),
        };
        let x = try_copy_slice(&state.x, scioxide_core::quantity!("qp_result"))?;
        let mut result =
            stopped_result(problem, h, x, phase.iterations() + iterations, termination)?;
        if termination == Stop::Converged {
            let assessment =
                model.assess(problem, &result.x, &state.dual, state.scale, t, session)?;
            termination = if assessment.accepted {
                Stop::Converged
            } else {
                Stop::Stagnation
            };
            result.diagnostics = assessment.diagnostics;
            result.evidence = assessment.evidence;
            result.verification = assessment.report;
        } else if termination == Stop::SuspectedUnbounded {
            // Repair roundoff in the candidate null direction without weakening
            // the verifier's exact original-Hessian cancellation requirement.
            for _ in 0..2 {
                for row in h.chunks_exact(n) {
                    let pivot = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            NumericScalar::abs(**a)
                                .partial_cmp(&NumericScalar::abs(**b))
                                .unwrap_or(core::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);
                    if let Some(j) = pivot
                        && row[j] != S::ZERO
                    {
                        let mut sum = scioxide_core::RangeAccumulator::new();
                        for (k, &a) in row.iter().enumerate() {
                            if k != j {
                                sum.add_product(a, state.direction[k])?;
                            }
                        }
                        state.direction[j] =
                            scioxide_core::checked_float_quotient(-sum.finish()?, row[j])?;
                    }
                }
            }
            let evidence = crate::verification::OwnedProgrammingCertificate::Unboundedness(
                try_copy_slice(&state.direction, scioxide_core::quantity!("qp_ray"))?,
            );
            let view = crate::QpVerificationProblem::from_problem(problem);
            let mut work = crate::VerificationWorkspace::for_qp(&view)?;
            let report = crate::verification::kernel::qp_session(
                &view,
                evidence.borrow(&result.x),
                t,
                &mut work,
                Some(session),
            )?;
            if report.is_accepted() {
                termination = Stop::Unbounded;
            }
            result.evidence = Some(evidence);
            result.verification = Some(report);
        }
        result.termination = termination;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn session(run: impl FnOnce(&SolveSession<'_, '_, '_>)) {
        crate::context::sequential_with_budgets(
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(1982),
                scioxide_core::algorithm_id!("optimize", "QpAllocationTest", 1),
                2,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |context| {
                run(&SolveSession::new(
                    context,
                    scioxide_core::argument!("context"),
                ));
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn working_set_changes_and_semidefinite_iterations_execute() {
        for h in [vec![2.0, 0., 0., 2.], vec![0.; 4]] {
            let p = QpProblem::new(Matrix::from_vec(2, 2, h.clone()).unwrap(), vec![-2., -6.])
                .unwrap()
                .with_bounds(scioxide_core::Bounds::new(vec![(Some(0.), Some(2.)); 2]).unwrap())
                .unwrap();
            let model = Model::new(&p).unwrap();
            let t = OptimalityTolerances::programming_default();
            session(|session| {
                let mut state = Storage::new(&h, vec![0.; 2], model.rows.len(), session).unwrap();
                let mut fallback = if state.factor.positive_definite() {
                    None
                } else {
                    let mut factor = crate::kkt::Factor::new(
                        None,
                        4,
                        Some(scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
                    )
                    .unwrap();
                    factor.factor_lent(None, &state.kkt, session).unwrap();
                    factor.invalidate();
                    Some(factor)
                };
                state.insert(0, &model, t, session).unwrap();
                state.insert(2, &model, t, session).unwrap();
                let outcome =
                    state.iterate(p.linear_objective(), &model, t, 100, &mut fallback, session);
                assert_eq!(outcome.unwrap().1, Stop::Converged);
                assert!(state.x[0] >= 1. - 1e-8 && (state.x[1] - 2.).abs() < 1e-8);
            });
        }
    }

    #[test]
    fn qr_insertion_deletion_reentry_and_dependence_preserve_kkt_equations() {
        let p = QpProblem::new(
            Matrix::from_vec(3, 3, vec![2., 0., 0., 0., 3., 0., 0., 0., 5.]).unwrap(),
            vec![1., -2., 3.],
        )
        .unwrap()
        .with_constraints(
            crate::LinearConstraints::new(
                Matrix::from_vec(3, 3, vec![1., 1., 0., 0., 1., 1., 2., 2., 0.]).unwrap(),
                vec![0.; 3],
                vec![crate::LinearConstraintSense::LessEqual; 3],
            )
            .unwrap(),
        )
        .unwrap();
        let model = Model::new(&p).unwrap();
        session(|session| {
            let h = Matrix::from_vec(3, 3, vec![2., 0., 0., 0., 3., 0., 0., 0., 5.]).unwrap();
            let mut factor = ActiveFactor::new(&h, session).unwrap();
            let mut active = Vec::with_capacity(3);
            let mut step = [0.; 3];
            let mut lambda = [0.; 3];
            let rank = OptimalityTolerances::programming_default().rank();
            {
                for (add, index) in [
                    (true, 0),
                    (true, 1),
                    (false, 0),
                    (true, 0),
                    (false, 1),
                    (true, 0),
                ] {
                    if add {
                        if factor.insert(&model.rows[index], rank).unwrap() {
                            active.push(index);
                        }
                    } else {
                        factor.delete(index, rank).unwrap();
                        active.remove(index);
                    }
                    factor
                        .solve(p.linear_objective(), &mut step, &mut lambda)
                        .unwrap();
                    for &i in &active {
                        assert!(model.rows[i].value(&step).unwrap().abs() < 1e-10);
                    }
                    for i in 0..3 {
                        let r = p.linear_objective()[i]
                            + h.as_slice()[i * 3 + i] * step[i]
                            + active
                                .iter()
                                .zip(&lambda)
                                .map(|(&r, &y)| model.rows[r].coefficient(i) * y)
                                .sum::<f64>();
                        assert!(r.abs() < 1e-10, "{r}");
                    }
                }
                if !active.contains(&0) {
                    assert!(factor.insert(&model.rows[0], rank).unwrap());
                }
                assert!(!factor.insert(&model.rows[2], rank).unwrap());
            };
        });
    }

    #[test]
    fn native_bound_border_and_deletion_updates_preserve_original_kkt_equations() {
        let h = alloc::vec![4., 1., 0.5, 1., 3., 1., 0.5, 1., 2.];
        let p = QpProblem::new(
            Matrix::from_vec(3, 3, h.clone()).unwrap(),
            alloc::vec![1., -2., 3.],
        )
        .unwrap()
        .with_bounds(scioxide_core::Bounds::new(alloc::vec![(Some(0.),Some(2.));3]).unwrap())
        .unwrap()
        .with_constraints(
            crate::LinearConstraints::new(
                Matrix::from_vec(1, 3, alloc::vec![1.;3]).unwrap(),
                alloc::vec![0.],
                alloc::vec![crate::LinearConstraintSense::LessEqual],
            )
            .unwrap(),
        )
        .unwrap();
        let model = Model::new(&p).unwrap();
        let rank = OptimalityTolerances::programming_default().rank();
        session(|session| {
            let mut factor =
                ActiveFactor::new(&Matrix::from_vec(3, 3, h.clone()).unwrap(), session).unwrap();
            let mut active = Vec::with_capacity(3);
            active.extend([0, 1, 5]);
            assert!(factor.initialize(&model, &active, rank, session).unwrap());
            let mut step = [0.; 3];
            let mut lambda = [0.; 3];
            for (add, index) in [(false, 1), (true, 3), (false, 0), (true, 0), (false, 1)] {
                if add {
                    assert!(factor.insert(&model.rows[index], rank).unwrap());
                    active.push(index);
                } else {
                    factor.delete(index, rank).unwrap();
                    active.remove(index);
                }
                factor
                    .solve(p.linear_objective(), &mut step, &mut lambda)
                    .unwrap();
                for &i in &active {
                    assert!(model.rows[i].value(&step).unwrap().abs() < 1e-10);
                }
                for i in 0..3 {
                    let residual = p.linear_objective()[i]
                        + h[i * 3..(i + 1) * 3]
                            .iter()
                            .zip(step)
                            .map(|(&a, x)| a * x)
                            .sum::<f64>()
                        + active
                            .iter()
                            .zip(lambda)
                            .map(|(&j, y)| model.rows[j].coefficient(i) * y)
                            .sum::<f64>();
                    assert!(residual.abs() < 1e-10, "{residual}");
                }
            }
        });
    }
}
