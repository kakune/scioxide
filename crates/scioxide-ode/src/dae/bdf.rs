//! Variable-step residual BDF and implicit-trapezoid stepping. A proposal changes the
//! accepted history only in `accept_bdf`; replay supplies its recorded geometry.
use super::*;
use crate::multistep::{
    BdfFormula,
    bdf::LocalError,
    polynomial::{CAPACITY, Functional, Weights, coordinate},
};

pub(crate) const MAX_ORDER: usize = 5;
// Reserve three quarters of the state-error budget for error propagation and
// nonlinear solve effects. This conservative LTE margin is not a global bound.
const LOCAL_ERROR_FRACTION: f64 = 0.25;
// DASSL's default first step (Brenan, Campbell & Petzold 1996, §5.4): one thousandth of
// the interval, and a first increment `h ‖ẋ₀‖` of at most half the error tolerance.
const DEFAULT_STEP_FRACTION: f64 = 1e-3;
const FIRST_INCREMENT: f64 = 0.5;
// The starting step of Hairer, Nørsett & Wanner (1993, II.4) for a start whose norms carry
// no scale, also the ODE initial-step fallback.
const UNSCALED_FIRST_STEP: f64 = 1e-6;
/// Normalized differential and algebraic local-error norms.
type Norms<S> = (S, S);
/// One-step residual formula of a recorded trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Formula {
    /// Variable-order residual BDF of orders one through `maximum_order`.
    Bdf { maximum_order: usize },
    /// Fixed-step implicit trapezoid (Crank–Nicolson): `F(t1, y1, v1) = 0` with
    /// `v1 = 2 (y1 - y0) / h - v0`, the rate `v0` carried from the previous step.
    Trapezoid,
}

/// Recorded equation geometry; integer metadata is never encoded as a scalar.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BdfStep<S: RealScalar> {
    pub order: usize,
    pub history: usize,
    pub alpha: [S; CAPACITY],
    pub nodes: [S; CAPACITY],
}
impl<S: RealScalar> BdfStep<S> {
    /// Geometry of an implicit-trapezoid step: endpoint node one, start node zero.
    pub(crate) fn trapezoid() -> Self {
        let mut nodes = [S::ZERO; CAPACITY];
        nodes[0] = S::ONE;
        Self {
            nodes,
            ..Self::default()
        }
    }
}
impl<S: RealScalar> Default for BdfStep<S> {
    fn default() -> Self {
        Self {
            order: 1,
            history: 0,
            alpha: [S::ZERO; CAPACITY],
            nodes: [S::ZERO; CAPACITY],
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) struct Trial<S: RealScalar> {
    pub geometry: BdfStep<S>,
    pub errors: (S, S),
    pub converged: bool,
    /// The stage converged only on an exact-factor or fresh-block Newton retry.
    retried: bool,
    next_order: usize,
}

impl<T: NumericScalar> DaeStorage<T>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    /// Default first step magnitude `min(|end − start|/1000, 0.5/‖ẋ₀‖)` from the
    /// accepted initial pair, at least the distance to the adjacent time and the smallest
    /// normal magnitude, which keeps the stage coefficient `α₀/h` finite. `‖ẋ₀‖` is the
    /// differential local-error norm of the initial differential rates with scales
    /// `|x₀|`; a norm beyond the scalar range is infinite. Components with an exact-zero
    /// threshold carry no scale and contribute zero; one that moves bounds the step by
    /// the unscaled starting step `10⁻⁶` instead. An empty interval takes no step.
    pub(crate) fn first_step(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        start: T::Real,
        end: T::Real,
    ) -> Result<T::Real, SciOxideError> {
        if start == end {
            return Ok(T::Real::ZERO);
        }
        let tolerance = plan.1.numerics().differential_error();
        let count = plan.0.partition().differential();
        let differential = plan
            .0
            .partition()
            .variables()
            .iter()
            .enumerate()
            .filter_map(|(i, &differential)| differential.then_some(i));
        let mut unscaled = false;
        for (k, i) in differential.enumerate() {
            self.scales[k] = self.accepted[i].abs();
            self.errors[k] = self.accepted_derivative[i].abs();
            if tolerance.threshold(k, count, self.scales[k])? == T::Real::ZERO {
                unscaled |= self.errors[k] != T::Real::ZERO;
                self.errors[k] = T::Real::ZERO;
            }
        }
        let rate = match tolerance.normalized_norm(&self.errors[..count], &self.scales[..count]) {
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                T::Real::INFINITY
            }
            rate => rate?,
        };
        let mut step = (end - start).abs() * real::<T::Real>(DEFAULT_STEP_FRACTION)?;
        if unscaled {
            step = step.min(real(UNSCALED_FIRST_STEP)?);
        }
        if rate * step > real::<T::Real>(FIRST_INCREMENT)? {
            step = real::<T::Real>(FIRST_INCREMENT)? / rate;
        }
        Ok(step
            .max(crate::safety::adjacent_time_step(start, end)?.abs())
            .max(T::Real::MIN_POSITIVE))
    }
    pub(crate) fn reset_bdf(&mut self, time: T::Real) {
        self.linear.invalidate_factor();
        self.history_count = 0;
        self.history_times.fill(time);
        self.order = 1;
        self.residence = 0;
        self.rejected = 0;
        self.limited = false;
        self.controller.reset();
    }
    /// Prepare a residual BDF trial from the last accepted pair.
    pub(crate) fn propose<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        maximum: usize,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Trial<T::Real>, Fault> {
        let q = self.order.min(maximum).min(self.history_count + 1);
        let mut old = [T::Real::ZERO; CAPACITY];
        for (i, node) in old.iter_mut().enumerate().skip(1).take(self.history_count) {
            *node = coordinate(self.history_times[i - 1], time, step)?;
        }
        let (alpha, _) =
            crate::multistep::bdf::coefficients(q, BdfFormula::Bdf, &old, self.history_count)?;
        let mut nodes = [T::Real::ONE; CAPACITY];
        nodes[1..=self.history_count + 1].copy_from_slice(&old[..=self.history_count]);
        let geometry = BdfStep {
            order: q,
            history: self.history_count,
            alpha,
            nodes,
        };
        let n = self.dimension();
        let predictor = if self.history_count == 0 {
            None
        } else {
            Some(Weights::new(
                &old[..=q.min(self.history_count)],
                Functional::Value(T::Real::ONE),
            )?)
        };
        for i in 0..n {
            self.candidate[i] = if let Some(predictor) = &predictor {
                predictor.affine(T::ZERO, T::Real::ONE, |j| {
                    if j == 0 {
                        self.accepted[i]
                    } else {
                        self.history[(j - 1) * n + i]
                    }
                })?
            } else {
                crate::safety::state_increment(self.accepted[i], step, self.accepted_derivative[i])?
            };
        }
        self.bdf_derivative(step, geometry, None)?;
        let (outcome, retried) =
            self.newton(plan, problem, time + step, alpha[0] / step, false, context)?;
        let mut converged = matches!(outcome, NewtonOutcome::Converged(_));
        let mut errors = (T::Real::ZERO, T::Real::ZERO);
        let mut estimates = [T::Real::ZERO; 3];
        if converged {
            for (slot, order) in [(0, q.saturating_sub(1)), (2, q + 1), (1, q)] {
                if order == 0 || order > maximum || order > self.history_count + 1 {
                    continue;
                }
                let norms = if self.history_count == 0 {
                    self.startup_errors(plan, problem, time + step, step, alpha[0], context)?
                } else {
                    Some(self.bdf_errors(plan, order, &old)?)
                };
                let Some(norms) = norms else {
                    converged = false;
                    break;
                };
                estimates[slot] = norms.0.max(norms.1);
                if slot == 1 {
                    errors = norms;
                }
            }
        }
        // A failed corrector keeps its order: a lower order predicts worse and
        // converges no better at the reduced step.
        let next_order = if converged {
            crate::multistep::select_order(
                q,
                maximum,
                self.history_count,
                self.residence,
                estimates,
            )?
        } else {
            q
        };
        Ok(Trial {
            geometry,
            errors,
            converged,
            retried,
            next_order,
        })
    }
    /// Prepare an implicit-trapezoid trial from the last accepted pair. The only
    /// acceptance test is the original residual check of [`Self::assess_bdf`].
    pub(crate) fn propose_trapezoid<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Trial<T::Real>, Fault> {
        let converged = self.trapezoid(plan, problem, time, step, context)?;
        Ok(Trial {
            geometry: BdfStep::trapezoid(),
            errors: (T::Real::ZERO, T::Real::ZERO),
            converged,
            retried: false,
            next_order: 1,
        })
    }
    /// Replay a recorded implicit-trapezoid equation from its restart pair
    /// `initial[..2n]` with the original predictor and Newton iteration.
    pub(crate) fn replay_trapezoid<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        initial: &[T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.linear.invalidate();
        let n = self.dimension();
        self.accepted.copy_from_slice(&initial[..n]);
        self.accepted_derivative.copy_from_slice(&initial[n..2 * n]);
        if !self.trapezoid(plan, problem, time, step, context)? {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::ReplayMismatch,
                scioxide_core::ConvergenceDiagnostics::new(),
                "recorded trapezoid equation no longer converges",
            ));
        }
        Ok(())
    }
    /// Solve `F(t + h, y1, 2 (y1 - y0) / h - v0) = 0` from the explicit Euler
    /// predictor. Newton moves the rate with the state at the inverse step `2 / h`,
    /// so every iterate keeps the trapezoid rate relation.
    fn trapezoid<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, Fault> {
        for i in 0..self.dimension() {
            let (state, rate) = (self.accepted[i], self.accepted_derivative[i]);
            self.candidate[i] = crate::safety::state_increment(state, step, rate)?;
            self.candidate_derivative[i] = crate::safety::stage_derivative(
                step,
                [
                    (T::Real::TWO, self.candidate[i]),
                    (-T::Real::TWO, state),
                    (-step, rate),
                ],
            )?;
        }
        Ok(matches!(
            self.newton(
                plan,
                problem,
                time + step,
                T::Real::TWO / step,
                false,
                context,
            )?,
            (NewtonOutcome::Converged(_), _)
        ))
    }
    /// Replay a recorded equation with the complete restart history. No controller,
    /// history mutation, acceptance accounting, or coefficient recomputation occurs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn replay_bdf<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        initial: &[T],
        geometry: BdfStep<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.linear.invalidate();
        let n = self.dimension();
        self.accepted.copy_from_slice(&initial[..n]);
        self.accepted_derivative.copy_from_slice(&initial[n..2 * n]);
        // Use the same actual-node predictor as the original proposal.
        let history = geometry.history;
        let predictor = if history == 0 {
            None
        } else {
            Some(Weights::new(
                &geometry.nodes[1..=geometry.order.min(history) + 1],
                Functional::Value(T::Real::ONE),
            )?)
        };
        for i in 0..n {
            self.candidate[i] = if let Some(predictor) = &predictor {
                predictor.affine(T::ZERO, T::Real::ONE, |j| {
                    if j == 0 {
                        initial[i]
                    } else {
                        initial[(j + 1) * n + i]
                    }
                })?
            } else {
                crate::safety::state_increment(initial[i], step, initial[n + i])?
            };
        }
        self.bdf_derivative(step, geometry, Some(&initial[2 * n..]))?;
        if let (NewtonOutcome::Failed(..), _) = self.newton(
            plan,
            problem,
            time + step,
            geometry.alpha[0] / step,
            false,
            context,
        )? {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::ReplayMismatch,
                scioxide_core::ConvergenceDiagnostics::new(),
                "recorded BDF equation no longer converges",
            ));
        }
        Ok(())
    }
    fn bdf_derivative(
        &mut self,
        step: T::Real,
        geometry: BdfStep<T::Real>,
        history: Option<&[T]>,
    ) -> Result<(), SciOxideError> {
        let n = self.dimension();
        let history = history.unwrap_or(&self.history);
        for i in 0..n {
            self.candidate_derivative[i] = crate::safety::stage_derivative(
                step,
                (0..=geometry.order).map(|j| {
                    let value = match j {
                        0 => self.candidate[i],
                        1 => self.accepted[i],
                        _ => history[(j - 2) * n + i],
                    };
                    (geometry.alpha[j], value)
                }),
            )?;
        }
        Ok(())
    }
    /// Normalized local errors of the converged candidate at `order` from its history.
    fn bdf_errors(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        order: usize,
        nodes: &[T::Real; CAPACITY],
    ) -> Result<Norms<T::Real>, SciOxideError> {
        let n = self.dimension();
        let estimate = LocalError::new(order, BdfFormula::Bdf, nodes, self.history_count)?;
        for i in 0..n {
            self.local[i] = estimate.component(self.candidate[i], self.accepted[i], |j| {
                self.history[j * n + i]
            })?;
        }
        self.local_norms(plan)
    }
    /// Normalized local errors of a converged first step, which has no history.
    ///
    /// The differential estimate is the defect `e_x = (x₁ − x₀ − h ẋ₀)/2 = (h/2)(ẋ₁ − ẋ₀)`
    /// of the first predictor, of local error power two as in ODE BDF startup; `x₀`, `ẋ₀`
    /// are the accepted differential states and rates and `x₁`, `ẋ₁` the candidate's. The
    /// residual defines no initial algebraic rates, so the algebraic estimate is the
    /// response of the linearized algebraic equations `G` to `e_x`: the algebraic part of
    /// `A⁻¹ [0; G_x e_x]` with the stage iteration matrix `A = F_y + (α₀/h) F_ẏ`, which
    /// is `G_z⁻¹ G_x e_x` up to a relative `O(h)` coupling term. It costs one product and
    /// one solve with the converged stage linearization; a GMRES solve above its verified
    /// defect yields no estimate, and the proposal fails like its stage.
    fn startup_errors<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        step: T::Real,
        alpha: T::Real,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<Norms<T::Real>>, Fault> {
        let partition = plan.0.partition();
        self.local.fill(T::ZERO);
        self.startup_defects(partition, step)?;
        // A candidate equal to its predictor converged without a stage linearization and
        // has no defect to respond to.
        if partition.differential() < self.dimension()
            && self.local.iter().any(|&value| value != T::ZERO)
        {
            let inverse = alpha / step;
            let preconditioner = if plan.0.operator_mode() {
                admit(context, EvaluationRole::Preconditioner)?;
                problem.preconditioner(
                    time,
                    &self.candidate,
                    &self.candidate_derivative,
                    false,
                    inverse,
                    context,
                )?
            } else {
                None
            };
            let mut effective = BoundDerivative {
                source: core::cell::Cell::new(None),
                problem,
                options: plan.1.numerics(),
                base: &self.residual,
                partition,
                time,
                state: &self.candidate,
                derivative: &self.candidate_derivative,
                initialization: false,
                inverse_step: inverse,
                scratch: plan.0.operator_scratch(),
                preconditioner,
            };
            self.linear
                .apply(&mut effective, &self.local, &mut self.correction, context)?;
            for (value, &differential) in self.correction.iter_mut().zip(partition.equations()) {
                if differential {
                    *value = T::ZERO;
                }
            }
            if !self.linear.solve_stage(
                &mut effective,
                &self.correction,
                &mut self.local,
                real(LINEAR_FORCING)?,
                context,
            )? {
                return Ok(None);
            }
            // The solve also returns differential responses; keep the defects.
            self.startup_defects(partition, step)?;
        }
        Ok(self.local_norms(plan).map(Some)?)
    }
    /// Write the differential first-predictor defects `(x₁ − x₀ − h ẋ₀)/2` into the local
    /// estimate, leaving algebraic entries unchanged.
    fn startup_defects(
        &mut self,
        partition: &DaePartition,
        step: T::Real,
    ) -> Result<(), SciOxideError> {
        for (i, &differential) in partition.variables().iter().enumerate() {
            if differential {
                let predictor = crate::safety::state_increment(
                    self.accepted[i],
                    step,
                    self.accepted_derivative[i],
                )?;
                self.local[i] = crate::safety::state_difference(self.candidate[i], predictor)?
                    * T::from_real(T::Real::HALF);
            }
        }
        Ok(())
    }
    /// Partitioned normalized norms of the local estimate, a quarter of each state-error
    /// budget reserved for it.
    fn local_norms(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
    ) -> Result<Norms<T::Real>, SciOxideError> {
        let options = plan.1.numerics();
        let mut norms = [T::Real::ZERO; 2];
        for (group, role) in [true, false].into_iter().enumerate() {
            let mut count = 0;
            for (i, &d) in plan.0.partition().variables().iter().enumerate() {
                if d == role {
                    self.errors[count] = self.local[i].abs();
                    self.scales[count] = self.candidate[i].abs().max(self.accepted[i].abs());
                    count += 1;
                }
            }
            norms[group] = if role {
                options.differential_error()
            } else {
                options.algebraic_error()
            }
            .normalized_norm(&self.errors[..count], &self.scales[..count])?
                / real::<T::Real>(LOCAL_ERROR_FRACTION)?;
        }
        Ok((norms[0], norms[1]))
    }
    /// Shared acceptance and retry policy, including optional objective quadrature.
    pub(crate) fn assess_bdf(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        mut trial: Trial<T::Real>,
        step: T::Real,
        extra: T::Real,
        fixed: bool,
    ) -> Result<(T::Real, bool), SciOxideError> {
        if trial.converged
            && !super::residual_ok(
                super::residual_norms(plan.0.partition(), &self.residual),
                plan.1.numerics(),
            )?
        {
            trial.converged = false;
        }
        let (mut next, accept) = if !trial.converged {
            (step * T::Real::HALF, false)
        } else if fixed {
            (step, true)
        } else {
            self.controller.compute_step(
                trial.errors.0.max(trial.errors.1).max(extra),
                step,
                trial.geometry.order + 1,
            )?
        };
        let growth = if trial.geometry.order == 1 {
            real(5.)?
        } else {
            T::Real::TWO
        };
        if next.abs() > step.abs() * growth {
            next = step * growth;
        }
        // Establish history after an order change, then suppress modest growth
        // to retain useful chord factors. Error-driven shrinkage is untouched.
        if accept
            && trial.next_order == trial.geometry.order
            && next.abs() > step.abs()
            && (self.residence < trial.geometry.order || crate::stiff::reuse::nearby(next / step)?)
        {
            next = step;
        }
        // A retry after rejection does not grow. After a stage failure at this order, a
        // stage that converges only on a Newton recovery retry is at the limit of the
        // corrector, not of the error test: growth would fail again, and the alternating
        // steps would keep low-order errors in the history that the order selection reads.
        if accept
            && (self.rejected > 0 || (self.limited && trial.retried))
            && next.abs() > step.abs()
        {
            next = step;
        }
        if !accept {
            self.linear.invalidate_factor();
            self.rejected += 1;
            let order = trial.next_order.min(trial.geometry.order);
            if self.order != order {
                self.residence = 0;
                self.limited = false;
            }
            self.limited |= !trial.converged;
            self.order = order;
            if fixed || self.rejected >= plan.1.numerics().rejection_limit().get() {
                return Err(SciOxideError::convergence_failure(
                    ConvergenceReason::NoSuccessfulCandidate,
                    scioxide_core::ConvergenceDiagnostics::new(),
                    "residual BDF rejection limit exhausted",
                ));
            }
        }
        Ok((next, accept))
    }
    pub(crate) fn accept_bdf(&mut self, time: T::Real, trial: Trial<T::Real>) {
        if self.order != trial.next_order {
            self.linear.invalidate_factor();
        }
        let n = self.dimension();
        self.history.copy_within(..(MAX_ORDER - 1) * n, n);
        self.history[..n].copy_from_slice(&self.accepted);
        self.history_times.copy_within(..MAX_ORDER - 1, 1);
        self.history_times[0] = time;
        self.history_count = (self.history_count + 1).min(MAX_ORDER);
        self.accepted.copy_from_slice(&self.candidate);
        self.accepted_derivative
            .copy_from_slice(&self.candidate_derivative);
        if self.order == trial.next_order {
            self.residence += 1;
        } else {
            self.residence = 0;
        }
        self.limited &= trial.retried && self.order == trial.next_order;
        self.order = trial.next_order;
        self.rejected = 0;
    }
    pub(crate) fn candidate(&self) -> (&[T], &[T]) {
        (&self.candidate, &self.candidate_derivative)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{DaePartition, OdeStageBackend};
    use scioxide_core::Budget;

    #[test]
    fn original_residual_is_required_even_for_a_converged_trial() {
        let plan = (
            DaeBdf::new(
                DaePartition::new(&[true], &[true]).unwrap(),
                OdeStageBackend::Dense,
            )
            .unwrap(),
            DaeOptions::new().with_numerics(
                crate::DaeNumericalOptions::new()
                    .with_rejection_limit(Budget::try_from(2).unwrap()),
            ),
        );
        let mut solve = plan.0.storage::<f64>().unwrap();
        solve.residual[0] = 1.;
        let trial = Trial {
            geometry: BdfStep::default(),
            errors: (0., 0.),
            converged: true,
            retried: false,
            next_order: 1,
        };
        let (next, accepted) = solve
            .assess_bdf((&plan.0, &plan.1), trial, 0.25, 0., false)
            .unwrap();
        assert!(!accepted);
        assert!((next - 0.125).abs() < f64::EPSILON);
        let error = solve
            .assess_bdf((&plan.0, &plan.1), trial, next, 0., false)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::NoSuccessfulCandidate,
                ..
            }
        ));
    }

    #[test]
    fn rejection_and_order_change_discard_factors_but_retain_raw_blocks() {
        use scioxide_core::{
            ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
        };
        let plan = (
            DaeBdf::new(
                DaePartition::new(&[true, false], &[true, false]).unwrap(),
                OdeStageBackend::Dense,
            )
            .unwrap(),
            DaeOptions::new(),
        );
        let mut solve = plan.0.storage::<f64>().unwrap();
        let mut epoch = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = epoch
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(2366),
                    scioxide_core::algorithm_id!("test", "dae-rejection", 1),
                    2,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        super::super::tests::prepared_correction(&mut solve, &plan, 0., 10., &mut context);
        let accepted = solve.accepted.clone();
        let history = solve.history.clone();
        let mut trial = Trial {
            geometry: BdfStep::default(),
            errors: (0., 0.),
            converged: false,
            retried: false,
            next_order: 1,
        };
        assert!(
            !solve
                .assess_bdf((&plan.0, &plan.1), trial, 0.1, 0., false)
                .unwrap()
                .1
        );
        assert_eq!(solve.accepted, accepted);
        assert_eq!(solve.history, history);
        super::super::tests::prepared_correction(&mut solve, &plan, 0., 10., &mut context);
        assert_eq!(context.counts().own().provider(EvaluationRole::Jacobian), 1);
        assert_eq!(
            context
                .counts()
                .own()
                .provider(EvaluationRole::Factorization),
            2
        );
        trial.converged = true;
        trial.next_order = 2;
        solve.accept_bdf(0., trial);
        super::super::tests::prepared_correction(&mut solve, &plan, 0., 10., &mut context);
        assert_eq!(context.counts().own().provider(EvaluationRole::Jacobian), 1);
        assert_eq!(
            context
                .counts()
                .own()
                .provider(EvaluationRole::Factorization),
            3
        );
    }
}
