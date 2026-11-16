//! Kernels for Newton–Krylov solves.
use super::*;
use crate::context::{KernelFault, KernelResult, admit, check_stop};

/// Jacobian action at `point`, whose checked residual is `residual`.
pub(crate) struct Linearization<'a, 'p, T: NumericScalar> {
    pub(crate) problem: &'a NewtonKrylovProblem<'p, T>,
    pub(crate) point: &'a [T],
    pub(crate) residual: &'a [T],
    pub(crate) row_scales: &'a [T::Real],
    pub(crate) variable_scales: &'a [T::Real],
    pub(crate) differences: Option<&'a RefCell<DifferenceStorage<T>>>,
}
impl<T: NumericScalar> LinearOperator<T> for Linearization<'_, '_, T> {
    fn rows(&self) -> usize {
        self.problem.dimension()
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        let provider = self
            .problem
            .jacobian
            .map_or(Ok(0), JacobianOperator::jvp_workspace_len)?;
        scioxide_core::backend::checked_add(
            provider,
            if self.problem.scaling.is_some() {
                self.cols()
            } else {
                0
            },
            scioxide_core::quantity!("scaled_product_scratch"),
        )
    }
    fn apply_into(&self, _: &[T], _: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("problem"),
            InvalidInputReason::Inconsistent,
            "Newton linearizations require the context-aware operator entrypoint",
        ))
    }
    fn apply_into_scoped(
        &self,
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        if direction.len() != self.cols()
            || output.len() != self.rows()
            || scratch.len() != self.apply_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                InvalidInputReason::Inconsistent,
                "linearization shape mismatch",
            ));
        }
        finite(direction, NonFiniteOrigin::Input)?;
        let (physical, scratch) = scratch.split_at_mut(if self.problem.scaling.is_some() {
            self.cols()
        } else {
            0
        });
        let direction = if self.problem.scaling.is_some() {
            for ((out, &value), &scale) in
                physical.iter_mut().zip(direction).zip(self.variable_scales)
            {
                *out = value * T::from_real(scale);
            }
            finite(physical, NonFiniteOrigin::Intermediate)?;
            &physical[..]
        } else {
            direction
        };
        if let Some(reason) = context.stop_reason()? {
            return Ok(scioxide_core::Invocation::ExecutionStopped(reason));
        }
        output.fill(T::from_real(T::Real::NAN));
        if let Some(jacobian) = self.problem.jacobian {
            jacobian.jvp_into_scoped(self.point, direction, output, scratch, context)?;
        } else {
            let storage = self.differences.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("w"),
                    InvalidInputReason::Inconsistent,
                    "missing differentiation workspace",
                )
            })?;
            let mut borrowed = storage.try_borrow_mut().map_err(|_| {
                invalid(
                    scioxide_core::argument!("problem"),
                    InvalidInputReason::Inconsistent,
                    "reentrant numerical JVP",
                )
            })?;
            let DifferenceStorage {
                point,
                residual_scratch,
            } = &mut *borrowed;
            let mut largest = T::Real::ZERO;
            for (i, (&x, &v)) in self.point.iter().zip(direction).enumerate() {
                let scale = if self.problem.scaling.is_some() {
                    self.variable_scales[i]
                } else {
                    x.abs().max(T::Real::ONE)
                };
                largest = largest.max(v.abs() / scale);
            }
            if largest == T::Real::ZERO {
                output.fill(T::ZERO);
                return Ok(scioxide_core::Invocation::Evaluated(()));
            }
            // `1/σ`: each displacement `vᵢσ` is at most `√ε·max(1,|xᵢ|)`.
            let inverse = T::from_real(largest / T::Real::SQRT_EPSILON);
            for ((p, &x), &v) in point.iter_mut().zip(self.point).zip(direction) {
                *p = x + v / inverse;
            }
            if let scioxide_core::Admission::Denied(report) =
                context.record_provider(EvaluationRole::Residual)?
            {
                return Ok(scioxide_core::Invocation::BudgetExhausted(report));
            }
            let scratch = crate::residual::scratch_slice(
                residual_scratch,
                self.problem.residual.residual_workspace_len()?,
            )?;
            self.problem
                .residual
                .residual_into_scoped(point, output, scratch, context)?;
            finite(output, NonFiniteOrigin::Callback)?;
            for (value, &base) in output.iter_mut().zip(self.residual) {
                *value = difference(*value, base)? * inverse;
            }
        }
        for (v, &scale) in output.iter_mut().zip(self.row_scales) {
            *v = *v / T::from_real(scale);
        }
        finite(output, NonFiniteOrigin::Callback).map(scioxide_core::Invocation::Evaluated)
    }
}
/// Merit `½‖F‖²` from the residual norm, in scaled arithmetic.
pub(crate) fn merit<R: RealScalar>(norm: R) -> Result<scioxide_core::Scaled<R>, SciOxideError> {
    scioxide_core::Scaled::from_value(norm)?
        .square()?
        .shifted(-1)
}

/// Accepted Newton–Krylov state carried between outer iterations.
struct Progress<R: RealScalar> {
    /// Forcing term `η` of the next linear solve.
    eta: R,
    /// Smallest allowed forcing: `√ε` for forward-difference products.
    floor: R,
    /// Completed outer iterations.
    outer: usize,
    /// Most recent linear observation.
    last: Option<NewtonKrylovObservation<R>>,
    /// Accepted residual norm.
    norm: R,
    /// Residual reference scale.
    scale: R,
    /// Zhang–Hager reference merit `C`.
    reference: scioxide_core::Scaled<R>,
    /// Zhang–Hager reference weight `Q`.
    weight: R,
    /// Budget stop reported if this iteration's inexact step is unusable:
    /// the inner solve hit its iteration limit with no outer iteration left.
    limited: Option<RootTermination>,
}

/// Result of the nonmonotone backtracking search.
enum Search<R> {
    /// An accepted step length and the trial residual norm.
    Accepted { alpha: R, norm: R },
    /// The search ended the solve.
    Stop(RootTermination),
}

impl<T: NumericScalar> NewtonKrylovSolver<T> {
    /// Newton iteration: setup, then per outer iteration the stopping checks,
    /// linear solve, check of the Jacobian action, backtracking and update, and
    /// finally the output write.
    pub(crate) fn run(
        &self,
        problem: &NewtonKrylovProblem<'_, T>,
        options: &NewtonKrylovOptions<T::Real>,
        result: &mut NewtonKrylovResult<T>,
        w: &mut NewtonKrylovWorkspace<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut state = setup(problem, options, w, context)?;
        let outcome: KernelResult<RootTermination> = (|| {
            loop {
                if let Some(termination) = gate(options, &mut state, context)? {
                    return Ok(termination);
                }
                if let Some(termination) = linear_solve(problem, options, w, &mut state, context)? {
                    return Ok(termination);
                }
                if let Some(termination) = verify(problem, options, w, &mut state, context)? {
                    return Ok(termination);
                }
                if problem.scaling.is_some() {
                    for (value, &scale) in w
                        .direction
                        .as_mut_slice()
                        .iter_mut()
                        .zip(&w.variable_scales)
                    {
                        *value = *value * T::from_real(scale);
                    }
                }
                let (alpha, norm) = match backtrack(problem, options, w, &state, context)? {
                    Search::Accepted { alpha, norm } => (alpha, norm),
                    Search::Stop(termination) => return Ok(termination),
                };
                if let Some(termination) =
                    commit(problem, options, w, &mut state, alpha, norm, context)?
                {
                    return Ok(termination);
                }
            }
        })();
        let termination = match outcome {
            Err(KernelFault::Stopped(reason)) => reason,
            Err(KernelFault::Error(error)) => return Err(error),
            Ok(termination) => termination,
        };
        publish(result, w, &state, termination, context)
    }
}

fn refresh_scales<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    w: &mut NewtonKrylovWorkspace<T>,
    context: &mut SolveContext<'_, '_>,
) -> Result<bool, SciOxideError> {
    for (old, &scale) in w.step.as_mut_slice().iter_mut().zip(&w.row_scales) {
        *old = T::from_real(scale);
    }
    if let Some(scaling) = problem.scaling {
        w.row_scales.fill(T::Real::NAN);
        w.variable_scales.fill(T::Real::NAN);
        scaling.scales_into(
            w.x.as_slice(),
            &mut w.row_scales,
            &mut w.variable_scales,
            &mut w.residual_scratch[..scaling.scaling_workspace_len()?],
            context,
        )?;
        if w.row_scales
            .iter()
            .chain(&w.variable_scales)
            .any(|&v| !v.is_finite())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "component divisor overflow or incomplete scaling provider",
            ));
        }
        if w.row_scales
            .iter()
            .chain(&w.variable_scales)
            .any(|&v| v <= T::Real::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("scaling"),
                InvalidInputReason::OutOfRange,
                "component divisors must be positive and finite",
            ));
        }
    } else {
        w.row_scales.fill(T::Real::ONE);
        w.variable_scales.fill(T::Real::ONE);
    }
    for ((out, &f), &scale) in w
        .scaled
        .as_mut_slice()
        .iter_mut()
        .zip(w.f.as_slice())
        .zip(&w.row_scales)
    {
        *out = f / T::from_real(scale);
    }
    finite(w.scaled.as_slice(), NonFiniteOrigin::Intermediate)?;
    Ok(w.step
        .as_slice()
        .iter()
        .zip(&w.row_scales)
        .any(|(&old, &scale)| old != T::from_real(scale)))
}

/// Evaluate and assess the initial point and seed the forcing and merit state.
fn setup<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    options: &NewtonKrylovOptions<T::Real>,
    w: &mut NewtonKrylovWorkspace<T>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Progress<T::Real>, SciOxideError> {
    w.history.clear();
    if let Some(reason) = evaluate(
        problem,
        w.x.as_slice(),
        w.f.as_mut_slice(),
        &mut w.residual_scratch,
        context,
    )? {
        return Err(initial_stop(reason, context));
    }
    context.assess_accepted(PartialResultTrust::ResidualChecked)?;
    // Forward-difference products carry O(√ε) relative error; a tighter
    // linear request could not be met or verified.
    let floor = if problem.jacobian.is_none() {
        T::Real::SQRT_EPSILON
    } else {
        T::Real::ZERO
    };
    refresh_scales(problem, w, context)?;
    let norm = norm(w.scaled.as_slice())?;
    let scale = options.scale.unwrap_or(norm);
    w.accepted = Some(crate::context::residual(norm, scale)?);
    Ok(Progress {
        eta: options.forcing.initial().max(floor),
        floor,
        outer: 0,
        last: None,
        norm,
        scale,
        reference: merit(norm)?,
        weight: T::Real::ONE,
        limited: None,
    })
}

/// Convergence, stop and iteration-limit checks before starting an iteration.
fn gate<R: RealScalar>(
    options: &NewtonKrylovOptions<R>,
    state: &mut Progress<R>,
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Option<RootTermination>> {
    if crate::termination::residual_ok(&options.residual.tolerance(), state.norm, state.scale)? {
        return Ok(Some(RootTermination::ResidualTolerance));
    }
    if let Some(reason) = context.stop_reason()? {
        return Ok(Some(RootTermination::from_execution(reason)));
    }
    if state.outer == options.iterations.get() {
        return Ok(Some(RootTermination::IterationLimit));
    }
    check_stop(context)?;
    crate::context::admitted(context.record_iteration()?)?;
    state.outer += 1;
    Ok(None)
}

/// Solve `J d = −F` to the forcing tolerance, record the observation and map
/// a terminal linear outcome.
fn linear_solve<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    options: &NewtonKrylovOptions<T::Real>,
    w: &mut NewtonKrylovWorkspace<T>,
    state: &mut Progress<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Option<RootTermination>> {
    w.rhs
        .as_mut_slice()
        .iter_mut()
        .zip(w.scaled.as_slice())
        .for_each(|(b, &f)| *b = -f);
    w.direction.as_mut_slice().fill(T::ZERO);
    state.limited = None;
    let linear = options.linear::<T>(state.eta)?;
    let tolerance = ResidualTolerance::new(T::Real::ZERO, state.eta)?;
    let operator = Linearization {
        problem,
        point: w.x.as_slice(),
        residual: w.f.as_slice(),
        row_scales: &w.row_scales,
        variable_scales: &w.variable_scales,
        differences: w.differences.as_ref(),
    };
    admit(context, EvaluationRole::LinearSolve)?;
    if !context.can_consume(BudgetKind::Evaluation, 1)?.is_granted() {
        return Ok(Some(RootTermination::WorkLimit(BudgetKind::Evaluation)));
    }
    let report = match problem.linear_solver {
        Some(provider) => provider.solve_into(
            &NonlinearLinearization {
                operator: &operator,
                point: w.x.as_slice(),
                residual: w.f.as_slice(),
                rows: &w.row_scales,
                variables: &w.variable_scales,
            },
            &w.rhs,
            &mut w.direction,
            tolerance,
            &mut *context,
        )?,
        None => linear.solve_into(
            &operator,
            &w.rhs,
            problem
                .preconditioner
                .map_or(Preconditioning::None, Preconditioning::Inverse),
            w.direction.view_mut(),
            w.linear.as_mut().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("workspace"),
                    InvalidInputReason::Inconsistent,
                    "missing GMRES storage",
                )
            })?,
            &mut *context,
        )?,
    };
    let termination = report.termination();
    let observation = NewtonKrylovObservation {
        residual_norm: state.norm,
        forcing: state.eta,
        linear_residual: report.diagnostic_residual_norm(),
        linear_iterations: report.iterations(),
        linear_termination: termination,
        step_length: T::Real::ZERO,
    };
    state.last = Some(observation);
    if options.history {
        w.history.push(observation);
    }
    Ok(match termination {
        LinearSolveTermination::EvaluationLimit => {
            Some(RootTermination::WorkLimit(BudgetKind::Evaluation))
        }
        LinearSolveTermination::ExecutionStopped(reason) => {
            Some(RootTermination::from_execution(reason))
        }
        // The step may still be usable; `verify` decides. The next gate then
        // reports the iteration stop after the step is taken.
        LinearSolveTermination::IterationLimit
            if !context.can_consume(BudgetKind::Iteration, 1)?.is_granted() =>
        {
            state.limited = Some(if state.outer == options.iterations.get() {
                RootTermination::IterationLimit
            } else {
                RootTermination::WorkLimit(BudgetKind::Iteration)
            });
            None
        }
        LinearSolveTermination::Breakdown(_) | LinearSolveTermination::ConditionLimit => {
            Some(RootTermination::LinearFailure(termination))
        }
        _ => None,
    })
}

/// Independently verify the actual Jacobian action `F + Jd` used for both the
/// forcing update and the merit slope, and reject an unusable inexact step.
///
/// An unmet request still yields an inexact Newton step while the verified
/// linear residual stays within the policy's largest forcing, which keeps the
/// merit slope below `−(1 − η_max)‖F‖²`, also on the last outer iteration. An
/// unusable step after an inner iteration-limit stop reports that stop.
fn verify<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    options: &NewtonKrylovOptions<T::Real>,
    w: &mut NewtonKrylovWorkspace<T>,
    state: &mut Progress<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Option<RootTermination>> {
    let operator = Linearization {
        problem,
        point: w.x.as_slice(),
        residual: w.f.as_slice(),
        row_scales: &w.row_scales,
        variable_scales: &w.variable_scales,
        differences: w.differences.as_ref(),
    };
    admit(context, EvaluationRole::LinearOperator)?;
    let scratch =
        crate::residual::scratch_slice(&mut w.product_scratch, operator.apply_workspace_len()?)?;
    crate::context::evaluated(operator.apply_into_scoped(
        w.direction.as_slice(),
        w.product.as_mut_slice(),
        scratch,
        &mut scioxide_core::OperatorContext::new(context),
    )?)?;
    for ((r, &f), &jp) in w
        .step
        .as_mut_slice()
        .iter_mut()
        .zip(w.scaled.as_slice())
        .zip(w.product.as_slice())
    {
        *r = affine(f, T::ONE, jp)?;
    }
    let verified = norm(w.step.as_slice())?;
    let termination =
        state
            .last
            .as_mut()
            .map_or(LinearSolveTermination::Converged, |observation| {
                observation.linear_residual = Some(verified);
                observation.linear_termination
            });
    if let Some(observation) = w.history.last_mut() {
        observation.linear_residual = Some(verified);
    }
    let within = |forcing| {
        ResidualTolerance::new(T::Real::ZERO, forcing)?.is_satisfied(verified, state.norm)
    };
    if !within(state.eta)? && !within(options.forcing.maximum().max(state.floor))? {
        return Ok(Some(state.limited.unwrap_or(
            RootTermination::LinearFailure(if termination.is_success() {
                LinearSolveTermination::Stagnation
            } else {
                termination
            }),
        )));
    }
    Ok(None)
}

/// Nonmonotone Armijo backtracking along `d` against the Zhang–Hager reference.
///
/// Trials outside the residual's domain shrink the step without counting as
/// merit failures.
fn backtrack<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    options: &NewtonKrylovOptions<T::Real>,
    w: &mut NewtonKrylovWorkspace<T>,
    state: &Progress<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Search<T::Real>> {
    let slope = dot(w.scaled.as_slice(), w.product.as_slice())?.real();
    if slope.cmp_value(scioxide_core::Scaled::ZERO) != core::cmp::Ordering::Less {
        return Ok(Search::Stop(RootTermination::GlobalizationFailure));
    }
    let relative = if problem.scaling.is_some() {
        w.direction
            .as_slice()
            .iter()
            .zip(&w.variable_scales)
            .fold(T::Real::ZERO, |largest, (&v, &scale)| {
                largest.max(v.abs() / scale)
            })
    } else {
        crate::nonlinear_math::relative_length(w.x.as_slice(), w.direction.as_slice())
    };
    let mut search = crate::nonlinear_math::Backtracking::with_allowance(
        slope,
        state.reference.sub(merit(state.norm)?)?,
    );
    let mut domain_rejected = false;
    loop {
        let alpha = search.step();
        if let Some(reason) = context.stop_reason()? {
            return Ok(Search::Stop(RootTermination::from_execution(reason)));
        }
        check_stop(context)?;
        crate::context::admitted(context.record_proposal()?)?;
        crate::nonlinear_math::affine_into(
            w.trial.as_mut_slice(),
            w.x.as_slice(),
            T::from_real(alpha),
            w.direction.as_slice(),
            crate::nonlinear_math::StepRounding::Separate,
        )?;
        if w.trial.as_slice() == w.x.as_slice() {
            return Ok(Search::Stop(if domain_rejected {
                RootTermination::GlobalizationFailure
            } else {
                RootTermination::Stagnation
            }));
        }
        match evaluate_trial(
            problem,
            w.trial.as_slice(),
            w.trial_f.as_mut_slice(),
            &mut w.residual_scratch,
            context,
        )? {
            TrialEvaluation::Stopped(reason) => return Ok(Search::Stop(reason)),
            TrialEvaluation::Rejected => {
                domain_rejected = true;
                search.reject_domain();
            }
            TrialEvaluation::Finite => {
                for ((out, &f), &scale) in w
                    .scaled_trial
                    .as_mut_slice()
                    .iter_mut()
                    .zip(w.trial_f.as_slice())
                    .zip(&w.row_scales)
                {
                    *out = f / T::from_real(scale);
                }
                if search.assess(
                    w.scaled.as_slice(),
                    w.scaled_trial.as_slice(),
                    options.armijo,
                )? {
                    return Ok(Search::Accepted {
                        alpha,
                        norm: norm(w.scaled_trial.as_slice())?,
                    });
                }
            }
        }
        context.record_rejected()?;
        if search.step() * relative < options.minimum_step {
            return Ok(Search::Stop(RootTermination::GlobalizationFailure));
        }
    }
}

/// Commit the accepted trial, update the forcing term and the nonmonotone
/// reference, and test convergence and stagnation.
fn commit<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    options: &NewtonKrylovOptions<T::Real>,
    w: &mut NewtonKrylovWorkspace<T>,
    state: &mut Progress<T::Real>,
    alpha: T::Real,
    next: T::Real,
    context: &mut SolveContext<'_, '_>,
) -> KernelResult<Option<RootTermination>> {
    let point_scale = norm(w.x.as_slice())?.max(T::Real::ONE);
    // Linear-model residual of the taken step, `‖(1 − α)F + α(F + Jd)‖`.
    for (r, &f) in w.step.as_mut_slice().iter_mut().zip(w.scaled.as_slice()) {
        *r = affine(f, T::from_real(alpha), difference(*r, f)?)?;
    }
    let model_norm = norm(w.step.as_slice())?;
    for ((step, &trial), &x) in w
        .step
        .as_mut_slice()
        .iter_mut()
        .zip(w.trial.as_slice())
        .zip(w.x.as_slice())
    {
        *step = difference(trial, x)?;
    }
    let small_step = if problem.scaling.is_some() {
        for (step, &scale) in w.step.as_mut_slice().iter_mut().zip(&w.variable_scales) {
            *step = *step / T::from_real(scale);
        }
        options
            .step
            .is_satisfied(norm(w.step.as_slice())?, T::Real::ONE)?
    } else {
        options
            .step
            .is_satisfied(norm(w.step.as_slice())?, point_scale)?
    };
    context.commit_accepted(PartialResultTrust::ResidualChecked)?;
    core::mem::swap(&mut w.x, &mut w.trial);
    core::mem::swap(&mut w.f, &mut w.trial_f);
    core::mem::swap(&mut w.scaled, &mut w.scaled_trial);
    if let Some(observation) = state.last.as_mut() {
        observation.step_length = alpha;
    }
    if let Some(observation) = w.history.last_mut() {
        observation.step_length = alpha;
    }
    state.eta = options
        .forcing
        .next(state.eta, state.norm, next, model_norm)
        .max(state.floor);
    let previous_weight = options.nonmonotone * state.weight;
    state.weight = previous_weight + T::Real::ONE;
    state.reference = scioxide_core::Scaled::from_value(previous_weight)?
        .mul(state.reference)?
        .add(merit(next)?)?
        .div(scioxide_core::Scaled::from_value(state.weight)?)?;
    let improved = next < state.norm;
    state.norm = next;
    if problem.scaling.is_some() {
        let changed = refresh_scales(problem, w, context)?;
        state.norm = norm(w.scaled.as_slice())?;
        if changed {
            // The old nonmonotone merit is comparable only in unchanged row units.
            state.reference = merit(state.norm)?;
            state.weight = T::Real::ONE;
        }
    }
    w.accepted = Some(crate::context::residual(state.norm, state.scale)?);
    if crate::termination::residual_ok(&options.residual.tolerance(), state.norm, state.scale)? {
        return Ok(Some(RootTermination::ResidualTolerance));
    }
    Ok((small_step && !improved).then_some(RootTermination::Stagnation))
}

/// Write the accepted point, residual, diagnostics and termination to the
/// output; a pending execution stop overrides a numerical termination.
fn publish<T: NumericScalar>(
    result: &mut NewtonKrylovResult<T>,
    w: &NewtonKrylovWorkspace<T>,
    state: &Progress<T::Real>,
    termination: RootTermination,
    context: &SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let termination = if matches!(
        termination,
        RootTermination::Cancelled | RootTermination::TimeLimit
    ) {
        termination
    } else {
        context
            .stop_reason()?
            .map_or(termination, RootTermination::from_execution)
    };
    let report = crate::context::residual(state.norm, state.scale)?;
    result.point.copy_from_slice(w.x.as_slice());
    result.residual.copy_from_slice(w.f.as_slice());
    result.report = Some(report);
    result.termination = termination;
    result.counts = context.counts();
    result.iterations = state.outer;
    result.last_linear = state.last;
    result.history.clear();
    result.history.extend_from_slice(&w.history);
    Ok(())
}
/// Checked residual evaluation at an accepted point; a stop is `Ok(Some(_))`.
///
/// Callback errors are returned unchanged.
pub(crate) fn evaluate<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    x: &[T],
    output: &mut [T],
    scratch: &mut [T],
    context: &mut SolveContext<'_, '_>,
) -> Result<Option<RootTermination>, SciOxideError> {
    match crate::residual::evaluate_scoped_raw(
        problem.residual,
        EvaluationRole::Residual,
        x,
        output,
        scratch,
        context,
    ) {
        Ok(result) => {
            result?;
            finite(output, NonFiniteOrigin::Callback)?;
            Ok(None)
        }
        Err(crate::context::KernelFault::Stopped(reason)) => Ok(Some(reason)),
        Err(crate::context::KernelFault::Error(error)) => Err(error),
    }
}

enum TrialEvaluation {
    Finite,
    Rejected,
    Stopped(RootTermination),
}

fn evaluate_trial<T: NumericScalar>(
    problem: &NewtonKrylovProblem<'_, T>,
    x: &[T],
    output: &mut [T],
    scratch: &mut [T],
    context: &mut SolveContext<'_, '_>,
) -> Result<TrialEvaluation, SciOxideError> {
    let evaluation = match crate::residual::evaluate_scoped_raw(
        problem.residual,
        EvaluationRole::Residual,
        x,
        output,
        scratch,
        context,
    ) {
        Ok(evaluation) => evaluation,
        Err(crate::context::KernelFault::Stopped(reason)) => {
            return Ok(TrialEvaluation::Stopped(reason));
        }
        Err(crate::context::KernelFault::Error(error)) => return Err(error),
    };
    Ok(match crate::residual::trial_status(evaluation, output)? {
        crate::residual::TrialStatus::Finite => TrialEvaluation::Finite,
        crate::residual::TrialStatus::Rejected => TrialEvaluation::Rejected,
    })
}

pub(crate) fn initial_stop(
    reason: RootTermination,
    context: &SolveContext<'_, '_>,
) -> SciOxideError {
    crate::context::stop_error(context, reason)
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod provider_tests;
