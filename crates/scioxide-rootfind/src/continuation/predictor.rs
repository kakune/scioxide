//! Predictor, corrector, tangent and step control of continuation.
use super::*;

/// Writes the border row `D⁻² t` for the tangent `t`.
pub(crate) fn make_row<R: RealScalar>(
    t: &[R],
    o: &ContinuationOptions<R>,
    row: &mut [R],
) -> Result<(), SciOxideError> {
    let n = t.len() - 1;
    for (i, (r, &v)) in row.iter_mut().zip(t).enumerate() {
        *r = (v / o.scale(i, n)) / o.scale(i, n);
    }
    finite(row, NonFiniteOrigin::Intermediate)
}
/// Returns the arclength constraint `rowᵀ (x − origin)`.
pub(crate) fn constraint<R: RealScalar>(
    x: &[R],
    origin: &[R],
    row: &[R],
) -> Result<R, SciOxideError> {
    let mut sum = scioxide_core::ScaledAccumulator::ZERO;
    for ((&a, &b), &r) in x.iter().zip(origin).zip(row) {
        sum.add_product(
            scioxide_core::Scaled::from_value(difference(a, b)?)?,
            scioxide_core::Scaled::from_value(r)?,
        )?;
    }
    sum.total()?.round()
}
/// The bordered map `(F(y), rowᵀ (y − origin))` solved by the corrector.
pub(crate) struct Augmented<'a, R: RealScalar> {
    pub(crate) provider: &'a dyn JacobianOperator<R>,
    pub(crate) origin: &'a [R],
    pub(crate) row: &'a [R],
}
impl<R: RealScalar> ResidualOperator<R> for Augmented<'_, R> {
    fn input_dimension(&self) -> usize {
        self.origin.len()
    }
    fn output_dimension(&self) -> usize {
        self.origin.len()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.residual_workspace_len()
    }
    fn residual_into(&self, x: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        let n = y.len() - 1;
        self.provider.residual_into(x, &mut y[..n], s)?;
        y[n] = constraint(x, self.origin, self.row)?;
        Ok(())
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = y.len() - 1;
        self.provider.residual_into_scoped(x, &mut y[..n], s, c)?;
        y[n] = constraint(x, self.origin, self.row)?;
        Ok(())
    }
}
impl<R: RealScalar> JacobianOperator<R> for Augmented<'_, R> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.provider.jvp_workspace_len()
    }
    fn jvp_into(&self, x: &[R], v: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        let n = y.len() - 1;
        self.provider.jvp_into(x, v, &mut y[..n], s)?;
        y[n] = dot(self.row, v)?.real().round()?;
        Ok(())
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        v: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = y.len() - 1;
        self.provider.jvp_into_scoped(x, v, &mut y[..n], s, c)?;
        y[n] = dot(self.row, v)?.real().round()?;
        Ok(())
    }
}
pub(crate) struct AtPoint<'a, 'b, R: RealScalar> {
    pub(crate) a: &'a Augmented<'b, R>,
    pub(crate) point: &'a [R],
}
impl<R: RealScalar> AtPoint<'_, '_, R> {
    pub(crate) fn validate(&self, v: &[R], y: &mut [R], s: &[R]) -> Result<(), SciOxideError> {
        if v.len() != self.point.len()
            || y.len() != self.point.len()
            || s.len() != self.a.jvp_workspace_len()?
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                InvalidInputReason::Inconsistent,
                "tangent operator shape mismatch",
            ));
        }
        finite(v, NonFiniteOrigin::Input)?;
        y.fill(R::NAN);
        Ok(())
    }
}
impl<R: RealScalar> LinearOperator<R> for AtPoint<'_, '_, R> {
    fn rows(&self) -> usize {
        self.point.len()
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.a.jvp_workspace_len()
    }
    fn apply_into(&self, v: &[R], y: &mut [R], s: &mut [R]) -> Result<(), SciOxideError> {
        self.validate(v, y, s)?;
        self.a.jvp_into(self.point, v, y, s)?;
        finite(y, NonFiniteOrigin::Callback)
    }
    fn apply_into_scoped(
        &self,
        v: &[R],
        y: &mut [R],
        s: &mut [R],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, R>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.validate(v, y, s)?;
        self.a.jvp_into_scoped(self.point, v, y, s, c)?;
        finite(y, NonFiniteOrigin::Callback).map(scioxide_core::Invocation::Evaluated)
    }
}
fn stop(reason: RootTermination) -> KernelResult<()> {
    Err(KernelFault::Stopped(reason))
}
/// Corrects `w.guess` with a nested Newton–Krylov solve of the bordered map.
pub(crate) fn correct<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
    o: &ContinuationOptions<R>,
    w: &mut ContinuationWorkspace<R>,
    c: &mut SolveContext<'_, '_>,
) -> KernelResult<()> {
    w.candidate.copy_from_slice(&w.guess);
    let initial_norm = verify(p, w, c)?;
    w.residual_reference = o.residual_scale.unwrap_or(initial_norm);
    let a = Augmented {
        provider: p.jacobian,
        origin: &w.origin,
        row: &w.row,
    };
    let problem = p.corrector(&a);
    // Admission ensures an initial residual is possible before entering the child.
    check_stop(c)?;
    if !c
        .can_consume(scioxide_core::BudgetKind::Evaluation, 1)?
        .is_granted()
    {
        return stop(RootTermination::WorkLimit(
            scioxide_core::BudgetKind::Evaluation,
        ));
    }
    NewtonKrylovSolver::new(o.corrector.clone())?.solve_into(
        &problem,
        &w.guess,
        &mut w.result,
        &mut w.newton,
        &mut *c,
    )?;
    let reason = match w.result.termination() {
        reason @ RootTermination::WorkLimit(_) => Some(reason),
        reason @ (RootTermination::Cancelled | RootTermination::TimeLimit) => Some(reason),
        RootTermination::IterationLimit
            if !c
                .can_consume(scioxide_core::BudgetKind::Iteration, 1)?
                .is_granted() =>
        {
            Some(RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Iteration,
            ))
        }
        _ => None,
    };
    if let Some(reason) = reason {
        return stop(reason);
    }
    Ok(())
}
/// Evaluates `F` at `w.candidate` and returns `‖F‖₂`.
pub(crate) fn verify<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
    w: &mut ContinuationWorkspace<R>,
    c: &mut SolveContext<'_, '_>,
) -> KernelResult<R> {
    crate::residual::evaluate_scoped(
        p.jacobian,
        EvaluationRole::Residual,
        &w.candidate,
        &mut w.residual,
        &mut w.scratch,
        c,
    )?;
    let n = p.dimension();
    w.linear_residual[..n].copy_from_slice(&w.residual);
    w.linear_residual[n] = R::ZERO;
    Ok(norm(&w.residual)?)
}
pub(crate) fn tangent_solver<R: RealScalar>(o: &ContinuationOptions<R>) -> GmresSolver<R> {
    GmresSolver::new(o.tangent_tolerance, o.tangent_iterations)
        .with_restart(o.tangent_restart)
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(o.policy),
        )
}
/// Solves the bordered tangent system and checks its residual; returns `false` on failure.
pub(crate) fn tangent<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
    o: &ContinuationOptions<R>,
    w: &mut ContinuationWorkspace<R>,
    c: &mut SolveContext<'_, '_>,
) -> KernelResult<bool> {
    let n = p.dimension();
    w.rhs.as_mut_slice().fill(R::ZERO);
    w.rhs.as_mut_slice()[n] = R::ONE;
    w.tangent.as_mut_slice().fill(R::ZERO);
    let a = Augmented {
        provider: p.jacobian,
        origin: &w.candidate,
        row: &w.row,
    };
    let operator = AtPoint {
        a: &a,
        point: &w.candidate,
    };
    let solver = tangent_solver(o);
    let tol = o.tangent_tolerance;
    let identity = match p.linear {
        Some(_) => SolveIdentity::new(
            c.identity().problem(),
            scioxide_core::algorithm_id!("rootfind", "tangent-correction", 1),
            (n + 1) as u128,
            0,
        ),
        None => crate::context::identity(solver.algorithm_id(), c.identity().problem(), n + 1, 0),
    };
    admit(c, EvaluationRole::LinearSolve)?;
    let term = scioxide_core::SolveScope::from(&mut *c).run(
        &mut w.child,
        identity,
        SolveBudgets::unbounded()
            .with_limit(scioxide_core::BudgetKind::Iteration, o.tangent_iterations),
        |child| {
            let report = match p.linear {
                Some(provider) => provider.solve_into(
                    &crate::NonlinearLinearization {
                        operator: &operator,
                        point: &w.candidate,
                        residual: &w.linear_residual,
                        rows: &w.units,
                        variables: &w.units,
                    },
                    &w.rhs,
                    &mut w.tangent,
                    tol,
                    &mut *child,
                )?,
                None => solver.solve_into(
                    &operator,
                    &w.rhs,
                    Preconditioning::None,
                    w.tangent.view_mut(),
                    w.linear.as_mut().ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("w"),
                            InvalidInputReason::Inconsistent,
                            "missing tangent workspace",
                        )
                    })?,
                    child,
                )?,
            };
            Ok(report.termination())
        },
    )?;
    match term {
        LinearSolveTermination::EvaluationLimit => {
            stop(RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Evaluation,
            ))?;
        }
        LinearSolveTermination::IterationLimit
            if !c
                .can_consume(scioxide_core::BudgetKind::Iteration, 1)?
                .is_granted() =>
        {
            stop(RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Iteration,
            ))?;
        }
        LinearSolveTermination::ExecutionStopped(r) => {
            stop(RootTermination::from_execution(r))?;
        }
        _ => {}
    }
    if !term.is_success() {
        return Ok(false);
    }
    finite(w.tangent.as_slice(), NonFiniteOrigin::Callback)?;
    admit(c, EvaluationRole::LinearOperator)?;
    let len = p.jacobian.jvp_workspace_len()?;
    crate::context::evaluated(operator.apply_into_scoped(
        w.tangent.as_slice(),
        &mut w.product,
        &mut w.scratch[..len],
        &mut scioxide_core::OperatorContext::new(c),
    )?)?;
    w.product[n] = difference(w.product[n], R::ONE)?;
    finite(&w.product, NonFiniteOrigin::Callback)?;
    if !tol.is_satisfied(norm(&w.product)?, R::ONE)? {
        return Ok(false);
    }
    for (i, (slot, &value)) in w.product.iter_mut().zip(w.tangent.as_slice()).enumerate() {
        *slot = value / o.scale(i, n);
    }
    let length = norm(&w.product)?;
    if length == R::ZERO {
        return Ok(false);
    }
    for value in w.tangent.as_mut_slice() {
        *value /= length;
    }
    finite(w.tangent.as_slice(), NonFiniteOrigin::Intermediate)?;
    Ok(true)
}

/// Outcome of the branch guards: reject, accept, or accept and grow the next step.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StepQuality {
    Reject,
    Accept,
    Grow,
}

/// Checks the change of the Jacobian along the correction direction. The
/// tangent-angle guard alone cannot tell apart parallel tangents on two nearby branches.
fn correction_quality<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
    point: &[R],
    o: &ContinuationOptions<R>,
    w: &mut ContinuationWorkspace<R>,
    c: &mut SolveContext<'_, '_>,
) -> KernelResult<StepQuality> {
    use scioxide_core::Scaled;
    let mut largest = Scaled::ZERO;
    for (&candidate, &origin) in w.candidate.iter().zip(&w.origin) {
        let value = Scaled::from_value(candidate)?.sub(Scaled::from_value(origin)?)?;
        if value.abs_cmp(largest).is_gt() {
            largest = value.abs();
        }
    }
    if largest.is_zero() {
        return Ok(StepQuality::Grow);
    }
    // Normalize before products so a tiny correction cannot underflow away a
    // change of Jacobian orientation, and large corrections stay representable.
    for ((slot, &candidate), &origin) in w.product.iter_mut().zip(&w.candidate).zip(&w.origin) {
        *slot = Scaled::from_value(candidate)?
            .sub(Scaled::from_value(origin)?)?
            .div(largest)?
            .round()?;
    }
    let len = p.jacobian.jvp_workspace_len()?;
    w.residual.fill(R::NAN);
    admit(c, EvaluationRole::LinearOperator)?;
    p.jacobian
        .jvp_into_scoped(point, &w.product, &mut w.residual, &mut w.scratch[..len], c)?;
    finite(&w.residual, NonFiniteOrigin::Callback)?;
    let n = p.dimension();
    let product = &mut w.rhs.as_mut_slice()[..n];
    product.fill(R::NAN);
    admit(c, EvaluationRole::LinearOperator)?;
    p.jacobian
        .jvp_into_scoped(&w.candidate, &w.product, product, &mut w.scratch[..len], c)?;
    finite(product, NonFiniteOrigin::Callback)?;
    let reference = norm(&w.residual)?;
    for (value, &old) in product.iter_mut().zip(&w.residual) {
        *value = difference(*value, old)?;
    }
    let change = norm(product)?;
    if !ResidualTolerance::new(R::ZERO, o.maximum_correction_contraction)?
        .is_satisfied(change, reference)?
    {
        return Ok(StepQuality::Reject);
    }
    Ok(
        if ResidualTolerance::new(R::ZERO, o.maximum_correction_contraction * R::HALF)?
            .is_satisfied(change, reference)?
        {
            StepQuality::Grow
        } else {
            StepQuality::Accept
        },
    )
}

/// Applies the tangent-angle and correction-ratio guards in the scaled metric.
fn step_quality<R: RealScalar>(
    point: &[R],
    tangent: &[R],
    o: &ContinuationOptions<R>,
    w: &ContinuationWorkspace<R>,
) -> Result<StepQuality, SciOxideError> {
    use scioxide_core::{Scaled, ScaledAccumulator};
    let n = point.len() - 1;
    let mut alignment = ScaledAccumulator::ZERO;
    let mut prediction = ScaledAccumulator::ZERO;
    let mut correction = ScaledAccumulator::ZERO;
    for (i, (&x, &t)) in point.iter().zip(tangent).enumerate() {
        let scale = Scaled::from_value(o.scale(i, n))?;
        alignment.add_product(
            Scaled::from_value(t)?.div(scale)?,
            Scaled::from_value(w.tangent.as_slice()[i])?.div(scale)?,
        )?;
        let predicted = Scaled::from_value(w.origin[i])?;
        let displacement = predicted.sub(Scaled::from_value(x)?)?.div(scale)?;
        prediction.add_product(displacement, displacement)?;
        let displacement = Scaled::from_value(w.candidate[i])?
            .sub(predicted)?
            .div(scale)?;
        correction.add_product(displacement, displacement)?;
    }
    let cosine = alignment.total()?.round()?.max(-R::ONE).min(R::ONE);
    let prediction = prediction.total()?;
    let correction = correction.total()?;
    let limit = prediction.mul(Scaled::from_value(o.maximum_correction_ratio)?.square()?)?;
    if prediction.is_zero() || cosine < o.minimum_tangent_cosine || !correction.abs_le(limit) {
        return Ok(StepQuality::Reject);
    }
    Ok(
        if R::ONE - cosine <= (R::ONE - o.minimum_tangent_cosine) * R::HALF
            && correction.abs_le(limit.mul(Scaled::HALF)?.mul(Scaled::HALF)?)
        {
            StepQuality::Grow
        } else {
            StepQuality::Accept
        },
    )
}
/// Computes the dense or injected branch-point diagnostic, if one is configured.
pub(crate) fn diagnose<R: RealScalar>(
    p: &ContinuationProblem<'_, R>,
    o: &ContinuationOptions<R>,
    w: &mut ContinuationWorkspace<R>,
    c: &mut SolveContext<'_, '_>,
) -> KernelResult<Option<ContinuationDiagnostic<R>>> {
    make_row(w.tangent.as_slice(), o, &mut w.row)?;
    if let Some(d) = w.dense.as_mut() {
        Ok(Some(d.evaluate(
            p.jacobian,
            &w.candidate,
            &w.row,
            o.rank_tolerance,
            &mut w.scratch,
            c,
        )?))
    } else if let Some(provider) = p.diagnostic {
        admit(c, EvaluationRole::Event)?;
        let d = provider.evaluate(p.jacobian, &w.candidate, w.tangent.as_slice(), c)?;
        d.validate()?;
        if d.augmented_rank.is_some_and(|rank| rank > p.dimension()) {
            return Err(invalid(
                scioxide_core::argument!("p"),
                InvalidInputReason::Inconsistent,
                "injected diagnostic rank exceeds the residual dimension",
            )
            .into());
        }
        Ok(Some(d))
    } else {
        Ok(None)
    }
}

impl<R: RealScalar> ContinuationState<R> {
    /// Corrects the seed with `λ` fixed, checks its residual and computes the
    /// first tangent. On failure the seed and history are unchanged, and the
    /// work done stays counted.
    ///
    /// # Errors
    ///
    /// Returns lifecycle, seed-correction, tangent, limit or provider errors.
    pub(crate) fn start(
        &mut self,
        p: &ContinuationProblem<'_, R>,
        w: &mut ContinuationWorkspace<R>,
        c: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.start_kernel(p, w, c).map_err(|fault| match fault {
            KernelFault::Stopped(reason) => crate::context::stop_error(c, reason),
            KernelFault::Error(error) => error,
        })
    }
    fn start_kernel(
        &mut self,
        p: &ContinuationProblem<'_, R>,
        w: &mut ContinuationWorkspace<R>,
        c: &mut SolveContext<'_, '_>,
    ) -> KernelResult<()> {
        self.preflight(p, w, c)?;
        if self.metadata.is_some() || c.counts() != scioxide_core::SolveCounts::new() {
            return Err(invalid(
                if self.metadata.is_some() {
                    scioxide_core::argument!("self")
                } else {
                    scioxide_core::argument!("c")
                },
                InvalidInputReason::Inconsistent,
                "start requires a fresh state and epoch",
            )
            .into());
        }
        w.guess.copy_from_slice(&self.point);
        w.origin.copy_from_slice(&self.point);
        w.row.fill(R::ZERO);
        let n = p.dimension();
        w.row[n] = R::ONE;
        correct(p, &self.options, w, c)?;
        if !w.result.is_success() {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::Stagnation,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(c.counts()),
                "initial correction did not converge",
            )
            .into());
        }
        w.candidate.copy_from_slice(w.result.point());
        let residual = verify(p, w, c)?;
        if !crate::termination::residual_ok(
            &self.options.residual.tolerance(),
            residual,
            w.residual_reference,
        )? {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::Stagnation,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(c.counts()),
                "initial point fails independently checked residual tolerance",
            )
            .into());
        }
        make_row(&self.tangent, &self.options, &mut w.row)?;
        if !tangent(p, &self.options, w, c)? {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::Stagnation,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(c.counts()),
                "initial tangent could not be verified",
            )
            .into());
        }
        let diagnostic = diagnose(p, &self.options, w, c)?;
        let report = crate::context::residual(residual, w.residual_reference)?;
        c.assess_accepted(PartialResultTrust::ResidualChecked)?;
        self.point.copy_from_slice(&w.candidate);
        self.residual.copy_from_slice(&w.residual);
        self.report = Some(report);
        self.tangent.copy_from_slice(w.tangent.as_slice());
        self.append(residual, w.result.iterations(), 0, false, diagnostic);
        self.update_completion();
        self.metadata = Some(c.checkpoint(self.state_token())?);
        Ok(())
    }
    /// Tries to accept one more point. A limit, cancellation or deadline sets
    /// the termination status; points accepted earlier are kept. Provider errors
    /// also keep them, with the work recorded in the context.
    ///
    /// # Errors
    ///
    /// Returns mismatched state or storage, provider and arithmetic errors.
    pub(crate) fn step(
        &mut self,
        p: &ContinuationProblem<'_, R>,
        w: &mut ContinuationWorkspace<R>,
        c: &mut SolveContext<'_, '_>,
    ) -> Result<RootTermination, SciOxideError> {
        self.preflight(p, w, c)?;
        if self.metadata.is_none() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                InvalidInputReason::Inconsistent,
                "initialize before advancing",
            ));
        }
        if self.status != RootTermination::Advanced {
            return Ok(self.status);
        }
        match self.advance(p, w, c) {
            Ok(()) => {}
            Err(KernelFault::Stopped(reason)) => self.status = reason,
            Err(KernelFault::Error(error)) => {
                self.metadata = Some(c.checkpoint(self.state_token())?);
                return Err(error);
            }
        }
        self.metadata = Some(c.checkpoint(self.state_token())?);
        Ok(self.status)
    }
    /// Proposes trial steps until one is accepted or the step control gives up.
    pub(crate) fn advance(
        &mut self,
        p: &ContinuationProblem<'_, R>,
        w: &mut ContinuationWorkspace<R>,
        c: &mut SolveContext<'_, '_>,
    ) -> KernelResult<()> {
        if self.history.len() == self.options.max_points.get() {
            self.status = RootTermination::HistoryLimit;
            return Ok(());
        }
        let n = p.dimension();
        let mut step = self.next_step;
        let mut failure = RootTermination::StepLimit;
        for rejected in 0..self.options.max_rejections.get() {
            failure = RootTermination::StepLimit;
            check_stop(c)?;
            if !crate::context::Session::new(c).try_iteration_only()? {
                return stop(RootTermination::WorkLimit(
                    scioxide_core::BudgetKind::Iteration,
                ));
            }
            if !crate::context::Session::new(c).try_proposal()? {
                return stop(RootTermination::WorkLimit(
                    scioxide_core::BudgetKind::Proposal,
                ));
            }
            w.row.fill(R::ZERO);
            let (actual, endpoint) = match &self.path {
                ContinuationPath::Parameters(targets) => {
                    let target = targets[self.next_target];
                    let delta = difference(target, self.point[n])?;
                    let amount = step.min(delta.abs());
                    let next = if amount == delta.abs() {
                        target
                    } else {
                        affine(self.point[n], delta.signum(), amount)?
                    };
                    if self.tangent[n].abs() <= R::EPSILON {
                        self.status = RootTermination::TangentFailure;
                        return Ok(());
                    }
                    let increment = difference(next, self.point[n])? / self.tangent[n];
                    affine_into(
                        &mut w.guess[..n],
                        &self.point[..n],
                        increment,
                        &self.tangent[..n],
                        StepRounding::Separate,
                    )?;
                    w.guess[n] = next;
                    w.row[n] = R::ONE;
                    (amount, next == target)
                }
                ContinuationPath::PseudoArclength(total) => {
                    let amount = step.min(*total - self.distance);
                    affine_into(
                        &mut w.guess[..=n],
                        &self.point[..=n],
                        amount,
                        &self.tangent[..=n],
                        StepRounding::Separate,
                    )?;
                    make_row(&self.tangent, &self.options, &mut w.row)?;
                    (amount, amount == *total - self.distance)
                }
            };
            if w.guess == self.point {
                self.status = RootTermination::StepLimit;
                return Ok(());
            }
            finite(&w.guess, NonFiniteOrigin::Intermediate)?;
            w.origin.copy_from_slice(&w.guess);
            correct(p, &self.options, w, c)?;
            if w.result.is_success() {
                w.candidate.copy_from_slice(w.result.point());
                // A fixed-parameter corrector must publish the requested parameter exactly.
                if matches!(self.path, ContinuationPath::Parameters(_)) {
                    w.candidate[n] = w.origin[n];
                }
                let raw = verify(p, w, c)?;
                let constraint = constraint(&w.candidate, &w.origin, &w.row)?;
                if crate::termination::residual_ok(
                    &self.options.residual.tolerance(),
                    raw,
                    w.residual_reference,
                )? && self
                    .options
                    .constraint_tolerance
                    .is_satisfied(constraint.abs(), R::ONE)?
                {
                    let contraction = correction_quality(p, &self.point, &self.options, w, c)?;
                    if contraction != StepQuality::Reject {
                        make_row(&self.tangent, &self.options, &mut w.row)?;
                        if tangent(p, &self.options, w, c)? {
                            let quality =
                                step_quality(&self.point, &self.tangent, &self.options, w)?
                                    .min(contraction);
                            if quality != StepQuality::Reject {
                                let diagnostic = diagnose(p, &self.options, w, c)?;
                                let turning = self.tangent[n] * w.tangent.as_slice()[n] < R::ZERO;
                                let report = crate::context::residual(raw, w.residual_reference)?;
                                c.commit_accepted(PartialResultTrust::ResidualChecked)?;
                                self.point.copy_from_slice(&w.candidate);
                                self.residual.copy_from_slice(&w.residual);
                                self.report = Some(report);
                                self.tangent.copy_from_slice(w.tangent.as_slice());
                                if matches!(self.path, ContinuationPath::PseudoArclength(_)) {
                                    self.distance = if endpoint {
                                        match self.path {
                                            ContinuationPath::PseudoArclength(s) => s,
                                            _ => self.distance,
                                        }
                                    } else {
                                        self.distance + actual
                                    };
                                }
                                self.append(
                                    raw,
                                    w.result.iterations(),
                                    rejected,
                                    turning,
                                    diagnostic,
                                );
                                let growth =
                                    if quality == StepQuality::Grow && w.result.iterations() <= 4 {
                                        R::ONE + R::HALF
                                    } else {
                                        R::ONE
                                    };
                                self.next_step = if actual > self.options.maximum_step / growth {
                                    self.options.maximum_step
                                } else {
                                    actual * growth
                                }
                                .max(self.options.minimum_step);
                                self.update_completion();
                                return Ok(());
                            }
                        } else {
                            failure = RootTermination::TangentFailure;
                        }
                    }
                }
            }
            c.record_rejected()?;
            if step <= self.options.minimum_step || actual < self.options.minimum_step {
                break;
            }
            step = (step * R::HALF).max(self.options.minimum_step);
        }
        self.status = failure;
        Ok(())
    }
}
