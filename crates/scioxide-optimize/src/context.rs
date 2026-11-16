//! Borrowed solve orchestration and callback observation.

use core::cell::{Cell, RefCell};

use scioxide_core::{
    AlgorithmId, Argument, Budget, BudgetKind, ConvergenceDiagnostics, ConvergenceReason,
    EvaluationCounts, EvaluationRole, ExecutionContext, ExecutionStopReason, InvalidInputReason,
    NumericScalar, PartialResultTrust, ProblemToken, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::Matrix;

use crate::{ObjectiveFunction, OptimizeTermination};

pub(crate) fn identity(
    algorithm: AlgorithmId,
    problem: ProblemToken,
    dimension: usize,
    options: u128,
) -> Result<SolveIdentity, SciOxideError> {
    Ok(SolveIdentity::new(
        problem,
        algorithm,
        dimension as u128,
        options,
    ))
}

/// Internal identity of a call-local epoch: method, dimension and configuration.
pub(crate) fn local_identity(
    algorithm: AlgorithmId,
    dimension: usize,
    options: &impl core::fmt::Debug,
) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(0),
        algorithm,
        dimension as u128,
        options_token(options),
    )
}

pub(crate) fn options_token(options: &impl core::fmt::Debug) -> u128 {
    struct Token(u128);
    impl core::fmt::Write for Token {
        fn write_str(&mut self, value: &str) -> core::fmt::Result {
            for byte in value.bytes() {
                self.0 = (self.0 ^ u128::from(byte)).wrapping_mul(0x100000001b3);
            }
            Ok(())
        }
    }
    let mut token = Token(0xcbf29ce484222325);
    let _ = core::fmt::write(&mut token, format_args!("{options:?}"));
    token.0
}

pub(crate) fn sequential<T>(
    identity: SolveIdentity,
    iterations: Budget,
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    sequential_with_budgets(
        identity,
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, iterations),
        run,
    )
}

pub(crate) fn sequential_with_budgets<T>(
    identity: SolveIdentity,
    mut budgets: SolveBudgets,
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    run(&mut context)
}

/// Translate a denied evaluation request without inventing caller-input errors.
pub(crate) fn budget_error(
    context: &SolveContext<'_, '_>,
    kind: BudgetKind,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::convergence_failure(
        match kind {
            BudgetKind::Iteration => ConvergenceReason::IterationLimit,
            BudgetKind::Evaluation => ConvergenceReason::EvaluationLimit,
            other => ConvergenceReason::ResourceLimit(other),
        },
        ConvergenceDiagnostics::new()
            .with_counts(context.counts())
            .with_budget(context.effective_budget_report(kind)),
        detail,
    )
}

fn borrow_error(argument: Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        InvalidInputReason::Inconsistent,
        "an observation cannot retain a mutable solve-context borrow across a callback",
    )
}

/// Borrow failure of an observation
/// and its parameter `argument` that owns the borrowed state.
pub(crate) fn observation_borrow_error(argument: Argument) -> SciOxideError {
    borrow_error(argument)
}

pub(crate) trait Observation {
    fn execution(&self) -> Result<ExecutionContext<'_>, SciOxideError>;
    fn counts(&self) -> Result<EvaluationCounts, SciOxideError>;
    fn cache_hit(&self) -> Result<(), SciOxideError>;
    fn proposal(&self) -> Result<(), SciOxideError>;
    fn reject(&self) -> Result<(), SciOxideError>;
    fn admit(&self, role: EvaluationRole) -> Result<(), SciOxideError>;
}

pub(crate) fn invoke<T>(
    observer: Option<&dyn Observation>,
    role: EvaluationRole,
    provider: impl FnOnce() -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    if let Some(observer) = observer {
        observer.admit(role)?;
    }
    provider()
}

/// Observe `objective` through `observer`; errors of the observed provider are
/// reported.
pub(crate) fn observe<'a, F: ?Sized>(
    objective: &'a F,
    observer: &'a dyn Observation,
) -> ObservedObjective<'a, F> {
    ObservedObjective {
        objective,
        observer,
    }
}

/// The mutable shell is explicit and local; numerical callback bodies stay pure.
///
/// Borrowed solve accounting and execution. The `argument` names the caller's context
/// parameter wrapped by the session.
pub(crate) struct SolveSession<'borrow, 'state, 'capability> {
    context: RefCell<&'borrow mut SolveContext<'state, 'capability>>,
    stopped: Cell<Option<OptimizeTermination>>,
    execution: Cell<ExecutionContext<'capability>>,

    argument: Argument,
}

impl<'borrow, 'state, 'capability> SolveSession<'borrow, 'state, 'capability> {
    pub(crate) fn new(
        context: &'borrow mut SolveContext<'state, 'capability>,
        argument: Argument,
    ) -> Self {
        Self {
            execution: Cell::new(context.execution()),
            context: RefCell::new(context),
            stopped: Cell::new(None),

            argument,
        }
    }

    #[inline]
    pub(crate) fn termination(&self) -> Result<Option<OptimizeTermination>, SciOxideError> {
        if let Some(stopped) = self.stopped.get() {
            return Ok(Some(stopped));
        }
        let reason = self.execution.get().stop_reason()?;
        let termination = match reason {
            Some(ExecutionStopReason::Cancelled) => Some(OptimizeTermination::Cancelled),
            Some(ExecutionStopReason::DeadlineReached) => Some(OptimizeTermination::TimeLimit),
            Some(_) => Some(OptimizeTermination::Cancelled),
            None => None,
        };
        self.stopped.set(termination);
        Ok(termination)
    }

    pub(crate) fn stopped(&self) -> Option<OptimizeTermination> {
        self.stopped.get()
    }

    pub(crate) fn check(&self) -> Result<(), SciOxideError> {
        if let Some(stop) = self.termination()? {
            return Err(self.stop_error(stop)?);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn trial_iteration(&self) -> Result<Option<OptimizeTermination>, SciOxideError> {
        if let Some(stop) = self.iteration()? {
            return Ok(Some(stop));
        }
        match self.proposal() {
            Ok(()) => Ok(None),
            Err(error) => match self.stopped() {
                Some(stop) => Ok(Some(stop)),
                None => Err(error),
            },
        }
    }

    pub(crate) fn set_stopped(&self, termination: OptimizeTermination) {
        self.stopped.set(Some(termination));
    }

    pub(crate) fn limit_time(&self, duration: core::time::Duration) -> Result<(), SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?;
        context.limit_time(duration)?;
        self.execution.set(context.execution());
        Ok(())
    }

    #[inline]
    pub(crate) fn provider_count(&self, role: EvaluationRole) -> Result<usize, SciOxideError> {
        self.context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .provider_count(role)
    }

    pub(crate) fn accounting(&self) -> Result<scioxide_core::SolveCounts, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .counts())
    }
    pub(crate) fn checkpoint(
        &self,
        state_token: u128,
    ) -> Result<scioxide_core::SolveCheckpointMetadata, SciOxideError> {
        self.context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .checkpoint(state_token)
    }
    pub(crate) fn combined_counts(&self) -> Result<EvaluationCounts, SciOxideError> {
        self.accounting()?.total()
    }
    #[inline]
    pub(crate) fn counts(&self) -> Result<EvaluationCounts, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .counts()
            .own())
    }

    pub(crate) fn algorithm(&self) -> Result<AlgorithmId, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .identity()
            .algorithm())
    }

    #[inline]
    pub(crate) fn iteration(&self) -> Result<Option<OptimizeTermination>, SciOxideError> {
        if let Some(stop) = self.termination()? {
            return Ok(Some(stop));
        }
        if !self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .record_iteration()?
            .is_granted()
        {
            self.stopped.set(Some(OptimizeTermination::IterationLimit));
        }
        Ok(self.stopped.get())
    }

    #[inline]
    pub(crate) fn accept(&self, trust: PartialResultTrust) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .commit_accepted(trust)
    }

    pub(crate) fn assess(&self, trust: PartialResultTrust) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .assess_accepted(trust)
    }

    #[inline]
    pub(crate) fn proposal(&self) -> Result<(), SciOxideError> {
        if let Some(stop) = self.termination()? {
            return Err(self.stop_error(stop)?);
        }
        if !self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .record_proposal()?
            .is_granted()
        {
            self.set_stopped(OptimizeTermination::ProposalLimit);
            return Err(self.stop_error(OptimizeTermination::ProposalLimit)?);
        }
        Ok(())
    }

    pub(crate) fn reject(&self) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .record_rejected()
    }

    pub(crate) fn cache_hit(&self) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .record_cache_hit()
    }

    pub(crate) fn samples(&self, amount: usize) -> Result<(), SciOxideError> {
        if let Some(stop) = self.termination()? {
            return Err(self.stop_error(stop)?);
        }
        if !self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .try_consume(BudgetKind::Sample, amount)?
            .is_granted()
        {
            self.set_stopped(OptimizeTermination::SampleLimit);
            return Err(self.stop_error(OptimizeTermination::SampleLimit)?);
        }
        Ok(())
    }

    /// Check a complete population before any of its objective slots are counted.
    pub(crate) fn require_evaluations(&self, amount: usize) -> Result<(), SciOxideError> {
        self.check()?;
        if !self
            .context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .can_consume(BudgetKind::Evaluation, amount)?
            .is_granted()
        {
            self.set_stopped(OptimizeTermination::EvaluationLimit);
            return Err(self.stop_error(OptimizeTermination::EvaluationLimit)?);
        }
        Ok(())
    }

    pub(crate) fn node(&self) -> Result<(), SciOxideError> {
        self.check()?;
        if !self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .try_consume(BudgetKind::Node, 1)?
            .is_granted()
        {
            self.set_stopped(OptimizeTermination::NodeLimit);
            return Err(self.stop_error(OptimizeTermination::NodeLimit)?);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn execution(&self) -> Result<ExecutionContext<'capability>, SciOxideError> {
        Ok(self.execution.get())
    }

    pub(crate) fn stop_error(
        &self,
        termination: OptimizeTermination,
    ) -> Result<SciOxideError, SciOxideError> {
        match termination {
            OptimizeTermination::Cancelled => Ok(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            )),
            OptimizeTermination::TimeLimit => Ok(SciOxideError::execution_stopped(
                ExecutionStopReason::DeadlineReached,
            )),
            _ => {
                let context = self
                    .context
                    .try_borrow()
                    .map_err(|_| borrow_error(self.argument))?;
                let kind = match termination {
                    OptimizeTermination::ProposalLimit => BudgetKind::Proposal,
                    OptimizeTermination::SampleLimit => BudgetKind::Sample,
                    OptimizeTermination::NodeLimit => BudgetKind::Node,
                    OptimizeTermination::IterationLimit => BudgetKind::Iteration,
                    _ => BudgetKind::Evaluation,
                };
                Ok(budget_error(
                    &context,
                    kind,
                    "observation was not admitted by the shared solve budget",
                ))
            }
        }
    }

    /// Low-level workspace operations can observe execution stops between
    /// counted provider calls. Keep their typed stop in the solve shell.
    pub(crate) fn record_execution_stop(&self, error: &SciOxideError) {
        if let scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } = (error).view() {
            self.set_stopped(match reason {
                ExecutionStopReason::DeadlineReached => OptimizeTermination::TimeLimit,
                _ => OptimizeTermination::Cancelled,
            });
        }
    }

    #[inline]
    pub(crate) fn invoke<T>(
        &self,
        role: EvaluationRole,
        provider: impl FnOnce() -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        self.admit(role)?;
        let outcome = provider();
        if let Err(error) = &outcome {
            self.record_execution_stop(error);
        }
        outcome
    }

    /// Child-local iteration limits do not stop an ancestor with remaining resources.
    pub(crate) fn inherit_child_stop(
        &self,
        stop: Option<OptimizeTermination>,
    ) -> Result<(), SciOxideError> {
        let Some(stop) = stop else {
            return Ok(());
        };
        let kind = match stop {
            OptimizeTermination::IterationLimit => BudgetKind::Iteration,
            OptimizeTermination::ProposalLimit => BudgetKind::Proposal,
            OptimizeTermination::EvaluationLimit => BudgetKind::Evaluation,
            OptimizeTermination::SampleLimit => BudgetKind::Sample,
            _ => {
                self.set_stopped(stop);
                return Ok(());
            }
        };
        if !self
            .context
            .try_borrow()
            .map_err(|_| borrow_error(self.argument))?
            .can_consume(kind, 1)?
            .is_granted()
        {
            self.set_stopped(stop);
        }
        Ok(())
    }

    pub(crate) fn with_child<T>(
        &self,
        state: &mut SolveState,
        identity: SolveIdentity,
        allowance: SolveBudgets,
        run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?;
        scioxide_core::SolveScope::from(&mut **context).run(state, identity, allowance, run)
    }

    pub(crate) fn with_context<T>(
        &self,
        run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> T,
    ) -> Result<T, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?;
        let output = run(&mut context);
        self.execution.set(context.execution());
        Ok(output)
    }

    pub(crate) fn observe<'a, F: ?Sized>(&'a self, objective: &'a F) -> ObservedObjective<'a, F> {
        ObservedObjective {
            objective,
            observer: self,
        }
    }

    pub(crate) fn observe_least_squares<'a, P: ?Sized>(
        &'a self,
        problem: &'a P,
    ) -> ObservedLeastSquares<'a, P> {
        ObservedLeastSquares {
            problem,
            observer: self,
        }
    }
}

impl Observation for SolveSession<'_, '_, '_> {
    fn counts(&self) -> Result<EvaluationCounts, SciOxideError> {
        self.counts()
    }
    fn execution(&self) -> Result<ExecutionContext<'_>, SciOxideError> {
        self.execution()
    }
    fn cache_hit(&self) -> Result<(), SciOxideError> {
        self.cache_hit()
    }
    fn proposal(&self) -> Result<(), SciOxideError> {
        self.proposal()
    }
    fn reject(&self) -> Result<(), SciOxideError> {
        self.reject()
    }
    #[inline(always)]
    fn admit(&self, role: EvaluationRole) -> Result<(), SciOxideError> {
        if let Some(stop) = self.termination()? {
            return Err(self.stop_error(stop)?);
        }
        let admitted = self
            .context
            .try_borrow_mut()
            .map_err(|_| borrow_error(self.argument))?
            .record_provider(role)?;
        if let scioxide_core::Admission::Denied(report) = admitted {
            self.stopped.set(Some(OptimizeTermination::EvaluationLimit));
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                ConvergenceDiagnostics::new().with_budget(report),
                "provider evaluation was denied by its solve allowance",
            ));
        }
        Ok(())
    }
}

pub(crate) struct ObservedLeastSquares<'a, P: ?Sized> {
    problem: &'a P,
    observer: &'a dyn Observation,
}

impl<S, P> crate::LeastSquaresProblem<S> for ObservedLeastSquares<'_, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: crate::LeastSquaresProblem<S> + ?Sized,
{
    fn param_dim(&self) -> usize {
        self.problem.param_dim()
    }
    fn residual_dim(&self) -> usize {
        self.problem.residual_dim()
    }
    fn parameter_scale(&self, index: usize) -> Option<S> {
        self.problem.parameter_scale(index)
    }
    fn residual_scale(&self, index: usize) -> S {
        self.problem.residual_scale(index)
    }
    fn residual_weight(&self, index: usize) -> S {
        self.problem.residual_weight(index)
    }
    fn residual_group_size(&self) -> usize {
        self.problem.residual_group_size()
    }
    fn jacobian_operator_workspace_len(&self) -> Result<Option<usize>, SciOxideError> {
        self.problem.jacobian_operator_workspace_len()
    }
    fn jacobian_product_into(
        &self,
        x: &[S],
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.problem
            .jacobian_product_into(x, input, output, scratch)
    }
    fn jacobian_adjoint_product_into(
        &self,
        x: &[S],
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.problem
            .jacobian_adjoint_product_into(x, input, output, scratch)
    }
    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Residual, || {
            self.problem.residual_into(x, output)
        })
    }
    fn jacobian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Jacobian, || {
            self.problem.jacobian_into(x, output)
        })
    }
    fn automatic_jacobian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Jacobian, || {
            self.problem.automatic_jacobian_into(x, output)
        })
    }
}

pub(crate) struct ObservedObjective<'a, F: ?Sized> {
    objective: &'a F,
    observer: &'a dyn Observation,
}

impl<S, F> ObjectiveFunction<S> for ObservedObjective<'_, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    fn dim(&self) -> usize {
        self.objective.dim()
    }
    fn bounds(&self) -> Option<&scioxide_core::Bounds<S>> {
        self.objective.bounds()
    }
    fn is_in_domain(&self, x: &[S]) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Domain, || {
            self.objective.is_in_domain(x)
        })
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Objective, || {
            self.objective.eval(x)
        })
    }
    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Gradient, || {
            self.objective.gradient_into(x, output)
        })
    }
    fn automatic_gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Gradient, || {
            self.objective.automatic_gradient_into(x, output)
        })
    }
    fn has_hessian_vector_product(&self) -> bool {
        self.objective.has_hessian_vector_product()
    }
    fn hessian_vector_product_into(
        &self,
        x: &[S],
        direction: &[S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::LinearOperator, || {
            self.objective
                .hessian_vector_product_into(x, direction, output)
        })?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "provider returned a nonfinite Hessian product",
            ));
        }
        Ok(())
    }

    fn hessian_into(&self, x: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Hessian, || {
            self.objective.hessian_into(x, output)
        })
    }
    fn automatic_hessian_into(
        &self,
        x: &[S],
        output: &mut Matrix<S>,
    ) -> Result<bool, SciOxideError> {
        invoke(Some(self.observer), EvaluationRole::Hessian, || {
            self.objective.automatic_hessian_into(x, output)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, Deadline, MonotonicInstant};

    #[test]
    fn budget_stops_preserve_the_exhausted_resource_without_an_accepted_state() {
        for (kind, stop, reason) in [
            (
                BudgetKind::Iteration,
                OptimizeTermination::IterationLimit,
                ConvergenceReason::IterationLimit,
            ),
            (
                BudgetKind::Evaluation,
                OptimizeTermination::EvaluationLimit,
                ConvergenceReason::EvaluationLimit,
            ),
            (
                BudgetKind::Proposal,
                OptimizeTermination::ProposalLimit,
                ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
            ),
            (
                BudgetKind::Sample,
                OptimizeTermination::SampleLimit,
                ConvergenceReason::ResourceLimit(BudgetKind::Sample),
            ),
            (
                BudgetKind::Node,
                OptimizeTermination::NodeLimit,
                ConvergenceReason::ResourceLimit(BudgetKind::Node),
            ),
        ] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    identity(
                        scioxide_core::algorithm_id!("optimize", "Bfgs", 1),
                        ProblemToken::new(1),
                        1,
                        0,
                    )
                    .unwrap(),
                    &mut budgets,
                )
                .unwrap();
            assert!(context.try_consume(kind, 1).unwrap().is_granted());
            let session = SolveSession::new(&mut context, scioxide_core::argument!("context"));
            let error = match kind {
                BudgetKind::Iteration => {
                    assert_eq!(session.iteration().unwrap(), Some(stop));
                    session.check().unwrap_err()
                }
                BudgetKind::Evaluation => session.require_evaluations(1).unwrap_err(),
                BudgetKind::Proposal => session.proposal().unwrap_err(),
                BudgetKind::Sample => session.samples(1).unwrap_err(),
                BudgetKind::Node => session.node().unwrap_err(),
                _ => unreachable!(),
            };
            assert_eq!(session.stopped(), Some(stop));
            assert!(
                matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { reason: actual, .. } if actual == reason),
                "{error:?}"
            );
        }
    }

    #[test]
    fn operator_kernel_borrows_the_clock_after_admission() {
        struct FailAfterAdmission(AtomicUsize);
        impl Clock for FailAfterAdmission {
            fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
                if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO))
                } else {
                    Err(CapabilityError::new("matrix operation clock failed"))
                }
            }
        }
        let clock = FailAfterAdmission(AtomicUsize::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let execution = ExecutionContext::sequential()
            .with_deadline(Deadline::at(
                &clock_binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
            ))
            .unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                execution,
                identity(
                    scioxide_core::algorithm_id!("optimize", "Bfgs", 1),
                    ProblemToken::new(1),
                    1,
                    0,
                )
                .unwrap(),
                &mut budgets,
            )
            .unwrap();
        let session = SolveSession::new(&mut context, scioxide_core::argument!("context"));
        let matrix = Matrix::from_vec(1, 1, alloc::vec![2.0_f64]).unwrap();
        let mut output = [17.0];
        let mut scratch = [0.0];
        let result = crate::dense::matrix_vector_into(
            matrix.view(),
            &[3.0],
            &mut output,
            &mut scratch,
            &session,
        );
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::CapabilityFailure { .. })
        ));
        assert_eq!(output, [17.0]);
        assert_eq!(
            session
                .counts()
                .unwrap()
                .provider(EvaluationRole::LinearOperator),
            1
        );
        assert_eq!(clock.0.load(Ordering::Relaxed), 2);
    }
}
