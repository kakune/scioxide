//! Solve contexts, budget checks before each callback, and the sequential
//! convenience forms.
use scioxide_core::{
    AlgorithmId, Budget, BudgetKind, ExecutionContext, InvalidInputReason, ProblemToken,
    SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState,
};

/// Internal control flow of a kernel. Stops are values, distinct from callback errors.
#[derive(Debug)]
pub(crate) enum KernelFault {
    Stopped(crate::RootTermination),
    Error(SciOxideError),
}
pub(crate) type KernelResult<T> = Result<T, KernelFault>;
impl From<SciOxideError> for KernelFault {
    fn from(error: SciOxideError) -> Self {
        match error.view() {
            scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                Self::Stopped(crate::RootTermination::from_execution(reason))
            }
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason:
                    scioxide_core::ConvergenceReason::IterationLimit
                    | scioxide_core::ConvergenceReason::EvaluationLimit
                    | scioxide_core::ConvergenceReason::ResourceLimit(_),
                diagnostics,
                ..
            } => match diagnostics.budget() {
                Some(report) => Self::Stopped(crate::RootTermination::WorkLimit(report.kind())),
                None => Self::Error(error),
            },
            _ => Self::Error(error),
        }
    }
}
pub(crate) fn evaluated(outcome: scioxide_core::Invocation<()>) -> KernelResult<()> {
    match outcome {
        scioxide_core::Invocation::Evaluated(()) => Ok(()),
        scioxide_core::Invocation::BudgetExhausted(report) => Err(KernelFault::Stopped(
            crate::RootTermination::WorkLimit(report.kind()),
        )),
        scioxide_core::Invocation::ExecutionStopped(reason) => Err(KernelFault::Stopped(
            crate::RootTermination::from_execution(reason),
        )),
    }
}

pub(crate) fn check_stop(context: &SolveContext<'_, '_>) -> KernelResult<()> {
    match context.stop_reason()? {
        Some(reason) => Err(KernelFault::Stopped(
            crate::RootTermination::from_execution(reason),
        )),
        None => Ok(()),
    }
}
pub(crate) fn admitted(admission: scioxide_core::Admission) -> KernelResult<()> {
    match admission {
        scioxide_core::Admission::Granted(()) => Ok(()),
        scioxide_core::Admission::Denied(report) => Err(KernelFault::Stopped(
            crate::RootTermination::WorkLimit(report.kind()),
        )),
    }
}
pub(crate) fn admit(
    context: &mut SolveContext<'_, '_>,
    role: scioxide_core::EvaluationRole,
) -> KernelResult<()> {
    check_stop(context)?;
    admitted(context.record_provider(role)?)
}

/// Run `kernel` in a fresh root or child epoch of `scope` with a local
/// `allowance`. The identity is internal: callers never build one.
#[inline(always)]
pub(crate) fn run<'scope, 'capability, T>(
    scope: SolveScope<'scope, 'capability>,
    algorithm: AlgorithmId,
    allowance: SolveBudgets,
    kernel: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scope.run(
        &mut SolveState::new(),
        SolveIdentity::new(ProblemToken::new(0), algorithm, 1, 0),
        allowance,
        kernel,
    )
}
pub(crate) fn identity(
    algorithm: AlgorithmId,
    problem: ProblemToken,
    dimension: usize,
    options: u128,
) -> SolveIdentity {
    SolveIdentity::new(problem, algorithm, dimension as u128, options)
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

pub(crate) fn residual<S: scioxide_core::RealScalar>(
    raw: S,
    scale: S,
) -> Result<scioxide_core::ResidualReport<S>, SciOxideError> {
    scioxide_core::ResidualReport::new(
        raw,
        scioxide_core::NormSelection::L2,
        scioxide_core::ResidualScaling::new(
            scioxide_core::ResidualScalingRule::Reference,
            if scale > S::ZERO { scale } else { S::ONE },
        )?,
    )
}

/// Error for a resource stop.
pub(crate) fn stop_error(
    context: &SolveContext<'_, '_>,
    reason: crate::RootTermination,
) -> SciOxideError {
    use crate::RootTermination;
    match reason {
        RootTermination::Cancelled => {
            SciOxideError::execution_stopped(scioxide_core::ExecutionStopReason::Cancelled)
        }
        RootTermination::TimeLimit => {
            SciOxideError::execution_stopped(scioxide_core::ExecutionStopReason::DeadlineReached)
        }
        RootTermination::WorkLimit(kind) => budget_error(context, kind),
        RootTermination::IterationLimit => budget_error(context, BudgetKind::Iteration),
        _ => SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::Stagnation,
            scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
            "no checked root was available before termination",
        ),
    }
}

fn budget_error(context: &SolveContext<'_, '_>, kind: BudgetKind) -> SciOxideError {
    SciOxideError::convergence_failure(
        crate::termination::budget_reason(kind),
        scioxide_core::ConvergenceDiagnostics::new()
            .with_counts(context.counts())
            .with_budget(context.effective_budget_report(kind)),
        "solve budget denied further work",
    )
}

/// Counts work and records stops for one solve. A scoped callback receives the
/// context through the same borrow that charged it, so it can run nested work.
/// Other callbacks run after the borrow is released.
pub(crate) struct Session<'a, 'state, 'capability> {
    context: core::cell::RefCell<&'a mut SolveContext<'state, 'capability>>,

    pub(crate) stopped: core::cell::Cell<Option<crate::RootTermination>>,
    pub(crate) accepted: core::cell::Cell<bool>,
    policy: scioxide_core::ExecutionPolicy,
}
impl<'a, 'state, 'capability> Session<'a, 'state, 'capability> {
    /// Borrow `context`.
    #[inline(always)]
    pub(crate) fn new(context: &'a mut SolveContext<'state, 'capability>) -> Self {
        Self {
            context: core::cell::RefCell::new(context),

            stopped: core::cell::Cell::new(None),
            accepted: core::cell::Cell::new(false),
            policy: scioxide_core::ExecutionPolicy::Sequential,
        }
    }
    pub(crate) fn with_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    pub(crate) fn policy(&self) -> scioxide_core::ExecutionPolicy {
        self.policy
    }
    #[cold]
    fn borrow_error(&self) -> SciOxideError {
        SciOxideError::invalid_input(
            scioxide_core::argument!("context"),
            InvalidInputReason::Inconsistent,
            "context borrow cannot span a provider invocation",
        )
    }
    #[inline(always)]
    pub(crate) fn accounting(&self) -> Result<scioxide_core::SolveCounts, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| self.borrow_error())?
            .counts())
    }
    #[inline(always)]
    pub(crate) fn algorithm(&self) -> Result<AlgorithmId, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| self.borrow_error())?
            .identity()
            .algorithm())
    }
    #[inline(always)]
    pub(crate) fn execution(&self) -> Result<ExecutionContext<'capability>, SciOxideError> {
        Ok(self
            .context
            .try_borrow()
            .map_err(|_| self.borrow_error())?
            .execution())
    }
    #[cold]
    pub(crate) fn stop(&self, reason: crate::RootTermination) -> Result<(), SciOxideError> {
        self.stopped.set(Some(reason));
        Err(stop_error(
            &**self.context.try_borrow().map_err(|_| self.borrow_error())?,
            reason,
        ))
    }
    #[inline(always)]
    pub(crate) fn check_stop(&self) -> Result<bool, SciOxideError> {
        let context = self.context.try_borrow().map_err(|_| self.borrow_error())?;
        self.check_context_stop(&context)
    }
    #[inline(always)]
    fn check_context_stop(&self, context: &SolveContext<'_, '_>) -> Result<bool, SciOxideError> {
        if let Some(reason) = context.stop_reason()? {
            self.stopped.set(Some(match reason {
                scioxide_core::ExecutionStopReason::DeadlineReached => {
                    crate::RootTermination::TimeLimit
                }
                _ => crate::RootTermination::Cancelled,
            }));
        }
        Ok(self.stopped.get().is_none())
    }
    #[inline(always)]
    pub(crate) fn try_admit(
        &self,
        role: scioxide_core::EvaluationRole,
    ) -> Result<bool, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        self.admit_context(&mut context, role)
    }
    /// Poll and charge through the same borrow used for a scoped callback.
    #[inline(always)]
    fn admit_context(
        &self,
        context: &mut SolveContext<'_, '_>,
        role: scioxide_core::EvaluationRole,
    ) -> Result<bool, SciOxideError> {
        if !self.check_context_stop(context)? {
            return Ok(false);
        }
        if !context.record_provider(role)?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Evaluation,
            )));
            return Ok(false);
        }
        Ok(true)
    }
    #[inline(always)]
    pub(crate) fn admit(&self, role: scioxide_core::EvaluationRole) -> Result<(), SciOxideError> {
        if !self.try_admit(role)?
            && let Some(reason) = self.stopped.get()
        {
            return self.stop(reason);
        }
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn invoke<T>(
        &self,
        role: scioxide_core::EvaluationRole,
        provider: impl FnOnce() -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        self.admit(role)?;
        provider()
    }
    #[inline(always)]
    pub(crate) fn invoke_context<T>(
        &self,
        role: scioxide_core::EvaluationRole,
        provider: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        if !self.admit_context(&mut context, role)? {
            let reason = self.stopped.get().ok_or_else(|| self.borrow_error())?;
            return Err(stop_error(&context, reason));
        }
        provider(&mut context)
    }
    /// Charges one further proposal within the current iteration.
    #[inline(always)]
    pub(crate) fn proposal(&self) -> Result<(), SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        if self.check_context_stop(&context)? && !context.record_proposal()?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Proposal,
            )));
        }
        match self.stopped.get() {
            Some(reason) => Err(stop_error(&context, reason)),
            None => Ok(()),
        }
    }

    /// Checks for a stop and charges an iteration whose proposals are counted
    /// separately; returns `false` on a stop.
    pub(crate) fn try_iteration_only(&self) -> Result<bool, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        if !self.check_context_stop(&context)? {
            return Ok(false);
        }
        if !context.record_iteration()?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                BudgetKind::Iteration,
            )));
            return Ok(false);
        }
        Ok(true)
    }
    /// Charges an iteration whose proposals are counted separately, stopping on
    /// cancellation or an exhausted budget.
    pub(crate) fn iteration_only(&self) -> Result<(), SciOxideError> {
        if !self.try_iteration_only()?
            && let Some(reason) = self.stopped.get()
        {
            return self.stop(reason);
        }
        Ok(())
    }
    /// Checks for a stop and charges a separately counted nonlinear trial;
    /// returns `false` on a stop.
    pub(crate) fn try_proposal(&self) -> Result<bool, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        if !self.check_context_stop(&context)? {
            return Ok(false);
        }
        if !context.record_proposal()?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Proposal,
            )));
            return Ok(false);
        }
        Ok(true)
    }
    #[inline(always)]
    pub(crate) fn try_iteration(&self) -> Result<bool, SciOxideError> {
        let mut context = self
            .context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?;
        if !self.check_context_stop(&context)? {
            return Ok(false);
        }
        if !context.record_iteration()?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                BudgetKind::Iteration,
            )));
            return Ok(false);
        }
        if !context.record_proposal()?.is_granted() {
            self.stopped.set(Some(crate::RootTermination::WorkLimit(
                scioxide_core::BudgetKind::Proposal,
            )));
            return Ok(false);
        }
        Ok(true)
    }
    #[inline(always)]
    pub(crate) fn iteration(&self) -> Result<(), SciOxideError> {
        if !self.try_iteration()?
            && let Some(reason) = self.stopped.get()
        {
            return self.stop(reason);
        }
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn assess(&self) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?
            .assess_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
        self.accepted.set(true);
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn accept(&self) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?
            .commit_accepted(scioxide_core::PartialResultTrust::ResidualChecked)?;
        self.accepted.set(true);
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn reject(&self) -> Result<(), SciOxideError> {
        self.context
            .try_borrow_mut()
            .map_err(|_| self.borrow_error())?
            .record_rejected()
    }
}

/// Tighten a local numerical limit without weakening an existing allowance.
pub(crate) fn cap(budgets: SolveBudgets, kind: BudgetKind, limit: Budget) -> SolveBudgets {
    if matches!(budgets.report(kind).limit(), scioxide_core::BudgetLimit::Finite(value) if value <= limit.get())
    {
        budgets
    } else {
        budgets.with_limit(kind, limit)
    }
}
