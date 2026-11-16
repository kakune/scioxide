//! Solve-state bookkeeping for calls that run in a solve scope.
use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, SciOxideError, SolveBudgets, SolveIdentity,
    SolveScope, SolveState,
};

pub(crate) struct SolveSlot<P> {
    state: SolveState,
    attempt: Option<AttemptRecord<P>>,
}
impl<P> Default for SolveSlot<P> {
    fn default() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
        }
    }
}
impl<P> SolveSlot<P> {
    pub(crate) fn reject(&mut self) {
        self.attempt = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
    }
    pub(crate) fn last_attempt(&self) -> Option<&AttemptRecord<P>> {
        self.attempt.as_ref()
    }
    pub(crate) fn run<'capability, T>(
        &mut self,
        scope: SolveScope<'_, 'capability>,
        identity: SolveIdentity,
        budgets: SolveBudgets,
        execute: impl FnOnce(
            &mut scioxide_core::SolveContext<'_, 'capability>,
        ) -> Result<T, SciOxideError>,
        classify: impl FnOnce(&Result<T, SciOxideError>) -> (AttemptOutcome, Option<P>),
    ) -> Result<T, SciOxideError> {
        let epoch = self.state.epoch();
        let result = scope.run(&mut self.state, identity, budgets, execute);
        let started = self.state.epoch() != epoch;
        let (outcome, progress) = if started {
            classify(&result)
        } else {
            (AttemptOutcome::PreflightRejected, None)
        };
        self.attempt = Some(AttemptRecord::new(
            if started { self.state.epoch() } else { None },
            AttemptKind::Call,
            outcome,
            self.state.counts(),
            progress,
        ));
        result
    }
}
