//! Workspace-owned solve state and attempt records shared by every family.
//!
//! A shell owns the local [`SolveState`] (so epochs are monotone across calls) and the
//! last [`AttemptRecord`]. Progress storage is recycled between attempts: the previous
//! record's progress, or a spare prepared at reservation, is handed to the next attempt,
//! so recording progress does not allocate.

use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, SciOxideError, SolveBudgets, SolveContext,
    SolveIdentity, SolveScope, SolveState,
};

/// Solve state, last attempt and recyclable progress storage.
#[derive(Debug)]
pub(crate) struct Shell<P> {
    state: SolveState,
    attempt: Option<AttemptRecord<P>>,
    spare: Option<P>,
}

impl<P> Default for Shell<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// The outcome of one epoch, before its attempt is recorded.
pub(crate) struct Executed<R, P> {
    pub(crate) result: Result<R, SciOxideError>,
    started: bool,
    recycled: Option<P>,
}

impl<P> Shell<P> {
    pub(crate) const fn new() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
            spare: None,
        }
    }

    /// The last attempt, if any.
    pub(crate) const fn last_attempt(&self) -> Option<&AttemptRecord<P>> {
        self.attempt.as_ref()
    }

    /// The last attempt's progress, consuming the shell.
    pub(crate) fn into_progress(self) -> Option<P> {
        self.attempt.and_then(AttemptRecord::into_progress)
    }

    /// Capture the last attempt's progress in an owned failure.
    pub(crate) fn into_failure(self, error: SciOxideError) -> scioxide_core::PartialFailure<P> {
        match self.into_progress() {
            Some(progress) => scioxide_core::PartialFailure::with_progress(error, || Ok(progress)),
            None => scioxide_core::PartialFailure::new(error),
        }
    }

    /// Whether recyclable progress storage satisfies `ready`.
    pub(crate) fn has_spare(&self, ready: impl Fn(&P) -> bool) -> bool {
        self.spare.as_ref().is_some_and(&ready)
            || self
                .attempt
                .as_ref()
                .and_then(AttemptRecord::progress)
                .is_some_and(ready)
    }

    /// Install recyclable progress storage unless suitable storage is already held.
    pub(crate) fn reserve_spare(
        &mut self,
        ready: impl Fn(&P) -> bool,
        create: impl FnOnce() -> Result<P, SciOxideError>,
    ) -> Result<(), SciOxideError> {
        if !self.has_spare(&ready) {
            self.spare = Some(create()?);
        }
        Ok(())
    }

    fn take_progress(&mut self) -> Option<P> {
        self.spare
            .take()
            .or_else(|| self.attempt.take().and_then(AttemptRecord::into_progress))
    }

    /// Record a preflight rejection and return `error`. No callback ran.
    pub(crate) fn reject<R>(&mut self, error: SciOxideError) -> Result<R, SciOxideError> {
        if let Some(progress) = self.take_progress() {
            self.spare = Some(progress);
        }
        self.attempt = Some(AttemptRecord::new(
            None,
            AttemptKind::Call,
            AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
        Err(error)
    }

    /// Run `execute` in a root or child epoch of `scope`.
    pub(crate) fn execute<R>(
        &mut self,
        scope: SolveScope<'_, '_>,
        identity: SolveIdentity,
        budgets: SolveBudgets,
        execute: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
    ) -> Executed<R, P> {
        let recycled = self.take_progress();
        let before = self.state.epoch();
        let result = scope.run(&mut self.state, identity, budgets, execute);
        let started = self.state.epoch() != before;
        Executed {
            result,
            started,
            recycled,
        }
    }

    /// Record the attempt of `executed`. `success` classifies an `Ok` result; `fill`
    /// writes the attempt's accepted progress into recycled (or new) storage.
    pub(crate) fn conclude<R>(
        &mut self,
        executed: Executed<R, P>,
        success: impl FnOnce(&R) -> bool,
        fill: impl FnOnce(Option<P>) -> Option<P>,
    ) -> Result<R, SciOxideError> {
        let Executed {
            result,
            started,
            recycled,
        } = executed;
        let (outcome, progress) = match (&result, started) {
            (_, false) => {
                self.spare = recycled;
                (AttemptOutcome::PreflightRejected, None)
            }
            (Ok(value), true) => (
                if success(value) {
                    AttemptOutcome::Succeeded
                } else {
                    AttemptOutcome::Stopped
                },
                fill(recycled),
            ),
            (Err(_), true) => (AttemptOutcome::Failed, fill(recycled)),
        };
        self.attempt = Some(AttemptRecord::new(
            self.state.epoch(),
            AttemptKind::Call,
            outcome,
            self.state.counts(),
            progress,
        ));
        result
    }
}
