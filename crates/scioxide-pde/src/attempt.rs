//! Workspace-owned solve state, attempt records and the crate's internal solve identity.
//!
//! PDE entry points take no caller identity: every epoch uses
//! the anonymous problem token, and caches are scoped by the owning workspace's epochs.
//! A [`Shell`] owns the local [`SolveState`], the last [`AttemptRecord`] and recyclable
//! progress storage, so recording progress reuses the previous attempt's buffers.

use scioxide_core::{
    AlgorithmId, AttemptKind, AttemptOutcome, AttemptRecord, ExecutionContext, ProblemToken,
    SciOxideError, SolveBudgets, SolveContext, SolveCounts, SolveIdentity, SolveScope, SolveState,
};

/// Identity of one internal epoch: the anonymous problem, `algorithm` and `shape`.
pub(crate) const fn identity(algorithm: AlgorithmId, shape: u128) -> SolveIdentity {
    SolveIdentity::new(ProblemToken::new(0), algorithm, shape, 0)
}

/// Run `body` in one root or child epoch of `scope` with an unbounded local allowance.
///
/// Construction and evaluation forms that keep no attempt record use this; ancestor
/// budgets, cancellation and deadlines still apply through `scope`.
pub(crate) fn scoped<'scope, 'capability, T>(
    scope: impl Into<SolveScope<'scope, 'capability>>,
    algorithm: AlgorithmId,
    body: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scope.into().run(
        &mut SolveState::new(),
        identity(algorithm, 0),
        SolveBudgets::unbounded(),
        body,
    )
}

/// Run `body` in a private root epoch under sequential native execution, for the
/// unscoped forms of operator and residual protocols; its counts are discarded.
pub(crate) fn sequential<T>(
    algorithm: AlgorithmId,
    body: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scoped(ExecutionContext::sequential_native(), algorithm, body)
}

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
    result: Result<R, SciOxideError>,
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

    fn take_progress(&mut self) -> Option<P> {
        self.spare
            .take()
            .or_else(|| self.attempt.take().and_then(AttemptRecord::into_progress))
    }

    /// Record a preflight rejection of a `kind` attempt and return `error`.
    /// No callback ran, no epoch began and no caller state changed.
    pub(crate) fn reject<R>(
        &mut self,
        kind: AttemptKind,
        error: SciOxideError,
    ) -> Result<R, SciOxideError> {
        if let Some(progress) = self.take_progress() {
            self.spare = Some(progress);
        }
        self.attempt = Some(AttemptRecord::new(
            None,
            kind,
            AttemptOutcome::PreflightRejected,
            SolveCounts::new(),
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

    /// Record the `kind` attempt of `executed`. `success` classifies an `Ok` result;
    /// `fill` writes the attempt's accepted progress into recycled (or new) storage,
    /// returning `None` when the attempt accepted nothing.
    pub(crate) fn conclude<R>(
        &mut self,
        kind: AttemptKind,
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
            kind,
            outcome,
            self.state.counts(),
            progress,
        ));
        result
    }
}
