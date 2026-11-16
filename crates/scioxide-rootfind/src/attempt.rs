//! Attempt bookkeeping used by every nonlinear solver.
//!
//! Each workspace owns a [`Shell`]: its reusable [`SolveState`] and the record
//! of its most recent attempt. [`attempt`] runs one call or resume:
//!
//! 1. record a `PreflightRejected` attempt before any validation;
//! 2. run the solver's preflight, which validates input, output and storage and
//!    returns the [`Entry`] (identity and local budgets) of the epoch;
//! 3. run the kernel in a fresh root or child epoch of the caller's scope, or
//!    resume a checkpointed epoch;
//! 4. record the attempt epoch, the counts at kernel exit, the outcome and the
//!    progress accepted by this attempt only.
//!
//! The outcome is `PreflightRejected` when no epoch began, `Failed` when the
//! epoch returned an error, and otherwise `Succeeded` or `Stopped` according to
//! the solver's success predicate. Progress is never recorded without an epoch.

use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, SciOxideError, SolveBudgets,
    SolveCheckpointMetadata, SolveContext, SolveCounts, SolveIdentity, SolveScope, SolveState,
};

/// Reusable solve state and the record of the most recent attempt.
#[derive(Debug, Clone)]
pub(crate) struct Shell<P> {
    state: SolveState,
    attempt: Option<AttemptRecord<P>>,
}

impl<P> Default for Shell<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> Shell<P> {
    /// Inactive state without an attempt record.
    pub(crate) const fn new() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
        }
    }

    /// Forget the attempt record, for example after storage is replaced.
    pub(crate) fn clear(&mut self) {
        self.attempt = None;
    }

    /// The solve state carried into replacement storage, without the attempt record.
    ///
    /// Keeping the state preserves epoch monotonicity across reservations.
    pub(crate) const fn retained(&self) -> Self {
        Self {
            state: self.state,
            attempt: None,
        }
    }

    /// Record of the most recent attempt, including preflight rejection.
    pub(crate) const fn last_attempt(&self) -> Option<&AttemptRecord<P>> {
        self.attempt.as_ref()
    }

    /// Progress accepted by the most recent attempt only.
    pub(crate) fn progress(&self) -> Option<&P> {
        self.attempt.as_ref()?.progress()
    }

    /// `value` when the most recent attempt recorded accepted progress.
    pub(crate) fn accepted<'a, T: ?Sized>(&self, value: &'a T) -> Option<&'a T> {
        self.attempt.as_ref()?.progress().map(|_| value)
    }
}

/// Prepared storage that owns one attempt [`Shell`].
pub(crate) trait Attempted {
    /// Accepted progress recorded by an attempt.
    type Progress;
    /// The workspace's shell.
    fn shell(&mut self) -> &mut Shell<Self::Progress>;
}

/// Implement [`Attempted`] for a workspace whose shell is the field `shell`.
macro_rules! attempted {
    ($workspace:ident<$scalar:ident>, [$($bound:tt)*], $progress:ty) => {
        impl<$scalar> crate::attempt::Attempted for $workspace<$scalar>
        where
            $($bound)*
        {
            type Progress = $progress;
            fn shell(&mut self) -> &mut crate::attempt::Shell<$progress> {
                &mut self.shell
            }
        }
    };
}
pub(crate) use attempted;

/// Fresh call or checkpoint continuation of one epoch.
#[expect(
    clippy::large_enum_variant,
    reason = "Entries live on the stack for one call; boxing would allocate per resume."
)]
pub(crate) enum Entry {
    /// Begin a fresh epoch with a local allowance.
    Call {
        identity: SolveIdentity,
        budgets: SolveBudgets,
    },
    /// Resume a validated checkpointed epoch.
    Resume {
        identity: SolveIdentity,
        state_token: u128,
        metadata: SolveCheckpointMetadata,
        budgets: SolveBudgets,
    },
}

impl Entry {
    /// A fresh call with `identity` and local `budgets`.
    pub(crate) const fn call(identity: SolveIdentity, budgets: SolveBudgets) -> Self {
        Self::Call { identity, budgets }
    }
}

/// How a completed epoch is classified and what it accepted.
pub(crate) struct Conclusion<P> {
    /// Whether a successful epoch met the solver's success criterion.
    pub(crate) succeeded: bool,
    /// Progress accepted by this attempt.
    pub(crate) progress: Option<P>,
}

/// Runs one attempt of `kind` over `workspace` and `output`.
///
/// `preflight` validates and prepares the attempt; its error leaves a
/// `PreflightRejected` record. `kernel` runs inside the epoch. `conclude`
/// classifies the result once the epoch has closed and the solve state is back
/// in the workspace; it is not called when no epoch began.
///
/// # Errors
/// Returns the preflight, epoch-entry or kernel error unchanged.
pub(crate) fn attempt<'scope, 'capability, W, O, T>(
    workspace: &mut W,
    output: &mut O,
    scope: SolveScope<'scope, 'capability>,
    kind: AttemptKind,
    preflight: impl FnOnce(&mut W, &mut O) -> Result<Entry, SciOxideError>,
    kernel: impl FnOnce(&mut W, &mut O, &mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    conclude: impl FnOnce(&W, &O, &Result<T, SciOxideError>) -> Conclusion<W::Progress>,
) -> Result<T, SciOxideError>
where
    W: Attempted,
    O: ?Sized,
{
    workspace.shell().attempt = Some(AttemptRecord::new(
        None,
        kind,
        AttemptOutcome::PreflightRejected,
        SolveCounts::new(),
        None,
    ));
    let entry = preflight(workspace, output)?;
    let mut state = core::mem::take(&mut workspace.shell().state);
    let mut epoch = None;
    let mut counts = SolveCounts::new();
    let run = |context: &mut SolveContext<'_, 'capability>| {
        epoch = context.accepted_state().map(|accepted| accepted.epoch());
        let result = kernel(workspace, output, context);
        counts = context.counts();
        result
    };
    let result = match entry {
        Entry::Call { identity, budgets } => scope.run(&mut state, identity, budgets, run),
        Entry::Resume {
            identity,
            state_token,
            metadata,
            mut budgets,
        } => scope.resume(
            &mut state,
            identity,
            state_token,
            metadata,
            &mut budgets,
            run,
        ),
    };
    workspace.shell().state = state;
    let (outcome, progress) = match epoch {
        None => (AttemptOutcome::PreflightRejected, None),
        Some(_) => {
            let Conclusion {
                succeeded,
                progress,
            } = conclude(workspace, output, &result);
            let outcome = if result.is_err() {
                AttemptOutcome::Failed
            } else if succeeded {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Stopped
            };
            (outcome, progress)
        }
    };
    workspace.shell().attempt = Some(AttemptRecord::new(epoch, kind, outcome, counts, progress));
    result
}
