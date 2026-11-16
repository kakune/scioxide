//! Workspace solve state, attempt records and accepted progress.
//!
//! Prepared solvers keep one [`Shell`] per workspace. A call records a
//! rejected request first, validates and reserves, then runs its kernel in a
//! fresh root or child epoch of the caller's scope. Kernels write a staged
//! result that is copied to the caller only after the epoch closes, so a
//! failed attempt never changes the output.

use core::time::Duration;

use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, Capability, PartialResultTrust, RealScalar,
    SciOxideError, SolveBudgets, SolveCheckpointMetadata, SolveContext, SolveCounts, SolveIdentity,
    SolveScope, SolveState,
};

use crate::OptimizeResult;

/// Summarizes the last accepted point of a local-minimizer call.
///
/// A workspace has none for a call rejected before it started, or that
/// stopped before accepting a point.
#[derive(Debug, Clone, Copy)]
pub struct MinimizerProgress<S: RealScalar> {
    value: S,
    iterations: usize,
    gradient_norm: Option<S>,
    trust: PartialResultTrust,
}

impl<S: RealScalar> MinimizerProgress<S> {
    /// Returns the objective value at the accepted point.
    pub const fn value(self) -> S {
        self.value
    }
    /// Returns the number of iterations completed.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the Euclidean norm of the gradient at the accepted point, if
    /// one was computed.
    pub const fn gradient_norm(self) -> Option<S> {
        self.gradient_norm
    }
    /// Returns how far the accepted point has been checked.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

/// Borrows the last accepted point of the most recent call from a workspace.
#[derive(Debug, Clone, Copy)]
pub struct MinimizerAccepted<'a, S: RealScalar> {
    point: &'a [S],
    gradient: Option<&'a [S]>,
    progress: MinimizerProgress<S>,
}

impl<'a, S: RealScalar> MinimizerAccepted<'a, S> {
    /// Returns the accepted point.
    pub const fn point(&self) -> &'a [S] {
        self.point
    }
    /// Returns the gradient at the accepted point, if one was computed.
    pub const fn gradient(&self) -> Option<&'a [S]> {
        self.gradient
    }
    /// Returns the objective value, iteration count and gradient norm.
    pub const fn progress(&self) -> MinimizerProgress<S> {
        self.progress
    }
}

/// Accepted state kept by solver storage after a kernel returns.
pub(crate) struct AcceptedState<'a, S> {
    pub(crate) point: &'a [S],
    pub(crate) value: S,
    pub(crate) gradient: Option<&'a [S]>,
    pub(crate) iterations: usize,
}

/// Storage that exposes the accepted state of its most recent kernel run.
pub(crate) trait Accepted<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>>;
}

/// Shape key that reserves and checks one family of solver storage.
pub(crate) trait Shape<St> {
    /// Whether `storage` already has exactly this shape.
    fn fits(&self, storage: &St) -> bool;
    /// Replace `storage` with this shape; on failure the existing storage is unchanged.
    fn prepare(&self, storage: &mut St) -> Result<(), SciOxideError>;
}

/// Fresh call or checkpoint continuation of one epoch.
// Entries live on the stack for one call; boxing would allocate per resume.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Entry {
    Call {
        identity: SolveIdentity,
        budgets: SolveBudgets,
    },
    Resume {
        identity: SolveIdentity,
        state_token: u128,
        metadata: SolveCheckpointMetadata,
        budgets: SolveBudgets,
    },
}

impl Entry {
    const fn kind(&self) -> AttemptKind {
        match self {
            Self::Call { .. } => AttemptKind::Call,
            Self::Resume { .. } => AttemptKind::Resume,
        }
    }
}

/// Solve state and the record of the most recent attempt.
#[derive(Debug)]
pub(crate) struct Shell<P> {
    pub(crate) state: SolveState,
    attempt: Option<AttemptRecord<P>>,
}

impl<P> Default for Shell<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> Shell<P> {
    pub(crate) const fn new() -> Self {
        Self {
            state: SolveState::new(),
            attempt: None,
        }
    }

    /// Record a rejected attempt before any validation runs.
    pub(crate) fn reject(&mut self, kind: AttemptKind) {
        self.attempt = Some(AttemptRecord::new(
            None,
            kind,
            AttemptOutcome::PreflightRejected,
            SolveCounts::new(),
            None,
        ));
    }

    pub(crate) const fn last_attempt(&self) -> Option<&AttemptRecord<P>> {
        self.attempt.as_ref()
    }

    pub(crate) fn progress(&self) -> Option<&P> {
        self.attempt.as_ref()?.progress()
    }

    /// Run `kernel` in one epoch of `scope`; the flag reports whether it began.
    pub(crate) fn execute<'scope, 'capability, T>(
        &mut self,
        scope: SolveScope<'scope, 'capability>,
        entry: Entry,
        kernel: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
    ) -> (Result<T, SciOxideError>, bool) {
        let mut started = false;
        let run = |context: &mut SolveContext<'_, 'capability>| {
            started = true;
            kernel(context)
        };
        let outcome = match entry {
            Entry::Call { identity, budgets } => scope.run(&mut self.state, identity, budgets, run),
            Entry::Resume {
                identity,
                state_token,
                metadata,
                mut budgets,
            } => scope.resume(
                &mut self.state,
                identity,
                state_token,
                metadata,
                &mut budgets,
                run,
            ),
        };
        (outcome, started)
    }

    /// Record the outcome of an attempt that passed preflight.
    pub(crate) fn finish(
        &mut self,
        kind: AttemptKind,
        started: bool,
        failed: bool,
        succeeded: bool,
        progress: Option<P>,
    ) {
        let outcome = if !started {
            AttemptOutcome::PreflightRejected
        } else if failed {
            AttemptOutcome::Failed
        } else if succeeded {
            AttemptOutcome::Succeeded
        } else {
            AttemptOutcome::Stopped
        };
        let counts = if started {
            self.state.counts()
        } else {
            SolveCounts::new()
        };
        self.attempt = Some(AttemptRecord::new(
            started.then(|| self.state.epoch()).flatten(),
            kind,
            outcome,
            counts,
            progress,
        ));
    }
}

/// Reject a configured time limit without a clock, then any pending stop.
pub(crate) fn preflight(
    scope: &SolveScope<'_, '_>,
    time_limit: Option<Duration>,
) -> Result<(), SciOxideError> {
    if time_limit.is_some() && scope.execution().clock().is_none() {
        return Err(SciOxideError::capability_unavailable(Capability::Clock));
    }
    scope.execution().check()
}

/// Result storage that a kernel stages before the attempt copies it out.
pub(crate) trait Staged {
    /// Clear observations while keeping capacity.
    fn reset(&mut self);
    /// Copy `source` into `self`; failure leaves `self` unchanged.
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError>;
    /// Whether the staged termination is a success.
    fn succeeded(&self) -> bool;
}

impl<S: RealScalar> Staged for OptimizeResult<S> {
    fn reset(&mut self) {
        self.reset_for_fit();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_copy_from(source)
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}

impl<S: RealScalar> Staged for crate::LeastSquaresResult<S> {
    fn reset(&mut self) {
        self.reset_for_fit();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_copy_from(source)
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}

/// Grow `values` to hold `len` elements; failure leaves its contents unchanged.
pub(crate) fn reserve_len<T>(
    values: &mut alloc::vec::Vec<T>,
    len: usize,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    scioxide_core::backend::try_reserve(values, len.saturating_sub(values.len()), quantity)
}

/// Replace reserved `values` by `source` without growth.
pub(crate) fn assign<T: Copy>(values: &mut alloc::vec::Vec<T>, source: &[T]) {
    values.clear();
    values.extend_from_slice(source);
}

/// Run one attempt, copy out its staged result and record the attempt.
///
/// `kernel` receives the epoch context, solver storage and staged output;
/// `progress` reads the storage after the kernel with the success flag.
/// `output` changes only when the epoch and the final copy succeed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<'scope, 'capability, St, R, P>(
    shell: &mut Shell<P>,
    storage: &mut St,
    staged: &mut R,
    output: &mut R,
    scope: SolveScope<'scope, 'capability>,
    entry: Entry,
    kernel: impl FnOnce(
        &mut SolveContext<'_, 'capability>,
        &mut St,
        &mut R,
    ) -> Result<(), SciOxideError>,
    progress: impl FnOnce(&St, bool) -> Option<P>,
) -> Result<(), SciOxideError>
where
    R: Staged,
{
    let kind = entry.kind();
    staged.reset();
    let (outcome, started) =
        shell.execute(scope, entry, |context| kernel(context, storage, staged));
    let outcome = outcome.and_then(|()| output.publish(staged));
    let succeeded = outcome.is_ok() && staged.succeeded();
    let progress = if started {
        progress(storage, succeeded)
    } else {
        None
    };
    shell.finish(kind, started, outcome.is_err(), succeeded, progress);
    outcome
}

/// Trust of an accepted state after an attempt.
pub(crate) const fn trust(succeeded: bool) -> PartialResultTrust {
    if succeeded {
        PartialResultTrust::ResidualChecked
    } else {
        PartialResultTrust::Admissible
    }
}

/// Progress of an accepted local-minimizer state.
pub(crate) fn progress<S>(state: AcceptedState<'_, S>, succeeded: bool) -> MinimizerProgress<S>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
{
    MinimizerProgress {
        value: state.value,
        iterations: state.iterations,
        gradient_norm: state
            .gradient
            .and_then(|gradient| crate::utils::norm2(gradient).ok()),
        trust: trust(succeeded),
    }
}

/// Run one local-minimizer attempt and copy out its staged result.
pub(crate) fn minimize<'scope, 'capability, S, St>(
    shell: &mut Shell<MinimizerProgress<S>>,
    storage: &mut St,
    staged: &mut OptimizeResult<S>,
    output: &mut OptimizeResult<S>,
    scope: SolveScope<'scope, 'capability>,
    entry: Entry,
    kernel: impl FnOnce(
        &mut SolveContext<'_, 'capability>,
        &mut St,
        &mut OptimizeResult<S>,
    ) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    St: Accepted<S>,
{
    run(
        shell,
        storage,
        staged,
        output,
        scope,
        entry,
        kernel,
        |storage, succeeded| storage.accepted().map(|state| progress(state, succeeded)),
    )
}

/// Borrow the accepted iterate recorded by the most recent attempt.
pub(crate) fn last_accepted<'a, S, St>(
    shell: &Shell<MinimizerProgress<S>>,
    storage: &'a St,
) -> Option<MinimizerAccepted<'a, S>>
where
    S: RealScalar,
    St: Accepted<S>,
{
    let progress = *shell.progress()?;
    let state = storage.accepted()?;
    Some(MinimizerAccepted {
        point: state.point,
        gradient: state.gradient,
        progress,
    })
}

/// Declare a prepared local-minimizer workspace over shape-keyed storage.
macro_rules! local_workspace {
    ($(#[$meta:meta])* $workspace:ident, $storage:ty, $requirements:ty) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $workspace<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> {
            pub(crate) storage: $storage,
            pub(crate) staged: crate::OptimizeResult<S>,
            pub(crate) shell: crate::attempt::Shell<crate::MinimizerProgress<S>>,
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> Default
            for $workspace<S>
        {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> $workspace<S> {
            /// Creates an empty workspace without allocating.
            pub fn new() -> Self {
                Self {
                    storage: Default::default(),
                    staged: crate::OptimizeResult::empty(),
                    shell: crate::attempt::Shell::new(),
                }
            }

            /// Returns whether the workspace is large enough for `requirements`.
            ///
            /// # Errors
            ///
            /// Never fails; this workspace has no byte limit.
            pub fn check(
                &self,
                requirements: &$requirements,
            ) -> Result<scioxide_core::Capacity, scioxide_core::SciOxideError> {
                use crate::attempt::Shape;
                Ok(
                    if requirements.fits(&self.storage)
                        && self.staged.fits(requirements.dimension())
                    {
                        scioxide_core::Capacity::Sufficient
                    } else {
                        scioxide_core::Capacity::NeedsGrowth
                    },
                )
            }

            /// Reserves storage for `requirements`.
            ///
            /// A different problem shape replaces the storage. On failure the
            /// existing storage is unchanged.
            ///
            /// # Errors
            ///
            /// Returns `SizeOverflow` or `AllocationFailure`.
            pub fn try_reserve(
                &mut self,
                requirements: &$requirements,
            ) -> Result<(), scioxide_core::SciOxideError> {
                use crate::attempt::Shape;
                if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
                    return Ok(());
                }
                requirements.prepare(&mut self.storage)?;
                self.staged.try_reserve(requirements.dimension())
            }

            /// Returns the record of the most recent call, or `None` before the
            /// first call.
            pub const fn last_attempt(
                &self,
            ) -> Option<&scioxide_core::AttemptRecord<crate::MinimizerProgress<S>>> {
                self.shell.last_attempt()
            }

            /// Returns the last accepted point of the most recent call, or `None`
            /// if that call accepted no point.
            pub fn last_accepted(&self) -> Option<crate::MinimizerAccepted<'_, S>> {
                crate::attempt::last_accepted(&self.shell, &self.storage)
            }
        }
    };
}
pub(crate) use local_workspace;
