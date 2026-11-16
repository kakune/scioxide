//! Records of solver attempts, and errors that carry partial progress.

use crate::{SciOxideError, SolveCounts, SolveEpoch};

/// Whether an attempt started a new solve or resumed a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttemptKind {
    /// A fresh call.
    Call,
    /// Continuation from a checkpoint.
    Resume,
}

/// Outcome of one solver attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttemptOutcome {
    /// Validation rejected the attempt before numerical work began.
    PreflightRejected,
    /// Execution returned an error.
    Failed,
    /// Execution stopped without success but with an accepted state.
    Stopped,
    /// Execution satisfied its success criterion.
    Succeeded,
}

/// Diagnostics for one solver attempt, stored in a workspace.
///
/// `progress` belongs to this attempt only. An attempt rejected during
/// validation has no epoch and no progress, even if the workspace still holds
/// buffers from an earlier attempt.
///
/// ```
/// use scioxide_core::{AttemptKind, AttemptOutcome, AttemptRecord, SolveCounts};
///
/// let record: AttemptRecord<usize> = AttemptRecord::new(
///     None,
///     AttemptKind::Call,
///     AttemptOutcome::PreflightRejected,
///     SolveCounts::new(),
///     None,
/// );
/// assert!(record.progress().is_none());
/// ```
#[derive(Debug, Clone)]
pub struct AttemptRecord<P> {
    epoch: Option<SolveEpoch>,
    kind: AttemptKind,
    outcome: AttemptOutcome,
    counts: SolveCounts,
    progress: Option<P>,
}

impl<P> AttemptRecord<P> {
    /// Creates a record of a finished or rejected attempt.
    #[inline(always)]
    pub fn new(
        epoch: Option<SolveEpoch>,
        kind: AttemptKind,
        outcome: AttemptOutcome,
        counts: SolveCounts,
        progress: Option<P>,
    ) -> Self {
        if matches!(outcome, AttemptOutcome::PreflightRejected) {
            return Self {
                epoch: None,
                kind,
                outcome,
                counts: SolveCounts::new(),
                progress: None,
            };
        }
        Self {
            epoch,
            kind,
            outcome,
            counts,
            progress,
        }
    }

    /// Returns the epoch, or `None` if the attempt was rejected during validation.
    pub const fn epoch(&self) -> Option<SolveEpoch> {
        self.epoch
    }

    /// Returns whether the attempt was a new call or a resume.
    pub const fn kind(&self) -> AttemptKind {
        self.kind
    }

    /// Returns the outcome.
    pub const fn outcome(&self) -> AttemptOutcome {
        self.outcome
    }

    /// Returns the work counted by the attempt and its nested solves.
    ///
    /// For a resume, this includes the work restored from the checkpoint.
    pub const fn counts(&self) -> SolveCounts {
        self.counts
    }

    /// Returns the progress of this attempt, if any.
    pub const fn progress(&self) -> Option<&P> {
        self.progress.as_ref()
    }

    /// Returns the progress by value, so its storage can be reused.
    ///
    /// ```
    /// use scioxide_core::{AttemptKind, AttemptOutcome, AttemptRecord, SolveCounts};
    ///
    /// let record = AttemptRecord::new(
    ///     None,
    ///     AttemptKind::Call,
    ///     AttemptOutcome::Failed,
    ///     SolveCounts::new(),
    ///     Some(vec![1.0_f64]),
    /// );
    /// assert_eq!(record.into_progress(), Some(vec![1.0]));
    /// ```
    pub fn into_progress(self) -> Option<P> {
        self.progress
    }
}

/// An error together with the progress made before it, if that could be saved.
///
/// If saving the progress fails, the original error is kept and the progress
/// is `None`.
///
/// ```
/// use scioxide_core::{SciOxideError, PartialFailure, quantity};
///
/// let error = SciOxideError::size_overflow(quantity!("size"));
/// let failure = PartialFailure::with_progress(error, || Ok::<_, SciOxideError>(vec![1_u8]));
/// assert_eq!(failure.progress().unwrap(), &[1]);
/// ```
#[derive(Debug)]
pub struct PartialFailure<P> {
    error: SciOxideError,
    progress: Option<P>,
}

impl<P> PartialFailure<P> {
    /// Creates a failure without progress.
    pub fn new(error: SciOxideError) -> Self {
        Self {
            error,
            progress: None,
        }
    }

    /// Creates a failure and saves the progress returned by `capture`.
    ///
    /// If `capture` fails, its error is discarded and the progress is `None`.
    pub fn with_progress(
        error: SciOxideError,
        capture: impl FnOnce() -> Result<P, SciOxideError>,
    ) -> Self {
        Self {
            error,
            progress: capture().ok(),
        }
    }

    /// Returns the error.
    pub const fn error(&self) -> &SciOxideError {
        &self.error
    }

    /// Returns the saved progress, if any.
    pub const fn progress(&self) -> Option<&P> {
        self.progress.as_ref()
    }

    /// Returns the error and the saved progress.
    pub fn into_parts(self) -> (SciOxideError, Option<P>) {
        (self.error, self.progress)
    }
}

impl<P> From<PartialFailure<P>> for SciOxideError {
    fn from(failure: PartialFailure<P>) -> Self {
        failure.error
    }
}
