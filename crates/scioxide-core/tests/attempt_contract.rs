//! Synthetic lifecycle and progress-storage contracts.

use scioxide_core::{
    AttemptKind, AttemptOutcome, AttemptRecord, PartialFailure, SciOxideError, SciOxideErrorKind,
    SolveCounts, argument,
};

fn primary_error() -> SciOxideError {
    SciOxideError::invalid_input(
        argument!("input"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "original failure",
    )
}

#[test]
fn preflight_record_has_no_progress_or_epoch() {
    let record: AttemptRecord<usize> = AttemptRecord::new(
        None,
        AttemptKind::Resume,
        AttemptOutcome::PreflightRejected,
        SolveCounts::new(),
        None,
    );
    assert_eq!(record.kind(), AttemptKind::Resume);
    assert_eq!(record.outcome(), AttemptOutcome::PreflightRejected);
    assert!(record.epoch().is_none());
    assert!(record.progress().is_none());
}

#[test]
fn conversion_discards_captured_progress() {
    let failure = PartialFailure::with_progress(primary_error(), || Ok(vec![1_u8, 2]));
    assert_eq!(failure.progress().unwrap(), &[1, 2]);
    let error: SciOxideError = failure.into();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
}

#[test]
fn recorded_progress_can_be_recycled() {
    let record = AttemptRecord::new(
        None,
        AttemptKind::Call,
        AttemptOutcome::Failed,
        SolveCounts::new(),
        Some(vec![3.0_f64, 4.0]),
    );
    assert_eq!(record.progress().unwrap(), &[3.0, 4.0]);
    assert_eq!(record.into_progress(), Some(vec![3.0, 4.0]));
    let rejected = AttemptRecord::new(
        None,
        AttemptKind::Call,
        AttemptOutcome::PreflightRejected,
        SolveCounts::new(),
        Some(vec![1.0_f64]),
    );
    assert!(rejected.into_progress().is_none());
}
