//! Deterministic cancellation and deadline capabilities for execution tests.
#![allow(
    dead_code,
    reason = "test targets use different execution capabilities"
)]

use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{Cancellation, CapabilityError, Clock, MonotonicInstant};

/// Permit `polls` cancellation checks, then report cancellation on every check.
pub struct CancelAfter {
    polls: AtomicUsize,
    limit: usize,
}

impl CancelAfter {
    pub const fn new(polls: usize) -> Self {
        Self {
            polls: AtomicUsize::new(0),
            limit: polls,
        }
    }
}

impl Cancellation for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.polls.fetch_add(1, Ordering::Relaxed) >= self.limit
    }
}

/// Always return two seconds after the monotonic epoch.
pub struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(2),
        ))
    }
}
