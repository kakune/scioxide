//! Deterministic clocks for time-limit contracts.
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use scioxide_core::{CapabilityError, Clock, MonotonicInstant};

/// A clock that always reads the epoch, so a positive time limit never expires.
pub struct Frozen;

impl Clock for Frozen {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO))
    }
}

/// A clock reading the stored number of whole seconds since the epoch.
pub struct TestClock(pub AtomicU64);

impl Clock for TestClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(
            Duration::from_secs(self.0.load(Ordering::Relaxed)),
        ))
    }
}
