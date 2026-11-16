//! Public root-path coverage for direct finite-difference lifecycle APIs.

#![allow(clippy::suboptimal_flops)]

use scioxide_core::{
    CancellationToken, Capability, CapabilityError, Clock, EntropySource, ExecutionContext,
    ExecutionPolicy, MonotonicInstant, SciOxideError,
};
use std::num::NonZeroUsize;
use std::time::Duration;

struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO))
    }
}

struct FixedEntropy;

impl EntropySource for FixedEntropy {
    fn fill_bytes(&self, destination: &mut [u8]) -> Result<(), CapabilityError> {
        destination.fill(0x5a);
        Ok(())
    }
}

#[test]
fn no_std_capability_contracts_are_injectable_from_the_root() -> Result<(), SciOxideError> {
    let cancellation = CancellationToken::new();
    let clock = FixedClock;
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let entropy = FixedEntropy;
    let context = ExecutionContext::sequential()
        .with_cancellation(&cancellation)
        .with_clock(&clock_binding)?
        .with_entropy(&entropy)
        .with_time_limit(Duration::from_secs(1))?;
    let plan = context.plan(
        ExecutionPolicy::Auto,
        4,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::MIN,
        },
    );
    context.execute(plan, &|_, (), _| Ok(()))?;

    let mut seed = [0_u8; 8];
    context
        .entropy()
        .ok_or_else(|| SciOxideError::capability_unavailable(Capability::Entropy))?
        .fill_bytes(&mut seed)
        .map_err(|error| {
            SciOxideError::capability_failure(Capability::Entropy, error.into_detail())
        })?;
    assert_eq!(seed, [0x5a; 8]);
    Ok(())
}
