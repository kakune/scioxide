//! Explicit timing, resource limits and caller-owned scoped work.
use scioxide_core::{
    ClockBinding, ExecutionContext, ExecutionPolicy, PlanLimits, SciOxideError, StdClock,
    ThreadExecutor,
};
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
const WORKERS: NonZeroUsize = NonZeroUsize::new(4).unwrap();
fn main() -> Result<(), SciOxideError> {
    let pool = ThreadExecutor::new(WORKERS);
    let clock = StdClock::new();
    let binding = ClockBinding::new(&clock);
    let context = ExecutionContext::new(&pool)
        .with_clock(&binding)?
        .with_time_limit(Duration::from_secs(10))?;
    let plan = context.plan(
        ExecutionPolicy::Parallel(WORKERS),
        128,
        PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: WORKERS,
        },
    );
    let visited = AtomicUsize::new(0);
    context.execute(plan, &|partition, (), scope| {
        assert_eq!(scope.worker_budget(), NonZeroUsize::MIN);
        visited.fetch_add(partition.range().len(), Ordering::Relaxed);
        Ok(())
    })?;
    assert_eq!(visited.load(Ordering::Relaxed), 128);
    Ok(())
}
