//! Clock identity, tightening and transactional configuration regressions.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Capability, CapabilityError, Clock, ClockBinding, Deadline, ExecutionContext,
    ExecutionStopReason, MonotonicInstant, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
struct FakeClock {
    ticks: AtomicU64,
    fail: AtomicBool,
}
impl Clock for FakeClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        if self.fail.load(Ordering::Relaxed) {
            Err(CapabilityError::new("clock failed"))
        } else {
            Ok(instant(self.ticks.load(Ordering::Relaxed)))
        }
    }
}
fn instant(ticks: u64) -> MonotonicInstant {
    MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(ticks))
}

#[test]
fn builder_orders_preserve_the_earliest_deadline() {
    let clock = FakeClock::default();
    let binding = ClockBinding::new(&clock);
    let early = Deadline::at(&binding, instant(5));
    let late = Deadline::at(&binding, instant(10));
    let initial = ExecutionContext::sequential();
    for context in [
        initial
            .with_clock(&binding)
            .unwrap()
            .with_deadline(early)
            .unwrap(),
        initial
            .with_deadline(early)
            .unwrap()
            .with_clock(&binding)
            .unwrap(),
        initial
            .with_deadline(late)
            .unwrap()
            .with_deadline(early)
            .unwrap(),
        initial
            .with_deadline(early)
            .unwrap()
            .with_deadline(late)
            .unwrap(),
        initial
            .with_deadline(early)
            .unwrap()
            .with_time_limit(Duration::from_nanos(20))
            .unwrap(),
        initial
            .with_deadline(late)
            .unwrap()
            .with_time_limit(Duration::from_nanos(5))
            .unwrap(),
    ] {
        clock.ticks.store(4, Ordering::Relaxed);
        context.check().unwrap();
        clock.ticks.store(5, Ordering::Relaxed);
        assert_eq!(
            context.stop_reason().unwrap(),
            Some(ExecutionStopReason::DeadlineReached)
        );
    }
}

#[test]
fn different_bindings_are_rejected_even_for_the_same_provider() {
    let clock = FakeClock::default();
    let a = ClockBinding::new(&clock);
    let b = ClockBinding::new(&clock);
    for context in [
        ExecutionContext::sequential().with_clock(&a).unwrap(),
        ExecutionContext::sequential()
            .with_deadline(Deadline::at(&a, instant(1)))
            .unwrap(),
    ] {
        for result in [
            context.with_clock(&b),
            context.with_deadline(Deadline::at(&b, instant(2))),
        ] {
            assert!(matches!(
                (result).as_ref().unwrap_err().view(),
                scioxide_core::SciOxideErrorView::CapabilityFailure {
                    capability: Capability::Clock,
                    ..
                }
            ));
        }
        assert!(core::ptr::eq(context.clock().unwrap(), &a));
    }
}

#[test]
fn missing_clock_limits_reject_without_mutating_a_solve() {
    for duration in [Duration::ZERO, Duration::from_secs(1)] {
        assert!(matches!(
            (ExecutionContext::sequential().with_time_limit(duration))
                .as_ref()
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::CapabilityUnavailable {
                capability: Capability::Clock,
                ..
            }
        ));
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(1),
                    scioxide_core::algorithm_id!("test", "timing", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let before = context.counts();
        assert!(matches!(
            (context.limit_time(duration)).as_ref().unwrap_err().view(),
            scioxide_core::SciOxideErrorView::CapabilityUnavailable { .. }
        ));
        assert_eq!(context.counts(), before);
        assert_eq!(context.stop_reason().unwrap(), None);
    }
}

#[test]
fn zero_overflow_and_provider_failures_are_explicit() {
    let clock = FakeClock::default();
    let binding = ClockBinding::new(&clock);
    let context = ExecutionContext::sequential().with_clock(&binding).unwrap();
    assert_eq!(
        context
            .with_time_limit(Duration::ZERO)
            .unwrap()
            .stop_reason()
            .unwrap(),
        Some(ExecutionStopReason::DeadlineReached)
    );
    clock.ticks.store(1, Ordering::Relaxed);
    assert!(matches!(
        (context.with_time_limit(Duration::MAX))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::CapabilityFailure { .. }
    ));
    clock.fail.store(true, Ordering::Relaxed);
    assert!(matches!(
        (context.with_time_limit(Duration::ZERO))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::CapabilityFailure { .. }
    ));
    // A bound clock without a deadline need not be read during ordinary work.
    context.check().unwrap();
}

#[test]
fn capability_detail_is_moved_without_cloning_its_allocation() {
    let owned = String::from("owned provider failure");
    let pointer = owned.as_ptr();
    let detail = CapabilityError::new(owned).into_detail();
    assert_eq!(detail.as_ptr(), pointer);
}
