//! Scoped RBF preparation: provider failures, stops, limits and atomic publication.
#![allow(clippy::unwrap_used)]
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use scioxide_core::{
    CancellationToken, CapabilityError, Clock, Deadline, ExecutionContext, ExecutionPlan, Executor,
    MonotonicInstant, PartitionFn, RankThreshold, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, SequentialExecutor,
};
use scioxide_interp::{ExtrapolationPolicy, Interpolant, RbfInterpolator, RbfKernel, RbfOptions};

const KERNEL: RbfKernel<f64> = RbfKernel::Gaussian(1.0);

fn options() -> RbfOptions<f64> {
    RbfOptions::new().with_rank_threshold(RankThreshold::new(0.0, 1e-12).unwrap())
}
const POINTS: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
const VALUES: [f64; 3] = [1.0, 2.0, 3.0];

fn build<'scope, 'capability>(
    options: RbfOptions<f64>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<RbfInterpolator<f64>, SciOxideError> {
    RbfInterpolator::build(POINTS.to_vec(), VALUES.to_vec(), KERNEL, options, scope)
}

struct RecordingExecutor(AtomicUsize);
impl Executor for RecordingExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        SequentialExecutor.execute(context, plan, operation)
    }
}

#[test]
fn scoped_preparation_matches_owned_preparation() {
    let owned =
        RbfInterpolator::with_options(POINTS.to_vec(), VALUES.to_vec(), KERNEL, options()).unwrap();
    let scoped = build(options(), ExecutionContext::sequential_native()).unwrap();
    let executor = RecordingExecutor(AtomicUsize::new(0));
    let injected = build(options(), ExecutionContext::new(&executor)).unwrap();
    assert_eq!(owned.conditioning(), scoped.conditioning());
    for (&point, expected) in POINTS.iter().zip(VALUES) {
        let actual = owned.evaluate(point, ExtrapolationPolicy::Reject).unwrap();
        assert!((actual - expected).abs() < 3e-12);
        assert_eq!(
            actual,
            scoped.evaluate(point, ExtrapolationPolicy::Reject).unwrap()
        );
        // The injected context uses portable kernels, equal within rounding.
        let portable = injected
            .evaluate(point, ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((portable - actual).abs() < 1e-12);
    }
    for query in [[0.25, 0.25], [0.5, 0.1]] {
        assert_eq!(
            owned.evaluate(query, ExtrapolationPolicy::Reject).unwrap(),
            scoped.evaluate(query, ExtrapolationPolicy::Reject).unwrap()
        );
    }
}

#[test]
fn cancellation_publishes_nothing() {
    let token = CancellationToken::new();
    token.cancel();
    let result = build(
        options(),
        ExecutionContext::sequential().with_cancellation(&token),
    );
    assert!(result.is_err_and(|error| error.kind() == SciOxideErrorKind::ExecutionStopped));
    // The next preparation is unaffected.
    assert!(build(options(), ExecutionContext::sequential()).is_ok());
}

struct FailingClock {
    reads: AtomicUsize,
    fail_after: usize,
}
impl Clock for FailingClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        if self.reads.fetch_add(1, Ordering::Relaxed) >= self.fail_after {
            Err(CapabilityError::new("interpolation clock sentinel"))
        } else {
            Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO))
        }
    }
}
#[test]
fn clock_failure_after_admission_is_propagated_and_next_preparation_succeeds() {
    let clock = FailingClock {
        reads: AtomicUsize::new(0),
        fail_after: 5,
    };
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let execution = ExecutionContext::sequential()
        .with_deadline(Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
        ))
        .unwrap();
    let result = build(options(), execution);
    assert!(
        matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::CapabilityFailure { .. }))
    );
    assert!(clock.reads.load(Ordering::Relaxed) > 5);
    assert!(build(options(), ExecutionContext::sequential()).is_ok());
}

#[test]
fn byte_limits_are_resource_limits_before_any_work() {
    let token = CancellationToken::new();
    token.cancel();
    // The limit is checked before the epoch begins, so it wins over cancellation.
    let result = build(
        options().with_maximum_bytes(64),
        ExecutionContext::sequential().with_cancellation(&token),
    );
    assert!(
        matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::ResourceLimit { .. }))
    );
    assert!(
        build(
            options().with_maximum_bytes(1 << 20),
            ExecutionContext::sequential()
        )
        .is_ok()
    );
}

struct TestExecutor {
    calls: AtomicUsize,
    reject: bool,
}

impl Executor for TestExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(2).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.reject {
            Err(SciOxideError::execution_failure(
                None,
                scioxide_core::ExecutionFailureReason::ProviderFailure,
                "injected rejection",
            ))
        } else {
            SequentialExecutor.execute(context, plan, operation)
        }
    }
}
#[test]
fn executor_is_used_and_its_failure_prevents_publication() {
    // Exact diagonal Gaussian matrix; enough rows to cross a shared GEMM panel.
    let points: Vec<_> = (0..133).map(|i| [40.0 * f64::from(i), 0.0]).collect();
    let values = vec![1.0; points.len()];
    let policy = options().with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(
        NonZeroUsize::new(2).unwrap(),
    ));
    for reject in [true, false] {
        let executor = TestExecutor {
            calls: AtomicUsize::new(0),
            reject,
        };
        let result = RbfInterpolator::build(
            points.clone(),
            values.clone(),
            KERNEL,
            policy,
            ExecutionContext::new(&executor),
        );
        assert!(executor.calls.load(Ordering::Relaxed) > 0);
        if reject {
            assert!(
                matches!(result, Err(error) if matches!(error.view(), SciOxideErrorView::ExecutionFailure { .. }))
            );
        } else {
            let interpolator = result.unwrap();
            for &point in points.iter().step_by(33) {
                let value = interpolator
                    .evaluate(point, ExtrapolationPolicy::Reject)
                    .unwrap();
                assert!((value - 1.0).abs() < 1e-12);
            }
        }
    }
}
