//! Transactional batch traversal and bounded cooperative stop polling.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    CancellationToken, CapabilityError, Clock, ClockBinding, Deadline, ExecutionContext,
    ExecutionPolicy, ExecutionStopReason, MonotonicInstant, NdView, NdViewMut, SciOxideError,
    SciOxideErrorView, StridedLayout, log_softmax_axis_into, logsumexp_axis_into,
    softmax_axis_into, transform_into,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[test]
fn mid_transform_cancellation_is_bounded_and_unpublished_for_every_layout() {
    for n in [1_usize, 255, 256, 257, 513] {
        for stride in [0, 1, 2] {
            let shape = [n];
            let strides = [stride];
            let input_layout = StridedLayout::new_broadcast(&shape, &strides).unwrap();
            let output_layout = StridedLayout::new(&shape, &[2]).unwrap();
            let input = vec![2.0_f64; input_layout.required_span()];
            for trigger in [0, 1, n] {
                let token = CancellationToken::new();
                let calls = AtomicUsize::new(0);
                if trigger == 0 {
                    token.cancel();
                }
                let mut output = vec![77.0; n * 2 + 3];
                let mut scratch = vec![0.0; n + 3];
                let error = transform_into(
                    NdView::new(&input, input_layout).unwrap(),
                    |x| {
                        if calls.fetch_add(1, Ordering::Relaxed) + 1 == trigger {
                            token.cancel();
                        }
                        Ok(x * 2.0)
                    },
                    ExecutionPolicy::Sequential,
                    NdViewMut::new(&mut output, output_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::sequential().with_cancellation(&token),
                )
                .unwrap_err();
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::ExecutionStopped {
                        reason: ExecutionStopReason::Cancelled,
                        ..
                    }
                ));
                assert_eq!(
                    calls.load(Ordering::Relaxed),
                    n.min(trigger.div_ceil(256) * 256)
                );
                assert!(output.iter().all(|&x| x == 77.0));
            }
        }
    }
}

struct WorkClock<'a> {
    calls: &'a AtomicUsize,
    reads: AtomicUsize,
    fail_at: usize,
}
impl Clock for WorkClock<'_> {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let work = self.calls.load(Ordering::Relaxed);
        if work >= self.fail_at {
            Err(CapabilityError::new("injected batch clock failure"))
        } else {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(work as u64),
            ))
        }
    }
}

#[test]
fn deadline_and_clock_failure_stop_at_chunk_boundaries_and_final_publication() {
    let n = 513;
    let shape = [n];
    let layout = StridedLayout::new(&shape, &[1]).unwrap();
    for trigger in [1_usize, 255, 256, 257, n, n + 1] {
        for fail in [false, true] {
            let calls = AtomicUsize::new(0);
            let clock = WorkClock {
                calls: &calls,
                reads: AtomicUsize::new(0),
                fail_at: if fail { trigger } else { usize::MAX },
            };
            let binding = ClockBinding::new(&clock);
            let context = ExecutionContext::sequential()
                .with_deadline(Deadline::at(
                    &binding,
                    MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(
                        trigger as u64,
                    )),
                ))
                .unwrap();
            let mut output = vec![77.0; n + 1];
            let mut scratch = vec![0.0; n];
            let input = vec![2.0_f64; n];
            let result = transform_into(
                NdView::new(&input, layout).unwrap(),
                |x| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(x * 2.0)
                },
                ExecutionPolicy::Sequential,
                NdViewMut::new(&mut output, layout).unwrap(),
                &mut scratch,
                context,
            );
            if trigger > n {
                result.unwrap();
                assert!(output[..n].iter().all(|&x| x == 4.0));
                assert_eq!(output[n], 77.0);
                // One admission, one callback boundary, three chunks, one commit check.
                assert_eq!(clock.reads.load(Ordering::Relaxed), 6);
            } else {
                let error = result.unwrap_err();
                if fail {
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::CapabilityFailure { .. }
                    ));
                } else {
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::ExecutionStopped {
                            reason: ExecutionStopReason::DeadlineReached,
                            ..
                        }
                    ));
                }
                assert!(output.iter().all(|&x| x == 77.0));
            }
            assert_eq!(
                calls.load(Ordering::Relaxed),
                n.min(trigger.div_ceil(256) * 256)
            );
        }
    }
}

#[test]
fn contiguous_singletons_broadcast_and_strided_transforms_keep_logical_order() {
    let shape = [2, 1, 3];
    let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    for strides in [[3, usize::MAX, 1], [1, 0, 2], [0, 0, 1]] {
        let layout = StridedLayout::new_broadcast(&shape, &strides).unwrap();
        let output_layout = StridedLayout::new(&shape, &[3, 0, 1]).unwrap();
        let mut output = [77.0; 8];
        let mut scratch = [0.0; 8];
        let view = NdView::new(&input, layout).unwrap();
        transform_into(
            view,
            |x| Ok(x * 2.0),
            ExecutionPolicy::Auto,
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let view = NdView::new(&input, layout).unwrap();
        assert!(
            output[..6]
                .iter()
                .zip(view.iter())
                .all(|(&actual, &x)| actual == x * 2.0)
        );
        assert_eq!(&output[6..], &[77.0; 2]);
    }
    for bad in [false, true] {
        let layout = StridedLayout::new(&[6], &[1]).unwrap();
        let mut output = [77.0; 6];
        let mut scratch = [0.0; 6];
        let error = transform_into(
            NdView::new(&input, layout).unwrap(),
            |x| {
                if x == 6.0 {
                    if bad {
                        Ok(f64::NAN)
                    } else {
                        Err(SciOxideError::singular(None))
                    }
                } else {
                    Ok(x)
                }
            },
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut output, layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            if bad {
                scioxide_core::SciOxideErrorKind::NonFinite
            } else {
                scioxide_core::SciOxideErrorKind::Singular
            }
        );
        assert_eq!(output, [77.0; 6]);
    }
    let empty = StridedLayout::new(&[0, 3], &[3, 1]).unwrap();
    transform_into(
        NdView::new(&[], empty).unwrap(),
        |_: f64| unreachable!(),
        ExecutionPolicy::Auto,
        NdViewMut::new(&mut [], empty).unwrap(),
        &mut [],
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[cfg(feature = "std")]
#[test]
fn concurrent_cancellation_has_a_per_partition_bound() {
    use std::num::NonZeroUsize;
    let workers = NonZeroUsize::new(4).unwrap();
    let executor = scioxide_core::ThreadExecutor::new(workers);
    let token = CancellationToken::new();
    let barrier = std::sync::Barrier::new(4);
    let calls: [AtomicUsize; 4] = core::array::from_fn(|_| AtomicUsize::new(0));
    let input: Vec<_> = (0..2048).map(|i| i as f64).collect();
    let layout = StridedLayout::new(&[2048], &[1]).unwrap();
    let mut output = [77.0; 2048];
    let mut scratch = [0.0; 2048];
    let error = transform_into(
        NdView::new(&input, layout).unwrap(),
        |x| {
            let partition = x as usize / 512;
            if calls[partition].fetch_add(1, Ordering::Relaxed) == 0 {
                barrier.wait();
                if partition == 0 {
                    token.cancel();
                }
                barrier.wait();
            }
            Ok(x)
        },
        ExecutionPolicy::Parallel(workers),
        NdViewMut::new(&mut output, layout).unwrap(),
        &mut scratch,
        ExecutionContext::new(&executor).with_cancellation(&token),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::Cancelled,
            ..
        }
    ));
    assert!(calls.iter().all(|n| n.load(Ordering::Relaxed) == 256));
    assert_eq!(output, [77.0; 2048]);
}

#[test]
fn prepared_transform_reuse_is_allocation_free() {
    let input = [1.0_f32; 1024];
    for strides in [[32, 1], [1, 32], [0, 1]] {
        let input_layout = StridedLayout::new_broadcast(&[32, 32], &strides).unwrap();
        let output_layout = StridedLayout::new(&[32, 32], &[32, 1]).unwrap();
        let mut output = [0.0; 1024];
        let mut scratch = [0.0; 1024];
        for _ in 0..2 {
            transform_into(
                NdView::new(&input, input_layout).unwrap(),
                scioxide_core::sigmoid,
                ExecutionPolicy::Auto,
                NdViewMut::new(&mut output, output_layout).unwrap(),
                &mut scratch,
                ExecutionContext::sequential(),
            )
            .unwrap();
        }
        assert!(
            output
                .iter()
                .all(|&x| x == scioxide_core::sigmoid(1.0_f32).unwrap())
        );
    }
}

#[test]
fn axis_stops_between_lanes_leave_all_output_unpublished() {
    struct CancelAfterLane(AtomicUsize);
    impl scioxide_core::Cancellation for CancelAfterLane {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 3
        }
    }
    let layout = StridedLayout::new(&[3, 2], &[2, 1]).unwrap();
    let reduced = StridedLayout::new(&[3], &[1]).unwrap();
    let input = [0.0_f64; 6];
    for kind in 0..3 {
        let cancellation = CancelAfterLane(AtomicUsize::new(0));
        let context = ExecutionContext::sequential().with_cancellation(&cancellation);
        let mut scratch = [77.0; 6];
        let mut output = [77.0; 6];
        let input = NdView::new(&input, layout).unwrap();
        let policy = ExecutionPolicy::Sequential;
        let error = match kind {
            0 => softmax_axis_into(
                input,
                1,
                policy,
                NdViewMut::new(&mut output, layout).unwrap(),
                &mut scratch,
                context,
            ),
            1 => log_softmax_axis_into(
                input,
                1,
                policy,
                NdViewMut::new(&mut output, layout).unwrap(),
                &mut scratch,
                context,
            ),
            _ => logsumexp_axis_into(
                input,
                1,
                policy,
                NdViewMut::new(&mut output, reduced).unwrap(),
                &mut scratch,
                context,
            ),
        }
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::Cancelled,
                ..
            }
        ));
        let width = if kind < 2 { 2 } else { 1 };
        assert!(scratch[..width].iter().all(|&x| x != 77.0));
        assert!(scratch[width..].iter().all(|&x| x == 77.0));
        assert_eq!(output, [77.0; 6]);
    }
}
