#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{ExecutionContext, RealScalar, StridedLayout};
use scioxide_stats::*;

fn run<S: RealScalar, M: IsotonicWeights>(
    view: BatchView<'_, S, M>,
    options: (Axis, IsotonicDirection, StatisticsOptions),
    context: ExecutionContext<'_>,
) -> (Vec<S>, Vec<IsotonicBlock<S>>, Vec<IsotonicLaneResult<S>>) {
    let r = IsotonicWorkspace::workspace_requirements(
        view.with_axis((options).0).unwrap(),
        (options).2,
        context,
    )
    .unwrap();
    let mut w = IsotonicWorkspace::default();
    w.try_reserve(&r).unwrap();
    let mut fits = vec![S::ZERO; r.slots()];
    let mut blocks = vec![IsotonicBlock::default(); r.slots()];
    let mut lanes = vec![IsotonicLaneResult::default(); r.batches()];
    isotonic_regression_axes_into(
        view.with_axis((options).0).unwrap(),
        (options).1,
        (options).2,
        IsotonicBatchOutput {
            fitted: &mut fits,
            blocks: &mut blocks,
            lanes: &mut lanes,
        },
        &mut w,
        context,
    )
    .unwrap();
    (fits, blocks, lanes)
}
fn layouts<S: RealScalar>() {
    let values = [3.0, 9.0, 1.0, 7.0, 5.0, 8.0].map(|x| S::try_from_f64_exact(x).unwrap());
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2, 3], &[1, 2]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let seq = ExecutionContext::sequential();
    let (fit, blocks, lanes) = run(
        view,
        (
            Axis::Index(1),
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
        ),
        seq,
    );
    let expected = [2.0, 2.0, 5.0, 8.0, 8.0, 8.0].map(|x| S::try_from_f64_exact(x).unwrap());
    assert_eq!(fit, expected);
    assert_eq!(lanes[0].diagnostics.block_count, 2);
    assert_eq!(lanes[1].block_offset, 3);
    assert_eq!(blocks[3].end, 3);
    let weights = [1.0, 3.0, 1.0, 1.0, 1.0, 1.0].map(|x| S::try_from_f64_exact(x).unwrap());
    let weighted = view
        .with_weights(Weights::<_, Mass>::new(&weights))
        .unwrap();
    let (fit, _, _) = run(
        weighted,
        (
            Axis::Index(1),
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
        ),
        seq,
    );
    assert_eq!(fit[0], S::try_from_f64_exact(1.5).unwrap());
    let flat = run(
        view,
        (
            Axis::All,
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
        ),
        seq,
    )
    .0;
    assert_eq!(
        flat,
        isotonic_regression_with_options(
            &[
                values[0], values[2], values[4], values[1], values[3], values[5]
            ],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new()
        )
        .unwrap()
        .fitted()
    );
    let broadcast = BatchView::new(
        &values[..3],
        StridedLayout::new_broadcast(&[4, 3], &[0, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let fits = run(
        broadcast,
        (
            Axis::Index(1),
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
        ),
        seq,
    )
    .0;
    let expected = isotonic_regression_with_options(
        &values[..3],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!(fits.chunks_exact(3).all(|row| row == expected.fitted()));
    let empty = BatchView::<S>::new(
        &[],
        StridedLayout::new(&[2, 0], &[0, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let (fit, blocks, lanes) = run(
        empty,
        (
            Axis::Index(1),
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
        ),
        seq,
    );
    assert!(fit.is_empty() && blocks.is_empty());
    assert_eq!(lanes.len(), 2);
    assert!(
        lanes
            .iter()
            .all(|l| l.diagnostics == IsotonicDiagnostics::default())
    );
    assert!(
        run(
            empty,
            (
                Axis::Index(0),
                IsotonicDirection::Increasing,
                StatisticsOptions::new()
            ),
            seq
        )
        .2
        .is_empty()
    );
    let scalar = BatchView::new(
        &values[..1],
        StridedLayout::new(&[], &[]).unwrap(),
        Axis::All,
    )
    .unwrap();
    assert_eq!(
        run(
            scalar,
            (
                Axis::All,
                IsotonicDirection::Increasing,
                StatisticsOptions::new()
            ),
            seq
        )
        .0,
        values[..1]
    );
    assert!(scalar.with_axis(Axis::Index(0)).is_err());
}
#[test]
fn layouts_f32() {
    layouts::<f32>();
}
#[test]
fn layouts_f64() {
    layouts::<f64>();
}

#[test]
fn admission_atomicity_and_reuse() {
    let layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
    let y = [3.0, 1.0, 4.0, 2.0, f64::NAN, 0.0];
    let view = BatchView::new(&y, layout, Axis::All).unwrap();
    let seq = ExecutionContext::sequential();
    let mut options = (
        Axis::Index(1),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    );
    let r = IsotonicWorkspace::workspace_requirements(
        view.with_axis((options).0).unwrap(),
        (options).2,
        seq,
    )
    .unwrap();
    let mut w = IsotonicWorkspace::default();
    let mut fits = [77.0; 6];
    let mut blocks = [IsotonicBlock::default(); 6];
    let mut lanes = [IsotonicLaneResult::default(); 2];
    let sentinel = blocks;
    for reserved in [false, true] {
        if reserved {
            w.try_reserve(&r).unwrap();
        }
        assert!(
            isotonic_regression_axes_into(
                view.with_axis((options).0).unwrap(),
                (options).1,
                (options).2,
                IsotonicBatchOutput {
                    fitted: &mut fits,
                    blocks: &mut blocks,
                    lanes: &mut lanes
                },
                &mut w,
                seq
            )
            .is_err()
        );
        assert_eq!(fits, [77.0; 6]);
        assert_eq!(blocks, sentinel);
        assert_eq!(lanes, [IsotonicLaneResult::default(); 2]);
    }
    options.2 = options.2.with_non_finite_policy(NonFinitePolicy::Skip);
    isotonic_regression_axes_into(
        view.with_axis((options).0).unwrap(),
        (options).1,
        (options).2,
        IsotonicBatchOutput {
            fitted: &mut fits,
            blocks: &mut blocks,
            lanes: &mut lanes,
        },
        &mut w,
        seq,
    )
    .unwrap();
    assert_eq!(fits, [2.0, 2.0, 4.0, 1.0, 1.0, 1.0]);
    assert_eq!(lanes[1].diagnostics.excluded, 1);
    let weights = [1.0, 1.0, 1.0, 1.0, f64::NAN, 1.0];
    let masked = view
        .with_validity(&[true, true, true, true, false, true])
        .unwrap()
        .with_weights(Weights::<_, Mass>::new(&weights))
        .unwrap();
    options.2 = options.2.with_missing_policy(MissingPolicy::Skip);
    assert_eq!(run(masked, options, seq).0, fits);
    options.2 = options.2.with_missing_policy(MissingPolicy::Reject);
    assert!(
        isotonic_regression_axes_into(
            masked.with_axis((options).0).unwrap(),
            (options).1,
            (options).2,
            IsotonicBatchOutput {
                fitted: &mut fits,
                blocks: &mut blocks,
                lanes: &mut lanes
            },
            &mut w,
            seq
        )
        .is_err()
    );
    let empty_mask = view.with_validity(&[false; 6]).unwrap();
    options.2 = options.2.with_missing_policy(MissingPolicy::Skip);
    assert!(
        isotonic_regression_axes_into(
            empty_mask.with_axis((options).0).unwrap(),
            (options).1,
            (options).2,
            IsotonicBatchOutput {
                fitted: &mut fits,
                blocks: &mut blocks,
                lanes: &mut lanes
            },
            &mut w,
            seq
        )
        .is_err()
    );

    assert_eq!(
        IsotonicWorkspace::workspace_requirements(
            view.with_axis(options.0).unwrap(),
            options.2,
            seq
        )
        .unwrap()
        .workers(),
        1
    );
    let empty = BatchView::<f64>::new(
        &[],
        StridedLayout::new(&[0, 1], &[1, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let capped = IsotonicWorkspace::workspace_requirements(
        empty,
        StatisticsOptions::new().with_max_bytes(0),
        seq,
    )
    .unwrap();
    assert!(w.check(&capped).is_err());
}

#[cfg(feature = "std")]
#[test]
fn executor_axes_bitwise_match_slice_fits() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let y: Vec<_> = (0..256).map(|i| ((i * 31) % 17) as f64).collect();
    let view = BatchView::new(
        &y,
        StridedLayout::new(&[4, 8, 8], &[64, 8, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    for axis in 0..3 {
        for direction in [IsotonicDirection::Increasing, IsotonicDirection::Decreasing] {
            let mut options = (
                Axis::Index(axis),
                IsotonicDirection::Increasing,
                StatisticsOptions::new(),
            );
            options.1 = direction;
            let expected = run(view, options, ExecutionContext::sequential());
            let layout = view.layout().axis(axis).unwrap();
            let len = view.layout().shape().extents()[axis];
            let trailing: usize = view.layout().shape().extents()[axis + 1..].iter().product();
            for batch in 0..layout.len() {
                let lane: Vec<_> = layout
                    .lane(batch)
                    .unwrap()
                    .offsets()
                    .map(|i| y[i])
                    .collect();
                let fit = isotonic_regression_with_options(
                    &lane,
                    Unweighted,
                    (direction, StatisticsOptions::new()).0,
                    (direction, StatisticsOptions::new()).1,
                )
                .unwrap();
                for (i, x) in fit.fitted().iter().enumerate() {
                    assert_eq!(
                        *x,
                        expected.0
                            [(batch / trailing) * len * trailing + batch % trailing + i * trailing]
                    );
                }
            }

            options.2 = options
                .2
                .with_execution_policy(ExecutionPolicy::Parallel(workers));
            for _ in 0..2 {
                assert_eq!(run(view, options, ExecutionContext::new(&pool)), expected);
            }
        }
    }
}

#[test]
fn executor_failure_cancellation_and_recovery() {
    use core::num::NonZeroUsize;
    use scioxide_core::{CancellationToken, ExecutionPlan, Executor, PartitionFn, SciOxideError};
    struct Incomplete;
    impl Executor for Incomplete {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::MIN
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    let view = BatchView::new(
        &[2.0, 1.0],
        StridedLayout::new(&[2], &[1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options = (
        Axis::All,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    );
    let seq = ExecutionContext::sequential();
    let mut w = IsotonicWorkspace::default();
    w.try_reserve(
        &(IsotonicWorkspace::workspace_requirements(
            view.with_axis((options).0).unwrap(),
            (options).2,
            seq,
        )
        .unwrap()),
    )
    .unwrap();
    let mut fits = [77.0; 2];
    let mut blocks = [IsotonicBlock::default(); 2];
    let mut lanes = [IsotonicLaneResult::default(); 1];
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    for context in [
        ExecutionContext::new(&Incomplete),
        seq.with_cancellation(&cancellation),
    ] {
        assert!(
            isotonic_regression_axes_into(
                view.with_axis((options).0).unwrap(),
                (options).1,
                (options).2,
                IsotonicBatchOutput {
                    fitted: &mut fits,
                    blocks: &mut blocks,
                    lanes: &mut lanes
                },
                &mut w,
                context
            )
            .is_err()
        );
        assert_eq!(fits, [77.0; 2]);
        assert_eq!(blocks, [IsotonicBlock::default(); 2]);
        assert_eq!(lanes, [IsotonicLaneResult::default(); 1]);
    }
    isotonic_regression_axes_into(
        view.with_axis((options).0).unwrap(),
        (options).1,
        (options).2,
        IsotonicBatchOutput {
            fitted: &mut fits,
            blocks: &mut blocks,
            lanes: &mut lanes,
        },
        &mut w,
        seq,
    )
    .unwrap();
    assert_eq!(fits, [1.5; 2]);
}
