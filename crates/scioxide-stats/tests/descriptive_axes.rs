#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_stats::*;
fn scalar(v: StatisticValue<f64>) -> f64 {
    v.scalar().unwrap()
}

#[test]
fn standardization_strides_and_late_failures_publish_atomically() {
    let layout = StridedLayout::new(&[2, 2], &[1, 2]).unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::workspace_requirements(
                (layout, Axis::Index(1)),
                None,
                options,
                context,
            )
            .unwrap(),
        )
        .unwrap();
    let mut output = [99.0; 4];
    for data in [[1.0, -f64::MAX, 3.0, f64::MAX], [1.0, 0.0, 3.0, 2.0]] {
        standardize_axes_into(
            BatchView::new(&data, layout, Axis::Index(1)).unwrap(),
            0,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        assert_eq!(output, [-1.0, 1.0, -1.0, 1.0]);
    }
    for data in [[1.0, 0.0, 3.0, f64::NAN], [1.0, 0.0, 3.0, 0.0]] {
        output.fill(99.0);
        assert!(
            standardize_axes_into(
                BatchView::new(&data, layout, Axis::Index(1)).unwrap(),
                0,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output, [99.0; 4]);
    }
    for (ddof, options) in [
        (2, options),
        (0, options.with_non_finite_policy(NonFinitePolicy::Skip)),
    ] {
        assert!(
            standardize_axes_into(
                BatchView::new(&[1.0, 0.0, 3.0, 2.0], layout, Axis::Index(1)).unwrap(),
                ddof,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output, [99.0; 4]);
    }
    let mut short = StatisticsWorkspace::default();
    assert!(
        standardize_axes_into(
            BatchView::new(&[1.0, 0.0, 3.0, 2.0], layout, Axis::Index(1)).unwrap(),
            0,
            options,
            &mut output,
            &mut short,
            context
        )
        .is_err()
    );
    assert_eq!(output, [99.0; 4]);
}

#[cfg(feature = "std")]
#[test]
fn complex_standardization_parallel_and_executor_failure() {
    struct Fails;
    impl Executor for Fails {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::MIN
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            SequentialExecutor.execute(context, plan, operation)?;
            Err(SciOxideError::invalid_input(
                argument!("executor"),
                InvalidInputReason::Inconsistent,
                "injected failure after staging",
            ))
        }
    }
    let layout = StridedLayout::new(&[4, 2], &[2, 1]).unwrap();
    let data: Vec<_> = [1.0, 1e200, f64::from_bits(1), 1.0]
        .into_iter()
        .flat_map(|scale| [Complex::new(0.0, 0.0), Complex::new(scale, -scale)])
        .collect();
    let workers = core::num::NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let mut expected = [Complex::new(0.0, 0.0); 8];
    for context in [ExecutionContext::sequential(), ExecutionContext::new(&pool)] {
        let options = StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Auto);
        let mut workspace = StatisticsWorkspace::default();
        workspace
            .try_reserve(
                &StatisticsWorkspace::workspace_requirements(
                    (layout, Axis::Index(1)),
                    None,
                    options,
                    context,
                )
                .unwrap(),
            )
            .unwrap();
        let mut output = [Complex::new(0.0, 0.0); 8];
        standardize_axes_into(
            BatchView::new(&data, layout, Axis::Index(1)).unwrap(),
            0,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        if expected[0].re == 0.0 {
            expected = output;
        }
        assert_eq!(output, expected);
    }
    let context = ExecutionContext::new(&Fails);
    let options = StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Auto);
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::workspace_requirements(
                (layout, Axis::Index(1)),
                None,
                options,
                context,
            )
            .unwrap(),
        )
        .unwrap();
    let sentinel = Complex::new(99.0, 99.0);
    let mut output = [sentinel; 8];
    assert!(
        standardize_axes_into(
            BatchView::new(&data, layout, Axis::Index(1)).unwrap(),
            0,
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, [sentinel; 8]);
}
#[test]
fn rolling_matches_independent_batches_and_preserves_output_on_failure() {
    let data = [1., 2., 4., 8., 3., 6., 9., 12.];
    let shape = [2, 4];
    let strides = [4, 1];
    let view = BatchView::new(
        &data,
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let opts = StatisticsOptions::new();
    let ctx = ExecutionContext::sequential();
    for boundary in [
        RollingBoundary::Truncate,
        RollingBoundary::Repeat,
        RollingBoundary::ZeroPad,
    ] {
        let window = RollingWindow::new(1, 2).with_boundary(boundary);
        let mut w = StatisticsWorkspace::default();
        w.try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), opts, ctx).unwrap()),
        )
        .unwrap();
        for stat in [
            Statistic::mean(),
            Statistic::variance(0),
            Statistic::quantile(
                QuantileProbability::new(0.5).unwrap(),
                QuantileRule::new(QuantileMethod::Linear),
            ),
            Statistic::median_absolute_deviation(QuantileMethod::Linear, 1.),
            Statistic::iqr(QuantileMethod::Linear),
        ] {
            let mut output = [StatisticValue::Location(99.); 8];
            rolling_statistics_into(view, stat, window, opts, &mut output, &mut w, ctx).unwrap();
            for batch in 0..2 {
                for center in 0..4 {
                    let mut expected = Vec::new();
                    for delta in -1isize..=2 {
                        let k = center as isize + delta;
                        if (0..4).contains(&k) {
                            expected.push(data[batch * 4 + k as usize]);
                        } else {
                            match boundary {
                                RollingBoundary::Truncate => (),
                                RollingBoundary::Repeat => {
                                    expected.push(data[batch * 4 + if k < 0 { 0 } else { 3 }])
                                }
                                RollingBoundary::ZeroPad => expected.push(0.),
                            }
                        }
                    }
                    let expected = scioxide_stats::statistic(&expected, Unweighted, stat)
                        .and_then(StatisticValue::scalar)
                        .unwrap();
                    assert!((scalar(output[batch * 4 + center]) - expected).abs() < 1e-12);
                }
            }
        }
        let mut output = [StatisticValue::Location(99.); 8];
        let too_short = window.with_left(20);
        if boundary != RollingBoundary::Truncate {
            assert!(
                rolling_statistics_into(
                    view,
                    Statistic::mean(),
                    too_short,
                    opts,
                    &mut output,
                    &mut w,
                    ctx
                )
                .is_err()
            );
            assert!(output.into_iter().all(|x| scalar(x) == 99.));
        }
    }
}
#[test]
fn strided_masked_weighted_lanes() {
    let data = [1., f64::NAN, 2., 4., 3., 6.];
    let mask = [true, false, true, true, true, true];
    let weights = [1u64, 1, 2, 2, 1, 1];
    let shape = [3, 2];
    let strides = [2, 1];
    let view = BatchView::new(
        &data,
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_validity(&mask)
    .unwrap()
    .with_weights(Weights::<_, Frequency>::new(&weights))
    .unwrap();
    // Mask and weights are batch-major/sample-minor, independently of physical layout.
    let opts = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Skip)
        .with_non_finite_policy(NonFinitePolicy::Skip);
    let ctx = ExecutionContext::sequential();
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(&(StatisticsWorkspace::workspace_requirements(view, None, opts, ctx).unwrap()))
        .unwrap();
    let mut out = [StatisticValue::Location(0.); 2];
    statistics_into(view, Statistic::mean(), opts, &mut out, &mut w, ctx).unwrap();
    assert!((scalar(out[0]) - 7. / 3.).abs() < 1e-14);
    assert_eq!(scalar(out[1]), 5.);
}
#[test]
fn directional_and_complex_axes() {
    let shape = [2, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let data = [3f64, 0., 0., 4.];
    let view = BatchView::new(&data, layout, Axis::Index(0))
        .and_then(|view| view.with_coordinates(Axis::Index(1)))
        .unwrap();
    let ctx = ExecutionContext::sequential();
    let opts = StatisticsOptions::new();
    let mut w = DirectionalWorkspace::default();
    w.try_reserve(&DirectionalWorkspace::workspace_requirements(view, opts, ctx).unwrap())
        .unwrap();
    let mut direction = [99.; 2];
    let mut report = [DirectionalSummary {
        resultant_length: 0.,
        direction_defined: false,
    }];
    directional_statistics_into(
        view,
        DirectionNormalization::Normalize,
        opts,
        &mut direction,
        &mut report,
        &mut w,
        ctx,
    )
    .unwrap();
    assert!((direction[0] - 0.5f64.sqrt()).abs() < 1e-14);
    assert!((report[0].resultant_length - 0.5f64.sqrt()).abs() < 1e-14);
    let before = direction;
    assert!(
        directional_statistics_into(
            view,
            DirectionNormalization::RequireUnit(1e-6),
            opts,
            &mut direction,
            &mut report,
            &mut w,
            ctx
        )
        .is_err()
    );
    assert_eq!(before, direction);
    let values = [
        Complex::new(1f64, 1.),
        Complex::new(3., 3.),
        Complex::new(0., 0.),
        Complex::new(0., 0.),
    ];
    let window = RollingWindow::new(1, 0).with_boundary(RollingBoundary::Truncate);
    let mut nw = StatisticsWorkspace::default();
    nw.try_reserve(
        &StatisticsWorkspace::workspace_requirements(
            (layout, Axis::Index(1)),
            Some(window),
            opts,
            ctx,
        )
        .unwrap(),
    )
    .unwrap();
    let mut out = [StatisticValue::Dispersion(99.); 4];
    rolling_statistics_into(
        BatchView::new(&values, layout, Axis::Index(1)).unwrap(),
        Statistic::variance(0),
        window,
        opts,
        &mut out,
        &mut nw,
        ctx,
    )
    .unwrap();
    assert!(matches!(out[1],StatisticValue::Dispersion(x) if (x-2.).abs()<1e-14));
    assert!(matches!(out[3], StatisticValue::Dispersion(0.)));
}

fn directional_precision<S: RealScalar>() {
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    let context = ExecutionContext::sequential();
    let options = StatisticsOptions::new();
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    for scale in [S::ONE, S::MAX / scalar(4.0), S::MIN_SUBNORMAL] {
        let data = [scale, S::ZERO, S::ZERO, scale];
        for masses in [
            [S::ONE, scalar(3.0)],
            [S::MAX / scalar(4.0), S::MAX / scalar(4.0) * scalar(3.0)],
            [S::MIN_SUBNORMAL, S::MIN_SUBNORMAL * scalar(3.0)],
        ] {
            let view = BatchView::new(&data, layout, Axis::Index(0))
                .and_then(|view| view.with_coordinates(Axis::Index(1)))
                .unwrap()
                .with_weights(Weights::<_, Mass>::new(&masses))
                .unwrap();
            let mut workspace = DirectionalWorkspace::default();
            workspace
                .try_reserve(
                    &DirectionalWorkspace::workspace_requirements(view, options, context).unwrap(),
                )
                .unwrap();
            let mut direction = [S::ZERO; 2];
            let mut reports = [DirectionalSummary {
                resultant_length: S::ZERO,
                direction_defined: false,
            }];
            directional_statistics_into(
                view,
                DirectionNormalization::Normalize,
                options,
                &mut direction,
                &mut reports,
                &mut workspace,
                context,
            )
            .unwrap();
            let tolerance = 128.0 * S::EPSILON.into_f64();
            assert!((direction[0].into_f64() - 1.0 / 10.0_f64.sqrt()).abs() <= tolerance);
            assert!((direction[1].into_f64() - 3.0 / 10.0_f64.sqrt()).abs() <= tolerance);
            assert!(
                (reports[0].resultant_length.into_f64() - 10.0_f64.sqrt() / 4.0).abs() <= tolerance
            );
            assert!(reports[0].direction_defined);
        }
    }
    let opposite = [S::ONE, S::ZERO, -S::ONE, S::ZERO];
    for (offset, defined) in [(8.0, false), (32.0, true)] {
        let masses = [S::ONE, S::ONE - scalar(offset) * S::EPSILON];
        let view = BatchView::new(&opposite, layout, Axis::Index(0))
            .and_then(|view| view.with_coordinates(Axis::Index(1)))
            .unwrap()
            .with_weights(Weights::<_, Mass>::new(&masses))
            .unwrap();
        let mut workspace = DirectionalWorkspace::default();
        workspace
            .try_reserve(
                &DirectionalWorkspace::workspace_requirements(view, options, context).unwrap(),
            )
            .unwrap();
        let mut direction = [S::ONE; 2];
        let mut reports = [DirectionalSummary {
            resultant_length: S::ZERO,
            direction_defined: false,
        }];
        directional_statistics_into(
            view,
            DirectionNormalization::Normalize,
            options,
            &mut direction,
            &mut reports,
            &mut workspace,
            context,
        )
        .unwrap();
        assert_eq!(reports[0].direction_defined, defined);
        assert_eq!(
            direction,
            if defined {
                [S::ONE, S::ZERO]
            } else {
                [S::ZERO; 2]
            }
        );
    }
}

#[test]
fn directional_native_and_extended_precision() {
    directional_precision::<f32>();
    directional_precision::<f64>();
}

#[test]
fn directional_strides_omission_and_transactional_failures() {
    // Logical axes: batch, sample, coordinate, stored coordinate-major.
    let layout = StridedLayout::new(&[2, 3, 2], &[3, 1, 6]).unwrap();
    let data = [
        1.0,
        0.0,
        f64::NAN,
        1.0,
        -1.0,
        f64::NAN,
        0.0,
        1.0,
        f64::NAN,
        0.0,
        0.0,
        f64::NAN,
    ];
    let masses = [1, 3, 0, 1, 1, 1];
    let validity = [true, true, true, true, true, false];
    let view = BatchView::new(&data, layout, Axis::Index(1))
        .and_then(|view| view.with_coordinates(Axis::Index(2)))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&masses))
        .unwrap()
        .with_validity(&validity)
        .unwrap();
    let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
    let context = ExecutionContext::sequential();
    let mut workspace = DirectionalWorkspace::default();
    workspace
        .try_reserve(&DirectionalWorkspace::workspace_requirements(view, options, context).unwrap())
        .unwrap();
    let mut directions = [99.0; 4];
    let sentinel = DirectionalSummary {
        resultant_length: 99.0,
        direction_defined: true,
    };
    let mut reports = [sentinel; 2];
    for normalization in [
        DirectionNormalization::Normalize,
        DirectionNormalization::RequireUnit(0.0),
    ] {
        directional_statistics_into(
            view,
            normalization,
            options,
            &mut directions,
            &mut reports,
            &mut workspace,
            context,
        )
        .unwrap();
        assert!((directions[0] - 1.0 / 10.0_f64.sqrt()).abs() < 1e-14);
        assert!((directions[1] - 3.0 / 10.0_f64.sqrt()).abs() < 1e-14);
        assert_eq!(&directions[2..], &[0.0, 0.0]);
        assert!(!reports[1].direction_defined);
    }
    for tolerance in [f64::NAN, -1.0] {
        assert!(
            directional_statistics_into(
                view,
                DirectionNormalization::RequireUnit(tolerance),
                options,
                &mut directions,
                &mut reports,
                &mut workspace,
                context
            )
            .is_err()
        );
    }
    let positive = [1; 6];
    let skip_nonfinite = options.with_non_finite_policy(NonFinitePolicy::Skip);
    let nonfinite_view = BatchView::new(&data, layout, Axis::Index(1))
        .and_then(|view| view.with_coordinates(Axis::Index(2)))
        .unwrap()
        .with_weights(Weights::<_, Frequency>::new(&positive))
        .unwrap();
    directional_statistics_into(
        nonfinite_view,
        DirectionNormalization::Normalize,
        skip_nonfinite,
        &mut directions,
        &mut reports,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!((directions[0] - 0.5_f64.sqrt()).abs() < 1e-14);
    assert_eq!(&directions[2..], &[0.0, 0.0]);
    // The first batch succeeds before a zero vector in the second batch fails.
    let invalid = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let invalid_view = BatchView::new(&invalid, layout, Axis::Index(1))
        .and_then(|view| view.with_coordinates(Axis::Index(2)))
        .unwrap();
    directions.fill(99.0);
    reports.fill(sentinel);
    assert!(
        directional_statistics_into(
            invalid_view,
            DirectionNormalization::Normalize,
            options,
            &mut directions,
            &mut reports,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(directions, [99.0; 4]);
    assert!(
        reports
            .iter()
            .all(|r| r.resultant_length == 99.0 && r.direction_defined)
    );
    assert!(
        directional_statistics_into(
            view,
            DirectionNormalization::Normalize,
            StatisticsOptions::new(),
            &mut directions,
            &mut reports,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(directions, [99.0; 4]);
    let mut empty = DirectionalWorkspace::default();
    assert!(
        directional_statistics_into(
            view,
            DirectionNormalization::Normalize,
            options,
            &mut directions,
            &mut reports,
            &mut empty,
            context
        )
        .is_err()
    );
    assert_eq!(directions, [99.0; 4]);
    struct Cancelled;
    impl Cancellation for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert!(
        directional_statistics_into(
            view,
            DirectionNormalization::Normalize,
            options,
            &mut directions,
            &mut reports,
            &mut workspace,
            context.with_cancellation(&Cancelled)
        )
        .is_err()
    );
    assert_eq!(directions, [99.0; 4]);
    assert!(
        reports
            .iter()
            .all(|r| r.resultant_length == 99.0 && r.direction_defined)
    );
}

#[cfg(feature = "std")]
#[test]
fn directional_parallel_batches_match_sequential() {
    let data: Vec<f64> = (0..8)
        .flat_map(|i| {
            let scale = if i % 2 == 0 { 1.0 } else { 1e200 };
            [scale, 0.0, 0.0, scale]
        })
        .collect();
    let layout = StridedLayout::new(&[8, 2, 2], &[4, 2, 1]).unwrap();
    let view = BatchView::new(&data, layout, Axis::Index(1))
        .and_then(|view| view.with_coordinates(Axis::Index(2)))
        .unwrap();
    let workers = core::num::NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let mut expected = [0.0; 16];
    for context in [ExecutionContext::sequential(), ExecutionContext::new(&pool)] {
        let options = StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Auto);
        let mut workspace = DirectionalWorkspace::default();
        workspace
            .try_reserve(
                &DirectionalWorkspace::workspace_requirements(view, options, context).unwrap(),
            )
            .unwrap();
        let mut directions = [0.0; 16];
        let mut reports = [DirectionalSummary {
            resultant_length: 0.0,
            direction_defined: false,
        }; 8];
        directional_statistics_into(
            view,
            DirectionNormalization::Normalize,
            options,
            &mut directions,
            &mut reports,
            &mut workspace,
            context,
        )
        .unwrap();
        if expected[0] == 0.0 {
            expected = directions;
        }
        assert_eq!(directions, expected);
        assert!(reports.iter().all(|r| r.direction_defined && (r.resultant_length - 0.5_f64.sqrt()).abs() < 1e-14));
    }
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_reproduces_rolling() {
    use core::num::NonZeroUsize;
    let x: Vec<f64> = (0..64).map(|i| (i % 13) as f64).collect();
    let shape = [8, 8];
    let strides = [8, 1];
    let view = BatchView::new(
        &x,
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let window = RollingWindow::new(2, 2).with_boundary(RollingBoundary::Repeat);
    let seq = ExecutionContext::sequential();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let ctx = ExecutionContext::new(&pool);
    for stat in [
        Statistic::l_moment(2, scioxide_stats::LMomentNormalization::Raw),
        Statistic::mean(),
        Statistic::variance(0),
        Statistic::standard_deviation(1),
    ] {
        let mut options = StatisticsOptions::new();
        let mut w = StatisticsWorkspace::default();
        w.try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, seq)
                .unwrap()),
        )
        .unwrap();
        let mut expected = vec![StatisticValue::Location(0.); 64];
        rolling_statistics_into(view, stat, window, options, &mut expected, &mut w, seq).unwrap();

        options = options.with_execution_policy(ExecutionPolicy::Parallel(workers));
        w.try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, ctx)
                .unwrap()),
        )
        .unwrap();
        let mut actual = vec![StatisticValue::Location(0.); 64];
        rolling_statistics_into(view, stat, window, options, &mut actual, &mut w, ctx).unwrap();
        assert_eq!(
            actual.into_iter().map(scalar).collect::<Vec<_>>(),
            expected.into_iter().map(scalar).collect::<Vec<_>>()
        );
    }
}
#[test]
fn admission_failures_and_empty_shapes_are_transactional() {
    let shape = [2, 3];
    let strides = [3, 1];
    let data = [1., 2., 3., 1., f64::NAN, 3.];
    let view = BatchView::new(
        &data,
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, None, options, context).unwrap()),
        )
        .unwrap();
    let mut out = [StatisticValue::Location(99.); 2];
    assert!(
        statistics_into(
            view,
            Statistic::mean(),
            options,
            &mut out,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert!(out.into_iter().all(|v| scalar(v) == 99.));
    let shape = [2, 0];
    let strides = [0, 1];
    let view = BatchView::<f64>::new(
        &[],
        StridedLayout::new(&shape, &strides).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let window = RollingWindow::new(1, 1).with_boundary(RollingBoundary::Truncate);
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)
                .unwrap()),
        )
        .unwrap();
    rolling_statistics_into(
        view,
        Statistic::mean(),
        window,
        options,
        &mut [],
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(
        RollingWindow::new(usize::MAX, 1)
            .with_boundary(RollingBoundary::Repeat)
            .width()
            .is_err()
    );
}
