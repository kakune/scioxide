//! Rolling updates against independent extended-range, centered batch reductions.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_stats::*;

fn close<S: RealScalar>(actual: S, expected: S, tolerance: f64) {
    let (a, e) = (actual.into_f64(), expected.into_f64());
    assert!(
        a == e || (a - e).abs() <= tolerance * e.abs(),
        "{a:e} != {e:e} (relative tolerance {tolerance:e})"
    );
}

fn numeric_close<T: NumericScalar>(actual: StatisticValue<T>, expected: StatisticValue<T>) {
    let tolerance = 8192.0 * T::Real::EPSILON.into_f64();
    match (actual, expected) {
        (StatisticValue::Location(a), StatisticValue::Location(e)) => {
            close(a.real_component(), e.real_component(), tolerance);
            close(a.imaginary_component(), e.imaginary_component(), tolerance);
        }
        (StatisticValue::Dispersion(a), StatisticValue::Dispersion(e)) => close(a, e, tolerance),
        _ => unreachable!(),
    }
}

fn batch_summary<T: NumericScalar>(values: &[T], statistic: Statistic<T>) -> StatisticValue<T> {
    if matches!(statistic, Statistic::Mean(_)) {
        // Sum components directly in extended compensated arithmetic, independent
        // of the rolling update and of a rounded origin/offset subtraction.
        let mut re = ScaledAccumulator::<f64>::ZERO;
        let mut im = ScaledAccumulator::<f64>::ZERO;
        for &x in values {
            re.add_value(x.real_component().into_f64()).unwrap();
            im.add_value(x.imaginary_component().into_f64()).unwrap();
        }
        let n = Scaled::from_value(values.len() as f64).unwrap();
        StatisticValue::Location(T::from_components(
            re.total().unwrap().div(n).unwrap().round_to().unwrap(),
            im.total().unwrap().div(n).unwrap().round_to().unwrap(),
        ))
    } else {
        scioxide_stats::statistic(values, Unweighted, statistic).unwrap()
    }
}

fn batch_window<T: Copy + AlgebraicScalar>(
    data: &[T],
    center: usize,
    window: RollingWindow,
) -> Vec<T> {
    (-(window.left() as isize)..=window.right() as isize)
        .filter_map(|delta| {
            let index = center as isize + delta;
            if index >= 0 && index < data.len() as isize {
                Some(data[index as usize])
            } else {
                match window.boundary() {
                    RollingBoundary::Truncate => None,
                    RollingBoundary::Repeat => {
                        Some(data[if index < 0 { 0 } else { data.len() - 1 }])
                    }
                    RollingBoundary::ZeroPad => Some(T::ZERO),
                }
            }
        })
        .collect()
}

fn numeric_windows<T: NumericScalar>(data: &[T]) {
    let shape = [data.len(), 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    // Two interleaved, identical lanes exercise resolved origins and strides.
    let interleaved: Vec<_> = data.iter().flat_map(|&x| [x, x]).collect();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    for (left, right) in [(0, 0), (0, 7), (8, 8), (19, 0), (90, 80)] {
        for boundary in [
            RollingBoundary::Truncate,
            RollingBoundary::Repeat,
            RollingBoundary::ZeroPad,
        ] {
            let window = RollingWindow::new(left, right).with_boundary(boundary);
            let mut workspace = StatisticsWorkspace::default();
            workspace
                .try_reserve(
                    &StatisticsWorkspace::workspace_requirements(
                        (layout, Axis::Index(0)),
                        Some(window),
                        options,
                        context,
                    )
                    .unwrap(),
                )
                .unwrap();
            let mut actual = vec![StatisticValue::Location(T::ZERO); interleaved.len()];
            for statistic in [
                Statistic::mean(),
                Statistic::variance(0),
                Statistic::standard_deviation(0),
            ] {
                rolling_statistics_into(
                    BatchView::new(&interleaved, layout, Axis::Index(0)).unwrap(),
                    statistic,
                    window,
                    options,
                    &mut actual,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (center, &value) in actual.iter().enumerate() {
                    let sample = batch_window(data, center % data.len(), window);
                    let expected = batch_summary(&sample, statistic);
                    numeric_close(value, expected);
                }
            }
        }
    }
}

fn real_windows<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let offset = s(1e12);
    let unit = offset * S::EPSILON * s(4.0);
    let tiny = S::MIN_POSITIVE.sqrt();
    for data in [
        vec![s(3.0); 65],
        (0..65).map(|i| s((i % 17) as f64 - 8.0)).collect(),
        (0..65)
            .map(|i| offset + s((i % 13) as f64) * unit)
            .collect(),
        (0..65).map(|i| s((i % 7 + 1) as f64) * tiny).collect(),
        (0..65)
            .map(|i| if i == 9 { s(1e18) } else { s(2.0) })
            .collect(),
    ] {
        numeric_windows(&data);
        let shape = [data.len()];
        let strides = [1];
        let view = BatchView::new(
            &data,
            StridedLayout::new(&shape, &strides).unwrap(),
            Axis::Index(0),
        )
        .unwrap();
        let options = StatisticsOptions::new();
        let context = ExecutionContext::sequential();
        for boundary in [
            RollingBoundary::Truncate,
            RollingBoundary::Repeat,
            RollingBoundary::ZeroPad,
        ] {
            let window = RollingWindow::new(8, 8).with_boundary(boundary);
            let mut workspace = StatisticsWorkspace::default();
            workspace
                .try_reserve(
                    &(StatisticsWorkspace::workspace_requirements(
                        view,
                        Some(window),
                        options,
                        context,
                    )
                    .unwrap()),
                )
                .unwrap();
            for statistic in [
                Statistic::mean(),
                Statistic::variance(0),
                Statistic::variance(1),
                Statistic::standard_deviation(1),
            ] {
                let mut actual = vec![StatisticValue::Location(S::ZERO); data.len()];
                rolling_statistics_into(
                    view,
                    statistic,
                    window,
                    options,
                    &mut actual,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (center, &value) in actual.iter().enumerate() {
                    let expected = scioxide_stats::statistic(
                        &batch_window(&data, center, window),
                        Unweighted,
                        statistic,
                    )
                    .and_then(StatisticValue::scalar)
                    .unwrap();
                    let value = value.scalar().unwrap();
                    close(value, expected, 8192.0 * S::EPSILON.into_f64());
                }
            }
        }
    }
}

#[test]
fn real_updates_match_stable_batches_across_scales_and_boundaries() {
    real_windows::<f32>();
    real_windows::<f64>();
}

#[test]
fn complex_updates_match_squared_modulus_batches() {
    let data: Vec<_> = (0..65)
        .map(|i| {
            Complex::new(
                1e12 + (i % 11) as f64 * 0.001,
                -3e11 + (i % 17) as f64 * 0.002,
            )
        })
        .collect();
    numeric_windows(&data);
    let data: Vec<_> = (0..65)
        .map(|i| Complex::new((i % 11) as f32, (i % 7) as f32))
        .collect();
    numeric_windows(&data);
}

#[test]
fn cancellation_in_mean_and_extended_range_fallback_are_local() {
    numeric_windows(&[1e16_f64, 1.0, -1e16, 2.0, 3.0, -3.0]);
    numeric_windows(&[1e18_f64, 3.0, 1e-18, 0.0, 0.0, 0.0]);
    numeric_windows(&[1e-300_f64, 2e-300, 3e-300, -2e-300, 0.0]);
    // Deviations remain representable although variances do not.
    for statistic in [
        Statistic::mean(),
        Statistic::standard_deviation(0),
        Statistic::variance(0),
    ] {
        let data = [1e300_f64, -1e300, 1.0, 1.0, 1.0];
        let shape = [data.len()];
        let strides = [1];
        let layout = StridedLayout::new(&shape, &strides).unwrap();
        let window = RollingWindow::new(1, 0).with_boundary(RollingBoundary::Truncate);
        let context = ExecutionContext::sequential();
        let options = StatisticsOptions::new();
        let mut workspace = StatisticsWorkspace::default();
        workspace
            .try_reserve(
                &StatisticsWorkspace::workspace_requirements(
                    (layout, Axis::Index(0)),
                    Some(window),
                    options,
                    context,
                )
                .unwrap(),
            )
            .unwrap();
        let mut actual = [StatisticValue::Location(99.0); 5];
        let result = rolling_statistics_into(
            BatchView::new(&data, layout, Axis::Index(0)).unwrap(),
            statistic,
            window,
            options,
            &mut actual,
            &mut workspace,
            context,
        );
        if matches!(statistic, Statistic::Variance(..)) {
            assert!(result.is_err());
            assert!(
                actual
                    .iter()
                    .all(|x| matches!(x, StatisticValue::Location(99.0)))
            );
        } else {
            result.unwrap();
            for (center, value) in actual.into_iter().enumerate() {
                numeric_close(
                    value,
                    scioxide_stats::statistic(
                        &batch_window(&data, center, window),
                        Unweighted,
                        statistic,
                    )
                    .unwrap(),
                );
            }
        }
    }
}

#[test]
fn missing_nonfinite_and_weighted_endpoints_follow_batch_admission() {
    let data = [1.0, f64::NAN, 4.0, 8.0, f64::INFINITY, 2.0];
    let present = [false, true, true, true, true, true];
    let masses = [0.0, 0.0, 1.0, 2.0, 0.0, 3.0];
    let frequencies = [0, 0, 1, 2, 0, 3];
    let shape = [data.len()];
    let strides = [1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let base = BatchView::new(&data, layout, Axis::Index(0))
        .unwrap()
        .with_validity(&present)
        .unwrap();
    let options = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Skip)
        .with_non_finite_policy(NonFinitePolicy::Skip);
    let context = ExecutionContext::sequential();
    macro_rules! check_weights {
        ($input:expr, $source:expr, $model:ty) => {{
            let weights = $input;
            let view = base.with_weights(weights).unwrap();
            for boundary in [RollingBoundary::Truncate, RollingBoundary::Repeat] {
                let window = RollingWindow::new(3, 3).with_boundary(boundary);
                let mut workspace = StatisticsWorkspace::default();
                workspace
                    .try_reserve(
                        &(StatisticsWorkspace::workspace_requirements(
                            view,
                            Some(window),
                            options,
                            context,
                        )
                        .unwrap()),
                    )
                    .unwrap();
                let mut actual = [StatisticValue::Location(99.0); 6];
                rolling_statistics_into(
                    view,
                    Statistic::mean(),
                    window,
                    options,
                    &mut actual,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (center, value) in actual.into_iter().enumerate() {
                    let indices: Vec<_> = (-(window.left() as isize)..=window.right() as isize)
                        .filter_map(|d| {
                            let i = center as isize + d;
                            let i = if boundary == RollingBoundary::Repeat {
                                i.clamp(0, data.len() as isize - 1)
                            } else {
                                i
                            };
                            (i >= 0 && i < data.len() as isize).then_some(i as usize)
                        })
                        .filter(|&i| present[i] && data[i].is_finite())
                        .collect();
                    let values: Vec<_> = indices.iter().map(|&i| data[i]).collect();
                    let gathered: Vec<_> = indices.iter().map(|&i| ($source)[i]).collect();
                    let weights = Weights::<_, $model>::new(&gathered);
                    let expected = scioxide_stats::statistic(&values, weights, Statistic::mean())
                        .and_then(StatisticValue::scalar)
                        .unwrap();
                    let value = value.scalar().unwrap();
                    close(value, expected, 1e-14);
                }
            }
        }};
    }
    check_weights!(Unweighted, [1.0; 6], Mass);
    check_weights!(Weights::<_, Mass>::new(&masses), masses, Mass);
    check_weights!(
        Weights::<_, Frequency>::new(&frequencies),
        frequencies,
        Frequency
    );
}

#[test]
fn failures_after_staged_windows_preserve_outputs_and_workspace_recovers() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter(AtomicUsize);
    impl Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 10
        }
    }
    let mut data = [2.0; 65];
    data[60] = f64::NAN;
    let shape = [data.len()];
    let strides = [1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let window = RollingWindow::new(8, 0).with_boundary(RollingBoundary::Truncate);
    let context = ExecutionContext::sequential();
    let options = StatisticsOptions::new();
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)
                .unwrap()),
        )
        .unwrap();
    let mut numeric = StatisticsWorkspace::default();
    numeric
        .try_reserve(
            &StatisticsWorkspace::workspace_requirements(
                (layout, Axis::Index(0)),
                Some(window),
                options,
                context,
            )
            .unwrap(),
        )
        .unwrap();
    let mut output = [StatisticValue::Location(99.0); 65];
    let mut numeric_output = [StatisticValue::Location(99.0); 65];
    for statistic in [
        Statistic::mean(),
        Statistic::variance(0),
        Statistic::standard_deviation(0),
    ] {
        assert!(
            rolling_statistics_into(
                view,
                statistic,
                window,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert!(
            output
                .iter()
                .all(|x| matches!(x, StatisticValue::Location(99.0)))
        );
    }
    assert!(
        rolling_statistics_into(
            BatchView::new(&data, layout, Axis::Index(0)).unwrap(),
            Statistic::mean(),
            window,
            options,
            &mut numeric_output,
            &mut numeric,
            context
        )
        .is_err()
    );
    assert!(
        numeric_output
            .iter()
            .all(|x| matches!(x, StatisticValue::Location(99.0)))
    );

    data[60] = 2.0;
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let cancellation = StopAfter(AtomicUsize::new(0));
    assert!(
        rolling_statistics_into(
            view,
            Statistic::mean(),
            window,
            options,
            &mut output,
            &mut workspace,
            context.with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert!(
        output
            .iter()
            .all(|x| matches!(x, StatisticValue::Location(99.0)))
    );
    for statistic in [Statistic::variance(1), Statistic::variance(100)] {
        assert!(
            rolling_statistics_into(
                view,
                statistic,
                window,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert!(
            output
                .iter()
                .all(|x| matches!(x, StatisticValue::Location(99.0)))
        );
    }
    rolling_statistics_into(
        view,
        Statistic::mean(),
        window,
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(
        output
            .iter()
            .all(|x| matches!(x, StatisticValue::Location(2.0)))
    );
    let options = options.with_non_finite_policy(NonFinitePolicy::Skip);
    let absent = [f64::NAN; 65];
    let view = BatchView::new(&absent, layout, Axis::Index(0)).unwrap();
    assert!(
        rolling_statistics_into(
            view,
            Statistic::mean(),
            window,
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert!(
        output
            .iter()
            .all(|x| matches!(x, StatisticValue::Location(2.0)))
    );
}

#[test]
fn long_lanes_rebuild_without_accumulating_drift() {
    let data: Vec<f64> = (0..4096)
        .map(|i| 1e12 + ((i * 7919) % 65537) as f64 * 0.001)
        .collect();
    let shape = [data.len()];
    let strides = [1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let window = RollingWindow::new(8, 8).with_boundary(RollingBoundary::Repeat);
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)
                .unwrap()),
        )
        .unwrap();
    let mut actual = vec![StatisticValue::Location(0.0); data.len()];
    rolling_statistics_into(
        view,
        Statistic::standard_deviation(1),
        window,
        options,
        &mut actual,
        &mut workspace,
        context,
    )
    .unwrap();
    for (center, value) in actual.into_iter().enumerate() {
        let StatisticValue::Dispersion(expected) = batch_summary(
            &batch_window(&data, center, window),
            Statistic::standard_deviation(1),
        ) else {
            unreachable!()
        };
        let value = value.scalar().unwrap();
        close(value, expected, 2e-12);
    }
}

#[test]
fn numeric_omission_includes_zero_padding_and_repeated_endpoints() {
    let data = [1.0, f64::NAN, 4.0, 8.0, f64::INFINITY, 2.0];
    let layout = StridedLayout::new(&[6], &[1]).unwrap();
    let options = StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip);
    let context = ExecutionContext::sequential();
    for boundary in [
        RollingBoundary::Truncate,
        RollingBoundary::Repeat,
        RollingBoundary::ZeroPad,
    ] {
        let window = RollingWindow::new(3, 3).with_boundary(boundary);
        let mut workspace = StatisticsWorkspace::default();
        workspace
            .try_reserve(
                &StatisticsWorkspace::workspace_requirements(
                    (layout, Axis::Index(0)),
                    Some(window),
                    options,
                    context,
                )
                .unwrap(),
            )
            .unwrap();
        let mut output = [StatisticValue::Location(99.0); 6];
        for statistic in [
            Statistic::mean(),
            Statistic::variance(0),
            Statistic::standard_deviation(1),
        ] {
            rolling_statistics_into(
                BatchView::new(&data, layout, Axis::Index(0)).unwrap(),
                statistic,
                window,
                options,
                &mut output,
                &mut workspace,
                context,
            )
            .unwrap();
            for (center, value) in output.into_iter().enumerate() {
                let sample: Vec<_> = batch_window(&data, center, window)
                    .into_iter()
                    .filter(|x| x.is_finite())
                    .collect();
                numeric_close(value, batch_summary(&sample, statistic));
            }
        }
    }
}

fn overflowing_variance<S: RealScalar>() {
    let high = S::MAX.sqrt() * S::try_from_f64_lossy(2.0).unwrap();
    let data = [high, -high];
    let layout = StridedLayout::new(&[2], &[1]).unwrap();
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let window = RollingWindow::new(1, 0).with_boundary(RollingBoundary::Repeat);
    let context = ExecutionContext::sequential();
    let options = StatisticsOptions::new();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)
                .unwrap()),
        )
        .unwrap();
    let mut output = [StatisticValue::Location(S::ONE); 2];
    assert!(
        rolling_statistics_into(
            view,
            Statistic::variance(0),
            window,
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert!(
        output
            .iter()
            .all(|x| matches!(x, StatisticValue::Location(x) if *x == S::ONE))
    );
    rolling_statistics_into(
        view,
        Statistic::standard_deviation(0),
        window,
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    assert_eq!(output[0].scalar().unwrap(), S::ZERO);
    let value = output[1].scalar().unwrap();
    close(value, high, 8.0 * S::EPSILON.into_f64());

    let complex = data.map(|x| Complex::new(x, x));
    let mut numeric = StatisticsWorkspace::default();
    numeric
        .try_reserve(
            &StatisticsWorkspace::workspace_requirements(
                (layout, Axis::Index(0)),
                Some(window),
                options,
                context,
            )
            .unwrap(),
        )
        .unwrap();
    let mut output = [StatisticValue::Dispersion(S::ONE); 2];
    rolling_statistics_into(
        BatchView::new(&complex, layout, Axis::Index(0)).unwrap(),
        Statistic::standard_deviation(0),
        window,
        options,
        &mut output,
        &mut numeric,
        context,
    )
    .unwrap();
    for (center, value) in output.into_iter().enumerate() {
        numeric_close(
            value,
            batch_summary(
                &batch_window(&complex, center, window),
                Statistic::standard_deviation(0),
            ),
        );
    }
}

#[test]
fn deviation_projects_independently_of_overflowing_variance() {
    overflowing_variance::<f32>();
    overflowing_variance::<f64>();
}

#[test]
fn rolling_and_batch_means_share_the_batch_center_contract_at_large_offsets() {
    // 4,096 three-point windows at offset 1e15 with sub-unit spacing: every
    // window mean needs the division residual of the shifted center, and the
    // rolling update must reproduce the batch kernel bit for bit.
    let data: Vec<f64> = (0..4_098)
        .map(|i| 1e15 + ((i * 7_919) % 1_009) as f64 * 0.125)
        .collect();
    let shape = [data.len()];
    let layout = StridedLayout::new(&shape, &[1]).unwrap();
    let window = RollingWindow::new(1, 1);
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::workspace_requirements(
                (layout, Axis::Index(0)),
                Some(window),
                options,
                context,
            )
            .unwrap(),
        )
        .unwrap();
    let mut rolling = vec![StatisticValue::Location(0.0); data.len()];
    rolling_statistics_into(
        BatchView::new(&data, layout, Axis::Index(0)).unwrap(),
        Statistic::mean(),
        window,
        options,
        &mut rolling,
        &mut workspace,
        context,
    )
    .unwrap();
    for center in 1..data.len() - 1 {
        let sample = &data[center - 1..=center + 1];
        let mut exact = RangeAccumulator::<f64>::new();
        sample.iter().for_each(|&x| exact.add(x).unwrap());
        let exact = exact.finish_quotient(3.0).unwrap();
        let batch = statistic(sample, Unweighted, Statistic::mean())
            .unwrap()
            .location()
            .unwrap();
        let rolled = rolling[center].location().unwrap();
        // The batch center is faithful: within one unit in the last place.
        assert!(
            batch.to_bits().abs_diff(exact.to_bits()) <= 1,
            "batch window {center}"
        );
        assert_eq!(rolled.to_bits(), batch.to_bits(), "rolling window {center}");
    }
}
