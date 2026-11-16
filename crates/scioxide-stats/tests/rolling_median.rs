//! Rolling selection agrees with independently constructed batch windows.
#![allow(clippy::unwrap_used)]

use scioxide_core::*;
use scioxide_stats::*;

#[path = "support/executor.rs"]
mod executor;

const METHODS: [QuantileMethod; 9] = [
    QuantileMethod::NearestRank,
    QuantileMethod::AveragedInverseCdf,
    QuantileMethod::ClosestObservation,
    QuantileMethod::InterpolatedInverseCdf,
    QuantileMethod::Hazen,
    QuantileMethod::Weibull,
    QuantileMethod::MedianUnbiased,
    QuantileMethod::NormalUnbiased,
    QuantileMethod::Linear,
];

fn scalar<S: RealScalar>(value: StatisticValue<S>) -> S {
    value.scalar().unwrap()
}

fn windows<S: RealScalar>(data: &[S], validity: &[bool], omit: bool) {
    let n = data.len() / 2;
    let shape = [n, 2];
    let layout = StridedLayout::new(&shape, &[2, 1]).unwrap();
    let view = BatchView::new(data, layout, Axis::Index(0))
        .unwrap()
        .with_validity(validity)
        .unwrap();
    let executor = executor::PartitionExecutor(core::num::NonZeroUsize::new(2).unwrap());
    let context = ExecutionContext::new(&executor);
    let options = StatisticsOptions::new()
        .with_missing_policy(if omit {
            MissingPolicy::Skip
        } else {
            MissingPolicy::Reject
        })
        .with_non_finite_policy(if omit {
            NonFinitePolicy::Skip
        } else {
            NonFinitePolicy::Reject
        })
        .with_execution_policy(ExecutionPolicy::Parallel(
            core::num::NonZeroUsize::new(2).unwrap(),
        ));
    for boundary in [
        RollingBoundary::Truncate,
        RollingBoundary::Repeat,
        RollingBoundary::ZeroPad,
    ] {
        for (left, right) in [(0, 0), (1, 0), (0, 3), (2, 2), (8, 8), (n + 2, n + 3)] {
            let window = RollingWindow::new(left, right).with_boundary(boundary);
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
            for method in METHODS {
                let mut expected = Vec::new();
                let mut empty = false;
                for batch in 0..2 {
                    for center in 0..n {
                        let mut sample = Vec::new();
                        for offset in 0..left + right + 1 {
                            let index = center as isize + offset as isize - left as isize;
                            let index = if (0..n as isize).contains(&index) {
                                Some(index as usize)
                            } else {
                                match boundary {
                                    RollingBoundary::Truncate => continue,
                                    RollingBoundary::Repeat => {
                                        Some(if index < 0 { 0 } else { n - 1 })
                                    }
                                    RollingBoundary::ZeroPad => None,
                                }
                            };
                            if let Some(i) = index {
                                let value = data[2 * i + batch];
                                if validity[batch * n + i] && value.is_finite() {
                                    sample.push(value);
                                }
                            } else {
                                sample.push(S::ZERO);
                            }
                        }
                        if sample.is_empty() {
                            empty = true;
                        } else {
                            expected.push(
                                quantile_in_place(
                                    &mut sample,
                                    QuantileProbability::new(0.5).unwrap(),
                                    QuantileRule::new(method),
                                )
                                .unwrap()
                                .value(),
                            );
                        }
                    }
                }
                let mut output = vec![StatisticValue::Location(S::ONE); 2 * n];
                let result = rolling_statistics_into(
                    view,
                    Statistic::quantile(
                        QuantileProbability::new(0.5).unwrap(),
                        QuantileRule::new(method),
                    ),
                    window,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                );
                if empty {
                    assert!(result.is_err());
                    assert!(output.into_iter().all(|value| scalar(value) == S::ONE));
                } else {
                    result.unwrap();
                    for (actual, expected) in output.into_iter().map(scalar).zip(expected) {
                        let scale = actual.abs().max(expected.abs()).max(S::ONE);
                        assert!(
                            ((actual / scale) - (expected / scale)).abs()
                                <= S::EPSILON * S::try_from_f64_lossy(8.0).unwrap(),
                            "{boundary:?}/{left}/{right}/{method:?}: {actual:?} != {expected:?}"
                        );
                        if actual == S::ZERO && expected == S::ZERO {
                            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                        }
                    }
                }
            }
        }
    }
}

fn cases<S: RealScalar>() {
    let mut data: Vec<_> = (0..66)
        .map(|i| S::try_from_f64_lossy(((i * 17) % 13) as f64 - 6.0).unwrap())
        .collect();
    let mut validity = vec![true; data.len()];
    windows(&data, &validity, false);
    data[0] = S::NAN;
    data[9] = S::INFINITY;
    data[65] = S::NEG_INFINITY;
    validity[3] = false;
    validity[36] = false;
    windows(&data, &validity, true);
    windows(
        &[
            S::MAX,
            S::ZERO,
            -S::MAX,
            -S::ZERO,
            S::MIN_POSITIVE,
            -S::MIN_POSITIVE,
        ],
        &[true; 6],
        false,
    );
    windows(&[-S::ZERO, S::ZERO], &[true; 2], false);
}

#[test]
fn independent_windows_f32() {
    cases::<f32>();
}

#[test]
fn independent_windows_f64() {
    cases::<f64>();
}

#[test]
fn median_failures_are_atomic_and_workspace_is_reusable() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter(AtomicUsize);
    impl Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 10
        }
    }
    let mut data = [2.0; 65];
    let layout = StridedLayout::new(&[65], &[1]).unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let window = RollingWindow::new(8, 0).with_boundary(RollingBoundary::Repeat);
    let statistic = Statistic::quantile(
        QuantileProbability::new(0.5).unwrap(),
        QuantileRule::new(QuantileMethod::Linear),
    );
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)
                .unwrap()),
        )
        .unwrap();
    let mut output = [StatisticValue::Location(99.0); 65];
    data[60] = f64::NAN;
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
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
    assert!(output.iter().all(|&value| scalar(value) == 99.0));
    data[60] = 2.0;
    let view = BatchView::new(&data, layout, Axis::Index(0)).unwrap();
    let mut validity = [true; 65];
    validity[60] = false;
    assert!(
        rolling_statistics_into(
            view.with_validity(&validity).unwrap(),
            statistic,
            window,
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert!(output.iter().all(|&value| scalar(value) == 99.0));
    let cancelled = StopAfter(AtomicUsize::new(0));
    assert!(
        rolling_statistics_into(
            view,
            statistic,
            window,
            options,
            &mut output,
            &mut workspace,
            context.with_cancellation(&cancelled)
        )
        .is_err()
    );
    assert!(output.iter().all(|&value| scalar(value) == 99.0));
    for invalid_window in [window.with_left(100), window.with_left(usize::MAX)] {
        assert!(
            rolling_statistics_into(
                view,
                statistic,
                invalid_window,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert!(output.iter().all(|&value| scalar(value) == 99.0));
    }
    let mut masses = [1.0; 65];
    masses[60] = -1.0;
    let mut frequencies = [1u64; 65];
    frequencies[60] = u64::MAX;
    macro_rules! check_weights {
        ($weights:expr, $rule:expr) => {{
            let request = Statistic::quantile(QuantileProbability::new(0.5).unwrap(), $rule);
            assert!(
                rolling_statistics_into(
                    view.with_weights($weights).unwrap(),
                    request,
                    window,
                    options,
                    &mut output,
                    &mut workspace,
                    context
                )
                .is_err()
            );
            assert!(output.iter().all(|&value| scalar(value) == 99.0));
        }};
    }
    check_weights!(
        Weights::<_, Mass>::new(&masses),
        QuantileRule::inverse_ecdf()
    );
    check_weights!(
        Weights::<_, Frequency>::new(&frequencies),
        QuantileRule::new(QuantileMethod::Linear)
    );
    rolling_statistics_into(
        view,
        statistic,
        window,
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(output.into_iter().all(|value| scalar(value) == 2.0));
}

#[test]
fn empty_axis_and_huge_truncated_extents() {
    for data in [&[][..], &[3.0][..]] {
        let shape = [data.len()];
        let view = BatchView::new(
            data,
            StridedLayout::new(&shape, &[1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap();
        let window = RollingWindow::new(usize::MAX - 1, 0).with_boundary(RollingBoundary::Truncate);
        let options = StatisticsOptions::new();
        let context = ExecutionContext::sequential();
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
        let mut output = vec![StatisticValue::Location(0.0); data.len()];
        rolling_statistics_into(
            view,
            Statistic::quantile(
                QuantileProbability::new(0.5).unwrap(),
                QuantileRule::new(QuantileMethod::Linear),
            ),
            window,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        assert!(output.into_iter().all(|value| scalar(value) == 3.0));
    }
}
