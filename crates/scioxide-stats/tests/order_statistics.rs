#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{ExecutionContext, RealScalar, StridedLayout};
use scioxide_stats::*;

fn expectile<S: RealScalar, const N: usize>(
    workspace: &mut StatisticsWorkspace<S>,
    pairs: &[(S, S); N],
    probability: QuantileProbability,
) -> Result<ExpectileResult<S>, scioxide_core::SciOxideError> {
    let values = pairs.map(|(x, _)| x);
    let masses = pairs.map(|(_, w)| w);
    match workspace.evaluate(
        &values,
        Weights::<_, Mass>::new(&masses),
        Statistic::expectile(probability),
    )? {
        StatisticValue::Order(OrderResult::Expectile(result)) => Ok(result),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected expectile",
        )),
    }
}
const METHODS: [QuantileMethod; 9] = [
    QuantileMethod::NearestRank,
    QuantileMethod::AveragedInverseCdf,
    QuantileMethod::ClosestObservation,
    QuantileMethod::InterpolatedInverseCdf,
    QuantileMethod::Hazen,
    QuantileMethod::Weibull,
    QuantileMethod::Linear,
    QuantileMethod::MedianUnbiased,
    QuantileMethod::NormalUnbiased,
];
fn p(value: f64) -> QuantileProbability {
    QuantileProbability::new(value).unwrap()
}

// Independent hand-derived H&F equation (1), n=4, p=1/4. Values are
// x=[0,10,20,30]; all ranks and interpolants below are exact rationals.
fn exact_definitions<S: RealScalar>() {
    let samples = [0.0, 10.0, 20.0, 30.0].map(|x| S::try_from_f64_exact(x).unwrap());
    let distribution = EmpiricalDistribution::new(&samples, Unweighted).unwrap();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(samples.len(), StatisticsOptions::new())
                .unwrap(),
        )
        .unwrap();
    for (method, expected) in
        METHODS
            .into_iter()
            .zip([0.0, 5.0, 0.0, 0.0, 5.0, 2.5, 7.5, 25.0 / 6.0, 35.0 / 8.0])
    {
        let value = distribution
            .quantile(p(0.25), QuantileRule::new(method))
            .unwrap()
            .value();
        let expected = S::try_from_f64_lossy(expected).unwrap();
        assert!(
            (value - expected).abs()
                <= S::try_from_f64_exact(0.000001)
                    .unwrap_or_else(|_| S::try_from_f64_lossy(0.000001).unwrap())
        );
        for probability in [0.0, f64::from_bits(1), 0.125, 0.25, 0.5, 0.625, 0.75, 1.0] {
            let mut work = [samples[3], samples[1], samples[0], samples[2]];
            let sorted = distribution
                .quantile(p(probability), QuantileRule::new(method))
                .unwrap();
            assert_eq!(
                quantile_in_place(&mut work, p(probability), QuantileRule::new(method)).unwrap(),
                sorted
            );
            assert_eq!(
                workspace
                    .evaluate(
                        &work,
                        Unweighted,
                        Statistic::quantile(p(probability), QuantileRule::new(method))
                    )
                    .and_then(|value| match value {
                        StatisticValue::Order(result) => Ok(result),
                        _ => Err(scioxide_core::SciOxideError::invalid_input(
                            scioxide_core::argument!("value"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "expected order result"
                        )),
                    })
                    .unwrap(),
                OrderResult::Quantile(sorted)
            );
            let mut descriptive = StatisticsWorkspace::default();
            descriptive
                .try_reserve(
                    &StatisticsWorkspace::scalar_requirements(work.len(), StatisticsOptions::new())
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(
                descriptive
                    .evaluate(
                        &work,
                        Unweighted,
                        Statistic::quantile(p(probability), QuantileRule::new(method))
                    )
                    .and_then(StatisticValue::scalar)
                    .unwrap(),
                sorted.value()
            );
            let mut frequency = [
                (samples[0], 1),
                (samples[3], 1),
                (samples[1], 1),
                (samples[2], 1),
            ];
            assert_eq!(
                quantile_in_place(
                    Weights::<_, Frequency>::new(&mut frequency[..]),
                    p(probability),
                    QuantileRule::new(method)
                )
                .unwrap()
                .value(),
                sorted.value()
            );
        }
    }
}
#[test]
fn all_nine_exact_f32_f64() {
    exact_definitions::<f32>();
    exact_definitions::<f64>();
}

#[test]
fn discontinuities_and_even_rank_ties() {
    let d = EmpiricalDistribution::new(&[1.0, 2.0, 3.0, 4.0], Unweighted).unwrap();
    for (prob, expected) in [(0.375, 2.0), (0.625, 2.0), (0.875, 4.0)] {
        assert_eq!(
            d.quantile(
                p(prob),
                QuantileRule::new(QuantileMethod::ClosestObservation)
            )
            .unwrap()
            .value(),
            expected
        );
    }
    let boundary = 0.5_f64;
    assert_eq!(
        d.quantile(
            p(f64::from_bits(boundary.to_bits() - 1)),
            QuantileRule::new(QuantileMethod::AveragedInverseCdf)
        )
        .unwrap()
        .value(),
        2.0
    );
    assert_eq!(
        d.quantile(
            p(boundary),
            QuantileRule::new(QuantileMethod::AveragedInverseCdf)
        )
        .unwrap()
        .value(),
        2.5
    );
    assert_eq!(
        d.quantile(
            p(f64::from_bits(boundary.to_bits() + 1)),
            QuantileRule::new(QuantileMethod::AveragedInverseCdf)
        )
        .unwrap()
        .value(),
        3.0
    );
}
#[test]
fn frequency_replication_and_large_counts() {
    for method in METHODS {
        let d = EmpiricalDistribution::new(&[1.0, 1.0, 4.0, 9.0, 9.0, 9.0], Unweighted).unwrap();
        for probability in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0] {
            let mut pairs = [(9.0, 3), (f64::NAN, 0), (1.0, 2), (4.0, 1)];
            assert_eq!(
                quantile_in_place(
                    Weights::<_, Frequency>::new(&mut pairs[..]),
                    p(probability),
                    QuantileRule::new(method)
                )
                .unwrap()
                .value(),
                d.quantile(p(probability), QuantileRule::new(method))
                    .unwrap()
                    .value()
            );
        }
        let mut huge = [(1.0, (usize::MAX - 1) as u64), (9.0, 1)];
        assert_eq!(
            quantile_in_place(
                Weights::<_, Frequency>::new(&mut huge[..]),
                p(1.0),
                QuantileRule::new(method)
            )
            .unwrap()
            .value(),
            9.0
        );
        assert_eq!(
            quantile_in_place(
                Weights::<_, Frequency>::new(&mut huge[..]),
                p(0.5),
                QuantileRule::new(method)
            )
            .unwrap()
            .value(),
            1.0
        );
    }
    assert!(
        quantile_in_place(
            Weights::<_, Frequency>::new(&mut [(1.0, usize::MAX as u64), (2.0, 1)][..]),
            p(0.5),
            QuantileRule::new(METHODS[0])
        )
        .is_err()
    );
}
#[test]
fn ranks_mode_percentiles_and_tie_correction() {
    let samples = [3.0, 1.0, 3.0, 2.0];
    for (method, expected) in [
        (RankMethod::Average, [3.5, 1.0, 3.5, 2.0]),
        (RankMethod::Minimum, [3.0, 1.0, 3.0, 2.0]),
        (RankMethod::Maximum, [4.0, 1.0, 4.0, 2.0]),
        (RankMethod::Dense, [3.0, 1.0, 3.0, 2.0]),
        (RankMethod::Ordinal, [3.0, 1.0, 4.0, 2.0]),
    ] {
        assert_eq!(
            rank_slice(&samples, method)
                .unwrap()
                .into_iter()
                .map(|r| r.to_f64().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    }
    assert_eq!(
        scioxide_stats::statistic(&samples, Unweighted, Statistic::mode())
            .and_then(|value| if let StatisticValue::Order(result) = value {
                Ok(result)
            } else {
                Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected order result",
                ))
            })
            .unwrap(),
        OrderResult::Mode(ModeResult {
            value: 3.0,
            count: 2
        })
    );
    assert_eq!(
        scioxide_stats::statistic(&samples, Unweighted, Statistic::tie_correction())
            .and_then(|value| if let StatisticValue::Order(result) = value {
                Ok(result)
            } else {
                Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected order result",
                ))
            })
            .unwrap(),
        OrderResult::TieCorrection(0.9)
    );
    for (kind, expected) in [
        (PercentileKind::Strict, 50.0),
        (PercentileKind::Weak, 100.0),
        (PercentileKind::Mean, 75.0),
        (PercentileKind::Rank, 87.5),
    ] {
        assert_eq!(
            scioxide_stats::statistic(
                &samples,
                Unweighted,
                Statistic::percentile_of_score(3.0, kind)
            )
            .and_then(|value| if let StatisticValue::Order(result) = value {
                Ok(result)
            } else {
                Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected order result",
                ))
            })
            .unwrap(),
            OrderResult::PercentileOfScore {
                percentage: expected,
                kind
            }
        );
    }
    assert_eq!(
        scioxide_stats::statistic(&[-0.0, 0.0, 1.0], Unweighted, Statistic::mode())
            .and_then(|value| if let StatisticValue::Order(result) = value {
                Ok(result)
            } else {
                Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected order result",
                ))
            })
            .unwrap(),
        OrderResult::Mode(ModeResult {
            value: 0.0,
            count: 2
        })
    );
}

#[test]
fn scalar_queries_preserve_zero_ties_boundaries_and_recover_after_errors() {
    fn check<S: RealScalar>() {
        let values = [-0.0, 3.0, 0.0, 1.0].map(|x| S::try_from_f64_lossy(x).unwrap());
        let mut workspace = StatisticsWorkspace::default();
        workspace
            .try_reserve(
                &StatisticsWorkspace::scalar_requirements(values.len(), StatisticsOptions::new())
                    .unwrap(),
            )
            .unwrap();
        for method in METHODS {
            let OrderResult::Quantile(q) = workspace
                .evaluate(
                    &values,
                    Unweighted,
                    Statistic::quantile(p(0.0), QuantileRule::new(method)),
                )
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result",
                    )),
                })
                .unwrap()
            else {
                unreachable!()
            };
            assert_eq!(q.value(), S::ZERO);
            assert!(!q.value().is_sign_negative());
            let sorted = EmpiricalDistribution::new(&values, Unweighted).unwrap();
            let expected = sorted
                .quantile(p(0.75), QuantileRule::new(method))
                .unwrap()
                .value()
                - sorted
                    .quantile(p(0.25), QuantileRule::new(method))
                    .unwrap()
                    .value();
            let OrderResult::Iqr { value, .. } = workspace
                .evaluate(&values, Unweighted, Statistic::iqr(method))
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result",
                    )),
                })
                .unwrap()
            else {
                unreachable!()
            };
            assert!((value - expected).abs() <= S::EPSILON * S::try_from_f64_lossy(4.0).unwrap());
        }
        for (score, expected) in [
            (-1.0, [0.0, 0.0, 0.0, 0.0]),
            (0.0, [0.0, 50.0, 25.0, 37.5]),
            (0.5, [50.0, 50.0, 50.0, 50.0]),
            (3.0, [75.0, 100.0, 87.5, 100.0]),
            (4.0, [100.0; 4]),
        ] {
            for (kind, expected) in [
                PercentileKind::Strict,
                PercentileKind::Weak,
                PercentileKind::Mean,
                PercentileKind::Rank,
            ]
            .into_iter()
            .zip(expected)
            {
                assert_eq!(
                    workspace
                        .evaluate(
                            &values,
                            Unweighted,
                            Statistic::percentile_of_score(
                                S::try_from_f64_lossy(score).unwrap(),
                                kind
                            )
                        )
                        .and_then(|value| match value {
                            StatisticValue::Order(result) => Ok(result),
                            _ => Err(scioxide_core::SciOxideError::invalid_input(
                                scioxide_core::argument!("value"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "expected order result"
                            )),
                        })
                        .unwrap(),
                    OrderResult::PercentileOfScore {
                        percentage: expected,
                        kind
                    }
                );
            }
        }
        for statistic in [
            Statistic::quantile(p(0.5), QuantileRule::new(QuantileMethod::Linear)),
            Statistic::iqr(QuantileMethod::Linear),
            Statistic::percentile_of_score(S::ZERO, PercentileKind::Rank),
        ] {
            assert!(
                workspace
                    .evaluate(&[], Unweighted, statistic)
                    .and_then(|value| match value {
                        StatisticValue::Order(result) => Ok(result),
                        _ => Err(scioxide_core::SciOxideError::invalid_input(
                            scioxide_core::argument!("value"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "expected order result"
                        )),
                    })
                    .is_err()
            );
            assert!(
                workspace
                    .evaluate(&[S::NAN], Unweighted, statistic)
                    .and_then(|value| match value {
                        StatisticValue::Order(result) => Ok(result),
                        _ => Err(scioxide_core::SciOxideError::invalid_input(
                            scioxide_core::argument!("value"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "expected order result"
                        )),
                    })
                    .is_err()
            );
            assert!(
                StatisticsWorkspace::<S>::default()
                    .evaluate(&values, Unweighted, statistic)
                    .is_err()
            );
            workspace
                .evaluate(&values, Unweighted, statistic)
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result",
                    )),
                })
                .unwrap();
            let singleton = workspace
                .evaluate(&[S::ONE], Unweighted, statistic)
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result",
                    )),
                })
                .unwrap();
            assert!(
                matches!(singleton, OrderResult::Quantile(q) if q.value() == S::ONE)
                    || matches!(singleton, OrderResult::Iqr { value, .. } if value == S::ZERO)
                    || matches!(
                        singleton,
                        OrderResult::PercentileOfScore {
                            percentage: 0.0,
                            ..
                        }
                    )
            );
        }
        assert!(
            workspace
                .evaluate(
                    &values,
                    Unweighted,
                    Statistic::percentile_of_score(S::NAN, PercentileKind::Rank)
                )
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result"
                    )),
                })
                .is_err()
        );
        assert!(
            workspace
                .evaluate(
                    &[-S::MAX, S::MAX],
                    Unweighted,
                    Statistic::iqr(QuantileMethod::NearestRank)
                )
                .and_then(|value| match value {
                    StatisticValue::Order(result) => Ok(result),
                    _ => Err(scioxide_core::SciOxideError::invalid_input(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "expected order result"
                    )),
                })
                .is_err()
        );
        workspace
            .evaluate(&values, Unweighted, Statistic::mode())
            .and_then(|value| match value {
                StatisticValue::Order(result) => Ok(result),
                _ => Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected order result",
                )),
            })
            .unwrap();
        let mut ranks = [None; 4];
        rank_slice_into(
            &values,
            RankMethod::Ordinal,
            &mut ranks,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            ranks.map(|rank| rank.unwrap().bounds()),
            [(1, 1), (4, 4), (2, 2), (3, 3)]
        );
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn scalar_axis_queries_roll_back_then_reuse_scratch_across_tasks() {
    let data = [3.0, 1.0, 0.0, 2.0, 9.0, f64::NAN];
    let layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
    let view = BatchView::new(&data, layout, Axis::All).unwrap();
    let context = ExecutionContext::sequential();
    let options_axis = Axis::Index(1);
    let mut options = StatisticsOptions::new();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(
                view.with_axis(options_axis).unwrap(),
                None,
                options,
                context,
            )
            .unwrap()),
        )
        .unwrap();
    let sentinel = StatisticValue::Order(OrderResult::TieCorrection(-1.0));
    for (statistic, expected) in [
        (
            Statistic::quantile(p(0.5), QuantileRule::new(QuantileMethod::Linear)),
            [1.0, 5.5],
        ),
        (Statistic::iqr(QuantileMethod::Linear), [1.5, 3.5]),
        (
            Statistic::percentile_of_score(2.0, PercentileKind::Rank),
            [200.0 / 3.0, 50.0],
        ),
    ] {
        let mut output = [sentinel; 2];
        options = options.with_non_finite_policy(NonFinitePolicy::Reject);
        assert!(
            statistics_into(
                view.with_axis(options_axis).unwrap(),
                statistic,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output, [sentinel; 2]);
        options = options.with_non_finite_policy(NonFinitePolicy::Skip);
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            statistic,
            options,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        for (result, expected) in output.into_iter().zip(expected) {
            let value = result.scalar().unwrap();
            assert!((value - expected).abs() <= 2e-14);
        }
        let mut ranks = [None; 6];
        rank_axes_into(
            view.with_axis(options_axis).unwrap(),
            RankMethod::Ordinal,
            options,
            &mut ranks,
            &mut workspace,
            context,
        )
        .unwrap();
        assert!(ranks[5].is_none());
        assert_eq!(ranks[0].unwrap().bounds(), (3, 3));
    }
}

#[test]
fn scalar_axes_share_definitions_under_strides_broadcasts_and_masks() {
    let data = [3.0_f64, 8.0, -0.0, 7.0, 1.0, 6.0];
    let mask = [true, false, true, true, true, false];
    let context = ExecutionContext::sequential();
    for strides in [[1, 2], [0, 2]] {
        let layout = StridedLayout::new_broadcast(&[2, 3], &strides).unwrap();
        let view = BatchView::new(&data, layout, Axis::All)
            .unwrap()
            .with_validity(&mask)
            .unwrap();
        let options_axis = Axis::Index(1);
        let mut options = StatisticsOptions::new();
        options = options.with_missing_policy(MissingPolicy::Skip);
        let mut workspace = StatisticsWorkspace::default();
        workspace
            .try_reserve(
                &(StatisticsWorkspace::workspace_requirements(
                    view.with_axis(options_axis).unwrap(),
                    None,
                    options,
                    context,
                )
                .unwrap()),
            )
            .unwrap();
        for method in METHODS {
            for statistic in [
                Statistic::quantile(p(0.0), QuantileRule::new(method)),
                Statistic::quantile(p(0.9), QuantileRule::new(method)),
                Statistic::iqr(method),
                Statistic::percentile_of_score(1.0, PercentileKind::Rank),
            ] {
                let mut output = [StatisticValue::Order(OrderResult::TieCorrection(-1.0)); 2];
                statistics_into(
                    view.with_axis(options_axis).unwrap(),
                    statistic,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                )
                .unwrap();
                for (lane, actual) in output.into_iter().enumerate() {
                    let retained: Vec<f64> = (0..3)
                        .filter(|&position| mask[lane * 3 + position])
                        .map(|position| data[lane * strides[0] + position * strides[1]])
                        .collect();
                    let expected = scioxide_stats::statistic(&retained, Unweighted, statistic)
                        .and_then(|value| {
                            if let StatisticValue::Order(result) = value {
                                Ok(result)
                            } else {
                                Err(scioxide_core::SciOxideError::invalid_input(
                                    scioxide_core::argument!("value"),
                                    scioxide_core::InvalidInputReason::Inconsistent,
                                    "expected order result",
                                ))
                            }
                        })
                        .unwrap();
                    assert_eq!(actual.ordered().unwrap(), expected);
                }
            }
        }
    }
}
#[test]
fn expectile_balance_and_extreme_values() {
    let mut workspace = StatisticsWorkspace::<f64>::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(4, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    for (prob, expected) in [
        (0.0, 0.0),
        (0.25, 2.5),
        (0.5, 5.0),
        (0.75, 7.5),
        (1.0, 10.0),
    ] {
        let result = expectile(&mut workspace, &[(0.0, 1.0), (10.0, 1.0)], p(prob)).unwrap();
        assert_eq!(result.value, expected);
        assert!(result.relative_balance_residual < 1e-14);
    }
    let result = expectile(&mut workspace, &[(-f64::MAX, 1.0), (f64::MAX, 1.0)], p(0.5)).unwrap();
    assert_eq!(result.value, 0.0);
}
#[test]
fn borrowed_axes_missing_and_transactional_output() {
    let values = [1.0, 4.0, 2.0, 3.0, 9.0, f64::NAN];
    let layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
    let view = BatchView::new(&values, layout, Axis::All).unwrap();
    let options_axis = Axis::Index(1);
    let mut options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let requirements = StatisticsWorkspace::workspace_requirements(
        view.with_axis(options_axis).unwrap(),
        None,
        options,
        context,
    )
    .unwrap();
    let mut workspace = StatisticsWorkspace::default();
    workspace.try_reserve(&requirements).unwrap();
    let sentinel = StatisticValue::Order(OrderResult::TieCorrection(-1.0));
    let mut output = [sentinel; 2];
    assert!(
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            Statistic::mode(),
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, [sentinel; 2]);
    options = options.with_non_finite_policy(NonFinitePolicy::Skip);
    statistics_into(
        view.with_axis(options_axis).unwrap(),
        Statistic::quantile(p(0.5), QuantileRule::new(QuantileMethod::Linear)),
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(matches!(output[0].ordered().unwrap(), OrderResult::Quantile(q) if q.value()==2.0));
    assert!(matches!(output[1].ordered().unwrap(), OrderResult::Quantile(q) if q.value()==6.0));
    let mut ranks = [None; 6];
    rank_axes_into(
        view.with_axis(options_axis).unwrap(),
        RankMethod::Average,
        options,
        &mut ranks,
        &mut workspace,
        context,
    )
    .unwrap();
    assert_eq!(
        ranks.map(|r| r.map(|r| r.to_f64().unwrap())),
        [Some(1.0), Some(3.0), Some(2.0), Some(1.0), Some(2.0), None]
    );
}

#[test]
fn independent_rational_fixtures_and_permutations() {
    fn check<S: RealScalar>(tolerance: f64) {
        for row in include_str!("data/quantiles.tsv")
            .lines()
            .filter(|row| !row.starts_with('#'))
        {
            let fields: Vec<f64> = row.split('\t').map(|x| x.parse().unwrap()).collect();
            let n = fields[0] as usize;
            let expected = fields[3] / fields[4];
            let method = METHODS[fields[2] as usize - 1];
            let probability = p(fields[1] / 16.0);
            let mut values: Vec<S> = (0..n)
                .map(|i| S::try_from_f64_exact(((i / 2) * (i / 2)) as f64 - 7.0).unwrap())
                .collect();
            let sorted = EmpiricalDistribution::new(&values, Unweighted).unwrap();
            let actual = sorted
                .quantile(probability, QuantileRule::new(method))
                .unwrap()
                .value();
            let mut workspace = StatisticsWorkspace::default();
            workspace
                .try_reserve(
                    &StatisticsWorkspace::scalar_requirements(n, StatisticsOptions::new()).unwrap(),
                )
                .unwrap();
            assert!(
                (actual.try_to_f64_exact().unwrap() - expected).abs() <= tolerance,
                "{row}"
            );
            for rotation in 0..n {
                values.rotate_left(rotation);
                let OrderResult::Quantile(result) = workspace
                    .evaluate(
                        &values,
                        Unweighted,
                        Statistic::quantile(probability, QuantileRule::new(method)),
                    )
                    .and_then(|value| match value {
                        StatisticValue::Order(result) => Ok(result),
                        _ => Err(scioxide_core::SciOxideError::invalid_input(
                            scioxide_core::argument!("value"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "expected order result",
                        )),
                    })
                    .unwrap()
                else {
                    unreachable!()
                };
                assert!(
                    (result.value().into_f64() - expected).abs() <= tolerance,
                    "{row}"
                );
                assert_eq!(
                    quantile_in_place(&mut values, probability, QuantileRule::new(method))
                        .unwrap()
                        .value(),
                    actual
                );
                values.reverse();
            }
        }
    }
    check::<f32>(4e-6);
    check::<f64>(2e-14);
}

#[test]
fn weighted_axes_and_admission_contracts() {
    let values = [0.0, 10.0, f64::NAN, 4.0];
    let layout = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    let view = BatchView::new(&values, layout, Axis::All).unwrap();
    let options_axis = Axis::Index(1);
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = StatisticsWorkspace::default();
    workspace
        .try_reserve(
            &(StatisticsWorkspace::workspace_requirements(
                view.with_axis(options_axis).unwrap(),
                None,
                options,
                context,
            )
            .unwrap()),
        )
        .unwrap();
    let mut out = [StatisticValue::Order(OrderResult::TieCorrection(0.0)); 2];
    let weighted = view
        .with_weights(Weights::<_, Mass>::new(&[1.0, 3.0, 0.0, 1.0]))
        .unwrap();
    statistics_into(
        weighted.with_axis(options_axis).unwrap(),
        Statistic::expectile(p(0.5)),
        options,
        &mut out,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(matches!(out[0].ordered().unwrap(), OrderResult::Expectile(e) if e.value==7.5));
    assert!(matches!(out[1].ordered().unwrap(), OrderResult::Expectile(e) if e.value==4.0));
    let before = out;
    let invalid = view
        .with_weights(Weights::<_, Mass>::new(&[-1.0, 3.0, 0.0, 1.0]))
        .unwrap();
    assert!(
        statistics_into(
            invalid.with_axis(options_axis).unwrap(),
            Statistic::quantile(p(0.5), QuantileRule::inverse_ecdf()),
            options,
            &mut out,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(out, before);
    let frequency = view
        .with_weights(Weights::<_, Frequency>::new(&[1, 3, 0, 1]))
        .unwrap();
    statistics_into(
        frequency.with_axis(options_axis).unwrap(),
        Statistic::quantile(p(0.25), QuantileRule::new(QuantileMethod::Linear)),
        options,
        &mut out,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(matches!(out[0].ordered().unwrap(), OrderResult::Quantile(q) if q.value()==7.5));
    let mut missing = options;
    missing = missing.with_missing_policy(MissingPolicy::Skip);
    let masked = view.with_validity(&[true, true, false, true]).unwrap();
    statistics_into(
        masked.with_axis(options_axis).unwrap(),
        Statistic::mode(),
        missing,
        &mut out,
        &mut workspace,
        context,
    )
    .unwrap();
    assert!(matches!(out[1].ordered().unwrap(), OrderResult::Mode(m) if m.value==4.0));
}

#[test]
fn empty_strided_rank_zero_and_scratch_boundaries() {
    let context = ExecutionContext::sequential();
    let layout = StridedLayout::new(&[2, 3], &[1, 2]).unwrap();
    let view = BatchView::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], layout, Axis::All).unwrap();
    let options_axis = Axis::Index(1);
    let options = StatisticsOptions::new();
    let mut w = StatisticsWorkspace::default();
    let mut out = [StatisticValue::Order(OrderResult::TieCorrection(-1.0)); 2];
    assert!(
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            Statistic::mode(),
            options,
            &mut out,
            &mut w,
            context
        )
        .is_err()
    );
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            view.with_axis(options_axis).unwrap(),
            None,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    statistics_into(
        view.with_axis(options_axis).unwrap(),
        Statistic::mode(),
        options,
        &mut out,
        &mut w,
        context,
    )
    .unwrap();
    assert!(matches!(out[1].ordered().unwrap(), OrderResult::Mode(m) if m.value==4.0));
    let empty = BatchView::<f64>::new(
        &[],
        StridedLayout::new(&[0, 3], &[3, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let r = StatisticsWorkspace::workspace_requirements(
        empty.with_axis(options_axis).unwrap(),
        None,
        options,
        context,
    )
    .unwrap();
    w.try_reserve(&r).unwrap();
    statistics_into(
        empty.with_axis(options_axis).unwrap(),
        Statistic::mode(),
        options,
        &mut [],
        &mut w,
        context,
    )
    .unwrap();
    let empty_lane_axis = Axis::Index(0);
    let empty_lane = StatisticsOptions::new();
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            empty.with_axis(empty_lane_axis).unwrap(),
            None,
            empty_lane,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    assert!(
        statistics_into(
            empty.with_axis(empty_lane_axis).unwrap(),
            Statistic::mode(),
            empty_lane,
            &mut [StatisticValue::Order(OrderResult::TieCorrection(1.0)); 3],
            &mut w,
            context
        )
        .is_err()
    );
    let scalar = BatchView::new(&[2.0], StridedLayout::new(&[], &[]).unwrap(), Axis::All).unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new();
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            scalar.with_axis(options_axis).unwrap(),
            None,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    statistics_into(
        scalar.with_axis(options_axis).unwrap(),
        Statistic::iqr(QuantileMethod::Linear),
        options,
        &mut out[..1],
        &mut w,
        context,
    )
    .unwrap();
    assert!(matches!(
        out[0].ordered().unwrap(),
        OrderResult::Iqr { value: 0.0, .. }
    ));
}

#[cfg(feature = "std")]
#[test]
fn parallel_axes_match_sequential_and_failed_calls_recover() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let values: Vec<f64> = (0..256).map(|i| (i % 7) as f64).collect();
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[4, 8, 8], &[64, 8, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options_axis = Axis::Index(1);
    let mut options = StatisticsOptions::new();
    let mut w = StatisticsWorkspace::default();
    let seq = ExecutionContext::sequential();
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            view.with_axis(options_axis).unwrap(),
            None,
            options,
            seq,
        )
        .unwrap()),
    )
    .unwrap();
    let mut expected = vec![StatisticValue::Order(OrderResult::TieCorrection(0.0)); 32];
    statistics_into(
        view.with_axis(options_axis).unwrap(),
        Statistic::expectile(p(0.7)),
        options,
        &mut expected,
        &mut w,
        seq,
    )
    .unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);

    options = options.with_execution_policy(ExecutionPolicy::Parallel(workers));
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            view.with_axis(options_axis).unwrap(),
            None,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    let mut actual = vec![StatisticValue::Order(OrderResult::TieCorrection(0.0)); 32];
    for _ in 0..3 {
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            Statistic::expectile(p(0.7)),
            options,
            &mut actual,
            &mut w,
            context,
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
    let mut ranks = vec![None; 256];
    rank_axes_into(
        view.with_axis(options_axis).unwrap(),
        RankMethod::Ordinal,
        options,
        &mut ranks,
        &mut w,
        context,
    )
    .unwrap();
    options = options.with_execution_policy(ExecutionPolicy::Sequential);
    let mut serial = vec![None; 256];
    rank_axes_into(
        view.with_axis(options_axis).unwrap(),
        RankMethod::Ordinal,
        options,
        &mut serial,
        &mut w,
        seq,
    )
    .unwrap();
    assert_eq!(ranks, serial);
}

#[test]
fn expectile_weight_scale_invariance_and_f32_balance() {
    let mut workspace = StatisticsWorkspace::<f64>::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(3, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    let original = expectile(
        &mut workspace,
        &[(0.0, 1.0), (10.0, 2.0), (30.0, 1.0)],
        p(0.75),
    )
    .unwrap();
    // Root in [10,30]: .75(30-e)=.25(e+2(e-10)), hence e=55/3.
    assert!((original.value - 55.0 / 3.0).abs() < 2e-14);
    for scale in [1e-300, 1e300] {
        let result = expectile(
            &mut workspace,
            &[(30.0, scale), (10.0, 2.0 * scale), (0.0, scale)],
            p(0.75),
        )
        .unwrap();
        assert!((result.value - original.value).abs() < 2e-14);
        assert!(result.relative_balance_residual < 2e-14);
    }
    let mut narrow = StatisticsWorkspace::default();
    narrow
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(3, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    let result = expectile(
        &mut narrow,
        &[(0.0_f32, 1.0), (10.0, 2.0), (30.0, 1.0)],
        p(0.75),
    )
    .unwrap();
    assert!((f64::from(result.value) - 55.0 / 3.0).abs() < 4e-6);
    assert!(result.relative_balance_residual < 2e-7);
}

// Evaluate the defining equation independently in extended range at the
// published scalar, including f32 projection error rather than solver error.
fn expectile_residual<S: RealScalar>(pairs: &[(S, S)], result: ExpectileResult<S>) -> f64 {
    use scioxide_core::{Scaled, ScaledAccumulator};
    let right = Scaled::from_value(result.probability.get()).unwrap();
    let left = Scaled::ONE.sub(right).unwrap();
    let center = Scaled::from_value(result.value.into_f64()).unwrap();
    let mut balance = ScaledAccumulator::ZERO;
    let mut magnitude = ScaledAccumulator::ZERO;
    for &(x, w) in pairs {
        if w == S::ZERO {
            continue;
        }
        let term = Scaled::from_value(x.into_f64())
            .unwrap()
            .sub(center)
            .unwrap()
            .mul(Scaled::from_value(w.into_f64()).unwrap())
            .unwrap()
            .mul(if x > result.value { right } else { left })
            .unwrap();
        balance.add_scaled(term).unwrap();
        magnitude.add_scaled(term.abs()).unwrap();
    }
    let magnitude = magnitude.total().unwrap();
    if magnitude.is_zero() {
        0.0
    } else {
        balance
            .total()
            .unwrap()
            .abs()
            .div(magnitude)
            .unwrap()
            .round()
            .unwrap()
    }
}

#[test]
fn expectile_partitions_match_extended_equations_and_permutations() {
    fn check<S: RealScalar>() {
        let mut workspace = StatisticsWorkspace::<S>::default();
        workspace
            .try_reserve(
                &StatisticsWorkspace::scalar_requirements(9, StatisticsOptions::new()).unwrap(),
            )
            .unwrap();
        let mut pairs = [
            (-13.0, 1.0),
            (-0.0, 2.0),
            (1.0, 0.0),
            (1.0, 3.0),
            (2.0, 1.0),
            (10.0, 4.0),
            (10.0, 1.0),
            (30.0, 2.0),
            (42.0, 1.0),
        ]
        .map(|(x, w)| {
            (
                S::try_from_f64_lossy(x).unwrap(),
                S::try_from_f64_lossy(w).unwrap(),
            )
        });
        let exponent = if S::MANTISSA_DIGITS > 24 { 500 } else { 70 };
        for probability in [0.0, 0.125, 0.25, 0.5, 0.7, 0.875, 1.0] {
            let expected = expectile(&mut workspace, &pairs, p(probability)).unwrap();
            for _ in 0..pairs.len() {
                pairs.rotate_left(1);
                let actual = expectile(&mut workspace, &pairs, p(probability)).unwrap();
                let tolerance =
                    32.0 * S::EPSILON.into_f64() * expected.value.into_f64().abs().max(1.0);
                assert!((actual.value.into_f64() - expected.value.into_f64()).abs() <= tolerance);
                assert!(
                    (actual.relative_balance_residual - expectile_residual(&pairs, actual)).abs()
                        < 2e-15
                );
                for power in [-exponent, exponent] {
                    let scale = S::try_from_f64_lossy(2.0_f64.powi(power)).unwrap();
                    let scaled = pairs.map(|(x, w)| (x, w * scale));
                    let extended = expectile(&mut workspace, &scaled, p(probability)).unwrap();
                    assert!(
                        (actual.value.into_f64() - extended.value.into_f64()).abs() <= tolerance
                    );
                }
            }
        }
        let values = pairs.map(|(x, _)| x);
        let unit = pairs.map(|(x, _)| (x, S::ONE));
        for probability in [0.125, 0.5, 0.75] {
            let StatisticValue::Order(OrderResult::Expectile(actual)) = workspace
                .evaluate(&values, Unweighted, Statistic::expectile(p(probability)))
                .unwrap()
            else {
                unreachable!()
            };
            let expected = expectile(&mut workspace, &unit, p(probability)).unwrap();
            assert!(
                (actual.value.into_f64() - expected.value.into_f64()).abs()
                    <= 8.0 * S::EPSILON.into_f64() * 42.0
            );
            assert!(
                (actual.relative_balance_residual - expectile_residual(&unit, actual)).abs()
                    < 2e-15
            );
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn expectile_selection_handles_boundaries_offsets_and_invalid_support() {
    let mut workspace = StatisticsWorkspace::<f64>::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(5, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    for probability in [
        f64::from_bits(1),
        2.0_f64.powi(-65),
        2.0_f64.powi(-64),
        0.25,
        0.5,
        0.75,
        1.0 - f64::EPSILON,
    ] {
        let pairs = [(-2.0, 3.0), (10.0, 2.0)];
        let actual = expectile(&mut workspace, &pairs, p(probability)).unwrap();
        let expected = ((1.0 - probability) * -6.0 + probability * 20.0)
            / ((1.0 - probability) * 3.0 + probability * 2.0);
        assert!((actual.value - expected).abs() <= 16.0 * f64::EPSILON * expected.abs().max(1.0));
        assert!(
            (actual.relative_balance_residual - expectile_residual(&pairs, actual)).abs() < 2e-15
        );
    }
    for (pairs, expected) in [
        (
            [(1e16, 1.0), (1e16 + 2.0, 1.0), (1e16 + 4.0, 1.0)],
            1e16 + 2.0,
        ),
        ([(-1e16, 1.0), (1.0, 1.0), (1e16, 1.0)], 1.0 / 3.0),
        ([(-0.0, 1.0), (0.0, 1.0), (f64::NAN, 0.0)], 0.0),
    ] {
        let actual = expectile(&mut workspace, &pairs, p(0.5)).unwrap();
        assert!((actual.value - expected).abs() <= 4.0 * f64::EPSILON * expected.abs().max(1.0));
        assert!(
            (actual.relative_balance_residual - expectile_residual(&pairs, actual)).abs() < 2e-15
        );
        if actual.value == 0.0 {
            assert!(!actual.value.is_sign_negative());
        }
    }
    assert!(expectile(&mut workspace, &[], p(0.5)).is_err());
    for pairs in [
        [(f64::NAN, 1.0)],
        [(f64::INFINITY, 1.0)],
        [(1.0, -1.0)],
        [(1.0, f64::NAN)],
        [(1.0, 0.0)],
    ] {
        for probability in [0.0, 0.5, 1.0] {
            assert!(expectile(&mut workspace, &pairs, p(probability)).is_err());
        }
    }
    assert_eq!(
        expectile(&mut workspace, &[(2.0, 1.0)], p(0.75))
            .unwrap()
            .value,
        2.0
    );
}

#[test]
fn expectile_retains_weighted_product_cancellation_and_block_edges() {
    let mut workspace = StatisticsWorkspace::<f64>::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(1026, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    let e = 2.0_f64.powi(-27);
    let pairs = [(1.0 + e, 1.0 - e), (-1.0, 1.0)];
    let actual = expectile(&mut workspace, &pairs, p(0.5)).unwrap();
    let expected = -(e * e) / (2.0 - e);
    assert!((actual.value - expected).abs() <= 4.0 * f64::EPSILON * expected.abs());
    for n in [254, 256, 258, 512, 514, 1026] {
        let values: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { -2.0 } else { 2.0 })
            .collect();
        let weights: Vec<_> = (0..n).map(|i| ((i / 2) % 7 + 1) as f64 * 0.125).collect();
        for probability in [0.125, 0.5, 0.75, 0.875] {
            let expected = -2.0 + 4.0 * probability;
            let actual = workspace
                .evaluate(
                    &values,
                    Weights::<_, Mass>::new(&weights),
                    Statistic::expectile(p(probability)),
                )
                .unwrap();
            let StatisticValue::Order(OrderResult::Expectile(actual)) = actual else {
                unreachable!()
            };
            assert!((actual.value - expected).abs() <= 4.0 * f64::EPSILON);
            assert!(actual.relative_balance_residual < 4.0 * f64::EPSILON);
        }
    }
}

#[test]
fn expectiles_bracket_exact_roots_at_offsets_and_extreme_probabilities() {
    let offset = 2.0_f64.powi(52);
    let skewed = [1.610390560973588, 0.4438190513776543, 0.1262714302226598];
    let half = 1.0 - 2.0_f64.powi(-53);
    // Exact rational roots of the defining balance on the binary64 inputs,
    // each rounded once. Rows 3/4 and 5/6 are a translation and a weight
    // rescaling of each other; the last row crosses the native weight frame.
    let cases: [(&[f64], &[f64], f64, f64); 7] = [
        (
            &[999_999_999_978.0, 999_999_999_984.0, 999_999_999_974.0],
            &[1.0; 3],
            1e-12,
            999_999_999_974.0,
        ),
        (&[4.0, 10.0, 0.0], &[1.0; 3], 1e-12, 1.399_999_999_998_6e-11),
        (
            &[offset + 40.0, offset, offset + 47.0],
            &skewed,
            0.999_999_999_999,
            offset + 47.0,
        ),
        (
            &[40.0, 0.0, 47.0],
            &skewed,
            0.999_999_999_999,
            46.999_999_999_745_53,
        ),
        (
            &[1.0, 51.0],
            &[2.0_f64.powi(64), 0.25],
            half,
            1.006_102_770_657_878_7,
        ),
        (
            &[1.0, 51.0],
            &[2.0_f64.powi(65), 0.5],
            half,
            1.006_102_770_657_878_7,
        ),
        (
            &[
                4_503_599_627_370_525.0,
                4_503_599_627_370_505.0,
                4_503_599_627_370_477.0,
                4_503_599_627_370_439.0,
                4_503_599_627_370_479.0,
                4_503_599_627_370_455.0,
                4_503_599_627_370_510.0,
                4_503_599_627_370_451.0,
            ],
            &[
                0.398_724_734_773_552_86,
                1.655_945_266_658_416_7,
                1.879_503_630_550_571_7,
                0.167_504_721_598_194_85,
                1.548_690_617_613_453_6,
                1.579_267_985_526_003_4,
                0.674_134_044_194_002,
                1.235_997_948_802_457_5,
            ],
            0.5,
            4_503_599_627_370_479.0,
        ),
    ];
    let mut workspace = StatisticsWorkspace::<f64>::default();
    workspace
        .try_reserve(
            &StatisticsWorkspace::scalar_requirements(8, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    for (values, weights, probability, expected) in cases {
        let StatisticValue::Order(OrderResult::Expectile(actual)) = workspace
            .evaluate(
                values,
                Weights::<_, Mass>::new(weights),
                Statistic::expectile(p(probability)),
            )
            .unwrap()
        else {
            unreachable!()
        };
        let ulp = f64::from_bits(expected.to_bits() + 1) - expected;
        assert!(
            (actual.value - expected).abs() <= ulp,
            "{values:?}: {} != {expected}",
            actual.value
        );
        let pairs: Vec<_> = values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .collect();
        assert!(
            (actual.relative_balance_residual - expectile_residual(&pairs, actual)).abs() < 2e-15
        );
        if weights.iter().all(|&w| w == 1.0) {
            let StatisticValue::Order(OrderResult::Expectile(unweighted)) = workspace
                .evaluate(values, Unweighted, Statistic::expectile(p(probability)))
                .unwrap()
            else {
                unreachable!()
            };
            assert_eq!(unweighted.value, actual.value);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn rank_boundaries_beyond_f64_integer_precision() {
    let half = 1_u64 << 53;
    for (method, expected) in METHODS
        .into_iter()
        .zip([0.0, 5.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    {
        let mut values = [(0.0, half), (10.0, half)];
        assert_eq!(
            quantile_in_place(
                Weights::<_, Frequency>::new(&mut values[..]),
                p(0.5),
                QuantileRule::new(method)
            )
            .unwrap()
            .value(),
            expected
        );
        assert_eq!(
            quantile_in_place(
                Weights::<_, Frequency>::new(&mut values[..]),
                p(f64::from_bits(0.5_f64.to_bits() - 1)),
                QuantileRule::new(method)
            )
            .unwrap()
            .value(),
            0.0
        );
        assert_eq!(
            quantile_in_place(
                Weights::<_, Frequency>::new(&mut values[..]),
                p(f64::from_bits(0.5_f64.to_bits() + 1)),
                QuantileRule::new(method)
            )
            .unwrap()
            .value(),
            10.0
        );
    }
}

#[test]
fn executor_and_cancellation_failures_do_not_commit_and_workspace_recovers() {
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
            _plan: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
    }
    let view = BatchView::new(
        &[1.0, 2.0],
        StridedLayout::new(&[2], &[1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new();
    let seq = ExecutionContext::sequential();
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(
            view.with_axis(options_axis).unwrap(),
            None,
            options,
            seq,
        )
        .unwrap()),
    )
    .unwrap();
    let sentinel = StatisticValue::Order(OrderResult::TieCorrection(-1.0));
    let mut output = [sentinel];
    assert!(
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            Statistic::mode(),
            options,
            &mut output,
            &mut w,
            ExecutionContext::new(&Incomplete)
        )
        .is_err()
    );
    assert_eq!(output, [sentinel]);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(
        statistics_into(
            view.with_axis(options_axis).unwrap(),
            Statistic::mode(),
            options,
            &mut output,
            &mut w,
            seq.with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert_eq!(output, [sentinel]);
    statistics_into(
        view.with_axis(options_axis).unwrap(),
        Statistic::mode(),
        options,
        &mut output,
        &mut w,
        seq,
    )
    .unwrap();
    assert!(matches!(output[0].ordered().unwrap(), OrderResult::Mode(m) if m.value==1.0));
}
#[test]
fn mode_counts_heavily_tied_values_and_prefers_the_smallest() {
    let mode = |values: &[f64]| {
        scioxide_stats::statistic(values, Unweighted, Statistic::mode()).map(|value| match value {
            StatisticValue::Order(OrderResult::Mode(result)) => Some(result),
            _ => None,
        })
    };
    // 32 distinct floors, the low ones one occurrence more frequent.
    let values: Vec<f64> = (0..4099).map(|i| ((i * 7919) % 4099 % 32) as f64).collect();
    let mut counts = [0_usize; 32];
    for &v in &values {
        counts[v as usize] += 1;
    }
    let most = *counts.iter().max().unwrap();
    let expected = counts.iter().position(|&c| c == most).unwrap() as f64;
    assert_eq!(
        mode(&values).unwrap(),
        Some(ModeResult {
            value: expected,
            count: most
        })
    );
    // Equal counts select the smallest value, wherever it occurs.
    assert_eq!(
        mode(&[5.0, 2.0, 5.0, 2.0, 9.0]).unwrap().map(|m| m.value),
        Some(2.0)
    );
    assert_eq!(
        mode(&[1.0, f64::NAN]).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
}
