#![allow(clippy::unwrap_used)]
use scioxide_core::Complex;
use scioxide_stats::*;

#[test]
fn circular_nonzero_resultant_survives_squared_underflow() {
    fn check<S: scioxide_core::RealScalar>(tiny: S) {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let angles = [S::ZERO, S::PI, -S::PI, S::ZERO];
        let weights = [scalar(2.0), S::ONE, S::ONE, tiny];
        let result = statistic(
            &angles,
            Weights::<_, Mass>::new(&weights),
            Statistic::circular(CircularOptions::default()),
        )
        .unwrap();
        let StatisticValue::Circular(result) = result else {
            unreachable!()
        };
        let expected = tiny.into_f64() / 4.0;
        let tolerance = 8.0 * S::EPSILON.into_f64();
        assert!((result.resultant_length.into_f64() / expected - 1.0).abs() < tolerance);
        assert!(result.mean.is_none());
        let expected_sd = (-2.0 * expected.ln()).sqrt();
        assert!(
            (result.standard_deviation.unwrap().into_f64() / expected_sd - 1.0).abs() < tolerance
        );
    }
    for tiny in [1e-200_f64, 1e-300] {
        check(tiny);
    }
    for tiny in [1e-25_f32, 1e-35] {
        check(tiny);
    }
}

#[test]
fn stouffer_preserves_positive_weights_at_probability_endpoints() {
    let weights = [f64::from_bits(1), f64::MAX];
    for (endpoint, expected) in [(0.0, 0.0), (1.0, 1.0)] {
        let result = combine_pvalues(
            &[endpoint, 0.5],
            PValueCombination::Stouffer,
            Some(&weights),
        )
        .unwrap();
        assert_eq!(result.p_value(), expected);
        assert!(result.statistic().is_infinite());
    }
    assert!(combine_pvalues(&[0.0, 1.0], PValueCombination::Stouffer, Some(&weights)).is_err());
    let result =
        combine_pvalues(&[0.0, 0.5], PValueCombination::Stouffer, Some(&[0.0, 1.0])).unwrap();
    assert!((result.p_value() - 0.5).abs() < 1e-15);
}
#[test]
fn stouffer_retains_finite_scores_after_cancellation() {
    for scale in [1e100_f64, 1e300] {
        let result = combine_pvalues(
            &[0.25, 0.25, 0.75],
            PValueCombination::Stouffer,
            Some(&[scale, 1.0, scale]),
        )
        .unwrap();
        let expected = 0.6744897501960817 / scale / 2.0_f64.sqrt();
        assert!((result.statistic() / expected - 1.0).abs() < 1e-13);
    }
}

fn check<S: scioxide_core::RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let data = [s(1.), s(2.), s(3.), s(4.)];
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(&StatisticsWorkspace::scalar_requirements(8, StatisticsOptions::new()).unwrap())
        .unwrap();
    let mut test = |stat, expected: f64| {
        let v = w
            .evaluate(&data, Unweighted, stat)
            .and_then(StatisticValue::scalar)
            .unwrap()
            .try_to_f64_exact()
            .unwrap();
        assert!((v - expected).abs() < 2e-6, "{v} != {expected}");
    };
    test(Statistic::mean(), 2.5);
    test(Statistic::geometric_mean(), 24f64.powf(0.25));
    test(Statistic::harmonic_mean(), 1.92);
    test(Statistic::power_mean(s(2.)), 7.5f64.sqrt());
    test(
        Statistic::l_moment(1, scioxide_stats::LMomentNormalization::Raw),
        2.5,
    );
    test(
        Statistic::l_moment(2, scioxide_stats::LMomentNormalization::Raw),
        5. / 6.,
    );
    test(
        Statistic::l_moment(3, scioxide_stats::LMomentNormalization::Raw),
        0.,
    );
    test(
        Statistic::l_moment(4, scioxide_stats::LMomentNormalization::Raw),
        0.,
    );
    test(
        Statistic::median_absolute_deviation(QuantileMethod::Linear, s(1.)),
        1.,
    );
    test(Statistic::trimmed_mean(s(0.25), TrimSide::Both), 2.5);
    let mut output = [s(9.); 4];
    w.obrien_transform_into(&data, &mut output).unwrap();
    let average = output
        .iter()
        .map(|x| x.try_to_f64_exact().unwrap())
        .sum::<f64>()
        / 4.;
    assert!((average - 5. / 3.).abs() < 2e-6);
    w.geometric_zscore_into(&[s(1.), s(2.), s(4.), s(8.)], 0, &mut output)
        .unwrap();
    assert!(
        output
            .iter()
            .map(|x| x.try_to_f64_exact().unwrap())
            .sum::<f64>()
            .abs()
            < 2e-6
    );
}
#[test]
fn exact_real_samples() {
    check::<f32>();
    check::<f64>();
}
#[test]
fn scale_and_weight_edges() {
    fn mean<'a, W: BatchWeights<'a, f64>>(x: &[f64], weights: W, p: f64) -> f64
    where
        W::Model: EmpiricalWeights,
    {
        scioxide_stats::statistic(x, weights, Statistic::power_mean(p))
            .and_then(StatisticValue::scalar)
            .unwrap()
    }
    assert_eq!(mean(&[f64::MAX, f64::MAX], Unweighted, 1.), f64::MAX);
    assert_eq!(mean(&[f64::MAX, f64::MAX], Unweighted, -1.), f64::MAX);
    assert_eq!(mean(&[0., 4.], Unweighted, -2.), 0.);
    assert!((mean(&[1., 4.], Unweighted, 1e-14) - 2.).abs() < 1e-12);
    assert!((mean(&[1., 4.], Weights::<_, Frequency>::new(&[2, 2]), 0.) - 2.).abs() < 1e-14);
    assert_eq!(
        mean(&[f64::NAN, 3.], Weights::<_, Mass>::new(&[0., 1.]), 1.),
        3.
    );
    assert!(
        scioxide_stats::statistic(&[-1., 2.], Unweighted, Statistic::geometric_mean())
            .and_then(StatisticValue::scalar)
            .is_err()
    );
    assert!(
        scioxide_stats::statistic(&[1., 2.], Unweighted, Statistic::variance(2))
            .and_then(StatisticValue::scalar)
            .is_err()
    );
}
#[test]
fn real_complex_standardization_and_transaction() {
    let z = standardize(&[1., 2., 3.], &[1., 2., 3.], 0).unwrap();
    assert!((z[0] + (1.5f64).sqrt()).abs() < 1e-14);
    let x = [Complex::new(1f64, 2.), Complex::new(3., 4.)];
    let z = standardize(&x, &x, 0).unwrap();
    assert!((z[0].re + 1. / 2f64.sqrt()).abs() < 1e-14);
    assert!((z[0].im + 1. / 2f64.sqrt()).abs() < 1e-14);
    let mut w = StandardizationWorkspace::default();
    w.try_reserve(
        &StandardizationWorkspace::workspace_requirements(2, StatisticsOptions::new()).unwrap(),
    )
    .unwrap();
    let mut out = [99.; 2];
    assert!(
        w.standardize_into(&[1., f64::NAN], &[1., 2.], 0, &mut out)
            .is_err()
    );
    assert_eq!(out, [99.; 2]);
}
#[test]
fn circular_wraparound() {
    let r = scioxide_stats::statistic(
        &[350f64, 10.],
        Unweighted,
        Statistic::circular(CircularOptions::new().with_low(0.).with_high(360.)),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    let mean = r.mean.unwrap();
    assert!(mean.abs() < 1e-12 || (mean - 360.).abs() < 1e-12);
    assert!((r.resultant_length - 10f64.to_radians().cos()).abs() < 1e-14);
    let r = scioxide_stats::statistic(
        &[0f32, core::f32::consts::PI],
        Unweighted,
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    assert!(r.mean.is_none());
}

fn circular_precision<S: scioxide_core::RealScalar>() {
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    let close = |a: S, b: f64| {
        assert!((a.into_f64() - b).abs() <= 128.0 * S::EPSILON.into_f64() * b.abs().max(1.0));
    };
    let angles = [S::ZERO, S::PI / scalar(2.0)];
    // Equal, replicated and relative masses have independent closed-form resultants.
    for scale in [S::ONE, S::MAX / scalar(4.0), S::MIN_SUBNORMAL] {
        let weights = [scale, scale * scalar(3.0)];
        let result = scioxide_stats::statistic(
            &angles,
            Weights::<_, Mass>::new(&weights),
            Statistic::circular(CircularOptions::default()),
        )
        .and_then(|value| match value {
            StatisticValue::Circular(value) => Ok(value),
            _ => Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("value"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected circular result",
            )),
        })
        .unwrap();
        close(result.mean.unwrap(), 3.0_f64.atan2(1.0));
        close(result.resultant_length, 10.0_f64.sqrt() / 4.0);
        close(result.variance, 1.0 - 10.0_f64.sqrt() / 4.0);
        close(
            result.standard_deviation.unwrap(),
            (-2.0 * (10.0_f64.sqrt() / 4.0).ln()).sqrt(),
        );
    }
    let frequencies = [1, 3];
    let replicated = [angles[0], angles[1], angles[1], angles[1]];
    let weighted = scioxide_stats::statistic(
        &angles,
        Weights::<_, Frequency>::new(&frequencies),
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    let expanded = scioxide_stats::statistic(
        &replicated,
        Unweighted,
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    close(weighted.mean.unwrap(), expanded.mean.unwrap().into_f64());
    close(
        weighted.resultant_length,
        expanded.resultant_length.into_f64(),
    );
    let half = [S::ZERO, S::PI];
    for (offset, defined) in [(8.0, false), (32.0, true)] {
        let masses = [S::ONE, S::ONE - scalar(offset) * S::EPSILON];
        let result = scioxide_stats::statistic(
            &half,
            Weights::<_, Mass>::new(&masses),
            Statistic::circular(CircularOptions::default()),
        )
        .and_then(|value| match value {
            StatisticValue::Circular(value) => Ok(value),
            _ => Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("value"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected circular result",
            )),
        })
        .unwrap();
        assert_eq!(result.mean.is_some(), defined);
    }
    // Default radians must not reduce huge coordinates with a rounded TAU.
    let huge = [S::MAX];
    let result = scioxide_stats::statistic(
        &huge,
        Unweighted,
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    let (sin, cos) = S::MAX.sin_cos();
    let expected = sin.atan2(cos);
    close(
        result.mean.unwrap(),
        (if expected < S::ZERO {
            expected + S::TAU
        } else {
            expected
        })
        .into_f64(),
    );
    let period = CircularOptions::new()
        .with_low(scalar(-3.0))
        .with_high(scalar(5.0));
    let result = scioxide_stats::statistic(
        &[scalar(2.0).scalbn(100), S::ZERO],
        Unweighted,
        Statistic::circular(period),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    close(result.mean.unwrap(), 0.0);
    let tiny_period = CircularOptions::new()
        .with_low(S::ZERO)
        .with_high(S::MIN_SUBNORMAL * scalar(8.0));
    let tiny = scioxide_stats::statistic(
        &[S::MIN_SUBNORMAL * scalar(2.0)],
        Unweighted,
        Statistic::circular(tiny_period),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    assert_eq!(tiny.mean, Some(S::MIN_SUBNORMAL * scalar(2.0)));
}

#[test]
fn circular_native_and_extended_precision() {
    circular_precision::<f32>();
    circular_precision::<f64>();
    // A native weight times a tiny phase can underflow before normalization;
    // restarting the whole reduction must recover the representable direction.
    let x = [1e-250_f64, 2e-250];
    let w = [1e-77, 1e-77];
    let result = scioxide_stats::statistic(
        &x,
        Weights::<_, Mass>::new(&w),
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    assert!((result.mean.unwrap() / 1.5e-250 - 1.0).abs() < 1e-14);
}

#[test]
fn circular_admission_and_period_errors() {
    let options = CircularOptions::<f64>::default();
    assert!(
        scioxide_stats::statistic(&[] as &[f64], Unweighted, Statistic::circular(options)).is_err()
    );
    assert!(
        scioxide_stats::statistic(
            &[1.0],
            Weights::<_, Mass>::new(&[]),
            Statistic::circular(options)
        )
        .and_then(|value| match value {
            StatisticValue::Circular(value) => Ok(value),
            _ => Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("value"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected circular result"
            )),
        })
        .is_err()
    );
    for masses in [
        [0.0, 0.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [f64::INFINITY, 1.0],
    ] {
        assert!(
            scioxide_stats::statistic(
                &[0.0, 1.0],
                Weights::<_, Mass>::new(&masses),
                Statistic::circular(options)
            )
            .and_then(|value| match value {
                StatisticValue::Circular(value) => Ok(value),
                _ => Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected circular result"
                )),
            })
            .is_err()
        );
    }
    let result = scioxide_stats::statistic(
        &[f64::NAN, 0.0],
        Weights::<_, Mass>::new(&[0.0, 1.0]),
        Statistic::circular(options),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    assert_eq!(result.mean, Some(0.0));
    assert!(
        scioxide_stats::statistic(&[f64::NAN], Unweighted, Statistic::circular(options))
            .and_then(|value| match value {
                StatisticValue::Circular(value) => Ok(value),
                _ => Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected circular result"
                )),
            })
            .is_err()
    );
    for (low, high) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (f64::NEG_INFINITY, 1.0),
        (-f64::MAX, f64::MAX),
    ] {
        assert!(
            scioxide_stats::statistic(
                &[0.0],
                Unweighted,
                Statistic::circular(CircularOptions::new().with_low(low).with_high(high))
            )
            .and_then(|value| match value {
                StatisticValue::Circular(value) => Ok(value),
                _ => Err(scioxide_core::SciOxideError::invalid_input(
                    scioxide_core::argument!("value"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected circular result"
                )),
            })
            .is_err()
        );
    }
}
#[test]
fn independent_enumerated_lmoment_oracle() {
    for line in include_str!("data/lmoments.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let x: Vec<f64> = fields[0].split(',').map(|v| v.parse().unwrap()).collect();
        let r = fields[1].parse().unwrap();
        let expected = fields[2].parse::<f64>().unwrap() / fields[3].parse::<f64>().unwrap();
        let stat = Statistic::l_moment(r, scioxide_stats::LMomentNormalization::Raw);
        let actual = scioxide_stats::statistic(&x, Unweighted, stat)
            .and_then(StatisticValue::scalar)
            .unwrap();
        assert!(
            (actual - expected).abs() < 2e-9,
            "{line}: {actual} != {expected}"
        );
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let stat = Statistic::l_moment(r, scioxide_stats::LMomentNormalization::Raw);
        let actual = scioxide_stats::statistic(&x32, Unweighted, stat)
            .and_then(StatisticValue::scalar)
            .unwrap() as f64;
        assert!(
            (actual - expected).abs() < 2e-6 * expected.abs().max(1.),
            "f32 {line}: {actual} != {expected}"
        );
    }
}
#[test]
fn mad_retains_small_deviations_beside_extreme_outlier() {
    let value = scioxide_stats::statistic(
        &[1e-300, 2e-300, f64::MAX],
        Unweighted,
        Statistic::median_absolute_deviation(QuantileMethod::Linear, 1.),
    )
    .and_then(StatisticValue::scalar)
    .unwrap();
    assert!((value / 1e-300 - 1.).abs() < 1e-14);
    let mean = scioxide_stats::statistic(
        &[1e-300_f64, 1e300],
        Unweighted,
        Statistic::power_mean(-0.001),
    )
    .and_then(StatisticValue::scalar)
    .unwrap();
    assert!(mean.is_finite() && mean > 0.);
}
#[test]
fn trimming_transforms_and_empty_identities() {
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(&StatisticsWorkspace::scalar_requirements(4, StatisticsOptions::new()).unwrap())
        .unwrap();
    let data = [4.0_f64, 1., 3., 2.];
    let mut out = [99.; 4];
    for (side, expected) in [
        (TrimSide::Lower, &[2., 3., 4.][..]),
        (TrimSide::Upper, &[1., 2., 3.][..]),
        (TrimSide::Both, &[2., 3.][..]),
    ] {
        let n = w.trim_into(&data, 0.25, side, &mut out).unwrap();
        assert_eq!(&out[..n], expected);
    }
    out.fill(99.);
    assert!(w.trim_into(&data, 0.5, TrimSide::Both, &mut out).is_err());
    assert_eq!(out, [99.; 4]);
    assert_eq!(
        scioxide_stats::statistic(&[] as &[f64], Unweighted, Statistic::sum())
            .and_then(StatisticValue::scalar)
            .unwrap(),
        0.
    );
    assert!(
        scioxide_stats::statistic(&[] as &[f64], Unweighted, Statistic::mean())
            .and_then(StatisticValue::scalar)
            .is_err()
    );
    let z = standardize(&[0., 2., 4.], &[1., 3.], 0).unwrap();
    assert_eq!(z, vec![-2., 0., 2.]);
    assert!(
        (w.evaluate(
            &[1., 1., 4., 4.],
            Unweighted,
            Statistic::geometric_standard_deviation(0)
        )
        .and_then(StatisticValue::scalar)
        .unwrap()
            - 2.)
            .abs()
            < 1e-14
    );
    let r = scioxide_stats::statistic(
        &[0., core::f64::consts::FRAC_PI_2],
        Unweighted,
        Statistic::circular(CircularOptions::default()),
    )
    .and_then(|value| match value {
        StatisticValue::Circular(value) => Ok(value),
        _ => Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "expected circular result",
        )),
    })
    .unwrap();
    assert!((r.standard_deviation.unwrap() - 2f64.ln().sqrt()).abs() < 1e-14);
}
#[test]
fn standardization_retains_unrepresentable_midpoint() {
    let x = [1e16_f64, 1e16 + 2.];
    let z = standardize(&x, &x, 0).unwrap();
    assert_eq!(z, vec![-1., 1.]);
    let mut w = StatisticsWorkspace::default();
    w.try_reserve(&StatisticsWorkspace::scalar_requirements(2, StatisticsOptions::new()).unwrap())
        .unwrap();
    let mut out = [0.; 2];
    w.geometric_zscore_into(&x, 0, &mut out).unwrap();
    assert!((out[0] + 1.).abs() < 1e-14);
    assert!((out[1] - 1.).abs() < 1e-14);
    let x = [
        scioxide_core::Complex::new(1e16, 1e16),
        scioxide_core::Complex::new(1e16 + 2., 1e16 + 2.),
    ];
    let z = standardize(&x, &x, 0).unwrap();
    assert!((z[0].re + 0.5f64.sqrt()).abs() < 1e-14);
}
#[test]
fn complex_f32_standardization() {
    let x = [
        scioxide_core::Complex::new(1f32, 2.),
        scioxide_core::Complex::new(3f32, 4.),
    ];
    let z = standardize(&x, &x, 0).unwrap();
    assert!((z[0].re + 0.5f32.sqrt()).abs() < 2e-6);
    assert!((z[0].im + 0.5f32.sqrt()).abs() < 2e-6);
}
#[test]
fn selected_mad_equals_the_sorted_deviation_quantile() {
    // Tied pseudo-random samples: selection must read the same order statistics
    // as a full sort of |x - median| for every interpolation rule.
    for n in [1_usize, 2, 3, 8, 33, 256] {
        let data: Vec<f64> = (0..n)
            .map(|i| ((i * 7919 + 13) % 61) as f64 * 0.125 - 3.0)
            .collect();
        for method in [
            QuantileMethod::Linear,
            QuantileMethod::NearestRank,
            QuantileMethod::AveragedInverseCdf,
            QuantileMethod::ClosestObservation,
            QuantileMethod::Hazen,
            QuantileMethod::MedianUnbiased,
        ] {
            // Independent reference: a fully sorted copy and the scalar quantile.
            let quantile = |values: &mut Vec<f64>| {
                values.sort_by(f64::total_cmp);
                scioxide_stats::statistic(
                    values,
                    Unweighted,
                    Statistic::quantile(
                        QuantileProbability::new(0.5).unwrap(),
                        QuantileRule::new(method),
                    ),
                )
                .and_then(StatisticValue::scalar)
                .unwrap()
            };
            let center = quantile(&mut data.clone());
            let expected = quantile(&mut data.iter().map(|x| (x - center).abs()).collect()) / 2.0;
            let actual = scioxide_stats::statistic(
                &data,
                Unweighted,
                Statistic::median_absolute_deviation(method, 2.0),
            )
            .and_then(StatisticValue::scalar)
            .unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits(), "n={n} {method:?}");
        }
    }
}
