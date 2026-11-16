//! Independent closed-form standardization and query-local dispersion checks.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_stats::*;

fn close<S: RealScalar>(actual: S, expected: f64) {
    assert!(
        (actual.into_f64() - expected).abs()
            <= 64.0 * S::EPSILON.into_f64() * expected.abs().max(1.0),
        "{} != {expected}",
        actual.into_f64()
    );
}

fn scalar<T: NumericScalar>(re: T::Real, im: T::Real) -> T {
    T::from_components(re, if T::IS_REAL { T::Real::ZERO } else { im })
}

fn precision<T: NumericScalar>() {
    let number = |x| T::Real::try_from_f64_lossy(x).unwrap();
    let factor = if T::IS_REAL { 1.0 } else { 0.5_f64.sqrt() };
    // Include both native guard boundaries, wider ranges and squared underflow.
    for step in [
        number(1.0),
        number(2.0_f64.powi(-65)),
        number(2.0_f64.powi(-64)),
        number(2.0_f64.powi(63)),
        number(2.0_f64.powi(64)),
        T::Real::MAX / number(2.0),
        T::Real::MIN_SUBNORMAL,
    ] {
        let reference = [T::ZERO, scalar::<T>(step + step, -(step + step))];
        for ddof in [0, 1] {
            let expected = factor / if ddof == 0 { 1.0 } else { 2.0_f64.sqrt() };
            let output = standardize(&reference, &reference, ddof).unwrap();
            for (&x, sign) in output.iter().zip([-1.0, 1.0]) {
                close(x.real_component(), sign * expected);
                close(
                    x.imaginary_component(),
                    if T::IS_REAL { 0.0 } else { -sign * expected },
                );
            }
        }
    }
    let reference = [
        scalar::<T>(number(1.0), number(-1.0)),
        scalar::<T>(number(3.0), number(-3.0)),
    ];
    let scores = [T::ZERO, scalar::<T>(number(4.0), number(-4.0))];
    let output = standardize(&scores, &reference, 0).unwrap();
    close(output[0].real_component(), -2.0 * factor);
    close(output[1].real_component(), 2.0 * factor);
    close(
        output[0].imaginary_component(),
        if T::IS_REAL { 0.0 } else { 2.0 * factor },
    );
    assert!(standardize::<T>(&[], &reference, 0).unwrap().is_empty());

    // A scalar mean cannot represent this midpoint in the source precision.
    let origin = if T::Real::EPSILON.into_f64() > 1e-10 {
        number(16_777_216.0)
    } else {
        number(1e16)
    };
    let translated = [
        scalar::<T>(origin, -origin),
        scalar::<T>(origin + number(2.0), -origin - number(2.0)),
    ];
    let output = standardize(&translated, &translated, 0).unwrap();
    close(output[0].real_component(), -factor);
    close(output[1].real_component(), factor);

    let cancellation = [
        scalar::<T>(-origin, origin),
        scalar::<T>(number(1.0), number(-1.0)),
        scalar::<T>(origin, -origin),
    ];
    let StatisticValue::Location(mean) =
        statistic(&cancellation, Unweighted, Statistic::mean()).unwrap()
    else {
        unreachable!()
    };
    close(mean.real_component(), 1.0 / 3.0);
    close(
        mean.imaginary_component(),
        if T::IS_REAL { 0.0 } else { -1.0 / 3.0 },
    );

    let huge = [T::ZERO, scalar::<T>(T::Real::MAX, T::Real::ZERO)];
    assert!(statistic(&huge, Unweighted, Statistic::variance(0)).is_err());
    let StatisticValue::Dispersion(sd) =
        statistic(&huge, Unweighted, Statistic::standard_deviation(0)).unwrap()
    else {
        unreachable!()
    };
    close(sd / T::Real::MAX, 0.5);
    let tiny = [
        T::ZERO,
        scalar::<T>(
            T::Real::MIN_SUBNORMAL + T::Real::MIN_SUBNORMAL,
            T::Real::ZERO,
        ),
    ];
    let StatisticValue::Dispersion(variance) =
        statistic(&tiny, Unweighted, Statistic::variance(0)).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(variance, T::Real::ZERO);
    let StatisticValue::Dispersion(sd) =
        statistic(&tiny, Unweighted, Statistic::standard_deviation(0)).unwrap()
    else {
        unreachable!()
    };
    assert_eq!(sd, T::Real::MIN_SUBNORMAL);
}

#[test]
fn all_scalar_precisions_and_ranges() {
    precision::<f32>();
    precision::<f64>();
    precision::<Complex<f32>>();
    precision::<Complex<f64>>();
}

fn rollback<T: NumericScalar>() {
    let number = |x| T::Real::try_from_f64_lossy(x).unwrap();
    let good = [T::ZERO, T::from_real(number(2.0))];
    let sentinel = T::from_real(number(99.0));
    let mut workspace = StandardizationWorkspace::default();
    let mut output = [sentinel; 2];
    assert!(
        workspace
            .standardize_into(&good, &good, 0, &mut output)
            .is_err()
    );
    workspace
        .try_reserve(
            &StandardizationWorkspace::workspace_requirements(2, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    for (scores, reference, ddof) in [
        (good, good, 2),
        (good, [T::ZERO; 2], 0),
        ([T::ZERO, T::from_real(T::Real::NAN)], good, 0),
        (good, [T::ZERO, T::from_real(T::Real::INFINITY)], 0),
        // Native reference, then wide score admission, then final overflow.
        (
            [T::ZERO, T::from_real(T::Real::MAX)],
            [T::ZERO, T::from_real(T::Real::MIN_SUBNORMAL)],
            0,
        ),
    ] {
        assert!(
            workspace
                .standardize_into(&scores, &reference, ddof, &mut output)
                .is_err()
        );
        assert_eq!(output, [sentinel; 2]);
    }
    if !T::IS_REAL {
        let invalid = [T::ZERO, T::from_components(T::Real::ZERO, T::Real::NAN)];
        assert!(
            workspace
                .standardize_into(&invalid, &good, 0, &mut output)
                .is_err()
        );
        assert!(
            workspace
                .standardize_into(&good, &invalid, 0, &mut output)
                .is_err()
        );
    }
    assert!(
        workspace
            .standardize_into(&good, &[], 0, &mut output)
            .is_err()
    );
    assert!(
        workspace
            .standardize_into(&good, &good, 0, &mut output[..1])
            .is_err()
    );
    assert_eq!(output, [sentinel; 2]);
    // An extreme score can be valid even when subtracting the mean overflows.
    let scores = [T::from_real(-T::Real::MAX), T::ZERO];
    let reference = [T::ZERO, T::from_real(T::Real::MAX)];
    workspace
        .standardize_into(&scores, &reference, 0, &mut output)
        .unwrap();
    close(output[0].real_component(), -3.0);
    close(output[1].real_component(), -1.0);
    workspace
        .standardize_into(&good, &good, 0, &mut output)
        .unwrap();
    close(output[0].real_component(), -1.0);
    close(output[1].real_component(), 1.0);
}

#[test]
fn failures_roll_back_and_workspace_remains_reusable() {
    rollback::<f32>();
    rollback::<f64>();
    rollback::<Complex<f32>>();
    rollback::<Complex<f64>>();
}

#[test]
fn native_f32_projection_overflow_is_numerical_and_transactional() {
    let mut workspace = StandardizationWorkspace::default();
    workspace
        .try_reserve(
            &StandardizationWorkspace::workspace_requirements(2, StatisticsOptions::new()).unwrap(),
        )
        .unwrap();
    let mut output = [99.0_f32; 2];
    let error = workspace
        .standardize_into(
            &[0.0, 2.0_f32.powi(64)],
            &[0.0, 2.0_f32.powi(-64)],
            0,
            &mut output,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    assert_eq!(output, [99.0; 2]);
}

#[test]
fn normalized_center_retains_non_dyadic_offsets_and_cancellation() {
    for origin in [2.0_f64.powi(-64), 1.0, 1e16, 2.0_f64.powi(64)] {
        let next = origin.next_up();
        let reference = [origin, origin, next];
        let output = standardize(&reference, &reference, 0).unwrap();
        for (actual, expected) in
            output
                .into_iter()
                .zip([-0.5_f64.sqrt(), -0.5_f64.sqrt(), 2.0_f64.sqrt()])
        {
            close(actual, expected);
        }
    }
    let output = standardize(&[0.0], &[1e16, 1.0, -1e16], 0).unwrap();
    let expected = -1.0 / (1e16 * 6.0_f64.sqrt());
    assert!((output[0] - expected).abs() <= 64.0 * f64::EPSILON * expected.abs());
}

#[test]
fn wide_scores_use_scaled_subtraction_and_allow_signed_underflow() {
    let output = standardize(&[-f64::MAX, 1.0], &[0.0, 2.0], 0).unwrap();
    assert_eq!(output, [-f64::MAX, 0.0]);
    let output = standardize(&[-f64::from_bits(1), 0.0], &[-f64::MAX, f64::MAX], 0).unwrap();
    assert_eq!(output[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(output[1], 0.0);
    assert!(matches!(
        scioxide_stats::statistic(&[] as &[f64], Unweighted, Statistic::sum()).unwrap(),
        StatisticValue::Location(0.0)
    ));
    assert!(scioxide_stats::statistic(&[] as &[f64], Unweighted, Statistic::mean()).is_err());
    assert!(statistic(&[1.0, 2.0], Unweighted, Statistic::variance(2)).is_err());
}

fn independent_reference<T: NumericScalar>() {
    // An odd count gives non-dyadic centers. Raw compensated scaled sums are
    // independent of the production shifted-location and native kernels.
    for scale in [
        T::Real::ONE,
        T::Real::MAX / T::Real::try_from_f64_lossy(1024.0).unwrap(),
        T::Real::MIN_SUBNORMAL * T::Real::try_from_f64_lossy(16.0).unwrap(),
    ] {
        let values: Vec<_> = (0..257)
            .map(|i| {
                scalar::<T>(
                    scale * T::Real::try_from_f64_lossy(f64::from(i % 17) - 7.0).unwrap(),
                    scale * T::Real::try_from_f64_lossy(f64::from(i % 11) - 3.0).unwrap(),
                )
            })
            .collect();
        let mut sums = [ScaledAccumulator::<f64>::ZERO; 2];
        for &x in &values {
            sums[0].add_value(x.real_component().into_f64()).unwrap();
            sums[1]
                .add_value(x.imaginary_component().into_f64())
                .unwrap();
        }
        let n = Scaled::from_value(values.len() as f64).unwrap();
        let mean = sums.map(|s| s.total().unwrap().div(n).unwrap());
        let deviations: Vec<_> = values
            .iter()
            .map(|x| {
                [
                    Scaled::from_value(x.real_component().into_f64())
                        .unwrap()
                        .sub(mean[0])
                        .unwrap(),
                    Scaled::from_value(x.imaginary_component().into_f64())
                        .unwrap()
                        .sub(mean[1])
                        .unwrap(),
                ]
            })
            .collect();
        for ddof in [0, 1, 256] {
            let mut squares = ScaledAccumulator::<f64>::ZERO;
            for d in &deviations {
                squares
                    .add_scaled(d[0].square().unwrap().add(d[1].square().unwrap()).unwrap())
                    .unwrap();
            }
            let sd = squares
                .total()
                .unwrap()
                .div(Scaled::from_value((values.len() - ddof) as f64).unwrap())
                .unwrap()
                .sqrt()
                .unwrap();
            let actual = standardize(&values, &values, ddof).unwrap();
            for (actual, d) in actual.iter().zip(&deviations) {
                close(
                    actual.real_component(),
                    d[0].div(sd).unwrap().round().unwrap(),
                );
                close(
                    actual.imaginary_component(),
                    d[1].div(sd).unwrap().round().unwrap(),
                );
            }
        }
    }
}

#[test]
fn native_and_extended_outputs_match_independent_scaled_equations() {
    independent_reference::<f32>();
    independent_reference::<f64>();
    independent_reference::<Complex<f32>>();
    independent_reference::<Complex<f64>>();
}

#[test]
fn dispersion_only_centers_remain_accurate_with_an_anchor_outlier() {
    fn check<S: RealScalar>() {
        let base = S::ONE.scalbn(if S::MANTISSA_DIGITS == 24 { 20 } else { 50 });
        let step = S::HALF;
        for count in [63, 64, 1023, 1024, 1025] {
            let mut values = vec![base; count];
            values[0] = base + step;
            let variance = statistic(&values, Unweighted, Statistic::variance(1))
                .unwrap()
                .scalar()
                .unwrap();
            let deviation = statistic(&values, Unweighted, Statistic::standard_deviation(1))
                .unwrap()
                .scalar()
                .unwrap();
            let expected = 0.25 / count as f64;
            assert!((variance.into_f64() / expected - 1.0).abs() <= 32.0 * S::EPSILON.into_f64());
            assert!(
                (deviation.into_f64() / expected.sqrt() - 1.0).abs()
                    <= 32.0 * S::EPSILON.into_f64()
            );
            values.reverse();
            let reversed = statistic(&values, Unweighted, Statistic::standard_deviation(1))
                .unwrap()
                .scalar()
                .unwrap();
            assert!(
                (reversed.into_f64() / expected.sqrt() - 1.0).abs() <= 32.0 * S::EPSILON.into_f64()
            );
            let varying: Vec<S> = (0..count)
                .map(|i| {
                    S::try_from_f64_lossy(((i * 73) % 1031) as f64 / 1031.0 + 0.03125).unwrap()
                })
                .collect();
            let reference = OnlineMoments::from_slice(&varying, MomentOrder::Second)
                .unwrap()
                .snapshot()
                .standard_deviation(1)
                .unwrap()
                .unwrap();
            let actual = statistic(&varying, Unweighted, Statistic::standard_deviation(1))
                .unwrap()
                .scalar()
                .unwrap();
            assert!(
                (actual.into_f64() / reference.into_f64() - 1.0).abs()
                    <= 64.0 * S::EPSILON.into_f64()
            );
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn entropy_selector_uses_requested_base_and_normalization() {
    let options = EntropyOptions::new()
        .with_base(2.0)
        .with_normalization(MassNormalization::Normalize);
    let actual = statistic(
        &[1.0_f64, 1.0, 0.0],
        Unweighted,
        Statistic::entropy(options),
    )
    .unwrap();
    let StatisticValue::Information(InformationValue::Finite(bits)) = actual else {
        unreachable!();
    };
    close(bits, 1.0);
}
