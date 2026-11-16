#![allow(clippy::unwrap_used)]
#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{
    ExecutionContext, NonFiniteOrigin, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;
use scioxide_random::{Pcg64, Philox4x64, Rng, SeedableRng};
use scioxide_stats::{
    ConfidenceProbability, MeanPowerDesign, MeanPowerModel, MeanPowerReference, PowerEstimate,
    StatisticsOptions, TestAlternative, power, sample_size, simulated_power, simulated_power_into,
};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

fn significance(alpha: f64) -> ConfidenceProbability {
    ConfidenceProbability::new(alpha).unwrap()
}

fn analytic(probability: f64) -> PowerEstimate {
    PowerEstimate::analytic(probability).unwrap()
}

fn simulated(rejections: usize, replicates: usize) -> PowerEstimate {
    let calls = AtomicUsize::new(0);
    simulated_power(
        |_, sample: &mut [f64]| {
            sample[0] = if calls.fetch_add(1, Ordering::Relaxed) < rejections {
                0.0
            } else {
                1.0
            };
            Ok(())
        },
        |x, _| Ok(x[0]),
        1,
        0,
        significance(0.5),
        replicates,
        &Pcg64::from_seed(558),
    )
    .unwrap()
}

fn provider_error() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("sample"),
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        "provider failed",
    )
}

#[test]
fn normal_power_matches_independent_gaussian_identity() {
    let model = MeanPowerModel {
        effect: 0.5,
        standard_deviation: 1.0,
        design: MeanPowerDesign::OneSample,
        reference: MeanPowerReference::KnownVariance,
        alternative: TestAlternative::Greater,
        significance: significance(0.05),
    };
    let result = power(model, 64).unwrap();
    // Phi(delta-z_.95), from an independent standard-normal integral.
    assert!((result.probability() - 0.990_742_294_6).abs() < 2e-9);
    let first = sample_size(2, 100, 0.8, |n| power(model, n))
        .unwrap()
        .unwrap();
    assert!(first.estimate.probability() >= 0.8);
    assert!(power(model, first.size - 1).unwrap().probability() < 0.8);
    let null = MeanPowerModel {
        effect: 0.0,
        ..model
    };
    assert!((power(null, 20).unwrap().probability() - 0.05).abs() < 1e-12);
    // The validated significance holds α itself, so tiny levels stay exact.
    let tiny = MeanPowerModel {
        significance: significance(1e-20),
        ..null
    };
    let tiny_power = power(tiny, 20).unwrap().probability();
    assert!((tiny_power - 1e-20).abs() / 1e-20 < 1e-8);
}

#[test]
fn student_null_and_invalid_inputs() {
    let model = MeanPowerModel {
        effect: 0.0,
        standard_deviation: 1.0,
        design: MeanPowerDesign::EqualGroups,
        reference: MeanPowerReference::EstimatedVariance,
        alternative: TestAlternative::TwoSided,
        significance: significance(0.05),
    };
    assert!((power(model, 8).unwrap().probability() - 0.05).abs() < 2e-9);
    assert!(
        power(
            MeanPowerModel {
                effect: 1.0,
                ..model
            },
            8
        )
        .unwrap()
        .probability()
            > 0.2
    );
    assert!(power(model, 1).is_err());
    assert!(
        power(
            MeanPowerModel {
                standard_deviation: 0.0,
                ..model
            },
            8
        )
        .is_err()
    );
    let single = MeanPowerModel {
        effect: 0.5_f32,
        standard_deviation: 1.0_f32,
        design: MeanPowerDesign::OneSample,
        reference: MeanPowerReference::KnownVariance,
        alternative: TestAlternative::Greater,
        significance: significance(0.05),
    };
    assert!((power(single, 64).unwrap().probability() - 0.990_742_294_6).abs() < 2e-9);
}

#[test]
fn simulated_power_replays_and_reports_wilson_uncertainty() {
    let sampler = |rng: &mut Pcg64, out: &mut [f64]| {
        out[0] = rng.next_float()?;
        Ok(())
    };
    let test = |x: &[f64], _: &[f64]| Ok(x[0]);
    let root = Pcg64::from_seed(558);
    let options = StatisticsOptions::new().with_max_bytes(8192);
    let mut a = vec![0.0; 512];
    let mut b = a.clone();
    let first = simulated_power_into(
        sampler,
        test,
        1,
        0,
        significance(0.5),
        &root,
        options,
        &mut a,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let second = simulated_power(sampler, test, 1, 0, significance(0.5), 512, &root).unwrap();
    assert_eq!(first, second);
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(NonZeroUsize::new(4).unwrap());
    let third = simulated_power_into(
        sampler,
        test,
        1,
        0,
        significance(0.5),
        &root,
        options.with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(
            NonZeroUsize::new(4).unwrap(),
        )),
        &mut b,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(first, third);
    assert_eq!(a, b);
    assert!((first.probability() - 0.5).abs() < 0.07);
    let interval = first.interval().unwrap();
    assert!(interval.lower() < first.probability() && first.probability() < interval.upper());
    let error = simulated_power_into(
        sampler,
        test,
        1,
        0,
        significance(0.5),
        &root,
        options.with_max_bytes(0),
        &mut b,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ResourceLimit { required, limit: 0, .. } if required > 0)
    );
    b.fill(-1.0);
    assert!(
        simulated_power_into(
            sampler,
            |_, _| Ok(f64::NAN),
            1,
            0,
            significance(0.5),
            &root,
            options,
            &mut b,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(b.iter().all(|&value| value == -1.0));
}

#[test]
fn wilson_interval_calibrates_repeated_independent_power_runs() {
    let mut pvalues = vec![0.0; 512];
    let covered = (0..64)
        .filter(|&seed| {
            let root = Philox4x64::from_seed(seed);
            let report = simulated_power_into(
                |rng: &mut Philox4x64, sample: &mut [f64]| {
                    sample[0] = rng.next_float()?;
                    Ok(())
                },
                |sample, _| Ok(sample[0]),
                1,
                0,
                significance(0.5),
                &root,
                StatisticsOptions::new().with_max_bytes(8192),
                &mut pvalues,
                ExecutionContext::sequential(),
            )
            .unwrap();
            report.interval().unwrap().contains(0.5)
        })
        .count();
    assert!(covered >= 55, "Wilson coverage: {covered}/64");

    let estimate = simulated(95, 100);
    let inconclusive = sample_size(2, 3, 0.9, |_| Ok(estimate)).unwrap();
    assert!(inconclusive.is_none());
}

#[test]
fn invalid_configuration_is_rejected_before_callbacks() {
    for alpha in [
        0.0,
        -0.0,
        1.0,
        -0.1,
        1.1,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let expected = if alpha.is_finite() {
            SciOxideErrorKind::InvalidInput
        } else {
            SciOxideErrorKind::NonFinite
        };
        // An invalid significance cannot reach a model or plan.
        assert_eq!(
            ConfidenceProbability::new(alpha).unwrap_err().kind(),
            expected
        );
    }
    for (first_size, second_size, replicates, expected) in [
        (0, 0, 2, SciOxideErrorKind::InvalidInput),
        (1, 0, 0, SciOxideErrorKind::InvalidInput),
        (1, 0, 1, SciOxideErrorKind::InvalidInput),
        (usize::MAX, 1, 2, SciOxideErrorKind::SizeOverflow),
    ] {
        let calls = AtomicUsize::new(0);
        let mut output = vec![-1.0; replicates];
        let error = simulated_power_into(
            |_, _: &mut [f64]| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
            |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(0.0)
            },
            first_size,
            second_size,
            significance(0.05),
            &Pcg64::from_seed(558),
            StatisticsOptions::new(),
            &mut output,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), expected);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
    let calls = AtomicUsize::new(0);
    let mut output = [-1.0; 2];
    let error = simulated_power_into(
        |_, _: &mut [f64]| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
        |_, _| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(0.0)
        },
        2,
        0,
        significance(0.05),
        &Pcg64::from_seed(558),
        StatisticsOptions::new().with_max_bytes(8),
        &mut output,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(output, [-1.0; 2]);
}

#[test]
fn invalid_search_configuration_never_evaluates_provider() {
    let mut calls = 0;
    for (minimum, maximum, target) in [
        (0, 3, 0.8),
        (1, 3, 0.8),
        (3, 2, 0.8),
        (2, 3, 0.0),
        (2, 3, -0.0),
        (2, 3, 1.0),
        (2, 3, -0.1),
        (2, 3, 1.1),
        (2, 3, f64::NAN),
        (2, 3, f64::INFINITY),
        (2, 3, f64::NEG_INFINITY),
    ] {
        let error = sample_size(minimum, maximum, target, |_| {
            calls += 1;
            Ok(analytic(1.0))
        })
        .unwrap_err();
        assert_eq!(
            error.kind(),
            if target.is_finite() {
                SciOxideErrorKind::InvalidInput
            } else {
                SciOxideErrorKind::NonFinite
            }
        );
    }
    assert_eq!(calls, 0);
}

#[test]
fn evidence_constructors_reject_inconsistent_values() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1, 2.0] {
        let error = PowerEstimate::analytic(value).unwrap_err();
        if value.is_finite() {
            assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        } else {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }
    for (rejections, replicates) in [(0, 0), (0, 1), (1, 1), (101, 100)] {
        assert!(matches!(
            PowerEstimate::simulated(rejections, replicates)
                .unwrap_err()
                .view(),
            SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            }
        ));
    }
    let analytic = analytic(0.9);
    assert_eq!(analytic.interval(), None);
    assert_eq!((analytic.replicates(), analytic.rejections()), (0, 0));
    let simulation = PowerEstimate::simulated(95, 100).unwrap();
    assert_eq!(simulation.probability(), 0.95);
    assert_eq!(
        (simulation.replicates(), simulation.rejections()),
        (100, 95)
    );
    assert!(simulation.interval().unwrap().contains(0.95));
    // Constructor errors returned by a provider stop the search at once.
    let mut calls = 0;
    let error = sample_size(2, 3, 0.8, |_| {
        calls += 1;
        PowerEstimate::simulated(101, 100)
    })
    .unwrap_err();
    assert_eq!(calls, 1);
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
}

#[test]
fn search_uses_first_qualifying_canonical_evidence() {
    let mut sizes = Vec::new();
    let analytic_result = sample_size(2, 5, 0.8, |n| {
        sizes.push(n);
        Ok(analytic(if n < 4 { 0.7 } else { 0.8 }))
    })
    .unwrap()
    .unwrap();
    assert_eq!(sizes, [2, 3, 4]);
    assert_eq!(analytic_result.size, 4);
    assert_eq!(analytic_result.evaluated, 3);
    assert_eq!(analytic_result.estimate, analytic(0.8));
    let low = simulated(80, 100);
    let high = simulated(95, 100);
    let result = sample_size(2, 5, 0.8, |n| Ok(if n < 4 { low } else { high }))
        .unwrap()
        .unwrap();
    assert_eq!(result.size, 4);
    assert_eq!(result.evaluated, 3);
    assert_eq!(result.estimate, high);

    // Simulated evidence qualifies by its count-derived Wilson lower bound.
    let lo = high.interval().unwrap().lower();
    let result = sample_size(2, 2, lo, |_| Ok(high)).unwrap().unwrap();
    assert_eq!(result.estimate, high);
    assert!(
        sample_size(2, 2, lo + f64::EPSILON, |_| Ok(high))
            .unwrap()
            .is_none()
    );
    for estimate in [analytic(0.0), low] {
        let mut calls = 0;
        assert!(
            sample_size(2, 4, 0.9, |_| {
                calls += 1;
                Ok(estimate)
            })
            .unwrap()
            .is_none()
        );
        assert_eq!(calls, 3);
    }
    assert_eq!(
        sample_size(2, 2, 0.8, |_| Ok(analytic(1.0)))
            .unwrap()
            .unwrap()
            .size,
        2
    );
    assert_eq!(
        sample_size(usize::MAX, usize::MAX, 0.8, |_| Ok(analytic(1.0)))
            .unwrap()
            .unwrap()
            .evaluated,
        1
    );
}

#[test]
fn simulation_boundaries_and_strict_rejection_rule() {
    for (rejections, expected_probability, endpoint) in [(0, 0.0, 0.0), (100, 1.0, 1.0)] {
        let estimate = simulated(rejections, 100);
        assert_eq!(estimate.probability(), expected_probability);
        let bounds = estimate.interval().unwrap();
        let observed = if rejections == 0 {
            bounds.lower()
        } else {
            bounds.upper()
        };
        assert_eq!(observed, endpoint);
        let other = if rejections == 0 {
            bounds.upper()
        } else {
            1.0 - bounds.lower()
        };
        // Independent Wilson endpoint fixture for n=100.
        assert!((other - 0.036_993_498_206_985_68).abs() < 2e-15);
        let result = sample_size(2, 2, 0.8, |_| Ok(estimate)).unwrap();
        assert_eq!(result.is_some(), rejections == 100);
    }
    let mut output = [-1.0; 2];
    let result = simulated_power_into(
        |_, _: &mut [f64]| Ok(()),
        |_, _| Ok(0.5),
        1,
        0,
        significance(0.5),
        &Pcg64::from_seed(558),
        StatisticsOptions::new().with_max_bytes(1024),
        &mut output,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(result.rejections(), 0);
    assert_eq!(output, [0.5; 2]);
}

#[test]
fn search_stops_at_callback_or_evidence_errors() {
    for malformed in [false, true] {
        let mut sizes = Vec::new();
        let error = sample_size(2, 5, 0.8, |n| {
            sizes.push(n);
            if n == 2 {
                Ok(analytic(0.5))
            } else if malformed {
                PowerEstimate::analytic(2.0)
            } else {
                Err(provider_error())
            }
        })
        .unwrap_err();
        assert_eq!(sizes, [2, 3]);
        if !malformed {
            assert_eq!(error, provider_error());
        }
    }
}

#[test]
fn simulation_failures_after_successful_replicates_preserve_output() {
    let root = Pcg64::from_seed(558);
    for fail_sampler in [false, true] {
        for invalid_value in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(-0.1),
            Some(1.1),
        ] {
            // Finite sample data outside [0,1] is legal; only p-values have that domain.
            if fail_sampler && invalid_value.is_some_and(f64::is_finite) {
                continue;
            }
            let sampler_calls = AtomicUsize::new(0);
            let test_calls = AtomicUsize::new(0);
            let mut output = [-1.0; 4];
            let error = simulated_power_into(
                |_, sample: &mut [f64]| {
                    let call = sampler_calls.fetch_add(1, Ordering::Relaxed);
                    if fail_sampler && call == 1 {
                        match invalid_value {
                            None => return Err(provider_error()),
                            Some(value) => sample[0] = value,
                        }
                    } else {
                        sample[0] = 0.0;
                    }
                    Ok(())
                },
                |_, _| {
                    let call = test_calls.fetch_add(1, Ordering::Relaxed);
                    if !fail_sampler && call == 1 {
                        invalid_value.ok_or_else(provider_error)
                    } else {
                        Ok(0.0)
                    }
                },
                1,
                0,
                significance(0.5),
                &root,
                StatisticsOptions::new().with_max_bytes(1024),
                &mut output,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
            assert_eq!(output, [-1.0; 4]);
            assert_eq!(sampler_calls.load(Ordering::Relaxed), 2);
            assert_eq!(
                test_calls.load(Ordering::Relaxed),
                if fail_sampler { 1 } else { 2 }
            );
            if invalid_value.is_none() {
                assert_eq!(error, provider_error());
            } else if invalid_value.is_some_and(|v| !v.is_finite()) {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Callback,
                        ..
                    }
                ));
            } else {
                assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
            }
        }
    }
}
