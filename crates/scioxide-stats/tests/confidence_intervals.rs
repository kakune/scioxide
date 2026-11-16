//! Independent critical values, metadata and repeated-sampling coverage.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/reductions.rs"]
mod reduction_test;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::BetaOptions;
use scioxide_stats::*;

fn probability() -> ConfidenceProbability {
    ConfidenceProbability::default()
}

#[test]
fn scaled_errors_preserve_margins_below_the_scalar_range() {
    use scioxide_core::Scaled;
    let plan = MeanIntervalPlan::new(
        probability(),
        MeanIntervalMethod::StudentT,
        Some(3.0),
        Default::default(),
    )
    .unwrap();
    fn check<S: RealScalar>(plan: MeanIntervalPlan) {
        let tiny = S::ZERO.next_after(S::ONE);
        let error = Scaled::from_value(tiny).unwrap().shifted(-2).unwrap();
        let interval = plan.interval_scaled(S::ZERO, error).unwrap();
        assert_eq!(interval.standard_error(), S::ZERO);
        assert_eq!(interval.bounds().lower(), -tiny);
        assert_eq!(interval.bounds().upper(), tiny);
        assert!(!interval.is_degenerate());
        assert!(
            plan.interval_scaled(S::ZERO, Scaled::ZERO)
                .unwrap()
                .is_degenerate()
        );
        assert!(plan.interval_scaled(S::NAN, error).is_err());
        assert!(
            plan.interval_scaled(S::ZERO, Scaled::from_value(-S::ONE).unwrap())
                .is_err()
        );
        assert!(
            plan.interval_scaled(S::MAX, Scaled::from_value(S::MAX).unwrap())
                .is_err()
        );
    }
    check::<f32>(plan);
    check::<f64>(plan);
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1e-300),
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn independent_critical_values() {
    for line in include_str!("data/intervals.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let p = ConfidenceProbability::new(row[1]).unwrap();
        let actual = p
            .student_t_critical_value_with_options(row[0], BetaOptions::default())
            .unwrap_or_else(|e| panic!("{line}: {e}"));
        close(actual, row[2], 2e-10);
        for (estimate, error) in [(0.0_f32, 1.0_f32), (2.0, 0.5)] {
            let result = scioxide_stats::MeanIntervalPlan::new(
                p,
                MeanIntervalMethod::StudentT,
                Some(row[0]),
                Default::default(),
            )
            .and_then(|plan| plan.interval(estimate, error))
            .unwrap();
            close(
                result.bounds().upper() as f64,
                estimate as f64 + error as f64 * row[2],
                5e-7,
            );
        }
    }
}

fn snapshot<S: RealScalar>(values: &[S]) -> scioxide_stats::MomentsSnapshot<S> {
    OnlineMoments::from_slice(values, MomentOrder::Second)
        .unwrap()
        .snapshot()
}

#[test]
fn defaults_methods_and_degeneracy() {
    let s = snapshot(&[1.0_f64, 2.0, 3.0]);
    let t = s
        .confidence_interval_of_mean_with_options(
            probability(),
            scioxide_stats::MeanIntervalMethod::StudentT,
            scioxide_special::BetaOptions::default(),
        )
        .unwrap()
        .unwrap();
    close(
        t.bounds().upper(),
        2.0 + 4.302652729749464 / 3.0_f64.sqrt(),
        1e-8,
    );
    assert_eq!(t.degrees_of_freedom(), Some(2.0));
    assert_eq!(t.assumptions(), MeanIntervalAssumptions::IndependentNormal);
    assert_eq!(t.confidence(), probability());
    assert_eq!(t.estimate(), 2.0);
    let z = s
        .confidence_interval_of_mean_with_options(
            probability(),
            MeanIntervalMethod::NormalAsymptotic,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(z.assumptions(), MeanIntervalAssumptions::AsymptoticNormal);
    assert!(z.bounds().upper() < t.bounds().upper());
    assert_eq!(z.degrees_of_freedom(), None);
    for values in [&[][..], &[1.0][..]] {
        assert!(
            snapshot(values)
                .confidence_interval_of_mean_with_options(
                    probability(),
                    scioxide_stats::MeanIntervalMethod::StudentT,
                    scioxide_special::BetaOptions::default()
                )
                .unwrap()
                .is_none()
        );
    }
    let zero = snapshot(&[7.0_f32; 3])
        .confidence_interval_of_mean_with_options(
            probability(),
            scioxide_stats::MeanIntervalMethod::StudentT,
            scioxide_special::BetaOptions::default(),
        )
        .unwrap()
        .unwrap();
    assert!(zero.is_degenerate());
    assert_eq!(zero.bounds().lower(), 7.0);
    let known = scioxide_stats::MeanIntervalPlan::new(
        probability(),
        MeanIntervalMethod::NormalKnownVariance,
        None,
        Default::default(),
    )
    .and_then(|plan| plan.interval(2.0_f64, 0.5))
    .unwrap();
    assert_eq!(
        known.assumptions(),
        MeanIntervalAssumptions::KnownNormalVariance
    );
    close(known.bounds().upper(), 2.979981992270027, 1e-13);
}

#[test]
fn invalid_inputs_and_exhaustion() {
    for p in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(ConfidenceProbability::new(p).is_err());
    }
    for degrees in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(
            probability()
                .student_t_critical_value_with_options(degrees, Default::default())
                .is_err()
        );
        assert!(
            scioxide_stats::MeanIntervalPlan::new(
                probability(),
                MeanIntervalMethod::StudentT,
                Some(degrees),
                Default::default()
            )
            .and_then(|plan| plan.interval(0.0, 0.0))
            .is_err()
        );
    }
    for (mean, se) in [
        (f64::NAN, 1.0),
        (0.0, -1.0),
        (0.0, f64::INFINITY),
        (f64::MAX, f64::MAX),
    ] {
        assert!(
            scioxide_stats::MeanIntervalPlan::new(
                probability(),
                MeanIntervalMethod::NormalKnownVariance,
                None,
                Default::default()
            )
            .and_then(|plan| plan.interval(mean, se))
            .is_err()
        );
    }
    assert!(
        matches ! (& (probability () . student_t_critical_value_with_options (3.0 , BetaOptions :: default () . with_iteration_budget (Budget :: ONE))) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ConvergenceFailure { .. }))
    );
    assert!(
        scioxide_stats::MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::StudentT,
            Some(3.0),
            BetaOptions::default().with_iteration_budget(Budget::ONE)
        )
        .and_then(|plan| plan.interval(7.0_f32, 0.0))
        .is_err()
    );
}

fn weighted<M: MomentWeights>(values: &[f64], weights: &[f64]) -> MeanConfidenceInterval<f64> {
    let mut a = OnlineWeightedMoments::<f64, M>::new(MomentOrder::Second);
    for (&x, &w) in values.iter().zip(weights) {
        a.observe(x, w).unwrap();
    }
    a.snapshot()
        .confidence_interval_of_weighted_mean_with_options(
            probability(),
            scioxide_stats::MeanIntervalMethod::StudentT,
            scioxide_special::BetaOptions::default(),
        )
        .unwrap()
        .unwrap()
}
type Interval = fn(&[f64], &[f64]) -> MeanConfidenceInterval<f64>;

#[test]
fn weight_models_and_summary_reuse() {
    let values = [1., 2., 4.];
    for (interval, mode, assumption, degrees) in [
        (
            weighted::<Frequency> as Interval,
            WeightSemantics::Frequency,
            MeanIntervalAssumptions::FrequencyNormal,
            5.0,
        ),
        (
            weighted::<Analytic> as Interval,
            WeightSemantics::Analytic,
            MeanIntervalAssumptions::AnalyticNormal,
            2.0,
        ),
        (
            weighted::<Reliability> as Interval,
            WeightSemantics::Reliability,
            MeanIntervalAssumptions::EffectiveSampleApproximation,
            36.0 / 14.0 - 1.0,
        ),
    ] {
        let a = interval(&values, &[1., 2., 3.]);
        assert_eq!(a.assumptions(), assumption);
        close(a.degrees_of_freedom().unwrap(), degrees, 1e-14);
        if mode != WeightSemantics::Frequency {
            let b = interval(&values, &[10., 20., 30.]);
            close(a.bounds().upper(), b.bounds().upper(), 1e-13);
        } else {
            let b = snapshot(&[1., 2., 2., 4., 4., 4.])
                .confidence_interval_of_mean_with_options(
                    probability(),
                    scioxide_stats::MeanIntervalMethod::StudentT,
                    scioxide_special::BetaOptions::default(),
                )
                .unwrap()
                .unwrap();
            close(a.bounds().upper(), b.bounds().upper(), 1e-13);
        }
    }
    let mut left = OnlineMoments::from_slice(&values[..1], MomentOrder::Second).unwrap();
    left.merge(&OnlineMoments::from_slice(&values[1..], MomentOrder::Second).unwrap())
        .unwrap();
    close(
        left.snapshot()
            .confidence_interval_of_mean_with_options(
                probability(),
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default(),
            )
            .unwrap()
            .unwrap()
            .bounds()
            .upper(),
        snapshot(&values)
            .confidence_interval_of_mean_with_options(
                probability(),
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default(),
            )
            .unwrap()
            .unwrap()
            .bounds()
            .upper(),
        1e-13,
    );
}

// Fixed SplitMix64 uniforms and Box-Muller normals, independent of SciOxide law kernels.
fn normal(seed: &mut u64) -> f64 {
    let mut uniform = || {
        *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / 9007199254740992.0
    };
    (-2.0 * uniform().ln()).sqrt() * (core::f64::consts::TAU * uniform()).cos()
}

#[test]
fn repeated_sampling_coverage() {
    let mut seed = 545_u64;
    let mut hits = [0_usize; 4];
    const REPLICATES: usize = 10_000;
    for _ in 0..REPLICATES {
        let xs = core::array::from_fn::<_, 5, _>(|_| normal(&mut seed));
        let s = snapshot(&xs);
        let t = s
            .confidence_interval_of_mean_with_options(
                probability(),
                scioxide_stats::MeanIntervalMethod::StudentT,
                scioxide_special::BetaOptions::default(),
            )
            .unwrap()
            .unwrap();
        let known = scioxide_stats::MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::NormalKnownVariance,
            None,
            Default::default(),
        )
        .and_then(|plan| plan.interval(s.mean().unwrap().unwrap(), 1.0 / 5.0_f64.sqrt()))
        .unwrap();
        let ys: Vec<_> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| x / ((i + 1) as f64).sqrt())
            .collect();
        let analytic = weighted::<Analytic>(&ys, &[1., 2., 3., 4., 5.]);
        let reliability = weighted::<Reliability>(&xs, &[1., 2., 3., 4., 5.]);
        for (count, result) in hits.iter_mut().zip([t, known, analytic, reliability]) {
            *count += usize::from(result.bounds().lower() <= 0.0 && result.bounds().upper() >= 0.0);
        }
    }
    // Six binomial standard errors: deterministic, predeclared, no rerun-to-pass.
    let radius = 6.0 * (0.95 * 0.05 / REPLICATES as f64).sqrt();
    for &count in &hits[..3] {
        assert!(
            (count as f64 / REPLICATES as f64 - 0.95).abs() < radius,
            "{hits:?}"
        );
    }
    eprintln!(
        "coverage t/known/analytic/reliability: {hits:?}/{REPLICATES}; exact-model band 0.95 +/- {radius}"
    );
}

#[test]
fn weighted_f32_and_asymptotic_metadata() {
    fn check<M: MomentWeights>(mode: WeightSemantics) {
        let mut a = OnlineWeightedMoments::<f32, M>::new(MomentOrder::Second);
        for (x, w) in [(1.0_f32, 1.0), (2., 2.), (4., 3.)] {
            a.observe(x, w).unwrap();
        }
        let s = a.snapshot();
        for method in [
            MeanIntervalMethod::StudentT,
            MeanIntervalMethod::NormalAsymptotic,
        ] {
            let result = s
                .confidence_interval_of_weighted_mean_with_options(
                    probability(),
                    method,
                    Default::default(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(result.weight_semantics(), Some(mode));
            assert!(result.bounds().lower() < result.estimate());
            assert!(result.bounds().upper() > result.estimate());
        }
    }
    check::<Frequency>(WeightSemantics::Frequency);
    check::<Analytic>(WeightSemantics::Analytic);
    check::<Reliability>(WeightSemantics::Reliability);
    assert!(
        probability()
            .student_t_critical_value_with_options(
                4096.,
                BetaOptions::default().with_iteration_budget(Budget::ONE)
            )
            .is_err()
    );
    assert!(
        probability()
            .student_t_critical_value_with_options(
                4096.,
                BetaOptions::new(Budget::ONE.saturating_add(3))
            )
            .is_ok()
    );
}

#[cfg(feature = "std")]
#[test]
fn borrowed_views_and_executor_summaries_retain_inference() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, ThreadExecutor};
    use scioxide_stats::*;
    let values: Vec<f32> = (0..1024).map(|i| (i % 7) as f32).collect();
    let view_shape = [1, values.len()];
    let view_strides = [values.len(), 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let workers = core::num::NonZeroUsize::MIN.saturating_add(1);
    let executor = ThreadExecutor::new(workers);
    let options_axis = Axis::All;
    let options = StatisticsOptions::new().with_merge_policy(MergePolicy::StablePartitions);
    let sequential = reduction_test::moments(
        view.with_axis(options_axis).unwrap(),
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let parallel = reduction_test::moments(
        view.with_axis(options_axis).unwrap(),
        options.with_execution_policy(ExecutionPolicy::Parallel(workers)),
        ExecutionContext::new(&executor),
    )
    .unwrap();
    let a = sequential[0]
        .confidence_interval_of_mean_with_options(
            probability(),
            scioxide_stats::MeanIntervalMethod::StudentT,
            scioxide_special::BetaOptions::default(),
        )
        .unwrap()
        .unwrap();
    let b = parallel[0]
        .confidence_interval_of_mean_with_options(
            probability(),
            scioxide_stats::MeanIntervalMethod::StudentT,
            scioxide_special::BetaOptions::default(),
        )
        .unwrap()
        .unwrap();
    close(a.bounds().upper() as f64, b.bounds().upper() as f64, 5e-7);
    assert_eq!(a.method(), b.method());
    assert_eq!(a.assumptions(), b.assumptions());
}

#[test]
fn prepared_intervals_reuse_critical_values_and_validate_each_estimate() {
    use scioxide_stats::MeanIntervalPlan;
    for (method, degrees) in [
        (MeanIntervalMethod::StudentT, Some(3.0)),
        (MeanIntervalMethod::StudentT, Some(4096.0)),
        (MeanIntervalMethod::NormalAsymptotic, None),
        (MeanIntervalMethod::NormalKnownVariance, None),
    ] {
        let plan =
            MeanIntervalPlan::new(probability(), method, degrees, BetaOptions::default()).unwrap();
        for (mean, error) in [(2.0_f64, 0.5), (-1e100, 1e99), (1.0, 0.0)] {
            let actual = plan.interval(mean, error).unwrap();
            let expected = scioxide_stats::MeanIntervalPlan::new(
                probability(),
                method,
                degrees,
                BetaOptions::default(),
            )
            .and_then(|plan| plan.interval(mean, error))
            .unwrap();
            assert_eq!(actual, expected);
            let actual = plan.interval(mean as f32, error as f32);
            let expected = scioxide_stats::MeanIntervalPlan::new(
                probability(),
                method,
                degrees,
                BetaOptions::default(),
            )
            .and_then(|plan| plan.interval(mean as f32, error as f32));
            assert_eq!(actual.is_ok(), expected.is_ok());
            if let (Ok(actual), Ok(expected)) = (actual, expected) {
                assert_eq!(actual, expected);
            }
        }
        for (mean, error) in [
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
            (1.0, -1.0),
            (f64::MAX, f64::MAX),
        ] {
            assert!(plan.interval(mean, error).is_err());
        }
    }
    assert!(
        MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::StudentT,
            None,
            BetaOptions::default()
        )
        .is_err()
    );
    assert!(
        MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::NormalAsymptotic,
            Some(1.0),
            BetaOptions::default()
        )
        .is_err()
    );
    let options = BetaOptions::default().with_iteration_budget(Budget::ONE);
    assert!(
        MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::StudentT,
            Some(3.0),
            options
        )
        .is_err()
    );
    assert!(
        scioxide_stats::MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::StudentT,
            Some(3.0),
            options
        )
        .and_then(|plan| plan.interval(1.0, 0.0))
        .is_err()
    );
}

#[test]
fn prepared_native_endpoints_match_scaled_rounding_at_range_boundaries() {
    fn check<S: RealScalar>() {
        use scioxide_core::Scaled;
        let critical = probability()
            .student_t_critical_value_with_options(31.0, BetaOptions::default())
            .unwrap();
        let plan = MeanIntervalPlan::new(
            probability(),
            MeanIntervalMethod::StudentT,
            Some(31.0),
            BetaOptions::default(),
        )
        .unwrap();
        for (estimate, error) in [
            (-S::ZERO, S::ZERO),
            (S::ONE, S::HALF),
            (
                S::MIN_SUBNORMAL * S::try_from_f64_lossy(3.0).unwrap(),
                S::MIN_SUBNORMAL,
            ),
            (
                S::MAX / S::try_from_f64_lossy(4.0).unwrap(),
                S::MAX / S::try_from_f64_lossy(8.0).unwrap(),
            ),
            (S::MAX, S::MAX),
        ] {
            let margin = Scaled::<f64>::from_value(error.into_f64())
                .unwrap()
                .mul(Scaled::from_value(critical).unwrap())
                .unwrap();
            let center = Scaled::<f64>::from_value(estimate.into_f64()).unwrap();
            let expected = center
                .sub(margin)
                .unwrap()
                .round_to::<S>()
                .and_then(|lo| center.add(margin)?.round_to::<S>().map(|hi| (lo, hi)));
            let actual = plan.interval(estimate, error);
            match (actual, expected) {
                (Ok(actual), Ok((lo, hi))) => {
                    assert_eq!(
                        actual.bounds().lower().into_f64().to_bits(),
                        lo.into_f64().to_bits()
                    );
                    assert_eq!(
                        actual.bounds().upper().into_f64().to_bits(),
                        hi.into_f64().to_bits()
                    );
                }
                (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
                _ => panic!("native/scaled projection mismatch"),
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
