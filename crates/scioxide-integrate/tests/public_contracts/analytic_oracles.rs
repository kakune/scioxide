//! Cross-family numerical oracles.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_integrate::GaussianRulePlan;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, NormSelection, RealScalar, SciOxideError,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, DoubleExponentialDomain, DoubleExponentialIntegrator,
    FixedRule, FourierWeight, GaussKronrod, GaussKronrodRule, IntegrationReport,
    IntegrationTermination, LevelOptions, MonteCarloIntegrator, MonteCarloOptions, ProductRule,
    RandomStreams, SubdivisionOptions, WeightedQuadraturePlan,
};
use scioxide_random::{Philox4x64, SeedableRng};

fn reference(name: &str) -> f64 {
    // generate_batteries.py
    include_str!("../data/constants.csv")
        .lines()
        .filter_map(|line| line.split_once(','))
        .find(|&(id, _)| id == name)
        .unwrap()
        .1
        .parse()
        .unwrap()
}
const GK_PAIRS: [(GaussKronrodRule, usize); 5] = [
    (GaussKronrodRule::G7K15, 15),
    (GaussKronrodRule::G10K21, 21),
    (GaussKronrodRule::G15K31, 31),
    (GaussKronrodRule::G20K41, 41),
    (GaussKronrodRule::G25K51, 51),
];

fn tolerance<S: RealScalar + Default>(value: S) -> ErrorTolerance<S> {
    ErrorTolerance::scalar(value, value, NormSelection::Infinity).unwrap()
}

/// A successful published estimate within `tolerance` of `expected`.
fn assert_report<S: RealScalar + Default>(
    result: Result<(S, IntegrationReport<S>), SciOxideError>,
    expected: S,
    tolerance: S,
) {
    let (value, report) = result.unwrap();
    assert!(report.is_success(), "termination: {}", report.termination());
    assert!(value.is_finite());
    assert!((value - expected).abs() <= tolerance);
    if let Some(error) = report.error_estimate() {
        assert!(error.is_finite() && error >= S::ZERO);
    }
}
fn assert_f64_result(
    result: (f64, scioxide_integrate::IntegrationReport<f64>),
    expected: f64,
    tolerance: f64,
    callbacks: usize,
) {
    assert!(
        result.1.is_success(),
        "termination: {}",
        result.1.termination()
    );
    assert!(result.0.is_finite());
    assert!((result.0 - expected).abs() <= tolerance);
    assert_eq!(result.1.evaluations(), callbacks);
    if let Some(error) = result.1.error_estimate() {
        assert!(error.is_finite() && error >= 0.0);
    }
}

fn assert_f32_result(
    result: (f32, scioxide_integrate::IntegrationReport<f32>),
    expected: f32,
    tolerance: f32,
    callbacks: usize,
) {
    assert!(
        result.1.is_success(),
        "termination: {}",
        result.1.termination()
    );
    assert!(result.0.is_finite());
    assert!((result.0 - expected).abs() <= tolerance);
    assert_eq!(result.1.evaluations(), callbacks);
    if let Some(error) = result.1.error_estimate() {
        assert!(error.is_finite() && error >= 0.0);
    }
}

fn assert_fixed_report<S: scioxide_core::RealScalar>(
    (value, report): (S, IntegrationReport<S>),
    expected: S,
    tolerance: S,
    callbacks: usize,
) {
    assert!(report.is_success(), "termination: {}", report.termination());
    assert!((value - expected).abs() <= tolerance);
    assert_eq!(report.evaluations(), callbacks);
    let error = report.error_estimate().unwrap();
    assert!(error.is_finite() && error >= S::ZERO);
}

#[test]
fn every_gauss_kronrod_pair_matches_the_gaussian_oracle_for_both_scalars() {
    let interval_f64 = Interval::new(0.0_f64, 1.0).unwrap();
    let interval_f32 = Interval::new(0.0_f32, 1.0).unwrap();

    for (pair, expected_callbacks) in GK_PAIRS {
        let rule = GaussKronrod::new(pair).unwrap();
        let calls_f64 = AtomicUsize::new(0);
        let result_f64 = rule
            .integrate(
                |x: f64| {
                    calls_f64.fetch_add(1, Ordering::Relaxed);
                    Ok((-x * x).exp())
                },
                interval_f64,
            )
            .unwrap();
        assert_eq!(calls_f64.load(Ordering::Relaxed), expected_callbacks);
        assert_fixed_report(
            result_f64,
            reference("gaussian_0_1"),
            2.0e-14,
            expected_callbacks,
        );

        let calls_f32 = AtomicUsize::new(0);
        let rule_f32 = GaussKronrod::<f32>::new(pair).unwrap();
        let result_f32 = rule_f32
            .integrate(
                |x: f32| {
                    calls_f32.fetch_add(1, Ordering::Relaxed);
                    Ok((-x * x).exp())
                },
                interval_f32,
            )
            .unwrap();
        assert_eq!(calls_f32.load(Ordering::Relaxed), expected_callbacks);
        assert_fixed_report(
            result_f32,
            reference("gaussian_0_1") as f32,
            2.0e-6,
            expected_callbacks,
        );
    }
}

#[test]
fn adaptive_and_transformed_families_match_analytic_oracles() {
    let unit_f64 = Interval::new(0.0_f64, 1.0).unwrap();
    let unit_f32 = Interval::new(0.0_f32, 1.0).unwrap();
    let peaked_expected = 5.0_f64.atan() / 5.0;

    let options_f64 = SubdivisionOptions::default()
        .with_tolerance(tolerance(1.0e-9_f64))
        .with_max_subdivisions(Budget::try_from(200).unwrap());
    let peaked_f64 = |x: f64| Ok(1.0 / 100.0_f64.mul_add((x - 0.5).powi(2), 1.0));
    for result in [
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options_f64.clone())
            .unwrap()
            .integrate(&peaked_f64, unit_f64),
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, options_f64)
            .unwrap()
            .integrate(&peaked_f64, unit_f64),
    ] {
        assert_report(result, peaked_expected, 1.0e-8);
    }

    let options_f32 = SubdivisionOptions::default()
        .with_tolerance(tolerance(1.0e-5_f32))
        .with_max_subdivisions(Budget::try_from(200).unwrap());
    let peaked_f32 = |x: f32| Ok(1.0 / 100.0_f32.mul_add((x - 0.5).powi(2), 1.0));
    for result in [
        AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options_f32.clone())
            .unwrap()
            .integrate(&peaked_f32, unit_f32),
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, options_f32)
            .unwrap()
            .integrate(&peaked_f32, unit_f32),
    ] {
        assert_report(result, peaked_expected as f32, 2.0e-4);
    }

    let de_f64 = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit_f64,
            singular_points: &[],
        },
        LevelOptions::default().with_tolerance(tolerance(1.0e-8_f64)),
    )
    .unwrap();
    let transformed_f64 = [
        de_f64.integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok(1.0 / x.sqrt())
        }),
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::HalfLine { lower: 0.0 },
            de_f64.options().clone(),
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok((-x).exp())
        }),
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::RealLine,
            de_f64.options().clone(),
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok((-x * x).exp())
        }),
    ];
    for (result, expected) in
        transformed_f64
            .into_iter()
            .zip([2.0_f64, 1.0, std::f64::consts::PI.sqrt()])
    {
        assert_report(result, expected, 1.0e-6);
    }
    let singular_calls = AtomicUsize::new(0);
    let singular_f64 = WeightedQuadraturePlan::breakpoints(
        unit_f64,
        &[0.5],
        scioxide_integrate::WeightedQuadratureOptions::new(
            SubdivisionOptions::default()
                .with_tolerance(tolerance(1e-6))
                .with_max_subdivisions(Budget::try_from(256).unwrap()),
        ),
    )
    .unwrap()
    .integrate(|x: f64| {
        singular_calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0 / (x - 0.5).abs().sqrt())
    })
    .unwrap();
    assert_f64_result(
        singular_f64,
        2.0 * std::f64::consts::SQRT_2,
        2.0e-6,
        singular_calls.load(Ordering::Relaxed),
    );

    let de_f32 = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: unit_f32,
            singular_points: &[],
        },
        LevelOptions::default().with_tolerance(tolerance(5.0e-5_f32)),
    )
    .unwrap();
    let transformed_f32 = [
        de_f32.integrate(|point: scioxide_integrate::DoubleExponentialPoint<f32>| {
            let x = point.coordinate();
            Ok(1.0 / x.sqrt())
        }),
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::HalfLine { lower: 0.0 },
            de_f32.options().clone(),
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f32>| {
            let x = point.coordinate();
            Ok((-x).exp())
        }),
        DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::RealLine,
            de_f32.options().clone(),
        )
        .unwrap()
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f32>| {
            let x = point.coordinate();
            Ok((-x * x).exp())
        }),
    ];
    for (result, expected) in
        transformed_f32
            .into_iter()
            .zip([2.0_f32, 1.0, std::f32::consts::PI.sqrt()])
    {
        assert_report(result, expected, 5.0e-3);
    }
}

#[test]
fn oscillatory_product_cubature_and_monte_carlo_oracles_are_deterministic() {
    let two_pi_f64 = Interval::new(0.0_f64, 2.0 * std::f64::consts::PI).unwrap();
    let oscillatory_f64 = SubdivisionOptions::default()
        .with_tolerance(tolerance(1.0e-6_f64))
        .with_max_subdivisions(Budget::try_from(500).unwrap());
    let result_f64 = WeightedQuadraturePlan::fourier(
        two_pi_f64,
        &[],
        10.0,
        FourierWeight::Sine,
        scioxide_integrate::WeightedQuadratureOptions::new(oscillatory_f64)
            .with_moment_levels(Budget::try_from(16).unwrap()),
    )
    .unwrap()
    .integrate(|_: f64| Ok(1.0))
    .unwrap();
    assert_f64_result(result_f64, 0.0, 1.0e-10, 25);

    let two_pi_f32 = Interval::new(0.0_f32, 2.0 * std::f32::consts::PI).unwrap();
    let oscillatory_f32 = SubdivisionOptions::default()
        .with_tolerance(tolerance(1.0e-4_f32))
        .with_max_subdivisions(Budget::try_from(500).unwrap());
    let result_f32 = WeightedQuadraturePlan::fourier(
        two_pi_f32,
        &[],
        10.0,
        FourierWeight::Sine,
        scioxide_integrate::WeightedQuadratureOptions::new(oscillatory_f32)
            .with_moment_levels(Budget::try_from(16).unwrap()),
    )
    .unwrap()
    .integrate(|_: f32| Ok(1.0))
    .unwrap();
    assert_f32_result(result_f32, 0.0, 5.0e-4, 25);

    let rule = GaussianRulePlan::gauss_legendre(5).unwrap();
    let product = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let bounds_f64 = [Interval::new(0.0_f64, 1.0).unwrap(); 4];
    let sum_squares_f64 = |point: &[f64]| Ok(point.iter().map(|x| x * x).sum());
    let product_f64 = product.integrate(&sum_squares_f64, &bounds_f64).unwrap();
    assert_f64_result(product_f64, 4.0 / 3.0, 2.0e-13, 625);

    let bounds_f32 = [Interval::new(0.0_f32, 1.0).unwrap(); 4];
    let sum_squares_f32 = |point: &[f32]| Ok(point.iter().map(|x| x * x).sum());
    let rule_single = GaussianRulePlan::<f32>::gauss_legendre(5).unwrap();
    let product_single =
        ProductRule::new(&rule_single, scioxide_integrate::ProductOptions::default());
    let product_f32 = product_single
        .integrate(&sum_squares_f32, &bounds_f32)
        .unwrap();
    assert_f32_result(product_f32, 4.0 / 3.0, 2.0e-5, 625);

    let cubature_options = SubdivisionOptions::default()
        .with_tolerance(tolerance(1.0e-7_f64))
        .with_max_subdivisions(Budget::try_from(500).unwrap());
    let bounds_2d = [Interval::new(0.0_f64, 1.0).unwrap(); 2];
    let calls = AtomicUsize::new(0);
    let anisotropic = |point: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        let offset = point[0] - 0.2;
        Ok((1.0 + point[1]) / 400.0_f64.mul_add(offset * offset, 1.0))
    };
    let cubature = scioxide_integrate::CubaturePlan::new(&bounds_2d, cubature_options)
        .and_then(|plan| plan.integrate(&anisotropic))
        .unwrap();
    assert_f64_result(
        cubature,
        reference("anisotropic_cubature"),
        3.0e-7,
        calls.load(Ordering::Relaxed),
    );

    let bounds_3d_f64 = [Interval::new(-3.0_f64, 3.0).unwrap(); 3];
    let gaussian_3d = |point: &[f64]| {
        Ok((-(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])).exp())
    };
    let gaussian_cubature = scioxide_integrate::CubaturePlan::new(
        &bounds_3d_f64,
        SubdivisionOptions::default()
            .with_tolerance(tolerance(1.0e-6_f64))
            .with_max_subdivisions(Budget::try_from(500).unwrap()),
    )
    .and_then(|plan| plan.integrate(&gaussian_3d))
    .unwrap();
    assert_eq!(
        gaussian_cubature.1.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
    );
    assert!((gaussian_cubature.0 - reference("truncated_gaussian_3d")).abs() < 1.0e-5);
    assert!(
        gaussian_cubature
            .1
            .error_estimate()
            .is_some_and(f64::is_finite)
    );

    assert_monte_carlo::<f64>(1.0, 10_000);
    assert_monte_carlo::<f32>(1.0, 10_000);
}

fn assert_monte_carlo<S: RealScalar + Default>(expected: S, samples: usize) {
    let bounds = [Interval::new(S::ZERO, S::ONE).unwrap(); 3];
    let options = MonteCarloOptions::default().with_samples(Budget::try_from(samples).unwrap());
    let plan = MonteCarloIntegrator::new(&bounds, options).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let root = Philox4x64::from_seed(42);
    let mut value = S::ZERO;
    let report = plan
        .integrate_into(
            |point: &[S]| Ok(point.iter().fold(S::ZERO, |sum, &x| sum + x * x)),
            RandomStreams::new(&root, 0),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::SampleBudgetCompleted
    );
    assert_eq!(report.evaluations(), samples);
    let statistics = workspace.statistics()[0].as_ref().unwrap();
    assert_eq!(statistics.samples(), samples);
    assert!(statistics.standard_error().is_finite() && statistics.standard_error() > S::ZERO);
    assert!(
        (value - expected).abs()
            <= S::try_from_f64_lossy(4.0).unwrap() * statistics.standard_error()
    );
}
