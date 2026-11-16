//! Extreme finite-geometry and scale-safe reduction regressions.

#![allow(clippy::unwrap_used)]

use scioxide_poly::OrthogonalFamily;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use scioxide_core::{Budget, ExecutionContext, Interval, SciOxideError};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, ClenshawCurtis, DoubleExponentialDomain,
    DoubleExponentialIntegrator, FixedRule, GaussKronrod, GaussKronrodRule, GaussianRulePlan,
    GaussianSupport, IntegrationReport, LevelOptions, MonteCarloIntegrator, MonteCarloOptions,
    MonteCarloStatistics, NewtonCotes, NewtonCotesEndpoints, RandomStreams, Romberg,
    SubdivisionOptions,
};
use scioxide_random::{Philox4x64, SeedableRng};

fn newton_cotes(nodes: usize) -> NewtonCotes<f64> {
    NewtonCotes::new(nodes, NewtonCotesEndpoints::Closed, Budget::ONE).unwrap()
}

fn is_breakdown(error: &SciOxideError) -> bool {
    matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    )
}

/// A rule either publishes a meaningful value or reports a geometry breakdown.
fn assert_not_false_zero(result: &Result<(f64, IntegrationReport<f64>), SciOxideError>) {
    match result {
        Ok((value, report)) => {
            assert!(value.is_finite());
            assert_ne!(*value, 0.0, "a collapsed grid must not certify zero");
            assert!(
                !report
                    .error_estimate()
                    .is_some_and(|error| error == 0.0 && *value == 0.0)
            );
        }
        Err(error) => assert!(is_breakdown(error), "{error:?}"),
    }
}

fn kronrod() -> AdaptiveIntegrator<f64> {
    AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, SubdivisionOptions::default()).unwrap()
}

fn simpson() -> AdaptiveIntegrator<f64> {
    AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default()).unwrap()
}

fn romberg() -> Romberg<f64> {
    Romberg::new(LevelOptions::default()).unwrap()
}

fn double_exponential(
    domain: DoubleExponentialDomain<'_, f64>,
) -> DoubleExponentialIntegrator<f64> {
    DoubleExponentialIntegrator::new(domain, LevelOptions::default()).unwrap()
}

fn adjacent_quadratic(x: f64, lower: f64, width: f64) -> f64 {
    let normalized = (x - lower) / width;
    normalized * normalized
}

#[test]
fn adjacent_high_offset_grids_never_certify_a_collapsed_zero() {
    let lower = 2.0_f64.powi(53);
    let upper = lower.next_up();
    let width = upper - lower;
    let interval = Interval::new(lower, upper).unwrap();

    let quadratic = |x: f64| Ok(adjacent_quadratic(x, lower, width));
    let fixed_results = [
        GaussianRulePlan::gauss_legendre(5)
            .unwrap()
            .integrate(&quadratic, interval),
        GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: 0.0,
                beta: 0.0,
            },
            5,
        )
        .unwrap()
        .integrate(&quadratic, interval),
        GaussKronrod::new(GaussKronrodRule::G7K15)
            .unwrap()
            .integrate(&quadratic, interval),
        ClenshawCurtis::new(8)
            .unwrap()
            .integrate(&quadratic, interval),
        newton_cotes(3).integrate(&quadratic, interval),
        newton_cotes(2).integrate(&quadratic, interval),
    ];
    fixed_results.iter().for_each(assert_not_false_zero);

    let adaptive_results = [
        kronrod().integrate(&quadratic, interval),
        simpson().integrate(&quadratic, interval),
        romberg().integrate(&quadratic, interval),
        double_exponential(DoubleExponentialDomain::Finite {
            interval,
            singular_points: &[],
        })
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| {
            quadratic(point.coordinate())
        }),
    ];
    adaptive_results.iter().for_each(assert_not_false_zero);
}

#[test]
fn extreme_spans_apply_scale_before_final_rounding() {
    let positive = Interval::new(0.0_f64, f64::MAX).unwrap();
    let symmetric = Interval::new(-f64::MAX, f64::MAX).unwrap();
    let tiny = Interval::new(0.0_f64, f64::MIN_POSITIVE).unwrap();

    let reciprocal = |_| Ok(1.0 / f64::MAX);
    for (value, _) in [
        GaussianRulePlan::gauss_legendre(8)
            .unwrap()
            .integrate(&reciprocal, positive)
            .unwrap(),
        GaussKronrod::new(GaussKronrodRule::G10K21)
            .unwrap()
            .integrate(&reciprocal, positive)
            .unwrap(),
        kronrod().integrate(&reciprocal, positive).unwrap(),
        simpson().integrate(&reciprocal, positive).unwrap(),
        romberg().integrate(&reciprocal, positive).unwrap(),
    ] {
        assert!((value - 1.0).abs() < 2.0e-13);
    }

    let legendre = GaussianRulePlan::gauss_legendre(8).unwrap();
    let (symmetric_value, _) = legendre
        .integrate(|_| Ok(0.5 / f64::MAX), symmetric)
        .unwrap();
    assert!((symmetric_value - 1.0).abs() < 2.0e-13);

    let (tiny_value, _) = legendre.integrate(|_| Ok(f64::MAX), tiny).unwrap();
    let expected = f64::MAX * f64::MIN_POSITIVE;
    assert!((tiny_value - expected).abs() < 2.0e-13 * expected);

    let subnormal_width = f64::from_bits(1024);
    let (subnormal_value, _) = legendre
        .integrate(|_| Ok(1.0), Interval::new(0.0, subnormal_width).unwrap())
        .unwrap();
    assert!((subnormal_value - subnormal_width).abs() <= 2.0 * f64::from_bits(1));
}

#[test]
fn chebyshev_constant_measure_is_pi_on_every_valid_interval() {
    let rule = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 8).unwrap();
    for interval in [
        Interval::new(-1.0, 1.0).unwrap(),
        Interval::new(2.0, 9.0).unwrap(),
        Interval::new(-1.0e200, 1.0e200).unwrap(),
    ] {
        let (value, _) = rule
            .integrate_weighted(|_| Ok(1.0), GaussianSupport::Interval(interval))
            .unwrap();
        assert!((value - std::f64::consts::PI).abs() < 2.0e-14);
    }
}

#[test]
fn large_shift_laguerre_is_meaningful_or_a_breakdown_before_callbacks() {
    let rule = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 8).unwrap();
    let meaningful_lower = 2.0_f64.powi(48);
    let (meaningful, _) = rule
        .integrate_weighted(
            |x| Ok(x - meaningful_lower),
            GaussianSupport::HalfLine(meaningful_lower),
        )
        .unwrap();
    assert!((meaningful - 1.0).abs() < 5.0e-2);

    let calls = AtomicUsize::new(0);
    let collapsed_lower = 2.0_f64.powi(53);
    let collapsed = rule.integrate_weighted(
        |x| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(x - collapsed_lower)
        },
        GaussianSupport::HalfLine(collapsed_lower),
    );
    assert!(is_breakdown(&collapsed.err().unwrap()));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn double_exponential_rejects_a_rounded_semi_infinite_center_before_callback() {
    let calls = AtomicUsize::new(0);
    let lower = 2.0_f64.powi(53);
    let result = double_exponential(DoubleExponentialDomain::HalfLine { lower }).integrate(|_| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    });
    assert!(is_breakdown(&result.unwrap_err()));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn newton_cotes_validates_all_callback_coordinates_before_sampling() {
    let lower = 2.0_f64.powi(53);
    let upper = lower.next_up();
    let interval = Interval::new(lower, upper).unwrap();
    let observed = Mutex::new(Vec::new());
    let result = newton_cotes(3).integrate(
        |x| {
            observed.lock().unwrap().push(x);
            Ok(1.0)
        },
        interval,
    );
    assert!(is_breakdown(&result.err().unwrap()));
    assert!(observed.lock().unwrap().is_empty());
}

fn monte_carlo(
    bounds: &[Interval<f64>],
    samples: usize,
    stream: u64,
    f: &(dyn Fn(&[f64]) -> Result<f64, SciOxideError> + Sync),
) -> (f64, MonteCarloStatistics<f64>) {
    let options = MonteCarloOptions::default().with_samples(Budget::try_from(samples).unwrap());
    let plan = MonteCarloIntegrator::new(bounds, options).unwrap();
    let mut workspace = plan
        .workspace(1, ExecutionContext::sequential_native())
        .unwrap();
    let root = Philox4x64::from_seed(337);
    let mut value = 0.0;
    let report = plan
        .integrate_into(
            f,
            RandomStreams::new(&root, stream),
            &mut value,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    (value, workspace.statistics()[0].clone().unwrap())
}

#[test]
fn monte_carlo_preserves_mixed_volume_and_high_offset_dispersion() {
    let mixed_bounds =
        [1.0e200, 1.0e200, 1.0e-200, 1.0e-200].map(|width| Interval::new(0.0, width).unwrap());
    let (constant, statistics) = monte_carlo(&mixed_bounds, 128, 0, &|_: &[f64]| Ok(1.0));
    assert!((constant - 1.0).abs() < 2.0e-14);
    assert_eq!(statistics.standard_error(), 0.0);

    let lower = 2.0_f64.powi(53);
    let width = 4096.0;
    let upper = lower + width;
    // Positive floats order like their bit patterns, so integer extrema track them.
    let minimum = AtomicU64::new(upper.to_bits());
    let maximum = AtomicU64::new(lower.to_bits());
    let normalized = |point: &[f64]| {
        let coordinate = point[0];
        assert!(coordinate >= lower && coordinate < upper);
        minimum.fetch_min(coordinate.to_bits(), Ordering::Relaxed);
        maximum.fetch_max(coordinate.to_bits(), Ordering::Relaxed);
        Ok((coordinate - lower) / width)
    };
    let (value, mut statistics) = monte_carlo(
        &[Interval::new(lower, upper).unwrap()],
        4096,
        1,
        &normalized,
    );
    assert!(minimum.load(Ordering::Relaxed) < maximum.load(Ordering::Relaxed));
    assert!((value / width - 0.5).abs() < 0.03);
    assert!(statistics.standard_error().is_finite());
    assert!(statistics.standard_error() > 0.0);
    assert!(
        statistics.confidence_interval().unwrap().lower()
            < statistics.confidence_interval().unwrap().upper()
    );
}
