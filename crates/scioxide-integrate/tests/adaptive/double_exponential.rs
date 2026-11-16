//! Prepared DE tails, recovery, accounting and reserved execution.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, NormSelection, RealScalar,
};
use scioxide_integrate::{
    DoubleExponentialDomain, DoubleExponentialIntegrator, DoubleExponentialPoint,
    IntegrationTermination, LevelOptions,
};
use std::{
    collections::BTreeSet,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
};

#[test]
fn finer_boundary_evidence_reopens_skipped_shells_without_repeating_samples() {
    let coordinate = |t: f64| (core::f64::consts::FRAC_PI_2 * t.sinh()).sinh();
    let inner = coordinate(1.75);
    let outer = coordinate(2.25);
    let seen = Mutex::new(BTreeSet::new());
    let plan = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::RealLine,
        LevelOptions::default()
            .with_max_levels(Budget::try_from(4).unwrap())
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap()),
    )
    .unwrap();
    let (_, report) = plan
        .integrate(|point: DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            assert!(seen.lock().unwrap().insert(x.to_bits()));
            // The coarse outer shells are zero. The finer inner bump must reopen
            // the shell containing the second bump, including missed older nodes.
            Ok((1.0 - x.abs()).max(0.0)
                + (1.0 - (x - inner).abs()).max(0.0)
                + (1.0 - (x - outer).abs()).max(0.0))
        })
        .unwrap();
    let seen = seen.lock().unwrap();
    for expected in [inner, outer] {
        assert!(
            seen.iter()
                .any(|&bits| (f64::from_bits(bits) - expected).abs()
                    <= 8.0 * f64::EPSILON * expected.abs())
        );
    }
    assert_eq!(seen.len(), report.evaluations());
    assert!(!report.is_success());

    let limited = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::RealLine,
        plan.options()
            .clone()
            .with_max_evaluations(Some(Budget::try_from(19).unwrap())),
    )
    .unwrap();
    let (_, report) = limited
        .integrate(|point: DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok((1.0 - x.abs()).max(0.0)
                + (1.0 - (x - inner).abs()).max(0.0)
                + (1.0 - (x - outer).abs()).max(0.0))
        })
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
    );
    assert_eq!(report.evaluations(), 19);
    assert!(report.error_estimate().is_none());
    assert_eq!(
        report
            .error_components()
            .unwrap()
            .tail_diagnostics()
            .unwrap()
            .status(),
        scioxide_integrate::TailStatus::Open
    );
}

#[test]
fn an_overflowing_pruning_bound_does_not_reject_finite_cancelling_samples() {
    let plan = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::Finite {
            interval: Interval::new(0.0, 1e308).unwrap(),
            singular_points: &[],
        },
        LevelOptions::default()
            .with_max_levels(Budget::try_from(4).unwrap())
            .with_max_evaluations(Some(Budget::try_from(7).unwrap())),
    )
    .unwrap();
    let (value, report) = plan
        .integrate(|point: DoubleExponentialPoint<f64>| {
            let (endpoint, distance) = point.nearest_endpoint().unwrap();
            let fraction = distance / 1e308;
            Ok(if (1e-5..1.3e-5).contains(&fraction) {
                if endpoint == 0.0 { 4000.0 } else { -4000.0 }
            } else {
                0.0
            })
        })
        .unwrap();
    assert!(value.abs() <= f64::EPSILON);
    assert_eq!(report.evaluations(), 7);
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
    );
    // No refined discretization estimate exists at this budget, but the
    // finite absolute mass still supplies an arithmetic diagnostic.
    assert!(report.error_estimate().is_none());
    assert!(
        report
            .error_components()
            .unwrap()
            .roundoff()
            .is_some_and(f64::is_finite)
    );
}

#[test]
fn coarse_batches_and_reused_workspaces_respect_every_small_evaluation_budget() {
    for budget in 1..=32 {
        let plan = DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::RealLine,
            LevelOptions::default().with_max_evaluations(Some(Budget::try_from(budget).unwrap())),
        )
        .unwrap();
        let mut workspace = plan.workspace(1).unwrap();
        for _ in 0..2 {
            let calls = AtomicUsize::new(0);
            let mut out = -7.0;
            let report = plan
                .integrate_into(
                    |point: DoubleExponentialPoint<f64>| {
                        calls.fetch_add(1, Relaxed);
                        Ok((-point.coordinate().powi(2)).exp())
                    },
                    &mut out,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(report.evaluations(), calls.load(Relaxed));
            assert!(report.evaluations() <= budget);
            assert!(out.is_finite());
            assert_eq!(
                report.termination(),
                IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
            );
        }
    }
}

/// A finite interval, its integrand and the exact integral.
type Case = (Interval<f64>, fn(f64) -> f64, f64);

fn solve<S: RealScalar + Default>(
    domain: DoubleExponentialDomain<'_, S>,
    tolerance: Option<(S, S)>,
    f: impl Fn(S) -> S + Sync,
) -> (S, scioxide_integrate::IntegrationReport<S>) {
    let mut options = LevelOptions::default();
    if let Some((absolute, relative)) = tolerance {
        options = options.with_tolerance(
            ErrorTolerance::scalar(absolute, relative, NormSelection::Infinity).unwrap(),
        );
    }
    DoubleExponentialIntegrator::new(domain, options)
        .unwrap()
        .integrate(|point: DoubleExponentialPoint<S>| Ok(f(point.coordinate())))
        .unwrap()
}

#[test]
fn growing_pre_asymptotic_changes_do_not_count_as_slow_decay() {
    // Each integrand needs several levels of feature discovery whose changes
    // grow; those levels once exhausted the slow-decay allowance at level four.
    let unit = Interval::new(0.0, 1.0).unwrap();
    let finite = |interval| DoubleExponentialDomain::Finite {
        interval,
        singular_points: &[],
    };
    let pi = core::f64::consts::PI;
    let cases: [Case; 4] = [
        (
            Interval::new(0.0, 100.0).unwrap(),
            |x| if x == 0.0 { 1.0 } else { x.sin() / x },
            1.562_225_466_889_056_3,
        ),
        (
            unit,
            |x| (32.0 * core::f64::consts::PI * x).sin().powi(2),
            0.5,
        ),
        (
            unit,
            |x| 1.0 / (1e-4 + (x - 0.3) * (x - 0.3)),
            100.0 * (70.0_f64.atan() + 30.0_f64.atan()),
        ),
        (
            unit,
            |x| (-((x - 0.37) / 0.01).powi(2)).exp(),
            0.01 * pi.sqrt(),
        ),
    ];
    for (interval, f, expected) in cases {
        let (value, report) = solve(finite(interval), Some((1e-10, 1e-10)), f);
        assert_eq!(
            report.termination(),
            IntegrationTermination::HeuristicToleranceReached,
            "{expected}: {report:?}"
        );
        assert!(
            (value - expected).abs() <= 1e-9 * expected.abs(),
            "{value} vs {expected}"
        );
    }
    // Relative-only control of a shifted Gaussian discovers its peak.
    let (value, report) = solve(
        DoubleExponentialDomain::RealLine,
        Some((0.0, 1e-10)),
        |x: f64| (-(x - 30.0) * (x - 30.0)).exp(),
    );
    assert!(report.is_success(), "{report:?}");
    assert!((value - pi.sqrt()).abs() < 1e-9);
}

#[test]
fn negligible_sampled_flanks_cannot_meet_an_absolute_tolerance() {
    // The level-three mesh straddles this narrow peak and samples only its
    // flanks, about 3e-20; an absolute tolerance alone once accepted that mass.
    for tolerance in [None, Some((1e-6, 1e-6)), Some((1e-12, 1e-12))] {
        let (value, report) = solve(DoubleExponentialDomain::RealLine, tolerance, |x: f64| {
            (-(x - 30.0) * (x - 30.0)).exp()
        });
        assert!(report.is_success(), "{tolerance:?}: {report:?}");
        assert!(
            (value - core::f64::consts::PI.sqrt()).abs() < 1e-6,
            "{tolerance:?}: {value}"
        );
    }
    // A genuinely small integral still converges under the default tolerance.
    let (value, report) = solve(DoubleExponentialDomain::RealLine, None, |x: f64| {
        1e-12 * (-x * x).exp()
    });
    assert!(report.is_success(), "{report:?}");
    assert!(
        (value / (1e-12 * core::f64::consts::PI.sqrt()) - 1.0).abs() < 1e-3,
        "{value} {report:?}"
    );
}

type Integrand<'a, S> = dyn Fn(S) -> S + Sync + 'a;

#[test]
fn nested_sums_near_the_scalar_maximum_scale_before_rounding() {
    fn check<S: RealScalar + Default>(c: S) {
        let unit = Interval::new(S::ZERO, S::ONE).unwrap();
        let root_pi = S::PI.sqrt();
        let constant = |_: S| c;
        let decaying = |x: S| c * (-x).exp();
        let gaussian = |x: S| c * (-x * x).exp() / root_pi;
        let cases: [(DoubleExponentialDomain<'_, S>, &Integrand<'_, S>); 3] = [
            (
                DoubleExponentialDomain::Finite {
                    interval: unit,
                    singular_points: &[],
                },
                &constant,
            ),
            (
                DoubleExponentialDomain::HalfLine { lower: S::ZERO },
                &decaying,
            ),
            (DoubleExponentialDomain::RealLine, &gaussian),
        ];
        for (domain, f) in cases {
            let (value, report) = solve(domain, None, f);
            assert!(report.is_success(), "{report:?}");
            assert!((value / c - S::ONE).abs() < S::SQRT_EPSILON, "{value:?}");
        }
    }
    check(1e308_f64);
    check(1e38_f32);
    let (value, report) = solve(
        DoubleExponentialDomain::Finite {
            interval: Interval::new(-1.0, 1.0).unwrap(),
            singular_points: &[],
        },
        Some((0.0, 1e-10)),
        |x: f64| 1e307 / (1.0 + 25.0 * x * x),
    );
    assert!(report.is_success(), "{report:?}");
    assert!((value / (1e307 * 0.4 * 5_f64.atan()) - 1.0).abs() < 1e-10);
}
