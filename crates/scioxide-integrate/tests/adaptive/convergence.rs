//! Independent analytic regressions for convergence and scalar-derived defaults.
#![allow(clippy::unwrap_used)]
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use scioxide_core::{
    Budget, BudgetKind, ConvergenceReason, ErrorTolerance, Interval, NormSelection,
    SciOxideErrorView,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, DoubleExponentialDomain, DoubleExponentialIntegrator,
    IntegrationTermination, LevelOptions, Romberg, SubdivisionOptions,
};

type Case = (&'static str, fn(f64) -> f64, f64);
fn peaks(x: f64) -> f64 {
    (10.0 * (x - 0.2)).cosh().recip().powi(2)
        + (100.0 * (x - 0.4)).cosh().recip().powi(4)
        + (1000.0 * (x - 0.6)).cosh().recip().powi(6)
}
fn peak_integral() -> f64 {
    // u = tanh(k(x-c)): primitives are u, u-u^3/3, u-2u^3/3+u^5/5.
    let primitive = |u: f64, power| match power {
        2 => u,
        4 => u - u.powi(3) / 3.0,
        _ => u - 2.0 * u.powi(3) / 3.0 + u.powi(5) / 5.0,
    };
    [(10.0, 0.2, 2), (100.0, 0.4, 4), (1000.0, 0.6, 6)]
        .into_iter()
        .map(|(k, c, p)| {
            (primitive((k * (1.0_f64 - c)).tanh(), p) - primitive((-k * c).tanh(), p)) / k
        })
        .sum()
}
#[test]
fn romberg_and_simpson_do_not_accept_aliased_or_underestimated_results() {
    let pi = core::f64::consts::PI;
    let cases: [Case; 8] = [
        (
            "periodic",
            |x| 2.0 / (2.0 + (10.0 * core::f64::consts::PI * x).sin()),
            2.0 / 3.0_f64.sqrt(),
        ),
        (
            "aliased product",
            |x| {
                4.0 * core::f64::consts::PI.powi(2)
                    * x
                    * (20.0 * core::f64::consts::PI * x).sin()
                    * (2.0 * core::f64::consts::PI * x).cos()
            },
            -20.0 * pi / 99.0,
        ),
        (
            "oscillation",
            |x| (100.0 * x).cos(),
            100.0_f64.sin() / 100.0,
        ),
        ("peaks", peaks, peak_integral()),
        ("step", |x| if x < 0.3 { 0.0 } else { 1.0 }, 0.7),
        ("sqrt", f64::sqrt, 2.0 / 3.0),
        (
            "sqrt log",
            |x| if x == 0.0 { 0.0 } else { x.sqrt() * x.ln() },
            -4.0 / 9.0,
        ),
        ("smooth", f64::exp, 1.0_f64.exp() - 1.0),
    ];
    let unit = Interval::new(0.0, 1.0).unwrap();
    for tolerance in [1e-6, 1e-10] {
        let policy = ErrorTolerance::scalar(tolerance, tolerance, NormSelection::Infinity).unwrap();
        let romberg_solver = Romberg::new(
            LevelOptions::default()
                .with_tolerance(policy.clone())
                .with_max_levels(Budget::try_from(20).unwrap()),
        )
        .unwrap();
        let simpson_solver = AdaptiveIntegrator::new(
            AdaptiveRule::Simpson,
            SubdivisionOptions::default()
                .with_tolerance(policy)
                .with_max_subdivisions(Budget::try_from(5000).unwrap()),
        )
        .unwrap();
        for (name, function, expected) in cases {
            for romberg in [false, true] {
                let calls = AtomicUsize::new(0);
                let f = |x| {
                    calls.fetch_add(1, Relaxed);
                    Ok(function(x))
                };
                let (value, report) = if romberg {
                    romberg_solver.integrate(&f, unit).unwrap()
                } else {
                    simpson_solver.integrate(&f, unit).unwrap()
                };
                assert_eq!(report.evaluations(), calls.load(Relaxed));
                let true_error = (value - expected).abs();
                let threshold = tolerance * (1.0 + expected.abs());
                assert!(
                    !report.is_success() || true_error <= threshold,
                    "{name}, romberg={romberg}, tol={tolerance}: {report:?}, true error={true_error}"
                );
                assert!(
                    true_error
                        <= report.error_estimate().unwrap() + 8.0 * f64::EPSILON * expected.abs(),
                    "{name}, romberg={romberg}, tol={tolerance}: {report:?}, true error={true_error}"
                );
                if matches!(
                    name,
                    "periodic" | "aliased product" | "oscillation" | "smooth"
                ) {
                    assert!(report.is_success(), "{name}: {report:?}");
                }
            }
        }
    }
}

macro_rules! defaults {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let subdivision = SubdivisionOptions::<$s>::default();
            let romberg = LevelOptions::<$s>::default();
            let transform = LevelOptions::<$s>::default();
            for policy in [
                subdivision.tolerance(),
                romberg.tolerance(),
                transform.tolerance(),
            ] {
                assert!(policy.threshold(0, 1, 1.0).unwrap() > 50.0 * <$s>::EPSILON);
                assert_eq!(policy.relative(), <$s>::EPSILON.sqrt());
            }
            let sine = |x: $s| Ok(x.sin());
            let exponential = |x: $s| Ok(x.exp());
            let pi = core::f64::consts::PI as $s;
            let sine_interval = Interval::new(0.0, pi).unwrap();
            let unit = Interval::new(0.0, 1.0).unwrap();
            let kronrod =
                AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, subdivision.clone()).unwrap();
            for (result, expected) in [
                (kronrod.integrate(&sine, sine_interval), 2.0),
                (
                    kronrod.integrate(&exponential, unit),
                    (1.0 as $s).exp() - 1.0,
                ),
                (
                    AdaptiveIntegrator::new(AdaptiveRule::Simpson, subdivision.clone())
                        .unwrap()
                        .integrate(&sine, sine_interval),
                    2.0,
                ),
                (
                    Romberg::new(romberg)
                        .unwrap()
                        .integrate(&sine, sine_interval),
                    2.0,
                ),
                (
                    DoubleExponentialIntegrator::new(
                        DoubleExponentialDomain::Finite {
                            interval: unit,
                            singular_points: &[],
                        },
                        transform,
                    )
                    .unwrap()
                    .integrate(
                        |point: scioxide_integrate::DoubleExponentialPoint<_>| {
                            exponential(point.coordinate())
                        },
                    ),
                    (1.0 as $s).exp() - 1.0,
                ),
            ] {
                let (value, report) = result.unwrap();
                assert!(report.is_success(), "{report:?}");
                assert!(
                    (value - expected).abs()
                        <= subdivision.tolerance().threshold(0, 1, expected).unwrap(),
                    "{value:?}: {report:?}"
                );
                assert!(
                    (value - expected).abs()
                        <= report.error_estimate().unwrap() + 8.0 * <$s>::EPSILON * expected.abs(),
                    "{value:?}: {report:?}"
                );
            }
            let f = (|x: &[$s]| Ok((x[0] + x[1]).exp()));
            let result = scioxide_integrate::CubaturePlan::new(
                &[unit, unit],
                scioxide_integrate::SubdivisionOptions::default(),
            )
            .and_then(|plan| plan.integrate(&f))
            .unwrap();
            let expected = ((1.0 as $s).exp() - 1.0).powi(2);
            assert!(result.1.is_success(), "{result:?}");
            assert!(
                (result.0 - expected).abs()
                    <= subdivision.tolerance().threshold(0, 1, expected).unwrap()
            );
            assert!(
                (result.0 - expected).abs()
                    <= result.1.error_estimate().unwrap() + 8.0 * <$s>::EPSILON * expected.abs(),
                "{result:?}"
            );
        }
    };
}
defaults!(f32_defaults_converge, f32);
defaults!(f64_defaults_converge, f64);

macro_rules! de {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let unit = DoubleExponentialDomain::Finite {
                interval: Interval::new(0.0, 1.0).unwrap(),
                singular_points: &[],
            };
            let solver =
                DoubleExponentialIntegrator::<$s>::new(unit, LevelOptions::default()).unwrap();
            let half_line = |lower| DoubleExponentialDomain::HalfLine { lower };
            let real_line = DoubleExponentialDomain::RealLine;
            for (result, expected) in [
                (
                    solver.integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                        let x = point.coordinate();
                        Ok(x.exp())
                    }),
                    (1.0 as $s).exp() - 1.0,
                ),
                (
                    solver.integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                        let x = point.coordinate();
                        Ok(1.0 / x.sqrt())
                    }),
                    2.0,
                ),
                (
                    DoubleExponentialIntegrator::new(half_line(0.0), solver.options().clone())
                        .unwrap()
                        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                            let x = point.coordinate();
                            Ok((-x).exp())
                        }),
                    1.0,
                ),
                (
                    DoubleExponentialIntegrator::new(half_line(1.0), solver.options().clone())
                        .unwrap()
                        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                            let x = point.coordinate();
                            Ok((1.0 - x).exp())
                        }),
                    1.0,
                ),
                (
                    DoubleExponentialIntegrator::new(half_line(0.0), solver.options().clone())
                        .unwrap()
                        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                            let x = point.coordinate();
                            Ok(1.0 / (1.0 + x * x))
                        }),
                    (core::f64::consts::PI as $s) / 2.0,
                ),
                (
                    DoubleExponentialIntegrator::new(real_line, solver.options().clone())
                        .unwrap()
                        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                            let x = point.coordinate();
                            Ok((-x * x).exp())
                        }),
                    (core::f64::consts::PI as $s).sqrt(),
                ),
                (
                    DoubleExponentialIntegrator::new(real_line, solver.options().clone())
                        .unwrap()
                        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                            let x = point.coordinate();
                            Ok(1.0 / (1.0 + x * x))
                        }),
                    core::f64::consts::PI as $s,
                ),
            ] {
                let (value, report) = result.unwrap();
                assert_eq!(
                    report.termination(),
                    IntegrationTermination::HeuristicToleranceReached,
                    "{report:?}"
                );
                assert!(report.is_success());
                assert!(
                    (value - expected).abs()
                        <= solver
                            .options()
                            .tolerance()
                            .threshold(0, 1, expected)
                            .unwrap(),
                    "{value:?}: {report:?}"
                );
            }
            let (_, spike) = solver
                .integrate(|point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                    let x = point.coordinate();
                    Ok(1.0 / ((x - 0.5).powi(2) + 1e-6))
                })
                .unwrap();
            assert!(!spike.is_success(), "{spike:?}");
            let zero_solver = DoubleExponentialIntegrator::new(
                half_line(0.0),
                LevelOptions::default().with_max_levels(Budget::try_from(4).unwrap()),
            )
            .unwrap();
            let (_, zero) = zero_solver
                .integrate(|_: scioxide_integrate::DoubleExponentialPoint<$s>| Ok(0.0))
                .unwrap();
            assert_eq!(zero.termination(), IntegrationTermination::RefinementLimit);
            assert!(!zero.is_success());
        }
    };
}
de!(f32_de_convergence, f32);
de!(f64_de_convergence, f64);

#[test]
fn romberg_explicit_unreachable_f32_tolerance_stagnates() {
    let solver =
        Romberg::new(LevelOptions::default().with_tolerance(
            ErrorTolerance::scalar(1e-8_f32, 1e-8, NormSelection::Infinity).unwrap(),
        ))
        .unwrap();
    let (_, report) = solver
        .integrate(
            |x: f32| Ok(x.sin()),
            Interval::new(0.0, core::f32::consts::PI).unwrap(),
        )
        .unwrap();
    assert_eq!(report.termination(), IntegrationTermination::Stagnation);
    assert!(!report.is_success());
    assert!(report.error_estimate().unwrap() > 1e-8 + 1e-8 * 2.0);
}

macro_rules! affine_de {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            use scioxide_core::ExecutionContext;

            let unit = Interval::new(0.0, 1.0).unwrap();
            for domain in 0..4 {
                let solver = DoubleExponentialIntegrator::<$s>::new(
                    match domain {
                        0 => DoubleExponentialDomain::Finite {
                            interval: unit,
                            singular_points: &[],
                        },
                        1 => DoubleExponentialDomain::Finite {
                            interval: unit,
                            singular_points: &[0.5],
                        },
                        2 => DoubleExponentialDomain::HalfLine { lower: 0.0 },
                        _ => DoubleExponentialDomain::RealLine,
                    },
                    LevelOptions::default(),
                )
                .unwrap();
                let tolerance = solver.options().tolerance();
                let mut workspace = solver.workspace(2).unwrap();

                let calls = AtomicUsize::new(0);
                let function = |point: scioxide_integrate::DoubleExponentialPoint<$s>| {
                    let x = point.coordinate();
                    calls.fetch_add(1, Relaxed);
                    let value = match domain {
                        0 => x.exp(),
                        1 => {
                            let distance = match point.nearest_endpoint() {
                                Some((0.5, distance)) => distance,
                                _ => (x - 0.5).abs(),
                            };
                            1.0 / distance.sqrt()
                        }
                        2 => (-x).exp(),
                        _ => (-x * x).exp(),
                    };
                    Ok([value, -2.0 * value])
                };
                let mut output = [99.0; 2];
                let report = solver
                    .integrate_into(
                        &function,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert_eq!(report.evaluations(), calls.load(Relaxed));
                let errors = workspace.component_errors();
                assert_eq!(
                    report.termination(),
                    IntegrationTermination::HeuristicToleranceReached,
                    "domain={domain}, output={output:?}, errors={errors:?}"
                );
                let expected = match domain {
                    0 => (1.0 as $s).exp() - 1.0,
                    1 => 2.0 * (2.0 as $s).sqrt(),
                    2 => 1.0,
                    _ => (core::f64::consts::PI as $s).sqrt(),
                };
                for (c, expected) in [expected, -2.0 * expected].into_iter().enumerate() {
                    let true_error = (output[c] - expected).abs();
                    assert!(
                        true_error <= tolerance.threshold(c, 2, expected.abs()).unwrap(),
                        "domain={domain}, output={output:?}, expected={expected}, errors={errors:?}"
                    );
                }
            }
        }
    };
}
affine_de!(f32_affine_de_convergence, f32);
affine_de!(f64_affine_de_convergence, f64);

#[test]
fn heuristic_success_is_shared_but_failure_stops_remain_unsuccessful() {
    for termination in [
        IntegrationTermination::FixedRuleCompleted,
        IntegrationTermination::ToleranceReached,
        IntegrationTermination::HeuristicToleranceReached,
        IntegrationTermination::SampleBudgetCompleted,
        IntegrationTermination::DegenerateDomain,
    ] {
        assert!(termination.is_success());
    }
    for termination in [
        IntegrationTermination::Stagnation,
        IntegrationTermination::GeometryFailure,
        IntegrationTermination::BudgetLimit(BudgetKind::Evaluation),
        IntegrationTermination::BudgetLimit(BudgetKind::Subdivision),
        IntegrationTermination::RefinementLimit,
    ] {
        assert!(!termination.is_success());
    }
}

#[test]
fn exploration_is_required_and_charged_to_complete_work_units() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    let limit = |budget: usize| Some(Budget::try_from(budget).unwrap());
    for romberg in [false, true] {
        let cases: &[(usize, usize, bool)] = if romberg {
            // Seventeen grid callbacks precede the witness; level five needs sixteen more.
            &[
                (2, 2, false),
                (32, 18, false),
                (33, 18, false),
                (34, 34, true),
            ]
        } else {
            &[
                (9, 0, false),
                (10, 10, false),
                (129, 122, false),
                (130, 130, true),
            ]
        };
        for &(budget, expected_calls, success) in cases {
            let calls = AtomicUsize::new(0);
            let function = |_: f64| {
                calls.fetch_add(1, Relaxed);
                Ok(1.0)
            };
            let result = if romberg {
                Romberg::new(LevelOptions::default().with_max_evaluations(limit(budget)))
                    .unwrap()
                    .integrate(&function, interval)
            } else {
                AdaptiveIntegrator::new(
                    AdaptiveRule::Simpson,
                    SubdivisionOptions::default().with_max_evaluations(limit(budget)),
                )
                .unwrap()
                .integrate(&function, interval)
            };
            assert_eq!(calls.load(Relaxed), expected_calls);
            if expected_calls == 0 {
                assert!(matches!(
                    result.unwrap_err().view(),
                    SciOxideErrorView::ConvergenceFailure {
                        reason: ConvergenceReason::EvaluationLimit,
                        ..
                    }
                ));
                continue;
            }
            let (value, report) = result.unwrap();
            assert_eq!(report.evaluations(), expected_calls);
            assert_eq!(report.is_success(), success);
            assert_eq!(value, 1.0);
            if !success {
                assert_eq!(
                    report.termination(),
                    IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
                );
            }
        }
    }
}

macro_rules! dyadic_aliasing {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            // Every Romberg and Simpson sample is dyadic. sin²(2^k πx) vanishes,
            // up to rounding noise, on every grid of spacing at least 2^-k, and
            // x + sin²(2^k πx) looks exactly linear there. Only the off-grid
            // witness can reject the aliased value; no wrong success is allowed.
            type S = $s;
            let unit = Interval::new(0.0 as S, 1.0).unwrap();
            let tolerances: [Option<S>; 3] = [None, Some(1e-6), Some(1e-10)];
            for tolerance in tolerances {
                let levels = tolerance.map_or_else(LevelOptions::default, |t| {
                    LevelOptions::default().with_tolerance(
                        ErrorTolerance::scalar(t, t, NormSelection::Infinity).unwrap(),
                    )
                });
                let subdivisions = tolerance.map_or_else(SubdivisionOptions::default, |t| {
                    SubdivisionOptions::default().with_tolerance(
                        ErrorTolerance::scalar(t, t, NormSelection::Infinity).unwrap(),
                    )
                });
                let romberg = Romberg::new(levels).unwrap();
                let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, subdivisions).unwrap();
                let allowed = 10.0 * tolerance.unwrap_or(<S>::EPSILON.sqrt()) * 2.0;
                for k in 3..=9 {
                    let frequency = (1u32 << k) as S * core::f64::consts::PI as S;
                    for shift in [0.0 as S, 1.0] {
                        let f = |x: S| Ok(shift * x + (frequency * x).sin().powi(2));
                        let exact = 0.5 + 0.5 * shift;
                        for (method, (value, report)) in [
                            ("romberg", romberg.integrate(&f, unit).unwrap()),
                            ("simpson", simpson.integrate(&f, unit).unwrap()),
                        ] {
                            assert!(
                                !report.is_success() || (value - exact).abs() <= allowed,
                                "{method} k={k} shift={shift} {tolerance:?}: {value} {report:?}"
                            );
                        }
                    }
                }
            }
            // The originally reported pairs refine to their exact values.
            let (value, report) = Romberg::new(LevelOptions::default())
                .unwrap()
                .integrate(
                    |x: S| Ok(x + (32.0 * core::f64::consts::PI as S * x).sin().powi(2)),
                    unit,
                )
                .unwrap();
            assert!(
                report.is_success() && (value - 1.0).abs() < 1e-4,
                "{value} {report:?}"
            );
            let (value, report) =
                AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
                    .unwrap()
                    .integrate(
                        |x: S| Ok(x + (128.0 * core::f64::consts::PI as S * x).sin().powi(2)),
                        unit,
                    )
                    .unwrap();
            assert!(
                !report.is_success() || (value - 1.0).abs() < 1e-4,
                "{value} {report:?}"
            );
        }
    };
}
dyadic_aliasing!(f32_dyadic_aliasing_never_succeeds_with_a_wrong_value, f32);
dyadic_aliasing!(f64_dyadic_aliasing_never_succeeds_with_a_wrong_value, f64);

#[test]
fn smooth_witness_costs_one_callback() {
    // A resolved integrand agrees with its witness, so convergence keeps its
    // grid and exploration and pays one callback over the dyadic samples.
    let unit = Interval::new(0.0_f64, 1.0).unwrap();
    let (value, report) = Romberg::new(LevelOptions::default())
        .unwrap()
        .integrate(|x: f64| Ok(x.exp()), unit)
        .unwrap();
    assert!(report.is_success());
    assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-12);
    assert!((report.evaluations() - 2).is_power_of_two(), "{report:?}");
    let (value, report) =
        AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
            .unwrap()
            .integrate(|x: f64| Ok(x.exp()), unit)
            .unwrap();
    assert!(report.is_success());
    assert!((value - (1.0_f64.exp() - 1.0)).abs() < 1e-10);
    assert_eq!(report.evaluations(), 130);
}

#[test]
fn impossible_refinement_configurations_are_rejected_at_construction() {
    for levels in [1, 3, 5, 31] {
        let error = Romberg::<f64>::new(
            LevelOptions::default().with_max_levels(Budget::try_from(levels).unwrap()),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(format!("{error:?}").contains("options.max_levels"));
    }
    assert!(
        Romberg::<f64>::new(LevelOptions::default().with_max_levels(Budget::try_from(6).unwrap()))
            .is_ok()
    );
    for subdivisions in [1, 3, 14] {
        let error = AdaptiveIntegrator::<f64>::new(
            AdaptiveRule::Simpson,
            SubdivisionOptions::default()
                .with_max_subdivisions(Budget::try_from(subdivisions).unwrap()),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(format!("{error:?}").contains("options.max_subdivisions"));
    }
    assert!(
        AdaptiveIntegrator::<f64>::new(
            AdaptiveRule::Simpson,
            SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(15).unwrap())
        )
        .is_ok()
    );
    use scioxide_integrate::{SeriesMethod, SeriesOptions, SeriesSummation};
    let error = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default().with_order(Budget::ONE),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(format!("{error:?}").contains("order"));
    assert!(
        SeriesSummation::<f64>::new(
            SeriesMethod::WynnEpsilon,
            0,
            SeriesOptions::default().with_order(Budget::try_from(2).unwrap())
        )
        .is_ok()
    );
}

macro_rules! mass_evidence {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            let interval = Interval::new(0.0 as $s, 1.0).unwrap();
            let romberg = Romberg::<$s>::new(LevelOptions::default()).unwrap();
            let simpson =
                AdaptiveIntegrator::<$s>::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
                    .unwrap();
            let de = DoubleExponentialIntegrator::<$s>::new(
                DoubleExponentialDomain::Finite {
                    interval,
                    singular_points: &[],
                },
                LevelOptions::default().with_max_levels(Budget::try_from(10).unwrap()),
            )
            .unwrap();
            for k in 3..=6 {
                let f = |x: $s| {
                    Ok(((1_u32 << k) as $s * std::f64::consts::PI as $s * x)
                        .sin()
                        .powi(2))
                };
                let results = [
                    romberg.integrate(&f, interval).unwrap(),
                    simpson.integrate(&f, interval).unwrap(),
                    de.integrate(|p: scioxide_integrate::DoubleExponentialPoint<$s>| {
                        f(p.coordinate())
                    })
                    .unwrap(),
                ];
                for (value, report) in results {
                    let requested = <$s>::EPSILON.sqrt() * 1.5;
                    assert!(
                        !report.is_success() || (value - 0.5).abs() <= requested,
                        "k={k}: {value:?}, {report:?}"
                    );
                }
            }
            let romberg = Romberg::<$s>::new(
                LevelOptions::default().with_max_levels(Budget::try_from(6).unwrap()),
            )
            .unwrap();
            let simpson = AdaptiveIntegrator::<$s>::new(
                AdaptiveRule::Simpson,
                SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(15).unwrap()),
            )
            .unwrap();
            let (zero, report) = romberg.integrate(|_: $s| Ok(0.0 as $s), interval).unwrap();
            assert_eq!(zero, 0.0);
            assert!(!report.is_success());
            assert_eq!(report.evaluations(), 34);
            let (zero, report) = simpson.integrate(|_: $s| Ok(0.0 as $s), interval).unwrap();
            assert_eq!(zero, 0.0);
            assert!(!report.is_success());
            assert_eq!(report.subdivisions(), 15);
            let de = DoubleExponentialIntegrator::<$s>::new(
                DoubleExponentialDomain::Finite {
                    interval,
                    singular_points: &[],
                },
                LevelOptions::default().with_max_levels(Budget::try_from(4).unwrap()),
            )
            .unwrap();
            let (_, report) = de
                .integrate(|_: scioxide_integrate::DoubleExponentialPoint<$s>| Ok(0.0 as $s))
                .unwrap();
            assert!(!report.is_success());
            let (values, report) = romberg
                .integrate(|x: $s| Ok([0.0 as $s, x, x - 0.5]), interval)
                .unwrap();
            assert!(report.is_success());
            assert_eq!(report.evaluations(), 34);
            assert!((values[1] - 0.5).abs() <= 8.0 * <$s>::EPSILON);
            let (_, report) = romberg
                .integrate(|x: $s| Ok((1.0 + x) * <$s>::MIN_POSITIVE), interval)
                .unwrap();
            assert!(report.is_success());
        }
    };
}
mass_evidence!(sampled_mass_evidence_f64, f64);
mass_evidence!(sampled_mass_evidence_f32, f32);
