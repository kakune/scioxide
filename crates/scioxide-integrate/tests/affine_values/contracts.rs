//! Independent whole-value accuracy, failure, and resource contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_integrate::*;
use scioxide_random::SeedableRng;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
fn run<T>(operation: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(589),
        scioxide_core::algorithm_id!("integrate", "affine-test", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    operation(&mut context)
}

macro_rules! precision {
    ($name:ident,$s:ty,$epsilon:expr) => {
        #[test]
        fn $name() {
            let tolerance =
                ErrorTolerance::scalar($epsilon, $epsilon, NormSelection::Infinity).unwrap();
            let interval = Interval::new(0.0 as $s, 1.0).unwrap();
            let options = SubdivisionOptions::default()
                .with_max_subdivisions(Budget::try_from(16).unwrap())
                .with_tolerance(tolerance.clone());
            let sample = Complex::new(0.0 as $s, 0.0);
            let mut output = sample;
            let function = |x: $s| Ok(Complex::new(x * x, x));
            let gk = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options.clone()).unwrap();
            let mut workspace = gk.workspace(2).unwrap();
            assert!(
                gk.integrate_into(
                    &function,
                    interval,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .is_ok()
            );
            assert!((output.re - 1.0 / 3.0).abs() < $epsilon);
            assert!((output.im - 0.5).abs() < $epsilon);
            let simpson = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options).unwrap();
            let mut workspace = simpson.workspace(2).unwrap();
            assert!(
                simpson
                    .integrate_into(
                        &function,
                        interval,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .is_ok()
            );
            assert!((output.re - 1.0 / 3.0).abs() < $epsilon);
            let romberg =
                Romberg::new(LevelOptions::default().with_tolerance(tolerance.clone())).unwrap();
            let mut workspace = romberg.workspace(2).unwrap();
            assert!(
                romberg
                    .integrate_into(
                        &function,
                        interval,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .is_ok()
            );
            assert!((output.re - 1.0 / 3.0).abs() < $epsilon);
            let rule = GaussianRulePlan::new(scioxide_poly::OrthogonalFamily::Laguerre, 4).unwrap();
            let mut fixed = rule.workspace(2).unwrap();
            assert!(
                run(|context| rule.integrate_weighted_into(
                    |x: $s| Ok(Complex::new(x * x, x)),
                    GaussianSupport::HalfLine(1.0),
                    &mut output,
                    &mut fixed,
                    context
                ))
                .unwrap()
                .is_success()
            );
            // E[(1+X)^2]=5, E[1+X]=2 for unit exponential X.
            assert!((output.re - 5.0).abs() < 20.0 * $epsilon);
            assert!((output.im - 2.0).abs() < 20.0 * $epsilon);
            let de = DoubleExponentialIntegrator::new(
                DoubleExponentialDomain::Finite {
                    interval,
                    singular_points: &[0.0, 0.0, 1.0, 1.0],
                },
                LevelOptions::default().with_tolerance(tolerance.clone()),
            )
            .unwrap();
            let mut workspace = de.workspace(2).unwrap();
            assert!(
                de.integrate_into(
                    |point: scioxide_integrate::DoubleExponentialPoint<_>| function(
                        point.coordinate()
                    ),
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )
                .is_ok()
            );
            assert!((output.re - 1.0 / 3.0).abs() < 10.0 * $epsilon);
            assert!((output.im - 0.5).abs() < 10.0 * $epsilon);
            let function = |x: &[$s], out: &mut Complex<$s>| {
                *out = Complex::new(x[0] * x[1], x[0] + x[1]);
                Ok::<(), SciOxideError>(())
            };
            let bounds = [interval; 2];
            let options =
                SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(8).unwrap());
            let cube = CubaturePlan::new(&bounds, options).unwrap();
            let mut cubature = cube.workspace(2).unwrap();
            assert!(
                run(|context| cube.integrate_into(
                    |point| {
                        let mut lane = (sample).clone();
                        scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                            *v = <_ as scioxide_core::RealScalar>::NAN
                        });
                        {
                            let result: Result<(), SciOxideError> = (&function)(point, &mut lane);
                            result?;
                        }
                        Ok(lane)
                    },
                    &mut output,
                    &mut cubature,
                    context
                ))
                .unwrap()
                .is_success()
            );
            assert!((output.re - 0.25).abs() < $epsilon);
            assert!((output.im - 1.0).abs() < $epsilon);
        }
    };
}
precision!(f32_complex_families, f32, 5e-5_f32);
precision!(f64_complex_families, f64, 1e-9_f64);

#[test]
fn adaptive_callback_failures_never_publish_and_are_counted() {
    let options = SubdivisionOptions::default()
        .with_max_subdivisions(Budget::try_from(8).unwrap())
        .with_tolerance(ErrorTolerance::scalar(1e-9, 1e-9, NormSelection::Infinity).unwrap());
    let solver = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options).unwrap();
    let mut workspace = solver.workspace(2).unwrap();
    let interval = Interval::new(0.0, 1.0).unwrap();
    let mut output = [91.0, 92.0];
    for failure in 1..=15 {
        let calls = AtomicUsize::new(0);
        run(|context| {
            let accepted = context.accepted_state();
            let result = solver.integrate_into(
                |x: f64| {
                    if calls.fetch_add(1, Ordering::Relaxed) + 1 == failure {
                        return Err(SciOxideError::invalid_input(
                            scioxide_core::argument!("test_callback"),
                            InvalidInputReason::Inconsistent,
                            "injected failure",
                        ));
                    }
                    Ok([x, x * x])
                },
                interval,
                &mut output,
                &mut workspace,
                &mut *context,
            );
            assert!(result.is_err());
            assert_eq!(calls.load(Ordering::Relaxed), failure);
            assert_eq!(
                context.counts().nested().provider(EvaluationRole::Function),
                failure
            );
            assert_eq!(context.accepted_state(), accepted);
        });
        assert_eq!(output, [91.0, 92.0]);
        assert!(workspace.component_errors().is_none());
    }
    let mut insufficient = AdaptiveIntegrator::new(
        AdaptiveRule::GaussKronrod,
        SubdivisionOptions::default().with_max_subdivisions(Budget::ONE),
    )
    .unwrap()
    .workspace(2)
    .unwrap();
    let calls = AtomicUsize::new(0);
    let error = solver
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok([x, x])
            },
            interval,
            &mut output,
            &mut insufficient,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(
        insufficient.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(output, [91.0, 92.0]);
}

#[test]
fn de_nested_grids_do_not_repeat_nodes_and_nondecaying_tails_remain_unestimated() {
    let points = Mutex::new(BTreeSet::new());
    let function = |x: f64| {
        assert!(points.lock().unwrap().insert(x.to_bits()), "repeated {x}");
        Ok((-x * x).exp())
    };
    let integrator = DoubleExponentialIntegrator::new(
        DoubleExponentialDomain::RealLine,
        LevelOptions::default(),
    )
    .unwrap();
    let (value, report) = integrator
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<_>| {
            function(point.coordinate())
        })
        .unwrap();
    assert_eq!(report.evaluations(), points.lock().unwrap().len());
    assert!((value - core::f64::consts::PI.sqrt()).abs() < 1e-8);
    let (_, report) = integrator
        .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
            let x = point.coordinate();
            Ok(x)
        })
        .unwrap();
    assert_eq!(report.termination(), IntegrationTermination::Stagnation);
    assert!(report.error_components().unwrap().tail().is_none());
}

#[test]
fn de_refinement_levels_retain_accurate_complete_estimates() {
    for levels in 4..=8 {
        let options = LevelOptions::default()
            .with_tolerance(ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap())
            .with_max_levels(Budget::try_from(levels).unwrap());
        let (value, report) =
            DoubleExponentialIntegrator::new(DoubleExponentialDomain::RealLine, options)
                .unwrap()
                .integrate(|point: scioxide_integrate::DoubleExponentialPoint<f64>| {
                    let x = point.coordinate();
                    Ok((-x * x).exp())
                })
                .unwrap();
        if levels >= 7 {
            // Resolved tails are not resampled at every finer mesh.
            assert!(report.evaluations() < 385);
            assert!((value - core::f64::consts::PI.sqrt()).abs() < 1e-10);
            assert_eq!(
                report.termination(),
                IntegrationTermination::HeuristicToleranceReached
            );
        } else if levels == 6 {
            // The guarded change still exceeds the request at this level.
            assert_eq!(
                report.termination(),
                IntegrationTermination::RefinementLimit
            );
        }
        assert!(value.is_finite());
    }
}

#[test]
fn empty_values_do_not_claim_zero_measure_and_rank_zero_still_evaluates() {
    let calls = AtomicUsize::new(0);
    let f = |_: &[f64], _: &mut [f64; 0]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok::<(), SciOxideError>(())
    };
    let sample: [f64; 0] = [];
    let mut output: [f64; 0] = [];
    let bounds = [Interval::new(0.0_f64, 1.0).unwrap()];
    let rule = NewtonCotes::<f64>::new(3, NewtonCotesEndpoints::Closed, Budget::ONE).unwrap();
    let product = ProductRule::new(&rule, scioxide_integrate::ProductOptions::default());
    let mut pw = product.workspace((1, 0)).unwrap();
    let result = run(|context| {
        product.integrate_into(
            |point| {
                let mut lane = sample;
                scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                    *v = <_ as scioxide_core::RealScalar>::NAN
                });
                {
                    let result: Result<(), SciOxideError> = f(point, &mut lane);
                    result?;
                }
                Ok(lane)
            },
            &bounds,
            &mut output,
            &mut pw,
            context,
        )
    })
    .unwrap();
    assert_eq!(
        result.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let mc_f = |_: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok([0.0_f64; 0])
    };
    let options = MonteCarloOptions::default().with_samples(Budget::try_from(8).unwrap());
    let root = scioxide_random::Philox4x64::from_seed(1);
    let streams = RandomStreams::new(&root, 0);
    let mc = MonteCarloIntegrator::new(&bounds, options).unwrap();
    let report = run(|context| {
        mc.integrate_into(
            &mc_f,
            streams,
            &mut output,
            &mut mc
                .workspace(0, ExecutionContext::sequential_native())
                .unwrap(),
            context,
        )
    })
    .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let tolerance = ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity).unwrap();
    let mut cw = CubaturePlan::new(&[bounds[0]; 2], SubdivisionOptions::default())
        .unwrap()
        .workspace(0)
        .unwrap();
    let result = run(|context| {
        scioxide_integrate::CubaturePlan::new(
            &[bounds[0]; 2],
            scioxide_integrate::SubdivisionOptions::default(),
        )
        .and_then(|plan| {
            plan.integrate_into(
                |point| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let result: Result<(), SciOxideError> = f(point, &mut lane);
                        result?;
                    }
                    Ok(lane)
                },
                &mut output,
                &mut cw,
                context,
            )
        })
    })
    .unwrap();
    assert_eq!(
        result.termination(),
        IntegrationTermination::FixedRuleCompleted
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    for interval in [bounds[0], Interval::new(0.0, f64::from_bits(1)).unwrap()] {
        let de = DoubleExponentialIntegrator::new(
            DoubleExponentialDomain::Finite {
                interval,
                singular_points: &[],
            },
            LevelOptions::default().with_tolerance(tolerance.clone()),
        )
        .unwrap();
        let mut dw = de.workspace(0).unwrap();
        let report = de
            .integrate_into(
                |_: scioxide_integrate::DoubleExponentialPoint<f64>| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok([0.0_f64; 0])
                },
                &mut output,
                &mut dw,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
    let mut pw = product.workspace((0, 0)).unwrap();
    let result = run(|context| {
        product.integrate_into(
            |point| {
                let mut lane = sample;
                scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                    *v = <_ as scioxide_core::RealScalar>::NAN
                });
                {
                    let result: Result<(), SciOxideError> = f(point, &mut lane);
                    result?;
                }
                Ok(lane)
            },
            &[],
            &mut output,
            &mut pw,
            context,
        )
    })
    .unwrap();
    assert_eq!(result.evaluations(), 1);
    let mc = MonteCarloIntegrator::new(&[], options).unwrap();
    let report = run(|context| {
        mc.integrate_into(
            &mc_f,
            streams,
            &mut output,
            &mut mc
                .workspace(0, ExecutionContext::sequential_native())
                .unwrap(),
            context,
        )
    })
    .unwrap();
    assert_eq!(report.evaluations(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    let mut cw = CubaturePlan::<f64>::new(&[], SubdivisionOptions::default())
        .unwrap()
        .workspace(0)
        .unwrap();
    let result = run(|context| {
        scioxide_integrate::CubaturePlan::new(
            &[],
            scioxide_integrate::SubdivisionOptions::default(),
        )
        .and_then(|plan| {
            plan.integrate_into(
                |point| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let result: Result<(), SciOxideError> = f(point, &mut lane);
                        result?;
                    }
                    Ok(lane)
                },
                &mut output,
                &mut cw,
                context,
            )
        })
    })
    .unwrap();
    assert_eq!(result.evaluations(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}
