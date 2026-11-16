//! Independent manufactured integrals and nested resource contracts.
#![allow(clippy::unwrap_used)]

#[cfg(feature = "std")]
use scioxide_core::ExecutionPolicy;
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, Interval, NormSelection, ProblemToken, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_integrate::{
    IdentityMap, IntegrationTermination, IteratedDomainPlan, IteratedOptions, SubdivisionOptions,
};
use std::sync::atomic::AtomicUsize as Cell;
fn context<T>(budgets: &mut SolveBudgets, f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("integrate", "iterated-test", 1),
        0,
        0,
    );
    f(&mut state
        .begin_solve(ExecutionContext::sequential(), identity, budgets)
        .unwrap())
}
fn plan(d: usize) -> IteratedDomainPlan<f64> {
    IteratedDomainPlan::new(
        d,
        IteratedOptions::new(
            SubdivisionOptions::default()
                .with_tolerance(
                    ErrorTolerance::scalar(1e-9, 1e-9, NormSelection::Infinity).unwrap(),
                )
                .with_max_subdivisions(Budget::try_from(500).unwrap()),
        ),
    )
    .unwrap()
}
fn unit(_: &[f64]) -> Result<Interval<f64>, SciOxideError> {
    Interval::new(0., 1.)
}

#[test]
fn separable_and_variable_polynomials() {
    for d in 1..=4 {
        let r = plan(d)
            .integrate(|p| Ok(p.iter().map(|x| x * x).product::<f64>()), &unit)
            .unwrap();
        assert!(r.1.is_success(), "{r:?}");
        assert!((r.0 - 3_f64.powi(-(i32::try_from(d).unwrap()))).abs() < 1e-12);
        assert_eq!(r.1.evaluations(), 15_usize.pow(u32::try_from(d).unwrap()));
    }
    let r = plan(3)
        .integrate(
            |_| Ok(1.),
            |p: &[f64]| Interval::new(0., if p.is_empty() { 1. } else { p[p.len() - 1] }),
        )
        .unwrap();
    assert!(r.1.is_success(), "{r:?}");
    assert!((r.0 - 1. / 6.).abs() < 1e-12);
    // Exact inner polynomials leave only arithmetic floors, which join roundoff.
    let e = r.1.error_components().unwrap();
    assert!(e.roundoff().unwrap() > 0.);
    assert!(e.inner().unwrap().magnitude() <= e.roundoff().unwrap());
}
#[test]
fn f32_variable_triangle() {
    let p =
        IteratedDomainPlan::new(
            2,
            IteratedOptions::new(SubdivisionOptions::<f32>::default().with_tolerance(
                ErrorTolerance::scalar(2e-5, 2e-5, NormSelection::Infinity).unwrap(),
            )),
        )
        .unwrap();
    let r = p
        .integrate(
            |point: &[f32]| Ok(point[0] + point[1]),
            |point: &[f32]| Interval::new(0., if point.is_empty() { 1. } else { point[0] }),
        )
        .unwrap();
    assert!(r.1.is_success(), "{r:?}");
    assert!((r.0 - 0.5).abs() < 2e-6);
}
#[test]
fn hints_resolve_narrow_feature() {
    // Compact polynomial bump has exact integral 16*w/15; no comparator oracle.
    let width = 0.0001;
    let center = 0.12345;
    let p = plan(2)
        .with_breakpoints(1, &[center - width, center, center + width])
        .unwrap();
    let r = p
        .integrate(
            |x| {
                let t = (x[1] - center) / width;
                Ok(if t.abs() < 1. {
                    (1. - t * t).powi(2)
                } else {
                    0.
                })
            },
            &unit,
        )
        .unwrap();
    assert!(r.1.is_success(), "{r:?}");
    assert!((r.0 - 16. * width / 15.).abs() < 1e-12);
}
#[test]
fn inner_difficulty_is_included_in_global_error() {
    let p = plan(2);
    let calls = Cell::new(0);
    let r = p
        .integrate(
            |x| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(1. / (1. + 10000. * (x[1] - 0.3).powi(2)))
            },
            &unit,
        )
        .unwrap();
    let exact = (70_f64.atan() + 30_f64.atan()) / 100.;
    assert!(r.1.is_success(), "{r:?}");
    assert!((r.0 - exact).abs() < 2e-9);
    assert_eq!(
        r.1.evaluations(),
        calls.load(core::sync::atomic::Ordering::Relaxed)
    );
    let e = r.1.error_components().unwrap();
    assert!(e.inner().unwrap().magnitude() > 0.);
    assert!(e.total().unwrap() >= e.inner().unwrap().magnitude());
}
#[test]
fn budget_shared_across_all_inner_calls() {
    for budget in [1, 14, 15, 224, 225, 240] {
        let p = IteratedDomainPlan::new(
            2,
            IteratedOptions::new(
                SubdivisionOptions::default()
                    .with_max_evaluations(Some(Budget::try_from(budget).unwrap())),
            ),
        )
        .unwrap();
        let mut ws = p.workspace(1).unwrap();
        let calls = Cell::new(0);
        let mut out = 99.0;
        let result = p.integrate_into(
            |x: &[f64]| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(x[1].exp())
            },
            &unit,
            &IdentityMap,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        );
        let count = calls.load(core::sync::atomic::Ordering::Relaxed);
        assert!(count <= budget);
        assert_eq!(result.is_ok(), budget >= 225);
        if let Ok(report) = result {
            assert!(report.is_success());
            assert_eq!(report.evaluations(), count);
        } else {
            assert_eq!(out, 99.0);
            assert!(ws.last_attempt().unwrap().progress().is_none());
        }
    }
}
#[test]
fn inner_budget_stop_cannot_be_convergence() {
    let p = IteratedDomainPlan::new(
        2,
        IteratedOptions::new(
            SubdivisionOptions::default()
                .with_max_subdivisions(Budget::ONE)
                .with_tolerance(
                    ErrorTolerance::scalar(1e-14, 0.0, NormSelection::Infinity).unwrap(),
                ),
        ),
    )
    .unwrap();
    let mut ws = p.workspace(1).unwrap();
    let mut out = 99.0;
    let error = p
        .integrate_into(
            |x: &[f64]| Ok(1.0 / (1.0 + 10000.0 * (x[1] - 0.3).powi(2))),
            &unit,
            &IdentityMap,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ResourceLimit(
                scioxide_core::BudgetKind::Subdivision
            ),
            ..
        }
    ));
    assert_eq!(out, 99.0);
}
#[test]
fn zero_dimension_and_degenerate_domain() {
    let calls = Cell::new(0);
    let r = plan(0)
        .integrate(
            |p| {
                assert!(p.is_empty());
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(3.)
            },
            |_| unreachable!(),
        )
        .unwrap();
    assert_eq!(r.0, 3.0);
    assert_eq!(r.1.evaluations(), 1);
    let r = plan(2)
        .integrate(
            |_| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(1.)
            },
            |_| Interval::new(0., 0.),
        )
        .unwrap();
    assert_eq!(r.0, 0.0);
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(r.1.termination(), IntegrationTermination::DegenerateDomain);
}

#[test]
fn executor_sequential_slice_values() {
    let p = plan(2);
    let mut ws = p.workspace(2).unwrap();
    let mut out = [0.0; 2];
    let report = p
        .integrate_into(
            |x: &[f64]| Ok([x[0] * x[1], x[0] + x[1]]),
            &unit,
            &IdentityMap,
            &mut out[..],
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((out[0] - 0.25).abs() < 1e-12);
    assert!((out[1] - 1.0).abs() < 1e-12);
}
#[test]
fn ordinary_failure_preserves_output() {
    let p = plan(2);
    let mut ws = p.workspace(1).unwrap();
    let mut output = 42.;
    let sample = 0.;
    context(&mut SolveBudgets::unbounded(), |ctx| {
        assert!(
            p.integrate_into(
                |_point: &[f64]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let v = &mut lane;

                        *v = f64::NAN;
                        Ok::<(), SciOxideError>(())
                    }?;
                    Ok(lane)
                },
                &unit,
                &IdentityMap,
                &mut output,
                &mut ws,
                ctx
            )
            .is_err()
        );
    });
    assert_eq!(output, 42.);
    assert!(ws.component_errors().is_none());
}
#[test]
fn invalid_partitions_shapes_and_infinite_limits() {
    for cuts in [&[0.][..], &[1.], &[0.3, 0.2], &[0.2, 0.2], &[f64::NAN]] {
        assert!(plan(2).with_breakpoints(0, cuts).is_err());
    }
    assert!(plan(2).with_breakpoints(2, &[]).is_err());
    assert!(
        plan(1)
            .integrate(|_| Ok(1.), |_| Interval::new(0., f64::INFINITY))
            .is_err()
    );
}

#[test]
fn complete_root_estimate_survives_budget_exhaustion() {
    let p = IteratedDomainPlan::new(
        2,
        IteratedOptions::new(
            SubdivisionOptions::default()
                .with_max_evaluations(Some(Budget::try_from(225).unwrap()))
                .with_tolerance(
                    ErrorTolerance::scalar(1e-12, 0., NormSelection::Infinity).unwrap(),
                ),
        ),
    )
    .unwrap();
    let r = p
        .integrate(|x| Ok(1. / (1. + 10000. * (x[0] - 0.3).powi(2))), &unit)
        .unwrap();
    assert_eq!(
        r.1.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
    );
    assert_eq!(r.1.evaluations(), 225);
    assert!(r.0.is_finite());
    assert!(r.1.error_components().unwrap().inner().is_some());
}
#[test]
fn context_evaluation_budget_is_shared_and_preserves_output() {
    let p = plan(2);
    let mut ws = p.workspace(1).unwrap();
    let mut out = 42.0;
    let mut budgets = SolveBudgets::unbounded().with_limit(
        scioxide_core::BudgetKind::Evaluation,
        Budget::try_from(224).unwrap(),
    );
    context(&mut budgets, |parent| {
        let error = p
            .integrate_into(
                |_: &[f64]| Ok(1.0),
                &unit,
                &IdentityMap,
                &mut out,
                &mut ws,
                &mut *parent,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
        assert_eq!(
            parent
                .counts()
                .total()
                .unwrap()
                .provider(scioxide_core::EvaluationRole::Function),
            210
        );
        assert!(ws.last_attempt().unwrap().progress().is_none());
    });
    assert_eq!(out, 42.0);
    assert!(ws.component_errors().is_none());
}

#[test]
fn cubature_roundoff_floor_prevents_false_success_under_cancellation() {
    let f = |x: &[f64]| Ok(1e10 * (x[0] - 0.5) + 1.);
    let r = scioxide_integrate::CubaturePlan::new(
        &[Interval::new(0., 1.).unwrap(); 2],
        SubdivisionOptions::default()
            .with_max_subdivisions(Budget::try_from(2).unwrap())
            .with_tolerance(ErrorTolerance::scalar(1e-9, 0., NormSelection::Infinity).unwrap()),
    )
    .and_then(|plan| plan.integrate(&f))
    .unwrap();
    assert!(!r.1.is_success());
    assert!(r.1.error_components().unwrap().roundoff().unwrap() > 1e-9);
    assert!((r.0 - 1.).abs() < 1e-5);
}
#[test]
fn f32_complex_dense_values_and_zero_component_shape_validation() {
    let p =
        IteratedDomainPlan::new(
            2,
            IteratedOptions::new(SubdivisionOptions::<f32>::default().with_tolerance(
                ErrorTolerance::scalar(1e-4, 1e-4, NormSelection::Infinity).unwrap(),
            )),
        )
        .unwrap();
    let mut ws = p.workspace(2).unwrap();
    let mut output = scioxide_core::Complex::new(0_f32, 0.);
    let sample = output;
    context(&mut SolveBudgets::unbounded(), |ctx| {
        let r = p
            .integrate_into(
                |point: &[f32]| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let x = point;
                        let v = &mut lane;

                        *v = scioxide_core::Complex::new(x[0], x[1]);
                        Ok::<(), SciOxideError>(())
                    }?;
                    Ok(lane)
                },
                |_| Interval::new(0., 1.),
                &IdentityMap,
                &mut output,
                &mut ws,
                ctx,
            )
            .unwrap();
        assert!(r.termination().is_success());
    });
    assert!((output.re - 0.5).abs() < 1e-6);
    assert!((output.im - 0.5).abs() < 1e-6);
}
#[cfg(feature = "std")]
#[test]
fn threaded_executor_matches_sequential_for_cubature_and_nested_rules() {
    let workers = std::num::NonZeroUsize::new(3).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    for dimension in [1, 2] {
        let bounds = vec![Interval::new(0.0, 1.0).unwrap(); dimension];
        let callback = |x: &[f64]| {
            Ok(if dimension == 1 {
                [x[0] * x[0], x[0]]
            } else {
                [x[0] * x[1], x[0] + x[1]]
            })
        };
        for iterated in [false, true] {
            let mut baseline = None;
            for policy in [
                ExecutionPolicy::Sequential,
                ExecutionPolicy::Parallel(workers),
            ] {
                let options = SubdivisionOptions::default()
                    .with_tolerance(
                        ErrorTolerance::scalar(1e-9, 1e-9, NormSelection::Infinity).unwrap(),
                    )
                    .with_execution_policy(policy);
                let mut out = [0.0; 2];
                let report = if iterated {
                    let p =
                        IteratedDomainPlan::new(dimension, IteratedOptions::new(options)).unwrap();
                    let mut ws = p.workspace(2).unwrap();
                    p.integrate_into(
                        &callback,
                        &unit,
                        &IdentityMap,
                        &mut out,
                        &mut ws,
                        ExecutionContext::new(&pool),
                    )
                    .unwrap()
                } else {
                    let p = scioxide_integrate::CubaturePlan::new(&bounds, options).unwrap();
                    let mut ws = p.workspace(2).unwrap();
                    p.integrate_into(&callback, &mut out, &mut ws, ExecutionContext::new(&pool))
                        .unwrap()
                };
                if let Some(expected) = baseline {
                    assert_eq!((out, report), expected);
                } else {
                    baseline = Some((out, report));
                }
            }
        }
    }
}

#[test]
fn extreme_dimensions_fail_safely() {
    assert!(
        IteratedDomainPlan::<f64>::new(33, IteratedOptions::new(SubdivisionOptions::default()))
            .is_err()
    );
}

#[test]
fn coincident_rule_nodes_are_rejected_before_callbacks() {
    let calls = Cell::new(0);
    let error = plan(1)
        .integrate(
            |_: &[f64]| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok::<f64, SciOxideError>(1.0)
            },
            |_| Interval::new(1.0, 1.0 + 16.0 * f64::EPSILON),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn cubature_collapsed_corner_and_edge_generators_are_rejected() {
    let calls = Cell::new(0);
    let p = scioxide_integrate::CubaturePlan::new(
        &[Interval::new(1.0, 1.0 + 5.0 * f64::EPSILON).unwrap(); 2],
        SubdivisionOptions::default(),
    )
    .unwrap();
    let error = p
        .integrate(|_: &[f64]| {
            calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok::<f64, SciOxideError>(1.0)
        })
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn virtual_rules_obey_work_budgets_even_when_inner_slices_have_zero_extent() {
    let p = IteratedDomainPlan::new(
        3,
        IteratedOptions::default().with_max_iterations(Some(Budget::ONE)),
    )
    .unwrap();
    let mut ws = p.workspace(1).unwrap();
    let mut out = 99.0;
    let mut budgets = SolveBudgets::unbounded();
    context(&mut budgets, |parent| {
        let error = p
            .integrate_into(
                |_: &[f64]| Ok(1.0),
                |outer: &[f64]| Interval::new(0.0, if outer.len() == 2 { 0.0 } else { 1.0 }),
                &IdentityMap,
                &mut out,
                &mut ws,
                parent,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert!(ws.last_attempt().unwrap().progress().is_none());
        assert_eq!(
            ws.last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .provider(scioxide_core::EvaluationRole::Function),
            0
        );
    });
    assert_eq!(out, 99.0);
}

#[test]
fn one_dimensional_cubature_honors_declared_workspace_capacity() {
    let bounds = [Interval::new(0.0, 1.0).unwrap()];
    let coarse = scioxide_integrate::CubaturePlan::new(
        &bounds,
        SubdivisionOptions::default().with_max_subdivisions(Budget::ONE),
    )
    .unwrap();
    let fine = scioxide_integrate::CubaturePlan::new(
        &bounds,
        SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(2).unwrap()),
    )
    .unwrap();
    let mut ws = coarse.workspace(1).unwrap();
    let calls = Cell::new(0);
    let mut out = 42.0;
    assert!(
        fine.integrate_into(
            |_: &[f64]| {
                calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                Ok(1.0)
            },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(out, 42.0);
    assert_eq!(
        ws.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
}

fn tolerance_plan<S: scioxide_core::RealScalar>(
    d: usize,
    absolute: S,
    relative: S,
) -> IteratedDomainPlan<S> {
    IteratedDomainPlan::new(
        d,
        IteratedOptions::new(SubdivisionOptions::default().with_tolerance(
            ErrorTolerance::scalar(absolute, relative, NormSelection::Infinity).unwrap(),
        )),
    )
    .unwrap()
}

#[test]
fn inner_floor_stagnation_publishes_its_uncertainty() {
    use core::f64::consts::PI;
    // The inner integral cancels near x = 0.3, where a relative target follows
    // the inner value below its roundoff floor. Exact integral 1/5.
    let (value, report) = tolerance_plan(2, 1e-15, 1e-13)
        .integrate(|x: &[f64]| Ok((2. * PI * x[1]).cos() + (x[0] - 0.3)), &unit)
        .unwrap();
    let (swapped, reference) = tolerance_plan(2, 1e-15, 1e-13)
        .integrate(|x: &[f64]| Ok((2. * PI * x[0]).cos() + (x[1] - 0.3)), &unit)
        .unwrap();
    for (value, report) in [(value, &report), (swapped, &reference)] {
        assert_eq!(
            report.termination(),
            IntegrationTermination::ToleranceReached
        );
        assert!((value - 0.2).abs() <= report.error_estimate().unwrap());
        assert!(report.error_estimate().unwrap() <= 1e-15 + 1e-13 * 0.2);
    }
    assert_eq!(report.evaluations(), reference.evaluations());
    // Below the propagated floor the outer solve stagnates honestly instead of
    // spending work or failing before its first estimate.
    let (value, report) = tolerance_plan(2, 0., 3e-14)
        .integrate(|x: &[f64]| Ok((2. * PI * x[1]).cos() + (x[0] - 0.3)), &unit)
        .unwrap();
    assert_eq!(report.termination(), IntegrationTermination::Stagnation);
    let estimate = report.error_estimate().unwrap();
    assert!(estimate > 3e-14 * 0.2 && (value - 0.2).abs() <= estimate);
    let evidence = report.error_components().unwrap();
    assert!(evidence.roundoff().unwrap() >= evidence.discretization().unwrap().magnitude());
    // Smooth integrands stop on the initial rules: 15^d callbacks.
    for (d, exact) in [(2, 1f64.exp() - 1.), (3, 1f64.exp() - 1.)] {
        let exact = exact.powi(i32::try_from(d).unwrap());
        let (value, report) = tolerance_plan(d, 0., 1e-14)
            .integrate(|x: &[f64]| Ok(x.iter().sum::<f64>().exp()), &unit)
            .unwrap();
        assert_eq!(report.termination(), IntegrationTermination::Stagnation);
        assert_eq!(
            report.evaluations(),
            15_usize.pow(u32::try_from(d).unwrap())
        );
        assert!((value - exact).abs() <= report.error_estimate().unwrap());
    }
    let (value, report) = tolerance_plan(2, 0f32, 1e-6)
        .integrate(
            |x: &[f32]| Ok((x[0] + x[1]).exp()),
            |_: &[f32]| Interval::new(0., 1.),
        )
        .unwrap();
    assert_eq!(report.termination(), IntegrationTermination::Stagnation);
    assert_eq!(report.evaluations(), 225);
    let exact = (1f64.exp() - 1.).powi(2);
    assert!((f64::from(value) - exact).abs() <= f64::from(report.error_estimate().unwrap()));
}

#[test]
fn vector_inner_stagnation_keeps_the_outer_solve() {
    // A converged and a kinked component share the inner solves; inner
    // stagnation of the pair publishes its uncertainty, so the outer solve runs
    // to its work limit with a complete estimate instead of failing.
    let p = IteratedDomainPlan::new(
        3,
        IteratedOptions::new(
            SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(200).unwrap()),
        ),
    )
    .unwrap();
    let mut ws = p.workspace(2).unwrap();
    let mut out = [f64::NAN; 2];
    let report = p
        .integrate_into(
            |x: &[f64]| Ok([(x[0] + x[1] + x[2]).exp(), (x[0] - x[1]).abs()]),
            &unit,
            &IdentityMap,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
    );
    let evidence = ws.component_errors().unwrap();
    for ((value, exact), evidence) in out
        .iter()
        .zip([(1f64.exp() - 1.).powi(3), 1. / 3.])
        .zip(evidence)
    {
        assert!(
            (value - exact).abs() <= evidence.total().unwrap(),
            "{out:?}"
        );
    }
}

#[test]
fn bounds_receive_the_outer_coordinates_only() {
    let lengths = std::cell::RefCell::new(std::collections::BTreeSet::new());
    let integral = scioxide_integrate::integrate_iterated(
        |_: &[f64]| Ok(1.0),
        |outer: &[f64]| {
            lengths.borrow_mut().insert(outer.len());
            Interval::new(0.0, 1.0)
        },
        3,
    )
    .unwrap();
    assert!((integral.value() - 1.0).abs() < 1e-12);
    assert_eq!(
        lengths.into_inner(),
        std::collections::BTreeSet::from([0, 1, 2])
    );
}
