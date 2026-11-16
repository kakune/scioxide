//! Resource, executor, allocation and publication boundaries for weighted rules.
#![allow(clippy::unwrap_used)]

use scioxide_core::*;
use scioxide_integrate::*;
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

#[test]
fn an_underflowed_weight_is_not_published_as_an_exact_zero() {
    let plan = WeightedQuadraturePlan::endpoint(
        Interval::new(0.0, 1e-200).unwrap(),
        &[],
        EndpointWeight::new(1.0, 0.0, false, false).unwrap(),
        scioxide_integrate::WeightedQuadratureOptions::new(SubdivisionOptions::default()),
    )
    .unwrap();
    // The singular region's scale (5e-201)^2 underflows on its own; applied in
    // extended range after the amplitude, ∫ 1e300 x dx = 5e-101 stays representable.
    let (value, report) = plan.integrate(|_: f64| Ok(1e300)).unwrap();
    assert!(report.is_success(), "{value} {report:?}");
    assert!((value - 5e-101).abs() <= 1e-12 * 5e-101, "{value}");
    // A unit amplitude contributes 5e-401: it rounds to zero, but its error
    // evidence keeps a positive roundoff floor instead of claiming an exact zero.
    let (value, report) = plan.integrate(|_: f64| Ok(1.0)).unwrap();
    assert_eq!(value, 0.0);
    let error = report.error_components().unwrap();
    assert!(
        error.roundoff().is_some_and(|floor| floor > 0.0),
        "{report:?}"
    );
    assert!(report.error_estimate().unwrap() > 0.0, "{report:?}");
}

#[test]
fn contours_support_single_precision() {
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(2e-4_f32, 2e-4, NormSelection::Infinity).unwrap());
    let plan = ContourPlan::new(
        Interval::new(0.0, core::f32::consts::PI).unwrap(),
        &[],
        options,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut out = Complex::new(0.0, 0.0);
    let report = plan
        .integrate_into(
            |z: Complex<f32>| Ok(z.conj()),
            |t: f32| {
                Ok((
                    Complex::new(t.cos(), t.sin()),
                    Complex::new(-t.sin(), t.cos()),
                ))
            },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!(out.re.abs() < 2e-4);
    assert!((out.im - core::f32::consts::PI).abs() < 2e-4);
}

#[test]
fn cancellation_and_moment_depth_keep_a_measured_parent() {
    let token = CancellationToken::new();
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-14, 1e-14, NormSelection::Infinity).unwrap());
    let prepare = |levels| {
        WeightedQuadraturePlan::fourier(
            Interval::new(0.0, 1.0).unwrap(),
            &[],
            20.0,
            FourierWeight::Cosine,
            scioxide_integrate::WeightedQuadratureOptions::new(options.clone())
                .with_moment_levels(levels),
        )
        .unwrap()
    };
    let plan = prepare(Budget::try_from(16).unwrap());
    let mut ws = plan.workspace(1).unwrap();
    let calls = AtomicUsize::new(0);
    let mut out = 99.0;
    let report = plan
        .integrate_into(
            |x: f64| {
                calls.fetch_add(1, Relaxed);
                token.cancel();
                Ok(1.0 / (1.0 + 1000.0 * (x - 0.4).powi(2)))
            },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native().with_cancellation(&token),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
    );
    assert_eq!(calls.load(Relaxed), 25);
    assert!(ws.component_errors().is_some());
    let saved = out;
    assert!(
        plan.integrate_into(
            |_: f64| Ok(0.0),
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(out, saved);
    assert!(ws.last_attempt().unwrap().progress().is_none());
    let (value, report) = prepare(Budget::ONE)
        .integrate(|x: f64| Ok(1.0 / (1.0 + 1000.0 * (x - 0.4).powi(2))))
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::RefinementLimit
    );
    assert_eq!(report.evaluations(), 25);
    assert!(value.is_finite());
}
#[test]
fn executor_order_workers_and_malformed_dispatch() {
    let prepare = |policy| {
        WeightedQuadraturePlan::endpoint(
            Interval::new(0.0, 1.0).unwrap(),
            &[],
            EndpointWeight::new(-0.5, 0.25, true, true).unwrap(),
            scioxide_integrate::WeightedQuadratureOptions::new(
                SubdivisionOptions::default().with_execution_policy(policy),
            ),
        )
        .unwrap()
    };
    let f = |x: f64| Ok([x.exp(), -2.0 * x.exp()]);
    let baseline = prepare(ExecutionPolicy::Sequential).integrate(&f).unwrap();
    for workers in [1, 2, 4] {
        for threaded in [false, true] {
            let executor = Reordered {
                workers,
                threaded,
                omit: false,
                duplicate: false,
            };
            let plan = prepare(ExecutionPolicy::Parallel(
                NonZeroUsize::new(workers).unwrap(),
            ));
            let mut ws = plan.workspace(2).unwrap();
            let mut out = [0.0; 2];
            let report = plan
                .integrate_into(&f, &mut out, &mut ws, ExecutionContext::new(&executor))
                .unwrap();
            assert_eq!((out, report), baseline);
        }
    }
    let plan = prepare(ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()));
    let mut ws = plan.workspace(2).unwrap();
    let mut out = baseline.0;
    for duplicate in [false, true] {
        let executor = Reordered {
            workers: 2,
            threaded: false,
            omit: !duplicate,
            duplicate,
        };
        let saved = out;
        assert!(
            plan.integrate_into(&f, &mut out, &mut ws, ExecutionContext::new(&executor))
                .is_err()
        );
        assert_eq!(out, saved);
        assert!(ws.component_errors().is_none());
        assert!(ws.last_attempt().unwrap().progress().is_none());
    }
}

#[test]
fn whole_child_rules_preserve_parent_and_reconcile_ancestors() {
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-12, 1e-12, NormSelection::Infinity).unwrap());
    let plan = WeightedQuadraturePlan::fourier(
        Interval::new(0.0, 1.0).unwrap(),
        &[],
        20.0,
        FourierWeight::Cosine,
        scioxide_integrate::WeightedQuadratureOptions::new(options)
            .with_moment_levels(Budget::try_from(16).unwrap()),
    )
    .unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let mut out = 99.0;
    let calls = AtomicUsize::new(0);
    let report = run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(74).unwrap()),
        |parent| {
            let report = plan
                .integrate_into(
                    |x: f64| {
                        calls.fetch_add(1, Relaxed);
                        Ok(1.0 / (1.0 + 1000.0 * (x - 0.4).powi(2)))
                    },
                    &mut out,
                    &mut ws,
                    &mut *parent,
                )
                .unwrap();
            assert!(
                parent
                    .can_consume(BudgetKind::Evaluation, 49)
                    .unwrap()
                    .is_granted()
            );
            assert!(
                !parent
                    .can_consume(BudgetKind::Evaluation, 50)
                    .unwrap()
                    .is_granted()
            );
            assert_eq!(parent.counts().own().provider(EvaluationRole::Function), 0);
            assert_eq!(
                parent.counts().nested().provider(EvaluationRole::Function),
                25
            );
            report
        },
    );
    assert_eq!(calls.load(Relaxed), 25);
    assert_eq!(report.evaluations(), 25);
    assert_eq!(report.subdivisions(), 0);
    assert_eq!(
        report.termination(),
        IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
    );
    assert_ne!(out, 99.0);
}
#[test]
fn contour_provider_failures_and_executor_match() {
    let plan = ContourPlan::new(
        Interval::new(0.0, 1.0).unwrap(),
        &[],
        SubdivisionOptions::default()
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap())),
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut out = Complex::new(11.0, 12.0);
    let calls = AtomicUsize::new(0);
    let map = |x: f64| {
        calls.fetch_add(1, Relaxed);
        Ok((Complex::new(x, x), Complex::new(1.0, 1.0)))
    };
    run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(29).unwrap()),
        |parent| {
            assert!(
                plan.integrate_into(|z: Complex<f64>| Ok(z), map, &mut out, &mut ws, parent)
                    .is_err()
            )
        },
    );
    assert_eq!(calls.load(Relaxed), 0);
    assert_eq!(out, Complex::new(11.0, 12.0));
    let executor = Reordered {
        workers: 3,
        threaded: true,
        omit: false,
        duplicate: false,
    };
    run(
        ExecutionContext::new(&executor),
        SolveBudgets::unbounded(),
        |parent| {
            let report = plan
                .integrate_into(
                    |z: Complex<f64>| Ok(z),
                    map,
                    &mut out,
                    &mut ws,
                    &mut *parent,
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(
                parent
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Function),
                2 * report.evaluations()
            );
        },
    );
    assert!(out.re.abs() < 1e-14);
    assert!((out.im - 1.0).abs() < 1e-14);
    let saved = out;
    run(
        ExecutionContext::sequential(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(30).unwrap()),
        |parent| {
            let error = plan
                .integrate_into(
                    |z: Complex<f64>| Ok(z),
                    |_: f64| -> Result<(Complex<f64>, Complex<f64>), SciOxideError> {
                        Err(SciOxideError::invalid_input(
                            scioxide_core::argument!("map"),
                            InvalidInputReason::OutOfRange,
                            "failure",
                        ))
                    },
                    &mut out,
                    &mut ws,
                    &mut *parent,
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput { .. }
            ));
            assert_eq!(out, saved);
            assert!(
                parent
                    .can_consume(BudgetKind::Evaluation, 29)
                    .unwrap()
                    .is_granted()
            );
            assert!(
                !parent
                    .can_consume(BudgetKind::Evaluation, 30)
                    .unwrap()
                    .is_granted()
            );
        },
    );
}
#[test]
fn improper_directions_zero_shapes_and_open_tails() {
    let options =
        SubdivisionOptions::default().with_max_subdivisions(Budget::try_from(128).unwrap());
    for (domain, weight, expected) in [
        (
            FourierDomain::Lower { upper: 0.0 },
            FourierWeight::Sine,
            -0.5,
        ),
        (FourierDomain::RealLine, FourierWeight::Cosine, 1.0),
    ] {
        let plan = FourierTailPlan::new(
            domain,
            1.0,
            weight,
            scioxide_integrate::FourierTailOptions::new(options.clone())
                .with_max_cycles(Budget::try_from(64).unwrap())
                .with_moment_levels(Budget::try_from(16).unwrap()),
        )
        .unwrap();
        let (value, report) = plan.integrate(|x: f64| Ok((-x.abs()).exp())).unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::HeuristicToleranceReached
        );
        assert!((value - expected).abs() < 1e-8);
    }
    let plan = FourierTailPlan::new(
        FourierDomain::Upper { lower: 0.0 },
        1.0,
        FourierWeight::Sine,
        scioxide_integrate::FourierTailOptions::new(options.clone())
            .with_max_cycles(Budget::try_from(12).unwrap())
            .with_moment_levels(Budget::try_from(16).unwrap()),
    )
    .unwrap();
    let (_, report) = plan.integrate(|_: f64| Ok(1.0)).unwrap();
    assert!(!report.is_success());
    assert_ne!(
        report.termination(),
        IntegrationTermination::HeuristicToleranceReached
    );
    assert!(report.error_estimate().is_none());
    let plan = WeightedQuadraturePlan::breakpoints(
        Interval::new(0.0, 1.0).unwrap(),
        &[],
        scioxide_integrate::WeightedQuadratureOptions::new(options),
    )
    .unwrap();
    let mut ws = plan.workspace(0).unwrap();
    let mut out: [f64; 0] = [];
    let report = plan
        .integrate_into(
            |_: f64| -> Result<[f64; 0], SciOxideError> {
                Err(SciOxideError::external("empty values must not sample"))
            },
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 0);
}
fn run<T>(
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
    op: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
) -> T {
    let mut state = SolveState::new();
    let id = SolveIdentity::new(
        ProblemToken::new(591),
        scioxide_core::algorithm_id!("integrate", "weighted-contracts", 1),
        0,
        0,
    );
    op(&mut state.begin_solve(execution, id, &mut budgets).unwrap())
}
struct Reordered {
    workers: usize,
    omit: bool,
    duplicate: bool,
    threaded: bool,
}
impl Executor for Reordered {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.workers).unwrap()
    }
    fn execute(
        &self,
        ctx: ExecutionContext<'_>,
        plan: ExecutionPlan,
        op: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let mut partitions: Vec<_> = plan.partitions().collect();
        partitions.reverse();
        if self.omit {
            partitions.pop();
        }
        let dispatch = |target| {
            SequentialExecutor.execute(ctx, plan, &|p, (), scope| {
                if &p == target {
                    op(p.clone(), (), scope)?;
                    if self.duplicate {
                        op(p, (), scope)?;
                    }
                    Ok::<(), SciOxideError>(())
                } else {
                    Ok::<(), SciOxideError>(())
                }
            })
        };
        if self.threaded {
            std::thread::scope(|scope| {
                let handles: Vec<_> = partitions
                    .iter()
                    .map(|p| scope.spawn(move || dispatch(p)))
                    .collect();
                for h in handles {
                    h.join().unwrap()?;
                }
                Ok::<(), SciOxideError>(())
            })?;
        } else {
            for p in &partitions {
                dispatch(p)?;
            }
        }
        SequentialExecutor.execute(ctx, plan, &|_, (), _| Ok::<(), SciOxideError>(()))
    }
}
