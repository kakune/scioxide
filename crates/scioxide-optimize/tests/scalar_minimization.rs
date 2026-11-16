//! Independent analytic and lifecycle tests for scalar minimum searches.
#![allow(clippy::unwrap_used)]
use core::cell::Cell;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, EvaluationRole, ExecutionContext, Interval,
    ProblemToken, ResidualTolerance, SolveBudgets, SolveState, StepTolerance, Tolerance,
};
use scioxide_optimize::{
    MinimumBracket, OptimizeTermination, ScalarMinimizer, ScalarMinimumMethod,
    ScalarMinimumOptions, ScalarMinimumStart,
};

fn parent(token: u128) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        1,
        0,
    )
}

macro_rules! numerical_cases {
    ($module:ident, $s:ty, $accuracy:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn analytic_and_boundary_minima() {
                for method in [
                    ScalarMinimumMethod::GoldenSection,
                    ScalarMinimumMethod::Brent,
                ] {
                    let solver =
                        ScalarMinimizer::<$s>::new(method, ScalarMinimumOptions::default());
                    for minimum in [-1.75, 0.0, 0.125, 2.25] {
                        let objective = |x: $s| {
                            assert!((-3.0..=4.0).contains(&x));
                            Ok((x - minimum) * (x - minimum))
                        };
                        let result = solver
                            .minimize(
                                &objective,
                                ScalarMinimumStart::Bounded(Interval::new(-3.0, 4.0).unwrap()),
                            )
                            .unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert!((result.x() - minimum).abs() <= $accuracy, "{result:?}");
                        assert_eq!(result.fun(), objective(result.x()).unwrap());
                        let interval = result.interval().unwrap().unwrap();
                        assert!(interval.contains(result.x()));
                        assert!(interval.contains(minimum), "{result:?}");
                        let samples = result.samples().unwrap();
                        for sample in samples {
                            assert_eq!(sample.value(), objective(sample.x()).unwrap());
                        }
                    }
                    for sign in [-1.0, 1.0] {
                        let objective = |x: $s| Ok(sign * x);
                        let result = solver
                            .minimize(
                                &objective,
                                ScalarMinimumStart::Bounded(Interval::new(-2.0, 3.0).unwrap()),
                            )
                            .unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert_eq!(result.x(), if sign > 0.0 { -2.0 } else { 3.0 });
                    }
                }
            }
            #[test]
            fn bracket_and_flat_nonsmooth_objectives() {
                for method in [
                    ScalarMinimumMethod::GoldenSection,
                    ScalarMinimumMethod::Brent,
                ] {
                    let solver =
                        ScalarMinimizer::<$s>::new(method, ScalarMinimumOptions::default());
                    for seeds in [(0.0, 1.0), (1.0, 0.0), (4.0, 5.0)] {
                        let quadratic = |x: $s| Ok((x - 2.5) * (x - 2.5));
                        let bracket = solver
                            .bracket_minimum(&quadratic, seeds.0, seeds.1)
                            .unwrap();
                        assert!(bracket.is_success(), "{bracket:?}");
                        let bracket = bracket.bracket().unwrap();
                        let result = solver
                            .minimize(&quadratic, ScalarMinimumStart::Bracket(bracket))
                            .unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert!((result.x() - 2.5).abs() <= $accuracy);
                        let auto = solver
                            .minimize(&quadratic, ScalarMinimumStart::Seeds(seeds.0, seeds.1))
                            .unwrap();
                        assert!(auto.is_success());
                    }
                    for power in [0, 1, 4] {
                        let objective = |x: $s| Ok((x - 0.3).abs().powi(power));
                        let result = solver
                            .minimize(
                                &objective,
                                ScalarMinimumStart::Bounded(Interval::new(-1.0, 2.0).unwrap()),
                            )
                            .unwrap();
                        assert!(result.is_success(), "{result:?}");
                        assert!(result.interval().unwrap().unwrap().contains(result.x()));
                        if power != 0 {
                            assert!((result.x() - 0.3).abs() <= $accuracy, "{result:?}");
                        }
                    }
                }
            }
            #[test]
            fn adjacent_singleton_and_extreme_coordinates() {
                let solver = ScalarMinimizer::<$s>::default();
                let next = <$s>::from_bits((1.0 as $s).to_bits() + 1);
                let options = ScalarMinimumOptions::default().with_step_tolerance(
                    StepTolerance::from_tolerance(Tolerance::new(0.0, 0.0).unwrap()),
                );
                let strict = ScalarMinimizer::<$s>::new(ScalarMinimumMethod::Brent, options);
                let result = strict
                    .minimize(
                        &|x| Ok(x),
                        ScalarMinimumStart::Bounded(Interval::new(1.0, next).unwrap()),
                    )
                    .unwrap();
                assert_eq!(result.termination(), OptimizeTermination::Stagnation);
                assert_eq!(result.fun_evals(), 2);
                let count = Cell::new(0);
                let result = solver
                    .minimize(
                        &|x: $s| {
                            count.set(count.get() + 1);
                            Ok(x * x)
                        },
                        ScalarMinimumStart::Bounded(Interval::new(-0.0, 0.0).unwrap()),
                    )
                    .unwrap();
                assert!(result.is_success());
                assert_eq!(count.get(), 1);
                assert!(result.x().is_sign_negative());
                let tiny = <$s>::from_bits(1);
                let result = strict
                    .minimize(
                        &|x: $s| Ok(x),
                        ScalarMinimumStart::Bounded(Interval::new(0.0, tiny * 4.0).unwrap()),
                    )
                    .unwrap();
                assert_eq!(result.termination(), OptimizeTermination::Stagnation);
                assert_eq!(result.x(), 0.0);
                let result = solver
                    .minimize(
                        &|x: $s| Ok((x / <$s>::MAX).powi(2)),
                        ScalarMinimumStart::Bounded(Interval::new(-<$s>::MAX, <$s>::MAX).unwrap()),
                    )
                    .unwrap();
                assert!(result.is_success(), "{result:?}");
                assert_eq!(result.x(), 0.0);
            }
        }
    };
}
numerical_cases!(single, f32, 0.004);
numerical_cases!(double, f64, 5e-8);

#[test]
fn invalid_initialization_and_callbacks() {
    let solver = ScalarMinimizer::<f64>::default();
    assert!(MinimumBracket::new([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]).is_err());
    assert!(MinimumBracket::new([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]).is_err());
    assert!(MinimumBracket::new([0.0, 0.0, 2.0], [1.0, 0.0, 2.0]).is_err());
    assert!(MinimumBracket::new([0.0, 1.0, f64::INFINITY], [1.0, 0.0, 2.0]).is_err());
    let callbacks = Cell::new(0);
    let objective = |x: f64| {
        callbacks.set(callbacks.get() + 1);
        Ok(x)
    };
    assert!(
        solver
            .minimize(&objective, ScalarMinimumStart::Seeds(1.0, 1.0))
            .is_err()
    );
    assert_eq!(callbacks.get(), 0);
    let bracket = MinimumBracket::new([0.0, 1.0, 2.0], [1.0, 0.0, 2.0]).unwrap();
    assert!(matches!(
        (solver.minimize(&objective, ScalarMinimumStart::Bracket(bracket)))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
    assert_eq!(callbacks.get(), 3);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            (solver.minimize(&|_: f64| Ok(value), ScalarMinimumStart::Seeds(0.0, 1.0)))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }
}

#[test]
fn exact_local_budgets_and_incomplete_initialization() {
    for budget in 1..12 {
        let solver = ScalarMinimizer::<f64>::new(
            ScalarMinimumMethod::Brent,
            ScalarMinimumOptions::default().with_max_evaluations(Budget::try_from(budget).unwrap()),
        );
        let calls = Cell::new(0);
        let result = solver
            .minimize(
                &|x: f64| {
                    calls.set(calls.get() + 1);
                    Ok((x - 0.2).powi(2))
                },
                ScalarMinimumStart::Bounded(Interval::new(-1.0, 1.0).unwrap()),
            )
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(result.fun_evals(), budget);
        assert_eq!(calls.get(), budget);
        assert_eq!(result.interval().unwrap().is_some(), budget >= 3);
    }
    let solver = ScalarMinimizer::new(
        ScalarMinimumMethod::Brent,
        ScalarMinimumOptions::default().with_max_iterations(Budget::ONE),
    );
    let result = solver.bracket_minimum(&|x: f64| Ok(-x), 0.0, 1.0).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
    assert!(result.bracket().is_none());
    assert_eq!(result.iterations(), 1);
}

#[test]
fn scoped_parent_budget_cancellation_and_reuse() {
    let solver = ScalarMinimizer::<f64>::default();
    let start = ScalarMinimumStart::Bounded(Interval::new(-1.0, 1.0).unwrap());
    let mut state = SolveState::new();
    let objective = |x: f64| Ok(x * x);
    for budget in [1, 2, 5, 1000] {
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(budget).unwrap());
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(12), &mut budgets)
            .unwrap();
        let result = solver
            .minimize_scoped(&objective, start, &mut context)
            .unwrap();
        assert!(result.fun_evals() <= budget);
        assert_eq!(result.counts().nested(), Default::default());
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::Objective),
            result.fun_evals()
        );
        assert_eq!(context.counts().own().total_providers().unwrap(), 0);
        // A later child shares the parent's remaining evaluation allowance.
        let second = solver.minimize_scoped(&objective, start, &mut context);
        assert_eq!(second.is_err(), result.fun_evals() == budget);
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let calls = Cell::new(0);
    let counted = |x: f64| {
        calls.set(calls.get() + 1);
        Ok(x * x)
    };
    assert!(matches!(
        (solver.minimize_scoped(
            &counted,
            start,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        ))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::ExecutionStopped { .. })
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn noise_does_not_converge_on_objective_stall_alone() {
    let options = ScalarMinimumOptions::default()
        .with_step_tolerance(StepTolerance::from_tolerance(
            Tolerance::new(1e-12, 0.0).unwrap(),
        ))
        .with_function_tolerance(ResidualTolerance::from_tolerance(
            Tolerance::new(1e-15, 0.0).unwrap(),
        ));
    let solver = ScalarMinimizer::new(ScalarMinimumMethod::Brent, options);
    let result = solver
        .minimize(
            &|x: f64| {
                Ok((x - 0.37).powi(2)
                    + if x.to_bits().is_multiple_of(2) {
                        1e-6
                    } else {
                        -1e-6
                    })
            },
            ScalarMinimumStart::Bounded(Interval::new(0.0, 1.0).unwrap()),
        )
        .unwrap();
    if result.is_success() {
        let samples = result.samples().unwrap();
        assert!(samples[2].x() - samples[0].x() <= 1e-12);
        assert!((samples[0].value() - samples[1].value()).abs() <= 1e-15);
        assert!((samples[2].value() - samples[1].value()).abs() <= 1e-15);
    } else {
        assert!(matches!(
            result.termination(),
            OptimizeTermination::Stagnation | OptimizeTermination::IterationLimit
        ));
    }
}

#[test]
fn successful_solves_execute() {
    let objective = |x: f64| Ok((x - 0.2).powi(2));
    for method in [
        ScalarMinimumMethod::GoldenSection,
        ScalarMinimumMethod::Brent,
    ] {
        let solver = ScalarMinimizer::new(method, ScalarMinimumOptions::default());
        {
            assert!(
                solver
                    .minimize(&objective, ScalarMinimumStart::Seeds(-1.0, 1.0))
                    .unwrap()
                    .is_success()
            );
            assert!(
                solver
                    .bracket_minimum(&objective, -1.0, 1.0)
                    .unwrap()
                    .is_success()
            );
        };
    }
}

#[test]
fn independently_observed_prefixes_contract_and_improve() {
    for method in [
        ScalarMinimumMethod::GoldenSection,
        ScalarMinimumMethod::Brent,
    ] {
        let mut previous = (-2.0, 3.0, f64::INFINITY);
        for budget in 3..120 {
            let solver = ScalarMinimizer::new(
                method,
                ScalarMinimumOptions::default()
                    .with_max_evaluations(Budget::try_from(budget).unwrap()),
            );
            let objective = |x: f64| Ok((x - 0.7).powi(2) * (1.0 + 0.1 * (x - 0.7).powi(2)));
            let result = solver
                .minimize(
                    &objective,
                    ScalarMinimumStart::Bounded(Interval::new(-2.0, 3.0).unwrap()),
                )
                .unwrap();
            let samples = result.samples().unwrap();
            let lower = samples[0].x();
            let upper = samples[2].x();
            assert!(lower >= previous.0 && upper <= previous.1);
            assert!(lower <= 0.7 && upper >= 0.7);
            assert!(result.fun() <= previous.2);
            assert_eq!(result.fun(), objective(result.x()).unwrap());
            assert!(samples.into_iter().all(|s| s.value() >= result.fun()));
            previous = (lower, upper, result.fun());
            if result.is_success() {
                break;
            }
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert!(budget < 119, "failed to converge");
        }
    }
}

#[test]
fn interruption_and_provider_failure_preserve_only_finite_observations() {
    let solver = ScalarMinimizer::<f64>::default();
    for singleton in [false, true] {
        let start = ScalarMinimumStart::Bounded(
            Interval::new(0.0, if singleton { 0.0 } else { 1.0 }).unwrap(),
        );
        let cancel = CancellationToken::new();
        let result = solver
            .minimize_scoped(
                &|x| {
                    cancel.cancel();
                    Ok(x)
                },
                start,
                ExecutionContext::sequential().with_cancellation(&cancel),
            )
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::Cancelled);
        assert_eq!(result.fun_evals(), 1);
        assert!(!result.is_success());
    }
    // Non-finite trials are rejected: the finite endpoint stays the best
    // observation and no non-finite sample is published.
    let start = ScalarMinimumStart::Bounded(Interval::new(0.0, 1.0).unwrap());
    let calls = Cell::new(0);
    let result = solver
        .minimize(
            &|x| {
                calls.set(calls.get() + 1);
                Ok(if x == 0.0 { 2.0 } else { f64::NAN })
            },
            start,
        )
        .unwrap();
    assert!(!result.is_success(), "{result:?}");
    assert_eq!((result.x(), result.fun()), (0.0, 2.0));
    assert!(result.samples().is_none() && result.bracket().is_none());
    assert_eq!(result.fun_evals(), calls.get());
    // Without any finite observation the start itself is invalid.
    let result = solver.minimize(&|_| Ok(f64::NAN), start);
    assert!(matches!(
        (result)
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
    ));
}

#[test]
fn bracketing_expands_geometrically_and_safeguards_range() {
    let solver = ScalarMinimizer::default();
    let result = solver
        .bracket_minimum(&|x: f64| Ok((x - 1e10).powi(2)), 0.0, 1.0)
        .unwrap();
    assert!(result.is_success());
    assert!(result.fun_evals() < 40, "geometric expansion: {result:?}");
    let result = solver
        .bracket_minimum(
            &|x: f64| Ok((x / f64::MAX + 0.1).powi(2)),
            f64::MAX,
            0.4 * f64::MAX,
        )
        .unwrap();
    assert!(result.is_success(), "range-safe expansion: {result:?}");
    let result = solver
        .bracket_minimum(&|x: f64| Ok(-x), f64::MAX / 2.0, f64::MAX)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert_eq!(result.fun_evals(), 2);
}

#[test]
fn local_and_parent_evaluation_allowances_are_both_enforced() {
    for local_limit in [true, false] {
        let options = if local_limit {
            ScalarMinimumOptions::default().with_max_evaluations(Budget::ONE)
        } else {
            ScalarMinimumOptions::default()
        };
        let solver = ScalarMinimizer::new(ScalarMinimumMethod::Brent, options);
        let start = ScalarMinimumStart::Bounded(Interval::new(-1.0, 1.0).unwrap());
        let mut state = SolveState::new();
        let mut budgets = if local_limit {
            SolveBudgets::unbounded()
        } else {
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(4).unwrap())
        };
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), parent(88), &mut budgets)
            .unwrap();
        let result = solver
            .minimize_scoped(&|x: f64| Ok(x * x), start, &mut context)
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(result.fun_evals(), if local_limit { 1 } else { 4 });
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::Objective),
            result.fun_evals()
        );
    }
}

#[test]
fn injected_deadline_stops_after_the_last_finite_observation() {
    use core::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};
    struct ManualClock(AtomicUsize);
    impl Clock for ManualClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(self.0.load(Ordering::Relaxed) as u64),
            ))
        }
    }
    let clock = ManualClock(AtomicUsize::new(0));
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let solver = ScalarMinimizer::default();
    let start = ScalarMinimumStart::Bounded(Interval::new(0.0, 1.0).unwrap());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential()
                .with_clock(&clock_binding)
                .unwrap(),
            parent(9),
            &mut budgets,
        )
        .unwrap();
    context.limit_time(Duration::from_millis(1)).unwrap();
    let result = solver
        .minimize_scoped(
            &|x: f64| {
                clock.0.store(1, Ordering::Relaxed);
                Ok(x * x)
            },
            start,
            &mut context,
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::TimeLimit);
    assert_eq!(result.fun_evals(), 1);
    assert_eq!(result.x(), 0.0);
}

#[test]
fn a_flat_line_stops_bracketing_without_expanding() {
    let solver = ScalarMinimizer::<f64>::default();
    let result = solver.bracket_minimum(&|_: f64| Ok(2.0), 0.0, 1.0).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert_eq!(result.fun_evals(), 3);
    assert!(result.bracket().is_none());
    let result = solver
        .minimize(&|_: f64| Ok(2.0), ScalarMinimumStart::Seeds(0.0, 1.0))
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert_eq!(result.fun(), 2.0);
    assert_eq!(result.fun_evals(), 3);
}
