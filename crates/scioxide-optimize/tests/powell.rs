#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{
    AttemptOutcome, Bounds, Budget, ExecutionContext, ProblemToken, RealScalar, SciOxideError,
    SolveIdentity,
};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_optimize::{
    BoundedObjective, ObjectiveFn, OptimizeResult, OptimizeTermination, Powell, PowellOptions,
};

fn directions<S: scioxide_core::NumericScalar>(rows: usize, values: &[S]) -> Matrix<S> {
    MatrixView::from_slice(rows, rows, values)
        .unwrap()
        .try_to_matrix()
        .unwrap()
}

fn parent(token: u128, dimension: usize) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        dimension as u128,
        0,
    )
}

fn quadratic<S: RealScalar>() {
    let objective = ObjectiveFn::new(2, |x: &[S]| {
        let a = x[0] - S::ONE;
        let b = x[1] + S::ONE;
        Ok((a + b) * (a + b) + S::TEN * (a - b) * (a - b))
    });
    let solver = Powell::<S>::default();
    let result = solver
        .minimize(
            &objective,
            &[
                S::try_from_f64_lossy(4.0).unwrap(),
                S::try_from_f64_lossy(3.0).unwrap(),
            ],
        )
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    let tolerance = S::SQRT_EPSILON * S::try_from_f64_lossy(30.0).unwrap();
    assert!((result.x()[0] - S::ONE).abs() < tolerance, "{result:?}");
    assert!((result.x()[1] + S::ONE).abs() < tolerance, "{result:?}");
    assert!(result.fun() < tolerance * tolerance, "{result:?}");
    assert_eq!(result.grad_evals(), 0);
    assert!(powell_of(&result).replacements() > 0);
}
#[test]
fn rotated_quadratic_both_widths() {
    quadratic::<f64>();
    quadratic::<f32>();
}

#[test]
fn curved_valley() {
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
    });
    let result = Powell::new(
        PowellOptions::default().with_max_evaluations(Budget::try_from(20_000).unwrap()),
    )
    .unwrap()
    .minimize(&objective, &[-1.2, 1.0])
    .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(result.fun() < 1e-12, "{result:?}");
}

#[test]
fn bounds_fixed_and_nonsmooth() {
    let objective = ObjectiveFn::new(3, |x: &[f64]| {
        assert!((-2.0..=2.0).contains(&x[0]));
        assert_eq!(x[1], 4.0);
        assert!((-1.0..=1.0).contains(&x[2]));
        Ok((x[0] - 0.25).abs() + (x[2] - 3.0).powi(2))
    });
    let bounds = Bounds::new(vec![
        (Some(-2.0), Some(2.0)),
        (Some(4.0), Some(4.0)),
        (Some(-1.0), Some(1.0)),
    ])
    .unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let result = Powell::default()
        .minimize(&problem, &[0.0, 4.0, 0.0])
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 0.25).abs() < 1e-7, "{result:?}");
    assert_eq!(result.x()[2], 1.0);
    assert_eq!(powell_of(&result).free_dimension(), 2);
}

#[test]
fn dependent_directions_rejected_before_callbacks() {
    let objective = ObjectiveFn::new(2, |_: &[f64]| -> Result<f64, SciOxideError> {
        panic!("must validate rank first")
    });
    let solver =
        Powell::new(PowellOptions::default().with_directions(directions(2, &[1.0, 2.0, 1.0, 2.0])))
            .unwrap();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    assert!(matches!(
        (solver.minimize_into(
            &objective,
            &[1.0, 1.0],
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
    let attempt = work.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(work.last_accepted().is_none());
    assert_eq!(output.termination(), OptimizeTermination::NotStarted);
}

#[test]
fn exact_local_budget_and_workspace_reuse() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(x.iter().map(|x| x * x).sum())
    });
    for limit in 1..25 {
        calls.store(0, Ordering::Relaxed);
        let solver = Powell::new(
            PowellOptions::default().with_max_evaluations(Budget::try_from(limit).unwrap()),
        )
        .unwrap();
        let mut work = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        solver
            .minimize_into(
                &objective,
                &[3.0, 2.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), limit, "{output:?}");
        assert_eq!(output.fun_evals(), limit);
        assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(output.fun(), output.x().iter().map(|x| x * x).sum::<f64>());
        let attempt = work.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
        assert_eq!(work.last_accepted().unwrap().point(), output.x());
        assert_eq!(
            work.last_accepted().unwrap().progress().value(),
            output.fun()
        );
        solver
            .minimize_into(
                &objective,
                &[3.0, 2.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2 * limit);
    }
}

#[test]
fn fixed_and_empty_evaluate_once() {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().sum()));
    let bounds = Bounds::new(vec![(Some(2.0), Some(2.0)), (Some(3.0), Some(3.0))]).unwrap();
    let result = Powell::default()
        .minimize(
            &BoundedObjective::new(&objective, &bounds).unwrap(),
            &[2.0, 3.0],
        )
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.fun_evals(), 1);
    struct Empty;
    impl scioxide_optimize::ObjectiveFunction<f64> for Empty {
        fn dim(&self) -> usize {
            0
        }
        fn eval(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(7.0)
        }
    }
    let result = Powell::default().minimize(&Empty, &[]).unwrap();
    assert!(result.is_success());
    assert_eq!(result.fun_evals(), 1);
}

#[test]
fn prepared_solves_execute_including_replacements() {
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] + x[1]).powi(2) + 10.0 * (x[0] - x[1]).powi(2))
    });
    let solver = Powell::default();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    {
        solver
            .minimize_into(
                &objective,
                &[4.0, 3.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success());
        assert!(powell_of(&output).replacements() > 0);
    };
}

#[test]
fn parent_admission_and_child_counts_are_conserved() {
    use scioxide_core::{BudgetKind, EvaluationRole, SolveBudgets, SolveState};
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
    let solver = Powell::default();
    let mut work = solver.workspace(2).unwrap();
    for limit in 1..25 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), parent(7, 2), &mut budgets)
            .unwrap();
        let mut result = OptimizeResult::empty();
        solver
            .minimize_into(&objective, &[3.0, 2.0], &mut result, &mut work, &mut ctx)
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(result.fun_evals(), limit);
        assert_eq!(result.counts(), work.last_attempt().unwrap().counts());
        assert_eq!(
            ctx.counts().nested().provider(EvaluationRole::Objective),
            limit
        );
        // The exhausted shared allowance denies the next child's first observation.
        assert!(
            solver
                .minimize_into(&objective, &[3.0, 2.0], &mut result, &mut work, &mut ctx)
                .is_err()
        );
    }
    let solver = Powell::new(PowellOptions::default().with_max_iterations(Budget::ONE)).unwrap();
    let result = solver.minimize(&objective, &[3.0, 2.0]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(result.iterations(), 1);
}

#[test]
fn callback_errors_are_counted_and_output_is_transactional() {
    use scioxide_core::{EvaluationRole, InvalidInputReason, SolveBudgets, SolveState};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let solver = Powell::default();
    let mut work = solver.workspace(2).unwrap();
    let good = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
    let mut output = solver.minimize(&good, &[1.0, 2.0]).unwrap();
    let saved = output.clone();
    // A provider error fails the solve wherever it occurs; a non-finite value
    // fails it only at the start (later ones are rejected trials, tested below).
    for (nonfinite, injections) in [(false, 1..12), (true, 1..2)] {
        for fail_at in injections {
            calls.store(0, Ordering::Relaxed);
            let bad = ObjectiveFn::new(2, |x: &[f64]| {
                if calls.fetch_add(1, Ordering::Relaxed) + 1 == fail_at {
                    if nonfinite {
                        return Ok(f64::NAN);
                    }
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("x"),
                        InvalidInputReason::Malformed,
                        "injected provider failure",
                    ));
                }
                Ok(x.iter().map(|x| x * x).sum())
            });
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut ctx = state
                .begin_solve(ExecutionContext::sequential(), parent(9, 2), &mut budgets)
                .unwrap();
            let error = solver
                .minimize_into(&bad, &[3.0, 2.0], &mut output, &mut work, &mut ctx)
                .unwrap_err();
            if nonfinite {
                assert!(matches!(
                    (error).view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
            } else {
                assert!(matches!(
                    (error).view(),
                    scioxide_core::SciOxideErrorView::InvalidInput { .. }
                ));
            }
            assert_eq!(
                ctx.counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Objective),
                fail_at
            );
            assert_eq!(output.x(), saved.x());
            assert_eq!(output.fun(), saved.fun());
            assert_eq!(output.counts(), saved.counts());
        }
    }
    solver
        .minimize_into(
            &good,
            &[3.0, 2.0],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.is_success());
}

#[test]
fn cancellation_before_and_during_nested_work() {
    use scioxide_core::CancellationToken;
    use std::sync::atomic::{AtomicUsize, Ordering};
    for cancel_at in [0, 1, 2, 5, 25] {
        let cancel = CancellationToken::new();
        let calls = AtomicUsize::new(0);
        if cancel_at == 0 {
            cancel.cancel();
        }
        let objective = ObjectiveFn::new(2, |x: &[f64]| {
            if calls.fetch_add(1, Ordering::Relaxed) + 1 == cancel_at {
                cancel.cancel();
            }
            Ok(x.iter().map(|x| x * x).sum())
        });
        let solver = Powell::default();
        let mut work = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        let result = solver
            .minimize_into(
                &objective,
                &[3.0, 2.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential().with_cancellation(&cancel),
            )
            .map(|()| output);
        if cancel_at == 0 {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::ExecutionStopped { .. })
            ));
        } else {
            let result = result.unwrap();
            assert_eq!(result.termination(), OptimizeTermination::Cancelled);
            assert_eq!(result.fun_evals(), cancel_at);
        }
        assert_eq!(calls.load(Ordering::Relaxed), cancel_at);
    }
}

#[test]
fn executor_context_is_sequential_and_matches_convenience() {
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};
    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            panic!("Powell must execute sequentially")
        }
    }
    let solver = Powell::default();
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
    let mut work = solver.workspace(2).unwrap();
    let mut sequential = OptimizeResult::empty();
    solver
        .minimize_into(
            &objective,
            &[3.0, 2.0],
            &mut sequential,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let mut injected = OptimizeResult::empty();
    solver
        .minimize_into(
            &objective,
            &[3.0, 2.0],
            &mut injected,
            &mut work,
            ExecutionContext::new(&CallerExecutor),
        )
        .unwrap();
    assert_eq!(injected.x(), sequential.x());
    assert_eq!(injected.counts(), sequential.counts());
}

#[test]
fn injected_deadline_preserves_finite_point() {
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};
    use std::sync::atomic::{AtomicUsize, Ordering};
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
    let solver =
        Powell::new(PowellOptions::default().with_time_limit(Duration::from_millis(1))).unwrap();
    let objective = ObjectiveFn::new(1, |x: &[f64]| {
        clock.0.store(1, Ordering::Relaxed);
        Ok(x[0] * x[0])
    });
    assert_eq!(
        solver.minimize(&objective, &[3.0]).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::CapabilityUnavailable
    );
    let mut work = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::empty();
    solver
        .minimize_into(
            &objective,
            &[3.0],
            &mut result,
            &mut work,
            ExecutionContext::sequential()
                .with_clock(&clock_binding)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::TimeLimit);
    assert_eq!(result.fun_evals(), 1);
    assert_eq!(result.x(), &[3.0]);
}

#[test]
fn strided_scaled_directions_and_shape_validation() {
    // Column-major view and extreme column magnitudes represent the coordinate basis.
    let directions = MatrixView::from_slice_with_strides(2, 2, &[1e-300, 0.0, 0.0, 1e300], 1, 2)
        .unwrap()
        .try_to_matrix()
        .unwrap();
    let solver = Powell::new(PowellOptions::default().with_directions(directions)).unwrap();
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2))
    });
    let result = solver.minimize(&objective, &[5.0, 4.0]).unwrap();
    assert!(result.is_success());
    assert!(result.fun() < 1e-15);
    assert!(solver.minimize(&objective, &[1.0]).is_err());
    let half_open = Bounds::new(vec![(Some(0.0), None); 2]).unwrap();
    let error = Powell::default()
        .minimize(
            &BoundedObjective::new(&objective, &half_open).unwrap(),
            &[1.0, 1.0],
        )
        .unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    let unit = Bounds::new(vec![(Some(0.0), Some(1.0)); 2]).unwrap();
    let bounded = BoundedObjective::new(&objective, &unit).unwrap();
    assert!(Powell::default().minimize(&bounded, &[2.0, 0.0]).is_err());
}

#[test]
fn non_finite_trials_are_rejected_after_the_start() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for fail_at in 2..12 {
        let calls = AtomicUsize::new(0);
        let objective = ObjectiveFn::new(2, |x: &[f64]| {
            Ok(if calls.fetch_add(1, Ordering::Relaxed) + 1 == fail_at {
                f64::NAN
            } else {
                x.iter().map(|x| x * x).sum()
            })
        });
        let result = Powell::default().minimize(&objective, &[3.0, 2.0]).unwrap();
        assert!(result.is_success(), "NaN at call {fail_at}: {result:?}");
        assert!(result.fun() < 1e-12 && result.fun().is_finite());
    }
}

#[test]
fn a_constant_objective_stops_bracketing_at_its_first_flat_triple() {
    // The old bracket expansion doubled its step until the evaluation budget ran out.
    let objective = ObjectiveFn::new(1, |_: &[f64]| Ok(1.0));
    let result =
        Powell::new(PowellOptions::default().with_max_evaluations(Budget::try_from(30).unwrap()))
            .unwrap()
            .minimize(&objective, &[0.0])
            .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Converged);
    assert_eq!(result.x(), &[0.0]);
    assert!(result.fun_evals() < 30, "{result:?}");
}

#[test]
fn rejected_replacements_preserve_a_spanning_basis() {
    use scioxide_core::{RankThreshold, Tolerance};
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] + x[1]).powi(2) + 10.0 * (x[0] - x[1]).powi(2))
    });
    let solver = Powell::new(
        PowellOptions::default()
            .with_rank_threshold(RankThreshold::from_tolerance(
                Tolerance::new(0.75, 0.0).unwrap(),
            ))
            .with_max_evaluations(Budget::try_from(20_000).unwrap()),
    )
    .unwrap();
    let result = solver.minimize(&objective, &[4.0, 3.0]).unwrap();
    assert!(powell_of(&result).rejected_replacements() > 0, "{result:?}");
    assert_eq!(powell_of(&result).rank().unwrap().rank(), 2);
    assert!(result.is_success(), "{result:?}");
    assert!(result.fun() < 1e-12);
}

#[test]
fn f32_finite_fixed_nonsmooth_and_near_dependent() {
    let objective = ObjectiveFn::new(2, |x: &[f32]| {
        assert_eq!(x[1], 1.0);
        Ok((x[0] - 0.25).abs())
    });
    let bounds = Bounds::new(vec![(Some(-1.0), Some(1.0)), (Some(1.0), Some(1.0))]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let result = Powell::default().minimize(&problem, &[0.0, 1.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 0.25).abs() < 0.002);
    let solver = Powell::new(
        PowellOptions::default().with_directions(directions(2, &[1.0f32, 1.0, 0.0, f32::EPSILON])),
    )
    .unwrap();
    // Without the box both coordinates are free, so the basis is rank deficient.
    assert!(solver.minimize(&objective, &[0.0, 1.0]).is_err());
}

#[test]
fn scale_changes_preserve_analytic_minimum_and_report_stopping() {
    use scioxide_core::{ResidualTolerance, StepTolerance, Tolerance};
    for scale in [1e-12, 1.0, 1e12] {
        let objective = ObjectiveFn::new(2, |x: &[f64]| {
            let a = x[0] / scale - 1.0;
            let b = x[1] / scale + 1.0;
            Ok(a * a + 2.0 * b * b)
        });
        let bounds = Bounds::new(vec![(Some(-4.0 * scale), Some(4.0 * scale)); 2]).unwrap();
        let problem = BoundedObjective::new(&objective, &bounds).unwrap();
        let solver = Powell::new(
            PowellOptions::default()
                .with_step_tolerance(StepTolerance::from_tolerance(
                    Tolerance::new(1e-8 * scale, 1e-8).unwrap(),
                ))
                .with_scalar_options(
                    scioxide_optimize::ScalarMinimumOptions::default()
                        .with_step_tolerance(StepTolerance::from_tolerance(
                            Tolerance::new(1e-10 * scale, 1e-10).unwrap(),
                        ))
                        .with_function_tolerance(ResidualTolerance::default()),
                ),
        )
        .unwrap();
        let result = solver
            .minimize(&problem, &[3.0 * scale, 2.0 * scale])
            .unwrap();
        assert!(result.is_success(), "scale={scale}, {result:?}");
        assert!((result.x()[0] / scale - 1.0).abs() < 1e-6, "{result:?}");
        assert!((result.x()[1] / scale + 1.0).abs() < 1e-6, "{result:?}");
        let d = powell_of(&result);
        assert_eq!(
            d.step_report().unwrap().unwrap().raw_norm(),
            d.step().unwrap()
        );
        assert!(
            d.function_report()
                .unwrap()
                .unwrap()
                .scaled_norm()
                .is_finite()
        );
    }
}

#[test]
fn input_errors_preserve_specific_shape_and_origin() {
    use scioxide_core::NonFiniteOrigin;
    assert!(matches!(
        (Powell::new(PowellOptions::<f64>::default().with_directions(directions(1, &[f64::NAN]))))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        })
    ));
    let nonsquare = Matrix::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
    assert_eq!(
        Powell::new(PowellOptions::default().with_directions(nonsquare))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    let solver =
        Powell::new(PowellOptions::default().with_directions(directions(1, &[1.0]))).unwrap();
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
    assert!(matches!(
        (solver.minimize(&objective, &[1.0, 2.0]))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
    ));
}

fn powell_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::PowellDiagnostics<S> {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::Powell(diagnostics) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}

#[test]
fn a_flat_direction_ends_its_line_not_the_solve() {
    // f is constant along x1. Bracketing along that line stops at the first
    // flat triple, and the line's stagnation never ends the solve.
    let objective = scioxide_optimize::ObjectiveFn::new(2, |x: &[f64]| Ok((x[0] - 1.0).powi(2)));
    let result = Powell::new(PowellOptions::default())
        .unwrap()
        .minimize(&objective, &[0.0, 0.0])
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 1.0).abs() < 1e-6, "{result:?}");
    assert!(result.fun_evals() < 200, "{result:?}");
}
