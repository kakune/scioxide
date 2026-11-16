//! Independent solutions, restoration transactions, and exact provider accounting.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Bounds, Budget, BudgetKind, EvaluationRole, ExecutionContext, ProblemToken, SciOxideError,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    AugmentedLagrangian, AugmentedLagrangianOptions, Bfgs, BfgsOptions, ConstrainedProblem,
    ConstraintFn, Constraints, InteriorPoint, ObjectiveFunction, OptimizeResult,
    OptimizeTermination, PenaltyMethod,
};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

macro_rules! numerical_contract {
    ($name:ident, $scalar:ty, $tol:expr) => {
        #[test]
        fn $name() {
            struct Sphere;
            impl ObjectiveFunction<$scalar> for Sphere {
                fn dim(&self) -> usize {
                    2
                }
                fn eval(&self, x: &[$scalar]) -> Result<$scalar, SciOxideError> {
                    Ok(x[0] * x[0] + x[1] * x[1])
                }
                fn gradient_into(
                    &self,
                    x: &[$scalar],
                    g: &mut [$scalar],
                ) -> Result<bool, SciOxideError> {
                    g[0] = 2.0 * x[0];
                    g[1] = 2.0 * x[1];
                    Ok(true)
                }
            }
            let equality = Constraints::new(2)
                .unwrap()
                .equality(
                    ConstraintFn::new(2, |x: &[$scalar]| Ok(x[0] + x[1] - 1.0)).with_jacobian(
                        |_: &[$scalar], output| {
                            output.copy_from_slice(&[1.0, 1.0]);
                            Ok(())
                        },
                    ),
                )
                .unwrap();
            let problem = ConstrainedProblem::new(&Sphere, &equality).unwrap();
            for start in [[-2.0, -1.0], [5.0, -4.0]] {
                for result in [
                    PenaltyMethod::default().minimize(&problem, &start).unwrap(),
                    AugmentedLagrangian::default()
                        .minimize(&problem, &start)
                        .unwrap(),
                ] {
                    assert!(result.is_success(), "{result:?}");
                    let x = result.x();
                    assert!(
                        (x[0] - 0.5).abs() < $tol && (x[1] - 0.5).abs() < $tol,
                        "{result:?}"
                    );
                    assert!((x[0] + x[1] - 1.0).abs() < $tol);
                    // Independent KKT multiplier for this problem is -1.
                    assert!((2.0 * x[0] - 1.0).abs() < 2.0 * $tol);
                    assert!(result.stationarity().unwrap() < $tol);
                }
            }
            let mixed = Constraints::new(2)
                .unwrap()
                .equality(
                    ConstraintFn::new(2, |x: &[$scalar]| Ok(x[0] + x[1] - 2.0)).with_jacobian(
                        |_: &[$scalar], output| {
                            output.copy_from_slice(&[1.0, 1.0]);
                            Ok(())
                        },
                    ),
                )
                .unwrap()
                .inequality(
                    ConstraintFn::new(2, |x: &[$scalar]| Ok(1.5 - x[0])).with_jacobian(
                        |_: &[$scalar], output| {
                            output.copy_from_slice(&[-1.0, 0.0]);
                            Ok(())
                        },
                    ),
                )
                .unwrap()
                .bounds(Bounds::new(vec![(None, None), (Some(0.0), None)]).unwrap())
                .unwrap();
            let problem = ConstrainedProblem::new(&Sphere, &mixed).unwrap();
            for result in [
                PenaltyMethod::default()
                    .minimize(&problem, &[-1.0, -1.0])
                    .unwrap(),
                AugmentedLagrangian::default()
                    .minimize(&problem, &[-1.0, -1.0])
                    .unwrap(),
            ] {
                assert!(result.is_success(), "{result:?}");
                let x = result.x();
                assert!(
                    (x[0] - 1.5).abs() < $tol && (x[1] - 0.5).abs() < $tol,
                    "{result:?}"
                );
                // Equality multiplier -1, active inequality multiplier 2, inactive bound multiplier 0.
                assert!((2.0 * x[0] - 3.0).abs() < 2.0 * $tol);
                assert!((2.0 * x[1] - 1.0).abs() < 2.0 * $tol);
                assert!(result.complementarity().unwrap() < $tol);
            }
            let constraints = Constraints::new(2)
                .unwrap()
                .bounds(Bounds::new(vec![(Some(1.0), None); 2]).unwrap())
                .unwrap();
            let problem = ConstrainedProblem::new(&Sphere, &constraints).unwrap();
            let barrier = InteriorPoint::default();
            assert!(
                barrier
                    .minimize(&problem, &[1.0, 2.0])
                    .unwrap()
                    .is_success()
            );
            let result = barrier.minimize(&problem, &[2.0, 2.0]).unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(result.x().iter().all(|&x| x > 1.0 && x - 1.0 < $tol));
            let slack = Constraints::new(2)
                .unwrap()
                .inequality(
                    ConstraintFn::new(2, |x: &[$scalar]| Ok(x[0] + x[1] - 1.0)).with_jacobian(
                        |_: &[$scalar], output| {
                            output.copy_from_slice(&[1.0, 1.0]);
                            Ok(())
                        },
                    ),
                )
                .unwrap();
            let objective: &dyn ObjectiveFunction<$scalar> = &Sphere;
            let problem = ConstrainedProblem::new(objective, &slack).unwrap();
            let result = barrier.minimize(&problem, &[0.1, 0.1]).unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(result.x().iter().all(|x| x.abs() < $tol));
            assert!(result.stationarity().unwrap() < $tol);
        }
    };
}
numerical_contract!(independent_f64_solutions, f64, 1e-4);
numerical_contract!(independent_f32_solutions, f32, 5e-3);

#[derive(Default)]
struct Counter(AtomicUsize);
impl Counter {
    fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
    fn set(&self, value: usize) {
        self.0.store(value, Ordering::Relaxed);
    }
}
#[derive(Default)]
struct Counts {
    objective: Counter,
    gradient: Counter,
    constraint: Counter,
    jacobian: Counter,
}
struct Counted {
    counts: Arc<Counts>,
    analytic: bool,
}
impl ObjectiveFunction<f64> for Counted {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.counts.objective.set(self.counts.objective.get() + 1);
        Ok(x[0] * x[0])
    }
    fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
        self.counts.gradient.set(self.counts.gradient.get() + 1);
        g[0] = 2.0 * x[0];
        Ok(self.analytic)
    }
    fn automatic_gradient_into(&self, _: &[f64], _: &mut [f64]) -> Result<bool, SciOxideError> {
        self.counts.gradient.set(self.counts.gradient.get() + 1);
        Ok(false)
    }
}

struct CountedConstraint {
    counts: Arc<Counts>,
    analytic: bool,
}
impl scioxide_optimize::ConstraintFunction<f64> for CountedConstraint {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.counts.constraint.set(self.counts.constraint.get() + 1);
        Ok(1.0 - x[0])
    }
    fn jacobian_into(&self, _: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
        self.counts.jacobian.set(self.counts.jacobian.get() + 1);
        g[0] = -1.0;
        Ok(self.analytic)
    }
    fn automatic_jacobian_into(&self, _: &[f64], _: &mut [f64]) -> Result<bool, SciOxideError> {
        self.counts.jacobian.set(self.counts.jacobian.get() + 1);
        Ok(false)
    }
}
fn counted_constraint(counts: &Arc<Counts>, analytic: bool) -> Constraints<f64> {
    Constraints::new(1)
        .unwrap()
        .inequality(CountedConstraint {
            counts: Arc::clone(counts),
            analytic,
        })
        .unwrap()
}

/// Identity of a caller-owned parent epoch; the solver opens a child epoch.
fn parent_identity(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "ConstrainedParent", 1),
        1,
        0,
    )
}

/// Restoration is the only outer step that records an own rejection.
fn restorations(output: &OptimizeResult<f64>) -> usize {
    output.counts().own().rejected()
}

#[test]
fn every_nested_and_finite_difference_callback_respects_the_parent_limit() {
    for analytic in [false, true] {
        for limit in 1..90 {
            let counts = Arc::new(Counts::default());
            let objective = Counted {
                counts: Arc::clone(&counts),
                analytic,
            };
            let constraints = counted_constraint(&counts, analytic);
            let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
            let solver = AugmentedLagrangian::<f64>::default();
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut result = OptimizeResult::empty();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(1),
                    &mut budgets,
                )
                .unwrap();
            solver
                .minimize_into(&problem, &[-2.0], &mut result, &mut workspace, &mut context)
                .unwrap();
            assert_eq!(
                result.counts().total().unwrap(),
                context.counts().total().unwrap()
            );
            assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Objective),
                counts.objective.get()
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Gradient),
                counts.gradient.get()
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Constraint),
                counts.constraint.get()
            );
            {
                assert_eq!(
                    result
                        .counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::ConstraintJacobian),
                    counts.jacobian.get()
                );
            }
            assert!(context.budget_report(BudgetKind::Evaluation).consumed() <= limit);
            assert_eq!(result.fun(), result.x()[0] * result.x()[0]);
            if !result.is_success() {
                assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            }
        }
    }
}

#[test]
fn repeated_value_and_gradient_requests_reuse_same_point_constraints() {
    let counts = Arc::new(Counts::default());
    let objective = Counted {
        counts: Arc::clone(&counts),
        analytic: true,
    };
    let constraints = counted_constraint(&counts, true);
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    let solver = PenaltyMethod::default().with_inner_optimizer(Bfgs::new(
        BfgsOptions::default().with_max_iterations(Budget::ONE),
    ));
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[-2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.counts().own().cache_hits() > 0);
    let work = workspace.last_attempt().unwrap().counts().total().unwrap();
    assert_eq!(
        work.provider(EvaluationRole::Objective),
        counts.objective.get()
    );
    assert_eq!(
        work.provider(EvaluationRole::Constraint),
        counts.constraint.get()
    );
}

#[test]
fn impossible_constraints_restore_then_stop_without_claiming_infeasibility() {
    let constraints = Constraints::new(1)
        .unwrap()
        .equality(
            ConstraintFn::new(1, |_: &[f64]| Ok(1.0)).with_jacobian(|_: &[f64], output| {
                output.copy_from_slice(&[0.0]);
                Ok(())
            }),
        )
        .unwrap();
    let objective = Counted {
        counts: Arc::new(Counts::default()),
        analytic: true,
    };
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    let solver = AugmentedLagrangian::<f64>::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::Stagnation);
    assert_eq!(output.primal_feasibility(), Some(1.0));
    assert_eq!(restorations(&output), 1);
    assert_eq!(output.fun(), output.x()[0] * output.x()[0]);
}

#[test]
fn restoration_recovers_from_a_stalled_short_inner_solve() {
    let objective = Counted {
        counts: Arc::new(Counts::default()),
        analytic: true,
    };
    let constraints = Constraints::new(1)
        .unwrap()
        .equality(
            ConstraintFn::new(1, |x: &[f64]| Ok(x[0] * x[0] - 1.0)).with_jacobian(
                |x: &[f64], output| {
                    output.copy_from_slice(&[2.0 * x[0]]);
                    Ok(())
                },
            ),
        )
        .unwrap();
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    let solver = AugmentedLagrangian::default().with_inner_optimizer(Bfgs::new(
        BfgsOptions::default().with_max_iterations(Budget::try_from(2).unwrap()),
    ));
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[0.1],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(restorations(&output) > 0, "{output:?}");
    assert!(output.is_success(), "{output:?}");
    assert!((output.x()[0].abs() - 1.0).abs() < 1e-5);
}

#[test]
fn restoration_and_certification_share_iteration_and_evaluation_limits() {
    let mut restored = false;
    // Evaluations are shared with the caller's parent epoch; the outer
    // iteration budget is the method's own option.
    for (kind, termination) in [
        (BudgetKind::Evaluation, OptimizeTermination::EvaluationLimit),
        (BudgetKind::Iteration, OptimizeTermination::IterationLimit),
    ] {
        for limit in 1..70 {
            let constraints = Constraints::new(1)
                .unwrap()
                .equality(
                    ConstraintFn::new(1, |x: &[f64]| Ok(x[0] * x[0] - 1.0)).with_jacobian(
                        |x: &[f64], output| {
                            output.copy_from_slice(&[2.0 * x[0]]);
                            Ok(())
                        },
                    ),
                )
                .unwrap();
            let budget = Budget::try_from(limit).unwrap();
            let options = if kind == BudgetKind::Iteration {
                AugmentedLagrangianOptions::default().with_max_outer_iterations(budget)
            } else {
                AugmentedLagrangianOptions::default()
            };
            let solver = AugmentedLagrangian::new(options).with_inner_optimizer(Bfgs::new(
                BfgsOptions::default().with_max_iterations(Budget::try_from(2).unwrap()),
            ));
            let objective = Counted {
                counts: Arc::new(Counts::default()),
                analytic: true,
            };
            let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut output = OptimizeResult::empty();
            let mut state = SolveState::new();
            let mut budgets = if kind == BudgetKind::Evaluation {
                SolveBudgets::unbounded().with_limit(kind, budget)
            } else {
                SolveBudgets::unbounded()
            };
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(3),
                    &mut budgets,
                )
                .unwrap();
            solver
                .minimize_into(&problem, &[0.1], &mut output, &mut workspace, &mut context)
                .unwrap();
            restored |= restorations(&output) > 0;
            if kind == BudgetKind::Evaluation {
                assert!(context.budget_report(kind).consumed() <= limit);
            } else {
                assert!(output.counts().own().iterations() <= limit);
            }
            assert_eq!(
                output.counts().total().unwrap(),
                context.counts().total().unwrap()
            );
            assert_eq!(output.fun(), output.x()[0] * output.x()[0]);
            if let Some(residual) = output.primal_feasibility() {
                assert_eq!(residual, (output.x()[0] * output.x()[0] - 1.0).abs());
            }
            if !output.is_success() {
                assert_eq!(output.termination(), termination, "{output:?}");
            }
        }
    }
    assert!(restored);
}

#[test]
fn trial_callback_failure_does_not_replace_output_or_commit_trial_state() {
    struct FailingSquare {
        fail: Cell<bool>,
    }
    impl ObjectiveFunction<f64> for FailingSquare {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            g[0] = 123.0;
            if self.fail.get() && x[0] != -2.0 {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::Malformed,
                    "trial derivative failure",
                ));
            }
            g[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    let f = FailingSquare {
        fail: Cell::new(false),
    };
    let constraints = counted_constraint(&Arc::new(Counts::default()), true);
    let problem = ConstrainedProblem::new(&f, &constraints).unwrap();
    let solver = PenaltyMethod::<f64>::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[-2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let before = output.clone();
    f.fail.set(true);
    assert!(matches!(
        (solver.minimize_into(
            &problem,
            &[-2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Malformed,
            ..
        })
    ));
    assert_eq!(output.x(), before.x());
    assert_eq!(output.fun(), before.fun());
    assert_eq!(output.diagnostics(), before.diagnostics());
    assert_eq!(output.counts(), before.counts());
}

#[test]
fn cancellation_during_trial_keeps_the_accepted_snapshot() {
    struct CancellingSquare(Arc<scioxide_core::CancellationToken>);
    impl ObjectiveFunction<f64> for CancellingSquare {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            if x[0] != -2.0 {
                self.0.cancel();
            }
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            g[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    let token = Arc::new(scioxide_core::CancellationToken::new());
    let f = CancellingSquare(Arc::clone(&token));
    let constraints = counted_constraint(&Arc::new(Counts::default()), true);
    let problem = ConstrainedProblem::new(&f, &constraints).unwrap();
    let solver = PenaltyMethod::<f64>::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(token.as_ref()),
            parent_identity(4),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(&problem, &[-2.0], &mut result, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Cancelled);
    assert_eq!(result.x(), &[-2.0]);
    assert_eq!(result.fun(), 4.0);
    assert_eq!(result.primal_feasibility(), Some(3.0));
}

#[test]
fn injected_deadline_during_a_trial_keeps_the_accepted_snapshot() {
    use std::time::Duration;
    struct Clock(AtomicUsize);
    impl scioxide_core::Clock for Clock {
        fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(self.0.load(Ordering::Relaxed) as u64),
            ))
        }
    }
    struct TimedSquare<'a>(&'a Clock);
    impl ObjectiveFunction<f64> for TimedSquare<'_> {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            if x[0] != -2.0 {
                self.0.0.store(1, Ordering::Relaxed);
            }
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            g[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    let clock = Clock(AtomicUsize::new(0));
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let solver = AugmentedLagrangian::new(
        AugmentedLagrangianOptions::default().with_time_limit(Duration::from_millis(1)),
    );
    let constraints = counted_constraint(&Arc::new(Counts::default()), true);
    let objective = TimedSquare(&clock);
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential()
                .with_clock(&clock_binding)
                .unwrap(),
            parent_identity(5),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(&problem, &[-2.0], &mut result, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::TimeLimit);
    assert_eq!(result.x(), &[-2.0]);
    assert_eq!(result.fun(), 4.0);
    assert_eq!(result.primal_feasibility(), Some(3.0));
}

#[test]
fn caller_executor_and_borrowed_objective_preserve_constrained_results() {
    use std::num::NonZeroUsize;
    struct CallerExecutor;
    impl scioxide_core::Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: scioxide_core::ExecutionPlan,
            _: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::Malformed,
                "sequential kernels must not request a worker team",
            ))
        }
    }
    let solver = AugmentedLagrangian::<f64>::default();
    let objective = Counted {
        counts: Arc::new(Counts::default()),
        analytic: true,
    };
    let borrowed: &dyn ObjectiveFunction<f64> = &objective;
    let constraints = counted_constraint(&Arc::new(Counts::default()), true);
    let problem = ConstrainedProblem::new(borrowed, &constraints).unwrap();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut sequential = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[-2.0],
            &mut sequential,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let mut injected = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[-2.0],
            &mut injected,
            &mut workspace,
            ExecutionContext::new(&CallerExecutor),
        )
        .unwrap();
    assert_eq!(sequential.x(), injected.x());
    assert_eq!(sequential.diagnostics(), injected.diagnostics());
    assert_eq!(sequential.counts(), injected.counts());
    assert!(
        injected
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Factorization)
            > 0
    );
}
