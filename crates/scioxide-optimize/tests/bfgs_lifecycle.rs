//! BFGS attempt, checkpoint, capacity and scoped accounting contracts.

#![allow(clippy::unwrap_used, clippy::panic)]

use core::cell::Cell;

use scioxide_core::{
    AttemptKind, AttemptOutcome, Capacity, EvaluationRole, ExecutionContext, InvalidInputReason,
    ProblemToken, SciOxideError, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_optimize::{
    Bfgs, BfgsOptions, ObjectiveFunction, OptimizeDiagnostics, OptimizeResult,
};

#[test]
fn unrepresentable_gradient_norm_is_rejected_before_a_trial() {
    struct LargeGradient(Cell<usize>);
    impl ObjectiveFunction<f64> for LargeGradient {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.0.set(self.0.get() + 1);
            scioxide_core::checked_sum_of_products(&[f64::MAX; 2], x)
        }
        fn gradient_into(&self, _: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            out.fill(f64::MAX);
            Ok(true)
        }
    }
    let objective = LargeGradient(Cell::new(0));
    let solver = Bfgs::<f64>::default();
    let mut workspace = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    let error = solver
        .minimize_into(
            &objective,
            &[0.; 2],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
    ));
    assert_eq!(objective.0.get(), 1);
    assert!(output.x().is_empty());
}

struct Quadratic {
    calls: Cell<usize>,
    fail_at: Option<usize>,
}

impl Quadratic {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at,
        }
    }
}

impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                InvalidInputReason::Malformed,
                "injected objective failure",
            ));
        }
        Ok(x[0] * x[0])
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * x[0];
        Ok(true)
    }
}

#[test]
fn failures_and_preflight_rejection_isolate_attempts_and_output() {
    let solver = Bfgs::<f64>::default();
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &Quadratic::new(None),
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let original_x = output.x().to_vec();
    let original_value = output.fun();
    let original_termination = output.termination();

    let first = Quadratic::new(Some(1));
    let error = solver
        .minimize_into(
            &first,
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(first.calls.get(), 1);
    assert_eq!(workspace.last_attempt().unwrap().kind(), AttemptKind::Call);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert!(workspace.last_accepted().is_none());
    assert_eq!(output.x(), original_x);
    assert_eq!(output.fun(), original_value);

    let later = Quadratic::new(Some(2));
    let error = solver
        .minimize_into(
            &later,
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert_eq!(workspace.last_accepted().unwrap().point(), &[2.0]);
    assert_eq!(workspace.last_accepted().unwrap().progress().value(), 4.0);
    let checkpoint = workspace.checkpoint().unwrap();
    let metadata = checkpoint.metadata();

    let mismatch = solver
        .minimize_into(
            &Quadratic::new(None),
            &[1.0, 2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(mismatch.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(workspace.last_accepted().is_none());
    assert_eq!(output.x(), original_x);
    assert_eq!(output.fun(), original_value);
    assert_eq!(output.termination(), original_termination);
    assert_eq!(checkpoint.point(), &[2.0]);
    assert_eq!(checkpoint.metadata(), metadata);
}

#[test]
fn owned_checkpoint_resumes_and_failed_validation_preserves_snapshot() {
    let solver = Bfgs::<f64>::default();
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    let _ = solver
        .minimize_into(
            &Quadratic::new(Some(2)),
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let checkpoint = workspace.checkpoint().unwrap();

    let changed = Bfgs::new(
        BfgsOptions::<f64>::default()
            .with_workspace_byte_limit(solver.workspace_requirements(1).unwrap().bytes() + 1),
    );
    let mut destination = changed.workspace(1).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    let before = checkpoint.metadata();
    let error = changed
        .resume_into(
            &Quadratic::new(None),
            &checkpoint,
            &mut result,
            &mut destination,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(
        destination.last_attempt().unwrap().kind(),
        AttemptKind::Resume
    );
    assert_eq!(
        destination.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(destination.last_accepted().is_none());
    assert_eq!(checkpoint.metadata(), before);

    let mut destination = solver.workspace(1).unwrap();
    solver
        .resume_into(
            &Quadratic::new(None),
            &checkpoint,
            &mut result,
            &mut destination,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.is_success());
    assert!(result.fun() <= 1e-12);
    assert_eq!(
        destination.last_attempt().unwrap().kind(),
        AttemptKind::Resume
    );
    assert_eq!(
        destination.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    assert_eq!(checkpoint.point(), &[2.0]);
}

#[test]
fn requirements_check_overflow_limits_retained_capacity_and_reuse() {
    let solver = Bfgs::<f64>::default();
    assert_eq!(
        solver
            .workspace_requirements(usize::MAX)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let large = solver.workspace_requirements(10).unwrap();
    let mut workspace = solver.workspace(10).unwrap();
    assert_eq!(workspace.check(&large).unwrap(), Capacity::Sufficient);
    workspace.try_reserve(&large).unwrap();
    let small = Bfgs::new(
        BfgsOptions::<f64>::default()
            .with_workspace_byte_limit(solver.workspace_requirements(1).unwrap().bytes()),
    );
    let requirement = small.workspace_requirements(1).unwrap();
    assert_eq!(
        workspace.check(&requirement).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        small.workspace_requirements(10).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn child_attempt_keeps_local_epoch_and_parent_counts_separate() {
    let solver = Bfgs::<f64>::default();
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(9),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        1,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    solver
        .minimize_into(
            &Quadratic::new(None),
            &[2.0],
            &mut output,
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    let first = workspace.last_attempt().unwrap().epoch().unwrap();
    assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Objective),
        output.counts().own().provider(EvaluationRole::Objective)
    );
    solver
        .minimize_into(
            &Quadratic::new(None),
            &[1.0],
            &mut output,
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_ne!(workspace.last_attempt().unwrap().epoch().unwrap(), first);
    assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Objective),
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .own()
            .provider(EvaluationRole::Objective)
            + 3
    );
}

#[test]
fn owned_and_prepared_results_share_first_order_diagnostics() {
    let solver = Bfgs::<f64>::default();
    let owned = solver.minimize(&Quadratic::new(None), &[2.0]).unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    let mut prepared = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &Quadratic::new(None),
            &[2.0],
            &mut prepared,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    for result in [&owned, &prepared] {
        let OptimizeDiagnostics::FirstOrder(globalization) = result.method_diagnostics() else {
            panic!("BFGS publishes first-order diagnostics");
        };
        assert!(globalization.updates() + globalization.damped_updates() >= 1);
    }
}

#[test]
fn resumed_child_charges_only_new_work_to_parent() {
    let solver = Bfgs::<f64>::default();
    let mut first_workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    let _ = solver
        .minimize_into(
            &Quadratic::new(Some(2)),
            &[2.0],
            &mut output,
            &mut first_workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let checkpoint = first_workspace.checkpoint().unwrap();
    let old_objectives = checkpoint
        .metadata()
        .counts()
        .own()
        .provider(EvaluationRole::Objective);

    let mut parent_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(10),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        1,
        0,
    );
    let mut parent = parent_state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    solver
        .resume_into(
            &Quadratic::new(None),
            &checkpoint,
            &mut output,
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert!(output.is_success());
    assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Objective),
        output.counts().own().provider(EvaluationRole::Objective) - old_objectives
    );
}

#[test]
fn derivative_candidates_preserve_bfgs_fallback_counts_and_provider_errors() {
    use scioxide_differentiate::DerivativeStrategy;
    struct Providers {
        analytic: bool,
        automatic: bool,
        fail: bool,
        evaluations: Cell<usize>,
        analytic_calls: Cell<usize>,
        automatic_calls: Cell<usize>,
    }
    impl ObjectiveFunction<f64> for Providers {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.evaluations.set(self.evaluations.get() + 1);
            Ok(x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            self.analytic_calls.set(self.analytic_calls.get() + 1);
            if self.fail {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("gradient"),
                    InvalidInputReason::Malformed,
                    "analytic provider failed",
                ));
            }
            output[0] = 2.0 * x[0];
            Ok(self.analytic)
        }
        fn automatic_gradient_into(
            &self,
            x: &[f64],
            output: &mut [f64],
        ) -> Result<bool, SciOxideError> {
            self.automatic_calls.set(self.automatic_calls.get() + 1);
            output[0] = 2.0 * x[0];
            Ok(self.automatic)
        }
    }
    for (strategy, analytic, automatic, fail) in [
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            true,
            true,
            false,
        ),
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            false,
            true,
            false,
        ),
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            false,
            false,
            false,
        ),
        (DerivativeStrategy::Central, false, false, false),
        (
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
            false,
            true,
            true,
        ),
    ] {
        let objective = Providers {
            analytic,
            automatic,
            fail,
            evaluations: Cell::new(0),
            analytic_calls: Cell::new(0),
            automatic_calls: Cell::new(0),
        };
        let solver = Bfgs::new(BfgsOptions::default().with_derivative_strategy(strategy));
        let result = solver.minimize(&objective, &[2.0]);
        if fail {
            let error = result.unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::Malformed,
                    ..
                }
            ));
            assert_eq!(objective.analytic_calls.get(), 1);
            assert_eq!(objective.automatic_calls.get(), 0);
        } else {
            let result = result.unwrap();
            assert!(result.is_success());
            assert!(result.x()[0].abs() < 1e-6);
            assert_eq!(result.fun_evals(), objective.evaluations.get());
            if strategy == DerivativeStrategy::Central {
                assert_eq!(objective.analytic_calls.get(), 0);
                assert_eq!(objective.automatic_calls.get(), 0);
            } else if analytic {
                assert!(objective.analytic_calls.get() > 0);
                assert_eq!(objective.automatic_calls.get(), 0);
            } else {
                assert!(objective.analytic_calls.get() > 0);
                assert_eq!(
                    objective.analytic_calls.get(),
                    objective.automatic_calls.get()
                );
            }
        }
    }
}

#[test]
fn default_line_search_methods_use_accurate_numerical_gradients() {
    use scioxide_differentiate::DerivativeStrategy;
    use scioxide_optimize::{
        Lbfgs, LbfgsOptions, NonlinearCg, NonlinearCgOptions, ObjectiveFn, OptimizeTermination,
    };
    let rosenbrock = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
    });
    let bowl = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
    });
    for results in [
        [
            Bfgs::default().minimize(&rosenbrock, &[-1.2, 1.0]),
            Lbfgs::default().minimize(&rosenbrock, &[-1.2, 1.0]),
            NonlinearCg::default().minimize(&rosenbrock, &[-1.2, 1.0]),
        ],
        [
            Bfgs::default().minimize(&bowl, &[0.0, 0.0]),
            Lbfgs::default().minimize(&bowl, &[0.0, 0.0]),
            NonlinearCg::default().minimize(&bowl, &[0.0, 0.0]),
        ],
    ] {
        for result in results {
            let result = result.unwrap();
            // Near the Rosenbrock minimum the last line search may fail below
            // the noise of the objective; the gradient is still accurate.
            assert!(
                result.termination() == OptimizeTermination::GradientSmall
                    || result.stationarity().is_some_and(|g| g < 1e-6)
            );
            assert!(result.fun() < 1e-12);
        }
    }
    let analytic = bowl.with_gradient(|x, g| {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] + 2.0);
        Ok(())
    });
    for (central, forward) in [
        (
            Bfgs::default().minimize(&analytic, &[0.0, 0.0]),
            Bfgs::new(
                BfgsOptions::default()
                    .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
            )
            .minimize(&analytic, &[0.0, 0.0]),
        ),
        (
            Lbfgs::default().minimize(&analytic, &[0.0, 0.0]),
            Lbfgs::new(
                LbfgsOptions::default()
                    .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
            )
            .minimize(&analytic, &[0.0, 0.0]),
        ),
        (
            NonlinearCg::default().minimize(&analytic, &[0.0, 0.0]),
            NonlinearCg::new(
                NonlinearCgOptions::default()
                    .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
            )
            .minimize(&analytic, &[0.0, 0.0]),
        ),
    ] {
        let (central, forward) = (central.unwrap(), forward.unwrap());
        assert_eq!(central.x(), forward.x());
        assert_eq!(central.fun().to_bits(), forward.fun().to_bits());
        assert_eq!(central.counts(), forward.counts());
    }
}
