#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use core::cell::Cell;
use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, Interval, ProblemToken, SciOxideError,
    SolveBudgets, SolveContext, SolveState,
};
use scioxide_rootfind::{
    Bisection, RootTermination, ScalarFn, ScalarFunction, ScalarRootOptions, ScalarRootSolver,
    Secant,
};

fn limited(kind: BudgetKind, limit: usize) -> SolveBudgets {
    SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap())
}

#[test]
fn scalar_evaluation_role_preserves_admission_and_nested_accounting() {
    use scioxide_rootfind::Brent;

    let calls = Cell::new(0);
    let function = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x * x - 2.0)
    });
    assert_eq!(
        ScalarRootOptions::<f64>::default().evaluation_role(),
        EvaluationRole::Function
    );
    for limit in [2, 100] {
        calls.set(0);
        with_parent(limited(BudgetKind::Evaluation, limit), |parent| {
            let options = ScalarRootOptions::default().with_evaluation_role(EvaluationRole::Event);
            let result = Brent
                .solve_scoped(
                    &function,
                    Interval::new(0.0, 2.0).unwrap(),
                    &options,
                    &mut *parent,
                )
                .unwrap();
            assert_eq!(result.counts().provider(EvaluationRole::Event), calls.get());
            assert_eq!(result.counts().provider(EvaluationRole::Function), 0);
            assert_eq!(
                parent.counts().nested().provider(EvaluationRole::Event),
                calls.get()
            );
            assert_eq!(
                parent.counts().nested().provider(EvaluationRole::Function),
                0
            );
            if limit == 2 {
                assert_eq!(calls.get(), limit);
                assert_eq!(
                    result.termination(),
                    RootTermination::WorkLimit(BudgetKind::Evaluation)
                );
            } else {
                assert!(result.is_success());
            }
        });
    }
}

#[test]
fn scalar_evaluation_role_includes_difference_probes_and_keeps_analytic_derivative_roles() {
    use scioxide_differentiate::DerivativeStrategy;
    use scioxide_rootfind::Newton;

    for strategy in [DerivativeStrategy::Central, DerivativeStrategy::Analytic] {
        let calls = Cell::new(0);
        let derivatives = Cell::new(0);
        let function = ScalarFn::new(|x: f64| {
            calls.set(calls.get() + 1);
            Ok(x * x - 2.0)
        })
        .with_derivative(|x| {
            derivatives.set(derivatives.get() + 1);
            Ok(2.0 * x)
        });
        let options = ScalarRootOptions::default().with_evaluation_role(EvaluationRole::Event);
        let result = Newton::new(strategy)
            .solve(&function, 1.0, &options)
            .unwrap();
        assert!(result.is_success());
        assert_eq!(result.counts().provider(EvaluationRole::Event), calls.get());
        let jacobians = result.counts().provider(EvaluationRole::Jacobian);
        assert!(jacobians > 0);
        if matches!(strategy, DerivativeStrategy::Analytic) {
            assert_eq!(jacobians, derivatives.get());
        } else {
            assert_eq!(derivatives.get(), 0);
        }
        assert_eq!(result.counts().provider(EvaluationRole::Function), 0);
    }
}

/// Run `body` under a root epoch whose budgets are `budgets`.
fn with_parent<T>(budgets: SolveBudgets, body: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("rootfind-test", "parent", 1),
        1,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = budgets;
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    body(&mut context)
}

#[test]
fn scalar_option_budgets_stop_exactly_and_every_call_starts_a_fresh_epoch() {
    let calls = Cell::new(0);
    let function = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x * x - 2.0)
    });
    let options = ScalarRootOptions::default().with_budgets(limited(BudgetKind::Evaluation, 2));
    for epoch in 1..=2 {
        let result = Bisection
            .solve(&function, Interval::new(0.0, 2.0).unwrap(), &options)
            .unwrap();
        assert_eq!(
            result.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(result.counts().provider(EvaluationRole::Function), 2);
        assert_eq!(result.accounting().nested().total_providers().unwrap(), 0);
        assert_eq!(calls.get(), 2 * epoch);
        assert!(result.bracket().is_some());
        assert!(result.residual().is_finite());
    }
    // An iteration allowance below `max_iterations` stops first.
    let options = ScalarRootOptions::default().with_budgets(limited(BudgetKind::Iteration, 3));
    let result = Bisection
        .solve(&function, Interval::new(0.0, 2.0).unwrap(), &options)
        .unwrap();
    assert_eq!(
        result.termination(),
        RootTermination::WorkLimit(BudgetKind::Iteration)
    );
    assert_eq!(result.counts().iterations(), 3);
}

#[test]
fn failed_scalar_provider_is_charged_and_preflight_rejection_calls_nothing() {
    let calls = Cell::new(0);
    let function = ScalarFn::new(|_: f64| -> Result<f64, SciOxideError> {
        calls.set(calls.get() + 1);
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("sentinel"),
            scioxide_core::InvalidInputReason::Malformed,
            "provider failure",
        ))
    });
    let options = ScalarRootOptions::default();
    assert_eq!(
        Secant
            .solve(&function, (f64::NAN, 1.0), &options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(calls.get(), 0);
    with_parent(SolveBudgets::unbounded(), |parent| {
        let error = Bisection
            .solve_scoped(
                &function,
                Interval::new(0.0, 2.0).unwrap(),
                &options,
                &mut *parent,
            )
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert_eq!(calls.get(), 1);
        assert_eq!(
            parent.counts().nested().provider(EvaluationRole::Function),
            1
        );
    });
}

#[test]
fn cancellation_after_initial_observation_returns_a_finite_scalar_state() {
    let cancellation = scioxide_core::CancellationToken::new();
    let function = ScalarFn::new(|x: f64| {
        if x == 2.0 {
            cancellation.cancel();
        }
        Ok(x * x - 2.0)
    });
    let result = Bisection
        .solve_scoped(
            &function,
            Interval::new(0.0, 2.0).unwrap(),
            &ScalarRootOptions::default(),
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Cancelled);
    assert_eq!(result.counts().provider(EvaluationRole::Function), 2);
    assert!(result.residual().is_finite());
}

#[test]
fn deadline_after_a_trial_publishes_the_latest_evaluated_bracket() {
    use core::sync::atomic::{AtomicU64, Ordering};
    use scioxide_core::{CapabilityError, Clock, ClockBinding, Deadline, MonotonicInstant};
    struct ManualClock(AtomicU64);
    impl Clock for ManualClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(self.0.load(Ordering::Relaxed)),
            ))
        }
    }
    let clock = ManualClock(AtomicU64::new(0));
    let binding = ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &binding,
        MonotonicInstant::from_duration_since_epoch(core::time::Duration::from_nanos(1)),
    );
    let function = ScalarFn::new(|x: f64| {
        if x == 1.0 {
            clock.0.store(1, Ordering::Relaxed);
        }
        Ok(x * x - 2.0)
    });
    let result = Bisection
        .solve_scoped(
            &function,
            Interval::new(0.0, 2.0).unwrap(),
            &ScalarRootOptions::default(),
            ExecutionContext::sequential()
                .with_deadline(deadline)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(result.termination(), RootTermination::TimeLimit);
    assert_eq!(result.counts().provider(EvaluationRole::Function), 3);
    let bracket = result.bracket().unwrap();
    assert!((bracket.lower() - 1.0).abs() <= f64::EPSILON);
    assert!((bracket.upper() - 2.0).abs() <= f64::EPSILON);
    assert_eq!(
        bracket.verify(&function).unwrap(),
        scioxide_rootfind::BracketVerification::Verified
    );
}

#[test]
fn child_solves_charge_their_parent_and_obey_its_shared_limits() {
    let function = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
    let options = ScalarRootOptions::default();
    with_parent(limited(BudgetKind::Evaluation, 5), |parent| {
        let first = Bisection
            .solve_scoped(
                &function,
                Interval::new(0.0, 2.0).unwrap(),
                &options,
                &mut *parent,
            )
            .unwrap();
        assert_eq!(
            first.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(first.counts().provider(EvaluationRole::Function), 5);
        // The parent's shared allowance is spent: a second child cannot evaluate.
        let second = Bisection
            .solve_scoped(
                &function,
                Interval::new(0.0, 2.0).unwrap(),
                &options,
                &mut *parent,
            )
            .unwrap_err();
        assert_eq!(
            second.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(
            parent.counts().nested().provider(EvaluationRole::Function),
            5
        );
    });
}

/// `√2` by bisection in a root or child epoch of `scope`.
fn sqrt2<'scope, 'capability>(
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<f64, SciOxideError> {
    let root = Bisection.solve_scoped(
        &ScalarFn::new(|y: f64| Ok(y * y - 2.0)),
        Interval::new(1.0, 2.0)?,
        &ScalarRootOptions::default(),
        scope,
    )?;
    Ok(root.point())
}

/// `f(x) = x - √2`, with `√2` found by a nested solve on every call.
struct NestedRoot;
impl ScalarFunction<f64> for NestedRoot {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(x - sqrt2(ExecutionContext::sequential())?)
    }
    fn eval_scoped(
        &self,
        x: f64,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<f64, SciOxideError> {
        Ok(x - sqrt2(context)?)
    }
}

#[test]
fn scoped_callbacks_report_nested_work_in_the_result() {
    let result = Bisection
        .solve(
            &NestedRoot,
            Interval::new(0.0, 3.0).unwrap(),
            &ScalarRootOptions::default(),
        )
        .unwrap();
    assert!(result.is_success());
    assert!((result.point() - 2.0_f64.sqrt()).abs() < 1e-12);
    let own = result.counts().provider(EvaluationRole::Function);
    let nested = result
        .accounting()
        .nested()
        .provider(EvaluationRole::Function);
    assert!(own > 2);
    // Every outer evaluation ran one complete inner bisection.
    assert_eq!(nested % own, 0);
    assert!(nested > own);
}

#[test]
fn borrowed_system_initialization_and_budget_stop_reuse_the_epoch() {
    use scioxide_differentiate::DerivativeStrategy;
    use scioxide_rootfind::{NewtonSystem, ResidualFn, SystemRootResult, SystemRootSolver};
    let calls = Cell::new(0);
    let function = ResidualFn::new(1, |x: &[f64], output: &mut [f64]| {
        calls.set(calls.get() + 1);
        output[0] = x[0] * x[0] - 2.0;
        Ok(())
    })
    .with_jacobian(|x: &[f64], output: &mut [f64]| {
        output[0] = 2.0 * x[0];
        Ok(())
    });
    let solver = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();

    let point = [1.0];
    let mut workspace = solver.workspace(&function).unwrap();
    let mut result = SystemRootResult::empty();
    result.try_reserve(1).unwrap();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(20),
        scioxide_core::algorithm_id!("test", "dense-parent", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    for epoch in 1..=2 {
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        solver
            .solve_into(&function, &point, &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(
            result.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(result.point(), &point);
        assert_eq!(
            context.counts().total().unwrap(),
            result.accounting().total().unwrap()
        );
        assert_eq!(calls.get(), epoch);
        assert_eq!(result.counts().provider(EvaluationRole::Residual), 1);
        assert_eq!(result.counts().provider(EvaluationRole::Jacobian), 0);
    }
}

#[test]
fn polynomial_budget_stop_publishes_partial_candidates_and_exact_counts() {
    use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace};
    let coefficients = [-1.0_f64, 0.0, 0.0, 1.0];
    let sentinel = scioxide_core::Complex::new(9.0, -3.0);
    let mut output = [sentinel; 3];
    let options = PolynomialRootOptions::default();
    let method = PolynomialRootMethod::Aberth;
    let mut workspace = PolynomialRootWorkspace::new();
    PolynomialRootWorkspace::workspace_requirements(
        3,
        &PolynomialRootOptions::<f64>::default().with_method(method),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(21),
        scioxide_core::algorithm_id!("test", "polynomial-parent", 1),
        3_u128,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    for _ in 0..2 {
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let report = scioxide_rootfind::polynomial_roots_into(
            &coefficients,
            &options.clone().with_method(method),
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            scioxide_rootfind::RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_ne!(output, [sentinel; 3]);
        assert!(
            workspace
                .diagnostics()
                .iter()
                .all(|d| d.is_available() && d.backward_error().is_none())
        );
        // Nothing was assessed, so the report claims no backward error.
        assert_eq!(report.backward_error(), None);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            1
        );
    }
}

#[test]
fn bracket_verification_rejects_changed_observations() {
    let function = ScalarFn::new(|x: f64| Ok(x - 1.0));
    let result = Bisection
        .solve(
            &function,
            Interval::new(0.0, 2.0).unwrap(),
            &ScalarRootOptions::default(),
        )
        .unwrap();
    let certificate = result.bracket().unwrap();
    assert_eq!(
        certificate.verify(&function).unwrap(),
        scioxide_rootfind::BracketVerification::Verified
    );
    assert_eq!(
        certificate
            .verify(&ScalarFn::new(|x: f64| Ok(x + 1.0)))
            .unwrap(),
        scioxide_rootfind::BracketVerification::Rejected
    );
}

#[test]
fn polynomial_workspace_requirements_follow_the_selected_family() {
    use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace};
    let analytic = PolynomialRootWorkspace::workspace_requirements(
        2,
        &PolynomialRootOptions::<f64>::default(),
    )
    .unwrap();
    // Seed/root radii, real coefficients and a real matrix accompany complex storage.
    assert_eq!(analytic.real_elements(), 11);
    assert_eq!(analytic.complex_elements(), 15);
    assert_eq!(analytic.companion_dimension(), 2);
    let iterative = PolynomialRootWorkspace::workspace_requirements(
        100,
        &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
    )
    .unwrap();
    assert_eq!(iterative.real_elements(), 200);
    assert_eq!(iterative.complex_elements(), 501);
    assert_eq!(iterative.companion_dimension(), 0);
    let auto = PolynomialRootWorkspace::workspace_requirements(
        100,
        &PolynomialRootOptions::<f64>::default(),
    )
    .unwrap();
    assert_eq!(auto.real_elements(), 200);
    assert_eq!(auto.companion_dimension(), 0);
    assert_eq!(
        PolynomialRootWorkspace::workspace_requirements(
            0,
            &PolynomialRootOptions::<f64>::default()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        PolynomialRootWorkspace::workspace_requirements(
            usize::MAX,
            &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth)
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
}

#[test]
fn companion_solves_the_reduced_degree_after_exact_zero_factors() {
    use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions};
    // x²(x³ - 1) has two exact zero roots and a cubic companion problem.
    let result = scioxide_rootfind::polynomial_roots_with_options(
        &[0.0_f64, 0.0, -1.0, 0.0, 0.0, 1.0],
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Companion),
    )
    .unwrap();
    assert_eq!(result.roots().len(), 5);
    assert_eq!(
        result
            .roots()
            .iter()
            .filter(|root| root.norm() == 0.0)
            .count(),
        2
    );
    for root in result.roots() {
        let value = *root * *root * (*root * *root * *root - scioxide_core::Complex::new(1.0, 0.0));
        assert!(value.norm() < 1e-8);
    }
}
