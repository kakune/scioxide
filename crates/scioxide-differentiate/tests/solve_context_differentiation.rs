//! Cross-crate regressions for the solve-accounting and differentiation contracts.

#![allow(clippy::unwrap_used)]

use core::time::Duration;

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, CapabilityError, Clock, Deadline, EvaluationRole,
    ExecutionContext, ExecutionStopReason, MonotonicInstant, PartialResultTrust, ProblemToken,
    RealScalar, SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::{
    DerivativeRequest, DerivativeShape, DifferenceScheme, FiniteDifferenceWorkspace,
    ZeroDirectionPolicy,
};

#[path = "support/fixed.rs"]
#[allow(dead_code)]
mod cached;

struct FixedClock(MonotonicInstant);

impl Clock for FixedClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(self.0)
    }
}

fn solve_identity(problem: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(problem),
        scioxide_core::algorithm_id!("contract", "property", 1),
        2,
        3,
    )
}

#[test]
fn jvp_output_obeys_power_of_two_direction_scaling() {
    let derivative = |direction: f64| {
        cached::jvp(
            &scioxide_differentiate::FiniteDifference::default()
                .with_scheme(DifferenceScheme::Central),
            None,
            ZeroDirectionPolicy::MathematicalZero,
            &[0.25],
            &[direction],
            1,
            Some(&[0.5]),
            |point: &[f64], output: &mut [f64]| {
                output[0] = 2.0 * point[0];
                Ok(())
            },
        )
        .map(|report| report.values().unwrap()[0])
    };
    let base = derivative(1.0).expect("ordinary direction must have a stencil");
    [-400, -40, 40, 400].into_iter().for_each(|exponent| {
        let scaled = derivative(1.0_f64.scalbn(exponent))
            .expect("power-of-two direction must have a stencil");
        assert_eq!(scaled.scalbn(-exponent), base);
    });
}

#[test]
fn child_budget_consumption_is_conserved_for_generated_allowances() {
    (1..=13).for_each(|parent_limit| {
        let mut parent = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::try_from(parent_limit).unwrap(),
        );
        let mut parent_state = SolveState::new();
        let mut root = parent_state
            .begin_solve(
                ExecutionContext::sequential(),
                solve_identity(1),
                &mut parent,
            )
            .unwrap();
        let mut accepted_total = 0;
        (1..=7).for_each(|allowance_limit| {
            let allowance = SolveBudgets::unbounded().with_limit(
                BudgetKind::Evaluation,
                Budget::try_from(allowance_limit).unwrap(),
            );
            let mut accepted_child = 0;
            {
                let mut child_state = SolveState::new();
                scioxide_core::SolveScope::from(&mut root)
                    .run(&mut child_state, solve_identity(2), allowance, |child| {
                        (0..allowance_limit + 2).for_each(|_| {
                            if child
                                .try_consume(BudgetKind::Evaluation, 1)
                                .unwrap()
                                .is_granted()
                            {
                                accepted_child += 1;
                            }
                        });
                        assert_eq!(
                            child.budget_report(BudgetKind::Evaluation).consumed(),
                            accepted_child
                        );
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(
                    root.budget_report(BudgetKind::Evaluation).consumed(),
                    accepted_total + accepted_child
                );
            }
            accepted_total += accepted_child;
        });
        let report = root.budget_report(BudgetKind::Evaluation);
        assert_eq!(report.consumed(), accepted_total);
        assert!(accepted_total <= parent_limit);
        assert_eq!(report.remaining(), Some(parent_limit - accepted_total));
    });
}

#[test]
fn cancellation_precedes_an_already_reached_deadline() {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let now = MonotonicInstant::from_duration_since_epoch(Duration::from_secs(2));
    let clock = FixedClock(now);
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &clock_binding,
        MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
    );
    let execution = ExecutionContext::sequential()
        .with_cancellation(&cancellation)
        .with_deadline(deadline)
        .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let context = state
        .begin_solve(execution, solve_identity(1), &mut budgets)
        .unwrap();
    assert_eq!(
        context.stop_reason().unwrap(),
        Some(ExecutionStopReason::Cancelled)
    );
    drop(context);

    cancellation.reset();
    let context = state
        .begin_solve(execution, solve_identity(2), &mut budgets)
        .unwrap();
    assert_eq!(
        context.stop_reason().unwrap(),
        Some(ExecutionStopReason::DeadlineReached)
    );
}

#[test]
fn budget_termination_preserves_the_last_accepted_state_and_exact_counts() {
    (1..=16).for_each(|limit| {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                solve_identity(limit as u128),
                &mut budgets,
            )
            .unwrap();
        while context.record_iteration().unwrap().is_granted() {
            assert!(context.record_proposal().unwrap().is_granted());
            context
                .commit_accepted(PartialResultTrust::Admissible)
                .unwrap();
        }
        let accepted = context.accepted_state().unwrap();
        assert_eq!(context.counts().own().iterations(), limit);
        assert_eq!(context.counts().own().proposals(), limit);
        assert_eq!(context.counts().own().accepted(), limit);
        assert_eq!(accepted.revision(), limit as u64);
        assert_eq!(accepted.trust(), PartialResultTrust::Admissible);
        assert!(context.budget_report(BudgetKind::Iteration).exhausted());
    });
}

#[test]
fn provider_errors_are_counted_once_and_never_reclassified() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            solve_identity(99),
            &mut budgets,
        )
        .unwrap();
    let error = context
        .invoke_provider(EvaluationRole::Residual, || {
            Err::<(), _>(SciOxideError::capability_failure(
                scioxide_core::Capability::Clock,
                "injected provider failure",
            ))
        })
        .unwrap_err();
    assert!(
        matches ! (& (error) , __scioxide_error if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: CapabilityFailure { .. }))
    );
    assert_eq!(context.counts().own().provider(EvaluationRole::Residual), 1);
    assert_eq!(context.counts().own().iterations(), 0);
}

#[test]
fn fresh_and_reused_jvp_workspaces_agree_across_direction_exponents() {
    let x = [0.25_f64, -0.5];
    let base = [0.5_f64, -1.0];
    let method = scioxide_differentiate::FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .with_zero_direction(ZeroDirectionPolicy::MathematicalZero);
    let request = DerivativeRequest::new(&method, DerivativeShape::jvp(2, 2));
    let mut workspace = FiniteDifferenceWorkspace::workspace(request).unwrap();
    let mut reused = [0.0; 2];
    let double = |point: &[f64], output: &mut [f64]| {
        output
            .iter_mut()
            .zip(point)
            .for_each(|(value, point)| *value = 2.0 * point);
        Ok(())
    };
    [-700, -70, 0, 70, 700].into_iter().for_each(|exponent| {
        let direction = [1.0_f64.scalbn(exponent), -1.0_f64.scalbn(exponent)];
        let fresh = cached::jvp(
            &scioxide_differentiate::FiniteDifference::default()
                .with_scheme(DifferenceScheme::Central),
            None,
            ZeroDirectionPolicy::MathematicalZero,
            &x,
            &direction,
            2,
            Some(&base),
            double,
        )
        .unwrap();
        let stats = method
            .jvp_into(
                double,
                &x,
                &direction,
                None,
                Some(&base),
                &mut reused,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(fresh.values().unwrap(), reused.as_slice());
        assert_eq!(fresh.report().evaluations(), stats.evaluations());
    });
}
