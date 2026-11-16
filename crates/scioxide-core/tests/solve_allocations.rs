//! Prepared accounting performs no heap allocation on success or failure.
#![allow(clippy::unwrap_used)]

use scioxide_core::backend::{PartitionedSolve, with_partitioned_solves};
use scioxide_core::{
    Admission, Budget, BudgetKind, EvaluationRole, ExecutionContext, InvalidInputReason,
    Invocation, ProblemToken, SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};

const ID: SolveIdentity = SolveIdentity::new(
    ProblemToken::new(1),
    scioxide_core::algorithm_id!("test", "allocations", 1),
    0,
    0,
);

#[test]
fn reborrowed_scopes_keep_local_epochs_and_shared_parent_allowance() {
    let mut state = SolveState::new();
    let mut child_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    {
        let mut parent = state
            .begin_solve(ExecutionContext::sequential(), ID, &mut budgets)
            .unwrap();
        {
            let mut scope = scioxide_core::SolveScope::from(&mut parent);
            for attempt in 0..2 {
                scope
                    .reborrow()
                    .run(&mut child_state, ID, SolveBudgets::unbounded(), |child| {
                        assert_eq!(child.accepted_state().unwrap().epoch().get(), attempt + 1);
                        assert_eq!(
                            child
                                .record_provider(EvaluationRole::Function)?
                                .is_granted(),
                            attempt == 0
                        );
                        Ok(())
                    })
                    .unwrap();
            }
        }
        assert_eq!(parent.counts().own().provider(EvaluationRole::Function), 0);
        assert_eq!(
            parent.counts().nested().provider(EvaluationRole::Function),
            1
        );
    };
}

#[test]
fn root_children_and_detached_ledgers_execute() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut child_state = SolveState::new();
    let mut slots = core::array::from_fn::<_, 3, _>(|_| PartitionedSolve::default());
    {
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), ID, &mut budgets)
            .unwrap();
        let mut attempts = 0;
        scioxide_core::SolveScope::from(&mut root)
            .run(&mut child_state, ID, SolveBudgets::unbounded(), |child| {
                let failed = child.invoke_provider::<()>(EvaluationRole::Function, || {
                    attempts += 1;
                    Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("provider"),
                        InvalidInputReason::Malformed,
                        "fixture",
                    ))
                });
                assert!(matches!(
                    (failed).as_ref().unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::InvalidInput { .. }
                ));
                assert!(matches!(
                    child.invoke_provider(EvaluationRole::Gradient, || {
                        attempts += 1;
                        Ok(())
                    })?,
                    Invocation::BudgetExhausted(_)
                ));
                Ok(())
            })
            .unwrap();
        assert_eq!(attempts, 1);
        assert_eq!(root.counts().own().total_providers().unwrap(), 0);
        assert_eq!(root.counts().total().unwrap().total_providers().unwrap(), 1);
        with_partitioned_solves(
            &mut root,
            &mut slots,
            SolveBudgets::unbounded(),
            |execution, children| {
                for child in children {
                    child.run(execution, ID, |context| {
                        assert!(matches!(
                            context.record_provider(EvaluationRole::Function)?,
                            Admission::Denied(_)
                        ));
                        assert_eq!(context.record_iteration()?, Admission::Granted(()));
                        Ok(())
                    })?;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(root.counts().nested().iterations(), 3);
        assert_eq!(
            root.try_consume(BudgetKind::Node, usize::MAX).unwrap(),
            Admission::Granted(())
        );
        assert!(matches!(
            (root.try_consume(BudgetKind::Node, 1))
                .as_ref()
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
    };
}

#[test]
fn bulk_iteration_admission_and_denial_execute() {
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, scioxide_core::budget!(8));
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), ID, &mut budgets)
        .unwrap();
    {
        assert!(!context.record_iterations(9).unwrap().is_granted());
        assert!(context.record_iterations(8).unwrap().is_granted());
        assert!(context.record_iterations(0).unwrap().is_granted());
        assert!(!context.record_iterations(1).unwrap().is_granted());
    };
    assert_eq!(context.counts().own().iterations(), 8);
}
