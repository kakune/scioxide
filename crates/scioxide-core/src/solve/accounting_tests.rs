//! Boundary and mutation tests for solve accounting.
use super::*;
use alloc::string::ToString;
use core::cell::Cell;

#[test]
fn provider_batches_match_single_attempts_through_ancestors() {
    for &role in EvaluationRole::ALL {
        let run = |batch: bool| {
            let mut state = SolveState::new();
            let mut child_state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut root = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            let own = SolveScope::from(&mut root)
                .run(
                    &mut child_state,
                    identity(),
                    SolveBudgets::unbounded(),
                    |child| {
                        if batch {
                            child.record_providers(role, 17)?.require()?;
                        } else {
                            for _ in 0..17 {
                                child.record_provider(role)?.require()?;
                            }
                        }
                        child.record_providers(role, 0)?.require()?;
                        Ok(child.counts())
                    },
                )
                .unwrap();
            (
                own,
                root.counts(),
                root.budget_report(BudgetKind::Evaluation),
            )
        };
        assert_eq!(run(true), run(false));
    }
}

#[test]
fn provider_batch_denial_and_overflow_preserve_every_ledger() {
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(3));
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut child_state = SolveState::new();
    SolveScope::from(&mut root)
        .run(
            &mut child_state,
            identity(),
            SolveBudgets::unbounded(),
            |child| {
                child
                    .record_providers(EvaluationRole::Function, 2)?
                    .require()?;
                let before = child.counts();
                let budget_before = child.budget_report(BudgetKind::Evaluation);
                assert!(
                    !child
                        .record_providers(EvaluationRole::Function, 2)?
                        .is_granted()
                );
                assert!(
                    child
                        .record_providers(EvaluationRole::Function, usize::MAX)
                        .is_err()
                );
                assert_eq!(child.counts(), before);
                assert_eq!(child.budget_report(BudgetKind::Evaluation), budget_before);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(root.counts().nested().provider(EvaluationRole::Function), 2);
    assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 2);
}

#[test]
fn every_counter_checks_combined_capacity_before_committing_the_chain() {
    for counter in Counter::all() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        root.admit(&Charge::Count {
            counter,
            scope: CountScope::Own,
            amount: usize::MAX - 1,
        })
        .unwrap()
        .require()
        .unwrap();
        let mut child_state = SolveState::new();
        SolveScope::from(&mut root)
            .run(
                &mut child_state,
                identity(),
                SolveBudgets::unbounded(),
                |child| {
                    child.admit(&Charge::count(counter))?.require()?;
                    let before = child.counts();
                    let budgets_before = *child.budgets;
                    let accepted_before = child.accepted_state();
                    match counter {
                        Counter::Iteration => assert!(child.can_record_iterations(1).is_err()),
                        Counter::Proposal => assert!(child.can_record_proposals(1).is_err()),
                        _ => {}
                    }

                    assert_eq!(
                        child.admit(&Charge::count(counter)).unwrap_err().kind(),
                        crate::SciOxideErrorKind::SizeOverflow
                    );
                    assert_eq!(child.counts(), before);
                    assert_eq!(*child.budgets, budgets_before);
                    assert_eq!(child.accepted_state(), accepted_before);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(counter.read(&root.counts().own()), usize::MAX - 1);
        assert_eq!(counter.read(&root.counts().nested()), 1);
        assert_eq!(counter.read(&root.counts().total().unwrap()), usize::MAX);
        if let Some(kind) = counter.budget() {
            let expected = if kind.nesting() == NestingPolicy::Shared {
                usize::MAX
            } else {
                usize::MAX - 1
            };
            assert_eq!(root.budget_report(kind).consumed(), expected);
        }
    }
}

#[test]
fn reserved_headroom_checks_every_counter_and_shared_resource_at_max_minus_one() {
    for counter in Counter::all() {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut cx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        cx.headroom_limit = usize::MAX - 1;
        cx.admit(&Charge::Count {
            counter,
            scope: CountScope::Own,
            amount: usize::MAX - 1,
        })
        .unwrap()
        .require()
        .unwrap();
        let before = cx.counts();
        let budgets_before = *cx.budgets;
        assert_eq!(
            cx.admit(&Charge::count(counter)).unwrap_err().kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(cx.counts(), before);
        assert_eq!(*cx.budgets, budgets_before);
    }
    for &kind in BudgetKind::ALL {
        if kind.nesting() != NestingPolicy::Shared {
            continue;
        }
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut cx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        cx.headroom_limit = usize::MAX - 1;
        cx.try_consume(kind, usize::MAX - 1)
            .unwrap()
            .require()
            .unwrap();
        let before = *cx.budgets;
        assert_eq!(
            cx.can_consume(kind, 1).unwrap_err().kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(
            cx.try_consume(kind, 1).unwrap_err().kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(*cx.budgets, before);
        assert_eq!(cx.counts(), SolveCounts::new());
    }
}

#[test]
fn mixed_provider_sum_survives_reconciliation_and_resume_and_resets_on_a_fresh_epoch() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let checkpoint = {
        let mut cx = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        cx.admit(&Charge::Count {
            counter: Counter::Provider(EvaluationRole::Function),
            scope: CountScope::Own,
            amount: usize::MAX - 2,
        })
        .unwrap()
        .require()
        .unwrap();
        let mut children = [PartitionedSolve::default()];
        with_partitioned_solves(
            &mut cx,
            &mut children,
            SolveBudgets::unbounded(),
            |execution, batch| {
                for permit in batch {
                    permit.run(execution, identity(), |child| {
                        child.record_provider(EvaluationRole::Gradient)?.require()
                    })?;
                }
                Ok(())
            },
        )
        .unwrap();
        cx.checkpoint(0).unwrap()
    };
    {
        let mut cx = state
            .resume(
                ExecutionContext::sequential(),
                identity(),
                0,
                checkpoint,
                &mut budgets,
            )
            .unwrap();
        cx.record_provider(EvaluationRole::Hessian)
            .unwrap()
            .require()
            .unwrap();
        assert_eq!(
            cx.counts().total().unwrap().total_providers().unwrap(),
            usize::MAX
        );
        let before = cx.counts();
        let budgets_before = *cx.budgets;
        assert_eq!(
            cx.record_provider(EvaluationRole::Jacobian)
                .unwrap_err()
                .kind(),
            crate::SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(cx.counts(), before);
        assert_eq!(*cx.budgets, budgets_before);
    }
    let mut cx = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    cx.record_provider(EvaluationRole::Jacobian)
        .unwrap()
        .require()
        .unwrap();
    assert_eq!(cx.counts().total().unwrap().total_providers().unwrap(), 1);
}

#[test]
fn reconciliation_denial_and_overflow_preserve_every_ledger() {
    for overflow in [false, true] {
        for failing_depth in 0..3 {
            let allowance = |depth| {
                if !overflow && depth == failing_depth {
                    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE)
                } else {
                    SolveBudgets::unbounded()
                }
            };
            let seed = |cx: &mut SolveContext<'_, '_>, depth| -> Result<(), SciOxideError> {
                if overflow && depth == failing_depth {
                    cx.record_iterations(usize::MAX - 1)?.require()?;
                }
                Ok(())
            };
            let mut state = SolveState::new();
            let mut budgets = allowance(0);
            let mut root = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            seed(&mut root, 0).unwrap();
            let root_before = *root.budgets;
            let mut child_state = SolveState::new();
            SolveScope::from(&mut root)
                .run(&mut child_state, identity(), allowance(1), |child| {
                    seed(child, 1)?;
                    let budgets_before = *child.budgets;
                    let mut grandchild_state = SolveState::new();
                    SolveScope::from(&mut *child).run(
                        &mut grandchild_state,
                        identity(),
                        allowance(2),
                        |grandchild| {
                            seed(grandchild, 2)?;
                            let before = grandchild.counts();
                            let budgets_before = *grandchild.budgets;
                            let mut resources = [0; BUDGET_KIND_COUNT];
                            let counts = if overflow {
                                EvaluationCounts::from_iterations(2)
                            } else {
                                resources[BudgetKind::Evaluation.index()] = 2;
                                EvaluationCounts::new()
                                    .with_provider_count(EvaluationRole::Function, 2)
                            };
                            let result = grandchild.merge(&resources, &counts);
                            if overflow {
                                assert_eq!(
                                    result.unwrap_err().kind(),
                                    crate::SciOxideErrorKind::SizeOverflow
                                );
                            } else {
                                let Admission::Denied(report) = result? else {
                                    unreachable!()
                                };
                                assert_eq!(report.limit(), BudgetLimit::Finite(1));
                                assert_eq!(report.consumed(), 0);
                            }
                            assert_eq!(grandchild.counts(), before);
                            assert_eq!(*grandchild.budgets, budgets_before);
                            Ok(())
                        },
                    )?;
                    assert_eq!(*child.budgets, budgets_before);
                    assert_eq!(
                        child.counts().total()?.iterations(),
                        if overflow && failing_depth >= 1 {
                            usize::MAX - 1
                        } else {
                            0
                        }
                    );
                    assert_eq!(child.counts().total()?.total_providers()?, 0);
                    Ok(())
                })
                .unwrap();
            assert_eq!(*root.budgets, root_before);
            assert_eq!(
                root.counts().total().unwrap().iterations(),
                if overflow { usize::MAX - 1 } else { 0 }
            );
            assert_eq!(root.counts().total().unwrap().total_providers().unwrap(), 0);
        }
    }
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        crate::algorithm_id!("test", "accounting", 1),
        0,
        0,
    )
}

#[test]
fn converted_scopes_keep_local_epochs_and_charge_ancestors() {
    let mut state = SolveState::new();
    let mut local = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(2));
    // Advance only the ancestor's epoch.
    drop(
        state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap(),
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let accepted = parent.accepted_state();
    for epoch in 1..=2 {
        SolveScope::from(&mut parent)
            .run(
                &mut local,
                identity(),
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
                |child| {
                    assert_eq!(child.accepted_state().unwrap().epoch().get(), epoch);
                    assert!(child.record_iteration()?.is_granted());
                    assert!(!child.record_iteration()?.is_granted());
                    assert!(
                        child
                            .record_provider(EvaluationRole::Function)?
                            .is_granted()
                    );
                    child.commit_accepted(PartialResultTrust::Admissible)
                },
            )
            .unwrap();
    }
    assert_eq!(parent.accepted_state(), accepted);
    assert_eq!(parent.counts().nested().iterations(), 2);
    assert_eq!(parent.budget_report(BudgetKind::Evaluation).consumed(), 2);
    SolveScope::from(&mut parent)
        .run(&mut local, identity(), SolveBudgets::unbounded(), |child| {
            assert!(
                !child
                    .record_provider(EvaluationRole::Function)?
                    .is_granted()
            );
            Ok(())
        })
        .unwrap();
    SolveScope::from(ExecutionContext::sequential())
        .run(&mut local, identity(), SolveBudgets::unbounded(), |root| {
            assert!(root.record_provider(EvaluationRole::Function)?.is_granted());
            assert_eq!(root.counts().nested(), EvaluationCounts::new());
            Ok(())
        })
        .unwrap();
}

#[test]
fn local_child_work_never_consumes_outer_iteration_or_proposal_limits() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Iteration, crate::budget!(3))
        .with_limit(BudgetKind::Proposal, Budget::ONE)
        .with_limit(BudgetKind::Evaluation, crate::budget!(2));
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut child = SolveState::new();
    crate::SolveScope::from(&mut root)
        .run(&mut child, identity(), SolveBudgets::unbounded(), |child| {
            for _ in 0..3 {
                assert_eq!(child.record_iteration()?, Admission::Granted(()));
                assert_eq!(child.record_proposal()?, Admission::Granted(()));
                child.commit_accepted(PartialResultTrust::Admissible)?;
            }
            assert_eq!(
                child.invoke_provider(EvaluationRole::Function, || Ok(4))?,
                Invocation::Evaluated(4)
            );
            let mut grandchild = SolveState::new();
            crate::SolveScope::from(&mut *child).run(
                &mut grandchild,
                identity(),
                SolveBudgets::unbounded(),
                |grandchild| {
                    assert_eq!(
                        grandchild.record_provider(EvaluationRole::Gradient)?,
                        Admission::Granted(())
                    );
                    assert!(matches!(
                        grandchild.record_provider(EvaluationRole::Function)?,
                        Admission::Denied(_)
                    ));
                    Ok(())
                },
            )?;
            assert_eq!(child.counts().own().total_providers()?, 1);
            assert_eq!(child.counts().nested().total_providers()?, 1);
            Ok(())
        })
        .unwrap();
    assert_eq!(root.counts().own(), EvaluationCounts::new());
    assert_eq!(root.counts().nested().iterations(), 3);
    assert_eq!(root.counts().nested().proposals(), 3);
    assert_eq!(root.counts().nested().accepted(), 3);
    assert_eq!(root.counts().total().unwrap().total_providers().unwrap(), 2);
    assert_eq!(root.budget_report(BudgetKind::Iteration).consumed(), 0);
    assert_eq!(root.record_iteration().unwrap(), Admission::Granted(()));
    assert_eq!(root.record_proposal().unwrap(), Admission::Granted(()));
    assert_eq!(root.accepted_state().unwrap().revision(), 0);
}

#[test]
fn provider_sum_capacity_is_checked_at_each_ancestor_before_any_mutation() {
    for saturated_depth in 0..3 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let full = SolveCounts::from_own(
            EvaluationCounts::new().with_provider_count(EvaluationRole::Function, usize::MAX),
        );
        if saturated_depth == 0 {
            *root.state.counts = full;
        }
        let mut child_state = SolveState::new();
        let mut grandchild_state = SolveState::new();
        crate::SolveScope::from(&mut root)
            .run(
                &mut child_state,
                identity(),
                SolveBudgets::unbounded(),
                |child| {
                    if saturated_depth == 1 {
                        *child.state.counts = full;
                    }
                    let child_before = child.counts();
                    crate::SolveScope::from(&mut *child).run(
                        &mut grandchild_state,
                        identity(),
                        SolveBudgets::unbounded(),
                        |grandchild| {
                            if saturated_depth == 2 {
                                *grandchild.state.counts = full;
                            }
                            let before = grandchild.counts();
                            let attempts = Cell::new(0);
                            let result =
                                grandchild.invoke_provider(EvaluationRole::Gradient, || {
                                    attempts.set(attempts.get() + 1);
                                    Ok(())
                                });
                            assert_eq!(
                                match result.unwrap_err().view() {
                                    crate::SciOxideErrorView::SizeOverflow { context, .. } =>
                                        Some(context),
                                    _ => None,
                                }
                                .unwrap(),
                                crate::quantity!("total_provider_evaluations"),
                            );
                            assert_eq!(attempts.get(), 0);
                            assert_eq!(grandchild.counts(), before);
                            assert_eq!(
                                grandchild.budget_report(BudgetKind::Evaluation).consumed(),
                                0
                            );
                            Ok(())
                        },
                    )?;
                    assert_eq!(child.counts(), child_before);
                    assert_eq!(child.budget_report(BudgetKind::Evaluation).consumed(), 0);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 0);
        assert_eq!(
            root.counts(),
            if saturated_depth == 0 {
                full
            } else {
                SolveCounts::new()
            }
        );
    }
}

#[test]
fn numerical_denial_at_every_depth_preserves_the_entire_chain() {
    for limited_depth in 0..3 {
        let allowance = |depth| {
            if depth == limited_depth {
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE)
            } else {
                SolveBudgets::unbounded()
            }
        };
        let mut state = SolveState::new();
        let mut budgets = allowance(0);
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut child_state = SolveState::new();
        let mut grandchild_state = SolveState::new();
        crate::SolveScope::from(&mut root)
            .run(&mut child_state, identity(), allowance(1), |child| {
                crate::SolveScope::from(&mut *child).run(
                    &mut grandchild_state,
                    identity(),
                    allowance(2),
                    |grandchild| {
                        assert_eq!(
                            grandchild.record_provider(EvaluationRole::Function)?,
                            Admission::Granted(())
                        );
                        let before = grandchild.counts();
                        let denial = grandchild.record_provider(EvaluationRole::Gradient)?;
                        let Admission::Denied(report) = denial else {
                            unreachable!()
                        };
                        assert_eq!(report.limit(), BudgetLimit::Finite(1));
                        assert_eq!(report.consumed(), 1);
                        assert_eq!(grandchild.counts(), before);
                        Ok(())
                    },
                )?;
                assert_eq!(child.counts().total()?.total_providers()?, 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(root.counts().total().unwrap().total_providers().unwrap(), 1);
        assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 1);
    }
}

#[test]
fn detached_capacity_is_reserved_even_for_unbounded_numerical_budgets() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    *root.state.counts = SolveCounts::from_own(
        EvaluationCounts::new().with_provider_count(EvaluationRole::Function, usize::MAX - 1),
    );
    root.budgets.consumed[BudgetKind::Evaluation.index()] = usize::MAX - 1;
    let mut storage = core::array::from_fn::<_, 2, _>(|_| PartitionedSolve::default());
    let attempts = Cell::new(0);
    with_partitioned_solves(
        &mut root,
        &mut storage,
        SolveBudgets::unbounded(),
        |execution, children| {
            for (index, child) in children.into_iter().enumerate() {
                child.run(execution, identity(), |context| {
                    let result = context.invoke_provider(EvaluationRole::Gradient, || {
                        attempts.set(attempts.get() + 1);
                        Ok(())
                    });
                    if index == 0 {
                        assert_eq!(result.unwrap(), Invocation::Evaluated(()));
                    } else {
                        assert!(matches!(
                            (result).as_ref().unwrap_err().view(),
                            crate::SciOxideErrorView::SizeOverflow { .. }
                        ));
                    }
                    Ok(())
                })?;
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(attempts.get(), 1);
    assert_eq!(
        root.counts().total().unwrap().total_providers().unwrap(),
        usize::MAX
    );
    assert_eq!(root.counts().nested().provider(EvaluationRole::Gradient), 1);
}

#[test]
fn zero_numerical_share_is_reported_as_finite_zero_without_callback() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut storage = core::array::from_fn::<_, 2, _>(|_| PartitionedSolve::default());
    with_partitioned_solves(
        &mut root,
        &mut storage,
        SolveBudgets::unbounded(),
        |execution, children| {
            let mut permits = children.into_iter();
            let _unused = permits.next().unwrap();
            permits
                .next()
                .unwrap()
                .run(execution, identity(), |context| {
                    let attempts = Cell::new(0);
                    let result = context.invoke_provider(EvaluationRole::Function, || {
                        attempts.set(1);
                        Ok(())
                    })?;
                    let Invocation::BudgetExhausted(report) = result else {
                        unreachable!()
                    };
                    assert_eq!(report.limit(), BudgetLimit::Finite(0));
                    assert_eq!(report.remaining(), Some(0));
                    assert_eq!(
                        crate::ConvergenceDiagnostics::new()
                            .with_budget(report)
                            .to_string(),
                        " [budget=evaluation:0/0]"
                    );
                    assert_eq!(attempts.get(), 0);
                    Ok(())
                })
        },
    )
    .unwrap();
    assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 0);
}

#[test]
fn malformed_first_ledger_does_not_discard_later_admitted_work() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut storage = core::array::from_fn::<_, 3, _>(|_| PartitionedSolve::default());
    let attempts = Cell::new(0);
    let result: Result<(), SciOxideError> = with_partitioned_solves(
        &mut root,
        &mut storage,
        SolveBudgets::unbounded(),
        |_execution, children| {
            let PartitionSolveBatch {
                children,
                execution,
            } = children;
            for storage in children.iter_mut() {
                let child = PartitionSolvePermit { storage, execution };
                child.run(execution, identity(), |context| {
                    assert_eq!(
                        context.invoke_provider(EvaluationRole::Function, || {
                            attempts.set(attempts.get() + 1);
                            Ok(())
                        })?,
                        Invocation::Evaluated(())
                    );
                    Ok(())
                })?;
            }
            // Corruption is impossible through the public permit API. Inject it here
            // to test recovery independently of the ordinary charging implementation.
            children[0].state.counts.nested.providers[EvaluationRole::Function.index()] =
                usize::MAX;
            Err(SciOxideError::execution_failure(
                None,
                crate::ExecutionFailureReason::ProviderFailure,
                "injected failure",
            ))
        },
    );
    assert!(matches!(
        (result).as_ref().unwrap_err().view(),
        crate::SciOxideErrorView::AccountingInvariant {
            partition: Some(0),
            ..
        }
    ));
    assert_eq!(attempts.get(), 3);
    assert_eq!(root.counts().nested().provider(EvaluationRole::Function), 2);
    assert_eq!(root.counts().integrity(), AccountingIntegrity::Invalid);
    assert!(root.counts().total().is_err());
    assert!(root.record_provider(EvaluationRole::Function).is_err());
    assert!(root.checkpoint(0).is_err());
    assert_eq!(
        storage[0].counts().own().provider(EvaluationRole::Function),
        1
    );
}

#[test]
fn successful_reconciliation_preserves_original_execution_and_provider_errors() {
    for execution_failure in [false, true] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut storage = [PartitionedSolve::default()];
        let calls = Cell::new(0);
        let failure = if execution_failure {
            SciOxideError::execution_failure(
                None,
                crate::ExecutionFailureReason::ProviderFailure,
                "original",
            )
        } else {
            SciOxideError::invalid_input(
                crate::argument!("provider"),
                InvalidInputReason::Malformed,
                "original",
            )
        };
        let result: Result<(), SciOxideError> = with_partitioned_solves(
            &mut root,
            &mut storage,
            SolveBudgets::unbounded(),
            |execution, children| {
                let result =
                    children
                        .into_iter()
                        .next()
                        .unwrap()
                        .run(execution, identity(), |context| {
                            context.invoke_provider::<()>(EvaluationRole::Function, || {
                                calls.set(calls.get() + 1);
                                Err(failure.clone())
                            })
                        });
                if execution_failure {
                    let _ = result;
                    Err(failure.clone())
                } else {
                    result.map(|_| ())
                }
            },
        );
        assert_eq!(result.unwrap_err().to_string(), failure.to_string());
        assert_eq!(calls.get(), 1);
        assert_eq!(
            root.counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Function),
            1
        );
    }
}

#[test]
fn tables_cover_every_index_and_policy() {
    for (index, &role) in EvaluationRole::ALL.iter().enumerate() {
        assert_eq!(role.index(), index);
    }
    for (index, &kind) in BudgetKind::ALL.iter().enumerate() {
        assert_eq!(kind.index(), index);
    }
}

#[test]
fn incomplete_nested_batch_retains_its_representable_prefix_in_every_ancestor() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut outer = [PartitionedSolve::default()];
    let mut inner = core::array::from_fn::<_, 3, _>(|_| PartitionedSolve::default());
    let error = with_partitioned_solves(
        &mut root,
        &mut outer,
        SolveBudgets::unbounded(),
        |execution, batch| {
            batch
                .into_iter()
                .next()
                .unwrap()
                .run(execution, identity(), |child| {
                    assert_eq!(
                        child.record_provider(EvaluationRole::Gradient)?,
                        Admission::Granted(())
                    );
                    with_partitioned_solves(
                        child,
                        &mut inner,
                        SolveBudgets::unbounded(),
                        |_execution, batch| {
                            let PartitionSolveBatch {
                                children,
                                execution,
                            } = batch;
                            for storage in children.iter_mut() {
                                let permit = PartitionSolvePermit { storage, execution };
                                permit.run(execution, identity(), |context| {
                                    assert_eq!(
                                        context.record_provider(EvaluationRole::Function)?,
                                        Admission::Granted(())
                                    );
                                    Ok(())
                                })?;
                            }
                            children[0].state.counts.nested.providers
                                [EvaluationRole::Function.index()] = usize::MAX;
                            Ok(())
                        },
                    )
                })
        },
    )
    .unwrap_err();
    assert!(matches!(
        (error).view(),
        crate::SciOxideErrorView::AccountingInvariant { .. }
    ));
    assert_eq!(root.counts().nested().provider(EvaluationRole::Function), 2);
    assert_eq!(root.counts().nested().provider(EvaluationRole::Gradient), 1);
    assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 3);
    assert_eq!(root.counts().integrity(), AccountingIntegrity::Invalid);
    assert!(root.counts().total().is_err());
}

#[test]
fn exceeded_numerical_share_cannot_charge_another_childs_reservation() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, crate::budget!(2));
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut children = core::array::from_fn::<_, 2, _>(|_| PartitionedSolve::default());
    let error = with_partitioned_solves(
        &mut root,
        &mut children,
        SolveBudgets::unbounded(),
        |_execution, batch| {
            let PartitionSolveBatch {
                children,
                execution,
            } = batch;
            for storage in children.iter_mut() {
                let child = PartitionSolvePermit { storage, execution };
                child.run(execution, identity(), |context| {
                    assert_eq!(
                        context.try_consume(BudgetKind::Node, 1)?,
                        Admission::Granted(())
                    );
                    Ok(())
                })?;
            }
            children[0].allowance.consumed[BudgetKind::Node.index()] = 2;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        (error).view(),
        crate::SciOxideErrorView::AccountingInvariant {
            partition: Some(0),
            ..
        }
    ));
    assert_eq!(root.budget_report(BudgetKind::Node).consumed(), 1);
    assert_eq!(root.counts().integrity(), AccountingIntegrity::Invalid);
}

#[test]
fn a_permit_cannot_remove_parent_cancellation() {
    let cancellation = crate::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let mut children = [PartitionedSolve::default()];
    with_partitioned_solves(
        &mut root,
        &mut children,
        SolveBudgets::unbounded(),
        |_, batch| {
            batch.into_iter().next().unwrap().run(
                ExecutionContext::sequential(),
                identity(),
                |child| {
                    assert_eq!(child.stop_reason()?, Some(ExecutionStopReason::Cancelled));
                    Ok(())
                },
            )
        },
    )
    .unwrap();
    assert_eq!(root.counts().total().unwrap(), EvaluationCounts::new());
}

#[test]
fn a_permit_cannot_remove_or_extend_parent_deadline() {
    use crate::{Clock, Deadline, MonotonicInstant};
    use core::time::Duration;
    struct AtDeadline;
    impl Clock for AtDeadline {
        fn now(&self) -> Result<MonotonicInstant, crate::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(2),
            ))
        }
    }
    let clock = AtDeadline;
    let clock_binding = crate::ClockBinding::new(&clock);
    let deadline = Deadline::at(
        &clock_binding,
        MonotonicInstant::from_duration_since_epoch(Duration::from_secs(2)),
    );
    for execution in [
        ExecutionContext::sequential(),
        ExecutionContext::sequential()
            .with_deadline(Deadline::at(
                &clock_binding,
                MonotonicInstant::from_duration_since_epoch(Duration::from_secs(3)),
            ))
            .unwrap(),
    ] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut root = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let mut children = [PartitionedSolve::default()];
        with_partitioned_solves(
            &mut root,
            &mut children,
            SolveBudgets::unbounded(),
            |_, batch| {
                batch
                    .into_iter()
                    .next()
                    .unwrap()
                    .run(execution, identity(), |child| {
                        assert_eq!(
                            child.stop_reason()?,
                            Some(ExecutionStopReason::DeadlineReached)
                        );
                        Ok(())
                    })
            },
        )
        .unwrap();
        assert_eq!(root.counts().total().unwrap(), EvaluationCounts::new());
    }
}

#[test]
fn reusable_partition_storage_never_reuses_an_epoch() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut children = [PartitionedSolve::default()];
    for expected in 1..=3 {
        let epoch = with_partitioned_solves(
            &mut root,
            &mut children,
            SolveBudgets::unbounded(),
            |execution, batch| {
                batch
                    .into_iter()
                    .next()
                    .unwrap()
                    .run(execution, identity(), |child| {
                        Ok(child.accepted_state().unwrap().epoch().get())
                    })
            },
        )
        .unwrap();
        assert_eq!(epoch, expected);
    }
}

#[test]
fn consuming_splits_keep_shares_and_dropped_permits_leave_work_unused() {
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, crate::budget!(3));
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut storage = core::array::from_fn::<_, 4, _>(|_| PartitionedSolve::default());
    with_partitioned_solves(
        &mut root,
        &mut storage,
        SolveBudgets::unbounded(),
        |cx, batch| {
            let (left, right) = batch.split_at(2)?;
            let mut left = left.into_iter();
            let _unused = left.next().unwrap();
            left.next().unwrap().run(cx, identity(), |child| {
                assert_eq!(
                    child.record_provider(EvaluationRole::Function)?,
                    Admission::Granted(())
                );
                Ok(())
            })?;
            for (index, chunk) in right.into_chunks(1)?.enumerate() {
                chunk
                    .into_iter()
                    .next()
                    .unwrap()
                    .run(cx, identity(), |child| {
                        assert_eq!(
                            child
                                .effective_budget_report(BudgetKind::Evaluation)
                                .remaining(),
                            Some(1 - index)
                        );
                        assert_eq!(
                            child
                                .record_provider(EvaluationRole::Function)?
                                .is_granted(),
                            index == 0
                        );
                        Ok(())
                    })?;
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(root.counts().total().unwrap().total_providers().unwrap(), 2);
    assert_eq!(root.budget_report(BudgetKind::Evaluation).consumed(), 2);
    for invalid_split in [true, false] {
        let before = root.counts();
        let result = with_partitioned_solves(
            &mut root,
            &mut storage,
            SolveBudgets::unbounded(),
            |_, batch| {
                if invalid_split {
                    batch.split_at(5).map(|_| ())
                } else {
                    batch.into_chunks(0).map(|_| ())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(root.counts(), before);
    }
}

#[test]
fn a_caught_inner_accounting_failure_cannot_be_reported_as_success() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut child_state = SolveState::new();
    let result = crate::SolveScope::from(&mut root).run(
        &mut child_state,
        identity(),
        SolveBudgets::unbounded(),
        |child| {
            child.invalidate();
            Ok(7)
        },
    );
    assert!(matches!(
        (result).as_ref().unwrap_err().view(),
        crate::SciOxideErrorView::AccountingInvariant { .. }
    ));
    assert_eq!(root.counts().integrity(), AccountingIntegrity::Invalid);
}

#[test]
fn effective_reports_identify_shared_limits_without_conflating_local_budgets() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Evaluation, Budget::ONE)
        .with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert_eq!(
        root.record_provider(EvaluationRole::Function).unwrap(),
        Admission::Granted(())
    );
    assert_eq!(root.record_iteration().unwrap(), Admission::Granted(()));
    let mut state = SolveState::new();
    crate::SolveScope::from(&mut root)
        .run(&mut state, identity(), SolveBudgets::unbounded(), |child| {
            assert_eq!(
                child.budget_report(BudgetKind::Evaluation).limit(),
                BudgetLimit::Unbounded
            );
            let shared = child.effective_budget_report(BudgetKind::Evaluation);
            assert_eq!(shared.limit(), BudgetLimit::Finite(1));
            assert_eq!(shared.consumed(), 1);
            assert_eq!(
                child.record_provider(EvaluationRole::Gradient)?,
                Admission::Denied(shared)
            );
            assert_eq!(
                child.effective_budget_report(BudgetKind::Iteration).limit(),
                BudgetLimit::Unbounded
            );
            assert_eq!(child.record_iteration()?, Admission::Granted(()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn bulk_iterations_are_atomic_and_keep_ancestor_limits_local() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let mut child_state = SolveState::new();
    SolveScope::from(&mut root)
        .run(
            &mut child_state,
            identity(),
            SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, crate::budget!(8)),
            |child| {
                let before = child.counts();
                assert!(child.record_iterations(0)?.is_granted());
                assert_eq!(child.counts(), before);
                assert!(!child.record_iterations(9)?.is_granted());
                assert_eq!(child.counts(), before);
                assert_eq!(child.budget_report(BudgetKind::Iteration).consumed(), 0);
                child.record_iterations(8)?.require()?;
                assert_eq!(child.counts().own().iterations(), 8);
                assert_eq!(child.budget_report(BudgetKind::Iteration).consumed(), 8);
                assert!(child.record_iterations(0)?.is_granted());
                assert!(!child.record_iteration()?.is_granted());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(root.counts().own().iterations(), 0);
    assert_eq!(root.counts().nested().iterations(), 8);
    assert_eq!(root.budget_report(BudgetKind::Iteration).consumed(), 0);
    assert!(root.record_iteration().unwrap().is_granted());
}

#[test]
fn bulk_iteration_overflow_at_any_depth_changes_no_ledger() {
    for saturated_depth in 0..3 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        if saturated_depth == 0 {
            assert!(root.record_iterations(usize::MAX - 1).unwrap().is_granted());
        }
        let root_before = root.counts();
        let mut child_state = SolveState::new();
        SolveScope::from(&mut root)
            .run(
                &mut child_state,
                identity(),
                SolveBudgets::unbounded(),
                |child| {
                    if saturated_depth == 1 {
                        assert!(child.record_iterations(usize::MAX - 1)?.is_granted());
                    }
                    let before = child.counts();
                    let mut grandchild_state = SolveState::new();
                    SolveScope::from(&mut *child).run(
                        &mut grandchild_state,
                        identity(),
                        SolveBudgets::unbounded(),
                        |grandchild| {
                            if saturated_depth == 2 {
                                assert!(grandchild.record_iterations(usize::MAX - 1)?.is_granted());
                            }
                            let before = grandchild.counts();
                            let budget = grandchild.budget_report(BudgetKind::Iteration);
                            assert_eq!(
                                grandchild.record_iterations(2).unwrap_err().kind(),
                                crate::SciOxideErrorKind::SizeOverflow
                            );
                            assert_eq!(grandchild.counts(), before);
                            assert_eq!(grandchild.budget_report(BudgetKind::Iteration), budget);
                            Ok(())
                        },
                    )?;
                    if saturated_depth < 2 {
                        assert_eq!(child.counts(), before);
                    }
                    Ok(())
                },
            )
            .unwrap();
        if saturated_depth == 0 {
            assert_eq!(root.counts(), root_before);
        }
        assert_eq!(root.counts().total().unwrap().iterations(), usize::MAX - 1);
    }
}

#[test]
fn iteration_and_proposal_batches_preflight_without_consuming_ancestors() {
    for kind in [BudgetKind::Iteration, BudgetKind::Proposal] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
        let mut root = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let mut child_state = SolveState::new();
        SolveScope::from(&mut root)
            .run(
                &mut child_state,
                identity(),
                SolveBudgets::unbounded().with_limit(kind, crate::budget!(3)),
                |child| {
                    let before = child.counts();
                    let preflight = |child: &SolveContext<'_, '_>, n| match kind {
                        BudgetKind::Iteration => child.can_record_iterations(n),
                        _ => child.can_record_proposals(n),
                    };
                    assert!(preflight(child, 3)?.is_granted());
                    assert!(!preflight(child, 4)?.is_granted());
                    assert_eq!(child.counts(), before);
                    match kind {
                        BudgetKind::Iteration => child.record_iterations(3)?.require()?,
                        _ => child.record_proposals(3)?.require()?,
                    }
                    assert!(preflight(child, 0)?.is_granted());
                    assert!(!preflight(child, 1)?.is_granted());
                    assert!(preflight(child, usize::MAX).is_err());
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(root.budget_report(kind).consumed(), 0);
        let counts = root.counts().nested();
        assert_eq!(
            if kind == BudgetKind::Iteration {
                counts.iterations()
            } else {
                counts.proposals()
            },
            3
        );
    }
}

#[test]
fn individual_provider_queries_match_totals_and_reject_invalid_or_overflowed_counts() {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut root = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    root.record_provider(EvaluationRole::Objective)
        .unwrap()
        .require()
        .unwrap();
    let mut child_state = SolveState::new();
    SolveScope::from(&mut root)
        .run(
            &mut child_state,
            identity(),
            SolveBudgets::unbounded(),
            |child| {
                child
                    .record_providers(EvaluationRole::Objective, 3)?
                    .require()?;
                child.record_provider(EvaluationRole::Gradient)?.require()?;
                Ok(())
            },
        )
        .unwrap();
    for &role in EvaluationRole::ALL {
        assert_eq!(
            root.provider_count(role).unwrap(),
            root.counts().total().unwrap().provider(role)
        );
    }
    assert_eq!(root.provider_count(EvaluationRole::Objective).unwrap(), 4);
    root.state.counts.own.providers[EvaluationRole::Objective.index()] = usize::MAX;
    assert!(root.provider_count(EvaluationRole::Objective).is_err());
    root.state.counts.integrity = AccountingIntegrity::Invalid;
    assert!(root.provider_count(EvaluationRole::Gradient).is_err());
}
