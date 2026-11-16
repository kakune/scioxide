//! Atomic publication, progressive attempts, typed capacity and retained-memory boundaries.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use scioxide_core::{
    AttemptKind, AttemptOutcome, Budget, Capacity, ConvergenceReason, ExecutionContext,
    InvalidInputReason, JacobianOperator, ProblemToken, ResidualOperator, SciOxideError,
    SciOxideErrorView, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_rootfind::{
    Continuation, ContinuationOptions, ContinuationPath, ContinuationProblem, NewtonSystem,
    NewtonSystemOptions, ResidualFn, SystemRootResult, SystemRootSolver,
};

struct Fold;
impl ResidualOperator<f64> for Fold {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0] - x[1];
        Ok(())
    }
}
impl JacobianOperator<f64> for Fold {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = 2.0 * x[0] * v[0] - v[1];
        Ok(())
    }
}
fn continuation(distance: f64) -> Continuation<f64> {
    Continuation::new(
        ContinuationPath::PseudoArclength(distance),
        ContinuationOptions::new(0.1, 1e-6, 0.2, Budget::try_from(32).unwrap()).unwrap(),
    )
    .unwrap()
}

#[test]
fn dense_empty_storage_and_retained_memory_have_explicit_boundaries() {
    let mut output = SystemRootResult::<f64>::empty();
    assert_eq!(
        output.termination(),
        scioxide_rootfind::RootTermination::NotStarted
    );
    assert_eq!(output.residual_norm(), None);
    assert!(!output.is_success());
    let function = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out.copy_from_slice(x);
        Ok(())
    })
    .with_jacobian(|_: &[f64], out: &mut [f64]| {
        out.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    });
    let options = NewtonSystemOptions::default();
    let solver = NewtonSystem::new(options.clone()).unwrap();
    let request = solver.workspace_requirements(&function).unwrap();
    let bounded = NewtonSystem::new(
        options
            .clone()
            .with_workspace_byte_limit(request.required_bytes()),
    )
    .unwrap();
    let mut workspace = bounded.workspace(&function).unwrap();
    assert_eq!(workspace.check(&request).unwrap(), Capacity::Sufficient);
    output.try_reserve(2).unwrap();
    bounded
        .solve_into(
            &function,
            &[1.0, 2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(output.is_success());
    let epoch = workspace.last_attempt().unwrap().epoch().unwrap().get();
    let changed = NewtonSystem::new(options.clone().with_residual_scale(2.0)).unwrap();
    assert_eq!(
        workspace
            .check(&changed.workspace_requirements(&function).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
    changed
        .solve_into(
            &function,
            &[1.0, 2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        workspace.last_attempt().unwrap().epoch().unwrap().get(),
        epoch + 1
    );
    let too_small =
        NewtonSystem::new(options.with_workspace_byte_limit(request.required_bytes() - 1)).unwrap();
    assert!(matches!(
        too_small
            .workspace_requirements(&function)
            .unwrap_err()
            .view(),
        SciOxideErrorView::ResourceLimit { .. }
    ));
    output.try_reserve(3).unwrap();
    assert_eq!(
        output.termination(),
        scioxide_rootfind::RootTermination::NotStarted
    );
    assert_eq!(output.residual_norm(), None);
}
#[test]
fn rejected_continuation_attempts_hide_stale_progress_and_preserve_parent_work() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let plan = continuation(1.0);
    let mut state = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    plan.advance(
        &problem,
        &mut state,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let checkpoint = state.checkpoint().unwrap();
    assert!(workspace.last_accepted().is_some());
    let before = state.point().to_vec();
    let counts = state.accounting();
    let other = continuation(2.0);
    let mut parent_state = SolveState::new();
    let mut allowance = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut parent = parent_state
        .begin_solve(ExecutionContext::sequential(), identity, &mut allowance)
        .unwrap();
    let parent_before = parent.counts();
    assert!(
        other
            .advance(
                &problem,
                &mut state,
                |_, _| Ok(()),
                &mut workspace,
                &mut parent
            )
            .is_err()
    );
    assert_eq!(state.point(), before);
    assert_eq!(state.accounting(), counts);
    assert_eq!(parent.counts(), parent_before);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(workspace.last_accepted().is_none());
    assert!(
        other
            .resume(
                &problem,
                &checkpoint,
                &mut state,
                |_, _| Ok(()),
                &mut workspace,
                &mut parent
            )
            .is_err()
    );
    assert_eq!(checkpoint.metadata().counts(), counts);
    assert_eq!(parent.counts(), parent_before);
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        AttemptKind::Resume
    );
    assert!(workspace.last_accepted().is_none());
}
#[test]
fn initial_numerical_failure_is_counted_convergence_failure() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let plan = continuation(1.0);
    let mut state = plan.state(&problem, &[0.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let error = plan
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::Stagnation,
            ..
        }
    ));
    assert_eq!(state.point(), &[0.0, 1.0]);
    assert!(state.history().is_empty());
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert!(attempt.counts().total().unwrap().total_providers().unwrap() > 0);
    assert!(workspace.last_accepted().is_none());
}
#[test]
fn sink_failure_retains_accepted_state_and_can_continue() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let plan = continuation(1.0);
    let mut state = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let mut emitted = Vec::new();
    let error = plan
        .advance(
            &problem,
            &mut state,
            |point, _| {
                emitted.push(point.to_vec());
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("sink"),
                    InvalidInputReason::Malformed,
                    "sink sentinel",
                ))
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("sink"))
    );
    assert_eq!(state.history().len(), 1);
    assert_eq!(emitted, vec![state.point().to_vec()]);
    assert_eq!(workspace.last_accepted().unwrap(), state.point());
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    plan.advance(
        &problem,
        &mut state,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(state.history().len(), 2);
    assert_eq!(emitted.len(), 1);
}

#[test]
fn continuation_resumes_after_an_ancestor_budget_stop_without_recharging_history() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let plan = continuation(1.0);
    let mut state = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    plan.advance(
        &problem,
        &mut state,
        |_, _| Ok(()),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let identity = SolveIdentity::new(
        ProblemToken::new(2),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut first_parent = SolveState::new();
    let mut allowance =
        SolveBudgets::unbounded().with_limit(scioxide_core::BudgetKind::Evaluation, Budget::ONE);
    let mut parent = first_parent
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity,
            &mut allowance,
        )
        .unwrap();
    let stopped = plan
        .advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        stopped.termination(),
        scioxide_rootfind::RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
    );
    let checkpoint = state.checkpoint().unwrap();
    let archived = checkpoint.metadata().counts().total().unwrap();
    let mut restored = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let mut next_parent = SolveState::new();
    let mut allowance = SolveBudgets::unbounded();
    let mut parent = next_parent
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity,
            &mut allowance,
        )
        .unwrap();
    let report = plan
        .resume(
            &problem,
            &checkpoint,
            &mut restored,
            |_, _| Ok(()),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        scioxide_rootfind::RootTermination::Advanced
    );
    assert_eq!(restored.history().len(), 2);
    assert!(restored.point()[1] > 1.0);
    let original = restored.point()[0] * restored.point()[0] - restored.point()[1];
    assert!(original.abs() < 1e-8);
    let new_work = parent.counts().total().unwrap().total_providers().unwrap();
    assert!(new_work > 0);
    assert_eq!(
        report
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap()
            - archived.total_providers().unwrap(),
        new_work
    );
    assert_eq!(
        workspace.last_attempt().unwrap().kind(),
        AttemptKind::Resume
    );
    assert_eq!(checkpoint.metadata().counts().total().unwrap(), archived);
}

#[test]
fn continuation_resume_preserves_exhausted_local_allowances() {
    let problem = ContinuationProblem::new(&Fold).unwrap();
    let options = ContinuationOptions::new(0.1, 1e-6, 0.2, Budget::try_from(32).unwrap())
        .unwrap()
        .with_budgets(
            SolveBudgets::unbounded().with_limit(scioxide_core::BudgetKind::Proposal, Budget::ONE),
        );
    let plan = Continuation::new(ContinuationPath::PseudoArclength(1.0), options).unwrap();
    let mut state = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    for _ in 0..2 {
        plan.advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    }
    assert_eq!(
        plan.advance(
            &problem,
            &mut state,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native()
        )
        .unwrap()
        .termination(),
        scioxide_rootfind::RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal)
    );
    let checkpoint = state.checkpoint().unwrap();
    let before = state.point().to_vec();
    let archived = checkpoint
        .metadata()
        .counts()
        .total()
        .unwrap()
        .total_providers()
        .unwrap();
    let mut restored = plan.state(&problem, &[1.0, 1.0]).unwrap();
    let mut workspace = plan.workspace(&problem).unwrap();
    let report = plan
        .resume(
            &problem,
            &checkpoint,
            &mut restored,
            |_, _| Ok(()),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        scioxide_rootfind::RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal)
    );
    assert_eq!(restored.point(), before);
    assert_eq!(
        report
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap(),
        archived
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}
