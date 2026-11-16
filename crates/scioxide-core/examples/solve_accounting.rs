//! Typed admission and disjoint nested work without bookkeeping allocation.
use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, Invocation, ProblemToken, SciOxideError,
    SolveBudgets, SolveIdentity, SolveState,
};

const ALGORITHM: scioxide_core::AlgorithmId =
    scioxide_core::algorithm_id!("example", "accounting", 1);

fn main() -> Result<(), SciOxideError> {
    let identity = SolveIdentity::new(ProblemToken::new(1), ALGORITHM, 0, 0);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Iteration, Budget::ONE)
        .with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    scioxide_core::SolveScope::from(&mut context).run(
        &mut SolveState::new(),
        identity,
        SolveBudgets::unbounded(),
        |child| {
            child.record_iteration()?.require()?;
            child.record_iteration()?.require()?;
            assert!(matches!(
                child.invoke_provider(EvaluationRole::Function, || Ok(7))?,
                Invocation::Evaluated(7)
            ));
            Ok(())
        },
    )?;
    // The child consumed the shared evaluation allowance, but no outer iteration.
    context.record_iteration()?.require()?;
    assert!(matches!(
        context.invoke_provider(EvaluationRole::Function, || Ok(8))?,
        Invocation::BudgetExhausted(_)
    ));
    assert_eq!(context.counts().own().iterations(), 1);
    assert_eq!(context.counts().nested().iterations(), 2);
    assert_eq!(
        context.counts().total()?.provider(EvaluationRole::Function),
        1
    );
    Ok(())
}
