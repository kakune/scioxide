//! Iteration accounting that turns an exhausted iteration limit into a convergence failure.
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, ExecutionContext, SciOxideError, SolveContext,
};

pub(crate) fn admit_iteration(
    execution: Option<ExecutionContext<'_>>,
    solve: Option<&mut SolveContext<'_, '_>>,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    if let Some(execution) = execution {
        execution.check()?;
    }
    if let Some(solve) = solve
        && !solve.record_iteration()?.is_granted()
    {
        return Err(SciOxideError::convergence_failure(
            ConvergenceReason::IterationLimit,
            ConvergenceDiagnostics::new().with_counts(solve.counts()),
            detail,
        ));
    }
    Ok(())
}
