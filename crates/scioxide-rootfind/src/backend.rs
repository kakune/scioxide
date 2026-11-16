//! Lower-level building blocks used by other SciOxide crates.
//!
//! Most users do not need this module. Its functions run inside a solve that
//! another solver manages, so they do not keep the output unchanged on error
//! and do not record attempts in the workspace.
use crate::{NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver, NewtonKrylovWorkspace};
use scioxide_core::backend::{PartitionedSolve, with_partitioned_solves};
use scioxide_core::{
    BudgetKind, InvalidInputReason, NumericScalar, ProblemToken, SciOxideError, SolveContext,
    SolveIdentity,
};

/// Runs one Newton–Krylov solve as a partitioned child of `parent`.
///
/// The solve counts its callback evaluations locally and adds its work,
/// including failed work, to `parent` when it finishes. It shares the parent's
/// evaluation budgets, cancellation and deadline, and keeps its own iteration
/// limit and work limits from the solver's options. Counting locally avoids
/// walking the chain of enclosing solves on every callback.
///
/// `output` may change on error, and the workspace's attempt record is
/// cleared. With a workspace and result sized for the problem, the call does
/// not allocate.
///
/// # Errors
///
/// - `InvalidInput` if `initial` or `output` does not match the problem
///   dimension, or `work` is not reserved for the problem.
/// - `NonFinite` if `initial` contains NaN or infinity.
/// - Errors from the residual callback, budget errors and execution stops of
///   the parent, as for [`NewtonKrylovSolver::solve_into`].
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::PartitionedSolve;
/// use scioxide_core::{
///     ExecutionContext, SciOxideError, ProblemToken, SolveBudgets, SolveIdentity, SolveScope,
///     SolveState,
/// };
/// use scioxide_rootfind::{
///     NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver,
///     ResidualFn, backend::solve_newton_partitioned,
/// };
///
/// // x² − 2 = 0 from x₀ = 1 converges to √2.
/// let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] - 2.0;
///     Ok(())
/// });
/// let problem = NewtonKrylovProblem::numerical(&f);
/// let solver = NewtonKrylovSolver::new(NewtonKrylovOptions::default())?;
/// let mut work = solver.workspace(&problem)?;
/// let mut result = NewtonKrylovResult::new(1, solver.options())?;
/// let mut slot = PartitionedSolve::default();
/// let id = scioxide_core::algorithm_id!("example", "nested-newton", 1);
/// SolveScope::from(ExecutionContext::sequential()).run(
///     &mut SolveState::new(),
///     SolveIdentity::new(ProblemToken::new(0), id, 1, 0),
///     SolveBudgets::unbounded(),
///     |cx| solve_newton_partitioned(&solver, &problem, &[1.0], &mut result, &mut work, &mut slot, cx),
/// )?;
/// assert!(result.is_success());
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn solve_newton_partitioned<T: NumericScalar>(
    solver: &NewtonKrylovSolver<T>,
    problem: &NewtonKrylovProblem<'_, T>,
    initial: &[T],
    output: &mut NewtonKrylovResult<T>,
    work: &mut NewtonKrylovWorkspace<T>,
    slot: &mut PartitionedSolve,
    parent: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    work.shell.clear();
    solver.prepare(problem, initial.into(), output, work)?;
    with_partitioned_solves(
        parent,
        core::slice::from_mut(slot),
        crate::context::cap(
            solver.options.budgets,
            BudgetKind::Iteration,
            solver.options.iterations,
        ),
        |execution, batch| {
            let permit = batch.into_iter().next().ok_or_else(|| {
                crate::invalid(
                    scioxide_core::argument!("workspace"),
                    InvalidInputReason::Inconsistent,
                    "missing prepared Newton permit",
                )
            })?;
            permit.run(
                execution,
                SolveIdentity::new(
                    ProblemToken::new(0),
                    scioxide_core::algorithm_id!("rootfind", "Newton-Krylov", 1),
                    problem.dimension() as u128,
                    0,
                ),
                |cx| solver.run(problem, &solver.options, output, work, cx),
            )
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{NewtonKrylovOptions, ResidualFn};
    use core::cell::Cell;
    use scioxide_core::{Budget, ExecutionContext, SolveBudgets, SolveState};
    #[test]
    fn delegated_work_matches_live_parent_admission_and_local_iteration_limits() {
        for limit in [1, 2, 8, 50] {
            for local_iteration_limit in [false, true] {
                let mut expected = None;
                for delegated in [false, true] {
                    let calls = Cell::new(0);
                    let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
                        calls.set(calls.get() + 1);
                        out[0] = x[0] * x[0] - 2.;
                        Ok(())
                    });
                    let problem = NewtonKrylovProblem::numerical(&f);
                    let options = if local_iteration_limit {
                        NewtonKrylovOptions::default().with_budgets(
                            SolveBudgets::unbounded()
                                .with_limit(BudgetKind::Iteration, Budget::ONE),
                        )
                    } else {
                        NewtonKrylovOptions::default()
                    };
                    let solver = NewtonKrylovSolver::new(options).unwrap();
                    let mut w = solver.workspace(&problem).unwrap();
                    let mut out = NewtonKrylovResult::new(1, solver.options()).unwrap();
                    let mut state = SolveState::new();
                    let mut allowance = SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
                    let mut cx = state
                        .begin_solve(
                            ExecutionContext::sequential(),
                            SolveIdentity::new(
                                ProblemToken::new(2368),
                                scioxide_core::algorithm_id!("test", "partitioned-newton", 1),
                                1,
                                0,
                            ),
                            &mut allowance,
                        )
                        .unwrap();
                    let result = if delegated {
                        solve_newton_partitioned(
                            &solver,
                            &problem,
                            &[1.],
                            &mut out,
                            &mut w,
                            &mut PartitionedSolve::default(),
                            &mut cx,
                        )
                    } else {
                        solver.solve_into(&problem, &[1.], &mut out, &mut w, &mut cx)
                    };
                    let observed = (
                        result.as_ref().err().map(|e| e.kind()),
                        if result.is_ok() {
                            Some(out.termination())
                        } else {
                            None
                        },
                        out.point().to_vec(),
                        calls.get(),
                        cx.counts().total().unwrap(),
                    );
                    if let Some(reference) = &expected {
                        assert_eq!(
                            &observed, reference,
                            "limit={limit} local_iteration_limit={local_iteration_limit}"
                        );
                    } else {
                        expected = Some(observed);
                    }
                }
            }
        }
    }
}
