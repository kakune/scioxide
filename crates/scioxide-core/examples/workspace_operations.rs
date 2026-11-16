//! Caller-owned storage, execution contexts and prepared operations.
//! Run with `cargo run -p scioxide-core --example workspace_operations`.
use scioxide_core::SciOxideError;

#[cfg(feature = "std")]
fn execution_std_threads_thread_executor() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionContext, ExecutionPolicy, PlanLimits, ThreadExecutor};

    let workers = NonZeroUsize::MIN.saturating_add(1);
    let executor = ThreadExecutor::new(workers);
    let cx = ExecutionContext::new(&executor);
    let limits = PlanLimits {
        work_per_item: NonZeroUsize::MIN,
        auto_threshold: NonZeroUsize::MIN,
        worker_limit: workers,
    };
    let plan = cx.plan(ExecutionPolicy::Parallel(workers), 8, limits);
    cx.execute(plan, &|partition, (), child| {
        assert!(!partition.range().is_empty());
        assert_eq!(child.worker_budget(), NonZeroUsize::MIN);
        Ok(())
    })?;
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn execution_execute_partitioned() -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionContext, ExecutionPolicy, PlanLimits, backend::PartitionSlot};

    let cx = ExecutionContext::sequential();
    let limits = PlanLimits {
        work_per_item: NonZeroUsize::MIN,
        auto_threshold: NonZeroUsize::MIN,
        worker_limit: NonZeroUsize::MIN,
    };
    let plan = cx.plan(ExecutionPolicy::Sequential, 2, limits);
    let mut output = [0; 2];
    let slots = [PartitionSlot::new(&mut output[..])];
    cx.execute_partitioned(plan, &slots, &|_, values, _| {
        values.fill(1);
        Ok(())
    })?;
    assert_eq!(output, [1, 1]);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn operator_closure_operator() -> Result<(), SciOxideError> {
    use scioxide_core::{AdjointOperator, ClosureOperator, LinearOperator};

    let a = ClosureOperator::<f64, _>::new(2, 2, |x, y| {
        y.copy_from_slice(&[x[0] + 2.0 * x[1], x[1]]);
        Ok(())
    })
    .with_adjoint(|x, y| {
        y.copy_from_slice(&[x[0], 2.0 * x[0] + x[1]]);
        Ok(())
    });
    let (mut y, mut workspace) = ([0.0; 2], [0.0; 2]);
    a.apply_into(&[1.0, 1.0], &mut y, &mut workspace)?;
    assert!((y[0] - 3.0).abs() < 1.0e-15 && (y[1] - 1.0).abs() < 1.0e-15);
    a.apply_adjoint_into(&[1.0, 1.0], &mut y, &mut workspace)?;
    assert!((y[0] - 1.0).abs() < 1.0e-15 && (y[1] - 3.0).abs() < 1.0e-15);
    Ok::<(), scioxide_core::SciOxideError>(())
}

fn solve_with_partitioned_solves() -> Result<(), SciOxideError> {
    use scioxide_core::backend::{PartitionedSolve, with_partitioned_solves};
    use scioxide_core::{
        EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };

    let algorithm = scioxide_core::algorithm_id!("example", "batch", 1);
    let id = SolveIdentity::new(ProblemToken::new(1), algorithm, 0, 0);
    let mut state = SolveState::new();
    let mut limits = SolveBudgets::unbounded();
    let mut parent = state.begin_solve(ExecutionContext::sequential(), id, &mut limits)?;
    let mut children = [PartitionedSolve::default(), PartitionedSolve::default()];
    let unbounded = SolveBudgets::unbounded();
    with_partitioned_solves(&mut parent, &mut children, unbounded, |execution, batch| {
        for permit in batch {
            permit.run(execution, id, |child| {
                child.record_provider(EvaluationRole::Function)?.require()?;
                Ok(())
            })?;
        }
        Ok(())
    })?;
    // Each of the two children counted one invocation.
    assert_eq!(parent.counts().total()?.total_providers()?, 2);
    Ok::<(), scioxide_core::SciOxideError>(())
}

pub fn main() -> Result<(), SciOxideError> {
    #[cfg(feature = "std")]
    execution_std_threads_thread_executor()?;
    execution_execute_partitioned()?;
    operator_closure_operator()?;
    solve_with_partitioned_solves()?;
    Ok(())
}
