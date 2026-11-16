//! Multiobjective search with NSGA-II and stochastic first-order methods.
mod pareto;
mod sampling;
pub use pareto::{ParetoArchive, ParetoDominance, ParetoEntry, pareto_dominance};
pub use sampling::{
    DeterministicGradient, NoiseModel, OptimizationSampleKey, OptimizationSampler, PhiloxSampler,
    ReplicationOptions, SampledGradient, SampledObjective,
};

use crate::OptimizeTermination;
use crate::validation::invalid;
use crate::validation::number;
use scioxide_core::{ExecutionStopReason, NumericScalar, RealScalar, SciOxideError, SolveContext};
fn finite<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    crate::validation::scalar(value, scioxide_core::NonFiniteOrigin::Intermediate)
}
fn stop(context: &SolveContext<'_, '_>) -> Result<Option<OptimizeTermination>, SciOxideError> {
    Ok(context.stop_reason()?.map(|s| match s {
        ExecutionStopReason::DeadlineReached => OptimizeTermination::TimeLimit,
        _ => OptimizeTermination::Cancelled,
    }))
}
fn finite_vector<S: NumericScalar>(values: &[S]) -> Result<(), SciOxideError> {
    crate::validation::finite_numeric(values, scioxide_core::NonFiniteOrigin::Callback)
}

mod stochastic;
pub use stochastic::{
    SgdOptions, SgdStep, StochasticCheckpoint, StochasticGradientDescent, StochasticProblem,
    StochasticProgress, StochasticResult, StochasticTrustOptions, StochasticTrustRegion,
    StochasticWorkspace, StochasticWorkspaceRequirements,
};
// Preserve allocation-free propagation of the shared primary failure.
#[allow(clippy::large_enum_variant)]
pub(super) enum FlowError {
    Stop(OptimizeTermination),
    Failure(SciOxideError),
}
impl From<SciOxideError> for FlowError {
    fn from(e: SciOxideError) -> Self {
        Self::Failure(e)
    }
}
type Flow<T> = Result<T, FlowError>;
fn admitted(stop: Option<OptimizeTermination>) -> Flow<()> {
    match stop {
        Some(s) => Err(FlowError::Stop(s)),
        None => Ok(()),
    }
}
fn initial_stop(termination: OptimizeTermination, context: &SolveContext<'_, '_>) -> SciOxideError {
    use scioxide_core::{BudgetKind, ConvergenceDiagnostics, ConvergenceReason};
    match termination {
        OptimizeTermination::Cancelled => {
            SciOxideError::execution_stopped(ExecutionStopReason::Cancelled)
        }
        OptimizeTermination::TimeLimit => {
            SciOxideError::execution_stopped(ExecutionStopReason::DeadlineReached)
        }
        _ => {
            let kind = match termination {
                OptimizeTermination::SampleLimit => BudgetKind::Sample,
                OptimizeTermination::IterationLimit => BudgetKind::Iteration,
                OptimizeTermination::ProposalLimit => BudgetKind::Proposal,
                _ => BudgetKind::Evaluation,
            };
            SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                ConvergenceDiagnostics::new()
                    .with_counts(context.counts())
                    .with_budget(context.effective_budget_report(kind)),
                "stopped before a complete initial observation",
            )
        }
    }
}
mod nsga;
pub use nsga::{
    Nsga2, Nsga2Checkpoint, Nsga2Options, Nsga2Problem, Nsga2Progress, Nsga2Result, Nsga2Workspace,
    Nsga2WorkspaceRequirements,
};
#[cfg(test)]
mod tests;
