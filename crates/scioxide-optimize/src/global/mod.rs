//! Global optimization over a finite box.
//!
//! The problem is a [`crate::BoundedObjective`] whose box is finite on every
//! variable. Every plan prepares a workspace with `workspace(dimension)` and
//! runs `minimize_into(problem, [x0,] output, workspace, scope)`. Randomness
//! comes only from the seed in the options, and the execution policy in the
//! options selects sequential or parallel evaluation.

use scioxide_core::{
    AttemptKind, ExecutionPolicy, InvalidInputReason, NumericScalar, RealScalar, SciOxideError,
    SolveContext, SolveScope,
};

use crate::attempt::{Accepted, Entry, Shell, Staged};
use crate::{BoundedObjective, MinimizerProgress, ObjectiveFunction};

mod chains;
mod cma_es;
mod design;
mod differential_evolution;
mod direct;
mod local;
mod multistart;
mod shgo;
mod surrogate;

pub use chains::{
    BasinHopping, BasinHoppingOptions, BasinProposal, DualAnnealing, DualAnnealingOptions,
    UniformBasinProposal,
};
pub use cma_es::{
    CmaEs, CmaEsGeneration, CmaEsOptions, CmaEsRestartPolicy, CmaEsWorkspace,
    CmaEsWorkspaceRequirements,
};
pub use design::{BorrowedStartDesign, StartDesign, UniformStartDesign};
pub use differential_evolution::{
    DeStrategy, DifferentialEvolution, DifferentialEvolutionBoundary,
    DifferentialEvolutionGeneration, DifferentialEvolutionOptions, DifferentialEvolutionWorkspace,
    DifferentialEvolutionWorkspaceRequirements, UpdatingMode,
};
pub use direct::{
    Direct, DirectOptions, DirectResult, DirectVariant, DirectWorkspace,
    DirectWorkspaceRequirements,
};
pub use local::{LocalSolverFactory, PowellLocalFactory, PowellSearchLocal, SearchLocalSolver};
pub use multistart::{
    GlobalSearchOptions, GlobalSearchResult, GlobalSearchWorkspace,
    GlobalSearchWorkspaceRequirements, MultiStart, SearchStartReport,
};
pub use shgo::{
    Shgo, ShgoOptions, ShgoRefinement, ShgoResult, ShgoWorkspace, ShgoWorkspaceRequirements,
};
pub use surrogate::{
    GaussianAcquisition, KrigingSurrogateFitter, RbfSurrogateFitter, SurrogateAcquisition,
    SurrogateFeasibility, SurrogateFitter, SurrogateHistory, SurrogateModel, SurrogateObservation,
    SurrogatePrediction, SurrogateSearch, SurrogateSearchOptions, SurrogateSearchReport,
    SurrogateSearchWorkspace,
};

/// Thread-safe objective required by a non-sequential execution policy.
///
/// The sequential policy needs none; any other policy evaluates on executor
/// workers and requires a problem built with [`BoundedObjective::new_sync`].
pub(super) fn shared_objective<'a, S: RealScalar + NumericScalar<Real = S>>(
    problem: &BoundedObjective<'a, S>,
    execution: ExecutionPolicy,
) -> Result<Option<&'a (dyn ObjectiveFunction<S> + Sync)>, SciOxideError> {
    if execution == ExecutionPolicy::Sequential {
        return Ok(None);
    }
    problem.sync_objective().map(Some).ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("problem"),
            InvalidInputReason::UnsupportedConfiguration,
            "a non-sequential execution policy requires BoundedObjective::new_sync",
        )
    })
}

/// Reject a non-sequential policy for an inherently sequential method.
pub(super) fn sequential_only(
    execution: ExecutionPolicy,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    if execution == ExecutionPolicy::Sequential {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("options"),
            InvalidInputReason::UnsupportedConfiguration,
            detail,
        ))
    }
}

/// Run one global attempt, copy its staged result to the output and record the attempt.
///
/// Unlike local minimizers, global results keep their accepted state in the
/// staged result itself, so the attempt progress is read from it.
pub(super) fn attempt<'scope, 'capability, S, St, R>(
    shell: &mut Shell<MinimizerProgress<S>>,
    storage: &mut St,
    staged: &mut R,
    output: &mut R,
    scope: SolveScope<'scope, 'capability>,
    entry: Entry,
    kernel: impl FnOnce(
        &mut SolveContext<'_, 'capability>,
        &mut St,
        &mut R,
    ) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    R: Staged + Accepted<S>,
{
    staged.reset();
    let (outcome, started) =
        shell.execute(scope, entry, |context| kernel(context, storage, staged));
    let outcome = outcome.and_then(|()| output.publish(staged));
    let succeeded = outcome.is_ok() && staged.succeeded();
    let progress = if started {
        staged
            .accepted()
            .map(|state| crate::attempt::progress(state, succeeded))
    } else {
        None
    };
    shell.finish(
        AttemptKind::Call,
        started,
        outcome.is_err(),
        succeeded,
        progress,
    );
    outcome
}
