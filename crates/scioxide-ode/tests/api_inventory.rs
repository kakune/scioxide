//! The authoritative scioxide-ode root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test fails
//! when this list and the `pub use` items of `src/lib.rs` differ. Plans and options are
//! asserted `Send + Sync + Clone`, every time-evolution termination enum carries the
//! shared vocabulary, and the requirements and execution forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_ode::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    solve_ivp, solve_ivp_at, solve_sde, solve_dde, solve_hamiltonian, solve_bvp, solve_ensemble_into,
    EventFn, DaeFn, SdeFn, BvpFn, HamiltonianFn,
    // Systems, closures, events and method selection.
    OdeSystem, OdeRhsStepLimit, OdeFn, JacobianCallback, NoJacobian, EventFunction, EventDirection,
    EventEvidence, EventOccurrence, EventSearchEvidence, EventSearchPolicy, OdeEventOptions,
    OdeMethod, OdeLinearStability, BuiltinOdeMethod,
    // Initial-value plans, workspaces, output and reports.
    OdeOptions, OdeStepControl, OdeOutputSchedule, OdeProblem, OdeState,
    OdeRequirements, OdeWorkspace, OdeOutputSink, OdeEvent, EventInfo, OdeTrajectory,
    OdeSolveReport, OdeProgress, OdeSolution, OdeTermination, ProportionalIntegralController,
    // Built-in methods and their configuration.
    ForwardEuler, ExplicitMidpoint, Heun, ClassicalRk4, Rkf45, DormandPrince5, DormandPrince853,
    Verner65, BackwardEuler, ImplicitTrapezoid, GaussRk, Sdirk, RadauIia, RosenbrockMethod,
    RosenbrockW, AdamsBashforth, AdamsMoulton, Bdf, BdfFormula,
    MultistepOrder, AdamsBdf, AutoStiff, OdeMethodSwitch, OdeSwitchReason,
    StiffnessSwitchOptions,
    // Stiff stages and injected stage solvers.
    NewtonMode, OdeStageBackend, OdeStageSolver, StageAccuracy, StageLinearSolver, NonlinearStageSolver,
    NonlinearStageReport,
    // Delay differential equations.
    DdeHistory, DdeHistoryBreak, DdeRhs, DdeOptions, DdePlan, DdeProblem, DdeRequirements,
    DdeWorkspace, DdeWorkspaceRequest, DdeReport, DdeProgress, DdeSolution, DdeTermination,
    // Index-1 DAEs.
    DaeSystem, DaePartition, DaeInitialization, DaeOptions, DaeNumericalOptions, DaeBdf, DaeProblem, DaeState,
    DaeOutputSink, DaeRequirements, DaeWorkspace, DaeReport, DaeConsistencyReport, DaeSolution,
    DaeTermination,
    // SDEs.
    SdeSystem, SdeInterpretation, SdeMethod, SdeOptions, SdeProblem, SdeRequirements,
    SdeWorkspace, SdeReport, SdeSolution, SdeTermination,
    // Geometric mechanics.
    SeparableHamiltonian, HolonomicConstraints, PhaseSpace, MechanicalOptions, RattleOptions,
    StormerVerlet, Rattle, RattleProblem, StormerVerletRequirements, StormerVerletWorkspace,
    RattleRequirements, RattleWorkspace, MechanicalOutputSink, MechanicalReport,
    MechanicalSolution, MechanicalTermination,
    // Boundary-value problems.
    BvpSystem, BvpJacobian, BvpBackend, BvpNewtonOptions, BvpOptions, BvpPlan, BvpProblem, BvpRequirements,
    BvpWorkspace, BvpSolution, BvpReport, BvpTermination,
    // Adjoints.
    AdjointMode, AdjointOptions, CheckpointPolicy, OdeAdjointOptions, DaeAdjointOptions, AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue,
    AdjointTerminalEvent, OdeAdjointSystem, DaeAdjointSystem, OdeAdjointResidual,
    OdeAdjointSolver, DaeAdjointSolver, OdeAdjointProblem, DaeAdjointProblem, OdeAdjointRequirements, DaeAdjointRequirements,
    OdeAdjointWorkspace, DaeAdjointWorkspace, AdjointReport, AdjointErrorReport, AdjointResult,
    AdjointTermination,
    // Forward sensitivities and ensembles.
    OdeSensitivitySystem, ForwardSensitivityRequirements,
    ForwardSensitivityWorkspace, ForwardSensitivitySystem, EventSensitivityStatus,
    EventSensitivityInput, EventIsolation, EnsembleParameters,
    EnsembleRequirements, EnsembleWorkspace, EnsembleReport, EnsembleOutcome,
}

/// Names re-exported by the `pub use` statements of a crate root. Each statement
/// re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("\npub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn inventory_lists_every_root_export_once() {
    let exported = root_exports(include_str!("../src/lib.rs"));
    let listed: BTreeSet<&str> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    let unlisted: Vec<_> = exported.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&exported).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "unlisted {unlisted:?}, stale {stale:?}"
    );
}

#[test]
fn plans_and_options_are_send_sync_clone() {
    fn defaults<T: Default>() {}
    defaults::<OdeOptions<f64>>();
    defaults::<OdeEventOptions<f64>>();
    defaults::<StiffnessSwitchOptions<f64>>();
    defaults::<DaeOptions<f64>>();
    defaults::<DaeNumericalOptions<f64>>();
    defaults::<DdeOptions<f64>>();
    defaults::<SdeOptions<f64>>();
    defaults::<MechanicalOptions>();
    defaults::<RattleOptions<f64>>();
    defaults::<BvpOptions<f64>>();
    defaults::<BvpNewtonOptions<f64>>();

    fn shareable<T: Send + Sync + Clone>() {}
    shareable::<DormandPrince5<f64>>();
    shareable::<RadauIia<f64>>();
    shareable::<Bdf<f64>>();
    shareable::<RosenbrockW<f64>>();
    shareable::<AdamsBdf<f64>>();
    shareable::<BuiltinOdeMethod<f64>>();
    shareable::<BuiltinOdeMethod<f64, scioxide_core::Complex<f64>>>();
    shareable::<DdePlan<f64>>();
    shareable::<DaeBdf>();
    shareable::<(SdeMethod, Vec<f64>, Vec<usize>, SdeOptions<f64>)>();
    shareable::<StormerVerlet>();
    shareable::<Rattle>();
    shareable::<BvpPlan<f64>>();
    shareable::<OdeAdjointSolver<f64>>();
    shareable::<DaeAdjointSolver<f64>>();
    shareable::<OdeOptions<f64>>();
    shareable::<OdeEventOptions<f64>>();
    shareable::<DdeOptions<f64>>();
    shareable::<DdeWorkspaceRequest>();
    shareable::<DaeOptions<f64>>();
    shareable::<SdeOptions<f64>>();
    shareable::<MechanicalOptions>();
    shareable::<RattleOptions<f64>>();
    shareable::<BvpOptions<f64>>();
    shareable::<BvpNewtonOptions<f64>>();
    shareable::<AdjointOptions<f64>>();
    shareable::<OdeAdjointOptions<f64>>();
    shareable::<DaeAdjointOptions<f64>>();
    shareable::<DaeNumericalOptions<f64>>();
    shareable::<StiffnessSwitchOptions<f64>>();
}

/// Every time-evolution family shares these termination names (issue #2302).
macro_rules! shared_termination_names {
    ($($termination:ty),* $(,)?) => {$(
        const _: fn() -> [$termination; 5] = || {
            use scioxide_core::{BudgetKind, ExecutionStopReason};
            [
                <$termination>::ReachedEnd,
                <$termination>::TerminalEvent,
                <$termination>::BudgetLimit(BudgetKind::Proposal),
                <$termination>::StepSizeLimit,
                <$termination>::ExecutionStopped(ExecutionStopReason::Cancelled),
            ]
        };
    )*};
}
shared_termination_names!(
    OdeTermination,
    DdeTermination,
    DaeTermination,
    SdeTermination,
    MechanicalTermination,
    AdjointTermination,
);

// One requirements protocol and one progressive form per family.
const _: fn() = || {
    use scioxide_core::{AttemptRecord, Capacity, SciOxideError};
    type Method = DormandPrince5<f64>;
    let _: fn(
        &OdeWorkspace<f64, Method>,
        &OdeRequirements<f64, Method>,
    ) -> Result<Capacity, SciOxideError> = OdeWorkspace::check;
    let _: fn(
        &mut OdeWorkspace<f64, Method>,
        &OdeRequirements<f64, Method>,
    ) -> Result<(), SciOxideError> = OdeWorkspace::try_reserve;
    let _: fn(&OdeWorkspace<f64, Method>) -> Option<&AttemptRecord<OdeProgress<f64>>> =
        OdeWorkspace::last_attempt;
    let _: fn(&DdePlan<f64>, DdeWorkspaceRequest) -> Result<DdeRequirements, SciOxideError> =
        DdePlan::workspace_requirements;
    let _: fn(&DdePlan<f64>, DdeWorkspaceRequest) -> Result<DdeWorkspace<f64>, SciOxideError> =
        DdePlan::workspace;
    let _: fn(&mut DdeWorkspace<f64>, &DdeRequirements) -> Result<(), SciOxideError> =
        DdeWorkspace::try_reserve;
    let _: fn(&DdeWorkspace<f64>, &DdeRequirements) -> Result<Capacity, SciOxideError> =
        DdeWorkspace::check;
    let _: fn(&DdeWorkspace<f64>) -> Option<&AttemptRecord<DdeProgress<f64>>> =
        DdeWorkspace::last_attempt;
    let _: fn(&DaeWorkspace<f64>) -> Option<&AttemptRecord<DaeReport<f64>>> =
        DaeWorkspace::last_attempt;
    let _: fn(&SdeWorkspace<f64>) -> Option<&AttemptRecord<SdeReport<f64>>> =
        SdeWorkspace::last_attempt;
    let _: fn(&StormerVerletWorkspace<f64>) -> Option<&AttemptRecord<MechanicalReport<f64>>> =
        StormerVerletWorkspace::last_attempt;
    let _: fn(&OdeAdjointWorkspace<f64>) -> Option<&AttemptRecord<AdjointReport<f64>>> =
        OdeAdjointWorkspace::last_attempt;
    let _: fn(&BvpWorkspace<f64>) -> Option<&AttemptRecord<BvpReport<f64>>> =
        BvpWorkspace::last_attempt;
    let _: fn(&EnsembleWorkspace<f64>) -> Option<&AttemptRecord<EnsembleReport>> =
        EnsembleWorkspace::last_attempt;
    let _: fn(OdeMethodSwitch<f64>) -> &'static str = |switch| switch.to();
};

// Every plan exposes its validated options.
const _: fn() = || {
    let _: fn(&DdePlan<f64>) -> &DdeOptions<f64> = DdePlan::options;
    let _: fn(&BvpPlan<f64>) -> &BvpOptions<f64> = BvpPlan::options;
    let _: fn(&OdeAdjointSolver<f64>) -> &OdeAdjointOptions<f64> = OdeAdjointSolver::options;
    let _: fn(&DaeAdjointSolver<f64>) -> &DaeAdjointOptions<f64> = DaeAdjointSolver::options;
};

// Requirements report their workspace bytes under the workspace-wide name.
const _: fn() = || {
    let _: fn(&DdeRequirements) -> usize = DdeRequirements::required_bytes;
    let _: fn(&SdeRequirements<f64>) -> usize = SdeRequirements::required_bytes;
    let _: fn(&BvpRequirements<f64>) -> usize = BvpRequirements::required_bytes;
    let _: fn(&OdeAdjointRequirements<f64>) -> usize = OdeAdjointRequirements::required_bytes;
    let _: fn(&DaeAdjointRequirements<f64>) -> usize = DaeAdjointRequirements::required_bytes;
};

// Newton options report every setting their setters accept.
const _: fn() = || {
    use scioxide_core::Budget;
    use scioxide_rootfind::NewtonKrylovForcing;
    let _: fn(&BvpNewtonOptions<f64>) -> Budget = BvpNewtonOptions::max_iterations;
    let _: fn(&BvpNewtonOptions<f64>) -> (f64, f64) = BvpNewtonOptions::backtracking;
    let _: fn(&BvpNewtonOptions<f64>) -> f64 = BvpNewtonOptions::nonmonotone;
    let _: fn(&BvpNewtonOptions<f64>) -> NewtonKrylovForcing<f64> = BvpNewtonOptions::forcing;
};

// Every evolution family exposes the same accessor names and value types.
macro_rules! shared_report_accessors {
    ($(($report:ident, $termination:ident)),* $(,)?) => {$(
        const _: fn() = || {
            use scioxide_core::{EvaluationCounts, PartialResultTrust, SolveCounts};
            let _: fn(&$report<f64>) -> Option<$termination> = $report::termination;
            let _: fn(&$report<f64>) -> bool = $report::is_success;
            let _: fn(&$report<f64>) -> f64 = $report::final_time;
            let _: fn(&$report<f64>) -> usize = $report::accepted_steps;
            let _: fn(&$report<f64>) -> usize = $report::rejected_steps;
            let _: fn(&$report<f64>) -> EvaluationCounts = $report::counts;
            let _: fn(&$report<f64>) -> SolveCounts = $report::accounting;
            let _: fn(&$report<f64>) -> PartialResultTrust = $report::trust;
        };
    )*};
}
shared_report_accessors!(
    (OdeSolveReport, OdeTermination),
    (DdeReport, DdeTermination),
    (DaeReport, DaeTermination),
    (SdeReport, SdeTermination),
    (MechanicalReport, MechanicalTermination),
    (AdjointReport, AdjointTermination),
);

const _: fn() = || {
    use scioxide_core::{SciOxideError, SolveCounts};
    let _: fn(usize, usize, &[f64], BvpOptions<f64>) -> Result<BvpPlan<f64>, SciOxideError> =
        BvpPlan::new;
    let _: fn(
        usize,
        usize,
        OdeAdjointOptions<f64>,
    ) -> Result<OdeAdjointSolver<f64>, SciOxideError> = OdeAdjointSolver::new;
    let _: fn(
        DaePartition,
        usize,
        OdeStageBackend,
        DaeAdjointOptions<f64>,
    ) -> Result<DaeAdjointSolver<f64>, SciOxideError> = DaeAdjointSolver::new;
    let _: fn(
        usize,
        usize,
        OdeStageBackend,
        DaeAdjointOptions<f64>,
    ) -> Result<DaeAdjointSolver<f64>, SciOxideError> = DaeAdjointSolver::backward_euler;
    let _: fn(
        usize,
        usize,
        OdeStageBackend,
        DaeAdjointOptions<f64>,
    ) -> Result<DaeAdjointSolver<f64>, SciOxideError> = DaeAdjointSolver::trapezoid;
    // Options store their documented defaults, and getters return effective values.
    let _: fn(
        AdjointOptions<f64>,
        DaeNumericalOptions<f64>,
    ) -> Result<DaeAdjointOptions<f64>, SciOxideError> = DaeAdjointOptions::new;
    let _: fn(&DaeAdjointOptions<f64>) -> scioxide_core::RelativeTolerance<f64> =
        DaeAdjointOptions::linear_residual;
    let _: fn(
        &BvpOptions<f64>,
    ) -> (
        scioxide_core::ResidualTolerance<f64>,
        scioxide_core::ResidualTolerance<f64>,
    ) = BvpOptions::tolerances;
    // Sparse factor limits belong to the sparse backend; `None` derives them per mesh.
    let _: fn(Option<scioxide_sparse::SparseLuLimits>) -> BvpBackend =
        |limits| BvpBackend::Sparse { limits };
    let _: fn(&BvpReport<f64>) -> SolveCounts = BvpReport::accounting;
    let _: fn(&BvpReport<f64>) -> bool = <BvpReport<f64> as EnsembleOutcome>::is_success;
    let _: fn(&EnsembleReport) -> bool = EnsembleReport::is_success;
    let _: fn(&OdeSolveReport<f64>) -> Option<f64> = OdeSolveReport::last_step_error_estimate;
    let _: fn(&DdeReport<f64>) -> Option<f64> = DdeReport::last_step_error_estimate;
    let _: fn(&DaeReport<f64>) -> Option<f64> = DaeReport::last_step_error_estimate;
    let _: fn(&SdeReport<f64>) -> Option<f64> = SdeReport::last_step_error_estimate;
    let _: fn(&AdjointReport<f64>) -> Option<f64> = AdjointReport::last_step_error_estimate;
};

// These generic bodies pin execution independently of any callback implementation.
#[allow(dead_code, reason = "Compile-time public signature assertions")]
fn adjoint_execution_signatures<P, O>()
where
    P: OdeAdjointSystem<f64> + DaeAdjointSystem<f64>,
    O: AdjointObjective<f64>,
{
    use scioxide_core::{PartialFailure, SciOxideError, SolveScope};
    type Report = AdjointReport<f64>;
    let _: fn(
        &OdeAdjointSolver<f64>,
        &OdeAdjointProblem<'_, f64, P, O>,
    ) -> Result<OdeAdjointRequirements<f64>, SciOxideError> =
        OdeAdjointSolver::workspace_requirements;
    let _: fn(
        &DaeAdjointSolver<f64>,
        &DaeAdjointProblem<'_, f64, P, O>,
    ) -> Result<DaeAdjointRequirements<f64>, SciOxideError> =
        DaeAdjointSolver::workspace_requirements;
    let _: fn(
        &OdeAdjointSolver<f64>,
        &OdeAdjointProblem<'_, f64, P, O>,
    ) -> Result<OdeAdjointWorkspace<f64>, SciOxideError> = OdeAdjointSolver::workspace;
    let _: fn(
        &DaeAdjointSolver<f64>,
        &DaeAdjointProblem<'_, f64, P, O>,
    ) -> Result<DaeAdjointWorkspace<f64>, SciOxideError> = DaeAdjointSolver::workspace;
    let _: fn(
        &OdeAdjointSolver<f64>,
        &OdeAdjointProblem<'_, f64, P, O>,
    ) -> Result<AdjointResult<f64>, PartialFailure<Report>> = OdeAdjointSolver::solve;
    let _: fn(
        &DaeAdjointSolver<f64>,
        &DaeAdjointProblem<'_, f64, P, O>,
    ) -> Result<AdjointResult<f64>, PartialFailure<Report>> = DaeAdjointSolver::solve;
    let _: fn(
        &OdeAdjointSolver<f64>,
        &OdeAdjointProblem<'_, f64, P, O>,
        &mut [f64],
        &mut [f64],
        &mut OdeAdjointWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<Report, SciOxideError> = |plan, problem, initial, parameter, workspace, scope| {
        plan.solve_into(problem, initial, parameter, workspace, scope)
    };
    let _: fn(
        &DaeAdjointSolver<f64>,
        &DaeAdjointProblem<'_, f64, P, O>,
        &mut [f64],
        &mut [f64],
        &mut DaeAdjointWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<Report, SciOxideError> = |plan, problem, initial, parameter, workspace, scope| {
        plan.solve_into(problem, initial, parameter, workspace, scope)
    };
}

// One-shot answers and configured execution take only their documented inputs.
const _: fn() = || {
    use scioxide_core::{ExecutionPolicy, PartialFailure, SciOxideError, SolveContext, SolveScope};
    use scioxide_linalg::{Matrix, MatrixView, Vector};
    type Rhs = fn(f64, &[f64], &mut [f64]) -> Result<(), SciOxideError>;
    type Diffusion = fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), SciOxideError>;
    type Boundary = fn(&[f64], &[f64], &mut [f64]) -> Result<(), SciOxideError>;
    type Hamiltonian = fn(&[f64], &mut [f64]) -> Result<(), SciOxideError>;
    type Delay = fn(f64, &[f64], MatrixView<'_, f64>, &mut [f64]) -> Result<(), SciOxideError>;
    type History = fn(f64, &mut [f64]) -> Result<(), SciOxideError>;
    let _: fn(Rhs, f64, f64, &[f64]) -> Result<Vector<f64>, PartialFailure<OdeProgress<f64>>> =
        solve_ivp;
    let _: fn(Rhs, &[f64], &[f64]) -> Result<Matrix<f64>, PartialFailure<OdeProgress<f64>>> =
        solve_ivp_at;
    let _: fn(
        Rhs,
        Diffusion,
        &[f64],
        MatrixView<'_, f64>,
        &[f64],
    ) -> Result<Matrix<f64>, PartialFailure<SdeReport<f64>>> = solve_sde;
    let _: fn(
        Delay,
        History,
        &[f64],
        f64,
        f64,
        usize,
    ) -> Result<Vector<f64>, PartialFailure<DdeProgress<f64>>> = solve_dde;
    let _: fn(
        Hamiltonian,
        Hamiltonian,
        &[f64],
        &[f64],
        f64,
        usize,
    ) -> Result<PhaseSpace<f64>, PartialFailure<MechanicalReport<f64>>> = solve_hamiltonian;
    let _: fn(
        Rhs,
        Boundary,
        &[f64],
        MatrixView<'_, f64>,
    ) -> Result<BvpSolution<f64>, SciOxideError> = solve_bvp;
    let _: fn(&BvpSolution<f64>, f64, &mut [f64], &mut [f64]) -> Result<(), SciOxideError> =
        BvpSolution::evaluate_with_derivative_into;
    type Run = fn(
        usize,
        &[f64],
        &mut f64,
        &mut SolveContext<'_, '_>,
    ) -> Result<OdeSolveReport<f64>, SciOxideError>;
    let _: fn(
        EnsembleParameters<'_, f64>,
        &Run,
        ExecutionPolicy,
        &mut [f64],
        &mut [Option<Result<OdeSolveReport<f64>, SciOxideError>>],
        &mut EnsembleWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<EnsembleReport, SciOxideError> =
        |parameters, run, policy, members, outcomes, workspace, scope| {
            solve_ensemble_into(parameters, run, policy, members, outcomes, workspace, scope)
        };
    let _: fn(
        &DormandPrince5<f64>,
        &OdeProblem<'_, f64, dyn OdeSystem<f64>>,
        &OdeOptions<f64>,
        OdeState<f64>,
    ) -> Result<OdeSolution<f64>, PartialFailure<OdeProgress<f64>>> = OdeMethod::solve;
    let _: fn(
        &DormandPrince5<f64>,
        &OdeProblem<'_, f64, dyn OdeSystem<f64>>,
        &OdeOptions<f64>,
        &mut OdeState<f64>,
        &mut (),
        &mut OdeWorkspace<f64, DormandPrince5<f64>>,
        SolveScope<'_, '_>,
    ) -> Result<OdeSolveReport<f64>, SciOxideError> =
        |method, problem, options, state, sink, workspace, scope| {
            method.advance(problem, options, state, sink, workspace, scope)
        };
    let _: fn(
        SdeMethod,
        &SdeProblem<'_, f64>,
        &SdeOptions<f64>,
        &mut OdeState<f64>,
        &mut (),
        &mut SdeWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<SdeReport<f64>, SciOxideError> =
        |method, problem, options, state, sink, workspace, scope| {
            method.advance(problem, options, state, sink, workspace, scope)
        };
    let _: fn(
        &StormerVerlet,
        &dyn SeparableHamiltonian<f64>,
        f64,
        usize,
        &MechanicalOptions,
        &mut PhaseSpace<f64>,
        &mut (),
        &mut StormerVerletWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<MechanicalReport<f64>, SciOxideError> =
        |method, problem, step, steps, options, state, sink, workspace, scope| {
            method.advance(problem, step, steps, options, state, sink, workspace, scope)
        };
    let _: fn(
        &DaeBdf,
        &DaeProblem<'_, f64, dyn DaeSystem<f64>>,
        &DaeOptions<f64>,
        &mut DaeState<f64>,
        &mut (),
        &mut DaeWorkspace<f64>,
        SolveScope<'_, '_>,
    ) -> Result<DaeReport<f64>, SciOxideError> =
        |method, problem, options, state, sink, workspace, scope| {
            method.advance(problem, options, state, sink, workspace, scope)
        };
};
