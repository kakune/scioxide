//! Reviewed crate-root inventory, compiled as an external Rust consumer.
#![expect(
    unused_imports,
    reason = "The external API inventory verifies crate-root exports by importing them."
)]
use scioxide_optimize::{
    Admm, AdmmBlock, AdmmBlockUpdate, AdmmOptions, AdmmProblem, AdmmProgress, AdmmResult,
    AdmmStart, AdmmUpdateInput, AdmmUpdateReport, AdmmWorkspace, AdmmWorkspaceRequirements,
    AugmentedLagrangian, AugmentedLagrangianOptions, AugmentedLagrangianWorkspace,
    AugmentedLagrangianWorkspaceRequirements, BacktrackingLineSearch, BasinHopping,
    BasinHoppingOptions, BasinProposal, BasisMatrix, BasisProvider, Bfgs, BfgsCheckpoint,
    BfgsOptions, BfgsWorkspace, BfgsWorkspaceRequirements, BorrowedStartDesign,
    BoundedLinearLeastSquares, BoundedLinearLeastSquaresWorkspace, BoundedObjective,
    BranchAndBound, BranchAndBoundWorkspace, CgRestartReason, CmaEs, CmaEsGeneration, CmaEsOptions,
    CmaEsRestartPolicy, CmaEsWorkspace, CmaEsWorkspaceRequirements, Cobyla, CobylaDiagnostics,
    CobylaOptions, CobylaWorkspace, CobylaWorkspaceRequirements, Cobyqa, CobyqaDiagnostics,
    CobyqaOptions, CobyqaWorkspace, CobyqaWorkspaceRequirements, CoefficientMatrix,
    ComplexLeastSquares, CompositeProblem, ConstrainedProblem, ConstraintFn, ConstraintFunction,
    ConstraintJacobianStats, ConstraintProfile, ConstraintValues, Constraints,
    ConstraintsWorkspace, CurvatureAction, CurvaturePolicy, DeStrategy, DenseBasisProvider,
    DenseKktProvider, DenseSqpSubproblem, DerivativeWorkspace, DeterministicGradient,
    DifferentialEvolution, DifferentialEvolutionBoundary, DifferentialEvolutionGeneration,
    DifferentialEvolutionOptions, DifferentialEvolutionWorkspace,
    DifferentialEvolutionWorkspaceRequirements, Direct, DirectOptions, DirectResult, DirectVariant,
    DirectWorkspace, DirectWorkspaceRequirements, DualAnnealing, DualAnnealingOptions,
    DualCertificate, FiniteBounds, Fista, GaussianAcquisition, GlobalSearchOptions,
    GlobalSearchResult, GlobalSearchWorkspace, GlobalSearchWorkspaceRequirements,
    GlobalizationDiagnostics, GradientDescent, GradientDescentOptions, GradientDescentWorkspace,
    GradientDescentWorkspaceRequirements, HessianForm, HessianSymmetry, HessianUpdate,
    HessianUpdateReport, HessianUpdateWorkspace, InteriorPoint, InteriorPointDiagnostics,
    InteriorPointLp, InteriorPointLpWorkspace, InteriorPointOptions, InteriorPointState,
    InteriorPointWorkspace, InteriorPointWorkspaceRequirements, KktProvider,
    KrigingSurrogateFitter, LagrangianHessian, Lbfgs, LbfgsB, LbfgsBFitSolver, LbfgsBOptions,
    LbfgsBWorkspace, LbfgsBWorkspaceRequirements, LbfgsOptions, LbfgsWorkspace,
    LbfgsWorkspaceRequirements, LeastSquares, LeastSquaresAccepted, LeastSquaresFitSolver,
    LeastSquaresFn, LeastSquaresGlobalization, LeastSquaresLinearSolver, LeastSquaresLoss,
    LeastSquaresOptions, LeastSquaresProblem, LeastSquaresProgress, LeastSquaresResult,
    LeastSquaresStopping, LeastSquaresWorkspace, LeastSquaresWorkspaceRequest,
    LeastSquaresWorkspaceRequirements, LineSearch, LineSearchFailure, LineSearchResult,
    LineSearchScope, LineSearchTermination, LinearConstraintSense, LinearConstraints,
    LinearLeastSquaresActivity, LinearLeastSquaresKkt, LinearLeastSquaresOptions,
    LinearLeastSquaresProblem, LinearLeastSquaresResult, LocalSolverFactory, LpProblem,
    LpVerificationProblem, MinimizerAccepted, MinimizerProgress, MinimumBracket, MinimumSample,
    MipBranching, MipCutKind, MipLimits, MipNode, MipPresolveMap, MipProblem, MipRelaxationSolver,
    MipResult, MipSearchState, MipSimplexWorkspace, MipStatistics, MultiStart, NelderMead,
    NelderMeadDiagnostics, NelderMeadGeometry, NelderMeadOptions, NelderMeadWorkspace,
    NelderMeadWorkspaceRequirements, NewtonCgTermination, NewtonDiagnostics, NewtonOptimizer,
    NewtonOptimizerWorkspace, NewtonOptimizerWorkspaceRequirements, NewtonOptions, Nnls,
    NnlsWorkspace, NoiseModel, NonlinearCg, NonlinearCgOptions, NonlinearCgWorkspace,
    NonlinearCgWorkspaceRequirements, Nsga2, Nsga2Checkpoint, Nsga2Options, Nsga2Problem,
    Nsga2Progress, Nsga2Result, Nsga2Workspace, Nsga2WorkspaceRequirements, ObjectiveFn,
    ObjectiveFunction, OptimalityDiagnostics, OptimalityTolerances, OptimizationSampleKey,
    OptimizationSampler, OptimizeDiagnostics, OptimizeResult, OptimizeTermination, Optimizer,
    OwnedDualCertificate, OwnedProgrammingCertificate, ParetoArchive, ParetoDominance, ParetoEntry,
    PenaltyMethod, PenaltyMethodWorkspace, PenaltyMethodWorkspaceRequirements, PenaltyOptions,
    PhiloxSampler, Powell, PowellDiagnostics, PowellLocalFactory, PowellOptions, PowellSearchLocal,
    PowellWorkspace, PowellWorkspaceRequirements, ProgrammingAccepted, ProgrammingCertificate,
    ProgrammingClaim, ProgrammingProgress, ProgrammingResult, ProgrammingVerifier,
    ProximalGradient, ProximalMap, ProximalOptions, ProximalProgress, ProximalResult,
    ProximalWorkspace, ProximalWorkspaceRequirements, QapDirection, QapMethod, QapOptions, QapPlan,
    QapProblemView, QapResult, QapWorkspace, QpMethod, QpProblem, QpSolver, QpVerificationProblem,
    QpWarmStart, QpWorkspace, RbfSurrogateFitter, ReplicationOptions, SampledGradient,
    SampledObjective, ScalarMinimizer, ScalarMinimumMethod, ScalarMinimumOptions,
    ScalarMinimumResult, ScalarMinimumStart, ScalarObjective, SearchLocalSolver, SearchRandom,
    SearchStartReport, SgdOptions, SgdStep, Shgo, ShgoOptions, ShgoRefinement, ShgoResult,
    ShgoWorkspace, ShgoWorkspaceRequirements, SimplexSolver, SimplexTransportSolver,
    SimplexWarmStart, SimplexWorkspace, SmoothGradient, SparseBasisProvider, SparseKktProvider,
    Sqp, SqpDiagnostics, SqpFitSolver, SqpOptions, SqpState, SqpSubproblem, SqpSubproblemObjective,
    SqpSubproblemProvider, SqpSubproblemReport, SqpWorkspace, SqpWorkspaceRequirements,
    StartDesign, StochasticCheckpoint, StochasticGradientDescent, StochasticProblem,
    StochasticProgress, StochasticResult, StochasticTrustOptions, StochasticTrustRegion,
    StochasticWorkspace, StochasticWorkspaceRequirements, StrongWolfeLineSearch,
    SurrogateAcquisition, SurrogateFeasibility, SurrogateFitter, SurrogateHistory, SurrogateModel,
    SurrogateObservation, SurrogatePrediction, SurrogateSearch, SurrogateSearchOptions,
    SurrogateSearchReport, SurrogateSearchWorkspace, TrustRegionOptimizer,
    TrustRegionOptimizerWorkspace, TrustRegionOptimizerWorkspaceRequirements, TrustRegionOptions,
    TrustRegionSubproblem, TrustRegionWorkspaceRequest, UniformBasinProposal, UniformStartDesign,
    UpdateSkipReason, UpdatingMode, VerificationOptions, VerificationProblem, VerificationReport,
    VerificationStatus, VerificationWorkspace, pareto_dominance,
};

use scioxide_core::{Complex, SciOxideError};

// Constructors that validate the whole configuration.
const _: fn(ProximalOptions<f64>) -> Result<ProximalGradient<f64>, SciOxideError> =
    ProximalGradient::new;
const _: fn(ProximalOptions<f64>) -> Result<Fista<f64>, SciOxideError> = Fista::new;
const _: fn(AdmmOptions<f64>) -> Result<Admm<f64>, SciOxideError> = Admm::new;
const _: fn(Nsga2Options<f64>) -> Result<Nsga2<f64>, SciOxideError> = Nsga2::new;
const _: fn(SgdOptions<f64>) -> Result<StochasticGradientDescent<f64>, SciOxideError> =
    StochasticGradientDescent::new;
const _: fn(StochasticTrustOptions<f64>) -> Result<StochasticTrustRegion<f64>, SciOxideError> =
    StochasticTrustRegion::new;
const _: fn(LeastSquaresOptions<f64>) -> Result<LeastSquares<f64>, SciOxideError> =
    LeastSquares::new;
const _: fn(
    DifferentialEvolutionOptions<f64>,
) -> Result<DifferentialEvolution<f64>, SciOxideError> = DifferentialEvolution::new;

// Checkpoint creation is fallible and copies the accepted state.
const _: fn(&BfgsWorkspace<f64>) -> Result<BfgsCheckpoint<f64>, SciOxideError> =
    BfgsWorkspace::checkpoint;
const _: fn(&Nsga2Workspace<f64>) -> Result<Nsga2Checkpoint<f64>, SciOxideError> =
    Nsga2Workspace::checkpoint;
const _: fn(&StochasticWorkspace<f64>) -> Result<StochasticCheckpoint<f64>, SciOxideError> =
    StochasticWorkspace::checkpoint;

#[test]
fn plans_and_options_are_clone_send_sync() {
    fn check<T: Clone + Send + Sync>() {}
    // Local minimizers, line searches and least squares.
    check::<Bfgs<f64>>();
    check::<BfgsOptions<f64>>();
    check::<Lbfgs<f64>>();
    check::<LbfgsB<f64>>();
    check::<GradientDescent<f64>>();
    check::<NonlinearCg<f64>>();
    check::<NewtonOptimizer<f64>>();
    check::<TrustRegionOptimizer<f64>>();
    check::<NelderMead<f64>>();
    check::<Powell<f64>>();
    check::<ScalarMinimizer<f64>>();
    check::<LeastSquares<f64>>();
    check::<LeastSquaresOptions<f64>>();
    check::<Nnls<f64>>();
    check::<BoundedLinearLeastSquares<f64>>();
    check::<BacktrackingLineSearch<f64>>();
    check::<StrongWolfeLineSearch<f64>>();
    // Nonlinear constrained.
    check::<PenaltyMethod<f64>>();
    check::<PenaltyOptions<f64>>();
    check::<AugmentedLagrangian<f64>>();
    check::<Sqp<f64>>();
    check::<SqpOptions<f64>>();
    check::<InteriorPoint<f64>>();
    check::<Cobyla<f64>>();
    check::<CobylaOptions<f64>>();
    check::<Cobyqa<f64>>();
    // Programming, verification and assignment.
    check::<SimplexSolver<f64>>();
    check::<InteriorPointLp<f64>>();
    check::<QpSolver<f64>>();
    check::<BranchAndBound<f64>>();
    check::<ProgrammingVerifier<f64>>();
    check::<VerificationOptions<f64>>();
    check::<QapOptions<f64>>();
    check::<QapPlan<f64>>();
    // Global search with standard configurations.
    check::<DifferentialEvolution<f64>>();
    check::<DifferentialEvolutionOptions<f64>>();
    check::<CmaEs<f64>>();
    check::<CmaEsOptions<f64>>();
    // Composite, splitting and sampled workflows.
    check::<ProximalGradient<f64>>();
    check::<Fista<f64>>();
    check::<ProximalOptions<f64>>();
    check::<Admm<f64>>();
    check::<AdmmOptions<f64>>();
    check::<Nsga2<f64>>();
    check::<Nsga2Options<f64>>();
    check::<StochasticGradientDescent<f64>>();
    check::<SgdOptions<f64>>();
    check::<StochasticTrustRegion<f64>>();
    check::<StochasticTrustOptions<f64>>();
    check::<SimplexTransportSolver<f64>>();
}

#[test]
fn workspaces_and_checkpoints_are_send() {
    fn check<T: Send>() {}
    check::<BfgsWorkspace<f64>>();
    check::<BfgsCheckpoint<f64>>();
    check::<LeastSquaresWorkspace<f64>>();
    check::<SqpWorkspace<f64>>();
    check::<InteriorPointWorkspace<f64>>();
    check::<SimplexWorkspace<f64>>();
    check::<InteriorPointLpWorkspace<f64>>();
    check::<QpWorkspace<f64>>();
    check::<BranchAndBoundWorkspace<f64>>();
    check::<QapWorkspace<f64>>();
    check::<ProximalWorkspace<Complex<f64>>>();
    check::<AdmmWorkspace<f64>>();
    check::<Nsga2Workspace<f64>>();
    check::<Nsga2Checkpoint<f64>>();
    check::<StochasticWorkspace<f64>>();
    check::<StochasticCheckpoint<f64>>();
}

#[allow(unused_imports)]
use scioxide_optimize::{
    bvls, curve_fit, least_squares, minimize, minimize_bounded, minimize_constrained,
    minimize_global, minimize_scalar, nnls, quadratic_assignment,
};
type ObjectiveCallback = fn(&[f64]) -> Result<f64, SciOxideError>;
type ResidualCallback = fn(&[f64], &mut [f64]) -> Result<(), SciOxideError>;
const _: fn(ObjectiveCallback, &[f64]) -> Result<OptimizeResult<f64>, SciOxideError> = minimize;
#[allow(clippy::type_complexity, reason = "pins the public function signature")]
const _: fn(ResidualCallback, &[f64], usize) -> Result<LeastSquaresResult<f64>, SciOxideError> =
    least_squares;
#[allow(clippy::type_complexity, reason = "pins the public function signature")]
const _: fn(ObjectiveCallback, &[(f64, f64)], u64) -> Result<OptimizeResult<f64>, SciOxideError> =
    minimize_global;
#[allow(clippy::type_complexity, reason = "pins the public method signature")]
const _: fn(
    &QapPlan<f64>,
    QapDirection,
    Option<scioxide_core::PermutationView<'_>>,
) -> Result<QapResult<f64>, SciOxideError> = QapPlan::solve;

#[test]
fn method_values_and_runtime_selection() {
    fn check<T: Clone + Send + Sync>() {}
    check::<Direct<f64>>();
    check::<MultiStart<f64, UniformStartDesign, PowellLocalFactory<f64>>>();
    check::<Shgo<f64, UniformStartDesign, PowellLocalFactory<f64>>>();
    check::<DualAnnealing<f64, PowellLocalFactory<f64>>>();
    check::<BasinHopping<f64, UniformBasinProposal<f64>, PowellLocalFactory<f64>>>();
    let _: [Box<dyn Optimizer<f64>>; 2] =
        [Box::new(Bfgs::default()), Box::new(NelderMead::default())];
    let _ = DifferentialEvolution::<f64>::default();
    let _ = CmaEs::<f64>::default();
    let _ = DualAnnealingOptions::<f64>::default();
    let _ = QapOptions::<f64>::default();
}
