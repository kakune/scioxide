//! Crate-root inventory, compiled as an external Rust consumer.
#![expect(
    unused_imports,
    reason = "The external API inventory verifies crate-root exports by importing them."
)]
use scioxide_rootfind::backend::solve_newton_partitioned;
use scioxide_rootfind::{
    BacktrackingLineSearch, Bisection, Bracket, BracketExpansion, BracketSearchDirection,
    BracketSearchInput, BracketSearchOptions, BracketVerification, Brent, Brenth, Broyden,
    BroydenOptions, BroydenUpdate, Chandrupatla, ComplexHalley, ComplexMuller, ComplexNewton,
    ComplexRootResult, ComplexRootSolver, ComplexScalarFn, ComplexScalarFnWithDerivative,
    ComplexScalarFnWithDerivatives, ComplexScalarFunction, ComplexSecant, Continuation,
    ContinuationCheckpoint, ContinuationDiagnostic, ContinuationDiagnosticProvider,
    ContinuationObservation, ContinuationOptions, ContinuationPath, ContinuationProblem,
    ContinuationReport, ContinuationState, ContinuationWorkspace,
    ContinuationWorkspaceRequirements, DeflatedProblem, DeflatedRootResult, DeflationOptions,
    DeflationSet, DeflationWorkspace, DeflationWorkspaceRequirements, DenseJacobian,
    FixedPointDiagnostics, FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointResult,
    FixedPointSolver, FixedPointWorkspace, FixedPointWorkspaceRequirements, Halley, HistoryRestart,
    Illinois, KelvinZeroFunction, LinearHomotopy, Muller, Newton, NewtonKrylovForcing,
    NewtonKrylovInput, NewtonKrylovObservation, NewtonKrylovOptions, NewtonKrylovProblem,
    NewtonKrylovResult, NewtonKrylovSolver, NewtonKrylovWorkspace,
    NewtonKrylovWorkspaceRequirements, NewtonSystem, NewtonSystemOptions, NonlinearLinearSolver,
    NonlinearLinearization, NonlinearScaling, PolynomialRootAlgorithm, PolynomialRootDiagnostic,
    PolynomialRootInput, PolynomialRootMethod, PolynomialRootOptions, PolynomialRootReport,
    PolynomialRootWorkspace, PolynomialRootWorkspaceRequirements, PolynomialRoots,
    RealifiedProblem, RegulaFalsi, ResidualFn, Ridder, RootLineSearch, RootResult, RootTermination,
    SafeguardedHalley, SafeguardedInitialGuess, SafeguardedNewton, ScalarFn,
    ScalarFnWithDerivative, ScalarFnWithDerivatives, ScalarFunction, ScalarRootInput,
    ScalarRootMethod, ScalarRootOptions, ScalarRootSolver, Secant, SpecialZero, SpecialZeroBranch,
    SpecialZeroFamily, SpecialZeroOptions, SpecialZeroReport, SpecialZeroWorkspace,
    SpecialZeroWorkspaceRequirements, SpecialZeros, SystemRootResult, SystemRootSolver,
    SystemWorkspace, SystemWorkspaceRequirements, Toms748, TrustRegionSystem,
    TrustRegionSystemOptions, find_bracket, find_bracket_scoped, find_bracket_with_options,
    find_root, fixed_point, polynomial_roots, polynomial_roots_into, polynomial_roots_with_options,
    solve_system, special_zeros, special_zeros_into, special_zeros_with_options,
};

const _: fn(NewtonSystemOptions<f64>) -> Result<NewtonSystem<f64>, scioxide_core::SciOxideError> =
    NewtonSystem::new;
const _: fn(
    BroydenUpdate,
    BroydenOptions<f64>,
) -> Result<Broyden<f64>, scioxide_core::SciOxideError> = Broyden::new;
const _: fn(
    TrustRegionSystemOptions<f64>,
) -> Result<TrustRegionSystem<f64>, scioxide_core::SciOxideError> = TrustRegionSystem::new;
type FixedPointConstructor = fn(
    FixedPointMethod,
    FixedPointInput,
    FixedPointOptions<f64>,
) -> Result<FixedPointSolver<f64>, scioxide_core::SciOxideError>;
const _: FixedPointConstructor = FixedPointSolver::new;
const _: fn(
    NewtonKrylovOptions<f64>,
) -> Result<NewtonKrylovSolver<f64>, scioxide_core::SciOxideError> = NewtonKrylovSolver::new;
type Coefficients = &'static [f64];
const _: fn(
    Coefficients,
) -> Result<Vec<scioxide_core::Complex<f64>>, scioxide_core::SciOxideError> =
    polynomial_roots::<f64, f64>;
const _: fn(
    Coefficients,
    &PolynomialRootOptions<f64>,
) -> Result<PolynomialRoots<f64>, scioxide_core::SciOxideError> =
    polynomial_roots_with_options::<f64, f64>;
type PolynomialRootsInto = fn(
    Coefficients,
    &PolynomialRootOptions<f64>,
    &mut [scioxide_core::Complex<f64>],
    &mut PolynomialRootWorkspace<f64>,
    scioxide_core::ExecutionContext<'static>,
) -> Result<PolynomialRootReport<f64>, scioxide_core::SciOxideError>;
const _: PolynomialRootsInto = polynomial_roots_into::<f64, f64>;
const _: fn(
    usize,
    &PolynomialRootOptions<f64>,
) -> Result<PolynomialRootWorkspaceRequirements<f64>, scioxide_core::SciOxideError> =
    PolynomialRootWorkspace::<f64>::workspace_requirements;
const _: fn(
    usize,
    &PolynomialRootOptions<f64>,
) -> Result<PolynomialRootWorkspace<f64>, scioxide_core::SciOxideError> =
    PolynomialRootWorkspace::<f64>::workspace;
const _: fn(PolynomialRootOptions<f64>, PolynomialRootMethod) -> PolynomialRootOptions<f64> =
    PolynomialRootOptions::with_method;
const _: fn(&PolynomialRootOptions<f64>) -> PolynomialRootMethod = PolynomialRootOptions::method;

type ScalarCallback = fn(f64) -> Result<f64, scioxide_core::SciOxideError>;
const _: fn(ScalarCallback, f64, f64) -> Result<f64, scioxide_core::SciOxideError> =
    find_root::<f64, ScalarCallback>;
const _: fn(ScalarCallback, f64) -> Result<Bracket<f64>, scioxide_core::SciOxideError> =
    find_bracket::<f64, ScalarCallback>;
const _: fn(SpecialZeroFamily<f64>, usize) -> Result<Vec<f64>, scioxide_core::SciOxideError> =
    special_zeros::<f64>;
type SpecialZerosWithOptions = fn(
    SpecialZeroFamily<f64>,
    SpecialZeroBranch,
    usize,
    usize,
    &SpecialZeroOptions<f64>,
) -> Result<SpecialZeros<f64>, scioxide_core::SciOxideError>;
const _: SpecialZerosWithOptions = special_zeros_with_options::<f64>;
type SpecialZerosInto = fn(
    SpecialZeroFamily<f64>,
    SpecialZeroBranch,
    usize,
    &SpecialZeroOptions<f64>,
    &mut [Option<SpecialZero<f64>>],
    &mut SpecialZeroWorkspace<f64>,
    scioxide_core::ExecutionContext<'static>,
) -> Result<SpecialZeroReport<f64>, scioxide_core::SciOxideError>;
const _: SpecialZerosInto = special_zeros_into;
const _: fn(
    usize,
    &SpecialZeroOptions<f64>,
) -> Result<SpecialZeroWorkspaceRequirements<f64>, scioxide_core::SciOxideError> =
    SpecialZeroWorkspace::<f64>::workspace_requirements;
const _: fn(
    usize,
    &SpecialZeroOptions<f64>,
) -> Result<SpecialZeroWorkspace<f64>, scioxide_core::SciOxideError> =
    SpecialZeroWorkspace::<f64>::workspace;
type VectorCallback = fn(&[f64], &mut [f64]) -> Result<(), scioxide_core::SciOxideError>;
const _: fn(VectorCallback, &[f64]) -> Result<Vec<f64>, scioxide_core::SciOxideError> =
    solve_system::<f64, VectorCallback>;
const _: fn(VectorCallback, &[f64]) -> Result<Vec<f64>, scioxide_core::SciOxideError> =
    fixed_point::<f64, VectorCallback>;

#[test]
fn standard_immutable_configuration_is_clone_send_sync() {
    fn check<T: Clone + Send + Sync>() {}
    check::<NewtonSystemOptions<f64>>();
    check::<BroydenOptions<f64>>();
    check::<TrustRegionSystemOptions<f64>>();
    check::<NewtonSystem<f64>>();
    check::<Broyden<f64>>();
    check::<TrustRegionSystem<f64>>();
    check::<FixedPointSolver<f64>>();
    check::<FixedPointSolver<scioxide_core::Complex<f64>>>();
    check::<NewtonKrylovSolver<f64>>();
    check::<NewtonKrylovSolver<scioxide_core::Complex<f64>>>();
    check::<Continuation<f64>>();
    check::<PolynomialRootOptions<f64>>();
    check::<SpecialZeroOptions<f64>>();
}

// The single-root result families share names, borrowed vectors and accounting types.
macro_rules! vector_result_accessors {
    ($result:ty) => {
        const _: fn(&$result) -> &[f64] = <$result>::point;
        const _: fn(&$result) -> &[f64] = <$result>::residual;
        const _: fn(&$result) -> Option<scioxide_core::ResidualReport<f64>> =
            <$result>::residual_report;
        const _: fn(&$result) -> Option<f64> = <$result>::residual_norm;
        const _: fn(&$result) -> RootTermination = <$result>::termination;
        const _: fn(&$result) -> usize = <$result>::iterations;
        const _: fn(&$result) -> scioxide_core::EvaluationCounts = <$result>::counts;
        const _: fn(&$result) -> scioxide_core::SolveCounts = <$result>::accounting;
    };
}
vector_result_accessors!(SystemRootResult<f64>);
vector_result_accessors!(FixedPointResult<f64>);
vector_result_accessors!(NewtonKrylovResult<f64>);
vector_result_accessors!(DeflatedRootResult<f64>);
vector_result_accessors!(ContinuationState<f64>);
type Follow =
    fn(
        &Continuation<f64>,
        &ContinuationProblem<'_, f64>,
        ContinuationState<f64>,
    )
        -> Result<ContinuationState<f64>, scioxide_core::PartialFailure<ContinuationState<f64>>>;
const _: Follow = Continuation::follow;
const _: fn(&RootResult<f64>) -> f64 = RootResult::point;
const _: fn(&RootResult<f64>) -> f64 = RootResult::residual;
const _: fn(&RootResult<f64>) -> scioxide_core::ResidualReport<f64> = RootResult::residual_report;
const _: fn(&RootResult<f64>) -> scioxide_core::EvaluationCounts = RootResult::counts;
const _: fn(&RootResult<f64>) -> scioxide_core::SolveCounts = RootResult::accounting;
const _: fn(&ComplexRootResult<f64>) -> scioxide_core::Complex<f64> = ComplexRootResult::point;
const _: fn(&SpecialZeroReport<f64>) -> RootTermination = SpecialZeroReport::termination;
const _: fn(&ContinuationReport<f64>) -> RootTermination = ContinuationReport::termination;
const _: fn(PolynomialRootReport<f64>) -> RootTermination = PolynomialRootReport::termination;
