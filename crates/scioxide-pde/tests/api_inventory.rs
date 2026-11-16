//! The authoritative scioxide-pde root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test fails
//! when this list and the `pub use` items of `src/lib.rs` differ. Every exported plan
//! and option type is asserted `Send + Sync + Clone`, and the requirements, execution
//! and attempt forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_pde::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    // Boundary vocabulary and data.
    ConstantDiffusion, IsotropicDiffusion,
    BoundaryCondition, BoundaryConditions, BoundaryData, BoundaryKind, BoundaryPoint,
    DirichletConflictPolicy, FieldData, FieldPoint, PdeData, StructuredBoundaries,
    TaggedBoundaries, TimeDependence,
    // Discretization plans for D = 1, 2, 3.
    SpatialDiscretization, SpatialLimits, StructuredFdmCoefficients, StructuredFdmPlan,
    FvmCoefficients, FvmFluxWorkspace, SimplexFvmPlan, StructuredFvmPlan,
    SimplexFemCoefficients, SimplexFemField, SimplexFemFieldView, SimplexFemGeometry,
    SimplexFemLimits, SimplexFemMappedPoint, SimplexFemMassOperator, SimplexFemOptions,
    SimplexFemOrder, SimplexFemPlan,
    SimplexCsrWorkspace, SimplexLinearization,
    // Steady solves.
    EllipticGauge, EllipticOptions, EllipticProblem, EllipticSolution,
    EllipticWorkspace, PdeDiagnostics, PdeNullspaceSource, PdeSolveReport, PdeTermination,
    // Transient solves, output and progress.
    ParabolicOptions, ParabolicScheme, RannacherStartup,
    HyperbolicOptions, HyperbolicProblem, ParabolicProblem, PdeState, PdeCheckpoint, PdeProgress,
    PdeSolution, PdeWorkspace, PdeObservation, PdeOutputSchedule, PdeOutputSink, PdeTrajectory,
    SemidiscreteSystem, CflPolicy, MethodOfLinesFields, MethodOfLinesOrder, MethodOfLinesPlan,
    MethodOfLinesStagePreconditioner, MethodOfLinesSystem, MethodOfLinesWorkspace,
    // Nonlinear residuals and tangents.
    NonlinearPdeProblem, NonlinearPdeWorkspace, PdeAdjointCallback,
    PdeAdjointJacobian, PdeFieldLayout, PdeFieldPreconditioner, PdeJacobianMode, PdeResidual,
    PdeTangent,
    // Elasticity.
    ElasticityEnergy, ElasticityField, ElasticityFieldView, ElasticityFormulation,
    ElasticityKinematics, ElasticityOptions, ElasticityRigidModes, ElasticityStress,
    HyperelasticityPlan, IsotropicElasticMaterial, LinearElasticityPlan, PlaneStressOptions,
    HyperelasticityLinearizationWorkspace,
    // Incompressible flow.
    IncompressibleDiagnostics, IncompressibleField, IncompressibleFieldView,
    IncompressibleOperator, IncompressibleOptions, IncompressiblePreconditioner,
    IncompressiblePreconditionerOptions, IncompressibleSaddle, IncompressibleSolveOptions,
    IncompressibleSystem, IncompressibleTimeWorkspace, IncompressibleVelocityInverse,
    NavierStokesPlan, NavierStokesWorkspace, StokesPlan, StokesWorkspace,
    NavierStokesLinearizationWorkspace,
    // Conservation laws.
    Burgers, ConservationBoundary, ConservationLaw, ConservationLimits, ConservationOptions,
    ConservationReconstruction, ConservationReport, ConservationSystem, ConservationWorkspace,
    Godunov, Hllc, Hlle, IdealGasEuler, LinearAdvection, NumericalFlux, RiemannData, RiemannReport, Rusanov,
    SimplexConservationPlan, StructuredConservationPlan,
    // Adaptive refinement.
    AmrBoundaryTags, AmrCellTransfer, AmrDiffusionData, AmrDiffusionPlan, AmrEstimator,
    AmrEstimatorWorkspace, AmrIndicators, AmrLimits, AmrMap, AmrMarking, AmrMarks, AmrMassSolve,
    AmrNodalField, AmrNodalFieldView, AmrNodalTransfer, AmrOptions, AmrOrder,
    AmrProjectionReport, AmrResidual, AmrSpace, AmrTransferWorkspace,
    // Discrete adjoints.
    DifferentiableDiscretization, PdeConditionTangent, PdeParameterTangent, PdeParameterization,
    PdeEllipticAdjointPlan, PdeEllipticAdjointWorkspace, PdeDiscretizationDerivatives,
    PdeLinearSolve, PdeSteadyAdjointOptions, PdeSteadyAdjointReport,
    PdeSteadyAdjointResult, PdeSteadyAdjointWorkspace, PdeTransferDerivatives,
    PdeTransferWorkspace, PdeTransientAdjointPlan, PdeTransientAdjointWorkspace,
    SemidiscreteAdjointSystem,
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
    assert_eq!(exported.len(), 163);
    assert_eq!(
        exported
            .iter()
            .filter(|name| name.ends_with("Plan"))
            .count(),
        14
    );
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
    fn shareable<T: Send + Sync + Clone>() {}
    // Discretization plans.
    shareable::<StructuredFdmPlan<f64, f64, 2>>();
    shareable::<StructuredFvmPlan<f64, f64, 2>>();
    shareable::<SimplexFvmPlan<f64, f64, 2>>();
    shareable::<SimplexFemPlan<f64, f64, 2>>();
    shareable::<SimplexFemPlan<f64, scioxide_core::Complex<f64>, 3>>();
    // Solve plans.
    shareable::<MethodOfLinesPlan<f64>>();
    // Physics plans.
    shareable::<LinearElasticityPlan<f64, f64, 2>>();
    shareable::<HyperelasticityPlan<f64, 3>>();
    shareable::<StokesPlan<f64, f64, 2>>();
    shareable::<NavierStokesPlan<f64, 2>>();
    shareable::<StructuredConservationPlan<f64, 2, 4>>();
    shareable::<SimplexConservationPlan<f64, 2, 1>>();
    // Adaptive refinement and adjoints; borrowed parts are immutable `Sync` products.
    shareable::<AmrDiffusionPlan<'static, f64, f64, 2>>();
    shareable::<AmrSpace<f64, 2>>();
    shareable::<AmrEstimator<'static, f64, 2>>();
    shareable::<AmrCellTransfer<f64>>();
    shareable::<AmrNodalTransfer<'static, f64, 2>>();
    shareable::<PdeEllipticAdjointPlan<f64>>();
    shareable::<PdeTransientAdjointPlan<f64>>();
    shareable::<PdeParameterization<f64, f64, 2>>();
    // Options.
    shareable::<EllipticOptions<f64>>();
    shareable::<ParabolicOptions<f64>>();
    shareable::<HyperbolicOptions<f64>>();
    shareable::<SimplexFemOptions<f64>>();
    shareable::<ElasticityOptions<f64>>();
    shareable::<PlaneStressOptions<f64>>();
    shareable::<IncompressibleOptions<f64, f64>>();
    shareable::<IncompressibleSolveOptions<f64>>();
    shareable::<IncompressiblePreconditionerOptions<f64>>();
    shareable::<ConservationOptions<f64>>();
    shareable::<AmrOptions<f64>>();
    shareable::<PdeSteadyAdjointOptions<f64>>();
    shareable::<ConstantDiffusion<f64>>();
    shareable::<SpatialLimits>();
    shareable::<SimplexFemLimits>();
    shareable::<ConservationLimits>();
    shareable::<AmrLimits>();
}

// One progressive and one atomic form per lifecycle, with workspace-owned attempts.
const _: fn() = pin_signatures;
fn pin_signatures() {
    use scioxide_core::{AttemptRecord, ExecutionContext, PartialFailure, SciOxideError};
    type Fdm = StructuredFdmPlan<f64, f64, 1>;
    let _: fn(
        &EllipticProblem<'static, f64, Fdm, 1>,
        &EllipticOptions<f64>,
        &mut [f64],
        &mut EllipticWorkspace<f64>,
        ExecutionContext<'static>,
    ) -> Result<PdeSolveReport<f64>, SciOxideError> = EllipticProblem::solve_into;
    let _: fn(&EllipticProblem<'static, f64, Fdm, 1>) -> Result<Vec<f64>, SciOxideError> =
        EllipticProblem::solve;
    let _: fn(&EllipticWorkspace<f64>) -> Option<&AttemptRecord<()>> =
        EllipticWorkspace::last_attempt;
    let _: fn(
        &ParabolicProblem<'static, f64, Fdm, 1>,
        &ParabolicOptions<f64>,
        &mut PdeState<f64>,
        &mut PdeTrajectory<f64>,
        &mut PdeWorkspace<f64>,
        ExecutionContext<'static>,
    ) -> Result<PdeSolveReport<f64>, SciOxideError> = ParabolicProblem::advance;
    let _: fn(
        &HyperbolicProblem<'static, f64, Fdm, 1>,
        &HyperbolicOptions<f64>,
        &PdeCheckpoint<f64>,
        &mut PdeState<f64>,
        &mut (),
        &mut PdeWorkspace<f64>,
        ExecutionContext<'static>,
    ) -> Result<PdeSolveReport<f64>, SciOxideError> = HyperbolicProblem::resume;
    let _: fn(
        &ParabolicProblem<'static, f64, Fdm, 1>,
        &FieldData<f64, f64, 1>,
        &ParabolicOptions<f64>,
    ) -> Result<PdeSolution<f64>, PartialFailure<PdeProgress<f64>>> =
        ParabolicProblem::solve_with_options;
    let _: fn(&PdeWorkspace<f64>) -> Option<&AttemptRecord<PdeProgress<f64>>> =
        PdeWorkspace::last_attempt;
    let _: fn(&PdeState<f64>) -> Result<PdeCheckpoint<f64>, SciOxideError> = PdeCheckpoint::new;
    let _: fn(
        &PdeTransientAdjointWorkspace<f64>,
    ) -> Option<&AttemptRecord<scioxide_ode::AdjointReport<f64>>> =
        PdeTransientAdjointWorkspace::last_attempt;
}
