//! PDE discretizations and solves in one to three dimensions: finite differences,
//! finite volumes and P1/P2 simplex finite elements behind one discretization protocol,
//! steady and transient solve plans, elasticity, incompressible flow, conservation laws,
//! adaptive refinement and discrete adjoints. Every operation follows configuration →
//! preparation → execution: immutable `Send + Sync + Clone` plans, caller workspaces that
//! own attempt records, and no caller identities.
//!
//! # Discretization plans
//!
//! Each family has one prepared plan serving `D = 1, 2, 3`: [`StructuredFdmPlan`]
//! (vertex-centred conservative finite differences on structured meshes, including
//! periodic axes), [`StructuredFvmPlan`] and [`SimplexFvmPlan`] (cell-centred finite
//! volumes) and [`SimplexFemPlan`] (P1/P2 finite elements on intervals, triangles and
//! tetrahedra; the order is a positional argument). They share [`SpatialDiscretization`]:
//! a time-invariant eliminated operator `A`, positive lumped mass, structural constant
//! modes and loads `b(t)` evaluated at any time. Application has one route, the core
//! `LinearOperator` (`apply_into` sequentially, `apply_into_scoped` with the caller's
//! capabilities and the options' execution policy).
//!
//! # Boundary conditions
//!
//! One vocabulary serves every family: [`BoundaryCondition`] combines a mesh-family
//! region (a structured `BoundarySet` or a `u32` face tag), an optional component, a
//! [`BoundaryKind`] and time-classified [`BoundaryData`]. [`BoundaryConditions`] validates
//! a complete list against one exact mesh revision through a single path. Natural data
//! is always the outward conormal flux `n·K∇u`, and Robin data is `n·K∇u + αu = g`.
//! Dirichlet unknowns are eliminated symmetrically, so symmetric problems assemble
//! symmetric matrices. Periodicity belongs to mesh topology. [`DirichletConflictPolicy`]
//! resolves shared Dirichlet nodes for every family.
//!
//! # Steady solves
//!
//! [`EllipticProblem`] binds a discretization, source and optional nullspace bases.
//! Its `solve()` returns the full field; `solve_with_options` returns diagnostics,
//! and `solve_into(options, out, workspace, scope)` uses caller storage. Compact bands
//! use tridiagonal factors, other operators fill-reducing sparse LU. An
//! [`EllipticWorkspace`] may own an iterative solver. Known nullspaces need an
//! explicit [`EllipticGauge`] through options.
//!
//! # Transient solves
//!
//! [`ParabolicProblem`] and [`HyperbolicProblem`] prepare `M u' = b(t) − A u` and
//! `M u'' = b(t) − A u` once from a discretization, source and end time. Parabolic
//! solves default to adaptive, fifth-order L-stable Radau IIA with sparse stage
//! factors. Hyperbolic solves default to Dormand–Prince 5(4) with automatic wave
//! stability limits. Both use 8-decimal RMS tolerances and 100,000 attempted steps.
//! `solve` returns the final full field and rejects unsuccessful termination;
//! `solve_with_options` retains the report, state and trajectory. Owned failures
//! carry accepted [`PdeProgress`] in `PartialFailure` when it can be stored.
//! `advance(options, state, sink, workspace, scope)` continues caller-owned state;
//! `resume` restores a compatible [`PdeCheckpoint`]. [`PdeState::coordinates`] exposes
//! free ODE coordinates; each problem's `values` reconstructs the physical field.
//! [`PdeOutputSchedule`] selects final, step or requested-time output; `()` ignores
//! output and [`PdeTrajectory`] stores it. [`PdeTermination`] is the sole status.
//!
//! # Physics, conservation, adaptation and derivatives
//!
//! [`LinearElasticityPlan`] and [`HyperelasticityPlan`] add isotropic Hooke and
//! neo-Hookean weak forms, plane stress/strain and 3D reductions and stable mixed P2/P1
//! pressure ([`ElasticityFormulation`]). [`StokesPlan`] and [`NavierStokesPlan`] provide
//! stable P2/P1 flow with explicit pressure gauges and library saddle preconditioning;
//! their workspaces own injected solvers, and [`IncompressibleSystem`] composes
//! constrained velocity with explicit ODE methods. [`NonlinearPdeProblem::bind`] binds a
//! physical provider at a fixed time to a [`NonlinearPdeWorkspace`] as a [`PdeResidual`],
//! which retains one validated dense or sparse Jacobian per derivative state. [`PdeTangent`]
//! borrows a provider JVP; simplex physics plans also expose [`SimplexLinearization`] from
//! caller-owned workspaces to evaluate a state once and reuse its CSR tangent.
//!
//! Nonlinear conservation laws use [`StructuredConservationPlan`] and
//! [`SimplexConservationPlan`] with [`ConservationLaw`]/[`NumericalFlux`] providers, slope
//! limiting and an admissibility-checked [`ConservationSystem`]. Error-driven AMR
//! ([`AmrSpace`], [`AmrEstimator`], [`AmrNodalTransfer`]) composes over shared adaptive
//! meshes with hanging constraints. Spaces cache physical face quadrature and
//! exterior-cell lookup. Estimation uses linear work; marking and transfer
//! preparation use at most `O(n log n)` work for `n` active source/destination
//! cells at fixed order, quadrature degree and bounded constraint support.
//!
//! # Discrete adjoints
//!
//! Linear plans differentiate their own data: [`DifferentiableDiscretization::parameterize`]
//! turns caller derivative data ([`PdeParameterTangent`]) into an exact
//! [`PdeParameterization`] of the eliminated residual. [`PdeEllipticAdjointPlan`] then
//! returns steady gradients and [`PdeTransientAdjointPlan`] fixed-step backward-Euler,
//! Crank–Nicolson or DP5 gradients through `scioxide-ode`'s discrete adjoints. Mesh
//! coordinates are not differentiated. [`PdeDiscretizationDerivatives`] differentiates other
//! residuals from caller-supplied derivatives.
//!
//! The crate is `no_std + alloc` without default features. Geometry and field locations
//! are owned by `scioxide-spatial`; field locations cannot be interchanged even when their
//! lengths agree:
//!
//! ```compile_fail
//! use scioxide_spatial::{FieldView, Vertex, Face};
//! fn vertices(_: FieldView<'_, f64, Vertex, 1>) {}
//! fn wrong_location(faces: FieldView<'_, f64, Face<0>, 1>) { vertices(faces); }
//! ```

#![forbid(unsafe_code)]
#![deny(unreachable_pub)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[macro_use]
extern crate alloc;
#[cfg(test)]
extern crate self as scioxide_pde;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod adjoint;
mod amr;
mod attempt;
mod coefficients;
pub use coefficients::{ConstantDiffusion, IsotropicDiffusion};
mod boundary;
mod conservation;
mod discretization;
mod fdm;
mod fem;
mod fvm;
mod helpers;
mod kernels;
#[cfg(test)]
#[path = "../tests/support/mesh.rs"]
mod mesh_fixture;
mod mol;
#[cfg(test)]
mod mol_tests;
mod nonlinear;
mod result;
mod semidiscrete;
mod spatial;
mod steady;
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod support;

pub use adjoint::{
    DifferentiableDiscretization, PdeConditionTangent, PdeDiscretizationDerivatives,
    PdeEllipticAdjointPlan, PdeEllipticAdjointWorkspace, PdeLinearSolve, PdeParameterTangent,
    PdeParameterization, PdeSteadyAdjointOptions, PdeSteadyAdjointReport, PdeSteadyAdjointResult,
    PdeSteadyAdjointWorkspace, PdeTransferDerivatives, PdeTransferWorkspace,
    PdeTransientAdjointPlan, PdeTransientAdjointWorkspace,
};
pub use amr::{
    AmrBoundaryTags, AmrCellTransfer, AmrDiffusionData, AmrDiffusionPlan, AmrEstimator,
    AmrEstimatorWorkspace, AmrIndicators, AmrLimits, AmrMap, AmrMarking, AmrMarks, AmrMassSolve,
    AmrNodalField, AmrNodalFieldView, AmrNodalTransfer, AmrOptions, AmrOrder, AmrProjectionReport,
    AmrResidual, AmrSpace, AmrTransferWorkspace,
};
pub use boundary::{
    BoundaryCondition, BoundaryConditions, BoundaryData, BoundaryKind, BoundaryPoint,
    DirichletConflictPolicy, FieldData, FieldPoint, PdeData, StructuredBoundaries,
    TaggedBoundaries, TimeDependence,
};
pub use conservation::{
    Burgers, ConservationBoundary, ConservationLaw, ConservationLimits, ConservationOptions,
    ConservationReconstruction, ConservationReport, ConservationSystem, ConservationWorkspace,
    Godunov, Hllc, Hlle, IdealGasEuler, LinearAdvection, NumericalFlux, RiemannData, RiemannReport,
    Rusanov, SimplexConservationPlan, StructuredConservationPlan,
};
pub use discretization::{SpatialDiscretization, SpatialLimits};
pub use fdm::{StructuredFdmCoefficients, StructuredFdmPlan};
pub use fem::simplex::elasticity::{
    ElasticityEnergy, ElasticityField, ElasticityFieldView, ElasticityFormulation,
    ElasticityKinematics, ElasticityOptions, ElasticityRigidModes, ElasticityStress,
    HyperelasticityLinearizationWorkspace, HyperelasticityPlan, IsotropicElasticMaterial,
    LinearElasticityPlan, PlaneStressOptions,
};
pub use fem::simplex::incompressible::{
    IncompressibleDiagnostics, IncompressibleField, IncompressibleFieldView,
    IncompressibleOperator, IncompressibleOptions, IncompressiblePreconditioner,
    IncompressiblePreconditionerOptions, IncompressibleSaddle, IncompressibleSolveOptions,
    IncompressibleSystem, IncompressibleTimeWorkspace, IncompressibleVelocityInverse,
    NavierStokesLinearizationWorkspace, NavierStokesPlan, NavierStokesWorkspace, StokesPlan,
    StokesWorkspace,
};
pub use fem::simplex::{
    SimplexCsrWorkspace, SimplexFemCoefficients, SimplexFemField, SimplexFemFieldView,
    SimplexFemGeometry, SimplexFemLimits, SimplexFemMappedPoint, SimplexFemMassOperator,
    SimplexFemOptions, SimplexFemOrder, SimplexFemPlan, SimplexLinearization,
};
pub use fvm::multidimensional::{
    FvmCoefficients, FvmFluxWorkspace, SimplexFvmPlan, StructuredFvmPlan,
};
pub use mol::{
    CflPolicy, MethodOfLinesFields, MethodOfLinesOrder, MethodOfLinesPlan,
    MethodOfLinesStagePreconditioner, MethodOfLinesSystem, MethodOfLinesWorkspace,
};
pub use nonlinear::{
    NonlinearPdeProblem, NonlinearPdeWorkspace, PdeAdjointCallback, PdeAdjointJacobian,
    PdeFieldLayout, PdeFieldPreconditioner, PdeJacobianMode, PdeResidual, PdeTangent,
};
pub use semidiscrete::{SemidiscreteAdjointSystem, SemidiscreteSystem};

pub use result::{PdeDiagnostics, PdeNullspaceSource, PdeSolveReport, PdeTermination};
pub use steady::{
    EllipticGauge, EllipticOptions, EllipticProblem, EllipticSolution, EllipticWorkspace,
};
pub use transient::{
    HyperbolicOptions, HyperbolicProblem, ParabolicOptions, ParabolicProblem, ParabolicScheme,
    PdeCheckpoint, PdeObservation, PdeOutputSchedule, PdeOutputSink, PdeProgress, PdeSolution,
    PdeState, PdeTrajectory, PdeWorkspace, RannacherStartup,
};

mod elliptic;
mod transient;
mod validation;
