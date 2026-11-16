# scioxide-pde

MOL dense derivatives require `workspace()?.with_dense_jacobian()?` before binding.
Default workspaces retain linear vectors and declared sparse-pattern storage; derivative callbacks publish only after normalization succeeds.

PDE discretizations and solvers: finite differences, P1/P2 simplex finite elements, finite volumes, method of lines, elasticity, incompressible flow, conservation laws, AMR and discrete adjoints.

## Main entry points

- Plans for D = 1, 2, 3: `StructuredFdmPlan`, `StructuredFvmPlan`, `SimplexFvmPlan`, `SimplexFemPlan` (simplex families include `IntervalMesh`), sharing `SpatialDiscretization`, the core `LinearOperator` and one `BoundaryConditions` vocabulary
- Steady solves: `EllipticProblem::{solve, solve_with_options, solve_into}`
- Transient solves: `ParabolicProblem`, `HyperbolicProblem` with a caller-owned `PdeState` (`advance`, `resume` from `PdeCheckpoint`, owned `solve`)
- Physics: `LinearElasticityPlan`, `HyperelasticityPlan`, `StokesPlan`, `NavierStokesPlan`, `IncompressibleSystem`; conservation: `StructuredConservationPlan`, `SimplexConservationPlan`
- Nonlinearity, adaptation and derivatives: `SemidiscreteSystem`, `MethodOfLinesPlan`, `NonlinearPdeProblem`, `PdeTangent`, `AmrSpace`
- Discrete adjoints of the plans' own operators: `DifferentiableDiscretization::parameterize` with `PdeParameterTangent` derivative data, `PdeEllipticAdjointPlan` (steady) and `PdeTransientAdjointPlan` (backward Euler, Crank–Nicolson, DP5); `PdeDiscretizationDerivatives` for caller-supplied residual derivatives

Natural data is the outward conormal flux `n·K∇u = g` and Robin data is `n·K∇u + αu = g` in every family; Dirichlet unknowns are eliminated symmetrically.
`load_into` uses caller scratch sized by `load_workspace_len()` and preserves output on failure.
Default elliptic solves use tridiagonal factors for bandwidth-one operators and fill-reducing sparse LU otherwise, precision-derived tolerances, and explicit gauges for known nullspaces.
Iterative solvers are opt-in and retain scoped execution and typed outcomes.
Transient solves run through one method-of-lines adapter with lumped mass and automatic step limits derived from the integrator's declared linear stability.
Parabolic problems default to adaptive Radau IIA with sparse stage factors; hyperbolic problems use Dormand–Prince 5(4).
Both default to 8-decimal RMS tolerances and 100,000 attempted steps.
States and checkpoints retain the free-node layout; changing that layout requires a new initial state.
A rejected resume preserves the destination state.
Structured stencil application and method-of-lines kernels use prepared native arithmetic with range-safe fallback and transactional RHS/product output.
Semidiscrete systems reuse successfully sampled loads when source and plan data are declared time-invariant; rebuild the problem after changing captured data.

FVM uses cell-owned diffusion and harmonic half-cell conductances.
Invalid geometric conductances reject preparation.
Simplex FEM defaults to positive interior quadrature and projects rounding-level Hermitian coefficient defects before checking positive definiteness.
Affine maps and declared constant coefficients reuse prepared geometry and reference moments.
`assembly_workspace()` and `assemble_csr_into` refill fixed CSR patterns, including numerical zeros.
Hyperelastic and Navier–Stokes `linearization_workspace()` and `linearize` evaluate a state once for repeated tangent products and CSR publication; reserved numerical calls allocate nothing.
`NonlinearPdeProblem` workspaces retain one validated dense or sparse Jacobian for repeated tangent and adjoint products at the same state.
Binding resets reuse; rebind after changing captured provider data.
Matrix-free products call the provider on every application.
Shared Dirichlet nodes reject conflicting traces by default; opt-in `LastConditionWins` supports discontinuous corner data in every family.
Flow solves default to library block preconditioning with sparse Cholesky velocity solves (AMG is selectable).
Transient flow reuses mass-saddle inverses and warm starts, restores divergence in the RHS, and computes viscous/convective step caps.
Conservation reconstruction respects boundary extrema; inadmissible SSP stages reject and retry within the ODE driver's bounds.
Euler face solves share primitive states, and native arithmetic falls back to scaled operations at range limits.
Explicit Runge–Kutta steps obtain CFL bounds with the first residual and honor the conservation execution policy.

Geometry comes from `scioxide-spatial`; linear algebra from `scioxide-linalg` and `scioxide-sparse`.

Parameterizations and adjoint plans belong to one prepared spatial operator; clones remain compatible, while independently prepared operators require new adjoint preparation.
Derivative products stage their output and use checked reductions at range limits.
Adjoint byte ceilings include adapter staging and reserved factor storage.

## Features

- `std` (default): standard-library implementations.
  Disable default features for `no_std + alloc`; the public API is the same.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
