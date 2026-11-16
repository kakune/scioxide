//! Numerical optimization: local, constrained, least-squares, linear, quadratic,
//! integer, global and multiobjective.
//!
//! The crate minimizes smooth and nonsmooth functions of real variables, with or
//! without derivatives, subject to bounds and nonlinear constraints. It fits
//! nonlinear and linear least-squares models, solves linear, quadratic and
//! mixed-integer linear programs, checks claimed LP and QP solutions, and
//! approximates quadratic assignment problems. Global and multiobjective
//! heuristics search a finite box. Composite problems `f(x) + g(x)` use
//! proximal and splitting methods, and sampled objectives use stochastic
//! first-order methods. Every method supports `f32` and `f64`.
//!
//! # Quick start
//!
//! ```
//! let result = scioxide_optimize::minimize(
//!     |x: &[f64]| Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)),
//!     &[0.0, 0.0],
//! )?;
//! assert!((result.x()[0] - 1.0).abs() < 1e-6);
//! assert!((result.x()[1] + 2.0).abs() < 1e-6);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! Supply derivatives through an adapter and a configured method:
//!
//! ```
//! use scioxide_optimize::{Bfgs, ObjectiveFn};
//! let f = ObjectiveFn::new(2, |x: &[f64]| {
//!     Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
//! }).with_gradient(|x: &[f64], g: &mut [f64]| {
//!     let t = x[1] - x[0] * x[0];
//!     g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * t;
//!     g[1] = 200.0 * t;
//!     Ok(())
//! });
//! let result = Bfgs::<f64>::default().minimize(&f, &[-1.0, 1.0])?;
//! result.require_success()?;
//! assert!(result.x().iter().all(|x| (x - 1.0).abs() < 1e-6));
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Entry points | Method | One call |
//! |---|---|---|---|
//! | Smooth, unconstrained, gradient available | [`Bfgs`], [`Lbfgs`] | BFGS and limited-memory BFGS with a line search | [`minimize`] (BFGS) |
//! | Smooth, many variables, low memory | [`Lbfgs`], [`NonlinearCg`], [`GradientDescent`] | L-BFGS; Polak–Ribière nonlinear conjugate gradients with a strong Wolfe search; steepest descent with an Armijo search |  |
//! | Smooth, Hessian or Hessian-vector products available | [`NewtonOptimizer`], [`TrustRegionOptimizer`] | Newton's method with a diagonal shift for indefinite Hessians; trust region with dogleg or Steihaug truncated CG steps ([`TrustRegionSubproblem`]) |  |
//! | Smooth, bound constraints only | [`LbfgsB`] | projected L-BFGS | [`minimize_bounded`] (L-BFGS-B) |
//! | No derivatives | [`NelderMead`], [`Powell`] | Nelder–Mead simplex; Powell's conjugate directions with Brent line minimization |  |
//! | One variable | [`ScalarMinimizer`] | golden-section search or Brent's method ([`ScalarMinimumMethod`]) | [`minimize_scalar`] (Brent) |
//! | Smooth nonlinear constraints | [`Sqp`], [`InteriorPoint`], [`AugmentedLagrangian`], [`PenaltyMethod`] | sequential quadratic programming; primal-dual interior point; augmented Lagrangian; quadratic penalty | [`minimize_constrained`] (SQP) |
//! | Nonlinear constraints, no derivatives | [`Cobyla`], [`Cobyqa`] | trust-region methods on linear or quadratic interpolation models |  |
//! | Nonlinear least squares, real or complex residuals | [`LeastSquares`] | Gauss–Newton with backtracking, or Levenberg–Marquardt; optional robust losses ([`LeastSquaresLoss`]) | [`least_squares`], [`curve_fit`] (Levenberg–Marquardt) |
//! | Linear least squares with `x ≥ 0` or bounds | [`Nnls`], [`BoundedLinearLeastSquares`] | Lawson–Hanson active set; bounded-variable least squares (BVLS) | [`nnls`], [`bvls`] |
//! | Linear programming | [`SimplexSolver`], [`InteriorPointLp`] | bounded revised simplex; homogeneous self-dual interior point | [`LpProblem::solve`] (simplex) |
//! | Quadratic programming | [`QpSolver`] | active set or primal-dual interior point ([`QpMethod`]) | [`QpProblem::solve`] (automatic selection) |
//! | Mixed-integer linear programming | [`BranchAndBound`] | LP-based branch and bound with best-bound search, presolve and cuts | [`MipProblem::solve`] (branch and bound) |
//! | Check a claimed LP or QP solution | [`ProgrammingVerifier`] | recomputes residuals, multiplier signs and gaps for the original problem |  |
//! | Quadratic assignment | [`QapPlan`] | FAQ relaxation, 2-opt, or both ([`QapMethod`]) | [`quadratic_assignment`] (FAQ and 2-opt) |
//! | `f(x) + g(x)` with smooth `f` and a proximal map for `g` | [`ProximalGradient`], [`Fista`] | proximal gradient; accelerated proximal gradient |  |
//! | Separable objective with linear coupling | [`Admm`] | alternating direction method of multipliers |  |
//! | Global search in a finite box | [`MultiStart`], [`BasinHopping`], [`DualAnnealing`], [`DifferentialEvolution`], [`CmaEs`], [`Direct`], [`Shgo`] | multistart local search, basin hopping, dual annealing, differential evolution, CMA-ES, DIRECT, simplicial homology global optimization | [`minimize_global`] (differential evolution) |
//! | Expensive objective, few evaluations | [`SurrogateSearch`] | Kriging or radial basis function surrogate with an acquisition function |  |
//! | Several objectives | [`Nsga2`], [`ParetoArchive`] | NSGA-II; nondominated archive |  |
//! | Objective known only through samples | [`StochasticGradientDescent`], [`StochasticTrustRegion`] | stochastic gradient descent; trust region on sampled models |  |
//!
//! [`minimize`] differentiates numerically and suits smooth objectives; use the
//! derivative-free methods above for nonsmooth problems.
//!
//! Each solver's page names its method, its options and its references.
//!
//! # Problems and callbacks
//!
//! You describe a problem by implementing a trait or by wrapping closures:
//!
//! - [`ObjectiveFunction`]: `f: ℝⁿ → ℝ`, with optional gradient and Hessian.
//!   [`ObjectiveFn`] wraps closures.
//! - [`LeastSquaresProblem`]: a residual vector `r(x)` and optional Jacobian.
//!   [`LeastSquaresFn`] wraps closures.
//! - [`ConstraintFunction`] and [`Constraints`]: equality and inequality
//!   constraints and bounds. [`ConstraintFn`] wraps closures.
//! - [`ConstrainedProblem`]: an objective with its constraints and an optional
//!   [`LagrangianHessian`].
//! - [`BoundedObjective`]: an objective with a box. Global methods and
//!   [`LbfgsB`] take this form.
//! - [`LpProblem`], [`QpProblem`] and [`MipProblem`]: linear and quadratic
//!   programs in matrix form.
//!
//! Adapter closures return scalar values or write solver buffers: gradients,
//! constraint rows and residuals take `(&[S], &mut [S])`; Hessians and Jacobians
//! take `(&[S], &mut Matrix<S>)`; Hessian-vector products take
//! `(&[S], &[S], &mut [S])`. Each writes every entry and returns `Result<(), SciOxideError>`.
//!
//! A derivative protocol method returns `Ok(true)` when it wrote the derivative and
//! `Ok(false)` when it does not provide one. The solver then follows its
//! [`DerivativeStrategy`](scioxide_differentiate::DerivativeStrategy). The
//! line-search minimizers BFGS, L-BFGS, gradient descent, nonlinear CG and Newton
//! default to `AnalyticOrCentral`; other solvers default to `AnalyticOrForward`.
//! Both use analytic derivatives when available, then their finite-difference scheme.
//!
//! An error returned by a callback stops the solve and is returned unchanged.
//! A trial point proposed by the solver whose objective or constraint value is
//! NaN or infinite is rejected, not reported as an error: line searches shorten
//! the step, trust-region methods shrink the radius, and direct-search methods
//! rank the trial last. The starting point, derivatives and the values of
//! global sampling methods must be finite; a non-finite value there is a
//! `NonFinite` error.
//!
//! # Termination
//!
//! Each solve ends with an [`OptimizeTermination`], available from
//! `termination()` on the result. [`OptimizeTermination::is_success`] is true
//! only for `Converged` and `GradientSmall`. Success means that the method's
//! stopping tests passed at a local solution; it does not certify a global
//! minimum.
//!
//! Configured methods return the best accepted point for an unsuccessful status.
//! [`OptimizeResult::require_success`] and its result-family counterparts convert
//! unsuccessful terminations to structured errors. One-shot functions require
//! success, except [`minimize_global`], which returns its result when differential
//! evolution completes its schedule (`PopulationStagnation` or `SearchBudgetCompleted`).
//! These schedule statuses make no convergence claim:
//!
//!
//! - `StepSmall`, `FunctionChangeSmall` and `Stagnation`: progress stopped
//!   before the stationarity test passed.
//! - `IterationLimit`, `EvaluationLimit`, `ProposalLimit`, `SampleLimit`,
//!   `NodeLimit`, `DepthLimit`, `MemoryLimit`, `TimeLimit` and `Cancelled`: a
//!   limit, a deadline or cancellation ended the solve.
//! - `LineSearchFailed`: the line search found no acceptable step.
//! - `SearchBudgetCompleted` and `PopulationStagnation`: a global heuristic
//!   used its budget or its population collapsed. Global heuristics never
//!   report success.
//! - `EstimatedStationarity` and `NoiseLimit`: a stochastic method reached a
//!   small sampled gradient, or could not resolve the noise.
//! - `Infeasible` and `Unbounded`: a linear, quadratic or integer program has
//!   no solution, confirmed by a checked certificate. `SuspectedInfeasible`
//!   and `SuspectedUnbounded` have no certificate.
//!
//! When a limit, cancellation or deadline stops a solve before the first point
//! is accepted, the solve returns an error instead
//! (see [Solve accounting](scioxide_core#solve-accounting)).
//!
//! Stopping thresholds are [`Tolerance`](scioxide_core::Tolerance) values of the
//! form `absolute + relative · scale`. Gradient, step, objective-change,
//! feasibility and complementarity tests have separate tolerances. Smooth local
//! solvers compare the Euclidean norm `‖∇f(x)‖₂` with the threshold at scale
//! `max(1, |f(x)| / max(1, ‖x‖∞))`; [`LbfgsB`] compares the infinity norm of
//! the projected gradient and reports its Euclidean norm. Nonlinear least squares minimizes
//! `½ Σ wᵢ rᵢ(x)²` and tests its gradient at unit scale.
//!
//! Solvers for nonlinear constraints (penalty, augmented Lagrangian, SQP,
//! interior point, COBYLA and COBYQA) normalize the problem once, at the
//! first point where they evaluate derivatives. The objective is divided by
//! `‖∇f(x₀)‖∞` and each constraint row `cᵢ` by `‖∇cᵢ(x₀)‖∞`. A zero row
//! gradient is replaced by `|cᵢ(x₀)|`, and a zero value by 1; a zero objective
//! gradient gives the scale 1 (`max(1, |f(x₀)|)` for SQP and interior point).
//! COBYLA and COBYQA take these gradients from their initial interpolation
//! models. The scales stay fixed during the solve. Stationarity, feasibility
//! and complementarity are tested on the normalized problem; the reported
//! diagnostics, and multipliers where a method reports them, are converted
//! back to the units of the original problem. [`OptimalityTolerances`] holds
//! these thresholds for constrained, programming and verification methods. LP, QP and MIP results
//! are successful only after the residuals of the original problem pass these
//! tests. A MIP stops when `incumbent − bound ≤ gap tolerance` at scale
//! `max(1, |incumbent|, |bound|)`; the bound is a floating-point estimate, not
//! a proof.
//!
//! Results also report iteration and evaluation counts in
//! [`SolveCounts`](scioxide_core::SolveCounts), including failed callback calls
//! and the work of nested solves, and one [`OptimizeDiagnostics`] variant with
//! details for the method family.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: invalid options, such as a nonpositive tolerance, radius
//!   or population size, or an empty or inconsistent box.
//! - `DimensionMismatch`: the starting point, bounds, output or workspace does
//!   not match the problem dimension.
//! - `NonFinite`: the starting point, a bound or a coefficient is NaN or
//!   infinite (origin `Input`), or a callback returned a non-finite value where
//!   the method needs a finite one (origin `Callback`), as described in
//!   [Problems and callbacks](#problems-and-callbacks).
//! - `ConvergenceFailure`, `ExecutionStopped`: the stopping test did not pass,
//!   or a limit, cancellation or deadline ended the solve. Configured methods
//!   return accepted partial results as described under [Termination](#termination).
//! - `DomainError`: a one-shot solve verified that a program is infeasible or
//!   unbounded below.
//! - `CapabilityUnavailable`: a time limit is set but the execution context has
//!   no clock.
//! - `ResourceLimit`, `AllocationFailure`, `SizeOverflow`: a workspace, a
//!   search tree or an archive exceeds its configured capacity or cannot be
//!   allocated.
//! - `NumericalInstability`, `Singular`: a factorization or model breaks down
//!   and the method cannot recover.
//!
//! Errors from your callbacks propagate unchanged. Each solver lists its own
//! conditions. On error, the output is unchanged as described in
//! [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing plans and workspaces
//!
//! One-shot functions take problem data and default settings. A method value
//! holds validated options and can solve many problems; its inherent `minimize`,
//! `solve` or `verify` allocates a workspace and returns a complete result.
//! [`Optimizer`] supports runtime selection of minimizers. [`QapPlan`] is the
//! only plan: it precomputes matrix copies and free-node blocks in `O(n²)` work
//! and storage. Owned forms run sequentially without a clock. For repeated solves, create a workspace once with
//! `workspace` (or `workspace_requirements` to size it first), then call
//! `minimize_into` or `solve_into` with an output you provide and an
//! [execution context](scioxide_core#execution-contexts) or a parent
//! [`SolveContext`](scioxide_core::SolveContext). See
//! [Plans and workspaces](scioxide_core#plans-and-workspaces). With a prepared
//! workspace, reserved output and callbacks that write in place, a prepared
//! call does not allocate.
//!
//! After a call, the workspace reports `last_attempt()` and, when a point was
//! accepted, `last_accepted()`. [`BfgsCheckpoint`], [`Nsga2Checkpoint`] and
//! [`StochasticCheckpoint`] save the state of an attempt so that `resume_into`
//! can continue it, and [`SimplexWarmStart`] restarts the simplex method from a
//! previous basis.
//!
//! ```
//! use scioxide_core::ExecutionContext;
//! use scioxide_optimize::{Bfgs, ObjectiveFn, OptimizeResult};
//!
//! let solver = Bfgs::<f64>::default();
//! let mut workspace = solver.workspace(2)?;
//! let mut result = OptimizeResult::<f64>::empty();
//! for shift in [1.0, 2.0, 3.0] {
//!     // f(x) = (x₀ − shift)² + x₁², minimized at (shift, 0)
//!     let f = ObjectiveFn::new(2, move |x: &[f64]| Ok((x[0] - shift).powi(2) + x[1] * x[1]));
//!     let cx = ExecutionContext::sequential();
//!     solver.minimize_into(&f, &[0.0, 1.0], &mut result, &mut workspace, cx)?;
//!     assert!((result.x()[0] - shift).abs() < 1e-4);
//! }
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Randomness and parallelism
//!
//! [`minimize_global`] takes its seed as an argument. Configured randomized
//! methods draw only from the seed in their options. Each start of
//! [`MultiStart`], [`Shgo`], [`BasinHopping`] and
//! [`DualAnnealing`] has its own stream derived from that seed;
//! [`DifferentialEvolution`] and [`CmaEs`] use one stream. Every random word
//! counts toward the sample limit. The same seed and options give the same
//! result.
//!
//! The execution policy of a global method is part of its options and defaults
//! to `Sequential`. Other policies evaluate independent starts or population
//! members in parallel and need a problem built with
//! [`BoundedObjective::new_sync`]. [`Direct`], [`BasinHopping`] and
//! [`DualAnnealing`] are sequential. [`Nsga2`] and the stochastic methods
//! run replicate batches under the policy in their options, with results that
//! do not depend on the number of workers. Linear, quadratic and integer
//! programming run sequentially, even when the execution context has an
//! executor.
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide crates this crate depends on,
//!   which provides the thread executor and clock of `scioxide-core`.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. The
//! API is the same.
//!
//! # References
//!
//! - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
//!   2006.
//! - R. P. Brent, *Algorithms for Minimization without Derivatives*,
//!   Prentice-Hall, 1973.
//! - J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and
//!   theory", in *Numerical Analysis*, Lecture Notes in Mathematics 630,
//!   Springer, 1978, 105–116.
//! - C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*,
//!   Prentice-Hall, 1974.
//! - M. J. D. Powell, "A direct search optimization method that models the
//!   objective and constraint functions by linear interpolation", in *Advances
//!   in Optimization and Numerical Analysis*, Kluwer, 1994, 51–67.
//! - V. Chvátal, *Linear Programming*, W. H. Freeman, 1983.
//! - A. Beck, *First-Order Methods in Optimization*, SIAM, 2017.
//!
//! References for individual methods are on each solver's page.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod attempt;
mod callbacks;
mod constrained;
mod constraints;
mod context;
mod dense;
mod derivatives;
mod geometry;
mod global;
mod globalization;
mod integer;
mod kkt;
mod least_squares;
mod line_search;
mod linear;
mod linear_least_squares;
mod one_shot;
mod options;
mod problem;
mod programming;
mod proximal;
mod quadratic;
mod quadratic_assignment;
mod quasi_newton;
mod random;
mod result;
mod scalar;
mod stats_adapters;
mod stats_transport;
mod storage;
mod traits;
mod unconstrained;
mod utils;
mod validation;
mod verification;
mod workflows;
// Reuse the oracle source without a dev-dependency cycle back to the consumer tool.

// Protocols, problems, callbacks and constraints
pub use callbacks::{ConstraintFn, LeastSquaresFn, ObjectiveFn};
pub use constraints::{
    ConstraintJacobianStats, ConstraintProfile, ConstraintValues, Constraints, ConstraintsWorkspace,
};
pub use derivatives::DerivativeWorkspace;
pub use geometry::FiniteBounds;
pub use problem::{
    BoundedObjective, CoefficientMatrix, ConstrainedProblem, HessianSymmetry,
    LinearConstraintSense, LinearConstraints, LpProblem, MipProblem, QpProblem,
};
pub use traits::{
    ConstraintFunction, LeastSquaresProblem, LineSearch, ObjectiveFunction, Optimizer,
};

// Shared results, termination, tolerances and attempt progress
pub use attempt::{MinimizerAccepted, MinimizerProgress};
pub use options::OptimalityTolerances;
pub use result::{OptimalityDiagnostics, OptimizeDiagnostics, OptimizeResult, OptimizeTermination};

// Local minimizers
pub use unconstrained::{
    Bfgs, BfgsCheckpoint, BfgsOptions, BfgsWorkspace, BfgsWorkspaceRequirements, GradientDescent,
    GradientDescentOptions, GradientDescentWorkspace, GradientDescentWorkspaceRequirements, Lbfgs,
    LbfgsB, LbfgsBOptions, LbfgsBWorkspace, LbfgsBWorkspaceRequirements, LbfgsOptions,
    LbfgsWorkspace, LbfgsWorkspaceRequirements, NelderMead, NelderMeadDiagnostics,
    NelderMeadGeometry, NelderMeadOptions, NelderMeadWorkspace, NelderMeadWorkspaceRequirements,
    NewtonCgTermination, NewtonDiagnostics, NewtonOptimizer, NewtonOptimizerWorkspace,
    NewtonOptimizerWorkspaceRequirements, NewtonOptions, NonlinearCg, NonlinearCgOptions,
    NonlinearCgWorkspace, NonlinearCgWorkspaceRequirements, Powell, PowellDiagnostics,
    PowellOptions, PowellWorkspace, PowellWorkspaceRequirements, TrustRegionOptimizer,
    TrustRegionOptimizerWorkspace, TrustRegionOptimizerWorkspaceRequirements, TrustRegionOptions,
    TrustRegionSubproblem, TrustRegionWorkspaceRequest,
};

// Line searches and curvature updates
pub use line_search::{
    BacktrackingLineSearch, LineSearchFailure, LineSearchResult, LineSearchScope,
    LineSearchTermination, StrongWolfeLineSearch,
};
pub use quasi_newton::{
    CgRestartReason, CurvatureAction, CurvaturePolicy, GlobalizationDiagnostics, HessianForm,
    HessianUpdate, HessianUpdateReport, HessianUpdateWorkspace, UpdateSkipReason,
};

// Scalar minimization
pub use scalar::{
    MinimumBracket, MinimumSample, ScalarMinimizer, ScalarMinimumMethod, ScalarMinimumOptions,
    ScalarMinimumResult, ScalarMinimumStart, ScalarObjective,
};

// Nonlinear and linear least squares
pub use least_squares::{
    ComplexLeastSquares, LeastSquares, LeastSquaresAccepted, LeastSquaresGlobalization,
    LeastSquaresLinearSolver, LeastSquaresLoss, LeastSquaresOptions, LeastSquaresProgress,
    LeastSquaresResult, LeastSquaresStopping, LeastSquaresWorkspace, LeastSquaresWorkspaceRequest,
    LeastSquaresWorkspaceRequirements,
};
pub use linear_least_squares::{
    BoundedLinearLeastSquares, BoundedLinearLeastSquaresWorkspace, LinearLeastSquaresActivity,
    LinearLeastSquaresKkt, LinearLeastSquaresOptions, LinearLeastSquaresProblem,
    LinearLeastSquaresResult, Nnls, NnlsWorkspace,
};

// Nonlinear constrained minimizers and KKT providers
pub use constrained::{
    AugmentedLagrangian, AugmentedLagrangianOptions, AugmentedLagrangianWorkspace,
    AugmentedLagrangianWorkspaceRequirements, Cobyla, CobylaDiagnostics, CobylaOptions,
    CobylaWorkspace, CobylaWorkspaceRequirements, Cobyqa, CobyqaDiagnostics, CobyqaOptions,
    CobyqaWorkspace, CobyqaWorkspaceRequirements, DenseSqpSubproblem, InteriorPoint,
    InteriorPointDiagnostics, InteriorPointOptions, InteriorPointState, InteriorPointWorkspace,
    InteriorPointWorkspaceRequirements, LagrangianHessian, PenaltyMethod, PenaltyMethodWorkspace,
    PenaltyMethodWorkspaceRequirements, PenaltyOptions, Sqp, SqpDiagnostics, SqpOptions, SqpState,
    SqpSubproblem, SqpSubproblemObjective, SqpSubproblemProvider, SqpSubproblemReport,
    SqpWorkspace, SqpWorkspaceRequirements,
};
pub use kkt::{DenseKktProvider, KktProvider, SparseKktProvider};

// Linear, quadratic and mixed-integer programming
pub use integer::{
    BranchAndBound, BranchAndBoundWorkspace, MipBranching, MipCutKind, MipLimits, MipNode,
    MipPresolveMap, MipRelaxationSolver, MipResult, MipSearchState, MipStatistics,
};
pub use linear::{
    BasisMatrix, BasisProvider, DenseBasisProvider, InteriorPointLp, InteriorPointLpWorkspace,
    MipSimplexWorkspace, SimplexSolver, SimplexWarmStart, SimplexWorkspace, SparseBasisProvider,
};
pub use programming::{ProgrammingAccepted, ProgrammingProgress, ProgrammingResult};
pub use quadratic::{QpMethod, QpSolver, QpWarmStart, QpWorkspace};

// Programming verification
pub use verification::{
    DualCertificate, LpVerificationProblem, OwnedDualCertificate, OwnedProgrammingCertificate,
    ProgrammingCertificate, ProgrammingClaim, ProgrammingVerifier, QpVerificationProblem,
    VerificationOptions, VerificationProblem, VerificationReport, VerificationStatus,
    VerificationWorkspace,
};

// Quadratic assignment
pub use quadratic_assignment::{
    QapDirection, QapMethod, QapOptions, QapPlan, QapProblemView, QapResult, QapWorkspace,
};

// Global search
pub use global::{
    BasinHopping, BasinHoppingOptions, BasinProposal, BorrowedStartDesign, CmaEs, CmaEsGeneration,
    CmaEsOptions, CmaEsRestartPolicy, CmaEsWorkspace, CmaEsWorkspaceRequirements, DeStrategy,
    DifferentialEvolution, DifferentialEvolutionBoundary, DifferentialEvolutionGeneration,
    DifferentialEvolutionOptions, DifferentialEvolutionWorkspace,
    DifferentialEvolutionWorkspaceRequirements, Direct, DirectOptions, DirectResult, DirectVariant,
    DirectWorkspace, DirectWorkspaceRequirements, DualAnnealing, DualAnnealingOptions,
    GaussianAcquisition, GlobalSearchOptions, GlobalSearchResult, GlobalSearchWorkspace,
    GlobalSearchWorkspaceRequirements, KrigingSurrogateFitter, LocalSolverFactory, MultiStart,
    PowellLocalFactory, PowellSearchLocal, RbfSurrogateFitter, SearchLocalSolver,
    SearchStartReport, Shgo, ShgoOptions, ShgoRefinement, ShgoResult, ShgoWorkspace,
    ShgoWorkspaceRequirements, StartDesign, SurrogateAcquisition, SurrogateFeasibility,
    SurrogateFitter, SurrogateHistory, SurrogateModel, SurrogateObservation, SurrogatePrediction,
    SurrogateSearch, SurrogateSearchOptions, SurrogateSearchReport, SurrogateSearchWorkspace,
    UniformBasinProposal, UniformStartDesign, UpdatingMode,
};
pub use random::SearchRandom;

// Composite and splitting methods
pub use proximal::{
    Admm, AdmmBlock, AdmmBlockUpdate, AdmmOptions, AdmmProblem, AdmmProgress, AdmmResult,
    AdmmStart, AdmmUpdateInput, AdmmUpdateReport, AdmmWorkspace, AdmmWorkspaceRequirements,
    CompositeProblem, Fista, ProximalGradient, ProximalMap, ProximalOptions, ProximalProgress,
    ProximalResult, ProximalWorkspace, ProximalWorkspaceRequirements, SmoothGradient,
};

// Multiobjective and sampled first-order workflows
pub use workflows::{
    DeterministicGradient, NoiseModel, Nsga2, Nsga2Checkpoint, Nsga2Options, Nsga2Problem,
    Nsga2Progress, Nsga2Result, Nsga2Workspace, Nsga2WorkspaceRequirements, OptimizationSampleKey,
    OptimizationSampler, ParetoArchive, ParetoDominance, ParetoEntry, PhiloxSampler,
    ReplicationOptions, SampledGradient, SampledObjective, SgdOptions, SgdStep,
    StochasticCheckpoint, StochasticGradientDescent, StochasticProblem, StochasticProgress,
    StochasticResult, StochasticTrustOptions, StochasticTrustRegion, StochasticWorkspace,
    StochasticWorkspaceRequirements, pareto_dominance,
};

// Statistical providers for scioxide-stats
pub use stats_adapters::{LbfgsBFitSolver, LeastSquaresFitSolver, SqpFitSolver};
pub use stats_transport::SimplexTransportSolver;

pub use one_shot::{
    bvls, curve_fit, least_squares, minimize, minimize_bounded, minimize_constrained,
    minimize_global, minimize_scalar, nnls, quadratic_assignment,
};
