//! Root finding: scalar equations, nonlinear systems, fixed points, solution
//! curves, polynomials and zeros of special functions.
//!
//! The crate solves `f(x) = 0` for a real or complex scalar `x`, and `F(x) = 0`
//! for a vector `x` of `n` unknowns and `n` equations. Scalar methods include bracketing methods
//! that keep a sign change and open methods that start from one or two points.
//! Systems are solved with Newton, quasi-Newton, trust-region, fixed-point and
//! Jacobian-free Newton–Krylov methods. Continuation follows a solution curve as
//! a parameter changes, and deflation finds further roots of the same system.
//! The crate also computes all roots of a polynomial and indexed zeros of Bessel,
//! Airy, Kelvin, error and Fresnel functions.
//!
//! # Quick start
//!
//! Each family has a one-shot form that takes the problem and returns the
//! answer with default settings:
//!
//! ```
//! use scioxide_rootfind::{SpecialZeroFamily, find_root, polynomial_roots, solve_system, special_zeros};
//!
//! // Wallis' cubic x³ − 2x − 5 = 0 has one real root, x = 2.0945514815423265…
//! let x = find_root(|x: f64| Ok(x * x * x - 2.0 * x - 5.0), 2.0, 3.0)?;
//! assert!((x - 2.094_551_481_542_326_5).abs() < 1e-12);
//! // x² = 2 as a system, x² − 1 = 0 by its ascending coefficients, and j₀,₁, j₀,₂, j₀,₃.
//! let square = |x: &[f64], out: &mut [f64]| {
//!     out[0] = x[0] * x[0] - 2.0;
//!     Ok(())
//! };
//! assert!((solve_system(square, &[1.0])?[0] - 2.0_f64.sqrt()).abs() < 1e-7);
//! assert!((polynomial_roots(&[-1.0_f64, 0.0, 1.0])?[1].re - 1.0).abs() < 1e-14);
//! let zeros = special_zeros(SpecialZeroFamily::BesselJ(0.0_f64), 3)?;
//! assert!((zeros[0] - 2.404_825_557_695_773).abs() < 1e-14);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! The configured forms in the table below take options, report the
//! termination and work counts, and reuse workspaces.
//!
//! # Choosing a method
//!
//! | Problem | Entry points | Method |
//! |---|---|---|
//! | Real root with a known sign change on `[a, b]` | [`find_root`], [`Brent`], [`Toms748`], [`Chandrupatla`], [`Brenth`] | Brent's method; Alefeld–Potra–Shi Algorithm 748; Chandrupatla's method; Brent's method with hyperbolic extrapolation |
//! | Same, with simple guaranteed steps | [`Bisection`], [`Ridder`], [`Illinois`], [`RegulaFalsi`] | bisection; Ridders' method; Illinois and plain false position |
//! | No bracket yet | [`find_bracket`], [`find_bracket_with_options`] | expands one or two starting points until the function changes sign |
//! | Real root, derivatives available, bracket known | [`SafeguardedNewton`], [`SafeguardedHalley`] | Newton or Halley steps that fall back to bisection inside the bracket |
//! | Real root from a starting point | [`Newton`], [`Halley`], [`Secant`], [`Muller`] | Newton–Raphson; Halley; secant; Muller's method |
//! | Complex root | [`ComplexNewton`], [`ComplexHalley`], [`ComplexSecant`], [`ComplexMuller`] | the same open methods in complex arithmetic |
//! | Scalar method chosen at run time | [`ScalarRootMethod`] | any of the scalar methods above |
//! | Square nonlinear system, Jacobian available or by differences | [`NewtonSystem`] | Newton's method with LU and an optional backtracking line search |
//! | Same, with an expensive Jacobian | [`Broyden`] | Broyden's first or second update ([`BroydenUpdate`]) |
//! | Square nonlinear system, from any starting point | [`solve_system`], [`TrustRegionSystem`] | Powell's hybrid dogleg trust-region method |
//! | Fixed point `x = G(x)` | [`fixed_point()`], [`FixedPointSolver`] | Anderson acceleration of depth 5 for [`fixed_point()`]; damped Picard, Steffensen, Anderson acceleration, Broyden-type and diagonal mixing ([`FixedPointMethod`]) |
//! | Large system with Jacobian–vector products only | [`NewtonKrylovSolver`] | inexact Newton with restarted GMRES and Eisenstat–Walker forcing |
//! | Solution curve of `F(x, λ) = 0` | [`Continuation`], [`LinearHomotopy`] | natural-parameter or pseudo-arclength predictor–corrector continuation |
//! | Further roots of a system | [`DeflationSet`], [`DeflatedProblem`] | shifted norm deflation of known roots |
//! | All roots of a polynomial | [`polynomial_roots`], [`polynomial_roots_with_options`] | closed forms, companion-matrix eigenvalues, Aberth–Ehrlich, Durand–Kerner or Laguerre ([`PolynomialRootMethod`]) |
//! | `k`-th zero of a special function | [`special_zeros()`], [`special_zeros_with_options`] | asymptotic and tabulated starting values refined by Newton's method |
//!
//! For a real scalar root, prefer a bracketing method: it keeps a sign change
//! of `f` at every step, so it cannot diverge. [`find_root`] uses [`Brent`]. Open methods
//! converge faster near a simple root but can diverge from a poor start.
//! [`PolynomialRootMethod::Auto`] uses closed forms through degree two,
//! companion-matrix eigenvalues through degree 64 and the Aberth–Ehrlich
//! iteration above. [`solve_system`] uses [`TrustRegionSystem`]: its trust
//! region converges from starting points where Newton's method without a line
//! search diverges, and it reuses a difference Jacobian across iterations.
//!
//! # Problems and callbacks
//!
//! A scalar equation implements [`ScalarFunction`] or [`ComplexScalarFunction`];
//! [`ScalarFn`], [`ScalarFnWithDerivative`] and [`ScalarFnWithDerivatives`] wrap
//! closures. A system implements [`ResidualOperator`](scioxide_core::ResidualOperator)
//! from `scioxide-core`, with [`DenseJacobian`] when a dense Jacobian is available.
//! [`ResidualFn`] wraps a residual closure, with an optional dense Jacobian
//! closure ([`ResidualFn::with_jacobian`]) or a Jacobian–vector product closure
//! ([`ResidualFn::with_jvp`]) for Newton–Krylov, deflation and continuation.
//! [`ResidualFn::with_parameter`] wraps `F(x, λ)` with the parameter `λ` as its
//! last input. The one-shot forms take the closures themselves.
//! Complex callbacks are never derived from real ones: you supply the complex
//! extension of the function.
//!
//! An error returned by a callback stops the solve and is returned unchanged. A
//! NaN or infinite callback value is a `NonFinite` error with origin `Callback`.
//!
//! # Termination
//!
//! Each result reports a [`RootTermination`] through `termination()`.
//! [`RootTermination::is_success`] is true for `ResidualTolerance`,
//! `StepTolerance`, `BracketTolerance`, `Complete` and `NoZeros`. Any other
//! status means that the returned point is the best accepted iterate, not a
//! verified root:
//!
//! - `IterationLimit`, `WorkLimit`, `Cancelled` and `TimeLimit`: an iteration
//!   limit, an evaluation or work budget, cancellation or a deadline ended the
//!   solve.
//! - `Stagnation`: the iterates stopped making progress before a stopping test
//!   passed.
//! - `GlobalizationFailure`: the line search or trust region found no acceptable
//!   step.
//! - `LinearFailure`: a nested linear solve, such as GMRES, failed.
//! - `Advanced`, `HistoryLimit`, `StepLimit` and `TangentFailure`: continuation
//!   states. `Advanced` means that one point was accepted and more of the path
//!   remains.
//! - `Unverified` and `NumericalFailure`: a polynomial or special-function root
//!   could not be verified or computed.
//!
//! When a limit, cancellation or deadline stops a solve before it accepts a
//! point, the solve returns an error instead (see
//! [Solve accounting](scioxide_core#solve-accounting)).
//!
//! The default stopping tests are scale-invariant. A bracketing method stops
//! when the bracket width is at most `4ε + 4ε·max(|a|, |b|)`, where `ε` is the
//! machine epsilon of the scalar type, or when it evaluates an exact zero. An
//! open scalar method needs both a step that small and a residual reduced by
//! `√ε` relative to the largest initial sample. A system solver needs
//! `‖F(x)‖ ≤ √ε · ‖F(x₀)‖`, with no absolute tolerance; a small step alone does
//! not count as success. [`ScalarRootOptions`] and the system options let you
//! set other tolerances.
//!
//! Results also report `point()`, `residual()`, `residual_norm()`,
//! `residual_report()` and `iterations()`. `counts()` holds this solve's own
//! callback evaluations, and `accounting()` adds the work of nested solves.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: invalid options, bracket endpoints whose function values
//!   have the same sign, a starting point outside its bracket, or a method that
//!   needs an analytic derivative the function does not provide.
//! - `NonFinite`: a NaN or infinite starting point, bracket endpoint or
//!   coefficient (origin `Input`), or callback value (origin `Callback`).
//! - `DimensionMismatch`: the starting point, output or workspace does not match
//!   the dimension of the problem.
//! - `ConvergenceFailure`, `ExecutionStopped`: a limit, cancellation or deadline
//!   stopped a solve before it accepted a point, or [`find_bracket`] reached the
//!   end of its domain or its expansion limit without a sign change.
//! - `Singular`: Newton or Broyden without a line search met a singular
//!   Jacobian.
//! - `NumericalInstability`: a line search, a trust-region model or the
//!   refinement of a special-function zero broke down.
//! - `ResourceLimit`, `SizeOverflow`, `AllocationFailure`: a workspace or
//!   history exceeds its configured memory limit or cannot be allocated.
//!
//! A one-shot form ([`find_root`], [`find_bracket`], [`solve_system`],
//! [`fixed_point()`], [`polynomial_roots`], [`special_zeros()`]) returns its
//! answer only after a successful termination. It reports any other termination
//! as `ConvergenceFailure` (iteration and work limits keep their budget reason,
//! other statuses are `Stagnation`), `ExecutionStopped` or
//! `NumericalInstability`, and its `InvalidInput` errors name only its own
//! parameters.
//!
//! Each item lists its own conditions. On error, the output is unchanged as
//! described in [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing workspaces
//!
//! Scalar solvers need no workspace: call `solve`, or `solve_scoped` to run
//! inside a parent solve or an [execution context](scioxide_core#execution-contexts).
//! System, fixed-point, Newton–Krylov and continuation solvers are values that
//! hold validated options and compute nothing when they are constructed. Their
//! `solve` method ([`Continuation::follow`] for a solution curve) allocates a
//! workspace and returns an owned result. For repeated solves, create a
//! workspace with `workspace` (or size it first with `workspace_requirements`)
//! and call `solve_into` with an output you provide. Polynomial roots and
//! special-function zeros have no solver value: [`polynomial_roots_into`] and
//! [`special_zeros_into`] take a workspace from
//! [`PolynomialRootWorkspace::workspace`]`(degree, &options)` and
//! [`SpecialZeroWorkspace::workspace`]`(count, &options)`. See
//! [Plans and workspaces](scioxide_core#plans-and-workspaces). With a prepared
//! workspace and reserved output, a `solve_into` call does not allocate unless
//! your callbacks do.
//!
//! ```
//! use scioxide_core::ExecutionContext;
//! use scioxide_rootfind::{
//!     NewtonSystem, NewtonSystemOptions, ResidualFn, SystemRootResult, SystemRootSolver,
//! };
//!
//! // x² − 2 = 0 has the positive root √2.
//! let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
//!     out[0] = x[0] * x[0] - 2.0;
//!     Ok(())
//! })
//! .with_jacobian(|x: &[f64], out: &mut [f64]| {
//!     out[0] = 2.0 * x[0];
//!     Ok(())
//! });
//! let solver = NewtonSystem::new(NewtonSystemOptions::default())?;
//! let mut workspace = solver.workspace(&f)?;
//! let mut output = SystemRootResult::empty();
//! output.try_reserve(1)?;
//! for x0 in [1.0, 3.0] {
//!     let cx = ExecutionContext::sequential_native();
//!     solver.solve_into(&f, &[x0], &mut output, &mut workspace, cx)?;
//!     assert!(output.is_success());
//!     assert!((output.point()[0] - 2.0_f64.sqrt()).abs() < 1e-7);
//! }
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! [`ContinuationState`] and [`ContinuationCheckpoint`] keep the state of a
//! path between calls, so a continuation can be advanced one point at a time
//! with [`Continuation::advance`] and a prepared workspace, and resumed from a
//! saved checkpoint.
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide dependencies. Without it the
//!   crate is `no_std` and needs `alloc`; the API is the same.
//!
//! # References
//!
//! - R. P. Brent, *Algorithms for Minimization without Derivatives*,
//!   Prentice-Hall, 1973, chapter 4.
//! - G. E. Alefeld, F. A. Potra and Y. Shi, "Algorithm 748: Enclosing zeros of
//!   continuous functions", *ACM Trans. Math. Softw.* 21(3), 1995.
//! - J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained
//!   Optimization and Nonlinear Equations*, SIAM, 1996.
//! - M. J. D. Powell, "A hybrid method for nonlinear equations", in
//!   P. Rabinowitz (ed.), *Numerical Methods for Nonlinear Algebraic Equations*,
//!   Gordon and Breach, 1970.
//! - C. T. Kelley, *Solving Nonlinear Equations with Newton's Method*, SIAM, 2003.
//! - H. F. Walker and P. Ni, "Anderson acceleration for fixed-point
//!   iterations", *SIAM J. Numer. Anal.* 49(4), 2011.
//! - E. L. Allgower and K. Georg, *Introduction to Numerical Continuation
//!   Methods*, SIAM, 2003.
//! - P. E. Farrell, Á. Birkisson and S. W. Funke, "Deflation techniques for
//!   finding distinct solutions of nonlinear partial differential equations",
//!   *SIAM J. Sci. Comput.* 37(4), 2015.
//! - D. A. Bini, "Numerical computation of polynomial zeros by means of
//!   Aberth's method", *Numer. Algorithms* 13, 1996.
//! - NIST Digital Library of Mathematical Functions, chapters 7, 9 and 10.
//!   <https://dlmf.nist.gov/>

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod backend;

mod attempt;
mod bracketing;
mod context;
mod continuation;
mod derivative;
mod fixed_point;
mod initialization;
mod newton_krylov;
mod nonlinear_math;
mod polynomial;
mod residual;
mod result;
mod scalar;
mod special_zeros;
mod system;
mod termination;
mod traits;
mod workspace;

// Traits and closure adapters
pub use traits::{
    ComplexRootSolver, ComplexScalarFn, ComplexScalarFnWithDerivative,
    ComplexScalarFnWithDerivatives, ComplexScalarFunction, RootLineSearch, ScalarFn,
    ScalarFnWithDerivative, ScalarFnWithDerivatives, ScalarFunction, ScalarRootSolver,
    SystemRootSolver,
};

// Residual protocol
pub use residual::{DenseJacobian, ResidualFn};

// Result / option types
pub use initialization::{Bracket, BracketVerification};
pub use result::{ComplexRootResult, RootResult, RootTermination, SystemRootResult};

// Bracketing utility
pub use bracketing::{
    BracketExpansion, BracketSearchDirection, BracketSearchInput, BracketSearchOptions,
    find_bracket, find_bracket_scoped, find_bracket_with_options,
};

// Scalar solvers
pub use scalar::{
    Bisection, Brent, Brenth, Chandrupatla, ComplexHalley, ComplexMuller, ComplexNewton,
    ComplexSecant, Halley, Illinois, Muller, Newton, RegulaFalsi, Ridder, SafeguardedHalley,
    SafeguardedInitialGuess, SafeguardedNewton, ScalarRootInput, ScalarRootMethod,
    ScalarRootOptions, Secant, Toms748, find_root,
};

// System solvers
pub use system::{
    BacktrackingLineSearch, Broyden, BroydenOptions, BroydenUpdate, NewtonSystem,
    NewtonSystemOptions, SystemWorkspace, SystemWorkspaceRequirements, TrustRegionSystem,
    TrustRegionSystemOptions, solve_system,
};

// Polynomial solvers
pub use polynomial::{
    PolynomialRootAlgorithm, PolynomialRootDiagnostic, PolynomialRootInput, PolynomialRootMethod,
    PolynomialRootOptions, PolynomialRootReport, PolynomialRootWorkspace,
    PolynomialRootWorkspaceRequirements, PolynomialRoots, polynomial_roots, polynomial_roots_into,
    polynomial_roots_with_options,
};

pub use fixed_point::{
    FixedPointDiagnostics, FixedPointInput, FixedPointMethod, FixedPointOptions, FixedPointResult,
    FixedPointSolver, FixedPointWorkspace, FixedPointWorkspaceRequirements, HistoryRestart,
    fixed_point,
};

pub use newton_krylov::{
    NewtonKrylovForcing, NewtonKrylovInput, NewtonKrylovObservation, NewtonKrylovOptions,
    NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver, NewtonKrylovWorkspace,
    NewtonKrylovWorkspaceRequirements, NonlinearLinearSolver, NonlinearLinearization,
    NonlinearScaling,
};

pub use continuation::{
    Continuation, ContinuationCheckpoint, ContinuationDiagnostic, ContinuationDiagnosticProvider,
    ContinuationObservation, ContinuationOptions, ContinuationPath, ContinuationProblem,
    ContinuationReport, ContinuationState, ContinuationWorkspace,
    ContinuationWorkspaceRequirements, DeflatedProblem, DeflatedRootResult, DeflationOptions,
    DeflationSet, DeflationWorkspace, DeflationWorkspaceRequirements, LinearHomotopy,
    RealifiedProblem,
};

pub use special_zeros::{
    KelvinZeroFunction, SpecialZero, SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions,
    SpecialZeroReport, SpecialZeroWorkspace, SpecialZeroWorkspaceRequirements, SpecialZeros,
    special_zeros, special_zeros_into, special_zeros_with_options,
};

/// Builds an `InvalidInput` error for `argument` with `reason`.
pub(crate) fn invalid(
    argument: scioxide_core::Argument,
    reason: scioxide_core::InvalidInputReason,
    detail: &'static str,
) -> scioxide_core::SciOxideError {
    scioxide_core::SciOxideError::invalid_input(argument, reason, detail)
}
