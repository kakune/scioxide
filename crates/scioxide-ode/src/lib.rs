//! Solves ordinary, differential-algebraic, delay and stochastic differential equations.
//!
//! The crate integrates initial-value problems `y′ = f(t, y)` with explicit and
//! implicit Runge–Kutta, Rosenbrock-W, Adams and BDF methods, and switches
//! between nonstiff and stiff methods automatically. It also integrates index-1
//! DAEs `F(t, y, y′) = 0`, delay equations with constant delays, Itô and
//! Stratonovich SDEs and Hamiltonian systems with or without holonomic
//! constraints. Two-point boundary-value problems are solved by collocation.
//! Gradients of a scalar objective with respect to parameters come from
//! forward sensitivities or from discrete and continuous adjoints. States are
//! real or complex, and time is `f32` or `f64`.
//!
//! # Quick start
//!
//! ```
//! use scioxide_ode::solve_ivp;
//! let y = solve_ivp(|_t: f64, y: &[f64], dy: &mut [f64]| {
//!     dy[0] = -y[0]; Ok(())
//! }, 0.0, 1.0, &[1.0]).map_err(scioxide_core::SciOxideError::from)?;
//! assert!((y.as_slice()[0] - (-1.0_f64).exp()).abs() < 1e-3);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Entry point | Method |
//! |---|---|---|
//! | Nonstiff IVP, general use | [`solve_ivp`], [`solve_ivp_at`] | explicit Runge–Kutta 5(4), adaptive |
//! | Nonstiff IVP, tight tolerances | [`OdeMethod`] with [`DormandPrince853`], [`Verner65`] | explicit Runge–Kutta 8(5,3) and 6(5) |
//! | Nonstiff IVP, expensive `f` | [`OdeMethod`] with [`AdamsBashforth`], [`AdamsMoulton`] | variable-order Adams, orders 1–5 |
//! | Stiff IVP | [`OdeMethod`] with [`RadauIia`], [`Bdf`], [`RosenbrockW`] | Radau IIA order 5; BDF/NDF orders 1–5; linearly implicit W-method |
//! | Stiffness unknown or changing | [`OdeMethod`] with [`AutoStiff`], [`AdamsBdf`] | automatic switching between a nonstiff and a stiff method |
//! | Fixed step, for example inside a PDE time loop | [`OdeMethod`] with [`BackwardEuler`], [`ImplicitTrapezoid`], [`Sdirk`], [`ClassicalRk4`] | one-step methods without error control |
//! | Hamiltonian system, long times | [`solve_hamiltonian`], [`GaussRk`], [`Rattle`] | symplectic Gauss collocation; Störmer–Verlet; RATTLE for `g(q) = 0` |
//! | Index-1 DAE `F(t, y, y′) = 0` | [`DaeBdf`] | variable-step BDF, orders 1–5 |
//! | Constant delays `y′ = f(t, y(t), y(t − τ₁), …)` | [`solve_dde`], [`DdePlan`] | method of steps on Dormand–Prince 5(4) |
//! | SDE `dX = a dt + b dW` | [`solve_sde`], [`SdeMethod`] | Euler–Maruyama (Itô), stochastic Heun (Stratonovich) |
//! | Two-point BVP with unknown parameters | [`solve_bvp`], [`BvpPlan`] | three-stage Lobatto IIIA collocation, fourth order, with residual control |
//! | Gradient of an objective, few parameters | [`ForwardSensitivitySystem::bind`] | forward sensitivity equations |
//! | Gradient of an objective, many parameters | [`OdeAdjointSolver`], [`DaeAdjointSolver`] | checkpointed discrete or continuous adjoint |
//! | Many independent problems | [`solve_ensemble_into`] | runs each member with its own parameters |
//!
//! [`BuiltinOdeMethod`] selects any built-in IVP method at run time.
//!
//! # Methods
//!
//! | Method | Type | Order | Step size | Dense output |
//! |---|---|---:|---|---|
//! | [`ForwardEuler`] | explicit RK | 1 | fixed | — |
//! | [`ExplicitMidpoint`], [`Heun`] | explicit RK | 2 | fixed | — |
//! | [`ClassicalRk4`] | explicit RK | 4 | fixed | — |
//! | [`Rkf45`] | explicit RK | 5(4) | adaptive | cubic Hermite |
//! | [`DormandPrince5`] | explicit RK | 5(4) | adaptive | quartic |
//! | [`Verner65`] | explicit RK | 6(5) | adaptive | cubic Hermite |
//! | [`DormandPrince853`] | explicit RK | 8(5,3) | adaptive | order 7 |
//! | [`RosenbrockW`] | linearly implicit W | 2 or 3 | adaptive | cubic Hermite |
//! | [`BackwardEuler`] | implicit RK, L-stable | 1 | fixed | — |
//! | [`ImplicitTrapezoid`] | implicit RK, A-stable | 2 | fixed | — |
//! | [`Sdirk`] | singly diagonally implicit RK, L-stable | 3 | fixed | — |
//! | [`GaussRk`] | implicit RK, symplectic | 4 | fixed | — |
//! | [`RadauIia`] | implicit RK, L-stable | 5(3) | adaptive | collocation polynomial |
//! | [`AdamsBashforth`], [`AdamsMoulton`] | Adams, PECE | 1–5 | adaptive with variable order, or fixed | interpolating polynomial |
//! | [`Bdf`] | BDF or NDF | 1–5 | adaptive with variable order, or fixed | interpolating polynomial |
//! | [`AdamsBdf`], [`AutoStiff`] | automatic switching | of the active method | adaptive | of the active method |
//! | [`StormerVerlet`], [`Rattle`] | partitioned, symplectic | 2 | fixed | — |
//!
//! A method marked "—" has no continuous extension, so it does not support
//! events or requested output times. [`OdeMethod::linear_stability`] gives the
//! part of each axis of the complex plane where a method's stability function
//! satisfies `|R(z)| ≤ 1`.
//!
//! # Callbacks
//!
//! An [`OdeSystem`] writes `f(t, y)` into an output slice. The closure wrapper
//! [`OdeFn`] covers the common case. [`EventFn`], [`DaeFn`], [`SdeFn`],
//! [`BvpFn`] and [`HamiltonianFn`] adapt the other equation and event protocols. Optional trait methods supply a Jacobian,
//! a sparsity pattern, a mass matrix `M` for `M y′ = f(t, y)`, a time
//! derivative, a step limit such as a CFL bound, and [`OdeSystem::admits`],
//! which rejects a trial state outside the domain of `f`. A rejected trial
//! state halves the step, with every method. Callbacks receive the
//! [`SolveContext`](scioxide_core::SolveContext) of the running solve, so a
//! nested solve inside a callback counts toward the same budgets.
//!
//! Output that a callback writes is checked before it is used: a NaN or
//! infinity is a `NonFinite` error with origin `Callback`. An error returned by
//! a callback stops the integration and is returned unchanged.
//!
//! # Error control
//!
//! Adaptive methods estimate the local error `eᵢ` of each step and accept the
//! step when the normalized norm
//!
//! ```text
//! r = ‖ eᵢ / (atolᵢ + rtol · max(|yᵢ|, |ŷᵢ|)) ‖ ≤ 1,
//! ```
//!
//! where `y` and `ŷ` are the states at the start and end of the step and the
//! norm is the one chosen in [`ErrorTolerance`](scioxide_core::ErrorTolerance).
//! [`OdeOptions::new`] uses `atol = 10⁻⁶`, `rtol = 10⁻³` and the RMS norm. The
//! tolerances control the local error per step, not the global error at the
//! end time, which can be larger by a factor that depends on the stability of
//! the problem.
//!
//! [`OdeStepControl`] chooses the next step. Adaptive explicit Runge–Kutta
//! methods default to integral control that keeps the current step when
//! `0.5 < r ≤ 0.75`, so small fluctuations of the error estimate do not change
//! the step. [`RadauIia`] defaults to predictive (Gustafsson) control and the
//! other adaptive methods to PI control. A [`ProportionalIntegralController`]
//! sets your own PI parameters. Without an initial step in
//! [`OdeOptions`], the first step is estimated from `f(t₀, y₀)` and one
//! explicit Euler trial step (Hairer, Nørsett and Wanner, §II.4).
//!
//! # Implicit stages
//!
//! Implicit Runge–Kutta, BDF and Rosenbrock-W methods and [`DaeBdf`] solve a
//! linear or nonlinear system in every step. The Jacobian `∂f/∂y` comes from
//! a [`DerivativeStrategy`](scioxide_differentiate::DerivativeStrategy): by
//! default an analytic [`OdeSystem::jacobian`] if you supply one, then
//! automatic differentiation, then forward differences with steps
//! `√ε · max(|yⱼ|, atolⱼ)`. [`OdeStageBackend`] selects dense LU (the
//! default), sparse LU for a given sparsity pattern, or restarted GMRES on
//! Jacobian–vector products. The sparse and Krylov backends need an identity
//! mass matrix.
//!
//! Nonlinear stages use a simplified Newton iteration by default
//! ([`NewtonMode`]), which keeps one Jacobian and factorization while the time
//! and state move little and the iteration converges. With the contraction
//! rate `θ = ‖Δₖ‖ / ‖Δₖ₋₁‖` of successive corrections `Δₖ`, the iteration
//! diverges for `θ ≥ 1` and converges when `θ/(1 − θ) · ‖Δₖ‖ ≤ 1`, in a norm
//! scaled to a tenth of the error tolerance (Hairer and Wanner, §IV.8). An
//! adaptive method rejects a step whose stage solve fails or whose iteration
//! matrix is singular, and retries with half the step. A fixed-step method
//! returns the error. For
//! [`BackwardEuler`], [`ImplicitTrapezoid`], [`GaussRk`], [`Sdirk`] and
//! [`RadauIia`], a [`StageLinearSolver`] or [`NonlinearStageSolver`] that you
//! provide through [`OdeProblem::with_stage_solver`] replaces the linear
//! solves or the whole iteration, and receives tolerances and accounting
//! through [`StageAccuracy`].
//!
//! # Events and output
//!
//! An [`EventFunction`] `g(t, y)` defines an event at each zero of `g` in the
//! chosen [`EventDirection`]. Events are located on the method's continuous
//! extension by the bracketing method of Alefeld, Potra and Shi, to a time
//! tolerance of `4ε` by default ([`OdeEventOptions`]). A terminal event stops
//! the integration at the located time. A zero of `g` at the start time is not
//! an event, so an integration can restart from an event.
//!
//! [`OdeOutputSchedule`] selects the output: the final state, every accepted
//! step, or interpolated states at requested times. Output goes to an
//! [`OdeOutputSink`] that you provide; [`OdeTrajectory`] is a sink that stores
//! everything in memory, and `()` discards it.
//!
//! # Termination
//!
//! One-shot functions return an error for every unsuccessful termination, with
//! the last accepted progress for time evolution.
//!
//! Each family has a termination enum, such as [`OdeTermination`], with the
//! variants `ReachedEnd`, `TerminalEvent`, `BudgetLimit`, `StepSizeLimit` and
//! `ExecutionStopped`, plus family-specific ones. When a step or evaluation
//! budget, the minimum step size, cancellation or a deadline stops an
//! integration after at least one accepted point, `advance` returns `Ok` with
//! an unsuccessful report and the state holds the last accepted point. Check
//! [`OdeSolveReport::is_success`]. Any other failure returns the error, the
//! state again holds the last accepted point, and `last_attempt` on the
//! workspace records the progress. The `solve` forms return that progress in
//! a [`PartialFailure`](scioxide_core::PartialFailure). Output already written
//! to the sink stays there. See
//! [Solve accounting](scioxide_core#solve-accounting).
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: invalid options, such as a nonpositive step bound or a
//!   tolerance vector of the wrong length; a method that does not support a
//!   requested feature, such as events or requested output times without
//!   dense output, or a step controller with a fixed-step method
//!   (`UnsupportedConfiguration`); a start time equal to the end time.
//! - `NonFinite`: a NaN or infinite time, initial state or option (origin
//!   `Input`), or callback output (origin `Callback`).
//! - `DimensionMismatch`: the system, the state or an output buffer does not
//!   have the problem's dimension.
//! - `ConvergenceFailure`, `Singular`, `NumericalInstability`: a stage solve
//!   of a fixed-step method failed, or a DAE initialization or BVP Newton
//!   iteration did not converge. An adaptive method instead reduces the step,
//!   and ends with `StepSizeLimit` when it cannot reduce it further.
//! - `ResourceLimit`, `AllocationFailure`, `SizeOverflow`: a workspace, history
//!   or checkpoint store exceeds its configured limit or cannot be allocated.
//!
//! Every check on the inputs runs before the first callback. A rejected call
//! leaves the state unchanged. The `solve_into` forms of [`BvpPlan`] and the
//! adjoint solvers follow [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing workspaces
//!
//! Method values take options on each call. Allocate an IVP workspace with
//! `method.workspace(&problem, &options)`, then call
//! `method.advance(&problem, &options, &mut state, &mut sink, &mut workspace, scope)`.
//! The workspace can serve different options and problems. [`DdePlan`] precomputes
//! a discontinuity schedule; [`BvpPlan::with_singular`] precomputes pseudo-inverses.
//! With sufficient storage and nonallocating callbacks and sinks, `advance`
//! does not allocate. Owned `solve` forms allocate storage and run sequentially.
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
//! - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
//!   Equations I: Nonstiff Problems*, 2nd ed., Springer, 1993.
//! - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems*, 2nd ed., Springer, 1996.
//! - E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical Integration*,
//!   2nd ed., Springer, 2006.
//! - K. E. Brenan, S. L. Campbell and L. R. Petzold, *Numerical Solution of
//!   Initial-Value Problems in Differential-Algebraic Equations*, SIAM, 1996.
//! - A. Bellen and M. Zennaro, *Numerical Methods for Delay Differential
//!   Equations*, Oxford University Press, 2003.
//! - P. E. Kloeden and E. Platen, *Numerical Solution of Stochastic
//!   Differential Equations*, Springer, 1992.
//! - U. M. Ascher, R. M. M. Mattheij and R. D. Russell, *Numerical Solution of
//!   Boundary Value Problems for Ordinary Differential Equations*, SIAM, 1995.
//! - Y. Cao, S. Li, L. Petzold and R. Serban, "Adjoint sensitivity analysis for
//!   differential-algebraic equations", *SIAM J. Sci. Comput.* 24(3),
//!   1076–1089, 2003.
//! - G. E. Alefeld, F. A. Potra and Y. Shi, "Algorithm 748: Enclosing zeros of
//!   continuous functions", *ACM Trans. Math. Softw.* 21(3), 327–344, 1995.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unreachable_pub)]

#[cfg_attr(test, macro_use)]
extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod adjoint;
mod automatic;
mod bvp;
mod callbacks;
mod dae;
mod dde;
mod dense_output;
mod ensemble;
mod explicit_rk;
mod implicit_rk;
mod ivp;
mod mechanical;
mod multistep;
mod result;
mod rosenbrock;
mod safety;
mod sde;
mod sensitivity;
mod shell;
mod step_control;
mod stepper;
mod stiff;
mod termination;
mod traits;
mod workspace;

pub use callbacks::{
    BvpFn, DaeFn, EventFn, HamiltonianFn, JacobianCallback, NoJacobian, OdeFn, SdeFn,
};
pub use dde::{
    DdeHistory, DdeHistoryBreak, DdeOptions, DdePlan, DdeProblem, DdeProgress, DdeReport,
    DdeRequirements, DdeRhs, DdeSolution, DdeTermination, DdeWorkspace, DdeWorkspaceRequest,
    solve_dde,
};
pub use explicit_rk::{
    ClassicalRk4, DormandPrince5, DormandPrince853, ExplicitMidpoint, ForwardEuler, Heun, Rkf45,
    Verner65,
};
pub use implicit_rk::{BackwardEuler, GaussRk, ImplicitTrapezoid, RadauIia, Sdirk};
pub use ivp::{
    BuiltinOdeMethod, EventInfo, OdeEvent, OdeOptions, OdeOutputSchedule, OdeOutputSink,
    OdeProblem, OdeProgress, OdeRequirements, OdeSolution, OdeSolveReport, OdeState,
    OdeStepControl, OdeTermination, OdeTrajectory, OdeWorkspace, solve_ivp, solve_ivp_at,
};
pub use multistep::{AdamsBashforth, AdamsMoulton, Bdf, BdfFormula, MultistepOrder};
pub use result::{
    EventDirection, EventEvidence, EventOccurrence, EventSearchEvidence, EventSearchPolicy,
    OdeEventOptions,
};
pub use step_control::ProportionalIntegralController;
pub use stepper::{OdeLinearStability, OdeMethod};
pub use traits::{
    EventFunction, NonlinearStageReport, NonlinearStageSolver, OdeRhsStepLimit, OdeSystem,
};

pub use stiff::NewtonMode;
pub use stiff::accuracy::StageAccuracy;
pub use stiff::linear::OdeStageBackend;
pub use stiff::stage::OdeStageSolver;
pub use traits::StageLinearSolver;

pub use mechanical::{
    HolonomicConstraints, Rattle, RattleOptions, RattleProblem, RattleRequirements, RattleWorkspace,
};
pub use mechanical::{
    MechanicalOptions, MechanicalOutputSink, MechanicalReport, MechanicalSolution,
    MechanicalTermination, PhaseSpace, SeparableHamiltonian, StormerVerlet,
    StormerVerletRequirements, StormerVerletWorkspace, solve_hamiltonian,
};

pub use rosenbrock::{RosenbrockMethod, RosenbrockW};

pub use adjoint::{
    AdjointErrorReport, AdjointMode, AdjointObjective, AdjointObjectivePart, AdjointObjectiveValue,
    AdjointOptions, AdjointReport, AdjointResult, AdjointTerminalEvent, AdjointTermination,
    CheckpointPolicy, DaeAdjointOptions, DaeAdjointProblem, DaeAdjointRequirements,
    DaeAdjointSolver, DaeAdjointSystem, DaeAdjointWorkspace, OdeAdjointOptions, OdeAdjointProblem,
    OdeAdjointRequirements, OdeAdjointResidual, OdeAdjointSolver, OdeAdjointSystem,
    OdeAdjointWorkspace,
};
pub use sensitivity::{
    EventIsolation, EventSensitivityInput, EventSensitivityStatus, ForwardSensitivityRequirements,
    ForwardSensitivitySystem, ForwardSensitivityWorkspace, OdeSensitivitySystem,
};

pub use ensemble::{
    EnsembleOutcome, EnsembleParameters, EnsembleReport, EnsembleRequirements, EnsembleWorkspace,
    solve_ensemble_into,
};

pub use automatic::{
    AdamsBdf, AutoStiff, OdeMethodSwitch, OdeSwitchReason, StiffnessSwitchOptions,
};

pub use sde::{
    SdeInterpretation, SdeMethod, SdeOptions, SdeProblem, SdeReport, SdeRequirements, SdeSolution,
    SdeSystem, SdeTermination, SdeWorkspace, solve_sde,
};

pub use dae::{
    DaeBdf, DaeConsistencyReport, DaeInitialization, DaeNumericalOptions, DaeOptions,
    DaeOutputSink, DaePartition, DaeProblem, DaeReport, DaeRequirements, DaeSolution, DaeState,
    DaeSystem, DaeTermination, DaeWorkspace,
};

pub use bvp::{
    BvpBackend, BvpJacobian, BvpNewtonOptions, BvpOptions, BvpPlan, BvpProblem, BvpReport,
    BvpRequirements, BvpSolution, BvpSystem, BvpTermination, BvpWorkspace, solve_bvp,
};
