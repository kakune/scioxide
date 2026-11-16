//! Checkpointed adjoint gradients of scalar objectives for ODEs (Dormand–Prince 5(4))
//! and residual index-1 DAEs (BDF, backward Euler and the implicit trapezoidal rule).
//!
//! [`OdeAdjointResidual`] writes an ODE as the residual `F(t, y, y′) = y′ − f(t, y)`
//! so that the DAE plans can differentiate implicit integrations of it.
//! [`AdjointOptions`] holds the settings both families share, and
//! [`OdeAdjointOptions`] and [`DaeAdjointOptions`] add the family-specific ones.
use crate::safety::invalid;
use crate::termination::admit;

use crate::safety::finite;
use crate::termination::{Stop, Terminal};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, EvaluationCounts, EvaluationRole, ExecutionStopReason,
    InvalidInputReason, NormSelection, NumericScalar, RealScalar, SciOxideError, SolveBudgets,
    SolveContext, SolveCounts, SolveIdentity, SolveScope,
};

mod execution;
pub use execution::{
    DaeAdjointProblem, DaeAdjointRequirements, DaeAdjointSolver, DaeAdjointWorkspace,
    OdeAdjointProblem, OdeAdjointRequirements, OdeAdjointSolver, OdeAdjointWorkspace,
};
mod configuration;
pub use configuration::{AdjointOptions, DaeAdjointOptions, OdeAdjointOptions};
mod checkpoint;
pub use checkpoint::CheckpointPolicy;
mod dae;
mod dae_linear;
mod ode;
mod residual;
mod reverse;
#[cfg(test)]
mod tests;
use dae::{DaeAdjointBuffers, DaeStorageRequirements, PreparedDaeAdjoint};
use ode::{OdeAdjointBuffers, OdeStorageRequirements, PreparedOdeAdjoint};
pub use residual::OdeAdjointResidual;

/// Selects whether an adjoint differentiates the numerical method or the differential
/// equation.
///
/// The discrete adjoint is the gradient of the computed objective with the step sizes
/// of the forward integration held fixed: it applies the transposed derivative of
/// every accepted step in reverse order. The continuous adjoint integrates
/// the adjoint differential equation backward in time with its own error control.
/// It approximates the gradient of the exact objective, and its error depends on the
/// adjoint tolerance as well as on the forward tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjointMode {
    /// Reverses the accepted forward steps on the forward step-size sequence.
    Discrete,
    /// Integrates the adjoint equation backward along the reconstructed forward
    /// trajectory.
    Continuous,
}

/// Accepted forward steps recorded by default.
const DEFAULT_TAPE_CAPACITY: usize = 1024;

// Numerical policies, checkpointing and resource limits of a prepared adjoint.
//
// Fields are private; setters are infallible and [`OdeAdjointSolver::new`] or
// [`DaeAdjointSolver::new`] validate the whole configuration. All error estimates are
// diagnostics, not certificates.
//
// The structural limits size and bound the prepared storage before any callback runs:
// [`Self::with_tape_capacity`] (accepted forward steps, default 1024),
// [`Self::with_max_replay_steps`] and [`Self::with_max_reverse_steps`] (unbounded by
// default) and [`Self::with_max_workspace_bytes`] (unbounded by default).
// [`Self::with_max_evaluations`] bounds the provider evaluations of one solve epoch.
#[derive(Debug, Clone)]
#[must_use]
pub(super) struct Policies<S: RealScalar> {
    mode: AdjointMode,
    initial_step: S,
    fixed_step: bool,
    interpolation_defect: bool,
    primal_error: ErrorTolerance<S>,
    adjoint_error: ErrorTolerance<S>,
    quadrature_error: ErrorTolerance<S>,
    replay_error: ErrorTolerance<S>,
    event_options: crate::OdeEventOptions<S>,
    checkpoint_policy: CheckpointPolicy,
    tape_capacity: usize,
    max_replay_steps: usize,
    max_reverse_steps: usize,
    max_workspace_bytes: usize,
    max_evaluations: Option<Budget>,
    max_steps: Option<Budget>,
}

// Structural limits admitted by plan construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Limits {
    tape_capacity: usize,
    max_replay_steps: usize,
    max_reverse_steps: usize,
    max_workspace_bytes: usize,
}

impl<S: RealScalar> Policies<S> {
    // Adaptive integration from a positive `initial_step` magnitude (the interval
    // supplies its direction), `sqrt(epsilon)` infinity-norm primal, adjoint and
    // quadrature tolerances, exact replay, the default [`crate::OdeEventOptions`]
    // event discovery and localization, [`CheckpointPolicy::StoreAll`], 1024 recorded
    // steps and no other limits.
    //
    // The plan constructors validate `initial_step`.
    //
    // # Errors
    // Propagates scalar/policy construction errors.
    pub(super) fn new(mode: AdjointMode, initial_step: S) -> Result<Self, SciOxideError> {
        let error =
            || ErrorTolerance::scalar(S::SQRT_EPSILON, S::SQRT_EPSILON, NormSelection::Infinity);
        Ok(Self {
            mode,
            initial_step,
            fixed_step: false,
            interpolation_defect: false,
            primal_error: error()?,
            adjoint_error: error()?,
            quadrature_error: error()?,
            replay_error: ErrorTolerance::scalar(S::ZERO, S::ZERO, NormSelection::Infinity)?,
            event_options: crate::OdeEventOptions::default(),
            checkpoint_policy: CheckpointPolicy::default(),
            tape_capacity: DEFAULT_TAPE_CAPACITY,
            max_replay_steps: usize::MAX,
            max_reverse_steps: usize::MAX,
            max_workspace_bytes: usize::MAX,
            max_evaluations: None,
            max_steps: None,
        })
    }

    // Replace the positive finite initial forward step magnitude.
    pub(super) fn with_initial_step(mut self, step: S) -> Self {
        self.initial_step = step;
        self
    }
    // Use the requested step magnitude for every forward step, subject to system limits,
    // observation boundaries and the final endpoint.
    pub(super) fn with_fixed_step(mut self, fixed: bool) -> Self {
        self.fixed_step = fixed;
        self
    }

    // DP5 forward component error policy. Residual DAE solvers use the separate
    // differential/algebraic policies of [`crate::DaeOptions`] instead.
    #[cfg(test)]
    pub(super) fn with_primal_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.primal_error = tolerance;
        self
    }
    // Continuous state-cotangent error policy.
    pub(super) fn with_adjoint_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.adjoint_error = tolerance;
        self
    }
    // Running-objective and continuous parameter-quadrature error policy.
    pub(super) fn with_quadrature_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.quadrature_error = tolerance;
        self
    }
    // Permitted discrepancy at retained replay boundaries; exact by default.
    pub(super) fn with_replay_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.replay_error = tolerance;
        self
    }
    // Discovery and localization policy of a terminal ODE event.

    // Checkpoint retention. [`CheckpointPolicy::MemoryBudget`] selects checkpoints so
    // that the complete workspace, not just checkpoint vectors, fits
    // [`Self::with_max_workspace_bytes`] within [`Self::with_max_replay_steps`].
    pub(super) fn with_checkpoint_policy(mut self, policy: CheckpointPolicy) -> Self {
        self.checkpoint_policy = policy;
        self
    }
    // Maximum accepted forward steps; sizes the recorded trajectory and must be positive.
    pub(super) fn with_tape_capacity(mut self, steps: usize) -> Self {
        self.tape_capacity = steps;
        self
    }
    // Maximum additional forward steps of checkpoint replay per solve.
    pub(super) fn with_max_replay_steps(mut self, steps: usize) -> Self {
        self.max_replay_steps = steps;
        self
    }
    // Maximum discrete reverse macro steps or attempted continuous substeps.
    pub(super) fn with_max_reverse_steps(mut self, steps: usize) -> Self {
        self.max_reverse_steps = steps;
        self
    }
    // Upper bound on all adjoint-managed numerical storage, including backend scratch.
    // Provider-owned storage, borrowed inputs/outputs and allocator overhead are excluded.
    // This is the sole byte budget of [`CheckpointPolicy::MemoryBudget`].
    pub(super) fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }
    // Bound the provider evaluations (dynamics, pullbacks, objectives, events and linear
    // solves) of one solve epoch, shared with nested consistency solves.
    pub(super) fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    // Discrete or continuous formulation.
    pub(super) const fn mode(&self) -> AdjointMode {
        self.mode
    }
    // Initial forward step magnitude.
    pub(super) const fn initial_step(&self) -> S {
        self.initial_step
    }
    // Whether every forward step uses the requested magnitude.
    pub(super) const fn fixed_step(&self) -> bool {
        self.fixed_step
    }
    // Continuous state-cotangent error policy.
    pub(super) const fn adjoint_error(&self) -> &ErrorTolerance<S> {
        &self.adjoint_error
    }
    // Running-objective and parameter-quadrature error policy.
    pub(super) const fn quadrature_error(&self) -> &ErrorTolerance<S> {
        &self.quadrature_error
    }
    // Permitted discrepancy at retained replay boundaries.
    pub(super) const fn replay_error(&self) -> &ErrorTolerance<S> {
        &self.replay_error
    }
    // Checkpoint retention policy.
    pub(super) const fn checkpoint_policy(&self) -> &CheckpointPolicy {
        &self.checkpoint_policy
    }
    // Maximum accepted forward steps.
    pub(super) const fn tape_capacity(&self) -> usize {
        self.tape_capacity
    }
    // Maximum checkpoint replay steps per solve.
    pub(super) const fn max_replay_steps(&self) -> usize {
        self.max_replay_steps
    }
    // Maximum attempted continuous reverse steps per solve.
    pub(super) const fn max_reverse_steps(&self) -> usize {
        self.max_reverse_steps
    }
    // Upper bound on adjoint-managed numerical storage.
    pub(super) const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
    // Provider-evaluation budget of one solve epoch.
    pub(super) const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    const fn limits(&self) -> Limits {
        Limits {
            tape_capacity: self.tape_capacity,
            max_replay_steps: self.max_replay_steps,
            max_reverse_steps: self.max_reverse_steps,
            max_workspace_bytes: self.max_workspace_bytes,
        }
    }

    // Local epoch allowance.
    fn budgets(&self) -> SolveBudgets {
        let budgets = match self.max_steps {
            Some(limit) => SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, limit),
            None => SolveBudgets::unbounded(),
        };
        match self.max_evaluations {
            Some(evaluations) => budgets.with_limit(BudgetKind::Evaluation, evaluations),
            None => budgets,
        }
    }

    fn validate(
        &self,
        n: usize,
        parameters: usize,
        restart: usize,
        primal: bool,
    ) -> Result<(), SciOxideError> {
        if !self.initial_step.is_finite() || self.initial_step <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options.initial_step"),
                "positive finite initial step required",
            ));
        }
        for i in 0..n {
            if primal {
                self.primal_error.threshold(i, n, S::ONE)?;
            }
            self.adjoint_error.threshold(i, n, S::ONE)?;
        }
        self.quadrature_error.threshold(0, 1, S::ONE)?;
        for i in 0..parameters {
            self.quadrature_error.threshold(i, parameters, S::ONE)?;
        }
        for i in 0..restart {
            self.replay_error.threshold(i, restart, S::ONE)?;
        }
        Ok(())
    }
}

/// Names the term of the objective that [`AdjointObjective::evaluate`] is asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjointObjectivePart {
    /// The terminal cost `Φ(t, y, p)`, evaluated at the final time.
    Terminal,
    /// The running cost `L(t, y, p)`, integrated over the solve interval in the
    /// direction of integration.
    Running,
}

/// Holds the value of one objective term and its partial derivative in time.
#[derive(Debug, Clone, Copy)]
pub struct AdjointObjectiveValue<S: RealScalar> {
    /// Finite value of the term.
    pub value: S,
    /// Partial derivative `∂/∂t` of the term, with state and parameters fixed.
    pub time_derivative: S,
}

/// Defines a real scalar objective and its first derivatives for an adjoint solve.
///
/// The objective of a solve on `[t₀, T]` is
///
/// ```text
/// J(y₀, p) = Φ(T, y(T), p) + ∫_{t₀}^{T} L(t, y(t), p) dt + Σₖ Gₖ(y(tₖ), p)
/// ```
///
/// where `y(t)` is the solution, `p` the real parameters, `Φ` the terminal cost, `L`
/// the running cost and `Gₖ` the costs at the observation times `tₖ`. When a terminal
/// event ends the integration, `T` is the event time.
///
/// For complex states the gradient with respect to `y` is the cotangent `ȳ` with
/// `dJ = Re(conj(ȳ) · dy)`. Parameters are real. Implementations must be
/// deterministic, write every element of each gradient output, and use no more
/// scratch than [`Self::workspace_len`] reports.
pub trait AdjointObjective<T: NumericScalar> {
    /// Returns the observation times `tₖ`, in nondecreasing order.
    ///
    /// Equal times are separate terms. Every time must lie in the solve interval, and
    /// the times are constants, not functions of `p`. Observation terms need
    /// [`AdjointMode::Discrete`] and a problem without a terminal event.
    fn observation_times(&self) -> &[T::Real] {
        &[]
    }
    /// Evaluates the observation term `Gₖ` with index `index` in
    /// [`Self::observation_times`] and writes its state and parameter gradients.
    ///
    /// The gradients include any direct dependence of the observation operator and its
    /// weights on `y` and `p`.
    ///
    /// # Errors
    ///
    /// The default implementation returns `InvalidInput`; implement this method
    /// whenever [`Self::observation_times`] is nonempty. Errors you return end the
    /// solve and are returned unchanged.
    #[allow(clippy::too_many_arguments)]
    fn observation(
        &self,
        _index: usize,
        _state: &[T],
        _parameters: &[T::Real],
        _state_gradient: &mut [T],
        _parameter_gradient: &mut [T::Real],
        _scratch: &mut [T],
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("self"),
            "observation derivatives are unavailable",
        ))
    }
    /// Returns the number of scratch scalars one evaluation needs.
    fn workspace_len(&self) -> usize {
        0
    }
    /// Evaluates the terminal or running cost at `(time, state, parameters)` and writes
    /// its gradients with respect to `state` and `parameters`.
    ///
    /// The call counts as one objective evaluation of the solve. Nested work you start
    /// should run in `context`; see [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged. A NaN or infinite
    /// gradient is a `NonFinite` error with origin `Callback`, and a non-finite value or
    /// time derivative is an `InvalidInput` error.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        part: AdjointObjectivePart,
        time: T::Real,
        state: &[T],
        parameters: &[T::Real],
        state_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError>;
}

/// Extends an [`OdeSystem`](crate::OdeSystem) `y′ = f(t, y, p)` with the transposed
/// Jacobian products an adjoint solve needs.
///
/// The right-hand side of the [`OdeSystem`](crate::OdeSystem) evaluates `f` at the
/// fixed parameters [`Self::parameters`]. The products can come from a dense or
/// sparse Jacobian, a matrix-free operator or reverse-mode automatic differentiation.
pub trait OdeAdjointSystem<T: NumericScalar>: crate::OdeSystem<T::Real, T> {
    /// Returns the real parameters `p` at which the right-hand side is evaluated.
    ///
    /// They must not change during a solve.
    fn parameters(&self) -> &[T::Real];
    /// Returns the number of real parameters.
    fn parameter_dimension(&self) -> usize {
        self.parameters().len()
    }

    /// Returns `true` if every callback returns the same output for the same input.
    ///
    /// Checkpointing evaluates the right-hand side again during the reverse sweep, so
    /// adjoint plans reject a system that returns `false`.
    fn is_replayable(&self) -> bool;
    /// Returns the number of scratch scalars one [`Self::pullback`] call needs.
    fn pullback_workspace_len(&self) -> usize {
        0
    }
    /// Writes the products `(∂f/∂y)ᴴ seed` to `state_output` and `(∂f/∂p)ᵀ seed` to
    /// `parameter_output`.
    ///
    /// For complex states, `ᴴ` is the adjoint of the real-linear map `dy ↦ (∂f/∂y) dy`
    /// under the inner product `Re(conj(u) · v)`. A right-hand side that is not
    /// holomorphic can supply that adjoint directly.
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged. A NaN or infinite
    /// output is a `NonFinite` error with origin `Callback`.
    #[allow(clippy::too_many_arguments)]
    fn pullback(
        &self,
        time: T::Real,
        state: &[T],
        seed: &[T],
        state_output: &mut [T],
        parameter_output: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Extends a [`DaeSystem`](crate::DaeSystem) `F(t, y, y′, p) = 0` with the
/// conjugate-transposed Jacobian products an adjoint solve needs.
///
/// The residual of the [`DaeSystem`](crate::DaeSystem) is evaluated at the fixed
/// parameters [`Self::parameters`]. For complex states the residual must be
/// holomorphic in `y` and `y′`, as for [`DaeSystem`](crate::DaeSystem).
pub trait DaeAdjointSystem<T: NumericScalar>: crate::DaeSystem<T> {
    /// Returns the real parameters `p` at which the residual is evaluated.
    ///
    /// They must not change during a solve.
    fn parameters(&self) -> &[T::Real];
    /// Returns the number of real parameters.
    fn parameter_dimension(&self) -> usize {
        self.parameters().len()
    }

    /// Returns `true` if every callback, including any linear solver or
    /// preconditioner you supply, returns the same output for the same input.
    ///
    /// Adjoint plans reject a system that returns `false`.
    fn is_replayable(&self) -> bool;
    /// Returns the number of scratch scalars one [`Self::pullback`] call needs.
    fn pullback_workspace_len(&self) -> usize {
        0
    }
    /// Writes `(∂F/∂y)ᴴ seed` to `state_output`, `(∂F/∂y′)ᴴ seed` to
    /// `derivative_output` and `(∂F/∂p)ᵀ seed` to `parameter_output`.
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged.
    #[allow(clippy::too_many_arguments)]
    fn pullback(
        &self,
        time: T::Real,
        state: &[T],
        derivative: &[T],
        seed: &[T],
        state_output: &mut [T],
        derivative_output: &mut [T],
        parameter_output: &mut [T::Real],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Returns a preconditioner for a transposed linear system of the reverse sweep, or
    /// `None` for no preconditioning.
    ///
    /// The system matrix is `Mᴴ`, where column `i` of `M` is
    /// `aᵢ ∂F/∂yᵢ + bᵢ ∂F/∂y′ᵢ` with `aᵢ = state_weights[i]` and
    /// `bᵢ = derivative_weights[i]`. A preconditioner of the forward integration is
    /// not used for these systems.
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged.
    fn adjoint_preconditioner(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _state_weights: &[T::Real],
        _derivative_weights: &[T::Real],
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<T>>, SciOxideError> {
        Ok(None)
    }
}

/// Defines a smooth event function `g(t, y, p)` whose first root ends an ODE adjoint
/// solve.
///
/// The final time `T` of the objective is then the root `g(T, y(T), p) = 0`, and the
/// gradient includes the dependence of `T` on `y₀` and `p`. The event has no reset
/// map. The root must be the only crossing in the interval; root finding on a
/// finite set of points cannot prove this. The gradient is unavailable, with
/// termination [`AdjointTermination::EventDerivativeUnavailable`], when the root is
/// at the initial time, when [`Self::unique_branch`] is `false`, or when the total
/// slope `|dg/dt| = |∂g/∂t + (∂g/∂y) f|` is not above
/// [`Self::transversality_threshold`]. Time and event values are real.
pub trait AdjointTerminalEvent<T: NumericScalar> {
    /// Returns the crossing direction to detect, in the direction of integration.
    fn direction(&self) -> crate::EventDirection {
        crate::EventDirection::Either
    }
    /// Returns `true` if you assert that the detected root is the only crossing of
    /// this event in the interval.
    fn unique_branch(&self) -> bool;
    /// Returns the positive lower bound on `|dg/dt|` at the root.
    fn transversality_threshold(&self) -> T::Real;
    /// Evaluates `g(time, state, p)`.
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged.
    fn value(&self, time: T::Real, state: &[T]) -> Result<T::Real, SciOxideError>;
    /// Writes `∂g/∂y` to `state_gradient` and `∂g/∂p` to `parameter_gradient`, and
    /// returns `∂g/∂t`.
    ///
    /// # Errors
    ///
    /// Errors you return end the solve and are returned unchanged.
    fn gradient(
        &self,
        time: T::Real,
        state: &[T],
        state_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
    ) -> Result<T::Real, SciOxideError>;
}

/// Says why an adjoint solve stopped.
///
/// An adjoint solve writes its gradients only when it succeeds, so a successful
/// [`AdjointReport`] always holds `ReachedEnd` or `TerminalEvent`. The other variants
/// classify the error of a failed attempt, in the report that
/// [`OdeAdjointWorkspace::last_attempt`] and [`DaeAdjointWorkspace::last_attempt`]
/// return. DAE adjoints have no events and never stop with `TerminalEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdjointTermination {
    /// The integration reached the end of the interval, and both gradients were
    /// computed.
    ReachedEnd,
    /// A transverse terminal event ended the integration, and both gradients were
    /// computed.
    TerminalEvent,
    /// A limit stopped the solve. The [`BudgetKind`] is `Proposal` for
    /// [`AdjointOptions::with_max_steps`], `Iteration` for the tape capacity
    /// ([`AdjointOptions::with_tape_capacity`]), the reverse-step limit
    /// ([`AdjointOptions::with_max_reverse_steps`]) or the Newton iterations of a DAE
    /// projection, and `Evaluation` for [`AdjointOptions::with_max_evaluations`] or
    /// the evaluation limit of an enclosing solve.
    BudgetLimit(BudgetKind),
    /// No representable forward or reverse step advances the time.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the solve.
    ExecutionStopped(ExecutionStopReason),
    /// Recomputing a step from a checkpoint did not reproduce the stored state within
    /// [`AdjointOptions::with_replay_error`], or the system parameters changed during
    /// the solve.
    ReplayMismatch,
    /// The terminal-event gradient does not exist: the root is at the initial time, it
    /// is not declared unique, or the event slope is below the transversality
    /// threshold.
    EventDerivativeUnavailable,
}

crate::termination::shared_termination!(AdjointTermination);

/// Lists the local error estimates and residuals of an adjoint solve.
///
/// These are diagnostics. None of them bounds the error of the gradient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjointErrorReport<S: RealScalar> {
    primal: Option<S>,
    adjoint: Option<S>,
    quadrature: Option<S>,
    replay: S,
    interpolation_defect: Option<S>,
    linear_residual: Option<S>,
}

impl<S: RealScalar> AdjointErrorReport<S> {
    /// Returns the largest normalized local error estimate of an accepted forward
    /// step, if the method has one.
    pub const fn primal(&self) -> Option<S> {
        self.primal
    }
    /// Returns the largest normalized local error estimate of an accepted step of the
    /// continuous adjoint equation.
    pub const fn adjoint(&self) -> Option<S> {
        self.adjoint
    }
    /// Returns the largest normalized local error estimate of the running-cost and
    /// parameter-gradient quadratures.
    pub const fn quadrature(&self) -> Option<S> {
        self.quadrature
    }
    /// Returns the largest absolute difference between a recomputed state and its
    /// stored checkpoint.
    pub const fn replay(&self) -> S {
        self.replay
    }
    /// Returns the largest sampled defect `|u′(t) − f(t, u(t))|` of the interpolant `u`
    /// of the forward trajectory, unscaled.
    ///
    /// It is reported only when
    /// [`OdeAdjointOptions::with_interpolation_defect_sampling`] is enabled.
    pub const fn interpolation_defect(&self) -> Option<S> {
        self.interpolation_defect
    }
    /// Returns the largest linear residual of a transposed linear system solved in
    /// the reverse sweep: the relative residual of a Krylov solve or the normwise
    /// backward error of a direct one (see [`DaeAdjointOptions::new`]).
    pub const fn linear_residual(&self) -> Option<S> {
        self.linear_residual
    }
}

/// Reports the termination, objective value, work and error estimates of one adjoint
/// solve.
///
/// A successful `solve_into` returns it. The workspace's `last_attempt` also holds it
/// as the progress of a failed attempt, and the owned `solve` returns it inside the
/// [`PartialFailure`](scioxide_core::PartialFailure). Step counts describe the original
/// forward integration; checkpoint recomputation and the reverse sweep have their own
/// counters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct AdjointReport<S: RealScalar> {
    termination: Option<AdjointTermination>,
    time: S,
    objective: Option<S>,
    accepted_steps: usize,
    rejected_steps: usize,
    trust: scioxide_core::PartialResultTrust,
    last_step_error_estimate: Option<S>,
    replayed_steps: usize,
    reverse_steps: usize,
    workspace_bytes: usize,
    counts: SolveCounts,
    errors: AdjointErrorReport<S>,
}

impl<S: RealScalar> AdjointReport<S> {
    /// Returns the report of a solve that has taken no step from `time`.
    const fn initial(time: S, workspace_bytes: usize) -> Self {
        Self {
            termination: None,
            time,
            objective: None,
            accepted_steps: 0,
            rejected_steps: 0,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            last_step_error_estimate: None,
            replayed_steps: 0,
            reverse_steps: 0,
            workspace_bytes,
            counts: SolveCounts::new(),
            errors: AdjointErrorReport {
                primal: None,
                adjoint: None,
                quadrature: None,
                replay: S::ZERO,
                interpolation_defect: None,
                linear_residual: None,
            },
        }
    }

    /// Returns why the solve stopped.
    ///
    /// It is `ReachedEnd` or `TerminalEvent` on success. In the report of a failed
    /// attempt it classifies a limit, step-size, cancellation, replay or event-gradient
    /// failure, and it is `None` for any other error.
    pub const fn termination(&self) -> Option<AdjointTermination> {
        self.termination
    }
    /// Returns `true` if both gradients were computed.
    pub const fn is_success(&self) -> bool {
        matches!(self.termination, Some(termination) if termination.is_success())
    }
    /// Returns the time of the last accepted forward step, or the initial time before
    /// the first one.
    pub const fn final_time(&self) -> S {
        self.time
    }
    /// Returns the objective value `J`, once the forward integration has finished.
    pub const fn objective(&self) -> Option<S> {
        self.objective
    }
    /// Returns the number of accepted steps of the forward integration.
    pub const fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }
    /// Returns the number of rejected steps of the forward integration, not counting
    /// recomputation or the reverse sweep.
    pub const fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }
    /// Returns how far the last accepted forward state can be trusted.
    pub const fn trust(&self) -> scioxide_core::PartialResultTrust {
        self.trust
    }
    /// Returns the normalized local error estimate of the last forward step, if the
    /// method has one.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
    /// Returns the number of forward steps recomputed from checkpoints.
    pub const fn replayed_steps(&self) -> usize {
        self.replayed_steps
    }
    /// Returns the number of recorded forward steps that the reverse sweep has
    /// completed.
    ///
    /// A continuous reverse step can consist of several attempted substeps; those count
    /// toward [`AdjointOptions::with_max_reverse_steps`] but not here.
    pub const fn reverse_steps(&self) -> usize {
        self.reverse_steps
    }
    /// Returns the upper bound, in bytes, on the numerical storage of the solve.
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    /// Returns the evaluations made by this solve itself, including recomputation and
    /// work on rejected steps.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this solve and of the solves nested in it.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the local error estimates and residuals.
    pub const fn errors(&self) -> AdjointErrorReport<S> {
        self.errors
    }
}

/// Holds the gradients and report returned by [`OdeAdjointSolver::solve`] and
/// [`DaeAdjointSolver::solve`].
#[derive(Debug, Clone)]
#[must_use]
pub struct AdjointResult<T: NumericScalar> {
    initial_gradient: Vec<T>,
    parameter_gradient: Vec<T::Real>,
    report: AdjointReport<T::Real>,
}

impl<T: NumericScalar> AdjointResult<T> {
    /// Returns the gradient `dJ/dy₀` with respect to the initial state.
    ///
    /// For a DAE it has one entry per differential coordinate, in the order of the
    /// [`DaePartition`](crate::DaePartition); the initial algebraic values are not
    /// independent variables.
    pub fn initial_gradient(&self) -> &[T] {
        &self.initial_gradient
    }
    /// Returns the gradient `dJ/dp` with respect to the real parameters.
    pub fn parameter_gradient(&self) -> &[T::Real] {
        &self.parameter_gradient
    }
    /// Returns the report of the solve.
    pub const fn report(&self) -> &AdjointReport<T::Real> {
        &self.report
    }
    /// Returns the initial-state gradient, the parameter gradient and the report.
    pub fn into_parts(self) -> (Vec<T>, Vec<T::Real>, AdjointReport<T::Real>) {
        (self.initial_gradient, self.parameter_gradient, self.report)
    }
}

fn convergence(reason: scioxide_core::ConvergenceReason, detail: &'static str) -> SciOxideError {
    SciOxideError::convergence_failure(reason, scioxide_core::ConvergenceDiagnostics::new(), detail)
}

fn scalar<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value)
}

/// Run one adjoint epoch of `scope` in `shell` and record its attempt.
///
/// `run` computes both gradients into workspace storage and returns whether a terminal
/// event ended the forward phase. A failure keeps the classification `run` recorded,
/// otherwise [`classify`]s its error; the report is the attempt's progress.
fn execute<S: RealScalar>(
    shell: &mut crate::shell::Shell<AdjointReport<S>>,
    scope: SolveScope<'_, '_>,
    identity: SolveIdentity,
    budgets: SolveBudgets,
    mut report: AdjointReport<S>,
    run: impl FnOnce(&mut AdjointReport<S>, &mut SolveContext<'_, '_>) -> Result<bool, SciOxideError>,
) -> Result<AdjointReport<S>, SciOxideError> {
    let executed = shell.execute(scope, identity, budgets, |cx| {
        let result = run(&mut report, cx);
        report.counts = cx.counts();
        match result {
            Ok(event) => {
                report.termination = Some(if event {
                    AdjointTermination::TerminalEvent
                } else {
                    AdjointTermination::ReachedEnd
                });
                Ok(report)
            }
            Err(error) => {
                if report.termination.is_none() {
                    report.termination = classify(&error);
                }
                Err(error)
            }
        }
    });
    shell.conclude(executed, AdjointReport::is_success, |_| Some(report))
}

/// The classification of an error that `run` left unclassified: changed replay
/// (including changed bound parameters) or a shared stop.
fn classify(error: &SciOxideError) -> Option<AdjointTermination> {
    match error.view() {
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::ReplayMismatch,
            ..
        } => Some(AdjointTermination::ReplayMismatch),
        _ => crate::termination::stop(error).map(AdjointTermination::from_stop),
    }
}

/// Owned convenience failure with the last attempt's progress.
fn partial<S: RealScalar>(
    error: SciOxideError,
    shell: crate::shell::Shell<AdjointReport<S>>,
) -> scioxide_core::PartialFailure<AdjointReport<S>> {
    shell.into_failure(error)
}

/// Zero-filled owned gradient outputs.
fn gradients<T: NumericScalar>(
    initial: usize,
    parameters: usize,
) -> Result<(Vec<T>, Vec<T::Real>), SciOxideError> {
    Ok((
        scioxide_core::backend::try_filled(
            initial,
            T::ZERO,
            scioxide_core::quantity!("initial_gradient"),
        )?,
        scioxide_core::backend::try_filled(
            parameters,
            T::Real::ZERO,
            scioxide_core::quantity!("parameter_gradient"),
        )?,
    ))
}

struct Products<T: NumericScalar> {
    point: Vec<T>,
    slope: Vec<T>,
    state: Vec<T>,
    parameters: Vec<T::Real>,
    provider: Vec<T>,
    objective: Vec<T>,
}
impl<T: NumericScalar> Products<T> {
    #[allow(clippy::too_many_arguments)]
    fn observations(
        &mut self,
        objective: &impl AdjointObjective<T>,
        times: &[T::Real],
        time: T::Real,
        state: &[T],
        parameters: &[T::Real],
        packed: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        packed.fill(T::ZERO);
        if packed.is_empty() {
            return Ok(());
        }
        let n = self.state.len();
        let p = self.parameters.len();
        let start = times.partition_point(|&t| t < time);
        let end = times.partition_point(|&t| t <= time);
        for index in start..end {
            admit(context, EvaluationRole::Objective)?;
            self.state.fill(T::from_real(T::Real::NAN));
            self.parameters.fill(T::Real::NAN);
            let value = objective.observation(
                index,
                &state[..n],
                parameters,
                &mut self.state,
                &mut self.parameters,
                &mut self.objective,
                context,
            )?;
            crate::safety::finite(&self.state, scioxide_core::NonFiniteOrigin::Callback)?;
            crate::safety::finite(&self.parameters, scioxide_core::NonFiniteOrigin::Callback)?;
            if !value.is_finite() {
                return Err(invalid(
                    scioxide_core::argument!("objective"),
                    "nonfinite observation cost",
                ));
            }
            for (out, &v) in packed[..n].iter_mut().zip(&self.state) {
                *out = *out + v;
            }
            for (out, &v) in packed[n..n + p].iter_mut().zip(&self.parameters) {
                *out = *out + T::from_real(v);
            }
            packed[n + p] = packed[n + p] + T::from_real(value);
        }
        finite(packed, scioxide_core::NonFiniteOrigin::Callback)
    }
    fn new(n: usize, p: usize, provider: usize, objective: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            point: scioxide_core::backend::try_filled(
                n + 1,
                T::ZERO,
                scioxide_core::quantity!("point"),
            )?,
            slope: scioxide_core::backend::try_filled(
                n + 1,
                T::ZERO,
                scioxide_core::quantity!("slope"),
            )?,
            state: scioxide_core::backend::try_filled(
                n,
                T::ZERO,
                scioxide_core::quantity!("state"),
            )?,
            parameters: scioxide_core::backend::try_filled(
                p,
                T::Real::ZERO,
                scioxide_core::quantity!("parameters"),
            )?,
            provider: scioxide_core::backend::try_filled(
                provider,
                T::ZERO,
                scioxide_core::quantity!("provider"),
            )?,
            objective: scioxide_core::backend::try_filled(
                objective,
                T::ZERO,
                scioxide_core::quantity!("objective"),
            )?,
        })
    }
    fn objective(
        &mut self,
        objective: &impl AdjointObjective<T>,
        part: AdjointObjectivePart,
        time: T::Real,
        state: &[T],
        parameters: &[T::Real],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AdjointObjectiveValue<T::Real>, SciOxideError> {
        admit(context, EvaluationRole::Objective)?;
        self.state.fill(T::from_real(T::Real::NAN));
        self.parameters.fill(T::Real::NAN);
        let value = objective.evaluate(
            part,
            time,
            state,
            parameters,
            &mut self.state,
            &mut self.parameters,
            &mut self.objective,
            context,
        )?;
        crate::safety::finite(&self.state, scioxide_core::NonFiniteOrigin::Callback)?;
        crate::safety::finite(&self.parameters, scioxide_core::NonFiniteOrigin::Callback)?;
        if !value.value.is_finite() || !value.time_derivative.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "nonfinite objective value or time derivative",
            ));
        }
        Ok(value)
    }
    fn pullback(
        &mut self,
        problem: &impl OdeAdjointSystem<T>,
        time: T::Real,
        state: &[T],
        seed: &[T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        admit(context, EvaluationRole::LinearOperator)?;
        self.state.fill(T::from_real(T::Real::NAN));
        self.parameters.fill(T::Real::NAN);
        problem.pullback(
            time,
            state,
            seed,
            &mut self.state,
            &mut self.parameters,
            &mut self.provider,
            context,
        )?;
        crate::safety::finite(&self.state, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(&self.parameters, scioxide_core::NonFiniteOrigin::Callback)
    }
}

fn prepare_observations<S: RealScalar>(
    times: &[S],
    mode: AdjointMode,
) -> Result<Vec<S>, SciOxideError> {
    if (!times.is_empty() && mode != AdjointMode::Discrete)
        || times.iter().any(|t| !t.is_finite())
        || times.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(
            scioxide_core::argument!("objective"),
            "observations require discrete mode and finite nondecreasing times",
        ));
    }
    let mut prepared = scioxide_core::backend::try_filled(
        times.len(),
        S::ZERO,
        scioxide_core::quantity!("observation_times"),
    )?;
    prepared.copy_from_slice(times);
    Ok(prepared)
}
fn check_observations<S: RealScalar>(prepared: &[S], actual: &[S]) -> Result<(), SciOxideError> {
    if prepared.len() != actual.len()
        || prepared
            .iter()
            .zip(actual)
            .any(|(a, b)| a.into_f64().to_bits() != b.into_f64().to_bits())
    {
        return Err(invalid(
            scioxide_core::argument!("objective"),
            "observation schedule changed after preparation",
        ));
    }
    Ok(())
}

/// Record `stop` as the attempt's classification and return its error.
fn stop_error<S: RealScalar>(stop: Stop, report: &mut AdjointReport<S>) -> SciOxideError {
    report.termination = Some(AdjointTermination::from_stop(stop));
    crate::termination::stop_error(stop, report.accounting())
}

/// Classify a shared recording or reverse drive; `Ok(true)` means a terminal event.
fn ivp_finish<S: RealScalar>(
    finish: crate::ivp::Finish,
    report: &mut AdjointReport<S>,
) -> Result<bool, SciOxideError> {
    use crate::ivp::Finish;
    match finish {
        Finish::ReachedEnd => Ok(false),
        Finish::TerminalEvent => Ok(true),
        Finish::Stop(stop) => Err(stop_error(stop, report)),
        Finish::HistoryCapacity => Err(convergence(
            scioxide_core::ConvergenceReason::NestedSolver,
            "initial-value integration failed",
        )),
    }
}

/// Sizes and observation times that depend on the callbacks, checked before storage is
/// reserved.
#[derive(Clone, Copy)]
struct Shape<'a, S: RealScalar> {
    dimension: usize,
    parameters: usize,
    provider_scratch: usize,
    objective_scratch: usize,
    observations: &'a [S],
}
impl<S: RealScalar> Shape<'_, S> {
    fn empty(dimension: usize, parameters: usize) -> Self {
        Self {
            dimension,
            parameters,
            provider_scratch: 0,
            objective_scratch: 0,
            observations: &[],
        }
    }
}
