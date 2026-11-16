//! Settings of ODE and DAE adjoint solves.
use super::*;
use crate::DaeNumericalOptions;
use scioxide_core::RelativeTolerance;

/// Holds the settings that ODE and DAE adjoint solves share: the mode, the step
/// policy, the error tolerances, checkpointing and resource limits.
///
/// Wrap it in [`OdeAdjointOptions`] or [`DaeAdjointOptions`] to build a plan. The
/// setters do not fail; the plan constructor checks the whole configuration.
///
/// Checkpointing trades memory for recomputation. With the default
/// [`CheckpointPolicy::StoreAll`] the forward integration stores every accepted step,
/// and the reverse sweep recomputes nothing. With
/// [`CheckpointPolicy::MemoryBudget`] the plan picks a binomial checkpoint schedule
/// (Griewank, 1992) with the most snapshots, at most 64, for which the workspace fits
/// [`Self::with_max_workspace_bytes`] and the recomputation fits
/// [`Self::with_max_replay_steps`]. [`CheckpointPolicy::Explicit`] uses a schedule you
/// provide, which must cover [`Self::with_tape_capacity`] steps.
///
/// # References
///
/// - A. Griewank, "Achieving logarithmic growth of temporal and spatial complexity in
///   reverse automatic differentiation", *Optim. Methods Softw.* 1(1), 35–54, 1992.
/// - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008,
///   chapter 12.
#[derive(Debug, Clone)]
#[must_use]
pub struct AdjointOptions<S: RealScalar> {
    pub(super) policies: Policies<S>,
    adjoint_error_configured: bool,
    quadrature_error_configured: bool,
}
impl<S: RealScalar> AdjointOptions<S> {
    pub(super) fn validate_formulation(&self, trapezoid: bool) -> Result<(), SciOxideError> {
        if self.mode() == AdjointMode::Discrete && self.adjoint_error_configured {
            return Err(invalid(
                scioxide_core::argument!("options.adjoint_error"),
                "adjoint integration error control requires continuous mode",
            ));
        }
        if trapezoid && self.quadrature_error_configured {
            return Err(invalid(
                scioxide_core::argument!("options.quadrature_error"),
                "fixed trapezoid has no quadrature error estimator",
            ));
        }
        Ok(())
    }

    /// Creates settings for `mode` with initial step magnitude `initial_step`.
    ///
    /// The defaults are adaptive steps; absolute and relative tolerances `√ε` in the
    /// maximum norm for the forward, adjoint and quadrature errors, where `ε` is the
    /// machine epsilon of `S`; exact agreement of recomputed states;
    /// [`CheckpointPolicy::StoreAll`]; a tape of 1024 steps; and no other limits.
    /// The plan constructor checks that `initial_step` is positive and finite.
    ///
    /// # Errors
    ///
    /// Does not fail for `f32` or `f64`. The `Result` covers the construction of the
    /// default tolerances.
    pub fn new(mode: AdjointMode, initial_step: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            policies: Policies::new(mode, initial_step)?,
            adjoint_error_configured: false,
            quadrature_error_configured: false,
        })
    }
    /// Returns the adjoint mode.
    pub const fn mode(&self) -> AdjointMode {
        self.policies.mode()
    }
    /// Limits the attempted integration steps of a solve, forward and continuous
    /// reverse steps together.
    ///
    /// Reaching it stops with [`AdjointTermination::BudgetLimit`] (`Proposal`). It is
    /// independent of [`Self::with_tape_capacity`].
    pub fn with_max_steps(mut self, steps: Budget) -> Self {
        self.policies.max_steps = Some(steps);
        self
    }
    /// Returns the limit on attempted integration steps, if any.
    pub const fn max_steps(&self) -> Option<Budget> {
        self.policies.max_steps
    }
    /// Sets the magnitude of the first forward step; it must be positive and finite.
    ///
    /// The direction comes from the solve interval.
    pub fn with_initial_step(mut self, step: S) -> Self {
        self.policies = self.policies.with_initial_step(step);
        self
    }
    /// Returns the magnitude of the first forward step.
    pub const fn initial_step(&self) -> S {
        self.policies.initial_step()
    }
    /// Makes every forward step use the initial step magnitude, without error control,
    /// if `fixed` is `true`.
    ///
    /// Steps are still shortened to meet step limits of the system, observation times
    /// and the end of the interval.
    pub fn with_fixed_step(mut self, fixed: bool) -> Self {
        self.policies = self.policies.with_fixed_step(fixed);
        self
    }
    /// Returns `true` if forward steps have a fixed size.
    pub const fn fixed_step(&self) -> bool {
        self.policies.fixed_step()
    }
    /// Sets the tolerance for the local error of the continuous adjoint state `λ`.
    ///
    /// The plan constructor rejects it in [`AdjointMode::Discrete`].
    pub fn with_adjoint_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.policies = self.policies.with_adjoint_error(tolerance);
        self.adjoint_error_configured = true;
        self
    }
    /// Returns the tolerance for the continuous adjoint state.
    pub const fn adjoint_error(&self) -> &ErrorTolerance<S> {
        self.policies.adjoint_error()
    }
    /// Sets the tolerance for the local error of the running-cost integral and of the
    /// continuous parameter-gradient integral.
    ///
    /// [`DaeAdjointSolver::trapezoid`] rejects it, since the fixed-step trapezoidal rule
    /// has no error estimate.
    pub fn with_quadrature_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.policies = self.policies.with_quadrature_error(tolerance);
        self.quadrature_error_configured = true;
        self
    }
    /// Returns the quadrature tolerance.
    pub const fn quadrature_error(&self) -> &ErrorTolerance<S> {
        self.policies.quadrature_error()
    }
    /// Sets how far a state recomputed from a checkpoint may differ from the stored
    /// state.
    ///
    /// The default, zero absolute and relative tolerance, requires bitwise agreement.
    /// A larger difference stops the solve with [`AdjointTermination::ReplayMismatch`].
    pub fn with_replay_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.policies = self.policies.with_replay_error(tolerance);
        self
    }
    /// Returns the tolerance for recomputed states.
    pub const fn replay_error(&self) -> &ErrorTolerance<S> {
        self.policies.replay_error()
    }
    /// Sets the checkpoint policy; see the type documentation.
    ///
    /// With [`CheckpointPolicy::MemoryBudget`], the byte limit covers the whole
    /// workspace, not only the checkpoints.
    pub fn with_checkpoint_policy(mut self, policy: CheckpointPolicy) -> Self {
        self.policies = self.policies.with_checkpoint_policy(policy);
        self
    }
    /// Returns the checkpoint policy.
    pub const fn checkpoint_policy(&self) -> &CheckpointPolicy {
        self.policies.checkpoint_policy()
    }
    /// Sets the largest number of accepted forward steps a solve can record; it must be
    /// positive and defaults to 1024.
    ///
    /// The workspace is sized for this many steps. A solve that needs more stops with
    /// [`AdjointTermination::BudgetLimit`] (`Iteration`).
    pub fn with_tape_capacity(mut self, steps: usize) -> Self {
        self.policies = self.policies.with_tape_capacity(steps);
        self
    }
    /// Returns the tape capacity.
    pub const fn tape_capacity(&self) -> usize {
        self.policies.tape_capacity()
    }
    /// Limits the forward steps recomputed from checkpoints in one solve.
    ///
    /// The plan constructor rejects a checkpoint schedule whose recomputation for a
    /// full tape exceeds the limit, so this limit never stops a solve.
    pub fn with_max_replay_steps(mut self, steps: usize) -> Self {
        self.policies = self.policies.with_max_replay_steps(steps);
        self
    }
    /// Returns the limit on recomputed forward steps.
    pub const fn max_replay_steps(&self) -> usize {
        self.policies.max_replay_steps()
    }
    /// Limits the reverse steps of a solve: one per recorded step in discrete mode, and
    /// every attempted substep in continuous mode.
    ///
    /// Reaching it stops with [`AdjointTermination::BudgetLimit`] (`Iteration`).
    pub fn with_max_reverse_steps(mut self, steps: usize) -> Self {
        self.policies = self.policies.with_max_reverse_steps(steps);
        self
    }
    /// Returns the limit on reverse steps.
    pub const fn max_reverse_steps(&self) -> usize {
        self.policies.max_reverse_steps()
    }
    /// Limits the numerical storage of the workspace, in bytes.
    ///
    /// The limit includes linear-solver storage. It excludes storage held by your
    /// callbacks, the borrowed inputs and outputs, and allocator overhead.
    pub fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.policies = self.policies.with_max_workspace_bytes(bytes);
        self
    }
    /// Returns the limit on workspace storage.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.policies.max_workspace_bytes()
    }
    /// Limits the callback evaluations of a solve: right-hand sides or residuals,
    /// pullbacks, objective terms, events and linear solves, including those of nested
    /// consistent-initialization solves.
    ///
    /// Reaching it stops with [`AdjointTermination::BudgetLimit`] (`Evaluation`).
    pub fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.policies = self.policies.with_max_evaluations(evaluations);
        self
    }
    /// Returns the limit on callback evaluations, if any.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.policies.max_evaluations()
    }
}
/// Holds the settings of an ODE adjoint solve: the shared [`AdjointOptions`], the
/// forward error tolerance, terminal-event detection and interpolation diagnostics.
#[derive(Debug, Clone)]
#[must_use]
pub struct OdeAdjointOptions<S: RealScalar> {
    pub(super) common: AdjointOptions<S>,
    primal_error: ErrorTolerance<S>,
    event_options: crate::OdeEventOptions<S>,
    interpolation_defect: bool,
}
impl<S: RealScalar> OdeAdjointOptions<S> {
    /// Returns the tolerance for the local error of the forward Dormand–Prince 5(4)
    /// integration.
    pub const fn primal_error(&self) -> &ErrorTolerance<S> {
        &self.primal_error
    }
    /// Returns the settings for detecting and locating the terminal event.
    pub const fn event_options(&self) -> crate::OdeEventOptions<S> {
        self.event_options
    }
    /// Returns `true` if the continuous reverse sweep samples the interpolation defect.
    pub const fn interpolation_defect_sampling(&self) -> bool {
        self.interpolation_defect
    }

    /// Creates ODE settings from `common`, with the forward tolerance of `common`
    /// (`√ε` by default) and the default [`OdeEventOptions`](crate::OdeEventOptions).
    pub fn new(common: AdjointOptions<S>) -> Self {
        Self {
            primal_error: common.policies.primal_error.clone(),
            event_options: common.policies.event_options,
            interpolation_defect: false,
            common,
        }
    }
    pub(super) fn policies(&self) -> Result<Policies<S>, SciOxideError> {
        self.common.validate_formulation(false)?;
        if self.interpolation_defect && self.common.mode() != AdjointMode::Continuous {
            return Err(invalid(
                scioxide_core::argument!("options.interpolation_defect"),
                "interpolation defect sampling requires continuous mode",
            ));
        }
        let mut policies = self.common.policies.clone();
        policies.primal_error = self.primal_error.clone();
        policies.event_options = self.event_options;
        policies.interpolation_defect = self.interpolation_defect;
        Ok(policies)
    }
    /// Sets the tolerance for the local error of the forward integration.
    pub fn with_primal_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.primal_error = tolerance;
        self
    }
    /// Sets the settings for detecting and locating the terminal event.
    pub fn with_event_options(mut self, options: crate::OdeEventOptions<S>) -> Self {
        self.event_options = options;
        self
    }
    /// Makes the continuous reverse sweep report
    /// [`AdjointErrorReport::interpolation_defect`] if `enabled` is `true`.
    ///
    /// Each sample costs one extra right-hand-side evaluation. The plan constructor
    /// rejects this setting in [`AdjointMode::Discrete`].
    pub fn with_interpolation_defect_sampling(mut self, enabled: bool) -> Self {
        self.interpolation_defect = enabled;
        self
    }
    /// Returns the shared settings.
    pub const fn common(&self) -> &AdjointOptions<S> {
        &self.common
    }
    /// Sets the workspace byte limit of the shared settings, as
    /// [`AdjointOptions::with_max_workspace_bytes`].
    pub fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.common = self.common.with_max_workspace_bytes(bytes);
        self
    }
}
/// Holds the settings of a DAE adjoint solve: the shared [`AdjointOptions`], the DAE
/// integrator settings and the residual tolerance of transposed linear systems.
///
/// The initial step and the step and evaluation limits come from the shared
/// settings; the DAE settings have none of their own. DAE adjoints have no terminal
/// event and no separate forward tolerance, as the examples below show.
///
/// ```compile_fail
/// use scioxide_ode::{AdjointMode, AdjointOptions, DaeAdjointOptions, DaeNumericalOptions};
/// let options = DaeAdjointOptions::new(
///     AdjointOptions::new(AdjointMode::Discrete, 0.1).unwrap(),
///     DaeNumericalOptions::new(),
/// )
/// .unwrap();
/// options.with_event_options(scioxide_ode::OdeEventOptions::new());
/// ```
///
/// ```compile_fail
/// use scioxide_ode::{AdjointMode, AdjointOptions, DaeAdjointOptions, DaeNumericalOptions};
/// let options = DaeAdjointOptions::new(
///     AdjointOptions::new(AdjointMode::Discrete, 0.1).unwrap(),
///     DaeNumericalOptions::new(),
/// )
/// .unwrap();
/// options.with_primal_error(scioxide_core::ErrorTolerance::default());
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct DaeAdjointOptions<S: RealScalar> {
    pub(super) common: AdjointOptions<S>,
    pub(super) numerics: DaeNumericalOptions<S>,
    pub(super) linear_residual: RelativeTolerance<S>,
}
impl<S: RealScalar> DaeAdjointOptions<S> {
    /// Creates DAE settings from `common` and the integrator settings `numerics`.
    ///
    /// Transposed linear systems `Aᵀλ = c` of the reverse sweep must reach a linear
    /// residual of `64ε` by default, where `ε` is the machine epsilon of `S`. The
    /// Krylov backend measures the relative residual `‖c − Aᵀλ‖₂ / ‖c‖₂`; the dense
    /// and sparse backends measure the normwise backward error
    /// `‖c − Aᵀλ‖∞ / (‖Aᵀ‖∞ ‖λ‖∞ + ‖c‖∞)`, which their pivoted LU attains on
    /// ill-conditioned systems as well.
    ///
    /// # Errors
    ///
    /// Does not fail for `f32` or `f64`. The `Result` covers the construction of the
    /// default residual tolerance.
    pub fn new(
        common: AdjointOptions<S>,
        numerics: DaeNumericalOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            common,
            numerics,
            linear_residual: RelativeTolerance::new(scalar::<S>(64.)? * S::EPSILON)?,
        })
    }
    /// Sets the linear residual that transposed linear systems must reach, measured
    /// as described in [`Self::new`].
    pub const fn with_linear_residual(mut self, tolerance: RelativeTolerance<S>) -> Self {
        self.linear_residual = tolerance;
        self
    }
    /// Returns the linear residual that transposed linear systems must reach.
    pub const fn linear_residual(&self) -> RelativeTolerance<S> {
        self.linear_residual
    }
    /// Returns the DAE integrator settings.
    pub const fn numerics(&self) -> &DaeNumericalOptions<S> {
        &self.numerics
    }
    /// Returns the shared settings.
    pub const fn common(&self) -> &AdjointOptions<S> {
        &self.common
    }
    /// Sets the workspace byte limit of the shared settings, as
    /// [`AdjointOptions::with_max_workspace_bytes`].
    pub fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.common = self.common.with_max_workspace_bytes(bytes);
        self
    }
}

impl<S: RealScalar> From<AdjointOptions<S>> for OdeAdjointOptions<S> {
    fn from(common: AdjointOptions<S>) -> Self {
        Self::new(common)
    }
}
