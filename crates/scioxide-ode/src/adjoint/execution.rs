//! Adjoint plans, problems and workspaces.
use super::*;
use alloc::sync::Arc;
use core::marker::PhantomData;
use scioxide_core::{AttemptRecord, Capacity, ExecutionContext, PartialFailure};

/// Describes one ODE adjoint solve: the system, the objective, the initial state, the
/// interval and an optional terminal event.
///
/// It only borrows its parts. The plan checks them before it evaluates any callback.
#[must_use]
pub struct OdeAdjointProblem<'a, T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>> {
    system: &'a P,
    objective: &'a O,
    initial: &'a [T],
    interval: (T::Real, T::Real),
    event: Option<&'a dyn AdjointTerminalEvent<T>>,
}
impl<'a, T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>>
    OdeAdjointProblem<'a, T, P, O>
{
    /// Creates a problem with initial state `initial` at `interval.0`, integrated
    /// toward `interval.1`.
    ///
    /// The end may lie before the start, for integration backward in time.
    pub const fn new(
        system: &'a P,
        objective: &'a O,
        initial: &'a [T],
        interval: (T::Real, T::Real),
    ) -> Self {
        Self {
            system,
            objective,
            initial,
            interval,
            event: None,
        }
    }
    /// Ends the forward integration at the first root of `event`, or removes the event
    /// with `None`.
    pub fn with_terminal_event(mut self, event: Option<&'a dyn AdjointTerminalEvent<T>>) -> Self {
        self.event = event;
        self
    }
    /// Returns the system.
    pub const fn system(&self) -> &'a P {
        self.system
    }
    /// Returns the objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }
    /// Returns the initial state.
    pub const fn initial(&self) -> &'a [T] {
        self.initial
    }
    /// Returns the initial and final time.
    pub const fn interval(&self) -> (T::Real, T::Real) {
        self.interval
    }
    fn shape(&self) -> Shape<'_, T::Real> {
        Shape {
            dimension: self.system.dimension(),
            parameters: self.system.parameter_dimension(),
            provider_scratch: self.system.pullback_workspace_len(),
            objective_scratch: self.objective.workspace_len(),
            observations: self.objective.observation_times(),
        }
    }
}
#[derive(Debug, Clone)]
struct OdeConfiguration<T: NumericScalar> {
    dimension: usize,
    parameters: usize,
    options: OdeAdjointOptions<T::Real>,
    scalar: PhantomData<T>,
}
/// Computes the gradient of a scalar objective of an ODE solution with respect to the
/// initial state and the parameters by the adjoint method.
///
/// For `y′ = f(t, y, p)`, `y(t₀) = y₀`, and the objective `J` defined by an
/// [`AdjointObjective`], a solve returns `dJ/dy₀` and `dJ/dp`. For real states the
/// continuous adjoint state `λ(t)` satisfies
///
/// ```text
/// λ′ = −(∂f/∂y)ᵀ λ − (∂L/∂y)ᵀ,      λ(T) = (∂Φ/∂y)ᵀ,
/// dJ/dy₀ = λ(t₀),
/// dJ/dp = ∂Φ/∂p + ∫_{t₀}^{T} (∂L/∂p + λᵀ ∂f/∂p) dt,
/// ```
///
/// where `Φ` and `L` are the terminal and running costs. Observation terms add jumps
/// to `λ` at their times, and a terminal event adds the dependence of `T`. States may
/// be real or complex; time and parameters are real.
///
/// The forward integration uses the Dormand–Prince 5(4) method and records each
/// accepted step, up to [`AdjointOptions::with_tape_capacity`] steps. In
/// [`AdjointMode::Discrete`] the reverse sweep applies the transposed derivative of
/// each recorded step; in [`AdjointMode::Continuous`] it integrates the equations
/// above backward with Dormand–Prince 5(4) on each recorded step, using the dense
/// output of the forward step. A solve costs one forward integration, the forward
/// steps recomputed from checkpoints (none with the default policy) and one reverse
/// sweep of [`OdeAdjointSystem::pullback`] products. Storage is bounded by the tape
/// capacity and the checkpoint policy of [`AdjointOptions`]. The local error
/// estimates in [`AdjointErrorReport`] do not bound the error of the gradient.
///
/// # Examples
///
/// ```
/// use scioxide_ode::{AdjointMode, AdjointOptions, OdeAdjointSolver};
///
/// let options = AdjointOptions::new(AdjointMode::Discrete, 0.1)?;
/// let plan = OdeAdjointSolver::<f64>::new(2, 1, options)?;
/// assert_eq!(plan.dimension(), 2);
/// assert_eq!(plan.parameter_dimension(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §§II.4–II.6.
/// - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008,
///   chapter 12.
#[derive(Debug, Clone)]
#[must_use]
pub struct OdeAdjointSolver<T: NumericScalar> {
    configuration: Arc<OdeConfiguration<T>>,
}

/// Describes the workspace an [`OdeAdjointSolver`] needs for one problem, returned by
/// [`OdeAdjointSolver::workspace_requirements`].
#[derive(Debug, Clone)]
pub struct OdeAdjointRequirements<T: NumericScalar> {
    configuration: Arc<OdeConfiguration<T>>,
    prepared: Arc<PreparedOdeAdjoint<T>>,
    storage: Arc<OdeStorageRequirements>,
}
impl<T: NumericScalar> OdeAdjointRequirements<T> {
    /// Returns the length of the initial-state gradient.
    pub fn initial_gradient_len(&self) -> usize {
        self.storage.initial_gradient_len()
    }
    /// Returns the length of the parameter gradient.
    pub fn parameter_gradient_len(&self) -> usize {
        self.storage.parameter_gradient_len()
    }
    /// Returns the upper bound, in bytes, on the numerical storage of the workspace.
    pub fn required_bytes(&self) -> usize {
        self.storage.workspace_bytes()
    }
}

/// Holds the storage of [`OdeAdjointSolver`] solves and the record of the last attempt.
///
/// See [Reusing workspaces](crate#reusing-workspaces).
#[derive(Debug)]
pub struct OdeAdjointWorkspace<T: NumericScalar> {
    buffers: OdeAdjointBuffers<T>,
    prepared: Option<OdeAdjointRequirements<T>>,
}
impl<T: NumericScalar> Default for OdeAdjointWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> OdeAdjointWorkspace<T> {
    /// Creates an empty workspace.
    pub const fn new() -> Self {
        Self {
            buffers: OdeAdjointBuffers::new(),
            prepared: None,
        }
    }
    /// Returns whether the workspace has storage of the shape `requirements` describes.
    ///
    /// # Errors
    ///
    /// Does not fail. The `Result` follows the shared
    /// [workspace protocol](scioxide_core#plans-and-workspaces).
    pub fn check(
        &self,
        requirements: &OdeAdjointRequirements<T>,
    ) -> Result<Capacity, SciOxideError> {
        self.buffers.check(&requirements.storage)
    }
    /// Prepares the workspace for `requirements`, replacing storage of another shape.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if memory cannot be reserved, and `SizeOverflow` if
    /// a storage size overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &OdeAdjointRequirements<T>,
    ) -> Result<(), SciOxideError> {
        self.buffers.try_reserve(&requirements.storage)?;
        self.prepared = Some(requirements.clone());
        Ok(())
    }
    /// Returns the record of the last solve, with its [`AdjointReport`] when the solve
    /// started.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<AdjointReport<T::Real>>> {
        self.buffers.last_attempt()
    }
}

impl<T: NumericScalar> OdeAdjointSolver<T> {
    /// Creates a plan for systems with `dimension` states and `parameter_dimension`
    /// real parameters.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` or the tape capacity is zero, if the initial
    ///   step is not positive and finite, if an adjoint tolerance or
    ///   interpolation-defect sampling is set in [`AdjointMode::Discrete`], or if no
    ///   checkpoint schedule meets the byte and recomputation limits.
    /// - `DimensionMismatch` if a per-component tolerance has the wrong length, or if
    ///   the storage of the checkpoint policy exceeds
    ///   [`AdjointOptions::with_max_workspace_bytes`].
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn new(
        dimension: usize,
        parameter_dimension: usize,
        options: impl Into<OdeAdjointOptions<T::Real>>,
    ) -> Result<Self, SciOxideError> {
        let plan = Self {
            configuration: Arc::new(OdeConfiguration {
                dimension,
                parameters: parameter_dimension,
                options: options.into(),
                scalar: PhantomData,
            }),
        };
        let _ = plan.prepare(Shape::empty(dimension, parameter_dimension))?;
        Ok(plan)
    }
    /// Returns the settings of the plan.
    pub fn options(&self) -> &OdeAdjointOptions<T::Real> {
        &self.configuration.options
    }
    /// Returns the number of states.
    pub fn dimension(&self) -> usize {
        self.configuration.dimension
    }
    /// Returns the number of real parameters.
    pub fn parameter_dimension(&self) -> usize {
        self.configuration.parameters
    }
    fn prepare(&self, shape: Shape<'_, T::Real>) -> Result<PreparedOdeAdjoint<T>, SciOxideError> {
        if shape.dimension != self.dimension() || shape.parameters != self.parameter_dimension() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "problem dimensions differ from the adjoint plan",
            ));
        }
        PreparedOdeAdjoint::new(shape, self.configuration.options.policies()?)
    }
    /// Returns the workspace requirements for `problem`.
    ///
    /// The requirements depend on the scratch lengths and observation times of the
    /// callbacks. No callback that evaluates the system or objective runs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the system is not replayable or has a mass matrix, if its
    ///   dimensions differ from the plan, or if the observation times are not finite
    ///   and nondecreasing or are given in [`AdjointMode::Continuous`].
    /// - `NonFinite` (origin `Callback`) if a system parameter is NaN or infinite.
    /// - The errors of [`Self::new`] for the problem's storage, and
    ///   `AllocationFailure`.
    pub fn workspace_requirements<P: OdeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &OdeAdjointProblem<'_, T, P, O>,
    ) -> Result<OdeAdjointRequirements<T>, SciOxideError> {
        if !problem.system.is_replayable() || problem.system.mass_matrix_revision().is_some() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "adjoints require replayable supported dynamics",
            ));
        }
        finite(
            problem.system.parameters(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        let prepared = Arc::new(self.prepare(problem.shape())?);
        let storage = Arc::new(prepared.workspace_requirements()?);
        Ok(OdeAdjointRequirements {
            configuration: Arc::clone(&self.configuration),
            prepared,
            storage,
        })
    }
    /// Returns a workspace that meets the requirements of `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`] and
    /// [`OdeAdjointWorkspace::try_reserve`].
    pub fn workspace<P: OdeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &OdeAdjointProblem<'_, T, P, O>,
    ) -> Result<OdeAdjointWorkspace<T>, SciOxideError> {
        let mut workspace = OdeAdjointWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(workspace)
    }
    /// Computes `dJ/dy₀` into `initial_gradient` and `dJ/dp` into
    /// `parameter_gradient`, and returns the report.
    ///
    /// Both outputs are written only on success. The report of a failed attempt is in
    /// [`OdeAdjointWorkspace::last_attempt`]. A workspace prepared for another problem
    /// shape is prepared again first. With a prepared workspace and callbacks that do
    /// not allocate, the solve does not allocate, also with a terminal event. The solve
    /// runs in `scope`, as described in [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `InvalidInput` if `initial`, `initial_gradient` or `parameter_gradient` does
    ///   not match the plan, if an interval end is not finite, if an observation time
    ///   lies outside the interval, if the problem has both observation times and a
    ///   terminal event, or if the objective value is not finite. It is also returned,
    ///   with termination [`AdjointTermination::EventDerivativeUnavailable`], when the
    ///   terminal-event gradient does not exist.
    /// - `NonFinite` with origin `Input` for a non-finite initial state, and with
    ///   origin `Callback` for non-finite callback output.
    /// - `ConvergenceFailure` when a limit, the step size or a checkpoint mismatch
    ///   stops the solve; the report's [`AdjointReport::termination`] says which.
    /// - `ExecutionStopped` for cancellation or a deadline.
    /// - Errors returned by your callbacks, unchanged.
    pub fn solve_into<'scope, 'capability, P: OdeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &OdeAdjointProblem<'_, T, P, O>,
        initial_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
        workspace: &mut OdeAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdjointReport<T::Real>, SciOxideError> {
        let cached = workspace.prepared.as_ref().filter(|requirements| {
            Arc::ptr_eq(&requirements.configuration, &self.configuration)
                && requirements.prepared.matches(problem.shape())
        });
        let requirements = match cached {
            Some(requirements) => requirements.clone(),
            None => match self.workspace_requirements(problem) {
                Ok(requirements) => requirements,
                Err(error) => return workspace.buffers.shell.reject(error),
            },
        };
        if let Err(error) = workspace.try_reserve(&requirements) {
            return workspace.buffers.shell.reject(error);
        }
        requirements.prepared.solve_into(
            problem.system,
            problem.objective,
            problem.initial,
            problem.interval,
            problem.event,
            initial_gradient,
            parameter_gradient,
            &mut workspace.buffers,
            scope,
        )
    }
    /// Computes both gradients with a new workspace on the calling thread and returns
    /// them with the report.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::solve_into`], in a [`PartialFailure`] that holds the
    /// [`AdjointReport`] of the failed attempt when the solve started. No partial
    /// gradient is returned.
    #[allow(clippy::result_large_err, reason = "owned conveniences carry progress")]
    pub fn solve<P: OdeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &OdeAdjointProblem<'_, T, P, O>,
    ) -> Result<AdjointResult<T>, PartialFailure<AdjointReport<T::Real>>> {
        let mut workspace = OdeAdjointWorkspace::new();
        let initial_len = self.dimension();
        let (mut initial_gradient, mut parameter_gradient) =
            gradients::<T>(initial_len, self.parameter_dimension()).map_err(PartialFailure::new)?;
        match self.solve_into(
            problem,
            &mut initial_gradient,
            &mut parameter_gradient,
            &mut workspace,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(AdjointResult {
                initial_gradient,
                parameter_gradient,
                report,
            }),
            Err(error) => Err(partial(error, workspace.buffers.shell)),
        }
    }
}
/// Describes one DAE adjoint solve: the system, the objective, the initial state and
/// derivative, and the interval.
///
/// It only borrows its parts. The plan checks them before it evaluates any callback.
#[must_use]
pub struct DaeAdjointProblem<'a, T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>> {
    system: &'a P,
    objective: &'a O,
    initial: &'a [T],
    interval: (T::Real, T::Real),
    derivative: &'a [T],
}
impl<'a, T: NumericScalar, P: DaeAdjointSystem<T>, O: AdjointObjective<T>>
    DaeAdjointProblem<'a, T, P, O>
{
    /// Creates a problem with initial state `initial` and initial derivative
    /// `derivative` at `interval.0`, integrated toward `interval.1`.
    ///
    /// The solve checks the initial pair against `F = 0`, or projects it onto the
    /// constraints, as the [`DaeInitialization`](crate::DaeInitialization) in the
    /// [`DaeNumericalOptions`](crate::DaeNumericalOptions) of the plan selects.
    pub const fn new(
        system: &'a P,
        objective: &'a O,
        initial: &'a [T],
        derivative: &'a [T],
        interval: (T::Real, T::Real),
    ) -> Self {
        Self {
            system,
            objective,
            initial,
            interval,
            derivative,
        }
    }
    /// Returns the system.
    pub const fn system(&self) -> &'a P {
        self.system
    }
    /// Returns the objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }
    /// Returns the initial state.
    pub const fn initial(&self) -> &'a [T] {
        self.initial
    }
    /// Returns the initial and final time.
    pub const fn interval(&self) -> (T::Real, T::Real) {
        self.interval
    }
    fn shape(&self) -> Shape<'_, T::Real> {
        Shape {
            dimension: self.system.dimension(),
            parameters: self.system.parameter_dimension(),
            provider_scratch: self.system.pullback_workspace_len(),
            objective_scratch: self.objective.workspace_len(),
            observations: self.objective.observation_times(),
        }
    }
}
#[derive(Debug, Clone)]
struct DaeConfiguration<T: NumericScalar> {
    dimension: usize,
    parameters: usize,
    options: DaeAdjointOptions<T::Real>,
    partition: crate::DaePartition,
    backend: crate::OdeStageBackend,
    method: DaeMethod,
    scalar: PhantomData<T>,
}
#[derive(Debug, Clone, Copy)]
enum DaeMethod {
    Bdf,
    BackwardEuler,
    Trapezoid,
}
/// Computes the gradient of a scalar objective of the solution of a residual index-1
/// DAE with respect to the free initial values and the parameters by the adjoint
/// method.
///
/// For `F(t, y, y′, p) = 0` and the objective `J` defined by an [`AdjointObjective`],
/// a solve returns `dJ/dp` and the gradient with respect to the differential
/// coordinates of the initial state, in the order of the
/// [`DaePartition`](crate::DaePartition). The initial algebraic values follow from
/// the constraints and are not independent variables. The DAE must be regular and of
/// index 1. DAE adjoints have no terminal events.
///
/// Three forward methods are available, each with its own constructor:
///
/// - [`Self::new`]: variable-order BDF, orders 1 to 5, as in
///   [`DaeBdf`](crate::DaeBdf).
/// - [`Self::backward_euler`]: the order-1 BDF, with every coordinate differential.
/// - [`Self::trapezoid`]: the implicit trapezoidal rule with fixed steps, with every
///   coordinate differential, in [`AdjointMode::Discrete`] only.
///
/// The reverse sweep solves transposed linear systems whose matrices combine
/// `∂F/∂y` and `∂F/∂y′`, to the linear residual of
/// [`DaeAdjointOptions::with_linear_residual`], on the dense, sparse or Krylov
/// backend of the plan. In
/// [`AdjointMode::Continuous`] it integrates the adjoint DAE with step doubling and
/// local extrapolation. A solve costs one forward integration, the forward steps
/// recomputed from checkpoints and one reverse sweep; storage is bounded by the tape
/// capacity and the checkpoint policy. The local error estimates in
/// [`AdjointErrorReport`] do not bound the error of the gradient.
///
/// # Examples
///
/// ```
/// use scioxide_ode::{
///     AdjointMode, AdjointOptions, DaeAdjointOptions, DaeAdjointSolver, DaeNumericalOptions,
///     OdeStageBackend,
/// };
///
/// let common = AdjointOptions::new(AdjointMode::Discrete, 0.1)?;
/// let options = DaeAdjointOptions::new(common, DaeNumericalOptions::new())?;
/// let plan = DaeAdjointSolver::<f64>::backward_euler(2, 1, OdeStageBackend::Dense, options)?;
/// assert_eq!(plan.dimension(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - Y. Cao, S. Li, L. Petzold and R. Serban, "Adjoint sensitivity analysis for
///   differential-algebraic equations: the adjoint DAE system and its numerical
///   solution", *SIAM J. Sci. Comput.* 24(3), 1076–1089, 2003.
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 2nd ed.,
///   Springer, 1996, chapter V.
/// - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008,
///   chapter 12.
#[derive(Debug, Clone)]
#[must_use]
pub struct DaeAdjointSolver<T: NumericScalar> {
    configuration: Arc<DaeConfiguration<T>>,
}

/// Describes the workspace a [`DaeAdjointSolver`] needs for one problem, returned by
/// [`DaeAdjointSolver::workspace_requirements`].
#[derive(Debug, Clone)]
pub struct DaeAdjointRequirements<T: NumericScalar> {
    configuration: Arc<DaeConfiguration<T>>,
    prepared: Arc<PreparedDaeAdjoint<T>>,
    storage: Arc<DaeStorageRequirements<T>>,
}
impl<T: NumericScalar> DaeAdjointRequirements<T> {
    /// Returns the length of the initial-state gradient: the number of differential
    /// coordinates.
    pub fn initial_gradient_len(&self) -> usize {
        self.storage.initial_gradient_len()
    }
    /// Returns the length of the parameter gradient.
    pub fn parameter_gradient_len(&self) -> usize {
        self.storage.parameter_gradient_len()
    }
    /// Returns the upper bound, in bytes, on the numerical storage of the workspace.
    pub fn required_bytes(&self) -> usize {
        self.storage.workspace_bytes()
    }
}

/// Holds the storage of [`DaeAdjointSolver`] solves and the record of the last attempt.
///
/// See [Reusing workspaces](crate#reusing-workspaces).
#[derive(Debug)]
pub struct DaeAdjointWorkspace<T: NumericScalar> {
    buffers: DaeAdjointBuffers<T>,
    prepared: Option<DaeAdjointRequirements<T>>,
}
impl<T: NumericScalar> Default for DaeAdjointWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> DaeAdjointWorkspace<T> {
    /// Creates an empty workspace.
    pub const fn new() -> Self {
        Self {
            buffers: DaeAdjointBuffers::new(),
            prepared: None,
        }
    }
    /// Returns whether the workspace has storage of the shape `requirements` describes.
    ///
    /// # Errors
    ///
    /// Does not fail. The `Result` follows the shared
    /// [workspace protocol](scioxide_core#plans-and-workspaces).
    pub fn check(
        &self,
        requirements: &DaeAdjointRequirements<T>,
    ) -> Result<Capacity, SciOxideError> {
        self.buffers.check(&requirements.storage)
    }
    /// Prepares the workspace for `requirements`, replacing storage of another shape or
    /// linear-solver backend.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if memory cannot be reserved, `SizeOverflow` if a
    /// storage size overflows `usize`, and the errors of preparing the linear-solver
    /// backend.
    pub fn try_reserve(
        &mut self,
        requirements: &DaeAdjointRequirements<T>,
    ) -> Result<(), SciOxideError> {
        self.buffers.try_reserve(&requirements.storage)?;
        self.prepared = Some(requirements.clone());
        Ok(())
    }
    /// Returns the record of the last solve, with its [`AdjointReport`] when the solve
    /// started.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<AdjointReport<T::Real>>> {
        self.buffers.last_attempt()
    }
}

impl<T: NumericScalar> DaeAdjointSolver<T> {
    /// Creates a plan that integrates forward with variable-order BDF on the
    /// coordinate roles of `partition`, for systems with `parameter_dimension` real
    /// parameters, solving linear systems with `backend`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the tape capacity is zero, if the initial step is not
    ///   positive and finite, if an adjoint tolerance is set in
    ///   [`AdjointMode::Discrete`], if the DAE settings do not suit `backend`, or if
    ///   no workspace or checkpoint schedule fits the byte and recomputation limits.
    /// - `DimensionMismatch` if a per-component tolerance has the wrong length, or if
    ///   the storage of the checkpoint policy exceeds
    ///   [`AdjointOptions::with_max_workspace_bytes`].
    /// - `SizeOverflow` and `AllocationFailure` for storage that cannot be sized or
    ///   allocated.
    pub fn new(
        partition: crate::DaePartition,
        parameter_dimension: usize,
        backend: crate::OdeStageBackend,
        options: DaeAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let dimension = partition.dimension();
        let plan = Self {
            configuration: Arc::new(DaeConfiguration {
                dimension,
                parameters: parameter_dimension,
                options,
                partition,
                backend,
                method: DaeMethod::Bdf,
                scalar: PhantomData,
            }),
        };
        let _ = plan.prepare(Shape::empty(dimension, parameter_dimension))?;
        Ok(plan)
    }
    /// Creates a plan that integrates forward with the backward Euler method, for
    /// systems with `dimension` coordinates, all differential.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `InvalidInput` if `dimension` is zero.
    pub fn backward_euler(
        dimension: usize,
        parameter_dimension: usize,
        backend: crate::OdeStageBackend,
        options: DaeAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let flags = scioxide_core::backend::try_filled(
            dimension,
            true,
            scioxide_core::quantity!("partition_flags"),
        )?;
        let partition = crate::DaePartition::new(&flags, &flags)?;
        let dimension = partition.dimension();
        let plan = Self {
            configuration: Arc::new(DaeConfiguration {
                dimension,
                parameters: parameter_dimension,
                options,
                partition,
                backend,
                method: DaeMethod::BackwardEuler,
                scalar: PhantomData,
            }),
        };
        let _ = plan.prepare(Shape::empty(dimension, parameter_dimension))?;
        Ok(plan)
    }
    /// Creates a plan that integrates forward with the implicit trapezoidal rule, for
    /// systems with `dimension` coordinates, all differential.
    ///
    /// Each step from `(t₀, y₀, y′₀)` solves `F(t₀ + h, y₁, y′₁) = 0` with
    /// `y′₁ = 2 (y₁ − y₀)/h − y′₀` by Newton's method, and the running cost uses the
    /// trapezoidal rule `h/2 (L(t₀, y₀) + L(t₁, y₁))`. For `F = M y′ − f(t, y)` with a
    /// constant regular `M`, such as [`OdeAdjointResidual`], a step is
    /// `M (y₁ − y₀) = h/2 (f(t₀, y₀) + f(t₁, y₁))`. The method has no error estimate,
    /// so every step has the initial step size, shortened only at observation times,
    /// step limits of the system and the end of the interval.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `InvalidInput` if `dimension` is zero, if the
    /// mode is [`AdjointMode::Continuous`], if
    /// [`AdjointOptions::with_fixed_step`] is not set, or if a quadrature tolerance is
    /// set.
    pub fn trapezoid(
        dimension: usize,
        parameter_dimension: usize,
        backend: crate::OdeStageBackend,
        options: DaeAdjointOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let flags = scioxide_core::backend::try_filled(
            dimension,
            true,
            scioxide_core::quantity!("partition_flags"),
        )?;
        let partition = crate::DaePartition::new(&flags, &flags)?;
        let dimension = partition.dimension();
        let plan = Self {
            configuration: Arc::new(DaeConfiguration {
                dimension,
                parameters: parameter_dimension,
                options,
                partition,
                backend,
                method: DaeMethod::Trapezoid,
                scalar: PhantomData,
            }),
        };
        let _ = plan.prepare(Shape::empty(dimension, parameter_dimension))?;
        Ok(plan)
    }
    /// Returns the roles of the coordinates: differential or algebraic.
    pub fn partition(&self) -> &crate::DaePartition {
        &self.configuration.partition
    }
    /// Returns the settings of the plan.
    pub fn options(&self) -> &DaeAdjointOptions<T::Real> {
        &self.configuration.options
    }
    /// Returns the number of coordinates.
    pub fn dimension(&self) -> usize {
        self.configuration.dimension
    }
    /// Returns the number of real parameters.
    pub fn parameter_dimension(&self) -> usize {
        self.configuration.parameters
    }
    fn prepare(&self, shape: Shape<'_, T::Real>) -> Result<PreparedDaeAdjoint<T>, SciOxideError> {
        if shape.dimension != self.dimension() || shape.parameters != self.parameter_dimension() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "problem dimensions differ from the adjoint plan",
            ));
        }
        let configuration = &self.configuration;
        configuration
            .options
            .common
            .validate_formulation(matches!(configuration.method, DaeMethod::Trapezoid))?;
        let numerics = configuration.options.numerics.clone();
        let policies = configuration.options.common.policies.clone();
        let linear_residual = configuration.options.linear_residual;
        match configuration.method {
            DaeMethod::Bdf => PreparedDaeAdjoint::new(
                shape,
                configuration.partition.clone(),
                configuration.backend.clone(),
                numerics,
                policies,
                linear_residual,
            ),
            DaeMethod::BackwardEuler => PreparedDaeAdjoint::backward_euler(
                shape,
                configuration.backend.clone(),
                numerics,
                policies,
                linear_residual,
            ),
            DaeMethod::Trapezoid => PreparedDaeAdjoint::trapezoid(
                shape,
                configuration.backend.clone(),
                numerics,
                policies,
                linear_residual,
            ),
        }
    }
    /// Returns the workspace requirements for `problem`.
    ///
    /// The requirements depend on the scratch lengths and observation times of the
    /// callbacks. No callback that evaluates the system or objective runs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the system is not replayable, if its dimensions differ from
    ///   the plan, or if the observation times are not finite and nondecreasing or are
    ///   given in [`AdjointMode::Continuous`].
    /// - `NonFinite` (origin `Callback`) if a system parameter is NaN or infinite.
    /// - The errors of [`Self::new`] for the problem's storage.
    pub fn workspace_requirements<P: DaeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &DaeAdjointProblem<'_, T, P, O>,
    ) -> Result<DaeAdjointRequirements<T>, SciOxideError> {
        if !problem.system.is_replayable() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "adjoints require replayable supported dynamics",
            ));
        }
        finite(
            problem.system.parameters(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        let prepared = Arc::new(self.prepare(problem.shape())?);
        let storage = Arc::new(prepared.workspace_requirements()?);
        Ok(DaeAdjointRequirements {
            configuration: Arc::clone(&self.configuration),
            prepared,
            storage,
        })
    }
    /// Returns a workspace that meets the requirements of `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`] and
    /// [`DaeAdjointWorkspace::try_reserve`].
    pub fn workspace<P: DaeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &DaeAdjointProblem<'_, T, P, O>,
    ) -> Result<DaeAdjointWorkspace<T>, SciOxideError> {
        let mut workspace = DaeAdjointWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(workspace)
    }
    /// Computes the initial-state gradient into `initial_gradient` and `dJ/dp` into
    /// `parameter_gradient`, and returns the report.
    ///
    /// `initial_gradient` has one entry per differential coordinate. Both outputs are
    /// written only on success. The report of a failed attempt is in
    /// [`DaeAdjointWorkspace::last_attempt`]. A workspace prepared for another problem
    /// shape is prepared again first. With a prepared workspace and callbacks that do
    /// not allocate, the solve does not allocate. The solve runs in `scope`, as
    /// described in [Solve accounting](scioxide_core#solve-accounting).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `InvalidInput` if `initial`, the initial derivative or an output does not
    ///   match the plan, if an interval end is not finite, if an observation time lies
    ///   outside the interval, or if the objective value is not finite. The reason is
    ///   [`Inconsistent`](scioxide_core::InvalidInputReason::Inconsistent) if the
    ///   initial pair does not satisfy the constraints and the initialization only
    ///   checks it.
    /// - `NonFinite` with origin `Input` for non-finite initial values, and with
    ///   origin `Callback` for non-finite callback output.
    /// - `ConvergenceFailure` when a limit, the step size or a checkpoint mismatch
    ///   stops the solve, including a projection of the initial pair that does not
    ///   converge; the report's [`AdjointReport::termination`] says which.
    /// - `ExecutionStopped` for cancellation or a deadline.
    /// - Errors returned by your callbacks, unchanged.
    pub fn solve_into<'scope, 'capability, P: DaeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &DaeAdjointProblem<'_, T, P, O>,
        initial_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
        workspace: &mut DaeAdjointWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdjointReport<T::Real>, SciOxideError> {
        let cached = workspace.prepared.as_ref().filter(|requirements| {
            Arc::ptr_eq(&requirements.configuration, &self.configuration)
                && requirements.prepared.matches(problem.shape())
        });
        let requirements = match cached {
            Some(requirements) => requirements.clone(),
            None => match self.workspace_requirements(problem) {
                Ok(requirements) => requirements,
                Err(error) => return workspace.buffers.shell.reject(error),
            },
        };
        if let Err(error) = workspace.try_reserve(&requirements) {
            return workspace.buffers.shell.reject(error);
        }
        requirements.prepared.solve_into(
            problem.system,
            problem.objective,
            problem.initial,
            problem.derivative,
            problem.interval,
            initial_gradient,
            parameter_gradient,
            &mut workspace.buffers,
            scope,
        )
    }
    /// Computes both gradients with a new workspace on the calling thread and returns
    /// them with the report.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::solve_into`], in a [`PartialFailure`] that holds the
    /// [`AdjointReport`] of the failed attempt when the solve started. No partial
    /// gradient is returned.
    #[allow(clippy::result_large_err, reason = "owned conveniences carry progress")]
    pub fn solve<P: DaeAdjointSystem<T>, O: AdjointObjective<T>>(
        &self,
        problem: &DaeAdjointProblem<'_, T, P, O>,
    ) -> Result<AdjointResult<T>, PartialFailure<AdjointReport<T::Real>>> {
        let mut workspace = DaeAdjointWorkspace::new();
        let initial_len = self.partition().differential();
        let (mut initial_gradient, mut parameter_gradient) =
            gradients::<T>(initial_len, self.parameter_dimension()).map_err(PartialFailure::new)?;
        match self.solve_into(
            problem,
            &mut initial_gradient,
            &mut parameter_gradient,
            &mut workspace,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(AdjointResult {
                initial_gradient,
                parameter_gradient,
                report,
            }),
            Err(error) => Err(partial(error, workspace.buffers.shell)),
        }
    }
}
