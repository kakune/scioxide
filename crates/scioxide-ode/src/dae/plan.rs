//! Residual DAE plans, problems, states, output sinks and workspaces.

use scioxide_core::{
    AttemptRecord, Capacity, ExecutionContext, NumericScalar, PartialFailure, ProblemToken,
    RealScalar, SciOxideError, SolveContext, SolveIdentity, SolveScope,
};
use scioxide_linalg::Vector;

use super::{DaeOptions, DaePartition, DaeReport, DaeStorage, DaeSystem, DaeTermination};
use crate::OdeStageBackend;
use crate::safety::invalid;
use crate::shell::Shell;
use crate::termination::{Fault, Terminal};

/// Pairs a [`DaeSystem`] with the end time of the integration.
///
/// The start time and the initial pair come from the [`DaeState`] passed to
/// [`DaeBdf::advance`]. Integration runs from `state.time()` to [`Self::end`] in either
/// direction.
#[must_use]
pub struct DaeProblem<'a, T: NumericScalar, Sys: ?Sized> {
    system: &'a Sys,
    end: T::Real,
}

impl<T: NumericScalar, Sys: ?Sized> Clone for DaeProblem<'_, T, Sys> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: NumericScalar, Sys: ?Sized> Copy for DaeProblem<'_, T, Sys> {}

impl<'a, T: NumericScalar, Sys: DaeSystem<T> + ?Sized> DaeProblem<'a, T, Sys> {
    /// Creates a problem that integrates `system` to `end`.
    ///
    /// An `end` equal to the start time makes [`DaeBdf::advance`] only check or project
    /// the initial pair.
    pub const fn new(system: &'a Sys, end: T::Real) -> Self {
        Self { system, end }
    }
    /// Returns the system.
    pub const fn system(&self) -> &'a Sys {
        self.system
    }
    /// Returns the end time.
    pub const fn end(&self) -> T::Real {
        self.end
    }
}

/// Holds the time `t`, the state `y` and its derivative `ẏ` of a DAE integration.
///
/// [`DaeBdf::advance`] reads the initial pair from it and leaves the last accepted pair
/// in it. If the call fails before the initial pair is accepted, the state is unchanged.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DaeState<T: NumericScalar> {
    time: T::Real,
    values: Vector<T>,
    derivative: Vector<T>,
}

impl<T: NumericScalar> DaeState<T> {
    /// Creates a state at `time` with `values` and their time `derivative`.
    pub const fn new(time: T::Real, values: Vector<T>, derivative: Vector<T>) -> Self {
        Self {
            time,
            values,
            derivative,
        }
    }

    /// Creates a state at `time` with copies of `values` and `derivative`.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if the copies cannot be allocated.
    pub fn from_slices(
        time: T::Real,
        values: &[T],
        derivative: &[T],
    ) -> Result<Self, SciOxideError> {
        let copy = |source: &[T]| {
            let mut vector = Vector::zeros(source.len())?;
            vector.as_mut_slice().copy_from_slice(source);
            Ok::<_, SciOxideError>(vector)
        };
        Ok(Self::new(time, copy(values)?, copy(derivative)?))
    }

    /// Returns the time.
    pub const fn time(&self) -> T::Real {
        self.time
    }
    /// Returns the state values.
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }
    /// Returns the time derivative of the state.
    pub fn derivative(&self) -> &[T] {
        self.derivative.as_slice()
    }

    /// Overwrites the time, values and derivative in place, without allocating.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `values` or `derivative` differs in length from the stored
    /// state.
    pub fn reset(
        &mut self,
        time: T::Real,
        values: &[T],
        derivative: &[T],
    ) -> Result<(), SciOxideError> {
        for (stored, supplied) in [
            (self.values.len(), values.len()),
            (self.derivative.len(), derivative.len()),
        ] {
            if stored != supplied {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(stored),
                    scioxide_core::Dimensions::one(supplied),
                ));
            }
        }
        self.values.as_mut_slice().copy_from_slice(values);
        self.derivative.as_mut_slice().copy_from_slice(derivative);
        self.time = time;
        Ok(())
    }

    /// Returns the time, values and derivative.
    pub fn into_parts(self) -> (T::Real, Vector<T>, Vector<T>) {
        (self.time, self.values, self.derivative)
    }

    fn parts_mut(&mut self) -> (&mut T::Real, &mut [T], &mut [T]) {
        (
            &mut self.time,
            self.values.as_mut_slice(),
            self.derivative.as_mut_slice(),
        )
    }
}

/// Receives the state at each accepted step of a DAE integration.
///
/// [`DaeBdf::advance`] calls [`emit`](Self::emit) once per accepted step, after writing
/// the step to the [`DaeState`]; the initial pair is not emitted. `()` ignores the output,
/// and a closure `FnMut(t, y, ẏ) -> Result<(), SciOxideError>` is a sink.
pub trait DaeOutputSink<S: RealScalar, T = S> {
    /// Receives the time, values and derivative of an accepted step.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged; the state keeps the
    /// emitted step.
    fn emit(&mut self, time: S, values: &[T], derivative: &[T]) -> Result<(), SciOxideError>;
}

/// Ignores all output.
impl<S: RealScalar, T> DaeOutputSink<S, T> for () {
    fn emit(&mut self, _: S, _: &[T], _: &[T]) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<S: RealScalar, T, F> DaeOutputSink<S, T> for F
where
    F: FnMut(S, &[T], &[T]) -> Result<(), SciOxideError>,
{
    fn emit(&mut self, time: S, values: &[T], derivative: &[T]) -> Result<(), SciOxideError> {
        self(time, values, derivative)
    }
}

/// Holds the final state and report of [`DaeBdf::solve`].
#[derive(Debug, Clone)]
#[must_use]
pub struct DaeSolution<T: NumericScalar> {
    state: DaeState<T>,
    report: DaeReport<T::Real>,
}

impl<T: NumericScalar> DaeSolution<T> {
    /// Returns the last accepted pair, or the checked or projected initial pair when no
    /// step was accepted.
    pub const fn state(&self) -> &DaeState<T> {
        &self.state
    }
    /// Returns the report of the integration.
    pub const fn report(&self) -> &DaeReport<T::Real> {
        &self.report
    }
    /// Returns the final state and the report.
    pub fn into_parts(self) -> (DaeState<T>, DaeReport<T::Real>) {
        (self.state, self.report)
    }
}

/// Describes the storage a [`DaeBdf`] needs, as returned by
/// [`DaeBdf::workspace_requirements`].
///
/// A [`DaeWorkspace`] holds storage for one dimension and stage backend at a time.
#[derive(Debug, Clone, PartialEq)]
pub struct DaeRequirements {
    dimension: usize,
    backend: OdeStageBackend,
}

impl DaeRequirements {
    /// Returns the state dimension.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Holds the BDF history, Newton and linear-solver storage of a [`DaeBdf`], and the
/// record of the last call.
///
/// [`DaeWorkspace::new`] creates an empty workspace; [`DaeBdf::workspace`] and
/// [`Self::try_reserve`] allocate its storage. See
/// [plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct DaeWorkspace<T: NumericScalar> {
    pub(super) storage: Option<DaeStorage<T>>,
    shell: Shell<DaeReport<T::Real>>,
}

impl<T: NumericScalar> Default for DaeWorkspace<T>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NumericScalar> core::fmt::Debug for DaeWorkspace<T>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DaeWorkspace")
            .field(
                "dimension",
                &self.storage.as_ref().map_or(0, DaeStorage::dimension),
            )
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> DaeWorkspace<T>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            storage: None,
            shell: Shell::new(),
        }
    }

    fn fits(&self, dimension: usize, backend: &OdeStageBackend) -> bool
    where
        T::Real: NumericScalar<Real = T::Real>,
    {
        self.storage
            .as_ref()
            .is_some_and(|storage| storage.fits(dimension, backend))
    }

    /// Reports whether this workspace holds storage for `requirements`.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    pub fn check(&self, requirements: &DaeRequirements) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.fits(requirements.dimension, &requirements.backend) {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Allocates storage for `requirements`, replacing storage for another dimension or
    /// backend.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` or `SizeOverflow` if the storage cannot be allocated, or the
    /// error of preparing the stage backend.
    pub fn try_reserve(&mut self, requirements: &DaeRequirements) -> Result<(), SciOxideError> {
        self.reserve(requirements.dimension, &requirements.backend)
    }

    fn reserve(
        &mut self,
        dimension: usize,
        backend: &OdeStageBackend,
    ) -> Result<(), SciOxideError> {
        if !self.fits(dimension, backend) {
            self.storage = None;
            self.storage = Some(DaeStorage::new(dimension, backend)?);
        }
        Ok(())
    }

    /// Returns the record of the last call: its outcome, work counts and the report of
    /// its progress.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<DaeReport<T::Real>>> {
        self.shell.last_attempt()
    }
}

/// Integrates a fully implicit index-1 DAE with variable-step, variable-order BDF.
///
/// The problem is
///
/// ```text
/// F(t, y, ẏ) = 0,   y(t₀) = y₀,   ẏ(t₀) = ẏ₀,
/// ```
///
/// where `F` is a [`DaeSystem`] and a [`DaePartition`] marks the differential and
/// algebraic variables and equations. Each call first checks or projects the initial
/// pair, as set by [`DaeInitialization`](crate::DaeInitialization). It then steps with
/// the backward differentiation formulas of orders one to five on the actual,
/// variable step sizes, choosing order and step size from local error estimates. Each
/// step solves the BDF stage equation by Newton's method with the dense, sparse or
/// matrix-free linear solver of the [`OdeStageBackend`]. The error control is described
/// on [`DaeNumericalOptions`](crate::DaeNumericalOptions).
///
/// Every call starts at order one without history. The first step estimates the
/// differential error from the defect `(h/2)(ẋ₁ − ẋ₀)` of the predictor `x₀ + h ẋ₀`,
/// where `h` is the step, `x₀`, `ẋ₀` are the initial differential states and
/// derivatives and `x₁`, `ẋ₁` those after the step. The algebraic error is the response
/// of the linearized algebraic equations to that defect. The supplied algebraic
/// derivatives, on which `F` does not depend, never enter an error test. Local error
/// estimates control the step; they do not bound the global error.
///
/// [`Self::advance`] continues from a [`DaeState`] you own and sends each accepted step
/// to a [`DaeOutputSink`]. [`Self::solve`] allocates a workspace and returns the final
/// state. A plan is immutable, `Send + Sync + Clone`, and can be shared between calls.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError, SolveContext};
/// use scioxide_ode::{
///     DaeOptions, DaePartition, DaeBdf, DaeProblem, DaeState, DaeSystem, OdeStageBackend,
/// };
///
/// // x′ = −x with the algebraic constraint z = x; x(t) = z(t) = e^(−t).
/// struct Decay;
///
/// impl DaeSystem<f64> for Decay {
///     fn dimension(&self) -> usize {
///         2
///     }
///
///     fn residual(
///         &self,
///         _: f64,
///         y: &[f64],
///         dy: &[f64],
///         out: &mut [f64],
///         _: &mut SolveContext<'_, '_>,
///     ) -> Result<(), SciOxideError> {
///         out[0] = dy[0] + y[0];
///         out[1] = y[1] - y[0];
///         Ok(())
///     }
/// }
///
/// let partition = DaePartition::new(&[true, false], &[true, false])?;
/// let options = DaeOptions::new().with_initial_step(0.1);
/// let plan = (DaeBdf::new(partition, OdeStageBackend::Dense)?, options);
/// let problem = DaeProblem::new(&Decay, 1.0);
/// let mut workspace = plan.0.workspace()?;
/// let mut state = DaeState::from_slices(0.0, &[1.0, 1.0], &[-1.0, 0.0])?;
/// let cx = ExecutionContext::sequential_native();
/// let report = plan.0.advance(&problem, &plan.1, &mut state, &mut (), &mut workspace, cx)?;
/// assert!(report.is_success());
/// assert!((state.values()[0] - (-1.0_f64).exp()).abs() < 1e-6);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - K. E. Brenan, S. L. Campbell and L. R. Petzold, *Numerical Solution of
///   Initial-Value Problems in Differential-Algebraic Equations*, SIAM, 1996.
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 2nd ed.,
///   Springer, 1996.
#[derive(Debug, Clone)]
#[must_use]
pub struct DaeBdf {
    partition: DaePartition,
    backend: OdeStageBackend,
    operator_scratch: usize,
}

impl DaeBdf {
    /// Creates a BDF method for `partition` with the linear solver `backend`.
    ///
    /// A sparse backend's pattern must cover the nonzeros of both `∂F/∂y` and `∂F/∂ẏ`
    /// and every diagonal entry. The operator scratch of a Krylov backend is the length
    /// your [`DaeSystem::derivative_product`] needs; the plan adds its own buffers. The
    /// Krylov backend never forms the Jacobian.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a sparse pattern has the wrong shape or lacks a diagonal entry,
    ///   or the backend configuration is invalid.
    /// - `SizeOverflow` if the Krylov scratch length overflows `usize`.
    pub fn new(
        partition: DaePartition,
        mut backend: OdeStageBackend,
    ) -> Result<Self, SciOxideError> {
        let n = partition.dimension();
        backend.validate(n)?;
        let operator_scratch = if let OdeStageBackend::Krylov {
            operator_scratch, ..
        } = &mut backend
        {
            let provider = *operator_scratch;
            *operator_scratch = provider
                .checked_add(scioxide_core::backend::checked_mul(
                    n,
                    6,
                    scioxide_core::quantity!("direction_scratch"),
                )?)
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("operator_scratch"))
                })?;
            *operator_scratch
        } else {
            0
        };
        Ok(Self {
            partition,
            backend,
            operator_scratch,
        })
    }

    /// Returns the variable and equation roles.
    pub const fn partition(&self) -> &DaePartition {
        &self.partition
    }
    /// Returns the number of equations and state components.
    pub fn dimension(&self) -> usize {
        self.partition.dimension()
    }

    pub(crate) const fn backend(&self) -> &OdeStageBackend {
        &self.backend
    }

    pub(crate) const fn operator_mode(&self) -> bool {
        matches!(self.backend, OdeStageBackend::Krylov { .. })
    }
    pub(crate) const fn operator_scratch(&self) -> usize {
        self.operator_scratch
    }
    /// Fresh storage for this plan.
    pub(crate) fn storage<T: NumericScalar>(&self) -> Result<DaeStorage<T>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
    {
        DaeStorage::new(self.dimension(), &self.backend)
    }
    /// Whether `storage` serves this plan.
    pub(crate) fn fits<T: NumericScalar>(&self, storage: &DaeStorage<T>) -> bool {
        storage.fits(self.dimension(), &self.backend)
    }

    /// Returns the storage that every integration with this plan needs.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    pub fn workspace_requirements(&self) -> Result<DaeRequirements, SciOxideError> {
        Ok(DaeRequirements {
            dimension: self.dimension(),
            backend: self.backend.clone(),
        })
    }

    /// Allocates a workspace for this plan.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` or `SizeOverflow` if the storage cannot be allocated, or the
    /// error of preparing the stage backend.
    pub fn workspace<T: NumericScalar>(&self) -> Result<DaeWorkspace<T>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
    {
        let mut workspace = DaeWorkspace::new();
        workspace.reserve(self.dimension(), &self.backend)?;
        Ok(workspace)
    }

    /// Shape, time and finiteness checks; grows an undersized workspace.
    fn preflight<T: NumericScalar, Sys: DaeSystem<T> + ?Sized>(
        &self,
        problem: &DaeProblem<'_, T, Sys>,
        options: &DaeOptions<T::Real>,
        state: &DaeState<T>,
        workspace: &mut DaeWorkspace<T>,
    ) -> Result<(), SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
    {
        options.validate(&self.partition)?;
        let n = self.dimension();
        let system = problem.system().dimension();
        if system != n || state.values().len() != n || state.derivative().len() != n {
            return Err(invalid(
                if system != n {
                    scioxide_core::argument!("problem")
                } else if state.values().len() != n {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("derivative")
                },
                "state/provider/workspace dimension mismatch",
            ));
        }
        let (start, end) = (state.time(), problem.end());
        if !start.is_finite() || !end.is_finite() || !(end - start).is_finite() {
            return Err(invalid(
                if start.is_finite() {
                    scioxide_core::argument!("end")
                } else {
                    scioxide_core::argument!("start")
                },
                "finite time span required",
            ));
        }
        if !state
            .values()
            .iter()
            .chain(state.derivative())
            .all(|value| value.has_finite_components())
        {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "initial state and derivative must be finite",
            ));
        }
        workspace.reserve(n, &self.backend)
    }

    /// Checks or projects the initial pair in `state`, then integrates `problem` to its
    /// end time.
    ///
    /// The initialization runs at the start of every call. If `problem.end()` equals
    /// `state.time()`, the call only initializes and returns `ReachedEnd`. Every call
    /// starts with fresh BDF history. Until the initial pair is accepted `state` is
    /// unchanged; afterwards it holds the last accepted pair, and `sink` has received
    /// every accepted step. A workspace for another dimension or backend is reallocated
    /// first. Stepping is sequential. `scope` is an
    /// [`ExecutionContext`](scioxide_core#execution-contexts) or the [`SolveContext`] of an
    /// enclosing solve, whose cancellation, deadline and budgets then apply (see
    /// [solve accounting](scioxide_core#solve-accounting)).
    ///
    /// After the initial pair is accepted, a step or evaluation budget, the rejection
    /// limit, a step size too small to advance, cancellation or a deadline ends the call
    /// with `Ok` and an unsuccessful [`DaeReport`]. A stop before that point is an error.
    /// [`DaeWorkspace::last_attempt`] records the progress of every call. With a large
    /// enough workspace, and callbacks and a sink that do not allocate, the call does not
    /// allocate.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the system, values or derivative differ in dimension from the
    ///   plan, or the start or end time or their difference is not finite.
    /// - `NonFinite` (origin `Input`) if the initial values or derivative are not finite.
    /// - `InvalidInput` (`Inconsistent`) if [`DaeInitialization::Check`] finds a residual
    ///   block above its tolerance.
    /// - `ConvergenceFailure` if [`DaeInitialization::Project`] does not converge; the
    ///   error carries the Newton reason and iteration count.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a budget, cancellation or a
    ///   deadline stops the call before the initial pair is accepted.
    /// - Errors returned by the system or the sink, unchanged.
    ///
    /// [`DaeInitialization::Check`]: crate::DaeInitialization::Check
    /// [`DaeInitialization::Project`]: crate::DaeInitialization::Project
    pub fn advance<'scope, 'capability, T: NumericScalar, Sys, K>(
        &self,
        problem: &DaeProblem<'_, T, Sys>,
        options: &DaeOptions<T::Real>,
        state: &mut DaeState<T>,
        sink: &mut K,
        workspace: &mut DaeWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DaeReport<T::Real>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Sys: DaeSystem<T> + ?Sized,
        K: DaeOutputSink<T::Real, T> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, options, state, workspace) {
            return workspace.shell.reject(error);
        }
        let DaeWorkspace { storage, shell } = workspace;
        let Some(storage) = storage.as_mut() else {
            return shell.reject(invalid(
                scioxide_core::argument!("workspace"),
                "DAE workspace storage is not prepared",
            ));
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "index1-bdf", 1),
            self.dimension() as u128,
            0,
        );
        let (start, end) = (state.time(), problem.end());
        let mut report = DaeReport::initial(start);
        let executed = shell.execute(scope, identity, options.budgets(), |cx| {
            let result = self.run(
                problem.system(),
                options,
                state,
                sink,
                storage,
                (start, end),
                cx,
                &mut report,
            );
            report.counts = cx.counts();
            match result {
                Ok(()) => {
                    report.termination =
                        Some(report.termination.unwrap_or(DaeTermination::ReachedEnd));
                    Ok(report)
                }
                // Only a stop that leaves the accepted initial pair, or a later one, is a
                // termination; earlier stops fail like any other error.
                Err(fault) => match fault.stop() {
                    Some(stop) if report.has_accepted_state => {
                        report.termination = Some(DaeTermination::from_stop(stop));
                        Ok(report)
                    }
                    _ => Err(fault.into()),
                },
            }
        });
        shell.conclude(executed, DaeReport::is_success, |_| Some(report))
    }

    #[allow(clippy::too_many_arguments)]
    fn run<T: NumericScalar, Sys, K>(
        &self,
        system: &Sys,
        options: &DaeOptions<T::Real>,
        state: &mut DaeState<T>,
        sink: &mut K,
        storage: &mut DaeStorage<T>,
        (start, end): (T::Real, T::Real),
        cx: &mut SolveContext<'_, '_>,
        report: &mut DaeReport<T::Real>,
    ) -> Result<(), Fault>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Sys: DaeSystem<T> + ?Sized,
        K: DaeOutputSink<T::Real, T> + ?Sized,
    {
        let (time, values, derivative) = state.parts_mut();
        storage
            .project(
                (self, options),
                system,
                values,
                derivative,
                start,
                cx,
                report,
            )?
            .map_err(super::Unaccepted::error)?;
        let step = match options.initial_step() {
            Some(step) => step,
            None => storage.first_step((self, options), start, end)?,
        };
        storage.run_steps(
            (self, options),
            system,
            start,
            end,
            step,
            cx,
            &mut |t, y, d| {
                values.copy_from_slice(y);
                derivative.copy_from_slice(d);
                *time = t;
                sink.emit(t, y, d)
            },
            report,
            &mut (),
        )
    }

    /// Integrates `problem` from `initial` with a new workspace on the calling thread and
    /// returns the final state.
    ///
    /// Intermediate steps are discarded. It runs [`Self::advance`] with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::advance`], wrapped in a
    /// [`PartialFailure`] together with the progress of the failed call.
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    pub fn solve<T: NumericScalar, Sys>(
        &self,
        problem: &DaeProblem<'_, T, Sys>,
        options: &DaeOptions<T::Real>,
        initial: DaeState<T>,
    ) -> Result<DaeSolution<T>, PartialFailure<DaeReport<T::Real>>>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Sys: DaeSystem<T> + ?Sized,
    {
        let mut workspace = self.workspace().map_err(PartialFailure::new)?;
        let mut state = initial;
        match self.advance(
            problem,
            options,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(DaeSolution { state, report }),
            Err(error) => Err(workspace.shell.into_failure(error)),
        }
    }
}
