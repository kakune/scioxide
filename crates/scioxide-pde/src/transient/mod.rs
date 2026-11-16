//! Transient PDE plans: progressive advances of `M u' = b(t) − A u` (parabolic) and
//! `M u'' = b(t) − A u` (hyperbolic) through the single method-of-lines adapter.
//!
//! An advance runs in at most two integration phases: Rannacher startup (fixed
//! backward-Euler half steps over `[t, min(n·h, end)]`) and the main method to the end
//! time. Requested output times are split between the phases by one cursor, and fully
//! constrained systems only reconstruct and emit their boundary data.

mod options;
mod output;
#[cfg(test)]
mod tests;

pub use options::{HyperbolicOptions, ParabolicOptions, ParabolicScheme, RannacherStartup};
pub use output::{PdeObservation, PdeOutputSchedule, PdeOutputSink, PdeTrajectory};

use alloc::vec::Vec;
use core::fmt;

use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, ErrorTolerance, EvaluationCounts, EvaluationRole,
    ExecutionContext, NumericScalar, PartialFailure, PartialResultTrust, RealScalar, SciOxideError,
    SolveContext, SolveScope,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    BuiltinOdeMethod, OdeLinearStability, OdeMethod, OdeOptions, OdeProblem, OdeState, OdeSystem,
    OdeWorkspace,
};
use scioxide_spatial::MeshToken;

use crate::attempt::{Shell, identity};
use crate::boundary::FieldData;
use crate::discretization::SpatialDiscretization;
use crate::mol::{
    MethodOfLinesOrder, MethodOfLinesPlan, MethodOfLinesSystem, MethodOfLinesWorkspace,
};
use crate::result::{PdeDiagnostics, PdeSolveReport, PdeTermination};
use crate::semidiscrete::SemidiscreteSystem;
use crate::validation::invalid;
use output::{PhaseSink, Requests, emit, phase_schedule};

/// A transient problem: a prepared discretization, its source and the end time.
///
/// Construction prepares the free-unknown semidiscrete system once (eliminated
/// Jacobian, lumped mass and pattern); every advance reuses it.
#[must_use]
struct PreparedProblem<'a, S: RealScalar, P, const D: usize> {
    system: SemidiscreteSystem<'a, S, P, D>,
    end: S,
}

impl<'a, S, P, const D: usize> PreparedProblem<'a, S, P, D>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: SpatialDiscretization<S, S, D>,
{
    /// Prepare `discretization` with `source` for integration to `end`.
    ///
    /// # Errors
    /// Rejects a non-finite or non-positive end time and propagates assembly and
    /// allocation failures.
    fn new(
        discretization: &'a P,
        source: &'a FieldData<S, S, D>,
        end: S,
    ) -> Result<Self, SciOxideError> {
        if !end.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "end time must be finite",
            ));
        }
        if end <= S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("end"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "end time must be positive",
            ));
        }
        Ok(Self {
            system: SemidiscreteSystem::new(discretization, source)?,
            end,
        })
    }

    /// End time.
    const fn end(&self) -> S {
        self.end
    }

    /// The prepared discretization.
    const fn discretization(&self) -> &'a P {
        self.system.plan()
    }

    /// The prepared free-unknown semidiscrete system.
    const fn system(&self) -> &SemidiscreteSystem<'a, S, P, D> {
        &self.system
    }
}

/// Caller-owned state of a transient advance.
///
/// It holds the time, the ODE coordinates (free unknowns, followed by their
/// velocities for hyperbolic plans), the end of any pending Rannacher startup and
/// the exact mesh revision and free-node layout. A preflight rejection leaves it unchanged; after any
/// accepted step it holds the last accepted point.
///
/// The pending startup end is fixed by the options passed to `initial_state`. An
/// advance or resume rejects the state with `InvalidInput` unless it matches the
/// startup its own options imply at the state's time: the end `n·h` while the time
/// is before it, and none otherwise.
#[derive(Debug, Clone)]
#[must_use]
pub struct PdeState<S: RealScalar + NumericScalar<Real = S>> {
    ode: OdeState<S>,
    free: Vec<usize>,
    full: usize,
    startup_end: Option<S>,
    mesh_token: MeshToken,
    order: MethodOfLinesOrder,
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeState<S> {
    /// Time of the state.
    pub const fn time(&self) -> S {
        self.ode.time()
    }

    /// ODE coordinates in free-unknown order (`[u, v]` for second order).
    pub fn coordinates(&self) -> &[S] {
        self.ode.values()
    }

    /// End of pending Rannacher startup, absent once startup is complete or disabled.
    pub const fn startup_end(&self) -> Option<S> {
        self.startup_end
    }

    /// Exact mesh revision of the discretization.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.mesh_token
    }

    /// First-order (parabolic) or second-order (hyperbolic) dynamics.
    pub const fn order(&self) -> MethodOfLinesOrder {
        self.order
    }

    fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            ode: OdeState::from_slice(self.time(), self.coordinates())?,
            free: scioxide_core::backend::try_copy_slice(
                &self.free,
                scioxide_core::quantity!("state_layout"),
            )?,
            full: self.full,
            startup_end: self.startup_end,
            mesh_token: self.mesh_token.clone(),
            order: self.order,
        })
    }

    /// Overwrite with `other`, reusing storage of equal length. On error `self` is unchanged.
    fn restore(&mut self, other: &Self) -> Result<(), SciOxideError> {
        if self.coordinates().len() == other.coordinates().len()
            && self.free.len() == other.free.len()
        {
            self.ode.reset(other.time(), other.coordinates())?;
            self.free.copy_from_slice(&other.free);
            self.full = other.full;
            self.startup_end = other.startup_end;
            self.mesh_token.clone_from(&other.mesh_token);
            self.order = other.order;
        } else {
            *self = other.try_clone()?;
        }
        Ok(())
    }
}

/// An owned snapshot of a [`PdeState`] for later [`ParabolicProblem::resume`] or
/// [`HyperbolicProblem::resume`].
///
/// Resume validates the snapshot's own structural data (mesh revision, free-node layout, order,
/// dimension, time and startup phase) before restoring it.
#[derive(Debug, Clone)]
#[must_use]
pub struct PdeCheckpoint<S: RealScalar + NumericScalar<Real = S>> {
    state: PdeState<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeCheckpoint<S> {
    /// Copy `state`.
    ///
    /// # Errors
    /// Returns an allocation failure.
    pub fn new(state: &PdeState<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            state: state.try_clone()?,
        })
    }

    /// Time of the snapshot.
    pub const fn time(&self) -> S {
        self.state.time()
    }

    /// Exact mesh revision of the snapshot.
    pub const fn mesh_token(&self) -> &MeshToken {
        self.state.mesh_token()
    }

    /// The snapshot state.
    pub const fn state(&self) -> &PdeState<S> {
        &self.state
    }
}

/// Accepted progress of one transient attempt.
///
/// Recorded in [`PdeWorkspace::last_attempt`] and carried by the owned
/// conveniences' [`PartialFailure`].
#[derive(Debug, Clone)]
pub struct PdeProgress<S: RealScalar + NumericScalar<Real = S>> {
    state: PdeState<S>,
    steps: EvaluationCounts,
    last_step_error_estimate: Option<S>,
    trust: PartialResultTrust,
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeProgress<S> {
    /// Last accepted state.
    pub const fn state(&self) -> &PdeState<S> {
        &self.state
    }

    /// Time of the last accepted state.
    pub const fn time(&self) -> S {
        self.state.time()
    }

    /// Attempted, accepted and rejected time steps of the attempt.
    pub const fn steps(&self) -> EvaluationCounts {
        self.steps
    }

    /// Normalized error estimate of the last attempted step.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }

    /// Trust of the last accepted state.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }

    /// The last accepted state, ready to continue with `advance`.
    pub fn into_state(self) -> PdeState<S> {
        self.state
    }
}

/// Reusable integration storage, solve state and last attempt of transient plans.
///
/// [`Self::new`] allocates nothing; an advance grows undersized storage during
/// preflight, before any callback. One workspace serves parabolic and hyperbolic
/// plans; it keeps the storage of its last problem.
pub struct PdeWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    /// The ODE workspace lives on the heap (zero or one element, grown fallibly)
    /// so that this workspace stays small to move and to hold on the stack.
    ode: Vec<OdeWorkspace<S, BuiltinOdeMethod<S>>>,
    mol: Option<MethodOfLinesWorkspace<S>>,
    field: Vec<S>,
    shell: Shell<PdeProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for PdeWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> fmt::Debug for PdeWorkspace<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PdeWorkspace")
            .field("field_len", &self.field.len())
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeWorkspace<S> {
    /// An empty workspace; allocates nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ode: Vec::new(),
            mol: None,
            field: Vec::new(),
            shell: Shell::new(),
        }
    }

    /// Record of the most recent advance or resume.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<PdeProgress<S>>> {
        self.shell.last_attempt()
    }

    /// State accepted by the most recent attempt only.
    pub fn last_accepted(&self) -> Option<&PdeState<S>> {
        self.last_attempt()
            .and_then(AttemptRecord::progress)
            .map(PdeProgress::state)
    }

    /// Grow the adapter and field storage for `adapter` and `field_len`.
    fn prepare(
        &mut self,
        adapter: &MethodOfLinesPlan<S>,
        field_len: usize,
    ) -> Result<(), SciOxideError> {
        if self.ode.is_empty() {
            self.ode.try_reserve_exact(1).map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("pde_ode_workspace"), 1)
            })?;
            self.ode.push(OdeWorkspace::new());
        }
        if !self.mol.as_ref().is_some_and(|work| work.fits(adapter)) {
            self.mol = Some(adapter.workspace()?);
        }
        if let Some(mol) = &mut self.mol {
            mol.reserve_field(field_len)?;
        }
        if self.field.len() != field_len {
            self.field = crate::discretization::zeros(field_len)?;
        }
        Ok(())
    }
}

/// Owned result of a transient `solve`.
#[derive(Debug, Clone)]
#[must_use]
pub struct PdeSolution<S: RealScalar + NumericScalar<Real = S>> {
    state: PdeState<S>,
    values: Vec<S>,
    report: PdeSolveReport<S>,
    trajectory: PdeTrajectory<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> PdeSolution<S> {
    /// Final accepted state.
    pub const fn state(&self) -> &PdeState<S> {
        &self.state
    }

    /// Full physical field of the final state, in discretization unknown order.
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Termination, diagnostics and trust.
    pub const fn report(&self) -> &PdeSolveReport<S> {
        &self.report
    }

    /// Samples emitted under the configured schedule.
    pub const fn trajectory(&self) -> &PdeTrajectory<S> {
        &self.trajectory
    }

    /// Separate the state, field, report and trajectory.
    pub fn into_parts(self) -> (PdeState<S>, Vec<S>, PdeSolveReport<S>, PdeTrajectory<S>) {
        (self.state, self.values, self.report, self.trajectory)
    }
}

/// Work and trust accumulated by one attempt's integration phases.
///
/// `recorded` marks that an ODE phase began its epoch: from then on the state is an
/// accepted point (the initial state when no step was accepted yet) and failures
/// carry it as progress, exactly as scioxide-ode records its own attempts.
#[derive(Debug, Clone, Copy)]
struct Tally<S> {
    steps: EvaluationCounts,
    last_step_error_estimate: Option<S>,
    trust: PartialResultTrust,
    recorded: bool,
}

impl<S: Copy> Tally<S> {
    const fn new() -> Self {
        Self {
            steps: EvaluationCounts::new(),
            last_step_error_estimate: None,
            trust: PartialResultTrust::Admissible,
            recorded: false,
        }
    }

    fn record(
        &mut self,
        steps: EvaluationCounts,
        estimate: Option<S>,
        trust: PartialResultTrust,
    ) -> Result<(), SciOxideError> {
        self.steps = self.steps.merge(steps)?;
        self.last_step_error_estimate = estimate;
        self.trust = trust;
        self.recorded = true;
        Ok(())
    }
}

/// Main and Rannacher startup methods; both absent without free dynamics.
type PhaseMethods<S> = (Option<BuiltinOdeMethod<S>>, Option<BuiltinOdeMethod<S>>);

/// Validated, callback-free preparation of one advance.
struct Prepared<S: RealScalar + NumericScalar<Real = S>> {
    adapter: MethodOfLinesPlan<S>,
    tolerance: ErrorTolerance<S>,
    main: Option<BuiltinOdeMethod<S>>,
    startup: Option<BuiltinOdeMethod<S>>,
}

/// Per-advance borrowed integration resources.
struct Run<'r, 's, 'b, S: RealScalar + NumericScalar<Real = S>, K: ?Sized> {
    system: &'s MethodOfLinesSystem<'b, S>,
    field: &'r mut [S],
    sink: &'r mut K,
    ode: &'r mut OdeWorkspace<S, BuiltinOdeMethod<S>>,
    tally: &'r mut Tally<S>,
    last_emitted: Option<S>,
}

/// The configuration shared by parabolic and hyperbolic plans.
#[derive(Debug, Clone)]
struct Transient<'o, S: RealScalar + NumericScalar<Real = S>> {
    options: &'o ParabolicOptions<S>,
    order: MethodOfLinesOrder,
}

/// Admit plan preparation as one provider evaluation before integration.
fn admit_preparation(context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    if let Some(reason) = context.stop_reason()? {
        return Err(SciOxideError::execution_stopped(reason));
    }
    if !context
        .record_provider(EvaluationRole::Function)?
        .is_granted()
    {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::EvaluationLimit,
            scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
            "preparation budget exhausted",
        ));
    }
    Ok(())
}

/// Copy the attempt's accepted point into recycled (or new) progress storage.
fn progress<S: RealScalar + NumericScalar<Real = S>>(
    recycled: Option<PdeProgress<S>>,
    state: &PdeState<S>,
    tally: &Tally<S>,
) -> Option<PdeProgress<S>> {
    let reused = recycled.and_then(|mut progress| {
        progress.state.restore(state).ok()?;
        Some(progress)
    });
    let mut progress = match reused {
        Some(progress) => progress,
        None => PdeProgress {
            state: state.try_clone().ok()?,
            steps: EvaluationCounts::new(),
            last_step_error_estimate: None,
            trust: PartialResultTrust::DiagnosticOnly,
        },
    };
    progress.steps = tally.steps;
    progress.last_step_error_estimate = tally.last_step_error_estimate;
    progress.trust = tally.trust;
    Some(progress)
}

impl<S: RealScalar + NumericScalar<Real = S>> Transient<'_, S> {
    const fn blocks(&self) -> usize {
        match self.order {
            MethodOfLinesOrder::First => 1,
            MethodOfLinesOrder::Second => 2,
        }
    }

    /// End of Rannacher startup, `n·h`, measured from time zero.
    fn startup_end(&self) -> Result<Option<S>, SciOxideError> {
        match self.options.startup() {
            RannacherStartup::Disabled => Ok(None),
            RannacherStartup::BackwardEuler { intervals } => {
                Ok(Some(scioxide_core::checked_float_product(
                    S::try_from_usize_exact(intervals.get())?,
                    self.options.step().ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("options.step"),
                            "startup requires a step",
                        )
                    })?,
                )?))
            }
        }
    }

    fn method_name(&self) -> &'static str {
        self.options
            .ode_method()
            .map_or(self.options.scheme().name(), BuiltinOdeMethod::name)
    }

    fn stability(&self) -> OdeLinearStability<S> {
        match self.options.ode_method() {
            Some(method) => method.linear_stability(),
            None => match self.options.scheme() {
                ParabolicScheme::ExplicitDopri5 => {
                    scioxide_ode::DormandPrince5::<S>::new().linear_stability()
                }
                ParabolicScheme::BackwardEuler
                | ParabolicScheme::CrankNicolson
                | ParabolicScheme::RadauIia => OdeLinearStability::unconditional(),
            },
        }
    }

    fn initial_state<P: SpatialDiscretization<S, S, D>, const D: usize>(
        &self,
        problem: &PreparedProblem<'_, S, P, D>,
        initial: &FieldData<S, S, D>,
        velocity: Option<&FieldData<S, S, D>>,
    ) -> Result<PdeState<S>, SciOxideError> {
        let mut values = problem.system.initial_state(initial)?;
        if let Some(velocity) = velocity {
            let velocity = problem.system.initial_state(velocity)?;
            values.try_reserve_exact(velocity.len()).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("pde_initial_velocity"),
                    velocity.len(),
                )
            })?;
            values.extend_from_slice(&velocity);
        }
        Ok(PdeState {
            ode: OdeState::new(S::ZERO, Vector::from_vec(values)),
            free: scioxide_core::backend::try_copy_slice(
                problem.system.free_indices(),
                scioxide_core::quantity!("state_layout"),
            )?,
            full: problem.discretization().unknowns(),
            startup_end: self.startup_end()?,
            mesh_token: problem.discretization().mesh_token().clone(),
            order: self.order,
        })
    }

    fn workspace<P: SpatialDiscretization<S, S, D>, const D: usize>(
        &self,
        problem: &PreparedProblem<'_, S, P, D>,
    ) -> Result<PdeWorkspace<S>, SciOxideError> {
        let mut workspace = PdeWorkspace::new();
        workspace.prepare(
            &problem.system.method_of_lines(self.order)?,
            problem.discretization().unknowns(),
        )?;
        Ok(workspace)
    }

    /// Check `state` against `problem` and the plan, and prepare everything that
    /// needs no callback: the step-limited adapter, the normalized tolerance and the
    /// phase methods.
    fn preflight<P: SpatialDiscretization<S, S, D>, const D: usize>(
        &self,
        problem: &PreparedProblem<'_, S, P, D>,
        state: &PdeState<S>,
    ) -> Result<Prepared<S>, SciOxideError> {
        self.options.validate()?;
        let semi = &problem.system;
        let plan = semi.plan();
        let argument = scioxide_core::argument!("state");
        if !state.mesh_token.is_compatible_with(plan.mesh_token()) {
            return Err(invalid(argument, "state belongs to another mesh revision"));
        }
        if state.order != self.order {
            return Err(invalid(argument, "state order differs from the plan"));
        }
        let free = semi.free_indices();
        if state.free != free || state.full != plan.unknowns() {
            return Err(invalid(
                argument,
                "state free-unknown layout differs from the plan",
            ));
        }
        let blocks = self.blocks();
        crate::validation::shape(state.coordinates().len(), free.len() * blocks)?;
        let time = state.time();
        if !time.is_finite() || time < S::ZERO || time > problem.end {
            return Err(invalid(argument, "state time must lie within [0, end]"));
        }
        crate::validation::finite_slice(
            state.coordinates(),
            scioxide_core::NonFiniteOrigin::Input,
        )?;
        if state.startup_end != self.startup_end()?.filter(|&end| time < end) {
            return Err(invalid(
                argument,
                "state startup phase differs from the plan",
            ));
        }
        if !Requests::new(self.options.output_schedule(), time).within(problem.end) {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "requested PDE times must lie within the solve span",
            ));
        }
        let field_len = plan.unknowns();
        let tolerance = self.options.error_tolerance();
        tolerance.absolute().component(0, field_len * blocks)?;
        let mut absolute = crate::discretization::reserve(free.len() * blocks)?;
        for block in 0..blocks {
            for &i in free {
                absolute.push(
                    tolerance
                        .absolute()
                        .component(block * field_len + i, field_len * blocks)?,
                );
            }
        }
        let physical =
            ErrorTolerance::try_vector(&absolute, tolerance.relative(), tolerance.norm())?;
        let adapter = semi
            .method_of_lines(self.order)?
            .with_step_limit(self.options.cfl_policy(), self.stability())?;
        let (main, startup) = if free.is_empty() {
            (None, None)
        } else {
            self.methods(semi)?
        };
        if matches!(self.options.output_schedule(), PdeOutputSchedule::Times(_)) {
            // Requested times are interpolated from dense output, which the built-in
            // implicit schemes and Rannacher startup do not provide.
            for method in main.iter().chain(&startup) {
                method.validate(
                    &(OdeOptions::new()
                        .with_output_schedule(scioxide_ode::OdeOutputSchedule::times(Vec::new()))),
                    free.len() * blocks,
                )?;
            }
        }
        Ok(Prepared {
            tolerance: adapter.error_tolerance(&physical)?,
            adapter,
            main,
            startup,
        })
    }

    /// Main and startup methods of a system with free dynamics.
    fn methods<P: SpatialDiscretization<S, S, D>, const D: usize>(
        &self,
        semi: &SemidiscreteSystem<'_, S, P, D>,
    ) -> Result<PhaseMethods<S>, SciOxideError> {
        let options = &self.options;
        if let Some(method) = options.ode_method() {
            return Ok((Some(method.clone()), None));
        }
        let backend = || match options.stage_backend() {
            Some(backend) => Ok(backend.clone()),
            None => crate::semidiscrete::sparse_backend(semi.jacobian_pattern()),
        };
        let main: BuiltinOdeMethod<S> = match options.scheme() {
            ParabolicScheme::ExplicitDopri5 => scioxide_ode::DormandPrince5::new().into(),
            ParabolicScheme::RadauIia => scioxide_ode::RadauIia::new()
                .with_backend(backend()?)
                .into(),
            ParabolicScheme::BackwardEuler => scioxide_ode::BackwardEuler::new()
                .with_backend(backend()?)
                .into(),
            ParabolicScheme::CrankNicolson => scioxide_ode::ImplicitTrapezoid::new()
                .with_backend(backend()?)
                .into(),
        };
        let startup = match options.startup() {
            RannacherStartup::Disabled => None,
            RannacherStartup::BackwardEuler { .. } => Some(
                scioxide_ode::BackwardEuler::new()
                    .with_backend(backend()?)
                    .into(),
            ),
        };
        Ok((Some(main), startup))
    }

    /// Validate, record and execute one attempt; a checkpoint makes it a resume.
    fn run<P, K, const D: usize>(
        &self,
        problem: &PreparedProblem<'_, S, P, D>,
        checkpoint: Option<&PdeCheckpoint<S>>,
        state: &mut PdeState<S>,
        sink: &mut K,
        workspace: &mut PdeWorkspace<S>,
        scope: SolveScope<'_, '_>,
    ) -> Result<PdeSolveReport<S>, SciOxideError>
    where
        P: SpatialDiscretization<S, S, D>,
        K: PdeOutputSink<S> + ?Sized,
    {
        let kind = if checkpoint.is_some() {
            AttemptKind::Resume
        } else {
            AttemptKind::Call
        };
        let prepared =
            match self.preflight(problem, checkpoint.map_or(&*state, PdeCheckpoint::state)) {
                Ok(prepared) => prepared,
                Err(error) => return workspace.shell.reject(kind, error),
            };
        if let Err(error) =
            workspace.prepare(&prepared.adapter, problem.discretization().unknowns())
        {
            return workspace.shell.reject(kind, error);
        }
        let PdeWorkspace {
            ode,
            mol,
            field,
            shell,
        } = workspace;
        let start = state.time();
        let mut tally = Tally::new();
        let executed = shell.execute(
            scope,
            identity(
                scioxide_core::algorithm_id!("pde", "method-of-lines", 4),
                state.coordinates().len() as u128,
            ),
            self.options.budgets(),
            |context| {
                admit_preparation(context)?;
                if let Some(saved) = checkpoint {
                    state.restore(&saved.state)?;
                }
                let (Some(mol), Some(ode)) = (mol.as_mut(), ode.first_mut()) else {
                    return Err(invalid(
                        scioxide_core::argument!("workspace"),
                        "adapter or ODE storage is missing",
                    ));
                };
                self.integrate(
                    problem, &prepared, state, sink, ode, mol, field, &mut tally, context,
                )
            },
        );
        state.startup_end = state.startup_end.filter(|&end| state.time() < end);
        let accepted = tally.recorded || state.time() != start;
        shell.conclude(kind, executed, PdeSolveReport::is_success, |recycled| {
            accepted
                .then(|| progress(recycled, state, &tally))
                .flatten()
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the epoch body borrows each workspace part and the caller state separately"
    )]
    fn integrate<P, K, const D: usize>(
        &self,
        problem: &PreparedProblem<'_, S, P, D>,
        prepared: &Prepared<S>,
        state: &mut PdeState<S>,
        sink: &mut K,
        ode: &mut OdeWorkspace<S, BuiltinOdeMethod<S>>,
        mol: &mut MethodOfLinesWorkspace<S>,
        field: &mut [S],
        tally: &mut Tally<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PdeSolveReport<S>, SciOxideError>
    where
        P: SpatialDiscretization<S, S, D>,
        K: PdeOutputSink<S> + ?Sized,
    {
        let semi = &problem.system;
        let system = prepared.adapter.bind_with_fields(semi, semi, mol)?;
        let output = self.options.output_schedule();
        let mut requests = Requests::new(output, state.time());
        let mut run = Run {
            system: &system,
            field,
            sink,
            ode,
            tally,
            last_emitted: None,
        };
        let termination = if system.dimension() == 0 {
            constrained_only(
                &mut run,
                state,
                problem.end,
                output,
                &mut requests,
                context.execution(),
            )?
        } else {
            self.phases(
                prepared,
                &mut run,
                state,
                problem.end,
                &mut requests,
                context,
            )?
        };
        let time = state.time();
        if matches!(output, PdeOutputSchedule::FinalOnly) {
            emit(
                &system,
                run.field,
                run.sink,
                time,
                state.coordinates(),
                None,
            )?;
        } else {
            system.reconstruct_into(time, state.coordinates(), run.field)?;
        }
        let work = context.counts().total()?;
        let mut counts = run.tally.steps;
        for role in [
            EvaluationRole::RightHandSide,
            EvaluationRole::Jacobian,
            EvaluationRole::Factorization,
            EvaluationRole::LinearSolve,
        ] {
            counts = counts.with_provider_count(role, work.provider(role));
        }
        Ok(PdeSolveReport::new(
            PdeDiagnostics::from_ode(
                self.method_name(),
                termination,
                time,
                run.tally.last_step_error_estimate,
                counts,
                scioxide_core::argument!("options"),
            )?,
            run.tally.trust,
        ))
    }

    /// Rannacher startup (when pending), then the main method to `end`.
    fn phases<K: PdeOutputSink<S> + ?Sized>(
        &self,
        prepared: &Prepared<S>,
        run: &mut Run<'_, '_, '_, S, K>,
        state: &mut PdeState<S>,
        end: S,
        requests: &mut Requests<'_, S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PdeTermination, SciOxideError> {
        let step = self.options.step();
        if let (Some(startup_end), Some(method)) = (state.startup_end, &prepared.startup)
            && state.time() < end
        {
            let termination = self.phase(
                prepared,
                method,
                startup_end.min(end),
                step.map(|h| h * S::HALF),
                run,
                state,
                requests,
                context,
            )?;
            if !termination.is_success() {
                return Ok(termination);
            }
        }
        if state.time() >= end {
            return Ok(PdeTermination::Completed);
        }
        let main = prepared.main.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("options"),
                "missing time integrator",
            )
        })?;
        self.phase(prepared, main, end, step, run, state, requests, context)
    }

    /// One ODE advance of `method` from the state to `end` with initial step `step`,
    /// within what remains of the attempted-step budget.
    #[expect(
        clippy::too_many_arguments,
        reason = "a phase is parameterized by its method, interval and step"
    )]
    fn phase<K: PdeOutputSink<S> + ?Sized>(
        &self,
        prepared: &Prepared<S>,
        method: &BuiltinOdeMethod<S>,
        end: S,
        step: Option<S>,
        run: &mut Run<'_, '_, '_, S, K>,
        state: &mut PdeState<S>,
        requests: &mut Requests<'_, S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PdeTermination, SciOxideError> {
        let budget = self.options.step_budget().get();
        let used = run.tally.steps.proposals();
        if used >= budget {
            return Ok(PdeTermination::TimeStepLimit);
        }
        if step == Some(S::ZERO) {
            return Ok(PdeTermination::TimeStepSizeLimit);
        }
        let output = self.options.output_schedule();
        let mut options = OdeOptions::new()
            .with_error_tolerance(prepared.tolerance.clone())
            .with_max_steps(Budget::try_from(budget - used)?)
            .with_output_schedule(phase_schedule(output, requests, end)?);
        if let Some(step) = step {
            options = options.with_initial_step(step);
        }
        let mut sink = PhaseSink {
            system: run.system,
            field: &mut *run.field,
            sink: &mut *run.sink,
            last_emitted: &mut run.last_emitted,
            base_index: requests.next(),
            steps: matches!(output, PdeOutputSchedule::TimeSteps),
        };
        match method.advance(
            &OdeProblem::new(run.system, end),
            &options,
            &mut state.ode,
            &mut sink,
            &mut *run.ode,
            &mut *context,
        ) {
            Ok(report) => {
                run.tally.record(
                    scioxide_core::EvaluationCounts::from_iterations(report.attempted_steps())
                        .with_proposals(
                            report.attempted_steps(),
                            report.accepted_steps(),
                            report.rejected_steps(),
                        ),
                    report.last_step_error_estimate(),
                    report.trust(),
                )?;
                let termination =
                    PdeTermination::from_ode(report.termination().ok_or_else(|| {
                        SciOxideError::invalid_input(
                            scioxide_core::argument!("report"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "ODE report has no termination",
                        )
                    })?)?;
                if termination.is_success() {
                    requests.serve_through(end);
                }
                Ok(termination)
            }
            Err(error) => {
                if let Some(progress) = run.ode.last_attempt().and_then(AttemptRecord::progress) {
                    run.tally.record(
                        scioxide_core::EvaluationCounts::from_iterations(
                            progress.attempted_steps(),
                        )
                        .with_proposals(
                            progress.attempted_steps(),
                            progress.accepted_steps(),
                            progress.rejected_steps(),
                        ),
                        progress.last_step_error_estimate(),
                        progress.trust(),
                    )?;
                }
                Err(error)
            }
        }
    }
}

/// A system without free unknowns reconstructs boundary data only: emit the
/// requested times (each one accepted as it is emitted), then reach `end`.
fn constrained_only<S, K>(
    run: &mut Run<'_, '_, '_, S, K>,
    state: &mut PdeState<S>,
    end: S,
    output: &PdeOutputSchedule<S>,
    requests: &mut Requests<'_, S>,
    execution: ExecutionContext<'_>,
) -> Result<PdeTermination, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    K: PdeOutputSink<S> + ?Sized,
{
    run.tally.recorded = true;
    run.tally.trust = PartialResultTrust::Admissible;
    let steps = matches!(output, PdeOutputSchedule::TimeSteps);
    if steps {
        emit(
            run.system,
            run.field,
            run.sink,
            state.time(),
            state.coordinates(),
            None,
        )?;
    }
    for (index, time) in requests.pending() {
        if let Some(reason) = execution.stop_reason()? {
            return Ok(PdeTermination::ExecutionStopped(reason));
        }
        emit(
            run.system,
            run.field,
            run.sink,
            time,
            state.coordinates(),
            Some(index),
        )?;
        state.ode.reset(time, &[])?;
        requests.serve(index);
    }
    if let Some(reason) = execution.stop_reason()? {
        return Ok(PdeTermination::ExecutionStopped(reason));
    }
    state.ode.reset(end, &[])?;
    if steps {
        emit(
            run.system,
            run.field,
            run.sink,
            end,
            state.coordinates(),
            None,
        )?;
    }
    Ok(PdeTermination::Completed)
}

/// Integrate with an internal workspace and trajectory under sequential native
/// execution, returning progress with any failure.
#[allow(
    clippy::result_large_err,
    reason = "owned conveniences return progress inside PartialFailure"
)]
fn solve_owned<S, P, const D: usize>(
    core: &Transient<'_, S>,
    problem: &PreparedProblem<'_, S, P, D>,
    state: Result<PdeState<S>, SciOxideError>,
    require_success: bool,
) -> Result<PdeSolution<S>, PartialFailure<PdeProgress<S>>>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: SpatialDiscretization<S, S, D>,
{
    let mut state = state.map_err(PartialFailure::new)?;
    let mut workspace = core.workspace(problem).map_err(PartialFailure::new)?;
    let mut trajectory = PdeTrajectory::new();
    match core.run(
        problem,
        None,
        &mut state,
        &mut trajectory,
        &mut workspace,
        ExecutionContext::sequential_native().into(),
    ) {
        Ok(report) => {
            if require_success && let Err(error) = report.require_success() {
                let progress = workspace.shell.into_progress();
                return Err(PartialFailure::with_progress(error, || {
                    progress
                        .ok_or_else(|| invalid(scioxide_core::argument!("state"), "no progress"))
                }));
            }
            Ok(PdeSolution {
                values: scioxide_core::backend::try_copy_slice(
                    &workspace.field,
                    scioxide_core::quantity!("pde_final_field"),
                )
                .map_err(PartialFailure::new)?,
                state,
                report,
                trajectory,
            })
        }
        Err(error) => {
            let progress = workspace.shell.into_progress();
            Err(PartialFailure::with_progress(error, || {
                progress.ok_or_else(|| invalid(scioxide_core::argument!("state"), "no progress"))
            }))
        }
    }
}

/// Prepare `M u' = b(t) − A u` and its free indices, lumped mass and Jacobian in `O(nnz)` work.
/// Every advance reuses the assembled system. Coefficients are time invariant.
#[must_use]
pub struct ParabolicProblem<'a, S: RealScalar, P, const D: usize> {
    core: PreparedProblem<'a, S, P, D>,
}
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains partial progress"
)]
impl<'a, S: RealScalar + NumericScalar<Real = S>, P: SpatialDiscretization<S, S, D>, const D: usize>
    ParabolicProblem<'a, S, P, D>
{
    /// Assemble the semidiscrete equation once for integration from zero to `end`.
    /// # Errors
    /// Rejects non-finite or non-positive end times and propagates assembly and allocation errors.
    pub fn new(
        discretization: &'a P,
        source: &'a FieldData<S, S, D>,
        end: S,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            core: PreparedProblem::new(discretization, source, end)?,
        })
    }
    /// Return the integration end time.
    pub const fn end(&self) -> S {
        self.core.end()
    }
    /// Return the prepared spatial discretization.
    pub const fn discretization(&self) -> &'a P {
        self.core.discretization()
    }
    /// Return the free-unknown semidiscrete system.
    pub const fn system(&self) -> &SemidiscreteSystem<'a, S, P, D> {
        self.core.system()
    }
    /// Sample the initial field at free unknowns at time zero.
    /// # Errors
    /// Propagates invalid options, callback, non-finite and allocation errors.
    pub fn initial_state(
        &self,
        initial: &FieldData<S, S, D>,
        options: &ParabolicOptions<S>,
    ) -> Result<PdeState<S>, SciOxideError> {
        let core = Transient {
            options,
            order: MethodOfLinesOrder::First,
        };
        options.validate()?;
        core.initial_state(&self.core, initial, None)
    }
    /// Build the initial state from full fields, ignoring constrained entries.
    /// # Errors
    /// Rejects incorrect lengths, non-finite free entries, invalid options and allocation failures.
    pub fn initial_state_from_values(
        &self,
        values: &[S],
        options: &ParabolicOptions<S>,
    ) -> Result<PdeState<S>, SciOxideError> {
        let core = Transient {
            options,
            order: MethodOfLinesOrder::First,
        };
        options.validate()?;
        let mut coordinates =
            crate::discretization::reserve(self.system().free_indices().len() * core.blocks())?;
        for field in [values] {
            crate::validation::shape(field.len(), self.discretization().unknowns())?;
            for &i in self.system().free_indices() {
                crate::validation::finite(field[i], scioxide_core::NonFiniteOrigin::Input)?;
                coordinates.push(field[i]);
            }
        }
        Ok(PdeState {
            ode: OdeState::new(S::ZERO, Vector::from_vec(coordinates)),
            free: scioxide_core::backend::try_copy_slice(
                self.system().free_indices(),
                scioxide_core::quantity!("state_layout"),
            )?,
            full: self.discretization().unknowns(),
            startup_end: core.startup_end()?,
            mesh_token: self.discretization().mesh_token().clone(),
            order: MethodOfLinesOrder::First,
        })
    }
    /// Prepare reusable storage for progressive integration.
    /// # Errors
    /// Propagates size and allocation failures.
    pub fn workspace(&self) -> Result<PdeWorkspace<S>, SciOxideError> {
        Transient {
            options: &ParabolicOptions::default(),
            order: MethodOfLinesOrder::First,
        }
        .workspace(&self.core)
    }
    /// Advance to the end time, retaining the last accepted state on failure.
    /// Preflight rejection leaves the state unchanged; emitted samples are irrevocable.
    /// # Errors
    /// Records invalid options, state, callback, sink and numerical failures in the workspace.
    pub fn advance<'scope, 'capability, K: PdeOutputSink<S> + ?Sized>(
        &self,
        options: &ParabolicOptions<S>,
        state: &mut PdeState<S>,
        sink: &mut K,
        workspace: &mut PdeWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        if let Err(error) = options.validate() {
            return workspace.shell.reject(AttemptKind::Call, error);
        }
        let core = Transient {
            options,
            order: MethodOfLinesOrder::First,
        };
        core.run(&self.core, None, state, sink, workspace, scope.into())
    }
    /// Restore a compatible checkpoint and advance with fresh local budgets.
    /// # Errors
    /// Records validation failures without changing the state or checkpoint; otherwise follows [`Self::advance`].
    pub fn resume<'scope, 'capability, K: PdeOutputSink<S> + ?Sized>(
        &self,
        options: &ParabolicOptions<S>,
        checkpoint: &PdeCheckpoint<S>,
        state: &mut PdeState<S>,
        sink: &mut K,
        workspace: &mut PdeWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        if let Err(error) = options.validate() {
            return workspace.shell.reject(AttemptKind::Resume, error);
        }
        let core = Transient {
            options,
            order: MethodOfLinesOrder::First,
        };
        core.run(
            &self.core,
            Some(checkpoint),
            state,
            sink,
            workspace,
            scope.into(),
        )
    }
    /// Reconstruct the full physical field without allocation, leaving output unchanged on error.
    /// # Errors
    /// Rejects foreign mesh revisions, equation order, dimensions and non-finite data; propagates boundary callback errors.
    pub fn values_into(&self, state: &PdeState<S>, out: &mut [S]) -> Result<(), SciOxideError> {
        if !state
            .mesh_token
            .is_compatible_with(self.discretization().mesh_token())
            || state.order != MethodOfLinesOrder::First
            || state.free != self.system().free_indices()
            || state.full != self.discretization().unknowns()
        {
            return Err(invalid(
                scioxide_core::argument!("state"),
                "state belongs to another mesh, free-node layout or equation order",
            ));
        }
        let n = self.system().free_indices().len();
        crate::validation::shape(state.coordinates().len(), n)?;
        crate::mol::MethodOfLinesFields::reconstruct_into(
            self.system(),
            state.time(),
            &state.coordinates()[..n],
            out,
        )
    }
    /// Reconstruct the full physical field in discretization unknown order.
    /// # Errors
    /// Returns [`Self::values_into`] errors and allocation failures.
    pub fn values(&self, state: &PdeState<S>) -> Result<Vec<S>, SciOxideError> {
        let mut values = scioxide_core::backend::try_filled(
            self.discretization().unknowns(),
            S::ZERO,
            scioxide_core::quantity!("pde_values"),
        )?;
        self.values_into(state, &mut values)?;
        Ok(values)
    }
    /// Integrate with configured options and return the field, report and trajectory.
    /// # Errors
    /// Returns integration errors with the last accepted progress when available.
    pub fn solve_with_options(
        &self,
        initial: &FieldData<S, S, D>,
        options: &ParabolicOptions<S>,
    ) -> Result<PdeSolution<S>, PartialFailure<PdeProgress<S>>> {
        let core = Transient {
            options,
            order: MethodOfLinesOrder::First,
        };
        solve_owned(
            &core,
            &self.core,
            options
                .validate()
                .and_then(|()| core.initial_state(&self.core, initial, None)),
            false,
        )
    }
    /// Integrate with the default method, 8-decimal RMS tolerance and 100,000 attempted steps.
    /// Returns the full final field in discretization unknown order.
    /// # Errors
    /// Returns every unsuccessful termination as an error with accepted progress, along with callback and numerical failures.
    /// Uses fifth-order L-stable Radau IIA. For unit diffusion to `t = 0.1`,
    /// a sine mode takes about 25 steps on 81–1281 nodes and 42 on 101 × 101 nodes;
    /// the maximum field error is below `1e-4`, dominated by spatial discretization.
    /// # References
    /// E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 1996, §IV.5 and §IV.8.
    /// ```
    /// use scioxide_pde::{FieldData, FieldPoint, ParabolicProblem, StructuredFdmPlan};
    /// let fdm = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [81])?;
    /// let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((core::f64::consts::PI * p.point[0]).sin()));
    /// let u = ParabolicProblem::new(&fdm, &FieldData::zero(), 0.1)?.solve(&initial)?;
    /// assert!((u[40] - (-core::f64::consts::PI.powi(2) * 0.1).exp()).abs() < 1e-4);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve(
        &self,
        initial: &FieldData<S, S, D>,
    ) -> Result<Vec<S>, PartialFailure<PdeProgress<S>>> {
        let options = ParabolicOptions::default();
        let core = Transient {
            options: &options,
            order: MethodOfLinesOrder::First,
        };
        let solution = solve_owned(
            &core,
            &self.core,
            core.initial_state(&self.core, initial, None),
            true,
        )?;
        Ok(solution.into_parts().1)
    }
}

/// Prepare `M u'' = b(t) − A u` and its free indices, lumped mass and Jacobian in `O(nnz)` work.
/// Every advance reuses the assembled system. Coefficients are time invariant.
#[must_use]
pub struct HyperbolicProblem<'a, S: RealScalar, P, const D: usize> {
    core: PreparedProblem<'a, S, P, D>,
}
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains partial progress"
)]
impl<'a, S: RealScalar + NumericScalar<Real = S>, P: SpatialDiscretization<S, S, D>, const D: usize>
    HyperbolicProblem<'a, S, P, D>
{
    /// Assemble the semidiscrete equation once for integration from zero to `end`.
    /// # Errors
    /// Rejects non-finite or non-positive end times and propagates assembly and allocation errors.
    pub fn new(
        discretization: &'a P,
        source: &'a FieldData<S, S, D>,
        end: S,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            core: PreparedProblem::new(discretization, source, end)?,
        })
    }
    /// Return the integration end time.
    pub const fn end(&self) -> S {
        self.core.end()
    }
    /// Return the prepared spatial discretization.
    pub const fn discretization(&self) -> &'a P {
        self.core.discretization()
    }
    /// Return the free-unknown semidiscrete system.
    pub const fn system(&self) -> &SemidiscreteSystem<'a, S, P, D> {
        self.core.system()
    }
    /// Sample the initial field at free unknowns at time zero.
    /// # Errors
    /// Propagates invalid options, callback, non-finite and allocation errors.
    pub fn initial_state(
        &self,
        displacement: &FieldData<S, S, D>,
        velocity: &FieldData<S, S, D>,
    ) -> Result<PdeState<S>, SciOxideError> {
        let options = HyperbolicOptions::default();
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        options.validate()?;
        core.initial_state(&self.core, displacement, Some(velocity))
    }
    /// Build the initial state from full fields, ignoring constrained entries.
    /// # Errors
    /// Rejects incorrect lengths, non-finite free entries, invalid options and allocation failures.
    pub fn initial_state_from_values(
        &self,
        displacement: &[S],
        velocity: &[S],
    ) -> Result<PdeState<S>, SciOxideError> {
        let options = HyperbolicOptions::default();
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        options.validate()?;
        let mut coordinates =
            crate::discretization::reserve(self.system().free_indices().len() * core.blocks())?;
        for field in [displacement, velocity] {
            crate::validation::shape(field.len(), self.discretization().unknowns())?;
            for &i in self.system().free_indices() {
                crate::validation::finite(field[i], scioxide_core::NonFiniteOrigin::Input)?;
                coordinates.push(field[i]);
            }
        }
        Ok(PdeState {
            ode: OdeState::new(S::ZERO, Vector::from_vec(coordinates)),
            free: scioxide_core::backend::try_copy_slice(
                self.system().free_indices(),
                scioxide_core::quantity!("state_layout"),
            )?,
            full: self.discretization().unknowns(),
            startup_end: core.startup_end()?,
            mesh_token: self.discretization().mesh_token().clone(),
            order: MethodOfLinesOrder::Second,
        })
    }
    /// Prepare reusable storage for progressive integration.
    /// # Errors
    /// Propagates size and allocation failures.
    pub fn workspace(&self) -> Result<PdeWorkspace<S>, SciOxideError> {
        Transient {
            options: &ParabolicOptions::default(),
            order: MethodOfLinesOrder::Second,
        }
        .workspace(&self.core)
    }
    /// Advance to the end time, retaining the last accepted state on failure.
    /// Preflight rejection leaves the state unchanged; emitted samples are irrevocable.
    /// # Errors
    /// Records invalid options, state, callback, sink and numerical failures in the workspace.
    pub fn advance<'scope, 'capability, K: PdeOutputSink<S> + ?Sized>(
        &self,
        options: &HyperbolicOptions<S>,
        state: &mut PdeState<S>,
        sink: &mut K,
        workspace: &mut PdeWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        if let Err(error) = options.validate() {
            return workspace.shell.reject(AttemptKind::Call, error);
        }
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        core.run(&self.core, None, state, sink, workspace, scope.into())
    }
    /// Restore a compatible checkpoint and advance with fresh local budgets.
    /// # Errors
    /// Records validation failures without changing the state or checkpoint; otherwise follows [`Self::advance`].
    pub fn resume<'scope, 'capability, K: PdeOutputSink<S> + ?Sized>(
        &self,
        options: &HyperbolicOptions<S>,
        checkpoint: &PdeCheckpoint<S>,
        state: &mut PdeState<S>,
        sink: &mut K,
        workspace: &mut PdeWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PdeSolveReport<S>, SciOxideError> {
        if let Err(error) = options.validate() {
            return workspace.shell.reject(AttemptKind::Resume, error);
        }
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        core.run(
            &self.core,
            Some(checkpoint),
            state,
            sink,
            workspace,
            scope.into(),
        )
    }
    /// Reconstruct the full physical field without allocation, leaving output unchanged on error.
    /// # Errors
    /// Rejects foreign mesh revisions, equation order, dimensions and non-finite data; propagates boundary callback errors.
    pub fn values_into(&self, state: &PdeState<S>, out: &mut [S]) -> Result<(), SciOxideError> {
        if !state
            .mesh_token
            .is_compatible_with(self.discretization().mesh_token())
            || state.order != MethodOfLinesOrder::Second
            || state.free != self.system().free_indices()
            || state.full != self.discretization().unknowns()
        {
            return Err(invalid(
                scioxide_core::argument!("state"),
                "state belongs to another mesh or equation order",
            ));
        }
        let n = self.system().free_indices().len();
        crate::validation::shape(state.coordinates().len(), n * 2)?;
        crate::mol::MethodOfLinesFields::reconstruct_into(
            self.system(),
            state.time(),
            &state.coordinates()[..n],
            out,
        )
    }
    /// Reconstruct the full physical field in discretization unknown order.
    /// # Errors
    /// Returns [`Self::values_into`] errors and allocation failures.
    pub fn values(&self, state: &PdeState<S>) -> Result<Vec<S>, SciOxideError> {
        let mut values = scioxide_core::backend::try_filled(
            self.discretization().unknowns(),
            S::ZERO,
            scioxide_core::quantity!("pde_values"),
        )?;
        self.values_into(state, &mut values)?;
        Ok(values)
    }
    /// Integrate with configured options and return the field, report and trajectory.
    /// # Errors
    /// Returns integration errors with the last accepted progress when available.
    pub fn solve_with_options(
        &self,
        displacement: &FieldData<S, S, D>,
        velocity: &FieldData<S, S, D>,
        options: &HyperbolicOptions<S>,
    ) -> Result<PdeSolution<S>, PartialFailure<PdeProgress<S>>> {
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        solve_owned(
            &core,
            &self.core,
            options
                .validate()
                .and_then(|()| core.initial_state(&self.core, displacement, Some(velocity))),
            false,
        )
    }
    /// Integrate with the default method, 8-decimal RMS tolerance and 100,000 attempted steps.
    /// Returns the full final field in discretization unknown order.
    /// # Errors
    /// Returns every unsuccessful termination as an error with accepted progress, along with callback and numerical failures.
    /// Uses Dormand–Prince 5(4). Automatic stability caps steps at
    /// `0.9 · 0.9971 / √ρ`, where `ρ` bounds the mass-normalized stiffness spectrum;
    /// unit-speed waves need approximately `2.23 √D/h` steps per unit time.
    /// # References
    /// E. Hairer, S. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I*, 1993, §II.5.
    /// ```
    /// use scioxide_pde::{FieldData, FieldPoint, HyperbolicProblem, StructuredFdmPlan};
    /// let fdm = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [81])?;
    /// let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok((core::f64::consts::PI * p.point[0]).sin()));
    /// let u = HyperbolicProblem::new(&fdm, &FieldData::zero(), 1.0)?.solve(&initial, &FieldData::zero())?;
    /// assert!((u[40] + 1.0).abs() < 1e-3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve(
        &self,
        displacement: &FieldData<S, S, D>,
        velocity: &FieldData<S, S, D>,
    ) -> Result<Vec<S>, PartialFailure<PdeProgress<S>>> {
        let options = HyperbolicOptions::default();
        let converted = options.parabolic();
        let core = Transient {
            options: &converted,
            order: MethodOfLinesOrder::Second,
        };
        let solution = solve_owned(
            &core,
            &self.core,
            core.initial_state(&self.core, displacement, Some(velocity)),
            true,
        )?;
        Ok(solution.into_parts().1)
    }
}
