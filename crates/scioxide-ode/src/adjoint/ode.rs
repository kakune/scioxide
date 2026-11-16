use super::checkpoint::{Kernel, Step, Tape, TapePlan, TapeShape, Trial};
use super::*;
use crate::OdeSystem;
use crate::dense_output::DenseExtension;
use crate::explicit_rk::dormand_prince5::DENSE_WEIGHTS;
use crate::explicit_rk::{ButcherTableau, perform_explicit_step};
use crate::shell::Shell;
use crate::stepper::{StepRequest, StepTrace};
use crate::termination::{Fault, Work};
use crate::workspace::ExplicitOdeWorkspace;
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{AttemptRecord, Capacity, EvaluationRole, ProblemToken};
#[cfg(test)]
use scioxide_core::{ExecutionContext, PartialFailure};
use scioxide_linalg::Vector;

// Prepared DP5 discrete/continuous adjoint of a scalar objective over real time.
//
// The plan owns immutable method coefficients and bounded checkpoint metadata; it takes
// `&self`, is `Clone` and holds no caller identity. Parameters are real; states and
// cotangents support real and complex widths. Discrete frames retain running-cost stage
// gradients; checkpoint memory admission includes these values and replay recomputes
// them. Use [`DaeAdjointSolver`] for residual-form stiff dynamics.
//
// [`Self::solve_into`] is the atomic execution form; [`Self::solve`] is the owned
// convenience.
//

#[derive(Debug, Clone)]
#[must_use]
pub(super) struct PreparedOdeAdjoint<T: NumericScalar> {
    pub(super) tape: TapePlan,
    observations: Vec<T::Real>,
    options: Policies<T::Real>,
    tableau: ButcherTableau<T::Real, 7, 7, true>,
    provider_scratch: usize,
    objective_scratch: usize,
    event_records: usize,
}

// Storage one [`PreparedOdeAdjoint`] needs, from
// [`PreparedOdeAdjoint::workspace_requirements`].
//
// A workspace serves exactly one requirement shape at a time; reserving another shape
// replaces its storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OdeStorageRequirements {
    tape: TapeShape,
    provider: usize,
    objective: usize,
    discrete: bool,
    event_records: usize,
    bytes: usize,
}

impl OdeStorageRequirements {
    // Length of the initial-gradient output: the state dimension.
    pub(super) const fn initial_gradient_len(&self) -> usize {
        self.tape.dimension()
    }
    // Length of the parameter-gradient output.
    pub(super) const fn parameter_gradient_len(&self) -> usize {
        self.tape.parameters()
    }
    // Admitted conservative upper bound on the numerical heap payload.
    pub(super) const fn workspace_bytes(&self) -> usize {
        self.bytes
    }
}

// Reusable DP5 adjoint trajectory, checkpoint and reverse storage, solve state and last
// attempt. Each solve starts a fresh tape.
//
// [`OdeAdjointBuffers::new`] allocates nothing; [`PreparedOdeAdjoint::workspace`] or
// [`Self::try_reserve`] prepare it.
pub(super) struct OdeAdjointBuffers<T: NumericScalar> {
    pub(super) storage: Option<OdeStorage<T>>,
    pub(super) shell: Shell<AdjointReport<T::Real>>,
}

pub(super) struct OdeStorage<T: NumericScalar> {
    requirements: OdeStorageRequirements,
    pub(super) tape: Tape<T>,
    work: OdeWork<T>,
}

impl<T: NumericScalar> Default for OdeAdjointBuffers<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NumericScalar> core::fmt::Debug for OdeAdjointBuffers<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OdeAdjointBuffers")
            .field(
                "requirements",
                &self.storage.as_ref().map(|storage| storage.requirements),
            )
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<T: NumericScalar> OdeAdjointBuffers<T> {
    // An empty workspace; allocates nothing.
    #[must_use]
    pub(super) const fn new() -> Self {
        Self {
            storage: None,
            shell: Shell::new(),
        }
    }

    fn fits(&self, requirements: &OdeStorageRequirements) -> bool {
        self.storage
            .as_ref()
            .is_some_and(|storage| storage.requirements == *requirements)
    }

    // Whether this workspace serves `requirements` without allocating.
    //
    // # Errors
    // This check is infallible today; the `Result` keeps the shared protocol.
    pub(super) fn check(
        &self,
        requirements: &OdeStorageRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.fits(requirements) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    // Prepare this workspace for `requirements`, replacing storage of another shape.
    //
    // # Errors
    // Returns allocation and size-overflow errors.
    pub(super) fn try_reserve(
        &mut self,
        requirements: &OdeStorageRequirements,
    ) -> Result<(), SciOxideError> {
        if !self.fits(requirements) {
            self.storage = None;
            self.storage = Some(OdeStorage::new(*requirements)?);
        }
        Ok(())
    }

    // The last attempt: its epoch, outcome, counts and [`AdjointReport`] progress.
    pub(super) const fn last_attempt(&self) -> Option<&AttemptRecord<AdjointReport<T::Real>>> {
        self.shell.last_attempt()
    }
}

struct OdeWork<T: NumericScalar> {
    ivp: crate::ivp::IvpStorage<T::Real, ExplicitOdeWorkspace<T::Real, T>, T>,
    primal_state: Vector<T>,
    forward: ExplicitOdeWorkspace<T::Real, T>,
    backward: crate::ivp::IvpStorage<T::Real, ExplicitOdeWorkspace<T::Real, T>, T>,
    reverse_time: Option<T::Real>,
    value: Vector<T>,
    reverse: Vector<T>,
    stage_bars: Vec<T>,
    stage_state: Vec<T>,
    stage_bar: Vec<T>,
    objective_gradients: Vec<T>,
    current_objective_gradient: Vec<T>,
    /// Terminal-event state and parameter gradients of total-derivative queries.
    event_state: Vec<T>,
    event_parameters: Vec<T::Real>,
    products: Products<T>,
    reverse_attempts: usize,
}

impl<T: NumericScalar> OdeStorage<T> {
    fn new(requirements: OdeStorageRequirements) -> Result<Self, SciOxideError> {
        let n = requirements.tape.dimension();
        let p = requirements.tape.parameters();
        let filled = |len, quantity| scioxide_core::backend::try_filled(len, T::ZERO, quantity);
        let gradients = |len: usize| if requirements.discrete { len } else { 0 };
        let mut forward = ExplicitOdeWorkspace::new();
        forward.try_resize(n + 1, 7, 4)?;
        let mut backward = crate::ivp::IvpStorage::default();
        backward.reserve(
            &crate::DormandPrince5::<T::Real, T>::new(),
            n + p,
            0,
            0,
            false,
        )?;
        let mut ivp = crate::ivp::IvpStorage::default();
        ivp.reserve(
            &crate::DormandPrince5::<T::Real, T>::new(),
            n,
            1,
            requirements.event_records,
            false,
        )?;
        Ok(Self {
            requirements,
            tape: Tape::new(requirements.tape)?,
            work: OdeWork {
                ivp,
                primal_state: Vector::zeros(n)?,
                forward,
                backward,
                reverse_time: None,
                value: Vector::zeros(n + 1)?,
                reverse: Vector::zeros(n + p)?,
                stage_bars: filled(7 * (n + 1), scioxide_core::quantity!("stage_bars"))?,
                stage_state: filled(n + 1, scioxide_core::quantity!("stage_state"))?,
                stage_bar: filled(n, scioxide_core::quantity!("stage_bar"))?,
                objective_gradients: filled(
                    gradients(7 * (n + p)),
                    scioxide_core::quantity!("objective_gradients"),
                )?,
                current_objective_gradient: filled(
                    gradients(n + p),
                    scioxide_core::quantity!("current_objective_gradient"),
                )?,
                event_state: filled(n, scioxide_core::quantity!("event_gradient"))?,
                event_parameters: scioxide_core::backend::try_filled(
                    p,
                    T::Real::ZERO,
                    scioxide_core::quantity!("event_gradient"),
                )?,
                products: Products::new(n, p, requirements.provider, requirements.objective)?,
                reverse_attempts: 0,
            },
        })
    }
}

impl<T: NumericScalar> PreparedOdeAdjoint<T> {
    pub(super) fn matches(&self, shape: Shape<'_, T::Real>) -> bool {
        self.tape.dimension == shape.dimension
            && self.tape.parameters == shape.parameters
            && self.provider_scratch == shape.provider_scratch
            && self.objective_scratch == shape.objective_scratch
            && check_observations(&self.observations, shape.observations).is_ok()
    }

    // Validate `options` for `problem` and `objective` and admit the complete reservation.
    //
    // [`CheckpointPolicy::MemoryBudget`] selects checkpoints so that the full workspace,
    // not just checkpoint vectors, fits [`Policies::max_workspace_bytes`].
    //
    // # Errors
    // Rejects an empty state, mass matrices, nonreplayable callbacks, nonfinite
    // parameters, invalid tolerances, event options or initial step, a zero step
    // capacity, observation schedules outside discrete mode, and limits that cannot
    // admit the checkpoint policy.
    pub(super) fn new(
        shape: Shape<'_, T::Real>,
        options: Policies<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let observations = prepare_observations(shape.observations, options.mode)?;
        let n = shape.dimension;
        let p = shape.parameters;
        options.validate(n, p, n, true)?;
        options.event_options.validate()?;
        // Forward driver, DP5 recording/replay and reverse workspaces, plus
        // products, event gradients, transactions and stage pullbacks.
        // A conservative byte bound includes Vec descriptors and small capacities.
        let width = checked_add(n, 1, scioxide_core::quantity!("forward_state"))?;
        let reverse_width = checked_add(n, p, scioxide_core::quantity!("reverse_state"))?;
        let base = checked_add(
            checked_mul(
                checked_add(
                    checked_mul(
                        checked_add(
                            width,
                            reverse_width,
                            scioxide_core::quantity!("integrator_values"),
                        )?,
                        256,
                        scioxide_core::quantity!("integrator_values"),
                    )?,
                    checked_add(
                        shape.provider_scratch,
                        shape.objective_scratch,
                        scioxide_core::quantity!("callback_scratch"),
                    )?,
                    scioxide_core::quantity!("workspace_values"),
                )?,
                core::mem::size_of::<T>(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            16384,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let base = checked_add(
            base,
            if options.mode == AdjointMode::Discrete {
                checked_mul(
                    checked_mul(
                        8,
                        reverse_width,
                        scioxide_core::quantity!("objective_gradients"),
                    )?,
                    core::mem::size_of::<T>(),
                    scioxide_core::quantity!("workspace_bytes"),
                )?
            } else {
                0
            },
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let tape = TapePlan::new::<T, ()>(
            n,
            p,
            checked_add(
                checked_mul(8, width, scioxide_core::quantity!("checkpoint_frame"))?,
                if options.mode == AdjointMode::Discrete {
                    checked_mul(
                        7,
                        reverse_width,
                        scioxide_core::quantity!("objective_gradients"),
                    )?
                } else {
                    0
                },
                scioxide_core::quantity!("checkpoint_frame"),
            )?,
            n,
            base,
            options.limits(),
            &options.checkpoint_policy,
            !observations.is_empty(),
        )?;
        let event_records = crate::OdeOptions::new()
            .with_event_options(options.event_options)
            .event_records(1)?;
        Ok(Self {
            tape,
            observations,
            options,
            tableau: crate::explicit_rk::dormand_prince5::tableau(),
            provider_scratch: shape.provider_scratch,
            objective_scratch: shape.objective_scratch,
            event_records,
        })
    }

    // The validated options.

    fn requirements(&self) -> OdeStorageRequirements {
        OdeStorageRequirements {
            tape: self.tape.shape(),
            provider: self.provider_scratch,
            objective: self.objective_scratch,
            discrete: self.options.mode == AdjointMode::Discrete,
            event_records: self.event_records,
            bytes: self.tape.bytes,
        }
    }

    // Storage needed by every solve with this plan.
    //
    // # Errors
    // Infallible today; the `Result` keeps the shared protocol.
    pub(super) fn workspace_requirements(&self) -> Result<OdeStorageRequirements, SciOxideError> {
        Ok(self.requirements())
    }

    // A workspace with every numerical buffer prepared; solves never grow it.
    //
    // # Errors
    // Returns allocation and size-overflow errors.
    #[cfg(test)]
    pub(super) fn workspace(&self) -> Result<OdeAdjointBuffers<T>, SciOxideError> {
        let mut workspace = OdeAdjointBuffers::new();
        workspace.try_reserve(&self.requirements())?;
        Ok(workspace)
    }

    // Shape, capability, interval and finiteness checks; grows an undersized workspace.
    #[allow(clippy::too_many_arguments)]
    fn preflight(
        &self,
        problem: &impl OdeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        interval: (T::Real, T::Real),
        event: Option<&dyn AdjointTerminalEvent<T>>,
        initial_gradient: &[T],
        parameter_gradient: &[T::Real],
        workspace: &mut OdeAdjointBuffers<T>,
    ) -> Result<(), SciOxideError> {
        check_observations(&self.observations, objective.observation_times())?;
        let n = self.tape.dimension;
        if problem.dimension() != n
            || initial.len() != n
            || initial_gradient.len() != n
            || parameter_gradient.len() != self.tape.parameters
            || problem.parameters().len() != self.tape.parameters
            || problem.pullback_workspace_len() > self.provider_scratch
            || objective.workspace_len() > self.objective_scratch
            || !problem.is_replayable()
            || problem.mass_matrix_revision().is_some()
        {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "adjoint plan, providers and gradient outputs must have matching capabilities and shapes",
            ));
        }
        if !interval.0.is_finite() || !interval.1.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("interval"),
                "finite integration endpoints required",
            ));
        }
        if event.is_some() && !self.observations.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("event"),
                "sampled objectives with terminal events are unsupported",
            ));
        }
        finite(
            problem.parameters(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        finite(initial, scioxide_core::NonFiniteOrigin::Input)?;
        let requirements = self.requirements();
        if !workspace.fits(&requirements) {
            workspace.try_reserve(&requirements)?;
        }
        Ok(())
    }

    // Compute the gradients of the objective with respect to the initial state and the
    // real parameters.
    //
    // Each call is one epoch of `scope` (a root epoch for an [`ExecutionContext`], a child
    // epoch for a `&mut SolveContext`) and records a fresh tape. Both gradient outputs
    // are written only on success and remain unchanged on every error; diagnostics of
    // a failed attempt, classified as in [`AdjointReport::termination`], are recorded in
    // [`OdeAdjointBuffers::last_attempt`]. A preflight rejection runs no callback; an
    // undersized workspace is grown first. An event is optional and must identify one
    // smooth transverse terminal branch. With a prepared workspace and nonallocating
    // callbacks, the numerical processing allocates nothing, with or without an event.
    //
    // # Errors
    // Returns shape, capability, interval and nonfinite-input errors at preflight, and
    // provider, event, replay, numerical, budget and cancellation errors during the solve.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve_into<'scope, 'capability>(
        &self,
        problem: &impl OdeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        interval: (T::Real, T::Real),
        event: Option<&dyn AdjointTerminalEvent<T>>,
        initial_gradient: &mut [T],
        parameter_gradient: &mut [T::Real],
        workspace: &mut OdeAdjointBuffers<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<AdjointReport<T::Real>, SciOxideError> {
        let scope = scope.into();
        if let Err(error) = self.preflight(
            problem,
            objective,
            initial,
            interval,
            event,
            initial_gradient,
            parameter_gradient,
            workspace,
        ) {
            return workspace.shell.reject(error);
        }
        let OdeAdjointBuffers { storage, shell } = workspace;
        let Some(storage) = storage.as_mut() else {
            return shell.reject(invalid(
                scioxide_core::argument!("workspace"),
                "adjoint workspace storage is not prepared",
            ));
        };
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "dp5-adjoint", 1),
            self.tape.dimension as u128,
            self.options.mode as u128,
        );
        let report = execute(
            shell,
            scope,
            identity,
            self.options.budgets(),
            AdjointReport::initial(interval.0, self.tape.bytes),
            |report, cx| {
                self.run(
                    problem, objective, initial, interval, event, storage, report, cx,
                )
            },
        )?;
        initial_gradient.copy_from_slice(&storage.tape.lambda);
        parameter_gradient.copy_from_slice(&storage.tape.gradient);
        Ok(report)
    }

    // The forward recording and reverse sweep of one epoch.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        problem: &impl OdeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        interval: (T::Real, T::Real),
        event: Option<&dyn AdjointTerminalEvent<T>>,
        storage: &mut OdeStorage<T>,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        storage.tape.state.copy_from_slice(initial);
        storage.tape.bind_parameters(problem.parameters())?;
        if let Some(event) = event {
            if !event.unique_branch()
                || !event.transversality_threshold().is_finite()
                || event.transversality_threshold() <= T::Real::ZERO
            {
                report.termination = Some(AdjointTermination::EventDerivativeUnavailable);
                return Err(invalid(
                    scioxide_core::argument!("event"),
                    "a unique transverse event branch must be declared",
                ));
            }
            admit(context, EvaluationRole::Event)?;
            let value = event.value(interval.0, &storage.tape.state)?;
            if !value.is_finite() || value == T::Real::ZERO {
                report.termination = Some(AdjointTermination::EventDerivativeUnavailable);
                return Err(invalid(
                    scioxide_core::argument!("event"),
                    "initial or nonfinite event roots have no selected smooth terminal derivative",
                ));
            }
        }
        storage.work.reverse_attempts = 0;
        storage.work.reverse_time = None;
        storage.work.forward.reset_epoch();
        let mut kernel = OdeKernel {
            plan: self,
            problem,
            objective,
            event,
            work: &mut storage.work,
        };
        let terminal_event = storage.tape.run(
            &self.tape,
            &self.options,
            &mut kernel,
            interval.0,
            interval.1,
            report,
            context,
        )?;
        check_observations(&self.observations, objective.observation_times())?;
        context.execution().check()?;
        Ok(terminal_event)
    }

    // Compute both gradients with an internal workspace under sequential native
    // execution (portable scalar dispatch without `std`).
    //
    // # Errors
    // Returns the [`Self::solve_into`] error together with the last attempt's progress
    // when one was recorded.
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    #[cfg(test)]
    pub(super) fn solve(
        &self,
        problem: &impl OdeAdjointSystem<T>,
        objective: &impl AdjointObjective<T>,
        initial: &[T],
        interval: (T::Real, T::Real),
        event: Option<&dyn AdjointTerminalEvent<T>>,
    ) -> Result<AdjointResult<T>, PartialFailure<AdjointReport<T::Real>>> {
        let mut workspace = self.workspace().map_err(PartialFailure::new)?;
        let (mut initial_gradient, mut parameter_gradient) =
            gradients::<T>(self.tape.dimension, self.tape.parameters)
                .map_err(PartialFailure::new)?;
        match self.solve_into(
            problem,
            objective,
            initial,
            interval,
            event,
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
            Err(error) => Err(partial(error, workspace.shell)),
        }
    }
}

struct OdeKernel<'a, T: NumericScalar, P, O> {
    plan: &'a PreparedOdeAdjoint<T>,
    problem: &'a P,
    objective: &'a O,
    event: Option<&'a dyn AdjointTerminalEvent<T>>,
    work: &'a mut OdeWork<T>,
}

// Self-metered replay system: admits its RHS, then its objective work, in `cx`.
struct Primal<'a, T: NumericScalar, P, O> {
    gradients: RefCell<&'a mut [T]>,
    stage: core::cell::Cell<usize>,
    problem: &'a P,
    objective: &'a O,
    products: RefCell<&'a mut Products<T>>,
}
impl<T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>> OdeSystem<T::Real, T>
    for Primal<'_, T, P, O>
{
    fn dimension(&self) -> usize {
        self.problem.dimension() + 1
    }
    fn rhs(
        &self,
        time: T::Real,
        state: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.problem.dimension();
        admit(cx, EvaluationRole::RightHandSide)?;
        output.fill(T::from_real(T::Real::NAN));
        self.problem.rhs(time, &state[..n], &mut output[..n], cx)?;
        let mut products = self.products.borrow_mut();
        let objective = products.objective(
            self.objective,
            AdjointObjectivePart::Running,
            time,
            &state[..n],
            self.problem.parameters(),
            cx,
        )?;
        let mut gradients = self.gradients.borrow_mut();
        if !gradients.is_empty() {
            let width = n + products.parameters.len();
            let stage = self.stage.get();
            gradients[stage * width..stage * width + n].copy_from_slice(&products.state);
            for (out, &value) in gradients[stage * width + n..(stage + 1) * width]
                .iter_mut()
                .zip(&products.parameters)
            {
                *out = T::from_real(value);
            }
            self.stage.set(stage + 1);
        }
        output[n] = T::from_real(objective.value);
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}

impl<T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>> Kernel<T>
    for OdeKernel<'_, T, P, O>
{
    type Metadata = ();
    fn accept(&mut self, _: T::Real) -> Result<(), SciOxideError> {
        self.work.forward.accept_step();
        let width = self.work.current_objective_gradient.len();
        if width != 0 {
            self.work
                .current_objective_gradient
                .copy_from_slice(&self.work.objective_gradients[6 * width..7 * width]);
        }
        Ok(())
    }
    fn restart(&mut self) {
        self.work.forward.reset_epoch();
    }
    fn observations(&self) -> &[T::Real] {
        &self.plan.observations
    }
    fn sample(
        &mut self,
        time: T::Real,
        state: &[T],
        packed: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.work.products.observations(
            self.objective,
            &self.plan.observations,
            time,
            state,
            self.problem.parameters(),
            packed,
            context,
        )
    }
    fn parameters(&self) -> &[T::Real] {
        self.problem.parameters()
    }
    fn record(
        &mut self,
        tape: &mut Tape<T>,
        start: T::Real,
        end: T::Real,
        cost: &mut T::Real,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        if start == end {
            return Ok(false);
        }
        self.work
            .primal_state
            .as_mut_slice()
            .copy_from_slice(&tape.state);
        let options = crate::OdeOptions::new()
            .with_initial_step(self.plan.options.initial_step)
            .with_event_options(self.plan.options.event_options);
        let event = TerminalEvent {
            event: self.event,
            state: RefCell::new(&mut self.work.event_state),
            parameters: RefCell::new(&mut self.work.event_parameters),
        };
        let mut recorder = Forward {
            plan: self.plan,
            problem: self.problem,
            objective: self.objective,
            tape,
            products: &mut self.work.products,
            stage_state: &mut self.work.stage_state,
            gradients: &mut self.work.objective_gradients,
            current_gradient: &mut self.work.current_objective_gradient,
            cached: None,
            cached_cost: T::Real::ZERO,
            error: T::Real::ZERO,
            quadrature: T::Real::ZERO,
            end,
            cost,
            report,
        };
        let method = crate::DormandPrince5::new();
        let event_function: &dyn crate::EventFunction<T::Real, T> = &event;
        let events = if self.event.is_some() {
            core::slice::from_ref(&event_function)
        } else {
            &[]
        };
        let run = crate::ivp::Drive {
            method: &method,
            info: crate::stepper::Stepper::<T::Real, T>::info(&method),
            options: &options,
            tolerance: &self.plan.options.primal_error,
            system: self.problem,
            events,
            delay: None,
            provider: crate::stiff::stage::Provider::Native,
            end,
        };
        let mut time = start;
        let rejected_before = context.counts().own().rejected();
        let result = crate::ivp::drive(
            &run,
            &mut self.work.ivp,
            &mut time,
            &mut self.work.primal_state,
            &mut (),
            &mut recorder,
            context,
        );
        recorder.report.rejected_steps += context.counts().own().rejected() - rejected_before;
        if self.work.ivp.steps.counts().accepted() > 0 {
            recorder.report.trust = self.work.ivp.trust;
        }
        if let Some(error) = self.work.ivp.last_step_error_estimate {
            recorder.report.last_step_error_estimate = Some(error);
        }
        let result = result?.finish;
        ivp_finish(result, recorder.report)
    }
    fn replay(
        &mut self,
        index: usize,
        step: Step<T::Real>,
        initial: &[T],
        candidate: &mut [T],
        frame: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let time = step.time;
        let width = step.width;
        let n = self.plan.tape.dimension;
        let m = n + 1;
        self.work.value.as_mut_slice()[..n].copy_from_slice(initial);
        self.work.value.as_mut_slice()[n] = T::ZERO;
        frame[..m].copy_from_slice(self.work.value.as_slice());
        let reuse = self.work.forward.cache.contains(Some(index));
        let gradient_width = self.work.current_objective_gradient.len();
        if reuse && gradient_width != 0 {
            self.work.objective_gradients[..gradient_width]
                .copy_from_slice(&self.work.current_objective_gradient);
        }
        {
            let system = Primal {
                gradients: RefCell::new(&mut self.work.objective_gradients),
                stage: core::cell::Cell::new(usize::from(reuse)),
                problem: self.problem,
                objective: self.objective,
                products: RefCell::new(&mut self.work.products),
            };
            let trace = StepTrace::default();
            perform_explicit_step(
                &system,
                time,
                &mut self.work.value,
                width,
                &self.plan.tableau,
                StepRequest {
                    accepted_step: Some(index),
                    self_metered: true,
                    ..StepRequest::new(&trace).with_dense_output()
                },
                &mut self.work.forward,
                None,
                context,
            )?;
        }
        for (i, stage) in self.work.forward.stages.iter().take(7).enumerate() {
            frame[(i + 1) * m..(i + 2) * m].copy_from_slice(stage.as_slice());
        }
        if gradient_width != 0 {
            self.work
                .current_objective_gradient
                .copy_from_slice(&self.work.objective_gradients[..gradient_width]);
            frame[8 * m..].copy_from_slice(&self.work.objective_gradients);
        }
        let fraction = step.fraction;
        if fraction != T::Real::ONE {
            self.work
                .forward
                .dense_output
                .eval_into(time + width * fraction, self.work.value.as_mut_slice())?;
        }
        candidate.copy_from_slice(&self.work.value.as_slice()[..n]);
        self.accept(time)?;
        Ok(())
    }
    fn terminal(
        &mut self,
        time: T::Real,
        state: &[T],
        event: bool,
        last: Option<Step<T::Real>>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        let value = self.work.products.objective(
            self.objective,
            AdjointObjectivePart::Terminal,
            time,
            state,
            self.problem.parameters(),
            context,
        )?;
        lambda.copy_from_slice(&self.work.products.state);
        gradient.copy_from_slice(&self.work.products.parameters);
        if event {
            let event = self.event.ok_or_else(|| {
                invalid(scioxide_core::argument!("event"), "missing terminal event")
            })?;
            let step = last.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("event"),
                    "initial event derivatives are unavailable",
                )
            })?;
            self.load_frame(step, frame)?;
            let n = lambda.len();
            if self.plan.options.mode == AdjointMode::Discrete {
                self.work
                    .forward
                    .dense_output
                    .derivative_into(time, &mut self.work.stage_state)?;
            } else {
                admit(context, EvaluationRole::RightHandSide)?;
                self.problem
                    .rhs(time, state, &mut self.work.stage_state[..n], context)?;
                let running = self.work.products.objective(
                    self.objective,
                    AdjointObjectivePart::Running,
                    time,
                    state,
                    self.problem.parameters(),
                    context,
                )?;
                self.work.stage_state[n] = T::from_real(running.value);
            }
            let numerator = value.time_derivative
                + pairing(lambda, &self.work.stage_state[..n])?
                + self.work.stage_state[n].real_component();
            admit(context, EvaluationRole::Gradient)?;
            self.work.products.state.fill(T::from_real(T::Real::NAN));
            self.work.products.parameters.fill(T::Real::NAN);
            let gt = event.gradient(
                time,
                state,
                &mut self.work.products.state,
                &mut self.work.products.parameters,
            )?;
            finite(
                &self.work.products.state,
                scioxide_core::NonFiniteOrigin::Callback,
            )?;
            finite(
                &self.work.products.parameters,
                scioxide_core::NonFiniteOrigin::Callback,
            )?;
            let slope = gt + pairing(&self.work.products.state, &self.work.stage_state[..n])?;
            if !slope.is_finite() || slope.abs() <= event.transversality_threshold() {
                report.termination = Some(AdjointTermination::EventDerivativeUnavailable);
                return Err(invalid(
                    scioxide_core::argument!("event"),
                    "terminal event is nontransverse",
                ));
            }
            let multiplier = numerator / slope;
            for (out, &g) in lambda.iter_mut().zip(&self.work.products.state) {
                *out = *out - T::from_real(multiplier) * g;
            }
            for (out, &g) in gradient.iter_mut().zip(&self.work.products.parameters) {
                *out -= multiplier * g;
            }
        }
        Ok(value.value)
    }
    fn reverse(
        &mut self,
        step: Step<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        match self.plan.options.mode {
            AdjointMode::Discrete => self.discrete(step, frame, lambda, gradient, context),
            AdjointMode::Continuous => {
                self.continuous(step, frame, lambda, gradient, report, context)
            }
        }
    }
    fn initial(
        &mut self,
        _: T::Real,
        _: &[T],
        _: &mut [T],
        _: &mut [T::Real],
        _: &mut AdjointReport<T::Real>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}

fn pairing<T: NumericScalar>(left: &[T], right: &[T]) -> Result<T::Real, SciOxideError> {
    let mut sum = scioxide_core::RangeAccumulator::new();
    for (&a, &b) in left.iter().zip(right) {
        sum.add_product(a.real_component(), b.real_component())?;
        sum.add_product(a.imaginary_component(), b.imaginary_component())?;
    }
    sum.finish()
}

impl<T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>> OdeKernel<'_, T, P, O> {
    fn load_frame(&mut self, step: Step<T::Real>, frame: &[T]) -> Result<(), SciOxideError> {
        self.work.forward.reset_epoch();
        let m = self.plan.tape.dimension + 1;
        self.work.value.as_mut_slice().copy_from_slice(&frame[..m]);
        for (i, stage) in self.work.forward.stages.iter_mut().take(7).enumerate() {
            stage
                .as_mut_slice()
                .copy_from_slice(&frame[(i + 1) * m..(i + 2) * m]);
        }
        for (i, out) in self
            .work
            .forward
            .trial
            .as_mut_slice()
            .iter_mut()
            .enumerate()
        {
            *out = crate::safety::stage_affine(
                frame[i],
                step.width,
                self.plan
                    .tableau
                    .b
                    .iter()
                    .enumerate()
                    .map(|(j, &b)| (b, frame[(j + 1) * m + i])),
            )?;
        }
        self.work.forward.dense_output.prepare_rk(
            DenseExtension::DormandPrince5,
            step.width,
            &self.work.value,
            &self.work.forward.trial,
            &self.work.forward.stages[0],
            &self.work.forward.stages[6],
            &self.work.forward.stages,
            &self.plan.tableau.dense[..1],
        )?;
        self.work.forward.dense_output.finish(
            DenseExtension::DormandPrince5,
            step.time,
            step.width,
            &self.work.value,
            &self.work.forward.trial,
        )?;
        Ok(())
    }
    fn discrete(
        &mut self,
        step: Step<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = lambda.len();
        let m = n + 1;
        for (j, &dense_weight) in DENSE_WEIGHTS.iter().enumerate() {
            let b = self.plan.tableau.b[j];
            let theta = step.fraction;
            let rest = T::Real::ONE - theta;
            let first = if j == 0 { T::Real::ONE } else { T::Real::ZERO };
            let last = if j == 6 { T::Real::ONE } else { T::Real::ZERO };
            let weight = if theta == T::Real::ONE {
                b
            } else {
                theta
                    * (b + rest
                        * (first - b
                            + theta
                                * (b + b - first - last + rest * scalar::<T::Real>(dense_weight)?)))
            };
            for (i, &value) in lambda.iter().enumerate() {
                self.work.stage_bars[j * m + i] = T::from_real(step.width * weight) * value;
            }
            self.work.stage_bars[j * m + n] = T::from_real(step.width * weight);
        }
        for j in (0..7).rev() {
            context.execution().check()?;
            // A zero seed has a zero pullback: on full steps b₇ = 0 seeds the last stage
            // with nothing, so its pullback is skipped.
            if self.work.stage_bars[j * m..j * m + n]
                .iter()
                .all(|&seed| seed == T::ZERO)
            {
                self.work.stage_bar.fill(T::ZERO);
            } else {
                for (i, out) in self.work.stage_state.iter_mut().enumerate() {
                    *out = crate::safety::stage_affine(
                        frame[i],
                        step.width,
                        self.plan.tableau.a[j][..j]
                            .iter()
                            .enumerate()
                            .map(|(k, &a)| (a, frame[(k + 1) * m + i])),
                    )?;
                }
                let time = crate::safety::checked_stage_time(
                    step.time,
                    step.width,
                    self.plan.tableau.c[j],
                )?;
                self.work.products.pullback(
                    self.problem,
                    time,
                    &self.work.stage_state[..n],
                    &self.work.stage_bars[j * m..j * m + n],
                    context,
                )?;
                self.work
                    .stage_bar
                    .copy_from_slice(&self.work.products.state);
                for (out, &value) in gradient.iter_mut().zip(&self.work.products.parameters) {
                    *out += value;
                }
            }
            let gradient_width = n + gradient.len();
            let offset = 8 * m + j * gradient_width;
            self.work
                .products
                .state
                .copy_from_slice(&frame[offset..offset + n]);
            for (out, &value) in self
                .work
                .products
                .parameters
                .iter_mut()
                .zip(&frame[offset + n..offset + gradient_width])
            {
                *out = value.real_component();
            }
            let seed = self.work.stage_bars[j * m + n].real_component();
            for (out, &value) in self
                .work
                .stage_bar
                .iter_mut()
                .zip(&self.work.products.state)
            {
                *out = *out + T::from_real(seed) * value;
            }
            for (out, &value) in gradient.iter_mut().zip(&self.work.products.parameters) {
                *out += seed * value;
            }
            for (out, &value) in lambda.iter_mut().zip(&self.work.stage_bar) {
                *out = *out + value;
            }
            for (k, &a) in self.plan.tableau.a[j][..j].iter().enumerate() {
                for (out, &value) in self.work.stage_bars[k * m..k * m + n]
                    .iter_mut()
                    .zip(&self.work.stage_bar)
                {
                    *out = *out + T::from_real(step.width * a) * value;
                }
            }
        }
        finite(lambda, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(gradient, scioxide_core::NonFiniteOrigin::Callback)
    }
    fn continuous(
        &mut self,
        step: Step<T::Real>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.load_frame(step, frame)?;
        let n = lambda.len();
        self.work.reverse.as_mut_slice()[..n].copy_from_slice(lambda);
        for (out, &value) in self.work.reverse.as_mut_slice()[n..]
            .iter_mut()
            .zip(gradient.iter())
        {
            *out = T::from_real(value);
        }
        let mut time = step.time + step.width * step.fraction;
        if time == step.time {
            report.errors.interpolation_defect = self
                .plan
                .options
                .interpolation_defect
                .then_some(report.errors.interpolation_defect.unwrap_or(T::Real::ZERO));
            return Ok(());
        }
        let restart = self.work.reverse_time != Some(time);
        let defect =
            core::cell::Cell::new(report.errors.interpolation_defect.unwrap_or(T::Real::ZERO));
        let system = Continuous {
            defect: self.plan.options.interpolation_defect.then_some(&defect),
            problem: self.problem,
            objective: self.objective,
            dense: &self.work.forward.dense_output,
            products: RefCell::new(&mut self.work.products),
        };
        let method = crate::DormandPrince5::new();
        let options = crate::OdeOptions::new().with_initial_step((time - step.time).abs());
        let run = crate::ivp::Drive {
            method: &method,
            info: crate::stepper::Stepper::<T::Real, T>::info(&method),
            options: &options,
            tolerance: &self.plan.options.adjoint_error,
            system: &system,
            events: &[],
            delay: None,
            provider: crate::stiff::stage::Provider::Native,
            end: step.time,
        };
        let shared_report = RefCell::new(&mut *report);
        let mut recorder = super::reverse::Reverse {
            options: &self.plan.options,
            dimension: n,
            attempts: &mut self.work.reverse_attempts,
            limit: self.plan.tape.limits.max_reverse_steps,
            restart,
            report: &shared_report,
            adjoint: T::Real::ZERO,
            quadrature: T::Real::ZERO,
        };
        let result = crate::ivp::drive(
            &run,
            &mut self.work.backward,
            &mut time,
            &mut self.work.reverse,
            &mut (),
            &mut recorder,
            context,
        );
        shared_report.borrow_mut().errors.interpolation_defect = self
            .plan
            .options
            .interpolation_defect
            .then_some(defect.get());
        ivp_finish(result?.finish, &mut shared_report.borrow_mut())?;
        self.work.reverse_time = Some(time);
        lambda.copy_from_slice(&self.work.reverse.as_slice()[..n]);
        for (out, value) in gradient.iter_mut().zip(&self.work.reverse.as_slice()[n..]) {
            *out = value.real_component();
        }
        Ok(())
    }
}

// Self-metered continuous adjoint system: admits every callback it makes in `cx`.
struct Continuous<'a, T: NumericScalar, P, O> {
    defect: Option<&'a core::cell::Cell<T::Real>>,
    problem: &'a P,
    objective: &'a O,
    dense: &'a crate::dense_output::DenseOutputStorage<T::Real, T>,
    products: RefCell<&'a mut Products<T>>,
}
impl<T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>> OdeSystem<T::Real, T>
    for Continuous<'_, T, P, O>
{
    fn dimension(&self) -> usize {
        self.problem.dimension() + self.problem.parameters().len()
    }
    fn rhs(
        &self,
        time: T::Real,
        state: &[T],
        output: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.problem.dimension();
        let mut work = self.products.borrow_mut();
        self.dense.eval_into(time, &mut work.point)?;
        let Products {
            point,
            slope,
            state: state_gradient,
            parameters: parameter_gradient,
            provider,
            objective,
        } = &mut **work;
        if let Some(defect) = self.defect {
            self.dense.derivative_into(time, slope)?;
            admit(context, EvaluationRole::RightHandSide)?;
            state_gradient.fill(T::from_real(T::Real::NAN));
            self.problem
                .rhs(time, &point[..n], state_gradient, context)?;
            finite(state_gradient, scioxide_core::NonFiniteOrigin::Callback)?;
            for (&a, &b) in state_gradient.iter().zip(&slope[..n]) {
                let difference = (a - b).abs();
                if !difference.is_finite() {
                    return Err(invalid(
                        scioxide_core::argument!("problem"),
                        "nonfinite interpolation defect",
                    ));
                }
                defect.set(defect.get().max(difference));
            }
        }
        admit(context, EvaluationRole::LinearOperator)?;
        state_gradient.fill(T::from_real(T::Real::NAN));
        parameter_gradient.fill(T::Real::NAN);
        self.problem.pullback(
            time,
            &point[..n],
            &state[..n],
            state_gradient,
            parameter_gradient,
            provider,
            context,
        )?;
        finite(state_gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(parameter_gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        for (out, &value) in output[..n].iter_mut().zip(state_gradient.iter()) {
            *out = -value;
        }
        for (out, &value) in output[n..].iter_mut().zip(parameter_gradient.iter()) {
            *out = -T::from_real(value);
        }
        admit(context, EvaluationRole::Objective)?;
        state_gradient.fill(T::from_real(T::Real::NAN));
        parameter_gradient.fill(T::Real::NAN);
        self.objective.evaluate(
            AdjointObjectivePart::Running,
            time,
            &point[..n],
            self.problem.parameters(),
            state_gradient,
            parameter_gradient,
            objective,
            context,
        )?;
        finite(state_gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(parameter_gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        for (out, &value) in output[..n].iter_mut().zip(state_gradient.iter()) {
            *out = *out - value;
        }
        for (out, &value) in output[n..].iter_mut().zip(parameter_gradient.iter()) {
            *out = *out - T::from_real(value);
        }
        finite(output, scioxide_core::NonFiniteOrigin::Callback)
    }
}

struct TerminalEvent<'a, T: NumericScalar> {
    event: Option<&'a dyn AdjointTerminalEvent<T>>,
    state: RefCell<&'a mut [T]>,
    parameters: RefCell<&'a mut [T::Real]>,
}
impl<T: NumericScalar> crate::EventFunction<T::Real, T> for TerminalEvent<'_, T> {
    fn eval(&self, time: T::Real, state: &[T]) -> Result<T::Real, SciOxideError> {
        self.event
            .ok_or_else(|| invalid(scioxide_core::argument!("event"), "missing terminal event"))?
            .value(time, state)
    }
    fn is_terminal(&self) -> bool {
        true
    }
    fn direction(&self) -> crate::EventDirection {
        self.event
            .map_or(crate::EventDirection::Either, |event| event.direction())
    }
    fn has_total_derivative(&self) -> bool {
        true
    }
    fn total_derivative(
        &self,
        time: T::Real,
        state: &[T],
        slope: &[T],
    ) -> Result<T::Real, SciOxideError> {
        let mut gradient = self.state.borrow_mut();
        gradient.fill(T::from_real(T::Real::NAN));
        self.parameters.borrow_mut().fill(T::Real::NAN);
        let partial = self
            .event
            .ok_or_else(|| invalid(scioxide_core::argument!("event"), "missing terminal event"))?
            .gradient(
                time,
                state,
                &mut gradient,
                &mut self.parameters.borrow_mut(),
            )?;
        finite(&gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(
            &self.parameters.borrow(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        Ok(partial + pairing(&gradient, slope)?)
    }
}

struct Forward<'a, T: NumericScalar, P, O> {
    plan: &'a PreparedOdeAdjoint<T>,
    problem: &'a P,
    objective: &'a O,
    tape: &'a mut Tape<T>,
    products: &'a mut Products<T>,
    stage_state: &'a mut [T],
    gradients: &'a mut [T],
    current_gradient: &'a mut [T],
    cached: Option<T::Real>,
    cached_cost: T::Real,
    error: T::Real,
    quadrature: T::Real,
    end: T::Real,
    cost: &'a mut T::Real,
    report: &'a mut AdjointReport<T::Real>,
}
impl<T: NumericScalar, P: OdeAdjointSystem<T>, O: AdjointObjective<T>>
    crate::ivp::Recorder<T::Real, T> for Forward<'_, T, P, O>
{
    fn boundary(
        &mut self,
        time: T::Real,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<T::Real>, SciOxideError> {
        super::checkpoint::check_parameters(&self.tape.parameters, self.problem.parameters())?;
        self.tape.boundary(
            &self.plan.tape,
            &self.plan.observations,
            time,
            self.end,
            self.report,
        )
    }
    fn admit(&mut self, context: &mut SolveContext<'_, '_>) -> Result<(), Fault> {
        crate::termination::record(context, Work::Proposal)
    }
    fn fixed_step(&self) -> bool {
        self.plan.options.fixed_step
    }
    fn requires_dense(&self) -> bool {
        true
    }
    fn trial(
        &mut self,
        time: T::Real,
        width: T::Real,
        initial: &Vector<T>,
        candidate: &Vector<T>,
        step: &crate::result::StepResult<'_, T::Real, T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        let n = self.plan.tape.dimension;
        let m = n + 1;
        let stages = step.stages.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                "DP5 stages required for recording",
            )
        })?;
        self.tape.frame[..n].copy_from_slice(initial.as_slice());
        self.tape.frame[n] = T::ZERO;
        let gradient_width = self.current_gradient.len();
        for j in 0..7 {
            self.tape.frame[(j + 1) * m..(j + 1) * m + n].copy_from_slice(stages[j].as_slice());
            let value = if j == 0 && self.cached == Some(time) {
                if gradient_width != 0 {
                    self.gradients[..gradient_width].copy_from_slice(self.current_gradient);
                }
                self.cached_cost
            } else {
                crate::safety::combine_stages(
                    Some(initial.as_slice()),
                    width,
                    &self.plan.tableau.a[j][..j],
                    &stages[..j],
                    &mut self.stage_state[..n],
                )?;
                let stage_time =
                    crate::safety::prepared_stage_time(time, width, self.plan.tableau.c[j]);
                let result = self.products.objective(
                    self.objective,
                    AdjointObjectivePart::Running,
                    stage_time,
                    &self.stage_state[..n],
                    self.problem.parameters(),
                    context,
                )?;
                if gradient_width != 0 {
                    self.gradients[j * gradient_width..j * gradient_width + n]
                        .copy_from_slice(&self.products.state);
                    for (out, &value) in self.gradients
                        [j * gradient_width + n..(j + 1) * gradient_width]
                        .iter_mut()
                        .zip(&self.products.parameters)
                    {
                        *out = T::from_real(value);
                    }
                }
                result.value
            };
            self.tape.frame[(j + 1) * m + n] = T::from_real(value);
        }
        self.cached = Some(time);
        self.cached_cost = self.tape.frame[m + n].real_component();
        if gradient_width != 0 {
            self.current_gradient
                .copy_from_slice(&self.gradients[..gradient_width]);
            self.tape.frame[8 * m..self.plan.tape.numerical_frame_len]
                .copy_from_slice(self.gradients);
        }
        self.error = self.plan.options.primal_error.normalized_norm_by(n, |i| {
            (
                step.raw_error.map_or(T::Real::ZERO, |e| e[i].abs()),
                initial.as_slice()[i]
                    .abs()
                    .max(candidate.as_slice()[i].abs()),
            )
        })?;
        let cost = crate::safety::stage_affine(
            T::Real::ZERO,
            width,
            self.plan
                .tableau
                .b
                .iter()
                .enumerate()
                .map(|(j, &b)| (b, self.tape.frame[(j + 1) * m + n].real_component())),
        )?;
        let errors = &self.plan.tableau.error;
        let error = crate::safety::stage_affine(
            T::Real::ZERO,
            width,
            errors
                .iter()
                .enumerate()
                .map(|(j, &b)| (b, self.tape.frame[(j + 1) * m + n].real_component())),
        )?;
        self.quadrature = self
            .plan
            .options
            .quadrature_error
            .normalized_norm(&[error.abs()], &[cost.abs()])?;
        Ok(self.quadrature)
    }
    fn accepted(
        &mut self,
        start: T::Real,
        width: T::Real,
        end: T::Real,
        state: &Vector<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.plan.tape.dimension;
        let m = n + 1;
        let theta = if end == start + width {
            T::Real::ONE
        } else {
            (end - start) / width
        };
        let rest = T::Real::ONE - theta;
        let mut weights = self.plan.tableau.b;
        if theta != T::Real::ONE {
            for (j, weight) in weights.iter_mut().enumerate() {
                let first = if j == 0 { T::Real::ONE } else { T::Real::ZERO };
                let last = if j == 6 { T::Real::ONE } else { T::Real::ZERO };
                let b = *weight;
                *weight = theta
                    * (b + rest
                        * (first - b
                            + theta
                                * (b + b - first - last
                                    + rest * scalar::<T::Real>(DENSE_WEIGHTS[j])?)));
            }
        }
        let cost = crate::safety::stage_affine(
            T::Real::ZERO,
            width,
            weights
                .iter()
                .enumerate()
                .map(|(j, &b)| (b, self.tape.frame[(j + 1) * m + n].real_component())),
        )?;
        self.products.observations(
            self.objective,
            &self.plan.observations,
            end,
            state.as_slice(),
            self.problem.parameters(),
            &mut self.tape.frame[self.plan.tape.numerical_frame_len..],
            context,
        )?;
        super::checkpoint::check_parameters(&self.tape.parameters, self.problem.parameters())?;
        self.tape.candidate.copy_from_slice(state.as_slice());
        self.cached = Some(end);
        self.cached_cost = self.tape.frame[7 * m + n].real_component();
        let gradient_width = self.current_gradient.len();
        if gradient_width != 0 {
            self.current_gradient
                .copy_from_slice(&self.gradients[6 * gradient_width..7 * gradient_width]);
        }
        self.tape.push(
            &self.plan.tape,
            start,
            width,
            end,
            Trial {
                error: Some(self.error),
                quadrature_error: Some(self.quadrature),
                cost,
                fraction: theta,
                metadata: (),
            },
            self.cost,
            self.report,
        )
    }
}
