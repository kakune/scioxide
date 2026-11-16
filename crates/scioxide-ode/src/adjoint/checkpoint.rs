//! One bounded trajectory recorder and checkpoint replay controller for both families.
use super::*;
use scioxide_ad::{CheckpointAction, CheckpointSchedule};

/// Selects which accepted-step states an adjoint plan stores and which it recomputes.
///
/// Storing the restart state of every accepted step makes the reverse pass cheapest.
/// Recomputing states from checkpoints trades forward steps for memory; the
/// schedules are those of [`CheckpointSchedule`] (Griewank and Walther, 2000).
///
/// ```
/// use scioxide_ad::CheckpointSchedule;
/// use scioxide_ode::{AdjointMode, AdjointOptions, CheckpointPolicy};
///
/// let schedule = CheckpointSchedule::binomial(64, 2)?;
/// let options = AdjointOptions::new(AdjointMode::Discrete, 0.01)?
///     .with_tape_capacity(64)
///     .with_checkpoint_policy(CheckpointPolicy::Explicit(schedule));
/// assert!(matches!(options.checkpoint_policy(), CheckpointPolicy::Explicit(_)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. Griewank and A. Walther, "Algorithm 799: revolve", *ACM Trans. Math. Softw.*
/// 26(1), 2000.
#[derive(Debug, Clone, Default)]
pub enum CheckpointPolicy {
    /// Stores the restart state of every accepted step, so the reverse pass never
    /// advances the forward solution again.
    #[default]
    StoreAll,
    /// Stores every state when the whole workspace fits the byte limit of
    /// [`AdjointOptions::with_max_workspace_bytes`]; otherwise uses the binomial
    /// schedule with the most slots that fits it and the replay limit.
    MemoryBudget,
    /// Uses the given schedule. It must cover the tape capacity, and its forward
    /// steps must stay within the replay limit.
    Explicit(CheckpointSchedule),
}
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul};

#[derive(Clone, Copy)]
pub(super) struct Step<S: RealScalar, M = ()> {
    pub time: S,
    pub endpoint: S,
    pub width: S,
    pub fraction: S,
    pub signature: u64,
    pub metadata: M,
}
impl<S: RealScalar, M: Copy + Default> Step<S, M> {
    fn empty() -> Self {
        Self {
            time: S::ZERO,
            endpoint: S::ZERO,
            width: S::ZERO,
            fraction: S::ONE,
            signature: 0,
            metadata: M::default(),
        }
    }
}

/// One accepted step. Absent error estimates leave the report diagnostics untouched.
pub(super) struct Trial<S: RealScalar, M = ()> {
    pub error: Option<S>,
    pub quadrature_error: Option<S>,
    pub cost: S,
    pub fraction: S,
    pub metadata: M,
}

pub(super) trait Kernel<T: NumericScalar> {
    type Metadata: Copy + Default;
    fn accept(&mut self, _time: T::Real) -> Result<(), SciOxideError> {
        Ok(())
    }
    /// Discard point-local caches after restoring a checkpoint.
    fn restart(&mut self) {}
    fn observations(&self) -> &[T::Real];
    fn sample(
        &mut self,
        time: T::Real,
        state: &[T],
        packed: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    fn parameters(&self) -> &[T::Real];
    fn record(
        &mut self,
        tape: &mut Tape<T, Self::Metadata>,
        start: T::Real,
        end: T::Real,
        cost: &mut T::Real,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError>;
    fn replay(
        &mut self,
        index: usize,
        step: Step<T::Real, Self::Metadata>,
        initial: &[T],
        candidate: &mut [T],
        frame: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    #[allow(clippy::too_many_arguments)]
    fn terminal(
        &mut self,
        time: T::Real,
        state: &[T],
        event: bool,
        last: Option<Step<T::Real, Self::Metadata>>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError>;
    fn reverse(
        &mut self,
        step: Step<T::Real, Self::Metadata>,
        frame: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    fn initial(
        &mut self,
        time: T::Real,
        state: &[T],
        lambda: &mut [T],
        gradient: &mut [T::Real],
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Storage lengths of a tape; workspaces serve exactly one shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TapeShape {
    restart: usize,
    dimension: usize,
    parameters: usize,
    frame: usize,
    sample: usize,
    steps: usize,
    frames: usize,
    snapshots: usize,
    control: usize,
}
impl TapeShape {
    pub(super) const fn dimension(self) -> usize {
        self.dimension
    }
    pub(super) const fn parameters(self) -> usize {
        self.parameters
    }
}

#[derive(Debug, Clone)]
pub(super) struct TapePlan {
    pub schedule: Option<CheckpointSchedule>,
    pub limits: Limits,
    pub frame_len: usize,
    pub numerical_frame_len: usize,
    sample_len: usize,
    pub restart_len: usize,
    pub dimension: usize,
    pub parameters: usize,
    pub bytes: usize,
}
impl TapePlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<T: NumericScalar, M: Copy + Default>(
        dimension: usize,
        parameters: usize,
        frame_len: usize,
        restart_len: usize,
        base_bytes: usize,
        limits: Limits,
        policy: &CheckpointPolicy,
        observations: bool,
    ) -> Result<Self, SciOxideError> {
        if limits.tape_capacity == 0 || dimension == 0 {
            return Err(invalid(
                if dimension == 0 {
                    scioxide_core::argument!("problem")
                } else {
                    scioxide_core::argument!("options.tape_capacity")
                },
                "positive state and step capacities are required",
            ));
        }
        let numerical_frame_len = frame_len;
        let sample_len = if observations {
            checked_add(
                checked_add(
                    dimension,
                    parameters,
                    scioxide_core::quantity!("observation_sample"),
                )?,
                1,
                scioxide_core::quantity!("observation_sample"),
            )?
        } else {
            0
        };
        let frame_len = checked_add(
            frame_len,
            sample_len,
            scioxide_core::quantity!("checkpoint_frame"),
        )?;
        let base_bytes = checked_add(
            base_bytes,
            checked_mul(
                sample_len,
                core::mem::size_of::<T>(),
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let cap = limits.max_workspace_bytes;
        let scalar_bytes = |n| {
            checked_mul(
                n,
                core::mem::size_of::<T>(),
                scioxide_core::quantity!("workspace_bytes"),
            )
        };
        let fixed = checked_add(
            base_bytes,
            checked_add(
                checked_mul(
                    limits.tape_capacity,
                    core::mem::size_of::<Step<T::Real, M>>(),
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                checked_add(
                    scalar_bytes(checked_add(
                        checked_mul(restart_len, 4, scioxide_core::quantity!("restart_values"))?,
                        dimension,
                        scioxide_core::quantity!("workspace_bytes"),
                    )?)?,
                    checked_mul(
                        parameters,
                        2 * core::mem::size_of::<T::Real>(),
                        scioxide_core::quantity!("workspace_bytes"),
                    )?,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let bytes_for = |frames, snapshots, controls| -> Result<usize, SciOxideError> {
            checked_add(
                fixed,
                checked_add(
                    scalar_bytes(checked_add(
                        checked_mul(
                            frame_len,
                            frames,
                            scioxide_core::quantity!("checkpoint_frames"),
                        )?,
                        checked_mul(
                            restart_len,
                            snapshots,
                            scioxide_core::quantity!("workspace_bytes"),
                        )?,
                        scioxide_core::quantity!("workspace_bytes"),
                    )?)?,
                    checked_add(
                        checked_mul(
                            controls,
                            core::mem::size_of::<u64>(),
                            scioxide_core::quantity!("workspace_bytes"),
                        )?,
                        checked_mul(
                            snapshots,
                            core::mem::size_of::<Option<usize>>(),
                            scioxide_core::quantity!("workspace_bytes"),
                        )?,
                        scioxide_core::quantity!("workspace_bytes"),
                    )?,
                    scioxide_core::quantity!("workspace_bytes"),
                )?,
                scioxide_core::quantity!("workspace_bytes"),
            )
        };
        // Keep a separate working frame even for StoreAll; terminal seeding uses it.
        let store_bytes = bytes_for(
            checked_add(
                limits.tape_capacity,
                1,
                scioxide_core::quantity!("checkpoint_frames"),
            )?,
            0,
            0,
        )?;
        let (schedule, bytes) = match policy {
            CheckpointPolicy::StoreAll => (None, store_bytes),
            CheckpointPolicy::MemoryBudget if store_bytes <= cap => (None, store_bytes),
            CheckpointPolicy::MemoryBudget => {
                let mut selected = None;
                for slots in (0..=64.min(limits.tape_capacity.saturating_sub(1))).rev() {
                    let candidate = CheckpointSchedule::binomial(limits.tape_capacity, slots)?;
                    let bytes = bytes_for(1, slots + 1, candidate.control_slots())?;
                    if bytes <= cap && candidate.forward_steps() <= limits.max_replay_steps {
                        selected = Some((Some(candidate), bytes));
                        break;
                    }
                }
                selected.ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("options.max_workspace_bytes"),
                        "workspace/replay limits cannot admit a checkpoint schedule",
                    )
                })?
            }
            CheckpointPolicy::Explicit(schedule) => {
                if schedule.steps() != limits.tape_capacity
                    || schedule.forward_steps() > limits.max_replay_steps
                {
                    return Err(invalid(
                        scioxide_core::argument!("options.checkpoint_policy"),
                        "explicit schedule must cover the step capacity within the replay limit",
                    ));
                }
                let bytes =
                    bytes_for(1, schedule.checkpoint_slots() + 1, schedule.control_slots())?;
                (Some(schedule.clone()), bytes)
            }
        };
        if bytes > cap {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(bytes),
                scioxide_core::Dimensions::one(cap),
            ));
        }
        Ok(Self {
            schedule,
            limits,
            frame_len,
            numerical_frame_len,
            sample_len,
            restart_len,
            dimension,
            parameters,
            bytes,
        })
    }

    pub(super) fn shape(&self) -> TapeShape {
        let schedule = self.schedule.as_ref();
        TapeShape {
            restart: self.restart_len,
            dimension: self.dimension,
            parameters: self.parameters,
            frame: self.frame_len,
            sample: self.sample_len,
            steps: self.limits.tape_capacity,
            frames: if schedule.is_none() {
                self.limits.tape_capacity.saturating_mul(self.frame_len)
            } else {
                0
            },
            snapshots: schedule.map_or(0, |s| s.checkpoint_slots() + 1),
            control: schedule.map_or(0, CheckpointSchedule::control_slots),
        }
    }
}

pub(super) struct Tape<T: NumericScalar, M = ()> {
    pub state: Vec<T>,
    pub candidate: Vec<T>,
    pub initial: Vec<T>,
    pub final_state: Vec<T>,
    pub lambda: Vec<T>,
    pub gradient: Vec<T::Real>,
    pub(super) parameters: Vec<T::Real>,
    pub frame: Vec<T>,
    initial_sample: Vec<T>,
    pub(super) frames: Vec<T>,
    restarts: Vec<T>,
    control: Vec<u64>,
    restart_steps: Vec<Option<usize>>,
    pub(super) steps: Vec<Step<T::Real, M>>,
}
impl<T: NumericScalar, M: Copy + Default> Tape<T, M>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    pub(super) fn fits(&self, shape: TapeShape) -> bool {
        self.state.len() == shape.restart
            && self.candidate.len() == shape.restart
            && self.initial.len() == shape.restart
            && self.final_state.len() == shape.restart
            && self.lambda.len() == shape.dimension
            && self.gradient.len() == shape.parameters
            && self.parameters.len() == shape.parameters
            && self.frame.len() == shape.frame
            && self.initial_sample.len() == shape.sample
            && self.steps.len() == shape.steps
            && self.frames.len() == shape.frames
            && self.restarts.len() == shape.snapshots.saturating_mul(shape.restart)
            && self.restart_steps.len() == shape.snapshots
            && self.control.len() == shape.control
    }
    pub(super) fn new(shape: TapeShape) -> Result<Self, SciOxideError> {
        let filled = |len, quantity| scioxide_core::backend::try_filled(len, T::ZERO, quantity);
        let real = |len, quantity| scioxide_core::backend::try_filled(len, T::Real::ZERO, quantity);
        Ok(Self {
            state: filled(shape.restart, scioxide_core::quantity!("state"))?,
            candidate: filled(shape.restart, scioxide_core::quantity!("candidate"))?,
            initial: filled(shape.restart, scioxide_core::quantity!("initial"))?,
            final_state: filled(shape.restart, scioxide_core::quantity!("final_state"))?,
            lambda: filled(shape.dimension, scioxide_core::quantity!("lambda"))?,
            gradient: real(shape.parameters, scioxide_core::quantity!("gradient"))?,
            parameters: real(shape.parameters, scioxide_core::quantity!("parameters"))?,
            frame: filled(shape.frame, scioxide_core::quantity!("frame"))?,
            initial_sample: filled(shape.sample, scioxide_core::quantity!("initial_sample"))?,
            frames: filled(shape.frames, scioxide_core::quantity!("frames"))?,
            restarts: filled(
                checked_mul(
                    shape.snapshots,
                    shape.restart,
                    scioxide_core::quantity!("restarts"),
                )?,
                scioxide_core::quantity!("restarts"),
            )?,
            restart_steps: scioxide_core::backend::try_filled(
                shape.snapshots,
                None,
                scioxide_core::quantity!("checkpoint_indices"),
            )?,
            control: scioxide_core::backend::try_filled(
                shape.control,
                0,
                scioxide_core::quantity!("control"),
            )?,
            steps: scioxide_core::backend::try_filled(
                shape.steps,
                Step::empty(),
                scioxide_core::quantity!("steps"),
            )?,
        })
    }

    pub(super) fn bind_parameters(&mut self, parameters: &[T::Real]) -> Result<(), SciOxideError> {
        if parameters.len() != self.parameters.len() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "parameter shape changed before binding",
            ));
        }
        finite(parameters, scioxide_core::NonFiniteOrigin::Callback)?;
        self.parameters.copy_from_slice(parameters);
        Ok(())
    }
    pub(super) fn boundary(
        &self,
        plan: &TapePlan,
        observations: &[T::Real],
        time: T::Real,
        end: T::Real,
        report: &mut AdjointReport<T::Real>,
    ) -> Result<Option<T::Real>, SciOxideError> {
        if report.accepted_steps == plan.limits.tape_capacity {
            report.termination = Some(AdjointTermination::BudgetLimit(BudgetKind::Iteration));
            return Err(convergence(
                scioxide_core::ConvergenceReason::IterationLimit,
                "accepted-step recording limit exhausted",
            ));
        }
        if end >= time {
            Ok(observations
                .get(observations.partition_point(|&t| t <= time))
                .copied())
        } else {
            Ok(observations
                .partition_point(|&t| t < time)
                .checked_sub(1)
                .map(|i| observations[i]))
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push(
        &mut self,
        plan: &TapePlan,
        time: T::Real,
        width: T::Real,
        endpoint: T::Real,
        trial: Trial<T::Real, M>,
        cost: &mut T::Real,
        report: &mut AdjointReport<T::Real>,
    ) -> Result<(), SciOxideError> {
        let actual_endpoint = endpoint;
        let step = Step {
            time,
            endpoint: actual_endpoint,
            width,
            fraction: trial.fraction,
            metadata: trial.metadata,
            signature: if plan.schedule.is_some() {
                signature(&self.candidate, &self.frame)
            } else {
                0
            },
        };
        self.steps[report.accepted_steps] = step;
        if plan.schedule.is_none() {
            let offset = report.accepted_steps * plan.frame_len;
            self.frames[offset..offset + plan.frame_len].copy_from_slice(&self.frame);
        }
        self.state.copy_from_slice(&self.candidate);

        *cost = *cost
            + trial.cost
            + self.frame[plan.numerical_frame_len..]
                .last()
                .map_or(T::Real::ZERO, |v| v.real_component());
        if !(*cost).is_finite() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "nonfinite accumulated objective",
            ));
        }
        report.accepted_steps += 1;
        report.time = actual_endpoint;
        let merge = |current: Option<T::Real>, estimate: Option<T::Real>| match estimate {
            Some(estimate) => Some(current.unwrap_or(T::Real::ZERO).max(estimate)),
            None => current,
        };
        report.errors.primal = merge(report.errors.primal, trial.error);
        report.errors.quadrature = merge(report.errors.quadrature, trial.quadrature_error);
        if let Some(boundaries) = plan
            .schedule
            .as_ref()
            .and_then(CheckpointSchedule::boundaries)
            && let Ok(slot) = boundaries.binary_search(&report.accepted_steps)
        {
            self.restart_steps[slot] = Some(report.accepted_steps);
            self.restarts[slot * plan.restart_len..(slot + 1) * plan.restart_len]
                .copy_from_slice(&self.state);
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &mut self,
        plan: &TapePlan,
        options: &Policies<T::Real>,
        kernel: &mut impl Kernel<T, Metadata = M>,
        start: T::Real,
        end: T::Real,
        report: &mut AdjointReport<T::Real>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        check_parameters(&self.parameters, kernel.parameters())?;
        let times = kernel.observations();
        if times
            .iter()
            .any(|&t| t < start.min(end) || t > start.max(end))
        {
            return Err(invalid(
                scioxide_core::argument!("interval"),
                "observation time outside solve interval",
            ));
        }
        kernel.sample(start, &self.state, &mut self.initial_sample, context)?;
        self.initial.copy_from_slice(&self.state);
        if plan.schedule.is_some() {
            self.restart_steps.fill(None);
            self.restart_steps[0] = Some(0);
            self.restarts[..plan.restart_len].copy_from_slice(&self.state);
        }
        self.lambda.fill(T::ZERO);
        self.gradient.fill(T::Real::ZERO);
        let mut cost = self
            .initial_sample
            .last()
            .map_or(T::Real::ZERO, |v| v.real_component());
        let terminal_event = kernel.record(self, start, end, &mut cost, report, context)?;
        let time = report.time;
        self.final_state.copy_from_slice(&self.state);
        let last = report.accepted_steps.checked_sub(1).map(|i| self.steps[i]);
        let terminal = kernel.terminal(
            time,
            &self.state,
            terminal_event,
            last,
            &self.frame[..plan.numerical_frame_len],
            &mut self.lambda,
            &mut self.gradient,
            report,
            context,
        )?;
        check_parameters(&self.parameters, kernel.parameters())?;
        let objective = cost + terminal;
        if !objective.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "nonfinite objective",
            ));
        }
        report.objective = Some(objective);
        // Plan construction admits only schedules whose replay work at the tape capacity,
        // and hence for every realized step count, is within the replay limit.
        if let Some(schedule) = &plan.schedule {
            let exact = (0..plan.restart_len).all(|i| {
                options
                    .replay_error
                    .threshold(i, plan.restart_len, T::Real::ONE)
                    .is_ok_and(|x| x == T::Real::ZERO)
            });
            schedule.visit(report.accepted_steps, &mut self.control, |action| {
                context.execution().check()?;
                match action {
                    CheckpointAction::Restore { slot, .. } => {
                        kernel.restart();
                        self.state.copy_from_slice(
                            &self.restarts[slot * plan.restart_len..(slot + 1) * plan.restart_len],
                        );
                    }
                    CheckpointAction::Save { slot, step } => {
                        self.restart_steps[slot] = Some(step);
                        self.restarts[slot * plan.restart_len..(slot + 1) * plan.restart_len]
                            .copy_from_slice(&self.state);
                    }
                    CheckpointAction::Advance { start, end } => {
                        for index in start..end {
                            let step = self.steps[index];
                            report.replayed_steps += 1;
                            kernel.replay(
                                index,
                                step,
                                &self.state,
                                &mut self.candidate,
                                &mut self.frame[..plan.numerical_frame_len],
                                context,
                            )?;
                            kernel.sample(
                                step.endpoint,
                                &self.candidate,
                                &mut self.frame[plan.numerical_frame_len..],
                                context,
                            )?;
                            check_parameters(&self.parameters, kernel.parameters())?;
                            if exact && signature(&self.candidate, &self.frame) != step.signature {
                                report.termination = Some(AdjointTermination::ReplayMismatch);
                                return Err(convergence(
                                    scioxide_core::ConvergenceReason::ReplayMismatch,
                                    "replayed accepted state changed",
                                ));
                            }
                            let retained = self
                                .restart_steps
                                .iter()
                                .position(|&step| step == Some(index + 1));
                            let reference = if index + 1 == report.accepted_steps {
                                Some(&self.final_state[..])
                            } else {
                                retained.map(|slot| {
                                    &self.restarts
                                        [slot * plan.restart_len..(slot + 1) * plan.restart_len]
                                })
                            };
                            if let Some(reference) = reference {
                                for (i, (&expected, &actual)) in
                                    reference.iter().zip(&self.candidate).enumerate()
                                {
                                    let difference = (actual - expected).abs();
                                    report.errors.replay = report.errors.replay.max(difference);
                                    let threshold = options.replay_error.threshold(
                                        i,
                                        plan.restart_len,
                                        expected.abs().max(actual.abs()),
                                    )?;
                                    if difference > threshold {
                                        report.termination =
                                            Some(AdjointTermination::ReplayMismatch);
                                        return Err(convergence(
                                            scioxide_core::ConvergenceReason::ReplayMismatch,
                                            "replayed retained boundary exceeds replay tolerance",
                                        ));
                                    }
                                }
                            }
                            self.state.copy_from_slice(&self.candidate);
                        }
                    }
                    CheckpointAction::Reverse { step } => {
                        add_sample(
                            &self.frame[plan.numerical_frame_len..],
                            &mut self.lambda,
                            &mut self.gradient,
                        )?;
                        admit_discrete_reverse(options, report)?;
                        kernel.reverse(
                            self.steps[step],
                            &self.frame[..plan.numerical_frame_len],
                            &mut self.lambda,
                            &mut self.gradient,
                            report,
                            context,
                        )?;
                        report.reverse_steps += 1;
                    }
                }
                check_parameters(&self.parameters, kernel.parameters())
            })?;
        } else {
            for index in (0..report.accepted_steps).rev() {
                context.execution().check()?;
                let offset = index * plan.frame_len;
                add_sample(
                    &self.frames[offset + plan.numerical_frame_len..offset + plan.frame_len],
                    &mut self.lambda,
                    &mut self.gradient,
                )?;
                admit_discrete_reverse(options, report)?;
                kernel.reverse(
                    self.steps[index],
                    &self.frames[offset..offset + plan.numerical_frame_len],
                    &mut self.lambda,
                    &mut self.gradient,
                    report,
                    context,
                )?;
                report.reverse_steps += 1;
                check_parameters(&self.parameters, kernel.parameters())?;
            }
        }
        add_sample(&self.initial_sample, &mut self.lambda, &mut self.gradient)?;
        kernel.initial(
            start,
            &self.initial,
            &mut self.lambda,
            &mut self.gradient,
            report,
            context,
        )?;
        finite(&self.lambda, scioxide_core::NonFiniteOrigin::Callback)?;
        check_parameters(&self.parameters, kernel.parameters())?;
        finite(&self.gradient, scioxide_core::NonFiniteOrigin::Callback)?;
        Ok(terminal_event)
    }
}

// A bounded per-step fingerprint detects changed replay without an O(steps*states)
// shadow trajectory. Retained boundary comparisons additionally obey replay tolerance.
fn signature<T: NumericScalar>(state: &[T], frame: &[T]) -> u64 {
    state
        .iter()
        .chain(frame)
        .fold(0xcbf29ce484222325_u64, |hash, value| {
            [value.real_component(), value.imaginary_component()]
                .iter()
                .fold(hash, |hash, x| {
                    (hash ^ x.into_f64().to_bits()).wrapping_mul(0x100000001b3)
                })
        })
}

pub(super) fn check_parameters<S: RealScalar>(
    bound: &[S],
    current: &[S],
) -> Result<(), SciOxideError> {
    if bound.len() != current.len()
        || bound
            .iter()
            .zip(current)
            .any(|(&a, &b)| a.into_f64().to_bits() != b.into_f64().to_bits())
    {
        return Err(convergence(
            scioxide_core::ConvergenceReason::ReplayMismatch,
            "bound parameter values changed during the solve epoch",
        ));
    }
    Ok(())
}

fn add_sample<T: NumericScalar>(
    packed: &[T],
    lambda: &mut [T],
    gradient: &mut [T::Real],
) -> Result<(), SciOxideError> {
    if !packed.is_empty() {
        let n = lambda.len();
        for (out, &v) in lambda.iter_mut().zip(&packed[..n]) {
            *out = *out + v;
        }
        for (out, &v) in gradient.iter_mut().zip(&packed[n..]) {
            *out += v.real_component();
        }
        finite(lambda, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(gradient, scioxide_core::NonFiniteOrigin::Callback)?;
    }
    Ok(())
}

fn admit_discrete_reverse<S: RealScalar>(
    options: &Policies<S>,
    report: &mut AdjointReport<S>,
) -> Result<(), SciOxideError> {
    if options.mode == AdjointMode::Discrete && report.reverse_steps >= options.max_reverse_steps {
        report.termination = Some(AdjointTermination::BudgetLimit(BudgetKind::Iteration));
        return Err(convergence(
            scioxide_core::ConvergenceReason::IterationLimit,
            "discrete reverse-step limit exhausted",
        ));
    }
    Ok(())
}
