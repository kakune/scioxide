//! The Störmer–Verlet method (kick–drift–kick) for separable Hamiltonians.

use core::marker::PhantomData;

use alloc::vec::Vec;

use scioxide_core::{
    AttemptRecord, Capacity, EvaluationRole, ExecutionContext, NonFiniteOrigin, NumericScalar,
    PartialFailure, PartialResultTrust, RealScalar, SciOxideError, SolveContext, SolveScope,
};

use super::{
    MechanicalOptions, MechanicalOutputSink, MechanicalReport, MechanicalSolution, PhaseSpace,
    accept, admit, attempt, identity, partial, poison, proposal, validate_state,
};
use crate::safety::finite;
use crate::shell::Shell;
use crate::termination::Fault;

/// Defines a separable Hamiltonian `H(q, p) = T(p) + V(q)` for [`StormerVerlet`].
///
/// The equations of motion are
///
/// ```text
/// q′ = ∇T(p),   p′ = −∇V(q),
/// ```
///
/// where `q` are the positions, `p` the conjugate momenta, `T` the kinetic and `V` the
/// potential energy. `H` does not depend on time.
///
/// For a complex scalar type, each complex component holds two real coordinates, and
/// the callbacks return the real and imaginary parts of the gradient with respect to
/// those two coordinates, not a complex derivative. The callbacks must be
/// deterministic and write every output component; unwritten, NaN and infinite
/// components are a `NonFinite` error with origin `Callback`. Each call counts as one
/// evaluation of the step that makes it; charge only nested work to `cx`.
pub trait SeparableHamiltonian<T: NumericScalar> {
    /// Returns the number of positions, which equals the number of momenta.
    fn dimension(&self) -> usize;
    /// Evaluates `∇T(p)` into `output`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the call returns unchanged.
    fn velocity(
        &self,
        momentum: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Evaluates `∇V(q)`, the negative force, into `output`.
    ///
    /// # Errors
    ///
    /// Returns the error of your implementation, which the call returns unchanged.
    fn gradient(
        &self,
        position: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Borrows or stores call-local execution parameters.
#[derive(Debug, Clone)]
#[must_use]
struct VerletRun<T: NumericScalar> {
    step: T::Real,
    steps: usize,
    dimension: usize,
    options: MechanicalOptions,
    scalar: PhantomData<fn() -> T>,
}

/// Describes the workspace storage of a [`StormerVerlet`] call; returned by
/// [`StormerVerlet::workspace_requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StormerVerletRequirements {
    dimension: usize,
}

impl StormerVerletRequirements {
    /// Returns the number of positions.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Holds the step buffers of [`StormerVerlet`] calls and the record of the last
/// attempt.
///
/// Create one with [`StormerVerlet::workspace`], or with [`Self::new`] and
/// [`Self::try_reserve`]. A workspace works with every problem of its dimension.
#[derive(Debug)]
pub struct StormerVerletWorkspace<T: NumericScalar> {
    storage: Storage<T>,
    pub(super) shell: Shell<MechanicalReport<T::Real>>,
}

#[derive(Debug)]
struct Storage<T> {
    q: Vec<T>,
    p: Vec<T>,
    derivative: Vec<T>,
    velocity: Vec<T>,
}

impl<T: NumericScalar> Storage<T> {
    fn fits(&self, dimension: usize) -> bool {
        [&self.q, &self.p, &self.derivative, &self.velocity]
            .iter()
            .all(|buffer| buffer.len() == dimension)
    }
}

impl<T: NumericScalar> Default for StormerVerletWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NumericScalar> StormerVerletWorkspace<T> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            storage: Storage {
                q: Vec::new(),
                p: Vec::new(),
                derivative: Vec::new(),
                velocity: Vec::new(),
            },
            shell: Shell::new(),
        }
    }

    /// Returns whether a call with `requirements` can run without growing this
    /// workspace.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the other workspaces.
    pub fn check(
        &self,
        requirements: &StormerVerletRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.storage.fits(requirements.dimension) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows this workspace to meet `requirements`.
    ///
    /// After a failed reservation the workspace may keep part of the new storage.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` or `SizeOverflow` if a buffer cannot be allocated.
    pub fn try_reserve(
        &mut self,
        requirements: &StormerVerletRequirements,
    ) -> Result<(), SciOxideError> {
        let n = requirements.dimension;
        if self.storage.fits(n) {
            return Ok(());
        }
        let Storage {
            q,
            p,
            derivative,
            velocity,
        } = &mut self.storage;
        for (buffer, quantity) in [
            (q, scioxide_core::quantity!("position_scratch")),
            (p, scioxide_core::quantity!("momentum_scratch")),
            (derivative, scioxide_core::quantity!("derivative_scratch")),
            (velocity, scioxide_core::quantity!("velocity_scratch")),
        ] {
            if buffer.len() != n {
                *buffer = scioxide_core::backend::try_filled(n, T::ZERO, quantity)?;
            }
        }
        Ok(())
    }

    /// Returns the record of the last call: its outcome, counts and the report at the
    /// last accepted step.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MechanicalReport<T::Real>>> {
        self.shell.last_attempt()
    }
}

impl<T: NumericScalar> VerletRun<T> {
    /// Creates a plan for `dimension` positions with the step and step count of
    /// `options`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if the step is NaN or infinite.
    /// - `InvalidInput` if the step is zero or half of it underflows to zero, also when
    ///   the step count is zero.
    fn new(
        problem: &(impl SeparableHamiltonian<T> + ?Sized),
        step: T::Real,
        steps: usize,
        options: MechanicalOptions,
    ) -> Result<Self, SciOxideError> {
        let dimension = problem.dimension();
        super::validate_step(step)?;
        Ok(Self {
            step,
            steps,
            dimension,
            options,
            scalar: PhantomData,
        })
    }

    /// Returns the workspace storage this plan needs.
    ///
    /// # Errors
    ///
    /// None; the `Result` matches the other plans.
    fn workspace_requirements(&self) -> Result<StormerVerletRequirements, SciOxideError> {
        Ok(StormerVerletRequirements {
            dimension: self.dimension,
        })
    }

    fn preflight<H: SeparableHamiltonian<T> + ?Sized>(
        &self,
        problem: &H,
        state: &PhaseSpace<T>,
        workspace: &mut StormerVerletWorkspace<T>,
    ) -> Result<(), SciOxideError> {
        validate_state(self.dimension, state.position(), state.momentum())?;
        if problem.dimension() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(problem.dimension()),
            ));
        }
        let requirements = self.workspace_requirements()?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            workspace.try_reserve(&requirements)?;
        }
        Ok(())
    }

    /// Takes the configured number of steps from `state` and passes each accepted state
    /// to `sink`.
    ///
    /// The call continues from the time and values in `state`, and runs in `scope`, an
    /// [`ExecutionContext`] or the `&mut SolveContext` of an enclosing solve (see
    /// [Solve accounting](scioxide_core#solve-accounting)). A workspace that is too small
    /// is grown first. A work limit, cancellation or a deadline ends the call with an
    /// unsuccessful report. On return, also after an error, `state` holds the last
    /// accepted step. With a large enough workspace and a sink and callbacks that do
    /// not allocate, the call does not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `state` or `problem` does not have the plan's dimension,
    ///   and `NonFinite` (origin `Input`) if `state` has a NaN or infinite entry. No
    ///   callback runs and `state` is unchanged.
    /// - `NonFinite` (origin `Callback`) if a gradient has a NaN, infinite or unwritten
    ///   component, and `NumericalInstability` (`Overflow`) if a position or momentum
    ///   is outside the finite scalar range.
    /// - Errors of `problem` and `sink`, unchanged, and `AllocationFailure` if the
    ///   workspace cannot grow.
    fn advance<'scope, 'capability, H, K>(
        &self,
        problem: &H,
        state: &mut PhaseSpace<T>,
        sink: &mut K,
        workspace: &mut StormerVerletWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<MechanicalReport<T::Real>, SciOxideError>
    where
        H: SeparableHamiltonian<T> + ?Sized,
        K: MechanicalOutputSink<T> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, state, workspace) {
            return workspace.shell.reject(error);
        }
        let StormerVerletWorkspace { storage, shell } = workspace;
        let initial_time = state.time();
        let (position, momentum, time) = state.parts_mut();
        attempt(
            shell,
            scope,
            identity(
                scioxide_core::algorithm_id!("ode-mechanical", "stormer-verlet", 1),
                self.dimension,
            ),
            self.options.budgets(),
            initial_time,
            |report, cx| self.steps(problem, position, momentum, time, storage, sink, report, cx),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn steps<H, K>(
        &self,
        problem: &H,
        position: &mut [T],
        momentum: &mut [T],
        time: &mut T::Real,
        storage: &mut Storage<T>,
        sink: &mut K,
        report: &mut MechanicalReport<T::Real>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault>
    where
        H: SeparableHamiltonian<T> + ?Sized,
        K: MechanicalOutputSink<T> + ?Sized,
    {
        let Storage {
            q,
            p,
            derivative,
            velocity,
        } = storage;
        let h = self.step;
        let half = self.step / T::Real::TWO;
        for step in 0..self.steps {
            proposal(cx)?;
            let Some(next_time) = super::next_time(*time, self.step) else {
                report.termination = Some(super::MechanicalTermination::StepSizeLimit);
                return Ok(());
            };
            if step == 0 {
                admit(cx, EvaluationRole::Gradient)?;
                poison(derivative);
                problem.gradient(position, derivative, cx)?;
                finite(derivative, NonFiniteOrigin::Callback)?;
            }
            p.iter_mut()
                .zip(momentum.iter().zip(derivative.iter()))
                .try_for_each(|(out, (&old, &g))| {
                    *out = crate::safety::state_increment(old, -half, g)?;
                    Ok::<_, SciOxideError>(())
                })?;
            finite(p, NonFiniteOrigin::Intermediate)?;
            admit(cx, EvaluationRole::RightHandSide)?;
            poison(velocity);
            problem.velocity(p, velocity, cx)?;
            finite(velocity, NonFiniteOrigin::Callback)?;
            q.iter_mut()
                .zip(position.iter().zip(velocity.iter()))
                .try_for_each(|(out, (&old, &v))| {
                    *out = crate::safety::state_increment(old, h, v)?;
                    Ok::<_, SciOxideError>(())
                })?;
            finite(q, NonFiniteOrigin::Intermediate)?;
            admit(cx, EvaluationRole::Gradient)?;
            poison(derivative);
            problem.gradient(q, derivative, cx)?;
            finite(derivative, NonFiniteOrigin::Callback)?;
            p.iter_mut()
                .zip(derivative.iter())
                .try_for_each(|(out, &g)| {
                    *out = crate::safety::state_increment(*out, -half, g)?;
                    Ok::<_, SciOxideError>(())
                })?;
            finite(p, NonFiniteOrigin::Intermediate)?;
            accept(cx, PartialResultTrust::Admissible)?;
            *time = next_time;
            report.time = next_time;
            report.trust = PartialResultTrust::Admissible;
            position.copy_from_slice(q);
            momentum.copy_from_slice(p);
            report.steps += 1;
            sink.emit(report.steps, position, momentum)?;
        }
        Ok(())
    }
}

/// Integrates a [`SeparableHamiltonian`] with the Störmer–Verlet method at a fixed step.
///
/// One step of size `h` is the kick–drift–kick sequence
///
/// ```text
/// p½ = pₙ − (h/2) ∇V(qₙ),
/// qₙ₊₁ = qₙ + h ∇T(p½),
/// pₙ₊₁ = p½ − (h/2) ∇V(qₙ₊₁).
/// ```
///
/// The method is explicit, of order 2, symplectic and time-reversible. The energy
/// error therefore stays bounded, of size `O(h²)`, over long times instead of
/// drifting. Each step evaluates `∇T` once and `∇V` once; the first step of each call
/// evaluates `∇V(q₀)` as well.
///
/// The signed step and number of steps are call arguments; [`MechanicalOptions`]
/// sets work limits. Kick and drift updates retain intermediate product range
/// until the final position or momentum is rounded.
///
/// # References
///
/// - E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical Integration*, 2nd ed.,
///   Springer, 2006, chapters I, VI and IX.
/// - B. Leimkuhler and S. Reich, *Simulating Hamiltonian Dynamics*, Cambridge
///   University Press, 2004.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StormerVerlet;
impl StormerVerlet {
    /// Computes reusable storage requirements from the problem dimensions.
    /// # Errors
    /// Returns operator scratch-query errors and storage size overflow.
    pub fn workspace_requirements<T: NumericScalar, H: SeparableHamiltonian<T> + ?Sized>(
        &self,
        problem: &H,
    ) -> Result<StormerVerletRequirements, SciOxideError> {
        Ok(StormerVerletRequirements {
            dimension: problem.dimension(),
        })
    }
    /// Allocates reusable storage for the problem.
    /// # Errors
    /// Returns requirement, allocation and factorization setup errors.
    pub fn workspace<T: NumericScalar, H: SeparableHamiltonian<T> + ?Sized>(
        &self,
        problem: &H,
    ) -> Result<StormerVerletWorkspace<T>, SciOxideError> {
        let mut ws = StormerVerletWorkspace::new();
        ws.try_reserve(&self.workspace_requirements(problem)?)?;
        Ok(ws)
    }
    /// Takes `steps` signed steps of size `step`, retaining each accepted state.
    /// Prepared execution allocates nothing when callbacks and sinks do not allocate.
    /// # Errors
    /// Rejects non-finite, zero or unrepresentable half steps, invalid shapes and
    /// options before changing state; returns callback and numerical errors unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn advance<
        'scope,
        'capability,
        T: NumericScalar,
        H: SeparableHamiltonian<T> + ?Sized,
        K: MechanicalOutputSink<T> + ?Sized,
    >(
        &self,
        problem: &H,
        step: T::Real,
        steps: usize,
        options: &MechanicalOptions,
        state: &mut PhaseSpace<T>,
        sink: &mut K,
        workspace: &mut StormerVerletWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<MechanicalReport<T::Real>, SciOxideError> {
        let run = match VerletRun::new(problem, step, steps, *options) {
            Ok(run) => run,
            Err(error) => return workspace.shell.reject(error),
        };
        run.advance(problem, state, sink, workspace, scope)
    }
    /// Allocates storage and returns the final state and termination report.
    /// # Errors
    /// Returns the errors of [`Self::advance`] with accepted progress.
    #[allow(
        clippy::result_large_err,
        reason = "owned evolution retains accepted progress"
    )]
    pub fn solve<T: NumericScalar, H: SeparableHamiltonian<T> + ?Sized>(
        &self,
        problem: &H,
        step: T::Real,
        steps: usize,
        options: &MechanicalOptions,
        initial: PhaseSpace<T>,
    ) -> Result<MechanicalSolution<T>, PartialFailure<MechanicalReport<T::Real>>> {
        let mut ws = self.workspace(problem).map_err(PartialFailure::new)?;
        let mut state = initial;
        match self.advance(
            problem,
            step,
            steps,
            options,
            &mut state,
            &mut (),
            &mut ws,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(MechanicalSolution { state, report }),
            Err(error) => Err(partial(error, ws.shell)),
        }
    }
}
