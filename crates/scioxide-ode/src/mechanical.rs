//! Fixed-step symplectic integrators for Hamiltonian mechanics: Störmer–Verlet for
//! separable Hamiltonians and RATTLE for holonomic constraints.
//!
//! Each `advance` takes the configured number of steps from a [`PhaseSpace`], so
//! successive calls continue one trajectory. The `solve` forms create their workspace
//! internally.

use crate::safety::checkpoint;
use crate::termination::{Work, admit};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{ExecutionContext, PartialFailure};

use scioxide_core::{
    Budget, BudgetKind, EvaluationCounts, ExecutionStopReason, InvalidInputReason, NonFiniteOrigin,
    NumericScalar, PartialResultTrust, RealScalar, SciOxideError, SolveBudgets, SolveContext,
    SolveCounts, SolveIdentity, SolveScope,
};

use crate::shell::Shell;
use crate::termination::{Fault, Stop, Terminal};

mod rattle;
mod verlet;

pub use rattle::{
    HolonomicConstraints, Rattle, RattleOptions, RattleProblem, RattleRequirements, RattleWorkspace,
};
pub use verlet::{
    SeparableHamiltonian, StormerVerlet, StormerVerletRequirements, StormerVerletWorkspace,
};

/// Stores the positions `q`, conjugate momenta `p` and time of a mechanical integration.
///
/// After [`StormerVerlet::advance`] or [`Rattle::advance`] the state holds the last
/// accepted step, also when the call ends with an error, so the next call continues
/// from it. If the call rejects its arguments before the first step, the state is
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PhaseSpace<T: NumericScalar> {
    position: Vec<T>,
    momentum: Vec<T>,
    time: T::Real,
}

impl<T: NumericScalar> PhaseSpace<T> {
    /// Creates a state at time zero from positions and momenta.
    ///
    /// `momentum` holds the conjugate momenta `p`, not the velocities `q′`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the two vectors differ in length.
    /// - `NonFinite` (origin `Input`) if an entry is NaN or infinite.
    pub fn new(position: Vec<T>, momentum: Vec<T>) -> Result<Self, SciOxideError> {
        validate_state(position.len(), &position, &momentum)?;
        Ok(Self {
            position,
            momentum,
            time: T::Real::ZERO,
        })
    }

    /// Returns the time reached, starting at zero and advanced by every accepted step.
    pub const fn time(&self) -> T::Real {
        self.time
    }

    /// Returns the positions `q`.
    pub fn position(&self) -> &[T] {
        &self.position
    }

    /// Returns the momenta `p`.
    pub fn momentum(&self) -> &[T] {
        &self.momentum
    }

    /// Returns the positions and momenta.
    pub fn into_parts(self) -> (Vec<T>, Vec<T>) {
        (self.position, self.momentum)
    }

    fn parts_mut(&mut self) -> (&mut [T], &mut [T], &mut T::Real) {
        (&mut self.position, &mut self.momentum, &mut self.time)
    }
}

/// Sets the work limits of
/// [`StormerVerlet`] and [`Rattle`].
///
/// The step is constant and may be negative, which integrates backward in time. Each
/// call takes the requested number of steps; there is no step size control.
/// A step whose endpoint is nonfinite or rounds back to the accepted time stops
/// with [`MechanicalTermination::StepSizeLimit`] before evaluating its stages.
/// The setters do not check their arguments; [`StormerVerlet::advance`] and [`Rattle::advance`]
/// validate the options.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use]
pub struct MechanicalOptions {
    max_steps: Option<Budget>,
    max_iterations: Option<Budget>,
    max_evaluations: Option<Budget>,
}

impl MechanicalOptions {
    /// Creates options without work limits.
    pub const fn new() -> Self {
        Self {
            max_steps: None,
            max_iterations: None,
            max_evaluations: None,
        }
    }

    /// Limits the number of attempted steps per call.
    ///
    /// A limit below the step count ends the call with
    /// [`MechanicalTermination::BudgetLimit`] with [`BudgetKind::Proposal`].
    pub fn with_max_steps(mut self, steps: Budget) -> Self {
        self.max_steps = Some(steps);
        self
    }

    /// Limits the iterations per call ([`BudgetKind::Iteration`]).
    ///
    /// Each step counts one iteration, and each Newton iteration of a RATTLE position projection
    /// counts one more.
    pub fn with_max_iterations(mut self, iterations: Budget) -> Self {
        self.max_iterations = Some(iterations);
        self
    }

    /// Limits the evaluations per call ([`BudgetKind::Evaluation`]).
    ///
    /// Gradients, velocities, constraint values and Jacobians, applications of `M⁻¹`,
    /// and the factorizations and solves of a RATTLE projection each count one. An
    /// evaluation limit of an enclosing scope also applies.
    pub fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    /// Returns the limit on attempted steps per call.
    pub const fn max_steps(&self) -> Option<Budget> {
        self.max_steps
    }
    /// Returns the iteration limit per call.
    pub const fn max_iterations(&self) -> Option<Budget> {
        self.max_iterations
    }
    /// Returns the evaluation limit per call.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    /// Returns the budgets of one call.
    fn budgets(&self) -> SolveBudgets {
        [
            (BudgetKind::Proposal, self.max_steps),
            (BudgetKind::Iteration, self.max_iterations),
            (BudgetKind::Evaluation, self.max_evaluations),
        ]
        .into_iter()
        .fold(
            SolveBudgets::unbounded(),
            |budgets, (kind, limit)| match limit {
                Some(limit) => budgets.with_limit(kind, limit),
                None => budgets,
            },
        )
    }
}
/// Rejects zero and non-finite steps and steps whose half underflows to zero, also
/// when no step is requested.
fn validate_step<S: RealScalar>(step: S) -> Result<(), SciOxideError> {
    if !step.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "step must be finite",
        ));
    }
    if step == S::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("step"),
            InvalidInputReason::OutOfRange,
            "step must be nonzero",
        ));
    }
    if step / S::TWO == S::ZERO {
        return Err(crate::safety::invalid(
            scioxide_core::argument!("step"),
            "half step is not representable",
        ));
    }
    Ok(())
}

/// A finite endpoint distinct from the accepted time, or exhausted time resolution.
fn next_time<S: RealScalar>(time: S, step: S) -> Option<S> {
    let next = time + step;
    (next.is_finite() && next != time).then_some(next)
}

/// Reports why a [`StormerVerlet`] or [`Rattle`] call stopped.
///
/// The variants shared with the other time integrators are described in
/// [Termination](crate#termination). The fixed-step mechanical integrators have no
/// events, so they never return `TerminalEvent`, and they never change the step, so
/// they return `StepSizeLimit` only when the time can no longer advance by the step.
/// After an unsuccessful termination the state holds the last accepted step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MechanicalTermination {
    /// Every configured step was accepted.
    ReachedEnd,
    /// Not returned by the mechanical integrators.
    TerminalEvent,
    /// A limit of [`MechanicalOptions`] or of an enclosing scope stopped the call.
    BudgetLimit(BudgetKind),
    /// The endpoint of the next step was nonfinite or rounded back to the accepted
    /// time, so the call stopped before evaluating that step.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the call.
    ExecutionStopped(ExecutionStopReason),
    /// A RATTLE step did not meet the position or momentum constraint tolerance of
    /// [`RattleOptions`]; the step was rejected.
    ProjectionLimit,
}

crate::termination::shared_termination!(MechanicalTermination);

/// Reports the termination, steps, time, residuals and work of one call of a mechanical
/// integrator.
///
/// After an error, the workspace's `last_attempt` and the
/// [`PartialFailure`](scioxide_core::PartialFailure) of `solve` hold a report of the last
/// accepted step without a termination.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MechanicalReport<S: RealScalar> {
    termination: Option<MechanicalTermination>,
    steps: usize,
    time: S,
    trust: PartialResultTrust,
    counts: SolveCounts,
    position_residual: Option<S>,
    momentum_residual: Option<S>,
}

impl<S: RealScalar> MechanicalReport<S> {
    /// Returns the number of rejected steps, which is at most one: a RATTLE step that
    /// ends the call with [`MechanicalTermination::ProjectionLimit`].
    pub const fn rejected_steps(&self) -> usize {
        self.counts.own().rejected()
    }

    const fn start(time: S) -> Self {
        Self {
            termination: None,
            steps: 0,
            time,
            trust: PartialResultTrust::DiagnosticOnly,
            counts: SolveCounts::new(),
            position_residual: None,
            momentum_residual: None,
        }
    }

    /// Returns why the call stopped.
    ///
    /// Every report returned by `advance` or `solve` has a termination. It is `None`
    /// only in the report of a call that ended with an error.
    pub const fn termination(&self) -> Option<MechanicalTermination> {
        self.termination
    }
    /// Returns whether every configured step was accepted.
    pub fn is_success(&self) -> bool {
        self.termination
            .is_some_and(MechanicalTermination::is_success)
    }
    /// Returns the time of the last accepted state, counted from the start of the first
    /// call on this state.
    pub const fn final_time(&self) -> S {
        self.time
    }
    /// Returns the trust level of the last state accepted in this call:
    /// `ResidualChecked` for RATTLE and `Admissible` for Störmer–Verlet.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the number of steps accepted in this call.
    pub const fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Returns the evaluations and iterations of this call, excluding nested solves.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the counts of this call together with its nested solves.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns `‖g(q)‖∞` at the last accepted RATTLE state; `None` for Störmer–Verlet.
    pub const fn position_residual(&self) -> Option<S> {
        self.position_residual
    }
    /// Returns `‖G(q) M⁻¹ p‖∞` at the last accepted RATTLE state, where `G = ∂g/∂q`;
    /// `None` for Störmer–Verlet.
    pub const fn momentum_residual(&self) -> Option<S> {
        self.momentum_residual
    }
}

/// Holds the final state and report of [`StormerVerlet::solve`] and [`Rattle::solve`].
#[derive(Debug, Clone)]
#[must_use]
pub struct MechanicalSolution<T: NumericScalar> {
    state: PhaseSpace<T>,
    report: MechanicalReport<T::Real>,
}

impl<T: NumericScalar> MechanicalSolution<T> {
    /// Returns the last accepted state.
    pub const fn state(&self) -> &PhaseSpace<T> {
        &self.state
    }
    /// Returns the report of the call.
    pub const fn report(&self) -> &MechanicalReport<T::Real> {
        &self.report
    }
    /// Returns the final state and the report.
    pub fn into_parts(self) -> (PhaseSpace<T>, MechanicalReport<T::Real>) {
        (self.state, self.report)
    }
}

/// Receives the state after every accepted step of a mechanical integrator.
///
/// `()` ignores the output, and a closure `FnMut(usize, &[T], &[T]) -> Result<(),
/// SciOxideError>` receives it. States passed to the sink are final: a later error does
/// not withdraw them.
pub trait MechanicalOutputSink<T> {
    /// Receives the positions and momenta after accepted step number `step` of the
    /// current call, counted from one.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged. The step it was
    /// called for stays accepted.
    fn emit(&mut self, step: usize, position: &[T], momentum: &[T]) -> Result<(), SciOxideError>;
}

/// Ignores all output.
impl<T> MechanicalOutputSink<T> for () {
    fn emit(&mut self, _: usize, _: &[T], _: &[T]) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<T, F> MechanicalOutputSink<T> for F
where
    F: FnMut(usize, &[T], &[T]) -> Result<(), SciOxideError>,
{
    fn emit(&mut self, step: usize, position: &[T], momentum: &[T]) -> Result<(), SciOxideError> {
        self(step, position, momentum)
    }
}

/// Runs one attempt of `steps` in `scope`, turns budget and execution stops into
/// terminations, and records the attempt with the running report.
fn attempt<S: RealScalar>(
    shell: &mut Shell<MechanicalReport<S>>,
    scope: SolveScope<'_, '_>,
    identity: SolveIdentity,
    budgets: SolveBudgets,
    initial_time: S,
    steps: impl FnOnce(&mut MechanicalReport<S>, &mut SolveContext<'_, '_>) -> Result<(), Fault>,
) -> Result<MechanicalReport<S>, SciOxideError> {
    let mut running = MechanicalReport::start(initial_time);
    let executed = shell.execute(scope, identity, budgets, |cx| {
        let outcome = steps(&mut running, cx);
        running.counts = cx.counts();
        match outcome {
            Ok(()) => {
                running
                    .termination
                    .get_or_insert(MechanicalTermination::ReachedEnd);
                Ok(running)
            }
            Err(fault) => match fault.stop() {
                Some(shared @ (Stop::Budget(_) | Stop::Execution(_))) => {
                    running.termination = Some(MechanicalTermination::from_stop(shared));
                    Ok(running)
                }
                Some(Stop::StepSize) | None => Err(fault.into()),
            },
        }
    });
    shell.conclude(executed, MechanicalReport::is_success, |_| Some(running))
}

fn identity(algorithm: scioxide_core::AlgorithmId, dimension: usize) -> SolveIdentity {
    SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        algorithm,
        dimension as u128,
        0,
    )
}

/// Counts one step: a proposal and its iteration.
fn proposal(cx: &mut SolveContext<'_, '_>) -> Result<(), Fault> {
    admit(cx, Work::Step)
}

/// Accepts a computed step unless execution stopped first.
fn accept(cx: &mut SolveContext<'_, '_>, trust: PartialResultTrust) -> Result<(), SciOxideError> {
    checkpoint(cx)?;
    cx.commit_accepted(trust)
}

fn poison<T: NumericScalar>(out: &mut [T]) {
    out.fill(T::from_components(T::Real::NAN, T::Real::NAN));
}

fn validate_state<T: NumericScalar>(n: usize, q: &[T], p: &[T]) -> Result<(), SciOxideError> {
    for len in [q.len(), p.len()] {
        if len != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(n),
                scioxide_core::Dimensions::one(len),
            ));
        }
    }
    crate::safety::finite(q, NonFiniteOrigin::Input)?;
    crate::safety::finite(p, NonFiniteOrigin::Input)
}

/// Builds the error of a `solve` form with the recorded report.
fn partial<S: RealScalar>(
    error: SciOxideError,
    shell: Shell<MechanicalReport<S>>,
) -> scioxide_core::PartialFailure<MechanicalReport<S>> {
    shell.into_failure(error)
}

/// Takes `steps` Störmer–Verlet steps of signed size `step` for `H(q,p)=T(p)+V(q)`.
///
/// `velocity` writes `∇T(p)` and `gradient` writes `∇V(q)`. Returns the final
/// positions and momenta. The method is symplectic and second order, with linear
/// state storage and a constant number of callback evaluations per step.
///
/// # Errors
/// Returns invalid step, shape and non-finite input errors, callback errors, and
/// an error with progress when integration stops before all steps are accepted.
///
/// # Examples
/// ```
/// let state = scioxide_ode::solve_hamiltonian(
///     |p: &[f64], v: &mut [f64]| { v.copy_from_slice(p); Ok(()) },
///     |q: &[f64], g: &mut [f64]| { g.copy_from_slice(q); Ok(()) },
///     &[1.0], &[0.0], 0.01, 100,
/// ).map_err(scioxide_core::SciOxideError::from)?;
/// assert!((state.position()[0] - 1.0_f64.cos()).abs() < 1e-4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Hairer, Lubich and Wanner, *Geometric Numerical Integration*, I.1.4.
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains accepted progress"
)]
pub fn solve_hamiltonian<T, V, G>(
    velocity: V,
    gradient: G,
    q0: &[T],
    p0: &[T],
    step: T::Real,
    steps: usize,
) -> Result<PhaseSpace<T>, PartialFailure<MechanicalReport<T::Real>>>
where
    T: NumericScalar,
    V: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
    G: Fn(&[T], &mut [T]) -> Result<(), SciOxideError>,
{
    use crate::termination::Terminal;
    let system = crate::HamiltonianFn::new(q0.len(), velocity, gradient);
    let mut state = PhaseSpace::new(
        scioxide_core::backend::try_copy_slice(q0, scioxide_core::quantity!("q0"))
            .map_err(PartialFailure::new)?,
        scioxide_core::backend::try_copy_slice(p0, scioxide_core::quantity!("p0"))
            .map_err(PartialFailure::new)?,
    )
    .map_err(PartialFailure::new)?;
    let mut ws = StormerVerlet
        .workspace(&system)
        .map_err(PartialFailure::new)?;
    match StormerVerlet.advance(
        &system,
        step,
        steps,
        &MechanicalOptions::default(),
        &mut state,
        &mut (),
        &mut ws,
        ExecutionContext::sequential_native(),
    ) {
        Ok(report) => {
            if let Some(stop) = report.termination().and_then(Terminal::stop) {
                return Err(ws
                    .shell
                    .into_failure(crate::termination::stop_error(stop, report.accounting())));
            }
            if !report.is_success() {
                return Err(ws.shell.into_failure(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::NestedSolver,
                    scioxide_core::ConvergenceDiagnostics::new().with_counts(report.accounting()),
                    "mechanical projection failed",
                )));
            }
            Ok(state)
        }
        Err(error) => Err(ws.shell.into_failure(error)),
    }
}
