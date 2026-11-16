//! Crate-private one-step protocol shared by every built-in method.
//!
//! Built-in method values implement [`Stepper`]; the public sealed [`OdeMethod`] marker
//! lets plans name them without exposing stepping, workspaces or metering. Every
//! callback receives the step's solve context explicitly: call sites count their own
//! provider work through [`admit_provider`] (unless the request is self-metered), which
//! marks budget denials, and record provider failures in the request's [`StepTrace`].

use core::cell::Cell;

use scioxide_core::{
    AlgorithmId, Budget, ErrorTolerance, EvaluationRole, NumericScalar, RealScalar, SciOxideError,
    SolveContext,
};
use scioxide_linalg::Vector;

use crate::result::StepResult;
use crate::traits::OdeSystem;

/// Default adaptive controller selected by a method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DefaultController {
    Integral,
    ProportionalIntegral,
    Radau,
}

/// Static metadata of one configured method.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MethodInfo {
    pub(crate) name: &'static str,
    pub(crate) algorithm: AlgorithmId,
    pub(crate) order: usize,
    pub(crate) initial_step_order: usize,
    pub(crate) controller: DefaultController,
    pub(crate) dense_output: bool,
    pub(crate) events: bool,
    pub(crate) adaptive: bool,
    pub(crate) mass_matrices: bool,
    /// Accepts injected stage linear or nonlinear solvers.
    pub(crate) injectable: bool,
}

/// Per-attempt classification flags written by call sites and read by the driver.
#[derive(Debug, Default)]
pub(crate) struct StepTrace {
    /// An explicit provider returned an error (never retried with a smaller step).
    pub(crate) provider_failed: Cell<bool>,
    /// The budget denial of a provider evaluation in this attempt.
    pub(crate) denial: crate::termination::Tunnel,
    /// The most recent RHS evaluation was at a trial (not the accepted base) point.
    pub(crate) trial_rhs: Cell<bool>,
    /// The system rejected a trial state through `OdeSystem::admits`; the attempt is
    /// retried with a smaller step.
    pub(crate) inadmissible: Cell<bool>,
}

impl StepTrace {
    pub(crate) fn reset(&self) {
        self.provider_failed.set(false);
        self.denial.clear();
        self.trial_rhs.set(false);
        self.inadmissible.set(false);
    }

    /// Mark denied work and return the core error of `fault`, which
    /// classifies the same stop where no driver reads this trace.
    pub(crate) fn fault(&self, fault: crate::termination::Fault) -> SciOxideError {
        if let crate::termination::Fault::Denied(report) = fault {
            self.denial.mark(report);
        }
        fault.into()
    }

    /// Record an error of an explicit provider that is not counted (products,
    /// preconditioners, step limits) and return the result unchanged.
    ///
    /// # Errors
    /// Propagates `result`.
    #[inline]
    pub(crate) fn provider<R>(&self, result: Result<R, SciOxideError>) -> Result<R, SciOxideError> {
        result.inspect_err(|_| self.provider_failed.set(true))
    }
}

/// Artifacts and policies requested from one step attempt.
#[derive(Clone, Copy)]
pub(crate) struct StepRequest<'a, S: RealScalar, T: NumericScalar<Real = S>> {
    /// Accepted base index within this workspace epoch. Unkeyed callers cannot reuse RHS.
    pub(crate) accepted_step: Option<usize>,
    pub(crate) dense_output: bool,
    pub(crate) error_tolerance: Option<&'a ErrorTolerance<S>>,
    pub(crate) inner_limit: Budget,
    pub(crate) trace: &'a StepTrace,
    /// Accepted point of the attempt, used to classify trial RHS evaluations.
    pub(crate) base: Option<(S, &'a Vector<T>)>,
    /// Already counted derivative at the accepted base, from the combined
    /// step-limit callback or the automatic initial-step heuristic.
    pub(crate) initial_rhs: Option<&'a [T]>,
    /// The system counts its own RHS work (DDE runtime, adjoint replay systems).
    pub(crate) self_metered: bool,
    /// Injected stage solver of the problem, if any.
    pub(crate) provider: crate::stiff::stage::Provider<'a, S, T>,
}

impl<'a, S: RealScalar, T: NumericScalar<Real = S>> StepRequest<'a, S, T> {
    /// Basic step with core default error control and at most 20 inner iterations.
    pub(crate) const fn new(trace: &'a StepTrace) -> Self {
        Self {
            accepted_step: None,
            dense_output: false,
            error_tolerance: None,
            inner_limit: Budget::ONE.saturating_add(19),
            trace,
            base: None,
            initial_rhs: None,
            self_metered: false,
            provider: crate::stiff::stage::Provider::Native,
        }
    }
    /// Reuse the driver's checked step geometry; direct calls check it here.
    pub(crate) fn endpoint(self, time: S, step: S) -> Result<S, SciOxideError> {
        if self.base.is_some() {
            Ok(time + step)
        } else {
            crate::safety::checked_step_endpoint(time, step)
        }
    }
    pub(crate) const fn with_dense_output(mut self) -> Self {
        self.dense_output = true;
        self
    }
    pub(crate) const fn without_dense_output(mut self) -> Self {
        self.dense_output = false;
        self
    }
    pub(crate) const fn with_error_tolerance(mut self, tolerance: &'a ErrorTolerance<S>) -> Self {
        self.error_tolerance = Some(tolerance);
        self
    }
    pub(crate) const fn requires_dense_output(self) -> bool {
        self.dense_output
    }
}

/// One-step method protocol. Methods are immutable configuration; all mutable numerical
/// state lives in `Workspace`.
pub(crate) trait Stepper<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    /// Static adaptivity of methods whose configuration cannot change it.
    const ADAPTIVE: Option<bool> = None;

    /// Read the accepted input from `StepRequest::base` when the driver supplies it.
    const BORROWS_BASE_STATE: bool = false;

    /// Establish finite, correctly shaped output and matching requested artifacts
    /// in the numerical kernel.
    const CHECKED_RESULT: bool = false;

    /// Method storage. `Default` is empty and does not allocate.
    type Workspace: Default;

    /// Borrow prepared linearization storage for automatic-method diagnostics.
    fn stage_kernel<'w>(
        &self,
        _workspace: &'w Self::Workspace,
    ) -> Option<&'w crate::stiff::kernel::StageKernel<S, T>> {
        None
    }

    fn info(&self) -> MethodInfo;

    /// Whether plans obtain the base derivative through the combined step-limit
    /// callback, which this method consumes as its first stage.
    fn accepts_initial_rhs(&self) -> bool {
        false
    }

    /// Whether `workspace` already serves `dimension` without allocation; `provided`
    /// selects storage for an injected stage solver instead of native factors.
    fn fits(&self, dimension: usize, provided: bool, workspace: &Self::Workspace) -> bool;

    /// Grow or re-prepare `workspace`; afterwards [`Self::fits`] holds.
    fn reserve(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Begin an independent epoch: discard history, keep storage. Requires `fits`.
    fn begin(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError>;

    /// Advance from the borrowed accepted base, or `(t, y)` when no base is borrowed.
    /// On success `y` holds the trial state.
    /// The caller validates problem features, tolerance shape and finite initial
    /// state, reserves matching storage, and begins the epoch before stepping.
    #[allow(clippy::too_many_arguments)]
    fn step<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        system: &System,
        t: S,
        y: &mut Vector<T>,
        h: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError>;

    /// Build the dense output that the last successful [`Self::step`] deferred
    /// (`StepResult::dense_deferred`), after the driver accepts the proposal and
    /// before [`Self::accept`]. `t`, `h` and `request` repeat the step's arguments.
    #[allow(clippy::too_many_arguments)]
    fn complete_dense<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        _system: &System,
        _t: S,
        _h: S,
        _request: StepRequest<'_, S, T>,
        _workspace: &'w mut Self::Workspace,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
        Err(crate::safety::invalid(
            scioxide_core::argument!("method"),
            "the method deferred no dense output",
        ))
    }

    /// Declared linear stability of the propagated solution.
    fn stability(&self) -> OdeLinearStability<S> {
        OdeLinearStability::Undeclared
    }

    /// Commit the last proposal at its full endpoint.
    fn accept(&self, _workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }

    /// Discard the last proposal (rejection or truncated terminal state).
    fn discard(&self, _workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        Ok(())
    }
}

/// Describes the linear stability region of a method on the test equation `y′ = λ y`
/// along the real and imaginary axes.
///
/// `Axes { real, imaginary }` states that the method is stable for every `z = h λ` on
/// the real segment `[−real, 0]` and on the imaginary segment
/// `[−i · imaginary, i · imaginary]`, where `h` is the step size. `None` means the whole
/// negative real half-axis or the whole imaginary axis, as for A-stable methods. A
/// zero bound means that no nonzero step is stable on that axis. Finite bounds are
/// decimal values rounded down from the boundary of the method's stability region.
/// Combine them with a bound on the spectrum of `∂f/∂y`, for example of a
/// method-of-lines discretization, to choose a stable explicit step. Methods without
/// a declared region return `Undeclared`.
///
/// # Examples
///
/// ```
/// use scioxide_ode::{DormandPrince5, OdeLinearStability, OdeMethod};
///
/// // The Dormand–Prince 5(4) region meets the negative real axis near −3.3065.
/// let OdeLinearStability::Axes { real, .. } =
///     OdeMethod::<f64>::linear_stability(&DormandPrince5::<f64>::new())
/// else {
///     unreachable!()
/// };
/// assert!((real.unwrap() - 3.3065).abs() < 1e-12);
/// ```
///
/// # References
///
/// E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 2nd ed.,
/// Springer, 1996, §IV.2–IV.3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdeLinearStability<S> {
    /// The method declares no stability region.
    Undeclared,
    /// Stable segments of the real and imaginary axes; `None` means unbounded.
    Axes {
        /// Length of the stable segment `[−real, 0]` of the negative real axis.
        real: Option<S>,
        /// Half-length of the stable segment of the imaginary axis.
        imaginary: Option<S>,
    },
}

impl<S: RealScalar> OdeLinearStability<S> {
    /// Returns `Axes` with both bounds `None`: stable on the whole negative real
    /// half-axis and the whole imaginary axis, as for A-stable methods.
    #[must_use]
    pub const fn unconditional() -> Self {
        Self::Axes {
            real: None,
            imaginary: None,
        }
    }

    /// Returns `Axes` with the finite bounds `real` and `imaginary`, converted to `S`.
    #[must_use]
    pub fn bounded(real: f64, imaginary: f64) -> Self {
        Self::Axes {
            real: Some(S::from_f64_ieee(real)),
            imaginary: Some(S::from_f64_ieee(imaginary)),
        }
    }
}

pub use crate::ivp::plan::OdeMethod;

/// Count one provider evaluation in `cx` through [`crate::termination::admit`], marking
/// a denial in `trace`.
///
/// # Errors
/// Returns execution stops and the core budget error of a denial.
#[inline]
pub(crate) fn admit_provider(
    cx: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
    trace: &StepTrace,
) -> Result<(), SciOxideError> {
    crate::termination::admit(cx, role).map_err(|fault| trace.fault(fault))
}

/// Count one provider evaluation without polling execution capabilities, as the
/// per-evaluation RHS and Jacobian boundaries always have.
#[inline(always)]
fn record(
    cx: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
    trace: &StepTrace,
) -> Result<(), SciOxideError> {
    crate::termination::record(cx, role).map_err(|fault| trace.fault(fault))
}

/// Evaluate the RHS through the request's metering policy.
///
/// Counts `RightHandSide` unless the request is self-metered, records provider failure and
/// whether the point is a trial point. Output finiteness is checked by callers. Like every
/// per-evaluation boundary, it does not poll execution capabilities.
///
/// # Errors
/// Propagates budget, execution and provider errors.
#[inline(always)]
pub(crate) fn rhs<S, T, System>(
    system: &System,
    time: S,
    state: &[T],
    output: &mut [T],
    at_base: bool,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let trial = if at_base {
        false
    } else {
        request.base.is_none_or(|(base_time, base_state)| {
            base_time != time || base_state.as_slice() != state
        })
    };
    if request.base.is_some() && trial && !request.trace.provider(system.admits(time, state, cx))? {
        request.trace.inadmissible.set(true);
        return Err(inadmissible());
    }
    if !request.self_metered {
        record(cx, EvaluationRole::RightHandSide, request.trace)?;
    }
    let result = request.trace.provider(system.rhs(time, state, output, cx));
    request.trace.trial_rhs.set(trial);
    result
}

/// The error that carries an admissibility rejection out of a step; the driver
/// recognizes it through [`StepTrace::inadmissible`] and retries with half the step.
pub(crate) fn inadmissible() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "trial state is inadmissible; a smaller step is required",
    )
}

/// Evaluate the RHS inside nested work (difference products of matrix-free stages).
///
/// `cx` is the child epoch (or the provider scope) running the product. The evaluation
/// is counted there once, unless the request is self-metered, and a denial is marked
/// in the trace like a step-level one. Trial classification is left to the step-level
/// evaluations.
///
/// # Errors
/// Propagates budget, execution and provider errors.
pub(crate) fn nested_rhs<S, T, System>(
    system: &System,
    time: S,
    state: &[T],
    output: &mut [T],
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    if !request.self_metered {
        record(cx, EvaluationRole::RightHandSide, request.trace)?;
    }
    request.trace.provider(system.rhs(time, state, output, cx))
}

/// Run `provider` after counting it under `role`, marking provider failure.
///
/// # Errors
/// Propagates budget, execution and provider errors.
#[inline]
pub(crate) fn invoke<S, T, R>(
    role: EvaluationRole,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
    provider: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    record(cx, role, request.trace)?;
    request.trace.provider(provider(cx))
}

/// Crate-internal test fixtures for direct stepping.
#[cfg(test)]
pub(crate) mod testing {
    use scioxide_core::{SciOxideError, SolveContext};

    /// Run `body` in a fresh sequential root epoch with unbounded budgets.
    pub(crate) fn with_context<R>(
        body: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
    ) -> Result<R, SciOxideError> {
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("ode", "test", 1),
            0,
            0,
        );
        scioxide_core::SolveScope::from(scioxide_core::ExecutionContext::sequential()).run(
            &mut scioxide_core::SolveState::new(),
            identity,
            scioxide_core::SolveBudgets::unbounded(),
            body,
        )
    }
}
