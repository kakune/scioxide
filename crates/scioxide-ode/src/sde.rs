//! Explicit stochastic differential equation solvers on a grid of Wiener increments
//! that you supply.
//!
//! A [`SdeSystem`] defines the drift and the diffusion; a [`SdeProblem`] pairs it with
//! the increments and time grid of one sample path. The problem holds the step
//! partition. [`SdeMethod`] takes [`SdeOptions`] per call. The solver never generates random
//! numbers and stores no path.

use crate::safety::invalid;
use crate::termination::Terminal;
use crate::{OdeOutputSink, OdeState};

use scioxide_core::backend::checked_mul;
use scioxide_core::checked_difference;
use scioxide_core::{
    Budget, BudgetKind, Capacity, ErrorTolerance, EvaluationCounts, ExecutionContext,
    ExecutionStopReason, NonFiniteOrigin, NumericScalar, PartialFailure, PartialResultTrust,
    ProblemToken, RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveCounts,
    SolveIdentity, SolveScope,
};
use scioxide_linalg::MatrixView;

mod kernel;
mod workspace;

pub use workspace::{SdeRequirements, SdeWorkspace};

/// Names the stochastic integral that defines the equation `dX = a dt + B dW`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdeInterpretation {
    /// Itô integral, evaluated at the left end of each interval.
    Ito,
    /// Stratonovich integral, evaluated symmetrically.
    ///
    /// The drift is taken as given; it is not converted from the Itô form.
    Stratonovich,
}

/// Selects an explicit one-step method for stochastic differential equations.
///
/// With step `h = t₁ − t₀`, Wiener increment `ΔW` over `[t₀, t₁]`, drift `a` and
/// diffusion `B`, the methods are:
///
/// ```text
/// Euler–Maruyama:   X₁ = X₀ + h a(t₀, X₀) + B(t₀, X₀) ΔW
/// stochastic Heun:  X̃ = X₀ + h a(t₀, X₀) + B(t₀, X₀) ΔW
///                   X₁ = X₀ + (h/2) (a(t₀, X₀) + a(t₁, X̃))
///                           + (1/2) (B(t₀, X₀) + B(t₁, X̃)) ΔW
/// ```
///
/// # References
///
/// P. E. Kloeden and E. Platen, *Numerical Solution of Stochastic Differential
/// Equations*, Springer, 1992.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdeMethod {
    /// Euler–Maruyama for Itô equations: strong order 1/2 and weak order 1 under the
    /// usual Lipschitz, linear growth and smoothness conditions on the coefficients.
    ///
    /// It supports fixed steps only; [`SdeMethod::advance`] rejects it with
    /// [`SdeOptions::with_refinement`], because refinement that depends on the path
    /// biases Itô methods.
    EulerMaruyama,
    /// Stochastic Heun for Stratonovich equations, a predictor–corrector that uses the
    /// same increment in both stages: strong order 1 for scalar or commutative noise,
    /// 1/2 in general, and weak order 1.
    StochasticHeun,
}
impl SdeMethod {
    /// Returns the stochastic integral the method solves.
    pub const fn interpretation(self) -> SdeInterpretation {
        match self {
            Self::EulerMaruyama => SdeInterpretation::Ito,
            Self::StochasticHeun => SdeInterpretation::Stratonovich,
        }
    }
    const fn algorithm_id(self) -> scioxide_core::AlgorithmId {
        match self {
            Self::EulerMaruyama => scioxide_core::algorithm_id!("sde", "euler-maruyama", 1),
            Self::StochasticHeun => scioxide_core::algorithm_id!("sde", "stochastic-heun", 1),
        }
    }
}

/// Defines the stochastic differential equation
///
/// ```text
/// dX = a(t, X) dt + B(t, X) dW,
/// ```
///
/// where `X(t) ∈ 𝕂ⁿ` is the real or complex state, `a(t, X) ∈ 𝕂ⁿ` the drift,
/// `B(t, X)` an `n × m` diffusion matrix and `W` an `m`-dimensional real Wiener
/// process with independent components.
///
/// The [`SdeMethod::interpretation`] of the method decides whether the equation is read
/// in the Itô or the Stratonovich sense; supply the coefficients for that sense. The
/// diffusion is given as its product with an increment, so it can be dense, sparse or
/// matrix-free; the product must be linear in the increment.
///
/// Callbacks write every output component and depend only on their arguments. A NaN or
/// infinite output is a `NonFinite` error with origin `Callback`. Each drift call counts
/// as one `RightHandSide` evaluation and each diffusion call as one `LinearOperator`
/// evaluation; charge only your own nested work to `cx` (see
/// [solve accounting](scioxide_core#solve-accounting)).
///
/// # Errors
///
/// An error returned by a callback stops the integration and is returned unchanged.
pub trait SdeSystem<T: NumericScalar> {
    /// Returns the number of state components `n`, which may be zero.
    fn dimension(&self) -> usize;
    /// Returns the number of Wiener components `m`, which may be zero.
    fn noise_dimension(&self) -> usize;
    /// Writes the drift `a(time, state)` to `output`.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged.
    fn drift(
        &self,
        time: T::Real,
        state: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Writes `B(time, state) · increment` to `output`.
    ///
    /// `increment` is the Wiener increment over the whole step, not scaled by the step
    /// size. It is not called when the noise dimension is zero.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged.
    fn diffusion_product(
        &self,
        time: T::Real,
        state: &[T],
        increment: &[T::Real],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

/// Pairs an [`SdeSystem`] with the Wiener increments of one sample path.
///
/// Row `i` of `increments` is `W(tᵢ₊₁) − W(tᵢ)` over the interval `[tᵢ, tᵢ₊₁]` of the
/// problem's time grid, and column `j` is the Wiener component `j`. For a standard Wiener
/// process each entry is normal with mean zero and variance `tᵢ₊₁ − tᵢ`. The view may
/// be strided. The solver checks its shape and that every entry is finite before any
/// callback runs.
#[derive(Clone, Copy)]
pub struct SdeProblem<'a, T: NumericScalar> {
    system: &'a dyn SdeSystem<T>,
    increments: MatrixView<'a, T::Real>,
    times: &'a [T::Real],
    boundaries: Option<&'a [usize]>,
}
impl<'a, T: NumericScalar> SdeProblem<'a, T> {
    /// Creates a problem driven by `increments`, with one row per grid interval and one
    /// column per Wiener component.
    pub const fn new(
        system: &'a dyn SdeSystem<T>,
        times: &'a [T::Real],
        increments: MatrixView<'a, T::Real>,
    ) -> Self {
        Self {
            system,
            times,
            increments,
            boundaries: None,
        }
    }
    /// Sets grid indices delimiting steps; omission makes every interval one step.
    #[must_use]
    pub const fn with_step_boundaries(mut self, boundaries: &'a [usize]) -> Self {
        self.boundaries = Some(boundaries);
        self
    }
    /// Returns the fine time grid.
    pub const fn times(&self) -> &'a [T::Real] {
        self.times
    }
    /// Returns explicit step boundaries, or `None` for every grid interval.
    pub const fn step_boundaries(&self) -> Option<&'a [usize]> {
        self.boundaries
    }
    /// Returns the system.
    pub const fn system(&self) -> &'a dyn SdeSystem<T> {
        self.system
    }
    /// Returns the Wiener increments.
    pub const fn increments(&self) -> MatrixView<'a, T::Real> {
        self.increments
    }
}

/// Configures [`SdeMethod`]: optional step refinement, a workspace size limit and
/// budgets.
///
/// Setters do not fail; [`SdeMethod::advance`] checks the whole configuration.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct SdeOptions<S: RealScalar> {
    refinement: Option<ErrorTolerance<S>>,
    max_workspace_bytes: Option<usize>,
    max_steps: Option<Budget>,
    max_evaluations: Option<Budget>,
}
impl<S: RealScalar> Default for SdeOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> SdeOptions<S> {
    /// Returns options for fixed steps on the problem's partition, with no workspace limit
    /// and no budgets.
    pub const fn new() -> Self {
        Self {
            refinement: None,
            max_workspace_bytes: None,
            max_steps: None,
            max_evaluations: None,
        }
    }
    /// Enables step refinement for [`SdeMethod::StochasticHeun`] with `tolerance`.
    ///
    /// Each interval of the partition is integrated with one step and with two half
    /// steps at the midpoint of its grid nodes. If the normalized difference of the two
    /// results exceeds one, both halves are refined in the same way; otherwise the
    /// two-half-step result is accepted. Every interval then needs at least two grid
    /// intervals: one that cannot be halved ends the integration with
    /// [`SdeTermination::StepSizeLimit`]. The difference is a local heuristic, not a
    /// bound on the strong or weak error.
    ///
    /// The decision for an interval uses that interval's own increments, so the accepted
    /// steps depend on the path. This samples the quadratic variation unevenly and
    /// biases Itô methods, so [`SdeMethod::advance`] rejects refinement with
    /// [`SdeMethod::EulerMaruyama`]. Stochastic Heun still converges to the Stratonovich
    /// solution.
    ///
    /// # References
    ///
    /// J. G. Gaines and T. J. Lyons, "Variable step size control in the numerical
    /// solution of stochastic differential equations", *SIAM J. Appl. Math.* 57(5),
    /// 1455–1484, 1997.
    pub fn with_refinement(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.refinement = Some(tolerance);
        self
    }
    /// Limits the bytes a workspace used by the method may hold.
    ///
    /// A workspace above the limit is a `ResourceLimit` error of
    /// [`SdeWorkspace::check`], [`SdeWorkspace::try_reserve`] and [`SdeMethod::advance`].
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = Some(bytes);
        self
    }
    /// Limits the number of attempted steps, including rejected refinement attempts.
    pub const fn with_max_steps(mut self, steps: Budget) -> Self {
        self.max_steps = Some(steps);
        self
    }
    /// Limits the number of drift and diffusion evaluations.
    pub const fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }
    /// Returns the refinement tolerance, or `None` for fixed steps.
    pub const fn refinement(&self) -> Option<&ErrorTolerance<S>> {
        self.refinement.as_ref()
    }
    /// Returns the workspace byte limit.
    pub const fn max_workspace_bytes(&self) -> Option<usize> {
        self.max_workspace_bytes
    }
    /// Returns the limit on attempted steps.
    pub const fn max_steps(&self) -> Option<Budget> {
        self.max_steps
    }
    /// Returns the limit on drift and diffusion evaluations.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }
    /// Budgets of one integration call.
    fn budgets(&self) -> SolveBudgets {
        let budgets = SolveBudgets::unbounded();
        let budgets = match self.max_steps {
            Some(steps) => budgets.with_limit(BudgetKind::Proposal, steps),
            None => budgets,
        };
        match self.max_evaluations {
            Some(evaluations) => budgets.with_limit(BudgetKind::Evaluation, evaluations),
            None => budgets,
        }
    }
}

/// Says why an SDE integration stopped.
///
/// The variants have the same meaning in every time-evolution family (see
/// [termination](crate#termination)). After an unsuccessful status the state holds the
/// last accepted point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SdeTermination {
    /// Every interval of the partition was accepted.
    ReachedEnd,
    /// Not produced: SDE integration has no events.
    TerminalEvent,
    /// A budget of the options or of an enclosing solve ran out: `Proposal` for steps,
    /// `Evaluation` for drift and diffusion calls.
    BudgetLimit(BudgetKind),
    /// Refinement needs a finer grid of increments than the one supplied.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the integration.
    ExecutionStopped(ExecutionStopReason),
}
crate::termination::shared_termination!(SdeTermination);

/// Reports the status, work and refinement diagnostics of one [`SdeMethod::advance`]
/// call.
///
/// The same report describes the progress of a failed call in
/// [`SdeWorkspace::last_attempt`] and in the [`PartialFailure`] of [`SdeMethod::solve`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct SdeReport<S: RealScalar> {
    termination: Option<SdeTermination>,
    final_time: S,
    grid_index: usize,
    steps: usize,
    trust: PartialResultTrust,
    counts: SolveCounts,
    last_step_error_estimate: Option<S>,
}
impl<S: RealScalar> SdeReport<S> {
    const fn initial(time: S) -> Self {
        Self {
            termination: None,
            final_time: time,
            grid_index: 0,
            steps: 0,
            trust: PartialResultTrust::DiagnosticOnly,
            counts: SolveCounts::new(),
            last_step_error_estimate: None,
        }
    }
    /// Returns why the integration stopped; `None` only in the progress of a failed call.
    pub const fn termination(&self) -> Option<SdeTermination> {
        self.termination
    }
    /// Returns `true` if every interval of the partition was accepted.
    pub const fn is_success(&self) -> bool {
        matches!(self.termination, Some(termination) if termination.is_success())
    }
    /// Returns the time of the last accepted state, or the first grid time if no step
    /// was accepted.
    pub const fn final_time(&self) -> S {
        self.final_time
    }
    /// Returns the grid index of the last accepted state; zero is the initial state.
    pub const fn grid_index(&self) -> usize {
        self.grid_index
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Returns the number of rejected refinement steps.
    pub const fn rejected_steps(&self) -> usize {
        self.counts.own().rejected()
    }

    /// Returns the trust level of the last accepted state; it is not an accuracy
    /// certificate.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the evaluations of this call, including failed and rejected attempts:
    /// `RightHandSide` counts drift calls and `LinearOperator` counts diffusion calls.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this call and of nested solves.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the normalized difference between one step and two half steps in the
    /// last refinement test, or `None` with fixed steps.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
}

/// Holds the final state and report of [`SdeMethod::solve`].
#[derive(Debug, Clone)]
#[must_use]
pub struct SdeSolution<S: RealScalar, T: NumericScalar = S> {
    state: OdeState<S, T>,
    report: SdeReport<S>,
}
impl<S: RealScalar, T: NumericScalar<Real = S>> SdeSolution<S, T> {
    /// Returns the last accepted state.
    pub const fn state(&self) -> &OdeState<S, T> {
        &self.state
    }
    /// Returns the report of the integration.
    pub const fn report(&self) -> &SdeReport<S> {
        &self.report
    }
    /// Returns the final state and the report.
    pub fn into_parts(self) -> (OdeState<S, T>, SdeReport<S>) {
        (self.state, self.report)
    }
}

/// Borrows or stores call-local execution parameters.
#[derive(Debug, Clone)]
#[must_use]
struct SdeRun<'a, S: RealScalar> {
    dimension: usize,
    noise_dimension: usize,
    method: SdeMethod,
    times: &'a [S],
    boundaries: Boundaries<'a>,
    options: &'a SdeOptions<S>,
    stack_capacity: usize,
}
impl<'a, S: RealScalar> SdeRun<'a, S> {
    /// Creates a plan for `dimension` state and `noise_dimension` Wiener components on
    /// the grid `times`, with steps between the grid indices in `boundaries`.
    ///
    /// `boundaries` must start at zero, end at `times.len() − 1` and increase strictly.
    /// Consecutive indices make a step of one grid interval; wider gaps sum the
    /// increments of the intervals they cover.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `times` or `boundaries` is empty, `times` is not strictly
    ///   increasing, `boundaries` does not partition the whole grid, or refinement is set
    ///   with [`SdeMethod::EulerMaruyama`].
    /// - `NonFinite` (origin `Input`) if a time is not finite.
    /// - `NumericalInstability` if `t_N − t₀` overflows.
    /// - `DimensionMismatch` if a vector refinement tolerance does not have `dimension`
    ///   entries.
    /// - `SizeOverflow` or `AllocationFailure` if the grids cannot be stored.
    fn new<T: NumericScalar<Real = S>>(
        method: SdeMethod,
        problem: &SdeProblem<'a, T>,
        options: &'a SdeOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let dimension = problem.system().dimension();
        let noise_dimension = problem.system().noise_dimension();
        let times = problem.times();
        let boundaries = Boundaries {
            explicit: problem.step_boundaries(),
            length: times.len(),
        };
        if times.is_empty() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("times"),
                scioxide_core::InvalidInputReason::Empty,
                "time grid must be nonempty",
            ));
        }
        if times.is_empty() || boundaries.len() == 0 {
            return Err(invalid(
                if times.is_empty() {
                    scioxide_core::argument!("times")
                } else {
                    scioxide_core::argument!("boundaries")
                },
                "time grid and boundary indices must be nonempty",
            ));
        }
        if times.iter().any(|t| !t.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "time grid must be finite",
            ));
        }
        if times.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                scioxide_core::argument!("times"),
                "time grid must be strictly increasing",
            ));
        }
        checked_difference(times[times.len() - 1], times[0])?;
        if boundaries.at(0) != 0
            || boundaries.at(boundaries.len() - 1) != times.len() - 1
            || boundaries.intervals().any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                scioxide_core::argument!("boundaries"),
                "step boundaries must partition the complete fine grid",
            ));
        }
        if options.refinement.is_some() && method == SdeMethod::EulerMaruyama {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "path-dependent refinement biases Euler–Maruyama; use fixed steps or stochastic Heun",
            ));
        }
        if let Some(absolute) = options
            .refinement
            .as_ref()
            .and_then(|tolerance| tolerance.absolute().as_vector())
            && absolute.len() != dimension
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(dimension),
                scioxide_core::Dimensions::one(absolute.len()),
            ));
        }
        checked_mul(
            times.len() - 1,
            noise_dimension,
            scioxide_core::quantity!("wiener_increments"),
        )?;
        let largest = boundaries
            .intervals()
            .map(|pair| pair[1] - pair[0])
            .max()
            .unwrap_or(0);
        // One pending right sibling per level, plus the current interval.
        let stack_capacity = if largest == 0 {
            0
        } else {
            (usize::BITS - (largest - 1).leading_zeros()) as usize + 1
        };
        Ok(Self {
            dimension,
            noise_dimension,
            method,
            times,
            boundaries,
            options,
            stack_capacity,
        })
    }
    /// Returns the shape `(times.len() − 1, noise_dimension)` of the increments.
    fn increment_shape(&self) -> (usize, usize) {
        (self.times.len() - 1, self.noise_dimension)
    }

    /// Returns the storage needed for states of scalar type `T`, with the workspace byte
    /// limit of the options.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the storage size overflows `usize`.
    fn workspace_requirements<T: NumericScalar<Real = S>>(
        &self,
    ) -> Result<SdeRequirements<T>, SciOxideError> {
        SdeRequirements::new(
            self.dimension,
            self.noise_dimension,
            self.stack_capacity,
            self.options.max_workspace_bytes,
        )
    }

    fn preflight<T: NumericScalar<Real = S>>(
        &self,
        problem: &SdeProblem<'_, T>,
        state: &OdeState<S, T>,
        workspace: &mut SdeWorkspace<T>,
    ) -> Result<(), SciOxideError> {
        let system = problem.system();
        for (expected, actual) in [
            (self.dimension, state.values().len()),
            (self.dimension, system.dimension()),
            (self.noise_dimension, system.noise_dimension()),
        ] {
            if actual != expected {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(expected),
                    scioxide_core::Dimensions::one(actual),
                ));
            }
        }
        let increments = problem.increments();
        let (rows, cols) = self.increment_shape();
        if (increments.rows(), increments.cols()) != (rows, cols) {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(rows, cols),
                scioxide_core::Dimensions::two(increments.rows(), increments.cols()),
            ));
        }
        if state.time() != self.times[0] {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("state"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "the state time must equal the first grid time",
            ));
        }
        crate::safety::finite(state.values(), NonFiniteOrigin::Input)?;
        for i in 0..rows {
            for j in 0..cols {
                if !RealScalar::is_finite(increments.get(i, j)?) {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "Wiener increments must be finite",
                    ));
                }
            }
        }
        let requirements = self.workspace_requirements::<T>()?;
        if workspace.check(&requirements)? == Capacity::NeedsGrowth {
            workspace.try_reserve(&requirements)?;
        }
        Ok(())
    }

    /// Integrates `problem` over the whole grid from `state`, which must be at the first
    /// grid time.
    ///
    /// Each accepted step is written to `state` and then sent to `sink`; the initial
    /// state is not sent. If an input check fails, `state` is unchanged and no callback
    /// runs. A workspace that is too small is grown first. A budget, a grid too coarse
    /// for refinement, cancellation or a deadline ends the call with `Ok` and an
    /// unsuccessful [`SdeReport`]; see [termination](crate#termination).
    /// [`SdeWorkspace::last_attempt`] records the progress of every call. `scope` is an
    /// [`ExecutionContext`](scioxide_core#execution-contexts) or the [`SolveContext`] of
    /// an enclosing solve. With a large enough workspace, and callbacks and a sink that
    /// do not allocate, the call does not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the state, the system or the increments do not match the
    ///   plan's dimensions or grid.
    /// - `InvalidInput` (`Inconsistent`) if `state.time()` is not the first grid time.
    /// - `NonFinite` (origin `Input`) if the state or an increment is not finite.
    /// - `NonFinite` (origin `Callback`) if the drift or diffusion returns NaN or
    ///   infinity.
    /// - `ResourceLimit`, `SizeOverflow` or `AllocationFailure` if the workspace cannot
    ///   be grown within the byte limit.
    /// - Errors returned by the system or the sink, unchanged.
    fn advance<'scope, 'capability, T, K>(
        &self,
        problem: &SdeProblem<'_, T>,
        state: &mut OdeState<S, T>,
        sink: &mut K,
        workspace: &mut SdeWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SdeReport<S>, SciOxideError>
    where
        T: NumericScalar<Real = S>,
        K: OdeOutputSink<S, T> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, state, workspace) {
            return workspace.shell.reject(error);
        }
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            self.method.algorithm_id(),
            ((self.dimension as u128) << 64) | self.noise_dimension as u128,
            0,
        );
        let SdeWorkspace { buffers, shell } = workspace;
        let mut snapshot = None;
        let executed = shell.execute(scope, identity, self.options.budgets(), |cx| {
            let mut report = SdeReport::initial(self.times[0]);
            let outcome = kernel::run(self, problem, state, sink, buffers, &mut report, cx);
            report.counts = cx.counts();
            report.trust = cx
                .accepted_state()
                .map_or(PartialResultTrust::DiagnosticOnly, |accepted| {
                    accepted.trust()
                });
            let result = match outcome {
                Ok(None) => Ok(SdeTermination::ReachedEnd),
                Ok(Some(stop)) => Ok(SdeTermination::from_stop(stop)),
                Err(fault) => fault
                    .stop()
                    .map(SdeTermination::from_stop)
                    .ok_or_else(|| fault.into()),
            };
            report.termination = result.as_ref().ok().copied();
            snapshot = Some(report);
            result.map(|_| report)
        });
        let failed = executed.result.is_err();
        let progress = snapshot.map(|mut report| {
            if failed {
                report.termination = None;
            }
            report
        });
        shell.conclude(executed, SdeReport::is_success, |_| progress)
    }
}

#[derive(Debug, Clone, Copy)]
struct Boundaries<'a> {
    explicit: Option<&'a [usize]>,
    length: usize,
}
impl Boundaries<'_> {
    fn len(self) -> usize {
        self.explicit.map_or(self.length, <[usize]>::len)
    }
    fn at(self, index: usize) -> usize {
        self.explicit.map_or(index, |b| b[index])
    }
    fn intervals(self) -> impl Iterator<Item = [usize; 2]> {
        (1..self.len()).map(move |i| [self.at(i - 1), self.at(i)])
    }
}
impl SdeMethod {
    /// Returns storage requirements after checking the grid and refinement options.
    ///
    /// # Errors
    /// Returns invalid grid, tolerance shape, overflow and configuration errors.
    pub fn workspace_requirements<T: NumericScalar>(
        self,
        problem: &SdeProblem<'_, T>,
        options: &SdeOptions<T::Real>,
    ) -> Result<SdeRequirements<T>, SciOxideError> {
        SdeRun::new(self, problem, options)?.workspace_requirements()
    }
    /// Allocates storage for the problem's state, noise and refinement stack.
    ///
    /// # Errors
    /// Returns the errors of [`Self::workspace_requirements`] and storage allocation.
    pub fn workspace<T: NumericScalar>(
        self,
        problem: &SdeProblem<'_, T>,
        options: &SdeOptions<T::Real>,
    ) -> Result<SdeWorkspace<T>, SciOxideError> {
        let mut ws = SdeWorkspace::new();
        ws.try_reserve(&self.workspace_requirements(problem, options)?)?;
        Ok(ws)
    }
    /// Integrates the problem grid, publishing each accepted step and its report.
    /// A prepared workspace and nonallocating callbacks make execution allocation-free.
    ///
    /// # Errors
    /// Preflight rejects invalid grids, shapes, non-finite inputs and refinement
    /// options without changing state. Callback and allocation errors are preserved.
    pub fn advance<'scope, 'capability, T: NumericScalar, K: OdeOutputSink<T::Real, T> + ?Sized>(
        self,
        problem: &SdeProblem<'_, T>,
        options: &SdeOptions<T::Real>,
        state: &mut OdeState<T::Real, T>,
        sink: &mut K,
        workspace: &mut SdeWorkspace<T>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SdeReport<T::Real>, SciOxideError> {
        let run = match SdeRun::new(self, problem, options) {
            Ok(run) => run,
            Err(error) => return workspace.shell.reject(error),
        };
        run.advance(problem, state, sink, workspace, scope)
    }
    /// Integrates from the first grid time with newly allocated storage.
    ///
    /// # Errors
    /// Returns the errors of [`Self::advance`] with accepted progress.
    #[allow(
        clippy::result_large_err,
        reason = "owned evolution retains accepted progress"
    )]
    #[allow(
        clippy::type_complexity,
        reason = "the owned form retains scalar-specific progress"
    )]
    pub fn solve<T: NumericScalar>(
        self,
        problem: &SdeProblem<'_, T>,
        options: &SdeOptions<T::Real>,
        initial: &[T],
    ) -> Result<SdeSolution<T::Real, T>, PartialFailure<SdeReport<T::Real>>> {
        let mut ws = self
            .workspace(problem, options)
            .map_err(PartialFailure::new)?;
        let mut state =
            OdeState::from_slice(problem.times()[0], initial).map_err(PartialFailure::new)?;
        match self.advance(
            problem,
            options,
            &mut state,
            &mut (),
            &mut ws,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(SdeSolution { state, report }),
            Err(error) => Err(ws.shell.into_failure(error)),
        }
    }
}

/// Integrates `dX = drift dt + B dW` by Euler–Maruyama and returns the full path.
///
/// Row `i` of `increments` is the Wiener increment over `[times[i], times[i+1]]`.
/// Row zero of the result is `y0`. Each interval is one step; storage is linear
/// in the path size. Strong order is `1/2` and weak order is `1` under the usual
/// smoothness and growth assumptions. The caller supplies all randomness.
///
/// # Errors
/// Returns the errors of [`SdeMethod::advance`], including an empty grid, invalid
/// shapes and non-finite inputs. An unsuccessful termination is an error with progress.
///
/// # Examples
/// ```
/// let increments = scioxide_linalg::MatrixView::from_slice(2, 1, &[0.25, -0.125])?;
/// let path = scioxide_ode::solve_sde(
///     |_t: f64, _x: &[f64], a: &mut [f64]| { a[0]=1.0; Ok(()) },
///     |_t: f64, _x: &[f64], dw: &[f64], b: &mut [f64]| { b[0]=0.5*dw[0]; Ok(()) },
///     &[0.0, 0.5, 1.0], increments, &[1.0],
/// ).map_err(scioxide_core::SciOxideError::from)?;
/// assert!((path.view().get(2, 0)? - 2.0625).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Kloeden and Platen, *Numerical Solution of Stochastic Differential Equations*, §10.2.
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains accepted progress"
)]
pub fn solve_sde<T, A, B>(
    drift: A,
    diffusion_product: B,
    times: &[T::Real],
    increments: MatrixView<'_, T::Real>,
    y0: &[T],
) -> Result<scioxide_linalg::Matrix<T>, PartialFailure<SdeReport<T::Real>>>
where
    T: NumericScalar,
    A: Fn(T::Real, &[T], &mut [T]) -> Result<(), SciOxideError>,
    B: Fn(T::Real, &[T], &[T::Real], &mut [T]) -> Result<(), SciOxideError>,
{
    let system = crate::SdeFn::new(y0.len(), increments.cols(), drift, diffusion_product);
    let problem = SdeProblem::new(&system, times, increments);
    let options = SdeOptions::default();
    let method = SdeMethod::EulerMaruyama;
    let mut ws = method
        .workspace(&problem, &options)
        .map_err(PartialFailure::new)?;
    let mut state = OdeState::from_slice(times[0], y0).map_err(PartialFailure::new)?;
    let mut path =
        scioxide_linalg::Matrix::zeros(times.len(), y0.len()).map_err(PartialFailure::new)?;
    for (j, &value) in y0.iter().enumerate() {
        path.view_mut()
            .set(0, j, value)
            .map_err(PartialFailure::new)?;
    }
    let mut row = 0;
    let mut sink = |_time, values: &[T]| {
        row += 1;
        for (j, &value) in values.iter().enumerate() {
            path.view_mut().set(row, j, value)?;
        }
        Ok(())
    };
    match method.advance(
        &problem,
        &options,
        &mut state,
        &mut sink,
        &mut ws,
        ExecutionContext::sequential_native(),
    ) {
        Ok(report) => {
            if let Some(stop) = report.termination().and_then(Terminal::stop) {
                return Err(ws
                    .shell
                    .into_failure(crate::termination::stop_error(stop, report.accounting())));
            }
            Ok(path)
        }
        Err(error) => Err(ws.shell.into_failure(error)),
    }
}
