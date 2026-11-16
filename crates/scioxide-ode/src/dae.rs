//! Fully implicit index-1 differential-algebraic equations `F(t, y, ẏ) = 0`.
//!
//! [`DaeBdf`] integrates a [`DaeProblem`] from a [`DaeState`] with variable-step,
//! variable-order BDF after checking or projecting the initial pair. The stage and
//! consistency equations are solved by Newton's method on dense, sparse or matrix-free
//! linear systems.
pub(crate) mod bdf;
pub(crate) mod derivatives;
pub(crate) mod linear;
mod options;
mod plan;
mod report;

pub use options::{DaeInitialization, DaeNumericalOptions, DaeOptions};
pub use plan::{
    DaeBdf, DaeOutputSink, DaeProblem, DaeRequirements, DaeSolution, DaeState, DaeWorkspace,
};
pub use report::{DaeConsistencyReport, DaeReport, DaeTermination};

use crate::OdeStageBackend;
use crate::ProportionalIntegralController;
use crate::safety::finite;
use crate::safety::invalid;
use crate::step_control::PiController;
use crate::stiff::newton::{NewtonOutcome, NewtonReport, NewtonStage, iterate};
use crate::termination::{Fault, Work, admit};
use alloc::vec::Vec;
use linear::{DaeLinear, EffectiveJacobian};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    ConvergenceReason, EvaluationRole, LinearOperator, NumericScalar, PartialResultTrust,
    Preconditioner, RealScalar, SciOxideError, SolveContext, StructuralPatternView,
};
use scioxide_linalg::{Matrix, MatrixViewMut};

// A linear defect of at most 1% leaves the nonlinear correction accuracy dominant.
const LINEAR_FORCING: f64 = 0.01;
fn real<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value)
}

/// Defines a fully implicit index-1 DAE through its residual and optional derivatives.
///
/// ```text
/// F(t, y, ẏ) = 0,   F(t, y, ẏ) ∈ 𝕂ⁿ,
/// ```
///
/// where `t` is the real time, `y ∈ 𝕂ⁿ` the state, `ẏ = dy/dt` its time derivative and
/// `𝕂` the real or complex scalar type `T`. A [`DaePartition`] marks each variable and
/// each equation as differential or algebraic. The system must satisfy:
///
/// - differential equations depend only on the derivatives of differential variables,
///   and algebraic equations depend on no derivatives;
/// - index one: the Jacobian of `F` with respect to the differential derivatives and the
///   algebraic states is nonsingular;
/// - for complex `T`, `F` is holomorphic in `y` and `ẏ`; write a nonholomorphic system
///   in real coordinates.
///
/// Only [`residual`](Self::residual) is required. Derivative blocks come from the
/// sources of [`DaeNumericalOptions::derivative_strategy`] in order: by default the
/// callback for the stage backend ([`derivatives`](Self::derivatives),
/// [`sparse_derivatives`](Self::sparse_derivatives) or
/// [`derivative_product`](Self::derivative_product)), then forward differences of the
/// residual. Automatic differentiation is not available for DAE residuals and is
/// skipped. A callback that returns `Ok(None)` is unavailable and the next source is
/// tried; an error is returned unchanged. Difference increments of the states scale with
/// the absolute state tolerance of their partition, and increments of the differential
/// derivatives with the absolute initial-derivative tolerance; a zero absolute tolerance
/// is replaced by one. A Newton iterate is accepted only when the residual itself meets
/// the differential and algebraic residual tolerances, so an approximate Jacobian
/// affects the iteration count rather than the accepted solution.
///
/// Callbacks write every output component as a finite value; NaN or infinity is a
/// `NonFinite` error with origin `Callback`. Input and output slices do not overlap.
/// Every callback receives the [`SolveContext`] of the integration. Calls count toward
/// the evaluation limit under the role named on each method; charge only your own nested
/// work, such as an inner solve, to `cx` (see
/// [solve accounting](scioxide_core#solve-accounting)).
///
/// # Errors
///
/// An error returned by any callback stops the integration and is returned unchanged.
pub trait DaeSystem<T: NumericScalar> {
    /// Returns the number of equations, which equals the number of state components.
    fn dimension(&self) -> usize;

    /// Returns an optional upper bound on the step magnitude `|h|` at the accepted point
    /// `(time, state, derivative)`.
    ///
    /// The integrator calls it before each step attempt and does not count the call as an
    /// evaluation. Return a positive finite bound, or `Ok(None)` (the default) for no
    /// bound.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged.
    fn step_limit(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<T::Real>, SciOxideError> {
        Ok(None)
    }

    /// Writes the residual `F(time, state, derivative)` to `output`.
    ///
    /// Each call counts as one `Residual` evaluation.
    ///
    /// # Errors
    ///
    /// An error, for example a domain error or a nested budget stop, ends the integration
    /// and is returned unchanged.
    fn residual(
        &self,
        time: T::Real,
        state: &[T],
        derivative: &[T],
        output: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;

    /// Writes the dense Jacobian blocks `∂F/∂y` to `state_block` and `∂F/∂ẏ` to
    /// `derivative_block`, or returns `Ok(None)` when they are unavailable (the default).
    ///
    /// The dense stage backend uses it. Entries of `∂F/∂ẏ` outside the rows of
    /// differential equations and the columns of differential variables must be zero.
    /// Each call counts as one `Jacobian` evaluation.
    ///
    /// # Errors
    ///
    /// An error is returned unchanged. A nonzero `∂F/∂ẏ` entry that the [`DaePartition`]
    /// forbids is an `InvalidInput` error.
    fn derivatives(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _state_block: &mut MatrixViewMut<'_, T>,
        _derivative_block: &mut MatrixViewMut<'_, T>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Writes the values of `∂F/∂y` and `∂F/∂ẏ` in the compressed sparse row `pattern`, or
    /// returns `Ok(None)` when they are unavailable (the default).
    ///
    /// The sparse stage backend uses it. Both blocks share the pattern of the backend,
    /// which covers the nonzeros of both blocks and every diagonal entry; write an
    /// explicit zero where a block has no entry. Each call counts as one `Jacobian`
    /// evaluation.
    ///
    /// # Errors
    ///
    /// An error is returned unchanged. A nonzero `∂F/∂ẏ` entry that the [`DaePartition`]
    /// forbids is an `InvalidInput` error.
    #[allow(clippy::too_many_arguments)]
    fn sparse_derivatives(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _pattern: StructuralPatternView<'_>,
        _state_values: &mut [T],
        _derivative_values: &mut [T],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Writes the directional derivative
    /// `(∂F/∂y) state_direction + (∂F/∂ẏ) derivative_direction` to `output`, or returns
    /// `Ok(None)` when it is unavailable (the default).
    ///
    /// The matrix-free ([`OdeStageBackend::Krylov`]) backend uses it. `scratch` holds at
    /// least the operator scratch length set in that backend. The integrator counts each
    /// operator application as one `LinearOperator` evaluation; this callback is not
    /// counted separately.
    ///
    /// # Errors
    ///
    /// An error is returned unchanged.
    #[allow(clippy::too_many_arguments)]
    fn derivative_product(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _state_direction: &[T],
        _derivative_direction: &[T],
        _output: &mut [T],
        _scratch: &mut [T],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Returns an optional right preconditioner for the Newton matrix of the matrix-free
    /// backend.
    ///
    /// During a BDF step the Newton matrix is `∂F/∂y + inverse_step · ∂F/∂ẏ`. When
    /// `initialization` is true it is the consistency Jacobian, whose columns are
    /// `∂F/∂ẏ` for differential variables and `∂F/∂y` for algebraic variables. You
    /// decide when to rebuild the preconditioner. Each call counts as one
    /// `Preconditioner` evaluation.
    ///
    /// # Errors
    ///
    /// An error is returned unchanged.
    fn preconditioner(
        &self,
        _time: T::Real,
        _state: &[T],
        _derivative: &[T],
        _initialization: bool,
        _inverse_step: T::Real,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn Preconditioner<T>>, SciOxideError> {
        Ok(None)
    }
}

/// Marks each variable and each equation of a DAE as differential or algebraic.
///
/// Entry `i` of `variables` is `true` when `yᵢ` is a differential variable, whose
/// derivative `ẏᵢ` appears in `F`; entry `j` of `equations` is `true` when equation `j`
/// is a differential equation. Both lists have the same number of `true` entries, but the
/// orders may differ.
#[derive(Debug, Clone)]
pub struct DaePartition {
    variables: Vec<bool>,
    equations: Vec<bool>,
    differential: usize,
}
impl DaePartition {
    /// Copies the variable and equation roles.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `variables` is empty, the two lists differ in length, or they
    ///   have different numbers of differential entries.
    /// - `AllocationFailure` if the copies cannot be allocated.
    pub fn new(variables: &[bool], equations: &[bool]) -> Result<Self, SciOxideError> {
        let differential = variables.iter().filter(|&&d| d).count();
        if variables.is_empty()
            || variables.len() != equations.len()
            || differential != equations.iter().filter(|&&d| d).count()
        {
            return Err(invalid(
                if variables.is_empty() {
                    scioxide_core::argument!("variables")
                } else {
                    scioxide_core::argument!("equations")
                },
                "variable and equation partitions must have equal nonzero dimensions and role counts",
            ));
        }
        let mut v = scioxide_core::backend::try_filled(
            variables.len(),
            false,
            scioxide_core::quantity!("variables"),
        )?;
        let mut e = scioxide_core::backend::try_filled(
            equations.len(),
            false,
            scioxide_core::quantity!("equations"),
        )?;
        v.copy_from_slice(variables);
        e.copy_from_slice(equations);
        Ok(Self {
            variables: v,
            equations: e,
            differential,
        })
    }
    /// Returns the number of variables, which equals the number of equations.
    pub fn dimension(&self) -> usize {
        self.variables.len()
    }
    /// Returns the variable roles; `true` marks a differential variable.
    pub fn variables(&self) -> &[bool] {
        &self.variables
    }
    /// Returns the equation roles; `true` marks a differential equation.
    pub fn equations(&self) -> &[bool] {
        &self.equations
    }
    /// Number of differential components.
    pub(crate) const fn differential(&self) -> usize {
        self.differential
    }
}

/// Numerical storage of residual BDF integration: the accepted pair, BDF history,
/// Newton candidates and linear storage of one prepared stage backend.
///
/// Each integration call starts fresh history; only converged, residual-consistent
/// endpoints satisfying both partitioned local error tests enter the history.
pub(crate) struct DaeStorage<T: NumericScalar> {
    backend: OdeStageBackend,
    linear: DaeLinear<T>,
    accepted: Vec<T>,
    accepted_derivative: Vec<T>,
    history: Vec<T>,
    history_times: [T::Real; bdf::MAX_ORDER],
    history_count: usize,
    order: usize,
    residence: usize,
    rejected: usize,
    /// A stage at the current order failed to converge, and every stage accepted since
    /// converged only on a Newton recovery retry.
    limited: bool,
    controller: PiController<T::Real>,
    candidate: Vec<T>,
    candidate_derivative: Vec<T>,
    residual: Vec<T>,
    correction: Vec<T>,
    /// Componentwise local error estimate of the candidate.
    local: Vec<T>,
    errors: Vec<T::Real>,
    scales: Vec<T::Real>,
}

/// Prepared recording hooks for residual integration; the formula defaults to
/// variable-order BDF.
#[allow(clippy::too_many_arguments)]
pub(crate) trait Recorder<T: NumericScalar>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    /// Reuse linearizations between ordinary forward proposals. Recorders disable
    /// this so replay depends only on recorded history, including after rejection.
    fn reuse_between_steps(&self) -> bool {
        true
    }
    fn boundary(&mut self, _time: T::Real) -> Result<Option<T::Real>, SciOxideError> {
        Ok(None)
    }
    fn fixed_step(&self) -> bool {
        false
    }
    fn formula(&self) -> bdf::Formula {
        bdf::Formula::Bdf {
            maximum_order: bdf::MAX_ORDER,
        }
    }
    fn trial(
        &mut self,
        _primal: &DaeStorage<T>,
        _time: T::Real,
        _width: T::Real,
        _trial: bdf::Trial<T::Real>,
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, SciOxideError> {
        Ok(T::Real::ZERO)
    }
    fn accepted(
        &mut self,
        _primal: &DaeStorage<T>,
        _start: T::Real,
        _width: T::Real,
        _end: T::Real,
        _trial: bdf::Trial<T::Real>,
        _context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}
impl<T: NumericScalar> Recorder<T> for () where T::Real: NumericScalar<Real = T::Real> {}

/// Initial data for which consistency processing accepted no pair; each consumer maps it
/// to its own error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unaccepted {
    /// `Check` found a residual block above its tolerance.
    Inconsistent,
    /// Projection Newton failed for this reason after this many iterations.
    Projection(ConvergenceReason, usize),
}

impl Unaccepted {
    /// The [`DaeBdf::advance`] error: invalid input or a convergence failure.
    pub(crate) fn error(self) -> SciOxideError {
        match self {
            Self::Inconsistent => SciOxideError::invalid_input(
                scioxide_core::argument!("state"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "initial pair violates a DAE residual tolerance",
            ),
            Self::Projection(reason, iterations) => SciOxideError::convergence_failure(
                reason,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(
                    scioxide_core::EvaluationCounts::from_iterations(iterations).into(),
                ),
                "DAE consistency projection did not converge",
            ),
        }
    }
}

impl<T: NumericScalar> DaeStorage<T>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    /// Reserve storage of dimension `n` for a prepared `backend`, whose Krylov operator
    /// scratch already includes the internal direction and difference buffers.
    /// # Errors
    /// Rejects backend dimensions, arithmetic overflow, resources, or allocation failure.
    pub(crate) fn new(n: usize, backend: &OdeStageBackend) -> Result<Self, SciOxideError> {
        let filled = |quantity| scioxide_core::backend::try_filled(n, T::ZERO, quantity);
        let real_filled = |quantity| scioxide_core::backend::try_filled(n, T::Real::ZERO, quantity);
        Ok(Self {
            backend: backend.clone(),
            linear: DaeLinear::new(n, backend, backend)?,
            accepted: filled(scioxide_core::quantity!("accepted"))?,
            accepted_derivative: filled(scioxide_core::quantity!("accepted_derivative"))?,
            history: scioxide_core::backend::try_filled(
                scioxide_core::backend::checked_mul(
                    n,
                    bdf::MAX_ORDER,
                    scioxide_core::quantity!("bdf_history"),
                )?,
                T::ZERO,
                scioxide_core::quantity!("bdf_history"),
            )?,
            history_times: [T::Real::ZERO; bdf::MAX_ORDER],
            history_count: 0,
            order: 1,
            residence: 0,
            rejected: 0,
            limited: false,
            controller: PiController::new(ProportionalIntegralController::integral_defaults()),
            candidate: filled(scioxide_core::quantity!("candidate"))?,
            candidate_derivative: filled(scioxide_core::quantity!("candidate_derivative"))?,
            residual: filled(scioxide_core::quantity!("residual"))?,
            correction: filled(scioxide_core::quantity!("correction"))?,
            local: filled(scioxide_core::quantity!("local_error"))?,
            errors: real_filled(scioxide_core::quantity!("errors"))?,
            scales: real_filled(scioxide_core::quantity!("scales"))?,
        })
    }

    /// Whether this storage serves dimension `n` on the prepared `backend`.
    pub(crate) fn fits(&self, n: usize, backend: &OdeStageBackend) -> bool {
        self.accepted.len() == n && self.backend == *backend
    }

    /// Dimension of the stored pair.
    pub(crate) fn dimension(&self) -> usize {
        self.accepted.len()
    }

    /// Consistency check or projection of `(state, derivative)` at `start`, shared by
    /// forward integration and reconstructed adjoint points. Both slices change only
    /// when a consistent pair is accepted; otherwise the inner value says why none was.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        state: &mut [T],
        derivative: &mut [T],
        start: T::Real,
        context: &mut SolveContext<'_, '_>,
        report: &mut DaeReport<T::Real>,
    ) -> Result<Result<(), Unaccepted>, Fault> {
        self.linear.invalidate();
        let n = self.dimension();
        if state.len() != n || derivative.len() != n {
            return Err(invalid(
                if state.len() != n {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("derivative")
                },
                "state/provider/workspace dimension mismatch",
            )
            .into());
        }
        if !state
            .iter()
            .chain(derivative.iter())
            .all(|value| value.has_finite_components())
        {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "initial state and derivative must be finite",
            )
            .into());
        }
        let options = plan.1.numerics();
        self.accepted.copy_from_slice(state);
        self.accepted_derivative.copy_from_slice(derivative);
        self.candidate.copy_from_slice(&self.accepted);
        self.candidate_derivative
            .copy_from_slice(&self.accepted_derivative);
        evaluate(
            problem,
            start,
            &self.candidate,
            &self.candidate_derivative,
            &mut self.residual,
            context,
        )?;
        let initial = residual_norms(plan.0.partition(), &self.residual);
        let mut consistency = DaeConsistencyReport {
            initial_differential_residual: initial.0,
            initial_algebraic_residual: initial.1,
            differential_residual: initial.0,
            algebraic_residual: initial.1,
            state_correction: T::Real::ZERO,
            derivative_correction: T::Real::ZERO,
        };
        report.consistency = Some(consistency);
        let check = options.initialization() == DaeInitialization::Check;
        if check && !residual_ok(initial, options)? {
            return Ok(Err(Unaccepted::Inconsistent));
        }
        // Even zero-residual initial data must exercise the mixed Jacobian. Direct
        // factors detect rank failure; matrix-free regularity remains a provider assumption.
        if let (NewtonOutcome::Failed(reason, newton), _) =
            self.newton(plan, problem, start, T::Real::ONE, true, context)?
        {
            return Ok(Err(Unaccepted::Projection(reason, newton.iterations)));
        }
        if check {
            // Check mode must preserve supplied values exactly, including unused derivatives.
            self.candidate.copy_from_slice(&self.accepted);
            self.candidate_derivative
                .copy_from_slice(&self.accepted_derivative);
            evaluate(
                problem,
                start,
                &self.candidate,
                &self.candidate_derivative,
                &mut self.residual,
                context,
            )?;
        }
        let final_residual = residual_norms(plan.0.partition(), &self.residual);
        if !residual_ok(final_residual, options)? {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "initial residual changed during consistency checking",
            )
            .into());
        }
        context.execution().check()?;
        consistency.differential_residual = final_residual.0;
        consistency.algebraic_residual = final_residual.1;
        for ((&candidate, &accepted), (&candidate_rate, &accepted_rate)) in
            self.candidate.iter().zip(&self.accepted).zip(
                self.candidate_derivative
                    .iter()
                    .zip(&self.accepted_derivative),
            )
        {
            consistency.state_correction = consistency
                .state_correction
                .max(crate::safety::state_difference(candidate, accepted)?.abs());
            consistency.derivative_correction = consistency
                .derivative_correction
                .max(crate::safety::state_difference(candidate_rate, accepted_rate)?.abs());
        }
        self.accepted.copy_from_slice(&self.candidate);
        self.accepted_derivative
            .copy_from_slice(&self.candidate_derivative);
        report.consistency = Some(consistency);
        report.differential_residual = final_residual.0;
        report.algebraic_residual = final_residual.1;
        context.assess_accepted(PartialResultTrust::ResidualChecked)?;
        state.copy_from_slice(&self.accepted);
        derivative.copy_from_slice(&self.accepted_derivative);
        report.has_accepted_state = true;
        Ok(Ok(()))
    }

    /// The single residual proposal, acceptance and continuation loop of the
    /// recorder's [`bdf::Formula`].
    /// `initial_step` is a positive magnitude; direction follows `end - start`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_steps<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        start: T::Real,
        end: T::Real,
        initial_step: T::Real,
        context: &mut SolveContext<'_, '_>,
        sink: &mut impl FnMut(T::Real, &[T], &[T]) -> Result<(), SciOxideError>,
        report: &mut DaeReport<T::Real>,
        recorder: &mut impl Recorder<T>,
    ) -> Result<(), Fault> {
        let direction = if end >= start {
            T::Real::ONE
        } else {
            -T::Real::ONE
        };
        let mut step = initial_step * direction;
        self.reset_bdf(start);
        while report.time != end {
            let boundary = recorder.boundary(report.time)?.unwrap_or(end);
            match admit(context, Work::Proposal) {
                Ok(()) => {}
                Err(Fault::Denied(denial)) => {
                    report.termination = Some(DaeTermination::BudgetLimit(denial.kind()));
                    return Ok(());
                }
                Err(fault) => return Err(fault),
            }
            let limit = problem.step_limit(
                report.time,
                &self.accepted,
                &self.accepted_derivative,
                context,
            )?;
            let Some((selected, endpoint)) = crate::safety::select_step(
                report.time,
                boundary,
                direction,
                step.abs(),
                None,
                limit,
                scioxide_core::argument!("self"),
            )?
            else {
                report.termination = Some(DaeTermination::StepSizeLimit);
                return Ok(());
            };
            step = selected;
            let formula = recorder.formula();
            let numerator = if formula == bdf::Formula::Trapezoid {
                T::Real::TWO
            } else {
                T::Real::ONE
            };
            if !(numerator / step).is_finite() {
                report.termination = Some(DaeTermination::StepSizeLimit);
                return Ok(());
            }
            if !recorder.reuse_between_steps() {
                self.linear.invalidate();
            }
            let trial = match formula {
                bdf::Formula::Bdf { maximum_order } => {
                    self.propose(plan, problem, report.time, step, maximum_order, context)?
                }
                bdf::Formula::Trapezoid => {
                    self.propose_trapezoid(plan, problem, report.time, step, context)?
                }
            };
            let (differential_error, algebraic_error) = trial.errors;
            let extra = recorder.trial(self, report.time, step, trial, context)?;
            report.last_step_error_estimate =
                if formula == bdf::Formula::Trapezoid || !trial.converged {
                    None
                } else {
                    Some(differential_error.max(algebraic_error).max(extra))
                };
            let (next, accept) =
                match self.assess_bdf(plan, trial, step, extra, recorder.fixed_step()) {
                    Ok(value) => value,
                    Err(error)
                        if matches!(
                            error.view(),
                            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                                reason: ConvergenceReason::NoSuccessfulCandidate,
                                ..
                            }
                        ) =>
                    {
                        context.record_rejected()?;
                        report.termination = Some(DaeTermination::RejectionLimit);
                        return Ok(());
                    }
                    Err(error) => return Err(error.into()),
                };
            if accept {
                context.execution().check()?;
                let norms = residual_norms(plan.0.partition(), &self.residual);
                recorder.accepted(self, report.time, step, endpoint, trial, context)?;
                self.accept_bdf(report.time, trial);
                context.commit_accepted(PartialResultTrust::ResidualChecked)?;
                report.time = endpoint;
                report.has_accepted_state = true;
                report.steps = report.steps.checked_add(1).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("accepted_steps"))
                })?;
                report.differential_error = differential_error;
                report.algebraic_error = algebraic_error;
                report.differential_residual = norms.0;
                report.algebraic_residual = norms.1;
                sink(endpoint, &self.accepted, &self.accepted_derivative)?;
            } else {
                context.record_rejected()?;
                if report.time + next == report.time {
                    report.termination = Some(DaeTermination::StepSizeLimit);
                    return Ok(());
                }
            }
            step = if recorder.fixed_step() {
                initial_step * direction
            } else {
                next
            };
        }
        Ok(())
    }

    /// Solve the stage (`initialization == false`) or mixed consistency equation; the
    /// failed outcome is the last recovery attempt's. The flag marks a stage that
    /// converged only on an exact-factor or fresh-block retry. A GMRES correction whose
    /// verified defect exceeds its forcing fails the attempt as a nested-solver failure.
    fn newton<P: DaeSystem<T> + ?Sized>(
        &mut self,
        plan: (&DaeBdf, &DaeOptions<T::Real>),
        problem: &P,
        time: T::Real,
        inverse_step: T::Real,
        initialization: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(NewtonOutcome<T::Real>, bool), Fault> {
        let mut stage = DaeStage {
            problem,
            plan,
            linear: &mut self.linear,
            candidate: &mut self.candidate,
            candidate_derivative: &mut self.candidate_derivative,
            accepted: &self.accepted,
            accepted_derivative: &self.accepted_derivative,
            residual: &mut self.residual,
            correction: &mut self.correction,
            errors: &mut self.errors,
            scales: &mut self.scales,
            time,
            inverse_step,
            initialization,
            context,
            evaluated: false,
            prepared: false,
            recovery: 0,
            iterations: 0,
            inexact: false,
        };
        if initialization && plan.1.numerics().initialization() == DaeInitialization::Check {
            // A rank probe: the mixed Jacobian must factor, but Check mode
            // preserves the supplied values exactly.
            stage.residual()?;
            stage.correction()?;
            return Ok((
                NewtonOutcome::Converged(NewtonReport {
                    iterations: 1,
                    corrections: 1,
                    rate: None,
                }),
                false,
            ));
        }
        let mut recovery = 0;
        loop {
            stage.recovery = recovery;
            stage.iterations = 0;
            let outcome = match iterate(&mut stage, plan.1.numerics().newton_limit().get()) {
                // The correction returned only to stop the iteration.
                Err(_) if core::mem::take(&mut stage.inexact) => NewtonOutcome::Failed(
                    ConvergenceReason::NestedSolver,
                    NewtonReport {
                        iterations: stage.iterations,
                        corrections: stage.iterations.saturating_sub(1),
                        rate: None,
                    },
                ),
                outcome => outcome?,
            };
            if let NewtonOutcome::Converged(report) = outcome {
                stage.linear.observe_newton(report)?;
                return Ok((outcome, recovery > 0));
            }
            if initialization || !stage.linear.frozen() {
                return Ok((outcome, false));
            }
            if recovery == 0 {
                stage.linear.invalidate_factor();
            } else {
                stage.linear.invalidate();
            }
            stage.prepared = false;
            stage.evaluated = false;
            if recovery == 2 {
                return Ok((outcome, false));
            }
            recovery += 1;
        }
    }
}

fn evaluate<T: NumericScalar, P: DaeSystem<T> + ?Sized>(
    problem: &P,
    time: T::Real,
    y: &[T],
    yd: &[T],
    out: &mut [T],
    context: &mut SolveContext<'_, '_>,
) -> Result<(), Fault> {
    finite(y, scioxide_core::NonFiniteOrigin::Callback)?;
    finite(yd, scioxide_core::NonFiniteOrigin::Callback)?;
    admit(context, EvaluationRole::Residual)?;
    out.fill(T::from_real(T::Real::NAN));
    problem.residual(time, y, yd, out, context)?;
    Ok(finite(out, scioxide_core::NonFiniteOrigin::Callback)?)
}
fn residual_norms<T: NumericScalar>(
    partition: &DaePartition,
    residual: &[T],
) -> (T::Real, T::Real) {
    let selected = |differential| {
        residual
            .iter()
            .zip(&partition.equations)
            .filter_map(move |(&value, &role)| (role == differential).then_some(value))
    };
    (
        crate::safety::max_abs(selected(true)),
        crate::safety::max_abs(selected(false)),
    )
}
fn residual_ok<S: RealScalar>(
    norms: (S, S),
    options: &DaeNumericalOptions<S>,
) -> Result<bool, SciOxideError> {
    Ok(norms.0
        <= options
            .residual_tolerances()?
            .0
            .tolerance()
            .threshold(options.differential_scale())?
        && norms.1
            <= options
                .residual_tolerances()?
                .1
                .tolerance()
                .threshold(options.algebraic_scale())?)
}
fn weights<S: RealScalar>(
    partition: &DaePartition,
    i: usize,
    initialization: bool,
    inverse: S,
) -> (S, S) {
    if initialization {
        if partition.variables[i] {
            (S::ZERO, S::ONE)
        } else {
            (S::ONE, S::ZERO)
        }
    } else {
        (S::ONE, inverse)
    }
}
/// GMRES corrections use one percent of the original residual tolerance, as a
/// relative defect between `√ε` and one percent; direct solves are not verified against
/// it. The original residual blocks still independently gate Newton acceptance.
fn linear_forcing<T: NumericScalar>(
    partition: &DaePartition,
    residual: &[T],
    options: &DaeNumericalOptions<T::Real>,
) -> Result<T::Real, SciOxideError> {
    let cap = real::<T::Real>(LINEAR_FORCING)?;
    let norm = crate::safety::scaled_rms_norm(residual)?;
    if norm == T::Real::ZERO {
        return Ok(cap);
    }
    let mut target = T::Real::INFINITY;
    if partition.differential > 0 {
        target = options
            .residual_tolerances()?
            .0
            .tolerance()
            .threshold(options.differential_scale())?;
    }
    if partition.differential < partition.dimension() {
        target = target.min(
            options
                .residual_tolerances()?
                .1
                .tolerance()
                .threshold(options.algebraic_scale())?,
        );
    }
    Ok((cap * target / norm).max(T::Real::SQRT_EPSILON).min(cap))
}
/// One consistency or residual BDF stage equation of the residual DAE.
struct DaeStage<'s, 'c, 'scope, 'capability, T: NumericScalar, P: DaeSystem<T> + ?Sized> {
    problem: &'s P,
    plan: (&'s DaeBdf, &'s DaeOptions<T::Real>),
    linear: &'s mut DaeLinear<T>,
    candidate: &'s mut [T],
    candidate_derivative: &'s mut [T],
    accepted: &'s [T],
    accepted_derivative: &'s [T],
    residual: &'s mut [T],
    correction: &'s mut [T],
    errors: &'s mut [T::Real],
    scales: &'s mut [T::Real],
    time: T::Real,
    inverse_step: T::Real,
    initialization: bool,
    context: &'c mut SolveContext<'scope, 'capability>,
    /// The residual already holds `F` at the iterate.
    evaluated: bool,
    /// Direct corrections share the first correction's frozen linearization.
    prepared: bool,
    recovery: usize,
    /// Iterations of the current attempt.
    iterations: usize,
    /// The last correction failed its verified GMRES defect.
    inexact: bool,
}

impl<T: NumericScalar, P: DaeSystem<T> + ?Sized> NewtonStage<T::Real>
    for DaeStage<'_, '_, '_, '_, T, P>
where
    T::Real: NumericScalar<Real = T::Real>,
{
    type Error = Fault;
    fn residual(&mut self) -> Result<bool, Fault> {
        admit(self.context, Work::Iteration)?;
        self.iterations += 1;
        if !core::mem::take(&mut self.evaluated) {
            evaluate(
                self.problem,
                self.time,
                self.candidate,
                self.candidate_derivative,
                self.residual,
                self.context,
            )?;
        }
        // Even zero-residual initial data must exercise the mixed Jacobian. Direct
        // factors detect rank failure; matrix-free regularity remains a provider assumption.
        Ok(!self.initialization && self.residual.iter().all(|value| *value == T::ZERO))
    }

    fn correction(&mut self) -> Result<(), Fault> {
        // Projection can start far from the consistency manifold and has no BDF
        // predictor. Its mixed state/rate equation uses full Newton.
        if self.initialization {
            self.linear.invalidate();
            self.prepared = false;
        }
        let preconditioner = if self.plan.0.operator_mode() {
            admit(self.context, EvaluationRole::Preconditioner)?;
            self.problem.preconditioner(
                self.time,
                self.candidate,
                self.candidate_derivative,
                self.initialization,
                self.inverse_step,
                self.context,
            )?
        } else {
            None
        };
        let partition = self.plan.0.partition();
        let mut effective = BoundDerivative {
            source: core::cell::Cell::new(None),
            problem: self.problem,
            options: self.plan.1.numerics(),
            base: self.residual,
            partition,
            time: self.time,
            state: self.candidate,
            derivative: self.candidate_derivative,
            initialization: self.initialization,
            inverse_step: self.inverse_step,
            scratch: self.plan.0.operator_scratch(),
            preconditioner,
        };
        if !self.prepared {
            if self.recovery == 1 && self.linear.frozen() && !self.initialization {
                self.linear
                    .refactor(&mut effective, self.inverse_step, self.context)?;
            } else {
                self.linear.prepare(
                    &mut effective,
                    Some(linear::NewtonPoint {
                        time: self.time,
                        state: self.candidate,
                        derivative: self.candidate_derivative,
                        strategy: self.plan.1.numerics().derivative_strategy(),
                        initialization: self.initialization,
                        inverse_step: self.inverse_step,
                    }),
                    self.context,
                )?;
            }
            self.prepared = true;
        }
        if self.linear.solve_stage(
            &mut effective,
            self.residual,
            self.correction,
            linear_forcing(partition, self.residual, self.plan.1.numerics())?,
            self.context,
        )? {
            return Ok(());
        }
        self.inexact = true;
        Err(SciOxideError::convergence_failure(
            ConvergenceReason::NestedSolver,
            scioxide_core::ConvergenceDiagnostics::new(),
            "linear defect exceeds requested accuracy",
        )
        .into())
    }

    fn frozen(&self) -> bool {
        !self.initialization && self.linear.frozen()
    }

    fn requires_rate(&self) -> bool {
        self.linear.requires_rate()
    }

    /// Partitioned correction norm; one is ten percent of each role's tolerance. Scales
    /// are the larger of the corrected and accepted magnitudes of the corrected unknown,
    /// as in endpoint acceptance, so a component that starts at zero can move.
    fn norm(&mut self) -> Result<T::Real, Fault> {
        let options = self.plan.1.numerics();
        let mut result = T::Real::ZERO;
        for role in [true, false] {
            let mut count = 0;
            for (i, &d) in self.plan.0.partition().variables.iter().enumerate() {
                if d == role {
                    let delta = self.correction[i];
                    self.errors[count] = delta.abs();
                    let (value, accepted) = if self.initialization && d {
                        (self.candidate_derivative[i], self.accepted_derivative[i])
                    } else {
                        (self.candidate[i], self.accepted[i])
                    };
                    self.scales[count] = crate::safety::state_difference(value, delta)?
                        .abs()
                        .max(accepted.abs());
                    count += 1;
                }
            }
            let tolerance = if role && self.initialization {
                options.initial_derivative_error()
            } else if role {
                options.differential_error()
            } else {
                options.algebraic_error()
            };
            result = result
                .max(tolerance.normalized_norm(&self.errors[..count], &self.scales[..count])?);
        }
        Ok(scioxide_core::checked_float_product(result, T::Real::TEN)?)
    }

    /// Without a contraction rate a correction within tolerance is accepted only after
    /// [`accept`](NewtonStage::accept) certifies both residual blocks.
    fn floor(&mut self) -> Result<T::Real, Fault> {
        Ok(T::Real::ONE)
    }

    fn apply(&mut self) -> Result<(), Fault> {
        let partition = self.plan.0.partition();
        for (i, &delta) in self.correction.iter().enumerate() {
            let (a, b) = weights(partition, i, self.initialization, self.inverse_step);
            self.candidate[i] = crate::safety::state_increment(self.candidate[i], -a, delta)?;
            self.candidate_derivative[i] =
                crate::safety::state_increment(self.candidate_derivative[i], -b, delta)?;
        }
        Ok(())
    }

    /// A converged pair must also satisfy both original residual blocks; otherwise the
    /// iteration continues from the residual evaluated here.
    fn accept(&mut self) -> Result<bool, Fault> {
        evaluate(
            self.problem,
            self.time,
            self.candidate,
            self.candidate_derivative,
            self.residual,
            self.context,
        )?;
        self.evaluated = true;
        Ok(residual_ok(
            residual_norms(self.plan.0.partition(), self.residual),
            self.plan.1.numerics(),
        )?)
    }
}

struct BoundDerivative<'a, T: NumericScalar, P: DaeSystem<T> + ?Sized> {
    source: core::cell::Cell<Option<scioxide_differentiate::DerivativeCandidate>>,
    problem: &'a P,
    options: &'a DaeNumericalOptions<T::Real>,
    base: &'a [T],
    partition: &'a DaePartition,
    time: T::Real,
    state: &'a [T],
    derivative: &'a [T],
    initialization: bool,
    inverse_step: T::Real,
    scratch: usize,
    preconditioner: Option<&'a dyn Preconditioner<T>>,
}
impl<T: NumericScalar, P: DaeSystem<T> + ?Sized> BoundDerivative<'_, T, P> {
    fn point(&self) -> derivatives::Point<'_, T, P> {
        derivatives::Point {
            problem: self.problem,
            partition: self.partition,
            options: self.options,
            time: self.time,
            state: self.state,
            derivative: self.derivative,
            base: Some(self.base),
        }
    }
    #[allow(clippy::type_complexity)]
    fn directions<'b>(
        &self,
        input: &[T],
        scratch: &'b mut [T],
    ) -> Result<(&'b [T], &'b [T], &'b mut [T]), SciOxideError> {
        let n = self.state.len();
        if input.len() != n || scratch.len() < self.scratch || self.scratch < 6 * n {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "derivative product shape/scratch mismatch",
            ));
        }
        let (y, rest) = scratch.split_at_mut(n);
        let (yd, rest) = rest.split_at_mut(n);
        for (i, &direction) in input.iter().enumerate() {
            let (a, b) = weights(self.partition, i, self.initialization, self.inverse_step);
            y[i] = crate::safety::state_increment(T::ZERO, a, direction)?;
            yd[i] = crate::safety::state_increment(T::ZERO, b, direction)?;
        }
        finite(y, scioxide_core::NonFiniteOrigin::Callback)?;
        finite(yd, scioxide_core::NonFiniteOrigin::Callback)?;
        Ok((y, yd, rest))
    }
}
impl<T: NumericScalar, P: DaeSystem<T> + ?Sized> LinearOperator<T> for BoundDerivative<'_, T, P> {
    fn rows(&self) -> usize {
        self.state.len()
    }
    fn cols(&self) -> usize {
        self.state.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    /// Products evaluate system callbacks, which need the invoking scope.
    fn apply_into(&self, _: &[T], _: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("operator"),
            "matrix-free DAE operators require scoped application",
        ))
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        out: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        if out.len() != self.rows() {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                "derivative product output shape",
            ));
        }
        let (y, yd, rest) = self.directions(input, scratch)?;
        let (provider, differences) = rest.split_at_mut(rest.len() - 4 * self.state.len());
        let mut evaluate = |candidate| match candidate {
            scioxide_differentiate::DerivativeCandidate::Analytic => {
                out.fill(T::from_real(T::Real::NAN));
                let available = self.problem.derivative_product(
                    self.time,
                    self.state,
                    self.derivative,
                    y,
                    yd,
                    out,
                    provider,
                    context,
                )?;
                if available.is_some() {
                    finite(out, scioxide_core::NonFiniteOrigin::Callback)?;
                }
                Ok(available)
            }
            scioxide_differentiate::DerivativeCandidate::AutomaticDifferentiation => Ok(None),
            scioxide_differentiate::DerivativeCandidate::FiniteDifference(scheme) => {
                derivatives::product(
                    self.point(),
                    self.initialization,
                    y,
                    yd,
                    out,
                    differences,
                    scheme,
                    context,
                )?;
                Ok(Some(()))
            }
        };
        if let Some(source) = self.source.get() {
            evaluate(source)?.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("system"),
                    "a selected residual derivative source became unavailable",
                )
            })?;
        } else {
            self.source.set(Some(
                self.options.derivative_strategy().select(&mut evaluate)?.0,
            ));
        }

        Ok(scioxide_core::Invocation::Evaluated(()))
    }
}
impl<T: NumericScalar, P: DaeSystem<T> + ?Sized> EffectiveJacobian<T>
    for BoundDerivative<'_, T, P>
{
    fn dense(
        &mut self,
        out: &mut Matrix<T>,
        other: &mut Matrix<T>,
        differences: &mut derivatives::Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        differences.acquire(
            self.point(),
            derivatives::Output::Dense(out, other),
            context,
        )
    }
    fn assemble_dense(
        &self,
        out: &mut Matrix<T>,
        state: &Matrix<T>,
        derivative: &Matrix<T>,
    ) -> Result<(), SciOxideError> {
        let n = self.state.len();
        for (index, (output, (&y, &yd))) in out
            .as_mut_slice()
            .iter_mut()
            .zip(state.as_slice().iter().zip(derivative.as_slice()))
            .enumerate()
        {
            let row = index / n;
            let col = index % n;
            *output = mixed_entry(
                self.partition,
                row,
                col,
                y,
                yd,
                weights(self.partition, col, self.initialization, self.inverse_step),
            )?;
        }
        finite(out.as_slice(), scioxide_core::NonFiniteOrigin::Callback)
    }
    fn sparse(
        &mut self,
        pattern: StructuralPatternView<'_>,
        out: &mut [T],
        other: &mut [T],
        differences: &mut derivatives::Blocks<T>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), Fault> {
        differences.acquire(
            self.point(),
            derivatives::Output::Sparse(pattern, out, other),
            context,
        )
    }
    fn assemble_sparse(
        &self,
        pattern: StructuralPatternView<'_>,
        out: &mut [T],
        state: &[T],
        derivative: &[T],
    ) -> Result<(), SciOxideError> {
        for row in 0..pattern.rows() {
            for index in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[index];
                out[index] = mixed_entry(
                    self.partition,
                    row,
                    col,
                    state[index],
                    derivative[index],
                    weights(self.partition, col, self.initialization, self.inverse_step),
                )?;
            }
        }
        finite(out, scioxide_core::NonFiniteOrigin::Callback)
    }
    fn prepare_product(
        &self,
        zero: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<T::Real, Fault> {
        admit(context, EvaluationRole::LinearOperator)?;
        self.apply_into_scoped(
            zero,
            output,
            scratch,
            &mut scioxide_core::OperatorContext::new(context),
        )?
        .require()?;
        // Numerical products are inexact linearizations. Use the established
        // one-percent forcing cap; nonlinear correction and original residual
        // tests independently enforce the requested DAE accuracy.
        if matches!(
            self.source.get(),
            Some(scioxide_differentiate::DerivativeCandidate::FiniteDifference(_))
        ) {
            Ok(real::<T::Real>(LINEAR_FORCING)?)
        } else {
            Ok(T::Real::ZERO)
        }
    }
    fn preconditioner(&self) -> Option<&dyn Preconditioner<T>> {
        self.preconditioner
    }
}

/// One original mixed-Jacobian entry; transpose storage conjugates after assembly.
pub(crate) fn mixed_entry<T: NumericScalar>(
    partition: &DaePartition,
    row: usize,
    column: usize,
    state: T,
    derivative: T,
    weights: (T::Real, T::Real),
) -> Result<T, SciOxideError> {
    if (!partition.variables[column] || !partition.equations[row]) && derivative != T::ZERO {
        return Err(invalid(
            scioxide_core::argument!("problem"),
            "derivative block violates declared DAE partition",
        ));
    }
    crate::safety::state_increment(state * T::from_real(weights.0), weights.1, derivative)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
