//! Traits that define an ODE system, its events and replacement stage solvers.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::{Matrix, Vector};

use crate::stepper::StepRequest;

/// Result of [`OdeSystem::rhs_with_step_limit`]: whether the system evaluated the
/// derivative together with a local step bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdeRhsStepLimit<S> {
    /// The system does not provide the combined evaluation. The output is untouched and no
    /// evaluation is counted.
    Unavailable,
    /// The system wrote the complete derivative at the accepted state.
    Evaluated {
        /// Largest allowed step magnitude, positive and finite, or `None` for no limit.
        maximum_step: Option<S>,
    },
}

/// An ODE system `M y′ = f(t, y)` with real time `t` and state `y ∈ Tⁿ`.
///
/// Only [`Self::dimension`] and [`Self::rhs`] are required. The other methods supply
/// optional information: analytic or automatic Jacobians, Jacobian products, a sparse
/// Jacobian, preconditioners, the partial time derivative for Rosenbrock methods, a
/// constant mass matrix `M`, a local step limit and a check that rejects trial states.
/// Each optional method has a default that reports the information as unavailable,
/// and implicit methods then use the next source of their derivative strategy (see
/// [Implicit stages](crate#implicit-stages)).
///
/// The mass matrix is the identity unless [`Self::mass_matrix_revision`] returns
/// `Some`, and it must be nonsingular. Use [`crate::DaeSystem`] for singular mass
/// matrices, residual-form index-1 systems and consistent initialization.
///
/// For a complex state `T = Complex<S>`, implicit methods and every Jacobian source,
/// including finite differences, assume that `f` is holomorphic in `y`, so that `∂f/∂y`
/// is complex-linear. Explicit methods accept any `f`. Write a nonholomorphic system as
/// a real system of twice the dimension to integrate it with an implicit method.
///
/// Every evaluation method receives the [`SolveContext`] of the step that calls it. The
/// integrator counts calls to [`Self::rhs`], the Jacobian methods, [`Self::time_derivative`]
/// and [`Self::mass_matrix`] toward the evaluation limit. Charge any nested work, such
/// as an inner solve, to `cx`. See [Callbacks](crate#callbacks) and
/// [Solve accounting](scioxide_core#solve-accounting).
pub trait OdeSystem<S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// Returns the state dimension `n`. It must not change while a plan advances the
    /// problem.
    fn dimension(&self) -> usize;

    /// Returns an optional upper bound on the step magnitude at the accepted state.
    ///
    /// Use it for a stability limit known to the model, such as a CFL condition. Before
    /// each step, including retries, the integrator takes the smallest of this limit,
    /// the configured maximum step, the step proposed by error control and the remaining
    /// interval. The bound is local: it does not guarantee accuracy or stability over
    /// the interval. The call is not counted as an evaluation; charge any work it does
    /// to `cx`. The default returns `None`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged. A limit that is zero,
    /// negative or not finite makes the integration return `InvalidInput` before any
    /// stage is evaluated.
    fn step_limit(
        &self,
        _time: S,
        _state: &[T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        Ok(None)
    }

    /// Evaluates `f(t, y)` into `derivative` together with the local step bound of
    /// [`Self::step_limit`], when the model computes both in one pass.
    ///
    /// Explicit Runge–Kutta plans use the derivative as their first stage and reuse it
    /// for retries from the same accepted state. Other methods always call
    /// [`Self::step_limit`] and [`Self::rhs`] separately. The implementation must give
    /// the same result for the same `(t, y)`, write every component of `derivative`
    /// when it returns `Evaluated`, and count one `RightHandSide` evaluation on `cx`
    /// before it computes the derivative; the integrator does not count it again. The
    /// default returns `Unavailable`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation, and budget or cancellation errors from `cx`, are
    /// returned unchanged. The integration returns `NonFinite` (origin `Callback`) for a
    /// derivative component that is NaN or infinite, and `InvalidInput` for a bound that
    /// is zero, negative or not finite.
    ///
    /// # References
    ///
    /// E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
    /// Equations I*, 2nd ed., Springer, 1993, §II.1.
    fn rhs_with_step_limit(
        &self,
        _time: S,
        _state: &[T],
        _derivative: &mut [T],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<OdeRhsStepLimit<S>, SciOxideError> {
        Ok(OdeRhsStepLimit::Unavailable)
    }

    /// Evaluates the right-hand side `f(t, y)` into `dydt`.
    ///
    /// `dydt` has length [`Self::dimension`] and unspecified contents on entry. Write
    /// every component with a finite value. The integrator counts each call as one
    /// `RightHandSide` evaluation; charge only nested work to `cx`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged. The integrator checks the
    /// output after every call and returns `DimensionMismatch` if its length changed, or
    /// `NonFinite` (origin `Callback`) for NaN or infinity. Adaptive methods treat a
    /// non-finite value at a trial stage as a rejected step and retry with a smaller
    /// step; a non-finite value at an accepted point is an error.
    fn rhs(
        &self,
        t: S,
        y: &[T],
        dydt: &mut [T],
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;

    /// Returns whether a trial `state` at `time` is physically valid, for example
    /// whether a concentration is nonnegative.
    ///
    /// The integrator calls it before every right-hand-side evaluation at a trial stage
    /// state (not at the accepted state), and once for the endpoint it is about to
    /// accept, after truncation at a terminal event and before any output. Returning
    /// `false` rejects the step independently of the error estimate, also for
    /// fixed-step methods. The integrator then retries from the accepted state with half
    /// the step, and stops with `StepSizeLimit` when the step bounds allow no smaller
    /// step. The call is not counted as an evaluation; charge any work it does to `cx`.
    /// The default returns `true`.
    ///
    /// # Errors
    ///
    /// An error from the implementation ends the integration and is returned unchanged;
    /// it is never retried with a smaller step.
    fn admits(
        &self,
        _time: S,
        _state: &[T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        Ok(true)
    }

    /// Writes the analytic Jacobian `∂f/∂y` at `(t, y)` into the `n × n` matrix `jac`.
    ///
    /// Return `Ok(Some(()))` after writing finite entries, or `Ok(None)` if no analytic
    /// Jacobian is available; implicit methods then try the next source of their
    /// derivative strategy, by default the automatic Jacobian and then finite
    /// differences. The integrator counts each call as one `Jacobian` evaluation. The
    /// default returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged. The integrator returns
    /// `DimensionMismatch` if the matrix shape changed and `NonFinite` (origin
    /// `Callback`) for NaN or infinity.
    fn jacobian(
        &self,
        _t: S,
        _y: &[T],
        _jac: &mut scioxide_linalg::MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Writes `∂f/∂y` at `(t, y)` computed by automatic differentiation into `jac`.
    ///
    /// The contract is that of [`Self::jacobian`]: return `Ok(Some(()))` after writing
    /// finite entries into the `n × n` matrix, or `Ok(None)` if unavailable. The
    /// integrator counts each call as one `Jacobian` evaluation; charge nested
    /// derivative work to `cx`. The default returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation and from `cx` are returned unchanged. The
    /// integrator returns `DimensionMismatch` if the matrix shape changed and
    /// `NonFinite` (origin `Callback`) for NaN or infinity.
    fn automatic_jacobian(
        &self,
        _t: S,
        _y: &[T],
        _jac: &mut scioxide_linalg::MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Returns the analytic Jacobian at `(t, y)` as a linear operator, if available.
    ///
    /// The operator may be a dense matrix, a sparse matrix or a closure. Implicit
    /// methods with the matrix-free backend [`crate::OdeStageBackend::Krylov`] apply it
    /// through the default [`Self::jacobian_product`]. Reserve the operator's scratch
    /// length in the backend configuration. The default returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged.
    fn jacobian_operator(
        &self,
        _t: S,
        _state: &[T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::LinearOperator<T>>, SciOxideError> {
        Ok(None)
    }

    /// Writes the Jacobian-vector product `J v` at `(t, y)` into `output`, where
    /// `J = ∂f/∂y` and `v` is `direction`.
    ///
    /// Return `Ok(Some(()))` after writing the product, or `Ok(None)` if no product is
    /// available, so that the stage solver falls back to finite differences. The call
    /// is not counted separately, because it is part of an operator application the
    /// integrator has already counted; charge further evaluations to `cx`. The default
    /// applies [`Self::jacobian_operator`] with `scratch` as its workspace.
    ///
    /// # Errors
    ///
    /// The default returns the operator's `DimensionMismatch` for wrong slice lengths
    /// or a short `scratch`. Errors from the implementation and from `cx` are returned
    /// unchanged.
    fn jacobian_product(
        &self,
        time: S,
        state: &[T],
        direction: &[T],
        output: &mut [T],
        scratch: &mut [T],
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let Some(operator) = self.jacobian_operator(time, state, cx)? else {
            return Ok(None);
        };
        operator
            .apply_into_scoped(
                direction,
                output,
                scratch,
                &mut scioxide_core::OperatorContext::new(cx),
            )
            .and_then(scioxide_core::Invocation::require)
            .map(Some)
    }

    /// Returns whether `∂f/∂y` is the same at every time and state.
    ///
    /// Return `true` only if `f(t, y) = J y + g(t)` with a constant matrix `J` and a
    /// forcing `g`. Implicit methods then reuse the Jacobian across steps, and with a
    /// direct stage factorization they solve each stage in one Newton correction. The
    /// default returns `false`.
    fn jacobian_is_constant(&self) -> bool {
        false
    }

    /// Writes the values of `∂f/∂y` at `(t, y)` for the positions of `pattern`, in its
    /// compressed sparse row order, into `values`.
    ///
    /// `pattern` is the sparsity pattern configured for the sparse stage backend. It
    /// must contain every structural dependency, including entries that are zero at
    /// this state. Return `Ok(None)` if the values are unavailable. The integrator counts
    /// each call as one `Jacobian` evaluation; charge further work, such as the column
    /// probes of a finite-difference implementation, to `cx`. The default returns
    /// `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation and from `cx` are returned unchanged.
    fn sparse_jacobian(
        &self,
        _time: S,
        _state: &[T],
        _pattern: scioxide_core::StructuralPatternView<'_>,
        _values: &mut [T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Returns a right preconditioner for the stage matrix `I − γ J`, where `γ` is
    /// `shift`, if available.
    ///
    /// Used by matrix-free stages. Prepare and update the preconditioner yourself. The
    /// call is not counted as an evaluation. The default returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged.
    fn stage_preconditioner(
        &self,
        _time: S,
        _state: &[T],
        _shift: S,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<T>>, SciOxideError> {
        Ok(None)
    }

    /// Returns a preconditioner for the coupled stage system of an implicit Runge–Kutta
    /// step, if available.
    ///
    /// `states` holds the `s` stage states one after another, `step` is the step size,
    /// `coefficients` is the `s × s` Runge–Kutta matrix in row-major order and `nodes`
    /// holds the `s` stage nodes. The call is not counted as an evaluation. The default
    /// returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged.
    fn coupled_preconditioner(
        &self,
        _time: S,
        _states: &[T],
        _step: S,
        _coefficients: &[S],
        _nodes: &[S],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<T>>, SciOxideError> {
        Ok(None)
    }

    /// Writes the partial time derivative `∂f/∂t` at `(t, y)` into `output`, for
    /// Rosenbrock methods.
    ///
    /// Return `Ok(None)` if unavailable; the derivative strategy may then use a finite
    /// difference in time. An autonomous system should write zeros and return
    /// `Ok(Some(()))`, which avoids the extra evaluations. The integrator counts each
    /// call as one `Jacobian` evaluation. The default returns `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged.
    fn time_derivative(
        &self,
        _time: S,
        _state: &[T],
        _output: &mut [T],
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Ok(None)
    }

    /// Returns the revision number of a constant mass matrix, or `None` for the
    /// identity.
    ///
    /// While the revision stays the same, [`Self::mass_matrix`] must write the same
    /// matrix. Whether the mass matrix is the identity must not change while a plan
    /// advances the problem, but implicit methods that support mass matrices pick up a
    /// new revision at the next step. The default returns `None`.
    fn mass_matrix_revision(&self) -> Option<u64> {
        None
    }

    /// Writes the constant mass matrix `M` into the `n × n` matrix `mass`.
    ///
    /// Solvers call this method only when [`Self::mass_matrix_revision`] returns
    /// `Some`, and count each call as one `LinearOperator` evaluation. The default
    /// writes the identity.
    ///
    /// # Errors
    ///
    /// The default returns `DimensionMismatch` if `mass` is not `n × n`.
    fn mass_matrix(
        &self,
        mass: &mut scioxide_linalg::MatrixViewMut<'_, T>,
        _cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let dimension = self.dimension();
        if mass.rows() != dimension || mass.cols() != dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(dimension, dimension),
                scioxide_core::Dimensions::two(mass.rows(), mass.cols()),
            ));
        }
        for row in 0..dimension {
            for col in 0..dimension {
                mass.set(row, col, if row == col { T::ONE } else { T::ZERO })?;
            }
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn validate_provider_vector<T: NumericScalar<Real = S>, S>(
    vector: &Vector<T>,
    dimension: usize,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    crate::safety::validate_vector(
        vector.as_slice(),
        dimension,
        scioxide_core::NonFiniteOrigin::Callback,
    )
}

#[inline]
pub(crate) fn validate_provider_matrix<T: NumericScalar>(
    matrix: &Matrix<T>,
    dimension: usize,
) -> Result<(), SciOxideError> {
    crate::safety::validate_matrix(dimension, matrix.view())
}

#[inline]
pub(crate) fn evaluate_rhs_staged<T: NumericScalar<Real = S>, S, System>(
    system: &System,
    time: S,
    state: &Vector<T>,
    derivative: &mut Vector<T>,
    candidate: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    evaluate_rhs_staged_slice(
        system,
        time,
        state.as_slice(),
        derivative,
        candidate,
        request,
        cx,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_rhs_staged_slice<T: NumericScalar<Real = S>, S, System>(
    system: &System,
    time: S,
    state: &[T],
    derivative: &mut Vector<T>,
    candidate: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    crate::safety::validate_finite_time(time, scioxide_core::NonFiniteOrigin::Input)?;
    crate::safety::finite(state, scioxide_core::NonFiniteOrigin::Input)?;
    evaluate_rhs_boundary(
        system, time, state, derivative, candidate, false, request, cx,
    )
}

/// Stage construction has already checked finite state components and time.
#[inline(always)]
pub(crate) fn evaluate_rhs_at_finite_state<T: NumericScalar<Real = S>, S, System>(
    system: &System,
    time: S,
    state: &Vector<T>,
    derivative: &mut Vector<T>,
    candidate: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    evaluate_rhs_boundary(
        system,
        time,
        state.as_slice(),
        derivative,
        candidate,
        false,
        request,
        cx,
    )
}

/// Evaluate a finite accepted base identified by the driver, without state comparison.
#[inline(always)]
pub(crate) fn evaluate_rhs_at_base<T: NumericScalar<Real = S>, S, System>(
    system: &System,
    time: S,
    state: &Vector<T>,
    derivative: &mut Vector<T>,
    candidate: &mut Vector<T>,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    evaluate_rhs_boundary(
        system,
        time,
        state.as_slice(),
        derivative,
        candidate,
        request.base.is_some(),
        request,
        cx,
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn evaluate_rhs_boundary<T: NumericScalar<Real = S>, S, System>(
    system: &System,
    time: S,
    state: &[T],
    derivative: &mut Vector<T>,
    candidate: &mut Vector<T>,
    at_base: bool,
    request: StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    debug_assert_eq!(derivative.len(), state.len());
    debug_assert_eq!(candidate.len(), state.len());
    // The four-component block preserves bounds for scalar component callbacks
    // without querying system metadata. Other shapes share the slice boundary.
    macro_rules! evaluate {
        ($state:expr, $output:expr) => {{
            let output = $output;
            crate::stepper::rhs(system, time, $state, output, at_base, request, cx)?;
            crate::safety::finite(output, scioxide_core::NonFiniteOrigin::Callback)?;
        }};
    }
    if state.len() == 4 {
        evaluate!(&state[..4], &mut candidate.as_mut_slice()[..4]);
    } else {
        evaluate!(state, candidate.as_mut_slice());
    }
    // Publish by exchanging prepared storage. Failed callbacks never touch the
    // committed derivative, and success needs no second pass over the state.
    core::mem::swap(derivative, candidate);
    Ok(())
}

/// An event function `g(t, y)` whose sign changes mark events during integration.
///
/// The detector samples `g` on each accepted step's continuous extension, at
/// the step endpoints and the points of the
/// [`EventSearchPolicy`](crate::EventSearchPolicy). A crossing is a change between
/// consecutive samples, in integration order, from negative to non-negative
/// ([`Rising`](crate::EventOccurrence::Rising)) or from positive to non-positive
/// ([`Falling`](crate::EventOccurrence::Falling)). A sample that is exactly zero starts
/// no crossing: the next nonzero sample sets the sign from which later crossings are
/// classified. A zero at the start of an integration is therefore never an event,
/// whatever the direction filter, so a restart from an event surface proceeds, and a
/// function that stays zero reports nothing. Occurrences are reported in integration
/// order, and integration stops after the first terminal event in that order.
pub trait EventFunction<S: RealScalar, T: NumericScalar<Real = S> = S> {
    /// Evaluates the event function `g(t, y)`.
    ///
    /// # Errors
    ///
    /// Errors from the implementation are returned unchanged. The detector returns
    /// `NonFinite` (origin `Callback`) for a NaN or infinite value.
    fn eval(&self, t: S, y: &[T]) -> Result<S, SciOxideError>;

    /// Returns whether [`Self::total_derivative`] is implemented. The default returns
    /// `false`.
    fn has_total_derivative(&self) -> bool {
        false
    }

    /// Evaluates the total derivative `∂g/∂t + (∂g/∂y) s` along the computed solution,
    /// where `s` is `slope`, the time derivative of the continuous extension at `t`.
    ///
    /// The detector calls it under
    /// [`EventSearchPolicy::DerivativeAssisted`](crate::EventSearchPolicy::DerivativeAssisted);
    /// `slope` comes from the interpolant, not from another right-hand-side call. Plans
    /// reject that policy with `InvalidInput` unless every event returns `true` from
    /// [`Self::has_total_derivative`].
    ///
    /// # Errors
    ///
    /// The default returns `InvalidInput` (`UnsupportedConfiguration`). Errors from an
    /// implementation are returned unchanged, and the detector returns `NonFinite`
    /// (origin `Callback`) for a NaN or infinite value.
    fn total_derivative(&self, _t: S, _state: &[T], _slope: &[T]) -> Result<S, SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "event has no total derivative",
        ))
    }

    /// Returns the crossing direction to detect, measured in integration order.
    fn direction(&self) -> crate::result::EventDirection;

    /// Returns whether integration stops at this event.
    fn is_terminal(&self) -> bool;
}

/// A nonlinear solver that replaces the Newton iteration of implicit Runge–Kutta
/// stages.
///
/// The solver receives the stage residual and the iteration matrix
/// `E = I ⊗ M − h A ⊗ J` of the step, where `A` is the Runge–Kutta matrix, `h` the step
/// size, `M` the mass matrix and `J` the Jacobian. The integrator prepares no stage
/// factorization for it; Radau IIA still factors the real matrix of its error
/// estimate. Returning `Ok` asserts convergence to the requested accuracy: the
/// integrator only checks that the state stays finite, and a report with zero
/// iterations must leave an exactly zero residual.
pub trait NonlinearStageSolver<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
>
{
    /// Solves the stage equations in place, starting from the iterate in `state`.
    ///
    /// On success `state` holds the solution with finite entries. `residual` and
    /// `correction` are scratch vectors of the same length. `residual_fn` evaluates the
    /// stage residual at an iterate and counts its own right-hand-side evaluations;
    /// pass it `cx` or a child context. Apply `operator` with
    /// [`scioxide_core::LinearOperator::apply_into_scoped`] on `cx`, since matrix-free
    /// stage operators support only that form. Call
    /// [`StageAccuracy::admit_iteration`](crate::StageAccuracy::admit_iteration) before
    /// each iteration, and use [`crate::StageAccuracy`] for the convergence test.
    ///
    /// # Errors
    ///
    /// Return an error when the iteration fails, for example `ConvergenceFailure`, or
    /// pass on an error from `residual_fn`, `operator` or `cx`. The integrator returns
    /// the error unchanged and does not retry the step with a smaller step size.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn solve_in_place(
        &self,
        residual_fn: &mut dyn FnMut(
            &[T],
            &mut [T],
            &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError>,
        operator: &dyn scioxide_core::LinearOperator<T>,
        state: &mut [T],
        residual: &mut [T],
        correction: &mut [T],
        accuracy: crate::StageAccuracy<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<NonlinearStageReport, SciOxideError>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct PartialRhsFailure;

    impl OdeSystem<f64> for PartialRhsFailure {
        fn dimension(&self) -> usize {
            2
        }

        fn rhs(
            &self,
            _time: f64,
            _state: &[f64],
            derivative: &mut [f64],
            _cx: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            derivative[0] = 99.0;
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("derivative"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "deliberate failure after a partial write",
            ))
        }
    }

    #[test]
    fn staged_rhs_failure_preserves_the_caller_buffer() {
        let state = Vector::from_vec(vec![1.0, 2.0]);
        let mut derivative = Vector::from_vec(vec![7.0, 8.0]);
        let mut candidate = Vector::zeros(2).unwrap();
        let trace = crate::stepper::StepTrace::default();
        let result = crate::stepper::testing::with_context(|cx| {
            evaluate_rhs_staged(
                &PartialRhsFailure,
                0.0,
                &state,
                &mut derivative,
                &mut candidate,
                StepRequest::new(&trace),
                cx,
            )
        });
        assert!(matches!(
            (result).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(derivative.as_slice(), &[7.0, 8.0]);
    }
}

/// Work done by one [`NonlinearStageSolver::solve_in_place`] call, added to the
/// integrator's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct NonlinearStageReport {
    iterations: usize,
    linear_solves: usize,
    factorizations: usize,
}

impl NonlinearStageReport {
    /// Creates a report from the numbers of iterations, linear solves and
    /// factorizations.
    pub const fn new(iterations: usize, linear_solves: usize, factorizations: usize) -> Self {
        Self {
            iterations,
            linear_solves,
            factorizations,
        }
    }

    /// Returns the number of nonlinear iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }

    /// Returns the number of linear solves.
    pub const fn linear_solves(self) -> usize {
        self.linear_solves
    }

    /// Returns the number of matrix factorizations.
    pub const fn factorizations(self) -> usize {
        self.factorizations
    }
}

/// A linear solver that replaces the factorization in the Newton iteration of
/// implicit stages.
///
/// The solver receives the iteration matrix as a matrix-free operator. The
/// integrator prepares no dense factorization for it; Radau IIA still factors the
/// real matrix of its error estimate. Prepare any storage before integration
/// starts; `operator` needs `apply_workspace_len()` scratch values per application.
/// `Ok(())` asserts that the solve converged; report an unsuccessful inner solve as an
/// error. The integrator checks each correction with one operator application and
/// rejects it if the residual exceeds the stage's linear tolerance.
pub trait StageLinearSolver<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S>
{
    /// Solves `E c = r` for the correction `c`, where `E` is `operator` and `r` is
    /// `residual`, and writes `c` into `correction`.
    ///
    /// `accuracy` gives the component scales and counts work on `cx`. Count every
    /// inner iteration and every operator or preconditioner application before doing
    /// it, and apply `operator` with
    /// [`scioxide_core::LinearOperator::apply_into_scoped`] on `cx` or a child context,
    /// since matrix-free stage operators support only that form.
    ///
    /// # Errors
    ///
    /// Return an error for a failed solve, such as `ConvergenceFailure` or `Singular`,
    /// or pass on an error from `operator` or `cx`. The integrator returns it
    /// unchanged.
    fn solve_into(
        &self,
        operator: &dyn scioxide_core::LinearOperator<T>,
        state: &[T],
        residual: &[T],
        correction: &mut [T],
        accuracy: crate::StageAccuracy<'_, S, T>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
}

pub(crate) fn validate_stage_buffers<T>(
    state: &[T],
    residual: &[T],
    dimension: usize,
) -> Result<(), SciOxideError> {
    if state.len() != dimension || residual.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(dimension, dimension),
            scioxide_core::Dimensions::two(state.len(), residual.len()),
        ));
    }
    Ok(())
}
