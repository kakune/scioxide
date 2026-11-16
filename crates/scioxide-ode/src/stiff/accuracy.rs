//! Component-scaled inner accuracy, shared by implicit method families.
//!
//! [`StageAccuracy::correction_norm`] embeds the Newton stopping constant: one equals a
//! correction of ten percent of the requested component scale.
use crate::stepper::{StepRequest, StepTrace};
use scioxide_core::{ErrorTolerance, EvaluationRole, NumericScalar, RealScalar, SciOxideError};

/// Measures stage corrections against the integration tolerance, for injected stage solvers.
///
/// An injected [`StageLinearSolver`](crate::StageLinearSolver) or
/// [`NonlinearStageSolver`](crate::NonlinearStageSolver) receives a value of this
/// type. It holds the state at the start of the step, the step's
/// [`ErrorTolerance`] and the iteration limit of one stage solve, and it counts work
/// in the solve's budgets.
///
/// For component `i`, the scale is `sᵢ = atolᵢ + rtol·max(|xᵢ|, |yᵢ|)`, where `x`
/// is the stage state and `y` the state at the start of the step. Corrections are
/// in state units. For coupled stages, whose vectors are several state vectors
/// placed one after another, each stage is normalized separately and the largest
/// norm is used.
#[derive(Clone, Copy, Debug)]
pub struct StageAccuracy<'a, S: RealScalar, T: NumericScalar<Real = S> = S> {
    previous: &'a [T],
    /// `None` selects the core default component policy.
    tolerance: Option<&'a ErrorTolerance<S>>,
    limit: usize,
    /// Step classification flags; absent for standalone accuracy policies.
    trace: Option<&'a StepTrace>,
}

impl<'a, S: RealScalar, T: NumericScalar<Real = S>> StageAccuracy<'a, S, T> {
    /// Borrow the accepted state, the requested component policy and the maximum
    /// number of inner iterations of one stage solve.
    #[cfg(test)]
    pub(crate) const fn new(
        previous: &'a [T],
        tolerance: &'a ErrorTolerance<S>,
        limit: usize,
    ) -> Self {
        Self {
            previous,
            tolerance: Some(tolerance),
            limit,
            trace: None,
        }
    }

    /// Accuracy of a step attempt: its tolerance, inner limit and classification flags.
    pub(crate) const fn from_request(previous: &'a [T], request: StepRequest<'a, S, T>) -> Self {
        Self {
            previous,
            tolerance: request.error_tolerance,
            limit: request.inner_limit.get(),
            trace: Some(request.trace),
        }
    }

    /// Returns the norm of `correction` relative to ten percent of the tolerance.
    ///
    /// The value is `10·‖(cᵢ/sᵢ)‖`, with the norm of the step's [`ErrorTolerance`],
    /// so a value of at most 1 means the correction is within a tenth of the
    /// tolerance. This is the stopping test of the native Newton iteration.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `state` and `correction` differ in length, or their
    ///   length is not a multiple of the state dimension.
    /// - `NonFinite` (`Input`) if `state` or the step's starting state contains NaN or
    ///   infinity.
    /// - `NonFinite` if `correction` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a nonzero correction meets a zero
    ///   scale, or the norm overflows.
    #[inline(always)]
    pub fn correction_norm(&self, state: &[T], correction: &[T]) -> Result<S, SciOxideError> {
        let n = self.previous.len();
        if n == 0 && state.is_empty() && correction.is_empty() {
            return Ok(S::ZERO);
        }
        if n == 0
            || state.len() != correction.len()
            || (state.len() != n && !state.len().is_multiple_of(n))
        {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(state.len()),
                scioxide_core::Dimensions::one(correction.len()),
            ));
        }
        crate::safety::finite(self.previous, scioxide_core::NonFiniteOrigin::Input)?;
        crate::safety::finite(state, scioxide_core::NonFiniteOrigin::Input)?;
        self.correction_norm_at_finite_state(state, correction)
    }

    /// Internal callers have already checked state shapes and finite values.
    #[inline(always)]
    pub(crate) fn correction_norm_at_finite_state(
        &self,
        state: &[T],
        correction: &[T],
    ) -> Result<S, SciOxideError> {
        self.scaled_norm(state, |index, _| correction[index].abs())
    }

    /// [`correction_norm`](Self::correction_norm) of a correction at the
    /// [`rounding_level`](super::newton::rounding_level) of every component scale of
    /// the prepared, finite `state`.
    pub(crate) fn rounding_norm(&self, state: &[T]) -> Result<S, SciOxideError> {
        self.scaled_norm(state, |_, scale| super::newton::rounding_level(scale))
    }

    /// Maximum stage norm, times ten, of `error(index, scale)` against the component
    /// scale `max(|stateᵢ|, |previousᵢ|)`.
    #[inline(always)]
    fn scaled_norm(&self, state: &[T], error: impl Fn(usize, S) -> S) -> Result<S, SciOxideError> {
        let n = self.previous.len();
        if n == 0 {
            return Ok(S::ZERO);
        }
        let default = ErrorTolerance::default();
        let tolerance = self.tolerance.unwrap_or(&default);
        let mut worst = S::ZERO;
        let stages = if state.len() == n { 1 } else { state.len() / n };
        for stage_index in 0..stages {
            let offset = stage_index * n;
            let stage = &state[offset..offset + n];
            let norm = tolerance.normalized_norm_by(n, |i| {
                let current = stage[i].abs();
                let old = self.previous[i].abs();
                let scale = if old > current { old } else { current };
                (error(offset + i, scale), scale)
            })?;
            if norm > worst {
                worst = norm;
            }
        }
        scioxide_core::checked_float_product(worst, S::TEN)
    }

    /// A strict component bound certifies ordinary infinity/RMS residuals;
    /// borderline values and other norms use the configured norm.
    pub(crate) fn residual_satisfied(
        &self,
        state: &[T],
        residual: &[T],
    ) -> Result<bool, SciOxideError> {
        let default = ErrorTolerance::default();
        let tolerance = self.tolerance.unwrap_or(&default);
        let n = self.previous.len();
        if n != 0
            && state.len() == n
            && residual.len() == n
            && matches!(
                tolerance.norm(),
                scioxide_core::NormSelection::Infinity
                    | scioxide_core::NormSelection::RootMeanSquare
            )
        {
            // Half of the usual ten-percent inner budget leaves rounding
            // headroom and implies both infinity and RMS acceptance.
            let fraction = S::ONE / (S::TEN + S::TEN);
            let within = state
                .iter()
                .zip(residual)
                .zip(self.previous)
                .enumerate()
                .all(|(i, ((&current, &error), &old))| {
                    let Ok(absolute) = tolerance.absolute().component(i, n) else {
                        return false;
                    };
                    let scale = current.abs().max(old.abs());
                    let threshold = (absolute + tolerance.relative() * scale) * fraction;
                    current.has_finite_components()
                        && old.has_finite_components()
                        && threshold.is_finite()
                        && error.abs() <= threshold
                });
            if within {
                return Ok(true);
            }
        }
        self.correction_norm(state, residual)
            .map(|norm| norm <= S::ONE)
    }
    /// Returns the tolerance scale `sᵢ = atolᵢ + rtol·max(|yᵢ|, |candidate|)` of
    /// component `index`, in state units.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `index` is not below the state dimension.
    /// - `NonFinite` (`Input`) if `candidate` or the starting component is NaN or
    ///   infinite.
    /// - `NumericalInstability` (`Overflow`) if the scale overflows.
    pub fn component_scale(self, index: usize, candidate: T) -> Result<S, SciOxideError> {
        let n = self.previous.len();
        if index >= n {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("implicit_component"),
                index,
                n,
            ));
        }
        if !candidate.has_finite_components() || !self.previous[index].has_finite_components() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "state must be finite",
            ));
        }
        let default = ErrorTolerance::default();
        self.tolerance.unwrap_or(&default).threshold(
            index,
            n,
            self.previous[index].abs().max(candidate.abs()),
        )
    }
    /// Number of state components; coupled stage vectors repeat this length.
    pub(crate) const fn dimension(self) -> usize {
        self.previous.len()
    }
    pub(crate) fn mark_provider_failure(self) {
        if let Some(trace) = self.trace {
            trace.provider_failed.set(true);
        }
    }
    /// Returns the maximum number of iterations of one stage solve.
    pub const fn iteration_limit(self) -> usize {
        self.limit
    }
    /// Counts one nonlinear iteration in `cx`; call it before each iteration.
    ///
    /// It first checks cancellation and the deadline of `cx`, then counts the
    /// iteration toward the nested iteration limit.
    ///
    /// # Errors
    ///
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    /// - `ConvergenceFailure` if a limit of `cx` or an enclosing solve is exhausted.
    pub fn admit_iteration(
        self,
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(crate::termination::admit(
            cx,
            crate::termination::Work::NestedIteration,
        )?)
    }
    /// Counts one evaluation with `role` in `cx`; call it before each operator,
    /// preconditioner or right-hand-side evaluation.
    ///
    /// It first checks cancellation and the deadline of `cx`, then counts the
    /// evaluation toward the evaluation limit.
    ///
    /// # Errors
    ///
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    /// - `ConvergenceFailure` if a limit of `cx` or an enclosing solve is exhausted.
    pub fn admit_provider(
        self,
        role: EvaluationRole,
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        match self.trace {
            Some(trace) => crate::stepper::admit_provider(cx, role, trace),
            None => crate::stepper::admit_provider(cx, role, &StepTrace::default()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{Complex, NormSelection};

    fn residual_boundaries<S: RealScalar + NumericScalar<Real = S>>() {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let previous = [S::ONE; 2];
        for norm in [
            NormSelection::Infinity,
            NormSelection::RootMeanSquare,
            NormSelection::L1,
            NormSelection::L2,
        ] {
            let tolerance = ErrorTolerance::scalar(scalar(1.25), S::ZERO, norm).unwrap();
            let accuracy = StageAccuracy::new(&previous, &tolerance, 20);
            assert!(
                accuracy
                    .residual_satisfied(&previous, &[scalar(0.001); 2])
                    .unwrap()
            );
            let boundary = scalar(0.125);
            assert_eq!(
                accuracy
                    .residual_satisfied(&previous, &[boundary; 2])
                    .unwrap(),
                matches!(
                    norm,
                    NormSelection::Infinity | NormSelection::RootMeanSquare
                )
            );
            assert!(
                !accuracy
                    .residual_satisfied(&previous, &[boundary.next_after(S::MAX); 2])
                    .unwrap()
            );
            assert!(
                accuracy
                    .residual_satisfied(&[S::NAN, S::ONE], &[S::ZERO; 2])
                    .is_err()
            );
            assert!(
                accuracy
                    .residual_satisfied(&previous, &[S::NAN, S::ZERO])
                    .is_err()
            );
        }
        let tolerance =
            ErrorTolerance::scalar(S::ZERO, S::ZERO, NormSelection::RootMeanSquare).unwrap();
        let accuracy = StageAccuracy::new(&previous, &tolerance, 20);
        assert!(
            accuracy
                .residual_satisfied(&previous, &[S::ZERO; 2])
                .unwrap()
        );
        assert!(
            accuracy
                .residual_satisfied(&previous, &[S::ONE, S::ZERO])
                .is_err()
        );
    }

    #[test]
    fn residual_boundaries_f32() {
        residual_boundaries::<f32>();
    }
    #[test]
    fn residual_boundaries_f64() {
        residual_boundaries::<f64>();
    }
    #[test]
    fn imaginary_errors_use_modulus_and_zero_tolerance_is_exact() {
        let old = [Complex::new(3_f64, 4.)];
        let tol = ErrorTolerance::scalar(0., 0.2, NormSelection::Infinity).unwrap();
        let accuracy = StageAccuracy::new(&old, &tol, 20);
        assert_eq!(accuracy.component_scale(0, old[0]).unwrap(), 1.);
        assert_eq!(
            accuracy
                .correction_norm(&old, &[Complex::new(0., 0.2)])
                .unwrap(),
            2.
        );
        let exact = ErrorTolerance::scalar(0., 0., NormSelection::Infinity).unwrap();
        let accuracy = StageAccuracy::new(&old, &exact, 20);
        assert_eq!(
            accuracy
                .correction_norm(&old, &[Complex::new(0., 0.)])
                .unwrap(),
            0.
        );
        assert!(
            accuracy
                .correction_norm(&old, &[Complex::new(0., f64::MIN_POSITIVE)])
                .is_err()
        );
        assert!(
            accuracy
                .correction_norm(&old, &[Complex::new(0., f64::NAN)])
                .is_err()
        );
    }
    #[test]
    fn component_norm_does_not_hide_small_components_or_dilute_coupled_stages() {
        let previous = [1.0e12, 1.0e-6];
        let policy =
            ErrorTolerance::try_vector(&[1.0, 1.0e-12], 0.0, NormSelection::Infinity).unwrap();
        let accuracy = StageAccuracy::new(&previous, &policy, 20);
        assert!(
            accuracy
                .correction_norm(&previous, &[0.0, 1.0e-10])
                .unwrap()
                > 1.0
        );
        let coupled = [1.0e12, 1.0e-6, 1.0e12, 1.0e-6];
        assert_eq!(
            accuracy
                .correction_norm(&coupled, &[0.0, 1.0e-10, 0.0, 0.0])
                .unwrap(),
            accuracy
                .correction_norm(&previous, &[0.0, 1.0e-10])
                .unwrap()
        );
    }
    #[test]
    fn rms_normalization_preserves_range_and_rejects_nonfinite_states() {
        let policy = ErrorTolerance::scalar(1.0, 0.0, NormSelection::RootMeanSquare).unwrap();
        let previous = [0.0; 400];
        let accuracy = StageAccuracy::new(&previous, &policy, 20);
        let norm = accuracy
            .correction_norm(&previous, &[f64::MAX / 20.0; 400])
            .unwrap();
        assert!((norm / (f64::MAX / 2.0) - 1.0).abs() < 4.0 * f64::EPSILON);
        assert!(matches!(
            (accuracy.correction_norm(&[f64::NAN; 400], &[0.0; 400]))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    #[test]
    fn empty_accuracy_has_zero_norm() {
        let default = ErrorTolerance::default();
        let empty = StageAccuracy::<f64>::new(&[], &default, 20);
        assert_eq!(empty.correction_norm(&[], &[]).unwrap(), 0.0);
    }
}
