//! Shared numerical safeguards for Krylov solvers.

use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    EvaluationRole, NonFiniteOrigin, NumericScalar, NumericalIssue, RealScalar, SciOxideError,
};

use scioxide_core::{AdjointOperator, LinearOperator};

/// CG and BiCGSTAB replace recursively updated residuals at this cadence.
// Periodic replacement bounds the drift between the recurrence and the true
// residual at the cost of one extra operator application per 50 iterations.
pub(super) const TRUE_RESIDUAL_RECOMPUTE_INTERVAL: usize = 50;

/// Unit roundoffs per operand in the rounding floor of a computed residual,
/// for real scalars; complex multiply–add rounds about twice as much.
const ROUNDING_FLOOR_UNITS: usize = 16;

/// Scratch of the probe product in [`ResidualFloor::reached`].
pub(super) struct FloorProbe<'a, S> {
    /// Receives the signed copy `s ∘ x`.
    pub(super) input: &'a mut [S],
    /// Receives `A (s ∘ x)`.
    pub(super) image: &'a mut [S],
    /// Operator scratch.
    pub(super) workspace: &'a mut [S],
}

/// Whether a verification compares the best residual with the rounding floor:
/// its residual does not improve on the best verified residual before it.
///
/// A residual that still decreases is never judged, so a pessimistic floor
/// cannot end a solve that is making progress.
pub(super) fn floor_test_due<R: RealScalar>(residual: R, best_before: R) -> bool {
    residual >= best_before
}

/// Rounding floor `ε c (‖b‖₂ + ‖A (s ∘ x)‖₂)` of the true residual
/// `‖b − A x‖₂` at the best verified iterate `x`, with `c = 16` for real and
/// `c = 32` for complex scalars.
///
/// Rounding `x` to working precision changes its residual by up to
/// `ε ‖ |A| |x| ‖₂`, and forming `b − A x` adds as much again. The fixed sign
/// pattern `sᵢ = ±1` keeps the cancellation between the products `aᵢⱼ xⱼ`,
/// which makes `‖A x‖₂` small near a solution, out of the estimate:
/// `‖A (s ∘ x)‖₂` measures `‖ |A| |x| ‖₂` with one more product with `A`,
/// counted toward the evaluation limit. The floor is measured at the best
/// verified iterate, not at the latest one: after a transient growth of the
/// recursive residuals the latest iterate can be far from the solution, and
/// the verified residual that replaces the recursive one at each checkpoint
/// removes the rounding errors accumulated before it (Sleijpen and
/// van der Vorst). A complex product rounds up to `√2 γ₂` relative to its
/// magnitude (Higham, §3.6), hence the larger `c`. The floor never changes the
/// convergence tolerance.
///
/// # References
///
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, §3.6.
/// - A. Greenbaum, "Estimating the attainable accuracy of recursively computed
///   residual methods", *SIAM J. Matrix Anal. Appl.* 18(3), 1997.
/// - G. L. G. Sleijpen and H. A. van der Vorst, "Reliable updated residuals in
///   hybrid Bi-CG methods", *Computing* 56(2), 1996.
#[derive(Debug, Clone, Copy)]
pub(super) struct ResidualFloor<R: RealScalar> {
    rhs_norm: R,
}

impl<R: RealScalar> ResidualFloor<R> {
    pub(super) const fn new(rhs_norm: R) -> Self {
        Self { rhs_norm }
    }

    /// Whether `best`, the smallest verified residual, lies within the floor at
    /// `solution`, the best verified iterate, or `None` when the evaluation
    /// limit denies the probe product.
    pub(super) fn reached<S: NumericScalar<Real = R>>(
        &self,
        operator: &dyn LinearOperator<S>,
        solution: &[S],
        best: R,
        probe: FloorProbe<'_, S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<Option<bool>, SciOxideError> {
        for (index, (out, &value)) in probe.input.iter_mut().zip(solution).enumerate() {
            *out = if flipped(index) { -value } else { value };
        }
        let maximum = crate::kernel::finite::maximum(probe.input, kernels.dispatch())?;
        let image = if maximum == R::ZERO {
            R::ZERO
        } else {
            if apply_operator_scoped(
                operator,
                probe.input,
                maximum,
                probe.image,
                probe.workspace,
                context,
            )?
            .is_exhausted()
            {
                return Ok(None);
            }
            crate::kernel::finite::norm(probe.image)?
        };
        let units = scioxide_core::Scaled::from_value(crate::checked::rounded_count::<R>(
            if S::IS_REAL {
                ROUNDING_FLOOR_UNITS
            } else {
                2 * ROUNDING_FLOOR_UNITS
            },
        )?)?;
        let floor = scioxide_core::Scaled::from_value(self.rhs_norm)?
            .add(scioxide_core::Scaled::from_value(image)?)?
            .mul(units)?
            .mul(scioxide_core::Scaled::from_value(R::EPSILON)?)?;
        Ok(Some(scioxide_core::Scaled::from_value(best)?.abs_le(floor)))
    }
}

/// Sign of the floor probe at `index`: the top bit of its Fibonacci hash.
fn flipped(index: usize) -> bool {
    (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63 == 1
}

/// Rounding model for Hermitian forms `xᴴ fl(B x)` of a Hermitian `B`.
///
/// Rounding in `fl(B x)` and in the reduction perturbs each component of the
/// form by about `dimension_epsilon(n) ‖B‖ ‖x‖²`, independent of the form's
/// magnitude: a positive definite form near `λ_min ‖x‖²` carries an
/// imaginary part of order `eps κ` relative to itself. `‖B‖` is modelled by
/// the running maximum of `‖B x‖ / ‖x‖` over the applications seen so far,
/// including the current one. CG and MINRES judge `pᴴAp`, `vᴴAv` and
/// `rᴴM⁻¹r` with this model and continue with the real part.
#[derive(Debug)]
pub(super) struct HermitianForms<R: RealScalar> {
    dimension: usize,
    operator: scioxide_core::Scaled<R>,
}

impl<R: RealScalar> HermitianForms<R> {
    pub(super) fn new(dimension: usize) -> Self {
        Self {
            dimension,
            operator: scioxide_core::Scaled::ZERO,
        }
    }

    /// Real part of `value = xᴴ fl(B x)` with `‖x‖ = input` and
    /// `‖fl(B x)‖ = image`, or `None` when the imaginary part exceeds the
    /// rounding bound, i.e. `B` is not Hermitian.
    pub(super) fn real_part<S: NumericScalar<Real = R>>(
        &mut self,
        value: scioxide_core::ScaledValue<S>,
        input: R,
        image: R,
    ) -> Result<Option<scioxide_core::Scaled<R>>, SciOxideError> {
        let input = scioxide_core::Scaled::from_value(input)?;
        if !input.is_zero() {
            let estimate = scioxide_core::Scaled::from_value(image)?.div(input)?;
            if self.operator.abs_le(estimate) {
                self.operator = estimate;
            }
        }
        let bound = self
            .operator
            .mul(input.square()?)?
            .mul(scioxide_core::Scaled::from_value(
                crate::checked::dimension_epsilon(self.dimension)?,
            )?)?;
        Ok(value.imaginary().abs_le(bound).then(|| value.real()))
    }

    /// Real part of a Hermitian form that is positive: real parts at most
    /// `dimension_epsilon(n) ‖x‖ ‖B x‖` are indistinguishable from zero, as
    /// for real scalars. `None` reports a non-Hermitian or non-positive form.
    pub(super) fn positive_part<S: NumericScalar<Real = R>>(
        &mut self,
        value: scioxide_core::ScaledValue<S>,
        input: R,
        image: R,
    ) -> Result<Option<scioxide_core::Scaled<R>>, SciOxideError> {
        let Some(real) = self.real_part(value, input, image)? else {
            return Ok(None);
        };
        let threshold = scioxide_core::Scaled::from_value(input)?
            .mul(scioxide_core::Scaled::from_value(image)?)?
            .mul(scioxide_core::Scaled::from_value(
                crate::checked::dimension_epsilon(self.dimension)?,
            )?)?;
        Ok((!real.is_sign_negative() && !real.abs_le(threshold)).then_some(real))
    }
}

/// Failure of one recurrence step.
///
/// Recurrence breakdown is a typed signal raised only by the recurrence guards
/// below, so that a failure of a provider or nested SciOxide call propagates
/// unchanged whatever its kind, including `NumericalInstability(Breakdown)`.
#[derive(Debug)]
pub(super) enum StepFault {
    /// A recurrence denominator, shadow product or normalization is unusable.
    Breakdown,
    /// Any other failure.
    Error(SciOxideError),
}

impl From<SciOxideError> for StepFault {
    fn from(error: SciOxideError) -> Self {
        Self::Error(error)
    }
}

/// Result of a recurrence step.
pub(super) type Step<T> = Result<T, StepFault>;

/// Classify the result of a crate-internal normalization of a recurrence
/// vector: the kernel reports a zero or unusable divisor as breakdown, which is
/// recurrence breakdown here. Never applied to provider results.
pub(super) fn normalized(result: Result<(), SciOxideError>) -> Step<()> {
    result.map_err(|error| match error.view() {
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        } => StepFault::Breakdown,
        _ => StepFault::Error(error),
    })
}

fn invariant(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Breakdown, detail)
}

/// Test an inner product against `||left||₂ ||right||₂`. Dividing by the
/// larger norm first avoids overflow in the scale product.
///
pub(super) fn inner_product_is_small<S: NumericScalar>(
    value: S,
    left_norm: S::Real,
    right_norm: S::Real,
    dimension: usize,
) -> Result<bool, SciOxideError> {
    if !value.has_finite_components() || !left_norm.is_finite() || !right_norm.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "inner product and operand norms must be finite",
        ));
    }
    if left_norm < S::Real::ZERO || right_norm < S::Real::ZERO {
        return Err(invariant("inner-product operand norms must be nonnegative"));
    }
    if left_norm == S::Real::ZERO || right_norm == S::Real::ZERO {
        return Ok(true);
    }

    let larger = left_norm.max(right_norm);
    let smaller = left_norm.min(right_norm);
    let first = value.abs() / larger;
    if !first.is_finite() {
        return Ok(false);
    }
    let normalized = first / smaller;
    if !normalized.is_finite() {
        return Ok(false);
    }
    Ok(normalized <= crate::checked::dimension_epsilon(dimension)?)
}

/// Shadow products may lose relative accuracy without breaking the recurrence.
/// Only magnitudes at most `epsilon² ||left||₂ ||right||₂` are unusable.
/// Keep the comparison in extended range, including for complex magnitudes.
pub(super) fn shadow_product_is_small<S: NumericScalar>(
    value: scioxide_core::ScaledValue<S>,
    left_norm: S::Real,
    right_norm: S::Real,
) -> Result<bool, SciOxideError> {
    let threshold = shadow_threshold::<S::Real>(left_norm, right_norm)?;
    Ok(value.magnitude()?.abs_le(threshold))
}

/// The epsilon-squared guard for a value known only within `error`:
/// `Some(small)` when every magnitude within `error` of `|value|` lies on one
/// side of the threshold, `None` when that interval contains it.
pub(super) fn certified_shadow_product_is_small<S: NumericScalar>(
    value: scioxide_core::ScaledValue<S>,
    error: scioxide_core::Scaled<S::Real>,
    left_norm: S::Real,
    right_norm: S::Real,
) -> Result<Option<bool>, SciOxideError> {
    let threshold = shadow_threshold::<S::Real>(left_norm, right_norm)?;
    let magnitude = value.magnitude()?;
    Ok(if magnitude.add(error)?.abs_le(threshold) {
        Some(true)
    } else if magnitude.abs_le(threshold.add(error)?) {
        None
    } else {
        Some(false)
    })
}

fn shadow_threshold<R: RealScalar>(
    left_norm: R,
    right_norm: R,
) -> Result<scioxide_core::Scaled<R>, SciOxideError> {
    if !left_norm.is_finite() || !right_norm.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "shadow-product operand norms must be finite",
        ));
    }
    if left_norm < R::ZERO || right_norm < R::ZERO {
        return Err(invariant(
            "shadow-product operand norms must be nonnegative",
        ));
    }
    scioxide_core::Scaled::from_value(R::EPSILON * R::EPSILON)?
        .mul(scioxide_core::Scaled::from_value(left_norm)?)?
        .mul(scioxide_core::Scaled::from_value(right_norm)?)
}

/// Divide two finite recurrence scalars. A zero denominator or a non-finite
/// quotient is recurrence breakdown.
pub(super) fn checked_divide<S: NumericScalar>(numerator: S, denominator: S) -> Step<S> {
    if !numerator.has_finite_components() || !denominator.has_finite_components() {
        return Err(StepFault::Error(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "recurrence division inputs must be finite",
        )));
    }
    if denominator == S::ZERO {
        return Err(StepFault::Breakdown);
    }
    let quotient = numerator / denominator;
    if !quotient.has_finite_components() {
        return Err(StepFault::Breakdown);
    }
    Ok(quotient)
}

/// Count one operator evaluation and apply the operator within the caller's scope.
pub(super) fn apply_operator_scoped<S: NumericScalar>(
    operator: &dyn LinearOperator<S>,
    input: &[S],
    maximum: S::Real,
    output: &mut [S],
    workspace: &mut [S],
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
    match context.record_provider(EvaluationRole::LinearOperator)? {
        scioxide_core::Admission::Granted(()) => operator
            .apply_finite_into_scoped(input, maximum, output, workspace, context)
            .and_then(|outcome| match outcome {
                scioxide_core::Invocation::Evaluated(()) => {
                    crate::checked::provider_output(output)?;
                    Ok(outcome)
                }
                scioxide_core::Invocation::ExecutionStopped(reason) => {
                    Err(SciOxideError::execution_stopped(reason))
                }
                stopped => Ok(stopped),
            }),
        scioxide_core::Admission::Denied(report) => {
            Ok(scioxide_core::Invocation::BudgetExhausted(report))
        }
    }
}

/// Count one operator evaluation and apply the conjugate adjoint within the caller's scope.
pub(super) fn apply_adjoint_scoped<S: NumericScalar>(
    operator: &dyn AdjointOperator<S>,
    input: &[S],
    maximum: S::Real,
    output: &mut [S],
    workspace: &mut [S],
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
    match context.record_provider(EvaluationRole::LinearOperator)? {
        scioxide_core::Admission::Granted(()) => operator
            .apply_adjoint_finite_into_scoped(input, maximum, output, workspace, context)
            .and_then(|outcome| match outcome {
                scioxide_core::Invocation::Evaluated(()) => {
                    crate::checked::provider_output(output)?;
                    Ok(outcome)
                }
                scioxide_core::Invocation::ExecutionStopped(reason) => {
                    Err(SciOxideError::execution_stopped(reason))
                }
                stopped => Ok(stopped),
            }),
        scioxide_core::Admission::Denied(report) => {
            Ok(scioxide_core::Invocation::BudgetExhausted(report))
        }
    }
}

/// Whether a recursively maintained residual must be replaced this iteration.
pub(super) const fn should_recompute_true_residual(iteration: usize) -> bool {
    iteration.is_multiple_of(TRUE_RESIDUAL_RECOMPUTE_INTERVAL)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use crate::Vector;
    pub(crate) fn true_residual<S: NumericScalar>(
        operator: &dyn LinearOperator<S>,
        b: &Vector<S>,
        x: &Vector<S>,
    ) -> Result<(Vector<S>, S::Real), SciOxideError> {
        let mut ax = Vector::zeros(operator.rows())?;
        let workspace_len = operator.apply_workspace_len()?;
        let mut workspace = scioxide_core::backend::try_filled(
            workspace_len,
            S::ZERO,
            scioxide_core::quantity!("true_residual_operator_workspace"),
        )?;
        operator.apply_into(x.as_slice(), ax.as_mut_slice(), &mut workspace)?;
        let residual = b.view().try_sub(&ax)?;
        let norm = residual.view().norm2_scoped(
            &crate::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok((residual, norm))
    }

    use super::*;
    use scioxide_core::{Scaled, ScaledValue};

    fn scaled(re: f64, im: f64) -> ScaledValue<scioxide_core::Complex<f64>> {
        ScaledValue::from_value(scioxide_core::Complex::new(re, im)).unwrap()
    }

    #[test]
    fn hermitian_forms_bound_imaginary_parts_by_the_operator_not_the_form() {
        let bound = 16. * f64::EPSILON;
        let mut forms = HermitianForms::new(16);
        // A large image fixes the running `‖B‖` model at 1.
        assert!(forms.real_part(scaled(1., 0.), 1., 1.).unwrap().is_some());
        // A tiny form `λ_min ‖x‖²` keeps an imaginary part at the operator bound.
        let real = forms
            .real_part(scaled(1e-12, 0.5 * bound), 1., 1e-12)
            .unwrap()
            .unwrap();
        assert_eq!(real.round().unwrap(), 1e-12);
        assert!(
            forms
                .positive_part(scaled(1e-12, 0.5 * bound), 1., 1e-12)
                .unwrap()
                .is_some()
        );
        assert!(
            forms
                .real_part(scaled(1., 2. * bound), 1., 1.)
                .unwrap()
                .is_none()
        );
        assert!(forms.real_part(scaled(1., 0.1), 1., 1.).unwrap().is_none());
    }

    #[test]
    fn hermitian_forms_are_scale_invariant() {
        for scale in [1e-150, 1., 1e150] {
            let mut forms = HermitianForms::new(4);
            let image = 2. * scale;
            let value = ScaledValue::from_value(scioxide_core::Complex::new(scale, 0.))
                .unwrap()
                .mul_real(Scaled::from_value(scale).unwrap())
                .unwrap();
            let real = forms.positive_part(value, scale, image).unwrap().unwrap();
            assert_eq!(real.round().unwrap(), scale * scale);
        }
    }

    #[test]
    fn hermitian_form_positivity_uses_the_real_part_boundary() {
        let threshold = 8. * f64::EPSILON;
        let mut forms = HermitianForms::<f64>::new(8);
        assert!(
            forms
                .positive_part(ScaledValue::from_value(threshold).unwrap(), 1., 1.)
                .unwrap()
                .is_none()
        );
        assert!(
            forms
                .positive_part(ScaledValue::from_value(2. * threshold).unwrap(), 1., 1.)
                .unwrap()
                .is_some()
        );
        assert!(
            forms
                .positive_part(ScaledValue::from_value(-1.).unwrap(), 1., 1.)
                .unwrap()
                .is_none()
        );
        assert!(
            forms
                .positive_part(ScaledValue::from_value(0.).unwrap(), 0., 0.)
                .unwrap()
                .is_none()
        );
    }

    macro_rules! shadow_threshold {
        ($name:ident, $real:ty, $scalar:ty, $phase:expr, $exponent:expr) => {
            #[test]
            fn $name() {
                let epsilon_squared = <$real>::EPSILON * <$real>::EPSILON;
                let phase: $scalar = $phase;
                for left in [
                    (2.0 as $real).powi(-$exponent),
                    1.,
                    (2.0 as $real).powi($exponent),
                ] {
                    for right in [
                        (2.0 as $real).powi(-$exponent),
                        1.,
                        (2.0 as $real).powi($exponent),
                    ] {
                        for (normalized, small) in [
                            (0., true),
                            (epsilon_squared / 2., true),
                            (epsilon_squared, true),
                            (2. * epsilon_squared, false),
                            (<$real>::EPSILON, false),
                            (1., false),
                        ] {
                            let value =
                                ScaledValue::from_value(phase * <$scalar>::from_real(normalized))
                                    .unwrap()
                                    .mul_real(Scaled::from_value(left).unwrap())
                                    .unwrap()
                                    .mul_real(Scaled::from_value(right).unwrap())
                                    .unwrap();
                            assert_eq!(shadow_product_is_small(value, left, right).unwrap(), small);
                        }
                    }
                }
                let zero = ScaledValue::<$scalar>::ZERO;
                assert!(shadow_product_is_small(zero, 0., 1.).unwrap());
                assert!(shadow_product_is_small(zero, 1., 0.).unwrap());
                assert!(shadow_product_is_small(zero, -1., 1.).is_err());
                assert!(shadow_product_is_small(zero, <$real>::NAN, 1.).is_err());
                assert!(shadow_product_is_small(zero, 1., <$real>::INFINITY).is_err());
            }
        };
    }
    shadow_threshold!(shadow_threshold_f32, f32, f32, 1., 100);
    shadow_threshold!(shadow_threshold_f64, f64, f64, 1., 900);
    shadow_threshold!(
        shadow_threshold_complex32,
        f32,
        scioxide_core::Complex<f32>,
        scioxide_core::Complex::new(0., -1.),
        100
    );
    shadow_threshold!(
        shadow_threshold_complex64,
        f64,
        scioxide_core::Complex<f64>,
        scioxide_core::Complex::new(0., -1.),
        900
    );

    #[test]
    fn certified_shadow_guard_decides_only_outside_the_error_interval() {
        let epsilon_squared = f64::EPSILON * f64::EPSILON;
        let scaled = |x: f64| Scaled::from_value(x).unwrap();
        for (left, right) in [
            (1., 1.),
            (2_f64.powi(-900), 2_f64.powi(900)),
            (1e100, 1e-300),
        ] {
            let threshold = epsilon_squared * left * right;
            for (magnitude, error, decision) in [
                (0., 0.5, Some(true)),
                (0.25, 0.5, Some(true)),
                (4., 1., Some(false)),
                (1., 0.5, None),
                (1.25, 0.5, None),
                (0.75, 0.5, None),
            ] {
                let value = ScaledValue::from_value(-magnitude)
                    .unwrap()
                    .mul_real(scaled(epsilon_squared))
                    .unwrap()
                    .mul_real(scaled(left))
                    .unwrap()
                    .mul_real(scaled(right))
                    .unwrap();
                let error = scaled(error * epsilon_squared)
                    .mul(scaled(left))
                    .unwrap()
                    .mul(scaled(right))
                    .unwrap();
                assert_eq!(
                    certified_shadow_product_is_small(value, error, left, right).unwrap(),
                    decision,
                    "{magnitude} ± {error:?} against {threshold}"
                );
            }
        }
        let zero = ScaledValue::<f64>::ZERO;
        assert!(certified_shadow_product_is_small(zero, Scaled::ZERO, -1., 1.).is_err());
        assert!(certified_shadow_product_is_small(zero, Scaled::ZERO, 1., f64::NAN).is_err());
    }

    #[test]
    fn inner_product_threshold_is_scale_invariant_for_f64() {
        assert!(!inner_product_is_small(1.0e-300_f64, 1.0e-100, 1.0e-200, 1).unwrap());
        assert!(inner_product_is_small(1.0e-320_f64, 1.0e-100, 1.0e-200, 1).unwrap());
    }

    #[test]
    fn inner_product_threshold_is_scale_invariant_for_f32() {
        assert!(!inner_product_is_small(1.0e-20_f32, 1.0e-10, 1.0e-10, 1).unwrap());
        assert!(inner_product_is_small(1.0e-28_f32, 1.0e-10, 1.0e-10, 1).unwrap());
    }

    #[test]
    fn true_residual_cadence_is_explicit() {
        assert!(!should_recompute_true_residual(49));
        assert!(should_recompute_true_residual(50));
        assert!(should_recompute_true_residual(100));
    }

    #[test]
    fn large_binary32_orders_do_not_classify_collinear_products_as_breakdown() {
        for n in [1 << 23, (1 << 24) + 1, usize::MAX] {
            assert!(!inner_product_is_small(2.0_f32, 1., 2., n).unwrap());
        }
    }
}
