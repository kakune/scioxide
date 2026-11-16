//! Step-size controllers for adaptive ODE methods.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

/// Built-in policies with concrete dispatch and per-epoch history.
pub(crate) enum ManagedController<S: RealScalar + NumericScalar<Real = S>> {
    DefaultIntegral(PiController<S>),
    ProportionalIntegral(PiController<S>),
    Radau(RadauController<S>),
}
impl<S: RealScalar + NumericScalar<Real = S>> ManagedController<S> {
    pub(crate) fn reset(&mut self) {
        match self {
            Self::DefaultIntegral(c) | Self::ProportionalIntegral(c) => c.reset(),
            Self::Radau(c) => c.previous = None,
        }
    }
    #[inline(always)]
    pub(crate) fn compute_step(
        &mut self,
        error: S,
        step: S,
        order: usize,
        nonlinear_iterations: usize,
    ) -> Result<(S, bool), SciOxideError> {
        match self {
            Self::DefaultIntegral(c) => {
                let half = S::ONE / (S::ONE + S::ONE);
                let upper = S::ONE - half * half;
                if error > half && error <= upper {
                    Ok((step, true))
                } else {
                    c.compute_step(error, step, order)
                }
            }
            Self::ProportionalIntegral(c) if c.parameters.beta == S::ZERO => {
                c.compute_step(error, step, order)
            }
            Self::ProportionalIntegral(c) => pi_step(c, error, step, order),
            Self::Radau(c) => c.compute_step(error, step, nonlinear_iterations),
        }
    }
}

/// The predictive (Gustafsson) controller of Hairer and Wanner's Radau IIA code,
/// Hairer & Wanner II, §IV.8. History contains accepted steps only. The safety factor
/// 0.9 and the bounds `h_new/h ∈ [0.2, 8]` follow that code; 7 is the Newton
/// iteration limit of managed Radau steps.
#[derive(Debug)]
pub(crate) struct RadauController<S> {
    previous: Option<(S, S)>,
}

impl<S> Default for RadauController<S> {
    fn default() -> Self {
        Self { previous: None }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> RadauController<S> {
    fn compute_step(
        &mut self,
        error: S,
        step: S,
        iterations: usize,
    ) -> Result<(S, bool), SciOxideError> {
        let safe = S::try_from_f64_lossy(0.9)?;
        let minimum = S::try_from_f64_lossy(0.2)?;
        let maximum = S::try_from_f64_lossy(8.0)?;
        let safety = safe.min(
            safe * S::try_from_f64_lossy(15.0)?
                / (S::try_from_f64_lossy(14.0)? + S::try_from_usize_exact(iterations)?),
        );
        let accepted = error <= S::ONE;
        let mut factor = if error == S::ZERO {
            maximum
        } else {
            (safety / error.sqrt().sqrt()).max(minimum).min(maximum)
        };
        if accepted {
            if error > S::ZERO
                && let Some((old_step, old_error)) = self.previous
            {
                let predictive =
                    safe * (step / old_step).abs() * old_error.sqrt().sqrt() / error.sqrt();
                factor = factor.min(predictive.max(minimum).min(maximum));
            }
            // Avoid refactoring for marginal growth, as in RADAU5's QUOT1/QUOT2.
            if factor >= S::ONE && factor <= S::try_from_f64_lossy(1.2)? {
                factor = S::ONE;
            }
            self.previous = Some((step, error.max(S::try_from_f64_lossy(0.01)?)));
        }
        Ok((bounded_scale_step(step, factor), accepted))
    }
}

// Keep two transcendental calls out of the integration loop's live registers.
#[inline(never)]
fn pi_step<S: RealScalar + NumericScalar<Real = S>>(
    controller: &mut PiController<S>,
    error: S,
    step: S,
    order: usize,
) -> Result<(S, bool), SciOxideError> {
    controller.compute_step(error, step, order)
}

/// Parameters of proportional-integral (PI) step-size control.
///
/// After a step with normalized local error `r` (the step is accepted when `r ≤ 1`),
/// the next step size is
///
/// ```text
/// h_new = h · clamp(s · r^(−α/k) · r_prev^(β/k), f_min, f_max)
/// ```
///
/// where `h` is the current step, `s` is `safety`, `α` and `β` are the exponents
/// `alpha` and `beta`, `k` is the power of `h` in the local error estimate,
/// `r_prev` is the error of the previous accepted step (one at the start), and
/// `f_min`, `f_max` are `min_factor`, `max_factor`. An error of exactly zero takes
/// the factor `f_max` and resets `r_prev` to one. Rejected steps do not change
/// `r_prev`. With `α = 1` and `β = 0` this is the classical integral (I)
/// controller.
///
/// The value holds parameters only; every integration keeps its own `r_prev`. Use it
/// through [`crate::OdeStepControl::ProportionalIntegral`].
///
/// # Examples
///
/// ```
/// use scioxide_ode::{OdeStepControl, ProportionalIntegralController};
///
/// // Integral control: α = 1, β = 0.
/// let parameters = ProportionalIntegralController::new(0.9_f64, 5., 0.2, 1., 0.)?;
/// assert!((parameters.safety() - 0.9).abs() < f64::EPSILON);
/// let control = OdeStepControl::ProportionalIntegral(parameters);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - K. Gustafsson, "Control theoretic techniques for stepsize selection in explicit
///   Runge–Kutta methods", *ACM Trans. Math. Softw.* 17(4), 533–554, 1991.
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 2nd ed.,
///   Springer, 1996, §IV.2.
#[derive(Debug, Clone, PartialEq)]
pub struct ProportionalIntegralController<S: RealScalar> {
    safety: S,
    max_factor: S,
    min_factor: S,
    alpha: S,
    beta: S,
}

impl<S: RealScalar> ProportionalIntegralController<S> {
    /// Creates PI parameters from the safety factor, the bounds on the step ratio
    /// and the exponents `alpha` and `beta`.
    ///
    /// [`Default`] gives `safety = 0.9`, `max_factor = 5`, `min_factor = 0.2`,
    /// `alpha = 0.7` and `beta = 0.4`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (`Input`) when a parameter is not finite.
    /// - `InvalidInput` (`OutOfRange`) when `safety` lies outside `(0, 1]`, a factor
    ///   bound is not positive or a gain is negative.
    /// - `InvalidInput` (`Inconsistent`) unless `min_factor ≤ 1 ≤ max_factor`.
    pub fn new(
        safety: S,
        max_factor: S,
        min_factor: S,
        alpha: S,
        beta: S,
    ) -> Result<Self, SciOxideError> {
        validate_controller_parameters(
            safety,
            max_factor,
            min_factor,
            &[alpha, beta],
            &[
                scioxide_core::argument!("safety"),
                scioxide_core::argument!("max_factor"),
                scioxide_core::argument!("min_factor"),
                scioxide_core::argument!("alpha"),
                scioxide_core::argument!("beta"),
            ],
        )?;
        Ok(Self {
            safety,
            max_factor,
            min_factor,
            alpha,
            beta,
        })
    }

    /// Returns the safety factor `s`.
    #[must_use]
    pub const fn safety(&self) -> S {
        self.safety
    }

    /// Returns the largest step ratio `f_max`.
    #[must_use]
    pub const fn max_factor(&self) -> S {
        self.max_factor
    }

    /// Returns the smallest step ratio `f_min`.
    #[must_use]
    pub const fn min_factor(&self) -> S {
        self.min_factor
    }

    /// Returns the exponent `α` of the current error.
    #[must_use]
    pub const fn alpha(&self) -> S {
        self.alpha
    }

    /// Returns the exponent `β` of the previous accepted error.
    #[must_use]
    pub const fn beta(&self) -> S {
        self.beta
    }
}

impl<S: RealScalar> Default for ProportionalIntegralController<S> {
    fn default() -> Self {
        let one: S = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let five = two + two + one;
        let ten = five + five;
        Self {
            safety: (ten - one) / ten,
            max_factor: five,
            min_factor: one / five,
            alpha: (ten - (one + one + one)) / ten,
            beta: two / five,
        }
    }
}

impl<S: RealScalar> ProportionalIntegralController<S> {
    pub(crate) fn integral_defaults() -> Self {
        Self {
            alpha: S::ONE,
            beta: S::ZERO,
            ..Self::default()
        }
    }
}

/// Drive-local PI history and estimator powers; configuration remains immutable.
pub(crate) struct PiController<S: RealScalar> {
    parameters: ProportionalIntegralController<S>,
    previous_error: S,
    powers: Option<(usize, S, S)>,
}

impl<S: RealScalar + NumericScalar<Real = S>> PiController<S> {
    pub(crate) fn new(parameters: ProportionalIntegralController<S>) -> Self {
        Self {
            parameters,
            previous_error: S::ONE,
            powers: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.previous_error = S::ONE;
        self.powers = None;
    }

    #[inline(always)]
    pub(crate) fn compute_step(
        &mut self,
        error: S,
        step: S,
        order: usize,
    ) -> Result<(S, bool), SciOxideError> {
        let p = &self.parameters;
        if error == S::ZERO {
            self.previous_error = S::ONE;
            return Ok((bounded_scale_step(step, p.max_factor), true));
        }
        let (alpha, beta) = match self.powers {
            Some((previous_order, alpha, beta)) if previous_order == order => (alpha, beta),
            _ => {
                let order_scalar = S::try_from_usize_exact(order)?;
                let alpha = -p.alpha / order_scalar;
                let beta = p.beta / order_scalar;
                self.powers = Some((order, alpha, beta));
                (alpha, beta)
            }
        };
        let current_factor = p.safety * error.powf(alpha);
        let raw_factor = if p.beta == S::ZERO {
            current_factor
        } else {
            current_factor * self.previous_error.powf(beta)
        };
        let factor = if raw_factor.is_finite() {
            raw_factor.min(p.max_factor).max(p.min_factor)
        } else {
            p.max_factor
        };
        let accepted = error <= S::ONE;
        if accepted {
            self.previous_error = error;
        }
        Ok((bounded_scale_step(step, factor), accepted))
    }
}

pub(crate) fn bounded_scale_step<S: RealScalar>(step: S, factor: S) -> S {
    let magnitude = step.abs();
    let maximum = S::MAX;
    let scaled = if magnitude > maximum / factor {
        maximum
    } else {
        // Keep built-in retries positive even when multiplication underflows.
        // The driver detects that the smallest retry cannot reduce the endpoint.
        (magnitude * factor).max(S::ZERO.next_after(S::ONE))
    };
    if step < S::ZERO { -scaled } else { scaled }
}

/// Validate controller factors and gains. `arguments` names
/// `safety`, `max_factor`, `min_factor` and then each gain, in order.
fn validate_controller_parameters<S: RealScalar>(
    safety: S,
    max_factor: S,
    min_factor: S,
    gains: &[S],
    arguments: &[scioxide_core::Argument],
) -> Result<(), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    if !safety.is_finite()
        || !max_factor.is_finite()
        || !min_factor.is_finite()
        || gains.iter().any(|gain| !gain.is_finite())
    {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "controller parameters must be finite",
        ));
    }
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    if safety <= zero || safety > one || max_factor <= zero || min_factor <= zero {
        let argument = if safety <= zero || safety > one {
            arguments[0]
        } else if max_factor <= zero {
            arguments[1]
        } else {
            arguments[2]
        };
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "controller factors must be positive and safety must not exceed one",
        ));
    }
    if let Some(index) = gains.iter().position(|gain| *gain < zero) {
        return Err(SciOxideError::invalid_input(
            arguments[3 + index],
            scioxide_core::InvalidInputReason::OutOfRange,
            "controller gains must be nonnegative",
        ));
    }
    if min_factor > one || max_factor < one || min_factor > max_factor {
        return Err(SciOxideError::invalid_input(
            if max_factor < one {
                arguments[1]
            } else {
                arguments[2]
            },
            scioxide_core::InvalidInputReason::Inconsistent,
            "controller factors must satisfy min_factor <= 1 <= max_factor",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn radau_safety_prediction_and_rejected_history_follow_radau5() {
        let mut controller = RadauController::default();
        let (step, accepted) = controller.compute_step(1.0_f64, 0.1, 7).unwrap();
        assert!(accepted);
        assert!((step - 0.1 * 0.9 * 15.0 / 21.0).abs() < 1e-15);
        controller.previous = Some((0.1, 0.25));
        let old = controller.previous;
        let (_, accepted) = controller.compute_step(16.0, 0.05, 3).unwrap();
        assert!(!accepted);
        assert_eq!(controller.previous, old);
        let (step, accepted) = controller.compute_step(0.16, 0.05, 2).unwrap();
        assert!(accepted);
        let predicted = 0.05 * 0.9 * 0.5 * 0.25_f64.sqrt().sqrt() / 0.16_f64.sqrt();
        assert!((step - predicted).abs() < 1e-15);
        let mut managed = ManagedController::Radau(controller);
        managed.reset();
        let ManagedController::Radau(controller) = managed else {
            unreachable!()
        };
        assert!(controller.previous.is_none());
    }

    #[test]
    fn radau_zero_error_and_extreme_errors_keep_direction_and_bounds() {
        for step in [-0.1, 0.1] {
            let mut controller = RadauController::default();
            assert_eq!(
                controller.compute_step(0.0_f64, step, 1).unwrap(),
                (step * 8.0, true)
            );
            let (reduced, accepted) = controller.compute_step(f64::MAX, step, 7).unwrap();
            assert!(!accepted);
            assert!((reduced / step - 0.2).abs() < 1e-15);
        }
    }

    #[test]
    fn controller_constructors_validate_parameters() {
        assert!(ProportionalIntegralController::new(0.9, 5.0, 0.2, 0.7, 0.4).is_ok());
        assert!(ProportionalIntegralController::new(0.9, 5.0, 0.2, 1.0, 0.0).is_ok());
        assert!(ProportionalIntegralController::new(1.1, 5.0, 0.2, 1.0, 0.0).is_err());
        assert!(matches!(
            (ProportionalIntegralController::new(0.9, 0.1, 0.2, 1.0, 0.0))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (ProportionalIntegralController::new(0.9, 5.0, 0.2, f64::NAN, 0.4))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn controller_accessors_expose_validated_configuration() {
        let controller = ProportionalIntegralController::new(0.9, 5.0, 0.2, 0.7, 0.4).unwrap();
        assert_eq!(controller.safety(), 0.9);
        assert_eq!(controller.max_factor(), 5.0);
        assert_eq!(controller.min_factor(), 0.2);
        assert_eq!(controller.alpha(), 0.7);
        assert_eq!(controller.beta(), 0.4);
    }

    #[test]
    fn exact_zero_error_uses_the_configured_maximum_growth() {
        let mut integral = PiController::new(
            ProportionalIntegralController::new(0.9, 5.0, 0.2, 1.0, 0.0).unwrap(),
        );
        let mut pi = PiController::new(
            ProportionalIntegralController::new(0.9, 4.0, 0.2, 0.7, 0.4).unwrap(),
        );

        assert_eq!(integral.compute_step(0.0, 0.25, 5).unwrap(), (1.25, true));
        assert_eq!(pi.compute_step(0.0, -0.25, 5).unwrap(), (-1.0, true));
    }

    #[test]
    fn default_integral_retains_only_accepted_steps_in_the_error_band() {
        let mut controller = ManagedController::DefaultIntegral(PiController::new(
            ProportionalIntegralController::integral_defaults(),
        ));
        for step in [-0.1_f64, 0.1] {
            for error in [0.625, 0.75] {
                assert_eq!(
                    controller.compute_step(error, step, 5, 0).unwrap(),
                    (step, true)
                );
            }
            let (reduced, accepted) = controller.compute_step(0.9, step, 5, 0).unwrap();
            assert!(accepted);
            assert!(reduced.abs() < step.abs());
            let (smaller, accepted) = controller.compute_step(1.01, step, 5, 0).unwrap();
            assert!(!accepted);
            assert!(smaller.abs() < step.abs());
            let (larger, accepted) = controller.compute_step(0.25, step, 5, 0).unwrap();
            assert!(accepted);
            assert!(larger.abs() > step.abs());
        }
    }

    #[test]
    fn integral_order_changes_refresh_the_error_power() {
        let mut controller = PiController::new(ProportionalIntegralController::integral_defaults());
        for order in [5, 8, 5, 2, 8] {
            for step in [-0.1, 0.1] {
                let (next, accepted) = controller.compute_step(0.25_f64, step, order).unwrap();
                let exponent = -1.0 / f64::try_from_usize_exact(order).unwrap();
                let expected = step * 0.9 * 0.25_f64.powf(exponent);
                assert!(accepted);
                assert!((next - expected).abs() < 8.0 * f64::EPSILON);
            }
        }
    }

    #[test]
    fn pi_history_uses_accepted_errors_and_resets_without_changing_parameters() {
        fn check<S: RealScalar + NumericScalar<Real = S>>() {
            let r = |value| S::try_from_f64_lossy(value).unwrap();
            let parameters =
                ProportionalIntegralController::new(r(0.9), r(5.), r(0.2), r(0.7), r(0.4)).unwrap();
            let mut controller = PiController::new(parameters.clone());
            for step in [r(-0.1), r(0.1)] {
                controller.reset();
                for order in [5, 2, 8] {
                    let old_error = controller.previous_error;
                    let (next, accepted) = controller.compute_step(r(0.25), step, order).unwrap();
                    let order = S::try_from_usize_exact(order).unwrap();
                    let expected = step
                        * r(0.9)
                        * r(0.25).powf(-r(0.7) / order)
                        * old_error.powf(r(0.4) / order);
                    assert!(accepted);
                    assert!((next - expected).abs() <= r(8.) * S::EPSILON);
                    let (_, accepted) = controller.compute_step(r(4.), step, 5).unwrap();
                    assert!(!accepted);
                    assert_eq!(controller.previous_error, r(0.25));
                }
                controller.compute_step(S::ZERO, step, 5).unwrap();
                assert_eq!(controller.previous_error, S::ONE);
                controller.compute_step(r(0.5), step, 5).unwrap();
                controller.reset();
                assert_eq!(controller.previous_error, S::ONE);
                assert!(controller.powers.is_none());
                assert_eq!(controller.parameters, parameters);
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
