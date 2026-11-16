//! Values stored as a phase and the logarithm of the magnitude.

use crate::Scaled;
use crate::ScaledValue;
use crate::{
    InvalidInputReason, NonFiniteOrigin, NumericScalar, NumericalIssue, RealScalar, SciOxideError,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum LogState<V: NumericScalar, R> {
    Zero(V),
    Nonzero {
        phase: ScaledValue<V>,
        log_magnitude: R,
    },
}

/// A zero, or a unit phase `p` and a finite logarithm `ℓ` representing `p · e^ℓ`.
///
/// For real values the phase is `±1`; for complex values it has `|p| = 1`. The
/// phase stores each component with its own exponent, so a tiny nonzero
/// component is not lost. Zero is a separate state ([`Self::ZERO`]) without an
/// infinite logarithm, which is how log-sum-exp functions such as
/// [`weighted_logsumexp`](crate::weighted_logsumexp) represent an empty or
/// cancelled sum.
///
/// Converting a finite scalar with [`Self::from_value`] always succeeds but
/// rounds the logarithm, unlike the exact conversion to
/// [`ScaledValue`](crate::ScaledValue). Converting back evaluates `p · e^ℓ` and
/// checks the range of each component.
///
/// # Examples
///
/// ```
/// use core::f64::consts::FRAC_1_SQRT_2;
/// use scioxide_core::{Complex64, LogValue};
///
/// // |z| = e^(ln(f64::MAX) + 0.2) overflows f64, but each component
/// // |z| / √2 = 0.86 f64::MAX is finite.
/// let phase = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
/// let value = LogValue::from_log(phase, f64::MAX.ln() + 0.2)?;
/// assert!(value.try_value()?.is_finite());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LogValue<V: NumericScalar, R: RealScalar> {
    state: LogState<V, R>,
}

impl<V: NumericScalar<Real = R>, R: RealScalar> LogValue<V, R> {
    /// Zero, with `+0` components.
    pub const ZERO: Self = Self {
        state: LogState::Zero(V::ZERO),
    };

    /// Returns `phase · e^(log_magnitude)` for a unit `phase`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `phase` or `log_magnitude` is not finite.
    /// - `InvalidInput` if a real `phase` is not `±1`, or `||phase| − 1| > 16 ε` for
    ///   a complex phase, where `ε` is the machine epsilon.
    pub fn from_log(phase: V, log_magnitude: R) -> Result<Self, SciOxideError> {
        Self::from_scaled_log(ScaledValue::from_value(phase)?, log_magnitude)
    }

    /// Returns `phase · e^(log_magnitude)` for a unit `phase` stored as a `ScaledValue`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `log_magnitude` is not finite.
    /// - `InvalidInput` if a real `phase` is not `±1`, or `||phase| − 1| > 16 ε` for
    ///   a complex phase, where `ε` is the machine epsilon.
    /// - `NumericalInstability` if computing `|phase|` leaves the exponent range.
    pub fn from_scaled_log(phase: ScaledValue<V>, log_magnitude: R) -> Result<Self, SciOxideError> {
        if !log_magnitude.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "log magnitude must be finite",
            ));
        }
        let invalid_phase = || {
            SciOxideError::invalid_input(
                crate::argument!("phase"),
                InvalidInputReason::OutOfRange,
                "real phase must be -1 or 1; complex phase must have unit magnitude",
            )
        };
        if V::from_components(R::ZERO, R::ONE)
            .imaginary_component()
            .is_zero()
            && phase.real().abs() != Scaled::ONE
        {
            return Err(invalid_phase());
        }
        let largest_component = Scaled::from_value(R::ONE + R::EPSILON.scalbn(4))?;
        if !phase.real().abs_le(largest_component) || !phase.imaginary().abs_le(largest_component) {
            return Err(invalid_phase());
        }
        let magnitude = phase.magnitude()?.round()?;
        if (magnitude - R::ONE).abs() > R::EPSILON.scalbn(4) {
            return Err(SciOxideError::invalid_input(
                crate::argument!("phase"),
                InvalidInputReason::OutOfRange,
                "phase must have unit magnitude to rounding accuracy",
            ));
        }
        Ok(Self {
            state: LogState::Nonzero {
                phase,
                log_magnitude,
            },
        })
    }

    /// Converts a finite scalar, keeping signed zeros and tiny phase components.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `value` is NaN or infinite.
    pub fn from_value(value: V) -> Result<Self, SciOxideError> {
        let scaled = ScaledValue::from_value(value)?;
        if scaled.is_zero() {
            return Ok(Self {
                state: LogState::Zero(value),
            });
        }
        // The magnitude can exceed the IEEE maximum; it remains scaled while
        // each component is divided independently, so a tiny phase never flushes.
        let magnitude = scaled.magnitude()?;
        let phase = ScaledValue::from_components(
            scaled.real().div(magnitude)?,
            scaled.imaginary().div(magnitude)?,
        );
        let log_magnitude =
            crate::complex::log_hypot(value.real_component(), value.imaginary_component());
        if !log_magnitude.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "finite nonzero components must produce a finite logarithm",
            ));
        }
        // Finite scalar decoding bounds all phase exponents by 2098. Hypot's
        // four-ULP envelope plus correctly rounded component division bounds
        // the computed unit phase within the constructor's 16-epsilon window.
        // Zero and nonfinite inputs were handled before these operations, and
        // the logarithm was checked above. Do not recompute the phase norm.
        Ok(Self {
            state: LogState::Nonzero {
                phase,
                log_magnitude,
            },
        })
    }

    /// Returns the phase `p`, or the signed zero components for zero.
    pub fn scaled_phase(self) -> ScaledValue<V> {
        match self.state {
            LogState::Zero(value) => ScaledValue::from_components(
                Scaled::from_normalized_parts(value.real_component(), 0),
                Scaled::from_normalized_parts(value.imaginary_component(), 0),
            ),
            LogState::Nonzero { phase, .. } => phase,
        }
    }
    /// Returns the logarithm `ℓ` of the magnitude, or `None` for zero.
    pub const fn log_magnitude(self) -> Option<R> {
        match self.state {
            LogState::Zero(_) => None,
            LogState::Nonzero { log_magnitude, .. } => Some(log_magnitude),
        }
    }
    /// Converts to `V`, allowing gradual underflow.
    ///
    /// The magnitude `e^ℓ` is never formed on its own; each component is scaled once,
    /// so a component may be finite even when `|p · e^ℓ|` is not.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a rounded component is infinite.
    pub fn round(self) -> Result<V, SciOxideError> {
        let (phase, log_magnitude) = match self.state {
            LogState::Zero(value) => return Ok(value),
            LogState::Nonzero {
                phase,
                log_magnitude,
            } => (phase, log_magnitude),
        };
        // A unit phase has at least one component >= 1/sqrt(2). These guards
        // classify logs beyond the extended exponent domain without converting
        // them to integers or losing zero-component signs.
        let two = R::TWO;
        if log_magnitude > R::MAX.ln() + two {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "at least one phase component exceeds the IEEE finite range",
            ));
        }
        let minimum = R::ZERO.next_after(R::ONE);
        if log_magnitude < minimum.ln() - two {
            return Ok(V::from_components(
                R::ZERO.copysign(phase.real().significand()),
                R::ZERO.copysign(phase.imaginary().significand()),
            ));
        }
        let magnitude = Scaled::from_log(R::ONE, log_magnitude)?;
        Ok(V::from_components(
            phase.real().mul_round(magnitude)?,
            phase.imaginary().mul_round(magnitude)?,
        ))
    }
    /// Converts to `V`, rejecting a nonzero component that would round to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a rounded component is infinite.
    /// - `NumericalInstability` (`Underflow`) if a nonzero component rounds to zero.
    pub fn try_value(self) -> Result<V, SciOxideError> {
        self.scaled_phase().check_nonzero_components(self.round()?)
    }
}

impl<S: RealScalar> LogValue<S, S> {
    /// Returns the sign of a nonzero real value, or the signed zero for zero.
    pub fn sign(self) -> S {
        self.scaled_phase().real().significand()
    }
}
