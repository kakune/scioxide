//! Real and complex values whose components each have an extended binary exponent.

use super::real::Wide;
use crate::{
    AlgebraicScalar, InvalidInputReason, NonFiniteOrigin, NumericScalar, NumericalIssue,
    ProjectiveValue, RealScalar, Scaled, SciOxideError,
};
use core::ops::Neg;

/// A real or complex number whose components each have the form `m · 2ᵉ`.
///
/// Each component has a significand with `1 ≤ |m| < 2` and its own `i64`
/// exponent, so the real and imaginary parts can differ in scale by more than the
/// `f64` range. Signed zeros keep their sign and have exponent zero. A real value
/// has imaginary part `+0`. Cancellation within an operation is resolved before
/// the exponent range is checked. A complex quotient has a relative error of at
/// most `γ₃ = 3u / (1 − 3u)` in each component before conversion, where
/// `u = ε/2` is the unit roundoff.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex64, ScaledValue};
///
/// // The components are the largest finite f64 and the smallest subnormal.
/// let original = Complex64::new(f64::MAX, f64::from_bits(1));
/// let value = ScaledValue::from_value(original)?;
/// let decoded = value.try_value()?;
/// assert_eq!(decoded.re.to_bits(), original.re.to_bits());
/// assert_eq!(decoded.im.to_bits(), original.im.to_bits());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledValue<V: NumericScalar> {
    re: Scaled<V::Real>,
    im: Scaled<V::Real>,
    value_type: core::marker::PhantomData<V>,
}
impl<V: NumericScalar> core::ops::Neg for ScaledValue<V> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_components(-self.re, -self.im)
    }
}
impl<V: NumericScalar> ScaledValue<V> {
    /// Zero.
    pub const ZERO: Self = Self {
        re: Scaled::ZERO,
        im: Scaled::ZERO,
        value_type: core::marker::PhantomData,
    };
    /// One.
    pub const ONE: Self = Self {
        re: Scaled::ONE,
        im: Scaled::ZERO,
        value_type: core::marker::PhantomData,
    };
    /// One half.
    pub const HALF: Self = Self {
        re: Scaled::HALF,
        im: Scaled::ZERO,
        value_type: core::marker::PhantomData,
    };
    /// Builds a value from its real and imaginary components.
    ///
    /// As with [`NumericScalar::from_components`](crate::NumericScalar::from_components),
    /// a real `V` ignores `im` and stores `+0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Complex64, Scaled, ScaledValue};
    ///
    /// let real = ScaledValue::<f64>::from_components(Scaled::ONE, Scaled::HALF);
    /// assert!(real.imaginary().is_zero());
    /// let complex = ScaledValue::<Complex64>::from_components(Scaled::ONE, Scaled::HALF);
    /// assert_eq!(complex.component_exponents(), [0, -1]);
    /// ```
    pub fn from_components(re: Scaled<V::Real>, im: Scaled<V::Real>) -> Self {
        let im = if V::IS_REAL { Scaled::ZERO } else { im };
        Self {
            re,
            im,
            value_type: core::marker::PhantomData,
        }
    }
    /// Converts a finite value exactly.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component is NaN or infinite.
    pub fn from_value(value: V) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            Scaled::from_value(value.real_component())?,
            Scaled::from_value(value.imaginary_component())?,
        ))
    }
    /// Returns the real component.
    pub fn real(self) -> Scaled<V::Real> {
        self.re
    }
    /// Returns the imaginary component, `+0` for a real `V`.
    pub fn imaginary(self) -> Scaled<V::Real> {
        self.im
    }
    /// Returns whether every component is zero.
    pub fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
    #[inline]
    /// Returns `self + other`, rounding each component once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn add(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            self.re.add(other.re)?,
            self.im.add(other.im)?,
        ))
    }

    #[inline]
    /// Returns `self − other`, rounding each component once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        self.add(other.neg())
    }

    /// Returns `self · b − c · d` componentwise, rounding each component once.
    ///
    /// Only the final exponents are checked, so intermediate products may exceed the
    /// `i64` exponent range.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn difference_of_products(self, b: Self, c: Self, d: Self) -> Result<Self, SciOxideError> {
        let re = Wide::from(self.re)
            .difference_of_products(b.re.into(), c.re.into(), d.re.into())?
            .sub(Wide::from(self.im).difference_of_products(
                b.im.into(),
                c.im.into(),
                d.im.into(),
            )?)?;
        let im = Wide::from(self.re)
            .difference_of_products(b.im.into(), c.re.into(), d.im.into())?
            .add(Wide::from(self.im).difference_of_products(
                b.re.into(),
                c.im.into(),
                d.re.into(),
            )?)?;
        Ok(Self::from_components(re.finish()?, im.finish()?))
    }

    /// Returns the principal square root.
    ///
    /// For a complex value on the negative real axis, the sign of the zero imaginary
    /// part selects the side of the branch cut: `−0` gives a root with negative
    /// imaginary part.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `V` is real and `self` is negative and nonzero.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn sqrt(self) -> Result<Self, SciOxideError> {
        if V::IS_REAL {
            return Ok(Self::from_components(self.re.sqrt()?, Scaled::ZERO));
        }
        if self.is_zero() {
            return Ok(Self::from_components(Scaled::ZERO, self.im));
        }
        if self.im.is_zero() {
            let root = self.re.abs().sqrt()?;
            return Ok(if self.re.is_sign_negative() {
                Self::from_components(
                    Scaled::ZERO,
                    if self.im.is_sign_negative() {
                        root.neg()
                    } else {
                        root
                    },
                )
            } else {
                Self::from_components(root, self.im)
            });
        }
        let re = Wide::from(self.re);
        let im = Wide::from(self.im);
        let large = re.hypot(im)?.add(re.abs())?.shifted(-1).sqrt()?;
        let small = im.abs().div(large.shifted(1))?;
        let (re, im) = if self.re.is_sign_negative() {
            (small, large)
        } else {
            (large, small)
        };
        let im = if self.im.is_sign_negative() {
            im.neg()
        } else {
            im
        };
        Ok(Self::from_components(re.finish()?, im.finish()?))
    }

    /// Returns `self · other`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        let re = Wide::from(self.re).difference_of_products(
            other.re.into(),
            self.im.into(),
            other.im.into(),
        )?;
        let im = Wide::from(self.re).difference_of_products(
            other.im.into(),
            Wide::from(self.im).neg(),
            other.re.into(),
        )?;
        Ok(Self::from_components(re.finish()?, im.finish()?))
    }

    /// Returns `self / other`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `other` is zero.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn div(self, other: Self) -> Result<Self, SciOxideError> {
        if other.im.is_zero() {
            return Ok(Self::from_components(
                self.re.div(other.re)?,
                self.im.div(other.re)?,
            ));
        }
        let ar = Wide::from(self.re);
        let ai = Wide::from(self.im);
        let br = Wide::from(other.re);
        let bi = Wide::from(other.im);
        let denominator = br.difference_of_products(br, bi.neg(), bi)?;
        let re = ar
            .difference_of_products(br, ai.neg(), bi)?
            .div(denominator)?;
        let im = ai.difference_of_products(br, ar, bi)?.div(denominator)?;
        Ok(Self::from_components(re.finish()?, im.finish()?))
    }

    /// Returns `|self| = √(re² + im²)` without intermediate overflow or underflow.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn magnitude(self) -> Result<Scaled<V::Real>, SciOxideError> {
        Wide::from(self.re).hypot(self.im.into())?.finish()
    }

    /// Converts to `V` with one rounding per component, allowing gradual underflow.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a rounded component is infinite.
    pub fn round(self) -> Result<V, SciOxideError> {
        Ok(V::from_components(self.re.round()?, self.im.round()?))
    }

    /// Returns `value · 2^exponent`, keeping the separate component scales.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `value` is NaN or infinite.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn from_parts(value: V, exponent: i64) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            Scaled::from_parts(value.real_component(), exponent)?,
            Scaled::from_parts(value.imaginary_component(), exponent)?,
        ))
    }
    /// Returns `(re · 2^e₀, im · 2^e₁)` for `value = (re, im)` and `exponents = [e₀, e₁]`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `value` is NaN or infinite.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn from_component_parts(value: V, exponents: [i64; 2]) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            Scaled::from_parts(value.real_component(), exponents[0])?,
            Scaled::from_parts(value.imaginary_component(), exponents[1])?,
        ))
    }
    /// Returns the significands of both components, each normalized separately.
    pub fn component_significands(self) -> V {
        V::from_components(self.real().significand(), self.imaginary().significand())
    }
    /// Returns the binary exponents of the real and imaginary components.
    pub fn component_exponents(self) -> [i64; 2] {
        [self.real().exponent(), self.imaginary().exponent()]
    }
    /// Returns the largest exponent of a nonzero component, or zero for zero.
    ///
    /// Normwise error estimates for this value use units of `2^scale_exponent`.
    pub fn scale_exponent(self) -> i64 {
        match (self.real().is_zero(), self.imaginary().is_zero()) {
            (true, true) => 0,
            (true, false) => self.imaginary().exponent(),
            (false, true) => self.real().exponent(),
            (false, false) => self.real().exponent().max(self.imaginary().exponent()),
        }
    }
    /// Returns `|self|` in units of `2^scale_exponent`, for normwise error estimates.
    ///
    /// The magnitude is computed with the scalar `hypot`; the stored components are
    /// not changed.
    pub fn scaled_magnitude(self) -> V::Real {
        let exponent = i128::from(self.scale_exponent());
        let coordinate = |part: Scaled<V::Real>| part.round_shifted(-exponent);
        coordinate(self.real()).hypot(coordinate(self.imaginary()))
    }
    /// Converts an absolute error to units of `2^scale_exponent`.
    ///
    /// Subnormal results round up, so the converted error is never smaller than the
    /// original.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `error` is NaN or infinite.
    /// - `InvalidInput` if `error` is negative.
    /// - `NumericalInstability` (`Overflow`) if the converted error is not
    ///   representable.
    pub fn scale_absolute_error(self, error: V::Real) -> Result<V::Real, SciOxideError> {
        if !error.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "absolute error must be finite",
            ));
        }
        if error < V::Real::ZERO {
            return Err(SciOxideError::invalid_input(
                crate::argument!("error"),
                InvalidInputReason::OutOfRange,
                "absolute error must be nonnegative",
            ));
        }
        let exponent = i128::from(self.scale_exponent());
        let scaled = Scaled::from_value(error)?.round_shifted(-exponent);
        if !scaled.is_finite() {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "absolute uncertainty exceeds scaled error coordinates",
            ));
        }
        let reconstructed = Scaled::from_value(scaled)?.round_shifted(exponent);
        Ok(if reconstructed < error {
            scaled.next_after(V::Real::MAX)
        } else {
            scaled
        })
    }
    /// Returns `self · 2^exponent` exactly.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn shifted(self, exponent: i64) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            self.real().shifted(exponent)?,
            self.imaginary().shifted(exponent)?,
        ))
    }
    /// Multiplies each component by a real `factor`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn mul_real(self, factor: Scaled<V::Real>) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            self.real().mul(factor)?,
            self.imaginary().mul(factor)?,
        ))
    }
    /// Converts to `V` with one rounding per component, rejecting underflow to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if a rounded component is infinite.
    /// - `NumericalInstability` (`Underflow`) if a nonzero component rounds to zero.
    pub fn try_value(self) -> Result<V, SciOxideError> {
        self.check_nonzero_components(self.round()?)
    }
    pub(crate) fn check_nonzero_components(self, value: V) -> Result<V, SciOxideError> {
        if (!self.real().is_zero() && value.real_component().is_zero())
            || (!self.imaginary().is_zero() && value.imaginary_component().is_zero())
        {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Underflow,
                "a nonzero component rounds to zero",
            ))
        } else {
            Ok(value)
        }
    }
    /// Converts to a [`ProjectiveValue`](crate::ProjectiveValue), whose components share one exponent.
    ///
    /// The result represents this value exactly. Not every value can be converted:
    /// the components may be too far apart in scale.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Underflow`) if a nonzero component would become zero.
    /// - `NumericalInstability` (`LossOfSignificance`) if the shared scale would
    ///   drop nonzero significand bits.
    pub fn try_projective(self) -> Result<ProjectiveValue<V>, SciOxideError> {
        ProjectiveValue::from_scaled(self)
    }
}

impl<S: RealScalar> ScaledValue<S> {
    /// Returns `sign · e^(log_magnitude)` without the IEEE range limit.
    ///
    /// Has the accuracy of [`Scaled::from_log`](crate::Scaled::from_log).
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `sign` or `log_magnitude` is not finite.
    /// - `InvalidInput` if `sign` is not `±1`.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if an exponent leaves the
    ///   `i64` range.
    pub fn from_log(sign: S, log_magnitude: S) -> Result<Self, SciOxideError> {
        Ok(Self::from_components(
            Scaled::from_log(sign, log_magnitude)?,
            Scaled::ZERO,
        ))
    }
}
