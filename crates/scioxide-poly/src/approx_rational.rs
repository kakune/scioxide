//! Rational functions with floating-point monomial-basis numerator and
//! denominator.

use crate::PolynomialBasis;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{NumericScalar, SciOxideError};

use crate::util::{round_complex, scaled_complex};
use crate::{
    ApproxPolynomial, ApproximateGcdReport, MonomialBasis, PolynomialEvaluation, approximate_gcd,
};

/// Holds the value of a rational function at a point, with the evaluations of
/// its numerator and denominator, as returned by
/// [`ApproxRationalFunction::evaluate_parts`].
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
/// let r = ApproxRationalFunction::new(p, one)?;
/// let evaluation = r.evaluate_parts(1.0)?;
/// assert!((evaluation.value() - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ApproximateRationalEvaluation<C: NumericScalar> {
    value: C,
    numerator: PolynomialEvaluation<C>,
    denominator: PolynomialEvaluation<C>,
}

impl<C: NumericScalar> ApproximateRationalEvaluation<C> {
    /// Returns the value `p(x)/q(x)`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let evaluation = r.evaluate_parts(1.0)?;
    /// assert!((evaluation.value() - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn value(self) -> C {
        self.value
    }

    /// Returns the evaluation of the numerator `p`, with its condition report.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let evaluation = r.evaluate_parts(1.0)?;
    /// assert!((evaluation.numerator().value() - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn numerator(self) -> PolynomialEvaluation<C> {
        self.numerator
    }

    /// Returns the evaluation of the denominator `q`, with its condition report.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let evaluation = r.evaluate_parts(1.0)?;
    /// assert_eq!(evaluation.denominator().value(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn denominator(self) -> PolynomialEvaluation<C> {
        self.denominator
    }
}

/// Represents a rational function `r(x) = p(x)/q(x)` with floating-point
/// coefficients in the monomial basis.
///
/// The denominator `q` is stored monic, and `p` is divided by the same leading
/// coefficient. Zero is stored as `0/1`. Arithmetic does not cancel common
/// factors; call [`simplify`](Self::simplify) to remove an approximate common
/// factor. Numerator and denominator carry the same
/// [`PolynomialApproximationPolicy`](crate::PolynomialApproximationPolicy).
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
/// let r = ApproxRationalFunction::new(p, one)?;
/// assert!((r.evaluate(1.0)? - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ApproxRationalFunction<C: NumericScalar> {
    numerator: ApproxPolynomial<C, MonomialBasis<C>>,
    denominator: ApproxPolynomial<C, MonomialBasis<C>>,
}

impl<C: NumericScalar> ApproxRationalFunction<C> {
    /// Creates `numerator / denominator` and normalizes the denominator to be
    /// monic.
    ///
    /// No common factor is removed. A zero numerator gives `0/1`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `denominator` is zero, or if the two polynomials have
    ///   different approximation policies.
    /// - `NumericalInstability` if a normalized coefficient overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// // r(x) = (x + 1)² / 1, so r(1) = 4.
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// assert!((r.evaluate(1.0)? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        numerator: ApproxPolynomial<C, MonomialBasis<C>>,
        denominator: ApproxPolynomial<C, MonomialBasis<C>>,
    ) -> Result<Self, SciOxideError> {
        if denominator.is_zero() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("denominator"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "the denominator polynomial must be nonzero",
            ));
        }
        crate::util::same_policy(
            numerator.policy(),
            denominator.policy(),
            scioxide_core::argument!("denominator.policy"),
        )?;
        let policy = numerator.policy();
        // Normal form: monic denominator with the numerator divided by the same
        // leading coefficient, so the represented function is unchanged. A zero
        // numerator collapses to `0 / 1`. No common factor is removed here; that is a
        // tolerance decision and belongs to the explicit `simplify`.
        if numerator.is_zero() {
            return Ok(Self {
                numerator,
                denominator: ApproxPolynomial::monomial_with_options(
                    try_filled(
                        1,
                        C::ONE,
                        scioxide_core::quantity!("zero_rational_denominator"),
                    )?,
                    policy,
                )?,
            });
        }
        let leading = scaled_complex(denominator.leading_coefficient())?;
        let mut coefficients = try_with_capacity(
            numerator.basis().coefficients().len(),
            scioxide_core::quantity!("rational_normalization"),
        )?;
        for coefficient in numerator.basis().coefficients() {
            coefficients.push(round_complex(scaled_complex(*coefficient)?.div(leading)?)?);
        }
        Ok(Self {
            numerator: ApproxPolynomial::monomial_with_options(coefficients, policy)?,
            denominator: denominator.monic()?,
        })
    }

    /// Adds two rational functions: `p₁/q₁ + p₂/q₂ = (p₁q₂ + p₂q₁)/(q₁q₂)`.
    ///
    /// Common factors are not cancelled, so degrees add up. Use
    /// [`simplify`](Self::simplify) to cancel them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operands have different approximation policies.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.try_add(&r)?;
    /// assert!((result.evaluate(1.0)? - 8.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_add(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator
                .try_mul(&other.denominator)?
                .try_add(&other.numerator.try_mul(&self.denominator)?)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Subtracts two rational functions: `p₁/q₁ − p₂/q₂ = (p₁q₂ − p₂q₁)/(q₁q₂)`.
    ///
    /// Common factors are not cancelled, so degrees add up. Use
    /// [`simplify`](Self::simplify) to cancel them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operands have different approximation policies.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.try_sub(&r)?;
    /// assert!((result.evaluate(1.0)? - 0.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_sub(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator
                .try_mul(&other.denominator)?
                .try_sub(&other.numerator.try_mul(&self.denominator)?)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Multiplies two rational functions: `(p₁/q₁)(p₂/q₂) = (p₁p₂)/(q₁q₂)`.
    ///
    /// Common factors are not cancelled, so degrees add up. Use
    /// [`simplify`](Self::simplify) to cancel them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the operands have different approximation policies.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.try_mul(&r)?;
    /// assert!((result.evaluate(1.0)? - 16.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_mul(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator.try_mul(&other.numerator)?,
            self.denominator.try_mul(&other.denominator)?,
        )
    }

    /// Divides two rational functions: `(p₁/q₁)/(p₂/q₂) = (p₁q₂)/(q₁p₂)`.
    ///
    /// Common factors are not cancelled, so degrees add up. Use
    /// [`simplify`](Self::simplify) to cancel them.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `other` is zero.
    /// - `InvalidInput` if the operands have different approximation policies.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.try_div(&r)?;
    /// assert!((result.evaluate(1.0)? - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_div(&self, other: &Self) -> Result<Self, SciOxideError> {
        Self::new(
            self.numerator.try_mul(&other.denominator)?,
            self.denominator.try_mul(&other.numerator)?,
        )
    }

    /// Returns the reciprocal `q/p`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the function is zero.
    /// - `NumericalInstability` if a normalized coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// assert!((r.reciprocal()?.evaluate(1.0)? - 0.25).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn reciprocal(&self) -> Result<Self, SciOxideError> {
        let copy = |p: &ApproxPolynomial<C, MonomialBasis<C>>| {
            let mut coefficients = try_with_capacity(
                p.basis().coefficients().len(),
                scioxide_core::quantity!("rational_reciprocal"),
            )?;
            coefficients.extend_from_slice(p.basis().coefficients());
            ApproxPolynomial::monomial_with_options(coefficients, p.policy())
        };
        Self::new(copy(&self.denominator)?, copy(&self.numerator)?)
    }

    /// Returns the numerator `p`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// assert_eq!(r.numerator().degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn numerator(&self) -> &ApproxPolynomial<C, MonomialBasis<C>> {
        &self.numerator
    }

    /// Returns the monic denominator `q`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// assert_eq!(r.denominator().basis().coefficients(), &[1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn denominator(&self) -> &ApproxPolynomial<C, MonomialBasis<C>> {
        &self.denominator
    }

    /// Evaluates `r(x) = p(x)/q(x)` with Horner's method.
    ///
    /// The point counts as a pole when `q(x) = 0` or when the forward-error
    /// estimate of `q(x)` relative to `|q(x)|` is at least one, that is, when the
    /// computed denominator could be zero. A small denominator that is computed
    /// accurately is accepted, whatever the coefficient tolerance. The quotient
    /// is formed in working precision, and in extended exponent range with one
    /// rounding when it would overflow or underflow. The cost is
    /// O(deg p + deg q), without derivatives and without allocation.
    /// [`evaluate_parts`](Self::evaluate_parts) also returns both polynomial
    /// evaluations; the two values agree to rounding.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `point` is NaN or infinite.
    /// - `DomainError` if `point` is a pole in the sense above.
    /// - `NumericalInstability` if the value overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// // (1 + x)² / (2 + x) at x = 1 is 4/3.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let q = ApproxPolynomial::monomial(vec![2.0_f64, 1.0])?;
    /// let r = ApproxRationalFunction::new(p, q)?;
    /// assert!((r.evaluate(1.0)? - 4.0 / 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: C) -> Result<C, SciOxideError> {
        crate::util::finite_input(point, "point must have finite components")?;
        let (numerator, _) = self.numerator.value_state(point)?;
        let (denominator, error) = self.denominator.value_state(point)?;
        if denominator == C::ZERO || error >= C::Real::ONE {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("point"),
                scioxide_core::DomainReason::OutsideDomain,
                "the denominator is zero or dominated by its evaluation error estimate",
            ));
        }
        if crate::admission::native_value(numerator) && crate::admission::native_value(denominator)
        {
            let value = numerator / denominator;
            if crate::admission::native_value(value) && (value != C::ZERO || numerator == C::ZERO) {
                return Ok(value);
            }
        }
        round_complex(scaled_complex(numerator)?.div(scaled_complex(denominator)?)?)
    }

    /// Evaluates `r(x) = p(x)/q(x)` and returns the value with both polynomial
    /// evaluations, their derivatives and condition reports.
    ///
    /// The pole test is that of [`evaluate`](Self::evaluate). The polynomials
    /// are evaluated with
    /// [`ApproxPolynomial::evaluate_with_derivative`], and the quotient is formed
    /// in extended exponent range and rounded once. The cost is
    /// O(deg p + deg q).
    ///
    /// # Errors
    ///
    /// The errors of [`evaluate`](Self::evaluate).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// // (1 + x)² / (2 + x) at x = 1: the parts are 4 and 3.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let q = ApproxPolynomial::monomial(vec![2.0_f64, 1.0])?;
    /// let r = ApproxRationalFunction::new(p, q)?;
    /// let parts = r.evaluate_parts(1.0)?;
    /// assert!((parts.value() - 4.0 / 3.0).abs() < 1e-14);
    /// assert!((parts.numerator().value() - 4.0).abs() < 1e-14);
    /// assert!((parts.denominator().value() - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_parts(
        &self,
        point: C,
    ) -> Result<ApproximateRationalEvaluation<C>, SciOxideError> {
        let numerator = self.numerator.evaluate_with_derivative(point)?;
        let denominator = self.denominator.evaluate_with_derivative(point)?;
        if denominator.value() == C::ZERO
            || denominator.condition().forward_error_estimate() >= C::Real::ONE
        {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("point"),
                scioxide_core::DomainReason::OutsideDomain,
                "the denominator is zero or dominated by its evaluation error estimate",
            ));
        }
        let value = round_complex(
            scaled_complex(numerator.value())?.div(scaled_complex(denominator.value())?)?,
        )?;
        Ok(ApproximateRationalEvaluation {
            value,
            numerator,
            denominator,
        })
    }

    /// Cancels the approximate common factor of numerator and denominator.
    ///
    /// The common factor is the [`approximate_gcd`] of `p` and `q`. The new
    /// numerator and denominator are the cofactors `p/g` and `q/g`. The
    /// cancellation is accepted only if the remainder of each division is within
    /// the tolerance of the approximation policy, relative to the largest
    /// coefficient of the operand. The cost is that of [`approximate_gcd`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`IllConditioned`) if a division remainder
    ///   exceeds the policy tolerance, so the factor is not a common factor
    ///   within that tolerance.
    /// - The errors of [`approximate_gcd`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// assert_eq!(r.simplify()?.rational().denominator().degree(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// Cancelling the common factor `x − 1`:
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // (x² − 1)/(x² + x − 2) = (x + 1)/(x + 2), which is 0.8 at x = 3.
    /// let p = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let q = ApproxPolynomial::monomial(vec![-2.0_f64, 1.0, 1.0])?;
    /// let simplified = ApproxRationalFunction::new(p, q)?.simplify()?;
    /// assert_eq!(simplified.gcd_report().effective_degree(), 1);
    /// assert_eq!(simplified.rational().denominator().degree(), 1);
    /// assert!((simplified.rational().evaluate(3.0)? - 0.8).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn simplify(&self) -> Result<ApproximateRationalSimplification<C>, SciOxideError> {
        let gcd = approximate_gcd(&self.numerator, &self.denominator)?;
        let ([numerator, denominator], [numerator_ok, denominator_ok], report) =
            gcd.into_cancellation();
        let [numerator_error, denominator_error] = report.operand_errors();
        if !numerator_ok || !denominator_ok {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "the rank-selected factor left a policy-significant remainder",
            ));
        }
        Ok(ApproximateRationalSimplification {
            rational: Self::new(numerator, denominator)?,
            gcd_report: report,
            numerator_error,
            denominator_error,
        })
    }
}

/// Holds the result of [`ApproxRationalFunction::simplify`].
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
/// let r = ApproxRationalFunction::new(p, one)?;
/// let result = r.simplify()?;
/// assert!(result.numerator_error() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ApproximateRationalSimplification<C: NumericScalar> {
    rational: ApproxRationalFunction<C>,
    gcd_report: ApproximateGcdReport<C::Real>,
    numerator_error: C::Real,
    denominator_error: C::Real,
}

impl<C: NumericScalar> ApproximateRationalSimplification<C> {
    /// Returns the simplified rational function.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.simplify()?;
    /// assert_eq!(result.rational().denominator().degree(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn rational(&self) -> &ApproxRationalFunction<C> {
        &self.rational
    }

    /// Returns the diagnostics of the approximate GCD that was cancelled.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.simplify()?;
    /// assert_eq!(result.gcd_report().effective_degree(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn gcd_report(&self) -> ApproximateGcdReport<C::Real> {
        self.gcd_report
    }

    /// Returns the remainder of dividing the numerator by the GCD, relative to
    /// the numerator's largest coefficient modulus.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.simplify()?;
    /// assert!(result.numerator_error() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn numerator_error(&self) -> C::Real {
        self.numerator_error
    }

    /// Returns the remainder of dividing the denominator by the GCD, relative to
    /// the denominator's largest coefficient modulus.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let one = ApproxPolynomial::monomial(vec![1.0_f64])?;
    /// let r = ApproxRationalFunction::new(p, one)?;
    /// let result = r.simplify()?;
    /// assert!(result.denominator_error() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn denominator_error(&self) -> C::Real {
        self.denominator_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PolynomialApproximationPolicy;

    #[test]
    fn approximate_rational_evaluation_and_simplification_report_policy() {
        let policy = PolynomialApproximationPolicy::<f64>::default();
        let common = ApproxPolynomial::monomial_with_options(vec![-1.0, 1.0], policy).unwrap();
        let numerator = common
            .try_mul(&ApproxPolynomial::monomial_with_options(vec![1.0, 1.0], policy).unwrap())
            .unwrap();
        let denominator = common
            .try_mul(&ApproxPolynomial::monomial_with_options(vec![2.0, 1.0], policy).unwrap())
            .unwrap();
        let rational = ApproxRationalFunction::new(numerator, denominator).unwrap();
        let simplified = rational.simplify().unwrap();
        assert_eq!(simplified.gcd_report().effective_degree(), 1);
        let evaluation = simplified.rational().evaluate_parts(3.0).unwrap();
        assert!((evaluation.value() - 0.8).abs() <= 1.0e-14);
        assert_eq!(simplified.rational().denominator().policy(), policy);
    }
}
