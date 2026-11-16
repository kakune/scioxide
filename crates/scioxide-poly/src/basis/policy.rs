//! Approximation policy, evaluation results and diagnostic reports.
use super::*;

/// Tolerances for the numerical decisions made on an [`ApproxPolynomial`].
///
/// The coefficient tolerance gives the threshold `atol + rtol · scale` below
/// which a coordinate counts as zero, where `scale` is the largest coordinate
/// magnitude. It decides the effective degree in
/// [`ApproxPolynomial::classify_coefficients`], whether
/// [`approximate_gcd`](crate::approximate_gcd) accepts a common factor, and the
/// reconstruction checks of partial fractions. The rank policy decides which
/// coordinates or singular values count toward a numerical rank.
///
/// The default uses `atol = rtol = √ε` and
/// [`RankPolicy::DimensionScaledEpsilon`](scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
/// where `ε` is the machine epsilon. A polynomial checks at construction that
/// both thresholds are finite for its coefficients.
///
/// ```
/// use scioxide_poly::*;
/// let policy = PolynomialApproximationPolicy::<f64>::default();
/// assert!(policy.tolerance().relative() >= 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialApproximationPolicy<R: RealScalar> {
    pub(super) tolerance: scioxide_core::Tolerance<R>,
    pub(super) rank_policy: RankPolicy<R>,
}

impl<R: RealScalar> PolynomialApproximationPolicy<R> {
    /// Creates a policy from a coefficient tolerance and a rank policy.
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// let policy = PolynomialApproximationPolicy::new(
    ///     scioxide_core::Tolerance::<f64>::from_sqrt_epsilon(),
    ///     scioxide_linalg::RankPolicy::DimensionScaledEpsilon,
    /// );
    /// assert!(policy.tolerance().relative() >= 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new(tolerance: scioxide_core::Tolerance<R>, rank_policy: RankPolicy<R>) -> Self {
        Self {
            tolerance,
            rank_policy,
        }
    }

    /// Returns the policy with the coefficient tolerance `tolerance`.
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// use scioxide_poly::PolynomialApproximationPolicy;
    ///
    /// let tolerance = Tolerance::<f64>::new(1e-10, 1e-8)?;
    /// let policy = PolynomialApproximationPolicy::default().with_tolerance(tolerance);
    /// assert_eq!(policy.tolerance(), tolerance);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn with_tolerance(self, tolerance: scioxide_core::Tolerance<R>) -> Self {
        Self { tolerance, ..self }
    }

    /// Returns the policy with the rank policy `rank_policy`.
    ///
    /// ```
    /// use scioxide_linalg::RankPolicy;
    /// use scioxide_poly::PolynomialApproximationPolicy;
    ///
    /// let policy =
    ///     PolynomialApproximationPolicy::<f64>::default().with_rank_policy(RankPolicy::Exact);
    /// assert_eq!(policy.rank_policy(), RankPolicy::Exact);
    /// ```
    pub const fn with_rank_policy(self, rank_policy: RankPolicy<R>) -> Self {
        Self {
            rank_policy,
            ..self
        }
    }

    /// Returns the coefficient tolerance.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let policy = PolynomialApproximationPolicy::<f64>::default();
    /// assert_eq!(policy.tolerance(), scioxide_core::Tolerance::from_sqrt_epsilon());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn tolerance(self) -> scioxide_core::Tolerance<R> {
        self.tolerance
    }

    /// Returns the rank policy.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let policy = PolynomialApproximationPolicy::<f64>::default();
    /// assert_eq!(policy.rank_policy(), scioxide_linalg::RankPolicy::DimensionScaledEpsilon);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn rank_policy(self) -> RankPolicy<R> {
        self.rank_policy
    }
}

impl<R: RealScalar> Default for PolynomialApproximationPolicy<R> {
    fn default() -> Self {
        Self::new(
            scioxide_core::Tolerance::from_sqrt_epsilon(),
            RankPolicy::DimensionScaledEpsilon,
        )
    }
}

/// Error estimates and conditioning of one operation on an [`ApproxPolynomial`].
///
/// Evaluation, basis conversion, division and the product and composition
/// condition functions return this report. It holds a backward-error estimate, a condition estimate and
/// their product, the forward-error estimate. These are first-order estimates
/// from a rounding model, not bounds.
///
/// The rounding model is `γₖ = k ε / (1 − k ε)`, where `ε` is the machine
/// epsilon and `k` is the number of stored coordinates.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let evaluation = p.evaluate_with_derivative(0.5)?;
/// let report = evaluation.condition();
/// assert!(report.condition_estimate() >= 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialConditionReport<R: RealScalar> {
    pub(super) coefficient_scale: R,
    pub(super) backward_error: R,
    pub(super) forward_error_estimate: R,
    pub(super) condition_estimate: R,
}

impl<R: RealScalar> PolynomialConditionReport<R> {
    /// Returns the largest magnitude of the result's stored coordinates
    /// (coefficients, nodal values or control coefficients).
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// let report = evaluation.condition();
    /// assert_eq!(report.coefficient_scale(), 2.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn coefficient_scale(self) -> R {
        self.coefficient_scale
    }

    /// Returns the relative backward-error estimate.
    ///
    /// Evaluation, conversion, product and composition report `γₙ`, where `n`
    /// is the number of coordinates (for a conversion, the larger of the source
    /// and destination counts). Division reports the measured residual
    ///
    /// ```text
    /// ‖a − (q b + r)‖₁ / (‖a‖₁ + ‖q‖₁ ‖b‖₁ + ‖r‖₁)
    /// ```
    ///
    /// for dividend `a`, divisor `b`, quotient `q` and remainder `r`, and zero
    /// when that denominator is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// let report = evaluation.condition();
    /// assert!(report.backward_error() >= 0.0 && report.backward_error() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn backward_error(self) -> R {
        self.backward_error
    }

    /// Returns the relative forward-error estimate `γₙ · κ`.
    ///
    /// `κ` is [`condition_estimate`](Self::condition_estimate) and `n` is the
    /// coordinate count used by [`backward_error`](Self::backward_error); for
    /// division, `n` is the number of dividend coefficients. The estimate is
    /// `+∞` when `κ` is infinite, for example when evaluating at a root.
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// let p = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let report = p.evaluate_with_derivative(1.0)?.condition();
    /// assert!(report.backward_error() < 1e-14);
    /// assert!(report.forward_error_estimate().is_infinite());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn forward_error_estimate(self) -> R {
        self.forward_error_estimate
    }

    /// Returns the condition estimate `κ ≥ 1` of the operation.
    ///
    /// - Evaluation: the absolute error envelope divided by `|p(x)|`; see
    ///   [`ApproxPolynomial::evaluate`]. A zero envelope gives `κ = 1`.
    /// - Conversion: the growth of the largest coordinate magnitude from the
    ///   source to the destination basis.
    /// - Product, from [`polynomial_product_condition`](crate::polynomial_product_condition):
    ///   `‖a‖₁ ‖b‖₁ / ‖a b‖₁`.
    /// - Composition `p(q(x))`, from
    ///   [`polynomial_composition_condition`](crate::polynomial_composition_condition):
    ///   `Σ |aᵢ| ‖q‖₁ⁱ / ‖p ∘ q‖₁`, where `aᵢ` are the coefficients of `p`.
    /// - Division: the first-order propagation of coefficient perturbations.
    ///
    /// `κ = +∞` means that the result is not sensitive to relative
    /// perturbations in a bounded way, for example a zero value with a nonzero
    /// envelope, or that the estimate overflows.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// let report = evaluation.condition();
    /// assert!((report.condition_estimate() - 1.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition_estimate(self) -> R {
        self.condition_estimate
    }
}

/// Effective degree and rank of a polynomial's stored coordinates.
///
/// [`ApproxPolynomial::classify_coefficients`] returns this report. The
/// effective degree uses coefficients that multiply a basis polynomial of known
/// degree; the rank and scale use the stored coordinates. The polynomial is
/// unchanged.
///
/// ```
/// use scioxide_poly::ApproxPolynomial;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0])?;
/// assert_eq!(p.classify_coefficients()?.effective_degree(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialCoefficientReport<R: RealScalar> {
    pub(super) coefficient_scale: R,
    pub(super) effective_degree: usize,
    pub(super) effective_rank: usize,
    pub(super) policy: PolynomialApproximationPolicy<R>,
}

impl<R: RealScalar> PolynomialCoefficientReport<R> {
    /// Returns the largest magnitude of the stored coordinates.
    pub const fn coefficient_scale(self) -> R {
        self.coefficient_scale
    }

    /// Returns the index of the last coefficient above the policy's tolerance.
    ///
    /// Monomial, Newton and orthogonal forms test their own coefficients.
    /// Barycentric forms test the Newton divided differences of the nodal
    /// values in node order. Bernstein forms test the power coefficients in
    /// `t = (x − a)/(b − a)`. The tolerance threshold uses the largest of the
    /// tested magnitudes as its scale. The zero polynomial has degree zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let report = p.classify_coefficients()?;
    /// assert_eq!(report.effective_degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn effective_degree(self) -> usize {
        self.effective_degree
    }

    /// Returns the number of stored coordinates whose magnitude exceeds the
    /// rank threshold of the policy.
    ///
    /// This counts significant coordinates; it is neither a degree nor a
    /// matrix rank.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let report = p.classify_coefficients()?;
    /// assert_eq!(report.effective_rank(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn effective_rank(self) -> usize {
        self.effective_rank
    }

    /// Returns the policy used for this classification.
    pub const fn policy(self) -> PolynomialApproximationPolicy<R> {
        self.policy
    }
}

/// Value, first derivative and error estimates from [`ApproxPolynomial::evaluate`].
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let evaluation = p.evaluate_with_derivative(0.5)?;
/// assert!((evaluation.value() - 2.25).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct PolynomialEvaluation<C: NumericScalar> {
    pub(crate) value: C,
    pub(crate) derivative: C,
    pub(crate) condition: PolynomialConditionReport<C::Real>,
}

impl<C: NumericScalar> PolynomialEvaluation<C> {
    /// Returns the value `p(x)`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// assert!((evaluation.value() - 2.25).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn value(self) -> C {
        self.value
    }

    /// Returns the first derivative `p′(x)`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// assert!((evaluation.derivative() - 3.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn derivative(self) -> C {
        self.derivative
    }

    /// Returns the condition and error estimates of the evaluation.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let evaluation = p.evaluate_with_derivative(0.5)?;
    /// assert!(evaluation.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition(self) -> PolynomialConditionReport<C::Real> {
        self.condition
    }
}

/// Polynomial converted to another basis, with the diagnostics of the conversion.
///
/// The `to_*` methods of [`ApproxPolynomial`] return this type.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let result = p.to_newton(vec![0.0_f64, 1.0, 2.0])?;
/// assert!((result.polynomial().evaluate(0.5)? - 2.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PolynomialConversion<C: NumericScalar, B: PolynomialBasis<C>> {
    pub(super) polynomial: ApproxPolynomial<C, B>,
    pub(super) condition: PolynomialConditionReport<C::Real>,
}

impl<C: NumericScalar, B: PolynomialBasis<C>> PolynomialConversion<C, B> {
    /// Returns the converted polynomial.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.to_newton(vec![0.0_f64, 1.0, 2.0])?;
    /// assert!((result.polynomial().evaluate(0.5)? - 2.25).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn polynomial(&self) -> &ApproxPolynomial<C, B> {
        &self.polynomial
    }

    /// Returns the condition and error estimates of the conversion.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.to_newton(vec![0.0_f64, 1.0, 2.0])?;
    /// assert!(result.condition().backward_error() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition(&self) -> PolynomialConditionReport<C::Real> {
        self.condition
    }

    /// Returns the converted polynomial, consuming `self`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.to_newton(vec![0.0_f64, 1.0, 2.0])?;
    /// let converted = result.into_polynomial();
    /// assert!((converted.evaluate(0.5)? - 2.25).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn into_polynomial(self) -> ApproxPolynomial<C, B> {
        self.polynomial
    }
}

/// Evaluation diagnostics, reporting arithmetic failures.
pub(crate) fn condition_report<C: NumericScalar>(
    coefficients: &[C],
    evaluation_bound: C::Real,
    value_magnitude: C::Real,
) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
    let condition_estimate = scale_ratio(evaluation_bound, value_magnitude);
    let backward_error = rounding_error::<C::Real>(coefficients.len())?;
    condition_report_from_scale(
        coefficient_scale(coefficients),
        backward_error,
        condition_estimate,
        coefficients.len(),
    )
}

/// Operation report from its stored-coordinate scale. Reports carry no
/// policy-derived thresholds, so none are formed here; construction and explicit
/// classification own threshold validation.
pub(crate) fn condition_report_from_scale<R: RealScalar>(
    scale: R,
    backward_error: R,
    condition_estimate: R,
    rounding_dimension: usize,
) -> Result<PolynomialConditionReport<R>, SciOxideError> {
    Ok(PolynomialConditionReport {
        coefficient_scale: scale,
        backward_error,
        forward_error_estimate: rounding_error::<R>(rounding_dimension)? * condition_estimate,
        condition_estimate,
    })
}

pub(crate) fn rank_threshold<R: RealScalar>(
    policy: RankPolicy<R>,
    scale: R,
    dimension: usize,
) -> Result<R, SciOxideError> {
    match policy {
        RankPolicy::Exact => Ok(R::ZERO),
        RankPolicy::DimensionScaledEpsilon => {
            let dimension = R::try_from_usize_exact(dimension)?;
            let product = scale * dimension;
            let threshold = product * R::EPSILON;
            // Epsilon is an exact binary power. With normal finite output,
            // this has the same significand rounding as the scaled product;
            // only exceptional intermediate/final range needs extended work.
            if product.is_finite()
                && threshold.is_finite()
                && (threshold >= R::MIN_POSITIVE || scale == R::ZERO || dimension == R::ZERO)
            {
                Ok(threshold)
            } else {
                Scaled::from_value(scale)?
                    .mul(Scaled::from_value(dimension)?)?
                    .mul(Scaled::from_value(R::EPSILON)?)?
                    .round()
            }
        }
        RankPolicy::Threshold(threshold) => threshold.threshold(scale),
    }
}

/// Absolute evaluation envelope `absolute + |p(z)| value_factor`.
///
/// Running-error envelopes have no value term. The barycentric second form adds
/// `|p(z)|` times its Lebesgue function. The parts stay separate so that neither
/// the condition nor the extended-range root diagnostics form a native sum that
/// overflows while the value itself is representable.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationEnvelope<R> {
    absolute: R,
    value_factor: R,
}

impl<R: RealScalar> EvaluationEnvelope<R> {
    #[inline(always)]
    pub(crate) fn new(absolute: R, value_factor: R) -> Self {
        Self {
            absolute,
            value_factor,
        }
    }

    /// `max(1, envelope / |p|)` without forming the envelope. A zero absolute
    /// part bounds `|p|` by zero, so the whole envelope vanishes: condition one.
    #[inline(always)]
    pub(crate) fn condition(self, magnitude: R) -> R {
        let condition = scale_ratio(self.absolute, magnitude);
        if self.absolute == R::ZERO {
            condition
        } else {
            condition + self.value_factor
        }
    }

    /// The whole envelope in extended range.
    pub(crate) fn scaled(self, magnitude: Scaled<R>) -> Result<Scaled<R>, SciOxideError> {
        Scaled::from_value(self.absolute)?
            .add(magnitude.mul(Scaled::from_value(self.value_factor)?)?)
    }
}

/// Attach a running-error envelope, which has no value term.
#[inline(always)]
pub(crate) fn absolute_envelope<C: NumericScalar>(
    (value, derivative, bound): (C, C, C::Real),
) -> (C, C, EvaluationEnvelope<C::Real>) {
    (
        value,
        derivative,
        EvaluationEnvelope::new(bound, C::Real::ZERO),
    )
}

pub(super) fn scale_ratio<R: RealScalar>(numerator: R, denominator: R) -> R {
    if numerator == R::ZERO {
        return R::ONE;
    }
    if denominator == R::ZERO {
        return R::INFINITY;
    }
    // Infinity is intentional diagnostic information, never a polynomial value.
    (numerator / denominator).max(R::ONE)
}

/// Common range policy for extended operation sensitivity ratios.
pub(crate) fn scaled_condition_ratio<R: RealScalar>(
    numerator: Scaled<R>,
    denominator: Scaled<R>,
) -> Result<R, SciOxideError> {
    if numerator.is_zero() {
        return Ok(R::ONE);
    }
    if denominator.is_zero() {
        return Ok(R::INFINITY);
    }
    let ratio = numerator.div(denominator)?;
    if !ratio.abs_le(Scaled::from_value(R::MAX)?) {
        return Ok(R::INFINITY);
    }
    Ok(ratio.round()?.max(R::ONE))
}

/// Classify degree-bearing coefficients without projecting them to scalar range.
pub(super) fn scaled_effective_degree<C: NumericScalar>(
    coefficients: &[ScaledValue<C>],
    tolerance: scioxide_core::Tolerance<C::Real>,
) -> Result<usize, SciOxideError> {
    let mut scale = Scaled::ZERO;
    for &coefficient in coefficients {
        let magnitude = coefficient.magnitude()?;
        if magnitude.abs_cmp(scale).is_gt() {
            scale = magnitude;
        }
    }
    let threshold = scioxide_core::backend::scaled::scaled_tolerance_threshold(&tolerance, scale)?;
    for (index, &coefficient) in coefficients.iter().enumerate().rev() {
        if coefficient.magnitude()?.abs_cmp(threshold).is_gt() {
            return Ok(index);
        }
    }
    Ok(0)
}
