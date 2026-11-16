//! Monomial (power) basis: construction, arithmetic and calculus.
use super::*;

/// Monomial (power) basis: coefficients `a₀, …, aₙ` of `p(x) = Σ aᵢ xⁱ`.
///
/// Coefficients are stored in ascending order, without exact trailing zeros.
/// The zero polynomial is stored as the single coefficient `+0`. Construct
/// with [`ApproxPolynomial::monomial`], [`ApproxPolynomial::from_roots`] or
/// [`ApproxPolynomial::from_conjugate_pairs`].
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// assert_eq!(p.basis().coefficients().len(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct MonomialBasis<C: NumericScalar> {
    pub(super) coefficients: crate::admission::Coefficients<C>,
}

impl<C: NumericScalar> MonomialBasis<C> {
    pub(super) fn new(coefficients: Vec<C>) -> Self {
        Self {
            coefficients: crate::admission::Coefficients::new(coefficients),
        }
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Creates `p(x) = Σ aᵢ xⁱ` from the coefficients `a₀, …, aₙ` in ascending order.
    ///
    /// Exact trailing zeros are dropped; small nonzero coefficients are kept.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `coefficients` is empty.
    /// - `NonFinite` if a coefficient has a NaN or infinite component.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // p(x) = 1 + 2x + 3x²: p(0.5) = 2.75 and p′(0.5) = 2 + 6x = 5.
    /// let polynomial = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0])?;
    /// let result = polynomial.evaluate_with_derivative(0.5)?;
    /// assert!((result.value() - 2.75).abs() < 1e-14);
    /// assert!((result.derivative() - 5.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monomial(coefficients: Vec<C>) -> Result<Self, SciOxideError> {
        Self::monomial_with_options(coefficients, PolynomialApproximationPolicy::default())
    }

    /// Creates `p(x) = Σ aᵢ xⁱ` from its coefficients with the policy `policy`.
    ///
    /// This is the configured form of [`monomial`](Self::monomial), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`monomial`](Self::monomial), and `NumericalInstability`
    /// (`Overflow`) if a threshold of `policy` overflows at the scale of the
    /// coefficients.
    pub fn monomial_with_options(
        coefficients: Vec<C>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::from_basis(
            MonomialBasis::new(canonical_coefficients(
                coefficients,
                scioxide_core::argument!("coefficients"),
            )?),
            policy,
        )
    }

    /// Value and relative forward-error estimate, without the derivative.
    pub(crate) fn value_state(&self, point: C) -> Result<(C, C::Real), SciOxideError> {
        let state = crate::horner::admitted_horner::<C, 0, true>(
            self.basis.coefficients.view(),
            point,
            || Ok(()),
        )?;
        let (value, _, bound) = state.rounded()?;
        Ok((
            value,
            self.metadata.operation_factor * policy::scale_ratio(bound, value.abs()),
        ))
    }

    /// Creates the monic polynomial `p(x) = Π (x − rⱼ)` with the given roots.
    ///
    /// The roots are put in Leja order: the first has the largest magnitude,
    /// and each next one maximizes the product of distances to the roots
    /// already chosen. Ties are broken by real, then imaginary part. The linear
    /// factors are then multiplied pairwise in a balanced tree. The ordering
    /// keeps intermediate coefficients small, and it makes the result
    /// independent of the order of `roots`. Intermediate products use extended
    /// exponent range; only the final coefficients must fit the scalar type.
    /// The cost is O(n²) for `n` roots. An empty slice gives `p(x) = 1`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a root has a NaN or infinite component.
    /// - `NumericalInstability` if a final coefficient overflows.
    /// - `AllocationFailure` if the factors cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, PolynomialBasis};
    ///
    /// // (x + 2)(x − 1) = x² + x − 2
    /// let p = ApproxPolynomial::from_roots(&[-2.0_f64, 1.0])?;
    /// let expected = [-2.0_f64, 1.0, 1.0];
    /// let coefficients = p.basis().coefficients();
    /// assert!(
    ///     coefficients
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_roots(roots: &[C]) -> Result<Self, SciOxideError> {
        Self::from_roots_with_options(roots, PolynomialApproximationPolicy::default())
    }

    /// Creates the monic polynomial with the given roots and the policy `policy`.
    ///
    /// This is the configured form of [`from_roots`](Self::from_roots), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`from_roots`](Self::from_roots), and `NumericalInstability`
    /// (`Overflow`) if a threshold of `policy` overflows at the scale of the
    /// coefficients.
    pub fn from_roots_with_options(
        roots: &[C],
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let coefficients = balanced_coefficients_from_roots(roots)?;
        Self::monomial_with_options(coefficients, policy)
    }

    /// Returns `true` if this is the zero polynomial.
    ///
    /// The test is exact: the single stored coefficient is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert!(!p.is_zero());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn is_zero(&self) -> bool {
        let coefficients = self.basis().coefficients();
        coefficients.len() == 1 && coefficients[0].abs() == C::Real::ZERO
    }

    /// Returns the leading coefficient `aₙ`, which is nonzero unless the
    /// polynomial is zero.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.leading_coefficient(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn leading_coefficient(&self) -> C {
        let coefficients = self.basis().coefficients();
        coefficients[coefficients.len() - 1]
    }

    /// Returns the sum `self + other`, coefficient by coefficient.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `other` has a different policy.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.try_add(&p)?;
    /// assert!((result.evaluate(1.0)? - 8.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_add(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.require_same_policy(other)?;
        let (left, right) = (self.basis().coefficients(), other.basis().coefficients());
        let length = left.len().max(right.len());
        let mut coefficients = try_filled(
            length,
            C::ZERO,
            scioxide_core::quantity!("monomial_polynomial_sum"),
        )?;
        (0..length).try_for_each(|index| {
            coefficients[index] = crate::util::finite_arithmetic(
                left.get(index).copied().unwrap_or(C::ZERO)
                    + right.get(index).copied().unwrap_or(C::ZERO),
                "monomial coefficient arithmetic overflow or breakdown",
            )?;
            Ok::<_, SciOxideError>(())
        })?;
        Self::monomial_with_options(coefficients, self.policy)
    }

    /// Returns the difference `self − other`, coefficient by coefficient.
    ///
    /// Exact trailing zeros of the difference are dropped, so `p − p` is the
    /// zero polynomial.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `other` has a different policy.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.try_sub(&p)?;
    /// assert!((result.evaluate(1.0)? - 0.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_sub(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.require_same_policy(other)?;
        let (left, right) = (self.basis().coefficients(), other.basis().coefficients());
        let length = left.len().max(right.len());
        let mut coefficients = try_filled(
            length,
            C::ZERO,
            scioxide_core::quantity!("monomial_polynomial_difference"),
        )?;
        (0..length).try_for_each(|index| {
            coefficients[index] = crate::util::finite_arithmetic(
                left.get(index).copied().unwrap_or(C::ZERO)
                    - right.get(index).copied().unwrap_or(C::ZERO),
                "monomial coefficient arithmetic overflow or breakdown",
            )?;
            Ok::<_, SciOxideError>(())
        })?;
        Self::monomial_with_options(coefficients, self.policy)
    }

    /// Returns the product `self · other`.
    ///
    /// Short operands are multiplied directly in O(nm) operations; above a
    /// crossover length the product uses FFT convolution in O((n + m) log(n + m)).
    /// Use [`PolynomialOperationPlan`](crate::PolynomialOperationPlan) to choose
    /// the algorithm or to reuse storage for repeated products.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `other` has a different policy.
    /// - `NumericalInstability` if a coefficient overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result or the FFT storage
    ///   cannot be sized or allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.try_mul(&p)?;
    /// assert!((result.evaluate(1.0)? - 16.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_mul(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.require_same_policy(other)?;
        if self.is_zero() || other.is_zero() {
            return Self::monomial_with_options(
                try_filled(
                    1,
                    C::ZERO,
                    scioxide_core::quantity!("zero_polynomial_product"),
                )?,
                self.policy,
            );
        }
        Self::monomial_with_options(
            crate::fast::multiply(self.basis().coefficients(), other.basis().coefficients())?,
            self.policy,
        )
    }

    /// Returns the derivative `p′(x) = Σ i aᵢ xⁱ⁻¹`.
    ///
    /// The derivative of a constant is the zero polynomial.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a coefficient `i aᵢ` overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.derivative()?.basis().coefficients(), &[2.0, 2.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative(&self) -> Result<Self, SciOxideError> {
        if self.degree() == 0 {
            return Self::monomial_with_options(
                try_filled(
                    1,
                    C::ZERO,
                    scioxide_core::quantity!("constant_polynomial_derivative"),
                )?,
                self.policy,
            );
        }
        let mut coefficients = try_with_capacity(
            self.degree(),
            scioxide_core::quantity!("polynomial_derivative"),
        )?;
        for (degree, coefficient) in self
            .basis()
            .coefficients()
            .iter()
            .copied()
            .enumerate()
            .skip(1)
        {
            coefficients.push(crate::util::finite_arithmetic(
                coefficient * C::from_real(C::Real::try_from_usize_exact(degree)?),
                "monomial coefficient arithmetic overflow or breakdown",
            )?);
        }
        Self::monomial_with_options(coefficients, self.policy)
    }

    /// Returns the antiderivative `F` with `F(anchor) = value`.
    ///
    /// For `p(x) = Σ aᵢ xⁱ` of degree `n`, `F(x) = b₀ + Σᵢ aᵢ xⁱ⁺¹ / (i + 1)`.
    /// Each `aᵢ / (i + 1)` is formed in extended exponent range and rounded once
    /// per component, with gradual underflow. The constant `b₀` is
    /// `value − G(anchor)`, where `G` is the antiderivative with `b₀ = 0`
    /// evaluated by Horner's rule, so it carries the error of one evaluation,
    /// whose relative estimate is the forward-error estimate of
    /// [`evaluate_with_derivative`](ApproxPolynomial::evaluate_with_derivative)
    /// on `G` at `anchor`. The cost is O(n) operations and one vector of `n + 2`
    /// coefficients. Exact trailing zeros are dropped, so the antiderivative of
    /// the zero polynomial is the constant `value`. The result keeps the policy.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `anchor` or `value` is NaN or infinite.
    /// - `NumericalInstability` if `G(anchor)` or the constant term overflows.
    /// - `ScalarConversion` if `i + 1` is not exactly representable in the
    ///   scalar type.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// // p(x) = 2 + 6x has F(x) = 1 + 2x + 3x² with F(0) = 1.
    /// let p = ApproxPolynomial::monomial(vec![2.0_f64, 6.0])?;
    /// let primitive = p.antiderivative(0.0, 1.0)?;
    /// assert_eq!(primitive.basis().coefficients(), &[1.0, 2.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn antiderivative(&self, anchor: C, value: C) -> Result<Self, SciOxideError> {
        validate_slice(&[anchor, value])?;
        let coefficients = self.basis().coefficients();
        let mut primitive = try_with_capacity(
            scioxide_core::backend::checked_add(
                coefficients.len(),
                1,
                scioxide_core::quantity!("antiderivative_coefficients"),
            )?,
            scioxide_core::quantity!("antiderivative_coefficients"),
        )?;
        primitive.push(C::ZERO);
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let divisor = C::from_real(C::Real::try_from_usize_exact(index + 1)?);
            primitive.push(round_complex::<C>(
                scaled_complex(coefficient)?.div(scaled_complex(divisor)?)?,
            )?);
        }
        let unanchored = Self::monomial_with_options(primitive, self.policy)?;
        let constant = scaled_complex(value)?.sub(scaled_complex(unanchored.evaluate(anchor)?)?)?;
        let mut coefficients = unanchored.basis.coefficients.into_vec();
        coefficients[0] = round_complex::<C>(constant)?;
        Self::monomial_with_options(coefficients, self.policy)
    }

    /// Returns the definite integral `∫ p(x) dx` from `lower` to `upper`.
    ///
    /// The integral is `F(upper)` for the antiderivative `F` of
    /// [`antiderivative`](Self::antiderivative) with `F(lower) = 0`, in O(n)
    /// operations. Equal limits give exactly zero. The relative accuracy is lost
    /// when `|F(upper)|` is small compared with the values of `F` between the
    /// limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a limit is NaN or infinite.
    /// - `NumericalInstability` if the constant term or the integral overflows.
    /// - `ScalarConversion` if a degree is not exactly representable in the
    ///   scalar type.
    /// - `AllocationFailure` if the antiderivative cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// // ∫₀¹ (1 + 2x + 3x²) dx = 1 + 1 + 1 = 3.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0])?;
    /// assert!((p.integral(0.0, 1.0)? - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integral(&self, lower: C, upper: C) -> Result<C, SciOxideError> {
        validate_slice(&[lower, upper])?;
        if lower == upper {
            return Ok(C::ZERO);
        }
        self.antiderivative(lower, C::ZERO)?.evaluate(upper)
    }

    /// Returns `p(x) / aₙ`, whose leading coefficient is one.
    ///
    /// The zero polynomial is returned unchanged. Each quotient is computed in
    /// extended exponent range and rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a quotient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.monic()?.leading_coefficient(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn monic(&self) -> Result<Self, SciOxideError> {
        if self.is_zero() {
            return Self::monomial_with_options(
                try_copy_slice(
                    self.basis().coefficients(),
                    scioxide_core::quantity!("monic_polynomial"),
                )?,
                self.policy,
            );
        }
        let leading = scaled_complex(self.leading_coefficient())?;
        let mut coefficients = try_with_capacity(
            self.basis().coefficients().len(),
            scioxide_core::quantity!("monic_polynomial"),
        )?;
        for coefficient in self.basis().coefficients() {
            coefficients.push(round_complex::<C>(
                scaled_complex(*coefficient)?.div(leading)?,
            )?);
        }
        Self::monomial_with_options(coefficients, self.policy)
    }

    /// Reject a policy mismatch of the `other` operand.
    fn require_same_policy(&self, other: &Self) -> Result<(), SciOxideError> {
        crate::util::same_policy(
            self.policy,
            other.policy,
            scioxide_core::argument!("other.policy"),
        )
    }
}

impl<R> ApproxPolynomial<R, MonomialBasis<R>>
where
    R: RealScalar + NumericScalar<Real = R>,
{
    /// Creates a real monic polynomial from real roots and complex conjugate pairs.
    ///
    /// Each entry `z` of `conjugate_representatives` stands for the pair
    /// `z, z̄` and contributes the real quadratic factor
    /// `x² − 2 Re(z) x + |z|²`, formed in extended exponent range. Real roots
    /// contribute `x − r`. The factors are ordered and multiplied as in
    /// [`from_roots`](Self::from_roots), so the coefficients are real by
    /// construction and do not depend on the input order.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a root has a NaN or infinite component.
    /// - `InvalidInput` (`OutOfRange`) if a representative has `Im z ≤ 0`.
    /// - `NumericalInstability` if a final coefficient overflows.
    /// - `AllocationFailure` if the factors cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::{ApproxPolynomial, PolynomialBasis};
    ///
    /// // The pair ±i gives x² + 1.
    /// let p = ApproxPolynomial::from_conjugate_pairs(&[], &[Complex::new(0.0_f64, 1.0)])?;
    /// let expected = [1.0_f64, 0.0, 1.0];
    /// let coefficients = p.basis().coefficients();
    /// assert!(
    ///     coefficients
    ///         .iter()
    ///         .zip(expected)
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_conjugate_pairs(
        real_roots: &[R],
        conjugate_representatives: &[scioxide_core::Complex<R>],
    ) -> Result<Self, SciOxideError> {
        Self::from_conjugate_pairs_with_options(
            real_roots,
            conjugate_representatives,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Creates a real monic polynomial from real roots and conjugate pairs with the
    /// policy `policy`.
    ///
    /// This is the configured form of
    /// [`from_conjugate_pairs`](Self::from_conjugate_pairs), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`from_conjugate_pairs`](Self::from_conjugate_pairs), and
    /// `NumericalInstability` (`Overflow`) if a threshold of `policy` overflows at the
    /// scale of the coefficients.
    pub fn from_conjugate_pairs_with_options(
        real_roots: &[R],
        conjugate_representatives: &[scioxide_core::Complex<R>],
        policy: PolynomialApproximationPolicy<R>,
    ) -> Result<Self, SciOxideError> {
        real_roots
            .iter()
            .copied()
            .try_for_each(|root| finite_input(root, "real root must be finite").map(drop))?;
        conjugate_representatives
            .iter()
            .copied()
            .try_for_each(|root| {
                if !root.re.is_finite() || !root.im.is_finite() {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "complex root must have finite components",
                    ));
                }
                (root.im > <R as scioxide_core::AlgebraicScalar>::ZERO)
                    .then_some(())
                    .ok_or_else(|| {
                        SciOxideError::invalid_input(
                            scioxide_core::argument!("conjugate_representatives"),
                            InvalidInputReason::OutOfRange,
                            "each representative must have a positive imaginary component",
                        )
                    })
            })?;

        let factor_count = scioxide_core::backend::checked_add(
            real_roots.len(),
            conjugate_representatives.len(),
            scioxide_core::quantity!("conjugate_root_factors"),
        )?;
        let mut factors = try_with_capacity(
            factor_count,
            scioxide_core::quantity!("conjugate_root_factors"),
        )?;
        for &root in real_roots {
            factors.push((
                scioxide_core::Complex::new(root, <R as scioxide_core::AlgebraicScalar>::ZERO),
                scaled_factor(&[-root, <R as scioxide_core::AlgebraicScalar>::ONE])?,
            ));
        }
        // Form |z|² and -2 Re(z) without an intermediate scalar overflow.
        for &root in conjugate_representatives {
            let re = ScaledValue::from_value(root.re)?;
            let im = ScaledValue::from_value(root.im)?;
            let mut quadratic =
                try_with_capacity(3, scioxide_core::quantity!("scaled_polynomial_factor"))?;
            quadratic.push(re.mul(re)?.add(im.mul(im)?)?);
            quadratic.push(-re.add(re)?);
            quadratic.push(ScaledValue::ONE);
            factors.push((root, quadratic));
        }
        leja_order(&mut factors, |factor| factor.0)?;
        let factors = crate::util::try_collect(
            factors.into_iter().map(|(_, factor)| Ok(factor)),
            scioxide_core::quantity!("conjugate_root_factors"),
        )?;
        Self::monomial_with_options(round_monic_product(balanced_product(factors)?)?, policy)
    }
}

impl<C: NumericScalar> PolynomialBasis<C> for MonomialBasis<C> {
    fn coefficients(&self) -> &[C] {
        &self.coefficients
    }
}

impl<C: NumericScalar> private::Prepared<C> for MonomialBasis<C> {
    fn root_condition_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<crate::RootConditionReport<C::Real>, SciOxideError> {
        crate::root_diagnostics::polynomial_root_condition(
            &self.coefficients,
            scioxide_core::Complex::new(point.real_component(), point.imaginary_component()),
        )
    }

    #[inline(always)]
    fn evaluate_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<(C, C, EvaluationEnvelope<C::Real>), SciOxideError> {
        crate::horner::admitted_horner::<_, 1, true>(self.coefficients.view(), point, || Ok(()))?
            .rounded()
            .map(absolute_envelope)
    }

    fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError> {
        try_copy_slice(
            &self.coefficients,
            scioxide_core::quantity!("monomial_coefficients"),
        )
    }
}

/// Shared monomial evaluation, projecting the fields requested by this basis.
pub(super) fn evaluate_monomial<C: NumericScalar>(
    coefficients: &[C],
    point: C,
) -> Result<(C, C, C::Real), SciOxideError> {
    crate::horner::monomial_horner::<_, 1, true>(coefficients.iter().copied(), point, || Ok(()))
        .and_then(crate::horner::HornerEvaluation::rounded)
}
