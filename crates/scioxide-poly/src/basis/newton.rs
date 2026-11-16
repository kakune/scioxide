//! Newton basis: divided-difference coefficients and their nodes.
use super::*;
use crate::admission::{
    Arithmetic, ArithmeticResult, Extended, Native, block_point, candidate, extended, require,
};

/// Newton basis: coefficients `c₀, …, cₙ` and nodes `x₀, …, xₙ₋₁` of
///
/// ```text
/// p(x) = c₀ + c₁ (x − x₀) + c₂ (x − x₀)(x − x₁) + … + cₙ (x − x₀)⋯(x − xₙ₋₁).
/// ```
///
/// For an interpolant, `cᵢ` is the divided difference `f[x₀, …, xᵢ]`. Construct
/// with [`ApproxPolynomial::newton`] or [`ApproxPolynomial::interpolate_newton`].
/// [`PolynomialBasis::coefficients`] returns `c₀, …, cₙ`, and
/// [`nodes`](Self::nodes) returns the nodes.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::newton(vec![1.0_f64, 2.0], vec![0.0_f64])?;
/// assert_eq!(p.degree(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct NewtonBasis<C: NumericScalar> {
    pub(super) coefficients: Vec<C>,
    pub(super) nodes: Vec<C>,
    block_native: bool,
}

impl<C: NumericScalar> NewtonBasis<C> {
    pub(super) fn prepared(coefficients: Vec<C>, nodes: Vec<C>) -> Self {
        let block_native = C::IS_REAL
            && C::Real::MANTISSA_DIGITS == 53
            && coefficients
                .iter()
                .all(|&coefficient| <Native as Arithmetic<C>>::checkpoint(coefficient, true))
            && nodes.iter().all(|&node| block_point(node));
        Self {
            coefficients,
            nodes,
            block_native,
        }
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, NewtonBasis<C>> {
    /// Creates a polynomial in Newton form from its coefficients and nodes.
    ///
    /// `nodes` must have one entry fewer than `coefficients`; the nodes need
    /// not be distinct. All coefficients and nodes are kept, including trailing
    /// zero coefficients, so [`degree`](Self::degree) is
    /// `coefficients.len() − 1`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `coefficients` is empty.
    /// - `DimensionMismatch` if `nodes.len() != coefficients.len() − 1`.
    /// - `NonFinite` if a coefficient or node has a NaN or infinite component.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // p(x) = 1 + 2(x − 0), so p(0.5) = 2.
    /// let p = ApproxPolynomial::newton(vec![1.0_f64, 2.0], vec![0.0_f64])?;
    /// let result = p.evaluate_with_derivative(0.5)?;
    /// assert!((result.value() - 2.0).abs() < 1e-13);
    /// assert!(result.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn newton(coefficients: Vec<C>, nodes: Vec<C>) -> Result<Self, SciOxideError> {
        Self::newton_with_options(
            coefficients,
            nodes,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Creates a polynomial in Newton form with the policy `policy`.
    ///
    /// This is the configured form of [`newton`](Self::newton), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`newton`](Self::newton), and `NumericalInstability` (`Overflow`)
    /// if a threshold of `policy` overflows at the scale of the coefficients.
    pub fn newton_with_options(
        coefficients: Vec<C>,
        nodes: Vec<C>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        if coefficients.is_empty() {
            return Err(empty_error(scioxide_core::argument!("coefficients")));
        }
        if nodes.len() != coefficients.len().saturating_sub(1) {
            return Err(layout_error(coefficients.len() - 1, nodes.len()));
        }
        validate_slice(&coefficients)?;
        validate_slice(&nodes)?;
        Self::from_basis(NewtonBasis::prepared(coefficients, nodes), policy)
    }
}

impl<R> ApproxPolynomial<R, NewtonBasis<R>>
where
    R: RealScalar + NumericScalar<Real = R>,
{
    /// Interpolates real samples `(xⱼ, yⱼ)` by a polynomial in Newton form.
    ///
    /// The result is the unique polynomial of degree at most `n − 1` through
    /// the `n` samples, with coefficients `cᵢ = f[x₀, …, xᵢ]` computed by the
    /// divided-difference table in O(n²) operations. An entry that would
    /// overflow or underflow is computed in extended exponent range; only the
    /// final coefficients must fit the scalar type. Exact trailing zero
    /// coefficients are dropped together with their nodes, so samples of a
    /// line give degree 1.
    ///
    /// Evaluation takes O(n) operations per point. For complex samples, use
    /// [`interpolate_barycentric`](ApproxPolynomial::interpolate_barycentric).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `nodes` is empty.
    /// - `DimensionMismatch` if `nodes` and `values` differ in length.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if two nodes are equal.
    /// - `NumericalInstability` if a final coefficient overflows, or a nonzero
    ///   one underflows to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // The line through (0, 1) and (1, 3) is p(x) = 1 + 2x, so p(0.5) = 2.
    /// let p = ApproxPolynomial::interpolate_newton(vec![0.0_f64, 1.0], vec![1.0_f64, 3.0])?;
    /// assert!((p.evaluate(0.5)? - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// K. E. Atkinson, *An Introduction to Numerical Analysis*, 2nd ed., Wiley,
    /// 1989, §3.2.
    #[inline(always)]
    pub fn interpolate_newton(nodes: Vec<R>, values: Vec<R>) -> Result<Self, SciOxideError> {
        Self::interpolate_newton_with_options(
            nodes,
            values,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Interpolates real samples in Newton form with the policy `policy`.
    ///
    /// This is the configured form of [`interpolate_newton`](Self::interpolate_newton),
    /// which uses [`PolynomialApproximationPolicy::default`]; the result carries
    /// `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`interpolate_newton`](Self::interpolate_newton), and
    /// `NumericalInstability` (`Overflow`) if a threshold of `policy` overflows at the
    /// scale of the coefficients.
    #[inline(always)]
    pub fn interpolate_newton_with_options(
        nodes: Vec<R>,
        values: Vec<R>,
        policy: PolynomialApproximationPolicy<R>,
    ) -> Result<Self, SciOxideError> {
        let validated = validate_interpolation_data(&nodes, &values)?;
        let coefficients = canonical_coefficients(
            numeric_divided_differences(validated, values)?.into_native()?,
            scioxide_core::argument!("values"),
        )?;
        let mut nodes = nodes;
        nodes.truncate(coefficients.len() - 1);
        Self::from_basis(NewtonBasis::prepared(coefficients, nodes), policy)
    }
}

impl<C: NumericScalar> NewtonBasis<C> {
    /// Returns the nodes `x₀, …, xₙ₋₁` in the order of the Newton form.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::newton(vec![1.0_f64, 2.0], vec![0.0_f64])?;
    /// assert_eq!(p.basis().nodes(), &[0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn nodes(&self) -> &[C] {
        &self.nodes
    }
}

impl<C: NumericScalar> PolynomialBasis<C> for NewtonBasis<C> {
    fn coefficients(&self) -> &[C] {
        &self.coefficients
    }
}

impl<C: NumericScalar> private::Prepared<C> for NewtonBasis<C> {
    #[inline(always)]
    fn evaluate_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<(C, C, EvaluationEnvelope<C::Real>), SciOxideError> {
        if self.block_native && block_point(point) {
            if let Some(state) = candidate(self.evaluate_arithmetic::<Native, true>(point))? {
                return Ok(absolute_envelope(state));
            }
        } else if let Some(state) = candidate(self.evaluate_arithmetic::<Native, false>(point))? {
            return Ok(absolute_envelope(state));
        }
        self.evaluate_extended(point).map(absolute_envelope)
    }

    fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError> {
        materialize_monomial(
            &self.nodes,
            self.coefficients.iter().copied().map(scaled_complex),
        )
    }
}

/// Expand a Newton table without projecting its intermediate coefficients.
/// Both stored scalar coefficients and extended nodal differences use this fold.
pub(super) fn materialize_monomial<C: NumericScalar>(
    nodes: &[C],
    coefficients: impl DoubleEndedIterator<
        Item = Result<ScaledValue<scioxide_core::Complex<C::Real>>, SciOxideError>,
    >,
) -> Result<Vec<C>, SciOxideError> {
    let mut result = try_filled(
        nodes.len() + 1,
        ScaledValue::ZERO,
        scioxide_core::quantity!("newton_monomial_conversion"),
    )?;
    crate::backend::polynomial_basis::newton_to_power_into(
        nodes,
        coefficients,
        &mut result,
        Ok,
        |addend, value, node| addend.sub(value.mul(scaled_complex(node)?)?),
    )?;
    project_coefficients::<C>(&result)
}

type NewtonState<C, A> = (
    <A as Arithmetic<C>>::Value,
    <A as Arithmetic<C>>::Value,
    <A as Arithmetic<C>>::Bound,
);

impl<C: NumericScalar> NewtonBasis<C> {
    #[cold]
    #[inline(never)]
    fn evaluate_extended(&self, point: C) -> Result<(C, C, C::Real), SciOxideError> {
        let (value, derivative, bound) =
            extended(self.evaluate_arithmetic::<Extended, false>(point))?;
        Ok((value.round()?, derivative.round()?, bound.round()?))
    }

    /// Value, derivative and the absolute Newton envelope share every gap.
    /// For moderate binary64 coordinates, gaps have quantum >= 2^-68 and
    /// magnitude <= 2^17. Twelve steps from the shared narrow interval
    /// [2^-128, 2^768] (or zero) keep quantum >= 2^-996 and magnitude
    /// < 2^976, including derivatives and envelopes. No intermediate can underflow or overflow between
    /// checkpoints, even under cancellation. Other inputs check every step.
    #[inline(always)]
    fn evaluate_arithmetic<A: Arithmetic<C>, const BLOCKED: bool>(
        &self,
        point: C,
    ) -> ArithmeticResult<NewtonState<C, A>> {
        let point = A::scalar(point)?;
        let mut value = A::scalar(self.coefficients[self.nodes.len()])?;
        let mut derivative = A::scalar(C::ZERO)?;
        let mut bound = A::magnitude(value)?;
        // The immutable blocked admission already covers every coefficient,
        // including the leading value and its real absolute envelope.
        if !BLOCKED {
            require(A::safe(value) && A::safe_bound(bound))?;
        }
        let mut remaining = 12;
        for (index, (&node, &coefficient)) in
            self.nodes.iter().zip(&self.coefficients).enumerate().rev()
        {
            let gap = A::sub(point, A::scalar(node)?)?;
            let coefficient = A::scalar(coefficient)?;
            if !BLOCKED {
                require(A::safe(gap) && A::safe(coefficient))?;
            }
            let radius = A::magnitude(gap)?;
            derivative = A::mul_add(derivative, gap, value)?;
            value = A::mul_add(value, gap, coefficient)?;
            bound = A::bound_mul_add(bound, radius, A::magnitude(coefficient)?)?;
            if BLOCKED {
                remaining -= 1;
                if remaining == 0 || index == 0 {
                    require(
                        A::checkpoint(value, true)
                            && A::checkpoint(derivative, true)
                            && A::bound_checkpoint(bound, true),
                    )?;
                    remaining = 12;
                }
            } else {
                require(A::safe(value) && A::safe(derivative) && A::safe_bound(bound))?;
            }
        }
        Ok((value, derivative, bound))
    }
}

impl<C: NumericScalar> NewtonBasis<C> {}
