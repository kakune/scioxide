//! Explicit conversions between polynomial bases.
use super::*;

/// Conversion diagnostics, reporting arithmetic failures.
pub(crate) fn conversion_result<C, B>(
    source_coefficients: &[C],
    polynomial: ApproxPolynomial<C, B>,
) -> Result<PolynomialConversion<C, B>, SciOxideError>
where
    C: NumericScalar,
    B: PolynomialBasis<C>,
{
    let source_scale = coefficient_scale(source_coefficients);
    let converted_scale = coefficient_scale(polynomial.basis.coefficients());
    // Conversion sensitivity is coordinate-scale growth, not a measured error.
    let condition_estimate = scale_ratio(converted_scale, source_scale);
    let backward_error = rounding_error::<C::Real>(
        source_coefficients
            .len()
            .max(polynomial.basis.coefficients().len()),
    )?;
    let condition = PolynomialConditionReport {
        coefficient_scale: polynomial.metadata.scale,
        backward_error,
        forward_error_estimate: backward_error * condition_estimate,
        condition_estimate,
    };
    Ok(PolynomialConversion {
        polynomial,
        condition,
    })
}

/// Horner composition in extended range, with one fixed-size output buffer.
/// Descending updates keep the preceding row without per-step allocation.
pub(super) fn compose_scaled_affine<R: RealScalar>(
    coefficients: &[ScaledValue<scioxide_core::Complex<R>>],
    constant: ScaledValue<scioxide_core::Complex<R>>,
    linear: ScaledValue<scioxide_core::Complex<R>>,
) -> Result<Vec<ScaledValue<scioxide_core::Complex<R>>>, SciOxideError> {
    let mut result = try_filled(
        coefficients.len(),
        ScaledValue::ZERO,
        scioxide_core::quantity!("affine_basis_conversion"),
    )?;
    for (degree, &coefficient) in coefficients.iter().rev().enumerate() {
        for power in (1..=degree).rev() {
            result[power] = result[power]
                .mul(constant)?
                .add(result[power - 1].mul(linear)?)?;
        }
        result[0] = result[0].mul(constant)?.add(coefficient)?;
    }
    Ok(result)
}

/// Project only final coefficients; losing a nonzero component is an error.
pub(super) fn project_coefficients<C: NumericScalar>(
    coefficients: &[ScaledValue<scioxide_core::Complex<C::Real>>],
) -> Result<Vec<C>, SciOxideError> {
    crate::util::try_collect(
        coefficients.iter().map(|&coefficient| {
            let value = coefficient.try_value()?;
            Ok(C::from_components(value.re, value.im))
        }),
        scioxide_core::quantity!("converted_coefficients"),
    )
}

impl<C: NumericScalar, B: PolynomialBasis<C>> ApproxPolynomial<C, B> {
    /// Converts to the monomial basis `p(x) = Σ aᵢ xⁱ`.
    ///
    /// Newton and barycentric forms are expanded through the nested Newton
    /// form, Bernstein forms through forward differences of the controls and
    /// the substitution `t = (x − a)/(b − a)`. These take O(n²) operations and
    /// O(n) storage. Intermediate coefficients use extended exponent range, so
    /// only the final coefficients must fit the scalar type. Exact trailing
    /// zeros are dropped. The power basis can be badly conditioned: the
    /// condition estimate of the result reports the growth of the largest
    /// coefficient magnitude, but cancellation in the conversion can lose
    /// further digits.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a final coefficient overflows, or a nonzero
    ///   one underflows to zero.
    /// - `AllocationFailure` if the coefficients cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let conversion = p.to_monomial()?;
    /// let result = conversion.polynomial();
    /// assert!((result.evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// assert!(conversion.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_monomial(&self) -> Result<PolynomialConversion<C, MonomialBasis<C>>, SciOxideError> {
        let coefficients = canonical_coefficients(
            self.basis.materialize_monomial()?,
            scioxide_core::argument!("self"),
        )?;
        conversion_result(
            self.basis.coefficients(),
            ApproxPolynomial::from_basis(MonomialBasis::new(coefficients), self.policy)?,
        )
    }
}
impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Converts to Newton form on `degree + 1` distinct nodes.
    ///
    /// The polynomial is evaluated at the nodes by Horner's rule, and the
    /// Newton coefficients are the divided differences of those values, in
    /// O(n²) operations. Exact trailing zero coefficients are dropped together
    /// with their nodes.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `nodes.len()` is not `degree + 1`.
    /// - `NonFinite` if a node has a NaN or infinite component.
    /// - `InvalidInput` (`Inconsistent`) if two nodes are equal.
    /// - `NumericalInstability` if a value or a final coefficient overflows, or
    ///   a nonzero coefficient underflows to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let conversion = p.to_newton(vec![0.0_f64, 1.0, 2.0])?;
    /// let result = conversion.polynomial();
    /// assert!((result.evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// assert!(conversion.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_newton(
        &self,
        nodes: Vec<C>,
    ) -> Result<PolynomialConversion<C, NewtonBasis<C>>, SciOxideError> {
        let validated = validate_conversion_nodes(&nodes, self.degree() + 1)?;
        let mut values = try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("newton_conversion_samples"),
        )?;
        for node in &nodes {
            values.push(evaluate_monomial(self.basis().coefficients(), *node)?.0);
        }
        let coefficients = canonical_coefficients(
            numeric_divided_differences(validated, values)?.into_native()?,
            scioxide_core::argument!("nodes"),
        )?;
        let mut nodes = nodes;
        nodes.truncate(coefficients.len() - 1);
        conversion_result(
            self.basis().coefficients(),
            ApproxPolynomial::from_basis(NewtonBasis::prepared(coefficients, nodes), self.policy)?,
        )
    }
    /// Converts to Bernstein form of the same degree on `[lower, upper]`.
    ///
    /// With `x = a + (b − a) t`, the power coefficients `âⱼ` in `t` give the
    /// controls `bᵢ = Σⱼ≤ᵢ C(i, j) / C(n, j) âⱼ`. The conversion takes O(n²)
    /// operations and O(n) storage, in extended exponent range. The condition
    /// estimate reports the growth of the largest coordinate magnitude; it
    /// does not account for cancellation in the sums.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower` or `upper` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `lower ≥ upper`.
    /// - `NumericalInstability` if a control coefficient overflows, or a nonzero
    ///   one underflows to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let conversion = p.to_bernstein(0.0, 1.0)?;
    /// let result = conversion.polynomial();
    /// assert!((result.evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// assert!(conversion.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// R. T. Farouki, "The Bernstein polynomial basis: a centennial
    /// retrospective", *Comput. Aided Geom. Design* 29(6), 2012, 379–419.
    pub fn to_bernstein(
        &self,
        lower: C::Real,
        upper: C::Real,
    ) -> Result<PolynomialConversion<C, BernsteinBasis<C>>, SciOxideError> {
        interval(lower, upper)?;
        // Substitute `x = lower + (upper - lower) t` to obtain `sum_j a_j t^j` on [0, 1],
        // then use `t^j = sum_{i >= j} [C(i, j) / C(n, j)] B_{i,n}(t)`, giving
        // `b_i = sum_{j <= i} C(i, j) / C(n, j) a_j`.
        let scaled_lower = scaled_complex(C::from_real(lower))?;
        let width = scaled_complex(C::from_real(upper))?.sub(scaled_lower)?;
        let source = crate::util::try_collect(
            self.basis()
                .coefficients()
                .iter()
                .copied()
                .map(scaled_complex),
            scioxide_core::quantity!("bernstein_conversion_source"),
        )?;
        let power = compose_scaled_affine(&source, scaled_lower, width)?;
        let degree = self.degree();
        let mut controls = try_filled(
            degree + 1,
            ScaledValue::ZERO,
            scioxide_core::quantity!("bernstein_conversion_coefficients"),
        )?;
        for (index, coefficient) in controls.iter_mut().enumerate() {
            let mut weight = Scaled::from_value(C::Real::ONE)?;
            for (power_index, &value) in power.iter().enumerate().take(index + 1) {
                *coefficient = coefficient.add(value.mul_real(weight)?)?;
                if power_index < index {
                    weight = weight
                        .mul(Scaled::from_value(C::Real::try_from_usize_exact(
                            index - power_index,
                        )?)?)?
                        .div(Scaled::from_value(C::Real::try_from_usize_exact(
                            degree - power_index,
                        )?)?)?;
                }
            }
        }
        let coefficients = project_coefficients::<C>(&controls)?;
        conversion_result(
            self.basis().coefficients(),
            ApproxPolynomial::bernstein_with_options(coefficients, lower, upper, self.policy)?,
        )
    }
}
impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Converts to barycentric form on `degree + 1` distinct nodes.
    ///
    /// The nodal values are computed by Horner's rule and the weights as in
    /// [`interpolate_barycentric`](ApproxPolynomial::interpolate_barycentric),
    /// in O(n²) operations.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `nodes.len()` is not `degree + 1`.
    /// - `NonFinite` if a node has a NaN or infinite component.
    /// - `InvalidInput` (`Inconsistent`) if two nodes are equal.
    /// - `NumericalInstability` if a value overflows or a scaled weight
    ///   underflows to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let conversion = p.to_barycentric(vec![0.0_f64, 1.0, 2.0])?;
    /// let result = conversion.polynomial();
    /// assert!((result.evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// assert!(conversion.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_barycentric(
        &self,
        nodes: Vec<C>,
    ) -> Result<PolynomialConversion<C, BarycentricBasis<C>>, SciOxideError> {
        let validated = validate_conversion_nodes(&nodes, self.degree() + 1)?;
        let mut values = try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("barycentric_conversion_samples"),
        )?;
        for &node in &nodes {
            values.push(evaluate_monomial(self.basis().coefficients(), node)?.0);
        }
        let weights = weights_for_nodes(validated)?;
        conversion_result(
            self.basis().coefficients(),
            ApproxPolynomial::from_basis(
                BarycentricBasis {
                    nodes,
                    values,
                    weights,
                },
                self.policy,
            )?,
        )
    }
}
