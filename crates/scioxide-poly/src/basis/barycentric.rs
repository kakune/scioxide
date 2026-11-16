//! Barycentric Lagrange basis: nodal values, nodes and weights.
use super::*;

/// Barycentric Lagrange basis: the values `yⱼ = p(xⱼ)` at distinct nodes `xⱼ`.
///
/// The polynomial is evaluated with the second (true) barycentric formula
///
/// ```text
/// p(x) = Σⱼ wⱼ yⱼ / (x − xⱼ)  /  Σⱼ wⱼ / (x − xⱼ),
/// wⱼ = 1 / Πₖ≠ⱼ (xⱼ − xₖ),
/// ```
///
/// with the weights `wⱼ` scaled so that `max |wⱼ| = 1`. Construct with
/// [`ApproxPolynomial::interpolate_barycentric`] or
/// [`ApproxPolynomial::to_barycentric`]. [`PolynomialBasis::coefficients`]
/// returns the values `yⱼ` in node order.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0], vec![1.0_f64, 3.0])?;
/// assert_eq!(p.degree(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BarycentricBasis<C: NumericScalar> {
    pub(super) values: Vec<C>,
    pub(super) nodes: Vec<C>,
    pub(super) weights: Vec<C>,
}

impl<C: NumericScalar> ApproxPolynomial<C, BarycentricBasis<C>> {
    /// Interpolates real or complex samples `(xⱼ, yⱼ)` in barycentric form.
    ///
    /// The result is the unique polynomial of degree at most `n − 1` through
    /// the `n` samples. The weights are computed from the nodes in O(n²)
    /// operations, in extended exponent range where needed; you cannot supply
    /// them, so the result is always the polynomial interpolant. Each
    /// evaluation then takes O(n) operations and does not allocate. Nodes may
    /// be in any order.
    ///
    /// The second formula is forward stable for node sets with a small
    /// Lebesgue constant, such as Chebyshev points (Higham, 2004). The condition
    /// estimate of [`evaluate`](ApproxPolynomial::evaluate) includes the
    /// Lebesgue function `Λ(x) = Σ |ℓⱼ(x)|`, so clustered or equispaced nodes
    /// show up as large condition estimates.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `nodes` is empty.
    /// - `DimensionMismatch` if `nodes` and `values` differ in length.
    /// - `NonFinite` if a node or value has a NaN or infinite component.
    /// - `InvalidInput` (`Inconsistent`) if two nodes are equal.
    /// - `NumericalInstability` (`Underflow`) if a scaled weight underflows to
    ///   zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // The line through (0, 1) and (1, 3) is p(x) = 1 + 2x, so p(0.5) = 2.
    /// let p = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0], vec![1.0, 3.0])?;
    /// assert!((p.evaluate(0.5)? - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
    ///   *SIAM Rev.* 46(3), 2004, 501–517.
    /// - N. J. Higham, "The numerical stability of barycentric Lagrange
    ///   interpolation", *IMA J. Numer. Anal.* 24(4), 2004, 547–556.
    pub fn interpolate_barycentric(nodes: Vec<C>, values: Vec<C>) -> Result<Self, SciOxideError> {
        Self::interpolate_barycentric_with_options(
            nodes,
            values,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Interpolates samples in barycentric form with the policy `policy`.
    ///
    /// This is the configured form of
    /// [`interpolate_barycentric`](Self::interpolate_barycentric), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`interpolate_barycentric`](Self::interpolate_barycentric), and
    /// `NumericalInstability` (`Overflow`) if a threshold of `policy` overflows at the
    /// scale of the coefficients.
    pub fn interpolate_barycentric_with_options(
        nodes: Vec<C>,
        values: Vec<C>,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let validated = validate_interpolation_data(&nodes, &values)?;
        let weights = weights_for_nodes(validated)?;
        Self::from_basis(
            BarycentricBasis {
                values,
                nodes,
                weights,
            },
            policy,
        )
    }
}

impl<C: NumericScalar> BarycentricBasis<C> {
    /// Returns the interpolation nodes `xⱼ` in the order given.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0], vec![1.0_f64, 3.0])?;
    /// assert_eq!(p.basis().nodes(), &[0.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn nodes(&self) -> &[C] {
        &self.nodes
    }

    /// Returns the barycentric weights `wⱼ`, scaled so that `max |wⱼ| = 1`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0], vec![1.0_f64, 3.0])?;
    /// let weights = p.basis().weights();
    /// assert_eq!(weights.len(), 2);
    /// assert!((weights[0] + weights[1]).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn weights(&self) -> &[C] {
        &self.weights
    }
}

impl<C: NumericScalar> PolynomialBasis<C> for BarycentricBasis<C> {
    fn coefficients(&self) -> &[C] {
        &self.values
    }
}

impl<C: NumericScalar> private::Prepared<C> for BarycentricBasis<C> {
    #[inline(always)]
    fn evaluate_prepared(
        &self,
        point: C,
        _scratch: &mut [C],
    ) -> Result<(C, C, EvaluationEnvelope<C::Real>), SciOxideError> {
        let mut reference = 0;
        let mut nearest = (point - self.nodes[0]).abs();
        for (index, &node) in self.nodes.iter().enumerate().skip(1) {
            let distance = (point - node).abs();
            if distance < nearest {
                nearest = distance;
                reference = index;
            }
        }
        let result = crate::backend::barycentric::evaluate::<C, _, true, true, true>(
            &self.nodes,
            &self.weights,
            |index| self.values[index],
            point,
            reference,
            C::Real::ZERO,
        )?;
        // The second form also rounds its weights and denominator. Relative
        // perturbations of those terms move the value by |p(z)| Λ(z) to first
        // order, where Λ is the Lebesgue function `denominator_condition`;
        // clustered nodes make it dominate the sample sum Σ|ℓ_j f_j|.
        Ok((
            result.value,
            result.derivative,
            EvaluationEnvelope::new(result.numerator_bound, result.denominator_condition),
        ))
    }

    fn effective_degree(
        &self,
        tolerance: scioxide_core::Tolerance<C::Real>,
    ) -> Result<usize, SciOxideError> {
        policy::scaled_effective_degree(&self.newton_coefficients()?, tolerance)
    }

    fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError> {
        let coefficients = self.newton_coefficients()?;
        newton::materialize_monomial(
            &self.nodes[..self.nodes.len() - 1],
            coefficients.into_iter().map(|value| {
                Ok(ScaledValue::from_components(
                    value.real(),
                    value.imaginary(),
                ))
            }),
        )
    }
}

impl<C: NumericScalar> BarycentricBasis<C> {
    fn newton_coefficients(&self) -> Result<Vec<ScaledValue<C>>, SciOxideError> {
        numeric_divided_differences(
            // The private basis can only be constructed with validated nodes.
            ValidatedNodes::prepared(&self.nodes),
            try_copy_slice(
                &self.values,
                scioxide_core::quantity!("divided_differences"),
            )?,
        )?
        .into_scaled()
    }
}
