//! The basis-generic polynomial type and its evaluation.
use super::*;

/// Polynomial with floating-point coefficients stored in an explicit basis.
///
/// `C` is the coefficient type, real (`f32`, `f64`) or complex, and `B` is the
/// basis. Each basis has its own constructors:
///
/// | Basis `B` | Stored coordinates | Constructors |
/// |---|---|---|
/// | [`MonomialBasis`] | `a₀, …, aₙ` of `p(x) = Σ aᵢ xⁱ` | [`monomial`](Self::monomial), [`from_roots`](Self::from_roots), [`from_conjugate_pairs`](Self::from_conjugate_pairs) |
/// | [`NewtonBasis`] | `c₀, …, cₙ` of `p(x) = Σ cᵢ (x − x₀)⋯(x − xᵢ₋₁)` | [`newton`](Self::newton), [`interpolate_newton`](Self::interpolate_newton) |
/// | [`BarycentricBasis`] | values `p(xⱼ)` at `n + 1` distinct nodes | [`interpolate_barycentric`](Self::interpolate_barycentric) |
/// | [`BernsteinBasis`] | control coefficients `b₀, …, bₙ` on `[a, b]` | [`bernstein`](Self::bernstein) |
/// | [`OrthogonalBasis`](crate::OrthogonalBasis) | coefficients of a classical orthogonal expansion | [`orthogonal`](Self::orthogonal) |
///
/// Operations work in the stored basis. [`evaluate`](Self::evaluate) returns
/// the value, the first derivative and a [`PolynomialConditionReport`].
/// Changing the basis is always explicit: the `to_*` methods, such as
/// [`to_monomial`](Self::to_monomial), return a [`PolynomialConversion`] with
/// the converted polynomial and its diagnostics.
///
/// Use the monomial basis for coefficient arithmetic, division and
/// elimination. Interpolate samples in Newton or barycentric form, which avoids
/// expanding the interpolant into powers. Use the Bernstein basis for curves
/// and for bounds from control coefficients, and an orthogonal basis for
/// expansions on an interval.
///
/// Every polynomial carries a [`PolynomialApproximationPolicy`], and binary
/// operations require both operands to carry the same one. The monomial
/// constructors drop exact trailing zeros; no constructor drops small nonzero
/// coefficients. [`degree`](Self::degree) is the number of stored coordinates
/// minus one, and [`classify_coefficients`](Self::classify_coefficients)
/// reports the degree under the policy's tolerance.
///
/// # Examples
///
/// ```
/// use scioxide_poly::ApproxPolynomial;
///
/// // p(x) = 1 + 2x + x² = (1 + x)², so p(0.5) = 2.25 and p′(0.5) = 3.
/// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
/// let y = p.evaluate_with_derivative(0.5)?;
/// assert!((y.value() - 2.25).abs() < 1e-14);
/// assert!((y.derivative() - 3.0).abs() < 1e-14);
///
/// // The same polynomial from its values at x = 0, 1, 2.
/// let q =
///     ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0, 2.0], vec![1.0, 4.0, 9.0])?;
/// assert!((q.evaluate(0.5)? - 2.25).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ApproxPolynomial<C: NumericScalar, B: PolynomialBasis<C>> {
    pub(crate) basis: B,
    pub(super) policy: PolynomialApproximationPolicy<C::Real>,
    coefficient: PhantomData<C>,
    pub(super) metadata: CoefficientMetadata<C::Real>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CoefficientMetadata<R: RealScalar> {
    pub(super) scale: R,
    pub(super) operation_factor: R,
}

impl<C: NumericScalar, B: PolynomialBasis<C>> ApproxPolynomial<C, B> {
    pub(crate) fn from_basis(
        basis: B,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        let coefficients = basis.coefficients();
        let scale = coefficient_scale(coefficients);
        // Validate the stored-coordinate policy even when degree classification
        // uses a different basis or extended-range coefficients.
        policy.tolerance.threshold(scale)?;
        policy::rank_threshold(policy.rank_policy, scale, coefficients.len())?;
        let metadata = CoefficientMetadata {
            scale,
            operation_factor: rounding_error::<C::Real>(coefficients.len())?,
        };
        Ok(Self {
            basis,
            policy,
            coefficient: PhantomData,
            metadata,
        })
    }

    /// Classifies the stored coefficients under this polynomial's policy.
    ///
    /// The report gives the effective degree, the number of significant stored
    /// coordinates and their scale; see [`PolynomialCoefficientReport`]. The
    /// polynomial itself is unchanged. Construction and evaluation never
    /// classify coefficients.
    ///
    /// Monomial, Newton and orthogonal forms classify their own coefficients in
    /// O(n) operations. Barycentric and Bernstein forms first compute Newton or
    /// power coefficients in extended exponent range, which takes O(n²)
    /// operations and O(n) temporary storage.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the temporary coefficients cannot be allocated.
    /// - `NumericalInstability` if the tolerance or rank threshold overflows.
    ///
    /// # Examples
    ///
    /// The values 1, 1, 1 at three nodes stored in barycentric form have
    /// structural degree 2 but describe the constant polynomial 1.
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// let p = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64, 1.0, 2.0], vec![1.0; 3])?;
    /// assert_eq!(p.degree(), 2);
    /// assert_eq!(p.classify_coefficients()?.effective_degree(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn classify_coefficients(
        &self,
    ) -> Result<PolynomialCoefficientReport<C::Real>, SciOxideError> {
        let coefficients = self.basis.coefficients();
        let threshold = policy::rank_threshold(
            self.policy.rank_policy,
            self.metadata.scale,
            coefficients.len(),
        )?;
        Ok(PolynomialCoefficientReport {
            coefficient_scale: self.metadata.scale,
            effective_degree: self.basis.effective_degree(self.policy.tolerance())?,
            effective_rank: coefficients.iter().filter(|c| c.abs() > threshold).count(),
            policy: self.policy,
        })
    }

    /// Returns the basis, which holds the stored coordinates.
    ///
    /// Read the coordinates with [`PolynomialBasis::coefficients`]; the
    /// concrete basis types add accessors such as nodes or the interval.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.basis().coefficients(), &[1.0, 2.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn basis(&self) -> &B {
        &self.basis
    }

    /// Returns the approximation policy of this polynomial.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.policy(), PolynomialApproximationPolicy::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn policy(&self) -> PolynomialApproximationPolicy<C::Real> {
        self.policy
    }

    /// Returns the structural degree: the number of stored coordinates minus one.
    ///
    /// The structural degree counts zero leading coordinates. Use
    /// [`classify_coefficients`](Self::classify_coefficients) for the degree
    /// under the policy's tolerance.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// assert_eq!(p.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn degree(&self) -> usize {
        self.basis.degree()
    }

    /// Returns the scratch length of the batch forms for `points` points.
    ///
    /// [`evaluate_batch_into`](Self::evaluate_batch_into) needs `m + s` entries
    /// and
    /// [`evaluate_batch_with_derivative_into`](Self::evaluate_batch_with_derivative_into)
    /// needs `2m + s`, where `m` is the number of points and `s` is the scratch
    /// of the basis: one entry per control coefficient for the Bernstein basis,
    /// none for the other bases. The returned length `2·points + s` serves both.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length does not fit in `usize`.
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 1.0], 0.0, 1.0)?;
    /// assert_eq!(p.workspace_requirements(4)?.scratch_len(), 11);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        let staged = checked_mul(points, 2, scioxide_core::quantity!("evaluation_scratch"))?;
        Ok(PolynomialScratchRequirements::new(checked_add(
            staged,
            self.basis.scratch_len(),
            scioxide_core::quantity!("evaluation_scratch"),
        )?))
    }

    /// Evaluates the polynomial at `point`.
    ///
    /// Evaluation works in the stored basis, without conversion:
    ///
    /// - monomial: Horner's rule;
    /// - Newton: nested multiplication of the Newton form;
    /// - barycentric: the second (true) barycentric formula;
    /// - Bernstein: de Casteljau's algorithm on `t = (x − a)/(b − a)`;
    /// - orthogonal: Clenshaw's recurrence.
    ///
    /// Each takes O(n) operations for degree `n`, except de Casteljau's
    /// algorithm, which takes O(n²). The kernel is that of
    /// [`evaluate_with_derivative`](Self::evaluate_with_derivative), which also
    /// computes the derivative and the error envelope, and the value is the same
    /// to the last bit. Points outside the interval of a Bernstein or orthogonal
    /// basis are accepted. The Bernstein basis allocates one vector of
    /// `degree + 1` entries; the other bases allocate nothing. For many points
    /// use [`evaluate_batch`](Self::evaluate_batch).
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `point` has a NaN or infinite component.
    /// - `NumericalInstability` if the value, the derivative or the envelope
    ///   overflows.
    /// - `AllocationFailure` if the Bernstein scratch cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// // Controls 1, 2, 4 on [0, 1]: p(t) = (1 − t)² + 4t(1 − t) + 4t², so p(0.5) = 2.25.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 4.0], 0.0, 1.0)?;
    /// assert!((p.evaluate(0.5)? - 2.25).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
    ///   SIAM, 2002, chapter 5.
    /// - J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
    ///   *SIAM Rev.* 46(3), 2004, 501–517.
    #[inline(always)]
    pub fn evaluate(&self, point: C) -> Result<C, SciOxideError> {
        self.evaluation_state(point).map(|(value, _, _)| value)
    }

    /// Evaluates the polynomial and its first derivative at `point`, with error
    /// estimates.
    ///
    /// The method and cost are those of [`evaluate`](Self::evaluate); the
    /// derivative is computed in the same pass. The
    /// [`condition`](PolynomialEvaluation::condition) report divides an
    /// absolute envelope by `|p(x)|`: `Σ |aᵢ| |x|ⁱ` for monomial coefficients
    /// (the condition number of evaluation, Higham §5.1), `Σ |cᵢ| Π |x − xⱼ|` for
    /// Newton coefficients, `Σ |cᵢ| |φᵢ(x)|` for orthogonal expansions, and
    /// `max |bᵢ|` for Bernstein coefficients at `t ∈ [0, 1]`. The barycentric
    /// envelope adds `|p(x)| Λ(x)`, where `Λ` is the Lebesgue function of the
    /// nodes. When native arithmetic would overflow or underflow, the evaluation
    /// restarts in extended exponent range.
    ///
    /// # Errors
    ///
    /// The errors of [`evaluate`](Self::evaluate). A zero value with a nonzero
    /// envelope gives an infinite condition estimate, not an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// // p(x) = 1 + 2x + x², so p(0.5) = 2.25 and p′(0.5) = 3.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.evaluate_with_derivative(0.5)?;
    /// assert!((result.value() - 2.25).abs() < 1e-14);
    /// assert!((result.derivative() - 3.0).abs() < 1e-14);
    /// assert!(result.condition().forward_error_estimate() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[inline(always)]
    pub fn evaluate_with_derivative(
        &self,
        point: C,
    ) -> Result<PolynomialEvaluation<C>, SciOxideError> {
        self.evaluation_from_state(self.evaluation_state(point)?)
    }

    /// Value, derivative and envelope with scratch owned by the call.
    #[inline(always)]
    fn evaluation_state(
        &self,
        point: C,
    ) -> Result<(C, C, super::EvaluationEnvelope<C::Real>), SciOxideError> {
        finite_input(point, "point must have finite components")?;
        let mut scratch = try_filled(
            self.basis.scratch_len(),
            C::ZERO,
            scioxide_core::quantity!("evaluation_scratch"),
        )?;
        self.basis.evaluate_prepared(point, &mut scratch)
    }

    /// Evaluates the polynomial at each of `points` and returns the values.
    ///
    /// Each value equals [`evaluate`](Self::evaluate) at that point to the last
    /// bit. The call allocates the result and the scratch of
    /// [`evaluate_batch_into`](Self::evaluate_batch_into), which it runs on the
    /// calling thread.
    ///
    /// # Errors
    ///
    /// The errors of [`evaluate_batch_into`](Self::evaluate_batch_into) for a
    /// correctly sized output, and `AllocationFailure`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    /// // p(x) = 1 + 2x + 3x².
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0])?;
    /// let values = p.evaluate_batch(&[0.0, 0.5, 1.0])?;
    /// assert!(
    ///     values
    ///         .iter()
    ///         .zip([1.0, 2.75, 6.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-14)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch(&self, points: &[C]) -> Result<Vec<C>, SciOxideError> {
        let mut output = try_filled(
            points.len(),
            C::ZERO,
            scioxide_core::quantity!("evaluation_values"),
        )?;
        let mut scratch = try_filled(
            checked_add(
                points.len(),
                self.basis.scratch_len(),
                scioxide_core::quantity!("evaluation_scratch"),
            )?,
            C::ZERO,
            scioxide_core::quantity!("evaluation_scratch"),
        )?;
        self.evaluate_batch_into(
            points,
            &mut output,
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Evaluates the polynomial at each of `points` and writes the values to
    /// `output`, without allocating.
    ///
    /// The points are evaluated one after another on the calling thread, each
    /// with the kernel of [`evaluate`](Self::evaluate), so every value equals the
    /// single-point result to the last bit. `scratch` needs `m + s` entries for
    /// `m` points and the basis scratch `s` of
    /// [`workspace_requirements`](Self::workspace_requirements). The values are
    /// staged in `scratch` and copied to `output` after the last point, so
    /// `output` is unchanged on error. For a parallel batch over borrowed
    /// orthogonal coefficients, use
    /// [`OrthogonalBasisPlan::evaluate_batch_into`](crate::OrthogonalBasisPlan::evaluate_batch_into).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len() ≠ points.len()` or `scratch` is too
    ///   short.
    /// - `NonFinite` if a point has a NaN or infinite component.
    /// - `NumericalInstability` if a value, derivative or envelope overflows.
    /// - `ExecutionStopped` on cancellation or deadline, polled before each
    ///   point and once more before the copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::ApproxPolynomial;
    /// // Controls 1, 3 on [0, 1] give the line p(x) = 1 + 2x.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// let points = [0.0, 0.5, 1.0];
    /// let mut values = [0.0; 3];
    /// let mut scratch = vec![0.0; p.workspace_requirements(points.len())?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// p.evaluate_batch_into(&points, &mut values, &mut scratch, cx)?;
    /// assert!(
    ///     values
    ///         .iter()
    ///         .zip([1.0, 2.0, 3.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-14)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_into(
        &self,
        points: &[C],
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let count = points.len();
        exact_len(count, output.len())?;
        minimum_len(
            checked_add(
                count,
                self.basis.scratch_len(),
                scioxide_core::quantity!("evaluation_scratch"),
            )?,
            scratch.len(),
        )?;
        validate_slice(points)?;
        let (values, basis_scratch) = scratch.split_at_mut(count);
        for (value, &point) in values.iter_mut().zip(points) {
            context.check()?;
            *value = self.basis.evaluate_prepared(point, basis_scratch)?.0;
        }
        context.check()?;
        output.copy_from_slice(values);
        Ok(())
    }

    /// Evaluates the polynomial and its first derivative at each of `points`
    /// and writes them to `values` and `derivatives`, without allocating.
    ///
    /// This is [`evaluate_batch_into`](Self::evaluate_batch_into) with the
    /// derivative: each derivative equals that of
    /// [`evaluate_with_derivative`](Self::evaluate_with_derivative) at the point
    /// to the last bit. `scratch` needs `2m + s` entries, the length of
    /// [`workspace_requirements`](Self::workspace_requirements). Both outputs are
    /// unchanged on error.
    ///
    /// # Errors
    ///
    /// The errors of [`evaluate_batch_into`](Self::evaluate_batch_into), with
    /// `DimensionMismatch` also for `derivatives`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::ApproxPolynomial;
    /// // p(x) = x², so p′(x) = 2x.
    /// let p = ApproxPolynomial::monomial(vec![0.0_f64, 0.0, 1.0])?;
    /// let points = [1.0, 2.0];
    /// let (mut values, mut slopes) = ([0.0; 2], [0.0; 2]);
    /// let mut scratch = vec![0.0; p.workspace_requirements(points.len())?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// p.evaluate_batch_with_derivative_into(&points, &mut values, &mut slopes, &mut scratch, cx)?;
    /// assert!(
    ///     values
    ///         .iter()
    ///         .zip([1.0, 4.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-14)
    /// );
    /// assert!(
    ///     slopes
    ///         .iter()
    ///         .zip([2.0, 4.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-14)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_with_derivative_into(
        &self,
        points: &[C],
        values: &mut [C],
        derivatives: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let count = points.len();
        exact_len(count, values.len())?;
        exact_len(count, derivatives.len())?;
        let staged = checked_mul(count, 2, scioxide_core::quantity!("evaluation_scratch"))?;
        minimum_len(
            checked_add(
                staged,
                self.basis.scratch_len(),
                scioxide_core::quantity!("evaluation_scratch"),
            )?,
            scratch.len(),
        )?;
        validate_slice(points)?;
        let (staged, basis_scratch) = scratch.split_at_mut(staged);
        let (staged_values, staged_derivatives) = staged.split_at_mut(count);
        for ((value, derivative), &point) in staged_values
            .iter_mut()
            .zip(staged_derivatives.iter_mut())
            .zip(points)
        {
            context.check()?;
            (*value, *derivative, _) = self.basis.evaluate_prepared(point, basis_scratch)?;
        }
        context.check()?;
        values.copy_from_slice(staged_values);
        derivatives.copy_from_slice(staged_derivatives);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn evaluation_from_state(
        &self,
        (value, derivative, envelope): (C, C, super::EvaluationEnvelope<C::Real>),
    ) -> Result<PolynomialEvaluation<C>, SciOxideError> {
        let condition_estimate = envelope.condition(value.abs());
        let backward_error = self.metadata.operation_factor;
        let forward_error_estimate = backward_error * condition_estimate;
        let condition = PolynomialConditionReport {
            coefficient_scale: self.metadata.scale,
            condition_estimate,
            backward_error,
            forward_error_estimate,
        };
        Ok(PolynomialEvaluation {
            value,
            derivative,
            condition,
        })
    }
}
