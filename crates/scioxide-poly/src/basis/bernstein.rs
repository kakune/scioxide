//! Bernstein basis on an interval: construction, calculus, subdivision and bounds.
use super::casteljau::{elevate_curve, split_curve, split_parameter};
use super::validation::interval;
use super::{
    EvaluationEnvelope, absolute_envelope, compose_scaled_affine, private, project_coefficients,
};
use crate::util::{coefficient_scale, round_complex, scaled_complex, validate_slice};
use crate::util::{empty_error, invalid, try_collect};
use crate::{
    ApproxPolynomial, OrthogonalBasis, OrthogonalBasisSpec, PolynomialApproximationPolicy,
    PolynomialBasis, PolynomialConversion,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled, try_with_capacity};
use scioxide_core::{
    AlgebraicScalar, Interval, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

/// Bernstein basis on a finite real interval `[a, b]`: control coefficients
/// `b₀, …, bₙ` of
///
/// ```text
/// p(x) = Σᵢ bᵢ C(n, i) tⁱ (1 − t)ⁿ⁻ⁱ,   t = (x − a) / (b − a),
/// ```
///
/// where `C(n, i)` is the binomial coefficient. On `[a, b]` the values lie
/// between the smallest and largest control coefficient (convex-hull
/// property). Construct with [`ApproxPolynomial::bernstein`] or
/// [`ApproxPolynomial::to_bernstein`]. [`PolynomialBasis::coefficients`]
/// returns `b₀, …, bₙ`.
///
/// ```
/// use scioxide_poly::*;
/// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
/// assert_eq!(p.degree(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct BernsteinBasis<C: NumericScalar> {
    pub(super) coefficients: Vec<C>,
    pub(super) bound: C::Real,
    pub(super) lower: C::Real,
    pub(super) upper: C::Real,
}

impl<C: NumericScalar> ApproxPolynomial<C, BernsteinBasis<C>> {
    /// Creates a polynomial in Bernstein form on `[lower, upper]` from its
    /// control coefficients `b₀, …, bₙ`.
    ///
    /// The degree `n` is `coefficients.len() − 1`. Trailing zeros are kept,
    /// because the degree determines the basis polynomials. Evaluation with
    /// [`evaluate`](ApproxPolynomial::evaluate) uses de Casteljau's algorithm
    /// and needs a scratch slice of `n + 1` entries.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower`, `upper` or a coefficient is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `lower ≥ upper`.
    /// - `InvalidInput` (`Empty`) if `coefficients` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // Controls 1, 3 on [0, 1] give the line p(x) = 1 + 2x, so p(0.5) = 2.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// let result = p.evaluate_with_derivative(0.5)?;
    /// assert!((result.value() - 2.0).abs() < 1e-13);
    /// assert!(result.condition().condition_estimate() >= 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. Farin, *Curves and Surfaces for CAGD: A Practical Guide*, 5th ed.,
    ///   Morgan Kaufmann, 2002, chapters 4–5.
    /// - R. T. Farouki, "The Bernstein polynomial basis: a centennial
    ///   retrospective", *Comput. Aided Geom. Design* 29(6), 2012, 379–419.
    pub fn bernstein(
        coefficients: Vec<C>,
        lower: C::Real,
        upper: C::Real,
    ) -> Result<Self, SciOxideError> {
        Self::bernstein_with_options(
            coefficients,
            lower,
            upper,
            PolynomialApproximationPolicy::default(),
        )
    }

    /// Creates a polynomial in Bernstein form on `[lower, upper]` with the policy
    /// `policy`.
    ///
    /// This is the configured form of [`bernstein`](Self::bernstein), which uses
    /// [`PolynomialApproximationPolicy::default`]; the result carries `policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`bernstein`](Self::bernstein), and `NumericalInstability`
    /// (`Overflow`) if a threshold of `policy` overflows at the scale of the
    /// coefficients.
    pub fn bernstein_with_options(
        coefficients: Vec<C>,
        lower: C::Real,
        upper: C::Real,
        policy: PolynomialApproximationPolicy<C::Real>,
    ) -> Result<Self, SciOxideError> {
        interval(lower, upper)?;
        if coefficients.is_empty() {
            return Err(empty_error(scioxide_core::argument!("coefficients")));
        }
        validate_slice(&coefficients)?;
        Self::from_basis(
            BernsteinBasis {
                bound: coefficient_scale(&coefficients),
                coefficients,
                lower,
                upper,
            },
            policy,
        )
    }
}

impl<C: NumericScalar> BernsteinBasis<C> {
    /// Unit-interval power coefficients: binomial(n,k) * Delta^k b_0.
    fn unit_power_coefficients(
        &self,
    ) -> Result<Vec<ScaledValue<scioxide_core::Complex<C::Real>>>, SciOxideError> {
        let mut differences = try_with_capacity(
            self.coefficients.len(),
            scioxide_core::quantity!("bernstein_degree"),
        )?;
        for &value in &self.coefficients {
            differences.push(scaled_complex(value)?);
        }
        let mut power = try_filled(
            differences.len(),
            ScaledValue::ZERO,
            scioxide_core::quantity!("bernstein_degree_coefficients"),
        )?;
        crate::backend::polynomial_basis::bernstein_to_power_into(
            &mut differences,
            &mut power,
            Scaled::from_value(C::Real::ONE)?,
            ScaledValue::sub,
            ScaledValue::mul_real,
            |binomial, numerator, denominator| {
                binomial
                    .mul(Scaled::from_value(C::Real::try_from_usize_exact(
                        numerator,
                    )?)?)?
                    .div(Scaled::from_value(C::Real::try_from_usize_exact(
                        denominator,
                    )?)?)
            },
        )?;
        Ok(power)
    }

    /// Returns the lower interval endpoint `a`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// assert_eq!(p.basis().lower(), 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn lower(&self) -> C::Real {
        self.lower
    }

    /// Returns the upper interval endpoint `b`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// assert_eq!(p.basis().upper(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn upper(&self) -> C::Real {
        self.upper
    }
}

impl<C: NumericScalar> PolynomialBasis<C> for BernsteinBasis<C> {
    fn coefficients(&self) -> &[C] {
        &self.coefficients
    }
}

impl<C: NumericScalar> private::Prepared<C> for BernsteinBasis<C> {
    fn effective_degree(
        &self,
        tolerance: scioxide_core::Tolerance<C::Real>,
    ) -> Result<usize, SciOxideError> {
        super::policy::scaled_effective_degree(&self.unit_power_coefficients()?, tolerance)
    }

    fn scratch_len(&self) -> usize {
        self.coefficients.len()
    }
    fn evaluate_prepared(
        &self,
        point: C,
        scratch: &mut [C],
    ) -> Result<(C, C, EvaluationEnvelope<C::Real>), SciOxideError> {
        let (value, derivative, t) = super::casteljau::value_and_derivative(
            &self.coefficients,
            point,
            scioxide_core::Interval::new(self.lower, self.upper)?,
            scratch,
        )?;
        let bound = if t.imaginary_component() == C::Real::ZERO
            && t.real_component() >= C::Real::ZERO
            && t.real_component() <= C::Real::ONE
        {
            self.bound
        } else {
            let mut growth = scaled_complex(t)?.magnitude()?.add(
                scaled_complex(C::ONE)?
                    .sub(scaled_complex(t)?)?
                    .magnitude()?,
            )?;
            let mut bound = Scaled::from_value(self.bound)?;
            let mut degree = self.degree();
            while degree != 0 {
                if degree & 1 != 0 {
                    bound = bound.mul(growth)?;
                }
                degree >>= 1;
                if degree != 0 {
                    growth = growth.square()?;
                }
            }
            bound.round()?
        };
        Ok(absolute_envelope((value, derivative, bound)))
    }

    fn materialize_monomial(&self) -> Result<Vec<C>, SciOxideError> {
        let power = self.unit_power_coefficients()?;
        let lower = scaled_complex(C::from_real(self.lower))?;
        let width = scaled_complex(C::from_real(self.upper))?.sub(lower)?;
        let linear = scaled_complex(C::ONE)?.div(width)?;
        let constant = scaled_complex(C::ZERO)?.sub(lower)?.div(width)?;
        project_coefficients::<C>(&compose_scaled_affine(&power, constant, linear)?)
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, crate::BernsteinBasis<C>> {
    /// Returns the derivative in Bernstein form on the same interval.
    ///
    /// The derivative of degree `n − 1` has controls
    /// `n (bᵢ₊₁ − bᵢ) / (b − a)`, computed in O(n) operations. The derivative of
    /// a constant is the zero constant.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a control coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // p(x) = 1 + 2x + x² on [0, 1] (controls 1, 2, 4), so p′(0.5) = 3.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 4.0], 0.0, 1.0)?;
    /// let derivative = p.derivative()?;
    /// assert!((derivative.evaluate(0.5)? - 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative(&self) -> Result<Self, SciOxideError> {
        let mut coefficients = try_filled(
            self.degree().max(1),
            C::ZERO,
            scioxide_core::quantity!("derivative_coefficients"),
        )?;
        bernstein_derivative_coefficients_into(
            self.basis().coefficients(),
            self.basis().lower(),
            self.basis().upper(),
            &mut coefficients,
        )?;
        Self::bernstein_with_options(
            coefficients,
            self.basis().lower(),
            self.basis().upper(),
            self.policy(),
        )
    }
    /// Returns the antiderivative `F` in Bernstein form with `F(anchor) = value`.
    ///
    /// `F` has degree `n + 1` on the same interval, with controls
    /// `d₀ = 0` and `dᵢ₊₁ = dᵢ + (b − a) bᵢ / (n + 1)`, shifted by a constant so
    /// that `F(anchor) = value`. The controls cost O(n) operations and the
    /// evaluation at `anchor` costs O(n²). `anchor` may lie outside `[a, b]`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `anchor` or `value` is NaN or infinite.
    /// - `NumericalInstability` if a control coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 4.0], 0.0, 1.0)?;
    /// let primitive = p.antiderivative(0.0, 2.0)?;
    /// assert!((primitive.evaluate(0.0)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn antiderivative(&self, anchor: C, value: C) -> Result<Self, SciOxideError> {
        validate_slice(&[anchor, value])?;
        let degree = checked_add(
            self.degree(),
            1,
            scioxide_core::quantity!("antiderivative_degree"),
        )?;
        // Bernstein integration: with `N = n + 1`, the primitive has control points
        // `d_0 = 0`, `d_{k+1} = d_k + (upper - lower) / N * c_k`. Bernstein bases form
        // a partition of unity, so adding a constant to every control point adds that
        // constant to the polynomial; this enforces the anchor condition below.
        let factor = scaled_complex(C::from_real(self.basis().upper()))?
            .sub(scaled_complex(C::from_real(self.basis().lower()))?)?
            .div(ScaledValue::from_value(scioxide_core::Complex::new(
                C::Real::try_from_usize_exact(degree)?,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))?)?;
        let mut coefficients = try_with_capacity(
            checked_add(
                degree,
                1,
                scioxide_core::quantity!("antiderivative_coefficients"),
            )?,
            scioxide_core::quantity!("antiderivative_coefficients"),
        )?;
        coefficients.push(C::ZERO);
        let mut sum = ScaledValue::ZERO;
        for &coefficient in self.basis().coefficients() {
            sum = sum.add(scaled_complex(coefficient)?.mul(factor)?)?;
            coefficients.push(round_complex(sum)?);
        }
        let primitive = Self::bernstein_with_options(
            coefficients,
            self.basis().lower(),
            self.basis().upper(),
            self.policy(),
        )?;
        let shift = scaled_complex(value)?.sub(scaled_complex(primitive.evaluate(anchor)?)?)?;
        let mut coefficients = try_with_capacity(
            degree + 1,
            scioxide_core::quantity!("antiderivative_coefficients"),
        )?;
        for &coefficient in primitive.basis().coefficients() {
            coefficients.push(round_complex(scaled_complex(coefficient)?.add(shift)?)?);
        }
        Self::bernstein_with_options(
            coefficients,
            self.basis().lower(),
            self.basis().upper(),
            self.policy(),
        )
    }
    /// Returns the definite integral `∫ p(x) dx` from `lower` to `upper`.
    ///
    /// The integral is `F(upper) − F(lower)` for the antiderivative of
    /// [`antiderivative`](Self::antiderivative), in O(n²) operations. The limits
    /// may lie outside the interval of the basis; equal limits give zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a limit is NaN or infinite.
    /// - `NumericalInstability` if a control coefficient or the value
    ///   overflows.
    /// - `AllocationFailure` if the antiderivative cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // ∫₀¹ (1 + 2x + x²) dx = 1 + 1 + 1/3 = 7/3.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 4.0], 0.0, 1.0)?;
    /// assert!((p.integral(0.0, 1.0)? - 7.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integral(&self, lower: C, upper: C) -> Result<C, SciOxideError> {
        validate_slice(&[lower, upper])?;
        if lower == upper {
            return Ok(C::ZERO);
        }
        self.antiderivative(lower, C::ZERO)?.evaluate(upper)
    }
    /// Converts to an orthogonal basis described by `target`.
    ///
    /// The conversion goes through the monomial basis, which can be
    /// ill-conditioned at high degree. The destination basis is prepared with
    /// the default [`PolynomialExecutionOptions`](crate::PolynomialExecutionOptions).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`to_monomial`](ApproxPolynomial::to_monomial) and
    /// of the monomial-to-orthogonal conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let spec = OrthogonalBasisSpec::new(
    ///     OrthogonalFamily::ChebyshevT,
    ///     BasisNormalization::Classical,
    ///     0.0_f64,
    ///     1.0,
    /// )?;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 2.0, 4.0], 0.0, 1.0)?;
    /// let result = p.to_orthogonal(spec)?;
    /// assert!((result.polynomial().evaluate(0.5)? - 2.25).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_orthogonal(
        &self,
        target: OrthogonalBasisSpec<C::Real>,
    ) -> Result<PolynomialConversion<C, OrthogonalBasis<C>>, SciOxideError> {
        self.to_monomial()?.polynomial().to_orthogonal(target)
    }
}

/// Bernstein derivative control points.
pub(crate) fn bernstein_derivative_coefficients_into<C: NumericScalar>(
    coefficients: &[C],
    lower: C::Real,
    upper: C::Real,
    output: &mut [C],
) -> Result<(), SciOxideError> {
    if coefficients.len() == 1 {
        output[0] = C::ZERO;
        return Ok(());
    }
    let degree = coefficients.len() - 1;
    // `d/dx sum b_i B_{i,n} = n / (upper - lower) * sum (b_{i+1} - b_i) B_{i,n-1}`.
    let scale = ScaledValue::from_value(scioxide_core::Complex::new(
        C::Real::try_from_usize_exact(degree)?,
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
    ))?
    .div(
        ScaledValue::from_value(scioxide_core::Complex::new(
            upper,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        ))?
        .sub(ScaledValue::from_value(scioxide_core::Complex::new(
            lower,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        ))?)?,
    )?;
    for (value, pair) in output.iter_mut().zip(coefficients.windows(2)) {
        *value = round_complex(
            scaled_complex(pair[1])?
                .sub(scaled_complex(pair[0])?)?
                .mul(scale)?,
        )?;
    }
    Ok(())
}

/// Coordinate-wise range of the control coefficients of a Bernstein form.
///
/// By the convex-hull property, each coordinate of the polynomial over its
/// interval or parameter box lies in the corresponding interval. The bounds
/// are computed in floating point from rounded control coefficients, so they
/// are estimates, not rigorous enclosures. Subdivision and degree elevation
/// tighten them.
///
/// ```
/// use scioxide_poly::*;
/// let tensor = TensorBernstein::new(
///     &[1],
///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
///     1,
///     vec![1.0_f64, 3.0],
/// )?;
/// let bounds = tensor.control_bounds()?;
/// assert_eq!(bounds.coordinates().len(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct BernsteinBounds<R: RealScalar> {
    coordinates: Vec<Interval<R>>,
}
impl<R: RealScalar> BernsteinBounds<R> {
    /// Returns one interval `[min bᵢ, max bᵢ]` per output coordinate.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let bounds = tensor.control_bounds()?;
    /// assert_eq!(
    ///     bounds.coordinates()[0],
    ///     scioxide_core::Interval::new(1.0, 3.0)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coordinates(&self) -> &[Interval<R>] {
        &self.coordinates
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, BernsteinBasis<C>> {
    /// Splits the polynomial at `point` into Bernstein forms on `[a, point]`
    /// and `[point, b]`.
    ///
    /// Both pieces represent the same polynomial. Their controls come from one
    /// run of de Casteljau's algorithm, in O(n²) operations.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `point` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) unless `a < point < b`.
    /// - `NumericalInstability` if a control coefficient overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// // p(x) = 1 + 2x on [0, 1]; both halves still evaluate to 1 + 2x.
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// let (left, right) = p.subdivide(0.5)?;
    /// assert!((left.evaluate(0.25)? - 1.5).abs() < 1e-13);
    /// assert!((right.evaluate(0.75)? - 2.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn subdivide(&self, point: C::Real) -> Result<(Self, Self), SciOxideError> {
        let b = self.basis();
        let t = split_parameter::<C>(point, Interval::new(b.lower(), b.upper())?)?;
        let (left, right) = split_curve(b.coefficients(), t)?;
        Ok((
            Self::bernstein_with_options(left, b.lower(), point, self.policy())?,
            Self::bernstein_with_options(right, point, b.upper(), self.policy())?,
        ))
    }
    /// Returns the same polynomial in the Bernstein basis of degree `degree`,
    /// on the same interval.
    ///
    /// Each step from degree `m` to `m + 1` forms the new controls as convex
    /// combinations of neighboring controls, so raising the degree to `m`
    /// takes O(m²) operations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `degree` is below the current degree.
    /// - `SizeOverflow` if `degree + 1` overflows `usize`.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// let result = p.elevate_degree(3)?;
    /// assert!((result.evaluate(0.5)? - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn elevate_degree(&self, degree: usize) -> Result<Self, SciOxideError> {
        let b = self.basis();
        let target = checked_add(degree, 1, scioxide_core::quantity!("axis_controls"))?;
        if target < b.coefficients().len() {
            return Err(invalid(
                scioxide_core::argument!("degree"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Bernstein degree reduction is unsupported",
            ));
        }
        let mut coefficients = try_filled(
            target,
            C::ZERO,
            scioxide_core::quantity!("elevated_controls"),
        )?;
        coefficients[..b.coefficients().len()].copy_from_slice(b.coefficients());
        elevate_curve(&mut coefficients, b.coefficients().len())?;
        Self::bernstein_with_options(coefficients, b.lower(), b.upper(), self.policy())
    }
}
impl<R: RealScalar> ApproxPolynomial<R, BernsteinBasis<R>> {
    /// Returns the range `[min bᵢ, max bᵢ]` of the control coefficients.
    ///
    /// By the convex-hull property, `p(x)` lies in this range for `x` in
    /// `[a, b]`; see [`BernsteinBounds`] for the limits of this estimate.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
    /// let bounds = p.control_bounds()?;
    /// assert!((bounds.coordinates()[0].lower() - 1.0).abs() < 1e-13);
    /// assert!((bounds.coordinates()[0].upper() - 3.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn control_bounds(&self) -> Result<BernsteinBounds<R>, SciOxideError> {
        control_bounds(self.basis().coefficients(), 1)
    }
}

pub(crate) fn control_bounds<R: RealScalar>(
    coefficients: &[R],
    components: usize,
) -> Result<BernsteinBounds<R>, SciOxideError> {
    let coordinates = try_collect(
        (0..components).map(|component| {
            let first = coefficients[component];
            // Convex-hull property: over its domain a Bernstein polynomial lies within
            // the range of its control coefficients, so these per-component extrema
            // bound the patch (a cheap, often loose, enclosure).
            let (lo, hi) = coefficients
                .iter()
                .skip(component)
                .step_by(components)
                .fold((first, first), |(lo, hi), &c| (lo.min(c), hi.max(c)));
            Interval::new(lo, hi)
        }),
        scioxide_core::quantity!("components"),
    )?;
    Ok(BernsteinBounds { coordinates })
}
