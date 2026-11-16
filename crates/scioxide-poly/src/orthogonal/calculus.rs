//! Differentiation and integration of orthogonal expansions in their own basis.
//!
//! In physical coordinates, Q_k = A_k Q'_{k+1} + B_k Q'_k + C_k Q'_{k-1}.
//! The constant coordinate of a primitive is arbitrary until anchoring. Thus
//! B_0 and C_1 are omitted, including Jacobi's removable singularity at k=1.
use super::basis::{BasisNormalization, OrthogonalBasis, OrthogonalBasisPlan, lengths};
use crate::PolynomialBasis;
use crate::util::{minimum_len, validate_slice};
use crate::{ApproxPolynomial, OrthogonalFamily, PolynomialScratchRequirements};
use alloc::borrow::Cow;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    ExecutionContext, LinearOperator, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

use crate::admission::{Arithmetic, ArithmeticResult, Extended, Native, extended};
use crate::orthogonal::recurrence;

impl<R: RealScalar> OrthogonalBasisPlan<R> {
    /// Returns the operator that differentiates expansions of exactly this
    /// plan's degree with respect to the physical variable `x`.
    ///
    /// See [`OrthogonalDerivative`] for its shape and cost. Creating it does not
    /// allocate.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::Legendre)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// // P₂′ = 3 P₁.
    /// let derivative = plan.derivative_operator().apply(&[0.0_f64, 0.0, 1.0])?;
    /// assert!(
    ///     derivative
    ///         .iter()
    ///         .zip([0.0_f64, 3.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative_operator(&self) -> OrthogonalDerivative<'_, R> {
        OrthogonalDerivative { plan: self }
    }

    pub(crate) fn resized(&self, degree: usize) -> Result<Self, SciOxideError> {
        let count = checked_add(degree, 1, scioxide_core::quantity!("recurrence_terms"))?;
        let recurrence = self
            .recurrence
            .resized(count, |n| self.spec.normalized_step(n))?;
        Ok(Self {
            spec: self.spec,
            options: self.options,
            recurrence,
            p0: self.p0,
        })
    }

    // Local normalization ratios, shared by native and scaled execution.
    #[inline(always)]
    fn band<C: NumericScalar<Real = R>, A: Arithmetic<C>>(
        &self,
        n: usize,
    ) -> ArithmeticResult<[A::Bound; 3]> {
        let s = A::checked_bound_scalar;
        let add = A::bound_add;
        let sub = A::bound_sub;
        let mul = A::bound_mul;
        let div = A::bound_div;
        let zero = s(R::ZERO)?;
        let one = s(R::ONE)?;
        let two = s(R::TWO)?;
        let k = s(R::try_from_usize_exact(n)?)?;
        let scale = s(self.spec.scale)?;
        let upper = A::checked_bound(div(one, mul(add(k, one)?, s(self.recurrence[n].0)?)?)?)?;
        if n == 0 {
            return Ok([upper, zero, zero]);
        }
        let negative = |value| sub(zero, value);
        let (middle, lower) = match self.spec.family {
            OrthogonalFamily::ChebyshevT => (
                zero,
                if n > 1 {
                    negative(div(one, mul(two, sub(k, one)?)?)?)?
                } else {
                    zero
                },
            ),
            OrthogonalFamily::ChebyshevU => (zero, negative(div(one, mul(two, add(k, one)?)?)?)?),
            OrthogonalFamily::Legendre => (zero, negative(div(one, add(add(k, k)?, one)?)?)?),
            OrthogonalFamily::Gegenbauer { lambda } => {
                (zero, negative(div(one, mul(two, add(k, s(lambda)?)?)?)?)?)
            }
            OrthogonalFamily::Laguerre | OrthogonalFamily::GeneralizedLaguerre { .. } => {
                (one, zero)
            }
            OrthogonalFamily::HermiteH | OrthogonalFamily::HermiteHe => (zero, zero),
            OrthogonalFamily::Jacobi { alpha, beta } => {
                if A::NATIVE {
                    crate::admission::require(super::admission::four_product_values([
                        alpha,
                        beta,
                        R::try_from_usize_exact(n)?,
                    ]))?;
                }
                let (alpha, beta) = (s(alpha)?, s(beta)?);
                let sum = add(alpha, beta)?;
                let q = add(add(k, k)?, sum)?;
                (
                    div(mul(two, sub(alpha, beta)?)?, mul(q, add(q, two)?)?)?,
                    if n > 1 {
                        negative(div(
                            mul(mul(two, add(k, alpha)?)?, add(k, beta)?)?,
                            mul(mul(add(k, sum)?, q)?, add(q, one)?)?,
                        )?)?
                    } else {
                        zero
                    },
                )
            }
        };
        let middle = A::checked_bound(middle)?;
        let lower = A::checked_bound(lower)?;
        let middle = A::checked_bound_mul(scale, middle)?;
        let lower = if n > 1 {
            let classical = s(self.spec.family.recurrence(n - 1)?.0)?;
            let normalized = s(self.recurrence[n - 1].0)?;
            let product = A::checked_bound_mul(scale, lower)?;
            let ratio = A::checked_bound_div(normalized, classical)?;
            A::checked_bound_mul(A::checked_bound_mul(product, ratio)?, scale)?
        } else {
            zero
        };
        Ok([upper, middle, lower])
    }

    fn differentiate<C: NumericScalar<Real = R>>(
        &self,
        input: &[C],
        candidate: &mut [C],
    ) -> Result<(), SciOxideError> {
        if crate::admission::candidate(self.derivative_kernel::<C, Native>(input, candidate))?
            .is_some()
        {
            return Ok(());
        }
        extended(self.derivative_kernel::<C, Extended>(input, candidate))
    }
    fn derivative_kernel<C: NumericScalar<Real = R>, A: Arithmetic<C, Projected = C>>(
        &self,
        input: &[C],
        candidate: &mut [C],
    ) -> ArithmeticResult<()> {
        if A::NATIVE {
            crate::admission::require(crate::admission::native_coefficients(
                input.iter().copied(),
            ))?;
        }
        let zero = A::scalar(C::ZERO)?;
        let mut next = zero;
        let mut next2 = zero;
        candidate.fill(C::ZERO);
        if self.spec.family == OrthogonalFamily::ChebyshevT
            && self.spec.normalization == BasisNormalization::Classical
        {
            // Analytically cancel the integration-band ratios in both representations.
            let inverse = A::checked_bound_div(
                A::checked_bound_scalar(R::ONE)?,
                A::checked_bound_scalar(self.spec.scale)?,
            )?;
            for k in (0..input.len() - 1).rev() {
                let factor = A::checked_bound(A::bound_mul(
                    A::bound_mul(
                        A::bound_scalar(R::try_from_usize_exact(k + 1)?)?,
                        A::bound_scalar(if k == 0 { R::ONE } else { R::TWO })?,
                    )?,
                    inverse,
                )?)?;
                let value = A::checked(A::add(
                    A::mul_bound(A::scalar(input[k + 1])?, factor)?,
                    A::mul_bound(
                        next2,
                        A::bound_scalar(if k == 0 { R::HALF } else { R::ONE })?,
                    )?,
                )?)?;
                candidate[k] = A::project(value)?;
                next2 = next;
                next = value;
            }
            return Ok(());
        }
        let z = A::bound_scalar(R::ZERO)?;
        let (mut middle, mut lower, mut next_lower) = (z, z, z);
        for k in (0..input.len() - 1).rev() {
            let band = self.band::<C, A>(k)?;
            let residual = A::checked(A::sub(
                A::sub(A::scalar(input[k + 1])?, A::mul_bound(next, middle)?)?,
                A::mul_bound(next2, lower)?,
            )?)?;
            let inverse = A::checked_bound(A::bound_mul(
                A::bound_scalar(R::try_from_usize_exact(k + 1)?)?,
                A::bound_scalar(self.recurrence[k].0)?,
            )?)?;
            let value = A::checked(A::mul_bound(residual, inverse)?)?;
            candidate[k] = A::project(value)?;
            next2 = next;
            next = value;
            middle = band[1];
            lower = next_lower;
            next_lower = band[2];
        }
        Ok(())
    }
}

/// Differentiates expansions in an orthogonal basis without leaving the basis.
///
/// The operator maps the `n + 1` coefficients of `Σₖ cₖ Qₖ(x)`, where `n` is the
/// plan's degree, to the coefficients of its derivative with respect to `x` in
/// the same basis. A constant (`n = 0`) gives one zero coefficient, so there are
/// `max(n, 1)` outputs. The derivative coefficients are found by a backward
/// recurrence from the integration relation
/// `Qₖ = Aₖ Q′ₖ₊₁ + Bₖ Q′ₖ + Cₖ Q′ₖ₋₁` of the family, in O(n) time. If a
/// value would overflow or underflow, the computation restarts with a wider
/// exponent range.
///
/// [`LinearOperator::apply_into`] needs `max(n, 1)` scratch entries and leaves
/// the output unchanged on error. Inputs must have exactly `n + 1` entries,
/// including trailing zeros.
///
/// ```
/// use scioxide_core::LinearOperator;
/// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
///
/// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
/// let plan = spec.prepare(2, Default::default())?;
/// let operator = plan.derivative_operator();
/// assert_eq!(LinearOperator::<f64>::rows(&operator), 2);
/// assert_eq!(LinearOperator::<f64>::cols(&operator), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §18.9. <https://dlmf.nist.gov/18.9>
#[derive(Debug, Clone, Copy)]
pub struct OrthogonalDerivative<'a, R: RealScalar> {
    plan: &'a OrthogonalBasisPlan<R>,
}
impl<R: RealScalar> OrthogonalDerivative<'_, R> {
    /// Differentiates the coefficients `input`, in ascending degree, and
    /// returns the derivative coefficients in a new vector.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `degree + 1` entries.
    /// - `NonFinite` if an entry of `input` is NaN or infinite.
    /// - `NumericalInstability` if a derivative coefficient overflows.
    /// - `AllocationFailure` if the output cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let plan = spec.prepare(2, Default::default())?;
    /// let operator = plan.derivative_operator();
    /// // T₂′ = 4 T₁.
    /// let result = operator.apply(&[0.0_f64, 0.0, 1.0])?;
    /// assert!(
    ///     result
    ///         .iter()
    ///         .zip([0.0_f64, 4.0])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply<C: NumericScalar<Real = R>>(&self, input: &[C]) -> Result<Vec<C>, SciOxideError> {
        lengths(<Self as LinearOperator<C>>::cols(self), input.len())?;
        validate_slice(input)?;
        let mut candidate = try_filled(
            <Self as LinearOperator<C>>::rows(self),
            C::ZERO,
            scioxide_core::quantity!("derivative_coefficients"),
        )?;
        self.plan.differentiate(input, &mut candidate)?;
        Ok(candidate)
    }
}
impl<C: NumericScalar> LinearOperator<C> for OrthogonalDerivative<'_, C::Real> {
    fn rows(&self) -> usize {
        self.plan.degree().max(1)
    }
    fn cols(&self) -> usize {
        self.plan.recurrence.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(<Self as LinearOperator<C>>::rows(self))
    }
    fn apply_into(
        &self,
        input: &[C],
        output: &mut [C],
        scratch: &mut [C],
    ) -> Result<(), SciOxideError> {
        let rows = <Self as LinearOperator<C>>::rows(self);
        lengths(<Self as LinearOperator<C>>::cols(self), input.len())?;
        lengths(rows, output.len())?;
        minimum_len(rows, scratch.len())?;
        validate_slice(input)?;
        self.plan.differentiate(input, &mut scratch[..rows])?;
        output.copy_from_slice(&scratch[..rows]);
        Ok(())
    }
}

/// Computes antiderivatives of expansions in an orthogonal basis, in the same
/// basis.
///
/// For an expansion of degree `n` the antiderivative `F` has degree `n + 1` and
/// is expressed in the [`output_plan`](Self::output_plan), which has the same
/// family, normalization and affine map. Each coefficient of `F` is a
/// combination of at most three input coefficients, from the integration
/// relation `Qₖ = Aₖ Q′ₖ₊₁ + Bₖ Q′ₖ + Cₖ Q′ₖ₋₁`, so it costs O(n) time.
/// The constant term is then chosen so that `F(anchor)` has the requested
/// value. [`apply_into`](Self::apply_into) does not allocate.
///
/// ```
/// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
///
/// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
/// let operator = spec.prepare(2, Default::default())?.antiderivative_plan()?;
/// assert_eq!(operator.output_plan().degree(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// NIST DLMF, §18.9. <https://dlmf.nist.gov/18.9>
#[derive(Debug, Clone, PartialEq)]
pub struct OrthogonalAntiderivative<R: RealScalar> {
    pub(super) plan: OrthogonalBasisPlan<R>,
}
impl<R: RealScalar> OrthogonalBasisPlan<R> {
    /// Prepares an [`OrthogonalAntiderivative`] for expansions of this plan's
    /// degree, in O(n) time and storage.
    ///
    /// # Errors
    ///
    /// The errors of [`OrthogonalBasisSpec::prepare`](crate::OrthogonalBasisSpec::prepare)
    /// for one degree higher.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::Legendre)?;
    /// let integral = spec.prepare(1, Default::default())?.antiderivative_plan()?;
    /// let mut output = [0.0_f64; 3];
    /// let mut scratch = vec![0.0_f64; integral.workspace_requirements(())?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// // ∫₀ˣ (1 + 2t) dt = x + x² = 1/3 + P₁ + (2/3) P₂.
    /// integral.apply_into(&[1.0_f64, 2.0], (0.0, 0.0), &mut output, &mut scratch, cx)?;
    /// assert!((output[1] - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn antiderivative_plan(&self) -> Result<OrthogonalAntiderivative<R>, SciOxideError> {
        Ok(OrthogonalAntiderivative {
            plan: self.resized(checked_add(
                self.degree(),
                1,
                scioxide_core::quantity!("antiderivative_degree"),
            )?)?,
        })
    }
}
impl<R: RealScalar> OrthogonalAntiderivative<R> {
    /// Returns the basis plan of the antiderivative, one degree higher than the
    /// source plan.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let operator = spec.prepare(2, Default::default())?.antiderivative_plan()?;
    /// assert_eq!(operator.output_plan().degree(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn output_plan(&self) -> &OrthogonalBasisPlan<R> {
        &self.plan
    }
    /// Returns the scratch length of [`apply_into`](Self::apply_into): one
    /// entry per output coefficient.
    ///
    /// # Errors
    ///
    /// Never fails; the `Result` keeps the signature of the other
    /// `workspace_requirements` methods.
    ///
    /// ```
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::ChebyshevT)?;
    /// let operator = spec.prepare(2, Default::default())?.antiderivative_plan()?;
    /// assert!(operator.workspace_requirements(())?.scratch_len() >= 4);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        Ok(PolynomialScratchRequirements::new(
            self.plan.recurrence.len(),
        ))
    }
    /// Computes the antiderivative `F` with `F′ = Σₖ inputₖ Qₖ` and
    /// `F(anchor) = value`, where `boundary = (anchor, value)`, and writes its
    /// coefficients to `output`.
    ///
    /// `input` has `degree + 1` entries for the source plan's degree, and
    /// `output` has `output_plan().degree() + 1`. `scratch` needs
    /// [`workspace_requirements`](Self::workspace_requirements) entries.
    /// `context` is checked for cancellation before and after the computation.
    /// On error `output` is unchanged; see
    /// [Output on error](scioxide_core#output-on-error).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has the wrong length, or
    ///   `scratch` is too short.
    /// - `NonFinite` if an entry of `input`, `anchor` or `value` is NaN or
    ///   infinite.
    /// - `NumericalInstability` if a coefficient of `F` overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let operator = spec.prepare(2, Default::default())?.antiderivative_plan()?;
    /// let mut output = [0.0_f64; 4];
    /// let mut scratch = vec![0.0_f64; operator.workspace_requirements(())?.scratch_len()];
    /// let cx = ExecutionContext::sequential();
    /// // F′ = 1 and F(0) = 2, so F(x) = x + 2.
    /// operator.apply_into(
    ///     &[1.0_f64, 0.0, 0.0],
    ///     (0.0, 2.0),
    ///     &mut output,
    ///     &mut scratch,
    ///     cx,
    /// )?;
    /// let at_zero = operator
    ///     .output_plan()
    ///     .evaluate_with_derivative(&output, 0.0)?;
    /// assert!((at_zero.value() - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn apply_into<C: NumericScalar<Real = R>>(
        &self,
        input: &[C],
        boundary: (C, C),
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let len = self.plan.recurrence.len();
        lengths(self.plan.degree(), input.len())?;
        lengths(len, output.len())?;
        minimum_len(len, scratch.len())?;
        validate_slice(input)?;
        validate_slice(&[boundary.0, boundary.1])?;
        let candidate = &mut scratch[..len];
        self.integrate(input, boundary, candidate)?;
        context.check()?;
        output.copy_from_slice(candidate);
        Ok(())
    }

    pub(super) fn integrate<C: NumericScalar<Real = R>>(
        &self,
        input: &[C],
        (anchor, value): (C, C),
        candidate: &mut [C],
    ) -> Result<(), SciOxideError> {
        if crate::admission::candidate(self.integral_kernel::<C, Native>(input, candidate))?
            .is_none()
        {
            extended(self.integral_kernel::<C, Extended>(input, candidate))?;
        }
        let at_anchor = recurrence::series::<C, 0, false, true>(
            candidate.iter().copied(),
            anchor,
            self.plan.p0,
            |n| Ok(self.plan.recurrence[n]),
        )?
        .into_scaled()?
        .value;
        candidate[0] = ScaledValue::from_value(value)?
            .sub(at_anchor)?
            .mul_real(Scaled::from_value(R::ONE)?.div(Scaled::from_value(self.plan.p0)?)?)?
            .round()?;
        Ok(())
    }
    fn integral_kernel<C: NumericScalar<Real = R>, A: Arithmetic<C, Projected = C>>(
        &self,
        input: &[C],
        candidate: &mut [C],
    ) -> ArithmeticResult<()> {
        if A::NATIVE {
            crate::admission::require(crate::admission::native_coefficients(
                input.iter().copied(),
            ))?;
        }
        candidate.fill(C::ZERO);
        let zero = A::bound_scalar(R::ZERO)?;
        let band = |n| {
            if n < input.len() {
                self.plan.band::<C, A>(n)
            } else {
                Ok([zero; 3])
            }
        };
        let mut previous = band(0)?;
        let mut current = band(1)?;
        let mut next = band(2)?;
        for (j, out) in candidate.iter_mut().enumerate().skip(1) {
            let mut coefficient = A::mul_bound(A::scalar(input[j - 1])?, previous[0])?;
            if j < input.len() {
                coefficient = A::add(coefficient, A::mul_bound(A::scalar(input[j])?, current[1])?)?;
            }
            if j + 1 < input.len() {
                coefficient = A::add(
                    coefficient,
                    A::mul_bound(A::scalar(input[j + 1])?, next[2])?,
                )?;
            }
            *out = A::project(A::checked(coefficient)?)?;
            previous = current;
            current = next;
            next = band(j + 2)?;
        }
        Ok(())
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, OrthogonalBasis<C>> {
    /// Returns the derivative with respect to `x`, in the same orthogonal basis.
    ///
    /// The method of [`OrthogonalDerivative`] takes O(n) time and storage. For
    /// repeated derivatives of coefficient slices without allocation, use
    /// [`OrthogonalBasisPlan::derivative_operator`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a derivative coefficient overflows.
    /// - `AllocationFailure` if the coefficients cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// // p = 2x² + 2x, so p′(0.5) = 4.
    /// let derivative = p.derivative()?;
    /// assert!((derivative.evaluate(0.5)? - 4.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative(&self) -> Result<Self, SciOxideError> {
        let coefficients = self
            .expansion_plan()?
            .derivative_operator()
            .apply(self.basis().coefficients())?;
        let plan = self.basis().plan.resized(coefficients.len() - 1)?;
        Self::from_basis(OrthogonalBasis::new(coefficients, plan), self.policy())
    }

    /// The prepared plan restricted to this expansion's length. Construction
    /// accepts expansions shorter than the plan's degree; calculus operators are
    /// sized by the coefficients, so a longer plan contributes only its
    /// recurrence prefix and options.
    fn expansion_plan(&self) -> Result<Cow<'_, OrthogonalBasisPlan<C::Real>>, SciOxideError> {
        let plan = &self.basis().plan;
        let degree = self.basis().coefficients().len() - 1;
        Ok(if plan.degree() == degree {
            Cow::Borrowed(plan)
        } else {
            Cow::Owned(plan.resized(degree)?)
        })
    }
    /// Returns the antiderivative `F` with `F′ = p` and `F(anchor) = value`, in
    /// the same orthogonal basis.
    ///
    /// `F` has one degree more than `p` and keeps its family, normalization and
    /// affine map. The integral is unweighted, with respect to `x`. The method of
    /// [`OrthogonalAntiderivative`] takes O(n) time and storage.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `anchor` or `value` is NaN or infinite.
    /// - `NumericalInstability` if a coefficient of `F` overflows.
    /// - The errors of [`OrthogonalBasisPlan::antiderivative_plan`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// let primitive = p.antiderivative(0.0, 2.0)?;
    /// assert!((primitive.evaluate(0.0)? - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn antiderivative(&self, anchor: C, value: C) -> Result<Self, SciOxideError> {
        let prepared = self.expansion_plan()?.antiderivative_plan()?;
        let mut coefficients = try_filled(
            prepared.plan.recurrence.len(),
            C::ZERO,
            scioxide_core::quantity!("antiderivative_coefficients"),
        )?;
        validate_slice(&[anchor, value])?;
        prepared.integrate(
            self.basis().coefficients(),
            (anchor, value),
            &mut coefficients,
        )?;
        Self::from_basis(
            OrthogonalBasis::new(coefficients, prepared.plan),
            self.policy(),
        )
    }

    /// Computes the unweighted integral `∫ p(x) dx` from `lower` to `upper`.
    ///
    /// The endpoints may be complex; the integral of a polynomial does not
    /// depend on the path. The value is `F(upper)`, where `F` is the
    /// [`antiderivative`](Self::antiderivative) with `F(lower) = 0`, in O(n)
    /// time and storage. When `|F(upper)|` is much smaller than the values of
    /// `F` along the way, cancellation limits the relative accuracy. Equal
    /// endpoints give exactly zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower` or `upper` is NaN or infinite.
    /// - `NumericalInstability` if the antiderivative or its value overflows.
    /// - The errors of [`OrthogonalBasisPlan::antiderivative_plan`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
    ///
    /// let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    /// let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 2.0, 1.0], spec)?;
    /// // ∫₀¹ (2x² + 2x) dx = 2/3 + 1 = 5/3.
    /// assert!((p.integral(0.0, 1.0)? - 5.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn integral(&self, lower: C, upper: C) -> Result<C, SciOxideError> {
        validate_slice(&[lower, upper])?;
        if lower == upper {
            return Ok(C::ZERO);
        }
        let primitive = self.antiderivative(lower, C::ZERO)?;
        let basis = primitive.basis();
        super::recurrence::series::<C, 0, false, true>(
            basis.coefficients.iter().copied(),
            upper,
            basis.plan.p0,
            |n| Ok(basis.plan.recurrence[n]),
        )?
        .value()
    }
}
