//! Division with remainder of floating-point monomial polynomials.
use crate::PolynomialBasis;
use crate::admission::native_value as safe;
use crate::util::{coefficient_scale, minimum_len, validate_slice};
use crate::{
    ApproxPolynomial, MonomialBasis, PolynomialConditionReport, PolynomialOperationPlan,
    PolynomialOperationRequest, PolynomialWorkspace,
};
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, InvalidInputReason, NumericScalar, Scaled, ScaledValue,
    SciOxideError,
};

type Polynomial<C> = ApproxPolynomial<C, MonomialBasis<C>>;

/// Holds the quotient `q`, the remainder `r` and the diagnostics of
/// `a = q·b + r`, as returned by [`ApproxPolynomial::div_rem`].
///
/// The remainder has degree below that of the divisor; a zero remainder is
/// stored as the single coefficient `[0]`.
///
/// # Examples
///
/// ```
/// use scioxide_poly::ApproxPolynomial;
///
/// // x² − 1 = (x + 1)(x − 1) + 0.
/// let a = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
/// let b = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
/// let result = a.div_rem(&b)?;
/// assert!(result.remainder().is_zero());
/// assert!(result.condition().backward_error() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct PolynomialDivision<C: NumericScalar> {
    quotient: Polynomial<C>,
    remainder: Polynomial<C>,
    condition: PolynomialConditionReport<C::Real>,
}
impl<C: NumericScalar> PolynomialDivision<C> {
    /// Returns the quotient `q`, without trailing zero coefficients.
    ///
    /// ```
    /// use scioxide_poly::{ApproxPolynomial, PolynomialBasis};
    ///
    /// // (x² − 1) / (x − 1) = 1 + x.
    /// let a = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
    /// let result = a.div_rem(&b)?;
    /// assert_eq!(result.quotient().basis().coefficients(), &[1.0, 1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn quotient(&self) -> &ApproxPolynomial<C, MonomialBasis<C>> {
        &self.quotient
    }
    /// Returns the remainder `r`, without trailing zero coefficients.
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// let a = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
    /// let result = a.div_rem(&b)?;
    /// assert!(result.remainder().is_zero());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn remainder(&self) -> &ApproxPolynomial<C, MonomialBasis<C>> {
        &self.remainder
    }
    /// Returns the diagnostics of the division.
    ///
    /// The backward error and condition estimate are defined on
    /// [`PolynomialOperationPlan::div_rem_into`].
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// let a = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
    /// let result = a.div_rem(&b)?;
    /// assert!(result.condition().backward_error() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition(&self) -> PolynomialConditionReport<C::Real> {
        self.condition
    }
}

impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Divides `self` by `divisor` with remainder, `self = q·divisor + r`, with
    /// `deg r < deg divisor`.
    ///
    /// Both polynomials must have the same approximation policy. The policy's
    /// tolerance does not change the quotient or remainder: only exact trailing
    /// zeros are removed. The method is long division; see
    /// [`PolynomialOperationPlan::div_rem_into`] for the cost, the error
    /// diagnostics and a form that reuses storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the divisor is zero or the two policies differ.
    /// - `NumericalInstability` if a quotient or remainder coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // x² − 1 = (x + 1)(x − 1).
    /// let a = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0])?;
    /// let b = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
    /// let result = a.div_rem(&b)?;
    /// assert!((result.quotient().evaluate(2.0)? - 3.0).abs() < 1e-14);
    /// assert!(result.remainder().is_zero());
    /// assert!(result.condition().backward_error() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn div_rem(&self, divisor: &Self) -> Result<PolynomialDivision<C>, SciOxideError> {
        crate::util::same_policy(
            self.policy(),
            divisor.policy(),
            scioxide_core::argument!("divisor.policy"),
        )?;
        let a = self.basis().coefficients();
        let b = divisor.basis().coefficients();
        let (qlen, rlen) = output_lengths(a, b)?;
        let request = PolynomialOperationRequest::Divide {
            dividend_len: a.len(),
            divisor_len: b.len(),
        };
        let plan = PolynomialOperationPlan::new(&[request])?;
        let mut workspace = plan.workspace(&[request])?;
        let mut quotient =
            try_filled(qlen, C::ZERO, scioxide_core::quantity!("division_quotient"))?;
        let mut remainder = try_filled(
            rlen,
            C::ZERO,
            scioxide_core::quantity!("division_remainder"),
        )?;
        let condition = plan.div_rem_into(
            a,
            b,
            &mut quotient,
            &mut remainder,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(PolynomialDivision {
            quotient: Self::monomial_with_options(quotient, self.policy())?,
            remainder: Self::monomial_with_options(remainder, self.policy())?,
            condition,
        })
    }
}

impl<C: NumericScalar> PolynomialOperationPlan<C> {
    /// Divides `a` by `b` with remainder, `a = q·b + r`, and writes `q` to
    /// `quotient` and `r` to `remainder`.
    ///
    /// `a` and `b` hold ascending monomial coefficients. Each must be nonempty
    /// and have no exact zero as its last coefficient, except the zero
    /// polynomial `[0]`. `quotient` must have `max(1, a.len() + 1 − b.len())`
    /// entries and `remainder` must have `max(1, min(a.len(), b.len() − 1))`;
    /// both keep padding zeros.
    ///
    /// The method is classical long division, `O(a.len() · b.len())` operations.
    /// If an operand or an intermediate value comes close to overflow or
    /// underflow, the whole division is repeated in extended-exponent arithmetic
    /// and each coefficient is rounded once at the end.
    ///
    /// The returned report gives two diagnostics, both estimates and not bounds.
    /// The backward error is
    ///
    /// ```text
    /// ‖a − (q·b + r)‖₁ / (‖a‖₁ + ‖q‖₁ ‖b‖₁ + ‖r‖₁),
    /// ```
    ///
    /// evaluated in extended-exponent arithmetic, and zero when the denominator
    /// is zero. Here `‖·‖₁` is the sum of the absolute values of the
    /// coefficients. The condition estimate propagates first-order coefficient
    /// perturbations `|δa| ≤ u|a|` and `|δb| ≤ u|b|`, with `u` the unit
    /// roundoff, through the long division, and divides the resulting bound on
    /// `|δq| + |δr|` by `‖q‖₁ + ‖r‖₁`. It is at least one and at most the
    /// largest finite value of the scalar type.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `a` or `b` is empty or ends in an exact zero, if `b` is
    ///   zero, or if the shape `Divide { dividend_len: a.len(), divisor_len: b.len() }`
    ///   was not prepared.
    /// - `DimensionMismatch` if `quotient` or `remainder` has the wrong length,
    ///   or `workspace` is too small.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a quotient or remainder coefficient overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // (x² − 1) / (x − 1) = 1 + x, remainder 0.
    /// let request = PolynomialOperationRequest::Divide {
    ///     dividend_len: 3,
    ///     divisor_len: 2,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let mut ws = plan.workspace(plan.requests())?;
    /// let (a, b, mut q, mut r) = ([-1.0_f64, 0.0, 1.0], [-1.0, 1.0], [0.0; 2], [0.0]);
    /// let cx = ExecutionContext::sequential();
    /// let report = plan.div_rem_into(&a, &b, &mut q, &mut r, &mut ws, cx)?;
    /// assert!(q.iter().all(|x| (x - 1.0).abs() < 1e-13) && r[0].abs() < 1e-13);
    /// assert!(report.backward_error() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn div_rem_into(
        &self,
        a: &[C],
        b: &[C],
        quotient: &mut [C],
        remainder: &mut [C],
        workspace: &mut PolynomialWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
        context.check()?;
        let (qlen, rlen) = output_lengths(a, b)?;
        for (expected, actual) in [(qlen, quotient.len()), (rlen, remainder.len())] {
            if expected != actual {
                return Err(SciOxideError::dimension_mismatch(expected, actual));
            }
        }
        self.admit(
            workspace,
            PolynomialOperationRequest::Divide {
                dividend_len: a.len(),
                divisor_len: b.len(),
            },
        )?;
        let (q, rest) = workspace.coefficients.split_at_mut(qlen);
        let r = &mut rest[..a.len()];
        division_into(a, b, q, r, &mut workspace.scaled, || context.check())?;
        let condition = division_condition(a, b, q, &r[..rlen], &mut workspace.scaled, || {
            context.check()
        })?;
        context.check()?;
        quotient.copy_from_slice(q);
        remainder.copy_from_slice(&r[..rlen]);
        Ok(condition)
    }
}

fn output_lengths<C: NumericScalar>(a: &[C], b: &[C]) -> Result<(usize, usize), SciOxideError> {
    validate_slice(a)?;
    validate_slice(b)?;
    for (values, argument) in [
        (a, scioxide_core::argument!("dividend")),
        (b, scioxide_core::argument!("divisor")),
    ] {
        if values.is_empty() {
            return Err(crate::util::empty_error(argument));
        }
        if values.len() > 1 && values[values.len() - 1] == C::ZERO {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::Inconsistent,
                "canonical coefficients have no exact trailing zeros",
            ));
        }
    }
    if b.len() == 1 && b[0] == C::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("divisor"),
            InvalidInputReason::OutOfRange,
            "divisor must be nonzero",
        ));
    }
    Ok((
        a.len().saturating_sub(b.len() - 1).max(1),
        a.len().min(b.len() - 1).max(1),
    ))
}
pub(crate) fn scratch_len(a: usize, b: usize) -> Result<usize, SciOxideError> {
    checked_add(a, b, scioxide_core::quantity!("division_scratch"))
}

/// Divides `a` by `b` into `q` and `r`. The inputs are nonempty without exact
/// trailing zeros, and the output and scratch slices are long enough.
pub(crate) fn division_into<C: NumericScalar>(
    a: &[C],
    b: &[C],
    q: &mut [C],
    r: &mut [C],
    scratch: &mut [ScaledValue<C>],
    mut check: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    q.fill(C::ZERO);
    r.fill(C::ZERO);
    r[..a.len()].copy_from_slice(a);
    if a.len() < b.len() {
        return Ok(());
    }
    let degree = b.len() - 1;
    let mut native = a.iter().chain(b).copied().all(safe);
    if native {
        'division: for shift in (0..=a.len() - b.len()).rev() {
            check()?;
            let factor = r[degree + shift] / b[degree];
            if !safe(factor) {
                native = false;
                break;
            }
            q[shift] = factor;
            for (index, &coefficient) in b[..degree].iter().enumerate() {
                let product = factor * coefficient;
                let next = r[index + shift] - product;
                if !safe(product) || !safe(next) {
                    native = false;
                    break 'division;
                }
                r[index + shift] = next;
            }
            r[degree + shift] = C::ZERO;
        }
    }
    if native {
        return Ok(());
    }
    minimum_len(a.len() + b.len(), scratch.len())?;
    let (extended, divisor) = scratch.split_at_mut(a.len());
    for (out, &value) in extended.iter_mut().zip(a) {
        *out = ScaledValue::from_value(value)?;
    }
    for (out, &value) in divisor.iter_mut().zip(b) {
        *out = ScaledValue::from_value(value)?;
    }
    let leading = divisor[degree];
    for shift in (0..=a.len() - b.len()).rev() {
        check()?;
        let factor = extended[degree + shift].div(leading)?;
        q[shift] = factor.round()?;
        for (index, &coefficient) in divisor[..degree].iter().enumerate() {
            extended[index + shift] = extended[index + shift].sub(factor.mul(coefficient)?)?;
        }
        extended[degree + shift] = ScaledValue::ZERO;
    }
    for (out, value) in r.iter_mut().zip(extended) {
        *out = value.round()?;
    }
    Ok(())
}

fn norm<C: NumericScalar>(values: &[C]) -> Result<Scaled<C::Real>, SciOxideError> {
    values.iter().try_fold(Scaled::ZERO, |sum, &value| {
        sum.add(ScaledValue::from_value(value)?.magnitude()?)
    })
}
fn real<C: NumericScalar>(value: Scaled<C::Real>) -> ScaledValue<C> {
    ScaledValue::from_components(value, Scaled::ZERO)
}
fn division_condition<C: NumericScalar>(
    a: &[C],
    b: &[C],
    q: &[C],
    r: &[C],
    scratch: &mut [ScaledValue<C>],
    mut check: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<PolynomialConditionReport<C::Real>, SciOxideError> {
    let mut residual = Scaled::ZERO;
    let (cached_quotient, cached_divisor) = scratch.split_at_mut(q.len());
    for (out, &value) in cached_quotient.iter_mut().zip(q) {
        *out = ScaledValue::from_value(value)?;
    }
    for (out, &value) in cached_divisor.iter_mut().zip(b) {
        *out = ScaledValue::from_value(value)?;
    }
    for (k, &value) in a.iter().enumerate() {
        check()?;
        let product = crate::fast::convolution_coefficient(
            k,
            q.len(),
            b.len(),
            |i| Ok(cached_quotient[i]),
            |i| Ok(cached_divisor[i]),
        )?;
        let difference =
            ScaledValue::from_value(value)?
                .sub(product)?
                .sub(ScaledValue::from_value(
                    r.get(k).copied().unwrap_or(C::ZERO),
                )?)?;
        residual = residual.add(difference.magnitude()?)?;
    }
    let denominator = norm(a)?.add(norm(q)?.mul(norm(b)?)?)?.add(norm(r)?)?;
    let backward = if denominator.is_zero() {
        C::Real::ZERO
    } else {
        residual.div(denominator)?.round()?
    };
    let (bounds, rest) = scratch.split_at_mut(a.len());
    let divisor_bounds = &mut rest[..b.len()];
    for (bound, &value) in divisor_bounds.iter_mut().zip(b) {
        *bound = real::<C>(ScaledValue::from_value(value)?.magnitude()?);
    }
    let mut quotient_bound = Scaled::ZERO;
    for (bound, &value) in bounds.iter_mut().zip(a) {
        *bound = real::<C>(ScaledValue::from_value(value)?.magnitude()?);
    }
    if a.len() >= b.len() {
        let degree = b.len() - 1;
        let leading = divisor_bounds[degree].real();
        for shift in (0..=a.len() - b.len()).rev() {
            check()?;
            let magnitude = ScaledValue::from_value(q[shift])?.magnitude()?;
            let bound = bounds[degree + shift].real().div(leading)?.add(magnitude)?;
            quotient_bound = quotient_bound.add(bound)?;
            let update_bound = bound.add(magnitude)?;
            for (index, coefficient) in divisor_bounds[..degree].iter().enumerate() {
                bounds[index + shift] = real::<C>(
                    bounds[index + shift]
                        .real()
                        .add(update_bound.mul(coefficient.real())?)?,
                );
            }
            bounds[degree + shift] = ScaledValue::ZERO;
        }
    }
    let majorant = bounds
        .iter()
        .try_fold(quotient_bound, |sum, value| sum.add(value.real()))?;
    let condition = crate::basis::scaled_condition_ratio(majorant, norm(q)?.add(norm(r)?)?)?;
    crate::basis::condition_report_from_scale(coefficient_scale(q), backward, condition, a.len())
}
