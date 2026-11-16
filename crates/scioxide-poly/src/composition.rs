//! Composition `p(q(x))` and Taylor shift `p(x + s)` of monomial coefficients.
use crate::PolynomialBasis;
use crate::fast::ProductScratch;
use crate::util::{round_complex, scaled_complex, validate_slice};
use crate::{
    ApproxPolynomial, MonomialBasis, PolynomialOperationPlan, PolynomialOperationRequest,
    PolynomialWorkspace,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Argument, ExecutionContext, NumericScalar, RealScalar, ScaledValue, SciOxideError,
};

/// Returns the coefficient count `(a − 1)(b − 1) + 1` of `p(q(x))` for operands
/// of lengths `a` and `b`. An empty operand is `Empty`, reported against
/// `outer` (length `a`) or `inner` (length `b`).
pub(crate) fn length(
    a: usize,
    b: usize,
    outer: Argument,
    inner: Argument,
) -> Result<usize, SciOxideError> {
    if a == 0 || b == 0 {
        return Err(crate::util::empty_error(if a == 0 { outer } else { inner }));
    }
    (a - 1)
        .checked_mul(b - 1)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("composition_coefficients"))
        })
}

/// Returns the largest power of two below the length of a nonconstant outer
/// polynomial.
pub(crate) fn split(len: usize) -> usize {
    1 << (len - 1).ilog2()
}

/// Describes one interior node of the composition recursion,
/// `p = low + x^block·high`.
///
/// Execution takes `low(q)` and then `high(q)` from the front of its stack,
/// recurses into `low` (always a power-of-two block) and `high` (the
/// remainder), and multiplies `high(q)` by `q^block`. Workspace sizing walks the
/// same nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    /// Coefficient count of `low`, the largest power of two below `p`.
    pub(crate) block: usize,
    /// Output lengths of `low(q)` and `high(q)`.
    pub(crate) low: usize,
    pub(crate) high: usize,
    /// Coefficient count of `q^block`.
    pub(crate) power: usize,
}

impl Split {
    /// Returns the node for operand lengths `p` and `q`, or `None` for a leaf: a
    /// constant outer or inner polynomial needs neither stack nor product.
    pub(crate) fn of(
        p: usize,
        q: usize,
        outer: Argument,
        inner: Argument,
    ) -> Result<Option<Self>, SciOxideError> {
        if p == 1 || q == 1 {
            return Ok(None);
        }
        let block = split(p);
        Ok(Some(Self {
            block,
            low: length(block, q, outer, inner)?,
            high: length(p - block, q, outer, inner)?,
            power: length(block + 1, q, outer, inner)?,
        }))
    }
}

/// Returns the number of slots for `q², q⁴, …, q^split(p)`, in ascending
/// power order. The first power is `q` itself and needs no slot.
pub(crate) fn powers_len(p: usize, q: usize) -> Result<usize, SciOxideError> {
    if p <= 2 || q == 1 {
        return Ok(0);
    }
    let m = split(p);
    // Sum of powers 2 + 4 + … + m is 2*(m-1), without overflowing 2*m.
    (m - 1)
        .checked_mul(2)
        .and_then(|degrees| degrees.checked_mul(q - 1))
        .and_then(|coefficients| coefficients.checked_add(m.ilog2() as usize))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("composition_powers")))
}
impl<C: NumericScalar> PolynomialOperationPlan<C> {
    /// Computes the monomial coefficients of `p(q(x))` and writes them to
    /// `output`.
    ///
    /// `p` and `q` hold ascending monomial coefficients, `p(x) = Σₖ pₖ xᵏ`.
    /// `output` must have exactly `(p.len() − 1)(q.len() − 1) + 1` entries; it
    /// keeps trailing zeros. The composition splits `p = low + xᵐ·high`, with
    /// `m` the largest power of two below `p.len()`, and evaluates
    /// `low(q) + qᵐ·high(q)` recursively. The powers `q², q⁴, …` are computed
    /// once. The cost is `O(M(N) log n)`, where `n = p.len()`, `N` is the output
    /// length and `M(N)` is the cost of one product with `N` coefficients (see
    /// [`Self::multiply_into`] for the product algorithm). Intermediate zero,
    /// subnormal or out-of-range coefficients retry with coefficient Horner in
    /// extended exponent range, at O(n N q.len()) cost and O(N) extra storage.
    /// Only the final coefficients are rounded in that fallback.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `p` or `q` is empty, or if the shape
    ///   `Compose { outer_len: p.len(), inner_len: q.len() }` was not prepared.
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace` is
    ///   too small.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` if a result coefficient overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // p(x) = (1 + x)², q(x) = 2x, so p(q(x)) = 1 + 4x + 4x².
    /// let request = PolynomialOperationRequest::Compose {
    ///     outer_len: 3,
    ///     inner_len: 2,
    /// };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let mut ws = plan.workspace(plan.requests())?;
    /// let mut out = [0.0_f64; 3];
    /// let cx = ExecutionContext::sequential();
    /// plan.compose_into(&[1.0_f64, 2.0, 1.0], &[0.0_f64, 2.0], &mut out, &mut ws, cx)?;
    /// let want = [1.0_f64, 4.0, 4.0];
    /// assert!(out.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compose_into(
        &self,
        p: &[C],
        q: &[C],
        output: &mut [C],
        workspace: &mut PolynomialWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        length(
            p.len(),
            q.len(),
            scioxide_core::argument!("p"),
            scioxide_core::argument!("q"),
        )?;
        self.admit(
            workspace,
            PolynomialOperationRequest::Compose {
                outer_len: p.len(),
                inner_len: q.len(),
            },
        )?;
        self.compose_prepared(p, q, output, workspace, context)
    }

    fn compose_prepared(
        &self,
        p: &[C],
        q: &[C],
        output: &mut [C],
        workspace: &mut PolynomialWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let len = length(
            p.len(),
            q.len(),
            scioxide_core::argument!("p"),
            scioxide_core::argument!("q"),
        )?;
        if output.len() != len {
            return Err(SciOxideError::dimension_mismatch(len, output.len()));
        }
        validate_slice(p)?;
        validate_slice(q)?;
        let (staged, stack) = workspace.coefficients.split_at_mut(output.len());
        let (powers, stack) = stack.split_at_mut(powers_len(p.len(), q.len())?);
        let mut scratch = ProductScratch {
            complex: &mut workspace.fft,
            real: &mut workspace.real,
        };
        let attempt = (|| {
            // Retain each binary power once, separate from the sibling-reused stack.
            let mut remaining = &mut *powers;
            let mut previous = q;
            while !remaining.is_empty() {
                let (power, rest) = remaining.split_at_mut(previous.len() + (previous.len() - 1));
                self.product(previous, previous, power, &mut scratch, context)?;
                if !normal_product(previous, previous, power) {
                    return Ok(false);
                }
                previous = power;
                remaining = rest;
            }
            self.compose_recursive(p, q, powers, staged, stack, &mut scratch, context)
        })();
        let retry = match attempt {
            Ok(complete) => !complete,
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow
                            | scioxide_core::NumericalIssue::Underflow,
                        ..
                    }
                ) =>
            {
                true
            }
            Err(error) => return Err(error),
        };
        if retry {
            compose_scaled(p, q, staged, &mut workspace.scaled, context)?;
        }
        context.check()?;
        output.copy_from_slice(staged);
        Ok(())
    }

    /// Computes the monomial coefficients of the Taylor shift `p(x + shift)` and
    /// writes them to `output`.
    ///
    /// The shift is computed as the composition `p(q(x))` with `q(x) = shift + x`,
    /// so no factorials appear. `output` must have `p.len()` entries.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `p` is empty or the shape `TaylorShift { len: p.len() }`
    ///   was not prepared.
    /// - `DimensionMismatch` if `output` has the wrong length or `workspace` is
    ///   too small.
    /// - `NonFinite` if a coefficient or `shift` is NaN or infinite.
    /// - `NumericalInstability` if a result coefficient overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::{PolynomialOperationPlan, PolynomialOperationRequest};
    ///
    /// // (x + 1 + 1)² = 4 + 4x + x².
    /// let request = PolynomialOperationRequest::TaylorShift { len: 3 };
    /// let plan = PolynomialOperationPlan::<f64>::new(&[request])?;
    /// let mut ws = plan.workspace(plan.requests())?;
    /// let mut out = [0.0_f64; 3];
    /// let cx = ExecutionContext::sequential();
    /// plan.taylor_shift_into(&[1.0_f64, 2.0, 1.0], 1.0, &mut out, &mut ws, cx)?;
    /// let want = [4.0_f64, 4.0, 1.0];
    /// assert!(out.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn taylor_shift_into(
        &self,
        p: &[C],
        shift: C,
        output: &mut [C],
        workspace: &mut PolynomialWorkspace<C>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        // `p(x + s)` is composition with `q = s + x`. This avoids the textbook
        // `sum_k p^{(k)}(s) / k! x^k` formula, whose factorials overflow early and
        // whose derivative sums cancel badly.
        context.check()?;
        crate::fast::nonempty(p, scioxide_core::argument!("p"))?;
        self.admit(
            workspace,
            PolynomialOperationRequest::TaylorShift { len: p.len() },
        )?;
        self.compose_prepared(p, &[shift, C::ONE], output, workspace, context)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "operands, shared powers, output, stack, scratch and context stay explicit"
    )]
    fn compose_recursive(
        &self,
        p: &[C],
        q: &[C],
        powers: &[C],
        out: &mut [C],
        stack: &mut [C],
        fft: &mut ProductScratch<'_, C::Real>,
        context: ExecutionContext<'_>,
    ) -> Result<bool, SciOxideError> {
        context.check()?;
        // A constant inner polynomial collapses the composition to evaluating `p` at
        // that constant, projecting only the requested value.
        // Split `p = low + x^m high`, so `p(q) = low(q) + q^m high(q)`. With fast
        // products this costs O(M(n) log n). A binary block makes q^m common to
        // all nodes using that block, even when the outer length is irregular.
        let Some(node) = Split::of(
            p.len(),
            q.len(),
            scioxide_core::argument!("p"),
            scioxide_core::argument!("q"),
        )?
        else {
            out[0] = if q.len() == 1 {
                crate::horner::monomial_horner::<_, 0, false>(p.iter().copied(), q[0], || {
                    context.check()
                })?
                .value()?
            } else {
                p[0]
            };
            return Ok(true);
        };
        let power = if node.block == 1 {
            q
        } else {
            let end = powers_len(p.len(), q.len())?;
            &powers[end - node.power..end]
        };
        let (low, stack) = stack.split_at_mut(node.low);
        let (high, stack) = stack.split_at_mut(node.high);
        if !self.compose_recursive(&p[..node.block], q, powers, low, stack, fft, context)?
            || !self.compose_recursive(&p[node.block..], q, powers, high, stack, fft, context)?
        {
            return Ok(false);
        }
        self.product(high, power, out, fft, context)?;
        if !normal_product(high, power, out) {
            return Ok(false);
        }
        // `low(q)` has lower degree than the product, so it only touches the leading
        // part of `out`. Addition rounds gradually; only non-finite components
        // need the extended retry, independently for each coefficient.
        for (v, a) in out.iter_mut().zip(low.iter()) {
            let sum = *v + *a;
            *v = if sum.has_finite_components() {
                sum
            } else {
                round_complex(scaled_complex(*v)?.add(scaled_complex(*a)?)?)?
            };
        }
        Ok(true)
    }
}
impl<C: NumericScalar> ApproxPolynomial<C, MonomialBasis<C>> {
    /// Returns the composition `self(inner(x))` in the monomial basis.
    ///
    /// Both polynomials must have the same approximation policy, which the
    /// result keeps. This builds a one-shot [`PolynomialOperationPlan`] and calls
    /// [`PolynomialOperationPlan::compose_into`]; use the plan directly to
    /// compose many polynomials of the same shapes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the two policies differ.
    /// - `SizeOverflow` or `AllocationFailure` if the result does not fit in
    ///   memory.
    /// - `NumericalInstability` if a result coefficient overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // p(x) = (1 + x)², so p(p(1)) = p(4) = 25.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.compose(&p)?;
    /// assert!((result.evaluate(1.0)? - 25.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn compose(&self, inner: &Self) -> Result<Self, SciOxideError> {
        crate::util::same_policy(
            self.policy(),
            inner.policy(),
            scioxide_core::argument!("inner.policy"),
        )?;
        let n = length(
            self.basis().coefficients().len(),
            inner.basis().coefficients().len(),
            scioxide_core::argument!("self"),
            scioxide_core::argument!("inner"),
        )?;
        let request = PolynomialOperationRequest::Compose {
            outer_len: self.basis().coefficients().len(),
            inner_len: inner.basis().coefficients().len(),
        };
        let plan = PolynomialOperationPlan::new(&[request])?;
        let mut work = plan.workspace(&[request])?;
        let mut output = scioxide_core::backend::try_filled(
            n,
            C::ZERO,
            scioxide_core::quantity!("composition_coefficients"),
        )?;
        plan.compose_into(
            self.basis().coefficients(),
            inner.basis().coefficients(),
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Self::monomial_with_options(output, self.policy())
    }
    /// Returns the Taylor shift `self(x + shift)` in the monomial basis.
    ///
    /// The method is that of [`PolynomialOperationPlan::taylor_shift_into`],
    /// with a one-shot plan.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shift` is NaN or infinite.
    /// - `NumericalInstability` if a result coefficient overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// // p(x) = (1 + x)², so p(1 + 1) = 9.
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let result = p.taylor_shift(1.0)?;
    /// assert!((result.evaluate(1.0)? - 9.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn taylor_shift(&self, shift: C) -> Result<Self, SciOxideError> {
        let request = PolynomialOperationRequest::TaylorShift {
            len: self.basis().coefficients().len(),
        };
        let plan = PolynomialOperationPlan::new(&[request])?;
        let mut work = plan.workspace(&[request])?;
        let mut output = scioxide_core::backend::try_filled(
            self.basis().coefficients().len(),
            C::ZERO,
            scioxide_core::quantity!("shifted_coefficients"),
        )?;
        plan.taylor_shift_into(
            self.basis().coefficients(),
            shift,
            &mut output,
            &mut work,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Self::monomial_with_options(output, self.policy())
    }
}

// A zero may be structural or a rounded nonzero term. Replay ambiguous lanes
// rather than allowing later multiplication to amplify an erased coefficient.
fn normal_product<C: NumericScalar>(a: &[C], b: &[C], values: &[C]) -> bool {
    let support = |coefficients: &[C]| {
        Some((
            coefficients.iter().position(|&c| c != C::ZERO)?,
            coefficients.iter().rposition(|&c| c != C::ZERO)?,
        ))
    };
    let (Some((a_first, a_last)), Some((b_first, b_last))) = (support(a), support(b)) else {
        return true;
    };
    // Coefficients outside the product's support are structurally zero.
    let values = &values[a_first + b_first..=a_last + b_last];
    let real = C::IS_REAL
        || a.iter()
            .chain(b)
            .all(|c| c.imaginary_component() == C::Real::ZERO);
    values
        .iter()
        .all(|c| c.real_component().is_normal() && (real || c.imaginary_component().is_normal()))
}

/// Coefficient Horner with extended exponents and two output-sized buffers.
fn compose_scaled<C: NumericScalar>(
    p: &[C],
    q: &[C],
    output: &mut [C],
    scratch: &mut [ScaledValue<C>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = output.len();
    let (mut current, mut next) = scratch[..2 * n].split_at_mut(n);
    current.fill(ScaledValue::ZERO);
    current[0] = ScaledValue::from_value(p[p.len() - 1])?;
    let mut length = 1;
    for &coefficient in p[..p.len() - 1].iter().rev() {
        context.check()?;
        let next_length = length + q.len() - 1;
        next[..next_length].fill(ScaledValue::ZERO);
        for (i, &value) in current[..length].iter().enumerate() {
            context.check()?;
            if value.is_zero() {
                continue;
            }
            for (j, &factor) in q.iter().enumerate() {
                if factor != C::ZERO {
                    next[i + j] = next[i + j].add(value.mul(ScaledValue::from_value(factor)?)?)?;
                }
            }
        }
        next[0] = next[0].add(ScaledValue::from_value(coefficient)?)?;
        core::mem::swap(&mut current, &mut next);
        length = next_length;
    }
    for (out, &value) in output.iter_mut().zip(current.iter()) {
        context.check()?;
        *out = value.round()?;
    }
    Ok(())
}
