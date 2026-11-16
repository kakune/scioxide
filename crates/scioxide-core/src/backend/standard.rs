//! Products, dot products and triangular-substitution steps that avoid overflow and underflow.
//!
//! These helpers support the `Standard` level of [`crate::Accuracy`]: they use
//! ordinary arithmetic and rescale by powers of two, or switch to exact
//! accumulation, only when a result would leave the normal range.

use crate::{AlgebraicScalar, FloatDecomposition, NumericScalar, RealScalar, SciOxideError};

/// Returns the binary exponent `e` with `maximum < 2^e`, or 0 for a zero bound.
///
/// # Errors
///
/// - `NumericalInstability` (`Overflow`) if `maximum` is NaN or infinite.
pub fn exponent<R: RealScalar>(maximum: R) -> Result<i32, SciOxideError> {
    match maximum.decompose() {
        FloatDecomposition::Finite { significand: 0, .. } => Ok(0),
        FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } => Ok(i32::from(exponent) + significand.ilog2() as i32 + 1),
        _ => Err(SciOxideError::numerical_instability(
            crate::NumericalIssue::Overflow,
            "component bound is not finite",
        )),
    }
}

/// Returns the largest absolute value of the real and imaginary components.
#[inline]
pub fn magnitude<S: NumericScalar>(value: S) -> S::Real {
    if S::IS_REAL {
        return value.real_component().abs();
    }
    value
        .real_component()
        .abs()
        .max(value.imaginary_component().abs())
}

/// Multiplies both components by `2^exponent`.
#[inline]
pub fn scale<S: NumericScalar>(value: S, exponent: i32) -> S {
    if exponent == 0 {
        value
    } else {
        S::from_components(
            value.real_component().scalbn(exponent),
            value.imaginary_component().scalbn(exponent),
        )
    }
}

/// Computes `a · x + c` with fused multiply-adds on each component.
///
/// Use it for vector updates in Krylov recurrences such as CGS, whose attainable
/// residual depends on the single rounding of the product. Reductions and
/// products use separate operations.
#[inline]
pub fn fused_update<S: NumericScalar>(a: S, x: S, c: S) -> S {
    let real = a
        .real_component()
        .mul_add(x.real_component(), c.real_component());
    if S::IS_REAL {
        S::from_real(real)
    } else {
        S::from_components(
            (-a.imaginary_component()).mul_add(x.imaginary_component(), real),
            a.real_component().mul_add(
                x.imaginary_component(),
                a.imaginary_component()
                    .mul_add(x.real_component(), c.imaginary_component()),
            ),
        )
    }
}

/// Sums the products `a · b` in four interleaved lanes with ordinary arithmetic.
///
/// The arguments must be finite and scaled so the sum cannot overflow, or the
/// result must be checked before use.
pub fn dot<S: NumericScalar>(products: impl Iterator<Item = (S, S)>) -> S {
    let mut sums = [S::ZERO; 4];
    for (index, (a, b)) in products.enumerate() {
        let lane = index % sums.len();
        sums[lane] = a * b + sums[lane];
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

/// Multiplies `value` by `2^exponent` and checks that the result is finite.
///
/// # Errors
///
/// - `NumericalInstability` (`Overflow`) if a component of the result is not finite.
pub fn restore<S: NumericScalar>(value: S, exponent: i32) -> Result<S, SciOxideError> {
    finite_candidate(scale(value, exponent))
}

/// Computes `(prior − Σ a·x) / pivot` for one row of a triangular substitution.
///
/// `residual` is the ordinary-arithmetic value of `prior − Σ a·x`. If dividing it
/// by `pivot` overflows, or a residual below the normal range meets a pivot below
/// one, the row is recomputed in a scaled frame. `bounds` gives precomputed
/// maxima of `|a|` and `|x|` so the retry need not rescan the row; it is ignored
/// for tiny residuals. A final result below the normal range is rounded as usual.
///
/// # Errors
///
/// - `Singular` if `pivot` is zero.
/// - `NumericalInstability` (`Overflow`) if the quotient does not fit in `S`.
#[inline]
pub fn finish_substitution<S: NumericScalar>(
    prior: S,
    pivot: S,
    residual: S,
    products: impl ExactSizeIterator<Item = (S, S)> + Clone,
    bounds: Option<(S::Real, S::Real)>,
) -> Result<S, SciOxideError> {
    if pivot == S::ZERO {
        return Err(SciOxideError::singular(None));
    }
    if let Some(candidate) = ordinary_quotient(residual, pivot) {
        return Ok(candidate);
    }
    let tiny = magnitude(residual) < S::Real::MIN_POSITIVE && magnitude(pivot) < S::Real::ONE;
    scaled_substitution(prior, pivot, products, bounds.filter(|_| !tiny))
}

/// Returns `residual / pivot` in ordinary arithmetic, or `None` if [`finish_substitution`] would retry.
///
/// It returns `None` when the quotient is not finite, or when a residual below
/// the normal range meets a pivot below one. `pivot` must be nonzero.
#[inline]
pub fn ordinary_quotient<S: NumericScalar>(residual: S, pivot: S) -> Option<S> {
    let candidate = if pivot == S::ONE {
        residual
    } else {
        residual / pivot
    };
    let tiny = magnitude(residual) < S::Real::MIN_POSITIVE && magnitude(pivot) < S::Real::ONE;
    (candidate.has_finite_components() && !tiny).then_some(candidate)
}

#[cold]
fn scaled_substitution<S: NumericScalar>(
    prior: S,
    pivot: S,
    products: impl ExactSizeIterator<Item = (S, S)> + Clone,
    bounds: Option<(S::Real, S::Real)>,
) -> Result<S, SciOxideError> {
    let (maximum, solved_maximum) = bounds.unwrap_or_else(|| {
        products
            .clone()
            .fold((S::Real::ZERO, S::Real::ZERO), |(am, xm), (a, x)| {
                if a == S::ZERO {
                    (am, xm)
                } else {
                    (am.max(magnitude(a)), xm.max(magnitude(x)))
                }
            })
    });
    let te = exponent(maximum)?;
    let xe = exponent(solved_maximum)?;
    let be = exponent(magnitude(prior))?;
    let de = exponent(magnitude(pivot))?;
    let growth = usize::BITS as i32 - products.len().max(1).leading_zeros() as i32 + 3;
    let frame = if products.len() == 0 || solved_maximum == S::Real::ZERO {
        be + 2
    } else {
        let prior_exponent = if prior == S::ZERO { i32::MIN } else { be + 2 };
        prior_exponent.max(te + xe + growth)
    };
    let mut sum = scale(prior, -frame);
    for (a, x) in products {
        if a != S::ZERO {
            sum = sum - scale(a, -te) * scale(scale(x, -xe), te + xe - frame);
        }
    }
    restore(sum / scale(pivot, -de), frame - de)
}

/// Computes the LU multiplier `a / b` and checks that it is representable.
///
/// A multiplier that underflows to zero, or overflows, can change a later pivot,
/// so such quotients are recomputed with extended exponent range and must fit in
/// `S`.
///
/// # Errors
///
/// - `Singular` if `b` is zero.
/// - `NumericalInstability` if the exact quotient overflows or underflows to zero.
#[inline(always)]
pub fn factor_quotient<S: NumericScalar>(a: S, b: S) -> Result<S, SciOxideError> {
    if b == S::ZERO {
        return Err(SciOxideError::singular(None));
    }
    let value = a / b;
    if value.has_finite_components() && (value != S::ZERO || a == S::ZERO) {
        Ok(value)
    } else {
        crate::ScaledValue::from_value(a)?
            .div(crate::ScaledValue::from_value(b)?)?
            .try_value()
    }
}

/// Computes `(prior − Σ a·x) / pivot` for one row of a triangular substitution.
///
/// The sum uses four interleaved lanes. Overflow and underflow are handled as in
/// [`finish_substitution`].
///
/// # Errors
///
/// - `Singular` if `pivot` is zero.
/// - `NumericalInstability` (`Overflow`) if the quotient does not fit in `S`.
#[inline(always)]
pub fn substitute_row<S: NumericScalar>(
    prior: S,
    pivot: S,
    products: impl ExactSizeIterator<Item = (S, S)> + Clone,
) -> Result<S, SciOxideError> {
    let mut sums = [prior, S::ZERO, S::ZERO, S::ZERO];
    for (index, (a, x)) in products.clone().enumerate() {
        let lane = index % sums.len();
        sums[lane] = sums[lane] - a * x;
    }
    finish_substitution(
        prior,
        pivot,
        (sums[0] + sums[1]) + (sums[2] + sums[3]),
        products,
        None,
    )
}

/// Evaluates an inner product unscaled first, then with scaling if it overflows.
///
/// `a` and `b` are the scale exponents of the two operands and `sum(ea, eb)`
/// evaluates the product sum with the operands multiplied by `2^−ea` and `2^−eb`.
/// If `a + b > 0`, the unscaled sum is tried first and returned when finite.
/// Otherwise, or if it overflows, the scaled sum is computed and rescaled by
/// `2^(a + b)`. Scaling each operand by its own maximum avoids overflow, but can
/// flush a small factor whose product with a large one is ordinary, such as
/// `MAX · MIN_POSITIVE`; the unscaled attempt avoids that loss.
///
/// # Errors
///
/// - `NumericalInstability` (`Overflow`) if the rescaled result is not finite.
pub fn unit_then_normalized<S: NumericScalar>(
    a: i32,
    b: i32,
    sum: impl Fn(i32, i32) -> S,
) -> Result<S, SciOxideError> {
    if a + b > 0 {
        let unscaled = sum(0, 0);
        if unscaled.has_finite_components() {
            return Ok(unscaled);
        }
    }
    restore(sum(a, b), a + b)
}

/// Returns the scale exponents for a sum of `terms` products of operands bounded by `a` and `b`.
///
/// Returns `(0, 0)` when the products and their sum stay in the normal range
/// with room for accumulation. Otherwise returns exponents that normalize each
/// operand separately so that no product overflows. Callers that risk only
/// overflow should try [`unit_then_normalized`] first.
///
/// # Errors
///
/// - `NumericalInstability` (`Overflow`) if `a` or `b` is not finite.
pub fn product_scales<R: RealScalar>(
    a: R,
    b: R,
    terms: usize,
) -> Result<(i32, i32), SciOxideError> {
    let ae = exponent(a)?;
    let be = exponent(b)?;
    let growth = usize::BITS as i32 - terms.max(1).leading_zeros() as i32 + 3;
    let high = exponent(R::MAX)?;
    let low = exponent(R::MIN_POSITIVE)?;
    if ae + be + growth < high && ae + be > low + R::MANTISSA_DIGITS as i32 {
        Ok((0, 0))
    } else {
        // A nonnegative restore exponent includes the destination subnormal grid.
        Ok((ae, be.max(-ae)))
    }
}

/// Returns `value` if both of its components are finite.
///
/// # Errors
///
/// - `NumericalInstability` (`Overflow`) if a component is NaN or infinite.
pub fn finite_candidate<S: NumericScalar>(value: S) -> Result<S, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            crate::NumericalIssue::Overflow,
            "finite operands produced a non-finite result",
        ))
    }
}

/// Returns whether the ordinary product `a · b = product` overflowed or underflowed.
///
/// A product with a zero factor never counts as underflow. For complex values,
/// each real cross product is checked separately, so cancellation in the
/// result cannot hide an overflowing partial product.
#[inline]
pub fn product_needs_retry<S: NumericScalar>(a: S, b: S, product: S) -> bool {
    fn component<R: RealScalar>(a: R, b: R, product: R) -> bool {
        !product.is_finite() || (a != R::ZERO && b != R::ZERO && product.abs() < R::MIN_POSITIVE)
    }
    if S::IS_REAL {
        component(
            a.real_component(),
            b.real_component(),
            product.real_component(),
        )
    } else {
        let (ar, ai) = (a.real_component(), a.imaginary_component());
        let (br, bi) = (b.real_component(), b.imaginary_component());
        !product.has_finite_components()
            || component(ar, br, ar * br)
            || component(ai, bi, ai * bi)
            || component(ar, bi, ar * bi)
            || component(ai, br, ai * br)
    }
}

/// Computes `Σ a·b` with the given [`crate::Accuracy`].
///
/// `Standard` sums in four ordinary lanes and falls back to exact accumulation
/// when a product overflows or underflows. `Exact` accumulates exactly and rounds
/// each component once. An empty sum is `+0`.
///
/// # Errors
///
/// - `NonFinite` if an operand is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if the result does not fit in `S`.
///
/// ```
/// use scioxide_core::{Accuracy, backend::standard::sum_products};
/// assert_eq!(sum_products([(2.0_f64, 3.0)].into_iter(), Accuracy::Standard)?, 6.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn sum_products<S: NumericScalar>(
    products: impl Iterator<Item = (S, S)> + Clone,
    accuracy: crate::Accuracy,
) -> Result<S, SciOxideError> {
    if accuracy == crate::Accuracy::Standard {
        let mut sums = [S::ZERO; 4];
        let mut retry = false;
        for (i, (a, b)) in products.clone().enumerate() {
            let product = a * b;
            retry |= product_needs_retry(a, b, product);
            sums[i % 4] = sums[i % 4] + product;
        }
        let value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        if !retry && value.has_finite_components() {
            return Ok(value);
        }
    }
    exact_sum_products(products)
}
// Keep the large exact state and its initialization off Standard hot stacks.
#[cold]
#[inline(never)]
fn exact_sum_products<S: NumericScalar>(
    products: impl Iterator<Item = (S, S)>,
) -> Result<S, SciOxideError> {
    let mut sum = crate::ProductSum::new();
    for (a, b) in products {
        sum.add_product(a, b)?;
    }
    sum.finish()
}

/// Multiplies `a · b` with the given [`crate::Accuracy`].
///
/// `Standard` returns the ordinary product unless it overflows or underflows, in
/// which case the product is computed exactly and rounded once.
///
/// # Errors
///
/// - `NonFinite` if an operand is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if the result does not fit in `S`.
#[inline]
pub fn product<S: NumericScalar>(
    a: S,
    b: S,
    accuracy: crate::Accuracy,
) -> Result<S, SciOxideError> {
    let p = a * b;
    if accuracy == crate::Accuracy::Standard && !product_needs_retry(a, b, p) {
        Ok(p)
    } else {
        exact_sum_products(core::iter::once((a, b)))
    }
}
