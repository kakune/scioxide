//! Real scalar kernels that report failures as a small `Copy` enum.
//!
//! Most functions here compute the crate-root function of the same name in `f64`
//! and round the result to `S`. They return a [`ScalarKernelError`] instead of a
//! [`crate::SciOxideError`], which suits kernels that evaluate many values;
//! `From<ScalarKernelError>` converts it.
//!
//! ```
//! use scioxide_core::backend::{ScalarKernelError, real_logit};
//! assert_eq!(real_logit(0.5_f64), Ok(0.0));
//! assert_eq!(real_logit(0.0_f64), Err(ScalarKernelError::DomainX));
//! ```
use crate::{NonFiniteOrigin, RangeAccumulator, RealScalar, SciOxideError};

/// Failure of a scalar kernel in this module.
///
/// Convert it to a [`crate::SciOxideError`] with `From`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKernelError {
    /// An argument is NaN or infinite.
    NonFiniteInput,
    /// An intermediate factor is NaN or infinite.
    NonFiniteIntermediate,
    /// The first argument is outside the function's domain.
    DomainX,
    /// The second argument is outside the function's domain.
    DomainY,
    /// The denominator is zero.
    ZeroDenominator,
    /// The result overflows the scalar type.
    Overflow,
}
impl From<ScalarKernelError> for SciOxideError {
    fn from(error: ScalarKernelError) -> Self {
        use ScalarKernelError::*;
        match error {
            NonFiniteInput => {
                Self::non_finite(NonFiniteOrigin::Input, "scalar kernel input must be finite")
            }
            NonFiniteIntermediate => Self::non_finite(
                NonFiniteOrigin::Intermediate,
                "an active compound factor is non-finite",
            ),
            DomainX | DomainY => Self::domain_error(
                if error == DomainX {
                    crate::argument!("x")
                } else {
                    crate::argument!("y")
                },
                crate::DomainReason::OutsideDomain,
                "argument is outside the transform domain",
            ),
            ZeroDenominator => Self::domain_error(
                crate::argument!("denominator"),
                crate::DomainReason::Undefined,
                "denominator must be nonzero",
            ),
            Overflow => Self::numerical_instability(
                crate::NumericalIssue::Overflow,
                "finite mathematical result exceeds scalar range",
            ),
        }
    }
}

fn finite<S: RealScalar>(values: &[S]) -> Result<(), ScalarKernelError> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(ScalarKernelError::NonFiniteInput)
    }
}
fn projection<S: RealScalar>(value: f64) -> Result<S, ScalarKernelError> {
    if !value.is_finite() {
        return Err(ScalarKernelError::Overflow);
    }
    let result = S::from_f64_ieee(value);
    if !result.is_finite() {
        Err(ScalarKernelError::Overflow)
    } else {
        Ok(result)
    }
}

/// Computes [`crate::softplus`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_softplus<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::softplus(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::sigmoid`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_sigmoid<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::sigmoid(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::log_sigmoid`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_log_sigmoid<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::log_sigmoid(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::logit`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainX`] if `x ≤ 0` or `x ≥ 1`.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_logit<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();
    if x <= 0.0 || x >= 1.0 {
        return Err(ScalarKernelError::DomainX);
    }
    projection(crate::logit(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::exprel`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_exprel<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::exprel(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::cosm1`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_cosm1<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::cosm1(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::sinc`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_sinc<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::sinc(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::sincpi`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_sincpi<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::sincpi(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::sin_pi`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_sin_pi<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::sin_pi(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::cos_pi`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_cos_pi<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::cos_pi(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::tan_pi`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainX`] if `cos(πx) = 0`, that is, `x` is a half-integer.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_tan_pi<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();
    if crate::stable::phase(x, 1.0).1 == 0.0 {
        return Err(ScalarKernelError::DomainX);
    }
    projection(crate::tan_pi(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::sin_degrees`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_sin_degrees<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::sin_degrees(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::cos_degrees`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_cos_degrees<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();

    projection(crate::cos_degrees(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::tan_degrees`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainX`] if `x` is an odd multiple of 90.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_tan_degrees<S: RealScalar>(x: S) -> Result<S, ScalarKernelError> {
    finite(&[x])?;
    let x = x.into_f64();
    if crate::stable::phase(x, 180.0).1 == 0.0 {
        return Err(ScalarKernelError::DomainX);
    }
    projection(crate::tan_degrees(x).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::xlogy`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainY`] if `x ≠ 0` and `y ≤ 0`.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_xlogy<S: RealScalar>(x: S, y: S) -> Result<S, ScalarKernelError> {
    finite(&[x, y])?;
    let (x, y) = (x.into_f64(), y.into_f64());
    if x != 0.0 && y <= 0.0 {
        return Err(ScalarKernelError::DomainY);
    }
    projection(crate::xlogy(x, y).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::xlog1py`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainY`] if `x ≠ 0` and `y ≤ −1`.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_xlog1py<S: RealScalar>(x: S, y: S) -> Result<S, ScalarKernelError> {
    finite(&[x, y])?;
    let (x, y) = (x.into_f64(), y.into_f64());
    if x != 0.0 && y <= -1.0 {
        return Err(ScalarKernelError::DomainY);
    }
    projection(crate::xlog1py(x, y).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes [`crate::powm1`] in `f64` and rounds the result to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::DomainY`] if `x = 0` and `y < 0`, or if `x < 0` and `y`
///   is not an integer.
/// - [`ScalarKernelError::Overflow`] if evaluation fails or the result exceeds `S`.
#[inline]
pub fn real_powm1<S: RealScalar>(x: S, y: S) -> Result<S, ScalarKernelError> {
    finite(&[x, y])?;
    let (x, y) = (x.into_f64(), y.into_f64());
    if (x == 0.0 && y < 0.0) || (x < 0.0 && y.trunc() != y) {
        return Err(ScalarKernelError::DomainY);
    }
    projection(crate::powm1(x, y).map_err(|_| ScalarKernelError::Overflow)?)
}

/// Computes `log(eˣ + eʸ)` like [`crate::logaddexp`].
///
/// An argument of `−∞` stands for a zero term.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or `+∞`.
/// - [`ScalarKernelError::Overflow`] if the result overflows.
pub fn real_logaddexp<S: RealScalar>(x: S, y: S) -> Result<S, ScalarKernelError> {
    if [x, y]
        .iter()
        .any(|x| !x.is_finite() && *x != S::NEG_INFINITY)
    {
        return Err(ScalarKernelError::NonFiniteInput);
    }
    crate::logaddexp(x, y).map_err(|_| ScalarKernelError::Overflow)
}

/// Multiplies two finite scalars with one rounding.
///
/// An exact zero product is `+0`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if the rounded product overflows.
#[inline]
pub fn scalar_product<S: RealScalar>(left: S, right: S) -> Result<S, ScalarKernelError> {
    finite::<S>(&[left, right])?;

    crate::checked_float_product(left, right).map_err(|_| ScalarKernelError::Overflow)
}

/// Computes `left · right + addend` with one rounding.
///
/// An exact zero result is `+0`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if an argument is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if the rounded result overflows.
#[inline]
pub fn scalar_mul_add<S: RealScalar>(left: S, right: S, addend: S) -> Result<S, ScalarKernelError> {
    finite::<S>(&[left, right, addend])?;

    crate::checked_mul_add(left, right, addend).map_err(|_| ScalarKernelError::Overflow)
}

/// Adds the exact product `left · right` to `sum`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if a factor is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if the accumulator rejects the update.
#[inline]
pub fn range_add_product<S: RealScalar>(
    sum: &mut RangeAccumulator<S>,
    left: S,
    right: S,
) -> Result<(), ScalarKernelError> {
    finite::<S>(&[left, right])?;

    sum.add_product(left, right)
        .map_err(|_| ScalarKernelError::Overflow)
}

/// Adds the exact product `first · second · third` to `sum`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if a factor is NaN or infinite.
/// - [`ScalarKernelError::Overflow`] if the accumulator rejects the update.
#[inline]
pub fn range_add_triple<S: RealScalar>(
    sum: &mut RangeAccumulator<S>,
    first: S,
    second: S,
    third: S,
) -> Result<(), ScalarKernelError> {
    finite::<S>(&[first, second, third])?;

    sum.add_triple_product(first, second, third)
        .map_err(|_| ScalarKernelError::Overflow)
}

/// Rounds the exact value of `sum` once to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::Overflow`] if the rounded value overflows.
#[inline]
pub fn range_finish<S: RealScalar>(sum: &RangeAccumulator<S>) -> Result<S, ScalarKernelError> {
    finite::<S>(&[])?;

    sum.finish().map_err(|_| ScalarKernelError::Overflow)
}

/// Divides the exact value of `sum` by `denominator` with one rounding.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if `denominator` is NaN or infinite.
/// - [`ScalarKernelError::ZeroDenominator`] if `denominator` is zero.
/// - [`ScalarKernelError::Overflow`] if the rounded quotient overflows.
#[inline]
pub fn range_finish_quotient<S: RealScalar>(
    sum: &RangeAccumulator<S>,
    denominator: S,
) -> Result<S, ScalarKernelError> {
    finite::<S>(&[denominator])?;
    if denominator == S::ZERO {
        return Err(ScalarKernelError::ZeroDenominator);
    }
    sum.finish_quotient(denominator)
        .map_err(|_| ScalarKernelError::Overflow)
}

/// Returns `(sin(πx), cos(πx))`, evaluated in `f64` and rounded to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if `x` is NaN or infinite.
pub fn real_sin_cos_pi<S: RealScalar>(x: S) -> Result<(S, S), ScalarKernelError> {
    finite(&[x])?;
    let (s, c) = crate::stable::phase(x.into_f64(), 1.0);
    Ok((S::from_f64_ieee(s), S::from_f64_ieee(c)))
}

/// Returns the sine and cosine of `x` degrees, evaluated in `f64` and rounded to `S`.
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if `x` is NaN or infinite.
pub fn real_sin_cos_degrees<S: RealScalar>(x: S) -> Result<(S, S), ScalarKernelError> {
    finite(&[x])?;
    let (s, c) = crate::stable::phase(x.into_f64(), 180.0);
    Ok((S::from_f64_ieee(s), S::from_f64_ieee(c)))
}

/// Reduces a radian argument like [`crate::backend::reduce_radians`].
///
/// # Errors
///
/// - [`ScalarKernelError::NonFiniteInput`] if `value` is NaN or infinite.
pub fn real_reduce_radians(value: f64) -> Result<(i32, f64, f64), ScalarKernelError> {
    finite(&[value])?;
    Ok(crate::math::reduce_radians(value))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    fn same<S: RealScalar>(
        actual: Result<S, ScalarKernelError>,
        expected: Result<S, SciOxideError>,
    ) {
        match (actual, expected) {
            (Ok(a), Ok(b)) => assert_eq!(a.into_f64().to_bits(), b.into_f64().to_bits()),
            (Err(a), Err(b)) => {
                let a = SciOxideError::from(a);
                assert_eq!(a.kind(), b.kind());
                match (a.view(), b.view()) {
                    (
                        crate::SciOxideErrorView::DomainError {
                            argument: a,
                            reason: ar,
                            ..
                        },
                        crate::SciOxideErrorView::DomainError {
                            argument: b,
                            reason: br,
                            ..
                        },
                    ) => assert_eq!((a, ar), (b, br)),
                    (
                        crate::SciOxideErrorView::ScalarConversion {
                            from_type: a,
                            to_type: b,
                            reason: c,
                        },
                        crate::SciOxideErrorView::ScalarConversion {
                            from_type: x,
                            to_type: y,
                            reason: z,
                        },
                    ) => assert_eq!((a, b, c), (x, y, z)),
                    (
                        crate::SciOxideErrorView::NonFinite { origin: a, .. },
                        crate::SciOxideErrorView::NonFinite { origin: b, .. },
                    ) => assert_eq!(a, b),
                    (
                        crate::SciOxideErrorView::NumericalInstability { issue: a, .. },
                        crate::SciOxideErrorView::NumericalInstability { issue: b, .. },
                    ) => assert_eq!(a, b),
                    values => panic!("different kernel error families: {values:?}"),
                }
            }
            values => panic!("different kernel results: {values:?}"),
        }
    }
    #[test]
    fn bounded_transform_kernels_preserve_scalar_values_and_error_payloads() {
        macro_rules! check {
            ($scalar:ty) => {{
                for x in [
                    0.0 as $scalar,
                    -0.0,
                    -0.7,
                    0.5,
                    1.0,
                    1.25,
                    <$scalar>::from_bits(1),
                    <$scalar>::MAX,
                    <$scalar>::NAN,
                    <$scalar>::INFINITY,
                ] {
                    macro_rules! unary {
                        ($kernel:ident,$public:ident) => {
                            same($kernel(x), crate::$public(x));
                        };
                    }
                    unary!(real_softplus, softplus);
                    unary!(real_sigmoid, sigmoid);
                    unary!(real_log_sigmoid, log_sigmoid);
                    unary!(real_logit, logit);
                    unary!(real_exprel, exprel);
                    unary!(real_cosm1, cosm1);
                    unary!(real_sinc, sinc);
                    unary!(real_sincpi, sincpi);
                    unary!(real_sin_pi, sin_pi);
                    unary!(real_cos_pi, cos_pi);
                    unary!(real_tan_pi, tan_pi);
                    unary!(real_sin_degrees, sin_degrees);
                    unary!(real_cos_degrees, cos_degrees);
                    unary!(real_tan_degrees, tan_degrees);
                    for y in [
                        -2.0 as $scalar,
                        -1.0,
                        0.0,
                        0.5,
                        1.0,
                        2.0,
                        <$scalar>::MAX,
                        <$scalar>::NEG_INFINITY,
                    ] {
                        same(real_xlogy(x, y), crate::xlogy(x, y));
                        same(real_xlog1py(x, y), crate::xlog1py(x, y));
                        same(real_powm1(x, y), crate::powm1(x, y));
                        same(real_logaddexp(x, y), crate::logaddexp(x, y));
                    }
                }
            }};
        }
        check!(f32);
        check!(f64);
    }
}
