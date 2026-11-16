//! Log-sum-exp and softmax shifted by the largest base value.
//!
//! The shift `m = maxⱼ xⱼ` is taken lane by lane from base values and is not
//! differentiated, so derivatives are those of the unshifted functions. A `−∞`
//! entry is masked: its exponential is exactly zero. A NaN entry makes the shift
//! and every result of its lane NaN; nothing is checked, as for every number type.

use scioxide_core::{DomainReason, RealScalar, SciOxideError};

use crate::{BaseReal, Mask, Real};

/// The largest base value of each lane, or NaN if the lane holds a NaN (`max`
/// alone would return the other operand).
fn shift<S: RealScalar, R: Real<S>>(values: &[R]) -> R::Base {
    values
        .iter()
        .fold(R::Base::constant(S::NEG_INFINITY), |m, v| {
            let v = v.base();
            R::Base::select(m.ge(m) & v.ge(v), m.max(v), m + v)
        })
}

/// Returns `ln Σⱼ e^(xⱼ)` computed as `m + ln Σⱼ e^(xⱼ − m)`.
///
/// The derivative is the softmax of the values where the shift is finite. An
/// empty or all-masked input gives `−∞` with zero derivative; a `+∞` entry gives
/// `+∞`, with zero derivative because the shift is passive; a NaN entry gives NaN.
///
/// # Accuracy
///
/// The shift keeps every exponent at most 0 and the largest term at 1, so nothing
/// overflows and, to first order, the absolute error of the result `y` over `n`
/// values is at most about `ε·(|y| + 5n)` with an `exp` within 2 ulps.
///
/// # Errors
///
/// None at present; the signature leaves room for checked forms.
///
/// # References
///
/// P. Blanchard, D. J. Higham and N. J. Higham, "Accurately computing the
/// log-sum-exp and softmax functions", *IMA J. Numer. Anal.* 41 (2021),
/// 2311–2330.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{grad, logsumexp};
/// let g = grad(|x| logsumexp(x), &[0.0, 0.0])?;
/// assert_eq!(g.as_slice(), [0.5, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn logsumexp<S: RealScalar, R: Real<S>>(values: &[R]) -> Result<R, SciOxideError> {
    let m = shift(values);
    let shifted = R::passive(m);
    let sum: R = values.iter().map(|&v| (v - shifted).exp()).sum();
    let finite = m.is_finite();
    Ok(R::select(finite, shifted + sum.ln(), shifted))
}

fn masked_everywhere<S: RealScalar, B: BaseReal<S>>(m: B) -> bool {
    m.le(B::constant(S::NEG_INFINITY)).any()
}

/// Writes the softmax `pᵢ = e^(xᵢ − m) / Σⱼ e^(xⱼ − m)` to `out`; masked entries
/// receive `+0`.
///
/// # Accuracy
///
/// To first order each entry has relative error at most about
/// `ε·(4n + |xᵢ − m|)`: the sum carries `(n − 1)ε` and the rounded shift `xᵢ − m`
/// enters the exponent (Blanchard, Higham and Higham 2021, see [`logsumexp`]).
///
/// # Errors
///
/// - `DimensionMismatch` if `out` and `values` differ in length.
/// - `DomainError` (argument `values`, `Undefined`) if `values` is empty or every
///   entry of a lane is masked.
///
/// A NaN entry is not an error: it makes every entry of its lane NaN.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Dual, Real, softmax_into};
/// // dp/dt along x = (0, 0) + t·(1, 0) is (p₀p₁, −p₀p₁) = (¼, −¼).
/// let x = [Dual::new(0.0, 1.0), Dual::new(0.0, 0.0)];
/// let mut p = [Dual::constant(0.0); 2];
/// softmax_into(&x, &mut p)?;
/// assert_eq!(p.map(|v| v.value()), [0.5, 0.5]);
/// assert_eq!(p.map(|v| v.derivative()), [0.25, -0.25]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn softmax_into<S: RealScalar, R: Real<S>>(
    values: &[R],
    out: &mut [R],
) -> Result<(), SciOxideError> {
    crate::support::check_len(out.len(), values.len())?;
    let m = shift(values);
    if values.is_empty() || masked_everywhere::<S, _>(m) {
        return Err(masked());
    }
    let shifted = R::passive(m);
    let sum: R = values.iter().map(|&v| (v - shifted).exp()).sum();
    for (target, &value) in out.iter_mut().zip(values) {
        *target = (value - shifted).exp() / sum;
    }
    Ok(())
}

/// Writes the log-softmax `ln pᵢ = (xᵢ − m) − ln Σⱼ e^(xⱼ − m)` to `out`; masked
/// entries receive `−∞`.
///
/// # Accuracy
///
/// To first order each entry has absolute error at most about
/// `ε·(|xᵢ − m| + |ln pᵢ| + 5n)`, from the shift, the logarithm of the sum and the
/// final subtraction.
///
/// # Errors
///
/// The errors of [`softmax_into`].
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, Var, grad, log_softmax_into};
/// // ∂ ln p₀ / ∂x = e₀ − p = (½, −½) at x = (0, 0).
/// let g = grad(
///     |x| {
///         let mut out = [Var::constant(0.0); 2];
///         log_softmax_into(x, &mut out)?;
///         Ok(out[0])
///     },
///     &[0.0, 0.0],
/// )?;
/// assert_eq!(g.as_slice(), [0.5, -0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_softmax_into<S: RealScalar, R: Real<S>>(
    values: &[R],
    out: &mut [R],
) -> Result<(), SciOxideError> {
    crate::support::check_len(out.len(), values.len())?;
    let m = shift(values);
    if values.is_empty() || masked_everywhere::<S, _>(m) {
        return Err(masked());
    }
    let shifted = R::passive(m);
    let log_sum = values.iter().map(|&v| (v - shifted).exp()).sum::<R>().ln();
    for (target, &value) in out.iter_mut().zip(values) {
        *target = (value - shifted) - log_sum;
    }
    Ok(())
}

fn masked() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("values"),
        DomainReason::Undefined,
        "the input is empty or every entry is masked",
    )
}
