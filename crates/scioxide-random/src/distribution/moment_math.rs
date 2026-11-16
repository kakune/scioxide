//! Cancellation-free scalar moment combinations, evaluated before final rounding.
use scioxide_core::{NumericalIssue, RealScalar, SciOxideError};

use crate::distribution::LawOptions;
use crate::distribution::quadrature::log_integral;
use crate::distribution::special::{average, log_gamma_ratio};
use crate::law::{charge, constant as number, finite};

pub(super) fn cancellation() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::Cancellation,
        "moment combination does not resolve the requested accuracy",
    )
}

/// Check a positive variance after combining its moment terms.
pub(super) fn positive_variance(value: f64) -> Result<f64, SciOxideError> {
    if value <= 0.0 || value.is_nan() {
        return Err(cancellation());
    }
    finite(value)
}

/// Round a positive mathematical variance once, rejecting lost range.
pub(super) fn variance<S: RealScalar>(log_value: f64) -> Result<S, SciOxideError> {
    let value = finite(log_value.exp())?;
    let rounded = S::try_from_f64_lossy(value).map_err(|_| {
        SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "variance exceeds the destination scalar range",
        )
    })?;
    if rounded == S::ZERO
        || (rounded.into_f64().ln() - log_value).abs() > S::SQRT_EPSILON.into_f64()
    {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Underflow,
            "positive variance loses relative accuracy at the scalar underflow boundary",
        ));
    }
    finite(rounded)
}

/// Log variance from Gamma(base+r*shift)/Gamma(base), r=1,2.
pub(super) fn gamma_variance<S: RealScalar>(base: f64, shift: f64) -> Result<S, SciOxideError> {
    variance(log_gamma_variance::<S>(base, shift)?)
}

/// Logarithm of the gamma moment variance, before scale factors or final rounding.
pub(super) fn log_gamma_variance<S: RealScalar>(
    base: f64,
    shift: f64,
) -> Result<f64, SciOxideError> {
    if base + shift.min(2.0 * shift) <= 0.0 {
        return Err(crate::law::undefined_moment(
            "requested gamma moment does not exist",
        ));
    }
    let first = log_gamma_ratio(base, shift)?;
    let log_delta = if shift.abs() <= base * 0.125 {
        // The exact second difference is shift² times the positive double
        // integral of trigamma. Never subtract the two first differences.
        let integral =
            average(|s| average(|t| scioxide_special::polygamma(1, base + shift * (s + t))))?;
        2.0 * shift.abs().ln() + integral.ln()
    } else {
        let second = log_gamma_ratio(base, 2.0 * shift)?;
        let delta = second - 2.0 * first;
        let error = 32.0 * f64::EPSILON * (second.abs() + 2.0 * first.abs());
        if delta <= 0.0 || error > S::SQRT_EPSILON.into_f64() * delta {
            return Err(cancellation());
        }
        delta.ln()
    };
    let log_relative = if log_delta < f64::EPSILON.ln() {
        log_delta
    } else {
        crate::law::log_expm1(log_delta.exp())?
    };
    Ok(2.0 * first + log_relative)
}

/// Gamma(shape,1) entropy with the linear and shape*log(shape) terms removed.
pub(super) fn gamma_entropy(shape: f64) -> Result<f64, SciOxideError> {
    if shape < 16.0 {
        return finite(
            shape
                + scioxide_special::ln_gamma(shape)?
                + (1.0 - shape) * scioxide_special::digamma(shape)?,
        );
    }
    // Bernoulli B_(2n)/(2n). At shape >= 16 the first omitted contribution
    // (n=9) is below 4e-19. Stirling's remainder is bounded by that term.
    const B: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 120.0,
        1.0 / 252.0,
        -1.0 / 240.0,
        1.0 / 132.0,
        -691.0 / 32760.0,
        1.0 / 12.0,
        -3617.0 / 8160.0,
    ];
    let inverse = shape.recip();
    let square = inverse * inverse;
    let mut power = inverse;
    let mut correction = -0.5 * inverse;
    for (index, b) in B.iter().enumerate() {
        correction += b * power * (1.0 / (2 * index + 1) as f64 + 1.0 - inverse);
        power *= square;
    }
    finite(0.5 * (core::f64::consts::TAU.ln() + 1.0 + shape.ln()) + correction)
}

pub(super) fn student_entropy(nu: f64) -> Result<f64, SciOxideError> {
    let z = nu * 0.5;
    if z < 4.0 {
        return finite(
            0.5 * nu.ln()
                + scioxide_special::ln_beta(z, 0.5)?
                + (z + 0.5) * (scioxide_special::digamma(z + 0.5)? - scioxide_special::digamma(z)?),
        );
    }
    let remainder = crate::distribution::special::log_gamma_ratio_remainder(z, 0.5)?;
    // Scale trigamma before multiplying by z; no tiny digamma difference is formed.
    let derivative = average(|t| Ok(z * scioxide_special::polygamma(1, z + 0.5 * t)?))?;
    finite(0.5 * core::f64::consts::TAU.ln() - remainder + 0.5 * (1.0 + 0.5 / z) * derivative)
}

/// Bradford raw moments: integrate x^r/(1+c*x) by its geometric series.
pub(super) fn bradford(shape: f64) -> (f64, f64) {
    if shape <= 0.125 {
        let mut normalization = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        let mut power = 1.0;
        // At c <= 1/8, the geometric remainder after 20 terms is < 1e-18.
        for j in 0..20 {
            normalization += power / (j + 1) as f64;
            first += power / (j + 2) as f64;
            second += power / (j + 3) as f64;
            power *= -shape;
        }
        let mean = first / normalization;
        (mean, second / normalization - mean * mean)
    } else {
        let inverse_log = shape.ln_1p().recip();
        let mean = inverse_log - shape.recip();
        (mean, 0.5 * inverse_log - mean / shape - mean * mean)
    }
}

pub(super) fn quantile_abs_moment<
    S: RealScalar,
    D: crate::Quantile<S> + crate::InverseSurvival<S>,
>(
    law: &D,
    order: u32,
    low: S,
    high: S,
    options: LawOptions<S>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    if low >= high {
        return Ok(S::ZERO);
    }
    if low == S::ZERO && high == S::ONE {
        return full_quantile_moment(law, order, options, left);
    }
    if low != S::ZERO && high != S::ONE {
        return Err(crate::law::invalid(
            scioxide_core::argument!("self"),
            "moment interval must touch a probability endpoint",
        ));
    }
    let tol = options.tolerance.relative();
    // E|X|^r over a one-sided probability window is \int |Q(p)|^r dp. The
    // substitution p = width * u^4 (dp = 4 width u^3 du) clusters nodes near
    // the probability endpoint, where the quantile of a heavy-tailed law
    // diverges, and turns the endpoint singularity into an integrable zero.
    // Upper windows use `isf` so tail quantiles keep full relative precision.
    let power = number::<S>(order as f64)?;
    let width = high - low;
    let four = number::<S>(4.)?;
    let three = number::<S>(3.)?;
    let mut f = |u: S| {
        charge(left)?;
        let tail = width * u.powi(4);
        if tail == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let q = if high == S::ONE {
            crate::law::scoped(left, |context| law.isf_scoped(tail, context))?
        } else {
            crate::law::scoped(left, |context| law.quantile_scoped(tail, context))?
        };
        Ok(if q == S::ZERO {
            S::NEG_INFINITY
        } else {
            four.ln() + width.ln() + three * u.ln() + power * q.abs().ln()
        })
    };
    crate::law::finite(log_integral(S::ZERO, S::ONE, tol, &mut f, None)?.exp())
}
pub(super) fn checked_variance<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value <= S::ZERO || !value.is_finite() {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "positive variance could not be resolved",
        ))
    } else {
        Ok(value)
    }
}
fn full_quantile_moment<S: RealScalar, D: crate::Quantile<S> + crate::InverseSurvival<S>>(
    law: &D,
    order: u32,
    options: LawOptions<S>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let tol = options.tolerance.relative();
    // Whole-line moment: split at p = 1/2 and integrate each half from its
    // endpoint with the same u^4 substitution as `quantile_abs_moment`, the
    // lower half through `quantile` and the upper half through `isf`.
    let power = number::<S>(order as f64)?;
    let half = S::HALF;
    let two = S::TWO;
    let three = two + S::ONE;
    let piece = |upper: bool, left: &mut usize| {
        let mut f = |u: S| {
            charge(left)?;
            let tail = half * u.powi(4);
            if tail == S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            let q = if upper {
                crate::law::scoped(left, |context| law.isf_scoped(tail, context))?
            } else {
                crate::law::scoped(left, |context| law.quantile_scoped(tail, context))?
            };
            Ok(if q == S::ZERO {
                S::NEG_INFINITY
            } else {
                two.ln() + three * u.ln() + power * q.abs().ln()
            })
        };
        log_integral(S::ZERO, S::ONE, tol, &mut f, None)
    };
    crate::law::finite(crate::law::logadd(piece(false, left)?, piece(true, left)?)?.exp())
}
pub(super) fn quantile_mean<S: RealScalar, D: crate::Quantile<S> + crate::InverseSurvival<S>>(
    law: &D,
    options: LawOptions<S>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    full_quantile_moment(law, 1, options, left)
}
pub(super) fn quantile_variance<
    S: RealScalar,
    D: crate::Quantile<S> + crate::InverseSurvival<S>,
>(
    law: &D,
    options: LawOptions<S>,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    let mean = full_quantile_moment(law, 1, options, left)?;
    let second = full_quantile_moment(law, 2, options, left)?;
    variance_from_moments(second, mean)
}

pub(super) fn variance_from_moments<S: RealScalar>(second: S, mean: S) -> Result<S, SciOxideError> {
    variance_from_moments_with_accuracy(second, mean, S::EPSILON, S::SQRT_EPSILON)
}

/// Reject raw-moment cancellation against both evaluation and rounding error.
pub(super) fn variance_from_moments_with_accuracy<S: RealScalar>(
    second: S,
    mean: S,
    relative_error: S,
    tolerance: S,
) -> Result<S, SciOxideError> {
    let square = mean * mean;
    let value = finite(second - square)?;
    // Scale before adding, and retain one subnormal ULP even when the relative
    // bound itself underflows. Otherwise tiny raw moments can appear exact.
    let scale = second.abs().max(square);
    let error = relative_error * second.abs()
        + (S::TWO * relative_error) * square
        + number::<S>(4.0)? * S::EPSILON * scale
        + S::MIN_POSITIVE * S::EPSILON;
    if value <= S::ZERO || error > tolerance * value {
        if scale < S::MIN_POSITIVE {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "raw moments lose relative accuracy at the scalar underflow boundary",
            ));
        }
        return Err(cancellation());
    }
    Ok(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod error_contract_tests {
    use super::*;
    #[test]
    fn nonpositive_moment_differences_are_cancellation() {
        for value in [0.0, -1.0, f64::NAN] {
            crate::error_assertions::numerical(
                positive_variance(value).unwrap_err(),
                NumericalIssue::Cancellation,
            );
        }
        crate::error_assertions::numerical(
            positive_variance(f64::INFINITY).unwrap_err(),
            NumericalIssue::Overflow,
        );
        assert_eq!(positive_variance(1.0).unwrap(), 1.0);
    }
}
