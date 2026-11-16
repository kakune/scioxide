//! Complete real gamma, log-gamma, digamma and polygamma kernels.
//!
//! Positive polygamma uses compensated, binary-scaled Hurwitz summation;
//! Negative arguments use a small cotangent polynomial through order 20 and
//! a constant-storage paired partial-fraction reflection at higher orders.

use crate::support::Work;
pub(crate) use crate::support::gamma_core::{
    GAMMA_ULPS, SMALL_FACTORIALS_F64, complex_log_gamma, digamma_kernel, gamma_log_parts,
    gamma_value, ln_gamma_kernel, validate_gamma_argument,
};
pub(crate) use crate::support::gamma_real as real;
use crate::support::{check_finite, project_real};

use crate::support::wide::Wide;
use crate::{
    GammaOptions, SpecialFunctionTermination,
    result::{Bound, Evaluation, Projection},
};
use scioxide_core::algorithm_id;
use scioxide_core::{RealScalar, SciOxideError};
pub(crate) fn ln_gamma<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    ln_gamma_kernel(x)
}

/// Evaluates the result before projecting it to the scalar type.
#[inline]
pub(crate) fn ln_gamma_evaluate<S: RealScalar>(
    x: S,
    options: GammaOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || ln_gamma_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    validate_gamma_argument(x)?;
    let exact_zero = x == S::ONE || x == S::TWO;
    let work = if exact_zero {
        0
    } else if x < S::ZERO {
        73
    } else {
        13
    };
    crate::support::admit_family_work(work, options.iteration_budget())?;
    let value = project_real(if x > S::ZERO {
        real::ln_positive(x.into_f64())
    } else {
        real::ln_negative(x.into_f64())
    })?;
    Ok(if exact_zero {
        Evaluation::finite(
            value,
            Bound::exact(),
            None,
            algorithm_id!("gamma", "exact_log_gamma_zero", 1),
            0,
            SpecialFunctionTermination::Exact,
        )
    } else {
        Evaluation::calibrated(
            value,
            GAMMA_ULPS,
            if x < S::ZERO {
                algorithm_id!("gamma", "conditioned_negative_log_gamma", 1)
            } else {
                algorithm_id!("gamma", "region_log_gamma", 1)
            },
            work,
            SpecialFunctionTermination::Converged,
        )
    })
}
pub(crate) fn gamma<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    validate_gamma_argument(x)?;
    let argument = x.into_f64();
    let value = match small_factorial_gamma(argument) {
        Some(value) => value,
        None => gamma_value(argument)?,
    };
    project_real(value)
}

fn small_factorial_gamma(x: f64) -> Option<f64> {
    (x >= 1.0 && x <= SMALL_FACTORIALS_F64.len() as f64 && x == x.floor())
        .then(|| SMALL_FACTORIALS_F64[x as usize - 1])
}

/// Evaluates the result before projecting it to the scalar type.
///
/// The ordinary value and signed logarithm share a report, preserving the direct value's rounding
/// while keeping logarithmic precision near Γ=1.
#[inline]
pub(crate) fn gamma_evaluate<S: RealScalar>(
    x: S,
    options: GammaOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    const DIRECT: scioxide_core::AlgorithmId = algorithm_id!("gamma", "direct_region_gamma", 1);
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    validate_gamma_argument(x)?;
    let argument = x.into_f64();
    let table_value = small_factorial_gamma(argument);
    let work = if table_value.is_some() {
        0
    } else if argument >= 16.0 {
        18
    } else {
        13
    };
    crate::support::admit_family_work(work, options.iteration_budget())?;
    if let Some(value) = table_value {
        return Ok(Evaluation::finite(
            S::try_from_f64_lossy(value)?,
            Bound::exact(),
            None,
            algorithm_id!("gamma", "factorial_table", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }
    let signed = if argument > 0.0 {
        real::gamma_positive(argument)?
    } else {
        real::gamma_negative(argument)?
    };
    let magnitude = signed.abs();
    let (ln_abs, sign) = gamma_log_parts(x)?;
    if magnitude <= S::MAX.into_f64() {
        let value = S::try_from_f64_lossy(signed)?;
        if value != S::ZERO {
            return Ok(Evaluation::logarithmic(
                sign,
                ln_abs,
                Bound::envelope(ln_abs, GAMMA_ULPS)?,
                Projection::Direct(value.abs(), Bound::envelope(value, GAMMA_ULPS)?),
                DIRECT,
                work,
            ));
        }
    }
    let projected = if magnitude > S::MAX.into_f64() {
        S::INFINITY
    } else {
        S::ZERO
    };
    Ok(Evaluation::logarithmic(
        sign,
        ln_abs,
        Bound::envelope(ln_abs, GAMMA_ULPS)?,
        Projection::Direct(projected, Bound::exact()),
        DIRECT,
        work,
    ))
}
pub(crate) fn digamma<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    digamma_kernel(x)
}

/// Evaluates the result before projecting it to the scalar type.
#[inline]
pub(crate) fn digamma_evaluate<S: RealScalar>(
    x: S,
    options: GammaOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || digamma_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    validate_gamma_argument(x)?;
    let work = 32;
    crate::support::admit_family_work(work, options.iteration_budget())?;
    digamma_evaluation(
        x,
        if x < S::ZERO {
            algorithm_id!("gamma", "digamma_reflection", 1)
        } else {
            algorithm_id!("gamma", "digamma_recurrence_asymptotic", 1)
        },
        work,
    )
}

/// Digamma with a four-unit envelope of the terms it sums: relative to `ψ`
/// on the positive axis, and absolute at the reflection's scale on the
/// negative axis, where the terms cancel near each zero.
fn digamma_evaluation<S: RealScalar>(
    x: S,
    algorithm: scioxide_core::AlgorithmId,
    work: usize,
) -> Result<Evaluation<S>, SciOxideError> {
    let (value, scale) = real::digamma_scaled(x.into_f64());
    let value = project_real(value)?;
    if x >= S::ZERO {
        return Ok(Evaluation::calibrated(
            value,
            GAMMA_ULPS,
            algorithm,
            work,
            SpecialFunctionTermination::Converged,
        ));
    }
    Ok(Evaluation::finite(
        value,
        Bound::envelope(S::try_from_f64_lossy(scale)?, GAMMA_ULPS)?,
        None,
        algorithm,
        work,
        SpecialFunctionTermination::Converged,
    ))
}
pub(crate) fn polygamma<S: RealScalar>(n: usize, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <GammaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    polygamma_evaluate(n, x, options, None)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
fn polygamma_evaluation_kernel<S: RealScalar>(
    n: usize,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    validate_gamma_argument(x)?;
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || polygamma_evaluation_kernel::<f64>(n, x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    if n == 0 {
        work.steps(13)?;
        return digamma_evaluation(x, algorithm_id!("gamma", "digamma", 1), work.used);
    }
    // The Hurwitz prefix uses n + 64 and exact binary64 integer indices.
    let largest_index = n
        .checked_add(64)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("polygamma_order")))?;
    f64::try_from_usize_exact(largest_index)?;
    if x < S::ZERO {
        let (value, scale) = polygamma_reflected(n, x, work)?;
        return Ok(Evaluation::finite(
            value,
            Bound::envelope(scale, GAMMA_ULPS)?,
            None,
            algorithm_id!("gamma", "cotangent_reflection", 1),
            work.used,
            SpecialFunctionTermination::Converged,
        ));
    }
    let value = polygamma_positive(n, x, work)?;
    Ok(Evaluation::calibrated(
        value,
        GAMMA_ULPS,
        algorithm_id!("gamma", "integer_hurwitz", 1),
        work.used,
        SpecialFunctionTermination::Converged,
    ))
}

// n!/x^n with an explicit binary exponent: neither the factorial nor its
// denominator has to be representable on its own.
fn factorial_power(n: usize, x: Wide, work: &mut Work) -> Result<(Wide, i32), SciOxideError> {
    let (_, x_exponent) = crate::support::wide::finite_parts(x.hi);
    let denominator = Wide {
        hi: x.hi.scalbn(-x_exponent),
        lo: x.lo.scalbn(-x_exponent),
    };
    let mut value = Wide::from(1.0);
    let mut scale = 0_i32;
    for index in 1..=n {
        work.try_step()?;
        value = value.mul_f64(index as f64) / denominator;
        let (_, exponent) = crate::support::wide::finite_parts(value.hi);
        let shift = exponent + 1;
        value = Wide {
            hi: value.hi.scalbn(-shift),
            lo: value.lo.scalbn(-shift),
        };
        scale = scale.checked_add(shift - x_exponent).ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::ExponentOutOfRange,
                "polygamma binary exponent exceeds i32",
            )
        })?;
    }
    Ok((value, scale))
}

fn polygamma_positive_wide(
    n: usize,
    q: Wide,
    cancellation: bool,
    work: &mut Work,
) -> Result<Wide, SciOxideError> {
    let order = n + 1;
    // Twofold continuation reserves roughly 2^-93 for the tail before a
    // potentially cancelling negative-axis recurrence. Ordinary projection
    // reserves seven eighths of one binary64 epsilon for rounding.
    let relative = if cancellation {
        2048.0 * f64::EPSILON * f64::EPSILON
    } else {
        0.125 * f64::EPSILON
    };
    let sum = crate::support::hurwitz::positive(
        Wide::from(order as f64),
        q,
        n + if cancellation { 64 } else { 16 },
        relative,
        |_, x| Ok((q / x).powu(order)),
        &mut || work.try_step(),
    )?;
    let (factor, scale) = factorial_power(n, q, work)?;
    let value = factor * (sum / q);
    let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    Ok(Wide {
        hi: (sign * value.hi).scalbn(scale),
        lo: (sign * value.lo).scalbn(scale),
    })
}

fn project_polygamma<S: RealScalar>(value: Wide) -> Result<S, SciOxideError> {
    if !value.hi.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "the polygamma magnitude exceeds the scalar range",
        ));
    }
    // A value below the scalar range keeps the sign of its leading word.
    let value = value.value().copysign(value.hi);
    check_finite(S::try_from_f64_lossy(value)?)
}

fn polygamma_positive<S: RealScalar>(n: usize, x: S, work: &mut Work) -> Result<S, SciOxideError> {
    let argument = x.into_f64();
    if n <= 4 && (1.0e-60..=1.0e60).contains(&argument) {
        // Low orders need neither a separately scaled factorial nor powers
        // of q/(q+k). One corrected reciprocal supplies each positive term.
        let prefix = (16.0 - argument).ceil().max(0.0) as usize;
        let sum = crate::support::hurwitz::positive(
            Wide::from((n + 1) as f64),
            Wide::from(argument),
            prefix,
            f64::EPSILON / 16384.0,
            |_, x| Ok((Wide::from(1.0) / x).powu(n + 1)),
            &mut || work.try_step(),
        )?;
        let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
        return project_polygamma(sum.mul_f64(sign * SMALL_FACTORIALS_F64[n]));
    }

    project_polygamma(polygamma_positive_wide(
        n,
        Wide::from(x.into_f64()),
        false,
        work,
    )?)
}

// π's low word and Taylor terms retain the exact reduced binary argument.
// The complementary angle keeps cotangent relatively accurate beside half-integers.
fn reflection_cotangent(x: f64, work: &mut Work) -> Result<Wide, SciOxideError> {
    let reduced = x - x.round();
    let complementary = reduced.abs() > 0.25;
    let argument = if complementary {
        0.5 - reduced.abs()
    } else {
        reduced.abs()
    };
    let pi = crate::support::constants::PI;
    let angle = pi.mul_f64(argument);
    let square = -(angle * angle);
    let mut sine = angle;
    let mut cosine = Wide::from(1.0);
    let mut sine_term = sine;
    let mut cosine_term = cosine;
    for k in 1..=16 {
        work.try_step()?;
        sine_term = sine_term * square / Wide::from((2 * k * (2 * k + 1)) as f64);
        cosine_term = cosine_term * square / Wide::from((2 * k * (2 * k - 1)) as f64);
        sine = sine + sine_term;
        cosine = cosine + cosine_term;
    }
    let cotangent = if complementary {
        sine / cosine
    } else {
        cosine / sine
    };
    Ok(cotangent.mul_f64(if reduced < 0.0 { -1.0 } else { 1.0 }))
}

fn polygamma_reflected<S: RealScalar>(
    n: usize,
    x: S,
    work: &mut Work,
) -> Result<(S, S), SciOxideError> {
    let argument = x.into_f64();
    let reflected = polygamma_positive_wide(n, Wide::from(1.0) - Wide::from(argument), true, work)?;
    let cotangent = reflection_cotangent(argument, work)?;
    // Every even derivative of cot vanishes exactly at a half-integer.
    if n.is_multiple_of(2) && cotangent.hi == 0.0 {
        let value = project_polygamma(reflected)?;
        return Ok((value, value.abs()));
    }
    if n > 20 {
        let derivative = polygamma_reflection_series(n, argument, work)?;
        let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
        let value = project_polygamma((reflected - derivative).mul_f64(sign))?;
        let scale = S::try_from_f64_lossy(reflected.value().abs() + derivative.value().abs())?;
        return Ok((value, scale));
    }
    // Q_n = (-1)^n P_n/n! has nonnegative coefficients. Store one row,
    // retaining each old left neighbour while replacing it from left to right.
    let mut coefficients = [Wide::ZERO; 22];
    coefficients[1] = Wide::from(1.0);
    for order in 1..=n {
        let mut previous = Wide::ZERO;
        for index in 0..=order + 1 {
            work.try_step()?;
            let current = coefficients[index];
            let left = if index == 0 {
                Wide::ZERO
            } else {
                previous.mul_f64((index - 1) as f64)
            };
            let right = if index < order {
                coefficients[index + 1].mul_f64((index + 1) as f64)
            } else {
                Wide::ZERO
            };
            coefficients[index] = (left + right).div_f64(order as f64);
            previous = current;
        }
    }
    let square = cotangent * cotangent;
    let parity = (n + 1) % 2;
    let mut polynomial = Wide::ZERO;
    for index in (parity..n + 2).step_by(2).rev() {
        work.try_step()?;
        polynomial = polynomial * square + coefficients[index];
    }
    if parity == 1 {
        polynomial = polynomial * cotangent;
    }
    let pi = crate::support::constants::PI;
    let (factor, scale) = factorial_power(n, Wide::from(1.0) / pi, work)?;
    let term = polynomial * pi * factor;
    let derivative = Wide {
        hi: term.hi.scalbn(scale),
        lo: term.lo.scalbn(scale),
    };
    let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
    let value = project_polygamma((reflected - derivative).mul_f64(sign))?;
    let scale = S::try_from_f64_lossy(reflected.value().abs() + derivative.value().abs())?;
    Ok((value, scale))
}

/// n! times the bilateral inverse-power sum, normalized by the nearest pole.
/// Pair distances u=k+d and v=k+1-d. For even n their difference is evaluated
/// as -expm1(-(n+1) log1p((v-u)/u)), keeping half-integer displacements.
/// The unsummed positive terms are bounded by their first term plus integral;
/// twice that bound also bounds the paired difference, including cancellation.
fn polygamma_reflection_series(n: usize, x: f64, work: &mut Work) -> Result<Wide, SciOxideError> {
    let reduced = x - x.round();
    let d = Wide::from(reduced.abs());
    let displacement = Wide::from(1.0) - d.mul_f64(2.0);
    let order = n + 1;
    let even = n.is_multiple_of(2);
    let mut sum = Wide::ZERO;
    let relative = 2048.0 * f64::EPSILON * f64::EPSILON;
    for k in 0usize.. {
        work.try_step()?;
        let u = d + Wide::from(f64::try_from_usize_exact(k)?);
        let leading = (d / u).powu(order);
        let pair = if even {
            -((displacement / u).ln_1p().mul_f64(-(order as f64))).exp_m1()
        } else {
            Wide::from(1.0) + (u / (u + displacement)).powu(order)
        };
        sum = sum + leading * pair;
        let next = u + Wide::from(1.0);
        let bound = 2.0 * (d / next).powu(order).value() * (1.0 + next.value() / n as f64);
        if bound <= relative * sum.value().abs() {
            break;
        }
    }
    let (factor, exponent) = factorial_power(n, d, work)?;
    let value = factor * (sum / d);
    let sign = if even && reduced < 0.0 { -1.0 } else { 1.0 };
    Ok(Wide {
        hi: (sign * value.hi).scalbn(exponent),
        lo: (sign * value.lo).scalbn(exponent),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "gamma_tests.rs"]
mod tests;

pub(crate) fn polygamma_evaluate<S: RealScalar>(
    n: usize,
    x: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = polygamma_evaluation_kernel(n, x, &mut work);
    work.check()?;
    result
}
