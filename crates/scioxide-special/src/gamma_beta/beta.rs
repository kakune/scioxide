//! Beta function family: B(a,b), ln B(a,b), regularized incomplete beta,
//! and inverse incomplete beta.
//!
//! Primitive `f32`/`f64` values use small integer/half-integer recurrences,
//! conditioned gamma ratios and stable logarithmic differences. Scalar
//! polynomial tails and Stirling corrections carry rounding uncertainty;
//! ambiguous log-beta rounding keeps the full two-term calculation. Near
//! its zeros, core's exact accumulator preserves cancelling components.
//! The incomplete integral applies symmetry, finite integer tails, an endpoint
//! power series and `f64` modified Lentz. An `f64` linear tail (`pow`
//! power factor, positive DLMF 8.17.7/8.17.8 series) is returned when its
//! first-order bound proves the envelope; a conditioned scalar normalization and
//! tail then precede compensated refinement, which reuses the recurrence.
//! Small integer/half-integer normalizations share the complete beta recurrence.
//! Integer-tail powers must remain normal; otherwise compensated logarithmic
//! evaluation preserves the complete tail before destination rounding.

use crate::support::Work;
use crate::support::wide::ln as ln_wide;
use crate::support::{check_finite, check_positive, checked_probability, require_finite_param};
mod cancellation;

use crate::gamma_beta::gamma::SMALL_FACTORIALS_F64;
use crate::gamma_beta::gamma::real::ln_positive_wide;
use crate::support::constants::LOG_SQRT_TWO_PI;
use crate::support::wide::Wide;
use crate::util::is_exact;
use crate::{
    BetaOptions, SpecialFunctionTermination,
    result::{Bound, Evaluation, Projection},
};
use scioxide_core::{AlgorithmComponent, algorithm_component};

/// Envelopes of the fixed beta kernels on the tested range, in ulps of the result.
const BETA_ULPS: f64 = 4.0;
const INCOMPLETE_ULPS: f64 = 32.0;
use scioxide_core::{RealScalar, SciOxideError};
pub(crate) fn ln_beta<S: RealScalar>(a: S, b: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    ln_beta_evaluate(a, b, options)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
pub(crate) fn ln_beta_evaluate<S: RealScalar>(
    a: S,
    b: S,
    options: BetaOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    check_positive(
        a,
        scioxide_core::argument!("a"),
        "beta parameter a must be positive",
    )?;
    check_positive(
        b,
        scioxide_core::argument!("b"),
        "beta parameter b must be positive",
    )?;
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || ln_beta_evaluate::<f64>(a.into_f64(), b.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    let work = if a == S::ONE || b == S::ONE { 0 } else { 256 };
    crate::support::admit_family_work(work, options.iteration_budget())?;
    let value = if a == S::ONE {
        -b.ln()
    } else if b == S::ONE {
        -a.ln()
    } else {
        S::try_from_f64_lossy(ln_beta_value(a.into_f64(), b.into_f64())?)?
    };
    Ok(Evaluation::calibrated(
        value,
        BETA_ULPS,
        scioxide_core::algorithm_id!("beta", "conditioned_gamma_ratio", 1),
        work,
        SpecialFunctionTermination::Converged,
    ))
}

/// Accept the cheaper approximation only if its uncertainty cannot change
/// the rounded log-beta. This preserves the measured half-ulp envelope.
fn ln_beta_value(a: f64, b: f64) -> Result<f64, SciOxideError> {
    let (value, uncertainty) = ln_beta_kernel(a, b, true)?;
    let rounded = value.value();
    let spacing = (rounded.next_up() - rounded).min(rounded - rounded.next_down());
    if (value - Wide::from(rounded)).value().abs() + uncertainty < 0.5 * spacing {
        Ok(rounded)
    } else {
        Ok(ln_beta_wide(a, b)?.value())
    }
}

pub(crate) fn ln_beta_wide(a: f64, b: f64) -> Result<Wide, SciOxideError> {
    Ok(ln_beta_kernel(a, b, false)?.0)
}

/// Scalar log normalization and absolute uncertainty for probability kernels.
/// These callers condition the error on the requested probability or quantile;
/// they do not need the public log-beta contract near its zeros.
pub(crate) fn probability_normalization(a: f64, b: f64) -> (f64, f64) {
    if let Some(beta) = beta_small_lattice_f64(a, b) {
        let value = beta.ln();
        return (
            value,
            (beta.next_up() - beta) / beta + 4.0 * (value.abs().next_up() - value.abs()),
        );
    }
    let (ga, ea) = super::incomplete_gamma::probability_normalization(a);
    let (gb, eb) = super::incomplete_gamma::probability_normalization(b);
    let sum = Wide::from(a) + Wide::from(b);
    let (gs, es) = super::incomplete_gamma::probability_normalization(sum.hi);
    let value = (ga + gb) - gs;
    // |psi(s)| <= 1/s + |log(s)| + 1 covers the displacement of a+b.
    let error = ea
        + eb
        + es
        + sum.lo.abs() * (1.0 / sum.hi + sum.hi.ln().abs() + 1.0)
        + f64::EPSILON * ((ga + gb).abs() + value.abs());
    (value, error)
}

fn ln_beta_kernel(a: f64, b: f64, fast: bool) -> Result<(Wide, f64), SciOxideError> {
    if a == 1.0 {
        return Ok((-Wide::from(b).ln(), 0.0));
    }
    if b == 1.0 {
        return Ok((-Wide::from(a).ln(), 0.0));
    }
    if let Some(value) = beta_small_lattice(a, b) {
        return Ok((ln_wide(value), 0.0));
    }
    let (a, b) = (a.min(b), a.max(b));
    let a = Wide::from(a);
    let b = Wide::from(b);
    let sum = a + b;
    if !sum.hi.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "a + b is not representable as a finite scalar",
        ));
    }
    let half = Wide::from(0.5);
    if b.hi < 16.0 || (fast && b.hi < 64.0) {
        if a.hi < 1.0e-150 {
            return Ok((
                ln_positive_wide(a) + ln_positive_wide(b) - ln_positive_wide(sum),
                0.0,
            ));
        }
        let (ratio, uncertainty) = if fast {
            crate::support::gamma_real::beta_moderate_fast(a, b)
        } else {
            (crate::support::gamma_real::beta_moderate(a, b), 0.0)
        };
        return Ok(if (ratio.hi - 1.0).abs() < f64::EPSILON.sqrt() {
            (cancellation::ln_beta(a.hi, b.hi)?, 0.0)
        } else {
            (ln_wide(ratio), 2.0 * uncertainty)
        });
    }
    let correction = |x: Wide| {
        if fast {
            let inverse = 1.0 / x.hi;
            let value = crate::support::gamma_series::stirling_f64(inverse, inverse * inverse);
            (Wide::from(value), 4.0 * f64::EPSILON * value.abs())
        } else {
            (crate::support::gamma_series::stirling_wide(x), 0.0)
        }
    };
    let (cb, eb) = correction(b);
    let (cs, es) = correction(sum);
    let ratio = a / b;
    let log_b_ratio = -if ratio.hi >= 0.125 {
        ln_wide(Wide::from(1.0) + ratio)
    } else {
        ratio.ln_1p()
    };
    let difference = (b - half) * log_b_ratio + a + cb - cs;
    if a.hi < 32.0 {
        let logarithm = ln_wide(sum);
        let (complete, ec) = if fast && a.hi >= 1.0e-150 {
            let (gamma, uncertainty) = crate::support::gamma_real::gamma_positive_fast(a);
            (ln_wide(gamma), 2.0 * uncertainty)
        } else {
            (ln_positive_wide(a), 0.0)
        };
        let value = (complete - a * logarithm) + difference;
        if a.hi < 1.0 && value.hi.abs() < f64::EPSILON.sqrt() * complete.hi.abs() {
            return Ok((cancellation::ln_beta(a.hi, b.hi)?, 0.0));
        }
        return Ok((value, ec + eb + es));
    }
    let (ca, ea) = correction(a);
    Ok((
        (a - half) * ln_wide(a / sum) + (b - half) * log_b_ratio - half * ln_wide(sum)
            + LOG_SQRT_TWO_PI
            + ca
            + cb
            - cs,
        ea + eb + es,
    ))
}
pub(crate) fn beta<S: RealScalar>(a: S, b: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    beta_evaluate(a, b, options)?.value()
}

/// Evaluates the result before projecting it to the scalar type.
pub(crate) fn beta_evaluate<S: RealScalar>(
    a: S,
    b: S,
    options: BetaOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    check_positive(
        a,
        scioxide_core::argument!("a"),
        "beta parameter a must be positive",
    )?;
    check_positive(
        b,
        scioxide_core::argument!("b"),
        "beta parameter b must be positive",
    )?;
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || beta_evaluate::<f64>(a.into_f64(), b.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    let work = if a == S::ONE || b == S::ONE {
        0
    } else if a + b <= S::try_from_f64_lossy(32.0)?
        && (a + a) == (a + a).floor()
        && (b + b) == (b + b).floor()
    {
        39
    } else {
        256
    };
    crate::support::admit_family_work(work, options.iteration_budget())?;
    if is_exact(a, S::ONE) || is_exact(b, S::ONE) {
        let shape = if is_exact(a, S::ONE) { b } else { a };
        let value = S::ONE / shape;
        let algorithm = scioxide_core::algorithm_id!("beta", "unit_shape", 1);
        if !value.is_finite() {
            let logarithm = -shape.ln();
            return Ok(Evaluation::logarithmic(
                S::ONE,
                logarithm,
                Bound::envelope(logarithm, BETA_ULPS)?,
                Projection::Direct(value, Bound::rounded(value)),
                algorithm,
                0,
            ));
        }
        return Ok(Evaluation::finite(
            value,
            Bound::rounded(value),
            None,
            algorithm,
            0,
            SpecialFunctionTermination::Converged,
        ));
    }
    if let Some(value) = beta_small_lattice_f64(a.into_f64(), b.into_f64()) {
        let value = S::try_from_f64_lossy(value)?;
        // A twofold lattice product rounded once.
        let value = check_finite(value)?;
        return Ok(Evaluation::finite(
            value,
            Bound::rounded(value),
            None,
            scioxide_core::algorithm_id!("beta", "small_lattice", 1),
            work,
            SpecialFunctionTermination::Converged,
        ));
    }
    // With both shapes below 64, all three recurrence products stay finite
    // (a+b < 128). Direct projection avoids an exponential rounding as well.
    if a.min(b).into_f64() >= 1.0e-150 && a.max(b).into_f64() < 64.0 {
        let (ratio, uncertainty) = crate::support::gamma_real::beta_moderate_fast(
            Wide::from(a.into_f64()),
            Wide::from(b.into_f64()),
        );
        let rounded = ratio.value();
        let spacing = (rounded.next_up() - rounded).min(rounded - rounded.next_down());
        let value = if uncertainty * rounded.abs() <= 0.5 * spacing {
            rounded
        } else {
            crate::support::gamma_real::beta_moderate(
                Wide::from(a.into_f64()),
                Wide::from(b.into_f64()),
            )
            .value()
        };
        let value = S::try_from_f64_lossy(value)?;
        return Ok(Evaluation::calibrated(
            value,
            BETA_ULPS,
            scioxide_core::algorithm_id!("beta", "direct_gamma_ratio", 1),
            work,
            SpecialFunctionTermination::Converged,
        ));
    }
    let logarithm = ln_beta_wide(a.into_f64(), b.into_f64())?;
    let log_value = S::try_from_f64_lossy(logarithm.value())?;
    let projection = crate::result::positive_projection(logarithm.exp())?;
    Ok(Evaluation::logarithmic(
        S::ONE,
        log_value,
        Bound::envelope(log_value, BETA_ULPS)?,
        Projection::Direct(projection, Bound::envelope(projection, BETA_ULPS)?),
        scioxide_core::algorithm_id!("beta", "conditioned_gamma_ratio", 1),
        work,
    ))
}

fn beta_small_lattice_f64(a: f64, b: f64) -> Option<f64> {
    beta_small_lattice(a, b).map(Wide::value)
}

/// The product of doubled half-integer factors while it stays exact (< 2^53).
fn exact_doubled_product(mut factors: impl Iterator<Item = f64>) -> Option<f64> {
    factors.try_fold(1.0_f64, |product, factor| {
        let next = product * (2.0 * factor);
        (next < 9_007_199_254_740_992.0).then_some(next)
    })
}

fn beta_small_lattice(a: f64, b: f64) -> Option<Wide> {
    let twice_a = a * 2.0;
    let twice_b = b * 2.0;
    if a <= 0.0
        || b <= 0.0
        || a + b > 32.0
        || twice_a != twice_a.floor()
        || twice_b != twice_b.floor()
    {
        return None;
    }
    let (base, start_a, start_b) = match (
        (twice_a as usize).is_multiple_of(2),
        (twice_b as usize).is_multiple_of(2),
    ) {
        (true, true) => {
            let a = a as usize;
            let b = b as usize;
            if a + b <= SMALL_FACTORIALS_F64.len() {
                return Some(
                    Wide::from(SMALL_FACTORIALS_F64[a - 1])
                        .mul_f64(SMALL_FACTORIALS_F64[b - 1])
                        .div_f64(SMALL_FACTORIALS_F64[a + b - 1]),
                );
            }
            (1.0, 1.0, 1.0)
        }
        (false, false) => (core::f64::consts::PI, 0.5, 0.5),
        (false, true) => (2.0, 0.5, 1.0),
        (true, false) => (2.0, 1.0, 0.5),
    };
    let a_steps = (a - start_a) as usize;
    let b_steps = (b - start_b) as usize;
    let base = if base == core::f64::consts::PI {
        crate::support::constants::PI
    } else {
        Wide::from(base)
    };
    // Doubled step factors are integers. While both products stay below
    // 2^53 they are exact, and one two-word product and quotient finish B.
    let numerator = exact_doubled_product(
        (0..a_steps)
            .map(|step| start_a + step as f64)
            .chain((0..b_steps).map(|step| start_b + step as f64)),
    );
    let denominator = exact_doubled_product(
        (0..a_steps)
            .map(|step| start_a + start_b + step as f64)
            .chain((0..b_steps).map(|step| a + start_b + step as f64)),
    );
    if let (Some(numerator), Some(denominator)) = (numerator, denominator) {
        return Some(base.mul_f64(numerator).div_f64(denominator));
    }
    let after_a = (0..a_steps).fold(base, |value, step| {
        let current = start_a + step as f64;
        value.mul_f64(current).div_f64(current + start_b)
    });
    Some((0..b_steps).fold(after_a, |value, step| {
        let current = start_b + step as f64;
        value.mul_f64(current).div_f64(a + current)
    }))
}
#[inline]
pub(crate) fn beta_inc<S: RealScalar>(a: S, b: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    beta_inc_value(a, b, x)
}

/// The plain incomplete beta: the detailed kernel.s value without its report.
fn beta_inc_value<S: RealScalar>(a: S, b: S, x: S) -> Result<S, SciOxideError> {
    if let Some(value) = crate::support::delegate_f32::<S, _, _>(
        || beta_inc_value::<f64>(a.into_f64(), b.into_f64(), x.into_f64()),
        S::try_from_f64_lossy,
    ) {
        return value;
    }
    if let Some(value) = forward_endpoint(a, b, x)? {
        return Ok(value);
    }
    if is_exact(a, S::ONE) && is_exact(b, S::ONE) {
        return Ok(x);
    }
    if is_exact(a, b) && is_exact(x, S::HALF) {
        return Ok(S::HALF);
    }
    let mut work = Work::new(BetaOptions::default().iteration_budget().get());
    Ok(beta_inc_evaluation_with_work(a, b, x, &mut work)?.value)
}

/// Evaluates the result before projecting it to the scalar type.
fn beta_inc_kernel<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || beta_inc_kernel::<f64>(a.into_f64(), b.into_f64(), x.into_f64(), work),
        |wide| wide.narrow(),
    ) {
        return result;
    }

    if let Some(value) = forward_endpoint(a, b, x)? {
        return Ok(Evaluation::finite(
            value,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("beta", "forward_endpoint", 1),
            work.used,
            if a == S::ZERO || b == S::ZERO {
                SpecialFunctionTermination::Limit
            } else {
                SpecialFunctionTermination::Exact
            },
        ));
    }

    // Special case: I_x(1,1) = x
    let one = S::ONE;
    if is_exact(a, one) && is_exact(b, one) {
        return Ok(Evaluation::finite(
            x,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("beta", "uniform_identity", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }
    if is_exact(a, b) && is_exact(x, S::HALF) {
        return Ok(Evaluation::finite(
            S::HALF,
            Bound::exact(),
            None,
            scioxide_core::algorithm_id!("beta", "symmetric_midpoint", 1),
            0,
            SpecialFunctionTermination::Exact,
        ));
    }

    let evaluation = beta_inc_evaluation_with_work(a, b, x, work)?;
    Ok(Evaluation::finite(
        evaluation.value,
        Bound::envelope(evaluation.value, INCOMPLETE_ULPS)?,
        None,
        scioxide_core::AlgorithmId::from_components(
            algorithm_component!("beta"),
            evaluation.branch_id,
            1,
        ),
        evaluation.iteration_count,
        SpecialFunctionTermination::Converged,
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct BetaIncEvaluation<S> {
    value: S,
    branch_id: AlgorithmComponent,
    iteration_count: usize,
}

fn beta_inc_evaluation_with_work<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    work: &mut Work,
) -> Result<BetaIncEvaluation<S>, SciOxideError> {
    beta_inc_core(a, b, x, true, work)
}

fn beta_inc_core<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    specialized: bool,
    work: &mut Work,
) -> Result<BetaIncEvaluation<S>, SciOxideError> {
    let one = S::ONE;
    let sum = a + b;
    let threshold = if sum.is_finite() {
        (a + one) / (sum + S::TWO)
    } else {
        let scale = a.max(b);
        let inverse_scale = one / scale;
        (a / scale + inverse_scale) / (a / scale + b / scale + S::TWO * inverse_scale)
    };
    let (tail_a, tail_b, tail_x, complement) = if x > threshold {
        (b, a, one - x, true)
    } else {
        (a, b, x, false)
    };

    if specialized {
        let a_integer = tail_a == tail_a.floor();
        let b_integer = tail_b == tail_b.floor();
        let total = (tail_a + tail_b).try_to_f64_exact()?;
        if a_integer && b_integer && total <= 33.0 {
            let a_terms = tail_a.try_to_f64_exact()? as usize;
            let b_terms = tail_b.try_to_f64_exact()? as usize;
            if b_terms > work.remaining {
                return beta_inc_core(a, b, x, false, work);
            }
            let Some(tail) = beta_inc_integer_tail(a_terms, b_terms, tail_x, work)? else {
                return beta_inc_core(a, b, x, false, work);
            };
            let value = if complement {
                checked_probability(one - tail)?
            } else {
                tail
            };
            return Ok(BetaIncEvaluation {
                value,
                branch_id: if complement {
                    algorithm_component!("symmetric_integer_finite")
                } else {
                    algorithm_component!("lower_integer_finite")
                },
                iteration_count: b_terms,
            });
        }
    }

    if let Some(normalization) = linear_normalization(a.into_f64(), b.into_f64()) {
        let before = work.used;
        if let Some(tail) = linear_tail(
            a.into_f64(),
            b.into_f64(),
            x.into_f64(),
            false,
            normalization,
            work,
        )?
        .filter(|tail| tail.linear_error() <= LINEAR_TAIL_ACCEPT)
        {
            return Ok(BetaIncEvaluation {
                value: checked_probability(S::try_from_f64_lossy(tail.value)?)?,
                branch_id: algorithm_component!("conditioned_scalar_tail"),
                iteration_count: work.used - before,
            });
        }
    }

    let wide_x = if complement {
        Wide::from(1.0) - Wide::from(x.into_f64())
    } else {
        Wide::from(x.into_f64())
    };
    let initial =
        1.0 - (tail_a.into_f64() + tail_b.into_f64()) * wide_x.hi / (tail_a.into_f64() + 1.0);
    let series = use_beta_series(tail_b.into_f64(), wide_x.hi);
    // Do not evaluate a scalar fraction when its first denominator or its
    // reflected coordinate already requires the compensated recurrence.
    let scalar_normalization = (series || (initial.abs() >= 0.5 && wide_x.lo == 0.0))
        .then(|| probability_normalization(a.into_f64(), b.into_f64()))
        .filter(|(value, error)| value.is_finite() && error.is_finite());
    let mut tail_normalization = None;
    let mut scalar_tail = None;
    if let Some((normalization, normalization_error)) = scalar_normalization {
        let before = work.used;
        let tail = probability_log_tail_fast(
            a.into_f64(),
            b.into_f64(),
            x.into_f64(),
            false,
            work,
            normalization,
            normalization_error,
        )?;
        let value = tail.logarithm.exp();
        let quantum = (value - value.next_down()).max(f64::from_bits(1));
        let bound = value * tail.error + 4.0 * quantum;
        if value.is_finite()
            && tail.error <= f64::EPSILON.sqrt()
            && bound.is_finite()
            && bound <= 8.0 * quantum
        {
            return Ok(BetaIncEvaluation {
                value: S::try_from_f64_lossy(value)?,
                branch_id: algorithm_component!("conditioned_scalar_tail"),
                iteration_count: work.used - before,
            });
        }
        // If the exponent or fraction needs compensation, a well-conditioned
        // normalization can still be reused. Reserve two relative units for
        // it, and avoid amplification by subtraction of a tail near one.
        if normalization_error <= 2.0 * f64::EPSILON
            && (!complement || tail.logarithm >= -core::f64::consts::LN_2)
        {
            tail_normalization = Some(Wide::from(normalization));
        }
        scalar_tail = Some(tail);
    }
    let normalization = match tail_normalization {
        Some(value) => value,
        None => ln_beta_kernel(a.into_f64(), b.into_f64(), true)?.0,
    };

    let cached = scalar_tail.filter(|tail| tail.reflected == complement && tail.series == series);
    let (log_tail, iteration_count) = if series {
        let (correction, count) = if let Some(tail) = cached {
            (Wide::from(tail.factor), tail.iterations)
        } else {
            let (value, count, _) =
                beta_series(tail_a.into_f64(), tail_b.into_f64(), wide_x.hi, work)?;
            (value, count)
        };
        (
            ln_wide(wide_x).mul_f64(tail_a.into_f64())
                - normalization
                - Wide::from(tail_a.into_f64()).ln()
                + correction,
            count,
        )
    } else if initial.abs() < 0.5 || wide_x.lo != 0.0 {
        // Cancellation in the first denominator magnifies subsequent scalar
        // fraction updates. Preserve the lost bits and reflected coordinate.
        let (tail, _, count) = beta_log_tail(
            tail_a.into_f64(),
            tail_b.into_f64(),
            wide_x,
            work,
            Some(normalization),
        )?;
        (tail, count)
    } else {
        let (fraction, iteration_count) = if let Some(tail) = cached {
            (tail.factor, tail.iterations)
        } else {
            beta_fraction(tail_a.into_f64(), tail_b.into_f64(), wide_x.hi, work)?
        };
        let log_tail = Wide::from(tail_a.into_f64()) * ln_wide(wide_x)
            + Wide::from(tail_b.into_f64()) * log_complement(wide_x)
            - normalization
            - ln_wide(Wide::from(tail_a.into_f64()))
            + ln_wide(Wide::from(fraction));
        if complement && log_tail.hi > -core::f64::consts::LN_2 {
            // The scalar fraction's relative error is amplified when the
            // requested result is the small complement of this direct tail.
            let (log_tail, _, extra) = beta_log_tail(
                tail_a.into_f64(),
                tail_b.into_f64(),
                wide_x,
                work,
                Some(ln_beta_wide(a.into_f64(), b.into_f64())?),
            )?;
            (log_tail, iteration_count + extra)
        } else {
            (log_tail, iteration_count)
        }
    };
    let value = if complement {
        // Reflection can select a tail close to one for very unequal shapes.
        // Preserve its small complement instead of subtracting a rounded exp.
        -log_tail.exp_m1().value()
    } else {
        log_tail.exp()
    };
    let value = checked_probability(S::try_from_f64_lossy(value)?)?;
    Ok(BetaIncEvaluation {
        value,
        branch_id: if series && complement {
            algorithm_component!("symmetric_power_series")
        } else if series {
            algorithm_component!("lower_power_series")
        } else if complement {
            algorithm_component!("symmetric_scaled_lentz")
        } else {
            algorithm_component!("lower_scaled_lentz")
        },
        iteration_count,
    })
}

fn beta_inc_integer_tail<S: RealScalar>(
    a: usize,
    b: usize,
    x: S,
    work: &mut Work,
) -> Result<Option<S>, SciOxideError> {
    work.try_step()?;
    let n = a + b - 1;
    let one_minus_x = S::ONE - x;
    let coefficient = (1..=a).try_fold(S::ONE, |value, index| {
        #[allow(clippy::cast_precision_loss)]
        let numerator = S::try_from_f64_lossy((n - a + index) as f64)?;
        #[allow(clippy::cast_precision_loss)]
        let denominator = S::try_from_f64_lossy(index as f64)?;
        Ok::<_, SciOxideError>(value * numerator / denominator)
    })?;
    let x_power = (0..a).fold(S::ONE, |value, _| value * x);
    let one_minus_x_power = (0..b - 1).fold(S::ONE, |value, _| value * one_minus_x);
    let initial = coefficient * x_power * one_minus_x_power;
    // A subnormal intermediate has already lost relative precision, even
    // when the binomial coefficient would restore a representable tail.
    // Let the compensated logarithmic path round only the complete tail.
    if x_power < S::MIN_POSITIVE || one_minus_x_power < S::MIN_POSITIVE || initial < S::MIN_POSITIVE
    {
        return Ok(None);
    }
    let (sum, _, _) = (a..n).try_fold(
        (initial, initial, S::ZERO),
        |(sum, term, correction), index| {
            work.try_step()?;
            #[allow(clippy::cast_precision_loss)]
            let numerator = S::try_from_f64_lossy((n - index) as f64)?;
            #[allow(clippy::cast_precision_loss)]
            let denominator = S::try_from_f64_lossy((index + 1) as f64)?;
            let next = term * numerator * x / (denominator * one_minus_x);
            let corrected = next - correction;
            let updated = sum + corrected;
            Ok::<_, SciOxideError>((updated, next, (updated - sum) - corrected))
        },
    )?;
    checked_probability(sum).map(Some)
}

fn use_beta_series(b: f64, x: f64) -> bool {
    x <= 0.5 && b * x <= 0.5
}

/// Two-word subtraction keeps the complement for the fixed logarithm.
/// Below sqrt(epsilon), the direct log1p series needs only a few terms and
/// also preserves displacements below the normal lattice.
fn log_complement(x: Wide) -> Wide {
    if x.hi >= f64::EPSILON.sqrt() {
        ln_wide(Wide::from(1.0) - x)
    } else {
        (-x).ln_1p()
    }
}

/// Log of 2F1(a, 1-b; a+1; x), using <https://dlmf.nist.gov/8.17.E7>.
/// Here x <= 1/2 and b*x <= 1/2 bound successive binomial terms by 1/2.
/// Keep the constant outside the scalar sum: for small a the correction is
/// O(a), and losing it would destroy the sensitivity of an inverse tail.
fn beta_series(
    a: f64,
    b: f64,
    x: f64,
    work: &mut Work,
) -> Result<(Wide, usize, f64), SciOxideError> {
    let (correction, count, error) = beta_series_correction(a, b, x, work)?;
    // The scalar series already has O(epsilon) relative error.
    // Its log1p keeps O(a) sensitivity without a two-word logarithm.
    let logarithm = correction.ln_1p();
    Ok((
        Wide::from(logarithm),
        count,
        error / (1.0 + correction) + 4.0 * (logarithm.abs().next_up() - logarithm.abs()),
    ))
}

/// The correction `c` of 2F1(a, 1-b; a+1; x) = 1 + c and its absolute error.
fn beta_series_correction(
    a: f64,
    b: f64,
    x: f64,
    work: &mut Work,
) -> Result<(f64, usize, f64), SciOxideError> {
    let mut term = 1.0;
    let mut sum = 0.0;
    let mut compensation = 0.0;
    let mut rounding = 0.0;
    let mut count = 0usize;
    loop {
        work.try_step()?;
        count += 1;
        let n = count as f64;
        term *= ((n - b) / n) * x;
        let next = term / (a + n);
        // Four rounded operations per coefficient recurrence (2 epsilon),
        // plus the division by a+n and compensated summation. Weight each
        // term's error before summing, so cancellation cannot hide it.
        rounding += (2.0 * n + 3.0) * next.abs();
        let corrected = next - compensation;
        let updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
        if !sum.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "incomplete-beta power series is not finite",
            ));
        }
        if next == 0.0 || next.abs() <= 0.25 * f64::EPSILON * sum.abs() {
            let correction = a * (sum - compensation);
            // Successive absolute terms decrease by at least a factor two.
            let error = f64::EPSILON * (a * rounding + correction.abs()) + a * next.abs();
            return Ok((correction, count, error));
        }
    }
}

/// An `f64` incomplete-beta tail evaluated in linear space.
#[derive(Clone, Copy)]
pub(crate) struct LinearBetaTail {
    /// The requested tail probability.
    pub(crate) value: f64,
    /// Calibrated relative uncertainty of `value`.
    pub(crate) error: f64,
    /// `x^a (1-x)^b / B(a,b)`, so `power / value` is the logit derivative.
    pub(crate) power: f64,
    /// The summed opposite tail when `value` is its complement.
    complemented: Option<f64>,
}

impl LinearBetaTail {
    /// Natural logarithm of `value` and its absolute uncertainty; a
    /// complement uses `ln1p` of the summed tail instead of rounding `1 - d`.
    pub(crate) fn log(&self) -> (f64, f64) {
        let log = self
            .complemented
            .map_or_else(|| self.value.ln(), |summed| (-summed).ln_1p());
        (log, self.error + f64::EPSILON * log.abs())
    }

    /// Relative uncertainty of the linear `value`, including the rounded
    /// complement subtraction.
    pub(crate) fn linear_error(&self) -> f64 {
        self.error
            + if self.complemented.is_some() {
                0.5 * f64::EPSILON
            } else {
                0.0
            }
    }
}

/// Relative uncertainty below which a linear tail is returned directly. A
/// bound of sixteen epsilon proves the documented 32-ulp incomplete envelope;
/// the reference tests measure a twelve-ulp maximum.
pub(crate) const LINEAR_TAIL_ACCEPT: f64 = 16.0 * f64::EPSILON;

/// `B(a,b)` and its relative uncertainty when an `f64` power-factor tail can
/// use it: the exact half-integer lattice, or the mixed-precision gamma ratio
/// while every recurrence product stays finite.
pub(crate) fn linear_normalization(a: f64, b: f64) -> Option<(f64, f64)> {
    let (value, uncertainty) = if let Some(beta) = beta_small_lattice(a, b) {
        (beta.value(), f64::EPSILON * f64::EPSILON)
    } else if a.min(b) >= 1.0e-150 && a + b < 128.0 {
        let (beta, uncertainty) =
            crate::support::gamma_real::beta_moderate_fast(Wide::from(a), Wide::from(b));
        (beta.value(), uncertainty)
    } else {
        return None;
    };
    (value.is_normal() && uncertainty.is_finite())
        .then_some((value, uncertainty + 0.5 * f64::EPSILON))
}

/// Integer shapes through `a + b = 33`: the finite binomial sums
/// `I_x(a,b) = sum_{j>=a} C(n,j) x^j (1-x)^(n-j)`, `n = a+b-1`, and the upper
/// tail over `j < a`, both positive, so either tail is summed directly.
fn integer_linear_tail(
    a: f64,
    b: f64,
    x: f64,
    upper: bool,
    (beta, beta_error): (f64, f64),
    work: &mut Work,
) -> Result<Option<LinearBetaTail>, SciOxideError> {
    let c = 1.0 - x;
    let c_lo = (1.0 - c) - x;
    let leading = x.powf(a);
    let trailing = c.powf(b - 1.0);
    if !(leading.is_normal() && trailing.is_normal() && c > 0.0) {
        return Ok(None);
    }
    let trailing = trailing * (1.0 + (b - 1.0) * (c_lo / c));
    // C(n, a) = 1 / (a B(a, b)).
    let product = leading * trailing;
    if !product.is_normal() {
        return Ok(None);
    }
    let first = product / (a * beta);
    if !first.is_normal() {
        return Ok(None);
    }
    let ratio = x / c * (1.0 - c_lo / c);
    let (n, a_index) = ((a + b - 1.0) as usize, a as usize);
    // Each step multiplies by an integer ratio and x/(1-x) (1.5 units, plus
    // the unit of the rounded ratio carried by every later term).
    let (sum, weighted) = if upper {
        let mut term = first * a / (n - a_index + 1) as f64 / ratio;
        let (mut sum, mut weighted) = (term, 0.0);
        for (steps, j) in (1..a_index).rev().enumerate() {
            work.try_step()?;
            term *= j as f64 / (n - j + 1) as f64 / ratio;
            sum += term;
            weighted += (steps + 1) as f64 * term;
        }
        (sum, weighted + sum)
    } else {
        let (mut term, mut sum, mut weighted) = (first, first, 0.0);
        for (steps, j) in (a_index..n).enumerate() {
            work.try_step()?;
            term *= (n - j) as f64 / (j + 1) as f64 * ratio;
            sum += term;
            weighted += (steps + 1) as f64 * term;
        }
        (sum, weighted)
    };
    if !(sum.is_finite() && sum > 0.0 && sum <= 1.0) {
        return Ok(None);
    }
    // Two powers, their combination with the normalization, the residual
    // correction, the step ratios and the summation of at most 33 terms.
    let terms = (n + 1) as f64;
    let error = 5.5 * f64::EPSILON
        + beta_error
        + 2.5 * f64::EPSILON * weighted / sum
        + 0.5 * f64::EPSILON * terms;
    Ok(Some(LinearBetaTail {
        value: sum,
        error,
        power: leading * trailing * c / beta,
        complemented: None,
    }))
}

/// Which positive series sums `I_y(p, q)` within a bounded number of terms.
#[derive(Clone, Copy, PartialEq)]
enum LinearSeries {
    /// DLMF 8.17.7 with alternating binomial terms bounded by one half.
    Endpoint,
    /// DLMF 8.17.8, `2F1(p+q, 1; p+1; y)`, with term ratios below the bound.
    Hypergeometric,
}

/// Largest accepted term ratio of DLMF 8.17.8: at most 166 terms.
const HYPERGEOMETRIC_RATIO: f64 = 0.8;

fn linear_series(p: f64, q: f64, y: f64) -> Option<LinearSeries> {
    if y <= 0.5 && q * y <= 0.5 {
        Some(LinearSeries::Endpoint)
    } else if (y * (p + q) / (p + 1.0)).max(y) <= HYPERGEOMETRIC_RATIO {
        Some(LinearSeries::Hypergeometric)
    } else {
        None
    }
}

/// The requested incomplete-beta tail in `f64` with a calibrated relative
/// uncertainty, or `None` where the power factor leaves the normal range or
/// neither orientation has a fast positive series.
///
/// The tail `I_y(p, q)` is summed in its own orientation (`(a, b, x)` for the
/// lower tail, `(b, a, 1-x)` for the upper) whenever DLMF 8.17.7 applies
/// (`y <= 1/2`, `qy <= 1/2`) or DLMF 8.17.8, `y^p (1-y)^q / (p B) *
/// 2F1(p+q, 1; p+1; y)`, has term ratios at most 4/5. Otherwise the opposite
/// orientation is complemented, and its uncertainty includes the cancellation
/// amplification. The power factor uses `pow`, whose sub-ulp error does not
/// grow with `|p ln y|` as a rounded logarithmic exponent would; the inexact
/// complement coordinate keeps its exact residual.
pub(crate) fn linear_tail(
    a: f64,
    b: f64,
    x: f64,
    upper: bool,
    (beta, beta_error): (f64, f64),
    work: &mut Work,
) -> Result<Option<LinearBetaTail>, SciOxideError> {
    if a == a.floor() && b == b.floor() && a + b <= 33.0 {
        return integer_linear_tail(a, b, x, upper, (beta, beta_error), work);
    }
    // y is the summed coordinate and c = 1 - y; one of the two is exact and
    // the other keeps its exact residual, so (y + y_lo)^p (c + c_lo)^q follows
    // from correctly ordered binary64 powers.
    let coordinates = |reflected: bool| {
        if reflected {
            let y = 1.0 - x;
            (b, a, y, (1.0 - y) - x, x, 0.0)
        } else {
            let c = 1.0 - x;
            (a, b, x, 0.0, c, (1.0 - c) - x)
        }
    };
    let own = coordinates(upper);
    let (complemented, (p, q, y, y_lo, c, c_lo), series) =
        if let Some(series) = linear_series(own.0, own.1, own.2) {
            (false, own, series)
        } else {
            let other = coordinates(!upper);
            match linear_series(other.0, other.1, other.2) {
                Some(series) => (true, other, series),
                None => return Ok(None),
            }
        };
    let leading = y.powf(p);
    let trailing = c.powf(q);
    if !(leading.is_normal() && trailing.is_normal()) {
        return Ok(None);
    }
    let leading = leading * (1.0 + p * (y_lo / y));
    let trailing = trailing * (1.0 + q * (c_lo / c));
    let inverse = 1.0 / (p * beta);
    // Two sub-ULP powers, one residual correction, the normalization and the
    // four rounded products or quotients that combine them.
    let mut error = 5.5 * f64::EPSILON + beta_error;
    let summed = match series {
        LinearSeries::Endpoint => {
            let (correction, _, absolute) = beta_series_correction(p, q, y, work)?;
            error += absolute / (1.0 + correction) + 0.5 * f64::EPSILON;
            leading * inverse * (1.0 + correction)
        }
        LinearSeries::Hypergeometric => {
            // Term ratios y (1 + (q-1)/(p+1+n)) are monotone toward y. Each
            // rounds within (1.5 + w) epsilon, where w bounds the relative
            // magnification of the rounded (q-1)/(p+1+n) in its sum with one;
            // a rounding at index k perturbs every later term.
            let shift = q - 1.0;
            let start = p + 1.0;
            let first_ratio = shift / start;
            let magnification = if shift < 0.0 {
                -first_ratio / (1.0 + first_ratio)
            } else {
                first_ratio.min(1.0)
            };
            let mut term = 1.0;
            let mut sum = 1.0;
            let mut compensation = 0.0;
            let mut weighted = 0.0;
            let mut count = 0usize;
            loop {
                work.try_step()?;
                let ratio = y * (1.0 + shift / (start + count as f64));
                count += 1;
                term *= ratio;
                let updated = sum + term;
                compensation += (sum - updated) + term;
                sum = updated;
                weighted += count as f64 * term;
                let bound = if shift >= 0.0 { ratio } else { y };
                if term <= 0.125 * f64::EPSILON * sum * (1.0 - bound) {
                    error += f64::EPSILON * (1.5 + magnification) * weighted / sum
                        + 1.5 * f64::EPSILON
                        + term * bound / ((1.0 - bound) * sum);
                    break;
                }
            }
            leading * trailing * inverse * (sum + compensation)
        }
    };
    if !(summed.is_finite() && summed > 0.0 && summed <= 1.0) {
        return Ok(None);
    }
    let power = leading * trailing / beta;
    Ok(Some(if !complemented {
        LinearBetaTail {
            value: summed,
            error,
            power,
            complemented: None,
        }
    } else {
        let value = 1.0 - summed;
        if value <= 0.0 {
            return Ok(None);
        }
        // d ln(1-d) = -dd / (1-d): the complement amplifies by d / (1-d);
        // the rounded subtraction adds half a unit to the linear value only.
        LinearBetaTail {
            value,
            error: error * (summed / value),
            power,
            complemented: Some(summed),
        }
    }))
}

#[allow(clippy::similar_names)]
fn beta_log_tail(
    a: f64,
    b: f64,
    x: Wide,
    work: &mut Work,
    normalization: Option<Wide>,
) -> Result<(Wide, Wide, usize), SciOxideError> {
    let normalization = match normalization {
        Some(value) => value,
        None => ln_beta_kernel(a, b, true)?.0,
    };
    if use_beta_series(b, x.hi) {
        let (correction, count, _) = beta_series(a, b, x.hi, work)?;
        let exponent = ln_wide(x).mul_f64(a) - normalization;
        return Ok((
            exponent - Wide::from(a).ln() + correction,
            exponent + log_complement(x).mul_f64(b),
            count,
        ));
    }
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let a_wide = Wide::from(a);
    let b_wide = Wide::from(b);
    let one = Wide::from(1.0);
    let floor_wide = |value: Wide| {
        if value.hi.abs() < tiny {
            Wide::from(tiny.copysign(value.hi))
        } else {
            value
        }
    };
    let mut c = one;
    let mut d = one / floor_wide(one - (a_wide + b_wide) * x / (a_wide + one));
    let mut fraction = d;
    let mut iteration = 0usize;
    loop {
        iteration += 1;
        work.try_step()?;
        let m = Wide::from(iteration as f64);
        let twice_m = m.mul_f64(2.0);
        let even = m * (b_wide - m) * x / ((a_wide - one + twice_m) * (a_wide + twice_m));
        d = one / floor_wide(even * d + one);
        c = floor_wide(one + even / c);
        fraction = fraction * d * c;
        let odd = -(a_wide + m) * (a_wide + b_wide + m) * x
            / ((a_wide + twice_m) * (a_wide + one + twice_m));
        d = one / floor_wide(odd * d + one);
        c = floor_wide(one + odd / c);
        let delta = d * c;
        fraction = fraction * delta;
        if !fraction.hi.is_finite() || fraction.hi <= 0.0 {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "continued-fraction scale is not finite and positive",
            ));
        }
        if (delta - one).value().abs() <= 32.0 * f64::EPSILON * f64::EPSILON {
            let power = a_wide * ln_wide(x) + b_wide * log_complement(x) - normalization;
            return Ok((
                power - ln_wide(a_wide) + ln_wide(fraction),
                power,
                iteration,
            ));
        }
    }
}

/// Binary64 modified Lentz fraction for ordinary forward and inverse steps.
fn beta_fraction(a: f64, b: f64, x: f64, work: &mut Work) -> Result<(f64, usize), SciOxideError> {
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let floor = |v: f64| crate::util::lentz_floor(v, tiny);
    let mut c = 1.0;
    let mut d = 1.0 / floor(-scioxide_core::fast_mul_add((a + b) / (a + 1.0), x, -1.0));
    let mut fraction = d;
    let mut iteration = 0usize;
    loop {
        iteration += 1;
        work.try_step()?;
        let m = iteration as f64;
        let center = a + 2.0 * m;
        let even = (m / center) * ((b - m) / (center - 1.0)) * x;
        d = 1.0 / floor(scioxide_core::fast_mul_add(even, d, 1.0));
        c = floor(1.0 + even / c);
        fraction *= d * c;
        let shifted = a + m;
        let odd = -(shifted / center) * ((shifted + b) / (center + 1.0)) * x;
        d = 1.0 / floor(scioxide_core::fast_mul_add(odd, d, 1.0));
        c = floor(1.0 + odd / c);
        let delta = d * c;
        fraction *= delta;
        if !fraction.is_finite() || fraction <= 0.0 {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "incomplete-beta continued fraction is not finite and positive",
            ));
        }
        if (delta - 1.0).abs() <= f64::EPSILON {
            return Ok((fraction, iteration));
        }
    }
}

/// A scalar tail keeps its series/fraction so compensating its exponent does
/// not repeat the recurrence or charge its work a second time.
#[derive(Clone, Copy)]
pub(crate) struct ScalarBetaTail {
    pub(crate) logarithm: f64,
    pub(crate) error: f64,
    pub(crate) power: f64,
    factor: f64,
    iterations: usize,
    series: bool,
    reflected: bool,
}

/// Log tail, rounding estimate, and logarithmic power factor for inverse steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probability_log_tail_fast(
    a: f64,
    b: f64,
    x: f64,
    upper: bool,
    work: &mut crate::support::Work,
    normalization: f64,
    normalization_error: f64,
) -> Result<ScalarBetaTail, SciOxideError> {
    let scale = a.max(b).max(1.0);
    let reflected = x > (a / scale + 1.0 / scale) / (a / scale + b / scale + 2.0 / scale);
    let original_x = x;
    let (a, b, x) = if reflected {
        (b, a, 1.0 - x)
    } else {
        (a, b, x)
    };
    let left = a * x.ln();
    let right = b * (-x).ln_1p();
    let power = left + right - normalization;
    let series = use_beta_series(b, x);
    let (direct, arithmetic_error, factor, iterations) = if series {
        let (correction, count, series_error) = beta_series(a, b, x, work)?;
        let log_a = a.ln();
        let direct = (left - normalization - log_a) + correction.hi;
        let error =
            series_error + f64::EPSILON * (4.0 * (left.abs() + log_a.abs()) + 2.0 * direct.abs());
        (direct, error, correction.hi, count)
    } else {
        let (fraction, count) = beta_fraction(a, b, x, work)?;
        let factor = (fraction / a).ln();
        let direct = power + factor;
        let initial = (1.0 - (a + b) * x / (a + 1.0)).abs();
        let error = f64::EPSILON
            * (4.0
                + 8.0 * count as f64
                + 4.0 * (left.abs() + right.abs() + factor.abs())
                + power.abs()
                + direct.abs())
            / initial.min(1.0);
        (direct, error, fraction, count)
    };
    let reflection_error = if reflected {
        ((1.0 - x) - original_x).abs() * (a / x + b / (1.0 - x))
    } else {
        0.0
    };
    let error = arithmetic_error + normalization_error + reflection_error;
    let direct = direct.min(0.0);
    let (logarithm, error) = if upper == reflected {
        (direct, error)
    } else {
        let opposite = crate::error_probability::probability::log1mexp(direct)?;
        (
            opposite,
            error * (direct - opposite).exp() + 4.0 * f64::EPSILON * opposite.abs(),
        )
    };
    Ok(ScalarBetaTail {
        logarithm,
        error,
        power,
        factor,
        iterations,
        series,
        reflected,
    })
}

/// Direct logarithmic tail and power factor, keeping reflected coordinates.
pub(crate) fn probability_log_tail_wide(
    a: f64,
    b: f64,
    x: f64,
    upper: bool,
    work: &mut crate::support::Work,
    normalization: Wide,
) -> Result<(Wide, Wide), SciOxideError> {
    // In Student-t coordinates the usual smaller-tail reflection chooses a
    // slowly converging fraction near one. DLMF 8.17.7 instead sums the central
    // half-shape series. Its modest cancellation is harmless in two words.
    let half = if a == 0.5 && (8.0..=2048.0).contains(&b) {
        Some((false, b, Wide::from(x)))
    } else if b == 0.5 && (8.0..=2048.0).contains(&a) {
        Some((true, a, Wide::from(1.0) - Wide::from(x)))
    } else {
        None
    };
    if let Some((reflected, shape, coordinate)) = half
        && coordinate.hi <= 0.25
        && (0.5..=4.0).contains(&(shape * coordinate.hi))
    {
        let mut term = Wide::from(1.0);
        let mut sum = term;
        let mut count = 0usize;
        loop {
            work.try_step()?;
            count += 1;
            let n = count as f64;
            term = (term * (Wide::from(n) - Wide::from(shape))).div_f64(n) * coordinate;
            let next = term.mul_f64(0.5).div_f64(n + 0.5);
            sum = sum + next;
            // After eight terms the ratio of successive absolute terms is
            // <= 1/2: shape*x <= 4 before the sign change, and x <= 1/4
            // afterwards. Twice the last term therefore bounds the tail.
            if count >= 8 && next.hi.abs() <= f64::EPSILON * f64::EPSILON / 64.0 * sum.hi.abs() {
                let exponent = ln_wide(coordinate).mul_f64(0.5) - normalization;
                let central = exponent + sum.ln() - Wide::from(0.5).ln();
                let power = exponent + log_complement(coordinate).mul_f64(shape);
                return Ok((
                    if upper == reflected {
                        central
                    } else {
                        crate::error_probability::probability::wide_log1mexp(central)
                    },
                    power,
                ));
            }
        }
    }
    let scale = a.max(b).max(1.0);
    let reflected = x > (a / scale + 1.0 / scale) / (a / scale + b / scale + 2.0 / scale);
    let (logarithm, power, _count) = if reflected {
        beta_log_tail(
            b,
            a,
            Wide::from(1.0) - Wide::from(x),
            work,
            Some(normalization),
        )?
    } else {
        beta_log_tail(a, b, Wide::from(x), work, Some(normalization))?
    };
    Ok((
        if upper == reflected {
            logarithm
        } else {
            crate::error_probability::probability::wide_log1mexp(logarithm)
        },
        power,
    ))
}

/// The requested (`upper`) tail, logarithmic when `logarithmic` is set.
///
/// A conditioned `f64` linear tail is returned directly; otherwise the
/// smaller tail is evaluated in two-word arithmetic and the requested one is
/// its compensated complement.
pub(crate) fn probability_tail<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    upper: bool,
    logarithmic: bool,
    work: &mut crate::support::Work,
) -> Result<(S, AlgorithmComponent), SciOxideError> {
    if let Some(lower) = forward_endpoint(a, b, x)? {
        let value = if upper { S::ONE - lower } else { lower };
        let value = if !logarithmic {
            value
        } else if value == S::ZERO {
            S::NEG_INFINITY
        } else {
            S::ZERO
        };
        return Ok((value, algorithm_component!("forward_endpoint")));
    }
    if work.remaining == 0 {
        work.complete = false;
        return Err(work.limit_error());
    }
    let (af, bf, xf) = (a.into_f64(), b.into_f64(), x.into_f64());
    if let Some(normalization) = linear_normalization(af, bf)
        && let Some(tail) = linear_tail(af, bf, xf, upper, normalization, work)?
    {
        let (value, accepted) = if logarithmic {
            let (log, error) = tail.log();
            (log, error <= LINEAR_TAIL_ACCEPT * log.abs())
        } else {
            (tail.value, tail.linear_error() <= LINEAR_TAIL_ACCEPT)
        };
        if accepted {
            return Ok((
                S::try_from_f64_lossy(value)?,
                if upper {
                    algorithm_component!("upper_linear_tail")
                } else {
                    algorithm_component!("lower_linear_tail")
                },
            ));
        }
    }
    let scale = a.max(b).max(S::ONE);
    let threshold = (a / scale + S::ONE / scale) / (a / scale + b / scale + (S::TWO) / scale);
    let reflected = x > threshold;
    let (direct, _) =
        probability_log_tail_wide(af, bf, xf, reflected, work, ln_beta_wide(af, bf)?)?;
    let log = if upper == reflected {
        direct
    } else {
        crate::error_probability::probability::wide_log1mexp(direct)
    };
    // Exponentiate both words: a rounded logarithm of a deep tail would cost
    // half an ULP of |ln T| in relative accuracy.
    let value = if !logarithmic {
        S::try_from_f64_lossy(log.exp())?
    } else if log.hi == f64::NEG_INFINITY {
        S::NEG_INFINITY
    } else {
        S::try_from_f64_lossy(log.value())?
    };
    Ok((
        value,
        if reflected {
            algorithm_component!("upper_log_tail")
        } else {
            algorithm_component!("lower_log_tail")
        },
    ))
}

pub(crate) fn beta_inc_evaluate<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    options: BetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let result = beta_inc_kernel(a, b, x, &mut work);
    work.check()?;
    result.map(|evaluation| {
        evaluation.with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
    })
}

/// Validate the forward domain and resolve its finite endpoint limits.
fn forward_endpoint<S: RealScalar>(a: S, b: S, x: S) -> Result<Option<S>, SciOxideError> {
    require_finite_param(a, "beta shape a must be finite")?;
    require_finite_param(b, "beta shape b must be finite")?;
    require_finite_param(x, "beta coordinate must be finite")?;
    for (value, argument) in [
        (a, scioxide_core::argument!("a")),
        (b, scioxide_core::argument!("b")),
    ] {
        if value < S::ZERO {
            return Err(crate::support::GAMMA_ERRORS.outside(argument));
        }
    }
    if x < S::ZERO || x > S::ONE {
        return Err(crate::support::GAMMA_ERRORS.outside(scioxide_core::argument!("x")));
    }
    if (a == S::ZERO && (b == S::ZERO || x == S::ZERO)) || (b == S::ZERO && x == S::ONE) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::Undefined,
            "regularized beta has no joint limit at this zero-shape corner",
        ));
    }
    Ok(if a == S::ZERO || x == S::ONE {
        Some(S::ONE)
    } else if b == S::ZERO || x == S::ZERO {
        Some(S::ZERO)
    } else {
        None
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {

    use super::*;
    use crate::beta_inc_inv;

    // ---- ln_beta / beta ----

    #[test]
    fn test_beta_known_value_f64() {
        // B(2, 3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12
        let val = beta(2.0_f64, 3.0).unwrap();
        assert!((val - 1.0 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_beta_symmetry_f64() {
        let ab = beta(3.0_f64, 5.0).unwrap();
        let ba = beta(5.0_f64, 3.0).unwrap();
        assert!((ab - ba).abs() < 1e-12);
    }

    #[test]
    fn test_ln_beta_f64() {
        let val = ln_beta(2.0_f64, 3.0).unwrap();
        assert!((val - (1.0_f64 / 12.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_beta_domain_error() {
        {
            let outcome = &(beta(-1.0_f64, 3.0));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
        {
            let outcome = &(beta(1.0_f64, 0.0));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_beta_f32() {
        let val = beta(2.0_f32, 3.0).unwrap();
        assert!((val - 1.0_f32 / 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_beta_large_unbalanced_shapes_are_cancellation_safe() {
        let a = 1.0e16_f64;
        assert_eq!(beta(a, 1.0).unwrap(), 1.0e-16);
        let expected = -a.ln();
        let actual = ln_beta(a, 1.0).unwrap();
        assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
    }

    // ---- beta_inc ----

    #[test]
    fn reflected_small_tail_at_series_boundary() {
        let a = 32.5;
        let b = 0.01;
        let x = f64::from_bits(0x3fef81f81f81f81f);
        let actual = beta_inc(a, b, x).unwrap();
        let expected = 0.005_672_547_916_204_945_f64;
        assert!((actual - expected).abs() <= 32.0 * (expected.next_up() - expected));
    }

    #[test]
    fn test_beta_inc_boundary_f64() {
        let val0 = beta_inc(2.0_f64, 3.0, 0.0).unwrap();
        let val1 = beta_inc(2.0_f64, 3.0, 1.0).unwrap();
        assert!(val0.abs() < 1e-15);
        assert!((val1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_beta_inc_symmetry_f64() {
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        let a = 2.5_f64;
        let b = 3.5;
        let x = 0.4;
        let lhs = beta_inc(a, b, x).unwrap();
        let rhs = 1.0 - beta_inc(b, a, 1.0 - x).unwrap();
        assert!((lhs - rhs).abs() < 1e-10);
    }

    #[test]
    fn test_beta_inc_known_value_f64() {
        // I_x(1, 1) = x (uniform)
        let x = 0.7_f64;
        let val = beta_inc(1.0, 1.0, x).unwrap();
        assert!((val - x).abs() < 1e-12);
    }

    #[test]
    fn test_beta_inc_extreme_symmetric_value() {
        assert_eq!(beta_inc(1.0e300_f64, 1.0e300, 0.5).unwrap(), 0.5);
        assert_eq!(beta_inc(f32::MAX, f32::MAX, 0.5).unwrap(), 0.5);
    }

    #[test]
    fn test_beta_inc_preserves_subnormal_argument() {
        let x = f64::from_bits(1);
        let value = beta_inc(0.5_f64, 1.0, x).unwrap();
        assert!(value.is_finite() && value > 0.0);
    }

    #[test]
    fn test_beta_inc_domain_error() {
        {
            let outcome = &(beta_inc(1.0_f64, 1.0, -0.1));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
        {
            let outcome = &(beta_inc(1.0_f64, 1.0, 1.1));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_beta_inc_f32() {
        let val = beta_inc(2.0_f32, 3.0, 0.5).unwrap();
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_incomplete_beta_iteration_budget_failure() {
        {
            let outcome = &(beta_log_tail(2.3, 4.7, Wide::from(0.2), &mut Work::new(0), None));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
        }
    }

    // ---- beta_inc_inv ----

    #[test]
    fn test_beta_inc_inv_roundtrip_f64() {
        let a = 2.0_f64;
        let b = 5.0;
        let x = 0.3;
        let p = beta_inc(a, b, x).unwrap();
        let x_inv = beta_inc_inv(a, b, p).unwrap();
        assert!((x_inv - x).abs() < 1e-10);
    }

    #[test]
    fn test_beta_inc_inv_boundary() {
        let val0 = beta_inc_inv(2.0_f64, 3.0, 0.0).unwrap();
        let val1 = beta_inc_inv(2.0_f64, 3.0, 1.0).unwrap();
        assert!(val0.abs() < 1e-15);
        assert!((val1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_beta_inc_inv_domain_error() {
        {
            let outcome = &(beta_inc_inv(1.0_f64, 1.0, -0.1));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_beta_inc_inv_non_finite_input() {
        {
            let outcome = &(beta_inc_inv(f64::NAN, 1.0, 0.5));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        {
            let outcome = &(beta_inc_inv(1.0_f64, 1.0, f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }
}
