//! Stable Gaussian probability kernels, erfcx and log erfc, and shared bounded
//! evaluation support.
//!
//! Normal tails keep the original input's compensated Gaussian exponent and
//! take their slowly varying factor from the crate's single erfcx kernel
//! ([`crate::error_probability::real_faddeeva`]). `f64` targets are eight ulps on
//! [-38,38] for forward and logarithmic tails, and four for inverses, including
//! subnormal probabilities; erfcx and log erfc target four. `f32` evaluates
//! in `f64` and rounds once.
//!
//! Plain and detailed normal entries share one value evaluation. Completed
//! evaluations defer their calibrated envelopes to the detailed result. Forward tails
//! correct one exponential with the half-square's low word; quantiles reuse
//! the fixed-cost inverse-error rational approximation without refinement.

use crate::error_probability::real_faddeeva;
use crate::expansions::real;
use crate::result::{Bound, Evaluation, Projection, Ulps};
use crate::support::PROBABILITY_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::wide::Wide;
use crate::support::{check_finite, require_finite_param};
use crate::{ProbabilityOptions, SpecialFunctionResult};
use core::f64::consts::{FRAC_2_SQRT_PI, LN_2};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Budget, RealScalar, SciOxideError, algorithm_component,
};

/// Stable algorithm family of the probability kernels.
const PROBABILITY: AlgorithmComponent = algorithm_component!("probability");

pub(crate) use scioxide_core::log1mexp;
/// Complement a wide log tail without rounding a probability close to one.
pub(crate) fn wide_log1mexp(x: Wide) -> Wide {
    if x.value() >= 0.0 {
        // Preserve the log-zero limit and IEEE invalid classification without
        // passing zero/negative arguments to the positive wide logarithm.
        return Wide::from((-x.value().exp_m1()).ln());
    }
    if x.hi < -core::f64::consts::LN_2 {
        let tail = Wide::from(-x.exp());
        if tail.hi.abs() >= f64::EPSILON.sqrt() {
            // Two-word subtraction preserves the complement. Its fixed
            // logarithm is cheaper than the log1p series for ordinary tails;
            // below sqrt(epsilon) the latter needs only a few terms.
            (Wide::from(1.0) + tail).ln()
        } else {
            tail.ln_1p()
        }
    } else {
        (-x.exp_m1()).ln()
    }
}
/// A probability evaluation of `v` charged to `work` under `method`.
pub(crate) fn finish<S: RealScalar>(
    v: S,
    bound: Bound<S>,
    work: &Work,
    method: AlgorithmComponent,
) -> Result<Evaluation<S>, SciOxideError> {
    let algorithm = AlgorithmId::from_components(PROBABILITY, method, 1);
    if v.is_infinite() && work.complete {
        return Ok(Evaluation::limit(v, algorithm));
    }
    crate::expansions::bounded(v, bound, algorithm, work.used, work.complete)
}
/// Reported envelope of the probability kernels, checked against reference values: 64 units of `v`.
pub(crate) fn error<S: RealScalar>(v: S) -> Result<Bound<S>, SciOxideError> {
    Bound::envelope(v, 64.0)
}

/// Defer a completed normal evaluation's envelope to the detailed result.
// Keep the successful status visible to value() so a scalar caller eliminates
// the Evaluation payload rather than copying its unpublished representations.
#[inline(always)]
fn normal_finish<S: RealScalar>(
    value: S,
    work: &Work,
    method: AlgorithmComponent,
) -> Result<Evaluation<S>, SciOxideError> {
    if !work.complete {
        return finish(value, error(value)?, work, method);
    }
    let algorithm = AlgorithmId::from_components(PROBABILITY, method, 1);
    if value.is_infinite() {
        return Ok(Evaluation::limit(value, algorithm));
    }
    let value = check_finite(value)?;
    // Below MIN_POSITIVE the 64-unit envelope is exactly 64 subnormal
    // quanta. Retain bounded()'s four extra quanta without forming a bound.
    let units = if value.abs() < S::MIN_POSITIVE {
        68.0
    } else {
        64.0
    };
    Ok(Evaluation::calibrated(
        value,
        units,
        algorithm,
        work.used,
        crate::SpecialFunctionTermination::Converged,
    ))
}

/// Reported envelopes of the `f64` erfcx and log erfc values, checked against reference values.
const ERFCX_ULPS: Ulps = Ulps::new(2.0, 1.0);
const LOG_ERFC_ULPS: Ulps = Ulps::new(4.0, 1.0);
pub(crate) fn erfcx<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    require_finite_param(x, "probability parameter x must be finite")?;
    let value = real_faddeeva::erfcx(x.into_f64());
    if value.is_infinite() {
        return Err(crate::result::projection_error(
            crate::ScalarRepresentation::Overflow,
        ));
    }
    S::try_from_f64_lossy(value)
        .map_err(|_| crate::result::projection_error(crate::ScalarRepresentation::Overflow))
}

#[inline]
pub(crate) fn erfcx_evaluation<S: RealScalar>(
    x: S,
    options: crate::ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    erfcx_evaluate(x, true, options.iteration_budget())
}

/// erfcx, with its derivative when `slope` is set; one fixed unit of work.
pub(crate) fn erfcx_evaluate<S: RealScalar>(
    x: S,
    slope: bool,
    budget: Budget,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || erfcx_evaluate::<f64>(x.into_f64(), slope, budget),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "probability parameter x must be finite")?;
    let algorithm = AlgorithmId::from_components(PROBABILITY, algorithm_component!("erfcx"), 1);
    crate::support::admit_family_work(1, budget)?;
    let x = x.into_f64();
    let value = real_faddeeva::erfcx(x);
    if value == f64::INFINITY {
        return erfcx_overflow(x, algorithm);
    }
    let value = S::try_from_f64_lossy(value)?;
    let bound = Bound::envelope(value, ERFCX_ULPS)?;
    // Near the overflow threshold the slope itself may overflow.
    let derivative = if slope {
        let (d, error) = real_faddeeva::erfcx_slope(x, value.into_f64(), bound.absolute.into_f64());
        d.is_finite()
            .then(|| {
                Ok::<_, SciOxideError>((
                    S::try_from_f64_lossy(d)?,
                    Bound::derived(S::try_from_f64_lossy(error.min(f64::MAX))?),
                ))
            })
            .transpose()?
    } else {
        None
    };
    Ok(Evaluation::finite(
        value,
        bound,
        derivative,
        algorithm,
        1,
        crate::SpecialFunctionTermination::Converged,
    ))
}

/// The logarithmic form of `erfcx(x) ≈ 2 exp(x²)` beyond the scalar range, with `x² = s + e` formed
/// exactly.
fn erfcx_overflow<S: RealScalar>(
    x: f64,
    algorithm: AlgorithmId,
) -> Result<Evaluation<S>, SciOxideError> {
    let s = check_finite(x * x)?;
    let log = s + (LN_2 + crate::support::wide::two_product(x, x).lo);
    Ok(Evaluation::logarithmic(
        S::ONE,
        S::try_from_f64_lossy(log)?,
        Bound::derived(S::try_from_f64_lossy(2.0 * f64::EPSILON * log)?),
        Projection::Exp,
        algorithm,
        1,
    ))
}
pub(crate) fn log_erfc<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    require_finite_param(x, "probability parameter x must be finite")?;
    let wide = x.into_f64();
    let value = if wide < 0.5 {
        (-crate::error_probability::error_fn::erf(wide)?).ln_1p()
    } else {
        real_faddeeva::erfcx(wide).ln() - wide * wide
    };
    S::try_from_f64_lossy(check_finite(value)?)
        .map_err(|_| crate::result::projection_error(crate::ScalarRepresentation::Overflow))
}

#[inline]
pub(crate) fn log_erfc_evaluation<S: RealScalar>(
    x: S,
    options: crate::ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    log_erfc_evaluate(x, true, options.iteration_budget())
}

/// log erfc, with its derivative when `slope` is set; one fixed unit of work.
pub(crate) fn log_erfc_evaluate<S: RealScalar>(
    x: S,
    slope: bool,
    budget: Budget,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || log_erfc_evaluate::<f64>(x.into_f64(), slope, budget),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "probability parameter x must be finite")?;
    let algorithm = AlgorithmId::from_components(PROBABILITY, algorithm_component!("log_erfc"), 1);
    crate::support::admit_family_work(1, budget)?;
    let x = x.into_f64();
    let (value, scaled) = if x < 0.5 {
        ((-crate::error_probability::error_fn::erf(x)?).ln_1p(), None)
    } else {
        let scaled = real_faddeeva::erfcx(x);
        (scaled.ln() - x * x, Some(scaled))
    };
    let value = S::try_from_f64_lossy(check_finite(value)?)?;
    let derivative = if slope {
        // d/dx log erfc(x) = -(2/√π) exp(-x²)/erfc(x); the rounded square
        // perturbs the negative branch by up to x² units.
        let (d, relative) = if x >= 0.0 {
            let scaled = scaled.unwrap_or_else(|| real_faddeeva::erfcx(x));
            (-FRAC_2_SQRT_PI / scaled, 4.0)
        } else {
            let tail = crate::error_probability::error_fn::erfc(x)?;
            (-FRAC_2_SQRT_PI * (-x * x).exp() / tail, x * x + 4.0)
        };
        let d = S::try_from_f64_lossy(d)?;
        let error = S::try_from_f64_lossy(relative * f64::EPSILON * d.into_f64().abs())?;
        Some((
            d,
            Bound::derived(error).with_subnormal_rounding(d.abs(), S::TWO),
        ))
    } else {
        None
    };
    Ok(Evaluation::finite(
        value,
        Bound::envelope(value, LOG_ERFC_ULPS)?,
        derivative,
        algorithm,
        1,
        crate::SpecialFunctionTermination::Converged,
    ))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the standard normal cumulative distribution function Φ(x).

```text
Φ(x) = (1/√(2π)) ∫₋∞ˣ exp(−t²/2) dt = ½ erfc(−x/√2)
```

for real `x`. Each tail is computed directly as `½ erfcx(|x|/√2) exp(−x²/2)`, with `x²/2` formed
exactly as a two-term sum, so it keeps full relative accuracy far into the tail instead of
being formed as `1 − Φ`. For `x < −38.5` the result underflows to zero in `f64`; use
[`normal_log_cdf`](crate::normal_log_cdf) for such tails.

# Examples

```
// Φ(1.96) = 0.9750021 (two-sided 95% point)
let p = scioxide_special::normal_cdf(1.96_f64)?;
assert!((p - 0.975_002_1).abs() < 1e-7);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
On `−38 ≤ x ≤ 38` the error is within 8 ulps for `f64`; for `f32` it is within 1 ulp. The
detailed form reports a bound of 64 units ([error estimates](crate#error-estimates)). See
the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    #[inline]
    pub normal_cdf [S: RealScalar] (x: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_cdf_with_options -> SpecialFunctionResult<S>;
    value {
        normal_tail_value(-x)
    }
    report {
        normal_cdf_evaluate(x, options)?.publish()
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_cdf_evaluate<S: RealScalar>(
    x: S,
    options: ProbabilityOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || normal_cdf_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "special-function parameter x must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let value = normal_cdf_kernel(x, &mut work)?;
    normal_finish(value, &work, algorithm_component!(stringify!(normal_cdf)))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the standard normal survival function 1 − Φ(x), the upper tail.

```text
Φ(x) = (1/√(2π)) ∫₋∞ˣ exp(−t²/2) dt = ½ erfc(−x/√2)
```

for real `x`. Each tail is computed directly as `½ erfcx(|x|/√2) exp(−x²/2)`, with `x²/2` formed
exactly as a two-term sum, so it keeps full relative accuracy far into the tail instead of
being formed as `1 − Φ`. For `x > 38.5` the result underflows to zero in `f64`; use
[`normal_log_sf`](crate::normal_log_sf) for such tails.

# Examples

```
// 1 − Φ(x) = ½ erfc(x/√2), here deep in the tail
let x = 10.0_f64;
let expected = 0.5 * scioxide_special::erfc(x / 2.0_f64.sqrt())?;
assert!((scioxide_special::normal_sf(x)? / expected - 1.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
On `−38 ≤ x ≤ 38` the error is within 8 ulps for `f64`; for `f32` it is within 1 ulp. The
detailed form reports a bound of 64 units ([error estimates](crate#error-estimates)). See
the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    #[inline]
    pub normal_sf [S: RealScalar] (x: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_sf_with_options -> SpecialFunctionResult<S>;
    value {
        normal_tail_value(x)
    }
    report {
        normal_sf_evaluate(x, options)?.publish()
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_sf_evaluate<S: RealScalar>(
    x: S,
    options: ProbabilityOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || normal_sf_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "special-function parameter x must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let value = normal_sf_kernel(x, &mut work)?;
    normal_finish(value, &work, algorithm_component!(stringify!(normal_sf)))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the logarithm of the standard normal distribution function, ln Φ(x).

```text
Φ(x) = (1/√(2π)) ∫₋∞ˣ exp(−t²/2) dt = ½ erfc(−x/√2)
```

for real `x`. For `x ≤ 0` the result is `ln(½ erfcx(−x/√2)) − x²/2` with `x²/2` formed
exactly; for `x > 0` it is `ln1p(−(1 − Φ(x)))`. The probability itself is never formed, so the
result stays accurate where `Φ(x)` underflows.

# Examples

```
// ln Φ(x) = ln erfc(−x/√2) − ln 2; Φ(−40) itself underflows
let x = -40.0_f64;
let expected = scioxide_special::log_erfc(-x / 2.0_f64.sqrt())? - core::f64::consts::LN_2;
assert!((scioxide_special::normal_log_cdf(x)? - expected).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
On `−38 ≤ x ≤ 38` the error is within 8 ulps for `f64`; for `f32` it is within 1 ulp. The
detailed form reports a bound of 64 units ([error estimates](crate#error-estimates)). See
the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `x²/2` overflows (about `x < −1.9·10¹⁵⁴` in `f64`) or `ln
  Φ(x)` is below the range of the scalar type.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    pub normal_log_cdf [S: RealScalar] (x: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_log_cdf_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        normal_log_cdf_evaluate(x, options)
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_log_cdf_evaluate<S: RealScalar>(
    x: S,
    options: ProbabilityOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || normal_log_cdf_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "special-function parameter x must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let value = normal_log_cdf_kernel(x, &mut work)?;
    normal_finish(
        value,
        &work,
        algorithm_component!(stringify!(normal_log_cdf)),
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the logarithm of the standard normal survival function, ln(1 − Φ(x)).

```text
Φ(x) = (1/√(2π)) ∫₋∞ˣ exp(−t²/2) dt = ½ erfc(−x/√2)
```

for real `x`. `ln(1 − Φ(x)) = ln Φ(−x)`, evaluated as in
[`normal_log_cdf`](crate::normal_log_cdf): the probability is never formed, so the result stays
accurate where `1 − Φ(x)` underflows.

# Examples

```
// ln(1 − Φ(x)) = ln Φ(−x)
let upper = scioxide_special::normal_log_sf(40.0_f64)?;
let lower = scioxide_special::normal_log_cdf(-40.0_f64)?;
assert!((upper - lower).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
On `−38 ≤ x ≤ 38` the error is within 8 ulps for `f64`; for `f32` it is within 1 ulp. The
detailed form reports a bound of 64 units ([error estimates](crate#error-estimates)). See
the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `x²/2` overflows (about `x > 1.9·10¹⁵⁴` in `f64`) or `ln(1
  − Φ(x))` is below the range of the scalar type.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    pub normal_log_sf [S: RealScalar] (x: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_log_sf_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        normal_log_sf_evaluate(x, options)
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_log_sf_evaluate<S: RealScalar>(
    x: S,
    options: ProbabilityOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || normal_log_sf_evaluate::<f64>(x.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(x, "special-function parameter x must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let value = normal_log_sf_kernel(x, &mut work)?;
    normal_finish(
        value,
        &work,
        algorithm_component!(stringify!(normal_log_sf)),
    )
}

// Inline the work admission with its caller's allowance. The erfcx arithmetic
// remains shared out of line; plain calls can discard Work and its diagnostics.
#[inline(always)]
fn normal_log_sf_kernel<S: RealScalar>(x: S, w: &mut Work) -> Result<S, SciOxideError> {
    if x < S::ZERO {
        return normal_sf_kernel(-x, w).map(|tail| (-tail).ln_1p());
    }
    let factor = S::HALF * normal_erfcx(x / S::TWO.sqrt(), w)?;
    let square = normal_half_square(x.into_f64())?;
    check_finite(S::try_from_f64_lossy(
        (Wide::from(factor.into_f64().ln()) - square).value(),
    )?)
}
#[inline]
fn normal_log_cdf_kernel<S: RealScalar>(x: S, w: &mut Work) -> Result<S, SciOxideError> {
    normal_log_sf_kernel(-x, w)
}
/// Plain upper normal tail `1 - Phi(x)` (`normal_cdf` passes `-x`): the
/// detailed value without forming its report. `f32` rounds the `f64`
/// kernel once.
#[inline]
fn normal_tail_value<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    if let Some(value) = crate::support::delegate_f32::<S, _, _>(
        || normal_tail_value::<f64>(x.into_f64()),
        S::try_from_f64_lossy,
    ) {
        return value;
    }
    require_finite_param(x, "special-function parameter x must be finite")?;
    normal_tail_kernel(x)
}

/// Upper normal tail `Q(x)`, charged to `w`.
#[inline(always)]
pub(crate) fn normal_sf_kernel<S: RealScalar>(x: S, w: &mut Work) -> Result<S, SciOxideError> {
    if w.step() {
        normal_tail_kernel(x)
    } else if x.abs() > real::<S>(39.)? {
        Ok(if x < S::ZERO { S::ONE } else { S::ZERO })
    } else {
        // Preserve the incomplete estimate produced when erfcx admission fails.
        normal_tail_projection(x, S::HALF)
    }
}

/// Shared fixed-cost arithmetic, independent of work accounting and the detailed result.
#[inline(always)]
fn normal_tail_kernel<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let absolute = x.abs();
    if absolute > real::<S>(39.)? {
        return Ok(if x < S::ZERO { S::ONE } else { S::ZERO });
    }
    let factor = S::HALF * normal_erfcx_value(absolute / real::<S>(core::f64::consts::SQRT_2)?)?;
    normal_tail_projection(x, factor)
}

#[inline(always)]
fn normal_tail_projection<S: RealScalar>(x: S, factor: S) -> Result<S, SciOxideError> {
    let square = normal_half_square(x.abs().into_f64())?;
    let base = (-square.hi).exp();
    // For |x| <= 39, the omitted quadratic correction is below 2^-89 relative.
    // Retain the original exponent and subnormal tails with one exponential.
    let exponential = base - base * square.lo;
    let tail = S::try_from_f64_lossy(exponential * factor.into_f64())?;
    Ok(if x < S::ZERO { S::ONE - tail } else { tail })
}

#[inline]
fn normal_cdf_kernel<S: RealScalar>(x: S, w: &mut Work) -> Result<S, SciOxideError> {
    normal_sf_kernel(-x, w)
}

/// The shared erfcx kernel, charged as one unit of `work`.
#[inline]
fn normal_erfcx<S: RealScalar>(x: S, work: &mut Work) -> Result<S, SciOxideError> {
    if !work.step() {
        return Ok(S::ONE);
    }
    normal_erfcx_value(x)
}

#[inline(always)]
fn normal_erfcx_value<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(real_faddeeva::erfcx(x.into_f64()))
}

// The product residual retains the exact exponent of the original normal input;
// the rounded erfcx argument only affects a slowly varying factor.
#[inline]
fn normal_half_square(x: f64) -> Result<Wide, SciOxideError> {
    let square = crate::support::wide::two_product(x * 0.5, x);
    check_finite(square.hi)?;
    Ok(square)
}

/// Inverse normal log-lower-tail kernel; reflects to the smaller tail before refinement.
fn normal_inverse<S: RealScalar>(log_p: S, work: &mut Work) -> Result<S, SciOxideError> {
    if log_p > S::ZERO {
        return Err(ERRORS.outside(scioxide_core::argument!("p")));
    }
    if log_p == S::ZERO {
        return Ok(S::INFINITY);
    }
    if log_p >= -S::TWO * S::LN_2 && log_p <= -S::HALF * S::LN_2 {
        work.step();
        // 1-2p must retain its small displacement at the median. Carry the
        // real ln(2) in two parts rather than subtracting rounded probabilities.
        let log_twice = Wide::from(log_p.into_f64()) + crate::support::constants::LN_2;
        let argument = S::try_from_f64_lossy(-log_twice.exp_m1().value())?;
        return Ok(-S::TWO.sqrt() * crate::error_probability::error_fn::erfinv(argument)?);
    }
    if log_p > -S::LN_2 {
        return normal_inverse(log1mexp(log_p)?, work).map(|x| -x);
    }
    if log_p > real::<S>(-36.)? {
        work.step();
        return Ok(
            -S::TWO.sqrt() * crate::error_probability::error_fn::erfcinv(S::TWO * log_p.exp())?
        );
    }
    let mut x = -(real::<S>(2.)?.sqrt() * (-log_p).sqrt());
    // The omitted logarithmic prefactor changes this seed by less than one
    // output ULP beyond epsilon^-2. Avoid squaring a rounded seed near MAX.
    if log_p < -S::ONE / (S::EPSILON * S::EPSILON) {
        work.step();
        return Ok(x);
    }
    while work.step() {
        let actual = normal_log_cdf_kernel(x, work)?;
        if !work.complete {
            break;
        }
        let derivative =
            (real::<S>(2.)? / S::PI).sqrt() / normal_erfcx(-x / real::<S>(2.)?.sqrt(), work)?;
        if !work.complete {
            break;
        }
        let delta = (actual - log_p) / derivative;
        let next = x - delta;
        if next == x || delta.abs() <= S::TWO * (x - x.next_after(S::ZERO)).abs() {
            return Ok(next);
        }
        x = next;
    }
    Ok(x)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the standard normal quantile function Φ⁻¹(p): the `x` with `Φ(x) = p`.

```text
Φ⁻¹(p) = −√2 erfcinv(2p),   0 ≤ p ≤ 1,
```

where `Φ` is the standard normal distribution function ([`normal_cdf`](crate::normal_cdf)).
`p = 0` returns `−∞` and `p = 1` returns `+∞`; there are no interior poles.

The method applies the fixed-cost rational approximation of [`erfinv`](crate::erfinv) to
the smaller tail `min(p, 1 − p)`, which is doubled exactly, with no iteration.

# Examples

```
// Φ⁻¹(0.975) = 1.959963984540054
let x = scioxide_special::normal_quantile(0.975_f64)?;
assert!((x - 1.959_963_984_540_054).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
The error is within 4 ulps for `f64`, including subnormal `p`, and 1 ulp for `f32` on the
tested range. The detailed form reports a bound of 64 units ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p < 0` or `p > 1`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2.
- NIST Digital Library of Mathematical Functions, §7.17. <https://dlmf.nist.gov/7.17>"###)]

    #[inline]
    pub normal_quantile [S: RealScalar] (p: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_quantile_with_options -> SpecialFunctionResult<S>;
    value {
        normal_quantile_value(p, false)
    }
    report {
        normal_quantile_evaluate(p, options, None)?.publish()
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_quantile_evaluate<S: RealScalar>(
    p: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    normal_inverse_evaluate(p, options, false, false, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of the standard normal survival function: the `x` with `1 − Φ(x) = p`.

```text
x = −Φ⁻¹(p) = √2 erfcinv(2p),   0 ≤ p ≤ 1,
```

where `Φ` is the standard normal distribution function ([`normal_cdf`](crate::normal_cdf)).
`p = 0` returns `+∞` and `p = 1` returns `−∞`; there are no interior poles.

The method applies the fixed-cost rational approximation of [`erfinv`](crate::erfinv) to
the smaller tail `min(p, 1 − p)`, which is doubled exactly, with no iteration.

# Examples

```
// upper 2.5% point: 1 − Φ(1.959963984540054) = 0.025
let x = scioxide_special::normal_isf(0.025_f64)?;
assert!((x - 1.959_963_984_540_054).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
The error is within 4 ulps for `f64`, including subnormal `p`, and 1 ulp for `f32` on the
tested range. The detailed form reports a bound of 64 units ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p < 0` or `p > 1`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    #[inline]
    pub normal_isf [S: RealScalar] (p: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_isf_with_options -> SpecialFunctionResult<S>;
    value {
        normal_quantile_value(p, true)
    }
    report {
        normal_isf_evaluate(p, options, None)?.publish()
    }

}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_isf_evaluate<S: RealScalar>(
    p: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    normal_inverse_evaluate(p, options, false, true, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of ln Φ: the `x` with `ln Φ(x) = p`.

The domain is `p ≤ 0`, with `p = 0` returning `+∞`. Because `p` is a logarithm, the inverse is
available for probabilities far below the smallest subnormal. Here `Φ` is the standard normal
distribution function ([`normal_cdf`](crate::normal_cdf)).

For `p > −36` the result comes from the inverse error function, with no iteration. For
deeper tails it uses Newton's method on `ln Φ(x) − p` from the asymptotic start
`x₀ = −√(−2p)`; each step counts toward the iteration budget of
[`ProbabilityOptions`](crate::ProbabilityOptions).

# Examples

```
// round trip through ln Φ(−40) ≈ −804.6, where Φ itself underflows
let log_p = scioxide_special::normal_log_cdf(-40.0_f64)?;
let x = scioxide_special::normal_inv_log_cdf(log_p)?;
assert!((x + 40.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target. The detailed form reports a bound of 64 units ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p > 0`.
- `NonFinite` if `p = −∞`; use [`normal_quantile`](crate::normal_quantile) with `p = 0` for that
  limit.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    pub normal_inv_log_cdf [S: RealScalar] (p: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_inv_log_cdf_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        normal_inv_log_cdf_evaluate(p, options, context)
    }
    scoped pub normal_inv_log_cdf_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_inv_log_cdf_evaluate<S: RealScalar>(
    p: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    normal_inverse_evaluate(p, options, true, false, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of ln(1 − Φ): the `x` with `ln(1 − Φ(x)) = p`.

The domain is `p ≤ 0`, with `p = 0` returning `−∞`. The result is the negative of
[`normal_inv_log_cdf`](crate::normal_inv_log_cdf)`(p)`. Here `Φ` is the standard normal
distribution function ([`normal_cdf`](crate::normal_cdf)).

For `p > −36` the result comes from the inverse error function, with no iteration. For
deeper tails it uses Newton's method on `ln Φ(x) − p` from the asymptotic start
`x₀ = −√(−2p)`; each step counts toward the iteration budget of
[`ProbabilityOptions`](crate::ProbabilityOptions).

# Examples

```
// round trip through ln(1 − Φ(40)) ≈ −804.6, where the tail itself underflows
let log_p = scioxide_special::normal_log_sf(40.0_f64)?;
let x = scioxide_special::normal_inv_log_sf(log_p)?;
assert!((x - 40.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target. The detailed form reports a bound of 64 units ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p > 0`.
- `NonFinite` if `p = −∞`; use [`normal_isf`](crate::normal_isf) with `p = 0` for that limit.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §26.2."###)]

    pub normal_inv_log_sf [S: RealScalar] (p: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed normal_inv_log_sf_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        normal_inv_log_sf_evaluate(p, options, context)
    }
    scoped pub normal_inv_log_sf_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the detailed result before it is projected and returned."]
#[inline(always)]
pub(crate) fn normal_inv_log_sf_evaluate<S: RealScalar>(
    p: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    normal_inverse_evaluate(p, options, true, true, context)
}

/// Shared normal inverse validation, fixed rational kernels, and logarithmic iteration.
fn normal_inverse_evaluate<S: RealScalar>(
    p: S,
    options: ProbabilityOptions,
    logarithmic: bool,
    upper: bool,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || {
            normal_inverse_evaluate::<f64>(
                p.into_f64(),
                options,
                logarithmic,
                upper,
                context.as_deref_mut(),
            )
        },
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(p, "probability parameter p must be finite")?;
    if (!logarithmic && (p < S::ZERO || p > S::ONE)) || (logarithmic && p > S::ZERO) {
        return Err(ERRORS.outside(scioxide_core::argument!("p")));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let v = if !logarithmic && p == S::ZERO {
        -S::INFINITY
    } else if !logarithmic && p == S::ONE {
        S::INFINITY
    } else if !logarithmic {
        work.step();
        S::try_from_f64_lossy(normal_quantile_kernel(p.into_f64())?)?
    } else {
        normal_inverse(p, &mut work)?
    };
    work.check()?;
    let v = if upper { -v } else { v };
    normal_finish(
        v,
        &work,
        match (logarithmic, upper) {
            (false, false) => algorithm_component!("normal_quantile"),
            (false, true) => algorithm_component!("normal_isf"),
            (true, false) => algorithm_component!("normal_inv_log_cdf"),
            (true, true) => algorithm_component!("normal_inv_log_sf"),
        },
    )
}

/// Plain normal quantile (`upper`: inverse survival): the detailed value
/// without forming its report. `f32` rounds the `f64` kernel once.
#[inline]
fn normal_quantile_value<S: RealScalar>(p: S, upper: bool) -> Result<S, SciOxideError> {
    require_finite_param(p, "probability parameter p must be finite")?;
    if p < S::ZERO || p > S::ONE {
        return Err(ERRORS.outside(scioxide_core::argument!("p")));
    }
    let v = if p == S::ZERO {
        -S::INFINITY
    } else if p == S::ONE {
        S::INFINITY
    } else {
        S::try_from_f64_lossy(normal_quantile_kernel(p.into_f64())?)?
    };
    Ok(if upper { -v } else { v })
}

/// Lower-tail standard normal quantile for `0 < p < 1` in `f64`.
///
/// The smaller tail doubles exactly into the inverse-error complement, so
/// the rational kernel never sees a rounded `1 - 2p`.
#[inline]
pub(crate) fn normal_quantile_kernel(p: f64) -> Result<f64, SciOxideError> {
    let tail = if p <= 0.5 { p } else { 1.0 - p };
    let complement = tail + tail;
    let magnitude =
        crate::error_probability::error_fn::inverse_error_rational(1.0 - complement, complement)?
            * core::f64::consts::SQRT_2;
    Ok(if p <= 0.5 { -magnitude } else { magnitude })
}
