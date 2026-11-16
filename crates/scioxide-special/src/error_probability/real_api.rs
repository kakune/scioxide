//! Real evaluation adapters and family accuracy envelopes.

use crate::result::{Bound, Evaluation, Ulps, enveloped};
use crate::{ErrorFunctionOptions, SpecialFunctionResult, SpecialFunctionTermination};
use scioxide_core::{RealScalar, SciOxideError, algorithm_id};

const INVERSE_ERROR_ULPS: Ulps = Ulps::new(4.0, 1.0);
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse error function: the real `x` with `erf(x) = p`.

The domain is `−1 ≤ p ≤ 1`. The function is odd, `erfinv(0) = 0` with the sign of zero kept,
and `erfinv(±1) = ±∞`. There are no interior poles. Near `p = ±1` the complement `1 − |p|` is
not representable to full precision; if it is known, use [`erfcinv`](crate::erfcinv) instead.

The method is a fixed-cost piecewise rational approximation in `p` for `|p| ≤ 1/2` and in
`√(−ln(1 − |p|))` beyond, with no iteration.

# Examples

```
// erfinv(erf(x)) = x
let p = scioxide_special::erf(0.5_f64)?;
assert!((scioxide_special::erfinv(p)? - 0.5).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
The error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range. The
detailed form reports a bound of 4 units for `f64` and 1 unit for `f32` ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p < −1` or `p > 1`.

# References

- NIST Digital Library of Mathematical Functions, §7.17. <https://dlmf.nist.gov/7.17>"###)]

    pub erfinv [S: RealScalar] (x: S) -> S
    where [];
    options options: ErrorFunctionOptions;
    detailed erfinv_with_options -> SpecialFunctionResult<S>;
    value { crate::error_probability::error_fn::erfinv(x) }
    report { erfinv_evaluate(x, options)?.publish() }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse complementary error function: the real `x` with `erfc(x) = p`.

The domain is `0 ≤ p ≤ 2`. `erfcinv(0) = +∞`, `erfcinv(1) = 0` and `erfcinv(2) = −∞`; there
are no interior poles. The tail logarithm is formed directly from `p`, so small probabilities,
including subnormal ones, are not rounded away.

The method is the fixed-cost piecewise rational approximation of [`erfinv`](crate::erfinv),
applied to `1 − p` and the complement `min(p, 2 − p)`, with no iteration.

# Examples

```
// erfcinv(erfc(x)) = x
let p = scioxide_special::erfc(2.0_f64)?;
assert!((scioxide_special::erfcinv(p)? - 2.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
The error is within 4 ulps for `f64`, including subnormal `p`, and 1 ulp for `f32` on the
tested range. The detailed form reports a bound of 4 units for `f64` and 1 unit for `f32`
([error estimates](crate#error-estimates)). See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `p < 0` or `p > 2`.

# References

- NIST Digital Library of Mathematical Functions, §7.17. <https://dlmf.nist.gov/7.17>"###)]

    pub erfcinv [S: RealScalar] (x: S) -> S
    where [];
    options options: ErrorFunctionOptions;
    detailed erfcinv_with_options -> SpecialFunctionResult<S>;
    value { crate::error_probability::error_fn::erfcinv(x) }
    report { erfcinv_evaluate(x, options)?.publish() }
}

/// Evaluate erf with its analytic derivative and detailed metadata.
///
/// See [`crate::erf`] for the mathematical definition, domain, limits and precision/accuracy
/// envelope. Inspect termination and estimate availability; exhausted work can keep diagnostics.
///
/// # Errors
///
/// The input, domain and arithmetic errors of [`crate::erf`] apply.
pub(crate) fn erf_evaluation<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
    _context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::error_probability::complex_error::real_error_function(
        x,
        false,
        options.iteration_budget(),
    )
}

/// Evaluate erfc with its analytic derivative and detailed metadata.
///
/// See [`crate::erfc`] for the mathematical definition, domain, limits and precision/accuracy
/// envelope. Inspect termination and estimate availability; exhausted work can keep diagnostics.
///
/// # Errors
///
/// The input, domain and arithmetic errors of [`crate::erfc`] apply.
pub(crate) fn erfc_evaluation<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
    _context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::error_probability::complex_error::real_error_function(
        x,
        true,
        options.iteration_budget(),
    )
}

/// Evaluates the detailed result before it is projected and returned.
fn erfinv_evaluate<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::support::admit_family_work(1, options.iteration_budget())?;
    enveloped(
        crate::error_probability::error_fn::erfinv(x)?,
        INVERSE_ERROR_ULPS,
        algorithm_id!("erfinv", "piecewise_rational", 1),
        1,
        SpecialFunctionTermination::Converged,
    )
}

/// Evaluates the detailed result before it is projected and returned.
fn erfcinv_evaluate<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::support::admit_family_work(1, options.iteration_budget())?;
    enveloped(
        crate::error_probability::error_fn::erfcinv(x)?,
        INVERSE_ERROR_ULPS,
        algorithm_id!("erfcinv", "piecewise_rational", 1),
        1,
        SpecialFunctionTermination::Converged,
    )
}

/// Reported error envelope of the `f64` Dawson kernel, checked against reference values.
const DAWSON_ULPS: Ulps = Ulps::new(2.0, 1.0);
#[inline]
pub(crate) fn dawson<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    crate::support::require_finite_param(x, "Dawson argument must be finite")?;
    S::try_from_f64_lossy(crate::error_probability::real_faddeeva::dawson(
        x.into_f64(),
    ))
}
#[inline]
pub(crate) fn dawson_evaluation<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
    _context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::support::admit_family_work(1, options.iteration_budget())?;
    dawson_evaluate(x, true)
}

/// Dawson, with its derivative when `slope` is set.
fn dawson_evaluate<S: RealScalar>(x: S, slope: bool) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || dawson_evaluate::<f64>(x.into_f64(), slope),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    crate::support::require_finite_param(x, "integral parameter x must be finite")?;
    let x = x.into_f64();
    let value = S::try_from_f64_lossy(crate::error_probability::real_faddeeva::dawson(x))?;
    let bound = Bound::envelope(value, DAWSON_ULPS)?;
    let derivative = if slope {
        let (d, error) = crate::error_probability::real_faddeeva::dawson_slope(
            x,
            value.into_f64(),
            bound.absolute.into_f64(),
        );
        Some((
            S::try_from_f64_lossy(d)?,
            Bound::derived(S::try_from_f64_lossy(error)?),
        ))
    } else {
        None
    };
    Ok(Evaluation::finite(
        value,
        bound,
        derivative,
        algorithm_id!("error", "dawson_binned_asymptotic", 1),
        1,
        SpecialFunctionTermination::Converged,
    ))
}
pub(crate) fn erfi<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    erfi_evaluate(x, false, false, ErrorFunctionOptions::default())?.value()
}

#[inline]
pub(crate) fn erfi_evaluation<S: RealScalar>(
    x: S,
    options: ErrorFunctionOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    erfi_evaluate(x, false, true, options)
}

/// One fixed-cost Dawson evaluation; exponential growth stays in the scale.
fn erfi_evaluate<S: RealScalar>(
    x: S,
    scaled: bool,
    detailed: bool,
    options: ErrorFunctionOptions,
) -> Result<crate::complex_support::ScaledEvaluation<S>, SciOxideError> {
    use crate::support::{Jet, JetData, JetEstimate, real};
    use core::f64::consts::FRAC_2_SQRT_PI;
    crate::support::require_finite_param(x, "error-function parameter x must be finite")?;
    crate::support::admit_family_work(1, options.iteration_budget())?;
    let x = x.into_f64();
    let f = crate::error_probability::real_faddeeva::dawson(x);
    let value = FRAC_2_SQRT_PI * f;
    // Two Dawson ULPs plus the final multiplication and subnormal rounding.
    let error = 4.0 * f64::EPSILON * value.abs() + 4.0 * f64::from_bits(1);
    let (derivative, derivative_error) = if !detailed {
        (0.0, 0.0)
    } else if scaled {
        let (slope, bound) =
            crate::error_probability::real_faddeeva::dawson_slope(x, f, error / FRAC_2_SQRT_PI);
        (
            FRAC_2_SQRT_PI * slope,
            FRAC_2_SQRT_PI * bound + 2.0 * f64::EPSILON * slope.abs(),
        )
    } else {
        (FRAC_2_SQRT_PI, 2.0 * f64::EPSILON * FRAC_2_SQRT_PI)
    };
    let exponent = if scaled {
        0.0
    } else {
        crate::support::check_finite(x * x)?
    };
    crate::complex_support::report_with_status(
        Ok(JetEstimate::from(JetData {
            y: Jet {
                v: real(value),
                d: real(derivative),
            },
            scale: Jet::constant(real(exponent)),
            error,
            derivative_error: Some(derivative_error),
            method: scioxide_core::algorithm_component!("dawson_binned_asymptotic"),
        })),
        detailed,
        if scaled {
            scioxide_core::algorithm_component!("erfi_scaled")
        } else {
            scioxide_core::algorithm_component!("erfi")
        },
        1,
        false,
    )
}
