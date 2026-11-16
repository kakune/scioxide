//! Real spherical Bessel kernels. Formulas: DLMF 10.49, 10.51,
//! 10.53; Miller normalization uses the larger of the two elementary seeds.

use crate::expansions::{bounded, real};
use crate::result::Evaluation;
use crate::support::Work;
use crate::{SpecialFunctionResult, SphericalBesselOptions};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, RealScalar, SciOxideError, algorithm_component,
};

type Detailed<S> = SpecialFunctionResult<S>;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    J,
    Y,
    I,
    K,
}

#[derive(Clone, Copy)]
struct Pair<S> {
    value: S,
    derivative: S,
    error: S,
    derivative_error: S,
    complete: bool,
    log_scale: S,
    model_based: bool,
}

fn partial<S: RealScalar>() -> Pair<S> {
    Pair {
        value: S::ZERO,
        derivative: S::ZERO,
        error: S::MAX,
        derivative_error: S::MAX,
        complete: false,
        log_scale: S::ZERO,
        model_based: false,
    }
}
pub(crate) fn spherical_jn<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::J, false, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_jn_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::J, false, false, order, x, options, context)
}

pub(crate) fn spherical_yn<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::Y, false, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_yn_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::Y, false, false, order, x, options, context)
}

pub(crate) fn spherical_in<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::I, false, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_in_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::I, false, false, order, x, options, context)
}

pub(crate) fn spherical_kn<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::K, false, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_kn_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::K, false, false, order, x, options, context)
}

pub(crate) fn spherical_in_scaled<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::I, true, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_in_scaled_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::I, true, false, order, x, options, context)
}

pub(crate) fn spherical_kn_scaled<S: RealScalar>(
    order: u32,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <SphericalBesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    evaluate(Kind::K, true, false, order, x, options, None)?.value()
}

#[inline]
pub(crate) fn spherical_kn_scaled_evaluation<S: RealScalar>(
    order: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::K, true, false, order, x, options, context)
}

fn validate<S: RealScalar>(kind: Kind, x: S) -> Result<(), SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "argument must be finite",
        ));
    }
    if matches!(kind, Kind::Y) && x == S::ZERO || matches!(kind, Kind::K) && x <= S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                scioxide_core::DomainReason::Pole
            } else {
                scioxide_core::DomainReason::OutsideDomain
            },
            "Y requires x != 0; K requires x > 0",
        ));
    }
    Ok(())
}

pub(crate) fn evaluate<S: RealScalar>(
    kind: Kind,
    scaled: bool,
    riccati: bool,
    n: u32,
    x: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !(riccati && matches!(kind, Kind::Y) && n == 0 && x == S::ZERO) {
        validate(kind, x)?;
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let a = x.abs();
    let (mut pair, method) = if a == S::ZERO {
        (
            Pair {
                value: if n == 0 {
                    if riccati && matches!(kind, Kind::Y) {
                        -S::ONE
                    } else {
                        S::ONE
                    }
                } else {
                    S::ZERO
                },
                derivative: if n == 1 { S::ONE / real(3.0)? } else { S::ZERO },
                error: S::ZERO,
                derivative_error: if n == 1 { S::EPSILON } else { S::ZERO },
                complete: true,
                log_scale: S::ZERO,
                model_based: true,
            },
            algorithm_component!("origin"),
        )
    } else if matches!(kind, Kind::J | Kind::I) && a <= S::ONE {
        (
            series(kind, n, a, &mut work)?,
            algorithm_component!("origin_series"),
        )
    } else {
        (
            recurrence(kind, riccati, n, a, &mut work)?,
            algorithm_component!("directional_recurrence"),
        )
    };
    work.check()?;
    if riccati && a <= S::ONE && matches!(kind, Kind::J) {
        pair.derivative = pair.value + a * pair.derivative;
        pair.derivative_error =
            pair.error + a * pair.derivative_error + S::EPSILON * pair.derivative.abs();
        pair.value = a * pair.value;
        pair.error = a * pair.error + S::EPSILON * pair.value.abs();
    }
    // The recurrence for modified functions is exponentially scaled throughout.
    let exponent = match kind {
        Kind::I if a <= S::ONE && scaled => -a,
        Kind::I if a > S::ONE && !scaled => a,
        Kind::K if !scaled => -a,
        _ => S::ZERO,
    };
    let exponent = exponent + pair.log_scale;
    if exponent != S::ZERO && pair.complete {
        pair.value = multiply_exp(pair.value, exponent);
        pair.derivative = multiply_exp(pair.derivative, exponent);
        pair.error = multiply_exp(pair.error, exponent).abs();
        pair.derivative_error = multiply_exp(pair.derivative_error, exponent).abs();
    }
    if x.is_sign_negative() {
        let odd = (n % 2 == 1) ^ matches!(kind, Kind::Y) ^ riccati;
        if odd {
            pair.value = -pair.value;
        }
        if !odd {
            pair.derivative = -pair.derivative;
        }
    }
    let family = if riccati {
        if matches!(kind, Kind::J) {
            algorithm_component!("riccati_jn")
        } else {
            algorithm_component!("riccati_yn")
        }
    } else {
        match (kind, scaled) {
            (Kind::J, _) => algorithm_component!("spherical_jn"),
            (Kind::Y, _) => algorithm_component!("spherical_yn"),
            (Kind::I, false) => algorithm_component!("spherical_in"),
            (Kind::I, true) => algorithm_component!("spherical_in_scaled"),
            (Kind::K, false) => algorithm_component!("spherical_kn"),
            (Kind::K, true) => algorithm_component!("spherical_kn_scaled"),
        }
    };
    let bound = |error| {
        if pair.model_based {
            crate::result::Bound::derived(error)
        } else {
            crate::result::Bound::heuristic(error)
        }
    };
    let derivative = (pair.derivative.is_finite() && pair.derivative_error.is_finite())
        .then(|| (pair.derivative, bound(pair.derivative_error)));
    Ok(bounded(
        pair.value,
        bound(pair.error),
        AlgorithmId::from_components(family, method, 1),
        work.used,
        pair.complete,
    )?
    .with_derivative(derivative))
}

fn multiply_exp<S: RealScalar>(value: S, exponent: S) -> S {
    if value == S::ZERO {
        return value;
    }
    let factor = exponent.exp();
    if factor.is_finite() && factor >= S::MIN_POSITIVE {
        value * factor
    } else {
        let magnitude = (value.abs().ln() + exponent).exp();
        if value < S::ZERO {
            -magnitude
        } else {
            magnitude
        }
    }
}

fn series<S: RealScalar>(
    kind: Kind,
    n: u32,
    x: S,
    work: &mut Work,
) -> Result<Pair<S>, SciOxideError> {
    // Build x^(n-1)/(2n+1)!! first to preserve the derivative when x^n underflows.
    let mut base = S::ONE;
    for k in 1..=n {
        if !work.step() {
            return Ok(partial());
        }
        base /= S::try_from_f64_exact(2.0 * f64::from(k) + 1.0)?;
        if k != n {
            base *= x;
        }
    }
    let ns = S::try_from_f64_exact(f64::from(n))?;
    let mut value = if n == 0 { S::ONE } else { base * x };
    let mut derivative = if n == 0 { S::ZERO } else { ns * base };
    let mut term = value;
    let mut derivative_term = derivative;
    let sign = if matches!(kind, Kind::J) {
        -S::ONE
    } else {
        S::ONE
    };
    let xx = x * x;
    let mut absolute = value.abs();
    let mut derivative_absolute = derivative.abs();
    let mut complete = false;
    while work.step() {
        let k = work.used - n as usize;
        let ks = S::try_from_usize_exact(k)?;
        let denominator = real::<S>(2.0)? * ks * (real::<S>(2.0)? * (ns + ks) + S::ONE);
        let ratio = sign * xx / denominator;
        if k == 1 && n == 0 {
            derivative_term = sign * x / real(3.0)?;
        } else {
            derivative_term = derivative_term * ratio * (ns + real::<S>(2.0)? * ks)
                / (ns + real::<S>(2.0)? * (ks - S::ONE));
        }
        term *= ratio;
        value += term;
        derivative += derivative_term;
        absolute += term.abs();
        derivative_absolute += derivative_term.abs();
        if term.abs() <= S::EPSILON * value.abs()
            && derivative_term.abs() <= S::EPSILON * derivative.abs()
        {
            complete = true;
            break;
        }
    }
    let rounding = real::<S>(8.0)? * S::EPSILON * S::try_from_usize_exact(work.used + 1)?;
    // Each of the 2n+2 leading operations may round into the subnormal range
    // by half a quantum; the factors x and 1/(2k+1) never amplify it. Above
    // |value| = η/ε and |derivative| = (n+1)η/ε the rounding term already
    // exceeds these quanta, so normal results skip the subnormal arithmetic.
    let order = S::try_from_usize_exact(n as usize + 1)?;
    let subnormal = if value.abs() < S::MIN_POSITIVE || derivative.abs() < order * S::MIN_POSITIVE {
        S::MIN_SUBNORMAL * (order + order)
    } else {
        S::ZERO
    };
    Ok(Pair {
        value,
        derivative,
        error: term.abs() + rounding * absolute + subnormal,
        derivative_error: derivative_term.abs()
            + rounding * derivative_absolute
            + subnormal * order,
        complete,
        log_scale: S::ZERO,
        model_based: false,
    })
}

fn recurrence<S: RealScalar>(
    kind: Kind,
    riccati: bool,
    n: u32,
    x: S,
    work: &mut Work,
) -> Result<Pair<S>, SciOxideError> {
    if !work.step() {
        return Ok(partial());
    }
    let inv = S::ONE / x;
    let (zero, one) = match kind {
        Kind::J if riccati => (x.sin(), x.sin() * inv - x.cos()),
        Kind::J => (x.sin() * inv, (x.sin() * inv - x.cos()) * inv),
        Kind::Y if riccati => (-x.cos(), -x.cos() * inv - x.sin()),
        Kind::Y => (-x.cos() * inv, (-x.cos() * inv - x.sin()) * inv),
        Kind::I => {
            let tail = (-x - x).exp();
            let zero = -(-x - x).exp_m1() * S::HALF * inv;
            (zero, ((S::ONE + tail) * S::HALF - zero) * inv)
        }
        Kind::K => {
            let zero = S::PI * S::HALF * inv;
            (zero, zero * (S::ONE + inv))
        }
    };
    let ns = S::try_from_f64_exact(f64::from(n))?;
    let use_miller = matches!(kind, Kind::J) && ns > x
        || matches!(kind, Kind::I) && x < real::<S>(4.0)? * (ns + S::ONE) * (ns + S::ONE);
    let mut log_scale = S::ZERO;
    // Seed subtraction is conditioned by its operands, not by the result.
    // Eight rounding units cover the elementary evaluation and divisions;
    // sixteen cover construction of the adjacent seed.
    let amplitude = if riccati { S::ONE } else { inv };
    let zero_error = real::<S>(8.0)? * S::EPSILON * amplitude.max(zero.abs());
    let one_error = real::<S>(16.0)? * S::EPSILON * (amplitude * (S::ONE + inv)).max(one.abs());
    let (value, next, value_error, next_error) = if use_miller {
        // Compare independently truncated downward recurrences. Every step is
        // charged, including repeated starts; no unbounded order conversion.
        let margin = (real::<S>(40.0)? * (ns + x + S::ONE)).sqrt().ceil() + real(24.0)?;
        if margin > S::try_from_usize_exact(work.remaining)? {
            return Ok(partial());
        }
        let mut top = (n as usize)
            .checked_add(margin.try_to_usize_exact()?)
            .ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("order"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "recurrence order overflow",
                )
            })?;
        let mut previous: Option<(S, S, S)> = None;
        loop {
            if top > work.remaining {
                return Ok(partial());
            }
            let (v, p, scale) = miller(kind, n as usize, top, inv, zero, one, work)?;
            if let Some((old, old_next, old_scale)) = previous {
                let factor = (old_scale - scale).exp();
                let delta = (v - old * factor).abs() + (p - old_next * factor).abs();
                if delta
                    <= (real::<S>(32.0)? + real::<S>(4.0)? * scale.abs())
                        * S::EPSILON
                        * (v.abs() + p.abs()).max(S::MIN_POSITIVE)
                {
                    log_scale = scale;
                    let seed_relative = if zero.abs() >= one.abs() {
                        zero_error / zero.abs()
                    } else {
                        one_error / one.abs()
                    };
                    let rounding = seed_relative
                        + real::<S>(16.0)? * S::EPSILON * S::try_from_usize_exact(work.used + 1)?;
                    break (v, p, delta + rounding * v.abs(), delta + rounding * p.abs());
                }
            }
            previous = Some((v, p, scale));
            top = top.saturating_add(24);
        }
    } else {
        let mut previous = zero;
        let mut current = one;
        let mut previous_error = zero_error;
        let mut current_error = one_error;
        for k in 1..=n {
            if !work.step() {
                return Ok(partial());
            }
            let coefficient = S::try_from_f64_exact(2.0 * f64::from(k) + 1.0)? * inv;
            if matches!(kind, Kind::K) {
                let threshold = S::MAX.sqrt();
                if current.abs() > threshold / coefficient.abs().max(S::ONE) {
                    previous /= threshold;
                    current /= threshold;
                    previous_error /= threshold;
                    current_error /= threshold;
                    log_scale += threshold.ln();
                }
            }
            let next = match kind {
                Kind::J | Kind::Y => coefficient * current - previous,
                Kind::I => previous - coefficient * current,
                Kind::K => previous + coefficient * current,
            };
            let next_error = (coefficient.abs() * current_error
                + previous_error
                + real::<S>(4.0)? * S::EPSILON * ((coefficient * current).abs() + previous.abs()))
            .min(S::MAX);
            previous = current;
            previous_error = current_error;
            current_error = next_error;
            current = if k == n {
                next
            } else {
                crate::support::check_finite(next)?
            };
        }
        (previous, current, previous_error, current_error)
    };
    let derivative_order = ns + if riccati { S::ONE } else { S::ZERO };
    let derivative = match kind {
        Kind::I => (derivative_order * inv) * value + next,
        Kind::J if riccati && n == 0 => x.cos(),
        Kind::Y if riccati && n == 0 => x.sin(),
        _ => (derivative_order * inv) * value - next,
    };
    let rounding = real::<S>(4.0)? * S::EPSILON;
    Ok(Pair {
        value,
        derivative,
        error: value_error,
        derivative_error: if riccati && n == 0 {
            rounding * derivative.abs()
        } else {
            ((derivative_order * inv).abs() * value_error
                + next_error
                + rounding * ((derivative_order * inv * value).abs() + next.abs()))
            .min(S::MAX)
        },
        complete: true,
        log_scale,
        model_based: !use_miller,
    })
}

fn miller<S: RealScalar>(
    kind: Kind,
    n: usize,
    top: usize,
    inv: S,
    zero: S,
    one: S,
    work: &mut Work,
) -> Result<(S, S, S), SciOxideError> {
    let mut current = S::ONE;
    let mut upper = S::ZERO;
    let mut target = S::ZERO;
    let mut target_next = S::ZERO;
    let threshold = S::MAX.sqrt();
    let mut target_log_scale = S::ZERO;
    for k in (1..=top).rev() {
        if !work.step() {
            return Ok((S::ZERO, S::ZERO, S::ZERO));
        }
        let coefficient = (real::<S>(2.0)? * S::try_from_usize_exact(k)? + S::ONE) * inv;
        if current.abs() > threshold / coefficient.abs().max(S::ONE) {
            current /= threshold;
            upper /= threshold;
            if k < n {
                target_log_scale -= threshold.ln();
            }
        }
        if k == n {
            target = current;
            target_next = upper;
        }
        let lower = coefficient * current
            + if matches!(kind, Kind::I) {
                upper
            } else {
                -upper
            };
        upper = current;
        current = lower;
    }
    if n == 0 {
        target = current;
        target_next = upper;
    }
    let (seed, denominator) = if zero.abs() >= one.abs() {
        (zero, current)
    } else {
        (one, upper)
    };
    let normalize = seed / denominator;
    let value = target * normalize;
    let next = target_next * normalize;
    if target_log_scale == S::ZERO
        && normalize.abs() >= S::MIN_POSITIVE
        && value.abs() >= S::MIN_POSITIVE
        && next.abs() >= S::MIN_POSITIVE
    {
        return Ok((value, next, S::ZERO));
    }
    let magnitude = target.abs().max(target_next.abs());
    let sign = if seed.is_sign_negative() == denominator.is_sign_negative() {
        S::ONE
    } else {
        -S::ONE
    };
    let log_scale = magnitude.ln() + seed.abs().ln() - denominator.abs().ln() + target_log_scale;
    Ok((
        sign * (target / magnitude),
        sign * (target_next / magnitude),
        log_scale,
    ))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the radial eigenfunction `H_l(λ, η)` of the Laplacian on three-dimensional hyperbolic space.

Here `l` is the nonnegative integer `order`, `λ` the real wave number (`lambda`) and `η ≥ 0`
the radial coordinate (`eta`). The functions are normalized by

```text
H₀(λ, η) = sin(λη) / (λ sinh η),
H₁(λ, η) = (coth η · H₀(λ, η) − cos(λη) / sinh η) / √(1 + λ²),
√(λ² + (l+1)²) H_{l+1} = (2l + 1) coth η · H_l − √(λ² + l²) H_{l−1},
```

so that `H_l(λ, η) → j_l(λη)` as `η → 0` with `λη` fixed. Up to a factor depending only on `λ`
and `l`, `H_l(λ, η)` equals `(sinh η)^(−1/2) P^(−1/2−l)_(−1/2+iλ)(cosh η)`, a conical function
(DLMF 14.20). `H_l` is even in `λ`, `λ = 0` takes the continuous limit, and at `η = 0`,
`H₀ = 1` while higher orders are zero.

For `η ≤ 1` and `|λ|η ≤ 1` the function sums a hypergeometric series. Otherwise it starts from
`H₀` and `H₁` and applies the recurrence above: upward when `l ≤ |λ| sinh η` or
`η > ln(l + 1) + 2`, and with Miller's backward recurrence otherwise. The cost grows linearly
with `l`.

# Examples

```
// H₀(λ, η) = sin(λη) / (λ sinh η)
let (lambda, eta) = (3.0_f64, 0.5);
let expected = (lambda * eta).sin() / (lambda * eta.sinh());
let h = scioxide_special::legendre_h3d(0, lambda, eta)?;
assert!((h - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error. The error estimate of
[`legendre_h3d_with_options`](crate::legendre_h3d_with_options) is heuristic when Miller's
recurrence is used. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `η < 0`.
- `NumericalInstability` (`LossOfSignificance`) if `λη` overflows.
- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`SphericalBesselOptions`](crate::SphericalBesselOptions) (65536 units by default). The cost
  grows linearly with `l`.

# References

NIST DLMF, §14.20. <https://dlmf.nist.gov/14.20>"###)]

    pub legendre_h3d [S: RealScalar] (order: u32, lambda: S, eta: S) -> S
    where [];
    options options: SphericalBesselOptions;
    detailed legendre_h3d_with_options -> Detailed<S>;
    evaluate |options, context| {
        legendre_h3d_evaluate(order, lambda, eta, options, context)
    }
    scoped pub legendre_h3d_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluate before projecting the value or building the detailed result.
fn legendre_h3d_evaluate<S: RealScalar>(
    order: u32,
    lambda: S,
    eta: S,
    options: SphericalBesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    validate(Kind::J, lambda)?;
    crate::expansions::nonnegative(eta, scioxide_core::argument!("eta"))?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let computed = h3d_kernel(order, lambda.abs(), eta, &mut work);
    work.check()?;
    let (value, error, complete, method) = computed?;
    bounded(
        value,
        if method == algorithm_component!("miller_recurrence") {
            crate::result::Bound::heuristic(error)
        } else {
            crate::result::Bound::derived(error)
        },
        scioxide_core::AlgorithmId::from_components(
            algorithm_component!("legendre_h3d"),
            method,
            1,
        ),
        work.used,
        complete,
    )
}

fn h3d_kernel<S: RealScalar>(
    n: u32,
    lambda: S,
    eta: S,
    work: &mut Work,
) -> Result<(S, S, bool, AlgorithmComponent), SciOxideError> {
    if eta == S::ZERO {
        return Ok((
            if n == 0 { S::ONE } else { S::ZERO },
            S::ZERO,
            true,
            algorithm_component!("origin"),
        ));
    }
    let ns = S::try_from_f64_exact(f64::from(n))?;
    let theta = lambda * eta;
    if eta <= S::ONE && theta <= S::ONE {
        // Defining associated-Legendre hypergeometric; GSL manual H3d
        // normalization, analytically cancelling the apparent lambda=0 pole.
        let half = eta * S::HALF;
        let sh = half.sinh();
        let th = half.tanh();
        let mut prefactor = S::ONE / half.cosh();
        for j in 1..=n {
            if !work.step() {
                return Ok((
                    S::ZERO,
                    S::MAX,
                    false,
                    algorithm_component!("origin_series"),
                ));
            }
            let js = S::try_from_f64_exact(f64::from(j))?;
            prefactor *= (lambda.hypot(js) * th) / (js + S::HALF);
        }
        let mut sum = S::ONE;
        let mut term = S::ONE;
        let mut absolute = S::ONE;
        let mut complete = false;
        let mut index = 0;
        while work.step() {
            index += 1;
            let k = S::try_from_usize_exact(index)?;
            let a = (k - S::HALF) * sh;
            let b = lambda * sh;
            term = -term * ((a * a + b * b) / k) / (ns + k + S::HALF);
            sum += term;
            absolute += term.abs();
            if term.abs() <= S::EPSILON * sum.abs() {
                complete = true;
                break;
            }
        }
        let value = prefactor * sum;
        let error = prefactor.abs()
            * (term.abs()
                + real::<S>(16.0)?
                    * S::EPSILON
                    * S::try_from_usize_exact(work.used + 1)?
                    * absolute);
        return Ok((
            value,
            error,
            complete,
            algorithm_component!("origin_series"),
        ));
    }
    if !work.step() {
        return Ok((
            S::ZERO,
            S::MAX,
            false,
            algorithm_component!("directional_recurrence"),
        ));
    }
    if !theta.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::LossOfSignificance,
            "lambda*eta exceeds the supported phase representation",
        ));
    }
    // Preserve the exact product phase as a high/low pair before reduction.
    let low = lambda.mul_add(eta, -theta);
    let sin = theta.sin() * low.cos() + theta.cos() * low.sin();
    let cos = theta.cos() * low.cos() - theta.sin() * low.sin();
    let inverse_sinh = if eta <= S::ONE {
        S::ONE / eta.sinh()
    } else {
        multiply_exp(real::<S>(2.0)?, -eta) / (-(-eta - eta).exp_m1())
    };
    let eta_over_sinh = if eta <= S::ONE {
        eta / eta.sinh()
    } else {
        multiply_exp(eta, real::<S>(2.0)?.ln() - eta) / (-(-eta - eta).exp_m1())
    };
    let coth = S::ONE / eta.tanh();
    let zero = if theta == S::ZERO {
        eta_over_sinh
    } else {
        (sin / theta) * eta_over_sinh
    };
    let one = (coth / lambda.hypot(S::ONE)) * zero - cos * (inverse_sinh / lambda.hypot(S::ONE));
    let zero_error = real::<S>(16.0)?
        * S::EPSILON
        * if theta == S::ZERO {
            eta_over_sinh.abs()
        } else {
            (eta_over_sinh / theta).abs()
        };
    let one_error = (coth / lambda.hypot(S::ONE)).abs() * zero_error
        + real::<S>(16.0)? * S::EPSILON * ((coth * zero).abs() + inverse_sinh.abs())
            / lambda.hypot(S::ONE);
    if n == 0 {
        return Ok((
            zero,
            zero_error,
            true,
            algorithm_component!("elementary_seed"),
        ));
    }
    if n == 1 {
        return Ok((
            one,
            one_error,
            true,
            algorithm_component!("elementary_seed"),
        ));
    }
    if ns <= lambda * eta.sinh() || eta > (ns + S::ONE).ln() + real(2.0)? {
        let mut previous = zero;
        let mut current = one;
        let mut previous_error = zero_error;
        let mut current_error = one_error;
        for j in 1..n {
            if !work.step() {
                return Ok((
                    S::ZERO,
                    S::MAX,
                    false,
                    algorithm_component!("upward_recurrence"),
                ));
            }
            let js = S::try_from_f64_exact(f64::from(j))?;
            let denominator = lambda.hypot(js + S::ONE);
            let a = (real::<S>(2.0)? * js + S::ONE) * (coth / denominator);
            let b = lambda.hypot(js) / denominator;
            let next = a * current - b * previous;
            let next_error = a.abs() * current_error
                + b.abs() * previous_error
                + real::<S>(16.0)? * S::EPSILON * ((a * current).abs() + (b * previous).abs());
            previous = current;
            current = next;
            previous_error = current_error;
            current_error = next_error;
        }
        return Ok((
            current,
            current_error,
            true,
            algorithm_component!("upward_recurrence"),
        ));
    }
    let margin = (real::<S>(32.0)? / eta.max(real(0.1)?)).ceil();
    if margin > S::try_from_usize_exact(work.remaining)? {
        return Ok((
            S::ZERO,
            S::MAX,
            false,
            algorithm_component!("miller_recurrence"),
        ));
    }
    let mut top = (n as usize).saturating_add(margin.try_to_usize_exact()?);
    let mut previous: Option<S> = None;
    loop {
        if top > work.remaining {
            return Ok((
                S::ZERO,
                S::MAX,
                false,
                algorithm_component!("miller_recurrence"),
            ));
        }
        let mut upper = S::ZERO;
        let mut current = S::ONE;
        let mut target = S::ZERO;
        let threshold = S::MAX.sqrt();
        for j in (1..=top).rev() {
            if !work.step() {
                return Ok((
                    S::ZERO,
                    S::MAX,
                    false,
                    algorithm_component!("miller_recurrence"),
                ));
            }
            let js = S::try_from_usize_exact(j)?;
            let a = (real::<S>(2.0)? * js + S::ONE) * (coth / lambda.hypot(js));
            let b = lambda.hypot(js + S::ONE) / lambda.hypot(js);
            if current.abs() > threshold / a.abs().max(S::ONE) {
                current /= threshold;
                upper /= threshold;
                target /= threshold;
            }
            if j == n as usize {
                target = current;
            }
            let lower = a * current - b * upper;
            upper = current;
            current = lower;
        }
        let value = if zero.abs() >= one.abs() {
            (target / current) * zero
        } else {
            (target / upper) * one
        };
        if let Some(old) = previous {
            let delta = (value - old).abs();
            if delta <= real::<S>(64.0)? * S::EPSILON * value.abs().max(S::MIN_POSITIVE) {
                let error = delta
                    + real::<S>(32.0)?
                        * S::EPSILON
                        * S::try_from_usize_exact(top + 1)?
                        * value.abs();
                return Ok((
                    value,
                    error,
                    true,
                    algorithm_component!("miller_recurrence"),
                ));
            }
        }
        previous = Some(value);
        top = top.saturating_add(24);
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Riccati–Bessel function `x j_n(x)` of nonnegative integer order `n`.

`j_n` is the spherical Bessel function of the first kind (DLMF 10.47.3), so `x j₀(x) = sin x`
and `x j₁(x) = sin x / x − cos x`. Every finite `x` is accepted. The function has parity
`(−1)^(n+1)` and vanishes at `x = 0`.

For `|x| ≤ 1` the function sums the power series (DLMF 10.53.1). Otherwise it starts from the
two closed forms above and recurs in the order (DLMF 10.51.1): upward for `n ≤ |x|`, and
with Miller's backward recurrence, normalized by the larger closed form, for `n > |x|`.

# Examples

```
// x j₀(x) = sin x
let s = scioxide_special::riccati_jn(0, 1.0_f64)?;
assert!((s - 1.0_f64.sin()).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`riccati_jn_with_options`](crate::riccati_jn_with_options) returns an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`SphericalBesselOptions`](crate::SphericalBesselOptions) (65536 units by default). The cost
  grows linearly with `n`.

# References

NIST DLMF, §10.47–10.53. <https://dlmf.nist.gov/10.47>"###)]

    pub riccati_jn [S: RealScalar] (order: u32, x: S) -> S
    where [];
    options options: SphericalBesselOptions;
    detailed riccati_jn_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate(Kind::J, false, true, order, x, options, context)
    }
    scoped pub riccati_jn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Riccati–Bessel function `x y_n(x)` of nonnegative integer order `n`.

`y_n` is the spherical Bessel function of the second kind (DLMF 10.47.4), so
`x y₀(x) = −cos x` and `x y₁(x) = −cos x / x − sin x`. Every finite nonzero `x` is accepted, and
the function has parity `(−1)^n`. At `x = 0`, order zero takes its limit `−1`; higher orders
have a pole.

The function starts from the two closed forms above and recurs upward in the order
(DLMF 10.51.1), which is stable for this solution.

# Examples

```
// x y₀(x) = −cos x
let c = scioxide_special::riccati_yn(0, 1.0_f64)?;
assert!((c + 1.0_f64.cos()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated. Tests compare values and error estimates with reference
values in `f32` and `f64`; near zeros they bound the absolute error.
[`riccati_yn_with_options`](crate::riccati_yn_with_options) returns an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0` for `n ≥ 1`.
- `NumericalInstability` (`Overflow`) if `|x y_n(x)|` exceeds the range of the scalar type,
  which happens for `n` large compared with `|x|`.
- `ConvergenceFailure` (`IterationLimit`) if the work exceeds the budget of
  [`SphericalBesselOptions`](crate::SphericalBesselOptions) (65536 units by default). The cost
  grows linearly with `n`.

# References

NIST DLMF, §10.47–10.53. <https://dlmf.nist.gov/10.47>"###)]

    pub riccati_yn [S: RealScalar] (order: u32, x: S) -> S
    where [];
    options options: SphericalBesselOptions;
    detailed riccati_yn_with_options -> Detailed<S>;
    evaluate |options, context| {
        evaluate(Kind::Y, false, true, order, x, options, context)
    }
    scoped pub riccati_yn_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
