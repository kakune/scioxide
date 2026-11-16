//! Real special integrals with explicit work and representation contracts.
//!
//! E1/Ei use scaled piecewise rational approximations; En uses a compensated
//! defining series or positive continued-fraction convergents. Si/Ci use central
//! and auxiliary rational expansions through 48, then relative asymptotic
//! expansions. Dense 160/240-digit reference values check an eight-ulp `f64` target
//! away from zeros, with absolute cancellation-aware bounds at zeros. Detailed
//! converged estimates are statistically calibrated; interrupted work keeps
//! conservative model bounds.

use crate::ValueWithError;
use crate::expansions::{chebyshev, real};
use crate::integrals::integral_coefficients as c;
use crate::integrals::real_coefficients as r;
use crate::result::{Evaluation, Forms, Publish};
use crate::support::INTEGRAL_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::wide::Wide;
use crate::{
    IntegralOptions, ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination,
};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, DomainReason, EstimateReliability, EvaluationCounts, LogValue,
    RealScalar, ScaledValue, SciOxideError, algorithm_component,
};

const FAMILY: AlgorithmComponent = algorithm_component!("special_integrals");
const EULER: f64 = 0.577_215_664_901_532_9;

type Detailed<S> = SpecialFunctionResult<S>;

fn table<S: RealScalar>(
    x: S,
    lo: f64,
    hi: f64,
    coefficients: &[f64],
    work: &mut Work,
) -> Result<(S, S), SciOxideError> {
    let t = ((x - real::<S>(lo)?) / real::<S>(hi - lo)?) * real::<S>(2.)? - S::ONE;
    chebyshev(t, coefficients, work)
}

fn project<S: RealScalar>(value: S, exponent: S) -> S {
    if exponent == S::ZERO || value == S::ZERO {
        return value;
    }
    let factor = exponent.exp();
    let direct = value * factor;
    if factor.is_normal() && direct.is_finite() && direct != S::ZERO {
        return direct;
    }
    let half = (exponent * S::HALF).exp();
    let direct = (value * half) * half;
    if direct.is_finite() && direct != S::ZERO {
        direct
    } else {
        S::ONE.copysign(value) * (value.abs().ln() + exponent).exp()
    }
}

/// A special-integral estimate `value·exp(exponent)`, the mathematical value, with its derivative.
/// Exponential factors are applied only at projection; tails never multiply an overflow by an
/// underflow.
#[derive(Clone, Copy)]
pub(crate) struct IntegralForm<S> {
    value: S,
    error: S,
    exponent: S,
    derivative: Option<S>,
    used: usize,
    complete: bool,
    qualified: bool,
}

/// Evaluation of one special integral.
pub(crate) type IntegralEvaluation<S> = Evaluation<S, IntegralForm<S>>;

#[inline]
fn finish<S: RealScalar>(
    value: S,
    error: S,
    exponent: S,
    derivative: Option<S>,
    work: &Work,
    method: AlgorithmComponent,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    let projected = project(value, exponent);
    // A zero bound on an unscaled value is an exact identity, such as Si(0).
    let termination = if !work.complete {
        SpecialFunctionTermination::IterationLimit
    } else if error == S::ZERO && exponent == S::ZERO {
        SpecialFunctionTermination::Exact
    } else {
        SpecialFunctionTermination::Converged
    };
    let representation = if !projected.is_finite() {
        ScalarRepresentation::Overflow
    } else {
        ScalarRepresentation::Finite
    };
    Ok(Evaluation::from_parts(
        projected.is_finite().then_some(projected),
        representation,
        termination,
        EvaluationCounts::from_iterations(work.used),
        AlgorithmId::from_components(FAMILY, method, 1),
        Some(IntegralForm {
            value,
            error,
            exponent,
            derivative,
            used: work.used,
            complete: work.complete,
            qualified: method == algorithm_component!("ei_piecewise_rational")
                || method == algorithm_component!("en_compensated_positive_fraction")
                || method == algorithm_component!("sici_auxiliary_rational"),
        }),
    ))
}

impl<S: RealScalar> Publish<S> for IntegralForm<S> {
    fn publish(self, ordinary: Option<S>) -> Result<Forms<S>, SciOxideError> {
        let Self {
            value,
            error,
            exponent,
            derivative,
            used,
            complete,
            qualified,
        } = self;
        // These kernels are qualified by paired-precision dense oracles.
        // Interrupted evaluations retain their conservative model bounds.
        let calibrated = qualified && complete;
        let eps = real::<S>(if calibrated { 2.0 } else { 32.0 })? * S::EPSILON;
        let sign = S::ONE.copysign(value);
        let log = if value == S::ZERO {
            None
        } else {
            Some(value.abs().ln() + exponent)
        };
        let projected = project(value, exponent);
        let floor = if used == 0 && value == S::ZERO && error == S::ZERO {
            S::ZERO
        } else {
            real::<S>(4.)? * S::ZERO.next_after(S::ONE)
        };
        let relative = if value == S::ZERO {
            S::ZERO
        } else {
            ((error + floor) / value.abs() + eps).min(S::MAX)
        };
        let absolute = if exponent == S::ZERO {
            error + eps * value.abs() + floor
        } else if value == S::ZERO {
            project(error + floor, exponent).abs().min(S::MAX) + floor
        } else {
            (relative * projected.abs()).min(S::MAX) + floor
        }
        .min(S::MAX);
        let reliability = if calibrated {
            EstimateReliability::StatisticallyCalibrated
        } else {
            EstimateReliability::ModelBased
        };
        let ordinary = ordinary
            .map(|ordinary| {
                ValueWithError::new(
                    ordinary,
                    absolute.min(S::MAX),
                    (ordinary != S::ZERO).then(|| (absolute / ordinary.abs()).min(S::MAX)),
                    reliability,
                )
            })
            .transpose()?;
        let log_error = if relative < S::HALF {
            (crate::result::log_error(relative, log.unwrap_or(S::ZERO)).unwrap_or(S::MAX)
                + (eps - S::EPSILON) * log.unwrap_or(S::ZERO).abs())
            .min(S::MAX)
        } else {
            S::MAX
        };
        let logarithmic = log
            .filter(|_| relative < S::HALF)
            .map(|l| {
                ValueWithError::new(LogValue::from_log(sign, l)?, log_error, None, reliability)
            })
            .transpose()?;
        // An interrupted evaluation's uncertainty can exceed the scaled coordinates of its
        // partial value; that form is then omitted rather than failing the whole report.
        let scaled_estimate = |scaled: ScaledValue<S>, relative: Option<S>| match scaled
            .scale_absolute_error(absolute)
        {
            Ok(error) => ValueWithError::new(scaled, error, relative, reliability).map(Some),
            Err(_) if !complete => Ok(None),
            Err(error) => Err(error),
        };
        let scaled = if exponent == S::ZERO {
            scaled_estimate(
                ScaledValue::from_value(value)?,
                (value != S::ZERO).then_some(relative),
            )?
        } else if let Some(l) = log {
            if l.abs() < real::<S>(6e18)? && log_error < S::HALF {
                let scale_error = log_error.exp_m1() + eps;
                let scaled = ScaledValue::from_log(sign, l)?;
                Some(crate::result::scaled_estimate(
                    scaled,
                    scale_error,
                    reliability,
                )?)
            } else {
                None
            }
        } else {
            scaled_estimate(ScaledValue::from_value(value)?, None)?
        };
        let derivative = derivative
            .filter(|d| complete && d.is_finite())
            .map(|d| {
                ValueWithError::new(
                    d,
                    (eps * d.abs() + real::<S>(16.)? * absolute + floor).min(S::MAX),
                    None,
                    EstimateReliability::ModelBased,
                )
            })
            .transpose()?;
        Ok(Forms {
            value: ordinary,
            scaled,
            logarithmic,
            derivative,
        })
    }
}

// The positive convergents of DLMF 8.19.17 avoid the cancellation in the
// modified Lentz denominators. Rescale all four states by the same power of two.
fn en<S: RealScalar>(n: u32, x: S, work: &mut Work) -> Result<(S, S), SciOxideError> {
    let (value, error) = en_f64(n, x.into_f64(), work);
    Ok((real(value)?, real(error.min(S::MAX.into_f64()))?))
}

fn en_f64(n: u32, x: f64, work: &mut Work) -> (f64, f64) {
    if x == 0.0 {
        let v = 1.0 / f64::from(n - 1);
        return (v, f64::EPSILON * v);
    }
    if n == 0 {
        let v = 1.0 / x;
        return (v, f64::EPSILON * v);
    }
    if n == 1 {
        return e1(x, work);
    }
    let nf = f64::from(n);
    if x <= 1.0 {
        let mut harmonic = 0.0;
        for k in 1..if n <= 32 { n } else { 1 } {
            if !work.step() {
                return (0.0, f64::MAX);
            }
            harmonic += 1.0 / f64::from(k);
        }
        let mut sum = Wide::from(1.0 / (nf - 1.0));
        let mut fact = 1.0;
        let mut last = 1.0;
        let mut i = 0usize;
        while work.step() {
            i += 1;
            let k = i as f64;
            fact *= -x / k;
            let delta = if i == n as usize - 1 {
                fact * (harmonic - x.ln() - EULER)
            } else {
                -fact / (k - nf + 1.0)
            };
            sum = sum + Wide::from(delta);
            last = delta.abs();
            if (i > n as usize || n > 32) && last <= 0.25 * f64::EPSILON * sum.value().abs() {
                break;
            }
        }
        let v = sum.value() * x.exp();
        return (v, last * x.exp() + 4.0 * f64::EPSILON * v.abs());
    }
    if x <= 2.0 && n <= 32 {
        // Upward order recurrence is well conditioned in this band. FMA
        // preserves the subtraction, and each successive order damps error.
        let mut value = e1(x, work).0;
        for order in 2..=n {
            if !work.step() {
                return (value, f64::MAX);
            }
            value = (-x).mul_add(value, 1.0) / f64::from(order - 1);
        }
        return (value, 6.0 * f64::EPSILON * value.abs());
    }
    if x <= 8.0 && n <= 32 {
        // The defining series converges much faster than the continued
        // fraction here. Two terms retain cancellation through exp(x) scaling.
        let mut harmonic = Wide::ZERO;
        for k in 1..n {
            if !work.step() {
                return (0.0, f64::MAX);
            }
            harmonic = harmonic + Wide::from(1.0).div_f64(f64::from(k));
        }
        let logarithm = Wide::from(x).ln();
        let constant = harmonic - logarithm - crate::support::constants::EULER_GAMMA;
        let mut sum = Wide::from(1.0).div_f64(nf - 1.0);
        let mut factor = Wide::from(1.0);
        let mut i = 0usize;
        let mut last = 1.0;
        while work.step() {
            i += 1;
            factor = factor.mul_f64(-x).div_f64(i as f64);
            let delta = if i == n as usize - 1 {
                factor * constant
            } else {
                factor.neg().div_f64(i as f64 - nf + 1.0)
            };
            sum = sum + delta;
            last = delta.hi.abs();
            if i > n as usize && last <= 0.125 * f64::EPSILON * sum.hi.abs() {
                break;
            }
        }
        let value = sum.value() * x.exp();
        return (value, last * x.exp() + 3.0 * f64::EPSILON * value.abs());
    }
    // The first convergent includes the order. Its relative correction is at
    // most n/(x+n)^2; multiply reciprocals to avoid overflowing the square.
    let inverse = 1.0 / (x + nf);
    let correction = (nf * inverse) * inverse;
    if correction < f64::EPSILON {
        return (inverse, (correction + 2.0 * f64::EPSILON) * inverse);
    }
    let (mut p0, mut q0) = (1.0, x);
    let (mut p1, mut q1) = (1.0, x + nf);
    let mut value = p1 / q1;
    let mut change = value;
    let mut k = 1usize;
    while work.step() {
        k += 1;
        let (a, b) = if k & 1 == 0 {
            ((k / 2) as f64, x)
        } else {
            (nf + ((k - 1) / 2) as f64, 1.0)
        };
        let p = p1.mul_add(b, p0 * a);
        let q = q1.mul_add(b, q0 * a);
        let next = p / q;
        change = (next - value).abs();
        (p0, p1, q0, q1) = (p1, p, q1, q);
        value = next;
        if change <= 0.5 * f64::EPSILON * value.abs() {
            break;
        }
        if q1 > 2.0_f64.powi(500) {
            // Common exact binary rescaling preserves the convergent.
            let scale = 2.0_f64.powi(-512);
            p0 *= scale;
            p1 *= scale;
            q0 *= scale;
            q1 *= scale;
        }
    }
    (value, change + 6.0 * f64::EPSILON * value.abs())
}

// Coefficients are in ascending monomial order. Fixed-cost approximations
// consume one work unit per coefficient, including the constant coefficients.
fn rational(x: f64, p: &[f64], q: &[f64], work: &mut Work) -> f64 {
    if !work.admit(p.len() + q.len()) {
        return 0.0;
    }
    let polynomial = |c: &[f64]| {
        crate::support::polynomial::polynomial(
            x,
            c,
            crate::support::polynomial::PolynomialPolicy::Fused,
        )
    };
    polynomial(p) / polynomial(q)
}

// Boost.Math's binary64 E1 approximation, kept scaled throughout the tail.
fn e1(x: f64, work: &mut Work) -> (f64, f64) {
    let value = if x <= 1.0 {
        let correction = rational(x, r::E1_0_P, r::E1_0_Q, work);
        (Wide::from(x) - Wide::from(x.ln()) - Wide::from(0.6637353897094727)
            + Wide::from(correction))
        .value()
            * x.exp()
    } else {
        let inv = 1.0 / x;
        (1.0 + rational(inv, r::E1_1_P, r::E1_1_Q, work)) * inv
    };
    (
        value,
        if work.complete {
            4.0 * f64::EPSILON * value.abs()
        } else {
            f64::MAX
        },
    )
}

fn ei<S: RealScalar>(x: S, work: &mut Work) -> Result<(S, S, S), SciOxideError> {
    let x = x.into_f64();
    let (value, error, exponent) = if x < 0.0 {
        let (v, e) = e1(-x, work);
        (-v, e, x)
    } else if x <= 6.0 {
        let t = x / 3.0 - 1.0;
        let ratio = rational(t, r::EI_0_P, r::EI_0_Q, work);
        let root_hi = 1677624236387711.0 / 4503599627370496.0;
        let root_lo = 1.3140183414386028e-17;
        let delta = Wide::from(x)
            - Wide {
                hi: root_hi,
                lo: root_lo,
            };
        let logarithm = if delta.hi.abs() < 0.1 {
            (delta.value() / 0.3725074107813666).ln_1p()
        } else {
            x.ln() - 0.3725074107813666_f64.ln()
        };
        let value = (delta.mul_f64(ratio) + Wide::from(logarithm)).value();
        if x <= 1.0 {
            (value, 6.0 * f64::EPSILON * value.abs(), 0.0)
        } else {
            let value = value * (-x).exp();
            (value, 6.0 * f64::EPSILON * value.abs(), x)
        }
    } else {
        let (t, y, p, q) = if x <= 10.0 {
            (x / 2.0 - 4.0, 1.1589851379394531, r::EI_1_P, r::EI_1_Q)
        } else if x <= 20.0 {
            (x / 5.0 - 3.0, 1.0869731903076172, r::EI_2_P, r::EI_2_Q)
        } else if x <= 40.0 {
            (x / 10.0 - 3.0, 1.0393753051757813, r::EI_3_P, r::EI_3_Q)
        } else {
            (1.0 / x, 1.0130653381347656, r::EI_4_P, r::EI_4_Q)
        };
        let value = (y + rational(t, p, q, work)) / x + x * (-x).exp();
        (value, 5.0 * f64::EPSILON * value.abs(), x)
    };
    Ok((
        real(value)?,
        real(if work.complete {
            error.min(S::MAX.into_f64())
        } else {
            S::MAX.into_f64()
        })?,
        real(exponent)?,
    ))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the generalized exponential integral E_n(x) for integer order `n ≥ 0`.

```text
E_n(x) = ∫₁^∞ exp(−xt) / tⁿ dt
```

(DLMF §8.19). For `n ≥ 2` the domain is `x ≥ 0` and `E_n(0) = 1/(n − 1)`. `E₀(x) = exp(−x)/x`
and `E₁` have a pole at zero; for `n = 1` negative `x` is accepted and returns the real
principal value `−Ei(−x)`, as [`expint_e1`](crate::expint_e1) does.

For `n ≥ 2` the method uses a compensated power series for `0 < x ≤ 1`. For `n ≤ 32`,
it uses upward order recurrence from `E₁` for `1 < x ≤ 2` and a two-word defining series for
`2 < x ≤ 8`. Remaining positive arguments use a continued fraction. These methods evaluate
the scaled form `exp(x) E_n(x)`. `n = 1` uses the rational approximations of
[`expint_e1`](crate::expint_e1).

# Examples

```
// E_n(0) = 1/(n − 1)
assert!((scioxide_special::expint_en(3, 0.0_f64)? - 0.5).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0` and `n ≤ 1`.
- `DomainError` (`OutsideDomain`) if `x < 0` and `n ≠ 1`.
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type (`n = 0`
  with tiny `x`, or `n = 1` with large negative `x`); the detailed form still returns the value in
  scaled and logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §8.19. <https://dlmf.nist.gov/8.19>"###)]

    pub expint_en [S: RealScalar] (n: u32, x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_en_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();

        en_evaluate(n, x, options, false, false, None)?.value()
    }
    report {
        en_evaluate(n, x, options, false, true, None)?.publish()
    }
    scoped pub expint_en_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            en_evaluate(n, x, options, false, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled generalized exponential integral exp(x) E_n(x).

`E_n` is defined in [`expint_en`](crate::expint_en), whose orders, domain, poles and limits
apply. The scaled function behaves like `1/(x + n)` for large `x`, where `E_n(x)` itself
underflows; the exponential is absorbed by the kernel rather than applied to an underflowed
value.

# Examples

```
// exp(1) E₃(1) = 0.29817368116159704
let v = scioxide_special::expint_en_scaled(3, 1.0_f64)?;
assert!((v - 0.298_173_681_161_597_04).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target for the scaled form; the detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0` and `n ≤ 1`.
- `DomainError` (`OutsideDomain`) if `x < 0` and `n ≠ 1`.
- `NumericalInstability` (`Overflow`) if `n = 0` and `exp(x) E₀(x) = 1/x` exceeds the range of the
  scalar type.

# References

- NIST Digital Library of Mathematical Functions, §8.19. <https://dlmf.nist.gov/8.19>"###)]

    pub expint_en_scaled [S: RealScalar] (n: u32, x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_en_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();

        en_evaluate(n, x, options, true, false, None)?.value()
    }
    report {
        en_evaluate(n, x, options, true, true, None)?.publish()
    }
    scoped pub expint_en_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            en_evaluate(n, x, options, true, true, Some(context))
        })?
        .publish()
    }
}

fn en_evaluate<S: RealScalar>(
    n: u32,
    x: S,
    options: IntegralOptions,
    scaled: bool,
    slope: bool,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    en_report(n, x, scaled, slope, &mut work)
}
/// `E_n(x)` or `exp(x) E_n(x)` charged to the caller's allowance.
fn en_report<S: RealScalar>(
    n: u32,
    x: S,
    scaled: bool,
    slope: bool,
    work: &mut Work,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    crate::support::require_finite_param(x, "integral parameter x must be finite")?;
    if x < S::ZERO {
        if n == 1 {
            return evaluate_ei(if scaled { Kind::E1Scaled } else { Kind::E1 }, x, work);
        }
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::OutsideDomain));
    }
    if x == S::ZERO && n < 2 {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::Pole));
    }
    if n == 0 && !(S::ONE / x).is_finite() {
        return finish(
            S::ONE,
            S::EPSILON,
            -x.ln() - if scaled { S::ZERO } else { x },
            None,
            work,
            algorithm_component!("en_zero_logarithm"),
        );
    }
    let (v, e) = en(n, x, work)?;
    let derivative = if !slope {
        None
    } else if x == S::ZERO {
        if n > 2 {
            Some(-S::ONE / real::<S>(f64::from(n - 2))? + if scaled { v } else { S::ZERO })
        } else {
            None
        }
    } else if n == 0 {
        Some(if scaled {
            -v / x
        } else {
            (-x).exp() * (-v - v / x)
        })
    } else if n == 1 {
        Some(if scaled {
            v - S::ONE / x
        } else {
            project(-S::ONE / x, -x)
        })
    } else {
        let previous = en(n - 1, x, work)?.0;
        Some(if scaled {
            v - previous
        } else {
            -previous * (-x).exp()
        })
    };
    work.check()?;
    finish(
        v,
        e,
        if scaled { S::ZERO } else { -x },
        derivative,
        work,
        algorithm_component!("en_compensated_positive_fraction"),
    )
}

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Ei,
    EiScaled,
    E1,
    E1Scaled,
    E2,
    E2Scaled,
    Si,
    Ci,
    Shi,
    Chi,
    ShiScaled,
    ChiScaled,
    FresnelC,
    FresnelS,
    Atanint,
    Expint3,
    Li,
    LiOffset,
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponential integral Ei(x).

```text
Ei(x) = PV ∫₋∞ˣ exp(t)/t dt
```

(DLMF §6.2), a Cauchy principal value for `x > 0`. Ei has a logarithmic pole at zero, a single
real zero near `x = 0.3725`, and grows like `exp(x)/x`. For `x < 0`, `Ei(x) = −E₁(−x)`.

The method is a set of piecewise rational approximations evaluated in scaled form, so the
exponential factor is applied once at the end.

# Examples

```
// Ei(1) = 1.8951178163559368
let v = scioxide_special::expint_ei(1.0_f64)?;
assert!((v - 1.895_117_816_355_936_8).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type (about `x >
  716` in `f64`); the detailed form still returns the value in scaled and logarithmic form.
  [`expint_ei_scaled`](crate::expint_ei_scaled) stays in range.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub expint_ei [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_ei_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Ei, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Ei, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled exponential integral exp(−x) Ei(x).

Ei is defined in [`expint_ei`](crate::expint_ei), whose domain and pole apply. The scaled
function behaves like `1/x` for large `|x|`, so it stays in range where `Ei(x)` overflows.

# Examples

```
// exp(−1) Ei(1) = 0.697174883235066
let v = scioxide_special::expint_ei_scaled(1.0_f64)?;
assert!((v - 0.697_174_883_235_066).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target for the scaled form; the detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub expint_ei_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_ei_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::EiScaled, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::EiScaled, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponential integral E₁(x).

```text
E₁(x) = ∫ₓ^∞ exp(−t)/t dt,   x > 0
```

(DLMF 6.2.1). For `x < 0` the result is the real principal value `−Ei(−x)`, the real part of
E₁ on either side of its branch cut, not a complex value. E₁ has a logarithmic pole at zero.

The method is a set of piecewise rational approximations evaluated in scaled form, so the
exponential factor is applied once at the end.

# Examples

```
// E₁(1) = 0.21938393439552029
let v = scioxide_special::expint_e1(1.0_f64)?;
assert!((v - 0.219_383_934_395_520_29).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type (about `x <
  −716` in `f64`); the detailed form still returns the value in scaled and logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub expint_e1 [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_e1_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::E1, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::E1, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled exponential integral exp(x) E₁(x).

E₁ is defined in [`expint_e1`](crate::expint_e1), whose domain and pole apply. The scaled
function behaves like `1/x` for large `|x|`, so it stays in range where `E₁(x)` underflows or,
for negative `x`, overflows.

# Examples

```
// scaled and unscaled forms agree: exp(x) E₁(x)
let scaled = scioxide_special::expint_e1_scaled(1.0_f64)?;
let expected = 1.0_f64.exp() * scioxide_special::expint_e1(1.0_f64)?;
assert!((scaled - expected).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target for the scaled form; the detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub expint_e1_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_e1_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::E1Scaled, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::E1Scaled, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the exponential integral E₂(x).

```text
E₂(x) = ∫₁^∞ exp(−xt)/t² dt,   x ≥ 0,
```

the case `n = 2` of [`expint_en`](crate::expint_en) (DLMF §8.19). `E₂(0) = 1`, and
`E₂(x) = exp(−x) − x E₁(x)`.

# Examples

```
// E₂(1) = exp(−1) − E₁(1)
let expected = (-1.0_f64).exp() - scioxide_special::expint_e1(1.0_f64)?;
assert!((scioxide_special::expint_e2(1.0_f64)? - expected).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §8.19. <https://dlmf.nist.gov/8.19>"###)]

    pub expint_e2 [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_e2_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::E2, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::E2, x, options, true, None)?.publish()
    }
    scoped pub expint_e2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::E2, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled exponential integral exp(x) E₂(x).

E₂ is defined in [`expint_e2`](crate::expint_e2), whose domain applies. The scaled function
behaves like `1/(x + 2)` for large `x`, where `E₂(x)` itself underflows.

# Examples

```
// exp(1) E₂(1) = 0.4036526376768059
let v = scioxide_special::expint_e2_scaled(1.0_f64)?;
assert!((v - 0.403_652_637_676_805_9).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No separate ulp target for the scaled form; the detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §8.19. <https://dlmf.nist.gov/8.19>"###)]

    pub expint_e2_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_e2_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::E2Scaled, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::E2Scaled, x, options, true, None)?.publish()
    }
    scoped pub expint_e2_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::E2Scaled, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the sine integral Si(x).

```text
Si(x) = ∫₀ˣ sin(t)/t dt
```

(DLMF 6.2.9). Si is entire and odd, with `Si(0) = 0` and `Si(x) → π/2` as `x → +∞`.

For `|x| ≤ 4` the method uses Chebyshev expansions in `t = (x² − 8)/8`. For
`4 < |x| < 48` it uses rational approximations of the auxiliary functions `f` and `g`;
for `|x| ≥ 48` it uses their asymptotic expansions.

# Examples

```
// Si(1) = 0.946083070367183
let v = scioxide_special::si(1.0_f64)?;
assert!((v - 0.946_083_070_367_183).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub si [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed si_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Si, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Si, x, options, true, None)?.publish()
    }
    scoped pub si_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::Si, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the cosine integral Ci(x).

```text
Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt,   x > 0,
```

where `γ` is the Euler–Mascheroni constant (DLMF §6.2). Ci has a logarithmic pole at zero
and `Ci(x) → 0` as `x → +∞`, oscillating about zero with infinitely many real zeros, the
first near `x = 0.6165`.

For `|x| ≤ 4` the method uses Chebyshev expansions in `t = (x² − 8)/8`. For
`4 < |x| < 48` it uses rational approximations of the auxiliary functions `f` and `g`;
for `|x| ≥ 48` it uses their asymptotic expansions.

# Examples

```
// Ci(1) = 0.33740392290096816
let v = scioxide_special::ci(1.0_f64)?;
assert!((v - 0.337_403_922_900_968_16).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub ci [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed ci_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Ci, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Ci, x, options, true, None)?.publish()
    }
    scoped pub ci_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::Ci, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the hyperbolic sine integral Shi(x).

```text
Shi(x) = ∫₀ˣ sinh(t)/t dt = (Ei(x) + E₁(x))/2,   x > 0
```

(DLMF §6.2). Shi is entire and odd, with `Shi(0) = 0`, and grows like `exp(|x|)/(2|x|)`.

For `|x| ≤ 1` the method is a power series; beyond, it combines the scaled kernels of
[`expint_ei`](crate::expint_ei) and [`expint_e1`](crate::expint_e1).

# Examples

```
// Shi(1) = 1.0572508753757285
let v = scioxide_special::shi(1.0_f64)?;
assert!((v - 1.057_250_875_375_728_5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type (about `|x|
  > 717` in `f64`); the detailed form still returns the value in scaled and logarithmic form.
  [`shi_scaled`](crate::shi_scaled) stays in range.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub shi [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed shi_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Shi, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Shi, x, options, true, None)?.publish()
    }
    scoped pub shi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::Shi, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the hyperbolic cosine integral Chi(x).

```text
Chi(x) = γ + ln x + ∫₀ˣ (cosh t − 1)/t dt = (Ei(x) − E₁(x))/2,   x > 0,
```

where `γ` is the Euler–Mascheroni constant (DLMF §6.2). Chi has a logarithmic pole at zero
and grows like `exp(x)/(2x)`.

For `|x| ≤ 1` the method is a power series; beyond, it combines the scaled kernels of
[`expint_ei`](crate::expint_ei) and [`expint_e1`](crate::expint_e1).

# Examples

```
// Shi(x) + Chi(x) = Ei(x)
let sum = scioxide_special::shi(1.0_f64)? + scioxide_special::chi(1.0_f64)?;
assert!((sum - scioxide_special::expint_ei(1.0_f64)?).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.
- `DomainError` (`OutsideDomain`) if `x < 0`.
- `NumericalInstability` (`Overflow`) if the value exceeds the range of the scalar type (about `x >
  717` in `f64`); the detailed form still returns the value in scaled and logarithmic form.
  [`chi_scaled`](crate::chi_scaled) stays in range.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub chi [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed chi_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Chi, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Chi, x, options, true, None)?.publish()
    }
    scoped pub chi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::Chi, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled hyperbolic sine integral exp(−|x|) Shi(x).

Shi is defined in [`shi`](crate::shi). The scaled function tends to `±1/(2|x|)` for large `|x|`,
so it stays in range where `Shi(x)` overflows.

# Examples

```
// exp(−1) Shi(1) = 0.3889408612112412
let v = scioxide_special::shi_scaled(1.0_f64)?;
assert!((v - 0.388_940_861_211_241_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub shi_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed shi_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::ShiScaled, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::ShiScaled, x, options, true, None)?.publish()
    }
    scoped pub shi_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::ShiScaled, x, options, true, Some(context))
        })?
        .publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled hyperbolic cosine integral exp(−x) Chi(x).

Chi is defined in [`chi`](crate::chi), whose domain and pole apply. The scaled function tends
to `1/(2x)` for large `x`, so it stays in range where `Chi(x)` overflows.

# Examples

```
// exp(−1) Chi(1) = 0.3082340220238248
let v = scioxide_special::chi_scaled(1.0_f64)?;
assert!((v - 0.308_234_022_023_824_8).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 0`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub chi_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed chi_scaled_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::ChiScaled, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::ChiScaled, x, options, true, None)?.publish()
    }
    scoped pub chi_scaled_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            evaluate(Kind::ChiScaled, x, options, true, Some(context))
        })?
        .publish()
    }
}

pub(crate) fn fresnel_c<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options: IntegralOptions = IntegralOptions::default();
    evaluate(Kind::FresnelC, x, options, false, None)?.value()
}

#[inline]
pub(crate) fn fresnel_c_evaluation<S: RealScalar>(
    x: S,
    options: IntegralOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::FresnelC, x, options, true, context)
}

pub(crate) fn fresnel_s<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options: IntegralOptions = IntegralOptions::default();
    evaluate(Kind::FresnelS, x, options, false, None)?.value()
}

#[inline]
pub(crate) fn fresnel_s_evaluation<S: RealScalar>(
    x: S,
    options: IntegralOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<S, impl crate::result::Publish<S> + use<S>>,
    scioxide_core::SciOxideError,
> {
    let _ = &context;
    evaluate(Kind::FresnelS, x, options, true, context)
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse tangent integral Ti₂(x).

```text
Ti₂(x) = ∫₀ˣ atan(t)/t dt
```

Ti₂ is odd, with `Ti₂(0) = 0` and `Ti₂(1) = G`, Catalan's constant. For `x > 1` it uses
`Ti₂(x) = Ti₂(1/x) + (π/2) ln x`, so the method is a Chebyshev expansion on `[0, 1]` plus that
reflection.

# Examples

```
// Ti₂(1) = G = 0.915965594177219 (Catalan's constant)
let v = scioxide_special::atanint(1.0_f64)?;
assert!((v - 0.915_965_594_177_219).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- L. Lewin, *Polylogarithms and Associated Functions*, North-Holland, 1981, ch. 2."###)]

    pub atanint [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed atanint_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Atanint, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Atanint, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the cubic exponential integral ∫₀ˣ exp(−t³) dt.

```text
∫₀ˣ exp(−t³) dt = γ(1/3, x³)/3,   x ≥ 0,
```

where `γ(a, z)` is the lower incomplete gamma function. The value is zero at the origin and
tends to `Γ(4/3) = Γ(1/3)/3` as `x → ∞`.

For `x ≤ 4` the method is a set of Chebyshev expansions on unit intervals. Beyond 4 the
result is the rounded limit `Γ(4/3)`, and the omitted tail, below `exp(−x³)/(3x²)`, is included
in the error estimate.

# Examples

```
// the limit Γ(4/3) = 0.8929795115692492
let v = scioxide_special::expint_3(10.0_f64)?;
assert!((v - 0.892_979_511_569_249_2).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
For real `x` the error is within 8 ulps for `f64` on the tested range, away from zeros of
the function; near a zero the error is bounded in absolute terms. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §8.2. <https://dlmf.nist.gov/8.2>"###)]

    pub expint_3 [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed expint_3_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Expint3, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Expint3, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the logarithmic integral li(x).

```text
li(x) = PV ∫₀ˣ dt / ln t = Ei(ln x),   x ≥ 0,
```

(DLMF §6.2), a Cauchy principal value for `x > 1`. `li(0) = 0`, li has a logarithmic pole
at `x = 1`, and its only zero is near `x = 1.4514`. The value is computed from the
[`expint_ei`](crate::expint_ei) kernel, with `ln x` formed through `ln1p` near one.

# Examples

```
// li(2) = 1.045163780117493
let v = scioxide_special::logarithmic_integral(2.0_f64)?;
assert!((v - 1.045_163_780_117_493).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 1`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub logarithmic_integral [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed logarithmic_integral_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::Li, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::Li, x, options, true, None)?.publish()
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the offset logarithmic integral Li(x) = li(x) − li(2).

li is defined in [`logarithmic_integral`](crate::logarithmic_integral), whose domain and pole
apply. `Li(2) = 0`; on `[1.5, 2.5]` a Chebyshev expansion about `x = 2` avoids the cancellation
in `li(x) − li(2)`.

# Examples

```
// Li(x) = li(x) − li(2)
let li = |x| scioxide_special::logarithmic_integral(x);
let v = scioxide_special::logarithmic_integral_offset(10.0_f64)?;
assert!((v - (li(10.0)? - li(2.0)?)).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds in `f32` and `f64`.
See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`Pole`) if `x = 1`.
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- NIST Digital Library of Mathematical Functions, §6.2. <https://dlmf.nist.gov/6.2>"###)]

    pub logarithmic_integral_offset [S: RealScalar] (x: S) -> S
    where [];
    options options: IntegralOptions;
    detailed logarithmic_integral_offset_with_options -> Detailed<S>;
    value {
        let options: IntegralOptions = IntegralOptions::default();
        evaluate(Kind::LiOffset, x, options, false, None)?.value()
    }
    report {
        evaluate(Kind::LiOffset, x, options, true, None)?.publish()
    }
}

// Half-weight Clenshaw, with FMA and a conservative interrupted-table bound.
fn real_chebyshev(x: f64, c: &[f64], work: &mut Work) -> (f64, f64) {
    match crate::support::polynomial::chebyshev(
        x,
        c,
        true,
        true,
        Ok::<_, core::convert::Infallible>,
        work,
    ) {
        Ok(result) => result,
        Err(never) => match never {},
    }
}

// Origin series needed by Shi/Chi only; oscillatory functions have their own kernel.
fn hyperbolic_origin<S: RealScalar>(
    x: S,
    cosine: bool,
    work: &mut Work,
) -> Result<(S, S), SciOxideError> {
    let xx = x * x;
    let mut term = if cosine { xx * S::HALF } else { x };
    let mut k = if cosine { 2usize } else { 1 };
    let mut sum = if cosine {
        real::<S>(EULER)? + x.ln()
    } else {
        S::ZERO
    };
    let mut round = sum.abs();
    let mut error = term.abs();
    while work.step() {
        let delta = term / real::<S>(k as f64)?;
        sum += delta;
        round += delta.abs();
        error = delta.abs();
        if error <= S::EPSILON * sum.abs() {
            break;
        }
        term *= xx / (real::<S>((k + 1) as f64)? * real::<S>((k + 2) as f64)?);
        k += 2;
    }
    Ok((sum, error + real::<S>(8.)? * S::EPSILON * round))
}

fn trigonometric<S: RealScalar>(
    x: S,
    cosine: bool,
    work: &mut Work,
) -> Result<(S, S), SciOxideError> {
    let x = x.into_f64();
    let (value, error) = if x <= 4.0 {
        let t = (x * x - 8.0) * 0.125;
        if cosine {
            let logarithm = if x >= 2.0 {
                Wide::from((0.5 * x).ln()) + crate::support::constants::LN_2
            } else {
                Wide::from(x.ln())
            };
            let (v, error) = real_chebyshev(t, r::CI, work);
            let mut value = (logarithm - Wide::from(0.5) + Wide::from(v)).value();
            if value.abs() < 0.25 && work.complete {
                // A nearby exact center keeps the elementary logarithm's
                // absolute rounding small. Preserve the division residual.
                let (center, log_center) = if x < 1.0 {
                    (0.75, crate::support::constants::LN_THREE_QUARTERS)
                } else {
                    (3.0, crate::support::constants::LN_3)
                };
                let displacement = (Wide::from(x) - Wide::from(center)).div_f64(center);
                let logarithm = log_center
                    + Wide::from(displacement.hi.ln_1p())
                    + Wide::from(displacement.lo / (1.0 + displacement.hi));
                let coordinate = (Wide::from(x).mul_f64(x) - Wide::from(8.0)).mul_f64(0.125);
                let (mut a, mut b) = (Wide::ZERO, Wide::ZERO);
                for (&hi, &lo) in r::CI[1..].iter().zip(&r::CI_LOW[1..]).rev() {
                    if !work.step() {
                        break;
                    }
                    (a, b) = (coordinate.mul_f64(2.0) * a - b + Wide { hi, lo }, a);
                }
                value = (logarithm - Wide::from(0.5) + coordinate * a - b
                    + Wide {
                        hi: r::CI[0] * 0.5,
                        lo: r::CI_LOW[0] * 0.5,
                    })
                .value();
            }
            (
                value,
                error + 4.0 * f64::EPSILON * (logarithm.value().abs() + 0.5),
            )
        } else {
            let (v, error) = real_chebyshev(t, r::SI, work);
            (x * (0.75 + v), x * error + 4.0 * f64::EPSILON * x)
        }
    } else {
        let inv = 1.0 / x;
        let square = inv * inv;
        let (f, g, fe, ge) = if x >= 48.0 {
            // The least asymptotic term is below binary64 rounding here.
            // Relative stopping is reached before the factorial terms grow.
            let (mut f, mut g, mut ft, mut gt) = (1.0, 1.0, 1.0, 1.0);
            let mut last = 1.0;
            let mut k = 0usize;
            while work.step() {
                k += 1;
                let a = (2 * k) as f64;
                ft *= -(a - 1.0) * a * square;
                gt *= -a * (a + 1.0) * square;
                f += ft;
                g += gt;
                last = ft.abs().max(gt.abs());
                if last <= 0.25 * f64::EPSILON {
                    break;
                }
            }
            (
                f * inv,
                (g * inv) * inv,
                last + 4.0 * f64::EPSILON,
                last + 4.0 * f64::EPSILON,
            )
        } else {
            let (fp, fq, gp, gq) = if x < 8.0 {
                (r::FN4, r::FD4, r::GN4, r::GD4)
            } else {
                (r::FN8, r::FD8, r::GN8, r::GD8)
            };
            let f = rational(square, fp, fq, work) * inv;
            let g = (rational(square, gp, gq, work) * inv) * inv;
            (f, g, 4.0 * f64::EPSILON, 4.0 * f64::EPSILON)
        };
        let (sin, cos) = x.sin_cos();
        let value = if cosine {
            (Wide::from(f).mul_f64(sin) - Wide::from(g).mul_f64(cos)).value()
        } else {
            (-f).mul_add(cos, (-g).mul_add(sin, core::f64::consts::FRAC_PI_2))
        };
        (
            value,
            fe * inv
                + (ge * inv) * inv
                + 8.0 * f64::EPSILON * (if cosine { inv } else { value.abs() }),
        )
    };
    Ok((
        real(value)?,
        real(if work.complete {
            error
        } else {
            S::MAX.into_f64()
        })?,
    ))
}

// Exact-product decomposition retains the low part of x²/2. Beyond 2^p every
// representable input is an even integer and this normalized phase is exact.
fn fresnel_phase<S: RealScalar>(x: S) -> Result<(S, S), SciOxideError> {
    if x.abs() >= real::<S>(2f64.powi(S::MANTISSA_DIGITS as i32))? {
        return Ok((S::ZERO, S::ONE));
    }
    let half = x * S::HALF;
    let high = x * half;
    let low = x.mul_add(half, -high);
    let (sh, ch) = scioxide_core::sin_cos_pi(high)?;
    let (sl, cl) = scioxide_core::sin_cos_pi(low)?;
    Ok((sh * cl + ch * sl, ch * cl - sh * sl))
}
fn fresnel<S: RealScalar>(x: S, sine: bool, work: &mut Work) -> Result<(S, S), SciOxideError> {
    let tables = if sine {
        [
            c::FRESNEL_S_0,
            c::FRESNEL_S_1,
            c::FRESNEL_S_2,
            c::FRESNEL_S_3,
        ]
    } else {
        [
            c::FRESNEL_C_0,
            c::FRESNEL_C_1,
            c::FRESNEL_C_2,
            c::FRESNEL_C_3,
        ]
    };
    if x <= real::<S>(8.)? {
        let (index, lo, hi) = if x <= S::ONE {
            (0, 0., 1.)
        } else if x <= real::<S>(2.)? {
            (1, 1., 2.)
        } else if x <= real::<S>(4.)? {
            (2, 2., 4.)
        } else {
            (3, 4., 8.)
        };
        let (v, e) = table(x, lo, hi, tables[index], work)?;
        return Ok(if sine {
            (((v * x) * x) * x, ((e * x) * x) * x)
        } else {
            (v * x, e * x)
        });
    }
    let inv = S::ONE / x;
    let pi = real::<S>(core::f64::consts::PI)?;
    let r = inv * inv / pi;
    let q = r * r;
    let mut f = S::ONE;
    let mut g = S::ONE;
    let mut ft = S::ONE;
    let mut gt = S::ONE;
    let mut last = S::ONE;
    let mut k = 0usize;
    while work.step() {
        k += 1;
        let a = real::<S>((4 * k) as f64)?;
        let nf = -ft * ((a - S::ONE) * (a - real::<S>(3.)?) * q);
        let ng = -gt * ((a + S::ONE) * (a - S::ONE) * q);
        if ng.abs() >= gt.abs() {
            break;
        }
        ft = nf;
        gt = ng;
        f += ft;
        g += gt;
        last = ft.abs().max(gt.abs());
        if last <= S::EPSILON {
            break;
        }
    }
    let (s, c) = fresnel_phase(x)?;
    let f = f * (inv / pi);
    let g = g * (inv / pi) * r;
    let value = if sine {
        S::HALF - f * c - g * s
    } else {
        S::HALF + f * s - g * c
    };
    Ok((
        value,
        last * inv + real::<S>(16.)? * S::EPSILON * value.abs(),
    ))
}

fn evaluate_ei<S: RealScalar>(
    kind: Kind,
    x: S,
    work: &mut Work,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    let zero = S::ZERO;
    let one = S::ONE;

    if x == zero {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::Pole));
    }
    let e1 = matches!(kind, Kind::E1 | Kind::E1Scaled);
    let argument = if e1 { -x } else { x };
    let (v, e, exponent) = ei(argument, work)?;
    let factor = if e1 { -one } else { one };
    let scaled = matches!(kind, Kind::EiScaled | Kind::E1Scaled);
    let exponent = exponent - if scaled { argument } else { zero };
    let ordinary = project(factor * v, exponent);
    let derivative = if scaled {
        if e1 {
            ordinary - one / x
        } else {
            one / x - ordinary
        }
    } else {
        if e1 {
            project(-one / x, -x)
        } else {
            project(one / x, x)
        }
    };
    work.check()?;
    finish(
        factor * v,
        e,
        exponent,
        Some(derivative),
        work,
        algorithm_component!("ei_piecewise_rational"),
    )
}

fn evaluate_li<S: RealScalar>(
    kind: Kind,
    x: S,
    work: &mut Work,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    let zero = S::ZERO;
    let one = S::ONE;

    if x == one {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::Pole));
    }
    let offset = matches!(kind, Kind::LiOffset);
    if x == zero {
        return finish(
            if offset {
                -real::<S>(1.045_163_780_117_493)?
            } else {
                zero
            },
            zero,
            zero,
            Some(zero),
            work,
            algorithm_component!("li_endpoint"),
        );
    }
    if offset && x >= real::<S>(1.5)? && x <= real::<S>(2.5)? {
        let (v, e) = table(x, 1.5, 2.5, c::LI_OFFSET_0, work)?;
        let delta = x - real::<S>(2.)?;
        return finish(
            v * delta,
            e * delta.abs(),
            zero,
            Some(one / x.ln()),
            work,
            algorithm_component!("li_offset_chebyshev"),
        );
    }
    let logarithm = if (x - one).abs() < S::HALF {
        (x - one).ln_1p()
    } else {
        x.ln()
    };
    let (v, e, exponent) = ei(logarithm, work)?;
    // li(x) is representable for finite positive x. Subtract the offset
    // only outside its cancellation-sensitive neighborhood.
    let factor = if exponent == zero { one } else { x };
    let value = v * factor
        - if offset {
            real::<S>(1.045_163_780_117_493)?
        } else {
            zero
        };
    work.check()?;
    finish(
        value,
        e * factor,
        zero,
        Some(one / logarithm),
        work,
        algorithm_component!("li_ei"),
    )
}

pub(crate) fn evaluate<S: RealScalar>(
    kind: Kind,
    x: S,
    options: IntegralOptions,
    slope: bool,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<IntegralEvaluation<S>, SciOxideError> {
    crate::support::require_finite_param(x, "integral parameter x must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let a = x.abs();
    let zero = S::ZERO;
    let one = S::ONE;
    let sign = one.copysign(x);
    if matches!(kind, Kind::E2 | Kind::E2Scaled) {
        return en_report(2, x, matches!(kind, Kind::E2Scaled), slope, &mut work);
    }
    if matches!(kind, Kind::E1 | Kind::E1Scaled) && x > zero {
        return en_report(1, x, matches!(kind, Kind::E1Scaled), slope, &mut work);
    }
    if matches!(kind, Kind::Ei | Kind::EiScaled | Kind::E1 | Kind::E1Scaled) {
        return evaluate_ei(kind, x, &mut work);
    }
    if matches!(kind, Kind::Ci | Kind::Chi | Kind::ChiScaled) && x <= zero {
        return Err(ERRORS.domain(
            scioxide_core::argument!("x"),
            if x == zero {
                DomainReason::Pole
            } else {
                DomainReason::OutsideDomain
            },
        ));
    }
    if matches!(kind, Kind::Expint3 | Kind::Li | Kind::LiOffset) && x < zero {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::OutsideDomain));
    }
    if matches!(kind, Kind::Li | Kind::LiOffset) {
        return evaluate_li(kind, x, &mut work);
    }
    if a == zero {
        let derivative = if matches!(kind, Kind::FresnelS) {
            zero
        } else {
            one
        };
        return finish(
            x,
            zero,
            zero,
            Some(derivative),
            &work,
            algorithm_component!("origin"),
        );
    }
    let (value, error, exponent, derivative, method) = match kind {
        Kind::Si | Kind::Ci => {
            let cosine = matches!(kind, Kind::Ci);
            let (v, e) = trigonometric(a, cosine, &mut work)?;
            (
                if cosine { v } else { sign * v },
                e,
                zero,
                Some(if cosine {
                    x.cos() / x
                } else {
                    scioxide_core::sinc(x)?
                }),
                algorithm_component!("sici_auxiliary_rational"),
            )
        }
        Kind::Shi | Kind::Chi | Kind::ShiScaled | Kind::ChiScaled => {
            let cosine = matches!(kind, Kind::Chi | Kind::ChiScaled);
            let scaled = matches!(kind, Kind::ShiScaled | Kind::ChiScaled);
            let (v, e) = if a <= one {
                let (v, e) = hyperbolic_origin(a, cosine, &mut work)?;
                let factor = (-a).exp();
                (v * factor, e * factor)
            } else {
                let (v, e, _) = ei(a, &mut work)?;
                let (w, we) = en(1, a, &mut work)?;
                let factor = (-a - a).exp();
                (
                    (v + if cosine { -w * factor } else { w * factor }) * S::HALF,
                    (e + we * factor) * S::HALF,
                )
            };
            let tail = (-a - a).exp();
            let dv = if cosine {
                (one + tail) * S::HALF / a
            } else {
                -(-a - a).exp_m1() * S::HALF / a
            };
            let derivative = if scaled { dv - v } else { project(dv, a) };
            (
                if cosine { v } else { sign * v },
                e,
                if scaled { zero } else { a },
                Some(derivative),
                algorithm_component!("shichi_scaled_series_ei"),
            )
        }
        Kind::FresnelC | Kind::FresnelS => {
            let sine = matches!(kind, Kind::FresnelS);
            let (v, e) = fresnel(a, sine, &mut work)?;
            let (s, c) = fresnel_phase(x)?;
            (
                sign * v,
                e,
                zero,
                Some(if sine { s } else { c }),
                algorithm_component!("fresnel_chebyshev_asymptotic"),
            )
        }
        Kind::Atanint => {
            let t = if a > one { one / a } else { a };
            let (v, e) = table(t, 0., 1., c::ATANINT_0, &mut work)?;
            let v = v * t
                + if a > one {
                    real::<S>(core::f64::consts::FRAC_PI_2)? * a.ln()
                } else {
                    zero
                };
            (
                sign * v,
                e * t,
                zero,
                Some(x.atan() / x),
                algorithm_component!("atanint_reciprocal_chebyshev"),
            )
        }
        Kind::Expint3 => {
            let (v, e) = if a <= one {
                table(a, 0., 1., c::EXPINT3_0, &mut work)?
            } else if a <= real::<S>(2.)? {
                table(a, 1., 2., c::EXPINT3_1, &mut work)?
            } else if a <= real::<S>(3.)? {
                table(a, 2., 3., c::EXPINT3_2, &mut work)?
            } else if a <= real::<S>(4.)? {
                table(a, 3., 4., c::EXPINT3_3, &mut work)?
            } else {
                // The omitted positive tail is bounded by exp(-x³)/(3x²); the
                // limit Γ(4/3) is rounded once.
                let limit = real::<S>(f64::from_bits(0x3fec_9349_c4c6_03ed))?;
                let tail = (-(a * a) * a).exp() / (real::<S>(3.)? * a * a);
                return finish(
                    limit,
                    tail + S::EPSILON * limit,
                    zero,
                    Some(if a < real::<S>(16.)? {
                        (-(a * a) * a).exp()
                    } else {
                        zero
                    }),
                    &work,
                    algorithm_component!("expint3_limit"),
                );
            };
            (
                v * a,
                e * a,
                zero,
                Some((-(a * a) * a).exp()),
                algorithm_component!("expint3_chebyshev"),
            )
        }
        _ => {
            return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::OutsideDomain));
        }
    };
    work.check()?;
    finish(value, error, exponent, derivative, &work, method)
}

pub(crate) mod complex_fresnel;
mod integral_coefficients;
mod real_coefficients;

/// Upper incomplete gamma at zero shape, sharing the scaled E1 kernel and allowance.
pub(crate) fn e1_logarithmic<S: RealScalar>(
    x: S,
    work: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    let (value, error) = en(1, x, work)?;
    work.check()?;
    if work.exhausted() {
        return Err(work.limit_error());
    }
    let logarithm = value.ln() - x;
    Ok(Evaluation::logarithmic(
        S::ONE,
        logarithm,
        crate::result::Bound::derived(error / value + S::EPSILON * logarithm.abs()),
        crate::result::Projection::Direct(
            project(value, -x),
            crate::result::Bound::derived(project(error, -x)),
        ),
        scioxide_core::algorithm_id!("gamma", "zero_shape_e1", 1),
        work.used,
    ))
}

pub(crate) mod api;
