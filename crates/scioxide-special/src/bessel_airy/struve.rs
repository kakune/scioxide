//! Struve H/L by the defining series in two-term arithmetic (DLMF 11.2.1,
//! 11.2.2) and the large-argument expansions (DLMF 11.6.1, 11.6.2).
//!
//! - From x = 20 the expansions H = Y_nu + K_nu and L = I_nu + M_nu are tried.
//!   K and M are summed until their smallest term past order nu - 1/2, where
//!   DLMF 11.6.1 bounds the remainder of K by its first neglected term with the
//!   same sign. H takes half that term and accepts when the other half is below
//!   eps/2 of max(|K|, sqrt(2/(pi x))), the amplitude of Y_nu; L accepts when
//!   the truncation of M and the exponentially small I_nu - I_-nu ambiguity are
//!   below eps/16 of I_nu.
//! - Otherwise the defining series is summed in two-term arithmetic: the
//!   cancellation in H then costs only two-term rounding, and the f64
//!   prefactor scales the sum without amplification.
//! - Orders nu + 1/2 in Z<=0 reduce to Bessel J/I of half-integer order.
//!
//! On nu in [-5, 10], x in [1e-3, 100] both functions are within 64 f64 ulps
//! of the reference values, except near zeros of H_nu, where the absolute error
//! is at most 4 eps max(1, |H_nu|). Value bounds are derived from the remainder
//! and rounding of the evaluation; where Bessel kernels contribute, their
//! tested error envelope makes the bound statistical.
use crate::bessel_airy::bessel_base::{self as wide, Wide};
use crate::bessel_airy::real_api::BESSEL_ULPS;
use crate::expansions::bounded;
use crate::gamma_beta::gamma::real::{gamma_positive_wide, ln_negative_wide, ln_positive_wide};
use crate::result::{Bound, Evaluation};
use crate::support::Work;
use crate::support::wide::project;
use crate::{SpecialFunctionResult, StruveOptions};
use scioxide_core::{AlgorithmComponent, RealScalar, SciOxideError, algorithm_component};
type Detailed<S> = SpecialFunctionResult<S>;

const EPS: f64 = f64::EPSILON;
/// Argument from which the large-argument expansions are attempted; below it Y_nu and K_nu may
/// cancel, and I_nu - I_-nu is not negligible against I_nu.
const ASYMPTOTIC_ARGUMENT: f64 = 20.0;
/// Remainders below this fraction of a sum do not change it.
const TAIL: f64 = EPS / 16.0;
/// Two-term resolution u^2 with u = 2^-53.
const RESOLUTION: f64 = EPS * EPS / 4.0;
/// Powers of two folded into the logarithmic scale before a series term overflows.
const RESCALE_EXPONENT: i32 = 960;
/// Relative error of a direct prefactor: the 4-unit `powf` contract and one rounding of the
/// two-term quotient.
const PREFACTOR_ULPS: f64 = 4.5;
/// 2/sqrt(pi) in two terms.
use crate::support::constants::FRAC_2_SQRT_PI as TWO_OVER_SQRT_PI;

/// Kernel output in value units; `envelope` is the Bessel magnitude covered by the tested
/// Bessel error envelope.
struct Pair {
    value: f64,
    derivative: f64,
    error: f64,
    envelope: f64,
    derivative_error: f64,
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Struve function `H_ν(x)` for real order `ν` and real `x`.

```text
H_ν(x) = Σ_{k=0}^∞ (−1)^k (x/2)^(ν+2k+1) / (Γ(k + 3/2) Γ(k + ν + 3/2))   (DLMF 11.2.1)
```

Terms where `Γ(k + ν + 3/2)` has a pole vanish. Every finite `ν` is accepted. For `x < 0` the
order must be an integer `n`, and `H_n(−x) = (−1)^(n+1) H_n(x)`. At `x = 0`, `H_ν(0) = 0` for
`ν > −1` and for `ν = −3/2, −5/2, …`, and `H_{−1}(0) = 2/π`; other orders have a pole there.
[`struve_with_options`](crate::struve_with_options) also returns `H′_ν(x)`, except at `x = 0`
for `−1 < ν < 0`, where it is infinite.

From `x = 20` the function tries `H_ν = Y_ν + K_ν`, where `Y_ν` is the Bessel function of the
second kind and `K_ν` is summed from its asymptotic expansion (DLMF 11.6.1). It accepts the
result once the truncation error is below `f64` rounding. Otherwise it sums the defining series
in double-word arithmetic, so cancellation between terms costs only double-word rounding.
Orders `ν = −(n + 1/2)` use `H_ν(x) = (−1)^n J_{n+1/2}(x)` (DLMF 11.4.4).

# Examples

```
// H_{1/2}(x) = √(2/(πx)) (1 − cos x) (DLMF 11.4)
let x = 2.0_f64;
let expected = (2.0 / (core::f64::consts::PI * x)).sqrt() * (1.0 - x.cos());
assert!((scioxide_special::struve(0.5, x)? - expected).abs() < 1e-13);

// H_{−2} has a pole at the origin.
let pole = scioxide_special::struve(-2.0_f64, 0.0).unwrap_err();
assert!(matches!(pole.view(), scioxide_core::SciOxideErrorView::DomainError {
    reason: scioxide_core::DomainReason::Pole, ..
}));
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For `f64`, `H_ν(x)` is within 64 ulps on `ν ∈ [−5, 10]`, `x ∈ [10⁻³, 100]`. Near zeros of
`H_ν` the absolute error is at most `4ε·max(1, |H_ν(x)|)` with `ε = f64::EPSILON`.
Outside this range and for `f32`, tests compare values, derivatives and error estimates with
reference values without a uniform ulp bound.
[`struve_with_options`](crate::struve_with_options) returns an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0` and `ν` is not an integer.
- `DomainError` (`Pole`) at `x = 0` if `ν < −1` and `ν ∉ {−3/2, −5/2, …}`.
- `NumericalInstability` (`Overflow`) if `|H_ν(x)|` exceeds the `f64` range. For `f32`, a
  value beyond the `f32` range returns `ScalarConversion`.

# References

NIST DLMF, §11.2–11.6. <https://dlmf.nist.gov/11>"###)]

    pub struve [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: StruveOptions;
    detailed struve_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(struve));
        evaluate(order, x, options, false, family, context)
    }
    scoped pub struve_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the modified Struve function `L_ν(x)` for real order `ν` and real `x`.

```text
L_ν(x) = Σ_{k=0}^∞ (x/2)^(ν+2k+1) / (Γ(k + 3/2) Γ(k + ν + 3/2))   (DLMF 11.2.2)
```

Terms where `Γ(k + ν + 3/2)` has a pole vanish. Every finite `ν` is accepted. For `x < 0` the
order must be an integer `n`, and `L_n(−x) = (−1)^(n+1) L_n(x)`. At `x = 0`, `L_ν(0) = 0` for
`ν > −1` and for `ν = −3/2, −5/2, …`, and `L_{−1}(0) = 2/π`; other orders have a pole there.
`L_ν(x)` grows like `exp(x)/√(2πx)`. [`modstruve_with_options`](crate::modstruve_with_options)
also returns `L′_ν(x)`, except at `x = 0` for `−1 < ν < 0`, where it is infinite.

From `x = 20` the function tries `L_ν = I_ν + M_ν`, where `I_ν` is the modified Bessel function
and `M_ν` is summed from its asymptotic expansion (DLMF 11.6.2). It accepts the result once the
truncation error and the exponentially small difference `I_ν − I_{−ν}` are below `f64`
rounding. Otherwise it sums the defining series in double-word arithmetic. Orders
`ν = −(n + 1/2)` use `L_ν(x) = I_{n+1/2}(x)` (DLMF 11.4.5).

# Examples

```
// L_{1/2}(x) = √(2/(πx)) (cosh x − 1) (DLMF 11.4)
let x = 2.0_f64;
let expected = (2.0 / (core::f64::consts::PI * x)).sqrt() * (x.cosh() - 1.0);
assert!((scioxide_special::modstruve(0.5, x)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For `f64`, `L_ν(x)` is within 64 ulps on `ν ∈ [−5, 10]`, `x ∈ [10⁻³, 100]`.
Outside this range and for `f32`, tests compare values, derivatives and error estimates with
reference values without a uniform ulp bound.
[`modstruve_with_options`](crate::modstruve_with_options) returns an error estimate. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0` and `ν` is not an integer.
- `DomainError` (`Pole`) at `x = 0` if `ν < −1` and `ν ∉ {−3/2, −5/2, …}`.
- `NumericalInstability` (`Overflow`) if `|L_ν(x)|` exceeds the `f64` range. For `f32`, a
  value beyond the `f32` range returns `ScalarConversion`.

# References

NIST DLMF, §11.2–11.6. <https://dlmf.nist.gov/11>"###)]

    pub modstruve [S: RealScalar] (order: S, x: S) -> S
    where [];
    options options: StruveOptions;
    detailed modstruve_with_options -> Detailed<S>;
    evaluate |options, context| {
        let family = algorithm_component!(stringify!(modstruve));
        evaluate(order, x, options, true, family, context)
    }
    scoped pub modstruve_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

fn evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: StruveOptions,
    modified: bool,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if !order.is_finite() || !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "order and argument must be finite",
        ));
    }
    if x < S::ZERO && order != order.round() {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "negative arguments require integral order",
        ));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let computed = kernel(
        order.try_to_f64_exact()?,
        x.abs().try_to_f64_exact()?,
        modified,
        &mut work,
    );
    work.check()?;
    let mut pair = match computed {
        Ok(pair) => pair,
        Err(_) if !work.complete => interrupted(),
        Err(error) => return Err(error),
    };
    if x.is_sign_negative() {
        let odd = order % (S::TWO) == S::ZERO;
        if odd {
            pair.value = -pair.value;
        } else {
            pair.derivative = -pair.derivative;
        }
    }
    crate::support::check_finite(pair.value)?;
    let ceiling = S::MAX.try_to_f64_exact()?;
    let value = crate::support::project_real::<S>(pair.value)?;
    let error = S::try_from_f64_lossy(pair.error.min(ceiling))? + S::EPSILON * value.abs();
    let bound = if pair.envelope > 0.0 {
        Bound::derived(error).plus(Bound::envelope(
            S::try_from_f64_lossy(pair.envelope.min(ceiling))?,
            BESSEL_ULPS,
        )?)
    } else {
        Bound::derived(error)
    };
    let derivative = if pair.derivative.is_finite()
        && S::try_from_f64_lossy(pair.derivative).is_ok()
    {
        let d = S::try_from_f64_lossy(pair.derivative)?;
        let e = S::try_from_f64_lossy(pair.derivative_error.min(ceiling))? + S::EPSILON * d.abs();
        e.is_finite().then(|| (d, Bound::heuristic(e)))
    } else {
        None
    };
    Ok(bounded(
        value,
        bound,
        scioxide_core::AlgorithmId::from_components(
            family,
            algorithm_component!("series_asymptotic"),
            1,
        ),
        work.used,
        work.complete,
    )?
    .with_derivative(derivative))
}

fn interrupted() -> Pair {
    Pair {
        value: 0.0,
        derivative: 0.0,
        error: f64::MAX,
        envelope: 0.0,
        derivative_error: f64::MAX,
    }
}

/// Amplitude sqrt(2/(pi x)) of the oscillatory Bessel functions.
fn amplitude(x: f64) -> f64 {
    (2.0 / (core::f64::consts::PI * x)).sqrt()
}

fn kernel(nu: f64, x: f64, modified: bool, w: &mut Work) -> Result<Pair, SciOxideError> {
    if x == 0.0 {
        return origin(nu);
    }
    let reflected = nu + 0.5;
    if reflected <= 0.0 && reflected == reflected.round() {
        return half_integer(nu, x, modified, w);
    }
    if let Some(pair) = asymptotic(nu, x, modified, w)? {
        return Ok(pair);
    }
    series(nu, x, modified, w)
}

/// Limits at x = 0: zero for nu > -1, 2/pi for H_-1 and L_-1, zero at the half-integer orders where
/// 1/Gamma(nu + 3/2) vanishes, otherwise a pole.
fn origin(nu: f64) -> Result<Pair, SciOxideError> {
    let half_integer = nu + 1.5;
    let gamma_pole = half_integer <= 0.0 && half_integer == half_integer.round();
    if nu < -1.0 && !gamma_pole {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::Pole,
            "origin is singular for this order",
        ));
    }
    let value = if nu == -1.0 {
        core::f64::consts::FRAC_2_PI
    } else {
        0.0
    };
    let derivative = if nu == 0.0 {
        core::f64::consts::FRAC_2_PI
    } else if nu > -1.0 && nu < 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    Ok(Pair {
        value,
        derivative,
        error: EPS * value.abs(),
        envelope: 0.0,
        derivative_error: EPS * derivative.abs(),
    })
}

/// H_-(n+1/2) = (-1)^n J_(n+1/2) and L_-(n+1/2) = I_(n+1/2) (DLMF 11.4.4-5).
fn half_integer(nu: f64, x: f64, modified: bool, w: &mut Work) -> Result<Pair, SciOxideError> {
    let order = -nu;
    let (value, lower) = if modified {
        (
            crate::bessel_airy::bessel::bessel_i(order, x, w)?,
            crate::bessel_airy::bessel::bessel_i(order - 1.0, x, w)?,
        )
    } else {
        (
            crate::bessel_airy::bessel::bessel_j(order, x, w)?,
            crate::bessel_airy::bessel::bessel_j(order - 1.0, x, w)?,
        )
    };
    let sign = if modified {
        1.0
    } else {
        crate::util::cos_pi(nu + 0.5)
    };
    let derivative = lower - (order * value) / x;
    Ok(Pair {
        value: sign * value,
        derivative: sign * derivative,
        error: 0.0,
        envelope: value.abs(),
        derivative_error: 64.0 * EPS * (lower.abs() + ((order * value) / x).abs()),
    })
}

/// A coefficient `value * exp(log)` with relative error `error`; `log` is zero in f64 range.
struct Scale {
    value: f64,
    log: f64,
    error: f64,
}

impl Scale {
    fn magnitude(&self, normalized: f64) -> f64 {
        project(
            Wide::from(self.value.abs() * normalized),
            Wide::from(self.log),
        )
    }
}

/// (2/sqrt(pi)) (x/2)^(nu+1) / Gamma(nu+3/2) for the defining series, or (1/sqrt(pi)) (x/2)^(nu-1)
/// / Gamma(nu+1/2) for the large-argument expansion. The power keeps nu exact and Gamma's argument
/// is an exact two-term sum, so only `powf` and the final rounding remain.
fn prefactor(nu: f64, x: f64, series: bool) -> Scale {
    let half = 0.5 * x;
    let (offset, constant, shift) = if series {
        (1.5, TWO_OVER_SQRT_PI, 1.0)
    } else {
        (0.5, TWO_OVER_SQRT_PI.mul_f64(0.5), -1.0)
    };
    let argument = Wide::from(nu) + Wide::from(offset);
    let power = half.powf(nu);
    if power.is_normal() && argument.hi.abs() < 60.0 {
        let quotient = constant * Wide::from(power) / gamma_wide(argument);
        let value = if series {
            quotient.mul_f64(half)
        } else {
            quotient.div_f64(half)
        }
        .value();
        if value.is_normal() {
            return Scale {
                value,
                log: 0.0,
                error: PREFACTOR_ULPS * EPS,
            };
        }
    }
    let (log_gamma, sign) = if argument.hi > 0.0 {
        (ln_positive_wide(argument), 1.0)
    } else {
        // Gamma alternates in sign between consecutive negative integers.
        let parity = (-argument.hi).ceil();
        (
            ln_negative_wide(argument),
            if parity % 2.0 == 0.0 { 1.0 } else { -1.0 },
        )
    };
    let log = (Wide::from(half).ln().mul_f64(nu + shift) + constant.ln() - log_gamma).value();
    Scale {
        value: sign,
        log,
        error: (PREFACTOR_ULPS + 2.0 + log.abs()) * EPS,
    }
}

/// Two-term Gamma(a) for |a| < 60 away from the poles; negative arguments recur upward with exact
/// two-term factors.
fn gamma_wide(argument: Wide) -> Wide {
    let mut shifted = argument;
    let mut product = Wide::from(1.0);
    while shifted.hi < 0.5 {
        product = product * shifted;
        shifted = shifted + Wide::from(1.0);
    }
    gamma_positive_wide(shifted) / product
}

/// DLMF 11.2.1 (H) and 11.2.2 (L) in two-term arithmetic: P sum_k t_k, t_(k+1)/t_k = -+(x/2)^2 /
/// ((k+3/2)(k+nu+3/2)), t_0 = 1.
fn series(nu: f64, x: f64, modified: bool, w: &mut Work) -> Result<Pair, SciOxideError> {
    if !w.step() {
        return Ok(interrupted());
    }
    let mut scale = prefactor(nu, x, true);
    let square = wide::two_product(0.5 * x, 0.5 * x);
    let factor = if modified { square } else { -square };
    // nu + 3/2 and nu + 1 exactly.
    let shift = wide::sum(Wide::from(nu), Wide::from(1.5));
    let order = wide::sum(Wide::from(nu), Wide::from(1.0));
    let mut term = Wide::from(1.0);
    let mut sum = term;
    // Sum of k t_k for the derivative (x/2)^(2k+nu+1)' = (2k+nu+1)/x.
    let mut moment = Wide::ZERO;
    // Sums of |t_k|, k |t_k| and |partial sums| for the rounding model.
    let (mut absolute, mut moment_absolute, mut partials) = (1.0_f64, 0.0_f64, 1.0_f64);
    let mut k = 0.0_f64;
    let (tail, moment_tail) = loop {
        if !w.step() {
            return Ok(interrupted());
        }
        // The ratio does not depend on the running term, keeping its
        // division off the dependent chain.
        let denominator = wide::scale(wide::sum(shift, Wide::from(k)), k + 1.5);
        let ratio = wide::quotient(factor, denominator);
        term = wide::product(term, ratio);
        k += 1.0;
        if term.hi.abs() > 1.0_f64.scalbn(RESCALE_EXPONENT) {
            let down = 1.0_f64.scalbn(-RESCALE_EXPONENT);
            term = wide::scale(term, down);
            sum = wide::scale(sum, down);
            moment = wide::scale(moment, down);
            absolute *= down;
            moment_absolute *= down;
            partials *= down;
            let lift = f64::from(RESCALE_EXPONENT) * core::f64::consts::LN_2;
            scale.log += lift;
            scale.error += lift * EPS;
        }
        sum = wide::sum(sum, term);
        moment = wide::sum(moment, wide::scale(term, k));
        absolute += term.hi.abs();
        moment_absolute += k * term.hi.abs();
        partials += sum.hi.abs();
        // Once k + nu + 3/2 > 0 the ratio magnitudes decrease, so the
        // remainder after t_k is geometric in the next ratio r.
        let base = k + shift.hi;
        let next = (factor.hi / ((k + 1.5) * base)).abs();
        if base > 0.0 && next < 1.0 {
            let tail = term.hi.abs() * next / (1.0 - next);
            let moment_tail = term.hi.abs() * next * (k + 1.0 / (1.0 - next)) / (1.0 - next);
            let derivative = order.hi * sum.hi + 2.0 * moment.hi;
            let value_done = tail <= TAIL * sum.hi.abs() || tail <= RESOLUTION * absolute;
            let derivative_done = order.hi.abs() * tail + 2.0 * moment_tail
                <= TAIL * derivative.abs()
                || order.hi.abs() * tail + 2.0 * moment_tail
                    <= RESOLUTION * (order.hi.abs() * absolute + 2.0 * moment_absolute);
            if value_done && derivative_done {
                break (tail, moment_tail);
            }
        }
    };
    // Each two-term operation errs by at most 4u^2 (u = 2^-53) of its
    // operands: products and quotients accumulate along the terms, sums
    // along the partial sums.
    let count = k + 1.0;
    let rounding =
        RESOLUTION * (8.0 * (count * sum.hi.abs() + partials) + 4.0 * (partials + absolute));
    let value = scale.value * sum.value();
    let derivative_sum = wide::sum(wide::product(order, sum), wide::scale(moment, 2.0));
    let derivative = scale.value * derivative_sum.value() / x;
    let derivative_rounding =
        RESOLUTION * (8.0 * count + 8.0) * (order.hi.abs() * absolute + 2.0 * moment_absolute);
    Ok(Pair {
        value: project(Wide::from(value), Wide::from(scale.log)),
        derivative: project(Wide::from(derivative), Wide::from(scale.log)),
        error: scale.magnitude(tail + rounding)
            + (scale.error + EPS) * project(Wide::from(value), Wide::from(scale.log)).abs(),
        envelope: 0.0,
        derivative_error: scale
            .magnitude((order.hi.abs() * tail + 2.0 * moment_tail + derivative_rounding) / x)
            + (scale.error + 2.0 * EPS)
                * project(Wide::from(derivative), Wide::from(scale.log)).abs(),
    })
}

/// DLMF 11.6.1 and 11.6.2: K_nu = H_nu - Y_nu and M_nu = L_nu - I_nu as +-Q sum_k t_k with Q =
/// (x/2)^(nu-1)/(sqrt(pi) Gamma(nu+1/2)), t_(k+1)/t_k = +-(2k+1)(2nu-2k-1)/x^2 and t_0 = 1, or None
/// when the expansion cannot reach f64 accuracy.
fn asymptotic(
    nu: f64,
    x: f64,
    modified: bool,
    w: &mut Work,
) -> Result<Option<Pair>, SciOxideError> {
    // The remainder bound needs at least nu - 1/2 terms.
    if x < ASYMPTOTIC_ARGUMENT || nu - 0.5 > w.remaining as f64 {
        return Ok(None);
    }
    let sign = if modified { -1.0 } else { 1.0 };
    let square = x * x;
    let mut term = 1.0_f64;
    let mut sum = Wide::from(1.0);
    // Sum of (nu-1-2k) t_k for the derivative; rounding weights of the terms.
    let mut moment = nu - 1.0;
    let mut weighted = 0.0_f64;
    let mut count = 1.0_f64;
    let omitted = loop {
        if !w.step() {
            return Ok(Some(interrupted()));
        }
        let index = count - 1.0;
        let next = term * sign * ((2.0 * index + 1.0) * (2.0 * (nu - index) - 1.0)) / square;
        if count >= nu - 0.5
            && (next == 0.0 || next.abs() >= term.abs() || next.abs() <= TAIL * sum.hi.abs())
        {
            break next;
        }
        term = next;
        sum = wide::sum(sum, Wide::from(term));
        moment += (nu - 1.0 - 2.0 * count) * term;
        // Each term carries at most five roundings per recurrence step.
        weighted += 5.0 * count * term.abs();
        count += 1.0;
    };
    let scale = prefactor(nu, x, false);
    let leading = sign * scale.value;
    if modified {
        let correction = project(Wide::from(leading * sum.value()), Wide::from(scale.log));
        let base = crate::bessel_airy::bessel::bessel_i(nu, x, w)?;
        // DLMF bounds no remainder of M: x times its first neglected term,
        // and 2 exp(nu^2/x - 2x) I_nu for I_nu - I_-nu, serve as envelopes.
        let truncation =
            scale.magnitude(x * omitted.abs()) + 2.0 * (nu * nu / x - 2.0 * x).exp() * base.abs();
        if truncation > TAIL * base.abs() {
            return Ok(None);
        }
        let upper = crate::bessel_airy::bessel::bessel_i(nu + 1.0, x, w)?;
        let d_correction = project(Wide::from(leading * moment), Wide::from(scale.log)) / x;
        return Ok(Some(Pair {
            value: base + correction,
            derivative: upper + nu / x * base + d_correction,
            error: truncation
                + scale.magnitude(weighted * EPS + RESOLUTION * count)
                + scale.error * correction.abs(),
            envelope: base.abs(),
            derivative_error: 64.0 * EPS * (upper.abs() + (nu / x * base).abs())
                + (scale.error + EPS * count) * d_correction.abs(),
        }));
    }
    // K's remainder has the sign of its first neglected term and is bounded
    // by it: taking half leaves at most half.
    let normalized = wide::sum(sum, Wide::from(0.5 * omitted)).value();
    let correction = project(Wide::from(leading * normalized), Wide::from(scale.log));
    let truncation = scale.magnitude(0.5 * omitted.abs());
    if truncation > 0.5 * EPS * correction.abs().max(amplitude(x)) {
        return Ok(None);
    }
    let base = crate::bessel_airy::bessel::bessel_y(nu, x, w)?;
    let value = base + correction;
    // Y_nu and K_nu cancel only for nu beyond x, where the series serves.
    if base.abs() + correction.abs() > 64.0 * value.abs().max(amplitude(x)) {
        return Ok(None);
    }
    let upper = crate::bessel_airy::bessel::bessel_y(nu + 1.0, x, w)?;
    let d_correction = project(Wide::from(leading * moment), Wide::from(scale.log)) / x;
    // Reflected orders below -1/2 err by eps times the amplitude, not |Y|.
    Ok(Some(Pair {
        value,
        derivative: nu / x * base - upper + d_correction,
        error: truncation
            + scale.magnitude(weighted * EPS + RESOLUTION * count)
            + scale.error * correction.abs(),
        envelope: base.abs().max(amplitude(x)),
        derivative_error: 64.0
            * EPS
            * (upper.abs().max(amplitude(x)) + (nu / x).abs() * base.abs().max(amplitude(x)))
            + (scale.error + EPS * count) * d_correction.abs(),
    }))
}
