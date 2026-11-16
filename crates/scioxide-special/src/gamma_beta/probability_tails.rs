//! Direct/logarithmic complementary gamma and beta tails and bounded inverses.
//!
//! Inverses use asymptotic seeds and scalar safeguarded Halley steps, switching
//! to compensated tails before updating an uncertain bracket. Final refinement
//! terminates on the output lattice, including tiny and rounded quantiles.
//! Wide tail residuals preserve small-shape sensitivity; correctly rounded
//! endpoint probabilities and quantiles keep harmless zero/one rounding.
//! Scalar shape normalization carries an uncertainty; compensated normalization
//! and the two-word target are prepared lazily and reused. Scalar steps prefer
//! the `f64` linear tails, whose power factor gives the derivative directly.
//! Scalar completion and compensated gamma-series refinement both reserve
//! sixteen output ulps, inside the 32-ulp inverse target.
use crate::error_probability::probability::{error, finish, wide_log1mexp};
use crate::expansions::real;
use crate::gamma_beta::incomplete_gamma::Normalization;
use crate::result::{Bound, Evaluation};
use crate::support::PROBABILITY_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::require_finite_param;
use crate::support::wide::Wide;
use crate::{BetaOptions, GammaOptions, ProbabilityOptions, SpecialFunctionResult};
use scioxide_core::AlgorithmComponent;
use scioxide_core::algorithm_component;
use scioxide_core::{RealScalar, SciOxideError};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln P(a, x)`, the logarithm of the regularized lower incomplete gamma function, without
forming `P`.

`P(a, x)` is defined at [`crate::gamma_p`]; `a ≥ 0` and `x ≥ 0`. Use this form when `P`
underflows, for example in log-likelihoods. `ln P(a, 0) = −∞` for `a > 0` is returned as a
value, not an error, and `ln P(0, x) = 0` for `x > 0`.

The tail is evaluated in logarithmic form with the methods of [`crate::gamma_p`]. When the
smaller tail is the complement, the result is formed with `ln(1 − T)` of the smaller tail `T`
without cancellation.

# Examples

```
// ln P(1, 1) = ln(1 − e^(−1))
let v = scioxide_special::gamma_log_p(1.0_f64, 1.0)?;
assert!((v - (-(-1.0_f64).exp()).ln_1p()).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The tails are within 32 ulps for `f64` and 1 ulp for `f32` on the tested range; the detailed
form returns an error estimate for the logarithm. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0` or `x < 0`.
- `DomainError` (`Undefined`) if `a = 0` and `x = 0`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_log_p [S: RealScalar] (a: S, x: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_log_p_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        gamma_log_p_evaluate(a, x, options, context)
    }
    scoped pub gamma_log_p_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn gamma_log_p_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_log_p_evaluate::<f64>(a.into_f64(), x.into_f64(), options, context.as_deref_mut()),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, branch) =
        crate::gamma_beta::incomplete_gamma::probability_log_tail(a, x, false, &mut w)?;
    w.check()?;
    tail_finish(v, &w, branch)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln Q(a, x)`, the logarithm of the regularized upper incomplete gamma function, without
forming `Q`.

`Q(a, x)` is defined at [`crate::gamma_q`]; `a ≥ 0` and `x ≥ 0`. Use this form for upper tails
that underflow. `ln Q(a, 0) = 0` for `a > 0`, and `ln Q(0, x) = −∞` for `x > 0` is returned as a
value, not an error.

The tail is evaluated in logarithmic form with the methods of [`crate::gamma_q`]. When the
smaller tail is the complement, the result is formed with `ln(1 − T)` of the smaller tail `T`
without cancellation.

# Examples

```
// Q(1, x) = e^(−x), so ln Q(1, 1) = −1
let v = scioxide_special::gamma_log_q(1.0_f64, 1.0)?;
assert!((v + 1.0).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The tails are within 32 ulps for `f64` and 1 ulp for `f32` on the tested range; the detailed
form returns an error estimate for the logarithm. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0` or `x < 0`.
- `DomainError` (`Undefined`) if `a = 0` and `x = 0`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_log_q [S: RealScalar] (a: S, x: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_log_q_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        gamma_log_q_evaluate(a, x, options, context)
    }
    scoped pub gamma_log_q_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn gamma_log_q_evaluate<S: RealScalar>(
    a: S,
    x: S,
    options: GammaOptions,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || gamma_log_q_evaluate::<f64>(a.into_f64(), x.into_f64(), options, context.as_deref_mut()),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, branch) =
        crate::gamma_beta::incomplete_gamma::probability_log_tail(a, x, true, &mut w)?;
    w.check()?;
    tail_finish(v, &w, branch)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complementary regularized incomplete beta function `1 − I_x(a, b) = I_{1−x}(b, a)`.

`I_x(a, b)` is defined at [`crate::beta_inc`]; `a, b ≥ 0` and `0 ≤ x ≤ 1`, with the same
zero-shape limits. For positive shapes the result is 1 at `x = 0` and 0 at `x = 1`. This is the
survival function of the beta distribution.

The upper tail is evaluated directly, not as `1 − I_x(a, b)`, so small upper tails keep their
relative accuracy.

# Examples

```
// 1 − I_{1/2}(2, 3) = 1 − 11/16 = 5/16
let q = scioxide_special::beta_inc_q(2.0_f64, 3.0, 0.5)?;
assert!((q - 0.3125).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range; the largest error observed on a broader 4,160-point `f64` sample is 12 ulps. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0`, `b < 0`, or `x` is outside `[0, 1]`.
- `DomainError` (`Undefined`) if `a = b = 0`, if `a = 0` and `x = 0`, or if `b = 0` and `x = 1`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_q [S: RealScalar] (a: S, b: S, x: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_q_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        beta_inc_q_evaluate(a, b, x, options, context)
    }
    scoped pub beta_inc_q_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn beta_inc_q_evaluate<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    options: BetaOptions,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || {
            beta_inc_q_evaluate::<f64>(
                a.into_f64(),
                b.into_f64(),
                x.into_f64(),
                options,
                context.as_deref_mut(),
            )
        },
        |wide| wide.narrow(),
    ) {
        return result;
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, branch) = crate::gamma_beta::beta::probability_tail(a, b, x, true, false, &mut w)?;
    w.check()?;
    tail_finish(v, &w, branch)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln I_x(a, b)`, the logarithm of the regularized incomplete beta function, without
forming `I_x`.

`I_x(a, b)` is defined at [`crate::beta_inc`]; `a, b ≥ 0` and `0 ≤ x ≤ 1`, with the same
zero-shape limits. For positive shapes the result is `−∞` at `x = 0`, returned as a value, and 0
at `x = 1`.

# Examples

```
// I_x(1, 1) = x, so ln I_{1/2}(1, 1) = −ln 2
let v = scioxide_special::beta_inc_log(1.0_f64, 1.0, 0.5)?;
assert!((v + core::f64::consts::LN_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The tails are within 32 ulps for `f64` and 1 ulp for `f32` on the tested range; the detailed
form returns an error estimate for the logarithm. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0`, `b < 0`, or `x` is outside `[0, 1]`.
- `DomainError` (`Undefined`) if `a = b = 0`, if `a = 0` and `x = 0`, or if `b = 0` and `x = 1`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_log [S: RealScalar] (a: S, b: S, x: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        beta_inc_log_evaluate(a, b, x, options, context)
    }
    scoped pub beta_inc_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn beta_inc_log_evaluate<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    options: BetaOptions,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || {
            beta_inc_log_evaluate::<f64>(
                a.into_f64(),
                b.into_f64(),
                x.into_f64(),
                options,
                context.as_deref_mut(),
            )
        },
        |wide| wide.narrow(),
    ) {
        return result;
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, branch) = crate::gamma_beta::beta::probability_tail(a, b, x, false, true, &mut w)?;
    w.check()?;
    tail_finish(v, &w, branch)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes `ln(1 − I_x(a, b))`, the logarithm of the complementary regularized incomplete beta
function, without forming `1 − I_x`.

`I_x(a, b)` is defined at [`crate::beta_inc`]; `a, b ≥ 0` and `0 ≤ x ≤ 1`, with the same
zero-shape limits. For positive shapes the result is 0 at `x = 0` and `−∞` at `x = 1`, returned
as a value. The upper tail is evaluated directly.

# Examples

```
// I_x(1, 1) = x, so ln(1 − I_{1/2}(1, 1)) = −ln 2
let v = scioxide_special::beta_inc_log_q(1.0_f64, 1.0, 0.5)?;
assert!((v + core::f64::consts::LN_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The tails are within 32 ulps for `f64` and 1 ulp for `f32` on the tested range; the detailed
form returns an error estimate for the logarithm. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a < 0`, `b < 0`, or `x` is outside `[0, 1]`.
- `DomainError` (`Undefined`) if `a = b = 0`, if `a = 0` and `x = 0`, or if `b = 0` and `x = 1`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_log_q [S: RealScalar] (a: S, b: S, x: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_log_q_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        beta_inc_log_q_evaluate(a, b, x, options, context)
    }
    scoped pub beta_inc_log_q_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

#[doc = "Evaluates the result before projecting it to the scalar type."]
fn beta_inc_log_q_evaluate<S: RealScalar>(
    a: S,
    b: S,
    x: S,
    options: BetaOptions,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || {
            beta_inc_log_q_evaluate::<f64>(
                a.into_f64(),
                b.into_f64(),
                x.into_f64(),
                options,
                context.as_deref_mut(),
            )
        },
        |wide| wide.narrow(),
    ) {
        return result;
    }
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, branch) = crate::gamma_beta::beta::probability_tail(a, b, x, true, true, &mut w)?;
    w.check()?;
    tail_finish(v, &w, branch)
}

/// Returns a tail or log tail with the probability kernels' envelope; the
/// infinite log-zero limit is exact.
fn tail_finish<S: RealScalar>(
    v: S,
    w: &Work,
    branch: AlgorithmComponent,
) -> Result<Evaluation<S>, SciOxideError> {
    finish(
        v,
        if v.is_finite() {
            error(v)?
        } else {
            Bound::exact()
        },
        w,
        branch,
    )
}

// A logarithmic coordinate spans tiny positive quantiles without thousands of
// linear bisections. Beta uses a logit coordinate for two-sided endpoint access.
fn coordinate<S: RealScalar>(t: S, beta: bool) -> S {
    if !beta {
        t.exp()
    } else if t < S::ZERO {
        let e = t.exp();
        e / (S::ONE + e)
    } else {
        S::ONE / (S::ONE + (-t).exp())
    }
}
/// Accurate normalization is prepared only when a scalar residual cannot
/// establish the next bracket update or the requested output accuracy.
fn inverse_normalization(a: f64, b: Option<f64>) -> Result<(Wide, Wide), SciOxideError> {
    Ok(match b {
        Some(b) => (super::beta::ln_beta_wide(a, b)?, Wide::ZERO),
        None => {
            let normalization = Normalization::wide(a);
            (normalization.value, normalization.shifted)
        }
    })
}

/// Bounded gamma (`b = None`) or beta quantile;
#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse<S: RealScalar>(
    a: S,
    b: Option<S>,
    p: S,
    log: bool,
    upper: bool,
    budget: scioxide_core::Budget,
    method: AlgorithmComponent,
    mut context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || {
            inverse::<f64>(
                a.into_f64(),
                b.map(RealScalar::into_f64),
                p.into_f64(),
                log,
                upper,
                budget,
                method,
                context.as_deref_mut(),
            )
        },
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(a, "shape parameter a must be finite")?;
    require_finite_param(p, "probability parameter p must be finite")?;
    if a <= S::ZERO {
        return Err(ERRORS.outside(scioxide_core::argument!("a")));
    }
    if let Some(b) = b {
        require_finite_param(b, "shape parameter b must be finite")?;
        if b <= S::ZERO {
            return Err(ERRORS.outside(scioxide_core::argument!("b")));
        }
    }
    if (log && p > S::ZERO) || (!log && (p < S::ZERO || p > S::ONE)) {
        return Err(ERRORS.outside(scioxide_core::argument!("p")));
    }
    let mut w = Work::with_context(budget.get(), context);
    let evaluation = search(a, b, p, log, upper, method, &mut w);
    w.check()?;
    evaluation
}

/// Safeguarded inverse search, charged to the caller's allowance.
fn search<S: RealScalar>(
    a: S,
    b: Option<S>,
    p: S,
    log: bool,
    mut upper: bool,
    method: AlgorithmComponent,
    w: &mut Work,
) -> Result<Evaluation<S>, SciOxideError> {
    if p == S::ZERO && !log || (if log { p == S::ZERO } else { p == S::ONE }) {
        let at_upper = (if log { true } else { p == S::ONE }) != upper;
        let x = if at_upper {
            if b.is_some() { S::ONE } else { S::INFINITY }
        } else {
            S::ZERO
        };
        return finish(x, Bound::exact(), w, method);
    }
    // Seeds, scalar residuals and clear endpoint decisions need only a binary64
    // target. Its two-word logarithm is formed when an endpoint decision is
    // close or the precise phase starts.
    let reflected_input = !log && p > S::HALF;
    if reflected_input {
        // Sterbenz subtraction is exact here. Reflect the input before
        // taking a logarithm instead of exponentiating its logarithm again.
        upper = !upper;
    }
    let complemented_log = log && p.into_f64() > -core::f64::consts::LN_2;
    if complemented_log {
        upper = !upper;
    }
    // The probability whose logarithm is the target, when it is exact.
    let target_probability = (!log).then(|| {
        if reflected_input {
            (S::ONE - p).into_f64()
        } else {
            p.into_f64()
        }
    });
    let exact_target = || match target_probability {
        Some(probability) => Wide::from(probability).ln(),
        None if complemented_log => wide_log1mexp(Wide::from(p.into_f64())),
        None => Wide::from(p.into_f64()),
    };
    let mut target_exact = log && !complemented_log;
    let mut target = match target_probability {
        Some(probability) => Wide::from(probability.ln()),
        None if complemented_log => Wide::from(crate::error_probability::probability::log1mexp(
            p.into_f64(),
        )?),
        None => Wide::from(p.into_f64()),
    };
    let target_value = S::try_from_f64_lossy(target.value())?;
    if b == Some(a) && !log && p == S::HALF {
        return finish(S::HALF, Bound::exact(), w, method);
    }
    if b.is_none() && a == S::ONE {
        w.step();
        let target = exact_target();
        let x = if upper {
            -target.value()
        } else {
            -wide_log1mexp(target).value()
        };
        let x = S::try_from_f64_lossy(x)?;
        return finish(x, error(x)?, w, method);
    }
    let af = a.into_f64();
    let bf = b.map(RealScalar::into_f64);
    // Small-shape and extreme-tail endpoint decisions need their low words
    // before division by a shape; ordinary seeds only need scalar accuracy.
    let scalar_normalization = (af >= 0.125 && bf.is_none_or(|b| b >= 0.125) && target.hi >= -36.0)
        .then(|| match bf {
            Some(b) => super::beta::probability_normalization(af, b),
            None => super::incomplete_gamma::probability_normalization(af),
        })
        .filter(|(value, error)| {
            value.is_finite()
                && error.is_finite()
                && *error <= f64::EPSILON.sqrt() * value.abs().max(1.0)
        });
    let (
        mut normalization,
        mut shifted_normalization,
        normalization_error,
        mut normalized_precisely,
    ) = if let Some((value, error)) = scalar_normalization {
        let shifted = if b.is_none() {
            if af >= 1.0 {
                value + af.ln()
            } else {
                super::gamma::real::ln_positive(af + 1.0)
            }
        } else {
            0.0
        };
        (Wide::from(value), Wide::from(shifted), error, false)
    } else {
        let (value, shifted) = inverse_normalization(af, bf)?;
        (value, shifted, 0.0, true)
    };
    crate::support::check_finite(normalization.value())?;
    let mut lo = S::ZERO.next_after(S::ONE).ln() - S::LN_2;
    let mut hi = if b.is_some() {
        -(S::EPSILON * S::try_from_f64_lossy(0.25)?).ln()
    } else {
        let scale = a.max(-target_value).max(S::ONE);
        (scale.ln() + real::<S>(4.)?).min(S::MAX.ln())
    };
    let (lo_value, hi_value) = (lo.into_f64(), hi.into_f64());
    // Check both output endpoints independently of the selected tail. At
    // these coordinates the power-series correction is below the lattice.
    // A binary64 argument decides unless its rounding margin (sixteen units
    // of every summand, plus the normalization uncertainty) reaches the
    // endpoint; then the original two-word comparison decides.
    let margin = |terms: &[f64]| {
        16.0 * f64::EPSILON * terms.iter().map(|term| term.abs()).sum::<f64>() + normalization_error
    };
    let lower_target_value = if upper {
        crate::error_probability::probability::log1mexp(target.value())?
    } else {
        target.value()
    };
    let lower_shift = match bf {
        Some(_) => normalization.value() + af.ln(),
        None => shifted_normalization.value(),
    };
    let lower_value = lower_target_value + lower_shift;
    let lower_bound = af * lo_value;
    if lower_value - margin(&[lower_target_value, lower_shift, lower_bound]) < lower_bound {
        if !target_exact {
            target = exact_target();
            target_exact = true;
        }
        let lower_target = if upper { wide_log1mexp(target) } else { target };
        let lower_argument = lower_target
            + match b {
                Some(_) => normalization + Wide::from(af).ln(),
                None => shifted_normalization,
            };
        // Compare before division: the logarithm of an unrepresentable
        // quantile can exceed the scalar range for very small, but valid,
        // shapes.
        if lower_argument.value() < Wide::from(af).mul_f64(lo_value).value() {
            w.step();
            return finish(S::ZERO, Bound::derived(S::MIN_SUBNORMAL), w, method);
        }
    }
    if let Some(b) = bf {
        let upper_target_value = if upper {
            target.value()
        } else {
            crate::error_probability::probability::log1mexp(target.value())?
        };
        let upper_shift = normalization.value() + b.ln();
        let upper_bound = -b * hi_value;
        if upper_target_value + upper_shift
            - margin(&[upper_target_value, upper_shift, upper_bound])
            < upper_bound
        {
            if !target_exact {
                target = exact_target();
                target_exact = true;
            }
            let upper_target = if upper { target } else { wide_log1mexp(target) };
            let mut upper_argument = upper_target + normalization + Wide::from(b).ln();
            // Recheck endpoint rounding with the accurate normalization even
            // when only the uncertainty interval reaches the upper endpoint.
            if !normalized_precisely && upper_argument.value() - normalization_error < b * -hi_value
            {
                (normalization, shifted_normalization) = inverse_normalization(af, bf)?;
                normalized_precisely = true;
                upper_argument = upper_target + normalization + Wide::from(b).ln();
            }
            if upper_argument.value() < Wide::from(b).mul_f64(-hi_value).value() {
                w.step();
                return finish(S::ONE, Bound::derived(S::EPSILON * S::HALF), w, method);
            }
        }
    }
    let seed = match bf {
        Some(b) => {
            let shape = if upper { b } else { af };
            let l = (target.value() + normalization.value() + shape.ln()) / shape;
            if upper { -l } else { l }
        }
        None if upper => a.max(-target_value).ln().into_f64(),
        None => (target.value() + shifted_normalization.value()) / af,
    };
    let lower_argument = lower_value;
    let normal_seed = bf.is_none_or(|b| b >= 1.0);
    let seed = if normal_seed && target_value > real::<S>(-36.)? && a > S::ONE && w.step() {
        let z = crate::error_probability::probability::normal_quantile_evaluate(
            target_value.exp(),
            ProbabilityOptions::default(),
            None,
        )?
        .value()?
            * if upper { -S::ONE } else { S::ONE };
        let (af, z) = (a.into_f64(), z.into_f64());
        match b {
            None => {
                // DiDonato--Morris (1986), equation 31.
                let root = af.sqrt();
                let square = z * z;
                let w = af + z * root + (square - 1.0) / 3.0 + z * (square - 7.0) / (36.0 * root)
                    - (3.0 * square * square + 7.0 * square - 16.0) / (810.0 * af)
                    + z * (9.0 * square * square + 256.0 * square - 433.0) / (38880.0 * af * root);
                if w > 0.0 { w.ln() } else { seed }
            }
            Some(b) if b > S::ONE && a + b >= real::<S>(10.)? => {
                // Temme (1992), sections 3 and 5: first correction to eta
                // and the small-eta inversion. Halley supplies the remaining bits.
                let r = af + b.into_f64();
                let m = af / r;
                let sc = (m * (1.0 - m)).sqrt();
                let eta0 = z / r.sqrt();
                let eta = eta0
                    + ((2.0 * m - 1.0) / (3.0 * sc)
                        + eta0 * (1.0 + 5.0 * m - 5.0 * m * m) / (36.0 * sc * sc))
                        / r;
                let x = m + eta
                    * (sc
                        + eta
                            * ((1.0 - 2.0 * m) / 3.0
                                + eta
                                    * ((1.0 - 13.0 * m + 13.0 * m * m) / (36.0 * sc)
                                        + eta * (1.0 + m * (21.0 + m * (-69.0 + 46.0 * m)))
                                            / (270.0 * sc * sc))));
                if eta.abs() < 0.7 && x > 0.0 && x < 1.0 {
                    x.ln() - (-x).ln_1p()
                } else {
                    seed
                }
            }
            Some(b) => {
                // Abramowitz and Stegun 26.5.22 for moderate shapes a, b >= 1,
                // in the logit coordinate: x = a / (a + b e^(2w)).
                let (b, s) = (b.into_f64(), -z);
                let lambda = (s * s - 3.0) / 6.0;
                let (left, right) = (1.0 / (2.0 * af - 1.0), 1.0 / (2.0 * b - 1.0));
                let h = 2.0 / (left + right);
                let w = s * (lambda + h).sqrt() / h
                    - (right - left) * (lambda + 5.0 / 6.0 - 2.0 / (3.0 * h));
                let t = af.ln() - b.ln() - 2.0 * w;
                if t.is_finite() { t } else { seed }
            }
        }
    } else {
        seed
    };
    let seed = if a < S::ONE || b.is_some_and(|b| b < S::ONE) {
        // Small-shape quantiles can be near zero even for an upper-tail
        // request; choose the endpoint from both shape and probability.
        let left = lower_argument / af;
        match bf {
            Some(b) => {
                // Both power laws bound the root from above when a >= 1 >= b:
                // I >= x^a / (aB) and 1 - I <= (1-x)^b / (bB).
                let upper_root = if af >= 1.0 && b <= 1.0 {
                    let upper_log = if upper {
                        target.value()
                    } else {
                        crate::error_probability::probability::log1mexp(target.value())?
                    };
                    -((upper_log + normalization.value() + b.ln()) / b).exp_m1()
                } else {
                    f64::INFINITY
                };
                let root = left.exp().min(upper_root);
                if root > 0.0 && root < 1.0 {
                    root.ln() - (-root).ln_1p()
                } else if left < -core::f64::consts::LN_2 {
                    left - (-left.exp()).ln_1p()
                } else {
                    seed
                }
            }
            None if left < -core::f64::consts::LN_2 => left,
            None => seed,
        }
    } else {
        seed
    };
    // Half-shape beta inverses are Student-t coordinates. The same A&S
    // 26.7.5 expansion used by the large-degree Student-t quantile supplies a
    // close seed at moderate degrees too. It is only a seed: the unchanged
    // safeguarded beta residual and output-lattice tests supply every final bit.
    let student = match bf {
        Some(b) if af == 0.5 && (8.0..=2048.0).contains(&b) => Some((
            2.0 * b,
            if upper {
                target.value().exp()
            } else {
                -target.value().exp_m1()
            },
            1.0,
        )),
        Some(0.5) if (8.0..=2048.0).contains(&af) => Some((
            2.0 * af,
            if upper {
                -target.value().exp_m1()
            } else {
                target.value().exp()
            },
            -1.0,
        )),
        _ => None,
    };
    let seed = if let Some((degrees, tail, sign)) = student
        && (1e-6..=0.5).contains(&tail)
        && w.remaining > 5
    {
        w.charge(5)?;
        let z = crate::error_probability::probability::normal_quantile_evaluate(
            1.0 - 0.5 * tail,
            ProbabilityOptions::default(),
            None,
        )?
        .value()?;
        let q = z * z;
        let r = 1.0 / degrees;
        let a = (q + 1.0) / 4.0;
        let b = ((5.0 * q + 16.0) * q + 3.0) / 96.0;
        let c = (((3.0 * q + 19.0) * q + 17.0) * q - 15.0) / 384.0;
        let d = ((((79.0 * q + 776.0) * q + 1482.0) * q - 1920.0) * q - 945.0) / 92160.0;
        let t = z * (1.0 + r * (a + r * (b + r * (c + r * d))));
        sign * (2.0 * t.ln() - degrees.ln())
    } else {
        seed
    };
    // One linear normalization serves every scalar beta tail of the search.
    let linear_normalization = match bf {
        Some(b) => crate::gamma_beta::beta::linear_normalization(af, b),
        None => crate::gamma_beta::incomplete_gamma::linear_normalization(af),
    };
    let mut proposal = Some(S::try_from_f64_lossy(seed)?);
    let mut x = S::ZERO;
    let mut best_x = S::ZERO;
    let mut best_residual = f64::INFINITY;
    let mut linear = false;
    let mut precise = false;
    while w.step() {
        // A safeguarded Halley proposal already proves the bracket has an
        // interior point. Compute the checked midpoint only for bisection.
        let candidate = match proposal
            .take()
            .filter(|t| t.is_finite() && *t > lo && *t < hi)
        {
            Some(value) => Some(value),
            None => crate::util::strict_bracket_candidate(lo, hi)?,
        };
        if candidate.is_none() && !linear {
            // Adjacent transformed values can span hundreds of output ULPs.
            // Widen their rounded images by one neighbour before refining x.
            lo = coordinate(lo, b.is_some()).next_after(S::ZERO);
            hi = coordinate(hi, b.is_some()).next_after(if b.is_some() {
                S::ONE
            } else {
                S::INFINITY
            });
            linear = true;
            proposal = Some(best_x);
            continue;
        }
        let quantum = (best_x - best_x.next_after(S::ZERO))
            .abs()
            .max(S::ZERO.next_after(S::ONE));
        if linear && (hi - lo).abs() <= real::<S>(4.)? * quantum {
            return finish(
                best_x,
                error(best_x)?.plus(Bound::derived(hi - lo)),
                w,
                method,
            );
        }
        let Some(t) = candidate else {
            return finish(
                best_x,
                error(best_x)?.plus(Bound::derived(hi - lo)),
                w,
                method,
            );
        };
        x = if linear {
            t
        } else {
            coordinate(t, b.is_some())
        };
        if x == S::ZERO || (b.is_some() && x == S::ONE) {
            if x == S::ZERO {
                lo = t;
            } else {
                hi = t;
            }
            continue;
        }
        if w.remaining == 0 {
            w.complete = false;
            break;
        }
        if !precise {
            // (log, its error, power factor, linear value, its relative error).
            let linear_tail = match (b, linear_normalization) {
                (Some(b), Some(normalization)) => crate::gamma_beta::beta::linear_tail(
                    af,
                    b.into_f64(),
                    x.into_f64(),
                    upper,
                    normalization,
                    w,
                )
                .map(|tail| {
                    tail.map(|tail| {
                        let (log, error) = tail.log();
                        (log, error, tail.power, tail.value, tail.linear_error())
                    })
                }),
                (None, Some(normalization)) => crate::gamma_beta::incomplete_gamma::linear_tail(
                    af,
                    x.into_f64(),
                    upper,
                    normalization,
                    w,
                )
                .map(|tail| {
                    tail.map(|tail| {
                        let (log, error) = tail.log();
                        (log, error, tail.power, tail.value, tail.linear_error())
                    })
                }),
                _ => Ok(None),
            };
            let fast = match (b, linear_tail) {
                (_, Err(error)) => Err(error),
                (_, Ok(Some((log, log_error, power, value, error)))) => {
                    Ok((log, log_error, power / value, Some((value, error))))
                }
                (Some(b), Ok(None)) => crate::gamma_beta::beta::probability_log_tail_fast(
                    a.into_f64(),
                    b.into_f64(),
                    x.into_f64(),
                    upper,
                    w,
                    normalization.value(),
                    normalization_error,
                )
                .map(|tail| {
                    (
                        tail.logarithm,
                        tail.error,
                        (tail.power - tail.logarithm).exp(),
                        None,
                    )
                }),
                (None, Ok(None)) => crate::gamma_beta::incomplete_gamma::probability_log_tail_fast(
                    a.into_f64(),
                    x.into_f64(),
                    upper,
                    w,
                    &Normalization {
                        value: normalization,
                        shifted: shifted_normalization,
                        error: normalization_error,
                    },
                )
                .map(|(log, error, power)| (log, error, (power - log).exp(), None)),
            };
            // (log tail, its error, logarithmic derivative magnitude, linear tail).
            let (fast, fast_error, slope, linear_value) = match fast {
                Ok(value) => value,
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::ConvergenceFailure {
                            reason: scioxide_core::ConvergenceReason::IterationLimit,
                            ..
                        }
                    ) =>
                {
                    w.charge(w.remaining)?;
                    w.complete = false;
                    break;
                }
                Err(error) => return Err(error),
            };
            // A linear tail against a linear target forms its log residual
            // directly; rounded logarithms would cost epsilon times their size.
            let (difference, fast_error) = match (linear_value, target_probability) {
                (Some((value, relative)), Some(probability)) => {
                    let difference = ((value - probability) / probability).ln_1p();
                    (difference, relative + f64::EPSILON * difference.abs())
                }
                // An exact logarithmic target only adds the subtraction rounding.
                (Some(_), None) if target_exact => {
                    let difference = fast - target.value();
                    (difference, fast_error + f64::EPSILON * difference.abs())
                }
                _ => (
                    fast - target.value(),
                    fast_error + f64::EPSILON * (fast.abs() + target.value().abs()),
                ),
            };
            let xf = x.into_f64();
            let af = a.into_f64();
            let scale = if b.is_some() { xf * (1.0 - xf) } else { xf };
            let derivative = slope * if upper { -1.0 } else { 1.0 };
            let quantum = (xf - xf.next_down()).max(f64::from_bits(1));
            // A logit or log step dt moves x by at most scale (e^|dt| - 1); the
            // first-order scale |dt| would understate long steps near an endpoint.
            let uncertainty = ((difference.abs() + fast_error) / derivative.abs()).exp_m1() * scale;
            // A sixteen-ULP output bound, the precise residual.s allowance,
            // stays inside the 32-ULP inverse envelope. Uncertain scalar tails
            // still require the precise residual before changing a bracket.
            if derivative.is_finite()
                && derivative != 0.0
                && uncertainty.is_finite()
                && uncertainty <= 16.0 * quantum
            {
                return finish(
                    x,
                    error(x)?.plus(Bound::derived(S::try_from_f64_lossy(uncertainty)?)),
                    w,
                    method,
                );
            }
            // Never update a bracket with a residual whose sign is uncertain.
            // Wide arithmetic is needed only for the final, conditioned correction.
            if !difference.is_finite() || difference.abs() <= fast_error || linear {
                precise = true;
            } else {
                let delta = difference / derivative;
                if delta.is_finite() && (delta * scale).abs() <= 32.0 * quantum {
                    precise = true;
                } else {
                    if (difference < 0.0) != upper {
                        lo = t;
                    } else {
                        hi = t;
                    }
                    let curvature = match b {
                        Some(b) => af * (1.0 - xf) - b.into_f64() * xf,
                        None => af - xf,
                    } - derivative;
                    let denominator = 1.0 - 0.5 * delta * curvature;
                    let step = if denominator > 0.5 && denominator.is_finite() {
                        delta / denominator
                    } else {
                        delta
                    };
                    if step.is_finite() {
                        proposal = Some(t - S::try_from_f64_lossy(step)?);
                    }
                    continue;
                }
            }
        }
        if !normalized_precisely {
            (normalization, shifted_normalization) = inverse_normalization(af, bf)?;
            normalized_precisely = true;
        }
        if !target_exact {
            target = exact_target();
            target_exact = true;
        }
        let actual = match b {
            Some(b) => crate::gamma_beta::beta::probability_log_tail_wide(
                a.into_f64(),
                b.into_f64(),
                x.into_f64(),
                upper,
                w,
                normalization,
            )
            .map(|(tail, power)| (tail, power, 0.0)),
            None => crate::gamma_beta::incomplete_gamma::probability_log_tail_wide(
                a.into_f64(),
                x.into_f64(),
                upper,
                w,
                &Normalization {
                    value: normalization,
                    shifted: shifted_normalization,
                    error: 0.0,
                },
            ),
        };
        let (actual, power, uncertainty) = match actual {
            Ok(value) => value,
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: scioxide_core::ConvergenceReason::IterationLimit,
                        ..
                    }
                ) =>
            {
                w.charge(w.remaining)?;
                w.complete = false;
                break;
            }
            Err(error) => return Err(error),
        };
        if !w.complete {
            break;
        }
        crate::support::check_finite(actual.value())?;
        let difference = (actual - target).value();
        if difference.abs() < best_residual {
            best_residual = difference.abs();
            best_x = x;
        }
        // The tail already evaluated x^a(1-x)^b/B(a,b), or x^a exp(-x)/Gamma(a).
        // Dividing that factor by the selected tail gives its derivative in
        // logit/log coordinates without repeating logarithms or normalization.
        let scale = if b.is_some() { x * (S::ONE - x) } else { x };
        let derivative = (power - actual).exp() * if upper { -1.0 } else { 1.0 };
        let delta = difference / derivative;
        if !delta.is_finite() {
            // A Newton proposal can leave the representable derivative range
            // far from the root. Keep the bracket and bisect instead.
            if (difference < 0.0) != upper {
                lo = t;
            } else {
                hi = t;
            }
            continue;
        }
        let output_delta = S::try_from_f64_lossy(delta)? * scale;
        let quantum = (x - x.next_after(S::ZERO))
            .abs()
            .max(S::ZERO.next_after(S::ONE));
        let uncertainty = S::try_from_f64_lossy(uncertainty)?;
        // The compensated exponent leaves a bounded scalar-series error of
        // at most eight output ULPs. Sixteen allows completion when the
        // residual interval includes zero, without guessing its sign.
        let allowance = if uncertainty > S::ZERO {
            real::<S>(16.0)?
        } else {
            S::ONE
        };
        if output_delta.abs() + uncertainty <= allowance * quantum {
            return finish(
                x,
                error(x)?.plus(Bound::derived(output_delta.abs() + uncertainty)),
                w,
                method,
            );
        }
        if (difference < 0.0) != upper {
            lo = t;
        } else {
            hi = t;
        }
        if linear {
            proposal = Some(x - output_delta);
        } else {
            let curvature = match b {
                Some(b) => a * (S::ONE - x) - b * x,
                None => a - x,
            }
            .into_f64()
                - derivative;
            let correction = 1.0 - 0.5 * delta * curvature;
            let step = if correction > 0.5 && correction.is_finite() {
                delta / correction
            } else {
                delta
            };
            let next = t - S::try_from_f64_lossy(step)?;
            if next == t || output_delta.abs() <= real::<S>(32.)? * quantum {
                lo = coordinate(lo, b.is_some()).next_after(S::ZERO);
                hi = coordinate(hi, b.is_some()).next_after(if b.is_some() {
                    S::ONE
                } else {
                    S::INFINITY
                });
                linear = true;
                proposal = Some(x - output_delta);
            } else {
                proposal = Some(next);
            }
        }
    }
    finish(x, error(x)?, w, method)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of the regularized upper incomplete gamma function: the `x ≥ 0` with
`Q(a, x) = p`.

`Q` is defined at [`crate::gamma_q`]; `a > 0`. This is the inverse survival function of the
gamma distribution with shape `a` and unit scale. `p = 0` gives `x = +∞`, and `p = 1` gives
`x = 0`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln x`, safeguarded by a bracket. The final steps use double-word tails and stop when
the result is accurate to the precision of the scalar type. For `a = 1` the inverse is evaluated
in closed form.

# Examples

```
// Q(1, x) = e^(−x), so Q(1, ln 2) = 1/2
let x = scioxide_special::gamma_q_inv(1.0_f64, 0.5)?;
assert!((x - core::f64::consts::LN_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range, which includes
shapes `20 < a ≤ 10¹⁰`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, or `p` is outside `[0, 1]`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_q_inv [S: RealScalar] (a: S, p: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_q_inv_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(gamma_q_inv));
        inverse(
            a,
            None,
            p,
            false,
            true,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub gamma_q_inv_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `x ≥ 0` with `ln P(a, x) = p`, the inverse of [`crate::gamma_log_p`].

`P` is defined at [`crate::gamma_p`]; `a > 0` and `p ≤ 0`. Use this form for probabilities below
the range of the scalar type. `p = 0` gives `x = +∞`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln x`, safeguarded by a bracket. The final steps use double-word tails and stop when
the result is accurate to the precision of the scalar type. For `a = 1` the inverse is evaluated
in closed form.

# Examples

```
// P(1, ln 2) = 1/2
let x = scioxide_special::gamma_p_inv_log(1.0_f64, -core::f64::consts::LN_2)?;
assert!((x - core::f64::consts::LN_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range, which includes
shapes `20 < a ≤ 10¹⁰`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0` or `p > 0`.
- `NonFinite` if `p = −∞`, the logarithm of probability zero.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_p_inv_log [S: RealScalar] (a: S, p: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_p_inv_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(gamma_p_inv_log));
        inverse(
            a,
            None,
            p,
            true,
            false,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub gamma_p_inv_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `x ≥ 0` with `ln Q(a, x) = p`, the inverse of [`crate::gamma_log_q`].

`Q` is defined at [`crate::gamma_q`]; `a > 0` and `p ≤ 0`. Use this form for upper-tail
probabilities below the range of the scalar type. `p = 0` gives `x = 0`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln x`, safeguarded by a bracket. The final steps use double-word tails and stop when
the result is accurate to the precision of the scalar type. For `a = 1` the inverse is evaluated
in closed form.

# Examples

```
// Q(1, ln 2) = 1/2
let x = scioxide_special::gamma_q_inv_log(1.0_f64, -core::f64::consts::LN_2)?;
assert!((x - core::f64::consts::LN_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range, which includes
shapes `20 < a ≤ 10¹⁰`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0` or `p > 0`.
- `NonFinite` if `p = −∞`, the logarithm of probability zero.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_q_inv_log [S: RealScalar] (a: S, p: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_q_inv_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(gamma_q_inv_log));
        inverse(
            a,
            None,
            p,
            true,
            true,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub gamma_q_inv_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of the complementary regularized incomplete beta function: the `x` in
`[0, 1]` with `1 − I_x(a, b) = p`.

`1 − I_x` is [`crate::beta_inc_q`]; `a > 0` and `b > 0`. This is the inverse survival function
of the beta distribution. `p = 0` gives `x = 1`, and `p = 1` gives `x = 0`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln(x/(1 − x))`, safeguarded by a bracket. The final steps use double-word tails and
stop when the result is accurate to the precision of the scalar type.

# Examples

```
// 1 − I_x(1, 1) = 1 − x
let x = scioxide_special::beta_inc_q_inv(1.0_f64, 1.0, 0.5)?;
assert!((x - 0.5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, `b ≤ 0`, or `p` is outside `[0, 1]`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_q_inv [S: RealScalar] (a: S, b: S, p: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_q_inv_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(beta_inc_q_inv));
        inverse(
            a,
            Some(b),
            p,
            false,
            true,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub beta_inc_q_inv_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `x` in `[0, 1]` with `ln I_x(a, b) = p`, the inverse of [`crate::beta_inc_log`].

`I_x` is defined at [`crate::beta_inc`]; `a > 0`, `b > 0` and `p ≤ 0`. Use this form for
probabilities below the range of the scalar type. `p = 0` gives `x = 1`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln(x/(1 − x))`, safeguarded by a bracket. The final steps use double-word tails and
stop when the result is accurate to the precision of the scalar type.

# Examples

```
// I_x(1, 1) = x, so ln I_0.5(1, 1) = −ln 2
let x = scioxide_special::beta_inc_inv_log(1.0_f64, 1.0, -core::f64::consts::LN_2)?;
assert!((x - 0.5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, `b ≤ 0` or `p > 0`.
- `NonFinite` if `p = −∞`, the logarithm of probability zero.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_inv_log [S: RealScalar] (a: S, b: S, p: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_inv_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(beta_inc_inv_log));
        inverse(
            a,
            Some(b),
            p,
            true,
            false,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub beta_inc_inv_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the `x` in `[0, 1]` with `ln(1 − I_x(a, b)) = p`, the inverse of
[`crate::beta_inc_log_q`].

`I_x` is defined at [`crate::beta_inc`]; `a > 0`, `b > 0` and `p ≤ 0`. Use this form for
upper-tail probabilities below the range of the scalar type. `p = 0` gives `x = 0`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln(x/(1 − x))`, safeguarded by a bracket. The final steps use double-word tails and
stop when the result is accurate to the precision of the scalar type.

# Examples

```
// 1 − I_x(1, 1) = 1 − x, so the inverse of −ln 2 is 1/2
let x = scioxide_special::beta_inc_q_inv_log(1.0_f64, 1.0, -core::f64::consts::LN_2)?;
assert!((x - 0.5).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, `b ≤ 0` or `p > 0`.
- `NonFinite` if `p = −∞`, the logarithm of probability zero.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_q_inv_log [S: RealScalar] (a: S, b: S, p: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_q_inv_log_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        let method = scioxide_core::algorithm_component!(stringify!(beta_inc_q_inv_log));
        inverse(
            a,
            Some(b),
            p,
            true,
            true,
            options.iteration_budget(),
            method,
            context,
        )
    }
    scoped pub beta_inc_q_inv_log_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of the regularized lower incomplete gamma function: the `x ≥ 0` with
`P(a, x) = p`.

`P` is defined at [`crate::gamma_p`]. This is the quantile function of the gamma distribution
with shape `a > 0` and unit scale. `p = 0` gives `x = 0`, and `p = 1` gives `x = +∞`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln x`, safeguarded by a bracket. The final steps use double-word tails and stop when
the result is accurate to the precision of the scalar type. For `a = 1` the inverse is evaluated
in closed form.

# Examples

```
// P(1, x) = 1 − e^(−x), so P(1, ln 2) = 1/2
let x = scioxide_special::gamma_p_inv(1.0_f64, 0.5)?;
assert!((x - core::f64::consts::LN_2).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range, which includes
shapes `20 < a ≤ 10¹⁰`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, or `p` is outside `[0, 1]`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]

    pub gamma_p_inv [S: RealScalar] (a: S, p: S) -> S
    where [];
    options options: GammaOptions;
    detailed gamma_p_inv_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        gamma_p_inv_evaluate(a, p, options, context)
    }
    scoped pub gamma_p_inv_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the result before projecting it to the scalar type.
fn gamma_p_inv_evaluate<S: RealScalar>(
    a: S,
    p: S,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::gamma_beta::probability_tails::inverse(
        a,
        None,
        p,
        false,
        false,
        options.iteration_budget(),
        algorithm_component!("gamma_p_inv"),
        context,
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the inverse of the regularized incomplete beta function: the `x` in `[0, 1]` with
`I_x(a, b) = p`.

`I_x` is defined at [`crate::beta_inc`]. This is the quantile function of the beta distribution
with shapes `a > 0` and `b > 0`. `p = 0` gives `x = 0`, and `p = 1` gives `x = 1`.

The search starts from an asymptotic initial approximation and takes Halley steps in the
coordinate `ln(x/(1 − x))`, safeguarded by a bracket. The final steps use double-word tails and
stop when the result is accurate to the precision of the scalar type.

# Examples

```
// I_x(1, 1) = x
let x = scioxide_special::beta_inc_inv(1.0_f64, 1.0, 0.5)?;
assert!((x - 0.5).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested range. See the [crate
accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `a ≤ 0`, `b ≤ 0`, or `p` is outside `[0, 1]`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]

    pub beta_inc_inv [S: RealScalar] (a: S, b: S, p: S) -> S
    where [];
    options options: BetaOptions;
    detailed beta_inc_inv_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        beta_inc_inv_evaluate(a, b, p, options, context)
    }
    scoped pub beta_inc_inv_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the result before projecting it to the scalar type.
fn beta_inc_inv_evaluate<S: RealScalar>(
    a: S,
    b: S,
    p: S,
    options: BetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    crate::gamma_beta::probability_tails::inverse(
        a,
        Some(b),
        p,
        false,
        false,
        options.iteration_budget(),
        algorithm_component!("beta_inc_inv"),
        context,
    )
}
