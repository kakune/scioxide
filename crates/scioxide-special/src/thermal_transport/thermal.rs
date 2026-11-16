//! Real Debye and Fermi–Dirac integrals with scaled values and shared work budgets.
use crate::ValueWithError;
use crate::expansions::{chebyshev, nonnegative};
use crate::result::{Evaluation, Forms, Publish};
use crate::support::Work;
use crate::thermal_transport::thermal_coefficients as tables;
use crate::{
    DebyeOptions, FermiDiracOptions, ScalarRepresentation, SpecialFunctionResult,
    SpecialFunctionTermination,
};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, DomainReason, EstimateReliability, EvaluationCounts, LogValue,
    RealScalar, ScaledValue, SciOxideError, algorithm_component,
};

const EPS: f64 = f64::EPSILON;

// Numerical bookkeeping is local to this family; public budgets and results are core contracts.

fn table(
    work: &mut Work<'_, '_, '_>,
    x: f64,
    coefficients: &[f64],
) -> Result<(f64, f64, bool), SciOxideError> {
    let (value, error) = chebyshev(x, coefficients, work)?;
    let complete = work.complete;
    Ok((value, error, complete))
}

fn outside(parameter: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::domain_error(
        parameter,
        DomainReason::OutsideDomain,
        "parameter outside the documented thermal-integral domain",
    )
}

/// Evaluation of one thermal integral.
type ThermalEvaluation<S> = Evaluation<S, ThermalForm>;

/// Thermal forms: an `f64` value with its logarithm, or the rounded-zero limit of an incomplete
/// integral below every representation.
#[derive(Clone, Copy)]
enum ThermalForm {
    Report {
        value: f64,
        sign: f64,
        log: Option<f64>,
        relative: f64,
        absolute: f64,
    },
    RoundedZero,
}

/// Natural-log error of a report: none once the relative interval may reach zero; it includes the
/// rounding of the destination logarithm.
fn report_log_error<S: RealScalar>(log: Option<f64>, relative: f64) -> Option<f64> {
    log.filter(|_| relative < 0.5).map(|l| {
        crate::result::log_error(relative, l).unwrap_or(f64::MAX)
            + 7.0 * EPS * l.abs()
            + (S::try_from_f64_lossy(l).map_or(l, RealScalar::into_f64) - l).abs()
    })
}

// Keep the ordinary projection independent of exp(log(value)) rounding. A log
// representation survives underflow and overflow; its error is in log units.
#[allow(clippy::too_many_arguments)]
fn report<S: RealScalar>(
    value: f64,
    sign: f64,
    log: Option<f64>,
    relative: f64,
    absolute: f64,
    method: AlgorithmComponent,
    work: &Work,
    complete: bool,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    // Core conversion rounds before deciding whether the target overflows;
    // values just above binary32 MAX can still round to finite MAX.
    if value.is_nan() {
        let _ = crate::support::check_finite(value)?;
    }
    let projected = if value.is_finite() {
        S::try_from_f64_lossy(value).ok()
    } else {
        None
    };
    let rounding = projected.map_or(0.0, |v| (v.into_f64() - value).abs());
    let representation = match projected {
        None => ScalarRepresentation::Overflow,
        _ => ScalarRepresentation::Finite,
    };
    let termination = if !complete {
        SpecialFunctionTermination::IterationLimit
    } else if absolute == 0.0
        && rounding == 0.0
        && report_log_error::<S>(log, relative).is_none_or(|error| error == 0.0)
    {
        SpecialFunctionTermination::Exact
    } else {
        SpecialFunctionTermination::Converged
    };
    Ok(Evaluation::from_parts(
        projected,
        representation,
        termination,
        EvaluationCounts::from_iterations(work.used),
        AlgorithmId::from_components(algorithm_component!("thermal"), method, 1),
        Some(ThermalForm::Report {
            value,
            sign,
            log,
            relative,
            absolute,
        }),
    ))
}

impl<S: RealScalar> Publish<S> for ThermalForm {
    fn publish(self, projected: Option<S>) -> Result<Forms<S>, SciOxideError> {
        let Self::Report {
            value,
            sign,
            log,
            relative,
            absolute,
        } = self
        else {
            return Ok(Forms {
                value: projected
                    .map(|zero| {
                        ValueWithError::new(
                            zero,
                            S::MIN_SUBNORMAL,
                            None,
                            EstimateReliability::ModelBased,
                        )
                    })
                    .transpose()?,
                scaled: None,
                logarithmic: None,
                derivative: None,
            });
        };
        let cast = S::try_from_f64_lossy;
        let maximum = S::MAX.into_f64();
        let rounding = projected.map_or(0.0, |v| (v.into_f64() - value).abs());
        let floor = if absolute == 0.0 && relative == 0.0 {
            0.0
        } else {
            2.0 * f64::from_bits(1)
        };
        let abs_error = (absolute + rounding + floor).min(maximum);
        let scalar = projected
            .map(|v| {
                ValueWithError::new(
                    v,
                    crate::result::outward(abs_error)?,
                    crate::result::relative_error(v, crate::result::outward(abs_error)?),
                    EstimateReliability::ModelBased,
                )
            })
            .transpose()?;
        let log_error = report_log_error::<S>(log, relative);
        let logarithmic = log
            .zip(log_error)
            .map(|(l, log_error)| {
                ValueWithError::new(
                    LogValue::from_log(cast(sign)?, cast(l)?)?,
                    crate::result::outward(log_error)?,
                    None,
                    EstimateReliability::ModelBased,
                )
            })
            .transpose()?;
        let scaled = match (log, log_error) {
            (Some(l), Some(log_error))
                if relative + log_error < 0.5
                    && l.abs() < (i64::MAX as f64) * core::f64::consts::LN_2 - 2048.0 =>
            {
                let wide = ScaledValue::from_log(sign, l)?;
                let scaled = ScaledValue::from_parts(
                    cast(wide.real().significand())?,
                    wide.real().exponent(),
                )?;
                let relative =
                    crate::result::outward::<S>(relative + log_error + S::EPSILON.into_f64())?;
                Some(crate::result::scaled_estimate(
                    scaled,
                    relative,
                    EstimateReliability::ModelBased,
                )?)
            }
            (None, _) if value == 0.0 => Some(ValueWithError::new(
                ScaledValue::from_value(cast(value)?)?,
                cast(abs_error)?,
                None,
                EstimateReliability::ModelBased,
            )?),
            _ => None,
        };
        Ok(Forms {
            value: scalar,
            scaled,
            logarithmic,
            derivative: None,
        })
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₁(x).

```text
D₁(x) = (1/x) ∫₀ˣ t / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₁(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₁
decreases from 1 at the origin and behaves like `c/x` for large `x`, with
`c = ζ(2) = π²/6 = 1.6449340668482264`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit.

# Examples

```
// D₁(0) = 1
assert!((scioxide_special::debye_1(0.0_f64)? - 1.0).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_1 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_1_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 1, options, context)
    }
    scoped pub debye_1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₂(x).

```text
D₂(x) = (2/x²) ∫₀ˣ t² / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₂(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₂
decreases from 1 at the origin and behaves like `c/x²` for large `x`, with
`c = 2·2! ζ(3) = 4.808227612638377`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit,
working with logarithms so that `x²` is never formed.

# Examples

```
// D₂(0) = 1
assert!((scioxide_special::debye_2(0.0_f64)? - 1.0).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_2 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_2_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 2, options, context)
    }
    scoped pub debye_2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₃(x).

```text
D₃(x) = (3/x³) ∫₀ˣ t³ / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₃(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₃
decreases from 1 at the origin and behaves like `c/x³` for large `x`, with
`c = 3·3! ζ(4) = 19.481818206800487`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit,
working with logarithms so that `x³` is never formed.

# Examples

```
// D₃(x) → π⁴/(5x³) for large x; the exponential tail is negligible at x = 50
let expected = core::f64::consts::PI.powi(4) / (5.0 * 50.0_f64.powi(3));
let v = scioxide_special::debye_3(50.0_f64)?;
assert!((v / expected - 1.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_3 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_3_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 3, options, context)
    }
    scoped pub debye_3_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₄(x).

```text
D₄(x) = (4/x⁴) ∫₀ˣ t⁴ / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₄(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₄
decreases from 1 at the origin and behaves like `c/x⁴` for large `x`, with
`c = 4·4! ζ(5) = 99.5450644937635`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit,
working with logarithms so that `x⁴` is never formed.

# Examples

```
// D₄(0) = 1
assert!((scioxide_special::debye_4(0.0_f64)? - 1.0).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_4 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_4_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 4, options, context)
    }
    scoped pub debye_4_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₅(x).

```text
D₅(x) = (5/x⁵) ∫₀ˣ t⁵ / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₅(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₅
decreases from 1 at the origin and behaves like `c/x⁵` for large `x`, with
`c = 5·5! ζ(6) = 610.4058371906695`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit,
working with logarithms so that `x⁵` is never formed.

# Examples

```
// D₅(0) = 1
assert!((scioxide_special::debye_5(0.0_f64)? - 1.0).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_5 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_5_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 5, options, context)
    }
    scoped pub debye_5_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Debye function D₆(x).

```text
D₆(x) = (6/x⁶) ∫₀ˣ t⁶ / (exp(t) − 1) dt,   x ≥ 0
```

normalized so that `D₆(0) = 1`; A&S §27.1 tabulates the unnormalized integral. D₆
decreases from 1 at the origin and behaves like `c/x⁶` for large `x`, with
`c = 6·6! ζ(7) = 4356.068878289907`.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`. Beyond 4 it subtracts the
exponentially convergent tail `∑ₖ exp(−kx)·(polynomial in 1/(kx))` from the large-`x` limit,
working with logarithms so that `x⁶` is never formed.

# Examples

```
// D₆(0) = 1
assert!((scioxide_special::debye_6(0.0_f64)? - 1.0).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.1."###)]

    pub debye_6 [S: RealScalar] (x: S) -> S
    where [];
    options options: DebyeOptions;
    detailed debye_6_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        debye(x, 6, options, context)
    }
    scoped pub debye_6_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

fn debye<S: RealScalar>(
    input: S,
    n: usize,
    options: DebyeOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    nonnegative(input, scioxide_core::argument!("x"))?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let evaluation = debye_region(input.into_f64(), n, &mut work);
    work.check()?;
    evaluation
}

fn debye_region<S: RealScalar>(
    x: f64,
    n: usize,
    work: &mut Work,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    if x == 0.0 {
        return report(
            1.0,
            1.0,
            Some(0.0),
            0.0,
            0.0,
            algorithm_component!("debye_origin"),
            work,
            true,
        );
    }
    let order = n as f64;
    if x <= 4.0 {
        let c = [
            tables::ADEB1,
            tables::ADEB2,
            tables::ADEB3,
            tables::ADEB4,
            tables::ADEB5,
            tables::ADEB6,
        ][n - 1];
        let (v, e, complete) = table(work, x * x / 8.0 - 1.0, c)?;
        let linear = order * x / (2.0 * (order + 1.0));
        let value = v - linear;
        let error = e + 8.0 * EPS * (v.abs() + linear);
        return report(
            value,
            value.signum(),
            (value > 0.0).then(|| value.ln()),
            if value > 0.0 { error / value } else { 1.0 },
            error,
            algorithm_component!("debye_origin_expansion"),
            work,
            complete,
        );
    }
    let infinity = [
        1.6449340668482264,
        4.808227612638377,
        19.481818206800487,
        99.545_064_493_763_5,
        610.4058371906695,
        4356.068878289907,
    ][n - 1];
    let log_leading = infinity.ln() - order * x.ln();
    let q = (-x).exp();
    let mut tail = 0.0;
    let mut qk = q;
    let mut remainder = 1.0;
    let mut complete = false;
    let mut k = 1;
    // Tail normalized by n*n!*zeta(n+1): never form x^n outside range.
    while work.step() {
        let inverse = 1.0 / (x * k as f64);
        let polynomial = (1..=n).fold(1.0, |p, j| p * j as f64 * inverse + 1.0);
        let term = if qk == 0.0 {
            0.0
        } else {
            (order.ln() + order * x.ln() - infinity.ln() + (qk).ln() - (k as f64).ln()
                + polynomial.ln())
            .exp()
        };
        tail += term;
        remainder = term * q / (1.0 - q);
        if remainder <= EPS * (1.0 - tail) {
            complete = true;
            break;
        }
        qk *= q;
        k += 1;
    }
    let log = log_leading + (-tail).ln_1p();
    let value = (0..n).fold(infinity, |v, _| v / x) * (1.0 - tail);
    let relative = remainder / (1.0 - tail)
        + 32.0 * EPS * (1.0 + log.abs() + x.min(1024.0) * tail / (1.0 - tail));
    report(
        value,
        1.0,
        Some(log),
        relative,
        value * relative,
        algorithm_component!("debye_exponential_tail"),
        work,
        complete,
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Fermi–Dirac integral of order −1, the logistic function.

```text
F₋₁(x) = 1 / (1 + exp(−x))
```

This is the analytic continuation of the gamma-normalized Fermi–Dirac integrals
([`fermi_dirac_0`](crate::fermi_dirac_0)) to `j = −1`, and the derivative of `F₀`. It is
evaluated in closed form, without overflow for either sign of `x`; for very negative `x` the
value underflows to zero and the detailed form keeps its logarithm.

# Examples

```
// F₋₁(0) = 1/2
assert!((scioxide_special::fermi_dirac_m1(0.0_f64)? - 0.5).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_m1 [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_m1_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, -1.0, options, None)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F₀(x) = ln(1 + exp(x)).

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `j = 0` the integral is `ln(1 + exp(x))`, evaluated in closed form as
`max(x, 0) + ln1p(exp(−|x|))`, without overflow. For very negative `x` the value underflows to
zero and the detailed form keeps its logarithm.

# Examples

```
// F₀(0) = ln 2
let v = scioxide_special::fermi_dirac_0(0.0_f64)?;
assert!((v - core::f64::consts::LN_2).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_0 [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_0_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, 0.0, options, None)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of order `j = 1`.

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `x < −1` the method sums the series `∑ₖ (−1)^(k+1) exp(kx)/k^(j+1)`, and for
`−1 ≤ x < 30` it uses Chebyshev expansions on four intervals. For `x ≥ 30` it uses the
exact inversion formula `F_j(x) = polynomial(x) ± F_j(−x)`, valid for integer `j`.

The value underflows to zero for very negative `x` (about `x < −745` in `f64`); the detailed
form keeps its logarithm.

# Examples

```
// F_j(0) = (1 − 2^(−j)) ζ(j + 1) = π²/12
let v = scioxide_special::fermi_dirac_1(0.0_f64)?;
assert!((v - 0.8224670334241132).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `F_j(x) ≈ x^(j+1)/Γ(j + 2)` exceeds the range of the scalar
  type for large `x`; the detailed form still returns the value in logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_1 [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_1_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, 1.0, options, context)
    }
    scoped pub fermi_dirac_1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of order `j = 2`.

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `x < −1` the method sums the series `∑ₖ (−1)^(k+1) exp(kx)/k^(j+1)`, and for
`−1 ≤ x < 30` it uses Chebyshev expansions on four intervals. For `x ≥ 30` it uses the
exact inversion formula `F_j(x) = polynomial(x) ± F_j(−x)`, valid for integer `j`.

The value underflows to zero for very negative `x` (about `x < −745` in `f64`); the detailed
form keeps its logarithm.

# Examples

```
// F_j(0) = (1 − 2^(−j)) ζ(j + 1) = (3/4) ζ(3)
let v = scioxide_special::fermi_dirac_2(0.0_f64)?;
assert!((v - 0.9015426773696957).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `F_j(x) ≈ x^(j+1)/Γ(j + 2)` exceeds the range of the scalar
  type for large `x`; the detailed form still returns the value in logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_2 [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_2_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, 2.0, options, context)
    }
    scoped pub fermi_dirac_2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of order `j = −1/2`.

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `x < −1` the method sums the series `∑ₖ (−1)^(k+1) exp(kx)/k^(j+1)`, and for
`−1 ≤ x < 30` it uses Chebyshev expansions on four intervals. For `x ≥ 30` it uses the
Sommerfeld asymptotic expansion.

The value underflows to zero for very negative `x` (about `x < −745` in `f64`); the detailed
form keeps its logarithm.

# Examples

```
// F_j(0) = (1 − 2^(−j)) ζ(j + 1) = (1 − √2) ζ(1/2)
let v = scioxide_special::fermi_dirac_mhalf(0.0_f64)?;
assert!((v - 0.6048986434216304).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `x` is accepted.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_mhalf [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_mhalf_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, -0.5, options, context)
    }
    scoped pub fermi_dirac_mhalf_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of order `j = 1/2`.

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `x < −1` the method sums the series `∑ₖ (−1)^(k+1) exp(kx)/k^(j+1)`, and for
`−1 ≤ x < 30` it uses Chebyshev expansions on four intervals. For `x ≥ 30` it uses the
Sommerfeld asymptotic expansion.

The value underflows to zero for very negative `x` (about `x < −745` in `f64`); the detailed
form keeps its logarithm.

# Examples

```
// F_j(0) = (1 − 2^(−j)) ζ(j + 1) = (1 − 2^(−1/2)) ζ(3/2)
let v = scioxide_special::fermi_dirac_half(0.0_f64)?;
assert!((v - 0.765147024625408).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `F_j(x) ≈ x^(j+1)/Γ(j + 2)` exceeds the range of the scalar
  type for large `x`; the detailed form still returns the value in logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_half [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_half_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, 0.5, options, context)
    }
    scoped pub fermi_dirac_half_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of order `j = 3/2`.

```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.

For `x < −1` the method sums the series `∑ₖ (−1)^(k+1) exp(kx)/k^(j+1)`, and for
`−1 ≤ x < 30` it uses Chebyshev expansions on four intervals. For `x ≥ 30` it uses the
Sommerfeld asymptotic expansion.

The value underflows to zero for very negative `x` (about `x < −745` in `f64`); the detailed
form keeps its logarithm.

# Examples

```
// F_j(0) = (1 − 2^(−j)) ζ(j + 1) = (1 − 2^(−3/2)) ζ(5/2)
let v = scioxide_special::fermi_dirac_3half(0.0_f64)?;
assert!((v - 0.8671998890121841).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `F_j(x) ≈ x^(j+1)/Γ(j + 2)` exceeds the range of the scalar
  type for large `x`; the detailed form still returns the value in logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_3half [S: RealScalar] (x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_3half_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi(x, 1.5, options, context)
    }
    scoped pub fermi_dirac_3half_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete Fermi–Dirac integral F_j(x) of integer order `j ≥ −101`.

For `j ≥ 0`,
```text
F_j(x) = (1/Γ(j + 1)) ∫₀^∞ t^j / (exp(t − x) + 1) dt = −Li_(j+1)(−exp(x))
```

(DLMF §25.12(iii)), where `Li` is the polylogarithm. This gamma-normalized form satisfies
`F′_j(x) = F_(j−1)(x)`.
`F₋₁(x) = 1/(1 + exp(−x))` is the logistic function, and for `j < −1` the function is its
`(−1 − j)`-th derivative, a rational function of `exp(x)`.

For `j ≥ 1` the method is a series in `exp(x)` for `x < −1` and Chebyshev expansions for
`j ≤ 2` and `x < 30`. Otherwise it uses the alternating series in `exp(x)`, accelerated by
Euler's transform, for `x ≤ 0`, and the exact inversion formula
`F_j(x) = polynomial(x) ± F_j(−x)` for `x > 0`; the polynomial has about `j/2` terms.

# Examples

```
// F₁(0) = π²/12
let v = scioxide_special::fermi_dirac_int(1, 0.0_f64)?;
assert!((v - core::f64::consts::PI.powi(2) / 12.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `j < −101`.
- `ConvergenceFailure` (`IterationLimit`) if `j` is large and `x` is not small compared with `j`, so
  the inversion polynomial needs more terms than the iteration budget of
  [`FermiDiracOptions`](crate::FermiDiracOptions) allows; with the default budget this happens, for
  example, at `j = 2·10⁵`, `x = 10⁵`.
- `NumericalInstability` (`Overflow`) if `F_j(x) ≈ x^(j+1)/Γ(j + 2)` exceeds the range of the scalar
  type; the detailed form still returns the value in logarithmic form.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_int [S: RealScalar] (j: i32, x: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_int_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi_dirac_int_evaluate(j, x, options, context)
    }
    scoped pub fermi_dirac_int_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the detailed result before it is projected and returned.
fn fermi_dirac_int_evaluate<S: RealScalar>(
    j: i32,
    x: S,
    options: FermiDiracOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    if j < -101 {
        return Err(outside(scioxide_core::argument!("j")));
    }
    fermi(x, f64::from(j), options, context)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the incomplete Fermi–Dirac integral of order zero.

```text
F₀(x, b) = ∫_b^∞ dt / (exp(t − x) + 1) = ln(1 + exp(x − b)),   b ≥ 0,
```

for real `x` and lower limit `b`. At `b = 0` it equals [`fermi_dirac_0`](crate::fermi_dirac_0).
The closed form is evaluated without overflow or cancellation; if `x − b` overflows to `−∞` the
result is the limit zero.

# Examples

```
// F₀(x, b) = ln(1 + exp(x − b)); here x = b
let v = scioxide_special::fermi_dirac_inc_0(1.0_f64, 1.0)?;
assert!((v - core::f64::consts::LN_2).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values and reported error bounds, including the scaled
and logarithmic forms, in `f32` and `f64`. The detailed form reports a derived error
bound. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `b < 0`.

# References

- NIST Digital Library of Mathematical Functions, §25.12(iii). <https://dlmf.nist.gov/25.12>"###)]

    pub fermi_dirac_inc_0 [S: RealScalar] (x: S, b: S) -> S
    where [];
    options options: FermiDiracOptions;
    detailed fermi_dirac_inc_0_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        fermi_dirac_inc_0_evaluate(x, b, options)
    }

}

/// Evaluates the detailed result before it is projected and returned.
fn fermi_dirac_inc_0_evaluate<S: RealScalar>(
    x: S,
    b: S,
    options: FermiDiracOptions,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    crate::support::require_finite_param(x.into_f64(), "thermal parameter value must be finite")?;
    nonnegative(b, scioxide_core::argument!("b"))?;
    let difference = x.into_f64() - b.into_f64();
    let work = Work::new(options.iteration_budget().get());
    if !difference.is_finite() {
        // Only the negative difference can overflow: b is nonnegative. Its
        // logarithm is outside scalar range; no finite log estimate is asserted.
        return Ok(Evaluation::from_parts(
            Some(S::ZERO),
            ScalarRepresentation::Finite,
            SpecialFunctionTermination::Limit,
            EvaluationCounts::from_iterations(work.used),
            scioxide_core::algorithm_id!("thermal", "incomplete_zero_rounded_limit", 1),
            Some(ThermalForm::RoundedZero),
        ));
    }
    fermi_kernel(difference, 0.0, options, None)
}

fn fermi<S: RealScalar>(
    x: S,
    j: f64,
    options: FermiDiracOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    crate::support::require_finite_param(x.into_f64(), "thermal parameter value must be finite")?;
    fermi_kernel(x.into_f64(), j, options, context)
}

fn fermi_kernel<S: RealScalar>(
    x: f64,
    j: f64,
    options: FermiDiracOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let evaluation = fermi_region(x, j, &mut work);
    work.check()?;
    evaluation
}

fn fermi_region<S: RealScalar>(
    x: f64,
    j: f64,
    work: &mut Work,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    if j == -1.0 || j == 0.0 {
        if !work.step() {
            return Err(crate::result::exhausted_work(work.used));
        }
        let q = (-x.abs()).exp();
        let value = if j == -1.0 {
            if x < 0.0 {
                q / (1.0 + q)
            } else {
                1.0 / (1.0 + q)
            }
        } else {
            x.max(0.0) + q.ln_1p()
        };
        let log = if x < -36.0 { x } else { value.ln() };
        let relative = 8.0 * EPS;
        return report(
            value,
            1.0,
            Some(log),
            relative,
            value * relative,
            algorithm_component!("fermi_algebraic"),
            work,
            true,
        );
    }
    if j < -1.0 {
        return negative_integer(x, j as i32, work);
    }
    if x < -1.0 {
        let (normalized, error, complete) = negative_series(x, j, work);
        let log = x + normalized.ln();
        let value = x.exp() * normalized;
        let relative = error / normalized + 16.0 * EPS;
        return report(
            value,
            1.0,
            Some(log),
            relative,
            value * relative,
            algorithm_component!("fermi_fugacity_series"),
            work,
            complete,
        );
    }
    if j <= 2.0 && x < 30.0 {
        let row = if j == -0.5 {
            0
        } else if j == 0.5 {
            1
        } else if j == 1.5 {
            2
        } else if j == 1.0 {
            3
        } else {
            4
        };
        let coefficients = [
            [
                tables::FD_MHALF_A,
                tables::FD_MHALF_B,
                tables::FD_MHALF_C,
                tables::FD_MHALF_D,
            ],
            [
                tables::FD_HALF_A,
                tables::FD_HALF_B,
                tables::FD_HALF_C,
                tables::FD_HALF_D,
            ],
            [
                tables::FD_3HALF_A,
                tables::FD_3HALF_B,
                tables::FD_3HALF_C,
                tables::FD_3HALF_D,
            ],
            [
                tables::FD_1_A,
                tables::FD_1_B,
                tables::FD_1_C,
                tables::FD_1_D,
            ],
            [
                tables::FD_2_A,
                tables::FD_2_B,
                tables::FD_2_C,
                tables::FD_2_D,
            ],
        ];
        let (column, t) = if x < 1.0 {
            (0, x)
        } else if x < 4.0 {
            (1, (x - 1.0) * (2.0 / 3.0) - 1.0)
        } else if x < 10.0 {
            (2, (x - 4.0) / 3.0 - 1.0)
        } else {
            (3, x * 0.1 - 2.0)
        };
        let (v, e, complete) = table(work, t, coefficients[row][column])?;
        let scale = if column == 3 { x.powf(j + 1.0) } else { 1.0 };
        let value = v * scale;
        let relative = e / v.abs() + 16.0 * EPS;
        return report(
            value,
            value.signum(),
            (value > 0.0).then(|| value.ln()),
            relative,
            e * scale + 16.0 * EPS * value.abs(),
            algorithm_component!("fermi_central_expansion"),
            work,
            complete,
        );
    }
    if j.fract() == 0.0 {
        integer_positive(x, j as u32, work)
    } else {
        sommerfeld(x, j, work)
    }
}

// Factor exp(x) before summation, retaining the logarithm after scalar underflow.
fn negative_series(x: f64, j: f64, work: &mut Work) -> (f64, f64, bool) {
    let q = x.exp();
    let mut term = 1.0;
    let mut sum = 0.0;
    let mut error = 1.0;
    let mut k = 1;
    while work.step() {
        sum += term;
        let ratio = k as f64 / (k + 1) as f64;
        term *= -q * ratio.powf(j + 1.0);
        error = term.abs();
        if error <= EPS * sum.abs() {
            return (sum, error + 8.0 * EPS * sum.abs(), true);
        }
        k += 1;
    }
    (sum, error, false)
}

// Eta constants are rounded mathematical data, not a separate zeta evaluator.
fn eta(n: u32) -> f64 {
    tables::ETA_POS_INT_TABLE
        .get(n as usize)
        .copied()
        .unwrap_or(1.0)
}

fn opposite(x: f64, j: u32, work: &mut Work) -> (f64, f64, bool) {
    if x == 0.0 {
        return (eta(j + 1), 2.0 * EPS, true);
    }
    if x >= 1.0 {
        let (v, e, c) = negative_series(-x, f64::from(j), work);
        return ((-x).exp() * v, (-x).exp() * e, c);
    }
    // Euler's transform of a completely monotone alternating sequence. N
    // rows leave at most 2^-N of the leading normalized term. Fixed storage.
    let q = (-x).exp();
    let mut differences = [0.0; 64];
    let mut sum = 0.0;
    let mut weight = 0.5;
    let mut power = 1.0;
    let mut error = 1.0;
    for n in 0..64 {
        if !work.step() {
            return ((-x).exp() * sum, error, false);
        }
        differences[n] = power / ((n + 1) as f64).powf(f64::from(j) + 1.0);
        for k in (0..n).rev() {
            if !work.step() {
                return ((-x).exp() * sum, error, false);
            }
            differences[k] -= differences[k + 1];
        }
        sum += weight * differences[0];
        error = weight;
        if weight <= EPS * sum {
            return (
                (-x).exp() * sum,
                (-x).exp() * (error + 64.0 * EPS * sum),
                true,
            );
        }
        weight *= 0.5;
        power *= q;
    }
    (
        (-x).exp() * sum,
        (-x).exp() * (error + 64.0 * EPS * sum),
        true,
    )
}

fn logadd(a: f64, b: f64) -> f64 {
    let hi = a.max(b);
    hi + (-(a - b).abs()).exp().ln_1p()
}

fn integer_positive<S: RealScalar>(
    x: f64,
    j: u32,
    work: &mut Work,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    if x <= 0.0 {
        let (value, error, complete) = opposite(-x, j, work);
        return report(
            value,
            1.0,
            Some(value.ln()),
            error / value,
            error,
            algorithm_component!("fermi_euler_series"),
            work,
            complete,
        );
    }
    // Exact integer inversion identity (DLMF 25.12.13). Build the positive
    // polynomial from its lowest power, in logarithms, without factorials.
    // Unlike an asymptotic truncation this remains valid when order grows.
    let last = j + 1;
    let mut p = last % 2;
    let lx = x.ln();
    let mut lt = if p == 0 { 0.0 } else { lx };
    let mut sum = f64::NEG_INFINITY;
    let mut relative = 0.0;
    let mut complete = false;
    while work.step() {
        let coefficient = 2.0 * eta(last - p);
        let term = lt + coefficient.ln();
        sum = if sum == f64::NEG_INFINITY {
            term
        } else {
            logadd(sum, term)
        };
        if p == last {
            complete = true;
            break;
        }
        let ratio = (x / f64::from(p + 1)) * (x / f64::from(p + 2));
        if ratio < 0.5 {
            let bound = (term - sum).exp() * 2.0 * ratio / (1.0 - ratio);
            if bound <= EPS {
                relative = bound;
                complete = true;
                break;
            }
        }
        p += 2;
        lt += 2.0 * lx - f64::from(p - 1).ln() - f64::from(p).ln();
    }
    if !complete {
        return Err(crate::result::exhausted_work(work.used));
    }
    let (negative, negative_error, negative_complete) = opposite(x, j, work);
    let correction = if j.is_multiple_of(2) {
        negative
    } else {
        -negative
    };
    let log = if correction > 0.0 {
        logadd(sum, correction.ln())
    } else {
        sum + (correction * (-sum).exp()).ln_1p()
    };
    let value = if sum < 700.0 {
        sum.exp() + correction
    } else {
        log.exp()
    };
    relative += negative_error * (-sum).exp() + 64.0 * EPS * (1.0 + sum.abs() + f64::from(p));
    report(
        value,
        1.0,
        Some(log),
        relative,
        value * relative,
        algorithm_component!("fermi_integer_inversion"),
        work,
        negative_complete,
    )
}

fn sommerfeld<S: RealScalar>(
    x: f64,
    j: f64,
    work: &mut Work,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    // Half orders are fixed: x >= 30. Integrated Fermi-edge expansion, with
    // least-term control and an exponentially small endpoint allowance.
    if work.remaining < 13 {
        return Err(crate::result::exhausted_work(work.used));
    }
    work.steps(13)?; // shared log-gamma kernel's positive-argument admission cost
    let gamma = crate::gamma_beta::gamma::ln_gamma_kernel(j + 2.0)?;
    let leading = (j + 1.0) * x.ln() - gamma;
    let mut term = 1.0;
    let mut sum: f64 = 1.0;
    let mut previous = f64::MAX;
    let mut remainder = 1.0;
    let mut complete = false;
    let mut k = 1;
    while work.step() {
        term *= ((j + 3.0 - 2.0 * f64::from(k)) / x) * ((j + 2.0 - 2.0 * f64::from(k)) / x);
        let add = 2.0 * eta(2 * k) * term;
        remainder = add.abs();
        if remainder >= previous {
            complete = true;
            break;
        }
        if remainder <= EPS * sum.abs() {
            complete = true;
            break;
        }
        sum += add;
        previous = remainder;
        k += 1;
    }
    let relative =
        (4.0 * remainder + 4.0 * (-x).exp()) / sum.abs() + 64.0 * EPS * (1.0 + leading.abs());
    let log = leading + sum.ln();
    let value = log.exp();
    report(
        value,
        1.0,
        Some(log),
        relative,
        value * relative,
        algorithm_component!("fermi_sommerfeld"),
        work,
        complete,
    )
}

fn negative_integer<S: RealScalar>(
    x: f64,
    j: i32,
    work: &mut Work,
) -> Result<ThermalEvaluation<S>, SciOxideError> {
    let n = (-(j + 1)) as usize;
    let e = crate::zeta_polylog::negative::fermi(n, x, work)?;
    let v = e.value.re;
    let log = (v != 0.0).then(|| v.abs().ln() + e.scale);
    let value = if e.scale == 0.0 {
        v
    } else {
        crate::support::wide::project(
            crate::support::wide::Wide::from(v),
            crate::support::wide::Wide::from(e.scale),
        )
    };
    let relative = if v == 0.0 { 1.0 } else { e.error / v.abs() };
    let absolute = crate::support::wide::project(
        crate::support::wide::Wide::from(e.error),
        crate::support::wide::Wide::from(e.scale),
    )
    .min(f64::MAX);
    report(
        value,
        v.signum(),
        log,
        relative,
        absolute,
        e.method,
        work,
        true,
    )
}
