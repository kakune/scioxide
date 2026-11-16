//! Real synchrotron functions with exponential tail scaling.

use crate::ValueWithError;
use crate::expansions::{bounded, chebyshev, nonnegative, real};
use crate::result::{Either, Evaluation, Form, Forms, Publish};
use crate::thermal_transport::transport_coefficients as coefficients;
use crate::{SpecialFunctionResult, SynchrotronOptions};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, EstimateReliability, LogValue, RealScalar, ScaledValue,
    SciOxideError, algorithm_component,
};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the first synchrotron function F(x).

```text
F(x) = x ∫ₓ^∞ K_(5/3)(t) dt,   x ≥ 0,
```

where `K_ν` is the modified Bessel function of the second kind. `F(0) = 0`, and
`F(x) ≈ √(πx/2) exp(−x)` for large `x`; the value underflows for `x` beyond about 745 in `f64`,
where [`synchrotron_1_scaled`](crate::synchrotron_1_scaled) stays in range.

The method uses Chebyshev expansions: in `x²/8` with the fractional powers of `x` factored out
for `x ≤ 1`, on `(1, 4]`, and in `1/(x + c)` for the scaled tail beyond 4.

# Examples

```
// scaled and unscaled forms agree: exp(x) F(x)
let scaled = scioxide_special::synchrotron_1_scaled(2.0_f64)?;
let expected = 2.0_f64.exp() * scioxide_special::synchrotron_1(2.0_f64)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley, 1979, ch. 6."###)]

    pub synchrotron_1 [S: RealScalar] (x: S) -> S
    where [];
    options options: SynchrotronOptions;
    detailed synchrotron_1_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            false,
            false,
            options,
            algorithm_component!(stringify!(synchrotron_1)),
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the second synchrotron function G(x).

```text
G(x) = x K_(2/3)(x),   x ≥ 0,
```

where `K_ν` is the modified Bessel function of the second kind. `G(0) = 0`, and
`G(x) ≈ √(πx/2) exp(−x)` for large `x`; the value underflows for `x` beyond about 745 in `f64`,
where [`synchrotron_2_scaled`](crate::synchrotron_2_scaled) stays in range.

The method uses Chebyshev expansions: in `x²/8` with the fractional powers of `x` factored out
for `x ≤ 1`, on `(1, 4]`, and in `1/(x + c)` for the scaled tail beyond 4.

# Examples

```
// scaled and unscaled forms agree: exp(x) G(x)
let scaled = scioxide_special::synchrotron_2_scaled(2.0_f64)?;
let expected = 2.0_f64.exp() * scioxide_special::synchrotron_2(2.0_f64)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley, 1979, ch. 6."###)]

    pub synchrotron_2 [S: RealScalar] (x: S) -> S
    where [];
    options options: SynchrotronOptions;
    detailed synchrotron_2_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            true,
            false,
            options,
            algorithm_component!(stringify!(synchrotron_2)),
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled synchrotron function exp(x) F(x).

F is defined in [`synchrotron_1`](crate::synchrotron_1), whose domain applies. The
scaled function behaves like `√(πx/2)` for large `x`, so it stays in range where F(x)
underflows.

# Examples

```
// exp(1) F(1) from the unscaled form
let scaled = scioxide_special::synchrotron_1_scaled(1.0_f64)?;
let expected = 1.0_f64.exp() * scioxide_special::synchrotron_1(1.0_f64)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley, 1979, ch. 6."###)]

    pub synchrotron_1_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: SynchrotronOptions;
    detailed synchrotron_1_scaled_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            false,
            true,
            options,
            algorithm_component!(stringify!(synchrotron_1_scaled)),
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the scaled synchrotron function exp(x) G(x).

G is defined in [`synchrotron_2`](crate::synchrotron_2), whose domain applies. The
scaled function behaves like `√(πx/2)` for large `x`, so it stays in range where G(x)
underflows.

# Examples

```
// exp(1) G(1) from the unscaled form
let scaled = scioxide_special::synchrotron_2_scaled(1.0_f64)?;
let expected = 1.0_f64.exp() * scioxide_special::synchrotron_2(1.0_f64)?;
assert!((scaled - expected).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`.

# References

- G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley, 1979, ch. 6."###)]

    pub synchrotron_2_scaled [S: RealScalar] (x: S) -> S
    where [];
    options options: SynchrotronOptions;
    detailed synchrotron_2_scaled_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            true,
            true,
            options,
            algorithm_component!(stringify!(synchrotron_2_scaled)),
        )
    }

}

/// Evaluation of a synchrotron function: a bounded expansion or the asymptotic tail.
type SynchrotronEvaluation<S> = Evaluation<S, Either<Form<S>, Tail<S>>>;

fn evaluate<S: RealScalar>(
    x: S,
    second: bool,
    scaled: bool,
    options: SynchrotronOptions,
    family: AlgorithmComponent,
) -> Result<SynchrotronEvaluation<S>, SciOxideError> {
    nonnegative(x, scioxide_core::argument!("x"))?;
    if x == S::ZERO {
        return Ok(bounded(
            S::ZERO,
            crate::result::Bound::derived(S::ZERO),
            AlgorithmId::from_components(family, algorithm_component!("origin"), 1),
            0,
            true,
        )?
        .map_form(Either::Left));
    }
    let budget = options.iteration_budget().get();
    let mut work = crate::support::Work::new(budget);
    let (mut value, mut error, method, required) = if x <= S::ONE {
        let z = x.cbrt();
        let t = x * x / real::<S>(8.0)? - S::ONE;
        let (a, ae) = chebyshev(
            t,
            if second {
                &coefficients::SYNCHROTRON21
            } else {
                &coefficients::SYNCHROTRON1
            },
            &mut work,
        )?;
        let (b, be) = chebyshev(
            t,
            if second {
                &coefficients::SYNCHROTRON22
            } else {
                &coefficients::SYNCHROTRON2
            },
            &mut work,
        )?;
        // x*z^2 and x^3*z^2 preserve the fractional powers without pow(x,1/3).
        let power = if second { x * z * z } else { x * x * x * z * z };
        let linear = if second {
            S::ZERO
        } else {
            real::<S>(core::f64::consts::PI / 1.732_050_807_568_877_2)? * x
        };
        let value = z * a - power * b - linear;
        let error = z * ae
            + power * be
            + real::<S>(16.0)? * S::EPSILON * ((z * a).abs() + (power * b).abs() + linear);
        (
            value,
            error,
            algorithm_component!("fractional_power_chebyshev"),
            if second { 26 } else { 25 },
        )
    } else if x <= real::<S>(4.0)? {
        let t = (real::<S>(2.0)? * x - real::<S>(5.0)?) / real::<S>(3.0)?;
        let table = if second {
            &super::transport_generated::CENTRAL2
        } else {
            &super::transport_generated::CENTRAL1
        };
        let (value, error) = chebyshev(t, table, &mut work)?;
        (
            value,
            error,
            algorithm_component!("central_scaled_chebyshev"),
            table.len(),
        )
    } else {
        let (offset, numerator, table): (f64, f64, &[f64]) = if second {
            (2.0, 12.0, &coefficients::SYNCHROTRON2A)
        } else {
            (4.0, 16.0, &coefficients::SYNCHROTRON1A)
        };
        let t = real::<S>(numerator)? / (x + real::<S>(offset)?) - S::ONE;
        let (value, error) = chebyshev(t, table, &mut work)?;
        let factor = real::<S>(1.253_314_137_315_500_3)? * x.sqrt();
        (
            factor * value,
            factor * (error + real::<S>(8.0)? * S::EPSILON * value.abs()),
            algorithm_component!("scaled_tail_chebyshev"),
            table.len(),
        )
    };
    let complete = budget >= required;
    let mut log_magnitude = None;
    let relative_error = if value > S::ZERO {
        error / value
    } else {
        S::ZERO
    };
    if x <= S::ONE {
        if scaled {
            let factor = x.exp();
            value *= factor;
            error = error * factor + real::<S>(8.0)? * S::EPSILON * value.abs();
        }
    } else if !scaled {
        if value > S::ZERO {
            log_magnitude = Some(value.ln() - x);
        }
        // Splitting the exponential avoids destroying a representable subnormal
        // by rounding exp(-x) to zero before multiplication by the prefactor.
        let factor = (-x * S::HALF).exp();
        value = (value * factor) * factor;
        error = (error * factor) * factor + real::<S>(16.0)? * S::EPSILON * value.abs();
    }
    let result = bounded(
        value,
        crate::result::Bound::derived(error),
        AlgorithmId::from_components(family, method, 1),
        work.used,
        complete,
    )?;
    let Some(log_magnitude) = log_magnitude else {
        return Ok(result.map_form(Either::Left));
    };
    let representation = crate::ScalarRepresentation::Finite;
    Ok(result
        .map_form(|inner| {
            Either::Right(Tail {
                inner,
                log_magnitude,
                relative_error,
            })
        })
        .with_projection(Some(value), representation))
}

/// An unscaled tail: the ordinary estimate, with scaled and logarithmic forms formed from its
/// logarithm, which survives underflow.
#[derive(Clone, Copy)]
pub(crate) struct Tail<S: RealScalar> {
    inner: Form<S>,
    log_magnitude: S,
    relative_error: S,
}

impl<S: RealScalar> Publish<S> for Tail<S> {
    fn publish(self, value: Option<S>) -> Result<Forms<S>, SciOxideError> {
        let Self {
            inner,
            log_magnitude,
            relative_error,
        } = self;
        let ordinary = inner.publish(value)?.value;
        let log_error =
            relative_error + real::<S>(32.0)? * S::EPSILON * log_magnitude.abs().max(S::ONE);
        let logarithmic = ValueWithError::new(
            LogValue::from_log(S::ONE, log_magnitude)?,
            log_error,
            None,
            EstimateReliability::ModelBased,
        )?;
        // A finite logarithm need not have an i64-sized base-two exponent or
        // enough absolute accuracy to produce a useful linear significand.
        let scaled = if log_error <= S::HALF
            && log_magnitude.abs() < real::<S>(9.0e18)? * real::<S>(core::f64::consts::LN_2)?
        {
            let relative_scale_error = log_error.exp_m1() + real::<S>(8.0)? * S::EPSILON;
            let scaled = ScaledValue::from_log(S::ONE, log_magnitude)?;
            Some(crate::result::scaled_estimate(
                scaled,
                relative_scale_error,
                EstimateReliability::ModelBased,
            )?)
        } else {
            None
        };
        Ok(Forms {
            value: ordinary,
            scaled,
            logarithmic: Some(logarithmic),
            derivative: None,
        })
    }
}
