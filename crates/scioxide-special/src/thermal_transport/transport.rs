//! Transport integrals of integer orders two through five.

use crate::expansions::{bounded, chebyshev, nonnegative, real};
use crate::result::Evaluation;
use crate::thermal_transport::transport_coefficients as coefficients;
use crate::{SpecialFunctionResult, TransportOptions};
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{RealScalar, SciOxideError};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the transport integral J₂(x).

```text
J₂(x) = ∫₀ˣ t² exp(t) / (exp(t) − 1)² dt,   x ≥ 0
```

`J₂(0) = 0`, and `J₂(x)` increases to `2! ζ(2) = π²/3` as `x → ∞`, where `ζ` is the
Riemann zeta function.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`, multiplied by `x`. Beyond 4 it
subtracts from the limit the exponentially convergent tail
`∑ₖ x² exp(−kx)·(polynomial in 1/(kx))`.

# Examples

```
// J₂(x) → 2! ζ(2) = π²/3; the tail is negligible at x = 50
let v = scioxide_special::transport_2(50.0_f64)?;
assert!((v - core::f64::consts::PI.powi(2) / 3.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`."###)]

    pub transport_2 [S: RealScalar] (x: S) -> S
    where [];
    options options: TransportOptions;
    detailed transport_2_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            2,
            options,
            algorithm_component!(stringify!(transport_2)),
            context,
        )
    }
    scoped pub transport_2_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the transport integral J₃(x).

```text
J₃(x) = ∫₀ˣ t³ exp(t) / (exp(t) − 1)² dt,   x ≥ 0
```

`J₃(0) = 0`, and `J₃(x)` increases to `3! ζ(3) = 7.212341418957566` as `x → ∞`, where `ζ` is the
Riemann zeta function.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`, multiplied by `x²`. Beyond 4 it
subtracts from the limit the exponentially convergent tail
`∑ₖ x³ exp(−kx)·(polynomial in 1/(kx))`.

# Examples

```
// J₃(x) → 3! ζ(3), ζ(3) = 1.2020569031595942; the tail is negligible at x = 50
let v = scioxide_special::transport_3(50.0_f64)?;
assert!((v - 6.0 * 1.202_056_903_159_594_2).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`."###)]

    pub transport_3 [S: RealScalar] (x: S) -> S
    where [];
    options options: TransportOptions;
    detailed transport_3_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            3,
            options,
            algorithm_component!(stringify!(transport_3)),
            context,
        )
    }
    scoped pub transport_3_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the transport integral J₄(x).

```text
J₄(x) = ∫₀ˣ t⁴ exp(t) / (exp(t) − 1)² dt,   x ≥ 0
```

`J₄(0) = 0`, and `J₄(x)` increases to `4! ζ(4) = 4π⁴/15` as `x → ∞`, where `ζ` is the
Riemann zeta function.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`, multiplied by `x³`. Beyond 4 it
subtracts from the limit the exponentially convergent tail
`∑ₖ x⁴ exp(−kx)·(polynomial in 1/(kx))`.

# Examples

```
// J₄(x) → 4! ζ(4) = 4π⁴/15; the tail is negligible at x = 50
let v = scioxide_special::transport_4(50.0_f64)?;
assert!((v - 4.0 * core::f64::consts::PI.powi(4) / 15.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`."###)]

    pub transport_4 [S: RealScalar] (x: S) -> S
    where [];
    options options: TransportOptions;
    detailed transport_4_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            4,
            options,
            algorithm_component!(stringify!(transport_4)),
            context,
        )
    }
    scoped pub transport_4_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the transport integral J₅(x).

```text
J₅(x) = ∫₀ˣ t⁵ exp(t) / (exp(t) − 1)² dt,   x ≥ 0
```

`J₅(0) = 0`, and `J₅(x)` increases to `5! ζ(5) = 124.4313306172044` as `x → ∞`, where `ζ` is the
Riemann zeta function.

For `x ≤ 4` the method is a Chebyshev expansion in `x²/8`, multiplied by `x⁴`. Beyond 4 it
subtracts from the limit the exponentially convergent tail
`∑ₖ x⁵ exp(−kx)·(polynomial in 1/(kx))`.

# Examples

```
// J₅(x) → 5! ζ(5), ζ(5) = 1.0369277551433699; the tail is negligible at x = 50
let v = scioxide_special::transport_5(50.0_f64)?;
assert!((v - 120.0 * 1.036_927_755_143_369_9).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; tests check values, logarithms, reported error bounds and
subnormal results in `f32` and `f64`. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `x < 0`."###)]

    pub transport_5 [S: RealScalar] (x: S) -> S
    where [];
    options options: TransportOptions;
    detailed transport_5_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        evaluate(
            x,
            5,
            options,
            algorithm_component!(stringify!(transport_5)),
            context,
        )
    }
    scoped pub transport_5_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

fn evaluate<S: RealScalar>(
    x: S,
    order: usize,
    options: TransportOptions,
    family: AlgorithmComponent,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    nonnegative(x, scioxide_core::argument!("x"))?;
    let budget = options.iteration_budget().get();
    let mut work = crate::support::Work::with_context(budget, context);
    work.check()?;
    if x == S::ZERO {
        return bounded(
            S::ZERO,
            crate::result::Bound::derived(S::ZERO),
            scioxide_core::AlgorithmId::from_components(family, algorithm_component!("origin"), 1),
            0,
            true,
        );
    }
    if x <= real::<S>(4.0)? {
        let table = match order {
            2 => &coefficients::TRANSPORT2,
            3 => &coefficients::TRANSPORT3,
            4 => &coefficients::TRANSPORT4,
            _ => &coefficients::TRANSPORT5,
        };
        let (v, e) = chebyshev(x * x / real::<S>(8.0)? - S::ONE, table, &mut work)?;
        work.check()?;
        // Multiply the normalized integral first: no premature division or
        // separately rounded x^(n-1) at the subnormal boundary.
        let value = (1..order).fold(v, |value, _| value * x);
        let error =
            (1..order).fold(e, |error, _| error * x) + real::<S>(8.0)? * S::EPSILON * value.abs();
        return bounded(
            value,
            crate::result::Bound::derived(error),
            scioxide_core::AlgorithmId::from_components(
                family,
                algorithm_component!("origin_chebyshev"),
                1,
            ),
            work.used,
            budget >= table.len(),
        );
    }
    let infinity = real::<S>(match order {
        2 => 3.289_868_133_696_453,
        3 => 7.212_341_418_957_566,
        4 => 25.975_757_609_067_315,
        _ => 124.431_330_617_204_4,
    })?;
    let q = (-x).exp();
    let n = S::try_from_usize_exact(order)?;
    let prefactor = (n * x.ln() - x).exp();
    let mut tail = S::ZERO;
    let mut q_power = S::ONE;
    let mut error = infinity;
    let mut complete = false;
    let mut k = 0;
    // Integral of each k*exp(-kt) term is an exact degree-n polynomial.
    // Its coefficients decrease with k, so q/(1-q) bounds the remaining tail.
    while work.step() {
        k += 1;
        let inverse = (S::ONE / x) / S::try_from_usize_exact(k)?;
        let polynomial = (1..=order).try_fold(S::ONE, |p, j| {
            Ok::<_, SciOxideError>(p * S::try_from_usize_exact(j)? * inverse + S::ONE)
        })?;
        let term = prefactor * q_power * polynomial;
        tail += term;
        let remainder = term * q / (S::ONE - q);
        error = remainder
            + real::<S>(16.0)?
                * S::EPSILON
                * (infinity + tail * (S::ONE + x.min(real::<S>(1024.0)?)));
        // Fixed accuracy: stop once the tail is below one unit of the limit.
        if remainder <= S::EPSILON * infinity {
            complete = true;
            break;
        }
        q_power *= q;
    }
    work.check()?;
    bounded(
        infinity - tail,
        crate::result::Bound::derived(error),
        scioxide_core::AlgorithmId::from_components(
            family,
            algorithm_component!("exponential_tail"),
            1,
        ),
        work.used,
        complete,
    )
}
