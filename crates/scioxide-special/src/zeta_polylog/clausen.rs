//! Real order-two Clausen function with full-range phase reduction.

use super::clausen_coefficients as coefficients;
use crate::expansions::{bounded, chebyshev, real};
use crate::result::Evaluation;
use crate::{ClausenOptions, SpecialFunctionResult};
use scioxide_core::algorithm_component;
use scioxide_core::{RealScalar, SciOxideError};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Clausen function `Cl₂(x)`.

```text
Cl₂(x) = Σₖ₌₁^∞ sin(kx) / k² = −∫₀^x ln|2 sin(t/2)| dt
```

`x` is in radians. `Cl₂` is odd, `2π`-periodic and continuous, with no poles, and
`Cl₂(x) = Im Li₂(eⁱˣ)`. Every finite `x` is accepted: the reduction modulo `2π` uses full-range
trigonometric argument reduction, so it stays accurate up to the largest finite arguments. The sign
of zero is preserved.

Next to a multiple of `2π`, a Chebyshev expansion gives `Cl₂(u) = u (c(u) − ln|u|)`, where `u` is
the reduced argument. Next to an odd multiple of `π`, the value is a power series in `(u/π)²`, where
`u` is the distance to that multiple. The cost is a fixed number of terms.

# Examples

```
// Cl₂(π/2) = G, Catalan's constant
let c = scioxide_special::clausen(core::f64::consts::FRAC_PI_2)?;
assert!((c - 0.915_965_594_177_219_0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound. The detailed result reports a derived error bound, and tests check
values and bounds in `f32` and `f64`, including subnormal arguments. See the [crate accuracy
table](crate#accuracy)."###, r###"This function has no domain errors: every finite argument is accepted.

# References

- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, §27.8.
- NIST DLMF, §25.12. <https://dlmf.nist.gov/25.12>"###)]

    pub clausen [S: RealScalar] (x: S) -> S
    where [];
    options options: ClausenOptions;
    detailed clausen_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        clausen_evaluate(x, options)
    }

}

/// Evaluate before projecting the value or forming the detailed result.
fn clausen_evaluate<S: RealScalar>(
    x: S,
    options: ClausenOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "x must be finite",
        ));
    }
    if x == S::ZERO {
        return bounded(
            x,
            crate::result::Bound::derived(S::ZERO),
            scioxide_core::algorithm_id!("clausen", "origin", 1),
            0,
            true,
        );
    }
    let (angle, near_pi) = if x.abs() <= S::PI * S::HALF {
        (x, false)
    } else {
        let (sine, cosine) = x.sin_cos();
        if cosine < S::ZERO {
            (sine.atan2(-cosine), true)
        } else {
            (sine.atan2(cosine), false)
        }
    };
    let u = angle.abs();
    let budget = options.iteration_budget().get();
    let mut work = crate::support::Work::new(budget);
    let (value, error, required, method) = if near_pi {
        let q = (u / S::PI) * (u / S::PI);
        let mut power = q;
        let mut sum = S::ZERO;
        let count = super::clausen_generated::CLAUSEN_PI
            .len()
            .min(work.remaining);
        work.steps(count)?;
        for &coefficient in &super::clausen_generated::CLAUSEN_PI[..count] {
            sum += real::<S>(coefficient)? * power;
            power *= q;
        }
        // The coefficients are positive and non-increasing, so the next one
        // (at most the last tabulated) bounds the geometric tail.
        let next = super::clausen_generated::CLAUSEN_PI
            .get(count)
            .or(super::clausen_generated::CLAUSEN_PI.last())
            .copied()
            .unwrap_or(0.0);
        let omitted = real::<S>(next)? * power / (S::ONE - q);
        let value = u * (real::<S>(core::f64::consts::LN_2)? - sum);
        let error = u * omitted + real::<S>(24.0)? * S::EPSILON * u;
        (
            value,
            error,
            super::clausen_generated::CLAUSEN_PI.len(),
            algorithm_component!("pi_log_cos_series"),
        )
    } else {
        // sin/atan2 reduction preserves tiny distances to a multiple of 2π.
        // The zero guard avoids evaluating log(0) if a reducer rounds to zero.
        if u == S::ZERO {
            return bounded(
                angle,
                crate::result::Bound::derived(S::ZERO),
                scioxide_core::algorithm_id!("clausen", "reduced_origin", 1),
                0,
                true,
            );
        }
        let t = real::<S>(2.0)? * (u / S::PI) * (u / S::PI) - S::ONE;
        let (v, e) = chebyshev(t, &coefficients::ACLAUS, &mut work)?;
        let logarithm = u.ln();
        let value = u * (v - logarithm);
        let error = u * e + real::<S>(24.0)? * S::EPSILON * u * (S::ONE + logarithm.abs());
        (
            value,
            error,
            coefficients::ACLAUS.len(),
            algorithm_component!("log_sine_chebyshev"),
        )
    };
    bounded(
        if angle < S::ZERO { -value } else { value },
        crate::result::Bound::derived(error),
        scioxide_core::AlgorithmId::from_components(algorithm_component!("clausen"), method, 1),
        work.used,
        budget >= required,
    )
}

#[cfg(test)]
mod tests {
    use super::super::clausen_generated::CLAUSEN_PI;

    #[test]
    fn near_pi_coefficients_bound_their_tail() {
        assert!(CLAUSEN_PI.iter().all(|&coefficient| coefficient > 0.0));
        assert!(CLAUSEN_PI.windows(2).all(|pair| pair[1] <= pair[0]));
    }
}
