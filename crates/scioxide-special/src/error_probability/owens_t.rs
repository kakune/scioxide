//! Owen T using symmetry reduction and the Gaussian-quadrature method T5.
//!
//! For `a ≤ 1` one 32-point Gauss–Legendre pass over `[0, min(a, 12/h)]`
//! resolves the integrand to below the `f64` rounding: the poles of
//! `1/(1+t²)` lie outside its Bernstein ellipse, and beyond `ht = 12` the
//! omitted normalized Gaussian tail is below 2e-33. Larger `a` reflects to
//! `1/a` through the upper normal tails. `f32` evaluates in `f64` and
//! rounds once.
use crate::error_probability::probability::{error, finish, normal_sf_kernel};
use crate::result::{Bound, Evaluation};
use crate::support::Work;
use crate::support::require_finite_param;
use crate::{ProbabilityOptions, SpecialFunctionResult};
use core::f64::consts::TAU;
use scioxide_core::{RealScalar, SciOxideError, algorithm_component};

/// `T(h, a)` for `h, a ≥ 0` and its absolute error bound, charged to `w`.
fn reduced(h: f64, a: f64, w: &mut Work) -> Result<(f64, f64), SciOxideError> {
    if h == 0.0 {
        return Ok((a.atan() / TAU, 0.0));
    }
    if a == 0.0 {
        return Ok((a, 0.0));
    }
    if a > 1.0 {
        // T(h,a)+T(ah,1/a) = (Q(h)+Q(ah))/2 - Q(h)Q(ah).
        let ah = a * h;
        let q = normal_sf_kernel(h, w)?;
        let (r, (t, e)) = if ah.is_finite() {
            (normal_sf_kernel(ah, w)?, reduced(ah, a.recip(), w)?)
        } else {
            (0.0, (0.0, 0.0))
        };
        if !w.complete {
            return Ok((0.0, 1.0));
        }
        let v = (q + r) * 0.5 - q * r - t;
        // Each tail contributes its own probability envelope.
        return Ok((
            v,
            e + error(v)?.absolute + error(q)?.absolute + error(r)?.absolute,
        ));
    }
    let factor = (-(h * 0.5) * h).exp() / TAU;
    if factor == 0.0 {
        // T ≤ a·exp(-h²/2)/(2π) with a ≤ 1 lies below half a quantum.
        return Ok((0.0, f64::from_bits(1)));
    }
    let half = a.min(12.0 / h) * 0.5;
    let mut sum = 0.0;
    for &(node, weight) in &crate::support::GL32 {
        if !w.step() {
            // The integrand is at most one, which bounds the whole integral.
            return Ok((factor * half * sum, factor * 2.0 * half));
        }
        let t = half * (1.0 + node);
        let ht = h * t;
        sum += weight * (-(ht * 0.5) * ht).exp() / (1.0 + t * t);
    }
    let v = factor * half * sum;
    Ok((v, error(v)?.absolute))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Owen's T function T(h, a).

```text
T(h, a) = (1/(2π)) ∫₀ᵃ exp(−h²(1 + t²)/2) / (1 + t²) dt
```

for real `h` and `a`. T is even in `h` and odd in `a`, with `T(0, a) = atan(a)/(2π)` and
`T(h, 0) = 0`. It gives bivariate normal probabilities over angular sectors.

For `|a| ≤ 1` the integral is evaluated by 32-point Gauss–Legendre quadrature on
`[0, min(|a|, 12/|h|)]` (method T5 of Patefield and Tandy); the omitted Gaussian tail is below
`2·10⁻³³`. For `|a| > 1` the reflection

```text
T(h, a) + T(ah, 1/a) = (Q(h) + Q(ah))/2 − Q(h) Q(ah),   h, a ≥ 0,
```

with `Q(x) = 1 − Φ(x)` the upper normal tail, maps the problem to `1/a`. `f32` arguments are
evaluated in `f64` and rounded once.

# Examples

```
// T(h, 1) = Φ(h) (1 − Φ(h)) / 2 (Owen, 1956)
let p = scioxide_special::normal_cdf(1.0_f64)?;
let t = scioxide_special::owens_t(1.0_f64, 1.0)?;
assert!((t - 0.5 * p * (1.0 - p)).abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target. The detailed form reports a derived error bound that includes the
quadrature and the error of the normal tails. See the [crate accuracy
table](crate#accuracy)."###, r###"
No error conditions of its own: every finite `h` and `a` is accepted.

# References

- D. B. Owen, "Tables for computing bivariate normal probabilities", *Ann. Math. Statist.* 27(4),
  1956, 1075–1090.
- M. Patefield and D. Tandy, "Fast and accurate calculation of Owen's T function", *J. Statist.
  Softw.* 5(5), 2000."###)]

    pub owens_t [S: RealScalar] (h: S, a: S) -> S
    where [];
    options options: ProbabilityOptions;
    detailed owens_t_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        owens_t_evaluate(h, a, options)
    }

}

/// Evaluates the detailed result before it is projected and returned.
fn owens_t_evaluate<S: RealScalar>(
    h: S,
    a: S,
    options: ProbabilityOptions,
) -> Result<Evaluation<S>, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || owens_t_evaluate::<f64>(h.into_f64(), a.into_f64(), options),
        |wide| wide.narrow(),
    ) {
        return result;
    }
    require_finite_param(h, "probability parameter h must be finite")?;
    require_finite_param(a, "probability parameter a must be finite")?;
    let (h, a) = (h.into_f64(), a.into_f64());
    let mut w = Work::new(options.iteration_budget().get());
    let (v, e) = reduced(h.abs(), a.abs(), &mut w)?;
    finish(
        S::try_from_f64_lossy(v.copysign(a))?,
        Bound::derived(S::try_from_f64_lossy(e)?),
        &w,
        if a.abs() > 1.0 {
            algorithm_component!("owen_reciprocal_t5")
        } else {
            algorithm_component!("owen_t5")
        },
    )
}
