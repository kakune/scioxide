//! Gamma, beta, digamma, polygamma and incomplete gamma and beta entry points.
use crate::SpecialFunctionResult;
use scioxide_core::{Complex, NumericScalar};

use crate::dispatch::{complex, convert_scalar};

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the gamma function Γ(z).

```text
Γ(z) = ∫₀^∞ t^(z−1) e^(−t) dt,   Re z > 0,
```

continued analytically to the whole complex plane (DLMF 5.2.1). Γ has simple poles at
`z = 0, −1, −2, …`, no zeros and no branch cut. For negative non-integer real `x` the result
carries its sign. `Γ(1) = Γ(2) = 1`.

Real arguments use exact factorials at the integers `1, …, 21`; elsewhere the recurrence
`Γ(x + 1) = x Γ(x)` reduces `x < 16` to a rational approximation on `[1, 2]`, and the Stirling
series covers `x ≥ 16`. Negative `x` uses the reflection `Γ(x) Γ(1 − x) = π / sin(πx)` (DLMF
5.5.3). `f32` arguments are evaluated in `f64` and rounded once. Complex arguments use
`exp(ln Γ(z))` with the Lanczos approximation and the same reflection for `Re z < 1/2`. The cost
is a fixed number of operations.

# Examples

```
// Γ(5) = 4! = 24
let g = scioxide_special::gamma(5.0_f64)?;
assert!((g - 24.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range;
the largest error observed on a 4,160-point `f64` sample is 3.7 ulps. Complex arguments have no
uniform ulp bound; the detailed form returns an error estimate. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0, −1, −2, …`.
- `NumericalInstability` (`Overflow`) if `|Γ(z)|` exceeds the range of the scalar type: for real
  `x` above about 171.62 (`f64`) or 35.04 (`f32`). [`crate::gamma_with_options`] still returns
  the signed logarithm; see [detailed results](crate#detailed-results).

# References

- NIST DLMF, §5.2–5.5 and §5.11. <https://dlmf.nist.gov/5>
- C. Lanczos, "A precision approximation of the gamma function", *SIAM J. Numer. Anal. Ser. B* 1
  (1964), 86–96."###)]
    #[inline]
    pub gamma [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::GammaOptions;
    detailed gamma_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::gamma::gamma(x.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::gamma::gamma_evaluate(x.real_component(), options)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the logarithm of the gamma function: `ln |Γ(x)|` for real `x` and the analytic
log-gamma function `ln Γ(z)` for complex `z`.

For real arguments the result is the logarithm of the magnitude, without the sign of `Γ(x)`;
[`crate::gamma_with_options`] returns the sign together with the logarithm.
`ln Γ(1) = ln Γ(2) = 0`.

For complex arguments the result is the analytic continuation of the real function from the
positive real axis. It is analytic except on the cut `(−∞, 0]`, where the sign of a zero
imaginary part selects the upper or lower side. Its imaginary part is not reduced to `(−π, π]`,
so it can differ from the principal logarithm of `Γ(z)` by a multiple of `2πi`.

Real arguments use rational approximations near the zeros at 1 and 2, reached by the recurrence
`Γ(x + 1) = x Γ(x)`, the Stirling series (DLMF 5.11.1) for `x ≥ 16`, and the reflection formula
for `x < 0`. Complex arguments use the Lanczos approximation, with the reflection
`Γ(z) Γ(1 − z) = π / sin(πz)` (DLMF 5.5.3) for `Re z < 1/2`.

# Examples

```
// ln Γ(5) = ln 24
let value = scioxide_special::ln_gamma(5.0_f64)?;
assert!((value - 24.0_f64.ln()).abs() < 1e-13);
// Just above the cut, Im ln Γ(−2.5 + 0i) = −3π.
let upper = scioxide_special::ln_gamma(scioxide_core::Complex::new(-2.5_f64, 0.0))?;
assert!((upper.im + 3.0 * core::f64::consts::PI).abs() < 1e-12);
let pole = scioxide_special::ln_gamma(0.0_f64).unwrap_err();
assert_eq!(pole.kind(), scioxide_core::SciOxideErrorKind::DomainError);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range,
including near the zeros of `ln |Γ(x)|`; the largest error observed on a 4,160-point `f64`
sample is 1.9 ulps. Complex arguments have no uniform ulp bound; the detailed form returns an
error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0, −1, −2, …`.

# References

- NIST DLMF, §5.2–5.5 and §5.11. <https://dlmf.nist.gov/5>
- C. Lanczos, "A precision approximation of the gamma function", *SIAM J. Numer. Anal. Ser. B* 1
  (1964), 86–96."###)]
    #[inline]
    pub ln_gamma [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::GammaOptions;
    detailed ln_gamma_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::gamma::ln_gamma(x.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::ln_gamma_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::gamma::ln_gamma_evaluate(x.real_component(), options)
    } complex {
        crate::gamma_beta::complex_gamma::ln_gamma_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the digamma function `ψ(z) = Γ′(z)/Γ(z)` (DLMF 5.2.2).

ψ is meromorphic, with simple poles at `z = 0, −1, −2, …` and no branch cut. `ψ(1) = −γ`, where
`γ` is the Euler–Mascheroni constant.

Real arguments use the recurrence `ψ(x + 1) = ψ(x) + 1/x` (DLMF 5.5.2) to reach `[1, 2]`, where
a rational approximation is evaluated relative to the positive zero `x₀ ≈ 1.4616`. For `x ≥ 16`
the asymptotic expansion (DLMF 5.11.2) is used, and for `x < 0` the reflection
`ψ(1 − x) − ψ(x) = π cot(πx)` (DLMF 5.5.4). Complex arguments differentiate the Lanczos
approximation of `ln Γ(z)`.

# Examples

```
// ψ(1) = −γ (DLMF 5.4.12)
let psi = scioxide_special::digamma(1.0_f64)?;
assert!((psi + 0.5772156649015329).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range;
the largest error observed on a 4,160-point `f64` sample is 2.1 ulps. For `x < 0` the bound is
absolute, 4 ulps of `|ψ(1 − x)| + |π cot(πx)|`, so relative accuracy is lost next to the zeros
of ψ on the negative axis. Complex arguments have no uniform ulp bound; the detailed form
returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0, −1, −2, …`.

# References

- NIST DLMF, §5.2–5.5 and §5.11. <https://dlmf.nist.gov/5>
- C. Lanczos, "A precision approximation of the gamma function", *SIAM J. Numer. Anal. Ser. B* 1
  (1964), 86–96."###)]
    #[inline]
    pub digamma [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::GammaOptions;
    detailed digamma_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::gamma::digamma(x.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::digamma_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::gamma::digamma_evaluate(x.real_component(), options)
    } complex {
        crate::gamma_beta::complex_gamma::digamma_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the regularized lower incomplete gamma function `P(a, z) = γ(a, z)/Γ(a)`.

```text
γ(a, z) = ∫₀^z t^(a−1) e^(−t) dt,   P(a, z) + Q(a, z) = 1
```

(DLMF 8.2.1, 8.2.4), where `a` is the shape and `Q` is [`crate::gamma_q`]. For real arguments
`a ≥ 0` and `x ≥ 0`. `P(a, 0) = 0` for `a > 0`, and `P(0, x) = 1` for `x > 0` is the limit
`a → 0`. `P(a, x)` is the cumulative distribution function of the gamma distribution with shape
`a` and unit scale.

For real arguments the smaller of `P` and `Q` is evaluated directly and the other as its
complement. Unit shape uses `P(1,x) = −expm1(−x)` and `Q(1,x) = exp(−x)`; half shape uses
`erf(√x)` and `erfc(√x)`. A small-shape upper series applies when `a < 1`, `x ≤ 2`, and
`a ln(x) > −ln(2)`. Integer and half-integer `a ≤ 32.5` use finite sums when `x ≥ a`,
`exp(−x)` is normal, and the work budget admits the sum. Otherwise Temme's uniform asymptotic
expansion (DLMF §8.12) applies for `a > 20` and `−a/4 < x − a < 2a/5` if `a < 100`,
or `|x − a| < 2a/5` if `a ≥ 100`. Outside that band the methods are the power series
(DLMF §8.7) for `x < a` and Legendre's continued fraction (DLMF 8.9.2) by the modified Lentz
method for `x ≥ a`. `f32` arguments are evaluated in `f64`.

For complex arguments `z^a` is the principal power, with a branch cut along the negative real
axis; the sign of a zero imaginary part selects the side. Complex arguments use the power series
near the origin and beside the cut, Legendre's continued fraction in the right half-plane, and
Tricomi's confluent hypergeometric function `U` elsewhere. For `z ≠ 0` and `a = 0, −1, −2, …`,
where `Γ(a)` has a pole, the result is the limit `P = 1`.

`a` may be real while `z` is complex.

# Examples

```
// P(1, x) = 1 − e^(−x)
let p = scioxide_special::gamma_p(1.0_f64, 1.0)?;
assert!((p + (-1.0_f64).exp_m1()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range, which includes shapes `20 < a ≤ 10¹⁰` and deep tails. The largest error observed on a
broader 4,160-point `f64` sample is 90 ulps. Complex arguments have no uniform ulp bound; the
detailed form returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if real `a < 0` or `x < 0`.
- `DomainError` (`Undefined`) if real `a = 0` and `x = 0`, or complex `z = 0` and `Re a ≤ 0`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]
    #[inline]
    pub gamma_p [T: NumericScalar] (a: impl crate::Parameter<T>, x: T) -> T
    where [];
    promote [a] to T;
    options options: crate::GammaOptions;
    detailed gamma_p_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::incomplete_gamma::gamma_p(a.real_component(), x.real_component())
            .map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_p_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::incomplete_gamma::gamma_p_evaluate(
            a.real_component(),
            x.real_component(),
            options,
            context,
        )
    } complex {
        crate::gamma_beta::complex_gamma::gamma_p_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub gamma_p_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the regularized upper incomplete gamma function `Q(a, z) = Γ(a, z)/Γ(a)`.

```text
Γ(a, z) = ∫_z^∞ t^(a−1) e^(−t) dt,   P(a, z) + Q(a, z) = 1
```

(DLMF 8.2.2, 8.2.4), where `a` is the shape and `P` is [`crate::gamma_p`]. For real arguments
`a ≥ 0` and `x ≥ 0`. `Q(a, 0) = 1` for `a > 0`, and `Q(0, x) = 0` for `x > 0` is the limit
`a → 0`. `Q(a, x)` is the survival function of the gamma distribution with shape `a` and unit
scale; small upper tails keep their relative accuracy.

For real arguments the smaller of `P` and `Q` is evaluated directly and the other as its
complement. Unit shape uses `P(1,x) = −expm1(−x)` and `Q(1,x) = exp(−x)`; half shape uses
`erf(√x)` and `erfc(√x)`. A small-shape upper series applies when `a < 1`, `x ≤ 2`, and
`a ln(x) > −ln(2)`. Integer and half-integer `a ≤ 32.5` use finite sums when `x ≥ a`,
`exp(−x)` is normal, and the work budget admits the sum. Otherwise Temme's uniform asymptotic
expansion (DLMF §8.12) applies for `a > 20` and `−a/4 < x − a < 2a/5` if `a < 100`,
or `|x − a| < 2a/5` if `a ≥ 100`. Outside that band the methods are the power series
(DLMF §8.7) for `x < a` and Legendre's continued fraction (DLMF 8.9.2) by the modified Lentz
method for `x ≥ a`. `f32` arguments are evaluated in `f64`.

For complex arguments `z^a` is the principal power, with a branch cut along the negative real
axis; the sign of a zero imaginary part selects the side. Complex arguments use the power series
near the origin and beside the cut, Legendre's continued fraction in the right half-plane, and
Tricomi's confluent hypergeometric function `U` elsewhere. For `z ≠ 0` and `a = 0, −1, −2, …`,
where `Γ(a)` has a pole, the result is the limit `Q = 0`.

`a` may be real while `z` is complex.

# Examples

```
// Q(1, x) = e^(−x)
let q = scioxide_special::gamma_q(1.0_f64, 1.0)?;
assert!((q - (-1.0_f64).exp()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range, which includes shapes `20 < a ≤ 10¹⁰` and deep tails. The largest error observed on a
broader 4,160-point `f64` sample is 90 ulps. Complex arguments have no uniform ulp bound; the
detailed form returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if real `a < 0` or `x < 0`.
- `DomainError` (`Undefined`) if real `a = 0` and `x = 0`, or complex `z = 0` and `Re a ≤ 0`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]
    #[inline]
    pub gamma_q [T: NumericScalar] (a: impl crate::Parameter<T>, x: T) -> T
    where [];
    promote [a] to T;
    options options: crate::GammaOptions;
    detailed gamma_q_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::incomplete_gamma::gamma_q(a.real_component(), x.real_component())
            .map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_q_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::incomplete_gamma::gamma_q_evaluate(
            a.real_component(),
            x.real_component(),
            options,
            context,
        )
    } complex {
        crate::gamma_beta::complex_gamma::gamma_q_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub gamma_q_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the lower incomplete gamma function `γ(a, z)`.

```text
γ(a, z) = ∫₀^z t^(a−1) e^(−t) dt = Γ(a) P(a, z)
```

(DLMF 8.2.1), where `a` is the shape. For real arguments `a > 0` and `x ≥ 0`, and `γ(a, 0) = 0`.

Real arguments add `ln P(a, x)` and `ln Γ(a)` in double-word arithmetic and exponentiate once,
with the tail methods of [`crate::gamma_p`]. For complex arguments `z^a` is the principal power,
with a branch cut along the negative real axis; the sign of a zero imaginary part selects the
side. Complex arguments use the power series near the origin and beside the cut, Legendre's
continued fraction in the right half-plane, and Tricomi's confluent hypergeometric function `U`
elsewhere.

`a` may be real while `z` is complex.

# Examples

```
// γ(1, x) = 1 − e^(−x)
let g = scioxide_special::gamma_lower(1.0_f64, 1.0)?;
assert!((g + (-1.0_f64).exp_m1()).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range, which includes shapes `20 < a ≤ 10¹⁰` and deep tails. The largest error observed on a
broader 4,160-point `f64` sample is 90 ulps. Complex arguments have no uniform ulp bound; the
detailed form returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if real `a = 0`, or complex `a = 0, −1, −2, …`.
- `DomainError` (`OutsideDomain`) if real `a < 0` or `x < 0`.
- `DomainError` (`Undefined`) if complex `z = 0` and `Re a ≤ 0`.
- `NumericalInstability` (`Overflow`) if `|γ(a, z)|` exceeds the range of the scalar type.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]
    #[inline]
    pub gamma_lower [T: NumericScalar] (a: impl crate::Parameter<T>, x: T) -> T
    where [];
    promote [a] to T;
    options options: crate::GammaOptions;
    detailed gamma_lower_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::incomplete_gamma::gamma_lower(a.real_component(), x.real_component())
            .map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_lower_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::incomplete_gamma::gamma_lower_evaluate(
            a.real_component(),
            x.real_component(),
            options,
            context,
        )
    } complex {
        crate::gamma_beta::complex_gamma::gamma_lower_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub gamma_lower_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the upper incomplete gamma function `Γ(a, z)`.

```text
Γ(a, z) = ∫_z^∞ t^(a−1) e^(−t) dt = Γ(a) Q(a, z),   γ(a, z) + Γ(a, z) = Γ(a)
```

(DLMF 8.2.2, 8.2.3), where `a` is the shape. For real arguments `a ≥ 0` and `x ≥ 0`.
`Γ(a, 0) = Γ(a)` for `a > 0`, and `Γ(0, x) = E₁(x)` for `x > 0` is the exponential integral
(DLMF 8.4.4).

Real arguments add `ln Q(a, x)` and `ln Γ(a)` in double-word arithmetic and exponentiate once,
with the tail methods of [`crate::gamma_q`]; for small `a` the pole of `Γ(a)` is cancelled
analytically. For complex arguments `z^a` is the principal power, with a branch cut along the
negative real axis; the sign of a zero imaginary part selects the side. Complex arguments use
the power series near the origin and beside the cut, Legendre's continued fraction in the right
half-plane, and Tricomi's confluent hypergeometric function `U` elsewhere.

`a` may be real while `z` is complex.

# Examples

```
use scioxide_core::Complex;
// γ(a, z) + Γ(a, z) = Γ(a) (DLMF 8.2.3)
let a = Complex::new(2.7881610989887076_f64, 0.03470671342846998);
let z = Complex::new(0.02842616704752339, -0.005068492501103518);
let upper = scioxide_special::gamma_upper(a, z)?;
let lower = scioxide_special::gamma_lower(a, z)?;
let complete = scioxide_special::gamma(a)?;
assert!((upper + lower - complete).norm() < 1e-12 * complete.norm());
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range, which includes shapes `20 < a ≤ 10¹⁰` and deep tails. The largest error observed on a
broader 4,160-point `f64` sample is 90 ulps. Complex arguments have no uniform ulp bound; the
detailed form returns an error estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if real `a < 0` or `x < 0`.
- `DomainError` (`Pole`) if `a = 0` and `z = 0`.
- `DomainError` (`Undefined`) if complex `z = 0`, `Re a ≤ 0` and `a ≠ 0`.
- `NumericalInstability` (`Overflow`) if `|Γ(a, z)|` exceeds the range of the scalar type, for
  example `Γ(a, 0) = Γ(a)` with real `a` above about 171.62 in `f64`.

# References

- NIST DLMF, §8.2–8.12. <https://dlmf.nist.gov/8>
- N. M. Temme, "The asymptotic expansion of the incomplete gamma functions", *SIAM J. Math.
  Anal.* 10 (1979), 757–766."###)]
    #[inline]
    pub gamma_upper [T: NumericScalar] (a: impl crate::Parameter<T>, x: T) -> T
    where [];
    promote [a] to T;
    options options: crate::GammaOptions;
    detailed gamma_upper_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::incomplete_gamma::gamma_upper(a.real_component(), x.real_component())
            .map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::gamma_upper_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::incomplete_gamma::gamma_upper_evaluate(
            a.real_component(),
            x.real_component(),
            options,
            context,
        )
    } complex {
        crate::gamma_beta::complex_gamma::gamma_upper_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub gamma_upper_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the beta function `B(a, b)`.

```text
B(a, b) = ∫₀¹ t^(a−1) (1 − t)^(b−1) dt = Γ(a) Γ(b) / Γ(a + b),   Re a > 0, Re b > 0,
```

continued analytically through the gamma quotient (DLMF 5.12.1). B is symmetric in `a` and `b`.
Real arguments require `a > 0` and `b > 0`.

Real arguments use exact recurrences for small integer and half-integer shapes and otherwise a
gamma ratio arranged to avoid overflow and cancellation; `f32` arguments are evaluated in `f64`.
Unit shapes use `B(1, b) = 1/b`; detailed results retain scaled and logarithmic forms when
the reciprocal overflows, and positive rounding allowances for subnormal reciprocals.
Complex arguments use `exp(ln Γ(a) + ln Γ(b) − ln Γ(a + b))` with the Lanczos approximation.
Where `a + b` is a pole of Γ but `a` and `b` are not, the result is zero.

# Examples

```
// B(2, 3) = Γ(2) Γ(3) / Γ(5) = 2/24
let b = scioxide_special::beta(2.0_f64, 3.0)?;
assert!((b - 1.0 / 12.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range;
the largest error observed on a 4,160-point `f64` sample is 1.0 ulp. Complex arguments have no
uniform ulp bound; the detailed form returns an error estimate. See the [crate accuracy
table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if real `a = 0` or `b = 0`, or complex `a` or `b` is `0, −1, −2, …`.
- `DomainError` (`OutsideDomain`) if real `a < 0` or `b < 0`.
- `NumericalInstability` (`Overflow`) if `B(a, b)` exceeds the range of the scalar type.

# References

- NIST DLMF, §5.12. <https://dlmf.nist.gov/5.12>"###)]
    #[inline]
    pub beta [T: NumericScalar] (a: T, b: T) -> T
    where [];
    options options: crate::BetaOptions;
    detailed beta_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::beta::beta(a.real_component(), b.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::beta_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(b),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::beta::beta_evaluate(
            a.real_component(),
            b.real_component(),
            options)
    } complex {
        crate::gamma_beta::complex_gamma::beta_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(b),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the logarithm of the beta function: `ln B(a, b)` for real `a, b > 0` and
`ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for complex arguments.

For real arguments the result does not overflow for large or tiny shapes, and keeps its relative
accuracy where `ln B(a, b)` is close to zero. For complex arguments each term is the analytic
log-gamma function of [`crate::ln_gamma`], with its cut conventions; the sum is not reduced to
the principal branch.

# Examples

```
// ln B(2, 3) = ln(1/12)
let value = scioxide_special::ln_beta(2.0_f64, 3.0)?;
assert!((value + 12.0_f64.ln()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range,
including near the zeros of `ln B`; the largest error observed on a 4,160-point `f64` sample is
0.5 ulps. Complex arguments have no uniform ulp bound; the detailed form returns an error
estimate. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if real `a = 0` or `b = 0`, or complex `a`, `b` or `a + b` is
  `0, −1, −2, …`.
- `DomainError` (`OutsideDomain`) if real `a < 0` or `b < 0`.

# References

- NIST DLMF, §5.12. <https://dlmf.nist.gov/5.12>"###)]
    #[inline]
    pub ln_beta [T: NumericScalar] (a: T, b: T) -> T
    where [];
    options options: crate::BetaOptions;
    detailed ln_beta_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::beta::ln_beta(a.real_component(), b.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::ln_beta_complex(
            crate::dispatch::complex(a),
            crate::dispatch::complex(b),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::beta::ln_beta_evaluate(
            a.real_component(),
            b.real_component(),
            options)
    } complex {
        crate::gamma_beta::complex_gamma::ln_beta_complex_evaluation(
            crate::dispatch::complex(a),
            crate::dispatch::complex(b),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the regularized incomplete beta function `I_x(a, b)`.

```text
I_x(a, b) = B_x(a, b) / B(a, b),   B_x(a, b) = ∫₀^x t^(a−1) (1 − t)^(b−1) dt
```

(DLMF 8.17.1, 8.17.2), where `B(a, b)` is the beta function. For real arguments `a, b ≥ 0` and
`0 ≤ x ≤ 1`. For positive shapes `I_0 = 0` and `I_1 = 1`. At zero shapes the limits are
`I_x(0, b) = 1` for `x > 0` and `I_x(a, 0) = 0` for `x < 1`. `I_x(a, b)` is the cumulative
distribution function of the beta distribution, and `I_x(a, b) = 1 − I_{1−x}(b, a)` (DLMF
8.17.4).

Real arguments use finite binomial sums for integer shapes with `a + b ≤ 33`, the power series
(DLMF 8.17.7, 8.17.8), and otherwise the continued fraction (DLMF §8.17(v)) by the modified
Lentz method, after choosing the orientation in which the series or fraction converges quickly.
`f32` arguments are evaluated in `f64`. Complex arguments use
`I_z(a, b) = z^a ₂F₁(a, 1 − b; a + 1; z) / (a B(a, b))` (DLMF 8.17.7) with the principal power
`z^a`; the branch cuts are `(−∞, 0]` and `[1, ∞)`, and the sign of a zero imaginary part selects
the side.

`a` and `b` may be real while `z` is complex.

# Examples

```
// I_{1/2}(2, 3) = (C(4, 2) + C(4, 3) + C(4, 4)) / 2⁴ = 11/16
let i = scioxide_special::beta_inc(2.0_f64, 3.0, 0.5)?;
assert!((i - 0.6875).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 32 ulps for `f64` and 1 ulp for `f32` on the tested
range; the largest error observed on a broader 4,160-point `f64` sample is 12 ulps. See the
[crate accuracy table](crate#accuracy). Complex arguments have no uniform ulp bound."###, r###"- `DomainError` (`OutsideDomain`) if real `a < 0`, `b < 0`, or `x` is outside `[0, 1]`.
- `DomainError` (`Undefined`) if `a = b = 0`; if `a = 0` and `x = 0`, or `b = 0` and `x = 1`
  (real); or if `z = 0` and `Re a ≤ 0`, or `z = 1` and `Re b ≤ 0` (complex).
- `DomainError` (`Pole`) if complex `a + b`, or `b` together with `a` or `a + b`, is
  `0, −1, −2, …`.

# References

- NIST DLMF, §8.17. <https://dlmf.nist.gov/8.17>"###)]
    pub beta_inc [T: NumericScalar] (a: impl crate::Parameter<T>, b: impl crate::Parameter<T>, x: T) -> T
    where [];
    promote [a, b] to T;
    options options: crate::BetaOptions;
    detailed beta_inc_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::beta::beta_inc(a.real_component(), b.real_component(), x.real_component())
            .map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::beta_inc_complex(complex(a), complex(b), complex(x))
            .map(convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::beta::beta_inc_evaluate(
            a.real_component(),
            b.real_component(),
            x.real_component(),
            options,
            context,
        )
    } complex {
        crate::gamma_beta::complex_gamma::beta_inc_run(
            complex(a),
            complex(b),
            complex(x),
            options,
            context,
        )
    }
    scoped pub beta_inc_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the polygamma function `ψ⁽ⁿ⁾(z)`, the `n`-th derivative of the digamma function, where
`n` is `order`.

```text
ψ⁽ⁿ⁾(z) = dⁿ⁺¹ ln Γ(z) / dzⁿ⁺¹ = (−1)ⁿ⁺¹ n! ζ(n + 1, z),   n ≥ 1,
```

where `ζ(s, z)` is the Hurwitz zeta function (DLMF §5.15, 25.11.12). `ψ⁽⁰⁾` is
[`crate::digamma`]. ψ⁽ⁿ⁾ has poles of order `n + 1` at `z = 0, −1, −2, …` and no branch cut.

Positive real arguments sum the Hurwitz series with an Euler–Maclaurin tail in double-word
arithmetic. Negative real arguments use the reflection formula of DLMF §5.15. Complex arguments
use the Hurwitz zeta function. The work grows linearly with `order` and counts toward the
iteration budget of [`crate::GammaOptions`].

# Examples

```
// ψ′(1) = ζ(2) = π²/6
let p = scioxide_special::polygamma(1, 1.0_f64)?;
assert!((p - core::f64::consts::PI.powi(2) / 6.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real arguments the error is within 4 ulps for `f64` and 1 ulp for `f32` on the tested range,
which covers orders up to 65 and non-integer arguments down to `−10⁸`; the largest error
observed on a 4,160-point `f64` sample is 0.7 ulps. For `x < 0` the bound is absolute, 4 ulps of
the sum of the magnitudes of the reflection terms, so relative accuracy is lost next to zeros.
Complex arguments have no uniform ulp bound; the detailed form returns an error estimate. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0, −1, −2, …`.
- `SizeOverflow` if `order + 64` (real) or `order + 1` (complex) overflows `usize`.
- `ScalarConversion` if that number is not exactly representable in `f64`.
- `NumericalInstability` (`Overflow`) if `|ψ⁽ⁿ⁾(z)|`, which grows like `n!`, exceeds the range
  of the scalar type.
- `ConvergenceFailure` if the work for `order` exceeds the iteration budget.

# References

- NIST DLMF, §5.15 and §25.11. <https://dlmf.nist.gov/5.15>"###)]
    pub polygamma [T: NumericScalar] (order: usize, z: T) -> T
    where [];
    options options: crate::GammaOptions;
    detailed polygamma_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::gamma_beta::gamma::polygamma(order, z.real_component()).map(T::from_real)
    } complex {
        crate::gamma_beta::complex_gamma::polygamma_complex(order, complex(z))
            .map(convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::gamma_beta::gamma::polygamma_evaluate(order, z.real_component(), options, context)
    } complex {
        crate::gamma_beta::complex_gamma::polygamma_complex_evaluation(
            order,
            complex(z),
            options,
            context,
        )
    }
    scoped pub polygamma_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
