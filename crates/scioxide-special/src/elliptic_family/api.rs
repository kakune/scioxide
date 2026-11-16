//! Family-owned plain, detailed, and scoped scalar entry points.
use crate::SpecialFunctionResult;
use scioxide_core::{Complex, NumericScalar};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete elliptic integral of the first kind `K(m)`.

```text
K(m) = ∫₀^(π/2) (1 − m sin²θ)^(−1/2) dθ
```

`m = k²` is the parameter; DLMF §19.2 writes the same function as `K(k)` with the modulus `k`. Real
`m` must satisfy `m ≤ 1`, and `K(1) = +∞` is returned as a limit. For complex `m` the integrand uses
the principal square root, so the branch cut is `m ∈ [1, ∞)`; the sign of a zero imaginary part
selects the side of the cut. For a complex type, `m = 1` is a pole.

Real `0 ≤ m < 1` uses a fixed-degree polynomial approximation in `1 − m` with a logarithmic term,
evaluated in `f64` and rounded once for `f32`, so the cost does not depend on `m`. Negative `m` is
mapped to `0 < m < 1` by the imaginary-modulus transformation (DLMF §19.7). Complex `m` uses
Carlson's duplication algorithm for `K(m) = R_F(0, 1 − m, 1)` (DLMF §19.25).

# Examples

```
// K(0) = π/2
let k = scioxide_special::elliptic_k(0.0_f64)?;
assert!((k - core::f64::consts::FRAC_PI_2).abs() < 1e-14);
// K(1/2) = Γ(1/4)² / (4√π)
let g = scioxide_special::gamma(0.25_f64)?;
let expected = g * g / (4.0 * core::f64::consts::PI.sqrt());
assert!((scioxide_special::elliptic_k(0.5_f64)? - expected).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `0 ≤ m < 1` the error is within 2 ulps for `f64` and 1 ulp for `f32` on the tested range,
and the detailed result reports an error estimate of the same size. Negative real `m` gives finite
results down to the most negative representable argument. Complex `m` has no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `m > 1` for a real type.
- `DomainError` (`Pole`) at `m = 1` for a complex type.

# References

- NIST DLMF, §19.2, §19.7 and §19.25. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]
    #[inline]
    pub elliptic_k [T: NumericScalar] (m: T) -> T
    where [];
    options options: crate::EllipticOptions;
    detailed elliptic_k_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::elliptic_family::real_api::elliptic_k(m.real_component()).map(T::from_real)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_k_complex(crate::dispatch::complex(m))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::elliptic_family::real_api::elliptic_k_evaluate(m.real_component(), options, context)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_k_complex_evaluation(
            crate::dispatch::complex(m),
            options,
            context,
        )
    }
    scoped pub elliptic_k_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete elliptic integral of the second kind `E(m)`.

```text
E(m) = ∫₀^(π/2) (1 − m sin²θ)^(1/2) dθ
```

`m = k²` is the parameter; DLMF §19.2 writes the same function as `E(k)` with the modulus `k`. Real
`m` must satisfy `m ≤ 1`, and `E(1) = 1`. For complex `m` the integrand uses the principal square
root, so the branch cut is `m ∈ [1, ∞)`; the sign of a zero imaginary part selects the side of the
cut.

Real `0 ≤ m < 1` uses a fixed-degree polynomial approximation in `1 − m` with a logarithmic term,
evaluated in `f64` and rounded once for `f32`, so the cost does not depend on `m`. Negative `m` is
mapped to `0 < m < 1` by the imaginary-modulus transformation (DLMF §19.7). Complex `m` uses
Carlson's duplication algorithm for `E(m) = R_F(0, 1 − m, 1) − (m/3) R_D(0, 1 − m, 1)` (DLMF
§19.25).

# Examples

```
// E(0) = π/2 and E(1) = 1
let e = scioxide_special::elliptic_e(0.0_f64)?;
assert!((e - core::f64::consts::FRAC_PI_2).abs() < 1e-14);
assert!((scioxide_special::elliptic_e(1.0_f64)? - 1.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `0 ≤ m < 1` the error is within 2 ulps for `f64` and 1 ulp for `f32` on the tested range,
and the detailed result reports an error estimate of the same size. Negative real `m` gives finite
results down to the most negative representable argument. Complex `m` has no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `m > 1` for a real type.

# References

- NIST DLMF, §19.2, §19.7 and §19.25. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]
    #[inline]
    pub elliptic_e [T: NumericScalar] (m: T) -> T
    where [];
    options options: crate::EllipticOptions;
    detailed elliptic_e_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::elliptic_family::real_api::elliptic_e(m.real_component()).map(T::from_real)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_e_complex(crate::dispatch::complex(m))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::elliptic_family::real_api::elliptic_e_evaluate(m.real_component(), options, context)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_e_complex_evaluation(
            crate::dispatch::complex(m),
            options,
            context,
        )
    }
    scoped pub elliptic_e_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the incomplete elliptic integral of the first kind `F(φ|m)`.

```text
F(φ|m) = ∫₀^φ (1 − m sin²θ)^(−1/2) dθ
```

`φ` is the amplitude in radians and `m = k²` the parameter; DLMF §19.2 writes `F(φ, k)` with the
modulus `k`. Real `m` must satisfy `m ≤ 1`. For `m < 1` every real `φ` is accepted: `F` is odd in
`φ`, and `F(φ + jπ|m) = F(φ|m) + 2jK(m)` for integer `j` reduces the amplitude to `[0, π/2]`. At
`m = 1`, `F(φ|1) = artanh(sin φ)` is finite for `|φ| < π/2`, is `±∞` at `φ = ±π/2` (the value of π/2
rounded to the scalar type), and is undefined beyond. Complex arguments use principal square roots;
the sign of a zero imaginary part selects the side of a branch cut.

Real values use `F(φ|m) = sin φ · R_F(cos²φ, 1 − m sin²φ, 1)` (DLMF §19.25) with Carlson's
duplication algorithm.

`m` may be real while `φ` is complex.

# Examples

```
// F(φ|0) = φ
let f = scioxide_special::elliptic_f(0.5_f64, 0.0)?;
assert!((f - 0.5).abs() < 1e-14);
// F(π/2|m) = K(m)
let f = scioxide_special::elliptic_f(core::f64::consts::FRAC_PI_2, 0.5)?;
assert!((f - scioxide_special::elliptic_k(0.5_f64)?).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; the measured maximum for `f64` is 5.7 ulps.
Complex arguments have no uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `m > 1` for a real type.
- `DomainError` (`Undefined`) if `m = 1` and `|φ| > π/2` for a real type: the path of integration
  crosses the singularity of the integrand.
- `NumericalInstability` (`Overflow`) if the result exceeds the range of the scalar type, which can
  happen only for very large `|φ|`.

# References

- NIST DLMF, §19.2, §19.7 and §19.25. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]
    #[inline]
    pub elliptic_f [T: NumericScalar] (phi: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: crate::EllipticOptions;
    detailed elliptic_f_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::elliptic_family::real_api::elliptic_f(phi.real_component(), m.real_component())
            .map(T::from_real)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_f_complex(
            crate::dispatch::complex(phi),
            crate::dispatch::complex(m),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::elliptic_family::real_api::elliptic_f_evaluate(
            phi.real_component(),
            m.real_component(),
            options,
            context,
        )
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_f_complex_evaluation(
            crate::dispatch::complex(phi),
            crate::dispatch::complex(m),
            options,
            context,
        )
    }
    scoped pub elliptic_f_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the incomplete elliptic integral of the second kind `E(φ|m)`.

```text
E(φ|m) = ∫₀^φ (1 − m sin²θ)^(1/2) dθ
```

`φ` is the amplitude in radians and `m = k²` the parameter; DLMF §19.2 writes `E(φ, k)` with the
modulus `k`. Real `m` must satisfy `m ≤ 1`, and every real `φ` is accepted: `E` is odd in `φ`, and
`E(φ + jπ|m) = E(φ|m) + 2jE(m)` for integer `j` reduces the amplitude to `[0, π/2]`. At `m = 1` the
integrand is `|cos θ|`, so `E` is finite for every finite `φ`. Complex arguments use principal
square roots; the sign of a zero imaginary part selects the side of a branch cut.

Real values use `E(φ|m) = s R_F(c², Δ², 1) − (m/3) s³ R_D(c², Δ², 1)` with `s = sin φ`, `c = cos φ`
and `Δ² = 1 − m s²` (DLMF §19.25), and Carlson's duplication algorithm.

`m` may be real while `φ` is complex.

# Examples

```
// E(φ|0) = φ and E(φ|1) = sin φ for |φ| ≤ π/2
let e = scioxide_special::elliptic_e_inc(0.5_f64, 0.0)?;
assert!((e - 0.5).abs() < 1e-14);
let e = scioxide_special::elliptic_e_inc(0.5_f64, 1.0)?;
assert!((e - 0.5_f64.sin()).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range; the measured maximum for `f64` is 5.7 ulps.
Complex arguments have no uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `m > 1` for a real type.
- `NumericalInstability` (`Overflow`) if the result exceeds the range of the scalar type, which can
  happen only for very large `|φ|`.

# References

- NIST DLMF, §19.2, §19.7 and §19.25. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]
    #[inline]
    pub elliptic_e_inc [T: NumericScalar] (phi: T, m: impl crate::Parameter<T>) -> T
    where [];
    promote [m] to T;
    options options: crate::EllipticOptions;
    detailed elliptic_e_inc_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::elliptic_family::real_api::elliptic_e_inc(phi.real_component(), m.real_component())
            .map(T::from_real)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_e_inc_complex(
            crate::dispatch::complex(phi),
            crate::dispatch::complex(m),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::elliptic_family::real_api::elliptic_e_inc_evaluate(
            phi.real_component(),
            m.real_component(),
            options,
            context,
        )
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_e_inc_complex_evaluation(
            crate::dispatch::complex(phi),
            crate::dispatch::complex(m),
            options,
            context,
        )
    }
    scoped pub elliptic_e_inc_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complete elliptic integral of the third kind `Π(n|m)`.

```text
Π(n|m) = ∫₀^(π/2) dθ / ((1 − n sin²θ) (1 − m sin²θ)^(1/2))
```

`n` is the characteristic and `m = k²` the parameter; DLMF §19.2 writes `Π(α², k)` with `n = α²` and
the modulus `k`. Real arguments must satisfy `m ≤ 1` and `n ≤ 1`: for `n > 1` the integrand has a
pole inside `(0, π/2)`, and the Cauchy principal value is not returned. `n = 1` or `m = 1` returns
`+∞` as a limit. Complex arguments use principal square roots; the sign of a zero imaginary part
selects the side of a branch cut, and `n = 1` and `m = 1` are poles.

Real values use `Π(n|m) = R_F(0, 1 − m, 1) + (n/3) R_J(0, 1 − m, 1, 1 − n)` (DLMF §19.25) with
Carlson's duplication algorithm, and the closed forms `Π(0|m) = K(m)`, `Π(m|m) = E(m)/(1 − m)` and
`Π(n|0) = π / (2√(1 − n))`.

# Examples

```
// Π(n|0) = π / (2√(1 − n))
let p = scioxide_special::elliptic_pi(0.75_f64, 0.0)?;
assert!((p - core::f64::consts::PI).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps on the tested range. Complex arguments have no uniform ulp bound. See
the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `m > 1` for a real type.
- `DomainError` (`Undefined`) if `n > 1` for a real type.
- `DomainError` (`Pole`) at `n = 1` or `m = 1` for a complex type.
- `DomainError` (`OutsideDomain`) for a complex type if `n` and `m` are both real and greater than
  one but their zero imaginary parts have opposite signs, which puts them on opposite sides of the
  branch cut.

# References

- NIST DLMF, §19.2, §19.7 and §19.25. <https://dlmf.nist.gov/19>
- B. C. Carlson, "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms*
  10 (1995), 13–26. <https://doi.org/10.1007/BF02198293>"###)]
    #[inline]
    pub elliptic_pi [T: NumericScalar] (n: T, m: T) -> T
    where [];
    options options: crate::EllipticOptions;
    detailed elliptic_pi_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::elliptic_family::real_api::elliptic_pi(n.real_component(), m.real_component())
            .map(T::from_real)
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_pi_complex(
            crate::dispatch::complex(n),
            crate::dispatch::complex(m),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::elliptic_family::real_api::elliptic_pi_evaluate(
            n.real_component(),
            m.real_component(),
            options,
            context,
        )
    } complex {
        crate::elliptic_family::complex_elliptic::elliptic_pi_complex_evaluation(
            crate::dispatch::complex(n),
            crate::dispatch::complex(m),
            options,
            context,
        )
    }
    scoped pub elliptic_pi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
