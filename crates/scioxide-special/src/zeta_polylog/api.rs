//! Family-owned plain, detailed, and scoped scalar entry points.
use crate::SpecialFunctionResult;
use crate::dispatch::{complex, convert_scalar};
use scioxide_core::{Complex, NumericScalar};
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Riemann zeta function `ζ(s)`.

```text
ζ(s) = Σₙ₌₁^∞ n^(−s),   Re s > 1
```

(DLMF 25.2.1), continued meromorphically to the whole plane. The only pole is the simple pole at
`s = 1`, and there is no branch cut. The trivial zeros `s = −2, −4, …` return exact zero. The
argument is named `x` in the signature.

Real arguments use piecewise Chebyshev expansions for `s > 0` and on `[−1, 0]`, and the functional
equation (DLMF §25.4) in compensated logarithmic form below `s = −1`. Complex arguments use
Euler–Maclaurin summation for `Re s ≥ −1/2` and the functional equation, with exact reduction of the
`πs/2` phase, for `Re s < −1/2`.

# Examples

```
// ζ(2) = π²/6 and ζ(−1) = −1/12
let z = scioxide_special::riemann_zeta(2.0_f64)?;
assert!((z - core::f64::consts::PI.powi(2) / 6.0).abs() < 1e-14);
assert!((scioxide_special::riemann_zeta(-1.0_f64)? + 1.0 / 12.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments are within 64 ulps for `f32` and `f64` on the tested range. Complex arguments have no
uniform ulp bound; the error estimate includes the rounding of the logarithmic factor in the
functional equation. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `s = 1`.
- `NumericalInstability` (`Overflow`) if `|ζ(s)|` exceeds the range of the scalar type, which
  happens for large negative `s`.

# References

- NIST DLMF, §25.2–§25.4 and §25.11. <https://dlmf.nist.gov/25>"###)]
    #[inline]
    pub riemann_zeta [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ZetaOptions;
    detailed riemann_zeta_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::real_api::riemann_zeta(x.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::complex_zeta::riemann_zeta_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::real_api::riemann_zeta_evaluate(x.real_component(), options, context)
    } complex {
        crate::zeta_polylog::complex_zeta::riemann_zeta_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub riemann_zeta_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Dirichlet eta function `η(s)`, the alternating zeta function.

```text
η(s) = Σₙ₌₁^∞ (−1)^(n−1) n^(−s) = (1 − 2^(1−s)) ζ(s),   Re s > 0
```

(DLMF §25.2), continued to an entire function. There is no pole: `η(1) = ln 2`, and `η(0) = 1/2`.
The argument is named `x` in the signature.

Real arguments use Chebyshev expansions on `[−1, 2]` and the relation with `ζ(s)` elsewhere, in
compensated logarithmic form below `s = −1`.

# Examples

```
// η(1) = ln 2
let e = scioxide_special::dirichlet_eta(1.0_f64)?;
assert!((e - core::f64::consts::LN_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments are within 64 ulps for `f32` and `f64` on the tested range. Complex arguments have no
uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|η(s)|` exceeds the range of the scalar type, which
  happens for large negative `s`.

# References

- NIST DLMF, §25.2–§25.4 and §25.11. <https://dlmf.nist.gov/25>"###)]
    #[inline]
    pub dirichlet_eta [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::ZetaOptions;
    detailed dirichlet_eta_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::real_api::dirichlet_eta(x.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::complex_zeta::dirichlet_eta_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::real_api::dirichlet_eta_evaluate(x.real_component(), options, context)
    } complex {
        crate::zeta_polylog::complex_zeta::dirichlet_eta_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub dirichlet_eta_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Hurwitz zeta function `ζ(s, q)`.

```text
ζ(s, q) = Σₙ₌₀^∞ (n + q)^(−s),   Re s > 1
```

(DLMF 25.11.1), continued meromorphically in `s` with a simple pole at `s = 1`. `ζ(s, 1) = ζ(s)`.
For a real type `q` must be positive. For a complex type the powers are principal,
`(n + q)^(−s) = exp(−s ln(n + q))`, so `q` has a branch cut on the negative real axis, where the
sign of a zero imaginary part selects the side; `q = 0, −1, −2, …` are poles.

Real `s > 1` uses Euler–Maclaurin summation normalized by `q^(−s)`. Real `s < 1` uses a centered
Taylor series about `q = 1` (DLMF 25.11.10) or the Fourier series (DLMF 25.11.9), with the shift
recurrence `ζ(s, q) = ζ(s, q + 1) + q^(−s)`, and reports a derived absolute error. Complex arguments
with `Re s ≤ 0` evaluate Hermite's integral (DLMF 25.11.29) with adaptive Gauss–Legendre panels
after a shift in `q`.

`q` may be real while `s` is complex.

# Examples

```
// ζ(s, 1/2) = (2^s − 1) ζ(s), so ζ(2, 1/2) = π²/2
let z = scioxide_special::hurwitz_zeta(2.0_f64, 0.5)?;
assert!((z - core::f64::consts::PI.powi(2) / 2.0).abs() < 1e-13);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `s > 1` the error is within 64 ulps for `f32` and `f64` on the tested range. For real
`s < 1` the error estimate is a derived absolute bound, tested for `s ∈ [−60, 1)` and
`q ∈ (0, 1000]`; next to a zero of `ζ(s, q)` the relative error can be large. Complex arguments have
no uniform ulp bound, and for large `|Im s|` the estimate can be as large as the value. See the
[crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `s = 1`.
- `DomainError` (`Pole`) at `q = 0` for a real type, and at `q = 0, −1, −2, …` for a complex type.
- `DomainError` (`OutsideDomain`) if `q < 0` for a real type.

# References

- NIST DLMF, §25.2–§25.4 and §25.11. <https://dlmf.nist.gov/25>"###)]
    #[inline]
    pub hurwitz_zeta [T: NumericScalar] (s: T, q: impl crate::Parameter<T>) -> T
    where [];
    promote [q] to T;
    options options: crate::ZetaOptions;
    detailed hurwitz_zeta_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::real_api::hurwitz_zeta(s.real_component(), q.real_component())
            .map(T::from_real)
    } complex {
        crate::zeta_polylog::complex_zeta::hurwitz_zeta_complex(
            crate::dispatch::complex(s),
            crate::dispatch::complex(q),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::real_api::hurwitz_zeta_evaluate(
            s.real_component(),
            q.real_component(),
            options,
            context,
        )
    } complex {
        crate::zeta_polylog::complex_zeta::hurwitz_zeta_complex_evaluation(
            crate::dispatch::complex(s),
            crate::dispatch::complex(q),
            options,
            context,
        )
    }
    scoped pub hurwitz_zeta_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Wright omega function `ω(z)`, the solution of `ω + ln ω = z`.

```text
ω(z) = W_K(z)(eᶻ),   K(z) = ⌈(Im z − π) / (2π)⌉
```

`W_k` is branch `k` of the Lambert W function and `K(z)` is the unwinding number. For real `x`,
`ω(x) = W₀(eˣ)` is positive and increasing; for very negative `x` it can underflow to zero, and the
detailed result then keeps a finite logarithm. For complex `z`, `ω` is single-valued and continuous
except across the two half-lines `Im z = ±π`, `Re z ≤ −1`, where the sign of a zero imaginary part
selects the side. At the branch points `z = −1 ± iπ`, `ω = −1`. The argument is named `x` in the
signature.

Real arguments use Halley iteration. Complex arguments evaluate `W_K(z)(eᶻ)` with Halley iteration,
which works in the branch coordinate next to `z = −1 ± iπ`.

# Examples

```
// ω(1) = 1, and ω(0) = W₀(1) = 0.567143290409783872… (the omega constant)
assert!((scioxide_special::wright_omega(1.0_f64)? - 1.0).abs() < 1e-14);
let w = scioxide_special::wright_omega(0.0_f64)?;
assert!((w - 0.567_143_290_409_783_9).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range. Complex arguments have no
uniform ulp bound; their error estimate includes the rounding to the output type, propagated through
`ω + ln ω`. See the [crate accuracy table](crate#accuracy)."###, r###"This function has no domain errors: every finite argument is accepted.

# References

- R. M. Corless and D. J. Jeffrey, "The Wright ω function", in *Artificial Intelligence, Automated
  Reasoning, and Symbolic Computation*, LNCS 2385, Springer (2002), 76–89.
- NIST DLMF, §4.13. <https://dlmf.nist.gov/4.13>
- R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey and D. E. Knuth, "On the Lambert W
  function", *Adv. Comput. Math.* 5 (1996), 329–359. <https://doi.org/10.1007/BF02124750>"###)]
    #[inline]
    pub wright_omega [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::LambertOptions;
    detailed wright_omega_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::lambert::wright_omega(x.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::lambert::wright_omega_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::lambert::wright_omega_real_evaluate(x.real_component(), options, context)
    } complex {
        crate::zeta_polylog::lambert::wright_omega_complex_evaluate(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
    scoped pub wright_omega_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the dilogarithm `Li₂(z)`.

```text
Li₂(z) = Σₖ₌₁^∞ zᵏ / k² = −∫₀^z ln(1 − t) / t dt,   |z| ≤ 1
```

(DLMF 25.12.1), continued analytically to the plane cut along `[1, ∞)`; on the cut the sign of a
zero imaginary part selects the side. `Li₂(0) = 0` and `Li₂(1) = π²/6`. For a real type every finite
`x` is accepted, and for `x > 1` the result is the real part of the principal value. The argument is
named `x` in the signature.

The reflection and inversion formulas (DLMF 25.12.4 and 25.12.6) reduce the argument to `[−1, 1/2]`,
where thirteen terms of the Bernoulli series in `w = −ln(1 − z)` give the real value.
Near the real zero, `12 ≤ x ≤ 13`, inversion instead uses the defining series at `1/x`.
Real reductions and series terms consume the iteration budget. Complex arguments use the same
reductions and a budgeted Bernoulli series.

# Examples

```
// Li₂(1) = π²/6 and Li₂(−1) = −π²/12
let pi2 = core::f64::consts::PI.powi(2);
assert!((scioxide_special::dilog(1.0_f64)? - pi2 / 6.0).abs() < 1e-14);
assert!((scioxide_special::dilog(-1.0_f64)? + pi2 / 12.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range. Complex arguments have no
uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `ConvergenceFailure` if a real or complex evaluation exhausts the iteration budget.

# References

- NIST DLMF, §25.12. <https://dlmf.nist.gov/25.12>"###)]
    #[inline]
    pub dilog [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::PolylogOptions;
    detailed dilog_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::polylog::dilog(x.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::polylog::dilog_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::polylog::dilog_evaluate(x.real_component(), options)
    } complex {
        crate::zeta_polylog::polylog::dilog_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes Spence's function `Li₂(1 − z)`.

```text
Li₂(1 − z) = ∫₁^z ln t / (1 − t) dt
```

`Li₂` is the dilogarithm ([`crate::dilog`]). The argument is named `x` in the signature; for a real
type it must be nonnegative. For a complex type the branch cut is `(−∞, 0]`, and the sign of a zero
imaginary part selects the side (the map `z ↦ 1 − z` reverses the side relative to `Li₂`). The
values at zero and one are `π²/6` and `0`.

# Examples

```
// Li₂(1 − 1) = 0 and Li₂(1 − 2) = Li₂(−1) = −π²/12
assert!(scioxide_special::spence(1.0_f64)?.abs() < 1e-14);
let s = scioxide_special::spence(2.0_f64)?;
assert!((s + core::f64::consts::PI.powi(2) / 12.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"Real arguments target 64 ulps for `f32` and `f64` on the tested range. Complex arguments have no
uniform ulp bound. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x < 0` for a real type.
- `ConvergenceFailure` if a real or complex evaluation exhausts the iteration budget.

# References

- NIST DLMF, §25.12. <https://dlmf.nist.gov/25.12>"###)]
    #[inline]
    pub spence [T: NumericScalar] (x: T) -> T
    where [];
    options options: crate::PolylogOptions;
    detailed spence_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::polylog::spence(x.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::polylog::spence_complex(crate::dispatch::complex(x))
            .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::polylog::spence_evaluate(x.real_component(), options)
    } complex {
        crate::zeta_polylog::polylog::spence_complex_evaluation(
            crate::dispatch::complex(x),
            options,
            context,
        )
    }
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the polylogarithm `Li_s(z)`.

```text
Li_s(z) = Σₖ₌₁^∞ zᵏ / kˢ,   |z| < 1
```

(DLMF 25.12.10), continued analytically in `z` to the plane cut along `[1, ∞)` with principal
powers; on the cut the sign of a zero imaginary part selects the side. `Li_s(0) = 0`,
`Li₁(z) = −ln(1 − z)`, `Li₂` is the dilogarithm ([`crate::dilog`]), and for `s = 0, −1, −2, …`
`Li_s` is a rational function of `z`. At `z = 1` the result is `ζ(s)` for every `s ≠ 1`; for
`Re s ≤ 1` this is the analytic continuation, not the limit of the divergent series. For a real type
`z ≤ 1` is accepted for every `s`; `z > 1` is accepted for `s = 2` (the real part of the principal
value, as in [`crate::dilog`]) and for nonpositive integer `s`.

The function is evaluated as `z Φ(z, s, 1)` with the Lerch transcendent ([`crate::lerch_phi`]).
Nonpositive integer orders use Eulerian rational functions, and negative noninteger orders near
`|z| = 1` use the logarithmic series or Jonquière's inversion formula (DLMF 25.12.12–25.12.13).

`s` may be real while `z` is complex.

# Examples

```
// Li₁(z) = −ln(1 − z), so Li₁(1/2) = ln 2
let l = scioxide_special::polylog(1.0_f64, 0.5)?;
assert!((l - core::f64::consts::LN_2).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"For real `s = 2` the accuracy is that of [`crate::dilog`], 64 ulps on the tested range. Other orders
and complex arguments have no uniform ulp bound; their values and derivatives are checked against
independent references. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `s = 1`, `z = 1`.
- `DomainError` (`OutsideDomain`) for a real type if `z > 1`, unless `s = 2` or `s` is a nonpositive
  integer.

# References

- NIST DLMF, §25.12. <https://dlmf.nist.gov/25.12>"###)]
    #[inline]
    pub polylog [T: NumericScalar] (s: impl crate::Parameter<T>, z: T) -> T
    where [];
    promote [s] to T;
    options options: crate::PolylogOptions;
    detailed polylog_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::polylog::polylog(s.real_component(), z.real_component()).map(T::from_real)
    } complex {
        crate::zeta_polylog::polylog::polylog_complex(
            crate::dispatch::complex(s),
            crate::dispatch::complex(z),
        )
        .map(crate::dispatch::convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::polylog::polylog_evaluate(
            s.real_component(),
            z.real_component(),
            options,
            context,
        )
    } complex {
        crate::zeta_polylog::polylog::poly_evaluate(
            crate::dispatch::complex(s),
            crate::dispatch::complex(z),
            options,
            context,
        )
    }
    scoped pub polylog_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Lerch transcendent `Φ(z, s, a)`.

```text
Φ(z, s, a) = Σₙ₌₀^∞ zⁿ / (a + n)ˢ,   |z| < 1
```

(DLMF 25.14.1), continued analytically in `z` to the plane cut along `[1, ∞)`. The powers are
principal, `(a + n)^(−s) = exp(−s ln(a + n))`. Special cases are `Φ(1, s, a) = ζ(s, a)` (the Hurwitz
zeta function) and `z Φ(z, s, 1) = Li_s(z)`. `a = 0, −1, −2, …` are poles unless `s` is a
nonpositive integer, in which case those terms are polynomials in `a + n`.

For a real type `z ≤ 1` is accepted; `z > 1` only for nonpositive integer `s`. A noninteger real `s`
requires `a > 0`; an integer `s` also accepts `a ≤ 0`.

Terms with `Re(a + n) < 1` are summed directly. The remainder uses the series for `|z| < 0.95`, an
expansion in `ln z` near `z = 1`, and otherwise the integral representation (DLMF §25.14) on a
rotated path. At `z = 1` the Hurwitz zeta function is used.

`s` and `a` may be real while `z` is complex.

# Examples

```
// Φ(z, 0, a) = 1/(1 − z)
let p = scioxide_special::lerch_phi(0.5_f64, 0.0, 1.0)?;
assert!((p - 2.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; values and derivatives are checked against independent references.
See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) if `a` is a nonpositive integer and `s` is a positive integer for a real
  type, or is not a nonpositive integer for a complex type. Real noninteger `s` with `a ≤ 0`
  is rejected as `OutsideDomain` before pole evaluation.
- `DomainError` (`Pole`) at `z = 1`, `s = 1`.
- `DomainError` (`OutsideDomain`) for a real type if `z > 1` and `s` is not a nonpositive integer,
  or if `a ≤ 0` and `s` is not an integer.

# References

- NIST DLMF, §25.14. <https://dlmf.nist.gov/25.14>"###)]
    pub lerch_phi [T: NumericScalar] (z: T, s: impl crate::Parameter<T>, a: impl crate::Parameter<T>) -> T
    where [];
    promote [s, a] to T;
    options options: crate::LerchOptions;
    detailed lerch_phi_with_options -> SpecialFunctionResult<T>;
    dispatch T;
    value real {
        crate::zeta_polylog::polylog::lerch_phi(
            z.real_component(),
            s.real_component(),
            a.real_component(),
        )
        .map(T::from_real)
    } complex {
        crate::zeta_polylog::polylog::lerch_phi_complex(complex(z), complex(s), complex(a))
            .map(convert_scalar::<Complex<T::Real>, T>)
    }
    evaluate |options, context| real {
        crate::zeta_polylog::polylog::lerch_phi_evaluate(
            z.real_component(),
            s.real_component(),
            a.real_component(),
            options,
            context,
        )
    } complex {
        crate::zeta_polylog::polylog::lerch_evaluate(
            complex(z),
            complex(s),
            complex(a),
            options,
            context,
        )
    }
    scoped pub lerch_phi_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}
